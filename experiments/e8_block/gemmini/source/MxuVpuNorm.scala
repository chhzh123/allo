package gen
import chisel3._
import chisel3.util._
import chisel3.stage.ChiselStage
import hardfloat._
import gemmini._

class NormTag extends Bundle with TagQueueTag {
  val id = UInt(8.W)
  override def make_this_garbage(dummy: Int = 0): Unit = { id := 255.U }
}

/** Gemmini as a transformer-block engine: the mesh, the statistics unit and
  * the scale/activation path, with LayerNorm, IGELU and softmax on device.
  *
  * `MxuVpu` (E3) is the same datapath with `has_normalizations = false`, which
  * leaves scale + ReLU. Turning normalizations on is not a flag: it forces a
  * **floating-point scale type**, and that is structural rather than a choice.
  * `Normalizer` drives `inv_stddev` and `inv_sum_exp` through `MulPipe`, and
  * `MulPipe` matches only `case Float(expWidth, sigWidth, false)`; so does
  * `Arithmetic.reciprocal`. Gemmini's own configs use `Float(8, 24)` for the
  * accumulator scale wherever normalizations are enabled. So the comparison
  * against SPMW has to be at float32 scale, and the E3 numbers -- `SInt(32.W)`
  * with a shift -- are a different design point, not a baseline for this.
  *
  * The wiring is direct: `Normalizer.io.out` is a `NormalizedOutput`, which is
  * an `AccumulatorReadResp` plus `mean`, `max`, `inv_stddev` and `inv_sum_exp`
  * -- exactly `AccumulatorScale.io.in.bits`.
  */
class MxuVpuNorm(val dim: Int, val scaleLatency: Int = 4) extends Module {
  val inW = 8
  val accW = 32
  val fullDataType = Vec(dim, Vec(1, SInt(accW.W)))
  val rDataType = Vec(dim, Vec(1, SInt(inW.W)))
  val scale_t = Float(8, 24)

  // Gemmini's own accumulator scale function, verbatim from `Configs.scala`:
  // integer to recoded float, a fused multiply-add against the scale, back to
  // integer with saturation. Not a reimplementation -- a reimplementation
  // would be one more thing that could differ from the baseline.
  //
  // The shift form cannot be used here: it reads the scalar as a shift count,
  // so `inv_stddev` would be truncated to five bits.
  val scale_func = (t: SInt, f: Float) => {
    val f_rec = recFNFromFN(f.expWidth, f.sigWidth, f.bits)

    val in_to_rec_fn = Module(new INToRecFN(t.getWidth, f.expWidth, f.sigWidth))
    in_to_rec_fn.io.signedIn := true.B
    in_to_rec_fn.io.in := t.asTypeOf(UInt(t.getWidth.W))
    in_to_rec_fn.io.roundingMode := consts.round_near_even
    in_to_rec_fn.io.detectTininess := consts.tininess_afterRounding

    val t_rec = in_to_rec_fn.io.out

    val muladder = Module(new MulAddRecFN(f.expWidth, f.sigWidth))
    muladder.io.op := 0.U
    muladder.io.roundingMode := consts.round_near_even
    muladder.io.detectTininess := consts.tininess_afterRounding
    muladder.io.a := t_rec
    muladder.io.b := f_rec
    muladder.io.c := 0.U

    val rec_fn_to_in = Module(new RecFNToIN(f.expWidth, f.sigWidth, t.getWidth))
    rec_fn_to_in.io.in := muladder.io.out
    rec_fn_to_in.io.roundingMode := consts.round_near_even
    rec_fn_to_in.io.signedOut := true.B

    val overflow = rec_fn_to_in.io.intExceptionFlags(1)
    val maxsat = ((1 << (t.getWidth - 1)) - 1).S
    val minsat = (-(1 << (t.getWidth - 1))).S
    val sign = rawFloatFromRecFN(f.expWidth, f.sigWidth, rec_fn_to_in.io.in).sign
    val sat = Mux(sign, minsat, maxsat)

    Mux(overflow, sat, rec_fn_to_in.io.out.asTypeOf(t))
  }

  val mxu = Module(new MeshWithDelays(SInt(inW.W), SInt(inW.W), SInt(accW.W), SInt(accW.W),
    new NormTag, Dataflow.WS, tree_reduction = false, tile_latency = 0,
    output_delay = 1, tileRows = 1, tileColumns = 1,
    meshRows = dim, meshColumns = dim, leftBanks = 1, upBanks = 1, outBanks = 1))

  val norm = Module(new Normalizer(max_len = 1024, num_reduce_lanes = -1, num_stats = 2,
    latency = 4, fullDataType = fullDataType, scale_t = scale_t))

  // `latency` does **not** break the scale's combinational path: in both of
  // `AccumulatorScale`'s branches the activation, the int-to-float, the
  // multiply-add and the float-to-int are one cloud, and `latency` is a
  // `Pipe` on its output.  At `latency = 1` this routes with a 29.7 ns data
  // path -- 107 logic levels and four chained DSP multiplies -- so the
  // registers are there to be *retimed* backwards into the cloud, and the
  // P&R script turns retiming on.  Four is Gemmini's own value.
  val vpu = Module(new AccumulatorScale(fullDataType, rDataType, scale_t,
    read_small_data = true, read_full_data = false,
    scale_func = scale_func, num_scale_units = -1, latency = scaleLatency,
    has_nonlinear_activations = true, has_normalizations = true))

  val io = IO(new Bundle {
    val a = Flipped(Decoupled(chiselTypeOf(mxu.io.a.bits)))
    val b = Flipped(Decoupled(chiselTypeOf(mxu.io.b.bits)))
    val d = Flipped(Decoupled(chiselTypeOf(mxu.io.d.bits)))
    val req = Flipped(Decoupled(chiselTypeOf(mxu.io.req.bits)))
    // The accumulator, which in Gemmini sits between these two and in this
    // scope sits outside the module.  `mesh_out` is what the mesh writes to
    // it; `acc_in` is the read that drives the normalise path.  They have to
    // be separate ports: the normaliser makes three passes over a row and the
    // second and third are accumulator reads, not recomputed matmuls -- and
    // the mesh could not carry them anyway, its operand ports being 8 bits
    // wide where an accumulator row is 32.
    val acc_in = Flipped(Decoupled(chiselTypeOf(mxu.io.resp.bits.data)))
    val mesh_out = Valid(chiselTypeOf(mxu.io.resp.bits.data))
    val scale = Input(scale_t.cloneType)
    val act = Input(UInt(Activation.bitwidth.W))
    val cmd = Input(NormCmd())
    val len = Input(UInt(11.W))
    // Which of the normaliser's two statistics banks this row uses.
    // Hardwiring it to zero would serialise the rows: with two banks a
    // row's sum accumulates while the row before it is still in the
    // divider, and `num_stats = 2` exists to allow exactly that.
    val stats_id = Input(UInt(1.W))
    val igelu_qb = Input(SInt(accW.W))
    val igelu_qc = Input(SInt(accW.W))
    val iexp_qln2 = Input(SInt(accW.W))
    val iexp_qln2_inv = Input(SInt(accW.W))
    val out = Decoupled(new AccumulatorScaleResp[SInt](fullDataType, rDataType))
  })

  mxu.io.a <> io.a
  mxu.io.b <> io.b
  mxu.io.d <> io.d
  mxu.io.req <> io.req
  dontTouch(mxu.io.tags_in_progress)

  io.mesh_out.valid := mxu.io.resp.valid
  io.mesh_out.bits := mxu.io.resp.bits.data

  norm.io.in.valid := io.acc_in.valid
  io.acc_in.ready := norm.io.in.ready
  norm.io.in.bits.acc_read_resp.data := io.acc_in.bits
  norm.io.in.bits.acc_read_resp.act := io.act
  norm.io.in.bits.acc_read_resp.scale := io.scale
  norm.io.in.bits.acc_read_resp.igelu_qb := io.igelu_qb
  norm.io.in.bits.acc_read_resp.igelu_qc := io.igelu_qc
  norm.io.in.bits.acc_read_resp.iexp_qln2 := io.iexp_qln2
  norm.io.in.bits.acc_read_resp.iexp_qln2_inv := io.iexp_qln2_inv
  norm.io.in.bits.acc_read_resp.fromDMA := false.B
  norm.io.in.bits.acc_read_resp.acc_bank_id := 0.U
  norm.io.in.bits.len := io.len
  norm.io.in.bits.stats_id := io.stats_id
  norm.io.in.bits.cmd := io.cmd

  vpu.io.in <> norm.io.out
  io.out <> vpu.io.out
}

object ElaborateMxuVpuNorm extends App {
  val dim = sys.env.getOrElse("MESH_DIM", "16").toInt
  val lat = sys.env.getOrElse("SCALE_LATENCY", "4").toInt
  val tag = if (lat == 4) "" else "_l" + lat
  println("MXUVPUNORM_ELABORATE_START dim=" + dim + " scaleLatency=" + lat)
  (new ChiselStage).emitVerilog(new MxuVpuNorm(dim, lat),
    Array("--target-dir", "mxuvpunorm_out_" + dim + tag))
  println("MXUVPUNORM_ELABORATE_OK dim=" + dim + " scaleLatency=" + lat)
}

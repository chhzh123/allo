package gen
import chisel3._
import chisel3.util._
import chisel3.stage.ChiselStage
import gemmini._

class SimpleTag extends Bundle with TagQueueTag {
  val id = UInt(8.W)
  override def make_this_garbage(dummy: Int = 0): Unit = { id := 255.U }
}

// One TPU-shaped datapath: Gemmini's weight-stationary mesh (the MXU) feeding
// its accumulator scale/activation path (the VPU). Scope is requantise then
// ReLU, so has_normalizations is false and LAYERNORM/IGELU/SOFTMAX are gated
// out; num_scale_units = -1 selects the simple combinational activate-then-
// scale path rather than the multi-unit scheduler.
//
// The mesh half is instantiated with exactly the parameters E1 measured, so
// the MXU here is the same hardware as the E1 Gemmini WS rows.
class MxuVpu(val dim: Int) extends Module {
  val inW = 8
  val accW = 32
  // The mesh emits one row of results per beat: Vec(meshCols, Vec(tileCols, T)).
  val fullDataType = Vec(dim, Vec(1, SInt(accW.W)))
  val rDataType = Vec(dim, Vec(1, SInt(inW.W)))
  val scale_t = SInt(accW.W)
  // Two requantisation forms, selected by SCALE_MODE.
  //   shift: an arithmetic shift right by the scale. This is what SPMW's VPU
  //          actually does -- its E3 programs requantise with SHR -- so it is
  //          the apple-to-apple form.
  //   mul:   multiply by the scale then shift, Gemmini's more general form.
  //          A 32x32 signed multiply plus shift and clip in one cycle does not
  //          close at 3.333 ns; the number is kept as a documented variant.
  val scaleMode = sys.env.getOrElse("SCALE_MODE", "shift")
  val scale_func = if (scaleMode == "mul") {
    (v: SInt, sc: SInt) => (v * sc) >> 8.U
  } else {
    (v: SInt, sc: SInt) => (v >> sc(4, 0).asUInt).asSInt
  }

  val mxu = Module(new MeshWithDelays(SInt(inW.W), SInt(inW.W), SInt(accW.W), SInt(accW.W),
    new SimpleTag, Dataflow.WS, tree_reduction = false, tile_latency = 0,
    output_delay = 1, tileRows = 1, tileColumns = 1,
    meshRows = dim, meshColumns = dim, leftBanks = 1, upBanks = 1, outBanks = 1))

  val vpu = Module(new AccumulatorScale(fullDataType, rDataType, scale_t,
    read_small_data = true, read_full_data = false,
    scale_func = scale_func, num_scale_units = -1, latency = 1,
    has_nonlinear_activations = true, has_normalizations = false))

  // The mesh's ports are exposed one by one rather than by flipping its whole
  // IO bundle: that bundle contains `tags_in_progress`, already an Output, and
  // flipping the aggregate makes both sides drivers of it. It is a debug
  // output this datapath does not need, so it is left unexposed and tied off.
  val io = IO(new Bundle {
    val a = Flipped(Decoupled(chiselTypeOf(mxu.io.a.bits)))
    val b = Flipped(Decoupled(chiselTypeOf(mxu.io.b.bits)))
    val d = Flipped(Decoupled(chiselTypeOf(mxu.io.d.bits)))
    val req = Flipped(Decoupled(chiselTypeOf(mxu.io.req.bits)))
    val scale = Input(SInt(accW.W))
    val out = Decoupled(new AccumulatorScaleResp[SInt](fullDataType, rDataType))
  })

  mxu.io.a <> io.a
  mxu.io.b <> io.b
  mxu.io.d <> io.d
  mxu.io.req <> io.req
  dontTouch(mxu.io.tags_in_progress)

  // Gemmini's real datapath is Mesh -> AccumulatorMem -> AccumulatorScale: the
  // mesh writes its rows into the accumulator and the scale path reads them
  // out afterwards, so a memory sits between the two. This scope excludes the
  // accumulator memory, and wiring the mesh straight into the scale path put
  // the 32x32 scale multiply in the same cycle as the mesh's output logic --
  // 4.694 ns and 20 logic levels at dim 4, against a 3.333 ns target. That is
  // an artefact of leaving the accumulator out, not a property of Gemmini, so
  // the boundary it would provide is modelled here by one register stage.
  val respValid = RegNext(mxu.io.resp.valid, false.B)
  val respData = RegNext(mxu.io.resp.bits.data)

  // The mesh's result bus is Valid, the scale path's input is Decoupled, so
  // the VPU cannot back-pressure the mesh. At this scope it never needs to:
  // the scale path accepts a row every cycle. Asserted rather than assumed.
  assert(!respValid || vpu.io.in.ready, "the VPU stalled while the MXU emitted a row")

  vpu.io.in.valid := respValid
  val r = vpu.io.in.bits.acc_read_resp
  r.data := respData
  r.scale := io.scale
  r.act := Activation.RELU
  r.fromDMA := false.B
  r.acc_bank_id := 0.U
  // igelu and iexp constants belong to activations this scope gates off.
  r.igelu_qb := 0.S
  r.igelu_qc := 0.S
  r.iexp_qln2 := 0.S
  r.iexp_qln2_inv := 0.S
  vpu.io.in.bits.mean := 0.S
  vpu.io.in.bits.max := 0.S
  vpu.io.in.bits.inv_stddev := 0.S
  vpu.io.in.bits.inv_sum_exp := 0.S

  io.out <> vpu.io.out
}

object ElaborateMxuVpu extends App {
  val dim = sys.env.getOrElse("MESH_DIM", "4").toInt
  println("MXUVPU_ELABORATE_START dim=" + dim + " scale=" + sys.env.getOrElse("SCALE_MODE", "shift"))
  val v = (new ChiselStage).emitVerilog(new MxuVpu(dim),
    Array("--target-dir", "mxuvpu_out_" + dim + "_" + sys.env.getOrElse("SCALE_MODE", "shift")))
  println("MXUVPU_ELABORATE_OK dim=" + dim + " verilog_chars=" + v.length)
}

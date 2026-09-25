package gen
import chisel3._
import chisel3.util._
import chisel3.stage.ChiselStage
import gemmini._

// Gemmini's datapath *with* its accumulator: MeshWithDelays -> AccumulatorMem
// -> AccumulatorScale, which is what Gemmini's Scratchpad wires and what E3's
// MxuVpu left out (it put one register where the accumulator goes).
//
// The accumulator is what a fixed workload on a narrow array needs. A K-deep
// tile on a `dim`-wide mesh is K/dim weight blocks whose partial sums must be
// added before the epilogue, and the mesh's top-of-array input is weightType
// -- eight bits -- so they cannot be fed back through it. Gemmini adds them
// in AccumulatorMem, read-modify-write, and so does this.
//
// Two banks, as Gemmini's accumulator has: an accumulating write takes the
// bank's read port, so a finished group is read out of one bank while the
// mesh accumulates the next group into the other. The adder is Gemmini's
// AccPipeShared, one for both banks, latency acc_latency - 1, as Scratchpad
// instantiates it.
//
// What Gemmini's ExecuteController would do is here as three tag bits and two
// counters. A request's tag says which bank its rows go to, whether they are
// written or added, whether this is a group's last K block, and which `dim`
// rows of the group they are (a request carries at most `dim` rows); bit 7 is
// Gemmini's own "garbage" marker. When a group's last row lands, its bank is
// read out, one row a cycle, through AccumulatorScale.
//
// `relu` picks the scale unit's activation: ReLU for E3's microbenchmark,
// none for a layer that only requantises (LLaMA's projections).
class MxuAccVpu(val dim: Int, val rows: Int, val relu: Boolean = true) extends Module {
  val inW = 8
  val accW = 32
  val fullDataType = Vec(dim, Vec(1, SInt(accW.W)))
  val rDataType = Vec(dim, Vec(1, SInt(inW.W)))
  val scale_t = SInt(accW.W)
  val scale_func = (v: SInt, sc: SInt) => (v >> sc(4, 0).asUInt).asSInt
  require(rows % dim == 0, "a group's rows come in requests of `dim`")
  val subs = rows / dim

  val mxu = Module(new MeshWithDelays(SInt(inW.W), SInt(inW.W), SInt(accW.W), SInt(accW.W),
    new SimpleTag, Dataflow.WS, tree_reduction = false, tile_latency = 0,
    output_delay = 1, tileRows = 1, tileColumns = 1,
    meshRows = dim, meshColumns = dim, leftBanks = 1, upBanks = 1, outBanks = 1))

  val accs = Seq.fill(2)(Module(new AccumulatorMem(rows, fullDataType, scale_func, scale_t,
    acc_singleported = false, acc_sub_banks = 1, use_shared_ext_mem = false,
    use_tl_ext_ram = false, acc_latency = 2, acc_type = SInt(accW.W), is_dummy = false)))
  val adders = Module(new AccPipeShared(1, fullDataType, 2))

  val vpu = Module(new AccumulatorScale(fullDataType, rDataType, scale_t,
    read_small_data = true, read_full_data = false,
    scale_func = scale_func, num_scale_units = -1, latency = 1,
    has_nonlinear_activations = true, has_normalizations = false))

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

  for (k <- 0 until 2) {
    adders.io.in_sel(k) := accs(k).io.adder.valid
    adders.io.ina(k) := accs(k).io.adder.op1
    adders.io.inb(k) := accs(k).io.adder.op2
    accs(k).io.adder.sum := adders.io.out
  }

  // -- mesh rows into the accumulator ------------------------------------------
  val resp = mxu.io.resp
  val tag = resp.bits.tag.id
  val live = resp.valid && !tag(7)
  val bank = tag(0)
  val accumulate = tag(1)
  val lastBlock = tag(2)
  val sub = tag(6, 3)
  val rowInReq = RegInit(0.U(log2Up(dim max 2).W))
  when (resp.valid) { rowInReq := Mux(resp.bits.last, 0.U, rowInReq + 1.U) }
  val addr = sub * dim.U + rowInReq

  for (k <- 0 until 2) {
    val w = accs(k).io.write
    w.valid := live && bank === k.U
    w.bits.addr := addr
    w.bits.data := resp.bits.data
    w.bits.acc := accumulate
    w.bits.mask.foreach(_ := true.B)
    // The mesh's result bus is Valid: nothing can hold it back.
    assert(!w.valid || w.ready, "the accumulator refused a row the mesh emitted")
  }

  // -- a finished group out through the scale unit -----------------------------
  val reading = RegInit(false.B)
  val rdBank = RegInit(0.U(1.W))
  val rdRow = RegInit(0.U(log2Up(rows).W))
  for (k <- 0 until 2) {
    val r = accs(k).io.read.req
    r.valid := reading && rdBank === k.U
    r.bits.addr := rdRow
    r.bits.scale := io.scale
    r.bits.act := (if (relu) Activation.RELU else Activation.NONE)
    r.bits.full := false.B
    r.bits.fromDMA := false.B
    r.bits.igelu_qb := 0.S
    r.bits.igelu_qc := 0.S
    r.bits.iexp_qln2 := 0.S
    r.bits.iexp_qln2_inv := 0.S
  }
  val rdFire = Mux(rdBank === 0.U, accs(0).io.read.req.fire, accs(1).io.read.req.fire)
  when (rdFire) {
    rdRow := rdRow + 1.U
    when (rdRow === (rows - 1).U) { reading := false.B }
  }
  val groupDone = live && lastBlock && resp.bits.last && sub === (subs - 1).U
  when (groupDone) {
    assert(!reading || (rdFire && rdRow === (rows - 1).U),
      "a group finished while the previous one was still being read out")
    reading := true.B
    rdBank := bank
    rdRow := 0.U
  }

  val v0 = accs(0).io.read.resp.valid
  vpu.io.in.valid := v0 || accs(1).io.read.resp.valid
  vpu.io.in.bits.acc_read_resp := Mux(v0, accs(0).io.read.resp.bits, accs(1).io.read.resp.bits)
  vpu.io.in.bits.acc_read_resp.acc_bank_id := 0.U
  vpu.io.in.bits.mean := 0.S
  vpu.io.in.bits.max := 0.S
  vpu.io.in.bits.inv_stddev := 0.S
  vpu.io.in.bits.inv_sum_exp := 0.S
  for (k <- 0 until 2) accs(k).io.read.resp.ready := vpu.io.in.ready

  io.out <> vpu.io.out
}

object ElaborateMxuAccVpu extends App {
  val dim = sys.env.getOrElse("MESH_DIM", "4").toInt
  val rows = sys.env.getOrElse("ACC_ROWS", "16").toInt
  val relu = sys.env.getOrElse("ACT", "relu") == "relu"
  // E3's configuration keeps its directory; any other names what differs.
  val dir = "mxuaccvpu_out_" + dim + (if (rows != 16) "_r" + rows else "") +
    (if (relu) "" else "_noact")
  println("MXUACCVPU_ELABORATE_START dim=" + dim + " rows=" + rows + " relu=" + relu)
  val v = (new ChiselStage).emitVerilog(new MxuAccVpu(dim, rows, relu),
    Array("--target-dir", dir))
  println("MXUACCVPU_ELABORATE_OK dim=" + dim + " dir=" + dir + " verilog_chars=" + v.length)
}

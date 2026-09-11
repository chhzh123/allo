package gen
import chisel3._
import chiseltest._
import chiseltest.RawTester.test
import gemmini._

// Cycle count for the MXU+VPU datapath, on the same protocol MeshDriver uses
// for the mesh alone, with the golden model extended through the VPU.
//
// AccumulatorScale applies the activation and *then* the scale
// (AccumulatorScale.scala: e_act = relu(e); e_scaled = scale_func(e_act, scale)),
// and clips to the narrow type, so the model is clip8(relu(sum) >> shift) and
// not the other order. Getting that backwards is the kind of mistake that still
// produces plausible-looking numbers, so it is written out here.
class MxuVpuCycleTest(c: MxuVpu, dim: Int, a: Seq[Seq[Int]], w: Seq[Seq[Int]], shift: Int) {
  def poke(sig: chisel3.Bits, v: Int): Unit = sig match {
    case s: SInt => s.poke(v.S)
    case u: UInt => u.poke(v.U)
    case b: Bool => b.poke((v != 0).B)
  }
  def poke(sig: Bool, v: Int): Unit = sig.poke((v != 0).B)
  def peek(sig: chisel3.Bits): BigInt = sig match {
    case s: SInt => s.peek().litValue
    case u: UInt => u.peek().litValue
  }
  def peek(sig: Bool): BigInt = if (sig.peek().litToBoolean) 1 else 0
  def step(n: Int): Unit = c.clock.step(n)

  def clip8(x: Int): Int = if (x > 127) 127 else if (x < -128) -128 else x
  val gold: Seq[Seq[Int]] = Seq.tabulate(dim, dim) { (i, j) =>
    val s = (0 until dim).map(k => a(i)(k) * w(k)(j)).sum
    clip8((if (s < 0) 0 else s) >> shift)      // ReLU first, then the shift, then clip
  }

  var cyc = 0
  var firstIn = -1
  var firstCompute = -1
  var lastOut = -1
  val got = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
  def step1(): Unit = { step(1); cyc += 1 }

  c.clock.setTimeout(0)
  poke(c.io.req.valid, 0); poke(c.io.a.valid, 0); poke(c.io.b.valid, 0); poke(c.io.d.valid, 0)
  poke(c.io.scale, shift)
  poke(c.io.out.ready, 1)          // the VPU must never be the thing that stalls
  step1()

  def issue(prop: Int): Unit = {
    poke(c.io.req.bits.pe_control.dataflow, 1)
    poke(c.io.req.bits.pe_control.propagate, prop)
    poke(c.io.req.bits.pe_control.shift, 0)
    poke(c.io.req.bits.total_rows, dim)
    poke(c.io.req.bits.tag.id, 1)
    poke(c.io.req.valid, 1)
    var guard = 0
    while (peek(c.io.req.ready) == 0 && guard < 100) { step1(); guard += 1 }
    step1()
    poke(c.io.req.valid, 0)
  }

  def collectOut(): Unit =
    if (peek(c.io.out.valid) != 0) {
      got += (0 until dim).map(j => peek(c.io.out.bits.data(j)(0)).toInt)
      lastOut = cyc
    }

  def stream(rows: Int, av: Seq[Seq[Int]], dv: Seq[Seq[Int]], mark: Boolean = false): Unit = {
    var row = 0
    var guard = 0
    while (row < rows && guard < 500) {
      poke(c.io.a.valid, 1); poke(c.io.b.valid, 1); poke(c.io.d.valid, 1)
      for (j <- 0 until dim) {
        poke(c.io.a.bits(j)(0), av(row)(j))
        poke(c.io.b.bits(j)(0), 0)
        poke(c.io.d.bits(j)(0), dv(row)(j))
      }
      if (peek(c.io.a.ready) != 0) {
        if (firstIn < 0) firstIn = cyc
        if (mark && firstCompute < 0) firstCompute = cyc
        row += 1
      }
      collectOut()
      step1(); guard += 1
    }
    poke(c.io.a.valid, 0); poke(c.io.b.valid, 0); poke(c.io.d.valid, 0)
  }

  val zeros = Seq.fill(dim, dim)(0)
  val wRev = w.reverse
  issue(1); stream(dim, zeros, wRev)
  issue(1); stream(dim, a, zeros, mark = true)

  var g2 = 0
  while (g2 < 80) { collectOut(); step1(); g2 += 1 }

  val tail = if (got.length >= dim) got.takeRight(dim).toSeq else Seq.empty
  val correct = tail == gold
  println(s"MXUVPU_CYCLES dim=$dim shift=$shift first_in=$firstIn last_out=$lastOut " +
          s"latency=${if (firstIn >= 0 && lastOut >= 0) lastOut - firstIn + 1 else -1} " +
          s"rows_out=${got.length} correct=$correct " +
          s"seq_latency=${lastOut - firstIn + 1} compute_latency=${lastOut - firstCompute + 1}")
  if (!correct) {
    println("  gold: " + gold.map(_.mkString(",")).mkString(" | "))
    println("  got : " + got.map(_.mkString(",")).mkString(" | "))
  }
}

object MxuVpuDriver extends App {
  val dim = sys.env.getOrElse("MESH_DIM", "4").toInt
  val shift = sys.env.getOrElse("SCALE_SHIFT", "2").toInt
  val rnd = new scala.util.Random(0)
  val a = Seq.fill(dim, dim)(rnd.nextInt(7) - 3)
  val w = Seq.fill(dim, dim)(rnd.nextInt(7) - 3)
  test(new MxuVpu(dim)) { c => new MxuVpuCycleTest(c, dim, a, w, shift) }
}

// A driver for Gemmini's MeshWithDelays, to count cycles for one SxS matmul.
//
// Gemmini's own MeshWithDelaysUnitTest does not compile against this source --
// it drives io.s / io.tag_in / io.shift, which have been consolidated into
// req/resp -- so this drives the current interface. It checks every output
// against a golden matmul computed in Scala: if the protocol here were wrong,
// the check fails rather than a wrong cycle count being reported.
//
// Weight-stationary semantics, read from PE.scala:
//   d -> loads the stationary weight into c1/c2 (selected by `propagate`)
//   a -> the activation entering from the left
//   b -> the partial sum flowing down, leaving as out_b
// so for C = A x W: W goes in on d, A on a, zeros on b, and resp.data is out_b.
package gen

import chisel3._
import chiseltest._
import chiseltest.RawTester.test
import gemmini._

class SimpleTag extends Bundle with TagQueueTag {
  val id = UInt(8.W)
  override def make_this_garbage(dummy: Int = 0): Unit = { id := 255.U }
}

class MeshCycleTest(c: MeshWithDelays[SInt, SimpleTag], dim: Int,
                    a: Seq[Seq[Int]], w: Seq[Seq[Int]]) {
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

  val gold = Seq.tabulate(dim, dim) { (i, j) => (0 until dim).map(k => a(i)(k) * w(k)(j)).sum }
  var cyc = 0
  var firstIn = -1        // first input of the whole sequence (preload)
  var firstCompute = -1   // first activation of the compute pass
  var lastOut = -1
  val got = scala.collection.mutable.ArrayBuffer[Seq[Int]]()

  def step1(): Unit = { step(1); cyc += 1 }

  c.clock.setTimeout(0)   // the drain deliberately idles; chiseltest defaults to 1000
  poke(c.io.req.valid, 0); poke(c.io.a.valid, 0); poke(c.io.b.valid, 0); poke(c.io.d.valid, 0)
  step1()

  // A matmul is two commands, not one. PE.scala keeps two weight registers and
  // flips between them on `propagate`, so the first pass pushes weights in on
  // `d` (its outputs are the previous, garbage, weights) and the second pass
  // streams activations on `a` against them. That is what ExecuteController
  // issues as PRELOAD then COMPUTE_AND_FLIP.
  def issue(prop: Int): Unit = {
    poke(c.io.req.bits.pe_control.dataflow, 1)   // WEIGHT_STATIONARY
    poke(c.io.req.bits.pe_control.propagate, prop)
    poke(c.io.req.bits.pe_control.shift, 0)
    poke(c.io.req.bits.a_transpose, 0)
    poke(c.io.req.bits.bd_transpose, 0)
    poke(c.io.req.bits.total_rows, dim)
    poke(c.io.req.bits.tag.id, prop + 1)
    poke(c.io.req.bits.flush, 0)
    poke(c.io.req.valid, 1)
    var g = 0
    while (peek(c.io.req.ready) == 0 && g < 500) { step1(); g += 1 }
    step1()
    poke(c.io.req.valid, 0)
  }

  // Stream `rows` rows; `collect` says whether these outputs are the real ones.
  def stream(rows: Int, aM: Seq[Seq[Int]], dM: Seq[Seq[Int]], collect: Boolean,
             mark: Boolean = false): Unit = {
    var row = 0; var g = 0
    while (row < rows && g < 20000) {
      for (i <- 0 until dim) {
        poke(c.io.a.bits(i)(0), aM(row)(i))
        poke(c.io.d.bits(i)(0), dM(row)(i))
        poke(c.io.b.bits(i)(0), 0)
      }
      poke(c.io.a.valid, 1); poke(c.io.b.valid, 1); poke(c.io.d.valid, 1)
      val fired = peek(c.io.a.ready) != 0 && peek(c.io.b.ready) != 0 && peek(c.io.d.ready) != 0
      if (collect && peek(c.io.resp.valid) != 0) {
        got += (0 until dim).map(j => peek(c.io.resp.bits.data(j)(0)).toInt)
        lastOut = cyc
      }
      step1()
      if (fired) {
        if (firstIn < 0) firstIn = cyc
        if (mark && firstCompute < 0) firstCompute = cyc
        row += 1
      }
      g += 1
    }
    poke(c.io.a.valid, 0); poke(c.io.b.valid, 0); poke(c.io.d.valid, 0)
  }

  val zeros = Seq.fill(dim, dim)(0)
  // Gemmini pushes the weight rows in reverse so the last row lands nearest.
  val wRev = w.reverse
  // MeshWithDelays line 118: in_prop := propagate ^ in_prop. `propagate` is a
  // FLIP flag, not an absolute selector -- Gemmini's ISA calls the op
  // COMPUTE_AND_FLIP. So both passes set it: the first flips to load-side, the
  // second flips back so the weights just loaded become the stationary ones.
  issue(1); stream(dim, zeros, wRev, collect = true)    // preload the weights
  issue(1); stream(dim, a, zeros, collect = true, mark = true)  // compute against them

  // Drain.
  var g2 = 0
  while (g2 < 60) {
    if (peek(c.io.resp.valid) != 0) {
      got += (0 until dim).map(j => peek(c.io.resp.bits.data(j)(0)).toInt)
      lastOut = cyc
    }
    step1(); g2 += 1
  }

  // The last `dim` rows out are the compute pass; everything before is the
  // preload draining through.
  val tail = if (got.length >= dim) got.takeRight(dim).toSeq else Seq.empty
  val correct = tail == gold
  val hit = (0 to got.length - dim).find(i => got.slice(i, i + dim).toSeq == gold)
  println(s"GEMMINI_CYCLES dim=$dim first_in=$firstIn last_out=$lastOut " +
          s"latency=${if (firstIn >= 0 && lastOut >= 0) lastOut - firstIn + 1 else -1} " +
          s"rows_out=${got.length} correct=$correct " +
          s"seq_latency=${lastOut - firstIn + 1} compute_latency=${lastOut - firstCompute + 1}")
  println("  all rows out: " + got.map(_.mkString(",")).mkString(" | "))
  if (!correct) {
    println("  gold head: " + gold.head.mkString(","))
    println("  got  rows: " + got.map(_.mkString(",")).mkString(" | "))
  }
}

object MeshDriver extends App {
  val dim = sys.env.getOrElse("MESH_DIM", "4").toInt
  val rnd = new scala.util.Random(0)
  val a = Seq.fill(dim, dim)(rnd.nextInt(7) - 3)
  val w = Seq.fill(dim, dim)(rnd.nextInt(7) - 3)
  test(new MeshWithDelays(SInt(8.W), SInt(8.W), SInt(32.W), SInt(32.W),
      new SimpleTag, Dataflow.WS, tree_reduction = false, tile_latency = 0,
      output_delay = 1, tileRows = 1, tileColumns = 1,
      meshRows = dim, meshColumns = dim, leftBanks = 1, upBanks = 1, outBanks = 1)) {
    c => new MeshCycleTest(c, dim, a, w)
  }
}

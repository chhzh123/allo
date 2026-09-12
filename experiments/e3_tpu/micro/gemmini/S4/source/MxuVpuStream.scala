package gen
import chisel3._
import chiseltest._
import chiseltest.RawTester.test
import gemmini._

// The E3 microbenchmark on Gemmini: `tiles` independent S x S x S int8 GEMMs
// streamed back to back through MeshWithDelays and AccumulatorScale, with the
// bias, the requantisation, the ReLU and the clip all on device.
//
// `MxuVpuDriver` runs one matmul in two passes and measures its latency. This
// runs a stream, because an interval is not a latency and neither substitutes
// for the other.
//
// The stimulus is not generated here. It is read from the file
// `tests/dataflow/spmw/test_spmw_tpu_micro.py` writes, so the two systems are
// checked against the same bytes rather than against two generators that agree
// today. If they disagree, that is the finding.
object Stim {
  def load(path: String): Map[String, Array[Int]] = {
    val src = scala.io.Source.fromFile(path)
    try {
      src
        .getLines()
        .filter(_.nonEmpty)
        .map { line =>
          val i = line.indexOf(' ')
          line.substring(0, i) -> line.substring(i + 1).trim.split("\\s+").map(_.toInt)
        }
        .toMap
    } finally src.close()
  }
}

// The weight-stationary double buffer, and why the passes are offset by one.
//
// MeshWithDelays *toggles* its internal propagate on every request --
// `in_prop := io.req.bits.pe_control.propagate ^ in_prop` -- and the PE writes
// `d` into whichever of c1/c2 it is not multiplying by. So a pass that computes
// with the weights loaded last time simultaneously shifts in the weights for
// next time, and `tiles` tiles take `tiles + 1` passes: a warm-up that loads
// W(0) against zero activations, then one pass per tile.
class MxuVpuStreamTest(
    c: MxuVpu,
    dim: Int,
    tiles: Int,
    shift: Int,
    a: Array[Int],
    w: Array[Int],
    bias: Array[Int],
    want: Array[Int]
) {
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

  val tile = dim * dim
  def at(buf: Array[Int], t: Int, r: Int, col: Int): Int = buf(t * tile + r * dim + col)

  var cyc = 0
  var firstIn = -1
  var lastOut = -1
  val got = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
  val tileDone = Array.fill(tiles)(-1)
  def step1(): Unit = { c.clock.step(1); cyc += 1 }

  // The rows the mesh has emitted, against the rows this pass expects. Pass 0
  // is the warm-up and its `dim` rows are discarded; after that row
  // `(t + 1) * dim + r` is tile t's row r.
  def collectOut(): Unit =
    if (peek(c.io.out.valid) != 0) {
      got += (0 until dim).map(j => peek(c.io.out.bits.data(j)(0)).toInt)
      lastOut = cyc
      val done = got.length / dim - 1
      if (done >= 1 && done <= tiles && got.length % dim == 0 && tileDone(done - 1) < 0)
        tileDone(done - 1) = cyc
    }

  c.clock.setTimeout(0)
  poke(c.io.req.valid, 0)
  poke(c.io.a.valid, 0); poke(c.io.b.valid, 0); poke(c.io.d.valid, 0)
  poke(c.io.scale, shift)
  poke(c.io.out.ready, 1) // the VPU must never be the thing that stalls
  step1()

  def issue(): Unit = {
    poke(c.io.req.bits.pe_control.dataflow, 1)
    poke(c.io.req.bits.pe_control.propagate, 1) // toggles in_prop; see above
    poke(c.io.req.bits.pe_control.shift, 0)
    poke(c.io.req.bits.a_transpose, 0)
    poke(c.io.req.bits.bd_transpose, 0)
    poke(c.io.req.bits.flush, 0)
    poke(c.io.req.bits.total_rows, dim)
    poke(c.io.req.bits.tag.id, 1)
    poke(c.io.req.valid, 1)
    var guard = 0
    while (peek(c.io.req.ready) == 0 && guard < 100) { collectOut(); step1(); guard += 1 }
    collectOut(); step1()
    poke(c.io.req.valid, 0)
  }

  // One pass: `dim` rows of activations, and the *next* tile's weights shifted
  // in behind them. `d` goes in bottom row first -- the weights climb the
  // array -- which is what `MxuVpuDriver`'s `w.reverse` is doing too.
  def pass(actTile: Int, wTile: Int): Unit = {
    var row = 0
    var guard = 0
    while (row < dim && guard < 500) {
      poke(c.io.a.valid, 1); poke(c.io.b.valid, 1); poke(c.io.d.valid, 1)
      for (j <- 0 until dim) {
        poke(c.io.a.bits(j)(0), if (actTile < 0) 0 else at(a, actTile, row, j))
        // The bias enters as the partial sum at the top of the array, on every
        // beat including the warm-up: it is constant down the rows, so feeding
        // it unconditionally makes it immune to the array's input skew.
        poke(c.io.b.bits(j)(0), bias(j))
        poke(c.io.d.bits(j)(0), if (wTile < 0) 0 else at(w, wTile, dim - 1 - row, j))
      }
      if (peek(c.io.a.ready) != 0) {
        if (firstIn < 0) firstIn = cyc
        row += 1
      }
      collectOut()
      step1(); guard += 1
    }
    poke(c.io.a.valid, 0); poke(c.io.b.valid, 0); poke(c.io.d.valid, 0)
  }

  issue(); pass(-1, 0) // warm-up: no activations, load W(0)
  for (t <- 0 until tiles) {
    issue()
    pass(t, if (t + 1 < tiles) t + 1 else -1)
  }

  var drain = 0
  while (got.length < (tiles + 1) * dim && drain < 200) { collectOut(); step1(); drain += 1 }

  // Every tile checked, not the last one: a stream that is right at the end and
  // wrong in the middle is exactly what a single check misses.
  val rows = got.toSeq
  var wrong = 0
  var firstBad = -1
  for (t <- 0 until tiles; r <- 0 until dim) {
    val idx = (t + 1) * dim + r
    val mine = if (idx < rows.length) rows(idx) else Seq.fill(dim)(Int.MinValue)
    val gold = (0 until dim).map(j => at(want, t, r, j))
    if (mine != gold) { wrong += 1; if (firstBad < 0) firstBad = t * dim + r }
  }
  val correct = wrong == 0 && rows.length == (tiles + 1) * dim

  val gaps = (1 until tiles).map(t => tileDone(t) - tileDone(t - 1)).filter(_ > 0)
  val steady = gaps.drop(gaps.length / 4).sorted
  val median = if (steady.isEmpty) -1 else steady(steady.length / 2)
  val latency = if (tileDone(0) >= 0 && firstIn >= 0) tileDone(0) - firstIn + 1 else -1

  println(
    s"MXUVPU_STREAM dim=$dim tiles=$tiles shift=$shift correct=$correct " +
      s"rows_out=${rows.length} want_rows=${(tiles + 1) * dim} wrong_rows=$wrong " +
      s"first_in=${firstIn + 1} last_out=${lastOut + 1} " +
      s"latency=$latency interval=$median " +
      s"interval_min=${if (steady.isEmpty) -1 else steady.head} " +
      s"interval_max=${if (steady.isEmpty) -1 else steady.last} " +
      s"total=${if (lastOut >= 0 && firstIn >= 0) lastOut - firstIn + 1 else -1}"
  )
  for (t <- 0 until tiles) println(s"MXUVPU_TILE t=$t done=${tileDone(t) + 1}")
  if (!correct) {
    println(s"  first wrong output row: $firstBad")
    if (firstBad >= 0) {
      val t = firstBad / dim
      val r = firstBad % dim
      val idx = (t + 1) * dim + r
      println("  gold: " + (0 until dim).map(j => at(want, t, r, j)).mkString(","))
      println("  got : " + (if (idx < rows.length) rows(idx).mkString(",") else "(missing)"))
    }
  }
}

object MxuVpuStreamDriver extends App {
  val path = sys.env.getOrElse("STIM", "stim.txt")
  val f = Stim.load(path)
  val dim = f("S")(0)
  val tiles = f("TILES")(0)
  val shift = f("SHIFT")(0)
  require(f("A").length == tiles * dim * dim, s"A has ${f("A").length} entries")
  require(f("B").length == tiles * dim * dim, s"B has ${f("B").length} entries")
  require(f("C").length == tiles * dim * dim, s"C has ${f("C").length} entries")
  require(f("BIAS").length == dim, s"BIAS has ${f("BIAS").length} entries")
  println(s"MXUVPU_STREAM_STIM $path dim=$dim tiles=$tiles shift=$shift")
  test(new MxuVpu(dim)) { c =>
    new MxuVpuStreamTest(c, dim, tiles, shift, f("A"), f("B"), f("BIAS"), f("C"))
  }
}

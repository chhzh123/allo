package gen
import chisel3._
import chisel3.stage.ChiselStage
import gemmini._

// Feasibility gate: can the accumulator scale path elaborate on its own, with
// no scratchpad, controller or RoCC? Scope is scale + ReLU, so
// has_normalizations = false gates LAYERNORM, IGELU and SOFTMAX off.
object Gate extends App {
  val dim = 4
  val inW = 8
  val accW = 32
  val full = Vec(dim, Vec(dim, SInt(accW.W)))
  val small = Vec(dim, Vec(dim, SInt(inW.W)))
  val scale_t = SInt(accW.W)
  // Gemmini's integer scale: multiply then arithmetic-shift -- the requantise
  // step SPMW spells as MUL followed by SHR.
  val scale_func = (v: SInt, s: SInt) => (v * s) >> 8.U
  println("GATE_ELABORATE_START")
  val v = (new ChiselStage).emitVerilog(
    new AccumulatorScale(full, small, scale_t,
      read_small_data = true, read_full_data = false,
      scale_func = scale_func,
      num_scale_units = -1, latency = 1,
      has_nonlinear_activations = true,
      has_normalizations = false),
    Array("--target-dir", "gate_out"))
  println("GATE_ELABORATE_OK verilog_chars=" + v.length)
}

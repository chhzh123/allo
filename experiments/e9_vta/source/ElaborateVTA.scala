// Elaborate VTA's datapath at its default configuration, for comparison with
// Gemmini's MxuVpuNorm and SPMW's block engine.
//
// `CoreConfig` is VTA's own default and needs no edit: batch = 1,
// blockIn = blockOut = 16, inpBits = wgtBits = 8, accBits = 32.  That is 256
// int8 multiply-accumulates into int32 -- the same arithmetic as the 16x16
// arrays on the other two sides, though reached by an adder tree rather than
// a systolic array.
//
// Three scopes are emitted because they answer different questions:
//   TensorGemm  the GEMM core alone -- the counterpart of Gemmini's mesh
//   TensorAlu   the tensor ALU -- the counterpart of the scale path, and the
//               module whose op set (min, max, add, shr) is the reason VTA
//               cannot run this block's nonlinearities on device
//   Core        fetch, load, compute, store and every scratchpad: VTA's whole
//               engine, which is a larger scope than either of the others
package gen

import chisel3._
import chisel3.stage.ChiselStage
import vta.core._
import vta.shell._
import vta.util.config._

// `CoreConfig` alone is not enough: `TensorParams` reads `ShellKey` for the
// memory interface width, so a shell config has to be mixed in.  `F1Config`
// is VTA's Xilinx PCIe card -- 64-bit AXI address and data -- which is the
// closest of the three to a U280; `PynqConfig` is a 32-bit Zynq part and
// `De10Config` is Intel.
class VTAOnU280 extends Config(new CoreConfig ++ new F1Config)

object ElaborateVTA extends App {
  implicit val p: Parameters = new VTAOnU280
  val what = sys.env.getOrElse("VTA_TOP", "TensorGemm")
  val dir = "vta_out_" + what
  println("VTA_ELABORATE_START " + what)
  val stage = new ChiselStage
  what match {
    case "TensorGemm" => stage.emitVerilog(new TensorGemm, Array("--target-dir", dir))
    case "TensorAlu"  => stage.emitVerilog(new TensorAlu, Array("--target-dir", dir))
    case "Core"       => stage.emitVerilog(new Core, Array("--target-dir", dir))
    case other        => sys.error("unknown VTA_TOP " + other)
  }
  println("VTA_ELABORATE_OK " + what)
}

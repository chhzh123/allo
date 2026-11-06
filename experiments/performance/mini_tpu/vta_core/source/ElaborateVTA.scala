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

/** The same at another array width, so VTA can be swept the way E3 sweeps
  * the other two.  Only `blockIn`/`blockOut` move; the widths, the buffer
  * depths and the shell stay exactly as VTA ships them. */
class VTAWidth(n: Int) extends Config(
  new Config((site, here, up) => {
    case CoreKey =>
      CoreParams(batch = 1, blockOut = n, blockOutFactor = 1, blockIn = n,
        inpBits = 8, wgtBits = 8, uopBits = 32, accBits = 32, outBits = 8,
        uopMemDepth = 2048, inpMemDepth = 2048, wgtMemDepth = 1024,
        accMemDepth = 2048, outMemDepth = 2048, instQueueEntries = 512)
  }) ++ new F1Config)

object ElaborateVTA extends App {
  val width = sys.env.getOrElse("VTA_WIDTH", "16").toInt
  implicit val p: Parameters =
    if (width == 16) new VTAOnU280 else new VTAWidth(width)
  val what = sys.env.getOrElse("VTA_TOP", "TensorGemm")
  val dir = "vta_out_" + what + (if (width == 16) "" else "_w" + width)
  println("VTA_ELABORATE_START " + what + " width=" + width)
  val stage = new ChiselStage
  what match {
    case "TensorGemm" => stage.emitVerilog(new TensorGemm, Array("--target-dir", dir))
    case "TensorAlu"  => stage.emitVerilog(new TensorAlu, Array("--target-dir", dir))
    case "Core"       => stage.emitVerilog(new Core, Array("--target-dir", dir))
    case other        => sys.error("unknown VTA_TOP " + other)
  }
  println("VTA_ELABORATE_OK " + what + " width=" + width)
}

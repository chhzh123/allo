// Elaborate Gemmini's Mesh on its own, at the size and dataflow given by the
// environment, so it can be placed and routed the same way the SPMW arrays are.
package gen
import chisel3._
import gemmini._

object Main extends App {
  val dim  = sys.env.getOrElse("MESH_DIM", "8").toInt
  val df   = sys.env.getOrElse("MESH_DF", "WS") match {
    case "WS" => Dataflow.WS
    case "OS" => Dataflow.OS
    case _    => Dataflow.BOTH
  }
  val outW = sys.env.getOrElse("MESH_OUT_BITS", "32").toInt
  val dir  = sys.env.getOrElse("MESH_OUT", "/scratch/hc676/gemmini_build/out")
  // int8 x int8 -> int32, one PE per tile: an dim x dim array of multipliers,
  // the same shape the SPMW GEMM arrays are built at.
  (new chisel3.stage.ChiselStage).emitVerilog(
    new Mesh(SInt(8.W), SInt(8.W), SInt(outW.W), SInt(32.W),
             df, tree_reduction = false, tile_latency = 0,
             max_simultaneous_matmuls = 5, output_delay = 1,
             tileRows = 1, tileColumns = 1,
             meshRows = dim, meshColumns = dim),
    Array("--target-dir", dir))
  println(s"GEMMINI_MESH_EMITTED dim=$dim df=${sys.env.getOrElse("MESH_DF","WS")} out=$dir")
}

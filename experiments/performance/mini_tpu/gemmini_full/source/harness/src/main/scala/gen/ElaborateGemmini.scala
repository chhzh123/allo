package gen

import chisel3._
import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.subsystem.{SystemBusKey, WithoutTLMonitors}
import gemmini._

// Gemmini's whole accelerator -- reservation station, load, store and execute
// controllers, scratchpad with its DMA and TLB, loop unrollers, mesh -- as it
// sits in a Rocket tile. Elaborating rocket-chip's own example system with
// Gemmini as its RoCC accelerator gives Gemmini exactly the parameters and the
// TileLink edge it has in a real SoC; the `Gemmini` module is then taken out of
// the emitted SystemVerilog and routed and simulated on its own.
//
//   MESH_DIM        4 | 8 | 16   the mesh, DIM x DIM cells
//   GEMMINI_CONFIG  lean         Gemmini's `leanConfig`, as shipped: weight
//                                stationary only, float32 scale on load and
//                                on accumulator read-out
//                   default      Gemmini's `defaultConfig`: both dataflows
//                   shift        `leanConfig` with the accumulator's float
//                                scale replaced by a right shift and no scale
//                                on load -- the epilogue SPMW and VTA have
//                   matmul       `shift` with what a GEMM never uses switched
//                                off through Gemmini's own options: the
//                                convolution loop unroller, pooling, training
//                                and depthwise convolutions, first-layer
//                                optimisations
//   GEMMINI_BUS     64 | 128     the tile's system bus. rocket-chip's is 64
//                                bits, which is also VTA's memory port.
//                                Gemmini's DMA is 128 bits wide, and Chipyard
//                                widens the system bus to match it
//                                (`WithSystemBusWidth(128)`); on a 64-bit bus
//                                Gemmini's own width adapter sits in between.
class WithSystemBusWidth(bits: Int) extends Config((site, here, up) => {
  case SystemBusKey => up(SystemBusKey, site).copy(beatBytes = bits / 8)
})

object ElaborateGemmini extends App {
  val dim = sys.env.getOrElse("MESH_DIM", "16").toInt
  val variant = sys.env.getOrElse("GEMMINI_CONFIG", "lean")
  // GEMMINI_SIM=1 keeps firtool's register and memory initialisation, which a
  // simulation needs: Gemmini's datapath registers have no reset.
  val sim = sys.env.getOrElse("GEMMINI_SIM", "0") == "1"
  val bus = sys.env.getOrElse("GEMMINI_BUS", "64").toInt
  val dir = s"out/${variant}_$dim" + (if (bus == 64) "" else s"_b$bus") + (if (sim) "_sim" else "")
  new java.io.File(dir).mkdirs()

  def emit[U <: Data, V <: Data](base: GemminiArrayConfig[SInt, U, V]): Unit = {
    val cfg = base.copy(meshRows = dim, meshColumns = dim,
      headerFileName = s"$dir/gemmini_params.h")
    println(s"GEMMINI_ELABORATE_START dim=$dim config=$variant")
    implicit val p: Parameters = new Config(
      new LeanGemminiConfig(cfg) ++
      new WithSystemBusWidth(bus) ++
      new WithoutTLMonitors ++
      new freechips.rocketchip.system.DefaultConfig)
    circt.stage.ChiselStage.emitSystemVerilogFile(
      new freechips.rocketchip.system.TestHarness()(p),
      args = Array("--target-dir", dir, "--split-verilog"),
      firtoolOpts = (if (sim) Array[String]() else Array("-disable-all-randomization")) ++ Array(
        // rocket-chip annotates its SRAMs, address maps and register fields for
        // its own flow; firtool has no handler for them and none is needed.
        "-disable-annotation-unknown",
        "-disable-annotation-classless",
        "-strip-debug-info",
        "--lowering-options=emittedLineLength=2048,noAlwaysComb,disallowLocalVariables,disallowPortDeclSharing"))
    println(s"GEMMINI_ELABORATE_OK dim=$dim config=$variant dir=$dir")
  }

  def shifting = GemminiConfigs.leanConfig.copy[SInt, Float, UInt](
    mvin_scale_args = None,
    acc_scale_args = Some(ScaleArguments(
      (t: SInt, u: UInt) => (t >> u),
      1, UInt(5.W), -1,
      identity = "0",
      c_str = "((x) >> (scale))")))

  variant match {
    case "lean" => emit(GemminiConfigs.leanConfig)
    case "default" => emit(GemminiConfigs.defaultConfig)
    case "shift" => emit(shifting)
    case "matmul" =>
      emit(shifting.copy(
        has_loop_conv = false,
        has_max_pool = false,
        has_training_convs = false,
        has_dw_convs = false,
        has_first_layer_optimizations = false))
  }
}

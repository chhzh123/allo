# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a spatial array the way SPMW is meant to be built.

HLS synthesises the *unit* -- one per role -- and Vivado assembles the array
from the exported IPs.  The C synthesis and IP export are paid once per role, so
the cost tracks the role count rather than the grid: a 2-D mesh has nine roles
whatever its size.

Run it on a machine with Vitis and Vivado on PATH::

    python3 scripts/spmw_build_array.py --size 4 --out /scratch/$USER/spmw_array

It stages one HLS project per role, synthesises and exports them *concurrently*
-- roles are independent by construction, which is what having roles means --
then writes a Vivado script that reads the exported IPs together with the
structural fabric and elaborates the whole array.  ``--cosim`` goes further and
simulates it against ``target="ref"``.

Measured on brg-zhang-xcel, systolic GEMM, xcu280: nine roles in **40.5s of wall
clock** (361s of CPU across 48 cores), and Vivado assembles the array in ~69s.
The same nine cost 332s run serially, and 545s before their unused headers were
trimmed.

The whole-array program also synthesises -- ``spmw.build`` partitions the tensor
arguments, without which HLS dataflow rejects the shared result array -- but it
is one monolithic ``csynth`` and cannot be split across cores: 39s at 9 sites,
280s at 144, 807s at 256.  Decomposing into roles is what *creates* the
parallelism, which is a second reason for the split beyond the flat cost.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw import rtl  # pylint: disable=wrong-import-position
from allo.spmw.cosim import (  # pylint: disable=wrong-import-position
    render_testbench,
)
from allo.spmw.role_ip import (  # pylint: disable=wrong-import-position
    MoverEmitter,
    UnitEmitter,
    axi_data_width,
    build_mover,
    build_unit,
    check_wrapper,
    mover_wrapper_sv,
    optimise,
    trim_includes,
    wrapper_sv,
)

PART = "xcu280-fsvh2892-2L-e"

ROLE_TCL = """open_project prj
set_top {name}_0
add_files kernel.cpp
open_solution sol
set_part {part}
create_clock -period {period:.2f} -name default
config_interface -clock_enable=0
set_directive_interface -mode ap_ctrl_none "{name}_0" return
csynth_design
export_design -format ip_catalog
exit
"""

MOVER_TCL = """open_project prj
set_top {name}_0
add_files kernel.cpp
open_solution sol
set_part {part}
create_clock -period {period:.2f} -name default
config_interface -clock_enable=0 -m_axi_max_widen_bitwidth {widen}
set_directive_interface -mode m_axi -offset direct -depth {depth} "{name}_0" {arg}
csynth_design
export_design -format ip_catalog
exit
"""

ASSEMBLE_TCL = """create_project -in_memory -part {part}
set root {root}
add_files [glob $root/*.sv]
foreach r {{{roles}}} {{
  set d $root/$r
  add_files [glob -nocomplain $d/*.sv]
  add_files [glob -nocomplain $d/prj/sol/syn/verilog/*.v]
  foreach x [glob -nocomplain $d/prj/sol/impl/ip/tmp.srcs/sources_1/ip/*/*.xci] {{
    read_ip $x
  }}
}}
generate_target synthesis [get_ips]
set_property top {top} [current_fileset]
{step}
"""

ELABORATE = (
    'synth_design -top {top} -part {part} -rtl -name rtl_elab\nputs "ELABORATION OK"'
)

# Synthesis, then place and route. Each stage prints its own elapsed seconds,
# because "the build took an hour" is not a measurement -- the question is which
# stage grows with the array and which does not.
IMPLEMENT = """add_files -fileset constrs_1 {root}/clock.xdc
proc stage {{name body}} {{
  set t0 [clock milliseconds]
  uplevel 1 $body
  puts "SPMW STAGE $name [expr {{([clock milliseconds] - $t0) / 1000.0}}]"
}}
stage synth  {{ synth_design -top {top} -part {part} }}
report_utilization -file util_synth.rpt
stage opt    {{ opt_design }}
stage place  {{ place_design }}
stage physopt {{ phys_opt_design }}
stage route  {{ route_design }}
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
report_route_status -file route.rpt
write_checkpoint -force routed.dcp
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "ARRAY WNS $wns"
set unrouted [llength [get_nets -quiet -filter {{ROUTE_STATUS != ROUTED && ROUTE_STATUS != INTRASITE}} -of [get_nets -quiet -hierarchical]]]
puts "SPMW UNROUTED $unrouted"
puts "IMPLEMENTATION OK"
"""

# Real synthesis of the whole fabric. The per-unit HLS estimate is not the
# array's clock: the fabric adds the FIFOs, the fanout of a shared boundary
# stream, and the routing between instances, none of which HLS ever saw.
SYNTHESISE = """add_files -fileset constrs_1 {root}/clock.xdc
synth_design -top {top} -part {part}
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "ARRAY WNS $wns"
puts "SYNTHESIS OK"
"""


def design(name, size):
    """One of the design doc's worked examples, by name.

    They come from the test fixtures rather than being restated here, so the
    thing built is the same thing the suite checks.
    """
    # pylint: disable=import-outside-toplevel
    if name == "gemm":
        from test_spmw_rolled import gemm_of

        return gemm_of(size)
    if name == "gemm8":
        # The same mesh in integers. Its float twin cannot reach II=1 because a
        # float add sits in a distance-1 recurrence; an integer add is
        # single-cycle, so this one can.
        from test_spmw_gemm_int8 import gemm_int8_of

        return gemm_int8_of(size)
    if name == "daisy":
        # The chained-drain mesh -- the structural match for AutoSA, whose
        # C_drain_IO network this replaces with a link.
        from test_spmw_daisy import daisy_of

        return daisy_of(size)
    if name == "autosa":
        # Structurally matched to what AutoSA emits: chained A/B distribution,
        # chained C drain, int8 into int32.
        from test_spmw_autosa_match import autosa_match_of

        return autosa_match_of(size)
    if name == "tpuvpu":
        # The TPU with a programmable vector unit: MXU, a VPU chain with a
        # register file, and a program streamed in as data.
        from test_spmw_tpu_vpu import tpu

        return tpu
    if name == "tpuisa":
        # Both units instruction-driven: the MXU's opcode selects the weight
        # matrix, so one load serves several matmuls.
        from test_spmw_tpu_isa import ONE_PASS

        return ONE_PASS
    if name == "transformer":
        # The instruction-driven engine a Transformer block is written for.
        # `scripts/spmw_transformer_rtl.py` replays the whole block on it.
        from test_spmw_transformer import engine

        return engine
    if name == "transformer16":
        # The same engine on a 16x16 array -- 272 instances, the same 12 roles.
        from test_spmw_transformer import BIG

        return BIG.engine
    if name == "tputiled":
        # The same engine reducing deeper than the array: NTILE weight tiles
        # accumulated by the lane's own ACCZ instructions.
        from test_spmw_tpu_vpu import tpu_tiled

        return tpu_tiled
    if name == "autosa-spec":
        # The fused design with the PE's row specialised -- the control that
        # says whether splitting the drain out was needed at all.
        from test_spmw_autosa_match import autosa_match_of

        return autosa_match_of(size, specialise=True)
    if name == "split":
        # The same, with the drain lifted out of the PE into its own placement,
        # as AutoSA's `C_drain_IO_L1_out` is. The PE becomes a pure MAC.
        from test_spmw_split_drain import split_drain_of

        return split_drain_of(size)
    if name == "split-spec":
        # And with the drain's row specialised, so its forwarding loop has a
        # compile-time trip count -- one role per row, synthesised concurrently.
        from test_spmw_split_drain import split_drain_of

        return split_drain_of(size, specialise=True)
    if name == "tpu":
        from test_spmw_tpu import tpu_matmul

        return tpu_matmul
    if name == "fft":
        from test_spmw_fft import fft_spatial

        return fft_spatial
    if name == "attention":
        from test_spmw_attention import attention_pv

        return attention_pv(2)
    if name == "tiled":
        from test_spmw_tiled import tiled_gemm

        return tiled_gemm
    raise SystemExit(f"unknown design {name!r}")


def operands(fabric, graph):
    """Inputs to drive the array with, and the outputs the design says to expect.

    Which tensors are outputs is read off the graph rather than assumed from
    their names. Inputs are small integers, exactly representable in every type
    here, so the RTL comparison is bit-exact rather than a tolerance.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    from allo.spmw.lower_df import Lowering  # pylint: disable=import-outside-toplevel

    written = Lowering(graph).written_tensors()
    rng = np.random.default_rng(0)
    arrays = {}
    for tensor in graph.tensors.values():
        kind = np.dtype(_NUMPY[str(tensor.dtype)])
        if tensor.name in written:
            arrays[tensor.name] = np.zeros(tensor.shape, dtype=kind)
        else:
            arrays[tensor.name] = rng.integers(0, 3, size=tensor.shape).astype(kind)
    # Some inputs are not data. A tensor holding a *program* has to be a legal
    # program, and small random integers decode to opcodes the design does not
    # have -- so a fabric may say what a given tensor must contain.
    for name, value in getattr(fabric, "spmw_operands", {}).items():
        if name not in arrays:
            raise SystemExit(
                f"`spmw_operands` names {name!r}, which is not one of this "
                f"design's tensors: {sorted(arrays)}"
            )
        arrays[name][...] = value
    spmw.build(fabric, target="ref")(*[arrays[t.name] for t in graph.tensors.values()])
    return arrays


_NUMPY = {
    "f32": "float32",
    "i16": "int16",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}


def stage(graph, out, part, frequency, ii=None):
    """Write one HLS project per role, plus the fabric that will hold them.

    Every placement, not just the first: a design can put more than one
    component down -- the mini-TPU is an MXU and an activation row joined by
    ``link`` -- and each contributes its own roles.
    """
    emitter = UnitEmitter(graph)
    names = []
    for placement in emitter.placements():
        for order in range(len(emitter.classes(placement))):
            name = emitter.role_name(placement, order)
            if ii is not None:
                spmw.pipeline(placement, ii=ii)
            built = build_unit(graph, placement, order, target="vhls")
            code, _bound = optimise(trim_includes(str(built.hls_code)), built)
            directory = os.path.join(out, name)
            os.makedirs(directory, exist_ok=True)
            _write(os.path.join(directory, "kernel.cpp"), code)
            _write(
                os.path.join(directory, f"{name}.sv"),
                wrapper_sv(graph, placement, order, code),
            )
            _write(
                os.path.join(directory, "run.tcl"),
                ROLE_TCL.format(name=name, part=part, period=1000.0 / frequency),
            )
            names.append(name)
    _write(os.path.join(out, "spmw_fifo.sv"), rtl.fifo_module())
    _write(os.path.join(out, "spmw_const.sv"), rtl.const_module())
    _write(os.path.join(out, "spmw_top.sv"), rtl.StructuralEmitter(graph).fabric())
    return names


def stage_movers(graph, out, part, frequency, widen=512):
    """Write one HLS project per *mover* -- the array's memory interface.

    A mover is not a role: it walks a whole tensor and stops, so it keeps
    ``ap_ctrl_hs`` and its tensor argument becomes an AXI master rather than a
    stream.  There is one per binding whatever the grid, which is why giving the
    fabric DRAM access costs a fixed number of extra synthesis runs.

    The wrapper is written twice.  Once here, from the element width, so the
    fabric elaborates before Vitis has run; and again after ``csynth``, from the
    width Vitis actually achieved -- widening is a ceiling, not a promise, and a
    16-byte operand cannot reach 512 bits however wide the request.
    """
    emitter = MoverEmitter(graph)
    names = []
    for index in range(len(emitter.movers())):
        name = emitter.name(index)
        built = build_mover(graph, index, target="vhls")
        code = trim_includes(str(built.hls_code))
        directory = os.path.join(out, name)
        os.makedirs(directory, exist_ok=True)
        _write(os.path.join(directory, "kernel.cpp"), code)
        _write(
            os.path.join(directory, f"{name}.sv"), mover_wrapper_sv(graph, index, code)
        )
        tensor = emitter.movers()[index].tensor
        _write(
            os.path.join(directory, "run.tcl"),
            MOVER_TCL.format(
                name=name,
                part=part,
                period=1000.0 / frequency,
                widen=widen,
                depth=_volume(tensor.base.shape),
                arg=_tensor_param(code, name),
            ),
        )
        names.append(name)
    return names


def _tensor_param(code, name):
    """Which generated parameter the ``m_axi`` directive has to name."""
    # pylint: disable=import-outside-toplevel
    from allo.spmw.role_ip import mover_interface

    return mover_interface(code, name)[0]


def _volume(shape):
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def rewrite_mover_wrappers(graph, out, names):
    """Re-emit each mover's wrapper at the AXI width Vitis actually built.

    Returns ``{mover index: data width}``, which the fabric needs so its own
    ports match the IPs it instantiates.
    """
    emitter = MoverEmitter(graph)
    widths = {}
    for index in range(len(emitter.movers())):
        name = emitter.name(index)
        if name not in names:
            continue
        directory = os.path.join(out, name)
        exported = os.path.join(
            directory, "prj", "sol", "syn", "verilog", f"{name}_0.v"
        )
        with open(exported, encoding="utf-8") as handle:
            netlist = handle.read()
        widths[index] = axi_data_width(netlist)
        with open(os.path.join(directory, "kernel.cpp"), encoding="utf-8") as handle:
            code = handle.read()
        wrapper = mover_wrapper_sv(graph, index, code, data_width=widths[index])
        _write(os.path.join(directory, f"{name}.sv"), wrapper)
        check_wrapper(wrapper, netlist)
    return widths


def tune(graph, out, part, frequency, candidates=(0, 2, 3, 4, 5, 6)):
    """Find the initiation interval that runs fastest and still closes timing.

    A unit with a loop-carried accumulator cannot reach II=1: the recurrence runs
    through a float add, so II = adder latency + 1. Asking for a shorter adder
    buys interval and spends combinational delay, and which trade wins is a
    property of the design *and* the clock -- so it is measured rather than
    guessed. One representative role is enough; the roles of a placement differ
    in wiring, not in arithmetic.

    Returns the best interval, and the table it was chosen from.
    """
    target = 1000.0 / frequency
    work = os.path.join(out, "_tune")
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    order = min(2, len(emitter.classes(placement)) - 1)
    name = emitter.role_name(placement, order)

    jobs = []
    for ii in candidates:
        spmw.pipeline(placement, ii=ii)
        built = build_unit(graph, placement, order, target="vhls")
        code, _bound = optimise(trim_includes(str(built.hls_code)), built)
        directory = os.path.join(work, f"ii{ii}")
        os.makedirs(directory, exist_ok=True)
        _write(os.path.join(directory, "kernel.cpp"), code)
        _write(
            os.path.join(directory, "run.tcl"),
            ROLE_TCL.format(name=name, part=part, period=target).replace(
                "export_design -format ip_catalog\n", ""
            ),
        )
        jobs.append((ii, directory))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(
            pool.map(
                lambda j: _run(["vitis_hls", "-f", "run.tcl"], j[1], check=False), jobs
            )
        )

    table = []
    for ii, directory in jobs:
        measured = _measure(directory, name)
        if measured is None:
            continue
        achieved, period = measured
        closes = period <= target
        table.append((ii, achieved, period, achieved * max(period, target), closes))
    if not table:
        raise SystemExit(f"no tuning run produced a report; see {work}")
    closing = [row for row in table if row[4]] or table
    best = min(closing, key=lambda row: row[3])
    print(f"  {'ask':>4} {'II':>4} {'period':>8} {'ns/iter':>8}  closes timing")
    for ii, achieved, period, cost, closes in table:
        mark = " <-- chosen" if ii == best[0] else ""
        print(
            f"  {ii:>4} {achieved:>4} {period:>8.3f} {cost:>8.1f}  "
            f"{'yes' if closes else 'NO':>3}{mark}"
        )
    return best[0], table


def _measure(directory, name):
    """The II and estimated period a tuning run reported."""
    import re  # pylint: disable=import-outside-toplevel

    reports = os.path.join(directory, "prj", "sol", "syn", "report")
    if not os.path.isdir(reports):
        return None
    achieved = period = None
    for filename in os.listdir(reports):
        path = os.path.join(reports, filename)
        text = open(
            path, encoding="utf-8"
        ).read()  # pylint: disable=consider-using-with
        if "Pipeline" in filename:
            match = re.search(r"\|-\s+\S+\s*\|[^|]*\|[^|]*\|[^|]*\|\s*(\d+)\|", text)
            if match:
                achieved = int(match.group(1))
        if filename == f"{name}_0_csynth.rpt":
            match = re.search(r"\|ap_clk\s*\|\s*[\d.]+ ns\|\s*([\d.]+) ns", text)
            if match:
                period = float(match.group(1))
    if achieved is None or period is None:
        return None
    return achieved, period


def synthesise(out, names, jobs=None):
    """One `csynth` + `export_design` per role -- the whole HLS cost.

    The roles are independent by construction: that is what having roles *means*.
    So they run concurrently, and the wall clock is one role rather than nine.
    Each ``vitis_hls`` is effectively single-threaded (25s of CPU in 31s elapsed)
    and holds ~300MB, so the useful limit is cores, not memory.
    """
    # One vitis_hls per role, but not more than the machine can carry: each is
    # effectively single-threaded and holds ~300MB, so cores are the limit. A
    # design with fifty roles should not launch fifty tools at once.
    jobs = jobs or max(1, min(len(names), (os.cpu_count() or 4) - 2))
    times = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_synthesise_one, out, name): name for name in names}
        for future in concurrent.futures.as_completed(futures):
            name, seconds, code = future.result()
            times[name] = seconds
            print(f"  {name}: rc={code} {seconds}s", flush=True)
            if code != 0:
                raise SystemExit(
                    f"{name} failed to synthesise; see {os.path.join(out, name)}/prj"
                )
    return times


def _synthesise_one(out, name):
    """Synthesise and export one role, then check its wrapper against the IP."""
    directory = os.path.join(out, name)
    start = time.time()
    done = subprocess.run(
        ["vitis_hls", "-f", "run.tcl"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = round(time.time() - start, 1)
    if done.returncode == 0:
        exported = os.path.join(
            directory, "prj", "sol", "syn", "verilog", f"{name}_0.v"
        )
        if os.path.exists(exported):
            with open(exported, encoding="utf-8") as handle:
                netlist = handle.read()
            with open(
                os.path.join(directory, f"{name}.sv"), encoding="utf-8"
            ) as handle:
                check_wrapper(handle.read(), netlist)
    return name, seconds, done.returncode


def assemble(out, part, names, top="spmw_top", frequency=None, pnr=False):
    """Vivado reads the exported IPs and builds the array.

    With ``frequency`` it runs real synthesis and reports the array's own clock
    and area; without it, elaboration only. The distinction matters: a unit's
    HLS period says nothing about the assembled fabric, which adds FIFOs,
    fanout and routing that HLS never saw.
    """
    period = 1000.0 / frequency if frequency else None
    if period:
        # The clock has to be a constraint *file*: `create_clock` needs an open
        # design, so issuing it before synth_design fails, and issuing it after
        # means synthesis ran with no timing target at all.
        _write(
            os.path.join(out, "clock.xdc"),
            f"create_clock -period {period:.3f} -name ap_clk [get_ports ap_clk]\n",
        )
        step = (IMPLEMENT if pnr else SYNTHESISE).format(top=top, part=part, root=out)
    else:
        step = ELABORATE.format(top=top, part=part)
    script = os.path.join(out, "assemble.tcl")
    _write(
        script,
        ASSEMBLE_TCL.format(
            part=part, root=out, top=top, roles=" ".join(names), step=step
        ),
    )
    start = time.time()
    done = subprocess.run(
        ["vivado", "-mode", "batch", "-source", script, "-nojournal", "-nolog"],
        cwd=out,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.time() - start, 1)
    if pnr:
        marker = "IMPLEMENTATION OK"
    elif period:
        marker = "SYNTHESIS OK"
    else:
        marker = "ELABORATION OK"
    if marker not in done.stdout:
        tail = "\n".join(done.stdout.splitlines()[-25:])
        raise SystemExit(f"Vivado did not finish the array:\n{tail}")
    wns = None
    for line in done.stdout.splitlines():
        if line.startswith("ARRAY WNS "):
            try:
                wns = float(line.split()[-1])
            except ValueError:
                wns = None
        # Each implementation stage times itself, so the cost can be attributed
        # rather than reported as one number.
        elif line.startswith("SPMW STAGE ") or line.startswith("SPMW UNROUTED "):
            print("  " + line.replace("SPMW ", "").lower())
    return elapsed, wns


IPGEN_TCL = """create_project -force ipgen {out}/ipgen -part {part}
foreach r {{{roles}}} {{
  foreach x [glob -nocomplain \
      {root}/$r/prj/sol/impl/ip/tmp.srcs/sources_1/ip/*/*.xci] {{
    import_ip $x
  }}
}}
generate_target {{simulation}} [get_ips]
"""


def cosim(graph, out, part, arrays, names):
    """Simulate the assembled array and compare against the reference.

    Elaborating is not computing, so this is the check that the mixed path is
    right rather than merely well-formed.
    """
    _write(os.path.join(out, "tb.sv"), render_testbench(graph, arrays, arrays))

    # The exported IPs instantiate Xilinx FP cores; xsim needs their generated
    # simulation models, which only Vivado can produce from the .xci files.
    _write(
        os.path.join(out, "genip.tcl"),
        IPGEN_TCL.format(out=out, part=part, root=out, roles=" ".join(names)),
    )
    _run(
        ["vivado", "-mode", "batch", "-source", "genip.tcl", "-nojournal", "-nolog"],
        out,
    )

    sim = os.path.join(out, "sim")
    os.makedirs(sim, exist_ok=True)
    sources = ["spmw_fifo.sv", "spmw_const.sv", "spmw_top.sv", "tb.sv"]
    for name in sources:
        shutil.copy(os.path.join(out, name), sim)
    # Each role contributes its wrapper -- which lives in the role's own project
    # directory, not at the top -- and its exported netlist.
    for role in names:
        wrapper = os.path.join(out, role, f"{role}.sv")
        if not os.path.isfile(wrapper):
            raise SystemExit(f"{role} has no wrapper at {wrapper}")
        shutil.copy(wrapper, sim)
        sources.append(f"{role}.sv")
        verilog = os.path.join(out, role, "prj", "sol", "syn", "verilog")
        if os.path.isdir(verilog):
            for name in os.listdir(verilog):
                if name.endswith(".v"):
                    shutil.copy(os.path.join(verilog, name), sim)
    for root, _dirs, files in os.walk(os.path.join(out, "ipgen")):
        if "sources_1" in root and os.sep + "ip" + os.sep in root + os.sep:
            for name in files:
                if name.endswith(".v"):
                    shutil.copy(os.path.join(root, name), sim)

    _run(["xvlog", "-sv"] + sources, sim)
    _run(["xvlog"] + [f for f in os.listdir(sim) if f.endswith(".v")], sim)
    _run(
        [
            "xelab",
            "tb",
            "-s",
            "tbsim",
            "-L",
            "floating_point_v7_1_16",
            "-L",
            "unisims_ver",
            "-L",
            "unimacro_ver",
            "-L",
            "secureip",
        ],
        sim,
    )
    done = _run(["xsim", "tbsim", "-runall"], sim, check=False)
    for line in done.stdout.splitlines():
        if "SPMW COSIM" in line or "SPMW CYCLES" in line or "MISMATCH" in line:
            print("  " + line.strip())
    if "SPMW COSIM PASS" not in done.stdout:
        raise SystemExit(f"cosim did not pass; see {sim}/xsim.log")


def _run(command, cwd, check=True):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if check and done.returncode != 0:
        raise SystemExit(
            f"{command[0]} failed in {cwd}:\n"
            + "\n".join((done.stdout + done.stderr).splitlines()[-15:])
        )
    return done


def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        default="gemm",
        choices=(
            "gemm",
            "gemm8",
            "daisy",
            "autosa",
            "autosa-spec",
            "split",
            "split-spec",
            "tpu",
            "tpuvpu",
            "tputiled",
            "tpuisa",
            "transformer",
            "transformer16",
            "fft",
            "attention",
            "tiled",
        ),
        help="which worked example to build",
    )
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument(
        "--ii",
        type=int,
        default=None,
        help="target initiation interval for the units' loops (0 leaves it alone)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="measure the best initiation interval before building",
    )
    parser.add_argument(
        "--synth",
        action="store_true",
        help="synthesise the whole array, for its own clock and area",
    )
    parser.add_argument(
        "--pnr",
        action="store_true",
        help="take the array all the way through place and route, timing each "
        "stage; implies --synth",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--part", default=PART)
    parser.add_argument("--frequency", type=float, default=300.0)
    parser.add_argument(
        "--stage-only", action="store_true", help="write the projects, run nothing"
    )
    parser.add_argument(
        "--cosim",
        action="store_true",
        help="simulate the assembled array against the reference",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="give the fabric a DRAM interface: build each binding's mover as "
        "an AXI master and drive the edge streams from inside",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="concurrent role syntheses (default: one per core, capped by roles)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fabric = design(args.design, args.size)
    graph = spmw.elaborate(fabric)
    cost = rtl.cost(graph)
    rtl.check_netlist(graph)
    print(f"{args.design}: {json.dumps(cost)}")

    ii = args.ii
    if args.tune:
        print("tuning the initiation interval:")
        ii, _table = tune(graph, args.out, args.part, args.frequency)
    start = time.time()
    names = stage(graph, args.out, args.part, args.frequency, ii=ii)
    print(
        f"staged {len(names)} role project(s) in {time.time() - start:.1f}s "
        f"(frontend + per-role HLS codegen)"
    )
    movers = []
    if args.memory:
        start = time.time()
        movers = stage_movers(graph, args.out, args.part, args.frequency)
        emitter = MoverEmitter(graph)
        masters = sum(emitter.instances(i) for i in range(len(movers)))
        print(
            f"staged {len(movers)} mover project(s) in {time.time() - start:.1f}s "
            f"-- {masters} AXI master(s) at this size"
        )
    if args.stage_only:
        return

    print(
        f"synthesising and exporting {len(names) + len(movers)} unit(s) concurrently:"
    )
    start = time.time()
    times = synthesise(args.out, names + movers, jobs=args.jobs or None)
    wall = round(time.time() - start, 1)
    print(
        f"HLS: {wall}s wall for {len(names)} roles and {len(movers)} movers "
        f"({round(sum(times.values()), 1)}s of CPU work, "
        f"{cost['instances']} instances)"
    )
    if args.memory:
        # The AXI width is only known now, and the fabric's own ports have to
        # match the IPs it holds -- so both are re-emitted against the truth.
        widths = rewrite_mover_wrappers(graph, args.out, movers)
        _write(
            os.path.join(args.out, "spmw_top.sv"),
            rtl.StructuralEmitter(graph).fabric(memory=True, axi_widths=widths),
        )
        print(
            "  AXI data widths: "
            + ", ".join(f"{movers[i]}={w}b" for i, w in sorted(widths.items()))
        )

    print("assembling the array in Vivado:")
    elapsed, wns = assemble(
        args.out,
        args.part,
        names + movers,
        frequency=args.frequency if (args.synth or args.pnr) else None,
        pnr=args.pnr,
    )
    verb = (
        "implemented" if args.pnr else ("synthesised" if args.synth else "elaborated")
    )
    print(f"Vivado {verb} {cost['instances']} instances in {elapsed}s")
    if wns is not None:
        period = 1000.0 / args.frequency
        print(
            f"  array clock: {period - wns:.3f} ns achieved against a "
            f"{period:.3f} ns target (WNS {wns:+.3f} ns)"
        )

    if args.cosim:
        print("simulating the assembled array:")
        cosim(graph, args.out, args.part, operands(fabric, graph), names)
    _write(
        os.path.join(args.out, "cost.json"),
        json.dumps(
            {
                "cost": cost,
                "hls_seconds": times,
                "hls_wall_seconds": wall,
                "vivado_seconds": elapsed,
            },
            indent=1,
        ),
    )


if __name__ == "__main__":
    main()

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a spatial array the way SPMW is meant to be built.

HLS synthesises the *unit* -- one per role -- and Vivado assembles the array
from the exported IPs.  The C synthesis and IP export are paid once per role, so
the cost tracks the role count rather than the grid: a 2-D mesh has nine roles
whatever its size.

Run it on a machine with Vitis and Vivado on PATH::

    python3 scripts/spmw_build_array.py --size 4 --out /scratch/$USER/spmw_array

It stages one HLS project per role, synthesises and exports each, then writes a
Vivado script that reads the exported IPs together with the structural fabric
and elaborates the whole array.  Measured on brg-zhang-xcel for a 4x4 GEMM: nine
exports in 544s, and Vivado elaborates the 16-instance array in 17s.

For contrast, the whole-array program the dataflow path emits does not
synthesise at all -- at any grid size.  Every site stores into the same result
tensor and HLS dataflow permits one writer per array, so ``csynth`` rejects it
with "failed dataflow checking".  Streaming each site's result out, which is what
the unit does, is what makes the design synthesisable rather than merely
cheaper.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw import rtl  # pylint: disable=wrong-import-position
from allo.spmw.role_ip import (  # pylint: disable=wrong-import-position
    UnitEmitter,
    build_unit,
    check_wrapper,
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

ASSEMBLE_TCL = """create_project -in_memory -part {part}
set root {root}
add_files [glob $root/*.sv]
foreach d [glob $root/pe_r*] {{
  add_files [glob -nocomplain $d/*.sv]
  add_files [glob -nocomplain $d/prj/sol/syn/verilog/*.v]
  foreach x [glob -nocomplain $d/prj/sol/impl/ip/tmp.srcs/sources_1/ip/*/*.xci] {{
    read_ip $x
  }}
}}
generate_target synthesis [get_ips]
set_property top {top} [current_fileset]
synth_design -top {top} -part {part} -rtl -name rtl_elab
puts "ELABORATION OK"
"""


def gemm(size):
    """The design under test: the design doc's systolic GEMM, at a chosen size."""
    from test_spmw_rolled import gemm_of  # pylint: disable=import-outside-toplevel

    return gemm_of(size)


def stage(graph, out, part, frequency):
    """Write one HLS project per role, plus the fabric that will hold them."""
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    names = []
    for order in range(len(emitter.classes(placement))):
        name = emitter.role_name(placement, order)
        code = str(build_unit(graph, placement, order, target="vhls").hls_code)
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
    _write(os.path.join(out, "spmw_top.sv"), rtl.StructuralEmitter(graph).fabric())
    return names


def synthesise(out, names):
    """One `csynth` + `export_design` per role -- the whole HLS cost."""
    times = {}
    for name in names:
        directory = os.path.join(out, name)
        start = time.time()
        done = subprocess.run(
            ["vitis_hls", "-f", "run.tcl"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        times[name] = round(time.time() - start, 1)
        print(f"  {name}: rc={done.returncode} {times[name]}s", flush=True)
        if done.returncode != 0:
            raise SystemExit(f"{name} failed to synthesise; see {directory}/prj")
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
    return times


def assemble(out, part, top="spmw_top"):
    """Vivado reads the exported IPs and elaborates the array."""
    script = os.path.join(out, "assemble.tcl")
    _write(script, ASSEMBLE_TCL.format(part=part, root=out, top=top))
    start = time.time()
    done = subprocess.run(
        ["vivado", "-mode", "batch", "-source", script, "-nojournal", "-nolog"],
        cwd=out,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.time() - start, 1)
    if "ELABORATION OK" not in done.stdout:
        tail = "\n".join(done.stdout.splitlines()[-25:])
        raise SystemExit(f"Vivado did not elaborate the array:\n{tail}")
    return elapsed


def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--part", default=PART)
    parser.add_argument("--frequency", type=float, default=300.0)
    parser.add_argument(
        "--stage-only", action="store_true", help="write the projects, run nothing"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    graph = spmw.elaborate(gemm(args.size))
    cost = rtl.cost(graph)
    rtl.check_netlist(graph)
    print(f"{args.size}x{args.size} GEMM: {json.dumps(cost)}")

    names = stage(graph, args.out, args.part, args.frequency)
    print(f"staged {len(names)} role project(s)")
    if args.stage_only:
        return

    print("synthesising and exporting each role:")
    times = synthesise(args.out, names)
    total = round(sum(times.values()), 1)
    print(
        f"HLS total: {total}s for {len(names)} roles "
        f"({cost['instances']} instances)"
    )

    print("assembling the array in Vivado:")
    elapsed = assemble(args.out, args.part)
    print(f"Vivado elaborated {cost['instances']} instances in {elapsed}s")
    _write(
        os.path.join(args.out, "cost.json"),
        json.dumps(
            {"cost": cost, "hls_seconds": times, "vivado_seconds": elapsed}, indent=1
        ),
    )


if __name__ == "__main__":
    main()

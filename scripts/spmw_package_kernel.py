# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Take the role-path array to an `.xclbin`.

`spmw_build_array.py --pnr` implements the fabric behind a synthetic harness,
which answers "what does this array cost and how fast is it" but does not put
it on a card. This script answers the other question: it wraps the same fabric
in the kernel `allo.spmw.kernel` describes, packages it as an RTL kernel, and
links it for the U280.

    python3 scripts/spmw_package_kernel.py --design transformer16 --size 16 \
        --out DIR --slots 4 --frequency 300

Stages, each timed, because which one grows with the array is the interesting
part:

    feeders    one HLS run per boundary family, concurrent
    fabric     the SystemVerilog the array is made of
    package    `package_xo` over the IP
    link       `v++ -l`, which is hours

``--sim`` stops before `v++` and runs the kernel against a behavioural AXI
slave instead, which is the cheap way to find out that the control map or the
edge FIFOs are wrong.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)
sys.path.insert(0, os.path.dirname(__file__))

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw import rtl, shell  # pylint: disable=wrong-import-position
from allo.spmw.kernel import (  # pylint: disable=wrong-import-position
    arguments,
    control_sv,
    feeder_tcl,
    kernel_sv,
    kernel_xml,
)
from allo.spmw.role_ip import axi_data_width  # pylint: disable=wrong-import-position
from allo.spmw.shell import _dma_name, families  # pylint: disable=wrong-import-position
from spmw_build_array import (  # pylint: disable=wrong-import-position
    PART,
    ROLE_TCL,
    design,
    stage,
    synthesise,
)

PLATFORM = "xilinx_u280_gen3x16_xdma_1_202211_1"

# Packaging as a Vivado IP first, rather than handing `package_xo` loose files,
# so the floorplan can ride along as a *scoped* constraint: the pblock names
# cells relative to the kernel instance, which is the only way to write it
# without knowing where in the shell's hierarchy the kernel will land.
PACKAGE_TCL = """create_project -in_memory -part {part}
add_files -norecurse [glob {root}/*.sv]
proc add_if_any {{pattern}} {{
  # `add_files` on an empty glob is an error, not a no-op.
  set found [glob -nocomplain $pattern]
  if {{[llength $found]}} {{ add_files -norecurse $found }}
}}
foreach base [list {root}/feeders {root}/roles] {{
  foreach d [glob -nocomplain -directory $base *] {{
    # The unit's own wrapper as well as the netlist Vitis exported: the
    # wrapper is what names the ports the fabric connects to.
    add_if_any $d/*.sv
    add_if_any $d/prj/sol/syn/verilog/*.v
  }}
}}
set_property top {top} [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir {root}/ip -vendor allo -library spmw \\
  -taxonomy /UserIP -import_files -force
set core [ipx::current_core]
set_property core_revision 1 $core
set_property supported_families {{virtexuplusHBM Production}} $core
# Every AXI interface has to be named as belonging to `ap_clk`, or the system
# linker gets an empty object list when it tries to connect them and stops with
# "Invalid option value '' specified for 'objects'" -- which says nothing about
# clocks at all.
foreach busif {{{busifs}}} {{
  ipx::associate_bus_interfaces -busif $busif -clock ap_clk $core
}}
set clk [ipx::get_bus_interfaces ap_clk -of_objects $core]
set p [ipx::add_bus_parameter ASSOCIATED_RESET $clk]
set_property value ap_rst_n $p
set rst [ipx::get_bus_interfaces ap_rst_n -of_objects $core]
set p [ipx::add_bus_parameter POLARITY $rst]
set_property value ACTIVE_LOW $p
{floorplan}
# Without these two, `package_xo` treats the IP as an ordinary one and
# re-derives its metadata, which silently drops ASSOCIATED_BUSIF -- and then
# the system linker cannot find a clock for the masters.
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
package_xo -force -xo_path {root}/{top}.xo -kernel_name {top} \\
  -ctrl_protocol ap_ctrl_hs -ip_directory {root}/ip -kernel_xml {root}/kernel.xml
exit
"""

# A pblock has to name cells relative to the kernel instance, because where in
# the shell's hierarchy `v++` puts the kernel is not knowable here. An IP's
# implementation file group with SCOPED_TO_REF is how that is said.
SCOPED_XDC = """file copy -force {xdc} {root}/ip/floorplan.xdc
set fg [ipx::add_file_group -type implementation {{}} $core]
set f [ipx::add_file floorplan.xdc $fg]
set_property type xdc $f
set_property used_in {{implementation}} $f
set_property SCOPED_TO_REF {top} $f
"""


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def stage_feeders(graph, out, part, frequency):
    """One HLS project per boundary family."""
    names = []
    for fam in families(graph):
        name = _dma_name(fam)
        directory = os.path.join(out, "feeders", name)
        _write(os.path.join(directory, "kernel.cpp"), shell.feeder_cpp(fam))
        _write(
            os.path.join(directory, "run.tcl"),
            feeder_tcl(fam, part, 1000.0 / frequency),
        )
        names.append(name)
    return names


def _run_hls(directory):
    start = time.time()
    done = subprocess.run(
        ["vitis_hls", "-f", "run.tcl"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    _write(os.path.join(directory, "hls.log"), done.stdout + done.stderr)
    return os.path.basename(directory), round(time.time() - start, 1), done.returncode


def synthesise_feeders(out, names, jobs=None):
    """Every feeder at once; they are independent and small."""
    times = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs or len(names)) as pool:
        futures = [
            pool.submit(_run_hls, os.path.join(out, "feeders", name)) for name in names
        ]
        for future in concurrent.futures.as_completed(futures):
            name, seconds, code = future.result()
            print(f"  {name}: rc={code} {seconds}s")
            if code:
                raise SystemExit(
                    f"HLS failed for {name}; see {out}/feeders/{name}/hls.log"
                )
            times[name] = seconds
    return times


def measured_widths(out, names):
    """The AXI width each feeder actually got, read back from its netlist."""
    widths = {}
    for name in names:
        path = os.path.join(
            out, "feeders", name, "prj", "sol", "syn", "verilog", f"{name}.v"
        )
        with open(path, encoding="utf-8") as handle:
            widths[name] = axi_data_width(handle.read())
    return widths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="transformer16")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--part", default=PART)
    parser.add_argument("--platform", default=PLATFORM)
    parser.add_argument("--frequency", type=float, default=300.0)
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--top", default="spmw_kernel")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--sim", action="store_true", help="stop before v++")
    parser.add_argument("--link-frequency", type=float, default=0.0)
    args = parser.parse_args()

    fabric = design(args.design, args.size)
    graph = spmw.elaborate(fabric)
    os.makedirs(args.out, exist_ok=True)

    anchors = {}
    floorplan = ""
    if args.slots:
        anchors = shell.crossing_families(graph, slots=args.slots)
        floorplan = shell.floorplan_xdc(
            graph, part=args.part, top="dut", slots=args.slots
        )
        print(f"floorplan: {args.slots} slot(s); anchors on {sorted(anchors)}")

    print("staging roles and feeders:")
    start = time.time()
    roles = stage(
        graph,
        os.path.join(args.out, "roles"),
        args.part,
        args.frequency,
        anchors=anchors,
    )
    feeders = stage_feeders(graph, args.out, args.part, args.frequency)
    # `stage` writes the fabric under roles/; the kernel wants it beside itself.
    for name in ("spmw_fifo.sv", "spmw_const.sv", "spmw_top.sv"):
        src = os.path.join(args.out, "roles", name)
        with open(src, encoding="utf-8") as handle:
            _write(os.path.join(args.out, name), handle.read())
    print(
        f"  staged {len(roles)} role(s) and {len(feeders)} feeder(s) "
        f"in {time.time() - start:.1f}s"
    )

    print(f"synthesising {len(roles) + len(feeders)} unit(s):")
    start = time.time()
    synthesise(os.path.join(args.out, "roles"), roles, jobs=args.jobs or None)
    synthesise_feeders(args.out, feeders, jobs=args.jobs or None)
    print(f"  HLS: {time.time() - start:.1f}s")

    widths = measured_widths(args.out, feeders)
    print("  AXI widths: " + ", ".join(f"{k}={v}b" for k, v in sorted(widths.items())))

    kargs = arguments(graph)
    _write(os.path.join(args.out, "spmw_control_s_axi.sv"), control_sv(kargs))
    _write(
        os.path.join(args.out, f"{args.top}.sv"),
        kernel_sv(graph, kargs, widths, top=args.top),
    )
    _write(
        os.path.join(args.out, "kernel.xml"), kernel_xml(kargs, widths, name=args.top)
    )
    _write(
        os.path.join(args.out, "args.json"),
        json.dumps(
            [
                {
                    "name": a.name,
                    "bits": a.bits,
                    "offset": a.offset,
                    "pointer": a.pointer,
                    "family": a.family["name"],
                    "channels": a.family["channels"],
                    "steps": a.family["steps"],
                    "reads": a.family["reads"],
                    "width": a.family["width"],
                    "tensor": a.family["tensor"],
                }
                for a in kargs
            ],
            indent=2,
        ),
    )
    print(
        f"  kernel: {len(kargs)} argument(s), "
        f"{sum(1 for a in kargs if a.pointer)} AXI master(s)"
    )

    if args.sim:
        print("stopping before v++ (--sim)")
        return

    if floorplan:
        _write(os.path.join(args.out, "floorplan.xdc"), floorplan)
    # Inside the IP, not beside it: a file referenced as `../floorplan.xdc`
    # resolves outside the packaged core and is simply not shipped in the .xo,
    # which surfaces four steps later as "failed to deliver one or more files".
    scoped = (
        SCOPED_XDC.format(
            xdc=os.path.join(args.out, "floorplan.xdc"), root=args.out, top=args.top
        )
        if floorplan
        else ""
    )
    _write(
        os.path.join(args.out, "package.tcl"),
        PACKAGE_TCL.format(
            part=args.part,
            root=args.out,
            top=args.top,
            floorplan=scoped,
            busifs=" ".join(
                [f"m_axi_gmem{i}" for i in range(sum(1 for a in kargs if a.pointer))]
                + ["s_axi_control"]
            ),
        ),
    )
    print("packaging the kernel:")
    start = time.time()
    done = subprocess.run(
        ["vivado", "-mode", "batch", "-source", "package.tcl", "-nojournal", "-nolog"],
        cwd=args.out,
        capture_output=True,
        text=True,
        check=False,
    )
    _write(os.path.join(args.out, "package.log"), done.stdout + done.stderr)
    xo = os.path.join(args.out, f"{args.top}.xo")
    if not os.path.exists(xo):
        tail = "\n".join(done.stdout.splitlines()[-30:])
        raise SystemExit(f"package_xo produced no .xo:\n{tail}")
    print(f"  SPMW STAGE package {time.time() - start:.1f}")

    link = args.link_frequency or args.frequency
    print(f"linking for {args.platform} at {link:.0f} MHz (hours):")
    start = time.time()
    done = subprocess.run(
        [
            "v++",
            "-l",
            "-t",
            "hw",
            "--platform",
            args.platform,
            "--kernel_frequency",
            str(int(link)),
            "-o",
            os.path.join(args.out, f"{args.top}.xclbin"),
            xo,
            "--temp_dir",
            os.path.join(args.out, "vpp"),
            "--report_level",
            "2",
            "--save-temps",
        ],
        cwd=args.out,
        capture_output=True,
        text=True,
        check=False,
    )
    _write(os.path.join(args.out, "link.log"), done.stdout + done.stderr)
    print(f"  SPMW STAGE link {time.time() - start:.1f}")
    if done.returncode:
        raise SystemExit(f"v++ failed; see {args.out}/link.log")
    print(f"xclbin: {os.path.join(args.out, args.top + '.xclbin')}")


if __name__ == "__main__":
    main()

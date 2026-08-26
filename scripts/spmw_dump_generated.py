# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump every stage of the SPMW pipeline, for reading side by side.

The output lives in ``examples/spmw/generated`` and is checked in, so a change
to any emitter can be diffed rather than described.  See that directory's
README for what each file is.

    python3 scripts/spmw_dump_generated.py --out examples/spmw/generated

Only ``12_exported_*.v`` needs a toolchain -- it is what ``export_design``
actually produced -- and it is left alone when Vitis is absent.
"""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw import rtl  # pylint: disable=wrong-import-position
from allo.spmw.cosim import render_testbench  # pylint: disable=wrong-import-position
from allo.spmw.lower_df import render_source  # pylint: disable=wrong-import-position
from allo.spmw.lower_mlir import render_module  # pylint: disable=wrong-import-position
from allo.spmw.role_ip import (  # pylint: disable=wrong-import-position
    UnitEmitter,
    build_unit,
    trim_includes,
    wrapper_sv,
)

DESIGNS = ("gemm", "gemm8", "tiled", "fft", "tpu", "attention")


def dump(name, out, size):
    """Every stage of one design."""
    # pylint: disable=import-outside-toplevel
    from spmw_build_array import design, operands

    fabric = design(name, size)
    graph = spmw.elaborate(fabric)
    directory = os.path.join(out, name)
    os.makedirs(directory, exist_ok=True)

    def write(filename, text):
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as handle:
            handle.write(text)

    write("01_dataflow.py", render_source(graph))
    write("02_rolled.mlir", render_module(graph))
    module = spmw.build(fabric, target="vhls")
    write("03_array.cpp", str(module.hls_code))
    write("04_array.mlir", str(module.module))

    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    order = min(2, len(emitter.classes(placement)) - 1)
    role = emitter.role_name(placement, order)
    program, _extras = emitter.program(placement, order)
    write(f"05_unit_{role}.py", program)
    code = trim_includes(
        str(build_unit(graph, placement, order, target="vhls").hls_code)
    )
    write(f"06_unit_{role}.cpp", code)
    write(f"07_unit_{role}_wrapper.sv", wrapper_sv(graph, placement, order, code))

    write("08_spmw_fifo.sv", rtl.fifo_module())
    write("09_spmw_const.sv", rtl.const_module())
    write("10_spmw_top.sv", rtl.StructuralEmitter(graph).fabric())
    arrays = operands(fabric, graph)
    write("11_tb.sv", render_testbench(graph, arrays, arrays))
    return rtl.cost(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=3, help="grid size where it varies")
    parser.add_argument("--design", choices=DESIGNS, help="just one of them")
    args = parser.parse_args()
    for name in [args.design] if args.design else DESIGNS:
        print(f"{name}: {dump(name, args.out, args.size)}", flush=True)


if __name__ == "__main__":
    main()

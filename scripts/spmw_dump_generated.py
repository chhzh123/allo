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
from allo.spmw.dram import (  # pylint: disable=wrong-import-position
    ram_module,
    render_memory_testbench,
)
from allo.spmw.role_ip import (  # pylint: disable=wrong-import-position
    MoverEmitter,
    UnitEmitter,
    build_mover,
    build_unit,
    mover_wrapper_sv,
    trim_includes,
    wrapper_sv,
)

DESIGNS = (
    "gemm",
    "gemm8",
    "tiled",
    "fft",
    "tpu",
    "attention",
    # The programmable TPU: a matrix unit, a vector unit with its own ISA, and
    # a program that arrives as data.
    "tpuvpu",
    "tputiled",
    # Both units instruction-driven, and the engine a Transformer runs on.
    "tpuisa",
    "transformer",
    # The AutoSA comparison, and the two things learned from it.
    "autosa",
    "autosa-spec",
    "split",
)

# Designs whose bindings synthesise movers, so the memory-mapped fabric and its
# AXI masters can be dumped too. A `MemOut` gathered per site has no mover, so
# the plain meshes stop at their edge streams.
MEMORY = ("autosa", "autosa-spec", "split")

# Sizes that differ from the default. A 3x3 mesh has nine sites in nine wiring
# classes -- every one is an edge or a corner -- so specialising a coordinate
# cannot split anything and `autosa-spec` would be a byte-for-byte copy of
# `autosa`. From 4x4 there is an interior to split, and the role counts diverge:
# 15 against 18 at 4x4, 15 against 21 at 5x5.
SIZES = {"autosa": 4, "autosa-spec": 4, "split": 4}


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

    # One representative role from *every* placement, not just the first. A
    # design with more than one component -- the TPU's matrix unit and its
    # vector unit, the matched GEMM's mesh and its feed chains -- is mostly
    # interesting in the parts the first placement is not.
    emitter = UnitEmitter(graph)
    for placement in emitter.placements():
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

    if name in MEMORY:
        _dump_memory(name, graph, arrays, write)
    return rtl.cost(graph)


def _dump_memory(name, graph, arrays, write):
    """The DRAM side: each binding's mover as an IP, and the fabric holding them.

    This is what makes the design an accelerator rather than a core, and it is
    the part the AutoSA comparison turns on -- so it is checked in too.
    """
    movers = MoverEmitter(graph)
    for index in range(len(movers.movers())):
        mover = movers.name(index)
        write(f"13_mover_{mover}.py", movers.program(index)[0])
        code = trim_includes(str(build_mover(graph, index, target="vhls").hls_code))
        write(f"14_mover_{mover}.cpp", code)
        write(f"15_mover_{mover}_wrapper.sv", mover_wrapper_sv(graph, index, code))
    write("16_spmw_top_memory.sv", rtl.StructuralEmitter(graph).fabric(memory=True))
    write("17_spmw_axi_ram.sv", ram_module())
    write("18_tb_memory.sv", render_memory_testbench(graph, arrays, arrays))
    del name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=3, help="grid size where it varies")
    parser.add_argument("--design", choices=DESIGNS, help="just one of them")
    args = parser.parse_args()
    for name in [args.design] if args.design else DESIGNS:
        size = SIZES.get(name, args.size)
        print(f"{name} ({size}x{size}): {dump(name, args.out, size)}", flush=True)


if __name__ == "__main__":
    main()

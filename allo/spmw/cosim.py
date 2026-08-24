# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulating the structural fabric against the reference.

The array elaborating is not the array computing.  This drives ``spmw_top`` in
xsim over its edge streams and compares what comes out against the values the
design says it should produce, so the RTL path is checked the way every other
path is: numerically.

The stimulus order is not written by hand.  It comes from
:func:`allo.spmw.rtl.boundary_plan`, which reads it off the same ``binding.imap``
the reference simulator uses -- so if the testbench and the array disagree about
which channel carries row *i*, that is a real disagreement rather than a mistake
in the harness.
"""

import struct

from .errors import SPMWBindingError
from .ports import IN
from .rtl import StructuralEmitter, boundary_plan


def _bits(value, dtype):
    """One token as a Verilog literal, in the family's own bit pattern."""
    name = str(dtype)
    if name in ("f32", "float32"):
        return f"32'h{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"
    if name in ("f64", "float64"):
        return f"64'h{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"
    width = {"i8": 8, "i16": 16, "i32": 32, "i64": 64}.get(name)
    if width is None:
        raise SPMWBindingError(f"no cosim literal for `{name}`")
    return f"{width}'h{int(value) & ((1 << width) - 1):0{width // 4}x}"


class Testbench:
    """A self-checking testbench for one fabric and one set of inputs."""

    def __init__(self, graph, data):
        self.graph = graph
        self.data = data
        self.emitter = StructuralEmitter(graph)
        self.plan = boundary_plan(graph)
        self.families = {f.name: f for f in self.emitter.families()[1]}

    def _dtype(self, name):
        return self.families[name].dtype

    def _width(self, name):
        fam = self.families[name]
        bits = {"f64": 64, "f32": 32, "i64": 64, "i32": 32, "i16": 16, "i8": 8}[
            str(fam.dtype)
        ]
        for extent in fam.block:
            bits *= int(extent)
        return bits

    def stimulus(self):
        """Per inbound channel, the tokens it must be handed, in order."""
        out = {}
        for name, entry in self.plan.items():
            if entry["direction"] != IN:
                continue
            array = self.data[entry["tensor"]]
            out[name] = [
                [_bits(array[idx], self._dtype(name)) for idx in channel]
                for channel in entry["channels"]
            ]
        return out

    def expected(self, results):
        """Per outbound channel, the single token it should produce."""
        out = {}
        for name, entry in self.plan.items():
            if entry["direction"] == IN:
                continue
            array = results[entry["tensor"]]
            out[name] = [
                [_bits(array[idx], self._dtype(name)) for idx in channel]
                for channel in entry["channels"]
            ]
        return out

    def render(self, results, cycles=200000, top="spmw_top"):
        """The testbench module."""
        stim, want = self.stimulus(), self.expected(results)
        lines = [
            "`timescale 1ns/1ps",
            "",
            "module tb;",
            "  reg clk = 0, rst_n = 0;",
            "  always #5 clk = ~clk;",
            "  integer errors = 0;",
            "  integer produced = 0;",
            f"  localparam integer TOTAL = {sum(len(c) for c in want.values())};",
        ]
        conns = [".ap_clk(clk)", ".ap_rst_n(rst_n)"]
        lines += self._inbound(stim, conns)
        lines += self._outbound(want, conns)
        lines.append(f"  {top} dut (" + ", ".join(conns) + ");")
        lines += [
            "  initial begin",
            "    repeat (4) @(posedge clk);",
            "    rst_n = 1;",
            f"    for (integer c = 0; c < {cycles}; c = c + 1) begin",
            "      @(posedge clk);",
            "      if (produced == TOTAL) begin",
            '        $display("SPMW COSIM %s (%0d/%0d tokens, %0d errors)",',
            '                 errors == 0 ? "PASS" : "FAIL", produced, TOTAL, errors);',
            "        $finish;",
            "      end",
            "    end",
            '    $display("SPMW COSIM TIMEOUT (%0d/%0d tokens, %0d errors)",',
            "             produced, TOTAL, errors);",
            "    $finish;",
            "  end",
            "endmodule",
        ]
        return "\n".join(lines) + "\n"

    def _inbound(self, stim, conns):
        """Drivers: present each channel's tokens in order, advance on `read`."""
        lines = []
        for name, channels in stim.items():
            width = self._width(name)
            count = len(channels)
            lines += [
                f"  wire [{width - 1}:0] {name}_dout [0:{count - 1}];",
                f"  wire {name}_empty_n [0:{count - 1}];",
                f"  wire {name}_read [0:{count - 1}];",
            ]
            for k, tokens in enumerate(channels):
                depth = len(tokens)
                lines += [
                    f"  reg [{width - 1}:0] {name}_src{k} [0:{max(depth - 1, 0)}];",
                    f"  integer {name}_p{k} = 0;",
                    "  initial begin",
                    *[
                        f"    {name}_src{k}[{i}] = {tok};"
                        for i, tok in enumerate(tokens)
                    ],
                    "  end",
                    # Hold the last value once drained; empty_n gates it anyway.
                    f"  assign {name}_dout[{k}] = {name}_src{k}["
                    f"{name}_p{k} < {depth} ? {name}_p{k} : {max(depth - 1, 0)}];",
                    f"  assign {name}_empty_n[{k}] = ({name}_p{k} < {depth});",
                    f"  always @(posedge clk) if (rst_n && {name}_read[{k}] && "
                    f"{name}_empty_n[{k}]) {name}_p{k} <= {name}_p{k} + 1;",
                ]
            conns += [
                f".{name}_dout({name}_dout)",
                f".{name}_empty_n({name}_empty_n)",
                f".{name}_read({name}_read)",
            ]
        return lines

    def _outbound(self, want, conns):
        """Collectors: always ready, check each token as it is written."""
        lines = []
        for name, channels in want.items():
            width = self._width(name)
            count = len(channels)
            lines += [
                f"  wire [{width - 1}:0] {name}_din [0:{count - 1}];",
                f"  wire {name}_write [0:{count - 1}];",
                f"  wire {name}_full_n [0:{count - 1}];",
            ]
            for k, tokens in enumerate(channels):
                depth = len(tokens)
                lines += [
                    f"  reg [{width - 1}:0] {name}_exp{k} [0:{max(depth - 1, 0)}];",
                    f"  integer {name}_q{k} = 0;",
                    "  initial begin",
                    *[
                        f"    {name}_exp{k}[{i}] = {tok};"
                        for i, tok in enumerate(tokens)
                    ],
                    "  end",
                    f"  assign {name}_full_n[{k}] = 1'b1;",
                    f"  always @(posedge clk) if (rst_n && {name}_write[{k}]) begin",
                    f"    if ({name}_q{k} < {depth}) begin",
                    f"      if ({name}_din[{k}] !== {name}_exp{k}[{name}_q{k}]) begin",
                    "        errors = errors + 1;",
                    f'        $display("MISMATCH {name}[%0d] step %0d: got %h want %h",',
                    f"                 {k}, {name}_q{k}, {name}_din[{k}], "
                    f"{name}_exp{k}[{name}_q{k}]);",
                    "      end",
                    f"      {name}_q{k} <= {name}_q{k} + 1;",
                    "      produced = produced + 1;",
                    "    end else begin",
                    "      errors = errors + 1;",
                    f'      $display("EXTRA TOKEN on {name}[%0d]", {k});',
                    "    end",
                    "  end",
                ]
            conns += [
                f".{name}_din({name}_din)",
                f".{name}_write({name}_write)",
                f".{name}_full_n({name}_full_n)",
            ]
        return lines


def render_testbench(graph, data, results, cycles=200000, top="spmw_top"):
    """A self-checking testbench for ``graph`` driven by ``data``.

    ``data`` and ``results`` are ``{tensor_name: numpy array}`` -- the inputs to
    feed and the outputs to expect. Use the reference simulator to produce the
    latter, so the RTL is compared against the design's own semantics.
    """
    return Testbench(graph, data).render(results, cycles=cycles, top=top)


__all__ = ["Testbench", "render_testbench"]

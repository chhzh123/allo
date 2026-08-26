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


_SCALAR_BITS = {
    "f32": 32,
    "f64": 64,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
}


def _one(value, name):
    """One element's bit pattern as an integer."""
    if name == "f32":
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if name == "f64":
        return struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    width = _SCALAR_BITS[name]
    return int(value) & ((1 << width) - 1)


def _bits(value, dtype, block=()):
    """One token as a Verilog literal, in the family's own bit pattern.

    A block-carrying port sends a whole array per token, which HLS packs into one
    word as ``hls::vector``: element zero in the low bits. Packing it the same way
    here is an assumption the cosim then *tests* -- the hardware unpacks with its
    own convention, so a mismatch shows up as wrong results rather than silence.
    """
    name = str(dtype)
    if name not in _SCALAR_BITS:
        raise SPMWBindingError(f"no cosim literal for `{name}`")
    element = _SCALAR_BITS[name]
    if not block:
        return f"{element}'h{_one(value, name):0{element // 4}x}"
    import numpy as np  # pylint: disable=import-outside-toplevel

    flat = np.asarray(value).reshape(-1)
    total = element * flat.size
    packed = 0
    for position, item in enumerate(flat):
        packed |= _one(item, name) << (element * position)
    return f"{total}'h{packed:0{max(total // 4, 1)}x}"


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
        bits = _SCALAR_BITS[str(fam.dtype)]
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
            block = self.families[name].block
            out[name] = [
                [_bits(array[idx], self._dtype(name), block) for idx in channel]
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
            block = self.families[name].block
            out[name] = [
                [_bits(array[idx], self._dtype(name), block) for idx in channel]
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
            "  integer first = -1;",
            # Every token on every channel, not every channel: counting
            # channels made the run stop at the first token of each and call
            # that a pass, so attention checked 2 of its 12 and still said PASS.
            f"  localparam integer TOTAL = "
            f"{sum(len(t) for ch in want.values() for t in ch)};",
        ]
        conns = [".ap_clk(clk)", ".ap_rst_n(rst_n)"]
        lines += self._inbound(stim, conns)
        lines += self._outbound(want, conns)
        lines.append(f"  {top} dut (" + ", ".join(conns) + ");")
        lines += [
            "  initial begin",
            "    repeat (4) @(posedge clk);",
            "    rst_n = 1;",
            # The cycle count is the point of running the whole array rather
            # than reading a unit's report: fill, drain and any FIFO stall are
            # in it, and none of them are visible to HLS, which only ever saw
            # one unit. `first` catches the wavefront reaching the output.
            f"    for (integer c = 0; c < {cycles}; c = c + 1) begin",
            "      @(posedge clk);",
            "      if (produced > 0 && first < 0) first = c;",
            "      if (produced == TOTAL) begin",
            '        $display("SPMW COSIM %s (%0d/%0d tokens, %0d errors)",',
            '                 errors == 0 ? "PASS" : "FAIL", produced, TOTAL, errors);',
            '        $display("SPMW CYCLES total=%0d first_out=%0d",',
            "                 c + 1, first + 1);",
            "        $finish;",
            "      end",
            "    end",
            '    $display("SPMW COSIM TIMEOUT (%0d/%0d tokens, %0d errors)",',
            "             produced, TOTAL, errors);",
            '    $display("SPMW CYCLES total=-1 first_out=%0d", first + 1);',
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

    # Floating point cannot be compared bit-for-bit across two implementations.
    # The FFT's twiddles are irrational whatever the input, so the reference and
    # the hardware round differently -- observed at 1 to 4 ULP. A relative
    # tolerance with an absolute floor keeps the check meaningful (a real wiring
    # error is O(1) wrong, not O(1e-7)) without failing on rounding.
    _REL_TOL = 1e-5
    _ABS_TOL = 1e-6

    def _differs(self, name, channel):
        """The mismatch condition for one token: exact, or tolerant per element."""
        fam = self.families[name]
        got = f"{name}_din[{channel}]"
        want = f"{name}_exp{channel}[{name}_q{channel}]"
        element = _SCALAR_BITS[str(fam.dtype)]
        if str(fam.dtype) not in ("f32", "f64"):
            return f"{got} !== {want}"
        count = 1
        for extent in fam.block:
            count *= int(extent)
        cast = "$bitstoshortreal" if element == 32 else "$bitstoreal"
        terms = []
        for pos in range(count):
            lo = element * pos
            g = f"{cast}({got}[{lo + element - 1}:{lo}])"
            w = f"{cast}({want}[{lo + element - 1}:{lo}])"
            # abs() without a function: the difference either way.
            terms.append(
                f"((({g}) - ({w}) > {self._ABS_TOL} + {self._REL_TOL} * "
                f"((({w}) < 0.0) ? -({w}) : ({w}))) || "
                f"((({w}) - ({g})) > {self._ABS_TOL} + {self._REL_TOL} * "
                f"((({w}) < 0.0) ? -({w}) : ({w}))))"
            )
        return " || ".join(terms)

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
                    f"      if ({self._differs(name, k)}) begin",
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

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shell that makes a fabric a *device*.

A fabric's interface is streams, and at any real size that is far more signal
than a package has pins. The 16x16 TPU's top asks for **11,044** of them --
``mac_w_mem_dout`` alone is 256 channels of 32 bits -- against roughly 600 on a
U280. `place_design` does not fail because the design is large; it fails because
the design has no boundary a chip can present. Synthesis never notices, because
synthesis never places anything.

So a fabric is not a deployable unit. What is deployable is a fabric plus a
shell: something that turns those thousands of stream wires into a handful of
AXI ports by holding the data on-chip and feeding the streams from inside.

That is what this generates, and the shape is the mover's, one level up. A mover
serves one *binding* and gives it an AXI master; a shell's feeder serves one
whole *family* -- every channel of it -- so the port count collapses to one per
family rather than one per channel:

    per-channel AXI masters, 16x16 TPU      314
    per-family  AXI masters, same design      6

Each feeder is an HLS function with an array of stream ports, so Vitis writes
the AXI master and the handshaking; nothing here hand-writes a burst. The host
lays each family's tokens out in the order its channels consume them -- the
order `rtl.boundary_plan` already states -- which keeps the index arithmetic on
the host where it is cheap and leaves the hardware a flat walk.
"""

from .abi import _width
from .rtl import StructuralEmitter, _volume, boundary_plan
from .ports import IN

#: C types by token width. A family's token is one word, whatever it packs.
_CTYPE = {8: "int8_t", 16: "int16_t", 32: "int32_t", 64: "int64_t"}


def families(graph):
    """Every boundary family, with what a feeder needs to know about it.

    ``steps`` is how many tokens each channel carries and ``channels`` how many
    there are; together they are the buffer the host has to lay out.
    """
    emitter = StructuralEmitter(graph)
    plan = boundary_plan(graph)
    _internal, boundary = emitter.families()
    out = []
    for fam in boundary:
        entry = plan.get(fam.name)
        if entry is None:
            raise ValueError(f"no transfer plan for boundary family `{fam.name}`")
        channels = entry["channels"]
        steps = len(channels[0])
        if any(len(c) != steps for c in channels):
            raise ValueError(
                f"`{fam.name}` channels carry {sorted({len(c) for c in channels})} "
                f"tokens; a feeder walks them in lockstep, so they must agree."
            )
        width = _width(fam)
        if width not in _CTYPE:
            raise ValueError(f"`{fam.name}` is {width} bits, which has no C type")
        out.append(
            {
                "name": fam.name,
                "reads": emitter.boundary_direction(fam) == IN,
                "width": width,
                "ctype": _CTYPE[width],
                "channels": len(channels),
                "steps": steps,
                "tensor": entry["tensor"],
                "plan": channels,
            }
        )
    return out


def feeder_cpp(fam):
    """One family's DMA, as HLS C++.

    The loop order is the array's: every channel takes its step-t token before
    any takes step t+1. Filling one channel at a time would stall on a FIFO two
    deep while the array waited for a neighbour.
    """
    name, ctype = fam["name"], fam["ctype"]
    verb, port = ("in", "out") if fam["reads"] else ("out", "in")
    body = (
        f"      {port}[c].write(src[t * {fam['channels']} + c]);"
        if fam["reads"]
        else f"      dst[t * {fam['channels']} + c] = {port}[c].read();"
    )
    buf = "const %s *src" % ctype if fam["reads"] else "%s *dst" % ctype
    total = fam["channels"] * fam["steps"]
    return f"""#include <hls_stream.h>
#include <stdint.h>

// {name}: {fam['channels']} channel(s) x {fam['steps']} token(s) of {fam['width']} bits,
// from `{fam['tensor']}`, laid out by the host in the order the channels read.
extern "C" {{
void {_dma_name(fam)}({buf}, hls::stream<{ctype}> {port}[{fam['channels']}]) {{
#pragma HLS interface m_axi port={'src' if fam['reads'] else 'dst'} \
offset=direct bundle=gmem depth={total}
  for (int t = 0; t < {fam['steps']}; t++) {{
    for (int c = 0; c < {fam['channels']}; c++) {{
#pragma HLS pipeline II=1
{body}
    }}
  }}
}}
}}
"""


def _dma_name(fam):
    return ("feed_" if fam["reads"] else "drain_") + fam["name"]


def host_buffer(fam, arrays):
    """The family's tokens, in the order its channels consume them.

    The same index map the reference simulator and the cosim testbench use, so
    the device is driven by the design's own account of what each channel wants
    rather than by a second guess at it.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    array = arrays[fam["tensor"]]
    out = np.zeros(fam["channels"] * fam["steps"], dtype=array.dtype)
    for channel, indices in enumerate(fam["plan"]):
        for step, index in enumerate(indices):
            out[step * fam["channels"] + channel] = array[tuple(index)]
    return out


def scatter(fam, flat, arrays):
    """The inverse: put a drained family's tokens back where they belong."""
    array = arrays[fam["tensor"]]
    for channel, indices in enumerate(fam["plan"]):
        for step, index in enumerate(indices):
            array[tuple(index)] = flat[step * fam["channels"] + channel]
    return array


def _lfsr_of(width):
    """The LFSR, shaped to a channel's token width.

    A part-select of a concatenation is not portable Verilog, so a wide token
    takes whole copies and a narrow one takes a slice.
    """
    if width <= 32:
        return f"lfsr[{width - 1}:0]" if width > 1 else "lfsr[0]"
    reps, rem = divmod(width, 32)
    if rem:
        raise ValueError(f"no LFSR shaping for a {width}-bit token")
    return f"{{{reps}{{lfsr}}}}"


def harness_sv(graph, top="spmw_top", name="spmw_harness"):
    """A wrapper that gives the fabric a boundary a chip can present.

    Not a memory system -- :func:`feeder_cpp` is that. This is the smaller
    question: can the fabric itself place and route, and what does each stage
    cost? To ask it at all the 11,044 stream signals have to stop being pins, so
    every inbound channel is driven by an on-chip LFSR and every outbound one is
    folded into a single register.

    The LFSR matters. Tying the inputs to constants would let synthesis fold the
    whole array away and report a wonderful, meaningless result; a maximal-length
    shift register is cheap, and its output is opaque enough that nothing
    upstream of it can be trimmed. Likewise the outputs are XORed into `sig`
    rather than left dangling, so the drain logic survives.

    What this measures is therefore the fabric's own implementation cost, with a
    wrapper that is a rounding error beside 272 instances. What it does *not*
    measure is a deployable design, which has to carry real DMA as well.
    """
    fams = families(graph)
    lines = [
        "`timescale 1ns/1ps",
        "",
        f"module {name} (",
        "  input  wire ap_clk,",
        "  input  wire ap_rst_n,",
        "  input  wire        start,",
        "  output reg  [31:0] sig",
        ");",
        "  // A maximal-length LFSR: cheap, and opaque enough that synthesis",
        "  // cannot fold the array away behind it.",
        "  reg [31:0] lfsr;",
        "  always @(posedge ap_clk)",
        "    if (!ap_rst_n) lfsr <= 32'h1;",
        "    else if (start) lfsr <= {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};",
        "",
    ]
    conns = [".ap_clk(ap_clk)", ".ap_rst_n(ap_rst_n)"]
    folds = []
    for fam in fams:
        count, width = fam["channels"], fam["width"]
        span = f"[{width - 1}:0] " if width > 1 else ""
        base = fam["name"]
        if fam["reads"]:
            lines += [
                f"  wire {span}{base}_dout [0:{count - 1}];",
                f"  wire {base}_empty_n [0:{count - 1}];",
                f"  wire {base}_read [0:{count - 1}];",
                f"  genvar g_{base};",
                "  generate",
                f"    for (g_{base} = 0; g_{base} < {count}; g_{base} = g_{base} + 1)"
                f" begin : gen_{base}",
                f"      assign {base}_dout[g_{base}] = {_lfsr_of(width)} ^ g_{base};",
                f"      assign {base}_empty_n[g_{base}] = start;",
                "    end",
                "  endgenerate",
            ]
            conns += [
                f".{base}_dout({base}_dout)",
                f".{base}_empty_n({base}_empty_n)",
                f".{base}_read({base}_read)",
            ]
            folds.append(f"{base}_read[0]")
        else:
            lines += [
                f"  wire {span}{base}_din [0:{count - 1}];",
                f"  wire {base}_write [0:{count - 1}];",
                f"  wire {base}_full_n [0:{count - 1}];",
                f"  genvar g_{base};",
                "  generate",
                f"    for (g_{base} = 0; g_{base} < {count}; g_{base} = g_{base} + 1)"
                f" begin : gen_{base}",
                f"      assign {base}_full_n[g_{base}] = 1'b1;",
                "    end",
                "  endgenerate",
            ]
            conns += [
                f".{base}_din({base}_din)",
                f".{base}_write({base}_write)",
                f".{base}_full_n({base}_full_n)",
            ]
            for channel in range(count):
                folds.append(f"{base}_din[{channel}][{min(31, width - 1)}:0]")
                folds.append(f"{{31'b0, {base}_write[{channel}]}}")
            del span
    lines += [
        "",
        f"  {top} dut (\n      " + ",\n      ".join(conns) + ");",
        "",
        "  // Fold every output into one register: a dangling result is a result",
        "  // synthesis is entitled to delete.",
        "  always @(posedge ap_clk)",
        "    if (!ap_rst_n) sig <= 32'b0;",
        "    else sig <= sig",
    ]
    for fold in folds:
        # Everything folds as 32 bits: a narrower signal is zero-extended, a
        # wider one sliced. The value is meaningless -- what matters is that
        # every output has a reader, so none of the array can be trimmed.
        wide = (
            fold
            if fold.startswith("{") or fold.endswith(":0]")
            else ("{31'b0, %s}" % fold)
        )
        lines.append(f"      ^ {wide}")
    lines += ["      ;", "endmodule", ""]
    return "\n".join(lines)


__all__ = ["families", "feeder_cpp", "harness_sv", "host_buffer", "scatter"]

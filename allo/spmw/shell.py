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
        # A C type is only needed by `feeder_cpp`, which builds real DMA. The
        # place-and-route harness needs the width alone, and a design whose
        # boundary carries a packed vector -- the daisy chain drains a whole
        # int16[16] column, 256 bits -- has no scalar C type and does not need
        # one to be implemented. So this records what it can and lets the
        # feeder refuse, rather than refusing here for every caller.
        out.append(
            {
                "name": fam.name,
                "reads": emitter.boundary_direction(fam) == IN,
                "width": width,
                "ctype": _CTYPE.get(width),
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

    ``steps`` is an argument, not a constant. The array reads its own trip count
    out of the instruction stream, so a shell that walked a compile-time number
    of tokens would be the one part of the machine that still had a shape built
    into it -- and it is the part that decides whether a bigger model needs a
    new bitstream. The channel *count* stays fixed: that is how many wires leave
    the array, which is hardware.

    A token wider than any scalar -- a cell's 256-tile weight file is 2,048
    bits -- is moved as ``ceil(width / 512)`` beats of the widest AXI word and
    reassembled on the way into the stream. The host lays every token out as
    contiguous little-endian bytes, so beat k is bits ``[512k, 512k+512)`` and
    nothing has to agree on an order that is not already the byte order.
    """
    name, ctype = fam["name"], fam["ctype"]
    width = fam["width"]
    channels = fam["channels"]
    reads = fam["reads"]
    port = "out" if reads else "in"
    total = fam["channels"] * fam["steps"]
    if ctype is not None:
        elem = ctype
        beats = 1
        depth = total
        includes = "#include <hls_stream.h>\n#include <stdint.h>"
        body = (
            f"      {port}[c].write(src[t * {channels} + c]);"
            if reads
            else f"      dst[t * {channels} + c] = {port}[c].read();"
        )
        buf = f"const {elem} *src" if reads else f"{elem} *dst"
    else:
        elem = f"ap_uint<{width}>"
        beats = -(-width // 512)
        depth = total * beats
        cap = max(1024, 1 << (width - 1).bit_length())
        includes = (
            f"#define AP_INT_MAX_W {cap}\n#include <ap_int.h>\n"
            "#include <hls_stream.h>\n#include <stdint.h>"
        )
        if reads:
            body = (
                f"      {elem} tok;\n"
                f"      for (int b = 0; b < {beats}; b++) {{\n"
                f"#pragma HLS unroll\n"
                f"        tok.range(b * 512 + 511, b * 512) = "
                f"src[(t * {channels} + c) * {beats} + b];\n"
                f"      }}\n"
                f"      {port}[c].write(tok);"
            )
        else:
            body = (
                f"      {elem} tok = {port}[c].read();\n"
                f"      for (int b = 0; b < {beats}; b++) {{\n"
                f"#pragma HLS unroll\n"
                f"        dst[(t * {channels} + c) * {beats} + b] = "
                f"tok.range(b * 512 + 511, b * 512);\n"
                f"      }}"
            )
        buf = "const ap_uint<512> *src" if reads else "ap_uint<512> *dst"
    return f"""{includes}

// {name}: {channels} channel(s) x `steps` token(s) of {width} bits,
// from `{fam['tensor']}`, laid out by the host in the order the channels read.
// {fam['steps']} is what this design was elaborated with and sizes the burst
// depth; the loop itself runs as far as the host says.
extern "C" {{
void {_dma_name(fam)}({buf}, int steps, hls::stream<{elem}> {port}[{channels}]) {{
#pragma HLS interface m_axi port={'src' if reads else 'dst'} \
offset=direct bundle=gmem depth={depth}
#pragma HLS interface ap_none port=steps
  for (int t = 0; t < steps; t++) {{
    for (int c = 0; c < {channels}; c++) {{
#pragma HLS pipeline II={beats}
{body}
    }}
  }}
}}
}}
"""


def _dma_name(fam):
    return ("feed_" if fam["reads"] else "drain_") + fam["name"]


def host_buffer(fam, arrays):
    """The family's tokens as bytes, in the order its channels consume them.

    The same index map the reference simulator and the cosim testbench use, so
    the device is driven by the design's own account of what each channel wants
    rather than by a second guess at it.

    A channel's token is not always a scalar. A weight cell holds ``NW`` of them
    and a bias ``NB``, which the fabric carries as one wider word -- 32 bits of
    four ``int8``, 64 bits of two ``int32`` -- so the token is the packed
    vector. `tobytes` already lays a row out with element 0 in the low bytes,
    which is where ``hls::vector`` reads it, so packing is the identity here
    rather than a convention this has to invent.

    Returns bytes, because the board host is numpy-free and every consumer
    wants the buffer rather than its dtype.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    array = arrays[fam["tensor"]]
    word = fam["width"] // 8
    out = bytearray(fam["channels"] * fam["steps"] * word)
    for channel, indices in enumerate(fam["plan"]):
        for step, index in enumerate(indices):
            raw = np.ascontiguousarray(array[tuple(index)]).tobytes()
            if len(raw) != word:
                raise ValueError(
                    f"`{fam['name']}` carries {word}-byte tokens but "
                    f"`{fam['tensor']}`{list(index)} is {len(raw)} bytes; the "
                    f"family's width and the tensor's element disagree."
                )
            offset = (step * fam["channels"] + channel) * word
            out[offset : offset + word] = raw
    return bytes(out)


def scatter(fam, flat, arrays):
    """The inverse: put a drained family's tokens back where they belong."""
    array = arrays[fam["tensor"]]
    for channel, indices in enumerate(fam["plan"]):
        for step, index in enumerate(indices):
            array[tuple(index)] = flat[step * fam["channels"] + channel]
    return array


def _lfsr_of(width, src="lfsr"):
    """The LFSR, shaped to a channel's token width.

    A part-select of a concatenation is not portable Verilog, so a wide token
    takes whole copies and a narrow one takes a slice.
    """
    if width <= 32:
        return f"{src}[{width - 1}:0]" if width > 1 else f"{src}[0]"
    reps, rem = divmod(width, 32)
    if rem:
        raise ValueError(f"no LFSR shaping for a {width}-bit token")
    return f"{{{reps}{{{src}}}}}"


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
        "  // A maximal-length LFSR per *channel*, not one shared.",
        "  //",
        "  // Sharing one was the first thing measured and it was wrong: a single",
        "  // net driving every channel of every family is a fanout of hundreds",
        "  // across the whole die, and it became the critical path -- 3.877 ns of",
        "  // route against 0.080 ns of logic, sourced at `lfsr_reg` and sinking",
        "  // in a MAC cell. That measured the harness, not the fabric. One small",
        "  // LFSR per channel is a few flip-flops each and keeps every source",
        "  // beside its load.",
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
                f"      reg [31:0] lf_{base};",
                "      always @(posedge ap_clk)",
                f"        if (!ap_rst_n) lf_{base} <= 32'h1 + g_{base};",
                f"        else if (start) lf_{base} <= {{lf_{base}[30:0], "
                f"lf_{base}[31]^lf_{base}[21]^lf_{base}[1]^lf_{base}[0]}};",
                f"      assign {base}_dout[g_{base}] = {_lfsr_of(width, f'lf_{base}')};",
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


#: Clock-region geometry of the parts we floorplan for. A U280 is 8 columns by
#: 12 rows, three SLRs of four rows each; crossing an SLR costs a hop through
#: dedicated silicon and is the thing a floorplan most wants to avoid.
_GEOMETRY = {"xcu280": {"cols": 8, "rows": 4, "slrs": 3}}


class Crossing:
    """Which of a family's channels cross a slot boundary, and where.

    A family says every channel makes the same *journey*; it does not say every
    channel makes it in the same *place*.  The southward links of a 16-by-16
    mesh all travel one row, but with the array cut into four bands only the 48
    that land on rows 4, 8 and 12 actually leave a band.  Anchoring the family
    wholesale would build 768 registers to do the work of 48.

    The distinction is free to make here because `channel_index` subscripts an
    affine family by its *destination* site, row-major, so the channel's index
    already carries the coordinate the cut is defined on.
    """

    __slots__ = ("cuts", "extent", "stride")

    def __init__(self, cuts, extent, stride):
        self.cuts = tuple(cuts)
        self.extent = int(extent)
        self.stride = int(stride)

    def __repr__(self):
        return f"Crossing(cuts={list(self.cuts)}, extent={self.extent})"

    def coordinate(self, index):
        """The cut coordinate of channel ``index`` -- what the generate computes."""
        return (index // self.stride) % self.extent

    def crosses(self, index):
        return self.coordinate(index) in self.cuts

    def channels(self, count):
        """How many of ``count`` channels this plan anchors."""
        return sum(1 for i in range(count) if self.crosses(i))


def crossing_families(graph, slots=4, axis=0):
    """Which channels cross a slot boundary, so a register can be put on them.

    This is the question AutoBridge answers with an ILP over a dataflow graph it
    recovered from C++, and RapidStream answers per inter-island net.  A spatial
    program does not have to ask it that way.  A ``link`` family has one
    ``offset`` -- the displacement from writer to reader, which the elaborator
    records only when every channel in the family agrees on it -- so the family
    says which *way* its channels go, and the channel's own index says where.
    Neither is a search.

    Returns a map from family name to a `Crossing`, one anchor stage per
    crossing channel.  A family travelling along an axis the floorplan does not
    cut is absent; so is a TABLE, which has no single journey to cut.

    Adding latency to the crossing channels is safe here for the reason cut-set
    pipelining is safe generally -- they *are* a cut-set.  The channels landing
    on one row all come from the row above, carry the same signal, and point the
    same way, so delaying them delays one frontier of the array by one cycle and
    re-balances nothing else.  AutoBridge has to solve an SDC to find such a
    set; here it is the unit of declaration.

    The one case this must refuse is a family on a dependency cycle, where added
    latency would cost throughput rather than just latency.  A mesh whose links
    all displace positively along an axis cannot close a cycle among themselves;
    a cycle has to come back through a binding, and a binding is not a family
    this returns.
    """
    emitter = StructuralEmitter(graph)
    internal, _boundary = emitter.families()
    cut = {}
    for placement in emitter.placements():
        grid = tuple(placement.grid)
        if len(grid) != 2:
            continue  # `floorplan_xdc` only cuts a mesh, so only a mesh crosses
        rows = int(grid[axis])
        # A band boundary sits where `floorplan_xdc` starts a new band; the
        # array's own edge is not a boundary, so slot 0 does not contribute.
        bounds = sorted({(s * rows) // slots for s in range(1, slots)})
        for fam in emitter.peer_families(placement):
            cut[id(fam)] = bounds
    out = {}
    for fam in internal:
        bounds = cut.get(id(fam))
        if not bounds:
            continue
        offset = getattr(fam, "offset", None)
        if not offset or axis >= len(offset) or not offset[axis]:
            continue  # a table, a fanout, or a journey the cut does not cross
        stride = 1
        for extent in fam.shape[axis + 1 :]:
            stride *= int(extent)
        out[fam.name] = Crossing(bounds, int(fam.shape[axis]), stride)
    return out


def _band_of_site(row, rows, slots):
    """Which band a mesh row falls in."""
    for slot in range(slots):
        if (slot * rows) // slots <= row < ((slot + 1) * rows) // slots:
            return slot
    return slots - 1


def _fifo_instances(top, name, index):
    """Every spelling the emitter might have used for one channel's FIFO.

    A plain internal family is `g_F[i].u`; an anchored one puts the same FIFO
    inside a generate-if, so it is `g_F[i].direct.u` on the channels that do not
    cross and `g_F[i].anchored.u` plus `g_F[i].anchored.u_anchor` on the ones
    that do. Naming all four and letting `-quiet` drop the misses is cheaper
    than re-deriving which branch each channel took.
    """
    stem = f"{top}/g_{name}[{index}]"
    return [
        f"{stem}.u",
        f"{stem}.direct.u",
        f"{stem}.anchored.u",
        f"{stem}.anchored.u_anchor",
    ]


def _channel_readers(emitter, placement, internal_ids):
    """(family name, channel) -> the site that reads it, for internal families.

    The reader rather than the writer: a FIFO's output drives the reader, and
    its occupancy counter -- which is what went critical at 32x32, 4.7 ns
    between two bits of one register -- sits beside that logic.
    """
    out = {}
    for site in placement.sites():
        for port, fam in emitter.site_ports(placement, site):
            if id(fam) not in internal_ids or port.direction is not IN:
                continue
            index = emitter.channel_index(placement, site, port, fam)
            if index >= 0:
                out[(fam.name, index)] = tuple(int(c) for c in site)
    return out


def floorplan_xdc(
    graph, part="xcu280", slr=1, top="dut", slots=4, parent=None, cols=None
):
    """A floorplan for the array, written from the grid it was placed on.

    This is the one thing a regular design should not have to be *told*. The
    emitter already knows the mesh is 16 by 16 and which instance sits at which
    site, so it can say where each row should go rather than leaving a placer to
    infer neighbourliness from a netlist.

    Two decisions, and they are the ones that matter on this part:

    * **Keep the array inside one SLR.** A systolic array's nets are all
      neighbour-to-neighbour, and an SLR crossing turns one of those into a trip
      through Laguna. Constraining the whole fabric to a single SLR costs
      nothing here -- 272 instances is 8% of the device.
    * **Give each band of mesh rows its own pblock**, stacked in the same order
      the rows are stacked logically, so a partial sum travelling south travels
      south on the die too.  Bands, not rows: the point of a floorplan is to
      say which logic is far apart, and sixteen constraints that pin sixteen
      rows into four clock regions say nothing four constraints do not.

    The result is advisory: `pblock` ranges are soft, so a design that does not
    fit degrades rather than failing.

    A floorplan on its own is not expected to help, and measured on this design
    it does not -- it takes options away from the placer without shortening
    anything.  It pays only together with anchor registers on the links that
    cross the slots it creates; `crossing_families` says which those are.
    """
    geom = _GEOMETRY.get(part.split("-")[0])
    if geom is None:
        raise ValueError(f"no clock-region geometry for `{part}`")
    # How many clock-region columns the array may actually use. The part has
    # eight, but inside a Vitis platform the reconfigurable partition is
    # narrower than the die: on this one `pblock_dynamic_region` covers
    # CLOCKREGION_X0Y4:X5Y10, and X7 belongs to `pblock_blp`, the static shell.
    # A pblock spanning the full width therefore overlaps static logic, which
    # is what HD.RECONFIGURABLE DRC objects to -- not the parenting.
    cols = geom["cols"] if cols is None else cols
    emitter = StructuralEmitter(graph)
    lines = [
        "# Generated by allo.spmw.shell.floorplan_xdc -- do not edit.",
        f"# One SLR, {slots} slot(s) of mesh rows, {part}.",
        "",
    ]
    base = slr * geom["rows"]
    # Bands are laid on consecutive clock-region rows starting at `slr`, so the
    # ceiling is how many rows remain above it -- not one SLR's worth. A 32 by
    # 32 array does not fit in a single SLR (about 300k LUTs holds ~950 cells at
    # ~305 each, so roughly 31 by 31), and constraining it to one would be
    # asking for the impossible rather than for a floorplan.
    slots = min(slots, geom["rows"] * geom["slrs"] - slr * geom["rows"])
    internal, _boundary = emitter.families()
    internal_ids = {id(fam) for fam in internal}

    mesh = [p for p in emitter.placements() if len(tuple(p.grid)) == 2]
    chains = [p for p in emitter.placements() if len(tuple(p.grid)) != 2]

    # Every unit and every FIFO gets a band. The first version of this banded
    # the mesh cells and nothing else, which at 32x32 cost 35% of the clock:
    # the design has 3,136 FIFOs and 32 vector lanes, and packing 1,024 mesh
    # cells into eight thin bands left all of them to fill the gaps. A FIFO
    # whose two counter bits land in different regions is a 4.7 ns path. A
    # partial floorplan is worse than none.
    bands = [[] for _ in range(slots)]
    for placement in mesh:
        names = emitter.role_names(placement)
        rows = int(tuple(placement.grid)[0])
        for order, (_sig, _rt, sites) in enumerate(emitter.classes(placement)):
            for site in sites:
                tag = "_".join(str(int(c)) for c in site)
                # Exact paths, not patterns. The instance name carries the site
                # as `u_mac_<role>_<row>_<col>`, and a glob over it is
                # ambiguous: `u_mac_*_0_*` matches row 10 as readily as row 0,
                # because the leading `*` will happily eat `r0_1`.
                bands[_band_of_site(int(site[0]), rows, slots)].append(
                    f"{top}/u_{names[order]}_{tag}"
                )
        for (name, index), site in _channel_readers(
            emitter, placement, internal_ids
        ).items():
            slot = _band_of_site(site[0], rows, slots)
            bands[slot] += _fifo_instances(top, name, index)

    # A chain is linked from the mesh's last row, so its home is the last band
    # -- not "wherever the placer likes", which is what leaving it out meant.
    for placement in chains:
        names = emitter.role_names(placement)
        for order, (_sig, _rt, sites) in enumerate(emitter.classes(placement)):
            for site in sites:
                tag = "_".join(str(int(c)) for c in site)
                bands[-1].append(f"{top}/u_{names[order]}_{tag}")
        for name, index in _channel_readers(emitter, placement, internal_ids):
            bands[-1] += _fifo_instances(top, name, index)

    for slot, cells in enumerate(bands):
        if not cells:
            continue
        cr = base + slot
        pb = f"pb_slot{slot}"
        lines += [
            f"create_pblock {pb}",
            f"resize_pblock {pb} -add "
            f"{{CLOCKREGION_X0Y{cr}:CLOCKREGION_X{cols - 1}Y{cr}}}",
            f"add_cells_to_pblock {pb} [get_cells -quiet {{{' '.join(cells)}}}]",
        ]
        if parent:
            # Matched by suffix, because a *scoped* XDC renames the pblock
            # after the instance it is scoped to -- the DRC message that led
            # here named `level0_i_ulp_spmw_kernel_1_inst_pb_mac_slot0`. A bare
            # `[get_pblocks pb_slot0]` finds nothing there and the reparenting
            # silently does not happen.
            lines.append(f"set_property PARENT {parent} [get_pblocks -quiet *{pb}]")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "families",
    "feeder_cpp",
    "Crossing",
    "crossing_families",
    "floorplan_xdc",
    "harness_sv",
    "host_buffer",
    "scatter",
]

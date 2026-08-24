# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural RTL for the fabric, over HLS-synthesised units.

**HLS builds the unit, RTL builds the array.** One C synthesis and one IP export
per *role* -- nine for a mesh, whatever its size -- and the replication and
wiring are a Verilog ``generate`` nest, which Vivado elaborates from an
already-synthesised module rather than scheduling again.  Synthesis cost tracks
the role count; the array's size is paid only in elaboration.

That is a different bargain from emitting the array as an unrolled loop inside
HLS C++, where the tool still elaborates every instance itself.

``feat/spmw`` established this flow and validated it on hardware, but its
emitter derived the wiring from a four-entry literal and refused anything but
the systolic GEMM.  Here the netlist is read off the elaborated graph -- the same
families, port maps and site classes that :mod:`allo.spmw.lower_mlir` puts on
``spmw.map`` -- so the shape of the design is an input rather than an assumption.

Two kinds of family come out of resolution and they become different things:

* a **peer** family (site to site) becomes internal FIFOs;
* a **boundary** family (a port bound to a loader or drain) becomes a top-level
  stream port, because that is where the array meets the DMA.
"""

from . import channels as ch
from .errors import SPMWBindingError
from .lower_df import _wiring_classes
from .lower_mlir import RolledEmitter
from .ports import IN, MEMORY, OUT, READWRITE, STREAM

# Width of one token, by declared type. A block-carrying port packs its elements
# into one word, as the HLS ap_fifo interface does.
_WIDTH = {
    "f64": 64,
    "f32": 32,
    "f16": 16,
    "i64": 64,
    "i32": 32,
    "i16": 16,
    "i8": 8,
    "i1": 1,
    "ui64": 64,
    "ui32": 32,
    "ui16": 16,
    "ui8": 8,
    "index": 64,
}


def _width(family):
    """Bit width of one token on a family."""
    name = str(family.dtype)
    bits = _WIDTH.get(name)
    if bits is None:
        raise SPMWBindingError(
            f"family `{family.name}` carries `{name}`, which has no RTL width; "
            f"add it to allo.spmw.rtl._WIDTH."
        )
    for extent in family.block:
        bits *= int(extent)
    return bits


def _volume(shape):
    n = 1
    for extent in shape:
        n *= int(extent)
    return n


def fifo_module():
    """A synchronous FIFO with the ap_fifo handshake HLS exports.

    Write side is ``din``/``full_n``/``write``, read side ``dout``/``empty_n``/
    ``read`` -- what a free-running (``ap_ctrl_none``) HLS IP presents for an
    ``hls::stream`` argument, so a role IP drops straight onto it.
    """
    return """`timescale 1ns/1ps

module spmw_fifo #(parameter DW = 32, parameter DEPTH = 2) (
  input  wire          clk,
  input  wire          rst_n,
  input  wire [DW-1:0] din,
  output wire          full_n,
  input  wire          write,
  output wire [DW-1:0] dout,
  output wire          empty_n,
  input  wire          read
);
  localparam AW = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
  reg [DW-1:0] mem [0:DEPTH-1];
  reg [AW:0]   count;
  reg [AW-1:0] rptr, wptr;
  assign full_n  = (count != DEPTH);
  assign empty_n = (count != 0);
  assign dout    = mem[rptr];
  wire do_wr = write & full_n;
  wire do_rd = read  & empty_n;
  always @(posedge clk) begin
    if (!rst_n) begin
      count <= 0; rptr <= 0; wptr <= 0;
    end else begin
      if (do_wr) begin mem[wptr] <= din; wptr <= (wptr == DEPTH-1) ? 0 : wptr + 1; end
      if (do_rd) rptr <= (rptr == DEPTH-1) ? 0 : rptr + 1;
      count <= count + (do_wr ? 1 : 0) - (do_rd ? 1 : 0);
    end
  end
endmodule
"""


class CoordPort:
    """A grid coordinate, presented to the unit as an ordinary input stream.

    A role whose body reads its own position -- an FFT butterfly needs its stage
    and index -- cannot be one module unless the position is an *input*.  Making
    it a stream rather than a new interface kind means the unit stays a
    free-running IP and the fabric wires it like anything else; the unit reads it
    once and holds it, exactly as it does a stationary weight.
    """

    __slots__ = ("name", "axis", "direction", "protocol", "dtype", "shape", "depth")

    def __init__(self, axis):
        # pylint: disable=import-outside-toplevel
        from allo.ir.types import int32

        self.axis = axis
        self.name = f"_pid{axis}"
        self.direction = IN
        self.protocol = STREAM
        # A real Allo type, not its spelling: the emitted program annotates the
        # held value with it, and `type_ann` injects the object itself.
        self.dtype = int32
        self.shape = ()
        self.depth = 1

    def __repr__(self):
        return f"<coord {self.name}>"


def const_module():
    """A stream that always holds one constant.

    Not a FIFO: a coordinate is not produced by anything, it simply *is*. Holding
    ``empty_n`` high means a unit can read it whenever it starts, and reading it
    never empties it, so no reset ordering matters.
    """
    return """`timescale 1ns/1ps

module spmw_const #(parameter DW = 32, parameter [63:0] VAL = 0) (
  output wire [DW-1:0] dout,
  output wire          empty_n,
  input  wire          read
);
  assign dout    = VAL[DW-1:0];
  assign empty_n = 1'b1;
endmodule
"""


def _port_signals(name, direction, width):
    """One stream port's ap_fifo signals, from the unit's point of view."""
    if direction == IN:
        return [
            (f"{name}_dout", "input", width),
            (f"{name}_empty_n", "input", 1),
            (f"{name}_read", "output", 1),
        ]
    return [
        (f"{name}_din", "output", width),
        (f"{name}_full_n", "input", 1),
        (f"{name}_write", "output", 1),
    ]


def _decl(signals, indent="  "):
    out = []
    for sig, kind, width in signals:
        span = f"[{width - 1}:0] " if width > 1 else "               "
        pad = "input  " if kind == "input" else "output "
        out.append(f"{indent}{pad}wire {span}{sig}")
    return out


class StructuralEmitter:
    """Emits the fabric: FIFOs for peer channels, one role instance per site."""

    def __init__(self, graph):
        self.graph = graph
        self.rolled = RolledEmitter(graph)
        self._mem_families = {}

    # -- inspection ---------------------------------------------------------

    def placements(self):
        """Every placement the fabric elaborated, in declaration order."""
        return self.rolled.placements()

    def role_names(self, placement):
        """One module name per wiring class -- the units HLS will synthesise."""
        res = self.rolled.low.resolutions[placement]
        return [
            f"{self.rolled.low.kernel_names[placement]}_r{k}"
            for k in range(len(_wiring_classes(placement, res)))
        ]

    def classes(self, placement):
        """The wiring classes: `(signature, routing, sites)` per role."""
        res = self.rolled.low.resolutions[placement]
        return _wiring_classes(placement, res)

    def peer_families(self, placement):
        """Families that link site to site; these become internal FIFOs."""
        return list(self.rolled.low.resolutions[placement].families.values())

    def bind_families(self, placement):
        """Families a binding owns at this placement: a loader, drain, or link."""
        out = []
        for (pl, _port), fam in self.rolled.low.bind_families.items():
            if pl is placement and fam not in out:
                out.append(fam)
        return out

    @staticmethod
    def is_internal(fam):
        """Does this family have both of its ends inside the fabric?

        A ``link`` joins one placement's output to another's input, so both ends
        are sites and it needs real FIFOs.  A loader or drain has only one end
        here -- the other is the DMA -- so it becomes a top-level port.  The
        difference shows up as whether the family's slots cover both directions,
        which is the property itself rather than a proxy for it.

        Getting this wrong is not cosmetic: a link emitted as a port would leave
        two placements silently unconnected.
        """
        directions = {port.direction for _site, port in fam.slots}
        return IN in directions and OUT in directions

    def families(self):
        """Every family in the design, deduplicated and classified.

        Deduplication is across placements, not within one: ``_plan_link``
        registers a single family under *both* sides, so a per-placement walk
        would declare it twice -- which the elaborator rejects as a redeclared
        port.
        """
        internal, boundary = [], []
        for placement in self.placements():
            for fam in self.peer_families(placement):
                if fam not in internal:
                    internal.append(fam)
            for fam in self.bind_families(placement):
                bucket = internal if self.is_internal(fam) else boundary
                if fam not in bucket:
                    bucket.append(fam)
            for fam in self.memory_families(placement):
                if fam not in boundary:
                    boundary.append(fam)
        return internal, boundary

    def memory_families(self, placement):
        """Every memory port that crosses the edge, as a stream.

        A resident port is skipped: its contents are compile-time, so it never
        reaches the DMA.
        """
        return [
            self.memory_family(placement, port)
            for port in placement.iface.ports()
            if port.protocol == MEMORY and not self.is_resident(placement, port)
        ]

    def boundary_families(self, placement):
        """The loader/drain families at this placement, as top-level ports."""
        return [f for f in self.bind_families(placement) if not self.is_internal(f)]

    def resolve(self, placement, site, port):
        """This port's family at this site, or None if it connects to nothing.

        Three things can serve a port and a site takes whichever it has: a peer
        link to a neighbour, a *binding* -- a loader, drain or link -- or, for a
        memory port, the edge stream its results leave on.

        A site signature holds only ports linked to a *peer*, so a port fed by a
        loader is absent from it. Deriving the port list from the signature
        therefore dropped every loader: the fabric declared the input streams,
        connected nothing to them, and elaborated cleanly with A and B never
        entering the array.
        """
        if port.protocol == MEMORY:
            # Compile-time contents are resident, not a channel: the array holds
            # them in a per-site ROM and so does the unit. Only data that comes
            # from outside crosses the fabric's edge.
            if self.is_resident(placement, port):
                return None
            return self.memory_family(placement, port)
        if port.protocol != STREAM:
            return None
        res = self.rolled.low.resolutions[placement]
        fam = res.families.get(res.site_family.get((site, port)))
        if fam is not None:
            return fam
        fam = self.rolled.low.bind_families.get((placement, port))
        # A binding covers a *bundle*, not the whole grid, so membership is the
        # question -- an interior site has no loader even though the port does.
        if fam is not None and (site, port) in fam.slots:
            return fam
        return None

    def is_resident(self, placement, port):
        """Is this memory port backed by contents known at compile time?

        A ``mem(..., init=...)`` brick is; a shard of a caller's tensor is not.
        The first belongs inside the unit, the second has to arrive on a port,
        and telling them apart is what keeps a resident ROM out of the
        interface.
        """
        low = self.rolled.low
        binding = low.mem_reads.get((placement, port)) or low.mem_writes.get(
            (placement, port)
        )
        if binding is None:
            return False
        source = low.resolve_storage(
            binding.source
            if binding.kind in {"shard", "stationary"}
            else binding.target
        )
        return getattr(source, "init", None) is not None

    def site_ports(self, placement, site):
        """Every port connected at this site, with its family, in a stable order."""
        found = []
        for port in placement.iface.ports():
            fam = self.resolve(placement, site, port)
            if fam is not None:
                found.append((port, fam))
        return sorted(found, key=lambda pf: (pf[0].protocol, pf[0].name))

    def coord_ports(self, placement, order):
        """The coordinates this role reads, as input ports.

        Which axes a role reads is a fact about its *body*, so it is computed
        where the body is rewritten -- :mod:`allo.spmw.role_ip` -- and imported
        here rather than guessed at from the interface.
        """
        # pylint: disable=import-outside-toplevel
        from .role_ip import coord_axes

        return [CoordPort(axis) for axis in coord_axes(self.graph, placement, order)]

    def coord_family(self, placement, axis):
        """One channel per site, each a constant source holding that coordinate."""
        key = (id(placement), "coord", axis)
        if key in self._mem_families:
            return self._mem_families[key]
        prefix = self.rolled.low.kernel_names[placement]
        sites = list(placement.sites())
        port = CoordPort(axis)
        fam = ch.Family(
            f"{prefix}_pid{axis}", port.dtype, (), 1, ch.TABLE, (len(sites),)
        )
        for pos, site in enumerate(sites):
            fam.slots[(site, port)] = pos
        fam.count = len(sites)
        fam.coord_axis = axis
        self._mem_families[key] = fam
        return fam

    def coord_families(self, placement):
        """Every coordinate axis any role of this placement reads."""
        axes = set()
        for order in range(len(self.classes(placement))):
            axes |= {p.axis for p in self.coord_ports(placement, order)}
        return [self.coord_family(placement, axis) for axis in sorted(axes)]

    def unit_ports(self, placement, order):
        """The ports a role's IP carries -- one interface for all its sites.

        Every site in a wiring class must agree, or the class does not describe
        one module. Classes are formed from peer routing alone, so agreement on
        the *binding*-fed ports is a separate fact and is checked here rather
        than assumed.
        """
        _signature, _routing, sites = self.classes(placement)[order]
        first = self.site_ports(placement, sites[0])
        names = [p.name for p, _f in first]
        for site in sites[1:]:
            other = [p.name for p, _f in self.site_ports(placement, site)]
            if other != names:
                raise SPMWBindingError(
                    f"sites {sites[0]} and {site} share a wiring class but "
                    f"connect different ports ({names} vs {other}), so one IP "
                    f"cannot stand for both."
                )
        # Coordinates come last, so a role that does not read its position has
        # exactly the interface it had before they existed.
        coords = [
            (port, self.coord_family(placement, port.axis))
            for port in self.coord_ports(placement, order)
        ]
        return first + coords

    def stream_ports(self, signature):
        """Just the FIFO ports of a signature, in name order."""
        return sorted(
            (p for p in signature if p.protocol == STREAM), key=lambda p: p.name
        )

    def memory_family(self, placement, port):
        """The result (or feed) stream a bound memory port becomes in RTL.

        A memory port is per-site storage the parent gathers or scatters, so at
        the array's edge it is one channel per site, drained or filled outside
        the fabric.  Modelling it as a stream rather than a shared memory is what
        keeps the unit a free-running IP.
        """
        key = (id(placement), port.name)
        if key in self._mem_families:
            return self._mem_families[key]
        if port.access == READWRITE:
            raise SPMWBindingError(
                f"`{placement.name}.{port.name}` is shared random-access memory, "
                f"which the structural path does not model; it needs a real "
                f"memory rather than a stream."
            )
        prefix = self.rolled.low.kernel_names[placement]
        sites = list(placement.sites())
        fam = ch.Family(
            f"{prefix}_{port.name}_mem",
            port.dtype,
            port.shape,
            placement.depths.get(port, port.depth),
            ch.TABLE,
            (len(sites),),
        )
        for pos, site in enumerate(sites):
            fam.slots[(site, port)] = pos
        fam.count = len(sites)
        self._mem_families[key] = fam
        return fam

    def family_at(self, placement, site, port):
        """Which family this port addresses *at this site*.

        Routing is per site: one port can reach a neighbour at an interior site
        and a drain at the edge, so the peer table is consulted first and the
        binding table is the fallback.
        """
        if port.protocol == MEMORY:
            return self.memory_family(placement, port)
        res = self.rolled.low.resolutions[placement]
        fam = res.families.get(res.site_family.get((site, port)))
        if fam is None:
            fam = self.rolled.low.bind_families.get((placement, port))
        if fam is None:
            raise SPMWBindingError(
                f"`{placement.name}.{port.name}` is in the site signature at "
                f"{site} but addresses no family; this is an emission bug."
            )
        return fam

    def channel_index(self, placement, site, port, fam):
        """Which channel of ``fam`` this site's port attaches to.

        Affine families are subscripted by the *destination* site, so a writer
        displaces by the family's offset and a reader uses its own coordinate --
        the same convention ``lower_df`` emits, and the reason both ends of a
        link land on one FIFO.
        """
        if getattr(fam, "coord_axis", None) is not None:
            return list(placement.sites()).index(tuple(site))
        if port.protocol == MEMORY:
            return fam.slots[(site, port)]
        if fam.kind == ch.AFFINE:
            offs = fam.offset if port.direction == OUT else (0,) * len(placement.grid)
            coord = [c + o for c, o in zip(site, offs)]
            for axis, (value, extent) in enumerate(zip(coord, fam.shape)):
                if not 0 <= value < extent:
                    raise SPMWBindingError(
                        f"`{placement.name}.{port.name}` at site {site} indexes "
                        f"channel {value} on axis {axis} of family "
                        f"`{fam.name}`, whose extent is {extent}. A port that "
                        f"leaves the grid should be unbound or bound to a drain."
                    )
            flat, stride = 0, 1
            for axis in reversed(range(len(fam.shape))):
                flat += coord[axis] * stride
                stride *= int(fam.shape[axis])
            return flat
        return fam.slots.get((site, port), -1)

    # -- emission -----------------------------------------------------------

    def role_stub(self, placement, order, name):
        """A black box with the exported role IP's interface.

        The real module comes from ``export_design``; this declares the shape so
        the fabric elaborates before, or without, running HLS.
        """
        signals = []
        for port, fam in self.unit_ports(placement, order):
            signals += _port_signals(port.name, port.direction, _width(fam))
        head = [
            "  input  wire                ap_clk",
            "  input  wire                ap_rst_n",
        ]
        body = ",\n".join(head + _decl(signals))
        return f"`timescale 1ns/1ps\n\nmodule {name} (\n{body}\n);\nendmodule\n"

    def _family_wires(self, fam, internal):
        """Wire declarations for one family's channels."""
        count = _volume(fam.shape)
        width = _width(fam)
        span = f"[{width - 1}:0] "
        lines = [
            f"  // family {fam.name}: {count} channel(s), {width}-bit, "
            f"depth {fam.depth}{'' if internal else ' (boundary)'}"
        ]
        for sig in ("din", "dout"):
            lines.append(f"  wire {span}{fam.name}_{sig} [0:{count - 1}];")
        for sig in ("full_n", "write", "empty_n", "read"):
            lines.append(f"  wire {fam.name}_{sig} [0:{count - 1}];")
        if internal:
            lines += [
                f"  genvar {fam.name}_i;",
                "  generate",
                f"    for ({fam.name}_i = 0; {fam.name}_i < {count}; "
                f"{fam.name}_i = {fam.name}_i + 1) begin : g_{fam.name}",
                f"      spmw_fifo #(.DW({width}), .DEPTH({fam.depth})) u ("
                f".clk(ap_clk), .rst_n(ap_rst_n)"
                + "".join(
                    f", .{s}({fam.name}_{s}[{fam.name}_i])"
                    for s in ("din", "full_n", "write", "dout", "empty_n", "read")
                )
                + ");",
                "    end",
                "  endgenerate",
            ]
        return lines

    def _connections(self, placement, site, order):
        """This site's port connections, each resolved to one channel."""
        conns = [".ap_clk(ap_clk)", ".ap_rst_n(ap_rst_n)"]
        for port, fam in self.site_ports(placement, site) + [
            (port, self.coord_family(placement, port.axis))
            for port in self.coord_ports(placement, order)
        ]:
            idx = self.channel_index(placement, site, port, fam)
            if idx < 0:
                raise SPMWBindingError(
                    f"`{placement.name}.{port.name}` at site {site} has no "
                    f"channel in family `{fam.name}`, but the site's signature "
                    f"says it is bound."
                )
            for sig, _kind, _w in _port_signals(port.name, port.direction, _width(fam)):
                wire = sig[len(port.name) + 1 :]
                conns.append(f".{sig}({fam.name}_{wire}[{idx}])")
        return conns

    def _instances(self, placement):
        """One instance per site, grouped by role."""
        lines = []
        names = self.role_names(placement)
        for order, (_signature, _routing, sites) in enumerate(self.classes(placement)):
            lines.append(f"  // role {names[order]}: {len(sites)} instance(s)")
            self.unit_ports(placement, order)  # every site must agree
            for site in sites:
                tag = "_".join(str(c) for c in site)
                lines.append(
                    f"  {names[order]} u_{names[order]}_{tag} (\n      "
                    + ",\n      ".join(self._connections(placement, site, order))
                    + ");"
                )
        return lines

    @staticmethod
    def boundary_direction(fam):
        """Which way data crosses the array's edge on this family.

        ``IN`` means the sites read it, so the fabric consumes and the outside
        supplies; ``OUT`` is the reverse.  Every site touching a boundary family
        touches it the same way -- a family with both directions has both ends
        inside the fabric and is internal.
        """
        directions = {port.direction for _site, port in fam.slots}
        if len(directions) != 1:
            raise SPMWBindingError(
                f"boundary family `{fam.name}` is touched in {sorted(directions)} "
                f"directions; it cannot be one edge port."
            )
        return directions.pop()

    def _coord_sources(self, placement):
        """A constant source per site per coordinate axis this placement reads."""
        lines = []
        sites = list(placement.sites())
        for fam in self.coord_families(placement):
            width = _width(fam)
            lines.append(
                f"  // coordinate axis {fam.coord_axis}: {len(sites)} constant "
                f"source(s)"
            )
            lines += [
                f"  wire [{width - 1}:0] {fam.name}_dout [0:{len(sites) - 1}];",
                f"  wire {fam.name}_empty_n [0:{len(sites) - 1}];",
                f"  wire {fam.name}_read [0:{len(sites) - 1}];",
            ]
            for pos, site in enumerate(sites):
                lines.append(
                    f"  spmw_const #(.DW({width}), .VAL({int(site[fam.coord_axis])})) "
                    f"u_{fam.name}_{pos} (.dout({fam.name}_dout[{pos}]), "
                    f".empty_n({fam.name}_empty_n[{pos}]), "
                    f".read({fam.name}_read[{pos}]));"
                )
        return lines

    def _boundary_ports(self, boundary):
        """The top's stream ports: one per channel of each edge family.

        Only the three signals that direction actually needs.  Declaring all six
        elaborates too, but leaves half of them dangling and makes the interface
        ambiguous to anything driving it -- a testbench cannot tell which side of
        the handshake it owns.
        """
        ports = ["  input  wire ap_clk", "  input  wire ap_rst_n"]
        for fam in boundary:
            count = _volume(fam.shape)
            span = f"[{_width(fam) - 1}:0] "
            if self.boundary_direction(fam) == IN:
                # The outside feeds the array: it supplies data and readiness,
                # the fabric acknowledges.
                signals = (
                    ("dout", "input", span),
                    ("empty_n", "input", ""),
                    ("read", "output", ""),
                )
            else:
                signals = (
                    ("din", "output", span),
                    ("write", "output", ""),
                    ("full_n", "input", ""),
                )
            for sig, kind, wide in signals:
                ports.append(f"  {kind:<6} wire {wide}{fam.name}_{sig} [0:{count - 1}]")
        return ports

    def fabric(self, top="spmw_top"):
        """The structural top: boundary ports, internal FIFOs, role instances."""
        internal, boundary = self.families()
        body = []
        for fam in internal:
            body += self._family_wires(fam, internal=True)
        for placement in self.placements():
            body += self._coord_sources(placement)
            body += self._instances(placement)
        return (
            "`timescale 1ns/1ps\n\n"
            + f"module {top} (\n"
            + ",\n".join(self._boundary_ports(boundary))
            + "\n);\n"
            + "\n".join(body)
            + "\nendmodule\n"
        )

    def stubs(self):
        """A black box per role, so the fabric elaborates before HLS runs."""
        out = []
        for placement in self.placements():
            for order, name in enumerate(self.role_names(placement)):
                out.append(self.role_stub(placement, order, name))
        return out


def emit_structural_verilog(graph, top="spmw_top"):
    """The whole fabric as one SystemVerilog source: FIFO, role stubs, top."""
    emitter = StructuralEmitter(graph)
    return "\n".join(
        [fifo_module(), const_module()] + emitter.stubs() + [emitter.fabric(top)]
    )


def boundary_plan(graph):
    """What a driver must feed each edge channel, and where its output belongs.

    Each boundary channel is one edge stream. For an inbound family the plan
    gives the tensor indices that channel must be handed, in order; for an
    outbound one, where each arriving token goes. The order comes from the same
    ``binding.imap.eval`` the reference simulator uses, so a testbench built from
    this drives the array the way the design says rather than the way its author
    guessed.

    Returns ``{family: {"direction", "tensor", "channels": [[idx, ...], ...]}}``.
    """
    emitter = StructuralEmitter(graph)
    low = emitter.rolled.low
    plan = {}
    for placement in emitter.placements():
        for fam in emitter.boundary_families(placement):
            mover = _mover_for(low, placement, fam)
            if mover is None:
                continue
            channels = [None] * _volume(fam.shape)
            for (site, _port), slot in fam.slots.items():
                channels[slot] = _mover_indices(mover, site)
            plan[fam.name] = {
                "direction": emitter.boundary_direction(fam),
                "tensor": getattr(mover.tensor, "base", mover.tensor).name,
                "channels": channels,
            }
        for port in placement.iface.ports():
            if port.protocol != MEMORY:
                continue
            entry = _memory_plan(low, placement, port)
            if entry is not None:
                fam = emitter.memory_family(placement, port)
                channels = [None] * _volume(fam.shape)
                for (site, _p), slot in fam.slots.items():
                    channels[slot] = [entry[1](site)]
                plan[fam.name] = {
                    "direction": port.direction,
                    "tensor": entry[0],
                    "channels": channels,
                }
    return plan


def _mover_for(low, placement, fam):
    """The loader or drain whose channels this boundary family carries."""
    for mover in low.movers:
        if mover.bundle.placement is not placement:
            continue
        if any((site, mover.bundle.port) in fam.slots for site in mover.bundle.sites):
            return mover
    return None


def _mover_indices(mover, site):
    """The tensor indices this site's channel carries, in transfer order.

    Shifted into the *base* tensor.  A mover may address a view -- a tile of a
    placed fabric reads its own slice of the caller's array -- and its index map
    is written in the view's coordinates.  ``lower_df`` applies the same shift
    when it emits the loader; without it here every tile claims the first tile's
    rows, and the driver feeds the array data that looks plausible and is wrong.
    """
    env = dict(mover.bundle.placement.env(site), __coords__=site)
    extent = mover.extent or 1
    return [
        _shift(mover.imap.eval(env, step=step), mover.tensor)
        for step in range(int(extent))
    ]


def _shift(subs, tensor):
    """View coordinates into base coordinates."""
    offsets = getattr(tensor, "offsets", None) or ()
    return tuple(
        int(v) + int(offsets[k] if k < len(offsets) else 0) for k, v in enumerate(subs)
    )


def _memory_plan(low, placement, port):
    """The tensor a memory port lands in, and where each site's value goes."""
    binding = low.mem_reads.get((placement, port)) or low.mem_writes.get(
        (placement, port)
    )
    if binding is None:
        return None
    source = low.resolve_storage(
        binding.source if binding.kind in {"shard", "stationary"} else binding.target
    )
    base = getattr(source, "base", None)
    if base is None or not hasattr(source, "name"):
        return None  # a brick is resident, not a tensor the driver touches

    def where(site):
        offsets = getattr(source, "offsets", None) or ()
        # A bare binding carries a SliceMap rather than an index expression; for
        # a scalar port every slice is one element, so its start *is* the index.
        # The reference simulator makes the same split.
        if hasattr(binding.imap, "slice_for"):
            subs = [start for start, _size in binding.imap.slice_for(site)]
        else:
            env = dict(placement.env(site), __coords__=site)
            subs = list(binding.imap.eval(env))
        return tuple(
            int(v) + int(offsets[k] if k < len(offsets) else 0)
            for k, v in enumerate(subs)
        )

    return base.name, where


def check_netlist(graph):
    """Verify the emitted wiring against the topology it came from.

    Every peer link in the topology must land both of its ends on one channel:
    the writer's port and each reader's port must compute the *same* index into
    the *same* family.  This is the property a mis-wire breaks silently, so it is
    checked against ``topology.channels`` -- the ground truth the emitter never
    consults -- rather than against the emitter's own tables.

    Returns the number of links checked.
    """
    emitter = StructuralEmitter(graph)
    checked = 0
    for placement in emitter.placements():
        for chan in placement.topology.channels.values():
            if chan.writer is None or not chan.readers:
                continue
            wsite, wport = chan.writer
            wfam = emitter.family_at(placement, wsite, wport)
            widx = emitter.channel_index(placement, wsite, wport, wfam)
            for rsite, rport in chan.readers:
                rfam = emitter.family_at(placement, rsite, rport)
                ridx = emitter.channel_index(placement, rsite, rport, rfam)
                if rfam.name != wfam.name or ridx != widx:
                    raise SPMWBindingError(
                        f"link {wsite}.{wport.name} -> {rsite}.{rport.name} is "
                        f"mis-wired: the writer reaches {wfam.name}[{widx}] but "
                        f"the reader reaches {rfam.name}[{ridx}]."
                    )
            checked += 1
    check_no_dangling_family(graph)
    return checked


def check_no_dangling_family(graph):
    """Every declared channel must be reached by some site.

    A family the fabric declares but no instance connects to is legal Verilog --
    an unused wire -- and elaborates without complaint. It is also a design that
    silently does nothing: the loaders were declared this way and never reached
    the array, so A and B never entered it.

    Returns the number of channels that carry a connection.
    """
    emitter = StructuralEmitter(graph)
    reached = set()
    for placement in emitter.placements():
        for site in placement.sites():
            for port, fam in emitter.site_ports(placement, site):
                idx = emitter.channel_index(placement, site, port, fam)
                if idx >= 0:
                    reached.add((fam.name, idx))
    internal, boundary = emitter.families()
    for fam in internal + boundary:
        touched = sum(1 for name, _i in reached if name == fam.name)
        if touched == 0:
            raise SPMWBindingError(
                f"family `{fam.name}` declares {_volume(fam.shape)} channel(s) "
                f"that no site connects to, so nothing it carries reaches the "
                f"array."
            )
    return len(reached)


def cost(graph):
    """What this design costs to build: HLS runs, and what they are reused over.

    The point of the split, as two numbers: ``roles`` syntheses are paid once,
    and ``instances`` come out of elaboration.
    """
    emitter = StructuralEmitter(graph)
    internal, boundary = emitter.families()
    return {
        "roles": sum(len(emitter.role_names(p)) for p in emitter.placements()),
        "instances": sum(_volume(p.grid) for p in emitter.placements()),
        "fifos": sum(_volume(f.shape) for f in internal),
        "streams": sum(_volume(f.shape) for f in boundary),
    }


__all__ = [
    "CoordPort",
    "StructuralEmitter",
    "const_module",
    "check_netlist",
    "check_no_dangling_family",
    "cost",
    "emit_structural_verilog",
    "fifo_module",
]

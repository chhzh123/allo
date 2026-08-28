# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The interface a synthesised unit presents, and the wires that carry it.

One place for what the fabric and the unit must agree on: how wide a token is,
which signals one port has, and the two primitives the array is built from.
Both emitters read it, and neither owns it -- keeping it here is what stops the
agreement being made twice and drifting.

The handshake is Vitis HLS's ``ap_fifo``: write side ``din``/``full_n``/
``write``, read side ``dout``/``empty_n``/``read``.  A free-running
(``ap_ctrl_none``) IP presents exactly that for an ``hls::stream`` argument, so
an exported role drops straight onto a FIFO.
"""

from .errors import SPMWBindingError
from .ports import IN, STREAM

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


# One AXI4 master's ports, in the order Vitis HLS declares them, as
# ``(suffix, direction, width)`` from the *master's* point of view. Written down
# rather than read back: a wrapper is emitted before synthesis has run, so the
# port list has to be known in advance. `check_wrapper` compares it against the
# exported IP afterwards, which is what keeps the table honest.
AXI_ADDR_WIDTH = 64
AXI_ID_WIDTH = 1
AXI_USER_WIDTH = 1


def axi_signals(bundle, data, addr=AXI_ADDR_WIDTH):
    """The ports an ``m_axi`` argument becomes on the synthesised IP."""
    idw, usr = AXI_ID_WIDTH, AXI_USER_WIDTH
    table = []
    for kind in ("AW", "AR"):
        table += [
            (f"{kind}VALID", "output", 1),
            (f"{kind}READY", "input", 1),
            (f"{kind}ADDR", "output", addr),
            (f"{kind}ID", "output", idw),
            (f"{kind}LEN", "output", 8),
            (f"{kind}SIZE", "output", 3),
            (f"{kind}BURST", "output", 2),
            (f"{kind}LOCK", "output", 2),
            (f"{kind}CACHE", "output", 4),
            (f"{kind}PROT", "output", 3),
            (f"{kind}QOS", "output", 4),
            (f"{kind}REGION", "output", 4),
            (f"{kind}USER", "output", usr),
        ]
        if kind == "AW":
            table += [
                ("WVALID", "output", 1),
                ("WREADY", "input", 1),
                ("WDATA", "output", data),
                ("WSTRB", "output", max(1, data // 8)),
                ("WLAST", "output", 1),
                ("WID", "output", idw),
                ("WUSER", "output", usr),
            ]
    table += [
        ("RVALID", "input", 1),
        ("RREADY", "output", 1),
        ("RDATA", "input", data),
        ("RLAST", "input", 1),
        ("RID", "input", idw),
        ("RUSER", "input", usr),
        ("RRESP", "input", 2),
        ("BVALID", "input", 1),
        ("BREADY", "output", 1),
        ("BRESP", "input", 2),
        ("BID", "input", idw),
        ("BUSER", "input", usr),
    ]
    return [(f"m_axi_{bundle}_{s}", d, w) for s, d, w in table]


def _decl(signals, indent="  "):
    out = []
    for sig, kind, width in signals:
        span = f"[{width - 1}:0] " if width > 1 else "               "
        pad = "input  " if kind == "input" else "output "
        out.append(f"{indent}{pad}wire {span}{sig}")
    return out


__all__ = [
    "CoordPort",
    "axi_signals",
    "const_module",
    "fifo_module",
]

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The array as something XRT can load: an RTL kernel around the fabric.

The role path -- one HLS project per role, a SystemVerilog fabric, Vivado
assembling them -- produces the better array: at 16x16 it is half the LUTs of
the whole-array route and closes at a clock the whole-array route could not
reach.  What it could not do was run on a card, because a card does not load a
netlist; it loads an `.xclbin`, and an `.xclbin` wants a kernel with a control
register map and AXI masters.

This module is that wrapper.  It puts three things beside the fabric:

* **one AXI4-Lite slave** carrying the standard Vitis control map, so XRT can
  set arguments and start the kernel;
* **one HLS feeder per boundary family** -- six, not the 49 a per-binding mover
  plan would need -- each an AXI master walking one family's tokens; and
* **a FIFO per boundary channel**, because a feeder presents the write side of
  a handshake and the fabric presents the read side.

The register map is the ordinary one, so no host-side special pleading:

    0x00  control     bit 0 start, 1 done, 2 idle, 3 ready, 7 auto-restart
    0x04  GIER
    0x08  IP_IER
    0x0C  IP_ISR
    0x10  first argument, one 8-byte slot each

Arguments are the six DRAM pointers, then the six token counts.  The counts
are arguments for the same reason the array reads its trip count out of the
instruction stream: a shell that walked a compile-time number of tokens would
put the shape back into the bitstream, and the shape is the thing a new model
changes.
"""

from .abi import AXI_ADDR_WIDTH as AXI_ADDR, axi_signals
from .rtl import StructuralEmitter, _volume, _width
from .shell import _dma_name, families

# The first argument sits above the four control words, and each argument gets
# a 64-bit slot whatever it holds -- what `v++` expects of a kernel it did not
# compile itself.
ARG_BASE = 0x10
ARG_STRIDE = 0x08


class Argument:
    """One entry of the control map: what it is called, how wide, and where."""

    __slots__ = ("bits", "family", "name", "offset", "pointer")

    def __init__(self, name, bits, offset, pointer, family):
        self.name = name
        self.bits = bits
        self.offset = offset
        self.pointer = pointer
        self.family = family

    def __repr__(self):
        kind = "ptr" if self.pointer else "scalar"
        return f"Argument({self.name}, {kind}, {self.bits}b, @{self.offset:#04x})"


#: The most tokens an edge FIFO buffers between a feeder and the array.
EDGE_DEPTH = 1024


def arguments(graph):
    """The kernel's arguments: a pointer per family, then a count per family.

    Pointers first so that a host walking the list can bind buffers before it
    knows any shape, which is the order `spmw_board_run.py` wants.
    """
    fams = families(graph)
    args = []
    for index, fam in enumerate(fams):
        args.append(
            Argument(
                f"{_dma_name(fam)}_ptr",
                64,
                ARG_BASE + ARG_STRIDE * index,
                True,
                fam,
            )
        )
    base = ARG_BASE + ARG_STRIDE * len(fams)
    for index, fam in enumerate(fams):
        args.append(
            Argument(
                f"{_dma_name(fam)}_steps",
                32,
                base + ARG_STRIDE * index,
                False,
                fam,
            )
        )
    return args


def _addr_width(args):
    """Enough address bits to reach the last argument's high word."""
    top = ARG_BASE + ARG_STRIDE * len(args) + 8
    width = 1
    while (1 << width) < top:
        width += 1
    return width


def control_sv(args, name="spmw_control_s_axi"):
    """The AXI4-Lite slave: the control word, and a register per argument.

    Written out rather than borrowed from an HLS project because the arguments
    are this design's, not one function's, and because the behaviour that
    matters is small and worth being able to read: `ap_start` is set by a write
    and cleared when the kernel takes it, `ap_done` latches until read, and
    everything else is a register file.
    """
    aw = _addr_width(args)
    decls, resets, writes, reads, outs = [], [], [], [], []
    for arg in args:
        words = 2 if arg.bits > 32 else 1
        decls.append(f"  reg [{arg.bits - 1}:0] r_{arg.name};")
        resets.append(f"      r_{arg.name} <= 0;")
        outs.append(f"  assign {arg.name} = r_{arg.name};")
        for word in range(words):
            addr = arg.offset + 4 * word
            lo, hi = 32 * word, min(32 * word + 31, arg.bits - 1)
            span = f"[{hi}:{lo}]" if arg.bits > 32 else ""
            writes.append(
                f"        {aw}'h{addr:02x}: r_{arg.name}{span} <= "
                f"w_data[{hi - lo}:0];"
            )
            reads.append(f"        {aw}'h{addr:02x}: rdata <= r_{arg.name}{span};")
    ports = "".join(f"  output wire [{a.bits - 1}:0] {a.name},\n" for a in args)
    return f"""`timescale 1ns/1ps
// Generated by allo.spmw.kernel.control_sv -- do not edit.
//
// The Vitis control map. `ap_start` is a write-one that the kernel clears when
// it accepts the job; `ap_done` latches so a host that polls slower than the
// kernel runs still sees it, and clears on read.
module {name} #(
  parameter AW = {aw},
  parameter DW = 32
)(
  input  wire          ACLK,
  input  wire          ARESET,
  input  wire [AW-1:0] AWADDR,
  input  wire          AWVALID,
  output wire          AWREADY,
  input  wire [DW-1:0] WDATA,
  input  wire [DW/8-1:0] WSTRB,
  input  wire          WVALID,
  output wire          WREADY,
  output wire [1:0]    BRESP,
  output wire          BVALID,
  input  wire          BREADY,
  input  wire [AW-1:0] ARADDR,
  input  wire          ARVALID,
  output wire          ARREADY,
  output wire [DW-1:0] RDATA,
  output wire [1:0]    RRESP,
  output wire          RVALID,
  input  wire          RREADY,
  output wire          interrupt,
{ports}  output wire          ap_start,
  input  wire          ap_done,
  input  wire          ap_idle,
  input  wire          ap_ready
);
  localparam ADDR_CTRL = {aw}'h00, ADDR_GIER = {aw}'h04,
             ADDR_IER  = {aw}'h08, ADDR_ISR  = {aw}'h0c;

  reg        r_valid, b_valid;
  reg [DW-1:0] rdata;
  reg        r_start, r_done, r_auto;
  reg        gier;
  reg [1:0]  ier, isr;
{chr(10).join(decls)}

  wire [DW-1:0] w_data = WDATA;
  // A write is taken only when the previous response has been collected;
  // otherwise a back-to-back write overwrites BVALID and the host waits for a
  // response that was already spent.
  wire          w_fire = AWVALID && WVALID && !b_valid;
  wire          r_fire = ARVALID && !r_valid;
  wire          ctrl_read = r_fire && (ARADDR == ADDR_CTRL);
  wire          ctrl_write = w_fire && (AWADDR == ADDR_CTRL);

  assign AWREADY = w_fire;
  assign WREADY  = w_fire;
  assign BRESP   = 2'b00;
  assign BVALID  = b_valid;
  assign ARREADY = r_fire;
  assign RDATA   = rdata;
  assign RRESP   = 2'b00;
  assign RVALID  = r_valid;
  assign ap_start  = r_start;
  assign interrupt = gier && |(isr & ier);
{chr(10).join(outs)}

  always @(posedge ACLK) begin
    if (ARESET) begin
      b_valid <= 0; r_valid <= 0;
      r_start <= 0; r_done <= 0; r_auto <= 0;
      gier <= 0; ier <= 0; isr <= 0; rdata <= 0;
{chr(10).join(resets)}
    end else begin
      // -- write channel: address and data together, which is all a control
      // map needs.
      if (w_fire) begin
        b_valid <= 1;
        case (AWADDR)
          ADDR_CTRL: r_auto <= w_data[7];
          ADDR_GIER: gier <= w_data[0];
          ADDR_IER:  ier  <= w_data[1:0];
          ADDR_ISR:  isr  <= isr & ~w_data[1:0];
{chr(10).join(writes)}
          default: ;
        endcase
      end else if (b_valid && BREADY) begin
        b_valid <= 0;
      end

      // -- start. Written by the host, spent when the kernel accepts it. The
      // write wins a tie, or a start issued in the cycle the previous job was
      // accepted would be dropped.
      if (ctrl_write && w_data[0]) r_start <= 1'b1;
      else if (ap_ready)           r_start <= r_auto;

      // -- done. Latched, and *set wins over clear*: the kernel's `ap_done` is
      // one cycle wide, and a host polling the control word will eventually
      // read it in that very cycle. Clearing then would lose the completion
      // for good, and the host would poll a finished kernel for ever.
      if (ap_done)        r_done <= 1'b1;
      else if (ctrl_read) r_done <= 1'b0;
      if (ap_done) isr[0] <= isr[0] | ier[0];

      // -- read channel.
      if (r_fire) begin
        r_valid <= 1;
        case (ARADDR)
          ADDR_CTRL:
            rdata <= {{24'd0, r_auto, 3'd0, ap_ready,
                      ap_idle, r_done | ap_done, r_start}};
          ADDR_GIER: rdata <= {{31'd0, gier}};
          ADDR_IER:  rdata <= {{30'd0, ier}};
          ADDR_ISR:  rdata <= {{30'd0, isr}};
{chr(10).join(reads)}
          default:   rdata <= 0;
        endcase
      end else if (r_valid && RREADY) begin
        r_valid <= 0;
      end
    end
  end
endmodule
"""


def feeder_tcl(fam, part, period):
    """The HLS run for one family's DMA.

    ``-offset direct`` puts the address on a port instead of in a register map
    of its own: this feeder is not the kernel, it is one master inside it, and
    the kernel's own slave supplies the pointer.
    """
    beats = 1 if fam["ctype"] is not None else -(-fam["width"] // 512)
    total = fam["channels"] * fam["steps"] * beats
    return f"""open_project prj
set_top {_dma_name(fam)}
add_files kernel.cpp
open_solution sol
set_part {part}
create_clock -period {period:.3f} -name default
config_interface -clock_enable=0 -m_axi_max_widen_bitwidth 512
set_directive_interface -mode m_axi -offset direct -depth {total} \
"{_dma_name(fam)}" {'src' if fam['reads'] else 'dst'}
csynth_design
export_design -format ip_catalog
exit
"""


def kernel_sv(graph, args, widths, top="spmw_kernel", fabric="spmw_top"):
    """The kernel: control slave, one feeder per family, and the fabric.

    A feeder presents the write side of a handshake and the fabric the read
    side (or the reverse, for the drain), so every boundary channel gets a FIFO
    between them -- the same `spmw_fifo` the fabric uses internally, so there is
    one handshake in the design rather than two that have to agree.
    """
    emitter = StructuralEmitter(graph)
    _internal, boundary = emitter.families()
    by_name = {fam.name: fam for fam in boundary}
    fams = families(graph)

    ports = [
        "  input  wire ap_clk",
        "  input  wire ap_rst_n",
    ]
    for index, fam in enumerate(fams):
        for sig, kind, width in axi_signals(f"gmem{index}", _axi_width(widths, fam)):
            span = f"[{width - 1}:0] " if width > 1 else ""
            ports.append(
                f"  {'input ' if kind == 'input' else 'output'} wire {span}{sig}"
            )
    aw = _addr_width(args)
    ports += [
        f"  input  wire [{aw - 1}:0] s_axi_control_AWADDR",
        "  input  wire s_axi_control_AWVALID",
        "  output wire s_axi_control_AWREADY",
        "  input  wire [31:0] s_axi_control_WDATA",
        "  input  wire [3:0] s_axi_control_WSTRB",
        "  input  wire s_axi_control_WVALID",
        "  output wire s_axi_control_WREADY",
        "  output wire [1:0] s_axi_control_BRESP",
        "  output wire s_axi_control_BVALID",
        "  input  wire s_axi_control_BREADY",
        f"  input  wire [{aw - 1}:0] s_axi_control_ARADDR",
        "  input  wire s_axi_control_ARVALID",
        "  output wire s_axi_control_ARREADY",
        "  output wire [31:0] s_axi_control_RDATA",
        "  output wire [1:0] s_axi_control_RRESP",
        "  output wire s_axi_control_RVALID",
        "  input  wire s_axi_control_RREADY",
        "  output wire interrupt",
    ]

    body = ["  wire ap_start, ap_done, ap_idle, ap_ready;", "  wire rst = ~ap_rst_n;"]
    for arg in args:
        body.append(f"  wire [{arg.bits - 1}:0] {arg.name};")
    body.append(f"  reg [{len(fams) - 1}:0] fin, pend;")
    body += _control_instance(args)

    # One FIFO per boundary channel, and the wires either side of it.
    for fam in fams:
        body += _edge_wires(by_name[fam["name"]], fam)
    body += _feeder_instances(fams, by_name, widths)
    body += _fabric_instance(fabric, by_name, fams)
    body += _completion(fams)

    return (
        "`timescale 1ns/1ps\n"
        "// Generated by allo.spmw.kernel.kernel_sv -- do not edit.\n\n"
        + f"module {top} (\n"
        + ",\n".join(ports)
        + "\n);\n"
        + "\n".join(body)
        + "\nendmodule\n"
    )


def _axi_width(widths, fam):
    """The master's data width, as built.

    Not as requested: `-m_axi_max_widen_bitwidth 512` is a ceiling, and a feeder
    whose index is computed (`src[t * channels + c]`) does not burst, so Vitis
    builds a 32-bit master however wide the ask.  Declaring the ceiling on the
    kernel's ports would leave the top wider than the IP inside it.
    """
    return widths[_dma_name(fam)]


def _control_instance(args):
    binds = "".join(f"    .{a.name}({a.name}),\n" for a in args)
    return [
        "  spmw_control_s_axi u_control (",
        "    .ACLK(ap_clk), .ARESET(rst),",
        "    .AWADDR(s_axi_control_AWADDR), .AWVALID(s_axi_control_AWVALID),",
        "    .AWREADY(s_axi_control_AWREADY),",
        "    .WDATA(s_axi_control_WDATA), .WSTRB(s_axi_control_WSTRB),",
        "    .WVALID(s_axi_control_WVALID), .WREADY(s_axi_control_WREADY),",
        "    .BRESP(s_axi_control_BRESP), .BVALID(s_axi_control_BVALID),",
        "    .BREADY(s_axi_control_BREADY),",
        "    .ARADDR(s_axi_control_ARADDR), .ARVALID(s_axi_control_ARVALID),",
        "    .ARREADY(s_axi_control_ARREADY),",
        "    .RDATA(s_axi_control_RDATA), .RRESP(s_axi_control_RRESP),",
        "    .RVALID(s_axi_control_RVALID), .RREADY(s_axi_control_RREADY),",
        "    .interrupt(interrupt),",
        binds.rstrip(",\n") + ",",
        "    .ap_start(ap_start), .ap_done(ap_done),",
        "    .ap_idle(ap_idle), .ap_ready(ap_ready));",
        "",
    ]


def _edge_wires(fam, plan):
    """The FIFO between one family's feeder and the fabric, per channel.

    Deep enough to hold the family's whole pass, which is not an optimisation
    but the thing that makes the shell work at all.

    A feeder is one sequential loop: it writes every channel's step-t token
    before any channel's step t+1. A systolic array does not consume them that
    way -- the last row of a 16x16 mesh is fifteen steps behind the first,
    because its partial sums have to come down through fourteen cells first. So
    the last channel's FIFO fills while the first channel is still being asked
    for more, the feeder blocks on the full one, and the first row starves
    waiting for a token the feeder cannot deliver until the last row moves --
    which it cannot, because it is waiting on the first. The array deadlocks
    with every FIFO empty and nothing to show for it.

    A per-channel driver would also fix it, and that is what the cosim
    testbench happens to be, which is why the fabric passed 11/11 and the
    kernel hung. Sizing the buffer is the cheaper answer: the depth is the
    family's own step count -- what the design was elaborated with -- so the
    feeder can lay down a whole pass and let the array take it in its own
    order.
    """
    count = _volume(fam.shape)
    width = _width(fam)
    span = f"[{width - 1}:0] "
    name = fam.name
    # "One whole pass" was the right size when a pass was 32 steps. A stage
    # engine's pass is 32,768, and a 32,768-deep FIFO on each of sixteen
    # activation channels is half a megabyte of block RAM for a buffer that
    # only ever needs to cover the array's skew -- the last row is sixteen
    # steps behind the first, not thirty thousand. So the depth is the pass or
    # EDGE_DEPTH, whichever is smaller; a feeder that outruns it blocks on a
    # full FIFO and resumes, which is a stall rather than the deadlock, because
    # every row's FIFO is now deep enough for every other row to catch up.
    depth = min(max(2, plan["steps"]), EDGE_DEPTH)
    lines = [
        f"  // {name}: {count} channel(s) of {width} bits between the "
        f"{'feeder' if plan['reads'] else 'drain'} and the array, "
        f"{depth} deep -- one pass, so the feeder never blocks"
    ]
    for sig in ("din", "dout"):
        lines.append(f"  wire {span}{name}_e_{sig} [0:{count - 1}];")
    for sig in ("full_n", "write", "empty_n", "read"):
        lines.append(f"  wire {name}_e_{sig} [0:{count - 1}];")
    lines += [
        f"  genvar {name}_i;",
        "  generate",
        f"    for ({name}_i = 0; {name}_i < {count}; {name}_i = {name}_i + 1)"
        f" begin : g_edge_{name}",
        f"      spmw_fifo #(.DW({width}), .DEPTH({depth})) u ("
        ".clk(ap_clk), .rst_n(ap_rst_n)"
        + "".join(
            f", .{s}({name}_e_{s}[{name}_i])"
            for s in ("din", "full_n", "write", "dout", "empty_n", "read")
        )
        + ");",
        "    end",
        "  endgenerate",
    ]
    return lines


def _feeder_instances(fams, by_name, widths):
    """One HLS DMA per family, started together and reporting done together."""
    lines = []
    for index, plan in enumerate(fams):
        fam = by_name[plan["name"]]
        name = _dma_name(plan)
        count = _volume(fam.shape)
        conns = [
            ".ap_clk(ap_clk)",
            ".ap_rst_n(ap_rst_n)",
            f".ap_start(start_{index})",
            f".ap_done(done_{index})",
            f".ap_idle(idle_{index})",
            f".ap_ready(ready_{index})",
            f".{'src' if plan['reads'] else 'dst'}({name}_ptr)",
            f".steps({name}_steps)",
        ]
        # The feeder owns the side of the edge FIFO the fabric does not: it
        # writes what the array will read, and reads what the array wrote.
        for channel in range(count):
            # Vitis names a one-element array of streams `out_r`, not `out_0`:
            # a single stream is not an array to it. Every other size counts
            # from zero as written.
            stem = "out" if plan["reads"] else "in"
            port = f"{stem}_r" if count == 1 else f"{stem}_{channel}"
            sides = (
                ("din", "full_n", "write")
                if plan["reads"]
                else ("dout", "empty_n", "read")
            )
            conns += [f".{port}_{sig}({fam.name}_e_{sig}[{channel}])" for sig in sides]
        for sig, _kind, _w in axi_signals(f"gmem{index}", _axi_width(widths, plan)):
            conns.append(f".m_axi_gmem{sig[len(f'm_axi_gmem{index}'):]}({sig})")
        lines += [
            f"  wire done_{index}, idle_{index}, ready_{index};",
            f"  wire start_{index} = pend[{index}];",
            f"  {name} u_{name} (\n      " + ",\n      ".join(conns) + ");",
            "",
        ]
    return lines


def _fabric_instance(fabric, by_name, fams):
    """The array, taking whichever side of each edge FIFO the feeder left.

    The fabric's boundary ports are whole unpacked arrays, one per family, so
    they connect as arrays -- there is nothing per-channel to do here.
    """
    conns = [".ap_clk(ap_clk)", ".ap_rst_n(ap_rst_n)"]
    for plan in fams:
        fam = by_name[plan["name"]]
        sides = (
            ("dout", "empty_n", "read") if plan["reads"] else ("din", "write", "full_n")
        )
        conns += [f".{fam.name}_{sig}({fam.name}_e_{sig})" for sig in sides]
    return [f"  {fabric} dut (\n      " + ",\n      ".join(conns) + ");", ""]


def _completion(fams):
    """The kernel is done when every feeder is, and idle when none has started.

    Latched, because the feeders finish at different times and their `ap_done`
    pulses never coincide -- ANDing them directly is a kernel that never
    finishes, which is a thing this design has already been caught doing once.
    """
    n = len(fams)
    lines = [
        "  // Exactly one job per feeder per invocation.",
        "  //",
        "  // Holding `ap_start` high until the slowest feeder is done makes an",
        "  // `ap_ctrl_hs` IP take the job again the moment it frees up, and",
        "  // gating on `ap_done` does not help -- the IP accepts on `ap_ready`,",
        "  // which it asserts in the same cycle. A feeder that runs twice",
        "  // writes its family twice, and the array, fed tokens it never asked",
        "  // for, stalls with nothing to say about why. So the start is a",
        "  // one-shot: raised when the kernel is kicked, dropped the moment",
        "  // that feeder accepts.",
        "  reg ap_start_d;",
        "  wire kick = ap_start & ~ap_start_d;",
        "  always @(posedge ap_clk) ap_start_d <= ap_rst_n ? ap_start : 1'b0;",
        "  always @(posedge ap_clk) begin",
        "    if (rst) begin",
        "      fin <= 0; pend <= 0;",
        "    end else if (ap_ready) begin",
        "      fin <= 0; pend <= 0;",
        "    end else begin",
    ]
    for index in range(n):
        lines.append(f"      if (kick) pend[{index}] <= 1'b1;")
        lines.append(f"      else if (ready_{index}) pend[{index}] <= 1'b0;")
        lines.append(f"      if (done_{index}) fin[{index}] <= 1'b1;")
    lines += [
        "    end",
        "  end",
        "  assign ap_done  = &fin;",
        "  assign ap_ready = ap_done;",
        "  assign ap_idle  = " + " & ".join(f"idle_{i}" for i in range(n)) + ";",
    ]
    return lines


def kernel_xml(args, widths, name="spmw_kernel"):
    """What `package_xo` needs in order to believe this is a kernel."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<root versionMajor="1" versionMinor="6">',
        f'  <kernel name="{name}" language="ip_c" vlnv="allo:spmw:{name}:1.0"'
        f' attributes="" preferredWorkGroupSizeMultiple="0" workGroupSize="1"'
        f' interrupt="true">',
        "    <ports>",
        '      <port name="s_axi_control" mode="slave" range="0x1000"'
        ' dataWidth="32" portType="addressable" base="0x0"/>',
    ]
    for index, arg in enumerate(a for a in args if a.pointer):
        lines.append(
            f'      <port name="m_axi_gmem{index}" mode="master"'
            f' range="0xFFFFFFFFFFFFFFFF" dataWidth="{widths[arg.family and _dma_name(arg.family)]}"'
            f' portType="addressable" base="0x0"/>'
        )
    lines += ["    </ports>", "    <args>"]
    pointer_index = 0
    for index, arg in enumerate(args):
        if arg.pointer:
            lines.append(
                f'      <arg name="{arg.name}" addressQualifier="1" id="{index}"'
                f' port="m_axi_gmem{pointer_index}" size="0x8"'
                f' offset="{arg.offset:#06x}" type="int*" hostOffset="0x0"'
                f' hostSize="0x8"/>'
            )
            pointer_index += 1
        else:
            lines.append(
                f'      <arg name="{arg.name}" addressQualifier="0" id="{index}"'
                f' port="s_axi_control" size="0x4" offset="{arg.offset:#06x}"'
                f' type="int" hostOffset="0x0" hostSize="0x4"/>'
            )
    lines += ["    </args>", "  </kernel>", "</root>"]
    return "\n".join(lines) + "\n"


def kernel_testbench(
    graph, args, widths, operands, expected, counts=None, top="spmw_kernel"
):
    """The kernel against behavioural DRAM, so a stall can be seen.

    A hang on the card is invisible: pyxrt exposes no way to read the control
    map, so "the kernel did not finish" is the entire diagnosis available. Here
    every master has its own `spmw_axi_ram`, the control map is driven by a
    small AXI4-Lite writer, and when it stalls the waveform says which feeder
    still has work left.

    ``operands`` maps a family's DMA name to the bytes its master should hold;
    ``expected`` is what the drain's memory must contain afterwards; ``counts``
    gives each family's step count, defaulting to what the design was
    elaborated with. It has to be settable: a smaller tile is a different set
    of counts against the same netlist, and a bench that always wrote the
    elaborated ones would over-supply -- which the array survives, because its
    shape comes from the instruction stream, but the feeders then never finish.
    """
    counts = counts or {}
    fams = families(graph)
    aw = _addr_width(args)
    lines = [
        "`timescale 1ns/1ps",
        "// Generated by allo.spmw.kernel.kernel_testbench -- do not edit.",
        "",
        "module tb;",
        "  reg ap_clk = 0, ap_rst_n = 0;",
        "  always #2 ap_clk = ~ap_clk;   // 250 MHz",
        "",
        f"  reg [{aw - 1}:0] awaddr; reg awvalid; wire awready;",
        "  reg [31:0] wdata; reg wvalid; wire wready;",
        "  wire bvalid; reg bready;",
        f"  reg [{aw - 1}:0] araddr; reg arvalid; wire arready;",
        "  wire [31:0] rdata; wire rvalid; reg rready;",
        "  wire interrupt;",
        "",
    ]
    for index, fam in enumerate(fams):
        width = _axi_width(widths, fam)
        nbytes = len(operands[_dma_name(fam)])
        for sig, _kind, sig_width in axi_signals(f"gmem{index}", width):
            span = f"[{sig_width - 1}:0] " if sig_width > 1 else ""
            lines.append(f"  wire {span}{sig};")
        lines += [
            f"  spmw_axi_ram #(.DW({width}), .AW({AXI_ADDR}), "
            f".BYTES({max(nbytes, 64)})) u_ram{index} (",
            "    .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),",
            *[
                f"    .{short}(m_axi_gmem{index}_{short}),"
                for short in (
                    "ARVALID",
                    "ARREADY",
                    "ARADDR",
                    "ARLEN",
                    "RVALID",
                    "RREADY",
                    "RDATA",
                    "RLAST",
                    "AWVALID",
                    "AWREADY",
                    "AWADDR",
                    "AWLEN",
                    "WVALID",
                    "WREADY",
                    "WDATA",
                    "WSTRB",
                    "WLAST",
                    "BVALID",
                )
            ],
            f"    .BREADY(m_axi_gmem{index}_BREADY));",
            "",
        ]
    conns = [".ap_clk(ap_clk)", ".ap_rst_n(ap_rst_n)"]
    for index, fam in enumerate(fams):
        for sig, _kind, _w in axi_signals(f"gmem{index}", _axi_width(widths, fam)):
            conns.append(f".{sig}({sig})")
    conns += [
        ".s_axi_control_AWADDR(awaddr)",
        ".s_axi_control_AWVALID(awvalid)",
        ".s_axi_control_AWREADY(awready)",
        ".s_axi_control_WDATA(wdata)",
        ".s_axi_control_WSTRB(4'hF)",
        ".s_axi_control_WVALID(wvalid)",
        ".s_axi_control_WREADY(wready)",
        ".s_axi_control_BRESP()",
        ".s_axi_control_BVALID(bvalid)",
        ".s_axi_control_BREADY(bready)",
        ".s_axi_control_ARADDR(araddr)",
        ".s_axi_control_ARVALID(arvalid)",
        ".s_axi_control_ARREADY(arready)",
        ".s_axi_control_RDATA(rdata)",
        ".s_axi_control_RRESP()",
        ".s_axi_control_RVALID(rvalid)",
        ".s_axi_control_RREADY(rready)",
        ".interrupt(interrupt)",
    ]
    lines += [
        f"  {top} dut (\n      " + ",\n      ".join(conns) + ");",
        "",
        "  integer i;",
        "  reg [31:0] status;",
        "",
        "  task wr(input [31:0] a, input [31:0] d);",
        "    begin",
        f"      @(negedge ap_clk); awaddr <= a[{aw - 1}:0]; wdata <= d;",
        "      awvalid <= 1; wvalid <= 1; bready <= 1;",
        "      @(posedge ap_clk);",
        "      while (!(awready && wready)) @(posedge ap_clk);",
        "      @(negedge ap_clk); awvalid <= 0; wvalid <= 0;",
        "      @(posedge ap_clk);",
        "    end",
        "  endtask",
        "",
        "  task rd(input [31:0] a, output [31:0] d);",
        "    begin",
        f"      @(negedge ap_clk); araddr <= a[{aw - 1}:0]; arvalid <= 1;"
        f" rready <= 1;",
        "      @(posedge ap_clk);",
        "      while (!arready) @(posedge ap_clk);",
        "      @(negedge ap_clk); arvalid <= 0;",
        "      while (!rvalid) @(posedge ap_clk);",
        "      d = rdata;",
        "      @(posedge ap_clk);",
        "    end",
        "  endtask",
        "",
        "  initial begin",
        "    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0;",
        "    awaddr = 0; wdata = 0; araddr = 0;",
        "    repeat (8) @(posedge ap_clk);",
        "    @(negedge ap_clk) ap_rst_n = 1;",
        "    repeat (8) @(posedge ap_clk);",
        "",
    ]
    # Preload every inbound master, and zero the drain's.
    for index, fam in enumerate(fams):
        raw = operands[_dma_name(fam)]
        if not fam["reads"]:
            continue
        lines.append(f"    // {_dma_name(fam)}: {len(raw)} bytes")
        for addr, byte in enumerate(raw):
            lines.append(f"    u_ram{index}.mem[{addr}] = 8'd{byte};")
    lines.append("")
    for arg in args:
        if arg.pointer:
            lines += [
                f"    wr(32'h{arg.offset:02x}, 32'd0);",
                f"    wr(32'h{arg.offset + 4:02x}, 32'd0);",
            ]
        else:
            fam = [f for f in fams if _dma_name(f) + "_steps" == arg.name][0]
            steps = counts.get(_dma_name(fam), fam["steps"])
            lines.append(f"    wr(32'h{arg.offset:02x}, 32'd{steps});")
    lines += [
        "",
        '    $display("SPMW TB: starting");',
        "    wr(32'h00, 32'd1);",
        "    status = 0;",
        "    for (i = 0; i < 200000 && !status[1]; i = i + 1) begin",
        "      rd(32'h00, status);",
        "    end",
        "    if (!status[1]) begin",
        '      $display("SPMW TB TIMEOUT after %0d polls, ctrl=%h", i, status);',
    ]
    drain_index = [i for i, f in enumerate(fams) if not f["reads"]][0]
    lines += [
        "      $finish;",
        "    end",
        '    $display("SPMW TB: done after %0d poll(s)", i);',
        "    begin : compare",
        "      integer bad;",
        "      bad = 0;",
    ]
    for addr, byte in enumerate(expected):
        lines.append(
            f"      if (u_ram{drain_index}.mem[{addr}] !== 8'd{byte}) bad = bad + 1;"
        )
    lines += [
        '      $display("SPMW TB RESULT %0d of %0d byte(s) wrong",'
        f" bad, {len(expected)});",
        "    end",
        "    $finish;",
        "  end",
        "endmodule",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ARG_BASE",
    "ARG_STRIDE",
    "Argument",
    "arguments",
    "control_sv",
    "feeder_tcl",
    "kernel_sv",
    "kernel_testbench",
    "kernel_xml",
]

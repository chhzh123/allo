# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A memory for the memory-mapped fabric to talk to, and the bench that drives it.

:mod:`allo.spmw.cosim` drives the array's *edge streams*: it holds every operand
itself and hands them over token by token.  That measures the array and nothing
else, which is what it is for.  A fabric built with ``memory=True`` has no edge
streams -- its movers read and write DRAM over AXI -- so it needs something on
the other side of those masters.

This is that something: a behavioural AXI4 slave with a byte-addressed backing
array.  What it models and what it does not is worth being explicit about,
because the numbers it produces get compared against AutoSA's:

* **one memory per master, no contention.**  A U280 has 32 HBM channels, so a
  design with a handful of masters really can have them independent.  It stops
  being true well before 32: this model will not show a design running out of
  channels, it will only show the port count, which is reported separately.
* **a settable read latency**, applied once per burst rather than per beat, and
  **one beat per cycle** once data starts.  Real DDR is worse on both counts.
* **one transaction at a time per port** -- no outstanding-request pipelining,
  which the HLS masters are built to exploit.  This makes latency hurt more
  here than on hardware, so the latency sweep is a bound rather than a
  prediction.

The point of the model is comparison between designs under identical
assumptions, not absolute cycle counts.
"""

import numpy as np

from .ports import IN
from .rtl import StructuralEmitter, _volume, boundary_plan

_NUMPY = {
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}


def ram_module():
    """A behavioural AXI4 slave over a byte array.

    Deliberately simple: one outstanding transaction, a fixed pre-burst latency,
    then a beat a cycle.  It answers the question the fabric asks -- do the
    movers issue the right addresses and does the array get the right data -- and
    keeps the timing assumptions in one readable place.
    """
    return """`timescale 1ns/1ps

module spmw_axi_ram #(
  parameter integer DW = 32,
  parameter integer AW = 64,
  parameter integer BYTES = 65536,
  parameter integer LATENCY = 0
) (
  input  wire          ap_clk,
  input  wire          ap_rst_n,
  // read address
  input  wire          ARVALID,
  output reg           ARREADY,
  input  wire [AW-1:0] ARADDR,
  input  wire [7:0]    ARLEN,
  // read data
  output reg           RVALID,
  input  wire          RREADY,
  output reg  [DW-1:0] RDATA,
  output reg           RLAST,
  // write address
  input  wire          AWVALID,
  output reg           AWREADY,
  input  wire [AW-1:0] AWADDR,
  input  wire [7:0]    AWLEN,
  // write data
  input  wire          WVALID,
  output reg           WREADY,
  input  wire [DW-1:0] WDATA,
  input  wire [DW/8-1:0] WSTRB,
  input  wire          WLAST,
  // write response
  output reg           BVALID,
  input  wire          BREADY
);
  localparam integer STEP = DW / 8;
  reg [7:0] mem [0:BYTES-1];

  integer rbeats, rwait, b;
  reg [AW-1:0] raddr;
  integer wbeats;
  reg [AW-1:0] waddr;

  // -- read -----------------------------------------------------------------
  always @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      ARREADY <= 1'b1; RVALID <= 1'b0; RLAST <= 1'b0; rbeats <= 0; rwait <= 0;
    end else begin
      if (ARREADY && ARVALID) begin
        raddr   <= ARADDR;
        rbeats  <= ARLEN + 1;
        rwait   <= LATENCY;
        ARREADY <= 1'b0;
      end else if (rbeats > 0 && !RVALID) begin
        if (rwait > 0) begin
          rwait <= rwait - 1;
        end else begin
          for (b = 0; b < STEP; b = b + 1)
            RDATA[b*8 +: 8] <= mem[raddr + b];
          RVALID <= 1'b1;
          RLAST  <= (rbeats == 1);
        end
      end else if (RVALID && RREADY) begin
        RVALID <= 1'b0;
        raddr  <= raddr + STEP;
        rbeats <= rbeats - 1;
        if (rbeats == 1) begin
          RLAST   <= 1'b0;
          ARREADY <= 1'b1;
        end
      end
    end
  end

  // -- write ----------------------------------------------------------------
  always @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      AWREADY <= 1'b1; WREADY <= 1'b0; BVALID <= 1'b0; wbeats <= 0;
    end else begin
      if (AWREADY && AWVALID) begin
        waddr   <= AWADDR;
        wbeats  <= AWLEN + 1;
        AWREADY <= 1'b0;
        WREADY  <= 1'b1;
      end else if (WREADY && WVALID) begin
        for (b = 0; b < STEP; b = b + 1)
          if (WSTRB[b]) mem[waddr + b] <= WDATA[b*8 +: 8];
        waddr  <= waddr + STEP;
        wbeats <= wbeats - 1;
        if (WLAST || wbeats == 1) begin
          WREADY <= 1'b0;
          BVALID <= 1'b1;
        end
      end else if (BVALID && BREADY) begin
        BVALID  <= 1'b0;
        AWREADY <= 1'b1;
      end
    end
  end
endmodule
"""


class MemoryBench:
    """A testbench for a fabric that reaches DRAM itself.

    One :func:`ram_module` per AXI master, preloaded with whatever that master
    reads and checked afterwards for whatever it writes.  The array is started
    once and the run ends at ``ap_done``, so the cycle count covers the loads,
    the compute and the drain together -- which is the number AutoSA's cosim
    reports, and the reason this exists.
    """

    def __init__(self, graph, data, results, latency=0, axi_widths=None):
        self.graph = graph
        self.emitter = StructuralEmitter(graph)
        self.emitter.movers.widths = dict(axi_widths or {})
        self.data = data
        self.results = results
        self.latency = latency

    def masters(self):
        """Every AXI master: which mover it belongs to, and what it carries.

        ``touches`` is the elements this one instance actually moves.  It
        matters on the way out: each drain writes its own slice of the result,
        so checking a master's memory against the whole tensor finds every other
        instance's share still uninitialised.
        """
        plan = boundary_plan(self.graph)
        out = []
        movers = self.emitter.movers
        for index, mover in enumerate(movers):
            tensor = getattr(mover.tensor, "base", mover.tensor)
            inbound = self.emitter.boundary_direction(mover.family) == IN
            channels = plan.get(mover.family.name, {}).get("channels", [])
            for pos, _coords, _site, channel in movers.instances(index):
                out.append(
                    {
                        "name": f"{movers.name(index)}_{pos}",
                        "tensor": tensor.name,
                        "dtype": str(tensor.dtype),
                        "shape": tuple(int(e) for e in tensor.shape),
                        "reads": inbound,
                        "width": self.emitter.movers.width(index),
                        "touches": (
                            channels[channel] if channel < len(channels) else None
                        ),
                    }
                )
        return out

    def render(self, cycles=2000000, top="spmw_top"):
        """The testbench module."""
        masters = self.masters()
        lines = [
            "`timescale 1ns/1ps",
            "",
            "module tb;",
            "  reg clk = 0, rst_n = 0, start = 0;",
            "  always #5 clk = ~clk;",
            "  wire done;",
            "  integer errors = 0;",
            "  integer cycle = 0;",
        ]
        conns = [
            ".ap_clk(clk)",
            ".ap_rst_n(rst_n)",
            ".ap_start(start)",
            ".ap_done(done)",
        ]
        for master in masters:
            lines += self._ram(master, conns)
        lines.append(f"  {top} dut (" + ",\n      ".join(conns) + ");")
        lines += self._timing(masters)
        lines += [
            "  initial begin",
            "    repeat (4) @(posedge clk);",
        ]
        for master in masters:
            if master["reads"]:
                lines += self._preload(master)
        lines += [
            "    rst_n = 1;",
            "    @(posedge clk);",
            "    start = 1;",
            f"    for (cycle = 0; cycle < {cycles}; cycle = cycle + 1) begin",
            "      @(posedge clk);",
            "      if (done) begin",
        ]
        for master in masters:
            if not master["reads"]:
                lines += self._check(master)
        for master in masters:
            lines.append(
                f'        $display("SPMW MASTER {master["name"]} '
                f'{"read" if master["reads"] else "write"} done=%0d", '
                # The last master to finish latches on the same edge the run
                # ends on, so its sampler has not fired yet; it finished now.
                f'{master["name"]}_at < 0 ? cycle : {master["name"]}_at);'
            )
        lines += [
            '        $display("SPMW COSIM %s (%0d errors)",',
            '                 errors == 0 ? "PASS" : "FAIL", errors);',
            f'        $display("SPMW MEMORY CYCLES total=%0d latency={self.latency}",',
            "                 cycle + 1);",
            "        $finish;",
            "      end",
            "    end",
            '    $display("SPMW COSIM TIMEOUT");',
            '    $display("SPMW MEMORY CYCLES total=-1 latency=%0d",'
            f" {self.latency});",
            "    $finish;",
            "  end",
            "endmodule",
        ]
        return "\n".join(lines) + "\n"

    def _timing(self, masters):
        """When each transfer finished, so the run can be broken down.

        A single end-to-end number cannot say whether the loads, the compute or
        the drain is the cost, and that is the question the comparison against
        AutoSA turns on -- their own report splits the same way.
        """
        lines = []
        for master in masters:
            name = master["name"]
            lines += [
                f"  integer {name}_at = -1;",
                f"  always @(posedge clk) if (rst_n && {name}_at < 0 && "
                f"dut.{name}_done_r) {name}_at = cycle;",
            ]
        return lines

    def _bytes(self, master):
        """How large this master's memory has to be."""
        elem = np.dtype(_NUMPY[master["dtype"]]).itemsize
        return max(64, _volume(master["shape"]) * elem)

    def _ram(self, master, conns):
        """One AXI slave, wired to one of the top's masters."""
        name = master["name"]
        wide = self._bytes(master)
        lines = [
            f"  // {name}: {master['tensor']}{list(master['shape'])}, "
            f"{'read' if master['reads'] else 'written'}, "
            f"{master['width']}-bit master",
        ]
        signals = [
            ("ARVALID", "output"),
            ("ARREADY", "input"),
            ("ARADDR", "output"),
            ("ARLEN", "output"),
            ("RVALID", "input"),
            ("RREADY", "output"),
            ("RDATA", "input"),
            ("RLAST", "input"),
            ("AWVALID", "output"),
            ("AWREADY", "input"),
            ("AWADDR", "output"),
            ("AWLEN", "output"),
            ("WVALID", "output"),
            ("WREADY", "input"),
            ("WDATA", "output"),
            ("WSTRB", "output"),
            ("WLAST", "output"),
            ("BVALID", "input"),
            ("BREADY", "output"),
        ]
        widths = {
            "ARADDR": 64,
            "AWADDR": 64,
            "ARLEN": 8,
            "AWLEN": 8,
            "RDATA": master["width"],
            "WDATA": master["width"],
            "WSTRB": max(1, master["width"] // 8),
        }
        for signal, _kind in signals:
            span = f"[{widths[signal] - 1}:0] " if signal in widths else ""
            lines.append(f"  wire {span}{name}_{signal};")
        # every AXI signal the top drives, whether the model uses it or not
        for signal, _kind in signals:
            conns.append(f".m_axi_{name}_{signal}({name}_{signal})")
        for unused in (
            "ARID ARSIZE ARBURST ARLOCK ARCACHE ARPROT ARQOS ARREGION ARUSER "
            "AWID AWSIZE AWBURST AWLOCK AWCACHE AWPROT AWQOS AWREGION AWUSER "
            "WID WUSER RID RUSER RRESP BRESP BID BUSER"
        ).split():
            conns.append(f".m_axi_{name}_{unused}()")
        # Each master addresses its own memory from zero: they are separate
        # ports on separate channels, which is what the offsets say.
        conns.append(f".{name}_offset(64'd0)")
        lines.append(
            f"  spmw_axi_ram #(.DW({master['width']}), .BYTES({wide}), "
            f".LATENCY({self.latency})) u_{name} (\n"
            f"      .ap_clk(clk), .ap_rst_n(rst_n),\n      "
            + ", ".join(f".{s}({name}_{s})" for s, _k in signals)
            + ");"
        )
        return lines

    def _words(self, master, array):
        """The tensor as bytes, little-endian, in address order."""
        flat = np.ascontiguousarray(array.astype(_NUMPY[master["dtype"]])).tobytes()
        return list(flat)

    def _preload(self, master):
        """Fill an input master's memory before the array starts."""
        values = self._words(master, self.data[master["tensor"]])
        return [f"    // {master['name']} <- {master['tensor']}"] + [
            f"    u_{master['name']}.mem[{addr}] = 8'd{byte};"
            for addr, byte in enumerate(values)
        ]

    def _check(self, master):
        """Compare an output master's memory against the reference.

        Only where this instance writes: a per-column drain fills one slice and
        leaves the rest of its private memory untouched, so comparing the whole
        tensor would report every other column as wrong.
        """
        array = self.results[master["tensor"]]
        values = self._words(master, array)
        elem = np.dtype(_NUMPY[master["dtype"]]).itemsize
        lines = [f"        // {master['name']} -> {master['tensor']}"]
        for flat in self._flat(master, array):
            for offset in range(elem):
                addr = flat * elem + offset
                lines.append(
                    f"        if (u_{master['name']}.mem[{addr}] !== "
                    f"8'd{values[addr]}) begin errors = errors + 1; "
                    f'$display("  {master["tensor"]} byte {addr} = %0d, want '
                    f'{values[addr]}", u_{master["name"]}.mem[{addr}]); end'
                )
        return lines

    def _flat(self, master, array):
        """The flat element positions this master moves, row-major."""
        touches = master["touches"]
        if touches is None:
            return range(array.size)
        out = []
        for index in touches:
            flat, stride = 0, 1
            for axis in reversed(range(len(master["shape"]))):
                flat += int(index[axis]) * stride
                stride *= master["shape"][axis]
            out.append(flat)
        return out


def render_memory_testbench(
    graph, data, results, latency=0, cycles=2000000, axi_widths=None
):
    """The whole bench for a memory-mapped fabric."""
    bench = MemoryBench(graph, data, results, latency=latency, axi_widths=axi_widths)
    return bench.render(cycles=cycles)


__all__ = ["MemoryBench", "ram_module", "render_memory_testbench"]

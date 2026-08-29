`timescale 1ns/1ps

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

`timescale 1ns/1ps

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

// The same handshake on a block RAM. `write` only ever arrives with `full_n`
// high and `read` with `empty_n` high -- that is the ap_fifo contract -- and
// the gating below keeps the macro's own overflow and underflow checks quiet
// through reset, when the HLS side is held too.
module spmw_fifo_bram #(parameter DW = 32, parameter DEPTH = 1024) (
  input  wire          clk,
  input  wire          rst_n,
  input  wire [DW-1:0] din,
  output wire          full_n,
  input  wire          write,
  output wire [DW-1:0] dout,
  output wire          empty_n,
  input  wire          read
);
  wire full, empty, wr_rst_busy, rd_rst_busy;
  assign full_n  = ~full & ~wr_rst_busy;
  assign empty_n = ~empty & ~rd_rst_busy;
  xpm_fifo_sync #(
    .FIFO_MEMORY_TYPE("block"),
    .FIFO_WRITE_DEPTH(DEPTH),
    .WRITE_DATA_WIDTH(DW),
    .READ_DATA_WIDTH(DW),
    .READ_MODE("fwft"),
    .FIFO_READ_LATENCY(0),
    .USE_ADV_FEATURES("0000"),
    .ECC_MODE("no_ecc"),
    .WAKEUP_TIME(0),
    .DOUT_RESET_VALUE("0"),
    .FULL_RESET_VALUE(1),
    .RD_DATA_COUNT_WIDTH(1),
    .WR_DATA_COUNT_WIDTH(1)
  ) u (
    .rst(~rst_n),
    .wr_clk(clk),
    .wr_en(write & full_n),
    .din(din),
    .full(full),
    .wr_rst_busy(wr_rst_busy),
    .rd_en(read & empty_n),
    .dout(dout),
    .empty(empty),
    .rd_rst_busy(rd_rst_busy),
    .sleep(1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .overflow(),
    .underflow(),
    .prog_full(),
    .prog_empty(),
    .wr_ack(),
    .almost_full(),
    .almost_empty(),
    .data_valid(),
    .rd_data_count(),
    .wr_data_count(),
    .sbiterr(),
    .dbiterr()
  );
endmodule

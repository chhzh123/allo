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
  generate
    if (DEPTH <= 2) begin : slice
      // q0 is the output register, q1 the skid behind it. full_n is ~v1 -- a
      // flop -- so a beat written while the output holds lands in the skid,
      // and the producer sees the slice fill one cycle later.
      reg [DW-1:0] q0, q1;
      reg          v0, v1;
      assign dout    = q0;
      assign empty_n = v0;
      assign full_n  = ~v1;
      wire pop  = read  & v0;
      wire push = write & ~v1;
      always @(posedge clk) begin
        if (!rst_n) begin
          v0 <= 1'b0; v1 <= 1'b0;
        end else if (!v0 | pop) begin
          if (v1) begin
            q0 <= q1; v0 <= 1'b1; v1 <= 1'b0;
          end else begin
            q0 <= din; v0 <= push;
          end
        end else if (push) begin
          q1 <= din; v1 <= 1'b1;
        end
      end
    end else begin : deep
      wire [DW-1:0] mid_d;
      wire          mid_v, mid_r;
      spmw_fifo_lutram #(.DW(DW), .DEPTH(DEPTH)) ram (
        .clk(clk), .rst_n(rst_n),
        .din(din), .full_n(full_n), .write(write),
        .dout(mid_d), .empty_n(mid_v), .read(mid_v & mid_r));
      spmw_fifo #(.DW(DW), .DEPTH(2)) out (
        .clk(clk), .rst_n(rst_n),
        .din(mid_d), .full_n(mid_r), .write(mid_v & mid_r),
        .dout(dout), .empty_n(empty_n), .read(read));
    end
  endgenerate
endmodule

// The LUT-RAM FIFO the deep variant is built on: `dout` is the RAM's
// asynchronous read, so it only ever feeds the slice above.
module spmw_fifo_lutram #(parameter DW = 32, parameter DEPTH = 4) (
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

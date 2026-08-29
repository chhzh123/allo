`timescale 1ns/1ps

module tb;
  reg clk = 0, rst_n = 0, start = 0;
  always #5 clk = ~clk;
  wire done;
  integer errors = 0;
  integer cycle = 0;
  // feed_up_load_io_0: At[4, 4], read, 8-bit master
  wire feed_up_load_io_0_ARVALID;
  wire feed_up_load_io_0_ARREADY;
  wire [63:0] feed_up_load_io_0_ARADDR;
  wire [7:0] feed_up_load_io_0_ARLEN;
  wire feed_up_load_io_0_RVALID;
  wire feed_up_load_io_0_RREADY;
  wire [7:0] feed_up_load_io_0_RDATA;
  wire feed_up_load_io_0_RLAST;
  wire feed_up_load_io_0_AWVALID;
  wire feed_up_load_io_0_AWREADY;
  wire [63:0] feed_up_load_io_0_AWADDR;
  wire [7:0] feed_up_load_io_0_AWLEN;
  wire feed_up_load_io_0_WVALID;
  wire feed_up_load_io_0_WREADY;
  wire [7:0] feed_up_load_io_0_WDATA;
  wire [0:0] feed_up_load_io_0_WSTRB;
  wire feed_up_load_io_0_WLAST;
  wire feed_up_load_io_0_BVALID;
  wire feed_up_load_io_0_BREADY;
  spmw_axi_ram #(.DW(8), .BYTES(64), .LATENCY(0)) u_feed_up_load_io_0 (
      .ap_clk(clk), .ap_rst_n(rst_n),
      .ARVALID(feed_up_load_io_0_ARVALID), .ARREADY(feed_up_load_io_0_ARREADY), .ARADDR(feed_up_load_io_0_ARADDR), .ARLEN(feed_up_load_io_0_ARLEN), .RVALID(feed_up_load_io_0_RVALID), .RREADY(feed_up_load_io_0_RREADY), .RDATA(feed_up_load_io_0_RDATA), .RLAST(feed_up_load_io_0_RLAST), .AWVALID(feed_up_load_io_0_AWVALID), .AWREADY(feed_up_load_io_0_AWREADY), .AWADDR(feed_up_load_io_0_AWADDR), .AWLEN(feed_up_load_io_0_AWLEN), .WVALID(feed_up_load_io_0_WVALID), .WREADY(feed_up_load_io_0_WREADY), .WDATA(feed_up_load_io_0_WDATA), .WSTRB(feed_up_load_io_0_WSTRB), .WLAST(feed_up_load_io_0_WLAST), .BVALID(feed_up_load_io_0_BVALID), .BREADY(feed_up_load_io_0_BREADY));
  // feed_3_up_load_io_0: Bt[4, 4], read, 8-bit master
  wire feed_3_up_load_io_0_ARVALID;
  wire feed_3_up_load_io_0_ARREADY;
  wire [63:0] feed_3_up_load_io_0_ARADDR;
  wire [7:0] feed_3_up_load_io_0_ARLEN;
  wire feed_3_up_load_io_0_RVALID;
  wire feed_3_up_load_io_0_RREADY;
  wire [7:0] feed_3_up_load_io_0_RDATA;
  wire feed_3_up_load_io_0_RLAST;
  wire feed_3_up_load_io_0_AWVALID;
  wire feed_3_up_load_io_0_AWREADY;
  wire [63:0] feed_3_up_load_io_0_AWADDR;
  wire [7:0] feed_3_up_load_io_0_AWLEN;
  wire feed_3_up_load_io_0_WVALID;
  wire feed_3_up_load_io_0_WREADY;
  wire [7:0] feed_3_up_load_io_0_WDATA;
  wire [0:0] feed_3_up_load_io_0_WSTRB;
  wire feed_3_up_load_io_0_WLAST;
  wire feed_3_up_load_io_0_BVALID;
  wire feed_3_up_load_io_0_BREADY;
  spmw_axi_ram #(.DW(8), .BYTES(64), .LATENCY(0)) u_feed_3_up_load_io_0 (
      .ap_clk(clk), .ap_rst_n(rst_n),
      .ARVALID(feed_3_up_load_io_0_ARVALID), .ARREADY(feed_3_up_load_io_0_ARREADY), .ARADDR(feed_3_up_load_io_0_ARADDR), .ARLEN(feed_3_up_load_io_0_ARLEN), .RVALID(feed_3_up_load_io_0_RVALID), .RREADY(feed_3_up_load_io_0_RREADY), .RDATA(feed_3_up_load_io_0_RDATA), .RLAST(feed_3_up_load_io_0_RLAST), .AWVALID(feed_3_up_load_io_0_AWVALID), .AWREADY(feed_3_up_load_io_0_AWREADY), .AWADDR(feed_3_up_load_io_0_AWADDR), .AWLEN(feed_3_up_load_io_0_AWLEN), .WVALID(feed_3_up_load_io_0_WVALID), .WREADY(feed_3_up_load_io_0_WREADY), .WDATA(feed_3_up_load_io_0_WDATA), .WSTRB(feed_3_up_load_io_0_WSTRB), .WLAST(feed_3_up_load_io_0_WLAST), .BVALID(feed_3_up_load_io_0_BVALID), .BREADY(feed_3_up_load_io_0_BREADY));
  // drain_down_drain_io_0: Ct[4, 4], written, 32-bit master
  wire drain_down_drain_io_0_ARVALID;
  wire drain_down_drain_io_0_ARREADY;
  wire [63:0] drain_down_drain_io_0_ARADDR;
  wire [7:0] drain_down_drain_io_0_ARLEN;
  wire drain_down_drain_io_0_RVALID;
  wire drain_down_drain_io_0_RREADY;
  wire [31:0] drain_down_drain_io_0_RDATA;
  wire drain_down_drain_io_0_RLAST;
  wire drain_down_drain_io_0_AWVALID;
  wire drain_down_drain_io_0_AWREADY;
  wire [63:0] drain_down_drain_io_0_AWADDR;
  wire [7:0] drain_down_drain_io_0_AWLEN;
  wire drain_down_drain_io_0_WVALID;
  wire drain_down_drain_io_0_WREADY;
  wire [31:0] drain_down_drain_io_0_WDATA;
  wire [3:0] drain_down_drain_io_0_WSTRB;
  wire drain_down_drain_io_0_WLAST;
  wire drain_down_drain_io_0_BVALID;
  wire drain_down_drain_io_0_BREADY;
  spmw_axi_ram #(.DW(32), .BYTES(64), .LATENCY(0)) u_drain_down_drain_io_0 (
      .ap_clk(clk), .ap_rst_n(rst_n),
      .ARVALID(drain_down_drain_io_0_ARVALID), .ARREADY(drain_down_drain_io_0_ARREADY), .ARADDR(drain_down_drain_io_0_ARADDR), .ARLEN(drain_down_drain_io_0_ARLEN), .RVALID(drain_down_drain_io_0_RVALID), .RREADY(drain_down_drain_io_0_RREADY), .RDATA(drain_down_drain_io_0_RDATA), .RLAST(drain_down_drain_io_0_RLAST), .AWVALID(drain_down_drain_io_0_AWVALID), .AWREADY(drain_down_drain_io_0_AWREADY), .AWADDR(drain_down_drain_io_0_AWADDR), .AWLEN(drain_down_drain_io_0_AWLEN), .WVALID(drain_down_drain_io_0_WVALID), .WREADY(drain_down_drain_io_0_WREADY), .WDATA(drain_down_drain_io_0_WDATA), .WSTRB(drain_down_drain_io_0_WSTRB), .WLAST(drain_down_drain_io_0_WLAST), .BVALID(drain_down_drain_io_0_BVALID), .BREADY(drain_down_drain_io_0_BREADY));
  // drain_down_drain_io_1: Ct[4, 4], written, 32-bit master
  wire drain_down_drain_io_1_ARVALID;
  wire drain_down_drain_io_1_ARREADY;
  wire [63:0] drain_down_drain_io_1_ARADDR;
  wire [7:0] drain_down_drain_io_1_ARLEN;
  wire drain_down_drain_io_1_RVALID;
  wire drain_down_drain_io_1_RREADY;
  wire [31:0] drain_down_drain_io_1_RDATA;
  wire drain_down_drain_io_1_RLAST;
  wire drain_down_drain_io_1_AWVALID;
  wire drain_down_drain_io_1_AWREADY;
  wire [63:0] drain_down_drain_io_1_AWADDR;
  wire [7:0] drain_down_drain_io_1_AWLEN;
  wire drain_down_drain_io_1_WVALID;
  wire drain_down_drain_io_1_WREADY;
  wire [31:0] drain_down_drain_io_1_WDATA;
  wire [3:0] drain_down_drain_io_1_WSTRB;
  wire drain_down_drain_io_1_WLAST;
  wire drain_down_drain_io_1_BVALID;
  wire drain_down_drain_io_1_BREADY;
  spmw_axi_ram #(.DW(32), .BYTES(64), .LATENCY(0)) u_drain_down_drain_io_1 (
      .ap_clk(clk), .ap_rst_n(rst_n),
      .ARVALID(drain_down_drain_io_1_ARVALID), .ARREADY(drain_down_drain_io_1_ARREADY), .ARADDR(drain_down_drain_io_1_ARADDR), .ARLEN(drain_down_drain_io_1_ARLEN), .RVALID(drain_down_drain_io_1_RVALID), .RREADY(drain_down_drain_io_1_RREADY), .RDATA(drain_down_drain_io_1_RDATA), .RLAST(drain_down_drain_io_1_RLAST), .AWVALID(drain_down_drain_io_1_AWVALID), .AWREADY(drain_down_drain_io_1_AWREADY), .AWADDR(drain_down_drain_io_1_AWADDR), .AWLEN(drain_down_drain_io_1_AWLEN), .WVALID(drain_down_drain_io_1_WVALID), .WREADY(drain_down_drain_io_1_WREADY), .WDATA(drain_down_drain_io_1_WDATA), .WSTRB(drain_down_drain_io_1_WSTRB), .WLAST(drain_down_drain_io_1_WLAST), .BVALID(drain_down_drain_io_1_BVALID), .BREADY(drain_down_drain_io_1_BREADY));
  // drain_down_drain_io_2: Ct[4, 4], written, 32-bit master
  wire drain_down_drain_io_2_ARVALID;
  wire drain_down_drain_io_2_ARREADY;
  wire [63:0] drain_down_drain_io_2_ARADDR;
  wire [7:0] drain_down_drain_io_2_ARLEN;
  wire drain_down_drain_io_2_RVALID;
  wire drain_down_drain_io_2_RREADY;
  wire [31:0] drain_down_drain_io_2_RDATA;
  wire drain_down_drain_io_2_RLAST;
  wire drain_down_drain_io_2_AWVALID;
  wire drain_down_drain_io_2_AWREADY;
  wire [63:0] drain_down_drain_io_2_AWADDR;
  wire [7:0] drain_down_drain_io_2_AWLEN;
  wire drain_down_drain_io_2_WVALID;
  wire drain_down_drain_io_2_WREADY;
  wire [31:0] drain_down_drain_io_2_WDATA;
  wire [3:0] drain_down_drain_io_2_WSTRB;
  wire drain_down_drain_io_2_WLAST;
  wire drain_down_drain_io_2_BVALID;
  wire drain_down_drain_io_2_BREADY;
  spmw_axi_ram #(.DW(32), .BYTES(64), .LATENCY(0)) u_drain_down_drain_io_2 (
      .ap_clk(clk), .ap_rst_n(rst_n),
      .ARVALID(drain_down_drain_io_2_ARVALID), .ARREADY(drain_down_drain_io_2_ARREADY), .ARADDR(drain_down_drain_io_2_ARADDR), .ARLEN(drain_down_drain_io_2_ARLEN), .RVALID(drain_down_drain_io_2_RVALID), .RREADY(drain_down_drain_io_2_RREADY), .RDATA(drain_down_drain_io_2_RDATA), .RLAST(drain_down_drain_io_2_RLAST), .AWVALID(drain_down_drain_io_2_AWVALID), .AWREADY(drain_down_drain_io_2_AWREADY), .AWADDR(drain_down_drain_io_2_AWADDR), .AWLEN(drain_down_drain_io_2_AWLEN), .WVALID(drain_down_drain_io_2_WVALID), .WREADY(drain_down_drain_io_2_WREADY), .WDATA(drain_down_drain_io_2_WDATA), .WSTRB(drain_down_drain_io_2_WSTRB), .WLAST(drain_down_drain_io_2_WLAST), .BVALID(drain_down_drain_io_2_BVALID), .BREADY(drain_down_drain_io_2_BREADY));
  // drain_down_drain_io_3: Ct[4, 4], written, 32-bit master
  wire drain_down_drain_io_3_ARVALID;
  wire drain_down_drain_io_3_ARREADY;
  wire [63:0] drain_down_drain_io_3_ARADDR;
  wire [7:0] drain_down_drain_io_3_ARLEN;
  wire drain_down_drain_io_3_RVALID;
  wire drain_down_drain_io_3_RREADY;
  wire [31:0] drain_down_drain_io_3_RDATA;
  wire drain_down_drain_io_3_RLAST;
  wire drain_down_drain_io_3_AWVALID;
  wire drain_down_drain_io_3_AWREADY;
  wire [63:0] drain_down_drain_io_3_AWADDR;
  wire [7:0] drain_down_drain_io_3_AWLEN;
  wire drain_down_drain_io_3_WVALID;
  wire drain_down_drain_io_3_WREADY;
  wire [31:0] drain_down_drain_io_3_WDATA;
  wire [3:0] drain_down_drain_io_3_WSTRB;
  wire drain_down_drain_io_3_WLAST;
  wire drain_down_drain_io_3_BVALID;
  wire drain_down_drain_io_3_BREADY;
  spmw_axi_ram #(.DW(32), .BYTES(64), .LATENCY(0)) u_drain_down_drain_io_3 (
      .ap_clk(clk), .ap_rst_n(rst_n),
      .ARVALID(drain_down_drain_io_3_ARVALID), .ARREADY(drain_down_drain_io_3_ARREADY), .ARADDR(drain_down_drain_io_3_ARADDR), .ARLEN(drain_down_drain_io_3_ARLEN), .RVALID(drain_down_drain_io_3_RVALID), .RREADY(drain_down_drain_io_3_RREADY), .RDATA(drain_down_drain_io_3_RDATA), .RLAST(drain_down_drain_io_3_RLAST), .AWVALID(drain_down_drain_io_3_AWVALID), .AWREADY(drain_down_drain_io_3_AWREADY), .AWADDR(drain_down_drain_io_3_AWADDR), .AWLEN(drain_down_drain_io_3_AWLEN), .WVALID(drain_down_drain_io_3_WVALID), .WREADY(drain_down_drain_io_3_WREADY), .WDATA(drain_down_drain_io_3_WDATA), .WSTRB(drain_down_drain_io_3_WSTRB), .WLAST(drain_down_drain_io_3_WLAST), .BVALID(drain_down_drain_io_3_BVALID), .BREADY(drain_down_drain_io_3_BREADY));
  spmw_top dut (.ap_clk(clk),
      .ap_rst_n(rst_n),
      .ap_start(start),
      .ap_done(done),
      .m_axi_feed_up_load_io_0_ARVALID(feed_up_load_io_0_ARVALID),
      .m_axi_feed_up_load_io_0_ARREADY(feed_up_load_io_0_ARREADY),
      .m_axi_feed_up_load_io_0_ARADDR(feed_up_load_io_0_ARADDR),
      .m_axi_feed_up_load_io_0_ARLEN(feed_up_load_io_0_ARLEN),
      .m_axi_feed_up_load_io_0_RVALID(feed_up_load_io_0_RVALID),
      .m_axi_feed_up_load_io_0_RREADY(feed_up_load_io_0_RREADY),
      .m_axi_feed_up_load_io_0_RDATA(feed_up_load_io_0_RDATA),
      .m_axi_feed_up_load_io_0_RLAST(feed_up_load_io_0_RLAST),
      .m_axi_feed_up_load_io_0_AWVALID(feed_up_load_io_0_AWVALID),
      .m_axi_feed_up_load_io_0_AWREADY(feed_up_load_io_0_AWREADY),
      .m_axi_feed_up_load_io_0_AWADDR(feed_up_load_io_0_AWADDR),
      .m_axi_feed_up_load_io_0_AWLEN(feed_up_load_io_0_AWLEN),
      .m_axi_feed_up_load_io_0_WVALID(feed_up_load_io_0_WVALID),
      .m_axi_feed_up_load_io_0_WREADY(feed_up_load_io_0_WREADY),
      .m_axi_feed_up_load_io_0_WDATA(feed_up_load_io_0_WDATA),
      .m_axi_feed_up_load_io_0_WSTRB(feed_up_load_io_0_WSTRB),
      .m_axi_feed_up_load_io_0_WLAST(feed_up_load_io_0_WLAST),
      .m_axi_feed_up_load_io_0_BVALID(feed_up_load_io_0_BVALID),
      .m_axi_feed_up_load_io_0_BREADY(feed_up_load_io_0_BREADY),
      .m_axi_feed_up_load_io_0_ARID(),
      .m_axi_feed_up_load_io_0_ARSIZE(),
      .m_axi_feed_up_load_io_0_ARBURST(),
      .m_axi_feed_up_load_io_0_ARLOCK(),
      .m_axi_feed_up_load_io_0_ARCACHE(),
      .m_axi_feed_up_load_io_0_ARPROT(),
      .m_axi_feed_up_load_io_0_ARQOS(),
      .m_axi_feed_up_load_io_0_ARREGION(),
      .m_axi_feed_up_load_io_0_ARUSER(),
      .m_axi_feed_up_load_io_0_AWID(),
      .m_axi_feed_up_load_io_0_AWSIZE(),
      .m_axi_feed_up_load_io_0_AWBURST(),
      .m_axi_feed_up_load_io_0_AWLOCK(),
      .m_axi_feed_up_load_io_0_AWCACHE(),
      .m_axi_feed_up_load_io_0_AWPROT(),
      .m_axi_feed_up_load_io_0_AWQOS(),
      .m_axi_feed_up_load_io_0_AWREGION(),
      .m_axi_feed_up_load_io_0_AWUSER(),
      .m_axi_feed_up_load_io_0_WID(),
      .m_axi_feed_up_load_io_0_WUSER(),
      .m_axi_feed_up_load_io_0_RID(),
      .m_axi_feed_up_load_io_0_RUSER(),
      .m_axi_feed_up_load_io_0_RRESP(),
      .m_axi_feed_up_load_io_0_BRESP(),
      .m_axi_feed_up_load_io_0_BID(),
      .m_axi_feed_up_load_io_0_BUSER(),
      .feed_up_load_io_0_offset(64'd0),
      .m_axi_feed_3_up_load_io_0_ARVALID(feed_3_up_load_io_0_ARVALID),
      .m_axi_feed_3_up_load_io_0_ARREADY(feed_3_up_load_io_0_ARREADY),
      .m_axi_feed_3_up_load_io_0_ARADDR(feed_3_up_load_io_0_ARADDR),
      .m_axi_feed_3_up_load_io_0_ARLEN(feed_3_up_load_io_0_ARLEN),
      .m_axi_feed_3_up_load_io_0_RVALID(feed_3_up_load_io_0_RVALID),
      .m_axi_feed_3_up_load_io_0_RREADY(feed_3_up_load_io_0_RREADY),
      .m_axi_feed_3_up_load_io_0_RDATA(feed_3_up_load_io_0_RDATA),
      .m_axi_feed_3_up_load_io_0_RLAST(feed_3_up_load_io_0_RLAST),
      .m_axi_feed_3_up_load_io_0_AWVALID(feed_3_up_load_io_0_AWVALID),
      .m_axi_feed_3_up_load_io_0_AWREADY(feed_3_up_load_io_0_AWREADY),
      .m_axi_feed_3_up_load_io_0_AWADDR(feed_3_up_load_io_0_AWADDR),
      .m_axi_feed_3_up_load_io_0_AWLEN(feed_3_up_load_io_0_AWLEN),
      .m_axi_feed_3_up_load_io_0_WVALID(feed_3_up_load_io_0_WVALID),
      .m_axi_feed_3_up_load_io_0_WREADY(feed_3_up_load_io_0_WREADY),
      .m_axi_feed_3_up_load_io_0_WDATA(feed_3_up_load_io_0_WDATA),
      .m_axi_feed_3_up_load_io_0_WSTRB(feed_3_up_load_io_0_WSTRB),
      .m_axi_feed_3_up_load_io_0_WLAST(feed_3_up_load_io_0_WLAST),
      .m_axi_feed_3_up_load_io_0_BVALID(feed_3_up_load_io_0_BVALID),
      .m_axi_feed_3_up_load_io_0_BREADY(feed_3_up_load_io_0_BREADY),
      .m_axi_feed_3_up_load_io_0_ARID(),
      .m_axi_feed_3_up_load_io_0_ARSIZE(),
      .m_axi_feed_3_up_load_io_0_ARBURST(),
      .m_axi_feed_3_up_load_io_0_ARLOCK(),
      .m_axi_feed_3_up_load_io_0_ARCACHE(),
      .m_axi_feed_3_up_load_io_0_ARPROT(),
      .m_axi_feed_3_up_load_io_0_ARQOS(),
      .m_axi_feed_3_up_load_io_0_ARREGION(),
      .m_axi_feed_3_up_load_io_0_ARUSER(),
      .m_axi_feed_3_up_load_io_0_AWID(),
      .m_axi_feed_3_up_load_io_0_AWSIZE(),
      .m_axi_feed_3_up_load_io_0_AWBURST(),
      .m_axi_feed_3_up_load_io_0_AWLOCK(),
      .m_axi_feed_3_up_load_io_0_AWCACHE(),
      .m_axi_feed_3_up_load_io_0_AWPROT(),
      .m_axi_feed_3_up_load_io_0_AWQOS(),
      .m_axi_feed_3_up_load_io_0_AWREGION(),
      .m_axi_feed_3_up_load_io_0_AWUSER(),
      .m_axi_feed_3_up_load_io_0_WID(),
      .m_axi_feed_3_up_load_io_0_WUSER(),
      .m_axi_feed_3_up_load_io_0_RID(),
      .m_axi_feed_3_up_load_io_0_RUSER(),
      .m_axi_feed_3_up_load_io_0_RRESP(),
      .m_axi_feed_3_up_load_io_0_BRESP(),
      .m_axi_feed_3_up_load_io_0_BID(),
      .m_axi_feed_3_up_load_io_0_BUSER(),
      .feed_3_up_load_io_0_offset(64'd0),
      .m_axi_drain_down_drain_io_0_ARVALID(drain_down_drain_io_0_ARVALID),
      .m_axi_drain_down_drain_io_0_ARREADY(drain_down_drain_io_0_ARREADY),
      .m_axi_drain_down_drain_io_0_ARADDR(drain_down_drain_io_0_ARADDR),
      .m_axi_drain_down_drain_io_0_ARLEN(drain_down_drain_io_0_ARLEN),
      .m_axi_drain_down_drain_io_0_RVALID(drain_down_drain_io_0_RVALID),
      .m_axi_drain_down_drain_io_0_RREADY(drain_down_drain_io_0_RREADY),
      .m_axi_drain_down_drain_io_0_RDATA(drain_down_drain_io_0_RDATA),
      .m_axi_drain_down_drain_io_0_RLAST(drain_down_drain_io_0_RLAST),
      .m_axi_drain_down_drain_io_0_AWVALID(drain_down_drain_io_0_AWVALID),
      .m_axi_drain_down_drain_io_0_AWREADY(drain_down_drain_io_0_AWREADY),
      .m_axi_drain_down_drain_io_0_AWADDR(drain_down_drain_io_0_AWADDR),
      .m_axi_drain_down_drain_io_0_AWLEN(drain_down_drain_io_0_AWLEN),
      .m_axi_drain_down_drain_io_0_WVALID(drain_down_drain_io_0_WVALID),
      .m_axi_drain_down_drain_io_0_WREADY(drain_down_drain_io_0_WREADY),
      .m_axi_drain_down_drain_io_0_WDATA(drain_down_drain_io_0_WDATA),
      .m_axi_drain_down_drain_io_0_WSTRB(drain_down_drain_io_0_WSTRB),
      .m_axi_drain_down_drain_io_0_WLAST(drain_down_drain_io_0_WLAST),
      .m_axi_drain_down_drain_io_0_BVALID(drain_down_drain_io_0_BVALID),
      .m_axi_drain_down_drain_io_0_BREADY(drain_down_drain_io_0_BREADY),
      .m_axi_drain_down_drain_io_0_ARID(),
      .m_axi_drain_down_drain_io_0_ARSIZE(),
      .m_axi_drain_down_drain_io_0_ARBURST(),
      .m_axi_drain_down_drain_io_0_ARLOCK(),
      .m_axi_drain_down_drain_io_0_ARCACHE(),
      .m_axi_drain_down_drain_io_0_ARPROT(),
      .m_axi_drain_down_drain_io_0_ARQOS(),
      .m_axi_drain_down_drain_io_0_ARREGION(),
      .m_axi_drain_down_drain_io_0_ARUSER(),
      .m_axi_drain_down_drain_io_0_AWID(),
      .m_axi_drain_down_drain_io_0_AWSIZE(),
      .m_axi_drain_down_drain_io_0_AWBURST(),
      .m_axi_drain_down_drain_io_0_AWLOCK(),
      .m_axi_drain_down_drain_io_0_AWCACHE(),
      .m_axi_drain_down_drain_io_0_AWPROT(),
      .m_axi_drain_down_drain_io_0_AWQOS(),
      .m_axi_drain_down_drain_io_0_AWREGION(),
      .m_axi_drain_down_drain_io_0_AWUSER(),
      .m_axi_drain_down_drain_io_0_WID(),
      .m_axi_drain_down_drain_io_0_WUSER(),
      .m_axi_drain_down_drain_io_0_RID(),
      .m_axi_drain_down_drain_io_0_RUSER(),
      .m_axi_drain_down_drain_io_0_RRESP(),
      .m_axi_drain_down_drain_io_0_BRESP(),
      .m_axi_drain_down_drain_io_0_BID(),
      .m_axi_drain_down_drain_io_0_BUSER(),
      .drain_down_drain_io_0_offset(64'd0),
      .m_axi_drain_down_drain_io_1_ARVALID(drain_down_drain_io_1_ARVALID),
      .m_axi_drain_down_drain_io_1_ARREADY(drain_down_drain_io_1_ARREADY),
      .m_axi_drain_down_drain_io_1_ARADDR(drain_down_drain_io_1_ARADDR),
      .m_axi_drain_down_drain_io_1_ARLEN(drain_down_drain_io_1_ARLEN),
      .m_axi_drain_down_drain_io_1_RVALID(drain_down_drain_io_1_RVALID),
      .m_axi_drain_down_drain_io_1_RREADY(drain_down_drain_io_1_RREADY),
      .m_axi_drain_down_drain_io_1_RDATA(drain_down_drain_io_1_RDATA),
      .m_axi_drain_down_drain_io_1_RLAST(drain_down_drain_io_1_RLAST),
      .m_axi_drain_down_drain_io_1_AWVALID(drain_down_drain_io_1_AWVALID),
      .m_axi_drain_down_drain_io_1_AWREADY(drain_down_drain_io_1_AWREADY),
      .m_axi_drain_down_drain_io_1_AWADDR(drain_down_drain_io_1_AWADDR),
      .m_axi_drain_down_drain_io_1_AWLEN(drain_down_drain_io_1_AWLEN),
      .m_axi_drain_down_drain_io_1_WVALID(drain_down_drain_io_1_WVALID),
      .m_axi_drain_down_drain_io_1_WREADY(drain_down_drain_io_1_WREADY),
      .m_axi_drain_down_drain_io_1_WDATA(drain_down_drain_io_1_WDATA),
      .m_axi_drain_down_drain_io_1_WSTRB(drain_down_drain_io_1_WSTRB),
      .m_axi_drain_down_drain_io_1_WLAST(drain_down_drain_io_1_WLAST),
      .m_axi_drain_down_drain_io_1_BVALID(drain_down_drain_io_1_BVALID),
      .m_axi_drain_down_drain_io_1_BREADY(drain_down_drain_io_1_BREADY),
      .m_axi_drain_down_drain_io_1_ARID(),
      .m_axi_drain_down_drain_io_1_ARSIZE(),
      .m_axi_drain_down_drain_io_1_ARBURST(),
      .m_axi_drain_down_drain_io_1_ARLOCK(),
      .m_axi_drain_down_drain_io_1_ARCACHE(),
      .m_axi_drain_down_drain_io_1_ARPROT(),
      .m_axi_drain_down_drain_io_1_ARQOS(),
      .m_axi_drain_down_drain_io_1_ARREGION(),
      .m_axi_drain_down_drain_io_1_ARUSER(),
      .m_axi_drain_down_drain_io_1_AWID(),
      .m_axi_drain_down_drain_io_1_AWSIZE(),
      .m_axi_drain_down_drain_io_1_AWBURST(),
      .m_axi_drain_down_drain_io_1_AWLOCK(),
      .m_axi_drain_down_drain_io_1_AWCACHE(),
      .m_axi_drain_down_drain_io_1_AWPROT(),
      .m_axi_drain_down_drain_io_1_AWQOS(),
      .m_axi_drain_down_drain_io_1_AWREGION(),
      .m_axi_drain_down_drain_io_1_AWUSER(),
      .m_axi_drain_down_drain_io_1_WID(),
      .m_axi_drain_down_drain_io_1_WUSER(),
      .m_axi_drain_down_drain_io_1_RID(),
      .m_axi_drain_down_drain_io_1_RUSER(),
      .m_axi_drain_down_drain_io_1_RRESP(),
      .m_axi_drain_down_drain_io_1_BRESP(),
      .m_axi_drain_down_drain_io_1_BID(),
      .m_axi_drain_down_drain_io_1_BUSER(),
      .drain_down_drain_io_1_offset(64'd0),
      .m_axi_drain_down_drain_io_2_ARVALID(drain_down_drain_io_2_ARVALID),
      .m_axi_drain_down_drain_io_2_ARREADY(drain_down_drain_io_2_ARREADY),
      .m_axi_drain_down_drain_io_2_ARADDR(drain_down_drain_io_2_ARADDR),
      .m_axi_drain_down_drain_io_2_ARLEN(drain_down_drain_io_2_ARLEN),
      .m_axi_drain_down_drain_io_2_RVALID(drain_down_drain_io_2_RVALID),
      .m_axi_drain_down_drain_io_2_RREADY(drain_down_drain_io_2_RREADY),
      .m_axi_drain_down_drain_io_2_RDATA(drain_down_drain_io_2_RDATA),
      .m_axi_drain_down_drain_io_2_RLAST(drain_down_drain_io_2_RLAST),
      .m_axi_drain_down_drain_io_2_AWVALID(drain_down_drain_io_2_AWVALID),
      .m_axi_drain_down_drain_io_2_AWREADY(drain_down_drain_io_2_AWREADY),
      .m_axi_drain_down_drain_io_2_AWADDR(drain_down_drain_io_2_AWADDR),
      .m_axi_drain_down_drain_io_2_AWLEN(drain_down_drain_io_2_AWLEN),
      .m_axi_drain_down_drain_io_2_WVALID(drain_down_drain_io_2_WVALID),
      .m_axi_drain_down_drain_io_2_WREADY(drain_down_drain_io_2_WREADY),
      .m_axi_drain_down_drain_io_2_WDATA(drain_down_drain_io_2_WDATA),
      .m_axi_drain_down_drain_io_2_WSTRB(drain_down_drain_io_2_WSTRB),
      .m_axi_drain_down_drain_io_2_WLAST(drain_down_drain_io_2_WLAST),
      .m_axi_drain_down_drain_io_2_BVALID(drain_down_drain_io_2_BVALID),
      .m_axi_drain_down_drain_io_2_BREADY(drain_down_drain_io_2_BREADY),
      .m_axi_drain_down_drain_io_2_ARID(),
      .m_axi_drain_down_drain_io_2_ARSIZE(),
      .m_axi_drain_down_drain_io_2_ARBURST(),
      .m_axi_drain_down_drain_io_2_ARLOCK(),
      .m_axi_drain_down_drain_io_2_ARCACHE(),
      .m_axi_drain_down_drain_io_2_ARPROT(),
      .m_axi_drain_down_drain_io_2_ARQOS(),
      .m_axi_drain_down_drain_io_2_ARREGION(),
      .m_axi_drain_down_drain_io_2_ARUSER(),
      .m_axi_drain_down_drain_io_2_AWID(),
      .m_axi_drain_down_drain_io_2_AWSIZE(),
      .m_axi_drain_down_drain_io_2_AWBURST(),
      .m_axi_drain_down_drain_io_2_AWLOCK(),
      .m_axi_drain_down_drain_io_2_AWCACHE(),
      .m_axi_drain_down_drain_io_2_AWPROT(),
      .m_axi_drain_down_drain_io_2_AWQOS(),
      .m_axi_drain_down_drain_io_2_AWREGION(),
      .m_axi_drain_down_drain_io_2_AWUSER(),
      .m_axi_drain_down_drain_io_2_WID(),
      .m_axi_drain_down_drain_io_2_WUSER(),
      .m_axi_drain_down_drain_io_2_RID(),
      .m_axi_drain_down_drain_io_2_RUSER(),
      .m_axi_drain_down_drain_io_2_RRESP(),
      .m_axi_drain_down_drain_io_2_BRESP(),
      .m_axi_drain_down_drain_io_2_BID(),
      .m_axi_drain_down_drain_io_2_BUSER(),
      .drain_down_drain_io_2_offset(64'd0),
      .m_axi_drain_down_drain_io_3_ARVALID(drain_down_drain_io_3_ARVALID),
      .m_axi_drain_down_drain_io_3_ARREADY(drain_down_drain_io_3_ARREADY),
      .m_axi_drain_down_drain_io_3_ARADDR(drain_down_drain_io_3_ARADDR),
      .m_axi_drain_down_drain_io_3_ARLEN(drain_down_drain_io_3_ARLEN),
      .m_axi_drain_down_drain_io_3_RVALID(drain_down_drain_io_3_RVALID),
      .m_axi_drain_down_drain_io_3_RREADY(drain_down_drain_io_3_RREADY),
      .m_axi_drain_down_drain_io_3_RDATA(drain_down_drain_io_3_RDATA),
      .m_axi_drain_down_drain_io_3_RLAST(drain_down_drain_io_3_RLAST),
      .m_axi_drain_down_drain_io_3_AWVALID(drain_down_drain_io_3_AWVALID),
      .m_axi_drain_down_drain_io_3_AWREADY(drain_down_drain_io_3_AWREADY),
      .m_axi_drain_down_drain_io_3_AWADDR(drain_down_drain_io_3_AWADDR),
      .m_axi_drain_down_drain_io_3_AWLEN(drain_down_drain_io_3_AWLEN),
      .m_axi_drain_down_drain_io_3_WVALID(drain_down_drain_io_3_WVALID),
      .m_axi_drain_down_drain_io_3_WREADY(drain_down_drain_io_3_WREADY),
      .m_axi_drain_down_drain_io_3_WDATA(drain_down_drain_io_3_WDATA),
      .m_axi_drain_down_drain_io_3_WSTRB(drain_down_drain_io_3_WSTRB),
      .m_axi_drain_down_drain_io_3_WLAST(drain_down_drain_io_3_WLAST),
      .m_axi_drain_down_drain_io_3_BVALID(drain_down_drain_io_3_BVALID),
      .m_axi_drain_down_drain_io_3_BREADY(drain_down_drain_io_3_BREADY),
      .m_axi_drain_down_drain_io_3_ARID(),
      .m_axi_drain_down_drain_io_3_ARSIZE(),
      .m_axi_drain_down_drain_io_3_ARBURST(),
      .m_axi_drain_down_drain_io_3_ARLOCK(),
      .m_axi_drain_down_drain_io_3_ARCACHE(),
      .m_axi_drain_down_drain_io_3_ARPROT(),
      .m_axi_drain_down_drain_io_3_ARQOS(),
      .m_axi_drain_down_drain_io_3_ARREGION(),
      .m_axi_drain_down_drain_io_3_ARUSER(),
      .m_axi_drain_down_drain_io_3_AWID(),
      .m_axi_drain_down_drain_io_3_AWSIZE(),
      .m_axi_drain_down_drain_io_3_AWBURST(),
      .m_axi_drain_down_drain_io_3_AWLOCK(),
      .m_axi_drain_down_drain_io_3_AWCACHE(),
      .m_axi_drain_down_drain_io_3_AWPROT(),
      .m_axi_drain_down_drain_io_3_AWQOS(),
      .m_axi_drain_down_drain_io_3_AWREGION(),
      .m_axi_drain_down_drain_io_3_AWUSER(),
      .m_axi_drain_down_drain_io_3_WID(),
      .m_axi_drain_down_drain_io_3_WUSER(),
      .m_axi_drain_down_drain_io_3_RID(),
      .m_axi_drain_down_drain_io_3_RUSER(),
      .m_axi_drain_down_drain_io_3_RRESP(),
      .m_axi_drain_down_drain_io_3_BRESP(),
      .m_axi_drain_down_drain_io_3_BID(),
      .m_axi_drain_down_drain_io_3_BUSER(),
      .drain_down_drain_io_3_offset(64'd0));
  integer feed_up_load_io_0_at = -1;
  always @(posedge clk) if (rst_n && feed_up_load_io_0_at < 0 && dut.feed_up_load_io_0_done_r) feed_up_load_io_0_at = cycle;
  integer feed_3_up_load_io_0_at = -1;
  always @(posedge clk) if (rst_n && feed_3_up_load_io_0_at < 0 && dut.feed_3_up_load_io_0_done_r) feed_3_up_load_io_0_at = cycle;
  integer drain_down_drain_io_0_at = -1;
  always @(posedge clk) if (rst_n && drain_down_drain_io_0_at < 0 && dut.drain_down_drain_io_0_done_r) drain_down_drain_io_0_at = cycle;
  integer drain_down_drain_io_1_at = -1;
  always @(posedge clk) if (rst_n && drain_down_drain_io_1_at < 0 && dut.drain_down_drain_io_1_done_r) drain_down_drain_io_1_at = cycle;
  integer drain_down_drain_io_2_at = -1;
  always @(posedge clk) if (rst_n && drain_down_drain_io_2_at < 0 && dut.drain_down_drain_io_2_done_r) drain_down_drain_io_2_at = cycle;
  integer drain_down_drain_io_3_at = -1;
  always @(posedge clk) if (rst_n && drain_down_drain_io_3_at < 0 && dut.drain_down_drain_io_3_done_r) drain_down_drain_io_3_at = cycle;
  initial begin
    repeat (4) @(posedge clk);
    // feed_up_load_io_0 <- At
    u_feed_up_load_io_0.mem[0] = 8'd2;
    u_feed_up_load_io_0.mem[1] = 8'd1;
    u_feed_up_load_io_0.mem[2] = 8'd1;
    u_feed_up_load_io_0.mem[3] = 8'd0;
    u_feed_up_load_io_0.mem[4] = 8'd0;
    u_feed_up_load_io_0.mem[5] = 8'd0;
    u_feed_up_load_io_0.mem[6] = 8'd0;
    u_feed_up_load_io_0.mem[7] = 8'd0;
    u_feed_up_load_io_0.mem[8] = 8'd0;
    u_feed_up_load_io_0.mem[9] = 8'd2;
    u_feed_up_load_io_0.mem[10] = 8'd1;
    u_feed_up_load_io_0.mem[11] = 8'd2;
    u_feed_up_load_io_0.mem[12] = 8'd1;
    u_feed_up_load_io_0.mem[13] = 8'd1;
    u_feed_up_load_io_0.mem[14] = 8'd2;
    u_feed_up_load_io_0.mem[15] = 8'd2;
    // feed_3_up_load_io_0 <- Bt
    u_feed_3_up_load_io_0.mem[0] = 8'd1;
    u_feed_3_up_load_io_0.mem[1] = 8'd1;
    u_feed_3_up_load_io_0.mem[2] = 8'd1;
    u_feed_3_up_load_io_0.mem[3] = 8'd2;
    u_feed_3_up_load_io_0.mem[4] = 8'd0;
    u_feed_3_up_load_io_0.mem[5] = 8'd2;
    u_feed_3_up_load_io_0.mem[6] = 8'd2;
    u_feed_3_up_load_io_0.mem[7] = 8'd0;
    u_feed_3_up_load_io_0.mem[8] = 8'd1;
    u_feed_3_up_load_io_0.mem[9] = 8'd2;
    u_feed_3_up_load_io_0.mem[10] = 8'd1;
    u_feed_3_up_load_io_0.mem[11] = 8'd0;
    u_feed_3_up_load_io_0.mem[12] = 8'd2;
    u_feed_3_up_load_io_0.mem[13] = 8'd2;
    u_feed_3_up_load_io_0.mem[14] = 8'd2;
    u_feed_3_up_load_io_0.mem[15] = 8'd0;
    rst_n = 1;
    @(posedge clk);
    start = 1;
    for (cycle = 0; cycle < 2000000; cycle = cycle + 1) begin
      @(posedge clk);
      if (done) begin
        // drain_down_drain_io_0 -> Ct
        if (u_drain_down_drain_io_0.mem[0] !== 8'd6) begin errors = errors + 1; $display("  Ct byte 0 = %0d, want 6", u_drain_down_drain_io_0.mem[0]); end
        if (u_drain_down_drain_io_0.mem[1] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 1 = %0d, want 0", u_drain_down_drain_io_0.mem[1]); end
        if (u_drain_down_drain_io_0.mem[2] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 2 = %0d, want 0", u_drain_down_drain_io_0.mem[2]); end
        if (u_drain_down_drain_io_0.mem[3] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 3 = %0d, want 0", u_drain_down_drain_io_0.mem[3]); end
        if (u_drain_down_drain_io_0.mem[4] !== 8'd6) begin errors = errors + 1; $display("  Ct byte 4 = %0d, want 6", u_drain_down_drain_io_0.mem[4]); end
        if (u_drain_down_drain_io_0.mem[5] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 5 = %0d, want 0", u_drain_down_drain_io_0.mem[5]); end
        if (u_drain_down_drain_io_0.mem[6] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 6 = %0d, want 0", u_drain_down_drain_io_0.mem[6]); end
        if (u_drain_down_drain_io_0.mem[7] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 7 = %0d, want 0", u_drain_down_drain_io_0.mem[7]); end
        if (u_drain_down_drain_io_0.mem[8] !== 8'd5) begin errors = errors + 1; $display("  Ct byte 8 = %0d, want 5", u_drain_down_drain_io_0.mem[8]); end
        if (u_drain_down_drain_io_0.mem[9] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 9 = %0d, want 0", u_drain_down_drain_io_0.mem[9]); end
        if (u_drain_down_drain_io_0.mem[10] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 10 = %0d, want 0", u_drain_down_drain_io_0.mem[10]); end
        if (u_drain_down_drain_io_0.mem[11] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 11 = %0d, want 0", u_drain_down_drain_io_0.mem[11]); end
        if (u_drain_down_drain_io_0.mem[12] !== 8'd4) begin errors = errors + 1; $display("  Ct byte 12 = %0d, want 4", u_drain_down_drain_io_0.mem[12]); end
        if (u_drain_down_drain_io_0.mem[13] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 13 = %0d, want 0", u_drain_down_drain_io_0.mem[13]); end
        if (u_drain_down_drain_io_0.mem[14] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 14 = %0d, want 0", u_drain_down_drain_io_0.mem[14]); end
        if (u_drain_down_drain_io_0.mem[15] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 15 = %0d, want 0", u_drain_down_drain_io_0.mem[15]); end
        // drain_down_drain_io_1 -> Ct
        if (u_drain_down_drain_io_1.mem[16] !== 8'd8) begin errors = errors + 1; $display("  Ct byte 16 = %0d, want 8", u_drain_down_drain_io_1.mem[16]); end
        if (u_drain_down_drain_io_1.mem[17] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 17 = %0d, want 0", u_drain_down_drain_io_1.mem[17]); end
        if (u_drain_down_drain_io_1.mem[18] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 18 = %0d, want 0", u_drain_down_drain_io_1.mem[18]); end
        if (u_drain_down_drain_io_1.mem[19] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 19 = %0d, want 0", u_drain_down_drain_io_1.mem[19]); end
        if (u_drain_down_drain_io_1.mem[20] !== 8'd7) begin errors = errors + 1; $display("  Ct byte 20 = %0d, want 7", u_drain_down_drain_io_1.mem[20]); end
        if (u_drain_down_drain_io_1.mem[21] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 21 = %0d, want 0", u_drain_down_drain_io_1.mem[21]); end
        if (u_drain_down_drain_io_1.mem[22] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 22 = %0d, want 0", u_drain_down_drain_io_1.mem[22]); end
        if (u_drain_down_drain_io_1.mem[23] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 23 = %0d, want 0", u_drain_down_drain_io_1.mem[23]); end
        if (u_drain_down_drain_io_1.mem[24] !== 8'd7) begin errors = errors + 1; $display("  Ct byte 24 = %0d, want 7", u_drain_down_drain_io_1.mem[24]); end
        if (u_drain_down_drain_io_1.mem[25] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 25 = %0d, want 0", u_drain_down_drain_io_1.mem[25]); end
        if (u_drain_down_drain_io_1.mem[26] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 26 = %0d, want 0", u_drain_down_drain_io_1.mem[26]); end
        if (u_drain_down_drain_io_1.mem[27] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 27 = %0d, want 0", u_drain_down_drain_io_1.mem[27]); end
        if (u_drain_down_drain_io_1.mem[28] !== 8'd4) begin errors = errors + 1; $display("  Ct byte 28 = %0d, want 4", u_drain_down_drain_io_1.mem[28]); end
        if (u_drain_down_drain_io_1.mem[29] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 29 = %0d, want 0", u_drain_down_drain_io_1.mem[29]); end
        if (u_drain_down_drain_io_1.mem[30] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 30 = %0d, want 0", u_drain_down_drain_io_1.mem[30]); end
        if (u_drain_down_drain_io_1.mem[31] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 31 = %0d, want 0", u_drain_down_drain_io_1.mem[31]); end
        // drain_down_drain_io_2 -> Ct
        if (u_drain_down_drain_io_2.mem[32] !== 8'd6) begin errors = errors + 1; $display("  Ct byte 32 = %0d, want 6", u_drain_down_drain_io_2.mem[32]); end
        if (u_drain_down_drain_io_2.mem[33] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 33 = %0d, want 0", u_drain_down_drain_io_2.mem[33]); end
        if (u_drain_down_drain_io_2.mem[34] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 34 = %0d, want 0", u_drain_down_drain_io_2.mem[34]); end
        if (u_drain_down_drain_io_2.mem[35] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 35 = %0d, want 0", u_drain_down_drain_io_2.mem[35]); end
        if (u_drain_down_drain_io_2.mem[36] !== 8'd6) begin errors = errors + 1; $display("  Ct byte 36 = %0d, want 6", u_drain_down_drain_io_2.mem[36]); end
        if (u_drain_down_drain_io_2.mem[37] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 37 = %0d, want 0", u_drain_down_drain_io_2.mem[37]); end
        if (u_drain_down_drain_io_2.mem[38] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 38 = %0d, want 0", u_drain_down_drain_io_2.mem[38]); end
        if (u_drain_down_drain_io_2.mem[39] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 39 = %0d, want 0", u_drain_down_drain_io_2.mem[39]); end
        if (u_drain_down_drain_io_2.mem[40] !== 8'd5) begin errors = errors + 1; $display("  Ct byte 40 = %0d, want 5", u_drain_down_drain_io_2.mem[40]); end
        if (u_drain_down_drain_io_2.mem[41] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 41 = %0d, want 0", u_drain_down_drain_io_2.mem[41]); end
        if (u_drain_down_drain_io_2.mem[42] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 42 = %0d, want 0", u_drain_down_drain_io_2.mem[42]); end
        if (u_drain_down_drain_io_2.mem[43] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 43 = %0d, want 0", u_drain_down_drain_io_2.mem[43]); end
        if (u_drain_down_drain_io_2.mem[44] !== 8'd4) begin errors = errors + 1; $display("  Ct byte 44 = %0d, want 4", u_drain_down_drain_io_2.mem[44]); end
        if (u_drain_down_drain_io_2.mem[45] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 45 = %0d, want 0", u_drain_down_drain_io_2.mem[45]); end
        if (u_drain_down_drain_io_2.mem[46] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 46 = %0d, want 0", u_drain_down_drain_io_2.mem[46]); end
        if (u_drain_down_drain_io_2.mem[47] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 47 = %0d, want 0", u_drain_down_drain_io_2.mem[47]); end
        // drain_down_drain_io_3 -> Ct
        if (u_drain_down_drain_io_3.mem[48] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 48 = %0d, want 0", u_drain_down_drain_io_3.mem[48]); end
        if (u_drain_down_drain_io_3.mem[49] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 49 = %0d, want 0", u_drain_down_drain_io_3.mem[49]); end
        if (u_drain_down_drain_io_3.mem[50] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 50 = %0d, want 0", u_drain_down_drain_io_3.mem[50]); end
        if (u_drain_down_drain_io_3.mem[51] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 51 = %0d, want 0", u_drain_down_drain_io_3.mem[51]); end
        if (u_drain_down_drain_io_3.mem[52] !== 8'd2) begin errors = errors + 1; $display("  Ct byte 52 = %0d, want 2", u_drain_down_drain_io_3.mem[52]); end
        if (u_drain_down_drain_io_3.mem[53] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 53 = %0d, want 0", u_drain_down_drain_io_3.mem[53]); end
        if (u_drain_down_drain_io_3.mem[54] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 54 = %0d, want 0", u_drain_down_drain_io_3.mem[54]); end
        if (u_drain_down_drain_io_3.mem[55] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 55 = %0d, want 0", u_drain_down_drain_io_3.mem[55]); end
        if (u_drain_down_drain_io_3.mem[56] !== 8'd2) begin errors = errors + 1; $display("  Ct byte 56 = %0d, want 2", u_drain_down_drain_io_3.mem[56]); end
        if (u_drain_down_drain_io_3.mem[57] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 57 = %0d, want 0", u_drain_down_drain_io_3.mem[57]); end
        if (u_drain_down_drain_io_3.mem[58] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 58 = %0d, want 0", u_drain_down_drain_io_3.mem[58]); end
        if (u_drain_down_drain_io_3.mem[59] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 59 = %0d, want 0", u_drain_down_drain_io_3.mem[59]); end
        if (u_drain_down_drain_io_3.mem[60] !== 8'd4) begin errors = errors + 1; $display("  Ct byte 60 = %0d, want 4", u_drain_down_drain_io_3.mem[60]); end
        if (u_drain_down_drain_io_3.mem[61] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 61 = %0d, want 0", u_drain_down_drain_io_3.mem[61]); end
        if (u_drain_down_drain_io_3.mem[62] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 62 = %0d, want 0", u_drain_down_drain_io_3.mem[62]); end
        if (u_drain_down_drain_io_3.mem[63] !== 8'd0) begin errors = errors + 1; $display("  Ct byte 63 = %0d, want 0", u_drain_down_drain_io_3.mem[63]); end
        $display("SPMW MASTER feed_up_load_io_0 read done=%0d", feed_up_load_io_0_at < 0 ? cycle : feed_up_load_io_0_at);
        $display("SPMW MASTER feed_3_up_load_io_0 read done=%0d", feed_3_up_load_io_0_at < 0 ? cycle : feed_3_up_load_io_0_at);
        $display("SPMW MASTER drain_down_drain_io_0 write done=%0d", drain_down_drain_io_0_at < 0 ? cycle : drain_down_drain_io_0_at);
        $display("SPMW MASTER drain_down_drain_io_1 write done=%0d", drain_down_drain_io_1_at < 0 ? cycle : drain_down_drain_io_1_at);
        $display("SPMW MASTER drain_down_drain_io_2 write done=%0d", drain_down_drain_io_2_at < 0 ? cycle : drain_down_drain_io_2_at);
        $display("SPMW MASTER drain_down_drain_io_3 write done=%0d", drain_down_drain_io_3_at < 0 ? cycle : drain_down_drain_io_3_at);
        $display("SPMW COSIM %s (%0d errors)",
                 errors == 0 ? "PASS" : "FAIL", errors);
        $display("SPMW MEMORY CYCLES total=%0d latency=0",
                 cycle + 1);
        $finish;
      end
    end
    $display("SPMW COSIM TIMEOUT");
    $display("SPMW MEMORY CYCLES total=-1 latency=%0d", 0);
    $finish;
  end
endmodule

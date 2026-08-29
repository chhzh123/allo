`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire ap_start,
  output wire ap_done,
  input  wire [63:0] feed_up_load_io_0_offset,
  output wire m_axi_feed_up_load_io_0_AWVALID,
  input  wire m_axi_feed_up_load_io_0_AWREADY,
  output wire [63:0] m_axi_feed_up_load_io_0_AWADDR,
  output wire m_axi_feed_up_load_io_0_AWID,
  output wire [7:0] m_axi_feed_up_load_io_0_AWLEN,
  output wire [2:0] m_axi_feed_up_load_io_0_AWSIZE,
  output wire [1:0] m_axi_feed_up_load_io_0_AWBURST,
  output wire [1:0] m_axi_feed_up_load_io_0_AWLOCK,
  output wire [3:0] m_axi_feed_up_load_io_0_AWCACHE,
  output wire [2:0] m_axi_feed_up_load_io_0_AWPROT,
  output wire [3:0] m_axi_feed_up_load_io_0_AWQOS,
  output wire [3:0] m_axi_feed_up_load_io_0_AWREGION,
  output wire m_axi_feed_up_load_io_0_AWUSER,
  output wire m_axi_feed_up_load_io_0_WVALID,
  input  wire m_axi_feed_up_load_io_0_WREADY,
  output wire [7:0] m_axi_feed_up_load_io_0_WDATA,
  output wire m_axi_feed_up_load_io_0_WSTRB,
  output wire m_axi_feed_up_load_io_0_WLAST,
  output wire m_axi_feed_up_load_io_0_WID,
  output wire m_axi_feed_up_load_io_0_WUSER,
  output wire m_axi_feed_up_load_io_0_ARVALID,
  input  wire m_axi_feed_up_load_io_0_ARREADY,
  output wire [63:0] m_axi_feed_up_load_io_0_ARADDR,
  output wire m_axi_feed_up_load_io_0_ARID,
  output wire [7:0] m_axi_feed_up_load_io_0_ARLEN,
  output wire [2:0] m_axi_feed_up_load_io_0_ARSIZE,
  output wire [1:0] m_axi_feed_up_load_io_0_ARBURST,
  output wire [1:0] m_axi_feed_up_load_io_0_ARLOCK,
  output wire [3:0] m_axi_feed_up_load_io_0_ARCACHE,
  output wire [2:0] m_axi_feed_up_load_io_0_ARPROT,
  output wire [3:0] m_axi_feed_up_load_io_0_ARQOS,
  output wire [3:0] m_axi_feed_up_load_io_0_ARREGION,
  output wire m_axi_feed_up_load_io_0_ARUSER,
  input  wire m_axi_feed_up_load_io_0_RVALID,
  output wire m_axi_feed_up_load_io_0_RREADY,
  input  wire [7:0] m_axi_feed_up_load_io_0_RDATA,
  input  wire m_axi_feed_up_load_io_0_RLAST,
  input  wire m_axi_feed_up_load_io_0_RID,
  input  wire m_axi_feed_up_load_io_0_RUSER,
  input  wire [1:0] m_axi_feed_up_load_io_0_RRESP,
  input  wire m_axi_feed_up_load_io_0_BVALID,
  output wire m_axi_feed_up_load_io_0_BREADY,
  input  wire [1:0] m_axi_feed_up_load_io_0_BRESP,
  input  wire m_axi_feed_up_load_io_0_BID,
  input  wire m_axi_feed_up_load_io_0_BUSER,
  input  wire [63:0] feed_2_up_load_io_0_offset,
  output wire m_axi_feed_2_up_load_io_0_AWVALID,
  input  wire m_axi_feed_2_up_load_io_0_AWREADY,
  output wire [63:0] m_axi_feed_2_up_load_io_0_AWADDR,
  output wire m_axi_feed_2_up_load_io_0_AWID,
  output wire [7:0] m_axi_feed_2_up_load_io_0_AWLEN,
  output wire [2:0] m_axi_feed_2_up_load_io_0_AWSIZE,
  output wire [1:0] m_axi_feed_2_up_load_io_0_AWBURST,
  output wire [1:0] m_axi_feed_2_up_load_io_0_AWLOCK,
  output wire [3:0] m_axi_feed_2_up_load_io_0_AWCACHE,
  output wire [2:0] m_axi_feed_2_up_load_io_0_AWPROT,
  output wire [3:0] m_axi_feed_2_up_load_io_0_AWQOS,
  output wire [3:0] m_axi_feed_2_up_load_io_0_AWREGION,
  output wire m_axi_feed_2_up_load_io_0_AWUSER,
  output wire m_axi_feed_2_up_load_io_0_WVALID,
  input  wire m_axi_feed_2_up_load_io_0_WREADY,
  output wire [7:0] m_axi_feed_2_up_load_io_0_WDATA,
  output wire m_axi_feed_2_up_load_io_0_WSTRB,
  output wire m_axi_feed_2_up_load_io_0_WLAST,
  output wire m_axi_feed_2_up_load_io_0_WID,
  output wire m_axi_feed_2_up_load_io_0_WUSER,
  output wire m_axi_feed_2_up_load_io_0_ARVALID,
  input  wire m_axi_feed_2_up_load_io_0_ARREADY,
  output wire [63:0] m_axi_feed_2_up_load_io_0_ARADDR,
  output wire m_axi_feed_2_up_load_io_0_ARID,
  output wire [7:0] m_axi_feed_2_up_load_io_0_ARLEN,
  output wire [2:0] m_axi_feed_2_up_load_io_0_ARSIZE,
  output wire [1:0] m_axi_feed_2_up_load_io_0_ARBURST,
  output wire [1:0] m_axi_feed_2_up_load_io_0_ARLOCK,
  output wire [3:0] m_axi_feed_2_up_load_io_0_ARCACHE,
  output wire [2:0] m_axi_feed_2_up_load_io_0_ARPROT,
  output wire [3:0] m_axi_feed_2_up_load_io_0_ARQOS,
  output wire [3:0] m_axi_feed_2_up_load_io_0_ARREGION,
  output wire m_axi_feed_2_up_load_io_0_ARUSER,
  input  wire m_axi_feed_2_up_load_io_0_RVALID,
  output wire m_axi_feed_2_up_load_io_0_RREADY,
  input  wire [7:0] m_axi_feed_2_up_load_io_0_RDATA,
  input  wire m_axi_feed_2_up_load_io_0_RLAST,
  input  wire m_axi_feed_2_up_load_io_0_RID,
  input  wire m_axi_feed_2_up_load_io_0_RUSER,
  input  wire [1:0] m_axi_feed_2_up_load_io_0_RRESP,
  input  wire m_axi_feed_2_up_load_io_0_BVALID,
  output wire m_axi_feed_2_up_load_io_0_BREADY,
  input  wire [1:0] m_axi_feed_2_up_load_io_0_BRESP,
  input  wire m_axi_feed_2_up_load_io_0_BID,
  input  wire m_axi_feed_2_up_load_io_0_BUSER,
  input  wire [63:0] pe_c_out_drain_io_0_offset,
  output wire m_axi_pe_c_out_drain_io_0_AWVALID,
  input  wire m_axi_pe_c_out_drain_io_0_AWREADY,
  output wire [63:0] m_axi_pe_c_out_drain_io_0_AWADDR,
  output wire m_axi_pe_c_out_drain_io_0_AWID,
  output wire [7:0] m_axi_pe_c_out_drain_io_0_AWLEN,
  output wire [2:0] m_axi_pe_c_out_drain_io_0_AWSIZE,
  output wire [1:0] m_axi_pe_c_out_drain_io_0_AWBURST,
  output wire [1:0] m_axi_pe_c_out_drain_io_0_AWLOCK,
  output wire [3:0] m_axi_pe_c_out_drain_io_0_AWCACHE,
  output wire [2:0] m_axi_pe_c_out_drain_io_0_AWPROT,
  output wire [3:0] m_axi_pe_c_out_drain_io_0_AWQOS,
  output wire [3:0] m_axi_pe_c_out_drain_io_0_AWREGION,
  output wire m_axi_pe_c_out_drain_io_0_AWUSER,
  output wire m_axi_pe_c_out_drain_io_0_WVALID,
  input  wire m_axi_pe_c_out_drain_io_0_WREADY,
  output wire [31:0] m_axi_pe_c_out_drain_io_0_WDATA,
  output wire [3:0] m_axi_pe_c_out_drain_io_0_WSTRB,
  output wire m_axi_pe_c_out_drain_io_0_WLAST,
  output wire m_axi_pe_c_out_drain_io_0_WID,
  output wire m_axi_pe_c_out_drain_io_0_WUSER,
  output wire m_axi_pe_c_out_drain_io_0_ARVALID,
  input  wire m_axi_pe_c_out_drain_io_0_ARREADY,
  output wire [63:0] m_axi_pe_c_out_drain_io_0_ARADDR,
  output wire m_axi_pe_c_out_drain_io_0_ARID,
  output wire [7:0] m_axi_pe_c_out_drain_io_0_ARLEN,
  output wire [2:0] m_axi_pe_c_out_drain_io_0_ARSIZE,
  output wire [1:0] m_axi_pe_c_out_drain_io_0_ARBURST,
  output wire [1:0] m_axi_pe_c_out_drain_io_0_ARLOCK,
  output wire [3:0] m_axi_pe_c_out_drain_io_0_ARCACHE,
  output wire [2:0] m_axi_pe_c_out_drain_io_0_ARPROT,
  output wire [3:0] m_axi_pe_c_out_drain_io_0_ARQOS,
  output wire [3:0] m_axi_pe_c_out_drain_io_0_ARREGION,
  output wire m_axi_pe_c_out_drain_io_0_ARUSER,
  input  wire m_axi_pe_c_out_drain_io_0_RVALID,
  output wire m_axi_pe_c_out_drain_io_0_RREADY,
  input  wire [31:0] m_axi_pe_c_out_drain_io_0_RDATA,
  input  wire m_axi_pe_c_out_drain_io_0_RLAST,
  input  wire m_axi_pe_c_out_drain_io_0_RID,
  input  wire m_axi_pe_c_out_drain_io_0_RUSER,
  input  wire [1:0] m_axi_pe_c_out_drain_io_0_RRESP,
  input  wire m_axi_pe_c_out_drain_io_0_BVALID,
  output wire m_axi_pe_c_out_drain_io_0_BREADY,
  input  wire [1:0] m_axi_pe_c_out_drain_io_0_BRESP,
  input  wire m_axi_pe_c_out_drain_io_0_BID,
  input  wire m_axi_pe_c_out_drain_io_0_BUSER,
  input  wire [63:0] pe_c_out_drain_io_1_offset,
  output wire m_axi_pe_c_out_drain_io_1_AWVALID,
  input  wire m_axi_pe_c_out_drain_io_1_AWREADY,
  output wire [63:0] m_axi_pe_c_out_drain_io_1_AWADDR,
  output wire m_axi_pe_c_out_drain_io_1_AWID,
  output wire [7:0] m_axi_pe_c_out_drain_io_1_AWLEN,
  output wire [2:0] m_axi_pe_c_out_drain_io_1_AWSIZE,
  output wire [1:0] m_axi_pe_c_out_drain_io_1_AWBURST,
  output wire [1:0] m_axi_pe_c_out_drain_io_1_AWLOCK,
  output wire [3:0] m_axi_pe_c_out_drain_io_1_AWCACHE,
  output wire [2:0] m_axi_pe_c_out_drain_io_1_AWPROT,
  output wire [3:0] m_axi_pe_c_out_drain_io_1_AWQOS,
  output wire [3:0] m_axi_pe_c_out_drain_io_1_AWREGION,
  output wire m_axi_pe_c_out_drain_io_1_AWUSER,
  output wire m_axi_pe_c_out_drain_io_1_WVALID,
  input  wire m_axi_pe_c_out_drain_io_1_WREADY,
  output wire [31:0] m_axi_pe_c_out_drain_io_1_WDATA,
  output wire [3:0] m_axi_pe_c_out_drain_io_1_WSTRB,
  output wire m_axi_pe_c_out_drain_io_1_WLAST,
  output wire m_axi_pe_c_out_drain_io_1_WID,
  output wire m_axi_pe_c_out_drain_io_1_WUSER,
  output wire m_axi_pe_c_out_drain_io_1_ARVALID,
  input  wire m_axi_pe_c_out_drain_io_1_ARREADY,
  output wire [63:0] m_axi_pe_c_out_drain_io_1_ARADDR,
  output wire m_axi_pe_c_out_drain_io_1_ARID,
  output wire [7:0] m_axi_pe_c_out_drain_io_1_ARLEN,
  output wire [2:0] m_axi_pe_c_out_drain_io_1_ARSIZE,
  output wire [1:0] m_axi_pe_c_out_drain_io_1_ARBURST,
  output wire [1:0] m_axi_pe_c_out_drain_io_1_ARLOCK,
  output wire [3:0] m_axi_pe_c_out_drain_io_1_ARCACHE,
  output wire [2:0] m_axi_pe_c_out_drain_io_1_ARPROT,
  output wire [3:0] m_axi_pe_c_out_drain_io_1_ARQOS,
  output wire [3:0] m_axi_pe_c_out_drain_io_1_ARREGION,
  output wire m_axi_pe_c_out_drain_io_1_ARUSER,
  input  wire m_axi_pe_c_out_drain_io_1_RVALID,
  output wire m_axi_pe_c_out_drain_io_1_RREADY,
  input  wire [31:0] m_axi_pe_c_out_drain_io_1_RDATA,
  input  wire m_axi_pe_c_out_drain_io_1_RLAST,
  input  wire m_axi_pe_c_out_drain_io_1_RID,
  input  wire m_axi_pe_c_out_drain_io_1_RUSER,
  input  wire [1:0] m_axi_pe_c_out_drain_io_1_RRESP,
  input  wire m_axi_pe_c_out_drain_io_1_BVALID,
  output wire m_axi_pe_c_out_drain_io_1_BREADY,
  input  wire [1:0] m_axi_pe_c_out_drain_io_1_BRESP,
  input  wire m_axi_pe_c_out_drain_io_1_BID,
  input  wire m_axi_pe_c_out_drain_io_1_BUSER,
  input  wire [63:0] pe_c_out_drain_io_2_offset,
  output wire m_axi_pe_c_out_drain_io_2_AWVALID,
  input  wire m_axi_pe_c_out_drain_io_2_AWREADY,
  output wire [63:0] m_axi_pe_c_out_drain_io_2_AWADDR,
  output wire m_axi_pe_c_out_drain_io_2_AWID,
  output wire [7:0] m_axi_pe_c_out_drain_io_2_AWLEN,
  output wire [2:0] m_axi_pe_c_out_drain_io_2_AWSIZE,
  output wire [1:0] m_axi_pe_c_out_drain_io_2_AWBURST,
  output wire [1:0] m_axi_pe_c_out_drain_io_2_AWLOCK,
  output wire [3:0] m_axi_pe_c_out_drain_io_2_AWCACHE,
  output wire [2:0] m_axi_pe_c_out_drain_io_2_AWPROT,
  output wire [3:0] m_axi_pe_c_out_drain_io_2_AWQOS,
  output wire [3:0] m_axi_pe_c_out_drain_io_2_AWREGION,
  output wire m_axi_pe_c_out_drain_io_2_AWUSER,
  output wire m_axi_pe_c_out_drain_io_2_WVALID,
  input  wire m_axi_pe_c_out_drain_io_2_WREADY,
  output wire [31:0] m_axi_pe_c_out_drain_io_2_WDATA,
  output wire [3:0] m_axi_pe_c_out_drain_io_2_WSTRB,
  output wire m_axi_pe_c_out_drain_io_2_WLAST,
  output wire m_axi_pe_c_out_drain_io_2_WID,
  output wire m_axi_pe_c_out_drain_io_2_WUSER,
  output wire m_axi_pe_c_out_drain_io_2_ARVALID,
  input  wire m_axi_pe_c_out_drain_io_2_ARREADY,
  output wire [63:0] m_axi_pe_c_out_drain_io_2_ARADDR,
  output wire m_axi_pe_c_out_drain_io_2_ARID,
  output wire [7:0] m_axi_pe_c_out_drain_io_2_ARLEN,
  output wire [2:0] m_axi_pe_c_out_drain_io_2_ARSIZE,
  output wire [1:0] m_axi_pe_c_out_drain_io_2_ARBURST,
  output wire [1:0] m_axi_pe_c_out_drain_io_2_ARLOCK,
  output wire [3:0] m_axi_pe_c_out_drain_io_2_ARCACHE,
  output wire [2:0] m_axi_pe_c_out_drain_io_2_ARPROT,
  output wire [3:0] m_axi_pe_c_out_drain_io_2_ARQOS,
  output wire [3:0] m_axi_pe_c_out_drain_io_2_ARREGION,
  output wire m_axi_pe_c_out_drain_io_2_ARUSER,
  input  wire m_axi_pe_c_out_drain_io_2_RVALID,
  output wire m_axi_pe_c_out_drain_io_2_RREADY,
  input  wire [31:0] m_axi_pe_c_out_drain_io_2_RDATA,
  input  wire m_axi_pe_c_out_drain_io_2_RLAST,
  input  wire m_axi_pe_c_out_drain_io_2_RID,
  input  wire m_axi_pe_c_out_drain_io_2_RUSER,
  input  wire [1:0] m_axi_pe_c_out_drain_io_2_RRESP,
  input  wire m_axi_pe_c_out_drain_io_2_BVALID,
  output wire m_axi_pe_c_out_drain_io_2_BREADY,
  input  wire [1:0] m_axi_pe_c_out_drain_io_2_BRESP,
  input  wire m_axi_pe_c_out_drain_io_2_BID,
  input  wire m_axi_pe_c_out_drain_io_2_BUSER,
  input  wire [63:0] pe_c_out_drain_io_3_offset,
  output wire m_axi_pe_c_out_drain_io_3_AWVALID,
  input  wire m_axi_pe_c_out_drain_io_3_AWREADY,
  output wire [63:0] m_axi_pe_c_out_drain_io_3_AWADDR,
  output wire m_axi_pe_c_out_drain_io_3_AWID,
  output wire [7:0] m_axi_pe_c_out_drain_io_3_AWLEN,
  output wire [2:0] m_axi_pe_c_out_drain_io_3_AWSIZE,
  output wire [1:0] m_axi_pe_c_out_drain_io_3_AWBURST,
  output wire [1:0] m_axi_pe_c_out_drain_io_3_AWLOCK,
  output wire [3:0] m_axi_pe_c_out_drain_io_3_AWCACHE,
  output wire [2:0] m_axi_pe_c_out_drain_io_3_AWPROT,
  output wire [3:0] m_axi_pe_c_out_drain_io_3_AWQOS,
  output wire [3:0] m_axi_pe_c_out_drain_io_3_AWREGION,
  output wire m_axi_pe_c_out_drain_io_3_AWUSER,
  output wire m_axi_pe_c_out_drain_io_3_WVALID,
  input  wire m_axi_pe_c_out_drain_io_3_WREADY,
  output wire [31:0] m_axi_pe_c_out_drain_io_3_WDATA,
  output wire [3:0] m_axi_pe_c_out_drain_io_3_WSTRB,
  output wire m_axi_pe_c_out_drain_io_3_WLAST,
  output wire m_axi_pe_c_out_drain_io_3_WID,
  output wire m_axi_pe_c_out_drain_io_3_WUSER,
  output wire m_axi_pe_c_out_drain_io_3_ARVALID,
  input  wire m_axi_pe_c_out_drain_io_3_ARREADY,
  output wire [63:0] m_axi_pe_c_out_drain_io_3_ARADDR,
  output wire m_axi_pe_c_out_drain_io_3_ARID,
  output wire [7:0] m_axi_pe_c_out_drain_io_3_ARLEN,
  output wire [2:0] m_axi_pe_c_out_drain_io_3_ARSIZE,
  output wire [1:0] m_axi_pe_c_out_drain_io_3_ARBURST,
  output wire [1:0] m_axi_pe_c_out_drain_io_3_ARLOCK,
  output wire [3:0] m_axi_pe_c_out_drain_io_3_ARCACHE,
  output wire [2:0] m_axi_pe_c_out_drain_io_3_ARPROT,
  output wire [3:0] m_axi_pe_c_out_drain_io_3_ARQOS,
  output wire [3:0] m_axi_pe_c_out_drain_io_3_ARREGION,
  output wire m_axi_pe_c_out_drain_io_3_ARUSER,
  input  wire m_axi_pe_c_out_drain_io_3_RVALID,
  output wire m_axi_pe_c_out_drain_io_3_RREADY,
  input  wire [31:0] m_axi_pe_c_out_drain_io_3_RDATA,
  input  wire m_axi_pe_c_out_drain_io_3_RLAST,
  input  wire m_axi_pe_c_out_drain_io_3_RID,
  input  wire m_axi_pe_c_out_drain_io_3_RUSER,
  input  wire [1:0] m_axi_pe_c_out_drain_io_3_RRESP,
  input  wire m_axi_pe_c_out_drain_io_3_BVALID,
  output wire m_axi_pe_c_out_drain_io_3_BREADY,
  input  wire [1:0] m_axi_pe_c_out_drain_io_3_BRESP,
  input  wire m_axi_pe_c_out_drain_io_3_BID,
  input  wire m_axi_pe_c_out_drain_io_3_BUSER
);
  // family pe_east_west: 16 channel(s), 8-bit, depth 2
  wire [7:0] pe_east_west_din [0:15];
  wire [7:0] pe_east_west_dout [0:15];
  wire pe_east_west_full_n [0:15];
  wire pe_east_west_write [0:15];
  wire pe_east_west_empty_n [0:15];
  wire pe_east_west_read [0:15];
  genvar pe_east_west_i;
  generate
    for (pe_east_west_i = 0; pe_east_west_i < 16; pe_east_west_i = pe_east_west_i + 1) begin : g_pe_east_west
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_east_west_din[pe_east_west_i]), .full_n(pe_east_west_full_n[pe_east_west_i]), .write(pe_east_west_write[pe_east_west_i]), .dout(pe_east_west_dout[pe_east_west_i]), .empty_n(pe_east_west_empty_n[pe_east_west_i]), .read(pe_east_west_read[pe_east_west_i]));
    end
  endgenerate
  // family pe_south_north: 16 channel(s), 8-bit, depth 2
  wire [7:0] pe_south_north_din [0:15];
  wire [7:0] pe_south_north_dout [0:15];
  wire pe_south_north_full_n [0:15];
  wire pe_south_north_write [0:15];
  wire pe_south_north_empty_n [0:15];
  wire pe_south_north_read [0:15];
  genvar pe_south_north_i;
  generate
    for (pe_south_north_i = 0; pe_south_north_i < 16; pe_south_north_i = pe_south_north_i + 1) begin : g_pe_south_north
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_south_north_din[pe_south_north_i]), .full_n(pe_south_north_full_n[pe_south_north_i]), .write(pe_south_north_write[pe_south_north_i]), .dout(pe_south_north_dout[pe_south_north_i]), .empty_n(pe_south_north_empty_n[pe_south_north_i]), .read(pe_south_north_read[pe_south_north_i]));
    end
  endgenerate
  // family pe_c_out_c_in: 16 channel(s), 32-bit, depth 2
  wire [31:0] pe_c_out_c_in_din [0:15];
  wire [31:0] pe_c_out_c_in_dout [0:15];
  wire pe_c_out_c_in_full_n [0:15];
  wire pe_c_out_c_in_write [0:15];
  wire pe_c_out_c_in_empty_n [0:15];
  wire pe_c_out_c_in_read [0:15];
  genvar pe_c_out_c_in_i;
  generate
    for (pe_c_out_c_in_i = 0; pe_c_out_c_in_i < 16; pe_c_out_c_in_i = pe_c_out_c_in_i + 1) begin : g_pe_c_out_c_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_c_out_c_in_din[pe_c_out_c_in_i]), .full_n(pe_c_out_c_in_full_n[pe_c_out_c_in_i]), .write(pe_c_out_c_in_write[pe_c_out_c_in_i]), .dout(pe_c_out_c_in_dout[pe_c_out_c_in_i]), .empty_n(pe_c_out_c_in_empty_n[pe_c_out_c_in_i]), .read(pe_c_out_c_in_read[pe_c_out_c_in_i]));
    end
  endgenerate
  // family pe_west_bind: 4 channel(s), 8-bit, depth 2
  wire [7:0] pe_west_bind_din [0:3];
  wire [7:0] pe_west_bind_dout [0:3];
  wire pe_west_bind_full_n [0:3];
  wire pe_west_bind_write [0:3];
  wire pe_west_bind_empty_n [0:3];
  wire pe_west_bind_read [0:3];
  genvar pe_west_bind_i;
  generate
    for (pe_west_bind_i = 0; pe_west_bind_i < 4; pe_west_bind_i = pe_west_bind_i + 1) begin : g_pe_west_bind
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_west_bind_din[pe_west_bind_i]), .full_n(pe_west_bind_full_n[pe_west_bind_i]), .write(pe_west_bind_write[pe_west_bind_i]), .dout(pe_west_bind_dout[pe_west_bind_i]), .empty_n(pe_west_bind_empty_n[pe_west_bind_i]), .read(pe_west_bind_read[pe_west_bind_i]));
    end
  endgenerate
  // family pe_north_bind: 4 channel(s), 8-bit, depth 2
  wire [7:0] pe_north_bind_din [0:3];
  wire [7:0] pe_north_bind_dout [0:3];
  wire pe_north_bind_full_n [0:3];
  wire pe_north_bind_write [0:3];
  wire pe_north_bind_empty_n [0:3];
  wire pe_north_bind_read [0:3];
  genvar pe_north_bind_i;
  generate
    for (pe_north_bind_i = 0; pe_north_bind_i < 4; pe_north_bind_i = pe_north_bind_i + 1) begin : g_pe_north_bind
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_north_bind_din[pe_north_bind_i]), .full_n(pe_north_bind_full_n[pe_north_bind_i]), .write(pe_north_bind_write[pe_north_bind_i]), .dout(pe_north_bind_dout[pe_north_bind_i]), .empty_n(pe_north_bind_empty_n[pe_north_bind_i]), .read(pe_north_bind_read[pe_north_bind_i]));
    end
  endgenerate
  // family feed_down_up: 4 channel(s), 32-bit, depth 2
  wire [31:0] feed_down_up_din [0:3];
  wire [31:0] feed_down_up_dout [0:3];
  wire feed_down_up_full_n [0:3];
  wire feed_down_up_write [0:3];
  wire feed_down_up_empty_n [0:3];
  wire feed_down_up_read [0:3];
  genvar feed_down_up_i;
  generate
    for (feed_down_up_i = 0; feed_down_up_i < 4; feed_down_up_i = feed_down_up_i + 1) begin : g_feed_down_up
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(feed_down_up_din[feed_down_up_i]), .full_n(feed_down_up_full_n[feed_down_up_i]), .write(feed_down_up_write[feed_down_up_i]), .dout(feed_down_up_dout[feed_down_up_i]), .empty_n(feed_down_up_empty_n[feed_down_up_i]), .read(feed_down_up_read[feed_down_up_i]));
    end
  endgenerate
  // family feed_2_down_up: 4 channel(s), 32-bit, depth 2
  wire [31:0] feed_2_down_up_din [0:3];
  wire [31:0] feed_2_down_up_dout [0:3];
  wire feed_2_down_up_full_n [0:3];
  wire feed_2_down_up_write [0:3];
  wire feed_2_down_up_empty_n [0:3];
  wire feed_2_down_up_read [0:3];
  genvar feed_2_down_up_i;
  generate
    for (feed_2_down_up_i = 0; feed_2_down_up_i < 4; feed_2_down_up_i = feed_2_down_up_i + 1) begin : g_feed_2_down_up
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(feed_2_down_up_din[feed_2_down_up_i]), .full_n(feed_2_down_up_full_n[feed_2_down_up_i]), .write(feed_2_down_up_write[feed_2_down_up_i]), .dout(feed_2_down_up_dout[feed_2_down_up_i]), .empty_n(feed_2_down_up_empty_n[feed_2_down_up_i]), .read(feed_2_down_up_read[feed_2_down_up_i]));
    end
  endgenerate
  // family pe_c_out_bind: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_c_out_bind_din [0:3];
  wire [31:0] pe_c_out_bind_dout [0:3];
  wire pe_c_out_bind_full_n [0:3];
  wire pe_c_out_bind_write [0:3];
  wire pe_c_out_bind_empty_n [0:3];
  wire pe_c_out_bind_read [0:3];
  genvar pe_c_out_bind_i;
  generate
    for (pe_c_out_bind_i = 0; pe_c_out_bind_i < 4; pe_c_out_bind_i = pe_c_out_bind_i + 1) begin : g_pe_c_out_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_c_out_bind_din[pe_c_out_bind_i]), .full_n(pe_c_out_bind_full_n[pe_c_out_bind_i]), .write(pe_c_out_bind_write[pe_c_out_bind_i]), .dout(pe_c_out_bind_dout[pe_c_out_bind_i]), .empty_n(pe_c_out_bind_empty_n[pe_c_out_bind_i]), .read(pe_c_out_bind_read[pe_c_out_bind_i]));
    end
  endgenerate
  // family feed_up_bind: 1 channel(s), 32-bit, depth 2
  wire [31:0] feed_up_bind_din [0:0];
  wire [31:0] feed_up_bind_dout [0:0];
  wire feed_up_bind_full_n [0:0];
  wire feed_up_bind_write [0:0];
  wire feed_up_bind_empty_n [0:0];
  wire feed_up_bind_read [0:0];
  genvar feed_up_bind_i;
  generate
    for (feed_up_bind_i = 0; feed_up_bind_i < 1; feed_up_bind_i = feed_up_bind_i + 1) begin : g_feed_up_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(feed_up_bind_din[feed_up_bind_i]), .full_n(feed_up_bind_full_n[feed_up_bind_i]), .write(feed_up_bind_write[feed_up_bind_i]), .dout(feed_up_bind_dout[feed_up_bind_i]), .empty_n(feed_up_bind_empty_n[feed_up_bind_i]), .read(feed_up_bind_read[feed_up_bind_i]));
    end
  endgenerate
  // family feed_2_up_bind: 1 channel(s), 32-bit, depth 2
  wire [31:0] feed_2_up_bind_din [0:0];
  wire [31:0] feed_2_up_bind_dout [0:0];
  wire feed_2_up_bind_full_n [0:0];
  wire feed_2_up_bind_write [0:0];
  wire feed_2_up_bind_empty_n [0:0];
  wire feed_2_up_bind_read [0:0];
  genvar feed_2_up_bind_i;
  generate
    for (feed_2_up_bind_i = 0; feed_2_up_bind_i < 1; feed_2_up_bind_i = feed_2_up_bind_i + 1) begin : g_feed_2_up_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(feed_2_up_bind_din[feed_2_up_bind_i]), .full_n(feed_2_up_bind_full_n[feed_2_up_bind_i]), .write(feed_2_up_bind_write[feed_2_up_bind_i]), .dout(feed_2_up_bind_dout[feed_2_up_bind_i]), .empty_n(feed_2_up_bind_empty_n[feed_2_up_bind_i]), .read(feed_2_up_bind_read[feed_2_up_bind_i]));
    end
  endgenerate
  wire feed_up_load_io_0_done, feed_up_load_io_0_idle, feed_up_load_io_0_ready;
  reg  feed_up_load_io_0_done_r, feed_up_load_io_0_run_r;
  wire feed_up_load_io_0_start = ap_start & ~feed_up_load_io_0_run_r;
  always @(posedge ap_clk)
    if (!ap_rst_n) begin
      feed_up_load_io_0_done_r <= 1'b0; feed_up_load_io_0_run_r <= 1'b0;
    end else begin
      if (feed_up_load_io_0_start && feed_up_load_io_0_ready) feed_up_load_io_0_run_r <= 1'b1;
      if (feed_up_load_io_0_done) feed_up_load_io_0_done_r <= 1'b1;
    end
  wire feed_2_up_load_io_0_done, feed_2_up_load_io_0_idle, feed_2_up_load_io_0_ready;
  reg  feed_2_up_load_io_0_done_r, feed_2_up_load_io_0_run_r;
  wire feed_2_up_load_io_0_start = ap_start & ~feed_2_up_load_io_0_run_r;
  always @(posedge ap_clk)
    if (!ap_rst_n) begin
      feed_2_up_load_io_0_done_r <= 1'b0; feed_2_up_load_io_0_run_r <= 1'b0;
    end else begin
      if (feed_2_up_load_io_0_start && feed_2_up_load_io_0_ready) feed_2_up_load_io_0_run_r <= 1'b1;
      if (feed_2_up_load_io_0_done) feed_2_up_load_io_0_done_r <= 1'b1;
    end
  wire pe_c_out_drain_io_0_done, pe_c_out_drain_io_0_idle, pe_c_out_drain_io_0_ready;
  reg  pe_c_out_drain_io_0_done_r, pe_c_out_drain_io_0_run_r;
  wire pe_c_out_drain_io_0_start = ap_start & ~pe_c_out_drain_io_0_run_r;
  always @(posedge ap_clk)
    if (!ap_rst_n) begin
      pe_c_out_drain_io_0_done_r <= 1'b0; pe_c_out_drain_io_0_run_r <= 1'b0;
    end else begin
      if (pe_c_out_drain_io_0_start && pe_c_out_drain_io_0_ready) pe_c_out_drain_io_0_run_r <= 1'b1;
      if (pe_c_out_drain_io_0_done) pe_c_out_drain_io_0_done_r <= 1'b1;
    end
  wire [31:0] pe_c_out_drain_io_0_pid0_dout; wire pe_c_out_drain_io_0_pid0_empty_n, pe_c_out_drain_io_0_pid0_read;
  wire pe_c_out_drain_io_1_done, pe_c_out_drain_io_1_idle, pe_c_out_drain_io_1_ready;
  reg  pe_c_out_drain_io_1_done_r, pe_c_out_drain_io_1_run_r;
  wire pe_c_out_drain_io_1_start = ap_start & ~pe_c_out_drain_io_1_run_r;
  always @(posedge ap_clk)
    if (!ap_rst_n) begin
      pe_c_out_drain_io_1_done_r <= 1'b0; pe_c_out_drain_io_1_run_r <= 1'b0;
    end else begin
      if (pe_c_out_drain_io_1_start && pe_c_out_drain_io_1_ready) pe_c_out_drain_io_1_run_r <= 1'b1;
      if (pe_c_out_drain_io_1_done) pe_c_out_drain_io_1_done_r <= 1'b1;
    end
  wire [31:0] pe_c_out_drain_io_1_pid0_dout; wire pe_c_out_drain_io_1_pid0_empty_n, pe_c_out_drain_io_1_pid0_read;
  wire pe_c_out_drain_io_2_done, pe_c_out_drain_io_2_idle, pe_c_out_drain_io_2_ready;
  reg  pe_c_out_drain_io_2_done_r, pe_c_out_drain_io_2_run_r;
  wire pe_c_out_drain_io_2_start = ap_start & ~pe_c_out_drain_io_2_run_r;
  always @(posedge ap_clk)
    if (!ap_rst_n) begin
      pe_c_out_drain_io_2_done_r <= 1'b0; pe_c_out_drain_io_2_run_r <= 1'b0;
    end else begin
      if (pe_c_out_drain_io_2_start && pe_c_out_drain_io_2_ready) pe_c_out_drain_io_2_run_r <= 1'b1;
      if (pe_c_out_drain_io_2_done) pe_c_out_drain_io_2_done_r <= 1'b1;
    end
  wire [31:0] pe_c_out_drain_io_2_pid0_dout; wire pe_c_out_drain_io_2_pid0_empty_n, pe_c_out_drain_io_2_pid0_read;
  wire pe_c_out_drain_io_3_done, pe_c_out_drain_io_3_idle, pe_c_out_drain_io_3_ready;
  reg  pe_c_out_drain_io_3_done_r, pe_c_out_drain_io_3_run_r;
  wire pe_c_out_drain_io_3_start = ap_start & ~pe_c_out_drain_io_3_run_r;
  always @(posedge ap_clk)
    if (!ap_rst_n) begin
      pe_c_out_drain_io_3_done_r <= 1'b0; pe_c_out_drain_io_3_run_r <= 1'b0;
    end else begin
      if (pe_c_out_drain_io_3_start && pe_c_out_drain_io_3_ready) pe_c_out_drain_io_3_run_r <= 1'b1;
      if (pe_c_out_drain_io_3_done) pe_c_out_drain_io_3_done_r <= 1'b1;
    end
  wire [31:0] pe_c_out_drain_io_3_pid0_dout; wire pe_c_out_drain_io_3_pid0_empty_n, pe_c_out_drain_io_3_pid0_read;
  assign ap_done = feed_up_load_io_0_done_r & feed_2_up_load_io_0_done_r & pe_c_out_drain_io_0_done_r & pe_c_out_drain_io_1_done_r & pe_c_out_drain_io_2_done_r & pe_c_out_drain_io_3_done_r;
  // coordinate axis 1: 16 constant source(s)
  wire [31:0] pe_pid1_dout [0:15];
  wire pe_pid1_empty_n [0:15];
  wire pe_pid1_read [0:15];
  spmw_const #(.DW(32), .VAL(0)) u_pe_pid1_0 (.dout(pe_pid1_dout[0]), .empty_n(pe_pid1_empty_n[0]), .read(pe_pid1_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_pe_pid1_1 (.dout(pe_pid1_dout[1]), .empty_n(pe_pid1_empty_n[1]), .read(pe_pid1_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_pe_pid1_2 (.dout(pe_pid1_dout[2]), .empty_n(pe_pid1_empty_n[2]), .read(pe_pid1_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_pe_pid1_3 (.dout(pe_pid1_dout[3]), .empty_n(pe_pid1_empty_n[3]), .read(pe_pid1_read[3]));
  spmw_const #(.DW(32), .VAL(0)) u_pe_pid1_4 (.dout(pe_pid1_dout[4]), .empty_n(pe_pid1_empty_n[4]), .read(pe_pid1_read[4]));
  spmw_const #(.DW(32), .VAL(1)) u_pe_pid1_5 (.dout(pe_pid1_dout[5]), .empty_n(pe_pid1_empty_n[5]), .read(pe_pid1_read[5]));
  spmw_const #(.DW(32), .VAL(2)) u_pe_pid1_6 (.dout(pe_pid1_dout[6]), .empty_n(pe_pid1_empty_n[6]), .read(pe_pid1_read[6]));
  spmw_const #(.DW(32), .VAL(3)) u_pe_pid1_7 (.dout(pe_pid1_dout[7]), .empty_n(pe_pid1_empty_n[7]), .read(pe_pid1_read[7]));
  spmw_const #(.DW(32), .VAL(0)) u_pe_pid1_8 (.dout(pe_pid1_dout[8]), .empty_n(pe_pid1_empty_n[8]), .read(pe_pid1_read[8]));
  spmw_const #(.DW(32), .VAL(1)) u_pe_pid1_9 (.dout(pe_pid1_dout[9]), .empty_n(pe_pid1_empty_n[9]), .read(pe_pid1_read[9]));
  spmw_const #(.DW(32), .VAL(2)) u_pe_pid1_10 (.dout(pe_pid1_dout[10]), .empty_n(pe_pid1_empty_n[10]), .read(pe_pid1_read[10]));
  spmw_const #(.DW(32), .VAL(3)) u_pe_pid1_11 (.dout(pe_pid1_dout[11]), .empty_n(pe_pid1_empty_n[11]), .read(pe_pid1_read[11]));
  spmw_const #(.DW(32), .VAL(0)) u_pe_pid1_12 (.dout(pe_pid1_dout[12]), .empty_n(pe_pid1_empty_n[12]), .read(pe_pid1_read[12]));
  spmw_const #(.DW(32), .VAL(1)) u_pe_pid1_13 (.dout(pe_pid1_dout[13]), .empty_n(pe_pid1_empty_n[13]), .read(pe_pid1_read[13]));
  spmw_const #(.DW(32), .VAL(2)) u_pe_pid1_14 (.dout(pe_pid1_dout[14]), .empty_n(pe_pid1_empty_n[14]), .read(pe_pid1_read[14]));
  spmw_const #(.DW(32), .VAL(3)) u_pe_pid1_15 (.dout(pe_pid1_dout[15]), .empty_n(pe_pid1_empty_n[15]), .read(pe_pid1_read[15]));
  // role pe_r0: 2 instance(s)
  pe_r0 u_pe_r0_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(pe_c_out_c_in_din[5]),
      .c_out_full_n(pe_c_out_c_in_full_n[5]),
      .c_out_write(pe_c_out_c_in_write[5]),
      .east_din(pe_east_west_din[2]),
      .east_full_n(pe_east_west_full_n[2]),
      .east_write(pe_east_west_write[2]),
      .north_dout(pe_north_bind_dout[1]),
      .north_empty_n(pe_north_bind_empty_n[1]),
      .north_read(pe_north_bind_read[1]),
      .south_din(pe_south_north_din[5]),
      .south_full_n(pe_south_north_full_n[5]),
      .south_write(pe_south_north_write[5]),
      .west_dout(pe_east_west_dout[1]),
      .west_empty_n(pe_east_west_empty_n[1]),
      .west_read(pe_east_west_read[1]),
      ._pid1_dout(pe_pid1_dout[1]),
      ._pid1_empty_n(pe_pid1_empty_n[1]),
      ._pid1_read(pe_pid1_read[1]));
  pe_r0 u_pe_r0_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(pe_c_out_c_in_din[6]),
      .c_out_full_n(pe_c_out_c_in_full_n[6]),
      .c_out_write(pe_c_out_c_in_write[6]),
      .east_din(pe_east_west_din[3]),
      .east_full_n(pe_east_west_full_n[3]),
      .east_write(pe_east_west_write[3]),
      .north_dout(pe_north_bind_dout[2]),
      .north_empty_n(pe_north_bind_empty_n[2]),
      .north_read(pe_north_bind_read[2]),
      .south_din(pe_south_north_din[6]),
      .south_full_n(pe_south_north_full_n[6]),
      .south_write(pe_south_north_write[6]),
      .west_dout(pe_east_west_dout[2]),
      .west_empty_n(pe_east_west_empty_n[2]),
      .west_read(pe_east_west_read[2]),
      ._pid1_dout(pe_pid1_dout[2]),
      ._pid1_empty_n(pe_pid1_empty_n[2]),
      ._pid1_read(pe_pid1_read[2]));
  // role pe_r1: 2 instance(s)
  pe_r1 u_pe_r1_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[5]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[5]),
      .c_in_read(pe_c_out_c_in_read[5]),
      .c_out_din(pe_c_out_c_in_din[9]),
      .c_out_full_n(pe_c_out_c_in_full_n[9]),
      .c_out_write(pe_c_out_c_in_write[9]),
      .east_din(pe_east_west_din[6]),
      .east_full_n(pe_east_west_full_n[6]),
      .east_write(pe_east_west_write[6]),
      .north_dout(pe_south_north_dout[5]),
      .north_empty_n(pe_south_north_empty_n[5]),
      .north_read(pe_south_north_read[5]),
      .south_din(pe_south_north_din[9]),
      .south_full_n(pe_south_north_full_n[9]),
      .south_write(pe_south_north_write[9]),
      .west_dout(pe_east_west_dout[5]),
      .west_empty_n(pe_east_west_empty_n[5]),
      .west_read(pe_east_west_read[5]),
      ._pid1_dout(pe_pid1_dout[5]),
      ._pid1_empty_n(pe_pid1_empty_n[5]),
      ._pid1_read(pe_pid1_read[5]));
  pe_r1 u_pe_r1_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[6]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[6]),
      .c_in_read(pe_c_out_c_in_read[6]),
      .c_out_din(pe_c_out_c_in_din[10]),
      .c_out_full_n(pe_c_out_c_in_full_n[10]),
      .c_out_write(pe_c_out_c_in_write[10]),
      .east_din(pe_east_west_din[7]),
      .east_full_n(pe_east_west_full_n[7]),
      .east_write(pe_east_west_write[7]),
      .north_dout(pe_south_north_dout[6]),
      .north_empty_n(pe_south_north_empty_n[6]),
      .north_read(pe_south_north_read[6]),
      .south_din(pe_south_north_din[10]),
      .south_full_n(pe_south_north_full_n[10]),
      .south_write(pe_south_north_write[10]),
      .west_dout(pe_east_west_dout[6]),
      .west_empty_n(pe_east_west_empty_n[6]),
      .west_read(pe_east_west_read[6]),
      ._pid1_dout(pe_pid1_dout[6]),
      ._pid1_empty_n(pe_pid1_empty_n[6]),
      ._pid1_read(pe_pid1_read[6]));
  // role pe_r2: 2 instance(s)
  pe_r2 u_pe_r2_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[9]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[9]),
      .c_in_read(pe_c_out_c_in_read[9]),
      .c_out_din(pe_c_out_c_in_din[13]),
      .c_out_full_n(pe_c_out_c_in_full_n[13]),
      .c_out_write(pe_c_out_c_in_write[13]),
      .east_din(pe_east_west_din[10]),
      .east_full_n(pe_east_west_full_n[10]),
      .east_write(pe_east_west_write[10]),
      .north_dout(pe_south_north_dout[9]),
      .north_empty_n(pe_south_north_empty_n[9]),
      .north_read(pe_south_north_read[9]),
      .south_din(pe_south_north_din[13]),
      .south_full_n(pe_south_north_full_n[13]),
      .south_write(pe_south_north_write[13]),
      .west_dout(pe_east_west_dout[9]),
      .west_empty_n(pe_east_west_empty_n[9]),
      .west_read(pe_east_west_read[9]),
      ._pid1_dout(pe_pid1_dout[9]),
      ._pid1_empty_n(pe_pid1_empty_n[9]),
      ._pid1_read(pe_pid1_read[9]));
  pe_r2 u_pe_r2_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[10]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[10]),
      .c_in_read(pe_c_out_c_in_read[10]),
      .c_out_din(pe_c_out_c_in_din[14]),
      .c_out_full_n(pe_c_out_c_in_full_n[14]),
      .c_out_write(pe_c_out_c_in_write[14]),
      .east_din(pe_east_west_din[11]),
      .east_full_n(pe_east_west_full_n[11]),
      .east_write(pe_east_west_write[11]),
      .north_dout(pe_south_north_dout[10]),
      .north_empty_n(pe_south_north_empty_n[10]),
      .north_read(pe_south_north_read[10]),
      .south_din(pe_south_north_din[14]),
      .south_full_n(pe_south_north_full_n[14]),
      .south_write(pe_south_north_write[14]),
      .west_dout(pe_east_west_dout[10]),
      .west_empty_n(pe_east_west_empty_n[10]),
      .west_read(pe_east_west_read[10]),
      ._pid1_dout(pe_pid1_dout[10]),
      ._pid1_empty_n(pe_pid1_empty_n[10]),
      ._pid1_read(pe_pid1_read[10]));
  // role pe_r3: 2 instance(s)
  pe_r3 u_pe_r3_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[13]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[13]),
      .c_in_read(pe_c_out_c_in_read[13]),
      .c_out_din(pe_c_out_bind_din[1]),
      .c_out_full_n(pe_c_out_bind_full_n[1]),
      .c_out_write(pe_c_out_bind_write[1]),
      .east_din(pe_east_west_din[14]),
      .east_full_n(pe_east_west_full_n[14]),
      .east_write(pe_east_west_write[14]),
      .north_dout(pe_south_north_dout[13]),
      .north_empty_n(pe_south_north_empty_n[13]),
      .north_read(pe_south_north_read[13]),
      .west_dout(pe_east_west_dout[13]),
      .west_empty_n(pe_east_west_empty_n[13]),
      .west_read(pe_east_west_read[13]),
      ._pid1_dout(pe_pid1_dout[13]),
      ._pid1_empty_n(pe_pid1_empty_n[13]),
      ._pid1_read(pe_pid1_read[13]));
  pe_r3 u_pe_r3_3_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[14]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[14]),
      .c_in_read(pe_c_out_c_in_read[14]),
      .c_out_din(pe_c_out_bind_din[2]),
      .c_out_full_n(pe_c_out_bind_full_n[2]),
      .c_out_write(pe_c_out_bind_write[2]),
      .east_din(pe_east_west_din[15]),
      .east_full_n(pe_east_west_full_n[15]),
      .east_write(pe_east_west_write[15]),
      .north_dout(pe_south_north_dout[14]),
      .north_empty_n(pe_south_north_empty_n[14]),
      .north_read(pe_south_north_read[14]),
      .west_dout(pe_east_west_dout[14]),
      .west_empty_n(pe_east_west_empty_n[14]),
      .west_read(pe_east_west_read[14]),
      ._pid1_dout(pe_pid1_dout[14]),
      ._pid1_empty_n(pe_pid1_empty_n[14]),
      ._pid1_read(pe_pid1_read[14]));
  // role pe_r4: 1 instance(s)
  pe_r4 u_pe_r4_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(pe_c_out_c_in_din[4]),
      .c_out_full_n(pe_c_out_c_in_full_n[4]),
      .c_out_write(pe_c_out_c_in_write[4]),
      .east_din(pe_east_west_din[1]),
      .east_full_n(pe_east_west_full_n[1]),
      .east_write(pe_east_west_write[1]),
      .north_dout(pe_north_bind_dout[0]),
      .north_empty_n(pe_north_bind_empty_n[0]),
      .north_read(pe_north_bind_read[0]),
      .south_din(pe_south_north_din[4]),
      .south_full_n(pe_south_north_full_n[4]),
      .south_write(pe_south_north_write[4]),
      .west_dout(pe_west_bind_dout[0]),
      .west_empty_n(pe_west_bind_empty_n[0]),
      .west_read(pe_west_bind_read[0]),
      ._pid1_dout(pe_pid1_dout[0]),
      ._pid1_empty_n(pe_pid1_empty_n[0]),
      ._pid1_read(pe_pid1_read[0]));
  // role pe_r5: 1 instance(s)
  pe_r5 u_pe_r5_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(pe_c_out_c_in_din[7]),
      .c_out_full_n(pe_c_out_c_in_full_n[7]),
      .c_out_write(pe_c_out_c_in_write[7]),
      .north_dout(pe_north_bind_dout[3]),
      .north_empty_n(pe_north_bind_empty_n[3]),
      .north_read(pe_north_bind_read[3]),
      .south_din(pe_south_north_din[7]),
      .south_full_n(pe_south_north_full_n[7]),
      .south_write(pe_south_north_write[7]),
      .west_dout(pe_east_west_dout[3]),
      .west_empty_n(pe_east_west_empty_n[3]),
      .west_read(pe_east_west_read[3]),
      ._pid1_dout(pe_pid1_dout[3]),
      ._pid1_empty_n(pe_pid1_empty_n[3]),
      ._pid1_read(pe_pid1_read[3]));
  // role pe_r6: 1 instance(s)
  pe_r6 u_pe_r6_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[4]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[4]),
      .c_in_read(pe_c_out_c_in_read[4]),
      .c_out_din(pe_c_out_c_in_din[8]),
      .c_out_full_n(pe_c_out_c_in_full_n[8]),
      .c_out_write(pe_c_out_c_in_write[8]),
      .east_din(pe_east_west_din[5]),
      .east_full_n(pe_east_west_full_n[5]),
      .east_write(pe_east_west_write[5]),
      .north_dout(pe_south_north_dout[4]),
      .north_empty_n(pe_south_north_empty_n[4]),
      .north_read(pe_south_north_read[4]),
      .south_din(pe_south_north_din[8]),
      .south_full_n(pe_south_north_full_n[8]),
      .south_write(pe_south_north_write[8]),
      .west_dout(pe_west_bind_dout[1]),
      .west_empty_n(pe_west_bind_empty_n[1]),
      .west_read(pe_west_bind_read[1]),
      ._pid1_dout(pe_pid1_dout[4]),
      ._pid1_empty_n(pe_pid1_empty_n[4]),
      ._pid1_read(pe_pid1_read[4]));
  // role pe_r7: 1 instance(s)
  pe_r7 u_pe_r7_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[7]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[7]),
      .c_in_read(pe_c_out_c_in_read[7]),
      .c_out_din(pe_c_out_c_in_din[11]),
      .c_out_full_n(pe_c_out_c_in_full_n[11]),
      .c_out_write(pe_c_out_c_in_write[11]),
      .north_dout(pe_south_north_dout[7]),
      .north_empty_n(pe_south_north_empty_n[7]),
      .north_read(pe_south_north_read[7]),
      .south_din(pe_south_north_din[11]),
      .south_full_n(pe_south_north_full_n[11]),
      .south_write(pe_south_north_write[11]),
      .west_dout(pe_east_west_dout[7]),
      .west_empty_n(pe_east_west_empty_n[7]),
      .west_read(pe_east_west_read[7]),
      ._pid1_dout(pe_pid1_dout[7]),
      ._pid1_empty_n(pe_pid1_empty_n[7]),
      ._pid1_read(pe_pid1_read[7]));
  // role pe_r8: 1 instance(s)
  pe_r8 u_pe_r8_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[8]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[8]),
      .c_in_read(pe_c_out_c_in_read[8]),
      .c_out_din(pe_c_out_c_in_din[12]),
      .c_out_full_n(pe_c_out_c_in_full_n[12]),
      .c_out_write(pe_c_out_c_in_write[12]),
      .east_din(pe_east_west_din[9]),
      .east_full_n(pe_east_west_full_n[9]),
      .east_write(pe_east_west_write[9]),
      .north_dout(pe_south_north_dout[8]),
      .north_empty_n(pe_south_north_empty_n[8]),
      .north_read(pe_south_north_read[8]),
      .south_din(pe_south_north_din[12]),
      .south_full_n(pe_south_north_full_n[12]),
      .south_write(pe_south_north_write[12]),
      .west_dout(pe_west_bind_dout[2]),
      .west_empty_n(pe_west_bind_empty_n[2]),
      .west_read(pe_west_bind_read[2]),
      ._pid1_dout(pe_pid1_dout[8]),
      ._pid1_empty_n(pe_pid1_empty_n[8]),
      ._pid1_read(pe_pid1_read[8]));
  // role pe_r9: 1 instance(s)
  pe_r9 u_pe_r9_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[11]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[11]),
      .c_in_read(pe_c_out_c_in_read[11]),
      .c_out_din(pe_c_out_c_in_din[15]),
      .c_out_full_n(pe_c_out_c_in_full_n[15]),
      .c_out_write(pe_c_out_c_in_write[15]),
      .north_dout(pe_south_north_dout[11]),
      .north_empty_n(pe_south_north_empty_n[11]),
      .north_read(pe_south_north_read[11]),
      .south_din(pe_south_north_din[15]),
      .south_full_n(pe_south_north_full_n[15]),
      .south_write(pe_south_north_write[15]),
      .west_dout(pe_east_west_dout[11]),
      .west_empty_n(pe_east_west_empty_n[11]),
      .west_read(pe_east_west_read[11]),
      ._pid1_dout(pe_pid1_dout[11]),
      ._pid1_empty_n(pe_pid1_empty_n[11]),
      ._pid1_read(pe_pid1_read[11]));
  // role pe_r10: 1 instance(s)
  pe_r10 u_pe_r10_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[12]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[12]),
      .c_in_read(pe_c_out_c_in_read[12]),
      .c_out_din(pe_c_out_bind_din[0]),
      .c_out_full_n(pe_c_out_bind_full_n[0]),
      .c_out_write(pe_c_out_bind_write[0]),
      .east_din(pe_east_west_din[13]),
      .east_full_n(pe_east_west_full_n[13]),
      .east_write(pe_east_west_write[13]),
      .north_dout(pe_south_north_dout[12]),
      .north_empty_n(pe_south_north_empty_n[12]),
      .north_read(pe_south_north_read[12]),
      .west_dout(pe_west_bind_dout[3]),
      .west_empty_n(pe_west_bind_empty_n[3]),
      .west_read(pe_west_bind_read[3]),
      ._pid1_dout(pe_pid1_dout[12]),
      ._pid1_empty_n(pe_pid1_empty_n[12]),
      ._pid1_read(pe_pid1_read[12]));
  // role pe_r11: 1 instance(s)
  pe_r11 u_pe_r11_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(pe_c_out_c_in_dout[15]),
      .c_in_empty_n(pe_c_out_c_in_empty_n[15]),
      .c_in_read(pe_c_out_c_in_read[15]),
      .c_out_din(pe_c_out_bind_din[3]),
      .c_out_full_n(pe_c_out_bind_full_n[3]),
      .c_out_write(pe_c_out_bind_write[3]),
      .north_dout(pe_south_north_dout[15]),
      .north_empty_n(pe_south_north_empty_n[15]),
      .north_read(pe_south_north_read[15]),
      .west_dout(pe_east_west_dout[15]),
      .west_empty_n(pe_east_west_empty_n[15]),
      .west_read(pe_east_west_read[15]),
      ._pid1_dout(pe_pid1_dout[15]),
      ._pid1_empty_n(pe_pid1_empty_n[15]),
      ._pid1_read(pe_pid1_read[15]));
  // coordinate axis 0: 4 constant source(s)
  wire [31:0] feed_pid0_dout [0:3];
  wire feed_pid0_empty_n [0:3];
  wire feed_pid0_read [0:3];
  spmw_const #(.DW(32), .VAL(0)) u_feed_pid0_0 (.dout(feed_pid0_dout[0]), .empty_n(feed_pid0_empty_n[0]), .read(feed_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_feed_pid0_1 (.dout(feed_pid0_dout[1]), .empty_n(feed_pid0_empty_n[1]), .read(feed_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_feed_pid0_2 (.dout(feed_pid0_dout[2]), .empty_n(feed_pid0_empty_n[2]), .read(feed_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_feed_pid0_3 (.dout(feed_pid0_dout[3]), .empty_n(feed_pid0_empty_n[3]), .read(feed_pid0_read[3]));
  // role feed_r0: 2 instance(s)
  feed_r0 u_feed_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_down_up_din[2]),
      .down_full_n(feed_down_up_full_n[2]),
      .down_write(feed_down_up_write[2]),
      .lane_din(pe_west_bind_din[1]),
      .lane_full_n(pe_west_bind_full_n[1]),
      .lane_write(pe_west_bind_write[1]),
      .up_dout(feed_down_up_dout[1]),
      .up_empty_n(feed_down_up_empty_n[1]),
      .up_read(feed_down_up_read[1]),
      ._pid0_dout(feed_pid0_dout[1]),
      ._pid0_empty_n(feed_pid0_empty_n[1]),
      ._pid0_read(feed_pid0_read[1]));
  feed_r0 u_feed_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_down_up_din[3]),
      .down_full_n(feed_down_up_full_n[3]),
      .down_write(feed_down_up_write[3]),
      .lane_din(pe_west_bind_din[2]),
      .lane_full_n(pe_west_bind_full_n[2]),
      .lane_write(pe_west_bind_write[2]),
      .up_dout(feed_down_up_dout[2]),
      .up_empty_n(feed_down_up_empty_n[2]),
      .up_read(feed_down_up_read[2]),
      ._pid0_dout(feed_pid0_dout[2]),
      ._pid0_empty_n(feed_pid0_empty_n[2]),
      ._pid0_read(feed_pid0_read[2]));
  // role feed_r1: 1 instance(s)
  feed_r1 u_feed_r1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_down_up_din[1]),
      .down_full_n(feed_down_up_full_n[1]),
      .down_write(feed_down_up_write[1]),
      .lane_din(pe_west_bind_din[0]),
      .lane_full_n(pe_west_bind_full_n[0]),
      .lane_write(pe_west_bind_write[0]),
      .up_dout(feed_up_bind_dout[0]),
      .up_empty_n(feed_up_bind_empty_n[0]),
      .up_read(feed_up_bind_read[0]),
      ._pid0_dout(feed_pid0_dout[0]),
      ._pid0_empty_n(feed_pid0_empty_n[0]),
      ._pid0_read(feed_pid0_read[0]));
  // role feed_r2: 1 instance(s)
  feed_r2 u_feed_r2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lane_din(pe_west_bind_din[3]),
      .lane_full_n(pe_west_bind_full_n[3]),
      .lane_write(pe_west_bind_write[3]),
      .up_dout(feed_down_up_dout[3]),
      .up_empty_n(feed_down_up_empty_n[3]),
      .up_read(feed_down_up_read[3]),
      ._pid0_dout(feed_pid0_dout[3]),
      ._pid0_empty_n(feed_pid0_empty_n[3]),
      ._pid0_read(feed_pid0_read[3]));
  // coordinate axis 0: 4 constant source(s)
  wire [31:0] feed_2_pid0_dout [0:3];
  wire feed_2_pid0_empty_n [0:3];
  wire feed_2_pid0_read [0:3];
  spmw_const #(.DW(32), .VAL(0)) u_feed_2_pid0_0 (.dout(feed_2_pid0_dout[0]), .empty_n(feed_2_pid0_empty_n[0]), .read(feed_2_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_feed_2_pid0_1 (.dout(feed_2_pid0_dout[1]), .empty_n(feed_2_pid0_empty_n[1]), .read(feed_2_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_feed_2_pid0_2 (.dout(feed_2_pid0_dout[2]), .empty_n(feed_2_pid0_empty_n[2]), .read(feed_2_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_feed_2_pid0_3 (.dout(feed_2_pid0_dout[3]), .empty_n(feed_2_pid0_empty_n[3]), .read(feed_2_pid0_read[3]));
  // role feed_2_r0: 2 instance(s)
  feed_2_r0 u_feed_2_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_2_down_up_din[2]),
      .down_full_n(feed_2_down_up_full_n[2]),
      .down_write(feed_2_down_up_write[2]),
      .lane_din(pe_north_bind_din[1]),
      .lane_full_n(pe_north_bind_full_n[1]),
      .lane_write(pe_north_bind_write[1]),
      .up_dout(feed_2_down_up_dout[1]),
      .up_empty_n(feed_2_down_up_empty_n[1]),
      .up_read(feed_2_down_up_read[1]),
      ._pid0_dout(feed_2_pid0_dout[1]),
      ._pid0_empty_n(feed_2_pid0_empty_n[1]),
      ._pid0_read(feed_2_pid0_read[1]));
  feed_2_r0 u_feed_2_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_2_down_up_din[3]),
      .down_full_n(feed_2_down_up_full_n[3]),
      .down_write(feed_2_down_up_write[3]),
      .lane_din(pe_north_bind_din[2]),
      .lane_full_n(pe_north_bind_full_n[2]),
      .lane_write(pe_north_bind_write[2]),
      .up_dout(feed_2_down_up_dout[2]),
      .up_empty_n(feed_2_down_up_empty_n[2]),
      .up_read(feed_2_down_up_read[2]),
      ._pid0_dout(feed_2_pid0_dout[2]),
      ._pid0_empty_n(feed_2_pid0_empty_n[2]),
      ._pid0_read(feed_2_pid0_read[2]));
  // role feed_2_r1: 1 instance(s)
  feed_2_r1 u_feed_2_r1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_2_down_up_din[1]),
      .down_full_n(feed_2_down_up_full_n[1]),
      .down_write(feed_2_down_up_write[1]),
      .lane_din(pe_north_bind_din[0]),
      .lane_full_n(pe_north_bind_full_n[0]),
      .lane_write(pe_north_bind_write[0]),
      .up_dout(feed_2_up_bind_dout[0]),
      .up_empty_n(feed_2_up_bind_empty_n[0]),
      .up_read(feed_2_up_bind_read[0]),
      ._pid0_dout(feed_2_pid0_dout[0]),
      ._pid0_empty_n(feed_2_pid0_empty_n[0]),
      ._pid0_read(feed_2_pid0_read[0]));
  // role feed_2_r2: 1 instance(s)
  feed_2_r2 u_feed_2_r2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lane_din(pe_north_bind_din[3]),
      .lane_full_n(pe_north_bind_full_n[3]),
      .lane_write(pe_north_bind_write[3]),
      .up_dout(feed_2_down_up_dout[3]),
      .up_empty_n(feed_2_down_up_empty_n[3]),
      .up_read(feed_2_down_up_read[3]),
      ._pid0_dout(feed_2_pid0_dout[3]),
      ._pid0_empty_n(feed_2_pid0_empty_n[3]),
      ._pid0_read(feed_2_pid0_read[3]));
  // mover feed_up_load_io: 1 instance(s)
  // site (0,) -> channel 0
  feed_up_load_io u_feed_up_load_io_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .ap_start(feed_up_load_io_0_start),
      .ap_done(feed_up_load_io_0_done),
      .ap_idle(feed_up_load_io_0_idle),
      .ap_ready(feed_up_load_io_0_ready),
      .offset(feed_up_load_io_0_offset),
      .m_axi_gmem_AWVALID(m_axi_feed_up_load_io_0_AWVALID),
      .m_axi_gmem_AWREADY(m_axi_feed_up_load_io_0_AWREADY),
      .m_axi_gmem_AWADDR(m_axi_feed_up_load_io_0_AWADDR),
      .m_axi_gmem_AWID(m_axi_feed_up_load_io_0_AWID),
      .m_axi_gmem_AWLEN(m_axi_feed_up_load_io_0_AWLEN),
      .m_axi_gmem_AWSIZE(m_axi_feed_up_load_io_0_AWSIZE),
      .m_axi_gmem_AWBURST(m_axi_feed_up_load_io_0_AWBURST),
      .m_axi_gmem_AWLOCK(m_axi_feed_up_load_io_0_AWLOCK),
      .m_axi_gmem_AWCACHE(m_axi_feed_up_load_io_0_AWCACHE),
      .m_axi_gmem_AWPROT(m_axi_feed_up_load_io_0_AWPROT),
      .m_axi_gmem_AWQOS(m_axi_feed_up_load_io_0_AWQOS),
      .m_axi_gmem_AWREGION(m_axi_feed_up_load_io_0_AWREGION),
      .m_axi_gmem_AWUSER(m_axi_feed_up_load_io_0_AWUSER),
      .m_axi_gmem_WVALID(m_axi_feed_up_load_io_0_WVALID),
      .m_axi_gmem_WREADY(m_axi_feed_up_load_io_0_WREADY),
      .m_axi_gmem_WDATA(m_axi_feed_up_load_io_0_WDATA),
      .m_axi_gmem_WSTRB(m_axi_feed_up_load_io_0_WSTRB),
      .m_axi_gmem_WLAST(m_axi_feed_up_load_io_0_WLAST),
      .m_axi_gmem_WID(m_axi_feed_up_load_io_0_WID),
      .m_axi_gmem_WUSER(m_axi_feed_up_load_io_0_WUSER),
      .m_axi_gmem_ARVALID(m_axi_feed_up_load_io_0_ARVALID),
      .m_axi_gmem_ARREADY(m_axi_feed_up_load_io_0_ARREADY),
      .m_axi_gmem_ARADDR(m_axi_feed_up_load_io_0_ARADDR),
      .m_axi_gmem_ARID(m_axi_feed_up_load_io_0_ARID),
      .m_axi_gmem_ARLEN(m_axi_feed_up_load_io_0_ARLEN),
      .m_axi_gmem_ARSIZE(m_axi_feed_up_load_io_0_ARSIZE),
      .m_axi_gmem_ARBURST(m_axi_feed_up_load_io_0_ARBURST),
      .m_axi_gmem_ARLOCK(m_axi_feed_up_load_io_0_ARLOCK),
      .m_axi_gmem_ARCACHE(m_axi_feed_up_load_io_0_ARCACHE),
      .m_axi_gmem_ARPROT(m_axi_feed_up_load_io_0_ARPROT),
      .m_axi_gmem_ARQOS(m_axi_feed_up_load_io_0_ARQOS),
      .m_axi_gmem_ARREGION(m_axi_feed_up_load_io_0_ARREGION),
      .m_axi_gmem_ARUSER(m_axi_feed_up_load_io_0_ARUSER),
      .m_axi_gmem_RVALID(m_axi_feed_up_load_io_0_RVALID),
      .m_axi_gmem_RREADY(m_axi_feed_up_load_io_0_RREADY),
      .m_axi_gmem_RDATA(m_axi_feed_up_load_io_0_RDATA),
      .m_axi_gmem_RLAST(m_axi_feed_up_load_io_0_RLAST),
      .m_axi_gmem_RID(m_axi_feed_up_load_io_0_RID),
      .m_axi_gmem_RUSER(m_axi_feed_up_load_io_0_RUSER),
      .m_axi_gmem_RRESP(m_axi_feed_up_load_io_0_RRESP),
      .m_axi_gmem_BVALID(m_axi_feed_up_load_io_0_BVALID),
      .m_axi_gmem_BREADY(m_axi_feed_up_load_io_0_BREADY),
      .m_axi_gmem_BRESP(m_axi_feed_up_load_io_0_BRESP),
      .m_axi_gmem_BID(m_axi_feed_up_load_io_0_BID),
      .m_axi_gmem_BUSER(m_axi_feed_up_load_io_0_BUSER),
      .chan_din(feed_up_bind_din[0]),
      .chan_full_n(feed_up_bind_full_n[0]),
      .chan_write(feed_up_bind_write[0]));
  // mover feed_2_up_load_io: 1 instance(s)
  // site (0,) -> channel 0
  feed_2_up_load_io u_feed_2_up_load_io_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .ap_start(feed_2_up_load_io_0_start),
      .ap_done(feed_2_up_load_io_0_done),
      .ap_idle(feed_2_up_load_io_0_idle),
      .ap_ready(feed_2_up_load_io_0_ready),
      .offset(feed_2_up_load_io_0_offset),
      .m_axi_gmem_AWVALID(m_axi_feed_2_up_load_io_0_AWVALID),
      .m_axi_gmem_AWREADY(m_axi_feed_2_up_load_io_0_AWREADY),
      .m_axi_gmem_AWADDR(m_axi_feed_2_up_load_io_0_AWADDR),
      .m_axi_gmem_AWID(m_axi_feed_2_up_load_io_0_AWID),
      .m_axi_gmem_AWLEN(m_axi_feed_2_up_load_io_0_AWLEN),
      .m_axi_gmem_AWSIZE(m_axi_feed_2_up_load_io_0_AWSIZE),
      .m_axi_gmem_AWBURST(m_axi_feed_2_up_load_io_0_AWBURST),
      .m_axi_gmem_AWLOCK(m_axi_feed_2_up_load_io_0_AWLOCK),
      .m_axi_gmem_AWCACHE(m_axi_feed_2_up_load_io_0_AWCACHE),
      .m_axi_gmem_AWPROT(m_axi_feed_2_up_load_io_0_AWPROT),
      .m_axi_gmem_AWQOS(m_axi_feed_2_up_load_io_0_AWQOS),
      .m_axi_gmem_AWREGION(m_axi_feed_2_up_load_io_0_AWREGION),
      .m_axi_gmem_AWUSER(m_axi_feed_2_up_load_io_0_AWUSER),
      .m_axi_gmem_WVALID(m_axi_feed_2_up_load_io_0_WVALID),
      .m_axi_gmem_WREADY(m_axi_feed_2_up_load_io_0_WREADY),
      .m_axi_gmem_WDATA(m_axi_feed_2_up_load_io_0_WDATA),
      .m_axi_gmem_WSTRB(m_axi_feed_2_up_load_io_0_WSTRB),
      .m_axi_gmem_WLAST(m_axi_feed_2_up_load_io_0_WLAST),
      .m_axi_gmem_WID(m_axi_feed_2_up_load_io_0_WID),
      .m_axi_gmem_WUSER(m_axi_feed_2_up_load_io_0_WUSER),
      .m_axi_gmem_ARVALID(m_axi_feed_2_up_load_io_0_ARVALID),
      .m_axi_gmem_ARREADY(m_axi_feed_2_up_load_io_0_ARREADY),
      .m_axi_gmem_ARADDR(m_axi_feed_2_up_load_io_0_ARADDR),
      .m_axi_gmem_ARID(m_axi_feed_2_up_load_io_0_ARID),
      .m_axi_gmem_ARLEN(m_axi_feed_2_up_load_io_0_ARLEN),
      .m_axi_gmem_ARSIZE(m_axi_feed_2_up_load_io_0_ARSIZE),
      .m_axi_gmem_ARBURST(m_axi_feed_2_up_load_io_0_ARBURST),
      .m_axi_gmem_ARLOCK(m_axi_feed_2_up_load_io_0_ARLOCK),
      .m_axi_gmem_ARCACHE(m_axi_feed_2_up_load_io_0_ARCACHE),
      .m_axi_gmem_ARPROT(m_axi_feed_2_up_load_io_0_ARPROT),
      .m_axi_gmem_ARQOS(m_axi_feed_2_up_load_io_0_ARQOS),
      .m_axi_gmem_ARREGION(m_axi_feed_2_up_load_io_0_ARREGION),
      .m_axi_gmem_ARUSER(m_axi_feed_2_up_load_io_0_ARUSER),
      .m_axi_gmem_RVALID(m_axi_feed_2_up_load_io_0_RVALID),
      .m_axi_gmem_RREADY(m_axi_feed_2_up_load_io_0_RREADY),
      .m_axi_gmem_RDATA(m_axi_feed_2_up_load_io_0_RDATA),
      .m_axi_gmem_RLAST(m_axi_feed_2_up_load_io_0_RLAST),
      .m_axi_gmem_RID(m_axi_feed_2_up_load_io_0_RID),
      .m_axi_gmem_RUSER(m_axi_feed_2_up_load_io_0_RUSER),
      .m_axi_gmem_RRESP(m_axi_feed_2_up_load_io_0_RRESP),
      .m_axi_gmem_BVALID(m_axi_feed_2_up_load_io_0_BVALID),
      .m_axi_gmem_BREADY(m_axi_feed_2_up_load_io_0_BREADY),
      .m_axi_gmem_BRESP(m_axi_feed_2_up_load_io_0_BRESP),
      .m_axi_gmem_BID(m_axi_feed_2_up_load_io_0_BID),
      .m_axi_gmem_BUSER(m_axi_feed_2_up_load_io_0_BUSER),
      .chan_din(feed_2_up_bind_din[0]),
      .chan_full_n(feed_2_up_bind_full_n[0]),
      .chan_write(feed_2_up_bind_write[0]));
  // mover pe_c_out_drain_io: 4 instance(s)
  spmw_const #(.DW(32), .VAL(0)) u_pe_c_out_drain_io_0_pid0 (.dout(pe_c_out_drain_io_0_pid0_dout), .empty_n(pe_c_out_drain_io_0_pid0_empty_n), .read(pe_c_out_drain_io_0_pid0_read));
  // site (3, 0) -> channel 0
  pe_c_out_drain_io u_pe_c_out_drain_io_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .ap_start(pe_c_out_drain_io_0_start),
      .ap_done(pe_c_out_drain_io_0_done),
      .ap_idle(pe_c_out_drain_io_0_idle),
      .ap_ready(pe_c_out_drain_io_0_ready),
      .offset(pe_c_out_drain_io_0_offset),
      .m_axi_gmem_AWVALID(m_axi_pe_c_out_drain_io_0_AWVALID),
      .m_axi_gmem_AWREADY(m_axi_pe_c_out_drain_io_0_AWREADY),
      .m_axi_gmem_AWADDR(m_axi_pe_c_out_drain_io_0_AWADDR),
      .m_axi_gmem_AWID(m_axi_pe_c_out_drain_io_0_AWID),
      .m_axi_gmem_AWLEN(m_axi_pe_c_out_drain_io_0_AWLEN),
      .m_axi_gmem_AWSIZE(m_axi_pe_c_out_drain_io_0_AWSIZE),
      .m_axi_gmem_AWBURST(m_axi_pe_c_out_drain_io_0_AWBURST),
      .m_axi_gmem_AWLOCK(m_axi_pe_c_out_drain_io_0_AWLOCK),
      .m_axi_gmem_AWCACHE(m_axi_pe_c_out_drain_io_0_AWCACHE),
      .m_axi_gmem_AWPROT(m_axi_pe_c_out_drain_io_0_AWPROT),
      .m_axi_gmem_AWQOS(m_axi_pe_c_out_drain_io_0_AWQOS),
      .m_axi_gmem_AWREGION(m_axi_pe_c_out_drain_io_0_AWREGION),
      .m_axi_gmem_AWUSER(m_axi_pe_c_out_drain_io_0_AWUSER),
      .m_axi_gmem_WVALID(m_axi_pe_c_out_drain_io_0_WVALID),
      .m_axi_gmem_WREADY(m_axi_pe_c_out_drain_io_0_WREADY),
      .m_axi_gmem_WDATA(m_axi_pe_c_out_drain_io_0_WDATA),
      .m_axi_gmem_WSTRB(m_axi_pe_c_out_drain_io_0_WSTRB),
      .m_axi_gmem_WLAST(m_axi_pe_c_out_drain_io_0_WLAST),
      .m_axi_gmem_WID(m_axi_pe_c_out_drain_io_0_WID),
      .m_axi_gmem_WUSER(m_axi_pe_c_out_drain_io_0_WUSER),
      .m_axi_gmem_ARVALID(m_axi_pe_c_out_drain_io_0_ARVALID),
      .m_axi_gmem_ARREADY(m_axi_pe_c_out_drain_io_0_ARREADY),
      .m_axi_gmem_ARADDR(m_axi_pe_c_out_drain_io_0_ARADDR),
      .m_axi_gmem_ARID(m_axi_pe_c_out_drain_io_0_ARID),
      .m_axi_gmem_ARLEN(m_axi_pe_c_out_drain_io_0_ARLEN),
      .m_axi_gmem_ARSIZE(m_axi_pe_c_out_drain_io_0_ARSIZE),
      .m_axi_gmem_ARBURST(m_axi_pe_c_out_drain_io_0_ARBURST),
      .m_axi_gmem_ARLOCK(m_axi_pe_c_out_drain_io_0_ARLOCK),
      .m_axi_gmem_ARCACHE(m_axi_pe_c_out_drain_io_0_ARCACHE),
      .m_axi_gmem_ARPROT(m_axi_pe_c_out_drain_io_0_ARPROT),
      .m_axi_gmem_ARQOS(m_axi_pe_c_out_drain_io_0_ARQOS),
      .m_axi_gmem_ARREGION(m_axi_pe_c_out_drain_io_0_ARREGION),
      .m_axi_gmem_ARUSER(m_axi_pe_c_out_drain_io_0_ARUSER),
      .m_axi_gmem_RVALID(m_axi_pe_c_out_drain_io_0_RVALID),
      .m_axi_gmem_RREADY(m_axi_pe_c_out_drain_io_0_RREADY),
      .m_axi_gmem_RDATA(m_axi_pe_c_out_drain_io_0_RDATA),
      .m_axi_gmem_RLAST(m_axi_pe_c_out_drain_io_0_RLAST),
      .m_axi_gmem_RID(m_axi_pe_c_out_drain_io_0_RID),
      .m_axi_gmem_RUSER(m_axi_pe_c_out_drain_io_0_RUSER),
      .m_axi_gmem_RRESP(m_axi_pe_c_out_drain_io_0_RRESP),
      .m_axi_gmem_BVALID(m_axi_pe_c_out_drain_io_0_BVALID),
      .m_axi_gmem_BREADY(m_axi_pe_c_out_drain_io_0_BREADY),
      .m_axi_gmem_BRESP(m_axi_pe_c_out_drain_io_0_BRESP),
      .m_axi_gmem_BID(m_axi_pe_c_out_drain_io_0_BID),
      .m_axi_gmem_BUSER(m_axi_pe_c_out_drain_io_0_BUSER),
      ._pid0_dout(pe_c_out_drain_io_0_pid0_dout),
      ._pid0_empty_n(pe_c_out_drain_io_0_pid0_empty_n),
      ._pid0_read(pe_c_out_drain_io_0_pid0_read),
      .chan_dout(pe_c_out_bind_dout[0]),
      .chan_empty_n(pe_c_out_bind_empty_n[0]),
      .chan_read(pe_c_out_bind_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_pe_c_out_drain_io_1_pid0 (.dout(pe_c_out_drain_io_1_pid0_dout), .empty_n(pe_c_out_drain_io_1_pid0_empty_n), .read(pe_c_out_drain_io_1_pid0_read));
  // site (3, 1) -> channel 1
  pe_c_out_drain_io u_pe_c_out_drain_io_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .ap_start(pe_c_out_drain_io_1_start),
      .ap_done(pe_c_out_drain_io_1_done),
      .ap_idle(pe_c_out_drain_io_1_idle),
      .ap_ready(pe_c_out_drain_io_1_ready),
      .offset(pe_c_out_drain_io_1_offset),
      .m_axi_gmem_AWVALID(m_axi_pe_c_out_drain_io_1_AWVALID),
      .m_axi_gmem_AWREADY(m_axi_pe_c_out_drain_io_1_AWREADY),
      .m_axi_gmem_AWADDR(m_axi_pe_c_out_drain_io_1_AWADDR),
      .m_axi_gmem_AWID(m_axi_pe_c_out_drain_io_1_AWID),
      .m_axi_gmem_AWLEN(m_axi_pe_c_out_drain_io_1_AWLEN),
      .m_axi_gmem_AWSIZE(m_axi_pe_c_out_drain_io_1_AWSIZE),
      .m_axi_gmem_AWBURST(m_axi_pe_c_out_drain_io_1_AWBURST),
      .m_axi_gmem_AWLOCK(m_axi_pe_c_out_drain_io_1_AWLOCK),
      .m_axi_gmem_AWCACHE(m_axi_pe_c_out_drain_io_1_AWCACHE),
      .m_axi_gmem_AWPROT(m_axi_pe_c_out_drain_io_1_AWPROT),
      .m_axi_gmem_AWQOS(m_axi_pe_c_out_drain_io_1_AWQOS),
      .m_axi_gmem_AWREGION(m_axi_pe_c_out_drain_io_1_AWREGION),
      .m_axi_gmem_AWUSER(m_axi_pe_c_out_drain_io_1_AWUSER),
      .m_axi_gmem_WVALID(m_axi_pe_c_out_drain_io_1_WVALID),
      .m_axi_gmem_WREADY(m_axi_pe_c_out_drain_io_1_WREADY),
      .m_axi_gmem_WDATA(m_axi_pe_c_out_drain_io_1_WDATA),
      .m_axi_gmem_WSTRB(m_axi_pe_c_out_drain_io_1_WSTRB),
      .m_axi_gmem_WLAST(m_axi_pe_c_out_drain_io_1_WLAST),
      .m_axi_gmem_WID(m_axi_pe_c_out_drain_io_1_WID),
      .m_axi_gmem_WUSER(m_axi_pe_c_out_drain_io_1_WUSER),
      .m_axi_gmem_ARVALID(m_axi_pe_c_out_drain_io_1_ARVALID),
      .m_axi_gmem_ARREADY(m_axi_pe_c_out_drain_io_1_ARREADY),
      .m_axi_gmem_ARADDR(m_axi_pe_c_out_drain_io_1_ARADDR),
      .m_axi_gmem_ARID(m_axi_pe_c_out_drain_io_1_ARID),
      .m_axi_gmem_ARLEN(m_axi_pe_c_out_drain_io_1_ARLEN),
      .m_axi_gmem_ARSIZE(m_axi_pe_c_out_drain_io_1_ARSIZE),
      .m_axi_gmem_ARBURST(m_axi_pe_c_out_drain_io_1_ARBURST),
      .m_axi_gmem_ARLOCK(m_axi_pe_c_out_drain_io_1_ARLOCK),
      .m_axi_gmem_ARCACHE(m_axi_pe_c_out_drain_io_1_ARCACHE),
      .m_axi_gmem_ARPROT(m_axi_pe_c_out_drain_io_1_ARPROT),
      .m_axi_gmem_ARQOS(m_axi_pe_c_out_drain_io_1_ARQOS),
      .m_axi_gmem_ARREGION(m_axi_pe_c_out_drain_io_1_ARREGION),
      .m_axi_gmem_ARUSER(m_axi_pe_c_out_drain_io_1_ARUSER),
      .m_axi_gmem_RVALID(m_axi_pe_c_out_drain_io_1_RVALID),
      .m_axi_gmem_RREADY(m_axi_pe_c_out_drain_io_1_RREADY),
      .m_axi_gmem_RDATA(m_axi_pe_c_out_drain_io_1_RDATA),
      .m_axi_gmem_RLAST(m_axi_pe_c_out_drain_io_1_RLAST),
      .m_axi_gmem_RID(m_axi_pe_c_out_drain_io_1_RID),
      .m_axi_gmem_RUSER(m_axi_pe_c_out_drain_io_1_RUSER),
      .m_axi_gmem_RRESP(m_axi_pe_c_out_drain_io_1_RRESP),
      .m_axi_gmem_BVALID(m_axi_pe_c_out_drain_io_1_BVALID),
      .m_axi_gmem_BREADY(m_axi_pe_c_out_drain_io_1_BREADY),
      .m_axi_gmem_BRESP(m_axi_pe_c_out_drain_io_1_BRESP),
      .m_axi_gmem_BID(m_axi_pe_c_out_drain_io_1_BID),
      .m_axi_gmem_BUSER(m_axi_pe_c_out_drain_io_1_BUSER),
      ._pid0_dout(pe_c_out_drain_io_1_pid0_dout),
      ._pid0_empty_n(pe_c_out_drain_io_1_pid0_empty_n),
      ._pid0_read(pe_c_out_drain_io_1_pid0_read),
      .chan_dout(pe_c_out_bind_dout[1]),
      .chan_empty_n(pe_c_out_bind_empty_n[1]),
      .chan_read(pe_c_out_bind_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_pe_c_out_drain_io_2_pid0 (.dout(pe_c_out_drain_io_2_pid0_dout), .empty_n(pe_c_out_drain_io_2_pid0_empty_n), .read(pe_c_out_drain_io_2_pid0_read));
  // site (3, 2) -> channel 2
  pe_c_out_drain_io u_pe_c_out_drain_io_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .ap_start(pe_c_out_drain_io_2_start),
      .ap_done(pe_c_out_drain_io_2_done),
      .ap_idle(pe_c_out_drain_io_2_idle),
      .ap_ready(pe_c_out_drain_io_2_ready),
      .offset(pe_c_out_drain_io_2_offset),
      .m_axi_gmem_AWVALID(m_axi_pe_c_out_drain_io_2_AWVALID),
      .m_axi_gmem_AWREADY(m_axi_pe_c_out_drain_io_2_AWREADY),
      .m_axi_gmem_AWADDR(m_axi_pe_c_out_drain_io_2_AWADDR),
      .m_axi_gmem_AWID(m_axi_pe_c_out_drain_io_2_AWID),
      .m_axi_gmem_AWLEN(m_axi_pe_c_out_drain_io_2_AWLEN),
      .m_axi_gmem_AWSIZE(m_axi_pe_c_out_drain_io_2_AWSIZE),
      .m_axi_gmem_AWBURST(m_axi_pe_c_out_drain_io_2_AWBURST),
      .m_axi_gmem_AWLOCK(m_axi_pe_c_out_drain_io_2_AWLOCK),
      .m_axi_gmem_AWCACHE(m_axi_pe_c_out_drain_io_2_AWCACHE),
      .m_axi_gmem_AWPROT(m_axi_pe_c_out_drain_io_2_AWPROT),
      .m_axi_gmem_AWQOS(m_axi_pe_c_out_drain_io_2_AWQOS),
      .m_axi_gmem_AWREGION(m_axi_pe_c_out_drain_io_2_AWREGION),
      .m_axi_gmem_AWUSER(m_axi_pe_c_out_drain_io_2_AWUSER),
      .m_axi_gmem_WVALID(m_axi_pe_c_out_drain_io_2_WVALID),
      .m_axi_gmem_WREADY(m_axi_pe_c_out_drain_io_2_WREADY),
      .m_axi_gmem_WDATA(m_axi_pe_c_out_drain_io_2_WDATA),
      .m_axi_gmem_WSTRB(m_axi_pe_c_out_drain_io_2_WSTRB),
      .m_axi_gmem_WLAST(m_axi_pe_c_out_drain_io_2_WLAST),
      .m_axi_gmem_WID(m_axi_pe_c_out_drain_io_2_WID),
      .m_axi_gmem_WUSER(m_axi_pe_c_out_drain_io_2_WUSER),
      .m_axi_gmem_ARVALID(m_axi_pe_c_out_drain_io_2_ARVALID),
      .m_axi_gmem_ARREADY(m_axi_pe_c_out_drain_io_2_ARREADY),
      .m_axi_gmem_ARADDR(m_axi_pe_c_out_drain_io_2_ARADDR),
      .m_axi_gmem_ARID(m_axi_pe_c_out_drain_io_2_ARID),
      .m_axi_gmem_ARLEN(m_axi_pe_c_out_drain_io_2_ARLEN),
      .m_axi_gmem_ARSIZE(m_axi_pe_c_out_drain_io_2_ARSIZE),
      .m_axi_gmem_ARBURST(m_axi_pe_c_out_drain_io_2_ARBURST),
      .m_axi_gmem_ARLOCK(m_axi_pe_c_out_drain_io_2_ARLOCK),
      .m_axi_gmem_ARCACHE(m_axi_pe_c_out_drain_io_2_ARCACHE),
      .m_axi_gmem_ARPROT(m_axi_pe_c_out_drain_io_2_ARPROT),
      .m_axi_gmem_ARQOS(m_axi_pe_c_out_drain_io_2_ARQOS),
      .m_axi_gmem_ARREGION(m_axi_pe_c_out_drain_io_2_ARREGION),
      .m_axi_gmem_ARUSER(m_axi_pe_c_out_drain_io_2_ARUSER),
      .m_axi_gmem_RVALID(m_axi_pe_c_out_drain_io_2_RVALID),
      .m_axi_gmem_RREADY(m_axi_pe_c_out_drain_io_2_RREADY),
      .m_axi_gmem_RDATA(m_axi_pe_c_out_drain_io_2_RDATA),
      .m_axi_gmem_RLAST(m_axi_pe_c_out_drain_io_2_RLAST),
      .m_axi_gmem_RID(m_axi_pe_c_out_drain_io_2_RID),
      .m_axi_gmem_RUSER(m_axi_pe_c_out_drain_io_2_RUSER),
      .m_axi_gmem_RRESP(m_axi_pe_c_out_drain_io_2_RRESP),
      .m_axi_gmem_BVALID(m_axi_pe_c_out_drain_io_2_BVALID),
      .m_axi_gmem_BREADY(m_axi_pe_c_out_drain_io_2_BREADY),
      .m_axi_gmem_BRESP(m_axi_pe_c_out_drain_io_2_BRESP),
      .m_axi_gmem_BID(m_axi_pe_c_out_drain_io_2_BID),
      .m_axi_gmem_BUSER(m_axi_pe_c_out_drain_io_2_BUSER),
      ._pid0_dout(pe_c_out_drain_io_2_pid0_dout),
      ._pid0_empty_n(pe_c_out_drain_io_2_pid0_empty_n),
      ._pid0_read(pe_c_out_drain_io_2_pid0_read),
      .chan_dout(pe_c_out_bind_dout[2]),
      .chan_empty_n(pe_c_out_bind_empty_n[2]),
      .chan_read(pe_c_out_bind_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_pe_c_out_drain_io_3_pid0 (.dout(pe_c_out_drain_io_3_pid0_dout), .empty_n(pe_c_out_drain_io_3_pid0_empty_n), .read(pe_c_out_drain_io_3_pid0_read));
  // site (3, 3) -> channel 3
  pe_c_out_drain_io u_pe_c_out_drain_io_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .ap_start(pe_c_out_drain_io_3_start),
      .ap_done(pe_c_out_drain_io_3_done),
      .ap_idle(pe_c_out_drain_io_3_idle),
      .ap_ready(pe_c_out_drain_io_3_ready),
      .offset(pe_c_out_drain_io_3_offset),
      .m_axi_gmem_AWVALID(m_axi_pe_c_out_drain_io_3_AWVALID),
      .m_axi_gmem_AWREADY(m_axi_pe_c_out_drain_io_3_AWREADY),
      .m_axi_gmem_AWADDR(m_axi_pe_c_out_drain_io_3_AWADDR),
      .m_axi_gmem_AWID(m_axi_pe_c_out_drain_io_3_AWID),
      .m_axi_gmem_AWLEN(m_axi_pe_c_out_drain_io_3_AWLEN),
      .m_axi_gmem_AWSIZE(m_axi_pe_c_out_drain_io_3_AWSIZE),
      .m_axi_gmem_AWBURST(m_axi_pe_c_out_drain_io_3_AWBURST),
      .m_axi_gmem_AWLOCK(m_axi_pe_c_out_drain_io_3_AWLOCK),
      .m_axi_gmem_AWCACHE(m_axi_pe_c_out_drain_io_3_AWCACHE),
      .m_axi_gmem_AWPROT(m_axi_pe_c_out_drain_io_3_AWPROT),
      .m_axi_gmem_AWQOS(m_axi_pe_c_out_drain_io_3_AWQOS),
      .m_axi_gmem_AWREGION(m_axi_pe_c_out_drain_io_3_AWREGION),
      .m_axi_gmem_AWUSER(m_axi_pe_c_out_drain_io_3_AWUSER),
      .m_axi_gmem_WVALID(m_axi_pe_c_out_drain_io_3_WVALID),
      .m_axi_gmem_WREADY(m_axi_pe_c_out_drain_io_3_WREADY),
      .m_axi_gmem_WDATA(m_axi_pe_c_out_drain_io_3_WDATA),
      .m_axi_gmem_WSTRB(m_axi_pe_c_out_drain_io_3_WSTRB),
      .m_axi_gmem_WLAST(m_axi_pe_c_out_drain_io_3_WLAST),
      .m_axi_gmem_WID(m_axi_pe_c_out_drain_io_3_WID),
      .m_axi_gmem_WUSER(m_axi_pe_c_out_drain_io_3_WUSER),
      .m_axi_gmem_ARVALID(m_axi_pe_c_out_drain_io_3_ARVALID),
      .m_axi_gmem_ARREADY(m_axi_pe_c_out_drain_io_3_ARREADY),
      .m_axi_gmem_ARADDR(m_axi_pe_c_out_drain_io_3_ARADDR),
      .m_axi_gmem_ARID(m_axi_pe_c_out_drain_io_3_ARID),
      .m_axi_gmem_ARLEN(m_axi_pe_c_out_drain_io_3_ARLEN),
      .m_axi_gmem_ARSIZE(m_axi_pe_c_out_drain_io_3_ARSIZE),
      .m_axi_gmem_ARBURST(m_axi_pe_c_out_drain_io_3_ARBURST),
      .m_axi_gmem_ARLOCK(m_axi_pe_c_out_drain_io_3_ARLOCK),
      .m_axi_gmem_ARCACHE(m_axi_pe_c_out_drain_io_3_ARCACHE),
      .m_axi_gmem_ARPROT(m_axi_pe_c_out_drain_io_3_ARPROT),
      .m_axi_gmem_ARQOS(m_axi_pe_c_out_drain_io_3_ARQOS),
      .m_axi_gmem_ARREGION(m_axi_pe_c_out_drain_io_3_ARREGION),
      .m_axi_gmem_ARUSER(m_axi_pe_c_out_drain_io_3_ARUSER),
      .m_axi_gmem_RVALID(m_axi_pe_c_out_drain_io_3_RVALID),
      .m_axi_gmem_RREADY(m_axi_pe_c_out_drain_io_3_RREADY),
      .m_axi_gmem_RDATA(m_axi_pe_c_out_drain_io_3_RDATA),
      .m_axi_gmem_RLAST(m_axi_pe_c_out_drain_io_3_RLAST),
      .m_axi_gmem_RID(m_axi_pe_c_out_drain_io_3_RID),
      .m_axi_gmem_RUSER(m_axi_pe_c_out_drain_io_3_RUSER),
      .m_axi_gmem_RRESP(m_axi_pe_c_out_drain_io_3_RRESP),
      .m_axi_gmem_BVALID(m_axi_pe_c_out_drain_io_3_BVALID),
      .m_axi_gmem_BREADY(m_axi_pe_c_out_drain_io_3_BREADY),
      .m_axi_gmem_BRESP(m_axi_pe_c_out_drain_io_3_BRESP),
      .m_axi_gmem_BID(m_axi_pe_c_out_drain_io_3_BID),
      .m_axi_gmem_BUSER(m_axi_pe_c_out_drain_io_3_BUSER),
      ._pid0_dout(pe_c_out_drain_io_3_pid0_dout),
      ._pid0_empty_n(pe_c_out_drain_io_3_pid0_empty_n),
      ._pid0_read(pe_c_out_drain_io_3_pid0_read),
      .chan_dout(pe_c_out_bind_dout[3]),
      .chan_empty_n(pe_c_out_bind_empty_n[3]),
      .chan_read(pe_c_out_bind_read[3]));
endmodule

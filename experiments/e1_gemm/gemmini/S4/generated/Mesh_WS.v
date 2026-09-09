module MacUnit(
  input  [7:0]  io_in_a, // @[src/main/scala/gemmini/PE.scala 16:14]
  input  [7:0]  io_in_b, // @[src/main/scala/gemmini/PE.scala 16:14]
  input  [31:0] io_in_c, // @[src/main/scala/gemmini/PE.scala 16:14]
  output [31:0] io_out_d // @[src/main/scala/gemmini/PE.scala 16:14]
);
  wire [15:0] _io_out_d_T = $signed(io_in_a) * $signed(io_in_b); // @[src/main/scala/gemmini/Arithmetic.scala 93:49]
  wire [31:0] _GEN_0 = {{16{_io_out_d_T[15]}},_io_out_d_T}; // @[src/main/scala/gemmini/Arithmetic.scala 93:54]
  assign io_out_d = $signed(_GEN_0) + $signed(io_in_c); // @[src/main/scala/gemmini/Arithmetic.scala 93:54]
endmodule
module PE(
  input         clock,
  input  [7:0]  io_in_a, // @[src/main/scala/gemmini/PE.scala 35:14]
  input  [31:0] io_in_b, // @[src/main/scala/gemmini/PE.scala 35:14]
  input  [31:0] io_in_d, // @[src/main/scala/gemmini/PE.scala 35:14]
  output [7:0]  io_out_a, // @[src/main/scala/gemmini/PE.scala 35:14]
  output [31:0] io_out_b, // @[src/main/scala/gemmini/PE.scala 35:14]
  output [31:0] io_out_c, // @[src/main/scala/gemmini/PE.scala 35:14]
  input         io_in_control_dataflow, // @[src/main/scala/gemmini/PE.scala 35:14]
  input         io_in_control_propagate, // @[src/main/scala/gemmini/PE.scala 35:14]
  input  [4:0]  io_in_control_shift, // @[src/main/scala/gemmini/PE.scala 35:14]
  output        io_out_control_dataflow, // @[src/main/scala/gemmini/PE.scala 35:14]
  output        io_out_control_propagate, // @[src/main/scala/gemmini/PE.scala 35:14]
  output [4:0]  io_out_control_shift, // @[src/main/scala/gemmini/PE.scala 35:14]
  input  [2:0]  io_in_id, // @[src/main/scala/gemmini/PE.scala 35:14]
  output [2:0]  io_out_id, // @[src/main/scala/gemmini/PE.scala 35:14]
  input         io_in_last, // @[src/main/scala/gemmini/PE.scala 35:14]
  output        io_out_last, // @[src/main/scala/gemmini/PE.scala 35:14]
  input         io_in_valid, // @[src/main/scala/gemmini/PE.scala 35:14]
  output        io_out_valid // @[src/main/scala/gemmini/PE.scala 35:14]
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
`endif // RANDOMIZE_REG_INIT
  wire [7:0] mac_unit_io_in_a; // @[src/main/scala/gemmini/PE.scala 64:24]
  wire [7:0] mac_unit_io_in_b; // @[src/main/scala/gemmini/PE.scala 64:24]
  wire [31:0] mac_unit_io_in_c; // @[src/main/scala/gemmini/PE.scala 64:24]
  wire [31:0] mac_unit_io_out_d; // @[src/main/scala/gemmini/PE.scala 64:24]
  reg [7:0] c1; // @[src/main/scala/gemmini/PE.scala 70:15]
  reg [7:0] c2; // @[src/main/scala/gemmini/PE.scala 71:15]
  wire [7:0] _GEN_7 = io_in_control_propagate ? $signed(c1) : $signed(c2); // @[src/main/scala/gemmini/PE.scala 119:30 120:16 126:16]
  wire [31:0] _GEN_11 = io_in_control_propagate ? $signed(io_in_d) : $signed({{24{c1[7]}},c1}); // @[src/main/scala/gemmini/PE.scala 119:30 124:10 70:15]
  wire [31:0] _GEN_12 = io_in_control_propagate ? $signed({{24{c2[7]}},c2}) : $signed(io_in_d); // @[src/main/scala/gemmini/PE.scala 119:30 130:10 71:15]
  wire [31:0] _GEN_27 = ~io_in_valid ? $signed({{24{c1[7]}},c1}) : $signed(_GEN_11); // @[src/main/scala/gemmini/PE.scala 141:17 142:8]
  wire [31:0] _GEN_28 = ~io_in_valid ? $signed({{24{c2[7]}},c2}) : $signed(_GEN_12); // @[src/main/scala/gemmini/PE.scala 141:17 143:8]
  MacUnit mac_unit ( // @[src/main/scala/gemmini/PE.scala 64:24]
    .io_in_a(mac_unit_io_in_a),
    .io_in_b(mac_unit_io_in_b),
    .io_in_c(mac_unit_io_in_c),
    .io_out_d(mac_unit_io_out_d)
  );
  assign io_out_a = io_in_a; // @[src/main/scala/gemmini/PE.scala 79:12]
  assign io_out_b = mac_unit_io_out_d; // @[src/main/scala/gemmini/PE.scala 119:30 123:16 129:16]
  assign io_out_c = {{24{_GEN_7[7]}},_GEN_7}; // @[src/main/scala/gemmini/PE.scala 102:95]
  assign io_out_control_dataflow = io_in_control_dataflow; // @[src/main/scala/gemmini/PE.scala 80:27]
  assign io_out_control_propagate = io_in_control_propagate; // @[src/main/scala/gemmini/PE.scala 81:28]
  assign io_out_control_shift = io_in_control_shift; // @[src/main/scala/gemmini/PE.scala 82:24]
  assign io_out_id = io_in_id; // @[src/main/scala/gemmini/PE.scala 83:13]
  assign io_out_last = io_in_last; // @[src/main/scala/gemmini/PE.scala 84:15]
  assign io_out_valid = io_in_valid; // @[src/main/scala/gemmini/PE.scala 85:16]
  assign mac_unit_io_in_a = io_in_a; // @[src/main/scala/gemmini/PE.scala 87:20]
  assign mac_unit_io_in_b = io_in_control_propagate ? $signed(c2) : $signed(c1); // @[src/main/scala/gemmini/PE.scala 119:30 121:24 127:24]
  assign mac_unit_io_in_c = io_in_b; // @[src/main/scala/gemmini/PE.scala 102:95]
  always @(posedge clock) begin
    c1 <= _GEN_27[7:0];
    c2 <= _GEN_28[7:0];
  end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  c1 = _RAND_0[7:0];
  _RAND_1 = {1{`RANDOM}};
  c2 = _RAND_1[7:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
module Tile(
  input         clock,
  input  [7:0]  io_in_a_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input  [31:0] io_in_b_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input  [31:0] io_in_d_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input         io_in_control_0_dataflow, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input         io_in_control_0_propagate, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input  [4:0]  io_in_control_0_shift, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input  [2:0]  io_in_id_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input         io_in_last_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [7:0]  io_out_a_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [31:0] io_out_c_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [31:0] io_out_b_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output        io_out_control_0_dataflow, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output        io_out_control_0_propagate, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [4:0]  io_out_control_0_shift, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [2:0]  io_out_id_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output        io_out_last_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input         io_in_valid_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output        io_out_valid_0 // @[src/main/scala/gemmini/Tile.scala 17:14]
);
  wire  tile_0_0_clock; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [7:0] tile_0_0_io_in_a; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [31:0] tile_0_0_io_in_b; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [31:0] tile_0_0_io_in_d; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [7:0] tile_0_0_io_out_a; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [31:0] tile_0_0_io_out_b; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [31:0] tile_0_0_io_out_c; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_in_control_dataflow; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_in_control_propagate; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [4:0] tile_0_0_io_in_control_shift; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_out_control_dataflow; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_out_control_propagate; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [4:0] tile_0_0_io_out_control_shift; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [2:0] tile_0_0_io_in_id; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [2:0] tile_0_0_io_out_id; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_in_last; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_out_last; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_in_valid; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_out_valid; // @[src/main/scala/gemmini/Tile.scala 42:44]
  PE tile_0_0 ( // @[src/main/scala/gemmini/Tile.scala 42:44]
    .clock(tile_0_0_clock),
    .io_in_a(tile_0_0_io_in_a),
    .io_in_b(tile_0_0_io_in_b),
    .io_in_d(tile_0_0_io_in_d),
    .io_out_a(tile_0_0_io_out_a),
    .io_out_b(tile_0_0_io_out_b),
    .io_out_c(tile_0_0_io_out_c),
    .io_in_control_dataflow(tile_0_0_io_in_control_dataflow),
    .io_in_control_propagate(tile_0_0_io_in_control_propagate),
    .io_in_control_shift(tile_0_0_io_in_control_shift),
    .io_out_control_dataflow(tile_0_0_io_out_control_dataflow),
    .io_out_control_propagate(tile_0_0_io_out_control_propagate),
    .io_out_control_shift(tile_0_0_io_out_control_shift),
    .io_in_id(tile_0_0_io_in_id),
    .io_out_id(tile_0_0_io_out_id),
    .io_in_last(tile_0_0_io_in_last),
    .io_out_last(tile_0_0_io_out_last),
    .io_in_valid(tile_0_0_io_in_valid),
    .io_out_valid(tile_0_0_io_out_valid)
  );
  assign io_out_a_0 = tile_0_0_io_out_a; // @[src/main/scala/gemmini/Tile.scala 130:17]
  assign io_out_c_0 = tile_0_0_io_out_c; // @[src/main/scala/gemmini/Tile.scala 111:17]
  assign io_out_b_0 = tile_0_0_io_out_b; // @[src/main/scala/gemmini/Tile.scala 117:17]
  assign io_out_control_0_dataflow = tile_0_0_io_out_control_dataflow; // @[src/main/scala/gemmini/Tile.scala 112:23]
  assign io_out_control_0_propagate = tile_0_0_io_out_control_propagate; // @[src/main/scala/gemmini/Tile.scala 112:23]
  assign io_out_control_0_shift = tile_0_0_io_out_control_shift; // @[src/main/scala/gemmini/Tile.scala 112:23]
  assign io_out_id_0 = tile_0_0_io_out_id; // @[src/main/scala/gemmini/Tile.scala 113:18]
  assign io_out_last_0 = tile_0_0_io_out_last; // @[src/main/scala/gemmini/Tile.scala 114:20]
  assign io_out_valid_0 = tile_0_0_io_out_valid; // @[src/main/scala/gemmini/Tile.scala 115:21]
  assign tile_0_0_clock = clock;
  assign tile_0_0_io_in_a = io_in_a_0; // @[src/main/scala/gemmini/Tile.scala 50:20]
  assign tile_0_0_io_in_b = io_in_b_0; // @[src/main/scala/gemmini/Tile.scala 59:20]
  assign tile_0_0_io_in_d = io_in_d_0; // @[src/main/scala/gemmini/Tile.scala 68:20]
  assign tile_0_0_io_in_control_dataflow = io_in_control_0_dataflow; // @[src/main/scala/gemmini/Tile.scala 77:26]
  assign tile_0_0_io_in_control_propagate = io_in_control_0_propagate; // @[src/main/scala/gemmini/Tile.scala 77:26]
  assign tile_0_0_io_in_control_shift = io_in_control_0_shift; // @[src/main/scala/gemmini/Tile.scala 77:26]
  assign tile_0_0_io_in_id = io_in_id_0; // @[src/main/scala/gemmini/Tile.scala 95:21]
  assign tile_0_0_io_in_last = io_in_last_0; // @[src/main/scala/gemmini/Tile.scala 104:23]
  assign tile_0_0_io_in_valid = io_in_valid_0; // @[src/main/scala/gemmini/Tile.scala 86:24]
endmodule
module Tile_12(
  input         clock,
  input  [7:0]  io_in_a_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input  [31:0] io_in_b_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input  [31:0] io_in_d_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input         io_in_control_0_dataflow, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input         io_in_control_0_propagate, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input  [4:0]  io_in_control_0_shift, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input  [2:0]  io_in_id_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input         io_in_last_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [7:0]  io_out_a_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [31:0] io_out_c_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [31:0] io_out_b_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output        io_out_control_0_dataflow, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output        io_out_control_0_propagate, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [4:0]  io_out_control_0_shift, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output [2:0]  io_out_id_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output        io_out_last_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  input         io_in_valid_0, // @[src/main/scala/gemmini/Tile.scala 17:14]
  output        io_out_valid_0 // @[src/main/scala/gemmini/Tile.scala 17:14]
);
  wire  tile_0_0_clock; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [7:0] tile_0_0_io_in_a; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [31:0] tile_0_0_io_in_b; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [31:0] tile_0_0_io_in_d; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [7:0] tile_0_0_io_out_a; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [31:0] tile_0_0_io_out_b; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [31:0] tile_0_0_io_out_c; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_in_control_dataflow; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_in_control_propagate; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [4:0] tile_0_0_io_in_control_shift; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_out_control_dataflow; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_out_control_propagate; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [4:0] tile_0_0_io_out_control_shift; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [2:0] tile_0_0_io_in_id; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire [2:0] tile_0_0_io_out_id; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_in_last; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_out_last; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_in_valid; // @[src/main/scala/gemmini/Tile.scala 42:44]
  wire  tile_0_0_io_out_valid; // @[src/main/scala/gemmini/Tile.scala 42:44]
  PE tile_0_0 ( // @[src/main/scala/gemmini/Tile.scala 42:44]
    .clock(tile_0_0_clock),
    .io_in_a(tile_0_0_io_in_a),
    .io_in_b(tile_0_0_io_in_b),
    .io_in_d(tile_0_0_io_in_d),
    .io_out_a(tile_0_0_io_out_a),
    .io_out_b(tile_0_0_io_out_b),
    .io_out_c(tile_0_0_io_out_c),
    .io_in_control_dataflow(tile_0_0_io_in_control_dataflow),
    .io_in_control_propagate(tile_0_0_io_in_control_propagate),
    .io_in_control_shift(tile_0_0_io_in_control_shift),
    .io_out_control_dataflow(tile_0_0_io_out_control_dataflow),
    .io_out_control_propagate(tile_0_0_io_out_control_propagate),
    .io_out_control_shift(tile_0_0_io_out_control_shift),
    .io_in_id(tile_0_0_io_in_id),
    .io_out_id(tile_0_0_io_out_id),
    .io_in_last(tile_0_0_io_in_last),
    .io_out_last(tile_0_0_io_out_last),
    .io_in_valid(tile_0_0_io_in_valid),
    .io_out_valid(tile_0_0_io_out_valid)
  );
  assign io_out_a_0 = tile_0_0_io_out_a; // @[src/main/scala/gemmini/Tile.scala 130:17]
  assign io_out_c_0 = tile_0_0_io_out_c; // @[src/main/scala/gemmini/Tile.scala 111:17]
  assign io_out_b_0 = tile_0_0_io_out_b; // @[src/main/scala/gemmini/Tile.scala 117:17]
  assign io_out_control_0_dataflow = tile_0_0_io_out_control_dataflow; // @[src/main/scala/gemmini/Tile.scala 112:23]
  assign io_out_control_0_propagate = tile_0_0_io_out_control_propagate; // @[src/main/scala/gemmini/Tile.scala 112:23]
  assign io_out_control_0_shift = tile_0_0_io_out_control_shift; // @[src/main/scala/gemmini/Tile.scala 112:23]
  assign io_out_id_0 = tile_0_0_io_out_id; // @[src/main/scala/gemmini/Tile.scala 113:18]
  assign io_out_last_0 = tile_0_0_io_out_last; // @[src/main/scala/gemmini/Tile.scala 114:20]
  assign io_out_valid_0 = tile_0_0_io_out_valid; // @[src/main/scala/gemmini/Tile.scala 115:21]
  assign tile_0_0_clock = clock;
  assign tile_0_0_io_in_a = io_in_a_0; // @[src/main/scala/gemmini/Tile.scala 50:20]
  assign tile_0_0_io_in_b = io_in_b_0; // @[src/main/scala/gemmini/Tile.scala 59:20]
  assign tile_0_0_io_in_d = io_in_d_0; // @[src/main/scala/gemmini/Tile.scala 68:20]
  assign tile_0_0_io_in_control_dataflow = io_in_control_0_dataflow; // @[src/main/scala/gemmini/Tile.scala 77:26]
  assign tile_0_0_io_in_control_propagate = io_in_control_0_propagate; // @[src/main/scala/gemmini/Tile.scala 77:26]
  assign tile_0_0_io_in_control_shift = io_in_control_0_shift; // @[src/main/scala/gemmini/Tile.scala 77:26]
  assign tile_0_0_io_in_id = io_in_id_0; // @[src/main/scala/gemmini/Tile.scala 95:21]
  assign tile_0_0_io_in_last = io_in_last_0; // @[src/main/scala/gemmini/Tile.scala 104:23]
  assign tile_0_0_io_in_valid = io_in_valid_0; // @[src/main/scala/gemmini/Tile.scala 86:24]
endmodule
module Mesh(
  input         clock,
  input         reset,
  input  [7:0]  io_in_a_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_a_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_a_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_a_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_b_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_b_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_b_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_b_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_d_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_d_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_d_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [7:0]  io_in_d_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_control_0_0_dataflow, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_control_0_0_propagate, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [4:0]  io_in_control_0_0_shift, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_control_1_0_dataflow, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_control_1_0_propagate, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [4:0]  io_in_control_1_0_shift, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_control_2_0_dataflow, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_control_2_0_propagate, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [4:0]  io_in_control_2_0_shift, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_control_3_0_dataflow, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_control_3_0_propagate, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [4:0]  io_in_control_3_0_shift, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [2:0]  io_in_id_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [2:0]  io_in_id_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [2:0]  io_in_id_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input  [2:0]  io_in_id_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_last_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_last_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_last_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_last_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [31:0] io_out_b_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [31:0] io_out_b_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [31:0] io_out_b_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [31:0] io_out_b_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [31:0] io_out_c_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [31:0] io_out_c_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [31:0] io_out_c_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [31:0] io_out_c_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_valid_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_valid_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_valid_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  input         io_in_valid_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_valid_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_valid_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_valid_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_valid_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_control_0_0_dataflow, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_control_0_0_propagate, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [4:0]  io_out_control_0_0_shift, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_control_1_0_dataflow, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_control_1_0_propagate, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [4:0]  io_out_control_1_0_shift, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_control_2_0_dataflow, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_control_2_0_propagate, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [4:0]  io_out_control_2_0_shift, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_control_3_0_dataflow, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_control_3_0_propagate, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [4:0]  io_out_control_3_0_shift, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [2:0]  io_out_id_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [2:0]  io_out_id_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [2:0]  io_out_id_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output [2:0]  io_out_id_3_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_last_0_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_last_1_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_last_2_0, // @[src/main/scala/gemmini/Mesh.scala 22:14]
  output        io_out_last_3_0 // @[src/main/scala/gemmini/Mesh.scala 22:14]
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
  reg [31:0] _RAND_3;
  reg [31:0] _RAND_4;
  reg [31:0] _RAND_5;
  reg [31:0] _RAND_6;
  reg [31:0] _RAND_7;
  reg [31:0] _RAND_8;
  reg [31:0] _RAND_9;
  reg [31:0] _RAND_10;
  reg [31:0] _RAND_11;
  reg [31:0] _RAND_12;
  reg [31:0] _RAND_13;
  reg [31:0] _RAND_14;
  reg [31:0] _RAND_15;
  reg [31:0] _RAND_16;
  reg [31:0] _RAND_17;
  reg [31:0] _RAND_18;
  reg [31:0] _RAND_19;
  reg [31:0] _RAND_20;
  reg [31:0] _RAND_21;
  reg [31:0] _RAND_22;
  reg [31:0] _RAND_23;
  reg [31:0] _RAND_24;
  reg [31:0] _RAND_25;
  reg [31:0] _RAND_26;
  reg [31:0] _RAND_27;
  reg [31:0] _RAND_28;
  reg [31:0] _RAND_29;
  reg [31:0] _RAND_30;
  reg [31:0] _RAND_31;
  reg [31:0] _RAND_32;
  reg [31:0] _RAND_33;
  reg [31:0] _RAND_34;
  reg [31:0] _RAND_35;
  reg [31:0] _RAND_36;
  reg [31:0] _RAND_37;
  reg [31:0] _RAND_38;
  reg [31:0] _RAND_39;
  reg [31:0] _RAND_40;
  reg [31:0] _RAND_41;
  reg [31:0] _RAND_42;
  reg [31:0] _RAND_43;
  reg [31:0] _RAND_44;
  reg [31:0] _RAND_45;
  reg [31:0] _RAND_46;
  reg [31:0] _RAND_47;
  reg [31:0] _RAND_48;
  reg [31:0] _RAND_49;
  reg [31:0] _RAND_50;
  reg [31:0] _RAND_51;
  reg [31:0] _RAND_52;
  reg [31:0] _RAND_53;
  reg [31:0] _RAND_54;
  reg [31:0] _RAND_55;
  reg [31:0] _RAND_56;
  reg [31:0] _RAND_57;
  reg [31:0] _RAND_58;
  reg [31:0] _RAND_59;
  reg [31:0] _RAND_60;
  reg [31:0] _RAND_61;
  reg [31:0] _RAND_62;
  reg [31:0] _RAND_63;
  reg [31:0] _RAND_64;
  reg [31:0] _RAND_65;
  reg [31:0] _RAND_66;
  reg [31:0] _RAND_67;
  reg [31:0] _RAND_68;
  reg [31:0] _RAND_69;
  reg [31:0] _RAND_70;
  reg [31:0] _RAND_71;
  reg [31:0] _RAND_72;
  reg [31:0] _RAND_73;
  reg [31:0] _RAND_74;
  reg [31:0] _RAND_75;
  reg [31:0] _RAND_76;
  reg [31:0] _RAND_77;
  reg [31:0] _RAND_78;
  reg [31:0] _RAND_79;
  reg [31:0] _RAND_80;
  reg [31:0] _RAND_81;
  reg [31:0] _RAND_82;
  reg [31:0] _RAND_83;
  reg [31:0] _RAND_84;
  reg [31:0] _RAND_85;
  reg [31:0] _RAND_86;
  reg [31:0] _RAND_87;
  reg [31:0] _RAND_88;
  reg [31:0] _RAND_89;
  reg [31:0] _RAND_90;
  reg [31:0] _RAND_91;
  reg [31:0] _RAND_92;
  reg [31:0] _RAND_93;
  reg [31:0] _RAND_94;
  reg [31:0] _RAND_95;
  reg [31:0] _RAND_96;
  reg [31:0] _RAND_97;
  reg [31:0] _RAND_98;
  reg [31:0] _RAND_99;
  reg [31:0] _RAND_100;
  reg [31:0] _RAND_101;
  reg [31:0] _RAND_102;
  reg [31:0] _RAND_103;
  reg [31:0] _RAND_104;
  reg [31:0] _RAND_105;
  reg [31:0] _RAND_106;
  reg [31:0] _RAND_107;
  reg [31:0] _RAND_108;
  reg [31:0] _RAND_109;
  reg [31:0] _RAND_110;
  reg [31:0] _RAND_111;
  reg [31:0] _RAND_112;
  reg [31:0] _RAND_113;
  reg [31:0] _RAND_114;
  reg [31:0] _RAND_115;
  reg [31:0] _RAND_116;
  reg [31:0] _RAND_117;
  reg [31:0] _RAND_118;
  reg [31:0] _RAND_119;
  reg [31:0] _RAND_120;
  reg [31:0] _RAND_121;
  reg [31:0] _RAND_122;
  reg [31:0] _RAND_123;
  reg [31:0] _RAND_124;
  reg [31:0] _RAND_125;
  reg [31:0] _RAND_126;
  reg [31:0] _RAND_127;
  reg [31:0] _RAND_128;
  reg [31:0] _RAND_129;
  reg [31:0] _RAND_130;
  reg [31:0] _RAND_131;
  reg [31:0] _RAND_132;
  reg [31:0] _RAND_133;
  reg [31:0] _RAND_134;
  reg [31:0] _RAND_135;
  reg [31:0] _RAND_136;
  reg [31:0] _RAND_137;
  reg [31:0] _RAND_138;
  reg [31:0] _RAND_139;
  reg [31:0] _RAND_140;
  reg [31:0] _RAND_141;
  reg [31:0] _RAND_142;
  reg [31:0] _RAND_143;
  reg [31:0] _RAND_144;
  reg [31:0] _RAND_145;
  reg [31:0] _RAND_146;
  reg [31:0] _RAND_147;
  reg [31:0] _RAND_148;
  reg [31:0] _RAND_149;
  reg [31:0] _RAND_150;
  reg [31:0] _RAND_151;
  reg [31:0] _RAND_152;
  reg [31:0] _RAND_153;
  reg [31:0] _RAND_154;
  reg [31:0] _RAND_155;
  reg [31:0] _RAND_156;
  reg [31:0] _RAND_157;
  reg [31:0] _RAND_158;
  reg [31:0] _RAND_159;
  reg [31:0] _RAND_160;
  reg [31:0] _RAND_161;
  reg [31:0] _RAND_162;
  reg [31:0] _RAND_163;
  reg [31:0] _RAND_164;
  reg [31:0] _RAND_165;
  reg [31:0] _RAND_166;
  reg [31:0] _RAND_167;
  reg [31:0] _RAND_168;
  reg [31:0] _RAND_169;
  reg [31:0] _RAND_170;
  reg [31:0] _RAND_171;
  reg [31:0] _RAND_172;
  reg [31:0] _RAND_173;
  reg [31:0] _RAND_174;
  reg [31:0] _RAND_175;
`endif // RANDOMIZE_REG_INIT
  wire  mesh_0_0_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_0_0_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_0_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_0_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_0_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_0_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_0_0_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_0_0_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_0_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_0_0_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_0_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_0_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_0_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_0_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_0_0_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_0_0_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_0_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_0_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_0_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_0_1_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_1_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_1_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_0_1_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_0_1_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_0_1_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_1_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_1_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_0_1_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_0_1_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_1_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_0_2_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_2_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_2_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_0_2_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_0_2_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_0_2_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_2_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_2_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_0_2_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_0_2_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_2_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_0_3_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_3_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_3_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_0_3_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_0_3_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_0_3_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_3_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_0_3_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_0_3_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_0_3_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_0_3_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_1_0_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_0_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_0_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_1_0_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_1_0_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_1_0_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_0_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_0_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_1_0_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_1_0_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_0_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_1_1_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_1_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_1_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_1_1_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_1_1_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_1_1_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_1_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_1_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_1_1_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_1_1_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_1_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_1_2_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_2_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_2_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_1_2_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_1_2_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_1_2_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_2_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_2_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_1_2_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_1_2_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_2_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_1_3_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_3_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_3_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_1_3_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_1_3_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_1_3_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_3_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_1_3_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_1_3_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_1_3_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_1_3_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_2_0_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_0_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_0_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_2_0_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_2_0_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_2_0_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_0_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_0_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_2_0_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_2_0_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_0_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_2_1_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_1_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_1_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_2_1_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_2_1_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_2_1_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_1_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_1_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_2_1_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_2_1_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_1_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_2_2_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_2_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_2_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_2_2_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_2_2_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_2_2_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_2_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_2_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_2_2_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_2_2_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_2_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_2_3_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_3_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_3_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_2_3_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_2_3_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_2_3_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_3_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_2_3_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_2_3_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_2_3_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_2_3_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_3_0_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_0_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_0_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_3_0_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_3_0_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_3_0_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_0_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_0_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_3_0_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_3_0_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_0_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_3_1_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_1_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_1_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_3_1_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_3_1_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_3_1_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_1_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_1_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_3_1_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_3_1_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_1_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_3_2_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_2_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_2_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_3_2_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_3_2_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_3_2_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_2_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_2_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_3_2_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_3_2_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_2_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_clock; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_3_3_io_in_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_3_io_in_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_3_io_in_d_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_io_in_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_io_in_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_3_3_io_in_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_3_3_io_in_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_io_in_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [7:0] mesh_3_3_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_3_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [31:0] mesh_3_3_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [4:0] mesh_3_3_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire [2:0] mesh_3_3_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_io_in_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  wire  mesh_3_3_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 39:71]
  reg [7:0] r_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_1_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_2_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_3_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_4_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_5_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_6_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_7_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_8_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_9_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_10_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_11_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_12_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_13_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_14_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] r_15_0; // @[src/main/scala/gemmini/Mesh.scala 53:38]
  reg [7:0] pipe_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_1_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_2_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_3_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [7:0] pipe_b_4_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_5_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_6_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_7_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [7:0] pipe_b_8_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_9_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_10_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_11_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [7:0] pipe_b_12_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_13_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_14_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_15_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [7:0] pipe_b_16_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_17_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_18_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_19_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [7:0] pipe_b_20_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_21_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_22_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_23_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [7:0] pipe_b_24_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_25_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_26_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_27_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [7:0] pipe_b_28_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_29_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_30_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [31:0] pipe_b_31_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_0_0_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_0_0_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_0_0_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_1_0_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_1_0_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_1_0_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_2_0_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_2_0_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_2_0_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_3_0_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_3_0_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_3_0_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_0_1_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_0_1_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_0_1_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_1_1_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_1_1_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_1_1_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_2_1_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_2_1_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_2_1_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_3_1_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_3_1_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_3_1_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_0_2_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_0_2_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_0_2_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_1_2_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_1_2_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_1_2_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_2_2_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_2_2_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_2_2_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_3_2_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_3_2_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_3_2_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_0_3_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_0_3_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_0_3_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_1_3_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_1_3_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_1_3_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_2_3_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_2_3_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_2_3_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg [4:0] mesh_3_3_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_3_3_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  mesh_3_3_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
  reg  r_16_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_17_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_18_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_19_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_20_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_21_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_22_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_23_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_24_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_25_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_26_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_27_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_28_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_29_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_30_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg  r_31_0; // @[src/main/scala/gemmini/Mesh.scala 94:42]
  reg [2:0] r_32_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_33_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_34_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_35_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_36_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_37_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_38_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_39_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_40_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_41_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_42_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_43_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_44_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_45_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_46_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg [2:0] r_47_0; // @[src/main/scala/gemmini/Mesh.scala 103:39]
  reg  r_48_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_49_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_50_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_51_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_52_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_53_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_54_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_55_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_56_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_57_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_58_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_59_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_60_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_61_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_62_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg  r_63_0; // @[src/main/scala/gemmini/Mesh.scala 112:41]
  reg [31:0] r_64_0; // @[src/main/scala/gemmini/Mesh.scala 122:23]
  reg [31:0] r_65_0; // @[src/main/scala/gemmini/Mesh.scala 123:23]
  reg  r_66_0; // @[src/main/scala/gemmini/Mesh.scala 124:23]
  reg  r_67_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg  r_67_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg [4:0] r_67_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg [2:0] r_68_0; // @[src/main/scala/gemmini/Mesh.scala 126:24]
  reg  r_69_0; // @[src/main/scala/gemmini/Mesh.scala 127:26]
  reg [31:0] r_70_0; // @[src/main/scala/gemmini/Mesh.scala 122:23]
  reg [31:0] r_71_0; // @[src/main/scala/gemmini/Mesh.scala 123:23]
  reg  r_72_0; // @[src/main/scala/gemmini/Mesh.scala 124:23]
  reg  r_73_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg  r_73_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg [4:0] r_73_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg [2:0] r_74_0; // @[src/main/scala/gemmini/Mesh.scala 126:24]
  reg  r_75_0; // @[src/main/scala/gemmini/Mesh.scala 127:26]
  reg [31:0] r_76_0; // @[src/main/scala/gemmini/Mesh.scala 122:23]
  reg [31:0] r_77_0; // @[src/main/scala/gemmini/Mesh.scala 123:23]
  reg  r_78_0; // @[src/main/scala/gemmini/Mesh.scala 124:23]
  reg  r_79_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg  r_79_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg [4:0] r_79_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg [2:0] r_80_0; // @[src/main/scala/gemmini/Mesh.scala 126:24]
  reg  r_81_0; // @[src/main/scala/gemmini/Mesh.scala 127:26]
  reg [31:0] r_82_0; // @[src/main/scala/gemmini/Mesh.scala 122:23]
  reg [31:0] r_83_0; // @[src/main/scala/gemmini/Mesh.scala 123:23]
  reg  r_84_0; // @[src/main/scala/gemmini/Mesh.scala 124:23]
  reg  r_85_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg  r_85_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg [4:0] r_85_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:26]
  reg [2:0] r_86_0; // @[src/main/scala/gemmini/Mesh.scala 126:24]
  reg  r_87_0; // @[src/main/scala/gemmini/Mesh.scala 127:26]
  Tile mesh_0_0 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_0_0_clock),
    .io_in_a_0(mesh_0_0_io_in_a_0),
    .io_in_b_0(mesh_0_0_io_in_b_0),
    .io_in_d_0(mesh_0_0_io_in_d_0),
    .io_in_control_0_dataflow(mesh_0_0_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_0_0_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_0_0_io_in_control_0_shift),
    .io_in_id_0(mesh_0_0_io_in_id_0),
    .io_in_last_0(mesh_0_0_io_in_last_0),
    .io_out_a_0(mesh_0_0_io_out_a_0),
    .io_out_c_0(mesh_0_0_io_out_c_0),
    .io_out_b_0(mesh_0_0_io_out_b_0),
    .io_out_control_0_dataflow(mesh_0_0_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_0_0_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_0_0_io_out_control_0_shift),
    .io_out_id_0(mesh_0_0_io_out_id_0),
    .io_out_last_0(mesh_0_0_io_out_last_0),
    .io_in_valid_0(mesh_0_0_io_in_valid_0),
    .io_out_valid_0(mesh_0_0_io_out_valid_0)
  );
  Tile mesh_0_1 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_0_1_clock),
    .io_in_a_0(mesh_0_1_io_in_a_0),
    .io_in_b_0(mesh_0_1_io_in_b_0),
    .io_in_d_0(mesh_0_1_io_in_d_0),
    .io_in_control_0_dataflow(mesh_0_1_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_0_1_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_0_1_io_in_control_0_shift),
    .io_in_id_0(mesh_0_1_io_in_id_0),
    .io_in_last_0(mesh_0_1_io_in_last_0),
    .io_out_a_0(mesh_0_1_io_out_a_0),
    .io_out_c_0(mesh_0_1_io_out_c_0),
    .io_out_b_0(mesh_0_1_io_out_b_0),
    .io_out_control_0_dataflow(mesh_0_1_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_0_1_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_0_1_io_out_control_0_shift),
    .io_out_id_0(mesh_0_1_io_out_id_0),
    .io_out_last_0(mesh_0_1_io_out_last_0),
    .io_in_valid_0(mesh_0_1_io_in_valid_0),
    .io_out_valid_0(mesh_0_1_io_out_valid_0)
  );
  Tile mesh_0_2 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_0_2_clock),
    .io_in_a_0(mesh_0_2_io_in_a_0),
    .io_in_b_0(mesh_0_2_io_in_b_0),
    .io_in_d_0(mesh_0_2_io_in_d_0),
    .io_in_control_0_dataflow(mesh_0_2_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_0_2_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_0_2_io_in_control_0_shift),
    .io_in_id_0(mesh_0_2_io_in_id_0),
    .io_in_last_0(mesh_0_2_io_in_last_0),
    .io_out_a_0(mesh_0_2_io_out_a_0),
    .io_out_c_0(mesh_0_2_io_out_c_0),
    .io_out_b_0(mesh_0_2_io_out_b_0),
    .io_out_control_0_dataflow(mesh_0_2_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_0_2_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_0_2_io_out_control_0_shift),
    .io_out_id_0(mesh_0_2_io_out_id_0),
    .io_out_last_0(mesh_0_2_io_out_last_0),
    .io_in_valid_0(mesh_0_2_io_in_valid_0),
    .io_out_valid_0(mesh_0_2_io_out_valid_0)
  );
  Tile mesh_0_3 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_0_3_clock),
    .io_in_a_0(mesh_0_3_io_in_a_0),
    .io_in_b_0(mesh_0_3_io_in_b_0),
    .io_in_d_0(mesh_0_3_io_in_d_0),
    .io_in_control_0_dataflow(mesh_0_3_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_0_3_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_0_3_io_in_control_0_shift),
    .io_in_id_0(mesh_0_3_io_in_id_0),
    .io_in_last_0(mesh_0_3_io_in_last_0),
    .io_out_a_0(mesh_0_3_io_out_a_0),
    .io_out_c_0(mesh_0_3_io_out_c_0),
    .io_out_b_0(mesh_0_3_io_out_b_0),
    .io_out_control_0_dataflow(mesh_0_3_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_0_3_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_0_3_io_out_control_0_shift),
    .io_out_id_0(mesh_0_3_io_out_id_0),
    .io_out_last_0(mesh_0_3_io_out_last_0),
    .io_in_valid_0(mesh_0_3_io_in_valid_0),
    .io_out_valid_0(mesh_0_3_io_out_valid_0)
  );
  Tile mesh_1_0 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_1_0_clock),
    .io_in_a_0(mesh_1_0_io_in_a_0),
    .io_in_b_0(mesh_1_0_io_in_b_0),
    .io_in_d_0(mesh_1_0_io_in_d_0),
    .io_in_control_0_dataflow(mesh_1_0_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_1_0_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_1_0_io_in_control_0_shift),
    .io_in_id_0(mesh_1_0_io_in_id_0),
    .io_in_last_0(mesh_1_0_io_in_last_0),
    .io_out_a_0(mesh_1_0_io_out_a_0),
    .io_out_c_0(mesh_1_0_io_out_c_0),
    .io_out_b_0(mesh_1_0_io_out_b_0),
    .io_out_control_0_dataflow(mesh_1_0_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_1_0_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_1_0_io_out_control_0_shift),
    .io_out_id_0(mesh_1_0_io_out_id_0),
    .io_out_last_0(mesh_1_0_io_out_last_0),
    .io_in_valid_0(mesh_1_0_io_in_valid_0),
    .io_out_valid_0(mesh_1_0_io_out_valid_0)
  );
  Tile mesh_1_1 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_1_1_clock),
    .io_in_a_0(mesh_1_1_io_in_a_0),
    .io_in_b_0(mesh_1_1_io_in_b_0),
    .io_in_d_0(mesh_1_1_io_in_d_0),
    .io_in_control_0_dataflow(mesh_1_1_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_1_1_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_1_1_io_in_control_0_shift),
    .io_in_id_0(mesh_1_1_io_in_id_0),
    .io_in_last_0(mesh_1_1_io_in_last_0),
    .io_out_a_0(mesh_1_1_io_out_a_0),
    .io_out_c_0(mesh_1_1_io_out_c_0),
    .io_out_b_0(mesh_1_1_io_out_b_0),
    .io_out_control_0_dataflow(mesh_1_1_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_1_1_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_1_1_io_out_control_0_shift),
    .io_out_id_0(mesh_1_1_io_out_id_0),
    .io_out_last_0(mesh_1_1_io_out_last_0),
    .io_in_valid_0(mesh_1_1_io_in_valid_0),
    .io_out_valid_0(mesh_1_1_io_out_valid_0)
  );
  Tile mesh_1_2 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_1_2_clock),
    .io_in_a_0(mesh_1_2_io_in_a_0),
    .io_in_b_0(mesh_1_2_io_in_b_0),
    .io_in_d_0(mesh_1_2_io_in_d_0),
    .io_in_control_0_dataflow(mesh_1_2_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_1_2_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_1_2_io_in_control_0_shift),
    .io_in_id_0(mesh_1_2_io_in_id_0),
    .io_in_last_0(mesh_1_2_io_in_last_0),
    .io_out_a_0(mesh_1_2_io_out_a_0),
    .io_out_c_0(mesh_1_2_io_out_c_0),
    .io_out_b_0(mesh_1_2_io_out_b_0),
    .io_out_control_0_dataflow(mesh_1_2_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_1_2_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_1_2_io_out_control_0_shift),
    .io_out_id_0(mesh_1_2_io_out_id_0),
    .io_out_last_0(mesh_1_2_io_out_last_0),
    .io_in_valid_0(mesh_1_2_io_in_valid_0),
    .io_out_valid_0(mesh_1_2_io_out_valid_0)
  );
  Tile mesh_1_3 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_1_3_clock),
    .io_in_a_0(mesh_1_3_io_in_a_0),
    .io_in_b_0(mesh_1_3_io_in_b_0),
    .io_in_d_0(mesh_1_3_io_in_d_0),
    .io_in_control_0_dataflow(mesh_1_3_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_1_3_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_1_3_io_in_control_0_shift),
    .io_in_id_0(mesh_1_3_io_in_id_0),
    .io_in_last_0(mesh_1_3_io_in_last_0),
    .io_out_a_0(mesh_1_3_io_out_a_0),
    .io_out_c_0(mesh_1_3_io_out_c_0),
    .io_out_b_0(mesh_1_3_io_out_b_0),
    .io_out_control_0_dataflow(mesh_1_3_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_1_3_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_1_3_io_out_control_0_shift),
    .io_out_id_0(mesh_1_3_io_out_id_0),
    .io_out_last_0(mesh_1_3_io_out_last_0),
    .io_in_valid_0(mesh_1_3_io_in_valid_0),
    .io_out_valid_0(mesh_1_3_io_out_valid_0)
  );
  Tile mesh_2_0 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_2_0_clock),
    .io_in_a_0(mesh_2_0_io_in_a_0),
    .io_in_b_0(mesh_2_0_io_in_b_0),
    .io_in_d_0(mesh_2_0_io_in_d_0),
    .io_in_control_0_dataflow(mesh_2_0_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_2_0_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_2_0_io_in_control_0_shift),
    .io_in_id_0(mesh_2_0_io_in_id_0),
    .io_in_last_0(mesh_2_0_io_in_last_0),
    .io_out_a_0(mesh_2_0_io_out_a_0),
    .io_out_c_0(mesh_2_0_io_out_c_0),
    .io_out_b_0(mesh_2_0_io_out_b_0),
    .io_out_control_0_dataflow(mesh_2_0_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_2_0_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_2_0_io_out_control_0_shift),
    .io_out_id_0(mesh_2_0_io_out_id_0),
    .io_out_last_0(mesh_2_0_io_out_last_0),
    .io_in_valid_0(mesh_2_0_io_in_valid_0),
    .io_out_valid_0(mesh_2_0_io_out_valid_0)
  );
  Tile mesh_2_1 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_2_1_clock),
    .io_in_a_0(mesh_2_1_io_in_a_0),
    .io_in_b_0(mesh_2_1_io_in_b_0),
    .io_in_d_0(mesh_2_1_io_in_d_0),
    .io_in_control_0_dataflow(mesh_2_1_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_2_1_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_2_1_io_in_control_0_shift),
    .io_in_id_0(mesh_2_1_io_in_id_0),
    .io_in_last_0(mesh_2_1_io_in_last_0),
    .io_out_a_0(mesh_2_1_io_out_a_0),
    .io_out_c_0(mesh_2_1_io_out_c_0),
    .io_out_b_0(mesh_2_1_io_out_b_0),
    .io_out_control_0_dataflow(mesh_2_1_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_2_1_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_2_1_io_out_control_0_shift),
    .io_out_id_0(mesh_2_1_io_out_id_0),
    .io_out_last_0(mesh_2_1_io_out_last_0),
    .io_in_valid_0(mesh_2_1_io_in_valid_0),
    .io_out_valid_0(mesh_2_1_io_out_valid_0)
  );
  Tile mesh_2_2 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_2_2_clock),
    .io_in_a_0(mesh_2_2_io_in_a_0),
    .io_in_b_0(mesh_2_2_io_in_b_0),
    .io_in_d_0(mesh_2_2_io_in_d_0),
    .io_in_control_0_dataflow(mesh_2_2_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_2_2_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_2_2_io_in_control_0_shift),
    .io_in_id_0(mesh_2_2_io_in_id_0),
    .io_in_last_0(mesh_2_2_io_in_last_0),
    .io_out_a_0(mesh_2_2_io_out_a_0),
    .io_out_c_0(mesh_2_2_io_out_c_0),
    .io_out_b_0(mesh_2_2_io_out_b_0),
    .io_out_control_0_dataflow(mesh_2_2_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_2_2_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_2_2_io_out_control_0_shift),
    .io_out_id_0(mesh_2_2_io_out_id_0),
    .io_out_last_0(mesh_2_2_io_out_last_0),
    .io_in_valid_0(mesh_2_2_io_in_valid_0),
    .io_out_valid_0(mesh_2_2_io_out_valid_0)
  );
  Tile mesh_2_3 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_2_3_clock),
    .io_in_a_0(mesh_2_3_io_in_a_0),
    .io_in_b_0(mesh_2_3_io_in_b_0),
    .io_in_d_0(mesh_2_3_io_in_d_0),
    .io_in_control_0_dataflow(mesh_2_3_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_2_3_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_2_3_io_in_control_0_shift),
    .io_in_id_0(mesh_2_3_io_in_id_0),
    .io_in_last_0(mesh_2_3_io_in_last_0),
    .io_out_a_0(mesh_2_3_io_out_a_0),
    .io_out_c_0(mesh_2_3_io_out_c_0),
    .io_out_b_0(mesh_2_3_io_out_b_0),
    .io_out_control_0_dataflow(mesh_2_3_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_2_3_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_2_3_io_out_control_0_shift),
    .io_out_id_0(mesh_2_3_io_out_id_0),
    .io_out_last_0(mesh_2_3_io_out_last_0),
    .io_in_valid_0(mesh_2_3_io_in_valid_0),
    .io_out_valid_0(mesh_2_3_io_out_valid_0)
  );
  Tile_12 mesh_3_0 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_3_0_clock),
    .io_in_a_0(mesh_3_0_io_in_a_0),
    .io_in_b_0(mesh_3_0_io_in_b_0),
    .io_in_d_0(mesh_3_0_io_in_d_0),
    .io_in_control_0_dataflow(mesh_3_0_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_3_0_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_3_0_io_in_control_0_shift),
    .io_in_id_0(mesh_3_0_io_in_id_0),
    .io_in_last_0(mesh_3_0_io_in_last_0),
    .io_out_a_0(mesh_3_0_io_out_a_0),
    .io_out_c_0(mesh_3_0_io_out_c_0),
    .io_out_b_0(mesh_3_0_io_out_b_0),
    .io_out_control_0_dataflow(mesh_3_0_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_3_0_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_3_0_io_out_control_0_shift),
    .io_out_id_0(mesh_3_0_io_out_id_0),
    .io_out_last_0(mesh_3_0_io_out_last_0),
    .io_in_valid_0(mesh_3_0_io_in_valid_0),
    .io_out_valid_0(mesh_3_0_io_out_valid_0)
  );
  Tile_12 mesh_3_1 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_3_1_clock),
    .io_in_a_0(mesh_3_1_io_in_a_0),
    .io_in_b_0(mesh_3_1_io_in_b_0),
    .io_in_d_0(mesh_3_1_io_in_d_0),
    .io_in_control_0_dataflow(mesh_3_1_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_3_1_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_3_1_io_in_control_0_shift),
    .io_in_id_0(mesh_3_1_io_in_id_0),
    .io_in_last_0(mesh_3_1_io_in_last_0),
    .io_out_a_0(mesh_3_1_io_out_a_0),
    .io_out_c_0(mesh_3_1_io_out_c_0),
    .io_out_b_0(mesh_3_1_io_out_b_0),
    .io_out_control_0_dataflow(mesh_3_1_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_3_1_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_3_1_io_out_control_0_shift),
    .io_out_id_0(mesh_3_1_io_out_id_0),
    .io_out_last_0(mesh_3_1_io_out_last_0),
    .io_in_valid_0(mesh_3_1_io_in_valid_0),
    .io_out_valid_0(mesh_3_1_io_out_valid_0)
  );
  Tile_12 mesh_3_2 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_3_2_clock),
    .io_in_a_0(mesh_3_2_io_in_a_0),
    .io_in_b_0(mesh_3_2_io_in_b_0),
    .io_in_d_0(mesh_3_2_io_in_d_0),
    .io_in_control_0_dataflow(mesh_3_2_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_3_2_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_3_2_io_in_control_0_shift),
    .io_in_id_0(mesh_3_2_io_in_id_0),
    .io_in_last_0(mesh_3_2_io_in_last_0),
    .io_out_a_0(mesh_3_2_io_out_a_0),
    .io_out_c_0(mesh_3_2_io_out_c_0),
    .io_out_b_0(mesh_3_2_io_out_b_0),
    .io_out_control_0_dataflow(mesh_3_2_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_3_2_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_3_2_io_out_control_0_shift),
    .io_out_id_0(mesh_3_2_io_out_id_0),
    .io_out_last_0(mesh_3_2_io_out_last_0),
    .io_in_valid_0(mesh_3_2_io_in_valid_0),
    .io_out_valid_0(mesh_3_2_io_out_valid_0)
  );
  Tile_12 mesh_3_3 ( // @[src/main/scala/gemmini/Mesh.scala 39:71]
    .clock(mesh_3_3_clock),
    .io_in_a_0(mesh_3_3_io_in_a_0),
    .io_in_b_0(mesh_3_3_io_in_b_0),
    .io_in_d_0(mesh_3_3_io_in_d_0),
    .io_in_control_0_dataflow(mesh_3_3_io_in_control_0_dataflow),
    .io_in_control_0_propagate(mesh_3_3_io_in_control_0_propagate),
    .io_in_control_0_shift(mesh_3_3_io_in_control_0_shift),
    .io_in_id_0(mesh_3_3_io_in_id_0),
    .io_in_last_0(mesh_3_3_io_in_last_0),
    .io_out_a_0(mesh_3_3_io_out_a_0),
    .io_out_c_0(mesh_3_3_io_out_c_0),
    .io_out_b_0(mesh_3_3_io_out_b_0),
    .io_out_control_0_dataflow(mesh_3_3_io_out_control_0_dataflow),
    .io_out_control_0_propagate(mesh_3_3_io_out_control_0_propagate),
    .io_out_control_0_shift(mesh_3_3_io_out_control_0_shift),
    .io_out_id_0(mesh_3_3_io_out_id_0),
    .io_out_last_0(mesh_3_3_io_out_last_0),
    .io_in_valid_0(mesh_3_3_io_in_valid_0),
    .io_out_valid_0(mesh_3_3_io_out_valid_0)
  );
  assign io_out_b_0_0 = r_64_0; // @[src/main/scala/gemmini/Mesh.scala 122:7]
  assign io_out_b_1_0 = r_70_0; // @[src/main/scala/gemmini/Mesh.scala 122:7]
  assign io_out_b_2_0 = r_76_0; // @[src/main/scala/gemmini/Mesh.scala 122:7]
  assign io_out_b_3_0 = r_82_0; // @[src/main/scala/gemmini/Mesh.scala 122:7]
  assign io_out_c_0_0 = r_65_0; // @[src/main/scala/gemmini/Mesh.scala 123:7]
  assign io_out_c_1_0 = r_71_0; // @[src/main/scala/gemmini/Mesh.scala 123:7]
  assign io_out_c_2_0 = r_77_0; // @[src/main/scala/gemmini/Mesh.scala 123:7]
  assign io_out_c_3_0 = r_83_0; // @[src/main/scala/gemmini/Mesh.scala 123:7]
  assign io_out_valid_0_0 = r_66_0; // @[src/main/scala/gemmini/Mesh.scala 124:7]
  assign io_out_valid_1_0 = r_72_0; // @[src/main/scala/gemmini/Mesh.scala 124:7]
  assign io_out_valid_2_0 = r_78_0; // @[src/main/scala/gemmini/Mesh.scala 124:7]
  assign io_out_valid_3_0 = r_84_0; // @[src/main/scala/gemmini/Mesh.scala 124:7]
  assign io_out_control_0_0_dataflow = r_67_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_0_0_propagate = r_67_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_0_0_shift = r_67_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_1_0_dataflow = r_73_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_1_0_propagate = r_73_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_1_0_shift = r_73_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_2_0_dataflow = r_79_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_2_0_propagate = r_79_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_2_0_shift = r_79_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_3_0_dataflow = r_85_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_3_0_propagate = r_85_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_control_3_0_shift = r_85_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:10]
  assign io_out_id_0_0 = r_68_0; // @[src/main/scala/gemmini/Mesh.scala 126:8]
  assign io_out_id_1_0 = r_74_0; // @[src/main/scala/gemmini/Mesh.scala 126:8]
  assign io_out_id_2_0 = r_80_0; // @[src/main/scala/gemmini/Mesh.scala 126:8]
  assign io_out_id_3_0 = r_86_0; // @[src/main/scala/gemmini/Mesh.scala 126:8]
  assign io_out_last_0_0 = r_69_0; // @[src/main/scala/gemmini/Mesh.scala 127:10]
  assign io_out_last_1_0 = r_75_0; // @[src/main/scala/gemmini/Mesh.scala 127:10]
  assign io_out_last_2_0 = r_81_0; // @[src/main/scala/gemmini/Mesh.scala 127:10]
  assign io_out_last_3_0 = r_87_0; // @[src/main/scala/gemmini/Mesh.scala 127:10]
  assign mesh_0_0_clock = clock;
  assign mesh_0_0_io_in_a_0 = r_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_0_0_io_in_b_0 = {{24{pipe_b_0[7]}},pipe_b_0}; // @[src/main/scala/gemmini/Mesh.scala 62:22]
  assign mesh_0_0_io_in_d_0 = {{24{pipe_b_16_0[7]}},pipe_b_16_0}; // @[src/main/scala/gemmini/Mesh.scala 71:22]
  assign mesh_0_0_io_in_control_0_dataflow = mesh_0_0_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_0_io_in_control_0_propagate = mesh_0_0_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_0_io_in_control_0_shift = mesh_0_0_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_0_io_in_id_0 = r_32_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_0_0_io_in_last_0 = r_48_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_0_0_io_in_valid_0 = r_16_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_0_1_clock = clock;
  assign mesh_0_1_io_in_a_0 = r_1_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_0_1_io_in_b_0 = {{24{pipe_b_4_0[7]}},pipe_b_4_0}; // @[src/main/scala/gemmini/Mesh.scala 62:22]
  assign mesh_0_1_io_in_d_0 = {{24{pipe_b_20_0[7]}},pipe_b_20_0}; // @[src/main/scala/gemmini/Mesh.scala 71:22]
  assign mesh_0_1_io_in_control_0_dataflow = mesh_0_1_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_1_io_in_control_0_propagate = mesh_0_1_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_1_io_in_control_0_shift = mesh_0_1_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_1_io_in_id_0 = r_36_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_0_1_io_in_last_0 = r_52_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_0_1_io_in_valid_0 = r_20_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_0_2_clock = clock;
  assign mesh_0_2_io_in_a_0 = r_2_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_0_2_io_in_b_0 = {{24{pipe_b_8_0[7]}},pipe_b_8_0}; // @[src/main/scala/gemmini/Mesh.scala 62:22]
  assign mesh_0_2_io_in_d_0 = {{24{pipe_b_24_0[7]}},pipe_b_24_0}; // @[src/main/scala/gemmini/Mesh.scala 71:22]
  assign mesh_0_2_io_in_control_0_dataflow = mesh_0_2_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_2_io_in_control_0_propagate = mesh_0_2_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_2_io_in_control_0_shift = mesh_0_2_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_2_io_in_id_0 = r_40_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_0_2_io_in_last_0 = r_56_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_0_2_io_in_valid_0 = r_24_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_0_3_clock = clock;
  assign mesh_0_3_io_in_a_0 = r_3_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_0_3_io_in_b_0 = {{24{pipe_b_12_0[7]}},pipe_b_12_0}; // @[src/main/scala/gemmini/Mesh.scala 62:22]
  assign mesh_0_3_io_in_d_0 = {{24{pipe_b_28_0[7]}},pipe_b_28_0}; // @[src/main/scala/gemmini/Mesh.scala 71:22]
  assign mesh_0_3_io_in_control_0_dataflow = mesh_0_3_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_3_io_in_control_0_propagate = mesh_0_3_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_3_io_in_control_0_shift = mesh_0_3_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_0_3_io_in_id_0 = r_44_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_0_3_io_in_last_0 = r_60_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_0_3_io_in_valid_0 = r_28_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_1_0_clock = clock;
  assign mesh_1_0_io_in_a_0 = r_4_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_1_0_io_in_b_0 = pipe_b_1_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_0_io_in_d_0 = pipe_b_17_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_0_io_in_control_0_dataflow = mesh_1_0_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_0_io_in_control_0_propagate = mesh_1_0_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_0_io_in_control_0_shift = mesh_1_0_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_0_io_in_id_0 = r_33_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_1_0_io_in_last_0 = r_49_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_1_0_io_in_valid_0 = r_17_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_1_1_clock = clock;
  assign mesh_1_1_io_in_a_0 = r_5_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_1_1_io_in_b_0 = pipe_b_5_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_1_io_in_d_0 = pipe_b_21_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_1_io_in_control_0_dataflow = mesh_1_1_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_1_io_in_control_0_propagate = mesh_1_1_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_1_io_in_control_0_shift = mesh_1_1_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_1_io_in_id_0 = r_37_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_1_1_io_in_last_0 = r_53_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_1_1_io_in_valid_0 = r_21_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_1_2_clock = clock;
  assign mesh_1_2_io_in_a_0 = r_6_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_1_2_io_in_b_0 = pipe_b_9_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_2_io_in_d_0 = pipe_b_25_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_2_io_in_control_0_dataflow = mesh_1_2_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_2_io_in_control_0_propagate = mesh_1_2_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_2_io_in_control_0_shift = mesh_1_2_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_2_io_in_id_0 = r_41_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_1_2_io_in_last_0 = r_57_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_1_2_io_in_valid_0 = r_25_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_1_3_clock = clock;
  assign mesh_1_3_io_in_a_0 = r_7_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_1_3_io_in_b_0 = pipe_b_13_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_3_io_in_d_0 = pipe_b_29_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_3_io_in_control_0_dataflow = mesh_1_3_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_3_io_in_control_0_propagate = mesh_1_3_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_3_io_in_control_0_shift = mesh_1_3_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_1_3_io_in_id_0 = r_45_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_1_3_io_in_last_0 = r_61_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_1_3_io_in_valid_0 = r_29_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_2_0_clock = clock;
  assign mesh_2_0_io_in_a_0 = r_8_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_2_0_io_in_b_0 = pipe_b_2_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_0_io_in_d_0 = pipe_b_18_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_0_io_in_control_0_dataflow = mesh_2_0_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_0_io_in_control_0_propagate = mesh_2_0_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_0_io_in_control_0_shift = mesh_2_0_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_0_io_in_id_0 = r_34_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_2_0_io_in_last_0 = r_50_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_2_0_io_in_valid_0 = r_18_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_2_1_clock = clock;
  assign mesh_2_1_io_in_a_0 = r_9_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_2_1_io_in_b_0 = pipe_b_6_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_1_io_in_d_0 = pipe_b_22_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_1_io_in_control_0_dataflow = mesh_2_1_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_1_io_in_control_0_propagate = mesh_2_1_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_1_io_in_control_0_shift = mesh_2_1_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_1_io_in_id_0 = r_38_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_2_1_io_in_last_0 = r_54_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_2_1_io_in_valid_0 = r_22_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_2_2_clock = clock;
  assign mesh_2_2_io_in_a_0 = r_10_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_2_2_io_in_b_0 = pipe_b_10_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_2_io_in_d_0 = pipe_b_26_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_2_io_in_control_0_dataflow = mesh_2_2_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_2_io_in_control_0_propagate = mesh_2_2_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_2_io_in_control_0_shift = mesh_2_2_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_2_io_in_id_0 = r_42_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_2_2_io_in_last_0 = r_58_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_2_2_io_in_valid_0 = r_26_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_2_3_clock = clock;
  assign mesh_2_3_io_in_a_0 = r_11_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_2_3_io_in_b_0 = pipe_b_14_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_3_io_in_d_0 = pipe_b_30_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_3_io_in_control_0_dataflow = mesh_2_3_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_3_io_in_control_0_propagate = mesh_2_3_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_3_io_in_control_0_shift = mesh_2_3_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_2_3_io_in_id_0 = r_46_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_2_3_io_in_last_0 = r_62_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_2_3_io_in_valid_0 = r_30_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_3_0_clock = clock;
  assign mesh_3_0_io_in_a_0 = r_12_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_3_0_io_in_b_0 = pipe_b_3_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_0_io_in_d_0 = pipe_b_19_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_0_io_in_control_0_dataflow = mesh_3_0_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_0_io_in_control_0_propagate = mesh_3_0_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_0_io_in_control_0_shift = mesh_3_0_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_0_io_in_id_0 = r_35_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_3_0_io_in_last_0 = r_51_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_3_0_io_in_valid_0 = r_19_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_3_1_clock = clock;
  assign mesh_3_1_io_in_a_0 = r_13_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_3_1_io_in_b_0 = pipe_b_7_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_1_io_in_d_0 = pipe_b_23_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_1_io_in_control_0_dataflow = mesh_3_1_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_1_io_in_control_0_propagate = mesh_3_1_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_1_io_in_control_0_shift = mesh_3_1_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_1_io_in_id_0 = r_39_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_3_1_io_in_last_0 = r_55_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_3_1_io_in_valid_0 = r_23_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_3_2_clock = clock;
  assign mesh_3_2_io_in_a_0 = r_14_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_3_2_io_in_b_0 = pipe_b_11_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_2_io_in_d_0 = pipe_b_27_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_2_io_in_control_0_dataflow = mesh_3_2_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_2_io_in_control_0_propagate = mesh_3_2_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_2_io_in_control_0_shift = mesh_3_2_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_2_io_in_id_0 = r_43_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_3_2_io_in_last_0 = r_59_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_3_2_io_in_valid_0 = r_27_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  assign mesh_3_3_clock = clock;
  assign mesh_3_3_io_in_a_0 = r_15_0; // @[src/main/scala/gemmini/Mesh.scala 53:22]
  assign mesh_3_3_io_in_b_0 = pipe_b_15_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_3_io_in_d_0 = pipe_b_31_0; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_3_io_in_control_0_dataflow = mesh_3_3_io_in_control_0_dataflow_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_3_io_in_control_0_propagate = mesh_3_3_io_in_control_0_propagate_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_3_io_in_control_0_shift = mesh_3_3_io_in_control_0_shift_pipe_b; // @[src/main/scala/chisel3/util/Valid.scala 123:21 125:16]
  assign mesh_3_3_io_in_id_0 = r_47_0; // @[src/main/scala/gemmini/Mesh.scala 103:23]
  assign mesh_3_3_io_in_last_0 = r_63_0; // @[src/main/scala/gemmini/Mesh.scala 112:25]
  assign mesh_3_3_io_in_valid_0 = r_31_0; // @[src/main/scala/gemmini/Mesh.scala 94:26]
  always @(posedge clock) begin
    r_0 <= io_in_a_0_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_1_0 <= mesh_0_0_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_2_0 <= mesh_0_1_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_3_0 <= mesh_0_2_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_4_0 <= io_in_a_1_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_5_0 <= mesh_1_0_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_6_0 <= mesh_1_1_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_7_0 <= mesh_1_2_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_8_0 <= io_in_a_2_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_9_0 <= mesh_2_0_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_10_0 <= mesh_2_1_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_11_0 <= mesh_2_2_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_12_0 <= io_in_a_3_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_13_0 <= mesh_3_0_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_14_0 <= mesh_3_1_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    r_15_0 <= mesh_3_2_io_out_a_0; // @[src/main/scala/gemmini/Mesh.scala 53:{38,38,38}]
    if (io_in_valid_0_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_0 <= io_in_b_0_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_1_0 <= mesh_0_0_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_2_0 <= mesh_1_0_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_3_0 <= mesh_2_0_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_1_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_4_0 <= io_in_b_1_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_5_0 <= mesh_0_1_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_6_0 <= mesh_1_1_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_7_0 <= mesh_2_1_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_2_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_8_0 <= io_in_b_2_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_9_0 <= mesh_0_2_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_10_0 <= mesh_1_2_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_11_0 <= mesh_2_2_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_3_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_12_0 <= io_in_b_3_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_13_0 <= mesh_0_3_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_14_0 <= mesh_1_3_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_15_0 <= mesh_2_3_io_out_b_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_0_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_16_0 <= io_in_d_0_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_17_0 <= mesh_0_0_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_18_0 <= mesh_1_0_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_19_0 <= mesh_2_0_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_1_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_20_0 <= io_in_d_1_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_21_0 <= mesh_0_1_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_22_0 <= mesh_1_1_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_23_0 <= mesh_2_1_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_2_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_24_0 <= io_in_d_2_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_25_0 <= mesh_0_2_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_26_0 <= mesh_1_2_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_27_0 <= mesh_2_2_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_3_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_28_0 <= io_in_d_3_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_29_0 <= mesh_0_3_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_30_0 <= mesh_1_3_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      pipe_b_31_0 <= mesh_2_3_io_out_c_0; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_0_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_0_io_in_control_0_shift_pipe_b <= io_in_control_0_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_0_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_0_io_in_control_0_dataflow_pipe_b <= io_in_control_0_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_0_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_0_io_in_control_0_propagate_pipe_b <= io_in_control_0_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_0_io_in_control_0_shift_pipe_b <= mesh_0_0_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_0_io_in_control_0_dataflow_pipe_b <= mesh_0_0_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_0_io_in_control_0_propagate_pipe_b <= mesh_0_0_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_0_io_in_control_0_shift_pipe_b <= mesh_1_0_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_0_io_in_control_0_dataflow_pipe_b <= mesh_1_0_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_0_io_in_control_0_propagate_pipe_b <= mesh_1_0_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_0_io_in_control_0_shift_pipe_b <= mesh_2_0_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_0_io_in_control_0_dataflow_pipe_b <= mesh_2_0_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_0_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_0_io_in_control_0_propagate_pipe_b <= mesh_2_0_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_1_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_1_io_in_control_0_shift_pipe_b <= io_in_control_1_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_1_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_1_io_in_control_0_dataflow_pipe_b <= io_in_control_1_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_1_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_1_io_in_control_0_propagate_pipe_b <= io_in_control_1_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_1_io_in_control_0_shift_pipe_b <= mesh_0_1_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_1_io_in_control_0_dataflow_pipe_b <= mesh_0_1_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_1_io_in_control_0_propagate_pipe_b <= mesh_0_1_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_1_io_in_control_0_shift_pipe_b <= mesh_1_1_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_1_io_in_control_0_dataflow_pipe_b <= mesh_1_1_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_1_io_in_control_0_propagate_pipe_b <= mesh_1_1_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_1_io_in_control_0_shift_pipe_b <= mesh_2_1_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_1_io_in_control_0_dataflow_pipe_b <= mesh_2_1_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_1_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_1_io_in_control_0_propagate_pipe_b <= mesh_2_1_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_2_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_2_io_in_control_0_shift_pipe_b <= io_in_control_2_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_2_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_2_io_in_control_0_dataflow_pipe_b <= io_in_control_2_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_2_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_2_io_in_control_0_propagate_pipe_b <= io_in_control_2_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_2_io_in_control_0_shift_pipe_b <= mesh_0_2_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_2_io_in_control_0_dataflow_pipe_b <= mesh_0_2_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_2_io_in_control_0_propagate_pipe_b <= mesh_0_2_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_2_io_in_control_0_shift_pipe_b <= mesh_1_2_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_2_io_in_control_0_dataflow_pipe_b <= mesh_1_2_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_2_io_in_control_0_propagate_pipe_b <= mesh_1_2_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_2_io_in_control_0_shift_pipe_b <= mesh_2_2_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_2_io_in_control_0_dataflow_pipe_b <= mesh_2_2_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_2_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_2_io_in_control_0_propagate_pipe_b <= mesh_2_2_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_3_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_3_io_in_control_0_shift_pipe_b <= io_in_control_3_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_3_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_3_io_in_control_0_dataflow_pipe_b <= io_in_control_3_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (io_in_valid_3_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_0_3_io_in_control_0_propagate_pipe_b <= io_in_control_3_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_3_io_in_control_0_shift_pipe_b <= mesh_0_3_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_3_io_in_control_0_dataflow_pipe_b <= mesh_0_3_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_0_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_1_3_io_in_control_0_propagate_pipe_b <= mesh_0_3_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_3_io_in_control_0_shift_pipe_b <= mesh_1_3_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_3_io_in_control_0_dataflow_pipe_b <= mesh_1_3_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_1_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_2_3_io_in_control_0_propagate_pipe_b <= mesh_1_3_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_3_io_in_control_0_shift_pipe_b <= mesh_2_3_io_out_control_0_shift; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_3_io_in_control_0_dataflow_pipe_b <= mesh_2_3_io_out_control_0_dataflow; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    if (mesh_2_3_io_out_valid_0) begin // @[src/main/scala/chisel3/util/Valid.scala 130:26]
      mesh_3_3_io_in_control_0_propagate_pipe_b <= mesh_2_3_io_out_control_0_propagate; // @[src/main/scala/chisel3/util/Valid.scala 130:26]
    end
    r_16_0 <= io_in_valid_0_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_17_0 <= mesh_0_0_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_18_0 <= mesh_1_0_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_19_0 <= mesh_2_0_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_20_0 <= io_in_valid_1_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_21_0 <= mesh_0_1_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_22_0 <= mesh_1_1_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_23_0 <= mesh_2_1_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_24_0 <= io_in_valid_2_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_25_0 <= mesh_0_2_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_26_0 <= mesh_1_2_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_27_0 <= mesh_2_2_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_28_0 <= io_in_valid_3_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_29_0 <= mesh_0_3_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_30_0 <= mesh_1_3_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_31_0 <= mesh_2_3_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 94:{42,42,42}]
    r_32_0 <= io_in_id_0_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_33_0 <= mesh_0_0_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_34_0 <= mesh_1_0_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_35_0 <= mesh_2_0_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_36_0 <= io_in_id_1_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_37_0 <= mesh_0_1_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_38_0 <= mesh_1_1_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_39_0 <= mesh_2_1_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_40_0 <= io_in_id_2_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_41_0 <= mesh_0_2_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_42_0 <= mesh_1_2_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_43_0 <= mesh_2_2_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_44_0 <= io_in_id_3_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_45_0 <= mesh_0_3_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_46_0 <= mesh_1_3_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_47_0 <= mesh_2_3_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 103:{39,39,39}]
    r_48_0 <= io_in_last_0_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_49_0 <= mesh_0_0_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_50_0 <= mesh_1_0_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_51_0 <= mesh_2_0_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_52_0 <= io_in_last_1_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_53_0 <= mesh_0_1_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_54_0 <= mesh_1_1_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_55_0 <= mesh_2_1_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_56_0 <= io_in_last_2_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_57_0 <= mesh_0_2_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_58_0 <= mesh_1_2_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_59_0 <= mesh_2_2_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_60_0 <= io_in_last_3_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_61_0 <= mesh_0_3_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_62_0 <= mesh_1_3_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_63_0 <= mesh_2_3_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 112:{41,41,41}]
    r_64_0 <= mesh_3_0_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 122:{23,23,23}]
    r_65_0 <= mesh_3_0_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 123:{23,23,23}]
    r_66_0 <= mesh_3_0_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 124:{23,23,23}]
    r_67_0_dataflow <= mesh_3_0_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_67_0_propagate <= mesh_3_0_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_67_0_shift <= mesh_3_0_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_68_0 <= mesh_3_0_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 126:{24,24,24}]
    r_69_0 <= mesh_3_0_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 127:{26,26,26}]
    r_70_0 <= mesh_3_1_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 122:{23,23,23}]
    r_71_0 <= mesh_3_1_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 123:{23,23,23}]
    r_72_0 <= mesh_3_1_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 124:{23,23,23}]
    r_73_0_dataflow <= mesh_3_1_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_73_0_propagate <= mesh_3_1_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_73_0_shift <= mesh_3_1_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_74_0 <= mesh_3_1_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 126:{24,24,24}]
    r_75_0 <= mesh_3_1_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 127:{26,26,26}]
    r_76_0 <= mesh_3_2_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 122:{23,23,23}]
    r_77_0 <= mesh_3_2_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 123:{23,23,23}]
    r_78_0 <= mesh_3_2_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 124:{23,23,23}]
    r_79_0_dataflow <= mesh_3_2_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_79_0_propagate <= mesh_3_2_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_79_0_shift <= mesh_3_2_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_80_0 <= mesh_3_2_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 126:{24,24,24}]
    r_81_0 <= mesh_3_2_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 127:{26,26,26}]
    r_82_0 <= mesh_3_3_io_out_b_0; // @[src/main/scala/gemmini/Mesh.scala 122:{23,23,23}]
    r_83_0 <= mesh_3_3_io_out_c_0; // @[src/main/scala/gemmini/Mesh.scala 123:{23,23,23}]
    r_84_0 <= mesh_3_3_io_out_valid_0; // @[src/main/scala/gemmini/Mesh.scala 124:{23,23,23}]
    r_85_0_dataflow <= mesh_3_3_io_out_control_0_dataflow; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_85_0_propagate <= mesh_3_3_io_out_control_0_propagate; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_85_0_shift <= mesh_3_3_io_out_control_0_shift; // @[src/main/scala/gemmini/Mesh.scala 125:{26,26,26}]
    r_86_0 <= mesh_3_3_io_out_id_0; // @[src/main/scala/gemmini/Mesh.scala 126:{24,24,24}]
    r_87_0 <= mesh_3_3_io_out_last_0; // @[src/main/scala/gemmini/Mesh.scala 127:{26,26,26}]
  end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  r_0 = _RAND_0[7:0];
  _RAND_1 = {1{`RANDOM}};
  r_1_0 = _RAND_1[7:0];
  _RAND_2 = {1{`RANDOM}};
  r_2_0 = _RAND_2[7:0];
  _RAND_3 = {1{`RANDOM}};
  r_3_0 = _RAND_3[7:0];
  _RAND_4 = {1{`RANDOM}};
  r_4_0 = _RAND_4[7:0];
  _RAND_5 = {1{`RANDOM}};
  r_5_0 = _RAND_5[7:0];
  _RAND_6 = {1{`RANDOM}};
  r_6_0 = _RAND_6[7:0];
  _RAND_7 = {1{`RANDOM}};
  r_7_0 = _RAND_7[7:0];
  _RAND_8 = {1{`RANDOM}};
  r_8_0 = _RAND_8[7:0];
  _RAND_9 = {1{`RANDOM}};
  r_9_0 = _RAND_9[7:0];
  _RAND_10 = {1{`RANDOM}};
  r_10_0 = _RAND_10[7:0];
  _RAND_11 = {1{`RANDOM}};
  r_11_0 = _RAND_11[7:0];
  _RAND_12 = {1{`RANDOM}};
  r_12_0 = _RAND_12[7:0];
  _RAND_13 = {1{`RANDOM}};
  r_13_0 = _RAND_13[7:0];
  _RAND_14 = {1{`RANDOM}};
  r_14_0 = _RAND_14[7:0];
  _RAND_15 = {1{`RANDOM}};
  r_15_0 = _RAND_15[7:0];
  _RAND_16 = {1{`RANDOM}};
  pipe_b_0 = _RAND_16[7:0];
  _RAND_17 = {1{`RANDOM}};
  pipe_b_1_0 = _RAND_17[31:0];
  _RAND_18 = {1{`RANDOM}};
  pipe_b_2_0 = _RAND_18[31:0];
  _RAND_19 = {1{`RANDOM}};
  pipe_b_3_0 = _RAND_19[31:0];
  _RAND_20 = {1{`RANDOM}};
  pipe_b_4_0 = _RAND_20[7:0];
  _RAND_21 = {1{`RANDOM}};
  pipe_b_5_0 = _RAND_21[31:0];
  _RAND_22 = {1{`RANDOM}};
  pipe_b_6_0 = _RAND_22[31:0];
  _RAND_23 = {1{`RANDOM}};
  pipe_b_7_0 = _RAND_23[31:0];
  _RAND_24 = {1{`RANDOM}};
  pipe_b_8_0 = _RAND_24[7:0];
  _RAND_25 = {1{`RANDOM}};
  pipe_b_9_0 = _RAND_25[31:0];
  _RAND_26 = {1{`RANDOM}};
  pipe_b_10_0 = _RAND_26[31:0];
  _RAND_27 = {1{`RANDOM}};
  pipe_b_11_0 = _RAND_27[31:0];
  _RAND_28 = {1{`RANDOM}};
  pipe_b_12_0 = _RAND_28[7:0];
  _RAND_29 = {1{`RANDOM}};
  pipe_b_13_0 = _RAND_29[31:0];
  _RAND_30 = {1{`RANDOM}};
  pipe_b_14_0 = _RAND_30[31:0];
  _RAND_31 = {1{`RANDOM}};
  pipe_b_15_0 = _RAND_31[31:0];
  _RAND_32 = {1{`RANDOM}};
  pipe_b_16_0 = _RAND_32[7:0];
  _RAND_33 = {1{`RANDOM}};
  pipe_b_17_0 = _RAND_33[31:0];
  _RAND_34 = {1{`RANDOM}};
  pipe_b_18_0 = _RAND_34[31:0];
  _RAND_35 = {1{`RANDOM}};
  pipe_b_19_0 = _RAND_35[31:0];
  _RAND_36 = {1{`RANDOM}};
  pipe_b_20_0 = _RAND_36[7:0];
  _RAND_37 = {1{`RANDOM}};
  pipe_b_21_0 = _RAND_37[31:0];
  _RAND_38 = {1{`RANDOM}};
  pipe_b_22_0 = _RAND_38[31:0];
  _RAND_39 = {1{`RANDOM}};
  pipe_b_23_0 = _RAND_39[31:0];
  _RAND_40 = {1{`RANDOM}};
  pipe_b_24_0 = _RAND_40[7:0];
  _RAND_41 = {1{`RANDOM}};
  pipe_b_25_0 = _RAND_41[31:0];
  _RAND_42 = {1{`RANDOM}};
  pipe_b_26_0 = _RAND_42[31:0];
  _RAND_43 = {1{`RANDOM}};
  pipe_b_27_0 = _RAND_43[31:0];
  _RAND_44 = {1{`RANDOM}};
  pipe_b_28_0 = _RAND_44[7:0];
  _RAND_45 = {1{`RANDOM}};
  pipe_b_29_0 = _RAND_45[31:0];
  _RAND_46 = {1{`RANDOM}};
  pipe_b_30_0 = _RAND_46[31:0];
  _RAND_47 = {1{`RANDOM}};
  pipe_b_31_0 = _RAND_47[31:0];
  _RAND_48 = {1{`RANDOM}};
  mesh_0_0_io_in_control_0_shift_pipe_b = _RAND_48[4:0];
  _RAND_49 = {1{`RANDOM}};
  mesh_0_0_io_in_control_0_dataflow_pipe_b = _RAND_49[0:0];
  _RAND_50 = {1{`RANDOM}};
  mesh_0_0_io_in_control_0_propagate_pipe_b = _RAND_50[0:0];
  _RAND_51 = {1{`RANDOM}};
  mesh_1_0_io_in_control_0_shift_pipe_b = _RAND_51[4:0];
  _RAND_52 = {1{`RANDOM}};
  mesh_1_0_io_in_control_0_dataflow_pipe_b = _RAND_52[0:0];
  _RAND_53 = {1{`RANDOM}};
  mesh_1_0_io_in_control_0_propagate_pipe_b = _RAND_53[0:0];
  _RAND_54 = {1{`RANDOM}};
  mesh_2_0_io_in_control_0_shift_pipe_b = _RAND_54[4:0];
  _RAND_55 = {1{`RANDOM}};
  mesh_2_0_io_in_control_0_dataflow_pipe_b = _RAND_55[0:0];
  _RAND_56 = {1{`RANDOM}};
  mesh_2_0_io_in_control_0_propagate_pipe_b = _RAND_56[0:0];
  _RAND_57 = {1{`RANDOM}};
  mesh_3_0_io_in_control_0_shift_pipe_b = _RAND_57[4:0];
  _RAND_58 = {1{`RANDOM}};
  mesh_3_0_io_in_control_0_dataflow_pipe_b = _RAND_58[0:0];
  _RAND_59 = {1{`RANDOM}};
  mesh_3_0_io_in_control_0_propagate_pipe_b = _RAND_59[0:0];
  _RAND_60 = {1{`RANDOM}};
  mesh_0_1_io_in_control_0_shift_pipe_b = _RAND_60[4:0];
  _RAND_61 = {1{`RANDOM}};
  mesh_0_1_io_in_control_0_dataflow_pipe_b = _RAND_61[0:0];
  _RAND_62 = {1{`RANDOM}};
  mesh_0_1_io_in_control_0_propagate_pipe_b = _RAND_62[0:0];
  _RAND_63 = {1{`RANDOM}};
  mesh_1_1_io_in_control_0_shift_pipe_b = _RAND_63[4:0];
  _RAND_64 = {1{`RANDOM}};
  mesh_1_1_io_in_control_0_dataflow_pipe_b = _RAND_64[0:0];
  _RAND_65 = {1{`RANDOM}};
  mesh_1_1_io_in_control_0_propagate_pipe_b = _RAND_65[0:0];
  _RAND_66 = {1{`RANDOM}};
  mesh_2_1_io_in_control_0_shift_pipe_b = _RAND_66[4:0];
  _RAND_67 = {1{`RANDOM}};
  mesh_2_1_io_in_control_0_dataflow_pipe_b = _RAND_67[0:0];
  _RAND_68 = {1{`RANDOM}};
  mesh_2_1_io_in_control_0_propagate_pipe_b = _RAND_68[0:0];
  _RAND_69 = {1{`RANDOM}};
  mesh_3_1_io_in_control_0_shift_pipe_b = _RAND_69[4:0];
  _RAND_70 = {1{`RANDOM}};
  mesh_3_1_io_in_control_0_dataflow_pipe_b = _RAND_70[0:0];
  _RAND_71 = {1{`RANDOM}};
  mesh_3_1_io_in_control_0_propagate_pipe_b = _RAND_71[0:0];
  _RAND_72 = {1{`RANDOM}};
  mesh_0_2_io_in_control_0_shift_pipe_b = _RAND_72[4:0];
  _RAND_73 = {1{`RANDOM}};
  mesh_0_2_io_in_control_0_dataflow_pipe_b = _RAND_73[0:0];
  _RAND_74 = {1{`RANDOM}};
  mesh_0_2_io_in_control_0_propagate_pipe_b = _RAND_74[0:0];
  _RAND_75 = {1{`RANDOM}};
  mesh_1_2_io_in_control_0_shift_pipe_b = _RAND_75[4:0];
  _RAND_76 = {1{`RANDOM}};
  mesh_1_2_io_in_control_0_dataflow_pipe_b = _RAND_76[0:0];
  _RAND_77 = {1{`RANDOM}};
  mesh_1_2_io_in_control_0_propagate_pipe_b = _RAND_77[0:0];
  _RAND_78 = {1{`RANDOM}};
  mesh_2_2_io_in_control_0_shift_pipe_b = _RAND_78[4:0];
  _RAND_79 = {1{`RANDOM}};
  mesh_2_2_io_in_control_0_dataflow_pipe_b = _RAND_79[0:0];
  _RAND_80 = {1{`RANDOM}};
  mesh_2_2_io_in_control_0_propagate_pipe_b = _RAND_80[0:0];
  _RAND_81 = {1{`RANDOM}};
  mesh_3_2_io_in_control_0_shift_pipe_b = _RAND_81[4:0];
  _RAND_82 = {1{`RANDOM}};
  mesh_3_2_io_in_control_0_dataflow_pipe_b = _RAND_82[0:0];
  _RAND_83 = {1{`RANDOM}};
  mesh_3_2_io_in_control_0_propagate_pipe_b = _RAND_83[0:0];
  _RAND_84 = {1{`RANDOM}};
  mesh_0_3_io_in_control_0_shift_pipe_b = _RAND_84[4:0];
  _RAND_85 = {1{`RANDOM}};
  mesh_0_3_io_in_control_0_dataflow_pipe_b = _RAND_85[0:0];
  _RAND_86 = {1{`RANDOM}};
  mesh_0_3_io_in_control_0_propagate_pipe_b = _RAND_86[0:0];
  _RAND_87 = {1{`RANDOM}};
  mesh_1_3_io_in_control_0_shift_pipe_b = _RAND_87[4:0];
  _RAND_88 = {1{`RANDOM}};
  mesh_1_3_io_in_control_0_dataflow_pipe_b = _RAND_88[0:0];
  _RAND_89 = {1{`RANDOM}};
  mesh_1_3_io_in_control_0_propagate_pipe_b = _RAND_89[0:0];
  _RAND_90 = {1{`RANDOM}};
  mesh_2_3_io_in_control_0_shift_pipe_b = _RAND_90[4:0];
  _RAND_91 = {1{`RANDOM}};
  mesh_2_3_io_in_control_0_dataflow_pipe_b = _RAND_91[0:0];
  _RAND_92 = {1{`RANDOM}};
  mesh_2_3_io_in_control_0_propagate_pipe_b = _RAND_92[0:0];
  _RAND_93 = {1{`RANDOM}};
  mesh_3_3_io_in_control_0_shift_pipe_b = _RAND_93[4:0];
  _RAND_94 = {1{`RANDOM}};
  mesh_3_3_io_in_control_0_dataflow_pipe_b = _RAND_94[0:0];
  _RAND_95 = {1{`RANDOM}};
  mesh_3_3_io_in_control_0_propagate_pipe_b = _RAND_95[0:0];
  _RAND_96 = {1{`RANDOM}};
  r_16_0 = _RAND_96[0:0];
  _RAND_97 = {1{`RANDOM}};
  r_17_0 = _RAND_97[0:0];
  _RAND_98 = {1{`RANDOM}};
  r_18_0 = _RAND_98[0:0];
  _RAND_99 = {1{`RANDOM}};
  r_19_0 = _RAND_99[0:0];
  _RAND_100 = {1{`RANDOM}};
  r_20_0 = _RAND_100[0:0];
  _RAND_101 = {1{`RANDOM}};
  r_21_0 = _RAND_101[0:0];
  _RAND_102 = {1{`RANDOM}};
  r_22_0 = _RAND_102[0:0];
  _RAND_103 = {1{`RANDOM}};
  r_23_0 = _RAND_103[0:0];
  _RAND_104 = {1{`RANDOM}};
  r_24_0 = _RAND_104[0:0];
  _RAND_105 = {1{`RANDOM}};
  r_25_0 = _RAND_105[0:0];
  _RAND_106 = {1{`RANDOM}};
  r_26_0 = _RAND_106[0:0];
  _RAND_107 = {1{`RANDOM}};
  r_27_0 = _RAND_107[0:0];
  _RAND_108 = {1{`RANDOM}};
  r_28_0 = _RAND_108[0:0];
  _RAND_109 = {1{`RANDOM}};
  r_29_0 = _RAND_109[0:0];
  _RAND_110 = {1{`RANDOM}};
  r_30_0 = _RAND_110[0:0];
  _RAND_111 = {1{`RANDOM}};
  r_31_0 = _RAND_111[0:0];
  _RAND_112 = {1{`RANDOM}};
  r_32_0 = _RAND_112[2:0];
  _RAND_113 = {1{`RANDOM}};
  r_33_0 = _RAND_113[2:0];
  _RAND_114 = {1{`RANDOM}};
  r_34_0 = _RAND_114[2:0];
  _RAND_115 = {1{`RANDOM}};
  r_35_0 = _RAND_115[2:0];
  _RAND_116 = {1{`RANDOM}};
  r_36_0 = _RAND_116[2:0];
  _RAND_117 = {1{`RANDOM}};
  r_37_0 = _RAND_117[2:0];
  _RAND_118 = {1{`RANDOM}};
  r_38_0 = _RAND_118[2:0];
  _RAND_119 = {1{`RANDOM}};
  r_39_0 = _RAND_119[2:0];
  _RAND_120 = {1{`RANDOM}};
  r_40_0 = _RAND_120[2:0];
  _RAND_121 = {1{`RANDOM}};
  r_41_0 = _RAND_121[2:0];
  _RAND_122 = {1{`RANDOM}};
  r_42_0 = _RAND_122[2:0];
  _RAND_123 = {1{`RANDOM}};
  r_43_0 = _RAND_123[2:0];
  _RAND_124 = {1{`RANDOM}};
  r_44_0 = _RAND_124[2:0];
  _RAND_125 = {1{`RANDOM}};
  r_45_0 = _RAND_125[2:0];
  _RAND_126 = {1{`RANDOM}};
  r_46_0 = _RAND_126[2:0];
  _RAND_127 = {1{`RANDOM}};
  r_47_0 = _RAND_127[2:0];
  _RAND_128 = {1{`RANDOM}};
  r_48_0 = _RAND_128[0:0];
  _RAND_129 = {1{`RANDOM}};
  r_49_0 = _RAND_129[0:0];
  _RAND_130 = {1{`RANDOM}};
  r_50_0 = _RAND_130[0:0];
  _RAND_131 = {1{`RANDOM}};
  r_51_0 = _RAND_131[0:0];
  _RAND_132 = {1{`RANDOM}};
  r_52_0 = _RAND_132[0:0];
  _RAND_133 = {1{`RANDOM}};
  r_53_0 = _RAND_133[0:0];
  _RAND_134 = {1{`RANDOM}};
  r_54_0 = _RAND_134[0:0];
  _RAND_135 = {1{`RANDOM}};
  r_55_0 = _RAND_135[0:0];
  _RAND_136 = {1{`RANDOM}};
  r_56_0 = _RAND_136[0:0];
  _RAND_137 = {1{`RANDOM}};
  r_57_0 = _RAND_137[0:0];
  _RAND_138 = {1{`RANDOM}};
  r_58_0 = _RAND_138[0:0];
  _RAND_139 = {1{`RANDOM}};
  r_59_0 = _RAND_139[0:0];
  _RAND_140 = {1{`RANDOM}};
  r_60_0 = _RAND_140[0:0];
  _RAND_141 = {1{`RANDOM}};
  r_61_0 = _RAND_141[0:0];
  _RAND_142 = {1{`RANDOM}};
  r_62_0 = _RAND_142[0:0];
  _RAND_143 = {1{`RANDOM}};
  r_63_0 = _RAND_143[0:0];
  _RAND_144 = {1{`RANDOM}};
  r_64_0 = _RAND_144[31:0];
  _RAND_145 = {1{`RANDOM}};
  r_65_0 = _RAND_145[31:0];
  _RAND_146 = {1{`RANDOM}};
  r_66_0 = _RAND_146[0:0];
  _RAND_147 = {1{`RANDOM}};
  r_67_0_dataflow = _RAND_147[0:0];
  _RAND_148 = {1{`RANDOM}};
  r_67_0_propagate = _RAND_148[0:0];
  _RAND_149 = {1{`RANDOM}};
  r_67_0_shift = _RAND_149[4:0];
  _RAND_150 = {1{`RANDOM}};
  r_68_0 = _RAND_150[2:0];
  _RAND_151 = {1{`RANDOM}};
  r_69_0 = _RAND_151[0:0];
  _RAND_152 = {1{`RANDOM}};
  r_70_0 = _RAND_152[31:0];
  _RAND_153 = {1{`RANDOM}};
  r_71_0 = _RAND_153[31:0];
  _RAND_154 = {1{`RANDOM}};
  r_72_0 = _RAND_154[0:0];
  _RAND_155 = {1{`RANDOM}};
  r_73_0_dataflow = _RAND_155[0:0];
  _RAND_156 = {1{`RANDOM}};
  r_73_0_propagate = _RAND_156[0:0];
  _RAND_157 = {1{`RANDOM}};
  r_73_0_shift = _RAND_157[4:0];
  _RAND_158 = {1{`RANDOM}};
  r_74_0 = _RAND_158[2:0];
  _RAND_159 = {1{`RANDOM}};
  r_75_0 = _RAND_159[0:0];
  _RAND_160 = {1{`RANDOM}};
  r_76_0 = _RAND_160[31:0];
  _RAND_161 = {1{`RANDOM}};
  r_77_0 = _RAND_161[31:0];
  _RAND_162 = {1{`RANDOM}};
  r_78_0 = _RAND_162[0:0];
  _RAND_163 = {1{`RANDOM}};
  r_79_0_dataflow = _RAND_163[0:0];
  _RAND_164 = {1{`RANDOM}};
  r_79_0_propagate = _RAND_164[0:0];
  _RAND_165 = {1{`RANDOM}};
  r_79_0_shift = _RAND_165[4:0];
  _RAND_166 = {1{`RANDOM}};
  r_80_0 = _RAND_166[2:0];
  _RAND_167 = {1{`RANDOM}};
  r_81_0 = _RAND_167[0:0];
  _RAND_168 = {1{`RANDOM}};
  r_82_0 = _RAND_168[31:0];
  _RAND_169 = {1{`RANDOM}};
  r_83_0 = _RAND_169[31:0];
  _RAND_170 = {1{`RANDOM}};
  r_84_0 = _RAND_170[0:0];
  _RAND_171 = {1{`RANDOM}};
  r_85_0_dataflow = _RAND_171[0:0];
  _RAND_172 = {1{`RANDOM}};
  r_85_0_propagate = _RAND_172[0:0];
  _RAND_173 = {1{`RANDOM}};
  r_85_0_shift = _RAND_173[4:0];
  _RAND_174 = {1{`RANDOM}};
  r_86_0 = _RAND_174[2:0];
  _RAND_175 = {1{`RANDOM}};
  r_87_0 = _RAND_175[0:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule

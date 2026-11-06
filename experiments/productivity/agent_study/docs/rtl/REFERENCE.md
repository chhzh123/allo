# SystemVerilog reference

You are writing synthesisable register-transfer-level SystemVerilog for an AMD
device, compiled by Vivado's simulator and synthesiser. Everything the design
does is something you state explicitly: every register, every wire, and the
handshake on every port.

## The subset that synthesises

Sequential logic goes in `always_ff @(posedge ap_clk)` with non-blocking
assignments. Combinational logic goes in `always_comb` with blocking
assignments, or in continuous `assign` statements. Do not mix the two styles in
one block.

    logic signed [31:0] acc;
    always_ff @(posedge ap_clk) begin
      if (!ap_rst_n) acc <= '0;
      else if (en)   acc <= acc + prod;
    end

`logic` is the general type; declare vectors as `logic [W-1:0]` and use
`logic signed` where the arithmetic is signed. Signed multiplication needs both
operands signed: `$signed(a) * $signed(b)`.

## Replication

`generate` with `genvar` instantiates a block once per index, which is how you
build an array without writing out its elements.

    genvar i, j;
    generate
      for (i = 0; i < 8; i = i + 1) begin : row
        for (j = 0; j < 8; j = j + 1) begin : col
          pe u_pe (.ap_clk(ap_clk), .a_in(h[i][j]), .a_out(h[i][j+1]), ...);
        end
      end
    endgenerate

Wires that connect neighbours are naturally declared as arrays one larger than
the grid in the direction of travel, so element `(i,j)` reads index `j` and
drives index `j+1`.

Unpacked arrays of wires are declared `wire [7:0] h [0:8][0:7];` and may be
connected element by element in a port map.

## The handshake on your ports

Every port on this design uses a two-signal handshake. For an input port `p`
the environment drives `p_dout` and `p_empty_n`, and you drive `p_read`; a
value transfers on a rising edge when `p_empty_n` and `p_read` are both high,
and you must consume `p_dout` on that edge because it may change afterwards.
For an output port `p` you drive `p_din` and `p_write`, the environment drives
`p_full_n`, and a value transfers when `p_write` and `p_full_n` are both high.

Do not make `p_read` depend combinationally on `p_empty_n` in a way that forms
a loop through the environment; drive it from registered state.

## Reset and timing

`ap_rst_n` is active low and is held for at least eight cycles before the first
transfer. Registers you rely on must be reset there.

The design must run at a 3.333 ns period. Long combinational chains are what
break this: a multiplication feeding an addition feeding another addition in
one cycle will usually not close timing, while a multiplication whose result is
registered before it is added will. Keep one arithmetic operation between
registers where you can.

Multiplications map to digital signal processing blocks when they are simple
and their operands are registered; a multiplication buried in a wide
combinational expression may be built from logic instead, which is slower and
larger.

## State machines

An explicit state register with a `case` in `always_ff` is the usual idiom, and
Vivado infers it cleanly. Keep the next-state logic and the datapath in
separate blocks where you can, so a timing failure points at one of them.

    typedef enum logic [1:0] {IDLE, RUN, DRAIN} state_t;
    state_t state;
    always_ff @(posedge ap_clk)
      if (!ap_rst_n) state <= IDLE;
      else case (state)
        IDLE:  if (start) state <= RUN;
        RUN:   if (last)  state <= DRAIN;
        DRAIN: if (done)  state <= IDLE;
      endcase

## Buffering for full throughput

A stage that holds one value can accept a new one only every other cycle when
the consumer is slow, because it must empty before it fills. A two-deep skid
buffer accepts one value per cycle regardless: keep a main register and a spare,
fill the spare when the output is blocked, and drain the spare first.

    always_ff @(posedge ap_clk) begin
      if (!ap_rst_n) begin n_main <= 0; n_skid <= 0; end
      else begin
        if (in_valid && in_ready) begin
          if (!n_main || out_ready) begin main <= in_data;  n_main <= 1; end
          else                      begin skid <= in_data;  n_skid <= 1; end
        end else if (n_main && out_ready) begin
          if (n_skid) begin main <= skid; n_skid <= 0; end else n_main <= 0;
        end
      end
    end
    assign in_ready = !n_skid;

Whether you need this depends on how often the environment stalls you. The test
harness never stalls an output and always has an input ready, so a simpler
one-deep stage is enough to reach one transfer per cycle there, but it will not
survive a consumer that pauses.

## Parameters

`module m #(parameter int W = 8) (...)` and instantiate with `m #(.W(16)) u
(...)`. Parameters can size ports and arrays and control `generate` conditions,
so one module covers positions whose wiring differs:

    generate
      if (J == 0) begin : first
        // this position has no western neighbour
      end else begin : rest
      end
    endgenerate

## Things the synthesiser will refuse or silently change

Multiple `always` blocks driving one signal is an error. A `for` loop inside
`always_ff` is unrolled, not sequenced, so it costs area rather than cycles.
An unassigned branch in `always_comb` infers a latch, which will not meet
timing; assign a default at the top of the block. Reading an unpacked array
element with a non-constant index in one cycle builds a multiplexer whose depth
grows with the array, which is a common cause of a failed timing path.

## Worked example: a four-stage scaling chain

Not related to your task. It shows replication by `generate`, wires between
neighbours, and the handshake driven from registered state.

    module stage #(parameter int K = 2) (
      input  logic ap_clk, input logic ap_rst_n,
      input  logic [31:0] x_in_dout,  input  logic x_in_empty_n,
      output logic        x_in_read,
      output logic [31:0] x_out_din,  input  logic x_out_full_n,
      output logic        x_out_write);

      logic [31:0] held;
      logic        full;
      assign x_in_read   = !full;              // take one when we have room
      assign x_out_din   = held;
      assign x_out_write = full;

      always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
          full <= 1'b0;
        end else begin
          if (!full && x_in_empty_n) begin
            held <= x_in_dout * K;             // registered before it leaves
            full <= 1'b1;
          end else if (full && x_out_full_n) begin
            full <= 1'b0;
          end
        end
      end
    endmodule

    module line_engine (
      input logic ap_clk, input logic ap_rst_n,
      input  logic [31:0] X_dout, input logic X_empty_n, output logic X_read,
      output logic [31:0] Y_din,  input logic Y_full_n,   output logic Y_write);

      localparam int NS = 4;
      logic [31:0] d [0:NS];  logic v [0:NS];  logic r [0:NS];
      assign d[0] = X_dout;  assign v[0] = X_empty_n;  assign X_read = r[0];
      assign Y_din = d[NS];  assign Y_write = v[NS];   assign r[NS] = Y_full_n;

      genvar s;
      generate
        for (s = 0; s < NS; s = s + 1) begin : chain
          stage #(.K(s + 2)) u (
            .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
            .x_in_dout(d[s]),    .x_in_empty_n(v[s]),   .x_in_read(r[s]),
            .x_out_din(d[s+1]),  .x_out_full_n(r[s+1]), .x_out_write(v[s+1]));
        end
      endgenerate
    endmodule

Each stage holds one value, so a token moves one stage per cycle once the chain
is full, and back-pressure propagates backwards one stage per cycle.

## Building

    build

compiles and elaborates your design with the test harness and simulates it
against the visible vectors. It prints the compiler's errors if it does not
elaborate, then the number of wrong values and the cycle counts.

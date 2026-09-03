// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written Vitis HLS baseline: two-level hierarchical tiled GEMM.
//
// A TILES x TILES grid of tile engines, each an inner PE x PE output-stationary
// mesh, over int8 into int32.
//
// An AXI bundle may be read by exactly one dataflow process, so the operands
// cannot be handed to the engines as pointers: one loader per operand reads
// DRAM and fans the data out to the engines that need it, and one storer
// collects their results. That plumbing -- three extra processes and a stream
// per engine edge -- is what the two-level structure costs in HLS, and it is
// what the SPMW version replaces with `spmw.shard`.

#include <ap_int.h>
#include <hls_stream.h>

#define TILES 2 // engines per side
#define PE 8    // PEs per side inside one engine
#define SIDE (TILES * PE)

typedef ap_int<8> data_t;
typedef ap_int<32> acc_t;

// One PE of an inner mesh.
static void pe(hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
               hls::stream<data_t> &b_in, hls::stream<data_t> &b_out,
               hls::stream<acc_t> &c_out, int steps) {
  acc_t acc = 0;
pe_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
    data_t a = a_in.read();
    data_t b = b_in.read();
    acc += (acc_t)a * (acc_t)b;
    a_out.write(a);
    b_out.write(b);
  }
  c_out.write(acc);
}

// One loader for A: reads gmem0 once and fans each row out to the TILES engines
// in that tile-row.
static void load_a(const data_t *A, hls::stream<data_t> out[TILES][TILES][PE],
                   int steps) {
load_a_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  load_a_ti:
    for (int ti = 0; ti < TILES; ti++) {
#pragma HLS UNROLL
    load_a_row:
      for (int r = 0; r < PE; r++) {
#pragma HLS UNROLL
        data_t value = A[(ti * PE + r) * steps + k];
      load_a_fan:
        for (int tj = 0; tj < TILES; tj++) {
#pragma HLS UNROLL
          out[ti][tj][r].write(value);
        }
      }
    }
  }
}

// One loader for B, fanning each column out down its tile-column.
static void load_b(const data_t *B, hls::stream<data_t> out[TILES][TILES][PE],
                   int steps) {
load_b_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  load_b_tj:
    for (int tj = 0; tj < TILES; tj++) {
#pragma HLS UNROLL
    load_b_col:
      for (int c = 0; c < PE; c++) {
#pragma HLS UNROLL
        data_t value = B[k * SIDE + tj * PE + c];
      load_b_fan:
        for (int ti = 0; ti < TILES; ti++) {
#pragma HLS UNROLL
          out[ti][tj][c].write(value);
        }
      }
    }
  }
}

static void sink(hls::stream<data_t> in[PE], int steps) {
sink_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  sink_lane:
    for (int i = 0; i < PE; i++) {
#pragma HLS UNROLL
      in[i].read();
    }
  }
}

// One storer for C: the only process that touches gmem2.
static void store_c(hls::stream<acc_t> in[TILES][TILES][PE][PE], acc_t *C) {
store_ti:
  for (int ti = 0; ti < TILES; ti++) {
#pragma HLS UNROLL
  store_tj:
    for (int tj = 0; tj < TILES; tj++) {
#pragma HLS UNROLL
    store_row:
      for (int r = 0; r < PE; r++) {
#pragma HLS UNROLL
      store_col:
        for (int c = 0; c < PE; c++) {
#pragma HLS UNROLL
          C[(ti * PE + r) * SIDE + tj * PE + c] = in[ti][tj][r][c].read();
        }
      }
    }
  }
}

// One tile engine: the inner mesh on streams it does not own.
static void tile_engine(hls::stream<data_t> a_west[PE],
                        hls::stream<data_t> b_north[PE],
                        hls::stream<acc_t> c_out[PE][PE], int steps) {
#pragma HLS DATAFLOW
  hls::stream<data_t> a[PE + 1][PE];
#pragma HLS ARRAY_PARTITION variable = a complete dim = 0
  hls::stream<data_t> b[PE + 1][PE];
#pragma HLS ARRAY_PARTITION variable = b complete dim = 0

engine_west:
  for (int r = 0; r < PE; r++) {
#pragma HLS UNROLL
    // The engine's west edge is the loader's stream, forwarded in.
  engine_west_steps:
    for (int k = 0; k < steps; k++)
      a[0][r].write(a_west[r].read());
  }
engine_north:
  for (int c = 0; c < PE; c++) {
#pragma HLS UNROLL
  engine_north_steps:
    for (int k = 0; k < steps; k++)
      b[0][c].write(b_north[c].read());
  }

engine_rows:
  for (int r = 0; r < PE; r++) {
#pragma HLS UNROLL
  engine_cols:
    for (int col = 0; col < PE; col++) {
#pragma HLS UNROLL
      pe(a[col][r], a[col + 1][r], b[r][col], b[r + 1][col], c_out[r][col],
         steps);
    }
  }

  sink(a[PE], steps);
  sink(b[PE], steps);
}

extern "C" {

void gemm_tiled(const data_t *A, const data_t *B, acc_t *C, int steps) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = steps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

  hls::stream<data_t> fa[TILES][TILES][PE];
#pragma HLS ARRAY_PARTITION variable = fa complete dim = 0
  hls::stream<data_t> fb[TILES][TILES][PE];
#pragma HLS ARRAY_PARTITION variable = fb complete dim = 0
  hls::stream<acc_t> fc[TILES][TILES][PE][PE];
#pragma HLS ARRAY_PARTITION variable = fc complete dim = 0

  load_a(A, fa, steps);
  load_b(B, fb, steps);

grid_rows:
  for (int i = 0; i < TILES; i++) {
#pragma HLS UNROLL
  grid_cols:
    for (int j = 0; j < TILES; j++) {
#pragma HLS UNROLL
      tile_engine(fa[i][j], fb[i][j], fc[i][j], steps);
    }
  }

  store_c(fc, C);
}

} // extern "C"

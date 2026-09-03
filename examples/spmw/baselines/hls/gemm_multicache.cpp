// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written Vitis HLS baseline: daisy-chain multi-cache GEMM.
//
// The same DIMxDIM output-stationary mesh as the systolic baseline, but the
// results leave down a chain rather than through one port per PE: each PE
// writes its own row's slot into a column vector passing through it, so the
// bottom of each column emits a whole column of C. int16 throughout, matching
// the SPMW design this is compared against.

#include <ap_int.h>
#include <hls_stream.h>

#define DIM 16

typedef ap_int<16> data_t;

// The drain chain carries a whole column per token, so it is a struct rather
// than a scalar stream.
struct column_t {
  data_t v[DIM];
};

// One PE: multiply-accumulate over K, forwarding both operands, then splice its
// accumulator into the column passing through.
static void pe(hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
               hls::stream<data_t> &b_in, hls::stream<data_t> &b_out,
               hls::stream<column_t> &c_in, hls::stream<column_t> &c_out,
               int row, int steps) {
  data_t acc = 0;
pe_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
    data_t a = a_in.read();
    data_t b = b_in.read();
    acc += a * b;
    a_out.write(a);
    b_out.write(b);
  }
  column_t col = c_in.read();
  col.v[row] = acc;
  c_out.write(col);
}

static void feed_west(const data_t *A, hls::stream<data_t> out[DIM], int steps) {
feed_west_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  feed_west_lane:
    for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
      out[r].write(A[r * steps + k]);
    }
  }
}

static void feed_north(const data_t *B, hls::stream<data_t> out[DIM],
                       int steps) {
feed_north_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  feed_north_lane:
    for (int c = 0; c < DIM; c++) {
#pragma HLS UNROLL
      out[c].write(B[k * DIM + c]);
    }
  }
}

static void sink(hls::stream<data_t> in[DIM], int steps) {
sink_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  sink_lane:
    for (int i = 0; i < DIM; i++) {
#pragma HLS UNROLL
      in[i].read();
    }
  }
}

// The chain starts empty at the top row.
static void seed_chain(hls::stream<column_t> out[DIM]) {
seed_chain_lane:
  for (int c = 0; c < DIM; c++) {
#pragma HLS UNROLL
    column_t zero;
  seed_chain_elem:
    for (int i = 0; i < DIM; i++)
#pragma HLS UNROLL
      zero.v[i] = 0;
    out[c].write(zero);
  }
}

static void drain_chain(hls::stream<column_t> in[DIM], data_t *C) {
drain_chain_lane:
  for (int c = 0; c < DIM; c++) {
#pragma HLS UNROLL
    column_t col = in[c].read();
  drain_chain_elem:
    for (int r = 0; r < DIM; r++)
#pragma HLS UNROLL
      C[r * DIM + c] = col.v[r];
  }
}

extern "C" {

void gemm_multicache(const data_t *A, const data_t *B, data_t *C, int steps) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = steps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

  hls::stream<data_t> a[DIM + 1][DIM];
#pragma HLS ARRAY_PARTITION variable = a complete dim = 0
  hls::stream<data_t> b[DIM + 1][DIM];
#pragma HLS ARRAY_PARTITION variable = b complete dim = 0
  hls::stream<column_t> c[DIM + 1][DIM];
#pragma HLS ARRAY_PARTITION variable = c complete dim = 0

  feed_west(A, a[0], steps);
  feed_north(B, b[0], steps);
  seed_chain(c[0]);

array_rows:
  for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
  array_cols:
    for (int col = 0; col < DIM; col++) {
#pragma HLS UNROLL
      pe(a[col][r], a[col + 1][r], b[r][col], b[r + 1][col], c[r][col],
         c[r + 1][col], r, steps);
    }
  }

  sink(a[DIM], steps);
  sink(b[DIM], steps);
  drain_chain(c[DIM], C);
}

} // extern "C"

// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written Vitis HLS baseline: the plain systolic GEMM.
//
// A DIMxDIM output-stationary mesh. Operands enter from the west and north
// edges and walk east and south; each PE accumulates over K and writes its own
// element of C. int8 operands into an int32 accumulator, matching the SPMW
// design this is compared against.

#include <ap_int.h>
#include <hls_stream.h>

#define DIM 16

typedef ap_int<8> data_t;
typedef ap_int<32> acc_t;

// One PE: multiply-accumulate over K, forwarding both operands onward.
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

// Every PE holds one element of C, so the drain is one read per site.
static void drain(hls::stream<acc_t> c[DIM][DIM], acc_t *C) {
drain_rows:
  for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
  drain_cols:
    for (int col = 0; col < DIM; col++) {
#pragma HLS UNROLL
      C[r * DIM + col] = c[r][col].read();
    }
  }
}

extern "C" {

void gemm_output_stationary(const data_t *A, const data_t *B, acc_t *C,
                            int steps) {
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
  hls::stream<acc_t> c[DIM][DIM];
#pragma HLS ARRAY_PARTITION variable = c complete dim = 0

  feed_west(A, a[0], steps);
  feed_north(B, b[0], steps);

array_rows:
  for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
  array_cols:
    for (int col = 0; col < DIM; col++) {
#pragma HLS UNROLL
      pe(a[col][r], a[col + 1][r], b[r][col], b[r + 1][col], c[r][col], steps);
    }
  }

  sink(a[DIM], steps);
  sink(b[DIM], steps);
  drain(c, C);
}

} // extern "C"

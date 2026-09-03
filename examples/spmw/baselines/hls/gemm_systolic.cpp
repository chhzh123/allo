// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written Vitis HLS baseline: 16x16 weight-stationary int8 systolic GEMM.
//
// This is the comparison point for Fig. 10 and Table 5, so it is written the
// way an engineer would write it directly in HLS -- one PE function, an array
// of streams, loaders and a drain -- rather than as a strawman. It computes the
// same thing as the SPMW mini-TPU MXU: int8 operands into an int32
// accumulator, then ReLU and an arithmetic shift back down to int8.
//
// The array is fixed at 16x16 and large GEMMs are tiled over it by the host,
// which is what makes one bitstream cover 64^3 through 1024^3.

#include <ap_int.h>
#include <hls_stream.h>

#define DIM 16
#define SHIFT 6

typedef ap_int<8> data_t;
typedef ap_int<32> acc_t;
typedef ap_uint<128> bus_t; // 16 int8 lanes per beat

// One processing element. The weight is stationary; activations walk east and
// partial sums walk south. `steps` is the tile's K extent.
static void pe(hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
               hls::stream<acc_t> &p_in, hls::stream<acc_t> &p_out, data_t w,
               int steps) {
pe_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
    data_t a = a_in.read();
    acc_t p = p_in.read();
    p_out.write(p + (acc_t)a * (acc_t)w);
    a_out.write(a);
  }
}

// Feed one row of A into the west edge of the array.
static void feed_a(const bus_t *A, hls::stream<data_t> a_out[DIM], int steps,
                   int tile) {
feed_a_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
    bus_t beat = A[tile * steps + k];
  feed_a_lane:
    for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
      a_out[r].write((data_t)beat.range(8 * r + 7, 8 * r));
    }
  }
}

// Zero the north edge: partial sums start at 0 and accumulate downward.
static void seed_p(hls::stream<acc_t> p_out[DIM], int steps) {
seed_p_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  seed_p_lane:
    for (int c = 0; c < DIM; c++) {
#pragma HLS UNROLL
      p_out[c].write(0);
    }
  }
}

// Sink the activations that fall off the east edge.
static void sink_a(hls::stream<data_t> a_in[DIM], int steps) {
sink_a_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  sink_a_lane:
    for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
      a_in[r].read();
    }
  }
}

// The 16-lane vector unit: ReLU then shift, packing int8 results out.
static void drain(hls::stream<acc_t> p_in[DIM], bus_t *Y, int steps, int tile) {
drain_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
    bus_t beat = 0;
  drain_lane:
    for (int c = 0; c < DIM; c++) {
#pragma HLS UNROLL
      acc_t z = p_in[c].read();
      acc_t r = z > 0 ? z : (acc_t)0;
      beat.range(8 * c + 7, 8 * c) = (data_t)(r >> SHIFT);
    }
    Y[tile * steps + k] = beat;
  }
}

// One tile: load the stationary weights, then stream the tile through.
//
// `a` is indexed [column][row] so that the east edge is a[DIM], one contiguous
// array of DIM streams; `p` is [row][column] so the south edge is p[DIM].
static void tile_pass(const bus_t *A, const data_t w[DIM][DIM], bus_t *Y,
                      int steps, int tile) {
#pragma HLS DATAFLOW
  hls::stream<data_t> a[DIM + 1][DIM];
#pragma HLS ARRAY_PARTITION variable = a complete dim = 0
#pragma HLS STREAM variable = a depth = 4
  hls::stream<acc_t> p[DIM + 1][DIM];
#pragma HLS ARRAY_PARTITION variable = p complete dim = 0
#pragma HLS STREAM variable = p depth = 4

  feed_a(A, a[0], steps, tile);
  seed_p(p[0], steps);

array_rows:
  for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
  array_cols:
    for (int c = 0; c < DIM; c++) {
#pragma HLS UNROLL
      pe(a[c][r], a[c + 1][r], p[r][c], p[r + 1][c], w[r][c], steps);
    }
  }

  sink_a(a[DIM], steps);
  drain(p[DIM], Y, steps, tile);
}

extern "C" {

// A: activations, packed 16 int8 per 128-bit beat, `steps` beats per tile.
// W: stationary weights, DIM*DIM int8 per tile.
// Y: results, same packing as A.
void gemm_systolic(const bus_t *A, const data_t *W, bus_t *Y, int steps,
                   int tiles) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = W offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = Y offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = W bundle = control
#pragma HLS INTERFACE s_axilite port = Y bundle = control
#pragma HLS INTERFACE s_axilite port = steps bundle = control
#pragma HLS INTERFACE s_axilite port = tiles bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  static data_t w[DIM][DIM];
#pragma HLS ARRAY_PARTITION variable = w complete dim = 0

tiles_loop:
  for (int t = 0; t < tiles; t++) {
  load_w:
    for (int i = 0; i < DIM * DIM; i++) {
#pragma HLS PIPELINE II = 1
      w[i / DIM][i % DIM] = W[t * DIM * DIM + i];
    }
    tile_pass(A, w, Y, steps, t);
  }
}

} // extern "C"

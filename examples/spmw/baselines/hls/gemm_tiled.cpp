// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written Vitis HLS baseline: two-level hierarchical tiled GEMM.
//
// A TILES x TILES grid of tile engines, each an inner PE x PE output-stationary
// mesh, over float32. Tile (i, j) owns the C block at (i, j) and is fed the i-th
// row-slab of A and the j-th column-slab of B.
//
// The hierarchy has to be written twice here -- once for the PE mesh and once
// for the grid of engines -- because an HLS dataflow region composes function
// calls, not placeable components. That duplication is what the SPMW version
// avoids by placing a fabric on a topology.

#include <hls_stream.h>

#define TILES 2 // engines per side
#define PE 8    // PEs per side inside one engine
#define KDEPTH 16

typedef float data_t;

// One PE of an inner mesh.
static void pe(hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
               hls::stream<data_t> &b_in, hls::stream<data_t> &b_out,
               hls::stream<data_t> &c_out, int steps) {
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
  c_out.write(acc);
}

static void tile_feed_west(const data_t *A, hls::stream<data_t> out[PE],
                           int steps) {
tile_feed_west_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  tile_feed_west_lane:
    for (int r = 0; r < PE; r++) {
#pragma HLS UNROLL
      out[r].write(A[r * steps + k]);
    }
  }
}

static void tile_feed_north(const data_t *B, hls::stream<data_t> out[PE],
                            int steps) {
tile_feed_north_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  tile_feed_north_lane:
    for (int c = 0; c < PE; c++) {
#pragma HLS UNROLL
      out[c].write(B[k * PE + c]);
    }
  }
}

static void tile_sink(hls::stream<data_t> in[PE], int steps) {
tile_sink_loop:
  for (int k = 0; k < steps; k++) {
#pragma HLS PIPELINE II = 1
  tile_sink_lane:
    for (int i = 0; i < PE; i++) {
#pragma HLS UNROLL
      in[i].read();
    }
  }
}

static void tile_drain(hls::stream<data_t> c[PE][PE], data_t *C, int stride) {
tile_drain_rows:
  for (int r = 0; r < PE; r++) {
#pragma HLS UNROLL
  tile_drain_cols:
    for (int col = 0; col < PE; col++) {
#pragma HLS UNROLL
      C[r * stride + col] = c[r][col].read();
    }
  }
}

// One tile engine: the inner mesh, complete with its own edges.
static void tile_engine(const data_t *A, const data_t *B, data_t *C, int steps,
                        int stride) {
#pragma HLS DATAFLOW
  hls::stream<data_t> a[PE + 1][PE];
#pragma HLS ARRAY_PARTITION variable = a complete dim = 0
  hls::stream<data_t> b[PE + 1][PE];
#pragma HLS ARRAY_PARTITION variable = b complete dim = 0
  hls::stream<data_t> c[PE][PE];
#pragma HLS ARRAY_PARTITION variable = c complete dim = 0

  tile_feed_west(A, a[0], steps);
  tile_feed_north(B, b[0], steps);

engine_rows:
  for (int r = 0; r < PE; r++) {
#pragma HLS UNROLL
  engine_cols:
    for (int col = 0; col < PE; col++) {
#pragma HLS UNROLL
      pe(a[col][r], a[col + 1][r], b[r][col], b[r + 1][col], c[r][col], steps);
    }
  }

  tile_sink(a[PE], steps);
  tile_sink(b[PE], steps);
  tile_drain(c, C, stride);
}

extern "C" {

void gemm_tiled(const data_t *A, const data_t *B, data_t *C, int steps) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = steps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

  // The outer level: one engine per tile, each on its own slab of A and B.
grid_rows:
  for (int i = 0; i < TILES; i++) {
#pragma HLS UNROLL
  grid_cols:
    for (int j = 0; j < TILES; j++) {
#pragma HLS UNROLL
      tile_engine(A + i * PE * steps, B + j * PE, C + (i * PE) * (TILES * PE) + j * PE,
                  steps, TILES * PE);
    }
  }
}

} // extern "C"

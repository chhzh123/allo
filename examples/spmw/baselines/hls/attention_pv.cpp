// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written Vitis HLS baseline: grouped attention-PV.
//
// The P.V GEMM on a DIMxDIM weight-stationary array cut into GROUPS column
// slabs. Activations move east inside a slab and stop at its edge; partial sums
// move south down a column and then cross into the top of the next slab, so the
// psum chain is the reduction network.
//
// GROUPS decides five separate things in this file: which PEs forward
// activations, where each bottom-row PE sends its partial sum, how many west
// columns the loader feeds, how many north inputs are seeded, and how wide the
// drain is. They are written out separately because none of them can be
// inferred from the others -- which is the comparison Table 3 is making, since
// in the SPMW description the same change is one argument.

#include <ap_int.h>
#include <hls_stream.h>

#define DIM 16
#ifndef GROUPS
#define GROUPS 1
#endif
#define SLAB (DIM / GROUPS) // output columns per slab == head dimension
#define SHIFT 2

typedef ap_int<8> data_t;
typedef ap_int<32> acc_t;

// Interior of a slab: the activation is forwarded east.
static void pe_forward(hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
                       hls::stream<acc_t> &p_in, hls::stream<acc_t> &p_out,
                       data_t w, int steps) {
pe_forward_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
    data_t a = a_in.read();
    acc_t p = p_in.read();
    p_out.write(p + (acc_t)a * (acc_t)w);
    a_out.write(a);
  }
}

// Slab edge: the activation stops here.
static void pe_edge(hls::stream<data_t> &a_in, hls::stream<acc_t> &p_in,
                    hls::stream<acc_t> &p_out, data_t w, int steps) {
pe_edge_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
    data_t a = a_in.read();
    acc_t p = p_in.read();
    p_out.write(p + (acc_t)a * (acc_t)w);
  }
}

// Every slab has its own west column, so there are GROUPS * DIM of them.
static void feed_a(const data_t *Pr, hls::stream<data_t> a[DIM][DIM],
                   int steps) {
feed_a_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
  feed_a_slab:
    for (int g = 0; g < GROUPS; g++) {
#pragma HLS UNROLL
    feed_a_row:
      for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
        a[g * SLAB][r].write(Pr[m * GROUPS * DIM + g * DIM + r]);
      }
    }
  }
}

// Only slab 0's top row is seeded; every other slab's top receives the previous
// slab's partial sum.
static void seed_p(hls::stream<acc_t> p[DIM][DIM], int steps) {
seed_p_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
  seed_p_lane:
    for (int c = 0; c < SLAB; c++) {
#pragma HLS UNROLL
      p[0][c].write(0);
    }
  }
}

// Only the last slab's bottom row drains, through a SLAB-wide activation unit.
static void drain(hls::stream<acc_t> d[SLAB], data_t *Y, int steps) {
drain_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
  drain_lane:
    for (int c = 0; c < SLAB; c++) {
#pragma HLS UNROLL
      acc_t z = d[c].read();
      acc_t r = z > 0 ? z : (acc_t)0;
      Y[m * SLAB + c] = (data_t)(r >> SHIFT);
    }
  }
}

extern "C" {

void attention_pv(const data_t *Pr, const data_t *V, data_t *Y, int steps) {
#pragma HLS INTERFACE m_axi port = Pr offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = V offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = Y offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = Pr bundle = control
#pragma HLS INTERFACE s_axilite port = V bundle = control
#pragma HLS INTERFACE s_axilite port = Y bundle = control
#pragma HLS INTERFACE s_axilite port = steps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  static data_t w[DIM][DIM];
#pragma HLS ARRAY_PARTITION variable = w complete dim = 0
load_w:
  for (int i = 0; i < DIM * DIM; i++) {
#pragma HLS PIPELINE II = 1
    // PE (k, c) in slab g holds V[g * DIM + k][c % SLAB].
    int k = i / DIM, c = i % DIM;
    w[k][c] = V[((c / SLAB) * DIM + k) * SLAB + (c % SLAB)];
  }

  {
#pragma HLS DATAFLOW
    // a[c][r] enters column c of row r; a slab's west column is fed, the rest
    // come from the PE to the west.
    hls::stream<data_t> a[DIM][DIM];
#pragma HLS ARRAY_PARTITION variable = a complete dim = 0
    // p[r][c] enters row r of column c; row 0 of slab 0 is seeded, row 0 of
    // every other slab is the previous slab's bottom.
    hls::stream<acc_t> p[DIM][DIM];
#pragma HLS ARRAY_PARTITION variable = p complete dim = 0
    hls::stream<acc_t> d[SLAB];
#pragma HLS ARRAY_PARTITION variable = d complete dim = 0

    feed_a(Pr, a, steps);
    seed_p(p, steps);

  array_rows:
    for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
    array_cols:
      for (int c = 0; c < DIM; c++) {
#pragma HLS UNROLL
        // Down a column, across to the next slab's top, or out to the drain.
        hls::stream<acc_t> &pdst =
            (r + 1 < DIM)
                ? p[r + 1][c]
                : ((c + SLAB < DIM) ? p[0][c + SLAB] : d[c - (DIM - SLAB)]);
        if (((c + 1) % SLAB) == 0)
          pe_edge(a[c][r], p[r][c], pdst, w[r][c], steps);
        else
          pe_forward(a[c][r], a[c + 1][r], p[r][c], pdst, w[r][c], steps);
      }
    }

    drain(d, Y, steps);
  }
}

} // extern "C"

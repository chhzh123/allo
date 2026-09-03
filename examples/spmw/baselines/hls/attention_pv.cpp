// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written Vitis HLS baseline: grouped attention-PV.
//
// The P.V GEMM on a DIMxDIM weight-stationary array cut into GROUPS column
// slabs. Activations move east inside a slab only; partial sums move south
// down a column and then serpentine into the top of the next slab, so the psum
// chain is the reduction network.
//
// GROUPS is a compile-time constant here because the wiring depends on it: the
// activation forwarding condition, the psum destination, the number of seeded
// north inputs, the number of drained south outputs, and the width of the
// activation unit all change with it. That is the point of the comparison in
// Table 3 -- in the SPMW description the same change is one argument.

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

// One PE. The weight is stationary; activations walk east within a slab.
static void pe(hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
               hls::stream<acc_t> &p_in, hls::stream<acc_t> &p_out, data_t w,
               int steps, bool forward_a) {
pe_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
    data_t a = a_in.read();
    acc_t p = p_in.read();
    p_out.write(p + (acc_t)a * (acc_t)w);
    if (forward_a)
      a_out.write(a);
  }
}

// The west edge of every slab is fed, so there are GROUPS * DIM feeds, not DIM.
static void feed_a(const data_t *Pr, hls::stream<data_t> a[GROUPS * DIM],
                   int steps) {
feed_a_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
  feed_a_lane:
    for (int i = 0; i < GROUPS * DIM; i++) {
#pragma HLS UNROLL
      a[i].write(Pr[m * GROUPS * DIM + i]);
    }
  }
}

// Only slab 0's top row is seeded; the others receive the previous slab's sum.
static void seed_p(hls::stream<acc_t> p[SLAB], int steps) {
seed_p_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
  seed_p_lane:
    for (int c = 0; c < SLAB; c++) {
#pragma HLS UNROLL
      p[c].write(0);
    }
  }
}

// Only the last slab's bottom row drains, so the activation unit is SLAB wide.
static void drain(hls::stream<acc_t> p[SLAB], data_t *Y, int steps) {
drain_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
  drain_lane:
    for (int c = 0; c < SLAB; c++) {
#pragma HLS UNROLL
      acc_t z = p[c].read();
      acc_t r = z > 0 ? z : (acc_t)0;
      Y[m * SLAB + c] = (data_t)(r >> SHIFT);
    }
  }
}

static void sink_a(hls::stream<data_t> &in, int steps) {
sink_a_loop:
  for (int m = 0; m < steps; m++) {
#pragma HLS PIPELINE II = 1
    in.read();
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
    hls::stream<data_t> a[DIM + 1][DIM];
#pragma HLS ARRAY_PARTITION variable = a complete dim = 0
    hls::stream<acc_t> p[DIM + 1][DIM];
#pragma HLS ARRAY_PARTITION variable = p complete dim = 0

    feed_a(Pr, a[0], steps);
    seed_p(p[0], steps);

  array_rows:
    for (int r = 0; r < DIM; r++) {
#pragma HLS UNROLL
    array_cols:
      for (int c = 0; c < DIM; c++) {
#pragma HLS UNROLL
        // Activations stop at a slab edge; partial sums cross to the next slab.
        bool last_in_slab = ((c + 1) % SLAB) == 0;
        pe(a[c][r], a[c + 1][r], p[r][c], p[r + 1][c], w[r][c], steps,
           !last_in_slab);
      }
    }

    drain(p[DIM], Y, steps);
  }
}

} // extern "C"

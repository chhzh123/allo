// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Fixture for scripts/spmw_loc2.py: C++ symbol selection.
#include <hls_stream.h> // KEEP
#define DIM 16 // KEEP
typedef int data_t; // KEEP

struct column_t { // KEEP
  data_t v[DIM]; // KEEP
}; // KEEP

// One PE.
static void pe(hls::stream<data_t> &a_in, // KEEP
               hls::stream<data_t> &a_out) { // KEEP
  a_out.write(a_in.read()); // KEEP
} // KEEP

extern "C" { // KEEP

void top(const data_t *A, data_t *C) { // KEEP
#pragma HLS INTERFACE m_axi port = A // KEEP
  hls::stream<data_t> s[DIM]; // KEEP
  pe(s[0], s[1]); // KEEP
} // KEEP

} // extern "C" KEEP

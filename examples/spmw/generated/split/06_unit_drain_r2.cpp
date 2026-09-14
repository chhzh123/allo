
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void drain_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t v5 = v3.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v5;	// L5
  int32_t v7 = v4.read();	// L6
  int32_t _st__pid1;	// L7
  _st__pid1 = v7;	// L8
  int32_t v9 = _st__pid0;	// L9
  int32_t row;	// L10
  row = v9;	// L11
  int32_t v11 = _st__pid1;	// L12
  int32_t _col;	// L13
  _col = v11;	// L14
  int32_t v13 = v1.read();	// L15
  v0.write(v13);	// L16
  int32_t v14 = row;	// L17
  int v15 = v14;	// L21
  for (int v16 = 0; v16 < v15; v16 += 1) {	// L25
    int32_t v17 = v2.read();	// L26
    v0.write(v17);	// L27
  }
}


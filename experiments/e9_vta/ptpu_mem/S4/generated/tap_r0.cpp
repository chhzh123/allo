
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void tap_r0_0(
  hls::stream< int64_t >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int64_t >& v2
) {	// L2
  bool go;	// L6
  go = 1;	// L7
  while (true) {	// L8
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v4 = go;	// L9
    if (!(v4)) break;
    int64_t v5 = v1.read();	// L16
    int64_t u;	// L17
    u = v5;	// L18
    int64_t v7 = u;	// L19
    v2.write(v7);	// L20
    int64_t v8 = u;	// L21
    v0.write(v8);	// L22
    int64_t v9 = u;	// L23
    int64_t v10 = v9 >> 10;	// L27
    int64_t v11 = v10 & 1;	// L31
    bool v12 = v11 != 0;	// L35
    if (v12) {	// L36
      go = 0;	// L40
    }
  }
}


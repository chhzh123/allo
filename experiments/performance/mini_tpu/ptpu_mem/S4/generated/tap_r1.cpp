
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void tap_r1_0(
  hls::stream< int64_t >& v0,
  hls::stream< int64_t >& v1
) {	// L2
  bool go;	// L6
  go = 1;	// L7
  while (true) {	// L8
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v3 = go;	// L9
    if (!(v3)) break;
    int64_t v4 = v1.read();	// L16
    int64_t u;	// L17
    u = v4;	// L18
    int64_t v6 = u;	// L19
    v0.write(v6);	// L20
    int64_t v7 = u;	// L21
    int64_t v8 = v7 >> 10;	// L25
    int64_t v9 = v8 & 1;	// L29
    bool v10 = v9 != 0;	// L33
    if (v10) {	// L34
      go = 0;	// L38
    }
  }
}


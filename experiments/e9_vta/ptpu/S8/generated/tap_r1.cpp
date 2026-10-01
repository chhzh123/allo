
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
  hls::stream< int16_t >& v0,
  hls::stream< int16_t >& v1
) {	// L2
  bool go;	// L6
  go = 1;	// L7
  while (true) {	// L8
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v3 = go;	// L9
    if (!(v3)) break;
    int16_t v4 = v1.read();	// L16
    int16_t u;	// L17
    u = v4;	// L18
    int16_t v6 = u;	// L19
    v0.write(v6);	// L20
    int16_t v7 = u;	// L21
    int16_t v8 = v7 >> 11;	// L25
    int32_t v9 = v8;	// L26
    int32_t v10 = v9 & 1;	// L29
    bool v11 = v10 != 0;	// L32
    if (v11) {	// L33
      go = 0;	// L37
    }
  }
}


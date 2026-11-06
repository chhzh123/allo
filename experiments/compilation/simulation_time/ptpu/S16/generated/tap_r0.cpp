
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
  hls::stream< int16_t >& v0,
  hls::stream< int16_t >& v1,
  hls::stream< int16_t >& v2
) {	// L2
  bool go;	// L6
  go = 1;	// L7
  while (true) {	// L8
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v4 = go;	// L9
    if (!(v4)) break;
    int16_t v5 = v1.read();	// L16
    int16_t u;	// L17
    u = v5;	// L18
    int16_t v7 = u;	// L19
    v2.write(v7);	// L20
    int16_t v8 = u;	// L21
    v0.write(v8);	// L22
    int16_t v9 = u;	// L23
    int16_t v10 = v9 >> 11;	// L27
    int32_t v11 = v10;	// L28
    int32_t v12 = v11 & 1;	// L31
    bool v13 = v12 != 0;	// L34
    if (v13) {	// L35
      go = 0;	// L39
    }
  }
}


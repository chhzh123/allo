
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void lane_k1n1_r0_0(
  hls::stream< hls::vector< int32_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t v3[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v3[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v4 = v3[1];	// L4
  int32_t v5 = v4 & 31;	// L6
  int32_t sh;	// L7
  sh = v5;	// L8
  l_S_s_0_s: for (int s = 0; s < 256; s++) {	// L9
  #pragma HLS pipeline II=1
    int32_t kb;	// L19
    kb = 0;	// L20
    int32_t nb;	// L30
    nb = 0;	// L31
    int32_t v10 = v2.read();	// L32
    int32_t z;	// L33
    z = v10;	// L34
    int32_t v12 = nb;	// L35
    int v13 = v12;	// L36
    int32_t v14 = v3[v13];	// L37
    int32_t base;	// L38
    base = v14;	// L39
    int32_t v16 = base;	// L40
    int32_t v17 = z;	// L41
    ap_int<33> v18 = v16;	// L42
    ap_int<33> v19 = v17;	// L43
    ap_int<33> v20 = v18 + v19;	// L44
    int32_t v21 = v20;	// L45
    int32_t v;	// L46
    v = v21;	// L47
    int32_t v23 = kb;	// L48
    ap_int<33> v24 = v23;	// L49
    bool v25 = v24 == 0;	// L50
    if (v25) {	// L51
      int32_t v26 = v;	// L52
      bool v27 = v26 < 0;	// L53
      if (v27) {	// L54
        v = 0;	// L55
      }
      int32_t v28 = v;	// L57
      int32_t v29 = sh;	// L58
      int32_t v30 = v28 >> v29;	// L59
      v = v30;	// L60
      int32_t v31 = v;	// L61
      bool v32 = v31 > 127;	// L63
      if (v32) {	// L64
        v = 127;	// L65
      }
      int32_t v33 = v;	// L67
      v1.write(v33);	// L68
    }
  }
}


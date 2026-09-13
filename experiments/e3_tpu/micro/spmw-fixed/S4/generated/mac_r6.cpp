
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void mac_r6_0(
  hls::stream< int32_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t wf[4];	// L14
  for (int v4 = 0; v4 < 4; v4++) {	// L15
    wf[v4] = 0;	// L15
  }
  int32_t v5 = v0.read();	// L16
  int32_t n;	// L17
  n = v5;	// L18
  l_S_i_0_i: for (int i = 0; i < 4; i++) {	// L19
    int32_t v8 = v0.read();	// L20
    wf[i] = v8;	// L21
  }
  int32_t v9 = n;	// L23
  ap_int<33> v10 = v9;	// L24
  ap_int<33> v11 = v10 - 4;	// L25
  int v12 = v11;	// L26
  for (int v13 = 0; v13 < v12; v13 += 1) {	// L27
    int32_t v14 = v0.read();	// L28
    int32_t fwd;	// L29
    fwd = v14;	// L30
  }
  l_S_r_2_r: for (int r = 0; r < 64; r++) {	// L32
    int8_t v17 = v1.read();	// L33
    int8_t a;	// L34
    a = v17;	// L35
    int32_t p;	// L36
    p = 0;	// L37
    int v20 = r >> 2;	// L38
    int32_t v21 = v20;	// L39
    int32_t idx;	// L40
    idx = v21;	// L41
    int32_t v23 = idx;	// L42
    int32_t v24 = v23 >> 2;	// L43
    int v25 = v24;	// L44
    int32_t v26 = wf[v25];	// L45
    int32_t packed;	// L46
    packed = v26;	// L47
    int32_t v28 = packed;	// L48
    int32_t v29 = idx;	// L49
    int32_t v30 = v29 & 3;	// L50
    int64_t v31 = v30;	// L51
    int64_t v32 = v31 * 8;	// L52
    int32_t v33 = v32;	// L53
    int32_t v34 = v28 >> v33;	// L54
    int32_t v35 = v34 & 255;	// L55
    int32_t byte;	// L56
    byte = v35;	// L57
    int32_t v37 = byte;	// L58
    int32_t v38 = v37 ^ 128;	// L59
    ap_int<33> v39 = v38;	// L60
    ap_int<33> v40 = v39 - 128;	// L61
    int32_t v41 = v40;	// L62
    int32_t wt;	// L63
    wt = v41;	// L64
    int32_t v43 = p;	// L65
    int8_t v44 = a;	// L66
    int32_t v45 = wt;	// L67
    ap_int<40> v46 = v44;	// L68
    ap_int<40> v47 = v45;	// L69
    ap_int<40> v48 = v46 * v47;	// L70
    ap_int<41> v49 = v43;	// L71
    ap_int<41> v50 = v48;	// L72
    ap_int<41> v51 = v49 + v50;	// L73
    v2.write(v51);	// L74
  }
}

/// This is top function.
void top(

) {	// L78
  #pragma HLS dataflow
  hls::stream< int8_t > v52;
  #pragma HLS stream variable=v52 depth=4	// L79
  hls::stream< int32_t > v53;
  #pragma HLS stream variable=v53 depth=2	// L80
  hls::stream< int32_t > v54;
  #pragma HLS stream variable=v54 depth=2	// L81
  mac_r6_0(v54, v52, v53);	// L82
}



//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void mac_r3_0(
  hls::stream< int32_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int32_t wf[4];	// L14
  for (int v5 = 0; v5 < 4; v5++) {	// L15
    wf[v5] = 0;	// L15
  }
  int32_t v6 = v0.read();	// L16
  int32_t n;	// L17
  n = v6;	// L18
  l_S_i_0_i: for (int i = 0; i < 4; i++) {	// L19
    int32_t v9 = v0.read();	// L20
    wf[i] = v9;	// L21
  }
  int32_t v10 = n;	// L23
  ap_int<33> v11 = v10;	// L24
  ap_int<33> v12 = v11 - 4;	// L25
  int v13 = v12;	// L26
  for (int v14 = 0; v14 < v13; v14 += 1) {	// L27
    int32_t v15 = v0.read();	// L28
    int32_t fwd;	// L29
    fwd = v15;	// L30
  }
  l_S_r_2_r: for (int r = 0; r < 128; r++) {	// L32
    int8_t v18 = v1.read();	// L33
    int8_t a;	// L34
    a = v18;	// L35
    int32_t v20 = v2.read();	// L36
    int32_t p;	// L37
    p = v20;	// L38
    int v22 = r >> 3;	// L39
    int32_t v23 = v22;	// L40
    int32_t idx;	// L41
    idx = v23;	// L42
    int32_t v25 = idx;	// L43
    int32_t v26 = v25 >> 2;	// L44
    int v27 = v26;	// L45
    int32_t v28 = wf[v27];	// L46
    int32_t packed;	// L47
    packed = v28;	// L48
    int32_t v30 = packed;	// L49
    int32_t v31 = idx;	// L50
    int32_t v32 = v31 & 3;	// L51
    int64_t v33 = v32;	// L52
    int64_t v34 = v33 * 8;	// L53
    int32_t v35 = v34;	// L54
    int32_t v36 = v30 >> v35;	// L55
    int32_t v37 = v36 & 255;	// L56
    int32_t byte;	// L57
    byte = v37;	// L58
    int32_t v39 = byte;	// L59
    int32_t v40 = v39 ^ 128;	// L60
    ap_int<33> v41 = v40;	// L61
    ap_int<33> v42 = v41 - 128;	// L62
    int32_t v43 = v42;	// L63
    int32_t wt;	// L64
    wt = v43;	// L65
    int32_t v45 = p;	// L66
    int8_t v46 = a;	// L67
    int32_t v47 = wt;	// L68
    ap_int<40> v48 = v46;	// L69
    ap_int<40> v49 = v47;	// L70
    ap_int<40> v50 = v48 * v49;	// L71
    ap_int<41> v51 = v45;	// L72
    ap_int<41> v52 = v50;	// L73
    ap_int<41> v53 = v51 + v52;	// L74
    v3.write(v53);	// L75
  }
}

/// This is top function.
void top(

) {	// L79
  #pragma HLS dataflow
  hls::stream< int8_t > v54;
  #pragma HLS stream variable=v54 depth=4	// L80
  hls::stream< int32_t > v55;
  #pragma HLS stream variable=v55 depth=4	// L81
  hls::stream< int32_t > v56;
  #pragma HLS stream variable=v56 depth=2	// L82
  hls::stream< int32_t > v57;
  #pragma HLS stream variable=v57 depth=2	// L83
  mac_r3_0(v57, v54, v55, v56);	// L84
}



//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void mac_r4_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int32_t >& v5
) {	// L2
  int32_t wf[4];	// L14
  for (int v7 = 0; v7 < 4; v7++) {	// L15
    wf[v7] = 0;	// L15
  }
  int32_t v8 = v0.read();	// L16
  int32_t n;	// L17
  n = v8;	// L18
  int32_t v10 = n;	// L19
  ap_int<33> v11 = v10;	// L20
  ap_int<33> v12 = v11 - 4;	// L21
  v1.write(v12);	// L22
  l_S_i_0_i: for (int i = 0; i < 4; i++) {	// L23
    int32_t v14 = v0.read();	// L24
    wf[i] = v14;	// L25
  }
  int32_t v15 = n;	// L27
  ap_int<33> v16 = v15;	// L28
  ap_int<33> v17 = v16 - 4;	// L29
  int v18 = v17;	// L30
  for (int v19 = 0; v19 < v18; v19 += 1) {	// L31
    int32_t v20 = v0.read();	// L32
    int32_t fwd;	// L33
    fwd = v20;	// L34
    int32_t v22 = fwd;	// L35
    v1.write(v22);	// L36
  }
  l_S_r_2_r: for (int r = 0; r < 128; r++) {	// L38
    int8_t v24 = v2.read();	// L39
    int8_t a;	// L40
    a = v24;	// L41
    int32_t v26 = v3.read();	// L42
    int32_t p;	// L43
    p = v26;	// L44
    int8_t v28 = a;	// L45
    v4.write(v28);	// L46
    int v29 = r >> 3;	// L47
    int32_t v30 = v29;	// L48
    int32_t idx;	// L49
    idx = v30;	// L50
    int32_t v32 = idx;	// L51
    int32_t v33 = v32 >> 2;	// L52
    int v34 = v33;	// L53
    int32_t v35 = wf[v34];	// L54
    int32_t packed;	// L55
    packed = v35;	// L56
    int32_t v37 = packed;	// L57
    int32_t v38 = idx;	// L58
    int32_t v39 = v38 & 3;	// L59
    int64_t v40 = v39;	// L60
    int64_t v41 = v40 * 8;	// L61
    int32_t v42 = v41;	// L62
    int32_t v43 = v37 >> v42;	// L63
    int32_t v44 = v43 & 255;	// L64
    int32_t byte;	// L65
    byte = v44;	// L66
    int32_t v46 = byte;	// L67
    int32_t v47 = v46 ^ 128;	// L68
    ap_int<33> v48 = v47;	// L69
    ap_int<33> v49 = v48 - 128;	// L70
    int32_t v50 = v49;	// L71
    int32_t wt;	// L72
    wt = v50;	// L73
    int32_t v52 = p;	// L74
    int8_t v53 = a;	// L75
    int32_t v54 = wt;	// L76
    ap_int<40> v55 = v53;	// L77
    ap_int<40> v56 = v54;	// L78
    ap_int<40> v57 = v55 * v56;	// L79
    ap_int<41> v58 = v52;	// L80
    ap_int<41> v59 = v57;	// L81
    ap_int<41> v60 = v58 + v59;	// L82
    v5.write(v60);	// L83
  }
}

/// This is top function.
void top(

) {	// L87
  #pragma HLS dataflow
  hls::stream< int8_t > v61;
  #pragma HLS stream variable=v61 depth=4	// L88
  hls::stream< int8_t > v62;
  #pragma HLS stream variable=v62 depth=2	// L89
  hls::stream< int32_t > v63;
  #pragma HLS stream variable=v63 depth=4	// L90
  hls::stream< int32_t > v64;
  #pragma HLS stream variable=v64 depth=2	// L91
  hls::stream< int32_t > v65;
  #pragma HLS stream variable=v65 depth=2	// L92
  hls::stream< int32_t > v66;
  #pragma HLS stream variable=v66 depth=2	// L93
  mac_r4_0(v65, v66, v61, v63, v62, v64);	// L94
}


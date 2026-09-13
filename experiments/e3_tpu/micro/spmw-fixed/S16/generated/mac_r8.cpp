
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void mac_r8_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t wf[4];	// L14
  for (int v6 = 0; v6 < 4; v6++) {	// L15
    wf[v6] = 0;	// L15
  }
  int32_t v7 = v0.read();	// L16
  int32_t n;	// L17
  n = v7;	// L18
  int32_t v9 = n;	// L19
  ap_int<33> v10 = v9;	// L20
  ap_int<33> v11 = v10 - 4;	// L21
  v1.write(v11);	// L22
  l_S_i_0_i: for (int i = 0; i < 4; i++) {	// L23
    int32_t v13 = v0.read();	// L24
    wf[i] = v13;	// L25
  }
  int32_t v14 = n;	// L27
  ap_int<33> v15 = v14;	// L28
  ap_int<33> v16 = v15 - 4;	// L29
  int v17 = v16;	// L30
  for (int v18 = 0; v18 < v17; v18 += 1) {	// L31
    int32_t v19 = v0.read();	// L32
    int32_t fwd;	// L33
    fwd = v19;	// L34
    int32_t v21 = fwd;	// L35
    v1.write(v21);	// L36
  }
  l_S_r_2_r: for (int r = 0; r < 256; r++) {	// L38
    int8_t v23 = v2.read();	// L39
    int8_t a;	// L40
    a = v23;	// L41
    int32_t p;	// L42
    p = 0;	// L43
    int8_t v26 = a;	// L44
    v3.write(v26);	// L45
    int v27 = r >> 4;	// L46
    int32_t v28 = v27;	// L47
    int32_t idx;	// L48
    idx = v28;	// L49
    int32_t v30 = idx;	// L50
    int32_t v31 = v30 >> 2;	// L51
    int v32 = v31;	// L52
    int32_t v33 = wf[v32];	// L53
    int32_t packed;	// L54
    packed = v33;	// L55
    int32_t v35 = packed;	// L56
    int32_t v36 = idx;	// L57
    int32_t v37 = v36 & 3;	// L58
    int64_t v38 = v37;	// L59
    int64_t v39 = v38 * 8;	// L60
    int32_t v40 = v39;	// L61
    int32_t v41 = v35 >> v40;	// L62
    int32_t v42 = v41 & 255;	// L63
    int32_t byte;	// L64
    byte = v42;	// L65
    int32_t v44 = byte;	// L66
    int32_t v45 = v44 ^ 128;	// L67
    ap_int<33> v46 = v45;	// L68
    ap_int<33> v47 = v46 - 128;	// L69
    int32_t v48 = v47;	// L70
    int32_t wt;	// L71
    wt = v48;	// L72
    int32_t v50 = p;	// L73
    int8_t v51 = a;	// L74
    int32_t v52 = wt;	// L75
    ap_int<40> v53 = v51;	// L76
    ap_int<40> v54 = v52;	// L77
    ap_int<40> v55 = v53 * v54;	// L78
    ap_int<41> v56 = v50;	// L79
    ap_int<41> v57 = v55;	// L80
    ap_int<41> v58 = v56 + v57;	// L81
    v4.write(v58);	// L82
  }
}

/// This is top function.
void top(

) {	// L86
  #pragma HLS dataflow
  hls::stream< int8_t > v59;
  #pragma HLS stream variable=v59 depth=4	// L87
  hls::stream< int8_t > v60;
  #pragma HLS stream variable=v60 depth=2	// L88
  hls::stream< int32_t > v61;
  #pragma HLS stream variable=v61 depth=2	// L89
  hls::stream< int32_t > v62;
  #pragma HLS stream variable=v62 depth=2	// L90
  hls::stream< int32_t > v63;
  #pragma HLS stream variable=v63 depth=2	// L91
  mac_r8_0(v62, v63, v59, v60, v61);	// L92
}



//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void blk2_r5_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int8_t >& v6,
  hls::stream< int8_t >& v7
) {	// L2
  int8_t n0_0;	// L5
  n0_0 = 0;	// L6
  int8_t n0_1;	// L7
  n0_1 = 0;	// L8
  int8_t n1_0;	// L9
  n1_0 = 0;	// L10
  int8_t n1_1;	// L11
  n1_1 = 0;	// L12
  l_S__k_0__k: for (int _k = 0; _k < 16; _k++) {	// L13
  #pragma HLS pipeline II=1
    int8_t v13 = n0_0;	// L14
    n0_1 = v13;	// L15
    int8_t v14 = v6.read();	// L16
    n0_0 = v14;	// L17
    int8_t v15 = n1_0;	// L18
    n1_1 = v15;	// L19
    int8_t v16 = v7.read();	// L20
    n1_0 = v16;	// L21
  }
  int8_t v17 = n0_0;	// L23
  int8_t c0_0;	// L24
  c0_0 = v17;	// L25
  int8_t v19 = n0_1;	// L26
  int8_t c0_1;	// L27
  c0_1 = v19;	// L28
  int8_t v21 = n1_0;	// L29
  int8_t c1_0;	// L30
  c1_0 = v21;	// L31
  int8_t v23 = n1_1;	// L32
  int8_t c1_1;	// L33
  c1_1 = v23;	// L34
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L35
  #pragma HLS pipeline II=1
    int8_t v26 = v0.read();	// L36
    int8_t a0;	// L37
    a0 = v26;	// L38
    int8_t v28 = v1.read();	// L39
    int8_t a1;	// L40
    a1 = v28;	// L41
    int32_t v30 = v2.read();	// L42
    int32_t p0;	// L43
    p0 = v30;	// L44
    int32_t v32 = v4.read();	// L45
    int32_t p1;	// L46
    p1 = v32;	// L47
    int32_t v34 = p0;	// L48
    int8_t v35 = a0;	// L49
    int8_t v36 = c0_0;	// L50
    int16_t v37 = v35;	// L51
    int16_t v38 = v36;	// L52
    int16_t v39 = v37 * v38;	// L53
    #pragma HLS bind_op variable=v39 op=mul impl=fabric
    int8_t v40 = a1;	// L54
    int8_t v41 = c1_0;	// L55
    int16_t v42 = v40;	// L56
    int16_t v43 = v41;	// L57
    int16_t v44 = v42 * v43;	// L58
    #pragma HLS bind_op variable=v44 op=mul impl=fabric
    ap_int<17> v45 = v39;	// L59
    ap_int<17> v46 = v44;	// L60
    ap_int<17> v47 = v45 + v46;	// L61
    ap_int<33> v48 = v34;	// L62
    ap_int<33> v49 = v47;	// L63
    ap_int<33> v50 = v48 + v49;	// L64
    v3.write(v50);	// L65
    int32_t v51 = p1;	// L66
    int8_t v52 = a0;	// L67
    int8_t v53 = c0_1;	// L68
    int16_t v54 = v52;	// L69
    int16_t v55 = v53;	// L70
    int16_t v56 = v54 * v55;	// L71
    #pragma HLS bind_op variable=v56 op=mul impl=fabric
    int8_t v57 = a1;	// L72
    int8_t v58 = c1_1;	// L73
    int16_t v59 = v57;	// L74
    int16_t v60 = v58;	// L75
    int16_t v61 = v59 * v60;	// L76
    #pragma HLS bind_op variable=v61 op=mul impl=fabric
    ap_int<17> v62 = v56;	// L77
    ap_int<17> v63 = v61;	// L78
    ap_int<17> v64 = v62 + v63;	// L79
    ap_int<33> v65 = v51;	// L80
    ap_int<33> v66 = v64;	// L81
    ap_int<33> v67 = v65 + v66;	// L82
    v5.write(v67);	// L83
    int8_t v68 = n0_0;	// L84
    n0_1 = v68;	// L85
    int8_t v69 = v6.read();	// L86
    n0_0 = v69;	// L87
    int8_t v70 = n1_0;	// L88
    n1_1 = v70;	// L89
    int8_t v71 = v7.read();	// L90
    n1_0 = v71;	// L91
    int32_t v72 = s;	// L92
    int32_t v73 = v72 & 15;	// L94
    bool v74 = v73 == 15;	// L95
    if (v74) {	// L96
      int8_t v75 = n0_0;	// L97
      c0_0 = v75;	// L98
      int8_t v76 = n0_1;	// L99
      c0_1 = v76;	// L100
      int8_t v77 = n1_0;	// L101
      c1_0 = v77;	// L102
      int8_t v78 = n1_1;	// L103
      c1_1 = v78;	// L104
    }
  }
}


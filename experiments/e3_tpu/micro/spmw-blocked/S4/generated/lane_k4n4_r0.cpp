
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
void lane_k4n4_r0_0(
  hls::stream< hls::vector< int32_t, 5 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t v3[5];
  {
    hls::vector< int32_t, 5 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 5; ++_iv0) {
      v3[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v4 = v3[4];	// L4
  int32_t v5 = v4 & 31;	// L6
  int32_t sh;	// L7
  sh = v5;	// L8
  int32_t r0;	// L10
  r0 = 0;	// L11
  int32_t r1;	// L12
  r1 = 0;	// L13
  int32_t r2;	// L14
  r2 = 0;	// L15
  int32_t r3;	// L16
  r3 = 0;	// L17
  int32_t r4;	// L18
  r4 = 0;	// L19
  int32_t r5;	// L20
  r5 = 0;	// L21
  int32_t r6;	// L22
  r6 = 0;	// L23
  int32_t r7;	// L24
  r7 = 0;	// L25
  int32_t r8;	// L26
  r8 = 0;	// L27
  int32_t r9;	// L28
  r9 = 0;	// L29
  int32_t r10;	// L30
  r10 = 0;	// L31
  int32_t r11;	// L32
  r11 = 0;	// L33
  int32_t r12;	// L34
  r12 = 0;	// L35
  int32_t r13;	// L36
  r13 = 0;	// L37
  int32_t r14;	// L38
  r14 = 0;	// L39
  int32_t r15;	// L40
  r15 = 0;	// L41
  l_S_s_0_s: for (int s = 0; s < 4096; s++) {	// L42
  #pragma HLS pipeline II=1
    int v24 = s >> 4;	// L45
    ap_int<33> v25 = v24;	// L50
    ap_int<33> v26 = v25 & 3;	// L51
    int32_t v27 = v26;	// L52
    int32_t kb;	// L53
    kb = v27;	// L54
    int v29 = s >> 6;	// L59
    ap_int<33> v30 = v29;	// L60
    ap_int<33> v31 = v30 & 3;	// L61
    int32_t v32 = v31;	// L62
    int32_t nb;	// L63
    nb = v32;	// L64
    int32_t v34 = v2.read();	// L65
    int32_t z;	// L66
    z = v34;	// L67
    int32_t v36 = nb;	// L68
    int v37 = v36;	// L69
    int32_t v38 = v3[v37];	// L70
    int32_t base;	// L71
    base = v38;	// L72
    int32_t v40 = kb;	// L73
    bool v41 = v40 != 0;	// L74
    if (v41) {	// L75
      int32_t v42 = r0;	// L76
      base = v42;	// L77
    }
    int32_t v43 = base;	// L79
    int32_t v44 = z;	// L80
    ap_int<33> v45 = v43;	// L81
    ap_int<33> v46 = v44;	// L82
    ap_int<33> v47 = v45 + v46;	// L83
    int32_t v48 = v47;	// L84
    int32_t v;	// L85
    v = v48;	// L86
    int32_t v50 = r1;	// L87
    r0 = v50;	// L88
    int32_t v51 = r2;	// L89
    r1 = v51;	// L90
    int32_t v52 = r3;	// L91
    r2 = v52;	// L92
    int32_t v53 = r4;	// L93
    r3 = v53;	// L94
    int32_t v54 = r5;	// L95
    r4 = v54;	// L96
    int32_t v55 = r6;	// L97
    r5 = v55;	// L98
    int32_t v56 = r7;	// L99
    r6 = v56;	// L100
    int32_t v57 = r8;	// L101
    r7 = v57;	// L102
    int32_t v58 = r9;	// L103
    r8 = v58;	// L104
    int32_t v59 = r10;	// L105
    r9 = v59;	// L106
    int32_t v60 = r11;	// L107
    r10 = v60;	// L108
    int32_t v61 = r12;	// L109
    r11 = v61;	// L110
    int32_t v62 = r13;	// L111
    r12 = v62;	// L112
    int32_t v63 = r14;	// L113
    r13 = v63;	// L114
    int32_t v64 = r15;	// L115
    r14 = v64;	// L116
    int32_t v65 = v;	// L117
    r15 = v65;	// L118
    int32_t v66 = kb;	// L119
    ap_int<33> v67 = v66;	// L120
    bool v68 = v67 == 3;	// L121
    if (v68) {	// L122
      int32_t v69 = v;	// L123
      bool v70 = v69 < 0;	// L124
      if (v70) {	// L125
        v = 0;	// L126
      }
      int32_t v71 = v;	// L128
      int32_t v72 = sh;	// L129
      int32_t v73 = v71 >> v72;	// L130
      v = v73;	// L131
      int32_t v74 = v;	// L132
      bool v75 = v74 > 127;	// L134
      if (v75) {	// L135
        v = 127;	// L136
      }
      int32_t v76 = v;	// L138
      v1.write(v76);	// L139
    }
  }
}


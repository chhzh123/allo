
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
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int8_t >& v6,
  hls::stream< int32_t >& v7
) {	// L2
  int32_t wf[64];	// L19
  for (int v9 = 0; v9 < 64; v9++) {	// L20
    wf[v9] = 0;	// L20
  }
  int32_t v10 = v0.read();	// L21
  int32_t count;	// L22
  count = v10;	// L23
  int32_t v12 = count;	// L24
  v1.write(v12);	// L25
  int32_t v13 = count;	// L26
  int v14 = v13;	// L27
  for (int v15 = 0; v15 < v14; v15 += 1) {	// L28
    int32_t v16 = v0.read();	// L29
    int32_t word;	// L30
    word = v16;	// L31
    int32_t v18 = word;	// L32
    int32_t v19 = v18 >> 24;	// L33
    int32_t v20 = v19 & 255;	// L34
    int32_t opcode;	// L35
    opcode = v20;	// L36
    int32_t v22 = word;	// L37
    int32_t v23 = v22 >> 16;	// L38
    int32_t v24 = v23 & 255;	// L39
    int32_t base;	// L40
    base = v24;	// L41
    int32_t v26 = word;	// L42
    int32_t v27 = v26 & 65535;	// L43
    int32_t n;	// L44
    n = v27;	// L45
    int32_t v29 = opcode;	// L46
    bool v30 = v29 == 6;	// L47
    if (v30) {	// L48
      int32_t v31 = n;	// L49
      ap_int<33> v32 = v31;	// L50
      ap_int<33> v33 = v32 - 64;	// L51
      ap_int<33> v34 = v33 | 100663296;	// L52
      v1.write(v34);	// L53
      l_S_i_0_i: for (int i = 0; i < 64; i++) {	// L54
        int32_t v36 = v2.read();	// L55
        wf[i] = v36;	// L56
      }
      int32_t v37 = n;	// L58
      ap_int<33> v38 = v37;	// L59
      ap_int<33> v39 = v38 - 64;	// L60
      int v40 = v39;	// L61
      for (int v41 = 0; v41 < v40; v41 += 1) {	// L62
        int32_t v42 = v2.read();	// L63
        int32_t x;	// L64
        x = v42;	// L65
        int32_t v44 = x;	// L66
        v3.write(v44);	// L67
      }
    } else {
      int32_t v45 = word;	// L70
      v1.write(v45);	// L71
      int32_t v46 = n;	// L72
      int v47 = v46;	// L73
      for (int v48 = 0; v48 < v47; v48 += 1) {	// L74
        int8_t v49 = v4.read();	// L75
        int8_t a;	// L76
        a = v49;	// L77
        int32_t v51 = v5.read();	// L78
        int32_t p;	// L79
        p = v51;	// L80
        int8_t v53 = a;	// L81
        v6.write(v53);	// L82
        int32_t v54 = opcode;	// L83
        bool v55 = v54 == 4;	// L84
        if (v55) {	// L85
          int32_t v56 = base;	// L86
          ap_int<34> v57 = v56;	// L87
          ap_int<34> v58 = v48;	// L88
          ap_int<34> v59 = v57 + v58;	// L89
          int32_t v60 = v59;	// L90
          int32_t idx;	// L91
          idx = v60;	// L92
          int32_t v62 = idx;	// L93
          int32_t v63 = v62 >> 2;	// L94
          int v64 = v63;	// L95
          int32_t v65 = wf[v64];	// L96
          int32_t packed;	// L97
          packed = v65;	// L98
          int32_t v67 = packed;	// L99
          int32_t v68 = idx;	// L100
          int32_t v69 = v68 & 3;	// L101
          int64_t v70 = v69;	// L102
          int64_t v71 = v70 * 8;	// L103
          int32_t v72 = v71;	// L104
          int32_t v73 = v67 >> v72;	// L105
          int32_t v74 = v73 & 255;	// L106
          int32_t byte;	// L107
          byte = v74;	// L108
          int32_t v76 = byte;	// L109
          int32_t v77 = v76 ^ 128;	// L110
          ap_int<33> v78 = v77;	// L111
          ap_int<33> v79 = v78 - 128;	// L112
          int32_t v80 = v79;	// L113
          int32_t wt;	// L114
          wt = v80;	// L115
          int32_t v82 = p;	// L116
          int8_t v83 = a;	// L117
          int32_t v84 = wt;	// L118
          ap_int<40> v85 = v83;	// L119
          ap_int<40> v86 = v84;	// L120
          ap_int<40> v87 = v85 * v86;	// L121
          ap_int<41> v88 = v82;	// L122
          ap_int<41> v89 = v87;	// L123
          ap_int<41> v90 = v88 + v89;	// L124
          v7.write(v90);	// L125
        } else {
          int32_t v91 = p;	// L127
          v7.write(v91);	// L128
        }
      }
    }
  }
}

/// This is top function.
void top(

) {	// L135
  #pragma HLS dataflow
  hls::stream< int8_t > v92;
  #pragma HLS stream variable=v92 depth=2	// L136
  hls::stream< int8_t > v93;
  #pragma HLS stream variable=v93 depth=2	// L137
  hls::stream< int32_t > v94;
  #pragma HLS stream variable=v94 depth=2	// L138
  hls::stream< int32_t > v95;
  #pragma HLS stream variable=v95 depth=2	// L139
  hls::stream< int32_t > v96;
  #pragma HLS stream variable=v96 depth=2	// L140
  hls::stream< int32_t > v97;
  #pragma HLS stream variable=v97 depth=2	// L141
  hls::stream< int32_t > v98;
  #pragma HLS stream variable=v98 depth=2	// L142
  hls::stream< int32_t > v99;
  #pragma HLS stream variable=v99 depth=2	// L143
  mac_r4_0(v94, v95, v98, v99, v92, v96, v93, v97);	// L144
}



//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void mac_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int8_t >& v5,
  hls::stream< int32_t >& v6
) {	// L2
  int32_t wf[64];	// L19
  for (int v8 = 0; v8 < 64; v8++) {	// L20
    wf[v8] = 0;	// L20
  }
  int32_t v9 = v0.read();	// L21
  int32_t count;	// L22
  count = v9;	// L23
  int32_t v11 = count;	// L24
  v1.write(v11);	// L25
  int32_t v12 = count;	// L26
  int v13 = v12;	// L27
  for (int v14 = 0; v14 < v13; v14 += 1) {	// L28
    int32_t v15 = v0.read();	// L29
    int32_t word;	// L30
    word = v15;	// L31
    int32_t v17 = word;	// L32
    int32_t v18 = v17 >> 24;	// L33
    int32_t v19 = v18 & 255;	// L34
    int32_t opcode;	// L35
    opcode = v19;	// L36
    int32_t v21 = word;	// L37
    int32_t v22 = v21 >> 16;	// L38
    int32_t v23 = v22 & 255;	// L39
    int32_t base;	// L40
    base = v23;	// L41
    int32_t v25 = word;	// L42
    int32_t v26 = v25 & 65535;	// L43
    int32_t n;	// L44
    n = v26;	// L45
    int32_t v28 = opcode;	// L46
    bool v29 = v28 == 6;	// L47
    if (v29) {	// L48
      int32_t v30 = n;	// L49
      ap_int<33> v31 = v30;	// L50
      ap_int<33> v32 = v31 - 64;	// L51
      ap_int<33> v33 = v32 | 100663296;	// L52
      v1.write(v33);	// L53
      l_S_i_0_i: for (int i = 0; i < 64; i++) {	// L54
        int32_t v35 = v2.read();	// L55
        wf[i] = v35;	// L56
      }
      int32_t v36 = n;	// L58
      ap_int<33> v37 = v36;	// L59
      ap_int<33> v38 = v37 - 64;	// L60
      int v39 = v38;	// L61
      for (int v40 = 0; v40 < v39; v40 += 1) {	// L62
        int32_t v41 = v2.read();	// L63
        int32_t x;	// L64
        x = v41;	// L65
        int32_t v43 = x;	// L66
        v3.write(v43);	// L67
      }
    } else {
      int32_t v44 = word;	// L70
      v1.write(v44);	// L71
      int32_t v45 = n;	// L72
      int v46 = v45;	// L73
      for (int v47 = 0; v47 < v46; v47 += 1) {	// L74
        int8_t v48 = v4.read();	// L75
        int8_t a;	// L76
        a = v48;	// L77
        int32_t p;	// L78
        p = 0;	// L79
        int8_t v51 = a;	// L80
        v5.write(v51);	// L81
        int32_t v52 = opcode;	// L82
        bool v53 = v52 == 4;	// L83
        if (v53) {	// L84
          int32_t v54 = base;	// L85
          ap_int<34> v55 = v54;	// L86
          ap_int<34> v56 = v47;	// L87
          ap_int<34> v57 = v55 + v56;	// L88
          int32_t v58 = v57;	// L89
          int32_t idx;	// L90
          idx = v58;	// L91
          int32_t v60 = idx;	// L92
          int32_t v61 = v60 >> 2;	// L93
          int v62 = v61;	// L94
          int32_t v63 = wf[v62];	// L95
          int32_t packed;	// L96
          packed = v63;	// L97
          int32_t v65 = packed;	// L98
          int32_t v66 = idx;	// L99
          int32_t v67 = v66 & 3;	// L100
          int64_t v68 = v67;	// L101
          int64_t v69 = v68 * 8;	// L102
          int32_t v70 = v69;	// L103
          int32_t v71 = v65 >> v70;	// L104
          int32_t v72 = v71 & 255;	// L105
          int32_t byte;	// L106
          byte = v72;	// L107
          int32_t v74 = byte;	// L108
          int32_t v75 = v74 ^ 128;	// L109
          ap_int<33> v76 = v75;	// L110
          ap_int<33> v77 = v76 - 128;	// L111
          int32_t v78 = v77;	// L112
          int32_t wt;	// L113
          wt = v78;	// L114
          int32_t v80 = p;	// L115
          int8_t v81 = a;	// L116
          int32_t v82 = wt;	// L117
          ap_int<40> v83 = v81;	// L118
          ap_int<40> v84 = v82;	// L119
          ap_int<40> v85 = v83 * v84;	// L120
          ap_int<41> v86 = v80;	// L121
          ap_int<41> v87 = v85;	// L122
          ap_int<41> v88 = v86 + v87;	// L123
          v6.write(v88);	// L124
        } else {
          int32_t v89 = p;	// L126
          v6.write(v89);	// L127
        }
      }
    }
  }
}

/// This is top function.
void top(

) {	// L134
  #pragma HLS dataflow
  hls::stream< int8_t > v90;
  #pragma HLS stream variable=v90 depth=2	// L135
  hls::stream< int8_t > v91;
  #pragma HLS stream variable=v91 depth=2	// L136
  hls::stream< int32_t > v92;
  #pragma HLS stream variable=v92 depth=2	// L137
  hls::stream< int32_t > v93;
  #pragma HLS stream variable=v93 depth=2	// L138
  hls::stream< int32_t > v94;
  #pragma HLS stream variable=v94 depth=2	// L139
  hls::stream< int32_t > v95;
  #pragma HLS stream variable=v95 depth=2	// L140
  hls::stream< int32_t > v96;
  #pragma HLS stream variable=v96 depth=2	// L141
  mac_r2_0(v92, v93, v95, v96, v90, v91, v94);	// L142
}



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
  hls::stream< int32_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int32_t wf[64];	// L18
  for (int v5 = 0; v5 < 64; v5++) {	// L19
    wf[v5] = 0;	// L19
  }
  int32_t v6 = v0.read();	// L20
  int32_t count;	// L21
  count = v6;	// L22
  int32_t v8 = count;	// L23
  int v9 = v8;	// L24
  for (int v10 = 0; v10 < v9; v10 += 1) {	// L25
    int32_t v11 = v0.read();	// L26
    int32_t word;	// L27
    word = v11;	// L28
    int32_t v13 = word;	// L29
    int32_t v14 = v13 >> 24;	// L30
    int32_t v15 = v14 & 255;	// L31
    int32_t opcode;	// L32
    opcode = v15;	// L33
    int32_t v17 = word;	// L34
    int32_t v18 = v17 >> 16;	// L35
    int32_t v19 = v18 & 255;	// L36
    int32_t base;	// L37
    base = v19;	// L38
    int32_t v21 = word;	// L39
    int32_t v22 = v21 & 65535;	// L40
    int32_t n;	// L41
    n = v22;	// L42
    int32_t v24 = opcode;	// L43
    bool v25 = v24 == 6;	// L44
    if (v25) {	// L45
      l_S_i_0_i: for (int i = 0; i < 64; i++) {	// L46
        int32_t v27 = v1.read();	// L47
        wf[i] = v27;	// L48
      }
      int32_t v28 = n;	// L50
      ap_int<33> v29 = v28;	// L51
      ap_int<33> v30 = v29 - 64;	// L52
      int v31 = v30;	// L53
      for (int v32 = 0; v32 < v31; v32 += 1) {	// L54
        int32_t v33 = v1.read();	// L55
        int32_t x;	// L56
        x = v33;	// L57
      }
    } else {
      int32_t v35 = n;	// L60
      int v36 = v35;	// L61
      for (int v37 = 0; v37 < v36; v37 += 1) {	// L62
        int8_t v38 = v2.read();	// L63
        int8_t a;	// L64
        a = v38;	// L65
        int32_t p;	// L66
        p = 0;	// L67
        int32_t v41 = opcode;	// L68
        bool v42 = v41 == 4;	// L69
        if (v42) {	// L70
          int32_t v43 = base;	// L71
          ap_int<34> v44 = v43;	// L72
          ap_int<34> v45 = v37;	// L73
          ap_int<34> v46 = v44 + v45;	// L74
          int32_t v47 = v46;	// L75
          int32_t idx;	// L76
          idx = v47;	// L77
          int32_t v49 = idx;	// L78
          int32_t v50 = v49 >> 2;	// L79
          int v51 = v50;	// L80
          int32_t v52 = wf[v51];	// L81
          int32_t packed;	// L82
          packed = v52;	// L83
          int32_t v54 = packed;	// L84
          int32_t v55 = idx;	// L85
          int32_t v56 = v55 & 3;	// L86
          int64_t v57 = v56;	// L87
          int64_t v58 = v57 * 8;	// L88
          int32_t v59 = v58;	// L89
          int32_t v60 = v54 >> v59;	// L90
          int32_t v61 = v60 & 255;	// L91
          int32_t byte;	// L92
          byte = v61;	// L93
          int32_t v63 = byte;	// L94
          int32_t v64 = v63 ^ 128;	// L95
          ap_int<33> v65 = v64;	// L96
          ap_int<33> v66 = v65 - 128;	// L97
          int32_t v67 = v66;	// L98
          int32_t wt;	// L99
          wt = v67;	// L100
          int32_t v69 = p;	// L101
          int8_t v70 = a;	// L102
          int32_t v71 = wt;	// L103
          ap_int<40> v72 = v70;	// L104
          ap_int<40> v73 = v71;	// L105
          ap_int<40> v74 = v72 * v73;	// L106
          ap_int<41> v75 = v69;	// L107
          ap_int<41> v76 = v74;	// L108
          ap_int<41> v77 = v75 + v76;	// L109
          v3.write(v77);	// L110
        } else {
          int32_t v78 = p;	// L112
          v3.write(v78);	// L113
        }
      }
    }
  }
}

/// This is top function.
void top(

) {	// L120
  #pragma HLS dataflow
  hls::stream< int8_t > v79;
  #pragma HLS stream variable=v79 depth=2	// L121
  hls::stream< int32_t > v80;
  #pragma HLS stream variable=v80 depth=2	// L122
  hls::stream< int32_t > v81;
  #pragma HLS stream variable=v81 depth=2	// L123
  hls::stream< int32_t > v82;
  #pragma HLS stream variable=v82 depth=2	// L124
  mac_r6_0(v80, v82, v79, v81);	// L125
}


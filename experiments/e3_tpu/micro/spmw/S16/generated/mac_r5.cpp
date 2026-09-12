
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void mac_r5_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t wf[4];	// L18
  for (int v6 = 0; v6 < 4; v6++) {	// L19
    wf[v6] = 0;	// L19
  }
  int32_t v7 = v0.read();	// L20
  int32_t count;	// L21
  count = v7;	// L22
  int32_t v9 = count;	// L23
  int v10 = v9;	// L24
  for (int v11 = 0; v11 < v10; v11 += 1) {	// L25
    int32_t v12 = v0.read();	// L26
    int32_t word;	// L27
    word = v12;	// L28
    int32_t v14 = word;	// L29
    int32_t v15 = v14 >> 24;	// L30
    int32_t v16 = v15 & 255;	// L31
    int32_t opcode;	// L32
    opcode = v16;	// L33
    int32_t v18 = word;	// L34
    int32_t v19 = v18 >> 16;	// L35
    int32_t v20 = v19 & 255;	// L36
    int32_t base;	// L37
    base = v20;	// L38
    int32_t v22 = word;	// L39
    int32_t v23 = v22 & 65535;	// L40
    int32_t n;	// L41
    n = v23;	// L42
    int32_t v25 = opcode;	// L43
    bool v26 = v25 == 6;	// L44
    if (v26) {	// L45
      l_S_i_0_i: for (int i = 0; i < 4; i++) {	// L46
        int32_t v28 = v1.read();	// L47
        wf[i] = v28;	// L48
      }
      int32_t v29 = n;	// L50
      ap_int<33> v30 = v29;	// L51
      ap_int<33> v31 = v30 - 4;	// L52
      int v32 = v31;	// L53
      for (int v33 = 0; v33 < v32; v33 += 1) {	// L54
        int32_t v34 = v1.read();	// L55
        int32_t x;	// L56
        x = v34;	// L57
      }
    } else {
      int32_t v36 = n;	// L60
      int v37 = v36;	// L61
      for (int v38 = 0; v38 < v37; v38 += 1) {	// L62
        int8_t v39 = v2.read();	// L63
        int8_t a;	// L64
        a = v39;	// L65
        int32_t v41 = v3.read();	// L66
        int32_t p;	// L67
        p = v41;	// L68
        int32_t v43 = opcode;	// L69
        bool v44 = v43 == 4;	// L70
        if (v44) {	// L71
          int32_t v45 = base;	// L72
          ap_int<34> v46 = v45;	// L73
          ap_int<34> v47 = v38;	// L74
          ap_int<34> v48 = v46 + v47;	// L75
          int32_t v49 = v48;	// L76
          int32_t idx;	// L77
          idx = v49;	// L78
          int32_t v51 = idx;	// L79
          int32_t v52 = v51 >> 2;	// L80
          int v53 = v52;	// L81
          int32_t v54 = wf[v53];	// L82
          int32_t packed;	// L83
          packed = v54;	// L84
          int32_t v56 = packed;	// L85
          int32_t v57 = idx;	// L86
          int32_t v58 = v57 & 3;	// L87
          int64_t v59 = v58;	// L88
          int64_t v60 = v59 * 8;	// L89
          int32_t v61 = v60;	// L90
          int32_t v62 = v56 >> v61;	// L91
          int32_t v63 = v62 & 255;	// L92
          int32_t byte;	// L93
          byte = v63;	// L94
          int32_t v65 = byte;	// L95
          int32_t v66 = v65 ^ 128;	// L96
          ap_int<33> v67 = v66;	// L97
          ap_int<33> v68 = v67 - 128;	// L98
          int32_t v69 = v68;	// L99
          int32_t wt;	// L100
          wt = v69;	// L101
          int32_t v71 = p;	// L102
          int8_t v72 = a;	// L103
          int32_t v73 = wt;	// L104
          ap_int<40> v74 = v72;	// L105
          ap_int<40> v75 = v73;	// L106
          ap_int<40> v76 = v74 * v75;	// L107
          ap_int<41> v77 = v71;	// L108
          ap_int<41> v78 = v76;	// L109
          ap_int<41> v79 = v77 + v78;	// L110
          v4.write(v79);	// L111
        } else {
          int32_t v80 = p;	// L113
          v4.write(v80);	// L114
        }
      }
    }
  }
}

/// This is top function.
void top(

) {	// L121
  #pragma HLS dataflow
  hls::stream< int8_t > v81;
  #pragma HLS stream variable=v81 depth=2	// L122
  hls::stream< int32_t > v82;
  #pragma HLS stream variable=v82 depth=2	// L123
  hls::stream< int32_t > v83;
  #pragma HLS stream variable=v83 depth=2	// L124
  hls::stream< int32_t > v84;
  #pragma HLS stream variable=v84 depth=2	// L125
  hls::stream< int32_t > v85;
  #pragma HLS stream variable=v85 depth=2	// L126
  mac_r5_0(v82, v85, v81, v83, v84);	// L127
}


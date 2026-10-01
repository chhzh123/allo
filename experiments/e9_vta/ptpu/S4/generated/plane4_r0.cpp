
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
void plane4_r0_0(
  hls::stream< int32_t >& v0,
  hls::stream< int16_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int32_t bias;	// L5
  bias = 0;	// L6
  bool go;	// L10
  go = 1;	// L11
  int32_t r0;	// L14
  r0 = 0;	// L15
  int32_t r1;	// L18
  r1 = 0;	// L19
  int32_t r2;	// L22
  r2 = 0;	// L23
  int32_t r3;	// L26
  r3 = 0;	// L27
  while (true) {	// L28
    #pragma HLS pipeline II=1 style=flp
    bool v10 = go;	// L29
    if (!(v10)) break;
    int16_t v11 = v1.read();	// L36
    int16_t c;	// L37
    c = v11;	// L38
    int32_t v13 = v3.read();	// L39
    int32_t z;	// L40
    z = v13;	// L41
    int16_t v15 = c;	// L42
    int16_t v16 = v15 >> 10;	// L46
    int32_t v17 = v16;	// L47
    int32_t v18 = v17 & 1;	// L50
    bool v19 = v18 != 0;	// L53
    if (v19) {	// L54
      int32_t v20 = v0.read();	// L55
      bias = v20;	// L56
    }
    int32_t v21 = r0;	// L58
    int32_t s;	// L59
    s = v21;	// L60
    int16_t v23 = c;	// L61
    int16_t v24 = v23 >> 8;	// L65
    int32_t v25 = v24;	// L66
    int32_t v26 = v25 & 1;	// L69
    bool v27 = v26 != 0;	// L72
    if (v27) {	// L73
      s = 0;	// L76
      int16_t v28 = c;	// L77
      int16_t v29 = v28 >> 7;	// L81
      int32_t v30 = v29;	// L82
      int32_t v31 = v30 & 1;	// L85
      bool v32 = v31 != 0;	// L88
      if (v32) {	// L89
        int32_t v33 = bias;	// L90
        s = v33;	// L91
      }
    }
    int32_t v34 = s;	// L94
    int32_t v35 = z;	// L95
    ap_int<33> v36 = v34;	// L96
    ap_int<33> v37 = v35;	// L97
    ap_int<33> v38 = v36 + v37;	// L98
    int32_t v39 = v38;	// L99
    int32_t v;	// L100
    v = v39;	// L101
    int32_t v41 = r1;	// L102
    r0 = v41;	// L103
    int32_t v42 = r2;	// L104
    r1 = v42;	// L105
    int32_t v43 = r3;	// L106
    r2 = v43;	// L107
    int32_t v44 = v;	// L108
    r3 = v44;	// L109
    int16_t v45 = c;	// L110
    int16_t v46 = v45 >> 9;	// L114
    int32_t v47 = v46;	// L115
    int32_t v48 = v47 & 1;	// L118
    bool v49 = v48 != 0;	// L121
    if (v49) {	// L122
      int32_t v50 = v;	// L123
      int32_t y;	// L124
      y = v50;	// L125
      int16_t v52 = c;	// L126
      int16_t v53 = v52 >> 6;	// L130
      int32_t v54 = v53;	// L131
      int32_t v55 = v54 & 1;	// L134
      bool v56 = v55 == 0;	// L137
      if (v56) {	// L138
        int16_t v57 = c;	// L139
        int16_t v58 = v57 >> 5;	// L143
        int32_t v59 = v58;	// L144
        int32_t v60 = v59 & 1;	// L147
        bool v61 = v60 != 0;	// L150
        if (v61) {	// L151
          int32_t v62 = y;	// L152
          bool v63 = v62 < 0;	// L155
          if (v63) {	// L156
            y = 0;	// L159
          }
        }
        int32_t v64 = y;	// L162
        int16_t v65 = c;	// L163
        int32_t v66 = v65;	// L164
        int32_t v67 = v66 & 31;	// L167
        int32_t v68 = v64 >> v67;	// L168
        y = v68;	// L169
        int32_t v69 = y;	// L170
        bool v70 = v69 < -128;	// L175
        if (v70) {	// L176
          y = -128;	// L181
        }
        int32_t v71 = y;	// L183
        bool v72 = v71 > 127;	// L186
        if (v72) {	// L187
          y = 127;	// L190
        }
      }
      int32_t v73 = y;	// L193
      v2.write(v73);	// L194
    }
    int16_t v74 = c;	// L196
    int16_t v75 = v74 >> 11;	// L200
    int32_t v76 = v75;	// L201
    int32_t v77 = v76 & 1;	// L204
    bool v78 = v77 != 0;	// L207
    if (v78) {	// L208
      go = 0;	// L212
    }
  }
}



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
void plane8_r0_0(
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
  int32_t r4;	// L30
  r4 = 0;	// L31
  int32_t r5;	// L34
  r5 = 0;	// L35
  int32_t r6;	// L38
  r6 = 0;	// L39
  int32_t r7;	// L42
  r7 = 0;	// L43
  while (true) {	// L44
    #pragma HLS pipeline II=1 style=flp
    bool v14 = go;	// L45
    if (!(v14)) break;
    int16_t v15 = v1.read();	// L52
    int16_t c;	// L53
    c = v15;	// L54
    int32_t v17 = v3.read();	// L55
    int32_t z;	// L56
    z = v17;	// L57
    int16_t v19 = c;	// L58
    int16_t v20 = v19 >> 10;	// L62
    int32_t v21 = v20;	// L63
    int32_t v22 = v21 & 1;	// L66
    bool v23 = v22 != 0;	// L69
    if (v23) {	// L70
      int32_t v24 = v0.read();	// L71
      bias = v24;	// L72
    }
    int32_t v25 = r0;	// L74
    int32_t s;	// L75
    s = v25;	// L76
    int16_t v27 = c;	// L77
    int16_t v28 = v27 >> 8;	// L81
    int32_t v29 = v28;	// L82
    int32_t v30 = v29 & 1;	// L85
    bool v31 = v30 != 0;	// L88
    if (v31) {	// L89
      s = 0;	// L92
      int16_t v32 = c;	// L93
      int16_t v33 = v32 >> 7;	// L97
      int32_t v34 = v33;	// L98
      int32_t v35 = v34 & 1;	// L101
      bool v36 = v35 != 0;	// L104
      if (v36) {	// L105
        int32_t v37 = bias;	// L106
        s = v37;	// L107
      }
    }
    int32_t v38 = s;	// L110
    int32_t v39 = z;	// L111
    ap_int<33> v40 = v38;	// L112
    ap_int<33> v41 = v39;	// L113
    ap_int<33> v42 = v40 + v41;	// L114
    int32_t v43 = v42;	// L115
    int32_t v;	// L116
    v = v43;	// L117
    int32_t v45 = r1;	// L118
    r0 = v45;	// L119
    int32_t v46 = r2;	// L120
    r1 = v46;	// L121
    int32_t v47 = r3;	// L122
    r2 = v47;	// L123
    int32_t v48 = r4;	// L124
    r3 = v48;	// L125
    int32_t v49 = r5;	// L126
    r4 = v49;	// L127
    int32_t v50 = r6;	// L128
    r5 = v50;	// L129
    int32_t v51 = r7;	// L130
    r6 = v51;	// L131
    int32_t v52 = v;	// L132
    r7 = v52;	// L133
    int16_t v53 = c;	// L134
    int16_t v54 = v53 >> 9;	// L138
    int32_t v55 = v54;	// L139
    int32_t v56 = v55 & 1;	// L142
    bool v57 = v56 != 0;	// L145
    if (v57) {	// L146
      int32_t v58 = v;	// L147
      int32_t y;	// L148
      y = v58;	// L149
      int16_t v60 = c;	// L150
      int16_t v61 = v60 >> 6;	// L154
      int32_t v62 = v61;	// L155
      int32_t v63 = v62 & 1;	// L158
      bool v64 = v63 == 0;	// L161
      if (v64) {	// L162
        int16_t v65 = c;	// L163
        int16_t v66 = v65 >> 5;	// L167
        int32_t v67 = v66;	// L168
        int32_t v68 = v67 & 1;	// L171
        bool v69 = v68 != 0;	// L174
        if (v69) {	// L175
          int32_t v70 = y;	// L176
          bool v71 = v70 < 0;	// L179
          if (v71) {	// L180
            y = 0;	// L183
          }
        }
        int32_t v72 = y;	// L186
        int16_t v73 = c;	// L187
        int32_t v74 = v73;	// L188
        int32_t v75 = v74 & 31;	// L191
        int32_t v76 = v72 >> v75;	// L192
        y = v76;	// L193
        int32_t v77 = y;	// L194
        bool v78 = v77 < -128;	// L199
        if (v78) {	// L200
          y = -128;	// L205
        }
        int32_t v79 = y;	// L207
        bool v80 = v79 > 127;	// L210
        if (v80) {	// L211
          y = 127;	// L214
        }
      }
      int32_t v81 = y;	// L217
      v2.write(v81);	// L218
    }
    int16_t v82 = c;	// L220
    int16_t v83 = v82 >> 11;	// L224
    int32_t v84 = v83;	// L225
    int32_t v85 = v84 & 1;	// L228
    bool v86 = v85 != 0;	// L231
    if (v86) {	// L232
      go = 0;	// L236
    }
  }
}


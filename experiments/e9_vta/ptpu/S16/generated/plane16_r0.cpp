
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
void plane16_r0_0(
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
  int32_t r8;	// L46
  r8 = 0;	// L47
  int32_t r9;	// L50
  r9 = 0;	// L51
  int32_t r10;	// L54
  r10 = 0;	// L55
  int32_t r11;	// L58
  r11 = 0;	// L59
  int32_t r12;	// L62
  r12 = 0;	// L63
  int32_t r13;	// L66
  r13 = 0;	// L67
  int32_t r14;	// L70
  r14 = 0;	// L71
  int32_t r15;	// L74
  r15 = 0;	// L75
  while (true) {	// L76
    #pragma HLS pipeline II=1 style=flp
    bool v22 = go;	// L77
    if (!(v22)) break;
    int16_t v23 = v1.read();	// L84
    int16_t c;	// L85
    c = v23;	// L86
    int32_t v25 = v3.read();	// L87
    int32_t z;	// L88
    z = v25;	// L89
    int16_t v27 = c;	// L90
    int16_t v28 = v27 >> 10;	// L94
    int32_t v29 = v28;	// L95
    int32_t v30 = v29 & 1;	// L98
    bool v31 = v30 != 0;	// L101
    if (v31) {	// L102
      int32_t v32 = v0.read();	// L103
      bias = v32;	// L104
    }
    int32_t v33 = r0;	// L106
    int32_t s;	// L107
    s = v33;	// L108
    int16_t v35 = c;	// L109
    int16_t v36 = v35 >> 8;	// L113
    int32_t v37 = v36;	// L114
    int32_t v38 = v37 & 1;	// L117
    bool v39 = v38 != 0;	// L120
    if (v39) {	// L121
      s = 0;	// L124
      int16_t v40 = c;	// L125
      int16_t v41 = v40 >> 7;	// L129
      int32_t v42 = v41;	// L130
      int32_t v43 = v42 & 1;	// L133
      bool v44 = v43 != 0;	// L136
      if (v44) {	// L137
        int32_t v45 = bias;	// L138
        s = v45;	// L139
      }
    }
    int32_t v46 = s;	// L142
    int32_t v47 = z;	// L143
    ap_int<33> v48 = v46;	// L144
    ap_int<33> v49 = v47;	// L145
    ap_int<33> v50 = v48 + v49;	// L146
    int32_t v51 = v50;	// L147
    int32_t v;	// L148
    v = v51;	// L149
    int32_t v53 = r1;	// L150
    r0 = v53;	// L151
    int32_t v54 = r2;	// L152
    r1 = v54;	// L153
    int32_t v55 = r3;	// L154
    r2 = v55;	// L155
    int32_t v56 = r4;	// L156
    r3 = v56;	// L157
    int32_t v57 = r5;	// L158
    r4 = v57;	// L159
    int32_t v58 = r6;	// L160
    r5 = v58;	// L161
    int32_t v59 = r7;	// L162
    r6 = v59;	// L163
    int32_t v60 = r8;	// L164
    r7 = v60;	// L165
    int32_t v61 = r9;	// L166
    r8 = v61;	// L167
    int32_t v62 = r10;	// L168
    r9 = v62;	// L169
    int32_t v63 = r11;	// L170
    r10 = v63;	// L171
    int32_t v64 = r12;	// L172
    r11 = v64;	// L173
    int32_t v65 = r13;	// L174
    r12 = v65;	// L175
    int32_t v66 = r14;	// L176
    r13 = v66;	// L177
    int32_t v67 = r15;	// L178
    r14 = v67;	// L179
    int32_t v68 = v;	// L180
    r15 = v68;	// L181
    int16_t v69 = c;	// L182
    int16_t v70 = v69 >> 9;	// L186
    int32_t v71 = v70;	// L187
    int32_t v72 = v71 & 1;	// L190
    bool v73 = v72 != 0;	// L193
    if (v73) {	// L194
      int32_t v74 = v;	// L195
      int32_t y;	// L196
      y = v74;	// L197
      int16_t v76 = c;	// L198
      int16_t v77 = v76 >> 6;	// L202
      int32_t v78 = v77;	// L203
      int32_t v79 = v78 & 1;	// L206
      bool v80 = v79 == 0;	// L209
      if (v80) {	// L210
        int16_t v81 = c;	// L211
        int16_t v82 = v81 >> 5;	// L215
        int32_t v83 = v82;	// L216
        int32_t v84 = v83 & 1;	// L219
        bool v85 = v84 != 0;	// L222
        if (v85) {	// L223
          int32_t v86 = y;	// L224
          bool v87 = v86 < 0;	// L227
          if (v87) {	// L228
            y = 0;	// L231
          }
        }
        int32_t v88 = y;	// L234
        int16_t v89 = c;	// L235
        int32_t v90 = v89;	// L236
        int32_t v91 = v90 & 31;	// L239
        int32_t v92 = v88 >> v91;	// L240
        y = v92;	// L241
        int32_t v93 = y;	// L242
        bool v94 = v93 < -128;	// L247
        if (v94) {	// L248
          y = -128;	// L253
        }
        int32_t v95 = y;	// L255
        bool v96 = v95 > 127;	// L258
        if (v96) {	// L259
          y = 127;	// L262
        }
      }
      int32_t v97 = y;	// L265
      v2.write(v97);	// L266
    }
    int16_t v98 = c;	// L268
    int16_t v99 = v98 >> 11;	// L272
    int32_t v100 = v99;	// L273
    int32_t v101 = v100 & 1;	// L276
    bool v102 = v101 != 0;	// L279
    if (v102) {	// L280
      go = 0;	// L284
    }
  }
}


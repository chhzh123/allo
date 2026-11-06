
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
void lane_r0_0(
  hls::stream< int64_t >& v0,
  hls::stream< int16_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int32_t v4 = v3.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v4;	// L5
  int32_t v6 = _st__pid0;	// L6
  int32_t slot;	// L7
  slot = v6;	// L8
  static int32_t buf[1024] = {0};	// L9
  int32_t bnext;	// L14
  bnext = 0;	// L15
  int32_t bias;	// L18
  bias = 0;	// L19
  int32_t p;	// L22
  p = 0;	// L23
  bool go;	// L27
  go = 1;	// L28
  while (true) {	// L29
    #pragma HLS dependence variable=buf type=inter direction=RAW distance=4 true
    #pragma HLS pipeline II=1 style=flp
    bool v14 = go;	// L30
    if (!(v14)) break;
    int64_t v15 = v0.read();	// L37
    int64_t u;	// L38
    u = v15;	// L39
    int64_t v17 = u;	// L40
    int64_t v18 = v17 >> 9;	// L44
    int64_t v19 = v18 & 1;	// L48
    bool v20 = v19 != 0;	// L52
    if (v20) {	// L53
      int32_t v21 = bnext;	// L54
      bias = v21;	// L55
    }
    int64_t v22 = u;	// L57
    int64_t v23 = v22 >> 12;	// L61
    int64_t v24 = v23 & 1;	// L65
    bool v25 = v24 != 0;	// L69
    if (v25) {	// L70
      int64_t v26 = u;	// L71
      int64_t v27 = v26 >> 13;	// L75
      int64_t v28 = v27 & 31;	// L79
      int32_t v29 = slot;	// L80
      int64_t v30 = v29;	// L81
      bool v31 = v28 == v30;	// L82
      if (v31) {	// L83
        int64_t v32 = u;	// L84
        int64_t v33 = v32 >> 32;	// L88
        int32_t v34 = v33;	// L89
        bnext = v34;	// L90
      }
    }
    int64_t v35 = u;	// L93
    int64_t v36 = v35 >> 18;	// L97
    int64_t v37 = v36 & 1;	// L101
    bool v38 = v37 != 0;	// L105
    if (v38) {	// L106
      int32_t v39 = v2.read();	// L107
      int32_t z;	// L108
      z = v39;	// L109
      int32_t v41 = p;	// L110
      int v42 = v41;	// L111
      int32_t v43 = buf[v42];	// L112
      int32_t s;	// L113
      s = v43;	// L114
      int64_t v45 = u;	// L115
      int64_t v46 = v45 >> 7;	// L119
      int64_t v47 = v46 & 1;	// L123
      bool v48 = v47 != 0;	// L127
      if (v48) {	// L128
        s = 0;	// L131
        int64_t v49 = u;	// L132
        int64_t v50 = v49 >> 6;	// L136
        int64_t v51 = v50 & 1;	// L140
        bool v52 = v51 != 0;	// L144
        if (v52) {	// L145
          int32_t v53 = bias;	// L146
          s = v53;	// L147
        }
      }
      int32_t v54 = s;	// L150
      int32_t v55 = z;	// L151
      ap_int<33> v56 = v54;	// L152
      ap_int<33> v57 = v55;	// L153
      ap_int<33> v58 = v56 + v57;	// L154
      int32_t v59 = v58;	// L155
      int32_t v;	// L156
      v = v59;	// L157
      int32_t v61 = v;	// L158
      int32_t v62 = p;	// L159
      int v63 = v62;	// L160
      buf[v63] = v61;	// L161
      int64_t v64 = u;	// L162
      int64_t v65 = v64 >> 8;	// L166
      int64_t v66 = v65 & 1;	// L170
      bool v67 = v66 != 0;	// L174
      if (v67) {	// L175
        int32_t v68 = v;	// L176
        int32_t y;	// L177
        y = v68;	// L178
        int64_t v70 = u;	// L179
        int64_t v71 = v70 >> 5;	// L183
        int64_t v72 = v71 & 1;	// L187
        bool v73 = v72 != 0;	// L191
        if (v73) {	// L192
          int32_t v74 = y;	// L193
          bool v75 = v74 < 0;	// L196
          if (v75) {	// L197
            y = 0;	// L200
          }
        }
        int64_t v76 = u;	// L203
        int64_t v77 = v76 & 31;	// L207
        int32_t v78 = v77;	// L208
        int32_t sh;	// L209
        sh = v78;	// L210
        int32_t v80 = y;	// L211
        int32_t v81 = sh;	// L212
        int32_t v82 = v80 >> v81;	// L213
        y = v82;	// L214
        int32_t v83 = y;	// L215
        bool v84 = v83 < -128;	// L220
        if (v84) {	// L221
          y = -128;	// L226
        }
        int32_t v85 = y;	// L228
        bool v86 = v85 > 127;	// L231
        if (v86) {	// L232
          y = 127;	// L235
        }
        int32_t v87 = y;	// L237
        int32_t v88 = v87 & 255;	// L240
        y = v88;	// L241
        int64_t v89 = u;	// L242
        int64_t v90 = v89 >> 10;	// L246
        int64_t v91 = v90 & 1;	// L250
        bool v92 = v91 != 0;	// L254
        if (v92) {	// L255
          int32_t v93 = y;	// L256
          int32_t v94 = v93 | 256;	// L259
          y = v94;	// L260
        }
        int32_t v95 = y;	// L262
        v1.write(v95);	// L263
      }
      int64_t v96 = u;	// L265
      int64_t v97 = v96 >> 11;	// L269
      int64_t v98 = v97 & 1;	// L273
      bool v99 = v98 != 0;	// L277
      if (v99) {	// L278
        p = 0;	// L281
      } else {
        int32_t v100 = p;	// L283
        ap_int<33> v101 = v100;	// L284
        ap_int<33> v102 = v101 + 1;	// L288
        int32_t v103 = v102;	// L289
        p = v103;	// L290
      }
    }
    int64_t v104 = u;	// L293
    int64_t v105 = v104 >> 10;	// L297
    int64_t v106 = v105 & 1;	// L301
    bool v107 = v106 != 0;	// L305
    if (v107) {	// L306
      go = 0;	// L310
    }
  }
}


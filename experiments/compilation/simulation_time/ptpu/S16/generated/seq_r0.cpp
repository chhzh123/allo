
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
void seq_r0_0(
  hls::stream< int64_t >& v0,
  hls::stream< int16_t >& v1
) {	// L2
  int16_t cfg;	// L6
  cfg = 0;	// L7
  bool final;	// L11
  final = 0;	// L12
  uint16_t kn;	// L16
  kn = 0;	// L17
  uint16_t kc;	// L21
  kc = 0;	// L22
  int32_t rc;	// L25
  rc = 0;	// L26
  bool first;	// L30
  first = 1;	// L31
  bool kone;	// L35
  kone = 0;	// L36
  bool klast;	// L40
  klast = 0;	// L41
  bool rlast;	// L45
  rlast = 0;	// L46
  bool need;	// L50
  need = 1;	// L51
  uint8_t ph;	// L55
  ph = 0;	// L56
  bool go;	// L60
  go = 1;	// L61
  while (true) {	// L62
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v14 = go;	// L63
    if (!(v14)) break;
    bool v15 = need;	// L70
    if (v15) {	// L75
      int64_t v16 = v0.read();	// L76
      int64_t ins;	// L77
      ins = v16;	// L78
      int64_t v18 = ins;	// L79
      int64_t v19 = v18 & 255;	// L83
      int16_t v20 = v19;	// L84
      cfg = v20;	// L85
      int64_t v21 = ins;	// L86
      int64_t v22 = v21 >> 8;	// L90
      int64_t v23 = v22 & 1;	// L94
      bool v24 = v23;	// L95
      final = v24;	// L96
      int64_t v25 = ins;	// L97
      int64_t v26 = v25 >> 16;	// L101
      int64_t v27 = v26 & 65535;	// L105
      uint16_t v28 = v27;	// L106
      kn = v28;	// L107
      int16_t v29 = kn;	// L108
      kc = v29;	// L109
      int64_t v30 = ins;	// L110
      int64_t v31 = v30 >> 32;	// L114
      int32_t v32 = v31;	// L115
      rc = v32;	// L116
      kone = 0;	// L120
      int16_t v33 = kn;	// L121
      int32_t v34 = v33;	// L122
      bool v35 = v34 == 0;	// L125
      if (v35) {	// L126
        kone = 1;	// L130
      }
      bool v36 = kone;	// L132
      klast = v36;	// L133
      rlast = 0;	// L137
      int32_t v37 = rc;	// L138
      bool v38 = v37 == 0;	// L141
      if (v38) {	// L142
        rlast = 1;	// L146
      }
      need = 0;	// L151
    }
    int16_t v39 = cfg;	// L153
    int16_t u;	// L154
    u = v39;	// L155
    bool v41 = first;	// L156
    if (v41) {	// L161
      int16_t v42 = u;	// L162
      int32_t v43 = v42;	// L163
      int32_t v44 = v43 | 256;	// L166
      int16_t v45 = v44;	// L167
      u = v45;	// L168
      int8_t v46 = ph;	// L169
      int32_t v47 = v46;	// L170
      bool v48 = v47 == 0;	// L173
      if (v48) {	// L174
        int16_t v49 = cfg;	// L175
        int16_t v50 = v49 >> 7;	// L179
        int32_t v51 = v50;	// L180
        int32_t v52 = v51 & 1;	// L183
        bool v53 = v52 != 0;	// L186
        if (v53) {	// L187
          int16_t v54 = u;	// L188
          int32_t v55 = v54;	// L189
          int32_t v56 = v55 | 1024;	// L192
          int16_t v57 = v56;	// L193
          u = v57;	// L194
        }
      }
    }
    bool v58 = klast;	// L198
    if (v58) {	// L203
      int16_t v59 = u;	// L204
      int32_t v60 = v59;	// L205
      int32_t v61 = v60 | 512;	// L208
      int16_t v62 = v61;	// L209
      u = v62;	// L210
    }
    int8_t v63 = ph;	// L212
    int32_t v64 = v63;	// L213
    bool v65 = v64 == 15;	// L216
    if (v65) {	// L217
      ph = 0;	// L221
      bool v66 = klast;	// L222
      if (v66) {	// L227
        int16_t v67 = kn;	// L228
        kc = v67;	// L229
        first = 1;	// L233
        bool v68 = kone;	// L234
        klast = v68;	// L235
        bool v69 = rlast;	// L236
        if (v69) {	// L241
          need = 1;	// L245
          bool v70 = final;	// L246
          if (v70) {	// L251
            int16_t v71 = u;	// L252
            int32_t v72 = v71;	// L253
            int32_t v73 = v72 | 2048;	// L256
            int16_t v74 = v73;	// L257
            u = v74;	// L258
            go = 0;	// L262
          }
        } else {
          int32_t v75 = rc;	// L265
          bool v76 = v75 == 1;	// L268
          if (v76) {	// L269
            rlast = 1;	// L273
          }
          int32_t v77 = rc;	// L275
          ap_int<33> v78 = v77;	// L276
          ap_int<33> v79 = v78 - 1;	// L280
          int32_t v80 = v79;	// L281
          rc = v80;	// L282
        }
      } else {
        first = 0;	// L288
        int16_t v81 = kc;	// L289
        int32_t v82 = v81;	// L290
        bool v83 = v82 == 1;	// L293
        if (v83) {	// L294
          klast = 1;	// L298
        }
        int16_t v84 = kc;	// L300
        ap_int<33> v85 = v84;	// L301
        ap_int<33> v86 = v85 - 1;	// L305
        uint16_t v87 = v86;	// L306
        kc = v87;	// L307
      }
    } else {
      int8_t v88 = ph;	// L310
      ap_int<33> v89 = v88;	// L311
      ap_int<33> v90 = v89 + 1;	// L315
      uint8_t v91 = v90;	// L316
      ph = v91;	// L317
    }
    int16_t v92 = u;	// L319
    v1.write(v92);	// L320
  }
}


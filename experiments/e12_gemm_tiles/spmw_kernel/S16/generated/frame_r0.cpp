
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
void frame_r0_0(
  hls::stream< int16_t >& v0,
  hls::stream< int8_t >& v1
) {	// L2
  int8_t v2 = v1.read();	// L3
  int8_t h0;	// L4
  h0 = v2;	// L5
  int8_t v4 = v1.read();	// L6
  int8_t h1;	// L7
  h1 = v4;	// L8
  int8_t v6 = v1.read();	// L9
  int8_t h2;	// L10
  h2 = v6;	// L11
  int8_t v8 = v1.read();	// L12
  int8_t h3;	// L13
  h3 = v8;	// L14
  int8_t v10 = v1.read();	// L15
  int8_t h4;	// L16
  h4 = v10;	// L17
  int8_t v12 = h0;	// L18
  int32_t v13 = v12;	// L19
  int32_t v14 = v13 & 255;	// L22
  int8_t v15 = h1;	// L23
  int32_t v16 = v15;	// L24
  int32_t v17 = v16 & 255;	// L27
  int32_t v18 = v17 << 8;	// L30
  int32_t v19 = v14 | v18;	// L31
  uint16_t v20 = v19;	// L32
  uint16_t kn;	// L33
  kn = v20;	// L34
  int8_t v22 = h2;	// L35
  int32_t v23 = v22;	// L36
  int32_t v24 = v23 & 255;	// L39
  int8_t v25 = h3;	// L40
  int32_t v26 = v25;	// L41
  int32_t v27 = v26 & 255;	// L44
  int32_t v28 = v27 << 8;	// L47
  int32_t v29 = v24 | v28;	// L48
  int8_t v30 = h4;	// L49
  int32_t v31 = v30;	// L50
  int32_t v32 = v31 & 255;	// L53
  int32_t v33 = v32 << 16;	// L56
  int32_t v34 = v29 | v33;	// L57
  int32_t tc;	// L58
  tc = v34;	// L59
  int16_t v36 = kn;	// L60
  uint16_t kc;	// L61
  kc = v36;	// L62
  bool kone;	// L66
  kone = 0;	// L67
  int16_t v39 = kn;	// L68
  int32_t v40 = v39;	// L69
  bool v41 = v40 == 0;	// L72
  if (v41) {	// L73
    kone = 1;	// L77
  }
  bool v42 = kone;	// L79
  bool klast;	// L80
  klast = v42;	// L81
  bool tlast;	// L85
  tlast = 0;	// L86
  int32_t v45 = tc;	// L87
  bool v46 = v45 == 0;	// L90
  if (v46) {	// L91
    tlast = 1;	// L95
  }
  bool go;	// L100
  go = 1;	// L101
  while (true) {	// L102
    bool v48 = go;	// L103
    if (!(v48)) break;
    int8_t v49 = v1.read();	// L110
    int8_t v;	// L111
    v = v49;	// L112
    int8_t v51 = v;	// L113
    int32_t v52 = v51;	// L114
    int32_t v53 = v52 & 255;	// L117
    int16_t v54 = v53;	// L118
    int16_t u;	// L119
    u = v54;	// L120
    bool v56 = klast;	// L121
    if (v56) {	// L126
      int16_t v57 = u;	// L127
      int32_t v58 = v57;	// L128
      int32_t v59 = v58 | 256;	// L131
      int16_t v60 = v59;	// L132
      u = v60;	// L133
      int16_t v61 = kn;	// L134
      kc = v61;	// L135
      bool v62 = kone;	// L136
      klast = v62;	// L137
      bool v63 = tlast;	// L138
      if (v63) {	// L143
        int16_t v64 = u;	// L144
        int32_t v65 = v64;	// L145
        int32_t v66 = v65 | 512;	// L148
        int16_t v67 = v66;	// L149
        u = v67;	// L150
        go = 0;	// L154
      } else {
        int32_t v68 = tc;	// L156
        bool v69 = v68 == 1;	// L159
        if (v69) {	// L160
          tlast = 1;	// L164
        }
        int32_t v70 = tc;	// L166
        ap_int<33> v71 = v70;	// L167
        ap_int<33> v72 = v71 - 1;	// L171
        int32_t v73 = v72;	// L172
        tc = v73;	// L173
      }
    } else {
      int16_t v74 = kc;	// L176
      int32_t v75 = v74;	// L177
      bool v76 = v75 == 1;	// L180
      if (v76) {	// L181
        klast = 1;	// L185
      }
      int16_t v77 = kc;	// L187
      ap_int<33> v78 = v77;	// L188
      ap_int<33> v79 = v78 - 1;	// L192
      uint16_t v80 = v79;	// L193
      kc = v80;	// L194
    }
    int16_t v81 = u;	// L196
    v0.write(v81);	// L197
  }
}


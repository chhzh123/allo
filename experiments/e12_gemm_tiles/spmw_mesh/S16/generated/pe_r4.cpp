
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
void pe_r4_0(
  hls::stream< int32_t >& v0,
  hls::stream< int16_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int16_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int32_t >& v6
) {	// L2
  int32_t v7 = v5.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v7;	// L5
  int32_t v9 = v6.read();	// L6
  int32_t _st__pid1;	// L7
  _st__pid1 = v9;	// L8
  int32_t v11 = _st__pid0;	// L9
  int32_t row;	// L10
  row = v11;	// L11
  int32_t v13 = _st__pid1;	// L12
  int32_t _col;	// L13
  _col = v13;	// L14
  int32_t v15 = row;	// L15
  uint8_t v16 = v15;	// L16
  uint8_t above;	// L17
  above = v16;	// L18
  int32_t acc;	// L21
  acc = 0;	// L22
  uint8_t rem;	// L26
  rem = 0;	// L27
  bool tail;	// L31
  tail = 0;	// L32
  bool go;	// L36
  go = 1;	// L37
  while (true) {	// L38
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v22 = go;	// L39
    if (!(v22)) break;
    bool v23 = tail;	// L46
    if (v23) {	// L51
      v0.write(0);	// L54
      int8_t v24 = rem;	// L55
      int32_t v25 = v24;	// L56
      bool v26 = v25 == 1;	// L59
      if (v26) {	// L60
        go = 0;	// L64
      }
      int8_t v27 = rem;	// L66
      ap_int<33> v28 = v27;	// L67
      ap_int<33> v29 = v28 - 1;	// L71
      uint8_t v30 = v29;	// L72
      rem = v30;	// L73
    } else {
      int16_t v31 = v4.read();	// L75
      int16_t t;	// L76
      t = v31;	// L77
      int8_t v33 = v2.read();	// L78
      int8_t b;	// L79
      b = v33;	// L80
      int16_t v35 = t;	// L81
      v1.write(v35);	// L82
      int8_t v36 = b;	// L83
      v3.write(v36);	// L84
      int16_t v37 = t;	// L85
      int32_t v38 = v37;	// L86
      int32_t v39 = v38 & 255;	// L89
      int32_t v40 = v39 ^ 128;	// L92
      ap_int<33> v41 = v40;	// L93
      ap_int<33> v42 = v41 - 128;	// L97
      int8_t v43 = v42;	// L98
      int8_t a;	// L99
      a = v43;	// L100
      int32_t v45 = acc;	// L101
      int8_t v46 = a;	// L102
      int8_t v47 = b;	// L103
      int16_t v48 = v46;	// L104
      int16_t v49 = v47;	// L105
      int16_t v50 = v48 * v49;	// L106
      ap_int<33> v51 = v45;	// L107
      ap_int<33> v52 = v50;	// L108
      ap_int<33> v53 = v51 + v52;	// L109
      int32_t v54 = v53;	// L110
      int32_t s;	// L111
      s = v54;	// L112
      int16_t v56 = t;	// L113
      int16_t v57 = v56 >> 8;	// L117
      int32_t v58 = v57;	// L118
      int32_t v59 = v58 & 1;	// L121
      bool v60 = v59 != 0;	// L124
      if (v60) {	// L125
        int32_t v61 = s;	// L126
        v0.write(v61);	// L127
        acc = 0;	// L130
        int8_t v62 = above;	// L131
        rem = v62;	// L132
      } else {
        int32_t v63 = s;	// L134
        acc = v63;	// L135
        int8_t v64 = rem;	// L136
        int32_t v65 = v64;	// L137
        bool v66 = v65 != 0;	// L140
        if (v66) {	// L141
          v0.write(0);	// L144
          int8_t v67 = rem;	// L145
          ap_int<33> v68 = v67;	// L146
          ap_int<33> v69 = v68 - 1;	// L150
          uint8_t v70 = v69;	// L151
          rem = v70;	// L152
        }
      }
      int16_t v71 = t;	// L155
      int16_t v72 = v71 >> 9;	// L159
      int32_t v73 = v72;	// L160
      int32_t v74 = v73 & 1;	// L163
      bool v75 = v74 != 0;	// L166
      if (v75) {	// L167
        tail = 1;	// L171
        int8_t v76 = above;	// L172
        int32_t v77 = v76;	// L173
        bool v78 = v77 == 0;	// L176
        if (v78) {	// L177
          go = 0;	// L181
        }
      }
    }
  }
}



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
void wreq8_r0_0(
  hls::stream< int64_t >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int64_t >& v2,
  hls::stream< int64_t >& v3
) {	// L2
  int32_t addr;	// L5
  addr = 0;	// L6
  int32_t left;	// L9
  left = 0;	// L10
  bool final;	// L14
  final = 0;	// L15
  bool owed;	// L19
  owed = 0;	// L20
  int32_t st;	// L23
  st = 0;	// L24
  bool go;	// L28
  go = 1;	// L29
  while (true) {	// L30
    #pragma HLS pipeline II=1 style=flp
    bool v10 = go;	// L31
    if (!(v10)) break;
    int32_t v11 = st;	// L38
    bool v12 = v11 == 0;	// L41
    if (v12) {	// L42
      int64_t v13 = v3.read();	// L43
      int64_t c;	// L44
      c = v13;	// L45
      int64_t v15 = c;	// L46
      int64_t v16 = v15 & 1073741823;	// L50
      int32_t v17 = v16;	// L51
      addr = v17;	// L52
      int64_t v18 = c;	// L53
      int64_t v19 = v18 >> 32;	// L57
      int64_t v20 = v19 & 1073741823;	// L61
      int32_t v21 = v20;	// L62
      left = v21;	// L63
      int64_t v22 = c;	// L64
      int64_t v23 = v22 >> 62;	// L68
      int64_t v24 = v23 & 1;	// L72
      bool v25 = v24;	// L73
      final = v25;	// L74
      st = 1;	// L77
    } else {
      int32_t v26 = st;	// L79
      bool v27 = v26 == 1;	// L82
      if (v27) {	// L83
        int32_t v28 = left;	// L84
        int32_t n;	// L85
        n = v28;	// L86
        int32_t v30 = left;	// L87
        bool v31 = v30 > 256;	// L90
        if (v31) {	// L91
          n = 256;	// L94
        }
        int32_t v32 = n;	// L96
        int64_t v33 = v32;	// L97
        int64_t nn;	// L98
        nn = v33;	// L99
        int32_t v35 = addr;	// L100
        int64_t v36 = v35;	// L101
        int64_t a64;	// L102
        a64 = v36;	// L103
        int64_t v38 = nn;	// L104
        int64_t v39 = v38 << 32;	// L108
        int64_t v40 = a64;	// L109
        int64_t v41 = v39 | v40;	// L110
        v2.write(v41);	// L111
        int32_t v42 = addr;	// L112
        int32_t v43 = n;	// L113
        int32_t v44 = v43 << 3;	// L116
        ap_int<33> v45 = v42;	// L117
        ap_int<33> v46 = v44;	// L118
        ap_int<33> v47 = v45 + v46;	// L119
        int32_t v48 = v47;	// L120
        addr = v48;	// L121
        int32_t v49 = left;	// L122
        int32_t v50 = n;	// L123
        ap_int<33> v51 = v49;	// L124
        ap_int<33> v52 = v50;	// L125
        ap_int<33> v53 = v51 - v52;	// L126
        int32_t v54 = v53;	// L127
        left = v54;	// L128
        st = 2;	// L131
      } else {
        int32_t v55 = st;	// L133
        bool v56 = v55 == 2;	// L136
        if (v56) {	// L137
          bool v57 = owed;	// L138
          if (v57) {	// L143
            int64_t v58 = v1.read();	// L144
            int64_t ack;	// L145
            ack = v58;	// L146
          }
          owed = 1;	// L151
          st = 0;	// L154
          int32_t v60 = left;	// L155
          bool v61 = v60 != 0;	// L158
          if (v61) {	// L159
            st = 1;	// L162
          } else {
            bool v62 = final;	// L164
            if (v62) {	// L169
              st = 3;	// L172
            }
          }
        } else {
          int64_t v63 = v1.read();	// L176
          int64_t ack2;	// L177
          ack2 = v63;	// L178
          v0.write(1);	// L181
          go = 0;	// L185
        }
      }
    }
  }
}


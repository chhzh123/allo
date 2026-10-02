
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
void wreq4_r0_0(
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
      int64_t v21 = v20 >> 1;	// L65
      int32_t v22 = v21;	// L66
      left = v22;	// L67
      int64_t v23 = c;	// L68
      int64_t v24 = v23 >> 62;	// L72
      int64_t v25 = v24 & 1;	// L76
      bool v26 = v25;	// L77
      final = v26;	// L78
      st = 1;	// L81
    } else {
      int32_t v27 = st;	// L83
      bool v28 = v27 == 1;	// L86
      if (v28) {	// L87
        int32_t v29 = left;	// L88
        int32_t n;	// L89
        n = v29;	// L90
        int32_t v31 = left;	// L91
        bool v32 = v31 > 256;	// L94
        if (v32) {	// L95
          n = 256;	// L98
        }
        int32_t v33 = n;	// L100
        int64_t v34 = v33;	// L101
        int64_t nn;	// L102
        nn = v34;	// L103
        int32_t v36 = addr;	// L104
        int64_t v37 = v36;	// L105
        int64_t a64;	// L106
        a64 = v37;	// L107
        int64_t v39 = nn;	// L108
        int64_t v40 = v39 << 32;	// L112
        int64_t v41 = a64;	// L113
        int64_t v42 = v40 | v41;	// L114
        v2.write(v42);	// L115
        int32_t v43 = addr;	// L116
        int32_t v44 = n;	// L117
        int32_t v45 = v44 << 3;	// L120
        ap_int<33> v46 = v43;	// L121
        ap_int<33> v47 = v45;	// L122
        ap_int<33> v48 = v46 + v47;	// L123
        int32_t v49 = v48;	// L124
        addr = v49;	// L125
        int32_t v50 = left;	// L126
        int32_t v51 = n;	// L127
        ap_int<33> v52 = v50;	// L128
        ap_int<33> v53 = v51;	// L129
        ap_int<33> v54 = v52 - v53;	// L130
        int32_t v55 = v54;	// L131
        left = v55;	// L132
        st = 2;	// L135
      } else {
        int32_t v56 = st;	// L137
        bool v57 = v56 == 2;	// L140
        if (v57) {	// L141
          bool v58 = owed;	// L142
          if (v58) {	// L147
            int64_t v59 = v1.read();	// L148
            int64_t ack;	// L149
            ack = v59;	// L150
          }
          owed = 1;	// L155
          st = 0;	// L158
          int32_t v61 = left;	// L159
          bool v62 = v61 != 0;	// L162
          if (v62) {	// L163
            st = 1;	// L166
          } else {
            bool v63 = final;	// L168
            if (v63) {	// L173
              st = 3;	// L176
            }
          }
        } else {
          int64_t v64 = v1.read();	// L180
          int64_t ack2;	// L181
          ack2 = v64;	// L182
          v0.write(1);	// L185
          go = 0;	// L189
        }
      }
    }
  }
}


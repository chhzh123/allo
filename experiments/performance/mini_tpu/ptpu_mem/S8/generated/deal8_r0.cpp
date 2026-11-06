
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void deal8_r0_0(
  hls::stream< hls::vector< int8_t, 8 > >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int64_t >& v2,
  hls::stream< int64_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< hls::vector< int8_t, 8 > >& v5
) {	// L2
  int32_t kind;	// L5
  kind = 0;	// L6
  int16_t n;	// L10
  n = 0;	// L11
  bool last;	// L15
  last = 0;	// L16
  bool more;	// L20
  more = 0;	// L21
  bool islast;	// L25
  islast = 0;	// L26
  int32_t st;	// L29
  st = 0;	// L30
  int64_t lo;	// L34
  lo = 0;	// L35
  bool part;	// L39
  part = 0;	// L40
  bool endp;	// L44
  endp = 0;	// L45
  bool go;	// L49
  go = 1;	// L50
  while (true) {	// L51
    #pragma HLS pipeline II=1 style=flp
    bool v16 = go;	// L52
    if (!(v16)) break;
    bool ends;	// L62
    ends = 0;	// L63
    int32_t v18 = st;	// L64
    bool v19 = v18 == 0;	// L67
    if (v19) {	// L68
      int32_t v20 = v4.read();	// L69
      int32_t t0;	// L70
      t0 = v20;	// L71
      int32_t v22 = t0;	// L72
      int32_t v23 = v22 & 3;	// L75
      kind = v23;	// L76
      int32_t v24 = t0;	// L77
      int32_t v25 = v24 >> 2;	// L80
      int32_t v26 = v25 & 1;	// L83
      bool v27 = v26;	// L84
      last = v27;	// L85
      int32_t v28 = t0;	// L86
      int32_t v29 = v28 >> 8;	// L89
      int32_t v30 = v29 & 255;	// L92
      int16_t v31 = v30;	// L93
      n = v31;	// L94
      int32_t v32 = t0;	// L95
      int32_t v33 = v32 >> 3;	// L98
      int32_t v34 = v33 & 1;	// L101
      bool v35 = v34;	// L102
      more = v35;	// L103
      islast = 0;	// L107
      st = 1;	// L110
    } else {
      int64_t v36 = v3.read();	// L112
      int64_t beat;	// L113
      beat = v36;	// L114
      bool v38 = islast;	// L115
      bool lastb;	// L116
      lastb = v38;	// L117
      islast = 0;	// L121
      int16_t v40 = n;	// L122
      int32_t v41 = v40;	// L123
      bool v42 = v41 == 2;	// L126
      if (v42) {	// L127
        islast = 1;	// L131
      }
      int16_t v43 = n;	// L133
      ap_int<33> v44 = v43;	// L134
      ap_int<33> v45 = v44 - 1;	// L138
      int16_t v46 = v45;	// L139
      n = v46;	// L140
      int32_t v47 = kind;	// L141
      bool v48 = v47 == 0;	// L144
      if (v48) {	// L145
        int64_t v49 = beat;	// L146
        v2.write(v49);	// L147
        bool v50 = lastb;	// L148
        ends = v50;	// L149
      } else {
        int32_t v51 = kind;	// L151
        bool v52 = v51 == 1;	// L154
        if (v52) {	// L155
          int64_t v53 = beat;	// L156
          v1.write(v53);	// L157
          bool v54 = lastb;	// L158
          ends = v54;	// L159
        } else {
          int8_t wd[8];	// L164
          for (int v56 = 0; v56 < 8; v56++) {	// L165
            wd[v56] = 0;	// L165
          }
          int64_t v57 = beat;	// L166
          int64_t v58 = v57 & 255;	// L174
          int64_t v59 = v58 ^ 128;	// L178
          ap_int<65> v60 = v59;	// L179
          ap_int<65> v61 = v60 - 128;	// L183
          int8_t v62 = v61;	// L184
          wd[0] = v62;	// L185
          int64_t v63 = beat;	// L186
          int64_t v64 = v63 >> 8;	// L190
          int64_t v65 = v64 & 255;	// L194
          int64_t v66 = v65 ^ 128;	// L198
          ap_int<65> v67 = v66;	// L199
          ap_int<65> v68 = v67 - 128;	// L203
          int8_t v69 = v68;	// L204
          wd[1] = v69;	// L205
          int64_t v70 = beat;	// L206
          int64_t v71 = v70 >> 16;	// L210
          int64_t v72 = v71 & 255;	// L214
          int64_t v73 = v72 ^ 128;	// L218
          ap_int<65> v74 = v73;	// L219
          ap_int<65> v75 = v74 - 128;	// L223
          int8_t v76 = v75;	// L224
          wd[2] = v76;	// L225
          int64_t v77 = beat;	// L226
          int64_t v78 = v77 >> 24;	// L230
          int64_t v79 = v78 & 255;	// L234
          int64_t v80 = v79 ^ 128;	// L238
          ap_int<65> v81 = v80;	// L239
          ap_int<65> v82 = v81 - 128;	// L243
          int8_t v83 = v82;	// L244
          wd[3] = v83;	// L245
          int64_t v84 = beat;	// L246
          int64_t v85 = v84 >> 32;	// L250
          int64_t v86 = v85 & 255;	// L254
          int64_t v87 = v86 ^ 128;	// L258
          ap_int<65> v88 = v87;	// L259
          ap_int<65> v89 = v88 - 128;	// L263
          int8_t v90 = v89;	// L264
          wd[4] = v90;	// L265
          int64_t v91 = beat;	// L266
          int64_t v92 = v91 >> 40;	// L270
          int64_t v93 = v92 & 255;	// L274
          int64_t v94 = v93 ^ 128;	// L278
          ap_int<65> v95 = v94;	// L279
          ap_int<65> v96 = v95 - 128;	// L283
          int8_t v97 = v96;	// L284
          wd[5] = v97;	// L285
          int64_t v98 = beat;	// L286
          int64_t v99 = v98 >> 48;	// L290
          int64_t v100 = v99 & 255;	// L294
          int64_t v101 = v100 ^ 128;	// L298
          ap_int<65> v102 = v101;	// L299
          ap_int<65> v103 = v102 - 128;	// L303
          int8_t v104 = v103;	// L304
          wd[6] = v104;	// L305
          int64_t v105 = beat;	// L306
          int64_t v106 = v105 >> 56;	// L310
          int64_t v107 = v106 & 255;	// L314
          int64_t v108 = v107 ^ 128;	// L318
          ap_int<65> v109 = v108;	// L319
          ap_int<65> v110 = v109 - 128;	// L323
          int8_t v111 = v110;	// L324
          wd[7] = v111;	// L325
          int32_t v112 = kind;	// L326
          bool v113 = v112 == 3;	// L329
          if (v113) {	// L330
            {
              hls::vector< int8_t, 8 > _vec;
              for (int _iv0 = 0; _iv0 < 8; ++_iv0) {
                _vec[_iv0] = wd[_iv0];
              }
              v0.write(_vec);
            }	// L331
          } else {
            {
              hls::vector< int8_t, 8 > _vec;
              for (int _iv0 = 0; _iv0 < 8; ++_iv0) {
                _vec[_iv0] = wd[_iv0];
              }
              v5.write(_vec);
            }	// L333
          }
          bool v114 = lastb;	// L335
          ends = v114;	// L336
        }
      }
    }
    bool v115 = ends;	// L340
    if (v115) {	// L345
      bool v116 = more;	// L346
      if (v116) {	// L351
        int32_t v117 = v4.read();	// L352
        int32_t t1;	// L353
        t1 = v117;	// L354
        int32_t v119 = t1;	// L355
        int32_t v120 = v119 & 3;	// L358
        kind = v120;	// L359
        int32_t v121 = t1;	// L360
        int32_t v122 = v121 >> 2;	// L363
        int32_t v123 = v122 & 1;	// L366
        bool v124 = v123;	// L367
        last = v124;	// L368
        int32_t v125 = t1;	// L369
        int32_t v126 = v125 >> 8;	// L372
        int32_t v127 = v126 & 255;	// L375
        int16_t v128 = v127;	// L376
        n = v128;	// L377
        int32_t v129 = t1;	// L378
        int32_t v130 = v129 >> 3;	// L381
        int32_t v131 = v130 & 1;	// L384
        bool v132 = v131;	// L385
        more = v132;	// L386
        islast = 0;	// L390
        st = 1;	// L393
      } else {
        bool v133 = last;	// L395
        if (v133) {	// L400
          go = 0;	// L404
        } else {
          st = 0;	// L408
        }
      }
    }
  }
}


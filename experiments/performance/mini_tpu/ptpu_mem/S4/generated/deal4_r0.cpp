
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
void deal4_r0_0(
  hls::stream< hls::vector< int8_t, 4 > >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int64_t >& v2,
  hls::stream< int64_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< hls::vector< int8_t, 4 > >& v5
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
      int32_t v36 = st;	// L112
      bool v37 = v36 == 2;	// L115
      if (v37) {	// L116
        int8_t wh[4];	// L120
        for (int v39 = 0; v39 < 4; v39++) {	// L121
          wh[v39] = 0;	// L121
        }
        int64_t v40 = lo;	// L122
        int64_t v41 = v40 & 255;	// L130
        int64_t v42 = v41 ^ 128;	// L134
        ap_int<65> v43 = v42;	// L135
        ap_int<65> v44 = v43 - 128;	// L139
        int8_t v45 = v44;	// L140
        wh[0] = v45;	// L141
        int64_t v46 = lo;	// L142
        int64_t v47 = v46 >> 8;	// L146
        int64_t v48 = v47 & 255;	// L150
        int64_t v49 = v48 ^ 128;	// L154
        ap_int<65> v50 = v49;	// L155
        ap_int<65> v51 = v50 - 128;	// L159
        int8_t v52 = v51;	// L160
        wh[1] = v52;	// L161
        int64_t v53 = lo;	// L162
        int64_t v54 = v53 >> 16;	// L166
        int64_t v55 = v54 & 255;	// L170
        int64_t v56 = v55 ^ 128;	// L174
        ap_int<65> v57 = v56;	// L175
        ap_int<65> v58 = v57 - 128;	// L179
        int8_t v59 = v58;	// L180
        wh[2] = v59;	// L181
        int64_t v60 = lo;	// L182
        int64_t v61 = v60 >> 24;	// L186
        int64_t v62 = v61 & 255;	// L190
        int64_t v63 = v62 ^ 128;	// L194
        ap_int<65> v64 = v63;	// L195
        ap_int<65> v65 = v64 - 128;	// L199
        int8_t v66 = v65;	// L200
        wh[3] = v66;	// L201
        int32_t v67 = kind;	// L202
        bool v68 = v67 == 3;	// L205
        if (v68) {	// L206
          {
            hls::vector< int8_t, 4 > _vec;
            for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
              _vec[_iv0] = wh[_iv0];
            }
            v0.write(_vec);
          }	// L207
        } else {
          {
            hls::vector< int8_t, 4 > _vec;
            for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
              _vec[_iv0] = wh[_iv0];
            }
            v5.write(_vec);
          }	// L209
        }
        bool v69 = endp;	// L211
        ends = v69;	// L212
        st = 1;	// L215
      } else {
        int64_t v70 = v3.read();	// L217
        int64_t beat;	// L218
        beat = v70;	// L219
        bool v72 = islast;	// L220
        bool lastb;	// L221
        lastb = v72;	// L222
        islast = 0;	// L226
        int16_t v74 = n;	// L227
        int32_t v75 = v74;	// L228
        bool v76 = v75 == 2;	// L231
        if (v76) {	// L232
          islast = 1;	// L236
        }
        int16_t v77 = n;	// L238
        ap_int<33> v78 = v77;	// L239
        ap_int<33> v79 = v78 - 1;	// L243
        int16_t v80 = v79;	// L244
        n = v80;	// L245
        int32_t v81 = kind;	// L246
        bool v82 = v81 == 0;	// L249
        if (v82) {	// L250
          int64_t v83 = beat;	// L251
          v2.write(v83);	// L252
          bool v84 = lastb;	// L253
          ends = v84;	// L254
        } else {
          int32_t v85 = kind;	// L256
          bool v86 = v85 == 1;	// L259
          if (v86) {	// L260
            int64_t v87 = beat;	// L261
            v1.write(v87);	// L262
            bool v88 = lastb;	// L263
            ends = v88;	// L264
          } else {
            int8_t wd[4];	// L269
            for (int v90 = 0; v90 < 4; v90++) {	// L270
              wd[v90] = 0;	// L270
            }
            int64_t v91 = beat;	// L271
            int64_t v92 = v91 & 255;	// L279
            int64_t v93 = v92 ^ 128;	// L283
            ap_int<65> v94 = v93;	// L284
            ap_int<65> v95 = v94 - 128;	// L288
            int8_t v96 = v95;	// L289
            wd[0] = v96;	// L290
            int64_t v97 = beat;	// L291
            int64_t v98 = v97 >> 8;	// L295
            int64_t v99 = v98 & 255;	// L299
            int64_t v100 = v99 ^ 128;	// L303
            ap_int<65> v101 = v100;	// L304
            ap_int<65> v102 = v101 - 128;	// L308
            int8_t v103 = v102;	// L309
            wd[1] = v103;	// L310
            int64_t v104 = beat;	// L311
            int64_t v105 = v104 >> 16;	// L315
            int64_t v106 = v105 & 255;	// L319
            int64_t v107 = v106 ^ 128;	// L323
            ap_int<65> v108 = v107;	// L324
            ap_int<65> v109 = v108 - 128;	// L328
            int8_t v110 = v109;	// L329
            wd[2] = v110;	// L330
            int64_t v111 = beat;	// L331
            int64_t v112 = v111 >> 24;	// L335
            int64_t v113 = v112 & 255;	// L339
            int64_t v114 = v113 ^ 128;	// L343
            ap_int<65> v115 = v114;	// L344
            ap_int<65> v116 = v115 - 128;	// L348
            int8_t v117 = v116;	// L349
            wd[3] = v117;	// L350
            int32_t v118 = kind;	// L351
            bool v119 = v118 == 3;	// L354
            if (v119) {	// L355
              {
                hls::vector< int8_t, 4 > _vec;
                for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
                  _vec[_iv0] = wd[_iv0];
                }
                v0.write(_vec);
              }	// L356
            } else {
              {
                hls::vector< int8_t, 4 > _vec;
                for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
                  _vec[_iv0] = wd[_iv0];
                }
                v5.write(_vec);
              }	// L358
            }
            int64_t v120 = beat;	// L360
            int64_t v121 = v120 >> 32;	// L364
            lo = v121;	// L365
            bool v122 = lastb;	// L366
            endp = v122;	// L367
            st = 2;	// L370
          }
        }
      }
    }
    bool v123 = ends;	// L375
    if (v123) {	// L380
      bool v124 = more;	// L381
      if (v124) {	// L386
        int32_t v125 = v4.read();	// L387
        int32_t t1;	// L388
        t1 = v125;	// L389
        int32_t v127 = t1;	// L390
        int32_t v128 = v127 & 3;	// L393
        kind = v128;	// L394
        int32_t v129 = t1;	// L395
        int32_t v130 = v129 >> 2;	// L398
        int32_t v131 = v130 & 1;	// L401
        bool v132 = v131;	// L402
        last = v132;	// L403
        int32_t v133 = t1;	// L404
        int32_t v134 = v133 >> 8;	// L407
        int32_t v135 = v134 & 255;	// L410
        int16_t v136 = v135;	// L411
        n = v136;	// L412
        int32_t v137 = t1;	// L413
        int32_t v138 = v137 >> 3;	// L416
        int32_t v139 = v138 & 1;	// L419
        bool v140 = v139;	// L420
        more = v140;	// L421
        islast = 0;	// L425
        st = 1;	// L428
      } else {
        bool v141 = last;	// L430
        if (v141) {	// L435
          go = 0;	// L439
        } else {
          st = 0;	// L443
        }
      }
    }
  }
}


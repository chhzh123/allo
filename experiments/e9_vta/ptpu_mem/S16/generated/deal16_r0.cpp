
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
void deal16_r0_0(
  hls::stream< hls::vector< int8_t, 16 > >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int64_t >& v2,
  hls::stream< int64_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< hls::vector< int8_t, 16 > >& v5
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
          bool v55 = part;	// L161
          int32_t v56 = v55;	// L162
          bool v57 = v56 == 0;	// L165
          if (v57) {	// L166
            int64_t v58 = beat;	// L167
            lo = v58;	// L168
            part = 1;	// L172
          } else {
            int8_t wd[16];	// L177
            for (int v60 = 0; v60 < 16; v60++) {	// L178
              wd[v60] = 0;	// L178
            }
            int64_t v61 = lo;	// L179
            int64_t v62 = v61 & 255;	// L187
            int64_t v63 = v62 ^ 128;	// L191
            ap_int<65> v64 = v63;	// L192
            ap_int<65> v65 = v64 - 128;	// L196
            int8_t v66 = v65;	// L197
            wd[0] = v66;	// L198
            int64_t v67 = lo;	// L199
            int64_t v68 = v67 >> 8;	// L203
            int64_t v69 = v68 & 255;	// L207
            int64_t v70 = v69 ^ 128;	// L211
            ap_int<65> v71 = v70;	// L212
            ap_int<65> v72 = v71 - 128;	// L216
            int8_t v73 = v72;	// L217
            wd[1] = v73;	// L218
            int64_t v74 = lo;	// L219
            int64_t v75 = v74 >> 16;	// L223
            int64_t v76 = v75 & 255;	// L227
            int64_t v77 = v76 ^ 128;	// L231
            ap_int<65> v78 = v77;	// L232
            ap_int<65> v79 = v78 - 128;	// L236
            int8_t v80 = v79;	// L237
            wd[2] = v80;	// L238
            int64_t v81 = lo;	// L239
            int64_t v82 = v81 >> 24;	// L243
            int64_t v83 = v82 & 255;	// L247
            int64_t v84 = v83 ^ 128;	// L251
            ap_int<65> v85 = v84;	// L252
            ap_int<65> v86 = v85 - 128;	// L256
            int8_t v87 = v86;	// L257
            wd[3] = v87;	// L258
            int64_t v88 = lo;	// L259
            int64_t v89 = v88 >> 32;	// L263
            int64_t v90 = v89 & 255;	// L267
            int64_t v91 = v90 ^ 128;	// L271
            ap_int<65> v92 = v91;	// L272
            ap_int<65> v93 = v92 - 128;	// L276
            int8_t v94 = v93;	// L277
            wd[4] = v94;	// L278
            int64_t v95 = lo;	// L279
            int64_t v96 = v95 >> 40;	// L283
            int64_t v97 = v96 & 255;	// L287
            int64_t v98 = v97 ^ 128;	// L291
            ap_int<65> v99 = v98;	// L292
            ap_int<65> v100 = v99 - 128;	// L296
            int8_t v101 = v100;	// L297
            wd[5] = v101;	// L298
            int64_t v102 = lo;	// L299
            int64_t v103 = v102 >> 48;	// L303
            int64_t v104 = v103 & 255;	// L307
            int64_t v105 = v104 ^ 128;	// L311
            ap_int<65> v106 = v105;	// L312
            ap_int<65> v107 = v106 - 128;	// L316
            int8_t v108 = v107;	// L317
            wd[6] = v108;	// L318
            int64_t v109 = lo;	// L319
            int64_t v110 = v109 >> 56;	// L323
            int64_t v111 = v110 & 255;	// L327
            int64_t v112 = v111 ^ 128;	// L331
            ap_int<65> v113 = v112;	// L332
            ap_int<65> v114 = v113 - 128;	// L336
            int8_t v115 = v114;	// L337
            wd[7] = v115;	// L338
            int64_t v116 = beat;	// L339
            int64_t v117 = v116 & 255;	// L347
            int64_t v118 = v117 ^ 128;	// L351
            ap_int<65> v119 = v118;	// L352
            ap_int<65> v120 = v119 - 128;	// L356
            int8_t v121 = v120;	// L357
            wd[8] = v121;	// L358
            int64_t v122 = beat;	// L359
            int64_t v123 = v122 >> 8;	// L363
            int64_t v124 = v123 & 255;	// L367
            int64_t v125 = v124 ^ 128;	// L371
            ap_int<65> v126 = v125;	// L372
            ap_int<65> v127 = v126 - 128;	// L376
            int8_t v128 = v127;	// L377
            wd[9] = v128;	// L378
            int64_t v129 = beat;	// L379
            int64_t v130 = v129 >> 16;	// L383
            int64_t v131 = v130 & 255;	// L387
            int64_t v132 = v131 ^ 128;	// L391
            ap_int<65> v133 = v132;	// L392
            ap_int<65> v134 = v133 - 128;	// L396
            int8_t v135 = v134;	// L397
            wd[10] = v135;	// L398
            int64_t v136 = beat;	// L399
            int64_t v137 = v136 >> 24;	// L403
            int64_t v138 = v137 & 255;	// L407
            int64_t v139 = v138 ^ 128;	// L411
            ap_int<65> v140 = v139;	// L412
            ap_int<65> v141 = v140 - 128;	// L416
            int8_t v142 = v141;	// L417
            wd[11] = v142;	// L418
            int64_t v143 = beat;	// L419
            int64_t v144 = v143 >> 32;	// L423
            int64_t v145 = v144 & 255;	// L427
            int64_t v146 = v145 ^ 128;	// L431
            ap_int<65> v147 = v146;	// L432
            ap_int<65> v148 = v147 - 128;	// L436
            int8_t v149 = v148;	// L437
            wd[12] = v149;	// L438
            int64_t v150 = beat;	// L439
            int64_t v151 = v150 >> 40;	// L443
            int64_t v152 = v151 & 255;	// L447
            int64_t v153 = v152 ^ 128;	// L451
            ap_int<65> v154 = v153;	// L452
            ap_int<65> v155 = v154 - 128;	// L456
            int8_t v156 = v155;	// L457
            wd[13] = v156;	// L458
            int64_t v157 = beat;	// L459
            int64_t v158 = v157 >> 48;	// L463
            int64_t v159 = v158 & 255;	// L467
            int64_t v160 = v159 ^ 128;	// L471
            ap_int<65> v161 = v160;	// L472
            ap_int<65> v162 = v161 - 128;	// L476
            int8_t v163 = v162;	// L477
            wd[14] = v163;	// L478
            int64_t v164 = beat;	// L479
            int64_t v165 = v164 >> 56;	// L483
            int64_t v166 = v165 & 255;	// L487
            int64_t v167 = v166 ^ 128;	// L491
            ap_int<65> v168 = v167;	// L492
            ap_int<65> v169 = v168 - 128;	// L496
            int8_t v170 = v169;	// L497
            wd[15] = v170;	// L498
            int32_t v171 = kind;	// L499
            bool v172 = v171 == 3;	// L502
            if (v172) {	// L503
              {
                hls::vector< int8_t, 16 > _vec;
                for (int _iv0 = 0; _iv0 < 16; ++_iv0) {
                  _vec[_iv0] = wd[_iv0];
                }
                v0.write(_vec);
              }	// L504
            } else {
              {
                hls::vector< int8_t, 16 > _vec;
                for (int _iv0 = 0; _iv0 < 16; ++_iv0) {
                  _vec[_iv0] = wd[_iv0];
                }
                v5.write(_vec);
              }	// L506
            }
            part = 0;	// L511
          }
          bool v173 = lastb;	// L513
          ends = v173;	// L514
        }
      }
    }
    bool v174 = ends;	// L518
    if (v174) {	// L523
      bool v175 = more;	// L524
      if (v175) {	// L529
        int32_t v176 = v4.read();	// L530
        int32_t t1;	// L531
        t1 = v176;	// L532
        int32_t v178 = t1;	// L533
        int32_t v179 = v178 & 3;	// L536
        kind = v179;	// L537
        int32_t v180 = t1;	// L538
        int32_t v181 = v180 >> 2;	// L541
        int32_t v182 = v181 & 1;	// L544
        bool v183 = v182;	// L545
        last = v183;	// L546
        int32_t v184 = t1;	// L547
        int32_t v185 = v184 >> 8;	// L550
        int32_t v186 = v185 & 255;	// L553
        int16_t v187 = v186;	// L554
        n = v187;	// L555
        int32_t v188 = t1;	// L556
        int32_t v189 = v188 >> 3;	// L559
        int32_t v190 = v189 & 1;	// L562
        bool v191 = v190;	// L563
        more = v191;	// L564
        islast = 0;	// L568
        st = 1;	// L571
      } else {
        bool v192 = last;	// L573
        if (v192) {	// L578
          go = 0;	// L582
        } else {
          st = 0;	// L586
        }
      }
    }
  }
}


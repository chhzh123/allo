
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
using namespace std;
int32_t _ix0[1][1] = {0};	// L3
void bfly_up_in_load_0(
  float v0[8][2],
  hls::stream< hls::vector< float, 2 > >& v1
) {	// L4
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  // placeholder for const int32_t _ix0	// L6
  l_S__t_0__t: for (int _t = 0; _t < 1; _t++) {	// L7
    float _blk[2];	// L8
    for (int v5 = 0; v5 < 2; v5++) {	// L9
      _blk[v5] = (float)0.000000;	// L9
    }
    l_S__b0_0__b0: for (int _b0 = 0; _b0 < 2; _b0++) {	// L10
      int32_t v7 = _ix0[_t][0];	// L11
      int v8 = v7;	// L12
      float v9 = v0[v8][_b0];	// L13
      _blk[_b0] = v9;	// L14
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk[_iv0];
      }
      v1.write(_vec);
    }	// L16
  }
}

int32_t _ix0_0[1][1] = {2};	// L20
void bfly_up_in_load_1(
  float v10[8][2],
  hls::stream< hls::vector< float, 2 > >& v11
) {	// L21
  #pragma HLS array_partition variable=v10 complete dim=1
  #pragma HLS array_partition variable=v10 complete dim=2

  // placeholder for const int32_t _ix0_0	// L23
  l_S__t_0__t1: for (int _t1 = 0; _t1 < 1; _t1++) {	// L24
    float _blk1[2];	// L25
    for (int v15 = 0; v15 < 2; v15++) {	// L26
      _blk1[v15] = (float)0.000000;	// L26
    }
    l_S__b0_0__b01: for (int _b01 = 0; _b01 < 2; _b01++) {	// L27
      int32_t v17 = _ix0_0[_t1][0];	// L28
      int v18 = v17;	// L29
      float v19 = v10[v18][_b01];	// L30
      _blk1[_b01] = v19;	// L31
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk1[_iv0];
      }
      v11.write(_vec);
    }	// L33
  }
}

int32_t _ix0_1[1][1] = {1};	// L37
void bfly_up_in_load_2(
  float v20[8][2],
  hls::stream< hls::vector< float, 2 > >& v21
) {	// L38
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  // placeholder for const int32_t _ix0_1	// L40
  l_S__t_0__t2: for (int _t2 = 0; _t2 < 1; _t2++) {	// L41
    float _blk2[2];	// L42
    for (int v25 = 0; v25 < 2; v25++) {	// L43
      _blk2[v25] = (float)0.000000;	// L43
    }
    l_S__b0_0__b02: for (int _b02 = 0; _b02 < 2; _b02++) {	// L44
      int32_t v27 = _ix0_1[_t2][0];	// L45
      int v28 = v27;	// L46
      float v29 = v20[v28][_b02];	// L47
      _blk2[_b02] = v29;	// L48
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk2[_iv0];
      }
      v21.write(_vec);
    }	// L50
  }
}

int32_t _ix0_2[1][1] = {3};	// L54
void bfly_up_in_load_3(
  float v30[8][2],
  hls::stream< hls::vector< float, 2 > >& v31
) {	// L55
  #pragma HLS array_partition variable=v30 complete dim=1
  #pragma HLS array_partition variable=v30 complete dim=2

  // placeholder for const int32_t _ix0_2	// L57
  l_S__t_0__t3: for (int _t3 = 0; _t3 < 1; _t3++) {	// L58
    float _blk3[2];	// L59
    for (int v35 = 0; v35 < 2; v35++) {	// L60
      _blk3[v35] = (float)0.000000;	// L60
    }
    l_S__b0_0__b03: for (int _b03 = 0; _b03 < 2; _b03++) {	// L61
      int32_t v37 = _ix0_2[_t3][0];	// L62
      int v38 = v37;	// L63
      float v39 = v30[v38][_b03];	// L64
      _blk3[_b03] = v39;	// L65
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk3[_iv0];
      }
      v31.write(_vec);
    }	// L67
  }
}

int32_t _ix0_3[1][1] = {4};	// L71
void bfly_lo_in_load_0(
  float v40[8][2],
  hls::stream< hls::vector< float, 2 > >& v41
) {	// L72
  #pragma HLS array_partition variable=v40 complete dim=1
  #pragma HLS array_partition variable=v40 complete dim=2

  // placeholder for const int32_t _ix0_3	// L74
  l_S__t_0__t4: for (int _t4 = 0; _t4 < 1; _t4++) {	// L75
    float _blk4[2];	// L76
    for (int v45 = 0; v45 < 2; v45++) {	// L77
      _blk4[v45] = (float)0.000000;	// L77
    }
    l_S__b0_0__b04: for (int _b04 = 0; _b04 < 2; _b04++) {	// L78
      int32_t v47 = _ix0_3[_t4][0];	// L79
      int v48 = v47;	// L80
      float v49 = v40[v48][_b04];	// L81
      _blk4[_b04] = v49;	// L82
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk4[_iv0];
      }
      v41.write(_vec);
    }	// L84
  }
}

int32_t _ix0_4[1][1] = {6};	// L88
void bfly_lo_in_load_1(
  float v50[8][2],
  hls::stream< hls::vector< float, 2 > >& v51
) {	// L89
  #pragma HLS array_partition variable=v50 complete dim=1
  #pragma HLS array_partition variable=v50 complete dim=2

  // placeholder for const int32_t _ix0_4	// L91
  l_S__t_0__t5: for (int _t5 = 0; _t5 < 1; _t5++) {	// L92
    float _blk5[2];	// L93
    for (int v55 = 0; v55 < 2; v55++) {	// L94
      _blk5[v55] = (float)0.000000;	// L94
    }
    l_S__b0_0__b05: for (int _b05 = 0; _b05 < 2; _b05++) {	// L95
      int32_t v57 = _ix0_4[_t5][0];	// L96
      int v58 = v57;	// L97
      float v59 = v50[v58][_b05];	// L98
      _blk5[_b05] = v59;	// L99
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk5[_iv0];
      }
      v51.write(_vec);
    }	// L101
  }
}

int32_t _ix0_5[1][1] = {5};	// L105
void bfly_lo_in_load_2(
  float v60[8][2],
  hls::stream< hls::vector< float, 2 > >& v61
) {	// L106
  #pragma HLS array_partition variable=v60 complete dim=1
  #pragma HLS array_partition variable=v60 complete dim=2

  // placeholder for const int32_t _ix0_5	// L108
  l_S__t_0__t6: for (int _t6 = 0; _t6 < 1; _t6++) {	// L109
    float _blk6[2];	// L110
    for (int v65 = 0; v65 < 2; v65++) {	// L111
      _blk6[v65] = (float)0.000000;	// L111
    }
    l_S__b0_0__b06: for (int _b06 = 0; _b06 < 2; _b06++) {	// L112
      int32_t v67 = _ix0_5[_t6][0];	// L113
      int v68 = v67;	// L114
      float v69 = v60[v68][_b06];	// L115
      _blk6[_b06] = v69;	// L116
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk6[_iv0];
      }
      v61.write(_vec);
    }	// L118
  }
}

int32_t _ix0_6[1][1] = {7};	// L122
void bfly_lo_in_load_3(
  float v70[8][2],
  hls::stream< hls::vector< float, 2 > >& v71
) {	// L123
  #pragma HLS array_partition variable=v70 complete dim=1
  #pragma HLS array_partition variable=v70 complete dim=2

  // placeholder for const int32_t _ix0_6	// L125
  l_S__t_0__t7: for (int _t7 = 0; _t7 < 1; _t7++) {	// L126
    float _blk7[2];	// L127
    for (int v75 = 0; v75 < 2; v75++) {	// L128
      _blk7[v75] = (float)0.000000;	// L128
    }
    l_S__b0_0__b07: for (int _b07 = 0; _b07 < 2; _b07++) {	// L129
      int32_t v77 = _ix0_6[_t7][0];	// L130
      int v78 = v77;	// L131
      float v79 = v70[v78][_b07];	// L132
      _blk7[_b07] = v79;	// L133
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk7[_iv0];
      }
      v71.write(_vec);
    }	// L135
  }
}

float _st_tw[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L139
void bfly_0_0(
  hls::stream< hls::vector< float, 2 > >& v80,
  hls::stream< hls::vector< float, 2 > >& v81,
  hls::stream< hls::vector< float, 2 > >& v82,
  hls::stream< hls::vector< float, 2 > >& v83
) {	// L140
  // placeholder for const float _st_tw	// L147
  int32_t span;	// L148
  span = 1;	// L149
  int32_t v86 = span;	// L150
  int32_t v87 = 0 % v86;	// L151
  int32_t v88 = 4 / v86;	// L152
  int64_t v89 = v87;	// L153
  int64_t v90 = v88;	// L154
  int64_t v91 = v89 * v90;	// L155
  int64_t k;	// L156
  k = v91;	// L157
  int64_t v93 = k;	// L158
  int v94 = v93;	// L159
  float v95 = _st_tw[v94][0];	// L160
  float wr;	// L161
  wr = v95;	// L162
  int64_t v97 = k;	// L163
  int v98 = v97;	// L164
  float v99 = _st_tw[v98][1];	// L165
  float wi;	// L166
  wi = v99;	// L167
  float v101[2];
  {
    hls::vector< float, 2 > _vec = v80.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v101[_iv0] = _vec[_iv0];
    }
  }	// L168
  float v102[2];
  {
    hls::vector< float, 2 > _vec = v81.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v102[_iv0] = _vec[_iv0];
    }
  }	// L169
  float v103 = wr;	// L170
  float v104 = v102[0];	// L171
  float v105 = v103 * v104;	// L172
  float v106 = wi;	// L173
  float v107 = v102[1];	// L174
  float v108 = v106 * v107;	// L175
  float v109 = v105 - v108;	// L176
  float tr;	// L177
  tr = v109;	// L178
  float v111 = wr;	// L179
  float v112 = v102[1];	// L180
  float v113 = v111 * v112;	// L181
  float v114 = wi;	// L182
  float v115 = v102[0];	// L183
  float v116 = v114 * v115;	// L184
  float v117 = v113 + v116;	// L185
  float ti;	// L186
  ti = v117;	// L187
  float u[2];	// L188
  for (int v120 = 0; v120 < 2; v120++) {	// L189
    u[v120] = (float)0.000000;	// L189
  }
  float l[2];	// L190
  for (int v122 = 0; v122 < 2; v122++) {	// L191
    l[v122] = (float)0.000000;	// L191
  }
  float v123 = v101[0];	// L192
  float v124 = tr;	// L193
  float v125 = v123 + v124;	// L194
  u[0] = v125;	// L195
  float v126 = v101[1];	// L196
  float v127 = ti;	// L197
  float v128 = v126 + v127;	// L198
  u[1] = v128;	// L199
  float v129 = v101[0];	// L200
  float v130 = tr;	// L201
  float v131 = v129 - v130;	// L202
  l[0] = v131;	// L203
  float v132 = v101[1];	// L204
  float v133 = ti;	// L205
  float v134 = v132 - v133;	// L206
  l[1] = v134;	// L207
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u[_iv0];
    }
    v82.write(_vec);
  }	// L208
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l[_iv0];
    }
    v83.write(_vec);
  }	// L209
}

float _st_tw_0[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L212
void bfly_0_1(
  hls::stream< hls::vector< float, 2 > >& v135,
  hls::stream< hls::vector< float, 2 > >& v136,
  hls::stream< hls::vector< float, 2 > >& v137,
  hls::stream< hls::vector< float, 2 > >& v138
) {	// L213
  // placeholder for const float _st_tw_0	// L219
  int32_t span1;	// L220
  span1 = 1;	// L221
  int32_t v141 = span1;	// L222
  int32_t v142 = 1 % v141;	// L223
  int32_t v143 = 4 / v141;	// L224
  int64_t v144 = v142;	// L225
  int64_t v145 = v143;	// L226
  int64_t v146 = v144 * v145;	// L227
  int64_t k1;	// L228
  k1 = v146;	// L229
  int64_t v148 = k1;	// L230
  int v149 = v148;	// L231
  float v150 = _st_tw_0[v149][0];	// L232
  float wr1;	// L233
  wr1 = v150;	// L234
  int64_t v152 = k1;	// L235
  int v153 = v152;	// L236
  float v154 = _st_tw_0[v153][1];	// L237
  float wi1;	// L238
  wi1 = v154;	// L239
  float v156[2];
  {
    hls::vector< float, 2 > _vec = v135.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v156[_iv0] = _vec[_iv0];
    }
  }	// L240
  float v157[2];
  {
    hls::vector< float, 2 > _vec = v136.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v157[_iv0] = _vec[_iv0];
    }
  }	// L241
  float v158 = wr1;	// L242
  float v159 = v157[0];	// L243
  float v160 = v158 * v159;	// L244
  float v161 = wi1;	// L245
  float v162 = v157[1];	// L246
  float v163 = v161 * v162;	// L247
  float v164 = v160 - v163;	// L248
  float tr1;	// L249
  tr1 = v164;	// L250
  float v166 = wr1;	// L251
  float v167 = v157[1];	// L252
  float v168 = v166 * v167;	// L253
  float v169 = wi1;	// L254
  float v170 = v157[0];	// L255
  float v171 = v169 * v170;	// L256
  float v172 = v168 + v171;	// L257
  float ti1;	// L258
  ti1 = v172;	// L259
  float u1[2];	// L260
  for (int v175 = 0; v175 < 2; v175++) {	// L261
    u1[v175] = (float)0.000000;	// L261
  }
  float l1[2];	// L262
  for (int v177 = 0; v177 < 2; v177++) {	// L263
    l1[v177] = (float)0.000000;	// L263
  }
  float v178 = v156[0];	// L264
  float v179 = tr1;	// L265
  float v180 = v178 + v179;	// L266
  u1[0] = v180;	// L267
  float v181 = v156[1];	// L268
  float v182 = ti1;	// L269
  float v183 = v181 + v182;	// L270
  u1[1] = v183;	// L271
  float v184 = v156[0];	// L272
  float v185 = tr1;	// L273
  float v186 = v184 - v185;	// L274
  l1[0] = v186;	// L275
  float v187 = v156[1];	// L276
  float v188 = ti1;	// L277
  float v189 = v187 - v188;	// L278
  l1[1] = v189;	// L279
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u1[_iv0];
    }
    v137.write(_vec);
  }	// L280
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l1[_iv0];
    }
    v138.write(_vec);
  }	// L281
}

float _st_tw_1[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L284
void bfly_0_2(
  hls::stream< hls::vector< float, 2 > >& v190,
  hls::stream< hls::vector< float, 2 > >& v191,
  hls::stream< hls::vector< float, 2 > >& v192,
  hls::stream< hls::vector< float, 2 > >& v193
) {	// L285
  // placeholder for const float _st_tw_1	// L292
  int32_t span2;	// L293
  span2 = 1;	// L294
  int32_t v196 = span2;	// L295
  int32_t v197 = 2 % v196;	// L296
  int32_t v198 = 4 / v196;	// L297
  int64_t v199 = v197;	// L298
  int64_t v200 = v198;	// L299
  int64_t v201 = v199 * v200;	// L300
  int64_t k2;	// L301
  k2 = v201;	// L302
  int64_t v203 = k2;	// L303
  int v204 = v203;	// L304
  float v205 = _st_tw_1[v204][0];	// L305
  float wr2;	// L306
  wr2 = v205;	// L307
  int64_t v207 = k2;	// L308
  int v208 = v207;	// L309
  float v209 = _st_tw_1[v208][1];	// L310
  float wi2;	// L311
  wi2 = v209;	// L312
  float v211[2];
  {
    hls::vector< float, 2 > _vec = v190.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v211[_iv0] = _vec[_iv0];
    }
  }	// L313
  float v212[2];
  {
    hls::vector< float, 2 > _vec = v191.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v212[_iv0] = _vec[_iv0];
    }
  }	// L314
  float v213 = wr2;	// L315
  float v214 = v212[0];	// L316
  float v215 = v213 * v214;	// L317
  float v216 = wi2;	// L318
  float v217 = v212[1];	// L319
  float v218 = v216 * v217;	// L320
  float v219 = v215 - v218;	// L321
  float tr2;	// L322
  tr2 = v219;	// L323
  float v221 = wr2;	// L324
  float v222 = v212[1];	// L325
  float v223 = v221 * v222;	// L326
  float v224 = wi2;	// L327
  float v225 = v212[0];	// L328
  float v226 = v224 * v225;	// L329
  float v227 = v223 + v226;	// L330
  float ti2;	// L331
  ti2 = v227;	// L332
  float u2[2];	// L333
  for (int v230 = 0; v230 < 2; v230++) {	// L334
    u2[v230] = (float)0.000000;	// L334
  }
  float l2[2];	// L335
  for (int v232 = 0; v232 < 2; v232++) {	// L336
    l2[v232] = (float)0.000000;	// L336
  }
  float v233 = v211[0];	// L337
  float v234 = tr2;	// L338
  float v235 = v233 + v234;	// L339
  u2[0] = v235;	// L340
  float v236 = v211[1];	// L341
  float v237 = ti2;	// L342
  float v238 = v236 + v237;	// L343
  u2[1] = v238;	// L344
  float v239 = v211[0];	// L345
  float v240 = tr2;	// L346
  float v241 = v239 - v240;	// L347
  l2[0] = v241;	// L348
  float v242 = v211[1];	// L349
  float v243 = ti2;	// L350
  float v244 = v242 - v243;	// L351
  l2[1] = v244;	// L352
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u2[_iv0];
    }
    v192.write(_vec);
  }	// L353
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l2[_iv0];
    }
    v193.write(_vec);
  }	// L354
}

float _st_tw_2[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L357
void bfly_0_3(
  hls::stream< hls::vector< float, 2 > >& v245,
  hls::stream< hls::vector< float, 2 > >& v246,
  hls::stream< hls::vector< float, 2 > >& v247,
  hls::stream< hls::vector< float, 2 > >& v248
) {	// L358
  // placeholder for const float _st_tw_2	// L365
  int32_t span3;	// L366
  span3 = 1;	// L367
  int32_t v251 = span3;	// L368
  int32_t v252 = 3 % v251;	// L369
  int32_t v253 = 4 / v251;	// L370
  int64_t v254 = v252;	// L371
  int64_t v255 = v253;	// L372
  int64_t v256 = v254 * v255;	// L373
  int64_t k3;	// L374
  k3 = v256;	// L375
  int64_t v258 = k3;	// L376
  int v259 = v258;	// L377
  float v260 = _st_tw_2[v259][0];	// L378
  float wr3;	// L379
  wr3 = v260;	// L380
  int64_t v262 = k3;	// L381
  int v263 = v262;	// L382
  float v264 = _st_tw_2[v263][1];	// L383
  float wi3;	// L384
  wi3 = v264;	// L385
  float v266[2];
  {
    hls::vector< float, 2 > _vec = v245.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v266[_iv0] = _vec[_iv0];
    }
  }	// L386
  float v267[2];
  {
    hls::vector< float, 2 > _vec = v246.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v267[_iv0] = _vec[_iv0];
    }
  }	// L387
  float v268 = wr3;	// L388
  float v269 = v267[0];	// L389
  float v270 = v268 * v269;	// L390
  float v271 = wi3;	// L391
  float v272 = v267[1];	// L392
  float v273 = v271 * v272;	// L393
  float v274 = v270 - v273;	// L394
  float tr3;	// L395
  tr3 = v274;	// L396
  float v276 = wr3;	// L397
  float v277 = v267[1];	// L398
  float v278 = v276 * v277;	// L399
  float v279 = wi3;	// L400
  float v280 = v267[0];	// L401
  float v281 = v279 * v280;	// L402
  float v282 = v278 + v281;	// L403
  float ti3;	// L404
  ti3 = v282;	// L405
  float u3[2];	// L406
  for (int v285 = 0; v285 < 2; v285++) {	// L407
    u3[v285] = (float)0.000000;	// L407
  }
  float l3[2];	// L408
  for (int v287 = 0; v287 < 2; v287++) {	// L409
    l3[v287] = (float)0.000000;	// L409
  }
  float v288 = v266[0];	// L410
  float v289 = tr3;	// L411
  float v290 = v288 + v289;	// L412
  u3[0] = v290;	// L413
  float v291 = v266[1];	// L414
  float v292 = ti3;	// L415
  float v293 = v291 + v292;	// L416
  u3[1] = v293;	// L417
  float v294 = v266[0];	// L418
  float v295 = tr3;	// L419
  float v296 = v294 - v295;	// L420
  l3[0] = v296;	// L421
  float v297 = v266[1];	// L422
  float v298 = ti3;	// L423
  float v299 = v297 - v298;	// L424
  l3[1] = v299;	// L425
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u3[_iv0];
    }
    v247.write(_vec);
  }	// L426
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l3[_iv0];
    }
    v248.write(_vec);
  }	// L427
}

float _st_tw_3[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L430
void bfly_1_0(
  hls::stream< hls::vector< float, 2 > >& v300,
  hls::stream< hls::vector< float, 2 > >& v301,
  hls::stream< hls::vector< float, 2 > >& v302,
  hls::stream< hls::vector< float, 2 > >& v303
) {	// L431
  // placeholder for const float _st_tw_3	// L438
  int32_t span4;	// L439
  span4 = 2;	// L440
  int32_t v306 = span4;	// L441
  int32_t v307 = 0 % v306;	// L442
  int32_t v308 = 4 / v306;	// L443
  int64_t v309 = v307;	// L444
  int64_t v310 = v308;	// L445
  int64_t v311 = v309 * v310;	// L446
  int64_t k4;	// L447
  k4 = v311;	// L448
  int64_t v313 = k4;	// L449
  int v314 = v313;	// L450
  float v315 = _st_tw_3[v314][0];	// L451
  float wr4;	// L452
  wr4 = v315;	// L453
  int64_t v317 = k4;	// L454
  int v318 = v317;	// L455
  float v319 = _st_tw_3[v318][1];	// L456
  float wi4;	// L457
  wi4 = v319;	// L458
  float v321[2];
  {
    hls::vector< float, 2 > _vec = v300.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v321[_iv0] = _vec[_iv0];
    }
  }	// L459
  float v322[2];
  {
    hls::vector< float, 2 > _vec = v301.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v322[_iv0] = _vec[_iv0];
    }
  }	// L460
  float v323 = wr4;	// L461
  float v324 = v322[0];	// L462
  float v325 = v323 * v324;	// L463
  float v326 = wi4;	// L464
  float v327 = v322[1];	// L465
  float v328 = v326 * v327;	// L466
  float v329 = v325 - v328;	// L467
  float tr4;	// L468
  tr4 = v329;	// L469
  float v331 = wr4;	// L470
  float v332 = v322[1];	// L471
  float v333 = v331 * v332;	// L472
  float v334 = wi4;	// L473
  float v335 = v322[0];	// L474
  float v336 = v334 * v335;	// L475
  float v337 = v333 + v336;	// L476
  float ti4;	// L477
  ti4 = v337;	// L478
  float u4[2];	// L479
  for (int v340 = 0; v340 < 2; v340++) {	// L480
    u4[v340] = (float)0.000000;	// L480
  }
  float l4[2];	// L481
  for (int v342 = 0; v342 < 2; v342++) {	// L482
    l4[v342] = (float)0.000000;	// L482
  }
  float v343 = v321[0];	// L483
  float v344 = tr4;	// L484
  float v345 = v343 + v344;	// L485
  u4[0] = v345;	// L486
  float v346 = v321[1];	// L487
  float v347 = ti4;	// L488
  float v348 = v346 + v347;	// L489
  u4[1] = v348;	// L490
  float v349 = v321[0];	// L491
  float v350 = tr4;	// L492
  float v351 = v349 - v350;	// L493
  l4[0] = v351;	// L494
  float v352 = v321[1];	// L495
  float v353 = ti4;	// L496
  float v354 = v352 - v353;	// L497
  l4[1] = v354;	// L498
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u4[_iv0];
    }
    v302.write(_vec);
  }	// L499
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l4[_iv0];
    }
    v303.write(_vec);
  }	// L500
}

float _st_tw_4[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L503
void bfly_1_1(
  hls::stream< hls::vector< float, 2 > >& v355,
  hls::stream< hls::vector< float, 2 > >& v356,
  hls::stream< hls::vector< float, 2 > >& v357,
  hls::stream< hls::vector< float, 2 > >& v358
) {	// L504
  // placeholder for const float _st_tw_4	// L511
  int32_t span5;	// L512
  span5 = 2;	// L513
  int32_t v361 = span5;	// L514
  int32_t v362 = 1 % v361;	// L515
  int32_t v363 = 4 / v361;	// L516
  int64_t v364 = v362;	// L517
  int64_t v365 = v363;	// L518
  int64_t v366 = v364 * v365;	// L519
  int64_t k5;	// L520
  k5 = v366;	// L521
  int64_t v368 = k5;	// L522
  int v369 = v368;	// L523
  float v370 = _st_tw_4[v369][0];	// L524
  float wr5;	// L525
  wr5 = v370;	// L526
  int64_t v372 = k5;	// L527
  int v373 = v372;	// L528
  float v374 = _st_tw_4[v373][1];	// L529
  float wi5;	// L530
  wi5 = v374;	// L531
  float v376[2];
  {
    hls::vector< float, 2 > _vec = v355.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v376[_iv0] = _vec[_iv0];
    }
  }	// L532
  float v377[2];
  {
    hls::vector< float, 2 > _vec = v356.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v377[_iv0] = _vec[_iv0];
    }
  }	// L533
  float v378 = wr5;	// L534
  float v379 = v377[0];	// L535
  float v380 = v378 * v379;	// L536
  float v381 = wi5;	// L537
  float v382 = v377[1];	// L538
  float v383 = v381 * v382;	// L539
  float v384 = v380 - v383;	// L540
  float tr5;	// L541
  tr5 = v384;	// L542
  float v386 = wr5;	// L543
  float v387 = v377[1];	// L544
  float v388 = v386 * v387;	// L545
  float v389 = wi5;	// L546
  float v390 = v377[0];	// L547
  float v391 = v389 * v390;	// L548
  float v392 = v388 + v391;	// L549
  float ti5;	// L550
  ti5 = v392;	// L551
  float u5[2];	// L552
  for (int v395 = 0; v395 < 2; v395++) {	// L553
    u5[v395] = (float)0.000000;	// L553
  }
  float l5[2];	// L554
  for (int v397 = 0; v397 < 2; v397++) {	// L555
    l5[v397] = (float)0.000000;	// L555
  }
  float v398 = v376[0];	// L556
  float v399 = tr5;	// L557
  float v400 = v398 + v399;	// L558
  u5[0] = v400;	// L559
  float v401 = v376[1];	// L560
  float v402 = ti5;	// L561
  float v403 = v401 + v402;	// L562
  u5[1] = v403;	// L563
  float v404 = v376[0];	// L564
  float v405 = tr5;	// L565
  float v406 = v404 - v405;	// L566
  l5[0] = v406;	// L567
  float v407 = v376[1];	// L568
  float v408 = ti5;	// L569
  float v409 = v407 - v408;	// L570
  l5[1] = v409;	// L571
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u5[_iv0];
    }
    v357.write(_vec);
  }	// L572
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l5[_iv0];
    }
    v358.write(_vec);
  }	// L573
}

float _st_tw_5[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L576
void bfly_1_2(
  hls::stream< hls::vector< float, 2 > >& v410,
  hls::stream< hls::vector< float, 2 > >& v411,
  hls::stream< hls::vector< float, 2 > >& v412,
  hls::stream< hls::vector< float, 2 > >& v413
) {	// L577
  // placeholder for const float _st_tw_5	// L583
  int32_t span6;	// L584
  span6 = 2;	// L585
  int32_t v416 = span6;	// L586
  int32_t v417 = 2 % v416;	// L587
  int32_t v418 = 4 / v416;	// L588
  int64_t v419 = v417;	// L589
  int64_t v420 = v418;	// L590
  int64_t v421 = v419 * v420;	// L591
  int64_t k6;	// L592
  k6 = v421;	// L593
  int64_t v423 = k6;	// L594
  int v424 = v423;	// L595
  float v425 = _st_tw_5[v424][0];	// L596
  float wr6;	// L597
  wr6 = v425;	// L598
  int64_t v427 = k6;	// L599
  int v428 = v427;	// L600
  float v429 = _st_tw_5[v428][1];	// L601
  float wi6;	// L602
  wi6 = v429;	// L603
  float v431[2];
  {
    hls::vector< float, 2 > _vec = v410.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v431[_iv0] = _vec[_iv0];
    }
  }	// L604
  float v432[2];
  {
    hls::vector< float, 2 > _vec = v411.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v432[_iv0] = _vec[_iv0];
    }
  }	// L605
  float v433 = wr6;	// L606
  float v434 = v432[0];	// L607
  float v435 = v433 * v434;	// L608
  float v436 = wi6;	// L609
  float v437 = v432[1];	// L610
  float v438 = v436 * v437;	// L611
  float v439 = v435 - v438;	// L612
  float tr6;	// L613
  tr6 = v439;	// L614
  float v441 = wr6;	// L615
  float v442 = v432[1];	// L616
  float v443 = v441 * v442;	// L617
  float v444 = wi6;	// L618
  float v445 = v432[0];	// L619
  float v446 = v444 * v445;	// L620
  float v447 = v443 + v446;	// L621
  float ti6;	// L622
  ti6 = v447;	// L623
  float u6[2];	// L624
  for (int v450 = 0; v450 < 2; v450++) {	// L625
    u6[v450] = (float)0.000000;	// L625
  }
  float l6[2];	// L626
  for (int v452 = 0; v452 < 2; v452++) {	// L627
    l6[v452] = (float)0.000000;	// L627
  }
  float v453 = v431[0];	// L628
  float v454 = tr6;	// L629
  float v455 = v453 + v454;	// L630
  u6[0] = v455;	// L631
  float v456 = v431[1];	// L632
  float v457 = ti6;	// L633
  float v458 = v456 + v457;	// L634
  u6[1] = v458;	// L635
  float v459 = v431[0];	// L636
  float v460 = tr6;	// L637
  float v461 = v459 - v460;	// L638
  l6[0] = v461;	// L639
  float v462 = v431[1];	// L640
  float v463 = ti6;	// L641
  float v464 = v462 - v463;	// L642
  l6[1] = v464;	// L643
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u6[_iv0];
    }
    v412.write(_vec);
  }	// L644
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l6[_iv0];
    }
    v413.write(_vec);
  }	// L645
}

float _st_tw_6[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L648
void bfly_1_3(
  hls::stream< hls::vector< float, 2 > >& v465,
  hls::stream< hls::vector< float, 2 > >& v466,
  hls::stream< hls::vector< float, 2 > >& v467,
  hls::stream< hls::vector< float, 2 > >& v468
) {	// L649
  // placeholder for const float _st_tw_6	// L656
  int32_t span7;	// L657
  span7 = 2;	// L658
  int32_t v471 = span7;	// L659
  int32_t v472 = 3 % v471;	// L660
  int32_t v473 = 4 / v471;	// L661
  int64_t v474 = v472;	// L662
  int64_t v475 = v473;	// L663
  int64_t v476 = v474 * v475;	// L664
  int64_t k7;	// L665
  k7 = v476;	// L666
  int64_t v478 = k7;	// L667
  int v479 = v478;	// L668
  float v480 = _st_tw_6[v479][0];	// L669
  float wr7;	// L670
  wr7 = v480;	// L671
  int64_t v482 = k7;	// L672
  int v483 = v482;	// L673
  float v484 = _st_tw_6[v483][1];	// L674
  float wi7;	// L675
  wi7 = v484;	// L676
  float v486[2];
  {
    hls::vector< float, 2 > _vec = v465.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v486[_iv0] = _vec[_iv0];
    }
  }	// L677
  float v487[2];
  {
    hls::vector< float, 2 > _vec = v466.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v487[_iv0] = _vec[_iv0];
    }
  }	// L678
  float v488 = wr7;	// L679
  float v489 = v487[0];	// L680
  float v490 = v488 * v489;	// L681
  float v491 = wi7;	// L682
  float v492 = v487[1];	// L683
  float v493 = v491 * v492;	// L684
  float v494 = v490 - v493;	// L685
  float tr7;	// L686
  tr7 = v494;	// L687
  float v496 = wr7;	// L688
  float v497 = v487[1];	// L689
  float v498 = v496 * v497;	// L690
  float v499 = wi7;	// L691
  float v500 = v487[0];	// L692
  float v501 = v499 * v500;	// L693
  float v502 = v498 + v501;	// L694
  float ti7;	// L695
  ti7 = v502;	// L696
  float u7[2];	// L697
  for (int v505 = 0; v505 < 2; v505++) {	// L698
    u7[v505] = (float)0.000000;	// L698
  }
  float l7[2];	// L699
  for (int v507 = 0; v507 < 2; v507++) {	// L700
    l7[v507] = (float)0.000000;	// L700
  }
  float v508 = v486[0];	// L701
  float v509 = tr7;	// L702
  float v510 = v508 + v509;	// L703
  u7[0] = v510;	// L704
  float v511 = v486[1];	// L705
  float v512 = ti7;	// L706
  float v513 = v511 + v512;	// L707
  u7[1] = v513;	// L708
  float v514 = v486[0];	// L709
  float v515 = tr7;	// L710
  float v516 = v514 - v515;	// L711
  l7[0] = v516;	// L712
  float v517 = v486[1];	// L713
  float v518 = ti7;	// L714
  float v519 = v517 - v518;	// L715
  l7[1] = v519;	// L716
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u7[_iv0];
    }
    v467.write(_vec);
  }	// L717
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l7[_iv0];
    }
    v468.write(_vec);
  }	// L718
}

float _st_tw_7[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L721
void bfly_2_0(
  hls::stream< hls::vector< float, 2 > >& v520,
  hls::stream< hls::vector< float, 2 > >& v521,
  hls::stream< hls::vector< float, 2 > >& v522,
  hls::stream< hls::vector< float, 2 > >& v523
) {	// L722
  // placeholder for const float _st_tw_7	// L728
  int32_t span8;	// L729
  span8 = 4;	// L730
  int32_t v526 = span8;	// L731
  int32_t v527 = 0 % v526;	// L732
  int32_t v528 = 4 / v526;	// L733
  int64_t v529 = v527;	// L734
  int64_t v530 = v528;	// L735
  int64_t v531 = v529 * v530;	// L736
  int64_t k8;	// L737
  k8 = v531;	// L738
  int64_t v533 = k8;	// L739
  int v534 = v533;	// L740
  float v535 = _st_tw_7[v534][0];	// L741
  float wr8;	// L742
  wr8 = v535;	// L743
  int64_t v537 = k8;	// L744
  int v538 = v537;	// L745
  float v539 = _st_tw_7[v538][1];	// L746
  float wi8;	// L747
  wi8 = v539;	// L748
  float v541[2];
  {
    hls::vector< float, 2 > _vec = v520.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v541[_iv0] = _vec[_iv0];
    }
  }	// L749
  float v542[2];
  {
    hls::vector< float, 2 > _vec = v521.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v542[_iv0] = _vec[_iv0];
    }
  }	// L750
  float v543 = wr8;	// L751
  float v544 = v542[0];	// L752
  float v545 = v543 * v544;	// L753
  float v546 = wi8;	// L754
  float v547 = v542[1];	// L755
  float v548 = v546 * v547;	// L756
  float v549 = v545 - v548;	// L757
  float tr8;	// L758
  tr8 = v549;	// L759
  float v551 = wr8;	// L760
  float v552 = v542[1];	// L761
  float v553 = v551 * v552;	// L762
  float v554 = wi8;	// L763
  float v555 = v542[0];	// L764
  float v556 = v554 * v555;	// L765
  float v557 = v553 + v556;	// L766
  float ti8;	// L767
  ti8 = v557;	// L768
  float u8[2];	// L769
  for (int v560 = 0; v560 < 2; v560++) {	// L770
    u8[v560] = (float)0.000000;	// L770
  }
  float l8[2];	// L771
  for (int v562 = 0; v562 < 2; v562++) {	// L772
    l8[v562] = (float)0.000000;	// L772
  }
  float v563 = v541[0];	// L773
  float v564 = tr8;	// L774
  float v565 = v563 + v564;	// L775
  u8[0] = v565;	// L776
  float v566 = v541[1];	// L777
  float v567 = ti8;	// L778
  float v568 = v566 + v567;	// L779
  u8[1] = v568;	// L780
  float v569 = v541[0];	// L781
  float v570 = tr8;	// L782
  float v571 = v569 - v570;	// L783
  l8[0] = v571;	// L784
  float v572 = v541[1];	// L785
  float v573 = ti8;	// L786
  float v574 = v572 - v573;	// L787
  l8[1] = v574;	// L788
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u8[_iv0];
    }
    v522.write(_vec);
  }	// L789
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l8[_iv0];
    }
    v523.write(_vec);
  }	// L790
}

float _st_tw_8[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L793
void bfly_2_1(
  hls::stream< hls::vector< float, 2 > >& v575,
  hls::stream< hls::vector< float, 2 > >& v576,
  hls::stream< hls::vector< float, 2 > >& v577,
  hls::stream< hls::vector< float, 2 > >& v578
) {	// L794
  // placeholder for const float _st_tw_8	// L800
  int32_t span9;	// L801
  span9 = 4;	// L802
  int32_t v581 = span9;	// L803
  int32_t v582 = 1 % v581;	// L804
  int32_t v583 = 4 / v581;	// L805
  int64_t v584 = v582;	// L806
  int64_t v585 = v583;	// L807
  int64_t v586 = v584 * v585;	// L808
  int64_t k9;	// L809
  k9 = v586;	// L810
  int64_t v588 = k9;	// L811
  int v589 = v588;	// L812
  float v590 = _st_tw_8[v589][0];	// L813
  float wr9;	// L814
  wr9 = v590;	// L815
  int64_t v592 = k9;	// L816
  int v593 = v592;	// L817
  float v594 = _st_tw_8[v593][1];	// L818
  float wi9;	// L819
  wi9 = v594;	// L820
  float v596[2];
  {
    hls::vector< float, 2 > _vec = v575.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v596[_iv0] = _vec[_iv0];
    }
  }	// L821
  float v597[2];
  {
    hls::vector< float, 2 > _vec = v576.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v597[_iv0] = _vec[_iv0];
    }
  }	// L822
  float v598 = wr9;	// L823
  float v599 = v597[0];	// L824
  float v600 = v598 * v599;	// L825
  float v601 = wi9;	// L826
  float v602 = v597[1];	// L827
  float v603 = v601 * v602;	// L828
  float v604 = v600 - v603;	// L829
  float tr9;	// L830
  tr9 = v604;	// L831
  float v606 = wr9;	// L832
  float v607 = v597[1];	// L833
  float v608 = v606 * v607;	// L834
  float v609 = wi9;	// L835
  float v610 = v597[0];	// L836
  float v611 = v609 * v610;	// L837
  float v612 = v608 + v611;	// L838
  float ti9;	// L839
  ti9 = v612;	// L840
  float u9[2];	// L841
  for (int v615 = 0; v615 < 2; v615++) {	// L842
    u9[v615] = (float)0.000000;	// L842
  }
  float l9[2];	// L843
  for (int v617 = 0; v617 < 2; v617++) {	// L844
    l9[v617] = (float)0.000000;	// L844
  }
  float v618 = v596[0];	// L845
  float v619 = tr9;	// L846
  float v620 = v618 + v619;	// L847
  u9[0] = v620;	// L848
  float v621 = v596[1];	// L849
  float v622 = ti9;	// L850
  float v623 = v621 + v622;	// L851
  u9[1] = v623;	// L852
  float v624 = v596[0];	// L853
  float v625 = tr9;	// L854
  float v626 = v624 - v625;	// L855
  l9[0] = v626;	// L856
  float v627 = v596[1];	// L857
  float v628 = ti9;	// L858
  float v629 = v627 - v628;	// L859
  l9[1] = v629;	// L860
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u9[_iv0];
    }
    v577.write(_vec);
  }	// L861
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l9[_iv0];
    }
    v578.write(_vec);
  }	// L862
}

float _st_tw_9[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L865
void bfly_2_2(
  hls::stream< hls::vector< float, 2 > >& v630,
  hls::stream< hls::vector< float, 2 > >& v631,
  hls::stream< hls::vector< float, 2 > >& v632,
  hls::stream< hls::vector< float, 2 > >& v633
) {	// L866
  // placeholder for const float _st_tw_9	// L872
  int32_t span10;	// L873
  span10 = 4;	// L874
  int32_t v636 = span10;	// L875
  int32_t v637 = 2 % v636;	// L876
  int32_t v638 = 4 / v636;	// L877
  int64_t v639 = v637;	// L878
  int64_t v640 = v638;	// L879
  int64_t v641 = v639 * v640;	// L880
  int64_t k10;	// L881
  k10 = v641;	// L882
  int64_t v643 = k10;	// L883
  int v644 = v643;	// L884
  float v645 = _st_tw_9[v644][0];	// L885
  float wr10;	// L886
  wr10 = v645;	// L887
  int64_t v647 = k10;	// L888
  int v648 = v647;	// L889
  float v649 = _st_tw_9[v648][1];	// L890
  float wi10;	// L891
  wi10 = v649;	// L892
  float v651[2];
  {
    hls::vector< float, 2 > _vec = v630.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v651[_iv0] = _vec[_iv0];
    }
  }	// L893
  float v652[2];
  {
    hls::vector< float, 2 > _vec = v631.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v652[_iv0] = _vec[_iv0];
    }
  }	// L894
  float v653 = wr10;	// L895
  float v654 = v652[0];	// L896
  float v655 = v653 * v654;	// L897
  float v656 = wi10;	// L898
  float v657 = v652[1];	// L899
  float v658 = v656 * v657;	// L900
  float v659 = v655 - v658;	// L901
  float tr10;	// L902
  tr10 = v659;	// L903
  float v661 = wr10;	// L904
  float v662 = v652[1];	// L905
  float v663 = v661 * v662;	// L906
  float v664 = wi10;	// L907
  float v665 = v652[0];	// L908
  float v666 = v664 * v665;	// L909
  float v667 = v663 + v666;	// L910
  float ti10;	// L911
  ti10 = v667;	// L912
  float u10[2];	// L913
  for (int v670 = 0; v670 < 2; v670++) {	// L914
    u10[v670] = (float)0.000000;	// L914
  }
  float l10[2];	// L915
  for (int v672 = 0; v672 < 2; v672++) {	// L916
    l10[v672] = (float)0.000000;	// L916
  }
  float v673 = v651[0];	// L917
  float v674 = tr10;	// L918
  float v675 = v673 + v674;	// L919
  u10[0] = v675;	// L920
  float v676 = v651[1];	// L921
  float v677 = ti10;	// L922
  float v678 = v676 + v677;	// L923
  u10[1] = v678;	// L924
  float v679 = v651[0];	// L925
  float v680 = tr10;	// L926
  float v681 = v679 - v680;	// L927
  l10[0] = v681;	// L928
  float v682 = v651[1];	// L929
  float v683 = ti10;	// L930
  float v684 = v682 - v683;	// L931
  l10[1] = v684;	// L932
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u10[_iv0];
    }
    v632.write(_vec);
  }	// L933
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l10[_iv0];
    }
    v633.write(_vec);
  }	// L934
}

float _st_tw_10[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L937
void bfly_2_3(
  hls::stream< hls::vector< float, 2 > >& v685,
  hls::stream< hls::vector< float, 2 > >& v686,
  hls::stream< hls::vector< float, 2 > >& v687,
  hls::stream< hls::vector< float, 2 > >& v688
) {	// L938
  // placeholder for const float _st_tw_10	// L944
  int32_t span11;	// L945
  span11 = 4;	// L946
  int32_t v691 = span11;	// L947
  int32_t v692 = 3 % v691;	// L948
  int32_t v693 = 4 / v691;	// L949
  int64_t v694 = v692;	// L950
  int64_t v695 = v693;	// L951
  int64_t v696 = v694 * v695;	// L952
  int64_t k11;	// L953
  k11 = v696;	// L954
  int64_t v698 = k11;	// L955
  int v699 = v698;	// L956
  float v700 = _st_tw_10[v699][0];	// L957
  float wr11;	// L958
  wr11 = v700;	// L959
  int64_t v702 = k11;	// L960
  int v703 = v702;	// L961
  float v704 = _st_tw_10[v703][1];	// L962
  float wi11;	// L963
  wi11 = v704;	// L964
  float v706[2];
  {
    hls::vector< float, 2 > _vec = v685.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v706[_iv0] = _vec[_iv0];
    }
  }	// L965
  float v707[2];
  {
    hls::vector< float, 2 > _vec = v686.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v707[_iv0] = _vec[_iv0];
    }
  }	// L966
  float v708 = wr11;	// L967
  float v709 = v707[0];	// L968
  float v710 = v708 * v709;	// L969
  float v711 = wi11;	// L970
  float v712 = v707[1];	// L971
  float v713 = v711 * v712;	// L972
  float v714 = v710 - v713;	// L973
  float tr11;	// L974
  tr11 = v714;	// L975
  float v716 = wr11;	// L976
  float v717 = v707[1];	// L977
  float v718 = v716 * v717;	// L978
  float v719 = wi11;	// L979
  float v720 = v707[0];	// L980
  float v721 = v719 * v720;	// L981
  float v722 = v718 + v721;	// L982
  float ti11;	// L983
  ti11 = v722;	// L984
  float u11[2];	// L985
  for (int v725 = 0; v725 < 2; v725++) {	// L986
    u11[v725] = (float)0.000000;	// L986
  }
  float l11[2];	// L987
  for (int v727 = 0; v727 < 2; v727++) {	// L988
    l11[v727] = (float)0.000000;	// L988
  }
  float v728 = v706[0];	// L989
  float v729 = tr11;	// L990
  float v730 = v728 + v729;	// L991
  u11[0] = v730;	// L992
  float v731 = v706[1];	// L993
  float v732 = ti11;	// L994
  float v733 = v731 + v732;	// L995
  u11[1] = v733;	// L996
  float v734 = v706[0];	// L997
  float v735 = tr11;	// L998
  float v736 = v734 - v735;	// L999
  l11[0] = v736;	// L1000
  float v737 = v706[1];	// L1001
  float v738 = ti11;	// L1002
  float v739 = v737 - v738;	// L1003
  l11[1] = v739;	// L1004
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u11[_iv0];
    }
    v687.write(_vec);
  }	// L1005
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l11[_iv0];
    }
    v688.write(_vec);
  }	// L1006
}

int32_t _ix0_7[1][1] = {0};	// L1009
void bfly_up_out_drain_0(
  float v740[8][2],
  hls::stream< hls::vector< float, 2 > >& v741
) {	// L1010
  #pragma HLS array_partition variable=v740 complete dim=1
  #pragma HLS array_partition variable=v740 complete dim=2

  // placeholder for const int32_t _ix0_7	// L1011
  l_S__t_0__t8: for (int _t8 = 0; _t8 < 1; _t8++) {	// L1012
    float v744[2];
    {
      hls::vector< float, 2 > _vec = v741.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v744[_iv0] = _vec[_iv0];
      }
    }	// L1013
    l_S__b0_0__b08: for (int _b08 = 0; _b08 < 2; _b08++) {	// L1014
      float v746 = v744[_b08];	// L1015
      int32_t v747 = _ix0_7[_t8][0];	// L1016
      int v748 = v747;	// L1017
      v740[v748][_b08] = v746;	// L1018
    }
  }
}

int32_t _ix0_8[1][1] = {1};	// L1023
void bfly_up_out_drain_1(
  float v749[8][2],
  hls::stream< hls::vector< float, 2 > >& v750
) {	// L1024
  #pragma HLS array_partition variable=v749 complete dim=1
  #pragma HLS array_partition variable=v749 complete dim=2

  // placeholder for const int32_t _ix0_8	// L1025
  l_S__t_0__t9: for (int _t9 = 0; _t9 < 1; _t9++) {	// L1026
    float v753[2];
    {
      hls::vector< float, 2 > _vec = v750.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v753[_iv0] = _vec[_iv0];
      }
    }	// L1027
    l_S__b0_0__b09: for (int _b09 = 0; _b09 < 2; _b09++) {	// L1028
      float v755 = v753[_b09];	// L1029
      int32_t v756 = _ix0_8[_t9][0];	// L1030
      int v757 = v756;	// L1031
      v749[v757][_b09] = v755;	// L1032
    }
  }
}

int32_t _ix0_9[1][1] = {2};	// L1037
void bfly_up_out_drain_2(
  float v758[8][2],
  hls::stream< hls::vector< float, 2 > >& v759
) {	// L1038
  #pragma HLS array_partition variable=v758 complete dim=1
  #pragma HLS array_partition variable=v758 complete dim=2

  // placeholder for const int32_t _ix0_9	// L1039
  l_S__t_0__t10: for (int _t10 = 0; _t10 < 1; _t10++) {	// L1040
    float v762[2];
    {
      hls::vector< float, 2 > _vec = v759.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v762[_iv0] = _vec[_iv0];
      }
    }	// L1041
    l_S__b0_0__b010: for (int _b010 = 0; _b010 < 2; _b010++) {	// L1042
      float v764 = v762[_b010];	// L1043
      int32_t v765 = _ix0_9[_t10][0];	// L1044
      int v766 = v765;	// L1045
      v758[v766][_b010] = v764;	// L1046
    }
  }
}

int32_t _ix0_10[1][1] = {3};	// L1051
void bfly_up_out_drain_3(
  float v767[8][2],
  hls::stream< hls::vector< float, 2 > >& v768
) {	// L1052
  #pragma HLS array_partition variable=v767 complete dim=1
  #pragma HLS array_partition variable=v767 complete dim=2

  // placeholder for const int32_t _ix0_10	// L1053
  l_S__t_0__t11: for (int _t11 = 0; _t11 < 1; _t11++) {	// L1054
    float v771[2];
    {
      hls::vector< float, 2 > _vec = v768.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v771[_iv0] = _vec[_iv0];
      }
    }	// L1055
    l_S__b0_0__b011: for (int _b011 = 0; _b011 < 2; _b011++) {	// L1056
      float v773 = v771[_b011];	// L1057
      int32_t v774 = _ix0_10[_t11][0];	// L1058
      int v775 = v774;	// L1059
      v767[v775][_b011] = v773;	// L1060
    }
  }
}

int32_t _ix0_11[1][1] = {4};	// L1065
void bfly_lo_out_drain_0(
  float v776[8][2],
  hls::stream< hls::vector< float, 2 > >& v777
) {	// L1066
  #pragma HLS array_partition variable=v776 complete dim=1
  #pragma HLS array_partition variable=v776 complete dim=2

  // placeholder for const int32_t _ix0_11	// L1067
  l_S__t_0__t12: for (int _t12 = 0; _t12 < 1; _t12++) {	// L1068
    float v780[2];
    {
      hls::vector< float, 2 > _vec = v777.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v780[_iv0] = _vec[_iv0];
      }
    }	// L1069
    l_S__b0_0__b012: for (int _b012 = 0; _b012 < 2; _b012++) {	// L1070
      float v782 = v780[_b012];	// L1071
      int32_t v783 = _ix0_11[_t12][0];	// L1072
      int v784 = v783;	// L1073
      v776[v784][_b012] = v782;	// L1074
    }
  }
}

int32_t _ix0_12[1][1] = {5};	// L1079
void bfly_lo_out_drain_1(
  float v785[8][2],
  hls::stream< hls::vector< float, 2 > >& v786
) {	// L1080
  #pragma HLS array_partition variable=v785 complete dim=1
  #pragma HLS array_partition variable=v785 complete dim=2

  // placeholder for const int32_t _ix0_12	// L1081
  l_S__t_0__t13: for (int _t13 = 0; _t13 < 1; _t13++) {	// L1082
    float v789[2];
    {
      hls::vector< float, 2 > _vec = v786.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v789[_iv0] = _vec[_iv0];
      }
    }	// L1083
    l_S__b0_0__b013: for (int _b013 = 0; _b013 < 2; _b013++) {	// L1084
      float v791 = v789[_b013];	// L1085
      int32_t v792 = _ix0_12[_t13][0];	// L1086
      int v793 = v792;	// L1087
      v785[v793][_b013] = v791;	// L1088
    }
  }
}

int32_t _ix0_13[1][1] = {6};	// L1093
void bfly_lo_out_drain_2(
  float v794[8][2],
  hls::stream< hls::vector< float, 2 > >& v795
) {	// L1094
  #pragma HLS array_partition variable=v794 complete dim=1
  #pragma HLS array_partition variable=v794 complete dim=2

  // placeholder for const int32_t _ix0_13	// L1095
  l_S__t_0__t14: for (int _t14 = 0; _t14 < 1; _t14++) {	// L1096
    float v798[2];
    {
      hls::vector< float, 2 > _vec = v795.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v798[_iv0] = _vec[_iv0];
      }
    }	// L1097
    l_S__b0_0__b014: for (int _b014 = 0; _b014 < 2; _b014++) {	// L1098
      float v800 = v798[_b014];	// L1099
      int32_t v801 = _ix0_13[_t14][0];	// L1100
      int v802 = v801;	// L1101
      v794[v802][_b014] = v800;	// L1102
    }
  }
}

int32_t _ix0_14[1][1] = {7};	// L1107
void bfly_lo_out_drain_3(
  float v803[8][2],
  hls::stream< hls::vector< float, 2 > >& v804
) {	// L1108
  #pragma HLS array_partition variable=v803 complete dim=1
  #pragma HLS array_partition variable=v803 complete dim=2

  // placeholder for const int32_t _ix0_14	// L1109
  l_S__t_0__t15: for (int _t15 = 0; _t15 < 1; _t15++) {	// L1110
    float v807[2];
    {
      hls::vector< float, 2 > _vec = v804.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v807[_iv0] = _vec[_iv0];
      }
    }	// L1111
    l_S__b0_0__b015: for (int _b015 = 0; _b015 < 2; _b015++) {	// L1112
      float v809 = v807[_b015];	// L1113
      int32_t v810 = _ix0_14[_t15][0];	// L1114
      int v811 = v810;	// L1115
      v803[v811][_b015] = v809;	// L1116
    }
  }
}

/// This is top function.
void top(
  float v812[8][2],
  float v813[8][2]
) {	// L1121
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v812 complete dim=1
  #pragma HLS array_partition variable=v812 complete dim=2

  #pragma HLS array_partition variable=v813 complete dim=1
  #pragma HLS array_partition variable=v813 complete dim=2

  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v814;
  #pragma HLS stream variable=v814 depth=2	// L1122
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v815;
  #pragma HLS stream variable=v815 depth=2	// L1123
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v816;
  #pragma HLS stream variable=v816 depth=2	// L1124
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v817;
  #pragma HLS stream variable=v817 depth=2	// L1125
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v818;
  #pragma HLS stream variable=v818 depth=2	// L1126
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v819;
  #pragma HLS stream variable=v819 depth=2	// L1127
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v820;
  #pragma HLS stream variable=v820 depth=2	// L1128
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v821;
  #pragma HLS stream variable=v821 depth=2	// L1129
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v822;
  #pragma HLS stream variable=v822 depth=2	// L1130
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v823;
  #pragma HLS stream variable=v823 depth=2	// L1131
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v824;
  #pragma HLS stream variable=v824 depth=2	// L1132
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v825;
  #pragma HLS stream variable=v825 depth=2	// L1133
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v826;
  #pragma HLS stream variable=v826 depth=2	// L1134
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v827;
  #pragma HLS stream variable=v827 depth=2	// L1135
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v828;
  #pragma HLS stream variable=v828 depth=2	// L1136
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v829;
  #pragma HLS stream variable=v829 depth=2	// L1137
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v830;
  #pragma HLS stream variable=v830 depth=2	// L1138
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v831;
  #pragma HLS stream variable=v831 depth=2	// L1139
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v832;
  #pragma HLS stream variable=v832 depth=2	// L1140
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v833;
  #pragma HLS stream variable=v833 depth=2	// L1141
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v834;
  #pragma HLS stream variable=v834 depth=2	// L1142
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v835;
  #pragma HLS stream variable=v835 depth=2	// L1143
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v836;
  #pragma HLS stream variable=v836 depth=2	// L1144
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v837;
  #pragma HLS stream variable=v837 depth=2	// L1145
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v838;
  #pragma HLS stream variable=v838 depth=2	// L1146
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v839;
  #pragma HLS stream variable=v839 depth=2	// L1147
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v840;
  #pragma HLS stream variable=v840 depth=2	// L1148
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v841;
  #pragma HLS stream variable=v841 depth=2	// L1149
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v842;
  #pragma HLS stream variable=v842 depth=2	// L1150
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v843;
  #pragma HLS stream variable=v843 depth=2	// L1151
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v844;
  #pragma HLS stream variable=v844 depth=2	// L1152
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v845;
  #pragma HLS stream variable=v845 depth=2	// L1153
  bfly_up_in_load_0(v812, v830);	// L1154
  bfly_up_in_load_1(v812, v831);	// L1155
  bfly_up_in_load_2(v812, v832);	// L1156
  bfly_up_in_load_3(v812, v833);	// L1157
  bfly_lo_in_load_0(v812, v834);	// L1158
  bfly_lo_in_load_1(v812, v835);	// L1159
  bfly_lo_in_load_2(v812, v836);	// L1160
  bfly_lo_in_load_3(v812, v837);	// L1161
  bfly_0_0(v830, v834, v814, v815);	// L1162
  bfly_0_1(v831, v835, v816, v817);	// L1163
  bfly_0_2(v832, v836, v818, v819);	// L1164
  bfly_0_3(v833, v837, v820, v821);	// L1165
  bfly_1_0(v814, v816, v822, v824);	// L1166
  bfly_1_1(v815, v817, v823, v825);	// L1167
  bfly_1_2(v818, v820, v826, v828);	// L1168
  bfly_1_3(v819, v821, v827, v829);	// L1169
  bfly_2_0(v822, v826, v838, v842);	// L1170
  bfly_2_1(v823, v827, v839, v843);	// L1171
  bfly_2_2(v824, v828, v840, v844);	// L1172
  bfly_2_3(v825, v829, v841, v845);	// L1173
  bfly_up_out_drain_0(v813, v838);	// L1174
  bfly_up_out_drain_1(v813, v839);	// L1175
  bfly_up_out_drain_2(v813, v840);	// L1176
  bfly_up_out_drain_3(v813, v841);	// L1177
  bfly_lo_out_drain_0(v813, v842);	// L1178
  bfly_lo_out_drain_1(v813, v843);	// L1179
  bfly_lo_out_drain_2(v813, v844);	// L1180
  bfly_lo_out_drain_3(v813, v845);	// L1181
}


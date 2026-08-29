
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
void feed_up_load_0(
  int8_t v0[4][4],
  hls::stream< hls::vector< int8_t, 4 > >& v1
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L5
    int8_t _blk[4];	// L6
    for (int v4 = 0; v4 < 4; v4++) {	// L7
      _blk[v4] = 0;	// L7
    }
    l_S__b0_0__b0: for (int _b0 = 0; _b0 < 4; _b0++) {	// L8
    #pragma HLS pipeline II=1
      int8_t v6 = v0[_t][_b0];	// L9
      _blk[_b0] = v6;	// L10
    }
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = _blk[_iv0];
      }
      v1.write(_vec);
    }	// L12
  }
}

void feed_3_up_load_0(
  int8_t v7[4][4],
  hls::stream< hls::vector< int8_t, 4 > >& v8
) {	// L16
  #pragma HLS array_partition variable=v7 complete dim=1
  #pragma HLS array_partition variable=v7 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 4; _t1++) {	// L18
    int8_t _blk1[4];	// L19
    for (int v11 = 0; v11 < 4; v11++) {	// L20
      _blk1[v11] = 0;	// L20
    }
    l_S__b0_0__b01: for (int _b01 = 0; _b01 < 4; _b01++) {	// L21
    #pragma HLS pipeline II=1
      int8_t v13 = v7[_t1][_b01];	// L22
      _blk1[_b01] = v13;	// L23
    }
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = _blk1[_iv0];
      }
      v8.write(_vec);
    }	// L25
  }
}

void pe_0_0(
  hls::stream< int8_t >& v14,
  hls::stream< int8_t >& v15,
  hls::stream< int8_t >& v16,
  hls::stream< int8_t >& v17,
  hls::stream< int32_t >& v18
) {	// L29
  int32_t acc;	// L31
  acc = 0;	// L32
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L33
  #pragma HLS pipeline II=1
    int8_t v21 = v14.read();	// L34
    int8_t a;	// L35
    a = v21;	// L36
    int8_t v23 = v15.read();	// L37
    int8_t b;	// L38
    b = v23;	// L39
    int8_t v25 = a;	// L40
    int8_t v26 = b;	// L41
    int16_t v27 = v25;	// L42
    int16_t v28 = v26;	// L43
    int16_t v29 = v27 * v28;	// L44
    int32_t v30 = acc;	// L45
    ap_int<33> v31 = v30;	// L46
    ap_int<33> v32 = v29;	// L47
    ap_int<33> v33 = v31 + v32;	// L48
    int32_t v34 = v33;	// L49
    acc = v34;	// L50
    int8_t v35 = a;	// L51
    v16.write(v35);	// L52
    int8_t v36 = b;	// L53
    v17.write(v36);	// L54
  }
  int32_t v37 = acc;	// L56
  v18.write(v37);	// L57
}

void pe_0_1(
  hls::stream< int8_t >& v38,
  hls::stream< int8_t >& v39,
  hls::stream< int8_t >& v40,
  hls::stream< int8_t >& v41,
  hls::stream< int32_t >& v42
) {	// L60
  int32_t acc1;	// L62
  acc1 = 0;	// L63
  l_S_k_0_k1: for (int k1 = 0; k1 < 4; k1++) {	// L64
  #pragma HLS pipeline II=1
    int8_t v45 = v38.read();	// L65
    int8_t a1;	// L66
    a1 = v45;	// L67
    int8_t v47 = v39.read();	// L68
    int8_t b1;	// L69
    b1 = v47;	// L70
    int8_t v49 = a1;	// L71
    int8_t v50 = b1;	// L72
    int16_t v51 = v49;	// L73
    int16_t v52 = v50;	// L74
    int16_t v53 = v51 * v52;	// L75
    int32_t v54 = acc1;	// L76
    ap_int<33> v55 = v54;	// L77
    ap_int<33> v56 = v53;	// L78
    ap_int<33> v57 = v55 + v56;	// L79
    int32_t v58 = v57;	// L80
    acc1 = v58;	// L81
    int8_t v59 = a1;	// L82
    v40.write(v59);	// L83
    int8_t v60 = b1;	// L84
    v41.write(v60);	// L85
  }
  int32_t v61 = acc1;	// L87
  v42.write(v61);	// L88
}

void pe_0_2(
  hls::stream< int8_t >& v62,
  hls::stream< int8_t >& v63,
  hls::stream< int8_t >& v64,
  hls::stream< int8_t >& v65,
  hls::stream< int32_t >& v66
) {	// L91
  int32_t acc2;	// L93
  acc2 = 0;	// L94
  l_S_k_0_k2: for (int k2 = 0; k2 < 4; k2++) {	// L95
  #pragma HLS pipeline II=1
    int8_t v69 = v62.read();	// L96
    int8_t a2;	// L97
    a2 = v69;	// L98
    int8_t v71 = v63.read();	// L99
    int8_t b2;	// L100
    b2 = v71;	// L101
    int8_t v73 = a2;	// L102
    int8_t v74 = b2;	// L103
    int16_t v75 = v73;	// L104
    int16_t v76 = v74;	// L105
    int16_t v77 = v75 * v76;	// L106
    int32_t v78 = acc2;	// L107
    ap_int<33> v79 = v78;	// L108
    ap_int<33> v80 = v77;	// L109
    ap_int<33> v81 = v79 + v80;	// L110
    int32_t v82 = v81;	// L111
    acc2 = v82;	// L112
    int8_t v83 = a2;	// L113
    v64.write(v83);	// L114
    int8_t v84 = b2;	// L115
    v65.write(v84);	// L116
  }
  int32_t v85 = acc2;	// L118
  v66.write(v85);	// L119
}

void pe_0_3(
  hls::stream< int8_t >& v86,
  hls::stream< int8_t >& v87,
  hls::stream< int8_t >& v88,
  hls::stream< int32_t >& v89
) {	// L122
  int32_t acc3;	// L124
  acc3 = 0;	// L125
  l_S_k_0_k3: for (int k3 = 0; k3 < 4; k3++) {	// L126
  #pragma HLS pipeline II=1
    int8_t v92 = v86.read();	// L127
    int8_t a3;	// L128
    a3 = v92;	// L129
    int8_t v94 = v87.read();	// L130
    int8_t b3;	// L131
    b3 = v94;	// L132
    int8_t v96 = a3;	// L133
    int8_t v97 = b3;	// L134
    int16_t v98 = v96;	// L135
    int16_t v99 = v97;	// L136
    int16_t v100 = v98 * v99;	// L137
    int32_t v101 = acc3;	// L138
    ap_int<33> v102 = v101;	// L139
    ap_int<33> v103 = v100;	// L140
    ap_int<33> v104 = v102 + v103;	// L141
    int32_t v105 = v104;	// L142
    acc3 = v105;	// L143
    int8_t v106 = b3;	// L144
    v88.write(v106);	// L145
  }
  int32_t v107 = acc3;	// L147
  v89.write(v107);	// L148
}

void pe_1_0(
  hls::stream< int8_t >& v108,
  hls::stream< int8_t >& v109,
  hls::stream< int8_t >& v110,
  hls::stream< int8_t >& v111,
  hls::stream< int32_t >& v112
) {	// L151
  int32_t acc4;	// L153
  acc4 = 0;	// L154
  l_S_k_0_k4: for (int k4 = 0; k4 < 4; k4++) {	// L155
  #pragma HLS pipeline II=1
    int8_t v115 = v108.read();	// L156
    int8_t a4;	// L157
    a4 = v115;	// L158
    int8_t v117 = v109.read();	// L159
    int8_t b4;	// L160
    b4 = v117;	// L161
    int8_t v119 = a4;	// L162
    int8_t v120 = b4;	// L163
    int16_t v121 = v119;	// L164
    int16_t v122 = v120;	// L165
    int16_t v123 = v121 * v122;	// L166
    int32_t v124 = acc4;	// L167
    ap_int<33> v125 = v124;	// L168
    ap_int<33> v126 = v123;	// L169
    ap_int<33> v127 = v125 + v126;	// L170
    int32_t v128 = v127;	// L171
    acc4 = v128;	// L172
    int8_t v129 = a4;	// L173
    v110.write(v129);	// L174
    int8_t v130 = b4;	// L175
    v111.write(v130);	// L176
  }
  int32_t v131 = acc4;	// L178
  v112.write(v131);	// L179
}

void pe_1_1(
  hls::stream< int8_t >& v132,
  hls::stream< int8_t >& v133,
  hls::stream< int8_t >& v134,
  hls::stream< int8_t >& v135,
  hls::stream< int32_t >& v136
) {	// L182
  int32_t acc5;	// L184
  acc5 = 0;	// L185
  l_S_k_0_k5: for (int k5 = 0; k5 < 4; k5++) {	// L186
  #pragma HLS pipeline II=1
    int8_t v139 = v132.read();	// L187
    int8_t a5;	// L188
    a5 = v139;	// L189
    int8_t v141 = v133.read();	// L190
    int8_t b5;	// L191
    b5 = v141;	// L192
    int8_t v143 = a5;	// L193
    int8_t v144 = b5;	// L194
    int16_t v145 = v143;	// L195
    int16_t v146 = v144;	// L196
    int16_t v147 = v145 * v146;	// L197
    int32_t v148 = acc5;	// L198
    ap_int<33> v149 = v148;	// L199
    ap_int<33> v150 = v147;	// L200
    ap_int<33> v151 = v149 + v150;	// L201
    int32_t v152 = v151;	// L202
    acc5 = v152;	// L203
    int8_t v153 = a5;	// L204
    v134.write(v153);	// L205
    int8_t v154 = b5;	// L206
    v135.write(v154);	// L207
  }
  int32_t v155 = acc5;	// L209
  v136.write(v155);	// L210
}

void pe_1_2(
  hls::stream< int8_t >& v156,
  hls::stream< int8_t >& v157,
  hls::stream< int8_t >& v158,
  hls::stream< int8_t >& v159,
  hls::stream< int32_t >& v160
) {	// L213
  int32_t acc6;	// L215
  acc6 = 0;	// L216
  l_S_k_0_k6: for (int k6 = 0; k6 < 4; k6++) {	// L217
  #pragma HLS pipeline II=1
    int8_t v163 = v156.read();	// L218
    int8_t a6;	// L219
    a6 = v163;	// L220
    int8_t v165 = v157.read();	// L221
    int8_t b6;	// L222
    b6 = v165;	// L223
    int8_t v167 = a6;	// L224
    int8_t v168 = b6;	// L225
    int16_t v169 = v167;	// L226
    int16_t v170 = v168;	// L227
    int16_t v171 = v169 * v170;	// L228
    int32_t v172 = acc6;	// L229
    ap_int<33> v173 = v172;	// L230
    ap_int<33> v174 = v171;	// L231
    ap_int<33> v175 = v173 + v174;	// L232
    int32_t v176 = v175;	// L233
    acc6 = v176;	// L234
    int8_t v177 = a6;	// L235
    v158.write(v177);	// L236
    int8_t v178 = b6;	// L237
    v159.write(v178);	// L238
  }
  int32_t v179 = acc6;	// L240
  v160.write(v179);	// L241
}

void pe_1_3(
  hls::stream< int8_t >& v180,
  hls::stream< int8_t >& v181,
  hls::stream< int8_t >& v182,
  hls::stream< int32_t >& v183
) {	// L244
  int32_t acc7;	// L246
  acc7 = 0;	// L247
  l_S_k_0_k7: for (int k7 = 0; k7 < 4; k7++) {	// L248
  #pragma HLS pipeline II=1
    int8_t v186 = v180.read();	// L249
    int8_t a7;	// L250
    a7 = v186;	// L251
    int8_t v188 = v181.read();	// L252
    int8_t b7;	// L253
    b7 = v188;	// L254
    int8_t v190 = a7;	// L255
    int8_t v191 = b7;	// L256
    int16_t v192 = v190;	// L257
    int16_t v193 = v191;	// L258
    int16_t v194 = v192 * v193;	// L259
    int32_t v195 = acc7;	// L260
    ap_int<33> v196 = v195;	// L261
    ap_int<33> v197 = v194;	// L262
    ap_int<33> v198 = v196 + v197;	// L263
    int32_t v199 = v198;	// L264
    acc7 = v199;	// L265
    int8_t v200 = b7;	// L266
    v182.write(v200);	// L267
  }
  int32_t v201 = acc7;	// L269
  v183.write(v201);	// L270
}

void pe_2_0(
  hls::stream< int8_t >& v202,
  hls::stream< int8_t >& v203,
  hls::stream< int8_t >& v204,
  hls::stream< int8_t >& v205,
  hls::stream< int32_t >& v206
) {	// L273
  int32_t acc8;	// L275
  acc8 = 0;	// L276
  l_S_k_0_k8: for (int k8 = 0; k8 < 4; k8++) {	// L277
  #pragma HLS pipeline II=1
    int8_t v209 = v202.read();	// L278
    int8_t a8;	// L279
    a8 = v209;	// L280
    int8_t v211 = v203.read();	// L281
    int8_t b8;	// L282
    b8 = v211;	// L283
    int8_t v213 = a8;	// L284
    int8_t v214 = b8;	// L285
    int16_t v215 = v213;	// L286
    int16_t v216 = v214;	// L287
    int16_t v217 = v215 * v216;	// L288
    int32_t v218 = acc8;	// L289
    ap_int<33> v219 = v218;	// L290
    ap_int<33> v220 = v217;	// L291
    ap_int<33> v221 = v219 + v220;	// L292
    int32_t v222 = v221;	// L293
    acc8 = v222;	// L294
    int8_t v223 = a8;	// L295
    v204.write(v223);	// L296
    int8_t v224 = b8;	// L297
    v205.write(v224);	// L298
  }
  int32_t v225 = acc8;	// L300
  v206.write(v225);	// L301
}

void pe_2_1(
  hls::stream< int8_t >& v226,
  hls::stream< int8_t >& v227,
  hls::stream< int8_t >& v228,
  hls::stream< int8_t >& v229,
  hls::stream< int32_t >& v230
) {	// L304
  int32_t acc9;	// L306
  acc9 = 0;	// L307
  l_S_k_0_k9: for (int k9 = 0; k9 < 4; k9++) {	// L308
  #pragma HLS pipeline II=1
    int8_t v233 = v226.read();	// L309
    int8_t a9;	// L310
    a9 = v233;	// L311
    int8_t v235 = v227.read();	// L312
    int8_t b9;	// L313
    b9 = v235;	// L314
    int8_t v237 = a9;	// L315
    int8_t v238 = b9;	// L316
    int16_t v239 = v237;	// L317
    int16_t v240 = v238;	// L318
    int16_t v241 = v239 * v240;	// L319
    int32_t v242 = acc9;	// L320
    ap_int<33> v243 = v242;	// L321
    ap_int<33> v244 = v241;	// L322
    ap_int<33> v245 = v243 + v244;	// L323
    int32_t v246 = v245;	// L324
    acc9 = v246;	// L325
    int8_t v247 = a9;	// L326
    v228.write(v247);	// L327
    int8_t v248 = b9;	// L328
    v229.write(v248);	// L329
  }
  int32_t v249 = acc9;	// L331
  v230.write(v249);	// L332
}

void pe_2_2(
  hls::stream< int8_t >& v250,
  hls::stream< int8_t >& v251,
  hls::stream< int8_t >& v252,
  hls::stream< int8_t >& v253,
  hls::stream< int32_t >& v254
) {	// L335
  int32_t acc10;	// L337
  acc10 = 0;	// L338
  l_S_k_0_k10: for (int k10 = 0; k10 < 4; k10++) {	// L339
  #pragma HLS pipeline II=1
    int8_t v257 = v250.read();	// L340
    int8_t a10;	// L341
    a10 = v257;	// L342
    int8_t v259 = v251.read();	// L343
    int8_t b10;	// L344
    b10 = v259;	// L345
    int8_t v261 = a10;	// L346
    int8_t v262 = b10;	// L347
    int16_t v263 = v261;	// L348
    int16_t v264 = v262;	// L349
    int16_t v265 = v263 * v264;	// L350
    int32_t v266 = acc10;	// L351
    ap_int<33> v267 = v266;	// L352
    ap_int<33> v268 = v265;	// L353
    ap_int<33> v269 = v267 + v268;	// L354
    int32_t v270 = v269;	// L355
    acc10 = v270;	// L356
    int8_t v271 = a10;	// L357
    v252.write(v271);	// L358
    int8_t v272 = b10;	// L359
    v253.write(v272);	// L360
  }
  int32_t v273 = acc10;	// L362
  v254.write(v273);	// L363
}

void pe_2_3(
  hls::stream< int8_t >& v274,
  hls::stream< int8_t >& v275,
  hls::stream< int8_t >& v276,
  hls::stream< int32_t >& v277
) {	// L366
  int32_t acc11;	// L368
  acc11 = 0;	// L369
  l_S_k_0_k11: for (int k11 = 0; k11 < 4; k11++) {	// L370
  #pragma HLS pipeline II=1
    int8_t v280 = v274.read();	// L371
    int8_t a11;	// L372
    a11 = v280;	// L373
    int8_t v282 = v275.read();	// L374
    int8_t b11;	// L375
    b11 = v282;	// L376
    int8_t v284 = a11;	// L377
    int8_t v285 = b11;	// L378
    int16_t v286 = v284;	// L379
    int16_t v287 = v285;	// L380
    int16_t v288 = v286 * v287;	// L381
    int32_t v289 = acc11;	// L382
    ap_int<33> v290 = v289;	// L383
    ap_int<33> v291 = v288;	// L384
    ap_int<33> v292 = v290 + v291;	// L385
    int32_t v293 = v292;	// L386
    acc11 = v293;	// L387
    int8_t v294 = b11;	// L388
    v276.write(v294);	// L389
  }
  int32_t v295 = acc11;	// L391
  v277.write(v295);	// L392
}

void pe_3_0(
  hls::stream< int8_t >& v296,
  hls::stream< int8_t >& v297,
  hls::stream< int8_t >& v298,
  hls::stream< int32_t >& v299
) {	// L395
  int32_t acc12;	// L397
  acc12 = 0;	// L398
  l_S_k_0_k12: for (int k12 = 0; k12 < 4; k12++) {	// L399
  #pragma HLS pipeline II=1
    int8_t v302 = v296.read();	// L400
    int8_t a12;	// L401
    a12 = v302;	// L402
    int8_t v304 = v297.read();	// L403
    int8_t b12;	// L404
    b12 = v304;	// L405
    int8_t v306 = a12;	// L406
    int8_t v307 = b12;	// L407
    int16_t v308 = v306;	// L408
    int16_t v309 = v307;	// L409
    int16_t v310 = v308 * v309;	// L410
    int32_t v311 = acc12;	// L411
    ap_int<33> v312 = v311;	// L412
    ap_int<33> v313 = v310;	// L413
    ap_int<33> v314 = v312 + v313;	// L414
    int32_t v315 = v314;	// L415
    acc12 = v315;	// L416
    int8_t v316 = a12;	// L417
    v298.write(v316);	// L418
  }
  int32_t v317 = acc12;	// L420
  v299.write(v317);	// L421
}

void pe_3_1(
  hls::stream< int8_t >& v318,
  hls::stream< int8_t >& v319,
  hls::stream< int8_t >& v320,
  hls::stream< int32_t >& v321
) {	// L424
  int32_t acc13;	// L426
  acc13 = 0;	// L427
  l_S_k_0_k13: for (int k13 = 0; k13 < 4; k13++) {	// L428
  #pragma HLS pipeline II=1
    int8_t v324 = v318.read();	// L429
    int8_t a13;	// L430
    a13 = v324;	// L431
    int8_t v326 = v319.read();	// L432
    int8_t b13;	// L433
    b13 = v326;	// L434
    int8_t v328 = a13;	// L435
    int8_t v329 = b13;	// L436
    int16_t v330 = v328;	// L437
    int16_t v331 = v329;	// L438
    int16_t v332 = v330 * v331;	// L439
    int32_t v333 = acc13;	// L440
    ap_int<33> v334 = v333;	// L441
    ap_int<33> v335 = v332;	// L442
    ap_int<33> v336 = v334 + v335;	// L443
    int32_t v337 = v336;	// L444
    acc13 = v337;	// L445
    int8_t v338 = a13;	// L446
    v320.write(v338);	// L447
  }
  int32_t v339 = acc13;	// L449
  v321.write(v339);	// L450
}

void pe_3_2(
  hls::stream< int8_t >& v340,
  hls::stream< int8_t >& v341,
  hls::stream< int8_t >& v342,
  hls::stream< int32_t >& v343
) {	// L453
  int32_t acc14;	// L455
  acc14 = 0;	// L456
  l_S_k_0_k14: for (int k14 = 0; k14 < 4; k14++) {	// L457
  #pragma HLS pipeline II=1
    int8_t v346 = v340.read();	// L458
    int8_t a14;	// L459
    a14 = v346;	// L460
    int8_t v348 = v341.read();	// L461
    int8_t b14;	// L462
    b14 = v348;	// L463
    int8_t v350 = a14;	// L464
    int8_t v351 = b14;	// L465
    int16_t v352 = v350;	// L466
    int16_t v353 = v351;	// L467
    int16_t v354 = v352 * v353;	// L468
    int32_t v355 = acc14;	// L469
    ap_int<33> v356 = v355;	// L470
    ap_int<33> v357 = v354;	// L471
    ap_int<33> v358 = v356 + v357;	// L472
    int32_t v359 = v358;	// L473
    acc14 = v359;	// L474
    int8_t v360 = a14;	// L475
    v342.write(v360);	// L476
  }
  int32_t v361 = acc14;	// L478
  v343.write(v361);	// L479
}

void pe_3_3(
  hls::stream< int8_t >& v362,
  hls::stream< int8_t >& v363,
  hls::stream< int32_t >& v364
) {	// L482
  int32_t acc15;	// L484
  acc15 = 0;	// L485
  l_S_k_0_k15: for (int k15 = 0; k15 < 4; k15++) {	// L486
  #pragma HLS pipeline II=1
    int8_t v367 = v362.read();	// L487
    int8_t a15;	// L488
    a15 = v367;	// L489
    int8_t v369 = v363.read();	// L490
    int8_t b15;	// L491
    b15 = v369;	// L492
    int8_t v371 = a15;	// L493
    int8_t v372 = b15;	// L494
    int16_t v373 = v371;	// L495
    int16_t v374 = v372;	// L496
    int16_t v375 = v373 * v374;	// L497
    int32_t v376 = acc15;	// L498
    ap_int<33> v377 = v376;	// L499
    ap_int<33> v378 = v375;	// L500
    ap_int<33> v379 = v377 + v378;	// L501
    int32_t v380 = v379;	// L502
    acc15 = v380;	// L503
  }
  int32_t v381 = acc15;	// L505
  v364.write(v381);	// L506
}

void drain_0_0(
  hls::stream< int32_t >& v382,
  hls::stream< int32_t >& v383
) {	// L509
  int32_t v384 = v382.read();	// L511
  v383.write(v384);	// L512
  l_S__i_0__i: for (int _i = 0; _i < 0; _i++) {	// L513
  #pragma HLS pipeline II=1
    v383.write(0);	// L514
  }
}

void drain_0_1(
  hls::stream< int32_t >& v386,
  hls::stream< int32_t >& v387
) {	// L518
  int32_t v388 = v386.read();	// L520
  v387.write(v388);	// L521
  l_S__i_0__i1: for (int _i1 = 0; _i1 < 0; _i1++) {	// L522
  #pragma HLS pipeline II=1
    v387.write(0);	// L523
  }
}

void drain_0_2(
  hls::stream< int32_t >& v390,
  hls::stream< int32_t >& v391
) {	// L527
  int32_t v392 = v390.read();	// L529
  v391.write(v392);	// L530
  l_S__i_0__i2: for (int _i2 = 0; _i2 < 0; _i2++) {	// L531
  #pragma HLS pipeline II=1
    v391.write(0);	// L532
  }
}

void drain_0_3(
  hls::stream< int32_t >& v394,
  hls::stream< int32_t >& v395
) {	// L536
  int32_t v396 = v394.read();	// L538
  v395.write(v396);	// L539
  l_S__i_0__i3: for (int _i3 = 0; _i3 < 0; _i3++) {	// L540
  #pragma HLS pipeline II=1
    v395.write(0);	// L541
  }
}

void drain_1_0(
  hls::stream< int32_t >& v398,
  hls::stream< int32_t >& v399,
  hls::stream< int32_t >& v400
) {	// L545
  int32_t v401 = v398.read();	// L546
  v399.write(v401);	// L547
  l_S__i_0__i4: for (int _i4 = 0; _i4 < 1; _i4++) {	// L548
  #pragma HLS pipeline II=1
    int32_t v403 = v400.read();	// L549
    v399.write(v403);	// L550
  }
}

void drain_1_1(
  hls::stream< int32_t >& v404,
  hls::stream< int32_t >& v405,
  hls::stream< int32_t >& v406
) {	// L554
  int32_t v407 = v404.read();	// L555
  v405.write(v407);	// L556
  l_S__i_0__i5: for (int _i5 = 0; _i5 < 1; _i5++) {	// L557
  #pragma HLS pipeline II=1
    int32_t v409 = v406.read();	// L558
    v405.write(v409);	// L559
  }
}

void drain_1_2(
  hls::stream< int32_t >& v410,
  hls::stream< int32_t >& v411,
  hls::stream< int32_t >& v412
) {	// L563
  int32_t v413 = v410.read();	// L564
  v411.write(v413);	// L565
  l_S__i_0__i6: for (int _i6 = 0; _i6 < 1; _i6++) {	// L566
  #pragma HLS pipeline II=1
    int32_t v415 = v412.read();	// L567
    v411.write(v415);	// L568
  }
}

void drain_1_3(
  hls::stream< int32_t >& v416,
  hls::stream< int32_t >& v417,
  hls::stream< int32_t >& v418
) {	// L572
  int32_t v419 = v416.read();	// L573
  v417.write(v419);	// L574
  l_S__i_0__i7: for (int _i7 = 0; _i7 < 1; _i7++) {	// L575
  #pragma HLS pipeline II=1
    int32_t v421 = v418.read();	// L576
    v417.write(v421);	// L577
  }
}

void drain_2_0(
  hls::stream< int32_t >& v422,
  hls::stream< int32_t >& v423,
  hls::stream< int32_t >& v424
) {	// L581
  int32_t v425 = v422.read();	// L582
  v423.write(v425);	// L583
  l_S__i_0__i8: for (int _i8 = 0; _i8 < 2; _i8++) {	// L584
  #pragma HLS pipeline II=1
    int32_t v427 = v424.read();	// L585
    v423.write(v427);	// L586
  }
}

void drain_2_1(
  hls::stream< int32_t >& v428,
  hls::stream< int32_t >& v429,
  hls::stream< int32_t >& v430
) {	// L590
  int32_t v431 = v428.read();	// L591
  v429.write(v431);	// L592
  l_S__i_0__i9: for (int _i9 = 0; _i9 < 2; _i9++) {	// L593
  #pragma HLS pipeline II=1
    int32_t v433 = v430.read();	// L594
    v429.write(v433);	// L595
  }
}

void drain_2_2(
  hls::stream< int32_t >& v434,
  hls::stream< int32_t >& v435,
  hls::stream< int32_t >& v436
) {	// L599
  int32_t v437 = v434.read();	// L600
  v435.write(v437);	// L601
  l_S__i_0__i10: for (int _i10 = 0; _i10 < 2; _i10++) {	// L602
  #pragma HLS pipeline II=1
    int32_t v439 = v436.read();	// L603
    v435.write(v439);	// L604
  }
}

void drain_2_3(
  hls::stream< int32_t >& v440,
  hls::stream< int32_t >& v441,
  hls::stream< int32_t >& v442
) {	// L608
  int32_t v443 = v440.read();	// L609
  v441.write(v443);	// L610
  l_S__i_0__i11: for (int _i11 = 0; _i11 < 2; _i11++) {	// L611
  #pragma HLS pipeline II=1
    int32_t v445 = v442.read();	// L612
    v441.write(v445);	// L613
  }
}

void drain_3_0(
  hls::stream< int32_t >& v446,
  hls::stream< int32_t >& v447,
  hls::stream< int32_t >& v448
) {	// L617
  int32_t v449 = v446.read();	// L618
  v447.write(v449);	// L619
  l_S__i_0__i12: for (int _i12 = 0; _i12 < 3; _i12++) {	// L620
  #pragma HLS pipeline II=1
    int32_t v451 = v448.read();	// L621
    v447.write(v451);	// L622
  }
}

void drain_3_1(
  hls::stream< int32_t >& v452,
  hls::stream< int32_t >& v453,
  hls::stream< int32_t >& v454
) {	// L626
  int32_t v455 = v452.read();	// L627
  v453.write(v455);	// L628
  l_S__i_0__i13: for (int _i13 = 0; _i13 < 3; _i13++) {	// L629
  #pragma HLS pipeline II=1
    int32_t v457 = v454.read();	// L630
    v453.write(v457);	// L631
  }
}

void drain_3_2(
  hls::stream< int32_t >& v458,
  hls::stream< int32_t >& v459,
  hls::stream< int32_t >& v460
) {	// L635
  int32_t v461 = v458.read();	// L636
  v459.write(v461);	// L637
  l_S__i_0__i14: for (int _i14 = 0; _i14 < 3; _i14++) {	// L638
  #pragma HLS pipeline II=1
    int32_t v463 = v460.read();	// L639
    v459.write(v463);	// L640
  }
}

void drain_3_3(
  hls::stream< int32_t >& v464,
  hls::stream< int32_t >& v465,
  hls::stream< int32_t >& v466
) {	// L644
  int32_t v467 = v464.read();	// L645
  v465.write(v467);	// L646
  l_S__i_0__i15: for (int _i15 = 0; _i15 < 3; _i15++) {	// L647
  #pragma HLS pipeline II=1
    int32_t v469 = v466.read();	// L648
    v465.write(v469);	// L649
  }
}

void feed_0(
  hls::stream< hls::vector< int8_t, 4 > >& v470,
  hls::stream< int8_t >& v471,
  hls::stream< hls::vector< int8_t, 4 > >& v472
) {	// L653
  l_S_k_0_k16: for (int k16 = 0; k16 < 4; k16++) {	// L654
  #pragma HLS pipeline II=1
    int8_t v474[4];
    {
      hls::vector< int8_t, 4 > _vec = v470.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v474[_iv0] = _vec[_iv0];
      }
    }	// L655
    int8_t v475 = v474[0];	// L656
    v471.write(v475);	// L657
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v474[_iv0];
      }
      v472.write(_vec);
    }	// L658
  }
}

void feed_1(
  hls::stream< hls::vector< int8_t, 4 > >& v476,
  hls::stream< int8_t >& v477,
  hls::stream< hls::vector< int8_t, 4 > >& v478
) {	// L662
  l_S_k_0_k17: for (int k17 = 0; k17 < 4; k17++) {	// L663
  #pragma HLS pipeline II=1
    int8_t v480[4];
    {
      hls::vector< int8_t, 4 > _vec = v476.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v480[_iv0] = _vec[_iv0];
      }
    }	// L664
    int8_t v481 = v480[1];	// L665
    v477.write(v481);	// L666
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v480[_iv0];
      }
      v478.write(_vec);
    }	// L667
  }
}

void feed_2(
  hls::stream< hls::vector< int8_t, 4 > >& v482,
  hls::stream< int8_t >& v483,
  hls::stream< hls::vector< int8_t, 4 > >& v484
) {	// L671
  l_S_k_0_k18: for (int k18 = 0; k18 < 4; k18++) {	// L672
  #pragma HLS pipeline II=1
    int8_t v486[4];
    {
      hls::vector< int8_t, 4 > _vec = v482.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v486[_iv0] = _vec[_iv0];
      }
    }	// L673
    int8_t v487 = v486[2];	// L674
    v483.write(v487);	// L675
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v486[_iv0];
      }
      v484.write(_vec);
    }	// L676
  }
}

void feed_3(
  hls::stream< hls::vector< int8_t, 4 > >& v488,
  hls::stream< int8_t >& v489
) {	// L680
  l_S_k_0_k19: for (int k19 = 0; k19 < 4; k19++) {	// L681
  #pragma HLS pipeline II=1
    int8_t v491[4];
    {
      hls::vector< int8_t, 4 > _vec = v488.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v491[_iv0] = _vec[_iv0];
      }
    }	// L682
    int8_t v492 = v491[3];	// L683
    v489.write(v492);	// L684
  }
}

void feed_3_0(
  hls::stream< hls::vector< int8_t, 4 > >& v493,
  hls::stream< int8_t >& v494,
  hls::stream< hls::vector< int8_t, 4 > >& v495
) {	// L688
  l_S_k_0_k20: for (int k20 = 0; k20 < 4; k20++) {	// L689
  #pragma HLS pipeline II=1
    int8_t v497[4];
    {
      hls::vector< int8_t, 4 > _vec = v493.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v497[_iv0] = _vec[_iv0];
      }
    }	// L690
    int8_t v498 = v497[0];	// L691
    v494.write(v498);	// L692
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v497[_iv0];
      }
      v495.write(_vec);
    }	// L693
  }
}

void feed_3_1(
  hls::stream< hls::vector< int8_t, 4 > >& v499,
  hls::stream< int8_t >& v500,
  hls::stream< hls::vector< int8_t, 4 > >& v501
) {	// L697
  l_S_k_0_k21: for (int k21 = 0; k21 < 4; k21++) {	// L698
  #pragma HLS pipeline II=1
    int8_t v503[4];
    {
      hls::vector< int8_t, 4 > _vec = v499.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v503[_iv0] = _vec[_iv0];
      }
    }	// L699
    int8_t v504 = v503[1];	// L700
    v500.write(v504);	// L701
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v503[_iv0];
      }
      v501.write(_vec);
    }	// L702
  }
}

void feed_3_2(
  hls::stream< hls::vector< int8_t, 4 > >& v505,
  hls::stream< int8_t >& v506,
  hls::stream< hls::vector< int8_t, 4 > >& v507
) {	// L706
  l_S_k_0_k22: for (int k22 = 0; k22 < 4; k22++) {	// L707
  #pragma HLS pipeline II=1
    int8_t v509[4];
    {
      hls::vector< int8_t, 4 > _vec = v505.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v509[_iv0] = _vec[_iv0];
      }
    }	// L708
    int8_t v510 = v509[2];	// L709
    v506.write(v510);	// L710
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v509[_iv0];
      }
      v507.write(_vec);
    }	// L711
  }
}

void feed_3_3(
  hls::stream< hls::vector< int8_t, 4 > >& v511,
  hls::stream< int8_t >& v512
) {	// L715
  l_S_k_0_k23: for (int k23 = 0; k23 < 4; k23++) {	// L716
  #pragma HLS pipeline II=1
    int8_t v514[4];
    {
      hls::vector< int8_t, 4 > _vec = v511.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v514[_iv0] = _vec[_iv0];
      }
    }	// L717
    int8_t v515 = v514[3];	// L718
    v512.write(v515);	// L719
  }
}

void drain_down_drain_0(
  int32_t v516[4][4],
  hls::stream< int32_t >& v517
) {	// L723
  #pragma HLS array_partition variable=v516 complete dim=1
  #pragma HLS array_partition variable=v516 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L724
  #pragma HLS pipeline II=1
    int32_t v519 = v517.read();	// L725
    v516[0][_t2] = v519;	// L726
  }
}

void drain_down_drain_1(
  int32_t v520[4][4],
  hls::stream< int32_t >& v521
) {	// L730
  #pragma HLS array_partition variable=v520 complete dim=1
  #pragma HLS array_partition variable=v520 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 4; _t3++) {	// L731
  #pragma HLS pipeline II=1
    int32_t v523 = v521.read();	// L732
    v520[1][_t3] = v523;	// L733
  }
}

void drain_down_drain_2(
  int32_t v524[4][4],
  hls::stream< int32_t >& v525
) {	// L737
  #pragma HLS array_partition variable=v524 complete dim=1
  #pragma HLS array_partition variable=v524 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 4; _t4++) {	// L738
  #pragma HLS pipeline II=1
    int32_t v527 = v525.read();	// L739
    v524[2][_t4] = v527;	// L740
  }
}

void drain_down_drain_3(
  int32_t v528[4][4],
  hls::stream< int32_t >& v529
) {	// L744
  #pragma HLS array_partition variable=v528 complete dim=1
  #pragma HLS array_partition variable=v528 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 4; _t5++) {	// L745
  #pragma HLS pipeline II=1
    int32_t v531 = v529.read();	// L746
    v528[3][_t5] = v531;	// L747
  }
}

/// This is top function.
void top(
  int8_t v532[4][4],
  int8_t v533[4][4],
  int32_t v534[4][4]
) {	// L751
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v532 complete dim=1
  #pragma HLS array_partition variable=v532 complete dim=2

  #pragma HLS array_partition variable=v533 complete dim=1
  #pragma HLS array_partition variable=v533 complete dim=2

  #pragma HLS array_partition variable=v534 complete dim=1
  #pragma HLS array_partition variable=v534 complete dim=2

  hls::stream< int8_t > v535;
  #pragma HLS stream variable=v535 depth=2	// L752
  hls::stream< int8_t > v536;
  #pragma HLS stream variable=v536 depth=2	// L753
  hls::stream< int8_t > v537;
  #pragma HLS stream variable=v537 depth=2	// L754
  hls::stream< int8_t > v538;
  #pragma HLS stream variable=v538 depth=2	// L755
  hls::stream< int8_t > v539;
  #pragma HLS stream variable=v539 depth=2	// L756
  hls::stream< int8_t > v540;
  #pragma HLS stream variable=v540 depth=2	// L757
  hls::stream< int8_t > v541;
  #pragma HLS stream variable=v541 depth=2	// L758
  hls::stream< int8_t > v542;
  #pragma HLS stream variable=v542 depth=2	// L759
  hls::stream< int8_t > v543;
  #pragma HLS stream variable=v543 depth=2	// L760
  hls::stream< int8_t > v544;
  #pragma HLS stream variable=v544 depth=2	// L761
  hls::stream< int8_t > v545;
  #pragma HLS stream variable=v545 depth=2	// L762
  hls::stream< int8_t > v546;
  #pragma HLS stream variable=v546 depth=2	// L763
  hls::stream< int8_t > v547;
  #pragma HLS stream variable=v547 depth=2	// L764
  hls::stream< int8_t > v548;
  #pragma HLS stream variable=v548 depth=2	// L765
  hls::stream< int8_t > v549;
  #pragma HLS stream variable=v549 depth=2	// L766
  hls::stream< int8_t > v550;
  #pragma HLS stream variable=v550 depth=2	// L767
  hls::stream< int8_t > v551;
  #pragma HLS stream variable=v551 depth=2	// L768
  hls::stream< int8_t > v552;
  #pragma HLS stream variable=v552 depth=2	// L769
  hls::stream< int8_t > v553;
  #pragma HLS stream variable=v553 depth=2	// L770
  hls::stream< int8_t > v554;
  #pragma HLS stream variable=v554 depth=2	// L771
  hls::stream< int8_t > v555;
  #pragma HLS stream variable=v555 depth=2	// L772
  hls::stream< int8_t > v556;
  #pragma HLS stream variable=v556 depth=2	// L773
  hls::stream< int8_t > v557;
  #pragma HLS stream variable=v557 depth=2	// L774
  hls::stream< int8_t > v558;
  #pragma HLS stream variable=v558 depth=2	// L775
  hls::stream< int8_t > v559;
  #pragma HLS stream variable=v559 depth=2	// L776
  hls::stream< int8_t > v560;
  #pragma HLS stream variable=v560 depth=2	// L777
  hls::stream< int8_t > v561;
  #pragma HLS stream variable=v561 depth=2	// L778
  hls::stream< int8_t > v562;
  #pragma HLS stream variable=v562 depth=2	// L779
  hls::stream< int8_t > v563;
  #pragma HLS stream variable=v563 depth=2	// L780
  hls::stream< int8_t > v564;
  #pragma HLS stream variable=v564 depth=2	// L781
  hls::stream< int8_t > v565;
  #pragma HLS stream variable=v565 depth=2	// L782
  hls::stream< int8_t > v566;
  #pragma HLS stream variable=v566 depth=2	// L783
  hls::stream< int32_t > v567;
  #pragma HLS stream variable=v567 depth=2	// L784
  hls::stream< int32_t > v568;
  #pragma HLS stream variable=v568 depth=2	// L785
  hls::stream< int32_t > v569;
  #pragma HLS stream variable=v569 depth=2	// L786
  hls::stream< int32_t > v570;
  #pragma HLS stream variable=v570 depth=2	// L787
  hls::stream< int32_t > v571;
  #pragma HLS stream variable=v571 depth=2	// L788
  hls::stream< int32_t > v572;
  #pragma HLS stream variable=v572 depth=2	// L789
  hls::stream< int32_t > v573;
  #pragma HLS stream variable=v573 depth=2	// L790
  hls::stream< int32_t > v574;
  #pragma HLS stream variable=v574 depth=2	// L791
  hls::stream< int32_t > v575;
  #pragma HLS stream variable=v575 depth=2	// L792
  hls::stream< int32_t > v576;
  #pragma HLS stream variable=v576 depth=2	// L793
  hls::stream< int32_t > v577;
  #pragma HLS stream variable=v577 depth=2	// L794
  hls::stream< int32_t > v578;
  #pragma HLS stream variable=v578 depth=2	// L795
  hls::stream< int32_t > v579;
  #pragma HLS stream variable=v579 depth=2	// L796
  hls::stream< int32_t > v580;
  #pragma HLS stream variable=v580 depth=2	// L797
  hls::stream< int32_t > v581;
  #pragma HLS stream variable=v581 depth=2	// L798
  hls::stream< int32_t > v582;
  #pragma HLS stream variable=v582 depth=2	// L799
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v583;
  #pragma HLS stream variable=v583 depth=2	// L800
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v584;
  #pragma HLS stream variable=v584 depth=2	// L801
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v585;
  #pragma HLS stream variable=v585 depth=2	// L802
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v586;
  #pragma HLS stream variable=v586 depth=2	// L803
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v587;
  #pragma HLS stream variable=v587 depth=2	// L804
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v588;
  #pragma HLS stream variable=v588 depth=2	// L805
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v589;
  #pragma HLS stream variable=v589 depth=2	// L806
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v590;
  #pragma HLS stream variable=v590 depth=2	// L807
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v591;
  #pragma HLS stream variable=v591 depth=2	// L808
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v592;
  #pragma HLS stream variable=v592 depth=2	// L809
  hls::stream< int8_t > v593;
  #pragma HLS stream variable=v593 depth=2	// L810
  hls::stream< int8_t > v594;
  #pragma HLS stream variable=v594 depth=2	// L811
  hls::stream< int8_t > v595;
  #pragma HLS stream variable=v595 depth=2	// L812
  hls::stream< int8_t > v596;
  #pragma HLS stream variable=v596 depth=2	// L813
  hls::stream< int8_t > v597;
  #pragma HLS stream variable=v597 depth=2	// L814
  hls::stream< int8_t > v598;
  #pragma HLS stream variable=v598 depth=2	// L815
  hls::stream< int8_t > v599;
  #pragma HLS stream variable=v599 depth=2	// L816
  hls::stream< int8_t > v600;
  #pragma HLS stream variable=v600 depth=2	// L817
  hls::stream< int32_t > v601;
  #pragma HLS stream variable=v601 depth=2	// L818
  hls::stream< int32_t > v602;
  #pragma HLS stream variable=v602 depth=2	// L819
  hls::stream< int32_t > v603;
  #pragma HLS stream variable=v603 depth=2	// L820
  hls::stream< int32_t > v604;
  #pragma HLS stream variable=v604 depth=2	// L821
  hls::stream< int32_t > v605;
  #pragma HLS stream variable=v605 depth=2	// L822
  hls::stream< int32_t > v606;
  #pragma HLS stream variable=v606 depth=2	// L823
  hls::stream< int32_t > v607;
  #pragma HLS stream variable=v607 depth=2	// L824
  hls::stream< int32_t > v608;
  #pragma HLS stream variable=v608 depth=2	// L825
  hls::stream< int32_t > v609;
  #pragma HLS stream variable=v609 depth=2	// L826
  hls::stream< int32_t > v610;
  #pragma HLS stream variable=v610 depth=2	// L827
  hls::stream< int32_t > v611;
  #pragma HLS stream variable=v611 depth=2	// L828
  hls::stream< int32_t > v612;
  #pragma HLS stream variable=v612 depth=2	// L829
  hls::stream< int32_t > v613;
  #pragma HLS stream variable=v613 depth=2	// L830
  hls::stream< int32_t > v614;
  #pragma HLS stream variable=v614 depth=2	// L831
  hls::stream< int32_t > v615;
  #pragma HLS stream variable=v615 depth=2	// L832
  hls::stream< int32_t > v616;
  #pragma HLS stream variable=v616 depth=2	// L833
  hls::stream< int32_t > v617;
  #pragma HLS stream variable=v617 depth=2	// L834
  hls::stream< int32_t > v618;
  #pragma HLS stream variable=v618 depth=2	// L835
  hls::stream< int32_t > v619;
  #pragma HLS stream variable=v619 depth=2	// L836
  hls::stream< int32_t > v620;
  #pragma HLS stream variable=v620 depth=2	// L837
  feed_up_load_0(v532, v591);	// L838
  feed_3_up_load_0(v533, v592);	// L839
  pe_0_0(v593, v597, v536, v555, v601);	// L840
  pe_0_1(v536, v598, v537, v556, v602);	// L841
  pe_0_2(v537, v599, v538, v557, v603);	// L842
  pe_0_3(v538, v600, v558, v604);	// L843
  pe_1_0(v594, v555, v540, v559, v605);	// L844
  pe_1_1(v540, v556, v541, v560, v606);	// L845
  pe_1_2(v541, v557, v542, v561, v607);	// L846
  pe_1_3(v542, v558, v562, v608);	// L847
  pe_2_0(v595, v559, v544, v563, v609);	// L848
  pe_2_1(v544, v560, v545, v564, v610);	// L849
  pe_2_2(v545, v561, v546, v565, v611);	// L850
  pe_2_3(v546, v562, v566, v612);	// L851
  pe_3_0(v596, v563, v548, v613);	// L852
  pe_3_1(v548, v564, v549, v614);	// L853
  pe_3_2(v549, v565, v550, v615);	// L854
  pe_3_3(v550, v566, v616);	// L855
  drain_0_0(v601, v571);	// L856
  drain_0_1(v602, v572);	// L857
  drain_0_2(v603, v573);	// L858
  drain_0_3(v604, v574);	// L859
  drain_1_0(v605, v575, v571);	// L860
  drain_1_1(v606, v576, v572);	// L861
  drain_1_2(v607, v577, v573);	// L862
  drain_1_3(v608, v578, v574);	// L863
  drain_2_0(v609, v579, v575);	// L864
  drain_2_1(v610, v580, v576);	// L865
  drain_2_2(v611, v581, v577);	// L866
  drain_2_3(v612, v582, v578);	// L867
  drain_3_0(v613, v617, v579);	// L868
  drain_3_1(v614, v618, v580);	// L869
  drain_3_2(v615, v619, v581);	// L870
  drain_3_3(v616, v620, v582);	// L871
  feed_0(v591, v593, v584);	// L872
  feed_1(v584, v594, v585);	// L873
  feed_2(v585, v595, v586);	// L874
  feed_3(v586, v596);	// L875
  feed_3_0(v592, v597, v588);	// L876
  feed_3_1(v588, v598, v589);	// L877
  feed_3_2(v589, v599, v590);	// L878
  feed_3_3(v590, v600);	// L879
  drain_down_drain_0(v534, v617);	// L880
  drain_down_drain_1(v534, v618);	// L881
  drain_down_drain_2(v534, v619);	// L882
  drain_down_drain_3(v534, v620);	// L883
}


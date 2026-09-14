
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
void feed_up_load(
  int8_t v0[4][4],
  int v1,
  hls::stream< hls::vector< int8_t, 4 > >& v2
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L4
    int8_t _blk[4];	// L7
    for (int v5 = 0; v5 < 4; v5++) {	// L8
      _blk[v5] = 0;	// L8
    }
    l_S__b0_0__b0: for (int _b0 = 0; _b0 < 4; _b0++) {	// L9
    #pragma HLS pipeline II=1
      int8_t v7 = v0[_t][_b0];	// L10
      _blk[_b0] = v7;	// L11
    }
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = _blk[_iv0];
      }
      v2.write(_vec);
    }	// L13
  }
}

void feed_2_up_load(
  int8_t v8[4][4],
  int v9,
  hls::stream< hls::vector< int8_t, 4 > >& v10
) {	// L17
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 4; _t1++) {	// L18
    int8_t _blk1[4];	// L21
    for (int v13 = 0; v13 < 4; v13++) {	// L22
      _blk1[v13] = 0;	// L22
    }
    l_S__b0_0__b01: for (int _b01 = 0; _b01 < 4; _b01++) {	// L23
    #pragma HLS pipeline II=1
      int8_t v15 = v8[_t1][_b01];	// L24
      _blk1[_b01] = v15;	// L25
    }
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = _blk1[_iv0];
      }
      v10.write(_vec);
    }	// L27
  }
}

void pe_r0(
  int v16,
  int v17,
  hls::stream< int32_t >& v18,
  hls::stream< int8_t >& v19,
  hls::stream< int8_t >& v20,
  hls::stream< int8_t >& v21,
  hls::stream< int8_t >& v22
) {	// L31
  int32_t acc;	// L33
  acc = 0;	// L34
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L35
  #pragma HLS pipeline II=1
    int8_t v25 = v22.read();	// L36
    int8_t a;	// L37
    a = v25;	// L38
    int8_t v27 = v20.read();	// L39
    int8_t b;	// L40
    b = v27;	// L41
    int8_t v29 = a;	// L42
    int8_t v30 = b;	// L43
    int16_t v31 = v29;	// L44
    int16_t v32 = v30;	// L45
    int16_t v33 = v31 * v32;	// L46
    int32_t v34 = acc;	// L47
    ap_int<33> v35 = v34;	// L48
    ap_int<33> v36 = v33;	// L49
    ap_int<33> v37 = v35 + v36;	// L50
    int32_t v38 = v37;	// L51
    acc = v38;	// L52
    int8_t v39 = a;	// L53
    v19.write(v39);	// L54
    int8_t v40 = b;	// L55
    v21.write(v40);	// L56
  }
  int32_t v41 = acc;	// L58
  v18.write(v41);	// L59
  l_S__i_1__i: for (int _i = 0; _i < 0; _i++) {	// L60
  #pragma HLS pipeline II=1
    v18.write(0);	// L61
  }
}

void pe_r1(
  int v43,
  int v44,
  hls::stream< int32_t >& v45,
  hls::stream< int32_t >& v46,
  hls::stream< int8_t >& v47,
  hls::stream< int8_t >& v48,
  hls::stream< int8_t >& v49,
  hls::stream< int8_t >& v50
) {	// L65
  int32_t acc1;	// L67
  acc1 = 0;	// L68
  l_S_k_0_k1: for (int k1 = 0; k1 < 4; k1++) {	// L69
  #pragma HLS pipeline II=1
    int8_t v53 = v50.read();	// L70
    int8_t a1;	// L71
    a1 = v53;	// L72
    int8_t v55 = v48.read();	// L73
    int8_t b1;	// L74
    b1 = v55;	// L75
    int8_t v57 = a1;	// L76
    int8_t v58 = b1;	// L77
    int16_t v59 = v57;	// L78
    int16_t v60 = v58;	// L79
    int16_t v61 = v59 * v60;	// L80
    int32_t v62 = acc1;	// L81
    ap_int<33> v63 = v62;	// L82
    ap_int<33> v64 = v61;	// L83
    ap_int<33> v65 = v63 + v64;	// L84
    int32_t v66 = v65;	// L85
    acc1 = v66;	// L86
    int8_t v67 = a1;	// L87
    v47.write(v67);	// L88
    int8_t v68 = b1;	// L89
    v49.write(v68);	// L90
  }
  int32_t v69 = acc1;	// L92
  v46.write(v69);	// L93
  l_S__i_1__i1: for (int _i1 = 0; _i1 < 1; _i1++) {	// L94
  #pragma HLS pipeline II=1
    int32_t v71 = v45.read();	// L95
    v46.write(v71);	// L96
  }
}

void pe_r2(
  int v72,
  int v73,
  hls::stream< int32_t >& v74,
  hls::stream< int32_t >& v75,
  hls::stream< int8_t >& v76,
  hls::stream< int8_t >& v77,
  hls::stream< int8_t >& v78,
  hls::stream< int8_t >& v79
) {	// L100
  int32_t acc2;	// L102
  acc2 = 0;	// L103
  l_S_k_0_k2: for (int k2 = 0; k2 < 4; k2++) {	// L104
  #pragma HLS pipeline II=1
    int8_t v82 = v79.read();	// L105
    int8_t a2;	// L106
    a2 = v82;	// L107
    int8_t v84 = v77.read();	// L108
    int8_t b2;	// L109
    b2 = v84;	// L110
    int8_t v86 = a2;	// L111
    int8_t v87 = b2;	// L112
    int16_t v88 = v86;	// L113
    int16_t v89 = v87;	// L114
    int16_t v90 = v88 * v89;	// L115
    int32_t v91 = acc2;	// L116
    ap_int<33> v92 = v91;	// L117
    ap_int<33> v93 = v90;	// L118
    ap_int<33> v94 = v92 + v93;	// L119
    int32_t v95 = v94;	// L120
    acc2 = v95;	// L121
    int8_t v96 = a2;	// L122
    v76.write(v96);	// L123
    int8_t v97 = b2;	// L124
    v78.write(v97);	// L125
  }
  int32_t v98 = acc2;	// L127
  v75.write(v98);	// L128
  l_S__i_1__i2: for (int _i2 = 0; _i2 < 2; _i2++) {	// L129
  #pragma HLS pipeline II=1
    int32_t v100 = v74.read();	// L130
    v75.write(v100);	// L131
  }
}

void pe_r3(
  int v101,
  int v102,
  hls::stream< int32_t >& v103,
  hls::stream< int32_t >& v104,
  hls::stream< int8_t >& v105,
  hls::stream< int8_t >& v106,
  hls::stream< int8_t >& v107
) {	// L135
  int32_t acc3;	// L137
  acc3 = 0;	// L138
  l_S_k_0_k3: for (int k3 = 0; k3 < 4; k3++) {	// L139
  #pragma HLS pipeline II=1
    int8_t v110 = v107.read();	// L140
    int8_t a3;	// L141
    a3 = v110;	// L142
    int8_t v112 = v106.read();	// L143
    int8_t b3;	// L144
    b3 = v112;	// L145
    int8_t v114 = a3;	// L146
    int8_t v115 = b3;	// L147
    int16_t v116 = v114;	// L148
    int16_t v117 = v115;	// L149
    int16_t v118 = v116 * v117;	// L150
    int32_t v119 = acc3;	// L151
    ap_int<33> v120 = v119;	// L152
    ap_int<33> v121 = v118;	// L153
    ap_int<33> v122 = v120 + v121;	// L154
    int32_t v123 = v122;	// L155
    acc3 = v123;	// L156
    int8_t v124 = a3;	// L157
    v105.write(v124);	// L158
  }
  int32_t v125 = acc3;	// L160
  v104.write(v125);	// L161
  l_S__i_1__i3: for (int _i3 = 0; _i3 < 3; _i3++) {	// L162
  #pragma HLS pipeline II=1
    int32_t v127 = v103.read();	// L163
    v104.write(v127);	// L164
  }
}

void pe_r4(
  int v128,
  int v129,
  hls::stream< int32_t >& v130,
  hls::stream< int8_t >& v131,
  hls::stream< int8_t >& v132,
  hls::stream< int8_t >& v133,
  hls::stream< int8_t >& v134
) {	// L168
  int32_t acc4;	// L170
  acc4 = 0;	// L171
  l_S_k_0_k4: for (int k4 = 0; k4 < 4; k4++) {	// L172
  #pragma HLS pipeline II=1
    int8_t v137 = v134.read();	// L173
    int8_t a4;	// L174
    a4 = v137;	// L175
    int8_t v139 = v132.read();	// L176
    int8_t b4;	// L177
    b4 = v139;	// L178
    int8_t v141 = a4;	// L179
    int8_t v142 = b4;	// L180
    int16_t v143 = v141;	// L181
    int16_t v144 = v142;	// L182
    int16_t v145 = v143 * v144;	// L183
    int32_t v146 = acc4;	// L184
    ap_int<33> v147 = v146;	// L185
    ap_int<33> v148 = v145;	// L186
    ap_int<33> v149 = v147 + v148;	// L187
    int32_t v150 = v149;	// L188
    acc4 = v150;	// L189
    int8_t v151 = a4;	// L190
    v131.write(v151);	// L191
    int8_t v152 = b4;	// L192
    v133.write(v152);	// L193
  }
  int32_t v153 = acc4;	// L195
  v130.write(v153);	// L196
  l_S__i_1__i4: for (int _i4 = 0; _i4 < 0; _i4++) {	// L197
  #pragma HLS pipeline II=1
    v130.write(0);	// L198
  }
}

void pe_r5(
  int v155,
  int v156,
  hls::stream< int32_t >& v157,
  hls::stream< int8_t >& v158,
  hls::stream< int8_t >& v159,
  hls::stream< int8_t >& v160
) {	// L202
  int32_t acc5;	// L204
  acc5 = 0;	// L205
  l_S_k_0_k5: for (int k5 = 0; k5 < 4; k5++) {	// L206
  #pragma HLS pipeline II=1
    int8_t v163 = v160.read();	// L207
    int8_t a5;	// L208
    a5 = v163;	// L209
    int8_t v165 = v158.read();	// L210
    int8_t b5;	// L211
    b5 = v165;	// L212
    int8_t v167 = a5;	// L213
    int8_t v168 = b5;	// L214
    int16_t v169 = v167;	// L215
    int16_t v170 = v168;	// L216
    int16_t v171 = v169 * v170;	// L217
    int32_t v172 = acc5;	// L218
    ap_int<33> v173 = v172;	// L219
    ap_int<33> v174 = v171;	// L220
    ap_int<33> v175 = v173 + v174;	// L221
    int32_t v176 = v175;	// L222
    acc5 = v176;	// L223
    int8_t v177 = b5;	// L224
    v159.write(v177);	// L225
  }
  int32_t v178 = acc5;	// L227
  v157.write(v178);	// L228
  l_S__i_1__i5: for (int _i5 = 0; _i5 < 0; _i5++) {	// L229
  #pragma HLS pipeline II=1
    v157.write(0);	// L230
  }
}

void pe_r6(
  int v180,
  int v181,
  hls::stream< int32_t >& v182,
  hls::stream< int32_t >& v183,
  hls::stream< int8_t >& v184,
  hls::stream< int8_t >& v185,
  hls::stream< int8_t >& v186,
  hls::stream< int8_t >& v187
) {	// L234
  int32_t acc6;	// L236
  acc6 = 0;	// L237
  l_S_k_0_k6: for (int k6 = 0; k6 < 4; k6++) {	// L238
  #pragma HLS pipeline II=1
    int8_t v190 = v187.read();	// L239
    int8_t a6;	// L240
    a6 = v190;	// L241
    int8_t v192 = v185.read();	// L242
    int8_t b6;	// L243
    b6 = v192;	// L244
    int8_t v194 = a6;	// L245
    int8_t v195 = b6;	// L246
    int16_t v196 = v194;	// L247
    int16_t v197 = v195;	// L248
    int16_t v198 = v196 * v197;	// L249
    int32_t v199 = acc6;	// L250
    ap_int<33> v200 = v199;	// L251
    ap_int<33> v201 = v198;	// L252
    ap_int<33> v202 = v200 + v201;	// L253
    int32_t v203 = v202;	// L254
    acc6 = v203;	// L255
    int8_t v204 = a6;	// L256
    v184.write(v204);	// L257
    int8_t v205 = b6;	// L258
    v186.write(v205);	// L259
  }
  int32_t v206 = acc6;	// L261
  v183.write(v206);	// L262
  l_S__i_1__i6: for (int _i6 = 0; _i6 < 1; _i6++) {	// L263
  #pragma HLS pipeline II=1
    int32_t v208 = v182.read();	// L264
    v183.write(v208);	// L265
  }
}

void pe_r7(
  int v209,
  int v210,
  hls::stream< int32_t >& v211,
  hls::stream< int32_t >& v212,
  hls::stream< int8_t >& v213,
  hls::stream< int8_t >& v214,
  hls::stream< int8_t >& v215
) {	// L269
  int32_t acc7;	// L271
  acc7 = 0;	// L272
  l_S_k_0_k7: for (int k7 = 0; k7 < 4; k7++) {	// L273
  #pragma HLS pipeline II=1
    int8_t v218 = v215.read();	// L274
    int8_t a7;	// L275
    a7 = v218;	// L276
    int8_t v220 = v213.read();	// L277
    int8_t b7;	// L278
    b7 = v220;	// L279
    int8_t v222 = a7;	// L280
    int8_t v223 = b7;	// L281
    int16_t v224 = v222;	// L282
    int16_t v225 = v223;	// L283
    int16_t v226 = v224 * v225;	// L284
    int32_t v227 = acc7;	// L285
    ap_int<33> v228 = v227;	// L286
    ap_int<33> v229 = v226;	// L287
    ap_int<33> v230 = v228 + v229;	// L288
    int32_t v231 = v230;	// L289
    acc7 = v231;	// L290
    int8_t v232 = b7;	// L291
    v214.write(v232);	// L292
  }
  int32_t v233 = acc7;	// L294
  v212.write(v233);	// L295
  l_S__i_1__i7: for (int _i7 = 0; _i7 < 1; _i7++) {	// L296
  #pragma HLS pipeline II=1
    int32_t v235 = v211.read();	// L297
    v212.write(v235);	// L298
  }
}

void pe_r8(
  int v236,
  int v237,
  hls::stream< int32_t >& v238,
  hls::stream< int32_t >& v239,
  hls::stream< int8_t >& v240,
  hls::stream< int8_t >& v241,
  hls::stream< int8_t >& v242,
  hls::stream< int8_t >& v243
) {	// L302
  int32_t acc8;	// L304
  acc8 = 0;	// L305
  l_S_k_0_k8: for (int k8 = 0; k8 < 4; k8++) {	// L306
  #pragma HLS pipeline II=1
    int8_t v246 = v243.read();	// L307
    int8_t a8;	// L308
    a8 = v246;	// L309
    int8_t v248 = v241.read();	// L310
    int8_t b8;	// L311
    b8 = v248;	// L312
    int8_t v250 = a8;	// L313
    int8_t v251 = b8;	// L314
    int16_t v252 = v250;	// L315
    int16_t v253 = v251;	// L316
    int16_t v254 = v252 * v253;	// L317
    int32_t v255 = acc8;	// L318
    ap_int<33> v256 = v255;	// L319
    ap_int<33> v257 = v254;	// L320
    ap_int<33> v258 = v256 + v257;	// L321
    int32_t v259 = v258;	// L322
    acc8 = v259;	// L323
    int8_t v260 = a8;	// L324
    v240.write(v260);	// L325
    int8_t v261 = b8;	// L326
    v242.write(v261);	// L327
  }
  int32_t v262 = acc8;	// L329
  v239.write(v262);	// L330
  l_S__i_1__i8: for (int _i8 = 0; _i8 < 2; _i8++) {	// L331
  #pragma HLS pipeline II=1
    int32_t v264 = v238.read();	// L332
    v239.write(v264);	// L333
  }
}

void pe_r9(
  int v265,
  int v266,
  hls::stream< int32_t >& v267,
  hls::stream< int32_t >& v268,
  hls::stream< int8_t >& v269,
  hls::stream< int8_t >& v270,
  hls::stream< int8_t >& v271
) {	// L337
  int32_t acc9;	// L339
  acc9 = 0;	// L340
  l_S_k_0_k9: for (int k9 = 0; k9 < 4; k9++) {	// L341
  #pragma HLS pipeline II=1
    int8_t v274 = v271.read();	// L342
    int8_t a9;	// L343
    a9 = v274;	// L344
    int8_t v276 = v269.read();	// L345
    int8_t b9;	// L346
    b9 = v276;	// L347
    int8_t v278 = a9;	// L348
    int8_t v279 = b9;	// L349
    int16_t v280 = v278;	// L350
    int16_t v281 = v279;	// L351
    int16_t v282 = v280 * v281;	// L352
    int32_t v283 = acc9;	// L353
    ap_int<33> v284 = v283;	// L354
    ap_int<33> v285 = v282;	// L355
    ap_int<33> v286 = v284 + v285;	// L356
    int32_t v287 = v286;	// L357
    acc9 = v287;	// L358
    int8_t v288 = b9;	// L359
    v270.write(v288);	// L360
  }
  int32_t v289 = acc9;	// L362
  v268.write(v289);	// L363
  l_S__i_1__i9: for (int _i9 = 0; _i9 < 2; _i9++) {	// L364
  #pragma HLS pipeline II=1
    int32_t v291 = v267.read();	// L365
    v268.write(v291);	// L366
  }
}

void pe_r10(
  int v292,
  int v293,
  hls::stream< int32_t >& v294,
  hls::stream< int32_t >& v295,
  hls::stream< int8_t >& v296,
  hls::stream< int8_t >& v297,
  hls::stream< int8_t >& v298
) {	// L370
  int32_t acc10;	// L372
  acc10 = 0;	// L373
  l_S_k_0_k10: for (int k10 = 0; k10 < 4; k10++) {	// L374
  #pragma HLS pipeline II=1
    int8_t v301 = v298.read();	// L375
    int8_t a10;	// L376
    a10 = v301;	// L377
    int8_t v303 = v297.read();	// L378
    int8_t b10;	// L379
    b10 = v303;	// L380
    int8_t v305 = a10;	// L381
    int8_t v306 = b10;	// L382
    int16_t v307 = v305;	// L383
    int16_t v308 = v306;	// L384
    int16_t v309 = v307 * v308;	// L385
    int32_t v310 = acc10;	// L386
    ap_int<33> v311 = v310;	// L387
    ap_int<33> v312 = v309;	// L388
    ap_int<33> v313 = v311 + v312;	// L389
    int32_t v314 = v313;	// L390
    acc10 = v314;	// L391
    int8_t v315 = a10;	// L392
    v296.write(v315);	// L393
  }
  int32_t v316 = acc10;	// L395
  v295.write(v316);	// L396
  l_S__i_1__i10: for (int _i10 = 0; _i10 < 3; _i10++) {	// L397
  #pragma HLS pipeline II=1
    int32_t v318 = v294.read();	// L398
    v295.write(v318);	// L399
  }
}

void pe_r11(
  int v319,
  int v320,
  hls::stream< int32_t >& v321,
  hls::stream< int32_t >& v322,
  hls::stream< int8_t >& v323,
  hls::stream< int8_t >& v324
) {	// L403
  int32_t acc11;	// L405
  acc11 = 0;	// L406
  l_S_k_0_k11: for (int k11 = 0; k11 < 4; k11++) {	// L407
  #pragma HLS pipeline II=1
    int8_t v327 = v324.read();	// L408
    int8_t a11;	// L409
    a11 = v327;	// L410
    int8_t v329 = v323.read();	// L411
    int8_t b11;	// L412
    b11 = v329;	// L413
    int8_t v331 = a11;	// L414
    int8_t v332 = b11;	// L415
    int16_t v333 = v331;	// L416
    int16_t v334 = v332;	// L417
    int16_t v335 = v333 * v334;	// L418
    int32_t v336 = acc11;	// L419
    ap_int<33> v337 = v336;	// L420
    ap_int<33> v338 = v335;	// L421
    ap_int<33> v339 = v337 + v338;	// L422
    int32_t v340 = v339;	// L423
    acc11 = v340;	// L424
  }
  int32_t v341 = acc11;	// L426
  v322.write(v341);	// L427
  l_S__i_1__i11: for (int _i11 = 0; _i11 < 3; _i11++) {	// L428
  #pragma HLS pipeline II=1
    int32_t v343 = v321.read();	// L429
    v322.write(v343);	// L430
  }
}

void feed_r0(
  int v344,
  hls::stream< hls::vector< int8_t, 4 > >& v345,
  hls::stream< int8_t >& v346,
  hls::stream< hls::vector< int8_t, 4 > >& v347
) {	// L434
  l_S_k_0_k12: for (int k12 = 0; k12 < 4; k12++) {	// L435
  #pragma HLS pipeline II=1
    int8_t v349[4];
    {
      hls::vector< int8_t, 4 > _vec = v347.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v349[_iv0] = _vec[_iv0];
      }
    }	// L436
    int8_t v350 = v349[v344];	// L437
    v346.write(v350);	// L438
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v349[_iv0];
      }
      v345.write(_vec);
    }	// L439
  }
}

void feed_r1(
  int v351,
  hls::stream< hls::vector< int8_t, 4 > >& v352,
  hls::stream< int8_t >& v353,
  hls::stream< hls::vector< int8_t, 4 > >& v354
) {	// L443
  l_S_k_0_k13: for (int k13 = 0; k13 < 4; k13++) {	// L444
  #pragma HLS pipeline II=1
    int8_t v356[4];
    {
      hls::vector< int8_t, 4 > _vec = v354.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v356[_iv0] = _vec[_iv0];
      }
    }	// L445
    int8_t v357 = v356[v351];	// L446
    v353.write(v357);	// L447
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v356[_iv0];
      }
      v352.write(_vec);
    }	// L448
  }
}

void feed_r2(
  int v358,
  hls::stream< int8_t >& v359,
  hls::stream< hls::vector< int8_t, 4 > >& v360
) {	// L452
  l_S_k_0_k14: for (int k14 = 0; k14 < 4; k14++) {	// L453
  #pragma HLS pipeline II=1
    int8_t v362[4];
    {
      hls::vector< int8_t, 4 > _vec = v360.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v362[_iv0] = _vec[_iv0];
      }
    }	// L454
    int8_t v363 = v362[v358];	// L455
    v359.write(v363);	// L456
  }
}

void feed_2_r0(
  int v364,
  hls::stream< hls::vector< int8_t, 4 > >& v365,
  hls::stream< int8_t >& v366,
  hls::stream< hls::vector< int8_t, 4 > >& v367
) {	// L460
  l_S_k_0_k15: for (int k15 = 0; k15 < 4; k15++) {	// L461
  #pragma HLS pipeline II=1
    int8_t v369[4];
    {
      hls::vector< int8_t, 4 > _vec = v367.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v369[_iv0] = _vec[_iv0];
      }
    }	// L462
    int8_t v370 = v369[v364];	// L463
    v366.write(v370);	// L464
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v369[_iv0];
      }
      v365.write(_vec);
    }	// L465
  }
}

void feed_2_r1(
  int v371,
  hls::stream< hls::vector< int8_t, 4 > >& v372,
  hls::stream< int8_t >& v373,
  hls::stream< hls::vector< int8_t, 4 > >& v374
) {	// L469
  l_S_k_0_k16: for (int k16 = 0; k16 < 4; k16++) {	// L470
  #pragma HLS pipeline II=1
    int8_t v376[4];
    {
      hls::vector< int8_t, 4 > _vec = v374.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v376[_iv0] = _vec[_iv0];
      }
    }	// L471
    int8_t v377 = v376[v371];	// L472
    v373.write(v377);	// L473
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v376[_iv0];
      }
      v372.write(_vec);
    }	// L474
  }
}

void feed_2_r2(
  int v378,
  hls::stream< int8_t >& v379,
  hls::stream< hls::vector< int8_t, 4 > >& v380
) {	// L478
  l_S_k_0_k17: for (int k17 = 0; k17 < 4; k17++) {	// L479
  #pragma HLS pipeline II=1
    int8_t v382[4];
    {
      hls::vector< int8_t, 4 > _vec = v380.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v382[_iv0] = _vec[_iv0];
      }
    }	// L480
    int8_t v383 = v382[v378];	// L481
    v379.write(v383);	// L482
  }
}

void pe_c_out_drain(
  int32_t v384[4][4],
  int v385,
  hls::stream< int32_t >& v386
) {	// L486
  #pragma HLS array_partition variable=v384 complete dim=1
  #pragma HLS array_partition variable=v384 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L487
  #pragma HLS pipeline II=1
    int32_t v388 = v386.read();	// L488
    v384[v385][_t2] = v388;	// L489
  }
}

/// This is top function.
void top(
  int8_t v389[4][4],
  int8_t v390[4][4],
  int32_t v391[4][4]
) {	// L493
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v389 complete dim=1
  #pragma HLS array_partition variable=v389 complete dim=2

  #pragma HLS array_partition variable=v390 complete dim=1
  #pragma HLS array_partition variable=v390 complete dim=2

  #pragma HLS array_partition variable=v391 complete dim=1
  #pragma HLS array_partition variable=v391 complete dim=2

  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v392;
  #pragma HLS stream variable=v392 depth=2	// L494
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v393;
  #pragma HLS stream variable=v393 depth=2	// L495
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v394;
  #pragma HLS stream variable=v394 depth=2	// L496
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v395;
  #pragma HLS stream variable=v395 depth=2	// L497
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v396;
  #pragma HLS stream variable=v396 depth=2	// L498
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v397;
  #pragma HLS stream variable=v397 depth=2	// L499
  hls::stream< int32_t > v398;
  #pragma HLS stream variable=v398 depth=4	// L500
  hls::stream< int8_t > v399;
  #pragma HLS stream variable=v399 depth=2	// L501
  hls::stream< int32_t > v400;
  #pragma HLS stream variable=v400 depth=4	// L502
  hls::stream< int8_t > v401;
  #pragma HLS stream variable=v401 depth=2	// L503
  hls::stream< int32_t > v402;
  #pragma HLS stream variable=v402 depth=4	// L504
  hls::stream< int8_t > v403;
  #pragma HLS stream variable=v403 depth=2	// L505
  hls::stream< int8_t > v404;
  #pragma HLS stream variable=v404 depth=2	// L506
  hls::stream< int32_t > v405;
  #pragma HLS stream variable=v405 depth=4	// L507
  hls::stream< int8_t > v406;
  #pragma HLS stream variable=v406 depth=2	// L508
  hls::stream< int32_t > v407;
  #pragma HLS stream variable=v407 depth=2	// L509
  hls::stream< int8_t > v408;
  #pragma HLS stream variable=v408 depth=2	// L510
  hls::stream< int8_t > v409;
  #pragma HLS stream variable=v409 depth=2	// L511
  hls::stream< int32_t > v410;
  #pragma HLS stream variable=v410 depth=2	// L512
  hls::stream< int8_t > v411;
  #pragma HLS stream variable=v411 depth=2	// L513
  hls::stream< int8_t > v412;
  #pragma HLS stream variable=v412 depth=2	// L514
  hls::stream< int32_t > v413;
  #pragma HLS stream variable=v413 depth=2	// L515
  hls::stream< int8_t > v414;
  #pragma HLS stream variable=v414 depth=2	// L516
  hls::stream< int8_t > v415;
  #pragma HLS stream variable=v415 depth=2	// L517
  hls::stream< int8_t > v416;
  #pragma HLS stream variable=v416 depth=2	// L518
  hls::stream< int32_t > v417;
  #pragma HLS stream variable=v417 depth=2	// L519
  hls::stream< int8_t > v418;
  #pragma HLS stream variable=v418 depth=2	// L520
  hls::stream< int32_t > v419;
  #pragma HLS stream variable=v419 depth=2	// L521
  hls::stream< int8_t > v420;
  #pragma HLS stream variable=v420 depth=2	// L522
  hls::stream< int8_t > v421;
  #pragma HLS stream variable=v421 depth=2	// L523
  hls::stream< int32_t > v422;
  #pragma HLS stream variable=v422 depth=2	// L524
  hls::stream< int8_t > v423;
  #pragma HLS stream variable=v423 depth=2	// L525
  hls::stream< int8_t > v424;
  #pragma HLS stream variable=v424 depth=2	// L526
  hls::stream< int32_t > v425;
  #pragma HLS stream variable=v425 depth=2	// L527
  hls::stream< int8_t > v426;
  #pragma HLS stream variable=v426 depth=2	// L528
  hls::stream< int8_t > v427;
  #pragma HLS stream variable=v427 depth=2	// L529
  hls::stream< int8_t > v428;
  #pragma HLS stream variable=v428 depth=2	// L530
  hls::stream< int32_t > v429;
  #pragma HLS stream variable=v429 depth=2	// L531
  hls::stream< int8_t > v430;
  #pragma HLS stream variable=v430 depth=2	// L532
  hls::stream< int8_t > v431;
  #pragma HLS stream variable=v431 depth=2	// L533
  hls::stream< int32_t > v432;
  #pragma HLS stream variable=v432 depth=2	// L534
  hls::stream< int8_t > v433;
  #pragma HLS stream variable=v433 depth=2	// L536
  hls::stream< int8_t > v434;
  #pragma HLS stream variable=v434 depth=2	// L537
  hls::stream< int8_t > v435;
  #pragma HLS stream variable=v435 depth=2	// L538
  hls::stream< int32_t > v436;
  #pragma HLS stream variable=v436 depth=2	// L539
  hls::stream< int8_t > v437;
  #pragma HLS stream variable=v437 depth=2	// L541
  hls::stream< int8_t > v438;
  #pragma HLS stream variable=v438 depth=2	// L542
  hls::stream< int8_t > v439;
  #pragma HLS stream variable=v439 depth=2	// L543
  hls::stream< int32_t > v440;
  #pragma HLS stream variable=v440 depth=2	// L544
  hls::stream< int8_t > v441;
  #pragma HLS stream variable=v441 depth=2	// L546
  hls::stream< int8_t > v442;
  #pragma HLS stream variable=v442 depth=2	// L547
  hls::stream< int8_t > v443;
  #pragma HLS stream variable=v443 depth=2	// L548
  hls::stream< int8_t > v444;
  #pragma HLS stream variable=v444 depth=2	// L549
  hls::stream< int32_t > v445;
  #pragma HLS stream variable=v445 depth=2	// L550
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v446;
  #pragma HLS stream variable=v446 depth=16	// L551
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v447;
  #pragma HLS stream variable=v447 depth=16	// L552
  feed_up_load(v389, 0, v447);	// L554
  feed_2_up_load(v390, 0, v446);	// L555
  pe_r4(0, 0, v445, v444, v443, v442, v441);	// L556
  pe_r0(0, 1, v440, v439, v438, v437, v444);	// L557
  pe_r0(0, 2, v436, v435, v434, v433, v439);	// L558
  pe_r5(0, 3, v432, v431, v430, v435);	// L559
  pe_r6(1, 0, v445, v429, v428, v442, v427, v426);	// L560
  pe_r1(1, 1, v440, v425, v424, v437, v423, v428);	// L561
  pe_r1(1, 2, v436, v422, v421, v433, v420, v424);	// L562
  pe_r7(1, 3, v432, v419, v430, v418, v421);	// L563
  pe_r8(2, 0, v429, v417, v416, v427, v415, v414);	// L564
  pe_r2(2, 1, v425, v413, v412, v423, v411, v416);	// L565
  pe_r2(2, 2, v422, v410, v409, v420, v408, v412);	// L566
  pe_r9(2, 3, v419, v407, v418, v406, v409);	// L567
  pe_r10(3, 0, v417, v405, v404, v415, v403);	// L568
  pe_r3(3, 1, v413, v402, v401, v411, v404);	// L569
  pe_r3(3, 2, v410, v400, v399, v408, v401);	// L570
  pe_r11(3, 3, v407, v398, v406, v399);	// L571
  feed_r1(0, v397, v441, v447);	// L572
  feed_r0(1, v396, v426, v397);	// L573
  feed_r0(2, v395, v414, v396);	// L574
  feed_r2(3, v403, v395);	// L575
  feed_2_r1(0, v394, v443, v446);	// L576
  feed_2_r0(1, v393, v438, v394);	// L577
  feed_2_r0(2, v392, v434, v393);	// L578
  feed_2_r2(3, v431, v392);	// L579
  pe_c_out_drain(v391, 0, v405);	// L580
  pe_c_out_drain(v391, 1, v402);	// L581
  pe_c_out_drain(v391, 2, v400);	// L582
  pe_c_out_drain(v391, 3, v398);	// L583
}


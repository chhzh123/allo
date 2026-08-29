
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

void feed_2_up_load_0(
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
  l_S__i_1__i: for (int _i = 0; _i < 0; _i++) {	// L58
  #pragma HLS pipeline II=1
    v18.write(0);	// L59
  }
}

void pe_0_1(
  hls::stream< int8_t >& v39,
  hls::stream< int8_t >& v40,
  hls::stream< int8_t >& v41,
  hls::stream< int8_t >& v42,
  hls::stream< int32_t >& v43
) {	// L63
  int32_t acc1;	// L65
  acc1 = 0;	// L66
  l_S_k_0_k1: for (int k1 = 0; k1 < 4; k1++) {	// L67
  #pragma HLS pipeline II=1
    int8_t v46 = v39.read();	// L68
    int8_t a1;	// L69
    a1 = v46;	// L70
    int8_t v48 = v40.read();	// L71
    int8_t b1;	// L72
    b1 = v48;	// L73
    int8_t v50 = a1;	// L74
    int8_t v51 = b1;	// L75
    int16_t v52 = v50;	// L76
    int16_t v53 = v51;	// L77
    int16_t v54 = v52 * v53;	// L78
    int32_t v55 = acc1;	// L79
    ap_int<33> v56 = v55;	// L80
    ap_int<33> v57 = v54;	// L81
    ap_int<33> v58 = v56 + v57;	// L82
    int32_t v59 = v58;	// L83
    acc1 = v59;	// L84
    int8_t v60 = a1;	// L85
    v41.write(v60);	// L86
    int8_t v61 = b1;	// L87
    v42.write(v61);	// L88
  }
  int32_t v62 = acc1;	// L90
  v43.write(v62);	// L91
  l_S__i_1__i1: for (int _i1 = 0; _i1 < 0; _i1++) {	// L92
  #pragma HLS pipeline II=1
    v43.write(0);	// L93
  }
}

void pe_0_2(
  hls::stream< int8_t >& v64,
  hls::stream< int8_t >& v65,
  hls::stream< int8_t >& v66,
  hls::stream< int8_t >& v67,
  hls::stream< int32_t >& v68
) {	// L97
  int32_t acc2;	// L99
  acc2 = 0;	// L100
  l_S_k_0_k2: for (int k2 = 0; k2 < 4; k2++) {	// L101
  #pragma HLS pipeline II=1
    int8_t v71 = v64.read();	// L102
    int8_t a2;	// L103
    a2 = v71;	// L104
    int8_t v73 = v65.read();	// L105
    int8_t b2;	// L106
    b2 = v73;	// L107
    int8_t v75 = a2;	// L108
    int8_t v76 = b2;	// L109
    int16_t v77 = v75;	// L110
    int16_t v78 = v76;	// L111
    int16_t v79 = v77 * v78;	// L112
    int32_t v80 = acc2;	// L113
    ap_int<33> v81 = v80;	// L114
    ap_int<33> v82 = v79;	// L115
    ap_int<33> v83 = v81 + v82;	// L116
    int32_t v84 = v83;	// L117
    acc2 = v84;	// L118
    int8_t v85 = a2;	// L119
    v66.write(v85);	// L120
    int8_t v86 = b2;	// L121
    v67.write(v86);	// L122
  }
  int32_t v87 = acc2;	// L124
  v68.write(v87);	// L125
  l_S__i_1__i2: for (int _i2 = 0; _i2 < 0; _i2++) {	// L126
  #pragma HLS pipeline II=1
    v68.write(0);	// L127
  }
}

void pe_0_3(
  hls::stream< int8_t >& v89,
  hls::stream< int8_t >& v90,
  hls::stream< int8_t >& v91,
  hls::stream< int32_t >& v92
) {	// L131
  int32_t acc3;	// L133
  acc3 = 0;	// L134
  l_S_k_0_k3: for (int k3 = 0; k3 < 4; k3++) {	// L135
  #pragma HLS pipeline II=1
    int8_t v95 = v89.read();	// L136
    int8_t a3;	// L137
    a3 = v95;	// L138
    int8_t v97 = v90.read();	// L139
    int8_t b3;	// L140
    b3 = v97;	// L141
    int8_t v99 = a3;	// L142
    int8_t v100 = b3;	// L143
    int16_t v101 = v99;	// L144
    int16_t v102 = v100;	// L145
    int16_t v103 = v101 * v102;	// L146
    int32_t v104 = acc3;	// L147
    ap_int<33> v105 = v104;	// L148
    ap_int<33> v106 = v103;	// L149
    ap_int<33> v107 = v105 + v106;	// L150
    int32_t v108 = v107;	// L151
    acc3 = v108;	// L152
    int8_t v109 = b3;	// L153
    v91.write(v109);	// L154
  }
  int32_t v110 = acc3;	// L156
  v92.write(v110);	// L157
  l_S__i_1__i3: for (int _i3 = 0; _i3 < 0; _i3++) {	// L158
  #pragma HLS pipeline II=1
    v92.write(0);	// L159
  }
}

void pe_1_0(
  hls::stream< int8_t >& v112,
  hls::stream< int8_t >& v113,
  hls::stream< int8_t >& v114,
  hls::stream< int8_t >& v115,
  hls::stream< int32_t >& v116,
  hls::stream< int32_t >& v117
) {	// L163
  int32_t acc4;	// L165
  acc4 = 0;	// L166
  l_S_k_0_k4: for (int k4 = 0; k4 < 4; k4++) {	// L167
  #pragma HLS pipeline II=1
    int8_t v120 = v112.read();	// L168
    int8_t a4;	// L169
    a4 = v120;	// L170
    int8_t v122 = v113.read();	// L171
    int8_t b4;	// L172
    b4 = v122;	// L173
    int8_t v124 = a4;	// L174
    int8_t v125 = b4;	// L175
    int16_t v126 = v124;	// L176
    int16_t v127 = v125;	// L177
    int16_t v128 = v126 * v127;	// L178
    int32_t v129 = acc4;	// L179
    ap_int<33> v130 = v129;	// L180
    ap_int<33> v131 = v128;	// L181
    ap_int<33> v132 = v130 + v131;	// L182
    int32_t v133 = v132;	// L183
    acc4 = v133;	// L184
    int8_t v134 = a4;	// L185
    v114.write(v134);	// L186
    int8_t v135 = b4;	// L187
    v115.write(v135);	// L188
  }
  int32_t v136 = acc4;	// L190
  v116.write(v136);	// L191
  l_S__i_1__i4: for (int _i4 = 0; _i4 < 1; _i4++) {	// L192
  #pragma HLS pipeline II=1
    int32_t v138 = v117.read();	// L193
    v116.write(v138);	// L194
  }
}

void pe_1_1(
  hls::stream< int8_t >& v139,
  hls::stream< int8_t >& v140,
  hls::stream< int8_t >& v141,
  hls::stream< int8_t >& v142,
  hls::stream< int32_t >& v143,
  hls::stream< int32_t >& v144
) {	// L198
  int32_t acc5;	// L200
  acc5 = 0;	// L201
  l_S_k_0_k5: for (int k5 = 0; k5 < 4; k5++) {	// L202
  #pragma HLS pipeline II=1
    int8_t v147 = v139.read();	// L203
    int8_t a5;	// L204
    a5 = v147;	// L205
    int8_t v149 = v140.read();	// L206
    int8_t b5;	// L207
    b5 = v149;	// L208
    int8_t v151 = a5;	// L209
    int8_t v152 = b5;	// L210
    int16_t v153 = v151;	// L211
    int16_t v154 = v152;	// L212
    int16_t v155 = v153 * v154;	// L213
    int32_t v156 = acc5;	// L214
    ap_int<33> v157 = v156;	// L215
    ap_int<33> v158 = v155;	// L216
    ap_int<33> v159 = v157 + v158;	// L217
    int32_t v160 = v159;	// L218
    acc5 = v160;	// L219
    int8_t v161 = a5;	// L220
    v141.write(v161);	// L221
    int8_t v162 = b5;	// L222
    v142.write(v162);	// L223
  }
  int32_t v163 = acc5;	// L225
  v143.write(v163);	// L226
  l_S__i_1__i5: for (int _i5 = 0; _i5 < 1; _i5++) {	// L227
  #pragma HLS pipeline II=1
    int32_t v165 = v144.read();	// L228
    v143.write(v165);	// L229
  }
}

void pe_1_2(
  hls::stream< int8_t >& v166,
  hls::stream< int8_t >& v167,
  hls::stream< int8_t >& v168,
  hls::stream< int8_t >& v169,
  hls::stream< int32_t >& v170,
  hls::stream< int32_t >& v171
) {	// L233
  int32_t acc6;	// L235
  acc6 = 0;	// L236
  l_S_k_0_k6: for (int k6 = 0; k6 < 4; k6++) {	// L237
  #pragma HLS pipeline II=1
    int8_t v174 = v166.read();	// L238
    int8_t a6;	// L239
    a6 = v174;	// L240
    int8_t v176 = v167.read();	// L241
    int8_t b6;	// L242
    b6 = v176;	// L243
    int8_t v178 = a6;	// L244
    int8_t v179 = b6;	// L245
    int16_t v180 = v178;	// L246
    int16_t v181 = v179;	// L247
    int16_t v182 = v180 * v181;	// L248
    int32_t v183 = acc6;	// L249
    ap_int<33> v184 = v183;	// L250
    ap_int<33> v185 = v182;	// L251
    ap_int<33> v186 = v184 + v185;	// L252
    int32_t v187 = v186;	// L253
    acc6 = v187;	// L254
    int8_t v188 = a6;	// L255
    v168.write(v188);	// L256
    int8_t v189 = b6;	// L257
    v169.write(v189);	// L258
  }
  int32_t v190 = acc6;	// L260
  v170.write(v190);	// L261
  l_S__i_1__i6: for (int _i6 = 0; _i6 < 1; _i6++) {	// L262
  #pragma HLS pipeline II=1
    int32_t v192 = v171.read();	// L263
    v170.write(v192);	// L264
  }
}

void pe_1_3(
  hls::stream< int8_t >& v193,
  hls::stream< int8_t >& v194,
  hls::stream< int8_t >& v195,
  hls::stream< int32_t >& v196,
  hls::stream< int32_t >& v197
) {	// L268
  int32_t acc7;	// L270
  acc7 = 0;	// L271
  l_S_k_0_k7: for (int k7 = 0; k7 < 4; k7++) {	// L272
  #pragma HLS pipeline II=1
    int8_t v200 = v193.read();	// L273
    int8_t a7;	// L274
    a7 = v200;	// L275
    int8_t v202 = v194.read();	// L276
    int8_t b7;	// L277
    b7 = v202;	// L278
    int8_t v204 = a7;	// L279
    int8_t v205 = b7;	// L280
    int16_t v206 = v204;	// L281
    int16_t v207 = v205;	// L282
    int16_t v208 = v206 * v207;	// L283
    int32_t v209 = acc7;	// L284
    ap_int<33> v210 = v209;	// L285
    ap_int<33> v211 = v208;	// L286
    ap_int<33> v212 = v210 + v211;	// L287
    int32_t v213 = v212;	// L288
    acc7 = v213;	// L289
    int8_t v214 = b7;	// L290
    v195.write(v214);	// L291
  }
  int32_t v215 = acc7;	// L293
  v196.write(v215);	// L294
  l_S__i_1__i7: for (int _i7 = 0; _i7 < 1; _i7++) {	// L295
  #pragma HLS pipeline II=1
    int32_t v217 = v197.read();	// L296
    v196.write(v217);	// L297
  }
}

void pe_2_0(
  hls::stream< int8_t >& v218,
  hls::stream< int8_t >& v219,
  hls::stream< int8_t >& v220,
  hls::stream< int8_t >& v221,
  hls::stream< int32_t >& v222,
  hls::stream< int32_t >& v223
) {	// L301
  int32_t acc8;	// L303
  acc8 = 0;	// L304
  l_S_k_0_k8: for (int k8 = 0; k8 < 4; k8++) {	// L305
  #pragma HLS pipeline II=1
    int8_t v226 = v218.read();	// L306
    int8_t a8;	// L307
    a8 = v226;	// L308
    int8_t v228 = v219.read();	// L309
    int8_t b8;	// L310
    b8 = v228;	// L311
    int8_t v230 = a8;	// L312
    int8_t v231 = b8;	// L313
    int16_t v232 = v230;	// L314
    int16_t v233 = v231;	// L315
    int16_t v234 = v232 * v233;	// L316
    int32_t v235 = acc8;	// L317
    ap_int<33> v236 = v235;	// L318
    ap_int<33> v237 = v234;	// L319
    ap_int<33> v238 = v236 + v237;	// L320
    int32_t v239 = v238;	// L321
    acc8 = v239;	// L322
    int8_t v240 = a8;	// L323
    v220.write(v240);	// L324
    int8_t v241 = b8;	// L325
    v221.write(v241);	// L326
  }
  int32_t v242 = acc8;	// L328
  v222.write(v242);	// L329
  l_S__i_1__i8: for (int _i8 = 0; _i8 < 2; _i8++) {	// L330
  #pragma HLS pipeline II=1
    int32_t v244 = v223.read();	// L331
    v222.write(v244);	// L332
  }
}

void pe_2_1(
  hls::stream< int8_t >& v245,
  hls::stream< int8_t >& v246,
  hls::stream< int8_t >& v247,
  hls::stream< int8_t >& v248,
  hls::stream< int32_t >& v249,
  hls::stream< int32_t >& v250
) {	// L336
  int32_t acc9;	// L338
  acc9 = 0;	// L339
  l_S_k_0_k9: for (int k9 = 0; k9 < 4; k9++) {	// L340
  #pragma HLS pipeline II=1
    int8_t v253 = v245.read();	// L341
    int8_t a9;	// L342
    a9 = v253;	// L343
    int8_t v255 = v246.read();	// L344
    int8_t b9;	// L345
    b9 = v255;	// L346
    int8_t v257 = a9;	// L347
    int8_t v258 = b9;	// L348
    int16_t v259 = v257;	// L349
    int16_t v260 = v258;	// L350
    int16_t v261 = v259 * v260;	// L351
    int32_t v262 = acc9;	// L352
    ap_int<33> v263 = v262;	// L353
    ap_int<33> v264 = v261;	// L354
    ap_int<33> v265 = v263 + v264;	// L355
    int32_t v266 = v265;	// L356
    acc9 = v266;	// L357
    int8_t v267 = a9;	// L358
    v247.write(v267);	// L359
    int8_t v268 = b9;	// L360
    v248.write(v268);	// L361
  }
  int32_t v269 = acc9;	// L363
  v249.write(v269);	// L364
  l_S__i_1__i9: for (int _i9 = 0; _i9 < 2; _i9++) {	// L365
  #pragma HLS pipeline II=1
    int32_t v271 = v250.read();	// L366
    v249.write(v271);	// L367
  }
}

void pe_2_2(
  hls::stream< int8_t >& v272,
  hls::stream< int8_t >& v273,
  hls::stream< int8_t >& v274,
  hls::stream< int8_t >& v275,
  hls::stream< int32_t >& v276,
  hls::stream< int32_t >& v277
) {	// L371
  int32_t acc10;	// L373
  acc10 = 0;	// L374
  l_S_k_0_k10: for (int k10 = 0; k10 < 4; k10++) {	// L375
  #pragma HLS pipeline II=1
    int8_t v280 = v272.read();	// L376
    int8_t a10;	// L377
    a10 = v280;	// L378
    int8_t v282 = v273.read();	// L379
    int8_t b10;	// L380
    b10 = v282;	// L381
    int8_t v284 = a10;	// L382
    int8_t v285 = b10;	// L383
    int16_t v286 = v284;	// L384
    int16_t v287 = v285;	// L385
    int16_t v288 = v286 * v287;	// L386
    int32_t v289 = acc10;	// L387
    ap_int<33> v290 = v289;	// L388
    ap_int<33> v291 = v288;	// L389
    ap_int<33> v292 = v290 + v291;	// L390
    int32_t v293 = v292;	// L391
    acc10 = v293;	// L392
    int8_t v294 = a10;	// L393
    v274.write(v294);	// L394
    int8_t v295 = b10;	// L395
    v275.write(v295);	// L396
  }
  int32_t v296 = acc10;	// L398
  v276.write(v296);	// L399
  l_S__i_1__i10: for (int _i10 = 0; _i10 < 2; _i10++) {	// L400
  #pragma HLS pipeline II=1
    int32_t v298 = v277.read();	// L401
    v276.write(v298);	// L402
  }
}

void pe_2_3(
  hls::stream< int8_t >& v299,
  hls::stream< int8_t >& v300,
  hls::stream< int8_t >& v301,
  hls::stream< int32_t >& v302,
  hls::stream< int32_t >& v303
) {	// L406
  int32_t acc11;	// L408
  acc11 = 0;	// L409
  l_S_k_0_k11: for (int k11 = 0; k11 < 4; k11++) {	// L410
  #pragma HLS pipeline II=1
    int8_t v306 = v299.read();	// L411
    int8_t a11;	// L412
    a11 = v306;	// L413
    int8_t v308 = v300.read();	// L414
    int8_t b11;	// L415
    b11 = v308;	// L416
    int8_t v310 = a11;	// L417
    int8_t v311 = b11;	// L418
    int16_t v312 = v310;	// L419
    int16_t v313 = v311;	// L420
    int16_t v314 = v312 * v313;	// L421
    int32_t v315 = acc11;	// L422
    ap_int<33> v316 = v315;	// L423
    ap_int<33> v317 = v314;	// L424
    ap_int<33> v318 = v316 + v317;	// L425
    int32_t v319 = v318;	// L426
    acc11 = v319;	// L427
    int8_t v320 = b11;	// L428
    v301.write(v320);	// L429
  }
  int32_t v321 = acc11;	// L431
  v302.write(v321);	// L432
  l_S__i_1__i11: for (int _i11 = 0; _i11 < 2; _i11++) {	// L433
  #pragma HLS pipeline II=1
    int32_t v323 = v303.read();	// L434
    v302.write(v323);	// L435
  }
}

void pe_3_0(
  hls::stream< int8_t >& v324,
  hls::stream< int8_t >& v325,
  hls::stream< int8_t >& v326,
  hls::stream< int32_t >& v327,
  hls::stream< int32_t >& v328
) {	// L439
  int32_t acc12;	// L441
  acc12 = 0;	// L442
  l_S_k_0_k12: for (int k12 = 0; k12 < 4; k12++) {	// L443
  #pragma HLS pipeline II=1
    int8_t v331 = v324.read();	// L444
    int8_t a12;	// L445
    a12 = v331;	// L446
    int8_t v333 = v325.read();	// L447
    int8_t b12;	// L448
    b12 = v333;	// L449
    int8_t v335 = a12;	// L450
    int8_t v336 = b12;	// L451
    int16_t v337 = v335;	// L452
    int16_t v338 = v336;	// L453
    int16_t v339 = v337 * v338;	// L454
    int32_t v340 = acc12;	// L455
    ap_int<33> v341 = v340;	// L456
    ap_int<33> v342 = v339;	// L457
    ap_int<33> v343 = v341 + v342;	// L458
    int32_t v344 = v343;	// L459
    acc12 = v344;	// L460
    int8_t v345 = a12;	// L461
    v326.write(v345);	// L462
  }
  int32_t v346 = acc12;	// L464
  v327.write(v346);	// L465
  l_S__i_1__i12: for (int _i12 = 0; _i12 < 3; _i12++) {	// L466
  #pragma HLS pipeline II=1
    int32_t v348 = v328.read();	// L467
    v327.write(v348);	// L468
  }
}

void pe_3_1(
  hls::stream< int8_t >& v349,
  hls::stream< int8_t >& v350,
  hls::stream< int8_t >& v351,
  hls::stream< int32_t >& v352,
  hls::stream< int32_t >& v353
) {	// L472
  int32_t acc13;	// L474
  acc13 = 0;	// L475
  l_S_k_0_k13: for (int k13 = 0; k13 < 4; k13++) {	// L476
  #pragma HLS pipeline II=1
    int8_t v356 = v349.read();	// L477
    int8_t a13;	// L478
    a13 = v356;	// L479
    int8_t v358 = v350.read();	// L480
    int8_t b13;	// L481
    b13 = v358;	// L482
    int8_t v360 = a13;	// L483
    int8_t v361 = b13;	// L484
    int16_t v362 = v360;	// L485
    int16_t v363 = v361;	// L486
    int16_t v364 = v362 * v363;	// L487
    int32_t v365 = acc13;	// L488
    ap_int<33> v366 = v365;	// L489
    ap_int<33> v367 = v364;	// L490
    ap_int<33> v368 = v366 + v367;	// L491
    int32_t v369 = v368;	// L492
    acc13 = v369;	// L493
    int8_t v370 = a13;	// L494
    v351.write(v370);	// L495
  }
  int32_t v371 = acc13;	// L497
  v352.write(v371);	// L498
  l_S__i_1__i13: for (int _i13 = 0; _i13 < 3; _i13++) {	// L499
  #pragma HLS pipeline II=1
    int32_t v373 = v353.read();	// L500
    v352.write(v373);	// L501
  }
}

void pe_3_2(
  hls::stream< int8_t >& v374,
  hls::stream< int8_t >& v375,
  hls::stream< int8_t >& v376,
  hls::stream< int32_t >& v377,
  hls::stream< int32_t >& v378
) {	// L505
  int32_t acc14;	// L507
  acc14 = 0;	// L508
  l_S_k_0_k14: for (int k14 = 0; k14 < 4; k14++) {	// L509
  #pragma HLS pipeline II=1
    int8_t v381 = v374.read();	// L510
    int8_t a14;	// L511
    a14 = v381;	// L512
    int8_t v383 = v375.read();	// L513
    int8_t b14;	// L514
    b14 = v383;	// L515
    int8_t v385 = a14;	// L516
    int8_t v386 = b14;	// L517
    int16_t v387 = v385;	// L518
    int16_t v388 = v386;	// L519
    int16_t v389 = v387 * v388;	// L520
    int32_t v390 = acc14;	// L521
    ap_int<33> v391 = v390;	// L522
    ap_int<33> v392 = v389;	// L523
    ap_int<33> v393 = v391 + v392;	// L524
    int32_t v394 = v393;	// L525
    acc14 = v394;	// L526
    int8_t v395 = a14;	// L527
    v376.write(v395);	// L528
  }
  int32_t v396 = acc14;	// L530
  v377.write(v396);	// L531
  l_S__i_1__i14: for (int _i14 = 0; _i14 < 3; _i14++) {	// L532
  #pragma HLS pipeline II=1
    int32_t v398 = v378.read();	// L533
    v377.write(v398);	// L534
  }
}

void pe_3_3(
  hls::stream< int8_t >& v399,
  hls::stream< int8_t >& v400,
  hls::stream< int32_t >& v401,
  hls::stream< int32_t >& v402
) {	// L538
  int32_t acc15;	// L540
  acc15 = 0;	// L541
  l_S_k_0_k15: for (int k15 = 0; k15 < 4; k15++) {	// L542
  #pragma HLS pipeline II=1
    int8_t v405 = v399.read();	// L543
    int8_t a15;	// L544
    a15 = v405;	// L545
    int8_t v407 = v400.read();	// L546
    int8_t b15;	// L547
    b15 = v407;	// L548
    int8_t v409 = a15;	// L549
    int8_t v410 = b15;	// L550
    int16_t v411 = v409;	// L551
    int16_t v412 = v410;	// L552
    int16_t v413 = v411 * v412;	// L553
    int32_t v414 = acc15;	// L554
    ap_int<33> v415 = v414;	// L555
    ap_int<33> v416 = v413;	// L556
    ap_int<33> v417 = v415 + v416;	// L557
    int32_t v418 = v417;	// L558
    acc15 = v418;	// L559
  }
  int32_t v419 = acc15;	// L561
  v401.write(v419);	// L562
  l_S__i_1__i15: for (int _i15 = 0; _i15 < 3; _i15++) {	// L563
  #pragma HLS pipeline II=1
    int32_t v421 = v402.read();	// L564
    v401.write(v421);	// L565
  }
}

void feed_0(
  hls::stream< hls::vector< int8_t, 4 > >& v422,
  hls::stream< int8_t >& v423,
  hls::stream< hls::vector< int8_t, 4 > >& v424
) {	// L569
  l_S_k_0_k16: for (int k16 = 0; k16 < 4; k16++) {	// L570
  #pragma HLS pipeline II=1
    int8_t v426[4];
    {
      hls::vector< int8_t, 4 > _vec = v422.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v426[_iv0] = _vec[_iv0];
      }
    }	// L571
    int8_t v427 = v426[0];	// L572
    v423.write(v427);	// L573
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v426[_iv0];
      }
      v424.write(_vec);
    }	// L574
  }
}

void feed_1(
  hls::stream< hls::vector< int8_t, 4 > >& v428,
  hls::stream< int8_t >& v429,
  hls::stream< hls::vector< int8_t, 4 > >& v430
) {	// L578
  l_S_k_0_k17: for (int k17 = 0; k17 < 4; k17++) {	// L579
  #pragma HLS pipeline II=1
    int8_t v432[4];
    {
      hls::vector< int8_t, 4 > _vec = v428.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v432[_iv0] = _vec[_iv0];
      }
    }	// L580
    int8_t v433 = v432[1];	// L581
    v429.write(v433);	// L582
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v432[_iv0];
      }
      v430.write(_vec);
    }	// L583
  }
}

void feed_2(
  hls::stream< hls::vector< int8_t, 4 > >& v434,
  hls::stream< int8_t >& v435,
  hls::stream< hls::vector< int8_t, 4 > >& v436
) {	// L587
  l_S_k_0_k18: for (int k18 = 0; k18 < 4; k18++) {	// L588
  #pragma HLS pipeline II=1
    int8_t v438[4];
    {
      hls::vector< int8_t, 4 > _vec = v434.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v438[_iv0] = _vec[_iv0];
      }
    }	// L589
    int8_t v439 = v438[2];	// L590
    v435.write(v439);	// L591
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v438[_iv0];
      }
      v436.write(_vec);
    }	// L592
  }
}

void feed_3(
  hls::stream< hls::vector< int8_t, 4 > >& v440,
  hls::stream< int8_t >& v441
) {	// L596
  l_S_k_0_k19: for (int k19 = 0; k19 < 4; k19++) {	// L597
  #pragma HLS pipeline II=1
    int8_t v443[4];
    {
      hls::vector< int8_t, 4 > _vec = v440.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v443[_iv0] = _vec[_iv0];
      }
    }	// L598
    int8_t v444 = v443[3];	// L599
    v441.write(v444);	// L600
  }
}

void feed_2_0(
  hls::stream< hls::vector< int8_t, 4 > >& v445,
  hls::stream< int8_t >& v446,
  hls::stream< hls::vector< int8_t, 4 > >& v447
) {	// L604
  l_S_k_0_k20: for (int k20 = 0; k20 < 4; k20++) {	// L605
  #pragma HLS pipeline II=1
    int8_t v449[4];
    {
      hls::vector< int8_t, 4 > _vec = v445.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v449[_iv0] = _vec[_iv0];
      }
    }	// L606
    int8_t v450 = v449[0];	// L607
    v446.write(v450);	// L608
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v449[_iv0];
      }
      v447.write(_vec);
    }	// L609
  }
}

void feed_2_1(
  hls::stream< hls::vector< int8_t, 4 > >& v451,
  hls::stream< int8_t >& v452,
  hls::stream< hls::vector< int8_t, 4 > >& v453
) {	// L613
  l_S_k_0_k21: for (int k21 = 0; k21 < 4; k21++) {	// L614
  #pragma HLS pipeline II=1
    int8_t v455[4];
    {
      hls::vector< int8_t, 4 > _vec = v451.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v455[_iv0] = _vec[_iv0];
      }
    }	// L615
    int8_t v456 = v455[1];	// L616
    v452.write(v456);	// L617
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v455[_iv0];
      }
      v453.write(_vec);
    }	// L618
  }
}

void feed_2_2(
  hls::stream< hls::vector< int8_t, 4 > >& v457,
  hls::stream< int8_t >& v458,
  hls::stream< hls::vector< int8_t, 4 > >& v459
) {	// L622
  l_S_k_0_k22: for (int k22 = 0; k22 < 4; k22++) {	// L623
  #pragma HLS pipeline II=1
    int8_t v461[4];
    {
      hls::vector< int8_t, 4 > _vec = v457.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v461[_iv0] = _vec[_iv0];
      }
    }	// L624
    int8_t v462 = v461[2];	// L625
    v458.write(v462);	// L626
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v461[_iv0];
      }
      v459.write(_vec);
    }	// L627
  }
}

void feed_2_3(
  hls::stream< hls::vector< int8_t, 4 > >& v463,
  hls::stream< int8_t >& v464
) {	// L631
  l_S_k_0_k23: for (int k23 = 0; k23 < 4; k23++) {	// L632
  #pragma HLS pipeline II=1
    int8_t v466[4];
    {
      hls::vector< int8_t, 4 > _vec = v463.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v466[_iv0] = _vec[_iv0];
      }
    }	// L633
    int8_t v467 = v466[3];	// L634
    v464.write(v467);	// L635
  }
}

void pe_c_out_drain_0(
  int32_t v468[4][4],
  hls::stream< int32_t >& v469
) {	// L639
  #pragma HLS array_partition variable=v468 complete dim=1
  #pragma HLS array_partition variable=v468 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L640
  #pragma HLS pipeline II=1
    int32_t v471 = v469.read();	// L641
    v468[0][_t2] = v471;	// L642
  }
}

void pe_c_out_drain_1(
  int32_t v472[4][4],
  hls::stream< int32_t >& v473
) {	// L646
  #pragma HLS array_partition variable=v472 complete dim=1
  #pragma HLS array_partition variable=v472 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 4; _t3++) {	// L647
  #pragma HLS pipeline II=1
    int32_t v475 = v473.read();	// L648
    v472[1][_t3] = v475;	// L649
  }
}

void pe_c_out_drain_2(
  int32_t v476[4][4],
  hls::stream< int32_t >& v477
) {	// L653
  #pragma HLS array_partition variable=v476 complete dim=1
  #pragma HLS array_partition variable=v476 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 4; _t4++) {	// L654
  #pragma HLS pipeline II=1
    int32_t v479 = v477.read();	// L655
    v476[2][_t4] = v479;	// L656
  }
}

void pe_c_out_drain_3(
  int32_t v480[4][4],
  hls::stream< int32_t >& v481
) {	// L660
  #pragma HLS array_partition variable=v480 complete dim=1
  #pragma HLS array_partition variable=v480 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 4; _t5++) {	// L661
  #pragma HLS pipeline II=1
    int32_t v483 = v481.read();	// L662
    v480[3][_t5] = v483;	// L663
  }
}

/// This is top function.
void top(
  int8_t v484[4][4],
  int8_t v485[4][4],
  int32_t v486[4][4]
) {	// L667
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v484 complete dim=1
  #pragma HLS array_partition variable=v484 complete dim=2

  #pragma HLS array_partition variable=v485 complete dim=1
  #pragma HLS array_partition variable=v485 complete dim=2

  #pragma HLS array_partition variable=v486 complete dim=1
  #pragma HLS array_partition variable=v486 complete dim=2

  hls::stream< int8_t > v487;
  #pragma HLS stream variable=v487 depth=2	// L668
  hls::stream< int8_t > v488;
  #pragma HLS stream variable=v488 depth=2	// L669
  hls::stream< int8_t > v489;
  #pragma HLS stream variable=v489 depth=2	// L670
  hls::stream< int8_t > v490;
  #pragma HLS stream variable=v490 depth=2	// L671
  hls::stream< int8_t > v491;
  #pragma HLS stream variable=v491 depth=2	// L672
  hls::stream< int8_t > v492;
  #pragma HLS stream variable=v492 depth=2	// L673
  hls::stream< int8_t > v493;
  #pragma HLS stream variable=v493 depth=2	// L674
  hls::stream< int8_t > v494;
  #pragma HLS stream variable=v494 depth=2	// L675
  hls::stream< int8_t > v495;
  #pragma HLS stream variable=v495 depth=2	// L676
  hls::stream< int8_t > v496;
  #pragma HLS stream variable=v496 depth=2	// L677
  hls::stream< int8_t > v497;
  #pragma HLS stream variable=v497 depth=2	// L678
  hls::stream< int8_t > v498;
  #pragma HLS stream variable=v498 depth=2	// L679
  hls::stream< int8_t > v499;
  #pragma HLS stream variable=v499 depth=2	// L680
  hls::stream< int8_t > v500;
  #pragma HLS stream variable=v500 depth=2	// L681
  hls::stream< int8_t > v501;
  #pragma HLS stream variable=v501 depth=2	// L682
  hls::stream< int8_t > v502;
  #pragma HLS stream variable=v502 depth=2	// L683
  hls::stream< int8_t > v503;
  #pragma HLS stream variable=v503 depth=2	// L684
  hls::stream< int8_t > v504;
  #pragma HLS stream variable=v504 depth=2	// L685
  hls::stream< int8_t > v505;
  #pragma HLS stream variable=v505 depth=2	// L686
  hls::stream< int8_t > v506;
  #pragma HLS stream variable=v506 depth=2	// L687
  hls::stream< int8_t > v507;
  #pragma HLS stream variable=v507 depth=2	// L688
  hls::stream< int8_t > v508;
  #pragma HLS stream variable=v508 depth=2	// L689
  hls::stream< int8_t > v509;
  #pragma HLS stream variable=v509 depth=2	// L690
  hls::stream< int8_t > v510;
  #pragma HLS stream variable=v510 depth=2	// L691
  hls::stream< int8_t > v511;
  #pragma HLS stream variable=v511 depth=2	// L692
  hls::stream< int8_t > v512;
  #pragma HLS stream variable=v512 depth=2	// L693
  hls::stream< int8_t > v513;
  #pragma HLS stream variable=v513 depth=2	// L694
  hls::stream< int8_t > v514;
  #pragma HLS stream variable=v514 depth=2	// L695
  hls::stream< int8_t > v515;
  #pragma HLS stream variable=v515 depth=2	// L696
  hls::stream< int8_t > v516;
  #pragma HLS stream variable=v516 depth=2	// L697
  hls::stream< int8_t > v517;
  #pragma HLS stream variable=v517 depth=2	// L698
  hls::stream< int8_t > v518;
  #pragma HLS stream variable=v518 depth=2	// L699
  hls::stream< int32_t > v519;
  #pragma HLS stream variable=v519 depth=2	// L700
  hls::stream< int32_t > v520;
  #pragma HLS stream variable=v520 depth=2	// L701
  hls::stream< int32_t > v521;
  #pragma HLS stream variable=v521 depth=2	// L702
  hls::stream< int32_t > v522;
  #pragma HLS stream variable=v522 depth=2	// L703
  hls::stream< int32_t > v523;
  #pragma HLS stream variable=v523 depth=2	// L704
  hls::stream< int32_t > v524;
  #pragma HLS stream variable=v524 depth=2	// L705
  hls::stream< int32_t > v525;
  #pragma HLS stream variable=v525 depth=2	// L706
  hls::stream< int32_t > v526;
  #pragma HLS stream variable=v526 depth=2	// L707
  hls::stream< int32_t > v527;
  #pragma HLS stream variable=v527 depth=2	// L708
  hls::stream< int32_t > v528;
  #pragma HLS stream variable=v528 depth=2	// L709
  hls::stream< int32_t > v529;
  #pragma HLS stream variable=v529 depth=2	// L710
  hls::stream< int32_t > v530;
  #pragma HLS stream variable=v530 depth=2	// L711
  hls::stream< int32_t > v531;
  #pragma HLS stream variable=v531 depth=2	// L712
  hls::stream< int32_t > v532;
  #pragma HLS stream variable=v532 depth=2	// L713
  hls::stream< int32_t > v533;
  #pragma HLS stream variable=v533 depth=2	// L714
  hls::stream< int32_t > v534;
  #pragma HLS stream variable=v534 depth=2	// L715
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v535;
  #pragma HLS stream variable=v535 depth=2	// L716
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v536;
  #pragma HLS stream variable=v536 depth=2	// L717
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v537;
  #pragma HLS stream variable=v537 depth=2	// L718
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v538;
  #pragma HLS stream variable=v538 depth=2	// L719
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v539;
  #pragma HLS stream variable=v539 depth=2	// L720
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v540;
  #pragma HLS stream variable=v540 depth=2	// L721
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v541;
  #pragma HLS stream variable=v541 depth=2	// L722
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v542;
  #pragma HLS stream variable=v542 depth=2	// L723
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v543;
  #pragma HLS stream variable=v543 depth=2	// L724
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v544;
  #pragma HLS stream variable=v544 depth=2	// L725
  hls::stream< int8_t > v545;
  #pragma HLS stream variable=v545 depth=2	// L726
  hls::stream< int8_t > v546;
  #pragma HLS stream variable=v546 depth=2	// L727
  hls::stream< int8_t > v547;
  #pragma HLS stream variable=v547 depth=2	// L728
  hls::stream< int8_t > v548;
  #pragma HLS stream variable=v548 depth=2	// L729
  hls::stream< int8_t > v549;
  #pragma HLS stream variable=v549 depth=2	// L730
  hls::stream< int8_t > v550;
  #pragma HLS stream variable=v550 depth=2	// L731
  hls::stream< int8_t > v551;
  #pragma HLS stream variable=v551 depth=2	// L732
  hls::stream< int8_t > v552;
  #pragma HLS stream variable=v552 depth=2	// L733
  hls::stream< int32_t > v553;
  #pragma HLS stream variable=v553 depth=2	// L734
  hls::stream< int32_t > v554;
  #pragma HLS stream variable=v554 depth=2	// L735
  hls::stream< int32_t > v555;
  #pragma HLS stream variable=v555 depth=2	// L736
  hls::stream< int32_t > v556;
  #pragma HLS stream variable=v556 depth=2	// L737
  feed_up_load_0(v484, v543);	// L738
  feed_2_up_load_0(v485, v544);	// L739
  pe_0_0(v545, v549, v488, v507, v523);	// L740
  pe_0_1(v488, v550, v489, v508, v524);	// L741
  pe_0_2(v489, v551, v490, v509, v525);	// L742
  pe_0_3(v490, v552, v510, v526);	// L743
  pe_1_0(v546, v507, v492, v511, v527, v523);	// L744
  pe_1_1(v492, v508, v493, v512, v528, v524);	// L745
  pe_1_2(v493, v509, v494, v513, v529, v525);	// L746
  pe_1_3(v494, v510, v514, v530, v526);	// L747
  pe_2_0(v547, v511, v496, v515, v531, v527);	// L748
  pe_2_1(v496, v512, v497, v516, v532, v528);	// L749
  pe_2_2(v497, v513, v498, v517, v533, v529);	// L750
  pe_2_3(v498, v514, v518, v534, v530);	// L751
  pe_3_0(v548, v515, v500, v553, v531);	// L752
  pe_3_1(v500, v516, v501, v554, v532);	// L753
  pe_3_2(v501, v517, v502, v555, v533);	// L754
  pe_3_3(v502, v518, v556, v534);	// L755
  feed_0(v543, v545, v536);	// L756
  feed_1(v536, v546, v537);	// L757
  feed_2(v537, v547, v538);	// L758
  feed_3(v538, v548);	// L759
  feed_2_0(v544, v549, v540);	// L760
  feed_2_1(v540, v550, v541);	// L761
  feed_2_2(v541, v551, v542);	// L762
  feed_2_3(v542, v552);	// L763
  pe_c_out_drain_0(v486, v553);	// L764
  pe_c_out_drain_1(v486, v554);	// L765
  pe_c_out_drain_2(v486, v555);	// L766
  pe_c_out_drain_3(v486, v556);	// L767
}


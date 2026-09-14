
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
void pe_west_load(
  float v0[4][4],
  int v1,
  hls::stream< float >& v2
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L4
  #pragma HLS pipeline II=1
    float v4 = v0[v1][_t];	// L5
    v2.write(v4);	// L6
  }
}

void pe_north_load(
  float v5[4][4],
  int v6,
  hls::stream< float >& v7
) {	// L10
  #pragma HLS array_partition variable=v5 complete dim=1
  #pragma HLS array_partition variable=v5 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 4; _t1++) {	// L11
  #pragma HLS pipeline II=1
    float v9 = v5[_t1][v6];	// L12
    v7.write(v9);	// L13
  }
}

void pe_1_west_load(
  float v10[4][4],
  int v11,
  hls::stream< float >& v12
) {	// L17
  #pragma HLS array_partition variable=v10 complete dim=1
  #pragma HLS array_partition variable=v10 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L18
  #pragma HLS pipeline II=1
    float v14 = v10[v11][_t2];	// L19
    v12.write(v14);	// L20
  }
}

void pe_1_north_load(
  float v15[4][4],
  int v16,
  hls::stream< float >& v17
) {	// L24
  #pragma HLS array_partition variable=v15 complete dim=1
  #pragma HLS array_partition variable=v15 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 4; _t3++) {	// L25
  #pragma HLS pipeline II=1
    float v19 = v15[_t3][(v16 + 2)];	// L26
    v17.write(v19);	// L27
  }
}

void pe_2_west_load(
  float v20[4][4],
  int v21,
  hls::stream< float >& v22
) {	// L31
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 4; _t4++) {	// L32
  #pragma HLS pipeline II=1
    float v24 = v20[(v21 + 2)][_t4];	// L33
    v22.write(v24);	// L34
  }
}

void pe_2_north_load(
  float v25[4][4],
  int v26,
  hls::stream< float >& v27
) {	// L38
  #pragma HLS array_partition variable=v25 complete dim=1
  #pragma HLS array_partition variable=v25 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 4; _t5++) {	// L39
  #pragma HLS pipeline II=1
    float v29 = v25[_t5][v26];	// L40
    v27.write(v29);	// L41
  }
}

void pe_3_west_load(
  float v30[4][4],
  int v31,
  hls::stream< float >& v32
) {	// L45
  #pragma HLS array_partition variable=v30 complete dim=1
  #pragma HLS array_partition variable=v30 complete dim=2

  l_S__t_0__t6: for (int _t6 = 0; _t6 < 4; _t6++) {	// L46
  #pragma HLS pipeline II=1
    float v34 = v30[(v31 + 2)][_t6];	// L47
    v32.write(v34);	// L48
  }
}

void pe_3_north_load(
  float v35[4][4],
  int v36,
  hls::stream< float >& v37
) {	// L52
  #pragma HLS array_partition variable=v35 complete dim=1
  #pragma HLS array_partition variable=v35 complete dim=2

  l_S__t_0__t7: for (int _t7 = 0; _t7 < 4; _t7++) {	// L53
  #pragma HLS pipeline II=1
    float v39 = v35[_t7][(v36 + 2)];	// L54
    v37.write(v39);	// L55
  }
}

void pe_r0(
  float v40[4][4],
  int v41,
  int v42,
  hls::stream< float >& v43,
  hls::stream< float >& v44,
  hls::stream< float >& v45
) {	// L59
  #pragma HLS array_partition variable=v40 complete dim=1
  #pragma HLS array_partition variable=v40 complete dim=2

  float acc;	// L62
  acc = (float)0.000000;	// L63
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L64
  #pragma HLS pipeline II=1
    float v48 = v45.read();	// L65
    float a;	// L66
    a = v48;	// L67
    float v50 = v44.read();	// L68
    float b;	// L69
    b = v50;	// L70
    float v52 = a;	// L71
    float v53 = b;	// L72
    float v54 = v52 * v53;	// L73
    float v55 = acc;	// L74
    float v56 = v55 + v54;	// L75
    acc = v56;	// L76
    float v57 = a;	// L77
    v43.write(v57);	// L78
  }
  float v58 = acc;	// L80
  v40[v41][v42] = v58;	// L81
}

void pe_r1(
  float v59[4][4],
  int v60,
  int v61,
  hls::stream< float >& v62,
  hls::stream< float >& v63,
  hls::stream< float >& v64,
  hls::stream< float >& v65
) {	// L84
  #pragma HLS array_partition variable=v59 complete dim=1
  #pragma HLS array_partition variable=v59 complete dim=2

  float acc1;	// L87
  acc1 = (float)0.000000;	// L88
  l_S_k_0_k1: for (int k1 = 0; k1 < 4; k1++) {	// L89
  #pragma HLS pipeline II=1
    float v68 = v65.read();	// L90
    float a1;	// L91
    a1 = v68;	// L92
    float v70 = v63.read();	// L93
    float b1;	// L94
    b1 = v70;	// L95
    float v72 = a1;	// L96
    float v73 = b1;	// L97
    float v74 = v72 * v73;	// L98
    float v75 = acc1;	// L99
    float v76 = v75 + v74;	// L100
    acc1 = v76;	// L101
    float v77 = a1;	// L102
    v62.write(v77);	// L103
    float v78 = b1;	// L104
    v64.write(v78);	// L105
  }
  float v79 = acc1;	// L107
  v59[v60][v61] = v79;	// L108
}

void pe_r2(
  float v80[4][4],
  int v81,
  int v82,
  hls::stream< float >& v83,
  hls::stream< float >& v84
) {	// L111
  #pragma HLS array_partition variable=v80 complete dim=1
  #pragma HLS array_partition variable=v80 complete dim=2

  float acc2;	// L114
  acc2 = (float)0.000000;	// L115
  l_S_k_0_k2: for (int k2 = 0; k2 < 4; k2++) {	// L116
  #pragma HLS pipeline II=1
    float v87 = v84.read();	// L117
    float a2;	// L118
    a2 = v87;	// L119
    float v89 = v83.read();	// L120
    float b2;	// L121
    b2 = v89;	// L122
    float v91 = a2;	// L123
    float v92 = b2;	// L124
    float v93 = v91 * v92;	// L125
    float v94 = acc2;	// L126
    float v95 = v94 + v93;	// L127
    acc2 = v95;	// L128
  }
  float v96 = acc2;	// L130
  v80[v81][v82] = v96;	// L131
}

void pe_r3(
  float v97[4][4],
  int v98,
  int v99,
  hls::stream< float >& v100,
  hls::stream< float >& v101,
  hls::stream< float >& v102
) {	// L134
  #pragma HLS array_partition variable=v97 complete dim=1
  #pragma HLS array_partition variable=v97 complete dim=2

  float acc3;	// L137
  acc3 = (float)0.000000;	// L138
  l_S_k_0_k3: for (int k3 = 0; k3 < 4; k3++) {	// L139
  #pragma HLS pipeline II=1
    float v105 = v102.read();	// L140
    float a3;	// L141
    a3 = v105;	// L142
    float v107 = v100.read();	// L143
    float b3;	// L144
    b3 = v107;	// L145
    float v109 = a3;	// L146
    float v110 = b3;	// L147
    float v111 = v109 * v110;	// L148
    float v112 = acc3;	// L149
    float v113 = v112 + v111;	// L150
    acc3 = v113;	// L151
    float v114 = b3;	// L152
    v101.write(v114);	// L153
  }
  float v115 = acc3;	// L155
  v97[v98][v99] = v115;	// L156
}

void pe_1_r0(
  float v116[4][4],
  int v117,
  int v118,
  hls::stream< float >& v119,
  hls::stream< float >& v120,
  hls::stream< float >& v121
) {	// L159
  #pragma HLS array_partition variable=v116 complete dim=1
  #pragma HLS array_partition variable=v116 complete dim=2

  float acc4;	// L162
  acc4 = (float)0.000000;	// L163
  l_S_k_0_k4: for (int k4 = 0; k4 < 4; k4++) {	// L164
  #pragma HLS pipeline II=1
    float v124 = v121.read();	// L165
    float a4;	// L166
    a4 = v124;	// L167
    float v126 = v120.read();	// L168
    float b4;	// L169
    b4 = v126;	// L170
    float v128 = a4;	// L171
    float v129 = b4;	// L172
    float v130 = v128 * v129;	// L173
    float v131 = acc4;	// L174
    float v132 = v131 + v130;	// L175
    acc4 = v132;	// L176
    float v133 = a4;	// L177
    v119.write(v133);	// L178
  }
  float v134 = acc4;	// L180
  v116[v117][(v118 + 2)] = v134;	// L181
}

void pe_1_r1(
  float v135[4][4],
  int v136,
  int v137,
  hls::stream< float >& v138,
  hls::stream< float >& v139,
  hls::stream< float >& v140,
  hls::stream< float >& v141
) {	// L184
  #pragma HLS array_partition variable=v135 complete dim=1
  #pragma HLS array_partition variable=v135 complete dim=2

  float acc5;	// L187
  acc5 = (float)0.000000;	// L188
  l_S_k_0_k5: for (int k5 = 0; k5 < 4; k5++) {	// L189
  #pragma HLS pipeline II=1
    float v144 = v141.read();	// L190
    float a5;	// L191
    a5 = v144;	// L192
    float v146 = v139.read();	// L193
    float b5;	// L194
    b5 = v146;	// L195
    float v148 = a5;	// L196
    float v149 = b5;	// L197
    float v150 = v148 * v149;	// L198
    float v151 = acc5;	// L199
    float v152 = v151 + v150;	// L200
    acc5 = v152;	// L201
    float v153 = a5;	// L202
    v138.write(v153);	// L203
    float v154 = b5;	// L204
    v140.write(v154);	// L205
  }
  float v155 = acc5;	// L207
  v135[v136][(v137 + 2)] = v155;	// L208
}

void pe_1_r2(
  float v156[4][4],
  int v157,
  int v158,
  hls::stream< float >& v159,
  hls::stream< float >& v160
) {	// L211
  #pragma HLS array_partition variable=v156 complete dim=1
  #pragma HLS array_partition variable=v156 complete dim=2

  float acc6;	// L214
  acc6 = (float)0.000000;	// L215
  l_S_k_0_k6: for (int k6 = 0; k6 < 4; k6++) {	// L216
  #pragma HLS pipeline II=1
    float v163 = v160.read();	// L217
    float a6;	// L218
    a6 = v163;	// L219
    float v165 = v159.read();	// L220
    float b6;	// L221
    b6 = v165;	// L222
    float v167 = a6;	// L223
    float v168 = b6;	// L224
    float v169 = v167 * v168;	// L225
    float v170 = acc6;	// L226
    float v171 = v170 + v169;	// L227
    acc6 = v171;	// L228
  }
  float v172 = acc6;	// L230
  v156[v157][(v158 + 2)] = v172;	// L231
}

void pe_1_r3(
  float v173[4][4],
  int v174,
  int v175,
  hls::stream< float >& v176,
  hls::stream< float >& v177,
  hls::stream< float >& v178
) {	// L234
  #pragma HLS array_partition variable=v173 complete dim=1
  #pragma HLS array_partition variable=v173 complete dim=2

  float acc7;	// L237
  acc7 = (float)0.000000;	// L238
  l_S_k_0_k7: for (int k7 = 0; k7 < 4; k7++) {	// L239
  #pragma HLS pipeline II=1
    float v181 = v178.read();	// L240
    float a7;	// L241
    a7 = v181;	// L242
    float v183 = v176.read();	// L243
    float b7;	// L244
    b7 = v183;	// L245
    float v185 = a7;	// L246
    float v186 = b7;	// L247
    float v187 = v185 * v186;	// L248
    float v188 = acc7;	// L249
    float v189 = v188 + v187;	// L250
    acc7 = v189;	// L251
    float v190 = b7;	// L252
    v177.write(v190);	// L253
  }
  float v191 = acc7;	// L255
  v173[v174][(v175 + 2)] = v191;	// L256
}

void pe_2_r0(
  float v192[4][4],
  int v193,
  int v194,
  hls::stream< float >& v195,
  hls::stream< float >& v196,
  hls::stream< float >& v197
) {	// L259
  #pragma HLS array_partition variable=v192 complete dim=1
  #pragma HLS array_partition variable=v192 complete dim=2

  float acc8;	// L262
  acc8 = (float)0.000000;	// L263
  l_S_k_0_k8: for (int k8 = 0; k8 < 4; k8++) {	// L264
  #pragma HLS pipeline II=1
    float v200 = v197.read();	// L265
    float a8;	// L266
    a8 = v200;	// L267
    float v202 = v196.read();	// L268
    float b8;	// L269
    b8 = v202;	// L270
    float v204 = a8;	// L271
    float v205 = b8;	// L272
    float v206 = v204 * v205;	// L273
    float v207 = acc8;	// L274
    float v208 = v207 + v206;	// L275
    acc8 = v208;	// L276
    float v209 = a8;	// L277
    v195.write(v209);	// L278
  }
  float v210 = acc8;	// L280
  v192[(v193 + 2)][v194] = v210;	// L281
}

void pe_2_r1(
  float v211[4][4],
  int v212,
  int v213,
  hls::stream< float >& v214,
  hls::stream< float >& v215,
  hls::stream< float >& v216,
  hls::stream< float >& v217
) {	// L284
  #pragma HLS array_partition variable=v211 complete dim=1
  #pragma HLS array_partition variable=v211 complete dim=2

  float acc9;	// L287
  acc9 = (float)0.000000;	// L288
  l_S_k_0_k9: for (int k9 = 0; k9 < 4; k9++) {	// L289
  #pragma HLS pipeline II=1
    float v220 = v217.read();	// L290
    float a9;	// L291
    a9 = v220;	// L292
    float v222 = v215.read();	// L293
    float b9;	// L294
    b9 = v222;	// L295
    float v224 = a9;	// L296
    float v225 = b9;	// L297
    float v226 = v224 * v225;	// L298
    float v227 = acc9;	// L299
    float v228 = v227 + v226;	// L300
    acc9 = v228;	// L301
    float v229 = a9;	// L302
    v214.write(v229);	// L303
    float v230 = b9;	// L304
    v216.write(v230);	// L305
  }
  float v231 = acc9;	// L307
  v211[(v212 + 2)][v213] = v231;	// L308
}

void pe_2_r2(
  float v232[4][4],
  int v233,
  int v234,
  hls::stream< float >& v235,
  hls::stream< float >& v236
) {	// L311
  #pragma HLS array_partition variable=v232 complete dim=1
  #pragma HLS array_partition variable=v232 complete dim=2

  float acc10;	// L314
  acc10 = (float)0.000000;	// L315
  l_S_k_0_k10: for (int k10 = 0; k10 < 4; k10++) {	// L316
  #pragma HLS pipeline II=1
    float v239 = v236.read();	// L317
    float a10;	// L318
    a10 = v239;	// L319
    float v241 = v235.read();	// L320
    float b10;	// L321
    b10 = v241;	// L322
    float v243 = a10;	// L323
    float v244 = b10;	// L324
    float v245 = v243 * v244;	// L325
    float v246 = acc10;	// L326
    float v247 = v246 + v245;	// L327
    acc10 = v247;	// L328
  }
  float v248 = acc10;	// L330
  v232[(v233 + 2)][v234] = v248;	// L331
}

void pe_2_r3(
  float v249[4][4],
  int v250,
  int v251,
  hls::stream< float >& v252,
  hls::stream< float >& v253,
  hls::stream< float >& v254
) {	// L334
  #pragma HLS array_partition variable=v249 complete dim=1
  #pragma HLS array_partition variable=v249 complete dim=2

  float acc11;	// L337
  acc11 = (float)0.000000;	// L338
  l_S_k_0_k11: for (int k11 = 0; k11 < 4; k11++) {	// L339
  #pragma HLS pipeline II=1
    float v257 = v254.read();	// L340
    float a11;	// L341
    a11 = v257;	// L342
    float v259 = v252.read();	// L343
    float b11;	// L344
    b11 = v259;	// L345
    float v261 = a11;	// L346
    float v262 = b11;	// L347
    float v263 = v261 * v262;	// L348
    float v264 = acc11;	// L349
    float v265 = v264 + v263;	// L350
    acc11 = v265;	// L351
    float v266 = b11;	// L352
    v253.write(v266);	// L353
  }
  float v267 = acc11;	// L355
  v249[(v250 + 2)][v251] = v267;	// L356
}

void pe_3_r0(
  float v268[4][4],
  int v269,
  int v270,
  hls::stream< float >& v271,
  hls::stream< float >& v272,
  hls::stream< float >& v273
) {	// L359
  #pragma HLS array_partition variable=v268 complete dim=1
  #pragma HLS array_partition variable=v268 complete dim=2

  float acc12;	// L362
  acc12 = (float)0.000000;	// L363
  l_S_k_0_k12: for (int k12 = 0; k12 < 4; k12++) {	// L364
  #pragma HLS pipeline II=1
    float v276 = v273.read();	// L365
    float a12;	// L366
    a12 = v276;	// L367
    float v278 = v272.read();	// L368
    float b12;	// L369
    b12 = v278;	// L370
    float v280 = a12;	// L371
    float v281 = b12;	// L372
    float v282 = v280 * v281;	// L373
    float v283 = acc12;	// L374
    float v284 = v283 + v282;	// L375
    acc12 = v284;	// L376
    float v285 = a12;	// L377
    v271.write(v285);	// L378
  }
  float v286 = acc12;	// L380
  v268[(v269 + 2)][(v270 + 2)] = v286;	// L381
}

void pe_3_r1(
  float v287[4][4],
  int v288,
  int v289,
  hls::stream< float >& v290,
  hls::stream< float >& v291,
  hls::stream< float >& v292,
  hls::stream< float >& v293
) {	// L384
  #pragma HLS array_partition variable=v287 complete dim=1
  #pragma HLS array_partition variable=v287 complete dim=2

  float acc13;	// L387
  acc13 = (float)0.000000;	// L388
  l_S_k_0_k13: for (int k13 = 0; k13 < 4; k13++) {	// L389
  #pragma HLS pipeline II=1
    float v296 = v293.read();	// L390
    float a13;	// L391
    a13 = v296;	// L392
    float v298 = v291.read();	// L393
    float b13;	// L394
    b13 = v298;	// L395
    float v300 = a13;	// L396
    float v301 = b13;	// L397
    float v302 = v300 * v301;	// L398
    float v303 = acc13;	// L399
    float v304 = v303 + v302;	// L400
    acc13 = v304;	// L401
    float v305 = a13;	// L402
    v290.write(v305);	// L403
    float v306 = b13;	// L404
    v292.write(v306);	// L405
  }
  float v307 = acc13;	// L407
  v287[(v288 + 2)][(v289 + 2)] = v307;	// L408
}

void pe_3_r2(
  float v308[4][4],
  int v309,
  int v310,
  hls::stream< float >& v311,
  hls::stream< float >& v312
) {	// L411
  #pragma HLS array_partition variable=v308 complete dim=1
  #pragma HLS array_partition variable=v308 complete dim=2

  float acc14;	// L414
  acc14 = (float)0.000000;	// L415
  l_S_k_0_k14: for (int k14 = 0; k14 < 4; k14++) {	// L416
  #pragma HLS pipeline II=1
    float v315 = v312.read();	// L417
    float a14;	// L418
    a14 = v315;	// L419
    float v317 = v311.read();	// L420
    float b14;	// L421
    b14 = v317;	// L422
    float v319 = a14;	// L423
    float v320 = b14;	// L424
    float v321 = v319 * v320;	// L425
    float v322 = acc14;	// L426
    float v323 = v322 + v321;	// L427
    acc14 = v323;	// L428
  }
  float v324 = acc14;	// L430
  v308[(v309 + 2)][(v310 + 2)] = v324;	// L431
}

void pe_3_r3(
  float v325[4][4],
  int v326,
  int v327,
  hls::stream< float >& v328,
  hls::stream< float >& v329,
  hls::stream< float >& v330
) {	// L434
  #pragma HLS array_partition variable=v325 complete dim=1
  #pragma HLS array_partition variable=v325 complete dim=2

  float acc15;	// L437
  acc15 = (float)0.000000;	// L438
  l_S_k_0_k15: for (int k15 = 0; k15 < 4; k15++) {	// L439
  #pragma HLS pipeline II=1
    float v333 = v330.read();	// L440
    float a15;	// L441
    a15 = v333;	// L442
    float v335 = v328.read();	// L443
    float b15;	// L444
    b15 = v335;	// L445
    float v337 = a15;	// L446
    float v338 = b15;	// L447
    float v339 = v337 * v338;	// L448
    float v340 = acc15;	// L449
    float v341 = v340 + v339;	// L450
    acc15 = v341;	// L451
    float v342 = b15;	// L452
    v329.write(v342);	// L453
  }
  float v343 = acc15;	// L455
  v325[(v326 + 2)][(v327 + 2)] = v343;	// L456
}

/// This is top function.
void top(
  float v344[4][4],
  float v345[4][4],
  float v346[4][4]
) {	// L459
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v344 complete dim=1
  #pragma HLS array_partition variable=v344 complete dim=2

  #pragma HLS array_partition variable=v345 complete dim=1
  #pragma HLS array_partition variable=v345 complete dim=2

  #pragma HLS array_partition variable=v346 complete dim=1
  #pragma HLS array_partition variable=v346 complete dim=2

  hls::stream< float > v347;
  #pragma HLS stream variable=v347 depth=2	// L460
  hls::stream< float > v348;
  #pragma HLS stream variable=v348 depth=2	// L461
  hls::stream< float > v349;
  #pragma HLS stream variable=v349 depth=2	// L462
  hls::stream< float > v350;
  #pragma HLS stream variable=v350 depth=2	// L463
  hls::stream< float > v351;
  #pragma HLS stream variable=v351 depth=2	// L464
  hls::stream< float > v352;
  #pragma HLS stream variable=v352 depth=2	// L465
  hls::stream< float > v353;
  #pragma HLS stream variable=v353 depth=2	// L466
  hls::stream< float > v354;
  #pragma HLS stream variable=v354 depth=2	// L467
  hls::stream< float > v355;
  #pragma HLS stream variable=v355 depth=2	// L468
  hls::stream< float > v356;
  #pragma HLS stream variable=v356 depth=2	// L469
  hls::stream< float > v357;
  #pragma HLS stream variable=v357 depth=2	// L470
  hls::stream< float > v358;
  #pragma HLS stream variable=v358 depth=2	// L471
  hls::stream< float > v359;
  #pragma HLS stream variable=v359 depth=2	// L472
  hls::stream< float > v360;
  #pragma HLS stream variable=v360 depth=2	// L473
  hls::stream< float > v361;
  #pragma HLS stream variable=v361 depth=2	// L474
  hls::stream< float > v362;
  #pragma HLS stream variable=v362 depth=2	// L475
  hls::stream< float > v363;
  #pragma HLS stream variable=v363 depth=4	// L476
  hls::stream< float > v364;
  #pragma HLS stream variable=v364 depth=4	// L477
  hls::stream< float > v365;
  #pragma HLS stream variable=v365 depth=4	// L478
  hls::stream< float > v366;
  #pragma HLS stream variable=v366 depth=4	// L479
  hls::stream< float > v367;
  #pragma HLS stream variable=v367 depth=4	// L480
  hls::stream< float > v368;
  #pragma HLS stream variable=v368 depth=4	// L481
  hls::stream< float > v369;
  #pragma HLS stream variable=v369 depth=4	// L482
  hls::stream< float > v370;
  #pragma HLS stream variable=v370 depth=4	// L483
  hls::stream< float > v371;
  #pragma HLS stream variable=v371 depth=4	// L484
  hls::stream< float > v372;
  #pragma HLS stream variable=v372 depth=4	// L485
  hls::stream< float > v373;
  #pragma HLS stream variable=v373 depth=4	// L486
  hls::stream< float > v374;
  #pragma HLS stream variable=v374 depth=4	// L487
  hls::stream< float > v375;
  #pragma HLS stream variable=v375 depth=4	// L488
  hls::stream< float > v376;
  #pragma HLS stream variable=v376 depth=4	// L489
  hls::stream< float > v377;
  #pragma HLS stream variable=v377 depth=4	// L490
  hls::stream< float > v378;
  #pragma HLS stream variable=v378 depth=4	// L492
  pe_west_load(v344, 0, v378);	// L494
  pe_west_load(v344, 1, v377);	// L495
  pe_north_load(v345, 0, v376);	// L496
  pe_north_load(v345, 1, v375);	// L497
  pe_1_west_load(v344, 0, v374);	// L498
  pe_1_west_load(v344, 1, v373);	// L499
  pe_1_north_load(v345, 0, v372);	// L500
  pe_1_north_load(v345, 1, v371);	// L501
  pe_2_west_load(v344, 0, v370);	// L502
  pe_2_west_load(v344, 1, v369);	// L503
  pe_2_north_load(v345, 0, v368);	// L504
  pe_2_north_load(v345, 1, v367);	// L505
  pe_3_west_load(v344, 0, v366);	// L506
  pe_3_west_load(v344, 1, v365);	// L507
  pe_3_north_load(v345, 0, v364);	// L508
  pe_3_north_load(v345, 1, v363);	// L509
  pe_r1(v346, 0, 0, v362, v376, v361, v378);	// L510
  pe_r3(v346, 0, 1, v375, v360, v362);	// L511
  pe_r0(v346, 1, 0, v359, v361, v377);	// L512
  pe_r2(v346, 1, 1, v360, v359);	// L513
  pe_1_r1(v346, 0, 0, v358, v372, v357, v374);	// L514
  pe_1_r3(v346, 0, 1, v371, v356, v358);	// L515
  pe_1_r0(v346, 1, 0, v355, v357, v373);	// L516
  pe_1_r2(v346, 1, 1, v356, v355);	// L517
  pe_2_r1(v346, 0, 0, v354, v368, v353, v370);	// L518
  pe_2_r3(v346, 0, 1, v367, v352, v354);	// L519
  pe_2_r0(v346, 1, 0, v351, v353, v369);	// L520
  pe_2_r2(v346, 1, 1, v352, v351);	// L521
  pe_3_r1(v346, 0, 0, v350, v364, v349, v366);	// L522
  pe_3_r3(v346, 0, 1, v363, v348, v350);	// L523
  pe_3_r0(v346, 1, 0, v347, v349, v365);	// L524
  pe_3_r2(v346, 1, 1, v348, v347);	// L525
}


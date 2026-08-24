
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
void pe_west_load_0(
  float v0[4][4],
  hls::stream< float >& v1
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L4
    float v3 = v0[0][_t];	// L5
    v1.write(v3);	// L6
  }
}

void pe_west_load_1(
  float v4[4][4],
  hls::stream< float >& v5
) {	// L10
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 4; _t1++) {	// L11
    float v7 = v4[1][_t1];	// L12
    v5.write(v7);	// L13
  }
}

void pe_north_load_0(
  float v8[4][4],
  hls::stream< float >& v9
) {	// L17
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L18
    float v11 = v8[_t2][0];	// L19
    v9.write(v11);	// L20
  }
}

void pe_north_load_1(
  float v12[4][4],
  hls::stream< float >& v13
) {	// L24
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 4; _t3++) {	// L25
    float v15 = v12[_t3][1];	// L26
    v13.write(v15);	// L27
  }
}

void pe_1_west_load_0(
  float v16[4][4],
  hls::stream< float >& v17
) {	// L31
  #pragma HLS array_partition variable=v16 complete dim=1
  #pragma HLS array_partition variable=v16 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 4; _t4++) {	// L32
    float v19 = v16[0][_t4];	// L33
    v17.write(v19);	// L34
  }
}

void pe_1_west_load_1(
  float v20[4][4],
  hls::stream< float >& v21
) {	// L38
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 4; _t5++) {	// L39
    float v23 = v20[1][_t5];	// L40
    v21.write(v23);	// L41
  }
}

void pe_1_north_load_0(
  float v24[4][4],
  hls::stream< float >& v25
) {	// L45
  #pragma HLS array_partition variable=v24 complete dim=1
  #pragma HLS array_partition variable=v24 complete dim=2

  l_S__t_0__t6: for (int _t6 = 0; _t6 < 4; _t6++) {	// L46
    float v27 = v24[_t6][2];	// L47
    v25.write(v27);	// L48
  }
}

void pe_1_north_load_1(
  float v28[4][4],
  hls::stream< float >& v29
) {	// L52
  #pragma HLS array_partition variable=v28 complete dim=1
  #pragma HLS array_partition variable=v28 complete dim=2

  l_S__t_0__t7: for (int _t7 = 0; _t7 < 4; _t7++) {	// L53
    float v31 = v28[_t7][3];	// L54
    v29.write(v31);	// L55
  }
}

void pe_2_west_load_0(
  float v32[4][4],
  hls::stream< float >& v33
) {	// L59
  #pragma HLS array_partition variable=v32 complete dim=1
  #pragma HLS array_partition variable=v32 complete dim=2

  l_S__t_0__t8: for (int _t8 = 0; _t8 < 4; _t8++) {	// L60
    float v35 = v32[2][_t8];	// L61
    v33.write(v35);	// L62
  }
}

void pe_2_west_load_1(
  float v36[4][4],
  hls::stream< float >& v37
) {	// L66
  #pragma HLS array_partition variable=v36 complete dim=1
  #pragma HLS array_partition variable=v36 complete dim=2

  l_S__t_0__t9: for (int _t9 = 0; _t9 < 4; _t9++) {	// L67
    float v39 = v36[3][_t9];	// L68
    v37.write(v39);	// L69
  }
}

void pe_2_north_load_0(
  float v40[4][4],
  hls::stream< float >& v41
) {	// L73
  #pragma HLS array_partition variable=v40 complete dim=1
  #pragma HLS array_partition variable=v40 complete dim=2

  l_S__t_0__t10: for (int _t10 = 0; _t10 < 4; _t10++) {	// L74
    float v43 = v40[_t10][0];	// L75
    v41.write(v43);	// L76
  }
}

void pe_2_north_load_1(
  float v44[4][4],
  hls::stream< float >& v45
) {	// L80
  #pragma HLS array_partition variable=v44 complete dim=1
  #pragma HLS array_partition variable=v44 complete dim=2

  l_S__t_0__t11: for (int _t11 = 0; _t11 < 4; _t11++) {	// L81
    float v47 = v44[_t11][1];	// L82
    v45.write(v47);	// L83
  }
}

void pe_3_west_load_0(
  float v48[4][4],
  hls::stream< float >& v49
) {	// L87
  #pragma HLS array_partition variable=v48 complete dim=1
  #pragma HLS array_partition variable=v48 complete dim=2

  l_S__t_0__t12: for (int _t12 = 0; _t12 < 4; _t12++) {	// L88
    float v51 = v48[2][_t12];	// L89
    v49.write(v51);	// L90
  }
}

void pe_3_west_load_1(
  float v52[4][4],
  hls::stream< float >& v53
) {	// L94
  #pragma HLS array_partition variable=v52 complete dim=1
  #pragma HLS array_partition variable=v52 complete dim=2

  l_S__t_0__t13: for (int _t13 = 0; _t13 < 4; _t13++) {	// L95
    float v55 = v52[3][_t13];	// L96
    v53.write(v55);	// L97
  }
}

void pe_3_north_load_0(
  float v56[4][4],
  hls::stream< float >& v57
) {	// L101
  #pragma HLS array_partition variable=v56 complete dim=1
  #pragma HLS array_partition variable=v56 complete dim=2

  l_S__t_0__t14: for (int _t14 = 0; _t14 < 4; _t14++) {	// L102
    float v59 = v56[_t14][2];	// L103
    v57.write(v59);	// L104
  }
}

void pe_3_north_load_1(
  float v60[4][4],
  hls::stream< float >& v61
) {	// L108
  #pragma HLS array_partition variable=v60 complete dim=1
  #pragma HLS array_partition variable=v60 complete dim=2

  l_S__t_0__t15: for (int _t15 = 0; _t15 < 4; _t15++) {	// L109
    float v63 = v60[_t15][3];	// L110
    v61.write(v63);	// L111
  }
}

void pe_0_0(
  float v64[4][4],
  hls::stream< float >& v65,
  hls::stream< float >& v66,
  hls::stream< float >& v67,
  hls::stream< float >& v68
) {	// L115
  #pragma HLS array_partition variable=v64 complete dim=1
  #pragma HLS array_partition variable=v64 complete dim=2

  float acc;	// L117
  acc = (float)0.000000;	// L118
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L119
    float v71 = v65.read();	// L120
    float a;	// L121
    a = v71;	// L122
    float v73 = v66.read();	// L123
    float b;	// L124
    b = v73;	// L125
    float v75 = a;	// L126
    float v76 = b;	// L127
    float v77 = v75 * v76;	// L128
    float v78 = acc;	// L129
    float v79 = v78 + v77;	// L130
    acc = v79;	// L131
    float v80 = a;	// L132
    v67.write(v80);	// L133
    float v81 = b;	// L134
    v68.write(v81);	// L135
  }
  float v82 = acc;	// L137
  v64[0][0] = v82;	// L138
}

void pe_0_1(
  float v83[4][4],
  hls::stream< float >& v84,
  hls::stream< float >& v85,
  hls::stream< float >& v86
) {	// L141
  #pragma HLS array_partition variable=v83 complete dim=1
  #pragma HLS array_partition variable=v83 complete dim=2

  float acc1;	// L143
  acc1 = (float)0.000000;	// L144
  l_S_k_0_k1: for (int k1 = 0; k1 < 4; k1++) {	// L145
    float v89 = v84.read();	// L146
    float a1;	// L147
    a1 = v89;	// L148
    float v91 = v85.read();	// L149
    float b1;	// L150
    b1 = v91;	// L151
    float v93 = a1;	// L152
    float v94 = b1;	// L153
    float v95 = v93 * v94;	// L154
    float v96 = acc1;	// L155
    float v97 = v96 + v95;	// L156
    acc1 = v97;	// L157
    float v98 = b1;	// L158
    v86.write(v98);	// L159
  }
  float v99 = acc1;	// L161
  v83[0][1] = v99;	// L162
}

void pe_1_0(
  float v100[4][4],
  hls::stream< float >& v101,
  hls::stream< float >& v102,
  hls::stream< float >& v103
) {	// L165
  #pragma HLS array_partition variable=v100 complete dim=1
  #pragma HLS array_partition variable=v100 complete dim=2

  float acc2;	// L167
  acc2 = (float)0.000000;	// L168
  l_S_k_0_k2: for (int k2 = 0; k2 < 4; k2++) {	// L169
    float v106 = v101.read();	// L170
    float a2;	// L171
    a2 = v106;	// L172
    float v108 = v102.read();	// L173
    float b2;	// L174
    b2 = v108;	// L175
    float v110 = a2;	// L176
    float v111 = b2;	// L177
    float v112 = v110 * v111;	// L178
    float v113 = acc2;	// L179
    float v114 = v113 + v112;	// L180
    acc2 = v114;	// L181
    float v115 = a2;	// L182
    v103.write(v115);	// L183
  }
  float v116 = acc2;	// L185
  v100[1][0] = v116;	// L186
}

void pe_1_1(
  float v117[4][4],
  hls::stream< float >& v118,
  hls::stream< float >& v119
) {	// L189
  #pragma HLS array_partition variable=v117 complete dim=1
  #pragma HLS array_partition variable=v117 complete dim=2

  float acc3;	// L191
  acc3 = (float)0.000000;	// L192
  l_S_k_0_k3: for (int k3 = 0; k3 < 4; k3++) {	// L193
    float v122 = v118.read();	// L194
    float a3;	// L195
    a3 = v122;	// L196
    float v124 = v119.read();	// L197
    float b3;	// L198
    b3 = v124;	// L199
    float v126 = a3;	// L200
    float v127 = b3;	// L201
    float v128 = v126 * v127;	// L202
    float v129 = acc3;	// L203
    float v130 = v129 + v128;	// L204
    acc3 = v130;	// L205
  }
  float v131 = acc3;	// L207
  v117[1][1] = v131;	// L208
}

void pe_1_0_0(
  float v132[4][4],
  hls::stream< float >& v133,
  hls::stream< float >& v134,
  hls::stream< float >& v135,
  hls::stream< float >& v136
) {	// L211
  #pragma HLS array_partition variable=v132 complete dim=1
  #pragma HLS array_partition variable=v132 complete dim=2

  float acc4;	// L213
  acc4 = (float)0.000000;	// L214
  l_S_k_0_k4: for (int k4 = 0; k4 < 4; k4++) {	// L215
    float v139 = v133.read();	// L216
    float a4;	// L217
    a4 = v139;	// L218
    float v141 = v134.read();	// L219
    float b4;	// L220
    b4 = v141;	// L221
    float v143 = a4;	// L222
    float v144 = b4;	// L223
    float v145 = v143 * v144;	// L224
    float v146 = acc4;	// L225
    float v147 = v146 + v145;	// L226
    acc4 = v147;	// L227
    float v148 = a4;	// L228
    v135.write(v148);	// L229
    float v149 = b4;	// L230
    v136.write(v149);	// L231
  }
  float v150 = acc4;	// L233
  v132[0][2] = v150;	// L234
}

void pe_1_0_1(
  float v151[4][4],
  hls::stream< float >& v152,
  hls::stream< float >& v153,
  hls::stream< float >& v154
) {	// L237
  #pragma HLS array_partition variable=v151 complete dim=1
  #pragma HLS array_partition variable=v151 complete dim=2

  float acc5;	// L239
  acc5 = (float)0.000000;	// L240
  l_S_k_0_k5: for (int k5 = 0; k5 < 4; k5++) {	// L241
    float v157 = v152.read();	// L242
    float a5;	// L243
    a5 = v157;	// L244
    float v159 = v153.read();	// L245
    float b5;	// L246
    b5 = v159;	// L247
    float v161 = a5;	// L248
    float v162 = b5;	// L249
    float v163 = v161 * v162;	// L250
    float v164 = acc5;	// L251
    float v165 = v164 + v163;	// L252
    acc5 = v165;	// L253
    float v166 = b5;	// L254
    v154.write(v166);	// L255
  }
  float v167 = acc5;	// L257
  v151[0][3] = v167;	// L258
}

void pe_1_1_0(
  float v168[4][4],
  hls::stream< float >& v169,
  hls::stream< float >& v170,
  hls::stream< float >& v171
) {	// L261
  #pragma HLS array_partition variable=v168 complete dim=1
  #pragma HLS array_partition variable=v168 complete dim=2

  float acc6;	// L263
  acc6 = (float)0.000000;	// L264
  l_S_k_0_k6: for (int k6 = 0; k6 < 4; k6++) {	// L265
    float v174 = v169.read();	// L266
    float a6;	// L267
    a6 = v174;	// L268
    float v176 = v170.read();	// L269
    float b6;	// L270
    b6 = v176;	// L271
    float v178 = a6;	// L272
    float v179 = b6;	// L273
    float v180 = v178 * v179;	// L274
    float v181 = acc6;	// L275
    float v182 = v181 + v180;	// L276
    acc6 = v182;	// L277
    float v183 = a6;	// L278
    v171.write(v183);	// L279
  }
  float v184 = acc6;	// L281
  v168[1][2] = v184;	// L282
}

void pe_1_1_1(
  float v185[4][4],
  hls::stream< float >& v186,
  hls::stream< float >& v187
) {	// L285
  #pragma HLS array_partition variable=v185 complete dim=1
  #pragma HLS array_partition variable=v185 complete dim=2

  float acc7;	// L287
  acc7 = (float)0.000000;	// L288
  l_S_k_0_k7: for (int k7 = 0; k7 < 4; k7++) {	// L289
    float v190 = v186.read();	// L290
    float a7;	// L291
    a7 = v190;	// L292
    float v192 = v187.read();	// L293
    float b7;	// L294
    b7 = v192;	// L295
    float v194 = a7;	// L296
    float v195 = b7;	// L297
    float v196 = v194 * v195;	// L298
    float v197 = acc7;	// L299
    float v198 = v197 + v196;	// L300
    acc7 = v198;	// L301
  }
  float v199 = acc7;	// L303
  v185[1][3] = v199;	// L304
}

void pe_2_0_0(
  float v200[4][4],
  hls::stream< float >& v201,
  hls::stream< float >& v202,
  hls::stream< float >& v203,
  hls::stream< float >& v204
) {	// L307
  #pragma HLS array_partition variable=v200 complete dim=1
  #pragma HLS array_partition variable=v200 complete dim=2

  float acc8;	// L309
  acc8 = (float)0.000000;	// L310
  l_S_k_0_k8: for (int k8 = 0; k8 < 4; k8++) {	// L311
    float v207 = v201.read();	// L312
    float a8;	// L313
    a8 = v207;	// L314
    float v209 = v202.read();	// L315
    float b8;	// L316
    b8 = v209;	// L317
    float v211 = a8;	// L318
    float v212 = b8;	// L319
    float v213 = v211 * v212;	// L320
    float v214 = acc8;	// L321
    float v215 = v214 + v213;	// L322
    acc8 = v215;	// L323
    float v216 = a8;	// L324
    v203.write(v216);	// L325
    float v217 = b8;	// L326
    v204.write(v217);	// L327
  }
  float v218 = acc8;	// L329
  v200[2][0] = v218;	// L330
}

void pe_2_0_1(
  float v219[4][4],
  hls::stream< float >& v220,
  hls::stream< float >& v221,
  hls::stream< float >& v222
) {	// L333
  #pragma HLS array_partition variable=v219 complete dim=1
  #pragma HLS array_partition variable=v219 complete dim=2

  float acc9;	// L335
  acc9 = (float)0.000000;	// L336
  l_S_k_0_k9: for (int k9 = 0; k9 < 4; k9++) {	// L337
    float v225 = v220.read();	// L338
    float a9;	// L339
    a9 = v225;	// L340
    float v227 = v221.read();	// L341
    float b9;	// L342
    b9 = v227;	// L343
    float v229 = a9;	// L344
    float v230 = b9;	// L345
    float v231 = v229 * v230;	// L346
    float v232 = acc9;	// L347
    float v233 = v232 + v231;	// L348
    acc9 = v233;	// L349
    float v234 = b9;	// L350
    v222.write(v234);	// L351
  }
  float v235 = acc9;	// L353
  v219[2][1] = v235;	// L354
}

void pe_2_1_0(
  float v236[4][4],
  hls::stream< float >& v237,
  hls::stream< float >& v238,
  hls::stream< float >& v239
) {	// L357
  #pragma HLS array_partition variable=v236 complete dim=1
  #pragma HLS array_partition variable=v236 complete dim=2

  float acc10;	// L359
  acc10 = (float)0.000000;	// L360
  l_S_k_0_k10: for (int k10 = 0; k10 < 4; k10++) {	// L361
    float v242 = v237.read();	// L362
    float a10;	// L363
    a10 = v242;	// L364
    float v244 = v238.read();	// L365
    float b10;	// L366
    b10 = v244;	// L367
    float v246 = a10;	// L368
    float v247 = b10;	// L369
    float v248 = v246 * v247;	// L370
    float v249 = acc10;	// L371
    float v250 = v249 + v248;	// L372
    acc10 = v250;	// L373
    float v251 = a10;	// L374
    v239.write(v251);	// L375
  }
  float v252 = acc10;	// L377
  v236[3][0] = v252;	// L378
}

void pe_2_1_1(
  float v253[4][4],
  hls::stream< float >& v254,
  hls::stream< float >& v255
) {	// L381
  #pragma HLS array_partition variable=v253 complete dim=1
  #pragma HLS array_partition variable=v253 complete dim=2

  float acc11;	// L383
  acc11 = (float)0.000000;	// L384
  l_S_k_0_k11: for (int k11 = 0; k11 < 4; k11++) {	// L385
    float v258 = v254.read();	// L386
    float a11;	// L387
    a11 = v258;	// L388
    float v260 = v255.read();	// L389
    float b11;	// L390
    b11 = v260;	// L391
    float v262 = a11;	// L392
    float v263 = b11;	// L393
    float v264 = v262 * v263;	// L394
    float v265 = acc11;	// L395
    float v266 = v265 + v264;	// L396
    acc11 = v266;	// L397
  }
  float v267 = acc11;	// L399
  v253[3][1] = v267;	// L400
}

void pe_3_0_0(
  float v268[4][4],
  hls::stream< float >& v269,
  hls::stream< float >& v270,
  hls::stream< float >& v271,
  hls::stream< float >& v272
) {	// L403
  #pragma HLS array_partition variable=v268 complete dim=1
  #pragma HLS array_partition variable=v268 complete dim=2

  float acc12;	// L405
  acc12 = (float)0.000000;	// L406
  l_S_k_0_k12: for (int k12 = 0; k12 < 4; k12++) {	// L407
    float v275 = v269.read();	// L408
    float a12;	// L409
    a12 = v275;	// L410
    float v277 = v270.read();	// L411
    float b12;	// L412
    b12 = v277;	// L413
    float v279 = a12;	// L414
    float v280 = b12;	// L415
    float v281 = v279 * v280;	// L416
    float v282 = acc12;	// L417
    float v283 = v282 + v281;	// L418
    acc12 = v283;	// L419
    float v284 = a12;	// L420
    v271.write(v284);	// L421
    float v285 = b12;	// L422
    v272.write(v285);	// L423
  }
  float v286 = acc12;	// L425
  v268[2][2] = v286;	// L426
}

void pe_3_0_1(
  float v287[4][4],
  hls::stream< float >& v288,
  hls::stream< float >& v289,
  hls::stream< float >& v290
) {	// L429
  #pragma HLS array_partition variable=v287 complete dim=1
  #pragma HLS array_partition variable=v287 complete dim=2

  float acc13;	// L431
  acc13 = (float)0.000000;	// L432
  l_S_k_0_k13: for (int k13 = 0; k13 < 4; k13++) {	// L433
    float v293 = v288.read();	// L434
    float a13;	// L435
    a13 = v293;	// L436
    float v295 = v289.read();	// L437
    float b13;	// L438
    b13 = v295;	// L439
    float v297 = a13;	// L440
    float v298 = b13;	// L441
    float v299 = v297 * v298;	// L442
    float v300 = acc13;	// L443
    float v301 = v300 + v299;	// L444
    acc13 = v301;	// L445
    float v302 = b13;	// L446
    v290.write(v302);	// L447
  }
  float v303 = acc13;	// L449
  v287[2][3] = v303;	// L450
}

void pe_3_1_0(
  float v304[4][4],
  hls::stream< float >& v305,
  hls::stream< float >& v306,
  hls::stream< float >& v307
) {	// L453
  #pragma HLS array_partition variable=v304 complete dim=1
  #pragma HLS array_partition variable=v304 complete dim=2

  float acc14;	// L455
  acc14 = (float)0.000000;	// L456
  l_S_k_0_k14: for (int k14 = 0; k14 < 4; k14++) {	// L457
    float v310 = v305.read();	// L458
    float a14;	// L459
    a14 = v310;	// L460
    float v312 = v306.read();	// L461
    float b14;	// L462
    b14 = v312;	// L463
    float v314 = a14;	// L464
    float v315 = b14;	// L465
    float v316 = v314 * v315;	// L466
    float v317 = acc14;	// L467
    float v318 = v317 + v316;	// L468
    acc14 = v318;	// L469
    float v319 = a14;	// L470
    v307.write(v319);	// L471
  }
  float v320 = acc14;	// L473
  v304[3][2] = v320;	// L474
}

void pe_3_1_1(
  float v321[4][4],
  hls::stream< float >& v322,
  hls::stream< float >& v323
) {	// L477
  #pragma HLS array_partition variable=v321 complete dim=1
  #pragma HLS array_partition variable=v321 complete dim=2

  float acc15;	// L479
  acc15 = (float)0.000000;	// L480
  l_S_k_0_k15: for (int k15 = 0; k15 < 4; k15++) {	// L481
    float v326 = v322.read();	// L482
    float a15;	// L483
    a15 = v326;	// L484
    float v328 = v323.read();	// L485
    float b15;	// L486
    b15 = v328;	// L487
    float v330 = a15;	// L488
    float v331 = b15;	// L489
    float v332 = v330 * v331;	// L490
    float v333 = acc15;	// L491
    float v334 = v333 + v332;	// L492
    acc15 = v334;	// L493
  }
  float v335 = acc15;	// L495
  v321[3][3] = v335;	// L496
}

/// This is top function.
void top(
  float v336[4][4],
  float v337[4][4],
  float v338[4][4]
) {	// L499
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v336 complete dim=1
  #pragma HLS array_partition variable=v336 complete dim=2

  #pragma HLS array_partition variable=v337 complete dim=1
  #pragma HLS array_partition variable=v337 complete dim=2

  #pragma HLS array_partition variable=v338 complete dim=1
  #pragma HLS array_partition variable=v338 complete dim=2

  hls::stream< float > v339;
  #pragma HLS stream variable=v339 depth=2	// L500
  hls::stream< float > v340;
  #pragma HLS stream variable=v340 depth=2	// L501
  hls::stream< float > v341;
  #pragma HLS stream variable=v341 depth=2	// L502
  hls::stream< float > v342;
  #pragma HLS stream variable=v342 depth=2	// L503
  hls::stream< float > v343;
  #pragma HLS stream variable=v343 depth=2	// L504
  hls::stream< float > v344;
  #pragma HLS stream variable=v344 depth=2	// L505
  hls::stream< float > v345;
  #pragma HLS stream variable=v345 depth=2	// L506
  hls::stream< float > v346;
  #pragma HLS stream variable=v346 depth=2	// L507
  hls::stream< float > v347;
  #pragma HLS stream variable=v347 depth=2	// L508
  hls::stream< float > v348;
  #pragma HLS stream variable=v348 depth=2	// L509
  hls::stream< float > v349;
  #pragma HLS stream variable=v349 depth=2	// L510
  hls::stream< float > v350;
  #pragma HLS stream variable=v350 depth=2	// L511
  hls::stream< float > v351;
  #pragma HLS stream variable=v351 depth=2	// L512
  hls::stream< float > v352;
  #pragma HLS stream variable=v352 depth=2	// L513
  hls::stream< float > v353;
  #pragma HLS stream variable=v353 depth=2	// L514
  hls::stream< float > v354;
  #pragma HLS stream variable=v354 depth=2	// L515
  hls::stream< float > v355;
  #pragma HLS stream variable=v355 depth=2	// L516
  hls::stream< float > v356;
  #pragma HLS stream variable=v356 depth=2	// L517
  hls::stream< float > v357;
  #pragma HLS stream variable=v357 depth=2	// L518
  hls::stream< float > v358;
  #pragma HLS stream variable=v358 depth=2	// L519
  hls::stream< float > v359;
  #pragma HLS stream variable=v359 depth=2	// L520
  hls::stream< float > v360;
  #pragma HLS stream variable=v360 depth=2	// L521
  hls::stream< float > v361;
  #pragma HLS stream variable=v361 depth=2	// L522
  hls::stream< float > v362;
  #pragma HLS stream variable=v362 depth=2	// L523
  hls::stream< float > v363;
  #pragma HLS stream variable=v363 depth=2	// L524
  hls::stream< float > v364;
  #pragma HLS stream variable=v364 depth=2	// L525
  hls::stream< float > v365;
  #pragma HLS stream variable=v365 depth=2	// L526
  hls::stream< float > v366;
  #pragma HLS stream variable=v366 depth=2	// L527
  hls::stream< float > v367;
  #pragma HLS stream variable=v367 depth=2	// L528
  hls::stream< float > v368;
  #pragma HLS stream variable=v368 depth=2	// L529
  hls::stream< float > v369;
  #pragma HLS stream variable=v369 depth=2	// L530
  hls::stream< float > v370;
  #pragma HLS stream variable=v370 depth=2	// L531
  hls::stream< float > v371;
  #pragma HLS stream variable=v371 depth=2	// L532
  hls::stream< float > v372;
  #pragma HLS stream variable=v372 depth=2	// L533
  hls::stream< float > v373;
  #pragma HLS stream variable=v373 depth=2	// L534
  hls::stream< float > v374;
  #pragma HLS stream variable=v374 depth=2	// L535
  hls::stream< float > v375;
  #pragma HLS stream variable=v375 depth=2	// L536
  hls::stream< float > v376;
  #pragma HLS stream variable=v376 depth=2	// L537
  hls::stream< float > v377;
  #pragma HLS stream variable=v377 depth=2	// L538
  hls::stream< float > v378;
  #pragma HLS stream variable=v378 depth=2	// L539
  hls::stream< float > v379;
  #pragma HLS stream variable=v379 depth=2	// L540
  hls::stream< float > v380;
  #pragma HLS stream variable=v380 depth=2	// L541
  hls::stream< float > v381;
  #pragma HLS stream variable=v381 depth=2	// L542
  hls::stream< float > v382;
  #pragma HLS stream variable=v382 depth=2	// L543
  hls::stream< float > v383;
  #pragma HLS stream variable=v383 depth=2	// L544
  hls::stream< float > v384;
  #pragma HLS stream variable=v384 depth=2	// L545
  hls::stream< float > v385;
  #pragma HLS stream variable=v385 depth=2	// L546
  hls::stream< float > v386;
  #pragma HLS stream variable=v386 depth=2	// L547
  pe_west_load_0(v336, v371);	// L548
  pe_west_load_1(v336, v372);	// L549
  pe_north_load_0(v337, v373);	// L550
  pe_north_load_1(v337, v374);	// L551
  pe_1_west_load_0(v336, v375);	// L552
  pe_1_west_load_1(v336, v376);	// L553
  pe_1_north_load_0(v337, v377);	// L554
  pe_1_north_load_1(v337, v378);	// L555
  pe_2_west_load_0(v336, v379);	// L556
  pe_2_west_load_1(v336, v380);	// L557
  pe_2_north_load_0(v337, v381);	// L558
  pe_2_north_load_1(v337, v382);	// L559
  pe_3_west_load_0(v336, v383);	// L560
  pe_3_west_load_1(v336, v384);	// L561
  pe_3_north_load_0(v337, v385);	// L562
  pe_3_north_load_1(v337, v386);	// L563
  pe_0_0(v338, v371, v373, v340, v345);	// L564
  pe_0_1(v338, v340, v374, v346);	// L565
  pe_1_0(v338, v372, v345, v342);	// L566
  pe_1_1(v338, v342, v346);	// L567
  pe_1_0_0(v338, v375, v377, v348, v353);	// L568
  pe_1_0_1(v338, v348, v378, v354);	// L569
  pe_1_1_0(v338, v376, v353, v350);	// L570
  pe_1_1_1(v338, v350, v354);	// L571
  pe_2_0_0(v338, v379, v381, v356, v361);	// L572
  pe_2_0_1(v338, v356, v382, v362);	// L573
  pe_2_1_0(v338, v380, v361, v358);	// L574
  pe_2_1_1(v338, v358, v362);	// L575
  pe_3_0_0(v338, v383, v385, v364, v369);	// L576
  pe_3_0_1(v338, v364, v386, v370);	// L577
  pe_3_1_0(v338, v384, v369, v366);	// L578
  pe_3_1_1(v338, v366, v370);	// L579
}


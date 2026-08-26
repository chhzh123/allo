
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
  int8_t v0[4][4],
  hls::stream< int8_t >& v1
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L4
  #pragma HLS pipeline II=1
    int8_t v3 = v0[0][_t];	// L5
    v1.write(v3);	// L6
  }
}

void pe_west_load_1(
  int8_t v4[4][4],
  hls::stream< int8_t >& v5
) {	// L10
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 4; _t1++) {	// L11
  #pragma HLS pipeline II=1
    int8_t v7 = v4[1][_t1];	// L12
    v5.write(v7);	// L13
  }
}

void pe_west_load_2(
  int8_t v8[4][4],
  hls::stream< int8_t >& v9
) {	// L17
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L18
  #pragma HLS pipeline II=1
    int8_t v11 = v8[2][_t2];	// L19
    v9.write(v11);	// L20
  }
}

void pe_west_load_3(
  int8_t v12[4][4],
  hls::stream< int8_t >& v13
) {	// L24
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 4; _t3++) {	// L25
  #pragma HLS pipeline II=1
    int8_t v15 = v12[3][_t3];	// L26
    v13.write(v15);	// L27
  }
}

void pe_north_load_0(
  int8_t v16[4][4],
  hls::stream< int8_t >& v17
) {	// L31
  #pragma HLS array_partition variable=v16 complete dim=1
  #pragma HLS array_partition variable=v16 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 4; _t4++) {	// L32
  #pragma HLS pipeline II=1
    int8_t v19 = v16[_t4][0];	// L33
    v17.write(v19);	// L34
  }
}

void pe_north_load_1(
  int8_t v20[4][4],
  hls::stream< int8_t >& v21
) {	// L38
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 4; _t5++) {	// L39
  #pragma HLS pipeline II=1
    int8_t v23 = v20[_t5][1];	// L40
    v21.write(v23);	// L41
  }
}

void pe_north_load_2(
  int8_t v24[4][4],
  hls::stream< int8_t >& v25
) {	// L45
  #pragma HLS array_partition variable=v24 complete dim=1
  #pragma HLS array_partition variable=v24 complete dim=2

  l_S__t_0__t6: for (int _t6 = 0; _t6 < 4; _t6++) {	// L46
  #pragma HLS pipeline II=1
    int8_t v27 = v24[_t6][2];	// L47
    v25.write(v27);	// L48
  }
}

void pe_north_load_3(
  int8_t v28[4][4],
  hls::stream< int8_t >& v29
) {	// L52
  #pragma HLS array_partition variable=v28 complete dim=1
  #pragma HLS array_partition variable=v28 complete dim=2

  l_S__t_0__t7: for (int _t7 = 0; _t7 < 4; _t7++) {	// L53
  #pragma HLS pipeline II=1
    int8_t v31 = v28[_t7][3];	// L54
    v29.write(v31);	// L55
  }
}

void pe_0_0(
  int32_t v32[4][4],
  hls::stream< int8_t >& v33,
  hls::stream< int8_t >& v34,
  hls::stream< int8_t >& v35,
  hls::stream< int8_t >& v36
) {	// L59
  #pragma HLS array_partition variable=v32 complete dim=1
  #pragma HLS array_partition variable=v32 complete dim=2

  int32_t acc;	// L61
  acc = 0;	// L62
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L63
  #pragma HLS pipeline II=1
    int8_t v39 = v33.read();	// L64
    int8_t a;	// L65
    a = v39;	// L66
    int8_t v41 = v34.read();	// L67
    int8_t b;	// L68
    b = v41;	// L69
    int8_t v43 = a;	// L70
    int8_t v44 = b;	// L71
    int16_t v45 = v43;	// L72
    int16_t v46 = v44;	// L73
    int16_t v47 = v45 * v46;	// L74
    int32_t v48 = acc;	// L75
    ap_int<33> v49 = v48;	// L76
    ap_int<33> v50 = v47;	// L77
    ap_int<33> v51 = v49 + v50;	// L78
    int32_t v52 = v51;	// L79
    acc = v52;	// L80
    int8_t v53 = a;	// L81
    v35.write(v53);	// L82
    int8_t v54 = b;	// L83
    v36.write(v54);	// L84
  }
  int32_t v55 = acc;	// L86
  v32[0][0] = v55;	// L87
}

void pe_0_1(
  int32_t v56[4][4],
  hls::stream< int8_t >& v57,
  hls::stream< int8_t >& v58,
  hls::stream< int8_t >& v59,
  hls::stream< int8_t >& v60
) {	// L90
  #pragma HLS array_partition variable=v56 complete dim=1
  #pragma HLS array_partition variable=v56 complete dim=2

  int32_t acc1;	// L92
  acc1 = 0;	// L93
  l_S_k_0_k1: for (int k1 = 0; k1 < 4; k1++) {	// L94
  #pragma HLS pipeline II=1
    int8_t v63 = v57.read();	// L95
    int8_t a1;	// L96
    a1 = v63;	// L97
    int8_t v65 = v58.read();	// L98
    int8_t b1;	// L99
    b1 = v65;	// L100
    int8_t v67 = a1;	// L101
    int8_t v68 = b1;	// L102
    int16_t v69 = v67;	// L103
    int16_t v70 = v68;	// L104
    int16_t v71 = v69 * v70;	// L105
    int32_t v72 = acc1;	// L106
    ap_int<33> v73 = v72;	// L107
    ap_int<33> v74 = v71;	// L108
    ap_int<33> v75 = v73 + v74;	// L109
    int32_t v76 = v75;	// L110
    acc1 = v76;	// L111
    int8_t v77 = a1;	// L112
    v59.write(v77);	// L113
    int8_t v78 = b1;	// L114
    v60.write(v78);	// L115
  }
  int32_t v79 = acc1;	// L117
  v56[0][1] = v79;	// L118
}

void pe_0_2(
  int32_t v80[4][4],
  hls::stream< int8_t >& v81,
  hls::stream< int8_t >& v82,
  hls::stream< int8_t >& v83,
  hls::stream< int8_t >& v84
) {	// L121
  #pragma HLS array_partition variable=v80 complete dim=1
  #pragma HLS array_partition variable=v80 complete dim=2

  int32_t acc2;	// L123
  acc2 = 0;	// L124
  l_S_k_0_k2: for (int k2 = 0; k2 < 4; k2++) {	// L125
  #pragma HLS pipeline II=1
    int8_t v87 = v81.read();	// L126
    int8_t a2;	// L127
    a2 = v87;	// L128
    int8_t v89 = v82.read();	// L129
    int8_t b2;	// L130
    b2 = v89;	// L131
    int8_t v91 = a2;	// L132
    int8_t v92 = b2;	// L133
    int16_t v93 = v91;	// L134
    int16_t v94 = v92;	// L135
    int16_t v95 = v93 * v94;	// L136
    int32_t v96 = acc2;	// L137
    ap_int<33> v97 = v96;	// L138
    ap_int<33> v98 = v95;	// L139
    ap_int<33> v99 = v97 + v98;	// L140
    int32_t v100 = v99;	// L141
    acc2 = v100;	// L142
    int8_t v101 = a2;	// L143
    v83.write(v101);	// L144
    int8_t v102 = b2;	// L145
    v84.write(v102);	// L146
  }
  int32_t v103 = acc2;	// L148
  v80[0][2] = v103;	// L149
}

void pe_0_3(
  int32_t v104[4][4],
  hls::stream< int8_t >& v105,
  hls::stream< int8_t >& v106,
  hls::stream< int8_t >& v107
) {	// L152
  #pragma HLS array_partition variable=v104 complete dim=1
  #pragma HLS array_partition variable=v104 complete dim=2

  int32_t acc3;	// L154
  acc3 = 0;	// L155
  l_S_k_0_k3: for (int k3 = 0; k3 < 4; k3++) {	// L156
  #pragma HLS pipeline II=1
    int8_t v110 = v105.read();	// L157
    int8_t a3;	// L158
    a3 = v110;	// L159
    int8_t v112 = v106.read();	// L160
    int8_t b3;	// L161
    b3 = v112;	// L162
    int8_t v114 = a3;	// L163
    int8_t v115 = b3;	// L164
    int16_t v116 = v114;	// L165
    int16_t v117 = v115;	// L166
    int16_t v118 = v116 * v117;	// L167
    int32_t v119 = acc3;	// L168
    ap_int<33> v120 = v119;	// L169
    ap_int<33> v121 = v118;	// L170
    ap_int<33> v122 = v120 + v121;	// L171
    int32_t v123 = v122;	// L172
    acc3 = v123;	// L173
    int8_t v124 = b3;	// L174
    v107.write(v124);	// L175
  }
  int32_t v125 = acc3;	// L177
  v104[0][3] = v125;	// L178
}

void pe_1_0(
  int32_t v126[4][4],
  hls::stream< int8_t >& v127,
  hls::stream< int8_t >& v128,
  hls::stream< int8_t >& v129,
  hls::stream< int8_t >& v130
) {	// L181
  #pragma HLS array_partition variable=v126 complete dim=1
  #pragma HLS array_partition variable=v126 complete dim=2

  int32_t acc4;	// L183
  acc4 = 0;	// L184
  l_S_k_0_k4: for (int k4 = 0; k4 < 4; k4++) {	// L185
  #pragma HLS pipeline II=1
    int8_t v133 = v127.read();	// L186
    int8_t a4;	// L187
    a4 = v133;	// L188
    int8_t v135 = v128.read();	// L189
    int8_t b4;	// L190
    b4 = v135;	// L191
    int8_t v137 = a4;	// L192
    int8_t v138 = b4;	// L193
    int16_t v139 = v137;	// L194
    int16_t v140 = v138;	// L195
    int16_t v141 = v139 * v140;	// L196
    int32_t v142 = acc4;	// L197
    ap_int<33> v143 = v142;	// L198
    ap_int<33> v144 = v141;	// L199
    ap_int<33> v145 = v143 + v144;	// L200
    int32_t v146 = v145;	// L201
    acc4 = v146;	// L202
    int8_t v147 = a4;	// L203
    v129.write(v147);	// L204
    int8_t v148 = b4;	// L205
    v130.write(v148);	// L206
  }
  int32_t v149 = acc4;	// L208
  v126[1][0] = v149;	// L209
}

void pe_1_1(
  int32_t v150[4][4],
  hls::stream< int8_t >& v151,
  hls::stream< int8_t >& v152,
  hls::stream< int8_t >& v153,
  hls::stream< int8_t >& v154
) {	// L212
  #pragma HLS array_partition variable=v150 complete dim=1
  #pragma HLS array_partition variable=v150 complete dim=2

  int32_t acc5;	// L214
  acc5 = 0;	// L215
  l_S_k_0_k5: for (int k5 = 0; k5 < 4; k5++) {	// L216
  #pragma HLS pipeline II=1
    int8_t v157 = v151.read();	// L217
    int8_t a5;	// L218
    a5 = v157;	// L219
    int8_t v159 = v152.read();	// L220
    int8_t b5;	// L221
    b5 = v159;	// L222
    int8_t v161 = a5;	// L223
    int8_t v162 = b5;	// L224
    int16_t v163 = v161;	// L225
    int16_t v164 = v162;	// L226
    int16_t v165 = v163 * v164;	// L227
    int32_t v166 = acc5;	// L228
    ap_int<33> v167 = v166;	// L229
    ap_int<33> v168 = v165;	// L230
    ap_int<33> v169 = v167 + v168;	// L231
    int32_t v170 = v169;	// L232
    acc5 = v170;	// L233
    int8_t v171 = a5;	// L234
    v153.write(v171);	// L235
    int8_t v172 = b5;	// L236
    v154.write(v172);	// L237
  }
  int32_t v173 = acc5;	// L239
  v150[1][1] = v173;	// L240
}

void pe_1_2(
  int32_t v174[4][4],
  hls::stream< int8_t >& v175,
  hls::stream< int8_t >& v176,
  hls::stream< int8_t >& v177,
  hls::stream< int8_t >& v178
) {	// L243
  #pragma HLS array_partition variable=v174 complete dim=1
  #pragma HLS array_partition variable=v174 complete dim=2

  int32_t acc6;	// L245
  acc6 = 0;	// L246
  l_S_k_0_k6: for (int k6 = 0; k6 < 4; k6++) {	// L247
  #pragma HLS pipeline II=1
    int8_t v181 = v175.read();	// L248
    int8_t a6;	// L249
    a6 = v181;	// L250
    int8_t v183 = v176.read();	// L251
    int8_t b6;	// L252
    b6 = v183;	// L253
    int8_t v185 = a6;	// L254
    int8_t v186 = b6;	// L255
    int16_t v187 = v185;	// L256
    int16_t v188 = v186;	// L257
    int16_t v189 = v187 * v188;	// L258
    int32_t v190 = acc6;	// L259
    ap_int<33> v191 = v190;	// L260
    ap_int<33> v192 = v189;	// L261
    ap_int<33> v193 = v191 + v192;	// L262
    int32_t v194 = v193;	// L263
    acc6 = v194;	// L264
    int8_t v195 = a6;	// L265
    v177.write(v195);	// L266
    int8_t v196 = b6;	// L267
    v178.write(v196);	// L268
  }
  int32_t v197 = acc6;	// L270
  v174[1][2] = v197;	// L271
}

void pe_1_3(
  int32_t v198[4][4],
  hls::stream< int8_t >& v199,
  hls::stream< int8_t >& v200,
  hls::stream< int8_t >& v201
) {	// L274
  #pragma HLS array_partition variable=v198 complete dim=1
  #pragma HLS array_partition variable=v198 complete dim=2

  int32_t acc7;	// L276
  acc7 = 0;	// L277
  l_S_k_0_k7: for (int k7 = 0; k7 < 4; k7++) {	// L278
  #pragma HLS pipeline II=1
    int8_t v204 = v199.read();	// L279
    int8_t a7;	// L280
    a7 = v204;	// L281
    int8_t v206 = v200.read();	// L282
    int8_t b7;	// L283
    b7 = v206;	// L284
    int8_t v208 = a7;	// L285
    int8_t v209 = b7;	// L286
    int16_t v210 = v208;	// L287
    int16_t v211 = v209;	// L288
    int16_t v212 = v210 * v211;	// L289
    int32_t v213 = acc7;	// L290
    ap_int<33> v214 = v213;	// L291
    ap_int<33> v215 = v212;	// L292
    ap_int<33> v216 = v214 + v215;	// L293
    int32_t v217 = v216;	// L294
    acc7 = v217;	// L295
    int8_t v218 = b7;	// L296
    v201.write(v218);	// L297
  }
  int32_t v219 = acc7;	// L299
  v198[1][3] = v219;	// L300
}

void pe_2_0(
  int32_t v220[4][4],
  hls::stream< int8_t >& v221,
  hls::stream< int8_t >& v222,
  hls::stream< int8_t >& v223,
  hls::stream< int8_t >& v224
) {	// L303
  #pragma HLS array_partition variable=v220 complete dim=1
  #pragma HLS array_partition variable=v220 complete dim=2

  int32_t acc8;	// L305
  acc8 = 0;	// L306
  l_S_k_0_k8: for (int k8 = 0; k8 < 4; k8++) {	// L307
  #pragma HLS pipeline II=1
    int8_t v227 = v221.read();	// L308
    int8_t a8;	// L309
    a8 = v227;	// L310
    int8_t v229 = v222.read();	// L311
    int8_t b8;	// L312
    b8 = v229;	// L313
    int8_t v231 = a8;	// L314
    int8_t v232 = b8;	// L315
    int16_t v233 = v231;	// L316
    int16_t v234 = v232;	// L317
    int16_t v235 = v233 * v234;	// L318
    int32_t v236 = acc8;	// L319
    ap_int<33> v237 = v236;	// L320
    ap_int<33> v238 = v235;	// L321
    ap_int<33> v239 = v237 + v238;	// L322
    int32_t v240 = v239;	// L323
    acc8 = v240;	// L324
    int8_t v241 = a8;	// L325
    v223.write(v241);	// L326
    int8_t v242 = b8;	// L327
    v224.write(v242);	// L328
  }
  int32_t v243 = acc8;	// L330
  v220[2][0] = v243;	// L331
}

void pe_2_1(
  int32_t v244[4][4],
  hls::stream< int8_t >& v245,
  hls::stream< int8_t >& v246,
  hls::stream< int8_t >& v247,
  hls::stream< int8_t >& v248
) {	// L334
  #pragma HLS array_partition variable=v244 complete dim=1
  #pragma HLS array_partition variable=v244 complete dim=2

  int32_t acc9;	// L336
  acc9 = 0;	// L337
  l_S_k_0_k9: for (int k9 = 0; k9 < 4; k9++) {	// L338
  #pragma HLS pipeline II=1
    int8_t v251 = v245.read();	// L339
    int8_t a9;	// L340
    a9 = v251;	// L341
    int8_t v253 = v246.read();	// L342
    int8_t b9;	// L343
    b9 = v253;	// L344
    int8_t v255 = a9;	// L345
    int8_t v256 = b9;	// L346
    int16_t v257 = v255;	// L347
    int16_t v258 = v256;	// L348
    int16_t v259 = v257 * v258;	// L349
    int32_t v260 = acc9;	// L350
    ap_int<33> v261 = v260;	// L351
    ap_int<33> v262 = v259;	// L352
    ap_int<33> v263 = v261 + v262;	// L353
    int32_t v264 = v263;	// L354
    acc9 = v264;	// L355
    int8_t v265 = a9;	// L356
    v247.write(v265);	// L357
    int8_t v266 = b9;	// L358
    v248.write(v266);	// L359
  }
  int32_t v267 = acc9;	// L361
  v244[2][1] = v267;	// L362
}

void pe_2_2(
  int32_t v268[4][4],
  hls::stream< int8_t >& v269,
  hls::stream< int8_t >& v270,
  hls::stream< int8_t >& v271,
  hls::stream< int8_t >& v272
) {	// L365
  #pragma HLS array_partition variable=v268 complete dim=1
  #pragma HLS array_partition variable=v268 complete dim=2

  int32_t acc10;	// L367
  acc10 = 0;	// L368
  l_S_k_0_k10: for (int k10 = 0; k10 < 4; k10++) {	// L369
  #pragma HLS pipeline II=1
    int8_t v275 = v269.read();	// L370
    int8_t a10;	// L371
    a10 = v275;	// L372
    int8_t v277 = v270.read();	// L373
    int8_t b10;	// L374
    b10 = v277;	// L375
    int8_t v279 = a10;	// L376
    int8_t v280 = b10;	// L377
    int16_t v281 = v279;	// L378
    int16_t v282 = v280;	// L379
    int16_t v283 = v281 * v282;	// L380
    int32_t v284 = acc10;	// L381
    ap_int<33> v285 = v284;	// L382
    ap_int<33> v286 = v283;	// L383
    ap_int<33> v287 = v285 + v286;	// L384
    int32_t v288 = v287;	// L385
    acc10 = v288;	// L386
    int8_t v289 = a10;	// L387
    v271.write(v289);	// L388
    int8_t v290 = b10;	// L389
    v272.write(v290);	// L390
  }
  int32_t v291 = acc10;	// L392
  v268[2][2] = v291;	// L393
}

void pe_2_3(
  int32_t v292[4][4],
  hls::stream< int8_t >& v293,
  hls::stream< int8_t >& v294,
  hls::stream< int8_t >& v295
) {	// L396
  #pragma HLS array_partition variable=v292 complete dim=1
  #pragma HLS array_partition variable=v292 complete dim=2

  int32_t acc11;	// L398
  acc11 = 0;	// L399
  l_S_k_0_k11: for (int k11 = 0; k11 < 4; k11++) {	// L400
  #pragma HLS pipeline II=1
    int8_t v298 = v293.read();	// L401
    int8_t a11;	// L402
    a11 = v298;	// L403
    int8_t v300 = v294.read();	// L404
    int8_t b11;	// L405
    b11 = v300;	// L406
    int8_t v302 = a11;	// L407
    int8_t v303 = b11;	// L408
    int16_t v304 = v302;	// L409
    int16_t v305 = v303;	// L410
    int16_t v306 = v304 * v305;	// L411
    int32_t v307 = acc11;	// L412
    ap_int<33> v308 = v307;	// L413
    ap_int<33> v309 = v306;	// L414
    ap_int<33> v310 = v308 + v309;	// L415
    int32_t v311 = v310;	// L416
    acc11 = v311;	// L417
    int8_t v312 = b11;	// L418
    v295.write(v312);	// L419
  }
  int32_t v313 = acc11;	// L421
  v292[2][3] = v313;	// L422
}

void pe_3_0(
  int32_t v314[4][4],
  hls::stream< int8_t >& v315,
  hls::stream< int8_t >& v316,
  hls::stream< int8_t >& v317
) {	// L425
  #pragma HLS array_partition variable=v314 complete dim=1
  #pragma HLS array_partition variable=v314 complete dim=2

  int32_t acc12;	// L427
  acc12 = 0;	// L428
  l_S_k_0_k12: for (int k12 = 0; k12 < 4; k12++) {	// L429
  #pragma HLS pipeline II=1
    int8_t v320 = v315.read();	// L430
    int8_t a12;	// L431
    a12 = v320;	// L432
    int8_t v322 = v316.read();	// L433
    int8_t b12;	// L434
    b12 = v322;	// L435
    int8_t v324 = a12;	// L436
    int8_t v325 = b12;	// L437
    int16_t v326 = v324;	// L438
    int16_t v327 = v325;	// L439
    int16_t v328 = v326 * v327;	// L440
    int32_t v329 = acc12;	// L441
    ap_int<33> v330 = v329;	// L442
    ap_int<33> v331 = v328;	// L443
    ap_int<33> v332 = v330 + v331;	// L444
    int32_t v333 = v332;	// L445
    acc12 = v333;	// L446
    int8_t v334 = a12;	// L447
    v317.write(v334);	// L448
  }
  int32_t v335 = acc12;	// L450
  v314[3][0] = v335;	// L451
}

void pe_3_1(
  int32_t v336[4][4],
  hls::stream< int8_t >& v337,
  hls::stream< int8_t >& v338,
  hls::stream< int8_t >& v339
) {	// L454
  #pragma HLS array_partition variable=v336 complete dim=1
  #pragma HLS array_partition variable=v336 complete dim=2

  int32_t acc13;	// L456
  acc13 = 0;	// L457
  l_S_k_0_k13: for (int k13 = 0; k13 < 4; k13++) {	// L458
  #pragma HLS pipeline II=1
    int8_t v342 = v337.read();	// L459
    int8_t a13;	// L460
    a13 = v342;	// L461
    int8_t v344 = v338.read();	// L462
    int8_t b13;	// L463
    b13 = v344;	// L464
    int8_t v346 = a13;	// L465
    int8_t v347 = b13;	// L466
    int16_t v348 = v346;	// L467
    int16_t v349 = v347;	// L468
    int16_t v350 = v348 * v349;	// L469
    int32_t v351 = acc13;	// L470
    ap_int<33> v352 = v351;	// L471
    ap_int<33> v353 = v350;	// L472
    ap_int<33> v354 = v352 + v353;	// L473
    int32_t v355 = v354;	// L474
    acc13 = v355;	// L475
    int8_t v356 = a13;	// L476
    v339.write(v356);	// L477
  }
  int32_t v357 = acc13;	// L479
  v336[3][1] = v357;	// L480
}

void pe_3_2(
  int32_t v358[4][4],
  hls::stream< int8_t >& v359,
  hls::stream< int8_t >& v360,
  hls::stream< int8_t >& v361
) {	// L483
  #pragma HLS array_partition variable=v358 complete dim=1
  #pragma HLS array_partition variable=v358 complete dim=2

  int32_t acc14;	// L485
  acc14 = 0;	// L486
  l_S_k_0_k14: for (int k14 = 0; k14 < 4; k14++) {	// L487
  #pragma HLS pipeline II=1
    int8_t v364 = v359.read();	// L488
    int8_t a14;	// L489
    a14 = v364;	// L490
    int8_t v366 = v360.read();	// L491
    int8_t b14;	// L492
    b14 = v366;	// L493
    int8_t v368 = a14;	// L494
    int8_t v369 = b14;	// L495
    int16_t v370 = v368;	// L496
    int16_t v371 = v369;	// L497
    int16_t v372 = v370 * v371;	// L498
    int32_t v373 = acc14;	// L499
    ap_int<33> v374 = v373;	// L500
    ap_int<33> v375 = v372;	// L501
    ap_int<33> v376 = v374 + v375;	// L502
    int32_t v377 = v376;	// L503
    acc14 = v377;	// L504
    int8_t v378 = a14;	// L505
    v361.write(v378);	// L506
  }
  int32_t v379 = acc14;	// L508
  v358[3][2] = v379;	// L509
}

void pe_3_3(
  int32_t v380[4][4],
  hls::stream< int8_t >& v381,
  hls::stream< int8_t >& v382
) {	// L512
  #pragma HLS array_partition variable=v380 complete dim=1
  #pragma HLS array_partition variable=v380 complete dim=2

  int32_t acc15;	// L514
  acc15 = 0;	// L515
  l_S_k_0_k15: for (int k15 = 0; k15 < 4; k15++) {	// L516
  #pragma HLS pipeline II=1
    int8_t v385 = v381.read();	// L517
    int8_t a15;	// L518
    a15 = v385;	// L519
    int8_t v387 = v382.read();	// L520
    int8_t b15;	// L521
    b15 = v387;	// L522
    int8_t v389 = a15;	// L523
    int8_t v390 = b15;	// L524
    int16_t v391 = v389;	// L525
    int16_t v392 = v390;	// L526
    int16_t v393 = v391 * v392;	// L527
    int32_t v394 = acc15;	// L528
    ap_int<33> v395 = v394;	// L529
    ap_int<33> v396 = v393;	// L530
    ap_int<33> v397 = v395 + v396;	// L531
    int32_t v398 = v397;	// L532
    acc15 = v398;	// L533
  }
  int32_t v399 = acc15;	// L535
  v380[3][3] = v399;	// L536
}

/// This is top function.
void top(
  int8_t v400[4][4],
  int8_t v401[4][4],
  int32_t v402[4][4]
) {	// L539
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v400 complete dim=1
  #pragma HLS array_partition variable=v400 complete dim=2

  #pragma HLS array_partition variable=v401 complete dim=1
  #pragma HLS array_partition variable=v401 complete dim=2

  #pragma HLS array_partition variable=v402 complete dim=1
  #pragma HLS array_partition variable=v402 complete dim=2

  hls::stream< int8_t > v403;
  #pragma HLS stream variable=v403 depth=2	// L540
  hls::stream< int8_t > v404;
  #pragma HLS stream variable=v404 depth=2	// L541
  hls::stream< int8_t > v405;
  #pragma HLS stream variable=v405 depth=2	// L542
  hls::stream< int8_t > v406;
  #pragma HLS stream variable=v406 depth=2	// L543
  hls::stream< int8_t > v407;
  #pragma HLS stream variable=v407 depth=2	// L544
  hls::stream< int8_t > v408;
  #pragma HLS stream variable=v408 depth=2	// L545
  hls::stream< int8_t > v409;
  #pragma HLS stream variable=v409 depth=2	// L546
  hls::stream< int8_t > v410;
  #pragma HLS stream variable=v410 depth=2	// L547
  hls::stream< int8_t > v411;
  #pragma HLS stream variable=v411 depth=2	// L548
  hls::stream< int8_t > v412;
  #pragma HLS stream variable=v412 depth=2	// L549
  hls::stream< int8_t > v413;
  #pragma HLS stream variable=v413 depth=2	// L550
  hls::stream< int8_t > v414;
  #pragma HLS stream variable=v414 depth=2	// L551
  hls::stream< int8_t > v415;
  #pragma HLS stream variable=v415 depth=2	// L552
  hls::stream< int8_t > v416;
  #pragma HLS stream variable=v416 depth=2	// L553
  hls::stream< int8_t > v417;
  #pragma HLS stream variable=v417 depth=2	// L554
  hls::stream< int8_t > v418;
  #pragma HLS stream variable=v418 depth=2	// L555
  hls::stream< int8_t > v419;
  #pragma HLS stream variable=v419 depth=2	// L556
  hls::stream< int8_t > v420;
  #pragma HLS stream variable=v420 depth=2	// L557
  hls::stream< int8_t > v421;
  #pragma HLS stream variable=v421 depth=2	// L558
  hls::stream< int8_t > v422;
  #pragma HLS stream variable=v422 depth=2	// L559
  hls::stream< int8_t > v423;
  #pragma HLS stream variable=v423 depth=2	// L560
  hls::stream< int8_t > v424;
  #pragma HLS stream variable=v424 depth=2	// L561
  hls::stream< int8_t > v425;
  #pragma HLS stream variable=v425 depth=2	// L562
  hls::stream< int8_t > v426;
  #pragma HLS stream variable=v426 depth=2	// L563
  hls::stream< int8_t > v427;
  #pragma HLS stream variable=v427 depth=2	// L564
  hls::stream< int8_t > v428;
  #pragma HLS stream variable=v428 depth=2	// L565
  hls::stream< int8_t > v429;
  #pragma HLS stream variable=v429 depth=2	// L566
  hls::stream< int8_t > v430;
  #pragma HLS stream variable=v430 depth=2	// L567
  hls::stream< int8_t > v431;
  #pragma HLS stream variable=v431 depth=2	// L568
  hls::stream< int8_t > v432;
  #pragma HLS stream variable=v432 depth=2	// L569
  hls::stream< int8_t > v433;
  #pragma HLS stream variable=v433 depth=2	// L570
  hls::stream< int8_t > v434;
  #pragma HLS stream variable=v434 depth=2	// L571
  hls::stream< int8_t > v435;
  #pragma HLS stream variable=v435 depth=2	// L572
  hls::stream< int8_t > v436;
  #pragma HLS stream variable=v436 depth=2	// L573
  hls::stream< int8_t > v437;
  #pragma HLS stream variable=v437 depth=2	// L574
  hls::stream< int8_t > v438;
  #pragma HLS stream variable=v438 depth=2	// L575
  hls::stream< int8_t > v439;
  #pragma HLS stream variable=v439 depth=2	// L576
  hls::stream< int8_t > v440;
  #pragma HLS stream variable=v440 depth=2	// L577
  hls::stream< int8_t > v441;
  #pragma HLS stream variable=v441 depth=2	// L578
  hls::stream< int8_t > v442;
  #pragma HLS stream variable=v442 depth=2	// L579
  pe_west_load_0(v400, v435);	// L580
  pe_west_load_1(v400, v436);	// L581
  pe_west_load_2(v400, v437);	// L582
  pe_west_load_3(v400, v438);	// L583
  pe_north_load_0(v401, v439);	// L584
  pe_north_load_1(v401, v440);	// L585
  pe_north_load_2(v401, v441);	// L586
  pe_north_load_3(v401, v442);	// L587
  pe_0_0(v402, v435, v439, v404, v423);	// L588
  pe_0_1(v402, v404, v440, v405, v424);	// L589
  pe_0_2(v402, v405, v441, v406, v425);	// L590
  pe_0_3(v402, v406, v442, v426);	// L591
  pe_1_0(v402, v436, v423, v408, v427);	// L592
  pe_1_1(v402, v408, v424, v409, v428);	// L593
  pe_1_2(v402, v409, v425, v410, v429);	// L594
  pe_1_3(v402, v410, v426, v430);	// L595
  pe_2_0(v402, v437, v427, v412, v431);	// L596
  pe_2_1(v402, v412, v428, v413, v432);	// L597
  pe_2_2(v402, v413, v429, v414, v433);	// L598
  pe_2_3(v402, v414, v430, v434);	// L599
  pe_3_0(v402, v438, v431, v416);	// L600
  pe_3_1(v402, v416, v432, v417);	// L601
  pe_3_2(v402, v417, v433, v418);	// L602
  pe_3_3(v402, v418, v434);	// L603
}



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
void mac_a_in_load_0(
  int8_t v0[6][4],
  hls::stream< int8_t >& v1
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 6; _t++) {	// L4
  #pragma HLS pipeline II=1
    int8_t v3 = v0[_t][0];	// L5
    v1.write(v3);	// L6
  }
}

void mac_a_in_load_1(
  int8_t v4[6][4],
  hls::stream< int8_t >& v5
) {	// L10
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 6; _t1++) {	// L11
  #pragma HLS pipeline II=1
    int8_t v7 = v4[_t1][1];	// L12
    v5.write(v7);	// L13
  }
}

void mac_a_in_load_2(
  int8_t v8[6][4],
  hls::stream< int8_t >& v9
) {	// L17
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 6; _t2++) {	// L18
  #pragma HLS pipeline II=1
    int8_t v11 = v8[_t2][2];	// L19
    v9.write(v11);	// L20
  }
}

void mac_a_in_load_3(
  int8_t v12[6][4],
  hls::stream< int8_t >& v13
) {	// L24
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 6; _t3++) {	// L25
  #pragma HLS pipeline II=1
    int8_t v15 = v12[_t3][3];	// L26
    v13.write(v15);	// L27
  }
}

void mac_0_0(
  int8_t v16[4][4],
  hls::stream< int8_t >& v17,
  hls::stream< int32_t >& v18,
  hls::stream< int8_t >& v19
) {	// L31
  #pragma HLS array_partition variable=v16 complete dim=1
  #pragma HLS array_partition variable=v16 complete dim=2

  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L33
  #pragma HLS pipeline II=1
    int8_t v21 = v17.read();	// L34
    int8_t a;	// L35
    a = v21;	// L36
    int32_t p;	// L37
    p = 0;	// L38
    int32_t v24 = p;	// L39
    int8_t v25 = a;	// L40
    int8_t v26 = v16[0][0];	// L41
    int16_t v27 = v25;	// L42
    int16_t v28 = v26;	// L43
    int16_t v29 = v27 * v28;	// L44
    ap_int<33> v30 = v24;	// L45
    ap_int<33> v31 = v29;	// L46
    ap_int<33> v32 = v30 + v31;	// L47
    v18.write(v32);	// L48
    int8_t v33 = a;	// L49
    v19.write(v33);	// L50
  }
}

void mac_0_1(
  int8_t v34[4][4],
  hls::stream< int8_t >& v35,
  hls::stream< int32_t >& v36,
  hls::stream< int8_t >& v37
) {	// L54
  #pragma HLS array_partition variable=v34 complete dim=1
  #pragma HLS array_partition variable=v34 complete dim=2

  l_S_m_0_m1: for (int m1 = 0; m1 < 6; m1++) {	// L56
  #pragma HLS pipeline II=1
    int8_t v39 = v35.read();	// L57
    int8_t a1;	// L58
    a1 = v39;	// L59
    int32_t p1;	// L60
    p1 = 0;	// L61
    int32_t v42 = p1;	// L62
    int8_t v43 = a1;	// L63
    int8_t v44 = v34[0][1];	// L64
    int16_t v45 = v43;	// L65
    int16_t v46 = v44;	// L66
    int16_t v47 = v45 * v46;	// L67
    ap_int<33> v48 = v42;	// L68
    ap_int<33> v49 = v47;	// L69
    ap_int<33> v50 = v48 + v49;	// L70
    v36.write(v50);	// L71
    int8_t v51 = a1;	// L72
    v37.write(v51);	// L73
  }
}

void mac_0_2(
  int8_t v52[4][4],
  hls::stream< int8_t >& v53,
  hls::stream< int32_t >& v54,
  hls::stream< int8_t >& v55
) {	// L77
  #pragma HLS array_partition variable=v52 complete dim=1
  #pragma HLS array_partition variable=v52 complete dim=2

  l_S_m_0_m2: for (int m2 = 0; m2 < 6; m2++) {	// L79
  #pragma HLS pipeline II=1
    int8_t v57 = v53.read();	// L80
    int8_t a2;	// L81
    a2 = v57;	// L82
    int32_t p2;	// L83
    p2 = 0;	// L84
    int32_t v60 = p2;	// L85
    int8_t v61 = a2;	// L86
    int8_t v62 = v52[0][2];	// L87
    int16_t v63 = v61;	// L88
    int16_t v64 = v62;	// L89
    int16_t v65 = v63 * v64;	// L90
    ap_int<33> v66 = v60;	// L91
    ap_int<33> v67 = v65;	// L92
    ap_int<33> v68 = v66 + v67;	// L93
    v54.write(v68);	// L94
    int8_t v69 = a2;	// L95
    v55.write(v69);	// L96
  }
}

void mac_0_3(
  int8_t v70[4][4],
  hls::stream< int8_t >& v71,
  hls::stream< int32_t >& v72
) {	// L100
  #pragma HLS array_partition variable=v70 complete dim=1
  #pragma HLS array_partition variable=v70 complete dim=2

  l_S_m_0_m3: for (int m3 = 0; m3 < 6; m3++) {	// L102
  #pragma HLS pipeline II=1
    int8_t v74 = v71.read();	// L103
    int8_t a3;	// L104
    a3 = v74;	// L105
    int32_t p3;	// L106
    p3 = 0;	// L107
    int32_t v77 = p3;	// L108
    int8_t v78 = a3;	// L109
    int8_t v79 = v70[0][3];	// L110
    int16_t v80 = v78;	// L111
    int16_t v81 = v79;	// L112
    int16_t v82 = v80 * v81;	// L113
    ap_int<33> v83 = v77;	// L114
    ap_int<33> v84 = v82;	// L115
    ap_int<33> v85 = v83 + v84;	// L116
    v72.write(v85);	// L117
  }
}

void mac_1_0(
  int8_t v86[4][4],
  hls::stream< int8_t >& v87,
  hls::stream< int32_t >& v88,
  hls::stream< int32_t >& v89,
  hls::stream< int8_t >& v90
) {	// L121
  #pragma HLS array_partition variable=v86 complete dim=1
  #pragma HLS array_partition variable=v86 complete dim=2

  l_S_m_0_m4: for (int m4 = 0; m4 < 6; m4++) {	// L122
  #pragma HLS pipeline II=1
    int8_t v92 = v87.read();	// L123
    int8_t a4;	// L124
    a4 = v92;	// L125
    int32_t v94 = v88.read();	// L126
    int32_t p4;	// L127
    p4 = v94;	// L128
    int32_t v96 = p4;	// L129
    int8_t v97 = a4;	// L130
    int8_t v98 = v86[1][0];	// L131
    int16_t v99 = v97;	// L132
    int16_t v100 = v98;	// L133
    int16_t v101 = v99 * v100;	// L134
    ap_int<33> v102 = v96;	// L135
    ap_int<33> v103 = v101;	// L136
    ap_int<33> v104 = v102 + v103;	// L137
    v89.write(v104);	// L138
    int8_t v105 = a4;	// L139
    v90.write(v105);	// L140
  }
}

void mac_1_1(
  int8_t v106[4][4],
  hls::stream< int8_t >& v107,
  hls::stream< int32_t >& v108,
  hls::stream< int32_t >& v109,
  hls::stream< int8_t >& v110
) {	// L144
  #pragma HLS array_partition variable=v106 complete dim=1
  #pragma HLS array_partition variable=v106 complete dim=2

  l_S_m_0_m5: for (int m5 = 0; m5 < 6; m5++) {	// L145
  #pragma HLS pipeline II=1
    int8_t v112 = v107.read();	// L146
    int8_t a5;	// L147
    a5 = v112;	// L148
    int32_t v114 = v108.read();	// L149
    int32_t p5;	// L150
    p5 = v114;	// L151
    int32_t v116 = p5;	// L152
    int8_t v117 = a5;	// L153
    int8_t v118 = v106[1][1];	// L154
    int16_t v119 = v117;	// L155
    int16_t v120 = v118;	// L156
    int16_t v121 = v119 * v120;	// L157
    ap_int<33> v122 = v116;	// L158
    ap_int<33> v123 = v121;	// L159
    ap_int<33> v124 = v122 + v123;	// L160
    v109.write(v124);	// L161
    int8_t v125 = a5;	// L162
    v110.write(v125);	// L163
  }
}

void mac_1_2(
  int8_t v126[4][4],
  hls::stream< int8_t >& v127,
  hls::stream< int32_t >& v128,
  hls::stream< int32_t >& v129,
  hls::stream< int8_t >& v130
) {	// L167
  #pragma HLS array_partition variable=v126 complete dim=1
  #pragma HLS array_partition variable=v126 complete dim=2

  l_S_m_0_m6: for (int m6 = 0; m6 < 6; m6++) {	// L168
  #pragma HLS pipeline II=1
    int8_t v132 = v127.read();	// L169
    int8_t a6;	// L170
    a6 = v132;	// L171
    int32_t v134 = v128.read();	// L172
    int32_t p6;	// L173
    p6 = v134;	// L174
    int32_t v136 = p6;	// L175
    int8_t v137 = a6;	// L176
    int8_t v138 = v126[1][2];	// L177
    int16_t v139 = v137;	// L178
    int16_t v140 = v138;	// L179
    int16_t v141 = v139 * v140;	// L180
    ap_int<33> v142 = v136;	// L181
    ap_int<33> v143 = v141;	// L182
    ap_int<33> v144 = v142 + v143;	// L183
    v129.write(v144);	// L184
    int8_t v145 = a6;	// L185
    v130.write(v145);	// L186
  }
}

void mac_1_3(
  int8_t v146[4][4],
  hls::stream< int8_t >& v147,
  hls::stream< int32_t >& v148,
  hls::stream< int32_t >& v149
) {	// L190
  #pragma HLS array_partition variable=v146 complete dim=1
  #pragma HLS array_partition variable=v146 complete dim=2

  l_S_m_0_m7: for (int m7 = 0; m7 < 6; m7++) {	// L191
  #pragma HLS pipeline II=1
    int8_t v151 = v147.read();	// L192
    int8_t a7;	// L193
    a7 = v151;	// L194
    int32_t v153 = v148.read();	// L195
    int32_t p7;	// L196
    p7 = v153;	// L197
    int32_t v155 = p7;	// L198
    int8_t v156 = a7;	// L199
    int8_t v157 = v146[1][3];	// L200
    int16_t v158 = v156;	// L201
    int16_t v159 = v157;	// L202
    int16_t v160 = v158 * v159;	// L203
    ap_int<33> v161 = v155;	// L204
    ap_int<33> v162 = v160;	// L205
    ap_int<33> v163 = v161 + v162;	// L206
    v149.write(v163);	// L207
  }
}

void mac_2_0(
  int8_t v164[4][4],
  hls::stream< int8_t >& v165,
  hls::stream< int32_t >& v166,
  hls::stream< int32_t >& v167,
  hls::stream< int8_t >& v168
) {	// L211
  #pragma HLS array_partition variable=v164 complete dim=1
  #pragma HLS array_partition variable=v164 complete dim=2

  l_S_m_0_m8: for (int m8 = 0; m8 < 6; m8++) {	// L212
  #pragma HLS pipeline II=1
    int8_t v170 = v165.read();	// L213
    int8_t a8;	// L214
    a8 = v170;	// L215
    int32_t v172 = v166.read();	// L216
    int32_t p8;	// L217
    p8 = v172;	// L218
    int32_t v174 = p8;	// L219
    int8_t v175 = a8;	// L220
    int8_t v176 = v164[2][0];	// L221
    int16_t v177 = v175;	// L222
    int16_t v178 = v176;	// L223
    int16_t v179 = v177 * v178;	// L224
    ap_int<33> v180 = v174;	// L225
    ap_int<33> v181 = v179;	// L226
    ap_int<33> v182 = v180 + v181;	// L227
    v167.write(v182);	// L228
    int8_t v183 = a8;	// L229
    v168.write(v183);	// L230
  }
}

void mac_2_1(
  int8_t v184[4][4],
  hls::stream< int8_t >& v185,
  hls::stream< int32_t >& v186,
  hls::stream< int32_t >& v187,
  hls::stream< int8_t >& v188
) {	// L234
  #pragma HLS array_partition variable=v184 complete dim=1
  #pragma HLS array_partition variable=v184 complete dim=2

  l_S_m_0_m9: for (int m9 = 0; m9 < 6; m9++) {	// L235
  #pragma HLS pipeline II=1
    int8_t v190 = v185.read();	// L236
    int8_t a9;	// L237
    a9 = v190;	// L238
    int32_t v192 = v186.read();	// L239
    int32_t p9;	// L240
    p9 = v192;	// L241
    int32_t v194 = p9;	// L242
    int8_t v195 = a9;	// L243
    int8_t v196 = v184[2][1];	// L244
    int16_t v197 = v195;	// L245
    int16_t v198 = v196;	// L246
    int16_t v199 = v197 * v198;	// L247
    ap_int<33> v200 = v194;	// L248
    ap_int<33> v201 = v199;	// L249
    ap_int<33> v202 = v200 + v201;	// L250
    v187.write(v202);	// L251
    int8_t v203 = a9;	// L252
    v188.write(v203);	// L253
  }
}

void mac_2_2(
  int8_t v204[4][4],
  hls::stream< int8_t >& v205,
  hls::stream< int32_t >& v206,
  hls::stream< int32_t >& v207,
  hls::stream< int8_t >& v208
) {	// L257
  #pragma HLS array_partition variable=v204 complete dim=1
  #pragma HLS array_partition variable=v204 complete dim=2

  l_S_m_0_m10: for (int m10 = 0; m10 < 6; m10++) {	// L258
  #pragma HLS pipeline II=1
    int8_t v210 = v205.read();	// L259
    int8_t a10;	// L260
    a10 = v210;	// L261
    int32_t v212 = v206.read();	// L262
    int32_t p10;	// L263
    p10 = v212;	// L264
    int32_t v214 = p10;	// L265
    int8_t v215 = a10;	// L266
    int8_t v216 = v204[2][2];	// L267
    int16_t v217 = v215;	// L268
    int16_t v218 = v216;	// L269
    int16_t v219 = v217 * v218;	// L270
    ap_int<33> v220 = v214;	// L271
    ap_int<33> v221 = v219;	// L272
    ap_int<33> v222 = v220 + v221;	// L273
    v207.write(v222);	// L274
    int8_t v223 = a10;	// L275
    v208.write(v223);	// L276
  }
}

void mac_2_3(
  int8_t v224[4][4],
  hls::stream< int8_t >& v225,
  hls::stream< int32_t >& v226,
  hls::stream< int32_t >& v227
) {	// L280
  #pragma HLS array_partition variable=v224 complete dim=1
  #pragma HLS array_partition variable=v224 complete dim=2

  l_S_m_0_m11: for (int m11 = 0; m11 < 6; m11++) {	// L281
  #pragma HLS pipeline II=1
    int8_t v229 = v225.read();	// L282
    int8_t a11;	// L283
    a11 = v229;	// L284
    int32_t v231 = v226.read();	// L285
    int32_t p11;	// L286
    p11 = v231;	// L287
    int32_t v233 = p11;	// L288
    int8_t v234 = a11;	// L289
    int8_t v235 = v224[2][3];	// L290
    int16_t v236 = v234;	// L291
    int16_t v237 = v235;	// L292
    int16_t v238 = v236 * v237;	// L293
    ap_int<33> v239 = v233;	// L294
    ap_int<33> v240 = v238;	// L295
    ap_int<33> v241 = v239 + v240;	// L296
    v227.write(v241);	// L297
  }
}

void mac_3_0(
  int8_t v242[4][4],
  hls::stream< int8_t >& v243,
  hls::stream< int32_t >& v244,
  hls::stream< int32_t >& v245,
  hls::stream< int8_t >& v246
) {	// L301
  #pragma HLS array_partition variable=v242 complete dim=1
  #pragma HLS array_partition variable=v242 complete dim=2

  l_S_m_0_m12: for (int m12 = 0; m12 < 6; m12++) {	// L302
  #pragma HLS pipeline II=1
    int8_t v248 = v243.read();	// L303
    int8_t a12;	// L304
    a12 = v248;	// L305
    int32_t v250 = v244.read();	// L306
    int32_t p12;	// L307
    p12 = v250;	// L308
    int32_t v252 = p12;	// L309
    int8_t v253 = a12;	// L310
    int8_t v254 = v242[3][0];	// L311
    int16_t v255 = v253;	// L312
    int16_t v256 = v254;	// L313
    int16_t v257 = v255 * v256;	// L314
    ap_int<33> v258 = v252;	// L315
    ap_int<33> v259 = v257;	// L316
    ap_int<33> v260 = v258 + v259;	// L317
    v245.write(v260);	// L318
    int8_t v261 = a12;	// L319
    v246.write(v261);	// L320
  }
}

void mac_3_1(
  int8_t v262[4][4],
  hls::stream< int8_t >& v263,
  hls::stream< int32_t >& v264,
  hls::stream< int32_t >& v265,
  hls::stream< int8_t >& v266
) {	// L324
  #pragma HLS array_partition variable=v262 complete dim=1
  #pragma HLS array_partition variable=v262 complete dim=2

  l_S_m_0_m13: for (int m13 = 0; m13 < 6; m13++) {	// L325
  #pragma HLS pipeline II=1
    int8_t v268 = v263.read();	// L326
    int8_t a13;	// L327
    a13 = v268;	// L328
    int32_t v270 = v264.read();	// L329
    int32_t p13;	// L330
    p13 = v270;	// L331
    int32_t v272 = p13;	// L332
    int8_t v273 = a13;	// L333
    int8_t v274 = v262[3][1];	// L334
    int16_t v275 = v273;	// L335
    int16_t v276 = v274;	// L336
    int16_t v277 = v275 * v276;	// L337
    ap_int<33> v278 = v272;	// L338
    ap_int<33> v279 = v277;	// L339
    ap_int<33> v280 = v278 + v279;	// L340
    v265.write(v280);	// L341
    int8_t v281 = a13;	// L342
    v266.write(v281);	// L343
  }
}

void mac_3_2(
  int8_t v282[4][4],
  hls::stream< int8_t >& v283,
  hls::stream< int32_t >& v284,
  hls::stream< int32_t >& v285,
  hls::stream< int8_t >& v286
) {	// L347
  #pragma HLS array_partition variable=v282 complete dim=1
  #pragma HLS array_partition variable=v282 complete dim=2

  l_S_m_0_m14: for (int m14 = 0; m14 < 6; m14++) {	// L348
  #pragma HLS pipeline II=1
    int8_t v288 = v283.read();	// L349
    int8_t a14;	// L350
    a14 = v288;	// L351
    int32_t v290 = v284.read();	// L352
    int32_t p14;	// L353
    p14 = v290;	// L354
    int32_t v292 = p14;	// L355
    int8_t v293 = a14;	// L356
    int8_t v294 = v282[3][2];	// L357
    int16_t v295 = v293;	// L358
    int16_t v296 = v294;	// L359
    int16_t v297 = v295 * v296;	// L360
    ap_int<33> v298 = v292;	// L361
    ap_int<33> v299 = v297;	// L362
    ap_int<33> v300 = v298 + v299;	// L363
    v285.write(v300);	// L364
    int8_t v301 = a14;	// L365
    v286.write(v301);	// L366
  }
}

void mac_3_3(
  int8_t v302[4][4],
  hls::stream< int8_t >& v303,
  hls::stream< int32_t >& v304,
  hls::stream< int32_t >& v305
) {	// L370
  #pragma HLS array_partition variable=v302 complete dim=1
  #pragma HLS array_partition variable=v302 complete dim=2

  l_S_m_0_m15: for (int m15 = 0; m15 < 6; m15++) {	// L371
  #pragma HLS pipeline II=1
    int8_t v307 = v303.read();	// L372
    int8_t a15;	// L373
    a15 = v307;	// L374
    int32_t v309 = v304.read();	// L375
    int32_t p15;	// L376
    p15 = v309;	// L377
    int32_t v311 = p15;	// L378
    int8_t v312 = a15;	// L379
    int8_t v313 = v302[3][3];	// L380
    int16_t v314 = v312;	// L381
    int16_t v315 = v313;	// L382
    int16_t v316 = v314 * v315;	// L383
    ap_int<33> v317 = v311;	// L384
    ap_int<33> v318 = v316;	// L385
    ap_int<33> v319 = v317 + v318;	// L386
    v305.write(v319);	// L387
  }
}

void act_0(
  hls::stream< int32_t >& v320,
  hls::stream< int8_t >& v321
) {	// L391
  l_S_m_0_m16: for (int m16 = 0; m16 < 6; m16++) {	// L394
  #pragma HLS pipeline II=1
    int32_t v323 = v320.read();	// L395
    int32_t z;	// L396
    z = v323;	// L397
    int32_t v325 = z;	// L398
    bool v326 = v325 < 0;	// L399
    if (v326) {	// L400
      z = 0;	// L401
    }
    int32_t v327 = z;	// L403
    int32_t v328 = v327 >> 4;	// L404
    int8_t v329 = v328;	// L405
    int8_t y;	// L406
    y = v329;	// L407
    int8_t v331 = y;	// L408
    v321.write(v331);	// L409
  }
}

void act_1(
  hls::stream< int32_t >& v332,
  hls::stream< int8_t >& v333
) {	// L413
  l_S_m_0_m17: for (int m17 = 0; m17 < 6; m17++) {	// L416
  #pragma HLS pipeline II=1
    int32_t v335 = v332.read();	// L417
    int32_t z1;	// L418
    z1 = v335;	// L419
    int32_t v337 = z1;	// L420
    bool v338 = v337 < 0;	// L421
    if (v338) {	// L422
      z1 = 0;	// L423
    }
    int32_t v339 = z1;	// L425
    int32_t v340 = v339 >> 4;	// L426
    int8_t v341 = v340;	// L427
    int8_t y1;	// L428
    y1 = v341;	// L429
    int8_t v343 = y1;	// L430
    v333.write(v343);	// L431
  }
}

void act_2(
  hls::stream< int32_t >& v344,
  hls::stream< int8_t >& v345
) {	// L435
  l_S_m_0_m18: for (int m18 = 0; m18 < 6; m18++) {	// L438
  #pragma HLS pipeline II=1
    int32_t v347 = v344.read();	// L439
    int32_t z2;	// L440
    z2 = v347;	// L441
    int32_t v349 = z2;	// L442
    bool v350 = v349 < 0;	// L443
    if (v350) {	// L444
      z2 = 0;	// L445
    }
    int32_t v351 = z2;	// L447
    int32_t v352 = v351 >> 4;	// L448
    int8_t v353 = v352;	// L449
    int8_t y2;	// L450
    y2 = v353;	// L451
    int8_t v355 = y2;	// L452
    v345.write(v355);	// L453
  }
}

void act_3(
  hls::stream< int32_t >& v356,
  hls::stream< int8_t >& v357
) {	// L457
  l_S_m_0_m19: for (int m19 = 0; m19 < 6; m19++) {	// L460
  #pragma HLS pipeline II=1
    int32_t v359 = v356.read();	// L461
    int32_t z3;	// L462
    z3 = v359;	// L463
    int32_t v361 = z3;	// L464
    bool v362 = v361 < 0;	// L465
    if (v362) {	// L466
      z3 = 0;	// L467
    }
    int32_t v363 = z3;	// L469
    int32_t v364 = v363 >> 4;	// L470
    int8_t v365 = v364;	// L471
    int8_t y3;	// L472
    y3 = v365;	// L473
    int8_t v367 = y3;	// L474
    v357.write(v367);	// L475
  }
}

void act_y_out_drain_0(
  int8_t v368[6][4],
  hls::stream< int8_t >& v369
) {	// L479
  #pragma HLS array_partition variable=v368 complete dim=1
  #pragma HLS array_partition variable=v368 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 6; _t4++) {	// L480
  #pragma HLS pipeline II=1
    int8_t v371 = v369.read();	// L481
    v368[_t4][0] = v371;	// L482
  }
}

void act_y_out_drain_1(
  int8_t v372[6][4],
  hls::stream< int8_t >& v373
) {	// L486
  #pragma HLS array_partition variable=v372 complete dim=1
  #pragma HLS array_partition variable=v372 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 6; _t5++) {	// L487
  #pragma HLS pipeline II=1
    int8_t v375 = v373.read();	// L488
    v372[_t5][1] = v375;	// L489
  }
}

void act_y_out_drain_2(
  int8_t v376[6][4],
  hls::stream< int8_t >& v377
) {	// L493
  #pragma HLS array_partition variable=v376 complete dim=1
  #pragma HLS array_partition variable=v376 complete dim=2

  l_S__t_0__t6: for (int _t6 = 0; _t6 < 6; _t6++) {	// L494
  #pragma HLS pipeline II=1
    int8_t v379 = v377.read();	// L495
    v376[_t6][2] = v379;	// L496
  }
}

void act_y_out_drain_3(
  int8_t v380[6][4],
  hls::stream< int8_t >& v381
) {	// L500
  #pragma HLS array_partition variable=v380 complete dim=1
  #pragma HLS array_partition variable=v380 complete dim=2

  l_S__t_0__t7: for (int _t7 = 0; _t7 < 6; _t7++) {	// L501
  #pragma HLS pipeline II=1
    int8_t v383 = v381.read();	// L502
    v380[_t7][3] = v383;	// L503
  }
}

/// This is top function.
void top(
  int8_t v384[6][4],
  int8_t v385[4][4],
  int8_t v386[6][4]
) {	// L507
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v384 complete dim=1
  #pragma HLS array_partition variable=v384 complete dim=2

  #pragma HLS array_partition variable=v385 complete dim=1
  #pragma HLS array_partition variable=v385 complete dim=2

  #pragma HLS array_partition variable=v386 complete dim=1
  #pragma HLS array_partition variable=v386 complete dim=2

  hls::stream< int8_t > v387;
  #pragma HLS stream variable=v387 depth=2	// L508
  hls::stream< int8_t > v388;
  #pragma HLS stream variable=v388 depth=2	// L509
  hls::stream< int8_t > v389;
  #pragma HLS stream variable=v389 depth=2	// L510
  hls::stream< int8_t > v390;
  #pragma HLS stream variable=v390 depth=2	// L511
  hls::stream< int8_t > v391;
  #pragma HLS stream variable=v391 depth=2	// L512
  hls::stream< int8_t > v392;
  #pragma HLS stream variable=v392 depth=2	// L513
  hls::stream< int8_t > v393;
  #pragma HLS stream variable=v393 depth=2	// L514
  hls::stream< int8_t > v394;
  #pragma HLS stream variable=v394 depth=2	// L515
  hls::stream< int8_t > v395;
  #pragma HLS stream variable=v395 depth=2	// L516
  hls::stream< int8_t > v396;
  #pragma HLS stream variable=v396 depth=2	// L517
  hls::stream< int8_t > v397;
  #pragma HLS stream variable=v397 depth=2	// L518
  hls::stream< int8_t > v398;
  #pragma HLS stream variable=v398 depth=2	// L519
  hls::stream< int8_t > v399;
  #pragma HLS stream variable=v399 depth=2	// L520
  hls::stream< int8_t > v400;
  #pragma HLS stream variable=v400 depth=2	// L521
  hls::stream< int8_t > v401;
  #pragma HLS stream variable=v401 depth=2	// L522
  hls::stream< int8_t > v402;
  #pragma HLS stream variable=v402 depth=2	// L523
  hls::stream< int32_t > v403;
  #pragma HLS stream variable=v403 depth=2	// L524
  hls::stream< int32_t > v404;
  #pragma HLS stream variable=v404 depth=2	// L525
  hls::stream< int32_t > v405;
  #pragma HLS stream variable=v405 depth=2	// L526
  hls::stream< int32_t > v406;
  #pragma HLS stream variable=v406 depth=2	// L527
  hls::stream< int32_t > v407;
  #pragma HLS stream variable=v407 depth=2	// L528
  hls::stream< int32_t > v408;
  #pragma HLS stream variable=v408 depth=2	// L529
  hls::stream< int32_t > v409;
  #pragma HLS stream variable=v409 depth=2	// L530
  hls::stream< int32_t > v410;
  #pragma HLS stream variable=v410 depth=2	// L531
  hls::stream< int32_t > v411;
  #pragma HLS stream variable=v411 depth=2	// L532
  hls::stream< int32_t > v412;
  #pragma HLS stream variable=v412 depth=2	// L533
  hls::stream< int32_t > v413;
  #pragma HLS stream variable=v413 depth=2	// L534
  hls::stream< int32_t > v414;
  #pragma HLS stream variable=v414 depth=2	// L535
  hls::stream< int32_t > v415;
  #pragma HLS stream variable=v415 depth=2	// L536
  hls::stream< int32_t > v416;
  #pragma HLS stream variable=v416 depth=2	// L537
  hls::stream< int32_t > v417;
  #pragma HLS stream variable=v417 depth=2	// L538
  hls::stream< int32_t > v418;
  #pragma HLS stream variable=v418 depth=2	// L539
  hls::stream< int8_t > v419;
  #pragma HLS stream variable=v419 depth=2	// L540
  hls::stream< int8_t > v420;
  #pragma HLS stream variable=v420 depth=2	// L541
  hls::stream< int8_t > v421;
  #pragma HLS stream variable=v421 depth=2	// L542
  hls::stream< int8_t > v422;
  #pragma HLS stream variable=v422 depth=2	// L543
  hls::stream< int32_t > v423;
  #pragma HLS stream variable=v423 depth=2	// L544
  hls::stream< int32_t > v424;
  #pragma HLS stream variable=v424 depth=2	// L545
  hls::stream< int32_t > v425;
  #pragma HLS stream variable=v425 depth=2	// L546
  hls::stream< int32_t > v426;
  #pragma HLS stream variable=v426 depth=2	// L547
  hls::stream< int8_t > v427;
  #pragma HLS stream variable=v427 depth=2	// L548
  hls::stream< int8_t > v428;
  #pragma HLS stream variable=v428 depth=2	// L549
  hls::stream< int8_t > v429;
  #pragma HLS stream variable=v429 depth=2	// L550
  hls::stream< int8_t > v430;
  #pragma HLS stream variable=v430 depth=2	// L551
  mac_a_in_load_0(v384, v419);	// L552
  mac_a_in_load_1(v384, v420);	// L553
  mac_a_in_load_2(v384, v421);	// L554
  mac_a_in_load_3(v384, v422);	// L555
  mac_0_0(v385, v419, v407, v388);	// L556
  mac_0_1(v385, v388, v408, v389);	// L557
  mac_0_2(v385, v389, v409, v390);	// L558
  mac_0_3(v385, v390, v410);	// L559
  mac_1_0(v385, v420, v407, v411, v392);	// L560
  mac_1_1(v385, v392, v408, v412, v393);	// L561
  mac_1_2(v385, v393, v409, v413, v394);	// L562
  mac_1_3(v385, v394, v410, v414);	// L563
  mac_2_0(v385, v421, v411, v415, v396);	// L564
  mac_2_1(v385, v396, v412, v416, v397);	// L565
  mac_2_2(v385, v397, v413, v417, v398);	// L566
  mac_2_3(v385, v398, v414, v418);	// L567
  mac_3_0(v385, v422, v415, v423, v400);	// L568
  mac_3_1(v385, v400, v416, v424, v401);	// L569
  mac_3_2(v385, v401, v417, v425, v402);	// L570
  mac_3_3(v385, v402, v418, v426);	// L571
  act_0(v423, v427);	// L572
  act_1(v424, v428);	// L573
  act_2(v425, v429);	// L574
  act_3(v426, v430);	// L575
  act_y_out_drain_0(v386, v427);	// L576
  act_y_out_drain_1(v386, v428);	// L577
  act_y_out_drain_2(v386, v429);	// L578
  act_y_out_drain_3(v386, v430);	// L579
}


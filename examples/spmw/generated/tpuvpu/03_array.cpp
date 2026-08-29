
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
) {	// L4
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 6; _t++) {	// L5
  #pragma HLS pipeline II=1
    int8_t v3 = v0[_t][0];	// L6
    v1.write(v3);	// L7
  }
}

void mac_a_in_load_1(
  int8_t v4[6][4],
  hls::stream< int8_t >& v5
) {	// L11
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 6; _t1++) {	// L12
  #pragma HLS pipeline II=1
    int8_t v7 = v4[_t1][1];	// L13
    v5.write(v7);	// L14
  }
}

void mac_a_in_load_2(
  int8_t v8[6][4],
  hls::stream< int8_t >& v9
) {	// L18
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 6; _t2++) {	// L19
  #pragma HLS pipeline II=1
    int8_t v11 = v8[_t2][2];	// L20
    v9.write(v11);	// L21
  }
}

void mac_a_in_load_3(
  int8_t v12[6][4],
  hls::stream< int8_t >& v13
) {	// L25
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 6; _t3++) {	// L26
  #pragma HLS pipeline II=1
    int8_t v15 = v12[_t3][3];	// L27
    v13.write(v15);	// L28
  }
}

void vpu_op_in_load_0(
  int32_t v16[8],
  hls::stream< int32_t >& v17
) {	// L32
  #pragma HLS array_partition variable=v16 complete dim=1

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 8; _t4++) {	// L33
  #pragma HLS pipeline II=1
    int32_t v19 = v16[_t4];	// L34
    v17.write(v19);	// L35
  }
}

void mac_0_0(
  int8_t v20[4][4],
  hls::stream< int8_t >& v21,
  hls::stream< int32_t >& v22,
  hls::stream< int8_t >& v23
) {	// L39
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L41
  #pragma HLS pipeline II=1
    int8_t v25 = v21.read();	// L42
    int8_t a;	// L43
    a = v25;	// L44
    int32_t p;	// L45
    p = 0;	// L46
    int32_t v28 = p;	// L47
    int8_t v29 = a;	// L48
    int8_t v30 = v20[0][0];	// L49
    int16_t v31 = v29;	// L50
    int16_t v32 = v30;	// L51
    int16_t v33 = v31 * v32;	// L52
    ap_int<33> v34 = v28;	// L53
    ap_int<33> v35 = v33;	// L54
    ap_int<33> v36 = v34 + v35;	// L55
    v22.write(v36);	// L56
    int8_t v37 = a;	// L57
    v23.write(v37);	// L58
  }
}

void mac_0_1(
  int8_t v38[4][4],
  hls::stream< int8_t >& v39,
  hls::stream< int32_t >& v40,
  hls::stream< int8_t >& v41
) {	// L62
  #pragma HLS array_partition variable=v38 complete dim=1
  #pragma HLS array_partition variable=v38 complete dim=2

  l_S_m_0_m1: for (int m1 = 0; m1 < 6; m1++) {	// L64
  #pragma HLS pipeline II=1
    int8_t v43 = v39.read();	// L65
    int8_t a1;	// L66
    a1 = v43;	// L67
    int32_t p1;	// L68
    p1 = 0;	// L69
    int32_t v46 = p1;	// L70
    int8_t v47 = a1;	// L71
    int8_t v48 = v38[0][1];	// L72
    int16_t v49 = v47;	// L73
    int16_t v50 = v48;	// L74
    int16_t v51 = v49 * v50;	// L75
    ap_int<33> v52 = v46;	// L76
    ap_int<33> v53 = v51;	// L77
    ap_int<33> v54 = v52 + v53;	// L78
    v40.write(v54);	// L79
    int8_t v55 = a1;	// L80
    v41.write(v55);	// L81
  }
}

void mac_0_2(
  int8_t v56[4][4],
  hls::stream< int8_t >& v57,
  hls::stream< int32_t >& v58,
  hls::stream< int8_t >& v59
) {	// L85
  #pragma HLS array_partition variable=v56 complete dim=1
  #pragma HLS array_partition variable=v56 complete dim=2

  l_S_m_0_m2: for (int m2 = 0; m2 < 6; m2++) {	// L87
  #pragma HLS pipeline II=1
    int8_t v61 = v57.read();	// L88
    int8_t a2;	// L89
    a2 = v61;	// L90
    int32_t p2;	// L91
    p2 = 0;	// L92
    int32_t v64 = p2;	// L93
    int8_t v65 = a2;	// L94
    int8_t v66 = v56[0][2];	// L95
    int16_t v67 = v65;	// L96
    int16_t v68 = v66;	// L97
    int16_t v69 = v67 * v68;	// L98
    ap_int<33> v70 = v64;	// L99
    ap_int<33> v71 = v69;	// L100
    ap_int<33> v72 = v70 + v71;	// L101
    v58.write(v72);	// L102
    int8_t v73 = a2;	// L103
    v59.write(v73);	// L104
  }
}

void mac_0_3(
  int8_t v74[4][4],
  hls::stream< int8_t >& v75,
  hls::stream< int32_t >& v76
) {	// L108
  #pragma HLS array_partition variable=v74 complete dim=1
  #pragma HLS array_partition variable=v74 complete dim=2

  l_S_m_0_m3: for (int m3 = 0; m3 < 6; m3++) {	// L110
  #pragma HLS pipeline II=1
    int8_t v78 = v75.read();	// L111
    int8_t a3;	// L112
    a3 = v78;	// L113
    int32_t p3;	// L114
    p3 = 0;	// L115
    int32_t v81 = p3;	// L116
    int8_t v82 = a3;	// L117
    int8_t v83 = v74[0][3];	// L118
    int16_t v84 = v82;	// L119
    int16_t v85 = v83;	// L120
    int16_t v86 = v84 * v85;	// L121
    ap_int<33> v87 = v81;	// L122
    ap_int<33> v88 = v86;	// L123
    ap_int<33> v89 = v87 + v88;	// L124
    v76.write(v89);	// L125
  }
}

void mac_1_0(
  int8_t v90[4][4],
  hls::stream< int8_t >& v91,
  hls::stream< int32_t >& v92,
  hls::stream< int32_t >& v93,
  hls::stream< int8_t >& v94
) {	// L129
  #pragma HLS array_partition variable=v90 complete dim=1
  #pragma HLS array_partition variable=v90 complete dim=2

  l_S_m_0_m4: for (int m4 = 0; m4 < 6; m4++) {	// L130
  #pragma HLS pipeline II=1
    int8_t v96 = v91.read();	// L131
    int8_t a4;	// L132
    a4 = v96;	// L133
    int32_t v98 = v92.read();	// L134
    int32_t p4;	// L135
    p4 = v98;	// L136
    int32_t v100 = p4;	// L137
    int8_t v101 = a4;	// L138
    int8_t v102 = v90[1][0];	// L139
    int16_t v103 = v101;	// L140
    int16_t v104 = v102;	// L141
    int16_t v105 = v103 * v104;	// L142
    ap_int<33> v106 = v100;	// L143
    ap_int<33> v107 = v105;	// L144
    ap_int<33> v108 = v106 + v107;	// L145
    v93.write(v108);	// L146
    int8_t v109 = a4;	// L147
    v94.write(v109);	// L148
  }
}

void mac_1_1(
  int8_t v110[4][4],
  hls::stream< int8_t >& v111,
  hls::stream< int32_t >& v112,
  hls::stream< int32_t >& v113,
  hls::stream< int8_t >& v114
) {	// L152
  #pragma HLS array_partition variable=v110 complete dim=1
  #pragma HLS array_partition variable=v110 complete dim=2

  l_S_m_0_m5: for (int m5 = 0; m5 < 6; m5++) {	// L153
  #pragma HLS pipeline II=1
    int8_t v116 = v111.read();	// L154
    int8_t a5;	// L155
    a5 = v116;	// L156
    int32_t v118 = v112.read();	// L157
    int32_t p5;	// L158
    p5 = v118;	// L159
    int32_t v120 = p5;	// L160
    int8_t v121 = a5;	// L161
    int8_t v122 = v110[1][1];	// L162
    int16_t v123 = v121;	// L163
    int16_t v124 = v122;	// L164
    int16_t v125 = v123 * v124;	// L165
    ap_int<33> v126 = v120;	// L166
    ap_int<33> v127 = v125;	// L167
    ap_int<33> v128 = v126 + v127;	// L168
    v113.write(v128);	// L169
    int8_t v129 = a5;	// L170
    v114.write(v129);	// L171
  }
}

void mac_1_2(
  int8_t v130[4][4],
  hls::stream< int8_t >& v131,
  hls::stream< int32_t >& v132,
  hls::stream< int32_t >& v133,
  hls::stream< int8_t >& v134
) {	// L175
  #pragma HLS array_partition variable=v130 complete dim=1
  #pragma HLS array_partition variable=v130 complete dim=2

  l_S_m_0_m6: for (int m6 = 0; m6 < 6; m6++) {	// L176
  #pragma HLS pipeline II=1
    int8_t v136 = v131.read();	// L177
    int8_t a6;	// L178
    a6 = v136;	// L179
    int32_t v138 = v132.read();	// L180
    int32_t p6;	// L181
    p6 = v138;	// L182
    int32_t v140 = p6;	// L183
    int8_t v141 = a6;	// L184
    int8_t v142 = v130[1][2];	// L185
    int16_t v143 = v141;	// L186
    int16_t v144 = v142;	// L187
    int16_t v145 = v143 * v144;	// L188
    ap_int<33> v146 = v140;	// L189
    ap_int<33> v147 = v145;	// L190
    ap_int<33> v148 = v146 + v147;	// L191
    v133.write(v148);	// L192
    int8_t v149 = a6;	// L193
    v134.write(v149);	// L194
  }
}

void mac_1_3(
  int8_t v150[4][4],
  hls::stream< int8_t >& v151,
  hls::stream< int32_t >& v152,
  hls::stream< int32_t >& v153
) {	// L198
  #pragma HLS array_partition variable=v150 complete dim=1
  #pragma HLS array_partition variable=v150 complete dim=2

  l_S_m_0_m7: for (int m7 = 0; m7 < 6; m7++) {	// L199
  #pragma HLS pipeline II=1
    int8_t v155 = v151.read();	// L200
    int8_t a7;	// L201
    a7 = v155;	// L202
    int32_t v157 = v152.read();	// L203
    int32_t p7;	// L204
    p7 = v157;	// L205
    int32_t v159 = p7;	// L206
    int8_t v160 = a7;	// L207
    int8_t v161 = v150[1][3];	// L208
    int16_t v162 = v160;	// L209
    int16_t v163 = v161;	// L210
    int16_t v164 = v162 * v163;	// L211
    ap_int<33> v165 = v159;	// L212
    ap_int<33> v166 = v164;	// L213
    ap_int<33> v167 = v165 + v166;	// L214
    v153.write(v167);	// L215
  }
}

void mac_2_0(
  int8_t v168[4][4],
  hls::stream< int8_t >& v169,
  hls::stream< int32_t >& v170,
  hls::stream< int32_t >& v171,
  hls::stream< int8_t >& v172
) {	// L219
  #pragma HLS array_partition variable=v168 complete dim=1
  #pragma HLS array_partition variable=v168 complete dim=2

  l_S_m_0_m8: for (int m8 = 0; m8 < 6; m8++) {	// L220
  #pragma HLS pipeline II=1
    int8_t v174 = v169.read();	// L221
    int8_t a8;	// L222
    a8 = v174;	// L223
    int32_t v176 = v170.read();	// L224
    int32_t p8;	// L225
    p8 = v176;	// L226
    int32_t v178 = p8;	// L227
    int8_t v179 = a8;	// L228
    int8_t v180 = v168[2][0];	// L229
    int16_t v181 = v179;	// L230
    int16_t v182 = v180;	// L231
    int16_t v183 = v181 * v182;	// L232
    ap_int<33> v184 = v178;	// L233
    ap_int<33> v185 = v183;	// L234
    ap_int<33> v186 = v184 + v185;	// L235
    v171.write(v186);	// L236
    int8_t v187 = a8;	// L237
    v172.write(v187);	// L238
  }
}

void mac_2_1(
  int8_t v188[4][4],
  hls::stream< int8_t >& v189,
  hls::stream< int32_t >& v190,
  hls::stream< int32_t >& v191,
  hls::stream< int8_t >& v192
) {	// L242
  #pragma HLS array_partition variable=v188 complete dim=1
  #pragma HLS array_partition variable=v188 complete dim=2

  l_S_m_0_m9: for (int m9 = 0; m9 < 6; m9++) {	// L243
  #pragma HLS pipeline II=1
    int8_t v194 = v189.read();	// L244
    int8_t a9;	// L245
    a9 = v194;	// L246
    int32_t v196 = v190.read();	// L247
    int32_t p9;	// L248
    p9 = v196;	// L249
    int32_t v198 = p9;	// L250
    int8_t v199 = a9;	// L251
    int8_t v200 = v188[2][1];	// L252
    int16_t v201 = v199;	// L253
    int16_t v202 = v200;	// L254
    int16_t v203 = v201 * v202;	// L255
    ap_int<33> v204 = v198;	// L256
    ap_int<33> v205 = v203;	// L257
    ap_int<33> v206 = v204 + v205;	// L258
    v191.write(v206);	// L259
    int8_t v207 = a9;	// L260
    v192.write(v207);	// L261
  }
}

void mac_2_2(
  int8_t v208[4][4],
  hls::stream< int8_t >& v209,
  hls::stream< int32_t >& v210,
  hls::stream< int32_t >& v211,
  hls::stream< int8_t >& v212
) {	// L265
  #pragma HLS array_partition variable=v208 complete dim=1
  #pragma HLS array_partition variable=v208 complete dim=2

  l_S_m_0_m10: for (int m10 = 0; m10 < 6; m10++) {	// L266
  #pragma HLS pipeline II=1
    int8_t v214 = v209.read();	// L267
    int8_t a10;	// L268
    a10 = v214;	// L269
    int32_t v216 = v210.read();	// L270
    int32_t p10;	// L271
    p10 = v216;	// L272
    int32_t v218 = p10;	// L273
    int8_t v219 = a10;	// L274
    int8_t v220 = v208[2][2];	// L275
    int16_t v221 = v219;	// L276
    int16_t v222 = v220;	// L277
    int16_t v223 = v221 * v222;	// L278
    ap_int<33> v224 = v218;	// L279
    ap_int<33> v225 = v223;	// L280
    ap_int<33> v226 = v224 + v225;	// L281
    v211.write(v226);	// L282
    int8_t v227 = a10;	// L283
    v212.write(v227);	// L284
  }
}

void mac_2_3(
  int8_t v228[4][4],
  hls::stream< int8_t >& v229,
  hls::stream< int32_t >& v230,
  hls::stream< int32_t >& v231
) {	// L288
  #pragma HLS array_partition variable=v228 complete dim=1
  #pragma HLS array_partition variable=v228 complete dim=2

  l_S_m_0_m11: for (int m11 = 0; m11 < 6; m11++) {	// L289
  #pragma HLS pipeline II=1
    int8_t v233 = v229.read();	// L290
    int8_t a11;	// L291
    a11 = v233;	// L292
    int32_t v235 = v230.read();	// L293
    int32_t p11;	// L294
    p11 = v235;	// L295
    int32_t v237 = p11;	// L296
    int8_t v238 = a11;	// L297
    int8_t v239 = v228[2][3];	// L298
    int16_t v240 = v238;	// L299
    int16_t v241 = v239;	// L300
    int16_t v242 = v240 * v241;	// L301
    ap_int<33> v243 = v237;	// L302
    ap_int<33> v244 = v242;	// L303
    ap_int<33> v245 = v243 + v244;	// L304
    v231.write(v245);	// L305
  }
}

void mac_3_0(
  int8_t v246[4][4],
  hls::stream< int8_t >& v247,
  hls::stream< int32_t >& v248,
  hls::stream< int32_t >& v249,
  hls::stream< int8_t >& v250
) {	// L309
  #pragma HLS array_partition variable=v246 complete dim=1
  #pragma HLS array_partition variable=v246 complete dim=2

  l_S_m_0_m12: for (int m12 = 0; m12 < 6; m12++) {	// L310
  #pragma HLS pipeline II=1
    int8_t v252 = v247.read();	// L311
    int8_t a12;	// L312
    a12 = v252;	// L313
    int32_t v254 = v248.read();	// L314
    int32_t p12;	// L315
    p12 = v254;	// L316
    int32_t v256 = p12;	// L317
    int8_t v257 = a12;	// L318
    int8_t v258 = v246[3][0];	// L319
    int16_t v259 = v257;	// L320
    int16_t v260 = v258;	// L321
    int16_t v261 = v259 * v260;	// L322
    ap_int<33> v262 = v256;	// L323
    ap_int<33> v263 = v261;	// L324
    ap_int<33> v264 = v262 + v263;	// L325
    v249.write(v264);	// L326
    int8_t v265 = a12;	// L327
    v250.write(v265);	// L328
  }
}

void mac_3_1(
  int8_t v266[4][4],
  hls::stream< int8_t >& v267,
  hls::stream< int32_t >& v268,
  hls::stream< int32_t >& v269,
  hls::stream< int8_t >& v270
) {	// L332
  #pragma HLS array_partition variable=v266 complete dim=1
  #pragma HLS array_partition variable=v266 complete dim=2

  l_S_m_0_m13: for (int m13 = 0; m13 < 6; m13++) {	// L333
  #pragma HLS pipeline II=1
    int8_t v272 = v267.read();	// L334
    int8_t a13;	// L335
    a13 = v272;	// L336
    int32_t v274 = v268.read();	// L337
    int32_t p13;	// L338
    p13 = v274;	// L339
    int32_t v276 = p13;	// L340
    int8_t v277 = a13;	// L341
    int8_t v278 = v266[3][1];	// L342
    int16_t v279 = v277;	// L343
    int16_t v280 = v278;	// L344
    int16_t v281 = v279 * v280;	// L345
    ap_int<33> v282 = v276;	// L346
    ap_int<33> v283 = v281;	// L347
    ap_int<33> v284 = v282 + v283;	// L348
    v269.write(v284);	// L349
    int8_t v285 = a13;	// L350
    v270.write(v285);	// L351
  }
}

void mac_3_2(
  int8_t v286[4][4],
  hls::stream< int8_t >& v287,
  hls::stream< int32_t >& v288,
  hls::stream< int32_t >& v289,
  hls::stream< int8_t >& v290
) {	// L355
  #pragma HLS array_partition variable=v286 complete dim=1
  #pragma HLS array_partition variable=v286 complete dim=2

  l_S_m_0_m14: for (int m14 = 0; m14 < 6; m14++) {	// L356
  #pragma HLS pipeline II=1
    int8_t v292 = v287.read();	// L357
    int8_t a14;	// L358
    a14 = v292;	// L359
    int32_t v294 = v288.read();	// L360
    int32_t p14;	// L361
    p14 = v294;	// L362
    int32_t v296 = p14;	// L363
    int8_t v297 = a14;	// L364
    int8_t v298 = v286[3][2];	// L365
    int16_t v299 = v297;	// L366
    int16_t v300 = v298;	// L367
    int16_t v301 = v299 * v300;	// L368
    ap_int<33> v302 = v296;	// L369
    ap_int<33> v303 = v301;	// L370
    ap_int<33> v304 = v302 + v303;	// L371
    v289.write(v304);	// L372
    int8_t v305 = a14;	// L373
    v290.write(v305);	// L374
  }
}

void mac_3_3(
  int8_t v306[4][4],
  hls::stream< int8_t >& v307,
  hls::stream< int32_t >& v308,
  hls::stream< int32_t >& v309
) {	// L378
  #pragma HLS array_partition variable=v306 complete dim=1
  #pragma HLS array_partition variable=v306 complete dim=2

  l_S_m_0_m15: for (int m15 = 0; m15 < 6; m15++) {	// L379
  #pragma HLS pipeline II=1
    int8_t v311 = v307.read();	// L380
    int8_t a15;	// L381
    a15 = v311;	// L382
    int32_t v313 = v308.read();	// L383
    int32_t p15;	// L384
    p15 = v313;	// L385
    int32_t v315 = p15;	// L386
    int8_t v316 = a15;	// L387
    int8_t v317 = v306[3][3];	// L388
    int16_t v318 = v316;	// L389
    int16_t v319 = v317;	// L390
    int16_t v320 = v318 * v319;	// L391
    ap_int<33> v321 = v315;	// L392
    ap_int<33> v322 = v320;	// L393
    ap_int<33> v323 = v321 + v322;	// L394
    v309.write(v323);	// L395
  }
}

void vpu_0(
  int32_t v324[4],
  hls::stream< int32_t >& v325,
  hls::stream< int32_t >& v326,
  hls::stream< int32_t >& v327,
  hls::stream< int32_t >& v328
) {	// L399
  #pragma HLS array_partition variable=v324 complete dim=1

  int32_t prog[8];	// L415
  for (int v330 = 0; v330 < 8; v330++) {	// L416
    prog[v330] = 0;	// L416
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 8; pc++) {	// L417
  #pragma HLS pipeline II=1
    int32_t v332 = v325.read();	// L418
    int32_t word;	// L419
    word = v332;	// L420
    int32_t v334 = word;	// L421
    prog[pc] = v334;	// L422
    int32_t v335 = word;	// L423
    v326.write(v335);	// L424
  }
  l_S_m_1_m16: for (int m16 = 0; m16 < 6; m16++) {	// L426
    int32_t v337 = v327.read();	// L427
    int32_t z;	// L428
    z = v337;	// L429
    int32_t reg[4];	// L430
    for (int v340 = 0; v340 < 4; v340++) {	// L431
      reg[v340] = 0;	// L431
    }
    l_S_step_1_step: for (int step = 0; step < 8; step++) {	// L432
    #pragma HLS pipeline II=1
      int32_t v342 = prog[step];	// L433
      int32_t word2;	// L434
      word2 = v342;	// L435
      int32_t v344 = word2;	// L436
      int32_t v345 = v344 >> 24;	// L437
      int32_t v346 = v345 & 255;	// L438
      int32_t opcode;	// L439
      opcode = v346;	// L440
      int32_t v348 = word2;	// L441
      int32_t v349 = v348 >> 20;	// L442
      int32_t v350 = v349 & 15;	// L443
      int32_t dst;	// L444
      dst = v350;	// L445
      int32_t v352 = word2;	// L446
      int32_t v353 = v352 >> 16;	// L447
      int32_t v354 = v353 & 15;	// L448
      int32_t src;	// L449
      src = v354;	// L450
      int32_t v356 = word2;	// L451
      int32_t v357 = v356 & 65535;	// L452
      int32_t imm;	// L453
      imm = v357;	// L454
      int32_t v359 = opcode;	// L455
      bool v360 = v359 == 1;	// L456
      if (v360) {	// L457
        int32_t v361 = z;	// L458
        int32_t v362 = dst;	// L459
        int v363 = v362;	// L460
        reg[v363] = v361;	// L461
      } else {
        int32_t v364 = opcode;	// L463
        bool v365 = v364 == 2;	// L464
        if (v365) {	// L465
          int32_t v366 = v324[0];	// L466
          int32_t v367 = dst;	// L467
          int v368 = v367;	// L468
          reg[v368] = v366;	// L469
        } else {
          int32_t v369 = opcode;	// L471
          bool v370 = v369 == 3;	// L472
          if (v370) {	// L473
            int32_t v371 = imm;	// L474
            int32_t v372 = dst;	// L475
            int v373 = v372;	// L476
            reg[v373] = v371;	// L477
          } else {
            int32_t v374 = opcode;	// L479
            bool v375 = v374 == 4;	// L480
            if (v375) {	// L481
              int32_t v376 = dst;	// L482
              int v377 = v376;	// L483
              int32_t v378 = reg[v377];	// L484
              int32_t v379 = src;	// L485
              int v380 = v379;	// L486
              int32_t v381 = reg[v380];	// L487
              ap_int<33> v382 = v378;	// L488
              ap_int<33> v383 = v381;	// L489
              ap_int<33> v384 = v382 + v383;	// L490
              int32_t v385 = v384;	// L491
              reg[v377] = v385;	// L492
            } else {
              int32_t v386 = opcode;	// L494
              bool v387 = v386 == 5;	// L495
              if (v387) {	// L496
                int32_t v388 = dst;	// L497
                int v389 = v388;	// L498
                int32_t v390 = reg[v389];	// L499
                int32_t v391 = src;	// L500
                int v392 = v391;	// L501
                int32_t v393 = reg[v392];	// L502
                int64_t v394 = v390;	// L503
                int64_t v395 = v393;	// L504
                int64_t v396 = v394 * v395;	// L505
                int32_t v397 = v396;	// L506
                reg[v389] = v397;	// L507
              } else {
                int32_t v398 = opcode;	// L509
                bool v399 = v398 == 6;	// L510
                if (v399) {	// L511
                  int32_t v400 = src;	// L512
                  int v401 = v400;	// L513
                  int32_t v402 = reg[v401];	// L514
                  int32_t v403 = dst;	// L515
                  int v404 = v403;	// L516
                  int32_t v405 = reg[v404];	// L517
                  bool v406 = v402 > v405;	// L518
                  if (v406) {	// L519
                    int32_t v407 = src;	// L520
                    int v408 = v407;	// L521
                    int32_t v409 = reg[v408];	// L522
                    int32_t v410 = dst;	// L523
                    int v411 = v410;	// L524
                    reg[v411] = v409;	// L525
                  }
                } else {
                  int32_t v412 = opcode;	// L528
                  bool v413 = v412 == 7;	// L529
                  if (v413) {	// L530
                    int32_t v414 = dst;	// L531
                    int v415 = v414;	// L532
                    int32_t v416 = reg[v415];	// L533
                    int32_t v417 = imm;	// L534
                    int32_t v418 = v416 >> v417;	// L535
                    reg[v415] = v418;	// L536
                  } else {
                    int32_t v419 = opcode;	// L538
                    bool v420 = v419 == 8;	// L539
                    if (v420) {	// L540
                      int32_t v421 = dst;	// L541
                      int v422 = v421;	// L542
                      int32_t v423 = reg[v422];	// L543
                      v328.write(v423);	// L544
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void vpu_1(
  int32_t v424[4],
  hls::stream< int32_t >& v425,
  hls::stream< int32_t >& v426,
  hls::stream< int32_t >& v427,
  hls::stream< int32_t >& v428
) {	// L557
  #pragma HLS array_partition variable=v424 complete dim=1

  int32_t prog1[8];	// L573
  for (int v430 = 0; v430 < 8; v430++) {	// L574
    prog1[v430] = 0;	// L574
  }
  l_S_pc_0_pc1: for (int pc1 = 0; pc1 < 8; pc1++) {	// L575
  #pragma HLS pipeline II=1
    int32_t v432 = v425.read();	// L576
    int32_t word1;	// L577
    word1 = v432;	// L578
    int32_t v434 = word1;	// L579
    prog1[pc1] = v434;	// L580
    int32_t v435 = word1;	// L581
    v426.write(v435);	// L582
  }
  l_S_m_1_m17: for (int m17 = 0; m17 < 6; m17++) {	// L584
    int32_t v437 = v427.read();	// L585
    int32_t z1;	// L586
    z1 = v437;	// L587
    int32_t reg1[4];	// L588
    for (int v440 = 0; v440 < 4; v440++) {	// L589
      reg1[v440] = 0;	// L589
    }
    l_S_step_1_step1: for (int step1 = 0; step1 < 8; step1++) {	// L590
    #pragma HLS pipeline II=1
      int32_t v442 = prog1[step1];	// L591
      int32_t word21;	// L592
      word21 = v442;	// L593
      int32_t v444 = word21;	// L594
      int32_t v445 = v444 >> 24;	// L595
      int32_t v446 = v445 & 255;	// L596
      int32_t opcode1;	// L597
      opcode1 = v446;	// L598
      int32_t v448 = word21;	// L599
      int32_t v449 = v448 >> 20;	// L600
      int32_t v450 = v449 & 15;	// L601
      int32_t dst1;	// L602
      dst1 = v450;	// L603
      int32_t v452 = word21;	// L604
      int32_t v453 = v452 >> 16;	// L605
      int32_t v454 = v453 & 15;	// L606
      int32_t src1;	// L607
      src1 = v454;	// L608
      int32_t v456 = word21;	// L609
      int32_t v457 = v456 & 65535;	// L610
      int32_t imm1;	// L611
      imm1 = v457;	// L612
      int32_t v459 = opcode1;	// L613
      bool v460 = v459 == 1;	// L614
      if (v460) {	// L615
        int32_t v461 = z1;	// L616
        int32_t v462 = dst1;	// L617
        int v463 = v462;	// L618
        reg1[v463] = v461;	// L619
      } else {
        int32_t v464 = opcode1;	// L621
        bool v465 = v464 == 2;	// L622
        if (v465) {	// L623
          int32_t v466 = v424[1];	// L624
          int32_t v467 = dst1;	// L625
          int v468 = v467;	// L626
          reg1[v468] = v466;	// L627
        } else {
          int32_t v469 = opcode1;	// L629
          bool v470 = v469 == 3;	// L630
          if (v470) {	// L631
            int32_t v471 = imm1;	// L632
            int32_t v472 = dst1;	// L633
            int v473 = v472;	// L634
            reg1[v473] = v471;	// L635
          } else {
            int32_t v474 = opcode1;	// L637
            bool v475 = v474 == 4;	// L638
            if (v475) {	// L639
              int32_t v476 = dst1;	// L640
              int v477 = v476;	// L641
              int32_t v478 = reg1[v477];	// L642
              int32_t v479 = src1;	// L643
              int v480 = v479;	// L644
              int32_t v481 = reg1[v480];	// L645
              ap_int<33> v482 = v478;	// L646
              ap_int<33> v483 = v481;	// L647
              ap_int<33> v484 = v482 + v483;	// L648
              int32_t v485 = v484;	// L649
              reg1[v477] = v485;	// L650
            } else {
              int32_t v486 = opcode1;	// L652
              bool v487 = v486 == 5;	// L653
              if (v487) {	// L654
                int32_t v488 = dst1;	// L655
                int v489 = v488;	// L656
                int32_t v490 = reg1[v489];	// L657
                int32_t v491 = src1;	// L658
                int v492 = v491;	// L659
                int32_t v493 = reg1[v492];	// L660
                int64_t v494 = v490;	// L661
                int64_t v495 = v493;	// L662
                int64_t v496 = v494 * v495;	// L663
                int32_t v497 = v496;	// L664
                reg1[v489] = v497;	// L665
              } else {
                int32_t v498 = opcode1;	// L667
                bool v499 = v498 == 6;	// L668
                if (v499) {	// L669
                  int32_t v500 = src1;	// L670
                  int v501 = v500;	// L671
                  int32_t v502 = reg1[v501];	// L672
                  int32_t v503 = dst1;	// L673
                  int v504 = v503;	// L674
                  int32_t v505 = reg1[v504];	// L675
                  bool v506 = v502 > v505;	// L676
                  if (v506) {	// L677
                    int32_t v507 = src1;	// L678
                    int v508 = v507;	// L679
                    int32_t v509 = reg1[v508];	// L680
                    int32_t v510 = dst1;	// L681
                    int v511 = v510;	// L682
                    reg1[v511] = v509;	// L683
                  }
                } else {
                  int32_t v512 = opcode1;	// L686
                  bool v513 = v512 == 7;	// L687
                  if (v513) {	// L688
                    int32_t v514 = dst1;	// L689
                    int v515 = v514;	// L690
                    int32_t v516 = reg1[v515];	// L691
                    int32_t v517 = imm1;	// L692
                    int32_t v518 = v516 >> v517;	// L693
                    reg1[v515] = v518;	// L694
                  } else {
                    int32_t v519 = opcode1;	// L696
                    bool v520 = v519 == 8;	// L697
                    if (v520) {	// L698
                      int32_t v521 = dst1;	// L699
                      int v522 = v521;	// L700
                      int32_t v523 = reg1[v522];	// L701
                      v428.write(v523);	// L702
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void vpu_2(
  int32_t v524[4],
  hls::stream< int32_t >& v525,
  hls::stream< int32_t >& v526,
  hls::stream< int32_t >& v527,
  hls::stream< int32_t >& v528
) {	// L715
  #pragma HLS array_partition variable=v524 complete dim=1

  int32_t prog2[8];	// L731
  for (int v530 = 0; v530 < 8; v530++) {	// L732
    prog2[v530] = 0;	// L732
  }
  l_S_pc_0_pc2: for (int pc2 = 0; pc2 < 8; pc2++) {	// L733
  #pragma HLS pipeline II=1
    int32_t v532 = v525.read();	// L734
    int32_t word2;	// L735
    word2 = v532;	// L736
    int32_t v534 = word2;	// L737
    prog2[pc2] = v534;	// L738
    int32_t v535 = word2;	// L739
    v526.write(v535);	// L740
  }
  l_S_m_1_m18: for (int m18 = 0; m18 < 6; m18++) {	// L742
    int32_t v537 = v527.read();	// L743
    int32_t z2;	// L744
    z2 = v537;	// L745
    int32_t reg2[4];	// L746
    for (int v540 = 0; v540 < 4; v540++) {	// L747
      reg2[v540] = 0;	// L747
    }
    l_S_step_1_step2: for (int step2 = 0; step2 < 8; step2++) {	// L748
    #pragma HLS pipeline II=1
      int32_t v542 = prog2[step2];	// L749
      int32_t word22;	// L750
      word22 = v542;	// L751
      int32_t v544 = word22;	// L752
      int32_t v545 = v544 >> 24;	// L753
      int32_t v546 = v545 & 255;	// L754
      int32_t opcode2;	// L755
      opcode2 = v546;	// L756
      int32_t v548 = word22;	// L757
      int32_t v549 = v548 >> 20;	// L758
      int32_t v550 = v549 & 15;	// L759
      int32_t dst2;	// L760
      dst2 = v550;	// L761
      int32_t v552 = word22;	// L762
      int32_t v553 = v552 >> 16;	// L763
      int32_t v554 = v553 & 15;	// L764
      int32_t src2;	// L765
      src2 = v554;	// L766
      int32_t v556 = word22;	// L767
      int32_t v557 = v556 & 65535;	// L768
      int32_t imm2;	// L769
      imm2 = v557;	// L770
      int32_t v559 = opcode2;	// L771
      bool v560 = v559 == 1;	// L772
      if (v560) {	// L773
        int32_t v561 = z2;	// L774
        int32_t v562 = dst2;	// L775
        int v563 = v562;	// L776
        reg2[v563] = v561;	// L777
      } else {
        int32_t v564 = opcode2;	// L779
        bool v565 = v564 == 2;	// L780
        if (v565) {	// L781
          int32_t v566 = v524[2];	// L782
          int32_t v567 = dst2;	// L783
          int v568 = v567;	// L784
          reg2[v568] = v566;	// L785
        } else {
          int32_t v569 = opcode2;	// L787
          bool v570 = v569 == 3;	// L788
          if (v570) {	// L789
            int32_t v571 = imm2;	// L790
            int32_t v572 = dst2;	// L791
            int v573 = v572;	// L792
            reg2[v573] = v571;	// L793
          } else {
            int32_t v574 = opcode2;	// L795
            bool v575 = v574 == 4;	// L796
            if (v575) {	// L797
              int32_t v576 = dst2;	// L798
              int v577 = v576;	// L799
              int32_t v578 = reg2[v577];	// L800
              int32_t v579 = src2;	// L801
              int v580 = v579;	// L802
              int32_t v581 = reg2[v580];	// L803
              ap_int<33> v582 = v578;	// L804
              ap_int<33> v583 = v581;	// L805
              ap_int<33> v584 = v582 + v583;	// L806
              int32_t v585 = v584;	// L807
              reg2[v577] = v585;	// L808
            } else {
              int32_t v586 = opcode2;	// L810
              bool v587 = v586 == 5;	// L811
              if (v587) {	// L812
                int32_t v588 = dst2;	// L813
                int v589 = v588;	// L814
                int32_t v590 = reg2[v589];	// L815
                int32_t v591 = src2;	// L816
                int v592 = v591;	// L817
                int32_t v593 = reg2[v592];	// L818
                int64_t v594 = v590;	// L819
                int64_t v595 = v593;	// L820
                int64_t v596 = v594 * v595;	// L821
                int32_t v597 = v596;	// L822
                reg2[v589] = v597;	// L823
              } else {
                int32_t v598 = opcode2;	// L825
                bool v599 = v598 == 6;	// L826
                if (v599) {	// L827
                  int32_t v600 = src2;	// L828
                  int v601 = v600;	// L829
                  int32_t v602 = reg2[v601];	// L830
                  int32_t v603 = dst2;	// L831
                  int v604 = v603;	// L832
                  int32_t v605 = reg2[v604];	// L833
                  bool v606 = v602 > v605;	// L834
                  if (v606) {	// L835
                    int32_t v607 = src2;	// L836
                    int v608 = v607;	// L837
                    int32_t v609 = reg2[v608];	// L838
                    int32_t v610 = dst2;	// L839
                    int v611 = v610;	// L840
                    reg2[v611] = v609;	// L841
                  }
                } else {
                  int32_t v612 = opcode2;	// L844
                  bool v613 = v612 == 7;	// L845
                  if (v613) {	// L846
                    int32_t v614 = dst2;	// L847
                    int v615 = v614;	// L848
                    int32_t v616 = reg2[v615];	// L849
                    int32_t v617 = imm2;	// L850
                    int32_t v618 = v616 >> v617;	// L851
                    reg2[v615] = v618;	// L852
                  } else {
                    int32_t v619 = opcode2;	// L854
                    bool v620 = v619 == 8;	// L855
                    if (v620) {	// L856
                      int32_t v621 = dst2;	// L857
                      int v622 = v621;	// L858
                      int32_t v623 = reg2[v622];	// L859
                      v528.write(v623);	// L860
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void vpu_3(
  int32_t v624[4],
  hls::stream< int32_t >& v625,
  hls::stream< int32_t >& v626,
  hls::stream< int32_t >& v627
) {	// L873
  #pragma HLS array_partition variable=v624 complete dim=1

  int32_t prog3[8];	// L889
  for (int v629 = 0; v629 < 8; v629++) {	// L890
    prog3[v629] = 0;	// L890
  }
  l_S_pc_0_pc3: for (int pc3 = 0; pc3 < 8; pc3++) {	// L891
  #pragma HLS pipeline II=1
    int32_t v631 = v625.read();	// L892
    int32_t word3;	// L893
    word3 = v631;	// L894
    int32_t v633 = word3;	// L895
    prog3[pc3] = v633;	// L896
  }
  l_S_m_1_m19: for (int m19 = 0; m19 < 6; m19++) {	// L898
    int32_t v635 = v626.read();	// L899
    int32_t z3;	// L900
    z3 = v635;	// L901
    int32_t reg3[4];	// L902
    for (int v638 = 0; v638 < 4; v638++) {	// L903
      reg3[v638] = 0;	// L903
    }
    l_S_step_1_step3: for (int step3 = 0; step3 < 8; step3++) {	// L904
    #pragma HLS pipeline II=1
      int32_t v640 = prog3[step3];	// L905
      int32_t word23;	// L906
      word23 = v640;	// L907
      int32_t v642 = word23;	// L908
      int32_t v643 = v642 >> 24;	// L909
      int32_t v644 = v643 & 255;	// L910
      int32_t opcode3;	// L911
      opcode3 = v644;	// L912
      int32_t v646 = word23;	// L913
      int32_t v647 = v646 >> 20;	// L914
      int32_t v648 = v647 & 15;	// L915
      int32_t dst3;	// L916
      dst3 = v648;	// L917
      int32_t v650 = word23;	// L918
      int32_t v651 = v650 >> 16;	// L919
      int32_t v652 = v651 & 15;	// L920
      int32_t src3;	// L921
      src3 = v652;	// L922
      int32_t v654 = word23;	// L923
      int32_t v655 = v654 & 65535;	// L924
      int32_t imm3;	// L925
      imm3 = v655;	// L926
      int32_t v657 = opcode3;	// L927
      bool v658 = v657 == 1;	// L928
      if (v658) {	// L929
        int32_t v659 = z3;	// L930
        int32_t v660 = dst3;	// L931
        int v661 = v660;	// L932
        reg3[v661] = v659;	// L933
      } else {
        int32_t v662 = opcode3;	// L935
        bool v663 = v662 == 2;	// L936
        if (v663) {	// L937
          int32_t v664 = v624[3];	// L938
          int32_t v665 = dst3;	// L939
          int v666 = v665;	// L940
          reg3[v666] = v664;	// L941
        } else {
          int32_t v667 = opcode3;	// L943
          bool v668 = v667 == 3;	// L944
          if (v668) {	// L945
            int32_t v669 = imm3;	// L946
            int32_t v670 = dst3;	// L947
            int v671 = v670;	// L948
            reg3[v671] = v669;	// L949
          } else {
            int32_t v672 = opcode3;	// L951
            bool v673 = v672 == 4;	// L952
            if (v673) {	// L953
              int32_t v674 = dst3;	// L954
              int v675 = v674;	// L955
              int32_t v676 = reg3[v675];	// L956
              int32_t v677 = src3;	// L957
              int v678 = v677;	// L958
              int32_t v679 = reg3[v678];	// L959
              ap_int<33> v680 = v676;	// L960
              ap_int<33> v681 = v679;	// L961
              ap_int<33> v682 = v680 + v681;	// L962
              int32_t v683 = v682;	// L963
              reg3[v675] = v683;	// L964
            } else {
              int32_t v684 = opcode3;	// L966
              bool v685 = v684 == 5;	// L967
              if (v685) {	// L968
                int32_t v686 = dst3;	// L969
                int v687 = v686;	// L970
                int32_t v688 = reg3[v687];	// L971
                int32_t v689 = src3;	// L972
                int v690 = v689;	// L973
                int32_t v691 = reg3[v690];	// L974
                int64_t v692 = v688;	// L975
                int64_t v693 = v691;	// L976
                int64_t v694 = v692 * v693;	// L977
                int32_t v695 = v694;	// L978
                reg3[v687] = v695;	// L979
              } else {
                int32_t v696 = opcode3;	// L981
                bool v697 = v696 == 6;	// L982
                if (v697) {	// L983
                  int32_t v698 = src3;	// L984
                  int v699 = v698;	// L985
                  int32_t v700 = reg3[v699];	// L986
                  int32_t v701 = dst3;	// L987
                  int v702 = v701;	// L988
                  int32_t v703 = reg3[v702];	// L989
                  bool v704 = v700 > v703;	// L990
                  if (v704) {	// L991
                    int32_t v705 = src3;	// L992
                    int v706 = v705;	// L993
                    int32_t v707 = reg3[v706];	// L994
                    int32_t v708 = dst3;	// L995
                    int v709 = v708;	// L996
                    reg3[v709] = v707;	// L997
                  }
                } else {
                  int32_t v710 = opcode3;	// L1000
                  bool v711 = v710 == 7;	// L1001
                  if (v711) {	// L1002
                    int32_t v712 = dst3;	// L1003
                    int v713 = v712;	// L1004
                    int32_t v714 = reg3[v713];	// L1005
                    int32_t v715 = imm3;	// L1006
                    int32_t v716 = v714 >> v715;	// L1007
                    reg3[v713] = v716;	// L1008
                  } else {
                    int32_t v717 = opcode3;	// L1010
                    bool v718 = v717 == 8;	// L1011
                    if (v718) {	// L1012
                      int32_t v719 = dst3;	// L1013
                      int v720 = v719;	// L1014
                      int32_t v721 = reg3[v720];	// L1015
                      v627.write(v721);	// L1016
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void vpu_y_out_drain_0(
  int32_t v722[6][4],
  hls::stream< int32_t >& v723
) {	// L1029
  #pragma HLS array_partition variable=v722 complete dim=1
  #pragma HLS array_partition variable=v722 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 6; _t5++) {	// L1030
  #pragma HLS pipeline II=1
    int32_t v725 = v723.read();	// L1031
    v722[_t5][0] = v725;	// L1032
  }
}

void vpu_y_out_drain_1(
  int32_t v726[6][4],
  hls::stream< int32_t >& v727
) {	// L1036
  #pragma HLS array_partition variable=v726 complete dim=1
  #pragma HLS array_partition variable=v726 complete dim=2

  l_S__t_0__t6: for (int _t6 = 0; _t6 < 6; _t6++) {	// L1037
  #pragma HLS pipeline II=1
    int32_t v729 = v727.read();	// L1038
    v726[_t6][1] = v729;	// L1039
  }
}

void vpu_y_out_drain_2(
  int32_t v730[6][4],
  hls::stream< int32_t >& v731
) {	// L1043
  #pragma HLS array_partition variable=v730 complete dim=1
  #pragma HLS array_partition variable=v730 complete dim=2

  l_S__t_0__t7: for (int _t7 = 0; _t7 < 6; _t7++) {	// L1044
  #pragma HLS pipeline II=1
    int32_t v733 = v731.read();	// L1045
    v730[_t7][2] = v733;	// L1046
  }
}

void vpu_y_out_drain_3(
  int32_t v734[6][4],
  hls::stream< int32_t >& v735
) {	// L1050
  #pragma HLS array_partition variable=v734 complete dim=1
  #pragma HLS array_partition variable=v734 complete dim=2

  l_S__t_0__t8: for (int _t8 = 0; _t8 < 6; _t8++) {	// L1051
  #pragma HLS pipeline II=1
    int32_t v737 = v735.read();	// L1052
    v734[_t8][3] = v737;	// L1053
  }
}

/// This is top function.
void top(
  int8_t v738[6][4],
  int32_t v739[8],
  int8_t v740[4][4],
  int32_t v741[4],
  int32_t v742[6][4]
) {	// L1057
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v738 complete dim=1
  #pragma HLS array_partition variable=v738 complete dim=2

  #pragma HLS array_partition variable=v739 complete dim=1

  #pragma HLS array_partition variable=v740 complete dim=1
  #pragma HLS array_partition variable=v740 complete dim=2

  #pragma HLS array_partition variable=v741 complete dim=1

  #pragma HLS array_partition variable=v742 complete dim=1
  #pragma HLS array_partition variable=v742 complete dim=2

  hls::stream< int8_t > v743;
  #pragma HLS stream variable=v743 depth=2	// L1058
  hls::stream< int8_t > v744;
  #pragma HLS stream variable=v744 depth=2	// L1059
  hls::stream< int8_t > v745;
  #pragma HLS stream variable=v745 depth=2	// L1060
  hls::stream< int8_t > v746;
  #pragma HLS stream variable=v746 depth=2	// L1061
  hls::stream< int8_t > v747;
  #pragma HLS stream variable=v747 depth=2	// L1062
  hls::stream< int8_t > v748;
  #pragma HLS stream variable=v748 depth=2	// L1063
  hls::stream< int8_t > v749;
  #pragma HLS stream variable=v749 depth=2	// L1064
  hls::stream< int8_t > v750;
  #pragma HLS stream variable=v750 depth=2	// L1065
  hls::stream< int8_t > v751;
  #pragma HLS stream variable=v751 depth=2	// L1066
  hls::stream< int8_t > v752;
  #pragma HLS stream variable=v752 depth=2	// L1067
  hls::stream< int8_t > v753;
  #pragma HLS stream variable=v753 depth=2	// L1068
  hls::stream< int8_t > v754;
  #pragma HLS stream variable=v754 depth=2	// L1069
  hls::stream< int8_t > v755;
  #pragma HLS stream variable=v755 depth=2	// L1070
  hls::stream< int8_t > v756;
  #pragma HLS stream variable=v756 depth=2	// L1071
  hls::stream< int8_t > v757;
  #pragma HLS stream variable=v757 depth=2	// L1072
  hls::stream< int8_t > v758;
  #pragma HLS stream variable=v758 depth=2	// L1073
  hls::stream< int32_t > v759;
  #pragma HLS stream variable=v759 depth=2	// L1074
  hls::stream< int32_t > v760;
  #pragma HLS stream variable=v760 depth=2	// L1075
  hls::stream< int32_t > v761;
  #pragma HLS stream variable=v761 depth=2	// L1076
  hls::stream< int32_t > v762;
  #pragma HLS stream variable=v762 depth=2	// L1077
  hls::stream< int32_t > v763;
  #pragma HLS stream variable=v763 depth=2	// L1078
  hls::stream< int32_t > v764;
  #pragma HLS stream variable=v764 depth=2	// L1079
  hls::stream< int32_t > v765;
  #pragma HLS stream variable=v765 depth=2	// L1080
  hls::stream< int32_t > v766;
  #pragma HLS stream variable=v766 depth=2	// L1081
  hls::stream< int32_t > v767;
  #pragma HLS stream variable=v767 depth=2	// L1082
  hls::stream< int32_t > v768;
  #pragma HLS stream variable=v768 depth=2	// L1083
  hls::stream< int32_t > v769;
  #pragma HLS stream variable=v769 depth=2	// L1084
  hls::stream< int32_t > v770;
  #pragma HLS stream variable=v770 depth=2	// L1085
  hls::stream< int32_t > v771;
  #pragma HLS stream variable=v771 depth=2	// L1086
  hls::stream< int32_t > v772;
  #pragma HLS stream variable=v772 depth=2	// L1087
  hls::stream< int32_t > v773;
  #pragma HLS stream variable=v773 depth=2	// L1088
  hls::stream< int32_t > v774;
  #pragma HLS stream variable=v774 depth=2	// L1089
  hls::stream< int32_t > v775;
  #pragma HLS stream variable=v775 depth=2	// L1090
  hls::stream< int32_t > v776;
  #pragma HLS stream variable=v776 depth=2	// L1091
  hls::stream< int32_t > v777;
  #pragma HLS stream variable=v777 depth=2	// L1092
  hls::stream< int32_t > v778;
  #pragma HLS stream variable=v778 depth=2	// L1093
  hls::stream< int8_t > v779;
  #pragma HLS stream variable=v779 depth=2	// L1094
  hls::stream< int8_t > v780;
  #pragma HLS stream variable=v780 depth=2	// L1095
  hls::stream< int8_t > v781;
  #pragma HLS stream variable=v781 depth=2	// L1096
  hls::stream< int8_t > v782;
  #pragma HLS stream variable=v782 depth=2	// L1097
  hls::stream< int32_t > v783;
  #pragma HLS stream variable=v783 depth=2	// L1098
  hls::stream< int32_t > v784;
  #pragma HLS stream variable=v784 depth=2	// L1099
  hls::stream< int32_t > v785;
  #pragma HLS stream variable=v785 depth=2	// L1100
  hls::stream< int32_t > v786;
  #pragma HLS stream variable=v786 depth=2	// L1101
  hls::stream< int32_t > v787;
  #pragma HLS stream variable=v787 depth=2	// L1102
  hls::stream< int32_t > v788;
  #pragma HLS stream variable=v788 depth=2	// L1103
  hls::stream< int32_t > v789;
  #pragma HLS stream variable=v789 depth=2	// L1104
  hls::stream< int32_t > v790;
  #pragma HLS stream variable=v790 depth=2	// L1105
  hls::stream< int32_t > v791;
  #pragma HLS stream variable=v791 depth=2	// L1106
  mac_a_in_load_0(v738, v779);	// L1107
  mac_a_in_load_1(v738, v780);	// L1108
  mac_a_in_load_2(v738, v781);	// L1109
  mac_a_in_load_3(v738, v782);	// L1110
  vpu_op_in_load_0(v739, v787);	// L1111
  mac_0_0(v740, v779, v763, v744);	// L1112
  mac_0_1(v740, v744, v764, v745);	// L1113
  mac_0_2(v740, v745, v765, v746);	// L1114
  mac_0_3(v740, v746, v766);	// L1115
  mac_1_0(v740, v780, v763, v767, v748);	// L1116
  mac_1_1(v740, v748, v764, v768, v749);	// L1117
  mac_1_2(v740, v749, v765, v769, v750);	// L1118
  mac_1_3(v740, v750, v766, v770);	// L1119
  mac_2_0(v740, v781, v767, v771, v752);	// L1120
  mac_2_1(v740, v752, v768, v772, v753);	// L1121
  mac_2_2(v740, v753, v769, v773, v754);	// L1122
  mac_2_3(v740, v754, v770, v774);	// L1123
  mac_3_0(v740, v782, v771, v783, v756);	// L1124
  mac_3_1(v740, v756, v772, v784, v757);	// L1125
  mac_3_2(v740, v757, v773, v785, v758);	// L1126
  mac_3_3(v740, v758, v774, v786);	// L1127
  vpu_0(v741, v787, v776, v783, v788);	// L1128
  vpu_1(v741, v776, v777, v784, v789);	// L1129
  vpu_2(v741, v777, v778, v785, v790);	// L1130
  vpu_3(v741, v778, v786, v791);	// L1131
  vpu_y_out_drain_0(v742, v788);	// L1132
  vpu_y_out_drain_1(v742, v789);	// L1133
  vpu_y_out_drain_2(v742, v790);	// L1134
  vpu_y_out_drain_3(v742, v791);	// L1135
}


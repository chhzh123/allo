
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
void mac_a_in_load_0_0(
  int8_t v0[6][8],
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

void mac_a_in_load_0_1(
  int8_t v4[6][8],
  hls::stream< int8_t >& v5
) {	// L10
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 6; _t1++) {	// L11
  #pragma HLS pipeline II=1
    int8_t v7 = v4[_t1][4];	// L12
    v5.write(v7);	// L13
  }
}

void mac_a_in_load_1_0(
  int8_t v8[6][8],
  hls::stream< int8_t >& v9
) {	// L17
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 6; _t2++) {	// L18
  #pragma HLS pipeline II=1
    int8_t v11 = v8[_t2][1];	// L19
    v9.write(v11);	// L20
  }
}

void mac_a_in_load_1_1(
  int8_t v12[6][8],
  hls::stream< int8_t >& v13
) {	// L24
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 6; _t3++) {	// L25
  #pragma HLS pipeline II=1
    int8_t v15 = v12[_t3][5];	// L26
    v13.write(v15);	// L27
  }
}

void mac_a_in_load_2_0(
  int8_t v16[6][8],
  hls::stream< int8_t >& v17
) {	// L31
  #pragma HLS array_partition variable=v16 complete dim=1
  #pragma HLS array_partition variable=v16 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 6; _t4++) {	// L32
  #pragma HLS pipeline II=1
    int8_t v19 = v16[_t4][2];	// L33
    v17.write(v19);	// L34
  }
}

void mac_a_in_load_2_1(
  int8_t v20[6][8],
  hls::stream< int8_t >& v21
) {	// L38
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 6; _t5++) {	// L39
  #pragma HLS pipeline II=1
    int8_t v23 = v20[_t5][6];	// L40
    v21.write(v23);	// L41
  }
}

void mac_a_in_load_3_0(
  int8_t v24[6][8],
  hls::stream< int8_t >& v25
) {	// L45
  #pragma HLS array_partition variable=v24 complete dim=1
  #pragma HLS array_partition variable=v24 complete dim=2

  l_S__t_0__t6: for (int _t6 = 0; _t6 < 6; _t6++) {	// L46
  #pragma HLS pipeline II=1
    int8_t v27 = v24[_t6][3];	// L47
    v25.write(v27);	// L48
  }
}

void mac_a_in_load_3_1(
  int8_t v28[6][8],
  hls::stream< int8_t >& v29
) {	// L52
  #pragma HLS array_partition variable=v28 complete dim=1
  #pragma HLS array_partition variable=v28 complete dim=2

  l_S__t_0__t7: for (int _t7 = 0; _t7 < 6; _t7++) {	// L53
  #pragma HLS pipeline II=1
    int8_t v31 = v28[_t7][7];	// L54
    v29.write(v31);	// L55
  }
}

void mac_0_0(
  int8_t v32[8][2],
  hls::stream< int8_t >& v33,
  hls::stream< int32_t >& v34,
  hls::stream< int8_t >& v35
) {	// L59
  #pragma HLS array_partition variable=v32 complete dim=1
  #pragma HLS array_partition variable=v32 complete dim=2

  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L61
  #pragma HLS pipeline II=1
    int8_t v37 = v33.read();	// L62
    int8_t a;	// L63
    a = v37;	// L64
    int32_t p;	// L65
    p = 0;	// L66
    int32_t v40 = p;	// L67
    int8_t v41 = a;	// L68
    int8_t v42 = v32[0][0];	// L69
    int16_t v43 = v41;	// L70
    int16_t v44 = v42;	// L71
    int16_t v45 = v43 * v44;	// L72
    ap_int<33> v46 = v40;	// L73
    ap_int<33> v47 = v45;	// L74
    ap_int<33> v48 = v46 + v47;	// L75
    v34.write(v48);	// L76
    int8_t v49 = a;	// L77
    v35.write(v49);	// L78
  }
}

void mac_0_1(
  int8_t v50[8][2],
  hls::stream< int8_t >& v51,
  hls::stream< int32_t >& v52
) {	// L82
  #pragma HLS array_partition variable=v50 complete dim=1
  #pragma HLS array_partition variable=v50 complete dim=2

  l_S_m_0_m1: for (int m1 = 0; m1 < 6; m1++) {	// L84
  #pragma HLS pipeline II=1
    int8_t v54 = v51.read();	// L85
    int8_t a1;	// L86
    a1 = v54;	// L87
    int32_t p1;	// L88
    p1 = 0;	// L89
    int32_t v57 = p1;	// L90
    int8_t v58 = a1;	// L91
    int8_t v59 = v50[0][1];	// L92
    int16_t v60 = v58;	// L93
    int16_t v61 = v59;	// L94
    int16_t v62 = v60 * v61;	// L95
    ap_int<33> v63 = v57;	// L96
    ap_int<33> v64 = v62;	// L97
    ap_int<33> v65 = v63 + v64;	// L98
    v52.write(v65);	// L99
  }
}

void mac_0_2(
  int8_t v66[8][2],
  hls::stream< int8_t >& v67,
  hls::stream< int32_t >& v68,
  hls::stream< int32_t >& v69,
  hls::stream< int8_t >& v70
) {	// L103
  #pragma HLS array_partition variable=v66 complete dim=1
  #pragma HLS array_partition variable=v66 complete dim=2

  l_S_m_0_m2: for (int m2 = 0; m2 < 6; m2++) {	// L104
  #pragma HLS pipeline II=1
    int8_t v72 = v67.read();	// L105
    int8_t a2;	// L106
    a2 = v72;	// L107
    int32_t v74 = v68.read();	// L108
    int32_t p2;	// L109
    p2 = v74;	// L110
    int32_t v76 = p2;	// L111
    int8_t v77 = a2;	// L112
    int8_t v78 = v66[4][0];	// L113
    int16_t v79 = v77;	// L114
    int16_t v80 = v78;	// L115
    int16_t v81 = v79 * v80;	// L116
    ap_int<33> v82 = v76;	// L117
    ap_int<33> v83 = v81;	// L118
    ap_int<33> v84 = v82 + v83;	// L119
    v69.write(v84);	// L120
    int8_t v85 = a2;	// L121
    v70.write(v85);	// L122
  }
}

void mac_0_3(
  int8_t v86[8][2],
  hls::stream< int8_t >& v87,
  hls::stream< int32_t >& v88,
  hls::stream< int32_t >& v89
) {	// L126
  #pragma HLS array_partition variable=v86 complete dim=1
  #pragma HLS array_partition variable=v86 complete dim=2

  l_S_m_0_m3: for (int m3 = 0; m3 < 6; m3++) {	// L127
  #pragma HLS pipeline II=1
    int8_t v91 = v87.read();	// L128
    int8_t a3;	// L129
    a3 = v91;	// L130
    int32_t v93 = v88.read();	// L131
    int32_t p3;	// L132
    p3 = v93;	// L133
    int32_t v95 = p3;	// L134
    int8_t v96 = a3;	// L135
    int8_t v97 = v86[4][1];	// L136
    int16_t v98 = v96;	// L137
    int16_t v99 = v97;	// L138
    int16_t v100 = v98 * v99;	// L139
    ap_int<33> v101 = v95;	// L140
    ap_int<33> v102 = v100;	// L141
    ap_int<33> v103 = v101 + v102;	// L142
    v89.write(v103);	// L143
  }
}

void mac_1_0(
  int8_t v104[8][2],
  hls::stream< int8_t >& v105,
  hls::stream< int32_t >& v106,
  hls::stream< int32_t >& v107,
  hls::stream< int8_t >& v108
) {	// L147
  #pragma HLS array_partition variable=v104 complete dim=1
  #pragma HLS array_partition variable=v104 complete dim=2

  l_S_m_0_m4: for (int m4 = 0; m4 < 6; m4++) {	// L148
  #pragma HLS pipeline II=1
    int8_t v110 = v105.read();	// L149
    int8_t a4;	// L150
    a4 = v110;	// L151
    int32_t v112 = v106.read();	// L152
    int32_t p4;	// L153
    p4 = v112;	// L154
    int32_t v114 = p4;	// L155
    int8_t v115 = a4;	// L156
    int8_t v116 = v104[1][0];	// L157
    int16_t v117 = v115;	// L158
    int16_t v118 = v116;	// L159
    int16_t v119 = v117 * v118;	// L160
    ap_int<33> v120 = v114;	// L161
    ap_int<33> v121 = v119;	// L162
    ap_int<33> v122 = v120 + v121;	// L163
    v107.write(v122);	// L164
    int8_t v123 = a4;	// L165
    v108.write(v123);	// L166
  }
}

void mac_1_1(
  int8_t v124[8][2],
  hls::stream< int8_t >& v125,
  hls::stream< int32_t >& v126,
  hls::stream< int32_t >& v127
) {	// L170
  #pragma HLS array_partition variable=v124 complete dim=1
  #pragma HLS array_partition variable=v124 complete dim=2

  l_S_m_0_m5: for (int m5 = 0; m5 < 6; m5++) {	// L171
  #pragma HLS pipeline II=1
    int8_t v129 = v125.read();	// L172
    int8_t a5;	// L173
    a5 = v129;	// L174
    int32_t v131 = v126.read();	// L175
    int32_t p5;	// L176
    p5 = v131;	// L177
    int32_t v133 = p5;	// L178
    int8_t v134 = a5;	// L179
    int8_t v135 = v124[1][1];	// L180
    int16_t v136 = v134;	// L181
    int16_t v137 = v135;	// L182
    int16_t v138 = v136 * v137;	// L183
    ap_int<33> v139 = v133;	// L184
    ap_int<33> v140 = v138;	// L185
    ap_int<33> v141 = v139 + v140;	// L186
    v127.write(v141);	// L187
  }
}

void mac_1_2(
  int8_t v142[8][2],
  hls::stream< int8_t >& v143,
  hls::stream< int32_t >& v144,
  hls::stream< int32_t >& v145,
  hls::stream< int8_t >& v146
) {	// L191
  #pragma HLS array_partition variable=v142 complete dim=1
  #pragma HLS array_partition variable=v142 complete dim=2

  l_S_m_0_m6: for (int m6 = 0; m6 < 6; m6++) {	// L192
  #pragma HLS pipeline II=1
    int8_t v148 = v143.read();	// L193
    int8_t a6;	// L194
    a6 = v148;	// L195
    int32_t v150 = v144.read();	// L196
    int32_t p6;	// L197
    p6 = v150;	// L198
    int32_t v152 = p6;	// L199
    int8_t v153 = a6;	// L200
    int8_t v154 = v142[5][0];	// L201
    int16_t v155 = v153;	// L202
    int16_t v156 = v154;	// L203
    int16_t v157 = v155 * v156;	// L204
    ap_int<33> v158 = v152;	// L205
    ap_int<33> v159 = v157;	// L206
    ap_int<33> v160 = v158 + v159;	// L207
    v145.write(v160);	// L208
    int8_t v161 = a6;	// L209
    v146.write(v161);	// L210
  }
}

void mac_1_3(
  int8_t v162[8][2],
  hls::stream< int8_t >& v163,
  hls::stream< int32_t >& v164,
  hls::stream< int32_t >& v165
) {	// L214
  #pragma HLS array_partition variable=v162 complete dim=1
  #pragma HLS array_partition variable=v162 complete dim=2

  l_S_m_0_m7: for (int m7 = 0; m7 < 6; m7++) {	// L215
  #pragma HLS pipeline II=1
    int8_t v167 = v163.read();	// L216
    int8_t a7;	// L217
    a7 = v167;	// L218
    int32_t v169 = v164.read();	// L219
    int32_t p7;	// L220
    p7 = v169;	// L221
    int32_t v171 = p7;	// L222
    int8_t v172 = a7;	// L223
    int8_t v173 = v162[5][1];	// L224
    int16_t v174 = v172;	// L225
    int16_t v175 = v173;	// L226
    int16_t v176 = v174 * v175;	// L227
    ap_int<33> v177 = v171;	// L228
    ap_int<33> v178 = v176;	// L229
    ap_int<33> v179 = v177 + v178;	// L230
    v165.write(v179);	// L231
  }
}

void mac_2_0(
  int8_t v180[8][2],
  hls::stream< int8_t >& v181,
  hls::stream< int32_t >& v182,
  hls::stream< int32_t >& v183,
  hls::stream< int8_t >& v184
) {	// L235
  #pragma HLS array_partition variable=v180 complete dim=1
  #pragma HLS array_partition variable=v180 complete dim=2

  l_S_m_0_m8: for (int m8 = 0; m8 < 6; m8++) {	// L236
  #pragma HLS pipeline II=1
    int8_t v186 = v181.read();	// L237
    int8_t a8;	// L238
    a8 = v186;	// L239
    int32_t v188 = v182.read();	// L240
    int32_t p8;	// L241
    p8 = v188;	// L242
    int32_t v190 = p8;	// L243
    int8_t v191 = a8;	// L244
    int8_t v192 = v180[2][0];	// L245
    int16_t v193 = v191;	// L246
    int16_t v194 = v192;	// L247
    int16_t v195 = v193 * v194;	// L248
    ap_int<33> v196 = v190;	// L249
    ap_int<33> v197 = v195;	// L250
    ap_int<33> v198 = v196 + v197;	// L251
    v183.write(v198);	// L252
    int8_t v199 = a8;	// L253
    v184.write(v199);	// L254
  }
}

void mac_2_1(
  int8_t v200[8][2],
  hls::stream< int8_t >& v201,
  hls::stream< int32_t >& v202,
  hls::stream< int32_t >& v203
) {	// L258
  #pragma HLS array_partition variable=v200 complete dim=1
  #pragma HLS array_partition variable=v200 complete dim=2

  l_S_m_0_m9: for (int m9 = 0; m9 < 6; m9++) {	// L259
  #pragma HLS pipeline II=1
    int8_t v205 = v201.read();	// L260
    int8_t a9;	// L261
    a9 = v205;	// L262
    int32_t v207 = v202.read();	// L263
    int32_t p9;	// L264
    p9 = v207;	// L265
    int32_t v209 = p9;	// L266
    int8_t v210 = a9;	// L267
    int8_t v211 = v200[2][1];	// L268
    int16_t v212 = v210;	// L269
    int16_t v213 = v211;	// L270
    int16_t v214 = v212 * v213;	// L271
    ap_int<33> v215 = v209;	// L272
    ap_int<33> v216 = v214;	// L273
    ap_int<33> v217 = v215 + v216;	// L274
    v203.write(v217);	// L275
  }
}

void mac_2_2(
  int8_t v218[8][2],
  hls::stream< int8_t >& v219,
  hls::stream< int32_t >& v220,
  hls::stream< int32_t >& v221,
  hls::stream< int8_t >& v222
) {	// L279
  #pragma HLS array_partition variable=v218 complete dim=1
  #pragma HLS array_partition variable=v218 complete dim=2

  l_S_m_0_m10: for (int m10 = 0; m10 < 6; m10++) {	// L280
  #pragma HLS pipeline II=1
    int8_t v224 = v219.read();	// L281
    int8_t a10;	// L282
    a10 = v224;	// L283
    int32_t v226 = v220.read();	// L284
    int32_t p10;	// L285
    p10 = v226;	// L286
    int32_t v228 = p10;	// L287
    int8_t v229 = a10;	// L288
    int8_t v230 = v218[6][0];	// L289
    int16_t v231 = v229;	// L290
    int16_t v232 = v230;	// L291
    int16_t v233 = v231 * v232;	// L292
    ap_int<33> v234 = v228;	// L293
    ap_int<33> v235 = v233;	// L294
    ap_int<33> v236 = v234 + v235;	// L295
    v221.write(v236);	// L296
    int8_t v237 = a10;	// L297
    v222.write(v237);	// L298
  }
}

void mac_2_3(
  int8_t v238[8][2],
  hls::stream< int8_t >& v239,
  hls::stream< int32_t >& v240,
  hls::stream< int32_t >& v241
) {	// L302
  #pragma HLS array_partition variable=v238 complete dim=1
  #pragma HLS array_partition variable=v238 complete dim=2

  l_S_m_0_m11: for (int m11 = 0; m11 < 6; m11++) {	// L303
  #pragma HLS pipeline II=1
    int8_t v243 = v239.read();	// L304
    int8_t a11;	// L305
    a11 = v243;	// L306
    int32_t v245 = v240.read();	// L307
    int32_t p11;	// L308
    p11 = v245;	// L309
    int32_t v247 = p11;	// L310
    int8_t v248 = a11;	// L311
    int8_t v249 = v238[6][1];	// L312
    int16_t v250 = v248;	// L313
    int16_t v251 = v249;	// L314
    int16_t v252 = v250 * v251;	// L315
    ap_int<33> v253 = v247;	// L316
    ap_int<33> v254 = v252;	// L317
    ap_int<33> v255 = v253 + v254;	// L318
    v241.write(v255);	// L319
  }
}

void mac_3_0(
  int8_t v256[8][2],
  hls::stream< int8_t >& v257,
  hls::stream< int32_t >& v258,
  hls::stream< int32_t >& v259,
  hls::stream< int8_t >& v260
) {	// L323
  #pragma HLS array_partition variable=v256 complete dim=1
  #pragma HLS array_partition variable=v256 complete dim=2

  l_S_m_0_m12: for (int m12 = 0; m12 < 6; m12++) {	// L324
  #pragma HLS pipeline II=1
    int8_t v262 = v257.read();	// L325
    int8_t a12;	// L326
    a12 = v262;	// L327
    int32_t v264 = v258.read();	// L328
    int32_t p12;	// L329
    p12 = v264;	// L330
    int32_t v266 = p12;	// L331
    int8_t v267 = a12;	// L332
    int8_t v268 = v256[3][0];	// L333
    int16_t v269 = v267;	// L334
    int16_t v270 = v268;	// L335
    int16_t v271 = v269 * v270;	// L336
    ap_int<33> v272 = v266;	// L337
    ap_int<33> v273 = v271;	// L338
    ap_int<33> v274 = v272 + v273;	// L339
    v259.write(v274);	// L340
    int8_t v275 = a12;	// L341
    v260.write(v275);	// L342
  }
}

void mac_3_1(
  int8_t v276[8][2],
  hls::stream< int8_t >& v277,
  hls::stream< int32_t >& v278,
  hls::stream< int32_t >& v279
) {	// L346
  #pragma HLS array_partition variable=v276 complete dim=1
  #pragma HLS array_partition variable=v276 complete dim=2

  l_S_m_0_m13: for (int m13 = 0; m13 < 6; m13++) {	// L347
  #pragma HLS pipeline II=1
    int8_t v281 = v277.read();	// L348
    int8_t a13;	// L349
    a13 = v281;	// L350
    int32_t v283 = v278.read();	// L351
    int32_t p13;	// L352
    p13 = v283;	// L353
    int32_t v285 = p13;	// L354
    int8_t v286 = a13;	// L355
    int8_t v287 = v276[3][1];	// L356
    int16_t v288 = v286;	// L357
    int16_t v289 = v287;	// L358
    int16_t v290 = v288 * v289;	// L359
    ap_int<33> v291 = v285;	// L360
    ap_int<33> v292 = v290;	// L361
    ap_int<33> v293 = v291 + v292;	// L362
    v279.write(v293);	// L363
  }
}

void mac_3_2(
  int8_t v294[8][2],
  hls::stream< int8_t >& v295,
  hls::stream< int32_t >& v296,
  hls::stream< int32_t >& v297,
  hls::stream< int8_t >& v298
) {	// L367
  #pragma HLS array_partition variable=v294 complete dim=1
  #pragma HLS array_partition variable=v294 complete dim=2

  l_S_m_0_m14: for (int m14 = 0; m14 < 6; m14++) {	// L368
  #pragma HLS pipeline II=1
    int8_t v300 = v295.read();	// L369
    int8_t a14;	// L370
    a14 = v300;	// L371
    int32_t v302 = v296.read();	// L372
    int32_t p14;	// L373
    p14 = v302;	// L374
    int32_t v304 = p14;	// L375
    int8_t v305 = a14;	// L376
    int8_t v306 = v294[7][0];	// L377
    int16_t v307 = v305;	// L378
    int16_t v308 = v306;	// L379
    int16_t v309 = v307 * v308;	// L380
    ap_int<33> v310 = v304;	// L381
    ap_int<33> v311 = v309;	// L382
    ap_int<33> v312 = v310 + v311;	// L383
    v297.write(v312);	// L384
    int8_t v313 = a14;	// L385
    v298.write(v313);	// L386
  }
}

void mac_3_3(
  int8_t v314[8][2],
  hls::stream< int8_t >& v315,
  hls::stream< int32_t >& v316,
  hls::stream< int32_t >& v317
) {	// L390
  #pragma HLS array_partition variable=v314 complete dim=1
  #pragma HLS array_partition variable=v314 complete dim=2

  l_S_m_0_m15: for (int m15 = 0; m15 < 6; m15++) {	// L391
  #pragma HLS pipeline II=1
    int8_t v319 = v315.read();	// L392
    int8_t a15;	// L393
    a15 = v319;	// L394
    int32_t v321 = v316.read();	// L395
    int32_t p15;	// L396
    p15 = v321;	// L397
    int32_t v323 = p15;	// L398
    int8_t v324 = a15;	// L399
    int8_t v325 = v314[7][1];	// L400
    int16_t v326 = v324;	// L401
    int16_t v327 = v325;	// L402
    int16_t v328 = v326 * v327;	// L403
    ap_int<33> v329 = v323;	// L404
    ap_int<33> v330 = v328;	// L405
    ap_int<33> v331 = v329 + v330;	// L406
    v317.write(v331);	// L407
  }
}

void act_0(
  hls::stream< int32_t >& v332,
  hls::stream< int8_t >& v333
) {	// L411
  l_S_m_0_m16: for (int m16 = 0; m16 < 6; m16++) {	// L414
  #pragma HLS pipeline II=1
    int32_t v335 = v332.read();	// L415
    int32_t z;	// L416
    z = v335;	// L417
    int32_t v337 = z;	// L418
    bool v338 = v337 < 0;	// L419
    if (v338) {	// L420
      z = 0;	// L421
    }
    int32_t v339 = z;	// L423
    int32_t v340 = v339 >> 2;	// L424
    int8_t v341 = v340;	// L425
    int8_t y;	// L426
    y = v341;	// L427
    int8_t v343 = y;	// L428
    v333.write(v343);	// L429
  }
}

void act_1(
  hls::stream< int32_t >& v344,
  hls::stream< int8_t >& v345
) {	// L433
  l_S_m_0_m17: for (int m17 = 0; m17 < 6; m17++) {	// L436
  #pragma HLS pipeline II=1
    int32_t v347 = v344.read();	// L437
    int32_t z1;	// L438
    z1 = v347;	// L439
    int32_t v349 = z1;	// L440
    bool v350 = v349 < 0;	// L441
    if (v350) {	// L442
      z1 = 0;	// L443
    }
    int32_t v351 = z1;	// L445
    int32_t v352 = v351 >> 2;	// L446
    int8_t v353 = v352;	// L447
    int8_t y1;	// L448
    y1 = v353;	// L449
    int8_t v355 = y1;	// L450
    v345.write(v355);	// L451
  }
}

void act_y_out_drain_0(
  int8_t v356[6][2],
  hls::stream< int8_t >& v357
) {	// L455
  #pragma HLS array_partition variable=v356 complete dim=1
  #pragma HLS array_partition variable=v356 complete dim=2

  l_S__t_0__t8: for (int _t8 = 0; _t8 < 6; _t8++) {	// L456
  #pragma HLS pipeline II=1
    int8_t v359 = v357.read();	// L457
    v356[_t8][0] = v359;	// L458
  }
}

void act_y_out_drain_1(
  int8_t v360[6][2],
  hls::stream< int8_t >& v361
) {	// L462
  #pragma HLS array_partition variable=v360 complete dim=1
  #pragma HLS array_partition variable=v360 complete dim=2

  l_S__t_0__t9: for (int _t9 = 0; _t9 < 6; _t9++) {	// L463
  #pragma HLS pipeline II=1
    int8_t v363 = v361.read();	// L464
    v360[_t9][1] = v363;	// L465
  }
}

/// This is top function.
void top(
  int8_t v364[6][8],
  int8_t v365[8][2],
  int8_t v366[6][2]
) {	// L469
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v364 complete dim=1
  #pragma HLS array_partition variable=v364 complete dim=2

  #pragma HLS array_partition variable=v365 complete dim=1
  #pragma HLS array_partition variable=v365 complete dim=2

  #pragma HLS array_partition variable=v366 complete dim=1
  #pragma HLS array_partition variable=v366 complete dim=2

  hls::stream< int8_t > v367;
  #pragma HLS stream variable=v367 depth=2	// L470
  hls::stream< int8_t > v368;
  #pragma HLS stream variable=v368 depth=2	// L471
  hls::stream< int8_t > v369;
  #pragma HLS stream variable=v369 depth=2	// L472
  hls::stream< int8_t > v370;
  #pragma HLS stream variable=v370 depth=2	// L473
  hls::stream< int8_t > v371;
  #pragma HLS stream variable=v371 depth=2	// L474
  hls::stream< int8_t > v372;
  #pragma HLS stream variable=v372 depth=2	// L475
  hls::stream< int8_t > v373;
  #pragma HLS stream variable=v373 depth=2	// L476
  hls::stream< int8_t > v374;
  #pragma HLS stream variable=v374 depth=2	// L477
  hls::stream< int8_t > v375;
  #pragma HLS stream variable=v375 depth=2	// L478
  hls::stream< int8_t > v376;
  #pragma HLS stream variable=v376 depth=2	// L479
  hls::stream< int8_t > v377;
  #pragma HLS stream variable=v377 depth=2	// L480
  hls::stream< int8_t > v378;
  #pragma HLS stream variable=v378 depth=2	// L481
  hls::stream< int8_t > v379;
  #pragma HLS stream variable=v379 depth=2	// L482
  hls::stream< int8_t > v380;
  #pragma HLS stream variable=v380 depth=2	// L483
  hls::stream< int8_t > v381;
  #pragma HLS stream variable=v381 depth=2	// L484
  hls::stream< int8_t > v382;
  #pragma HLS stream variable=v382 depth=2	// L485
  hls::stream< int32_t > v383;
  #pragma HLS stream variable=v383 depth=2	// L486
  hls::stream< int32_t > v384;
  #pragma HLS stream variable=v384 depth=2	// L487
  hls::stream< int32_t > v385;
  #pragma HLS stream variable=v385 depth=2	// L488
  hls::stream< int32_t > v386;
  #pragma HLS stream variable=v386 depth=2	// L489
  hls::stream< int32_t > v387;
  #pragma HLS stream variable=v387 depth=2	// L490
  hls::stream< int32_t > v388;
  #pragma HLS stream variable=v388 depth=2	// L491
  hls::stream< int32_t > v389;
  #pragma HLS stream variable=v389 depth=2	// L492
  hls::stream< int32_t > v390;
  #pragma HLS stream variable=v390 depth=2	// L493
  hls::stream< int32_t > v391;
  #pragma HLS stream variable=v391 depth=2	// L494
  hls::stream< int32_t > v392;
  #pragma HLS stream variable=v392 depth=2	// L495
  hls::stream< int32_t > v393;
  #pragma HLS stream variable=v393 depth=2	// L496
  hls::stream< int32_t > v394;
  #pragma HLS stream variable=v394 depth=2	// L497
  hls::stream< int32_t > v395;
  #pragma HLS stream variable=v395 depth=2	// L498
  hls::stream< int32_t > v396;
  #pragma HLS stream variable=v396 depth=2	// L499
  hls::stream< int8_t > v397;
  #pragma HLS stream variable=v397 depth=2	// L500
  hls::stream< int8_t > v398;
  #pragma HLS stream variable=v398 depth=2	// L501
  hls::stream< int8_t > v399;
  #pragma HLS stream variable=v399 depth=2	// L502
  hls::stream< int8_t > v400;
  #pragma HLS stream variable=v400 depth=2	// L503
  hls::stream< int8_t > v401;
  #pragma HLS stream variable=v401 depth=2	// L504
  hls::stream< int8_t > v402;
  #pragma HLS stream variable=v402 depth=2	// L505
  hls::stream< int8_t > v403;
  #pragma HLS stream variable=v403 depth=2	// L506
  hls::stream< int8_t > v404;
  #pragma HLS stream variable=v404 depth=2	// L507
  hls::stream< int32_t > v405;
  #pragma HLS stream variable=v405 depth=2	// L508
  hls::stream< int32_t > v406;
  #pragma HLS stream variable=v406 depth=2	// L509
  hls::stream< int8_t > v407;
  #pragma HLS stream variable=v407 depth=2	// L510
  hls::stream< int8_t > v408;
  #pragma HLS stream variable=v408 depth=2	// L511
  mac_a_in_load_0_0(v364, v397);	// L512
  mac_a_in_load_0_1(v364, v398);	// L513
  mac_a_in_load_1_0(v364, v399);	// L514
  mac_a_in_load_1_1(v364, v400);	// L515
  mac_a_in_load_2_0(v364, v401);	// L516
  mac_a_in_load_2_1(v364, v402);	// L517
  mac_a_in_load_3_0(v364, v403);	// L518
  mac_a_in_load_3_1(v364, v404);	// L519
  mac_0_0(v365, v397, v383, v368);	// L520
  mac_0_1(v365, v368, v384);	// L521
  mac_0_2(v365, v398, v395, v385, v370);	// L522
  mac_0_3(v365, v370, v396, v386);	// L523
  mac_1_0(v365, v399, v383, v387, v372);	// L524
  mac_1_1(v365, v372, v384, v388);	// L525
  mac_1_2(v365, v400, v385, v389, v374);	// L526
  mac_1_3(v365, v374, v386, v390);	// L527
  mac_2_0(v365, v401, v387, v391, v376);	// L528
  mac_2_1(v365, v376, v388, v392);	// L529
  mac_2_2(v365, v402, v389, v393, v378);	// L530
  mac_2_3(v365, v378, v390, v394);	// L531
  mac_3_0(v365, v403, v391, v395, v380);	// L532
  mac_3_1(v365, v380, v392, v396);	// L533
  mac_3_2(v365, v404, v393, v405, v382);	// L534
  mac_3_3(v365, v382, v394, v406);	// L535
  act_0(v405, v407);	// L536
  act_1(v406, v408);	// L537
  act_y_out_drain_0(v366, v407);	// L538
  act_y_out_drain_1(v366, v408);	// L539
}


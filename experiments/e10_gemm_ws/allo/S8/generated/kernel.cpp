
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

extern "C" {

void PE_kernel_ws_gemm_0_0(
  hls::stream< int8_t > &v0 /* v0[8] */,
  hls::stream< int32_t > &v1 /* v1[8] */,
  hls::stream< int8_t > &v2 /* v2[8] */,
  hls::stream< int32_t > &v3 /* v3[8] */,
  int8_t v4[8][8],
  int v5,
  int v6
) {	// L4
  #pragma HLS stream variable=v0 depth=9
  #pragma HLS stream variable=v1 depth=9
  #pragma HLS stream variable=v2 depth=9
  #pragma HLS stream variable=v3 depth=9
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  int8_t v7 = v4[v5][v6];	// L5
  int8_t w;	// L6
  w = v7;	// L7
  l_step_m: for (int m = 0; m < 8; m++) {	// L8
  #pragma HLS pipeline II=1
    int8_t v10 = v0.read(); // v0[m];	// L9
    int8_t a;	// L10
    a = v10;	// L11
    int32_t v12 = v1.read(); // v1[m];	// L12
    int32_t p;	// L13
    p = v12;	// L14
    int32_t v14 = p;	// L15
    int8_t v15 = a;	// L16
    int8_t v16 = w;	// L17
    int16_t v17 = v15;	// L18
    int16_t v18 = v16;	// L19
    int16_t v19 = v17 * v18;	// L20
    ap_int<33> v20 = v14;	// L21
    ap_int<33> v21 = v19;	// L22
    ap_int<33> v22 = v20 + v21;	// L23
    int32_t v23 = v22;	// L24
    v3.write(v23); // v3[m] = v23;	// L25
    int8_t v24 = a;	// L26
    v2.write(v24); // v2[m] = v24;	// L27
  }
}

void PE_kernel_ws_gemm_1_0(
  hls::stream< int8_t > &v25 /* v25[8] */,
  hls::stream< int32_t > &v26 /* v26[8] */,
  hls::stream< int8_t > &v27 /* v27[8] */,
  hls::stream< int32_t > &v28 /* v28[8] */,
  int8_t v29[8][8],
  int v30,
  int v31
) {	// L31
  #pragma HLS stream variable=v25 depth=9
  #pragma HLS stream variable=v26 depth=9
  #pragma HLS stream variable=v27 depth=9
  #pragma HLS stream variable=v28 depth=9
  #pragma HLS array_partition variable=v29 complete dim=1
  #pragma HLS array_partition variable=v29 complete dim=2

  int8_t v32 = v29[v30][v31];	// L32
  int8_t w1;	// L33
  w1 = v32;	// L34
  l_step_m1: for (int m1 = 0; m1 < 8; m1++) {	// L35
  #pragma HLS pipeline II=1
    int8_t v35 = v25.read(); // v25[m1];	// L36
    int8_t a1;	// L37
    a1 = v35;	// L38
    int32_t v37 = v26.read(); // v26[m1];	// L39
    int32_t p1;	// L40
    p1 = v37;	// L41
    int32_t v39 = p1;	// L42
    int8_t v40 = a1;	// L43
    int8_t v41 = w1;	// L44
    int16_t v42 = v40;	// L45
    int16_t v43 = v41;	// L46
    int16_t v44 = v42 * v43;	// L47
    ap_int<33> v45 = v39;	// L48
    ap_int<33> v46 = v44;	// L49
    ap_int<33> v47 = v45 + v46;	// L50
    int32_t v48 = v47;	// L51
    v28.write(v48); // v28[m1] = v48;	// L52
    int8_t v49 = a1;	// L53
    v27.write(v49); // v27[m1] = v49;	// L54
  }
}

void PE_kernel_ws_gemm_2_0(
  hls::stream< int8_t > &v50 /* v50[8] */,
  hls::stream< int32_t > &v51 /* v51[8] */,
  hls::stream< int8_t > &v52 /* v52[8] */,
  hls::stream< int32_t > &v53 /* v53[8] */,
  int8_t v54[8][8],
  int v55,
  int v56
) {	// L58
  #pragma HLS stream variable=v50 depth=9
  #pragma HLS stream variable=v51 depth=9
  #pragma HLS stream variable=v52 depth=9
  #pragma HLS stream variable=v53 depth=9
  #pragma HLS array_partition variable=v54 complete dim=1
  #pragma HLS array_partition variable=v54 complete dim=2

  int8_t v57 = v54[v55][v56];	// L59
  int8_t w2;	// L60
  w2 = v57;	// L61
  l_step_m2: for (int m2 = 0; m2 < 8; m2++) {	// L62
  #pragma HLS pipeline II=1
    int8_t v60 = v50.read(); // v50[m2];	// L63
    int8_t a2;	// L64
    a2 = v60;	// L65
    int32_t v62 = v51.read(); // v51[m2];	// L66
    int32_t p2;	// L67
    p2 = v62;	// L68
    int32_t v64 = p2;	// L69
    int8_t v65 = a2;	// L70
    int8_t v66 = w2;	// L71
    int16_t v67 = v65;	// L72
    int16_t v68 = v66;	// L73
    int16_t v69 = v67 * v68;	// L74
    ap_int<33> v70 = v64;	// L75
    ap_int<33> v71 = v69;	// L76
    ap_int<33> v72 = v70 + v71;	// L77
    int32_t v73 = v72;	// L78
    v53.write(v73); // v53[m2] = v73;	// L79
    int8_t v74 = a2;	// L80
    v52.write(v74); // v52[m2] = v74;	// L81
  }
}

void PE_kernel_ws_gemm_3_0(
  hls::stream< int8_t > &v75 /* v75[8] */,
  hls::stream< int32_t > &v76 /* v76[8] */,
  hls::stream< int8_t > &v77 /* v77[8] */,
  hls::stream< int32_t > &v78 /* v78[8] */,
  int8_t v79[8][8],
  int v80,
  int v81
) {	// L85
  #pragma HLS stream variable=v75 depth=9
  #pragma HLS stream variable=v76 depth=9
  #pragma HLS stream variable=v77 depth=9
  #pragma HLS stream variable=v78 depth=9
  #pragma HLS array_partition variable=v79 complete dim=1
  #pragma HLS array_partition variable=v79 complete dim=2

  int8_t v82 = v79[v80][v81];	// L86
  int8_t w3;	// L87
  w3 = v82;	// L88
  l_step_m3: for (int m3 = 0; m3 < 8; m3++) {	// L89
  #pragma HLS pipeline II=1
    int8_t v85 = v75.read(); // v75[m3];	// L90
    int8_t a3;	// L91
    a3 = v85;	// L92
    int32_t v87 = v76.read(); // v76[m3];	// L93
    int32_t p3;	// L94
    p3 = v87;	// L95
    int32_t v89 = p3;	// L96
    int8_t v90 = a3;	// L97
    int8_t v91 = w3;	// L98
    int16_t v92 = v90;	// L99
    int16_t v93 = v91;	// L100
    int16_t v94 = v92 * v93;	// L101
    ap_int<33> v95 = v89;	// L102
    ap_int<33> v96 = v94;	// L103
    ap_int<33> v97 = v95 + v96;	// L104
    int32_t v98 = v97;	// L105
    v78.write(v98); // v78[m3] = v98;	// L106
    int8_t v99 = a3;	// L107
    v77.write(v99); // v77[m3] = v99;	// L108
  }
}

void PE_kernel_ws_gemm_4_0(
  hls::stream< int8_t > &v100 /* v100[8] */,
  hls::stream< int32_t > &v101 /* v101[8] */,
  hls::stream< int8_t > &v102 /* v102[8] */,
  hls::stream< int32_t > &v103 /* v103[8] */,
  int8_t v104[8][8],
  int v105,
  int v106
) {	// L112
  #pragma HLS stream variable=v100 depth=9
  #pragma HLS stream variable=v101 depth=9
  #pragma HLS stream variable=v102 depth=9
  #pragma HLS stream variable=v103 depth=9
  #pragma HLS array_partition variable=v104 complete dim=1
  #pragma HLS array_partition variable=v104 complete dim=2

  int8_t v107 = v104[v105][v106];	// L113
  int8_t w4;	// L114
  w4 = v107;	// L115
  l_step_m4: for (int m4 = 0; m4 < 8; m4++) {	// L116
  #pragma HLS pipeline II=1
    int8_t v110 = v100.read(); // v100[m4];	// L117
    int8_t a4;	// L118
    a4 = v110;	// L119
    int32_t v112 = v101.read(); // v101[m4];	// L120
    int32_t p4;	// L121
    p4 = v112;	// L122
    int32_t v114 = p4;	// L123
    int8_t v115 = a4;	// L124
    int8_t v116 = w4;	// L125
    int16_t v117 = v115;	// L126
    int16_t v118 = v116;	// L127
    int16_t v119 = v117 * v118;	// L128
    ap_int<33> v120 = v114;	// L129
    ap_int<33> v121 = v119;	// L130
    ap_int<33> v122 = v120 + v121;	// L131
    int32_t v123 = v122;	// L132
    v103.write(v123); // v103[m4] = v123;	// L133
    int8_t v124 = a4;	// L134
    v102.write(v124); // v102[m4] = v124;	// L135
  }
}

void PE_kernel_ws_gemm_5_0(
  hls::stream< int8_t > &v125 /* v125[8] */,
  hls::stream< int32_t > &v126 /* v126[8] */,
  hls::stream< int8_t > &v127 /* v127[8] */,
  hls::stream< int32_t > &v128 /* v128[8] */,
  int8_t v129[8][8],
  int v130,
  int v131
) {	// L139
  #pragma HLS stream variable=v125 depth=9
  #pragma HLS stream variable=v126 depth=9
  #pragma HLS stream variable=v127 depth=9
  #pragma HLS stream variable=v128 depth=9
  #pragma HLS array_partition variable=v129 complete dim=1
  #pragma HLS array_partition variable=v129 complete dim=2

  int8_t v132 = v129[v130][v131];	// L140
  int8_t w5;	// L141
  w5 = v132;	// L142
  l_step_m5: for (int m5 = 0; m5 < 8; m5++) {	// L143
  #pragma HLS pipeline II=1
    int8_t v135 = v125.read(); // v125[m5];	// L144
    int8_t a5;	// L145
    a5 = v135;	// L146
    int32_t v137 = v126.read(); // v126[m5];	// L147
    int32_t p5;	// L148
    p5 = v137;	// L149
    int32_t v139 = p5;	// L150
    int8_t v140 = a5;	// L151
    int8_t v141 = w5;	// L152
    int16_t v142 = v140;	// L153
    int16_t v143 = v141;	// L154
    int16_t v144 = v142 * v143;	// L155
    ap_int<33> v145 = v139;	// L156
    ap_int<33> v146 = v144;	// L157
    ap_int<33> v147 = v145 + v146;	// L158
    int32_t v148 = v147;	// L159
    v128.write(v148); // v128[m5] = v148;	// L160
    int8_t v149 = a5;	// L161
    v127.write(v149); // v127[m5] = v149;	// L162
  }
}

void PE_kernel_ws_gemm_6_0(
  hls::stream< int8_t > &v150 /* v150[8] */,
  hls::stream< int32_t > &v151 /* v151[8] */,
  hls::stream< int8_t > &v152 /* v152[8] */,
  hls::stream< int32_t > &v153 /* v153[8] */,
  int8_t v154[8][8],
  int v155,
  int v156
) {	// L166
  #pragma HLS stream variable=v150 depth=9
  #pragma HLS stream variable=v151 depth=9
  #pragma HLS stream variable=v152 depth=9
  #pragma HLS stream variable=v153 depth=9
  #pragma HLS array_partition variable=v154 complete dim=1
  #pragma HLS array_partition variable=v154 complete dim=2

  int8_t v157 = v154[v155][v156];	// L167
  int8_t w6;	// L168
  w6 = v157;	// L169
  l_step_m6: for (int m6 = 0; m6 < 8; m6++) {	// L170
  #pragma HLS pipeline II=1
    int8_t v160 = v150.read(); // v150[m6];	// L171
    int8_t a6;	// L172
    a6 = v160;	// L173
    int32_t v162 = v151.read(); // v151[m6];	// L174
    int32_t p6;	// L175
    p6 = v162;	// L176
    int32_t v164 = p6;	// L177
    int8_t v165 = a6;	// L178
    int8_t v166 = w6;	// L179
    int16_t v167 = v165;	// L180
    int16_t v168 = v166;	// L181
    int16_t v169 = v167 * v168;	// L182
    ap_int<33> v170 = v164;	// L183
    ap_int<33> v171 = v169;	// L184
    ap_int<33> v172 = v170 + v171;	// L185
    int32_t v173 = v172;	// L186
    v153.write(v173); // v153[m6] = v173;	// L187
    int8_t v174 = a6;	// L188
    v152.write(v174); // v152[m6] = v174;	// L189
  }
}

void PE_kernel_ws_gemm_7_0(
  hls::stream< int8_t > &v175 /* v175[8] */,
  hls::stream< int32_t > &v176 /* v176[8] */,
  hls::stream< int8_t > &v177 /* v177[8] */,
  hls::stream< int32_t > &v178 /* v178[8] */,
  int8_t v179[8][8],
  int v180,
  int v181
) {	// L193
  #pragma HLS stream variable=v175 depth=9
  #pragma HLS stream variable=v176 depth=9
  #pragma HLS stream variable=v177 depth=9
  #pragma HLS stream variable=v178 depth=9
  #pragma HLS array_partition variable=v179 complete dim=1
  #pragma HLS array_partition variable=v179 complete dim=2

  int8_t v182 = v179[v180][v181];	// L194
  int8_t w7;	// L195
  w7 = v182;	// L196
  l_step_m7: for (int m7 = 0; m7 < 8; m7++) {	// L197
  #pragma HLS pipeline II=1
    int8_t v185 = v175.read(); // v175[m7];	// L198
    int8_t a7;	// L199
    a7 = v185;	// L200
    int32_t v187 = v176.read(); // v176[m7];	// L201
    int32_t p7;	// L202
    p7 = v187;	// L203
    int32_t v189 = p7;	// L204
    int8_t v190 = a7;	// L205
    int8_t v191 = w7;	// L206
    int16_t v192 = v190;	// L207
    int16_t v193 = v191;	// L208
    int16_t v194 = v192 * v193;	// L209
    ap_int<33> v195 = v189;	// L210
    ap_int<33> v196 = v194;	// L211
    ap_int<33> v197 = v195 + v196;	// L212
    int32_t v198 = v197;	// L213
    v178.write(v198); // v178[m7] = v198;	// L214
    int8_t v199 = a7;	// L215
    v177.write(v199); // v177[m7] = v199;	// L216
  }
}

void PE_kernel_ws_gemm_0_1(
  hls::stream< int8_t > &v200 /* v200[8] */,
  hls::stream< int32_t > &v201 /* v201[8] */,
  hls::stream< int8_t > &v202 /* v202[8] */,
  hls::stream< int32_t > &v203 /* v203[8] */,
  int8_t v204[8][8],
  int v205,
  int v206
) {	// L220
  #pragma HLS stream variable=v200 depth=9
  #pragma HLS stream variable=v201 depth=9
  #pragma HLS stream variable=v202 depth=9
  #pragma HLS stream variable=v203 depth=9
  #pragma HLS array_partition variable=v204 complete dim=1
  #pragma HLS array_partition variable=v204 complete dim=2

  int8_t v207 = v204[v205][v206];	// L221
  int8_t w8;	// L222
  w8 = v207;	// L223
  l_step_m8: for (int m8 = 0; m8 < 8; m8++) {	// L224
  #pragma HLS pipeline II=1
    int8_t v210 = v200.read(); // v200[m8];	// L225
    int8_t a8;	// L226
    a8 = v210;	// L227
    int32_t v212 = v201.read(); // v201[m8];	// L228
    int32_t p8;	// L229
    p8 = v212;	// L230
    int32_t v214 = p8;	// L231
    int8_t v215 = a8;	// L232
    int8_t v216 = w8;	// L233
    int16_t v217 = v215;	// L234
    int16_t v218 = v216;	// L235
    int16_t v219 = v217 * v218;	// L236
    ap_int<33> v220 = v214;	// L237
    ap_int<33> v221 = v219;	// L238
    ap_int<33> v222 = v220 + v221;	// L239
    int32_t v223 = v222;	// L240
    v203.write(v223); // v203[m8] = v223;	// L241
    int8_t v224 = a8;	// L242
    v202.write(v224); // v202[m8] = v224;	// L243
  }
}

void PE_kernel_ws_gemm_1_1(
  hls::stream< int8_t > &v225 /* v225[8] */,
  hls::stream< int32_t > &v226 /* v226[8] */,
  hls::stream< int8_t > &v227 /* v227[8] */,
  hls::stream< int32_t > &v228 /* v228[8] */,
  int8_t v229[8][8],
  int v230,
  int v231
) {	// L247
  #pragma HLS stream variable=v225 depth=9
  #pragma HLS stream variable=v226 depth=9
  #pragma HLS stream variable=v227 depth=9
  #pragma HLS stream variable=v228 depth=9
  #pragma HLS array_partition variable=v229 complete dim=1
  #pragma HLS array_partition variable=v229 complete dim=2

  int8_t v232 = v229[v230][v231];	// L248
  int8_t w9;	// L249
  w9 = v232;	// L250
  l_step_m9: for (int m9 = 0; m9 < 8; m9++) {	// L251
  #pragma HLS pipeline II=1
    int8_t v235 = v225.read(); // v225[m9];	// L252
    int8_t a9;	// L253
    a9 = v235;	// L254
    int32_t v237 = v226.read(); // v226[m9];	// L255
    int32_t p9;	// L256
    p9 = v237;	// L257
    int32_t v239 = p9;	// L258
    int8_t v240 = a9;	// L259
    int8_t v241 = w9;	// L260
    int16_t v242 = v240;	// L261
    int16_t v243 = v241;	// L262
    int16_t v244 = v242 * v243;	// L263
    ap_int<33> v245 = v239;	// L264
    ap_int<33> v246 = v244;	// L265
    ap_int<33> v247 = v245 + v246;	// L266
    int32_t v248 = v247;	// L267
    v228.write(v248); // v228[m9] = v248;	// L268
    int8_t v249 = a9;	// L269
    v227.write(v249); // v227[m9] = v249;	// L270
  }
}

void PE_kernel_ws_gemm_2_1(
  hls::stream< int8_t > &v250 /* v250[8] */,
  hls::stream< int32_t > &v251 /* v251[8] */,
  hls::stream< int8_t > &v252 /* v252[8] */,
  hls::stream< int32_t > &v253 /* v253[8] */,
  int8_t v254[8][8],
  int v255,
  int v256
) {	// L274
  #pragma HLS stream variable=v250 depth=9
  #pragma HLS stream variable=v251 depth=9
  #pragma HLS stream variable=v252 depth=9
  #pragma HLS stream variable=v253 depth=9
  #pragma HLS array_partition variable=v254 complete dim=1
  #pragma HLS array_partition variable=v254 complete dim=2

  int8_t v257 = v254[v255][v256];	// L275
  int8_t w10;	// L276
  w10 = v257;	// L277
  l_step_m10: for (int m10 = 0; m10 < 8; m10++) {	// L278
  #pragma HLS pipeline II=1
    int8_t v260 = v250.read(); // v250[m10];	// L279
    int8_t a10;	// L280
    a10 = v260;	// L281
    int32_t v262 = v251.read(); // v251[m10];	// L282
    int32_t p10;	// L283
    p10 = v262;	// L284
    int32_t v264 = p10;	// L285
    int8_t v265 = a10;	// L286
    int8_t v266 = w10;	// L287
    int16_t v267 = v265;	// L288
    int16_t v268 = v266;	// L289
    int16_t v269 = v267 * v268;	// L290
    ap_int<33> v270 = v264;	// L291
    ap_int<33> v271 = v269;	// L292
    ap_int<33> v272 = v270 + v271;	// L293
    int32_t v273 = v272;	// L294
    v253.write(v273); // v253[m10] = v273;	// L295
    int8_t v274 = a10;	// L296
    v252.write(v274); // v252[m10] = v274;	// L297
  }
}

void PE_kernel_ws_gemm_3_1(
  hls::stream< int8_t > &v275 /* v275[8] */,
  hls::stream< int32_t > &v276 /* v276[8] */,
  hls::stream< int8_t > &v277 /* v277[8] */,
  hls::stream< int32_t > &v278 /* v278[8] */,
  int8_t v279[8][8],
  int v280,
  int v281
) {	// L301
  #pragma HLS stream variable=v275 depth=9
  #pragma HLS stream variable=v276 depth=9
  #pragma HLS stream variable=v277 depth=9
  #pragma HLS stream variable=v278 depth=9
  #pragma HLS array_partition variable=v279 complete dim=1
  #pragma HLS array_partition variable=v279 complete dim=2

  int8_t v282 = v279[v280][v281];	// L302
  int8_t w11;	// L303
  w11 = v282;	// L304
  l_step_m11: for (int m11 = 0; m11 < 8; m11++) {	// L305
  #pragma HLS pipeline II=1
    int8_t v285 = v275.read(); // v275[m11];	// L306
    int8_t a11;	// L307
    a11 = v285;	// L308
    int32_t v287 = v276.read(); // v276[m11];	// L309
    int32_t p11;	// L310
    p11 = v287;	// L311
    int32_t v289 = p11;	// L312
    int8_t v290 = a11;	// L313
    int8_t v291 = w11;	// L314
    int16_t v292 = v290;	// L315
    int16_t v293 = v291;	// L316
    int16_t v294 = v292 * v293;	// L317
    ap_int<33> v295 = v289;	// L318
    ap_int<33> v296 = v294;	// L319
    ap_int<33> v297 = v295 + v296;	// L320
    int32_t v298 = v297;	// L321
    v278.write(v298); // v278[m11] = v298;	// L322
    int8_t v299 = a11;	// L323
    v277.write(v299); // v277[m11] = v299;	// L324
  }
}

void PE_kernel_ws_gemm_4_1(
  hls::stream< int8_t > &v300 /* v300[8] */,
  hls::stream< int32_t > &v301 /* v301[8] */,
  hls::stream< int8_t > &v302 /* v302[8] */,
  hls::stream< int32_t > &v303 /* v303[8] */,
  int8_t v304[8][8],
  int v305,
  int v306
) {	// L328
  #pragma HLS stream variable=v300 depth=9
  #pragma HLS stream variable=v301 depth=9
  #pragma HLS stream variable=v302 depth=9
  #pragma HLS stream variable=v303 depth=9
  #pragma HLS array_partition variable=v304 complete dim=1
  #pragma HLS array_partition variable=v304 complete dim=2

  int8_t v307 = v304[v305][v306];	// L329
  int8_t w12;	// L330
  w12 = v307;	// L331
  l_step_m12: for (int m12 = 0; m12 < 8; m12++) {	// L332
  #pragma HLS pipeline II=1
    int8_t v310 = v300.read(); // v300[m12];	// L333
    int8_t a12;	// L334
    a12 = v310;	// L335
    int32_t v312 = v301.read(); // v301[m12];	// L336
    int32_t p12;	// L337
    p12 = v312;	// L338
    int32_t v314 = p12;	// L339
    int8_t v315 = a12;	// L340
    int8_t v316 = w12;	// L341
    int16_t v317 = v315;	// L342
    int16_t v318 = v316;	// L343
    int16_t v319 = v317 * v318;	// L344
    ap_int<33> v320 = v314;	// L345
    ap_int<33> v321 = v319;	// L346
    ap_int<33> v322 = v320 + v321;	// L347
    int32_t v323 = v322;	// L348
    v303.write(v323); // v303[m12] = v323;	// L349
    int8_t v324 = a12;	// L350
    v302.write(v324); // v302[m12] = v324;	// L351
  }
}

void PE_kernel_ws_gemm_5_1(
  hls::stream< int8_t > &v325 /* v325[8] */,
  hls::stream< int32_t > &v326 /* v326[8] */,
  hls::stream< int8_t > &v327 /* v327[8] */,
  hls::stream< int32_t > &v328 /* v328[8] */,
  int8_t v329[8][8],
  int v330,
  int v331
) {	// L355
  #pragma HLS stream variable=v325 depth=9
  #pragma HLS stream variable=v326 depth=9
  #pragma HLS stream variable=v327 depth=9
  #pragma HLS stream variable=v328 depth=9
  #pragma HLS array_partition variable=v329 complete dim=1
  #pragma HLS array_partition variable=v329 complete dim=2

  int8_t v332 = v329[v330][v331];	// L356
  int8_t w13;	// L357
  w13 = v332;	// L358
  l_step_m13: for (int m13 = 0; m13 < 8; m13++) {	// L359
  #pragma HLS pipeline II=1
    int8_t v335 = v325.read(); // v325[m13];	// L360
    int8_t a13;	// L361
    a13 = v335;	// L362
    int32_t v337 = v326.read(); // v326[m13];	// L363
    int32_t p13;	// L364
    p13 = v337;	// L365
    int32_t v339 = p13;	// L366
    int8_t v340 = a13;	// L367
    int8_t v341 = w13;	// L368
    int16_t v342 = v340;	// L369
    int16_t v343 = v341;	// L370
    int16_t v344 = v342 * v343;	// L371
    ap_int<33> v345 = v339;	// L372
    ap_int<33> v346 = v344;	// L373
    ap_int<33> v347 = v345 + v346;	// L374
    int32_t v348 = v347;	// L375
    v328.write(v348); // v328[m13] = v348;	// L376
    int8_t v349 = a13;	// L377
    v327.write(v349); // v327[m13] = v349;	// L378
  }
}

void PE_kernel_ws_gemm_6_1(
  hls::stream< int8_t > &v350 /* v350[8] */,
  hls::stream< int32_t > &v351 /* v351[8] */,
  hls::stream< int8_t > &v352 /* v352[8] */,
  hls::stream< int32_t > &v353 /* v353[8] */,
  int8_t v354[8][8],
  int v355,
  int v356
) {	// L382
  #pragma HLS stream variable=v350 depth=9
  #pragma HLS stream variable=v351 depth=9
  #pragma HLS stream variable=v352 depth=9
  #pragma HLS stream variable=v353 depth=9
  #pragma HLS array_partition variable=v354 complete dim=1
  #pragma HLS array_partition variable=v354 complete dim=2

  int8_t v357 = v354[v355][v356];	// L383
  int8_t w14;	// L384
  w14 = v357;	// L385
  l_step_m14: for (int m14 = 0; m14 < 8; m14++) {	// L386
  #pragma HLS pipeline II=1
    int8_t v360 = v350.read(); // v350[m14];	// L387
    int8_t a14;	// L388
    a14 = v360;	// L389
    int32_t v362 = v351.read(); // v351[m14];	// L390
    int32_t p14;	// L391
    p14 = v362;	// L392
    int32_t v364 = p14;	// L393
    int8_t v365 = a14;	// L394
    int8_t v366 = w14;	// L395
    int16_t v367 = v365;	// L396
    int16_t v368 = v366;	// L397
    int16_t v369 = v367 * v368;	// L398
    ap_int<33> v370 = v364;	// L399
    ap_int<33> v371 = v369;	// L400
    ap_int<33> v372 = v370 + v371;	// L401
    int32_t v373 = v372;	// L402
    v353.write(v373); // v353[m14] = v373;	// L403
    int8_t v374 = a14;	// L404
    v352.write(v374); // v352[m14] = v374;	// L405
  }
}

void PE_kernel_ws_gemm_7_1(
  hls::stream< int8_t > &v375 /* v375[8] */,
  hls::stream< int32_t > &v376 /* v376[8] */,
  hls::stream< int8_t > &v377 /* v377[8] */,
  hls::stream< int32_t > &v378 /* v378[8] */,
  int8_t v379[8][8],
  int v380,
  int v381
) {	// L409
  #pragma HLS stream variable=v375 depth=9
  #pragma HLS stream variable=v376 depth=9
  #pragma HLS stream variable=v377 depth=9
  #pragma HLS stream variable=v378 depth=9
  #pragma HLS array_partition variable=v379 complete dim=1
  #pragma HLS array_partition variable=v379 complete dim=2

  int8_t v382 = v379[v380][v381];	// L410
  int8_t w15;	// L411
  w15 = v382;	// L412
  l_step_m15: for (int m15 = 0; m15 < 8; m15++) {	// L413
  #pragma HLS pipeline II=1
    int8_t v385 = v375.read(); // v375[m15];	// L414
    int8_t a15;	// L415
    a15 = v385;	// L416
    int32_t v387 = v376.read(); // v376[m15];	// L417
    int32_t p15;	// L418
    p15 = v387;	// L419
    int32_t v389 = p15;	// L420
    int8_t v390 = a15;	// L421
    int8_t v391 = w15;	// L422
    int16_t v392 = v390;	// L423
    int16_t v393 = v391;	// L424
    int16_t v394 = v392 * v393;	// L425
    ap_int<33> v395 = v389;	// L426
    ap_int<33> v396 = v394;	// L427
    ap_int<33> v397 = v395 + v396;	// L428
    int32_t v398 = v397;	// L429
    v378.write(v398); // v378[m15] = v398;	// L430
    int8_t v399 = a15;	// L431
    v377.write(v399); // v377[m15] = v399;	// L432
  }
}

void PE_kernel_ws_gemm_0_2(
  hls::stream< int8_t > &v400 /* v400[8] */,
  hls::stream< int32_t > &v401 /* v401[8] */,
  hls::stream< int8_t > &v402 /* v402[8] */,
  hls::stream< int32_t > &v403 /* v403[8] */,
  int8_t v404[8][8],
  int v405,
  int v406
) {	// L436
  #pragma HLS stream variable=v400 depth=9
  #pragma HLS stream variable=v401 depth=9
  #pragma HLS stream variable=v402 depth=9
  #pragma HLS stream variable=v403 depth=9
  #pragma HLS array_partition variable=v404 complete dim=1
  #pragma HLS array_partition variable=v404 complete dim=2

  int8_t v407 = v404[v405][v406];	// L437
  int8_t w16;	// L438
  w16 = v407;	// L439
  l_step_m16: for (int m16 = 0; m16 < 8; m16++) {	// L440
  #pragma HLS pipeline II=1
    int8_t v410 = v400.read(); // v400[m16];	// L441
    int8_t a16;	// L442
    a16 = v410;	// L443
    int32_t v412 = v401.read(); // v401[m16];	// L444
    int32_t p16;	// L445
    p16 = v412;	// L446
    int32_t v414 = p16;	// L447
    int8_t v415 = a16;	// L448
    int8_t v416 = w16;	// L449
    int16_t v417 = v415;	// L450
    int16_t v418 = v416;	// L451
    int16_t v419 = v417 * v418;	// L452
    ap_int<33> v420 = v414;	// L453
    ap_int<33> v421 = v419;	// L454
    ap_int<33> v422 = v420 + v421;	// L455
    int32_t v423 = v422;	// L456
    v403.write(v423); // v403[m16] = v423;	// L457
    int8_t v424 = a16;	// L458
    v402.write(v424); // v402[m16] = v424;	// L459
  }
}

void PE_kernel_ws_gemm_1_2(
  hls::stream< int8_t > &v425 /* v425[8] */,
  hls::stream< int32_t > &v426 /* v426[8] */,
  hls::stream< int8_t > &v427 /* v427[8] */,
  hls::stream< int32_t > &v428 /* v428[8] */,
  int8_t v429[8][8],
  int v430,
  int v431
) {	// L463
  #pragma HLS stream variable=v425 depth=9
  #pragma HLS stream variable=v426 depth=9
  #pragma HLS stream variable=v427 depth=9
  #pragma HLS stream variable=v428 depth=9
  #pragma HLS array_partition variable=v429 complete dim=1
  #pragma HLS array_partition variable=v429 complete dim=2

  int8_t v432 = v429[v430][v431];	// L464
  int8_t w17;	// L465
  w17 = v432;	// L466
  l_step_m17: for (int m17 = 0; m17 < 8; m17++) {	// L467
  #pragma HLS pipeline II=1
    int8_t v435 = v425.read(); // v425[m17];	// L468
    int8_t a17;	// L469
    a17 = v435;	// L470
    int32_t v437 = v426.read(); // v426[m17];	// L471
    int32_t p17;	// L472
    p17 = v437;	// L473
    int32_t v439 = p17;	// L474
    int8_t v440 = a17;	// L475
    int8_t v441 = w17;	// L476
    int16_t v442 = v440;	// L477
    int16_t v443 = v441;	// L478
    int16_t v444 = v442 * v443;	// L479
    ap_int<33> v445 = v439;	// L480
    ap_int<33> v446 = v444;	// L481
    ap_int<33> v447 = v445 + v446;	// L482
    int32_t v448 = v447;	// L483
    v428.write(v448); // v428[m17] = v448;	// L484
    int8_t v449 = a17;	// L485
    v427.write(v449); // v427[m17] = v449;	// L486
  }
}

void PE_kernel_ws_gemm_2_2(
  hls::stream< int8_t > &v450 /* v450[8] */,
  hls::stream< int32_t > &v451 /* v451[8] */,
  hls::stream< int8_t > &v452 /* v452[8] */,
  hls::stream< int32_t > &v453 /* v453[8] */,
  int8_t v454[8][8],
  int v455,
  int v456
) {	// L490
  #pragma HLS stream variable=v450 depth=9
  #pragma HLS stream variable=v451 depth=9
  #pragma HLS stream variable=v452 depth=9
  #pragma HLS stream variable=v453 depth=9
  #pragma HLS array_partition variable=v454 complete dim=1
  #pragma HLS array_partition variable=v454 complete dim=2

  int8_t v457 = v454[v455][v456];	// L491
  int8_t w18;	// L492
  w18 = v457;	// L493
  l_step_m18: for (int m18 = 0; m18 < 8; m18++) {	// L494
  #pragma HLS pipeline II=1
    int8_t v460 = v450.read(); // v450[m18];	// L495
    int8_t a18;	// L496
    a18 = v460;	// L497
    int32_t v462 = v451.read(); // v451[m18];	// L498
    int32_t p18;	// L499
    p18 = v462;	// L500
    int32_t v464 = p18;	// L501
    int8_t v465 = a18;	// L502
    int8_t v466 = w18;	// L503
    int16_t v467 = v465;	// L504
    int16_t v468 = v466;	// L505
    int16_t v469 = v467 * v468;	// L506
    ap_int<33> v470 = v464;	// L507
    ap_int<33> v471 = v469;	// L508
    ap_int<33> v472 = v470 + v471;	// L509
    int32_t v473 = v472;	// L510
    v453.write(v473); // v453[m18] = v473;	// L511
    int8_t v474 = a18;	// L512
    v452.write(v474); // v452[m18] = v474;	// L513
  }
}

void PE_kernel_ws_gemm_3_2(
  hls::stream< int8_t > &v475 /* v475[8] */,
  hls::stream< int32_t > &v476 /* v476[8] */,
  hls::stream< int8_t > &v477 /* v477[8] */,
  hls::stream< int32_t > &v478 /* v478[8] */,
  int8_t v479[8][8],
  int v480,
  int v481
) {	// L517
  #pragma HLS stream variable=v475 depth=9
  #pragma HLS stream variable=v476 depth=9
  #pragma HLS stream variable=v477 depth=9
  #pragma HLS stream variable=v478 depth=9
  #pragma HLS array_partition variable=v479 complete dim=1
  #pragma HLS array_partition variable=v479 complete dim=2

  int8_t v482 = v479[v480][v481];	// L518
  int8_t w19;	// L519
  w19 = v482;	// L520
  l_step_m19: for (int m19 = 0; m19 < 8; m19++) {	// L521
  #pragma HLS pipeline II=1
    int8_t v485 = v475.read(); // v475[m19];	// L522
    int8_t a19;	// L523
    a19 = v485;	// L524
    int32_t v487 = v476.read(); // v476[m19];	// L525
    int32_t p19;	// L526
    p19 = v487;	// L527
    int32_t v489 = p19;	// L528
    int8_t v490 = a19;	// L529
    int8_t v491 = w19;	// L530
    int16_t v492 = v490;	// L531
    int16_t v493 = v491;	// L532
    int16_t v494 = v492 * v493;	// L533
    ap_int<33> v495 = v489;	// L534
    ap_int<33> v496 = v494;	// L535
    ap_int<33> v497 = v495 + v496;	// L536
    int32_t v498 = v497;	// L537
    v478.write(v498); // v478[m19] = v498;	// L538
    int8_t v499 = a19;	// L539
    v477.write(v499); // v477[m19] = v499;	// L540
  }
}

void PE_kernel_ws_gemm_4_2(
  hls::stream< int8_t > &v500 /* v500[8] */,
  hls::stream< int32_t > &v501 /* v501[8] */,
  hls::stream< int8_t > &v502 /* v502[8] */,
  hls::stream< int32_t > &v503 /* v503[8] */,
  int8_t v504[8][8],
  int v505,
  int v506
) {	// L544
  #pragma HLS stream variable=v500 depth=9
  #pragma HLS stream variable=v501 depth=9
  #pragma HLS stream variable=v502 depth=9
  #pragma HLS stream variable=v503 depth=9
  #pragma HLS array_partition variable=v504 complete dim=1
  #pragma HLS array_partition variable=v504 complete dim=2

  int8_t v507 = v504[v505][v506];	// L545
  int8_t w20;	// L546
  w20 = v507;	// L547
  l_step_m20: for (int m20 = 0; m20 < 8; m20++) {	// L548
  #pragma HLS pipeline II=1
    int8_t v510 = v500.read(); // v500[m20];	// L549
    int8_t a20;	// L550
    a20 = v510;	// L551
    int32_t v512 = v501.read(); // v501[m20];	// L552
    int32_t p20;	// L553
    p20 = v512;	// L554
    int32_t v514 = p20;	// L555
    int8_t v515 = a20;	// L556
    int8_t v516 = w20;	// L557
    int16_t v517 = v515;	// L558
    int16_t v518 = v516;	// L559
    int16_t v519 = v517 * v518;	// L560
    ap_int<33> v520 = v514;	// L561
    ap_int<33> v521 = v519;	// L562
    ap_int<33> v522 = v520 + v521;	// L563
    int32_t v523 = v522;	// L564
    v503.write(v523); // v503[m20] = v523;	// L565
    int8_t v524 = a20;	// L566
    v502.write(v524); // v502[m20] = v524;	// L567
  }
}

void PE_kernel_ws_gemm_5_2(
  hls::stream< int8_t > &v525 /* v525[8] */,
  hls::stream< int32_t > &v526 /* v526[8] */,
  hls::stream< int8_t > &v527 /* v527[8] */,
  hls::stream< int32_t > &v528 /* v528[8] */,
  int8_t v529[8][8],
  int v530,
  int v531
) {	// L571
  #pragma HLS stream variable=v525 depth=9
  #pragma HLS stream variable=v526 depth=9
  #pragma HLS stream variable=v527 depth=9
  #pragma HLS stream variable=v528 depth=9
  #pragma HLS array_partition variable=v529 complete dim=1
  #pragma HLS array_partition variable=v529 complete dim=2

  int8_t v532 = v529[v530][v531];	// L572
  int8_t w21;	// L573
  w21 = v532;	// L574
  l_step_m21: for (int m21 = 0; m21 < 8; m21++) {	// L575
  #pragma HLS pipeline II=1
    int8_t v535 = v525.read(); // v525[m21];	// L576
    int8_t a21;	// L577
    a21 = v535;	// L578
    int32_t v537 = v526.read(); // v526[m21];	// L579
    int32_t p21;	// L580
    p21 = v537;	// L581
    int32_t v539 = p21;	// L582
    int8_t v540 = a21;	// L583
    int8_t v541 = w21;	// L584
    int16_t v542 = v540;	// L585
    int16_t v543 = v541;	// L586
    int16_t v544 = v542 * v543;	// L587
    ap_int<33> v545 = v539;	// L588
    ap_int<33> v546 = v544;	// L589
    ap_int<33> v547 = v545 + v546;	// L590
    int32_t v548 = v547;	// L591
    v528.write(v548); // v528[m21] = v548;	// L592
    int8_t v549 = a21;	// L593
    v527.write(v549); // v527[m21] = v549;	// L594
  }
}

void PE_kernel_ws_gemm_6_2(
  hls::stream< int8_t > &v550 /* v550[8] */,
  hls::stream< int32_t > &v551 /* v551[8] */,
  hls::stream< int8_t > &v552 /* v552[8] */,
  hls::stream< int32_t > &v553 /* v553[8] */,
  int8_t v554[8][8],
  int v555,
  int v556
) {	// L598
  #pragma HLS stream variable=v550 depth=9
  #pragma HLS stream variable=v551 depth=9
  #pragma HLS stream variable=v552 depth=9
  #pragma HLS stream variable=v553 depth=9
  #pragma HLS array_partition variable=v554 complete dim=1
  #pragma HLS array_partition variable=v554 complete dim=2

  int8_t v557 = v554[v555][v556];	// L599
  int8_t w22;	// L600
  w22 = v557;	// L601
  l_step_m22: for (int m22 = 0; m22 < 8; m22++) {	// L602
  #pragma HLS pipeline II=1
    int8_t v560 = v550.read(); // v550[m22];	// L603
    int8_t a22;	// L604
    a22 = v560;	// L605
    int32_t v562 = v551.read(); // v551[m22];	// L606
    int32_t p22;	// L607
    p22 = v562;	// L608
    int32_t v564 = p22;	// L609
    int8_t v565 = a22;	// L610
    int8_t v566 = w22;	// L611
    int16_t v567 = v565;	// L612
    int16_t v568 = v566;	// L613
    int16_t v569 = v567 * v568;	// L614
    ap_int<33> v570 = v564;	// L615
    ap_int<33> v571 = v569;	// L616
    ap_int<33> v572 = v570 + v571;	// L617
    int32_t v573 = v572;	// L618
    v553.write(v573); // v553[m22] = v573;	// L619
    int8_t v574 = a22;	// L620
    v552.write(v574); // v552[m22] = v574;	// L621
  }
}

void PE_kernel_ws_gemm_7_2(
  hls::stream< int8_t > &v575 /* v575[8] */,
  hls::stream< int32_t > &v576 /* v576[8] */,
  hls::stream< int8_t > &v577 /* v577[8] */,
  hls::stream< int32_t > &v578 /* v578[8] */,
  int8_t v579[8][8],
  int v580,
  int v581
) {	// L625
  #pragma HLS stream variable=v575 depth=9
  #pragma HLS stream variable=v576 depth=9
  #pragma HLS stream variable=v577 depth=9
  #pragma HLS stream variable=v578 depth=9
  #pragma HLS array_partition variable=v579 complete dim=1
  #pragma HLS array_partition variable=v579 complete dim=2

  int8_t v582 = v579[v580][v581];	// L626
  int8_t w23;	// L627
  w23 = v582;	// L628
  l_step_m23: for (int m23 = 0; m23 < 8; m23++) {	// L629
  #pragma HLS pipeline II=1
    int8_t v585 = v575.read(); // v575[m23];	// L630
    int8_t a23;	// L631
    a23 = v585;	// L632
    int32_t v587 = v576.read(); // v576[m23];	// L633
    int32_t p23;	// L634
    p23 = v587;	// L635
    int32_t v589 = p23;	// L636
    int8_t v590 = a23;	// L637
    int8_t v591 = w23;	// L638
    int16_t v592 = v590;	// L639
    int16_t v593 = v591;	// L640
    int16_t v594 = v592 * v593;	// L641
    ap_int<33> v595 = v589;	// L642
    ap_int<33> v596 = v594;	// L643
    ap_int<33> v597 = v595 + v596;	// L644
    int32_t v598 = v597;	// L645
    v578.write(v598); // v578[m23] = v598;	// L646
    int8_t v599 = a23;	// L647
    v577.write(v599); // v577[m23] = v599;	// L648
  }
}

void PE_kernel_ws_gemm_0_3(
  hls::stream< int8_t > &v600 /* v600[8] */,
  hls::stream< int32_t > &v601 /* v601[8] */,
  hls::stream< int8_t > &v602 /* v602[8] */,
  hls::stream< int32_t > &v603 /* v603[8] */,
  int8_t v604[8][8],
  int v605,
  int v606
) {	// L652
  #pragma HLS stream variable=v600 depth=9
  #pragma HLS stream variable=v601 depth=9
  #pragma HLS stream variable=v602 depth=9
  #pragma HLS stream variable=v603 depth=9
  #pragma HLS array_partition variable=v604 complete dim=1
  #pragma HLS array_partition variable=v604 complete dim=2

  int8_t v607 = v604[v605][v606];	// L653
  int8_t w24;	// L654
  w24 = v607;	// L655
  l_step_m24: for (int m24 = 0; m24 < 8; m24++) {	// L656
  #pragma HLS pipeline II=1
    int8_t v610 = v600.read(); // v600[m24];	// L657
    int8_t a24;	// L658
    a24 = v610;	// L659
    int32_t v612 = v601.read(); // v601[m24];	// L660
    int32_t p24;	// L661
    p24 = v612;	// L662
    int32_t v614 = p24;	// L663
    int8_t v615 = a24;	// L664
    int8_t v616 = w24;	// L665
    int16_t v617 = v615;	// L666
    int16_t v618 = v616;	// L667
    int16_t v619 = v617 * v618;	// L668
    ap_int<33> v620 = v614;	// L669
    ap_int<33> v621 = v619;	// L670
    ap_int<33> v622 = v620 + v621;	// L671
    int32_t v623 = v622;	// L672
    v603.write(v623); // v603[m24] = v623;	// L673
    int8_t v624 = a24;	// L674
    v602.write(v624); // v602[m24] = v624;	// L675
  }
}

void PE_kernel_ws_gemm_1_3(
  hls::stream< int8_t > &v625 /* v625[8] */,
  hls::stream< int32_t > &v626 /* v626[8] */,
  hls::stream< int8_t > &v627 /* v627[8] */,
  hls::stream< int32_t > &v628 /* v628[8] */,
  int8_t v629[8][8],
  int v630,
  int v631
) {	// L679
  #pragma HLS stream variable=v625 depth=9
  #pragma HLS stream variable=v626 depth=9
  #pragma HLS stream variable=v627 depth=9
  #pragma HLS stream variable=v628 depth=9
  #pragma HLS array_partition variable=v629 complete dim=1
  #pragma HLS array_partition variable=v629 complete dim=2

  int8_t v632 = v629[v630][v631];	// L680
  int8_t w25;	// L681
  w25 = v632;	// L682
  l_step_m25: for (int m25 = 0; m25 < 8; m25++) {	// L683
  #pragma HLS pipeline II=1
    int8_t v635 = v625.read(); // v625[m25];	// L684
    int8_t a25;	// L685
    a25 = v635;	// L686
    int32_t v637 = v626.read(); // v626[m25];	// L687
    int32_t p25;	// L688
    p25 = v637;	// L689
    int32_t v639 = p25;	// L690
    int8_t v640 = a25;	// L691
    int8_t v641 = w25;	// L692
    int16_t v642 = v640;	// L693
    int16_t v643 = v641;	// L694
    int16_t v644 = v642 * v643;	// L695
    ap_int<33> v645 = v639;	// L696
    ap_int<33> v646 = v644;	// L697
    ap_int<33> v647 = v645 + v646;	// L698
    int32_t v648 = v647;	// L699
    v628.write(v648); // v628[m25] = v648;	// L700
    int8_t v649 = a25;	// L701
    v627.write(v649); // v627[m25] = v649;	// L702
  }
}

void PE_kernel_ws_gemm_2_3(
  hls::stream< int8_t > &v650 /* v650[8] */,
  hls::stream< int32_t > &v651 /* v651[8] */,
  hls::stream< int8_t > &v652 /* v652[8] */,
  hls::stream< int32_t > &v653 /* v653[8] */,
  int8_t v654[8][8],
  int v655,
  int v656
) {	// L706
  #pragma HLS stream variable=v650 depth=9
  #pragma HLS stream variable=v651 depth=9
  #pragma HLS stream variable=v652 depth=9
  #pragma HLS stream variable=v653 depth=9
  #pragma HLS array_partition variable=v654 complete dim=1
  #pragma HLS array_partition variable=v654 complete dim=2

  int8_t v657 = v654[v655][v656];	// L707
  int8_t w26;	// L708
  w26 = v657;	// L709
  l_step_m26: for (int m26 = 0; m26 < 8; m26++) {	// L710
  #pragma HLS pipeline II=1
    int8_t v660 = v650.read(); // v650[m26];	// L711
    int8_t a26;	// L712
    a26 = v660;	// L713
    int32_t v662 = v651.read(); // v651[m26];	// L714
    int32_t p26;	// L715
    p26 = v662;	// L716
    int32_t v664 = p26;	// L717
    int8_t v665 = a26;	// L718
    int8_t v666 = w26;	// L719
    int16_t v667 = v665;	// L720
    int16_t v668 = v666;	// L721
    int16_t v669 = v667 * v668;	// L722
    ap_int<33> v670 = v664;	// L723
    ap_int<33> v671 = v669;	// L724
    ap_int<33> v672 = v670 + v671;	// L725
    int32_t v673 = v672;	// L726
    v653.write(v673); // v653[m26] = v673;	// L727
    int8_t v674 = a26;	// L728
    v652.write(v674); // v652[m26] = v674;	// L729
  }
}

void PE_kernel_ws_gemm_3_3(
  hls::stream< int8_t > &v675 /* v675[8] */,
  hls::stream< int32_t > &v676 /* v676[8] */,
  hls::stream< int8_t > &v677 /* v677[8] */,
  hls::stream< int32_t > &v678 /* v678[8] */,
  int8_t v679[8][8],
  int v680,
  int v681
) {	// L733
  #pragma HLS stream variable=v675 depth=9
  #pragma HLS stream variable=v676 depth=9
  #pragma HLS stream variable=v677 depth=9
  #pragma HLS stream variable=v678 depth=9
  #pragma HLS array_partition variable=v679 complete dim=1
  #pragma HLS array_partition variable=v679 complete dim=2

  int8_t v682 = v679[v680][v681];	// L734
  int8_t w27;	// L735
  w27 = v682;	// L736
  l_step_m27: for (int m27 = 0; m27 < 8; m27++) {	// L737
  #pragma HLS pipeline II=1
    int8_t v685 = v675.read(); // v675[m27];	// L738
    int8_t a27;	// L739
    a27 = v685;	// L740
    int32_t v687 = v676.read(); // v676[m27];	// L741
    int32_t p27;	// L742
    p27 = v687;	// L743
    int32_t v689 = p27;	// L744
    int8_t v690 = a27;	// L745
    int8_t v691 = w27;	// L746
    int16_t v692 = v690;	// L747
    int16_t v693 = v691;	// L748
    int16_t v694 = v692 * v693;	// L749
    ap_int<33> v695 = v689;	// L750
    ap_int<33> v696 = v694;	// L751
    ap_int<33> v697 = v695 + v696;	// L752
    int32_t v698 = v697;	// L753
    v678.write(v698); // v678[m27] = v698;	// L754
    int8_t v699 = a27;	// L755
    v677.write(v699); // v677[m27] = v699;	// L756
  }
}

void PE_kernel_ws_gemm_4_3(
  hls::stream< int8_t > &v700 /* v700[8] */,
  hls::stream< int32_t > &v701 /* v701[8] */,
  hls::stream< int8_t > &v702 /* v702[8] */,
  hls::stream< int32_t > &v703 /* v703[8] */,
  int8_t v704[8][8],
  int v705,
  int v706
) {	// L760
  #pragma HLS stream variable=v700 depth=9
  #pragma HLS stream variable=v701 depth=9
  #pragma HLS stream variable=v702 depth=9
  #pragma HLS stream variable=v703 depth=9
  #pragma HLS array_partition variable=v704 complete dim=1
  #pragma HLS array_partition variable=v704 complete dim=2

  int8_t v707 = v704[v705][v706];	// L761
  int8_t w28;	// L762
  w28 = v707;	// L763
  l_step_m28: for (int m28 = 0; m28 < 8; m28++) {	// L764
  #pragma HLS pipeline II=1
    int8_t v710 = v700.read(); // v700[m28];	// L765
    int8_t a28;	// L766
    a28 = v710;	// L767
    int32_t v712 = v701.read(); // v701[m28];	// L768
    int32_t p28;	// L769
    p28 = v712;	// L770
    int32_t v714 = p28;	// L771
    int8_t v715 = a28;	// L772
    int8_t v716 = w28;	// L773
    int16_t v717 = v715;	// L774
    int16_t v718 = v716;	// L775
    int16_t v719 = v717 * v718;	// L776
    ap_int<33> v720 = v714;	// L777
    ap_int<33> v721 = v719;	// L778
    ap_int<33> v722 = v720 + v721;	// L779
    int32_t v723 = v722;	// L780
    v703.write(v723); // v703[m28] = v723;	// L781
    int8_t v724 = a28;	// L782
    v702.write(v724); // v702[m28] = v724;	// L783
  }
}

void PE_kernel_ws_gemm_5_3(
  hls::stream< int8_t > &v725 /* v725[8] */,
  hls::stream< int32_t > &v726 /* v726[8] */,
  hls::stream< int8_t > &v727 /* v727[8] */,
  hls::stream< int32_t > &v728 /* v728[8] */,
  int8_t v729[8][8],
  int v730,
  int v731
) {	// L787
  #pragma HLS stream variable=v725 depth=9
  #pragma HLS stream variable=v726 depth=9
  #pragma HLS stream variable=v727 depth=9
  #pragma HLS stream variable=v728 depth=9
  #pragma HLS array_partition variable=v729 complete dim=1
  #pragma HLS array_partition variable=v729 complete dim=2

  int8_t v732 = v729[v730][v731];	// L788
  int8_t w29;	// L789
  w29 = v732;	// L790
  l_step_m29: for (int m29 = 0; m29 < 8; m29++) {	// L791
  #pragma HLS pipeline II=1
    int8_t v735 = v725.read(); // v725[m29];	// L792
    int8_t a29;	// L793
    a29 = v735;	// L794
    int32_t v737 = v726.read(); // v726[m29];	// L795
    int32_t p29;	// L796
    p29 = v737;	// L797
    int32_t v739 = p29;	// L798
    int8_t v740 = a29;	// L799
    int8_t v741 = w29;	// L800
    int16_t v742 = v740;	// L801
    int16_t v743 = v741;	// L802
    int16_t v744 = v742 * v743;	// L803
    ap_int<33> v745 = v739;	// L804
    ap_int<33> v746 = v744;	// L805
    ap_int<33> v747 = v745 + v746;	// L806
    int32_t v748 = v747;	// L807
    v728.write(v748); // v728[m29] = v748;	// L808
    int8_t v749 = a29;	// L809
    v727.write(v749); // v727[m29] = v749;	// L810
  }
}

void PE_kernel_ws_gemm_6_3(
  hls::stream< int8_t > &v750 /* v750[8] */,
  hls::stream< int32_t > &v751 /* v751[8] */,
  hls::stream< int8_t > &v752 /* v752[8] */,
  hls::stream< int32_t > &v753 /* v753[8] */,
  int8_t v754[8][8],
  int v755,
  int v756
) {	// L814
  #pragma HLS stream variable=v750 depth=9
  #pragma HLS stream variable=v751 depth=9
  #pragma HLS stream variable=v752 depth=9
  #pragma HLS stream variable=v753 depth=9
  #pragma HLS array_partition variable=v754 complete dim=1
  #pragma HLS array_partition variable=v754 complete dim=2

  int8_t v757 = v754[v755][v756];	// L815
  int8_t w30;	// L816
  w30 = v757;	// L817
  l_step_m30: for (int m30 = 0; m30 < 8; m30++) {	// L818
  #pragma HLS pipeline II=1
    int8_t v760 = v750.read(); // v750[m30];	// L819
    int8_t a30;	// L820
    a30 = v760;	// L821
    int32_t v762 = v751.read(); // v751[m30];	// L822
    int32_t p30;	// L823
    p30 = v762;	// L824
    int32_t v764 = p30;	// L825
    int8_t v765 = a30;	// L826
    int8_t v766 = w30;	// L827
    int16_t v767 = v765;	// L828
    int16_t v768 = v766;	// L829
    int16_t v769 = v767 * v768;	// L830
    ap_int<33> v770 = v764;	// L831
    ap_int<33> v771 = v769;	// L832
    ap_int<33> v772 = v770 + v771;	// L833
    int32_t v773 = v772;	// L834
    v753.write(v773); // v753[m30] = v773;	// L835
    int8_t v774 = a30;	// L836
    v752.write(v774); // v752[m30] = v774;	// L837
  }
}

void PE_kernel_ws_gemm_7_3(
  hls::stream< int8_t > &v775 /* v775[8] */,
  hls::stream< int32_t > &v776 /* v776[8] */,
  hls::stream< int8_t > &v777 /* v777[8] */,
  hls::stream< int32_t > &v778 /* v778[8] */,
  int8_t v779[8][8],
  int v780,
  int v781
) {	// L841
  #pragma HLS stream variable=v775 depth=9
  #pragma HLS stream variable=v776 depth=9
  #pragma HLS stream variable=v777 depth=9
  #pragma HLS stream variable=v778 depth=9
  #pragma HLS array_partition variable=v779 complete dim=1
  #pragma HLS array_partition variable=v779 complete dim=2

  int8_t v782 = v779[v780][v781];	// L842
  int8_t w31;	// L843
  w31 = v782;	// L844
  l_step_m31: for (int m31 = 0; m31 < 8; m31++) {	// L845
  #pragma HLS pipeline II=1
    int8_t v785 = v775.read(); // v775[m31];	// L846
    int8_t a31;	// L847
    a31 = v785;	// L848
    int32_t v787 = v776.read(); // v776[m31];	// L849
    int32_t p31;	// L850
    p31 = v787;	// L851
    int32_t v789 = p31;	// L852
    int8_t v790 = a31;	// L853
    int8_t v791 = w31;	// L854
    int16_t v792 = v790;	// L855
    int16_t v793 = v791;	// L856
    int16_t v794 = v792 * v793;	// L857
    ap_int<33> v795 = v789;	// L858
    ap_int<33> v796 = v794;	// L859
    ap_int<33> v797 = v795 + v796;	// L860
    int32_t v798 = v797;	// L861
    v778.write(v798); // v778[m31] = v798;	// L862
    int8_t v799 = a31;	// L863
    v777.write(v799); // v777[m31] = v799;	// L864
  }
}

void PE_kernel_ws_gemm_0_4(
  hls::stream< int8_t > &v800 /* v800[8] */,
  hls::stream< int32_t > &v801 /* v801[8] */,
  hls::stream< int8_t > &v802 /* v802[8] */,
  hls::stream< int32_t > &v803 /* v803[8] */,
  int8_t v804[8][8],
  int v805,
  int v806
) {	// L868
  #pragma HLS stream variable=v800 depth=9
  #pragma HLS stream variable=v801 depth=9
  #pragma HLS stream variable=v802 depth=9
  #pragma HLS stream variable=v803 depth=9
  #pragma HLS array_partition variable=v804 complete dim=1
  #pragma HLS array_partition variable=v804 complete dim=2

  int8_t v807 = v804[v805][v806];	// L869
  int8_t w32;	// L870
  w32 = v807;	// L871
  l_step_m32: for (int m32 = 0; m32 < 8; m32++) {	// L872
  #pragma HLS pipeline II=1
    int8_t v810 = v800.read(); // v800[m32];	// L873
    int8_t a32;	// L874
    a32 = v810;	// L875
    int32_t v812 = v801.read(); // v801[m32];	// L876
    int32_t p32;	// L877
    p32 = v812;	// L878
    int32_t v814 = p32;	// L879
    int8_t v815 = a32;	// L880
    int8_t v816 = w32;	// L881
    int16_t v817 = v815;	// L882
    int16_t v818 = v816;	// L883
    int16_t v819 = v817 * v818;	// L884
    ap_int<33> v820 = v814;	// L885
    ap_int<33> v821 = v819;	// L886
    ap_int<33> v822 = v820 + v821;	// L887
    int32_t v823 = v822;	// L888
    v803.write(v823); // v803[m32] = v823;	// L889
    int8_t v824 = a32;	// L890
    v802.write(v824); // v802[m32] = v824;	// L891
  }
}

void PE_kernel_ws_gemm_1_4(
  hls::stream< int8_t > &v825 /* v825[8] */,
  hls::stream< int32_t > &v826 /* v826[8] */,
  hls::stream< int8_t > &v827 /* v827[8] */,
  hls::stream< int32_t > &v828 /* v828[8] */,
  int8_t v829[8][8],
  int v830,
  int v831
) {	// L895
  #pragma HLS stream variable=v825 depth=9
  #pragma HLS stream variable=v826 depth=9
  #pragma HLS stream variable=v827 depth=9
  #pragma HLS stream variable=v828 depth=9
  #pragma HLS array_partition variable=v829 complete dim=1
  #pragma HLS array_partition variable=v829 complete dim=2

  int8_t v832 = v829[v830][v831];	// L896
  int8_t w33;	// L897
  w33 = v832;	// L898
  l_step_m33: for (int m33 = 0; m33 < 8; m33++) {	// L899
  #pragma HLS pipeline II=1
    int8_t v835 = v825.read(); // v825[m33];	// L900
    int8_t a33;	// L901
    a33 = v835;	// L902
    int32_t v837 = v826.read(); // v826[m33];	// L903
    int32_t p33;	// L904
    p33 = v837;	// L905
    int32_t v839 = p33;	// L906
    int8_t v840 = a33;	// L907
    int8_t v841 = w33;	// L908
    int16_t v842 = v840;	// L909
    int16_t v843 = v841;	// L910
    int16_t v844 = v842 * v843;	// L911
    ap_int<33> v845 = v839;	// L912
    ap_int<33> v846 = v844;	// L913
    ap_int<33> v847 = v845 + v846;	// L914
    int32_t v848 = v847;	// L915
    v828.write(v848); // v828[m33] = v848;	// L916
    int8_t v849 = a33;	// L917
    v827.write(v849); // v827[m33] = v849;	// L918
  }
}

void PE_kernel_ws_gemm_2_4(
  hls::stream< int8_t > &v850 /* v850[8] */,
  hls::stream< int32_t > &v851 /* v851[8] */,
  hls::stream< int8_t > &v852 /* v852[8] */,
  hls::stream< int32_t > &v853 /* v853[8] */,
  int8_t v854[8][8],
  int v855,
  int v856
) {	// L922
  #pragma HLS stream variable=v850 depth=9
  #pragma HLS stream variable=v851 depth=9
  #pragma HLS stream variable=v852 depth=9
  #pragma HLS stream variable=v853 depth=9
  #pragma HLS array_partition variable=v854 complete dim=1
  #pragma HLS array_partition variable=v854 complete dim=2

  int8_t v857 = v854[v855][v856];	// L923
  int8_t w34;	// L924
  w34 = v857;	// L925
  l_step_m34: for (int m34 = 0; m34 < 8; m34++) {	// L926
  #pragma HLS pipeline II=1
    int8_t v860 = v850.read(); // v850[m34];	// L927
    int8_t a34;	// L928
    a34 = v860;	// L929
    int32_t v862 = v851.read(); // v851[m34];	// L930
    int32_t p34;	// L931
    p34 = v862;	// L932
    int32_t v864 = p34;	// L933
    int8_t v865 = a34;	// L934
    int8_t v866 = w34;	// L935
    int16_t v867 = v865;	// L936
    int16_t v868 = v866;	// L937
    int16_t v869 = v867 * v868;	// L938
    ap_int<33> v870 = v864;	// L939
    ap_int<33> v871 = v869;	// L940
    ap_int<33> v872 = v870 + v871;	// L941
    int32_t v873 = v872;	// L942
    v853.write(v873); // v853[m34] = v873;	// L943
    int8_t v874 = a34;	// L944
    v852.write(v874); // v852[m34] = v874;	// L945
  }
}

void PE_kernel_ws_gemm_3_4(
  hls::stream< int8_t > &v875 /* v875[8] */,
  hls::stream< int32_t > &v876 /* v876[8] */,
  hls::stream< int8_t > &v877 /* v877[8] */,
  hls::stream< int32_t > &v878 /* v878[8] */,
  int8_t v879[8][8],
  int v880,
  int v881
) {	// L949
  #pragma HLS stream variable=v875 depth=9
  #pragma HLS stream variable=v876 depth=9
  #pragma HLS stream variable=v877 depth=9
  #pragma HLS stream variable=v878 depth=9
  #pragma HLS array_partition variable=v879 complete dim=1
  #pragma HLS array_partition variable=v879 complete dim=2

  int8_t v882 = v879[v880][v881];	// L950
  int8_t w35;	// L951
  w35 = v882;	// L952
  l_step_m35: for (int m35 = 0; m35 < 8; m35++) {	// L953
  #pragma HLS pipeline II=1
    int8_t v885 = v875.read(); // v875[m35];	// L954
    int8_t a35;	// L955
    a35 = v885;	// L956
    int32_t v887 = v876.read(); // v876[m35];	// L957
    int32_t p35;	// L958
    p35 = v887;	// L959
    int32_t v889 = p35;	// L960
    int8_t v890 = a35;	// L961
    int8_t v891 = w35;	// L962
    int16_t v892 = v890;	// L963
    int16_t v893 = v891;	// L964
    int16_t v894 = v892 * v893;	// L965
    ap_int<33> v895 = v889;	// L966
    ap_int<33> v896 = v894;	// L967
    ap_int<33> v897 = v895 + v896;	// L968
    int32_t v898 = v897;	// L969
    v878.write(v898); // v878[m35] = v898;	// L970
    int8_t v899 = a35;	// L971
    v877.write(v899); // v877[m35] = v899;	// L972
  }
}

void PE_kernel_ws_gemm_4_4(
  hls::stream< int8_t > &v900 /* v900[8] */,
  hls::stream< int32_t > &v901 /* v901[8] */,
  hls::stream< int8_t > &v902 /* v902[8] */,
  hls::stream< int32_t > &v903 /* v903[8] */,
  int8_t v904[8][8],
  int v905,
  int v906
) {	// L976
  #pragma HLS stream variable=v900 depth=9
  #pragma HLS stream variable=v901 depth=9
  #pragma HLS stream variable=v902 depth=9
  #pragma HLS stream variable=v903 depth=9
  #pragma HLS array_partition variable=v904 complete dim=1
  #pragma HLS array_partition variable=v904 complete dim=2

  int8_t v907 = v904[v905][v906];	// L977
  int8_t w36;	// L978
  w36 = v907;	// L979
  l_step_m36: for (int m36 = 0; m36 < 8; m36++) {	// L980
  #pragma HLS pipeline II=1
    int8_t v910 = v900.read(); // v900[m36];	// L981
    int8_t a36;	// L982
    a36 = v910;	// L983
    int32_t v912 = v901.read(); // v901[m36];	// L984
    int32_t p36;	// L985
    p36 = v912;	// L986
    int32_t v914 = p36;	// L987
    int8_t v915 = a36;	// L988
    int8_t v916 = w36;	// L989
    int16_t v917 = v915;	// L990
    int16_t v918 = v916;	// L991
    int16_t v919 = v917 * v918;	// L992
    ap_int<33> v920 = v914;	// L993
    ap_int<33> v921 = v919;	// L994
    ap_int<33> v922 = v920 + v921;	// L995
    int32_t v923 = v922;	// L996
    v903.write(v923); // v903[m36] = v923;	// L997
    int8_t v924 = a36;	// L998
    v902.write(v924); // v902[m36] = v924;	// L999
  }
}

void PE_kernel_ws_gemm_5_4(
  hls::stream< int8_t > &v925 /* v925[8] */,
  hls::stream< int32_t > &v926 /* v926[8] */,
  hls::stream< int8_t > &v927 /* v927[8] */,
  hls::stream< int32_t > &v928 /* v928[8] */,
  int8_t v929[8][8],
  int v930,
  int v931
) {	// L1003
  #pragma HLS stream variable=v925 depth=9
  #pragma HLS stream variable=v926 depth=9
  #pragma HLS stream variable=v927 depth=9
  #pragma HLS stream variable=v928 depth=9
  #pragma HLS array_partition variable=v929 complete dim=1
  #pragma HLS array_partition variable=v929 complete dim=2

  int8_t v932 = v929[v930][v931];	// L1004
  int8_t w37;	// L1005
  w37 = v932;	// L1006
  l_step_m37: for (int m37 = 0; m37 < 8; m37++) {	// L1007
  #pragma HLS pipeline II=1
    int8_t v935 = v925.read(); // v925[m37];	// L1008
    int8_t a37;	// L1009
    a37 = v935;	// L1010
    int32_t v937 = v926.read(); // v926[m37];	// L1011
    int32_t p37;	// L1012
    p37 = v937;	// L1013
    int32_t v939 = p37;	// L1014
    int8_t v940 = a37;	// L1015
    int8_t v941 = w37;	// L1016
    int16_t v942 = v940;	// L1017
    int16_t v943 = v941;	// L1018
    int16_t v944 = v942 * v943;	// L1019
    ap_int<33> v945 = v939;	// L1020
    ap_int<33> v946 = v944;	// L1021
    ap_int<33> v947 = v945 + v946;	// L1022
    int32_t v948 = v947;	// L1023
    v928.write(v948); // v928[m37] = v948;	// L1024
    int8_t v949 = a37;	// L1025
    v927.write(v949); // v927[m37] = v949;	// L1026
  }
}

void PE_kernel_ws_gemm_6_4(
  hls::stream< int8_t > &v950 /* v950[8] */,
  hls::stream< int32_t > &v951 /* v951[8] */,
  hls::stream< int8_t > &v952 /* v952[8] */,
  hls::stream< int32_t > &v953 /* v953[8] */,
  int8_t v954[8][8],
  int v955,
  int v956
) {	// L1030
  #pragma HLS stream variable=v950 depth=9
  #pragma HLS stream variable=v951 depth=9
  #pragma HLS stream variable=v952 depth=9
  #pragma HLS stream variable=v953 depth=9
  #pragma HLS array_partition variable=v954 complete dim=1
  #pragma HLS array_partition variable=v954 complete dim=2

  int8_t v957 = v954[v955][v956];	// L1031
  int8_t w38;	// L1032
  w38 = v957;	// L1033
  l_step_m38: for (int m38 = 0; m38 < 8; m38++) {	// L1034
  #pragma HLS pipeline II=1
    int8_t v960 = v950.read(); // v950[m38];	// L1035
    int8_t a38;	// L1036
    a38 = v960;	// L1037
    int32_t v962 = v951.read(); // v951[m38];	// L1038
    int32_t p38;	// L1039
    p38 = v962;	// L1040
    int32_t v964 = p38;	// L1041
    int8_t v965 = a38;	// L1042
    int8_t v966 = w38;	// L1043
    int16_t v967 = v965;	// L1044
    int16_t v968 = v966;	// L1045
    int16_t v969 = v967 * v968;	// L1046
    ap_int<33> v970 = v964;	// L1047
    ap_int<33> v971 = v969;	// L1048
    ap_int<33> v972 = v970 + v971;	// L1049
    int32_t v973 = v972;	// L1050
    v953.write(v973); // v953[m38] = v973;	// L1051
    int8_t v974 = a38;	// L1052
    v952.write(v974); // v952[m38] = v974;	// L1053
  }
}

void PE_kernel_ws_gemm_7_4(
  hls::stream< int8_t > &v975 /* v975[8] */,
  hls::stream< int32_t > &v976 /* v976[8] */,
  hls::stream< int8_t > &v977 /* v977[8] */,
  hls::stream< int32_t > &v978 /* v978[8] */,
  int8_t v979[8][8],
  int v980,
  int v981
) {	// L1057
  #pragma HLS stream variable=v975 depth=9
  #pragma HLS stream variable=v976 depth=9
  #pragma HLS stream variable=v977 depth=9
  #pragma HLS stream variable=v978 depth=9
  #pragma HLS array_partition variable=v979 complete dim=1
  #pragma HLS array_partition variable=v979 complete dim=2

  int8_t v982 = v979[v980][v981];	// L1058
  int8_t w39;	// L1059
  w39 = v982;	// L1060
  l_step_m39: for (int m39 = 0; m39 < 8; m39++) {	// L1061
  #pragma HLS pipeline II=1
    int8_t v985 = v975.read(); // v975[m39];	// L1062
    int8_t a39;	// L1063
    a39 = v985;	// L1064
    int32_t v987 = v976.read(); // v976[m39];	// L1065
    int32_t p39;	// L1066
    p39 = v987;	// L1067
    int32_t v989 = p39;	// L1068
    int8_t v990 = a39;	// L1069
    int8_t v991 = w39;	// L1070
    int16_t v992 = v990;	// L1071
    int16_t v993 = v991;	// L1072
    int16_t v994 = v992 * v993;	// L1073
    ap_int<33> v995 = v989;	// L1074
    ap_int<33> v996 = v994;	// L1075
    ap_int<33> v997 = v995 + v996;	// L1076
    int32_t v998 = v997;	// L1077
    v978.write(v998); // v978[m39] = v998;	// L1078
    int8_t v999 = a39;	// L1079
    v977.write(v999); // v977[m39] = v999;	// L1080
  }
}

void PE_kernel_ws_gemm_0_5(
  hls::stream< int8_t > &v1000 /* v1000[8] */,
  hls::stream< int32_t > &v1001 /* v1001[8] */,
  hls::stream< int8_t > &v1002 /* v1002[8] */,
  hls::stream< int32_t > &v1003 /* v1003[8] */,
  int8_t v1004[8][8],
  int v1005,
  int v1006
) {	// L1084
  #pragma HLS stream variable=v1000 depth=9
  #pragma HLS stream variable=v1001 depth=9
  #pragma HLS stream variable=v1002 depth=9
  #pragma HLS stream variable=v1003 depth=9
  #pragma HLS array_partition variable=v1004 complete dim=1
  #pragma HLS array_partition variable=v1004 complete dim=2

  int8_t v1007 = v1004[v1005][v1006];	// L1085
  int8_t w40;	// L1086
  w40 = v1007;	// L1087
  l_step_m40: for (int m40 = 0; m40 < 8; m40++) {	// L1088
  #pragma HLS pipeline II=1
    int8_t v1010 = v1000.read(); // v1000[m40];	// L1089
    int8_t a40;	// L1090
    a40 = v1010;	// L1091
    int32_t v1012 = v1001.read(); // v1001[m40];	// L1092
    int32_t p40;	// L1093
    p40 = v1012;	// L1094
    int32_t v1014 = p40;	// L1095
    int8_t v1015 = a40;	// L1096
    int8_t v1016 = w40;	// L1097
    int16_t v1017 = v1015;	// L1098
    int16_t v1018 = v1016;	// L1099
    int16_t v1019 = v1017 * v1018;	// L1100
    ap_int<33> v1020 = v1014;	// L1101
    ap_int<33> v1021 = v1019;	// L1102
    ap_int<33> v1022 = v1020 + v1021;	// L1103
    int32_t v1023 = v1022;	// L1104
    v1003.write(v1023); // v1003[m40] = v1023;	// L1105
    int8_t v1024 = a40;	// L1106
    v1002.write(v1024); // v1002[m40] = v1024;	// L1107
  }
}

void PE_kernel_ws_gemm_1_5(
  hls::stream< int8_t > &v1025 /* v1025[8] */,
  hls::stream< int32_t > &v1026 /* v1026[8] */,
  hls::stream< int8_t > &v1027 /* v1027[8] */,
  hls::stream< int32_t > &v1028 /* v1028[8] */,
  int8_t v1029[8][8],
  int v1030,
  int v1031
) {	// L1111
  #pragma HLS stream variable=v1025 depth=9
  #pragma HLS stream variable=v1026 depth=9
  #pragma HLS stream variable=v1027 depth=9
  #pragma HLS stream variable=v1028 depth=9
  #pragma HLS array_partition variable=v1029 complete dim=1
  #pragma HLS array_partition variable=v1029 complete dim=2

  int8_t v1032 = v1029[v1030][v1031];	// L1112
  int8_t w41;	// L1113
  w41 = v1032;	// L1114
  l_step_m41: for (int m41 = 0; m41 < 8; m41++) {	// L1115
  #pragma HLS pipeline II=1
    int8_t v1035 = v1025.read(); // v1025[m41];	// L1116
    int8_t a41;	// L1117
    a41 = v1035;	// L1118
    int32_t v1037 = v1026.read(); // v1026[m41];	// L1119
    int32_t p41;	// L1120
    p41 = v1037;	// L1121
    int32_t v1039 = p41;	// L1122
    int8_t v1040 = a41;	// L1123
    int8_t v1041 = w41;	// L1124
    int16_t v1042 = v1040;	// L1125
    int16_t v1043 = v1041;	// L1126
    int16_t v1044 = v1042 * v1043;	// L1127
    ap_int<33> v1045 = v1039;	// L1128
    ap_int<33> v1046 = v1044;	// L1129
    ap_int<33> v1047 = v1045 + v1046;	// L1130
    int32_t v1048 = v1047;	// L1131
    v1028.write(v1048); // v1028[m41] = v1048;	// L1132
    int8_t v1049 = a41;	// L1133
    v1027.write(v1049); // v1027[m41] = v1049;	// L1134
  }
}

void PE_kernel_ws_gemm_2_5(
  hls::stream< int8_t > &v1050 /* v1050[8] */,
  hls::stream< int32_t > &v1051 /* v1051[8] */,
  hls::stream< int8_t > &v1052 /* v1052[8] */,
  hls::stream< int32_t > &v1053 /* v1053[8] */,
  int8_t v1054[8][8],
  int v1055,
  int v1056
) {	// L1138
  #pragma HLS stream variable=v1050 depth=9
  #pragma HLS stream variable=v1051 depth=9
  #pragma HLS stream variable=v1052 depth=9
  #pragma HLS stream variable=v1053 depth=9
  #pragma HLS array_partition variable=v1054 complete dim=1
  #pragma HLS array_partition variable=v1054 complete dim=2

  int8_t v1057 = v1054[v1055][v1056];	// L1139
  int8_t w42;	// L1140
  w42 = v1057;	// L1141
  l_step_m42: for (int m42 = 0; m42 < 8; m42++) {	// L1142
  #pragma HLS pipeline II=1
    int8_t v1060 = v1050.read(); // v1050[m42];	// L1143
    int8_t a42;	// L1144
    a42 = v1060;	// L1145
    int32_t v1062 = v1051.read(); // v1051[m42];	// L1146
    int32_t p42;	// L1147
    p42 = v1062;	// L1148
    int32_t v1064 = p42;	// L1149
    int8_t v1065 = a42;	// L1150
    int8_t v1066 = w42;	// L1151
    int16_t v1067 = v1065;	// L1152
    int16_t v1068 = v1066;	// L1153
    int16_t v1069 = v1067 * v1068;	// L1154
    ap_int<33> v1070 = v1064;	// L1155
    ap_int<33> v1071 = v1069;	// L1156
    ap_int<33> v1072 = v1070 + v1071;	// L1157
    int32_t v1073 = v1072;	// L1158
    v1053.write(v1073); // v1053[m42] = v1073;	// L1159
    int8_t v1074 = a42;	// L1160
    v1052.write(v1074); // v1052[m42] = v1074;	// L1161
  }
}

void PE_kernel_ws_gemm_3_5(
  hls::stream< int8_t > &v1075 /* v1075[8] */,
  hls::stream< int32_t > &v1076 /* v1076[8] */,
  hls::stream< int8_t > &v1077 /* v1077[8] */,
  hls::stream< int32_t > &v1078 /* v1078[8] */,
  int8_t v1079[8][8],
  int v1080,
  int v1081
) {	// L1165
  #pragma HLS stream variable=v1075 depth=9
  #pragma HLS stream variable=v1076 depth=9
  #pragma HLS stream variable=v1077 depth=9
  #pragma HLS stream variable=v1078 depth=9
  #pragma HLS array_partition variable=v1079 complete dim=1
  #pragma HLS array_partition variable=v1079 complete dim=2

  int8_t v1082 = v1079[v1080][v1081];	// L1166
  int8_t w43;	// L1167
  w43 = v1082;	// L1168
  l_step_m43: for (int m43 = 0; m43 < 8; m43++) {	// L1169
  #pragma HLS pipeline II=1
    int8_t v1085 = v1075.read(); // v1075[m43];	// L1170
    int8_t a43;	// L1171
    a43 = v1085;	// L1172
    int32_t v1087 = v1076.read(); // v1076[m43];	// L1173
    int32_t p43;	// L1174
    p43 = v1087;	// L1175
    int32_t v1089 = p43;	// L1176
    int8_t v1090 = a43;	// L1177
    int8_t v1091 = w43;	// L1178
    int16_t v1092 = v1090;	// L1179
    int16_t v1093 = v1091;	// L1180
    int16_t v1094 = v1092 * v1093;	// L1181
    ap_int<33> v1095 = v1089;	// L1182
    ap_int<33> v1096 = v1094;	// L1183
    ap_int<33> v1097 = v1095 + v1096;	// L1184
    int32_t v1098 = v1097;	// L1185
    v1078.write(v1098); // v1078[m43] = v1098;	// L1186
    int8_t v1099 = a43;	// L1187
    v1077.write(v1099); // v1077[m43] = v1099;	// L1188
  }
}

void PE_kernel_ws_gemm_4_5(
  hls::stream< int8_t > &v1100 /* v1100[8] */,
  hls::stream< int32_t > &v1101 /* v1101[8] */,
  hls::stream< int8_t > &v1102 /* v1102[8] */,
  hls::stream< int32_t > &v1103 /* v1103[8] */,
  int8_t v1104[8][8],
  int v1105,
  int v1106
) {	// L1192
  #pragma HLS stream variable=v1100 depth=9
  #pragma HLS stream variable=v1101 depth=9
  #pragma HLS stream variable=v1102 depth=9
  #pragma HLS stream variable=v1103 depth=9
  #pragma HLS array_partition variable=v1104 complete dim=1
  #pragma HLS array_partition variable=v1104 complete dim=2

  int8_t v1107 = v1104[v1105][v1106];	// L1193
  int8_t w44;	// L1194
  w44 = v1107;	// L1195
  l_step_m44: for (int m44 = 0; m44 < 8; m44++) {	// L1196
  #pragma HLS pipeline II=1
    int8_t v1110 = v1100.read(); // v1100[m44];	// L1197
    int8_t a44;	// L1198
    a44 = v1110;	// L1199
    int32_t v1112 = v1101.read(); // v1101[m44];	// L1200
    int32_t p44;	// L1201
    p44 = v1112;	// L1202
    int32_t v1114 = p44;	// L1203
    int8_t v1115 = a44;	// L1204
    int8_t v1116 = w44;	// L1205
    int16_t v1117 = v1115;	// L1206
    int16_t v1118 = v1116;	// L1207
    int16_t v1119 = v1117 * v1118;	// L1208
    ap_int<33> v1120 = v1114;	// L1209
    ap_int<33> v1121 = v1119;	// L1210
    ap_int<33> v1122 = v1120 + v1121;	// L1211
    int32_t v1123 = v1122;	// L1212
    v1103.write(v1123); // v1103[m44] = v1123;	// L1213
    int8_t v1124 = a44;	// L1214
    v1102.write(v1124); // v1102[m44] = v1124;	// L1215
  }
}

void PE_kernel_ws_gemm_5_5(
  hls::stream< int8_t > &v1125 /* v1125[8] */,
  hls::stream< int32_t > &v1126 /* v1126[8] */,
  hls::stream< int8_t > &v1127 /* v1127[8] */,
  hls::stream< int32_t > &v1128 /* v1128[8] */,
  int8_t v1129[8][8],
  int v1130,
  int v1131
) {	// L1219
  #pragma HLS stream variable=v1125 depth=9
  #pragma HLS stream variable=v1126 depth=9
  #pragma HLS stream variable=v1127 depth=9
  #pragma HLS stream variable=v1128 depth=9
  #pragma HLS array_partition variable=v1129 complete dim=1
  #pragma HLS array_partition variable=v1129 complete dim=2

  int8_t v1132 = v1129[v1130][v1131];	// L1220
  int8_t w45;	// L1221
  w45 = v1132;	// L1222
  l_step_m45: for (int m45 = 0; m45 < 8; m45++) {	// L1223
  #pragma HLS pipeline II=1
    int8_t v1135 = v1125.read(); // v1125[m45];	// L1224
    int8_t a45;	// L1225
    a45 = v1135;	// L1226
    int32_t v1137 = v1126.read(); // v1126[m45];	// L1227
    int32_t p45;	// L1228
    p45 = v1137;	// L1229
    int32_t v1139 = p45;	// L1230
    int8_t v1140 = a45;	// L1231
    int8_t v1141 = w45;	// L1232
    int16_t v1142 = v1140;	// L1233
    int16_t v1143 = v1141;	// L1234
    int16_t v1144 = v1142 * v1143;	// L1235
    ap_int<33> v1145 = v1139;	// L1236
    ap_int<33> v1146 = v1144;	// L1237
    ap_int<33> v1147 = v1145 + v1146;	// L1238
    int32_t v1148 = v1147;	// L1239
    v1128.write(v1148); // v1128[m45] = v1148;	// L1240
    int8_t v1149 = a45;	// L1241
    v1127.write(v1149); // v1127[m45] = v1149;	// L1242
  }
}

void PE_kernel_ws_gemm_6_5(
  hls::stream< int8_t > &v1150 /* v1150[8] */,
  hls::stream< int32_t > &v1151 /* v1151[8] */,
  hls::stream< int8_t > &v1152 /* v1152[8] */,
  hls::stream< int32_t > &v1153 /* v1153[8] */,
  int8_t v1154[8][8],
  int v1155,
  int v1156
) {	// L1246
  #pragma HLS stream variable=v1150 depth=9
  #pragma HLS stream variable=v1151 depth=9
  #pragma HLS stream variable=v1152 depth=9
  #pragma HLS stream variable=v1153 depth=9
  #pragma HLS array_partition variable=v1154 complete dim=1
  #pragma HLS array_partition variable=v1154 complete dim=2

  int8_t v1157 = v1154[v1155][v1156];	// L1247
  int8_t w46;	// L1248
  w46 = v1157;	// L1249
  l_step_m46: for (int m46 = 0; m46 < 8; m46++) {	// L1250
  #pragma HLS pipeline II=1
    int8_t v1160 = v1150.read(); // v1150[m46];	// L1251
    int8_t a46;	// L1252
    a46 = v1160;	// L1253
    int32_t v1162 = v1151.read(); // v1151[m46];	// L1254
    int32_t p46;	// L1255
    p46 = v1162;	// L1256
    int32_t v1164 = p46;	// L1257
    int8_t v1165 = a46;	// L1258
    int8_t v1166 = w46;	// L1259
    int16_t v1167 = v1165;	// L1260
    int16_t v1168 = v1166;	// L1261
    int16_t v1169 = v1167 * v1168;	// L1262
    ap_int<33> v1170 = v1164;	// L1263
    ap_int<33> v1171 = v1169;	// L1264
    ap_int<33> v1172 = v1170 + v1171;	// L1265
    int32_t v1173 = v1172;	// L1266
    v1153.write(v1173); // v1153[m46] = v1173;	// L1267
    int8_t v1174 = a46;	// L1268
    v1152.write(v1174); // v1152[m46] = v1174;	// L1269
  }
}

void PE_kernel_ws_gemm_7_5(
  hls::stream< int8_t > &v1175 /* v1175[8] */,
  hls::stream< int32_t > &v1176 /* v1176[8] */,
  hls::stream< int8_t > &v1177 /* v1177[8] */,
  hls::stream< int32_t > &v1178 /* v1178[8] */,
  int8_t v1179[8][8],
  int v1180,
  int v1181
) {	// L1273
  #pragma HLS stream variable=v1175 depth=9
  #pragma HLS stream variable=v1176 depth=9
  #pragma HLS stream variable=v1177 depth=9
  #pragma HLS stream variable=v1178 depth=9
  #pragma HLS array_partition variable=v1179 complete dim=1
  #pragma HLS array_partition variable=v1179 complete dim=2

  int8_t v1182 = v1179[v1180][v1181];	// L1274
  int8_t w47;	// L1275
  w47 = v1182;	// L1276
  l_step_m47: for (int m47 = 0; m47 < 8; m47++) {	// L1277
  #pragma HLS pipeline II=1
    int8_t v1185 = v1175.read(); // v1175[m47];	// L1278
    int8_t a47;	// L1279
    a47 = v1185;	// L1280
    int32_t v1187 = v1176.read(); // v1176[m47];	// L1281
    int32_t p47;	// L1282
    p47 = v1187;	// L1283
    int32_t v1189 = p47;	// L1284
    int8_t v1190 = a47;	// L1285
    int8_t v1191 = w47;	// L1286
    int16_t v1192 = v1190;	// L1287
    int16_t v1193 = v1191;	// L1288
    int16_t v1194 = v1192 * v1193;	// L1289
    ap_int<33> v1195 = v1189;	// L1290
    ap_int<33> v1196 = v1194;	// L1291
    ap_int<33> v1197 = v1195 + v1196;	// L1292
    int32_t v1198 = v1197;	// L1293
    v1178.write(v1198); // v1178[m47] = v1198;	// L1294
    int8_t v1199 = a47;	// L1295
    v1177.write(v1199); // v1177[m47] = v1199;	// L1296
  }
}

void PE_kernel_ws_gemm_0_6(
  hls::stream< int8_t > &v1200 /* v1200[8] */,
  hls::stream< int32_t > &v1201 /* v1201[8] */,
  hls::stream< int8_t > &v1202 /* v1202[8] */,
  hls::stream< int32_t > &v1203 /* v1203[8] */,
  int8_t v1204[8][8],
  int v1205,
  int v1206
) {	// L1300
  #pragma HLS stream variable=v1200 depth=9
  #pragma HLS stream variable=v1201 depth=9
  #pragma HLS stream variable=v1202 depth=9
  #pragma HLS stream variable=v1203 depth=9
  #pragma HLS array_partition variable=v1204 complete dim=1
  #pragma HLS array_partition variable=v1204 complete dim=2

  int8_t v1207 = v1204[v1205][v1206];	// L1301
  int8_t w48;	// L1302
  w48 = v1207;	// L1303
  l_step_m48: for (int m48 = 0; m48 < 8; m48++) {	// L1304
  #pragma HLS pipeline II=1
    int8_t v1210 = v1200.read(); // v1200[m48];	// L1305
    int8_t a48;	// L1306
    a48 = v1210;	// L1307
    int32_t v1212 = v1201.read(); // v1201[m48];	// L1308
    int32_t p48;	// L1309
    p48 = v1212;	// L1310
    int32_t v1214 = p48;	// L1311
    int8_t v1215 = a48;	// L1312
    int8_t v1216 = w48;	// L1313
    int16_t v1217 = v1215;	// L1314
    int16_t v1218 = v1216;	// L1315
    int16_t v1219 = v1217 * v1218;	// L1316
    ap_int<33> v1220 = v1214;	// L1317
    ap_int<33> v1221 = v1219;	// L1318
    ap_int<33> v1222 = v1220 + v1221;	// L1319
    int32_t v1223 = v1222;	// L1320
    v1203.write(v1223); // v1203[m48] = v1223;	// L1321
    int8_t v1224 = a48;	// L1322
    v1202.write(v1224); // v1202[m48] = v1224;	// L1323
  }
}

void PE_kernel_ws_gemm_1_6(
  hls::stream< int8_t > &v1225 /* v1225[8] */,
  hls::stream< int32_t > &v1226 /* v1226[8] */,
  hls::stream< int8_t > &v1227 /* v1227[8] */,
  hls::stream< int32_t > &v1228 /* v1228[8] */,
  int8_t v1229[8][8],
  int v1230,
  int v1231
) {	// L1327
  #pragma HLS stream variable=v1225 depth=9
  #pragma HLS stream variable=v1226 depth=9
  #pragma HLS stream variable=v1227 depth=9
  #pragma HLS stream variable=v1228 depth=9
  #pragma HLS array_partition variable=v1229 complete dim=1
  #pragma HLS array_partition variable=v1229 complete dim=2

  int8_t v1232 = v1229[v1230][v1231];	// L1328
  int8_t w49;	// L1329
  w49 = v1232;	// L1330
  l_step_m49: for (int m49 = 0; m49 < 8; m49++) {	// L1331
  #pragma HLS pipeline II=1
    int8_t v1235 = v1225.read(); // v1225[m49];	// L1332
    int8_t a49;	// L1333
    a49 = v1235;	// L1334
    int32_t v1237 = v1226.read(); // v1226[m49];	// L1335
    int32_t p49;	// L1336
    p49 = v1237;	// L1337
    int32_t v1239 = p49;	// L1338
    int8_t v1240 = a49;	// L1339
    int8_t v1241 = w49;	// L1340
    int16_t v1242 = v1240;	// L1341
    int16_t v1243 = v1241;	// L1342
    int16_t v1244 = v1242 * v1243;	// L1343
    ap_int<33> v1245 = v1239;	// L1344
    ap_int<33> v1246 = v1244;	// L1345
    ap_int<33> v1247 = v1245 + v1246;	// L1346
    int32_t v1248 = v1247;	// L1347
    v1228.write(v1248); // v1228[m49] = v1248;	// L1348
    int8_t v1249 = a49;	// L1349
    v1227.write(v1249); // v1227[m49] = v1249;	// L1350
  }
}

void PE_kernel_ws_gemm_2_6(
  hls::stream< int8_t > &v1250 /* v1250[8] */,
  hls::stream< int32_t > &v1251 /* v1251[8] */,
  hls::stream< int8_t > &v1252 /* v1252[8] */,
  hls::stream< int32_t > &v1253 /* v1253[8] */,
  int8_t v1254[8][8],
  int v1255,
  int v1256
) {	// L1354
  #pragma HLS stream variable=v1250 depth=9
  #pragma HLS stream variable=v1251 depth=9
  #pragma HLS stream variable=v1252 depth=9
  #pragma HLS stream variable=v1253 depth=9
  #pragma HLS array_partition variable=v1254 complete dim=1
  #pragma HLS array_partition variable=v1254 complete dim=2

  int8_t v1257 = v1254[v1255][v1256];	// L1355
  int8_t w50;	// L1356
  w50 = v1257;	// L1357
  l_step_m50: for (int m50 = 0; m50 < 8; m50++) {	// L1358
  #pragma HLS pipeline II=1
    int8_t v1260 = v1250.read(); // v1250[m50];	// L1359
    int8_t a50;	// L1360
    a50 = v1260;	// L1361
    int32_t v1262 = v1251.read(); // v1251[m50];	// L1362
    int32_t p50;	// L1363
    p50 = v1262;	// L1364
    int32_t v1264 = p50;	// L1365
    int8_t v1265 = a50;	// L1366
    int8_t v1266 = w50;	// L1367
    int16_t v1267 = v1265;	// L1368
    int16_t v1268 = v1266;	// L1369
    int16_t v1269 = v1267 * v1268;	// L1370
    ap_int<33> v1270 = v1264;	// L1371
    ap_int<33> v1271 = v1269;	// L1372
    ap_int<33> v1272 = v1270 + v1271;	// L1373
    int32_t v1273 = v1272;	// L1374
    v1253.write(v1273); // v1253[m50] = v1273;	// L1375
    int8_t v1274 = a50;	// L1376
    v1252.write(v1274); // v1252[m50] = v1274;	// L1377
  }
}

void PE_kernel_ws_gemm_3_6(
  hls::stream< int8_t > &v1275 /* v1275[8] */,
  hls::stream< int32_t > &v1276 /* v1276[8] */,
  hls::stream< int8_t > &v1277 /* v1277[8] */,
  hls::stream< int32_t > &v1278 /* v1278[8] */,
  int8_t v1279[8][8],
  int v1280,
  int v1281
) {	// L1381
  #pragma HLS stream variable=v1275 depth=9
  #pragma HLS stream variable=v1276 depth=9
  #pragma HLS stream variable=v1277 depth=9
  #pragma HLS stream variable=v1278 depth=9
  #pragma HLS array_partition variable=v1279 complete dim=1
  #pragma HLS array_partition variable=v1279 complete dim=2

  int8_t v1282 = v1279[v1280][v1281];	// L1382
  int8_t w51;	// L1383
  w51 = v1282;	// L1384
  l_step_m51: for (int m51 = 0; m51 < 8; m51++) {	// L1385
  #pragma HLS pipeline II=1
    int8_t v1285 = v1275.read(); // v1275[m51];	// L1386
    int8_t a51;	// L1387
    a51 = v1285;	// L1388
    int32_t v1287 = v1276.read(); // v1276[m51];	// L1389
    int32_t p51;	// L1390
    p51 = v1287;	// L1391
    int32_t v1289 = p51;	// L1392
    int8_t v1290 = a51;	// L1393
    int8_t v1291 = w51;	// L1394
    int16_t v1292 = v1290;	// L1395
    int16_t v1293 = v1291;	// L1396
    int16_t v1294 = v1292 * v1293;	// L1397
    ap_int<33> v1295 = v1289;	// L1398
    ap_int<33> v1296 = v1294;	// L1399
    ap_int<33> v1297 = v1295 + v1296;	// L1400
    int32_t v1298 = v1297;	// L1401
    v1278.write(v1298); // v1278[m51] = v1298;	// L1402
    int8_t v1299 = a51;	// L1403
    v1277.write(v1299); // v1277[m51] = v1299;	// L1404
  }
}

void PE_kernel_ws_gemm_4_6(
  hls::stream< int8_t > &v1300 /* v1300[8] */,
  hls::stream< int32_t > &v1301 /* v1301[8] */,
  hls::stream< int8_t > &v1302 /* v1302[8] */,
  hls::stream< int32_t > &v1303 /* v1303[8] */,
  int8_t v1304[8][8],
  int v1305,
  int v1306
) {	// L1408
  #pragma HLS stream variable=v1300 depth=9
  #pragma HLS stream variable=v1301 depth=9
  #pragma HLS stream variable=v1302 depth=9
  #pragma HLS stream variable=v1303 depth=9
  #pragma HLS array_partition variable=v1304 complete dim=1
  #pragma HLS array_partition variable=v1304 complete dim=2

  int8_t v1307 = v1304[v1305][v1306];	// L1409
  int8_t w52;	// L1410
  w52 = v1307;	// L1411
  l_step_m52: for (int m52 = 0; m52 < 8; m52++) {	// L1412
  #pragma HLS pipeline II=1
    int8_t v1310 = v1300.read(); // v1300[m52];	// L1413
    int8_t a52;	// L1414
    a52 = v1310;	// L1415
    int32_t v1312 = v1301.read(); // v1301[m52];	// L1416
    int32_t p52;	// L1417
    p52 = v1312;	// L1418
    int32_t v1314 = p52;	// L1419
    int8_t v1315 = a52;	// L1420
    int8_t v1316 = w52;	// L1421
    int16_t v1317 = v1315;	// L1422
    int16_t v1318 = v1316;	// L1423
    int16_t v1319 = v1317 * v1318;	// L1424
    ap_int<33> v1320 = v1314;	// L1425
    ap_int<33> v1321 = v1319;	// L1426
    ap_int<33> v1322 = v1320 + v1321;	// L1427
    int32_t v1323 = v1322;	// L1428
    v1303.write(v1323); // v1303[m52] = v1323;	// L1429
    int8_t v1324 = a52;	// L1430
    v1302.write(v1324); // v1302[m52] = v1324;	// L1431
  }
}

void PE_kernel_ws_gemm_5_6(
  hls::stream< int8_t > &v1325 /* v1325[8] */,
  hls::stream< int32_t > &v1326 /* v1326[8] */,
  hls::stream< int8_t > &v1327 /* v1327[8] */,
  hls::stream< int32_t > &v1328 /* v1328[8] */,
  int8_t v1329[8][8],
  int v1330,
  int v1331
) {	// L1435
  #pragma HLS stream variable=v1325 depth=9
  #pragma HLS stream variable=v1326 depth=9
  #pragma HLS stream variable=v1327 depth=9
  #pragma HLS stream variable=v1328 depth=9
  #pragma HLS array_partition variable=v1329 complete dim=1
  #pragma HLS array_partition variable=v1329 complete dim=2

  int8_t v1332 = v1329[v1330][v1331];	// L1436
  int8_t w53;	// L1437
  w53 = v1332;	// L1438
  l_step_m53: for (int m53 = 0; m53 < 8; m53++) {	// L1439
  #pragma HLS pipeline II=1
    int8_t v1335 = v1325.read(); // v1325[m53];	// L1440
    int8_t a53;	// L1441
    a53 = v1335;	// L1442
    int32_t v1337 = v1326.read(); // v1326[m53];	// L1443
    int32_t p53;	// L1444
    p53 = v1337;	// L1445
    int32_t v1339 = p53;	// L1446
    int8_t v1340 = a53;	// L1447
    int8_t v1341 = w53;	// L1448
    int16_t v1342 = v1340;	// L1449
    int16_t v1343 = v1341;	// L1450
    int16_t v1344 = v1342 * v1343;	// L1451
    ap_int<33> v1345 = v1339;	// L1452
    ap_int<33> v1346 = v1344;	// L1453
    ap_int<33> v1347 = v1345 + v1346;	// L1454
    int32_t v1348 = v1347;	// L1455
    v1328.write(v1348); // v1328[m53] = v1348;	// L1456
    int8_t v1349 = a53;	// L1457
    v1327.write(v1349); // v1327[m53] = v1349;	// L1458
  }
}

void PE_kernel_ws_gemm_6_6(
  hls::stream< int8_t > &v1350 /* v1350[8] */,
  hls::stream< int32_t > &v1351 /* v1351[8] */,
  hls::stream< int8_t > &v1352 /* v1352[8] */,
  hls::stream< int32_t > &v1353 /* v1353[8] */,
  int8_t v1354[8][8],
  int v1355,
  int v1356
) {	// L1462
  #pragma HLS stream variable=v1350 depth=9
  #pragma HLS stream variable=v1351 depth=9
  #pragma HLS stream variable=v1352 depth=9
  #pragma HLS stream variable=v1353 depth=9
  #pragma HLS array_partition variable=v1354 complete dim=1
  #pragma HLS array_partition variable=v1354 complete dim=2

  int8_t v1357 = v1354[v1355][v1356];	// L1463
  int8_t w54;	// L1464
  w54 = v1357;	// L1465
  l_step_m54: for (int m54 = 0; m54 < 8; m54++) {	// L1466
  #pragma HLS pipeline II=1
    int8_t v1360 = v1350.read(); // v1350[m54];	// L1467
    int8_t a54;	// L1468
    a54 = v1360;	// L1469
    int32_t v1362 = v1351.read(); // v1351[m54];	// L1470
    int32_t p54;	// L1471
    p54 = v1362;	// L1472
    int32_t v1364 = p54;	// L1473
    int8_t v1365 = a54;	// L1474
    int8_t v1366 = w54;	// L1475
    int16_t v1367 = v1365;	// L1476
    int16_t v1368 = v1366;	// L1477
    int16_t v1369 = v1367 * v1368;	// L1478
    ap_int<33> v1370 = v1364;	// L1479
    ap_int<33> v1371 = v1369;	// L1480
    ap_int<33> v1372 = v1370 + v1371;	// L1481
    int32_t v1373 = v1372;	// L1482
    v1353.write(v1373); // v1353[m54] = v1373;	// L1483
    int8_t v1374 = a54;	// L1484
    v1352.write(v1374); // v1352[m54] = v1374;	// L1485
  }
}

void PE_kernel_ws_gemm_7_6(
  hls::stream< int8_t > &v1375 /* v1375[8] */,
  hls::stream< int32_t > &v1376 /* v1376[8] */,
  hls::stream< int8_t > &v1377 /* v1377[8] */,
  hls::stream< int32_t > &v1378 /* v1378[8] */,
  int8_t v1379[8][8],
  int v1380,
  int v1381
) {	// L1489
  #pragma HLS stream variable=v1375 depth=9
  #pragma HLS stream variable=v1376 depth=9
  #pragma HLS stream variable=v1377 depth=9
  #pragma HLS stream variable=v1378 depth=9
  #pragma HLS array_partition variable=v1379 complete dim=1
  #pragma HLS array_partition variable=v1379 complete dim=2

  int8_t v1382 = v1379[v1380][v1381];	// L1490
  int8_t w55;	// L1491
  w55 = v1382;	// L1492
  l_step_m55: for (int m55 = 0; m55 < 8; m55++) {	// L1493
  #pragma HLS pipeline II=1
    int8_t v1385 = v1375.read(); // v1375[m55];	// L1494
    int8_t a55;	// L1495
    a55 = v1385;	// L1496
    int32_t v1387 = v1376.read(); // v1376[m55];	// L1497
    int32_t p55;	// L1498
    p55 = v1387;	// L1499
    int32_t v1389 = p55;	// L1500
    int8_t v1390 = a55;	// L1501
    int8_t v1391 = w55;	// L1502
    int16_t v1392 = v1390;	// L1503
    int16_t v1393 = v1391;	// L1504
    int16_t v1394 = v1392 * v1393;	// L1505
    ap_int<33> v1395 = v1389;	// L1506
    ap_int<33> v1396 = v1394;	// L1507
    ap_int<33> v1397 = v1395 + v1396;	// L1508
    int32_t v1398 = v1397;	// L1509
    v1378.write(v1398); // v1378[m55] = v1398;	// L1510
    int8_t v1399 = a55;	// L1511
    v1377.write(v1399); // v1377[m55] = v1399;	// L1512
  }
}

void PE_kernel_ws_gemm_0_7(
  hls::stream< int8_t > &v1400 /* v1400[8] */,
  hls::stream< int32_t > &v1401 /* v1401[8] */,
  hls::stream< int8_t > &v1402 /* v1402[8] */,
  hls::stream< int32_t > &v1403 /* v1403[8] */,
  int8_t v1404[8][8],
  int v1405,
  int v1406
) {	// L1516
  #pragma HLS stream variable=v1400 depth=9
  #pragma HLS stream variable=v1401 depth=9
  #pragma HLS stream variable=v1402 depth=9
  #pragma HLS stream variable=v1403 depth=9
  #pragma HLS array_partition variable=v1404 complete dim=1
  #pragma HLS array_partition variable=v1404 complete dim=2

  int8_t v1407 = v1404[v1405][v1406];	// L1517
  int8_t w56;	// L1518
  w56 = v1407;	// L1519
  l_step_m56: for (int m56 = 0; m56 < 8; m56++) {	// L1520
  #pragma HLS pipeline II=1
    int8_t v1410 = v1400.read(); // v1400[m56];	// L1521
    int8_t a56;	// L1522
    a56 = v1410;	// L1523
    int32_t v1412 = v1401.read(); // v1401[m56];	// L1524
    int32_t p56;	// L1525
    p56 = v1412;	// L1526
    int32_t v1414 = p56;	// L1527
    int8_t v1415 = a56;	// L1528
    int8_t v1416 = w56;	// L1529
    int16_t v1417 = v1415;	// L1530
    int16_t v1418 = v1416;	// L1531
    int16_t v1419 = v1417 * v1418;	// L1532
    ap_int<33> v1420 = v1414;	// L1533
    ap_int<33> v1421 = v1419;	// L1534
    ap_int<33> v1422 = v1420 + v1421;	// L1535
    int32_t v1423 = v1422;	// L1536
    v1403.write(v1423); // v1403[m56] = v1423;	// L1537
    int8_t v1424 = a56;	// L1538
    v1402.write(v1424); // v1402[m56] = v1424;	// L1539
  }
}

void PE_kernel_ws_gemm_1_7(
  hls::stream< int8_t > &v1425 /* v1425[8] */,
  hls::stream< int32_t > &v1426 /* v1426[8] */,
  hls::stream< int8_t > &v1427 /* v1427[8] */,
  hls::stream< int32_t > &v1428 /* v1428[8] */,
  int8_t v1429[8][8],
  int v1430,
  int v1431
) {	// L1543
  #pragma HLS stream variable=v1425 depth=9
  #pragma HLS stream variable=v1426 depth=9
  #pragma HLS stream variable=v1427 depth=9
  #pragma HLS stream variable=v1428 depth=9
  #pragma HLS array_partition variable=v1429 complete dim=1
  #pragma HLS array_partition variable=v1429 complete dim=2

  int8_t v1432 = v1429[v1430][v1431];	// L1544
  int8_t w57;	// L1545
  w57 = v1432;	// L1546
  l_step_m57: for (int m57 = 0; m57 < 8; m57++) {	// L1547
  #pragma HLS pipeline II=1
    int8_t v1435 = v1425.read(); // v1425[m57];	// L1548
    int8_t a57;	// L1549
    a57 = v1435;	// L1550
    int32_t v1437 = v1426.read(); // v1426[m57];	// L1551
    int32_t p57;	// L1552
    p57 = v1437;	// L1553
    int32_t v1439 = p57;	// L1554
    int8_t v1440 = a57;	// L1555
    int8_t v1441 = w57;	// L1556
    int16_t v1442 = v1440;	// L1557
    int16_t v1443 = v1441;	// L1558
    int16_t v1444 = v1442 * v1443;	// L1559
    ap_int<33> v1445 = v1439;	// L1560
    ap_int<33> v1446 = v1444;	// L1561
    ap_int<33> v1447 = v1445 + v1446;	// L1562
    int32_t v1448 = v1447;	// L1563
    v1428.write(v1448); // v1428[m57] = v1448;	// L1564
    int8_t v1449 = a57;	// L1565
    v1427.write(v1449); // v1427[m57] = v1449;	// L1566
  }
}

void PE_kernel_ws_gemm_2_7(
  hls::stream< int8_t > &v1450 /* v1450[8] */,
  hls::stream< int32_t > &v1451 /* v1451[8] */,
  hls::stream< int8_t > &v1452 /* v1452[8] */,
  hls::stream< int32_t > &v1453 /* v1453[8] */,
  int8_t v1454[8][8],
  int v1455,
  int v1456
) {	// L1570
  #pragma HLS stream variable=v1450 depth=9
  #pragma HLS stream variable=v1451 depth=9
  #pragma HLS stream variable=v1452 depth=9
  #pragma HLS stream variable=v1453 depth=9
  #pragma HLS array_partition variable=v1454 complete dim=1
  #pragma HLS array_partition variable=v1454 complete dim=2

  int8_t v1457 = v1454[v1455][v1456];	// L1571
  int8_t w58;	// L1572
  w58 = v1457;	// L1573
  l_step_m58: for (int m58 = 0; m58 < 8; m58++) {	// L1574
  #pragma HLS pipeline II=1
    int8_t v1460 = v1450.read(); // v1450[m58];	// L1575
    int8_t a58;	// L1576
    a58 = v1460;	// L1577
    int32_t v1462 = v1451.read(); // v1451[m58];	// L1578
    int32_t p58;	// L1579
    p58 = v1462;	// L1580
    int32_t v1464 = p58;	// L1581
    int8_t v1465 = a58;	// L1582
    int8_t v1466 = w58;	// L1583
    int16_t v1467 = v1465;	// L1584
    int16_t v1468 = v1466;	// L1585
    int16_t v1469 = v1467 * v1468;	// L1586
    ap_int<33> v1470 = v1464;	// L1587
    ap_int<33> v1471 = v1469;	// L1588
    ap_int<33> v1472 = v1470 + v1471;	// L1589
    int32_t v1473 = v1472;	// L1590
    v1453.write(v1473); // v1453[m58] = v1473;	// L1591
    int8_t v1474 = a58;	// L1592
    v1452.write(v1474); // v1452[m58] = v1474;	// L1593
  }
}

void PE_kernel_ws_gemm_3_7(
  hls::stream< int8_t > &v1475 /* v1475[8] */,
  hls::stream< int32_t > &v1476 /* v1476[8] */,
  hls::stream< int8_t > &v1477 /* v1477[8] */,
  hls::stream< int32_t > &v1478 /* v1478[8] */,
  int8_t v1479[8][8],
  int v1480,
  int v1481
) {	// L1597
  #pragma HLS stream variable=v1475 depth=9
  #pragma HLS stream variable=v1476 depth=9
  #pragma HLS stream variable=v1477 depth=9
  #pragma HLS stream variable=v1478 depth=9
  #pragma HLS array_partition variable=v1479 complete dim=1
  #pragma HLS array_partition variable=v1479 complete dim=2

  int8_t v1482 = v1479[v1480][v1481];	// L1598
  int8_t w59;	// L1599
  w59 = v1482;	// L1600
  l_step_m59: for (int m59 = 0; m59 < 8; m59++) {	// L1601
  #pragma HLS pipeline II=1
    int8_t v1485 = v1475.read(); // v1475[m59];	// L1602
    int8_t a59;	// L1603
    a59 = v1485;	// L1604
    int32_t v1487 = v1476.read(); // v1476[m59];	// L1605
    int32_t p59;	// L1606
    p59 = v1487;	// L1607
    int32_t v1489 = p59;	// L1608
    int8_t v1490 = a59;	// L1609
    int8_t v1491 = w59;	// L1610
    int16_t v1492 = v1490;	// L1611
    int16_t v1493 = v1491;	// L1612
    int16_t v1494 = v1492 * v1493;	// L1613
    ap_int<33> v1495 = v1489;	// L1614
    ap_int<33> v1496 = v1494;	// L1615
    ap_int<33> v1497 = v1495 + v1496;	// L1616
    int32_t v1498 = v1497;	// L1617
    v1478.write(v1498); // v1478[m59] = v1498;	// L1618
    int8_t v1499 = a59;	// L1619
    v1477.write(v1499); // v1477[m59] = v1499;	// L1620
  }
}

void PE_kernel_ws_gemm_4_7(
  hls::stream< int8_t > &v1500 /* v1500[8] */,
  hls::stream< int32_t > &v1501 /* v1501[8] */,
  hls::stream< int8_t > &v1502 /* v1502[8] */,
  hls::stream< int32_t > &v1503 /* v1503[8] */,
  int8_t v1504[8][8],
  int v1505,
  int v1506
) {	// L1624
  #pragma HLS stream variable=v1500 depth=9
  #pragma HLS stream variable=v1501 depth=9
  #pragma HLS stream variable=v1502 depth=9
  #pragma HLS stream variable=v1503 depth=9
  #pragma HLS array_partition variable=v1504 complete dim=1
  #pragma HLS array_partition variable=v1504 complete dim=2

  int8_t v1507 = v1504[v1505][v1506];	// L1625
  int8_t w60;	// L1626
  w60 = v1507;	// L1627
  l_step_m60: for (int m60 = 0; m60 < 8; m60++) {	// L1628
  #pragma HLS pipeline II=1
    int8_t v1510 = v1500.read(); // v1500[m60];	// L1629
    int8_t a60;	// L1630
    a60 = v1510;	// L1631
    int32_t v1512 = v1501.read(); // v1501[m60];	// L1632
    int32_t p60;	// L1633
    p60 = v1512;	// L1634
    int32_t v1514 = p60;	// L1635
    int8_t v1515 = a60;	// L1636
    int8_t v1516 = w60;	// L1637
    int16_t v1517 = v1515;	// L1638
    int16_t v1518 = v1516;	// L1639
    int16_t v1519 = v1517 * v1518;	// L1640
    ap_int<33> v1520 = v1514;	// L1641
    ap_int<33> v1521 = v1519;	// L1642
    ap_int<33> v1522 = v1520 + v1521;	// L1643
    int32_t v1523 = v1522;	// L1644
    v1503.write(v1523); // v1503[m60] = v1523;	// L1645
    int8_t v1524 = a60;	// L1646
    v1502.write(v1524); // v1502[m60] = v1524;	// L1647
  }
}

void PE_kernel_ws_gemm_5_7(
  hls::stream< int8_t > &v1525 /* v1525[8] */,
  hls::stream< int32_t > &v1526 /* v1526[8] */,
  hls::stream< int8_t > &v1527 /* v1527[8] */,
  hls::stream< int32_t > &v1528 /* v1528[8] */,
  int8_t v1529[8][8],
  int v1530,
  int v1531
) {	// L1651
  #pragma HLS stream variable=v1525 depth=9
  #pragma HLS stream variable=v1526 depth=9
  #pragma HLS stream variable=v1527 depth=9
  #pragma HLS stream variable=v1528 depth=9
  #pragma HLS array_partition variable=v1529 complete dim=1
  #pragma HLS array_partition variable=v1529 complete dim=2

  int8_t v1532 = v1529[v1530][v1531];	// L1652
  int8_t w61;	// L1653
  w61 = v1532;	// L1654
  l_step_m61: for (int m61 = 0; m61 < 8; m61++) {	// L1655
  #pragma HLS pipeline II=1
    int8_t v1535 = v1525.read(); // v1525[m61];	// L1656
    int8_t a61;	// L1657
    a61 = v1535;	// L1658
    int32_t v1537 = v1526.read(); // v1526[m61];	// L1659
    int32_t p61;	// L1660
    p61 = v1537;	// L1661
    int32_t v1539 = p61;	// L1662
    int8_t v1540 = a61;	// L1663
    int8_t v1541 = w61;	// L1664
    int16_t v1542 = v1540;	// L1665
    int16_t v1543 = v1541;	// L1666
    int16_t v1544 = v1542 * v1543;	// L1667
    ap_int<33> v1545 = v1539;	// L1668
    ap_int<33> v1546 = v1544;	// L1669
    ap_int<33> v1547 = v1545 + v1546;	// L1670
    int32_t v1548 = v1547;	// L1671
    v1528.write(v1548); // v1528[m61] = v1548;	// L1672
    int8_t v1549 = a61;	// L1673
    v1527.write(v1549); // v1527[m61] = v1549;	// L1674
  }
}

void PE_kernel_ws_gemm_6_7(
  hls::stream< int8_t > &v1550 /* v1550[8] */,
  hls::stream< int32_t > &v1551 /* v1551[8] */,
  hls::stream< int8_t > &v1552 /* v1552[8] */,
  hls::stream< int32_t > &v1553 /* v1553[8] */,
  int8_t v1554[8][8],
  int v1555,
  int v1556
) {	// L1678
  #pragma HLS stream variable=v1550 depth=9
  #pragma HLS stream variable=v1551 depth=9
  #pragma HLS stream variable=v1552 depth=9
  #pragma HLS stream variable=v1553 depth=9
  #pragma HLS array_partition variable=v1554 complete dim=1
  #pragma HLS array_partition variable=v1554 complete dim=2

  int8_t v1557 = v1554[v1555][v1556];	// L1679
  int8_t w62;	// L1680
  w62 = v1557;	// L1681
  l_step_m62: for (int m62 = 0; m62 < 8; m62++) {	// L1682
  #pragma HLS pipeline II=1
    int8_t v1560 = v1550.read(); // v1550[m62];	// L1683
    int8_t a62;	// L1684
    a62 = v1560;	// L1685
    int32_t v1562 = v1551.read(); // v1551[m62];	// L1686
    int32_t p62;	// L1687
    p62 = v1562;	// L1688
    int32_t v1564 = p62;	// L1689
    int8_t v1565 = a62;	// L1690
    int8_t v1566 = w62;	// L1691
    int16_t v1567 = v1565;	// L1692
    int16_t v1568 = v1566;	// L1693
    int16_t v1569 = v1567 * v1568;	// L1694
    ap_int<33> v1570 = v1564;	// L1695
    ap_int<33> v1571 = v1569;	// L1696
    ap_int<33> v1572 = v1570 + v1571;	// L1697
    int32_t v1573 = v1572;	// L1698
    v1553.write(v1573); // v1553[m62] = v1573;	// L1699
    int8_t v1574 = a62;	// L1700
    v1552.write(v1574); // v1552[m62] = v1574;	// L1701
  }
}

void PE_kernel_ws_gemm_7_7(
  hls::stream< int8_t > &v1575 /* v1575[8] */,
  hls::stream< int32_t > &v1576 /* v1576[8] */,
  hls::stream< int8_t > &v1577 /* v1577[8] */,
  hls::stream< int32_t > &v1578 /* v1578[8] */,
  int8_t v1579[8][8],
  int v1580,
  int v1581
) {	// L1705
  #pragma HLS stream variable=v1575 depth=9
  #pragma HLS stream variable=v1576 depth=9
  #pragma HLS stream variable=v1577 depth=9
  #pragma HLS stream variable=v1578 depth=9
  #pragma HLS array_partition variable=v1579 complete dim=1
  #pragma HLS array_partition variable=v1579 complete dim=2

  int8_t v1582 = v1579[v1580][v1581];	// L1706
  int8_t w63;	// L1707
  w63 = v1582;	// L1708
  l_step_m63: for (int m63 = 0; m63 < 8; m63++) {	// L1709
  #pragma HLS pipeline II=1
    int8_t v1585 = v1575.read(); // v1575[m63];	// L1710
    int8_t a63;	// L1711
    a63 = v1585;	// L1712
    int32_t v1587 = v1576.read(); // v1576[m63];	// L1713
    int32_t p63;	// L1714
    p63 = v1587;	// L1715
    int32_t v1589 = p63;	// L1716
    int8_t v1590 = a63;	// L1717
    int8_t v1591 = w63;	// L1718
    int16_t v1592 = v1590;	// L1719
    int16_t v1593 = v1591;	// L1720
    int16_t v1594 = v1592 * v1593;	// L1721
    ap_int<33> v1595 = v1589;	// L1722
    ap_int<33> v1596 = v1594;	// L1723
    ap_int<33> v1597 = v1595 + v1596;	// L1724
    int32_t v1598 = v1597;	// L1725
    v1578.write(v1598); // v1578[m63] = v1598;	// L1726
    int8_t v1599 = a63;	// L1727
    v1577.write(v1599); // v1577[m63] = v1599;	// L1728
  }
}

void systolic_tile_ws_gemm(
  int8_t v1600[8][8],
  int8_t v1601[8][8],
  int32_t v1602[8][8]
) {	// L1732
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v1600 complete dim=2

  #pragma HLS array_partition variable=v1601 complete dim=1
  #pragma HLS array_partition variable=v1601 complete dim=2

  #pragma HLS array_partition variable=v1602 complete dim=2

  hls::stream< int8_t > A_fifo[8][9] /* A_fifo[8][9][8] */;	// L1733
  #pragma HLS stream variable=A_fifo depth=9
  hls::stream< int32_t > P_fifo[8][9] /* P_fifo[8][9][8] */;	// L1734
  #pragma HLS stream variable=P_fifo depth=9
  int8_t A_drain[8];	// L1735
  l_data_load_m64: for (int m64 = 0; m64 < 8; m64++) {	// L1736
    l_S_k_0_k: for (int k = 0; k < 8; k++) {	// L1737
      int8_t v1608 = v1600[m64][k];	// L1738
      A_fifo[k][0].write(v1608); // A_fifo[k][0][m64] = v1608;	// L1739
    }
    l_S_n_1_n: for (int n = 0; n < 8; n++) {	// L1741
      P_fifo[n][0].write(0); // P_fifo[n][0][m64] = 0;	// L1743
    }
  }
  hls::stream< int8_t > &v1610 /* v1610[8] */ = A_fifo[0][0];	// L1747
  hls::stream< int32_t > &v1611 /* v1611[8] */ = P_fifo[0][0];	// L1748
  hls::stream< int8_t > &v1612 /* v1612[8] */ = A_fifo[0][1];	// L1754
  hls::stream< int32_t > &v1613 /* v1613[8] */ = P_fifo[0][1];	// L1755
  PE_kernel_ws_gemm_0_0(v1610, v1611, v1612, v1613, v1601, 0, 0);	// L1756
  hls::stream< int8_t > &v1614 /* v1614[8] */ = A_fifo[0][1];	// L1758
  hls::stream< int32_t > &v1615 /* v1615[8] */ = P_fifo[1][0];	// L1759
  hls::stream< int8_t > &v1616 /* v1616[8] */ = A_fifo[0][2];	// L1763
  hls::stream< int32_t > &v1617 /* v1617[8] */ = P_fifo[1][1];	// L1764
  PE_kernel_ws_gemm_1_0(v1614, v1615, v1616, v1617, v1601, 0, 1);	// L1765
  hls::stream< int8_t > &v1618 /* v1618[8] */ = A_fifo[0][2];	// L1767
  hls::stream< int32_t > &v1619 /* v1619[8] */ = P_fifo[2][0];	// L1768
  hls::stream< int8_t > &v1620 /* v1620[8] */ = A_fifo[0][3];	// L1772
  hls::stream< int32_t > &v1621 /* v1621[8] */ = P_fifo[2][1];	// L1773
  PE_kernel_ws_gemm_2_0(v1618, v1619, v1620, v1621, v1601, 0, 2);	// L1774
  hls::stream< int8_t > &v1622 /* v1622[8] */ = A_fifo[0][3];	// L1776
  hls::stream< int32_t > &v1623 /* v1623[8] */ = P_fifo[3][0];	// L1777
  hls::stream< int8_t > &v1624 /* v1624[8] */ = A_fifo[0][4];	// L1781
  hls::stream< int32_t > &v1625 /* v1625[8] */ = P_fifo[3][1];	// L1782
  PE_kernel_ws_gemm_3_0(v1622, v1623, v1624, v1625, v1601, 0, 3);	// L1783
  hls::stream< int8_t > &v1626 /* v1626[8] */ = A_fifo[0][4];	// L1785
  hls::stream< int32_t > &v1627 /* v1627[8] */ = P_fifo[4][0];	// L1786
  hls::stream< int8_t > &v1628 /* v1628[8] */ = A_fifo[0][5];	// L1790
  hls::stream< int32_t > &v1629 /* v1629[8] */ = P_fifo[4][1];	// L1791
  PE_kernel_ws_gemm_4_0(v1626, v1627, v1628, v1629, v1601, 0, 4);	// L1792
  hls::stream< int8_t > &v1630 /* v1630[8] */ = A_fifo[0][5];	// L1794
  hls::stream< int32_t > &v1631 /* v1631[8] */ = P_fifo[5][0];	// L1795
  hls::stream< int8_t > &v1632 /* v1632[8] */ = A_fifo[0][6];	// L1799
  hls::stream< int32_t > &v1633 /* v1633[8] */ = P_fifo[5][1];	// L1800
  PE_kernel_ws_gemm_5_0(v1630, v1631, v1632, v1633, v1601, 0, 5);	// L1801
  hls::stream< int8_t > &v1634 /* v1634[8] */ = A_fifo[0][6];	// L1803
  hls::stream< int32_t > &v1635 /* v1635[8] */ = P_fifo[6][0];	// L1804
  hls::stream< int8_t > &v1636 /* v1636[8] */ = A_fifo[0][7];	// L1808
  hls::stream< int32_t > &v1637 /* v1637[8] */ = P_fifo[6][1];	// L1809
  PE_kernel_ws_gemm_6_0(v1634, v1635, v1636, v1637, v1601, 0, 6);	// L1810
  hls::stream< int8_t > &v1638 /* v1638[8] */ = A_fifo[0][7];	// L1812
  hls::stream< int32_t > &v1639 /* v1639[8] */ = P_fifo[7][0];	// L1813
  hls::stream< int8_t > &v1640 /* v1640[8] */ = A_fifo[0][8];	// L1817
  hls::stream< int32_t > &v1641 /* v1641[8] */ = P_fifo[7][1];	// L1818
  PE_kernel_ws_gemm_7_0(v1638, v1639, v1640, v1641, v1601, 0, 7);	// L1819
  hls::stream< int8_t > &v1642 /* v1642[8] */ = A_fifo[1][0];	// L1820
  hls::stream< int32_t > &v1643 /* v1643[8] */ = P_fifo[0][1];	// L1821
  hls::stream< int8_t > &v1644 /* v1644[8] */ = A_fifo[1][1];	// L1822
  hls::stream< int32_t > &v1645 /* v1645[8] */ = P_fifo[0][2];	// L1823
  PE_kernel_ws_gemm_0_1(v1642, v1643, v1644, v1645, v1601, 1, 0);	// L1824
  hls::stream< int8_t > &v1646 /* v1646[8] */ = A_fifo[1][1];	// L1825
  hls::stream< int32_t > &v1647 /* v1647[8] */ = P_fifo[1][1];	// L1826
  hls::stream< int8_t > &v1648 /* v1648[8] */ = A_fifo[1][2];	// L1827
  hls::stream< int32_t > &v1649 /* v1649[8] */ = P_fifo[1][2];	// L1828
  PE_kernel_ws_gemm_1_1(v1646, v1647, v1648, v1649, v1601, 1, 1);	// L1829
  hls::stream< int8_t > &v1650 /* v1650[8] */ = A_fifo[1][2];	// L1830
  hls::stream< int32_t > &v1651 /* v1651[8] */ = P_fifo[2][1];	// L1831
  hls::stream< int8_t > &v1652 /* v1652[8] */ = A_fifo[1][3];	// L1832
  hls::stream< int32_t > &v1653 /* v1653[8] */ = P_fifo[2][2];	// L1833
  PE_kernel_ws_gemm_2_1(v1650, v1651, v1652, v1653, v1601, 1, 2);	// L1834
  hls::stream< int8_t > &v1654 /* v1654[8] */ = A_fifo[1][3];	// L1835
  hls::stream< int32_t > &v1655 /* v1655[8] */ = P_fifo[3][1];	// L1836
  hls::stream< int8_t > &v1656 /* v1656[8] */ = A_fifo[1][4];	// L1837
  hls::stream< int32_t > &v1657 /* v1657[8] */ = P_fifo[3][2];	// L1838
  PE_kernel_ws_gemm_3_1(v1654, v1655, v1656, v1657, v1601, 1, 3);	// L1839
  hls::stream< int8_t > &v1658 /* v1658[8] */ = A_fifo[1][4];	// L1840
  hls::stream< int32_t > &v1659 /* v1659[8] */ = P_fifo[4][1];	// L1841
  hls::stream< int8_t > &v1660 /* v1660[8] */ = A_fifo[1][5];	// L1842
  hls::stream< int32_t > &v1661 /* v1661[8] */ = P_fifo[4][2];	// L1843
  PE_kernel_ws_gemm_4_1(v1658, v1659, v1660, v1661, v1601, 1, 4);	// L1844
  hls::stream< int8_t > &v1662 /* v1662[8] */ = A_fifo[1][5];	// L1845
  hls::stream< int32_t > &v1663 /* v1663[8] */ = P_fifo[5][1];	// L1846
  hls::stream< int8_t > &v1664 /* v1664[8] */ = A_fifo[1][6];	// L1847
  hls::stream< int32_t > &v1665 /* v1665[8] */ = P_fifo[5][2];	// L1848
  PE_kernel_ws_gemm_5_1(v1662, v1663, v1664, v1665, v1601, 1, 5);	// L1849
  hls::stream< int8_t > &v1666 /* v1666[8] */ = A_fifo[1][6];	// L1850
  hls::stream< int32_t > &v1667 /* v1667[8] */ = P_fifo[6][1];	// L1851
  hls::stream< int8_t > &v1668 /* v1668[8] */ = A_fifo[1][7];	// L1852
  hls::stream< int32_t > &v1669 /* v1669[8] */ = P_fifo[6][2];	// L1853
  PE_kernel_ws_gemm_6_1(v1666, v1667, v1668, v1669, v1601, 1, 6);	// L1854
  hls::stream< int8_t > &v1670 /* v1670[8] */ = A_fifo[1][7];	// L1855
  hls::stream< int32_t > &v1671 /* v1671[8] */ = P_fifo[7][1];	// L1856
  hls::stream< int8_t > &v1672 /* v1672[8] */ = A_fifo[1][8];	// L1857
  hls::stream< int32_t > &v1673 /* v1673[8] */ = P_fifo[7][2];	// L1858
  PE_kernel_ws_gemm_7_1(v1670, v1671, v1672, v1673, v1601, 1, 7);	// L1859
  hls::stream< int8_t > &v1674 /* v1674[8] */ = A_fifo[2][0];	// L1860
  hls::stream< int32_t > &v1675 /* v1675[8] */ = P_fifo[0][2];	// L1861
  hls::stream< int8_t > &v1676 /* v1676[8] */ = A_fifo[2][1];	// L1862
  hls::stream< int32_t > &v1677 /* v1677[8] */ = P_fifo[0][3];	// L1863
  PE_kernel_ws_gemm_0_2(v1674, v1675, v1676, v1677, v1601, 2, 0);	// L1864
  hls::stream< int8_t > &v1678 /* v1678[8] */ = A_fifo[2][1];	// L1865
  hls::stream< int32_t > &v1679 /* v1679[8] */ = P_fifo[1][2];	// L1866
  hls::stream< int8_t > &v1680 /* v1680[8] */ = A_fifo[2][2];	// L1867
  hls::stream< int32_t > &v1681 /* v1681[8] */ = P_fifo[1][3];	// L1868
  PE_kernel_ws_gemm_1_2(v1678, v1679, v1680, v1681, v1601, 2, 1);	// L1869
  hls::stream< int8_t > &v1682 /* v1682[8] */ = A_fifo[2][2];	// L1870
  hls::stream< int32_t > &v1683 /* v1683[8] */ = P_fifo[2][2];	// L1871
  hls::stream< int8_t > &v1684 /* v1684[8] */ = A_fifo[2][3];	// L1872
  hls::stream< int32_t > &v1685 /* v1685[8] */ = P_fifo[2][3];	// L1873
  PE_kernel_ws_gemm_2_2(v1682, v1683, v1684, v1685, v1601, 2, 2);	// L1874
  hls::stream< int8_t > &v1686 /* v1686[8] */ = A_fifo[2][3];	// L1875
  hls::stream< int32_t > &v1687 /* v1687[8] */ = P_fifo[3][2];	// L1876
  hls::stream< int8_t > &v1688 /* v1688[8] */ = A_fifo[2][4];	// L1877
  hls::stream< int32_t > &v1689 /* v1689[8] */ = P_fifo[3][3];	// L1878
  PE_kernel_ws_gemm_3_2(v1686, v1687, v1688, v1689, v1601, 2, 3);	// L1879
  hls::stream< int8_t > &v1690 /* v1690[8] */ = A_fifo[2][4];	// L1880
  hls::stream< int32_t > &v1691 /* v1691[8] */ = P_fifo[4][2];	// L1881
  hls::stream< int8_t > &v1692 /* v1692[8] */ = A_fifo[2][5];	// L1882
  hls::stream< int32_t > &v1693 /* v1693[8] */ = P_fifo[4][3];	// L1883
  PE_kernel_ws_gemm_4_2(v1690, v1691, v1692, v1693, v1601, 2, 4);	// L1884
  hls::stream< int8_t > &v1694 /* v1694[8] */ = A_fifo[2][5];	// L1885
  hls::stream< int32_t > &v1695 /* v1695[8] */ = P_fifo[5][2];	// L1886
  hls::stream< int8_t > &v1696 /* v1696[8] */ = A_fifo[2][6];	// L1887
  hls::stream< int32_t > &v1697 /* v1697[8] */ = P_fifo[5][3];	// L1888
  PE_kernel_ws_gemm_5_2(v1694, v1695, v1696, v1697, v1601, 2, 5);	// L1889
  hls::stream< int8_t > &v1698 /* v1698[8] */ = A_fifo[2][6];	// L1890
  hls::stream< int32_t > &v1699 /* v1699[8] */ = P_fifo[6][2];	// L1891
  hls::stream< int8_t > &v1700 /* v1700[8] */ = A_fifo[2][7];	// L1892
  hls::stream< int32_t > &v1701 /* v1701[8] */ = P_fifo[6][3];	// L1893
  PE_kernel_ws_gemm_6_2(v1698, v1699, v1700, v1701, v1601, 2, 6);	// L1894
  hls::stream< int8_t > &v1702 /* v1702[8] */ = A_fifo[2][7];	// L1895
  hls::stream< int32_t > &v1703 /* v1703[8] */ = P_fifo[7][2];	// L1896
  hls::stream< int8_t > &v1704 /* v1704[8] */ = A_fifo[2][8];	// L1897
  hls::stream< int32_t > &v1705 /* v1705[8] */ = P_fifo[7][3];	// L1898
  PE_kernel_ws_gemm_7_2(v1702, v1703, v1704, v1705, v1601, 2, 7);	// L1899
  hls::stream< int8_t > &v1706 /* v1706[8] */ = A_fifo[3][0];	// L1900
  hls::stream< int32_t > &v1707 /* v1707[8] */ = P_fifo[0][3];	// L1901
  hls::stream< int8_t > &v1708 /* v1708[8] */ = A_fifo[3][1];	// L1902
  hls::stream< int32_t > &v1709 /* v1709[8] */ = P_fifo[0][4];	// L1903
  PE_kernel_ws_gemm_0_3(v1706, v1707, v1708, v1709, v1601, 3, 0);	// L1904
  hls::stream< int8_t > &v1710 /* v1710[8] */ = A_fifo[3][1];	// L1905
  hls::stream< int32_t > &v1711 /* v1711[8] */ = P_fifo[1][3];	// L1906
  hls::stream< int8_t > &v1712 /* v1712[8] */ = A_fifo[3][2];	// L1907
  hls::stream< int32_t > &v1713 /* v1713[8] */ = P_fifo[1][4];	// L1908
  PE_kernel_ws_gemm_1_3(v1710, v1711, v1712, v1713, v1601, 3, 1);	// L1909
  hls::stream< int8_t > &v1714 /* v1714[8] */ = A_fifo[3][2];	// L1910
  hls::stream< int32_t > &v1715 /* v1715[8] */ = P_fifo[2][3];	// L1911
  hls::stream< int8_t > &v1716 /* v1716[8] */ = A_fifo[3][3];	// L1912
  hls::stream< int32_t > &v1717 /* v1717[8] */ = P_fifo[2][4];	// L1913
  PE_kernel_ws_gemm_2_3(v1714, v1715, v1716, v1717, v1601, 3, 2);	// L1914
  hls::stream< int8_t > &v1718 /* v1718[8] */ = A_fifo[3][3];	// L1915
  hls::stream< int32_t > &v1719 /* v1719[8] */ = P_fifo[3][3];	// L1916
  hls::stream< int8_t > &v1720 /* v1720[8] */ = A_fifo[3][4];	// L1917
  hls::stream< int32_t > &v1721 /* v1721[8] */ = P_fifo[3][4];	// L1918
  PE_kernel_ws_gemm_3_3(v1718, v1719, v1720, v1721, v1601, 3, 3);	// L1919
  hls::stream< int8_t > &v1722 /* v1722[8] */ = A_fifo[3][4];	// L1920
  hls::stream< int32_t > &v1723 /* v1723[8] */ = P_fifo[4][3];	// L1921
  hls::stream< int8_t > &v1724 /* v1724[8] */ = A_fifo[3][5];	// L1922
  hls::stream< int32_t > &v1725 /* v1725[8] */ = P_fifo[4][4];	// L1923
  PE_kernel_ws_gemm_4_3(v1722, v1723, v1724, v1725, v1601, 3, 4);	// L1924
  hls::stream< int8_t > &v1726 /* v1726[8] */ = A_fifo[3][5];	// L1925
  hls::stream< int32_t > &v1727 /* v1727[8] */ = P_fifo[5][3];	// L1926
  hls::stream< int8_t > &v1728 /* v1728[8] */ = A_fifo[3][6];	// L1927
  hls::stream< int32_t > &v1729 /* v1729[8] */ = P_fifo[5][4];	// L1928
  PE_kernel_ws_gemm_5_3(v1726, v1727, v1728, v1729, v1601, 3, 5);	// L1929
  hls::stream< int8_t > &v1730 /* v1730[8] */ = A_fifo[3][6];	// L1930
  hls::stream< int32_t > &v1731 /* v1731[8] */ = P_fifo[6][3];	// L1931
  hls::stream< int8_t > &v1732 /* v1732[8] */ = A_fifo[3][7];	// L1932
  hls::stream< int32_t > &v1733 /* v1733[8] */ = P_fifo[6][4];	// L1933
  PE_kernel_ws_gemm_6_3(v1730, v1731, v1732, v1733, v1601, 3, 6);	// L1934
  hls::stream< int8_t > &v1734 /* v1734[8] */ = A_fifo[3][7];	// L1935
  hls::stream< int32_t > &v1735 /* v1735[8] */ = P_fifo[7][3];	// L1936
  hls::stream< int8_t > &v1736 /* v1736[8] */ = A_fifo[3][8];	// L1937
  hls::stream< int32_t > &v1737 /* v1737[8] */ = P_fifo[7][4];	// L1938
  PE_kernel_ws_gemm_7_3(v1734, v1735, v1736, v1737, v1601, 3, 7);	// L1939
  hls::stream< int8_t > &v1738 /* v1738[8] */ = A_fifo[4][0];	// L1940
  hls::stream< int32_t > &v1739 /* v1739[8] */ = P_fifo[0][4];	// L1941
  hls::stream< int8_t > &v1740 /* v1740[8] */ = A_fifo[4][1];	// L1942
  hls::stream< int32_t > &v1741 /* v1741[8] */ = P_fifo[0][5];	// L1943
  PE_kernel_ws_gemm_0_4(v1738, v1739, v1740, v1741, v1601, 4, 0);	// L1944
  hls::stream< int8_t > &v1742 /* v1742[8] */ = A_fifo[4][1];	// L1945
  hls::stream< int32_t > &v1743 /* v1743[8] */ = P_fifo[1][4];	// L1946
  hls::stream< int8_t > &v1744 /* v1744[8] */ = A_fifo[4][2];	// L1947
  hls::stream< int32_t > &v1745 /* v1745[8] */ = P_fifo[1][5];	// L1948
  PE_kernel_ws_gemm_1_4(v1742, v1743, v1744, v1745, v1601, 4, 1);	// L1949
  hls::stream< int8_t > &v1746 /* v1746[8] */ = A_fifo[4][2];	// L1950
  hls::stream< int32_t > &v1747 /* v1747[8] */ = P_fifo[2][4];	// L1951
  hls::stream< int8_t > &v1748 /* v1748[8] */ = A_fifo[4][3];	// L1952
  hls::stream< int32_t > &v1749 /* v1749[8] */ = P_fifo[2][5];	// L1953
  PE_kernel_ws_gemm_2_4(v1746, v1747, v1748, v1749, v1601, 4, 2);	// L1954
  hls::stream< int8_t > &v1750 /* v1750[8] */ = A_fifo[4][3];	// L1955
  hls::stream< int32_t > &v1751 /* v1751[8] */ = P_fifo[3][4];	// L1956
  hls::stream< int8_t > &v1752 /* v1752[8] */ = A_fifo[4][4];	// L1957
  hls::stream< int32_t > &v1753 /* v1753[8] */ = P_fifo[3][5];	// L1958
  PE_kernel_ws_gemm_3_4(v1750, v1751, v1752, v1753, v1601, 4, 3);	// L1959
  hls::stream< int8_t > &v1754 /* v1754[8] */ = A_fifo[4][4];	// L1960
  hls::stream< int32_t > &v1755 /* v1755[8] */ = P_fifo[4][4];	// L1961
  hls::stream< int8_t > &v1756 /* v1756[8] */ = A_fifo[4][5];	// L1962
  hls::stream< int32_t > &v1757 /* v1757[8] */ = P_fifo[4][5];	// L1963
  PE_kernel_ws_gemm_4_4(v1754, v1755, v1756, v1757, v1601, 4, 4);	// L1964
  hls::stream< int8_t > &v1758 /* v1758[8] */ = A_fifo[4][5];	// L1965
  hls::stream< int32_t > &v1759 /* v1759[8] */ = P_fifo[5][4];	// L1966
  hls::stream< int8_t > &v1760 /* v1760[8] */ = A_fifo[4][6];	// L1967
  hls::stream< int32_t > &v1761 /* v1761[8] */ = P_fifo[5][5];	// L1968
  PE_kernel_ws_gemm_5_4(v1758, v1759, v1760, v1761, v1601, 4, 5);	// L1969
  hls::stream< int8_t > &v1762 /* v1762[8] */ = A_fifo[4][6];	// L1970
  hls::stream< int32_t > &v1763 /* v1763[8] */ = P_fifo[6][4];	// L1971
  hls::stream< int8_t > &v1764 /* v1764[8] */ = A_fifo[4][7];	// L1972
  hls::stream< int32_t > &v1765 /* v1765[8] */ = P_fifo[6][5];	// L1973
  PE_kernel_ws_gemm_6_4(v1762, v1763, v1764, v1765, v1601, 4, 6);	// L1974
  hls::stream< int8_t > &v1766 /* v1766[8] */ = A_fifo[4][7];	// L1975
  hls::stream< int32_t > &v1767 /* v1767[8] */ = P_fifo[7][4];	// L1976
  hls::stream< int8_t > &v1768 /* v1768[8] */ = A_fifo[4][8];	// L1977
  hls::stream< int32_t > &v1769 /* v1769[8] */ = P_fifo[7][5];	// L1978
  PE_kernel_ws_gemm_7_4(v1766, v1767, v1768, v1769, v1601, 4, 7);	// L1979
  hls::stream< int8_t > &v1770 /* v1770[8] */ = A_fifo[5][0];	// L1980
  hls::stream< int32_t > &v1771 /* v1771[8] */ = P_fifo[0][5];	// L1981
  hls::stream< int8_t > &v1772 /* v1772[8] */ = A_fifo[5][1];	// L1982
  hls::stream< int32_t > &v1773 /* v1773[8] */ = P_fifo[0][6];	// L1983
  PE_kernel_ws_gemm_0_5(v1770, v1771, v1772, v1773, v1601, 5, 0);	// L1984
  hls::stream< int8_t > &v1774 /* v1774[8] */ = A_fifo[5][1];	// L1985
  hls::stream< int32_t > &v1775 /* v1775[8] */ = P_fifo[1][5];	// L1986
  hls::stream< int8_t > &v1776 /* v1776[8] */ = A_fifo[5][2];	// L1987
  hls::stream< int32_t > &v1777 /* v1777[8] */ = P_fifo[1][6];	// L1988
  PE_kernel_ws_gemm_1_5(v1774, v1775, v1776, v1777, v1601, 5, 1);	// L1989
  hls::stream< int8_t > &v1778 /* v1778[8] */ = A_fifo[5][2];	// L1990
  hls::stream< int32_t > &v1779 /* v1779[8] */ = P_fifo[2][5];	// L1991
  hls::stream< int8_t > &v1780 /* v1780[8] */ = A_fifo[5][3];	// L1992
  hls::stream< int32_t > &v1781 /* v1781[8] */ = P_fifo[2][6];	// L1993
  PE_kernel_ws_gemm_2_5(v1778, v1779, v1780, v1781, v1601, 5, 2);	// L1994
  hls::stream< int8_t > &v1782 /* v1782[8] */ = A_fifo[5][3];	// L1995
  hls::stream< int32_t > &v1783 /* v1783[8] */ = P_fifo[3][5];	// L1996
  hls::stream< int8_t > &v1784 /* v1784[8] */ = A_fifo[5][4];	// L1997
  hls::stream< int32_t > &v1785 /* v1785[8] */ = P_fifo[3][6];	// L1998
  PE_kernel_ws_gemm_3_5(v1782, v1783, v1784, v1785, v1601, 5, 3);	// L1999
  hls::stream< int8_t > &v1786 /* v1786[8] */ = A_fifo[5][4];	// L2000
  hls::stream< int32_t > &v1787 /* v1787[8] */ = P_fifo[4][5];	// L2001
  hls::stream< int8_t > &v1788 /* v1788[8] */ = A_fifo[5][5];	// L2002
  hls::stream< int32_t > &v1789 /* v1789[8] */ = P_fifo[4][6];	// L2003
  PE_kernel_ws_gemm_4_5(v1786, v1787, v1788, v1789, v1601, 5, 4);	// L2004
  hls::stream< int8_t > &v1790 /* v1790[8] */ = A_fifo[5][5];	// L2005
  hls::stream< int32_t > &v1791 /* v1791[8] */ = P_fifo[5][5];	// L2006
  hls::stream< int8_t > &v1792 /* v1792[8] */ = A_fifo[5][6];	// L2007
  hls::stream< int32_t > &v1793 /* v1793[8] */ = P_fifo[5][6];	// L2008
  PE_kernel_ws_gemm_5_5(v1790, v1791, v1792, v1793, v1601, 5, 5);	// L2009
  hls::stream< int8_t > &v1794 /* v1794[8] */ = A_fifo[5][6];	// L2010
  hls::stream< int32_t > &v1795 /* v1795[8] */ = P_fifo[6][5];	// L2011
  hls::stream< int8_t > &v1796 /* v1796[8] */ = A_fifo[5][7];	// L2012
  hls::stream< int32_t > &v1797 /* v1797[8] */ = P_fifo[6][6];	// L2013
  PE_kernel_ws_gemm_6_5(v1794, v1795, v1796, v1797, v1601, 5, 6);	// L2014
  hls::stream< int8_t > &v1798 /* v1798[8] */ = A_fifo[5][7];	// L2015
  hls::stream< int32_t > &v1799 /* v1799[8] */ = P_fifo[7][5];	// L2016
  hls::stream< int8_t > &v1800 /* v1800[8] */ = A_fifo[5][8];	// L2017
  hls::stream< int32_t > &v1801 /* v1801[8] */ = P_fifo[7][6];	// L2018
  PE_kernel_ws_gemm_7_5(v1798, v1799, v1800, v1801, v1601, 5, 7);	// L2019
  hls::stream< int8_t > &v1802 /* v1802[8] */ = A_fifo[6][0];	// L2020
  hls::stream< int32_t > &v1803 /* v1803[8] */ = P_fifo[0][6];	// L2021
  hls::stream< int8_t > &v1804 /* v1804[8] */ = A_fifo[6][1];	// L2022
  hls::stream< int32_t > &v1805 /* v1805[8] */ = P_fifo[0][7];	// L2023
  PE_kernel_ws_gemm_0_6(v1802, v1803, v1804, v1805, v1601, 6, 0);	// L2024
  hls::stream< int8_t > &v1806 /* v1806[8] */ = A_fifo[6][1];	// L2025
  hls::stream< int32_t > &v1807 /* v1807[8] */ = P_fifo[1][6];	// L2026
  hls::stream< int8_t > &v1808 /* v1808[8] */ = A_fifo[6][2];	// L2027
  hls::stream< int32_t > &v1809 /* v1809[8] */ = P_fifo[1][7];	// L2028
  PE_kernel_ws_gemm_1_6(v1806, v1807, v1808, v1809, v1601, 6, 1);	// L2029
  hls::stream< int8_t > &v1810 /* v1810[8] */ = A_fifo[6][2];	// L2030
  hls::stream< int32_t > &v1811 /* v1811[8] */ = P_fifo[2][6];	// L2031
  hls::stream< int8_t > &v1812 /* v1812[8] */ = A_fifo[6][3];	// L2032
  hls::stream< int32_t > &v1813 /* v1813[8] */ = P_fifo[2][7];	// L2033
  PE_kernel_ws_gemm_2_6(v1810, v1811, v1812, v1813, v1601, 6, 2);	// L2034
  hls::stream< int8_t > &v1814 /* v1814[8] */ = A_fifo[6][3];	// L2035
  hls::stream< int32_t > &v1815 /* v1815[8] */ = P_fifo[3][6];	// L2036
  hls::stream< int8_t > &v1816 /* v1816[8] */ = A_fifo[6][4];	// L2037
  hls::stream< int32_t > &v1817 /* v1817[8] */ = P_fifo[3][7];	// L2038
  PE_kernel_ws_gemm_3_6(v1814, v1815, v1816, v1817, v1601, 6, 3);	// L2039
  hls::stream< int8_t > &v1818 /* v1818[8] */ = A_fifo[6][4];	// L2040
  hls::stream< int32_t > &v1819 /* v1819[8] */ = P_fifo[4][6];	// L2041
  hls::stream< int8_t > &v1820 /* v1820[8] */ = A_fifo[6][5];	// L2042
  hls::stream< int32_t > &v1821 /* v1821[8] */ = P_fifo[4][7];	// L2043
  PE_kernel_ws_gemm_4_6(v1818, v1819, v1820, v1821, v1601, 6, 4);	// L2044
  hls::stream< int8_t > &v1822 /* v1822[8] */ = A_fifo[6][5];	// L2045
  hls::stream< int32_t > &v1823 /* v1823[8] */ = P_fifo[5][6];	// L2046
  hls::stream< int8_t > &v1824 /* v1824[8] */ = A_fifo[6][6];	// L2047
  hls::stream< int32_t > &v1825 /* v1825[8] */ = P_fifo[5][7];	// L2048
  PE_kernel_ws_gemm_5_6(v1822, v1823, v1824, v1825, v1601, 6, 5);	// L2049
  hls::stream< int8_t > &v1826 /* v1826[8] */ = A_fifo[6][6];	// L2050
  hls::stream< int32_t > &v1827 /* v1827[8] */ = P_fifo[6][6];	// L2051
  hls::stream< int8_t > &v1828 /* v1828[8] */ = A_fifo[6][7];	// L2052
  hls::stream< int32_t > &v1829 /* v1829[8] */ = P_fifo[6][7];	// L2053
  PE_kernel_ws_gemm_6_6(v1826, v1827, v1828, v1829, v1601, 6, 6);	// L2054
  hls::stream< int8_t > &v1830 /* v1830[8] */ = A_fifo[6][7];	// L2055
  hls::stream< int32_t > &v1831 /* v1831[8] */ = P_fifo[7][6];	// L2056
  hls::stream< int8_t > &v1832 /* v1832[8] */ = A_fifo[6][8];	// L2057
  hls::stream< int32_t > &v1833 /* v1833[8] */ = P_fifo[7][7];	// L2058
  PE_kernel_ws_gemm_7_6(v1830, v1831, v1832, v1833, v1601, 6, 7);	// L2059
  hls::stream< int8_t > &v1834 /* v1834[8] */ = A_fifo[7][0];	// L2060
  hls::stream< int32_t > &v1835 /* v1835[8] */ = P_fifo[0][7];	// L2061
  hls::stream< int8_t > &v1836 /* v1836[8] */ = A_fifo[7][1];	// L2062
  hls::stream< int32_t > &v1837 /* v1837[8] */ = P_fifo[0][8];	// L2063
  PE_kernel_ws_gemm_0_7(v1834, v1835, v1836, v1837, v1601, 7, 0);	// L2064
  hls::stream< int8_t > &v1838 /* v1838[8] */ = A_fifo[7][1];	// L2065
  hls::stream< int32_t > &v1839 /* v1839[8] */ = P_fifo[1][7];	// L2066
  hls::stream< int8_t > &v1840 /* v1840[8] */ = A_fifo[7][2];	// L2067
  hls::stream< int32_t > &v1841 /* v1841[8] */ = P_fifo[1][8];	// L2068
  PE_kernel_ws_gemm_1_7(v1838, v1839, v1840, v1841, v1601, 7, 1);	// L2069
  hls::stream< int8_t > &v1842 /* v1842[8] */ = A_fifo[7][2];	// L2070
  hls::stream< int32_t > &v1843 /* v1843[8] */ = P_fifo[2][7];	// L2071
  hls::stream< int8_t > &v1844 /* v1844[8] */ = A_fifo[7][3];	// L2072
  hls::stream< int32_t > &v1845 /* v1845[8] */ = P_fifo[2][8];	// L2073
  PE_kernel_ws_gemm_2_7(v1842, v1843, v1844, v1845, v1601, 7, 2);	// L2074
  hls::stream< int8_t > &v1846 /* v1846[8] */ = A_fifo[7][3];	// L2075
  hls::stream< int32_t > &v1847 /* v1847[8] */ = P_fifo[3][7];	// L2076
  hls::stream< int8_t > &v1848 /* v1848[8] */ = A_fifo[7][4];	// L2077
  hls::stream< int32_t > &v1849 /* v1849[8] */ = P_fifo[3][8];	// L2078
  PE_kernel_ws_gemm_3_7(v1846, v1847, v1848, v1849, v1601, 7, 3);	// L2079
  hls::stream< int8_t > &v1850 /* v1850[8] */ = A_fifo[7][4];	// L2080
  hls::stream< int32_t > &v1851 /* v1851[8] */ = P_fifo[4][7];	// L2081
  hls::stream< int8_t > &v1852 /* v1852[8] */ = A_fifo[7][5];	// L2082
  hls::stream< int32_t > &v1853 /* v1853[8] */ = P_fifo[4][8];	// L2083
  PE_kernel_ws_gemm_4_7(v1850, v1851, v1852, v1853, v1601, 7, 4);	// L2084
  hls::stream< int8_t > &v1854 /* v1854[8] */ = A_fifo[7][5];	// L2085
  hls::stream< int32_t > &v1855 /* v1855[8] */ = P_fifo[5][7];	// L2086
  hls::stream< int8_t > &v1856 /* v1856[8] */ = A_fifo[7][6];	// L2087
  hls::stream< int32_t > &v1857 /* v1857[8] */ = P_fifo[5][8];	// L2088
  PE_kernel_ws_gemm_5_7(v1854, v1855, v1856, v1857, v1601, 7, 5);	// L2089
  hls::stream< int8_t > &v1858 /* v1858[8] */ = A_fifo[7][6];	// L2090
  hls::stream< int32_t > &v1859 /* v1859[8] */ = P_fifo[6][7];	// L2091
  hls::stream< int8_t > &v1860 /* v1860[8] */ = A_fifo[7][7];	// L2092
  hls::stream< int32_t > &v1861 /* v1861[8] */ = P_fifo[6][8];	// L2093
  PE_kernel_ws_gemm_6_7(v1858, v1859, v1860, v1861, v1601, 7, 6);	// L2094
  hls::stream< int8_t > &v1862 /* v1862[8] */ = A_fifo[7][7];	// L2095
  hls::stream< int32_t > &v1863 /* v1863[8] */ = P_fifo[7][7];	// L2096
  hls::stream< int8_t > &v1864 /* v1864[8] */ = A_fifo[7][8];	// L2097
  hls::stream< int32_t > &v1865 /* v1865[8] */ = P_fifo[7][8];	// L2098
  PE_kernel_ws_gemm_7_7(v1862, v1863, v1864, v1865, v1601, 7, 7);	// L2099
  l_data_drain_m65: for (int m65 = 0; m65 < 8; m65++) {	// L2100
    l_S_k_4_k1: for (int k1 = 0; k1 < 8; k1++) {	// L2101
      int8_t v1868 = A_fifo[k1][8].read(); // A_fifo[k1][8][m65];	// L2102
      A_drain[k1] = v1868;	// L2103
    }
    l_S_n_5_n1: for (int n1 = 0; n1 < 8; n1++) {	// L2105
      int32_t v1870 = P_fifo[n1][8].read(); // P_fifo[n1][8][m65];	// L2106
      v1602[m65][n1] = v1870;	// L2107
    }
  }
}

void systolic_ws_gemm(
  int8_t v1871[8][8],
  int8_t v1872[8][8],
  int32_t v1873[8][8]
) {	// L2112
  int8_t local_A[8][8];	// L2113
  #pragma HLS array_partition variable=local_A complete dim=2

  int8_t local_B[8][8];	// L2114
  #pragma HLS array_partition variable=local_B complete dim=1
  #pragma HLS array_partition variable=local_B complete dim=2

  int32_t local_C[8][8];	// L2115
  #pragma HLS array_partition variable=local_C complete dim=2

  l_load_A_am: for (int am = 0; am < 8; am++) {	// L2116
  #pragma HLS pipeline II=1
    l_ak: for (int ak = 0; ak < 8; ak++) {	// L2117
      int8_t v1879 = v1871[am][ak];	// L2118
      local_A[am][ak] = v1879;	// L2119
    }
  }
  l_load_B_bk: for (int bk = 0; bk < 8; bk++) {	// L2122
  #pragma HLS pipeline II=1
    l_bn: for (int bn = 0; bn < 8; bn++) {	// L2123
      int8_t v1882 = v1872[bk][bn];	// L2124
      local_B[bk][bn] = v1882;	// L2125
    }
  }
  systolic_tile_ws_gemm(local_A, local_B, local_C);	// L2128
  l_store_C_sm: for (int sm = 0; sm < 8; sm++) {	// L2129
  #pragma HLS pipeline II=1
    l_sn: for (int sn = 0; sn < 8; sn++) {	// L2130
      int32_t v1885 = local_C[sm][sn];	// L2131
      v1873[sm][sn] = v1885;	// L2132
    }
  }
}

void load_buf0(
  int8_t v1886[64],
  int8_t v1887[8][8]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 8; load_buf0_l_0++) {	//
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 8; load_buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v1890 = v1886[((load_buf0_l_0 * 8) + load_buf0_l_1)];	//
      v1887[load_buf0_l_0][load_buf0_l_1] = v1890;	//
    }
  }
}

void load_buf1(
  int8_t v1891[64],
  int8_t v1892[8][8]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 8; load_buf1_l_0++) {	//
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 8; load_buf1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v1895 = v1891[((load_buf1_l_0 * 8) + load_buf1_l_1)];	//
      v1892[load_buf1_l_0][load_buf1_l_1] = v1895;	//
    }
  }
}

void store_res2(
  int32_t v1896[8][8],
  int32_t v1897[64]
) {	//
  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 8; store_res2_l_0++) {	//
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 8; store_res2_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int32_t v1900 = v1896[store_res2_l_0][store_res2_l_1];	//
      v1897[((store_res2_l_0 * 8) + store_res2_l_1)] = v1900;	//
    }
  }
}

/// This is top function.
void gemm(
  int8_t *v1901,
  int8_t *v1902,
  int32_t *v1903
) {	// L2137
  #pragma HLS interface m_axi port=v1901 offset=slave bundle=gmem0 depth=64
  #pragma HLS interface m_axi port=v1902 offset=slave bundle=gmem1 depth=64
  #pragma HLS interface m_axi port=v1903 offset=slave bundle=gmem2 depth=64
  int8_t buf0[8][8];	//
  load_buf0(v1901, buf0);	//
  int8_t buf1[8][8];	//
  load_buf1(v1902, buf1);	//
  int32_t buf2[8][8];	//
  systolic_ws_gemm(buf0, buf1, buf2);	// L2138
  store_res2(buf2, v1903);	//
}


} // extern "C"

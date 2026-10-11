
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
  hls::stream< int8_t > &v0 /* v0[4] */,
  hls::stream< int32_t > &v1 /* v1[4] */,
  hls::stream< int8_t > &v2 /* v2[4] */,
  hls::stream< int32_t > &v3 /* v3[4] */,
  int8_t v4[4][4],
  int v5,
  int v6
) {	// L4
  #pragma HLS stream variable=v0 depth=5
  #pragma HLS stream variable=v1 depth=5
  #pragma HLS stream variable=v2 depth=5
  #pragma HLS stream variable=v3 depth=5
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  int8_t v7 = v4[v5][v6];	// L5
  int8_t w;	// L6
  w = v7;	// L7
  l_step_m: for (int m = 0; m < 4; m++) {	// L8
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
  hls::stream< int8_t > &v25 /* v25[4] */,
  hls::stream< int32_t > &v26 /* v26[4] */,
  hls::stream< int8_t > &v27 /* v27[4] */,
  hls::stream< int32_t > &v28 /* v28[4] */,
  int8_t v29[4][4],
  int v30,
  int v31
) {	// L31
  #pragma HLS stream variable=v25 depth=5
  #pragma HLS stream variable=v26 depth=5
  #pragma HLS stream variable=v27 depth=5
  #pragma HLS stream variable=v28 depth=5
  #pragma HLS array_partition variable=v29 complete dim=1
  #pragma HLS array_partition variable=v29 complete dim=2

  int8_t v32 = v29[v30][v31];	// L32
  int8_t w1;	// L33
  w1 = v32;	// L34
  l_step_m1: for (int m1 = 0; m1 < 4; m1++) {	// L35
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
  hls::stream< int8_t > &v50 /* v50[4] */,
  hls::stream< int32_t > &v51 /* v51[4] */,
  hls::stream< int8_t > &v52 /* v52[4] */,
  hls::stream< int32_t > &v53 /* v53[4] */,
  int8_t v54[4][4],
  int v55,
  int v56
) {	// L58
  #pragma HLS stream variable=v50 depth=5
  #pragma HLS stream variable=v51 depth=5
  #pragma HLS stream variable=v52 depth=5
  #pragma HLS stream variable=v53 depth=5
  #pragma HLS array_partition variable=v54 complete dim=1
  #pragma HLS array_partition variable=v54 complete dim=2

  int8_t v57 = v54[v55][v56];	// L59
  int8_t w2;	// L60
  w2 = v57;	// L61
  l_step_m2: for (int m2 = 0; m2 < 4; m2++) {	// L62
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
  hls::stream< int8_t > &v75 /* v75[4] */,
  hls::stream< int32_t > &v76 /* v76[4] */,
  hls::stream< int8_t > &v77 /* v77[4] */,
  hls::stream< int32_t > &v78 /* v78[4] */,
  int8_t v79[4][4],
  int v80,
  int v81
) {	// L85
  #pragma HLS stream variable=v75 depth=5
  #pragma HLS stream variable=v76 depth=5
  #pragma HLS stream variable=v77 depth=5
  #pragma HLS stream variable=v78 depth=5
  #pragma HLS array_partition variable=v79 complete dim=1
  #pragma HLS array_partition variable=v79 complete dim=2

  int8_t v82 = v79[v80][v81];	// L86
  int8_t w3;	// L87
  w3 = v82;	// L88
  l_step_m3: for (int m3 = 0; m3 < 4; m3++) {	// L89
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

void PE_kernel_ws_gemm_0_1(
  hls::stream< int8_t > &v100 /* v100[4] */,
  hls::stream< int32_t > &v101 /* v101[4] */,
  hls::stream< int8_t > &v102 /* v102[4] */,
  hls::stream< int32_t > &v103 /* v103[4] */,
  int8_t v104[4][4],
  int v105,
  int v106
) {	// L112
  #pragma HLS stream variable=v100 depth=5
  #pragma HLS stream variable=v101 depth=5
  #pragma HLS stream variable=v102 depth=5
  #pragma HLS stream variable=v103 depth=5
  #pragma HLS array_partition variable=v104 complete dim=1
  #pragma HLS array_partition variable=v104 complete dim=2

  int8_t v107 = v104[v105][v106];	// L113
  int8_t w4;	// L114
  w4 = v107;	// L115
  l_step_m4: for (int m4 = 0; m4 < 4; m4++) {	// L116
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

void PE_kernel_ws_gemm_1_1(
  hls::stream< int8_t > &v125 /* v125[4] */,
  hls::stream< int32_t > &v126 /* v126[4] */,
  hls::stream< int8_t > &v127 /* v127[4] */,
  hls::stream< int32_t > &v128 /* v128[4] */,
  int8_t v129[4][4],
  int v130,
  int v131
) {	// L139
  #pragma HLS stream variable=v125 depth=5
  #pragma HLS stream variable=v126 depth=5
  #pragma HLS stream variable=v127 depth=5
  #pragma HLS stream variable=v128 depth=5
  #pragma HLS array_partition variable=v129 complete dim=1
  #pragma HLS array_partition variable=v129 complete dim=2

  int8_t v132 = v129[v130][v131];	// L140
  int8_t w5;	// L141
  w5 = v132;	// L142
  l_step_m5: for (int m5 = 0; m5 < 4; m5++) {	// L143
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

void PE_kernel_ws_gemm_2_1(
  hls::stream< int8_t > &v150 /* v150[4] */,
  hls::stream< int32_t > &v151 /* v151[4] */,
  hls::stream< int8_t > &v152 /* v152[4] */,
  hls::stream< int32_t > &v153 /* v153[4] */,
  int8_t v154[4][4],
  int v155,
  int v156
) {	// L166
  #pragma HLS stream variable=v150 depth=5
  #pragma HLS stream variable=v151 depth=5
  #pragma HLS stream variable=v152 depth=5
  #pragma HLS stream variable=v153 depth=5
  #pragma HLS array_partition variable=v154 complete dim=1
  #pragma HLS array_partition variable=v154 complete dim=2

  int8_t v157 = v154[v155][v156];	// L167
  int8_t w6;	// L168
  w6 = v157;	// L169
  l_step_m6: for (int m6 = 0; m6 < 4; m6++) {	// L170
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

void PE_kernel_ws_gemm_3_1(
  hls::stream< int8_t > &v175 /* v175[4] */,
  hls::stream< int32_t > &v176 /* v176[4] */,
  hls::stream< int8_t > &v177 /* v177[4] */,
  hls::stream< int32_t > &v178 /* v178[4] */,
  int8_t v179[4][4],
  int v180,
  int v181
) {	// L193
  #pragma HLS stream variable=v175 depth=5
  #pragma HLS stream variable=v176 depth=5
  #pragma HLS stream variable=v177 depth=5
  #pragma HLS stream variable=v178 depth=5
  #pragma HLS array_partition variable=v179 complete dim=1
  #pragma HLS array_partition variable=v179 complete dim=2

  int8_t v182 = v179[v180][v181];	// L194
  int8_t w7;	// L195
  w7 = v182;	// L196
  l_step_m7: for (int m7 = 0; m7 < 4; m7++) {	// L197
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

void PE_kernel_ws_gemm_0_2(
  hls::stream< int8_t > &v200 /* v200[4] */,
  hls::stream< int32_t > &v201 /* v201[4] */,
  hls::stream< int8_t > &v202 /* v202[4] */,
  hls::stream< int32_t > &v203 /* v203[4] */,
  int8_t v204[4][4],
  int v205,
  int v206
) {	// L220
  #pragma HLS stream variable=v200 depth=5
  #pragma HLS stream variable=v201 depth=5
  #pragma HLS stream variable=v202 depth=5
  #pragma HLS stream variable=v203 depth=5
  #pragma HLS array_partition variable=v204 complete dim=1
  #pragma HLS array_partition variable=v204 complete dim=2

  int8_t v207 = v204[v205][v206];	// L221
  int8_t w8;	// L222
  w8 = v207;	// L223
  l_step_m8: for (int m8 = 0; m8 < 4; m8++) {	// L224
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

void PE_kernel_ws_gemm_1_2(
  hls::stream< int8_t > &v225 /* v225[4] */,
  hls::stream< int32_t > &v226 /* v226[4] */,
  hls::stream< int8_t > &v227 /* v227[4] */,
  hls::stream< int32_t > &v228 /* v228[4] */,
  int8_t v229[4][4],
  int v230,
  int v231
) {	// L247
  #pragma HLS stream variable=v225 depth=5
  #pragma HLS stream variable=v226 depth=5
  #pragma HLS stream variable=v227 depth=5
  #pragma HLS stream variable=v228 depth=5
  #pragma HLS array_partition variable=v229 complete dim=1
  #pragma HLS array_partition variable=v229 complete dim=2

  int8_t v232 = v229[v230][v231];	// L248
  int8_t w9;	// L249
  w9 = v232;	// L250
  l_step_m9: for (int m9 = 0; m9 < 4; m9++) {	// L251
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

void PE_kernel_ws_gemm_2_2(
  hls::stream< int8_t > &v250 /* v250[4] */,
  hls::stream< int32_t > &v251 /* v251[4] */,
  hls::stream< int8_t > &v252 /* v252[4] */,
  hls::stream< int32_t > &v253 /* v253[4] */,
  int8_t v254[4][4],
  int v255,
  int v256
) {	// L274
  #pragma HLS stream variable=v250 depth=5
  #pragma HLS stream variable=v251 depth=5
  #pragma HLS stream variable=v252 depth=5
  #pragma HLS stream variable=v253 depth=5
  #pragma HLS array_partition variable=v254 complete dim=1
  #pragma HLS array_partition variable=v254 complete dim=2

  int8_t v257 = v254[v255][v256];	// L275
  int8_t w10;	// L276
  w10 = v257;	// L277
  l_step_m10: for (int m10 = 0; m10 < 4; m10++) {	// L278
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

void PE_kernel_ws_gemm_3_2(
  hls::stream< int8_t > &v275 /* v275[4] */,
  hls::stream< int32_t > &v276 /* v276[4] */,
  hls::stream< int8_t > &v277 /* v277[4] */,
  hls::stream< int32_t > &v278 /* v278[4] */,
  int8_t v279[4][4],
  int v280,
  int v281
) {	// L301
  #pragma HLS stream variable=v275 depth=5
  #pragma HLS stream variable=v276 depth=5
  #pragma HLS stream variable=v277 depth=5
  #pragma HLS stream variable=v278 depth=5
  #pragma HLS array_partition variable=v279 complete dim=1
  #pragma HLS array_partition variable=v279 complete dim=2

  int8_t v282 = v279[v280][v281];	// L302
  int8_t w11;	// L303
  w11 = v282;	// L304
  l_step_m11: for (int m11 = 0; m11 < 4; m11++) {	// L305
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

void PE_kernel_ws_gemm_0_3(
  hls::stream< int8_t > &v300 /* v300[4] */,
  hls::stream< int32_t > &v301 /* v301[4] */,
  hls::stream< int8_t > &v302 /* v302[4] */,
  hls::stream< int32_t > &v303 /* v303[4] */,
  int8_t v304[4][4],
  int v305,
  int v306
) {	// L328
  #pragma HLS stream variable=v300 depth=5
  #pragma HLS stream variable=v301 depth=5
  #pragma HLS stream variable=v302 depth=5
  #pragma HLS stream variable=v303 depth=5
  #pragma HLS array_partition variable=v304 complete dim=1
  #pragma HLS array_partition variable=v304 complete dim=2

  int8_t v307 = v304[v305][v306];	// L329
  int8_t w12;	// L330
  w12 = v307;	// L331
  l_step_m12: for (int m12 = 0; m12 < 4; m12++) {	// L332
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

void PE_kernel_ws_gemm_1_3(
  hls::stream< int8_t > &v325 /* v325[4] */,
  hls::stream< int32_t > &v326 /* v326[4] */,
  hls::stream< int8_t > &v327 /* v327[4] */,
  hls::stream< int32_t > &v328 /* v328[4] */,
  int8_t v329[4][4],
  int v330,
  int v331
) {	// L355
  #pragma HLS stream variable=v325 depth=5
  #pragma HLS stream variable=v326 depth=5
  #pragma HLS stream variable=v327 depth=5
  #pragma HLS stream variable=v328 depth=5
  #pragma HLS array_partition variable=v329 complete dim=1
  #pragma HLS array_partition variable=v329 complete dim=2

  int8_t v332 = v329[v330][v331];	// L356
  int8_t w13;	// L357
  w13 = v332;	// L358
  l_step_m13: for (int m13 = 0; m13 < 4; m13++) {	// L359
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

void PE_kernel_ws_gemm_2_3(
  hls::stream< int8_t > &v350 /* v350[4] */,
  hls::stream< int32_t > &v351 /* v351[4] */,
  hls::stream< int8_t > &v352 /* v352[4] */,
  hls::stream< int32_t > &v353 /* v353[4] */,
  int8_t v354[4][4],
  int v355,
  int v356
) {	// L382
  #pragma HLS stream variable=v350 depth=5
  #pragma HLS stream variable=v351 depth=5
  #pragma HLS stream variable=v352 depth=5
  #pragma HLS stream variable=v353 depth=5
  #pragma HLS array_partition variable=v354 complete dim=1
  #pragma HLS array_partition variable=v354 complete dim=2

  int8_t v357 = v354[v355][v356];	// L383
  int8_t w14;	// L384
  w14 = v357;	// L385
  l_step_m14: for (int m14 = 0; m14 < 4; m14++) {	// L386
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

void PE_kernel_ws_gemm_3_3(
  hls::stream< int8_t > &v375 /* v375[4] */,
  hls::stream< int32_t > &v376 /* v376[4] */,
  hls::stream< int8_t > &v377 /* v377[4] */,
  hls::stream< int32_t > &v378 /* v378[4] */,
  int8_t v379[4][4],
  int v380,
  int v381
) {	// L409
  #pragma HLS stream variable=v375 depth=5
  #pragma HLS stream variable=v376 depth=5
  #pragma HLS stream variable=v377 depth=5
  #pragma HLS stream variable=v378 depth=5
  #pragma HLS array_partition variable=v379 complete dim=1
  #pragma HLS array_partition variable=v379 complete dim=2

  int8_t v382 = v379[v380][v381];	// L410
  int8_t w15;	// L411
  w15 = v382;	// L412
  l_step_m15: for (int m15 = 0; m15 < 4; m15++) {	// L413
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

void systolic_tile_ws_gemm(
  int8_t v400[4][4],
  int8_t v401[4][4],
  int32_t v402[4][4]
) {	// L436
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v400 complete dim=2

  #pragma HLS array_partition variable=v401 complete dim=1
  #pragma HLS array_partition variable=v401 complete dim=2

  #pragma HLS array_partition variable=v402 complete dim=2

  hls::stream< int8_t > A_fifo[4][5] /* A_fifo[4][5][4] */;	// L437
  #pragma HLS stream variable=A_fifo depth=5
  hls::stream< int32_t > P_fifo[4][5] /* P_fifo[4][5][4] */;	// L438
  #pragma HLS stream variable=P_fifo depth=5
  int8_t A_drain[4];	// L439
  l_data_load_m16: for (int m16 = 0; m16 < 4; m16++) {	// L440
    l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L441
      int8_t v408 = v400[m16][k];	// L442
      A_fifo[k][0].write(v408); // A_fifo[k][0][m16] = v408;	// L443
    }
    l_S_n_1_n: for (int n = 0; n < 4; n++) {	// L445
      P_fifo[n][0].write(0); // P_fifo[n][0][m16] = 0;	// L447
    }
  }
  hls::stream< int8_t > &v410 /* v410[4] */ = A_fifo[0][0];	// L451
  hls::stream< int32_t > &v411 /* v411[4] */ = P_fifo[0][0];	// L452
  hls::stream< int8_t > &v412 /* v412[4] */ = A_fifo[0][1];	// L458
  hls::stream< int32_t > &v413 /* v413[4] */ = P_fifo[0][1];	// L459
  PE_kernel_ws_gemm_0_0(v410, v411, v412, v413, v401, 0, 0);	// L460
  hls::stream< int8_t > &v414 /* v414[4] */ = A_fifo[0][1];	// L462
  hls::stream< int32_t > &v415 /* v415[4] */ = P_fifo[1][0];	// L463
  hls::stream< int8_t > &v416 /* v416[4] */ = A_fifo[0][2];	// L467
  hls::stream< int32_t > &v417 /* v417[4] */ = P_fifo[1][1];	// L468
  PE_kernel_ws_gemm_1_0(v414, v415, v416, v417, v401, 0, 1);	// L469
  hls::stream< int8_t > &v418 /* v418[4] */ = A_fifo[0][2];	// L471
  hls::stream< int32_t > &v419 /* v419[4] */ = P_fifo[2][0];	// L472
  hls::stream< int8_t > &v420 /* v420[4] */ = A_fifo[0][3];	// L476
  hls::stream< int32_t > &v421 /* v421[4] */ = P_fifo[2][1];	// L477
  PE_kernel_ws_gemm_2_0(v418, v419, v420, v421, v401, 0, 2);	// L478
  hls::stream< int8_t > &v422 /* v422[4] */ = A_fifo[0][3];	// L480
  hls::stream< int32_t > &v423 /* v423[4] */ = P_fifo[3][0];	// L481
  hls::stream< int8_t > &v424 /* v424[4] */ = A_fifo[0][4];	// L485
  hls::stream< int32_t > &v425 /* v425[4] */ = P_fifo[3][1];	// L486
  PE_kernel_ws_gemm_3_0(v422, v423, v424, v425, v401, 0, 3);	// L487
  hls::stream< int8_t > &v426 /* v426[4] */ = A_fifo[1][0];	// L488
  hls::stream< int32_t > &v427 /* v427[4] */ = P_fifo[0][1];	// L489
  hls::stream< int8_t > &v428 /* v428[4] */ = A_fifo[1][1];	// L490
  hls::stream< int32_t > &v429 /* v429[4] */ = P_fifo[0][2];	// L491
  PE_kernel_ws_gemm_0_1(v426, v427, v428, v429, v401, 1, 0);	// L492
  hls::stream< int8_t > &v430 /* v430[4] */ = A_fifo[1][1];	// L493
  hls::stream< int32_t > &v431 /* v431[4] */ = P_fifo[1][1];	// L494
  hls::stream< int8_t > &v432 /* v432[4] */ = A_fifo[1][2];	// L495
  hls::stream< int32_t > &v433 /* v433[4] */ = P_fifo[1][2];	// L496
  PE_kernel_ws_gemm_1_1(v430, v431, v432, v433, v401, 1, 1);	// L497
  hls::stream< int8_t > &v434 /* v434[4] */ = A_fifo[1][2];	// L498
  hls::stream< int32_t > &v435 /* v435[4] */ = P_fifo[2][1];	// L499
  hls::stream< int8_t > &v436 /* v436[4] */ = A_fifo[1][3];	// L500
  hls::stream< int32_t > &v437 /* v437[4] */ = P_fifo[2][2];	// L501
  PE_kernel_ws_gemm_2_1(v434, v435, v436, v437, v401, 1, 2);	// L502
  hls::stream< int8_t > &v438 /* v438[4] */ = A_fifo[1][3];	// L503
  hls::stream< int32_t > &v439 /* v439[4] */ = P_fifo[3][1];	// L504
  hls::stream< int8_t > &v440 /* v440[4] */ = A_fifo[1][4];	// L505
  hls::stream< int32_t > &v441 /* v441[4] */ = P_fifo[3][2];	// L506
  PE_kernel_ws_gemm_3_1(v438, v439, v440, v441, v401, 1, 3);	// L507
  hls::stream< int8_t > &v442 /* v442[4] */ = A_fifo[2][0];	// L508
  hls::stream< int32_t > &v443 /* v443[4] */ = P_fifo[0][2];	// L509
  hls::stream< int8_t > &v444 /* v444[4] */ = A_fifo[2][1];	// L510
  hls::stream< int32_t > &v445 /* v445[4] */ = P_fifo[0][3];	// L511
  PE_kernel_ws_gemm_0_2(v442, v443, v444, v445, v401, 2, 0);	// L512
  hls::stream< int8_t > &v446 /* v446[4] */ = A_fifo[2][1];	// L513
  hls::stream< int32_t > &v447 /* v447[4] */ = P_fifo[1][2];	// L514
  hls::stream< int8_t > &v448 /* v448[4] */ = A_fifo[2][2];	// L515
  hls::stream< int32_t > &v449 /* v449[4] */ = P_fifo[1][3];	// L516
  PE_kernel_ws_gemm_1_2(v446, v447, v448, v449, v401, 2, 1);	// L517
  hls::stream< int8_t > &v450 /* v450[4] */ = A_fifo[2][2];	// L518
  hls::stream< int32_t > &v451 /* v451[4] */ = P_fifo[2][2];	// L519
  hls::stream< int8_t > &v452 /* v452[4] */ = A_fifo[2][3];	// L520
  hls::stream< int32_t > &v453 /* v453[4] */ = P_fifo[2][3];	// L521
  PE_kernel_ws_gemm_2_2(v450, v451, v452, v453, v401, 2, 2);	// L522
  hls::stream< int8_t > &v454 /* v454[4] */ = A_fifo[2][3];	// L523
  hls::stream< int32_t > &v455 /* v455[4] */ = P_fifo[3][2];	// L524
  hls::stream< int8_t > &v456 /* v456[4] */ = A_fifo[2][4];	// L525
  hls::stream< int32_t > &v457 /* v457[4] */ = P_fifo[3][3];	// L526
  PE_kernel_ws_gemm_3_2(v454, v455, v456, v457, v401, 2, 3);	// L527
  hls::stream< int8_t > &v458 /* v458[4] */ = A_fifo[3][0];	// L528
  hls::stream< int32_t > &v459 /* v459[4] */ = P_fifo[0][3];	// L529
  hls::stream< int8_t > &v460 /* v460[4] */ = A_fifo[3][1];	// L530
  hls::stream< int32_t > &v461 /* v461[4] */ = P_fifo[0][4];	// L531
  PE_kernel_ws_gemm_0_3(v458, v459, v460, v461, v401, 3, 0);	// L532
  hls::stream< int8_t > &v462 /* v462[4] */ = A_fifo[3][1];	// L533
  hls::stream< int32_t > &v463 /* v463[4] */ = P_fifo[1][3];	// L534
  hls::stream< int8_t > &v464 /* v464[4] */ = A_fifo[3][2];	// L535
  hls::stream< int32_t > &v465 /* v465[4] */ = P_fifo[1][4];	// L536
  PE_kernel_ws_gemm_1_3(v462, v463, v464, v465, v401, 3, 1);	// L537
  hls::stream< int8_t > &v466 /* v466[4] */ = A_fifo[3][2];	// L538
  hls::stream< int32_t > &v467 /* v467[4] */ = P_fifo[2][3];	// L539
  hls::stream< int8_t > &v468 /* v468[4] */ = A_fifo[3][3];	// L540
  hls::stream< int32_t > &v469 /* v469[4] */ = P_fifo[2][4];	// L541
  PE_kernel_ws_gemm_2_3(v466, v467, v468, v469, v401, 3, 2);	// L542
  hls::stream< int8_t > &v470 /* v470[4] */ = A_fifo[3][3];	// L543
  hls::stream< int32_t > &v471 /* v471[4] */ = P_fifo[3][3];	// L544
  hls::stream< int8_t > &v472 /* v472[4] */ = A_fifo[3][4];	// L545
  hls::stream< int32_t > &v473 /* v473[4] */ = P_fifo[3][4];	// L546
  PE_kernel_ws_gemm_3_3(v470, v471, v472, v473, v401, 3, 3);	// L547
  l_data_drain_m17: for (int m17 = 0; m17 < 4; m17++) {	// L548
    l_S_k_4_k1: for (int k1 = 0; k1 < 4; k1++) {	// L549
      int8_t v476 = A_fifo[k1][4].read(); // A_fifo[k1][4][m17];	// L550
      A_drain[k1] = v476;	// L551
    }
    l_S_n_5_n1: for (int n1 = 0; n1 < 4; n1++) {	// L553
      int32_t v478 = P_fifo[n1][4].read(); // P_fifo[n1][4][m17];	// L554
      v402[m17][n1] = v478;	// L555
    }
  }
}

void systolic_ws_gemm(
  int8_t v479[4][4],
  int8_t v480[4][4],
  int32_t v481[4][4]
) {	// L560
  int8_t local_A[4][4];	// L561
  #pragma HLS array_partition variable=local_A complete dim=2

  int8_t local_B[4][4];	// L562
  #pragma HLS array_partition variable=local_B complete dim=1
  #pragma HLS array_partition variable=local_B complete dim=2

  int32_t local_C[4][4];	// L563
  #pragma HLS array_partition variable=local_C complete dim=2

  l_load_A_am: for (int am = 0; am < 4; am++) {	// L564
  #pragma HLS pipeline II=1
    l_ak: for (int ak = 0; ak < 4; ak++) {	// L565
      int8_t v487 = v479[am][ak];	// L566
      local_A[am][ak] = v487;	// L567
    }
  }
  l_load_B_bk: for (int bk = 0; bk < 4; bk++) {	// L570
  #pragma HLS pipeline II=1
    l_bn: for (int bn = 0; bn < 4; bn++) {	// L571
      int8_t v490 = v480[bk][bn];	// L572
      local_B[bk][bn] = v490;	// L573
    }
  }
  systolic_tile_ws_gemm(local_A, local_B, local_C);	// L576
  l_store_C_sm: for (int sm = 0; sm < 4; sm++) {	// L577
  #pragma HLS pipeline II=1
    l_sn: for (int sn = 0; sn < 4; sn++) {	// L578
      int32_t v493 = local_C[sm][sn];	// L579
      v481[sm][sn] = v493;	// L580
    }
  }
}

void load_buf0(
  int8_t v494[16],
  int8_t v495[4][4]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 4; load_buf0_l_0++) {	//
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 4; load_buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v498 = v494[((load_buf0_l_0 * 4) + load_buf0_l_1)];	//
      v495[load_buf0_l_0][load_buf0_l_1] = v498;	//
    }
  }
}

void load_buf1(
  int8_t v499[16],
  int8_t v500[4][4]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 4; load_buf1_l_0++) {	//
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 4; load_buf1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v503 = v499[((load_buf1_l_0 * 4) + load_buf1_l_1)];	//
      v500[load_buf1_l_0][load_buf1_l_1] = v503;	//
    }
  }
}

void store_res2(
  int32_t v504[4][4],
  int32_t v505[16]
) {	//
  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 4; store_res2_l_0++) {	//
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 4; store_res2_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int32_t v508 = v504[store_res2_l_0][store_res2_l_1];	//
      v505[((store_res2_l_0 * 4) + store_res2_l_1)] = v508;	//
    }
  }
}

/// This is top function.
void gemm(
  int8_t *v509,
  int8_t *v510,
  int32_t *v511
) {	// L585
  #pragma HLS interface m_axi port=v509 offset=slave bundle=gmem0 depth=16
  #pragma HLS interface m_axi port=v510 offset=slave bundle=gmem1 depth=16
  #pragma HLS interface m_axi port=v511 offset=slave bundle=gmem2 depth=16
  int8_t buf0[4][4];	//
  load_buf0(v509, buf0);	//
  int8_t buf1[4][4];	//
  load_buf1(v510, buf1);	//
  int32_t buf2[4][4];	//
  systolic_ws_gemm(buf0, buf1, buf2);	// L586
  store_res2(buf2, v511);	//
}


} // extern "C"

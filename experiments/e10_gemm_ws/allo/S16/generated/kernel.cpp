
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
  hls::stream< int8_t > &v0 /* v0[16] */,
  hls::stream< int32_t > &v1 /* v1[16] */,
  hls::stream< int8_t > &v2 /* v2[16] */,
  hls::stream< int32_t > &v3 /* v3[16] */,
  int8_t v4[16][16],
  int v5,
  int v6
) {	// L4
  #pragma HLS stream variable=v0 depth=17
  #pragma HLS stream variable=v1 depth=17
  #pragma HLS stream variable=v2 depth=17
  #pragma HLS stream variable=v3 depth=17
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  int8_t v7 = v4[v5][v6];	// L5
  int8_t w;	// L6
  w = v7;	// L7
  l_step_m: for (int m = 0; m < 16; m++) {	// L8
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
  hls::stream< int8_t > &v25 /* v25[16] */,
  hls::stream< int32_t > &v26 /* v26[16] */,
  hls::stream< int8_t > &v27 /* v27[16] */,
  hls::stream< int32_t > &v28 /* v28[16] */,
  int8_t v29[16][16],
  int v30,
  int v31
) {	// L31
  #pragma HLS stream variable=v25 depth=17
  #pragma HLS stream variable=v26 depth=17
  #pragma HLS stream variable=v27 depth=17
  #pragma HLS stream variable=v28 depth=17
  #pragma HLS array_partition variable=v29 complete dim=1
  #pragma HLS array_partition variable=v29 complete dim=2

  int8_t v32 = v29[v30][v31];	// L32
  int8_t w1;	// L33
  w1 = v32;	// L34
  l_step_m1: for (int m1 = 0; m1 < 16; m1++) {	// L35
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
  hls::stream< int8_t > &v50 /* v50[16] */,
  hls::stream< int32_t > &v51 /* v51[16] */,
  hls::stream< int8_t > &v52 /* v52[16] */,
  hls::stream< int32_t > &v53 /* v53[16] */,
  int8_t v54[16][16],
  int v55,
  int v56
) {	// L58
  #pragma HLS stream variable=v50 depth=17
  #pragma HLS stream variable=v51 depth=17
  #pragma HLS stream variable=v52 depth=17
  #pragma HLS stream variable=v53 depth=17
  #pragma HLS array_partition variable=v54 complete dim=1
  #pragma HLS array_partition variable=v54 complete dim=2

  int8_t v57 = v54[v55][v56];	// L59
  int8_t w2;	// L60
  w2 = v57;	// L61
  l_step_m2: for (int m2 = 0; m2 < 16; m2++) {	// L62
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
  hls::stream< int8_t > &v75 /* v75[16] */,
  hls::stream< int32_t > &v76 /* v76[16] */,
  hls::stream< int8_t > &v77 /* v77[16] */,
  hls::stream< int32_t > &v78 /* v78[16] */,
  int8_t v79[16][16],
  int v80,
  int v81
) {	// L85
  #pragma HLS stream variable=v75 depth=17
  #pragma HLS stream variable=v76 depth=17
  #pragma HLS stream variable=v77 depth=17
  #pragma HLS stream variable=v78 depth=17
  #pragma HLS array_partition variable=v79 complete dim=1
  #pragma HLS array_partition variable=v79 complete dim=2

  int8_t v82 = v79[v80][v81];	// L86
  int8_t w3;	// L87
  w3 = v82;	// L88
  l_step_m3: for (int m3 = 0; m3 < 16; m3++) {	// L89
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
  hls::stream< int8_t > &v100 /* v100[16] */,
  hls::stream< int32_t > &v101 /* v101[16] */,
  hls::stream< int8_t > &v102 /* v102[16] */,
  hls::stream< int32_t > &v103 /* v103[16] */,
  int8_t v104[16][16],
  int v105,
  int v106
) {	// L112
  #pragma HLS stream variable=v100 depth=17
  #pragma HLS stream variable=v101 depth=17
  #pragma HLS stream variable=v102 depth=17
  #pragma HLS stream variable=v103 depth=17
  #pragma HLS array_partition variable=v104 complete dim=1
  #pragma HLS array_partition variable=v104 complete dim=2

  int8_t v107 = v104[v105][v106];	// L113
  int8_t w4;	// L114
  w4 = v107;	// L115
  l_step_m4: for (int m4 = 0; m4 < 16; m4++) {	// L116
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
  hls::stream< int8_t > &v125 /* v125[16] */,
  hls::stream< int32_t > &v126 /* v126[16] */,
  hls::stream< int8_t > &v127 /* v127[16] */,
  hls::stream< int32_t > &v128 /* v128[16] */,
  int8_t v129[16][16],
  int v130,
  int v131
) {	// L139
  #pragma HLS stream variable=v125 depth=17
  #pragma HLS stream variable=v126 depth=17
  #pragma HLS stream variable=v127 depth=17
  #pragma HLS stream variable=v128 depth=17
  #pragma HLS array_partition variable=v129 complete dim=1
  #pragma HLS array_partition variable=v129 complete dim=2

  int8_t v132 = v129[v130][v131];	// L140
  int8_t w5;	// L141
  w5 = v132;	// L142
  l_step_m5: for (int m5 = 0; m5 < 16; m5++) {	// L143
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
  hls::stream< int8_t > &v150 /* v150[16] */,
  hls::stream< int32_t > &v151 /* v151[16] */,
  hls::stream< int8_t > &v152 /* v152[16] */,
  hls::stream< int32_t > &v153 /* v153[16] */,
  int8_t v154[16][16],
  int v155,
  int v156
) {	// L166
  #pragma HLS stream variable=v150 depth=17
  #pragma HLS stream variable=v151 depth=17
  #pragma HLS stream variable=v152 depth=17
  #pragma HLS stream variable=v153 depth=17
  #pragma HLS array_partition variable=v154 complete dim=1
  #pragma HLS array_partition variable=v154 complete dim=2

  int8_t v157 = v154[v155][v156];	// L167
  int8_t w6;	// L168
  w6 = v157;	// L169
  l_step_m6: for (int m6 = 0; m6 < 16; m6++) {	// L170
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
  hls::stream< int8_t > &v175 /* v175[16] */,
  hls::stream< int32_t > &v176 /* v176[16] */,
  hls::stream< int8_t > &v177 /* v177[16] */,
  hls::stream< int32_t > &v178 /* v178[16] */,
  int8_t v179[16][16],
  int v180,
  int v181
) {	// L193
  #pragma HLS stream variable=v175 depth=17
  #pragma HLS stream variable=v176 depth=17
  #pragma HLS stream variable=v177 depth=17
  #pragma HLS stream variable=v178 depth=17
  #pragma HLS array_partition variable=v179 complete dim=1
  #pragma HLS array_partition variable=v179 complete dim=2

  int8_t v182 = v179[v180][v181];	// L194
  int8_t w7;	// L195
  w7 = v182;	// L196
  l_step_m7: for (int m7 = 0; m7 < 16; m7++) {	// L197
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

void PE_kernel_ws_gemm_8_0(
  hls::stream< int8_t > &v200 /* v200[16] */,
  hls::stream< int32_t > &v201 /* v201[16] */,
  hls::stream< int8_t > &v202 /* v202[16] */,
  hls::stream< int32_t > &v203 /* v203[16] */,
  int8_t v204[16][16],
  int v205,
  int v206
) {	// L220
  #pragma HLS stream variable=v200 depth=17
  #pragma HLS stream variable=v201 depth=17
  #pragma HLS stream variable=v202 depth=17
  #pragma HLS stream variable=v203 depth=17
  #pragma HLS array_partition variable=v204 complete dim=1
  #pragma HLS array_partition variable=v204 complete dim=2

  int8_t v207 = v204[v205][v206];	// L221
  int8_t w8;	// L222
  w8 = v207;	// L223
  l_step_m8: for (int m8 = 0; m8 < 16; m8++) {	// L224
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

void PE_kernel_ws_gemm_9_0(
  hls::stream< int8_t > &v225 /* v225[16] */,
  hls::stream< int32_t > &v226 /* v226[16] */,
  hls::stream< int8_t > &v227 /* v227[16] */,
  hls::stream< int32_t > &v228 /* v228[16] */,
  int8_t v229[16][16],
  int v230,
  int v231
) {	// L247
  #pragma HLS stream variable=v225 depth=17
  #pragma HLS stream variable=v226 depth=17
  #pragma HLS stream variable=v227 depth=17
  #pragma HLS stream variable=v228 depth=17
  #pragma HLS array_partition variable=v229 complete dim=1
  #pragma HLS array_partition variable=v229 complete dim=2

  int8_t v232 = v229[v230][v231];	// L248
  int8_t w9;	// L249
  w9 = v232;	// L250
  l_step_m9: for (int m9 = 0; m9 < 16; m9++) {	// L251
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

void PE_kernel_ws_gemm_10_0(
  hls::stream< int8_t > &v250 /* v250[16] */,
  hls::stream< int32_t > &v251 /* v251[16] */,
  hls::stream< int8_t > &v252 /* v252[16] */,
  hls::stream< int32_t > &v253 /* v253[16] */,
  int8_t v254[16][16],
  int v255,
  int v256
) {	// L274
  #pragma HLS stream variable=v250 depth=17
  #pragma HLS stream variable=v251 depth=17
  #pragma HLS stream variable=v252 depth=17
  #pragma HLS stream variable=v253 depth=17
  #pragma HLS array_partition variable=v254 complete dim=1
  #pragma HLS array_partition variable=v254 complete dim=2

  int8_t v257 = v254[v255][v256];	// L275
  int8_t w10;	// L276
  w10 = v257;	// L277
  l_step_m10: for (int m10 = 0; m10 < 16; m10++) {	// L278
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

void PE_kernel_ws_gemm_11_0(
  hls::stream< int8_t > &v275 /* v275[16] */,
  hls::stream< int32_t > &v276 /* v276[16] */,
  hls::stream< int8_t > &v277 /* v277[16] */,
  hls::stream< int32_t > &v278 /* v278[16] */,
  int8_t v279[16][16],
  int v280,
  int v281
) {	// L301
  #pragma HLS stream variable=v275 depth=17
  #pragma HLS stream variable=v276 depth=17
  #pragma HLS stream variable=v277 depth=17
  #pragma HLS stream variable=v278 depth=17
  #pragma HLS array_partition variable=v279 complete dim=1
  #pragma HLS array_partition variable=v279 complete dim=2

  int8_t v282 = v279[v280][v281];	// L302
  int8_t w11;	// L303
  w11 = v282;	// L304
  l_step_m11: for (int m11 = 0; m11 < 16; m11++) {	// L305
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

void PE_kernel_ws_gemm_12_0(
  hls::stream< int8_t > &v300 /* v300[16] */,
  hls::stream< int32_t > &v301 /* v301[16] */,
  hls::stream< int8_t > &v302 /* v302[16] */,
  hls::stream< int32_t > &v303 /* v303[16] */,
  int8_t v304[16][16],
  int v305,
  int v306
) {	// L328
  #pragma HLS stream variable=v300 depth=17
  #pragma HLS stream variable=v301 depth=17
  #pragma HLS stream variable=v302 depth=17
  #pragma HLS stream variable=v303 depth=17
  #pragma HLS array_partition variable=v304 complete dim=1
  #pragma HLS array_partition variable=v304 complete dim=2

  int8_t v307 = v304[v305][v306];	// L329
  int8_t w12;	// L330
  w12 = v307;	// L331
  l_step_m12: for (int m12 = 0; m12 < 16; m12++) {	// L332
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

void PE_kernel_ws_gemm_13_0(
  hls::stream< int8_t > &v325 /* v325[16] */,
  hls::stream< int32_t > &v326 /* v326[16] */,
  hls::stream< int8_t > &v327 /* v327[16] */,
  hls::stream< int32_t > &v328 /* v328[16] */,
  int8_t v329[16][16],
  int v330,
  int v331
) {	// L355
  #pragma HLS stream variable=v325 depth=17
  #pragma HLS stream variable=v326 depth=17
  #pragma HLS stream variable=v327 depth=17
  #pragma HLS stream variable=v328 depth=17
  #pragma HLS array_partition variable=v329 complete dim=1
  #pragma HLS array_partition variable=v329 complete dim=2

  int8_t v332 = v329[v330][v331];	// L356
  int8_t w13;	// L357
  w13 = v332;	// L358
  l_step_m13: for (int m13 = 0; m13 < 16; m13++) {	// L359
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

void PE_kernel_ws_gemm_14_0(
  hls::stream< int8_t > &v350 /* v350[16] */,
  hls::stream< int32_t > &v351 /* v351[16] */,
  hls::stream< int8_t > &v352 /* v352[16] */,
  hls::stream< int32_t > &v353 /* v353[16] */,
  int8_t v354[16][16],
  int v355,
  int v356
) {	// L382
  #pragma HLS stream variable=v350 depth=17
  #pragma HLS stream variable=v351 depth=17
  #pragma HLS stream variable=v352 depth=17
  #pragma HLS stream variable=v353 depth=17
  #pragma HLS array_partition variable=v354 complete dim=1
  #pragma HLS array_partition variable=v354 complete dim=2

  int8_t v357 = v354[v355][v356];	// L383
  int8_t w14;	// L384
  w14 = v357;	// L385
  l_step_m14: for (int m14 = 0; m14 < 16; m14++) {	// L386
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

void PE_kernel_ws_gemm_15_0(
  hls::stream< int8_t > &v375 /* v375[16] */,
  hls::stream< int32_t > &v376 /* v376[16] */,
  hls::stream< int8_t > &v377 /* v377[16] */,
  hls::stream< int32_t > &v378 /* v378[16] */,
  int8_t v379[16][16],
  int v380,
  int v381
) {	// L409
  #pragma HLS stream variable=v375 depth=17
  #pragma HLS stream variable=v376 depth=17
  #pragma HLS stream variable=v377 depth=17
  #pragma HLS stream variable=v378 depth=17
  #pragma HLS array_partition variable=v379 complete dim=1
  #pragma HLS array_partition variable=v379 complete dim=2

  int8_t v382 = v379[v380][v381];	// L410
  int8_t w15;	// L411
  w15 = v382;	// L412
  l_step_m15: for (int m15 = 0; m15 < 16; m15++) {	// L413
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

void PE_kernel_ws_gemm_0_1(
  hls::stream< int8_t > &v400 /* v400[16] */,
  hls::stream< int32_t > &v401 /* v401[16] */,
  hls::stream< int8_t > &v402 /* v402[16] */,
  hls::stream< int32_t > &v403 /* v403[16] */,
  int8_t v404[16][16],
  int v405,
  int v406
) {	// L436
  #pragma HLS stream variable=v400 depth=17
  #pragma HLS stream variable=v401 depth=17
  #pragma HLS stream variable=v402 depth=17
  #pragma HLS stream variable=v403 depth=17
  #pragma HLS array_partition variable=v404 complete dim=1
  #pragma HLS array_partition variable=v404 complete dim=2

  int8_t v407 = v404[v405][v406];	// L437
  int8_t w16;	// L438
  w16 = v407;	// L439
  l_step_m16: for (int m16 = 0; m16 < 16; m16++) {	// L440
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

void PE_kernel_ws_gemm_1_1(
  hls::stream< int8_t > &v425 /* v425[16] */,
  hls::stream< int32_t > &v426 /* v426[16] */,
  hls::stream< int8_t > &v427 /* v427[16] */,
  hls::stream< int32_t > &v428 /* v428[16] */,
  int8_t v429[16][16],
  int v430,
  int v431
) {	// L463
  #pragma HLS stream variable=v425 depth=17
  #pragma HLS stream variable=v426 depth=17
  #pragma HLS stream variable=v427 depth=17
  #pragma HLS stream variable=v428 depth=17
  #pragma HLS array_partition variable=v429 complete dim=1
  #pragma HLS array_partition variable=v429 complete dim=2

  int8_t v432 = v429[v430][v431];	// L464
  int8_t w17;	// L465
  w17 = v432;	// L466
  l_step_m17: for (int m17 = 0; m17 < 16; m17++) {	// L467
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

void PE_kernel_ws_gemm_2_1(
  hls::stream< int8_t > &v450 /* v450[16] */,
  hls::stream< int32_t > &v451 /* v451[16] */,
  hls::stream< int8_t > &v452 /* v452[16] */,
  hls::stream< int32_t > &v453 /* v453[16] */,
  int8_t v454[16][16],
  int v455,
  int v456
) {	// L490
  #pragma HLS stream variable=v450 depth=17
  #pragma HLS stream variable=v451 depth=17
  #pragma HLS stream variable=v452 depth=17
  #pragma HLS stream variable=v453 depth=17
  #pragma HLS array_partition variable=v454 complete dim=1
  #pragma HLS array_partition variable=v454 complete dim=2

  int8_t v457 = v454[v455][v456];	// L491
  int8_t w18;	// L492
  w18 = v457;	// L493
  l_step_m18: for (int m18 = 0; m18 < 16; m18++) {	// L494
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

void PE_kernel_ws_gemm_3_1(
  hls::stream< int8_t > &v475 /* v475[16] */,
  hls::stream< int32_t > &v476 /* v476[16] */,
  hls::stream< int8_t > &v477 /* v477[16] */,
  hls::stream< int32_t > &v478 /* v478[16] */,
  int8_t v479[16][16],
  int v480,
  int v481
) {	// L517
  #pragma HLS stream variable=v475 depth=17
  #pragma HLS stream variable=v476 depth=17
  #pragma HLS stream variable=v477 depth=17
  #pragma HLS stream variable=v478 depth=17
  #pragma HLS array_partition variable=v479 complete dim=1
  #pragma HLS array_partition variable=v479 complete dim=2

  int8_t v482 = v479[v480][v481];	// L518
  int8_t w19;	// L519
  w19 = v482;	// L520
  l_step_m19: for (int m19 = 0; m19 < 16; m19++) {	// L521
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

void PE_kernel_ws_gemm_4_1(
  hls::stream< int8_t > &v500 /* v500[16] */,
  hls::stream< int32_t > &v501 /* v501[16] */,
  hls::stream< int8_t > &v502 /* v502[16] */,
  hls::stream< int32_t > &v503 /* v503[16] */,
  int8_t v504[16][16],
  int v505,
  int v506
) {	// L544
  #pragma HLS stream variable=v500 depth=17
  #pragma HLS stream variable=v501 depth=17
  #pragma HLS stream variable=v502 depth=17
  #pragma HLS stream variable=v503 depth=17
  #pragma HLS array_partition variable=v504 complete dim=1
  #pragma HLS array_partition variable=v504 complete dim=2

  int8_t v507 = v504[v505][v506];	// L545
  int8_t w20;	// L546
  w20 = v507;	// L547
  l_step_m20: for (int m20 = 0; m20 < 16; m20++) {	// L548
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

void PE_kernel_ws_gemm_5_1(
  hls::stream< int8_t > &v525 /* v525[16] */,
  hls::stream< int32_t > &v526 /* v526[16] */,
  hls::stream< int8_t > &v527 /* v527[16] */,
  hls::stream< int32_t > &v528 /* v528[16] */,
  int8_t v529[16][16],
  int v530,
  int v531
) {	// L571
  #pragma HLS stream variable=v525 depth=17
  #pragma HLS stream variable=v526 depth=17
  #pragma HLS stream variable=v527 depth=17
  #pragma HLS stream variable=v528 depth=17
  #pragma HLS array_partition variable=v529 complete dim=1
  #pragma HLS array_partition variable=v529 complete dim=2

  int8_t v532 = v529[v530][v531];	// L572
  int8_t w21;	// L573
  w21 = v532;	// L574
  l_step_m21: for (int m21 = 0; m21 < 16; m21++) {	// L575
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

void PE_kernel_ws_gemm_6_1(
  hls::stream< int8_t > &v550 /* v550[16] */,
  hls::stream< int32_t > &v551 /* v551[16] */,
  hls::stream< int8_t > &v552 /* v552[16] */,
  hls::stream< int32_t > &v553 /* v553[16] */,
  int8_t v554[16][16],
  int v555,
  int v556
) {	// L598
  #pragma HLS stream variable=v550 depth=17
  #pragma HLS stream variable=v551 depth=17
  #pragma HLS stream variable=v552 depth=17
  #pragma HLS stream variable=v553 depth=17
  #pragma HLS array_partition variable=v554 complete dim=1
  #pragma HLS array_partition variable=v554 complete dim=2

  int8_t v557 = v554[v555][v556];	// L599
  int8_t w22;	// L600
  w22 = v557;	// L601
  l_step_m22: for (int m22 = 0; m22 < 16; m22++) {	// L602
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

void PE_kernel_ws_gemm_7_1(
  hls::stream< int8_t > &v575 /* v575[16] */,
  hls::stream< int32_t > &v576 /* v576[16] */,
  hls::stream< int8_t > &v577 /* v577[16] */,
  hls::stream< int32_t > &v578 /* v578[16] */,
  int8_t v579[16][16],
  int v580,
  int v581
) {	// L625
  #pragma HLS stream variable=v575 depth=17
  #pragma HLS stream variable=v576 depth=17
  #pragma HLS stream variable=v577 depth=17
  #pragma HLS stream variable=v578 depth=17
  #pragma HLS array_partition variable=v579 complete dim=1
  #pragma HLS array_partition variable=v579 complete dim=2

  int8_t v582 = v579[v580][v581];	// L626
  int8_t w23;	// L627
  w23 = v582;	// L628
  l_step_m23: for (int m23 = 0; m23 < 16; m23++) {	// L629
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

void PE_kernel_ws_gemm_8_1(
  hls::stream< int8_t > &v600 /* v600[16] */,
  hls::stream< int32_t > &v601 /* v601[16] */,
  hls::stream< int8_t > &v602 /* v602[16] */,
  hls::stream< int32_t > &v603 /* v603[16] */,
  int8_t v604[16][16],
  int v605,
  int v606
) {	// L652
  #pragma HLS stream variable=v600 depth=17
  #pragma HLS stream variable=v601 depth=17
  #pragma HLS stream variable=v602 depth=17
  #pragma HLS stream variable=v603 depth=17
  #pragma HLS array_partition variable=v604 complete dim=1
  #pragma HLS array_partition variable=v604 complete dim=2

  int8_t v607 = v604[v605][v606];	// L653
  int8_t w24;	// L654
  w24 = v607;	// L655
  l_step_m24: for (int m24 = 0; m24 < 16; m24++) {	// L656
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

void PE_kernel_ws_gemm_9_1(
  hls::stream< int8_t > &v625 /* v625[16] */,
  hls::stream< int32_t > &v626 /* v626[16] */,
  hls::stream< int8_t > &v627 /* v627[16] */,
  hls::stream< int32_t > &v628 /* v628[16] */,
  int8_t v629[16][16],
  int v630,
  int v631
) {	// L679
  #pragma HLS stream variable=v625 depth=17
  #pragma HLS stream variable=v626 depth=17
  #pragma HLS stream variable=v627 depth=17
  #pragma HLS stream variable=v628 depth=17
  #pragma HLS array_partition variable=v629 complete dim=1
  #pragma HLS array_partition variable=v629 complete dim=2

  int8_t v632 = v629[v630][v631];	// L680
  int8_t w25;	// L681
  w25 = v632;	// L682
  l_step_m25: for (int m25 = 0; m25 < 16; m25++) {	// L683
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

void PE_kernel_ws_gemm_10_1(
  hls::stream< int8_t > &v650 /* v650[16] */,
  hls::stream< int32_t > &v651 /* v651[16] */,
  hls::stream< int8_t > &v652 /* v652[16] */,
  hls::stream< int32_t > &v653 /* v653[16] */,
  int8_t v654[16][16],
  int v655,
  int v656
) {	// L706
  #pragma HLS stream variable=v650 depth=17
  #pragma HLS stream variable=v651 depth=17
  #pragma HLS stream variable=v652 depth=17
  #pragma HLS stream variable=v653 depth=17
  #pragma HLS array_partition variable=v654 complete dim=1
  #pragma HLS array_partition variable=v654 complete dim=2

  int8_t v657 = v654[v655][v656];	// L707
  int8_t w26;	// L708
  w26 = v657;	// L709
  l_step_m26: for (int m26 = 0; m26 < 16; m26++) {	// L710
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

void PE_kernel_ws_gemm_11_1(
  hls::stream< int8_t > &v675 /* v675[16] */,
  hls::stream< int32_t > &v676 /* v676[16] */,
  hls::stream< int8_t > &v677 /* v677[16] */,
  hls::stream< int32_t > &v678 /* v678[16] */,
  int8_t v679[16][16],
  int v680,
  int v681
) {	// L733
  #pragma HLS stream variable=v675 depth=17
  #pragma HLS stream variable=v676 depth=17
  #pragma HLS stream variable=v677 depth=17
  #pragma HLS stream variable=v678 depth=17
  #pragma HLS array_partition variable=v679 complete dim=1
  #pragma HLS array_partition variable=v679 complete dim=2

  int8_t v682 = v679[v680][v681];	// L734
  int8_t w27;	// L735
  w27 = v682;	// L736
  l_step_m27: for (int m27 = 0; m27 < 16; m27++) {	// L737
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

void PE_kernel_ws_gemm_12_1(
  hls::stream< int8_t > &v700 /* v700[16] */,
  hls::stream< int32_t > &v701 /* v701[16] */,
  hls::stream< int8_t > &v702 /* v702[16] */,
  hls::stream< int32_t > &v703 /* v703[16] */,
  int8_t v704[16][16],
  int v705,
  int v706
) {	// L760
  #pragma HLS stream variable=v700 depth=17
  #pragma HLS stream variable=v701 depth=17
  #pragma HLS stream variable=v702 depth=17
  #pragma HLS stream variable=v703 depth=17
  #pragma HLS array_partition variable=v704 complete dim=1
  #pragma HLS array_partition variable=v704 complete dim=2

  int8_t v707 = v704[v705][v706];	// L761
  int8_t w28;	// L762
  w28 = v707;	// L763
  l_step_m28: for (int m28 = 0; m28 < 16; m28++) {	// L764
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

void PE_kernel_ws_gemm_13_1(
  hls::stream< int8_t > &v725 /* v725[16] */,
  hls::stream< int32_t > &v726 /* v726[16] */,
  hls::stream< int8_t > &v727 /* v727[16] */,
  hls::stream< int32_t > &v728 /* v728[16] */,
  int8_t v729[16][16],
  int v730,
  int v731
) {	// L787
  #pragma HLS stream variable=v725 depth=17
  #pragma HLS stream variable=v726 depth=17
  #pragma HLS stream variable=v727 depth=17
  #pragma HLS stream variable=v728 depth=17
  #pragma HLS array_partition variable=v729 complete dim=1
  #pragma HLS array_partition variable=v729 complete dim=2

  int8_t v732 = v729[v730][v731];	// L788
  int8_t w29;	// L789
  w29 = v732;	// L790
  l_step_m29: for (int m29 = 0; m29 < 16; m29++) {	// L791
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

void PE_kernel_ws_gemm_14_1(
  hls::stream< int8_t > &v750 /* v750[16] */,
  hls::stream< int32_t > &v751 /* v751[16] */,
  hls::stream< int8_t > &v752 /* v752[16] */,
  hls::stream< int32_t > &v753 /* v753[16] */,
  int8_t v754[16][16],
  int v755,
  int v756
) {	// L814
  #pragma HLS stream variable=v750 depth=17
  #pragma HLS stream variable=v751 depth=17
  #pragma HLS stream variable=v752 depth=17
  #pragma HLS stream variable=v753 depth=17
  #pragma HLS array_partition variable=v754 complete dim=1
  #pragma HLS array_partition variable=v754 complete dim=2

  int8_t v757 = v754[v755][v756];	// L815
  int8_t w30;	// L816
  w30 = v757;	// L817
  l_step_m30: for (int m30 = 0; m30 < 16; m30++) {	// L818
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

void PE_kernel_ws_gemm_15_1(
  hls::stream< int8_t > &v775 /* v775[16] */,
  hls::stream< int32_t > &v776 /* v776[16] */,
  hls::stream< int8_t > &v777 /* v777[16] */,
  hls::stream< int32_t > &v778 /* v778[16] */,
  int8_t v779[16][16],
  int v780,
  int v781
) {	// L841
  #pragma HLS stream variable=v775 depth=17
  #pragma HLS stream variable=v776 depth=17
  #pragma HLS stream variable=v777 depth=17
  #pragma HLS stream variable=v778 depth=17
  #pragma HLS array_partition variable=v779 complete dim=1
  #pragma HLS array_partition variable=v779 complete dim=2

  int8_t v782 = v779[v780][v781];	// L842
  int8_t w31;	// L843
  w31 = v782;	// L844
  l_step_m31: for (int m31 = 0; m31 < 16; m31++) {	// L845
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

void PE_kernel_ws_gemm_0_2(
  hls::stream< int8_t > &v800 /* v800[16] */,
  hls::stream< int32_t > &v801 /* v801[16] */,
  hls::stream< int8_t > &v802 /* v802[16] */,
  hls::stream< int32_t > &v803 /* v803[16] */,
  int8_t v804[16][16],
  int v805,
  int v806
) {	// L868
  #pragma HLS stream variable=v800 depth=17
  #pragma HLS stream variable=v801 depth=17
  #pragma HLS stream variable=v802 depth=17
  #pragma HLS stream variable=v803 depth=17
  #pragma HLS array_partition variable=v804 complete dim=1
  #pragma HLS array_partition variable=v804 complete dim=2

  int8_t v807 = v804[v805][v806];	// L869
  int8_t w32;	// L870
  w32 = v807;	// L871
  l_step_m32: for (int m32 = 0; m32 < 16; m32++) {	// L872
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

void PE_kernel_ws_gemm_1_2(
  hls::stream< int8_t > &v825 /* v825[16] */,
  hls::stream< int32_t > &v826 /* v826[16] */,
  hls::stream< int8_t > &v827 /* v827[16] */,
  hls::stream< int32_t > &v828 /* v828[16] */,
  int8_t v829[16][16],
  int v830,
  int v831
) {	// L895
  #pragma HLS stream variable=v825 depth=17
  #pragma HLS stream variable=v826 depth=17
  #pragma HLS stream variable=v827 depth=17
  #pragma HLS stream variable=v828 depth=17
  #pragma HLS array_partition variable=v829 complete dim=1
  #pragma HLS array_partition variable=v829 complete dim=2

  int8_t v832 = v829[v830][v831];	// L896
  int8_t w33;	// L897
  w33 = v832;	// L898
  l_step_m33: for (int m33 = 0; m33 < 16; m33++) {	// L899
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

void PE_kernel_ws_gemm_2_2(
  hls::stream< int8_t > &v850 /* v850[16] */,
  hls::stream< int32_t > &v851 /* v851[16] */,
  hls::stream< int8_t > &v852 /* v852[16] */,
  hls::stream< int32_t > &v853 /* v853[16] */,
  int8_t v854[16][16],
  int v855,
  int v856
) {	// L922
  #pragma HLS stream variable=v850 depth=17
  #pragma HLS stream variable=v851 depth=17
  #pragma HLS stream variable=v852 depth=17
  #pragma HLS stream variable=v853 depth=17
  #pragma HLS array_partition variable=v854 complete dim=1
  #pragma HLS array_partition variable=v854 complete dim=2

  int8_t v857 = v854[v855][v856];	// L923
  int8_t w34;	// L924
  w34 = v857;	// L925
  l_step_m34: for (int m34 = 0; m34 < 16; m34++) {	// L926
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

void PE_kernel_ws_gemm_3_2(
  hls::stream< int8_t > &v875 /* v875[16] */,
  hls::stream< int32_t > &v876 /* v876[16] */,
  hls::stream< int8_t > &v877 /* v877[16] */,
  hls::stream< int32_t > &v878 /* v878[16] */,
  int8_t v879[16][16],
  int v880,
  int v881
) {	// L949
  #pragma HLS stream variable=v875 depth=17
  #pragma HLS stream variable=v876 depth=17
  #pragma HLS stream variable=v877 depth=17
  #pragma HLS stream variable=v878 depth=17
  #pragma HLS array_partition variable=v879 complete dim=1
  #pragma HLS array_partition variable=v879 complete dim=2

  int8_t v882 = v879[v880][v881];	// L950
  int8_t w35;	// L951
  w35 = v882;	// L952
  l_step_m35: for (int m35 = 0; m35 < 16; m35++) {	// L953
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

void PE_kernel_ws_gemm_4_2(
  hls::stream< int8_t > &v900 /* v900[16] */,
  hls::stream< int32_t > &v901 /* v901[16] */,
  hls::stream< int8_t > &v902 /* v902[16] */,
  hls::stream< int32_t > &v903 /* v903[16] */,
  int8_t v904[16][16],
  int v905,
  int v906
) {	// L976
  #pragma HLS stream variable=v900 depth=17
  #pragma HLS stream variable=v901 depth=17
  #pragma HLS stream variable=v902 depth=17
  #pragma HLS stream variable=v903 depth=17
  #pragma HLS array_partition variable=v904 complete dim=1
  #pragma HLS array_partition variable=v904 complete dim=2

  int8_t v907 = v904[v905][v906];	// L977
  int8_t w36;	// L978
  w36 = v907;	// L979
  l_step_m36: for (int m36 = 0; m36 < 16; m36++) {	// L980
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

void PE_kernel_ws_gemm_5_2(
  hls::stream< int8_t > &v925 /* v925[16] */,
  hls::stream< int32_t > &v926 /* v926[16] */,
  hls::stream< int8_t > &v927 /* v927[16] */,
  hls::stream< int32_t > &v928 /* v928[16] */,
  int8_t v929[16][16],
  int v930,
  int v931
) {	// L1003
  #pragma HLS stream variable=v925 depth=17
  #pragma HLS stream variable=v926 depth=17
  #pragma HLS stream variable=v927 depth=17
  #pragma HLS stream variable=v928 depth=17
  #pragma HLS array_partition variable=v929 complete dim=1
  #pragma HLS array_partition variable=v929 complete dim=2

  int8_t v932 = v929[v930][v931];	// L1004
  int8_t w37;	// L1005
  w37 = v932;	// L1006
  l_step_m37: for (int m37 = 0; m37 < 16; m37++) {	// L1007
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

void PE_kernel_ws_gemm_6_2(
  hls::stream< int8_t > &v950 /* v950[16] */,
  hls::stream< int32_t > &v951 /* v951[16] */,
  hls::stream< int8_t > &v952 /* v952[16] */,
  hls::stream< int32_t > &v953 /* v953[16] */,
  int8_t v954[16][16],
  int v955,
  int v956
) {	// L1030
  #pragma HLS stream variable=v950 depth=17
  #pragma HLS stream variable=v951 depth=17
  #pragma HLS stream variable=v952 depth=17
  #pragma HLS stream variable=v953 depth=17
  #pragma HLS array_partition variable=v954 complete dim=1
  #pragma HLS array_partition variable=v954 complete dim=2

  int8_t v957 = v954[v955][v956];	// L1031
  int8_t w38;	// L1032
  w38 = v957;	// L1033
  l_step_m38: for (int m38 = 0; m38 < 16; m38++) {	// L1034
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

void PE_kernel_ws_gemm_7_2(
  hls::stream< int8_t > &v975 /* v975[16] */,
  hls::stream< int32_t > &v976 /* v976[16] */,
  hls::stream< int8_t > &v977 /* v977[16] */,
  hls::stream< int32_t > &v978 /* v978[16] */,
  int8_t v979[16][16],
  int v980,
  int v981
) {	// L1057
  #pragma HLS stream variable=v975 depth=17
  #pragma HLS stream variable=v976 depth=17
  #pragma HLS stream variable=v977 depth=17
  #pragma HLS stream variable=v978 depth=17
  #pragma HLS array_partition variable=v979 complete dim=1
  #pragma HLS array_partition variable=v979 complete dim=2

  int8_t v982 = v979[v980][v981];	// L1058
  int8_t w39;	// L1059
  w39 = v982;	// L1060
  l_step_m39: for (int m39 = 0; m39 < 16; m39++) {	// L1061
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

void PE_kernel_ws_gemm_8_2(
  hls::stream< int8_t > &v1000 /* v1000[16] */,
  hls::stream< int32_t > &v1001 /* v1001[16] */,
  hls::stream< int8_t > &v1002 /* v1002[16] */,
  hls::stream< int32_t > &v1003 /* v1003[16] */,
  int8_t v1004[16][16],
  int v1005,
  int v1006
) {	// L1084
  #pragma HLS stream variable=v1000 depth=17
  #pragma HLS stream variable=v1001 depth=17
  #pragma HLS stream variable=v1002 depth=17
  #pragma HLS stream variable=v1003 depth=17
  #pragma HLS array_partition variable=v1004 complete dim=1
  #pragma HLS array_partition variable=v1004 complete dim=2

  int8_t v1007 = v1004[v1005][v1006];	// L1085
  int8_t w40;	// L1086
  w40 = v1007;	// L1087
  l_step_m40: for (int m40 = 0; m40 < 16; m40++) {	// L1088
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

void PE_kernel_ws_gemm_9_2(
  hls::stream< int8_t > &v1025 /* v1025[16] */,
  hls::stream< int32_t > &v1026 /* v1026[16] */,
  hls::stream< int8_t > &v1027 /* v1027[16] */,
  hls::stream< int32_t > &v1028 /* v1028[16] */,
  int8_t v1029[16][16],
  int v1030,
  int v1031
) {	// L1111
  #pragma HLS stream variable=v1025 depth=17
  #pragma HLS stream variable=v1026 depth=17
  #pragma HLS stream variable=v1027 depth=17
  #pragma HLS stream variable=v1028 depth=17
  #pragma HLS array_partition variable=v1029 complete dim=1
  #pragma HLS array_partition variable=v1029 complete dim=2

  int8_t v1032 = v1029[v1030][v1031];	// L1112
  int8_t w41;	// L1113
  w41 = v1032;	// L1114
  l_step_m41: for (int m41 = 0; m41 < 16; m41++) {	// L1115
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

void PE_kernel_ws_gemm_10_2(
  hls::stream< int8_t > &v1050 /* v1050[16] */,
  hls::stream< int32_t > &v1051 /* v1051[16] */,
  hls::stream< int8_t > &v1052 /* v1052[16] */,
  hls::stream< int32_t > &v1053 /* v1053[16] */,
  int8_t v1054[16][16],
  int v1055,
  int v1056
) {	// L1138
  #pragma HLS stream variable=v1050 depth=17
  #pragma HLS stream variable=v1051 depth=17
  #pragma HLS stream variable=v1052 depth=17
  #pragma HLS stream variable=v1053 depth=17
  #pragma HLS array_partition variable=v1054 complete dim=1
  #pragma HLS array_partition variable=v1054 complete dim=2

  int8_t v1057 = v1054[v1055][v1056];	// L1139
  int8_t w42;	// L1140
  w42 = v1057;	// L1141
  l_step_m42: for (int m42 = 0; m42 < 16; m42++) {	// L1142
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

void PE_kernel_ws_gemm_11_2(
  hls::stream< int8_t > &v1075 /* v1075[16] */,
  hls::stream< int32_t > &v1076 /* v1076[16] */,
  hls::stream< int8_t > &v1077 /* v1077[16] */,
  hls::stream< int32_t > &v1078 /* v1078[16] */,
  int8_t v1079[16][16],
  int v1080,
  int v1081
) {	// L1165
  #pragma HLS stream variable=v1075 depth=17
  #pragma HLS stream variable=v1076 depth=17
  #pragma HLS stream variable=v1077 depth=17
  #pragma HLS stream variable=v1078 depth=17
  #pragma HLS array_partition variable=v1079 complete dim=1
  #pragma HLS array_partition variable=v1079 complete dim=2

  int8_t v1082 = v1079[v1080][v1081];	// L1166
  int8_t w43;	// L1167
  w43 = v1082;	// L1168
  l_step_m43: for (int m43 = 0; m43 < 16; m43++) {	// L1169
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

void PE_kernel_ws_gemm_12_2(
  hls::stream< int8_t > &v1100 /* v1100[16] */,
  hls::stream< int32_t > &v1101 /* v1101[16] */,
  hls::stream< int8_t > &v1102 /* v1102[16] */,
  hls::stream< int32_t > &v1103 /* v1103[16] */,
  int8_t v1104[16][16],
  int v1105,
  int v1106
) {	// L1192
  #pragma HLS stream variable=v1100 depth=17
  #pragma HLS stream variable=v1101 depth=17
  #pragma HLS stream variable=v1102 depth=17
  #pragma HLS stream variable=v1103 depth=17
  #pragma HLS array_partition variable=v1104 complete dim=1
  #pragma HLS array_partition variable=v1104 complete dim=2

  int8_t v1107 = v1104[v1105][v1106];	// L1193
  int8_t w44;	// L1194
  w44 = v1107;	// L1195
  l_step_m44: for (int m44 = 0; m44 < 16; m44++) {	// L1196
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

void PE_kernel_ws_gemm_13_2(
  hls::stream< int8_t > &v1125 /* v1125[16] */,
  hls::stream< int32_t > &v1126 /* v1126[16] */,
  hls::stream< int8_t > &v1127 /* v1127[16] */,
  hls::stream< int32_t > &v1128 /* v1128[16] */,
  int8_t v1129[16][16],
  int v1130,
  int v1131
) {	// L1219
  #pragma HLS stream variable=v1125 depth=17
  #pragma HLS stream variable=v1126 depth=17
  #pragma HLS stream variable=v1127 depth=17
  #pragma HLS stream variable=v1128 depth=17
  #pragma HLS array_partition variable=v1129 complete dim=1
  #pragma HLS array_partition variable=v1129 complete dim=2

  int8_t v1132 = v1129[v1130][v1131];	// L1220
  int8_t w45;	// L1221
  w45 = v1132;	// L1222
  l_step_m45: for (int m45 = 0; m45 < 16; m45++) {	// L1223
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

void PE_kernel_ws_gemm_14_2(
  hls::stream< int8_t > &v1150 /* v1150[16] */,
  hls::stream< int32_t > &v1151 /* v1151[16] */,
  hls::stream< int8_t > &v1152 /* v1152[16] */,
  hls::stream< int32_t > &v1153 /* v1153[16] */,
  int8_t v1154[16][16],
  int v1155,
  int v1156
) {	// L1246
  #pragma HLS stream variable=v1150 depth=17
  #pragma HLS stream variable=v1151 depth=17
  #pragma HLS stream variable=v1152 depth=17
  #pragma HLS stream variable=v1153 depth=17
  #pragma HLS array_partition variable=v1154 complete dim=1
  #pragma HLS array_partition variable=v1154 complete dim=2

  int8_t v1157 = v1154[v1155][v1156];	// L1247
  int8_t w46;	// L1248
  w46 = v1157;	// L1249
  l_step_m46: for (int m46 = 0; m46 < 16; m46++) {	// L1250
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

void PE_kernel_ws_gemm_15_2(
  hls::stream< int8_t > &v1175 /* v1175[16] */,
  hls::stream< int32_t > &v1176 /* v1176[16] */,
  hls::stream< int8_t > &v1177 /* v1177[16] */,
  hls::stream< int32_t > &v1178 /* v1178[16] */,
  int8_t v1179[16][16],
  int v1180,
  int v1181
) {	// L1273
  #pragma HLS stream variable=v1175 depth=17
  #pragma HLS stream variable=v1176 depth=17
  #pragma HLS stream variable=v1177 depth=17
  #pragma HLS stream variable=v1178 depth=17
  #pragma HLS array_partition variable=v1179 complete dim=1
  #pragma HLS array_partition variable=v1179 complete dim=2

  int8_t v1182 = v1179[v1180][v1181];	// L1274
  int8_t w47;	// L1275
  w47 = v1182;	// L1276
  l_step_m47: for (int m47 = 0; m47 < 16; m47++) {	// L1277
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

void PE_kernel_ws_gemm_0_3(
  hls::stream< int8_t > &v1200 /* v1200[16] */,
  hls::stream< int32_t > &v1201 /* v1201[16] */,
  hls::stream< int8_t > &v1202 /* v1202[16] */,
  hls::stream< int32_t > &v1203 /* v1203[16] */,
  int8_t v1204[16][16],
  int v1205,
  int v1206
) {	// L1300
  #pragma HLS stream variable=v1200 depth=17
  #pragma HLS stream variable=v1201 depth=17
  #pragma HLS stream variable=v1202 depth=17
  #pragma HLS stream variable=v1203 depth=17
  #pragma HLS array_partition variable=v1204 complete dim=1
  #pragma HLS array_partition variable=v1204 complete dim=2

  int8_t v1207 = v1204[v1205][v1206];	// L1301
  int8_t w48;	// L1302
  w48 = v1207;	// L1303
  l_step_m48: for (int m48 = 0; m48 < 16; m48++) {	// L1304
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

void PE_kernel_ws_gemm_1_3(
  hls::stream< int8_t > &v1225 /* v1225[16] */,
  hls::stream< int32_t > &v1226 /* v1226[16] */,
  hls::stream< int8_t > &v1227 /* v1227[16] */,
  hls::stream< int32_t > &v1228 /* v1228[16] */,
  int8_t v1229[16][16],
  int v1230,
  int v1231
) {	// L1327
  #pragma HLS stream variable=v1225 depth=17
  #pragma HLS stream variable=v1226 depth=17
  #pragma HLS stream variable=v1227 depth=17
  #pragma HLS stream variable=v1228 depth=17
  #pragma HLS array_partition variable=v1229 complete dim=1
  #pragma HLS array_partition variable=v1229 complete dim=2

  int8_t v1232 = v1229[v1230][v1231];	// L1328
  int8_t w49;	// L1329
  w49 = v1232;	// L1330
  l_step_m49: for (int m49 = 0; m49 < 16; m49++) {	// L1331
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

void PE_kernel_ws_gemm_2_3(
  hls::stream< int8_t > &v1250 /* v1250[16] */,
  hls::stream< int32_t > &v1251 /* v1251[16] */,
  hls::stream< int8_t > &v1252 /* v1252[16] */,
  hls::stream< int32_t > &v1253 /* v1253[16] */,
  int8_t v1254[16][16],
  int v1255,
  int v1256
) {	// L1354
  #pragma HLS stream variable=v1250 depth=17
  #pragma HLS stream variable=v1251 depth=17
  #pragma HLS stream variable=v1252 depth=17
  #pragma HLS stream variable=v1253 depth=17
  #pragma HLS array_partition variable=v1254 complete dim=1
  #pragma HLS array_partition variable=v1254 complete dim=2

  int8_t v1257 = v1254[v1255][v1256];	// L1355
  int8_t w50;	// L1356
  w50 = v1257;	// L1357
  l_step_m50: for (int m50 = 0; m50 < 16; m50++) {	// L1358
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

void PE_kernel_ws_gemm_3_3(
  hls::stream< int8_t > &v1275 /* v1275[16] */,
  hls::stream< int32_t > &v1276 /* v1276[16] */,
  hls::stream< int8_t > &v1277 /* v1277[16] */,
  hls::stream< int32_t > &v1278 /* v1278[16] */,
  int8_t v1279[16][16],
  int v1280,
  int v1281
) {	// L1381
  #pragma HLS stream variable=v1275 depth=17
  #pragma HLS stream variable=v1276 depth=17
  #pragma HLS stream variable=v1277 depth=17
  #pragma HLS stream variable=v1278 depth=17
  #pragma HLS array_partition variable=v1279 complete dim=1
  #pragma HLS array_partition variable=v1279 complete dim=2

  int8_t v1282 = v1279[v1280][v1281];	// L1382
  int8_t w51;	// L1383
  w51 = v1282;	// L1384
  l_step_m51: for (int m51 = 0; m51 < 16; m51++) {	// L1385
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

void PE_kernel_ws_gemm_4_3(
  hls::stream< int8_t > &v1300 /* v1300[16] */,
  hls::stream< int32_t > &v1301 /* v1301[16] */,
  hls::stream< int8_t > &v1302 /* v1302[16] */,
  hls::stream< int32_t > &v1303 /* v1303[16] */,
  int8_t v1304[16][16],
  int v1305,
  int v1306
) {	// L1408
  #pragma HLS stream variable=v1300 depth=17
  #pragma HLS stream variable=v1301 depth=17
  #pragma HLS stream variable=v1302 depth=17
  #pragma HLS stream variable=v1303 depth=17
  #pragma HLS array_partition variable=v1304 complete dim=1
  #pragma HLS array_partition variable=v1304 complete dim=2

  int8_t v1307 = v1304[v1305][v1306];	// L1409
  int8_t w52;	// L1410
  w52 = v1307;	// L1411
  l_step_m52: for (int m52 = 0; m52 < 16; m52++) {	// L1412
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

void PE_kernel_ws_gemm_5_3(
  hls::stream< int8_t > &v1325 /* v1325[16] */,
  hls::stream< int32_t > &v1326 /* v1326[16] */,
  hls::stream< int8_t > &v1327 /* v1327[16] */,
  hls::stream< int32_t > &v1328 /* v1328[16] */,
  int8_t v1329[16][16],
  int v1330,
  int v1331
) {	// L1435
  #pragma HLS stream variable=v1325 depth=17
  #pragma HLS stream variable=v1326 depth=17
  #pragma HLS stream variable=v1327 depth=17
  #pragma HLS stream variable=v1328 depth=17
  #pragma HLS array_partition variable=v1329 complete dim=1
  #pragma HLS array_partition variable=v1329 complete dim=2

  int8_t v1332 = v1329[v1330][v1331];	// L1436
  int8_t w53;	// L1437
  w53 = v1332;	// L1438
  l_step_m53: for (int m53 = 0; m53 < 16; m53++) {	// L1439
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

void PE_kernel_ws_gemm_6_3(
  hls::stream< int8_t > &v1350 /* v1350[16] */,
  hls::stream< int32_t > &v1351 /* v1351[16] */,
  hls::stream< int8_t > &v1352 /* v1352[16] */,
  hls::stream< int32_t > &v1353 /* v1353[16] */,
  int8_t v1354[16][16],
  int v1355,
  int v1356
) {	// L1462
  #pragma HLS stream variable=v1350 depth=17
  #pragma HLS stream variable=v1351 depth=17
  #pragma HLS stream variable=v1352 depth=17
  #pragma HLS stream variable=v1353 depth=17
  #pragma HLS array_partition variable=v1354 complete dim=1
  #pragma HLS array_partition variable=v1354 complete dim=2

  int8_t v1357 = v1354[v1355][v1356];	// L1463
  int8_t w54;	// L1464
  w54 = v1357;	// L1465
  l_step_m54: for (int m54 = 0; m54 < 16; m54++) {	// L1466
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

void PE_kernel_ws_gemm_7_3(
  hls::stream< int8_t > &v1375 /* v1375[16] */,
  hls::stream< int32_t > &v1376 /* v1376[16] */,
  hls::stream< int8_t > &v1377 /* v1377[16] */,
  hls::stream< int32_t > &v1378 /* v1378[16] */,
  int8_t v1379[16][16],
  int v1380,
  int v1381
) {	// L1489
  #pragma HLS stream variable=v1375 depth=17
  #pragma HLS stream variable=v1376 depth=17
  #pragma HLS stream variable=v1377 depth=17
  #pragma HLS stream variable=v1378 depth=17
  #pragma HLS array_partition variable=v1379 complete dim=1
  #pragma HLS array_partition variable=v1379 complete dim=2

  int8_t v1382 = v1379[v1380][v1381];	// L1490
  int8_t w55;	// L1491
  w55 = v1382;	// L1492
  l_step_m55: for (int m55 = 0; m55 < 16; m55++) {	// L1493
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

void PE_kernel_ws_gemm_8_3(
  hls::stream< int8_t > &v1400 /* v1400[16] */,
  hls::stream< int32_t > &v1401 /* v1401[16] */,
  hls::stream< int8_t > &v1402 /* v1402[16] */,
  hls::stream< int32_t > &v1403 /* v1403[16] */,
  int8_t v1404[16][16],
  int v1405,
  int v1406
) {	// L1516
  #pragma HLS stream variable=v1400 depth=17
  #pragma HLS stream variable=v1401 depth=17
  #pragma HLS stream variable=v1402 depth=17
  #pragma HLS stream variable=v1403 depth=17
  #pragma HLS array_partition variable=v1404 complete dim=1
  #pragma HLS array_partition variable=v1404 complete dim=2

  int8_t v1407 = v1404[v1405][v1406];	// L1517
  int8_t w56;	// L1518
  w56 = v1407;	// L1519
  l_step_m56: for (int m56 = 0; m56 < 16; m56++) {	// L1520
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

void PE_kernel_ws_gemm_9_3(
  hls::stream< int8_t > &v1425 /* v1425[16] */,
  hls::stream< int32_t > &v1426 /* v1426[16] */,
  hls::stream< int8_t > &v1427 /* v1427[16] */,
  hls::stream< int32_t > &v1428 /* v1428[16] */,
  int8_t v1429[16][16],
  int v1430,
  int v1431
) {	// L1543
  #pragma HLS stream variable=v1425 depth=17
  #pragma HLS stream variable=v1426 depth=17
  #pragma HLS stream variable=v1427 depth=17
  #pragma HLS stream variable=v1428 depth=17
  #pragma HLS array_partition variable=v1429 complete dim=1
  #pragma HLS array_partition variable=v1429 complete dim=2

  int8_t v1432 = v1429[v1430][v1431];	// L1544
  int8_t w57;	// L1545
  w57 = v1432;	// L1546
  l_step_m57: for (int m57 = 0; m57 < 16; m57++) {	// L1547
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

void PE_kernel_ws_gemm_10_3(
  hls::stream< int8_t > &v1450 /* v1450[16] */,
  hls::stream< int32_t > &v1451 /* v1451[16] */,
  hls::stream< int8_t > &v1452 /* v1452[16] */,
  hls::stream< int32_t > &v1453 /* v1453[16] */,
  int8_t v1454[16][16],
  int v1455,
  int v1456
) {	// L1570
  #pragma HLS stream variable=v1450 depth=17
  #pragma HLS stream variable=v1451 depth=17
  #pragma HLS stream variable=v1452 depth=17
  #pragma HLS stream variable=v1453 depth=17
  #pragma HLS array_partition variable=v1454 complete dim=1
  #pragma HLS array_partition variable=v1454 complete dim=2

  int8_t v1457 = v1454[v1455][v1456];	// L1571
  int8_t w58;	// L1572
  w58 = v1457;	// L1573
  l_step_m58: for (int m58 = 0; m58 < 16; m58++) {	// L1574
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

void PE_kernel_ws_gemm_11_3(
  hls::stream< int8_t > &v1475 /* v1475[16] */,
  hls::stream< int32_t > &v1476 /* v1476[16] */,
  hls::stream< int8_t > &v1477 /* v1477[16] */,
  hls::stream< int32_t > &v1478 /* v1478[16] */,
  int8_t v1479[16][16],
  int v1480,
  int v1481
) {	// L1597
  #pragma HLS stream variable=v1475 depth=17
  #pragma HLS stream variable=v1476 depth=17
  #pragma HLS stream variable=v1477 depth=17
  #pragma HLS stream variable=v1478 depth=17
  #pragma HLS array_partition variable=v1479 complete dim=1
  #pragma HLS array_partition variable=v1479 complete dim=2

  int8_t v1482 = v1479[v1480][v1481];	// L1598
  int8_t w59;	// L1599
  w59 = v1482;	// L1600
  l_step_m59: for (int m59 = 0; m59 < 16; m59++) {	// L1601
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

void PE_kernel_ws_gemm_12_3(
  hls::stream< int8_t > &v1500 /* v1500[16] */,
  hls::stream< int32_t > &v1501 /* v1501[16] */,
  hls::stream< int8_t > &v1502 /* v1502[16] */,
  hls::stream< int32_t > &v1503 /* v1503[16] */,
  int8_t v1504[16][16],
  int v1505,
  int v1506
) {	// L1624
  #pragma HLS stream variable=v1500 depth=17
  #pragma HLS stream variable=v1501 depth=17
  #pragma HLS stream variable=v1502 depth=17
  #pragma HLS stream variable=v1503 depth=17
  #pragma HLS array_partition variable=v1504 complete dim=1
  #pragma HLS array_partition variable=v1504 complete dim=2

  int8_t v1507 = v1504[v1505][v1506];	// L1625
  int8_t w60;	// L1626
  w60 = v1507;	// L1627
  l_step_m60: for (int m60 = 0; m60 < 16; m60++) {	// L1628
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

void PE_kernel_ws_gemm_13_3(
  hls::stream< int8_t > &v1525 /* v1525[16] */,
  hls::stream< int32_t > &v1526 /* v1526[16] */,
  hls::stream< int8_t > &v1527 /* v1527[16] */,
  hls::stream< int32_t > &v1528 /* v1528[16] */,
  int8_t v1529[16][16],
  int v1530,
  int v1531
) {	// L1651
  #pragma HLS stream variable=v1525 depth=17
  #pragma HLS stream variable=v1526 depth=17
  #pragma HLS stream variable=v1527 depth=17
  #pragma HLS stream variable=v1528 depth=17
  #pragma HLS array_partition variable=v1529 complete dim=1
  #pragma HLS array_partition variable=v1529 complete dim=2

  int8_t v1532 = v1529[v1530][v1531];	// L1652
  int8_t w61;	// L1653
  w61 = v1532;	// L1654
  l_step_m61: for (int m61 = 0; m61 < 16; m61++) {	// L1655
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

void PE_kernel_ws_gemm_14_3(
  hls::stream< int8_t > &v1550 /* v1550[16] */,
  hls::stream< int32_t > &v1551 /* v1551[16] */,
  hls::stream< int8_t > &v1552 /* v1552[16] */,
  hls::stream< int32_t > &v1553 /* v1553[16] */,
  int8_t v1554[16][16],
  int v1555,
  int v1556
) {	// L1678
  #pragma HLS stream variable=v1550 depth=17
  #pragma HLS stream variable=v1551 depth=17
  #pragma HLS stream variable=v1552 depth=17
  #pragma HLS stream variable=v1553 depth=17
  #pragma HLS array_partition variable=v1554 complete dim=1
  #pragma HLS array_partition variable=v1554 complete dim=2

  int8_t v1557 = v1554[v1555][v1556];	// L1679
  int8_t w62;	// L1680
  w62 = v1557;	// L1681
  l_step_m62: for (int m62 = 0; m62 < 16; m62++) {	// L1682
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

void PE_kernel_ws_gemm_15_3(
  hls::stream< int8_t > &v1575 /* v1575[16] */,
  hls::stream< int32_t > &v1576 /* v1576[16] */,
  hls::stream< int8_t > &v1577 /* v1577[16] */,
  hls::stream< int32_t > &v1578 /* v1578[16] */,
  int8_t v1579[16][16],
  int v1580,
  int v1581
) {	// L1705
  #pragma HLS stream variable=v1575 depth=17
  #pragma HLS stream variable=v1576 depth=17
  #pragma HLS stream variable=v1577 depth=17
  #pragma HLS stream variable=v1578 depth=17
  #pragma HLS array_partition variable=v1579 complete dim=1
  #pragma HLS array_partition variable=v1579 complete dim=2

  int8_t v1582 = v1579[v1580][v1581];	// L1706
  int8_t w63;	// L1707
  w63 = v1582;	// L1708
  l_step_m63: for (int m63 = 0; m63 < 16; m63++) {	// L1709
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

void PE_kernel_ws_gemm_0_4(
  hls::stream< int8_t > &v1600 /* v1600[16] */,
  hls::stream< int32_t > &v1601 /* v1601[16] */,
  hls::stream< int8_t > &v1602 /* v1602[16] */,
  hls::stream< int32_t > &v1603 /* v1603[16] */,
  int8_t v1604[16][16],
  int v1605,
  int v1606
) {	// L1732
  #pragma HLS stream variable=v1600 depth=17
  #pragma HLS stream variable=v1601 depth=17
  #pragma HLS stream variable=v1602 depth=17
  #pragma HLS stream variable=v1603 depth=17
  #pragma HLS array_partition variable=v1604 complete dim=1
  #pragma HLS array_partition variable=v1604 complete dim=2

  int8_t v1607 = v1604[v1605][v1606];	// L1733
  int8_t w64;	// L1734
  w64 = v1607;	// L1735
  l_step_m64: for (int m64 = 0; m64 < 16; m64++) {	// L1736
  #pragma HLS pipeline II=1
    int8_t v1610 = v1600.read(); // v1600[m64];	// L1737
    int8_t a64;	// L1738
    a64 = v1610;	// L1739
    int32_t v1612 = v1601.read(); // v1601[m64];	// L1740
    int32_t p64;	// L1741
    p64 = v1612;	// L1742
    int32_t v1614 = p64;	// L1743
    int8_t v1615 = a64;	// L1744
    int8_t v1616 = w64;	// L1745
    int16_t v1617 = v1615;	// L1746
    int16_t v1618 = v1616;	// L1747
    int16_t v1619 = v1617 * v1618;	// L1748
    ap_int<33> v1620 = v1614;	// L1749
    ap_int<33> v1621 = v1619;	// L1750
    ap_int<33> v1622 = v1620 + v1621;	// L1751
    int32_t v1623 = v1622;	// L1752
    v1603.write(v1623); // v1603[m64] = v1623;	// L1753
    int8_t v1624 = a64;	// L1754
    v1602.write(v1624); // v1602[m64] = v1624;	// L1755
  }
}

void PE_kernel_ws_gemm_1_4(
  hls::stream< int8_t > &v1625 /* v1625[16] */,
  hls::stream< int32_t > &v1626 /* v1626[16] */,
  hls::stream< int8_t > &v1627 /* v1627[16] */,
  hls::stream< int32_t > &v1628 /* v1628[16] */,
  int8_t v1629[16][16],
  int v1630,
  int v1631
) {	// L1759
  #pragma HLS stream variable=v1625 depth=17
  #pragma HLS stream variable=v1626 depth=17
  #pragma HLS stream variable=v1627 depth=17
  #pragma HLS stream variable=v1628 depth=17
  #pragma HLS array_partition variable=v1629 complete dim=1
  #pragma HLS array_partition variable=v1629 complete dim=2

  int8_t v1632 = v1629[v1630][v1631];	// L1760
  int8_t w65;	// L1761
  w65 = v1632;	// L1762
  l_step_m65: for (int m65 = 0; m65 < 16; m65++) {	// L1763
  #pragma HLS pipeline II=1
    int8_t v1635 = v1625.read(); // v1625[m65];	// L1764
    int8_t a65;	// L1765
    a65 = v1635;	// L1766
    int32_t v1637 = v1626.read(); // v1626[m65];	// L1767
    int32_t p65;	// L1768
    p65 = v1637;	// L1769
    int32_t v1639 = p65;	// L1770
    int8_t v1640 = a65;	// L1771
    int8_t v1641 = w65;	// L1772
    int16_t v1642 = v1640;	// L1773
    int16_t v1643 = v1641;	// L1774
    int16_t v1644 = v1642 * v1643;	// L1775
    ap_int<33> v1645 = v1639;	// L1776
    ap_int<33> v1646 = v1644;	// L1777
    ap_int<33> v1647 = v1645 + v1646;	// L1778
    int32_t v1648 = v1647;	// L1779
    v1628.write(v1648); // v1628[m65] = v1648;	// L1780
    int8_t v1649 = a65;	// L1781
    v1627.write(v1649); // v1627[m65] = v1649;	// L1782
  }
}

void PE_kernel_ws_gemm_2_4(
  hls::stream< int8_t > &v1650 /* v1650[16] */,
  hls::stream< int32_t > &v1651 /* v1651[16] */,
  hls::stream< int8_t > &v1652 /* v1652[16] */,
  hls::stream< int32_t > &v1653 /* v1653[16] */,
  int8_t v1654[16][16],
  int v1655,
  int v1656
) {	// L1786
  #pragma HLS stream variable=v1650 depth=17
  #pragma HLS stream variable=v1651 depth=17
  #pragma HLS stream variable=v1652 depth=17
  #pragma HLS stream variable=v1653 depth=17
  #pragma HLS array_partition variable=v1654 complete dim=1
  #pragma HLS array_partition variable=v1654 complete dim=2

  int8_t v1657 = v1654[v1655][v1656];	// L1787
  int8_t w66;	// L1788
  w66 = v1657;	// L1789
  l_step_m66: for (int m66 = 0; m66 < 16; m66++) {	// L1790
  #pragma HLS pipeline II=1
    int8_t v1660 = v1650.read(); // v1650[m66];	// L1791
    int8_t a66;	// L1792
    a66 = v1660;	// L1793
    int32_t v1662 = v1651.read(); // v1651[m66];	// L1794
    int32_t p66;	// L1795
    p66 = v1662;	// L1796
    int32_t v1664 = p66;	// L1797
    int8_t v1665 = a66;	// L1798
    int8_t v1666 = w66;	// L1799
    int16_t v1667 = v1665;	// L1800
    int16_t v1668 = v1666;	// L1801
    int16_t v1669 = v1667 * v1668;	// L1802
    ap_int<33> v1670 = v1664;	// L1803
    ap_int<33> v1671 = v1669;	// L1804
    ap_int<33> v1672 = v1670 + v1671;	// L1805
    int32_t v1673 = v1672;	// L1806
    v1653.write(v1673); // v1653[m66] = v1673;	// L1807
    int8_t v1674 = a66;	// L1808
    v1652.write(v1674); // v1652[m66] = v1674;	// L1809
  }
}

void PE_kernel_ws_gemm_3_4(
  hls::stream< int8_t > &v1675 /* v1675[16] */,
  hls::stream< int32_t > &v1676 /* v1676[16] */,
  hls::stream< int8_t > &v1677 /* v1677[16] */,
  hls::stream< int32_t > &v1678 /* v1678[16] */,
  int8_t v1679[16][16],
  int v1680,
  int v1681
) {	// L1813
  #pragma HLS stream variable=v1675 depth=17
  #pragma HLS stream variable=v1676 depth=17
  #pragma HLS stream variable=v1677 depth=17
  #pragma HLS stream variable=v1678 depth=17
  #pragma HLS array_partition variable=v1679 complete dim=1
  #pragma HLS array_partition variable=v1679 complete dim=2

  int8_t v1682 = v1679[v1680][v1681];	// L1814
  int8_t w67;	// L1815
  w67 = v1682;	// L1816
  l_step_m67: for (int m67 = 0; m67 < 16; m67++) {	// L1817
  #pragma HLS pipeline II=1
    int8_t v1685 = v1675.read(); // v1675[m67];	// L1818
    int8_t a67;	// L1819
    a67 = v1685;	// L1820
    int32_t v1687 = v1676.read(); // v1676[m67];	// L1821
    int32_t p67;	// L1822
    p67 = v1687;	// L1823
    int32_t v1689 = p67;	// L1824
    int8_t v1690 = a67;	// L1825
    int8_t v1691 = w67;	// L1826
    int16_t v1692 = v1690;	// L1827
    int16_t v1693 = v1691;	// L1828
    int16_t v1694 = v1692 * v1693;	// L1829
    ap_int<33> v1695 = v1689;	// L1830
    ap_int<33> v1696 = v1694;	// L1831
    ap_int<33> v1697 = v1695 + v1696;	// L1832
    int32_t v1698 = v1697;	// L1833
    v1678.write(v1698); // v1678[m67] = v1698;	// L1834
    int8_t v1699 = a67;	// L1835
    v1677.write(v1699); // v1677[m67] = v1699;	// L1836
  }
}

void PE_kernel_ws_gemm_4_4(
  hls::stream< int8_t > &v1700 /* v1700[16] */,
  hls::stream< int32_t > &v1701 /* v1701[16] */,
  hls::stream< int8_t > &v1702 /* v1702[16] */,
  hls::stream< int32_t > &v1703 /* v1703[16] */,
  int8_t v1704[16][16],
  int v1705,
  int v1706
) {	// L1840
  #pragma HLS stream variable=v1700 depth=17
  #pragma HLS stream variable=v1701 depth=17
  #pragma HLS stream variable=v1702 depth=17
  #pragma HLS stream variable=v1703 depth=17
  #pragma HLS array_partition variable=v1704 complete dim=1
  #pragma HLS array_partition variable=v1704 complete dim=2

  int8_t v1707 = v1704[v1705][v1706];	// L1841
  int8_t w68;	// L1842
  w68 = v1707;	// L1843
  l_step_m68: for (int m68 = 0; m68 < 16; m68++) {	// L1844
  #pragma HLS pipeline II=1
    int8_t v1710 = v1700.read(); // v1700[m68];	// L1845
    int8_t a68;	// L1846
    a68 = v1710;	// L1847
    int32_t v1712 = v1701.read(); // v1701[m68];	// L1848
    int32_t p68;	// L1849
    p68 = v1712;	// L1850
    int32_t v1714 = p68;	// L1851
    int8_t v1715 = a68;	// L1852
    int8_t v1716 = w68;	// L1853
    int16_t v1717 = v1715;	// L1854
    int16_t v1718 = v1716;	// L1855
    int16_t v1719 = v1717 * v1718;	// L1856
    ap_int<33> v1720 = v1714;	// L1857
    ap_int<33> v1721 = v1719;	// L1858
    ap_int<33> v1722 = v1720 + v1721;	// L1859
    int32_t v1723 = v1722;	// L1860
    v1703.write(v1723); // v1703[m68] = v1723;	// L1861
    int8_t v1724 = a68;	// L1862
    v1702.write(v1724); // v1702[m68] = v1724;	// L1863
  }
}

void PE_kernel_ws_gemm_5_4(
  hls::stream< int8_t > &v1725 /* v1725[16] */,
  hls::stream< int32_t > &v1726 /* v1726[16] */,
  hls::stream< int8_t > &v1727 /* v1727[16] */,
  hls::stream< int32_t > &v1728 /* v1728[16] */,
  int8_t v1729[16][16],
  int v1730,
  int v1731
) {	// L1867
  #pragma HLS stream variable=v1725 depth=17
  #pragma HLS stream variable=v1726 depth=17
  #pragma HLS stream variable=v1727 depth=17
  #pragma HLS stream variable=v1728 depth=17
  #pragma HLS array_partition variable=v1729 complete dim=1
  #pragma HLS array_partition variable=v1729 complete dim=2

  int8_t v1732 = v1729[v1730][v1731];	// L1868
  int8_t w69;	// L1869
  w69 = v1732;	// L1870
  l_step_m69: for (int m69 = 0; m69 < 16; m69++) {	// L1871
  #pragma HLS pipeline II=1
    int8_t v1735 = v1725.read(); // v1725[m69];	// L1872
    int8_t a69;	// L1873
    a69 = v1735;	// L1874
    int32_t v1737 = v1726.read(); // v1726[m69];	// L1875
    int32_t p69;	// L1876
    p69 = v1737;	// L1877
    int32_t v1739 = p69;	// L1878
    int8_t v1740 = a69;	// L1879
    int8_t v1741 = w69;	// L1880
    int16_t v1742 = v1740;	// L1881
    int16_t v1743 = v1741;	// L1882
    int16_t v1744 = v1742 * v1743;	// L1883
    ap_int<33> v1745 = v1739;	// L1884
    ap_int<33> v1746 = v1744;	// L1885
    ap_int<33> v1747 = v1745 + v1746;	// L1886
    int32_t v1748 = v1747;	// L1887
    v1728.write(v1748); // v1728[m69] = v1748;	// L1888
    int8_t v1749 = a69;	// L1889
    v1727.write(v1749); // v1727[m69] = v1749;	// L1890
  }
}

void PE_kernel_ws_gemm_6_4(
  hls::stream< int8_t > &v1750 /* v1750[16] */,
  hls::stream< int32_t > &v1751 /* v1751[16] */,
  hls::stream< int8_t > &v1752 /* v1752[16] */,
  hls::stream< int32_t > &v1753 /* v1753[16] */,
  int8_t v1754[16][16],
  int v1755,
  int v1756
) {	// L1894
  #pragma HLS stream variable=v1750 depth=17
  #pragma HLS stream variable=v1751 depth=17
  #pragma HLS stream variable=v1752 depth=17
  #pragma HLS stream variable=v1753 depth=17
  #pragma HLS array_partition variable=v1754 complete dim=1
  #pragma HLS array_partition variable=v1754 complete dim=2

  int8_t v1757 = v1754[v1755][v1756];	// L1895
  int8_t w70;	// L1896
  w70 = v1757;	// L1897
  l_step_m70: for (int m70 = 0; m70 < 16; m70++) {	// L1898
  #pragma HLS pipeline II=1
    int8_t v1760 = v1750.read(); // v1750[m70];	// L1899
    int8_t a70;	// L1900
    a70 = v1760;	// L1901
    int32_t v1762 = v1751.read(); // v1751[m70];	// L1902
    int32_t p70;	// L1903
    p70 = v1762;	// L1904
    int32_t v1764 = p70;	// L1905
    int8_t v1765 = a70;	// L1906
    int8_t v1766 = w70;	// L1907
    int16_t v1767 = v1765;	// L1908
    int16_t v1768 = v1766;	// L1909
    int16_t v1769 = v1767 * v1768;	// L1910
    ap_int<33> v1770 = v1764;	// L1911
    ap_int<33> v1771 = v1769;	// L1912
    ap_int<33> v1772 = v1770 + v1771;	// L1913
    int32_t v1773 = v1772;	// L1914
    v1753.write(v1773); // v1753[m70] = v1773;	// L1915
    int8_t v1774 = a70;	// L1916
    v1752.write(v1774); // v1752[m70] = v1774;	// L1917
  }
}

void PE_kernel_ws_gemm_7_4(
  hls::stream< int8_t > &v1775 /* v1775[16] */,
  hls::stream< int32_t > &v1776 /* v1776[16] */,
  hls::stream< int8_t > &v1777 /* v1777[16] */,
  hls::stream< int32_t > &v1778 /* v1778[16] */,
  int8_t v1779[16][16],
  int v1780,
  int v1781
) {	// L1921
  #pragma HLS stream variable=v1775 depth=17
  #pragma HLS stream variable=v1776 depth=17
  #pragma HLS stream variable=v1777 depth=17
  #pragma HLS stream variable=v1778 depth=17
  #pragma HLS array_partition variable=v1779 complete dim=1
  #pragma HLS array_partition variable=v1779 complete dim=2

  int8_t v1782 = v1779[v1780][v1781];	// L1922
  int8_t w71;	// L1923
  w71 = v1782;	// L1924
  l_step_m71: for (int m71 = 0; m71 < 16; m71++) {	// L1925
  #pragma HLS pipeline II=1
    int8_t v1785 = v1775.read(); // v1775[m71];	// L1926
    int8_t a71;	// L1927
    a71 = v1785;	// L1928
    int32_t v1787 = v1776.read(); // v1776[m71];	// L1929
    int32_t p71;	// L1930
    p71 = v1787;	// L1931
    int32_t v1789 = p71;	// L1932
    int8_t v1790 = a71;	// L1933
    int8_t v1791 = w71;	// L1934
    int16_t v1792 = v1790;	// L1935
    int16_t v1793 = v1791;	// L1936
    int16_t v1794 = v1792 * v1793;	// L1937
    ap_int<33> v1795 = v1789;	// L1938
    ap_int<33> v1796 = v1794;	// L1939
    ap_int<33> v1797 = v1795 + v1796;	// L1940
    int32_t v1798 = v1797;	// L1941
    v1778.write(v1798); // v1778[m71] = v1798;	// L1942
    int8_t v1799 = a71;	// L1943
    v1777.write(v1799); // v1777[m71] = v1799;	// L1944
  }
}

void PE_kernel_ws_gemm_8_4(
  hls::stream< int8_t > &v1800 /* v1800[16] */,
  hls::stream< int32_t > &v1801 /* v1801[16] */,
  hls::stream< int8_t > &v1802 /* v1802[16] */,
  hls::stream< int32_t > &v1803 /* v1803[16] */,
  int8_t v1804[16][16],
  int v1805,
  int v1806
) {	// L1948
  #pragma HLS stream variable=v1800 depth=17
  #pragma HLS stream variable=v1801 depth=17
  #pragma HLS stream variable=v1802 depth=17
  #pragma HLS stream variable=v1803 depth=17
  #pragma HLS array_partition variable=v1804 complete dim=1
  #pragma HLS array_partition variable=v1804 complete dim=2

  int8_t v1807 = v1804[v1805][v1806];	// L1949
  int8_t w72;	// L1950
  w72 = v1807;	// L1951
  l_step_m72: for (int m72 = 0; m72 < 16; m72++) {	// L1952
  #pragma HLS pipeline II=1
    int8_t v1810 = v1800.read(); // v1800[m72];	// L1953
    int8_t a72;	// L1954
    a72 = v1810;	// L1955
    int32_t v1812 = v1801.read(); // v1801[m72];	// L1956
    int32_t p72;	// L1957
    p72 = v1812;	// L1958
    int32_t v1814 = p72;	// L1959
    int8_t v1815 = a72;	// L1960
    int8_t v1816 = w72;	// L1961
    int16_t v1817 = v1815;	// L1962
    int16_t v1818 = v1816;	// L1963
    int16_t v1819 = v1817 * v1818;	// L1964
    ap_int<33> v1820 = v1814;	// L1965
    ap_int<33> v1821 = v1819;	// L1966
    ap_int<33> v1822 = v1820 + v1821;	// L1967
    int32_t v1823 = v1822;	// L1968
    v1803.write(v1823); // v1803[m72] = v1823;	// L1969
    int8_t v1824 = a72;	// L1970
    v1802.write(v1824); // v1802[m72] = v1824;	// L1971
  }
}

void PE_kernel_ws_gemm_9_4(
  hls::stream< int8_t > &v1825 /* v1825[16] */,
  hls::stream< int32_t > &v1826 /* v1826[16] */,
  hls::stream< int8_t > &v1827 /* v1827[16] */,
  hls::stream< int32_t > &v1828 /* v1828[16] */,
  int8_t v1829[16][16],
  int v1830,
  int v1831
) {	// L1975
  #pragma HLS stream variable=v1825 depth=17
  #pragma HLS stream variable=v1826 depth=17
  #pragma HLS stream variable=v1827 depth=17
  #pragma HLS stream variable=v1828 depth=17
  #pragma HLS array_partition variable=v1829 complete dim=1
  #pragma HLS array_partition variable=v1829 complete dim=2

  int8_t v1832 = v1829[v1830][v1831];	// L1976
  int8_t w73;	// L1977
  w73 = v1832;	// L1978
  l_step_m73: for (int m73 = 0; m73 < 16; m73++) {	// L1979
  #pragma HLS pipeline II=1
    int8_t v1835 = v1825.read(); // v1825[m73];	// L1980
    int8_t a73;	// L1981
    a73 = v1835;	// L1982
    int32_t v1837 = v1826.read(); // v1826[m73];	// L1983
    int32_t p73;	// L1984
    p73 = v1837;	// L1985
    int32_t v1839 = p73;	// L1986
    int8_t v1840 = a73;	// L1987
    int8_t v1841 = w73;	// L1988
    int16_t v1842 = v1840;	// L1989
    int16_t v1843 = v1841;	// L1990
    int16_t v1844 = v1842 * v1843;	// L1991
    ap_int<33> v1845 = v1839;	// L1992
    ap_int<33> v1846 = v1844;	// L1993
    ap_int<33> v1847 = v1845 + v1846;	// L1994
    int32_t v1848 = v1847;	// L1995
    v1828.write(v1848); // v1828[m73] = v1848;	// L1996
    int8_t v1849 = a73;	// L1997
    v1827.write(v1849); // v1827[m73] = v1849;	// L1998
  }
}

void PE_kernel_ws_gemm_10_4(
  hls::stream< int8_t > &v1850 /* v1850[16] */,
  hls::stream< int32_t > &v1851 /* v1851[16] */,
  hls::stream< int8_t > &v1852 /* v1852[16] */,
  hls::stream< int32_t > &v1853 /* v1853[16] */,
  int8_t v1854[16][16],
  int v1855,
  int v1856
) {	// L2002
  #pragma HLS stream variable=v1850 depth=17
  #pragma HLS stream variable=v1851 depth=17
  #pragma HLS stream variable=v1852 depth=17
  #pragma HLS stream variable=v1853 depth=17
  #pragma HLS array_partition variable=v1854 complete dim=1
  #pragma HLS array_partition variable=v1854 complete dim=2

  int8_t v1857 = v1854[v1855][v1856];	// L2003
  int8_t w74;	// L2004
  w74 = v1857;	// L2005
  l_step_m74: for (int m74 = 0; m74 < 16; m74++) {	// L2006
  #pragma HLS pipeline II=1
    int8_t v1860 = v1850.read(); // v1850[m74];	// L2007
    int8_t a74;	// L2008
    a74 = v1860;	// L2009
    int32_t v1862 = v1851.read(); // v1851[m74];	// L2010
    int32_t p74;	// L2011
    p74 = v1862;	// L2012
    int32_t v1864 = p74;	// L2013
    int8_t v1865 = a74;	// L2014
    int8_t v1866 = w74;	// L2015
    int16_t v1867 = v1865;	// L2016
    int16_t v1868 = v1866;	// L2017
    int16_t v1869 = v1867 * v1868;	// L2018
    ap_int<33> v1870 = v1864;	// L2019
    ap_int<33> v1871 = v1869;	// L2020
    ap_int<33> v1872 = v1870 + v1871;	// L2021
    int32_t v1873 = v1872;	// L2022
    v1853.write(v1873); // v1853[m74] = v1873;	// L2023
    int8_t v1874 = a74;	// L2024
    v1852.write(v1874); // v1852[m74] = v1874;	// L2025
  }
}

void PE_kernel_ws_gemm_11_4(
  hls::stream< int8_t > &v1875 /* v1875[16] */,
  hls::stream< int32_t > &v1876 /* v1876[16] */,
  hls::stream< int8_t > &v1877 /* v1877[16] */,
  hls::stream< int32_t > &v1878 /* v1878[16] */,
  int8_t v1879[16][16],
  int v1880,
  int v1881
) {	// L2029
  #pragma HLS stream variable=v1875 depth=17
  #pragma HLS stream variable=v1876 depth=17
  #pragma HLS stream variable=v1877 depth=17
  #pragma HLS stream variable=v1878 depth=17
  #pragma HLS array_partition variable=v1879 complete dim=1
  #pragma HLS array_partition variable=v1879 complete dim=2

  int8_t v1882 = v1879[v1880][v1881];	// L2030
  int8_t w75;	// L2031
  w75 = v1882;	// L2032
  l_step_m75: for (int m75 = 0; m75 < 16; m75++) {	// L2033
  #pragma HLS pipeline II=1
    int8_t v1885 = v1875.read(); // v1875[m75];	// L2034
    int8_t a75;	// L2035
    a75 = v1885;	// L2036
    int32_t v1887 = v1876.read(); // v1876[m75];	// L2037
    int32_t p75;	// L2038
    p75 = v1887;	// L2039
    int32_t v1889 = p75;	// L2040
    int8_t v1890 = a75;	// L2041
    int8_t v1891 = w75;	// L2042
    int16_t v1892 = v1890;	// L2043
    int16_t v1893 = v1891;	// L2044
    int16_t v1894 = v1892 * v1893;	// L2045
    ap_int<33> v1895 = v1889;	// L2046
    ap_int<33> v1896 = v1894;	// L2047
    ap_int<33> v1897 = v1895 + v1896;	// L2048
    int32_t v1898 = v1897;	// L2049
    v1878.write(v1898); // v1878[m75] = v1898;	// L2050
    int8_t v1899 = a75;	// L2051
    v1877.write(v1899); // v1877[m75] = v1899;	// L2052
  }
}

void PE_kernel_ws_gemm_12_4(
  hls::stream< int8_t > &v1900 /* v1900[16] */,
  hls::stream< int32_t > &v1901 /* v1901[16] */,
  hls::stream< int8_t > &v1902 /* v1902[16] */,
  hls::stream< int32_t > &v1903 /* v1903[16] */,
  int8_t v1904[16][16],
  int v1905,
  int v1906
) {	// L2056
  #pragma HLS stream variable=v1900 depth=17
  #pragma HLS stream variable=v1901 depth=17
  #pragma HLS stream variable=v1902 depth=17
  #pragma HLS stream variable=v1903 depth=17
  #pragma HLS array_partition variable=v1904 complete dim=1
  #pragma HLS array_partition variable=v1904 complete dim=2

  int8_t v1907 = v1904[v1905][v1906];	// L2057
  int8_t w76;	// L2058
  w76 = v1907;	// L2059
  l_step_m76: for (int m76 = 0; m76 < 16; m76++) {	// L2060
  #pragma HLS pipeline II=1
    int8_t v1910 = v1900.read(); // v1900[m76];	// L2061
    int8_t a76;	// L2062
    a76 = v1910;	// L2063
    int32_t v1912 = v1901.read(); // v1901[m76];	// L2064
    int32_t p76;	// L2065
    p76 = v1912;	// L2066
    int32_t v1914 = p76;	// L2067
    int8_t v1915 = a76;	// L2068
    int8_t v1916 = w76;	// L2069
    int16_t v1917 = v1915;	// L2070
    int16_t v1918 = v1916;	// L2071
    int16_t v1919 = v1917 * v1918;	// L2072
    ap_int<33> v1920 = v1914;	// L2073
    ap_int<33> v1921 = v1919;	// L2074
    ap_int<33> v1922 = v1920 + v1921;	// L2075
    int32_t v1923 = v1922;	// L2076
    v1903.write(v1923); // v1903[m76] = v1923;	// L2077
    int8_t v1924 = a76;	// L2078
    v1902.write(v1924); // v1902[m76] = v1924;	// L2079
  }
}

void PE_kernel_ws_gemm_13_4(
  hls::stream< int8_t > &v1925 /* v1925[16] */,
  hls::stream< int32_t > &v1926 /* v1926[16] */,
  hls::stream< int8_t > &v1927 /* v1927[16] */,
  hls::stream< int32_t > &v1928 /* v1928[16] */,
  int8_t v1929[16][16],
  int v1930,
  int v1931
) {	// L2083
  #pragma HLS stream variable=v1925 depth=17
  #pragma HLS stream variable=v1926 depth=17
  #pragma HLS stream variable=v1927 depth=17
  #pragma HLS stream variable=v1928 depth=17
  #pragma HLS array_partition variable=v1929 complete dim=1
  #pragma HLS array_partition variable=v1929 complete dim=2

  int8_t v1932 = v1929[v1930][v1931];	// L2084
  int8_t w77;	// L2085
  w77 = v1932;	// L2086
  l_step_m77: for (int m77 = 0; m77 < 16; m77++) {	// L2087
  #pragma HLS pipeline II=1
    int8_t v1935 = v1925.read(); // v1925[m77];	// L2088
    int8_t a77;	// L2089
    a77 = v1935;	// L2090
    int32_t v1937 = v1926.read(); // v1926[m77];	// L2091
    int32_t p77;	// L2092
    p77 = v1937;	// L2093
    int32_t v1939 = p77;	// L2094
    int8_t v1940 = a77;	// L2095
    int8_t v1941 = w77;	// L2096
    int16_t v1942 = v1940;	// L2097
    int16_t v1943 = v1941;	// L2098
    int16_t v1944 = v1942 * v1943;	// L2099
    ap_int<33> v1945 = v1939;	// L2100
    ap_int<33> v1946 = v1944;	// L2101
    ap_int<33> v1947 = v1945 + v1946;	// L2102
    int32_t v1948 = v1947;	// L2103
    v1928.write(v1948); // v1928[m77] = v1948;	// L2104
    int8_t v1949 = a77;	// L2105
    v1927.write(v1949); // v1927[m77] = v1949;	// L2106
  }
}

void PE_kernel_ws_gemm_14_4(
  hls::stream< int8_t > &v1950 /* v1950[16] */,
  hls::stream< int32_t > &v1951 /* v1951[16] */,
  hls::stream< int8_t > &v1952 /* v1952[16] */,
  hls::stream< int32_t > &v1953 /* v1953[16] */,
  int8_t v1954[16][16],
  int v1955,
  int v1956
) {	// L2110
  #pragma HLS stream variable=v1950 depth=17
  #pragma HLS stream variable=v1951 depth=17
  #pragma HLS stream variable=v1952 depth=17
  #pragma HLS stream variable=v1953 depth=17
  #pragma HLS array_partition variable=v1954 complete dim=1
  #pragma HLS array_partition variable=v1954 complete dim=2

  int8_t v1957 = v1954[v1955][v1956];	// L2111
  int8_t w78;	// L2112
  w78 = v1957;	// L2113
  l_step_m78: for (int m78 = 0; m78 < 16; m78++) {	// L2114
  #pragma HLS pipeline II=1
    int8_t v1960 = v1950.read(); // v1950[m78];	// L2115
    int8_t a78;	// L2116
    a78 = v1960;	// L2117
    int32_t v1962 = v1951.read(); // v1951[m78];	// L2118
    int32_t p78;	// L2119
    p78 = v1962;	// L2120
    int32_t v1964 = p78;	// L2121
    int8_t v1965 = a78;	// L2122
    int8_t v1966 = w78;	// L2123
    int16_t v1967 = v1965;	// L2124
    int16_t v1968 = v1966;	// L2125
    int16_t v1969 = v1967 * v1968;	// L2126
    ap_int<33> v1970 = v1964;	// L2127
    ap_int<33> v1971 = v1969;	// L2128
    ap_int<33> v1972 = v1970 + v1971;	// L2129
    int32_t v1973 = v1972;	// L2130
    v1953.write(v1973); // v1953[m78] = v1973;	// L2131
    int8_t v1974 = a78;	// L2132
    v1952.write(v1974); // v1952[m78] = v1974;	// L2133
  }
}

void PE_kernel_ws_gemm_15_4(
  hls::stream< int8_t > &v1975 /* v1975[16] */,
  hls::stream< int32_t > &v1976 /* v1976[16] */,
  hls::stream< int8_t > &v1977 /* v1977[16] */,
  hls::stream< int32_t > &v1978 /* v1978[16] */,
  int8_t v1979[16][16],
  int v1980,
  int v1981
) {	// L2137
  #pragma HLS stream variable=v1975 depth=17
  #pragma HLS stream variable=v1976 depth=17
  #pragma HLS stream variable=v1977 depth=17
  #pragma HLS stream variable=v1978 depth=17
  #pragma HLS array_partition variable=v1979 complete dim=1
  #pragma HLS array_partition variable=v1979 complete dim=2

  int8_t v1982 = v1979[v1980][v1981];	// L2138
  int8_t w79;	// L2139
  w79 = v1982;	// L2140
  l_step_m79: for (int m79 = 0; m79 < 16; m79++) {	// L2141
  #pragma HLS pipeline II=1
    int8_t v1985 = v1975.read(); // v1975[m79];	// L2142
    int8_t a79;	// L2143
    a79 = v1985;	// L2144
    int32_t v1987 = v1976.read(); // v1976[m79];	// L2145
    int32_t p79;	// L2146
    p79 = v1987;	// L2147
    int32_t v1989 = p79;	// L2148
    int8_t v1990 = a79;	// L2149
    int8_t v1991 = w79;	// L2150
    int16_t v1992 = v1990;	// L2151
    int16_t v1993 = v1991;	// L2152
    int16_t v1994 = v1992 * v1993;	// L2153
    ap_int<33> v1995 = v1989;	// L2154
    ap_int<33> v1996 = v1994;	// L2155
    ap_int<33> v1997 = v1995 + v1996;	// L2156
    int32_t v1998 = v1997;	// L2157
    v1978.write(v1998); // v1978[m79] = v1998;	// L2158
    int8_t v1999 = a79;	// L2159
    v1977.write(v1999); // v1977[m79] = v1999;	// L2160
  }
}

void PE_kernel_ws_gemm_0_5(
  hls::stream< int8_t > &v2000 /* v2000[16] */,
  hls::stream< int32_t > &v2001 /* v2001[16] */,
  hls::stream< int8_t > &v2002 /* v2002[16] */,
  hls::stream< int32_t > &v2003 /* v2003[16] */,
  int8_t v2004[16][16],
  int v2005,
  int v2006
) {	// L2164
  #pragma HLS stream variable=v2000 depth=17
  #pragma HLS stream variable=v2001 depth=17
  #pragma HLS stream variable=v2002 depth=17
  #pragma HLS stream variable=v2003 depth=17
  #pragma HLS array_partition variable=v2004 complete dim=1
  #pragma HLS array_partition variable=v2004 complete dim=2

  int8_t v2007 = v2004[v2005][v2006];	// L2165
  int8_t w80;	// L2166
  w80 = v2007;	// L2167
  l_step_m80: for (int m80 = 0; m80 < 16; m80++) {	// L2168
  #pragma HLS pipeline II=1
    int8_t v2010 = v2000.read(); // v2000[m80];	// L2169
    int8_t a80;	// L2170
    a80 = v2010;	// L2171
    int32_t v2012 = v2001.read(); // v2001[m80];	// L2172
    int32_t p80;	// L2173
    p80 = v2012;	// L2174
    int32_t v2014 = p80;	// L2175
    int8_t v2015 = a80;	// L2176
    int8_t v2016 = w80;	// L2177
    int16_t v2017 = v2015;	// L2178
    int16_t v2018 = v2016;	// L2179
    int16_t v2019 = v2017 * v2018;	// L2180
    ap_int<33> v2020 = v2014;	// L2181
    ap_int<33> v2021 = v2019;	// L2182
    ap_int<33> v2022 = v2020 + v2021;	// L2183
    int32_t v2023 = v2022;	// L2184
    v2003.write(v2023); // v2003[m80] = v2023;	// L2185
    int8_t v2024 = a80;	// L2186
    v2002.write(v2024); // v2002[m80] = v2024;	// L2187
  }
}

void PE_kernel_ws_gemm_1_5(
  hls::stream< int8_t > &v2025 /* v2025[16] */,
  hls::stream< int32_t > &v2026 /* v2026[16] */,
  hls::stream< int8_t > &v2027 /* v2027[16] */,
  hls::stream< int32_t > &v2028 /* v2028[16] */,
  int8_t v2029[16][16],
  int v2030,
  int v2031
) {	// L2191
  #pragma HLS stream variable=v2025 depth=17
  #pragma HLS stream variable=v2026 depth=17
  #pragma HLS stream variable=v2027 depth=17
  #pragma HLS stream variable=v2028 depth=17
  #pragma HLS array_partition variable=v2029 complete dim=1
  #pragma HLS array_partition variable=v2029 complete dim=2

  int8_t v2032 = v2029[v2030][v2031];	// L2192
  int8_t w81;	// L2193
  w81 = v2032;	// L2194
  l_step_m81: for (int m81 = 0; m81 < 16; m81++) {	// L2195
  #pragma HLS pipeline II=1
    int8_t v2035 = v2025.read(); // v2025[m81];	// L2196
    int8_t a81;	// L2197
    a81 = v2035;	// L2198
    int32_t v2037 = v2026.read(); // v2026[m81];	// L2199
    int32_t p81;	// L2200
    p81 = v2037;	// L2201
    int32_t v2039 = p81;	// L2202
    int8_t v2040 = a81;	// L2203
    int8_t v2041 = w81;	// L2204
    int16_t v2042 = v2040;	// L2205
    int16_t v2043 = v2041;	// L2206
    int16_t v2044 = v2042 * v2043;	// L2207
    ap_int<33> v2045 = v2039;	// L2208
    ap_int<33> v2046 = v2044;	// L2209
    ap_int<33> v2047 = v2045 + v2046;	// L2210
    int32_t v2048 = v2047;	// L2211
    v2028.write(v2048); // v2028[m81] = v2048;	// L2212
    int8_t v2049 = a81;	// L2213
    v2027.write(v2049); // v2027[m81] = v2049;	// L2214
  }
}

void PE_kernel_ws_gemm_2_5(
  hls::stream< int8_t > &v2050 /* v2050[16] */,
  hls::stream< int32_t > &v2051 /* v2051[16] */,
  hls::stream< int8_t > &v2052 /* v2052[16] */,
  hls::stream< int32_t > &v2053 /* v2053[16] */,
  int8_t v2054[16][16],
  int v2055,
  int v2056
) {	// L2218
  #pragma HLS stream variable=v2050 depth=17
  #pragma HLS stream variable=v2051 depth=17
  #pragma HLS stream variable=v2052 depth=17
  #pragma HLS stream variable=v2053 depth=17
  #pragma HLS array_partition variable=v2054 complete dim=1
  #pragma HLS array_partition variable=v2054 complete dim=2

  int8_t v2057 = v2054[v2055][v2056];	// L2219
  int8_t w82;	// L2220
  w82 = v2057;	// L2221
  l_step_m82: for (int m82 = 0; m82 < 16; m82++) {	// L2222
  #pragma HLS pipeline II=1
    int8_t v2060 = v2050.read(); // v2050[m82];	// L2223
    int8_t a82;	// L2224
    a82 = v2060;	// L2225
    int32_t v2062 = v2051.read(); // v2051[m82];	// L2226
    int32_t p82;	// L2227
    p82 = v2062;	// L2228
    int32_t v2064 = p82;	// L2229
    int8_t v2065 = a82;	// L2230
    int8_t v2066 = w82;	// L2231
    int16_t v2067 = v2065;	// L2232
    int16_t v2068 = v2066;	// L2233
    int16_t v2069 = v2067 * v2068;	// L2234
    ap_int<33> v2070 = v2064;	// L2235
    ap_int<33> v2071 = v2069;	// L2236
    ap_int<33> v2072 = v2070 + v2071;	// L2237
    int32_t v2073 = v2072;	// L2238
    v2053.write(v2073); // v2053[m82] = v2073;	// L2239
    int8_t v2074 = a82;	// L2240
    v2052.write(v2074); // v2052[m82] = v2074;	// L2241
  }
}

void PE_kernel_ws_gemm_3_5(
  hls::stream< int8_t > &v2075 /* v2075[16] */,
  hls::stream< int32_t > &v2076 /* v2076[16] */,
  hls::stream< int8_t > &v2077 /* v2077[16] */,
  hls::stream< int32_t > &v2078 /* v2078[16] */,
  int8_t v2079[16][16],
  int v2080,
  int v2081
) {	// L2245
  #pragma HLS stream variable=v2075 depth=17
  #pragma HLS stream variable=v2076 depth=17
  #pragma HLS stream variable=v2077 depth=17
  #pragma HLS stream variable=v2078 depth=17
  #pragma HLS array_partition variable=v2079 complete dim=1
  #pragma HLS array_partition variable=v2079 complete dim=2

  int8_t v2082 = v2079[v2080][v2081];	// L2246
  int8_t w83;	// L2247
  w83 = v2082;	// L2248
  l_step_m83: for (int m83 = 0; m83 < 16; m83++) {	// L2249
  #pragma HLS pipeline II=1
    int8_t v2085 = v2075.read(); // v2075[m83];	// L2250
    int8_t a83;	// L2251
    a83 = v2085;	// L2252
    int32_t v2087 = v2076.read(); // v2076[m83];	// L2253
    int32_t p83;	// L2254
    p83 = v2087;	// L2255
    int32_t v2089 = p83;	// L2256
    int8_t v2090 = a83;	// L2257
    int8_t v2091 = w83;	// L2258
    int16_t v2092 = v2090;	// L2259
    int16_t v2093 = v2091;	// L2260
    int16_t v2094 = v2092 * v2093;	// L2261
    ap_int<33> v2095 = v2089;	// L2262
    ap_int<33> v2096 = v2094;	// L2263
    ap_int<33> v2097 = v2095 + v2096;	// L2264
    int32_t v2098 = v2097;	// L2265
    v2078.write(v2098); // v2078[m83] = v2098;	// L2266
    int8_t v2099 = a83;	// L2267
    v2077.write(v2099); // v2077[m83] = v2099;	// L2268
  }
}

void PE_kernel_ws_gemm_4_5(
  hls::stream< int8_t > &v2100 /* v2100[16] */,
  hls::stream< int32_t > &v2101 /* v2101[16] */,
  hls::stream< int8_t > &v2102 /* v2102[16] */,
  hls::stream< int32_t > &v2103 /* v2103[16] */,
  int8_t v2104[16][16],
  int v2105,
  int v2106
) {	// L2272
  #pragma HLS stream variable=v2100 depth=17
  #pragma HLS stream variable=v2101 depth=17
  #pragma HLS stream variable=v2102 depth=17
  #pragma HLS stream variable=v2103 depth=17
  #pragma HLS array_partition variable=v2104 complete dim=1
  #pragma HLS array_partition variable=v2104 complete dim=2

  int8_t v2107 = v2104[v2105][v2106];	// L2273
  int8_t w84;	// L2274
  w84 = v2107;	// L2275
  l_step_m84: for (int m84 = 0; m84 < 16; m84++) {	// L2276
  #pragma HLS pipeline II=1
    int8_t v2110 = v2100.read(); // v2100[m84];	// L2277
    int8_t a84;	// L2278
    a84 = v2110;	// L2279
    int32_t v2112 = v2101.read(); // v2101[m84];	// L2280
    int32_t p84;	// L2281
    p84 = v2112;	// L2282
    int32_t v2114 = p84;	// L2283
    int8_t v2115 = a84;	// L2284
    int8_t v2116 = w84;	// L2285
    int16_t v2117 = v2115;	// L2286
    int16_t v2118 = v2116;	// L2287
    int16_t v2119 = v2117 * v2118;	// L2288
    ap_int<33> v2120 = v2114;	// L2289
    ap_int<33> v2121 = v2119;	// L2290
    ap_int<33> v2122 = v2120 + v2121;	// L2291
    int32_t v2123 = v2122;	// L2292
    v2103.write(v2123); // v2103[m84] = v2123;	// L2293
    int8_t v2124 = a84;	// L2294
    v2102.write(v2124); // v2102[m84] = v2124;	// L2295
  }
}

void PE_kernel_ws_gemm_5_5(
  hls::stream< int8_t > &v2125 /* v2125[16] */,
  hls::stream< int32_t > &v2126 /* v2126[16] */,
  hls::stream< int8_t > &v2127 /* v2127[16] */,
  hls::stream< int32_t > &v2128 /* v2128[16] */,
  int8_t v2129[16][16],
  int v2130,
  int v2131
) {	// L2299
  #pragma HLS stream variable=v2125 depth=17
  #pragma HLS stream variable=v2126 depth=17
  #pragma HLS stream variable=v2127 depth=17
  #pragma HLS stream variable=v2128 depth=17
  #pragma HLS array_partition variable=v2129 complete dim=1
  #pragma HLS array_partition variable=v2129 complete dim=2

  int8_t v2132 = v2129[v2130][v2131];	// L2300
  int8_t w85;	// L2301
  w85 = v2132;	// L2302
  l_step_m85: for (int m85 = 0; m85 < 16; m85++) {	// L2303
  #pragma HLS pipeline II=1
    int8_t v2135 = v2125.read(); // v2125[m85];	// L2304
    int8_t a85;	// L2305
    a85 = v2135;	// L2306
    int32_t v2137 = v2126.read(); // v2126[m85];	// L2307
    int32_t p85;	// L2308
    p85 = v2137;	// L2309
    int32_t v2139 = p85;	// L2310
    int8_t v2140 = a85;	// L2311
    int8_t v2141 = w85;	// L2312
    int16_t v2142 = v2140;	// L2313
    int16_t v2143 = v2141;	// L2314
    int16_t v2144 = v2142 * v2143;	// L2315
    ap_int<33> v2145 = v2139;	// L2316
    ap_int<33> v2146 = v2144;	// L2317
    ap_int<33> v2147 = v2145 + v2146;	// L2318
    int32_t v2148 = v2147;	// L2319
    v2128.write(v2148); // v2128[m85] = v2148;	// L2320
    int8_t v2149 = a85;	// L2321
    v2127.write(v2149); // v2127[m85] = v2149;	// L2322
  }
}

void PE_kernel_ws_gemm_6_5(
  hls::stream< int8_t > &v2150 /* v2150[16] */,
  hls::stream< int32_t > &v2151 /* v2151[16] */,
  hls::stream< int8_t > &v2152 /* v2152[16] */,
  hls::stream< int32_t > &v2153 /* v2153[16] */,
  int8_t v2154[16][16],
  int v2155,
  int v2156
) {	// L2326
  #pragma HLS stream variable=v2150 depth=17
  #pragma HLS stream variable=v2151 depth=17
  #pragma HLS stream variable=v2152 depth=17
  #pragma HLS stream variable=v2153 depth=17
  #pragma HLS array_partition variable=v2154 complete dim=1
  #pragma HLS array_partition variable=v2154 complete dim=2

  int8_t v2157 = v2154[v2155][v2156];	// L2327
  int8_t w86;	// L2328
  w86 = v2157;	// L2329
  l_step_m86: for (int m86 = 0; m86 < 16; m86++) {	// L2330
  #pragma HLS pipeline II=1
    int8_t v2160 = v2150.read(); // v2150[m86];	// L2331
    int8_t a86;	// L2332
    a86 = v2160;	// L2333
    int32_t v2162 = v2151.read(); // v2151[m86];	// L2334
    int32_t p86;	// L2335
    p86 = v2162;	// L2336
    int32_t v2164 = p86;	// L2337
    int8_t v2165 = a86;	// L2338
    int8_t v2166 = w86;	// L2339
    int16_t v2167 = v2165;	// L2340
    int16_t v2168 = v2166;	// L2341
    int16_t v2169 = v2167 * v2168;	// L2342
    ap_int<33> v2170 = v2164;	// L2343
    ap_int<33> v2171 = v2169;	// L2344
    ap_int<33> v2172 = v2170 + v2171;	// L2345
    int32_t v2173 = v2172;	// L2346
    v2153.write(v2173); // v2153[m86] = v2173;	// L2347
    int8_t v2174 = a86;	// L2348
    v2152.write(v2174); // v2152[m86] = v2174;	// L2349
  }
}

void PE_kernel_ws_gemm_7_5(
  hls::stream< int8_t > &v2175 /* v2175[16] */,
  hls::stream< int32_t > &v2176 /* v2176[16] */,
  hls::stream< int8_t > &v2177 /* v2177[16] */,
  hls::stream< int32_t > &v2178 /* v2178[16] */,
  int8_t v2179[16][16],
  int v2180,
  int v2181
) {	// L2353
  #pragma HLS stream variable=v2175 depth=17
  #pragma HLS stream variable=v2176 depth=17
  #pragma HLS stream variable=v2177 depth=17
  #pragma HLS stream variable=v2178 depth=17
  #pragma HLS array_partition variable=v2179 complete dim=1
  #pragma HLS array_partition variable=v2179 complete dim=2

  int8_t v2182 = v2179[v2180][v2181];	// L2354
  int8_t w87;	// L2355
  w87 = v2182;	// L2356
  l_step_m87: for (int m87 = 0; m87 < 16; m87++) {	// L2357
  #pragma HLS pipeline II=1
    int8_t v2185 = v2175.read(); // v2175[m87];	// L2358
    int8_t a87;	// L2359
    a87 = v2185;	// L2360
    int32_t v2187 = v2176.read(); // v2176[m87];	// L2361
    int32_t p87;	// L2362
    p87 = v2187;	// L2363
    int32_t v2189 = p87;	// L2364
    int8_t v2190 = a87;	// L2365
    int8_t v2191 = w87;	// L2366
    int16_t v2192 = v2190;	// L2367
    int16_t v2193 = v2191;	// L2368
    int16_t v2194 = v2192 * v2193;	// L2369
    ap_int<33> v2195 = v2189;	// L2370
    ap_int<33> v2196 = v2194;	// L2371
    ap_int<33> v2197 = v2195 + v2196;	// L2372
    int32_t v2198 = v2197;	// L2373
    v2178.write(v2198); // v2178[m87] = v2198;	// L2374
    int8_t v2199 = a87;	// L2375
    v2177.write(v2199); // v2177[m87] = v2199;	// L2376
  }
}

void PE_kernel_ws_gemm_8_5(
  hls::stream< int8_t > &v2200 /* v2200[16] */,
  hls::stream< int32_t > &v2201 /* v2201[16] */,
  hls::stream< int8_t > &v2202 /* v2202[16] */,
  hls::stream< int32_t > &v2203 /* v2203[16] */,
  int8_t v2204[16][16],
  int v2205,
  int v2206
) {	// L2380
  #pragma HLS stream variable=v2200 depth=17
  #pragma HLS stream variable=v2201 depth=17
  #pragma HLS stream variable=v2202 depth=17
  #pragma HLS stream variable=v2203 depth=17
  #pragma HLS array_partition variable=v2204 complete dim=1
  #pragma HLS array_partition variable=v2204 complete dim=2

  int8_t v2207 = v2204[v2205][v2206];	// L2381
  int8_t w88;	// L2382
  w88 = v2207;	// L2383
  l_step_m88: for (int m88 = 0; m88 < 16; m88++) {	// L2384
  #pragma HLS pipeline II=1
    int8_t v2210 = v2200.read(); // v2200[m88];	// L2385
    int8_t a88;	// L2386
    a88 = v2210;	// L2387
    int32_t v2212 = v2201.read(); // v2201[m88];	// L2388
    int32_t p88;	// L2389
    p88 = v2212;	// L2390
    int32_t v2214 = p88;	// L2391
    int8_t v2215 = a88;	// L2392
    int8_t v2216 = w88;	// L2393
    int16_t v2217 = v2215;	// L2394
    int16_t v2218 = v2216;	// L2395
    int16_t v2219 = v2217 * v2218;	// L2396
    ap_int<33> v2220 = v2214;	// L2397
    ap_int<33> v2221 = v2219;	// L2398
    ap_int<33> v2222 = v2220 + v2221;	// L2399
    int32_t v2223 = v2222;	// L2400
    v2203.write(v2223); // v2203[m88] = v2223;	// L2401
    int8_t v2224 = a88;	// L2402
    v2202.write(v2224); // v2202[m88] = v2224;	// L2403
  }
}

void PE_kernel_ws_gemm_9_5(
  hls::stream< int8_t > &v2225 /* v2225[16] */,
  hls::stream< int32_t > &v2226 /* v2226[16] */,
  hls::stream< int8_t > &v2227 /* v2227[16] */,
  hls::stream< int32_t > &v2228 /* v2228[16] */,
  int8_t v2229[16][16],
  int v2230,
  int v2231
) {	// L2407
  #pragma HLS stream variable=v2225 depth=17
  #pragma HLS stream variable=v2226 depth=17
  #pragma HLS stream variable=v2227 depth=17
  #pragma HLS stream variable=v2228 depth=17
  #pragma HLS array_partition variable=v2229 complete dim=1
  #pragma HLS array_partition variable=v2229 complete dim=2

  int8_t v2232 = v2229[v2230][v2231];	// L2408
  int8_t w89;	// L2409
  w89 = v2232;	// L2410
  l_step_m89: for (int m89 = 0; m89 < 16; m89++) {	// L2411
  #pragma HLS pipeline II=1
    int8_t v2235 = v2225.read(); // v2225[m89];	// L2412
    int8_t a89;	// L2413
    a89 = v2235;	// L2414
    int32_t v2237 = v2226.read(); // v2226[m89];	// L2415
    int32_t p89;	// L2416
    p89 = v2237;	// L2417
    int32_t v2239 = p89;	// L2418
    int8_t v2240 = a89;	// L2419
    int8_t v2241 = w89;	// L2420
    int16_t v2242 = v2240;	// L2421
    int16_t v2243 = v2241;	// L2422
    int16_t v2244 = v2242 * v2243;	// L2423
    ap_int<33> v2245 = v2239;	// L2424
    ap_int<33> v2246 = v2244;	// L2425
    ap_int<33> v2247 = v2245 + v2246;	// L2426
    int32_t v2248 = v2247;	// L2427
    v2228.write(v2248); // v2228[m89] = v2248;	// L2428
    int8_t v2249 = a89;	// L2429
    v2227.write(v2249); // v2227[m89] = v2249;	// L2430
  }
}

void PE_kernel_ws_gemm_10_5(
  hls::stream< int8_t > &v2250 /* v2250[16] */,
  hls::stream< int32_t > &v2251 /* v2251[16] */,
  hls::stream< int8_t > &v2252 /* v2252[16] */,
  hls::stream< int32_t > &v2253 /* v2253[16] */,
  int8_t v2254[16][16],
  int v2255,
  int v2256
) {	// L2434
  #pragma HLS stream variable=v2250 depth=17
  #pragma HLS stream variable=v2251 depth=17
  #pragma HLS stream variable=v2252 depth=17
  #pragma HLS stream variable=v2253 depth=17
  #pragma HLS array_partition variable=v2254 complete dim=1
  #pragma HLS array_partition variable=v2254 complete dim=2

  int8_t v2257 = v2254[v2255][v2256];	// L2435
  int8_t w90;	// L2436
  w90 = v2257;	// L2437
  l_step_m90: for (int m90 = 0; m90 < 16; m90++) {	// L2438
  #pragma HLS pipeline II=1
    int8_t v2260 = v2250.read(); // v2250[m90];	// L2439
    int8_t a90;	// L2440
    a90 = v2260;	// L2441
    int32_t v2262 = v2251.read(); // v2251[m90];	// L2442
    int32_t p90;	// L2443
    p90 = v2262;	// L2444
    int32_t v2264 = p90;	// L2445
    int8_t v2265 = a90;	// L2446
    int8_t v2266 = w90;	// L2447
    int16_t v2267 = v2265;	// L2448
    int16_t v2268 = v2266;	// L2449
    int16_t v2269 = v2267 * v2268;	// L2450
    ap_int<33> v2270 = v2264;	// L2451
    ap_int<33> v2271 = v2269;	// L2452
    ap_int<33> v2272 = v2270 + v2271;	// L2453
    int32_t v2273 = v2272;	// L2454
    v2253.write(v2273); // v2253[m90] = v2273;	// L2455
    int8_t v2274 = a90;	// L2456
    v2252.write(v2274); // v2252[m90] = v2274;	// L2457
  }
}

void PE_kernel_ws_gemm_11_5(
  hls::stream< int8_t > &v2275 /* v2275[16] */,
  hls::stream< int32_t > &v2276 /* v2276[16] */,
  hls::stream< int8_t > &v2277 /* v2277[16] */,
  hls::stream< int32_t > &v2278 /* v2278[16] */,
  int8_t v2279[16][16],
  int v2280,
  int v2281
) {	// L2461
  #pragma HLS stream variable=v2275 depth=17
  #pragma HLS stream variable=v2276 depth=17
  #pragma HLS stream variable=v2277 depth=17
  #pragma HLS stream variable=v2278 depth=17
  #pragma HLS array_partition variable=v2279 complete dim=1
  #pragma HLS array_partition variable=v2279 complete dim=2

  int8_t v2282 = v2279[v2280][v2281];	// L2462
  int8_t w91;	// L2463
  w91 = v2282;	// L2464
  l_step_m91: for (int m91 = 0; m91 < 16; m91++) {	// L2465
  #pragma HLS pipeline II=1
    int8_t v2285 = v2275.read(); // v2275[m91];	// L2466
    int8_t a91;	// L2467
    a91 = v2285;	// L2468
    int32_t v2287 = v2276.read(); // v2276[m91];	// L2469
    int32_t p91;	// L2470
    p91 = v2287;	// L2471
    int32_t v2289 = p91;	// L2472
    int8_t v2290 = a91;	// L2473
    int8_t v2291 = w91;	// L2474
    int16_t v2292 = v2290;	// L2475
    int16_t v2293 = v2291;	// L2476
    int16_t v2294 = v2292 * v2293;	// L2477
    ap_int<33> v2295 = v2289;	// L2478
    ap_int<33> v2296 = v2294;	// L2479
    ap_int<33> v2297 = v2295 + v2296;	// L2480
    int32_t v2298 = v2297;	// L2481
    v2278.write(v2298); // v2278[m91] = v2298;	// L2482
    int8_t v2299 = a91;	// L2483
    v2277.write(v2299); // v2277[m91] = v2299;	// L2484
  }
}

void PE_kernel_ws_gemm_12_5(
  hls::stream< int8_t > &v2300 /* v2300[16] */,
  hls::stream< int32_t > &v2301 /* v2301[16] */,
  hls::stream< int8_t > &v2302 /* v2302[16] */,
  hls::stream< int32_t > &v2303 /* v2303[16] */,
  int8_t v2304[16][16],
  int v2305,
  int v2306
) {	// L2488
  #pragma HLS stream variable=v2300 depth=17
  #pragma HLS stream variable=v2301 depth=17
  #pragma HLS stream variable=v2302 depth=17
  #pragma HLS stream variable=v2303 depth=17
  #pragma HLS array_partition variable=v2304 complete dim=1
  #pragma HLS array_partition variable=v2304 complete dim=2

  int8_t v2307 = v2304[v2305][v2306];	// L2489
  int8_t w92;	// L2490
  w92 = v2307;	// L2491
  l_step_m92: for (int m92 = 0; m92 < 16; m92++) {	// L2492
  #pragma HLS pipeline II=1
    int8_t v2310 = v2300.read(); // v2300[m92];	// L2493
    int8_t a92;	// L2494
    a92 = v2310;	// L2495
    int32_t v2312 = v2301.read(); // v2301[m92];	// L2496
    int32_t p92;	// L2497
    p92 = v2312;	// L2498
    int32_t v2314 = p92;	// L2499
    int8_t v2315 = a92;	// L2500
    int8_t v2316 = w92;	// L2501
    int16_t v2317 = v2315;	// L2502
    int16_t v2318 = v2316;	// L2503
    int16_t v2319 = v2317 * v2318;	// L2504
    ap_int<33> v2320 = v2314;	// L2505
    ap_int<33> v2321 = v2319;	// L2506
    ap_int<33> v2322 = v2320 + v2321;	// L2507
    int32_t v2323 = v2322;	// L2508
    v2303.write(v2323); // v2303[m92] = v2323;	// L2509
    int8_t v2324 = a92;	// L2510
    v2302.write(v2324); // v2302[m92] = v2324;	// L2511
  }
}

void PE_kernel_ws_gemm_13_5(
  hls::stream< int8_t > &v2325 /* v2325[16] */,
  hls::stream< int32_t > &v2326 /* v2326[16] */,
  hls::stream< int8_t > &v2327 /* v2327[16] */,
  hls::stream< int32_t > &v2328 /* v2328[16] */,
  int8_t v2329[16][16],
  int v2330,
  int v2331
) {	// L2515
  #pragma HLS stream variable=v2325 depth=17
  #pragma HLS stream variable=v2326 depth=17
  #pragma HLS stream variable=v2327 depth=17
  #pragma HLS stream variable=v2328 depth=17
  #pragma HLS array_partition variable=v2329 complete dim=1
  #pragma HLS array_partition variable=v2329 complete dim=2

  int8_t v2332 = v2329[v2330][v2331];	// L2516
  int8_t w93;	// L2517
  w93 = v2332;	// L2518
  l_step_m93: for (int m93 = 0; m93 < 16; m93++) {	// L2519
  #pragma HLS pipeline II=1
    int8_t v2335 = v2325.read(); // v2325[m93];	// L2520
    int8_t a93;	// L2521
    a93 = v2335;	// L2522
    int32_t v2337 = v2326.read(); // v2326[m93];	// L2523
    int32_t p93;	// L2524
    p93 = v2337;	// L2525
    int32_t v2339 = p93;	// L2526
    int8_t v2340 = a93;	// L2527
    int8_t v2341 = w93;	// L2528
    int16_t v2342 = v2340;	// L2529
    int16_t v2343 = v2341;	// L2530
    int16_t v2344 = v2342 * v2343;	// L2531
    ap_int<33> v2345 = v2339;	// L2532
    ap_int<33> v2346 = v2344;	// L2533
    ap_int<33> v2347 = v2345 + v2346;	// L2534
    int32_t v2348 = v2347;	// L2535
    v2328.write(v2348); // v2328[m93] = v2348;	// L2536
    int8_t v2349 = a93;	// L2537
    v2327.write(v2349); // v2327[m93] = v2349;	// L2538
  }
}

void PE_kernel_ws_gemm_14_5(
  hls::stream< int8_t > &v2350 /* v2350[16] */,
  hls::stream< int32_t > &v2351 /* v2351[16] */,
  hls::stream< int8_t > &v2352 /* v2352[16] */,
  hls::stream< int32_t > &v2353 /* v2353[16] */,
  int8_t v2354[16][16],
  int v2355,
  int v2356
) {	// L2542
  #pragma HLS stream variable=v2350 depth=17
  #pragma HLS stream variable=v2351 depth=17
  #pragma HLS stream variable=v2352 depth=17
  #pragma HLS stream variable=v2353 depth=17
  #pragma HLS array_partition variable=v2354 complete dim=1
  #pragma HLS array_partition variable=v2354 complete dim=2

  int8_t v2357 = v2354[v2355][v2356];	// L2543
  int8_t w94;	// L2544
  w94 = v2357;	// L2545
  l_step_m94: for (int m94 = 0; m94 < 16; m94++) {	// L2546
  #pragma HLS pipeline II=1
    int8_t v2360 = v2350.read(); // v2350[m94];	// L2547
    int8_t a94;	// L2548
    a94 = v2360;	// L2549
    int32_t v2362 = v2351.read(); // v2351[m94];	// L2550
    int32_t p94;	// L2551
    p94 = v2362;	// L2552
    int32_t v2364 = p94;	// L2553
    int8_t v2365 = a94;	// L2554
    int8_t v2366 = w94;	// L2555
    int16_t v2367 = v2365;	// L2556
    int16_t v2368 = v2366;	// L2557
    int16_t v2369 = v2367 * v2368;	// L2558
    ap_int<33> v2370 = v2364;	// L2559
    ap_int<33> v2371 = v2369;	// L2560
    ap_int<33> v2372 = v2370 + v2371;	// L2561
    int32_t v2373 = v2372;	// L2562
    v2353.write(v2373); // v2353[m94] = v2373;	// L2563
    int8_t v2374 = a94;	// L2564
    v2352.write(v2374); // v2352[m94] = v2374;	// L2565
  }
}

void PE_kernel_ws_gemm_15_5(
  hls::stream< int8_t > &v2375 /* v2375[16] */,
  hls::stream< int32_t > &v2376 /* v2376[16] */,
  hls::stream< int8_t > &v2377 /* v2377[16] */,
  hls::stream< int32_t > &v2378 /* v2378[16] */,
  int8_t v2379[16][16],
  int v2380,
  int v2381
) {	// L2569
  #pragma HLS stream variable=v2375 depth=17
  #pragma HLS stream variable=v2376 depth=17
  #pragma HLS stream variable=v2377 depth=17
  #pragma HLS stream variable=v2378 depth=17
  #pragma HLS array_partition variable=v2379 complete dim=1
  #pragma HLS array_partition variable=v2379 complete dim=2

  int8_t v2382 = v2379[v2380][v2381];	// L2570
  int8_t w95;	// L2571
  w95 = v2382;	// L2572
  l_step_m95: for (int m95 = 0; m95 < 16; m95++) {	// L2573
  #pragma HLS pipeline II=1
    int8_t v2385 = v2375.read(); // v2375[m95];	// L2574
    int8_t a95;	// L2575
    a95 = v2385;	// L2576
    int32_t v2387 = v2376.read(); // v2376[m95];	// L2577
    int32_t p95;	// L2578
    p95 = v2387;	// L2579
    int32_t v2389 = p95;	// L2580
    int8_t v2390 = a95;	// L2581
    int8_t v2391 = w95;	// L2582
    int16_t v2392 = v2390;	// L2583
    int16_t v2393 = v2391;	// L2584
    int16_t v2394 = v2392 * v2393;	// L2585
    ap_int<33> v2395 = v2389;	// L2586
    ap_int<33> v2396 = v2394;	// L2587
    ap_int<33> v2397 = v2395 + v2396;	// L2588
    int32_t v2398 = v2397;	// L2589
    v2378.write(v2398); // v2378[m95] = v2398;	// L2590
    int8_t v2399 = a95;	// L2591
    v2377.write(v2399); // v2377[m95] = v2399;	// L2592
  }
}

void PE_kernel_ws_gemm_0_6(
  hls::stream< int8_t > &v2400 /* v2400[16] */,
  hls::stream< int32_t > &v2401 /* v2401[16] */,
  hls::stream< int8_t > &v2402 /* v2402[16] */,
  hls::stream< int32_t > &v2403 /* v2403[16] */,
  int8_t v2404[16][16],
  int v2405,
  int v2406
) {	// L2596
  #pragma HLS stream variable=v2400 depth=17
  #pragma HLS stream variable=v2401 depth=17
  #pragma HLS stream variable=v2402 depth=17
  #pragma HLS stream variable=v2403 depth=17
  #pragma HLS array_partition variable=v2404 complete dim=1
  #pragma HLS array_partition variable=v2404 complete dim=2

  int8_t v2407 = v2404[v2405][v2406];	// L2597
  int8_t w96;	// L2598
  w96 = v2407;	// L2599
  l_step_m96: for (int m96 = 0; m96 < 16; m96++) {	// L2600
  #pragma HLS pipeline II=1
    int8_t v2410 = v2400.read(); // v2400[m96];	// L2601
    int8_t a96;	// L2602
    a96 = v2410;	// L2603
    int32_t v2412 = v2401.read(); // v2401[m96];	// L2604
    int32_t p96;	// L2605
    p96 = v2412;	// L2606
    int32_t v2414 = p96;	// L2607
    int8_t v2415 = a96;	// L2608
    int8_t v2416 = w96;	// L2609
    int16_t v2417 = v2415;	// L2610
    int16_t v2418 = v2416;	// L2611
    int16_t v2419 = v2417 * v2418;	// L2612
    ap_int<33> v2420 = v2414;	// L2613
    ap_int<33> v2421 = v2419;	// L2614
    ap_int<33> v2422 = v2420 + v2421;	// L2615
    int32_t v2423 = v2422;	// L2616
    v2403.write(v2423); // v2403[m96] = v2423;	// L2617
    int8_t v2424 = a96;	// L2618
    v2402.write(v2424); // v2402[m96] = v2424;	// L2619
  }
}

void PE_kernel_ws_gemm_1_6(
  hls::stream< int8_t > &v2425 /* v2425[16] */,
  hls::stream< int32_t > &v2426 /* v2426[16] */,
  hls::stream< int8_t > &v2427 /* v2427[16] */,
  hls::stream< int32_t > &v2428 /* v2428[16] */,
  int8_t v2429[16][16],
  int v2430,
  int v2431
) {	// L2623
  #pragma HLS stream variable=v2425 depth=17
  #pragma HLS stream variable=v2426 depth=17
  #pragma HLS stream variable=v2427 depth=17
  #pragma HLS stream variable=v2428 depth=17
  #pragma HLS array_partition variable=v2429 complete dim=1
  #pragma HLS array_partition variable=v2429 complete dim=2

  int8_t v2432 = v2429[v2430][v2431];	// L2624
  int8_t w97;	// L2625
  w97 = v2432;	// L2626
  l_step_m97: for (int m97 = 0; m97 < 16; m97++) {	// L2627
  #pragma HLS pipeline II=1
    int8_t v2435 = v2425.read(); // v2425[m97];	// L2628
    int8_t a97;	// L2629
    a97 = v2435;	// L2630
    int32_t v2437 = v2426.read(); // v2426[m97];	// L2631
    int32_t p97;	// L2632
    p97 = v2437;	// L2633
    int32_t v2439 = p97;	// L2634
    int8_t v2440 = a97;	// L2635
    int8_t v2441 = w97;	// L2636
    int16_t v2442 = v2440;	// L2637
    int16_t v2443 = v2441;	// L2638
    int16_t v2444 = v2442 * v2443;	// L2639
    ap_int<33> v2445 = v2439;	// L2640
    ap_int<33> v2446 = v2444;	// L2641
    ap_int<33> v2447 = v2445 + v2446;	// L2642
    int32_t v2448 = v2447;	// L2643
    v2428.write(v2448); // v2428[m97] = v2448;	// L2644
    int8_t v2449 = a97;	// L2645
    v2427.write(v2449); // v2427[m97] = v2449;	// L2646
  }
}

void PE_kernel_ws_gemm_2_6(
  hls::stream< int8_t > &v2450 /* v2450[16] */,
  hls::stream< int32_t > &v2451 /* v2451[16] */,
  hls::stream< int8_t > &v2452 /* v2452[16] */,
  hls::stream< int32_t > &v2453 /* v2453[16] */,
  int8_t v2454[16][16],
  int v2455,
  int v2456
) {	// L2650
  #pragma HLS stream variable=v2450 depth=17
  #pragma HLS stream variable=v2451 depth=17
  #pragma HLS stream variable=v2452 depth=17
  #pragma HLS stream variable=v2453 depth=17
  #pragma HLS array_partition variable=v2454 complete dim=1
  #pragma HLS array_partition variable=v2454 complete dim=2

  int8_t v2457 = v2454[v2455][v2456];	// L2651
  int8_t w98;	// L2652
  w98 = v2457;	// L2653
  l_step_m98: for (int m98 = 0; m98 < 16; m98++) {	// L2654
  #pragma HLS pipeline II=1
    int8_t v2460 = v2450.read(); // v2450[m98];	// L2655
    int8_t a98;	// L2656
    a98 = v2460;	// L2657
    int32_t v2462 = v2451.read(); // v2451[m98];	// L2658
    int32_t p98;	// L2659
    p98 = v2462;	// L2660
    int32_t v2464 = p98;	// L2661
    int8_t v2465 = a98;	// L2662
    int8_t v2466 = w98;	// L2663
    int16_t v2467 = v2465;	// L2664
    int16_t v2468 = v2466;	// L2665
    int16_t v2469 = v2467 * v2468;	// L2666
    ap_int<33> v2470 = v2464;	// L2667
    ap_int<33> v2471 = v2469;	// L2668
    ap_int<33> v2472 = v2470 + v2471;	// L2669
    int32_t v2473 = v2472;	// L2670
    v2453.write(v2473); // v2453[m98] = v2473;	// L2671
    int8_t v2474 = a98;	// L2672
    v2452.write(v2474); // v2452[m98] = v2474;	// L2673
  }
}

void PE_kernel_ws_gemm_3_6(
  hls::stream< int8_t > &v2475 /* v2475[16] */,
  hls::stream< int32_t > &v2476 /* v2476[16] */,
  hls::stream< int8_t > &v2477 /* v2477[16] */,
  hls::stream< int32_t > &v2478 /* v2478[16] */,
  int8_t v2479[16][16],
  int v2480,
  int v2481
) {	// L2677
  #pragma HLS stream variable=v2475 depth=17
  #pragma HLS stream variable=v2476 depth=17
  #pragma HLS stream variable=v2477 depth=17
  #pragma HLS stream variable=v2478 depth=17
  #pragma HLS array_partition variable=v2479 complete dim=1
  #pragma HLS array_partition variable=v2479 complete dim=2

  int8_t v2482 = v2479[v2480][v2481];	// L2678
  int8_t w99;	// L2679
  w99 = v2482;	// L2680
  l_step_m99: for (int m99 = 0; m99 < 16; m99++) {	// L2681
  #pragma HLS pipeline II=1
    int8_t v2485 = v2475.read(); // v2475[m99];	// L2682
    int8_t a99;	// L2683
    a99 = v2485;	// L2684
    int32_t v2487 = v2476.read(); // v2476[m99];	// L2685
    int32_t p99;	// L2686
    p99 = v2487;	// L2687
    int32_t v2489 = p99;	// L2688
    int8_t v2490 = a99;	// L2689
    int8_t v2491 = w99;	// L2690
    int16_t v2492 = v2490;	// L2691
    int16_t v2493 = v2491;	// L2692
    int16_t v2494 = v2492 * v2493;	// L2693
    ap_int<33> v2495 = v2489;	// L2694
    ap_int<33> v2496 = v2494;	// L2695
    ap_int<33> v2497 = v2495 + v2496;	// L2696
    int32_t v2498 = v2497;	// L2697
    v2478.write(v2498); // v2478[m99] = v2498;	// L2698
    int8_t v2499 = a99;	// L2699
    v2477.write(v2499); // v2477[m99] = v2499;	// L2700
  }
}

void PE_kernel_ws_gemm_4_6(
  hls::stream< int8_t > &v2500 /* v2500[16] */,
  hls::stream< int32_t > &v2501 /* v2501[16] */,
  hls::stream< int8_t > &v2502 /* v2502[16] */,
  hls::stream< int32_t > &v2503 /* v2503[16] */,
  int8_t v2504[16][16],
  int v2505,
  int v2506
) {	// L2704
  #pragma HLS stream variable=v2500 depth=17
  #pragma HLS stream variable=v2501 depth=17
  #pragma HLS stream variable=v2502 depth=17
  #pragma HLS stream variable=v2503 depth=17
  #pragma HLS array_partition variable=v2504 complete dim=1
  #pragma HLS array_partition variable=v2504 complete dim=2

  int8_t v2507 = v2504[v2505][v2506];	// L2705
  int8_t w100;	// L2706
  w100 = v2507;	// L2707
  l_step_m100: for (int m100 = 0; m100 < 16; m100++) {	// L2708
  #pragma HLS pipeline II=1
    int8_t v2510 = v2500.read(); // v2500[m100];	// L2709
    int8_t a100;	// L2710
    a100 = v2510;	// L2711
    int32_t v2512 = v2501.read(); // v2501[m100];	// L2712
    int32_t p100;	// L2713
    p100 = v2512;	// L2714
    int32_t v2514 = p100;	// L2715
    int8_t v2515 = a100;	// L2716
    int8_t v2516 = w100;	// L2717
    int16_t v2517 = v2515;	// L2718
    int16_t v2518 = v2516;	// L2719
    int16_t v2519 = v2517 * v2518;	// L2720
    ap_int<33> v2520 = v2514;	// L2721
    ap_int<33> v2521 = v2519;	// L2722
    ap_int<33> v2522 = v2520 + v2521;	// L2723
    int32_t v2523 = v2522;	// L2724
    v2503.write(v2523); // v2503[m100] = v2523;	// L2725
    int8_t v2524 = a100;	// L2726
    v2502.write(v2524); // v2502[m100] = v2524;	// L2727
  }
}

void PE_kernel_ws_gemm_5_6(
  hls::stream< int8_t > &v2525 /* v2525[16] */,
  hls::stream< int32_t > &v2526 /* v2526[16] */,
  hls::stream< int8_t > &v2527 /* v2527[16] */,
  hls::stream< int32_t > &v2528 /* v2528[16] */,
  int8_t v2529[16][16],
  int v2530,
  int v2531
) {	// L2731
  #pragma HLS stream variable=v2525 depth=17
  #pragma HLS stream variable=v2526 depth=17
  #pragma HLS stream variable=v2527 depth=17
  #pragma HLS stream variable=v2528 depth=17
  #pragma HLS array_partition variable=v2529 complete dim=1
  #pragma HLS array_partition variable=v2529 complete dim=2

  int8_t v2532 = v2529[v2530][v2531];	// L2732
  int8_t w101;	// L2733
  w101 = v2532;	// L2734
  l_step_m101: for (int m101 = 0; m101 < 16; m101++) {	// L2735
  #pragma HLS pipeline II=1
    int8_t v2535 = v2525.read(); // v2525[m101];	// L2736
    int8_t a101;	// L2737
    a101 = v2535;	// L2738
    int32_t v2537 = v2526.read(); // v2526[m101];	// L2739
    int32_t p101;	// L2740
    p101 = v2537;	// L2741
    int32_t v2539 = p101;	// L2742
    int8_t v2540 = a101;	// L2743
    int8_t v2541 = w101;	// L2744
    int16_t v2542 = v2540;	// L2745
    int16_t v2543 = v2541;	// L2746
    int16_t v2544 = v2542 * v2543;	// L2747
    ap_int<33> v2545 = v2539;	// L2748
    ap_int<33> v2546 = v2544;	// L2749
    ap_int<33> v2547 = v2545 + v2546;	// L2750
    int32_t v2548 = v2547;	// L2751
    v2528.write(v2548); // v2528[m101] = v2548;	// L2752
    int8_t v2549 = a101;	// L2753
    v2527.write(v2549); // v2527[m101] = v2549;	// L2754
  }
}

void PE_kernel_ws_gemm_6_6(
  hls::stream< int8_t > &v2550 /* v2550[16] */,
  hls::stream< int32_t > &v2551 /* v2551[16] */,
  hls::stream< int8_t > &v2552 /* v2552[16] */,
  hls::stream< int32_t > &v2553 /* v2553[16] */,
  int8_t v2554[16][16],
  int v2555,
  int v2556
) {	// L2758
  #pragma HLS stream variable=v2550 depth=17
  #pragma HLS stream variable=v2551 depth=17
  #pragma HLS stream variable=v2552 depth=17
  #pragma HLS stream variable=v2553 depth=17
  #pragma HLS array_partition variable=v2554 complete dim=1
  #pragma HLS array_partition variable=v2554 complete dim=2

  int8_t v2557 = v2554[v2555][v2556];	// L2759
  int8_t w102;	// L2760
  w102 = v2557;	// L2761
  l_step_m102: for (int m102 = 0; m102 < 16; m102++) {	// L2762
  #pragma HLS pipeline II=1
    int8_t v2560 = v2550.read(); // v2550[m102];	// L2763
    int8_t a102;	// L2764
    a102 = v2560;	// L2765
    int32_t v2562 = v2551.read(); // v2551[m102];	// L2766
    int32_t p102;	// L2767
    p102 = v2562;	// L2768
    int32_t v2564 = p102;	// L2769
    int8_t v2565 = a102;	// L2770
    int8_t v2566 = w102;	// L2771
    int16_t v2567 = v2565;	// L2772
    int16_t v2568 = v2566;	// L2773
    int16_t v2569 = v2567 * v2568;	// L2774
    ap_int<33> v2570 = v2564;	// L2775
    ap_int<33> v2571 = v2569;	// L2776
    ap_int<33> v2572 = v2570 + v2571;	// L2777
    int32_t v2573 = v2572;	// L2778
    v2553.write(v2573); // v2553[m102] = v2573;	// L2779
    int8_t v2574 = a102;	// L2780
    v2552.write(v2574); // v2552[m102] = v2574;	// L2781
  }
}

void PE_kernel_ws_gemm_7_6(
  hls::stream< int8_t > &v2575 /* v2575[16] */,
  hls::stream< int32_t > &v2576 /* v2576[16] */,
  hls::stream< int8_t > &v2577 /* v2577[16] */,
  hls::stream< int32_t > &v2578 /* v2578[16] */,
  int8_t v2579[16][16],
  int v2580,
  int v2581
) {	// L2785
  #pragma HLS stream variable=v2575 depth=17
  #pragma HLS stream variable=v2576 depth=17
  #pragma HLS stream variable=v2577 depth=17
  #pragma HLS stream variable=v2578 depth=17
  #pragma HLS array_partition variable=v2579 complete dim=1
  #pragma HLS array_partition variable=v2579 complete dim=2

  int8_t v2582 = v2579[v2580][v2581];	// L2786
  int8_t w103;	// L2787
  w103 = v2582;	// L2788
  l_step_m103: for (int m103 = 0; m103 < 16; m103++) {	// L2789
  #pragma HLS pipeline II=1
    int8_t v2585 = v2575.read(); // v2575[m103];	// L2790
    int8_t a103;	// L2791
    a103 = v2585;	// L2792
    int32_t v2587 = v2576.read(); // v2576[m103];	// L2793
    int32_t p103;	// L2794
    p103 = v2587;	// L2795
    int32_t v2589 = p103;	// L2796
    int8_t v2590 = a103;	// L2797
    int8_t v2591 = w103;	// L2798
    int16_t v2592 = v2590;	// L2799
    int16_t v2593 = v2591;	// L2800
    int16_t v2594 = v2592 * v2593;	// L2801
    ap_int<33> v2595 = v2589;	// L2802
    ap_int<33> v2596 = v2594;	// L2803
    ap_int<33> v2597 = v2595 + v2596;	// L2804
    int32_t v2598 = v2597;	// L2805
    v2578.write(v2598); // v2578[m103] = v2598;	// L2806
    int8_t v2599 = a103;	// L2807
    v2577.write(v2599); // v2577[m103] = v2599;	// L2808
  }
}

void PE_kernel_ws_gemm_8_6(
  hls::stream< int8_t > &v2600 /* v2600[16] */,
  hls::stream< int32_t > &v2601 /* v2601[16] */,
  hls::stream< int8_t > &v2602 /* v2602[16] */,
  hls::stream< int32_t > &v2603 /* v2603[16] */,
  int8_t v2604[16][16],
  int v2605,
  int v2606
) {	// L2812
  #pragma HLS stream variable=v2600 depth=17
  #pragma HLS stream variable=v2601 depth=17
  #pragma HLS stream variable=v2602 depth=17
  #pragma HLS stream variable=v2603 depth=17
  #pragma HLS array_partition variable=v2604 complete dim=1
  #pragma HLS array_partition variable=v2604 complete dim=2

  int8_t v2607 = v2604[v2605][v2606];	// L2813
  int8_t w104;	// L2814
  w104 = v2607;	// L2815
  l_step_m104: for (int m104 = 0; m104 < 16; m104++) {	// L2816
  #pragma HLS pipeline II=1
    int8_t v2610 = v2600.read(); // v2600[m104];	// L2817
    int8_t a104;	// L2818
    a104 = v2610;	// L2819
    int32_t v2612 = v2601.read(); // v2601[m104];	// L2820
    int32_t p104;	// L2821
    p104 = v2612;	// L2822
    int32_t v2614 = p104;	// L2823
    int8_t v2615 = a104;	// L2824
    int8_t v2616 = w104;	// L2825
    int16_t v2617 = v2615;	// L2826
    int16_t v2618 = v2616;	// L2827
    int16_t v2619 = v2617 * v2618;	// L2828
    ap_int<33> v2620 = v2614;	// L2829
    ap_int<33> v2621 = v2619;	// L2830
    ap_int<33> v2622 = v2620 + v2621;	// L2831
    int32_t v2623 = v2622;	// L2832
    v2603.write(v2623); // v2603[m104] = v2623;	// L2833
    int8_t v2624 = a104;	// L2834
    v2602.write(v2624); // v2602[m104] = v2624;	// L2835
  }
}

void PE_kernel_ws_gemm_9_6(
  hls::stream< int8_t > &v2625 /* v2625[16] */,
  hls::stream< int32_t > &v2626 /* v2626[16] */,
  hls::stream< int8_t > &v2627 /* v2627[16] */,
  hls::stream< int32_t > &v2628 /* v2628[16] */,
  int8_t v2629[16][16],
  int v2630,
  int v2631
) {	// L2839
  #pragma HLS stream variable=v2625 depth=17
  #pragma HLS stream variable=v2626 depth=17
  #pragma HLS stream variable=v2627 depth=17
  #pragma HLS stream variable=v2628 depth=17
  #pragma HLS array_partition variable=v2629 complete dim=1
  #pragma HLS array_partition variable=v2629 complete dim=2

  int8_t v2632 = v2629[v2630][v2631];	// L2840
  int8_t w105;	// L2841
  w105 = v2632;	// L2842
  l_step_m105: for (int m105 = 0; m105 < 16; m105++) {	// L2843
  #pragma HLS pipeline II=1
    int8_t v2635 = v2625.read(); // v2625[m105];	// L2844
    int8_t a105;	// L2845
    a105 = v2635;	// L2846
    int32_t v2637 = v2626.read(); // v2626[m105];	// L2847
    int32_t p105;	// L2848
    p105 = v2637;	// L2849
    int32_t v2639 = p105;	// L2850
    int8_t v2640 = a105;	// L2851
    int8_t v2641 = w105;	// L2852
    int16_t v2642 = v2640;	// L2853
    int16_t v2643 = v2641;	// L2854
    int16_t v2644 = v2642 * v2643;	// L2855
    ap_int<33> v2645 = v2639;	// L2856
    ap_int<33> v2646 = v2644;	// L2857
    ap_int<33> v2647 = v2645 + v2646;	// L2858
    int32_t v2648 = v2647;	// L2859
    v2628.write(v2648); // v2628[m105] = v2648;	// L2860
    int8_t v2649 = a105;	// L2861
    v2627.write(v2649); // v2627[m105] = v2649;	// L2862
  }
}

void PE_kernel_ws_gemm_10_6(
  hls::stream< int8_t > &v2650 /* v2650[16] */,
  hls::stream< int32_t > &v2651 /* v2651[16] */,
  hls::stream< int8_t > &v2652 /* v2652[16] */,
  hls::stream< int32_t > &v2653 /* v2653[16] */,
  int8_t v2654[16][16],
  int v2655,
  int v2656
) {	// L2866
  #pragma HLS stream variable=v2650 depth=17
  #pragma HLS stream variable=v2651 depth=17
  #pragma HLS stream variable=v2652 depth=17
  #pragma HLS stream variable=v2653 depth=17
  #pragma HLS array_partition variable=v2654 complete dim=1
  #pragma HLS array_partition variable=v2654 complete dim=2

  int8_t v2657 = v2654[v2655][v2656];	// L2867
  int8_t w106;	// L2868
  w106 = v2657;	// L2869
  l_step_m106: for (int m106 = 0; m106 < 16; m106++) {	// L2870
  #pragma HLS pipeline II=1
    int8_t v2660 = v2650.read(); // v2650[m106];	// L2871
    int8_t a106;	// L2872
    a106 = v2660;	// L2873
    int32_t v2662 = v2651.read(); // v2651[m106];	// L2874
    int32_t p106;	// L2875
    p106 = v2662;	// L2876
    int32_t v2664 = p106;	// L2877
    int8_t v2665 = a106;	// L2878
    int8_t v2666 = w106;	// L2879
    int16_t v2667 = v2665;	// L2880
    int16_t v2668 = v2666;	// L2881
    int16_t v2669 = v2667 * v2668;	// L2882
    ap_int<33> v2670 = v2664;	// L2883
    ap_int<33> v2671 = v2669;	// L2884
    ap_int<33> v2672 = v2670 + v2671;	// L2885
    int32_t v2673 = v2672;	// L2886
    v2653.write(v2673); // v2653[m106] = v2673;	// L2887
    int8_t v2674 = a106;	// L2888
    v2652.write(v2674); // v2652[m106] = v2674;	// L2889
  }
}

void PE_kernel_ws_gemm_11_6(
  hls::stream< int8_t > &v2675 /* v2675[16] */,
  hls::stream< int32_t > &v2676 /* v2676[16] */,
  hls::stream< int8_t > &v2677 /* v2677[16] */,
  hls::stream< int32_t > &v2678 /* v2678[16] */,
  int8_t v2679[16][16],
  int v2680,
  int v2681
) {	// L2893
  #pragma HLS stream variable=v2675 depth=17
  #pragma HLS stream variable=v2676 depth=17
  #pragma HLS stream variable=v2677 depth=17
  #pragma HLS stream variable=v2678 depth=17
  #pragma HLS array_partition variable=v2679 complete dim=1
  #pragma HLS array_partition variable=v2679 complete dim=2

  int8_t v2682 = v2679[v2680][v2681];	// L2894
  int8_t w107;	// L2895
  w107 = v2682;	// L2896
  l_step_m107: for (int m107 = 0; m107 < 16; m107++) {	// L2897
  #pragma HLS pipeline II=1
    int8_t v2685 = v2675.read(); // v2675[m107];	// L2898
    int8_t a107;	// L2899
    a107 = v2685;	// L2900
    int32_t v2687 = v2676.read(); // v2676[m107];	// L2901
    int32_t p107;	// L2902
    p107 = v2687;	// L2903
    int32_t v2689 = p107;	// L2904
    int8_t v2690 = a107;	// L2905
    int8_t v2691 = w107;	// L2906
    int16_t v2692 = v2690;	// L2907
    int16_t v2693 = v2691;	// L2908
    int16_t v2694 = v2692 * v2693;	// L2909
    ap_int<33> v2695 = v2689;	// L2910
    ap_int<33> v2696 = v2694;	// L2911
    ap_int<33> v2697 = v2695 + v2696;	// L2912
    int32_t v2698 = v2697;	// L2913
    v2678.write(v2698); // v2678[m107] = v2698;	// L2914
    int8_t v2699 = a107;	// L2915
    v2677.write(v2699); // v2677[m107] = v2699;	// L2916
  }
}

void PE_kernel_ws_gemm_12_6(
  hls::stream< int8_t > &v2700 /* v2700[16] */,
  hls::stream< int32_t > &v2701 /* v2701[16] */,
  hls::stream< int8_t > &v2702 /* v2702[16] */,
  hls::stream< int32_t > &v2703 /* v2703[16] */,
  int8_t v2704[16][16],
  int v2705,
  int v2706
) {	// L2920
  #pragma HLS stream variable=v2700 depth=17
  #pragma HLS stream variable=v2701 depth=17
  #pragma HLS stream variable=v2702 depth=17
  #pragma HLS stream variable=v2703 depth=17
  #pragma HLS array_partition variable=v2704 complete dim=1
  #pragma HLS array_partition variable=v2704 complete dim=2

  int8_t v2707 = v2704[v2705][v2706];	// L2921
  int8_t w108;	// L2922
  w108 = v2707;	// L2923
  l_step_m108: for (int m108 = 0; m108 < 16; m108++) {	// L2924
  #pragma HLS pipeline II=1
    int8_t v2710 = v2700.read(); // v2700[m108];	// L2925
    int8_t a108;	// L2926
    a108 = v2710;	// L2927
    int32_t v2712 = v2701.read(); // v2701[m108];	// L2928
    int32_t p108;	// L2929
    p108 = v2712;	// L2930
    int32_t v2714 = p108;	// L2931
    int8_t v2715 = a108;	// L2932
    int8_t v2716 = w108;	// L2933
    int16_t v2717 = v2715;	// L2934
    int16_t v2718 = v2716;	// L2935
    int16_t v2719 = v2717 * v2718;	// L2936
    ap_int<33> v2720 = v2714;	// L2937
    ap_int<33> v2721 = v2719;	// L2938
    ap_int<33> v2722 = v2720 + v2721;	// L2939
    int32_t v2723 = v2722;	// L2940
    v2703.write(v2723); // v2703[m108] = v2723;	// L2941
    int8_t v2724 = a108;	// L2942
    v2702.write(v2724); // v2702[m108] = v2724;	// L2943
  }
}

void PE_kernel_ws_gemm_13_6(
  hls::stream< int8_t > &v2725 /* v2725[16] */,
  hls::stream< int32_t > &v2726 /* v2726[16] */,
  hls::stream< int8_t > &v2727 /* v2727[16] */,
  hls::stream< int32_t > &v2728 /* v2728[16] */,
  int8_t v2729[16][16],
  int v2730,
  int v2731
) {	// L2947
  #pragma HLS stream variable=v2725 depth=17
  #pragma HLS stream variable=v2726 depth=17
  #pragma HLS stream variable=v2727 depth=17
  #pragma HLS stream variable=v2728 depth=17
  #pragma HLS array_partition variable=v2729 complete dim=1
  #pragma HLS array_partition variable=v2729 complete dim=2

  int8_t v2732 = v2729[v2730][v2731];	// L2948
  int8_t w109;	// L2949
  w109 = v2732;	// L2950
  l_step_m109: for (int m109 = 0; m109 < 16; m109++) {	// L2951
  #pragma HLS pipeline II=1
    int8_t v2735 = v2725.read(); // v2725[m109];	// L2952
    int8_t a109;	// L2953
    a109 = v2735;	// L2954
    int32_t v2737 = v2726.read(); // v2726[m109];	// L2955
    int32_t p109;	// L2956
    p109 = v2737;	// L2957
    int32_t v2739 = p109;	// L2958
    int8_t v2740 = a109;	// L2959
    int8_t v2741 = w109;	// L2960
    int16_t v2742 = v2740;	// L2961
    int16_t v2743 = v2741;	// L2962
    int16_t v2744 = v2742 * v2743;	// L2963
    ap_int<33> v2745 = v2739;	// L2964
    ap_int<33> v2746 = v2744;	// L2965
    ap_int<33> v2747 = v2745 + v2746;	// L2966
    int32_t v2748 = v2747;	// L2967
    v2728.write(v2748); // v2728[m109] = v2748;	// L2968
    int8_t v2749 = a109;	// L2969
    v2727.write(v2749); // v2727[m109] = v2749;	// L2970
  }
}

void PE_kernel_ws_gemm_14_6(
  hls::stream< int8_t > &v2750 /* v2750[16] */,
  hls::stream< int32_t > &v2751 /* v2751[16] */,
  hls::stream< int8_t > &v2752 /* v2752[16] */,
  hls::stream< int32_t > &v2753 /* v2753[16] */,
  int8_t v2754[16][16],
  int v2755,
  int v2756
) {	// L2974
  #pragma HLS stream variable=v2750 depth=17
  #pragma HLS stream variable=v2751 depth=17
  #pragma HLS stream variable=v2752 depth=17
  #pragma HLS stream variable=v2753 depth=17
  #pragma HLS array_partition variable=v2754 complete dim=1
  #pragma HLS array_partition variable=v2754 complete dim=2

  int8_t v2757 = v2754[v2755][v2756];	// L2975
  int8_t w110;	// L2976
  w110 = v2757;	// L2977
  l_step_m110: for (int m110 = 0; m110 < 16; m110++) {	// L2978
  #pragma HLS pipeline II=1
    int8_t v2760 = v2750.read(); // v2750[m110];	// L2979
    int8_t a110;	// L2980
    a110 = v2760;	// L2981
    int32_t v2762 = v2751.read(); // v2751[m110];	// L2982
    int32_t p110;	// L2983
    p110 = v2762;	// L2984
    int32_t v2764 = p110;	// L2985
    int8_t v2765 = a110;	// L2986
    int8_t v2766 = w110;	// L2987
    int16_t v2767 = v2765;	// L2988
    int16_t v2768 = v2766;	// L2989
    int16_t v2769 = v2767 * v2768;	// L2990
    ap_int<33> v2770 = v2764;	// L2991
    ap_int<33> v2771 = v2769;	// L2992
    ap_int<33> v2772 = v2770 + v2771;	// L2993
    int32_t v2773 = v2772;	// L2994
    v2753.write(v2773); // v2753[m110] = v2773;	// L2995
    int8_t v2774 = a110;	// L2996
    v2752.write(v2774); // v2752[m110] = v2774;	// L2997
  }
}

void PE_kernel_ws_gemm_15_6(
  hls::stream< int8_t > &v2775 /* v2775[16] */,
  hls::stream< int32_t > &v2776 /* v2776[16] */,
  hls::stream< int8_t > &v2777 /* v2777[16] */,
  hls::stream< int32_t > &v2778 /* v2778[16] */,
  int8_t v2779[16][16],
  int v2780,
  int v2781
) {	// L3001
  #pragma HLS stream variable=v2775 depth=17
  #pragma HLS stream variable=v2776 depth=17
  #pragma HLS stream variable=v2777 depth=17
  #pragma HLS stream variable=v2778 depth=17
  #pragma HLS array_partition variable=v2779 complete dim=1
  #pragma HLS array_partition variable=v2779 complete dim=2

  int8_t v2782 = v2779[v2780][v2781];	// L3002
  int8_t w111;	// L3003
  w111 = v2782;	// L3004
  l_step_m111: for (int m111 = 0; m111 < 16; m111++) {	// L3005
  #pragma HLS pipeline II=1
    int8_t v2785 = v2775.read(); // v2775[m111];	// L3006
    int8_t a111;	// L3007
    a111 = v2785;	// L3008
    int32_t v2787 = v2776.read(); // v2776[m111];	// L3009
    int32_t p111;	// L3010
    p111 = v2787;	// L3011
    int32_t v2789 = p111;	// L3012
    int8_t v2790 = a111;	// L3013
    int8_t v2791 = w111;	// L3014
    int16_t v2792 = v2790;	// L3015
    int16_t v2793 = v2791;	// L3016
    int16_t v2794 = v2792 * v2793;	// L3017
    ap_int<33> v2795 = v2789;	// L3018
    ap_int<33> v2796 = v2794;	// L3019
    ap_int<33> v2797 = v2795 + v2796;	// L3020
    int32_t v2798 = v2797;	// L3021
    v2778.write(v2798); // v2778[m111] = v2798;	// L3022
    int8_t v2799 = a111;	// L3023
    v2777.write(v2799); // v2777[m111] = v2799;	// L3024
  }
}

void PE_kernel_ws_gemm_0_7(
  hls::stream< int8_t > &v2800 /* v2800[16] */,
  hls::stream< int32_t > &v2801 /* v2801[16] */,
  hls::stream< int8_t > &v2802 /* v2802[16] */,
  hls::stream< int32_t > &v2803 /* v2803[16] */,
  int8_t v2804[16][16],
  int v2805,
  int v2806
) {	// L3028
  #pragma HLS stream variable=v2800 depth=17
  #pragma HLS stream variable=v2801 depth=17
  #pragma HLS stream variable=v2802 depth=17
  #pragma HLS stream variable=v2803 depth=17
  #pragma HLS array_partition variable=v2804 complete dim=1
  #pragma HLS array_partition variable=v2804 complete dim=2

  int8_t v2807 = v2804[v2805][v2806];	// L3029
  int8_t w112;	// L3030
  w112 = v2807;	// L3031
  l_step_m112: for (int m112 = 0; m112 < 16; m112++) {	// L3032
  #pragma HLS pipeline II=1
    int8_t v2810 = v2800.read(); // v2800[m112];	// L3033
    int8_t a112;	// L3034
    a112 = v2810;	// L3035
    int32_t v2812 = v2801.read(); // v2801[m112];	// L3036
    int32_t p112;	// L3037
    p112 = v2812;	// L3038
    int32_t v2814 = p112;	// L3039
    int8_t v2815 = a112;	// L3040
    int8_t v2816 = w112;	// L3041
    int16_t v2817 = v2815;	// L3042
    int16_t v2818 = v2816;	// L3043
    int16_t v2819 = v2817 * v2818;	// L3044
    ap_int<33> v2820 = v2814;	// L3045
    ap_int<33> v2821 = v2819;	// L3046
    ap_int<33> v2822 = v2820 + v2821;	// L3047
    int32_t v2823 = v2822;	// L3048
    v2803.write(v2823); // v2803[m112] = v2823;	// L3049
    int8_t v2824 = a112;	// L3050
    v2802.write(v2824); // v2802[m112] = v2824;	// L3051
  }
}

void PE_kernel_ws_gemm_1_7(
  hls::stream< int8_t > &v2825 /* v2825[16] */,
  hls::stream< int32_t > &v2826 /* v2826[16] */,
  hls::stream< int8_t > &v2827 /* v2827[16] */,
  hls::stream< int32_t > &v2828 /* v2828[16] */,
  int8_t v2829[16][16],
  int v2830,
  int v2831
) {	// L3055
  #pragma HLS stream variable=v2825 depth=17
  #pragma HLS stream variable=v2826 depth=17
  #pragma HLS stream variable=v2827 depth=17
  #pragma HLS stream variable=v2828 depth=17
  #pragma HLS array_partition variable=v2829 complete dim=1
  #pragma HLS array_partition variable=v2829 complete dim=2

  int8_t v2832 = v2829[v2830][v2831];	// L3056
  int8_t w113;	// L3057
  w113 = v2832;	// L3058
  l_step_m113: for (int m113 = 0; m113 < 16; m113++) {	// L3059
  #pragma HLS pipeline II=1
    int8_t v2835 = v2825.read(); // v2825[m113];	// L3060
    int8_t a113;	// L3061
    a113 = v2835;	// L3062
    int32_t v2837 = v2826.read(); // v2826[m113];	// L3063
    int32_t p113;	// L3064
    p113 = v2837;	// L3065
    int32_t v2839 = p113;	// L3066
    int8_t v2840 = a113;	// L3067
    int8_t v2841 = w113;	// L3068
    int16_t v2842 = v2840;	// L3069
    int16_t v2843 = v2841;	// L3070
    int16_t v2844 = v2842 * v2843;	// L3071
    ap_int<33> v2845 = v2839;	// L3072
    ap_int<33> v2846 = v2844;	// L3073
    ap_int<33> v2847 = v2845 + v2846;	// L3074
    int32_t v2848 = v2847;	// L3075
    v2828.write(v2848); // v2828[m113] = v2848;	// L3076
    int8_t v2849 = a113;	// L3077
    v2827.write(v2849); // v2827[m113] = v2849;	// L3078
  }
}

void PE_kernel_ws_gemm_2_7(
  hls::stream< int8_t > &v2850 /* v2850[16] */,
  hls::stream< int32_t > &v2851 /* v2851[16] */,
  hls::stream< int8_t > &v2852 /* v2852[16] */,
  hls::stream< int32_t > &v2853 /* v2853[16] */,
  int8_t v2854[16][16],
  int v2855,
  int v2856
) {	// L3082
  #pragma HLS stream variable=v2850 depth=17
  #pragma HLS stream variable=v2851 depth=17
  #pragma HLS stream variable=v2852 depth=17
  #pragma HLS stream variable=v2853 depth=17
  #pragma HLS array_partition variable=v2854 complete dim=1
  #pragma HLS array_partition variable=v2854 complete dim=2

  int8_t v2857 = v2854[v2855][v2856];	// L3083
  int8_t w114;	// L3084
  w114 = v2857;	// L3085
  l_step_m114: for (int m114 = 0; m114 < 16; m114++) {	// L3086
  #pragma HLS pipeline II=1
    int8_t v2860 = v2850.read(); // v2850[m114];	// L3087
    int8_t a114;	// L3088
    a114 = v2860;	// L3089
    int32_t v2862 = v2851.read(); // v2851[m114];	// L3090
    int32_t p114;	// L3091
    p114 = v2862;	// L3092
    int32_t v2864 = p114;	// L3093
    int8_t v2865 = a114;	// L3094
    int8_t v2866 = w114;	// L3095
    int16_t v2867 = v2865;	// L3096
    int16_t v2868 = v2866;	// L3097
    int16_t v2869 = v2867 * v2868;	// L3098
    ap_int<33> v2870 = v2864;	// L3099
    ap_int<33> v2871 = v2869;	// L3100
    ap_int<33> v2872 = v2870 + v2871;	// L3101
    int32_t v2873 = v2872;	// L3102
    v2853.write(v2873); // v2853[m114] = v2873;	// L3103
    int8_t v2874 = a114;	// L3104
    v2852.write(v2874); // v2852[m114] = v2874;	// L3105
  }
}

void PE_kernel_ws_gemm_3_7(
  hls::stream< int8_t > &v2875 /* v2875[16] */,
  hls::stream< int32_t > &v2876 /* v2876[16] */,
  hls::stream< int8_t > &v2877 /* v2877[16] */,
  hls::stream< int32_t > &v2878 /* v2878[16] */,
  int8_t v2879[16][16],
  int v2880,
  int v2881
) {	// L3109
  #pragma HLS stream variable=v2875 depth=17
  #pragma HLS stream variable=v2876 depth=17
  #pragma HLS stream variable=v2877 depth=17
  #pragma HLS stream variable=v2878 depth=17
  #pragma HLS array_partition variable=v2879 complete dim=1
  #pragma HLS array_partition variable=v2879 complete dim=2

  int8_t v2882 = v2879[v2880][v2881];	// L3110
  int8_t w115;	// L3111
  w115 = v2882;	// L3112
  l_step_m115: for (int m115 = 0; m115 < 16; m115++) {	// L3113
  #pragma HLS pipeline II=1
    int8_t v2885 = v2875.read(); // v2875[m115];	// L3114
    int8_t a115;	// L3115
    a115 = v2885;	// L3116
    int32_t v2887 = v2876.read(); // v2876[m115];	// L3117
    int32_t p115;	// L3118
    p115 = v2887;	// L3119
    int32_t v2889 = p115;	// L3120
    int8_t v2890 = a115;	// L3121
    int8_t v2891 = w115;	// L3122
    int16_t v2892 = v2890;	// L3123
    int16_t v2893 = v2891;	// L3124
    int16_t v2894 = v2892 * v2893;	// L3125
    ap_int<33> v2895 = v2889;	// L3126
    ap_int<33> v2896 = v2894;	// L3127
    ap_int<33> v2897 = v2895 + v2896;	// L3128
    int32_t v2898 = v2897;	// L3129
    v2878.write(v2898); // v2878[m115] = v2898;	// L3130
    int8_t v2899 = a115;	// L3131
    v2877.write(v2899); // v2877[m115] = v2899;	// L3132
  }
}

void PE_kernel_ws_gemm_4_7(
  hls::stream< int8_t > &v2900 /* v2900[16] */,
  hls::stream< int32_t > &v2901 /* v2901[16] */,
  hls::stream< int8_t > &v2902 /* v2902[16] */,
  hls::stream< int32_t > &v2903 /* v2903[16] */,
  int8_t v2904[16][16],
  int v2905,
  int v2906
) {	// L3136
  #pragma HLS stream variable=v2900 depth=17
  #pragma HLS stream variable=v2901 depth=17
  #pragma HLS stream variable=v2902 depth=17
  #pragma HLS stream variable=v2903 depth=17
  #pragma HLS array_partition variable=v2904 complete dim=1
  #pragma HLS array_partition variable=v2904 complete dim=2

  int8_t v2907 = v2904[v2905][v2906];	// L3137
  int8_t w116;	// L3138
  w116 = v2907;	// L3139
  l_step_m116: for (int m116 = 0; m116 < 16; m116++) {	// L3140
  #pragma HLS pipeline II=1
    int8_t v2910 = v2900.read(); // v2900[m116];	// L3141
    int8_t a116;	// L3142
    a116 = v2910;	// L3143
    int32_t v2912 = v2901.read(); // v2901[m116];	// L3144
    int32_t p116;	// L3145
    p116 = v2912;	// L3146
    int32_t v2914 = p116;	// L3147
    int8_t v2915 = a116;	// L3148
    int8_t v2916 = w116;	// L3149
    int16_t v2917 = v2915;	// L3150
    int16_t v2918 = v2916;	// L3151
    int16_t v2919 = v2917 * v2918;	// L3152
    ap_int<33> v2920 = v2914;	// L3153
    ap_int<33> v2921 = v2919;	// L3154
    ap_int<33> v2922 = v2920 + v2921;	// L3155
    int32_t v2923 = v2922;	// L3156
    v2903.write(v2923); // v2903[m116] = v2923;	// L3157
    int8_t v2924 = a116;	// L3158
    v2902.write(v2924); // v2902[m116] = v2924;	// L3159
  }
}

void PE_kernel_ws_gemm_5_7(
  hls::stream< int8_t > &v2925 /* v2925[16] */,
  hls::stream< int32_t > &v2926 /* v2926[16] */,
  hls::stream< int8_t > &v2927 /* v2927[16] */,
  hls::stream< int32_t > &v2928 /* v2928[16] */,
  int8_t v2929[16][16],
  int v2930,
  int v2931
) {	// L3163
  #pragma HLS stream variable=v2925 depth=17
  #pragma HLS stream variable=v2926 depth=17
  #pragma HLS stream variable=v2927 depth=17
  #pragma HLS stream variable=v2928 depth=17
  #pragma HLS array_partition variable=v2929 complete dim=1
  #pragma HLS array_partition variable=v2929 complete dim=2

  int8_t v2932 = v2929[v2930][v2931];	// L3164
  int8_t w117;	// L3165
  w117 = v2932;	// L3166
  l_step_m117: for (int m117 = 0; m117 < 16; m117++) {	// L3167
  #pragma HLS pipeline II=1
    int8_t v2935 = v2925.read(); // v2925[m117];	// L3168
    int8_t a117;	// L3169
    a117 = v2935;	// L3170
    int32_t v2937 = v2926.read(); // v2926[m117];	// L3171
    int32_t p117;	// L3172
    p117 = v2937;	// L3173
    int32_t v2939 = p117;	// L3174
    int8_t v2940 = a117;	// L3175
    int8_t v2941 = w117;	// L3176
    int16_t v2942 = v2940;	// L3177
    int16_t v2943 = v2941;	// L3178
    int16_t v2944 = v2942 * v2943;	// L3179
    ap_int<33> v2945 = v2939;	// L3180
    ap_int<33> v2946 = v2944;	// L3181
    ap_int<33> v2947 = v2945 + v2946;	// L3182
    int32_t v2948 = v2947;	// L3183
    v2928.write(v2948); // v2928[m117] = v2948;	// L3184
    int8_t v2949 = a117;	// L3185
    v2927.write(v2949); // v2927[m117] = v2949;	// L3186
  }
}

void PE_kernel_ws_gemm_6_7(
  hls::stream< int8_t > &v2950 /* v2950[16] */,
  hls::stream< int32_t > &v2951 /* v2951[16] */,
  hls::stream< int8_t > &v2952 /* v2952[16] */,
  hls::stream< int32_t > &v2953 /* v2953[16] */,
  int8_t v2954[16][16],
  int v2955,
  int v2956
) {	// L3190
  #pragma HLS stream variable=v2950 depth=17
  #pragma HLS stream variable=v2951 depth=17
  #pragma HLS stream variable=v2952 depth=17
  #pragma HLS stream variable=v2953 depth=17
  #pragma HLS array_partition variable=v2954 complete dim=1
  #pragma HLS array_partition variable=v2954 complete dim=2

  int8_t v2957 = v2954[v2955][v2956];	// L3191
  int8_t w118;	// L3192
  w118 = v2957;	// L3193
  l_step_m118: for (int m118 = 0; m118 < 16; m118++) {	// L3194
  #pragma HLS pipeline II=1
    int8_t v2960 = v2950.read(); // v2950[m118];	// L3195
    int8_t a118;	// L3196
    a118 = v2960;	// L3197
    int32_t v2962 = v2951.read(); // v2951[m118];	// L3198
    int32_t p118;	// L3199
    p118 = v2962;	// L3200
    int32_t v2964 = p118;	// L3201
    int8_t v2965 = a118;	// L3202
    int8_t v2966 = w118;	// L3203
    int16_t v2967 = v2965;	// L3204
    int16_t v2968 = v2966;	// L3205
    int16_t v2969 = v2967 * v2968;	// L3206
    ap_int<33> v2970 = v2964;	// L3207
    ap_int<33> v2971 = v2969;	// L3208
    ap_int<33> v2972 = v2970 + v2971;	// L3209
    int32_t v2973 = v2972;	// L3210
    v2953.write(v2973); // v2953[m118] = v2973;	// L3211
    int8_t v2974 = a118;	// L3212
    v2952.write(v2974); // v2952[m118] = v2974;	// L3213
  }
}

void PE_kernel_ws_gemm_7_7(
  hls::stream< int8_t > &v2975 /* v2975[16] */,
  hls::stream< int32_t > &v2976 /* v2976[16] */,
  hls::stream< int8_t > &v2977 /* v2977[16] */,
  hls::stream< int32_t > &v2978 /* v2978[16] */,
  int8_t v2979[16][16],
  int v2980,
  int v2981
) {	// L3217
  #pragma HLS stream variable=v2975 depth=17
  #pragma HLS stream variable=v2976 depth=17
  #pragma HLS stream variable=v2977 depth=17
  #pragma HLS stream variable=v2978 depth=17
  #pragma HLS array_partition variable=v2979 complete dim=1
  #pragma HLS array_partition variable=v2979 complete dim=2

  int8_t v2982 = v2979[v2980][v2981];	// L3218
  int8_t w119;	// L3219
  w119 = v2982;	// L3220
  l_step_m119: for (int m119 = 0; m119 < 16; m119++) {	// L3221
  #pragma HLS pipeline II=1
    int8_t v2985 = v2975.read(); // v2975[m119];	// L3222
    int8_t a119;	// L3223
    a119 = v2985;	// L3224
    int32_t v2987 = v2976.read(); // v2976[m119];	// L3225
    int32_t p119;	// L3226
    p119 = v2987;	// L3227
    int32_t v2989 = p119;	// L3228
    int8_t v2990 = a119;	// L3229
    int8_t v2991 = w119;	// L3230
    int16_t v2992 = v2990;	// L3231
    int16_t v2993 = v2991;	// L3232
    int16_t v2994 = v2992 * v2993;	// L3233
    ap_int<33> v2995 = v2989;	// L3234
    ap_int<33> v2996 = v2994;	// L3235
    ap_int<33> v2997 = v2995 + v2996;	// L3236
    int32_t v2998 = v2997;	// L3237
    v2978.write(v2998); // v2978[m119] = v2998;	// L3238
    int8_t v2999 = a119;	// L3239
    v2977.write(v2999); // v2977[m119] = v2999;	// L3240
  }
}

void PE_kernel_ws_gemm_8_7(
  hls::stream< int8_t > &v3000 /* v3000[16] */,
  hls::stream< int32_t > &v3001 /* v3001[16] */,
  hls::stream< int8_t > &v3002 /* v3002[16] */,
  hls::stream< int32_t > &v3003 /* v3003[16] */,
  int8_t v3004[16][16],
  int v3005,
  int v3006
) {	// L3244
  #pragma HLS stream variable=v3000 depth=17
  #pragma HLS stream variable=v3001 depth=17
  #pragma HLS stream variable=v3002 depth=17
  #pragma HLS stream variable=v3003 depth=17
  #pragma HLS array_partition variable=v3004 complete dim=1
  #pragma HLS array_partition variable=v3004 complete dim=2

  int8_t v3007 = v3004[v3005][v3006];	// L3245
  int8_t w120;	// L3246
  w120 = v3007;	// L3247
  l_step_m120: for (int m120 = 0; m120 < 16; m120++) {	// L3248
  #pragma HLS pipeline II=1
    int8_t v3010 = v3000.read(); // v3000[m120];	// L3249
    int8_t a120;	// L3250
    a120 = v3010;	// L3251
    int32_t v3012 = v3001.read(); // v3001[m120];	// L3252
    int32_t p120;	// L3253
    p120 = v3012;	// L3254
    int32_t v3014 = p120;	// L3255
    int8_t v3015 = a120;	// L3256
    int8_t v3016 = w120;	// L3257
    int16_t v3017 = v3015;	// L3258
    int16_t v3018 = v3016;	// L3259
    int16_t v3019 = v3017 * v3018;	// L3260
    ap_int<33> v3020 = v3014;	// L3261
    ap_int<33> v3021 = v3019;	// L3262
    ap_int<33> v3022 = v3020 + v3021;	// L3263
    int32_t v3023 = v3022;	// L3264
    v3003.write(v3023); // v3003[m120] = v3023;	// L3265
    int8_t v3024 = a120;	// L3266
    v3002.write(v3024); // v3002[m120] = v3024;	// L3267
  }
}

void PE_kernel_ws_gemm_9_7(
  hls::stream< int8_t > &v3025 /* v3025[16] */,
  hls::stream< int32_t > &v3026 /* v3026[16] */,
  hls::stream< int8_t > &v3027 /* v3027[16] */,
  hls::stream< int32_t > &v3028 /* v3028[16] */,
  int8_t v3029[16][16],
  int v3030,
  int v3031
) {	// L3271
  #pragma HLS stream variable=v3025 depth=17
  #pragma HLS stream variable=v3026 depth=17
  #pragma HLS stream variable=v3027 depth=17
  #pragma HLS stream variable=v3028 depth=17
  #pragma HLS array_partition variable=v3029 complete dim=1
  #pragma HLS array_partition variable=v3029 complete dim=2

  int8_t v3032 = v3029[v3030][v3031];	// L3272
  int8_t w121;	// L3273
  w121 = v3032;	// L3274
  l_step_m121: for (int m121 = 0; m121 < 16; m121++) {	// L3275
  #pragma HLS pipeline II=1
    int8_t v3035 = v3025.read(); // v3025[m121];	// L3276
    int8_t a121;	// L3277
    a121 = v3035;	// L3278
    int32_t v3037 = v3026.read(); // v3026[m121];	// L3279
    int32_t p121;	// L3280
    p121 = v3037;	// L3281
    int32_t v3039 = p121;	// L3282
    int8_t v3040 = a121;	// L3283
    int8_t v3041 = w121;	// L3284
    int16_t v3042 = v3040;	// L3285
    int16_t v3043 = v3041;	// L3286
    int16_t v3044 = v3042 * v3043;	// L3287
    ap_int<33> v3045 = v3039;	// L3288
    ap_int<33> v3046 = v3044;	// L3289
    ap_int<33> v3047 = v3045 + v3046;	// L3290
    int32_t v3048 = v3047;	// L3291
    v3028.write(v3048); // v3028[m121] = v3048;	// L3292
    int8_t v3049 = a121;	// L3293
    v3027.write(v3049); // v3027[m121] = v3049;	// L3294
  }
}

void PE_kernel_ws_gemm_10_7(
  hls::stream< int8_t > &v3050 /* v3050[16] */,
  hls::stream< int32_t > &v3051 /* v3051[16] */,
  hls::stream< int8_t > &v3052 /* v3052[16] */,
  hls::stream< int32_t > &v3053 /* v3053[16] */,
  int8_t v3054[16][16],
  int v3055,
  int v3056
) {	// L3298
  #pragma HLS stream variable=v3050 depth=17
  #pragma HLS stream variable=v3051 depth=17
  #pragma HLS stream variable=v3052 depth=17
  #pragma HLS stream variable=v3053 depth=17
  #pragma HLS array_partition variable=v3054 complete dim=1
  #pragma HLS array_partition variable=v3054 complete dim=2

  int8_t v3057 = v3054[v3055][v3056];	// L3299
  int8_t w122;	// L3300
  w122 = v3057;	// L3301
  l_step_m122: for (int m122 = 0; m122 < 16; m122++) {	// L3302
  #pragma HLS pipeline II=1
    int8_t v3060 = v3050.read(); // v3050[m122];	// L3303
    int8_t a122;	// L3304
    a122 = v3060;	// L3305
    int32_t v3062 = v3051.read(); // v3051[m122];	// L3306
    int32_t p122;	// L3307
    p122 = v3062;	// L3308
    int32_t v3064 = p122;	// L3309
    int8_t v3065 = a122;	// L3310
    int8_t v3066 = w122;	// L3311
    int16_t v3067 = v3065;	// L3312
    int16_t v3068 = v3066;	// L3313
    int16_t v3069 = v3067 * v3068;	// L3314
    ap_int<33> v3070 = v3064;	// L3315
    ap_int<33> v3071 = v3069;	// L3316
    ap_int<33> v3072 = v3070 + v3071;	// L3317
    int32_t v3073 = v3072;	// L3318
    v3053.write(v3073); // v3053[m122] = v3073;	// L3319
    int8_t v3074 = a122;	// L3320
    v3052.write(v3074); // v3052[m122] = v3074;	// L3321
  }
}

void PE_kernel_ws_gemm_11_7(
  hls::stream< int8_t > &v3075 /* v3075[16] */,
  hls::stream< int32_t > &v3076 /* v3076[16] */,
  hls::stream< int8_t > &v3077 /* v3077[16] */,
  hls::stream< int32_t > &v3078 /* v3078[16] */,
  int8_t v3079[16][16],
  int v3080,
  int v3081
) {	// L3325
  #pragma HLS stream variable=v3075 depth=17
  #pragma HLS stream variable=v3076 depth=17
  #pragma HLS stream variable=v3077 depth=17
  #pragma HLS stream variable=v3078 depth=17
  #pragma HLS array_partition variable=v3079 complete dim=1
  #pragma HLS array_partition variable=v3079 complete dim=2

  int8_t v3082 = v3079[v3080][v3081];	// L3326
  int8_t w123;	// L3327
  w123 = v3082;	// L3328
  l_step_m123: for (int m123 = 0; m123 < 16; m123++) {	// L3329
  #pragma HLS pipeline II=1
    int8_t v3085 = v3075.read(); // v3075[m123];	// L3330
    int8_t a123;	// L3331
    a123 = v3085;	// L3332
    int32_t v3087 = v3076.read(); // v3076[m123];	// L3333
    int32_t p123;	// L3334
    p123 = v3087;	// L3335
    int32_t v3089 = p123;	// L3336
    int8_t v3090 = a123;	// L3337
    int8_t v3091 = w123;	// L3338
    int16_t v3092 = v3090;	// L3339
    int16_t v3093 = v3091;	// L3340
    int16_t v3094 = v3092 * v3093;	// L3341
    ap_int<33> v3095 = v3089;	// L3342
    ap_int<33> v3096 = v3094;	// L3343
    ap_int<33> v3097 = v3095 + v3096;	// L3344
    int32_t v3098 = v3097;	// L3345
    v3078.write(v3098); // v3078[m123] = v3098;	// L3346
    int8_t v3099 = a123;	// L3347
    v3077.write(v3099); // v3077[m123] = v3099;	// L3348
  }
}

void PE_kernel_ws_gemm_12_7(
  hls::stream< int8_t > &v3100 /* v3100[16] */,
  hls::stream< int32_t > &v3101 /* v3101[16] */,
  hls::stream< int8_t > &v3102 /* v3102[16] */,
  hls::stream< int32_t > &v3103 /* v3103[16] */,
  int8_t v3104[16][16],
  int v3105,
  int v3106
) {	// L3352
  #pragma HLS stream variable=v3100 depth=17
  #pragma HLS stream variable=v3101 depth=17
  #pragma HLS stream variable=v3102 depth=17
  #pragma HLS stream variable=v3103 depth=17
  #pragma HLS array_partition variable=v3104 complete dim=1
  #pragma HLS array_partition variable=v3104 complete dim=2

  int8_t v3107 = v3104[v3105][v3106];	// L3353
  int8_t w124;	// L3354
  w124 = v3107;	// L3355
  l_step_m124: for (int m124 = 0; m124 < 16; m124++) {	// L3356
  #pragma HLS pipeline II=1
    int8_t v3110 = v3100.read(); // v3100[m124];	// L3357
    int8_t a124;	// L3358
    a124 = v3110;	// L3359
    int32_t v3112 = v3101.read(); // v3101[m124];	// L3360
    int32_t p124;	// L3361
    p124 = v3112;	// L3362
    int32_t v3114 = p124;	// L3363
    int8_t v3115 = a124;	// L3364
    int8_t v3116 = w124;	// L3365
    int16_t v3117 = v3115;	// L3366
    int16_t v3118 = v3116;	// L3367
    int16_t v3119 = v3117 * v3118;	// L3368
    ap_int<33> v3120 = v3114;	// L3369
    ap_int<33> v3121 = v3119;	// L3370
    ap_int<33> v3122 = v3120 + v3121;	// L3371
    int32_t v3123 = v3122;	// L3372
    v3103.write(v3123); // v3103[m124] = v3123;	// L3373
    int8_t v3124 = a124;	// L3374
    v3102.write(v3124); // v3102[m124] = v3124;	// L3375
  }
}

void PE_kernel_ws_gemm_13_7(
  hls::stream< int8_t > &v3125 /* v3125[16] */,
  hls::stream< int32_t > &v3126 /* v3126[16] */,
  hls::stream< int8_t > &v3127 /* v3127[16] */,
  hls::stream< int32_t > &v3128 /* v3128[16] */,
  int8_t v3129[16][16],
  int v3130,
  int v3131
) {	// L3379
  #pragma HLS stream variable=v3125 depth=17
  #pragma HLS stream variable=v3126 depth=17
  #pragma HLS stream variable=v3127 depth=17
  #pragma HLS stream variable=v3128 depth=17
  #pragma HLS array_partition variable=v3129 complete dim=1
  #pragma HLS array_partition variable=v3129 complete dim=2

  int8_t v3132 = v3129[v3130][v3131];	// L3380
  int8_t w125;	// L3381
  w125 = v3132;	// L3382
  l_step_m125: for (int m125 = 0; m125 < 16; m125++) {	// L3383
  #pragma HLS pipeline II=1
    int8_t v3135 = v3125.read(); // v3125[m125];	// L3384
    int8_t a125;	// L3385
    a125 = v3135;	// L3386
    int32_t v3137 = v3126.read(); // v3126[m125];	// L3387
    int32_t p125;	// L3388
    p125 = v3137;	// L3389
    int32_t v3139 = p125;	// L3390
    int8_t v3140 = a125;	// L3391
    int8_t v3141 = w125;	// L3392
    int16_t v3142 = v3140;	// L3393
    int16_t v3143 = v3141;	// L3394
    int16_t v3144 = v3142 * v3143;	// L3395
    ap_int<33> v3145 = v3139;	// L3396
    ap_int<33> v3146 = v3144;	// L3397
    ap_int<33> v3147 = v3145 + v3146;	// L3398
    int32_t v3148 = v3147;	// L3399
    v3128.write(v3148); // v3128[m125] = v3148;	// L3400
    int8_t v3149 = a125;	// L3401
    v3127.write(v3149); // v3127[m125] = v3149;	// L3402
  }
}

void PE_kernel_ws_gemm_14_7(
  hls::stream< int8_t > &v3150 /* v3150[16] */,
  hls::stream< int32_t > &v3151 /* v3151[16] */,
  hls::stream< int8_t > &v3152 /* v3152[16] */,
  hls::stream< int32_t > &v3153 /* v3153[16] */,
  int8_t v3154[16][16],
  int v3155,
  int v3156
) {	// L3406
  #pragma HLS stream variable=v3150 depth=17
  #pragma HLS stream variable=v3151 depth=17
  #pragma HLS stream variable=v3152 depth=17
  #pragma HLS stream variable=v3153 depth=17
  #pragma HLS array_partition variable=v3154 complete dim=1
  #pragma HLS array_partition variable=v3154 complete dim=2

  int8_t v3157 = v3154[v3155][v3156];	// L3407
  int8_t w126;	// L3408
  w126 = v3157;	// L3409
  l_step_m126: for (int m126 = 0; m126 < 16; m126++) {	// L3410
  #pragma HLS pipeline II=1
    int8_t v3160 = v3150.read(); // v3150[m126];	// L3411
    int8_t a126;	// L3412
    a126 = v3160;	// L3413
    int32_t v3162 = v3151.read(); // v3151[m126];	// L3414
    int32_t p126;	// L3415
    p126 = v3162;	// L3416
    int32_t v3164 = p126;	// L3417
    int8_t v3165 = a126;	// L3418
    int8_t v3166 = w126;	// L3419
    int16_t v3167 = v3165;	// L3420
    int16_t v3168 = v3166;	// L3421
    int16_t v3169 = v3167 * v3168;	// L3422
    ap_int<33> v3170 = v3164;	// L3423
    ap_int<33> v3171 = v3169;	// L3424
    ap_int<33> v3172 = v3170 + v3171;	// L3425
    int32_t v3173 = v3172;	// L3426
    v3153.write(v3173); // v3153[m126] = v3173;	// L3427
    int8_t v3174 = a126;	// L3428
    v3152.write(v3174); // v3152[m126] = v3174;	// L3429
  }
}

void PE_kernel_ws_gemm_15_7(
  hls::stream< int8_t > &v3175 /* v3175[16] */,
  hls::stream< int32_t > &v3176 /* v3176[16] */,
  hls::stream< int8_t > &v3177 /* v3177[16] */,
  hls::stream< int32_t > &v3178 /* v3178[16] */,
  int8_t v3179[16][16],
  int v3180,
  int v3181
) {	// L3433
  #pragma HLS stream variable=v3175 depth=17
  #pragma HLS stream variable=v3176 depth=17
  #pragma HLS stream variable=v3177 depth=17
  #pragma HLS stream variable=v3178 depth=17
  #pragma HLS array_partition variable=v3179 complete dim=1
  #pragma HLS array_partition variable=v3179 complete dim=2

  int8_t v3182 = v3179[v3180][v3181];	// L3434
  int8_t w127;	// L3435
  w127 = v3182;	// L3436
  l_step_m127: for (int m127 = 0; m127 < 16; m127++) {	// L3437
  #pragma HLS pipeline II=1
    int8_t v3185 = v3175.read(); // v3175[m127];	// L3438
    int8_t a127;	// L3439
    a127 = v3185;	// L3440
    int32_t v3187 = v3176.read(); // v3176[m127];	// L3441
    int32_t p127;	// L3442
    p127 = v3187;	// L3443
    int32_t v3189 = p127;	// L3444
    int8_t v3190 = a127;	// L3445
    int8_t v3191 = w127;	// L3446
    int16_t v3192 = v3190;	// L3447
    int16_t v3193 = v3191;	// L3448
    int16_t v3194 = v3192 * v3193;	// L3449
    ap_int<33> v3195 = v3189;	// L3450
    ap_int<33> v3196 = v3194;	// L3451
    ap_int<33> v3197 = v3195 + v3196;	// L3452
    int32_t v3198 = v3197;	// L3453
    v3178.write(v3198); // v3178[m127] = v3198;	// L3454
    int8_t v3199 = a127;	// L3455
    v3177.write(v3199); // v3177[m127] = v3199;	// L3456
  }
}

void PE_kernel_ws_gemm_0_8(
  hls::stream< int8_t > &v3200 /* v3200[16] */,
  hls::stream< int32_t > &v3201 /* v3201[16] */,
  hls::stream< int8_t > &v3202 /* v3202[16] */,
  hls::stream< int32_t > &v3203 /* v3203[16] */,
  int8_t v3204[16][16],
  int v3205,
  int v3206
) {	// L3460
  #pragma HLS stream variable=v3200 depth=17
  #pragma HLS stream variable=v3201 depth=17
  #pragma HLS stream variable=v3202 depth=17
  #pragma HLS stream variable=v3203 depth=17
  #pragma HLS array_partition variable=v3204 complete dim=1
  #pragma HLS array_partition variable=v3204 complete dim=2

  int8_t v3207 = v3204[v3205][v3206];	// L3461
  int8_t w128;	// L3462
  w128 = v3207;	// L3463
  l_step_m128: for (int m128 = 0; m128 < 16; m128++) {	// L3464
  #pragma HLS pipeline II=1
    int8_t v3210 = v3200.read(); // v3200[m128];	// L3465
    int8_t a128;	// L3466
    a128 = v3210;	// L3467
    int32_t v3212 = v3201.read(); // v3201[m128];	// L3468
    int32_t p128;	// L3469
    p128 = v3212;	// L3470
    int32_t v3214 = p128;	// L3471
    int8_t v3215 = a128;	// L3472
    int8_t v3216 = w128;	// L3473
    int16_t v3217 = v3215;	// L3474
    int16_t v3218 = v3216;	// L3475
    int16_t v3219 = v3217 * v3218;	// L3476
    ap_int<33> v3220 = v3214;	// L3477
    ap_int<33> v3221 = v3219;	// L3478
    ap_int<33> v3222 = v3220 + v3221;	// L3479
    int32_t v3223 = v3222;	// L3480
    v3203.write(v3223); // v3203[m128] = v3223;	// L3481
    int8_t v3224 = a128;	// L3482
    v3202.write(v3224); // v3202[m128] = v3224;	// L3483
  }
}

void PE_kernel_ws_gemm_1_8(
  hls::stream< int8_t > &v3225 /* v3225[16] */,
  hls::stream< int32_t > &v3226 /* v3226[16] */,
  hls::stream< int8_t > &v3227 /* v3227[16] */,
  hls::stream< int32_t > &v3228 /* v3228[16] */,
  int8_t v3229[16][16],
  int v3230,
  int v3231
) {	// L3487
  #pragma HLS stream variable=v3225 depth=17
  #pragma HLS stream variable=v3226 depth=17
  #pragma HLS stream variable=v3227 depth=17
  #pragma HLS stream variable=v3228 depth=17
  #pragma HLS array_partition variable=v3229 complete dim=1
  #pragma HLS array_partition variable=v3229 complete dim=2

  int8_t v3232 = v3229[v3230][v3231];	// L3488
  int8_t w129;	// L3489
  w129 = v3232;	// L3490
  l_step_m129: for (int m129 = 0; m129 < 16; m129++) {	// L3491
  #pragma HLS pipeline II=1
    int8_t v3235 = v3225.read(); // v3225[m129];	// L3492
    int8_t a129;	// L3493
    a129 = v3235;	// L3494
    int32_t v3237 = v3226.read(); // v3226[m129];	// L3495
    int32_t p129;	// L3496
    p129 = v3237;	// L3497
    int32_t v3239 = p129;	// L3498
    int8_t v3240 = a129;	// L3499
    int8_t v3241 = w129;	// L3500
    int16_t v3242 = v3240;	// L3501
    int16_t v3243 = v3241;	// L3502
    int16_t v3244 = v3242 * v3243;	// L3503
    ap_int<33> v3245 = v3239;	// L3504
    ap_int<33> v3246 = v3244;	// L3505
    ap_int<33> v3247 = v3245 + v3246;	// L3506
    int32_t v3248 = v3247;	// L3507
    v3228.write(v3248); // v3228[m129] = v3248;	// L3508
    int8_t v3249 = a129;	// L3509
    v3227.write(v3249); // v3227[m129] = v3249;	// L3510
  }
}

void PE_kernel_ws_gemm_2_8(
  hls::stream< int8_t > &v3250 /* v3250[16] */,
  hls::stream< int32_t > &v3251 /* v3251[16] */,
  hls::stream< int8_t > &v3252 /* v3252[16] */,
  hls::stream< int32_t > &v3253 /* v3253[16] */,
  int8_t v3254[16][16],
  int v3255,
  int v3256
) {	// L3514
  #pragma HLS stream variable=v3250 depth=17
  #pragma HLS stream variable=v3251 depth=17
  #pragma HLS stream variable=v3252 depth=17
  #pragma HLS stream variable=v3253 depth=17
  #pragma HLS array_partition variable=v3254 complete dim=1
  #pragma HLS array_partition variable=v3254 complete dim=2

  int8_t v3257 = v3254[v3255][v3256];	// L3515
  int8_t w130;	// L3516
  w130 = v3257;	// L3517
  l_step_m130: for (int m130 = 0; m130 < 16; m130++) {	// L3518
  #pragma HLS pipeline II=1
    int8_t v3260 = v3250.read(); // v3250[m130];	// L3519
    int8_t a130;	// L3520
    a130 = v3260;	// L3521
    int32_t v3262 = v3251.read(); // v3251[m130];	// L3522
    int32_t p130;	// L3523
    p130 = v3262;	// L3524
    int32_t v3264 = p130;	// L3525
    int8_t v3265 = a130;	// L3526
    int8_t v3266 = w130;	// L3527
    int16_t v3267 = v3265;	// L3528
    int16_t v3268 = v3266;	// L3529
    int16_t v3269 = v3267 * v3268;	// L3530
    ap_int<33> v3270 = v3264;	// L3531
    ap_int<33> v3271 = v3269;	// L3532
    ap_int<33> v3272 = v3270 + v3271;	// L3533
    int32_t v3273 = v3272;	// L3534
    v3253.write(v3273); // v3253[m130] = v3273;	// L3535
    int8_t v3274 = a130;	// L3536
    v3252.write(v3274); // v3252[m130] = v3274;	// L3537
  }
}

void PE_kernel_ws_gemm_3_8(
  hls::stream< int8_t > &v3275 /* v3275[16] */,
  hls::stream< int32_t > &v3276 /* v3276[16] */,
  hls::stream< int8_t > &v3277 /* v3277[16] */,
  hls::stream< int32_t > &v3278 /* v3278[16] */,
  int8_t v3279[16][16],
  int v3280,
  int v3281
) {	// L3541
  #pragma HLS stream variable=v3275 depth=17
  #pragma HLS stream variable=v3276 depth=17
  #pragma HLS stream variable=v3277 depth=17
  #pragma HLS stream variable=v3278 depth=17
  #pragma HLS array_partition variable=v3279 complete dim=1
  #pragma HLS array_partition variable=v3279 complete dim=2

  int8_t v3282 = v3279[v3280][v3281];	// L3542
  int8_t w131;	// L3543
  w131 = v3282;	// L3544
  l_step_m131: for (int m131 = 0; m131 < 16; m131++) {	// L3545
  #pragma HLS pipeline II=1
    int8_t v3285 = v3275.read(); // v3275[m131];	// L3546
    int8_t a131;	// L3547
    a131 = v3285;	// L3548
    int32_t v3287 = v3276.read(); // v3276[m131];	// L3549
    int32_t p131;	// L3550
    p131 = v3287;	// L3551
    int32_t v3289 = p131;	// L3552
    int8_t v3290 = a131;	// L3553
    int8_t v3291 = w131;	// L3554
    int16_t v3292 = v3290;	// L3555
    int16_t v3293 = v3291;	// L3556
    int16_t v3294 = v3292 * v3293;	// L3557
    ap_int<33> v3295 = v3289;	// L3558
    ap_int<33> v3296 = v3294;	// L3559
    ap_int<33> v3297 = v3295 + v3296;	// L3560
    int32_t v3298 = v3297;	// L3561
    v3278.write(v3298); // v3278[m131] = v3298;	// L3562
    int8_t v3299 = a131;	// L3563
    v3277.write(v3299); // v3277[m131] = v3299;	// L3564
  }
}

void PE_kernel_ws_gemm_4_8(
  hls::stream< int8_t > &v3300 /* v3300[16] */,
  hls::stream< int32_t > &v3301 /* v3301[16] */,
  hls::stream< int8_t > &v3302 /* v3302[16] */,
  hls::stream< int32_t > &v3303 /* v3303[16] */,
  int8_t v3304[16][16],
  int v3305,
  int v3306
) {	// L3568
  #pragma HLS stream variable=v3300 depth=17
  #pragma HLS stream variable=v3301 depth=17
  #pragma HLS stream variable=v3302 depth=17
  #pragma HLS stream variable=v3303 depth=17
  #pragma HLS array_partition variable=v3304 complete dim=1
  #pragma HLS array_partition variable=v3304 complete dim=2

  int8_t v3307 = v3304[v3305][v3306];	// L3569
  int8_t w132;	// L3570
  w132 = v3307;	// L3571
  l_step_m132: for (int m132 = 0; m132 < 16; m132++) {	// L3572
  #pragma HLS pipeline II=1
    int8_t v3310 = v3300.read(); // v3300[m132];	// L3573
    int8_t a132;	// L3574
    a132 = v3310;	// L3575
    int32_t v3312 = v3301.read(); // v3301[m132];	// L3576
    int32_t p132;	// L3577
    p132 = v3312;	// L3578
    int32_t v3314 = p132;	// L3579
    int8_t v3315 = a132;	// L3580
    int8_t v3316 = w132;	// L3581
    int16_t v3317 = v3315;	// L3582
    int16_t v3318 = v3316;	// L3583
    int16_t v3319 = v3317 * v3318;	// L3584
    ap_int<33> v3320 = v3314;	// L3585
    ap_int<33> v3321 = v3319;	// L3586
    ap_int<33> v3322 = v3320 + v3321;	// L3587
    int32_t v3323 = v3322;	// L3588
    v3303.write(v3323); // v3303[m132] = v3323;	// L3589
    int8_t v3324 = a132;	// L3590
    v3302.write(v3324); // v3302[m132] = v3324;	// L3591
  }
}

void PE_kernel_ws_gemm_5_8(
  hls::stream< int8_t > &v3325 /* v3325[16] */,
  hls::stream< int32_t > &v3326 /* v3326[16] */,
  hls::stream< int8_t > &v3327 /* v3327[16] */,
  hls::stream< int32_t > &v3328 /* v3328[16] */,
  int8_t v3329[16][16],
  int v3330,
  int v3331
) {	// L3595
  #pragma HLS stream variable=v3325 depth=17
  #pragma HLS stream variable=v3326 depth=17
  #pragma HLS stream variable=v3327 depth=17
  #pragma HLS stream variable=v3328 depth=17
  #pragma HLS array_partition variable=v3329 complete dim=1
  #pragma HLS array_partition variable=v3329 complete dim=2

  int8_t v3332 = v3329[v3330][v3331];	// L3596
  int8_t w133;	// L3597
  w133 = v3332;	// L3598
  l_step_m133: for (int m133 = 0; m133 < 16; m133++) {	// L3599
  #pragma HLS pipeline II=1
    int8_t v3335 = v3325.read(); // v3325[m133];	// L3600
    int8_t a133;	// L3601
    a133 = v3335;	// L3602
    int32_t v3337 = v3326.read(); // v3326[m133];	// L3603
    int32_t p133;	// L3604
    p133 = v3337;	// L3605
    int32_t v3339 = p133;	// L3606
    int8_t v3340 = a133;	// L3607
    int8_t v3341 = w133;	// L3608
    int16_t v3342 = v3340;	// L3609
    int16_t v3343 = v3341;	// L3610
    int16_t v3344 = v3342 * v3343;	// L3611
    ap_int<33> v3345 = v3339;	// L3612
    ap_int<33> v3346 = v3344;	// L3613
    ap_int<33> v3347 = v3345 + v3346;	// L3614
    int32_t v3348 = v3347;	// L3615
    v3328.write(v3348); // v3328[m133] = v3348;	// L3616
    int8_t v3349 = a133;	// L3617
    v3327.write(v3349); // v3327[m133] = v3349;	// L3618
  }
}

void PE_kernel_ws_gemm_6_8(
  hls::stream< int8_t > &v3350 /* v3350[16] */,
  hls::stream< int32_t > &v3351 /* v3351[16] */,
  hls::stream< int8_t > &v3352 /* v3352[16] */,
  hls::stream< int32_t > &v3353 /* v3353[16] */,
  int8_t v3354[16][16],
  int v3355,
  int v3356
) {	// L3622
  #pragma HLS stream variable=v3350 depth=17
  #pragma HLS stream variable=v3351 depth=17
  #pragma HLS stream variable=v3352 depth=17
  #pragma HLS stream variable=v3353 depth=17
  #pragma HLS array_partition variable=v3354 complete dim=1
  #pragma HLS array_partition variable=v3354 complete dim=2

  int8_t v3357 = v3354[v3355][v3356];	// L3623
  int8_t w134;	// L3624
  w134 = v3357;	// L3625
  l_step_m134: for (int m134 = 0; m134 < 16; m134++) {	// L3626
  #pragma HLS pipeline II=1
    int8_t v3360 = v3350.read(); // v3350[m134];	// L3627
    int8_t a134;	// L3628
    a134 = v3360;	// L3629
    int32_t v3362 = v3351.read(); // v3351[m134];	// L3630
    int32_t p134;	// L3631
    p134 = v3362;	// L3632
    int32_t v3364 = p134;	// L3633
    int8_t v3365 = a134;	// L3634
    int8_t v3366 = w134;	// L3635
    int16_t v3367 = v3365;	// L3636
    int16_t v3368 = v3366;	// L3637
    int16_t v3369 = v3367 * v3368;	// L3638
    ap_int<33> v3370 = v3364;	// L3639
    ap_int<33> v3371 = v3369;	// L3640
    ap_int<33> v3372 = v3370 + v3371;	// L3641
    int32_t v3373 = v3372;	// L3642
    v3353.write(v3373); // v3353[m134] = v3373;	// L3643
    int8_t v3374 = a134;	// L3644
    v3352.write(v3374); // v3352[m134] = v3374;	// L3645
  }
}

void PE_kernel_ws_gemm_7_8(
  hls::stream< int8_t > &v3375 /* v3375[16] */,
  hls::stream< int32_t > &v3376 /* v3376[16] */,
  hls::stream< int8_t > &v3377 /* v3377[16] */,
  hls::stream< int32_t > &v3378 /* v3378[16] */,
  int8_t v3379[16][16],
  int v3380,
  int v3381
) {	// L3649
  #pragma HLS stream variable=v3375 depth=17
  #pragma HLS stream variable=v3376 depth=17
  #pragma HLS stream variable=v3377 depth=17
  #pragma HLS stream variable=v3378 depth=17
  #pragma HLS array_partition variable=v3379 complete dim=1
  #pragma HLS array_partition variable=v3379 complete dim=2

  int8_t v3382 = v3379[v3380][v3381];	// L3650
  int8_t w135;	// L3651
  w135 = v3382;	// L3652
  l_step_m135: for (int m135 = 0; m135 < 16; m135++) {	// L3653
  #pragma HLS pipeline II=1
    int8_t v3385 = v3375.read(); // v3375[m135];	// L3654
    int8_t a135;	// L3655
    a135 = v3385;	// L3656
    int32_t v3387 = v3376.read(); // v3376[m135];	// L3657
    int32_t p135;	// L3658
    p135 = v3387;	// L3659
    int32_t v3389 = p135;	// L3660
    int8_t v3390 = a135;	// L3661
    int8_t v3391 = w135;	// L3662
    int16_t v3392 = v3390;	// L3663
    int16_t v3393 = v3391;	// L3664
    int16_t v3394 = v3392 * v3393;	// L3665
    ap_int<33> v3395 = v3389;	// L3666
    ap_int<33> v3396 = v3394;	// L3667
    ap_int<33> v3397 = v3395 + v3396;	// L3668
    int32_t v3398 = v3397;	// L3669
    v3378.write(v3398); // v3378[m135] = v3398;	// L3670
    int8_t v3399 = a135;	// L3671
    v3377.write(v3399); // v3377[m135] = v3399;	// L3672
  }
}

void PE_kernel_ws_gemm_8_8(
  hls::stream< int8_t > &v3400 /* v3400[16] */,
  hls::stream< int32_t > &v3401 /* v3401[16] */,
  hls::stream< int8_t > &v3402 /* v3402[16] */,
  hls::stream< int32_t > &v3403 /* v3403[16] */,
  int8_t v3404[16][16],
  int v3405,
  int v3406
) {	// L3676
  #pragma HLS stream variable=v3400 depth=17
  #pragma HLS stream variable=v3401 depth=17
  #pragma HLS stream variable=v3402 depth=17
  #pragma HLS stream variable=v3403 depth=17
  #pragma HLS array_partition variable=v3404 complete dim=1
  #pragma HLS array_partition variable=v3404 complete dim=2

  int8_t v3407 = v3404[v3405][v3406];	// L3677
  int8_t w136;	// L3678
  w136 = v3407;	// L3679
  l_step_m136: for (int m136 = 0; m136 < 16; m136++) {	// L3680
  #pragma HLS pipeline II=1
    int8_t v3410 = v3400.read(); // v3400[m136];	// L3681
    int8_t a136;	// L3682
    a136 = v3410;	// L3683
    int32_t v3412 = v3401.read(); // v3401[m136];	// L3684
    int32_t p136;	// L3685
    p136 = v3412;	// L3686
    int32_t v3414 = p136;	// L3687
    int8_t v3415 = a136;	// L3688
    int8_t v3416 = w136;	// L3689
    int16_t v3417 = v3415;	// L3690
    int16_t v3418 = v3416;	// L3691
    int16_t v3419 = v3417 * v3418;	// L3692
    ap_int<33> v3420 = v3414;	// L3693
    ap_int<33> v3421 = v3419;	// L3694
    ap_int<33> v3422 = v3420 + v3421;	// L3695
    int32_t v3423 = v3422;	// L3696
    v3403.write(v3423); // v3403[m136] = v3423;	// L3697
    int8_t v3424 = a136;	// L3698
    v3402.write(v3424); // v3402[m136] = v3424;	// L3699
  }
}

void PE_kernel_ws_gemm_9_8(
  hls::stream< int8_t > &v3425 /* v3425[16] */,
  hls::stream< int32_t > &v3426 /* v3426[16] */,
  hls::stream< int8_t > &v3427 /* v3427[16] */,
  hls::stream< int32_t > &v3428 /* v3428[16] */,
  int8_t v3429[16][16],
  int v3430,
  int v3431
) {	// L3703
  #pragma HLS stream variable=v3425 depth=17
  #pragma HLS stream variable=v3426 depth=17
  #pragma HLS stream variable=v3427 depth=17
  #pragma HLS stream variable=v3428 depth=17
  #pragma HLS array_partition variable=v3429 complete dim=1
  #pragma HLS array_partition variable=v3429 complete dim=2

  int8_t v3432 = v3429[v3430][v3431];	// L3704
  int8_t w137;	// L3705
  w137 = v3432;	// L3706
  l_step_m137: for (int m137 = 0; m137 < 16; m137++) {	// L3707
  #pragma HLS pipeline II=1
    int8_t v3435 = v3425.read(); // v3425[m137];	// L3708
    int8_t a137;	// L3709
    a137 = v3435;	// L3710
    int32_t v3437 = v3426.read(); // v3426[m137];	// L3711
    int32_t p137;	// L3712
    p137 = v3437;	// L3713
    int32_t v3439 = p137;	// L3714
    int8_t v3440 = a137;	// L3715
    int8_t v3441 = w137;	// L3716
    int16_t v3442 = v3440;	// L3717
    int16_t v3443 = v3441;	// L3718
    int16_t v3444 = v3442 * v3443;	// L3719
    ap_int<33> v3445 = v3439;	// L3720
    ap_int<33> v3446 = v3444;	// L3721
    ap_int<33> v3447 = v3445 + v3446;	// L3722
    int32_t v3448 = v3447;	// L3723
    v3428.write(v3448); // v3428[m137] = v3448;	// L3724
    int8_t v3449 = a137;	// L3725
    v3427.write(v3449); // v3427[m137] = v3449;	// L3726
  }
}

void PE_kernel_ws_gemm_10_8(
  hls::stream< int8_t > &v3450 /* v3450[16] */,
  hls::stream< int32_t > &v3451 /* v3451[16] */,
  hls::stream< int8_t > &v3452 /* v3452[16] */,
  hls::stream< int32_t > &v3453 /* v3453[16] */,
  int8_t v3454[16][16],
  int v3455,
  int v3456
) {	// L3730
  #pragma HLS stream variable=v3450 depth=17
  #pragma HLS stream variable=v3451 depth=17
  #pragma HLS stream variable=v3452 depth=17
  #pragma HLS stream variable=v3453 depth=17
  #pragma HLS array_partition variable=v3454 complete dim=1
  #pragma HLS array_partition variable=v3454 complete dim=2

  int8_t v3457 = v3454[v3455][v3456];	// L3731
  int8_t w138;	// L3732
  w138 = v3457;	// L3733
  l_step_m138: for (int m138 = 0; m138 < 16; m138++) {	// L3734
  #pragma HLS pipeline II=1
    int8_t v3460 = v3450.read(); // v3450[m138];	// L3735
    int8_t a138;	// L3736
    a138 = v3460;	// L3737
    int32_t v3462 = v3451.read(); // v3451[m138];	// L3738
    int32_t p138;	// L3739
    p138 = v3462;	// L3740
    int32_t v3464 = p138;	// L3741
    int8_t v3465 = a138;	// L3742
    int8_t v3466 = w138;	// L3743
    int16_t v3467 = v3465;	// L3744
    int16_t v3468 = v3466;	// L3745
    int16_t v3469 = v3467 * v3468;	// L3746
    ap_int<33> v3470 = v3464;	// L3747
    ap_int<33> v3471 = v3469;	// L3748
    ap_int<33> v3472 = v3470 + v3471;	// L3749
    int32_t v3473 = v3472;	// L3750
    v3453.write(v3473); // v3453[m138] = v3473;	// L3751
    int8_t v3474 = a138;	// L3752
    v3452.write(v3474); // v3452[m138] = v3474;	// L3753
  }
}

void PE_kernel_ws_gemm_11_8(
  hls::stream< int8_t > &v3475 /* v3475[16] */,
  hls::stream< int32_t > &v3476 /* v3476[16] */,
  hls::stream< int8_t > &v3477 /* v3477[16] */,
  hls::stream< int32_t > &v3478 /* v3478[16] */,
  int8_t v3479[16][16],
  int v3480,
  int v3481
) {	// L3757
  #pragma HLS stream variable=v3475 depth=17
  #pragma HLS stream variable=v3476 depth=17
  #pragma HLS stream variable=v3477 depth=17
  #pragma HLS stream variable=v3478 depth=17
  #pragma HLS array_partition variable=v3479 complete dim=1
  #pragma HLS array_partition variable=v3479 complete dim=2

  int8_t v3482 = v3479[v3480][v3481];	// L3758
  int8_t w139;	// L3759
  w139 = v3482;	// L3760
  l_step_m139: for (int m139 = 0; m139 < 16; m139++) {	// L3761
  #pragma HLS pipeline II=1
    int8_t v3485 = v3475.read(); // v3475[m139];	// L3762
    int8_t a139;	// L3763
    a139 = v3485;	// L3764
    int32_t v3487 = v3476.read(); // v3476[m139];	// L3765
    int32_t p139;	// L3766
    p139 = v3487;	// L3767
    int32_t v3489 = p139;	// L3768
    int8_t v3490 = a139;	// L3769
    int8_t v3491 = w139;	// L3770
    int16_t v3492 = v3490;	// L3771
    int16_t v3493 = v3491;	// L3772
    int16_t v3494 = v3492 * v3493;	// L3773
    ap_int<33> v3495 = v3489;	// L3774
    ap_int<33> v3496 = v3494;	// L3775
    ap_int<33> v3497 = v3495 + v3496;	// L3776
    int32_t v3498 = v3497;	// L3777
    v3478.write(v3498); // v3478[m139] = v3498;	// L3778
    int8_t v3499 = a139;	// L3779
    v3477.write(v3499); // v3477[m139] = v3499;	// L3780
  }
}

void PE_kernel_ws_gemm_12_8(
  hls::stream< int8_t > &v3500 /* v3500[16] */,
  hls::stream< int32_t > &v3501 /* v3501[16] */,
  hls::stream< int8_t > &v3502 /* v3502[16] */,
  hls::stream< int32_t > &v3503 /* v3503[16] */,
  int8_t v3504[16][16],
  int v3505,
  int v3506
) {	// L3784
  #pragma HLS stream variable=v3500 depth=17
  #pragma HLS stream variable=v3501 depth=17
  #pragma HLS stream variable=v3502 depth=17
  #pragma HLS stream variable=v3503 depth=17
  #pragma HLS array_partition variable=v3504 complete dim=1
  #pragma HLS array_partition variable=v3504 complete dim=2

  int8_t v3507 = v3504[v3505][v3506];	// L3785
  int8_t w140;	// L3786
  w140 = v3507;	// L3787
  l_step_m140: for (int m140 = 0; m140 < 16; m140++) {	// L3788
  #pragma HLS pipeline II=1
    int8_t v3510 = v3500.read(); // v3500[m140];	// L3789
    int8_t a140;	// L3790
    a140 = v3510;	// L3791
    int32_t v3512 = v3501.read(); // v3501[m140];	// L3792
    int32_t p140;	// L3793
    p140 = v3512;	// L3794
    int32_t v3514 = p140;	// L3795
    int8_t v3515 = a140;	// L3796
    int8_t v3516 = w140;	// L3797
    int16_t v3517 = v3515;	// L3798
    int16_t v3518 = v3516;	// L3799
    int16_t v3519 = v3517 * v3518;	// L3800
    ap_int<33> v3520 = v3514;	// L3801
    ap_int<33> v3521 = v3519;	// L3802
    ap_int<33> v3522 = v3520 + v3521;	// L3803
    int32_t v3523 = v3522;	// L3804
    v3503.write(v3523); // v3503[m140] = v3523;	// L3805
    int8_t v3524 = a140;	// L3806
    v3502.write(v3524); // v3502[m140] = v3524;	// L3807
  }
}

void PE_kernel_ws_gemm_13_8(
  hls::stream< int8_t > &v3525 /* v3525[16] */,
  hls::stream< int32_t > &v3526 /* v3526[16] */,
  hls::stream< int8_t > &v3527 /* v3527[16] */,
  hls::stream< int32_t > &v3528 /* v3528[16] */,
  int8_t v3529[16][16],
  int v3530,
  int v3531
) {	// L3811
  #pragma HLS stream variable=v3525 depth=17
  #pragma HLS stream variable=v3526 depth=17
  #pragma HLS stream variable=v3527 depth=17
  #pragma HLS stream variable=v3528 depth=17
  #pragma HLS array_partition variable=v3529 complete dim=1
  #pragma HLS array_partition variable=v3529 complete dim=2

  int8_t v3532 = v3529[v3530][v3531];	// L3812
  int8_t w141;	// L3813
  w141 = v3532;	// L3814
  l_step_m141: for (int m141 = 0; m141 < 16; m141++) {	// L3815
  #pragma HLS pipeline II=1
    int8_t v3535 = v3525.read(); // v3525[m141];	// L3816
    int8_t a141;	// L3817
    a141 = v3535;	// L3818
    int32_t v3537 = v3526.read(); // v3526[m141];	// L3819
    int32_t p141;	// L3820
    p141 = v3537;	// L3821
    int32_t v3539 = p141;	// L3822
    int8_t v3540 = a141;	// L3823
    int8_t v3541 = w141;	// L3824
    int16_t v3542 = v3540;	// L3825
    int16_t v3543 = v3541;	// L3826
    int16_t v3544 = v3542 * v3543;	// L3827
    ap_int<33> v3545 = v3539;	// L3828
    ap_int<33> v3546 = v3544;	// L3829
    ap_int<33> v3547 = v3545 + v3546;	// L3830
    int32_t v3548 = v3547;	// L3831
    v3528.write(v3548); // v3528[m141] = v3548;	// L3832
    int8_t v3549 = a141;	// L3833
    v3527.write(v3549); // v3527[m141] = v3549;	// L3834
  }
}

void PE_kernel_ws_gemm_14_8(
  hls::stream< int8_t > &v3550 /* v3550[16] */,
  hls::stream< int32_t > &v3551 /* v3551[16] */,
  hls::stream< int8_t > &v3552 /* v3552[16] */,
  hls::stream< int32_t > &v3553 /* v3553[16] */,
  int8_t v3554[16][16],
  int v3555,
  int v3556
) {	// L3838
  #pragma HLS stream variable=v3550 depth=17
  #pragma HLS stream variable=v3551 depth=17
  #pragma HLS stream variable=v3552 depth=17
  #pragma HLS stream variable=v3553 depth=17
  #pragma HLS array_partition variable=v3554 complete dim=1
  #pragma HLS array_partition variable=v3554 complete dim=2

  int8_t v3557 = v3554[v3555][v3556];	// L3839
  int8_t w142;	// L3840
  w142 = v3557;	// L3841
  l_step_m142: for (int m142 = 0; m142 < 16; m142++) {	// L3842
  #pragma HLS pipeline II=1
    int8_t v3560 = v3550.read(); // v3550[m142];	// L3843
    int8_t a142;	// L3844
    a142 = v3560;	// L3845
    int32_t v3562 = v3551.read(); // v3551[m142];	// L3846
    int32_t p142;	// L3847
    p142 = v3562;	// L3848
    int32_t v3564 = p142;	// L3849
    int8_t v3565 = a142;	// L3850
    int8_t v3566 = w142;	// L3851
    int16_t v3567 = v3565;	// L3852
    int16_t v3568 = v3566;	// L3853
    int16_t v3569 = v3567 * v3568;	// L3854
    ap_int<33> v3570 = v3564;	// L3855
    ap_int<33> v3571 = v3569;	// L3856
    ap_int<33> v3572 = v3570 + v3571;	// L3857
    int32_t v3573 = v3572;	// L3858
    v3553.write(v3573); // v3553[m142] = v3573;	// L3859
    int8_t v3574 = a142;	// L3860
    v3552.write(v3574); // v3552[m142] = v3574;	// L3861
  }
}

void PE_kernel_ws_gemm_15_8(
  hls::stream< int8_t > &v3575 /* v3575[16] */,
  hls::stream< int32_t > &v3576 /* v3576[16] */,
  hls::stream< int8_t > &v3577 /* v3577[16] */,
  hls::stream< int32_t > &v3578 /* v3578[16] */,
  int8_t v3579[16][16],
  int v3580,
  int v3581
) {	// L3865
  #pragma HLS stream variable=v3575 depth=17
  #pragma HLS stream variable=v3576 depth=17
  #pragma HLS stream variable=v3577 depth=17
  #pragma HLS stream variable=v3578 depth=17
  #pragma HLS array_partition variable=v3579 complete dim=1
  #pragma HLS array_partition variable=v3579 complete dim=2

  int8_t v3582 = v3579[v3580][v3581];	// L3866
  int8_t w143;	// L3867
  w143 = v3582;	// L3868
  l_step_m143: for (int m143 = 0; m143 < 16; m143++) {	// L3869
  #pragma HLS pipeline II=1
    int8_t v3585 = v3575.read(); // v3575[m143];	// L3870
    int8_t a143;	// L3871
    a143 = v3585;	// L3872
    int32_t v3587 = v3576.read(); // v3576[m143];	// L3873
    int32_t p143;	// L3874
    p143 = v3587;	// L3875
    int32_t v3589 = p143;	// L3876
    int8_t v3590 = a143;	// L3877
    int8_t v3591 = w143;	// L3878
    int16_t v3592 = v3590;	// L3879
    int16_t v3593 = v3591;	// L3880
    int16_t v3594 = v3592 * v3593;	// L3881
    ap_int<33> v3595 = v3589;	// L3882
    ap_int<33> v3596 = v3594;	// L3883
    ap_int<33> v3597 = v3595 + v3596;	// L3884
    int32_t v3598 = v3597;	// L3885
    v3578.write(v3598); // v3578[m143] = v3598;	// L3886
    int8_t v3599 = a143;	// L3887
    v3577.write(v3599); // v3577[m143] = v3599;	// L3888
  }
}

void PE_kernel_ws_gemm_0_9(
  hls::stream< int8_t > &v3600 /* v3600[16] */,
  hls::stream< int32_t > &v3601 /* v3601[16] */,
  hls::stream< int8_t > &v3602 /* v3602[16] */,
  hls::stream< int32_t > &v3603 /* v3603[16] */,
  int8_t v3604[16][16],
  int v3605,
  int v3606
) {	// L3892
  #pragma HLS stream variable=v3600 depth=17
  #pragma HLS stream variable=v3601 depth=17
  #pragma HLS stream variable=v3602 depth=17
  #pragma HLS stream variable=v3603 depth=17
  #pragma HLS array_partition variable=v3604 complete dim=1
  #pragma HLS array_partition variable=v3604 complete dim=2

  int8_t v3607 = v3604[v3605][v3606];	// L3893
  int8_t w144;	// L3894
  w144 = v3607;	// L3895
  l_step_m144: for (int m144 = 0; m144 < 16; m144++) {	// L3896
  #pragma HLS pipeline II=1
    int8_t v3610 = v3600.read(); // v3600[m144];	// L3897
    int8_t a144;	// L3898
    a144 = v3610;	// L3899
    int32_t v3612 = v3601.read(); // v3601[m144];	// L3900
    int32_t p144;	// L3901
    p144 = v3612;	// L3902
    int32_t v3614 = p144;	// L3903
    int8_t v3615 = a144;	// L3904
    int8_t v3616 = w144;	// L3905
    int16_t v3617 = v3615;	// L3906
    int16_t v3618 = v3616;	// L3907
    int16_t v3619 = v3617 * v3618;	// L3908
    ap_int<33> v3620 = v3614;	// L3909
    ap_int<33> v3621 = v3619;	// L3910
    ap_int<33> v3622 = v3620 + v3621;	// L3911
    int32_t v3623 = v3622;	// L3912
    v3603.write(v3623); // v3603[m144] = v3623;	// L3913
    int8_t v3624 = a144;	// L3914
    v3602.write(v3624); // v3602[m144] = v3624;	// L3915
  }
}

void PE_kernel_ws_gemm_1_9(
  hls::stream< int8_t > &v3625 /* v3625[16] */,
  hls::stream< int32_t > &v3626 /* v3626[16] */,
  hls::stream< int8_t > &v3627 /* v3627[16] */,
  hls::stream< int32_t > &v3628 /* v3628[16] */,
  int8_t v3629[16][16],
  int v3630,
  int v3631
) {	// L3919
  #pragma HLS stream variable=v3625 depth=17
  #pragma HLS stream variable=v3626 depth=17
  #pragma HLS stream variable=v3627 depth=17
  #pragma HLS stream variable=v3628 depth=17
  #pragma HLS array_partition variable=v3629 complete dim=1
  #pragma HLS array_partition variable=v3629 complete dim=2

  int8_t v3632 = v3629[v3630][v3631];	// L3920
  int8_t w145;	// L3921
  w145 = v3632;	// L3922
  l_step_m145: for (int m145 = 0; m145 < 16; m145++) {	// L3923
  #pragma HLS pipeline II=1
    int8_t v3635 = v3625.read(); // v3625[m145];	// L3924
    int8_t a145;	// L3925
    a145 = v3635;	// L3926
    int32_t v3637 = v3626.read(); // v3626[m145];	// L3927
    int32_t p145;	// L3928
    p145 = v3637;	// L3929
    int32_t v3639 = p145;	// L3930
    int8_t v3640 = a145;	// L3931
    int8_t v3641 = w145;	// L3932
    int16_t v3642 = v3640;	// L3933
    int16_t v3643 = v3641;	// L3934
    int16_t v3644 = v3642 * v3643;	// L3935
    ap_int<33> v3645 = v3639;	// L3936
    ap_int<33> v3646 = v3644;	// L3937
    ap_int<33> v3647 = v3645 + v3646;	// L3938
    int32_t v3648 = v3647;	// L3939
    v3628.write(v3648); // v3628[m145] = v3648;	// L3940
    int8_t v3649 = a145;	// L3941
    v3627.write(v3649); // v3627[m145] = v3649;	// L3942
  }
}

void PE_kernel_ws_gemm_2_9(
  hls::stream< int8_t > &v3650 /* v3650[16] */,
  hls::stream< int32_t > &v3651 /* v3651[16] */,
  hls::stream< int8_t > &v3652 /* v3652[16] */,
  hls::stream< int32_t > &v3653 /* v3653[16] */,
  int8_t v3654[16][16],
  int v3655,
  int v3656
) {	// L3946
  #pragma HLS stream variable=v3650 depth=17
  #pragma HLS stream variable=v3651 depth=17
  #pragma HLS stream variable=v3652 depth=17
  #pragma HLS stream variable=v3653 depth=17
  #pragma HLS array_partition variable=v3654 complete dim=1
  #pragma HLS array_partition variable=v3654 complete dim=2

  int8_t v3657 = v3654[v3655][v3656];	// L3947
  int8_t w146;	// L3948
  w146 = v3657;	// L3949
  l_step_m146: for (int m146 = 0; m146 < 16; m146++) {	// L3950
  #pragma HLS pipeline II=1
    int8_t v3660 = v3650.read(); // v3650[m146];	// L3951
    int8_t a146;	// L3952
    a146 = v3660;	// L3953
    int32_t v3662 = v3651.read(); // v3651[m146];	// L3954
    int32_t p146;	// L3955
    p146 = v3662;	// L3956
    int32_t v3664 = p146;	// L3957
    int8_t v3665 = a146;	// L3958
    int8_t v3666 = w146;	// L3959
    int16_t v3667 = v3665;	// L3960
    int16_t v3668 = v3666;	// L3961
    int16_t v3669 = v3667 * v3668;	// L3962
    ap_int<33> v3670 = v3664;	// L3963
    ap_int<33> v3671 = v3669;	// L3964
    ap_int<33> v3672 = v3670 + v3671;	// L3965
    int32_t v3673 = v3672;	// L3966
    v3653.write(v3673); // v3653[m146] = v3673;	// L3967
    int8_t v3674 = a146;	// L3968
    v3652.write(v3674); // v3652[m146] = v3674;	// L3969
  }
}

void PE_kernel_ws_gemm_3_9(
  hls::stream< int8_t > &v3675 /* v3675[16] */,
  hls::stream< int32_t > &v3676 /* v3676[16] */,
  hls::stream< int8_t > &v3677 /* v3677[16] */,
  hls::stream< int32_t > &v3678 /* v3678[16] */,
  int8_t v3679[16][16],
  int v3680,
  int v3681
) {	// L3973
  #pragma HLS stream variable=v3675 depth=17
  #pragma HLS stream variable=v3676 depth=17
  #pragma HLS stream variable=v3677 depth=17
  #pragma HLS stream variable=v3678 depth=17
  #pragma HLS array_partition variable=v3679 complete dim=1
  #pragma HLS array_partition variable=v3679 complete dim=2

  int8_t v3682 = v3679[v3680][v3681];	// L3974
  int8_t w147;	// L3975
  w147 = v3682;	// L3976
  l_step_m147: for (int m147 = 0; m147 < 16; m147++) {	// L3977
  #pragma HLS pipeline II=1
    int8_t v3685 = v3675.read(); // v3675[m147];	// L3978
    int8_t a147;	// L3979
    a147 = v3685;	// L3980
    int32_t v3687 = v3676.read(); // v3676[m147];	// L3981
    int32_t p147;	// L3982
    p147 = v3687;	// L3983
    int32_t v3689 = p147;	// L3984
    int8_t v3690 = a147;	// L3985
    int8_t v3691 = w147;	// L3986
    int16_t v3692 = v3690;	// L3987
    int16_t v3693 = v3691;	// L3988
    int16_t v3694 = v3692 * v3693;	// L3989
    ap_int<33> v3695 = v3689;	// L3990
    ap_int<33> v3696 = v3694;	// L3991
    ap_int<33> v3697 = v3695 + v3696;	// L3992
    int32_t v3698 = v3697;	// L3993
    v3678.write(v3698); // v3678[m147] = v3698;	// L3994
    int8_t v3699 = a147;	// L3995
    v3677.write(v3699); // v3677[m147] = v3699;	// L3996
  }
}

void PE_kernel_ws_gemm_4_9(
  hls::stream< int8_t > &v3700 /* v3700[16] */,
  hls::stream< int32_t > &v3701 /* v3701[16] */,
  hls::stream< int8_t > &v3702 /* v3702[16] */,
  hls::stream< int32_t > &v3703 /* v3703[16] */,
  int8_t v3704[16][16],
  int v3705,
  int v3706
) {	// L4000
  #pragma HLS stream variable=v3700 depth=17
  #pragma HLS stream variable=v3701 depth=17
  #pragma HLS stream variable=v3702 depth=17
  #pragma HLS stream variable=v3703 depth=17
  #pragma HLS array_partition variable=v3704 complete dim=1
  #pragma HLS array_partition variable=v3704 complete dim=2

  int8_t v3707 = v3704[v3705][v3706];	// L4001
  int8_t w148;	// L4002
  w148 = v3707;	// L4003
  l_step_m148: for (int m148 = 0; m148 < 16; m148++) {	// L4004
  #pragma HLS pipeline II=1
    int8_t v3710 = v3700.read(); // v3700[m148];	// L4005
    int8_t a148;	// L4006
    a148 = v3710;	// L4007
    int32_t v3712 = v3701.read(); // v3701[m148];	// L4008
    int32_t p148;	// L4009
    p148 = v3712;	// L4010
    int32_t v3714 = p148;	// L4011
    int8_t v3715 = a148;	// L4012
    int8_t v3716 = w148;	// L4013
    int16_t v3717 = v3715;	// L4014
    int16_t v3718 = v3716;	// L4015
    int16_t v3719 = v3717 * v3718;	// L4016
    ap_int<33> v3720 = v3714;	// L4017
    ap_int<33> v3721 = v3719;	// L4018
    ap_int<33> v3722 = v3720 + v3721;	// L4019
    int32_t v3723 = v3722;	// L4020
    v3703.write(v3723); // v3703[m148] = v3723;	// L4021
    int8_t v3724 = a148;	// L4022
    v3702.write(v3724); // v3702[m148] = v3724;	// L4023
  }
}

void PE_kernel_ws_gemm_5_9(
  hls::stream< int8_t > &v3725 /* v3725[16] */,
  hls::stream< int32_t > &v3726 /* v3726[16] */,
  hls::stream< int8_t > &v3727 /* v3727[16] */,
  hls::stream< int32_t > &v3728 /* v3728[16] */,
  int8_t v3729[16][16],
  int v3730,
  int v3731
) {	// L4027
  #pragma HLS stream variable=v3725 depth=17
  #pragma HLS stream variable=v3726 depth=17
  #pragma HLS stream variable=v3727 depth=17
  #pragma HLS stream variable=v3728 depth=17
  #pragma HLS array_partition variable=v3729 complete dim=1
  #pragma HLS array_partition variable=v3729 complete dim=2

  int8_t v3732 = v3729[v3730][v3731];	// L4028
  int8_t w149;	// L4029
  w149 = v3732;	// L4030
  l_step_m149: for (int m149 = 0; m149 < 16; m149++) {	// L4031
  #pragma HLS pipeline II=1
    int8_t v3735 = v3725.read(); // v3725[m149];	// L4032
    int8_t a149;	// L4033
    a149 = v3735;	// L4034
    int32_t v3737 = v3726.read(); // v3726[m149];	// L4035
    int32_t p149;	// L4036
    p149 = v3737;	// L4037
    int32_t v3739 = p149;	// L4038
    int8_t v3740 = a149;	// L4039
    int8_t v3741 = w149;	// L4040
    int16_t v3742 = v3740;	// L4041
    int16_t v3743 = v3741;	// L4042
    int16_t v3744 = v3742 * v3743;	// L4043
    ap_int<33> v3745 = v3739;	// L4044
    ap_int<33> v3746 = v3744;	// L4045
    ap_int<33> v3747 = v3745 + v3746;	// L4046
    int32_t v3748 = v3747;	// L4047
    v3728.write(v3748); // v3728[m149] = v3748;	// L4048
    int8_t v3749 = a149;	// L4049
    v3727.write(v3749); // v3727[m149] = v3749;	// L4050
  }
}

void PE_kernel_ws_gemm_6_9(
  hls::stream< int8_t > &v3750 /* v3750[16] */,
  hls::stream< int32_t > &v3751 /* v3751[16] */,
  hls::stream< int8_t > &v3752 /* v3752[16] */,
  hls::stream< int32_t > &v3753 /* v3753[16] */,
  int8_t v3754[16][16],
  int v3755,
  int v3756
) {	// L4054
  #pragma HLS stream variable=v3750 depth=17
  #pragma HLS stream variable=v3751 depth=17
  #pragma HLS stream variable=v3752 depth=17
  #pragma HLS stream variable=v3753 depth=17
  #pragma HLS array_partition variable=v3754 complete dim=1
  #pragma HLS array_partition variable=v3754 complete dim=2

  int8_t v3757 = v3754[v3755][v3756];	// L4055
  int8_t w150;	// L4056
  w150 = v3757;	// L4057
  l_step_m150: for (int m150 = 0; m150 < 16; m150++) {	// L4058
  #pragma HLS pipeline II=1
    int8_t v3760 = v3750.read(); // v3750[m150];	// L4059
    int8_t a150;	// L4060
    a150 = v3760;	// L4061
    int32_t v3762 = v3751.read(); // v3751[m150];	// L4062
    int32_t p150;	// L4063
    p150 = v3762;	// L4064
    int32_t v3764 = p150;	// L4065
    int8_t v3765 = a150;	// L4066
    int8_t v3766 = w150;	// L4067
    int16_t v3767 = v3765;	// L4068
    int16_t v3768 = v3766;	// L4069
    int16_t v3769 = v3767 * v3768;	// L4070
    ap_int<33> v3770 = v3764;	// L4071
    ap_int<33> v3771 = v3769;	// L4072
    ap_int<33> v3772 = v3770 + v3771;	// L4073
    int32_t v3773 = v3772;	// L4074
    v3753.write(v3773); // v3753[m150] = v3773;	// L4075
    int8_t v3774 = a150;	// L4076
    v3752.write(v3774); // v3752[m150] = v3774;	// L4077
  }
}

void PE_kernel_ws_gemm_7_9(
  hls::stream< int8_t > &v3775 /* v3775[16] */,
  hls::stream< int32_t > &v3776 /* v3776[16] */,
  hls::stream< int8_t > &v3777 /* v3777[16] */,
  hls::stream< int32_t > &v3778 /* v3778[16] */,
  int8_t v3779[16][16],
  int v3780,
  int v3781
) {	// L4081
  #pragma HLS stream variable=v3775 depth=17
  #pragma HLS stream variable=v3776 depth=17
  #pragma HLS stream variable=v3777 depth=17
  #pragma HLS stream variable=v3778 depth=17
  #pragma HLS array_partition variable=v3779 complete dim=1
  #pragma HLS array_partition variable=v3779 complete dim=2

  int8_t v3782 = v3779[v3780][v3781];	// L4082
  int8_t w151;	// L4083
  w151 = v3782;	// L4084
  l_step_m151: for (int m151 = 0; m151 < 16; m151++) {	// L4085
  #pragma HLS pipeline II=1
    int8_t v3785 = v3775.read(); // v3775[m151];	// L4086
    int8_t a151;	// L4087
    a151 = v3785;	// L4088
    int32_t v3787 = v3776.read(); // v3776[m151];	// L4089
    int32_t p151;	// L4090
    p151 = v3787;	// L4091
    int32_t v3789 = p151;	// L4092
    int8_t v3790 = a151;	// L4093
    int8_t v3791 = w151;	// L4094
    int16_t v3792 = v3790;	// L4095
    int16_t v3793 = v3791;	// L4096
    int16_t v3794 = v3792 * v3793;	// L4097
    ap_int<33> v3795 = v3789;	// L4098
    ap_int<33> v3796 = v3794;	// L4099
    ap_int<33> v3797 = v3795 + v3796;	// L4100
    int32_t v3798 = v3797;	// L4101
    v3778.write(v3798); // v3778[m151] = v3798;	// L4102
    int8_t v3799 = a151;	// L4103
    v3777.write(v3799); // v3777[m151] = v3799;	// L4104
  }
}

void PE_kernel_ws_gemm_8_9(
  hls::stream< int8_t > &v3800 /* v3800[16] */,
  hls::stream< int32_t > &v3801 /* v3801[16] */,
  hls::stream< int8_t > &v3802 /* v3802[16] */,
  hls::stream< int32_t > &v3803 /* v3803[16] */,
  int8_t v3804[16][16],
  int v3805,
  int v3806
) {	// L4108
  #pragma HLS stream variable=v3800 depth=17
  #pragma HLS stream variable=v3801 depth=17
  #pragma HLS stream variable=v3802 depth=17
  #pragma HLS stream variable=v3803 depth=17
  #pragma HLS array_partition variable=v3804 complete dim=1
  #pragma HLS array_partition variable=v3804 complete dim=2

  int8_t v3807 = v3804[v3805][v3806];	// L4109
  int8_t w152;	// L4110
  w152 = v3807;	// L4111
  l_step_m152: for (int m152 = 0; m152 < 16; m152++) {	// L4112
  #pragma HLS pipeline II=1
    int8_t v3810 = v3800.read(); // v3800[m152];	// L4113
    int8_t a152;	// L4114
    a152 = v3810;	// L4115
    int32_t v3812 = v3801.read(); // v3801[m152];	// L4116
    int32_t p152;	// L4117
    p152 = v3812;	// L4118
    int32_t v3814 = p152;	// L4119
    int8_t v3815 = a152;	// L4120
    int8_t v3816 = w152;	// L4121
    int16_t v3817 = v3815;	// L4122
    int16_t v3818 = v3816;	// L4123
    int16_t v3819 = v3817 * v3818;	// L4124
    ap_int<33> v3820 = v3814;	// L4125
    ap_int<33> v3821 = v3819;	// L4126
    ap_int<33> v3822 = v3820 + v3821;	// L4127
    int32_t v3823 = v3822;	// L4128
    v3803.write(v3823); // v3803[m152] = v3823;	// L4129
    int8_t v3824 = a152;	// L4130
    v3802.write(v3824); // v3802[m152] = v3824;	// L4131
  }
}

void PE_kernel_ws_gemm_9_9(
  hls::stream< int8_t > &v3825 /* v3825[16] */,
  hls::stream< int32_t > &v3826 /* v3826[16] */,
  hls::stream< int8_t > &v3827 /* v3827[16] */,
  hls::stream< int32_t > &v3828 /* v3828[16] */,
  int8_t v3829[16][16],
  int v3830,
  int v3831
) {	// L4135
  #pragma HLS stream variable=v3825 depth=17
  #pragma HLS stream variable=v3826 depth=17
  #pragma HLS stream variable=v3827 depth=17
  #pragma HLS stream variable=v3828 depth=17
  #pragma HLS array_partition variable=v3829 complete dim=1
  #pragma HLS array_partition variable=v3829 complete dim=2

  int8_t v3832 = v3829[v3830][v3831];	// L4136
  int8_t w153;	// L4137
  w153 = v3832;	// L4138
  l_step_m153: for (int m153 = 0; m153 < 16; m153++) {	// L4139
  #pragma HLS pipeline II=1
    int8_t v3835 = v3825.read(); // v3825[m153];	// L4140
    int8_t a153;	// L4141
    a153 = v3835;	// L4142
    int32_t v3837 = v3826.read(); // v3826[m153];	// L4143
    int32_t p153;	// L4144
    p153 = v3837;	// L4145
    int32_t v3839 = p153;	// L4146
    int8_t v3840 = a153;	// L4147
    int8_t v3841 = w153;	// L4148
    int16_t v3842 = v3840;	// L4149
    int16_t v3843 = v3841;	// L4150
    int16_t v3844 = v3842 * v3843;	// L4151
    ap_int<33> v3845 = v3839;	// L4152
    ap_int<33> v3846 = v3844;	// L4153
    ap_int<33> v3847 = v3845 + v3846;	// L4154
    int32_t v3848 = v3847;	// L4155
    v3828.write(v3848); // v3828[m153] = v3848;	// L4156
    int8_t v3849 = a153;	// L4157
    v3827.write(v3849); // v3827[m153] = v3849;	// L4158
  }
}

void PE_kernel_ws_gemm_10_9(
  hls::stream< int8_t > &v3850 /* v3850[16] */,
  hls::stream< int32_t > &v3851 /* v3851[16] */,
  hls::stream< int8_t > &v3852 /* v3852[16] */,
  hls::stream< int32_t > &v3853 /* v3853[16] */,
  int8_t v3854[16][16],
  int v3855,
  int v3856
) {	// L4162
  #pragma HLS stream variable=v3850 depth=17
  #pragma HLS stream variable=v3851 depth=17
  #pragma HLS stream variable=v3852 depth=17
  #pragma HLS stream variable=v3853 depth=17
  #pragma HLS array_partition variable=v3854 complete dim=1
  #pragma HLS array_partition variable=v3854 complete dim=2

  int8_t v3857 = v3854[v3855][v3856];	// L4163
  int8_t w154;	// L4164
  w154 = v3857;	// L4165
  l_step_m154: for (int m154 = 0; m154 < 16; m154++) {	// L4166
  #pragma HLS pipeline II=1
    int8_t v3860 = v3850.read(); // v3850[m154];	// L4167
    int8_t a154;	// L4168
    a154 = v3860;	// L4169
    int32_t v3862 = v3851.read(); // v3851[m154];	// L4170
    int32_t p154;	// L4171
    p154 = v3862;	// L4172
    int32_t v3864 = p154;	// L4173
    int8_t v3865 = a154;	// L4174
    int8_t v3866 = w154;	// L4175
    int16_t v3867 = v3865;	// L4176
    int16_t v3868 = v3866;	// L4177
    int16_t v3869 = v3867 * v3868;	// L4178
    ap_int<33> v3870 = v3864;	// L4179
    ap_int<33> v3871 = v3869;	// L4180
    ap_int<33> v3872 = v3870 + v3871;	// L4181
    int32_t v3873 = v3872;	// L4182
    v3853.write(v3873); // v3853[m154] = v3873;	// L4183
    int8_t v3874 = a154;	// L4184
    v3852.write(v3874); // v3852[m154] = v3874;	// L4185
  }
}

void PE_kernel_ws_gemm_11_9(
  hls::stream< int8_t > &v3875 /* v3875[16] */,
  hls::stream< int32_t > &v3876 /* v3876[16] */,
  hls::stream< int8_t > &v3877 /* v3877[16] */,
  hls::stream< int32_t > &v3878 /* v3878[16] */,
  int8_t v3879[16][16],
  int v3880,
  int v3881
) {	// L4189
  #pragma HLS stream variable=v3875 depth=17
  #pragma HLS stream variable=v3876 depth=17
  #pragma HLS stream variable=v3877 depth=17
  #pragma HLS stream variable=v3878 depth=17
  #pragma HLS array_partition variable=v3879 complete dim=1
  #pragma HLS array_partition variable=v3879 complete dim=2

  int8_t v3882 = v3879[v3880][v3881];	// L4190
  int8_t w155;	// L4191
  w155 = v3882;	// L4192
  l_step_m155: for (int m155 = 0; m155 < 16; m155++) {	// L4193
  #pragma HLS pipeline II=1
    int8_t v3885 = v3875.read(); // v3875[m155];	// L4194
    int8_t a155;	// L4195
    a155 = v3885;	// L4196
    int32_t v3887 = v3876.read(); // v3876[m155];	// L4197
    int32_t p155;	// L4198
    p155 = v3887;	// L4199
    int32_t v3889 = p155;	// L4200
    int8_t v3890 = a155;	// L4201
    int8_t v3891 = w155;	// L4202
    int16_t v3892 = v3890;	// L4203
    int16_t v3893 = v3891;	// L4204
    int16_t v3894 = v3892 * v3893;	// L4205
    ap_int<33> v3895 = v3889;	// L4206
    ap_int<33> v3896 = v3894;	// L4207
    ap_int<33> v3897 = v3895 + v3896;	// L4208
    int32_t v3898 = v3897;	// L4209
    v3878.write(v3898); // v3878[m155] = v3898;	// L4210
    int8_t v3899 = a155;	// L4211
    v3877.write(v3899); // v3877[m155] = v3899;	// L4212
  }
}

void PE_kernel_ws_gemm_12_9(
  hls::stream< int8_t > &v3900 /* v3900[16] */,
  hls::stream< int32_t > &v3901 /* v3901[16] */,
  hls::stream< int8_t > &v3902 /* v3902[16] */,
  hls::stream< int32_t > &v3903 /* v3903[16] */,
  int8_t v3904[16][16],
  int v3905,
  int v3906
) {	// L4216
  #pragma HLS stream variable=v3900 depth=17
  #pragma HLS stream variable=v3901 depth=17
  #pragma HLS stream variable=v3902 depth=17
  #pragma HLS stream variable=v3903 depth=17
  #pragma HLS array_partition variable=v3904 complete dim=1
  #pragma HLS array_partition variable=v3904 complete dim=2

  int8_t v3907 = v3904[v3905][v3906];	// L4217
  int8_t w156;	// L4218
  w156 = v3907;	// L4219
  l_step_m156: for (int m156 = 0; m156 < 16; m156++) {	// L4220
  #pragma HLS pipeline II=1
    int8_t v3910 = v3900.read(); // v3900[m156];	// L4221
    int8_t a156;	// L4222
    a156 = v3910;	// L4223
    int32_t v3912 = v3901.read(); // v3901[m156];	// L4224
    int32_t p156;	// L4225
    p156 = v3912;	// L4226
    int32_t v3914 = p156;	// L4227
    int8_t v3915 = a156;	// L4228
    int8_t v3916 = w156;	// L4229
    int16_t v3917 = v3915;	// L4230
    int16_t v3918 = v3916;	// L4231
    int16_t v3919 = v3917 * v3918;	// L4232
    ap_int<33> v3920 = v3914;	// L4233
    ap_int<33> v3921 = v3919;	// L4234
    ap_int<33> v3922 = v3920 + v3921;	// L4235
    int32_t v3923 = v3922;	// L4236
    v3903.write(v3923); // v3903[m156] = v3923;	// L4237
    int8_t v3924 = a156;	// L4238
    v3902.write(v3924); // v3902[m156] = v3924;	// L4239
  }
}

void PE_kernel_ws_gemm_13_9(
  hls::stream< int8_t > &v3925 /* v3925[16] */,
  hls::stream< int32_t > &v3926 /* v3926[16] */,
  hls::stream< int8_t > &v3927 /* v3927[16] */,
  hls::stream< int32_t > &v3928 /* v3928[16] */,
  int8_t v3929[16][16],
  int v3930,
  int v3931
) {	// L4243
  #pragma HLS stream variable=v3925 depth=17
  #pragma HLS stream variable=v3926 depth=17
  #pragma HLS stream variable=v3927 depth=17
  #pragma HLS stream variable=v3928 depth=17
  #pragma HLS array_partition variable=v3929 complete dim=1
  #pragma HLS array_partition variable=v3929 complete dim=2

  int8_t v3932 = v3929[v3930][v3931];	// L4244
  int8_t w157;	// L4245
  w157 = v3932;	// L4246
  l_step_m157: for (int m157 = 0; m157 < 16; m157++) {	// L4247
  #pragma HLS pipeline II=1
    int8_t v3935 = v3925.read(); // v3925[m157];	// L4248
    int8_t a157;	// L4249
    a157 = v3935;	// L4250
    int32_t v3937 = v3926.read(); // v3926[m157];	// L4251
    int32_t p157;	// L4252
    p157 = v3937;	// L4253
    int32_t v3939 = p157;	// L4254
    int8_t v3940 = a157;	// L4255
    int8_t v3941 = w157;	// L4256
    int16_t v3942 = v3940;	// L4257
    int16_t v3943 = v3941;	// L4258
    int16_t v3944 = v3942 * v3943;	// L4259
    ap_int<33> v3945 = v3939;	// L4260
    ap_int<33> v3946 = v3944;	// L4261
    ap_int<33> v3947 = v3945 + v3946;	// L4262
    int32_t v3948 = v3947;	// L4263
    v3928.write(v3948); // v3928[m157] = v3948;	// L4264
    int8_t v3949 = a157;	// L4265
    v3927.write(v3949); // v3927[m157] = v3949;	// L4266
  }
}

void PE_kernel_ws_gemm_14_9(
  hls::stream< int8_t > &v3950 /* v3950[16] */,
  hls::stream< int32_t > &v3951 /* v3951[16] */,
  hls::stream< int8_t > &v3952 /* v3952[16] */,
  hls::stream< int32_t > &v3953 /* v3953[16] */,
  int8_t v3954[16][16],
  int v3955,
  int v3956
) {	// L4270
  #pragma HLS stream variable=v3950 depth=17
  #pragma HLS stream variable=v3951 depth=17
  #pragma HLS stream variable=v3952 depth=17
  #pragma HLS stream variable=v3953 depth=17
  #pragma HLS array_partition variable=v3954 complete dim=1
  #pragma HLS array_partition variable=v3954 complete dim=2

  int8_t v3957 = v3954[v3955][v3956];	// L4271
  int8_t w158;	// L4272
  w158 = v3957;	// L4273
  l_step_m158: for (int m158 = 0; m158 < 16; m158++) {	// L4274
  #pragma HLS pipeline II=1
    int8_t v3960 = v3950.read(); // v3950[m158];	// L4275
    int8_t a158;	// L4276
    a158 = v3960;	// L4277
    int32_t v3962 = v3951.read(); // v3951[m158];	// L4278
    int32_t p158;	// L4279
    p158 = v3962;	// L4280
    int32_t v3964 = p158;	// L4281
    int8_t v3965 = a158;	// L4282
    int8_t v3966 = w158;	// L4283
    int16_t v3967 = v3965;	// L4284
    int16_t v3968 = v3966;	// L4285
    int16_t v3969 = v3967 * v3968;	// L4286
    ap_int<33> v3970 = v3964;	// L4287
    ap_int<33> v3971 = v3969;	// L4288
    ap_int<33> v3972 = v3970 + v3971;	// L4289
    int32_t v3973 = v3972;	// L4290
    v3953.write(v3973); // v3953[m158] = v3973;	// L4291
    int8_t v3974 = a158;	// L4292
    v3952.write(v3974); // v3952[m158] = v3974;	// L4293
  }
}

void PE_kernel_ws_gemm_15_9(
  hls::stream< int8_t > &v3975 /* v3975[16] */,
  hls::stream< int32_t > &v3976 /* v3976[16] */,
  hls::stream< int8_t > &v3977 /* v3977[16] */,
  hls::stream< int32_t > &v3978 /* v3978[16] */,
  int8_t v3979[16][16],
  int v3980,
  int v3981
) {	// L4297
  #pragma HLS stream variable=v3975 depth=17
  #pragma HLS stream variable=v3976 depth=17
  #pragma HLS stream variable=v3977 depth=17
  #pragma HLS stream variable=v3978 depth=17
  #pragma HLS array_partition variable=v3979 complete dim=1
  #pragma HLS array_partition variable=v3979 complete dim=2

  int8_t v3982 = v3979[v3980][v3981];	// L4298
  int8_t w159;	// L4299
  w159 = v3982;	// L4300
  l_step_m159: for (int m159 = 0; m159 < 16; m159++) {	// L4301
  #pragma HLS pipeline II=1
    int8_t v3985 = v3975.read(); // v3975[m159];	// L4302
    int8_t a159;	// L4303
    a159 = v3985;	// L4304
    int32_t v3987 = v3976.read(); // v3976[m159];	// L4305
    int32_t p159;	// L4306
    p159 = v3987;	// L4307
    int32_t v3989 = p159;	// L4308
    int8_t v3990 = a159;	// L4309
    int8_t v3991 = w159;	// L4310
    int16_t v3992 = v3990;	// L4311
    int16_t v3993 = v3991;	// L4312
    int16_t v3994 = v3992 * v3993;	// L4313
    ap_int<33> v3995 = v3989;	// L4314
    ap_int<33> v3996 = v3994;	// L4315
    ap_int<33> v3997 = v3995 + v3996;	// L4316
    int32_t v3998 = v3997;	// L4317
    v3978.write(v3998); // v3978[m159] = v3998;	// L4318
    int8_t v3999 = a159;	// L4319
    v3977.write(v3999); // v3977[m159] = v3999;	// L4320
  }
}

void PE_kernel_ws_gemm_0_10(
  hls::stream< int8_t > &v4000 /* v4000[16] */,
  hls::stream< int32_t > &v4001 /* v4001[16] */,
  hls::stream< int8_t > &v4002 /* v4002[16] */,
  hls::stream< int32_t > &v4003 /* v4003[16] */,
  int8_t v4004[16][16],
  int v4005,
  int v4006
) {	// L4324
  #pragma HLS stream variable=v4000 depth=17
  #pragma HLS stream variable=v4001 depth=17
  #pragma HLS stream variable=v4002 depth=17
  #pragma HLS stream variable=v4003 depth=17
  #pragma HLS array_partition variable=v4004 complete dim=1
  #pragma HLS array_partition variable=v4004 complete dim=2

  int8_t v4007 = v4004[v4005][v4006];	// L4325
  int8_t w160;	// L4326
  w160 = v4007;	// L4327
  l_step_m160: for (int m160 = 0; m160 < 16; m160++) {	// L4328
  #pragma HLS pipeline II=1
    int8_t v4010 = v4000.read(); // v4000[m160];	// L4329
    int8_t a160;	// L4330
    a160 = v4010;	// L4331
    int32_t v4012 = v4001.read(); // v4001[m160];	// L4332
    int32_t p160;	// L4333
    p160 = v4012;	// L4334
    int32_t v4014 = p160;	// L4335
    int8_t v4015 = a160;	// L4336
    int8_t v4016 = w160;	// L4337
    int16_t v4017 = v4015;	// L4338
    int16_t v4018 = v4016;	// L4339
    int16_t v4019 = v4017 * v4018;	// L4340
    ap_int<33> v4020 = v4014;	// L4341
    ap_int<33> v4021 = v4019;	// L4342
    ap_int<33> v4022 = v4020 + v4021;	// L4343
    int32_t v4023 = v4022;	// L4344
    v4003.write(v4023); // v4003[m160] = v4023;	// L4345
    int8_t v4024 = a160;	// L4346
    v4002.write(v4024); // v4002[m160] = v4024;	// L4347
  }
}

void PE_kernel_ws_gemm_1_10(
  hls::stream< int8_t > &v4025 /* v4025[16] */,
  hls::stream< int32_t > &v4026 /* v4026[16] */,
  hls::stream< int8_t > &v4027 /* v4027[16] */,
  hls::stream< int32_t > &v4028 /* v4028[16] */,
  int8_t v4029[16][16],
  int v4030,
  int v4031
) {	// L4351
  #pragma HLS stream variable=v4025 depth=17
  #pragma HLS stream variable=v4026 depth=17
  #pragma HLS stream variable=v4027 depth=17
  #pragma HLS stream variable=v4028 depth=17
  #pragma HLS array_partition variable=v4029 complete dim=1
  #pragma HLS array_partition variable=v4029 complete dim=2

  int8_t v4032 = v4029[v4030][v4031];	// L4352
  int8_t w161;	// L4353
  w161 = v4032;	// L4354
  l_step_m161: for (int m161 = 0; m161 < 16; m161++) {	// L4355
  #pragma HLS pipeline II=1
    int8_t v4035 = v4025.read(); // v4025[m161];	// L4356
    int8_t a161;	// L4357
    a161 = v4035;	// L4358
    int32_t v4037 = v4026.read(); // v4026[m161];	// L4359
    int32_t p161;	// L4360
    p161 = v4037;	// L4361
    int32_t v4039 = p161;	// L4362
    int8_t v4040 = a161;	// L4363
    int8_t v4041 = w161;	// L4364
    int16_t v4042 = v4040;	// L4365
    int16_t v4043 = v4041;	// L4366
    int16_t v4044 = v4042 * v4043;	// L4367
    ap_int<33> v4045 = v4039;	// L4368
    ap_int<33> v4046 = v4044;	// L4369
    ap_int<33> v4047 = v4045 + v4046;	// L4370
    int32_t v4048 = v4047;	// L4371
    v4028.write(v4048); // v4028[m161] = v4048;	// L4372
    int8_t v4049 = a161;	// L4373
    v4027.write(v4049); // v4027[m161] = v4049;	// L4374
  }
}

void PE_kernel_ws_gemm_2_10(
  hls::stream< int8_t > &v4050 /* v4050[16] */,
  hls::stream< int32_t > &v4051 /* v4051[16] */,
  hls::stream< int8_t > &v4052 /* v4052[16] */,
  hls::stream< int32_t > &v4053 /* v4053[16] */,
  int8_t v4054[16][16],
  int v4055,
  int v4056
) {	// L4378
  #pragma HLS stream variable=v4050 depth=17
  #pragma HLS stream variable=v4051 depth=17
  #pragma HLS stream variable=v4052 depth=17
  #pragma HLS stream variable=v4053 depth=17
  #pragma HLS array_partition variable=v4054 complete dim=1
  #pragma HLS array_partition variable=v4054 complete dim=2

  int8_t v4057 = v4054[v4055][v4056];	// L4379
  int8_t w162;	// L4380
  w162 = v4057;	// L4381
  l_step_m162: for (int m162 = 0; m162 < 16; m162++) {	// L4382
  #pragma HLS pipeline II=1
    int8_t v4060 = v4050.read(); // v4050[m162];	// L4383
    int8_t a162;	// L4384
    a162 = v4060;	// L4385
    int32_t v4062 = v4051.read(); // v4051[m162];	// L4386
    int32_t p162;	// L4387
    p162 = v4062;	// L4388
    int32_t v4064 = p162;	// L4389
    int8_t v4065 = a162;	// L4390
    int8_t v4066 = w162;	// L4391
    int16_t v4067 = v4065;	// L4392
    int16_t v4068 = v4066;	// L4393
    int16_t v4069 = v4067 * v4068;	// L4394
    ap_int<33> v4070 = v4064;	// L4395
    ap_int<33> v4071 = v4069;	// L4396
    ap_int<33> v4072 = v4070 + v4071;	// L4397
    int32_t v4073 = v4072;	// L4398
    v4053.write(v4073); // v4053[m162] = v4073;	// L4399
    int8_t v4074 = a162;	// L4400
    v4052.write(v4074); // v4052[m162] = v4074;	// L4401
  }
}

void PE_kernel_ws_gemm_3_10(
  hls::stream< int8_t > &v4075 /* v4075[16] */,
  hls::stream< int32_t > &v4076 /* v4076[16] */,
  hls::stream< int8_t > &v4077 /* v4077[16] */,
  hls::stream< int32_t > &v4078 /* v4078[16] */,
  int8_t v4079[16][16],
  int v4080,
  int v4081
) {	// L4405
  #pragma HLS stream variable=v4075 depth=17
  #pragma HLS stream variable=v4076 depth=17
  #pragma HLS stream variable=v4077 depth=17
  #pragma HLS stream variable=v4078 depth=17
  #pragma HLS array_partition variable=v4079 complete dim=1
  #pragma HLS array_partition variable=v4079 complete dim=2

  int8_t v4082 = v4079[v4080][v4081];	// L4406
  int8_t w163;	// L4407
  w163 = v4082;	// L4408
  l_step_m163: for (int m163 = 0; m163 < 16; m163++) {	// L4409
  #pragma HLS pipeline II=1
    int8_t v4085 = v4075.read(); // v4075[m163];	// L4410
    int8_t a163;	// L4411
    a163 = v4085;	// L4412
    int32_t v4087 = v4076.read(); // v4076[m163];	// L4413
    int32_t p163;	// L4414
    p163 = v4087;	// L4415
    int32_t v4089 = p163;	// L4416
    int8_t v4090 = a163;	// L4417
    int8_t v4091 = w163;	// L4418
    int16_t v4092 = v4090;	// L4419
    int16_t v4093 = v4091;	// L4420
    int16_t v4094 = v4092 * v4093;	// L4421
    ap_int<33> v4095 = v4089;	// L4422
    ap_int<33> v4096 = v4094;	// L4423
    ap_int<33> v4097 = v4095 + v4096;	// L4424
    int32_t v4098 = v4097;	// L4425
    v4078.write(v4098); // v4078[m163] = v4098;	// L4426
    int8_t v4099 = a163;	// L4427
    v4077.write(v4099); // v4077[m163] = v4099;	// L4428
  }
}

void PE_kernel_ws_gemm_4_10(
  hls::stream< int8_t > &v4100 /* v4100[16] */,
  hls::stream< int32_t > &v4101 /* v4101[16] */,
  hls::stream< int8_t > &v4102 /* v4102[16] */,
  hls::stream< int32_t > &v4103 /* v4103[16] */,
  int8_t v4104[16][16],
  int v4105,
  int v4106
) {	// L4432
  #pragma HLS stream variable=v4100 depth=17
  #pragma HLS stream variable=v4101 depth=17
  #pragma HLS stream variable=v4102 depth=17
  #pragma HLS stream variable=v4103 depth=17
  #pragma HLS array_partition variable=v4104 complete dim=1
  #pragma HLS array_partition variable=v4104 complete dim=2

  int8_t v4107 = v4104[v4105][v4106];	// L4433
  int8_t w164;	// L4434
  w164 = v4107;	// L4435
  l_step_m164: for (int m164 = 0; m164 < 16; m164++) {	// L4436
  #pragma HLS pipeline II=1
    int8_t v4110 = v4100.read(); // v4100[m164];	// L4437
    int8_t a164;	// L4438
    a164 = v4110;	// L4439
    int32_t v4112 = v4101.read(); // v4101[m164];	// L4440
    int32_t p164;	// L4441
    p164 = v4112;	// L4442
    int32_t v4114 = p164;	// L4443
    int8_t v4115 = a164;	// L4444
    int8_t v4116 = w164;	// L4445
    int16_t v4117 = v4115;	// L4446
    int16_t v4118 = v4116;	// L4447
    int16_t v4119 = v4117 * v4118;	// L4448
    ap_int<33> v4120 = v4114;	// L4449
    ap_int<33> v4121 = v4119;	// L4450
    ap_int<33> v4122 = v4120 + v4121;	// L4451
    int32_t v4123 = v4122;	// L4452
    v4103.write(v4123); // v4103[m164] = v4123;	// L4453
    int8_t v4124 = a164;	// L4454
    v4102.write(v4124); // v4102[m164] = v4124;	// L4455
  }
}

void PE_kernel_ws_gemm_5_10(
  hls::stream< int8_t > &v4125 /* v4125[16] */,
  hls::stream< int32_t > &v4126 /* v4126[16] */,
  hls::stream< int8_t > &v4127 /* v4127[16] */,
  hls::stream< int32_t > &v4128 /* v4128[16] */,
  int8_t v4129[16][16],
  int v4130,
  int v4131
) {	// L4459
  #pragma HLS stream variable=v4125 depth=17
  #pragma HLS stream variable=v4126 depth=17
  #pragma HLS stream variable=v4127 depth=17
  #pragma HLS stream variable=v4128 depth=17
  #pragma HLS array_partition variable=v4129 complete dim=1
  #pragma HLS array_partition variable=v4129 complete dim=2

  int8_t v4132 = v4129[v4130][v4131];	// L4460
  int8_t w165;	// L4461
  w165 = v4132;	// L4462
  l_step_m165: for (int m165 = 0; m165 < 16; m165++) {	// L4463
  #pragma HLS pipeline II=1
    int8_t v4135 = v4125.read(); // v4125[m165];	// L4464
    int8_t a165;	// L4465
    a165 = v4135;	// L4466
    int32_t v4137 = v4126.read(); // v4126[m165];	// L4467
    int32_t p165;	// L4468
    p165 = v4137;	// L4469
    int32_t v4139 = p165;	// L4470
    int8_t v4140 = a165;	// L4471
    int8_t v4141 = w165;	// L4472
    int16_t v4142 = v4140;	// L4473
    int16_t v4143 = v4141;	// L4474
    int16_t v4144 = v4142 * v4143;	// L4475
    ap_int<33> v4145 = v4139;	// L4476
    ap_int<33> v4146 = v4144;	// L4477
    ap_int<33> v4147 = v4145 + v4146;	// L4478
    int32_t v4148 = v4147;	// L4479
    v4128.write(v4148); // v4128[m165] = v4148;	// L4480
    int8_t v4149 = a165;	// L4481
    v4127.write(v4149); // v4127[m165] = v4149;	// L4482
  }
}

void PE_kernel_ws_gemm_6_10(
  hls::stream< int8_t > &v4150 /* v4150[16] */,
  hls::stream< int32_t > &v4151 /* v4151[16] */,
  hls::stream< int8_t > &v4152 /* v4152[16] */,
  hls::stream< int32_t > &v4153 /* v4153[16] */,
  int8_t v4154[16][16],
  int v4155,
  int v4156
) {	// L4486
  #pragma HLS stream variable=v4150 depth=17
  #pragma HLS stream variable=v4151 depth=17
  #pragma HLS stream variable=v4152 depth=17
  #pragma HLS stream variable=v4153 depth=17
  #pragma HLS array_partition variable=v4154 complete dim=1
  #pragma HLS array_partition variable=v4154 complete dim=2

  int8_t v4157 = v4154[v4155][v4156];	// L4487
  int8_t w166;	// L4488
  w166 = v4157;	// L4489
  l_step_m166: for (int m166 = 0; m166 < 16; m166++) {	// L4490
  #pragma HLS pipeline II=1
    int8_t v4160 = v4150.read(); // v4150[m166];	// L4491
    int8_t a166;	// L4492
    a166 = v4160;	// L4493
    int32_t v4162 = v4151.read(); // v4151[m166];	// L4494
    int32_t p166;	// L4495
    p166 = v4162;	// L4496
    int32_t v4164 = p166;	// L4497
    int8_t v4165 = a166;	// L4498
    int8_t v4166 = w166;	// L4499
    int16_t v4167 = v4165;	// L4500
    int16_t v4168 = v4166;	// L4501
    int16_t v4169 = v4167 * v4168;	// L4502
    ap_int<33> v4170 = v4164;	// L4503
    ap_int<33> v4171 = v4169;	// L4504
    ap_int<33> v4172 = v4170 + v4171;	// L4505
    int32_t v4173 = v4172;	// L4506
    v4153.write(v4173); // v4153[m166] = v4173;	// L4507
    int8_t v4174 = a166;	// L4508
    v4152.write(v4174); // v4152[m166] = v4174;	// L4509
  }
}

void PE_kernel_ws_gemm_7_10(
  hls::stream< int8_t > &v4175 /* v4175[16] */,
  hls::stream< int32_t > &v4176 /* v4176[16] */,
  hls::stream< int8_t > &v4177 /* v4177[16] */,
  hls::stream< int32_t > &v4178 /* v4178[16] */,
  int8_t v4179[16][16],
  int v4180,
  int v4181
) {	// L4513
  #pragma HLS stream variable=v4175 depth=17
  #pragma HLS stream variable=v4176 depth=17
  #pragma HLS stream variable=v4177 depth=17
  #pragma HLS stream variable=v4178 depth=17
  #pragma HLS array_partition variable=v4179 complete dim=1
  #pragma HLS array_partition variable=v4179 complete dim=2

  int8_t v4182 = v4179[v4180][v4181];	// L4514
  int8_t w167;	// L4515
  w167 = v4182;	// L4516
  l_step_m167: for (int m167 = 0; m167 < 16; m167++) {	// L4517
  #pragma HLS pipeline II=1
    int8_t v4185 = v4175.read(); // v4175[m167];	// L4518
    int8_t a167;	// L4519
    a167 = v4185;	// L4520
    int32_t v4187 = v4176.read(); // v4176[m167];	// L4521
    int32_t p167;	// L4522
    p167 = v4187;	// L4523
    int32_t v4189 = p167;	// L4524
    int8_t v4190 = a167;	// L4525
    int8_t v4191 = w167;	// L4526
    int16_t v4192 = v4190;	// L4527
    int16_t v4193 = v4191;	// L4528
    int16_t v4194 = v4192 * v4193;	// L4529
    ap_int<33> v4195 = v4189;	// L4530
    ap_int<33> v4196 = v4194;	// L4531
    ap_int<33> v4197 = v4195 + v4196;	// L4532
    int32_t v4198 = v4197;	// L4533
    v4178.write(v4198); // v4178[m167] = v4198;	// L4534
    int8_t v4199 = a167;	// L4535
    v4177.write(v4199); // v4177[m167] = v4199;	// L4536
  }
}

void PE_kernel_ws_gemm_8_10(
  hls::stream< int8_t > &v4200 /* v4200[16] */,
  hls::stream< int32_t > &v4201 /* v4201[16] */,
  hls::stream< int8_t > &v4202 /* v4202[16] */,
  hls::stream< int32_t > &v4203 /* v4203[16] */,
  int8_t v4204[16][16],
  int v4205,
  int v4206
) {	// L4540
  #pragma HLS stream variable=v4200 depth=17
  #pragma HLS stream variable=v4201 depth=17
  #pragma HLS stream variable=v4202 depth=17
  #pragma HLS stream variable=v4203 depth=17
  #pragma HLS array_partition variable=v4204 complete dim=1
  #pragma HLS array_partition variable=v4204 complete dim=2

  int8_t v4207 = v4204[v4205][v4206];	// L4541
  int8_t w168;	// L4542
  w168 = v4207;	// L4543
  l_step_m168: for (int m168 = 0; m168 < 16; m168++) {	// L4544
  #pragma HLS pipeline II=1
    int8_t v4210 = v4200.read(); // v4200[m168];	// L4545
    int8_t a168;	// L4546
    a168 = v4210;	// L4547
    int32_t v4212 = v4201.read(); // v4201[m168];	// L4548
    int32_t p168;	// L4549
    p168 = v4212;	// L4550
    int32_t v4214 = p168;	// L4551
    int8_t v4215 = a168;	// L4552
    int8_t v4216 = w168;	// L4553
    int16_t v4217 = v4215;	// L4554
    int16_t v4218 = v4216;	// L4555
    int16_t v4219 = v4217 * v4218;	// L4556
    ap_int<33> v4220 = v4214;	// L4557
    ap_int<33> v4221 = v4219;	// L4558
    ap_int<33> v4222 = v4220 + v4221;	// L4559
    int32_t v4223 = v4222;	// L4560
    v4203.write(v4223); // v4203[m168] = v4223;	// L4561
    int8_t v4224 = a168;	// L4562
    v4202.write(v4224); // v4202[m168] = v4224;	// L4563
  }
}

void PE_kernel_ws_gemm_9_10(
  hls::stream< int8_t > &v4225 /* v4225[16] */,
  hls::stream< int32_t > &v4226 /* v4226[16] */,
  hls::stream< int8_t > &v4227 /* v4227[16] */,
  hls::stream< int32_t > &v4228 /* v4228[16] */,
  int8_t v4229[16][16],
  int v4230,
  int v4231
) {	// L4567
  #pragma HLS stream variable=v4225 depth=17
  #pragma HLS stream variable=v4226 depth=17
  #pragma HLS stream variable=v4227 depth=17
  #pragma HLS stream variable=v4228 depth=17
  #pragma HLS array_partition variable=v4229 complete dim=1
  #pragma HLS array_partition variable=v4229 complete dim=2

  int8_t v4232 = v4229[v4230][v4231];	// L4568
  int8_t w169;	// L4569
  w169 = v4232;	// L4570
  l_step_m169: for (int m169 = 0; m169 < 16; m169++) {	// L4571
  #pragma HLS pipeline II=1
    int8_t v4235 = v4225.read(); // v4225[m169];	// L4572
    int8_t a169;	// L4573
    a169 = v4235;	// L4574
    int32_t v4237 = v4226.read(); // v4226[m169];	// L4575
    int32_t p169;	// L4576
    p169 = v4237;	// L4577
    int32_t v4239 = p169;	// L4578
    int8_t v4240 = a169;	// L4579
    int8_t v4241 = w169;	// L4580
    int16_t v4242 = v4240;	// L4581
    int16_t v4243 = v4241;	// L4582
    int16_t v4244 = v4242 * v4243;	// L4583
    ap_int<33> v4245 = v4239;	// L4584
    ap_int<33> v4246 = v4244;	// L4585
    ap_int<33> v4247 = v4245 + v4246;	// L4586
    int32_t v4248 = v4247;	// L4587
    v4228.write(v4248); // v4228[m169] = v4248;	// L4588
    int8_t v4249 = a169;	// L4589
    v4227.write(v4249); // v4227[m169] = v4249;	// L4590
  }
}

void PE_kernel_ws_gemm_10_10(
  hls::stream< int8_t > &v4250 /* v4250[16] */,
  hls::stream< int32_t > &v4251 /* v4251[16] */,
  hls::stream< int8_t > &v4252 /* v4252[16] */,
  hls::stream< int32_t > &v4253 /* v4253[16] */,
  int8_t v4254[16][16],
  int v4255,
  int v4256
) {	// L4594
  #pragma HLS stream variable=v4250 depth=17
  #pragma HLS stream variable=v4251 depth=17
  #pragma HLS stream variable=v4252 depth=17
  #pragma HLS stream variable=v4253 depth=17
  #pragma HLS array_partition variable=v4254 complete dim=1
  #pragma HLS array_partition variable=v4254 complete dim=2

  int8_t v4257 = v4254[v4255][v4256];	// L4595
  int8_t w170;	// L4596
  w170 = v4257;	// L4597
  l_step_m170: for (int m170 = 0; m170 < 16; m170++) {	// L4598
  #pragma HLS pipeline II=1
    int8_t v4260 = v4250.read(); // v4250[m170];	// L4599
    int8_t a170;	// L4600
    a170 = v4260;	// L4601
    int32_t v4262 = v4251.read(); // v4251[m170];	// L4602
    int32_t p170;	// L4603
    p170 = v4262;	// L4604
    int32_t v4264 = p170;	// L4605
    int8_t v4265 = a170;	// L4606
    int8_t v4266 = w170;	// L4607
    int16_t v4267 = v4265;	// L4608
    int16_t v4268 = v4266;	// L4609
    int16_t v4269 = v4267 * v4268;	// L4610
    ap_int<33> v4270 = v4264;	// L4611
    ap_int<33> v4271 = v4269;	// L4612
    ap_int<33> v4272 = v4270 + v4271;	// L4613
    int32_t v4273 = v4272;	// L4614
    v4253.write(v4273); // v4253[m170] = v4273;	// L4615
    int8_t v4274 = a170;	// L4616
    v4252.write(v4274); // v4252[m170] = v4274;	// L4617
  }
}

void PE_kernel_ws_gemm_11_10(
  hls::stream< int8_t > &v4275 /* v4275[16] */,
  hls::stream< int32_t > &v4276 /* v4276[16] */,
  hls::stream< int8_t > &v4277 /* v4277[16] */,
  hls::stream< int32_t > &v4278 /* v4278[16] */,
  int8_t v4279[16][16],
  int v4280,
  int v4281
) {	// L4621
  #pragma HLS stream variable=v4275 depth=17
  #pragma HLS stream variable=v4276 depth=17
  #pragma HLS stream variable=v4277 depth=17
  #pragma HLS stream variable=v4278 depth=17
  #pragma HLS array_partition variable=v4279 complete dim=1
  #pragma HLS array_partition variable=v4279 complete dim=2

  int8_t v4282 = v4279[v4280][v4281];	// L4622
  int8_t w171;	// L4623
  w171 = v4282;	// L4624
  l_step_m171: for (int m171 = 0; m171 < 16; m171++) {	// L4625
  #pragma HLS pipeline II=1
    int8_t v4285 = v4275.read(); // v4275[m171];	// L4626
    int8_t a171;	// L4627
    a171 = v4285;	// L4628
    int32_t v4287 = v4276.read(); // v4276[m171];	// L4629
    int32_t p171;	// L4630
    p171 = v4287;	// L4631
    int32_t v4289 = p171;	// L4632
    int8_t v4290 = a171;	// L4633
    int8_t v4291 = w171;	// L4634
    int16_t v4292 = v4290;	// L4635
    int16_t v4293 = v4291;	// L4636
    int16_t v4294 = v4292 * v4293;	// L4637
    ap_int<33> v4295 = v4289;	// L4638
    ap_int<33> v4296 = v4294;	// L4639
    ap_int<33> v4297 = v4295 + v4296;	// L4640
    int32_t v4298 = v4297;	// L4641
    v4278.write(v4298); // v4278[m171] = v4298;	// L4642
    int8_t v4299 = a171;	// L4643
    v4277.write(v4299); // v4277[m171] = v4299;	// L4644
  }
}

void PE_kernel_ws_gemm_12_10(
  hls::stream< int8_t > &v4300 /* v4300[16] */,
  hls::stream< int32_t > &v4301 /* v4301[16] */,
  hls::stream< int8_t > &v4302 /* v4302[16] */,
  hls::stream< int32_t > &v4303 /* v4303[16] */,
  int8_t v4304[16][16],
  int v4305,
  int v4306
) {	// L4648
  #pragma HLS stream variable=v4300 depth=17
  #pragma HLS stream variable=v4301 depth=17
  #pragma HLS stream variable=v4302 depth=17
  #pragma HLS stream variable=v4303 depth=17
  #pragma HLS array_partition variable=v4304 complete dim=1
  #pragma HLS array_partition variable=v4304 complete dim=2

  int8_t v4307 = v4304[v4305][v4306];	// L4649
  int8_t w172;	// L4650
  w172 = v4307;	// L4651
  l_step_m172: for (int m172 = 0; m172 < 16; m172++) {	// L4652
  #pragma HLS pipeline II=1
    int8_t v4310 = v4300.read(); // v4300[m172];	// L4653
    int8_t a172;	// L4654
    a172 = v4310;	// L4655
    int32_t v4312 = v4301.read(); // v4301[m172];	// L4656
    int32_t p172;	// L4657
    p172 = v4312;	// L4658
    int32_t v4314 = p172;	// L4659
    int8_t v4315 = a172;	// L4660
    int8_t v4316 = w172;	// L4661
    int16_t v4317 = v4315;	// L4662
    int16_t v4318 = v4316;	// L4663
    int16_t v4319 = v4317 * v4318;	// L4664
    ap_int<33> v4320 = v4314;	// L4665
    ap_int<33> v4321 = v4319;	// L4666
    ap_int<33> v4322 = v4320 + v4321;	// L4667
    int32_t v4323 = v4322;	// L4668
    v4303.write(v4323); // v4303[m172] = v4323;	// L4669
    int8_t v4324 = a172;	// L4670
    v4302.write(v4324); // v4302[m172] = v4324;	// L4671
  }
}

void PE_kernel_ws_gemm_13_10(
  hls::stream< int8_t > &v4325 /* v4325[16] */,
  hls::stream< int32_t > &v4326 /* v4326[16] */,
  hls::stream< int8_t > &v4327 /* v4327[16] */,
  hls::stream< int32_t > &v4328 /* v4328[16] */,
  int8_t v4329[16][16],
  int v4330,
  int v4331
) {	// L4675
  #pragma HLS stream variable=v4325 depth=17
  #pragma HLS stream variable=v4326 depth=17
  #pragma HLS stream variable=v4327 depth=17
  #pragma HLS stream variable=v4328 depth=17
  #pragma HLS array_partition variable=v4329 complete dim=1
  #pragma HLS array_partition variable=v4329 complete dim=2

  int8_t v4332 = v4329[v4330][v4331];	// L4676
  int8_t w173;	// L4677
  w173 = v4332;	// L4678
  l_step_m173: for (int m173 = 0; m173 < 16; m173++) {	// L4679
  #pragma HLS pipeline II=1
    int8_t v4335 = v4325.read(); // v4325[m173];	// L4680
    int8_t a173;	// L4681
    a173 = v4335;	// L4682
    int32_t v4337 = v4326.read(); // v4326[m173];	// L4683
    int32_t p173;	// L4684
    p173 = v4337;	// L4685
    int32_t v4339 = p173;	// L4686
    int8_t v4340 = a173;	// L4687
    int8_t v4341 = w173;	// L4688
    int16_t v4342 = v4340;	// L4689
    int16_t v4343 = v4341;	// L4690
    int16_t v4344 = v4342 * v4343;	// L4691
    ap_int<33> v4345 = v4339;	// L4692
    ap_int<33> v4346 = v4344;	// L4693
    ap_int<33> v4347 = v4345 + v4346;	// L4694
    int32_t v4348 = v4347;	// L4695
    v4328.write(v4348); // v4328[m173] = v4348;	// L4696
    int8_t v4349 = a173;	// L4697
    v4327.write(v4349); // v4327[m173] = v4349;	// L4698
  }
}

void PE_kernel_ws_gemm_14_10(
  hls::stream< int8_t > &v4350 /* v4350[16] */,
  hls::stream< int32_t > &v4351 /* v4351[16] */,
  hls::stream< int8_t > &v4352 /* v4352[16] */,
  hls::stream< int32_t > &v4353 /* v4353[16] */,
  int8_t v4354[16][16],
  int v4355,
  int v4356
) {	// L4702
  #pragma HLS stream variable=v4350 depth=17
  #pragma HLS stream variable=v4351 depth=17
  #pragma HLS stream variable=v4352 depth=17
  #pragma HLS stream variable=v4353 depth=17
  #pragma HLS array_partition variable=v4354 complete dim=1
  #pragma HLS array_partition variable=v4354 complete dim=2

  int8_t v4357 = v4354[v4355][v4356];	// L4703
  int8_t w174;	// L4704
  w174 = v4357;	// L4705
  l_step_m174: for (int m174 = 0; m174 < 16; m174++) {	// L4706
  #pragma HLS pipeline II=1
    int8_t v4360 = v4350.read(); // v4350[m174];	// L4707
    int8_t a174;	// L4708
    a174 = v4360;	// L4709
    int32_t v4362 = v4351.read(); // v4351[m174];	// L4710
    int32_t p174;	// L4711
    p174 = v4362;	// L4712
    int32_t v4364 = p174;	// L4713
    int8_t v4365 = a174;	// L4714
    int8_t v4366 = w174;	// L4715
    int16_t v4367 = v4365;	// L4716
    int16_t v4368 = v4366;	// L4717
    int16_t v4369 = v4367 * v4368;	// L4718
    ap_int<33> v4370 = v4364;	// L4719
    ap_int<33> v4371 = v4369;	// L4720
    ap_int<33> v4372 = v4370 + v4371;	// L4721
    int32_t v4373 = v4372;	// L4722
    v4353.write(v4373); // v4353[m174] = v4373;	// L4723
    int8_t v4374 = a174;	// L4724
    v4352.write(v4374); // v4352[m174] = v4374;	// L4725
  }
}

void PE_kernel_ws_gemm_15_10(
  hls::stream< int8_t > &v4375 /* v4375[16] */,
  hls::stream< int32_t > &v4376 /* v4376[16] */,
  hls::stream< int8_t > &v4377 /* v4377[16] */,
  hls::stream< int32_t > &v4378 /* v4378[16] */,
  int8_t v4379[16][16],
  int v4380,
  int v4381
) {	// L4729
  #pragma HLS stream variable=v4375 depth=17
  #pragma HLS stream variable=v4376 depth=17
  #pragma HLS stream variable=v4377 depth=17
  #pragma HLS stream variable=v4378 depth=17
  #pragma HLS array_partition variable=v4379 complete dim=1
  #pragma HLS array_partition variable=v4379 complete dim=2

  int8_t v4382 = v4379[v4380][v4381];	// L4730
  int8_t w175;	// L4731
  w175 = v4382;	// L4732
  l_step_m175: for (int m175 = 0; m175 < 16; m175++) {	// L4733
  #pragma HLS pipeline II=1
    int8_t v4385 = v4375.read(); // v4375[m175];	// L4734
    int8_t a175;	// L4735
    a175 = v4385;	// L4736
    int32_t v4387 = v4376.read(); // v4376[m175];	// L4737
    int32_t p175;	// L4738
    p175 = v4387;	// L4739
    int32_t v4389 = p175;	// L4740
    int8_t v4390 = a175;	// L4741
    int8_t v4391 = w175;	// L4742
    int16_t v4392 = v4390;	// L4743
    int16_t v4393 = v4391;	// L4744
    int16_t v4394 = v4392 * v4393;	// L4745
    ap_int<33> v4395 = v4389;	// L4746
    ap_int<33> v4396 = v4394;	// L4747
    ap_int<33> v4397 = v4395 + v4396;	// L4748
    int32_t v4398 = v4397;	// L4749
    v4378.write(v4398); // v4378[m175] = v4398;	// L4750
    int8_t v4399 = a175;	// L4751
    v4377.write(v4399); // v4377[m175] = v4399;	// L4752
  }
}

void PE_kernel_ws_gemm_0_11(
  hls::stream< int8_t > &v4400 /* v4400[16] */,
  hls::stream< int32_t > &v4401 /* v4401[16] */,
  hls::stream< int8_t > &v4402 /* v4402[16] */,
  hls::stream< int32_t > &v4403 /* v4403[16] */,
  int8_t v4404[16][16],
  int v4405,
  int v4406
) {	// L4756
  #pragma HLS stream variable=v4400 depth=17
  #pragma HLS stream variable=v4401 depth=17
  #pragma HLS stream variable=v4402 depth=17
  #pragma HLS stream variable=v4403 depth=17
  #pragma HLS array_partition variable=v4404 complete dim=1
  #pragma HLS array_partition variable=v4404 complete dim=2

  int8_t v4407 = v4404[v4405][v4406];	// L4757
  int8_t w176;	// L4758
  w176 = v4407;	// L4759
  l_step_m176: for (int m176 = 0; m176 < 16; m176++) {	// L4760
  #pragma HLS pipeline II=1
    int8_t v4410 = v4400.read(); // v4400[m176];	// L4761
    int8_t a176;	// L4762
    a176 = v4410;	// L4763
    int32_t v4412 = v4401.read(); // v4401[m176];	// L4764
    int32_t p176;	// L4765
    p176 = v4412;	// L4766
    int32_t v4414 = p176;	// L4767
    int8_t v4415 = a176;	// L4768
    int8_t v4416 = w176;	// L4769
    int16_t v4417 = v4415;	// L4770
    int16_t v4418 = v4416;	// L4771
    int16_t v4419 = v4417 * v4418;	// L4772
    ap_int<33> v4420 = v4414;	// L4773
    ap_int<33> v4421 = v4419;	// L4774
    ap_int<33> v4422 = v4420 + v4421;	// L4775
    int32_t v4423 = v4422;	// L4776
    v4403.write(v4423); // v4403[m176] = v4423;	// L4777
    int8_t v4424 = a176;	// L4778
    v4402.write(v4424); // v4402[m176] = v4424;	// L4779
  }
}

void PE_kernel_ws_gemm_1_11(
  hls::stream< int8_t > &v4425 /* v4425[16] */,
  hls::stream< int32_t > &v4426 /* v4426[16] */,
  hls::stream< int8_t > &v4427 /* v4427[16] */,
  hls::stream< int32_t > &v4428 /* v4428[16] */,
  int8_t v4429[16][16],
  int v4430,
  int v4431
) {	// L4783
  #pragma HLS stream variable=v4425 depth=17
  #pragma HLS stream variable=v4426 depth=17
  #pragma HLS stream variable=v4427 depth=17
  #pragma HLS stream variable=v4428 depth=17
  #pragma HLS array_partition variable=v4429 complete dim=1
  #pragma HLS array_partition variable=v4429 complete dim=2

  int8_t v4432 = v4429[v4430][v4431];	// L4784
  int8_t w177;	// L4785
  w177 = v4432;	// L4786
  l_step_m177: for (int m177 = 0; m177 < 16; m177++) {	// L4787
  #pragma HLS pipeline II=1
    int8_t v4435 = v4425.read(); // v4425[m177];	// L4788
    int8_t a177;	// L4789
    a177 = v4435;	// L4790
    int32_t v4437 = v4426.read(); // v4426[m177];	// L4791
    int32_t p177;	// L4792
    p177 = v4437;	// L4793
    int32_t v4439 = p177;	// L4794
    int8_t v4440 = a177;	// L4795
    int8_t v4441 = w177;	// L4796
    int16_t v4442 = v4440;	// L4797
    int16_t v4443 = v4441;	// L4798
    int16_t v4444 = v4442 * v4443;	// L4799
    ap_int<33> v4445 = v4439;	// L4800
    ap_int<33> v4446 = v4444;	// L4801
    ap_int<33> v4447 = v4445 + v4446;	// L4802
    int32_t v4448 = v4447;	// L4803
    v4428.write(v4448); // v4428[m177] = v4448;	// L4804
    int8_t v4449 = a177;	// L4805
    v4427.write(v4449); // v4427[m177] = v4449;	// L4806
  }
}

void PE_kernel_ws_gemm_2_11(
  hls::stream< int8_t > &v4450 /* v4450[16] */,
  hls::stream< int32_t > &v4451 /* v4451[16] */,
  hls::stream< int8_t > &v4452 /* v4452[16] */,
  hls::stream< int32_t > &v4453 /* v4453[16] */,
  int8_t v4454[16][16],
  int v4455,
  int v4456
) {	// L4810
  #pragma HLS stream variable=v4450 depth=17
  #pragma HLS stream variable=v4451 depth=17
  #pragma HLS stream variable=v4452 depth=17
  #pragma HLS stream variable=v4453 depth=17
  #pragma HLS array_partition variable=v4454 complete dim=1
  #pragma HLS array_partition variable=v4454 complete dim=2

  int8_t v4457 = v4454[v4455][v4456];	// L4811
  int8_t w178;	// L4812
  w178 = v4457;	// L4813
  l_step_m178: for (int m178 = 0; m178 < 16; m178++) {	// L4814
  #pragma HLS pipeline II=1
    int8_t v4460 = v4450.read(); // v4450[m178];	// L4815
    int8_t a178;	// L4816
    a178 = v4460;	// L4817
    int32_t v4462 = v4451.read(); // v4451[m178];	// L4818
    int32_t p178;	// L4819
    p178 = v4462;	// L4820
    int32_t v4464 = p178;	// L4821
    int8_t v4465 = a178;	// L4822
    int8_t v4466 = w178;	// L4823
    int16_t v4467 = v4465;	// L4824
    int16_t v4468 = v4466;	// L4825
    int16_t v4469 = v4467 * v4468;	// L4826
    ap_int<33> v4470 = v4464;	// L4827
    ap_int<33> v4471 = v4469;	// L4828
    ap_int<33> v4472 = v4470 + v4471;	// L4829
    int32_t v4473 = v4472;	// L4830
    v4453.write(v4473); // v4453[m178] = v4473;	// L4831
    int8_t v4474 = a178;	// L4832
    v4452.write(v4474); // v4452[m178] = v4474;	// L4833
  }
}

void PE_kernel_ws_gemm_3_11(
  hls::stream< int8_t > &v4475 /* v4475[16] */,
  hls::stream< int32_t > &v4476 /* v4476[16] */,
  hls::stream< int8_t > &v4477 /* v4477[16] */,
  hls::stream< int32_t > &v4478 /* v4478[16] */,
  int8_t v4479[16][16],
  int v4480,
  int v4481
) {	// L4837
  #pragma HLS stream variable=v4475 depth=17
  #pragma HLS stream variable=v4476 depth=17
  #pragma HLS stream variable=v4477 depth=17
  #pragma HLS stream variable=v4478 depth=17
  #pragma HLS array_partition variable=v4479 complete dim=1
  #pragma HLS array_partition variable=v4479 complete dim=2

  int8_t v4482 = v4479[v4480][v4481];	// L4838
  int8_t w179;	// L4839
  w179 = v4482;	// L4840
  l_step_m179: for (int m179 = 0; m179 < 16; m179++) {	// L4841
  #pragma HLS pipeline II=1
    int8_t v4485 = v4475.read(); // v4475[m179];	// L4842
    int8_t a179;	// L4843
    a179 = v4485;	// L4844
    int32_t v4487 = v4476.read(); // v4476[m179];	// L4845
    int32_t p179;	// L4846
    p179 = v4487;	// L4847
    int32_t v4489 = p179;	// L4848
    int8_t v4490 = a179;	// L4849
    int8_t v4491 = w179;	// L4850
    int16_t v4492 = v4490;	// L4851
    int16_t v4493 = v4491;	// L4852
    int16_t v4494 = v4492 * v4493;	// L4853
    ap_int<33> v4495 = v4489;	// L4854
    ap_int<33> v4496 = v4494;	// L4855
    ap_int<33> v4497 = v4495 + v4496;	// L4856
    int32_t v4498 = v4497;	// L4857
    v4478.write(v4498); // v4478[m179] = v4498;	// L4858
    int8_t v4499 = a179;	// L4859
    v4477.write(v4499); // v4477[m179] = v4499;	// L4860
  }
}

void PE_kernel_ws_gemm_4_11(
  hls::stream< int8_t > &v4500 /* v4500[16] */,
  hls::stream< int32_t > &v4501 /* v4501[16] */,
  hls::stream< int8_t > &v4502 /* v4502[16] */,
  hls::stream< int32_t > &v4503 /* v4503[16] */,
  int8_t v4504[16][16],
  int v4505,
  int v4506
) {	// L4864
  #pragma HLS stream variable=v4500 depth=17
  #pragma HLS stream variable=v4501 depth=17
  #pragma HLS stream variable=v4502 depth=17
  #pragma HLS stream variable=v4503 depth=17
  #pragma HLS array_partition variable=v4504 complete dim=1
  #pragma HLS array_partition variable=v4504 complete dim=2

  int8_t v4507 = v4504[v4505][v4506];	// L4865
  int8_t w180;	// L4866
  w180 = v4507;	// L4867
  l_step_m180: for (int m180 = 0; m180 < 16; m180++) {	// L4868
  #pragma HLS pipeline II=1
    int8_t v4510 = v4500.read(); // v4500[m180];	// L4869
    int8_t a180;	// L4870
    a180 = v4510;	// L4871
    int32_t v4512 = v4501.read(); // v4501[m180];	// L4872
    int32_t p180;	// L4873
    p180 = v4512;	// L4874
    int32_t v4514 = p180;	// L4875
    int8_t v4515 = a180;	// L4876
    int8_t v4516 = w180;	// L4877
    int16_t v4517 = v4515;	// L4878
    int16_t v4518 = v4516;	// L4879
    int16_t v4519 = v4517 * v4518;	// L4880
    ap_int<33> v4520 = v4514;	// L4881
    ap_int<33> v4521 = v4519;	// L4882
    ap_int<33> v4522 = v4520 + v4521;	// L4883
    int32_t v4523 = v4522;	// L4884
    v4503.write(v4523); // v4503[m180] = v4523;	// L4885
    int8_t v4524 = a180;	// L4886
    v4502.write(v4524); // v4502[m180] = v4524;	// L4887
  }
}

void PE_kernel_ws_gemm_5_11(
  hls::stream< int8_t > &v4525 /* v4525[16] */,
  hls::stream< int32_t > &v4526 /* v4526[16] */,
  hls::stream< int8_t > &v4527 /* v4527[16] */,
  hls::stream< int32_t > &v4528 /* v4528[16] */,
  int8_t v4529[16][16],
  int v4530,
  int v4531
) {	// L4891
  #pragma HLS stream variable=v4525 depth=17
  #pragma HLS stream variable=v4526 depth=17
  #pragma HLS stream variable=v4527 depth=17
  #pragma HLS stream variable=v4528 depth=17
  #pragma HLS array_partition variable=v4529 complete dim=1
  #pragma HLS array_partition variable=v4529 complete dim=2

  int8_t v4532 = v4529[v4530][v4531];	// L4892
  int8_t w181;	// L4893
  w181 = v4532;	// L4894
  l_step_m181: for (int m181 = 0; m181 < 16; m181++) {	// L4895
  #pragma HLS pipeline II=1
    int8_t v4535 = v4525.read(); // v4525[m181];	// L4896
    int8_t a181;	// L4897
    a181 = v4535;	// L4898
    int32_t v4537 = v4526.read(); // v4526[m181];	// L4899
    int32_t p181;	// L4900
    p181 = v4537;	// L4901
    int32_t v4539 = p181;	// L4902
    int8_t v4540 = a181;	// L4903
    int8_t v4541 = w181;	// L4904
    int16_t v4542 = v4540;	// L4905
    int16_t v4543 = v4541;	// L4906
    int16_t v4544 = v4542 * v4543;	// L4907
    ap_int<33> v4545 = v4539;	// L4908
    ap_int<33> v4546 = v4544;	// L4909
    ap_int<33> v4547 = v4545 + v4546;	// L4910
    int32_t v4548 = v4547;	// L4911
    v4528.write(v4548); // v4528[m181] = v4548;	// L4912
    int8_t v4549 = a181;	// L4913
    v4527.write(v4549); // v4527[m181] = v4549;	// L4914
  }
}

void PE_kernel_ws_gemm_6_11(
  hls::stream< int8_t > &v4550 /* v4550[16] */,
  hls::stream< int32_t > &v4551 /* v4551[16] */,
  hls::stream< int8_t > &v4552 /* v4552[16] */,
  hls::stream< int32_t > &v4553 /* v4553[16] */,
  int8_t v4554[16][16],
  int v4555,
  int v4556
) {	// L4918
  #pragma HLS stream variable=v4550 depth=17
  #pragma HLS stream variable=v4551 depth=17
  #pragma HLS stream variable=v4552 depth=17
  #pragma HLS stream variable=v4553 depth=17
  #pragma HLS array_partition variable=v4554 complete dim=1
  #pragma HLS array_partition variable=v4554 complete dim=2

  int8_t v4557 = v4554[v4555][v4556];	// L4919
  int8_t w182;	// L4920
  w182 = v4557;	// L4921
  l_step_m182: for (int m182 = 0; m182 < 16; m182++) {	// L4922
  #pragma HLS pipeline II=1
    int8_t v4560 = v4550.read(); // v4550[m182];	// L4923
    int8_t a182;	// L4924
    a182 = v4560;	// L4925
    int32_t v4562 = v4551.read(); // v4551[m182];	// L4926
    int32_t p182;	// L4927
    p182 = v4562;	// L4928
    int32_t v4564 = p182;	// L4929
    int8_t v4565 = a182;	// L4930
    int8_t v4566 = w182;	// L4931
    int16_t v4567 = v4565;	// L4932
    int16_t v4568 = v4566;	// L4933
    int16_t v4569 = v4567 * v4568;	// L4934
    ap_int<33> v4570 = v4564;	// L4935
    ap_int<33> v4571 = v4569;	// L4936
    ap_int<33> v4572 = v4570 + v4571;	// L4937
    int32_t v4573 = v4572;	// L4938
    v4553.write(v4573); // v4553[m182] = v4573;	// L4939
    int8_t v4574 = a182;	// L4940
    v4552.write(v4574); // v4552[m182] = v4574;	// L4941
  }
}

void PE_kernel_ws_gemm_7_11(
  hls::stream< int8_t > &v4575 /* v4575[16] */,
  hls::stream< int32_t > &v4576 /* v4576[16] */,
  hls::stream< int8_t > &v4577 /* v4577[16] */,
  hls::stream< int32_t > &v4578 /* v4578[16] */,
  int8_t v4579[16][16],
  int v4580,
  int v4581
) {	// L4945
  #pragma HLS stream variable=v4575 depth=17
  #pragma HLS stream variable=v4576 depth=17
  #pragma HLS stream variable=v4577 depth=17
  #pragma HLS stream variable=v4578 depth=17
  #pragma HLS array_partition variable=v4579 complete dim=1
  #pragma HLS array_partition variable=v4579 complete dim=2

  int8_t v4582 = v4579[v4580][v4581];	// L4946
  int8_t w183;	// L4947
  w183 = v4582;	// L4948
  l_step_m183: for (int m183 = 0; m183 < 16; m183++) {	// L4949
  #pragma HLS pipeline II=1
    int8_t v4585 = v4575.read(); // v4575[m183];	// L4950
    int8_t a183;	// L4951
    a183 = v4585;	// L4952
    int32_t v4587 = v4576.read(); // v4576[m183];	// L4953
    int32_t p183;	// L4954
    p183 = v4587;	// L4955
    int32_t v4589 = p183;	// L4956
    int8_t v4590 = a183;	// L4957
    int8_t v4591 = w183;	// L4958
    int16_t v4592 = v4590;	// L4959
    int16_t v4593 = v4591;	// L4960
    int16_t v4594 = v4592 * v4593;	// L4961
    ap_int<33> v4595 = v4589;	// L4962
    ap_int<33> v4596 = v4594;	// L4963
    ap_int<33> v4597 = v4595 + v4596;	// L4964
    int32_t v4598 = v4597;	// L4965
    v4578.write(v4598); // v4578[m183] = v4598;	// L4966
    int8_t v4599 = a183;	// L4967
    v4577.write(v4599); // v4577[m183] = v4599;	// L4968
  }
}

void PE_kernel_ws_gemm_8_11(
  hls::stream< int8_t > &v4600 /* v4600[16] */,
  hls::stream< int32_t > &v4601 /* v4601[16] */,
  hls::stream< int8_t > &v4602 /* v4602[16] */,
  hls::stream< int32_t > &v4603 /* v4603[16] */,
  int8_t v4604[16][16],
  int v4605,
  int v4606
) {	// L4972
  #pragma HLS stream variable=v4600 depth=17
  #pragma HLS stream variable=v4601 depth=17
  #pragma HLS stream variable=v4602 depth=17
  #pragma HLS stream variable=v4603 depth=17
  #pragma HLS array_partition variable=v4604 complete dim=1
  #pragma HLS array_partition variable=v4604 complete dim=2

  int8_t v4607 = v4604[v4605][v4606];	// L4973
  int8_t w184;	// L4974
  w184 = v4607;	// L4975
  l_step_m184: for (int m184 = 0; m184 < 16; m184++) {	// L4976
  #pragma HLS pipeline II=1
    int8_t v4610 = v4600.read(); // v4600[m184];	// L4977
    int8_t a184;	// L4978
    a184 = v4610;	// L4979
    int32_t v4612 = v4601.read(); // v4601[m184];	// L4980
    int32_t p184;	// L4981
    p184 = v4612;	// L4982
    int32_t v4614 = p184;	// L4983
    int8_t v4615 = a184;	// L4984
    int8_t v4616 = w184;	// L4985
    int16_t v4617 = v4615;	// L4986
    int16_t v4618 = v4616;	// L4987
    int16_t v4619 = v4617 * v4618;	// L4988
    ap_int<33> v4620 = v4614;	// L4989
    ap_int<33> v4621 = v4619;	// L4990
    ap_int<33> v4622 = v4620 + v4621;	// L4991
    int32_t v4623 = v4622;	// L4992
    v4603.write(v4623); // v4603[m184] = v4623;	// L4993
    int8_t v4624 = a184;	// L4994
    v4602.write(v4624); // v4602[m184] = v4624;	// L4995
  }
}

void PE_kernel_ws_gemm_9_11(
  hls::stream< int8_t > &v4625 /* v4625[16] */,
  hls::stream< int32_t > &v4626 /* v4626[16] */,
  hls::stream< int8_t > &v4627 /* v4627[16] */,
  hls::stream< int32_t > &v4628 /* v4628[16] */,
  int8_t v4629[16][16],
  int v4630,
  int v4631
) {	// L4999
  #pragma HLS stream variable=v4625 depth=17
  #pragma HLS stream variable=v4626 depth=17
  #pragma HLS stream variable=v4627 depth=17
  #pragma HLS stream variable=v4628 depth=17
  #pragma HLS array_partition variable=v4629 complete dim=1
  #pragma HLS array_partition variable=v4629 complete dim=2

  int8_t v4632 = v4629[v4630][v4631];	// L5000
  int8_t w185;	// L5001
  w185 = v4632;	// L5002
  l_step_m185: for (int m185 = 0; m185 < 16; m185++) {	// L5003
  #pragma HLS pipeline II=1
    int8_t v4635 = v4625.read(); // v4625[m185];	// L5004
    int8_t a185;	// L5005
    a185 = v4635;	// L5006
    int32_t v4637 = v4626.read(); // v4626[m185];	// L5007
    int32_t p185;	// L5008
    p185 = v4637;	// L5009
    int32_t v4639 = p185;	// L5010
    int8_t v4640 = a185;	// L5011
    int8_t v4641 = w185;	// L5012
    int16_t v4642 = v4640;	// L5013
    int16_t v4643 = v4641;	// L5014
    int16_t v4644 = v4642 * v4643;	// L5015
    ap_int<33> v4645 = v4639;	// L5016
    ap_int<33> v4646 = v4644;	// L5017
    ap_int<33> v4647 = v4645 + v4646;	// L5018
    int32_t v4648 = v4647;	// L5019
    v4628.write(v4648); // v4628[m185] = v4648;	// L5020
    int8_t v4649 = a185;	// L5021
    v4627.write(v4649); // v4627[m185] = v4649;	// L5022
  }
}

void PE_kernel_ws_gemm_10_11(
  hls::stream< int8_t > &v4650 /* v4650[16] */,
  hls::stream< int32_t > &v4651 /* v4651[16] */,
  hls::stream< int8_t > &v4652 /* v4652[16] */,
  hls::stream< int32_t > &v4653 /* v4653[16] */,
  int8_t v4654[16][16],
  int v4655,
  int v4656
) {	// L5026
  #pragma HLS stream variable=v4650 depth=17
  #pragma HLS stream variable=v4651 depth=17
  #pragma HLS stream variable=v4652 depth=17
  #pragma HLS stream variable=v4653 depth=17
  #pragma HLS array_partition variable=v4654 complete dim=1
  #pragma HLS array_partition variable=v4654 complete dim=2

  int8_t v4657 = v4654[v4655][v4656];	// L5027
  int8_t w186;	// L5028
  w186 = v4657;	// L5029
  l_step_m186: for (int m186 = 0; m186 < 16; m186++) {	// L5030
  #pragma HLS pipeline II=1
    int8_t v4660 = v4650.read(); // v4650[m186];	// L5031
    int8_t a186;	// L5032
    a186 = v4660;	// L5033
    int32_t v4662 = v4651.read(); // v4651[m186];	// L5034
    int32_t p186;	// L5035
    p186 = v4662;	// L5036
    int32_t v4664 = p186;	// L5037
    int8_t v4665 = a186;	// L5038
    int8_t v4666 = w186;	// L5039
    int16_t v4667 = v4665;	// L5040
    int16_t v4668 = v4666;	// L5041
    int16_t v4669 = v4667 * v4668;	// L5042
    ap_int<33> v4670 = v4664;	// L5043
    ap_int<33> v4671 = v4669;	// L5044
    ap_int<33> v4672 = v4670 + v4671;	// L5045
    int32_t v4673 = v4672;	// L5046
    v4653.write(v4673); // v4653[m186] = v4673;	// L5047
    int8_t v4674 = a186;	// L5048
    v4652.write(v4674); // v4652[m186] = v4674;	// L5049
  }
}

void PE_kernel_ws_gemm_11_11(
  hls::stream< int8_t > &v4675 /* v4675[16] */,
  hls::stream< int32_t > &v4676 /* v4676[16] */,
  hls::stream< int8_t > &v4677 /* v4677[16] */,
  hls::stream< int32_t > &v4678 /* v4678[16] */,
  int8_t v4679[16][16],
  int v4680,
  int v4681
) {	// L5053
  #pragma HLS stream variable=v4675 depth=17
  #pragma HLS stream variable=v4676 depth=17
  #pragma HLS stream variable=v4677 depth=17
  #pragma HLS stream variable=v4678 depth=17
  #pragma HLS array_partition variable=v4679 complete dim=1
  #pragma HLS array_partition variable=v4679 complete dim=2

  int8_t v4682 = v4679[v4680][v4681];	// L5054
  int8_t w187;	// L5055
  w187 = v4682;	// L5056
  l_step_m187: for (int m187 = 0; m187 < 16; m187++) {	// L5057
  #pragma HLS pipeline II=1
    int8_t v4685 = v4675.read(); // v4675[m187];	// L5058
    int8_t a187;	// L5059
    a187 = v4685;	// L5060
    int32_t v4687 = v4676.read(); // v4676[m187];	// L5061
    int32_t p187;	// L5062
    p187 = v4687;	// L5063
    int32_t v4689 = p187;	// L5064
    int8_t v4690 = a187;	// L5065
    int8_t v4691 = w187;	// L5066
    int16_t v4692 = v4690;	// L5067
    int16_t v4693 = v4691;	// L5068
    int16_t v4694 = v4692 * v4693;	// L5069
    ap_int<33> v4695 = v4689;	// L5070
    ap_int<33> v4696 = v4694;	// L5071
    ap_int<33> v4697 = v4695 + v4696;	// L5072
    int32_t v4698 = v4697;	// L5073
    v4678.write(v4698); // v4678[m187] = v4698;	// L5074
    int8_t v4699 = a187;	// L5075
    v4677.write(v4699); // v4677[m187] = v4699;	// L5076
  }
}

void PE_kernel_ws_gemm_12_11(
  hls::stream< int8_t > &v4700 /* v4700[16] */,
  hls::stream< int32_t > &v4701 /* v4701[16] */,
  hls::stream< int8_t > &v4702 /* v4702[16] */,
  hls::stream< int32_t > &v4703 /* v4703[16] */,
  int8_t v4704[16][16],
  int v4705,
  int v4706
) {	// L5080
  #pragma HLS stream variable=v4700 depth=17
  #pragma HLS stream variable=v4701 depth=17
  #pragma HLS stream variable=v4702 depth=17
  #pragma HLS stream variable=v4703 depth=17
  #pragma HLS array_partition variable=v4704 complete dim=1
  #pragma HLS array_partition variable=v4704 complete dim=2

  int8_t v4707 = v4704[v4705][v4706];	// L5081
  int8_t w188;	// L5082
  w188 = v4707;	// L5083
  l_step_m188: for (int m188 = 0; m188 < 16; m188++) {	// L5084
  #pragma HLS pipeline II=1
    int8_t v4710 = v4700.read(); // v4700[m188];	// L5085
    int8_t a188;	// L5086
    a188 = v4710;	// L5087
    int32_t v4712 = v4701.read(); // v4701[m188];	// L5088
    int32_t p188;	// L5089
    p188 = v4712;	// L5090
    int32_t v4714 = p188;	// L5091
    int8_t v4715 = a188;	// L5092
    int8_t v4716 = w188;	// L5093
    int16_t v4717 = v4715;	// L5094
    int16_t v4718 = v4716;	// L5095
    int16_t v4719 = v4717 * v4718;	// L5096
    ap_int<33> v4720 = v4714;	// L5097
    ap_int<33> v4721 = v4719;	// L5098
    ap_int<33> v4722 = v4720 + v4721;	// L5099
    int32_t v4723 = v4722;	// L5100
    v4703.write(v4723); // v4703[m188] = v4723;	// L5101
    int8_t v4724 = a188;	// L5102
    v4702.write(v4724); // v4702[m188] = v4724;	// L5103
  }
}

void PE_kernel_ws_gemm_13_11(
  hls::stream< int8_t > &v4725 /* v4725[16] */,
  hls::stream< int32_t > &v4726 /* v4726[16] */,
  hls::stream< int8_t > &v4727 /* v4727[16] */,
  hls::stream< int32_t > &v4728 /* v4728[16] */,
  int8_t v4729[16][16],
  int v4730,
  int v4731
) {	// L5107
  #pragma HLS stream variable=v4725 depth=17
  #pragma HLS stream variable=v4726 depth=17
  #pragma HLS stream variable=v4727 depth=17
  #pragma HLS stream variable=v4728 depth=17
  #pragma HLS array_partition variable=v4729 complete dim=1
  #pragma HLS array_partition variable=v4729 complete dim=2

  int8_t v4732 = v4729[v4730][v4731];	// L5108
  int8_t w189;	// L5109
  w189 = v4732;	// L5110
  l_step_m189: for (int m189 = 0; m189 < 16; m189++) {	// L5111
  #pragma HLS pipeline II=1
    int8_t v4735 = v4725.read(); // v4725[m189];	// L5112
    int8_t a189;	// L5113
    a189 = v4735;	// L5114
    int32_t v4737 = v4726.read(); // v4726[m189];	// L5115
    int32_t p189;	// L5116
    p189 = v4737;	// L5117
    int32_t v4739 = p189;	// L5118
    int8_t v4740 = a189;	// L5119
    int8_t v4741 = w189;	// L5120
    int16_t v4742 = v4740;	// L5121
    int16_t v4743 = v4741;	// L5122
    int16_t v4744 = v4742 * v4743;	// L5123
    ap_int<33> v4745 = v4739;	// L5124
    ap_int<33> v4746 = v4744;	// L5125
    ap_int<33> v4747 = v4745 + v4746;	// L5126
    int32_t v4748 = v4747;	// L5127
    v4728.write(v4748); // v4728[m189] = v4748;	// L5128
    int8_t v4749 = a189;	// L5129
    v4727.write(v4749); // v4727[m189] = v4749;	// L5130
  }
}

void PE_kernel_ws_gemm_14_11(
  hls::stream< int8_t > &v4750 /* v4750[16] */,
  hls::stream< int32_t > &v4751 /* v4751[16] */,
  hls::stream< int8_t > &v4752 /* v4752[16] */,
  hls::stream< int32_t > &v4753 /* v4753[16] */,
  int8_t v4754[16][16],
  int v4755,
  int v4756
) {	// L5134
  #pragma HLS stream variable=v4750 depth=17
  #pragma HLS stream variable=v4751 depth=17
  #pragma HLS stream variable=v4752 depth=17
  #pragma HLS stream variable=v4753 depth=17
  #pragma HLS array_partition variable=v4754 complete dim=1
  #pragma HLS array_partition variable=v4754 complete dim=2

  int8_t v4757 = v4754[v4755][v4756];	// L5135
  int8_t w190;	// L5136
  w190 = v4757;	// L5137
  l_step_m190: for (int m190 = 0; m190 < 16; m190++) {	// L5138
  #pragma HLS pipeline II=1
    int8_t v4760 = v4750.read(); // v4750[m190];	// L5139
    int8_t a190;	// L5140
    a190 = v4760;	// L5141
    int32_t v4762 = v4751.read(); // v4751[m190];	// L5142
    int32_t p190;	// L5143
    p190 = v4762;	// L5144
    int32_t v4764 = p190;	// L5145
    int8_t v4765 = a190;	// L5146
    int8_t v4766 = w190;	// L5147
    int16_t v4767 = v4765;	// L5148
    int16_t v4768 = v4766;	// L5149
    int16_t v4769 = v4767 * v4768;	// L5150
    ap_int<33> v4770 = v4764;	// L5151
    ap_int<33> v4771 = v4769;	// L5152
    ap_int<33> v4772 = v4770 + v4771;	// L5153
    int32_t v4773 = v4772;	// L5154
    v4753.write(v4773); // v4753[m190] = v4773;	// L5155
    int8_t v4774 = a190;	// L5156
    v4752.write(v4774); // v4752[m190] = v4774;	// L5157
  }
}

void PE_kernel_ws_gemm_15_11(
  hls::stream< int8_t > &v4775 /* v4775[16] */,
  hls::stream< int32_t > &v4776 /* v4776[16] */,
  hls::stream< int8_t > &v4777 /* v4777[16] */,
  hls::stream< int32_t > &v4778 /* v4778[16] */,
  int8_t v4779[16][16],
  int v4780,
  int v4781
) {	// L5161
  #pragma HLS stream variable=v4775 depth=17
  #pragma HLS stream variable=v4776 depth=17
  #pragma HLS stream variable=v4777 depth=17
  #pragma HLS stream variable=v4778 depth=17
  #pragma HLS array_partition variable=v4779 complete dim=1
  #pragma HLS array_partition variable=v4779 complete dim=2

  int8_t v4782 = v4779[v4780][v4781];	// L5162
  int8_t w191;	// L5163
  w191 = v4782;	// L5164
  l_step_m191: for (int m191 = 0; m191 < 16; m191++) {	// L5165
  #pragma HLS pipeline II=1
    int8_t v4785 = v4775.read(); // v4775[m191];	// L5166
    int8_t a191;	// L5167
    a191 = v4785;	// L5168
    int32_t v4787 = v4776.read(); // v4776[m191];	// L5169
    int32_t p191;	// L5170
    p191 = v4787;	// L5171
    int32_t v4789 = p191;	// L5172
    int8_t v4790 = a191;	// L5173
    int8_t v4791 = w191;	// L5174
    int16_t v4792 = v4790;	// L5175
    int16_t v4793 = v4791;	// L5176
    int16_t v4794 = v4792 * v4793;	// L5177
    ap_int<33> v4795 = v4789;	// L5178
    ap_int<33> v4796 = v4794;	// L5179
    ap_int<33> v4797 = v4795 + v4796;	// L5180
    int32_t v4798 = v4797;	// L5181
    v4778.write(v4798); // v4778[m191] = v4798;	// L5182
    int8_t v4799 = a191;	// L5183
    v4777.write(v4799); // v4777[m191] = v4799;	// L5184
  }
}

void PE_kernel_ws_gemm_0_12(
  hls::stream< int8_t > &v4800 /* v4800[16] */,
  hls::stream< int32_t > &v4801 /* v4801[16] */,
  hls::stream< int8_t > &v4802 /* v4802[16] */,
  hls::stream< int32_t > &v4803 /* v4803[16] */,
  int8_t v4804[16][16],
  int v4805,
  int v4806
) {	// L5188
  #pragma HLS stream variable=v4800 depth=17
  #pragma HLS stream variable=v4801 depth=17
  #pragma HLS stream variable=v4802 depth=17
  #pragma HLS stream variable=v4803 depth=17
  #pragma HLS array_partition variable=v4804 complete dim=1
  #pragma HLS array_partition variable=v4804 complete dim=2

  int8_t v4807 = v4804[v4805][v4806];	// L5189
  int8_t w192;	// L5190
  w192 = v4807;	// L5191
  l_step_m192: for (int m192 = 0; m192 < 16; m192++) {	// L5192
  #pragma HLS pipeline II=1
    int8_t v4810 = v4800.read(); // v4800[m192];	// L5193
    int8_t a192;	// L5194
    a192 = v4810;	// L5195
    int32_t v4812 = v4801.read(); // v4801[m192];	// L5196
    int32_t p192;	// L5197
    p192 = v4812;	// L5198
    int32_t v4814 = p192;	// L5199
    int8_t v4815 = a192;	// L5200
    int8_t v4816 = w192;	// L5201
    int16_t v4817 = v4815;	// L5202
    int16_t v4818 = v4816;	// L5203
    int16_t v4819 = v4817 * v4818;	// L5204
    ap_int<33> v4820 = v4814;	// L5205
    ap_int<33> v4821 = v4819;	// L5206
    ap_int<33> v4822 = v4820 + v4821;	// L5207
    int32_t v4823 = v4822;	// L5208
    v4803.write(v4823); // v4803[m192] = v4823;	// L5209
    int8_t v4824 = a192;	// L5210
    v4802.write(v4824); // v4802[m192] = v4824;	// L5211
  }
}

void PE_kernel_ws_gemm_1_12(
  hls::stream< int8_t > &v4825 /* v4825[16] */,
  hls::stream< int32_t > &v4826 /* v4826[16] */,
  hls::stream< int8_t > &v4827 /* v4827[16] */,
  hls::stream< int32_t > &v4828 /* v4828[16] */,
  int8_t v4829[16][16],
  int v4830,
  int v4831
) {	// L5215
  #pragma HLS stream variable=v4825 depth=17
  #pragma HLS stream variable=v4826 depth=17
  #pragma HLS stream variable=v4827 depth=17
  #pragma HLS stream variable=v4828 depth=17
  #pragma HLS array_partition variable=v4829 complete dim=1
  #pragma HLS array_partition variable=v4829 complete dim=2

  int8_t v4832 = v4829[v4830][v4831];	// L5216
  int8_t w193;	// L5217
  w193 = v4832;	// L5218
  l_step_m193: for (int m193 = 0; m193 < 16; m193++) {	// L5219
  #pragma HLS pipeline II=1
    int8_t v4835 = v4825.read(); // v4825[m193];	// L5220
    int8_t a193;	// L5221
    a193 = v4835;	// L5222
    int32_t v4837 = v4826.read(); // v4826[m193];	// L5223
    int32_t p193;	// L5224
    p193 = v4837;	// L5225
    int32_t v4839 = p193;	// L5226
    int8_t v4840 = a193;	// L5227
    int8_t v4841 = w193;	// L5228
    int16_t v4842 = v4840;	// L5229
    int16_t v4843 = v4841;	// L5230
    int16_t v4844 = v4842 * v4843;	// L5231
    ap_int<33> v4845 = v4839;	// L5232
    ap_int<33> v4846 = v4844;	// L5233
    ap_int<33> v4847 = v4845 + v4846;	// L5234
    int32_t v4848 = v4847;	// L5235
    v4828.write(v4848); // v4828[m193] = v4848;	// L5236
    int8_t v4849 = a193;	// L5237
    v4827.write(v4849); // v4827[m193] = v4849;	// L5238
  }
}

void PE_kernel_ws_gemm_2_12(
  hls::stream< int8_t > &v4850 /* v4850[16] */,
  hls::stream< int32_t > &v4851 /* v4851[16] */,
  hls::stream< int8_t > &v4852 /* v4852[16] */,
  hls::stream< int32_t > &v4853 /* v4853[16] */,
  int8_t v4854[16][16],
  int v4855,
  int v4856
) {	// L5242
  #pragma HLS stream variable=v4850 depth=17
  #pragma HLS stream variable=v4851 depth=17
  #pragma HLS stream variable=v4852 depth=17
  #pragma HLS stream variable=v4853 depth=17
  #pragma HLS array_partition variable=v4854 complete dim=1
  #pragma HLS array_partition variable=v4854 complete dim=2

  int8_t v4857 = v4854[v4855][v4856];	// L5243
  int8_t w194;	// L5244
  w194 = v4857;	// L5245
  l_step_m194: for (int m194 = 0; m194 < 16; m194++) {	// L5246
  #pragma HLS pipeline II=1
    int8_t v4860 = v4850.read(); // v4850[m194];	// L5247
    int8_t a194;	// L5248
    a194 = v4860;	// L5249
    int32_t v4862 = v4851.read(); // v4851[m194];	// L5250
    int32_t p194;	// L5251
    p194 = v4862;	// L5252
    int32_t v4864 = p194;	// L5253
    int8_t v4865 = a194;	// L5254
    int8_t v4866 = w194;	// L5255
    int16_t v4867 = v4865;	// L5256
    int16_t v4868 = v4866;	// L5257
    int16_t v4869 = v4867 * v4868;	// L5258
    ap_int<33> v4870 = v4864;	// L5259
    ap_int<33> v4871 = v4869;	// L5260
    ap_int<33> v4872 = v4870 + v4871;	// L5261
    int32_t v4873 = v4872;	// L5262
    v4853.write(v4873); // v4853[m194] = v4873;	// L5263
    int8_t v4874 = a194;	// L5264
    v4852.write(v4874); // v4852[m194] = v4874;	// L5265
  }
}

void PE_kernel_ws_gemm_3_12(
  hls::stream< int8_t > &v4875 /* v4875[16] */,
  hls::stream< int32_t > &v4876 /* v4876[16] */,
  hls::stream< int8_t > &v4877 /* v4877[16] */,
  hls::stream< int32_t > &v4878 /* v4878[16] */,
  int8_t v4879[16][16],
  int v4880,
  int v4881
) {	// L5269
  #pragma HLS stream variable=v4875 depth=17
  #pragma HLS stream variable=v4876 depth=17
  #pragma HLS stream variable=v4877 depth=17
  #pragma HLS stream variable=v4878 depth=17
  #pragma HLS array_partition variable=v4879 complete dim=1
  #pragma HLS array_partition variable=v4879 complete dim=2

  int8_t v4882 = v4879[v4880][v4881];	// L5270
  int8_t w195;	// L5271
  w195 = v4882;	// L5272
  l_step_m195: for (int m195 = 0; m195 < 16; m195++) {	// L5273
  #pragma HLS pipeline II=1
    int8_t v4885 = v4875.read(); // v4875[m195];	// L5274
    int8_t a195;	// L5275
    a195 = v4885;	// L5276
    int32_t v4887 = v4876.read(); // v4876[m195];	// L5277
    int32_t p195;	// L5278
    p195 = v4887;	// L5279
    int32_t v4889 = p195;	// L5280
    int8_t v4890 = a195;	// L5281
    int8_t v4891 = w195;	// L5282
    int16_t v4892 = v4890;	// L5283
    int16_t v4893 = v4891;	// L5284
    int16_t v4894 = v4892 * v4893;	// L5285
    ap_int<33> v4895 = v4889;	// L5286
    ap_int<33> v4896 = v4894;	// L5287
    ap_int<33> v4897 = v4895 + v4896;	// L5288
    int32_t v4898 = v4897;	// L5289
    v4878.write(v4898); // v4878[m195] = v4898;	// L5290
    int8_t v4899 = a195;	// L5291
    v4877.write(v4899); // v4877[m195] = v4899;	// L5292
  }
}

void PE_kernel_ws_gemm_4_12(
  hls::stream< int8_t > &v4900 /* v4900[16] */,
  hls::stream< int32_t > &v4901 /* v4901[16] */,
  hls::stream< int8_t > &v4902 /* v4902[16] */,
  hls::stream< int32_t > &v4903 /* v4903[16] */,
  int8_t v4904[16][16],
  int v4905,
  int v4906
) {	// L5296
  #pragma HLS stream variable=v4900 depth=17
  #pragma HLS stream variable=v4901 depth=17
  #pragma HLS stream variable=v4902 depth=17
  #pragma HLS stream variable=v4903 depth=17
  #pragma HLS array_partition variable=v4904 complete dim=1
  #pragma HLS array_partition variable=v4904 complete dim=2

  int8_t v4907 = v4904[v4905][v4906];	// L5297
  int8_t w196;	// L5298
  w196 = v4907;	// L5299
  l_step_m196: for (int m196 = 0; m196 < 16; m196++) {	// L5300
  #pragma HLS pipeline II=1
    int8_t v4910 = v4900.read(); // v4900[m196];	// L5301
    int8_t a196;	// L5302
    a196 = v4910;	// L5303
    int32_t v4912 = v4901.read(); // v4901[m196];	// L5304
    int32_t p196;	// L5305
    p196 = v4912;	// L5306
    int32_t v4914 = p196;	// L5307
    int8_t v4915 = a196;	// L5308
    int8_t v4916 = w196;	// L5309
    int16_t v4917 = v4915;	// L5310
    int16_t v4918 = v4916;	// L5311
    int16_t v4919 = v4917 * v4918;	// L5312
    ap_int<33> v4920 = v4914;	// L5313
    ap_int<33> v4921 = v4919;	// L5314
    ap_int<33> v4922 = v4920 + v4921;	// L5315
    int32_t v4923 = v4922;	// L5316
    v4903.write(v4923); // v4903[m196] = v4923;	// L5317
    int8_t v4924 = a196;	// L5318
    v4902.write(v4924); // v4902[m196] = v4924;	// L5319
  }
}

void PE_kernel_ws_gemm_5_12(
  hls::stream< int8_t > &v4925 /* v4925[16] */,
  hls::stream< int32_t > &v4926 /* v4926[16] */,
  hls::stream< int8_t > &v4927 /* v4927[16] */,
  hls::stream< int32_t > &v4928 /* v4928[16] */,
  int8_t v4929[16][16],
  int v4930,
  int v4931
) {	// L5323
  #pragma HLS stream variable=v4925 depth=17
  #pragma HLS stream variable=v4926 depth=17
  #pragma HLS stream variable=v4927 depth=17
  #pragma HLS stream variable=v4928 depth=17
  #pragma HLS array_partition variable=v4929 complete dim=1
  #pragma HLS array_partition variable=v4929 complete dim=2

  int8_t v4932 = v4929[v4930][v4931];	// L5324
  int8_t w197;	// L5325
  w197 = v4932;	// L5326
  l_step_m197: for (int m197 = 0; m197 < 16; m197++) {	// L5327
  #pragma HLS pipeline II=1
    int8_t v4935 = v4925.read(); // v4925[m197];	// L5328
    int8_t a197;	// L5329
    a197 = v4935;	// L5330
    int32_t v4937 = v4926.read(); // v4926[m197];	// L5331
    int32_t p197;	// L5332
    p197 = v4937;	// L5333
    int32_t v4939 = p197;	// L5334
    int8_t v4940 = a197;	// L5335
    int8_t v4941 = w197;	// L5336
    int16_t v4942 = v4940;	// L5337
    int16_t v4943 = v4941;	// L5338
    int16_t v4944 = v4942 * v4943;	// L5339
    ap_int<33> v4945 = v4939;	// L5340
    ap_int<33> v4946 = v4944;	// L5341
    ap_int<33> v4947 = v4945 + v4946;	// L5342
    int32_t v4948 = v4947;	// L5343
    v4928.write(v4948); // v4928[m197] = v4948;	// L5344
    int8_t v4949 = a197;	// L5345
    v4927.write(v4949); // v4927[m197] = v4949;	// L5346
  }
}

void PE_kernel_ws_gemm_6_12(
  hls::stream< int8_t > &v4950 /* v4950[16] */,
  hls::stream< int32_t > &v4951 /* v4951[16] */,
  hls::stream< int8_t > &v4952 /* v4952[16] */,
  hls::stream< int32_t > &v4953 /* v4953[16] */,
  int8_t v4954[16][16],
  int v4955,
  int v4956
) {	// L5350
  #pragma HLS stream variable=v4950 depth=17
  #pragma HLS stream variable=v4951 depth=17
  #pragma HLS stream variable=v4952 depth=17
  #pragma HLS stream variable=v4953 depth=17
  #pragma HLS array_partition variable=v4954 complete dim=1
  #pragma HLS array_partition variable=v4954 complete dim=2

  int8_t v4957 = v4954[v4955][v4956];	// L5351
  int8_t w198;	// L5352
  w198 = v4957;	// L5353
  l_step_m198: for (int m198 = 0; m198 < 16; m198++) {	// L5354
  #pragma HLS pipeline II=1
    int8_t v4960 = v4950.read(); // v4950[m198];	// L5355
    int8_t a198;	// L5356
    a198 = v4960;	// L5357
    int32_t v4962 = v4951.read(); // v4951[m198];	// L5358
    int32_t p198;	// L5359
    p198 = v4962;	// L5360
    int32_t v4964 = p198;	// L5361
    int8_t v4965 = a198;	// L5362
    int8_t v4966 = w198;	// L5363
    int16_t v4967 = v4965;	// L5364
    int16_t v4968 = v4966;	// L5365
    int16_t v4969 = v4967 * v4968;	// L5366
    ap_int<33> v4970 = v4964;	// L5367
    ap_int<33> v4971 = v4969;	// L5368
    ap_int<33> v4972 = v4970 + v4971;	// L5369
    int32_t v4973 = v4972;	// L5370
    v4953.write(v4973); // v4953[m198] = v4973;	// L5371
    int8_t v4974 = a198;	// L5372
    v4952.write(v4974); // v4952[m198] = v4974;	// L5373
  }
}

void PE_kernel_ws_gemm_7_12(
  hls::stream< int8_t > &v4975 /* v4975[16] */,
  hls::stream< int32_t > &v4976 /* v4976[16] */,
  hls::stream< int8_t > &v4977 /* v4977[16] */,
  hls::stream< int32_t > &v4978 /* v4978[16] */,
  int8_t v4979[16][16],
  int v4980,
  int v4981
) {	// L5377
  #pragma HLS stream variable=v4975 depth=17
  #pragma HLS stream variable=v4976 depth=17
  #pragma HLS stream variable=v4977 depth=17
  #pragma HLS stream variable=v4978 depth=17
  #pragma HLS array_partition variable=v4979 complete dim=1
  #pragma HLS array_partition variable=v4979 complete dim=2

  int8_t v4982 = v4979[v4980][v4981];	// L5378
  int8_t w199;	// L5379
  w199 = v4982;	// L5380
  l_step_m199: for (int m199 = 0; m199 < 16; m199++) {	// L5381
  #pragma HLS pipeline II=1
    int8_t v4985 = v4975.read(); // v4975[m199];	// L5382
    int8_t a199;	// L5383
    a199 = v4985;	// L5384
    int32_t v4987 = v4976.read(); // v4976[m199];	// L5385
    int32_t p199;	// L5386
    p199 = v4987;	// L5387
    int32_t v4989 = p199;	// L5388
    int8_t v4990 = a199;	// L5389
    int8_t v4991 = w199;	// L5390
    int16_t v4992 = v4990;	// L5391
    int16_t v4993 = v4991;	// L5392
    int16_t v4994 = v4992 * v4993;	// L5393
    ap_int<33> v4995 = v4989;	// L5394
    ap_int<33> v4996 = v4994;	// L5395
    ap_int<33> v4997 = v4995 + v4996;	// L5396
    int32_t v4998 = v4997;	// L5397
    v4978.write(v4998); // v4978[m199] = v4998;	// L5398
    int8_t v4999 = a199;	// L5399
    v4977.write(v4999); // v4977[m199] = v4999;	// L5400
  }
}

void PE_kernel_ws_gemm_8_12(
  hls::stream< int8_t > &v5000 /* v5000[16] */,
  hls::stream< int32_t > &v5001 /* v5001[16] */,
  hls::stream< int8_t > &v5002 /* v5002[16] */,
  hls::stream< int32_t > &v5003 /* v5003[16] */,
  int8_t v5004[16][16],
  int v5005,
  int v5006
) {	// L5404
  #pragma HLS stream variable=v5000 depth=17
  #pragma HLS stream variable=v5001 depth=17
  #pragma HLS stream variable=v5002 depth=17
  #pragma HLS stream variable=v5003 depth=17
  #pragma HLS array_partition variable=v5004 complete dim=1
  #pragma HLS array_partition variable=v5004 complete dim=2

  int8_t v5007 = v5004[v5005][v5006];	// L5405
  int8_t w200;	// L5406
  w200 = v5007;	// L5407
  l_step_m200: for (int m200 = 0; m200 < 16; m200++) {	// L5408
  #pragma HLS pipeline II=1
    int8_t v5010 = v5000.read(); // v5000[m200];	// L5409
    int8_t a200;	// L5410
    a200 = v5010;	// L5411
    int32_t v5012 = v5001.read(); // v5001[m200];	// L5412
    int32_t p200;	// L5413
    p200 = v5012;	// L5414
    int32_t v5014 = p200;	// L5415
    int8_t v5015 = a200;	// L5416
    int8_t v5016 = w200;	// L5417
    int16_t v5017 = v5015;	// L5418
    int16_t v5018 = v5016;	// L5419
    int16_t v5019 = v5017 * v5018;	// L5420
    ap_int<33> v5020 = v5014;	// L5421
    ap_int<33> v5021 = v5019;	// L5422
    ap_int<33> v5022 = v5020 + v5021;	// L5423
    int32_t v5023 = v5022;	// L5424
    v5003.write(v5023); // v5003[m200] = v5023;	// L5425
    int8_t v5024 = a200;	// L5426
    v5002.write(v5024); // v5002[m200] = v5024;	// L5427
  }
}

void PE_kernel_ws_gemm_9_12(
  hls::stream< int8_t > &v5025 /* v5025[16] */,
  hls::stream< int32_t > &v5026 /* v5026[16] */,
  hls::stream< int8_t > &v5027 /* v5027[16] */,
  hls::stream< int32_t > &v5028 /* v5028[16] */,
  int8_t v5029[16][16],
  int v5030,
  int v5031
) {	// L5431
  #pragma HLS stream variable=v5025 depth=17
  #pragma HLS stream variable=v5026 depth=17
  #pragma HLS stream variable=v5027 depth=17
  #pragma HLS stream variable=v5028 depth=17
  #pragma HLS array_partition variable=v5029 complete dim=1
  #pragma HLS array_partition variable=v5029 complete dim=2

  int8_t v5032 = v5029[v5030][v5031];	// L5432
  int8_t w201;	// L5433
  w201 = v5032;	// L5434
  l_step_m201: for (int m201 = 0; m201 < 16; m201++) {	// L5435
  #pragma HLS pipeline II=1
    int8_t v5035 = v5025.read(); // v5025[m201];	// L5436
    int8_t a201;	// L5437
    a201 = v5035;	// L5438
    int32_t v5037 = v5026.read(); // v5026[m201];	// L5439
    int32_t p201;	// L5440
    p201 = v5037;	// L5441
    int32_t v5039 = p201;	// L5442
    int8_t v5040 = a201;	// L5443
    int8_t v5041 = w201;	// L5444
    int16_t v5042 = v5040;	// L5445
    int16_t v5043 = v5041;	// L5446
    int16_t v5044 = v5042 * v5043;	// L5447
    ap_int<33> v5045 = v5039;	// L5448
    ap_int<33> v5046 = v5044;	// L5449
    ap_int<33> v5047 = v5045 + v5046;	// L5450
    int32_t v5048 = v5047;	// L5451
    v5028.write(v5048); // v5028[m201] = v5048;	// L5452
    int8_t v5049 = a201;	// L5453
    v5027.write(v5049); // v5027[m201] = v5049;	// L5454
  }
}

void PE_kernel_ws_gemm_10_12(
  hls::stream< int8_t > &v5050 /* v5050[16] */,
  hls::stream< int32_t > &v5051 /* v5051[16] */,
  hls::stream< int8_t > &v5052 /* v5052[16] */,
  hls::stream< int32_t > &v5053 /* v5053[16] */,
  int8_t v5054[16][16],
  int v5055,
  int v5056
) {	// L5458
  #pragma HLS stream variable=v5050 depth=17
  #pragma HLS stream variable=v5051 depth=17
  #pragma HLS stream variable=v5052 depth=17
  #pragma HLS stream variable=v5053 depth=17
  #pragma HLS array_partition variable=v5054 complete dim=1
  #pragma HLS array_partition variable=v5054 complete dim=2

  int8_t v5057 = v5054[v5055][v5056];	// L5459
  int8_t w202;	// L5460
  w202 = v5057;	// L5461
  l_step_m202: for (int m202 = 0; m202 < 16; m202++) {	// L5462
  #pragma HLS pipeline II=1
    int8_t v5060 = v5050.read(); // v5050[m202];	// L5463
    int8_t a202;	// L5464
    a202 = v5060;	// L5465
    int32_t v5062 = v5051.read(); // v5051[m202];	// L5466
    int32_t p202;	// L5467
    p202 = v5062;	// L5468
    int32_t v5064 = p202;	// L5469
    int8_t v5065 = a202;	// L5470
    int8_t v5066 = w202;	// L5471
    int16_t v5067 = v5065;	// L5472
    int16_t v5068 = v5066;	// L5473
    int16_t v5069 = v5067 * v5068;	// L5474
    ap_int<33> v5070 = v5064;	// L5475
    ap_int<33> v5071 = v5069;	// L5476
    ap_int<33> v5072 = v5070 + v5071;	// L5477
    int32_t v5073 = v5072;	// L5478
    v5053.write(v5073); // v5053[m202] = v5073;	// L5479
    int8_t v5074 = a202;	// L5480
    v5052.write(v5074); // v5052[m202] = v5074;	// L5481
  }
}

void PE_kernel_ws_gemm_11_12(
  hls::stream< int8_t > &v5075 /* v5075[16] */,
  hls::stream< int32_t > &v5076 /* v5076[16] */,
  hls::stream< int8_t > &v5077 /* v5077[16] */,
  hls::stream< int32_t > &v5078 /* v5078[16] */,
  int8_t v5079[16][16],
  int v5080,
  int v5081
) {	// L5485
  #pragma HLS stream variable=v5075 depth=17
  #pragma HLS stream variable=v5076 depth=17
  #pragma HLS stream variable=v5077 depth=17
  #pragma HLS stream variable=v5078 depth=17
  #pragma HLS array_partition variable=v5079 complete dim=1
  #pragma HLS array_partition variable=v5079 complete dim=2

  int8_t v5082 = v5079[v5080][v5081];	// L5486
  int8_t w203;	// L5487
  w203 = v5082;	// L5488
  l_step_m203: for (int m203 = 0; m203 < 16; m203++) {	// L5489
  #pragma HLS pipeline II=1
    int8_t v5085 = v5075.read(); // v5075[m203];	// L5490
    int8_t a203;	// L5491
    a203 = v5085;	// L5492
    int32_t v5087 = v5076.read(); // v5076[m203];	// L5493
    int32_t p203;	// L5494
    p203 = v5087;	// L5495
    int32_t v5089 = p203;	// L5496
    int8_t v5090 = a203;	// L5497
    int8_t v5091 = w203;	// L5498
    int16_t v5092 = v5090;	// L5499
    int16_t v5093 = v5091;	// L5500
    int16_t v5094 = v5092 * v5093;	// L5501
    ap_int<33> v5095 = v5089;	// L5502
    ap_int<33> v5096 = v5094;	// L5503
    ap_int<33> v5097 = v5095 + v5096;	// L5504
    int32_t v5098 = v5097;	// L5505
    v5078.write(v5098); // v5078[m203] = v5098;	// L5506
    int8_t v5099 = a203;	// L5507
    v5077.write(v5099); // v5077[m203] = v5099;	// L5508
  }
}

void PE_kernel_ws_gemm_12_12(
  hls::stream< int8_t > &v5100 /* v5100[16] */,
  hls::stream< int32_t > &v5101 /* v5101[16] */,
  hls::stream< int8_t > &v5102 /* v5102[16] */,
  hls::stream< int32_t > &v5103 /* v5103[16] */,
  int8_t v5104[16][16],
  int v5105,
  int v5106
) {	// L5512
  #pragma HLS stream variable=v5100 depth=17
  #pragma HLS stream variable=v5101 depth=17
  #pragma HLS stream variable=v5102 depth=17
  #pragma HLS stream variable=v5103 depth=17
  #pragma HLS array_partition variable=v5104 complete dim=1
  #pragma HLS array_partition variable=v5104 complete dim=2

  int8_t v5107 = v5104[v5105][v5106];	// L5513
  int8_t w204;	// L5514
  w204 = v5107;	// L5515
  l_step_m204: for (int m204 = 0; m204 < 16; m204++) {	// L5516
  #pragma HLS pipeline II=1
    int8_t v5110 = v5100.read(); // v5100[m204];	// L5517
    int8_t a204;	// L5518
    a204 = v5110;	// L5519
    int32_t v5112 = v5101.read(); // v5101[m204];	// L5520
    int32_t p204;	// L5521
    p204 = v5112;	// L5522
    int32_t v5114 = p204;	// L5523
    int8_t v5115 = a204;	// L5524
    int8_t v5116 = w204;	// L5525
    int16_t v5117 = v5115;	// L5526
    int16_t v5118 = v5116;	// L5527
    int16_t v5119 = v5117 * v5118;	// L5528
    ap_int<33> v5120 = v5114;	// L5529
    ap_int<33> v5121 = v5119;	// L5530
    ap_int<33> v5122 = v5120 + v5121;	// L5531
    int32_t v5123 = v5122;	// L5532
    v5103.write(v5123); // v5103[m204] = v5123;	// L5533
    int8_t v5124 = a204;	// L5534
    v5102.write(v5124); // v5102[m204] = v5124;	// L5535
  }
}

void PE_kernel_ws_gemm_13_12(
  hls::stream< int8_t > &v5125 /* v5125[16] */,
  hls::stream< int32_t > &v5126 /* v5126[16] */,
  hls::stream< int8_t > &v5127 /* v5127[16] */,
  hls::stream< int32_t > &v5128 /* v5128[16] */,
  int8_t v5129[16][16],
  int v5130,
  int v5131
) {	// L5539
  #pragma HLS stream variable=v5125 depth=17
  #pragma HLS stream variable=v5126 depth=17
  #pragma HLS stream variable=v5127 depth=17
  #pragma HLS stream variable=v5128 depth=17
  #pragma HLS array_partition variable=v5129 complete dim=1
  #pragma HLS array_partition variable=v5129 complete dim=2

  int8_t v5132 = v5129[v5130][v5131];	// L5540
  int8_t w205;	// L5541
  w205 = v5132;	// L5542
  l_step_m205: for (int m205 = 0; m205 < 16; m205++) {	// L5543
  #pragma HLS pipeline II=1
    int8_t v5135 = v5125.read(); // v5125[m205];	// L5544
    int8_t a205;	// L5545
    a205 = v5135;	// L5546
    int32_t v5137 = v5126.read(); // v5126[m205];	// L5547
    int32_t p205;	// L5548
    p205 = v5137;	// L5549
    int32_t v5139 = p205;	// L5550
    int8_t v5140 = a205;	// L5551
    int8_t v5141 = w205;	// L5552
    int16_t v5142 = v5140;	// L5553
    int16_t v5143 = v5141;	// L5554
    int16_t v5144 = v5142 * v5143;	// L5555
    ap_int<33> v5145 = v5139;	// L5556
    ap_int<33> v5146 = v5144;	// L5557
    ap_int<33> v5147 = v5145 + v5146;	// L5558
    int32_t v5148 = v5147;	// L5559
    v5128.write(v5148); // v5128[m205] = v5148;	// L5560
    int8_t v5149 = a205;	// L5561
    v5127.write(v5149); // v5127[m205] = v5149;	// L5562
  }
}

void PE_kernel_ws_gemm_14_12(
  hls::stream< int8_t > &v5150 /* v5150[16] */,
  hls::stream< int32_t > &v5151 /* v5151[16] */,
  hls::stream< int8_t > &v5152 /* v5152[16] */,
  hls::stream< int32_t > &v5153 /* v5153[16] */,
  int8_t v5154[16][16],
  int v5155,
  int v5156
) {	// L5566
  #pragma HLS stream variable=v5150 depth=17
  #pragma HLS stream variable=v5151 depth=17
  #pragma HLS stream variable=v5152 depth=17
  #pragma HLS stream variable=v5153 depth=17
  #pragma HLS array_partition variable=v5154 complete dim=1
  #pragma HLS array_partition variable=v5154 complete dim=2

  int8_t v5157 = v5154[v5155][v5156];	// L5567
  int8_t w206;	// L5568
  w206 = v5157;	// L5569
  l_step_m206: for (int m206 = 0; m206 < 16; m206++) {	// L5570
  #pragma HLS pipeline II=1
    int8_t v5160 = v5150.read(); // v5150[m206];	// L5571
    int8_t a206;	// L5572
    a206 = v5160;	// L5573
    int32_t v5162 = v5151.read(); // v5151[m206];	// L5574
    int32_t p206;	// L5575
    p206 = v5162;	// L5576
    int32_t v5164 = p206;	// L5577
    int8_t v5165 = a206;	// L5578
    int8_t v5166 = w206;	// L5579
    int16_t v5167 = v5165;	// L5580
    int16_t v5168 = v5166;	// L5581
    int16_t v5169 = v5167 * v5168;	// L5582
    ap_int<33> v5170 = v5164;	// L5583
    ap_int<33> v5171 = v5169;	// L5584
    ap_int<33> v5172 = v5170 + v5171;	// L5585
    int32_t v5173 = v5172;	// L5586
    v5153.write(v5173); // v5153[m206] = v5173;	// L5587
    int8_t v5174 = a206;	// L5588
    v5152.write(v5174); // v5152[m206] = v5174;	// L5589
  }
}

void PE_kernel_ws_gemm_15_12(
  hls::stream< int8_t > &v5175 /* v5175[16] */,
  hls::stream< int32_t > &v5176 /* v5176[16] */,
  hls::stream< int8_t > &v5177 /* v5177[16] */,
  hls::stream< int32_t > &v5178 /* v5178[16] */,
  int8_t v5179[16][16],
  int v5180,
  int v5181
) {	// L5593
  #pragma HLS stream variable=v5175 depth=17
  #pragma HLS stream variable=v5176 depth=17
  #pragma HLS stream variable=v5177 depth=17
  #pragma HLS stream variable=v5178 depth=17
  #pragma HLS array_partition variable=v5179 complete dim=1
  #pragma HLS array_partition variable=v5179 complete dim=2

  int8_t v5182 = v5179[v5180][v5181];	// L5594
  int8_t w207;	// L5595
  w207 = v5182;	// L5596
  l_step_m207: for (int m207 = 0; m207 < 16; m207++) {	// L5597
  #pragma HLS pipeline II=1
    int8_t v5185 = v5175.read(); // v5175[m207];	// L5598
    int8_t a207;	// L5599
    a207 = v5185;	// L5600
    int32_t v5187 = v5176.read(); // v5176[m207];	// L5601
    int32_t p207;	// L5602
    p207 = v5187;	// L5603
    int32_t v5189 = p207;	// L5604
    int8_t v5190 = a207;	// L5605
    int8_t v5191 = w207;	// L5606
    int16_t v5192 = v5190;	// L5607
    int16_t v5193 = v5191;	// L5608
    int16_t v5194 = v5192 * v5193;	// L5609
    ap_int<33> v5195 = v5189;	// L5610
    ap_int<33> v5196 = v5194;	// L5611
    ap_int<33> v5197 = v5195 + v5196;	// L5612
    int32_t v5198 = v5197;	// L5613
    v5178.write(v5198); // v5178[m207] = v5198;	// L5614
    int8_t v5199 = a207;	// L5615
    v5177.write(v5199); // v5177[m207] = v5199;	// L5616
  }
}

void PE_kernel_ws_gemm_0_13(
  hls::stream< int8_t > &v5200 /* v5200[16] */,
  hls::stream< int32_t > &v5201 /* v5201[16] */,
  hls::stream< int8_t > &v5202 /* v5202[16] */,
  hls::stream< int32_t > &v5203 /* v5203[16] */,
  int8_t v5204[16][16],
  int v5205,
  int v5206
) {	// L5620
  #pragma HLS stream variable=v5200 depth=17
  #pragma HLS stream variable=v5201 depth=17
  #pragma HLS stream variable=v5202 depth=17
  #pragma HLS stream variable=v5203 depth=17
  #pragma HLS array_partition variable=v5204 complete dim=1
  #pragma HLS array_partition variable=v5204 complete dim=2

  int8_t v5207 = v5204[v5205][v5206];	// L5621
  int8_t w208;	// L5622
  w208 = v5207;	// L5623
  l_step_m208: for (int m208 = 0; m208 < 16; m208++) {	// L5624
  #pragma HLS pipeline II=1
    int8_t v5210 = v5200.read(); // v5200[m208];	// L5625
    int8_t a208;	// L5626
    a208 = v5210;	// L5627
    int32_t v5212 = v5201.read(); // v5201[m208];	// L5628
    int32_t p208;	// L5629
    p208 = v5212;	// L5630
    int32_t v5214 = p208;	// L5631
    int8_t v5215 = a208;	// L5632
    int8_t v5216 = w208;	// L5633
    int16_t v5217 = v5215;	// L5634
    int16_t v5218 = v5216;	// L5635
    int16_t v5219 = v5217 * v5218;	// L5636
    ap_int<33> v5220 = v5214;	// L5637
    ap_int<33> v5221 = v5219;	// L5638
    ap_int<33> v5222 = v5220 + v5221;	// L5639
    int32_t v5223 = v5222;	// L5640
    v5203.write(v5223); // v5203[m208] = v5223;	// L5641
    int8_t v5224 = a208;	// L5642
    v5202.write(v5224); // v5202[m208] = v5224;	// L5643
  }
}

void PE_kernel_ws_gemm_1_13(
  hls::stream< int8_t > &v5225 /* v5225[16] */,
  hls::stream< int32_t > &v5226 /* v5226[16] */,
  hls::stream< int8_t > &v5227 /* v5227[16] */,
  hls::stream< int32_t > &v5228 /* v5228[16] */,
  int8_t v5229[16][16],
  int v5230,
  int v5231
) {	// L5647
  #pragma HLS stream variable=v5225 depth=17
  #pragma HLS stream variable=v5226 depth=17
  #pragma HLS stream variable=v5227 depth=17
  #pragma HLS stream variable=v5228 depth=17
  #pragma HLS array_partition variable=v5229 complete dim=1
  #pragma HLS array_partition variable=v5229 complete dim=2

  int8_t v5232 = v5229[v5230][v5231];	// L5648
  int8_t w209;	// L5649
  w209 = v5232;	// L5650
  l_step_m209: for (int m209 = 0; m209 < 16; m209++) {	// L5651
  #pragma HLS pipeline II=1
    int8_t v5235 = v5225.read(); // v5225[m209];	// L5652
    int8_t a209;	// L5653
    a209 = v5235;	// L5654
    int32_t v5237 = v5226.read(); // v5226[m209];	// L5655
    int32_t p209;	// L5656
    p209 = v5237;	// L5657
    int32_t v5239 = p209;	// L5658
    int8_t v5240 = a209;	// L5659
    int8_t v5241 = w209;	// L5660
    int16_t v5242 = v5240;	// L5661
    int16_t v5243 = v5241;	// L5662
    int16_t v5244 = v5242 * v5243;	// L5663
    ap_int<33> v5245 = v5239;	// L5664
    ap_int<33> v5246 = v5244;	// L5665
    ap_int<33> v5247 = v5245 + v5246;	// L5666
    int32_t v5248 = v5247;	// L5667
    v5228.write(v5248); // v5228[m209] = v5248;	// L5668
    int8_t v5249 = a209;	// L5669
    v5227.write(v5249); // v5227[m209] = v5249;	// L5670
  }
}

void PE_kernel_ws_gemm_2_13(
  hls::stream< int8_t > &v5250 /* v5250[16] */,
  hls::stream< int32_t > &v5251 /* v5251[16] */,
  hls::stream< int8_t > &v5252 /* v5252[16] */,
  hls::stream< int32_t > &v5253 /* v5253[16] */,
  int8_t v5254[16][16],
  int v5255,
  int v5256
) {	// L5674
  #pragma HLS stream variable=v5250 depth=17
  #pragma HLS stream variable=v5251 depth=17
  #pragma HLS stream variable=v5252 depth=17
  #pragma HLS stream variable=v5253 depth=17
  #pragma HLS array_partition variable=v5254 complete dim=1
  #pragma HLS array_partition variable=v5254 complete dim=2

  int8_t v5257 = v5254[v5255][v5256];	// L5675
  int8_t w210;	// L5676
  w210 = v5257;	// L5677
  l_step_m210: for (int m210 = 0; m210 < 16; m210++) {	// L5678
  #pragma HLS pipeline II=1
    int8_t v5260 = v5250.read(); // v5250[m210];	// L5679
    int8_t a210;	// L5680
    a210 = v5260;	// L5681
    int32_t v5262 = v5251.read(); // v5251[m210];	// L5682
    int32_t p210;	// L5683
    p210 = v5262;	// L5684
    int32_t v5264 = p210;	// L5685
    int8_t v5265 = a210;	// L5686
    int8_t v5266 = w210;	// L5687
    int16_t v5267 = v5265;	// L5688
    int16_t v5268 = v5266;	// L5689
    int16_t v5269 = v5267 * v5268;	// L5690
    ap_int<33> v5270 = v5264;	// L5691
    ap_int<33> v5271 = v5269;	// L5692
    ap_int<33> v5272 = v5270 + v5271;	// L5693
    int32_t v5273 = v5272;	// L5694
    v5253.write(v5273); // v5253[m210] = v5273;	// L5695
    int8_t v5274 = a210;	// L5696
    v5252.write(v5274); // v5252[m210] = v5274;	// L5697
  }
}

void PE_kernel_ws_gemm_3_13(
  hls::stream< int8_t > &v5275 /* v5275[16] */,
  hls::stream< int32_t > &v5276 /* v5276[16] */,
  hls::stream< int8_t > &v5277 /* v5277[16] */,
  hls::stream< int32_t > &v5278 /* v5278[16] */,
  int8_t v5279[16][16],
  int v5280,
  int v5281
) {	// L5701
  #pragma HLS stream variable=v5275 depth=17
  #pragma HLS stream variable=v5276 depth=17
  #pragma HLS stream variable=v5277 depth=17
  #pragma HLS stream variable=v5278 depth=17
  #pragma HLS array_partition variable=v5279 complete dim=1
  #pragma HLS array_partition variable=v5279 complete dim=2

  int8_t v5282 = v5279[v5280][v5281];	// L5702
  int8_t w211;	// L5703
  w211 = v5282;	// L5704
  l_step_m211: for (int m211 = 0; m211 < 16; m211++) {	// L5705
  #pragma HLS pipeline II=1
    int8_t v5285 = v5275.read(); // v5275[m211];	// L5706
    int8_t a211;	// L5707
    a211 = v5285;	// L5708
    int32_t v5287 = v5276.read(); // v5276[m211];	// L5709
    int32_t p211;	// L5710
    p211 = v5287;	// L5711
    int32_t v5289 = p211;	// L5712
    int8_t v5290 = a211;	// L5713
    int8_t v5291 = w211;	// L5714
    int16_t v5292 = v5290;	// L5715
    int16_t v5293 = v5291;	// L5716
    int16_t v5294 = v5292 * v5293;	// L5717
    ap_int<33> v5295 = v5289;	// L5718
    ap_int<33> v5296 = v5294;	// L5719
    ap_int<33> v5297 = v5295 + v5296;	// L5720
    int32_t v5298 = v5297;	// L5721
    v5278.write(v5298); // v5278[m211] = v5298;	// L5722
    int8_t v5299 = a211;	// L5723
    v5277.write(v5299); // v5277[m211] = v5299;	// L5724
  }
}

void PE_kernel_ws_gemm_4_13(
  hls::stream< int8_t > &v5300 /* v5300[16] */,
  hls::stream< int32_t > &v5301 /* v5301[16] */,
  hls::stream< int8_t > &v5302 /* v5302[16] */,
  hls::stream< int32_t > &v5303 /* v5303[16] */,
  int8_t v5304[16][16],
  int v5305,
  int v5306
) {	// L5728
  #pragma HLS stream variable=v5300 depth=17
  #pragma HLS stream variable=v5301 depth=17
  #pragma HLS stream variable=v5302 depth=17
  #pragma HLS stream variable=v5303 depth=17
  #pragma HLS array_partition variable=v5304 complete dim=1
  #pragma HLS array_partition variable=v5304 complete dim=2

  int8_t v5307 = v5304[v5305][v5306];	// L5729
  int8_t w212;	// L5730
  w212 = v5307;	// L5731
  l_step_m212: for (int m212 = 0; m212 < 16; m212++) {	// L5732
  #pragma HLS pipeline II=1
    int8_t v5310 = v5300.read(); // v5300[m212];	// L5733
    int8_t a212;	// L5734
    a212 = v5310;	// L5735
    int32_t v5312 = v5301.read(); // v5301[m212];	// L5736
    int32_t p212;	// L5737
    p212 = v5312;	// L5738
    int32_t v5314 = p212;	// L5739
    int8_t v5315 = a212;	// L5740
    int8_t v5316 = w212;	// L5741
    int16_t v5317 = v5315;	// L5742
    int16_t v5318 = v5316;	// L5743
    int16_t v5319 = v5317 * v5318;	// L5744
    ap_int<33> v5320 = v5314;	// L5745
    ap_int<33> v5321 = v5319;	// L5746
    ap_int<33> v5322 = v5320 + v5321;	// L5747
    int32_t v5323 = v5322;	// L5748
    v5303.write(v5323); // v5303[m212] = v5323;	// L5749
    int8_t v5324 = a212;	// L5750
    v5302.write(v5324); // v5302[m212] = v5324;	// L5751
  }
}

void PE_kernel_ws_gemm_5_13(
  hls::stream< int8_t > &v5325 /* v5325[16] */,
  hls::stream< int32_t > &v5326 /* v5326[16] */,
  hls::stream< int8_t > &v5327 /* v5327[16] */,
  hls::stream< int32_t > &v5328 /* v5328[16] */,
  int8_t v5329[16][16],
  int v5330,
  int v5331
) {	// L5755
  #pragma HLS stream variable=v5325 depth=17
  #pragma HLS stream variable=v5326 depth=17
  #pragma HLS stream variable=v5327 depth=17
  #pragma HLS stream variable=v5328 depth=17
  #pragma HLS array_partition variable=v5329 complete dim=1
  #pragma HLS array_partition variable=v5329 complete dim=2

  int8_t v5332 = v5329[v5330][v5331];	// L5756
  int8_t w213;	// L5757
  w213 = v5332;	// L5758
  l_step_m213: for (int m213 = 0; m213 < 16; m213++) {	// L5759
  #pragma HLS pipeline II=1
    int8_t v5335 = v5325.read(); // v5325[m213];	// L5760
    int8_t a213;	// L5761
    a213 = v5335;	// L5762
    int32_t v5337 = v5326.read(); // v5326[m213];	// L5763
    int32_t p213;	// L5764
    p213 = v5337;	// L5765
    int32_t v5339 = p213;	// L5766
    int8_t v5340 = a213;	// L5767
    int8_t v5341 = w213;	// L5768
    int16_t v5342 = v5340;	// L5769
    int16_t v5343 = v5341;	// L5770
    int16_t v5344 = v5342 * v5343;	// L5771
    ap_int<33> v5345 = v5339;	// L5772
    ap_int<33> v5346 = v5344;	// L5773
    ap_int<33> v5347 = v5345 + v5346;	// L5774
    int32_t v5348 = v5347;	// L5775
    v5328.write(v5348); // v5328[m213] = v5348;	// L5776
    int8_t v5349 = a213;	// L5777
    v5327.write(v5349); // v5327[m213] = v5349;	// L5778
  }
}

void PE_kernel_ws_gemm_6_13(
  hls::stream< int8_t > &v5350 /* v5350[16] */,
  hls::stream< int32_t > &v5351 /* v5351[16] */,
  hls::stream< int8_t > &v5352 /* v5352[16] */,
  hls::stream< int32_t > &v5353 /* v5353[16] */,
  int8_t v5354[16][16],
  int v5355,
  int v5356
) {	// L5782
  #pragma HLS stream variable=v5350 depth=17
  #pragma HLS stream variable=v5351 depth=17
  #pragma HLS stream variable=v5352 depth=17
  #pragma HLS stream variable=v5353 depth=17
  #pragma HLS array_partition variable=v5354 complete dim=1
  #pragma HLS array_partition variable=v5354 complete dim=2

  int8_t v5357 = v5354[v5355][v5356];	// L5783
  int8_t w214;	// L5784
  w214 = v5357;	// L5785
  l_step_m214: for (int m214 = 0; m214 < 16; m214++) {	// L5786
  #pragma HLS pipeline II=1
    int8_t v5360 = v5350.read(); // v5350[m214];	// L5787
    int8_t a214;	// L5788
    a214 = v5360;	// L5789
    int32_t v5362 = v5351.read(); // v5351[m214];	// L5790
    int32_t p214;	// L5791
    p214 = v5362;	// L5792
    int32_t v5364 = p214;	// L5793
    int8_t v5365 = a214;	// L5794
    int8_t v5366 = w214;	// L5795
    int16_t v5367 = v5365;	// L5796
    int16_t v5368 = v5366;	// L5797
    int16_t v5369 = v5367 * v5368;	// L5798
    ap_int<33> v5370 = v5364;	// L5799
    ap_int<33> v5371 = v5369;	// L5800
    ap_int<33> v5372 = v5370 + v5371;	// L5801
    int32_t v5373 = v5372;	// L5802
    v5353.write(v5373); // v5353[m214] = v5373;	// L5803
    int8_t v5374 = a214;	// L5804
    v5352.write(v5374); // v5352[m214] = v5374;	// L5805
  }
}

void PE_kernel_ws_gemm_7_13(
  hls::stream< int8_t > &v5375 /* v5375[16] */,
  hls::stream< int32_t > &v5376 /* v5376[16] */,
  hls::stream< int8_t > &v5377 /* v5377[16] */,
  hls::stream< int32_t > &v5378 /* v5378[16] */,
  int8_t v5379[16][16],
  int v5380,
  int v5381
) {	// L5809
  #pragma HLS stream variable=v5375 depth=17
  #pragma HLS stream variable=v5376 depth=17
  #pragma HLS stream variable=v5377 depth=17
  #pragma HLS stream variable=v5378 depth=17
  #pragma HLS array_partition variable=v5379 complete dim=1
  #pragma HLS array_partition variable=v5379 complete dim=2

  int8_t v5382 = v5379[v5380][v5381];	// L5810
  int8_t w215;	// L5811
  w215 = v5382;	// L5812
  l_step_m215: for (int m215 = 0; m215 < 16; m215++) {	// L5813
  #pragma HLS pipeline II=1
    int8_t v5385 = v5375.read(); // v5375[m215];	// L5814
    int8_t a215;	// L5815
    a215 = v5385;	// L5816
    int32_t v5387 = v5376.read(); // v5376[m215];	// L5817
    int32_t p215;	// L5818
    p215 = v5387;	// L5819
    int32_t v5389 = p215;	// L5820
    int8_t v5390 = a215;	// L5821
    int8_t v5391 = w215;	// L5822
    int16_t v5392 = v5390;	// L5823
    int16_t v5393 = v5391;	// L5824
    int16_t v5394 = v5392 * v5393;	// L5825
    ap_int<33> v5395 = v5389;	// L5826
    ap_int<33> v5396 = v5394;	// L5827
    ap_int<33> v5397 = v5395 + v5396;	// L5828
    int32_t v5398 = v5397;	// L5829
    v5378.write(v5398); // v5378[m215] = v5398;	// L5830
    int8_t v5399 = a215;	// L5831
    v5377.write(v5399); // v5377[m215] = v5399;	// L5832
  }
}

void PE_kernel_ws_gemm_8_13(
  hls::stream< int8_t > &v5400 /* v5400[16] */,
  hls::stream< int32_t > &v5401 /* v5401[16] */,
  hls::stream< int8_t > &v5402 /* v5402[16] */,
  hls::stream< int32_t > &v5403 /* v5403[16] */,
  int8_t v5404[16][16],
  int v5405,
  int v5406
) {	// L5836
  #pragma HLS stream variable=v5400 depth=17
  #pragma HLS stream variable=v5401 depth=17
  #pragma HLS stream variable=v5402 depth=17
  #pragma HLS stream variable=v5403 depth=17
  #pragma HLS array_partition variable=v5404 complete dim=1
  #pragma HLS array_partition variable=v5404 complete dim=2

  int8_t v5407 = v5404[v5405][v5406];	// L5837
  int8_t w216;	// L5838
  w216 = v5407;	// L5839
  l_step_m216: for (int m216 = 0; m216 < 16; m216++) {	// L5840
  #pragma HLS pipeline II=1
    int8_t v5410 = v5400.read(); // v5400[m216];	// L5841
    int8_t a216;	// L5842
    a216 = v5410;	// L5843
    int32_t v5412 = v5401.read(); // v5401[m216];	// L5844
    int32_t p216;	// L5845
    p216 = v5412;	// L5846
    int32_t v5414 = p216;	// L5847
    int8_t v5415 = a216;	// L5848
    int8_t v5416 = w216;	// L5849
    int16_t v5417 = v5415;	// L5850
    int16_t v5418 = v5416;	// L5851
    int16_t v5419 = v5417 * v5418;	// L5852
    ap_int<33> v5420 = v5414;	// L5853
    ap_int<33> v5421 = v5419;	// L5854
    ap_int<33> v5422 = v5420 + v5421;	// L5855
    int32_t v5423 = v5422;	// L5856
    v5403.write(v5423); // v5403[m216] = v5423;	// L5857
    int8_t v5424 = a216;	// L5858
    v5402.write(v5424); // v5402[m216] = v5424;	// L5859
  }
}

void PE_kernel_ws_gemm_9_13(
  hls::stream< int8_t > &v5425 /* v5425[16] */,
  hls::stream< int32_t > &v5426 /* v5426[16] */,
  hls::stream< int8_t > &v5427 /* v5427[16] */,
  hls::stream< int32_t > &v5428 /* v5428[16] */,
  int8_t v5429[16][16],
  int v5430,
  int v5431
) {	// L5863
  #pragma HLS stream variable=v5425 depth=17
  #pragma HLS stream variable=v5426 depth=17
  #pragma HLS stream variable=v5427 depth=17
  #pragma HLS stream variable=v5428 depth=17
  #pragma HLS array_partition variable=v5429 complete dim=1
  #pragma HLS array_partition variable=v5429 complete dim=2

  int8_t v5432 = v5429[v5430][v5431];	// L5864
  int8_t w217;	// L5865
  w217 = v5432;	// L5866
  l_step_m217: for (int m217 = 0; m217 < 16; m217++) {	// L5867
  #pragma HLS pipeline II=1
    int8_t v5435 = v5425.read(); // v5425[m217];	// L5868
    int8_t a217;	// L5869
    a217 = v5435;	// L5870
    int32_t v5437 = v5426.read(); // v5426[m217];	// L5871
    int32_t p217;	// L5872
    p217 = v5437;	// L5873
    int32_t v5439 = p217;	// L5874
    int8_t v5440 = a217;	// L5875
    int8_t v5441 = w217;	// L5876
    int16_t v5442 = v5440;	// L5877
    int16_t v5443 = v5441;	// L5878
    int16_t v5444 = v5442 * v5443;	// L5879
    ap_int<33> v5445 = v5439;	// L5880
    ap_int<33> v5446 = v5444;	// L5881
    ap_int<33> v5447 = v5445 + v5446;	// L5882
    int32_t v5448 = v5447;	// L5883
    v5428.write(v5448); // v5428[m217] = v5448;	// L5884
    int8_t v5449 = a217;	// L5885
    v5427.write(v5449); // v5427[m217] = v5449;	// L5886
  }
}

void PE_kernel_ws_gemm_10_13(
  hls::stream< int8_t > &v5450 /* v5450[16] */,
  hls::stream< int32_t > &v5451 /* v5451[16] */,
  hls::stream< int8_t > &v5452 /* v5452[16] */,
  hls::stream< int32_t > &v5453 /* v5453[16] */,
  int8_t v5454[16][16],
  int v5455,
  int v5456
) {	// L5890
  #pragma HLS stream variable=v5450 depth=17
  #pragma HLS stream variable=v5451 depth=17
  #pragma HLS stream variable=v5452 depth=17
  #pragma HLS stream variable=v5453 depth=17
  #pragma HLS array_partition variable=v5454 complete dim=1
  #pragma HLS array_partition variable=v5454 complete dim=2

  int8_t v5457 = v5454[v5455][v5456];	// L5891
  int8_t w218;	// L5892
  w218 = v5457;	// L5893
  l_step_m218: for (int m218 = 0; m218 < 16; m218++) {	// L5894
  #pragma HLS pipeline II=1
    int8_t v5460 = v5450.read(); // v5450[m218];	// L5895
    int8_t a218;	// L5896
    a218 = v5460;	// L5897
    int32_t v5462 = v5451.read(); // v5451[m218];	// L5898
    int32_t p218;	// L5899
    p218 = v5462;	// L5900
    int32_t v5464 = p218;	// L5901
    int8_t v5465 = a218;	// L5902
    int8_t v5466 = w218;	// L5903
    int16_t v5467 = v5465;	// L5904
    int16_t v5468 = v5466;	// L5905
    int16_t v5469 = v5467 * v5468;	// L5906
    ap_int<33> v5470 = v5464;	// L5907
    ap_int<33> v5471 = v5469;	// L5908
    ap_int<33> v5472 = v5470 + v5471;	// L5909
    int32_t v5473 = v5472;	// L5910
    v5453.write(v5473); // v5453[m218] = v5473;	// L5911
    int8_t v5474 = a218;	// L5912
    v5452.write(v5474); // v5452[m218] = v5474;	// L5913
  }
}

void PE_kernel_ws_gemm_11_13(
  hls::stream< int8_t > &v5475 /* v5475[16] */,
  hls::stream< int32_t > &v5476 /* v5476[16] */,
  hls::stream< int8_t > &v5477 /* v5477[16] */,
  hls::stream< int32_t > &v5478 /* v5478[16] */,
  int8_t v5479[16][16],
  int v5480,
  int v5481
) {	// L5917
  #pragma HLS stream variable=v5475 depth=17
  #pragma HLS stream variable=v5476 depth=17
  #pragma HLS stream variable=v5477 depth=17
  #pragma HLS stream variable=v5478 depth=17
  #pragma HLS array_partition variable=v5479 complete dim=1
  #pragma HLS array_partition variable=v5479 complete dim=2

  int8_t v5482 = v5479[v5480][v5481];	// L5918
  int8_t w219;	// L5919
  w219 = v5482;	// L5920
  l_step_m219: for (int m219 = 0; m219 < 16; m219++) {	// L5921
  #pragma HLS pipeline II=1
    int8_t v5485 = v5475.read(); // v5475[m219];	// L5922
    int8_t a219;	// L5923
    a219 = v5485;	// L5924
    int32_t v5487 = v5476.read(); // v5476[m219];	// L5925
    int32_t p219;	// L5926
    p219 = v5487;	// L5927
    int32_t v5489 = p219;	// L5928
    int8_t v5490 = a219;	// L5929
    int8_t v5491 = w219;	// L5930
    int16_t v5492 = v5490;	// L5931
    int16_t v5493 = v5491;	// L5932
    int16_t v5494 = v5492 * v5493;	// L5933
    ap_int<33> v5495 = v5489;	// L5934
    ap_int<33> v5496 = v5494;	// L5935
    ap_int<33> v5497 = v5495 + v5496;	// L5936
    int32_t v5498 = v5497;	// L5937
    v5478.write(v5498); // v5478[m219] = v5498;	// L5938
    int8_t v5499 = a219;	// L5939
    v5477.write(v5499); // v5477[m219] = v5499;	// L5940
  }
}

void PE_kernel_ws_gemm_12_13(
  hls::stream< int8_t > &v5500 /* v5500[16] */,
  hls::stream< int32_t > &v5501 /* v5501[16] */,
  hls::stream< int8_t > &v5502 /* v5502[16] */,
  hls::stream< int32_t > &v5503 /* v5503[16] */,
  int8_t v5504[16][16],
  int v5505,
  int v5506
) {	// L5944
  #pragma HLS stream variable=v5500 depth=17
  #pragma HLS stream variable=v5501 depth=17
  #pragma HLS stream variable=v5502 depth=17
  #pragma HLS stream variable=v5503 depth=17
  #pragma HLS array_partition variable=v5504 complete dim=1
  #pragma HLS array_partition variable=v5504 complete dim=2

  int8_t v5507 = v5504[v5505][v5506];	// L5945
  int8_t w220;	// L5946
  w220 = v5507;	// L5947
  l_step_m220: for (int m220 = 0; m220 < 16; m220++) {	// L5948
  #pragma HLS pipeline II=1
    int8_t v5510 = v5500.read(); // v5500[m220];	// L5949
    int8_t a220;	// L5950
    a220 = v5510;	// L5951
    int32_t v5512 = v5501.read(); // v5501[m220];	// L5952
    int32_t p220;	// L5953
    p220 = v5512;	// L5954
    int32_t v5514 = p220;	// L5955
    int8_t v5515 = a220;	// L5956
    int8_t v5516 = w220;	// L5957
    int16_t v5517 = v5515;	// L5958
    int16_t v5518 = v5516;	// L5959
    int16_t v5519 = v5517 * v5518;	// L5960
    ap_int<33> v5520 = v5514;	// L5961
    ap_int<33> v5521 = v5519;	// L5962
    ap_int<33> v5522 = v5520 + v5521;	// L5963
    int32_t v5523 = v5522;	// L5964
    v5503.write(v5523); // v5503[m220] = v5523;	// L5965
    int8_t v5524 = a220;	// L5966
    v5502.write(v5524); // v5502[m220] = v5524;	// L5967
  }
}

void PE_kernel_ws_gemm_13_13(
  hls::stream< int8_t > &v5525 /* v5525[16] */,
  hls::stream< int32_t > &v5526 /* v5526[16] */,
  hls::stream< int8_t > &v5527 /* v5527[16] */,
  hls::stream< int32_t > &v5528 /* v5528[16] */,
  int8_t v5529[16][16],
  int v5530,
  int v5531
) {	// L5971
  #pragma HLS stream variable=v5525 depth=17
  #pragma HLS stream variable=v5526 depth=17
  #pragma HLS stream variable=v5527 depth=17
  #pragma HLS stream variable=v5528 depth=17
  #pragma HLS array_partition variable=v5529 complete dim=1
  #pragma HLS array_partition variable=v5529 complete dim=2

  int8_t v5532 = v5529[v5530][v5531];	// L5972
  int8_t w221;	// L5973
  w221 = v5532;	// L5974
  l_step_m221: for (int m221 = 0; m221 < 16; m221++) {	// L5975
  #pragma HLS pipeline II=1
    int8_t v5535 = v5525.read(); // v5525[m221];	// L5976
    int8_t a221;	// L5977
    a221 = v5535;	// L5978
    int32_t v5537 = v5526.read(); // v5526[m221];	// L5979
    int32_t p221;	// L5980
    p221 = v5537;	// L5981
    int32_t v5539 = p221;	// L5982
    int8_t v5540 = a221;	// L5983
    int8_t v5541 = w221;	// L5984
    int16_t v5542 = v5540;	// L5985
    int16_t v5543 = v5541;	// L5986
    int16_t v5544 = v5542 * v5543;	// L5987
    ap_int<33> v5545 = v5539;	// L5988
    ap_int<33> v5546 = v5544;	// L5989
    ap_int<33> v5547 = v5545 + v5546;	// L5990
    int32_t v5548 = v5547;	// L5991
    v5528.write(v5548); // v5528[m221] = v5548;	// L5992
    int8_t v5549 = a221;	// L5993
    v5527.write(v5549); // v5527[m221] = v5549;	// L5994
  }
}

void PE_kernel_ws_gemm_14_13(
  hls::stream< int8_t > &v5550 /* v5550[16] */,
  hls::stream< int32_t > &v5551 /* v5551[16] */,
  hls::stream< int8_t > &v5552 /* v5552[16] */,
  hls::stream< int32_t > &v5553 /* v5553[16] */,
  int8_t v5554[16][16],
  int v5555,
  int v5556
) {	// L5998
  #pragma HLS stream variable=v5550 depth=17
  #pragma HLS stream variable=v5551 depth=17
  #pragma HLS stream variable=v5552 depth=17
  #pragma HLS stream variable=v5553 depth=17
  #pragma HLS array_partition variable=v5554 complete dim=1
  #pragma HLS array_partition variable=v5554 complete dim=2

  int8_t v5557 = v5554[v5555][v5556];	// L5999
  int8_t w222;	// L6000
  w222 = v5557;	// L6001
  l_step_m222: for (int m222 = 0; m222 < 16; m222++) {	// L6002
  #pragma HLS pipeline II=1
    int8_t v5560 = v5550.read(); // v5550[m222];	// L6003
    int8_t a222;	// L6004
    a222 = v5560;	// L6005
    int32_t v5562 = v5551.read(); // v5551[m222];	// L6006
    int32_t p222;	// L6007
    p222 = v5562;	// L6008
    int32_t v5564 = p222;	// L6009
    int8_t v5565 = a222;	// L6010
    int8_t v5566 = w222;	// L6011
    int16_t v5567 = v5565;	// L6012
    int16_t v5568 = v5566;	// L6013
    int16_t v5569 = v5567 * v5568;	// L6014
    ap_int<33> v5570 = v5564;	// L6015
    ap_int<33> v5571 = v5569;	// L6016
    ap_int<33> v5572 = v5570 + v5571;	// L6017
    int32_t v5573 = v5572;	// L6018
    v5553.write(v5573); // v5553[m222] = v5573;	// L6019
    int8_t v5574 = a222;	// L6020
    v5552.write(v5574); // v5552[m222] = v5574;	// L6021
  }
}

void PE_kernel_ws_gemm_15_13(
  hls::stream< int8_t > &v5575 /* v5575[16] */,
  hls::stream< int32_t > &v5576 /* v5576[16] */,
  hls::stream< int8_t > &v5577 /* v5577[16] */,
  hls::stream< int32_t > &v5578 /* v5578[16] */,
  int8_t v5579[16][16],
  int v5580,
  int v5581
) {	// L6025
  #pragma HLS stream variable=v5575 depth=17
  #pragma HLS stream variable=v5576 depth=17
  #pragma HLS stream variable=v5577 depth=17
  #pragma HLS stream variable=v5578 depth=17
  #pragma HLS array_partition variable=v5579 complete dim=1
  #pragma HLS array_partition variable=v5579 complete dim=2

  int8_t v5582 = v5579[v5580][v5581];	// L6026
  int8_t w223;	// L6027
  w223 = v5582;	// L6028
  l_step_m223: for (int m223 = 0; m223 < 16; m223++) {	// L6029
  #pragma HLS pipeline II=1
    int8_t v5585 = v5575.read(); // v5575[m223];	// L6030
    int8_t a223;	// L6031
    a223 = v5585;	// L6032
    int32_t v5587 = v5576.read(); // v5576[m223];	// L6033
    int32_t p223;	// L6034
    p223 = v5587;	// L6035
    int32_t v5589 = p223;	// L6036
    int8_t v5590 = a223;	// L6037
    int8_t v5591 = w223;	// L6038
    int16_t v5592 = v5590;	// L6039
    int16_t v5593 = v5591;	// L6040
    int16_t v5594 = v5592 * v5593;	// L6041
    ap_int<33> v5595 = v5589;	// L6042
    ap_int<33> v5596 = v5594;	// L6043
    ap_int<33> v5597 = v5595 + v5596;	// L6044
    int32_t v5598 = v5597;	// L6045
    v5578.write(v5598); // v5578[m223] = v5598;	// L6046
    int8_t v5599 = a223;	// L6047
    v5577.write(v5599); // v5577[m223] = v5599;	// L6048
  }
}

void PE_kernel_ws_gemm_0_14(
  hls::stream< int8_t > &v5600 /* v5600[16] */,
  hls::stream< int32_t > &v5601 /* v5601[16] */,
  hls::stream< int8_t > &v5602 /* v5602[16] */,
  hls::stream< int32_t > &v5603 /* v5603[16] */,
  int8_t v5604[16][16],
  int v5605,
  int v5606
) {	// L6052
  #pragma HLS stream variable=v5600 depth=17
  #pragma HLS stream variable=v5601 depth=17
  #pragma HLS stream variable=v5602 depth=17
  #pragma HLS stream variable=v5603 depth=17
  #pragma HLS array_partition variable=v5604 complete dim=1
  #pragma HLS array_partition variable=v5604 complete dim=2

  int8_t v5607 = v5604[v5605][v5606];	// L6053
  int8_t w224;	// L6054
  w224 = v5607;	// L6055
  l_step_m224: for (int m224 = 0; m224 < 16; m224++) {	// L6056
  #pragma HLS pipeline II=1
    int8_t v5610 = v5600.read(); // v5600[m224];	// L6057
    int8_t a224;	// L6058
    a224 = v5610;	// L6059
    int32_t v5612 = v5601.read(); // v5601[m224];	// L6060
    int32_t p224;	// L6061
    p224 = v5612;	// L6062
    int32_t v5614 = p224;	// L6063
    int8_t v5615 = a224;	// L6064
    int8_t v5616 = w224;	// L6065
    int16_t v5617 = v5615;	// L6066
    int16_t v5618 = v5616;	// L6067
    int16_t v5619 = v5617 * v5618;	// L6068
    ap_int<33> v5620 = v5614;	// L6069
    ap_int<33> v5621 = v5619;	// L6070
    ap_int<33> v5622 = v5620 + v5621;	// L6071
    int32_t v5623 = v5622;	// L6072
    v5603.write(v5623); // v5603[m224] = v5623;	// L6073
    int8_t v5624 = a224;	// L6074
    v5602.write(v5624); // v5602[m224] = v5624;	// L6075
  }
}

void PE_kernel_ws_gemm_1_14(
  hls::stream< int8_t > &v5625 /* v5625[16] */,
  hls::stream< int32_t > &v5626 /* v5626[16] */,
  hls::stream< int8_t > &v5627 /* v5627[16] */,
  hls::stream< int32_t > &v5628 /* v5628[16] */,
  int8_t v5629[16][16],
  int v5630,
  int v5631
) {	// L6079
  #pragma HLS stream variable=v5625 depth=17
  #pragma HLS stream variable=v5626 depth=17
  #pragma HLS stream variable=v5627 depth=17
  #pragma HLS stream variable=v5628 depth=17
  #pragma HLS array_partition variable=v5629 complete dim=1
  #pragma HLS array_partition variable=v5629 complete dim=2

  int8_t v5632 = v5629[v5630][v5631];	// L6080
  int8_t w225;	// L6081
  w225 = v5632;	// L6082
  l_step_m225: for (int m225 = 0; m225 < 16; m225++) {	// L6083
  #pragma HLS pipeline II=1
    int8_t v5635 = v5625.read(); // v5625[m225];	// L6084
    int8_t a225;	// L6085
    a225 = v5635;	// L6086
    int32_t v5637 = v5626.read(); // v5626[m225];	// L6087
    int32_t p225;	// L6088
    p225 = v5637;	// L6089
    int32_t v5639 = p225;	// L6090
    int8_t v5640 = a225;	// L6091
    int8_t v5641 = w225;	// L6092
    int16_t v5642 = v5640;	// L6093
    int16_t v5643 = v5641;	// L6094
    int16_t v5644 = v5642 * v5643;	// L6095
    ap_int<33> v5645 = v5639;	// L6096
    ap_int<33> v5646 = v5644;	// L6097
    ap_int<33> v5647 = v5645 + v5646;	// L6098
    int32_t v5648 = v5647;	// L6099
    v5628.write(v5648); // v5628[m225] = v5648;	// L6100
    int8_t v5649 = a225;	// L6101
    v5627.write(v5649); // v5627[m225] = v5649;	// L6102
  }
}

void PE_kernel_ws_gemm_2_14(
  hls::stream< int8_t > &v5650 /* v5650[16] */,
  hls::stream< int32_t > &v5651 /* v5651[16] */,
  hls::stream< int8_t > &v5652 /* v5652[16] */,
  hls::stream< int32_t > &v5653 /* v5653[16] */,
  int8_t v5654[16][16],
  int v5655,
  int v5656
) {	// L6106
  #pragma HLS stream variable=v5650 depth=17
  #pragma HLS stream variable=v5651 depth=17
  #pragma HLS stream variable=v5652 depth=17
  #pragma HLS stream variable=v5653 depth=17
  #pragma HLS array_partition variable=v5654 complete dim=1
  #pragma HLS array_partition variable=v5654 complete dim=2

  int8_t v5657 = v5654[v5655][v5656];	// L6107
  int8_t w226;	// L6108
  w226 = v5657;	// L6109
  l_step_m226: for (int m226 = 0; m226 < 16; m226++) {	// L6110
  #pragma HLS pipeline II=1
    int8_t v5660 = v5650.read(); // v5650[m226];	// L6111
    int8_t a226;	// L6112
    a226 = v5660;	// L6113
    int32_t v5662 = v5651.read(); // v5651[m226];	// L6114
    int32_t p226;	// L6115
    p226 = v5662;	// L6116
    int32_t v5664 = p226;	// L6117
    int8_t v5665 = a226;	// L6118
    int8_t v5666 = w226;	// L6119
    int16_t v5667 = v5665;	// L6120
    int16_t v5668 = v5666;	// L6121
    int16_t v5669 = v5667 * v5668;	// L6122
    ap_int<33> v5670 = v5664;	// L6123
    ap_int<33> v5671 = v5669;	// L6124
    ap_int<33> v5672 = v5670 + v5671;	// L6125
    int32_t v5673 = v5672;	// L6126
    v5653.write(v5673); // v5653[m226] = v5673;	// L6127
    int8_t v5674 = a226;	// L6128
    v5652.write(v5674); // v5652[m226] = v5674;	// L6129
  }
}

void PE_kernel_ws_gemm_3_14(
  hls::stream< int8_t > &v5675 /* v5675[16] */,
  hls::stream< int32_t > &v5676 /* v5676[16] */,
  hls::stream< int8_t > &v5677 /* v5677[16] */,
  hls::stream< int32_t > &v5678 /* v5678[16] */,
  int8_t v5679[16][16],
  int v5680,
  int v5681
) {	// L6133
  #pragma HLS stream variable=v5675 depth=17
  #pragma HLS stream variable=v5676 depth=17
  #pragma HLS stream variable=v5677 depth=17
  #pragma HLS stream variable=v5678 depth=17
  #pragma HLS array_partition variable=v5679 complete dim=1
  #pragma HLS array_partition variable=v5679 complete dim=2

  int8_t v5682 = v5679[v5680][v5681];	// L6134
  int8_t w227;	// L6135
  w227 = v5682;	// L6136
  l_step_m227: for (int m227 = 0; m227 < 16; m227++) {	// L6137
  #pragma HLS pipeline II=1
    int8_t v5685 = v5675.read(); // v5675[m227];	// L6138
    int8_t a227;	// L6139
    a227 = v5685;	// L6140
    int32_t v5687 = v5676.read(); // v5676[m227];	// L6141
    int32_t p227;	// L6142
    p227 = v5687;	// L6143
    int32_t v5689 = p227;	// L6144
    int8_t v5690 = a227;	// L6145
    int8_t v5691 = w227;	// L6146
    int16_t v5692 = v5690;	// L6147
    int16_t v5693 = v5691;	// L6148
    int16_t v5694 = v5692 * v5693;	// L6149
    ap_int<33> v5695 = v5689;	// L6150
    ap_int<33> v5696 = v5694;	// L6151
    ap_int<33> v5697 = v5695 + v5696;	// L6152
    int32_t v5698 = v5697;	// L6153
    v5678.write(v5698); // v5678[m227] = v5698;	// L6154
    int8_t v5699 = a227;	// L6155
    v5677.write(v5699); // v5677[m227] = v5699;	// L6156
  }
}

void PE_kernel_ws_gemm_4_14(
  hls::stream< int8_t > &v5700 /* v5700[16] */,
  hls::stream< int32_t > &v5701 /* v5701[16] */,
  hls::stream< int8_t > &v5702 /* v5702[16] */,
  hls::stream< int32_t > &v5703 /* v5703[16] */,
  int8_t v5704[16][16],
  int v5705,
  int v5706
) {	// L6160
  #pragma HLS stream variable=v5700 depth=17
  #pragma HLS stream variable=v5701 depth=17
  #pragma HLS stream variable=v5702 depth=17
  #pragma HLS stream variable=v5703 depth=17
  #pragma HLS array_partition variable=v5704 complete dim=1
  #pragma HLS array_partition variable=v5704 complete dim=2

  int8_t v5707 = v5704[v5705][v5706];	// L6161
  int8_t w228;	// L6162
  w228 = v5707;	// L6163
  l_step_m228: for (int m228 = 0; m228 < 16; m228++) {	// L6164
  #pragma HLS pipeline II=1
    int8_t v5710 = v5700.read(); // v5700[m228];	// L6165
    int8_t a228;	// L6166
    a228 = v5710;	// L6167
    int32_t v5712 = v5701.read(); // v5701[m228];	// L6168
    int32_t p228;	// L6169
    p228 = v5712;	// L6170
    int32_t v5714 = p228;	// L6171
    int8_t v5715 = a228;	// L6172
    int8_t v5716 = w228;	// L6173
    int16_t v5717 = v5715;	// L6174
    int16_t v5718 = v5716;	// L6175
    int16_t v5719 = v5717 * v5718;	// L6176
    ap_int<33> v5720 = v5714;	// L6177
    ap_int<33> v5721 = v5719;	// L6178
    ap_int<33> v5722 = v5720 + v5721;	// L6179
    int32_t v5723 = v5722;	// L6180
    v5703.write(v5723); // v5703[m228] = v5723;	// L6181
    int8_t v5724 = a228;	// L6182
    v5702.write(v5724); // v5702[m228] = v5724;	// L6183
  }
}

void PE_kernel_ws_gemm_5_14(
  hls::stream< int8_t > &v5725 /* v5725[16] */,
  hls::stream< int32_t > &v5726 /* v5726[16] */,
  hls::stream< int8_t > &v5727 /* v5727[16] */,
  hls::stream< int32_t > &v5728 /* v5728[16] */,
  int8_t v5729[16][16],
  int v5730,
  int v5731
) {	// L6187
  #pragma HLS stream variable=v5725 depth=17
  #pragma HLS stream variable=v5726 depth=17
  #pragma HLS stream variable=v5727 depth=17
  #pragma HLS stream variable=v5728 depth=17
  #pragma HLS array_partition variable=v5729 complete dim=1
  #pragma HLS array_partition variable=v5729 complete dim=2

  int8_t v5732 = v5729[v5730][v5731];	// L6188
  int8_t w229;	// L6189
  w229 = v5732;	// L6190
  l_step_m229: for (int m229 = 0; m229 < 16; m229++) {	// L6191
  #pragma HLS pipeline II=1
    int8_t v5735 = v5725.read(); // v5725[m229];	// L6192
    int8_t a229;	// L6193
    a229 = v5735;	// L6194
    int32_t v5737 = v5726.read(); // v5726[m229];	// L6195
    int32_t p229;	// L6196
    p229 = v5737;	// L6197
    int32_t v5739 = p229;	// L6198
    int8_t v5740 = a229;	// L6199
    int8_t v5741 = w229;	// L6200
    int16_t v5742 = v5740;	// L6201
    int16_t v5743 = v5741;	// L6202
    int16_t v5744 = v5742 * v5743;	// L6203
    ap_int<33> v5745 = v5739;	// L6204
    ap_int<33> v5746 = v5744;	// L6205
    ap_int<33> v5747 = v5745 + v5746;	// L6206
    int32_t v5748 = v5747;	// L6207
    v5728.write(v5748); // v5728[m229] = v5748;	// L6208
    int8_t v5749 = a229;	// L6209
    v5727.write(v5749); // v5727[m229] = v5749;	// L6210
  }
}

void PE_kernel_ws_gemm_6_14(
  hls::stream< int8_t > &v5750 /* v5750[16] */,
  hls::stream< int32_t > &v5751 /* v5751[16] */,
  hls::stream< int8_t > &v5752 /* v5752[16] */,
  hls::stream< int32_t > &v5753 /* v5753[16] */,
  int8_t v5754[16][16],
  int v5755,
  int v5756
) {	// L6214
  #pragma HLS stream variable=v5750 depth=17
  #pragma HLS stream variable=v5751 depth=17
  #pragma HLS stream variable=v5752 depth=17
  #pragma HLS stream variable=v5753 depth=17
  #pragma HLS array_partition variable=v5754 complete dim=1
  #pragma HLS array_partition variable=v5754 complete dim=2

  int8_t v5757 = v5754[v5755][v5756];	// L6215
  int8_t w230;	// L6216
  w230 = v5757;	// L6217
  l_step_m230: for (int m230 = 0; m230 < 16; m230++) {	// L6218
  #pragma HLS pipeline II=1
    int8_t v5760 = v5750.read(); // v5750[m230];	// L6219
    int8_t a230;	// L6220
    a230 = v5760;	// L6221
    int32_t v5762 = v5751.read(); // v5751[m230];	// L6222
    int32_t p230;	// L6223
    p230 = v5762;	// L6224
    int32_t v5764 = p230;	// L6225
    int8_t v5765 = a230;	// L6226
    int8_t v5766 = w230;	// L6227
    int16_t v5767 = v5765;	// L6228
    int16_t v5768 = v5766;	// L6229
    int16_t v5769 = v5767 * v5768;	// L6230
    ap_int<33> v5770 = v5764;	// L6231
    ap_int<33> v5771 = v5769;	// L6232
    ap_int<33> v5772 = v5770 + v5771;	// L6233
    int32_t v5773 = v5772;	// L6234
    v5753.write(v5773); // v5753[m230] = v5773;	// L6235
    int8_t v5774 = a230;	// L6236
    v5752.write(v5774); // v5752[m230] = v5774;	// L6237
  }
}

void PE_kernel_ws_gemm_7_14(
  hls::stream< int8_t > &v5775 /* v5775[16] */,
  hls::stream< int32_t > &v5776 /* v5776[16] */,
  hls::stream< int8_t > &v5777 /* v5777[16] */,
  hls::stream< int32_t > &v5778 /* v5778[16] */,
  int8_t v5779[16][16],
  int v5780,
  int v5781
) {	// L6241
  #pragma HLS stream variable=v5775 depth=17
  #pragma HLS stream variable=v5776 depth=17
  #pragma HLS stream variable=v5777 depth=17
  #pragma HLS stream variable=v5778 depth=17
  #pragma HLS array_partition variable=v5779 complete dim=1
  #pragma HLS array_partition variable=v5779 complete dim=2

  int8_t v5782 = v5779[v5780][v5781];	// L6242
  int8_t w231;	// L6243
  w231 = v5782;	// L6244
  l_step_m231: for (int m231 = 0; m231 < 16; m231++) {	// L6245
  #pragma HLS pipeline II=1
    int8_t v5785 = v5775.read(); // v5775[m231];	// L6246
    int8_t a231;	// L6247
    a231 = v5785;	// L6248
    int32_t v5787 = v5776.read(); // v5776[m231];	// L6249
    int32_t p231;	// L6250
    p231 = v5787;	// L6251
    int32_t v5789 = p231;	// L6252
    int8_t v5790 = a231;	// L6253
    int8_t v5791 = w231;	// L6254
    int16_t v5792 = v5790;	// L6255
    int16_t v5793 = v5791;	// L6256
    int16_t v5794 = v5792 * v5793;	// L6257
    ap_int<33> v5795 = v5789;	// L6258
    ap_int<33> v5796 = v5794;	// L6259
    ap_int<33> v5797 = v5795 + v5796;	// L6260
    int32_t v5798 = v5797;	// L6261
    v5778.write(v5798); // v5778[m231] = v5798;	// L6262
    int8_t v5799 = a231;	// L6263
    v5777.write(v5799); // v5777[m231] = v5799;	// L6264
  }
}

void PE_kernel_ws_gemm_8_14(
  hls::stream< int8_t > &v5800 /* v5800[16] */,
  hls::stream< int32_t > &v5801 /* v5801[16] */,
  hls::stream< int8_t > &v5802 /* v5802[16] */,
  hls::stream< int32_t > &v5803 /* v5803[16] */,
  int8_t v5804[16][16],
  int v5805,
  int v5806
) {	// L6268
  #pragma HLS stream variable=v5800 depth=17
  #pragma HLS stream variable=v5801 depth=17
  #pragma HLS stream variable=v5802 depth=17
  #pragma HLS stream variable=v5803 depth=17
  #pragma HLS array_partition variable=v5804 complete dim=1
  #pragma HLS array_partition variable=v5804 complete dim=2

  int8_t v5807 = v5804[v5805][v5806];	// L6269
  int8_t w232;	// L6270
  w232 = v5807;	// L6271
  l_step_m232: for (int m232 = 0; m232 < 16; m232++) {	// L6272
  #pragma HLS pipeline II=1
    int8_t v5810 = v5800.read(); // v5800[m232];	// L6273
    int8_t a232;	// L6274
    a232 = v5810;	// L6275
    int32_t v5812 = v5801.read(); // v5801[m232];	// L6276
    int32_t p232;	// L6277
    p232 = v5812;	// L6278
    int32_t v5814 = p232;	// L6279
    int8_t v5815 = a232;	// L6280
    int8_t v5816 = w232;	// L6281
    int16_t v5817 = v5815;	// L6282
    int16_t v5818 = v5816;	// L6283
    int16_t v5819 = v5817 * v5818;	// L6284
    ap_int<33> v5820 = v5814;	// L6285
    ap_int<33> v5821 = v5819;	// L6286
    ap_int<33> v5822 = v5820 + v5821;	// L6287
    int32_t v5823 = v5822;	// L6288
    v5803.write(v5823); // v5803[m232] = v5823;	// L6289
    int8_t v5824 = a232;	// L6290
    v5802.write(v5824); // v5802[m232] = v5824;	// L6291
  }
}

void PE_kernel_ws_gemm_9_14(
  hls::stream< int8_t > &v5825 /* v5825[16] */,
  hls::stream< int32_t > &v5826 /* v5826[16] */,
  hls::stream< int8_t > &v5827 /* v5827[16] */,
  hls::stream< int32_t > &v5828 /* v5828[16] */,
  int8_t v5829[16][16],
  int v5830,
  int v5831
) {	// L6295
  #pragma HLS stream variable=v5825 depth=17
  #pragma HLS stream variable=v5826 depth=17
  #pragma HLS stream variable=v5827 depth=17
  #pragma HLS stream variable=v5828 depth=17
  #pragma HLS array_partition variable=v5829 complete dim=1
  #pragma HLS array_partition variable=v5829 complete dim=2

  int8_t v5832 = v5829[v5830][v5831];	// L6296
  int8_t w233;	// L6297
  w233 = v5832;	// L6298
  l_step_m233: for (int m233 = 0; m233 < 16; m233++) {	// L6299
  #pragma HLS pipeline II=1
    int8_t v5835 = v5825.read(); // v5825[m233];	// L6300
    int8_t a233;	// L6301
    a233 = v5835;	// L6302
    int32_t v5837 = v5826.read(); // v5826[m233];	// L6303
    int32_t p233;	// L6304
    p233 = v5837;	// L6305
    int32_t v5839 = p233;	// L6306
    int8_t v5840 = a233;	// L6307
    int8_t v5841 = w233;	// L6308
    int16_t v5842 = v5840;	// L6309
    int16_t v5843 = v5841;	// L6310
    int16_t v5844 = v5842 * v5843;	// L6311
    ap_int<33> v5845 = v5839;	// L6312
    ap_int<33> v5846 = v5844;	// L6313
    ap_int<33> v5847 = v5845 + v5846;	// L6314
    int32_t v5848 = v5847;	// L6315
    v5828.write(v5848); // v5828[m233] = v5848;	// L6316
    int8_t v5849 = a233;	// L6317
    v5827.write(v5849); // v5827[m233] = v5849;	// L6318
  }
}

void PE_kernel_ws_gemm_10_14(
  hls::stream< int8_t > &v5850 /* v5850[16] */,
  hls::stream< int32_t > &v5851 /* v5851[16] */,
  hls::stream< int8_t > &v5852 /* v5852[16] */,
  hls::stream< int32_t > &v5853 /* v5853[16] */,
  int8_t v5854[16][16],
  int v5855,
  int v5856
) {	// L6322
  #pragma HLS stream variable=v5850 depth=17
  #pragma HLS stream variable=v5851 depth=17
  #pragma HLS stream variable=v5852 depth=17
  #pragma HLS stream variable=v5853 depth=17
  #pragma HLS array_partition variable=v5854 complete dim=1
  #pragma HLS array_partition variable=v5854 complete dim=2

  int8_t v5857 = v5854[v5855][v5856];	// L6323
  int8_t w234;	// L6324
  w234 = v5857;	// L6325
  l_step_m234: for (int m234 = 0; m234 < 16; m234++) {	// L6326
  #pragma HLS pipeline II=1
    int8_t v5860 = v5850.read(); // v5850[m234];	// L6327
    int8_t a234;	// L6328
    a234 = v5860;	// L6329
    int32_t v5862 = v5851.read(); // v5851[m234];	// L6330
    int32_t p234;	// L6331
    p234 = v5862;	// L6332
    int32_t v5864 = p234;	// L6333
    int8_t v5865 = a234;	// L6334
    int8_t v5866 = w234;	// L6335
    int16_t v5867 = v5865;	// L6336
    int16_t v5868 = v5866;	// L6337
    int16_t v5869 = v5867 * v5868;	// L6338
    ap_int<33> v5870 = v5864;	// L6339
    ap_int<33> v5871 = v5869;	// L6340
    ap_int<33> v5872 = v5870 + v5871;	// L6341
    int32_t v5873 = v5872;	// L6342
    v5853.write(v5873); // v5853[m234] = v5873;	// L6343
    int8_t v5874 = a234;	// L6344
    v5852.write(v5874); // v5852[m234] = v5874;	// L6345
  }
}

void PE_kernel_ws_gemm_11_14(
  hls::stream< int8_t > &v5875 /* v5875[16] */,
  hls::stream< int32_t > &v5876 /* v5876[16] */,
  hls::stream< int8_t > &v5877 /* v5877[16] */,
  hls::stream< int32_t > &v5878 /* v5878[16] */,
  int8_t v5879[16][16],
  int v5880,
  int v5881
) {	// L6349
  #pragma HLS stream variable=v5875 depth=17
  #pragma HLS stream variable=v5876 depth=17
  #pragma HLS stream variable=v5877 depth=17
  #pragma HLS stream variable=v5878 depth=17
  #pragma HLS array_partition variable=v5879 complete dim=1
  #pragma HLS array_partition variable=v5879 complete dim=2

  int8_t v5882 = v5879[v5880][v5881];	// L6350
  int8_t w235;	// L6351
  w235 = v5882;	// L6352
  l_step_m235: for (int m235 = 0; m235 < 16; m235++) {	// L6353
  #pragma HLS pipeline II=1
    int8_t v5885 = v5875.read(); // v5875[m235];	// L6354
    int8_t a235;	// L6355
    a235 = v5885;	// L6356
    int32_t v5887 = v5876.read(); // v5876[m235];	// L6357
    int32_t p235;	// L6358
    p235 = v5887;	// L6359
    int32_t v5889 = p235;	// L6360
    int8_t v5890 = a235;	// L6361
    int8_t v5891 = w235;	// L6362
    int16_t v5892 = v5890;	// L6363
    int16_t v5893 = v5891;	// L6364
    int16_t v5894 = v5892 * v5893;	// L6365
    ap_int<33> v5895 = v5889;	// L6366
    ap_int<33> v5896 = v5894;	// L6367
    ap_int<33> v5897 = v5895 + v5896;	// L6368
    int32_t v5898 = v5897;	// L6369
    v5878.write(v5898); // v5878[m235] = v5898;	// L6370
    int8_t v5899 = a235;	// L6371
    v5877.write(v5899); // v5877[m235] = v5899;	// L6372
  }
}

void PE_kernel_ws_gemm_12_14(
  hls::stream< int8_t > &v5900 /* v5900[16] */,
  hls::stream< int32_t > &v5901 /* v5901[16] */,
  hls::stream< int8_t > &v5902 /* v5902[16] */,
  hls::stream< int32_t > &v5903 /* v5903[16] */,
  int8_t v5904[16][16],
  int v5905,
  int v5906
) {	// L6376
  #pragma HLS stream variable=v5900 depth=17
  #pragma HLS stream variable=v5901 depth=17
  #pragma HLS stream variable=v5902 depth=17
  #pragma HLS stream variable=v5903 depth=17
  #pragma HLS array_partition variable=v5904 complete dim=1
  #pragma HLS array_partition variable=v5904 complete dim=2

  int8_t v5907 = v5904[v5905][v5906];	// L6377
  int8_t w236;	// L6378
  w236 = v5907;	// L6379
  l_step_m236: for (int m236 = 0; m236 < 16; m236++) {	// L6380
  #pragma HLS pipeline II=1
    int8_t v5910 = v5900.read(); // v5900[m236];	// L6381
    int8_t a236;	// L6382
    a236 = v5910;	// L6383
    int32_t v5912 = v5901.read(); // v5901[m236];	// L6384
    int32_t p236;	// L6385
    p236 = v5912;	// L6386
    int32_t v5914 = p236;	// L6387
    int8_t v5915 = a236;	// L6388
    int8_t v5916 = w236;	// L6389
    int16_t v5917 = v5915;	// L6390
    int16_t v5918 = v5916;	// L6391
    int16_t v5919 = v5917 * v5918;	// L6392
    ap_int<33> v5920 = v5914;	// L6393
    ap_int<33> v5921 = v5919;	// L6394
    ap_int<33> v5922 = v5920 + v5921;	// L6395
    int32_t v5923 = v5922;	// L6396
    v5903.write(v5923); // v5903[m236] = v5923;	// L6397
    int8_t v5924 = a236;	// L6398
    v5902.write(v5924); // v5902[m236] = v5924;	// L6399
  }
}

void PE_kernel_ws_gemm_13_14(
  hls::stream< int8_t > &v5925 /* v5925[16] */,
  hls::stream< int32_t > &v5926 /* v5926[16] */,
  hls::stream< int8_t > &v5927 /* v5927[16] */,
  hls::stream< int32_t > &v5928 /* v5928[16] */,
  int8_t v5929[16][16],
  int v5930,
  int v5931
) {	// L6403
  #pragma HLS stream variable=v5925 depth=17
  #pragma HLS stream variable=v5926 depth=17
  #pragma HLS stream variable=v5927 depth=17
  #pragma HLS stream variable=v5928 depth=17
  #pragma HLS array_partition variable=v5929 complete dim=1
  #pragma HLS array_partition variable=v5929 complete dim=2

  int8_t v5932 = v5929[v5930][v5931];	// L6404
  int8_t w237;	// L6405
  w237 = v5932;	// L6406
  l_step_m237: for (int m237 = 0; m237 < 16; m237++) {	// L6407
  #pragma HLS pipeline II=1
    int8_t v5935 = v5925.read(); // v5925[m237];	// L6408
    int8_t a237;	// L6409
    a237 = v5935;	// L6410
    int32_t v5937 = v5926.read(); // v5926[m237];	// L6411
    int32_t p237;	// L6412
    p237 = v5937;	// L6413
    int32_t v5939 = p237;	// L6414
    int8_t v5940 = a237;	// L6415
    int8_t v5941 = w237;	// L6416
    int16_t v5942 = v5940;	// L6417
    int16_t v5943 = v5941;	// L6418
    int16_t v5944 = v5942 * v5943;	// L6419
    ap_int<33> v5945 = v5939;	// L6420
    ap_int<33> v5946 = v5944;	// L6421
    ap_int<33> v5947 = v5945 + v5946;	// L6422
    int32_t v5948 = v5947;	// L6423
    v5928.write(v5948); // v5928[m237] = v5948;	// L6424
    int8_t v5949 = a237;	// L6425
    v5927.write(v5949); // v5927[m237] = v5949;	// L6426
  }
}

void PE_kernel_ws_gemm_14_14(
  hls::stream< int8_t > &v5950 /* v5950[16] */,
  hls::stream< int32_t > &v5951 /* v5951[16] */,
  hls::stream< int8_t > &v5952 /* v5952[16] */,
  hls::stream< int32_t > &v5953 /* v5953[16] */,
  int8_t v5954[16][16],
  int v5955,
  int v5956
) {	// L6430
  #pragma HLS stream variable=v5950 depth=17
  #pragma HLS stream variable=v5951 depth=17
  #pragma HLS stream variable=v5952 depth=17
  #pragma HLS stream variable=v5953 depth=17
  #pragma HLS array_partition variable=v5954 complete dim=1
  #pragma HLS array_partition variable=v5954 complete dim=2

  int8_t v5957 = v5954[v5955][v5956];	// L6431
  int8_t w238;	// L6432
  w238 = v5957;	// L6433
  l_step_m238: for (int m238 = 0; m238 < 16; m238++) {	// L6434
  #pragma HLS pipeline II=1
    int8_t v5960 = v5950.read(); // v5950[m238];	// L6435
    int8_t a238;	// L6436
    a238 = v5960;	// L6437
    int32_t v5962 = v5951.read(); // v5951[m238];	// L6438
    int32_t p238;	// L6439
    p238 = v5962;	// L6440
    int32_t v5964 = p238;	// L6441
    int8_t v5965 = a238;	// L6442
    int8_t v5966 = w238;	// L6443
    int16_t v5967 = v5965;	// L6444
    int16_t v5968 = v5966;	// L6445
    int16_t v5969 = v5967 * v5968;	// L6446
    ap_int<33> v5970 = v5964;	// L6447
    ap_int<33> v5971 = v5969;	// L6448
    ap_int<33> v5972 = v5970 + v5971;	// L6449
    int32_t v5973 = v5972;	// L6450
    v5953.write(v5973); // v5953[m238] = v5973;	// L6451
    int8_t v5974 = a238;	// L6452
    v5952.write(v5974); // v5952[m238] = v5974;	// L6453
  }
}

void PE_kernel_ws_gemm_15_14(
  hls::stream< int8_t > &v5975 /* v5975[16] */,
  hls::stream< int32_t > &v5976 /* v5976[16] */,
  hls::stream< int8_t > &v5977 /* v5977[16] */,
  hls::stream< int32_t > &v5978 /* v5978[16] */,
  int8_t v5979[16][16],
  int v5980,
  int v5981
) {	// L6457
  #pragma HLS stream variable=v5975 depth=17
  #pragma HLS stream variable=v5976 depth=17
  #pragma HLS stream variable=v5977 depth=17
  #pragma HLS stream variable=v5978 depth=17
  #pragma HLS array_partition variable=v5979 complete dim=1
  #pragma HLS array_partition variable=v5979 complete dim=2

  int8_t v5982 = v5979[v5980][v5981];	// L6458
  int8_t w239;	// L6459
  w239 = v5982;	// L6460
  l_step_m239: for (int m239 = 0; m239 < 16; m239++) {	// L6461
  #pragma HLS pipeline II=1
    int8_t v5985 = v5975.read(); // v5975[m239];	// L6462
    int8_t a239;	// L6463
    a239 = v5985;	// L6464
    int32_t v5987 = v5976.read(); // v5976[m239];	// L6465
    int32_t p239;	// L6466
    p239 = v5987;	// L6467
    int32_t v5989 = p239;	// L6468
    int8_t v5990 = a239;	// L6469
    int8_t v5991 = w239;	// L6470
    int16_t v5992 = v5990;	// L6471
    int16_t v5993 = v5991;	// L6472
    int16_t v5994 = v5992 * v5993;	// L6473
    ap_int<33> v5995 = v5989;	// L6474
    ap_int<33> v5996 = v5994;	// L6475
    ap_int<33> v5997 = v5995 + v5996;	// L6476
    int32_t v5998 = v5997;	// L6477
    v5978.write(v5998); // v5978[m239] = v5998;	// L6478
    int8_t v5999 = a239;	// L6479
    v5977.write(v5999); // v5977[m239] = v5999;	// L6480
  }
}

void PE_kernel_ws_gemm_0_15(
  hls::stream< int8_t > &v6000 /* v6000[16] */,
  hls::stream< int32_t > &v6001 /* v6001[16] */,
  hls::stream< int8_t > &v6002 /* v6002[16] */,
  hls::stream< int32_t > &v6003 /* v6003[16] */,
  int8_t v6004[16][16],
  int v6005,
  int v6006
) {	// L6484
  #pragma HLS stream variable=v6000 depth=17
  #pragma HLS stream variable=v6001 depth=17
  #pragma HLS stream variable=v6002 depth=17
  #pragma HLS stream variable=v6003 depth=17
  #pragma HLS array_partition variable=v6004 complete dim=1
  #pragma HLS array_partition variable=v6004 complete dim=2

  int8_t v6007 = v6004[v6005][v6006];	// L6485
  int8_t w240;	// L6486
  w240 = v6007;	// L6487
  l_step_m240: for (int m240 = 0; m240 < 16; m240++) {	// L6488
  #pragma HLS pipeline II=1
    int8_t v6010 = v6000.read(); // v6000[m240];	// L6489
    int8_t a240;	// L6490
    a240 = v6010;	// L6491
    int32_t v6012 = v6001.read(); // v6001[m240];	// L6492
    int32_t p240;	// L6493
    p240 = v6012;	// L6494
    int32_t v6014 = p240;	// L6495
    int8_t v6015 = a240;	// L6496
    int8_t v6016 = w240;	// L6497
    int16_t v6017 = v6015;	// L6498
    int16_t v6018 = v6016;	// L6499
    int16_t v6019 = v6017 * v6018;	// L6500
    ap_int<33> v6020 = v6014;	// L6501
    ap_int<33> v6021 = v6019;	// L6502
    ap_int<33> v6022 = v6020 + v6021;	// L6503
    int32_t v6023 = v6022;	// L6504
    v6003.write(v6023); // v6003[m240] = v6023;	// L6505
    int8_t v6024 = a240;	// L6506
    v6002.write(v6024); // v6002[m240] = v6024;	// L6507
  }
}

void PE_kernel_ws_gemm_1_15(
  hls::stream< int8_t > &v6025 /* v6025[16] */,
  hls::stream< int32_t > &v6026 /* v6026[16] */,
  hls::stream< int8_t > &v6027 /* v6027[16] */,
  hls::stream< int32_t > &v6028 /* v6028[16] */,
  int8_t v6029[16][16],
  int v6030,
  int v6031
) {	// L6511
  #pragma HLS stream variable=v6025 depth=17
  #pragma HLS stream variable=v6026 depth=17
  #pragma HLS stream variable=v6027 depth=17
  #pragma HLS stream variable=v6028 depth=17
  #pragma HLS array_partition variable=v6029 complete dim=1
  #pragma HLS array_partition variable=v6029 complete dim=2

  int8_t v6032 = v6029[v6030][v6031];	// L6512
  int8_t w241;	// L6513
  w241 = v6032;	// L6514
  l_step_m241: for (int m241 = 0; m241 < 16; m241++) {	// L6515
  #pragma HLS pipeline II=1
    int8_t v6035 = v6025.read(); // v6025[m241];	// L6516
    int8_t a241;	// L6517
    a241 = v6035;	// L6518
    int32_t v6037 = v6026.read(); // v6026[m241];	// L6519
    int32_t p241;	// L6520
    p241 = v6037;	// L6521
    int32_t v6039 = p241;	// L6522
    int8_t v6040 = a241;	// L6523
    int8_t v6041 = w241;	// L6524
    int16_t v6042 = v6040;	// L6525
    int16_t v6043 = v6041;	// L6526
    int16_t v6044 = v6042 * v6043;	// L6527
    ap_int<33> v6045 = v6039;	// L6528
    ap_int<33> v6046 = v6044;	// L6529
    ap_int<33> v6047 = v6045 + v6046;	// L6530
    int32_t v6048 = v6047;	// L6531
    v6028.write(v6048); // v6028[m241] = v6048;	// L6532
    int8_t v6049 = a241;	// L6533
    v6027.write(v6049); // v6027[m241] = v6049;	// L6534
  }
}

void PE_kernel_ws_gemm_2_15(
  hls::stream< int8_t > &v6050 /* v6050[16] */,
  hls::stream< int32_t > &v6051 /* v6051[16] */,
  hls::stream< int8_t > &v6052 /* v6052[16] */,
  hls::stream< int32_t > &v6053 /* v6053[16] */,
  int8_t v6054[16][16],
  int v6055,
  int v6056
) {	// L6538
  #pragma HLS stream variable=v6050 depth=17
  #pragma HLS stream variable=v6051 depth=17
  #pragma HLS stream variable=v6052 depth=17
  #pragma HLS stream variable=v6053 depth=17
  #pragma HLS array_partition variable=v6054 complete dim=1
  #pragma HLS array_partition variable=v6054 complete dim=2

  int8_t v6057 = v6054[v6055][v6056];	// L6539
  int8_t w242;	// L6540
  w242 = v6057;	// L6541
  l_step_m242: for (int m242 = 0; m242 < 16; m242++) {	// L6542
  #pragma HLS pipeline II=1
    int8_t v6060 = v6050.read(); // v6050[m242];	// L6543
    int8_t a242;	// L6544
    a242 = v6060;	// L6545
    int32_t v6062 = v6051.read(); // v6051[m242];	// L6546
    int32_t p242;	// L6547
    p242 = v6062;	// L6548
    int32_t v6064 = p242;	// L6549
    int8_t v6065 = a242;	// L6550
    int8_t v6066 = w242;	// L6551
    int16_t v6067 = v6065;	// L6552
    int16_t v6068 = v6066;	// L6553
    int16_t v6069 = v6067 * v6068;	// L6554
    ap_int<33> v6070 = v6064;	// L6555
    ap_int<33> v6071 = v6069;	// L6556
    ap_int<33> v6072 = v6070 + v6071;	// L6557
    int32_t v6073 = v6072;	// L6558
    v6053.write(v6073); // v6053[m242] = v6073;	// L6559
    int8_t v6074 = a242;	// L6560
    v6052.write(v6074); // v6052[m242] = v6074;	// L6561
  }
}

void PE_kernel_ws_gemm_3_15(
  hls::stream< int8_t > &v6075 /* v6075[16] */,
  hls::stream< int32_t > &v6076 /* v6076[16] */,
  hls::stream< int8_t > &v6077 /* v6077[16] */,
  hls::stream< int32_t > &v6078 /* v6078[16] */,
  int8_t v6079[16][16],
  int v6080,
  int v6081
) {	// L6565
  #pragma HLS stream variable=v6075 depth=17
  #pragma HLS stream variable=v6076 depth=17
  #pragma HLS stream variable=v6077 depth=17
  #pragma HLS stream variable=v6078 depth=17
  #pragma HLS array_partition variable=v6079 complete dim=1
  #pragma HLS array_partition variable=v6079 complete dim=2

  int8_t v6082 = v6079[v6080][v6081];	// L6566
  int8_t w243;	// L6567
  w243 = v6082;	// L6568
  l_step_m243: for (int m243 = 0; m243 < 16; m243++) {	// L6569
  #pragma HLS pipeline II=1
    int8_t v6085 = v6075.read(); // v6075[m243];	// L6570
    int8_t a243;	// L6571
    a243 = v6085;	// L6572
    int32_t v6087 = v6076.read(); // v6076[m243];	// L6573
    int32_t p243;	// L6574
    p243 = v6087;	// L6575
    int32_t v6089 = p243;	// L6576
    int8_t v6090 = a243;	// L6577
    int8_t v6091 = w243;	// L6578
    int16_t v6092 = v6090;	// L6579
    int16_t v6093 = v6091;	// L6580
    int16_t v6094 = v6092 * v6093;	// L6581
    ap_int<33> v6095 = v6089;	// L6582
    ap_int<33> v6096 = v6094;	// L6583
    ap_int<33> v6097 = v6095 + v6096;	// L6584
    int32_t v6098 = v6097;	// L6585
    v6078.write(v6098); // v6078[m243] = v6098;	// L6586
    int8_t v6099 = a243;	// L6587
    v6077.write(v6099); // v6077[m243] = v6099;	// L6588
  }
}

void PE_kernel_ws_gemm_4_15(
  hls::stream< int8_t > &v6100 /* v6100[16] */,
  hls::stream< int32_t > &v6101 /* v6101[16] */,
  hls::stream< int8_t > &v6102 /* v6102[16] */,
  hls::stream< int32_t > &v6103 /* v6103[16] */,
  int8_t v6104[16][16],
  int v6105,
  int v6106
) {	// L6592
  #pragma HLS stream variable=v6100 depth=17
  #pragma HLS stream variable=v6101 depth=17
  #pragma HLS stream variable=v6102 depth=17
  #pragma HLS stream variable=v6103 depth=17
  #pragma HLS array_partition variable=v6104 complete dim=1
  #pragma HLS array_partition variable=v6104 complete dim=2

  int8_t v6107 = v6104[v6105][v6106];	// L6593
  int8_t w244;	// L6594
  w244 = v6107;	// L6595
  l_step_m244: for (int m244 = 0; m244 < 16; m244++) {	// L6596
  #pragma HLS pipeline II=1
    int8_t v6110 = v6100.read(); // v6100[m244];	// L6597
    int8_t a244;	// L6598
    a244 = v6110;	// L6599
    int32_t v6112 = v6101.read(); // v6101[m244];	// L6600
    int32_t p244;	// L6601
    p244 = v6112;	// L6602
    int32_t v6114 = p244;	// L6603
    int8_t v6115 = a244;	// L6604
    int8_t v6116 = w244;	// L6605
    int16_t v6117 = v6115;	// L6606
    int16_t v6118 = v6116;	// L6607
    int16_t v6119 = v6117 * v6118;	// L6608
    ap_int<33> v6120 = v6114;	// L6609
    ap_int<33> v6121 = v6119;	// L6610
    ap_int<33> v6122 = v6120 + v6121;	// L6611
    int32_t v6123 = v6122;	// L6612
    v6103.write(v6123); // v6103[m244] = v6123;	// L6613
    int8_t v6124 = a244;	// L6614
    v6102.write(v6124); // v6102[m244] = v6124;	// L6615
  }
}

void PE_kernel_ws_gemm_5_15(
  hls::stream< int8_t > &v6125 /* v6125[16] */,
  hls::stream< int32_t > &v6126 /* v6126[16] */,
  hls::stream< int8_t > &v6127 /* v6127[16] */,
  hls::stream< int32_t > &v6128 /* v6128[16] */,
  int8_t v6129[16][16],
  int v6130,
  int v6131
) {	// L6619
  #pragma HLS stream variable=v6125 depth=17
  #pragma HLS stream variable=v6126 depth=17
  #pragma HLS stream variable=v6127 depth=17
  #pragma HLS stream variable=v6128 depth=17
  #pragma HLS array_partition variable=v6129 complete dim=1
  #pragma HLS array_partition variable=v6129 complete dim=2

  int8_t v6132 = v6129[v6130][v6131];	// L6620
  int8_t w245;	// L6621
  w245 = v6132;	// L6622
  l_step_m245: for (int m245 = 0; m245 < 16; m245++) {	// L6623
  #pragma HLS pipeline II=1
    int8_t v6135 = v6125.read(); // v6125[m245];	// L6624
    int8_t a245;	// L6625
    a245 = v6135;	// L6626
    int32_t v6137 = v6126.read(); // v6126[m245];	// L6627
    int32_t p245;	// L6628
    p245 = v6137;	// L6629
    int32_t v6139 = p245;	// L6630
    int8_t v6140 = a245;	// L6631
    int8_t v6141 = w245;	// L6632
    int16_t v6142 = v6140;	// L6633
    int16_t v6143 = v6141;	// L6634
    int16_t v6144 = v6142 * v6143;	// L6635
    ap_int<33> v6145 = v6139;	// L6636
    ap_int<33> v6146 = v6144;	// L6637
    ap_int<33> v6147 = v6145 + v6146;	// L6638
    int32_t v6148 = v6147;	// L6639
    v6128.write(v6148); // v6128[m245] = v6148;	// L6640
    int8_t v6149 = a245;	// L6641
    v6127.write(v6149); // v6127[m245] = v6149;	// L6642
  }
}

void PE_kernel_ws_gemm_6_15(
  hls::stream< int8_t > &v6150 /* v6150[16] */,
  hls::stream< int32_t > &v6151 /* v6151[16] */,
  hls::stream< int8_t > &v6152 /* v6152[16] */,
  hls::stream< int32_t > &v6153 /* v6153[16] */,
  int8_t v6154[16][16],
  int v6155,
  int v6156
) {	// L6646
  #pragma HLS stream variable=v6150 depth=17
  #pragma HLS stream variable=v6151 depth=17
  #pragma HLS stream variable=v6152 depth=17
  #pragma HLS stream variable=v6153 depth=17
  #pragma HLS array_partition variable=v6154 complete dim=1
  #pragma HLS array_partition variable=v6154 complete dim=2

  int8_t v6157 = v6154[v6155][v6156];	// L6647
  int8_t w246;	// L6648
  w246 = v6157;	// L6649
  l_step_m246: for (int m246 = 0; m246 < 16; m246++) {	// L6650
  #pragma HLS pipeline II=1
    int8_t v6160 = v6150.read(); // v6150[m246];	// L6651
    int8_t a246;	// L6652
    a246 = v6160;	// L6653
    int32_t v6162 = v6151.read(); // v6151[m246];	// L6654
    int32_t p246;	// L6655
    p246 = v6162;	// L6656
    int32_t v6164 = p246;	// L6657
    int8_t v6165 = a246;	// L6658
    int8_t v6166 = w246;	// L6659
    int16_t v6167 = v6165;	// L6660
    int16_t v6168 = v6166;	// L6661
    int16_t v6169 = v6167 * v6168;	// L6662
    ap_int<33> v6170 = v6164;	// L6663
    ap_int<33> v6171 = v6169;	// L6664
    ap_int<33> v6172 = v6170 + v6171;	// L6665
    int32_t v6173 = v6172;	// L6666
    v6153.write(v6173); // v6153[m246] = v6173;	// L6667
    int8_t v6174 = a246;	// L6668
    v6152.write(v6174); // v6152[m246] = v6174;	// L6669
  }
}

void PE_kernel_ws_gemm_7_15(
  hls::stream< int8_t > &v6175 /* v6175[16] */,
  hls::stream< int32_t > &v6176 /* v6176[16] */,
  hls::stream< int8_t > &v6177 /* v6177[16] */,
  hls::stream< int32_t > &v6178 /* v6178[16] */,
  int8_t v6179[16][16],
  int v6180,
  int v6181
) {	// L6673
  #pragma HLS stream variable=v6175 depth=17
  #pragma HLS stream variable=v6176 depth=17
  #pragma HLS stream variable=v6177 depth=17
  #pragma HLS stream variable=v6178 depth=17
  #pragma HLS array_partition variable=v6179 complete dim=1
  #pragma HLS array_partition variable=v6179 complete dim=2

  int8_t v6182 = v6179[v6180][v6181];	// L6674
  int8_t w247;	// L6675
  w247 = v6182;	// L6676
  l_step_m247: for (int m247 = 0; m247 < 16; m247++) {	// L6677
  #pragma HLS pipeline II=1
    int8_t v6185 = v6175.read(); // v6175[m247];	// L6678
    int8_t a247;	// L6679
    a247 = v6185;	// L6680
    int32_t v6187 = v6176.read(); // v6176[m247];	// L6681
    int32_t p247;	// L6682
    p247 = v6187;	// L6683
    int32_t v6189 = p247;	// L6684
    int8_t v6190 = a247;	// L6685
    int8_t v6191 = w247;	// L6686
    int16_t v6192 = v6190;	// L6687
    int16_t v6193 = v6191;	// L6688
    int16_t v6194 = v6192 * v6193;	// L6689
    ap_int<33> v6195 = v6189;	// L6690
    ap_int<33> v6196 = v6194;	// L6691
    ap_int<33> v6197 = v6195 + v6196;	// L6692
    int32_t v6198 = v6197;	// L6693
    v6178.write(v6198); // v6178[m247] = v6198;	// L6694
    int8_t v6199 = a247;	// L6695
    v6177.write(v6199); // v6177[m247] = v6199;	// L6696
  }
}

void PE_kernel_ws_gemm_8_15(
  hls::stream< int8_t > &v6200 /* v6200[16] */,
  hls::stream< int32_t > &v6201 /* v6201[16] */,
  hls::stream< int8_t > &v6202 /* v6202[16] */,
  hls::stream< int32_t > &v6203 /* v6203[16] */,
  int8_t v6204[16][16],
  int v6205,
  int v6206
) {	// L6700
  #pragma HLS stream variable=v6200 depth=17
  #pragma HLS stream variable=v6201 depth=17
  #pragma HLS stream variable=v6202 depth=17
  #pragma HLS stream variable=v6203 depth=17
  #pragma HLS array_partition variable=v6204 complete dim=1
  #pragma HLS array_partition variable=v6204 complete dim=2

  int8_t v6207 = v6204[v6205][v6206];	// L6701
  int8_t w248;	// L6702
  w248 = v6207;	// L6703
  l_step_m248: for (int m248 = 0; m248 < 16; m248++) {	// L6704
  #pragma HLS pipeline II=1
    int8_t v6210 = v6200.read(); // v6200[m248];	// L6705
    int8_t a248;	// L6706
    a248 = v6210;	// L6707
    int32_t v6212 = v6201.read(); // v6201[m248];	// L6708
    int32_t p248;	// L6709
    p248 = v6212;	// L6710
    int32_t v6214 = p248;	// L6711
    int8_t v6215 = a248;	// L6712
    int8_t v6216 = w248;	// L6713
    int16_t v6217 = v6215;	// L6714
    int16_t v6218 = v6216;	// L6715
    int16_t v6219 = v6217 * v6218;	// L6716
    ap_int<33> v6220 = v6214;	// L6717
    ap_int<33> v6221 = v6219;	// L6718
    ap_int<33> v6222 = v6220 + v6221;	// L6719
    int32_t v6223 = v6222;	// L6720
    v6203.write(v6223); // v6203[m248] = v6223;	// L6721
    int8_t v6224 = a248;	// L6722
    v6202.write(v6224); // v6202[m248] = v6224;	// L6723
  }
}

void PE_kernel_ws_gemm_9_15(
  hls::stream< int8_t > &v6225 /* v6225[16] */,
  hls::stream< int32_t > &v6226 /* v6226[16] */,
  hls::stream< int8_t > &v6227 /* v6227[16] */,
  hls::stream< int32_t > &v6228 /* v6228[16] */,
  int8_t v6229[16][16],
  int v6230,
  int v6231
) {	// L6727
  #pragma HLS stream variable=v6225 depth=17
  #pragma HLS stream variable=v6226 depth=17
  #pragma HLS stream variable=v6227 depth=17
  #pragma HLS stream variable=v6228 depth=17
  #pragma HLS array_partition variable=v6229 complete dim=1
  #pragma HLS array_partition variable=v6229 complete dim=2

  int8_t v6232 = v6229[v6230][v6231];	// L6728
  int8_t w249;	// L6729
  w249 = v6232;	// L6730
  l_step_m249: for (int m249 = 0; m249 < 16; m249++) {	// L6731
  #pragma HLS pipeline II=1
    int8_t v6235 = v6225.read(); // v6225[m249];	// L6732
    int8_t a249;	// L6733
    a249 = v6235;	// L6734
    int32_t v6237 = v6226.read(); // v6226[m249];	// L6735
    int32_t p249;	// L6736
    p249 = v6237;	// L6737
    int32_t v6239 = p249;	// L6738
    int8_t v6240 = a249;	// L6739
    int8_t v6241 = w249;	// L6740
    int16_t v6242 = v6240;	// L6741
    int16_t v6243 = v6241;	// L6742
    int16_t v6244 = v6242 * v6243;	// L6743
    ap_int<33> v6245 = v6239;	// L6744
    ap_int<33> v6246 = v6244;	// L6745
    ap_int<33> v6247 = v6245 + v6246;	// L6746
    int32_t v6248 = v6247;	// L6747
    v6228.write(v6248); // v6228[m249] = v6248;	// L6748
    int8_t v6249 = a249;	// L6749
    v6227.write(v6249); // v6227[m249] = v6249;	// L6750
  }
}

void PE_kernel_ws_gemm_10_15(
  hls::stream< int8_t > &v6250 /* v6250[16] */,
  hls::stream< int32_t > &v6251 /* v6251[16] */,
  hls::stream< int8_t > &v6252 /* v6252[16] */,
  hls::stream< int32_t > &v6253 /* v6253[16] */,
  int8_t v6254[16][16],
  int v6255,
  int v6256
) {	// L6754
  #pragma HLS stream variable=v6250 depth=17
  #pragma HLS stream variable=v6251 depth=17
  #pragma HLS stream variable=v6252 depth=17
  #pragma HLS stream variable=v6253 depth=17
  #pragma HLS array_partition variable=v6254 complete dim=1
  #pragma HLS array_partition variable=v6254 complete dim=2

  int8_t v6257 = v6254[v6255][v6256];	// L6755
  int8_t w250;	// L6756
  w250 = v6257;	// L6757
  l_step_m250: for (int m250 = 0; m250 < 16; m250++) {	// L6758
  #pragma HLS pipeline II=1
    int8_t v6260 = v6250.read(); // v6250[m250];	// L6759
    int8_t a250;	// L6760
    a250 = v6260;	// L6761
    int32_t v6262 = v6251.read(); // v6251[m250];	// L6762
    int32_t p250;	// L6763
    p250 = v6262;	// L6764
    int32_t v6264 = p250;	// L6765
    int8_t v6265 = a250;	// L6766
    int8_t v6266 = w250;	// L6767
    int16_t v6267 = v6265;	// L6768
    int16_t v6268 = v6266;	// L6769
    int16_t v6269 = v6267 * v6268;	// L6770
    ap_int<33> v6270 = v6264;	// L6771
    ap_int<33> v6271 = v6269;	// L6772
    ap_int<33> v6272 = v6270 + v6271;	// L6773
    int32_t v6273 = v6272;	// L6774
    v6253.write(v6273); // v6253[m250] = v6273;	// L6775
    int8_t v6274 = a250;	// L6776
    v6252.write(v6274); // v6252[m250] = v6274;	// L6777
  }
}

void PE_kernel_ws_gemm_11_15(
  hls::stream< int8_t > &v6275 /* v6275[16] */,
  hls::stream< int32_t > &v6276 /* v6276[16] */,
  hls::stream< int8_t > &v6277 /* v6277[16] */,
  hls::stream< int32_t > &v6278 /* v6278[16] */,
  int8_t v6279[16][16],
  int v6280,
  int v6281
) {	// L6781
  #pragma HLS stream variable=v6275 depth=17
  #pragma HLS stream variable=v6276 depth=17
  #pragma HLS stream variable=v6277 depth=17
  #pragma HLS stream variable=v6278 depth=17
  #pragma HLS array_partition variable=v6279 complete dim=1
  #pragma HLS array_partition variable=v6279 complete dim=2

  int8_t v6282 = v6279[v6280][v6281];	// L6782
  int8_t w251;	// L6783
  w251 = v6282;	// L6784
  l_step_m251: for (int m251 = 0; m251 < 16; m251++) {	// L6785
  #pragma HLS pipeline II=1
    int8_t v6285 = v6275.read(); // v6275[m251];	// L6786
    int8_t a251;	// L6787
    a251 = v6285;	// L6788
    int32_t v6287 = v6276.read(); // v6276[m251];	// L6789
    int32_t p251;	// L6790
    p251 = v6287;	// L6791
    int32_t v6289 = p251;	// L6792
    int8_t v6290 = a251;	// L6793
    int8_t v6291 = w251;	// L6794
    int16_t v6292 = v6290;	// L6795
    int16_t v6293 = v6291;	// L6796
    int16_t v6294 = v6292 * v6293;	// L6797
    ap_int<33> v6295 = v6289;	// L6798
    ap_int<33> v6296 = v6294;	// L6799
    ap_int<33> v6297 = v6295 + v6296;	// L6800
    int32_t v6298 = v6297;	// L6801
    v6278.write(v6298); // v6278[m251] = v6298;	// L6802
    int8_t v6299 = a251;	// L6803
    v6277.write(v6299); // v6277[m251] = v6299;	// L6804
  }
}

void PE_kernel_ws_gemm_12_15(
  hls::stream< int8_t > &v6300 /* v6300[16] */,
  hls::stream< int32_t > &v6301 /* v6301[16] */,
  hls::stream< int8_t > &v6302 /* v6302[16] */,
  hls::stream< int32_t > &v6303 /* v6303[16] */,
  int8_t v6304[16][16],
  int v6305,
  int v6306
) {	// L6808
  #pragma HLS stream variable=v6300 depth=17
  #pragma HLS stream variable=v6301 depth=17
  #pragma HLS stream variable=v6302 depth=17
  #pragma HLS stream variable=v6303 depth=17
  #pragma HLS array_partition variable=v6304 complete dim=1
  #pragma HLS array_partition variable=v6304 complete dim=2

  int8_t v6307 = v6304[v6305][v6306];	// L6809
  int8_t w252;	// L6810
  w252 = v6307;	// L6811
  l_step_m252: for (int m252 = 0; m252 < 16; m252++) {	// L6812
  #pragma HLS pipeline II=1
    int8_t v6310 = v6300.read(); // v6300[m252];	// L6813
    int8_t a252;	// L6814
    a252 = v6310;	// L6815
    int32_t v6312 = v6301.read(); // v6301[m252];	// L6816
    int32_t p252;	// L6817
    p252 = v6312;	// L6818
    int32_t v6314 = p252;	// L6819
    int8_t v6315 = a252;	// L6820
    int8_t v6316 = w252;	// L6821
    int16_t v6317 = v6315;	// L6822
    int16_t v6318 = v6316;	// L6823
    int16_t v6319 = v6317 * v6318;	// L6824
    ap_int<33> v6320 = v6314;	// L6825
    ap_int<33> v6321 = v6319;	// L6826
    ap_int<33> v6322 = v6320 + v6321;	// L6827
    int32_t v6323 = v6322;	// L6828
    v6303.write(v6323); // v6303[m252] = v6323;	// L6829
    int8_t v6324 = a252;	// L6830
    v6302.write(v6324); // v6302[m252] = v6324;	// L6831
  }
}

void PE_kernel_ws_gemm_13_15(
  hls::stream< int8_t > &v6325 /* v6325[16] */,
  hls::stream< int32_t > &v6326 /* v6326[16] */,
  hls::stream< int8_t > &v6327 /* v6327[16] */,
  hls::stream< int32_t > &v6328 /* v6328[16] */,
  int8_t v6329[16][16],
  int v6330,
  int v6331
) {	// L6835
  #pragma HLS stream variable=v6325 depth=17
  #pragma HLS stream variable=v6326 depth=17
  #pragma HLS stream variable=v6327 depth=17
  #pragma HLS stream variable=v6328 depth=17
  #pragma HLS array_partition variable=v6329 complete dim=1
  #pragma HLS array_partition variable=v6329 complete dim=2

  int8_t v6332 = v6329[v6330][v6331];	// L6836
  int8_t w253;	// L6837
  w253 = v6332;	// L6838
  l_step_m253: for (int m253 = 0; m253 < 16; m253++) {	// L6839
  #pragma HLS pipeline II=1
    int8_t v6335 = v6325.read(); // v6325[m253];	// L6840
    int8_t a253;	// L6841
    a253 = v6335;	// L6842
    int32_t v6337 = v6326.read(); // v6326[m253];	// L6843
    int32_t p253;	// L6844
    p253 = v6337;	// L6845
    int32_t v6339 = p253;	// L6846
    int8_t v6340 = a253;	// L6847
    int8_t v6341 = w253;	// L6848
    int16_t v6342 = v6340;	// L6849
    int16_t v6343 = v6341;	// L6850
    int16_t v6344 = v6342 * v6343;	// L6851
    ap_int<33> v6345 = v6339;	// L6852
    ap_int<33> v6346 = v6344;	// L6853
    ap_int<33> v6347 = v6345 + v6346;	// L6854
    int32_t v6348 = v6347;	// L6855
    v6328.write(v6348); // v6328[m253] = v6348;	// L6856
    int8_t v6349 = a253;	// L6857
    v6327.write(v6349); // v6327[m253] = v6349;	// L6858
  }
}

void PE_kernel_ws_gemm_14_15(
  hls::stream< int8_t > &v6350 /* v6350[16] */,
  hls::stream< int32_t > &v6351 /* v6351[16] */,
  hls::stream< int8_t > &v6352 /* v6352[16] */,
  hls::stream< int32_t > &v6353 /* v6353[16] */,
  int8_t v6354[16][16],
  int v6355,
  int v6356
) {	// L6862
  #pragma HLS stream variable=v6350 depth=17
  #pragma HLS stream variable=v6351 depth=17
  #pragma HLS stream variable=v6352 depth=17
  #pragma HLS stream variable=v6353 depth=17
  #pragma HLS array_partition variable=v6354 complete dim=1
  #pragma HLS array_partition variable=v6354 complete dim=2

  int8_t v6357 = v6354[v6355][v6356];	// L6863
  int8_t w254;	// L6864
  w254 = v6357;	// L6865
  l_step_m254: for (int m254 = 0; m254 < 16; m254++) {	// L6866
  #pragma HLS pipeline II=1
    int8_t v6360 = v6350.read(); // v6350[m254];	// L6867
    int8_t a254;	// L6868
    a254 = v6360;	// L6869
    int32_t v6362 = v6351.read(); // v6351[m254];	// L6870
    int32_t p254;	// L6871
    p254 = v6362;	// L6872
    int32_t v6364 = p254;	// L6873
    int8_t v6365 = a254;	// L6874
    int8_t v6366 = w254;	// L6875
    int16_t v6367 = v6365;	// L6876
    int16_t v6368 = v6366;	// L6877
    int16_t v6369 = v6367 * v6368;	// L6878
    ap_int<33> v6370 = v6364;	// L6879
    ap_int<33> v6371 = v6369;	// L6880
    ap_int<33> v6372 = v6370 + v6371;	// L6881
    int32_t v6373 = v6372;	// L6882
    v6353.write(v6373); // v6353[m254] = v6373;	// L6883
    int8_t v6374 = a254;	// L6884
    v6352.write(v6374); // v6352[m254] = v6374;	// L6885
  }
}

void PE_kernel_ws_gemm_15_15(
  hls::stream< int8_t > &v6375 /* v6375[16] */,
  hls::stream< int32_t > &v6376 /* v6376[16] */,
  hls::stream< int8_t > &v6377 /* v6377[16] */,
  hls::stream< int32_t > &v6378 /* v6378[16] */,
  int8_t v6379[16][16],
  int v6380,
  int v6381
) {	// L6889
  #pragma HLS stream variable=v6375 depth=17
  #pragma HLS stream variable=v6376 depth=17
  #pragma HLS stream variable=v6377 depth=17
  #pragma HLS stream variable=v6378 depth=17
  #pragma HLS array_partition variable=v6379 complete dim=1
  #pragma HLS array_partition variable=v6379 complete dim=2

  int8_t v6382 = v6379[v6380][v6381];	// L6890
  int8_t w255;	// L6891
  w255 = v6382;	// L6892
  l_step_m255: for (int m255 = 0; m255 < 16; m255++) {	// L6893
  #pragma HLS pipeline II=1
    int8_t v6385 = v6375.read(); // v6375[m255];	// L6894
    int8_t a255;	// L6895
    a255 = v6385;	// L6896
    int32_t v6387 = v6376.read(); // v6376[m255];	// L6897
    int32_t p255;	// L6898
    p255 = v6387;	// L6899
    int32_t v6389 = p255;	// L6900
    int8_t v6390 = a255;	// L6901
    int8_t v6391 = w255;	// L6902
    int16_t v6392 = v6390;	// L6903
    int16_t v6393 = v6391;	// L6904
    int16_t v6394 = v6392 * v6393;	// L6905
    ap_int<33> v6395 = v6389;	// L6906
    ap_int<33> v6396 = v6394;	// L6907
    ap_int<33> v6397 = v6395 + v6396;	// L6908
    int32_t v6398 = v6397;	// L6909
    v6378.write(v6398); // v6378[m255] = v6398;	// L6910
    int8_t v6399 = a255;	// L6911
    v6377.write(v6399); // v6377[m255] = v6399;	// L6912
  }
}

void systolic_tile_ws_gemm(
  int8_t v6400[16][16],
  int8_t v6401[16][16],
  int32_t v6402[16][16]
) {	// L6916
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v6400 complete dim=2

  #pragma HLS array_partition variable=v6401 complete dim=1
  #pragma HLS array_partition variable=v6401 complete dim=2

  #pragma HLS array_partition variable=v6402 complete dim=2

  hls::stream< int8_t > A_fifo[16][17] /* A_fifo[16][17][16] */;	// L6917
  #pragma HLS stream variable=A_fifo depth=17
  hls::stream< int32_t > P_fifo[16][17] /* P_fifo[16][17][16] */;	// L6918
  #pragma HLS stream variable=P_fifo depth=17
  int8_t A_drain[16];	// L6919
  l_data_load_m256: for (int m256 = 0; m256 < 16; m256++) {	// L6920
    l_S_k_0_k: for (int k = 0; k < 16; k++) {	// L6921
      int8_t v6408 = v6400[m256][k];	// L6922
      A_fifo[k][0].write(v6408); // A_fifo[k][0][m256] = v6408;	// L6923
    }
    l_S_n_1_n: for (int n = 0; n < 16; n++) {	// L6925
      P_fifo[n][0].write(0); // P_fifo[n][0][m256] = 0;	// L6927
    }
  }
  hls::stream< int8_t > &v6410 /* v6410[16] */ = A_fifo[0][0];	// L6931
  hls::stream< int32_t > &v6411 /* v6411[16] */ = P_fifo[0][0];	// L6932
  hls::stream< int8_t > &v6412 /* v6412[16] */ = A_fifo[0][1];	// L6938
  hls::stream< int32_t > &v6413 /* v6413[16] */ = P_fifo[0][1];	// L6939
  PE_kernel_ws_gemm_0_0(v6410, v6411, v6412, v6413, v6401, 0, 0);	// L6940
  hls::stream< int8_t > &v6414 /* v6414[16] */ = A_fifo[0][1];	// L6942
  hls::stream< int32_t > &v6415 /* v6415[16] */ = P_fifo[1][0];	// L6943
  hls::stream< int8_t > &v6416 /* v6416[16] */ = A_fifo[0][2];	// L6947
  hls::stream< int32_t > &v6417 /* v6417[16] */ = P_fifo[1][1];	// L6948
  PE_kernel_ws_gemm_1_0(v6414, v6415, v6416, v6417, v6401, 0, 1);	// L6949
  hls::stream< int8_t > &v6418 /* v6418[16] */ = A_fifo[0][2];	// L6951
  hls::stream< int32_t > &v6419 /* v6419[16] */ = P_fifo[2][0];	// L6952
  hls::stream< int8_t > &v6420 /* v6420[16] */ = A_fifo[0][3];	// L6956
  hls::stream< int32_t > &v6421 /* v6421[16] */ = P_fifo[2][1];	// L6957
  PE_kernel_ws_gemm_2_0(v6418, v6419, v6420, v6421, v6401, 0, 2);	// L6958
  hls::stream< int8_t > &v6422 /* v6422[16] */ = A_fifo[0][3];	// L6960
  hls::stream< int32_t > &v6423 /* v6423[16] */ = P_fifo[3][0];	// L6961
  hls::stream< int8_t > &v6424 /* v6424[16] */ = A_fifo[0][4];	// L6965
  hls::stream< int32_t > &v6425 /* v6425[16] */ = P_fifo[3][1];	// L6966
  PE_kernel_ws_gemm_3_0(v6422, v6423, v6424, v6425, v6401, 0, 3);	// L6967
  hls::stream< int8_t > &v6426 /* v6426[16] */ = A_fifo[0][4];	// L6969
  hls::stream< int32_t > &v6427 /* v6427[16] */ = P_fifo[4][0];	// L6970
  hls::stream< int8_t > &v6428 /* v6428[16] */ = A_fifo[0][5];	// L6974
  hls::stream< int32_t > &v6429 /* v6429[16] */ = P_fifo[4][1];	// L6975
  PE_kernel_ws_gemm_4_0(v6426, v6427, v6428, v6429, v6401, 0, 4);	// L6976
  hls::stream< int8_t > &v6430 /* v6430[16] */ = A_fifo[0][5];	// L6978
  hls::stream< int32_t > &v6431 /* v6431[16] */ = P_fifo[5][0];	// L6979
  hls::stream< int8_t > &v6432 /* v6432[16] */ = A_fifo[0][6];	// L6983
  hls::stream< int32_t > &v6433 /* v6433[16] */ = P_fifo[5][1];	// L6984
  PE_kernel_ws_gemm_5_0(v6430, v6431, v6432, v6433, v6401, 0, 5);	// L6985
  hls::stream< int8_t > &v6434 /* v6434[16] */ = A_fifo[0][6];	// L6987
  hls::stream< int32_t > &v6435 /* v6435[16] */ = P_fifo[6][0];	// L6988
  hls::stream< int8_t > &v6436 /* v6436[16] */ = A_fifo[0][7];	// L6992
  hls::stream< int32_t > &v6437 /* v6437[16] */ = P_fifo[6][1];	// L6993
  PE_kernel_ws_gemm_6_0(v6434, v6435, v6436, v6437, v6401, 0, 6);	// L6994
  hls::stream< int8_t > &v6438 /* v6438[16] */ = A_fifo[0][7];	// L6996
  hls::stream< int32_t > &v6439 /* v6439[16] */ = P_fifo[7][0];	// L6997
  hls::stream< int8_t > &v6440 /* v6440[16] */ = A_fifo[0][8];	// L7001
  hls::stream< int32_t > &v6441 /* v6441[16] */ = P_fifo[7][1];	// L7002
  PE_kernel_ws_gemm_7_0(v6438, v6439, v6440, v6441, v6401, 0, 7);	// L7003
  hls::stream< int8_t > &v6442 /* v6442[16] */ = A_fifo[0][8];	// L7005
  hls::stream< int32_t > &v6443 /* v6443[16] */ = P_fifo[8][0];	// L7006
  hls::stream< int8_t > &v6444 /* v6444[16] */ = A_fifo[0][9];	// L7010
  hls::stream< int32_t > &v6445 /* v6445[16] */ = P_fifo[8][1];	// L7011
  PE_kernel_ws_gemm_8_0(v6442, v6443, v6444, v6445, v6401, 0, 8);	// L7012
  hls::stream< int8_t > &v6446 /* v6446[16] */ = A_fifo[0][9];	// L7014
  hls::stream< int32_t > &v6447 /* v6447[16] */ = P_fifo[9][0];	// L7015
  hls::stream< int8_t > &v6448 /* v6448[16] */ = A_fifo[0][10];	// L7019
  hls::stream< int32_t > &v6449 /* v6449[16] */ = P_fifo[9][1];	// L7020
  PE_kernel_ws_gemm_9_0(v6446, v6447, v6448, v6449, v6401, 0, 9);	// L7021
  hls::stream< int8_t > &v6450 /* v6450[16] */ = A_fifo[0][10];	// L7023
  hls::stream< int32_t > &v6451 /* v6451[16] */ = P_fifo[10][0];	// L7024
  hls::stream< int8_t > &v6452 /* v6452[16] */ = A_fifo[0][11];	// L7028
  hls::stream< int32_t > &v6453 /* v6453[16] */ = P_fifo[10][1];	// L7029
  PE_kernel_ws_gemm_10_0(v6450, v6451, v6452, v6453, v6401, 0, 10);	// L7030
  hls::stream< int8_t > &v6454 /* v6454[16] */ = A_fifo[0][11];	// L7032
  hls::stream< int32_t > &v6455 /* v6455[16] */ = P_fifo[11][0];	// L7033
  hls::stream< int8_t > &v6456 /* v6456[16] */ = A_fifo[0][12];	// L7037
  hls::stream< int32_t > &v6457 /* v6457[16] */ = P_fifo[11][1];	// L7038
  PE_kernel_ws_gemm_11_0(v6454, v6455, v6456, v6457, v6401, 0, 11);	// L7039
  hls::stream< int8_t > &v6458 /* v6458[16] */ = A_fifo[0][12];	// L7041
  hls::stream< int32_t > &v6459 /* v6459[16] */ = P_fifo[12][0];	// L7042
  hls::stream< int8_t > &v6460 /* v6460[16] */ = A_fifo[0][13];	// L7046
  hls::stream< int32_t > &v6461 /* v6461[16] */ = P_fifo[12][1];	// L7047
  PE_kernel_ws_gemm_12_0(v6458, v6459, v6460, v6461, v6401, 0, 12);	// L7048
  hls::stream< int8_t > &v6462 /* v6462[16] */ = A_fifo[0][13];	// L7050
  hls::stream< int32_t > &v6463 /* v6463[16] */ = P_fifo[13][0];	// L7051
  hls::stream< int8_t > &v6464 /* v6464[16] */ = A_fifo[0][14];	// L7055
  hls::stream< int32_t > &v6465 /* v6465[16] */ = P_fifo[13][1];	// L7056
  PE_kernel_ws_gemm_13_0(v6462, v6463, v6464, v6465, v6401, 0, 13);	// L7057
  hls::stream< int8_t > &v6466 /* v6466[16] */ = A_fifo[0][14];	// L7059
  hls::stream< int32_t > &v6467 /* v6467[16] */ = P_fifo[14][0];	// L7060
  hls::stream< int8_t > &v6468 /* v6468[16] */ = A_fifo[0][15];	// L7064
  hls::stream< int32_t > &v6469 /* v6469[16] */ = P_fifo[14][1];	// L7065
  PE_kernel_ws_gemm_14_0(v6466, v6467, v6468, v6469, v6401, 0, 14);	// L7066
  hls::stream< int8_t > &v6470 /* v6470[16] */ = A_fifo[0][15];	// L7068
  hls::stream< int32_t > &v6471 /* v6471[16] */ = P_fifo[15][0];	// L7069
  hls::stream< int8_t > &v6472 /* v6472[16] */ = A_fifo[0][16];	// L7073
  hls::stream< int32_t > &v6473 /* v6473[16] */ = P_fifo[15][1];	// L7074
  PE_kernel_ws_gemm_15_0(v6470, v6471, v6472, v6473, v6401, 0, 15);	// L7075
  hls::stream< int8_t > &v6474 /* v6474[16] */ = A_fifo[1][0];	// L7076
  hls::stream< int32_t > &v6475 /* v6475[16] */ = P_fifo[0][1];	// L7077
  hls::stream< int8_t > &v6476 /* v6476[16] */ = A_fifo[1][1];	// L7078
  hls::stream< int32_t > &v6477 /* v6477[16] */ = P_fifo[0][2];	// L7079
  PE_kernel_ws_gemm_0_1(v6474, v6475, v6476, v6477, v6401, 1, 0);	// L7080
  hls::stream< int8_t > &v6478 /* v6478[16] */ = A_fifo[1][1];	// L7081
  hls::stream< int32_t > &v6479 /* v6479[16] */ = P_fifo[1][1];	// L7082
  hls::stream< int8_t > &v6480 /* v6480[16] */ = A_fifo[1][2];	// L7083
  hls::stream< int32_t > &v6481 /* v6481[16] */ = P_fifo[1][2];	// L7084
  PE_kernel_ws_gemm_1_1(v6478, v6479, v6480, v6481, v6401, 1, 1);	// L7085
  hls::stream< int8_t > &v6482 /* v6482[16] */ = A_fifo[1][2];	// L7086
  hls::stream< int32_t > &v6483 /* v6483[16] */ = P_fifo[2][1];	// L7087
  hls::stream< int8_t > &v6484 /* v6484[16] */ = A_fifo[1][3];	// L7088
  hls::stream< int32_t > &v6485 /* v6485[16] */ = P_fifo[2][2];	// L7089
  PE_kernel_ws_gemm_2_1(v6482, v6483, v6484, v6485, v6401, 1, 2);	// L7090
  hls::stream< int8_t > &v6486 /* v6486[16] */ = A_fifo[1][3];	// L7091
  hls::stream< int32_t > &v6487 /* v6487[16] */ = P_fifo[3][1];	// L7092
  hls::stream< int8_t > &v6488 /* v6488[16] */ = A_fifo[1][4];	// L7093
  hls::stream< int32_t > &v6489 /* v6489[16] */ = P_fifo[3][2];	// L7094
  PE_kernel_ws_gemm_3_1(v6486, v6487, v6488, v6489, v6401, 1, 3);	// L7095
  hls::stream< int8_t > &v6490 /* v6490[16] */ = A_fifo[1][4];	// L7096
  hls::stream< int32_t > &v6491 /* v6491[16] */ = P_fifo[4][1];	// L7097
  hls::stream< int8_t > &v6492 /* v6492[16] */ = A_fifo[1][5];	// L7098
  hls::stream< int32_t > &v6493 /* v6493[16] */ = P_fifo[4][2];	// L7099
  PE_kernel_ws_gemm_4_1(v6490, v6491, v6492, v6493, v6401, 1, 4);	// L7100
  hls::stream< int8_t > &v6494 /* v6494[16] */ = A_fifo[1][5];	// L7101
  hls::stream< int32_t > &v6495 /* v6495[16] */ = P_fifo[5][1];	// L7102
  hls::stream< int8_t > &v6496 /* v6496[16] */ = A_fifo[1][6];	// L7103
  hls::stream< int32_t > &v6497 /* v6497[16] */ = P_fifo[5][2];	// L7104
  PE_kernel_ws_gemm_5_1(v6494, v6495, v6496, v6497, v6401, 1, 5);	// L7105
  hls::stream< int8_t > &v6498 /* v6498[16] */ = A_fifo[1][6];	// L7106
  hls::stream< int32_t > &v6499 /* v6499[16] */ = P_fifo[6][1];	// L7107
  hls::stream< int8_t > &v6500 /* v6500[16] */ = A_fifo[1][7];	// L7108
  hls::stream< int32_t > &v6501 /* v6501[16] */ = P_fifo[6][2];	// L7109
  PE_kernel_ws_gemm_6_1(v6498, v6499, v6500, v6501, v6401, 1, 6);	// L7110
  hls::stream< int8_t > &v6502 /* v6502[16] */ = A_fifo[1][7];	// L7111
  hls::stream< int32_t > &v6503 /* v6503[16] */ = P_fifo[7][1];	// L7112
  hls::stream< int8_t > &v6504 /* v6504[16] */ = A_fifo[1][8];	// L7113
  hls::stream< int32_t > &v6505 /* v6505[16] */ = P_fifo[7][2];	// L7114
  PE_kernel_ws_gemm_7_1(v6502, v6503, v6504, v6505, v6401, 1, 7);	// L7115
  hls::stream< int8_t > &v6506 /* v6506[16] */ = A_fifo[1][8];	// L7116
  hls::stream< int32_t > &v6507 /* v6507[16] */ = P_fifo[8][1];	// L7117
  hls::stream< int8_t > &v6508 /* v6508[16] */ = A_fifo[1][9];	// L7118
  hls::stream< int32_t > &v6509 /* v6509[16] */ = P_fifo[8][2];	// L7119
  PE_kernel_ws_gemm_8_1(v6506, v6507, v6508, v6509, v6401, 1, 8);	// L7120
  hls::stream< int8_t > &v6510 /* v6510[16] */ = A_fifo[1][9];	// L7121
  hls::stream< int32_t > &v6511 /* v6511[16] */ = P_fifo[9][1];	// L7122
  hls::stream< int8_t > &v6512 /* v6512[16] */ = A_fifo[1][10];	// L7123
  hls::stream< int32_t > &v6513 /* v6513[16] */ = P_fifo[9][2];	// L7124
  PE_kernel_ws_gemm_9_1(v6510, v6511, v6512, v6513, v6401, 1, 9);	// L7125
  hls::stream< int8_t > &v6514 /* v6514[16] */ = A_fifo[1][10];	// L7126
  hls::stream< int32_t > &v6515 /* v6515[16] */ = P_fifo[10][1];	// L7127
  hls::stream< int8_t > &v6516 /* v6516[16] */ = A_fifo[1][11];	// L7128
  hls::stream< int32_t > &v6517 /* v6517[16] */ = P_fifo[10][2];	// L7129
  PE_kernel_ws_gemm_10_1(v6514, v6515, v6516, v6517, v6401, 1, 10);	// L7130
  hls::stream< int8_t > &v6518 /* v6518[16] */ = A_fifo[1][11];	// L7131
  hls::stream< int32_t > &v6519 /* v6519[16] */ = P_fifo[11][1];	// L7132
  hls::stream< int8_t > &v6520 /* v6520[16] */ = A_fifo[1][12];	// L7133
  hls::stream< int32_t > &v6521 /* v6521[16] */ = P_fifo[11][2];	// L7134
  PE_kernel_ws_gemm_11_1(v6518, v6519, v6520, v6521, v6401, 1, 11);	// L7135
  hls::stream< int8_t > &v6522 /* v6522[16] */ = A_fifo[1][12];	// L7136
  hls::stream< int32_t > &v6523 /* v6523[16] */ = P_fifo[12][1];	// L7137
  hls::stream< int8_t > &v6524 /* v6524[16] */ = A_fifo[1][13];	// L7138
  hls::stream< int32_t > &v6525 /* v6525[16] */ = P_fifo[12][2];	// L7139
  PE_kernel_ws_gemm_12_1(v6522, v6523, v6524, v6525, v6401, 1, 12);	// L7140
  hls::stream< int8_t > &v6526 /* v6526[16] */ = A_fifo[1][13];	// L7141
  hls::stream< int32_t > &v6527 /* v6527[16] */ = P_fifo[13][1];	// L7142
  hls::stream< int8_t > &v6528 /* v6528[16] */ = A_fifo[1][14];	// L7143
  hls::stream< int32_t > &v6529 /* v6529[16] */ = P_fifo[13][2];	// L7144
  PE_kernel_ws_gemm_13_1(v6526, v6527, v6528, v6529, v6401, 1, 13);	// L7145
  hls::stream< int8_t > &v6530 /* v6530[16] */ = A_fifo[1][14];	// L7146
  hls::stream< int32_t > &v6531 /* v6531[16] */ = P_fifo[14][1];	// L7147
  hls::stream< int8_t > &v6532 /* v6532[16] */ = A_fifo[1][15];	// L7148
  hls::stream< int32_t > &v6533 /* v6533[16] */ = P_fifo[14][2];	// L7149
  PE_kernel_ws_gemm_14_1(v6530, v6531, v6532, v6533, v6401, 1, 14);	// L7150
  hls::stream< int8_t > &v6534 /* v6534[16] */ = A_fifo[1][15];	// L7151
  hls::stream< int32_t > &v6535 /* v6535[16] */ = P_fifo[15][1];	// L7152
  hls::stream< int8_t > &v6536 /* v6536[16] */ = A_fifo[1][16];	// L7153
  hls::stream< int32_t > &v6537 /* v6537[16] */ = P_fifo[15][2];	// L7154
  PE_kernel_ws_gemm_15_1(v6534, v6535, v6536, v6537, v6401, 1, 15);	// L7155
  hls::stream< int8_t > &v6538 /* v6538[16] */ = A_fifo[2][0];	// L7156
  hls::stream< int32_t > &v6539 /* v6539[16] */ = P_fifo[0][2];	// L7157
  hls::stream< int8_t > &v6540 /* v6540[16] */ = A_fifo[2][1];	// L7158
  hls::stream< int32_t > &v6541 /* v6541[16] */ = P_fifo[0][3];	// L7159
  PE_kernel_ws_gemm_0_2(v6538, v6539, v6540, v6541, v6401, 2, 0);	// L7160
  hls::stream< int8_t > &v6542 /* v6542[16] */ = A_fifo[2][1];	// L7161
  hls::stream< int32_t > &v6543 /* v6543[16] */ = P_fifo[1][2];	// L7162
  hls::stream< int8_t > &v6544 /* v6544[16] */ = A_fifo[2][2];	// L7163
  hls::stream< int32_t > &v6545 /* v6545[16] */ = P_fifo[1][3];	// L7164
  PE_kernel_ws_gemm_1_2(v6542, v6543, v6544, v6545, v6401, 2, 1);	// L7165
  hls::stream< int8_t > &v6546 /* v6546[16] */ = A_fifo[2][2];	// L7166
  hls::stream< int32_t > &v6547 /* v6547[16] */ = P_fifo[2][2];	// L7167
  hls::stream< int8_t > &v6548 /* v6548[16] */ = A_fifo[2][3];	// L7168
  hls::stream< int32_t > &v6549 /* v6549[16] */ = P_fifo[2][3];	// L7169
  PE_kernel_ws_gemm_2_2(v6546, v6547, v6548, v6549, v6401, 2, 2);	// L7170
  hls::stream< int8_t > &v6550 /* v6550[16] */ = A_fifo[2][3];	// L7171
  hls::stream< int32_t > &v6551 /* v6551[16] */ = P_fifo[3][2];	// L7172
  hls::stream< int8_t > &v6552 /* v6552[16] */ = A_fifo[2][4];	// L7173
  hls::stream< int32_t > &v6553 /* v6553[16] */ = P_fifo[3][3];	// L7174
  PE_kernel_ws_gemm_3_2(v6550, v6551, v6552, v6553, v6401, 2, 3);	// L7175
  hls::stream< int8_t > &v6554 /* v6554[16] */ = A_fifo[2][4];	// L7176
  hls::stream< int32_t > &v6555 /* v6555[16] */ = P_fifo[4][2];	// L7177
  hls::stream< int8_t > &v6556 /* v6556[16] */ = A_fifo[2][5];	// L7178
  hls::stream< int32_t > &v6557 /* v6557[16] */ = P_fifo[4][3];	// L7179
  PE_kernel_ws_gemm_4_2(v6554, v6555, v6556, v6557, v6401, 2, 4);	// L7180
  hls::stream< int8_t > &v6558 /* v6558[16] */ = A_fifo[2][5];	// L7181
  hls::stream< int32_t > &v6559 /* v6559[16] */ = P_fifo[5][2];	// L7182
  hls::stream< int8_t > &v6560 /* v6560[16] */ = A_fifo[2][6];	// L7183
  hls::stream< int32_t > &v6561 /* v6561[16] */ = P_fifo[5][3];	// L7184
  PE_kernel_ws_gemm_5_2(v6558, v6559, v6560, v6561, v6401, 2, 5);	// L7185
  hls::stream< int8_t > &v6562 /* v6562[16] */ = A_fifo[2][6];	// L7186
  hls::stream< int32_t > &v6563 /* v6563[16] */ = P_fifo[6][2];	// L7187
  hls::stream< int8_t > &v6564 /* v6564[16] */ = A_fifo[2][7];	// L7188
  hls::stream< int32_t > &v6565 /* v6565[16] */ = P_fifo[6][3];	// L7189
  PE_kernel_ws_gemm_6_2(v6562, v6563, v6564, v6565, v6401, 2, 6);	// L7190
  hls::stream< int8_t > &v6566 /* v6566[16] */ = A_fifo[2][7];	// L7191
  hls::stream< int32_t > &v6567 /* v6567[16] */ = P_fifo[7][2];	// L7192
  hls::stream< int8_t > &v6568 /* v6568[16] */ = A_fifo[2][8];	// L7193
  hls::stream< int32_t > &v6569 /* v6569[16] */ = P_fifo[7][3];	// L7194
  PE_kernel_ws_gemm_7_2(v6566, v6567, v6568, v6569, v6401, 2, 7);	// L7195
  hls::stream< int8_t > &v6570 /* v6570[16] */ = A_fifo[2][8];	// L7196
  hls::stream< int32_t > &v6571 /* v6571[16] */ = P_fifo[8][2];	// L7197
  hls::stream< int8_t > &v6572 /* v6572[16] */ = A_fifo[2][9];	// L7198
  hls::stream< int32_t > &v6573 /* v6573[16] */ = P_fifo[8][3];	// L7199
  PE_kernel_ws_gemm_8_2(v6570, v6571, v6572, v6573, v6401, 2, 8);	// L7200
  hls::stream< int8_t > &v6574 /* v6574[16] */ = A_fifo[2][9];	// L7201
  hls::stream< int32_t > &v6575 /* v6575[16] */ = P_fifo[9][2];	// L7202
  hls::stream< int8_t > &v6576 /* v6576[16] */ = A_fifo[2][10];	// L7203
  hls::stream< int32_t > &v6577 /* v6577[16] */ = P_fifo[9][3];	// L7204
  PE_kernel_ws_gemm_9_2(v6574, v6575, v6576, v6577, v6401, 2, 9);	// L7205
  hls::stream< int8_t > &v6578 /* v6578[16] */ = A_fifo[2][10];	// L7206
  hls::stream< int32_t > &v6579 /* v6579[16] */ = P_fifo[10][2];	// L7207
  hls::stream< int8_t > &v6580 /* v6580[16] */ = A_fifo[2][11];	// L7208
  hls::stream< int32_t > &v6581 /* v6581[16] */ = P_fifo[10][3];	// L7209
  PE_kernel_ws_gemm_10_2(v6578, v6579, v6580, v6581, v6401, 2, 10);	// L7210
  hls::stream< int8_t > &v6582 /* v6582[16] */ = A_fifo[2][11];	// L7211
  hls::stream< int32_t > &v6583 /* v6583[16] */ = P_fifo[11][2];	// L7212
  hls::stream< int8_t > &v6584 /* v6584[16] */ = A_fifo[2][12];	// L7213
  hls::stream< int32_t > &v6585 /* v6585[16] */ = P_fifo[11][3];	// L7214
  PE_kernel_ws_gemm_11_2(v6582, v6583, v6584, v6585, v6401, 2, 11);	// L7215
  hls::stream< int8_t > &v6586 /* v6586[16] */ = A_fifo[2][12];	// L7216
  hls::stream< int32_t > &v6587 /* v6587[16] */ = P_fifo[12][2];	// L7217
  hls::stream< int8_t > &v6588 /* v6588[16] */ = A_fifo[2][13];	// L7218
  hls::stream< int32_t > &v6589 /* v6589[16] */ = P_fifo[12][3];	// L7219
  PE_kernel_ws_gemm_12_2(v6586, v6587, v6588, v6589, v6401, 2, 12);	// L7220
  hls::stream< int8_t > &v6590 /* v6590[16] */ = A_fifo[2][13];	// L7221
  hls::stream< int32_t > &v6591 /* v6591[16] */ = P_fifo[13][2];	// L7222
  hls::stream< int8_t > &v6592 /* v6592[16] */ = A_fifo[2][14];	// L7223
  hls::stream< int32_t > &v6593 /* v6593[16] */ = P_fifo[13][3];	// L7224
  PE_kernel_ws_gemm_13_2(v6590, v6591, v6592, v6593, v6401, 2, 13);	// L7225
  hls::stream< int8_t > &v6594 /* v6594[16] */ = A_fifo[2][14];	// L7226
  hls::stream< int32_t > &v6595 /* v6595[16] */ = P_fifo[14][2];	// L7227
  hls::stream< int8_t > &v6596 /* v6596[16] */ = A_fifo[2][15];	// L7228
  hls::stream< int32_t > &v6597 /* v6597[16] */ = P_fifo[14][3];	// L7229
  PE_kernel_ws_gemm_14_2(v6594, v6595, v6596, v6597, v6401, 2, 14);	// L7230
  hls::stream< int8_t > &v6598 /* v6598[16] */ = A_fifo[2][15];	// L7231
  hls::stream< int32_t > &v6599 /* v6599[16] */ = P_fifo[15][2];	// L7232
  hls::stream< int8_t > &v6600 /* v6600[16] */ = A_fifo[2][16];	// L7233
  hls::stream< int32_t > &v6601 /* v6601[16] */ = P_fifo[15][3];	// L7234
  PE_kernel_ws_gemm_15_2(v6598, v6599, v6600, v6601, v6401, 2, 15);	// L7235
  hls::stream< int8_t > &v6602 /* v6602[16] */ = A_fifo[3][0];	// L7236
  hls::stream< int32_t > &v6603 /* v6603[16] */ = P_fifo[0][3];	// L7237
  hls::stream< int8_t > &v6604 /* v6604[16] */ = A_fifo[3][1];	// L7238
  hls::stream< int32_t > &v6605 /* v6605[16] */ = P_fifo[0][4];	// L7239
  PE_kernel_ws_gemm_0_3(v6602, v6603, v6604, v6605, v6401, 3, 0);	// L7240
  hls::stream< int8_t > &v6606 /* v6606[16] */ = A_fifo[3][1];	// L7241
  hls::stream< int32_t > &v6607 /* v6607[16] */ = P_fifo[1][3];	// L7242
  hls::stream< int8_t > &v6608 /* v6608[16] */ = A_fifo[3][2];	// L7243
  hls::stream< int32_t > &v6609 /* v6609[16] */ = P_fifo[1][4];	// L7244
  PE_kernel_ws_gemm_1_3(v6606, v6607, v6608, v6609, v6401, 3, 1);	// L7245
  hls::stream< int8_t > &v6610 /* v6610[16] */ = A_fifo[3][2];	// L7246
  hls::stream< int32_t > &v6611 /* v6611[16] */ = P_fifo[2][3];	// L7247
  hls::stream< int8_t > &v6612 /* v6612[16] */ = A_fifo[3][3];	// L7248
  hls::stream< int32_t > &v6613 /* v6613[16] */ = P_fifo[2][4];	// L7249
  PE_kernel_ws_gemm_2_3(v6610, v6611, v6612, v6613, v6401, 3, 2);	// L7250
  hls::stream< int8_t > &v6614 /* v6614[16] */ = A_fifo[3][3];	// L7251
  hls::stream< int32_t > &v6615 /* v6615[16] */ = P_fifo[3][3];	// L7252
  hls::stream< int8_t > &v6616 /* v6616[16] */ = A_fifo[3][4];	// L7253
  hls::stream< int32_t > &v6617 /* v6617[16] */ = P_fifo[3][4];	// L7254
  PE_kernel_ws_gemm_3_3(v6614, v6615, v6616, v6617, v6401, 3, 3);	// L7255
  hls::stream< int8_t > &v6618 /* v6618[16] */ = A_fifo[3][4];	// L7256
  hls::stream< int32_t > &v6619 /* v6619[16] */ = P_fifo[4][3];	// L7257
  hls::stream< int8_t > &v6620 /* v6620[16] */ = A_fifo[3][5];	// L7258
  hls::stream< int32_t > &v6621 /* v6621[16] */ = P_fifo[4][4];	// L7259
  PE_kernel_ws_gemm_4_3(v6618, v6619, v6620, v6621, v6401, 3, 4);	// L7260
  hls::stream< int8_t > &v6622 /* v6622[16] */ = A_fifo[3][5];	// L7261
  hls::stream< int32_t > &v6623 /* v6623[16] */ = P_fifo[5][3];	// L7262
  hls::stream< int8_t > &v6624 /* v6624[16] */ = A_fifo[3][6];	// L7263
  hls::stream< int32_t > &v6625 /* v6625[16] */ = P_fifo[5][4];	// L7264
  PE_kernel_ws_gemm_5_3(v6622, v6623, v6624, v6625, v6401, 3, 5);	// L7265
  hls::stream< int8_t > &v6626 /* v6626[16] */ = A_fifo[3][6];	// L7266
  hls::stream< int32_t > &v6627 /* v6627[16] */ = P_fifo[6][3];	// L7267
  hls::stream< int8_t > &v6628 /* v6628[16] */ = A_fifo[3][7];	// L7268
  hls::stream< int32_t > &v6629 /* v6629[16] */ = P_fifo[6][4];	// L7269
  PE_kernel_ws_gemm_6_3(v6626, v6627, v6628, v6629, v6401, 3, 6);	// L7270
  hls::stream< int8_t > &v6630 /* v6630[16] */ = A_fifo[3][7];	// L7271
  hls::stream< int32_t > &v6631 /* v6631[16] */ = P_fifo[7][3];	// L7272
  hls::stream< int8_t > &v6632 /* v6632[16] */ = A_fifo[3][8];	// L7273
  hls::stream< int32_t > &v6633 /* v6633[16] */ = P_fifo[7][4];	// L7274
  PE_kernel_ws_gemm_7_3(v6630, v6631, v6632, v6633, v6401, 3, 7);	// L7275
  hls::stream< int8_t > &v6634 /* v6634[16] */ = A_fifo[3][8];	// L7276
  hls::stream< int32_t > &v6635 /* v6635[16] */ = P_fifo[8][3];	// L7277
  hls::stream< int8_t > &v6636 /* v6636[16] */ = A_fifo[3][9];	// L7278
  hls::stream< int32_t > &v6637 /* v6637[16] */ = P_fifo[8][4];	// L7279
  PE_kernel_ws_gemm_8_3(v6634, v6635, v6636, v6637, v6401, 3, 8);	// L7280
  hls::stream< int8_t > &v6638 /* v6638[16] */ = A_fifo[3][9];	// L7281
  hls::stream< int32_t > &v6639 /* v6639[16] */ = P_fifo[9][3];	// L7282
  hls::stream< int8_t > &v6640 /* v6640[16] */ = A_fifo[3][10];	// L7283
  hls::stream< int32_t > &v6641 /* v6641[16] */ = P_fifo[9][4];	// L7284
  PE_kernel_ws_gemm_9_3(v6638, v6639, v6640, v6641, v6401, 3, 9);	// L7285
  hls::stream< int8_t > &v6642 /* v6642[16] */ = A_fifo[3][10];	// L7286
  hls::stream< int32_t > &v6643 /* v6643[16] */ = P_fifo[10][3];	// L7287
  hls::stream< int8_t > &v6644 /* v6644[16] */ = A_fifo[3][11];	// L7288
  hls::stream< int32_t > &v6645 /* v6645[16] */ = P_fifo[10][4];	// L7289
  PE_kernel_ws_gemm_10_3(v6642, v6643, v6644, v6645, v6401, 3, 10);	// L7290
  hls::stream< int8_t > &v6646 /* v6646[16] */ = A_fifo[3][11];	// L7291
  hls::stream< int32_t > &v6647 /* v6647[16] */ = P_fifo[11][3];	// L7292
  hls::stream< int8_t > &v6648 /* v6648[16] */ = A_fifo[3][12];	// L7293
  hls::stream< int32_t > &v6649 /* v6649[16] */ = P_fifo[11][4];	// L7294
  PE_kernel_ws_gemm_11_3(v6646, v6647, v6648, v6649, v6401, 3, 11);	// L7295
  hls::stream< int8_t > &v6650 /* v6650[16] */ = A_fifo[3][12];	// L7296
  hls::stream< int32_t > &v6651 /* v6651[16] */ = P_fifo[12][3];	// L7297
  hls::stream< int8_t > &v6652 /* v6652[16] */ = A_fifo[3][13];	// L7298
  hls::stream< int32_t > &v6653 /* v6653[16] */ = P_fifo[12][4];	// L7299
  PE_kernel_ws_gemm_12_3(v6650, v6651, v6652, v6653, v6401, 3, 12);	// L7300
  hls::stream< int8_t > &v6654 /* v6654[16] */ = A_fifo[3][13];	// L7301
  hls::stream< int32_t > &v6655 /* v6655[16] */ = P_fifo[13][3];	// L7302
  hls::stream< int8_t > &v6656 /* v6656[16] */ = A_fifo[3][14];	// L7303
  hls::stream< int32_t > &v6657 /* v6657[16] */ = P_fifo[13][4];	// L7304
  PE_kernel_ws_gemm_13_3(v6654, v6655, v6656, v6657, v6401, 3, 13);	// L7305
  hls::stream< int8_t > &v6658 /* v6658[16] */ = A_fifo[3][14];	// L7306
  hls::stream< int32_t > &v6659 /* v6659[16] */ = P_fifo[14][3];	// L7307
  hls::stream< int8_t > &v6660 /* v6660[16] */ = A_fifo[3][15];	// L7308
  hls::stream< int32_t > &v6661 /* v6661[16] */ = P_fifo[14][4];	// L7309
  PE_kernel_ws_gemm_14_3(v6658, v6659, v6660, v6661, v6401, 3, 14);	// L7310
  hls::stream< int8_t > &v6662 /* v6662[16] */ = A_fifo[3][15];	// L7311
  hls::stream< int32_t > &v6663 /* v6663[16] */ = P_fifo[15][3];	// L7312
  hls::stream< int8_t > &v6664 /* v6664[16] */ = A_fifo[3][16];	// L7313
  hls::stream< int32_t > &v6665 /* v6665[16] */ = P_fifo[15][4];	// L7314
  PE_kernel_ws_gemm_15_3(v6662, v6663, v6664, v6665, v6401, 3, 15);	// L7315
  hls::stream< int8_t > &v6666 /* v6666[16] */ = A_fifo[4][0];	// L7316
  hls::stream< int32_t > &v6667 /* v6667[16] */ = P_fifo[0][4];	// L7317
  hls::stream< int8_t > &v6668 /* v6668[16] */ = A_fifo[4][1];	// L7318
  hls::stream< int32_t > &v6669 /* v6669[16] */ = P_fifo[0][5];	// L7319
  PE_kernel_ws_gemm_0_4(v6666, v6667, v6668, v6669, v6401, 4, 0);	// L7320
  hls::stream< int8_t > &v6670 /* v6670[16] */ = A_fifo[4][1];	// L7321
  hls::stream< int32_t > &v6671 /* v6671[16] */ = P_fifo[1][4];	// L7322
  hls::stream< int8_t > &v6672 /* v6672[16] */ = A_fifo[4][2];	// L7323
  hls::stream< int32_t > &v6673 /* v6673[16] */ = P_fifo[1][5];	// L7324
  PE_kernel_ws_gemm_1_4(v6670, v6671, v6672, v6673, v6401, 4, 1);	// L7325
  hls::stream< int8_t > &v6674 /* v6674[16] */ = A_fifo[4][2];	// L7326
  hls::stream< int32_t > &v6675 /* v6675[16] */ = P_fifo[2][4];	// L7327
  hls::stream< int8_t > &v6676 /* v6676[16] */ = A_fifo[4][3];	// L7328
  hls::stream< int32_t > &v6677 /* v6677[16] */ = P_fifo[2][5];	// L7329
  PE_kernel_ws_gemm_2_4(v6674, v6675, v6676, v6677, v6401, 4, 2);	// L7330
  hls::stream< int8_t > &v6678 /* v6678[16] */ = A_fifo[4][3];	// L7331
  hls::stream< int32_t > &v6679 /* v6679[16] */ = P_fifo[3][4];	// L7332
  hls::stream< int8_t > &v6680 /* v6680[16] */ = A_fifo[4][4];	// L7333
  hls::stream< int32_t > &v6681 /* v6681[16] */ = P_fifo[3][5];	// L7334
  PE_kernel_ws_gemm_3_4(v6678, v6679, v6680, v6681, v6401, 4, 3);	// L7335
  hls::stream< int8_t > &v6682 /* v6682[16] */ = A_fifo[4][4];	// L7336
  hls::stream< int32_t > &v6683 /* v6683[16] */ = P_fifo[4][4];	// L7337
  hls::stream< int8_t > &v6684 /* v6684[16] */ = A_fifo[4][5];	// L7338
  hls::stream< int32_t > &v6685 /* v6685[16] */ = P_fifo[4][5];	// L7339
  PE_kernel_ws_gemm_4_4(v6682, v6683, v6684, v6685, v6401, 4, 4);	// L7340
  hls::stream< int8_t > &v6686 /* v6686[16] */ = A_fifo[4][5];	// L7341
  hls::stream< int32_t > &v6687 /* v6687[16] */ = P_fifo[5][4];	// L7342
  hls::stream< int8_t > &v6688 /* v6688[16] */ = A_fifo[4][6];	// L7343
  hls::stream< int32_t > &v6689 /* v6689[16] */ = P_fifo[5][5];	// L7344
  PE_kernel_ws_gemm_5_4(v6686, v6687, v6688, v6689, v6401, 4, 5);	// L7345
  hls::stream< int8_t > &v6690 /* v6690[16] */ = A_fifo[4][6];	// L7346
  hls::stream< int32_t > &v6691 /* v6691[16] */ = P_fifo[6][4];	// L7347
  hls::stream< int8_t > &v6692 /* v6692[16] */ = A_fifo[4][7];	// L7348
  hls::stream< int32_t > &v6693 /* v6693[16] */ = P_fifo[6][5];	// L7349
  PE_kernel_ws_gemm_6_4(v6690, v6691, v6692, v6693, v6401, 4, 6);	// L7350
  hls::stream< int8_t > &v6694 /* v6694[16] */ = A_fifo[4][7];	// L7351
  hls::stream< int32_t > &v6695 /* v6695[16] */ = P_fifo[7][4];	// L7352
  hls::stream< int8_t > &v6696 /* v6696[16] */ = A_fifo[4][8];	// L7353
  hls::stream< int32_t > &v6697 /* v6697[16] */ = P_fifo[7][5];	// L7354
  PE_kernel_ws_gemm_7_4(v6694, v6695, v6696, v6697, v6401, 4, 7);	// L7355
  hls::stream< int8_t > &v6698 /* v6698[16] */ = A_fifo[4][8];	// L7356
  hls::stream< int32_t > &v6699 /* v6699[16] */ = P_fifo[8][4];	// L7357
  hls::stream< int8_t > &v6700 /* v6700[16] */ = A_fifo[4][9];	// L7358
  hls::stream< int32_t > &v6701 /* v6701[16] */ = P_fifo[8][5];	// L7359
  PE_kernel_ws_gemm_8_4(v6698, v6699, v6700, v6701, v6401, 4, 8);	// L7360
  hls::stream< int8_t > &v6702 /* v6702[16] */ = A_fifo[4][9];	// L7361
  hls::stream< int32_t > &v6703 /* v6703[16] */ = P_fifo[9][4];	// L7362
  hls::stream< int8_t > &v6704 /* v6704[16] */ = A_fifo[4][10];	// L7363
  hls::stream< int32_t > &v6705 /* v6705[16] */ = P_fifo[9][5];	// L7364
  PE_kernel_ws_gemm_9_4(v6702, v6703, v6704, v6705, v6401, 4, 9);	// L7365
  hls::stream< int8_t > &v6706 /* v6706[16] */ = A_fifo[4][10];	// L7366
  hls::stream< int32_t > &v6707 /* v6707[16] */ = P_fifo[10][4];	// L7367
  hls::stream< int8_t > &v6708 /* v6708[16] */ = A_fifo[4][11];	// L7368
  hls::stream< int32_t > &v6709 /* v6709[16] */ = P_fifo[10][5];	// L7369
  PE_kernel_ws_gemm_10_4(v6706, v6707, v6708, v6709, v6401, 4, 10);	// L7370
  hls::stream< int8_t > &v6710 /* v6710[16] */ = A_fifo[4][11];	// L7371
  hls::stream< int32_t > &v6711 /* v6711[16] */ = P_fifo[11][4];	// L7372
  hls::stream< int8_t > &v6712 /* v6712[16] */ = A_fifo[4][12];	// L7373
  hls::stream< int32_t > &v6713 /* v6713[16] */ = P_fifo[11][5];	// L7374
  PE_kernel_ws_gemm_11_4(v6710, v6711, v6712, v6713, v6401, 4, 11);	// L7375
  hls::stream< int8_t > &v6714 /* v6714[16] */ = A_fifo[4][12];	// L7376
  hls::stream< int32_t > &v6715 /* v6715[16] */ = P_fifo[12][4];	// L7377
  hls::stream< int8_t > &v6716 /* v6716[16] */ = A_fifo[4][13];	// L7378
  hls::stream< int32_t > &v6717 /* v6717[16] */ = P_fifo[12][5];	// L7379
  PE_kernel_ws_gemm_12_4(v6714, v6715, v6716, v6717, v6401, 4, 12);	// L7380
  hls::stream< int8_t > &v6718 /* v6718[16] */ = A_fifo[4][13];	// L7381
  hls::stream< int32_t > &v6719 /* v6719[16] */ = P_fifo[13][4];	// L7382
  hls::stream< int8_t > &v6720 /* v6720[16] */ = A_fifo[4][14];	// L7383
  hls::stream< int32_t > &v6721 /* v6721[16] */ = P_fifo[13][5];	// L7384
  PE_kernel_ws_gemm_13_4(v6718, v6719, v6720, v6721, v6401, 4, 13);	// L7385
  hls::stream< int8_t > &v6722 /* v6722[16] */ = A_fifo[4][14];	// L7386
  hls::stream< int32_t > &v6723 /* v6723[16] */ = P_fifo[14][4];	// L7387
  hls::stream< int8_t > &v6724 /* v6724[16] */ = A_fifo[4][15];	// L7388
  hls::stream< int32_t > &v6725 /* v6725[16] */ = P_fifo[14][5];	// L7389
  PE_kernel_ws_gemm_14_4(v6722, v6723, v6724, v6725, v6401, 4, 14);	// L7390
  hls::stream< int8_t > &v6726 /* v6726[16] */ = A_fifo[4][15];	// L7391
  hls::stream< int32_t > &v6727 /* v6727[16] */ = P_fifo[15][4];	// L7392
  hls::stream< int8_t > &v6728 /* v6728[16] */ = A_fifo[4][16];	// L7393
  hls::stream< int32_t > &v6729 /* v6729[16] */ = P_fifo[15][5];	// L7394
  PE_kernel_ws_gemm_15_4(v6726, v6727, v6728, v6729, v6401, 4, 15);	// L7395
  hls::stream< int8_t > &v6730 /* v6730[16] */ = A_fifo[5][0];	// L7396
  hls::stream< int32_t > &v6731 /* v6731[16] */ = P_fifo[0][5];	// L7397
  hls::stream< int8_t > &v6732 /* v6732[16] */ = A_fifo[5][1];	// L7398
  hls::stream< int32_t > &v6733 /* v6733[16] */ = P_fifo[0][6];	// L7399
  PE_kernel_ws_gemm_0_5(v6730, v6731, v6732, v6733, v6401, 5, 0);	// L7400
  hls::stream< int8_t > &v6734 /* v6734[16] */ = A_fifo[5][1];	// L7401
  hls::stream< int32_t > &v6735 /* v6735[16] */ = P_fifo[1][5];	// L7402
  hls::stream< int8_t > &v6736 /* v6736[16] */ = A_fifo[5][2];	// L7403
  hls::stream< int32_t > &v6737 /* v6737[16] */ = P_fifo[1][6];	// L7404
  PE_kernel_ws_gemm_1_5(v6734, v6735, v6736, v6737, v6401, 5, 1);	// L7405
  hls::stream< int8_t > &v6738 /* v6738[16] */ = A_fifo[5][2];	// L7406
  hls::stream< int32_t > &v6739 /* v6739[16] */ = P_fifo[2][5];	// L7407
  hls::stream< int8_t > &v6740 /* v6740[16] */ = A_fifo[5][3];	// L7408
  hls::stream< int32_t > &v6741 /* v6741[16] */ = P_fifo[2][6];	// L7409
  PE_kernel_ws_gemm_2_5(v6738, v6739, v6740, v6741, v6401, 5, 2);	// L7410
  hls::stream< int8_t > &v6742 /* v6742[16] */ = A_fifo[5][3];	// L7411
  hls::stream< int32_t > &v6743 /* v6743[16] */ = P_fifo[3][5];	// L7412
  hls::stream< int8_t > &v6744 /* v6744[16] */ = A_fifo[5][4];	// L7413
  hls::stream< int32_t > &v6745 /* v6745[16] */ = P_fifo[3][6];	// L7414
  PE_kernel_ws_gemm_3_5(v6742, v6743, v6744, v6745, v6401, 5, 3);	// L7415
  hls::stream< int8_t > &v6746 /* v6746[16] */ = A_fifo[5][4];	// L7416
  hls::stream< int32_t > &v6747 /* v6747[16] */ = P_fifo[4][5];	// L7417
  hls::stream< int8_t > &v6748 /* v6748[16] */ = A_fifo[5][5];	// L7418
  hls::stream< int32_t > &v6749 /* v6749[16] */ = P_fifo[4][6];	// L7419
  PE_kernel_ws_gemm_4_5(v6746, v6747, v6748, v6749, v6401, 5, 4);	// L7420
  hls::stream< int8_t > &v6750 /* v6750[16] */ = A_fifo[5][5];	// L7421
  hls::stream< int32_t > &v6751 /* v6751[16] */ = P_fifo[5][5];	// L7422
  hls::stream< int8_t > &v6752 /* v6752[16] */ = A_fifo[5][6];	// L7423
  hls::stream< int32_t > &v6753 /* v6753[16] */ = P_fifo[5][6];	// L7424
  PE_kernel_ws_gemm_5_5(v6750, v6751, v6752, v6753, v6401, 5, 5);	// L7425
  hls::stream< int8_t > &v6754 /* v6754[16] */ = A_fifo[5][6];	// L7426
  hls::stream< int32_t > &v6755 /* v6755[16] */ = P_fifo[6][5];	// L7427
  hls::stream< int8_t > &v6756 /* v6756[16] */ = A_fifo[5][7];	// L7428
  hls::stream< int32_t > &v6757 /* v6757[16] */ = P_fifo[6][6];	// L7429
  PE_kernel_ws_gemm_6_5(v6754, v6755, v6756, v6757, v6401, 5, 6);	// L7430
  hls::stream< int8_t > &v6758 /* v6758[16] */ = A_fifo[5][7];	// L7431
  hls::stream< int32_t > &v6759 /* v6759[16] */ = P_fifo[7][5];	// L7432
  hls::stream< int8_t > &v6760 /* v6760[16] */ = A_fifo[5][8];	// L7433
  hls::stream< int32_t > &v6761 /* v6761[16] */ = P_fifo[7][6];	// L7434
  PE_kernel_ws_gemm_7_5(v6758, v6759, v6760, v6761, v6401, 5, 7);	// L7435
  hls::stream< int8_t > &v6762 /* v6762[16] */ = A_fifo[5][8];	// L7436
  hls::stream< int32_t > &v6763 /* v6763[16] */ = P_fifo[8][5];	// L7437
  hls::stream< int8_t > &v6764 /* v6764[16] */ = A_fifo[5][9];	// L7438
  hls::stream< int32_t > &v6765 /* v6765[16] */ = P_fifo[8][6];	// L7439
  PE_kernel_ws_gemm_8_5(v6762, v6763, v6764, v6765, v6401, 5, 8);	// L7440
  hls::stream< int8_t > &v6766 /* v6766[16] */ = A_fifo[5][9];	// L7441
  hls::stream< int32_t > &v6767 /* v6767[16] */ = P_fifo[9][5];	// L7442
  hls::stream< int8_t > &v6768 /* v6768[16] */ = A_fifo[5][10];	// L7443
  hls::stream< int32_t > &v6769 /* v6769[16] */ = P_fifo[9][6];	// L7444
  PE_kernel_ws_gemm_9_5(v6766, v6767, v6768, v6769, v6401, 5, 9);	// L7445
  hls::stream< int8_t > &v6770 /* v6770[16] */ = A_fifo[5][10];	// L7446
  hls::stream< int32_t > &v6771 /* v6771[16] */ = P_fifo[10][5];	// L7447
  hls::stream< int8_t > &v6772 /* v6772[16] */ = A_fifo[5][11];	// L7448
  hls::stream< int32_t > &v6773 /* v6773[16] */ = P_fifo[10][6];	// L7449
  PE_kernel_ws_gemm_10_5(v6770, v6771, v6772, v6773, v6401, 5, 10);	// L7450
  hls::stream< int8_t > &v6774 /* v6774[16] */ = A_fifo[5][11];	// L7451
  hls::stream< int32_t > &v6775 /* v6775[16] */ = P_fifo[11][5];	// L7452
  hls::stream< int8_t > &v6776 /* v6776[16] */ = A_fifo[5][12];	// L7453
  hls::stream< int32_t > &v6777 /* v6777[16] */ = P_fifo[11][6];	// L7454
  PE_kernel_ws_gemm_11_5(v6774, v6775, v6776, v6777, v6401, 5, 11);	// L7455
  hls::stream< int8_t > &v6778 /* v6778[16] */ = A_fifo[5][12];	// L7456
  hls::stream< int32_t > &v6779 /* v6779[16] */ = P_fifo[12][5];	// L7457
  hls::stream< int8_t > &v6780 /* v6780[16] */ = A_fifo[5][13];	// L7458
  hls::stream< int32_t > &v6781 /* v6781[16] */ = P_fifo[12][6];	// L7459
  PE_kernel_ws_gemm_12_5(v6778, v6779, v6780, v6781, v6401, 5, 12);	// L7460
  hls::stream< int8_t > &v6782 /* v6782[16] */ = A_fifo[5][13];	// L7461
  hls::stream< int32_t > &v6783 /* v6783[16] */ = P_fifo[13][5];	// L7462
  hls::stream< int8_t > &v6784 /* v6784[16] */ = A_fifo[5][14];	// L7463
  hls::stream< int32_t > &v6785 /* v6785[16] */ = P_fifo[13][6];	// L7464
  PE_kernel_ws_gemm_13_5(v6782, v6783, v6784, v6785, v6401, 5, 13);	// L7465
  hls::stream< int8_t > &v6786 /* v6786[16] */ = A_fifo[5][14];	// L7466
  hls::stream< int32_t > &v6787 /* v6787[16] */ = P_fifo[14][5];	// L7467
  hls::stream< int8_t > &v6788 /* v6788[16] */ = A_fifo[5][15];	// L7468
  hls::stream< int32_t > &v6789 /* v6789[16] */ = P_fifo[14][6];	// L7469
  PE_kernel_ws_gemm_14_5(v6786, v6787, v6788, v6789, v6401, 5, 14);	// L7470
  hls::stream< int8_t > &v6790 /* v6790[16] */ = A_fifo[5][15];	// L7471
  hls::stream< int32_t > &v6791 /* v6791[16] */ = P_fifo[15][5];	// L7472
  hls::stream< int8_t > &v6792 /* v6792[16] */ = A_fifo[5][16];	// L7473
  hls::stream< int32_t > &v6793 /* v6793[16] */ = P_fifo[15][6];	// L7474
  PE_kernel_ws_gemm_15_5(v6790, v6791, v6792, v6793, v6401, 5, 15);	// L7475
  hls::stream< int8_t > &v6794 /* v6794[16] */ = A_fifo[6][0];	// L7476
  hls::stream< int32_t > &v6795 /* v6795[16] */ = P_fifo[0][6];	// L7477
  hls::stream< int8_t > &v6796 /* v6796[16] */ = A_fifo[6][1];	// L7478
  hls::stream< int32_t > &v6797 /* v6797[16] */ = P_fifo[0][7];	// L7479
  PE_kernel_ws_gemm_0_6(v6794, v6795, v6796, v6797, v6401, 6, 0);	// L7480
  hls::stream< int8_t > &v6798 /* v6798[16] */ = A_fifo[6][1];	// L7481
  hls::stream< int32_t > &v6799 /* v6799[16] */ = P_fifo[1][6];	// L7482
  hls::stream< int8_t > &v6800 /* v6800[16] */ = A_fifo[6][2];	// L7483
  hls::stream< int32_t > &v6801 /* v6801[16] */ = P_fifo[1][7];	// L7484
  PE_kernel_ws_gemm_1_6(v6798, v6799, v6800, v6801, v6401, 6, 1);	// L7485
  hls::stream< int8_t > &v6802 /* v6802[16] */ = A_fifo[6][2];	// L7486
  hls::stream< int32_t > &v6803 /* v6803[16] */ = P_fifo[2][6];	// L7487
  hls::stream< int8_t > &v6804 /* v6804[16] */ = A_fifo[6][3];	// L7488
  hls::stream< int32_t > &v6805 /* v6805[16] */ = P_fifo[2][7];	// L7489
  PE_kernel_ws_gemm_2_6(v6802, v6803, v6804, v6805, v6401, 6, 2);	// L7490
  hls::stream< int8_t > &v6806 /* v6806[16] */ = A_fifo[6][3];	// L7491
  hls::stream< int32_t > &v6807 /* v6807[16] */ = P_fifo[3][6];	// L7492
  hls::stream< int8_t > &v6808 /* v6808[16] */ = A_fifo[6][4];	// L7493
  hls::stream< int32_t > &v6809 /* v6809[16] */ = P_fifo[3][7];	// L7494
  PE_kernel_ws_gemm_3_6(v6806, v6807, v6808, v6809, v6401, 6, 3);	// L7495
  hls::stream< int8_t > &v6810 /* v6810[16] */ = A_fifo[6][4];	// L7496
  hls::stream< int32_t > &v6811 /* v6811[16] */ = P_fifo[4][6];	// L7497
  hls::stream< int8_t > &v6812 /* v6812[16] */ = A_fifo[6][5];	// L7498
  hls::stream< int32_t > &v6813 /* v6813[16] */ = P_fifo[4][7];	// L7499
  PE_kernel_ws_gemm_4_6(v6810, v6811, v6812, v6813, v6401, 6, 4);	// L7500
  hls::stream< int8_t > &v6814 /* v6814[16] */ = A_fifo[6][5];	// L7501
  hls::stream< int32_t > &v6815 /* v6815[16] */ = P_fifo[5][6];	// L7502
  hls::stream< int8_t > &v6816 /* v6816[16] */ = A_fifo[6][6];	// L7503
  hls::stream< int32_t > &v6817 /* v6817[16] */ = P_fifo[5][7];	// L7504
  PE_kernel_ws_gemm_5_6(v6814, v6815, v6816, v6817, v6401, 6, 5);	// L7505
  hls::stream< int8_t > &v6818 /* v6818[16] */ = A_fifo[6][6];	// L7506
  hls::stream< int32_t > &v6819 /* v6819[16] */ = P_fifo[6][6];	// L7507
  hls::stream< int8_t > &v6820 /* v6820[16] */ = A_fifo[6][7];	// L7508
  hls::stream< int32_t > &v6821 /* v6821[16] */ = P_fifo[6][7];	// L7509
  PE_kernel_ws_gemm_6_6(v6818, v6819, v6820, v6821, v6401, 6, 6);	// L7510
  hls::stream< int8_t > &v6822 /* v6822[16] */ = A_fifo[6][7];	// L7511
  hls::stream< int32_t > &v6823 /* v6823[16] */ = P_fifo[7][6];	// L7512
  hls::stream< int8_t > &v6824 /* v6824[16] */ = A_fifo[6][8];	// L7513
  hls::stream< int32_t > &v6825 /* v6825[16] */ = P_fifo[7][7];	// L7514
  PE_kernel_ws_gemm_7_6(v6822, v6823, v6824, v6825, v6401, 6, 7);	// L7515
  hls::stream< int8_t > &v6826 /* v6826[16] */ = A_fifo[6][8];	// L7516
  hls::stream< int32_t > &v6827 /* v6827[16] */ = P_fifo[8][6];	// L7517
  hls::stream< int8_t > &v6828 /* v6828[16] */ = A_fifo[6][9];	// L7518
  hls::stream< int32_t > &v6829 /* v6829[16] */ = P_fifo[8][7];	// L7519
  PE_kernel_ws_gemm_8_6(v6826, v6827, v6828, v6829, v6401, 6, 8);	// L7520
  hls::stream< int8_t > &v6830 /* v6830[16] */ = A_fifo[6][9];	// L7521
  hls::stream< int32_t > &v6831 /* v6831[16] */ = P_fifo[9][6];	// L7522
  hls::stream< int8_t > &v6832 /* v6832[16] */ = A_fifo[6][10];	// L7523
  hls::stream< int32_t > &v6833 /* v6833[16] */ = P_fifo[9][7];	// L7524
  PE_kernel_ws_gemm_9_6(v6830, v6831, v6832, v6833, v6401, 6, 9);	// L7525
  hls::stream< int8_t > &v6834 /* v6834[16] */ = A_fifo[6][10];	// L7526
  hls::stream< int32_t > &v6835 /* v6835[16] */ = P_fifo[10][6];	// L7527
  hls::stream< int8_t > &v6836 /* v6836[16] */ = A_fifo[6][11];	// L7528
  hls::stream< int32_t > &v6837 /* v6837[16] */ = P_fifo[10][7];	// L7529
  PE_kernel_ws_gemm_10_6(v6834, v6835, v6836, v6837, v6401, 6, 10);	// L7530
  hls::stream< int8_t > &v6838 /* v6838[16] */ = A_fifo[6][11];	// L7531
  hls::stream< int32_t > &v6839 /* v6839[16] */ = P_fifo[11][6];	// L7532
  hls::stream< int8_t > &v6840 /* v6840[16] */ = A_fifo[6][12];	// L7533
  hls::stream< int32_t > &v6841 /* v6841[16] */ = P_fifo[11][7];	// L7534
  PE_kernel_ws_gemm_11_6(v6838, v6839, v6840, v6841, v6401, 6, 11);	// L7535
  hls::stream< int8_t > &v6842 /* v6842[16] */ = A_fifo[6][12];	// L7536
  hls::stream< int32_t > &v6843 /* v6843[16] */ = P_fifo[12][6];	// L7537
  hls::stream< int8_t > &v6844 /* v6844[16] */ = A_fifo[6][13];	// L7538
  hls::stream< int32_t > &v6845 /* v6845[16] */ = P_fifo[12][7];	// L7539
  PE_kernel_ws_gemm_12_6(v6842, v6843, v6844, v6845, v6401, 6, 12);	// L7540
  hls::stream< int8_t > &v6846 /* v6846[16] */ = A_fifo[6][13];	// L7541
  hls::stream< int32_t > &v6847 /* v6847[16] */ = P_fifo[13][6];	// L7542
  hls::stream< int8_t > &v6848 /* v6848[16] */ = A_fifo[6][14];	// L7543
  hls::stream< int32_t > &v6849 /* v6849[16] */ = P_fifo[13][7];	// L7544
  PE_kernel_ws_gemm_13_6(v6846, v6847, v6848, v6849, v6401, 6, 13);	// L7545
  hls::stream< int8_t > &v6850 /* v6850[16] */ = A_fifo[6][14];	// L7546
  hls::stream< int32_t > &v6851 /* v6851[16] */ = P_fifo[14][6];	// L7547
  hls::stream< int8_t > &v6852 /* v6852[16] */ = A_fifo[6][15];	// L7548
  hls::stream< int32_t > &v6853 /* v6853[16] */ = P_fifo[14][7];	// L7549
  PE_kernel_ws_gemm_14_6(v6850, v6851, v6852, v6853, v6401, 6, 14);	// L7550
  hls::stream< int8_t > &v6854 /* v6854[16] */ = A_fifo[6][15];	// L7551
  hls::stream< int32_t > &v6855 /* v6855[16] */ = P_fifo[15][6];	// L7552
  hls::stream< int8_t > &v6856 /* v6856[16] */ = A_fifo[6][16];	// L7553
  hls::stream< int32_t > &v6857 /* v6857[16] */ = P_fifo[15][7];	// L7554
  PE_kernel_ws_gemm_15_6(v6854, v6855, v6856, v6857, v6401, 6, 15);	// L7555
  hls::stream< int8_t > &v6858 /* v6858[16] */ = A_fifo[7][0];	// L7556
  hls::stream< int32_t > &v6859 /* v6859[16] */ = P_fifo[0][7];	// L7557
  hls::stream< int8_t > &v6860 /* v6860[16] */ = A_fifo[7][1];	// L7558
  hls::stream< int32_t > &v6861 /* v6861[16] */ = P_fifo[0][8];	// L7559
  PE_kernel_ws_gemm_0_7(v6858, v6859, v6860, v6861, v6401, 7, 0);	// L7560
  hls::stream< int8_t > &v6862 /* v6862[16] */ = A_fifo[7][1];	// L7561
  hls::stream< int32_t > &v6863 /* v6863[16] */ = P_fifo[1][7];	// L7562
  hls::stream< int8_t > &v6864 /* v6864[16] */ = A_fifo[7][2];	// L7563
  hls::stream< int32_t > &v6865 /* v6865[16] */ = P_fifo[1][8];	// L7564
  PE_kernel_ws_gemm_1_7(v6862, v6863, v6864, v6865, v6401, 7, 1);	// L7565
  hls::stream< int8_t > &v6866 /* v6866[16] */ = A_fifo[7][2];	// L7566
  hls::stream< int32_t > &v6867 /* v6867[16] */ = P_fifo[2][7];	// L7567
  hls::stream< int8_t > &v6868 /* v6868[16] */ = A_fifo[7][3];	// L7568
  hls::stream< int32_t > &v6869 /* v6869[16] */ = P_fifo[2][8];	// L7569
  PE_kernel_ws_gemm_2_7(v6866, v6867, v6868, v6869, v6401, 7, 2);	// L7570
  hls::stream< int8_t > &v6870 /* v6870[16] */ = A_fifo[7][3];	// L7571
  hls::stream< int32_t > &v6871 /* v6871[16] */ = P_fifo[3][7];	// L7572
  hls::stream< int8_t > &v6872 /* v6872[16] */ = A_fifo[7][4];	// L7573
  hls::stream< int32_t > &v6873 /* v6873[16] */ = P_fifo[3][8];	// L7574
  PE_kernel_ws_gemm_3_7(v6870, v6871, v6872, v6873, v6401, 7, 3);	// L7575
  hls::stream< int8_t > &v6874 /* v6874[16] */ = A_fifo[7][4];	// L7576
  hls::stream< int32_t > &v6875 /* v6875[16] */ = P_fifo[4][7];	// L7577
  hls::stream< int8_t > &v6876 /* v6876[16] */ = A_fifo[7][5];	// L7578
  hls::stream< int32_t > &v6877 /* v6877[16] */ = P_fifo[4][8];	// L7579
  PE_kernel_ws_gemm_4_7(v6874, v6875, v6876, v6877, v6401, 7, 4);	// L7580
  hls::stream< int8_t > &v6878 /* v6878[16] */ = A_fifo[7][5];	// L7581
  hls::stream< int32_t > &v6879 /* v6879[16] */ = P_fifo[5][7];	// L7582
  hls::stream< int8_t > &v6880 /* v6880[16] */ = A_fifo[7][6];	// L7583
  hls::stream< int32_t > &v6881 /* v6881[16] */ = P_fifo[5][8];	// L7584
  PE_kernel_ws_gemm_5_7(v6878, v6879, v6880, v6881, v6401, 7, 5);	// L7585
  hls::stream< int8_t > &v6882 /* v6882[16] */ = A_fifo[7][6];	// L7586
  hls::stream< int32_t > &v6883 /* v6883[16] */ = P_fifo[6][7];	// L7587
  hls::stream< int8_t > &v6884 /* v6884[16] */ = A_fifo[7][7];	// L7588
  hls::stream< int32_t > &v6885 /* v6885[16] */ = P_fifo[6][8];	// L7589
  PE_kernel_ws_gemm_6_7(v6882, v6883, v6884, v6885, v6401, 7, 6);	// L7590
  hls::stream< int8_t > &v6886 /* v6886[16] */ = A_fifo[7][7];	// L7591
  hls::stream< int32_t > &v6887 /* v6887[16] */ = P_fifo[7][7];	// L7592
  hls::stream< int8_t > &v6888 /* v6888[16] */ = A_fifo[7][8];	// L7593
  hls::stream< int32_t > &v6889 /* v6889[16] */ = P_fifo[7][8];	// L7594
  PE_kernel_ws_gemm_7_7(v6886, v6887, v6888, v6889, v6401, 7, 7);	// L7595
  hls::stream< int8_t > &v6890 /* v6890[16] */ = A_fifo[7][8];	// L7596
  hls::stream< int32_t > &v6891 /* v6891[16] */ = P_fifo[8][7];	// L7597
  hls::stream< int8_t > &v6892 /* v6892[16] */ = A_fifo[7][9];	// L7598
  hls::stream< int32_t > &v6893 /* v6893[16] */ = P_fifo[8][8];	// L7599
  PE_kernel_ws_gemm_8_7(v6890, v6891, v6892, v6893, v6401, 7, 8);	// L7600
  hls::stream< int8_t > &v6894 /* v6894[16] */ = A_fifo[7][9];	// L7601
  hls::stream< int32_t > &v6895 /* v6895[16] */ = P_fifo[9][7];	// L7602
  hls::stream< int8_t > &v6896 /* v6896[16] */ = A_fifo[7][10];	// L7603
  hls::stream< int32_t > &v6897 /* v6897[16] */ = P_fifo[9][8];	// L7604
  PE_kernel_ws_gemm_9_7(v6894, v6895, v6896, v6897, v6401, 7, 9);	// L7605
  hls::stream< int8_t > &v6898 /* v6898[16] */ = A_fifo[7][10];	// L7606
  hls::stream< int32_t > &v6899 /* v6899[16] */ = P_fifo[10][7];	// L7607
  hls::stream< int8_t > &v6900 /* v6900[16] */ = A_fifo[7][11];	// L7608
  hls::stream< int32_t > &v6901 /* v6901[16] */ = P_fifo[10][8];	// L7609
  PE_kernel_ws_gemm_10_7(v6898, v6899, v6900, v6901, v6401, 7, 10);	// L7610
  hls::stream< int8_t > &v6902 /* v6902[16] */ = A_fifo[7][11];	// L7611
  hls::stream< int32_t > &v6903 /* v6903[16] */ = P_fifo[11][7];	// L7612
  hls::stream< int8_t > &v6904 /* v6904[16] */ = A_fifo[7][12];	// L7613
  hls::stream< int32_t > &v6905 /* v6905[16] */ = P_fifo[11][8];	// L7614
  PE_kernel_ws_gemm_11_7(v6902, v6903, v6904, v6905, v6401, 7, 11);	// L7615
  hls::stream< int8_t > &v6906 /* v6906[16] */ = A_fifo[7][12];	// L7616
  hls::stream< int32_t > &v6907 /* v6907[16] */ = P_fifo[12][7];	// L7617
  hls::stream< int8_t > &v6908 /* v6908[16] */ = A_fifo[7][13];	// L7618
  hls::stream< int32_t > &v6909 /* v6909[16] */ = P_fifo[12][8];	// L7619
  PE_kernel_ws_gemm_12_7(v6906, v6907, v6908, v6909, v6401, 7, 12);	// L7620
  hls::stream< int8_t > &v6910 /* v6910[16] */ = A_fifo[7][13];	// L7621
  hls::stream< int32_t > &v6911 /* v6911[16] */ = P_fifo[13][7];	// L7622
  hls::stream< int8_t > &v6912 /* v6912[16] */ = A_fifo[7][14];	// L7623
  hls::stream< int32_t > &v6913 /* v6913[16] */ = P_fifo[13][8];	// L7624
  PE_kernel_ws_gemm_13_7(v6910, v6911, v6912, v6913, v6401, 7, 13);	// L7625
  hls::stream< int8_t > &v6914 /* v6914[16] */ = A_fifo[7][14];	// L7626
  hls::stream< int32_t > &v6915 /* v6915[16] */ = P_fifo[14][7];	// L7627
  hls::stream< int8_t > &v6916 /* v6916[16] */ = A_fifo[7][15];	// L7628
  hls::stream< int32_t > &v6917 /* v6917[16] */ = P_fifo[14][8];	// L7629
  PE_kernel_ws_gemm_14_7(v6914, v6915, v6916, v6917, v6401, 7, 14);	// L7630
  hls::stream< int8_t > &v6918 /* v6918[16] */ = A_fifo[7][15];	// L7631
  hls::stream< int32_t > &v6919 /* v6919[16] */ = P_fifo[15][7];	// L7632
  hls::stream< int8_t > &v6920 /* v6920[16] */ = A_fifo[7][16];	// L7633
  hls::stream< int32_t > &v6921 /* v6921[16] */ = P_fifo[15][8];	// L7634
  PE_kernel_ws_gemm_15_7(v6918, v6919, v6920, v6921, v6401, 7, 15);	// L7635
  hls::stream< int8_t > &v6922 /* v6922[16] */ = A_fifo[8][0];	// L7636
  hls::stream< int32_t > &v6923 /* v6923[16] */ = P_fifo[0][8];	// L7637
  hls::stream< int8_t > &v6924 /* v6924[16] */ = A_fifo[8][1];	// L7638
  hls::stream< int32_t > &v6925 /* v6925[16] */ = P_fifo[0][9];	// L7639
  PE_kernel_ws_gemm_0_8(v6922, v6923, v6924, v6925, v6401, 8, 0);	// L7640
  hls::stream< int8_t > &v6926 /* v6926[16] */ = A_fifo[8][1];	// L7641
  hls::stream< int32_t > &v6927 /* v6927[16] */ = P_fifo[1][8];	// L7642
  hls::stream< int8_t > &v6928 /* v6928[16] */ = A_fifo[8][2];	// L7643
  hls::stream< int32_t > &v6929 /* v6929[16] */ = P_fifo[1][9];	// L7644
  PE_kernel_ws_gemm_1_8(v6926, v6927, v6928, v6929, v6401, 8, 1);	// L7645
  hls::stream< int8_t > &v6930 /* v6930[16] */ = A_fifo[8][2];	// L7646
  hls::stream< int32_t > &v6931 /* v6931[16] */ = P_fifo[2][8];	// L7647
  hls::stream< int8_t > &v6932 /* v6932[16] */ = A_fifo[8][3];	// L7648
  hls::stream< int32_t > &v6933 /* v6933[16] */ = P_fifo[2][9];	// L7649
  PE_kernel_ws_gemm_2_8(v6930, v6931, v6932, v6933, v6401, 8, 2);	// L7650
  hls::stream< int8_t > &v6934 /* v6934[16] */ = A_fifo[8][3];	// L7651
  hls::stream< int32_t > &v6935 /* v6935[16] */ = P_fifo[3][8];	// L7652
  hls::stream< int8_t > &v6936 /* v6936[16] */ = A_fifo[8][4];	// L7653
  hls::stream< int32_t > &v6937 /* v6937[16] */ = P_fifo[3][9];	// L7654
  PE_kernel_ws_gemm_3_8(v6934, v6935, v6936, v6937, v6401, 8, 3);	// L7655
  hls::stream< int8_t > &v6938 /* v6938[16] */ = A_fifo[8][4];	// L7656
  hls::stream< int32_t > &v6939 /* v6939[16] */ = P_fifo[4][8];	// L7657
  hls::stream< int8_t > &v6940 /* v6940[16] */ = A_fifo[8][5];	// L7658
  hls::stream< int32_t > &v6941 /* v6941[16] */ = P_fifo[4][9];	// L7659
  PE_kernel_ws_gemm_4_8(v6938, v6939, v6940, v6941, v6401, 8, 4);	// L7660
  hls::stream< int8_t > &v6942 /* v6942[16] */ = A_fifo[8][5];	// L7661
  hls::stream< int32_t > &v6943 /* v6943[16] */ = P_fifo[5][8];	// L7662
  hls::stream< int8_t > &v6944 /* v6944[16] */ = A_fifo[8][6];	// L7663
  hls::stream< int32_t > &v6945 /* v6945[16] */ = P_fifo[5][9];	// L7664
  PE_kernel_ws_gemm_5_8(v6942, v6943, v6944, v6945, v6401, 8, 5);	// L7665
  hls::stream< int8_t > &v6946 /* v6946[16] */ = A_fifo[8][6];	// L7666
  hls::stream< int32_t > &v6947 /* v6947[16] */ = P_fifo[6][8];	// L7667
  hls::stream< int8_t > &v6948 /* v6948[16] */ = A_fifo[8][7];	// L7668
  hls::stream< int32_t > &v6949 /* v6949[16] */ = P_fifo[6][9];	// L7669
  PE_kernel_ws_gemm_6_8(v6946, v6947, v6948, v6949, v6401, 8, 6);	// L7670
  hls::stream< int8_t > &v6950 /* v6950[16] */ = A_fifo[8][7];	// L7671
  hls::stream< int32_t > &v6951 /* v6951[16] */ = P_fifo[7][8];	// L7672
  hls::stream< int8_t > &v6952 /* v6952[16] */ = A_fifo[8][8];	// L7673
  hls::stream< int32_t > &v6953 /* v6953[16] */ = P_fifo[7][9];	// L7674
  PE_kernel_ws_gemm_7_8(v6950, v6951, v6952, v6953, v6401, 8, 7);	// L7675
  hls::stream< int8_t > &v6954 /* v6954[16] */ = A_fifo[8][8];	// L7676
  hls::stream< int32_t > &v6955 /* v6955[16] */ = P_fifo[8][8];	// L7677
  hls::stream< int8_t > &v6956 /* v6956[16] */ = A_fifo[8][9];	// L7678
  hls::stream< int32_t > &v6957 /* v6957[16] */ = P_fifo[8][9];	// L7679
  PE_kernel_ws_gemm_8_8(v6954, v6955, v6956, v6957, v6401, 8, 8);	// L7680
  hls::stream< int8_t > &v6958 /* v6958[16] */ = A_fifo[8][9];	// L7681
  hls::stream< int32_t > &v6959 /* v6959[16] */ = P_fifo[9][8];	// L7682
  hls::stream< int8_t > &v6960 /* v6960[16] */ = A_fifo[8][10];	// L7683
  hls::stream< int32_t > &v6961 /* v6961[16] */ = P_fifo[9][9];	// L7684
  PE_kernel_ws_gemm_9_8(v6958, v6959, v6960, v6961, v6401, 8, 9);	// L7685
  hls::stream< int8_t > &v6962 /* v6962[16] */ = A_fifo[8][10];	// L7686
  hls::stream< int32_t > &v6963 /* v6963[16] */ = P_fifo[10][8];	// L7687
  hls::stream< int8_t > &v6964 /* v6964[16] */ = A_fifo[8][11];	// L7688
  hls::stream< int32_t > &v6965 /* v6965[16] */ = P_fifo[10][9];	// L7689
  PE_kernel_ws_gemm_10_8(v6962, v6963, v6964, v6965, v6401, 8, 10);	// L7690
  hls::stream< int8_t > &v6966 /* v6966[16] */ = A_fifo[8][11];	// L7691
  hls::stream< int32_t > &v6967 /* v6967[16] */ = P_fifo[11][8];	// L7692
  hls::stream< int8_t > &v6968 /* v6968[16] */ = A_fifo[8][12];	// L7693
  hls::stream< int32_t > &v6969 /* v6969[16] */ = P_fifo[11][9];	// L7694
  PE_kernel_ws_gemm_11_8(v6966, v6967, v6968, v6969, v6401, 8, 11);	// L7695
  hls::stream< int8_t > &v6970 /* v6970[16] */ = A_fifo[8][12];	// L7696
  hls::stream< int32_t > &v6971 /* v6971[16] */ = P_fifo[12][8];	// L7697
  hls::stream< int8_t > &v6972 /* v6972[16] */ = A_fifo[8][13];	// L7698
  hls::stream< int32_t > &v6973 /* v6973[16] */ = P_fifo[12][9];	// L7699
  PE_kernel_ws_gemm_12_8(v6970, v6971, v6972, v6973, v6401, 8, 12);	// L7700
  hls::stream< int8_t > &v6974 /* v6974[16] */ = A_fifo[8][13];	// L7701
  hls::stream< int32_t > &v6975 /* v6975[16] */ = P_fifo[13][8];	// L7702
  hls::stream< int8_t > &v6976 /* v6976[16] */ = A_fifo[8][14];	// L7703
  hls::stream< int32_t > &v6977 /* v6977[16] */ = P_fifo[13][9];	// L7704
  PE_kernel_ws_gemm_13_8(v6974, v6975, v6976, v6977, v6401, 8, 13);	// L7705
  hls::stream< int8_t > &v6978 /* v6978[16] */ = A_fifo[8][14];	// L7706
  hls::stream< int32_t > &v6979 /* v6979[16] */ = P_fifo[14][8];	// L7707
  hls::stream< int8_t > &v6980 /* v6980[16] */ = A_fifo[8][15];	// L7708
  hls::stream< int32_t > &v6981 /* v6981[16] */ = P_fifo[14][9];	// L7709
  PE_kernel_ws_gemm_14_8(v6978, v6979, v6980, v6981, v6401, 8, 14);	// L7710
  hls::stream< int8_t > &v6982 /* v6982[16] */ = A_fifo[8][15];	// L7711
  hls::stream< int32_t > &v6983 /* v6983[16] */ = P_fifo[15][8];	// L7712
  hls::stream< int8_t > &v6984 /* v6984[16] */ = A_fifo[8][16];	// L7713
  hls::stream< int32_t > &v6985 /* v6985[16] */ = P_fifo[15][9];	// L7714
  PE_kernel_ws_gemm_15_8(v6982, v6983, v6984, v6985, v6401, 8, 15);	// L7715
  hls::stream< int8_t > &v6986 /* v6986[16] */ = A_fifo[9][0];	// L7716
  hls::stream< int32_t > &v6987 /* v6987[16] */ = P_fifo[0][9];	// L7717
  hls::stream< int8_t > &v6988 /* v6988[16] */ = A_fifo[9][1];	// L7718
  hls::stream< int32_t > &v6989 /* v6989[16] */ = P_fifo[0][10];	// L7719
  PE_kernel_ws_gemm_0_9(v6986, v6987, v6988, v6989, v6401, 9, 0);	// L7720
  hls::stream< int8_t > &v6990 /* v6990[16] */ = A_fifo[9][1];	// L7721
  hls::stream< int32_t > &v6991 /* v6991[16] */ = P_fifo[1][9];	// L7722
  hls::stream< int8_t > &v6992 /* v6992[16] */ = A_fifo[9][2];	// L7723
  hls::stream< int32_t > &v6993 /* v6993[16] */ = P_fifo[1][10];	// L7724
  PE_kernel_ws_gemm_1_9(v6990, v6991, v6992, v6993, v6401, 9, 1);	// L7725
  hls::stream< int8_t > &v6994 /* v6994[16] */ = A_fifo[9][2];	// L7726
  hls::stream< int32_t > &v6995 /* v6995[16] */ = P_fifo[2][9];	// L7727
  hls::stream< int8_t > &v6996 /* v6996[16] */ = A_fifo[9][3];	// L7728
  hls::stream< int32_t > &v6997 /* v6997[16] */ = P_fifo[2][10];	// L7729
  PE_kernel_ws_gemm_2_9(v6994, v6995, v6996, v6997, v6401, 9, 2);	// L7730
  hls::stream< int8_t > &v6998 /* v6998[16] */ = A_fifo[9][3];	// L7731
  hls::stream< int32_t > &v6999 /* v6999[16] */ = P_fifo[3][9];	// L7732
  hls::stream< int8_t > &v7000 /* v7000[16] */ = A_fifo[9][4];	// L7733
  hls::stream< int32_t > &v7001 /* v7001[16] */ = P_fifo[3][10];	// L7734
  PE_kernel_ws_gemm_3_9(v6998, v6999, v7000, v7001, v6401, 9, 3);	// L7735
  hls::stream< int8_t > &v7002 /* v7002[16] */ = A_fifo[9][4];	// L7736
  hls::stream< int32_t > &v7003 /* v7003[16] */ = P_fifo[4][9];	// L7737
  hls::stream< int8_t > &v7004 /* v7004[16] */ = A_fifo[9][5];	// L7738
  hls::stream< int32_t > &v7005 /* v7005[16] */ = P_fifo[4][10];	// L7739
  PE_kernel_ws_gemm_4_9(v7002, v7003, v7004, v7005, v6401, 9, 4);	// L7740
  hls::stream< int8_t > &v7006 /* v7006[16] */ = A_fifo[9][5];	// L7741
  hls::stream< int32_t > &v7007 /* v7007[16] */ = P_fifo[5][9];	// L7742
  hls::stream< int8_t > &v7008 /* v7008[16] */ = A_fifo[9][6];	// L7743
  hls::stream< int32_t > &v7009 /* v7009[16] */ = P_fifo[5][10];	// L7744
  PE_kernel_ws_gemm_5_9(v7006, v7007, v7008, v7009, v6401, 9, 5);	// L7745
  hls::stream< int8_t > &v7010 /* v7010[16] */ = A_fifo[9][6];	// L7746
  hls::stream< int32_t > &v7011 /* v7011[16] */ = P_fifo[6][9];	// L7747
  hls::stream< int8_t > &v7012 /* v7012[16] */ = A_fifo[9][7];	// L7748
  hls::stream< int32_t > &v7013 /* v7013[16] */ = P_fifo[6][10];	// L7749
  PE_kernel_ws_gemm_6_9(v7010, v7011, v7012, v7013, v6401, 9, 6);	// L7750
  hls::stream< int8_t > &v7014 /* v7014[16] */ = A_fifo[9][7];	// L7751
  hls::stream< int32_t > &v7015 /* v7015[16] */ = P_fifo[7][9];	// L7752
  hls::stream< int8_t > &v7016 /* v7016[16] */ = A_fifo[9][8];	// L7753
  hls::stream< int32_t > &v7017 /* v7017[16] */ = P_fifo[7][10];	// L7754
  PE_kernel_ws_gemm_7_9(v7014, v7015, v7016, v7017, v6401, 9, 7);	// L7755
  hls::stream< int8_t > &v7018 /* v7018[16] */ = A_fifo[9][8];	// L7756
  hls::stream< int32_t > &v7019 /* v7019[16] */ = P_fifo[8][9];	// L7757
  hls::stream< int8_t > &v7020 /* v7020[16] */ = A_fifo[9][9];	// L7758
  hls::stream< int32_t > &v7021 /* v7021[16] */ = P_fifo[8][10];	// L7759
  PE_kernel_ws_gemm_8_9(v7018, v7019, v7020, v7021, v6401, 9, 8);	// L7760
  hls::stream< int8_t > &v7022 /* v7022[16] */ = A_fifo[9][9];	// L7761
  hls::stream< int32_t > &v7023 /* v7023[16] */ = P_fifo[9][9];	// L7762
  hls::stream< int8_t > &v7024 /* v7024[16] */ = A_fifo[9][10];	// L7763
  hls::stream< int32_t > &v7025 /* v7025[16] */ = P_fifo[9][10];	// L7764
  PE_kernel_ws_gemm_9_9(v7022, v7023, v7024, v7025, v6401, 9, 9);	// L7765
  hls::stream< int8_t > &v7026 /* v7026[16] */ = A_fifo[9][10];	// L7766
  hls::stream< int32_t > &v7027 /* v7027[16] */ = P_fifo[10][9];	// L7767
  hls::stream< int8_t > &v7028 /* v7028[16] */ = A_fifo[9][11];	// L7768
  hls::stream< int32_t > &v7029 /* v7029[16] */ = P_fifo[10][10];	// L7769
  PE_kernel_ws_gemm_10_9(v7026, v7027, v7028, v7029, v6401, 9, 10);	// L7770
  hls::stream< int8_t > &v7030 /* v7030[16] */ = A_fifo[9][11];	// L7771
  hls::stream< int32_t > &v7031 /* v7031[16] */ = P_fifo[11][9];	// L7772
  hls::stream< int8_t > &v7032 /* v7032[16] */ = A_fifo[9][12];	// L7773
  hls::stream< int32_t > &v7033 /* v7033[16] */ = P_fifo[11][10];	// L7774
  PE_kernel_ws_gemm_11_9(v7030, v7031, v7032, v7033, v6401, 9, 11);	// L7775
  hls::stream< int8_t > &v7034 /* v7034[16] */ = A_fifo[9][12];	// L7776
  hls::stream< int32_t > &v7035 /* v7035[16] */ = P_fifo[12][9];	// L7777
  hls::stream< int8_t > &v7036 /* v7036[16] */ = A_fifo[9][13];	// L7778
  hls::stream< int32_t > &v7037 /* v7037[16] */ = P_fifo[12][10];	// L7779
  PE_kernel_ws_gemm_12_9(v7034, v7035, v7036, v7037, v6401, 9, 12);	// L7780
  hls::stream< int8_t > &v7038 /* v7038[16] */ = A_fifo[9][13];	// L7781
  hls::stream< int32_t > &v7039 /* v7039[16] */ = P_fifo[13][9];	// L7782
  hls::stream< int8_t > &v7040 /* v7040[16] */ = A_fifo[9][14];	// L7783
  hls::stream< int32_t > &v7041 /* v7041[16] */ = P_fifo[13][10];	// L7784
  PE_kernel_ws_gemm_13_9(v7038, v7039, v7040, v7041, v6401, 9, 13);	// L7785
  hls::stream< int8_t > &v7042 /* v7042[16] */ = A_fifo[9][14];	// L7786
  hls::stream< int32_t > &v7043 /* v7043[16] */ = P_fifo[14][9];	// L7787
  hls::stream< int8_t > &v7044 /* v7044[16] */ = A_fifo[9][15];	// L7788
  hls::stream< int32_t > &v7045 /* v7045[16] */ = P_fifo[14][10];	// L7789
  PE_kernel_ws_gemm_14_9(v7042, v7043, v7044, v7045, v6401, 9, 14);	// L7790
  hls::stream< int8_t > &v7046 /* v7046[16] */ = A_fifo[9][15];	// L7791
  hls::stream< int32_t > &v7047 /* v7047[16] */ = P_fifo[15][9];	// L7792
  hls::stream< int8_t > &v7048 /* v7048[16] */ = A_fifo[9][16];	// L7793
  hls::stream< int32_t > &v7049 /* v7049[16] */ = P_fifo[15][10];	// L7794
  PE_kernel_ws_gemm_15_9(v7046, v7047, v7048, v7049, v6401, 9, 15);	// L7795
  hls::stream< int8_t > &v7050 /* v7050[16] */ = A_fifo[10][0];	// L7796
  hls::stream< int32_t > &v7051 /* v7051[16] */ = P_fifo[0][10];	// L7797
  hls::stream< int8_t > &v7052 /* v7052[16] */ = A_fifo[10][1];	// L7798
  hls::stream< int32_t > &v7053 /* v7053[16] */ = P_fifo[0][11];	// L7799
  PE_kernel_ws_gemm_0_10(v7050, v7051, v7052, v7053, v6401, 10, 0);	// L7800
  hls::stream< int8_t > &v7054 /* v7054[16] */ = A_fifo[10][1];	// L7801
  hls::stream< int32_t > &v7055 /* v7055[16] */ = P_fifo[1][10];	// L7802
  hls::stream< int8_t > &v7056 /* v7056[16] */ = A_fifo[10][2];	// L7803
  hls::stream< int32_t > &v7057 /* v7057[16] */ = P_fifo[1][11];	// L7804
  PE_kernel_ws_gemm_1_10(v7054, v7055, v7056, v7057, v6401, 10, 1);	// L7805
  hls::stream< int8_t > &v7058 /* v7058[16] */ = A_fifo[10][2];	// L7806
  hls::stream< int32_t > &v7059 /* v7059[16] */ = P_fifo[2][10];	// L7807
  hls::stream< int8_t > &v7060 /* v7060[16] */ = A_fifo[10][3];	// L7808
  hls::stream< int32_t > &v7061 /* v7061[16] */ = P_fifo[2][11];	// L7809
  PE_kernel_ws_gemm_2_10(v7058, v7059, v7060, v7061, v6401, 10, 2);	// L7810
  hls::stream< int8_t > &v7062 /* v7062[16] */ = A_fifo[10][3];	// L7811
  hls::stream< int32_t > &v7063 /* v7063[16] */ = P_fifo[3][10];	// L7812
  hls::stream< int8_t > &v7064 /* v7064[16] */ = A_fifo[10][4];	// L7813
  hls::stream< int32_t > &v7065 /* v7065[16] */ = P_fifo[3][11];	// L7814
  PE_kernel_ws_gemm_3_10(v7062, v7063, v7064, v7065, v6401, 10, 3);	// L7815
  hls::stream< int8_t > &v7066 /* v7066[16] */ = A_fifo[10][4];	// L7816
  hls::stream< int32_t > &v7067 /* v7067[16] */ = P_fifo[4][10];	// L7817
  hls::stream< int8_t > &v7068 /* v7068[16] */ = A_fifo[10][5];	// L7818
  hls::stream< int32_t > &v7069 /* v7069[16] */ = P_fifo[4][11];	// L7819
  PE_kernel_ws_gemm_4_10(v7066, v7067, v7068, v7069, v6401, 10, 4);	// L7820
  hls::stream< int8_t > &v7070 /* v7070[16] */ = A_fifo[10][5];	// L7821
  hls::stream< int32_t > &v7071 /* v7071[16] */ = P_fifo[5][10];	// L7822
  hls::stream< int8_t > &v7072 /* v7072[16] */ = A_fifo[10][6];	// L7823
  hls::stream< int32_t > &v7073 /* v7073[16] */ = P_fifo[5][11];	// L7824
  PE_kernel_ws_gemm_5_10(v7070, v7071, v7072, v7073, v6401, 10, 5);	// L7825
  hls::stream< int8_t > &v7074 /* v7074[16] */ = A_fifo[10][6];	// L7826
  hls::stream< int32_t > &v7075 /* v7075[16] */ = P_fifo[6][10];	// L7827
  hls::stream< int8_t > &v7076 /* v7076[16] */ = A_fifo[10][7];	// L7828
  hls::stream< int32_t > &v7077 /* v7077[16] */ = P_fifo[6][11];	// L7829
  PE_kernel_ws_gemm_6_10(v7074, v7075, v7076, v7077, v6401, 10, 6);	// L7830
  hls::stream< int8_t > &v7078 /* v7078[16] */ = A_fifo[10][7];	// L7831
  hls::stream< int32_t > &v7079 /* v7079[16] */ = P_fifo[7][10];	// L7832
  hls::stream< int8_t > &v7080 /* v7080[16] */ = A_fifo[10][8];	// L7833
  hls::stream< int32_t > &v7081 /* v7081[16] */ = P_fifo[7][11];	// L7834
  PE_kernel_ws_gemm_7_10(v7078, v7079, v7080, v7081, v6401, 10, 7);	// L7835
  hls::stream< int8_t > &v7082 /* v7082[16] */ = A_fifo[10][8];	// L7836
  hls::stream< int32_t > &v7083 /* v7083[16] */ = P_fifo[8][10];	// L7837
  hls::stream< int8_t > &v7084 /* v7084[16] */ = A_fifo[10][9];	// L7838
  hls::stream< int32_t > &v7085 /* v7085[16] */ = P_fifo[8][11];	// L7839
  PE_kernel_ws_gemm_8_10(v7082, v7083, v7084, v7085, v6401, 10, 8);	// L7840
  hls::stream< int8_t > &v7086 /* v7086[16] */ = A_fifo[10][9];	// L7841
  hls::stream< int32_t > &v7087 /* v7087[16] */ = P_fifo[9][10];	// L7842
  hls::stream< int8_t > &v7088 /* v7088[16] */ = A_fifo[10][10];	// L7843
  hls::stream< int32_t > &v7089 /* v7089[16] */ = P_fifo[9][11];	// L7844
  PE_kernel_ws_gemm_9_10(v7086, v7087, v7088, v7089, v6401, 10, 9);	// L7845
  hls::stream< int8_t > &v7090 /* v7090[16] */ = A_fifo[10][10];	// L7846
  hls::stream< int32_t > &v7091 /* v7091[16] */ = P_fifo[10][10];	// L7847
  hls::stream< int8_t > &v7092 /* v7092[16] */ = A_fifo[10][11];	// L7848
  hls::stream< int32_t > &v7093 /* v7093[16] */ = P_fifo[10][11];	// L7849
  PE_kernel_ws_gemm_10_10(v7090, v7091, v7092, v7093, v6401, 10, 10);	// L7850
  hls::stream< int8_t > &v7094 /* v7094[16] */ = A_fifo[10][11];	// L7851
  hls::stream< int32_t > &v7095 /* v7095[16] */ = P_fifo[11][10];	// L7852
  hls::stream< int8_t > &v7096 /* v7096[16] */ = A_fifo[10][12];	// L7853
  hls::stream< int32_t > &v7097 /* v7097[16] */ = P_fifo[11][11];	// L7854
  PE_kernel_ws_gemm_11_10(v7094, v7095, v7096, v7097, v6401, 10, 11);	// L7855
  hls::stream< int8_t > &v7098 /* v7098[16] */ = A_fifo[10][12];	// L7856
  hls::stream< int32_t > &v7099 /* v7099[16] */ = P_fifo[12][10];	// L7857
  hls::stream< int8_t > &v7100 /* v7100[16] */ = A_fifo[10][13];	// L7858
  hls::stream< int32_t > &v7101 /* v7101[16] */ = P_fifo[12][11];	// L7859
  PE_kernel_ws_gemm_12_10(v7098, v7099, v7100, v7101, v6401, 10, 12);	// L7860
  hls::stream< int8_t > &v7102 /* v7102[16] */ = A_fifo[10][13];	// L7861
  hls::stream< int32_t > &v7103 /* v7103[16] */ = P_fifo[13][10];	// L7862
  hls::stream< int8_t > &v7104 /* v7104[16] */ = A_fifo[10][14];	// L7863
  hls::stream< int32_t > &v7105 /* v7105[16] */ = P_fifo[13][11];	// L7864
  PE_kernel_ws_gemm_13_10(v7102, v7103, v7104, v7105, v6401, 10, 13);	// L7865
  hls::stream< int8_t > &v7106 /* v7106[16] */ = A_fifo[10][14];	// L7866
  hls::stream< int32_t > &v7107 /* v7107[16] */ = P_fifo[14][10];	// L7867
  hls::stream< int8_t > &v7108 /* v7108[16] */ = A_fifo[10][15];	// L7868
  hls::stream< int32_t > &v7109 /* v7109[16] */ = P_fifo[14][11];	// L7869
  PE_kernel_ws_gemm_14_10(v7106, v7107, v7108, v7109, v6401, 10, 14);	// L7870
  hls::stream< int8_t > &v7110 /* v7110[16] */ = A_fifo[10][15];	// L7871
  hls::stream< int32_t > &v7111 /* v7111[16] */ = P_fifo[15][10];	// L7872
  hls::stream< int8_t > &v7112 /* v7112[16] */ = A_fifo[10][16];	// L7873
  hls::stream< int32_t > &v7113 /* v7113[16] */ = P_fifo[15][11];	// L7874
  PE_kernel_ws_gemm_15_10(v7110, v7111, v7112, v7113, v6401, 10, 15);	// L7875
  hls::stream< int8_t > &v7114 /* v7114[16] */ = A_fifo[11][0];	// L7876
  hls::stream< int32_t > &v7115 /* v7115[16] */ = P_fifo[0][11];	// L7877
  hls::stream< int8_t > &v7116 /* v7116[16] */ = A_fifo[11][1];	// L7878
  hls::stream< int32_t > &v7117 /* v7117[16] */ = P_fifo[0][12];	// L7879
  PE_kernel_ws_gemm_0_11(v7114, v7115, v7116, v7117, v6401, 11, 0);	// L7880
  hls::stream< int8_t > &v7118 /* v7118[16] */ = A_fifo[11][1];	// L7881
  hls::stream< int32_t > &v7119 /* v7119[16] */ = P_fifo[1][11];	// L7882
  hls::stream< int8_t > &v7120 /* v7120[16] */ = A_fifo[11][2];	// L7883
  hls::stream< int32_t > &v7121 /* v7121[16] */ = P_fifo[1][12];	// L7884
  PE_kernel_ws_gemm_1_11(v7118, v7119, v7120, v7121, v6401, 11, 1);	// L7885
  hls::stream< int8_t > &v7122 /* v7122[16] */ = A_fifo[11][2];	// L7886
  hls::stream< int32_t > &v7123 /* v7123[16] */ = P_fifo[2][11];	// L7887
  hls::stream< int8_t > &v7124 /* v7124[16] */ = A_fifo[11][3];	// L7888
  hls::stream< int32_t > &v7125 /* v7125[16] */ = P_fifo[2][12];	// L7889
  PE_kernel_ws_gemm_2_11(v7122, v7123, v7124, v7125, v6401, 11, 2);	// L7890
  hls::stream< int8_t > &v7126 /* v7126[16] */ = A_fifo[11][3];	// L7891
  hls::stream< int32_t > &v7127 /* v7127[16] */ = P_fifo[3][11];	// L7892
  hls::stream< int8_t > &v7128 /* v7128[16] */ = A_fifo[11][4];	// L7893
  hls::stream< int32_t > &v7129 /* v7129[16] */ = P_fifo[3][12];	// L7894
  PE_kernel_ws_gemm_3_11(v7126, v7127, v7128, v7129, v6401, 11, 3);	// L7895
  hls::stream< int8_t > &v7130 /* v7130[16] */ = A_fifo[11][4];	// L7896
  hls::stream< int32_t > &v7131 /* v7131[16] */ = P_fifo[4][11];	// L7897
  hls::stream< int8_t > &v7132 /* v7132[16] */ = A_fifo[11][5];	// L7898
  hls::stream< int32_t > &v7133 /* v7133[16] */ = P_fifo[4][12];	// L7899
  PE_kernel_ws_gemm_4_11(v7130, v7131, v7132, v7133, v6401, 11, 4);	// L7900
  hls::stream< int8_t > &v7134 /* v7134[16] */ = A_fifo[11][5];	// L7901
  hls::stream< int32_t > &v7135 /* v7135[16] */ = P_fifo[5][11];	// L7902
  hls::stream< int8_t > &v7136 /* v7136[16] */ = A_fifo[11][6];	// L7903
  hls::stream< int32_t > &v7137 /* v7137[16] */ = P_fifo[5][12];	// L7904
  PE_kernel_ws_gemm_5_11(v7134, v7135, v7136, v7137, v6401, 11, 5);	// L7905
  hls::stream< int8_t > &v7138 /* v7138[16] */ = A_fifo[11][6];	// L7906
  hls::stream< int32_t > &v7139 /* v7139[16] */ = P_fifo[6][11];	// L7907
  hls::stream< int8_t > &v7140 /* v7140[16] */ = A_fifo[11][7];	// L7908
  hls::stream< int32_t > &v7141 /* v7141[16] */ = P_fifo[6][12];	// L7909
  PE_kernel_ws_gemm_6_11(v7138, v7139, v7140, v7141, v6401, 11, 6);	// L7910
  hls::stream< int8_t > &v7142 /* v7142[16] */ = A_fifo[11][7];	// L7911
  hls::stream< int32_t > &v7143 /* v7143[16] */ = P_fifo[7][11];	// L7912
  hls::stream< int8_t > &v7144 /* v7144[16] */ = A_fifo[11][8];	// L7913
  hls::stream< int32_t > &v7145 /* v7145[16] */ = P_fifo[7][12];	// L7914
  PE_kernel_ws_gemm_7_11(v7142, v7143, v7144, v7145, v6401, 11, 7);	// L7915
  hls::stream< int8_t > &v7146 /* v7146[16] */ = A_fifo[11][8];	// L7916
  hls::stream< int32_t > &v7147 /* v7147[16] */ = P_fifo[8][11];	// L7917
  hls::stream< int8_t > &v7148 /* v7148[16] */ = A_fifo[11][9];	// L7918
  hls::stream< int32_t > &v7149 /* v7149[16] */ = P_fifo[8][12];	// L7919
  PE_kernel_ws_gemm_8_11(v7146, v7147, v7148, v7149, v6401, 11, 8);	// L7920
  hls::stream< int8_t > &v7150 /* v7150[16] */ = A_fifo[11][9];	// L7921
  hls::stream< int32_t > &v7151 /* v7151[16] */ = P_fifo[9][11];	// L7922
  hls::stream< int8_t > &v7152 /* v7152[16] */ = A_fifo[11][10];	// L7923
  hls::stream< int32_t > &v7153 /* v7153[16] */ = P_fifo[9][12];	// L7924
  PE_kernel_ws_gemm_9_11(v7150, v7151, v7152, v7153, v6401, 11, 9);	// L7925
  hls::stream< int8_t > &v7154 /* v7154[16] */ = A_fifo[11][10];	// L7926
  hls::stream< int32_t > &v7155 /* v7155[16] */ = P_fifo[10][11];	// L7927
  hls::stream< int8_t > &v7156 /* v7156[16] */ = A_fifo[11][11];	// L7928
  hls::stream< int32_t > &v7157 /* v7157[16] */ = P_fifo[10][12];	// L7929
  PE_kernel_ws_gemm_10_11(v7154, v7155, v7156, v7157, v6401, 11, 10);	// L7930
  hls::stream< int8_t > &v7158 /* v7158[16] */ = A_fifo[11][11];	// L7931
  hls::stream< int32_t > &v7159 /* v7159[16] */ = P_fifo[11][11];	// L7932
  hls::stream< int8_t > &v7160 /* v7160[16] */ = A_fifo[11][12];	// L7933
  hls::stream< int32_t > &v7161 /* v7161[16] */ = P_fifo[11][12];	// L7934
  PE_kernel_ws_gemm_11_11(v7158, v7159, v7160, v7161, v6401, 11, 11);	// L7935
  hls::stream< int8_t > &v7162 /* v7162[16] */ = A_fifo[11][12];	// L7936
  hls::stream< int32_t > &v7163 /* v7163[16] */ = P_fifo[12][11];	// L7937
  hls::stream< int8_t > &v7164 /* v7164[16] */ = A_fifo[11][13];	// L7938
  hls::stream< int32_t > &v7165 /* v7165[16] */ = P_fifo[12][12];	// L7939
  PE_kernel_ws_gemm_12_11(v7162, v7163, v7164, v7165, v6401, 11, 12);	// L7940
  hls::stream< int8_t > &v7166 /* v7166[16] */ = A_fifo[11][13];	// L7941
  hls::stream< int32_t > &v7167 /* v7167[16] */ = P_fifo[13][11];	// L7942
  hls::stream< int8_t > &v7168 /* v7168[16] */ = A_fifo[11][14];	// L7943
  hls::stream< int32_t > &v7169 /* v7169[16] */ = P_fifo[13][12];	// L7944
  PE_kernel_ws_gemm_13_11(v7166, v7167, v7168, v7169, v6401, 11, 13);	// L7945
  hls::stream< int8_t > &v7170 /* v7170[16] */ = A_fifo[11][14];	// L7946
  hls::stream< int32_t > &v7171 /* v7171[16] */ = P_fifo[14][11];	// L7947
  hls::stream< int8_t > &v7172 /* v7172[16] */ = A_fifo[11][15];	// L7948
  hls::stream< int32_t > &v7173 /* v7173[16] */ = P_fifo[14][12];	// L7949
  PE_kernel_ws_gemm_14_11(v7170, v7171, v7172, v7173, v6401, 11, 14);	// L7950
  hls::stream< int8_t > &v7174 /* v7174[16] */ = A_fifo[11][15];	// L7951
  hls::stream< int32_t > &v7175 /* v7175[16] */ = P_fifo[15][11];	// L7952
  hls::stream< int8_t > &v7176 /* v7176[16] */ = A_fifo[11][16];	// L7953
  hls::stream< int32_t > &v7177 /* v7177[16] */ = P_fifo[15][12];	// L7954
  PE_kernel_ws_gemm_15_11(v7174, v7175, v7176, v7177, v6401, 11, 15);	// L7955
  hls::stream< int8_t > &v7178 /* v7178[16] */ = A_fifo[12][0];	// L7956
  hls::stream< int32_t > &v7179 /* v7179[16] */ = P_fifo[0][12];	// L7957
  hls::stream< int8_t > &v7180 /* v7180[16] */ = A_fifo[12][1];	// L7958
  hls::stream< int32_t > &v7181 /* v7181[16] */ = P_fifo[0][13];	// L7959
  PE_kernel_ws_gemm_0_12(v7178, v7179, v7180, v7181, v6401, 12, 0);	// L7960
  hls::stream< int8_t > &v7182 /* v7182[16] */ = A_fifo[12][1];	// L7961
  hls::stream< int32_t > &v7183 /* v7183[16] */ = P_fifo[1][12];	// L7962
  hls::stream< int8_t > &v7184 /* v7184[16] */ = A_fifo[12][2];	// L7963
  hls::stream< int32_t > &v7185 /* v7185[16] */ = P_fifo[1][13];	// L7964
  PE_kernel_ws_gemm_1_12(v7182, v7183, v7184, v7185, v6401, 12, 1);	// L7965
  hls::stream< int8_t > &v7186 /* v7186[16] */ = A_fifo[12][2];	// L7966
  hls::stream< int32_t > &v7187 /* v7187[16] */ = P_fifo[2][12];	// L7967
  hls::stream< int8_t > &v7188 /* v7188[16] */ = A_fifo[12][3];	// L7968
  hls::stream< int32_t > &v7189 /* v7189[16] */ = P_fifo[2][13];	// L7969
  PE_kernel_ws_gemm_2_12(v7186, v7187, v7188, v7189, v6401, 12, 2);	// L7970
  hls::stream< int8_t > &v7190 /* v7190[16] */ = A_fifo[12][3];	// L7971
  hls::stream< int32_t > &v7191 /* v7191[16] */ = P_fifo[3][12];	// L7972
  hls::stream< int8_t > &v7192 /* v7192[16] */ = A_fifo[12][4];	// L7973
  hls::stream< int32_t > &v7193 /* v7193[16] */ = P_fifo[3][13];	// L7974
  PE_kernel_ws_gemm_3_12(v7190, v7191, v7192, v7193, v6401, 12, 3);	// L7975
  hls::stream< int8_t > &v7194 /* v7194[16] */ = A_fifo[12][4];	// L7976
  hls::stream< int32_t > &v7195 /* v7195[16] */ = P_fifo[4][12];	// L7977
  hls::stream< int8_t > &v7196 /* v7196[16] */ = A_fifo[12][5];	// L7978
  hls::stream< int32_t > &v7197 /* v7197[16] */ = P_fifo[4][13];	// L7979
  PE_kernel_ws_gemm_4_12(v7194, v7195, v7196, v7197, v6401, 12, 4);	// L7980
  hls::stream< int8_t > &v7198 /* v7198[16] */ = A_fifo[12][5];	// L7981
  hls::stream< int32_t > &v7199 /* v7199[16] */ = P_fifo[5][12];	// L7982
  hls::stream< int8_t > &v7200 /* v7200[16] */ = A_fifo[12][6];	// L7983
  hls::stream< int32_t > &v7201 /* v7201[16] */ = P_fifo[5][13];	// L7984
  PE_kernel_ws_gemm_5_12(v7198, v7199, v7200, v7201, v6401, 12, 5);	// L7985
  hls::stream< int8_t > &v7202 /* v7202[16] */ = A_fifo[12][6];	// L7986
  hls::stream< int32_t > &v7203 /* v7203[16] */ = P_fifo[6][12];	// L7987
  hls::stream< int8_t > &v7204 /* v7204[16] */ = A_fifo[12][7];	// L7988
  hls::stream< int32_t > &v7205 /* v7205[16] */ = P_fifo[6][13];	// L7989
  PE_kernel_ws_gemm_6_12(v7202, v7203, v7204, v7205, v6401, 12, 6);	// L7990
  hls::stream< int8_t > &v7206 /* v7206[16] */ = A_fifo[12][7];	// L7991
  hls::stream< int32_t > &v7207 /* v7207[16] */ = P_fifo[7][12];	// L7992
  hls::stream< int8_t > &v7208 /* v7208[16] */ = A_fifo[12][8];	// L7993
  hls::stream< int32_t > &v7209 /* v7209[16] */ = P_fifo[7][13];	// L7994
  PE_kernel_ws_gemm_7_12(v7206, v7207, v7208, v7209, v6401, 12, 7);	// L7995
  hls::stream< int8_t > &v7210 /* v7210[16] */ = A_fifo[12][8];	// L7996
  hls::stream< int32_t > &v7211 /* v7211[16] */ = P_fifo[8][12];	// L7997
  hls::stream< int8_t > &v7212 /* v7212[16] */ = A_fifo[12][9];	// L7998
  hls::stream< int32_t > &v7213 /* v7213[16] */ = P_fifo[8][13];	// L7999
  PE_kernel_ws_gemm_8_12(v7210, v7211, v7212, v7213, v6401, 12, 8);	// L8000
  hls::stream< int8_t > &v7214 /* v7214[16] */ = A_fifo[12][9];	// L8001
  hls::stream< int32_t > &v7215 /* v7215[16] */ = P_fifo[9][12];	// L8002
  hls::stream< int8_t > &v7216 /* v7216[16] */ = A_fifo[12][10];	// L8003
  hls::stream< int32_t > &v7217 /* v7217[16] */ = P_fifo[9][13];	// L8004
  PE_kernel_ws_gemm_9_12(v7214, v7215, v7216, v7217, v6401, 12, 9);	// L8005
  hls::stream< int8_t > &v7218 /* v7218[16] */ = A_fifo[12][10];	// L8006
  hls::stream< int32_t > &v7219 /* v7219[16] */ = P_fifo[10][12];	// L8007
  hls::stream< int8_t > &v7220 /* v7220[16] */ = A_fifo[12][11];	// L8008
  hls::stream< int32_t > &v7221 /* v7221[16] */ = P_fifo[10][13];	// L8009
  PE_kernel_ws_gemm_10_12(v7218, v7219, v7220, v7221, v6401, 12, 10);	// L8010
  hls::stream< int8_t > &v7222 /* v7222[16] */ = A_fifo[12][11];	// L8011
  hls::stream< int32_t > &v7223 /* v7223[16] */ = P_fifo[11][12];	// L8012
  hls::stream< int8_t > &v7224 /* v7224[16] */ = A_fifo[12][12];	// L8013
  hls::stream< int32_t > &v7225 /* v7225[16] */ = P_fifo[11][13];	// L8014
  PE_kernel_ws_gemm_11_12(v7222, v7223, v7224, v7225, v6401, 12, 11);	// L8015
  hls::stream< int8_t > &v7226 /* v7226[16] */ = A_fifo[12][12];	// L8016
  hls::stream< int32_t > &v7227 /* v7227[16] */ = P_fifo[12][12];	// L8017
  hls::stream< int8_t > &v7228 /* v7228[16] */ = A_fifo[12][13];	// L8018
  hls::stream< int32_t > &v7229 /* v7229[16] */ = P_fifo[12][13];	// L8019
  PE_kernel_ws_gemm_12_12(v7226, v7227, v7228, v7229, v6401, 12, 12);	// L8020
  hls::stream< int8_t > &v7230 /* v7230[16] */ = A_fifo[12][13];	// L8021
  hls::stream< int32_t > &v7231 /* v7231[16] */ = P_fifo[13][12];	// L8022
  hls::stream< int8_t > &v7232 /* v7232[16] */ = A_fifo[12][14];	// L8023
  hls::stream< int32_t > &v7233 /* v7233[16] */ = P_fifo[13][13];	// L8024
  PE_kernel_ws_gemm_13_12(v7230, v7231, v7232, v7233, v6401, 12, 13);	// L8025
  hls::stream< int8_t > &v7234 /* v7234[16] */ = A_fifo[12][14];	// L8026
  hls::stream< int32_t > &v7235 /* v7235[16] */ = P_fifo[14][12];	// L8027
  hls::stream< int8_t > &v7236 /* v7236[16] */ = A_fifo[12][15];	// L8028
  hls::stream< int32_t > &v7237 /* v7237[16] */ = P_fifo[14][13];	// L8029
  PE_kernel_ws_gemm_14_12(v7234, v7235, v7236, v7237, v6401, 12, 14);	// L8030
  hls::stream< int8_t > &v7238 /* v7238[16] */ = A_fifo[12][15];	// L8031
  hls::stream< int32_t > &v7239 /* v7239[16] */ = P_fifo[15][12];	// L8032
  hls::stream< int8_t > &v7240 /* v7240[16] */ = A_fifo[12][16];	// L8033
  hls::stream< int32_t > &v7241 /* v7241[16] */ = P_fifo[15][13];	// L8034
  PE_kernel_ws_gemm_15_12(v7238, v7239, v7240, v7241, v6401, 12, 15);	// L8035
  hls::stream< int8_t > &v7242 /* v7242[16] */ = A_fifo[13][0];	// L8036
  hls::stream< int32_t > &v7243 /* v7243[16] */ = P_fifo[0][13];	// L8037
  hls::stream< int8_t > &v7244 /* v7244[16] */ = A_fifo[13][1];	// L8038
  hls::stream< int32_t > &v7245 /* v7245[16] */ = P_fifo[0][14];	// L8039
  PE_kernel_ws_gemm_0_13(v7242, v7243, v7244, v7245, v6401, 13, 0);	// L8040
  hls::stream< int8_t > &v7246 /* v7246[16] */ = A_fifo[13][1];	// L8041
  hls::stream< int32_t > &v7247 /* v7247[16] */ = P_fifo[1][13];	// L8042
  hls::stream< int8_t > &v7248 /* v7248[16] */ = A_fifo[13][2];	// L8043
  hls::stream< int32_t > &v7249 /* v7249[16] */ = P_fifo[1][14];	// L8044
  PE_kernel_ws_gemm_1_13(v7246, v7247, v7248, v7249, v6401, 13, 1);	// L8045
  hls::stream< int8_t > &v7250 /* v7250[16] */ = A_fifo[13][2];	// L8046
  hls::stream< int32_t > &v7251 /* v7251[16] */ = P_fifo[2][13];	// L8047
  hls::stream< int8_t > &v7252 /* v7252[16] */ = A_fifo[13][3];	// L8048
  hls::stream< int32_t > &v7253 /* v7253[16] */ = P_fifo[2][14];	// L8049
  PE_kernel_ws_gemm_2_13(v7250, v7251, v7252, v7253, v6401, 13, 2);	// L8050
  hls::stream< int8_t > &v7254 /* v7254[16] */ = A_fifo[13][3];	// L8051
  hls::stream< int32_t > &v7255 /* v7255[16] */ = P_fifo[3][13];	// L8052
  hls::stream< int8_t > &v7256 /* v7256[16] */ = A_fifo[13][4];	// L8053
  hls::stream< int32_t > &v7257 /* v7257[16] */ = P_fifo[3][14];	// L8054
  PE_kernel_ws_gemm_3_13(v7254, v7255, v7256, v7257, v6401, 13, 3);	// L8055
  hls::stream< int8_t > &v7258 /* v7258[16] */ = A_fifo[13][4];	// L8056
  hls::stream< int32_t > &v7259 /* v7259[16] */ = P_fifo[4][13];	// L8057
  hls::stream< int8_t > &v7260 /* v7260[16] */ = A_fifo[13][5];	// L8058
  hls::stream< int32_t > &v7261 /* v7261[16] */ = P_fifo[4][14];	// L8059
  PE_kernel_ws_gemm_4_13(v7258, v7259, v7260, v7261, v6401, 13, 4);	// L8060
  hls::stream< int8_t > &v7262 /* v7262[16] */ = A_fifo[13][5];	// L8061
  hls::stream< int32_t > &v7263 /* v7263[16] */ = P_fifo[5][13];	// L8062
  hls::stream< int8_t > &v7264 /* v7264[16] */ = A_fifo[13][6];	// L8063
  hls::stream< int32_t > &v7265 /* v7265[16] */ = P_fifo[5][14];	// L8064
  PE_kernel_ws_gemm_5_13(v7262, v7263, v7264, v7265, v6401, 13, 5);	// L8065
  hls::stream< int8_t > &v7266 /* v7266[16] */ = A_fifo[13][6];	// L8066
  hls::stream< int32_t > &v7267 /* v7267[16] */ = P_fifo[6][13];	// L8067
  hls::stream< int8_t > &v7268 /* v7268[16] */ = A_fifo[13][7];	// L8068
  hls::stream< int32_t > &v7269 /* v7269[16] */ = P_fifo[6][14];	// L8069
  PE_kernel_ws_gemm_6_13(v7266, v7267, v7268, v7269, v6401, 13, 6);	// L8070
  hls::stream< int8_t > &v7270 /* v7270[16] */ = A_fifo[13][7];	// L8071
  hls::stream< int32_t > &v7271 /* v7271[16] */ = P_fifo[7][13];	// L8072
  hls::stream< int8_t > &v7272 /* v7272[16] */ = A_fifo[13][8];	// L8073
  hls::stream< int32_t > &v7273 /* v7273[16] */ = P_fifo[7][14];	// L8074
  PE_kernel_ws_gemm_7_13(v7270, v7271, v7272, v7273, v6401, 13, 7);	// L8075
  hls::stream< int8_t > &v7274 /* v7274[16] */ = A_fifo[13][8];	// L8076
  hls::stream< int32_t > &v7275 /* v7275[16] */ = P_fifo[8][13];	// L8077
  hls::stream< int8_t > &v7276 /* v7276[16] */ = A_fifo[13][9];	// L8078
  hls::stream< int32_t > &v7277 /* v7277[16] */ = P_fifo[8][14];	// L8079
  PE_kernel_ws_gemm_8_13(v7274, v7275, v7276, v7277, v6401, 13, 8);	// L8080
  hls::stream< int8_t > &v7278 /* v7278[16] */ = A_fifo[13][9];	// L8081
  hls::stream< int32_t > &v7279 /* v7279[16] */ = P_fifo[9][13];	// L8082
  hls::stream< int8_t > &v7280 /* v7280[16] */ = A_fifo[13][10];	// L8083
  hls::stream< int32_t > &v7281 /* v7281[16] */ = P_fifo[9][14];	// L8084
  PE_kernel_ws_gemm_9_13(v7278, v7279, v7280, v7281, v6401, 13, 9);	// L8085
  hls::stream< int8_t > &v7282 /* v7282[16] */ = A_fifo[13][10];	// L8086
  hls::stream< int32_t > &v7283 /* v7283[16] */ = P_fifo[10][13];	// L8087
  hls::stream< int8_t > &v7284 /* v7284[16] */ = A_fifo[13][11];	// L8088
  hls::stream< int32_t > &v7285 /* v7285[16] */ = P_fifo[10][14];	// L8089
  PE_kernel_ws_gemm_10_13(v7282, v7283, v7284, v7285, v6401, 13, 10);	// L8090
  hls::stream< int8_t > &v7286 /* v7286[16] */ = A_fifo[13][11];	// L8091
  hls::stream< int32_t > &v7287 /* v7287[16] */ = P_fifo[11][13];	// L8092
  hls::stream< int8_t > &v7288 /* v7288[16] */ = A_fifo[13][12];	// L8093
  hls::stream< int32_t > &v7289 /* v7289[16] */ = P_fifo[11][14];	// L8094
  PE_kernel_ws_gemm_11_13(v7286, v7287, v7288, v7289, v6401, 13, 11);	// L8095
  hls::stream< int8_t > &v7290 /* v7290[16] */ = A_fifo[13][12];	// L8096
  hls::stream< int32_t > &v7291 /* v7291[16] */ = P_fifo[12][13];	// L8097
  hls::stream< int8_t > &v7292 /* v7292[16] */ = A_fifo[13][13];	// L8098
  hls::stream< int32_t > &v7293 /* v7293[16] */ = P_fifo[12][14];	// L8099
  PE_kernel_ws_gemm_12_13(v7290, v7291, v7292, v7293, v6401, 13, 12);	// L8100
  hls::stream< int8_t > &v7294 /* v7294[16] */ = A_fifo[13][13];	// L8101
  hls::stream< int32_t > &v7295 /* v7295[16] */ = P_fifo[13][13];	// L8102
  hls::stream< int8_t > &v7296 /* v7296[16] */ = A_fifo[13][14];	// L8103
  hls::stream< int32_t > &v7297 /* v7297[16] */ = P_fifo[13][14];	// L8104
  PE_kernel_ws_gemm_13_13(v7294, v7295, v7296, v7297, v6401, 13, 13);	// L8105
  hls::stream< int8_t > &v7298 /* v7298[16] */ = A_fifo[13][14];	// L8106
  hls::stream< int32_t > &v7299 /* v7299[16] */ = P_fifo[14][13];	// L8107
  hls::stream< int8_t > &v7300 /* v7300[16] */ = A_fifo[13][15];	// L8108
  hls::stream< int32_t > &v7301 /* v7301[16] */ = P_fifo[14][14];	// L8109
  PE_kernel_ws_gemm_14_13(v7298, v7299, v7300, v7301, v6401, 13, 14);	// L8110
  hls::stream< int8_t > &v7302 /* v7302[16] */ = A_fifo[13][15];	// L8111
  hls::stream< int32_t > &v7303 /* v7303[16] */ = P_fifo[15][13];	// L8112
  hls::stream< int8_t > &v7304 /* v7304[16] */ = A_fifo[13][16];	// L8113
  hls::stream< int32_t > &v7305 /* v7305[16] */ = P_fifo[15][14];	// L8114
  PE_kernel_ws_gemm_15_13(v7302, v7303, v7304, v7305, v6401, 13, 15);	// L8115
  hls::stream< int8_t > &v7306 /* v7306[16] */ = A_fifo[14][0];	// L8116
  hls::stream< int32_t > &v7307 /* v7307[16] */ = P_fifo[0][14];	// L8117
  hls::stream< int8_t > &v7308 /* v7308[16] */ = A_fifo[14][1];	// L8118
  hls::stream< int32_t > &v7309 /* v7309[16] */ = P_fifo[0][15];	// L8119
  PE_kernel_ws_gemm_0_14(v7306, v7307, v7308, v7309, v6401, 14, 0);	// L8120
  hls::stream< int8_t > &v7310 /* v7310[16] */ = A_fifo[14][1];	// L8121
  hls::stream< int32_t > &v7311 /* v7311[16] */ = P_fifo[1][14];	// L8122
  hls::stream< int8_t > &v7312 /* v7312[16] */ = A_fifo[14][2];	// L8123
  hls::stream< int32_t > &v7313 /* v7313[16] */ = P_fifo[1][15];	// L8124
  PE_kernel_ws_gemm_1_14(v7310, v7311, v7312, v7313, v6401, 14, 1);	// L8125
  hls::stream< int8_t > &v7314 /* v7314[16] */ = A_fifo[14][2];	// L8126
  hls::stream< int32_t > &v7315 /* v7315[16] */ = P_fifo[2][14];	// L8127
  hls::stream< int8_t > &v7316 /* v7316[16] */ = A_fifo[14][3];	// L8128
  hls::stream< int32_t > &v7317 /* v7317[16] */ = P_fifo[2][15];	// L8129
  PE_kernel_ws_gemm_2_14(v7314, v7315, v7316, v7317, v6401, 14, 2);	// L8130
  hls::stream< int8_t > &v7318 /* v7318[16] */ = A_fifo[14][3];	// L8131
  hls::stream< int32_t > &v7319 /* v7319[16] */ = P_fifo[3][14];	// L8132
  hls::stream< int8_t > &v7320 /* v7320[16] */ = A_fifo[14][4];	// L8133
  hls::stream< int32_t > &v7321 /* v7321[16] */ = P_fifo[3][15];	// L8134
  PE_kernel_ws_gemm_3_14(v7318, v7319, v7320, v7321, v6401, 14, 3);	// L8135
  hls::stream< int8_t > &v7322 /* v7322[16] */ = A_fifo[14][4];	// L8136
  hls::stream< int32_t > &v7323 /* v7323[16] */ = P_fifo[4][14];	// L8137
  hls::stream< int8_t > &v7324 /* v7324[16] */ = A_fifo[14][5];	// L8138
  hls::stream< int32_t > &v7325 /* v7325[16] */ = P_fifo[4][15];	// L8139
  PE_kernel_ws_gemm_4_14(v7322, v7323, v7324, v7325, v6401, 14, 4);	// L8140
  hls::stream< int8_t > &v7326 /* v7326[16] */ = A_fifo[14][5];	// L8141
  hls::stream< int32_t > &v7327 /* v7327[16] */ = P_fifo[5][14];	// L8142
  hls::stream< int8_t > &v7328 /* v7328[16] */ = A_fifo[14][6];	// L8143
  hls::stream< int32_t > &v7329 /* v7329[16] */ = P_fifo[5][15];	// L8144
  PE_kernel_ws_gemm_5_14(v7326, v7327, v7328, v7329, v6401, 14, 5);	// L8145
  hls::stream< int8_t > &v7330 /* v7330[16] */ = A_fifo[14][6];	// L8146
  hls::stream< int32_t > &v7331 /* v7331[16] */ = P_fifo[6][14];	// L8147
  hls::stream< int8_t > &v7332 /* v7332[16] */ = A_fifo[14][7];	// L8148
  hls::stream< int32_t > &v7333 /* v7333[16] */ = P_fifo[6][15];	// L8149
  PE_kernel_ws_gemm_6_14(v7330, v7331, v7332, v7333, v6401, 14, 6);	// L8150
  hls::stream< int8_t > &v7334 /* v7334[16] */ = A_fifo[14][7];	// L8151
  hls::stream< int32_t > &v7335 /* v7335[16] */ = P_fifo[7][14];	// L8152
  hls::stream< int8_t > &v7336 /* v7336[16] */ = A_fifo[14][8];	// L8153
  hls::stream< int32_t > &v7337 /* v7337[16] */ = P_fifo[7][15];	// L8154
  PE_kernel_ws_gemm_7_14(v7334, v7335, v7336, v7337, v6401, 14, 7);	// L8155
  hls::stream< int8_t > &v7338 /* v7338[16] */ = A_fifo[14][8];	// L8156
  hls::stream< int32_t > &v7339 /* v7339[16] */ = P_fifo[8][14];	// L8157
  hls::stream< int8_t > &v7340 /* v7340[16] */ = A_fifo[14][9];	// L8158
  hls::stream< int32_t > &v7341 /* v7341[16] */ = P_fifo[8][15];	// L8159
  PE_kernel_ws_gemm_8_14(v7338, v7339, v7340, v7341, v6401, 14, 8);	// L8160
  hls::stream< int8_t > &v7342 /* v7342[16] */ = A_fifo[14][9];	// L8161
  hls::stream< int32_t > &v7343 /* v7343[16] */ = P_fifo[9][14];	// L8162
  hls::stream< int8_t > &v7344 /* v7344[16] */ = A_fifo[14][10];	// L8163
  hls::stream< int32_t > &v7345 /* v7345[16] */ = P_fifo[9][15];	// L8164
  PE_kernel_ws_gemm_9_14(v7342, v7343, v7344, v7345, v6401, 14, 9);	// L8165
  hls::stream< int8_t > &v7346 /* v7346[16] */ = A_fifo[14][10];	// L8166
  hls::stream< int32_t > &v7347 /* v7347[16] */ = P_fifo[10][14];	// L8167
  hls::stream< int8_t > &v7348 /* v7348[16] */ = A_fifo[14][11];	// L8168
  hls::stream< int32_t > &v7349 /* v7349[16] */ = P_fifo[10][15];	// L8169
  PE_kernel_ws_gemm_10_14(v7346, v7347, v7348, v7349, v6401, 14, 10);	// L8170
  hls::stream< int8_t > &v7350 /* v7350[16] */ = A_fifo[14][11];	// L8171
  hls::stream< int32_t > &v7351 /* v7351[16] */ = P_fifo[11][14];	// L8172
  hls::stream< int8_t > &v7352 /* v7352[16] */ = A_fifo[14][12];	// L8173
  hls::stream< int32_t > &v7353 /* v7353[16] */ = P_fifo[11][15];	// L8174
  PE_kernel_ws_gemm_11_14(v7350, v7351, v7352, v7353, v6401, 14, 11);	// L8175
  hls::stream< int8_t > &v7354 /* v7354[16] */ = A_fifo[14][12];	// L8176
  hls::stream< int32_t > &v7355 /* v7355[16] */ = P_fifo[12][14];	// L8177
  hls::stream< int8_t > &v7356 /* v7356[16] */ = A_fifo[14][13];	// L8178
  hls::stream< int32_t > &v7357 /* v7357[16] */ = P_fifo[12][15];	// L8179
  PE_kernel_ws_gemm_12_14(v7354, v7355, v7356, v7357, v6401, 14, 12);	// L8180
  hls::stream< int8_t > &v7358 /* v7358[16] */ = A_fifo[14][13];	// L8181
  hls::stream< int32_t > &v7359 /* v7359[16] */ = P_fifo[13][14];	// L8182
  hls::stream< int8_t > &v7360 /* v7360[16] */ = A_fifo[14][14];	// L8183
  hls::stream< int32_t > &v7361 /* v7361[16] */ = P_fifo[13][15];	// L8184
  PE_kernel_ws_gemm_13_14(v7358, v7359, v7360, v7361, v6401, 14, 13);	// L8185
  hls::stream< int8_t > &v7362 /* v7362[16] */ = A_fifo[14][14];	// L8186
  hls::stream< int32_t > &v7363 /* v7363[16] */ = P_fifo[14][14];	// L8187
  hls::stream< int8_t > &v7364 /* v7364[16] */ = A_fifo[14][15];	// L8188
  hls::stream< int32_t > &v7365 /* v7365[16] */ = P_fifo[14][15];	// L8189
  PE_kernel_ws_gemm_14_14(v7362, v7363, v7364, v7365, v6401, 14, 14);	// L8190
  hls::stream< int8_t > &v7366 /* v7366[16] */ = A_fifo[14][15];	// L8191
  hls::stream< int32_t > &v7367 /* v7367[16] */ = P_fifo[15][14];	// L8192
  hls::stream< int8_t > &v7368 /* v7368[16] */ = A_fifo[14][16];	// L8193
  hls::stream< int32_t > &v7369 /* v7369[16] */ = P_fifo[15][15];	// L8194
  PE_kernel_ws_gemm_15_14(v7366, v7367, v7368, v7369, v6401, 14, 15);	// L8195
  hls::stream< int8_t > &v7370 /* v7370[16] */ = A_fifo[15][0];	// L8196
  hls::stream< int32_t > &v7371 /* v7371[16] */ = P_fifo[0][15];	// L8197
  hls::stream< int8_t > &v7372 /* v7372[16] */ = A_fifo[15][1];	// L8198
  hls::stream< int32_t > &v7373 /* v7373[16] */ = P_fifo[0][16];	// L8199
  PE_kernel_ws_gemm_0_15(v7370, v7371, v7372, v7373, v6401, 15, 0);	// L8200
  hls::stream< int8_t > &v7374 /* v7374[16] */ = A_fifo[15][1];	// L8201
  hls::stream< int32_t > &v7375 /* v7375[16] */ = P_fifo[1][15];	// L8202
  hls::stream< int8_t > &v7376 /* v7376[16] */ = A_fifo[15][2];	// L8203
  hls::stream< int32_t > &v7377 /* v7377[16] */ = P_fifo[1][16];	// L8204
  PE_kernel_ws_gemm_1_15(v7374, v7375, v7376, v7377, v6401, 15, 1);	// L8205
  hls::stream< int8_t > &v7378 /* v7378[16] */ = A_fifo[15][2];	// L8206
  hls::stream< int32_t > &v7379 /* v7379[16] */ = P_fifo[2][15];	// L8207
  hls::stream< int8_t > &v7380 /* v7380[16] */ = A_fifo[15][3];	// L8208
  hls::stream< int32_t > &v7381 /* v7381[16] */ = P_fifo[2][16];	// L8209
  PE_kernel_ws_gemm_2_15(v7378, v7379, v7380, v7381, v6401, 15, 2);	// L8210
  hls::stream< int8_t > &v7382 /* v7382[16] */ = A_fifo[15][3];	// L8211
  hls::stream< int32_t > &v7383 /* v7383[16] */ = P_fifo[3][15];	// L8212
  hls::stream< int8_t > &v7384 /* v7384[16] */ = A_fifo[15][4];	// L8213
  hls::stream< int32_t > &v7385 /* v7385[16] */ = P_fifo[3][16];	// L8214
  PE_kernel_ws_gemm_3_15(v7382, v7383, v7384, v7385, v6401, 15, 3);	// L8215
  hls::stream< int8_t > &v7386 /* v7386[16] */ = A_fifo[15][4];	// L8216
  hls::stream< int32_t > &v7387 /* v7387[16] */ = P_fifo[4][15];	// L8217
  hls::stream< int8_t > &v7388 /* v7388[16] */ = A_fifo[15][5];	// L8218
  hls::stream< int32_t > &v7389 /* v7389[16] */ = P_fifo[4][16];	// L8219
  PE_kernel_ws_gemm_4_15(v7386, v7387, v7388, v7389, v6401, 15, 4);	// L8220
  hls::stream< int8_t > &v7390 /* v7390[16] */ = A_fifo[15][5];	// L8221
  hls::stream< int32_t > &v7391 /* v7391[16] */ = P_fifo[5][15];	// L8222
  hls::stream< int8_t > &v7392 /* v7392[16] */ = A_fifo[15][6];	// L8223
  hls::stream< int32_t > &v7393 /* v7393[16] */ = P_fifo[5][16];	// L8224
  PE_kernel_ws_gemm_5_15(v7390, v7391, v7392, v7393, v6401, 15, 5);	// L8225
  hls::stream< int8_t > &v7394 /* v7394[16] */ = A_fifo[15][6];	// L8226
  hls::stream< int32_t > &v7395 /* v7395[16] */ = P_fifo[6][15];	// L8227
  hls::stream< int8_t > &v7396 /* v7396[16] */ = A_fifo[15][7];	// L8228
  hls::stream< int32_t > &v7397 /* v7397[16] */ = P_fifo[6][16];	// L8229
  PE_kernel_ws_gemm_6_15(v7394, v7395, v7396, v7397, v6401, 15, 6);	// L8230
  hls::stream< int8_t > &v7398 /* v7398[16] */ = A_fifo[15][7];	// L8231
  hls::stream< int32_t > &v7399 /* v7399[16] */ = P_fifo[7][15];	// L8232
  hls::stream< int8_t > &v7400 /* v7400[16] */ = A_fifo[15][8];	// L8233
  hls::stream< int32_t > &v7401 /* v7401[16] */ = P_fifo[7][16];	// L8234
  PE_kernel_ws_gemm_7_15(v7398, v7399, v7400, v7401, v6401, 15, 7);	// L8235
  hls::stream< int8_t > &v7402 /* v7402[16] */ = A_fifo[15][8];	// L8236
  hls::stream< int32_t > &v7403 /* v7403[16] */ = P_fifo[8][15];	// L8237
  hls::stream< int8_t > &v7404 /* v7404[16] */ = A_fifo[15][9];	// L8238
  hls::stream< int32_t > &v7405 /* v7405[16] */ = P_fifo[8][16];	// L8239
  PE_kernel_ws_gemm_8_15(v7402, v7403, v7404, v7405, v6401, 15, 8);	// L8240
  hls::stream< int8_t > &v7406 /* v7406[16] */ = A_fifo[15][9];	// L8241
  hls::stream< int32_t > &v7407 /* v7407[16] */ = P_fifo[9][15];	// L8242
  hls::stream< int8_t > &v7408 /* v7408[16] */ = A_fifo[15][10];	// L8243
  hls::stream< int32_t > &v7409 /* v7409[16] */ = P_fifo[9][16];	// L8244
  PE_kernel_ws_gemm_9_15(v7406, v7407, v7408, v7409, v6401, 15, 9);	// L8245
  hls::stream< int8_t > &v7410 /* v7410[16] */ = A_fifo[15][10];	// L8246
  hls::stream< int32_t > &v7411 /* v7411[16] */ = P_fifo[10][15];	// L8247
  hls::stream< int8_t > &v7412 /* v7412[16] */ = A_fifo[15][11];	// L8248
  hls::stream< int32_t > &v7413 /* v7413[16] */ = P_fifo[10][16];	// L8249
  PE_kernel_ws_gemm_10_15(v7410, v7411, v7412, v7413, v6401, 15, 10);	// L8250
  hls::stream< int8_t > &v7414 /* v7414[16] */ = A_fifo[15][11];	// L8251
  hls::stream< int32_t > &v7415 /* v7415[16] */ = P_fifo[11][15];	// L8252
  hls::stream< int8_t > &v7416 /* v7416[16] */ = A_fifo[15][12];	// L8253
  hls::stream< int32_t > &v7417 /* v7417[16] */ = P_fifo[11][16];	// L8254
  PE_kernel_ws_gemm_11_15(v7414, v7415, v7416, v7417, v6401, 15, 11);	// L8255
  hls::stream< int8_t > &v7418 /* v7418[16] */ = A_fifo[15][12];	// L8256
  hls::stream< int32_t > &v7419 /* v7419[16] */ = P_fifo[12][15];	// L8257
  hls::stream< int8_t > &v7420 /* v7420[16] */ = A_fifo[15][13];	// L8258
  hls::stream< int32_t > &v7421 /* v7421[16] */ = P_fifo[12][16];	// L8259
  PE_kernel_ws_gemm_12_15(v7418, v7419, v7420, v7421, v6401, 15, 12);	// L8260
  hls::stream< int8_t > &v7422 /* v7422[16] */ = A_fifo[15][13];	// L8261
  hls::stream< int32_t > &v7423 /* v7423[16] */ = P_fifo[13][15];	// L8262
  hls::stream< int8_t > &v7424 /* v7424[16] */ = A_fifo[15][14];	// L8263
  hls::stream< int32_t > &v7425 /* v7425[16] */ = P_fifo[13][16];	// L8264
  PE_kernel_ws_gemm_13_15(v7422, v7423, v7424, v7425, v6401, 15, 13);	// L8265
  hls::stream< int8_t > &v7426 /* v7426[16] */ = A_fifo[15][14];	// L8266
  hls::stream< int32_t > &v7427 /* v7427[16] */ = P_fifo[14][15];	// L8267
  hls::stream< int8_t > &v7428 /* v7428[16] */ = A_fifo[15][15];	// L8268
  hls::stream< int32_t > &v7429 /* v7429[16] */ = P_fifo[14][16];	// L8269
  PE_kernel_ws_gemm_14_15(v7426, v7427, v7428, v7429, v6401, 15, 14);	// L8270
  hls::stream< int8_t > &v7430 /* v7430[16] */ = A_fifo[15][15];	// L8271
  hls::stream< int32_t > &v7431 /* v7431[16] */ = P_fifo[15][15];	// L8272
  hls::stream< int8_t > &v7432 /* v7432[16] */ = A_fifo[15][16];	// L8273
  hls::stream< int32_t > &v7433 /* v7433[16] */ = P_fifo[15][16];	// L8274
  PE_kernel_ws_gemm_15_15(v7430, v7431, v7432, v7433, v6401, 15, 15);	// L8275
  l_data_drain_m257: for (int m257 = 0; m257 < 16; m257++) {	// L8276
    l_S_k_4_k1: for (int k1 = 0; k1 < 16; k1++) {	// L8277
      int8_t v7436 = A_fifo[k1][16].read(); // A_fifo[k1][16][m257];	// L8278
      A_drain[k1] = v7436;	// L8279
    }
    l_S_n_5_n1: for (int n1 = 0; n1 < 16; n1++) {	// L8281
      int32_t v7438 = P_fifo[n1][16].read(); // P_fifo[n1][16][m257];	// L8282
      v6402[m257][n1] = v7438;	// L8283
    }
  }
}

void systolic_ws_gemm(
  int8_t v7439[16][16],
  int8_t v7440[16][16],
  int32_t v7441[16][16]
) {	// L8288
  int8_t local_A[16][16];	// L8289
  #pragma HLS array_partition variable=local_A complete dim=2

  int8_t local_B[16][16];	// L8290
  #pragma HLS array_partition variable=local_B complete dim=1
  #pragma HLS array_partition variable=local_B complete dim=2

  int32_t local_C[16][16];	// L8291
  #pragma HLS array_partition variable=local_C complete dim=2

  l_load_A_am: for (int am = 0; am < 16; am++) {	// L8292
  #pragma HLS pipeline II=1
    l_ak: for (int ak = 0; ak < 16; ak++) {	// L8293
      int8_t v7447 = v7439[am][ak];	// L8294
      local_A[am][ak] = v7447;	// L8295
    }
  }
  l_load_B_bk: for (int bk = 0; bk < 16; bk++) {	// L8298
  #pragma HLS pipeline II=1
    l_bn: for (int bn = 0; bn < 16; bn++) {	// L8299
      int8_t v7450 = v7440[bk][bn];	// L8300
      local_B[bk][bn] = v7450;	// L8301
    }
  }
  systolic_tile_ws_gemm(local_A, local_B, local_C);	// L8304
  l_store_C_sm: for (int sm = 0; sm < 16; sm++) {	// L8305
  #pragma HLS pipeline II=1
    l_sn: for (int sn = 0; sn < 16; sn++) {	// L8306
      int32_t v7453 = local_C[sm][sn];	// L8307
      v7441[sm][sn] = v7453;	// L8308
    }
  }
}

void load_buf0(
  int8_t v7454[256],
  int8_t v7455[16][16]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 16; load_buf0_l_0++) {	//
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 16; load_buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v7458 = v7454[((load_buf0_l_0 * 16) + load_buf0_l_1)];	//
      v7455[load_buf0_l_0][load_buf0_l_1] = v7458;	//
    }
  }
}

void load_buf1(
  int8_t v7459[256],
  int8_t v7460[16][16]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 16; load_buf1_l_0++) {	//
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 16; load_buf1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v7463 = v7459[((load_buf1_l_0 * 16) + load_buf1_l_1)];	//
      v7460[load_buf1_l_0][load_buf1_l_1] = v7463;	//
    }
  }
}

void store_res2(
  int32_t v7464[16][16],
  int32_t v7465[256]
) {	//
  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 16; store_res2_l_0++) {	//
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 16; store_res2_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int32_t v7468 = v7464[store_res2_l_0][store_res2_l_1];	//
      v7465[((store_res2_l_0 * 16) + store_res2_l_1)] = v7468;	//
    }
  }
}

/// This is top function.
void gemm(
  int8_t *v7469,
  int8_t *v7470,
  int32_t *v7471
) {	// L8313
  #pragma HLS interface m_axi port=v7469 offset=slave bundle=gmem0 depth=256
  #pragma HLS interface m_axi port=v7470 offset=slave bundle=gmem1 depth=256
  #pragma HLS interface m_axi port=v7471 offset=slave bundle=gmem2 depth=256
  int8_t buf0[16][16];	//
  load_buf0(v7469, buf0);	//
  int8_t buf1[16][16];	//
  load_buf1(v7470, buf1);	//
  int32_t buf2[16][16];	//
  systolic_ws_gemm(buf0, buf1, buf2);	// L8314
  store_res2(buf2, v7471);	//
}


} // extern "C"

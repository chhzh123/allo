
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

void PE_kernel_gemm_0_0(
  hls::stream< int8_t > &v0 /* v0[8] */,
  hls::stream< int8_t > &v1 /* v1[8] */,
  hls::stream< int8_t > &v2 /* v2[8] */,
  hls::stream< int8_t > &v3 /* v3[8] */,
  int32_t v4[8][8],
  int v5,
  int v6
) {	// L5
  #pragma HLS stream variable=v0 depth=9
  #pragma HLS stream variable=v1 depth=9
  #pragma HLS stream variable=v2 depth=9
  #pragma HLS stream variable=v3 depth=9
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  int32_t v;	// L7
  v = 0;	// L8
  l_reduction_k: for (int k = 0; k < 8; k++) {	// L9
  #pragma HLS pipeline II=1
    int8_t v9 = v0.read(); // v0[k];	// L10
    int8_t a;	// L11
    a = v9;	// L12
    int8_t v11 = v1.read(); // v1[k];	// L13
    int8_t b;	// L14
    b = v11;	// L15
    int8_t v13 = a;	// L16
    int8_t v14 = b;	// L17
    int16_t v15 = v13;	// L18
    int16_t v16 = v14;	// L19
    int16_t v17 = v15 * v16;	// L20
    int32_t v18 = v;	// L21
    ap_int<33> v19 = v18;	// L22
    ap_int<33> v20 = v17;	// L23
    ap_int<33> v21 = v19 + v20;	// L24
    int32_t v22 = v21;	// L25
    v = v22;	// L26
    int8_t v23 = a;	// L27
    v2.write(v23); // v2[k] = v23;	// L28
    int8_t v24 = b;	// L29
    v3.write(v24); // v3[k] = v24;	// L30
  }
  int32_t v25 = v;	// L32
  v4[v5][v6] = v25;	// L33
}

void PE_kernel_gemm_1_0(
  hls::stream< int8_t > &v26 /* v26[8] */,
  hls::stream< int8_t > &v27 /* v27[8] */,
  hls::stream< int8_t > &v28 /* v28[8] */,
  hls::stream< int8_t > &v29 /* v29[8] */,
  int32_t v30[8][8],
  int v31,
  int v32
) {	// L36
  #pragma HLS stream variable=v26 depth=9
  #pragma HLS stream variable=v27 depth=9
  #pragma HLS stream variable=v28 depth=9
  #pragma HLS stream variable=v29 depth=9
  #pragma HLS array_partition variable=v30 complete dim=1
  #pragma HLS array_partition variable=v30 complete dim=2

  int32_t v1;	// L38
  v1 = 0;	// L39
  l_reduction_k1: for (int k1 = 0; k1 < 8; k1++) {	// L40
  #pragma HLS pipeline II=1
    int8_t v35 = v26.read(); // v26[k1];	// L41
    int8_t a1;	// L42
    a1 = v35;	// L43
    int8_t v37 = v27.read(); // v27[k1];	// L44
    int8_t b1;	// L45
    b1 = v37;	// L46
    int8_t v39 = a1;	// L47
    int8_t v40 = b1;	// L48
    int16_t v41 = v39;	// L49
    int16_t v42 = v40;	// L50
    int16_t v43 = v41 * v42;	// L51
    int32_t v44 = v1;	// L52
    ap_int<33> v45 = v44;	// L53
    ap_int<33> v46 = v43;	// L54
    ap_int<33> v47 = v45 + v46;	// L55
    int32_t v48 = v47;	// L56
    v1 = v48;	// L57
    int8_t v49 = a1;	// L58
    v28.write(v49); // v28[k1] = v49;	// L59
    int8_t v50 = b1;	// L60
    v29.write(v50); // v29[k1] = v50;	// L61
  }
  int32_t v51 = v1;	// L63
  v30[v31][v32] = v51;	// L64
}

void PE_kernel_gemm_2_0(
  hls::stream< int8_t > &v52 /* v52[8] */,
  hls::stream< int8_t > &v53 /* v53[8] */,
  hls::stream< int8_t > &v54 /* v54[8] */,
  hls::stream< int8_t > &v55 /* v55[8] */,
  int32_t v56[8][8],
  int v57,
  int v58
) {	// L67
  #pragma HLS stream variable=v52 depth=9
  #pragma HLS stream variable=v53 depth=9
  #pragma HLS stream variable=v54 depth=9
  #pragma HLS stream variable=v55 depth=9
  #pragma HLS array_partition variable=v56 complete dim=1
  #pragma HLS array_partition variable=v56 complete dim=2

  int32_t v2;	// L69
  v2 = 0;	// L70
  l_reduction_k2: for (int k2 = 0; k2 < 8; k2++) {	// L71
  #pragma HLS pipeline II=1
    int8_t v61 = v52.read(); // v52[k2];	// L72
    int8_t a2;	// L73
    a2 = v61;	// L74
    int8_t v63 = v53.read(); // v53[k2];	// L75
    int8_t b2;	// L76
    b2 = v63;	// L77
    int8_t v65 = a2;	// L78
    int8_t v66 = b2;	// L79
    int16_t v67 = v65;	// L80
    int16_t v68 = v66;	// L81
    int16_t v69 = v67 * v68;	// L82
    int32_t v70 = v2;	// L83
    ap_int<33> v71 = v70;	// L84
    ap_int<33> v72 = v69;	// L85
    ap_int<33> v73 = v71 + v72;	// L86
    int32_t v74 = v73;	// L87
    v2 = v74;	// L88
    int8_t v75 = a2;	// L89
    v54.write(v75); // v54[k2] = v75;	// L90
    int8_t v76 = b2;	// L91
    v55.write(v76); // v55[k2] = v76;	// L92
  }
  int32_t v77 = v2;	// L94
  v56[v57][v58] = v77;	// L95
}

void PE_kernel_gemm_3_0(
  hls::stream< int8_t > &v78 /* v78[8] */,
  hls::stream< int8_t > &v79 /* v79[8] */,
  hls::stream< int8_t > &v80 /* v80[8] */,
  hls::stream< int8_t > &v81 /* v81[8] */,
  int32_t v82[8][8],
  int v83,
  int v84
) {	// L98
  #pragma HLS stream variable=v78 depth=9
  #pragma HLS stream variable=v79 depth=9
  #pragma HLS stream variable=v80 depth=9
  #pragma HLS stream variable=v81 depth=9
  #pragma HLS array_partition variable=v82 complete dim=1
  #pragma HLS array_partition variable=v82 complete dim=2

  int32_t v3;	// L100
  v3 = 0;	// L101
  l_reduction_k3: for (int k3 = 0; k3 < 8; k3++) {	// L102
  #pragma HLS pipeline II=1
    int8_t v87 = v78.read(); // v78[k3];	// L103
    int8_t a3;	// L104
    a3 = v87;	// L105
    int8_t v89 = v79.read(); // v79[k3];	// L106
    int8_t b3;	// L107
    b3 = v89;	// L108
    int8_t v91 = a3;	// L109
    int8_t v92 = b3;	// L110
    int16_t v93 = v91;	// L111
    int16_t v94 = v92;	// L112
    int16_t v95 = v93 * v94;	// L113
    int32_t v96 = v3;	// L114
    ap_int<33> v97 = v96;	// L115
    ap_int<33> v98 = v95;	// L116
    ap_int<33> v99 = v97 + v98;	// L117
    int32_t v100 = v99;	// L118
    v3 = v100;	// L119
    int8_t v101 = a3;	// L120
    v80.write(v101); // v80[k3] = v101;	// L121
    int8_t v102 = b3;	// L122
    v81.write(v102); // v81[k3] = v102;	// L123
  }
  int32_t v103 = v3;	// L125
  v82[v83][v84] = v103;	// L126
}

void PE_kernel_gemm_4_0(
  hls::stream< int8_t > &v104 /* v104[8] */,
  hls::stream< int8_t > &v105 /* v105[8] */,
  hls::stream< int8_t > &v106 /* v106[8] */,
  hls::stream< int8_t > &v107 /* v107[8] */,
  int32_t v108[8][8],
  int v109,
  int v110
) {	// L129
  #pragma HLS stream variable=v104 depth=9
  #pragma HLS stream variable=v105 depth=9
  #pragma HLS stream variable=v106 depth=9
  #pragma HLS stream variable=v107 depth=9
  #pragma HLS array_partition variable=v108 complete dim=1
  #pragma HLS array_partition variable=v108 complete dim=2

  int32_t v4;	// L131
  v4 = 0;	// L132
  l_reduction_k4: for (int k4 = 0; k4 < 8; k4++) {	// L133
  #pragma HLS pipeline II=1
    int8_t v113 = v104.read(); // v104[k4];	// L134
    int8_t a4;	// L135
    a4 = v113;	// L136
    int8_t v115 = v105.read(); // v105[k4];	// L137
    int8_t b4;	// L138
    b4 = v115;	// L139
    int8_t v117 = a4;	// L140
    int8_t v118 = b4;	// L141
    int16_t v119 = v117;	// L142
    int16_t v120 = v118;	// L143
    int16_t v121 = v119 * v120;	// L144
    int32_t v122 = v4;	// L145
    ap_int<33> v123 = v122;	// L146
    ap_int<33> v124 = v121;	// L147
    ap_int<33> v125 = v123 + v124;	// L148
    int32_t v126 = v125;	// L149
    v4 = v126;	// L150
    int8_t v127 = a4;	// L151
    v106.write(v127); // v106[k4] = v127;	// L152
    int8_t v128 = b4;	// L153
    v107.write(v128); // v107[k4] = v128;	// L154
  }
  int32_t v129 = v4;	// L156
  v108[v109][v110] = v129;	// L157
}

void PE_kernel_gemm_5_0(
  hls::stream< int8_t > &v130 /* v130[8] */,
  hls::stream< int8_t > &v131 /* v131[8] */,
  hls::stream< int8_t > &v132 /* v132[8] */,
  hls::stream< int8_t > &v133 /* v133[8] */,
  int32_t v134[8][8],
  int v135,
  int v136
) {	// L160
  #pragma HLS stream variable=v130 depth=9
  #pragma HLS stream variable=v131 depth=9
  #pragma HLS stream variable=v132 depth=9
  #pragma HLS stream variable=v133 depth=9
  #pragma HLS array_partition variable=v134 complete dim=1
  #pragma HLS array_partition variable=v134 complete dim=2

  int32_t v5;	// L162
  v5 = 0;	// L163
  l_reduction_k5: for (int k5 = 0; k5 < 8; k5++) {	// L164
  #pragma HLS pipeline II=1
    int8_t v139 = v130.read(); // v130[k5];	// L165
    int8_t a5;	// L166
    a5 = v139;	// L167
    int8_t v141 = v131.read(); // v131[k5];	// L168
    int8_t b5;	// L169
    b5 = v141;	// L170
    int8_t v143 = a5;	// L171
    int8_t v144 = b5;	// L172
    int16_t v145 = v143;	// L173
    int16_t v146 = v144;	// L174
    int16_t v147 = v145 * v146;	// L175
    int32_t v148 = v5;	// L176
    ap_int<33> v149 = v148;	// L177
    ap_int<33> v150 = v147;	// L178
    ap_int<33> v151 = v149 + v150;	// L179
    int32_t v152 = v151;	// L180
    v5 = v152;	// L181
    int8_t v153 = a5;	// L182
    v132.write(v153); // v132[k5] = v153;	// L183
    int8_t v154 = b5;	// L184
    v133.write(v154); // v133[k5] = v154;	// L185
  }
  int32_t v155 = v5;	// L187
  v134[v135][v136] = v155;	// L188
}

void PE_kernel_gemm_6_0(
  hls::stream< int8_t > &v156 /* v156[8] */,
  hls::stream< int8_t > &v157 /* v157[8] */,
  hls::stream< int8_t > &v158 /* v158[8] */,
  hls::stream< int8_t > &v159 /* v159[8] */,
  int32_t v160[8][8],
  int v161,
  int v162
) {	// L191
  #pragma HLS stream variable=v156 depth=9
  #pragma HLS stream variable=v157 depth=9
  #pragma HLS stream variable=v158 depth=9
  #pragma HLS stream variable=v159 depth=9
  #pragma HLS array_partition variable=v160 complete dim=1
  #pragma HLS array_partition variable=v160 complete dim=2

  int32_t v6;	// L193
  v6 = 0;	// L194
  l_reduction_k6: for (int k6 = 0; k6 < 8; k6++) {	// L195
  #pragma HLS pipeline II=1
    int8_t v165 = v156.read(); // v156[k6];	// L196
    int8_t a6;	// L197
    a6 = v165;	// L198
    int8_t v167 = v157.read(); // v157[k6];	// L199
    int8_t b6;	// L200
    b6 = v167;	// L201
    int8_t v169 = a6;	// L202
    int8_t v170 = b6;	// L203
    int16_t v171 = v169;	// L204
    int16_t v172 = v170;	// L205
    int16_t v173 = v171 * v172;	// L206
    int32_t v174 = v6;	// L207
    ap_int<33> v175 = v174;	// L208
    ap_int<33> v176 = v173;	// L209
    ap_int<33> v177 = v175 + v176;	// L210
    int32_t v178 = v177;	// L211
    v6 = v178;	// L212
    int8_t v179 = a6;	// L213
    v158.write(v179); // v158[k6] = v179;	// L214
    int8_t v180 = b6;	// L215
    v159.write(v180); // v159[k6] = v180;	// L216
  }
  int32_t v181 = v6;	// L218
  v160[v161][v162] = v181;	// L219
}

void PE_kernel_gemm_7_0(
  hls::stream< int8_t > &v182 /* v182[8] */,
  hls::stream< int8_t > &v183 /* v183[8] */,
  hls::stream< int8_t > &v184 /* v184[8] */,
  hls::stream< int8_t > &v185 /* v185[8] */,
  int32_t v186[8][8],
  int v187,
  int v188
) {	// L222
  #pragma HLS stream variable=v182 depth=9
  #pragma HLS stream variable=v183 depth=9
  #pragma HLS stream variable=v184 depth=9
  #pragma HLS stream variable=v185 depth=9
  #pragma HLS array_partition variable=v186 complete dim=1
  #pragma HLS array_partition variable=v186 complete dim=2

  int32_t v7;	// L224
  v7 = 0;	// L225
  l_reduction_k7: for (int k7 = 0; k7 < 8; k7++) {	// L226
  #pragma HLS pipeline II=1
    int8_t v191 = v182.read(); // v182[k7];	// L227
    int8_t a7;	// L228
    a7 = v191;	// L229
    int8_t v193 = v183.read(); // v183[k7];	// L230
    int8_t b7;	// L231
    b7 = v193;	// L232
    int8_t v195 = a7;	// L233
    int8_t v196 = b7;	// L234
    int16_t v197 = v195;	// L235
    int16_t v198 = v196;	// L236
    int16_t v199 = v197 * v198;	// L237
    int32_t v200 = v7;	// L238
    ap_int<33> v201 = v200;	// L239
    ap_int<33> v202 = v199;	// L240
    ap_int<33> v203 = v201 + v202;	// L241
    int32_t v204 = v203;	// L242
    v7 = v204;	// L243
    int8_t v205 = a7;	// L244
    v184.write(v205); // v184[k7] = v205;	// L245
    int8_t v206 = b7;	// L246
    v185.write(v206); // v185[k7] = v206;	// L247
  }
  int32_t v207 = v7;	// L249
  v186[v187][v188] = v207;	// L250
}

void PE_kernel_gemm_0_1(
  hls::stream< int8_t > &v208 /* v208[8] */,
  hls::stream< int8_t > &v209 /* v209[8] */,
  hls::stream< int8_t > &v210 /* v210[8] */,
  hls::stream< int8_t > &v211 /* v211[8] */,
  int32_t v212[8][8],
  int v213,
  int v214
) {	// L253
  #pragma HLS stream variable=v208 depth=9
  #pragma HLS stream variable=v209 depth=9
  #pragma HLS stream variable=v210 depth=9
  #pragma HLS stream variable=v211 depth=9
  #pragma HLS array_partition variable=v212 complete dim=1
  #pragma HLS array_partition variable=v212 complete dim=2

  int32_t v8;	// L255
  v8 = 0;	// L256
  l_reduction_k8: for (int k8 = 0; k8 < 8; k8++) {	// L257
  #pragma HLS pipeline II=1
    int8_t v217 = v208.read(); // v208[k8];	// L258
    int8_t a8;	// L259
    a8 = v217;	// L260
    int8_t v219 = v209.read(); // v209[k8];	// L261
    int8_t b8;	// L262
    b8 = v219;	// L263
    int8_t v221 = a8;	// L264
    int8_t v222 = b8;	// L265
    int16_t v223 = v221;	// L266
    int16_t v224 = v222;	// L267
    int16_t v225 = v223 * v224;	// L268
    int32_t v226 = v8;	// L269
    ap_int<33> v227 = v226;	// L270
    ap_int<33> v228 = v225;	// L271
    ap_int<33> v229 = v227 + v228;	// L272
    int32_t v230 = v229;	// L273
    v8 = v230;	// L274
    int8_t v231 = a8;	// L275
    v210.write(v231); // v210[k8] = v231;	// L276
    int8_t v232 = b8;	// L277
    v211.write(v232); // v211[k8] = v232;	// L278
  }
  int32_t v233 = v8;	// L280
  v212[v213][v214] = v233;	// L281
}

void PE_kernel_gemm_1_1(
  hls::stream< int8_t > &v234 /* v234[8] */,
  hls::stream< int8_t > &v235 /* v235[8] */,
  hls::stream< int8_t > &v236 /* v236[8] */,
  hls::stream< int8_t > &v237 /* v237[8] */,
  int32_t v238[8][8],
  int v239,
  int v240
) {	// L284
  #pragma HLS stream variable=v234 depth=9
  #pragma HLS stream variable=v235 depth=9
  #pragma HLS stream variable=v236 depth=9
  #pragma HLS stream variable=v237 depth=9
  #pragma HLS array_partition variable=v238 complete dim=1
  #pragma HLS array_partition variable=v238 complete dim=2

  int32_t v9;	// L286
  v9 = 0;	// L287
  l_reduction_k9: for (int k9 = 0; k9 < 8; k9++) {	// L288
  #pragma HLS pipeline II=1
    int8_t v243 = v234.read(); // v234[k9];	// L289
    int8_t a9;	// L290
    a9 = v243;	// L291
    int8_t v245 = v235.read(); // v235[k9];	// L292
    int8_t b9;	// L293
    b9 = v245;	// L294
    int8_t v247 = a9;	// L295
    int8_t v248 = b9;	// L296
    int16_t v249 = v247;	// L297
    int16_t v250 = v248;	// L298
    int16_t v251 = v249 * v250;	// L299
    int32_t v252 = v9;	// L300
    ap_int<33> v253 = v252;	// L301
    ap_int<33> v254 = v251;	// L302
    ap_int<33> v255 = v253 + v254;	// L303
    int32_t v256 = v255;	// L304
    v9 = v256;	// L305
    int8_t v257 = a9;	// L306
    v236.write(v257); // v236[k9] = v257;	// L307
    int8_t v258 = b9;	// L308
    v237.write(v258); // v237[k9] = v258;	// L309
  }
  int32_t v259 = v9;	// L311
  v238[v239][v240] = v259;	// L312
}

void PE_kernel_gemm_2_1(
  hls::stream< int8_t > &v260 /* v260[8] */,
  hls::stream< int8_t > &v261 /* v261[8] */,
  hls::stream< int8_t > &v262 /* v262[8] */,
  hls::stream< int8_t > &v263 /* v263[8] */,
  int32_t v264[8][8],
  int v265,
  int v266
) {	// L315
  #pragma HLS stream variable=v260 depth=9
  #pragma HLS stream variable=v261 depth=9
  #pragma HLS stream variable=v262 depth=9
  #pragma HLS stream variable=v263 depth=9
  #pragma HLS array_partition variable=v264 complete dim=1
  #pragma HLS array_partition variable=v264 complete dim=2

  int32_t v10;	// L317
  v10 = 0;	// L318
  l_reduction_k10: for (int k10 = 0; k10 < 8; k10++) {	// L319
  #pragma HLS pipeline II=1
    int8_t v269 = v260.read(); // v260[k10];	// L320
    int8_t a10;	// L321
    a10 = v269;	// L322
    int8_t v271 = v261.read(); // v261[k10];	// L323
    int8_t b10;	// L324
    b10 = v271;	// L325
    int8_t v273 = a10;	// L326
    int8_t v274 = b10;	// L327
    int16_t v275 = v273;	// L328
    int16_t v276 = v274;	// L329
    int16_t v277 = v275 * v276;	// L330
    int32_t v278 = v10;	// L331
    ap_int<33> v279 = v278;	// L332
    ap_int<33> v280 = v277;	// L333
    ap_int<33> v281 = v279 + v280;	// L334
    int32_t v282 = v281;	// L335
    v10 = v282;	// L336
    int8_t v283 = a10;	// L337
    v262.write(v283); // v262[k10] = v283;	// L338
    int8_t v284 = b10;	// L339
    v263.write(v284); // v263[k10] = v284;	// L340
  }
  int32_t v285 = v10;	// L342
  v264[v265][v266] = v285;	// L343
}

void PE_kernel_gemm_3_1(
  hls::stream< int8_t > &v286 /* v286[8] */,
  hls::stream< int8_t > &v287 /* v287[8] */,
  hls::stream< int8_t > &v288 /* v288[8] */,
  hls::stream< int8_t > &v289 /* v289[8] */,
  int32_t v290[8][8],
  int v291,
  int v292
) {	// L346
  #pragma HLS stream variable=v286 depth=9
  #pragma HLS stream variable=v287 depth=9
  #pragma HLS stream variable=v288 depth=9
  #pragma HLS stream variable=v289 depth=9
  #pragma HLS array_partition variable=v290 complete dim=1
  #pragma HLS array_partition variable=v290 complete dim=2

  int32_t v11;	// L348
  v11 = 0;	// L349
  l_reduction_k11: for (int k11 = 0; k11 < 8; k11++) {	// L350
  #pragma HLS pipeline II=1
    int8_t v295 = v286.read(); // v286[k11];	// L351
    int8_t a11;	// L352
    a11 = v295;	// L353
    int8_t v297 = v287.read(); // v287[k11];	// L354
    int8_t b11;	// L355
    b11 = v297;	// L356
    int8_t v299 = a11;	// L357
    int8_t v300 = b11;	// L358
    int16_t v301 = v299;	// L359
    int16_t v302 = v300;	// L360
    int16_t v303 = v301 * v302;	// L361
    int32_t v304 = v11;	// L362
    ap_int<33> v305 = v304;	// L363
    ap_int<33> v306 = v303;	// L364
    ap_int<33> v307 = v305 + v306;	// L365
    int32_t v308 = v307;	// L366
    v11 = v308;	// L367
    int8_t v309 = a11;	// L368
    v288.write(v309); // v288[k11] = v309;	// L369
    int8_t v310 = b11;	// L370
    v289.write(v310); // v289[k11] = v310;	// L371
  }
  int32_t v311 = v11;	// L373
  v290[v291][v292] = v311;	// L374
}

void PE_kernel_gemm_4_1(
  hls::stream< int8_t > &v312 /* v312[8] */,
  hls::stream< int8_t > &v313 /* v313[8] */,
  hls::stream< int8_t > &v314 /* v314[8] */,
  hls::stream< int8_t > &v315 /* v315[8] */,
  int32_t v316[8][8],
  int v317,
  int v318
) {	// L377
  #pragma HLS stream variable=v312 depth=9
  #pragma HLS stream variable=v313 depth=9
  #pragma HLS stream variable=v314 depth=9
  #pragma HLS stream variable=v315 depth=9
  #pragma HLS array_partition variable=v316 complete dim=1
  #pragma HLS array_partition variable=v316 complete dim=2

  int32_t v12;	// L379
  v12 = 0;	// L380
  l_reduction_k12: for (int k12 = 0; k12 < 8; k12++) {	// L381
  #pragma HLS pipeline II=1
    int8_t v321 = v312.read(); // v312[k12];	// L382
    int8_t a12;	// L383
    a12 = v321;	// L384
    int8_t v323 = v313.read(); // v313[k12];	// L385
    int8_t b12;	// L386
    b12 = v323;	// L387
    int8_t v325 = a12;	// L388
    int8_t v326 = b12;	// L389
    int16_t v327 = v325;	// L390
    int16_t v328 = v326;	// L391
    int16_t v329 = v327 * v328;	// L392
    int32_t v330 = v12;	// L393
    ap_int<33> v331 = v330;	// L394
    ap_int<33> v332 = v329;	// L395
    ap_int<33> v333 = v331 + v332;	// L396
    int32_t v334 = v333;	// L397
    v12 = v334;	// L398
    int8_t v335 = a12;	// L399
    v314.write(v335); // v314[k12] = v335;	// L400
    int8_t v336 = b12;	// L401
    v315.write(v336); // v315[k12] = v336;	// L402
  }
  int32_t v337 = v12;	// L404
  v316[v317][v318] = v337;	// L405
}

void PE_kernel_gemm_5_1(
  hls::stream< int8_t > &v338 /* v338[8] */,
  hls::stream< int8_t > &v339 /* v339[8] */,
  hls::stream< int8_t > &v340 /* v340[8] */,
  hls::stream< int8_t > &v341 /* v341[8] */,
  int32_t v342[8][8],
  int v343,
  int v344
) {	// L408
  #pragma HLS stream variable=v338 depth=9
  #pragma HLS stream variable=v339 depth=9
  #pragma HLS stream variable=v340 depth=9
  #pragma HLS stream variable=v341 depth=9
  #pragma HLS array_partition variable=v342 complete dim=1
  #pragma HLS array_partition variable=v342 complete dim=2

  int32_t v13;	// L410
  v13 = 0;	// L411
  l_reduction_k13: for (int k13 = 0; k13 < 8; k13++) {	// L412
  #pragma HLS pipeline II=1
    int8_t v347 = v338.read(); // v338[k13];	// L413
    int8_t a13;	// L414
    a13 = v347;	// L415
    int8_t v349 = v339.read(); // v339[k13];	// L416
    int8_t b13;	// L417
    b13 = v349;	// L418
    int8_t v351 = a13;	// L419
    int8_t v352 = b13;	// L420
    int16_t v353 = v351;	// L421
    int16_t v354 = v352;	// L422
    int16_t v355 = v353 * v354;	// L423
    int32_t v356 = v13;	// L424
    ap_int<33> v357 = v356;	// L425
    ap_int<33> v358 = v355;	// L426
    ap_int<33> v359 = v357 + v358;	// L427
    int32_t v360 = v359;	// L428
    v13 = v360;	// L429
    int8_t v361 = a13;	// L430
    v340.write(v361); // v340[k13] = v361;	// L431
    int8_t v362 = b13;	// L432
    v341.write(v362); // v341[k13] = v362;	// L433
  }
  int32_t v363 = v13;	// L435
  v342[v343][v344] = v363;	// L436
}

void PE_kernel_gemm_6_1(
  hls::stream< int8_t > &v364 /* v364[8] */,
  hls::stream< int8_t > &v365 /* v365[8] */,
  hls::stream< int8_t > &v366 /* v366[8] */,
  hls::stream< int8_t > &v367 /* v367[8] */,
  int32_t v368[8][8],
  int v369,
  int v370
) {	// L439
  #pragma HLS stream variable=v364 depth=9
  #pragma HLS stream variable=v365 depth=9
  #pragma HLS stream variable=v366 depth=9
  #pragma HLS stream variable=v367 depth=9
  #pragma HLS array_partition variable=v368 complete dim=1
  #pragma HLS array_partition variable=v368 complete dim=2

  int32_t v14;	// L441
  v14 = 0;	// L442
  l_reduction_k14: for (int k14 = 0; k14 < 8; k14++) {	// L443
  #pragma HLS pipeline II=1
    int8_t v373 = v364.read(); // v364[k14];	// L444
    int8_t a14;	// L445
    a14 = v373;	// L446
    int8_t v375 = v365.read(); // v365[k14];	// L447
    int8_t b14;	// L448
    b14 = v375;	// L449
    int8_t v377 = a14;	// L450
    int8_t v378 = b14;	// L451
    int16_t v379 = v377;	// L452
    int16_t v380 = v378;	// L453
    int16_t v381 = v379 * v380;	// L454
    int32_t v382 = v14;	// L455
    ap_int<33> v383 = v382;	// L456
    ap_int<33> v384 = v381;	// L457
    ap_int<33> v385 = v383 + v384;	// L458
    int32_t v386 = v385;	// L459
    v14 = v386;	// L460
    int8_t v387 = a14;	// L461
    v366.write(v387); // v366[k14] = v387;	// L462
    int8_t v388 = b14;	// L463
    v367.write(v388); // v367[k14] = v388;	// L464
  }
  int32_t v389 = v14;	// L466
  v368[v369][v370] = v389;	// L467
}

void PE_kernel_gemm_7_1(
  hls::stream< int8_t > &v390 /* v390[8] */,
  hls::stream< int8_t > &v391 /* v391[8] */,
  hls::stream< int8_t > &v392 /* v392[8] */,
  hls::stream< int8_t > &v393 /* v393[8] */,
  int32_t v394[8][8],
  int v395,
  int v396
) {	// L470
  #pragma HLS stream variable=v390 depth=9
  #pragma HLS stream variable=v391 depth=9
  #pragma HLS stream variable=v392 depth=9
  #pragma HLS stream variable=v393 depth=9
  #pragma HLS array_partition variable=v394 complete dim=1
  #pragma HLS array_partition variable=v394 complete dim=2

  int32_t v15;	// L472
  v15 = 0;	// L473
  l_reduction_k15: for (int k15 = 0; k15 < 8; k15++) {	// L474
  #pragma HLS pipeline II=1
    int8_t v399 = v390.read(); // v390[k15];	// L475
    int8_t a15;	// L476
    a15 = v399;	// L477
    int8_t v401 = v391.read(); // v391[k15];	// L478
    int8_t b15;	// L479
    b15 = v401;	// L480
    int8_t v403 = a15;	// L481
    int8_t v404 = b15;	// L482
    int16_t v405 = v403;	// L483
    int16_t v406 = v404;	// L484
    int16_t v407 = v405 * v406;	// L485
    int32_t v408 = v15;	// L486
    ap_int<33> v409 = v408;	// L487
    ap_int<33> v410 = v407;	// L488
    ap_int<33> v411 = v409 + v410;	// L489
    int32_t v412 = v411;	// L490
    v15 = v412;	// L491
    int8_t v413 = a15;	// L492
    v392.write(v413); // v392[k15] = v413;	// L493
    int8_t v414 = b15;	// L494
    v393.write(v414); // v393[k15] = v414;	// L495
  }
  int32_t v415 = v15;	// L497
  v394[v395][v396] = v415;	// L498
}

void PE_kernel_gemm_0_2(
  hls::stream< int8_t > &v416 /* v416[8] */,
  hls::stream< int8_t > &v417 /* v417[8] */,
  hls::stream< int8_t > &v418 /* v418[8] */,
  hls::stream< int8_t > &v419 /* v419[8] */,
  int32_t v420[8][8],
  int v421,
  int v422
) {	// L501
  #pragma HLS stream variable=v416 depth=9
  #pragma HLS stream variable=v417 depth=9
  #pragma HLS stream variable=v418 depth=9
  #pragma HLS stream variable=v419 depth=9
  #pragma HLS array_partition variable=v420 complete dim=1
  #pragma HLS array_partition variable=v420 complete dim=2

  int32_t v16;	// L503
  v16 = 0;	// L504
  l_reduction_k16: for (int k16 = 0; k16 < 8; k16++) {	// L505
  #pragma HLS pipeline II=1
    int8_t v425 = v416.read(); // v416[k16];	// L506
    int8_t a16;	// L507
    a16 = v425;	// L508
    int8_t v427 = v417.read(); // v417[k16];	// L509
    int8_t b16;	// L510
    b16 = v427;	// L511
    int8_t v429 = a16;	// L512
    int8_t v430 = b16;	// L513
    int16_t v431 = v429;	// L514
    int16_t v432 = v430;	// L515
    int16_t v433 = v431 * v432;	// L516
    int32_t v434 = v16;	// L517
    ap_int<33> v435 = v434;	// L518
    ap_int<33> v436 = v433;	// L519
    ap_int<33> v437 = v435 + v436;	// L520
    int32_t v438 = v437;	// L521
    v16 = v438;	// L522
    int8_t v439 = a16;	// L523
    v418.write(v439); // v418[k16] = v439;	// L524
    int8_t v440 = b16;	// L525
    v419.write(v440); // v419[k16] = v440;	// L526
  }
  int32_t v441 = v16;	// L528
  v420[v421][v422] = v441;	// L529
}

void PE_kernel_gemm_1_2(
  hls::stream< int8_t > &v442 /* v442[8] */,
  hls::stream< int8_t > &v443 /* v443[8] */,
  hls::stream< int8_t > &v444 /* v444[8] */,
  hls::stream< int8_t > &v445 /* v445[8] */,
  int32_t v446[8][8],
  int v447,
  int v448
) {	// L532
  #pragma HLS stream variable=v442 depth=9
  #pragma HLS stream variable=v443 depth=9
  #pragma HLS stream variable=v444 depth=9
  #pragma HLS stream variable=v445 depth=9
  #pragma HLS array_partition variable=v446 complete dim=1
  #pragma HLS array_partition variable=v446 complete dim=2

  int32_t v17;	// L534
  v17 = 0;	// L535
  l_reduction_k17: for (int k17 = 0; k17 < 8; k17++) {	// L536
  #pragma HLS pipeline II=1
    int8_t v451 = v442.read(); // v442[k17];	// L537
    int8_t a17;	// L538
    a17 = v451;	// L539
    int8_t v453 = v443.read(); // v443[k17];	// L540
    int8_t b17;	// L541
    b17 = v453;	// L542
    int8_t v455 = a17;	// L543
    int8_t v456 = b17;	// L544
    int16_t v457 = v455;	// L545
    int16_t v458 = v456;	// L546
    int16_t v459 = v457 * v458;	// L547
    int32_t v460 = v17;	// L548
    ap_int<33> v461 = v460;	// L549
    ap_int<33> v462 = v459;	// L550
    ap_int<33> v463 = v461 + v462;	// L551
    int32_t v464 = v463;	// L552
    v17 = v464;	// L553
    int8_t v465 = a17;	// L554
    v444.write(v465); // v444[k17] = v465;	// L555
    int8_t v466 = b17;	// L556
    v445.write(v466); // v445[k17] = v466;	// L557
  }
  int32_t v467 = v17;	// L559
  v446[v447][v448] = v467;	// L560
}

void PE_kernel_gemm_2_2(
  hls::stream< int8_t > &v468 /* v468[8] */,
  hls::stream< int8_t > &v469 /* v469[8] */,
  hls::stream< int8_t > &v470 /* v470[8] */,
  hls::stream< int8_t > &v471 /* v471[8] */,
  int32_t v472[8][8],
  int v473,
  int v474
) {	// L563
  #pragma HLS stream variable=v468 depth=9
  #pragma HLS stream variable=v469 depth=9
  #pragma HLS stream variable=v470 depth=9
  #pragma HLS stream variable=v471 depth=9
  #pragma HLS array_partition variable=v472 complete dim=1
  #pragma HLS array_partition variable=v472 complete dim=2

  int32_t v18;	// L565
  v18 = 0;	// L566
  l_reduction_k18: for (int k18 = 0; k18 < 8; k18++) {	// L567
  #pragma HLS pipeline II=1
    int8_t v477 = v468.read(); // v468[k18];	// L568
    int8_t a18;	// L569
    a18 = v477;	// L570
    int8_t v479 = v469.read(); // v469[k18];	// L571
    int8_t b18;	// L572
    b18 = v479;	// L573
    int8_t v481 = a18;	// L574
    int8_t v482 = b18;	// L575
    int16_t v483 = v481;	// L576
    int16_t v484 = v482;	// L577
    int16_t v485 = v483 * v484;	// L578
    int32_t v486 = v18;	// L579
    ap_int<33> v487 = v486;	// L580
    ap_int<33> v488 = v485;	// L581
    ap_int<33> v489 = v487 + v488;	// L582
    int32_t v490 = v489;	// L583
    v18 = v490;	// L584
    int8_t v491 = a18;	// L585
    v470.write(v491); // v470[k18] = v491;	// L586
    int8_t v492 = b18;	// L587
    v471.write(v492); // v471[k18] = v492;	// L588
  }
  int32_t v493 = v18;	// L590
  v472[v473][v474] = v493;	// L591
}

void PE_kernel_gemm_3_2(
  hls::stream< int8_t > &v494 /* v494[8] */,
  hls::stream< int8_t > &v495 /* v495[8] */,
  hls::stream< int8_t > &v496 /* v496[8] */,
  hls::stream< int8_t > &v497 /* v497[8] */,
  int32_t v498[8][8],
  int v499,
  int v500
) {	// L594
  #pragma HLS stream variable=v494 depth=9
  #pragma HLS stream variable=v495 depth=9
  #pragma HLS stream variable=v496 depth=9
  #pragma HLS stream variable=v497 depth=9
  #pragma HLS array_partition variable=v498 complete dim=1
  #pragma HLS array_partition variable=v498 complete dim=2

  int32_t v19;	// L596
  v19 = 0;	// L597
  l_reduction_k19: for (int k19 = 0; k19 < 8; k19++) {	// L598
  #pragma HLS pipeline II=1
    int8_t v503 = v494.read(); // v494[k19];	// L599
    int8_t a19;	// L600
    a19 = v503;	// L601
    int8_t v505 = v495.read(); // v495[k19];	// L602
    int8_t b19;	// L603
    b19 = v505;	// L604
    int8_t v507 = a19;	// L605
    int8_t v508 = b19;	// L606
    int16_t v509 = v507;	// L607
    int16_t v510 = v508;	// L608
    int16_t v511 = v509 * v510;	// L609
    int32_t v512 = v19;	// L610
    ap_int<33> v513 = v512;	// L611
    ap_int<33> v514 = v511;	// L612
    ap_int<33> v515 = v513 + v514;	// L613
    int32_t v516 = v515;	// L614
    v19 = v516;	// L615
    int8_t v517 = a19;	// L616
    v496.write(v517); // v496[k19] = v517;	// L617
    int8_t v518 = b19;	// L618
    v497.write(v518); // v497[k19] = v518;	// L619
  }
  int32_t v519 = v19;	// L621
  v498[v499][v500] = v519;	// L622
}

void PE_kernel_gemm_4_2(
  hls::stream< int8_t > &v520 /* v520[8] */,
  hls::stream< int8_t > &v521 /* v521[8] */,
  hls::stream< int8_t > &v522 /* v522[8] */,
  hls::stream< int8_t > &v523 /* v523[8] */,
  int32_t v524[8][8],
  int v525,
  int v526
) {	// L625
  #pragma HLS stream variable=v520 depth=9
  #pragma HLS stream variable=v521 depth=9
  #pragma HLS stream variable=v522 depth=9
  #pragma HLS stream variable=v523 depth=9
  #pragma HLS array_partition variable=v524 complete dim=1
  #pragma HLS array_partition variable=v524 complete dim=2

  int32_t v20;	// L627
  v20 = 0;	// L628
  l_reduction_k20: for (int k20 = 0; k20 < 8; k20++) {	// L629
  #pragma HLS pipeline II=1
    int8_t v529 = v520.read(); // v520[k20];	// L630
    int8_t a20;	// L631
    a20 = v529;	// L632
    int8_t v531 = v521.read(); // v521[k20];	// L633
    int8_t b20;	// L634
    b20 = v531;	// L635
    int8_t v533 = a20;	// L636
    int8_t v534 = b20;	// L637
    int16_t v535 = v533;	// L638
    int16_t v536 = v534;	// L639
    int16_t v537 = v535 * v536;	// L640
    int32_t v538 = v20;	// L641
    ap_int<33> v539 = v538;	// L642
    ap_int<33> v540 = v537;	// L643
    ap_int<33> v541 = v539 + v540;	// L644
    int32_t v542 = v541;	// L645
    v20 = v542;	// L646
    int8_t v543 = a20;	// L647
    v522.write(v543); // v522[k20] = v543;	// L648
    int8_t v544 = b20;	// L649
    v523.write(v544); // v523[k20] = v544;	// L650
  }
  int32_t v545 = v20;	// L652
  v524[v525][v526] = v545;	// L653
}

void PE_kernel_gemm_5_2(
  hls::stream< int8_t > &v546 /* v546[8] */,
  hls::stream< int8_t > &v547 /* v547[8] */,
  hls::stream< int8_t > &v548 /* v548[8] */,
  hls::stream< int8_t > &v549 /* v549[8] */,
  int32_t v550[8][8],
  int v551,
  int v552
) {	// L656
  #pragma HLS stream variable=v546 depth=9
  #pragma HLS stream variable=v547 depth=9
  #pragma HLS stream variable=v548 depth=9
  #pragma HLS stream variable=v549 depth=9
  #pragma HLS array_partition variable=v550 complete dim=1
  #pragma HLS array_partition variable=v550 complete dim=2

  int32_t v21;	// L658
  v21 = 0;	// L659
  l_reduction_k21: for (int k21 = 0; k21 < 8; k21++) {	// L660
  #pragma HLS pipeline II=1
    int8_t v555 = v546.read(); // v546[k21];	// L661
    int8_t a21;	// L662
    a21 = v555;	// L663
    int8_t v557 = v547.read(); // v547[k21];	// L664
    int8_t b21;	// L665
    b21 = v557;	// L666
    int8_t v559 = a21;	// L667
    int8_t v560 = b21;	// L668
    int16_t v561 = v559;	// L669
    int16_t v562 = v560;	// L670
    int16_t v563 = v561 * v562;	// L671
    int32_t v564 = v21;	// L672
    ap_int<33> v565 = v564;	// L673
    ap_int<33> v566 = v563;	// L674
    ap_int<33> v567 = v565 + v566;	// L675
    int32_t v568 = v567;	// L676
    v21 = v568;	// L677
    int8_t v569 = a21;	// L678
    v548.write(v569); // v548[k21] = v569;	// L679
    int8_t v570 = b21;	// L680
    v549.write(v570); // v549[k21] = v570;	// L681
  }
  int32_t v571 = v21;	// L683
  v550[v551][v552] = v571;	// L684
}

void PE_kernel_gemm_6_2(
  hls::stream< int8_t > &v572 /* v572[8] */,
  hls::stream< int8_t > &v573 /* v573[8] */,
  hls::stream< int8_t > &v574 /* v574[8] */,
  hls::stream< int8_t > &v575 /* v575[8] */,
  int32_t v576[8][8],
  int v577,
  int v578
) {	// L687
  #pragma HLS stream variable=v572 depth=9
  #pragma HLS stream variable=v573 depth=9
  #pragma HLS stream variable=v574 depth=9
  #pragma HLS stream variable=v575 depth=9
  #pragma HLS array_partition variable=v576 complete dim=1
  #pragma HLS array_partition variable=v576 complete dim=2

  int32_t v22;	// L689
  v22 = 0;	// L690
  l_reduction_k22: for (int k22 = 0; k22 < 8; k22++) {	// L691
  #pragma HLS pipeline II=1
    int8_t v581 = v572.read(); // v572[k22];	// L692
    int8_t a22;	// L693
    a22 = v581;	// L694
    int8_t v583 = v573.read(); // v573[k22];	// L695
    int8_t b22;	// L696
    b22 = v583;	// L697
    int8_t v585 = a22;	// L698
    int8_t v586 = b22;	// L699
    int16_t v587 = v585;	// L700
    int16_t v588 = v586;	// L701
    int16_t v589 = v587 * v588;	// L702
    int32_t v590 = v22;	// L703
    ap_int<33> v591 = v590;	// L704
    ap_int<33> v592 = v589;	// L705
    ap_int<33> v593 = v591 + v592;	// L706
    int32_t v594 = v593;	// L707
    v22 = v594;	// L708
    int8_t v595 = a22;	// L709
    v574.write(v595); // v574[k22] = v595;	// L710
    int8_t v596 = b22;	// L711
    v575.write(v596); // v575[k22] = v596;	// L712
  }
  int32_t v597 = v22;	// L714
  v576[v577][v578] = v597;	// L715
}

void PE_kernel_gemm_7_2(
  hls::stream< int8_t > &v598 /* v598[8] */,
  hls::stream< int8_t > &v599 /* v599[8] */,
  hls::stream< int8_t > &v600 /* v600[8] */,
  hls::stream< int8_t > &v601 /* v601[8] */,
  int32_t v602[8][8],
  int v603,
  int v604
) {	// L718
  #pragma HLS stream variable=v598 depth=9
  #pragma HLS stream variable=v599 depth=9
  #pragma HLS stream variable=v600 depth=9
  #pragma HLS stream variable=v601 depth=9
  #pragma HLS array_partition variable=v602 complete dim=1
  #pragma HLS array_partition variable=v602 complete dim=2

  int32_t v23;	// L720
  v23 = 0;	// L721
  l_reduction_k23: for (int k23 = 0; k23 < 8; k23++) {	// L722
  #pragma HLS pipeline II=1
    int8_t v607 = v598.read(); // v598[k23];	// L723
    int8_t a23;	// L724
    a23 = v607;	// L725
    int8_t v609 = v599.read(); // v599[k23];	// L726
    int8_t b23;	// L727
    b23 = v609;	// L728
    int8_t v611 = a23;	// L729
    int8_t v612 = b23;	// L730
    int16_t v613 = v611;	// L731
    int16_t v614 = v612;	// L732
    int16_t v615 = v613 * v614;	// L733
    int32_t v616 = v23;	// L734
    ap_int<33> v617 = v616;	// L735
    ap_int<33> v618 = v615;	// L736
    ap_int<33> v619 = v617 + v618;	// L737
    int32_t v620 = v619;	// L738
    v23 = v620;	// L739
    int8_t v621 = a23;	// L740
    v600.write(v621); // v600[k23] = v621;	// L741
    int8_t v622 = b23;	// L742
    v601.write(v622); // v601[k23] = v622;	// L743
  }
  int32_t v623 = v23;	// L745
  v602[v603][v604] = v623;	// L746
}

void PE_kernel_gemm_0_3(
  hls::stream< int8_t > &v624 /* v624[8] */,
  hls::stream< int8_t > &v625 /* v625[8] */,
  hls::stream< int8_t > &v626 /* v626[8] */,
  hls::stream< int8_t > &v627 /* v627[8] */,
  int32_t v628[8][8],
  int v629,
  int v630
) {	// L749
  #pragma HLS stream variable=v624 depth=9
  #pragma HLS stream variable=v625 depth=9
  #pragma HLS stream variable=v626 depth=9
  #pragma HLS stream variable=v627 depth=9
  #pragma HLS array_partition variable=v628 complete dim=1
  #pragma HLS array_partition variable=v628 complete dim=2

  int32_t v24;	// L751
  v24 = 0;	// L752
  l_reduction_k24: for (int k24 = 0; k24 < 8; k24++) {	// L753
  #pragma HLS pipeline II=1
    int8_t v633 = v624.read(); // v624[k24];	// L754
    int8_t a24;	// L755
    a24 = v633;	// L756
    int8_t v635 = v625.read(); // v625[k24];	// L757
    int8_t b24;	// L758
    b24 = v635;	// L759
    int8_t v637 = a24;	// L760
    int8_t v638 = b24;	// L761
    int16_t v639 = v637;	// L762
    int16_t v640 = v638;	// L763
    int16_t v641 = v639 * v640;	// L764
    int32_t v642 = v24;	// L765
    ap_int<33> v643 = v642;	// L766
    ap_int<33> v644 = v641;	// L767
    ap_int<33> v645 = v643 + v644;	// L768
    int32_t v646 = v645;	// L769
    v24 = v646;	// L770
    int8_t v647 = a24;	// L771
    v626.write(v647); // v626[k24] = v647;	// L772
    int8_t v648 = b24;	// L773
    v627.write(v648); // v627[k24] = v648;	// L774
  }
  int32_t v649 = v24;	// L776
  v628[v629][v630] = v649;	// L777
}

void PE_kernel_gemm_1_3(
  hls::stream< int8_t > &v650 /* v650[8] */,
  hls::stream< int8_t > &v651 /* v651[8] */,
  hls::stream< int8_t > &v652 /* v652[8] */,
  hls::stream< int8_t > &v653 /* v653[8] */,
  int32_t v654[8][8],
  int v655,
  int v656
) {	// L780
  #pragma HLS stream variable=v650 depth=9
  #pragma HLS stream variable=v651 depth=9
  #pragma HLS stream variable=v652 depth=9
  #pragma HLS stream variable=v653 depth=9
  #pragma HLS array_partition variable=v654 complete dim=1
  #pragma HLS array_partition variable=v654 complete dim=2

  int32_t v25;	// L782
  v25 = 0;	// L783
  l_reduction_k25: for (int k25 = 0; k25 < 8; k25++) {	// L784
  #pragma HLS pipeline II=1
    int8_t v659 = v650.read(); // v650[k25];	// L785
    int8_t a25;	// L786
    a25 = v659;	// L787
    int8_t v661 = v651.read(); // v651[k25];	// L788
    int8_t b25;	// L789
    b25 = v661;	// L790
    int8_t v663 = a25;	// L791
    int8_t v664 = b25;	// L792
    int16_t v665 = v663;	// L793
    int16_t v666 = v664;	// L794
    int16_t v667 = v665 * v666;	// L795
    int32_t v668 = v25;	// L796
    ap_int<33> v669 = v668;	// L797
    ap_int<33> v670 = v667;	// L798
    ap_int<33> v671 = v669 + v670;	// L799
    int32_t v672 = v671;	// L800
    v25 = v672;	// L801
    int8_t v673 = a25;	// L802
    v652.write(v673); // v652[k25] = v673;	// L803
    int8_t v674 = b25;	// L804
    v653.write(v674); // v653[k25] = v674;	// L805
  }
  int32_t v675 = v25;	// L807
  v654[v655][v656] = v675;	// L808
}

void PE_kernel_gemm_2_3(
  hls::stream< int8_t > &v676 /* v676[8] */,
  hls::stream< int8_t > &v677 /* v677[8] */,
  hls::stream< int8_t > &v678 /* v678[8] */,
  hls::stream< int8_t > &v679 /* v679[8] */,
  int32_t v680[8][8],
  int v681,
  int v682
) {	// L811
  #pragma HLS stream variable=v676 depth=9
  #pragma HLS stream variable=v677 depth=9
  #pragma HLS stream variable=v678 depth=9
  #pragma HLS stream variable=v679 depth=9
  #pragma HLS array_partition variable=v680 complete dim=1
  #pragma HLS array_partition variable=v680 complete dim=2

  int32_t v26;	// L813
  v26 = 0;	// L814
  l_reduction_k26: for (int k26 = 0; k26 < 8; k26++) {	// L815
  #pragma HLS pipeline II=1
    int8_t v685 = v676.read(); // v676[k26];	// L816
    int8_t a26;	// L817
    a26 = v685;	// L818
    int8_t v687 = v677.read(); // v677[k26];	// L819
    int8_t b26;	// L820
    b26 = v687;	// L821
    int8_t v689 = a26;	// L822
    int8_t v690 = b26;	// L823
    int16_t v691 = v689;	// L824
    int16_t v692 = v690;	// L825
    int16_t v693 = v691 * v692;	// L826
    int32_t v694 = v26;	// L827
    ap_int<33> v695 = v694;	// L828
    ap_int<33> v696 = v693;	// L829
    ap_int<33> v697 = v695 + v696;	// L830
    int32_t v698 = v697;	// L831
    v26 = v698;	// L832
    int8_t v699 = a26;	// L833
    v678.write(v699); // v678[k26] = v699;	// L834
    int8_t v700 = b26;	// L835
    v679.write(v700); // v679[k26] = v700;	// L836
  }
  int32_t v701 = v26;	// L838
  v680[v681][v682] = v701;	// L839
}

void PE_kernel_gemm_3_3(
  hls::stream< int8_t > &v702 /* v702[8] */,
  hls::stream< int8_t > &v703 /* v703[8] */,
  hls::stream< int8_t > &v704 /* v704[8] */,
  hls::stream< int8_t > &v705 /* v705[8] */,
  int32_t v706[8][8],
  int v707,
  int v708
) {	// L842
  #pragma HLS stream variable=v702 depth=9
  #pragma HLS stream variable=v703 depth=9
  #pragma HLS stream variable=v704 depth=9
  #pragma HLS stream variable=v705 depth=9
  #pragma HLS array_partition variable=v706 complete dim=1
  #pragma HLS array_partition variable=v706 complete dim=2

  int32_t v27;	// L844
  v27 = 0;	// L845
  l_reduction_k27: for (int k27 = 0; k27 < 8; k27++) {	// L846
  #pragma HLS pipeline II=1
    int8_t v711 = v702.read(); // v702[k27];	// L847
    int8_t a27;	// L848
    a27 = v711;	// L849
    int8_t v713 = v703.read(); // v703[k27];	// L850
    int8_t b27;	// L851
    b27 = v713;	// L852
    int8_t v715 = a27;	// L853
    int8_t v716 = b27;	// L854
    int16_t v717 = v715;	// L855
    int16_t v718 = v716;	// L856
    int16_t v719 = v717 * v718;	// L857
    int32_t v720 = v27;	// L858
    ap_int<33> v721 = v720;	// L859
    ap_int<33> v722 = v719;	// L860
    ap_int<33> v723 = v721 + v722;	// L861
    int32_t v724 = v723;	// L862
    v27 = v724;	// L863
    int8_t v725 = a27;	// L864
    v704.write(v725); // v704[k27] = v725;	// L865
    int8_t v726 = b27;	// L866
    v705.write(v726); // v705[k27] = v726;	// L867
  }
  int32_t v727 = v27;	// L869
  v706[v707][v708] = v727;	// L870
}

void PE_kernel_gemm_4_3(
  hls::stream< int8_t > &v728 /* v728[8] */,
  hls::stream< int8_t > &v729 /* v729[8] */,
  hls::stream< int8_t > &v730 /* v730[8] */,
  hls::stream< int8_t > &v731 /* v731[8] */,
  int32_t v732[8][8],
  int v733,
  int v734
) {	// L873
  #pragma HLS stream variable=v728 depth=9
  #pragma HLS stream variable=v729 depth=9
  #pragma HLS stream variable=v730 depth=9
  #pragma HLS stream variable=v731 depth=9
  #pragma HLS array_partition variable=v732 complete dim=1
  #pragma HLS array_partition variable=v732 complete dim=2

  int32_t v28;	// L875
  v28 = 0;	// L876
  l_reduction_k28: for (int k28 = 0; k28 < 8; k28++) {	// L877
  #pragma HLS pipeline II=1
    int8_t v737 = v728.read(); // v728[k28];	// L878
    int8_t a28;	// L879
    a28 = v737;	// L880
    int8_t v739 = v729.read(); // v729[k28];	// L881
    int8_t b28;	// L882
    b28 = v739;	// L883
    int8_t v741 = a28;	// L884
    int8_t v742 = b28;	// L885
    int16_t v743 = v741;	// L886
    int16_t v744 = v742;	// L887
    int16_t v745 = v743 * v744;	// L888
    int32_t v746 = v28;	// L889
    ap_int<33> v747 = v746;	// L890
    ap_int<33> v748 = v745;	// L891
    ap_int<33> v749 = v747 + v748;	// L892
    int32_t v750 = v749;	// L893
    v28 = v750;	// L894
    int8_t v751 = a28;	// L895
    v730.write(v751); // v730[k28] = v751;	// L896
    int8_t v752 = b28;	// L897
    v731.write(v752); // v731[k28] = v752;	// L898
  }
  int32_t v753 = v28;	// L900
  v732[v733][v734] = v753;	// L901
}

void PE_kernel_gemm_5_3(
  hls::stream< int8_t > &v754 /* v754[8] */,
  hls::stream< int8_t > &v755 /* v755[8] */,
  hls::stream< int8_t > &v756 /* v756[8] */,
  hls::stream< int8_t > &v757 /* v757[8] */,
  int32_t v758[8][8],
  int v759,
  int v760
) {	// L904
  #pragma HLS stream variable=v754 depth=9
  #pragma HLS stream variable=v755 depth=9
  #pragma HLS stream variable=v756 depth=9
  #pragma HLS stream variable=v757 depth=9
  #pragma HLS array_partition variable=v758 complete dim=1
  #pragma HLS array_partition variable=v758 complete dim=2

  int32_t v29;	// L906
  v29 = 0;	// L907
  l_reduction_k29: for (int k29 = 0; k29 < 8; k29++) {	// L908
  #pragma HLS pipeline II=1
    int8_t v763 = v754.read(); // v754[k29];	// L909
    int8_t a29;	// L910
    a29 = v763;	// L911
    int8_t v765 = v755.read(); // v755[k29];	// L912
    int8_t b29;	// L913
    b29 = v765;	// L914
    int8_t v767 = a29;	// L915
    int8_t v768 = b29;	// L916
    int16_t v769 = v767;	// L917
    int16_t v770 = v768;	// L918
    int16_t v771 = v769 * v770;	// L919
    int32_t v772 = v29;	// L920
    ap_int<33> v773 = v772;	// L921
    ap_int<33> v774 = v771;	// L922
    ap_int<33> v775 = v773 + v774;	// L923
    int32_t v776 = v775;	// L924
    v29 = v776;	// L925
    int8_t v777 = a29;	// L926
    v756.write(v777); // v756[k29] = v777;	// L927
    int8_t v778 = b29;	// L928
    v757.write(v778); // v757[k29] = v778;	// L929
  }
  int32_t v779 = v29;	// L931
  v758[v759][v760] = v779;	// L932
}

void PE_kernel_gemm_6_3(
  hls::stream< int8_t > &v780 /* v780[8] */,
  hls::stream< int8_t > &v781 /* v781[8] */,
  hls::stream< int8_t > &v782 /* v782[8] */,
  hls::stream< int8_t > &v783 /* v783[8] */,
  int32_t v784[8][8],
  int v785,
  int v786
) {	// L935
  #pragma HLS stream variable=v780 depth=9
  #pragma HLS stream variable=v781 depth=9
  #pragma HLS stream variable=v782 depth=9
  #pragma HLS stream variable=v783 depth=9
  #pragma HLS array_partition variable=v784 complete dim=1
  #pragma HLS array_partition variable=v784 complete dim=2

  int32_t v30;	// L937
  v30 = 0;	// L938
  l_reduction_k30: for (int k30 = 0; k30 < 8; k30++) {	// L939
  #pragma HLS pipeline II=1
    int8_t v789 = v780.read(); // v780[k30];	// L940
    int8_t a30;	// L941
    a30 = v789;	// L942
    int8_t v791 = v781.read(); // v781[k30];	// L943
    int8_t b30;	// L944
    b30 = v791;	// L945
    int8_t v793 = a30;	// L946
    int8_t v794 = b30;	// L947
    int16_t v795 = v793;	// L948
    int16_t v796 = v794;	// L949
    int16_t v797 = v795 * v796;	// L950
    int32_t v798 = v30;	// L951
    ap_int<33> v799 = v798;	// L952
    ap_int<33> v800 = v797;	// L953
    ap_int<33> v801 = v799 + v800;	// L954
    int32_t v802 = v801;	// L955
    v30 = v802;	// L956
    int8_t v803 = a30;	// L957
    v782.write(v803); // v782[k30] = v803;	// L958
    int8_t v804 = b30;	// L959
    v783.write(v804); // v783[k30] = v804;	// L960
  }
  int32_t v805 = v30;	// L962
  v784[v785][v786] = v805;	// L963
}

void PE_kernel_gemm_7_3(
  hls::stream< int8_t > &v806 /* v806[8] */,
  hls::stream< int8_t > &v807 /* v807[8] */,
  hls::stream< int8_t > &v808 /* v808[8] */,
  hls::stream< int8_t > &v809 /* v809[8] */,
  int32_t v810[8][8],
  int v811,
  int v812
) {	// L966
  #pragma HLS stream variable=v806 depth=9
  #pragma HLS stream variable=v807 depth=9
  #pragma HLS stream variable=v808 depth=9
  #pragma HLS stream variable=v809 depth=9
  #pragma HLS array_partition variable=v810 complete dim=1
  #pragma HLS array_partition variable=v810 complete dim=2

  int32_t v31;	// L968
  v31 = 0;	// L969
  l_reduction_k31: for (int k31 = 0; k31 < 8; k31++) {	// L970
  #pragma HLS pipeline II=1
    int8_t v815 = v806.read(); // v806[k31];	// L971
    int8_t a31;	// L972
    a31 = v815;	// L973
    int8_t v817 = v807.read(); // v807[k31];	// L974
    int8_t b31;	// L975
    b31 = v817;	// L976
    int8_t v819 = a31;	// L977
    int8_t v820 = b31;	// L978
    int16_t v821 = v819;	// L979
    int16_t v822 = v820;	// L980
    int16_t v823 = v821 * v822;	// L981
    int32_t v824 = v31;	// L982
    ap_int<33> v825 = v824;	// L983
    ap_int<33> v826 = v823;	// L984
    ap_int<33> v827 = v825 + v826;	// L985
    int32_t v828 = v827;	// L986
    v31 = v828;	// L987
    int8_t v829 = a31;	// L988
    v808.write(v829); // v808[k31] = v829;	// L989
    int8_t v830 = b31;	// L990
    v809.write(v830); // v809[k31] = v830;	// L991
  }
  int32_t v831 = v31;	// L993
  v810[v811][v812] = v831;	// L994
}

void PE_kernel_gemm_0_4(
  hls::stream< int8_t > &v832 /* v832[8] */,
  hls::stream< int8_t > &v833 /* v833[8] */,
  hls::stream< int8_t > &v834 /* v834[8] */,
  hls::stream< int8_t > &v835 /* v835[8] */,
  int32_t v836[8][8],
  int v837,
  int v838
) {	// L997
  #pragma HLS stream variable=v832 depth=9
  #pragma HLS stream variable=v833 depth=9
  #pragma HLS stream variable=v834 depth=9
  #pragma HLS stream variable=v835 depth=9
  #pragma HLS array_partition variable=v836 complete dim=1
  #pragma HLS array_partition variable=v836 complete dim=2

  int32_t v32;	// L999
  v32 = 0;	// L1000
  l_reduction_k32: for (int k32 = 0; k32 < 8; k32++) {	// L1001
  #pragma HLS pipeline II=1
    int8_t v841 = v832.read(); // v832[k32];	// L1002
    int8_t a32;	// L1003
    a32 = v841;	// L1004
    int8_t v843 = v833.read(); // v833[k32];	// L1005
    int8_t b32;	// L1006
    b32 = v843;	// L1007
    int8_t v845 = a32;	// L1008
    int8_t v846 = b32;	// L1009
    int16_t v847 = v845;	// L1010
    int16_t v848 = v846;	// L1011
    int16_t v849 = v847 * v848;	// L1012
    int32_t v850 = v32;	// L1013
    ap_int<33> v851 = v850;	// L1014
    ap_int<33> v852 = v849;	// L1015
    ap_int<33> v853 = v851 + v852;	// L1016
    int32_t v854 = v853;	// L1017
    v32 = v854;	// L1018
    int8_t v855 = a32;	// L1019
    v834.write(v855); // v834[k32] = v855;	// L1020
    int8_t v856 = b32;	// L1021
    v835.write(v856); // v835[k32] = v856;	// L1022
  }
  int32_t v857 = v32;	// L1024
  v836[v837][v838] = v857;	// L1025
}

void PE_kernel_gemm_1_4(
  hls::stream< int8_t > &v858 /* v858[8] */,
  hls::stream< int8_t > &v859 /* v859[8] */,
  hls::stream< int8_t > &v860 /* v860[8] */,
  hls::stream< int8_t > &v861 /* v861[8] */,
  int32_t v862[8][8],
  int v863,
  int v864
) {	// L1028
  #pragma HLS stream variable=v858 depth=9
  #pragma HLS stream variable=v859 depth=9
  #pragma HLS stream variable=v860 depth=9
  #pragma HLS stream variable=v861 depth=9
  #pragma HLS array_partition variable=v862 complete dim=1
  #pragma HLS array_partition variable=v862 complete dim=2

  int32_t v33;	// L1030
  v33 = 0;	// L1031
  l_reduction_k33: for (int k33 = 0; k33 < 8; k33++) {	// L1032
  #pragma HLS pipeline II=1
    int8_t v867 = v858.read(); // v858[k33];	// L1033
    int8_t a33;	// L1034
    a33 = v867;	// L1035
    int8_t v869 = v859.read(); // v859[k33];	// L1036
    int8_t b33;	// L1037
    b33 = v869;	// L1038
    int8_t v871 = a33;	// L1039
    int8_t v872 = b33;	// L1040
    int16_t v873 = v871;	// L1041
    int16_t v874 = v872;	// L1042
    int16_t v875 = v873 * v874;	// L1043
    int32_t v876 = v33;	// L1044
    ap_int<33> v877 = v876;	// L1045
    ap_int<33> v878 = v875;	// L1046
    ap_int<33> v879 = v877 + v878;	// L1047
    int32_t v880 = v879;	// L1048
    v33 = v880;	// L1049
    int8_t v881 = a33;	// L1050
    v860.write(v881); // v860[k33] = v881;	// L1051
    int8_t v882 = b33;	// L1052
    v861.write(v882); // v861[k33] = v882;	// L1053
  }
  int32_t v883 = v33;	// L1055
  v862[v863][v864] = v883;	// L1056
}

void PE_kernel_gemm_2_4(
  hls::stream< int8_t > &v884 /* v884[8] */,
  hls::stream< int8_t > &v885 /* v885[8] */,
  hls::stream< int8_t > &v886 /* v886[8] */,
  hls::stream< int8_t > &v887 /* v887[8] */,
  int32_t v888[8][8],
  int v889,
  int v890
) {	// L1059
  #pragma HLS stream variable=v884 depth=9
  #pragma HLS stream variable=v885 depth=9
  #pragma HLS stream variable=v886 depth=9
  #pragma HLS stream variable=v887 depth=9
  #pragma HLS array_partition variable=v888 complete dim=1
  #pragma HLS array_partition variable=v888 complete dim=2

  int32_t v34;	// L1061
  v34 = 0;	// L1062
  l_reduction_k34: for (int k34 = 0; k34 < 8; k34++) {	// L1063
  #pragma HLS pipeline II=1
    int8_t v893 = v884.read(); // v884[k34];	// L1064
    int8_t a34;	// L1065
    a34 = v893;	// L1066
    int8_t v895 = v885.read(); // v885[k34];	// L1067
    int8_t b34;	// L1068
    b34 = v895;	// L1069
    int8_t v897 = a34;	// L1070
    int8_t v898 = b34;	// L1071
    int16_t v899 = v897;	// L1072
    int16_t v900 = v898;	// L1073
    int16_t v901 = v899 * v900;	// L1074
    int32_t v902 = v34;	// L1075
    ap_int<33> v903 = v902;	// L1076
    ap_int<33> v904 = v901;	// L1077
    ap_int<33> v905 = v903 + v904;	// L1078
    int32_t v906 = v905;	// L1079
    v34 = v906;	// L1080
    int8_t v907 = a34;	// L1081
    v886.write(v907); // v886[k34] = v907;	// L1082
    int8_t v908 = b34;	// L1083
    v887.write(v908); // v887[k34] = v908;	// L1084
  }
  int32_t v909 = v34;	// L1086
  v888[v889][v890] = v909;	// L1087
}

void PE_kernel_gemm_3_4(
  hls::stream< int8_t > &v910 /* v910[8] */,
  hls::stream< int8_t > &v911 /* v911[8] */,
  hls::stream< int8_t > &v912 /* v912[8] */,
  hls::stream< int8_t > &v913 /* v913[8] */,
  int32_t v914[8][8],
  int v915,
  int v916
) {	// L1090
  #pragma HLS stream variable=v910 depth=9
  #pragma HLS stream variable=v911 depth=9
  #pragma HLS stream variable=v912 depth=9
  #pragma HLS stream variable=v913 depth=9
  #pragma HLS array_partition variable=v914 complete dim=1
  #pragma HLS array_partition variable=v914 complete dim=2

  int32_t v35;	// L1092
  v35 = 0;	// L1093
  l_reduction_k35: for (int k35 = 0; k35 < 8; k35++) {	// L1094
  #pragma HLS pipeline II=1
    int8_t v919 = v910.read(); // v910[k35];	// L1095
    int8_t a35;	// L1096
    a35 = v919;	// L1097
    int8_t v921 = v911.read(); // v911[k35];	// L1098
    int8_t b35;	// L1099
    b35 = v921;	// L1100
    int8_t v923 = a35;	// L1101
    int8_t v924 = b35;	// L1102
    int16_t v925 = v923;	// L1103
    int16_t v926 = v924;	// L1104
    int16_t v927 = v925 * v926;	// L1105
    int32_t v928 = v35;	// L1106
    ap_int<33> v929 = v928;	// L1107
    ap_int<33> v930 = v927;	// L1108
    ap_int<33> v931 = v929 + v930;	// L1109
    int32_t v932 = v931;	// L1110
    v35 = v932;	// L1111
    int8_t v933 = a35;	// L1112
    v912.write(v933); // v912[k35] = v933;	// L1113
    int8_t v934 = b35;	// L1114
    v913.write(v934); // v913[k35] = v934;	// L1115
  }
  int32_t v935 = v35;	// L1117
  v914[v915][v916] = v935;	// L1118
}

void PE_kernel_gemm_4_4(
  hls::stream< int8_t > &v936 /* v936[8] */,
  hls::stream< int8_t > &v937 /* v937[8] */,
  hls::stream< int8_t > &v938 /* v938[8] */,
  hls::stream< int8_t > &v939 /* v939[8] */,
  int32_t v940[8][8],
  int v941,
  int v942
) {	// L1121
  #pragma HLS stream variable=v936 depth=9
  #pragma HLS stream variable=v937 depth=9
  #pragma HLS stream variable=v938 depth=9
  #pragma HLS stream variable=v939 depth=9
  #pragma HLS array_partition variable=v940 complete dim=1
  #pragma HLS array_partition variable=v940 complete dim=2

  int32_t v36;	// L1123
  v36 = 0;	// L1124
  l_reduction_k36: for (int k36 = 0; k36 < 8; k36++) {	// L1125
  #pragma HLS pipeline II=1
    int8_t v945 = v936.read(); // v936[k36];	// L1126
    int8_t a36;	// L1127
    a36 = v945;	// L1128
    int8_t v947 = v937.read(); // v937[k36];	// L1129
    int8_t b36;	// L1130
    b36 = v947;	// L1131
    int8_t v949 = a36;	// L1132
    int8_t v950 = b36;	// L1133
    int16_t v951 = v949;	// L1134
    int16_t v952 = v950;	// L1135
    int16_t v953 = v951 * v952;	// L1136
    int32_t v954 = v36;	// L1137
    ap_int<33> v955 = v954;	// L1138
    ap_int<33> v956 = v953;	// L1139
    ap_int<33> v957 = v955 + v956;	// L1140
    int32_t v958 = v957;	// L1141
    v36 = v958;	// L1142
    int8_t v959 = a36;	// L1143
    v938.write(v959); // v938[k36] = v959;	// L1144
    int8_t v960 = b36;	// L1145
    v939.write(v960); // v939[k36] = v960;	// L1146
  }
  int32_t v961 = v36;	// L1148
  v940[v941][v942] = v961;	// L1149
}

void PE_kernel_gemm_5_4(
  hls::stream< int8_t > &v962 /* v962[8] */,
  hls::stream< int8_t > &v963 /* v963[8] */,
  hls::stream< int8_t > &v964 /* v964[8] */,
  hls::stream< int8_t > &v965 /* v965[8] */,
  int32_t v966[8][8],
  int v967,
  int v968
) {	// L1152
  #pragma HLS stream variable=v962 depth=9
  #pragma HLS stream variable=v963 depth=9
  #pragma HLS stream variable=v964 depth=9
  #pragma HLS stream variable=v965 depth=9
  #pragma HLS array_partition variable=v966 complete dim=1
  #pragma HLS array_partition variable=v966 complete dim=2

  int32_t v37;	// L1154
  v37 = 0;	// L1155
  l_reduction_k37: for (int k37 = 0; k37 < 8; k37++) {	// L1156
  #pragma HLS pipeline II=1
    int8_t v971 = v962.read(); // v962[k37];	// L1157
    int8_t a37;	// L1158
    a37 = v971;	// L1159
    int8_t v973 = v963.read(); // v963[k37];	// L1160
    int8_t b37;	// L1161
    b37 = v973;	// L1162
    int8_t v975 = a37;	// L1163
    int8_t v976 = b37;	// L1164
    int16_t v977 = v975;	// L1165
    int16_t v978 = v976;	// L1166
    int16_t v979 = v977 * v978;	// L1167
    int32_t v980 = v37;	// L1168
    ap_int<33> v981 = v980;	// L1169
    ap_int<33> v982 = v979;	// L1170
    ap_int<33> v983 = v981 + v982;	// L1171
    int32_t v984 = v983;	// L1172
    v37 = v984;	// L1173
    int8_t v985 = a37;	// L1174
    v964.write(v985); // v964[k37] = v985;	// L1175
    int8_t v986 = b37;	// L1176
    v965.write(v986); // v965[k37] = v986;	// L1177
  }
  int32_t v987 = v37;	// L1179
  v966[v967][v968] = v987;	// L1180
}

void PE_kernel_gemm_6_4(
  hls::stream< int8_t > &v988 /* v988[8] */,
  hls::stream< int8_t > &v989 /* v989[8] */,
  hls::stream< int8_t > &v990 /* v990[8] */,
  hls::stream< int8_t > &v991 /* v991[8] */,
  int32_t v992[8][8],
  int v993,
  int v994
) {	// L1183
  #pragma HLS stream variable=v988 depth=9
  #pragma HLS stream variable=v989 depth=9
  #pragma HLS stream variable=v990 depth=9
  #pragma HLS stream variable=v991 depth=9
  #pragma HLS array_partition variable=v992 complete dim=1
  #pragma HLS array_partition variable=v992 complete dim=2

  int32_t v38;	// L1185
  v38 = 0;	// L1186
  l_reduction_k38: for (int k38 = 0; k38 < 8; k38++) {	// L1187
  #pragma HLS pipeline II=1
    int8_t v997 = v988.read(); // v988[k38];	// L1188
    int8_t a38;	// L1189
    a38 = v997;	// L1190
    int8_t v999 = v989.read(); // v989[k38];	// L1191
    int8_t b38;	// L1192
    b38 = v999;	// L1193
    int8_t v1001 = a38;	// L1194
    int8_t v1002 = b38;	// L1195
    int16_t v1003 = v1001;	// L1196
    int16_t v1004 = v1002;	// L1197
    int16_t v1005 = v1003 * v1004;	// L1198
    int32_t v1006 = v38;	// L1199
    ap_int<33> v1007 = v1006;	// L1200
    ap_int<33> v1008 = v1005;	// L1201
    ap_int<33> v1009 = v1007 + v1008;	// L1202
    int32_t v1010 = v1009;	// L1203
    v38 = v1010;	// L1204
    int8_t v1011 = a38;	// L1205
    v990.write(v1011); // v990[k38] = v1011;	// L1206
    int8_t v1012 = b38;	// L1207
    v991.write(v1012); // v991[k38] = v1012;	// L1208
  }
  int32_t v1013 = v38;	// L1210
  v992[v993][v994] = v1013;	// L1211
}

void PE_kernel_gemm_7_4(
  hls::stream< int8_t > &v1014 /* v1014[8] */,
  hls::stream< int8_t > &v1015 /* v1015[8] */,
  hls::stream< int8_t > &v1016 /* v1016[8] */,
  hls::stream< int8_t > &v1017 /* v1017[8] */,
  int32_t v1018[8][8],
  int v1019,
  int v1020
) {	// L1214
  #pragma HLS stream variable=v1014 depth=9
  #pragma HLS stream variable=v1015 depth=9
  #pragma HLS stream variable=v1016 depth=9
  #pragma HLS stream variable=v1017 depth=9
  #pragma HLS array_partition variable=v1018 complete dim=1
  #pragma HLS array_partition variable=v1018 complete dim=2

  int32_t v39;	// L1216
  v39 = 0;	// L1217
  l_reduction_k39: for (int k39 = 0; k39 < 8; k39++) {	// L1218
  #pragma HLS pipeline II=1
    int8_t v1023 = v1014.read(); // v1014[k39];	// L1219
    int8_t a39;	// L1220
    a39 = v1023;	// L1221
    int8_t v1025 = v1015.read(); // v1015[k39];	// L1222
    int8_t b39;	// L1223
    b39 = v1025;	// L1224
    int8_t v1027 = a39;	// L1225
    int8_t v1028 = b39;	// L1226
    int16_t v1029 = v1027;	// L1227
    int16_t v1030 = v1028;	// L1228
    int16_t v1031 = v1029 * v1030;	// L1229
    int32_t v1032 = v39;	// L1230
    ap_int<33> v1033 = v1032;	// L1231
    ap_int<33> v1034 = v1031;	// L1232
    ap_int<33> v1035 = v1033 + v1034;	// L1233
    int32_t v1036 = v1035;	// L1234
    v39 = v1036;	// L1235
    int8_t v1037 = a39;	// L1236
    v1016.write(v1037); // v1016[k39] = v1037;	// L1237
    int8_t v1038 = b39;	// L1238
    v1017.write(v1038); // v1017[k39] = v1038;	// L1239
  }
  int32_t v1039 = v39;	// L1241
  v1018[v1019][v1020] = v1039;	// L1242
}

void PE_kernel_gemm_0_5(
  hls::stream< int8_t > &v1040 /* v1040[8] */,
  hls::stream< int8_t > &v1041 /* v1041[8] */,
  hls::stream< int8_t > &v1042 /* v1042[8] */,
  hls::stream< int8_t > &v1043 /* v1043[8] */,
  int32_t v1044[8][8],
  int v1045,
  int v1046
) {	// L1245
  #pragma HLS stream variable=v1040 depth=9
  #pragma HLS stream variable=v1041 depth=9
  #pragma HLS stream variable=v1042 depth=9
  #pragma HLS stream variable=v1043 depth=9
  #pragma HLS array_partition variable=v1044 complete dim=1
  #pragma HLS array_partition variable=v1044 complete dim=2

  int32_t v40;	// L1247
  v40 = 0;	// L1248
  l_reduction_k40: for (int k40 = 0; k40 < 8; k40++) {	// L1249
  #pragma HLS pipeline II=1
    int8_t v1049 = v1040.read(); // v1040[k40];	// L1250
    int8_t a40;	// L1251
    a40 = v1049;	// L1252
    int8_t v1051 = v1041.read(); // v1041[k40];	// L1253
    int8_t b40;	// L1254
    b40 = v1051;	// L1255
    int8_t v1053 = a40;	// L1256
    int8_t v1054 = b40;	// L1257
    int16_t v1055 = v1053;	// L1258
    int16_t v1056 = v1054;	// L1259
    int16_t v1057 = v1055 * v1056;	// L1260
    int32_t v1058 = v40;	// L1261
    ap_int<33> v1059 = v1058;	// L1262
    ap_int<33> v1060 = v1057;	// L1263
    ap_int<33> v1061 = v1059 + v1060;	// L1264
    int32_t v1062 = v1061;	// L1265
    v40 = v1062;	// L1266
    int8_t v1063 = a40;	// L1267
    v1042.write(v1063); // v1042[k40] = v1063;	// L1268
    int8_t v1064 = b40;	// L1269
    v1043.write(v1064); // v1043[k40] = v1064;	// L1270
  }
  int32_t v1065 = v40;	// L1272
  v1044[v1045][v1046] = v1065;	// L1273
}

void PE_kernel_gemm_1_5(
  hls::stream< int8_t > &v1066 /* v1066[8] */,
  hls::stream< int8_t > &v1067 /* v1067[8] */,
  hls::stream< int8_t > &v1068 /* v1068[8] */,
  hls::stream< int8_t > &v1069 /* v1069[8] */,
  int32_t v1070[8][8],
  int v1071,
  int v1072
) {	// L1276
  #pragma HLS stream variable=v1066 depth=9
  #pragma HLS stream variable=v1067 depth=9
  #pragma HLS stream variable=v1068 depth=9
  #pragma HLS stream variable=v1069 depth=9
  #pragma HLS array_partition variable=v1070 complete dim=1
  #pragma HLS array_partition variable=v1070 complete dim=2

  int32_t v41;	// L1278
  v41 = 0;	// L1279
  l_reduction_k41: for (int k41 = 0; k41 < 8; k41++) {	// L1280
  #pragma HLS pipeline II=1
    int8_t v1075 = v1066.read(); // v1066[k41];	// L1281
    int8_t a41;	// L1282
    a41 = v1075;	// L1283
    int8_t v1077 = v1067.read(); // v1067[k41];	// L1284
    int8_t b41;	// L1285
    b41 = v1077;	// L1286
    int8_t v1079 = a41;	// L1287
    int8_t v1080 = b41;	// L1288
    int16_t v1081 = v1079;	// L1289
    int16_t v1082 = v1080;	// L1290
    int16_t v1083 = v1081 * v1082;	// L1291
    int32_t v1084 = v41;	// L1292
    ap_int<33> v1085 = v1084;	// L1293
    ap_int<33> v1086 = v1083;	// L1294
    ap_int<33> v1087 = v1085 + v1086;	// L1295
    int32_t v1088 = v1087;	// L1296
    v41 = v1088;	// L1297
    int8_t v1089 = a41;	// L1298
    v1068.write(v1089); // v1068[k41] = v1089;	// L1299
    int8_t v1090 = b41;	// L1300
    v1069.write(v1090); // v1069[k41] = v1090;	// L1301
  }
  int32_t v1091 = v41;	// L1303
  v1070[v1071][v1072] = v1091;	// L1304
}

void PE_kernel_gemm_2_5(
  hls::stream< int8_t > &v1092 /* v1092[8] */,
  hls::stream< int8_t > &v1093 /* v1093[8] */,
  hls::stream< int8_t > &v1094 /* v1094[8] */,
  hls::stream< int8_t > &v1095 /* v1095[8] */,
  int32_t v1096[8][8],
  int v1097,
  int v1098
) {	// L1307
  #pragma HLS stream variable=v1092 depth=9
  #pragma HLS stream variable=v1093 depth=9
  #pragma HLS stream variable=v1094 depth=9
  #pragma HLS stream variable=v1095 depth=9
  #pragma HLS array_partition variable=v1096 complete dim=1
  #pragma HLS array_partition variable=v1096 complete dim=2

  int32_t v42;	// L1309
  v42 = 0;	// L1310
  l_reduction_k42: for (int k42 = 0; k42 < 8; k42++) {	// L1311
  #pragma HLS pipeline II=1
    int8_t v1101 = v1092.read(); // v1092[k42];	// L1312
    int8_t a42;	// L1313
    a42 = v1101;	// L1314
    int8_t v1103 = v1093.read(); // v1093[k42];	// L1315
    int8_t b42;	// L1316
    b42 = v1103;	// L1317
    int8_t v1105 = a42;	// L1318
    int8_t v1106 = b42;	// L1319
    int16_t v1107 = v1105;	// L1320
    int16_t v1108 = v1106;	// L1321
    int16_t v1109 = v1107 * v1108;	// L1322
    int32_t v1110 = v42;	// L1323
    ap_int<33> v1111 = v1110;	// L1324
    ap_int<33> v1112 = v1109;	// L1325
    ap_int<33> v1113 = v1111 + v1112;	// L1326
    int32_t v1114 = v1113;	// L1327
    v42 = v1114;	// L1328
    int8_t v1115 = a42;	// L1329
    v1094.write(v1115); // v1094[k42] = v1115;	// L1330
    int8_t v1116 = b42;	// L1331
    v1095.write(v1116); // v1095[k42] = v1116;	// L1332
  }
  int32_t v1117 = v42;	// L1334
  v1096[v1097][v1098] = v1117;	// L1335
}

void PE_kernel_gemm_3_5(
  hls::stream< int8_t > &v1118 /* v1118[8] */,
  hls::stream< int8_t > &v1119 /* v1119[8] */,
  hls::stream< int8_t > &v1120 /* v1120[8] */,
  hls::stream< int8_t > &v1121 /* v1121[8] */,
  int32_t v1122[8][8],
  int v1123,
  int v1124
) {	// L1338
  #pragma HLS stream variable=v1118 depth=9
  #pragma HLS stream variable=v1119 depth=9
  #pragma HLS stream variable=v1120 depth=9
  #pragma HLS stream variable=v1121 depth=9
  #pragma HLS array_partition variable=v1122 complete dim=1
  #pragma HLS array_partition variable=v1122 complete dim=2

  int32_t v43;	// L1340
  v43 = 0;	// L1341
  l_reduction_k43: for (int k43 = 0; k43 < 8; k43++) {	// L1342
  #pragma HLS pipeline II=1
    int8_t v1127 = v1118.read(); // v1118[k43];	// L1343
    int8_t a43;	// L1344
    a43 = v1127;	// L1345
    int8_t v1129 = v1119.read(); // v1119[k43];	// L1346
    int8_t b43;	// L1347
    b43 = v1129;	// L1348
    int8_t v1131 = a43;	// L1349
    int8_t v1132 = b43;	// L1350
    int16_t v1133 = v1131;	// L1351
    int16_t v1134 = v1132;	// L1352
    int16_t v1135 = v1133 * v1134;	// L1353
    int32_t v1136 = v43;	// L1354
    ap_int<33> v1137 = v1136;	// L1355
    ap_int<33> v1138 = v1135;	// L1356
    ap_int<33> v1139 = v1137 + v1138;	// L1357
    int32_t v1140 = v1139;	// L1358
    v43 = v1140;	// L1359
    int8_t v1141 = a43;	// L1360
    v1120.write(v1141); // v1120[k43] = v1141;	// L1361
    int8_t v1142 = b43;	// L1362
    v1121.write(v1142); // v1121[k43] = v1142;	// L1363
  }
  int32_t v1143 = v43;	// L1365
  v1122[v1123][v1124] = v1143;	// L1366
}

void PE_kernel_gemm_4_5(
  hls::stream< int8_t > &v1144 /* v1144[8] */,
  hls::stream< int8_t > &v1145 /* v1145[8] */,
  hls::stream< int8_t > &v1146 /* v1146[8] */,
  hls::stream< int8_t > &v1147 /* v1147[8] */,
  int32_t v1148[8][8],
  int v1149,
  int v1150
) {	// L1369
  #pragma HLS stream variable=v1144 depth=9
  #pragma HLS stream variable=v1145 depth=9
  #pragma HLS stream variable=v1146 depth=9
  #pragma HLS stream variable=v1147 depth=9
  #pragma HLS array_partition variable=v1148 complete dim=1
  #pragma HLS array_partition variable=v1148 complete dim=2

  int32_t v44;	// L1371
  v44 = 0;	// L1372
  l_reduction_k44: for (int k44 = 0; k44 < 8; k44++) {	// L1373
  #pragma HLS pipeline II=1
    int8_t v1153 = v1144.read(); // v1144[k44];	// L1374
    int8_t a44;	// L1375
    a44 = v1153;	// L1376
    int8_t v1155 = v1145.read(); // v1145[k44];	// L1377
    int8_t b44;	// L1378
    b44 = v1155;	// L1379
    int8_t v1157 = a44;	// L1380
    int8_t v1158 = b44;	// L1381
    int16_t v1159 = v1157;	// L1382
    int16_t v1160 = v1158;	// L1383
    int16_t v1161 = v1159 * v1160;	// L1384
    int32_t v1162 = v44;	// L1385
    ap_int<33> v1163 = v1162;	// L1386
    ap_int<33> v1164 = v1161;	// L1387
    ap_int<33> v1165 = v1163 + v1164;	// L1388
    int32_t v1166 = v1165;	// L1389
    v44 = v1166;	// L1390
    int8_t v1167 = a44;	// L1391
    v1146.write(v1167); // v1146[k44] = v1167;	// L1392
    int8_t v1168 = b44;	// L1393
    v1147.write(v1168); // v1147[k44] = v1168;	// L1394
  }
  int32_t v1169 = v44;	// L1396
  v1148[v1149][v1150] = v1169;	// L1397
}

void PE_kernel_gemm_5_5(
  hls::stream< int8_t > &v1170 /* v1170[8] */,
  hls::stream< int8_t > &v1171 /* v1171[8] */,
  hls::stream< int8_t > &v1172 /* v1172[8] */,
  hls::stream< int8_t > &v1173 /* v1173[8] */,
  int32_t v1174[8][8],
  int v1175,
  int v1176
) {	// L1400
  #pragma HLS stream variable=v1170 depth=9
  #pragma HLS stream variable=v1171 depth=9
  #pragma HLS stream variable=v1172 depth=9
  #pragma HLS stream variable=v1173 depth=9
  #pragma HLS array_partition variable=v1174 complete dim=1
  #pragma HLS array_partition variable=v1174 complete dim=2

  int32_t v45;	// L1402
  v45 = 0;	// L1403
  l_reduction_k45: for (int k45 = 0; k45 < 8; k45++) {	// L1404
  #pragma HLS pipeline II=1
    int8_t v1179 = v1170.read(); // v1170[k45];	// L1405
    int8_t a45;	// L1406
    a45 = v1179;	// L1407
    int8_t v1181 = v1171.read(); // v1171[k45];	// L1408
    int8_t b45;	// L1409
    b45 = v1181;	// L1410
    int8_t v1183 = a45;	// L1411
    int8_t v1184 = b45;	// L1412
    int16_t v1185 = v1183;	// L1413
    int16_t v1186 = v1184;	// L1414
    int16_t v1187 = v1185 * v1186;	// L1415
    int32_t v1188 = v45;	// L1416
    ap_int<33> v1189 = v1188;	// L1417
    ap_int<33> v1190 = v1187;	// L1418
    ap_int<33> v1191 = v1189 + v1190;	// L1419
    int32_t v1192 = v1191;	// L1420
    v45 = v1192;	// L1421
    int8_t v1193 = a45;	// L1422
    v1172.write(v1193); // v1172[k45] = v1193;	// L1423
    int8_t v1194 = b45;	// L1424
    v1173.write(v1194); // v1173[k45] = v1194;	// L1425
  }
  int32_t v1195 = v45;	// L1427
  v1174[v1175][v1176] = v1195;	// L1428
}

void PE_kernel_gemm_6_5(
  hls::stream< int8_t > &v1196 /* v1196[8] */,
  hls::stream< int8_t > &v1197 /* v1197[8] */,
  hls::stream< int8_t > &v1198 /* v1198[8] */,
  hls::stream< int8_t > &v1199 /* v1199[8] */,
  int32_t v1200[8][8],
  int v1201,
  int v1202
) {	// L1431
  #pragma HLS stream variable=v1196 depth=9
  #pragma HLS stream variable=v1197 depth=9
  #pragma HLS stream variable=v1198 depth=9
  #pragma HLS stream variable=v1199 depth=9
  #pragma HLS array_partition variable=v1200 complete dim=1
  #pragma HLS array_partition variable=v1200 complete dim=2

  int32_t v46;	// L1433
  v46 = 0;	// L1434
  l_reduction_k46: for (int k46 = 0; k46 < 8; k46++) {	// L1435
  #pragma HLS pipeline II=1
    int8_t v1205 = v1196.read(); // v1196[k46];	// L1436
    int8_t a46;	// L1437
    a46 = v1205;	// L1438
    int8_t v1207 = v1197.read(); // v1197[k46];	// L1439
    int8_t b46;	// L1440
    b46 = v1207;	// L1441
    int8_t v1209 = a46;	// L1442
    int8_t v1210 = b46;	// L1443
    int16_t v1211 = v1209;	// L1444
    int16_t v1212 = v1210;	// L1445
    int16_t v1213 = v1211 * v1212;	// L1446
    int32_t v1214 = v46;	// L1447
    ap_int<33> v1215 = v1214;	// L1448
    ap_int<33> v1216 = v1213;	// L1449
    ap_int<33> v1217 = v1215 + v1216;	// L1450
    int32_t v1218 = v1217;	// L1451
    v46 = v1218;	// L1452
    int8_t v1219 = a46;	// L1453
    v1198.write(v1219); // v1198[k46] = v1219;	// L1454
    int8_t v1220 = b46;	// L1455
    v1199.write(v1220); // v1199[k46] = v1220;	// L1456
  }
  int32_t v1221 = v46;	// L1458
  v1200[v1201][v1202] = v1221;	// L1459
}

void PE_kernel_gemm_7_5(
  hls::stream< int8_t > &v1222 /* v1222[8] */,
  hls::stream< int8_t > &v1223 /* v1223[8] */,
  hls::stream< int8_t > &v1224 /* v1224[8] */,
  hls::stream< int8_t > &v1225 /* v1225[8] */,
  int32_t v1226[8][8],
  int v1227,
  int v1228
) {	// L1462
  #pragma HLS stream variable=v1222 depth=9
  #pragma HLS stream variable=v1223 depth=9
  #pragma HLS stream variable=v1224 depth=9
  #pragma HLS stream variable=v1225 depth=9
  #pragma HLS array_partition variable=v1226 complete dim=1
  #pragma HLS array_partition variable=v1226 complete dim=2

  int32_t v47;	// L1464
  v47 = 0;	// L1465
  l_reduction_k47: for (int k47 = 0; k47 < 8; k47++) {	// L1466
  #pragma HLS pipeline II=1
    int8_t v1231 = v1222.read(); // v1222[k47];	// L1467
    int8_t a47;	// L1468
    a47 = v1231;	// L1469
    int8_t v1233 = v1223.read(); // v1223[k47];	// L1470
    int8_t b47;	// L1471
    b47 = v1233;	// L1472
    int8_t v1235 = a47;	// L1473
    int8_t v1236 = b47;	// L1474
    int16_t v1237 = v1235;	// L1475
    int16_t v1238 = v1236;	// L1476
    int16_t v1239 = v1237 * v1238;	// L1477
    int32_t v1240 = v47;	// L1478
    ap_int<33> v1241 = v1240;	// L1479
    ap_int<33> v1242 = v1239;	// L1480
    ap_int<33> v1243 = v1241 + v1242;	// L1481
    int32_t v1244 = v1243;	// L1482
    v47 = v1244;	// L1483
    int8_t v1245 = a47;	// L1484
    v1224.write(v1245); // v1224[k47] = v1245;	// L1485
    int8_t v1246 = b47;	// L1486
    v1225.write(v1246); // v1225[k47] = v1246;	// L1487
  }
  int32_t v1247 = v47;	// L1489
  v1226[v1227][v1228] = v1247;	// L1490
}

void PE_kernel_gemm_0_6(
  hls::stream< int8_t > &v1248 /* v1248[8] */,
  hls::stream< int8_t > &v1249 /* v1249[8] */,
  hls::stream< int8_t > &v1250 /* v1250[8] */,
  hls::stream< int8_t > &v1251 /* v1251[8] */,
  int32_t v1252[8][8],
  int v1253,
  int v1254
) {	// L1493
  #pragma HLS stream variable=v1248 depth=9
  #pragma HLS stream variable=v1249 depth=9
  #pragma HLS stream variable=v1250 depth=9
  #pragma HLS stream variable=v1251 depth=9
  #pragma HLS array_partition variable=v1252 complete dim=1
  #pragma HLS array_partition variable=v1252 complete dim=2

  int32_t v48;	// L1495
  v48 = 0;	// L1496
  l_reduction_k48: for (int k48 = 0; k48 < 8; k48++) {	// L1497
  #pragma HLS pipeline II=1
    int8_t v1257 = v1248.read(); // v1248[k48];	// L1498
    int8_t a48;	// L1499
    a48 = v1257;	// L1500
    int8_t v1259 = v1249.read(); // v1249[k48];	// L1501
    int8_t b48;	// L1502
    b48 = v1259;	// L1503
    int8_t v1261 = a48;	// L1504
    int8_t v1262 = b48;	// L1505
    int16_t v1263 = v1261;	// L1506
    int16_t v1264 = v1262;	// L1507
    int16_t v1265 = v1263 * v1264;	// L1508
    int32_t v1266 = v48;	// L1509
    ap_int<33> v1267 = v1266;	// L1510
    ap_int<33> v1268 = v1265;	// L1511
    ap_int<33> v1269 = v1267 + v1268;	// L1512
    int32_t v1270 = v1269;	// L1513
    v48 = v1270;	// L1514
    int8_t v1271 = a48;	// L1515
    v1250.write(v1271); // v1250[k48] = v1271;	// L1516
    int8_t v1272 = b48;	// L1517
    v1251.write(v1272); // v1251[k48] = v1272;	// L1518
  }
  int32_t v1273 = v48;	// L1520
  v1252[v1253][v1254] = v1273;	// L1521
}

void PE_kernel_gemm_1_6(
  hls::stream< int8_t > &v1274 /* v1274[8] */,
  hls::stream< int8_t > &v1275 /* v1275[8] */,
  hls::stream< int8_t > &v1276 /* v1276[8] */,
  hls::stream< int8_t > &v1277 /* v1277[8] */,
  int32_t v1278[8][8],
  int v1279,
  int v1280
) {	// L1524
  #pragma HLS stream variable=v1274 depth=9
  #pragma HLS stream variable=v1275 depth=9
  #pragma HLS stream variable=v1276 depth=9
  #pragma HLS stream variable=v1277 depth=9
  #pragma HLS array_partition variable=v1278 complete dim=1
  #pragma HLS array_partition variable=v1278 complete dim=2

  int32_t v49;	// L1526
  v49 = 0;	// L1527
  l_reduction_k49: for (int k49 = 0; k49 < 8; k49++) {	// L1528
  #pragma HLS pipeline II=1
    int8_t v1283 = v1274.read(); // v1274[k49];	// L1529
    int8_t a49;	// L1530
    a49 = v1283;	// L1531
    int8_t v1285 = v1275.read(); // v1275[k49];	// L1532
    int8_t b49;	// L1533
    b49 = v1285;	// L1534
    int8_t v1287 = a49;	// L1535
    int8_t v1288 = b49;	// L1536
    int16_t v1289 = v1287;	// L1537
    int16_t v1290 = v1288;	// L1538
    int16_t v1291 = v1289 * v1290;	// L1539
    int32_t v1292 = v49;	// L1540
    ap_int<33> v1293 = v1292;	// L1541
    ap_int<33> v1294 = v1291;	// L1542
    ap_int<33> v1295 = v1293 + v1294;	// L1543
    int32_t v1296 = v1295;	// L1544
    v49 = v1296;	// L1545
    int8_t v1297 = a49;	// L1546
    v1276.write(v1297); // v1276[k49] = v1297;	// L1547
    int8_t v1298 = b49;	// L1548
    v1277.write(v1298); // v1277[k49] = v1298;	// L1549
  }
  int32_t v1299 = v49;	// L1551
  v1278[v1279][v1280] = v1299;	// L1552
}

void PE_kernel_gemm_2_6(
  hls::stream< int8_t > &v1300 /* v1300[8] */,
  hls::stream< int8_t > &v1301 /* v1301[8] */,
  hls::stream< int8_t > &v1302 /* v1302[8] */,
  hls::stream< int8_t > &v1303 /* v1303[8] */,
  int32_t v1304[8][8],
  int v1305,
  int v1306
) {	// L1555
  #pragma HLS stream variable=v1300 depth=9
  #pragma HLS stream variable=v1301 depth=9
  #pragma HLS stream variable=v1302 depth=9
  #pragma HLS stream variable=v1303 depth=9
  #pragma HLS array_partition variable=v1304 complete dim=1
  #pragma HLS array_partition variable=v1304 complete dim=2

  int32_t v50;	// L1557
  v50 = 0;	// L1558
  l_reduction_k50: for (int k50 = 0; k50 < 8; k50++) {	// L1559
  #pragma HLS pipeline II=1
    int8_t v1309 = v1300.read(); // v1300[k50];	// L1560
    int8_t a50;	// L1561
    a50 = v1309;	// L1562
    int8_t v1311 = v1301.read(); // v1301[k50];	// L1563
    int8_t b50;	// L1564
    b50 = v1311;	// L1565
    int8_t v1313 = a50;	// L1566
    int8_t v1314 = b50;	// L1567
    int16_t v1315 = v1313;	// L1568
    int16_t v1316 = v1314;	// L1569
    int16_t v1317 = v1315 * v1316;	// L1570
    int32_t v1318 = v50;	// L1571
    ap_int<33> v1319 = v1318;	// L1572
    ap_int<33> v1320 = v1317;	// L1573
    ap_int<33> v1321 = v1319 + v1320;	// L1574
    int32_t v1322 = v1321;	// L1575
    v50 = v1322;	// L1576
    int8_t v1323 = a50;	// L1577
    v1302.write(v1323); // v1302[k50] = v1323;	// L1578
    int8_t v1324 = b50;	// L1579
    v1303.write(v1324); // v1303[k50] = v1324;	// L1580
  }
  int32_t v1325 = v50;	// L1582
  v1304[v1305][v1306] = v1325;	// L1583
}

void PE_kernel_gemm_3_6(
  hls::stream< int8_t > &v1326 /* v1326[8] */,
  hls::stream< int8_t > &v1327 /* v1327[8] */,
  hls::stream< int8_t > &v1328 /* v1328[8] */,
  hls::stream< int8_t > &v1329 /* v1329[8] */,
  int32_t v1330[8][8],
  int v1331,
  int v1332
) {	// L1586
  #pragma HLS stream variable=v1326 depth=9
  #pragma HLS stream variable=v1327 depth=9
  #pragma HLS stream variable=v1328 depth=9
  #pragma HLS stream variable=v1329 depth=9
  #pragma HLS array_partition variable=v1330 complete dim=1
  #pragma HLS array_partition variable=v1330 complete dim=2

  int32_t v51;	// L1588
  v51 = 0;	// L1589
  l_reduction_k51: for (int k51 = 0; k51 < 8; k51++) {	// L1590
  #pragma HLS pipeline II=1
    int8_t v1335 = v1326.read(); // v1326[k51];	// L1591
    int8_t a51;	// L1592
    a51 = v1335;	// L1593
    int8_t v1337 = v1327.read(); // v1327[k51];	// L1594
    int8_t b51;	// L1595
    b51 = v1337;	// L1596
    int8_t v1339 = a51;	// L1597
    int8_t v1340 = b51;	// L1598
    int16_t v1341 = v1339;	// L1599
    int16_t v1342 = v1340;	// L1600
    int16_t v1343 = v1341 * v1342;	// L1601
    int32_t v1344 = v51;	// L1602
    ap_int<33> v1345 = v1344;	// L1603
    ap_int<33> v1346 = v1343;	// L1604
    ap_int<33> v1347 = v1345 + v1346;	// L1605
    int32_t v1348 = v1347;	// L1606
    v51 = v1348;	// L1607
    int8_t v1349 = a51;	// L1608
    v1328.write(v1349); // v1328[k51] = v1349;	// L1609
    int8_t v1350 = b51;	// L1610
    v1329.write(v1350); // v1329[k51] = v1350;	// L1611
  }
  int32_t v1351 = v51;	// L1613
  v1330[v1331][v1332] = v1351;	// L1614
}

void PE_kernel_gemm_4_6(
  hls::stream< int8_t > &v1352 /* v1352[8] */,
  hls::stream< int8_t > &v1353 /* v1353[8] */,
  hls::stream< int8_t > &v1354 /* v1354[8] */,
  hls::stream< int8_t > &v1355 /* v1355[8] */,
  int32_t v1356[8][8],
  int v1357,
  int v1358
) {	// L1617
  #pragma HLS stream variable=v1352 depth=9
  #pragma HLS stream variable=v1353 depth=9
  #pragma HLS stream variable=v1354 depth=9
  #pragma HLS stream variable=v1355 depth=9
  #pragma HLS array_partition variable=v1356 complete dim=1
  #pragma HLS array_partition variable=v1356 complete dim=2

  int32_t v52;	// L1619
  v52 = 0;	// L1620
  l_reduction_k52: for (int k52 = 0; k52 < 8; k52++) {	// L1621
  #pragma HLS pipeline II=1
    int8_t v1361 = v1352.read(); // v1352[k52];	// L1622
    int8_t a52;	// L1623
    a52 = v1361;	// L1624
    int8_t v1363 = v1353.read(); // v1353[k52];	// L1625
    int8_t b52;	// L1626
    b52 = v1363;	// L1627
    int8_t v1365 = a52;	// L1628
    int8_t v1366 = b52;	// L1629
    int16_t v1367 = v1365;	// L1630
    int16_t v1368 = v1366;	// L1631
    int16_t v1369 = v1367 * v1368;	// L1632
    int32_t v1370 = v52;	// L1633
    ap_int<33> v1371 = v1370;	// L1634
    ap_int<33> v1372 = v1369;	// L1635
    ap_int<33> v1373 = v1371 + v1372;	// L1636
    int32_t v1374 = v1373;	// L1637
    v52 = v1374;	// L1638
    int8_t v1375 = a52;	// L1639
    v1354.write(v1375); // v1354[k52] = v1375;	// L1640
    int8_t v1376 = b52;	// L1641
    v1355.write(v1376); // v1355[k52] = v1376;	// L1642
  }
  int32_t v1377 = v52;	// L1644
  v1356[v1357][v1358] = v1377;	// L1645
}

void PE_kernel_gemm_5_6(
  hls::stream< int8_t > &v1378 /* v1378[8] */,
  hls::stream< int8_t > &v1379 /* v1379[8] */,
  hls::stream< int8_t > &v1380 /* v1380[8] */,
  hls::stream< int8_t > &v1381 /* v1381[8] */,
  int32_t v1382[8][8],
  int v1383,
  int v1384
) {	// L1648
  #pragma HLS stream variable=v1378 depth=9
  #pragma HLS stream variable=v1379 depth=9
  #pragma HLS stream variable=v1380 depth=9
  #pragma HLS stream variable=v1381 depth=9
  #pragma HLS array_partition variable=v1382 complete dim=1
  #pragma HLS array_partition variable=v1382 complete dim=2

  int32_t v53;	// L1650
  v53 = 0;	// L1651
  l_reduction_k53: for (int k53 = 0; k53 < 8; k53++) {	// L1652
  #pragma HLS pipeline II=1
    int8_t v1387 = v1378.read(); // v1378[k53];	// L1653
    int8_t a53;	// L1654
    a53 = v1387;	// L1655
    int8_t v1389 = v1379.read(); // v1379[k53];	// L1656
    int8_t b53;	// L1657
    b53 = v1389;	// L1658
    int8_t v1391 = a53;	// L1659
    int8_t v1392 = b53;	// L1660
    int16_t v1393 = v1391;	// L1661
    int16_t v1394 = v1392;	// L1662
    int16_t v1395 = v1393 * v1394;	// L1663
    int32_t v1396 = v53;	// L1664
    ap_int<33> v1397 = v1396;	// L1665
    ap_int<33> v1398 = v1395;	// L1666
    ap_int<33> v1399 = v1397 + v1398;	// L1667
    int32_t v1400 = v1399;	// L1668
    v53 = v1400;	// L1669
    int8_t v1401 = a53;	// L1670
    v1380.write(v1401); // v1380[k53] = v1401;	// L1671
    int8_t v1402 = b53;	// L1672
    v1381.write(v1402); // v1381[k53] = v1402;	// L1673
  }
  int32_t v1403 = v53;	// L1675
  v1382[v1383][v1384] = v1403;	// L1676
}

void PE_kernel_gemm_6_6(
  hls::stream< int8_t > &v1404 /* v1404[8] */,
  hls::stream< int8_t > &v1405 /* v1405[8] */,
  hls::stream< int8_t > &v1406 /* v1406[8] */,
  hls::stream< int8_t > &v1407 /* v1407[8] */,
  int32_t v1408[8][8],
  int v1409,
  int v1410
) {	// L1679
  #pragma HLS stream variable=v1404 depth=9
  #pragma HLS stream variable=v1405 depth=9
  #pragma HLS stream variable=v1406 depth=9
  #pragma HLS stream variable=v1407 depth=9
  #pragma HLS array_partition variable=v1408 complete dim=1
  #pragma HLS array_partition variable=v1408 complete dim=2

  int32_t v54;	// L1681
  v54 = 0;	// L1682
  l_reduction_k54: for (int k54 = 0; k54 < 8; k54++) {	// L1683
  #pragma HLS pipeline II=1
    int8_t v1413 = v1404.read(); // v1404[k54];	// L1684
    int8_t a54;	// L1685
    a54 = v1413;	// L1686
    int8_t v1415 = v1405.read(); // v1405[k54];	// L1687
    int8_t b54;	// L1688
    b54 = v1415;	// L1689
    int8_t v1417 = a54;	// L1690
    int8_t v1418 = b54;	// L1691
    int16_t v1419 = v1417;	// L1692
    int16_t v1420 = v1418;	// L1693
    int16_t v1421 = v1419 * v1420;	// L1694
    int32_t v1422 = v54;	// L1695
    ap_int<33> v1423 = v1422;	// L1696
    ap_int<33> v1424 = v1421;	// L1697
    ap_int<33> v1425 = v1423 + v1424;	// L1698
    int32_t v1426 = v1425;	// L1699
    v54 = v1426;	// L1700
    int8_t v1427 = a54;	// L1701
    v1406.write(v1427); // v1406[k54] = v1427;	// L1702
    int8_t v1428 = b54;	// L1703
    v1407.write(v1428); // v1407[k54] = v1428;	// L1704
  }
  int32_t v1429 = v54;	// L1706
  v1408[v1409][v1410] = v1429;	// L1707
}

void PE_kernel_gemm_7_6(
  hls::stream< int8_t > &v1430 /* v1430[8] */,
  hls::stream< int8_t > &v1431 /* v1431[8] */,
  hls::stream< int8_t > &v1432 /* v1432[8] */,
  hls::stream< int8_t > &v1433 /* v1433[8] */,
  int32_t v1434[8][8],
  int v1435,
  int v1436
) {	// L1710
  #pragma HLS stream variable=v1430 depth=9
  #pragma HLS stream variable=v1431 depth=9
  #pragma HLS stream variable=v1432 depth=9
  #pragma HLS stream variable=v1433 depth=9
  #pragma HLS array_partition variable=v1434 complete dim=1
  #pragma HLS array_partition variable=v1434 complete dim=2

  int32_t v55;	// L1712
  v55 = 0;	// L1713
  l_reduction_k55: for (int k55 = 0; k55 < 8; k55++) {	// L1714
  #pragma HLS pipeline II=1
    int8_t v1439 = v1430.read(); // v1430[k55];	// L1715
    int8_t a55;	// L1716
    a55 = v1439;	// L1717
    int8_t v1441 = v1431.read(); // v1431[k55];	// L1718
    int8_t b55;	// L1719
    b55 = v1441;	// L1720
    int8_t v1443 = a55;	// L1721
    int8_t v1444 = b55;	// L1722
    int16_t v1445 = v1443;	// L1723
    int16_t v1446 = v1444;	// L1724
    int16_t v1447 = v1445 * v1446;	// L1725
    int32_t v1448 = v55;	// L1726
    ap_int<33> v1449 = v1448;	// L1727
    ap_int<33> v1450 = v1447;	// L1728
    ap_int<33> v1451 = v1449 + v1450;	// L1729
    int32_t v1452 = v1451;	// L1730
    v55 = v1452;	// L1731
    int8_t v1453 = a55;	// L1732
    v1432.write(v1453); // v1432[k55] = v1453;	// L1733
    int8_t v1454 = b55;	// L1734
    v1433.write(v1454); // v1433[k55] = v1454;	// L1735
  }
  int32_t v1455 = v55;	// L1737
  v1434[v1435][v1436] = v1455;	// L1738
}

void PE_kernel_gemm_0_7(
  hls::stream< int8_t > &v1456 /* v1456[8] */,
  hls::stream< int8_t > &v1457 /* v1457[8] */,
  hls::stream< int8_t > &v1458 /* v1458[8] */,
  hls::stream< int8_t > &v1459 /* v1459[8] */,
  int32_t v1460[8][8],
  int v1461,
  int v1462
) {	// L1741
  #pragma HLS stream variable=v1456 depth=9
  #pragma HLS stream variable=v1457 depth=9
  #pragma HLS stream variable=v1458 depth=9
  #pragma HLS stream variable=v1459 depth=9
  #pragma HLS array_partition variable=v1460 complete dim=1
  #pragma HLS array_partition variable=v1460 complete dim=2

  int32_t v56;	// L1743
  v56 = 0;	// L1744
  l_reduction_k56: for (int k56 = 0; k56 < 8; k56++) {	// L1745
  #pragma HLS pipeline II=1
    int8_t v1465 = v1456.read(); // v1456[k56];	// L1746
    int8_t a56;	// L1747
    a56 = v1465;	// L1748
    int8_t v1467 = v1457.read(); // v1457[k56];	// L1749
    int8_t b56;	// L1750
    b56 = v1467;	// L1751
    int8_t v1469 = a56;	// L1752
    int8_t v1470 = b56;	// L1753
    int16_t v1471 = v1469;	// L1754
    int16_t v1472 = v1470;	// L1755
    int16_t v1473 = v1471 * v1472;	// L1756
    int32_t v1474 = v56;	// L1757
    ap_int<33> v1475 = v1474;	// L1758
    ap_int<33> v1476 = v1473;	// L1759
    ap_int<33> v1477 = v1475 + v1476;	// L1760
    int32_t v1478 = v1477;	// L1761
    v56 = v1478;	// L1762
    int8_t v1479 = a56;	// L1763
    v1458.write(v1479); // v1458[k56] = v1479;	// L1764
    int8_t v1480 = b56;	// L1765
    v1459.write(v1480); // v1459[k56] = v1480;	// L1766
  }
  int32_t v1481 = v56;	// L1768
  v1460[v1461][v1462] = v1481;	// L1769
}

void PE_kernel_gemm_1_7(
  hls::stream< int8_t > &v1482 /* v1482[8] */,
  hls::stream< int8_t > &v1483 /* v1483[8] */,
  hls::stream< int8_t > &v1484 /* v1484[8] */,
  hls::stream< int8_t > &v1485 /* v1485[8] */,
  int32_t v1486[8][8],
  int v1487,
  int v1488
) {	// L1772
  #pragma HLS stream variable=v1482 depth=9
  #pragma HLS stream variable=v1483 depth=9
  #pragma HLS stream variable=v1484 depth=9
  #pragma HLS stream variable=v1485 depth=9
  #pragma HLS array_partition variable=v1486 complete dim=1
  #pragma HLS array_partition variable=v1486 complete dim=2

  int32_t v57;	// L1774
  v57 = 0;	// L1775
  l_reduction_k57: for (int k57 = 0; k57 < 8; k57++) {	// L1776
  #pragma HLS pipeline II=1
    int8_t v1491 = v1482.read(); // v1482[k57];	// L1777
    int8_t a57;	// L1778
    a57 = v1491;	// L1779
    int8_t v1493 = v1483.read(); // v1483[k57];	// L1780
    int8_t b57;	// L1781
    b57 = v1493;	// L1782
    int8_t v1495 = a57;	// L1783
    int8_t v1496 = b57;	// L1784
    int16_t v1497 = v1495;	// L1785
    int16_t v1498 = v1496;	// L1786
    int16_t v1499 = v1497 * v1498;	// L1787
    int32_t v1500 = v57;	// L1788
    ap_int<33> v1501 = v1500;	// L1789
    ap_int<33> v1502 = v1499;	// L1790
    ap_int<33> v1503 = v1501 + v1502;	// L1791
    int32_t v1504 = v1503;	// L1792
    v57 = v1504;	// L1793
    int8_t v1505 = a57;	// L1794
    v1484.write(v1505); // v1484[k57] = v1505;	// L1795
    int8_t v1506 = b57;	// L1796
    v1485.write(v1506); // v1485[k57] = v1506;	// L1797
  }
  int32_t v1507 = v57;	// L1799
  v1486[v1487][v1488] = v1507;	// L1800
}

void PE_kernel_gemm_2_7(
  hls::stream< int8_t > &v1508 /* v1508[8] */,
  hls::stream< int8_t > &v1509 /* v1509[8] */,
  hls::stream< int8_t > &v1510 /* v1510[8] */,
  hls::stream< int8_t > &v1511 /* v1511[8] */,
  int32_t v1512[8][8],
  int v1513,
  int v1514
) {	// L1803
  #pragma HLS stream variable=v1508 depth=9
  #pragma HLS stream variable=v1509 depth=9
  #pragma HLS stream variable=v1510 depth=9
  #pragma HLS stream variable=v1511 depth=9
  #pragma HLS array_partition variable=v1512 complete dim=1
  #pragma HLS array_partition variable=v1512 complete dim=2

  int32_t v58;	// L1805
  v58 = 0;	// L1806
  l_reduction_k58: for (int k58 = 0; k58 < 8; k58++) {	// L1807
  #pragma HLS pipeline II=1
    int8_t v1517 = v1508.read(); // v1508[k58];	// L1808
    int8_t a58;	// L1809
    a58 = v1517;	// L1810
    int8_t v1519 = v1509.read(); // v1509[k58];	// L1811
    int8_t b58;	// L1812
    b58 = v1519;	// L1813
    int8_t v1521 = a58;	// L1814
    int8_t v1522 = b58;	// L1815
    int16_t v1523 = v1521;	// L1816
    int16_t v1524 = v1522;	// L1817
    int16_t v1525 = v1523 * v1524;	// L1818
    int32_t v1526 = v58;	// L1819
    ap_int<33> v1527 = v1526;	// L1820
    ap_int<33> v1528 = v1525;	// L1821
    ap_int<33> v1529 = v1527 + v1528;	// L1822
    int32_t v1530 = v1529;	// L1823
    v58 = v1530;	// L1824
    int8_t v1531 = a58;	// L1825
    v1510.write(v1531); // v1510[k58] = v1531;	// L1826
    int8_t v1532 = b58;	// L1827
    v1511.write(v1532); // v1511[k58] = v1532;	// L1828
  }
  int32_t v1533 = v58;	// L1830
  v1512[v1513][v1514] = v1533;	// L1831
}

void PE_kernel_gemm_3_7(
  hls::stream< int8_t > &v1534 /* v1534[8] */,
  hls::stream< int8_t > &v1535 /* v1535[8] */,
  hls::stream< int8_t > &v1536 /* v1536[8] */,
  hls::stream< int8_t > &v1537 /* v1537[8] */,
  int32_t v1538[8][8],
  int v1539,
  int v1540
) {	// L1834
  #pragma HLS stream variable=v1534 depth=9
  #pragma HLS stream variable=v1535 depth=9
  #pragma HLS stream variable=v1536 depth=9
  #pragma HLS stream variable=v1537 depth=9
  #pragma HLS array_partition variable=v1538 complete dim=1
  #pragma HLS array_partition variable=v1538 complete dim=2

  int32_t v59;	// L1836
  v59 = 0;	// L1837
  l_reduction_k59: for (int k59 = 0; k59 < 8; k59++) {	// L1838
  #pragma HLS pipeline II=1
    int8_t v1543 = v1534.read(); // v1534[k59];	// L1839
    int8_t a59;	// L1840
    a59 = v1543;	// L1841
    int8_t v1545 = v1535.read(); // v1535[k59];	// L1842
    int8_t b59;	// L1843
    b59 = v1545;	// L1844
    int8_t v1547 = a59;	// L1845
    int8_t v1548 = b59;	// L1846
    int16_t v1549 = v1547;	// L1847
    int16_t v1550 = v1548;	// L1848
    int16_t v1551 = v1549 * v1550;	// L1849
    int32_t v1552 = v59;	// L1850
    ap_int<33> v1553 = v1552;	// L1851
    ap_int<33> v1554 = v1551;	// L1852
    ap_int<33> v1555 = v1553 + v1554;	// L1853
    int32_t v1556 = v1555;	// L1854
    v59 = v1556;	// L1855
    int8_t v1557 = a59;	// L1856
    v1536.write(v1557); // v1536[k59] = v1557;	// L1857
    int8_t v1558 = b59;	// L1858
    v1537.write(v1558); // v1537[k59] = v1558;	// L1859
  }
  int32_t v1559 = v59;	// L1861
  v1538[v1539][v1540] = v1559;	// L1862
}

void PE_kernel_gemm_4_7(
  hls::stream< int8_t > &v1560 /* v1560[8] */,
  hls::stream< int8_t > &v1561 /* v1561[8] */,
  hls::stream< int8_t > &v1562 /* v1562[8] */,
  hls::stream< int8_t > &v1563 /* v1563[8] */,
  int32_t v1564[8][8],
  int v1565,
  int v1566
) {	// L1865
  #pragma HLS stream variable=v1560 depth=9
  #pragma HLS stream variable=v1561 depth=9
  #pragma HLS stream variable=v1562 depth=9
  #pragma HLS stream variable=v1563 depth=9
  #pragma HLS array_partition variable=v1564 complete dim=1
  #pragma HLS array_partition variable=v1564 complete dim=2

  int32_t v60;	// L1867
  v60 = 0;	// L1868
  l_reduction_k60: for (int k60 = 0; k60 < 8; k60++) {	// L1869
  #pragma HLS pipeline II=1
    int8_t v1569 = v1560.read(); // v1560[k60];	// L1870
    int8_t a60;	// L1871
    a60 = v1569;	// L1872
    int8_t v1571 = v1561.read(); // v1561[k60];	// L1873
    int8_t b60;	// L1874
    b60 = v1571;	// L1875
    int8_t v1573 = a60;	// L1876
    int8_t v1574 = b60;	// L1877
    int16_t v1575 = v1573;	// L1878
    int16_t v1576 = v1574;	// L1879
    int16_t v1577 = v1575 * v1576;	// L1880
    int32_t v1578 = v60;	// L1881
    ap_int<33> v1579 = v1578;	// L1882
    ap_int<33> v1580 = v1577;	// L1883
    ap_int<33> v1581 = v1579 + v1580;	// L1884
    int32_t v1582 = v1581;	// L1885
    v60 = v1582;	// L1886
    int8_t v1583 = a60;	// L1887
    v1562.write(v1583); // v1562[k60] = v1583;	// L1888
    int8_t v1584 = b60;	// L1889
    v1563.write(v1584); // v1563[k60] = v1584;	// L1890
  }
  int32_t v1585 = v60;	// L1892
  v1564[v1565][v1566] = v1585;	// L1893
}

void PE_kernel_gemm_5_7(
  hls::stream< int8_t > &v1586 /* v1586[8] */,
  hls::stream< int8_t > &v1587 /* v1587[8] */,
  hls::stream< int8_t > &v1588 /* v1588[8] */,
  hls::stream< int8_t > &v1589 /* v1589[8] */,
  int32_t v1590[8][8],
  int v1591,
  int v1592
) {	// L1896
  #pragma HLS stream variable=v1586 depth=9
  #pragma HLS stream variable=v1587 depth=9
  #pragma HLS stream variable=v1588 depth=9
  #pragma HLS stream variable=v1589 depth=9
  #pragma HLS array_partition variable=v1590 complete dim=1
  #pragma HLS array_partition variable=v1590 complete dim=2

  int32_t v61;	// L1898
  v61 = 0;	// L1899
  l_reduction_k61: for (int k61 = 0; k61 < 8; k61++) {	// L1900
  #pragma HLS pipeline II=1
    int8_t v1595 = v1586.read(); // v1586[k61];	// L1901
    int8_t a61;	// L1902
    a61 = v1595;	// L1903
    int8_t v1597 = v1587.read(); // v1587[k61];	// L1904
    int8_t b61;	// L1905
    b61 = v1597;	// L1906
    int8_t v1599 = a61;	// L1907
    int8_t v1600 = b61;	// L1908
    int16_t v1601 = v1599;	// L1909
    int16_t v1602 = v1600;	// L1910
    int16_t v1603 = v1601 * v1602;	// L1911
    int32_t v1604 = v61;	// L1912
    ap_int<33> v1605 = v1604;	// L1913
    ap_int<33> v1606 = v1603;	// L1914
    ap_int<33> v1607 = v1605 + v1606;	// L1915
    int32_t v1608 = v1607;	// L1916
    v61 = v1608;	// L1917
    int8_t v1609 = a61;	// L1918
    v1588.write(v1609); // v1588[k61] = v1609;	// L1919
    int8_t v1610 = b61;	// L1920
    v1589.write(v1610); // v1589[k61] = v1610;	// L1921
  }
  int32_t v1611 = v61;	// L1923
  v1590[v1591][v1592] = v1611;	// L1924
}

void PE_kernel_gemm_6_7(
  hls::stream< int8_t > &v1612 /* v1612[8] */,
  hls::stream< int8_t > &v1613 /* v1613[8] */,
  hls::stream< int8_t > &v1614 /* v1614[8] */,
  hls::stream< int8_t > &v1615 /* v1615[8] */,
  int32_t v1616[8][8],
  int v1617,
  int v1618
) {	// L1927
  #pragma HLS stream variable=v1612 depth=9
  #pragma HLS stream variable=v1613 depth=9
  #pragma HLS stream variable=v1614 depth=9
  #pragma HLS stream variable=v1615 depth=9
  #pragma HLS array_partition variable=v1616 complete dim=1
  #pragma HLS array_partition variable=v1616 complete dim=2

  int32_t v62;	// L1929
  v62 = 0;	// L1930
  l_reduction_k62: for (int k62 = 0; k62 < 8; k62++) {	// L1931
  #pragma HLS pipeline II=1
    int8_t v1621 = v1612.read(); // v1612[k62];	// L1932
    int8_t a62;	// L1933
    a62 = v1621;	// L1934
    int8_t v1623 = v1613.read(); // v1613[k62];	// L1935
    int8_t b62;	// L1936
    b62 = v1623;	// L1937
    int8_t v1625 = a62;	// L1938
    int8_t v1626 = b62;	// L1939
    int16_t v1627 = v1625;	// L1940
    int16_t v1628 = v1626;	// L1941
    int16_t v1629 = v1627 * v1628;	// L1942
    int32_t v1630 = v62;	// L1943
    ap_int<33> v1631 = v1630;	// L1944
    ap_int<33> v1632 = v1629;	// L1945
    ap_int<33> v1633 = v1631 + v1632;	// L1946
    int32_t v1634 = v1633;	// L1947
    v62 = v1634;	// L1948
    int8_t v1635 = a62;	// L1949
    v1614.write(v1635); // v1614[k62] = v1635;	// L1950
    int8_t v1636 = b62;	// L1951
    v1615.write(v1636); // v1615[k62] = v1636;	// L1952
  }
  int32_t v1637 = v62;	// L1954
  v1616[v1617][v1618] = v1637;	// L1955
}

void PE_kernel_gemm_7_7(
  hls::stream< int8_t > &v1638 /* v1638[8] */,
  hls::stream< int8_t > &v1639 /* v1639[8] */,
  hls::stream< int8_t > &v1640 /* v1640[8] */,
  hls::stream< int8_t > &v1641 /* v1641[8] */,
  int32_t v1642[8][8],
  int v1643,
  int v1644
) {	// L1958
  #pragma HLS stream variable=v1638 depth=9
  #pragma HLS stream variable=v1639 depth=9
  #pragma HLS stream variable=v1640 depth=9
  #pragma HLS stream variable=v1641 depth=9
  #pragma HLS array_partition variable=v1642 complete dim=1
  #pragma HLS array_partition variable=v1642 complete dim=2

  int32_t v63;	// L1960
  v63 = 0;	// L1961
  l_reduction_k63: for (int k63 = 0; k63 < 8; k63++) {	// L1962
  #pragma HLS pipeline II=1
    int8_t v1647 = v1638.read(); // v1638[k63];	// L1963
    int8_t a63;	// L1964
    a63 = v1647;	// L1965
    int8_t v1649 = v1639.read(); // v1639[k63];	// L1966
    int8_t b63;	// L1967
    b63 = v1649;	// L1968
    int8_t v1651 = a63;	// L1969
    int8_t v1652 = b63;	// L1970
    int16_t v1653 = v1651;	// L1971
    int16_t v1654 = v1652;	// L1972
    int16_t v1655 = v1653 * v1654;	// L1973
    int32_t v1656 = v63;	// L1974
    ap_int<33> v1657 = v1656;	// L1975
    ap_int<33> v1658 = v1655;	// L1976
    ap_int<33> v1659 = v1657 + v1658;	// L1977
    int32_t v1660 = v1659;	// L1978
    v63 = v1660;	// L1979
    int8_t v1661 = a63;	// L1980
    v1640.write(v1661); // v1640[k63] = v1661;	// L1981
    int8_t v1662 = b63;	// L1982
    v1641.write(v1662); // v1641[k63] = v1662;	// L1983
  }
  int32_t v1663 = v63;	// L1985
  v1642[v1643][v1644] = v1663;	// L1986
}

void systolic_tile_gemm(
  int8_t v1664[8][8],
  int8_t v1665[8][8],
  int32_t v1666[8][8]
) {	// L1989
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v1664 complete dim=1

  #pragma HLS array_partition variable=v1665 complete dim=2

  #pragma HLS array_partition variable=v1666 complete dim=1
  #pragma HLS array_partition variable=v1666 complete dim=2

  hls::stream< int8_t > A_fifo[8][9] /* A_fifo[8][9][8] */;	// L1990
  #pragma HLS stream variable=A_fifo depth=9
  hls::stream< int8_t > B_fifo[8][9] /* B_fifo[8][9][8] */;	// L1991
  #pragma HLS stream variable=B_fifo depth=9
  int8_t A_drain[8];	// L1992
  int8_t B_drain[8];	// L1993
  l_data_load_k64: for (int k64 = 0; k64 < 8; k64++) {	// L1994
    l_S_m_0_m: for (int m = 0; m < 8; m++) {	// L1995
      int8_t v1673 = v1664[m][k64];	// L1996
      A_fifo[m][0].write(v1673); // A_fifo[m][0][k64] = v1673;	// L1997
    }
    l_S_n_1_n: for (int n = 0; n < 8; n++) {	// L1999
      int8_t v1675 = v1665[k64][n];	// L2000
      B_fifo[n][0].write(v1675); // B_fifo[n][0][k64] = v1675;	// L2001
    }
  }
  hls::stream< int8_t > &v1676 /* v1676[8] */ = A_fifo[0][0];	// L2005
  hls::stream< int8_t > &v1677 /* v1677[8] */ = B_fifo[0][0];	// L2006
  hls::stream< int8_t > &v1678 /* v1678[8] */ = A_fifo[0][1];	// L2012
  hls::stream< int8_t > &v1679 /* v1679[8] */ = B_fifo[0][1];	// L2013
  PE_kernel_gemm_0_0(v1676, v1677, v1678, v1679, v1666, 0, 0);	// L2014
  hls::stream< int8_t > &v1680 /* v1680[8] */ = A_fifo[0][1];	// L2016
  hls::stream< int8_t > &v1681 /* v1681[8] */ = B_fifo[1][0];	// L2017
  hls::stream< int8_t > &v1682 /* v1682[8] */ = A_fifo[0][2];	// L2021
  hls::stream< int8_t > &v1683 /* v1683[8] */ = B_fifo[1][1];	// L2022
  PE_kernel_gemm_1_0(v1680, v1681, v1682, v1683, v1666, 0, 1);	// L2023
  hls::stream< int8_t > &v1684 /* v1684[8] */ = A_fifo[0][2];	// L2025
  hls::stream< int8_t > &v1685 /* v1685[8] */ = B_fifo[2][0];	// L2026
  hls::stream< int8_t > &v1686 /* v1686[8] */ = A_fifo[0][3];	// L2030
  hls::stream< int8_t > &v1687 /* v1687[8] */ = B_fifo[2][1];	// L2031
  PE_kernel_gemm_2_0(v1684, v1685, v1686, v1687, v1666, 0, 2);	// L2032
  hls::stream< int8_t > &v1688 /* v1688[8] */ = A_fifo[0][3];	// L2034
  hls::stream< int8_t > &v1689 /* v1689[8] */ = B_fifo[3][0];	// L2035
  hls::stream< int8_t > &v1690 /* v1690[8] */ = A_fifo[0][4];	// L2039
  hls::stream< int8_t > &v1691 /* v1691[8] */ = B_fifo[3][1];	// L2040
  PE_kernel_gemm_3_0(v1688, v1689, v1690, v1691, v1666, 0, 3);	// L2041
  hls::stream< int8_t > &v1692 /* v1692[8] */ = A_fifo[0][4];	// L2043
  hls::stream< int8_t > &v1693 /* v1693[8] */ = B_fifo[4][0];	// L2044
  hls::stream< int8_t > &v1694 /* v1694[8] */ = A_fifo[0][5];	// L2048
  hls::stream< int8_t > &v1695 /* v1695[8] */ = B_fifo[4][1];	// L2049
  PE_kernel_gemm_4_0(v1692, v1693, v1694, v1695, v1666, 0, 4);	// L2050
  hls::stream< int8_t > &v1696 /* v1696[8] */ = A_fifo[0][5];	// L2052
  hls::stream< int8_t > &v1697 /* v1697[8] */ = B_fifo[5][0];	// L2053
  hls::stream< int8_t > &v1698 /* v1698[8] */ = A_fifo[0][6];	// L2057
  hls::stream< int8_t > &v1699 /* v1699[8] */ = B_fifo[5][1];	// L2058
  PE_kernel_gemm_5_0(v1696, v1697, v1698, v1699, v1666, 0, 5);	// L2059
  hls::stream< int8_t > &v1700 /* v1700[8] */ = A_fifo[0][6];	// L2061
  hls::stream< int8_t > &v1701 /* v1701[8] */ = B_fifo[6][0];	// L2062
  hls::stream< int8_t > &v1702 /* v1702[8] */ = A_fifo[0][7];	// L2066
  hls::stream< int8_t > &v1703 /* v1703[8] */ = B_fifo[6][1];	// L2067
  PE_kernel_gemm_6_0(v1700, v1701, v1702, v1703, v1666, 0, 6);	// L2068
  hls::stream< int8_t > &v1704 /* v1704[8] */ = A_fifo[0][7];	// L2070
  hls::stream< int8_t > &v1705 /* v1705[8] */ = B_fifo[7][0];	// L2071
  hls::stream< int8_t > &v1706 /* v1706[8] */ = A_fifo[0][8];	// L2075
  hls::stream< int8_t > &v1707 /* v1707[8] */ = B_fifo[7][1];	// L2076
  PE_kernel_gemm_7_0(v1704, v1705, v1706, v1707, v1666, 0, 7);	// L2077
  hls::stream< int8_t > &v1708 /* v1708[8] */ = A_fifo[1][0];	// L2078
  hls::stream< int8_t > &v1709 /* v1709[8] */ = B_fifo[0][1];	// L2079
  hls::stream< int8_t > &v1710 /* v1710[8] */ = A_fifo[1][1];	// L2080
  hls::stream< int8_t > &v1711 /* v1711[8] */ = B_fifo[0][2];	// L2081
  PE_kernel_gemm_0_1(v1708, v1709, v1710, v1711, v1666, 1, 0);	// L2082
  hls::stream< int8_t > &v1712 /* v1712[8] */ = A_fifo[1][1];	// L2083
  hls::stream< int8_t > &v1713 /* v1713[8] */ = B_fifo[1][1];	// L2084
  hls::stream< int8_t > &v1714 /* v1714[8] */ = A_fifo[1][2];	// L2085
  hls::stream< int8_t > &v1715 /* v1715[8] */ = B_fifo[1][2];	// L2086
  PE_kernel_gemm_1_1(v1712, v1713, v1714, v1715, v1666, 1, 1);	// L2087
  hls::stream< int8_t > &v1716 /* v1716[8] */ = A_fifo[1][2];	// L2088
  hls::stream< int8_t > &v1717 /* v1717[8] */ = B_fifo[2][1];	// L2089
  hls::stream< int8_t > &v1718 /* v1718[8] */ = A_fifo[1][3];	// L2090
  hls::stream< int8_t > &v1719 /* v1719[8] */ = B_fifo[2][2];	// L2091
  PE_kernel_gemm_2_1(v1716, v1717, v1718, v1719, v1666, 1, 2);	// L2092
  hls::stream< int8_t > &v1720 /* v1720[8] */ = A_fifo[1][3];	// L2093
  hls::stream< int8_t > &v1721 /* v1721[8] */ = B_fifo[3][1];	// L2094
  hls::stream< int8_t > &v1722 /* v1722[8] */ = A_fifo[1][4];	// L2095
  hls::stream< int8_t > &v1723 /* v1723[8] */ = B_fifo[3][2];	// L2096
  PE_kernel_gemm_3_1(v1720, v1721, v1722, v1723, v1666, 1, 3);	// L2097
  hls::stream< int8_t > &v1724 /* v1724[8] */ = A_fifo[1][4];	// L2098
  hls::stream< int8_t > &v1725 /* v1725[8] */ = B_fifo[4][1];	// L2099
  hls::stream< int8_t > &v1726 /* v1726[8] */ = A_fifo[1][5];	// L2100
  hls::stream< int8_t > &v1727 /* v1727[8] */ = B_fifo[4][2];	// L2101
  PE_kernel_gemm_4_1(v1724, v1725, v1726, v1727, v1666, 1, 4);	// L2102
  hls::stream< int8_t > &v1728 /* v1728[8] */ = A_fifo[1][5];	// L2103
  hls::stream< int8_t > &v1729 /* v1729[8] */ = B_fifo[5][1];	// L2104
  hls::stream< int8_t > &v1730 /* v1730[8] */ = A_fifo[1][6];	// L2105
  hls::stream< int8_t > &v1731 /* v1731[8] */ = B_fifo[5][2];	// L2106
  PE_kernel_gemm_5_1(v1728, v1729, v1730, v1731, v1666, 1, 5);	// L2107
  hls::stream< int8_t > &v1732 /* v1732[8] */ = A_fifo[1][6];	// L2108
  hls::stream< int8_t > &v1733 /* v1733[8] */ = B_fifo[6][1];	// L2109
  hls::stream< int8_t > &v1734 /* v1734[8] */ = A_fifo[1][7];	// L2110
  hls::stream< int8_t > &v1735 /* v1735[8] */ = B_fifo[6][2];	// L2111
  PE_kernel_gemm_6_1(v1732, v1733, v1734, v1735, v1666, 1, 6);	// L2112
  hls::stream< int8_t > &v1736 /* v1736[8] */ = A_fifo[1][7];	// L2113
  hls::stream< int8_t > &v1737 /* v1737[8] */ = B_fifo[7][1];	// L2114
  hls::stream< int8_t > &v1738 /* v1738[8] */ = A_fifo[1][8];	// L2115
  hls::stream< int8_t > &v1739 /* v1739[8] */ = B_fifo[7][2];	// L2116
  PE_kernel_gemm_7_1(v1736, v1737, v1738, v1739, v1666, 1, 7);	// L2117
  hls::stream< int8_t > &v1740 /* v1740[8] */ = A_fifo[2][0];	// L2118
  hls::stream< int8_t > &v1741 /* v1741[8] */ = B_fifo[0][2];	// L2119
  hls::stream< int8_t > &v1742 /* v1742[8] */ = A_fifo[2][1];	// L2120
  hls::stream< int8_t > &v1743 /* v1743[8] */ = B_fifo[0][3];	// L2121
  PE_kernel_gemm_0_2(v1740, v1741, v1742, v1743, v1666, 2, 0);	// L2122
  hls::stream< int8_t > &v1744 /* v1744[8] */ = A_fifo[2][1];	// L2123
  hls::stream< int8_t > &v1745 /* v1745[8] */ = B_fifo[1][2];	// L2124
  hls::stream< int8_t > &v1746 /* v1746[8] */ = A_fifo[2][2];	// L2125
  hls::stream< int8_t > &v1747 /* v1747[8] */ = B_fifo[1][3];	// L2126
  PE_kernel_gemm_1_2(v1744, v1745, v1746, v1747, v1666, 2, 1);	// L2127
  hls::stream< int8_t > &v1748 /* v1748[8] */ = A_fifo[2][2];	// L2128
  hls::stream< int8_t > &v1749 /* v1749[8] */ = B_fifo[2][2];	// L2129
  hls::stream< int8_t > &v1750 /* v1750[8] */ = A_fifo[2][3];	// L2130
  hls::stream< int8_t > &v1751 /* v1751[8] */ = B_fifo[2][3];	// L2131
  PE_kernel_gemm_2_2(v1748, v1749, v1750, v1751, v1666, 2, 2);	// L2132
  hls::stream< int8_t > &v1752 /* v1752[8] */ = A_fifo[2][3];	// L2133
  hls::stream< int8_t > &v1753 /* v1753[8] */ = B_fifo[3][2];	// L2134
  hls::stream< int8_t > &v1754 /* v1754[8] */ = A_fifo[2][4];	// L2135
  hls::stream< int8_t > &v1755 /* v1755[8] */ = B_fifo[3][3];	// L2136
  PE_kernel_gemm_3_2(v1752, v1753, v1754, v1755, v1666, 2, 3);	// L2137
  hls::stream< int8_t > &v1756 /* v1756[8] */ = A_fifo[2][4];	// L2138
  hls::stream< int8_t > &v1757 /* v1757[8] */ = B_fifo[4][2];	// L2139
  hls::stream< int8_t > &v1758 /* v1758[8] */ = A_fifo[2][5];	// L2140
  hls::stream< int8_t > &v1759 /* v1759[8] */ = B_fifo[4][3];	// L2141
  PE_kernel_gemm_4_2(v1756, v1757, v1758, v1759, v1666, 2, 4);	// L2142
  hls::stream< int8_t > &v1760 /* v1760[8] */ = A_fifo[2][5];	// L2143
  hls::stream< int8_t > &v1761 /* v1761[8] */ = B_fifo[5][2];	// L2144
  hls::stream< int8_t > &v1762 /* v1762[8] */ = A_fifo[2][6];	// L2145
  hls::stream< int8_t > &v1763 /* v1763[8] */ = B_fifo[5][3];	// L2146
  PE_kernel_gemm_5_2(v1760, v1761, v1762, v1763, v1666, 2, 5);	// L2147
  hls::stream< int8_t > &v1764 /* v1764[8] */ = A_fifo[2][6];	// L2148
  hls::stream< int8_t > &v1765 /* v1765[8] */ = B_fifo[6][2];	// L2149
  hls::stream< int8_t > &v1766 /* v1766[8] */ = A_fifo[2][7];	// L2150
  hls::stream< int8_t > &v1767 /* v1767[8] */ = B_fifo[6][3];	// L2151
  PE_kernel_gemm_6_2(v1764, v1765, v1766, v1767, v1666, 2, 6);	// L2152
  hls::stream< int8_t > &v1768 /* v1768[8] */ = A_fifo[2][7];	// L2153
  hls::stream< int8_t > &v1769 /* v1769[8] */ = B_fifo[7][2];	// L2154
  hls::stream< int8_t > &v1770 /* v1770[8] */ = A_fifo[2][8];	// L2155
  hls::stream< int8_t > &v1771 /* v1771[8] */ = B_fifo[7][3];	// L2156
  PE_kernel_gemm_7_2(v1768, v1769, v1770, v1771, v1666, 2, 7);	// L2157
  hls::stream< int8_t > &v1772 /* v1772[8] */ = A_fifo[3][0];	// L2158
  hls::stream< int8_t > &v1773 /* v1773[8] */ = B_fifo[0][3];	// L2159
  hls::stream< int8_t > &v1774 /* v1774[8] */ = A_fifo[3][1];	// L2160
  hls::stream< int8_t > &v1775 /* v1775[8] */ = B_fifo[0][4];	// L2161
  PE_kernel_gemm_0_3(v1772, v1773, v1774, v1775, v1666, 3, 0);	// L2162
  hls::stream< int8_t > &v1776 /* v1776[8] */ = A_fifo[3][1];	// L2163
  hls::stream< int8_t > &v1777 /* v1777[8] */ = B_fifo[1][3];	// L2164
  hls::stream< int8_t > &v1778 /* v1778[8] */ = A_fifo[3][2];	// L2165
  hls::stream< int8_t > &v1779 /* v1779[8] */ = B_fifo[1][4];	// L2166
  PE_kernel_gemm_1_3(v1776, v1777, v1778, v1779, v1666, 3, 1);	// L2167
  hls::stream< int8_t > &v1780 /* v1780[8] */ = A_fifo[3][2];	// L2168
  hls::stream< int8_t > &v1781 /* v1781[8] */ = B_fifo[2][3];	// L2169
  hls::stream< int8_t > &v1782 /* v1782[8] */ = A_fifo[3][3];	// L2170
  hls::stream< int8_t > &v1783 /* v1783[8] */ = B_fifo[2][4];	// L2171
  PE_kernel_gemm_2_3(v1780, v1781, v1782, v1783, v1666, 3, 2);	// L2172
  hls::stream< int8_t > &v1784 /* v1784[8] */ = A_fifo[3][3];	// L2173
  hls::stream< int8_t > &v1785 /* v1785[8] */ = B_fifo[3][3];	// L2174
  hls::stream< int8_t > &v1786 /* v1786[8] */ = A_fifo[3][4];	// L2175
  hls::stream< int8_t > &v1787 /* v1787[8] */ = B_fifo[3][4];	// L2176
  PE_kernel_gemm_3_3(v1784, v1785, v1786, v1787, v1666, 3, 3);	// L2177
  hls::stream< int8_t > &v1788 /* v1788[8] */ = A_fifo[3][4];	// L2178
  hls::stream< int8_t > &v1789 /* v1789[8] */ = B_fifo[4][3];	// L2179
  hls::stream< int8_t > &v1790 /* v1790[8] */ = A_fifo[3][5];	// L2180
  hls::stream< int8_t > &v1791 /* v1791[8] */ = B_fifo[4][4];	// L2181
  PE_kernel_gemm_4_3(v1788, v1789, v1790, v1791, v1666, 3, 4);	// L2182
  hls::stream< int8_t > &v1792 /* v1792[8] */ = A_fifo[3][5];	// L2183
  hls::stream< int8_t > &v1793 /* v1793[8] */ = B_fifo[5][3];	// L2184
  hls::stream< int8_t > &v1794 /* v1794[8] */ = A_fifo[3][6];	// L2185
  hls::stream< int8_t > &v1795 /* v1795[8] */ = B_fifo[5][4];	// L2186
  PE_kernel_gemm_5_3(v1792, v1793, v1794, v1795, v1666, 3, 5);	// L2187
  hls::stream< int8_t > &v1796 /* v1796[8] */ = A_fifo[3][6];	// L2188
  hls::stream< int8_t > &v1797 /* v1797[8] */ = B_fifo[6][3];	// L2189
  hls::stream< int8_t > &v1798 /* v1798[8] */ = A_fifo[3][7];	// L2190
  hls::stream< int8_t > &v1799 /* v1799[8] */ = B_fifo[6][4];	// L2191
  PE_kernel_gemm_6_3(v1796, v1797, v1798, v1799, v1666, 3, 6);	// L2192
  hls::stream< int8_t > &v1800 /* v1800[8] */ = A_fifo[3][7];	// L2193
  hls::stream< int8_t > &v1801 /* v1801[8] */ = B_fifo[7][3];	// L2194
  hls::stream< int8_t > &v1802 /* v1802[8] */ = A_fifo[3][8];	// L2195
  hls::stream< int8_t > &v1803 /* v1803[8] */ = B_fifo[7][4];	// L2196
  PE_kernel_gemm_7_3(v1800, v1801, v1802, v1803, v1666, 3, 7);	// L2197
  hls::stream< int8_t > &v1804 /* v1804[8] */ = A_fifo[4][0];	// L2198
  hls::stream< int8_t > &v1805 /* v1805[8] */ = B_fifo[0][4];	// L2199
  hls::stream< int8_t > &v1806 /* v1806[8] */ = A_fifo[4][1];	// L2200
  hls::stream< int8_t > &v1807 /* v1807[8] */ = B_fifo[0][5];	// L2201
  PE_kernel_gemm_0_4(v1804, v1805, v1806, v1807, v1666, 4, 0);	// L2202
  hls::stream< int8_t > &v1808 /* v1808[8] */ = A_fifo[4][1];	// L2203
  hls::stream< int8_t > &v1809 /* v1809[8] */ = B_fifo[1][4];	// L2204
  hls::stream< int8_t > &v1810 /* v1810[8] */ = A_fifo[4][2];	// L2205
  hls::stream< int8_t > &v1811 /* v1811[8] */ = B_fifo[1][5];	// L2206
  PE_kernel_gemm_1_4(v1808, v1809, v1810, v1811, v1666, 4, 1);	// L2207
  hls::stream< int8_t > &v1812 /* v1812[8] */ = A_fifo[4][2];	// L2208
  hls::stream< int8_t > &v1813 /* v1813[8] */ = B_fifo[2][4];	// L2209
  hls::stream< int8_t > &v1814 /* v1814[8] */ = A_fifo[4][3];	// L2210
  hls::stream< int8_t > &v1815 /* v1815[8] */ = B_fifo[2][5];	// L2211
  PE_kernel_gemm_2_4(v1812, v1813, v1814, v1815, v1666, 4, 2);	// L2212
  hls::stream< int8_t > &v1816 /* v1816[8] */ = A_fifo[4][3];	// L2213
  hls::stream< int8_t > &v1817 /* v1817[8] */ = B_fifo[3][4];	// L2214
  hls::stream< int8_t > &v1818 /* v1818[8] */ = A_fifo[4][4];	// L2215
  hls::stream< int8_t > &v1819 /* v1819[8] */ = B_fifo[3][5];	// L2216
  PE_kernel_gemm_3_4(v1816, v1817, v1818, v1819, v1666, 4, 3);	// L2217
  hls::stream< int8_t > &v1820 /* v1820[8] */ = A_fifo[4][4];	// L2218
  hls::stream< int8_t > &v1821 /* v1821[8] */ = B_fifo[4][4];	// L2219
  hls::stream< int8_t > &v1822 /* v1822[8] */ = A_fifo[4][5];	// L2220
  hls::stream< int8_t > &v1823 /* v1823[8] */ = B_fifo[4][5];	// L2221
  PE_kernel_gemm_4_4(v1820, v1821, v1822, v1823, v1666, 4, 4);	// L2222
  hls::stream< int8_t > &v1824 /* v1824[8] */ = A_fifo[4][5];	// L2223
  hls::stream< int8_t > &v1825 /* v1825[8] */ = B_fifo[5][4];	// L2224
  hls::stream< int8_t > &v1826 /* v1826[8] */ = A_fifo[4][6];	// L2225
  hls::stream< int8_t > &v1827 /* v1827[8] */ = B_fifo[5][5];	// L2226
  PE_kernel_gemm_5_4(v1824, v1825, v1826, v1827, v1666, 4, 5);	// L2227
  hls::stream< int8_t > &v1828 /* v1828[8] */ = A_fifo[4][6];	// L2228
  hls::stream< int8_t > &v1829 /* v1829[8] */ = B_fifo[6][4];	// L2229
  hls::stream< int8_t > &v1830 /* v1830[8] */ = A_fifo[4][7];	// L2230
  hls::stream< int8_t > &v1831 /* v1831[8] */ = B_fifo[6][5];	// L2231
  PE_kernel_gemm_6_4(v1828, v1829, v1830, v1831, v1666, 4, 6);	// L2232
  hls::stream< int8_t > &v1832 /* v1832[8] */ = A_fifo[4][7];	// L2233
  hls::stream< int8_t > &v1833 /* v1833[8] */ = B_fifo[7][4];	// L2234
  hls::stream< int8_t > &v1834 /* v1834[8] */ = A_fifo[4][8];	// L2235
  hls::stream< int8_t > &v1835 /* v1835[8] */ = B_fifo[7][5];	// L2236
  PE_kernel_gemm_7_4(v1832, v1833, v1834, v1835, v1666, 4, 7);	// L2237
  hls::stream< int8_t > &v1836 /* v1836[8] */ = A_fifo[5][0];	// L2238
  hls::stream< int8_t > &v1837 /* v1837[8] */ = B_fifo[0][5];	// L2239
  hls::stream< int8_t > &v1838 /* v1838[8] */ = A_fifo[5][1];	// L2240
  hls::stream< int8_t > &v1839 /* v1839[8] */ = B_fifo[0][6];	// L2241
  PE_kernel_gemm_0_5(v1836, v1837, v1838, v1839, v1666, 5, 0);	// L2242
  hls::stream< int8_t > &v1840 /* v1840[8] */ = A_fifo[5][1];	// L2243
  hls::stream< int8_t > &v1841 /* v1841[8] */ = B_fifo[1][5];	// L2244
  hls::stream< int8_t > &v1842 /* v1842[8] */ = A_fifo[5][2];	// L2245
  hls::stream< int8_t > &v1843 /* v1843[8] */ = B_fifo[1][6];	// L2246
  PE_kernel_gemm_1_5(v1840, v1841, v1842, v1843, v1666, 5, 1);	// L2247
  hls::stream< int8_t > &v1844 /* v1844[8] */ = A_fifo[5][2];	// L2248
  hls::stream< int8_t > &v1845 /* v1845[8] */ = B_fifo[2][5];	// L2249
  hls::stream< int8_t > &v1846 /* v1846[8] */ = A_fifo[5][3];	// L2250
  hls::stream< int8_t > &v1847 /* v1847[8] */ = B_fifo[2][6];	// L2251
  PE_kernel_gemm_2_5(v1844, v1845, v1846, v1847, v1666, 5, 2);	// L2252
  hls::stream< int8_t > &v1848 /* v1848[8] */ = A_fifo[5][3];	// L2253
  hls::stream< int8_t > &v1849 /* v1849[8] */ = B_fifo[3][5];	// L2254
  hls::stream< int8_t > &v1850 /* v1850[8] */ = A_fifo[5][4];	// L2255
  hls::stream< int8_t > &v1851 /* v1851[8] */ = B_fifo[3][6];	// L2256
  PE_kernel_gemm_3_5(v1848, v1849, v1850, v1851, v1666, 5, 3);	// L2257
  hls::stream< int8_t > &v1852 /* v1852[8] */ = A_fifo[5][4];	// L2258
  hls::stream< int8_t > &v1853 /* v1853[8] */ = B_fifo[4][5];	// L2259
  hls::stream< int8_t > &v1854 /* v1854[8] */ = A_fifo[5][5];	// L2260
  hls::stream< int8_t > &v1855 /* v1855[8] */ = B_fifo[4][6];	// L2261
  PE_kernel_gemm_4_5(v1852, v1853, v1854, v1855, v1666, 5, 4);	// L2262
  hls::stream< int8_t > &v1856 /* v1856[8] */ = A_fifo[5][5];	// L2263
  hls::stream< int8_t > &v1857 /* v1857[8] */ = B_fifo[5][5];	// L2264
  hls::stream< int8_t > &v1858 /* v1858[8] */ = A_fifo[5][6];	// L2265
  hls::stream< int8_t > &v1859 /* v1859[8] */ = B_fifo[5][6];	// L2266
  PE_kernel_gemm_5_5(v1856, v1857, v1858, v1859, v1666, 5, 5);	// L2267
  hls::stream< int8_t > &v1860 /* v1860[8] */ = A_fifo[5][6];	// L2268
  hls::stream< int8_t > &v1861 /* v1861[8] */ = B_fifo[6][5];	// L2269
  hls::stream< int8_t > &v1862 /* v1862[8] */ = A_fifo[5][7];	// L2270
  hls::stream< int8_t > &v1863 /* v1863[8] */ = B_fifo[6][6];	// L2271
  PE_kernel_gemm_6_5(v1860, v1861, v1862, v1863, v1666, 5, 6);	// L2272
  hls::stream< int8_t > &v1864 /* v1864[8] */ = A_fifo[5][7];	// L2273
  hls::stream< int8_t > &v1865 /* v1865[8] */ = B_fifo[7][5];	// L2274
  hls::stream< int8_t > &v1866 /* v1866[8] */ = A_fifo[5][8];	// L2275
  hls::stream< int8_t > &v1867 /* v1867[8] */ = B_fifo[7][6];	// L2276
  PE_kernel_gemm_7_5(v1864, v1865, v1866, v1867, v1666, 5, 7);	// L2277
  hls::stream< int8_t > &v1868 /* v1868[8] */ = A_fifo[6][0];	// L2278
  hls::stream< int8_t > &v1869 /* v1869[8] */ = B_fifo[0][6];	// L2279
  hls::stream< int8_t > &v1870 /* v1870[8] */ = A_fifo[6][1];	// L2280
  hls::stream< int8_t > &v1871 /* v1871[8] */ = B_fifo[0][7];	// L2281
  PE_kernel_gemm_0_6(v1868, v1869, v1870, v1871, v1666, 6, 0);	// L2282
  hls::stream< int8_t > &v1872 /* v1872[8] */ = A_fifo[6][1];	// L2283
  hls::stream< int8_t > &v1873 /* v1873[8] */ = B_fifo[1][6];	// L2284
  hls::stream< int8_t > &v1874 /* v1874[8] */ = A_fifo[6][2];	// L2285
  hls::stream< int8_t > &v1875 /* v1875[8] */ = B_fifo[1][7];	// L2286
  PE_kernel_gemm_1_6(v1872, v1873, v1874, v1875, v1666, 6, 1);	// L2287
  hls::stream< int8_t > &v1876 /* v1876[8] */ = A_fifo[6][2];	// L2288
  hls::stream< int8_t > &v1877 /* v1877[8] */ = B_fifo[2][6];	// L2289
  hls::stream< int8_t > &v1878 /* v1878[8] */ = A_fifo[6][3];	// L2290
  hls::stream< int8_t > &v1879 /* v1879[8] */ = B_fifo[2][7];	// L2291
  PE_kernel_gemm_2_6(v1876, v1877, v1878, v1879, v1666, 6, 2);	// L2292
  hls::stream< int8_t > &v1880 /* v1880[8] */ = A_fifo[6][3];	// L2293
  hls::stream< int8_t > &v1881 /* v1881[8] */ = B_fifo[3][6];	// L2294
  hls::stream< int8_t > &v1882 /* v1882[8] */ = A_fifo[6][4];	// L2295
  hls::stream< int8_t > &v1883 /* v1883[8] */ = B_fifo[3][7];	// L2296
  PE_kernel_gemm_3_6(v1880, v1881, v1882, v1883, v1666, 6, 3);	// L2297
  hls::stream< int8_t > &v1884 /* v1884[8] */ = A_fifo[6][4];	// L2298
  hls::stream< int8_t > &v1885 /* v1885[8] */ = B_fifo[4][6];	// L2299
  hls::stream< int8_t > &v1886 /* v1886[8] */ = A_fifo[6][5];	// L2300
  hls::stream< int8_t > &v1887 /* v1887[8] */ = B_fifo[4][7];	// L2301
  PE_kernel_gemm_4_6(v1884, v1885, v1886, v1887, v1666, 6, 4);	// L2302
  hls::stream< int8_t > &v1888 /* v1888[8] */ = A_fifo[6][5];	// L2303
  hls::stream< int8_t > &v1889 /* v1889[8] */ = B_fifo[5][6];	// L2304
  hls::stream< int8_t > &v1890 /* v1890[8] */ = A_fifo[6][6];	// L2305
  hls::stream< int8_t > &v1891 /* v1891[8] */ = B_fifo[5][7];	// L2306
  PE_kernel_gemm_5_6(v1888, v1889, v1890, v1891, v1666, 6, 5);	// L2307
  hls::stream< int8_t > &v1892 /* v1892[8] */ = A_fifo[6][6];	// L2308
  hls::stream< int8_t > &v1893 /* v1893[8] */ = B_fifo[6][6];	// L2309
  hls::stream< int8_t > &v1894 /* v1894[8] */ = A_fifo[6][7];	// L2310
  hls::stream< int8_t > &v1895 /* v1895[8] */ = B_fifo[6][7];	// L2311
  PE_kernel_gemm_6_6(v1892, v1893, v1894, v1895, v1666, 6, 6);	// L2312
  hls::stream< int8_t > &v1896 /* v1896[8] */ = A_fifo[6][7];	// L2313
  hls::stream< int8_t > &v1897 /* v1897[8] */ = B_fifo[7][6];	// L2314
  hls::stream< int8_t > &v1898 /* v1898[8] */ = A_fifo[6][8];	// L2315
  hls::stream< int8_t > &v1899 /* v1899[8] */ = B_fifo[7][7];	// L2316
  PE_kernel_gemm_7_6(v1896, v1897, v1898, v1899, v1666, 6, 7);	// L2317
  hls::stream< int8_t > &v1900 /* v1900[8] */ = A_fifo[7][0];	// L2318
  hls::stream< int8_t > &v1901 /* v1901[8] */ = B_fifo[0][7];	// L2319
  hls::stream< int8_t > &v1902 /* v1902[8] */ = A_fifo[7][1];	// L2320
  hls::stream< int8_t > &v1903 /* v1903[8] */ = B_fifo[0][8];	// L2321
  PE_kernel_gemm_0_7(v1900, v1901, v1902, v1903, v1666, 7, 0);	// L2322
  hls::stream< int8_t > &v1904 /* v1904[8] */ = A_fifo[7][1];	// L2323
  hls::stream< int8_t > &v1905 /* v1905[8] */ = B_fifo[1][7];	// L2324
  hls::stream< int8_t > &v1906 /* v1906[8] */ = A_fifo[7][2];	// L2325
  hls::stream< int8_t > &v1907 /* v1907[8] */ = B_fifo[1][8];	// L2326
  PE_kernel_gemm_1_7(v1904, v1905, v1906, v1907, v1666, 7, 1);	// L2327
  hls::stream< int8_t > &v1908 /* v1908[8] */ = A_fifo[7][2];	// L2328
  hls::stream< int8_t > &v1909 /* v1909[8] */ = B_fifo[2][7];	// L2329
  hls::stream< int8_t > &v1910 /* v1910[8] */ = A_fifo[7][3];	// L2330
  hls::stream< int8_t > &v1911 /* v1911[8] */ = B_fifo[2][8];	// L2331
  PE_kernel_gemm_2_7(v1908, v1909, v1910, v1911, v1666, 7, 2);	// L2332
  hls::stream< int8_t > &v1912 /* v1912[8] */ = A_fifo[7][3];	// L2333
  hls::stream< int8_t > &v1913 /* v1913[8] */ = B_fifo[3][7];	// L2334
  hls::stream< int8_t > &v1914 /* v1914[8] */ = A_fifo[7][4];	// L2335
  hls::stream< int8_t > &v1915 /* v1915[8] */ = B_fifo[3][8];	// L2336
  PE_kernel_gemm_3_7(v1912, v1913, v1914, v1915, v1666, 7, 3);	// L2337
  hls::stream< int8_t > &v1916 /* v1916[8] */ = A_fifo[7][4];	// L2338
  hls::stream< int8_t > &v1917 /* v1917[8] */ = B_fifo[4][7];	// L2339
  hls::stream< int8_t > &v1918 /* v1918[8] */ = A_fifo[7][5];	// L2340
  hls::stream< int8_t > &v1919 /* v1919[8] */ = B_fifo[4][8];	// L2341
  PE_kernel_gemm_4_7(v1916, v1917, v1918, v1919, v1666, 7, 4);	// L2342
  hls::stream< int8_t > &v1920 /* v1920[8] */ = A_fifo[7][5];	// L2343
  hls::stream< int8_t > &v1921 /* v1921[8] */ = B_fifo[5][7];	// L2344
  hls::stream< int8_t > &v1922 /* v1922[8] */ = A_fifo[7][6];	// L2345
  hls::stream< int8_t > &v1923 /* v1923[8] */ = B_fifo[5][8];	// L2346
  PE_kernel_gemm_5_7(v1920, v1921, v1922, v1923, v1666, 7, 5);	// L2347
  hls::stream< int8_t > &v1924 /* v1924[8] */ = A_fifo[7][6];	// L2348
  hls::stream< int8_t > &v1925 /* v1925[8] */ = B_fifo[6][7];	// L2349
  hls::stream< int8_t > &v1926 /* v1926[8] */ = A_fifo[7][7];	// L2350
  hls::stream< int8_t > &v1927 /* v1927[8] */ = B_fifo[6][8];	// L2351
  PE_kernel_gemm_6_7(v1924, v1925, v1926, v1927, v1666, 7, 6);	// L2352
  hls::stream< int8_t > &v1928 /* v1928[8] */ = A_fifo[7][7];	// L2353
  hls::stream< int8_t > &v1929 /* v1929[8] */ = B_fifo[7][7];	// L2354
  hls::stream< int8_t > &v1930 /* v1930[8] */ = A_fifo[7][8];	// L2355
  hls::stream< int8_t > &v1931 /* v1931[8] */ = B_fifo[7][8];	// L2356
  PE_kernel_gemm_7_7(v1928, v1929, v1930, v1931, v1666, 7, 7);	// L2357
  l_data_drain_k65: for (int k65 = 0; k65 < 8; k65++) {	// L2358
    l_S_m_4_m1: for (int m1 = 0; m1 < 8; m1++) {	// L2359
      int8_t v1934 = A_fifo[m1][8].read(); // A_fifo[m1][8][k65];	// L2360
      A_drain[m1] = v1934;	// L2361
    }
    l_S_n_5_n1: for (int n1 = 0; n1 < 8; n1++) {	// L2363
      int8_t v1936 = B_fifo[n1][8].read(); // B_fifo[n1][8][k65];	// L2364
      B_drain[n1] = v1936;	// L2365
    }
  }
}

void systolic_gemm(
  int8_t v1937[8][8],
  int8_t v1938[8][8],
  int32_t v1939[8][8]
) {	// L2370
  int8_t local_A[8][8];	// L2371
  #pragma HLS array_partition variable=local_A complete dim=1

  int8_t local_B[8][8];	// L2372
  #pragma HLS array_partition variable=local_B complete dim=2

  int32_t local_C[8][8];	// L2373
  #pragma HLS array_partition variable=local_C complete dim=1
  #pragma HLS array_partition variable=local_C complete dim=2

  l_outer_tile_mi_ni_fused: for (int mi_ni_fused = 0; mi_ni_fused < 1; mi_ni_fused++) {	// L2374
    l_load_A_tile_ak: for (int ak = 0; ak < 8; ak++) {	// L2376
    #pragma HLS pipeline II=1
      l_ai: for (int ai = 0; ai < 8; ai++) {	// L2377
        if (1) {	// L2382
          int8_t v1946 = v1937[((mi_ni_fused * 8) + ai)][ak];	// L2383
          local_A[ai][ak] = v1946;	// L2384
        }
      }
    }
    l_load_B_tile_bk: for (int bk = 0; bk < 8; bk++) {	// L2388
    #pragma HLS pipeline II=1
      l_bj: for (int bj = 0; bj < 8; bj++) {	// L2389
        int8_t v1949 = v1938[bk][((0 * 8) + bj)];	// L2390
        local_B[bk][bj] = v1949;	// L2391
      }
    }
    systolic_tile_gemm(local_A, local_B, local_C);	// L2394
    l_store_C_tile_sj: for (int sj = 0; sj < 8; sj++) {	// L2395
    #pragma HLS pipeline II=1
      l_si: for (int si = 0; si < 8; si++) {	// L2396
        int32_t v1952 = local_C[si][sj];	// L2397
        v1939[((mi_ni_fused * 8) + si)][((0 * 8) + sj)] = v1952;	// L2398
      }
    }
  }
}

void load_buf0(
  int8_t v1953[64],
  int8_t v1954[8][8]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 8; load_buf0_l_0++) {	//
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 8; load_buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v1957 = v1953[((load_buf0_l_0 * 8) + load_buf0_l_1)];	//
      v1954[load_buf0_l_0][load_buf0_l_1] = v1957;	//
    }
  }
}

void load_buf1(
  int8_t v1958[64],
  int8_t v1959[8][8]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 8; load_buf1_l_0++) {	//
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 8; load_buf1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v1962 = v1958[((load_buf1_l_0 * 8) + load_buf1_l_1)];	//
      v1959[load_buf1_l_0][load_buf1_l_1] = v1962;	//
    }
  }
}

void store_res2(
  int32_t v1963[8][8],
  int32_t v1964[64]
) {	//
  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 8; store_res2_l_0++) {	//
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 8; store_res2_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int32_t v1967 = v1963[store_res2_l_0][store_res2_l_1];	//
      v1964[((store_res2_l_0 * 8) + store_res2_l_1)] = v1967;	//
    }
  }
}

/// This is top function.
void gemm(
  int8_t *v1968,
  int8_t *v1969,
  int32_t *v1970
) {	// L2404
  #pragma HLS interface m_axi port=v1968 offset=slave bundle=gmem0 depth=64
  #pragma HLS interface m_axi port=v1969 offset=slave bundle=gmem1 depth=64
  #pragma HLS interface m_axi port=v1970 offset=slave bundle=gmem2 depth=64
  int8_t buf0[8][8];	//
  load_buf0(v1968, buf0);	//
  int8_t buf1[8][8];	//
  load_buf1(v1969, buf1);	//
  int32_t buf2[8][8];	//
  systolic_gemm(buf0, buf1, buf2);	// L2405
  store_res2(buf2, v1970);	//
}


} // extern "C"

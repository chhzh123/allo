
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
  hls::stream< int8_t > &v0 /* v0[4] */,
  hls::stream< int8_t > &v1 /* v1[4] */,
  hls::stream< int8_t > &v2 /* v2[4] */,
  hls::stream< int8_t > &v3 /* v3[4] */,
  int32_t v4[4][4],
  int v5,
  int v6
) {	// L5
  #pragma HLS stream variable=v0 depth=5
  #pragma HLS stream variable=v1 depth=5
  #pragma HLS stream variable=v2 depth=5
  #pragma HLS stream variable=v3 depth=5
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  int32_t v;	// L7
  v = 0;	// L8
  l_reduction_k: for (int k = 0; k < 4; k++) {	// L9
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
  hls::stream< int8_t > &v26 /* v26[4] */,
  hls::stream< int8_t > &v27 /* v27[4] */,
  hls::stream< int8_t > &v28 /* v28[4] */,
  hls::stream< int8_t > &v29 /* v29[4] */,
  int32_t v30[4][4],
  int v31,
  int v32
) {	// L36
  #pragma HLS stream variable=v26 depth=5
  #pragma HLS stream variable=v27 depth=5
  #pragma HLS stream variable=v28 depth=5
  #pragma HLS stream variable=v29 depth=5
  #pragma HLS array_partition variable=v30 complete dim=1
  #pragma HLS array_partition variable=v30 complete dim=2

  int32_t v1;	// L38
  v1 = 0;	// L39
  l_reduction_k1: for (int k1 = 0; k1 < 4; k1++) {	// L40
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
  hls::stream< int8_t > &v52 /* v52[4] */,
  hls::stream< int8_t > &v53 /* v53[4] */,
  hls::stream< int8_t > &v54 /* v54[4] */,
  hls::stream< int8_t > &v55 /* v55[4] */,
  int32_t v56[4][4],
  int v57,
  int v58
) {	// L67
  #pragma HLS stream variable=v52 depth=5
  #pragma HLS stream variable=v53 depth=5
  #pragma HLS stream variable=v54 depth=5
  #pragma HLS stream variable=v55 depth=5
  #pragma HLS array_partition variable=v56 complete dim=1
  #pragma HLS array_partition variable=v56 complete dim=2

  int32_t v2;	// L69
  v2 = 0;	// L70
  l_reduction_k2: for (int k2 = 0; k2 < 4; k2++) {	// L71
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
  hls::stream< int8_t > &v78 /* v78[4] */,
  hls::stream< int8_t > &v79 /* v79[4] */,
  hls::stream< int8_t > &v80 /* v80[4] */,
  hls::stream< int8_t > &v81 /* v81[4] */,
  int32_t v82[4][4],
  int v83,
  int v84
) {	// L98
  #pragma HLS stream variable=v78 depth=5
  #pragma HLS stream variable=v79 depth=5
  #pragma HLS stream variable=v80 depth=5
  #pragma HLS stream variable=v81 depth=5
  #pragma HLS array_partition variable=v82 complete dim=1
  #pragma HLS array_partition variable=v82 complete dim=2

  int32_t v3;	// L100
  v3 = 0;	// L101
  l_reduction_k3: for (int k3 = 0; k3 < 4; k3++) {	// L102
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

void PE_kernel_gemm_0_1(
  hls::stream< int8_t > &v104 /* v104[4] */,
  hls::stream< int8_t > &v105 /* v105[4] */,
  hls::stream< int8_t > &v106 /* v106[4] */,
  hls::stream< int8_t > &v107 /* v107[4] */,
  int32_t v108[4][4],
  int v109,
  int v110
) {	// L129
  #pragma HLS stream variable=v104 depth=5
  #pragma HLS stream variable=v105 depth=5
  #pragma HLS stream variable=v106 depth=5
  #pragma HLS stream variable=v107 depth=5
  #pragma HLS array_partition variable=v108 complete dim=1
  #pragma HLS array_partition variable=v108 complete dim=2

  int32_t v4;	// L131
  v4 = 0;	// L132
  l_reduction_k4: for (int k4 = 0; k4 < 4; k4++) {	// L133
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

void PE_kernel_gemm_1_1(
  hls::stream< int8_t > &v130 /* v130[4] */,
  hls::stream< int8_t > &v131 /* v131[4] */,
  hls::stream< int8_t > &v132 /* v132[4] */,
  hls::stream< int8_t > &v133 /* v133[4] */,
  int32_t v134[4][4],
  int v135,
  int v136
) {	// L160
  #pragma HLS stream variable=v130 depth=5
  #pragma HLS stream variable=v131 depth=5
  #pragma HLS stream variable=v132 depth=5
  #pragma HLS stream variable=v133 depth=5
  #pragma HLS array_partition variable=v134 complete dim=1
  #pragma HLS array_partition variable=v134 complete dim=2

  int32_t v5;	// L162
  v5 = 0;	// L163
  l_reduction_k5: for (int k5 = 0; k5 < 4; k5++) {	// L164
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

void PE_kernel_gemm_2_1(
  hls::stream< int8_t > &v156 /* v156[4] */,
  hls::stream< int8_t > &v157 /* v157[4] */,
  hls::stream< int8_t > &v158 /* v158[4] */,
  hls::stream< int8_t > &v159 /* v159[4] */,
  int32_t v160[4][4],
  int v161,
  int v162
) {	// L191
  #pragma HLS stream variable=v156 depth=5
  #pragma HLS stream variable=v157 depth=5
  #pragma HLS stream variable=v158 depth=5
  #pragma HLS stream variable=v159 depth=5
  #pragma HLS array_partition variable=v160 complete dim=1
  #pragma HLS array_partition variable=v160 complete dim=2

  int32_t v6;	// L193
  v6 = 0;	// L194
  l_reduction_k6: for (int k6 = 0; k6 < 4; k6++) {	// L195
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

void PE_kernel_gemm_3_1(
  hls::stream< int8_t > &v182 /* v182[4] */,
  hls::stream< int8_t > &v183 /* v183[4] */,
  hls::stream< int8_t > &v184 /* v184[4] */,
  hls::stream< int8_t > &v185 /* v185[4] */,
  int32_t v186[4][4],
  int v187,
  int v188
) {	// L222
  #pragma HLS stream variable=v182 depth=5
  #pragma HLS stream variable=v183 depth=5
  #pragma HLS stream variable=v184 depth=5
  #pragma HLS stream variable=v185 depth=5
  #pragma HLS array_partition variable=v186 complete dim=1
  #pragma HLS array_partition variable=v186 complete dim=2

  int32_t v7;	// L224
  v7 = 0;	// L225
  l_reduction_k7: for (int k7 = 0; k7 < 4; k7++) {	// L226
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

void PE_kernel_gemm_0_2(
  hls::stream< int8_t > &v208 /* v208[4] */,
  hls::stream< int8_t > &v209 /* v209[4] */,
  hls::stream< int8_t > &v210 /* v210[4] */,
  hls::stream< int8_t > &v211 /* v211[4] */,
  int32_t v212[4][4],
  int v213,
  int v214
) {	// L253
  #pragma HLS stream variable=v208 depth=5
  #pragma HLS stream variable=v209 depth=5
  #pragma HLS stream variable=v210 depth=5
  #pragma HLS stream variable=v211 depth=5
  #pragma HLS array_partition variable=v212 complete dim=1
  #pragma HLS array_partition variable=v212 complete dim=2

  int32_t v8;	// L255
  v8 = 0;	// L256
  l_reduction_k8: for (int k8 = 0; k8 < 4; k8++) {	// L257
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

void PE_kernel_gemm_1_2(
  hls::stream< int8_t > &v234 /* v234[4] */,
  hls::stream< int8_t > &v235 /* v235[4] */,
  hls::stream< int8_t > &v236 /* v236[4] */,
  hls::stream< int8_t > &v237 /* v237[4] */,
  int32_t v238[4][4],
  int v239,
  int v240
) {	// L284
  #pragma HLS stream variable=v234 depth=5
  #pragma HLS stream variable=v235 depth=5
  #pragma HLS stream variable=v236 depth=5
  #pragma HLS stream variable=v237 depth=5
  #pragma HLS array_partition variable=v238 complete dim=1
  #pragma HLS array_partition variable=v238 complete dim=2

  int32_t v9;	// L286
  v9 = 0;	// L287
  l_reduction_k9: for (int k9 = 0; k9 < 4; k9++) {	// L288
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

void PE_kernel_gemm_2_2(
  hls::stream< int8_t > &v260 /* v260[4] */,
  hls::stream< int8_t > &v261 /* v261[4] */,
  hls::stream< int8_t > &v262 /* v262[4] */,
  hls::stream< int8_t > &v263 /* v263[4] */,
  int32_t v264[4][4],
  int v265,
  int v266
) {	// L315
  #pragma HLS stream variable=v260 depth=5
  #pragma HLS stream variable=v261 depth=5
  #pragma HLS stream variable=v262 depth=5
  #pragma HLS stream variable=v263 depth=5
  #pragma HLS array_partition variable=v264 complete dim=1
  #pragma HLS array_partition variable=v264 complete dim=2

  int32_t v10;	// L317
  v10 = 0;	// L318
  l_reduction_k10: for (int k10 = 0; k10 < 4; k10++) {	// L319
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

void PE_kernel_gemm_3_2(
  hls::stream< int8_t > &v286 /* v286[4] */,
  hls::stream< int8_t > &v287 /* v287[4] */,
  hls::stream< int8_t > &v288 /* v288[4] */,
  hls::stream< int8_t > &v289 /* v289[4] */,
  int32_t v290[4][4],
  int v291,
  int v292
) {	// L346
  #pragma HLS stream variable=v286 depth=5
  #pragma HLS stream variable=v287 depth=5
  #pragma HLS stream variable=v288 depth=5
  #pragma HLS stream variable=v289 depth=5
  #pragma HLS array_partition variable=v290 complete dim=1
  #pragma HLS array_partition variable=v290 complete dim=2

  int32_t v11;	// L348
  v11 = 0;	// L349
  l_reduction_k11: for (int k11 = 0; k11 < 4; k11++) {	// L350
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

void PE_kernel_gemm_0_3(
  hls::stream< int8_t > &v312 /* v312[4] */,
  hls::stream< int8_t > &v313 /* v313[4] */,
  hls::stream< int8_t > &v314 /* v314[4] */,
  hls::stream< int8_t > &v315 /* v315[4] */,
  int32_t v316[4][4],
  int v317,
  int v318
) {	// L377
  #pragma HLS stream variable=v312 depth=5
  #pragma HLS stream variable=v313 depth=5
  #pragma HLS stream variable=v314 depth=5
  #pragma HLS stream variable=v315 depth=5
  #pragma HLS array_partition variable=v316 complete dim=1
  #pragma HLS array_partition variable=v316 complete dim=2

  int32_t v12;	// L379
  v12 = 0;	// L380
  l_reduction_k12: for (int k12 = 0; k12 < 4; k12++) {	// L381
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

void PE_kernel_gemm_1_3(
  hls::stream< int8_t > &v338 /* v338[4] */,
  hls::stream< int8_t > &v339 /* v339[4] */,
  hls::stream< int8_t > &v340 /* v340[4] */,
  hls::stream< int8_t > &v341 /* v341[4] */,
  int32_t v342[4][4],
  int v343,
  int v344
) {	// L408
  #pragma HLS stream variable=v338 depth=5
  #pragma HLS stream variable=v339 depth=5
  #pragma HLS stream variable=v340 depth=5
  #pragma HLS stream variable=v341 depth=5
  #pragma HLS array_partition variable=v342 complete dim=1
  #pragma HLS array_partition variable=v342 complete dim=2

  int32_t v13;	// L410
  v13 = 0;	// L411
  l_reduction_k13: for (int k13 = 0; k13 < 4; k13++) {	// L412
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

void PE_kernel_gemm_2_3(
  hls::stream< int8_t > &v364 /* v364[4] */,
  hls::stream< int8_t > &v365 /* v365[4] */,
  hls::stream< int8_t > &v366 /* v366[4] */,
  hls::stream< int8_t > &v367 /* v367[4] */,
  int32_t v368[4][4],
  int v369,
  int v370
) {	// L439
  #pragma HLS stream variable=v364 depth=5
  #pragma HLS stream variable=v365 depth=5
  #pragma HLS stream variable=v366 depth=5
  #pragma HLS stream variable=v367 depth=5
  #pragma HLS array_partition variable=v368 complete dim=1
  #pragma HLS array_partition variable=v368 complete dim=2

  int32_t v14;	// L441
  v14 = 0;	// L442
  l_reduction_k14: for (int k14 = 0; k14 < 4; k14++) {	// L443
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

void PE_kernel_gemm_3_3(
  hls::stream< int8_t > &v390 /* v390[4] */,
  hls::stream< int8_t > &v391 /* v391[4] */,
  hls::stream< int8_t > &v392 /* v392[4] */,
  hls::stream< int8_t > &v393 /* v393[4] */,
  int32_t v394[4][4],
  int v395,
  int v396
) {	// L470
  #pragma HLS stream variable=v390 depth=5
  #pragma HLS stream variable=v391 depth=5
  #pragma HLS stream variable=v392 depth=5
  #pragma HLS stream variable=v393 depth=5
  #pragma HLS array_partition variable=v394 complete dim=1
  #pragma HLS array_partition variable=v394 complete dim=2

  int32_t v15;	// L472
  v15 = 0;	// L473
  l_reduction_k15: for (int k15 = 0; k15 < 4; k15++) {	// L474
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

void systolic_tile_gemm(
  int8_t v416[4][4],
  int8_t v417[4][4],
  int32_t v418[4][4]
) {	// L501
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v416 complete dim=1

  #pragma HLS array_partition variable=v417 complete dim=2

  #pragma HLS array_partition variable=v418 complete dim=1
  #pragma HLS array_partition variable=v418 complete dim=2

  hls::stream< int8_t > A_fifo[4][5] /* A_fifo[4][5][4] */;	// L502
  #pragma HLS stream variable=A_fifo depth=5
  hls::stream< int8_t > B_fifo[4][5] /* B_fifo[4][5][4] */;	// L503
  #pragma HLS stream variable=B_fifo depth=5
  int8_t A_drain[4];	// L504
  int8_t B_drain[4];	// L505
  l_data_load_k16: for (int k16 = 0; k16 < 4; k16++) {	// L506
    l_S_m_0_m: for (int m = 0; m < 4; m++) {	// L507
      int8_t v425 = v416[m][k16];	// L508
      A_fifo[m][0].write(v425); // A_fifo[m][0][k16] = v425;	// L509
    }
    l_S_n_1_n: for (int n = 0; n < 4; n++) {	// L511
      int8_t v427 = v417[k16][n];	// L512
      B_fifo[n][0].write(v427); // B_fifo[n][0][k16] = v427;	// L513
    }
  }
  hls::stream< int8_t > &v428 /* v428[4] */ = A_fifo[0][0];	// L517
  hls::stream< int8_t > &v429 /* v429[4] */ = B_fifo[0][0];	// L518
  hls::stream< int8_t > &v430 /* v430[4] */ = A_fifo[0][1];	// L524
  hls::stream< int8_t > &v431 /* v431[4] */ = B_fifo[0][1];	// L525
  PE_kernel_gemm_0_0(v428, v429, v430, v431, v418, 0, 0);	// L526
  hls::stream< int8_t > &v432 /* v432[4] */ = A_fifo[0][1];	// L528
  hls::stream< int8_t > &v433 /* v433[4] */ = B_fifo[1][0];	// L529
  hls::stream< int8_t > &v434 /* v434[4] */ = A_fifo[0][2];	// L533
  hls::stream< int8_t > &v435 /* v435[4] */ = B_fifo[1][1];	// L534
  PE_kernel_gemm_1_0(v432, v433, v434, v435, v418, 0, 1);	// L535
  hls::stream< int8_t > &v436 /* v436[4] */ = A_fifo[0][2];	// L537
  hls::stream< int8_t > &v437 /* v437[4] */ = B_fifo[2][0];	// L538
  hls::stream< int8_t > &v438 /* v438[4] */ = A_fifo[0][3];	// L542
  hls::stream< int8_t > &v439 /* v439[4] */ = B_fifo[2][1];	// L543
  PE_kernel_gemm_2_0(v436, v437, v438, v439, v418, 0, 2);	// L544
  hls::stream< int8_t > &v440 /* v440[4] */ = A_fifo[0][3];	// L546
  hls::stream< int8_t > &v441 /* v441[4] */ = B_fifo[3][0];	// L547
  hls::stream< int8_t > &v442 /* v442[4] */ = A_fifo[0][4];	// L551
  hls::stream< int8_t > &v443 /* v443[4] */ = B_fifo[3][1];	// L552
  PE_kernel_gemm_3_0(v440, v441, v442, v443, v418, 0, 3);	// L553
  hls::stream< int8_t > &v444 /* v444[4] */ = A_fifo[1][0];	// L554
  hls::stream< int8_t > &v445 /* v445[4] */ = B_fifo[0][1];	// L555
  hls::stream< int8_t > &v446 /* v446[4] */ = A_fifo[1][1];	// L556
  hls::stream< int8_t > &v447 /* v447[4] */ = B_fifo[0][2];	// L557
  PE_kernel_gemm_0_1(v444, v445, v446, v447, v418, 1, 0);	// L558
  hls::stream< int8_t > &v448 /* v448[4] */ = A_fifo[1][1];	// L559
  hls::stream< int8_t > &v449 /* v449[4] */ = B_fifo[1][1];	// L560
  hls::stream< int8_t > &v450 /* v450[4] */ = A_fifo[1][2];	// L561
  hls::stream< int8_t > &v451 /* v451[4] */ = B_fifo[1][2];	// L562
  PE_kernel_gemm_1_1(v448, v449, v450, v451, v418, 1, 1);	// L563
  hls::stream< int8_t > &v452 /* v452[4] */ = A_fifo[1][2];	// L564
  hls::stream< int8_t > &v453 /* v453[4] */ = B_fifo[2][1];	// L565
  hls::stream< int8_t > &v454 /* v454[4] */ = A_fifo[1][3];	// L566
  hls::stream< int8_t > &v455 /* v455[4] */ = B_fifo[2][2];	// L567
  PE_kernel_gemm_2_1(v452, v453, v454, v455, v418, 1, 2);	// L568
  hls::stream< int8_t > &v456 /* v456[4] */ = A_fifo[1][3];	// L569
  hls::stream< int8_t > &v457 /* v457[4] */ = B_fifo[3][1];	// L570
  hls::stream< int8_t > &v458 /* v458[4] */ = A_fifo[1][4];	// L571
  hls::stream< int8_t > &v459 /* v459[4] */ = B_fifo[3][2];	// L572
  PE_kernel_gemm_3_1(v456, v457, v458, v459, v418, 1, 3);	// L573
  hls::stream< int8_t > &v460 /* v460[4] */ = A_fifo[2][0];	// L574
  hls::stream< int8_t > &v461 /* v461[4] */ = B_fifo[0][2];	// L575
  hls::stream< int8_t > &v462 /* v462[4] */ = A_fifo[2][1];	// L576
  hls::stream< int8_t > &v463 /* v463[4] */ = B_fifo[0][3];	// L577
  PE_kernel_gemm_0_2(v460, v461, v462, v463, v418, 2, 0);	// L578
  hls::stream< int8_t > &v464 /* v464[4] */ = A_fifo[2][1];	// L579
  hls::stream< int8_t > &v465 /* v465[4] */ = B_fifo[1][2];	// L580
  hls::stream< int8_t > &v466 /* v466[4] */ = A_fifo[2][2];	// L581
  hls::stream< int8_t > &v467 /* v467[4] */ = B_fifo[1][3];	// L582
  PE_kernel_gemm_1_2(v464, v465, v466, v467, v418, 2, 1);	// L583
  hls::stream< int8_t > &v468 /* v468[4] */ = A_fifo[2][2];	// L584
  hls::stream< int8_t > &v469 /* v469[4] */ = B_fifo[2][2];	// L585
  hls::stream< int8_t > &v470 /* v470[4] */ = A_fifo[2][3];	// L586
  hls::stream< int8_t > &v471 /* v471[4] */ = B_fifo[2][3];	// L587
  PE_kernel_gemm_2_2(v468, v469, v470, v471, v418, 2, 2);	// L588
  hls::stream< int8_t > &v472 /* v472[4] */ = A_fifo[2][3];	// L589
  hls::stream< int8_t > &v473 /* v473[4] */ = B_fifo[3][2];	// L590
  hls::stream< int8_t > &v474 /* v474[4] */ = A_fifo[2][4];	// L591
  hls::stream< int8_t > &v475 /* v475[4] */ = B_fifo[3][3];	// L592
  PE_kernel_gemm_3_2(v472, v473, v474, v475, v418, 2, 3);	// L593
  hls::stream< int8_t > &v476 /* v476[4] */ = A_fifo[3][0];	// L594
  hls::stream< int8_t > &v477 /* v477[4] */ = B_fifo[0][3];	// L595
  hls::stream< int8_t > &v478 /* v478[4] */ = A_fifo[3][1];	// L596
  hls::stream< int8_t > &v479 /* v479[4] */ = B_fifo[0][4];	// L597
  PE_kernel_gemm_0_3(v476, v477, v478, v479, v418, 3, 0);	// L598
  hls::stream< int8_t > &v480 /* v480[4] */ = A_fifo[3][1];	// L599
  hls::stream< int8_t > &v481 /* v481[4] */ = B_fifo[1][3];	// L600
  hls::stream< int8_t > &v482 /* v482[4] */ = A_fifo[3][2];	// L601
  hls::stream< int8_t > &v483 /* v483[4] */ = B_fifo[1][4];	// L602
  PE_kernel_gemm_1_3(v480, v481, v482, v483, v418, 3, 1);	// L603
  hls::stream< int8_t > &v484 /* v484[4] */ = A_fifo[3][2];	// L604
  hls::stream< int8_t > &v485 /* v485[4] */ = B_fifo[2][3];	// L605
  hls::stream< int8_t > &v486 /* v486[4] */ = A_fifo[3][3];	// L606
  hls::stream< int8_t > &v487 /* v487[4] */ = B_fifo[2][4];	// L607
  PE_kernel_gemm_2_3(v484, v485, v486, v487, v418, 3, 2);	// L608
  hls::stream< int8_t > &v488 /* v488[4] */ = A_fifo[3][3];	// L609
  hls::stream< int8_t > &v489 /* v489[4] */ = B_fifo[3][3];	// L610
  hls::stream< int8_t > &v490 /* v490[4] */ = A_fifo[3][4];	// L611
  hls::stream< int8_t > &v491 /* v491[4] */ = B_fifo[3][4];	// L612
  PE_kernel_gemm_3_3(v488, v489, v490, v491, v418, 3, 3);	// L613
  l_data_drain_k17: for (int k17 = 0; k17 < 4; k17++) {	// L614
    l_S_m_4_m1: for (int m1 = 0; m1 < 4; m1++) {	// L615
      int8_t v494 = A_fifo[m1][4].read(); // A_fifo[m1][4][k17];	// L616
      A_drain[m1] = v494;	// L617
    }
    l_S_n_5_n1: for (int n1 = 0; n1 < 4; n1++) {	// L619
      int8_t v496 = B_fifo[n1][4].read(); // B_fifo[n1][4][k17];	// L620
      B_drain[n1] = v496;	// L621
    }
  }
}

void systolic_gemm(
  int8_t v497[4][4],
  int8_t v498[4][4],
  int32_t v499[4][4]
) {	// L626
  int8_t local_A[4][4];	// L627
  #pragma HLS array_partition variable=local_A complete dim=1

  int8_t local_B[4][4];	// L628
  #pragma HLS array_partition variable=local_B complete dim=2

  int32_t local_C[4][4];	// L629
  #pragma HLS array_partition variable=local_C complete dim=1
  #pragma HLS array_partition variable=local_C complete dim=2

  l_outer_tile_mi_ni_fused: for (int mi_ni_fused = 0; mi_ni_fused < 1; mi_ni_fused++) {	// L630
    l_load_A_tile_ak: for (int ak = 0; ak < 4; ak++) {	// L632
    #pragma HLS pipeline II=1
      l_ai: for (int ai = 0; ai < 4; ai++) {	// L633
        if (1) {	// L638
          int8_t v506 = v497[((mi_ni_fused * 4) + ai)][ak];	// L639
          local_A[ai][ak] = v506;	// L640
        }
      }
    }
    l_load_B_tile_bk: for (int bk = 0; bk < 4; bk++) {	// L644
    #pragma HLS pipeline II=1
      l_bj: for (int bj = 0; bj < 4; bj++) {	// L645
        int8_t v509 = v498[bk][((0 * 4) + bj)];	// L646
        local_B[bk][bj] = v509;	// L647
      }
    }
    systolic_tile_gemm(local_A, local_B, local_C);	// L650
    l_store_C_tile_sj: for (int sj = 0; sj < 4; sj++) {	// L651
    #pragma HLS pipeline II=1
      l_si: for (int si = 0; si < 4; si++) {	// L652
        int32_t v512 = local_C[si][sj];	// L653
        v499[((mi_ni_fused * 4) + si)][((0 * 4) + sj)] = v512;	// L654
      }
    }
  }
}

void load_buf0(
  int8_t v513[16],
  int8_t v514[4][4]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 4; load_buf0_l_0++) {	//
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 4; load_buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v517 = v513[((load_buf0_l_0 * 4) + load_buf0_l_1)];	//
      v514[load_buf0_l_0][load_buf0_l_1] = v517;	//
    }
  }
}

void load_buf1(
  int8_t v518[16],
  int8_t v519[4][4]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 4; load_buf1_l_0++) {	//
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 4; load_buf1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v522 = v518[((load_buf1_l_0 * 4) + load_buf1_l_1)];	//
      v519[load_buf1_l_0][load_buf1_l_1] = v522;	//
    }
  }
}

void store_res2(
  int32_t v523[4][4],
  int32_t v524[16]
) {	//
  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 4; store_res2_l_0++) {	//
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 4; store_res2_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int32_t v527 = v523[store_res2_l_0][store_res2_l_1];	//
      v524[((store_res2_l_0 * 4) + store_res2_l_1)] = v527;	//
    }
  }
}

/// This is top function.
void gemm(
  int8_t *v528,
  int8_t *v529,
  int32_t *v530
) {	// L660
  #pragma HLS interface m_axi port=v528 offset=slave bundle=gmem0 depth=16
  #pragma HLS interface m_axi port=v529 offset=slave bundle=gmem1 depth=16
  #pragma HLS interface m_axi port=v530 offset=slave bundle=gmem2 depth=16
  int8_t buf0[4][4];	//
  load_buf0(v528, buf0);	//
  int8_t buf1[4][4];	//
  load_buf1(v529, buf1);	//
  int32_t buf2[4][4];	//
  systolic_gemm(buf0, buf1, buf2);	// L661
  store_res2(buf2, v530);	//
}


} // extern "C"

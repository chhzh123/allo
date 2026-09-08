
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
  hls::stream< int8_t > &v0 /* v0[16] */,
  hls::stream< int8_t > &v1 /* v1[16] */,
  hls::stream< int8_t > &v2 /* v2[16] */,
  hls::stream< int8_t > &v3 /* v3[16] */,
  int32_t v4[16][16],
  int v5,
  int v6
) {	// L5
  #pragma HLS stream variable=v0 depth=17
  #pragma HLS stream variable=v1 depth=17
  #pragma HLS stream variable=v2 depth=17
  #pragma HLS stream variable=v3 depth=17
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  int32_t v;	// L7
  v = 0;	// L8
  l_reduction_k: for (int k = 0; k < 16; k++) {	// L9
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
  hls::stream< int8_t > &v26 /* v26[16] */,
  hls::stream< int8_t > &v27 /* v27[16] */,
  hls::stream< int8_t > &v28 /* v28[16] */,
  hls::stream< int8_t > &v29 /* v29[16] */,
  int32_t v30[16][16],
  int v31,
  int v32
) {	// L36
  #pragma HLS stream variable=v26 depth=17
  #pragma HLS stream variable=v27 depth=17
  #pragma HLS stream variable=v28 depth=17
  #pragma HLS stream variable=v29 depth=17
  #pragma HLS array_partition variable=v30 complete dim=1
  #pragma HLS array_partition variable=v30 complete dim=2

  int32_t v1;	// L38
  v1 = 0;	// L39
  l_reduction_k1: for (int k1 = 0; k1 < 16; k1++) {	// L40
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
  hls::stream< int8_t > &v52 /* v52[16] */,
  hls::stream< int8_t > &v53 /* v53[16] */,
  hls::stream< int8_t > &v54 /* v54[16] */,
  hls::stream< int8_t > &v55 /* v55[16] */,
  int32_t v56[16][16],
  int v57,
  int v58
) {	// L67
  #pragma HLS stream variable=v52 depth=17
  #pragma HLS stream variable=v53 depth=17
  #pragma HLS stream variable=v54 depth=17
  #pragma HLS stream variable=v55 depth=17
  #pragma HLS array_partition variable=v56 complete dim=1
  #pragma HLS array_partition variable=v56 complete dim=2

  int32_t v2;	// L69
  v2 = 0;	// L70
  l_reduction_k2: for (int k2 = 0; k2 < 16; k2++) {	// L71
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
  hls::stream< int8_t > &v78 /* v78[16] */,
  hls::stream< int8_t > &v79 /* v79[16] */,
  hls::stream< int8_t > &v80 /* v80[16] */,
  hls::stream< int8_t > &v81 /* v81[16] */,
  int32_t v82[16][16],
  int v83,
  int v84
) {	// L98
  #pragma HLS stream variable=v78 depth=17
  #pragma HLS stream variable=v79 depth=17
  #pragma HLS stream variable=v80 depth=17
  #pragma HLS stream variable=v81 depth=17
  #pragma HLS array_partition variable=v82 complete dim=1
  #pragma HLS array_partition variable=v82 complete dim=2

  int32_t v3;	// L100
  v3 = 0;	// L101
  l_reduction_k3: for (int k3 = 0; k3 < 16; k3++) {	// L102
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
  hls::stream< int8_t > &v104 /* v104[16] */,
  hls::stream< int8_t > &v105 /* v105[16] */,
  hls::stream< int8_t > &v106 /* v106[16] */,
  hls::stream< int8_t > &v107 /* v107[16] */,
  int32_t v108[16][16],
  int v109,
  int v110
) {	// L129
  #pragma HLS stream variable=v104 depth=17
  #pragma HLS stream variable=v105 depth=17
  #pragma HLS stream variable=v106 depth=17
  #pragma HLS stream variable=v107 depth=17
  #pragma HLS array_partition variable=v108 complete dim=1
  #pragma HLS array_partition variable=v108 complete dim=2

  int32_t v4;	// L131
  v4 = 0;	// L132
  l_reduction_k4: for (int k4 = 0; k4 < 16; k4++) {	// L133
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
  hls::stream< int8_t > &v130 /* v130[16] */,
  hls::stream< int8_t > &v131 /* v131[16] */,
  hls::stream< int8_t > &v132 /* v132[16] */,
  hls::stream< int8_t > &v133 /* v133[16] */,
  int32_t v134[16][16],
  int v135,
  int v136
) {	// L160
  #pragma HLS stream variable=v130 depth=17
  #pragma HLS stream variable=v131 depth=17
  #pragma HLS stream variable=v132 depth=17
  #pragma HLS stream variable=v133 depth=17
  #pragma HLS array_partition variable=v134 complete dim=1
  #pragma HLS array_partition variable=v134 complete dim=2

  int32_t v5;	// L162
  v5 = 0;	// L163
  l_reduction_k5: for (int k5 = 0; k5 < 16; k5++) {	// L164
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
  hls::stream< int8_t > &v156 /* v156[16] */,
  hls::stream< int8_t > &v157 /* v157[16] */,
  hls::stream< int8_t > &v158 /* v158[16] */,
  hls::stream< int8_t > &v159 /* v159[16] */,
  int32_t v160[16][16],
  int v161,
  int v162
) {	// L191
  #pragma HLS stream variable=v156 depth=17
  #pragma HLS stream variable=v157 depth=17
  #pragma HLS stream variable=v158 depth=17
  #pragma HLS stream variable=v159 depth=17
  #pragma HLS array_partition variable=v160 complete dim=1
  #pragma HLS array_partition variable=v160 complete dim=2

  int32_t v6;	// L193
  v6 = 0;	// L194
  l_reduction_k6: for (int k6 = 0; k6 < 16; k6++) {	// L195
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
  hls::stream< int8_t > &v182 /* v182[16] */,
  hls::stream< int8_t > &v183 /* v183[16] */,
  hls::stream< int8_t > &v184 /* v184[16] */,
  hls::stream< int8_t > &v185 /* v185[16] */,
  int32_t v186[16][16],
  int v187,
  int v188
) {	// L222
  #pragma HLS stream variable=v182 depth=17
  #pragma HLS stream variable=v183 depth=17
  #pragma HLS stream variable=v184 depth=17
  #pragma HLS stream variable=v185 depth=17
  #pragma HLS array_partition variable=v186 complete dim=1
  #pragma HLS array_partition variable=v186 complete dim=2

  int32_t v7;	// L224
  v7 = 0;	// L225
  l_reduction_k7: for (int k7 = 0; k7 < 16; k7++) {	// L226
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

void PE_kernel_gemm_8_0(
  hls::stream< int8_t > &v208 /* v208[16] */,
  hls::stream< int8_t > &v209 /* v209[16] */,
  hls::stream< int8_t > &v210 /* v210[16] */,
  hls::stream< int8_t > &v211 /* v211[16] */,
  int32_t v212[16][16],
  int v213,
  int v214
) {	// L253
  #pragma HLS stream variable=v208 depth=17
  #pragma HLS stream variable=v209 depth=17
  #pragma HLS stream variable=v210 depth=17
  #pragma HLS stream variable=v211 depth=17
  #pragma HLS array_partition variable=v212 complete dim=1
  #pragma HLS array_partition variable=v212 complete dim=2

  int32_t v8;	// L255
  v8 = 0;	// L256
  l_reduction_k8: for (int k8 = 0; k8 < 16; k8++) {	// L257
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

void PE_kernel_gemm_9_0(
  hls::stream< int8_t > &v234 /* v234[16] */,
  hls::stream< int8_t > &v235 /* v235[16] */,
  hls::stream< int8_t > &v236 /* v236[16] */,
  hls::stream< int8_t > &v237 /* v237[16] */,
  int32_t v238[16][16],
  int v239,
  int v240
) {	// L284
  #pragma HLS stream variable=v234 depth=17
  #pragma HLS stream variable=v235 depth=17
  #pragma HLS stream variable=v236 depth=17
  #pragma HLS stream variable=v237 depth=17
  #pragma HLS array_partition variable=v238 complete dim=1
  #pragma HLS array_partition variable=v238 complete dim=2

  int32_t v9;	// L286
  v9 = 0;	// L287
  l_reduction_k9: for (int k9 = 0; k9 < 16; k9++) {	// L288
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

void PE_kernel_gemm_10_0(
  hls::stream< int8_t > &v260 /* v260[16] */,
  hls::stream< int8_t > &v261 /* v261[16] */,
  hls::stream< int8_t > &v262 /* v262[16] */,
  hls::stream< int8_t > &v263 /* v263[16] */,
  int32_t v264[16][16],
  int v265,
  int v266
) {	// L315
  #pragma HLS stream variable=v260 depth=17
  #pragma HLS stream variable=v261 depth=17
  #pragma HLS stream variable=v262 depth=17
  #pragma HLS stream variable=v263 depth=17
  #pragma HLS array_partition variable=v264 complete dim=1
  #pragma HLS array_partition variable=v264 complete dim=2

  int32_t v10;	// L317
  v10 = 0;	// L318
  l_reduction_k10: for (int k10 = 0; k10 < 16; k10++) {	// L319
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

void PE_kernel_gemm_11_0(
  hls::stream< int8_t > &v286 /* v286[16] */,
  hls::stream< int8_t > &v287 /* v287[16] */,
  hls::stream< int8_t > &v288 /* v288[16] */,
  hls::stream< int8_t > &v289 /* v289[16] */,
  int32_t v290[16][16],
  int v291,
  int v292
) {	// L346
  #pragma HLS stream variable=v286 depth=17
  #pragma HLS stream variable=v287 depth=17
  #pragma HLS stream variable=v288 depth=17
  #pragma HLS stream variable=v289 depth=17
  #pragma HLS array_partition variable=v290 complete dim=1
  #pragma HLS array_partition variable=v290 complete dim=2

  int32_t v11;	// L348
  v11 = 0;	// L349
  l_reduction_k11: for (int k11 = 0; k11 < 16; k11++) {	// L350
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

void PE_kernel_gemm_12_0(
  hls::stream< int8_t > &v312 /* v312[16] */,
  hls::stream< int8_t > &v313 /* v313[16] */,
  hls::stream< int8_t > &v314 /* v314[16] */,
  hls::stream< int8_t > &v315 /* v315[16] */,
  int32_t v316[16][16],
  int v317,
  int v318
) {	// L377
  #pragma HLS stream variable=v312 depth=17
  #pragma HLS stream variable=v313 depth=17
  #pragma HLS stream variable=v314 depth=17
  #pragma HLS stream variable=v315 depth=17
  #pragma HLS array_partition variable=v316 complete dim=1
  #pragma HLS array_partition variable=v316 complete dim=2

  int32_t v12;	// L379
  v12 = 0;	// L380
  l_reduction_k12: for (int k12 = 0; k12 < 16; k12++) {	// L381
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

void PE_kernel_gemm_13_0(
  hls::stream< int8_t > &v338 /* v338[16] */,
  hls::stream< int8_t > &v339 /* v339[16] */,
  hls::stream< int8_t > &v340 /* v340[16] */,
  hls::stream< int8_t > &v341 /* v341[16] */,
  int32_t v342[16][16],
  int v343,
  int v344
) {	// L408
  #pragma HLS stream variable=v338 depth=17
  #pragma HLS stream variable=v339 depth=17
  #pragma HLS stream variable=v340 depth=17
  #pragma HLS stream variable=v341 depth=17
  #pragma HLS array_partition variable=v342 complete dim=1
  #pragma HLS array_partition variable=v342 complete dim=2

  int32_t v13;	// L410
  v13 = 0;	// L411
  l_reduction_k13: for (int k13 = 0; k13 < 16; k13++) {	// L412
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

void PE_kernel_gemm_14_0(
  hls::stream< int8_t > &v364 /* v364[16] */,
  hls::stream< int8_t > &v365 /* v365[16] */,
  hls::stream< int8_t > &v366 /* v366[16] */,
  hls::stream< int8_t > &v367 /* v367[16] */,
  int32_t v368[16][16],
  int v369,
  int v370
) {	// L439
  #pragma HLS stream variable=v364 depth=17
  #pragma HLS stream variable=v365 depth=17
  #pragma HLS stream variable=v366 depth=17
  #pragma HLS stream variable=v367 depth=17
  #pragma HLS array_partition variable=v368 complete dim=1
  #pragma HLS array_partition variable=v368 complete dim=2

  int32_t v14;	// L441
  v14 = 0;	// L442
  l_reduction_k14: for (int k14 = 0; k14 < 16; k14++) {	// L443
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

void PE_kernel_gemm_15_0(
  hls::stream< int8_t > &v390 /* v390[16] */,
  hls::stream< int8_t > &v391 /* v391[16] */,
  hls::stream< int8_t > &v392 /* v392[16] */,
  hls::stream< int8_t > &v393 /* v393[16] */,
  int32_t v394[16][16],
  int v395,
  int v396
) {	// L470
  #pragma HLS stream variable=v390 depth=17
  #pragma HLS stream variable=v391 depth=17
  #pragma HLS stream variable=v392 depth=17
  #pragma HLS stream variable=v393 depth=17
  #pragma HLS array_partition variable=v394 complete dim=1
  #pragma HLS array_partition variable=v394 complete dim=2

  int32_t v15;	// L472
  v15 = 0;	// L473
  l_reduction_k15: for (int k15 = 0; k15 < 16; k15++) {	// L474
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

void PE_kernel_gemm_0_1(
  hls::stream< int8_t > &v416 /* v416[16] */,
  hls::stream< int8_t > &v417 /* v417[16] */,
  hls::stream< int8_t > &v418 /* v418[16] */,
  hls::stream< int8_t > &v419 /* v419[16] */,
  int32_t v420[16][16],
  int v421,
  int v422
) {	// L501
  #pragma HLS stream variable=v416 depth=17
  #pragma HLS stream variable=v417 depth=17
  #pragma HLS stream variable=v418 depth=17
  #pragma HLS stream variable=v419 depth=17
  #pragma HLS array_partition variable=v420 complete dim=1
  #pragma HLS array_partition variable=v420 complete dim=2

  int32_t v16;	// L503
  v16 = 0;	// L504
  l_reduction_k16: for (int k16 = 0; k16 < 16; k16++) {	// L505
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

void PE_kernel_gemm_1_1(
  hls::stream< int8_t > &v442 /* v442[16] */,
  hls::stream< int8_t > &v443 /* v443[16] */,
  hls::stream< int8_t > &v444 /* v444[16] */,
  hls::stream< int8_t > &v445 /* v445[16] */,
  int32_t v446[16][16],
  int v447,
  int v448
) {	// L532
  #pragma HLS stream variable=v442 depth=17
  #pragma HLS stream variable=v443 depth=17
  #pragma HLS stream variable=v444 depth=17
  #pragma HLS stream variable=v445 depth=17
  #pragma HLS array_partition variable=v446 complete dim=1
  #pragma HLS array_partition variable=v446 complete dim=2

  int32_t v17;	// L534
  v17 = 0;	// L535
  l_reduction_k17: for (int k17 = 0; k17 < 16; k17++) {	// L536
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

void PE_kernel_gemm_2_1(
  hls::stream< int8_t > &v468 /* v468[16] */,
  hls::stream< int8_t > &v469 /* v469[16] */,
  hls::stream< int8_t > &v470 /* v470[16] */,
  hls::stream< int8_t > &v471 /* v471[16] */,
  int32_t v472[16][16],
  int v473,
  int v474
) {	// L563
  #pragma HLS stream variable=v468 depth=17
  #pragma HLS stream variable=v469 depth=17
  #pragma HLS stream variable=v470 depth=17
  #pragma HLS stream variable=v471 depth=17
  #pragma HLS array_partition variable=v472 complete dim=1
  #pragma HLS array_partition variable=v472 complete dim=2

  int32_t v18;	// L565
  v18 = 0;	// L566
  l_reduction_k18: for (int k18 = 0; k18 < 16; k18++) {	// L567
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

void PE_kernel_gemm_3_1(
  hls::stream< int8_t > &v494 /* v494[16] */,
  hls::stream< int8_t > &v495 /* v495[16] */,
  hls::stream< int8_t > &v496 /* v496[16] */,
  hls::stream< int8_t > &v497 /* v497[16] */,
  int32_t v498[16][16],
  int v499,
  int v500
) {	// L594
  #pragma HLS stream variable=v494 depth=17
  #pragma HLS stream variable=v495 depth=17
  #pragma HLS stream variable=v496 depth=17
  #pragma HLS stream variable=v497 depth=17
  #pragma HLS array_partition variable=v498 complete dim=1
  #pragma HLS array_partition variable=v498 complete dim=2

  int32_t v19;	// L596
  v19 = 0;	// L597
  l_reduction_k19: for (int k19 = 0; k19 < 16; k19++) {	// L598
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

void PE_kernel_gemm_4_1(
  hls::stream< int8_t > &v520 /* v520[16] */,
  hls::stream< int8_t > &v521 /* v521[16] */,
  hls::stream< int8_t > &v522 /* v522[16] */,
  hls::stream< int8_t > &v523 /* v523[16] */,
  int32_t v524[16][16],
  int v525,
  int v526
) {	// L625
  #pragma HLS stream variable=v520 depth=17
  #pragma HLS stream variable=v521 depth=17
  #pragma HLS stream variable=v522 depth=17
  #pragma HLS stream variable=v523 depth=17
  #pragma HLS array_partition variable=v524 complete dim=1
  #pragma HLS array_partition variable=v524 complete dim=2

  int32_t v20;	// L627
  v20 = 0;	// L628
  l_reduction_k20: for (int k20 = 0; k20 < 16; k20++) {	// L629
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

void PE_kernel_gemm_5_1(
  hls::stream< int8_t > &v546 /* v546[16] */,
  hls::stream< int8_t > &v547 /* v547[16] */,
  hls::stream< int8_t > &v548 /* v548[16] */,
  hls::stream< int8_t > &v549 /* v549[16] */,
  int32_t v550[16][16],
  int v551,
  int v552
) {	// L656
  #pragma HLS stream variable=v546 depth=17
  #pragma HLS stream variable=v547 depth=17
  #pragma HLS stream variable=v548 depth=17
  #pragma HLS stream variable=v549 depth=17
  #pragma HLS array_partition variable=v550 complete dim=1
  #pragma HLS array_partition variable=v550 complete dim=2

  int32_t v21;	// L658
  v21 = 0;	// L659
  l_reduction_k21: for (int k21 = 0; k21 < 16; k21++) {	// L660
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

void PE_kernel_gemm_6_1(
  hls::stream< int8_t > &v572 /* v572[16] */,
  hls::stream< int8_t > &v573 /* v573[16] */,
  hls::stream< int8_t > &v574 /* v574[16] */,
  hls::stream< int8_t > &v575 /* v575[16] */,
  int32_t v576[16][16],
  int v577,
  int v578
) {	// L687
  #pragma HLS stream variable=v572 depth=17
  #pragma HLS stream variable=v573 depth=17
  #pragma HLS stream variable=v574 depth=17
  #pragma HLS stream variable=v575 depth=17
  #pragma HLS array_partition variable=v576 complete dim=1
  #pragma HLS array_partition variable=v576 complete dim=2

  int32_t v22;	// L689
  v22 = 0;	// L690
  l_reduction_k22: for (int k22 = 0; k22 < 16; k22++) {	// L691
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

void PE_kernel_gemm_7_1(
  hls::stream< int8_t > &v598 /* v598[16] */,
  hls::stream< int8_t > &v599 /* v599[16] */,
  hls::stream< int8_t > &v600 /* v600[16] */,
  hls::stream< int8_t > &v601 /* v601[16] */,
  int32_t v602[16][16],
  int v603,
  int v604
) {	// L718
  #pragma HLS stream variable=v598 depth=17
  #pragma HLS stream variable=v599 depth=17
  #pragma HLS stream variable=v600 depth=17
  #pragma HLS stream variable=v601 depth=17
  #pragma HLS array_partition variable=v602 complete dim=1
  #pragma HLS array_partition variable=v602 complete dim=2

  int32_t v23;	// L720
  v23 = 0;	// L721
  l_reduction_k23: for (int k23 = 0; k23 < 16; k23++) {	// L722
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

void PE_kernel_gemm_8_1(
  hls::stream< int8_t > &v624 /* v624[16] */,
  hls::stream< int8_t > &v625 /* v625[16] */,
  hls::stream< int8_t > &v626 /* v626[16] */,
  hls::stream< int8_t > &v627 /* v627[16] */,
  int32_t v628[16][16],
  int v629,
  int v630
) {	// L749
  #pragma HLS stream variable=v624 depth=17
  #pragma HLS stream variable=v625 depth=17
  #pragma HLS stream variable=v626 depth=17
  #pragma HLS stream variable=v627 depth=17
  #pragma HLS array_partition variable=v628 complete dim=1
  #pragma HLS array_partition variable=v628 complete dim=2

  int32_t v24;	// L751
  v24 = 0;	// L752
  l_reduction_k24: for (int k24 = 0; k24 < 16; k24++) {	// L753
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

void PE_kernel_gemm_9_1(
  hls::stream< int8_t > &v650 /* v650[16] */,
  hls::stream< int8_t > &v651 /* v651[16] */,
  hls::stream< int8_t > &v652 /* v652[16] */,
  hls::stream< int8_t > &v653 /* v653[16] */,
  int32_t v654[16][16],
  int v655,
  int v656
) {	// L780
  #pragma HLS stream variable=v650 depth=17
  #pragma HLS stream variable=v651 depth=17
  #pragma HLS stream variable=v652 depth=17
  #pragma HLS stream variable=v653 depth=17
  #pragma HLS array_partition variable=v654 complete dim=1
  #pragma HLS array_partition variable=v654 complete dim=2

  int32_t v25;	// L782
  v25 = 0;	// L783
  l_reduction_k25: for (int k25 = 0; k25 < 16; k25++) {	// L784
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

void PE_kernel_gemm_10_1(
  hls::stream< int8_t > &v676 /* v676[16] */,
  hls::stream< int8_t > &v677 /* v677[16] */,
  hls::stream< int8_t > &v678 /* v678[16] */,
  hls::stream< int8_t > &v679 /* v679[16] */,
  int32_t v680[16][16],
  int v681,
  int v682
) {	// L811
  #pragma HLS stream variable=v676 depth=17
  #pragma HLS stream variable=v677 depth=17
  #pragma HLS stream variable=v678 depth=17
  #pragma HLS stream variable=v679 depth=17
  #pragma HLS array_partition variable=v680 complete dim=1
  #pragma HLS array_partition variable=v680 complete dim=2

  int32_t v26;	// L813
  v26 = 0;	// L814
  l_reduction_k26: for (int k26 = 0; k26 < 16; k26++) {	// L815
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

void PE_kernel_gemm_11_1(
  hls::stream< int8_t > &v702 /* v702[16] */,
  hls::stream< int8_t > &v703 /* v703[16] */,
  hls::stream< int8_t > &v704 /* v704[16] */,
  hls::stream< int8_t > &v705 /* v705[16] */,
  int32_t v706[16][16],
  int v707,
  int v708
) {	// L842
  #pragma HLS stream variable=v702 depth=17
  #pragma HLS stream variable=v703 depth=17
  #pragma HLS stream variable=v704 depth=17
  #pragma HLS stream variable=v705 depth=17
  #pragma HLS array_partition variable=v706 complete dim=1
  #pragma HLS array_partition variable=v706 complete dim=2

  int32_t v27;	// L844
  v27 = 0;	// L845
  l_reduction_k27: for (int k27 = 0; k27 < 16; k27++) {	// L846
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

void PE_kernel_gemm_12_1(
  hls::stream< int8_t > &v728 /* v728[16] */,
  hls::stream< int8_t > &v729 /* v729[16] */,
  hls::stream< int8_t > &v730 /* v730[16] */,
  hls::stream< int8_t > &v731 /* v731[16] */,
  int32_t v732[16][16],
  int v733,
  int v734
) {	// L873
  #pragma HLS stream variable=v728 depth=17
  #pragma HLS stream variable=v729 depth=17
  #pragma HLS stream variable=v730 depth=17
  #pragma HLS stream variable=v731 depth=17
  #pragma HLS array_partition variable=v732 complete dim=1
  #pragma HLS array_partition variable=v732 complete dim=2

  int32_t v28;	// L875
  v28 = 0;	// L876
  l_reduction_k28: for (int k28 = 0; k28 < 16; k28++) {	// L877
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

void PE_kernel_gemm_13_1(
  hls::stream< int8_t > &v754 /* v754[16] */,
  hls::stream< int8_t > &v755 /* v755[16] */,
  hls::stream< int8_t > &v756 /* v756[16] */,
  hls::stream< int8_t > &v757 /* v757[16] */,
  int32_t v758[16][16],
  int v759,
  int v760
) {	// L904
  #pragma HLS stream variable=v754 depth=17
  #pragma HLS stream variable=v755 depth=17
  #pragma HLS stream variable=v756 depth=17
  #pragma HLS stream variable=v757 depth=17
  #pragma HLS array_partition variable=v758 complete dim=1
  #pragma HLS array_partition variable=v758 complete dim=2

  int32_t v29;	// L906
  v29 = 0;	// L907
  l_reduction_k29: for (int k29 = 0; k29 < 16; k29++) {	// L908
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

void PE_kernel_gemm_14_1(
  hls::stream< int8_t > &v780 /* v780[16] */,
  hls::stream< int8_t > &v781 /* v781[16] */,
  hls::stream< int8_t > &v782 /* v782[16] */,
  hls::stream< int8_t > &v783 /* v783[16] */,
  int32_t v784[16][16],
  int v785,
  int v786
) {	// L935
  #pragma HLS stream variable=v780 depth=17
  #pragma HLS stream variable=v781 depth=17
  #pragma HLS stream variable=v782 depth=17
  #pragma HLS stream variable=v783 depth=17
  #pragma HLS array_partition variable=v784 complete dim=1
  #pragma HLS array_partition variable=v784 complete dim=2

  int32_t v30;	// L937
  v30 = 0;	// L938
  l_reduction_k30: for (int k30 = 0; k30 < 16; k30++) {	// L939
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

void PE_kernel_gemm_15_1(
  hls::stream< int8_t > &v806 /* v806[16] */,
  hls::stream< int8_t > &v807 /* v807[16] */,
  hls::stream< int8_t > &v808 /* v808[16] */,
  hls::stream< int8_t > &v809 /* v809[16] */,
  int32_t v810[16][16],
  int v811,
  int v812
) {	// L966
  #pragma HLS stream variable=v806 depth=17
  #pragma HLS stream variable=v807 depth=17
  #pragma HLS stream variable=v808 depth=17
  #pragma HLS stream variable=v809 depth=17
  #pragma HLS array_partition variable=v810 complete dim=1
  #pragma HLS array_partition variable=v810 complete dim=2

  int32_t v31;	// L968
  v31 = 0;	// L969
  l_reduction_k31: for (int k31 = 0; k31 < 16; k31++) {	// L970
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

void PE_kernel_gemm_0_2(
  hls::stream< int8_t > &v832 /* v832[16] */,
  hls::stream< int8_t > &v833 /* v833[16] */,
  hls::stream< int8_t > &v834 /* v834[16] */,
  hls::stream< int8_t > &v835 /* v835[16] */,
  int32_t v836[16][16],
  int v837,
  int v838
) {	// L997
  #pragma HLS stream variable=v832 depth=17
  #pragma HLS stream variable=v833 depth=17
  #pragma HLS stream variable=v834 depth=17
  #pragma HLS stream variable=v835 depth=17
  #pragma HLS array_partition variable=v836 complete dim=1
  #pragma HLS array_partition variable=v836 complete dim=2

  int32_t v32;	// L999
  v32 = 0;	// L1000
  l_reduction_k32: for (int k32 = 0; k32 < 16; k32++) {	// L1001
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

void PE_kernel_gemm_1_2(
  hls::stream< int8_t > &v858 /* v858[16] */,
  hls::stream< int8_t > &v859 /* v859[16] */,
  hls::stream< int8_t > &v860 /* v860[16] */,
  hls::stream< int8_t > &v861 /* v861[16] */,
  int32_t v862[16][16],
  int v863,
  int v864
) {	// L1028
  #pragma HLS stream variable=v858 depth=17
  #pragma HLS stream variable=v859 depth=17
  #pragma HLS stream variable=v860 depth=17
  #pragma HLS stream variable=v861 depth=17
  #pragma HLS array_partition variable=v862 complete dim=1
  #pragma HLS array_partition variable=v862 complete dim=2

  int32_t v33;	// L1030
  v33 = 0;	// L1031
  l_reduction_k33: for (int k33 = 0; k33 < 16; k33++) {	// L1032
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

void PE_kernel_gemm_2_2(
  hls::stream< int8_t > &v884 /* v884[16] */,
  hls::stream< int8_t > &v885 /* v885[16] */,
  hls::stream< int8_t > &v886 /* v886[16] */,
  hls::stream< int8_t > &v887 /* v887[16] */,
  int32_t v888[16][16],
  int v889,
  int v890
) {	// L1059
  #pragma HLS stream variable=v884 depth=17
  #pragma HLS stream variable=v885 depth=17
  #pragma HLS stream variable=v886 depth=17
  #pragma HLS stream variable=v887 depth=17
  #pragma HLS array_partition variable=v888 complete dim=1
  #pragma HLS array_partition variable=v888 complete dim=2

  int32_t v34;	// L1061
  v34 = 0;	// L1062
  l_reduction_k34: for (int k34 = 0; k34 < 16; k34++) {	// L1063
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

void PE_kernel_gemm_3_2(
  hls::stream< int8_t > &v910 /* v910[16] */,
  hls::stream< int8_t > &v911 /* v911[16] */,
  hls::stream< int8_t > &v912 /* v912[16] */,
  hls::stream< int8_t > &v913 /* v913[16] */,
  int32_t v914[16][16],
  int v915,
  int v916
) {	// L1090
  #pragma HLS stream variable=v910 depth=17
  #pragma HLS stream variable=v911 depth=17
  #pragma HLS stream variable=v912 depth=17
  #pragma HLS stream variable=v913 depth=17
  #pragma HLS array_partition variable=v914 complete dim=1
  #pragma HLS array_partition variable=v914 complete dim=2

  int32_t v35;	// L1092
  v35 = 0;	// L1093
  l_reduction_k35: for (int k35 = 0; k35 < 16; k35++) {	// L1094
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

void PE_kernel_gemm_4_2(
  hls::stream< int8_t > &v936 /* v936[16] */,
  hls::stream< int8_t > &v937 /* v937[16] */,
  hls::stream< int8_t > &v938 /* v938[16] */,
  hls::stream< int8_t > &v939 /* v939[16] */,
  int32_t v940[16][16],
  int v941,
  int v942
) {	// L1121
  #pragma HLS stream variable=v936 depth=17
  #pragma HLS stream variable=v937 depth=17
  #pragma HLS stream variable=v938 depth=17
  #pragma HLS stream variable=v939 depth=17
  #pragma HLS array_partition variable=v940 complete dim=1
  #pragma HLS array_partition variable=v940 complete dim=2

  int32_t v36;	// L1123
  v36 = 0;	// L1124
  l_reduction_k36: for (int k36 = 0; k36 < 16; k36++) {	// L1125
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

void PE_kernel_gemm_5_2(
  hls::stream< int8_t > &v962 /* v962[16] */,
  hls::stream< int8_t > &v963 /* v963[16] */,
  hls::stream< int8_t > &v964 /* v964[16] */,
  hls::stream< int8_t > &v965 /* v965[16] */,
  int32_t v966[16][16],
  int v967,
  int v968
) {	// L1152
  #pragma HLS stream variable=v962 depth=17
  #pragma HLS stream variable=v963 depth=17
  #pragma HLS stream variable=v964 depth=17
  #pragma HLS stream variable=v965 depth=17
  #pragma HLS array_partition variable=v966 complete dim=1
  #pragma HLS array_partition variable=v966 complete dim=2

  int32_t v37;	// L1154
  v37 = 0;	// L1155
  l_reduction_k37: for (int k37 = 0; k37 < 16; k37++) {	// L1156
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

void PE_kernel_gemm_6_2(
  hls::stream< int8_t > &v988 /* v988[16] */,
  hls::stream< int8_t > &v989 /* v989[16] */,
  hls::stream< int8_t > &v990 /* v990[16] */,
  hls::stream< int8_t > &v991 /* v991[16] */,
  int32_t v992[16][16],
  int v993,
  int v994
) {	// L1183
  #pragma HLS stream variable=v988 depth=17
  #pragma HLS stream variable=v989 depth=17
  #pragma HLS stream variable=v990 depth=17
  #pragma HLS stream variable=v991 depth=17
  #pragma HLS array_partition variable=v992 complete dim=1
  #pragma HLS array_partition variable=v992 complete dim=2

  int32_t v38;	// L1185
  v38 = 0;	// L1186
  l_reduction_k38: for (int k38 = 0; k38 < 16; k38++) {	// L1187
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

void PE_kernel_gemm_7_2(
  hls::stream< int8_t > &v1014 /* v1014[16] */,
  hls::stream< int8_t > &v1015 /* v1015[16] */,
  hls::stream< int8_t > &v1016 /* v1016[16] */,
  hls::stream< int8_t > &v1017 /* v1017[16] */,
  int32_t v1018[16][16],
  int v1019,
  int v1020
) {	// L1214
  #pragma HLS stream variable=v1014 depth=17
  #pragma HLS stream variable=v1015 depth=17
  #pragma HLS stream variable=v1016 depth=17
  #pragma HLS stream variable=v1017 depth=17
  #pragma HLS array_partition variable=v1018 complete dim=1
  #pragma HLS array_partition variable=v1018 complete dim=2

  int32_t v39;	// L1216
  v39 = 0;	// L1217
  l_reduction_k39: for (int k39 = 0; k39 < 16; k39++) {	// L1218
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

void PE_kernel_gemm_8_2(
  hls::stream< int8_t > &v1040 /* v1040[16] */,
  hls::stream< int8_t > &v1041 /* v1041[16] */,
  hls::stream< int8_t > &v1042 /* v1042[16] */,
  hls::stream< int8_t > &v1043 /* v1043[16] */,
  int32_t v1044[16][16],
  int v1045,
  int v1046
) {	// L1245
  #pragma HLS stream variable=v1040 depth=17
  #pragma HLS stream variable=v1041 depth=17
  #pragma HLS stream variable=v1042 depth=17
  #pragma HLS stream variable=v1043 depth=17
  #pragma HLS array_partition variable=v1044 complete dim=1
  #pragma HLS array_partition variable=v1044 complete dim=2

  int32_t v40;	// L1247
  v40 = 0;	// L1248
  l_reduction_k40: for (int k40 = 0; k40 < 16; k40++) {	// L1249
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

void PE_kernel_gemm_9_2(
  hls::stream< int8_t > &v1066 /* v1066[16] */,
  hls::stream< int8_t > &v1067 /* v1067[16] */,
  hls::stream< int8_t > &v1068 /* v1068[16] */,
  hls::stream< int8_t > &v1069 /* v1069[16] */,
  int32_t v1070[16][16],
  int v1071,
  int v1072
) {	// L1276
  #pragma HLS stream variable=v1066 depth=17
  #pragma HLS stream variable=v1067 depth=17
  #pragma HLS stream variable=v1068 depth=17
  #pragma HLS stream variable=v1069 depth=17
  #pragma HLS array_partition variable=v1070 complete dim=1
  #pragma HLS array_partition variable=v1070 complete dim=2

  int32_t v41;	// L1278
  v41 = 0;	// L1279
  l_reduction_k41: for (int k41 = 0; k41 < 16; k41++) {	// L1280
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

void PE_kernel_gemm_10_2(
  hls::stream< int8_t > &v1092 /* v1092[16] */,
  hls::stream< int8_t > &v1093 /* v1093[16] */,
  hls::stream< int8_t > &v1094 /* v1094[16] */,
  hls::stream< int8_t > &v1095 /* v1095[16] */,
  int32_t v1096[16][16],
  int v1097,
  int v1098
) {	// L1307
  #pragma HLS stream variable=v1092 depth=17
  #pragma HLS stream variable=v1093 depth=17
  #pragma HLS stream variable=v1094 depth=17
  #pragma HLS stream variable=v1095 depth=17
  #pragma HLS array_partition variable=v1096 complete dim=1
  #pragma HLS array_partition variable=v1096 complete dim=2

  int32_t v42;	// L1309
  v42 = 0;	// L1310
  l_reduction_k42: for (int k42 = 0; k42 < 16; k42++) {	// L1311
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

void PE_kernel_gemm_11_2(
  hls::stream< int8_t > &v1118 /* v1118[16] */,
  hls::stream< int8_t > &v1119 /* v1119[16] */,
  hls::stream< int8_t > &v1120 /* v1120[16] */,
  hls::stream< int8_t > &v1121 /* v1121[16] */,
  int32_t v1122[16][16],
  int v1123,
  int v1124
) {	// L1338
  #pragma HLS stream variable=v1118 depth=17
  #pragma HLS stream variable=v1119 depth=17
  #pragma HLS stream variable=v1120 depth=17
  #pragma HLS stream variable=v1121 depth=17
  #pragma HLS array_partition variable=v1122 complete dim=1
  #pragma HLS array_partition variable=v1122 complete dim=2

  int32_t v43;	// L1340
  v43 = 0;	// L1341
  l_reduction_k43: for (int k43 = 0; k43 < 16; k43++) {	// L1342
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

void PE_kernel_gemm_12_2(
  hls::stream< int8_t > &v1144 /* v1144[16] */,
  hls::stream< int8_t > &v1145 /* v1145[16] */,
  hls::stream< int8_t > &v1146 /* v1146[16] */,
  hls::stream< int8_t > &v1147 /* v1147[16] */,
  int32_t v1148[16][16],
  int v1149,
  int v1150
) {	// L1369
  #pragma HLS stream variable=v1144 depth=17
  #pragma HLS stream variable=v1145 depth=17
  #pragma HLS stream variable=v1146 depth=17
  #pragma HLS stream variable=v1147 depth=17
  #pragma HLS array_partition variable=v1148 complete dim=1
  #pragma HLS array_partition variable=v1148 complete dim=2

  int32_t v44;	// L1371
  v44 = 0;	// L1372
  l_reduction_k44: for (int k44 = 0; k44 < 16; k44++) {	// L1373
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

void PE_kernel_gemm_13_2(
  hls::stream< int8_t > &v1170 /* v1170[16] */,
  hls::stream< int8_t > &v1171 /* v1171[16] */,
  hls::stream< int8_t > &v1172 /* v1172[16] */,
  hls::stream< int8_t > &v1173 /* v1173[16] */,
  int32_t v1174[16][16],
  int v1175,
  int v1176
) {	// L1400
  #pragma HLS stream variable=v1170 depth=17
  #pragma HLS stream variable=v1171 depth=17
  #pragma HLS stream variable=v1172 depth=17
  #pragma HLS stream variable=v1173 depth=17
  #pragma HLS array_partition variable=v1174 complete dim=1
  #pragma HLS array_partition variable=v1174 complete dim=2

  int32_t v45;	// L1402
  v45 = 0;	// L1403
  l_reduction_k45: for (int k45 = 0; k45 < 16; k45++) {	// L1404
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

void PE_kernel_gemm_14_2(
  hls::stream< int8_t > &v1196 /* v1196[16] */,
  hls::stream< int8_t > &v1197 /* v1197[16] */,
  hls::stream< int8_t > &v1198 /* v1198[16] */,
  hls::stream< int8_t > &v1199 /* v1199[16] */,
  int32_t v1200[16][16],
  int v1201,
  int v1202
) {	// L1431
  #pragma HLS stream variable=v1196 depth=17
  #pragma HLS stream variable=v1197 depth=17
  #pragma HLS stream variable=v1198 depth=17
  #pragma HLS stream variable=v1199 depth=17
  #pragma HLS array_partition variable=v1200 complete dim=1
  #pragma HLS array_partition variable=v1200 complete dim=2

  int32_t v46;	// L1433
  v46 = 0;	// L1434
  l_reduction_k46: for (int k46 = 0; k46 < 16; k46++) {	// L1435
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

void PE_kernel_gemm_15_2(
  hls::stream< int8_t > &v1222 /* v1222[16] */,
  hls::stream< int8_t > &v1223 /* v1223[16] */,
  hls::stream< int8_t > &v1224 /* v1224[16] */,
  hls::stream< int8_t > &v1225 /* v1225[16] */,
  int32_t v1226[16][16],
  int v1227,
  int v1228
) {	// L1462
  #pragma HLS stream variable=v1222 depth=17
  #pragma HLS stream variable=v1223 depth=17
  #pragma HLS stream variable=v1224 depth=17
  #pragma HLS stream variable=v1225 depth=17
  #pragma HLS array_partition variable=v1226 complete dim=1
  #pragma HLS array_partition variable=v1226 complete dim=2

  int32_t v47;	// L1464
  v47 = 0;	// L1465
  l_reduction_k47: for (int k47 = 0; k47 < 16; k47++) {	// L1466
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

void PE_kernel_gemm_0_3(
  hls::stream< int8_t > &v1248 /* v1248[16] */,
  hls::stream< int8_t > &v1249 /* v1249[16] */,
  hls::stream< int8_t > &v1250 /* v1250[16] */,
  hls::stream< int8_t > &v1251 /* v1251[16] */,
  int32_t v1252[16][16],
  int v1253,
  int v1254
) {	// L1493
  #pragma HLS stream variable=v1248 depth=17
  #pragma HLS stream variable=v1249 depth=17
  #pragma HLS stream variable=v1250 depth=17
  #pragma HLS stream variable=v1251 depth=17
  #pragma HLS array_partition variable=v1252 complete dim=1
  #pragma HLS array_partition variable=v1252 complete dim=2

  int32_t v48;	// L1495
  v48 = 0;	// L1496
  l_reduction_k48: for (int k48 = 0; k48 < 16; k48++) {	// L1497
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

void PE_kernel_gemm_1_3(
  hls::stream< int8_t > &v1274 /* v1274[16] */,
  hls::stream< int8_t > &v1275 /* v1275[16] */,
  hls::stream< int8_t > &v1276 /* v1276[16] */,
  hls::stream< int8_t > &v1277 /* v1277[16] */,
  int32_t v1278[16][16],
  int v1279,
  int v1280
) {	// L1524
  #pragma HLS stream variable=v1274 depth=17
  #pragma HLS stream variable=v1275 depth=17
  #pragma HLS stream variable=v1276 depth=17
  #pragma HLS stream variable=v1277 depth=17
  #pragma HLS array_partition variable=v1278 complete dim=1
  #pragma HLS array_partition variable=v1278 complete dim=2

  int32_t v49;	// L1526
  v49 = 0;	// L1527
  l_reduction_k49: for (int k49 = 0; k49 < 16; k49++) {	// L1528
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

void PE_kernel_gemm_2_3(
  hls::stream< int8_t > &v1300 /* v1300[16] */,
  hls::stream< int8_t > &v1301 /* v1301[16] */,
  hls::stream< int8_t > &v1302 /* v1302[16] */,
  hls::stream< int8_t > &v1303 /* v1303[16] */,
  int32_t v1304[16][16],
  int v1305,
  int v1306
) {	// L1555
  #pragma HLS stream variable=v1300 depth=17
  #pragma HLS stream variable=v1301 depth=17
  #pragma HLS stream variable=v1302 depth=17
  #pragma HLS stream variable=v1303 depth=17
  #pragma HLS array_partition variable=v1304 complete dim=1
  #pragma HLS array_partition variable=v1304 complete dim=2

  int32_t v50;	// L1557
  v50 = 0;	// L1558
  l_reduction_k50: for (int k50 = 0; k50 < 16; k50++) {	// L1559
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

void PE_kernel_gemm_3_3(
  hls::stream< int8_t > &v1326 /* v1326[16] */,
  hls::stream< int8_t > &v1327 /* v1327[16] */,
  hls::stream< int8_t > &v1328 /* v1328[16] */,
  hls::stream< int8_t > &v1329 /* v1329[16] */,
  int32_t v1330[16][16],
  int v1331,
  int v1332
) {	// L1586
  #pragma HLS stream variable=v1326 depth=17
  #pragma HLS stream variable=v1327 depth=17
  #pragma HLS stream variable=v1328 depth=17
  #pragma HLS stream variable=v1329 depth=17
  #pragma HLS array_partition variable=v1330 complete dim=1
  #pragma HLS array_partition variable=v1330 complete dim=2

  int32_t v51;	// L1588
  v51 = 0;	// L1589
  l_reduction_k51: for (int k51 = 0; k51 < 16; k51++) {	// L1590
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

void PE_kernel_gemm_4_3(
  hls::stream< int8_t > &v1352 /* v1352[16] */,
  hls::stream< int8_t > &v1353 /* v1353[16] */,
  hls::stream< int8_t > &v1354 /* v1354[16] */,
  hls::stream< int8_t > &v1355 /* v1355[16] */,
  int32_t v1356[16][16],
  int v1357,
  int v1358
) {	// L1617
  #pragma HLS stream variable=v1352 depth=17
  #pragma HLS stream variable=v1353 depth=17
  #pragma HLS stream variable=v1354 depth=17
  #pragma HLS stream variable=v1355 depth=17
  #pragma HLS array_partition variable=v1356 complete dim=1
  #pragma HLS array_partition variable=v1356 complete dim=2

  int32_t v52;	// L1619
  v52 = 0;	// L1620
  l_reduction_k52: for (int k52 = 0; k52 < 16; k52++) {	// L1621
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

void PE_kernel_gemm_5_3(
  hls::stream< int8_t > &v1378 /* v1378[16] */,
  hls::stream< int8_t > &v1379 /* v1379[16] */,
  hls::stream< int8_t > &v1380 /* v1380[16] */,
  hls::stream< int8_t > &v1381 /* v1381[16] */,
  int32_t v1382[16][16],
  int v1383,
  int v1384
) {	// L1648
  #pragma HLS stream variable=v1378 depth=17
  #pragma HLS stream variable=v1379 depth=17
  #pragma HLS stream variable=v1380 depth=17
  #pragma HLS stream variable=v1381 depth=17
  #pragma HLS array_partition variable=v1382 complete dim=1
  #pragma HLS array_partition variable=v1382 complete dim=2

  int32_t v53;	// L1650
  v53 = 0;	// L1651
  l_reduction_k53: for (int k53 = 0; k53 < 16; k53++) {	// L1652
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

void PE_kernel_gemm_6_3(
  hls::stream< int8_t > &v1404 /* v1404[16] */,
  hls::stream< int8_t > &v1405 /* v1405[16] */,
  hls::stream< int8_t > &v1406 /* v1406[16] */,
  hls::stream< int8_t > &v1407 /* v1407[16] */,
  int32_t v1408[16][16],
  int v1409,
  int v1410
) {	// L1679
  #pragma HLS stream variable=v1404 depth=17
  #pragma HLS stream variable=v1405 depth=17
  #pragma HLS stream variable=v1406 depth=17
  #pragma HLS stream variable=v1407 depth=17
  #pragma HLS array_partition variable=v1408 complete dim=1
  #pragma HLS array_partition variable=v1408 complete dim=2

  int32_t v54;	// L1681
  v54 = 0;	// L1682
  l_reduction_k54: for (int k54 = 0; k54 < 16; k54++) {	// L1683
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

void PE_kernel_gemm_7_3(
  hls::stream< int8_t > &v1430 /* v1430[16] */,
  hls::stream< int8_t > &v1431 /* v1431[16] */,
  hls::stream< int8_t > &v1432 /* v1432[16] */,
  hls::stream< int8_t > &v1433 /* v1433[16] */,
  int32_t v1434[16][16],
  int v1435,
  int v1436
) {	// L1710
  #pragma HLS stream variable=v1430 depth=17
  #pragma HLS stream variable=v1431 depth=17
  #pragma HLS stream variable=v1432 depth=17
  #pragma HLS stream variable=v1433 depth=17
  #pragma HLS array_partition variable=v1434 complete dim=1
  #pragma HLS array_partition variable=v1434 complete dim=2

  int32_t v55;	// L1712
  v55 = 0;	// L1713
  l_reduction_k55: for (int k55 = 0; k55 < 16; k55++) {	// L1714
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

void PE_kernel_gemm_8_3(
  hls::stream< int8_t > &v1456 /* v1456[16] */,
  hls::stream< int8_t > &v1457 /* v1457[16] */,
  hls::stream< int8_t > &v1458 /* v1458[16] */,
  hls::stream< int8_t > &v1459 /* v1459[16] */,
  int32_t v1460[16][16],
  int v1461,
  int v1462
) {	// L1741
  #pragma HLS stream variable=v1456 depth=17
  #pragma HLS stream variable=v1457 depth=17
  #pragma HLS stream variable=v1458 depth=17
  #pragma HLS stream variable=v1459 depth=17
  #pragma HLS array_partition variable=v1460 complete dim=1
  #pragma HLS array_partition variable=v1460 complete dim=2

  int32_t v56;	// L1743
  v56 = 0;	// L1744
  l_reduction_k56: for (int k56 = 0; k56 < 16; k56++) {	// L1745
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

void PE_kernel_gemm_9_3(
  hls::stream< int8_t > &v1482 /* v1482[16] */,
  hls::stream< int8_t > &v1483 /* v1483[16] */,
  hls::stream< int8_t > &v1484 /* v1484[16] */,
  hls::stream< int8_t > &v1485 /* v1485[16] */,
  int32_t v1486[16][16],
  int v1487,
  int v1488
) {	// L1772
  #pragma HLS stream variable=v1482 depth=17
  #pragma HLS stream variable=v1483 depth=17
  #pragma HLS stream variable=v1484 depth=17
  #pragma HLS stream variable=v1485 depth=17
  #pragma HLS array_partition variable=v1486 complete dim=1
  #pragma HLS array_partition variable=v1486 complete dim=2

  int32_t v57;	// L1774
  v57 = 0;	// L1775
  l_reduction_k57: for (int k57 = 0; k57 < 16; k57++) {	// L1776
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

void PE_kernel_gemm_10_3(
  hls::stream< int8_t > &v1508 /* v1508[16] */,
  hls::stream< int8_t > &v1509 /* v1509[16] */,
  hls::stream< int8_t > &v1510 /* v1510[16] */,
  hls::stream< int8_t > &v1511 /* v1511[16] */,
  int32_t v1512[16][16],
  int v1513,
  int v1514
) {	// L1803
  #pragma HLS stream variable=v1508 depth=17
  #pragma HLS stream variable=v1509 depth=17
  #pragma HLS stream variable=v1510 depth=17
  #pragma HLS stream variable=v1511 depth=17
  #pragma HLS array_partition variable=v1512 complete dim=1
  #pragma HLS array_partition variable=v1512 complete dim=2

  int32_t v58;	// L1805
  v58 = 0;	// L1806
  l_reduction_k58: for (int k58 = 0; k58 < 16; k58++) {	// L1807
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

void PE_kernel_gemm_11_3(
  hls::stream< int8_t > &v1534 /* v1534[16] */,
  hls::stream< int8_t > &v1535 /* v1535[16] */,
  hls::stream< int8_t > &v1536 /* v1536[16] */,
  hls::stream< int8_t > &v1537 /* v1537[16] */,
  int32_t v1538[16][16],
  int v1539,
  int v1540
) {	// L1834
  #pragma HLS stream variable=v1534 depth=17
  #pragma HLS stream variable=v1535 depth=17
  #pragma HLS stream variable=v1536 depth=17
  #pragma HLS stream variable=v1537 depth=17
  #pragma HLS array_partition variable=v1538 complete dim=1
  #pragma HLS array_partition variable=v1538 complete dim=2

  int32_t v59;	// L1836
  v59 = 0;	// L1837
  l_reduction_k59: for (int k59 = 0; k59 < 16; k59++) {	// L1838
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

void PE_kernel_gemm_12_3(
  hls::stream< int8_t > &v1560 /* v1560[16] */,
  hls::stream< int8_t > &v1561 /* v1561[16] */,
  hls::stream< int8_t > &v1562 /* v1562[16] */,
  hls::stream< int8_t > &v1563 /* v1563[16] */,
  int32_t v1564[16][16],
  int v1565,
  int v1566
) {	// L1865
  #pragma HLS stream variable=v1560 depth=17
  #pragma HLS stream variable=v1561 depth=17
  #pragma HLS stream variable=v1562 depth=17
  #pragma HLS stream variable=v1563 depth=17
  #pragma HLS array_partition variable=v1564 complete dim=1
  #pragma HLS array_partition variable=v1564 complete dim=2

  int32_t v60;	// L1867
  v60 = 0;	// L1868
  l_reduction_k60: for (int k60 = 0; k60 < 16; k60++) {	// L1869
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

void PE_kernel_gemm_13_3(
  hls::stream< int8_t > &v1586 /* v1586[16] */,
  hls::stream< int8_t > &v1587 /* v1587[16] */,
  hls::stream< int8_t > &v1588 /* v1588[16] */,
  hls::stream< int8_t > &v1589 /* v1589[16] */,
  int32_t v1590[16][16],
  int v1591,
  int v1592
) {	// L1896
  #pragma HLS stream variable=v1586 depth=17
  #pragma HLS stream variable=v1587 depth=17
  #pragma HLS stream variable=v1588 depth=17
  #pragma HLS stream variable=v1589 depth=17
  #pragma HLS array_partition variable=v1590 complete dim=1
  #pragma HLS array_partition variable=v1590 complete dim=2

  int32_t v61;	// L1898
  v61 = 0;	// L1899
  l_reduction_k61: for (int k61 = 0; k61 < 16; k61++) {	// L1900
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

void PE_kernel_gemm_14_3(
  hls::stream< int8_t > &v1612 /* v1612[16] */,
  hls::stream< int8_t > &v1613 /* v1613[16] */,
  hls::stream< int8_t > &v1614 /* v1614[16] */,
  hls::stream< int8_t > &v1615 /* v1615[16] */,
  int32_t v1616[16][16],
  int v1617,
  int v1618
) {	// L1927
  #pragma HLS stream variable=v1612 depth=17
  #pragma HLS stream variable=v1613 depth=17
  #pragma HLS stream variable=v1614 depth=17
  #pragma HLS stream variable=v1615 depth=17
  #pragma HLS array_partition variable=v1616 complete dim=1
  #pragma HLS array_partition variable=v1616 complete dim=2

  int32_t v62;	// L1929
  v62 = 0;	// L1930
  l_reduction_k62: for (int k62 = 0; k62 < 16; k62++) {	// L1931
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

void PE_kernel_gemm_15_3(
  hls::stream< int8_t > &v1638 /* v1638[16] */,
  hls::stream< int8_t > &v1639 /* v1639[16] */,
  hls::stream< int8_t > &v1640 /* v1640[16] */,
  hls::stream< int8_t > &v1641 /* v1641[16] */,
  int32_t v1642[16][16],
  int v1643,
  int v1644
) {	// L1958
  #pragma HLS stream variable=v1638 depth=17
  #pragma HLS stream variable=v1639 depth=17
  #pragma HLS stream variable=v1640 depth=17
  #pragma HLS stream variable=v1641 depth=17
  #pragma HLS array_partition variable=v1642 complete dim=1
  #pragma HLS array_partition variable=v1642 complete dim=2

  int32_t v63;	// L1960
  v63 = 0;	// L1961
  l_reduction_k63: for (int k63 = 0; k63 < 16; k63++) {	// L1962
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

void PE_kernel_gemm_0_4(
  hls::stream< int8_t > &v1664 /* v1664[16] */,
  hls::stream< int8_t > &v1665 /* v1665[16] */,
  hls::stream< int8_t > &v1666 /* v1666[16] */,
  hls::stream< int8_t > &v1667 /* v1667[16] */,
  int32_t v1668[16][16],
  int v1669,
  int v1670
) {	// L1989
  #pragma HLS stream variable=v1664 depth=17
  #pragma HLS stream variable=v1665 depth=17
  #pragma HLS stream variable=v1666 depth=17
  #pragma HLS stream variable=v1667 depth=17
  #pragma HLS array_partition variable=v1668 complete dim=1
  #pragma HLS array_partition variable=v1668 complete dim=2

  int32_t v64;	// L1991
  v64 = 0;	// L1992
  l_reduction_k64: for (int k64 = 0; k64 < 16; k64++) {	// L1993
  #pragma HLS pipeline II=1
    int8_t v1673 = v1664.read(); // v1664[k64];	// L1994
    int8_t a64;	// L1995
    a64 = v1673;	// L1996
    int8_t v1675 = v1665.read(); // v1665[k64];	// L1997
    int8_t b64;	// L1998
    b64 = v1675;	// L1999
    int8_t v1677 = a64;	// L2000
    int8_t v1678 = b64;	// L2001
    int16_t v1679 = v1677;	// L2002
    int16_t v1680 = v1678;	// L2003
    int16_t v1681 = v1679 * v1680;	// L2004
    int32_t v1682 = v64;	// L2005
    ap_int<33> v1683 = v1682;	// L2006
    ap_int<33> v1684 = v1681;	// L2007
    ap_int<33> v1685 = v1683 + v1684;	// L2008
    int32_t v1686 = v1685;	// L2009
    v64 = v1686;	// L2010
    int8_t v1687 = a64;	// L2011
    v1666.write(v1687); // v1666[k64] = v1687;	// L2012
    int8_t v1688 = b64;	// L2013
    v1667.write(v1688); // v1667[k64] = v1688;	// L2014
  }
  int32_t v1689 = v64;	// L2016
  v1668[v1669][v1670] = v1689;	// L2017
}

void PE_kernel_gemm_1_4(
  hls::stream< int8_t > &v1690 /* v1690[16] */,
  hls::stream< int8_t > &v1691 /* v1691[16] */,
  hls::stream< int8_t > &v1692 /* v1692[16] */,
  hls::stream< int8_t > &v1693 /* v1693[16] */,
  int32_t v1694[16][16],
  int v1695,
  int v1696
) {	// L2020
  #pragma HLS stream variable=v1690 depth=17
  #pragma HLS stream variable=v1691 depth=17
  #pragma HLS stream variable=v1692 depth=17
  #pragma HLS stream variable=v1693 depth=17
  #pragma HLS array_partition variable=v1694 complete dim=1
  #pragma HLS array_partition variable=v1694 complete dim=2

  int32_t v65;	// L2022
  v65 = 0;	// L2023
  l_reduction_k65: for (int k65 = 0; k65 < 16; k65++) {	// L2024
  #pragma HLS pipeline II=1
    int8_t v1699 = v1690.read(); // v1690[k65];	// L2025
    int8_t a65;	// L2026
    a65 = v1699;	// L2027
    int8_t v1701 = v1691.read(); // v1691[k65];	// L2028
    int8_t b65;	// L2029
    b65 = v1701;	// L2030
    int8_t v1703 = a65;	// L2031
    int8_t v1704 = b65;	// L2032
    int16_t v1705 = v1703;	// L2033
    int16_t v1706 = v1704;	// L2034
    int16_t v1707 = v1705 * v1706;	// L2035
    int32_t v1708 = v65;	// L2036
    ap_int<33> v1709 = v1708;	// L2037
    ap_int<33> v1710 = v1707;	// L2038
    ap_int<33> v1711 = v1709 + v1710;	// L2039
    int32_t v1712 = v1711;	// L2040
    v65 = v1712;	// L2041
    int8_t v1713 = a65;	// L2042
    v1692.write(v1713); // v1692[k65] = v1713;	// L2043
    int8_t v1714 = b65;	// L2044
    v1693.write(v1714); // v1693[k65] = v1714;	// L2045
  }
  int32_t v1715 = v65;	// L2047
  v1694[v1695][v1696] = v1715;	// L2048
}

void PE_kernel_gemm_2_4(
  hls::stream< int8_t > &v1716 /* v1716[16] */,
  hls::stream< int8_t > &v1717 /* v1717[16] */,
  hls::stream< int8_t > &v1718 /* v1718[16] */,
  hls::stream< int8_t > &v1719 /* v1719[16] */,
  int32_t v1720[16][16],
  int v1721,
  int v1722
) {	// L2051
  #pragma HLS stream variable=v1716 depth=17
  #pragma HLS stream variable=v1717 depth=17
  #pragma HLS stream variable=v1718 depth=17
  #pragma HLS stream variable=v1719 depth=17
  #pragma HLS array_partition variable=v1720 complete dim=1
  #pragma HLS array_partition variable=v1720 complete dim=2

  int32_t v66;	// L2053
  v66 = 0;	// L2054
  l_reduction_k66: for (int k66 = 0; k66 < 16; k66++) {	// L2055
  #pragma HLS pipeline II=1
    int8_t v1725 = v1716.read(); // v1716[k66];	// L2056
    int8_t a66;	// L2057
    a66 = v1725;	// L2058
    int8_t v1727 = v1717.read(); // v1717[k66];	// L2059
    int8_t b66;	// L2060
    b66 = v1727;	// L2061
    int8_t v1729 = a66;	// L2062
    int8_t v1730 = b66;	// L2063
    int16_t v1731 = v1729;	// L2064
    int16_t v1732 = v1730;	// L2065
    int16_t v1733 = v1731 * v1732;	// L2066
    int32_t v1734 = v66;	// L2067
    ap_int<33> v1735 = v1734;	// L2068
    ap_int<33> v1736 = v1733;	// L2069
    ap_int<33> v1737 = v1735 + v1736;	// L2070
    int32_t v1738 = v1737;	// L2071
    v66 = v1738;	// L2072
    int8_t v1739 = a66;	// L2073
    v1718.write(v1739); // v1718[k66] = v1739;	// L2074
    int8_t v1740 = b66;	// L2075
    v1719.write(v1740); // v1719[k66] = v1740;	// L2076
  }
  int32_t v1741 = v66;	// L2078
  v1720[v1721][v1722] = v1741;	// L2079
}

void PE_kernel_gemm_3_4(
  hls::stream< int8_t > &v1742 /* v1742[16] */,
  hls::stream< int8_t > &v1743 /* v1743[16] */,
  hls::stream< int8_t > &v1744 /* v1744[16] */,
  hls::stream< int8_t > &v1745 /* v1745[16] */,
  int32_t v1746[16][16],
  int v1747,
  int v1748
) {	// L2082
  #pragma HLS stream variable=v1742 depth=17
  #pragma HLS stream variable=v1743 depth=17
  #pragma HLS stream variable=v1744 depth=17
  #pragma HLS stream variable=v1745 depth=17
  #pragma HLS array_partition variable=v1746 complete dim=1
  #pragma HLS array_partition variable=v1746 complete dim=2

  int32_t v67;	// L2084
  v67 = 0;	// L2085
  l_reduction_k67: for (int k67 = 0; k67 < 16; k67++) {	// L2086
  #pragma HLS pipeline II=1
    int8_t v1751 = v1742.read(); // v1742[k67];	// L2087
    int8_t a67;	// L2088
    a67 = v1751;	// L2089
    int8_t v1753 = v1743.read(); // v1743[k67];	// L2090
    int8_t b67;	// L2091
    b67 = v1753;	// L2092
    int8_t v1755 = a67;	// L2093
    int8_t v1756 = b67;	// L2094
    int16_t v1757 = v1755;	// L2095
    int16_t v1758 = v1756;	// L2096
    int16_t v1759 = v1757 * v1758;	// L2097
    int32_t v1760 = v67;	// L2098
    ap_int<33> v1761 = v1760;	// L2099
    ap_int<33> v1762 = v1759;	// L2100
    ap_int<33> v1763 = v1761 + v1762;	// L2101
    int32_t v1764 = v1763;	// L2102
    v67 = v1764;	// L2103
    int8_t v1765 = a67;	// L2104
    v1744.write(v1765); // v1744[k67] = v1765;	// L2105
    int8_t v1766 = b67;	// L2106
    v1745.write(v1766); // v1745[k67] = v1766;	// L2107
  }
  int32_t v1767 = v67;	// L2109
  v1746[v1747][v1748] = v1767;	// L2110
}

void PE_kernel_gemm_4_4(
  hls::stream< int8_t > &v1768 /* v1768[16] */,
  hls::stream< int8_t > &v1769 /* v1769[16] */,
  hls::stream< int8_t > &v1770 /* v1770[16] */,
  hls::stream< int8_t > &v1771 /* v1771[16] */,
  int32_t v1772[16][16],
  int v1773,
  int v1774
) {	// L2113
  #pragma HLS stream variable=v1768 depth=17
  #pragma HLS stream variable=v1769 depth=17
  #pragma HLS stream variable=v1770 depth=17
  #pragma HLS stream variable=v1771 depth=17
  #pragma HLS array_partition variable=v1772 complete dim=1
  #pragma HLS array_partition variable=v1772 complete dim=2

  int32_t v68;	// L2115
  v68 = 0;	// L2116
  l_reduction_k68: for (int k68 = 0; k68 < 16; k68++) {	// L2117
  #pragma HLS pipeline II=1
    int8_t v1777 = v1768.read(); // v1768[k68];	// L2118
    int8_t a68;	// L2119
    a68 = v1777;	// L2120
    int8_t v1779 = v1769.read(); // v1769[k68];	// L2121
    int8_t b68;	// L2122
    b68 = v1779;	// L2123
    int8_t v1781 = a68;	// L2124
    int8_t v1782 = b68;	// L2125
    int16_t v1783 = v1781;	// L2126
    int16_t v1784 = v1782;	// L2127
    int16_t v1785 = v1783 * v1784;	// L2128
    int32_t v1786 = v68;	// L2129
    ap_int<33> v1787 = v1786;	// L2130
    ap_int<33> v1788 = v1785;	// L2131
    ap_int<33> v1789 = v1787 + v1788;	// L2132
    int32_t v1790 = v1789;	// L2133
    v68 = v1790;	// L2134
    int8_t v1791 = a68;	// L2135
    v1770.write(v1791); // v1770[k68] = v1791;	// L2136
    int8_t v1792 = b68;	// L2137
    v1771.write(v1792); // v1771[k68] = v1792;	// L2138
  }
  int32_t v1793 = v68;	// L2140
  v1772[v1773][v1774] = v1793;	// L2141
}

void PE_kernel_gemm_5_4(
  hls::stream< int8_t > &v1794 /* v1794[16] */,
  hls::stream< int8_t > &v1795 /* v1795[16] */,
  hls::stream< int8_t > &v1796 /* v1796[16] */,
  hls::stream< int8_t > &v1797 /* v1797[16] */,
  int32_t v1798[16][16],
  int v1799,
  int v1800
) {	// L2144
  #pragma HLS stream variable=v1794 depth=17
  #pragma HLS stream variable=v1795 depth=17
  #pragma HLS stream variable=v1796 depth=17
  #pragma HLS stream variable=v1797 depth=17
  #pragma HLS array_partition variable=v1798 complete dim=1
  #pragma HLS array_partition variable=v1798 complete dim=2

  int32_t v69;	// L2146
  v69 = 0;	// L2147
  l_reduction_k69: for (int k69 = 0; k69 < 16; k69++) {	// L2148
  #pragma HLS pipeline II=1
    int8_t v1803 = v1794.read(); // v1794[k69];	// L2149
    int8_t a69;	// L2150
    a69 = v1803;	// L2151
    int8_t v1805 = v1795.read(); // v1795[k69];	// L2152
    int8_t b69;	// L2153
    b69 = v1805;	// L2154
    int8_t v1807 = a69;	// L2155
    int8_t v1808 = b69;	// L2156
    int16_t v1809 = v1807;	// L2157
    int16_t v1810 = v1808;	// L2158
    int16_t v1811 = v1809 * v1810;	// L2159
    int32_t v1812 = v69;	// L2160
    ap_int<33> v1813 = v1812;	// L2161
    ap_int<33> v1814 = v1811;	// L2162
    ap_int<33> v1815 = v1813 + v1814;	// L2163
    int32_t v1816 = v1815;	// L2164
    v69 = v1816;	// L2165
    int8_t v1817 = a69;	// L2166
    v1796.write(v1817); // v1796[k69] = v1817;	// L2167
    int8_t v1818 = b69;	// L2168
    v1797.write(v1818); // v1797[k69] = v1818;	// L2169
  }
  int32_t v1819 = v69;	// L2171
  v1798[v1799][v1800] = v1819;	// L2172
}

void PE_kernel_gemm_6_4(
  hls::stream< int8_t > &v1820 /* v1820[16] */,
  hls::stream< int8_t > &v1821 /* v1821[16] */,
  hls::stream< int8_t > &v1822 /* v1822[16] */,
  hls::stream< int8_t > &v1823 /* v1823[16] */,
  int32_t v1824[16][16],
  int v1825,
  int v1826
) {	// L2175
  #pragma HLS stream variable=v1820 depth=17
  #pragma HLS stream variable=v1821 depth=17
  #pragma HLS stream variable=v1822 depth=17
  #pragma HLS stream variable=v1823 depth=17
  #pragma HLS array_partition variable=v1824 complete dim=1
  #pragma HLS array_partition variable=v1824 complete dim=2

  int32_t v70;	// L2177
  v70 = 0;	// L2178
  l_reduction_k70: for (int k70 = 0; k70 < 16; k70++) {	// L2179
  #pragma HLS pipeline II=1
    int8_t v1829 = v1820.read(); // v1820[k70];	// L2180
    int8_t a70;	// L2181
    a70 = v1829;	// L2182
    int8_t v1831 = v1821.read(); // v1821[k70];	// L2183
    int8_t b70;	// L2184
    b70 = v1831;	// L2185
    int8_t v1833 = a70;	// L2186
    int8_t v1834 = b70;	// L2187
    int16_t v1835 = v1833;	// L2188
    int16_t v1836 = v1834;	// L2189
    int16_t v1837 = v1835 * v1836;	// L2190
    int32_t v1838 = v70;	// L2191
    ap_int<33> v1839 = v1838;	// L2192
    ap_int<33> v1840 = v1837;	// L2193
    ap_int<33> v1841 = v1839 + v1840;	// L2194
    int32_t v1842 = v1841;	// L2195
    v70 = v1842;	// L2196
    int8_t v1843 = a70;	// L2197
    v1822.write(v1843); // v1822[k70] = v1843;	// L2198
    int8_t v1844 = b70;	// L2199
    v1823.write(v1844); // v1823[k70] = v1844;	// L2200
  }
  int32_t v1845 = v70;	// L2202
  v1824[v1825][v1826] = v1845;	// L2203
}

void PE_kernel_gemm_7_4(
  hls::stream< int8_t > &v1846 /* v1846[16] */,
  hls::stream< int8_t > &v1847 /* v1847[16] */,
  hls::stream< int8_t > &v1848 /* v1848[16] */,
  hls::stream< int8_t > &v1849 /* v1849[16] */,
  int32_t v1850[16][16],
  int v1851,
  int v1852
) {	// L2206
  #pragma HLS stream variable=v1846 depth=17
  #pragma HLS stream variable=v1847 depth=17
  #pragma HLS stream variable=v1848 depth=17
  #pragma HLS stream variable=v1849 depth=17
  #pragma HLS array_partition variable=v1850 complete dim=1
  #pragma HLS array_partition variable=v1850 complete dim=2

  int32_t v71;	// L2208
  v71 = 0;	// L2209
  l_reduction_k71: for (int k71 = 0; k71 < 16; k71++) {	// L2210
  #pragma HLS pipeline II=1
    int8_t v1855 = v1846.read(); // v1846[k71];	// L2211
    int8_t a71;	// L2212
    a71 = v1855;	// L2213
    int8_t v1857 = v1847.read(); // v1847[k71];	// L2214
    int8_t b71;	// L2215
    b71 = v1857;	// L2216
    int8_t v1859 = a71;	// L2217
    int8_t v1860 = b71;	// L2218
    int16_t v1861 = v1859;	// L2219
    int16_t v1862 = v1860;	// L2220
    int16_t v1863 = v1861 * v1862;	// L2221
    int32_t v1864 = v71;	// L2222
    ap_int<33> v1865 = v1864;	// L2223
    ap_int<33> v1866 = v1863;	// L2224
    ap_int<33> v1867 = v1865 + v1866;	// L2225
    int32_t v1868 = v1867;	// L2226
    v71 = v1868;	// L2227
    int8_t v1869 = a71;	// L2228
    v1848.write(v1869); // v1848[k71] = v1869;	// L2229
    int8_t v1870 = b71;	// L2230
    v1849.write(v1870); // v1849[k71] = v1870;	// L2231
  }
  int32_t v1871 = v71;	// L2233
  v1850[v1851][v1852] = v1871;	// L2234
}

void PE_kernel_gemm_8_4(
  hls::stream< int8_t > &v1872 /* v1872[16] */,
  hls::stream< int8_t > &v1873 /* v1873[16] */,
  hls::stream< int8_t > &v1874 /* v1874[16] */,
  hls::stream< int8_t > &v1875 /* v1875[16] */,
  int32_t v1876[16][16],
  int v1877,
  int v1878
) {	// L2237
  #pragma HLS stream variable=v1872 depth=17
  #pragma HLS stream variable=v1873 depth=17
  #pragma HLS stream variable=v1874 depth=17
  #pragma HLS stream variable=v1875 depth=17
  #pragma HLS array_partition variable=v1876 complete dim=1
  #pragma HLS array_partition variable=v1876 complete dim=2

  int32_t v72;	// L2239
  v72 = 0;	// L2240
  l_reduction_k72: for (int k72 = 0; k72 < 16; k72++) {	// L2241
  #pragma HLS pipeline II=1
    int8_t v1881 = v1872.read(); // v1872[k72];	// L2242
    int8_t a72;	// L2243
    a72 = v1881;	// L2244
    int8_t v1883 = v1873.read(); // v1873[k72];	// L2245
    int8_t b72;	// L2246
    b72 = v1883;	// L2247
    int8_t v1885 = a72;	// L2248
    int8_t v1886 = b72;	// L2249
    int16_t v1887 = v1885;	// L2250
    int16_t v1888 = v1886;	// L2251
    int16_t v1889 = v1887 * v1888;	// L2252
    int32_t v1890 = v72;	// L2253
    ap_int<33> v1891 = v1890;	// L2254
    ap_int<33> v1892 = v1889;	// L2255
    ap_int<33> v1893 = v1891 + v1892;	// L2256
    int32_t v1894 = v1893;	// L2257
    v72 = v1894;	// L2258
    int8_t v1895 = a72;	// L2259
    v1874.write(v1895); // v1874[k72] = v1895;	// L2260
    int8_t v1896 = b72;	// L2261
    v1875.write(v1896); // v1875[k72] = v1896;	// L2262
  }
  int32_t v1897 = v72;	// L2264
  v1876[v1877][v1878] = v1897;	// L2265
}

void PE_kernel_gemm_9_4(
  hls::stream< int8_t > &v1898 /* v1898[16] */,
  hls::stream< int8_t > &v1899 /* v1899[16] */,
  hls::stream< int8_t > &v1900 /* v1900[16] */,
  hls::stream< int8_t > &v1901 /* v1901[16] */,
  int32_t v1902[16][16],
  int v1903,
  int v1904
) {	// L2268
  #pragma HLS stream variable=v1898 depth=17
  #pragma HLS stream variable=v1899 depth=17
  #pragma HLS stream variable=v1900 depth=17
  #pragma HLS stream variable=v1901 depth=17
  #pragma HLS array_partition variable=v1902 complete dim=1
  #pragma HLS array_partition variable=v1902 complete dim=2

  int32_t v73;	// L2270
  v73 = 0;	// L2271
  l_reduction_k73: for (int k73 = 0; k73 < 16; k73++) {	// L2272
  #pragma HLS pipeline II=1
    int8_t v1907 = v1898.read(); // v1898[k73];	// L2273
    int8_t a73;	// L2274
    a73 = v1907;	// L2275
    int8_t v1909 = v1899.read(); // v1899[k73];	// L2276
    int8_t b73;	// L2277
    b73 = v1909;	// L2278
    int8_t v1911 = a73;	// L2279
    int8_t v1912 = b73;	// L2280
    int16_t v1913 = v1911;	// L2281
    int16_t v1914 = v1912;	// L2282
    int16_t v1915 = v1913 * v1914;	// L2283
    int32_t v1916 = v73;	// L2284
    ap_int<33> v1917 = v1916;	// L2285
    ap_int<33> v1918 = v1915;	// L2286
    ap_int<33> v1919 = v1917 + v1918;	// L2287
    int32_t v1920 = v1919;	// L2288
    v73 = v1920;	// L2289
    int8_t v1921 = a73;	// L2290
    v1900.write(v1921); // v1900[k73] = v1921;	// L2291
    int8_t v1922 = b73;	// L2292
    v1901.write(v1922); // v1901[k73] = v1922;	// L2293
  }
  int32_t v1923 = v73;	// L2295
  v1902[v1903][v1904] = v1923;	// L2296
}

void PE_kernel_gemm_10_4(
  hls::stream< int8_t > &v1924 /* v1924[16] */,
  hls::stream< int8_t > &v1925 /* v1925[16] */,
  hls::stream< int8_t > &v1926 /* v1926[16] */,
  hls::stream< int8_t > &v1927 /* v1927[16] */,
  int32_t v1928[16][16],
  int v1929,
  int v1930
) {	// L2299
  #pragma HLS stream variable=v1924 depth=17
  #pragma HLS stream variable=v1925 depth=17
  #pragma HLS stream variable=v1926 depth=17
  #pragma HLS stream variable=v1927 depth=17
  #pragma HLS array_partition variable=v1928 complete dim=1
  #pragma HLS array_partition variable=v1928 complete dim=2

  int32_t v74;	// L2301
  v74 = 0;	// L2302
  l_reduction_k74: for (int k74 = 0; k74 < 16; k74++) {	// L2303
  #pragma HLS pipeline II=1
    int8_t v1933 = v1924.read(); // v1924[k74];	// L2304
    int8_t a74;	// L2305
    a74 = v1933;	// L2306
    int8_t v1935 = v1925.read(); // v1925[k74];	// L2307
    int8_t b74;	// L2308
    b74 = v1935;	// L2309
    int8_t v1937 = a74;	// L2310
    int8_t v1938 = b74;	// L2311
    int16_t v1939 = v1937;	// L2312
    int16_t v1940 = v1938;	// L2313
    int16_t v1941 = v1939 * v1940;	// L2314
    int32_t v1942 = v74;	// L2315
    ap_int<33> v1943 = v1942;	// L2316
    ap_int<33> v1944 = v1941;	// L2317
    ap_int<33> v1945 = v1943 + v1944;	// L2318
    int32_t v1946 = v1945;	// L2319
    v74 = v1946;	// L2320
    int8_t v1947 = a74;	// L2321
    v1926.write(v1947); // v1926[k74] = v1947;	// L2322
    int8_t v1948 = b74;	// L2323
    v1927.write(v1948); // v1927[k74] = v1948;	// L2324
  }
  int32_t v1949 = v74;	// L2326
  v1928[v1929][v1930] = v1949;	// L2327
}

void PE_kernel_gemm_11_4(
  hls::stream< int8_t > &v1950 /* v1950[16] */,
  hls::stream< int8_t > &v1951 /* v1951[16] */,
  hls::stream< int8_t > &v1952 /* v1952[16] */,
  hls::stream< int8_t > &v1953 /* v1953[16] */,
  int32_t v1954[16][16],
  int v1955,
  int v1956
) {	// L2330
  #pragma HLS stream variable=v1950 depth=17
  #pragma HLS stream variable=v1951 depth=17
  #pragma HLS stream variable=v1952 depth=17
  #pragma HLS stream variable=v1953 depth=17
  #pragma HLS array_partition variable=v1954 complete dim=1
  #pragma HLS array_partition variable=v1954 complete dim=2

  int32_t v75;	// L2332
  v75 = 0;	// L2333
  l_reduction_k75: for (int k75 = 0; k75 < 16; k75++) {	// L2334
  #pragma HLS pipeline II=1
    int8_t v1959 = v1950.read(); // v1950[k75];	// L2335
    int8_t a75;	// L2336
    a75 = v1959;	// L2337
    int8_t v1961 = v1951.read(); // v1951[k75];	// L2338
    int8_t b75;	// L2339
    b75 = v1961;	// L2340
    int8_t v1963 = a75;	// L2341
    int8_t v1964 = b75;	// L2342
    int16_t v1965 = v1963;	// L2343
    int16_t v1966 = v1964;	// L2344
    int16_t v1967 = v1965 * v1966;	// L2345
    int32_t v1968 = v75;	// L2346
    ap_int<33> v1969 = v1968;	// L2347
    ap_int<33> v1970 = v1967;	// L2348
    ap_int<33> v1971 = v1969 + v1970;	// L2349
    int32_t v1972 = v1971;	// L2350
    v75 = v1972;	// L2351
    int8_t v1973 = a75;	// L2352
    v1952.write(v1973); // v1952[k75] = v1973;	// L2353
    int8_t v1974 = b75;	// L2354
    v1953.write(v1974); // v1953[k75] = v1974;	// L2355
  }
  int32_t v1975 = v75;	// L2357
  v1954[v1955][v1956] = v1975;	// L2358
}

void PE_kernel_gemm_12_4(
  hls::stream< int8_t > &v1976 /* v1976[16] */,
  hls::stream< int8_t > &v1977 /* v1977[16] */,
  hls::stream< int8_t > &v1978 /* v1978[16] */,
  hls::stream< int8_t > &v1979 /* v1979[16] */,
  int32_t v1980[16][16],
  int v1981,
  int v1982
) {	// L2361
  #pragma HLS stream variable=v1976 depth=17
  #pragma HLS stream variable=v1977 depth=17
  #pragma HLS stream variable=v1978 depth=17
  #pragma HLS stream variable=v1979 depth=17
  #pragma HLS array_partition variable=v1980 complete dim=1
  #pragma HLS array_partition variable=v1980 complete dim=2

  int32_t v76;	// L2363
  v76 = 0;	// L2364
  l_reduction_k76: for (int k76 = 0; k76 < 16; k76++) {	// L2365
  #pragma HLS pipeline II=1
    int8_t v1985 = v1976.read(); // v1976[k76];	// L2366
    int8_t a76;	// L2367
    a76 = v1985;	// L2368
    int8_t v1987 = v1977.read(); // v1977[k76];	// L2369
    int8_t b76;	// L2370
    b76 = v1987;	// L2371
    int8_t v1989 = a76;	// L2372
    int8_t v1990 = b76;	// L2373
    int16_t v1991 = v1989;	// L2374
    int16_t v1992 = v1990;	// L2375
    int16_t v1993 = v1991 * v1992;	// L2376
    int32_t v1994 = v76;	// L2377
    ap_int<33> v1995 = v1994;	// L2378
    ap_int<33> v1996 = v1993;	// L2379
    ap_int<33> v1997 = v1995 + v1996;	// L2380
    int32_t v1998 = v1997;	// L2381
    v76 = v1998;	// L2382
    int8_t v1999 = a76;	// L2383
    v1978.write(v1999); // v1978[k76] = v1999;	// L2384
    int8_t v2000 = b76;	// L2385
    v1979.write(v2000); // v1979[k76] = v2000;	// L2386
  }
  int32_t v2001 = v76;	// L2388
  v1980[v1981][v1982] = v2001;	// L2389
}

void PE_kernel_gemm_13_4(
  hls::stream< int8_t > &v2002 /* v2002[16] */,
  hls::stream< int8_t > &v2003 /* v2003[16] */,
  hls::stream< int8_t > &v2004 /* v2004[16] */,
  hls::stream< int8_t > &v2005 /* v2005[16] */,
  int32_t v2006[16][16],
  int v2007,
  int v2008
) {	// L2392
  #pragma HLS stream variable=v2002 depth=17
  #pragma HLS stream variable=v2003 depth=17
  #pragma HLS stream variable=v2004 depth=17
  #pragma HLS stream variable=v2005 depth=17
  #pragma HLS array_partition variable=v2006 complete dim=1
  #pragma HLS array_partition variable=v2006 complete dim=2

  int32_t v77;	// L2394
  v77 = 0;	// L2395
  l_reduction_k77: for (int k77 = 0; k77 < 16; k77++) {	// L2396
  #pragma HLS pipeline II=1
    int8_t v2011 = v2002.read(); // v2002[k77];	// L2397
    int8_t a77;	// L2398
    a77 = v2011;	// L2399
    int8_t v2013 = v2003.read(); // v2003[k77];	// L2400
    int8_t b77;	// L2401
    b77 = v2013;	// L2402
    int8_t v2015 = a77;	// L2403
    int8_t v2016 = b77;	// L2404
    int16_t v2017 = v2015;	// L2405
    int16_t v2018 = v2016;	// L2406
    int16_t v2019 = v2017 * v2018;	// L2407
    int32_t v2020 = v77;	// L2408
    ap_int<33> v2021 = v2020;	// L2409
    ap_int<33> v2022 = v2019;	// L2410
    ap_int<33> v2023 = v2021 + v2022;	// L2411
    int32_t v2024 = v2023;	// L2412
    v77 = v2024;	// L2413
    int8_t v2025 = a77;	// L2414
    v2004.write(v2025); // v2004[k77] = v2025;	// L2415
    int8_t v2026 = b77;	// L2416
    v2005.write(v2026); // v2005[k77] = v2026;	// L2417
  }
  int32_t v2027 = v77;	// L2419
  v2006[v2007][v2008] = v2027;	// L2420
}

void PE_kernel_gemm_14_4(
  hls::stream< int8_t > &v2028 /* v2028[16] */,
  hls::stream< int8_t > &v2029 /* v2029[16] */,
  hls::stream< int8_t > &v2030 /* v2030[16] */,
  hls::stream< int8_t > &v2031 /* v2031[16] */,
  int32_t v2032[16][16],
  int v2033,
  int v2034
) {	// L2423
  #pragma HLS stream variable=v2028 depth=17
  #pragma HLS stream variable=v2029 depth=17
  #pragma HLS stream variable=v2030 depth=17
  #pragma HLS stream variable=v2031 depth=17
  #pragma HLS array_partition variable=v2032 complete dim=1
  #pragma HLS array_partition variable=v2032 complete dim=2

  int32_t v78;	// L2425
  v78 = 0;	// L2426
  l_reduction_k78: for (int k78 = 0; k78 < 16; k78++) {	// L2427
  #pragma HLS pipeline II=1
    int8_t v2037 = v2028.read(); // v2028[k78];	// L2428
    int8_t a78;	// L2429
    a78 = v2037;	// L2430
    int8_t v2039 = v2029.read(); // v2029[k78];	// L2431
    int8_t b78;	// L2432
    b78 = v2039;	// L2433
    int8_t v2041 = a78;	// L2434
    int8_t v2042 = b78;	// L2435
    int16_t v2043 = v2041;	// L2436
    int16_t v2044 = v2042;	// L2437
    int16_t v2045 = v2043 * v2044;	// L2438
    int32_t v2046 = v78;	// L2439
    ap_int<33> v2047 = v2046;	// L2440
    ap_int<33> v2048 = v2045;	// L2441
    ap_int<33> v2049 = v2047 + v2048;	// L2442
    int32_t v2050 = v2049;	// L2443
    v78 = v2050;	// L2444
    int8_t v2051 = a78;	// L2445
    v2030.write(v2051); // v2030[k78] = v2051;	// L2446
    int8_t v2052 = b78;	// L2447
    v2031.write(v2052); // v2031[k78] = v2052;	// L2448
  }
  int32_t v2053 = v78;	// L2450
  v2032[v2033][v2034] = v2053;	// L2451
}

void PE_kernel_gemm_15_4(
  hls::stream< int8_t > &v2054 /* v2054[16] */,
  hls::stream< int8_t > &v2055 /* v2055[16] */,
  hls::stream< int8_t > &v2056 /* v2056[16] */,
  hls::stream< int8_t > &v2057 /* v2057[16] */,
  int32_t v2058[16][16],
  int v2059,
  int v2060
) {	// L2454
  #pragma HLS stream variable=v2054 depth=17
  #pragma HLS stream variable=v2055 depth=17
  #pragma HLS stream variable=v2056 depth=17
  #pragma HLS stream variable=v2057 depth=17
  #pragma HLS array_partition variable=v2058 complete dim=1
  #pragma HLS array_partition variable=v2058 complete dim=2

  int32_t v79;	// L2456
  v79 = 0;	// L2457
  l_reduction_k79: for (int k79 = 0; k79 < 16; k79++) {	// L2458
  #pragma HLS pipeline II=1
    int8_t v2063 = v2054.read(); // v2054[k79];	// L2459
    int8_t a79;	// L2460
    a79 = v2063;	// L2461
    int8_t v2065 = v2055.read(); // v2055[k79];	// L2462
    int8_t b79;	// L2463
    b79 = v2065;	// L2464
    int8_t v2067 = a79;	// L2465
    int8_t v2068 = b79;	// L2466
    int16_t v2069 = v2067;	// L2467
    int16_t v2070 = v2068;	// L2468
    int16_t v2071 = v2069 * v2070;	// L2469
    int32_t v2072 = v79;	// L2470
    ap_int<33> v2073 = v2072;	// L2471
    ap_int<33> v2074 = v2071;	// L2472
    ap_int<33> v2075 = v2073 + v2074;	// L2473
    int32_t v2076 = v2075;	// L2474
    v79 = v2076;	// L2475
    int8_t v2077 = a79;	// L2476
    v2056.write(v2077); // v2056[k79] = v2077;	// L2477
    int8_t v2078 = b79;	// L2478
    v2057.write(v2078); // v2057[k79] = v2078;	// L2479
  }
  int32_t v2079 = v79;	// L2481
  v2058[v2059][v2060] = v2079;	// L2482
}

void PE_kernel_gemm_0_5(
  hls::stream< int8_t > &v2080 /* v2080[16] */,
  hls::stream< int8_t > &v2081 /* v2081[16] */,
  hls::stream< int8_t > &v2082 /* v2082[16] */,
  hls::stream< int8_t > &v2083 /* v2083[16] */,
  int32_t v2084[16][16],
  int v2085,
  int v2086
) {	// L2485
  #pragma HLS stream variable=v2080 depth=17
  #pragma HLS stream variable=v2081 depth=17
  #pragma HLS stream variable=v2082 depth=17
  #pragma HLS stream variable=v2083 depth=17
  #pragma HLS array_partition variable=v2084 complete dim=1
  #pragma HLS array_partition variable=v2084 complete dim=2

  int32_t v80;	// L2487
  v80 = 0;	// L2488
  l_reduction_k80: for (int k80 = 0; k80 < 16; k80++) {	// L2489
  #pragma HLS pipeline II=1
    int8_t v2089 = v2080.read(); // v2080[k80];	// L2490
    int8_t a80;	// L2491
    a80 = v2089;	// L2492
    int8_t v2091 = v2081.read(); // v2081[k80];	// L2493
    int8_t b80;	// L2494
    b80 = v2091;	// L2495
    int8_t v2093 = a80;	// L2496
    int8_t v2094 = b80;	// L2497
    int16_t v2095 = v2093;	// L2498
    int16_t v2096 = v2094;	// L2499
    int16_t v2097 = v2095 * v2096;	// L2500
    int32_t v2098 = v80;	// L2501
    ap_int<33> v2099 = v2098;	// L2502
    ap_int<33> v2100 = v2097;	// L2503
    ap_int<33> v2101 = v2099 + v2100;	// L2504
    int32_t v2102 = v2101;	// L2505
    v80 = v2102;	// L2506
    int8_t v2103 = a80;	// L2507
    v2082.write(v2103); // v2082[k80] = v2103;	// L2508
    int8_t v2104 = b80;	// L2509
    v2083.write(v2104); // v2083[k80] = v2104;	// L2510
  }
  int32_t v2105 = v80;	// L2512
  v2084[v2085][v2086] = v2105;	// L2513
}

void PE_kernel_gemm_1_5(
  hls::stream< int8_t > &v2106 /* v2106[16] */,
  hls::stream< int8_t > &v2107 /* v2107[16] */,
  hls::stream< int8_t > &v2108 /* v2108[16] */,
  hls::stream< int8_t > &v2109 /* v2109[16] */,
  int32_t v2110[16][16],
  int v2111,
  int v2112
) {	// L2516
  #pragma HLS stream variable=v2106 depth=17
  #pragma HLS stream variable=v2107 depth=17
  #pragma HLS stream variable=v2108 depth=17
  #pragma HLS stream variable=v2109 depth=17
  #pragma HLS array_partition variable=v2110 complete dim=1
  #pragma HLS array_partition variable=v2110 complete dim=2

  int32_t v81;	// L2518
  v81 = 0;	// L2519
  l_reduction_k81: for (int k81 = 0; k81 < 16; k81++) {	// L2520
  #pragma HLS pipeline II=1
    int8_t v2115 = v2106.read(); // v2106[k81];	// L2521
    int8_t a81;	// L2522
    a81 = v2115;	// L2523
    int8_t v2117 = v2107.read(); // v2107[k81];	// L2524
    int8_t b81;	// L2525
    b81 = v2117;	// L2526
    int8_t v2119 = a81;	// L2527
    int8_t v2120 = b81;	// L2528
    int16_t v2121 = v2119;	// L2529
    int16_t v2122 = v2120;	// L2530
    int16_t v2123 = v2121 * v2122;	// L2531
    int32_t v2124 = v81;	// L2532
    ap_int<33> v2125 = v2124;	// L2533
    ap_int<33> v2126 = v2123;	// L2534
    ap_int<33> v2127 = v2125 + v2126;	// L2535
    int32_t v2128 = v2127;	// L2536
    v81 = v2128;	// L2537
    int8_t v2129 = a81;	// L2538
    v2108.write(v2129); // v2108[k81] = v2129;	// L2539
    int8_t v2130 = b81;	// L2540
    v2109.write(v2130); // v2109[k81] = v2130;	// L2541
  }
  int32_t v2131 = v81;	// L2543
  v2110[v2111][v2112] = v2131;	// L2544
}

void PE_kernel_gemm_2_5(
  hls::stream< int8_t > &v2132 /* v2132[16] */,
  hls::stream< int8_t > &v2133 /* v2133[16] */,
  hls::stream< int8_t > &v2134 /* v2134[16] */,
  hls::stream< int8_t > &v2135 /* v2135[16] */,
  int32_t v2136[16][16],
  int v2137,
  int v2138
) {	// L2547
  #pragma HLS stream variable=v2132 depth=17
  #pragma HLS stream variable=v2133 depth=17
  #pragma HLS stream variable=v2134 depth=17
  #pragma HLS stream variable=v2135 depth=17
  #pragma HLS array_partition variable=v2136 complete dim=1
  #pragma HLS array_partition variable=v2136 complete dim=2

  int32_t v82;	// L2549
  v82 = 0;	// L2550
  l_reduction_k82: for (int k82 = 0; k82 < 16; k82++) {	// L2551
  #pragma HLS pipeline II=1
    int8_t v2141 = v2132.read(); // v2132[k82];	// L2552
    int8_t a82;	// L2553
    a82 = v2141;	// L2554
    int8_t v2143 = v2133.read(); // v2133[k82];	// L2555
    int8_t b82;	// L2556
    b82 = v2143;	// L2557
    int8_t v2145 = a82;	// L2558
    int8_t v2146 = b82;	// L2559
    int16_t v2147 = v2145;	// L2560
    int16_t v2148 = v2146;	// L2561
    int16_t v2149 = v2147 * v2148;	// L2562
    int32_t v2150 = v82;	// L2563
    ap_int<33> v2151 = v2150;	// L2564
    ap_int<33> v2152 = v2149;	// L2565
    ap_int<33> v2153 = v2151 + v2152;	// L2566
    int32_t v2154 = v2153;	// L2567
    v82 = v2154;	// L2568
    int8_t v2155 = a82;	// L2569
    v2134.write(v2155); // v2134[k82] = v2155;	// L2570
    int8_t v2156 = b82;	// L2571
    v2135.write(v2156); // v2135[k82] = v2156;	// L2572
  }
  int32_t v2157 = v82;	// L2574
  v2136[v2137][v2138] = v2157;	// L2575
}

void PE_kernel_gemm_3_5(
  hls::stream< int8_t > &v2158 /* v2158[16] */,
  hls::stream< int8_t > &v2159 /* v2159[16] */,
  hls::stream< int8_t > &v2160 /* v2160[16] */,
  hls::stream< int8_t > &v2161 /* v2161[16] */,
  int32_t v2162[16][16],
  int v2163,
  int v2164
) {	// L2578
  #pragma HLS stream variable=v2158 depth=17
  #pragma HLS stream variable=v2159 depth=17
  #pragma HLS stream variable=v2160 depth=17
  #pragma HLS stream variable=v2161 depth=17
  #pragma HLS array_partition variable=v2162 complete dim=1
  #pragma HLS array_partition variable=v2162 complete dim=2

  int32_t v83;	// L2580
  v83 = 0;	// L2581
  l_reduction_k83: for (int k83 = 0; k83 < 16; k83++) {	// L2582
  #pragma HLS pipeline II=1
    int8_t v2167 = v2158.read(); // v2158[k83];	// L2583
    int8_t a83;	// L2584
    a83 = v2167;	// L2585
    int8_t v2169 = v2159.read(); // v2159[k83];	// L2586
    int8_t b83;	// L2587
    b83 = v2169;	// L2588
    int8_t v2171 = a83;	// L2589
    int8_t v2172 = b83;	// L2590
    int16_t v2173 = v2171;	// L2591
    int16_t v2174 = v2172;	// L2592
    int16_t v2175 = v2173 * v2174;	// L2593
    int32_t v2176 = v83;	// L2594
    ap_int<33> v2177 = v2176;	// L2595
    ap_int<33> v2178 = v2175;	// L2596
    ap_int<33> v2179 = v2177 + v2178;	// L2597
    int32_t v2180 = v2179;	// L2598
    v83 = v2180;	// L2599
    int8_t v2181 = a83;	// L2600
    v2160.write(v2181); // v2160[k83] = v2181;	// L2601
    int8_t v2182 = b83;	// L2602
    v2161.write(v2182); // v2161[k83] = v2182;	// L2603
  }
  int32_t v2183 = v83;	// L2605
  v2162[v2163][v2164] = v2183;	// L2606
}

void PE_kernel_gemm_4_5(
  hls::stream< int8_t > &v2184 /* v2184[16] */,
  hls::stream< int8_t > &v2185 /* v2185[16] */,
  hls::stream< int8_t > &v2186 /* v2186[16] */,
  hls::stream< int8_t > &v2187 /* v2187[16] */,
  int32_t v2188[16][16],
  int v2189,
  int v2190
) {	// L2609
  #pragma HLS stream variable=v2184 depth=17
  #pragma HLS stream variable=v2185 depth=17
  #pragma HLS stream variable=v2186 depth=17
  #pragma HLS stream variable=v2187 depth=17
  #pragma HLS array_partition variable=v2188 complete dim=1
  #pragma HLS array_partition variable=v2188 complete dim=2

  int32_t v84;	// L2611
  v84 = 0;	// L2612
  l_reduction_k84: for (int k84 = 0; k84 < 16; k84++) {	// L2613
  #pragma HLS pipeline II=1
    int8_t v2193 = v2184.read(); // v2184[k84];	// L2614
    int8_t a84;	// L2615
    a84 = v2193;	// L2616
    int8_t v2195 = v2185.read(); // v2185[k84];	// L2617
    int8_t b84;	// L2618
    b84 = v2195;	// L2619
    int8_t v2197 = a84;	// L2620
    int8_t v2198 = b84;	// L2621
    int16_t v2199 = v2197;	// L2622
    int16_t v2200 = v2198;	// L2623
    int16_t v2201 = v2199 * v2200;	// L2624
    int32_t v2202 = v84;	// L2625
    ap_int<33> v2203 = v2202;	// L2626
    ap_int<33> v2204 = v2201;	// L2627
    ap_int<33> v2205 = v2203 + v2204;	// L2628
    int32_t v2206 = v2205;	// L2629
    v84 = v2206;	// L2630
    int8_t v2207 = a84;	// L2631
    v2186.write(v2207); // v2186[k84] = v2207;	// L2632
    int8_t v2208 = b84;	// L2633
    v2187.write(v2208); // v2187[k84] = v2208;	// L2634
  }
  int32_t v2209 = v84;	// L2636
  v2188[v2189][v2190] = v2209;	// L2637
}

void PE_kernel_gemm_5_5(
  hls::stream< int8_t > &v2210 /* v2210[16] */,
  hls::stream< int8_t > &v2211 /* v2211[16] */,
  hls::stream< int8_t > &v2212 /* v2212[16] */,
  hls::stream< int8_t > &v2213 /* v2213[16] */,
  int32_t v2214[16][16],
  int v2215,
  int v2216
) {	// L2640
  #pragma HLS stream variable=v2210 depth=17
  #pragma HLS stream variable=v2211 depth=17
  #pragma HLS stream variable=v2212 depth=17
  #pragma HLS stream variable=v2213 depth=17
  #pragma HLS array_partition variable=v2214 complete dim=1
  #pragma HLS array_partition variable=v2214 complete dim=2

  int32_t v85;	// L2642
  v85 = 0;	// L2643
  l_reduction_k85: for (int k85 = 0; k85 < 16; k85++) {	// L2644
  #pragma HLS pipeline II=1
    int8_t v2219 = v2210.read(); // v2210[k85];	// L2645
    int8_t a85;	// L2646
    a85 = v2219;	// L2647
    int8_t v2221 = v2211.read(); // v2211[k85];	// L2648
    int8_t b85;	// L2649
    b85 = v2221;	// L2650
    int8_t v2223 = a85;	// L2651
    int8_t v2224 = b85;	// L2652
    int16_t v2225 = v2223;	// L2653
    int16_t v2226 = v2224;	// L2654
    int16_t v2227 = v2225 * v2226;	// L2655
    int32_t v2228 = v85;	// L2656
    ap_int<33> v2229 = v2228;	// L2657
    ap_int<33> v2230 = v2227;	// L2658
    ap_int<33> v2231 = v2229 + v2230;	// L2659
    int32_t v2232 = v2231;	// L2660
    v85 = v2232;	// L2661
    int8_t v2233 = a85;	// L2662
    v2212.write(v2233); // v2212[k85] = v2233;	// L2663
    int8_t v2234 = b85;	// L2664
    v2213.write(v2234); // v2213[k85] = v2234;	// L2665
  }
  int32_t v2235 = v85;	// L2667
  v2214[v2215][v2216] = v2235;	// L2668
}

void PE_kernel_gemm_6_5(
  hls::stream< int8_t > &v2236 /* v2236[16] */,
  hls::stream< int8_t > &v2237 /* v2237[16] */,
  hls::stream< int8_t > &v2238 /* v2238[16] */,
  hls::stream< int8_t > &v2239 /* v2239[16] */,
  int32_t v2240[16][16],
  int v2241,
  int v2242
) {	// L2671
  #pragma HLS stream variable=v2236 depth=17
  #pragma HLS stream variable=v2237 depth=17
  #pragma HLS stream variable=v2238 depth=17
  #pragma HLS stream variable=v2239 depth=17
  #pragma HLS array_partition variable=v2240 complete dim=1
  #pragma HLS array_partition variable=v2240 complete dim=2

  int32_t v86;	// L2673
  v86 = 0;	// L2674
  l_reduction_k86: for (int k86 = 0; k86 < 16; k86++) {	// L2675
  #pragma HLS pipeline II=1
    int8_t v2245 = v2236.read(); // v2236[k86];	// L2676
    int8_t a86;	// L2677
    a86 = v2245;	// L2678
    int8_t v2247 = v2237.read(); // v2237[k86];	// L2679
    int8_t b86;	// L2680
    b86 = v2247;	// L2681
    int8_t v2249 = a86;	// L2682
    int8_t v2250 = b86;	// L2683
    int16_t v2251 = v2249;	// L2684
    int16_t v2252 = v2250;	// L2685
    int16_t v2253 = v2251 * v2252;	// L2686
    int32_t v2254 = v86;	// L2687
    ap_int<33> v2255 = v2254;	// L2688
    ap_int<33> v2256 = v2253;	// L2689
    ap_int<33> v2257 = v2255 + v2256;	// L2690
    int32_t v2258 = v2257;	// L2691
    v86 = v2258;	// L2692
    int8_t v2259 = a86;	// L2693
    v2238.write(v2259); // v2238[k86] = v2259;	// L2694
    int8_t v2260 = b86;	// L2695
    v2239.write(v2260); // v2239[k86] = v2260;	// L2696
  }
  int32_t v2261 = v86;	// L2698
  v2240[v2241][v2242] = v2261;	// L2699
}

void PE_kernel_gemm_7_5(
  hls::stream< int8_t > &v2262 /* v2262[16] */,
  hls::stream< int8_t > &v2263 /* v2263[16] */,
  hls::stream< int8_t > &v2264 /* v2264[16] */,
  hls::stream< int8_t > &v2265 /* v2265[16] */,
  int32_t v2266[16][16],
  int v2267,
  int v2268
) {	// L2702
  #pragma HLS stream variable=v2262 depth=17
  #pragma HLS stream variable=v2263 depth=17
  #pragma HLS stream variable=v2264 depth=17
  #pragma HLS stream variable=v2265 depth=17
  #pragma HLS array_partition variable=v2266 complete dim=1
  #pragma HLS array_partition variable=v2266 complete dim=2

  int32_t v87;	// L2704
  v87 = 0;	// L2705
  l_reduction_k87: for (int k87 = 0; k87 < 16; k87++) {	// L2706
  #pragma HLS pipeline II=1
    int8_t v2271 = v2262.read(); // v2262[k87];	// L2707
    int8_t a87;	// L2708
    a87 = v2271;	// L2709
    int8_t v2273 = v2263.read(); // v2263[k87];	// L2710
    int8_t b87;	// L2711
    b87 = v2273;	// L2712
    int8_t v2275 = a87;	// L2713
    int8_t v2276 = b87;	// L2714
    int16_t v2277 = v2275;	// L2715
    int16_t v2278 = v2276;	// L2716
    int16_t v2279 = v2277 * v2278;	// L2717
    int32_t v2280 = v87;	// L2718
    ap_int<33> v2281 = v2280;	// L2719
    ap_int<33> v2282 = v2279;	// L2720
    ap_int<33> v2283 = v2281 + v2282;	// L2721
    int32_t v2284 = v2283;	// L2722
    v87 = v2284;	// L2723
    int8_t v2285 = a87;	// L2724
    v2264.write(v2285); // v2264[k87] = v2285;	// L2725
    int8_t v2286 = b87;	// L2726
    v2265.write(v2286); // v2265[k87] = v2286;	// L2727
  }
  int32_t v2287 = v87;	// L2729
  v2266[v2267][v2268] = v2287;	// L2730
}

void PE_kernel_gemm_8_5(
  hls::stream< int8_t > &v2288 /* v2288[16] */,
  hls::stream< int8_t > &v2289 /* v2289[16] */,
  hls::stream< int8_t > &v2290 /* v2290[16] */,
  hls::stream< int8_t > &v2291 /* v2291[16] */,
  int32_t v2292[16][16],
  int v2293,
  int v2294
) {	// L2733
  #pragma HLS stream variable=v2288 depth=17
  #pragma HLS stream variable=v2289 depth=17
  #pragma HLS stream variable=v2290 depth=17
  #pragma HLS stream variable=v2291 depth=17
  #pragma HLS array_partition variable=v2292 complete dim=1
  #pragma HLS array_partition variable=v2292 complete dim=2

  int32_t v88;	// L2735
  v88 = 0;	// L2736
  l_reduction_k88: for (int k88 = 0; k88 < 16; k88++) {	// L2737
  #pragma HLS pipeline II=1
    int8_t v2297 = v2288.read(); // v2288[k88];	// L2738
    int8_t a88;	// L2739
    a88 = v2297;	// L2740
    int8_t v2299 = v2289.read(); // v2289[k88];	// L2741
    int8_t b88;	// L2742
    b88 = v2299;	// L2743
    int8_t v2301 = a88;	// L2744
    int8_t v2302 = b88;	// L2745
    int16_t v2303 = v2301;	// L2746
    int16_t v2304 = v2302;	// L2747
    int16_t v2305 = v2303 * v2304;	// L2748
    int32_t v2306 = v88;	// L2749
    ap_int<33> v2307 = v2306;	// L2750
    ap_int<33> v2308 = v2305;	// L2751
    ap_int<33> v2309 = v2307 + v2308;	// L2752
    int32_t v2310 = v2309;	// L2753
    v88 = v2310;	// L2754
    int8_t v2311 = a88;	// L2755
    v2290.write(v2311); // v2290[k88] = v2311;	// L2756
    int8_t v2312 = b88;	// L2757
    v2291.write(v2312); // v2291[k88] = v2312;	// L2758
  }
  int32_t v2313 = v88;	// L2760
  v2292[v2293][v2294] = v2313;	// L2761
}

void PE_kernel_gemm_9_5(
  hls::stream< int8_t > &v2314 /* v2314[16] */,
  hls::stream< int8_t > &v2315 /* v2315[16] */,
  hls::stream< int8_t > &v2316 /* v2316[16] */,
  hls::stream< int8_t > &v2317 /* v2317[16] */,
  int32_t v2318[16][16],
  int v2319,
  int v2320
) {	// L2764
  #pragma HLS stream variable=v2314 depth=17
  #pragma HLS stream variable=v2315 depth=17
  #pragma HLS stream variable=v2316 depth=17
  #pragma HLS stream variable=v2317 depth=17
  #pragma HLS array_partition variable=v2318 complete dim=1
  #pragma HLS array_partition variable=v2318 complete dim=2

  int32_t v89;	// L2766
  v89 = 0;	// L2767
  l_reduction_k89: for (int k89 = 0; k89 < 16; k89++) {	// L2768
  #pragma HLS pipeline II=1
    int8_t v2323 = v2314.read(); // v2314[k89];	// L2769
    int8_t a89;	// L2770
    a89 = v2323;	// L2771
    int8_t v2325 = v2315.read(); // v2315[k89];	// L2772
    int8_t b89;	// L2773
    b89 = v2325;	// L2774
    int8_t v2327 = a89;	// L2775
    int8_t v2328 = b89;	// L2776
    int16_t v2329 = v2327;	// L2777
    int16_t v2330 = v2328;	// L2778
    int16_t v2331 = v2329 * v2330;	// L2779
    int32_t v2332 = v89;	// L2780
    ap_int<33> v2333 = v2332;	// L2781
    ap_int<33> v2334 = v2331;	// L2782
    ap_int<33> v2335 = v2333 + v2334;	// L2783
    int32_t v2336 = v2335;	// L2784
    v89 = v2336;	// L2785
    int8_t v2337 = a89;	// L2786
    v2316.write(v2337); // v2316[k89] = v2337;	// L2787
    int8_t v2338 = b89;	// L2788
    v2317.write(v2338); // v2317[k89] = v2338;	// L2789
  }
  int32_t v2339 = v89;	// L2791
  v2318[v2319][v2320] = v2339;	// L2792
}

void PE_kernel_gemm_10_5(
  hls::stream< int8_t > &v2340 /* v2340[16] */,
  hls::stream< int8_t > &v2341 /* v2341[16] */,
  hls::stream< int8_t > &v2342 /* v2342[16] */,
  hls::stream< int8_t > &v2343 /* v2343[16] */,
  int32_t v2344[16][16],
  int v2345,
  int v2346
) {	// L2795
  #pragma HLS stream variable=v2340 depth=17
  #pragma HLS stream variable=v2341 depth=17
  #pragma HLS stream variable=v2342 depth=17
  #pragma HLS stream variable=v2343 depth=17
  #pragma HLS array_partition variable=v2344 complete dim=1
  #pragma HLS array_partition variable=v2344 complete dim=2

  int32_t v90;	// L2797
  v90 = 0;	// L2798
  l_reduction_k90: for (int k90 = 0; k90 < 16; k90++) {	// L2799
  #pragma HLS pipeline II=1
    int8_t v2349 = v2340.read(); // v2340[k90];	// L2800
    int8_t a90;	// L2801
    a90 = v2349;	// L2802
    int8_t v2351 = v2341.read(); // v2341[k90];	// L2803
    int8_t b90;	// L2804
    b90 = v2351;	// L2805
    int8_t v2353 = a90;	// L2806
    int8_t v2354 = b90;	// L2807
    int16_t v2355 = v2353;	// L2808
    int16_t v2356 = v2354;	// L2809
    int16_t v2357 = v2355 * v2356;	// L2810
    int32_t v2358 = v90;	// L2811
    ap_int<33> v2359 = v2358;	// L2812
    ap_int<33> v2360 = v2357;	// L2813
    ap_int<33> v2361 = v2359 + v2360;	// L2814
    int32_t v2362 = v2361;	// L2815
    v90 = v2362;	// L2816
    int8_t v2363 = a90;	// L2817
    v2342.write(v2363); // v2342[k90] = v2363;	// L2818
    int8_t v2364 = b90;	// L2819
    v2343.write(v2364); // v2343[k90] = v2364;	// L2820
  }
  int32_t v2365 = v90;	// L2822
  v2344[v2345][v2346] = v2365;	// L2823
}

void PE_kernel_gemm_11_5(
  hls::stream< int8_t > &v2366 /* v2366[16] */,
  hls::stream< int8_t > &v2367 /* v2367[16] */,
  hls::stream< int8_t > &v2368 /* v2368[16] */,
  hls::stream< int8_t > &v2369 /* v2369[16] */,
  int32_t v2370[16][16],
  int v2371,
  int v2372
) {	// L2826
  #pragma HLS stream variable=v2366 depth=17
  #pragma HLS stream variable=v2367 depth=17
  #pragma HLS stream variable=v2368 depth=17
  #pragma HLS stream variable=v2369 depth=17
  #pragma HLS array_partition variable=v2370 complete dim=1
  #pragma HLS array_partition variable=v2370 complete dim=2

  int32_t v91;	// L2828
  v91 = 0;	// L2829
  l_reduction_k91: for (int k91 = 0; k91 < 16; k91++) {	// L2830
  #pragma HLS pipeline II=1
    int8_t v2375 = v2366.read(); // v2366[k91];	// L2831
    int8_t a91;	// L2832
    a91 = v2375;	// L2833
    int8_t v2377 = v2367.read(); // v2367[k91];	// L2834
    int8_t b91;	// L2835
    b91 = v2377;	// L2836
    int8_t v2379 = a91;	// L2837
    int8_t v2380 = b91;	// L2838
    int16_t v2381 = v2379;	// L2839
    int16_t v2382 = v2380;	// L2840
    int16_t v2383 = v2381 * v2382;	// L2841
    int32_t v2384 = v91;	// L2842
    ap_int<33> v2385 = v2384;	// L2843
    ap_int<33> v2386 = v2383;	// L2844
    ap_int<33> v2387 = v2385 + v2386;	// L2845
    int32_t v2388 = v2387;	// L2846
    v91 = v2388;	// L2847
    int8_t v2389 = a91;	// L2848
    v2368.write(v2389); // v2368[k91] = v2389;	// L2849
    int8_t v2390 = b91;	// L2850
    v2369.write(v2390); // v2369[k91] = v2390;	// L2851
  }
  int32_t v2391 = v91;	// L2853
  v2370[v2371][v2372] = v2391;	// L2854
}

void PE_kernel_gemm_12_5(
  hls::stream< int8_t > &v2392 /* v2392[16] */,
  hls::stream< int8_t > &v2393 /* v2393[16] */,
  hls::stream< int8_t > &v2394 /* v2394[16] */,
  hls::stream< int8_t > &v2395 /* v2395[16] */,
  int32_t v2396[16][16],
  int v2397,
  int v2398
) {	// L2857
  #pragma HLS stream variable=v2392 depth=17
  #pragma HLS stream variable=v2393 depth=17
  #pragma HLS stream variable=v2394 depth=17
  #pragma HLS stream variable=v2395 depth=17
  #pragma HLS array_partition variable=v2396 complete dim=1
  #pragma HLS array_partition variable=v2396 complete dim=2

  int32_t v92;	// L2859
  v92 = 0;	// L2860
  l_reduction_k92: for (int k92 = 0; k92 < 16; k92++) {	// L2861
  #pragma HLS pipeline II=1
    int8_t v2401 = v2392.read(); // v2392[k92];	// L2862
    int8_t a92;	// L2863
    a92 = v2401;	// L2864
    int8_t v2403 = v2393.read(); // v2393[k92];	// L2865
    int8_t b92;	// L2866
    b92 = v2403;	// L2867
    int8_t v2405 = a92;	// L2868
    int8_t v2406 = b92;	// L2869
    int16_t v2407 = v2405;	// L2870
    int16_t v2408 = v2406;	// L2871
    int16_t v2409 = v2407 * v2408;	// L2872
    int32_t v2410 = v92;	// L2873
    ap_int<33> v2411 = v2410;	// L2874
    ap_int<33> v2412 = v2409;	// L2875
    ap_int<33> v2413 = v2411 + v2412;	// L2876
    int32_t v2414 = v2413;	// L2877
    v92 = v2414;	// L2878
    int8_t v2415 = a92;	// L2879
    v2394.write(v2415); // v2394[k92] = v2415;	// L2880
    int8_t v2416 = b92;	// L2881
    v2395.write(v2416); // v2395[k92] = v2416;	// L2882
  }
  int32_t v2417 = v92;	// L2884
  v2396[v2397][v2398] = v2417;	// L2885
}

void PE_kernel_gemm_13_5(
  hls::stream< int8_t > &v2418 /* v2418[16] */,
  hls::stream< int8_t > &v2419 /* v2419[16] */,
  hls::stream< int8_t > &v2420 /* v2420[16] */,
  hls::stream< int8_t > &v2421 /* v2421[16] */,
  int32_t v2422[16][16],
  int v2423,
  int v2424
) {	// L2888
  #pragma HLS stream variable=v2418 depth=17
  #pragma HLS stream variable=v2419 depth=17
  #pragma HLS stream variable=v2420 depth=17
  #pragma HLS stream variable=v2421 depth=17
  #pragma HLS array_partition variable=v2422 complete dim=1
  #pragma HLS array_partition variable=v2422 complete dim=2

  int32_t v93;	// L2890
  v93 = 0;	// L2891
  l_reduction_k93: for (int k93 = 0; k93 < 16; k93++) {	// L2892
  #pragma HLS pipeline II=1
    int8_t v2427 = v2418.read(); // v2418[k93];	// L2893
    int8_t a93;	// L2894
    a93 = v2427;	// L2895
    int8_t v2429 = v2419.read(); // v2419[k93];	// L2896
    int8_t b93;	// L2897
    b93 = v2429;	// L2898
    int8_t v2431 = a93;	// L2899
    int8_t v2432 = b93;	// L2900
    int16_t v2433 = v2431;	// L2901
    int16_t v2434 = v2432;	// L2902
    int16_t v2435 = v2433 * v2434;	// L2903
    int32_t v2436 = v93;	// L2904
    ap_int<33> v2437 = v2436;	// L2905
    ap_int<33> v2438 = v2435;	// L2906
    ap_int<33> v2439 = v2437 + v2438;	// L2907
    int32_t v2440 = v2439;	// L2908
    v93 = v2440;	// L2909
    int8_t v2441 = a93;	// L2910
    v2420.write(v2441); // v2420[k93] = v2441;	// L2911
    int8_t v2442 = b93;	// L2912
    v2421.write(v2442); // v2421[k93] = v2442;	// L2913
  }
  int32_t v2443 = v93;	// L2915
  v2422[v2423][v2424] = v2443;	// L2916
}

void PE_kernel_gemm_14_5(
  hls::stream< int8_t > &v2444 /* v2444[16] */,
  hls::stream< int8_t > &v2445 /* v2445[16] */,
  hls::stream< int8_t > &v2446 /* v2446[16] */,
  hls::stream< int8_t > &v2447 /* v2447[16] */,
  int32_t v2448[16][16],
  int v2449,
  int v2450
) {	// L2919
  #pragma HLS stream variable=v2444 depth=17
  #pragma HLS stream variable=v2445 depth=17
  #pragma HLS stream variable=v2446 depth=17
  #pragma HLS stream variable=v2447 depth=17
  #pragma HLS array_partition variable=v2448 complete dim=1
  #pragma HLS array_partition variable=v2448 complete dim=2

  int32_t v94;	// L2921
  v94 = 0;	// L2922
  l_reduction_k94: for (int k94 = 0; k94 < 16; k94++) {	// L2923
  #pragma HLS pipeline II=1
    int8_t v2453 = v2444.read(); // v2444[k94];	// L2924
    int8_t a94;	// L2925
    a94 = v2453;	// L2926
    int8_t v2455 = v2445.read(); // v2445[k94];	// L2927
    int8_t b94;	// L2928
    b94 = v2455;	// L2929
    int8_t v2457 = a94;	// L2930
    int8_t v2458 = b94;	// L2931
    int16_t v2459 = v2457;	// L2932
    int16_t v2460 = v2458;	// L2933
    int16_t v2461 = v2459 * v2460;	// L2934
    int32_t v2462 = v94;	// L2935
    ap_int<33> v2463 = v2462;	// L2936
    ap_int<33> v2464 = v2461;	// L2937
    ap_int<33> v2465 = v2463 + v2464;	// L2938
    int32_t v2466 = v2465;	// L2939
    v94 = v2466;	// L2940
    int8_t v2467 = a94;	// L2941
    v2446.write(v2467); // v2446[k94] = v2467;	// L2942
    int8_t v2468 = b94;	// L2943
    v2447.write(v2468); // v2447[k94] = v2468;	// L2944
  }
  int32_t v2469 = v94;	// L2946
  v2448[v2449][v2450] = v2469;	// L2947
}

void PE_kernel_gemm_15_5(
  hls::stream< int8_t > &v2470 /* v2470[16] */,
  hls::stream< int8_t > &v2471 /* v2471[16] */,
  hls::stream< int8_t > &v2472 /* v2472[16] */,
  hls::stream< int8_t > &v2473 /* v2473[16] */,
  int32_t v2474[16][16],
  int v2475,
  int v2476
) {	// L2950
  #pragma HLS stream variable=v2470 depth=17
  #pragma HLS stream variable=v2471 depth=17
  #pragma HLS stream variable=v2472 depth=17
  #pragma HLS stream variable=v2473 depth=17
  #pragma HLS array_partition variable=v2474 complete dim=1
  #pragma HLS array_partition variable=v2474 complete dim=2

  int32_t v95;	// L2952
  v95 = 0;	// L2953
  l_reduction_k95: for (int k95 = 0; k95 < 16; k95++) {	// L2954
  #pragma HLS pipeline II=1
    int8_t v2479 = v2470.read(); // v2470[k95];	// L2955
    int8_t a95;	// L2956
    a95 = v2479;	// L2957
    int8_t v2481 = v2471.read(); // v2471[k95];	// L2958
    int8_t b95;	// L2959
    b95 = v2481;	// L2960
    int8_t v2483 = a95;	// L2961
    int8_t v2484 = b95;	// L2962
    int16_t v2485 = v2483;	// L2963
    int16_t v2486 = v2484;	// L2964
    int16_t v2487 = v2485 * v2486;	// L2965
    int32_t v2488 = v95;	// L2966
    ap_int<33> v2489 = v2488;	// L2967
    ap_int<33> v2490 = v2487;	// L2968
    ap_int<33> v2491 = v2489 + v2490;	// L2969
    int32_t v2492 = v2491;	// L2970
    v95 = v2492;	// L2971
    int8_t v2493 = a95;	// L2972
    v2472.write(v2493); // v2472[k95] = v2493;	// L2973
    int8_t v2494 = b95;	// L2974
    v2473.write(v2494); // v2473[k95] = v2494;	// L2975
  }
  int32_t v2495 = v95;	// L2977
  v2474[v2475][v2476] = v2495;	// L2978
}

void PE_kernel_gemm_0_6(
  hls::stream< int8_t > &v2496 /* v2496[16] */,
  hls::stream< int8_t > &v2497 /* v2497[16] */,
  hls::stream< int8_t > &v2498 /* v2498[16] */,
  hls::stream< int8_t > &v2499 /* v2499[16] */,
  int32_t v2500[16][16],
  int v2501,
  int v2502
) {	// L2981
  #pragma HLS stream variable=v2496 depth=17
  #pragma HLS stream variable=v2497 depth=17
  #pragma HLS stream variable=v2498 depth=17
  #pragma HLS stream variable=v2499 depth=17
  #pragma HLS array_partition variable=v2500 complete dim=1
  #pragma HLS array_partition variable=v2500 complete dim=2

  int32_t v96;	// L2983
  v96 = 0;	// L2984
  l_reduction_k96: for (int k96 = 0; k96 < 16; k96++) {	// L2985
  #pragma HLS pipeline II=1
    int8_t v2505 = v2496.read(); // v2496[k96];	// L2986
    int8_t a96;	// L2987
    a96 = v2505;	// L2988
    int8_t v2507 = v2497.read(); // v2497[k96];	// L2989
    int8_t b96;	// L2990
    b96 = v2507;	// L2991
    int8_t v2509 = a96;	// L2992
    int8_t v2510 = b96;	// L2993
    int16_t v2511 = v2509;	// L2994
    int16_t v2512 = v2510;	// L2995
    int16_t v2513 = v2511 * v2512;	// L2996
    int32_t v2514 = v96;	// L2997
    ap_int<33> v2515 = v2514;	// L2998
    ap_int<33> v2516 = v2513;	// L2999
    ap_int<33> v2517 = v2515 + v2516;	// L3000
    int32_t v2518 = v2517;	// L3001
    v96 = v2518;	// L3002
    int8_t v2519 = a96;	// L3003
    v2498.write(v2519); // v2498[k96] = v2519;	// L3004
    int8_t v2520 = b96;	// L3005
    v2499.write(v2520); // v2499[k96] = v2520;	// L3006
  }
  int32_t v2521 = v96;	// L3008
  v2500[v2501][v2502] = v2521;	// L3009
}

void PE_kernel_gemm_1_6(
  hls::stream< int8_t > &v2522 /* v2522[16] */,
  hls::stream< int8_t > &v2523 /* v2523[16] */,
  hls::stream< int8_t > &v2524 /* v2524[16] */,
  hls::stream< int8_t > &v2525 /* v2525[16] */,
  int32_t v2526[16][16],
  int v2527,
  int v2528
) {	// L3012
  #pragma HLS stream variable=v2522 depth=17
  #pragma HLS stream variable=v2523 depth=17
  #pragma HLS stream variable=v2524 depth=17
  #pragma HLS stream variable=v2525 depth=17
  #pragma HLS array_partition variable=v2526 complete dim=1
  #pragma HLS array_partition variable=v2526 complete dim=2

  int32_t v97;	// L3014
  v97 = 0;	// L3015
  l_reduction_k97: for (int k97 = 0; k97 < 16; k97++) {	// L3016
  #pragma HLS pipeline II=1
    int8_t v2531 = v2522.read(); // v2522[k97];	// L3017
    int8_t a97;	// L3018
    a97 = v2531;	// L3019
    int8_t v2533 = v2523.read(); // v2523[k97];	// L3020
    int8_t b97;	// L3021
    b97 = v2533;	// L3022
    int8_t v2535 = a97;	// L3023
    int8_t v2536 = b97;	// L3024
    int16_t v2537 = v2535;	// L3025
    int16_t v2538 = v2536;	// L3026
    int16_t v2539 = v2537 * v2538;	// L3027
    int32_t v2540 = v97;	// L3028
    ap_int<33> v2541 = v2540;	// L3029
    ap_int<33> v2542 = v2539;	// L3030
    ap_int<33> v2543 = v2541 + v2542;	// L3031
    int32_t v2544 = v2543;	// L3032
    v97 = v2544;	// L3033
    int8_t v2545 = a97;	// L3034
    v2524.write(v2545); // v2524[k97] = v2545;	// L3035
    int8_t v2546 = b97;	// L3036
    v2525.write(v2546); // v2525[k97] = v2546;	// L3037
  }
  int32_t v2547 = v97;	// L3039
  v2526[v2527][v2528] = v2547;	// L3040
}

void PE_kernel_gemm_2_6(
  hls::stream< int8_t > &v2548 /* v2548[16] */,
  hls::stream< int8_t > &v2549 /* v2549[16] */,
  hls::stream< int8_t > &v2550 /* v2550[16] */,
  hls::stream< int8_t > &v2551 /* v2551[16] */,
  int32_t v2552[16][16],
  int v2553,
  int v2554
) {	// L3043
  #pragma HLS stream variable=v2548 depth=17
  #pragma HLS stream variable=v2549 depth=17
  #pragma HLS stream variable=v2550 depth=17
  #pragma HLS stream variable=v2551 depth=17
  #pragma HLS array_partition variable=v2552 complete dim=1
  #pragma HLS array_partition variable=v2552 complete dim=2

  int32_t v98;	// L3045
  v98 = 0;	// L3046
  l_reduction_k98: for (int k98 = 0; k98 < 16; k98++) {	// L3047
  #pragma HLS pipeline II=1
    int8_t v2557 = v2548.read(); // v2548[k98];	// L3048
    int8_t a98;	// L3049
    a98 = v2557;	// L3050
    int8_t v2559 = v2549.read(); // v2549[k98];	// L3051
    int8_t b98;	// L3052
    b98 = v2559;	// L3053
    int8_t v2561 = a98;	// L3054
    int8_t v2562 = b98;	// L3055
    int16_t v2563 = v2561;	// L3056
    int16_t v2564 = v2562;	// L3057
    int16_t v2565 = v2563 * v2564;	// L3058
    int32_t v2566 = v98;	// L3059
    ap_int<33> v2567 = v2566;	// L3060
    ap_int<33> v2568 = v2565;	// L3061
    ap_int<33> v2569 = v2567 + v2568;	// L3062
    int32_t v2570 = v2569;	// L3063
    v98 = v2570;	// L3064
    int8_t v2571 = a98;	// L3065
    v2550.write(v2571); // v2550[k98] = v2571;	// L3066
    int8_t v2572 = b98;	// L3067
    v2551.write(v2572); // v2551[k98] = v2572;	// L3068
  }
  int32_t v2573 = v98;	// L3070
  v2552[v2553][v2554] = v2573;	// L3071
}

void PE_kernel_gemm_3_6(
  hls::stream< int8_t > &v2574 /* v2574[16] */,
  hls::stream< int8_t > &v2575 /* v2575[16] */,
  hls::stream< int8_t > &v2576 /* v2576[16] */,
  hls::stream< int8_t > &v2577 /* v2577[16] */,
  int32_t v2578[16][16],
  int v2579,
  int v2580
) {	// L3074
  #pragma HLS stream variable=v2574 depth=17
  #pragma HLS stream variable=v2575 depth=17
  #pragma HLS stream variable=v2576 depth=17
  #pragma HLS stream variable=v2577 depth=17
  #pragma HLS array_partition variable=v2578 complete dim=1
  #pragma HLS array_partition variable=v2578 complete dim=2

  int32_t v99;	// L3076
  v99 = 0;	// L3077
  l_reduction_k99: for (int k99 = 0; k99 < 16; k99++) {	// L3078
  #pragma HLS pipeline II=1
    int8_t v2583 = v2574.read(); // v2574[k99];	// L3079
    int8_t a99;	// L3080
    a99 = v2583;	// L3081
    int8_t v2585 = v2575.read(); // v2575[k99];	// L3082
    int8_t b99;	// L3083
    b99 = v2585;	// L3084
    int8_t v2587 = a99;	// L3085
    int8_t v2588 = b99;	// L3086
    int16_t v2589 = v2587;	// L3087
    int16_t v2590 = v2588;	// L3088
    int16_t v2591 = v2589 * v2590;	// L3089
    int32_t v2592 = v99;	// L3090
    ap_int<33> v2593 = v2592;	// L3091
    ap_int<33> v2594 = v2591;	// L3092
    ap_int<33> v2595 = v2593 + v2594;	// L3093
    int32_t v2596 = v2595;	// L3094
    v99 = v2596;	// L3095
    int8_t v2597 = a99;	// L3096
    v2576.write(v2597); // v2576[k99] = v2597;	// L3097
    int8_t v2598 = b99;	// L3098
    v2577.write(v2598); // v2577[k99] = v2598;	// L3099
  }
  int32_t v2599 = v99;	// L3101
  v2578[v2579][v2580] = v2599;	// L3102
}

void PE_kernel_gemm_4_6(
  hls::stream< int8_t > &v2600 /* v2600[16] */,
  hls::stream< int8_t > &v2601 /* v2601[16] */,
  hls::stream< int8_t > &v2602 /* v2602[16] */,
  hls::stream< int8_t > &v2603 /* v2603[16] */,
  int32_t v2604[16][16],
  int v2605,
  int v2606
) {	// L3105
  #pragma HLS stream variable=v2600 depth=17
  #pragma HLS stream variable=v2601 depth=17
  #pragma HLS stream variable=v2602 depth=17
  #pragma HLS stream variable=v2603 depth=17
  #pragma HLS array_partition variable=v2604 complete dim=1
  #pragma HLS array_partition variable=v2604 complete dim=2

  int32_t v100;	// L3107
  v100 = 0;	// L3108
  l_reduction_k100: for (int k100 = 0; k100 < 16; k100++) {	// L3109
  #pragma HLS pipeline II=1
    int8_t v2609 = v2600.read(); // v2600[k100];	// L3110
    int8_t a100;	// L3111
    a100 = v2609;	// L3112
    int8_t v2611 = v2601.read(); // v2601[k100];	// L3113
    int8_t b100;	// L3114
    b100 = v2611;	// L3115
    int8_t v2613 = a100;	// L3116
    int8_t v2614 = b100;	// L3117
    int16_t v2615 = v2613;	// L3118
    int16_t v2616 = v2614;	// L3119
    int16_t v2617 = v2615 * v2616;	// L3120
    int32_t v2618 = v100;	// L3121
    ap_int<33> v2619 = v2618;	// L3122
    ap_int<33> v2620 = v2617;	// L3123
    ap_int<33> v2621 = v2619 + v2620;	// L3124
    int32_t v2622 = v2621;	// L3125
    v100 = v2622;	// L3126
    int8_t v2623 = a100;	// L3127
    v2602.write(v2623); // v2602[k100] = v2623;	// L3128
    int8_t v2624 = b100;	// L3129
    v2603.write(v2624); // v2603[k100] = v2624;	// L3130
  }
  int32_t v2625 = v100;	// L3132
  v2604[v2605][v2606] = v2625;	// L3133
}

void PE_kernel_gemm_5_6(
  hls::stream< int8_t > &v2626 /* v2626[16] */,
  hls::stream< int8_t > &v2627 /* v2627[16] */,
  hls::stream< int8_t > &v2628 /* v2628[16] */,
  hls::stream< int8_t > &v2629 /* v2629[16] */,
  int32_t v2630[16][16],
  int v2631,
  int v2632
) {	// L3136
  #pragma HLS stream variable=v2626 depth=17
  #pragma HLS stream variable=v2627 depth=17
  #pragma HLS stream variable=v2628 depth=17
  #pragma HLS stream variable=v2629 depth=17
  #pragma HLS array_partition variable=v2630 complete dim=1
  #pragma HLS array_partition variable=v2630 complete dim=2

  int32_t v101;	// L3138
  v101 = 0;	// L3139
  l_reduction_k101: for (int k101 = 0; k101 < 16; k101++) {	// L3140
  #pragma HLS pipeline II=1
    int8_t v2635 = v2626.read(); // v2626[k101];	// L3141
    int8_t a101;	// L3142
    a101 = v2635;	// L3143
    int8_t v2637 = v2627.read(); // v2627[k101];	// L3144
    int8_t b101;	// L3145
    b101 = v2637;	// L3146
    int8_t v2639 = a101;	// L3147
    int8_t v2640 = b101;	// L3148
    int16_t v2641 = v2639;	// L3149
    int16_t v2642 = v2640;	// L3150
    int16_t v2643 = v2641 * v2642;	// L3151
    int32_t v2644 = v101;	// L3152
    ap_int<33> v2645 = v2644;	// L3153
    ap_int<33> v2646 = v2643;	// L3154
    ap_int<33> v2647 = v2645 + v2646;	// L3155
    int32_t v2648 = v2647;	// L3156
    v101 = v2648;	// L3157
    int8_t v2649 = a101;	// L3158
    v2628.write(v2649); // v2628[k101] = v2649;	// L3159
    int8_t v2650 = b101;	// L3160
    v2629.write(v2650); // v2629[k101] = v2650;	// L3161
  }
  int32_t v2651 = v101;	// L3163
  v2630[v2631][v2632] = v2651;	// L3164
}

void PE_kernel_gemm_6_6(
  hls::stream< int8_t > &v2652 /* v2652[16] */,
  hls::stream< int8_t > &v2653 /* v2653[16] */,
  hls::stream< int8_t > &v2654 /* v2654[16] */,
  hls::stream< int8_t > &v2655 /* v2655[16] */,
  int32_t v2656[16][16],
  int v2657,
  int v2658
) {	// L3167
  #pragma HLS stream variable=v2652 depth=17
  #pragma HLS stream variable=v2653 depth=17
  #pragma HLS stream variable=v2654 depth=17
  #pragma HLS stream variable=v2655 depth=17
  #pragma HLS array_partition variable=v2656 complete dim=1
  #pragma HLS array_partition variable=v2656 complete dim=2

  int32_t v102;	// L3169
  v102 = 0;	// L3170
  l_reduction_k102: for (int k102 = 0; k102 < 16; k102++) {	// L3171
  #pragma HLS pipeline II=1
    int8_t v2661 = v2652.read(); // v2652[k102];	// L3172
    int8_t a102;	// L3173
    a102 = v2661;	// L3174
    int8_t v2663 = v2653.read(); // v2653[k102];	// L3175
    int8_t b102;	// L3176
    b102 = v2663;	// L3177
    int8_t v2665 = a102;	// L3178
    int8_t v2666 = b102;	// L3179
    int16_t v2667 = v2665;	// L3180
    int16_t v2668 = v2666;	// L3181
    int16_t v2669 = v2667 * v2668;	// L3182
    int32_t v2670 = v102;	// L3183
    ap_int<33> v2671 = v2670;	// L3184
    ap_int<33> v2672 = v2669;	// L3185
    ap_int<33> v2673 = v2671 + v2672;	// L3186
    int32_t v2674 = v2673;	// L3187
    v102 = v2674;	// L3188
    int8_t v2675 = a102;	// L3189
    v2654.write(v2675); // v2654[k102] = v2675;	// L3190
    int8_t v2676 = b102;	// L3191
    v2655.write(v2676); // v2655[k102] = v2676;	// L3192
  }
  int32_t v2677 = v102;	// L3194
  v2656[v2657][v2658] = v2677;	// L3195
}

void PE_kernel_gemm_7_6(
  hls::stream< int8_t > &v2678 /* v2678[16] */,
  hls::stream< int8_t > &v2679 /* v2679[16] */,
  hls::stream< int8_t > &v2680 /* v2680[16] */,
  hls::stream< int8_t > &v2681 /* v2681[16] */,
  int32_t v2682[16][16],
  int v2683,
  int v2684
) {	// L3198
  #pragma HLS stream variable=v2678 depth=17
  #pragma HLS stream variable=v2679 depth=17
  #pragma HLS stream variable=v2680 depth=17
  #pragma HLS stream variable=v2681 depth=17
  #pragma HLS array_partition variable=v2682 complete dim=1
  #pragma HLS array_partition variable=v2682 complete dim=2

  int32_t v103;	// L3200
  v103 = 0;	// L3201
  l_reduction_k103: for (int k103 = 0; k103 < 16; k103++) {	// L3202
  #pragma HLS pipeline II=1
    int8_t v2687 = v2678.read(); // v2678[k103];	// L3203
    int8_t a103;	// L3204
    a103 = v2687;	// L3205
    int8_t v2689 = v2679.read(); // v2679[k103];	// L3206
    int8_t b103;	// L3207
    b103 = v2689;	// L3208
    int8_t v2691 = a103;	// L3209
    int8_t v2692 = b103;	// L3210
    int16_t v2693 = v2691;	// L3211
    int16_t v2694 = v2692;	// L3212
    int16_t v2695 = v2693 * v2694;	// L3213
    int32_t v2696 = v103;	// L3214
    ap_int<33> v2697 = v2696;	// L3215
    ap_int<33> v2698 = v2695;	// L3216
    ap_int<33> v2699 = v2697 + v2698;	// L3217
    int32_t v2700 = v2699;	// L3218
    v103 = v2700;	// L3219
    int8_t v2701 = a103;	// L3220
    v2680.write(v2701); // v2680[k103] = v2701;	// L3221
    int8_t v2702 = b103;	// L3222
    v2681.write(v2702); // v2681[k103] = v2702;	// L3223
  }
  int32_t v2703 = v103;	// L3225
  v2682[v2683][v2684] = v2703;	// L3226
}

void PE_kernel_gemm_8_6(
  hls::stream< int8_t > &v2704 /* v2704[16] */,
  hls::stream< int8_t > &v2705 /* v2705[16] */,
  hls::stream< int8_t > &v2706 /* v2706[16] */,
  hls::stream< int8_t > &v2707 /* v2707[16] */,
  int32_t v2708[16][16],
  int v2709,
  int v2710
) {	// L3229
  #pragma HLS stream variable=v2704 depth=17
  #pragma HLS stream variable=v2705 depth=17
  #pragma HLS stream variable=v2706 depth=17
  #pragma HLS stream variable=v2707 depth=17
  #pragma HLS array_partition variable=v2708 complete dim=1
  #pragma HLS array_partition variable=v2708 complete dim=2

  int32_t v104;	// L3231
  v104 = 0;	// L3232
  l_reduction_k104: for (int k104 = 0; k104 < 16; k104++) {	// L3233
  #pragma HLS pipeline II=1
    int8_t v2713 = v2704.read(); // v2704[k104];	// L3234
    int8_t a104;	// L3235
    a104 = v2713;	// L3236
    int8_t v2715 = v2705.read(); // v2705[k104];	// L3237
    int8_t b104;	// L3238
    b104 = v2715;	// L3239
    int8_t v2717 = a104;	// L3240
    int8_t v2718 = b104;	// L3241
    int16_t v2719 = v2717;	// L3242
    int16_t v2720 = v2718;	// L3243
    int16_t v2721 = v2719 * v2720;	// L3244
    int32_t v2722 = v104;	// L3245
    ap_int<33> v2723 = v2722;	// L3246
    ap_int<33> v2724 = v2721;	// L3247
    ap_int<33> v2725 = v2723 + v2724;	// L3248
    int32_t v2726 = v2725;	// L3249
    v104 = v2726;	// L3250
    int8_t v2727 = a104;	// L3251
    v2706.write(v2727); // v2706[k104] = v2727;	// L3252
    int8_t v2728 = b104;	// L3253
    v2707.write(v2728); // v2707[k104] = v2728;	// L3254
  }
  int32_t v2729 = v104;	// L3256
  v2708[v2709][v2710] = v2729;	// L3257
}

void PE_kernel_gemm_9_6(
  hls::stream< int8_t > &v2730 /* v2730[16] */,
  hls::stream< int8_t > &v2731 /* v2731[16] */,
  hls::stream< int8_t > &v2732 /* v2732[16] */,
  hls::stream< int8_t > &v2733 /* v2733[16] */,
  int32_t v2734[16][16],
  int v2735,
  int v2736
) {	// L3260
  #pragma HLS stream variable=v2730 depth=17
  #pragma HLS stream variable=v2731 depth=17
  #pragma HLS stream variable=v2732 depth=17
  #pragma HLS stream variable=v2733 depth=17
  #pragma HLS array_partition variable=v2734 complete dim=1
  #pragma HLS array_partition variable=v2734 complete dim=2

  int32_t v105;	// L3262
  v105 = 0;	// L3263
  l_reduction_k105: for (int k105 = 0; k105 < 16; k105++) {	// L3264
  #pragma HLS pipeline II=1
    int8_t v2739 = v2730.read(); // v2730[k105];	// L3265
    int8_t a105;	// L3266
    a105 = v2739;	// L3267
    int8_t v2741 = v2731.read(); // v2731[k105];	// L3268
    int8_t b105;	// L3269
    b105 = v2741;	// L3270
    int8_t v2743 = a105;	// L3271
    int8_t v2744 = b105;	// L3272
    int16_t v2745 = v2743;	// L3273
    int16_t v2746 = v2744;	// L3274
    int16_t v2747 = v2745 * v2746;	// L3275
    int32_t v2748 = v105;	// L3276
    ap_int<33> v2749 = v2748;	// L3277
    ap_int<33> v2750 = v2747;	// L3278
    ap_int<33> v2751 = v2749 + v2750;	// L3279
    int32_t v2752 = v2751;	// L3280
    v105 = v2752;	// L3281
    int8_t v2753 = a105;	// L3282
    v2732.write(v2753); // v2732[k105] = v2753;	// L3283
    int8_t v2754 = b105;	// L3284
    v2733.write(v2754); // v2733[k105] = v2754;	// L3285
  }
  int32_t v2755 = v105;	// L3287
  v2734[v2735][v2736] = v2755;	// L3288
}

void PE_kernel_gemm_10_6(
  hls::stream< int8_t > &v2756 /* v2756[16] */,
  hls::stream< int8_t > &v2757 /* v2757[16] */,
  hls::stream< int8_t > &v2758 /* v2758[16] */,
  hls::stream< int8_t > &v2759 /* v2759[16] */,
  int32_t v2760[16][16],
  int v2761,
  int v2762
) {	// L3291
  #pragma HLS stream variable=v2756 depth=17
  #pragma HLS stream variable=v2757 depth=17
  #pragma HLS stream variable=v2758 depth=17
  #pragma HLS stream variable=v2759 depth=17
  #pragma HLS array_partition variable=v2760 complete dim=1
  #pragma HLS array_partition variable=v2760 complete dim=2

  int32_t v106;	// L3293
  v106 = 0;	// L3294
  l_reduction_k106: for (int k106 = 0; k106 < 16; k106++) {	// L3295
  #pragma HLS pipeline II=1
    int8_t v2765 = v2756.read(); // v2756[k106];	// L3296
    int8_t a106;	// L3297
    a106 = v2765;	// L3298
    int8_t v2767 = v2757.read(); // v2757[k106];	// L3299
    int8_t b106;	// L3300
    b106 = v2767;	// L3301
    int8_t v2769 = a106;	// L3302
    int8_t v2770 = b106;	// L3303
    int16_t v2771 = v2769;	// L3304
    int16_t v2772 = v2770;	// L3305
    int16_t v2773 = v2771 * v2772;	// L3306
    int32_t v2774 = v106;	// L3307
    ap_int<33> v2775 = v2774;	// L3308
    ap_int<33> v2776 = v2773;	// L3309
    ap_int<33> v2777 = v2775 + v2776;	// L3310
    int32_t v2778 = v2777;	// L3311
    v106 = v2778;	// L3312
    int8_t v2779 = a106;	// L3313
    v2758.write(v2779); // v2758[k106] = v2779;	// L3314
    int8_t v2780 = b106;	// L3315
    v2759.write(v2780); // v2759[k106] = v2780;	// L3316
  }
  int32_t v2781 = v106;	// L3318
  v2760[v2761][v2762] = v2781;	// L3319
}

void PE_kernel_gemm_11_6(
  hls::stream< int8_t > &v2782 /* v2782[16] */,
  hls::stream< int8_t > &v2783 /* v2783[16] */,
  hls::stream< int8_t > &v2784 /* v2784[16] */,
  hls::stream< int8_t > &v2785 /* v2785[16] */,
  int32_t v2786[16][16],
  int v2787,
  int v2788
) {	// L3322
  #pragma HLS stream variable=v2782 depth=17
  #pragma HLS stream variable=v2783 depth=17
  #pragma HLS stream variable=v2784 depth=17
  #pragma HLS stream variable=v2785 depth=17
  #pragma HLS array_partition variable=v2786 complete dim=1
  #pragma HLS array_partition variable=v2786 complete dim=2

  int32_t v107;	// L3324
  v107 = 0;	// L3325
  l_reduction_k107: for (int k107 = 0; k107 < 16; k107++) {	// L3326
  #pragma HLS pipeline II=1
    int8_t v2791 = v2782.read(); // v2782[k107];	// L3327
    int8_t a107;	// L3328
    a107 = v2791;	// L3329
    int8_t v2793 = v2783.read(); // v2783[k107];	// L3330
    int8_t b107;	// L3331
    b107 = v2793;	// L3332
    int8_t v2795 = a107;	// L3333
    int8_t v2796 = b107;	// L3334
    int16_t v2797 = v2795;	// L3335
    int16_t v2798 = v2796;	// L3336
    int16_t v2799 = v2797 * v2798;	// L3337
    int32_t v2800 = v107;	// L3338
    ap_int<33> v2801 = v2800;	// L3339
    ap_int<33> v2802 = v2799;	// L3340
    ap_int<33> v2803 = v2801 + v2802;	// L3341
    int32_t v2804 = v2803;	// L3342
    v107 = v2804;	// L3343
    int8_t v2805 = a107;	// L3344
    v2784.write(v2805); // v2784[k107] = v2805;	// L3345
    int8_t v2806 = b107;	// L3346
    v2785.write(v2806); // v2785[k107] = v2806;	// L3347
  }
  int32_t v2807 = v107;	// L3349
  v2786[v2787][v2788] = v2807;	// L3350
}

void PE_kernel_gemm_12_6(
  hls::stream< int8_t > &v2808 /* v2808[16] */,
  hls::stream< int8_t > &v2809 /* v2809[16] */,
  hls::stream< int8_t > &v2810 /* v2810[16] */,
  hls::stream< int8_t > &v2811 /* v2811[16] */,
  int32_t v2812[16][16],
  int v2813,
  int v2814
) {	// L3353
  #pragma HLS stream variable=v2808 depth=17
  #pragma HLS stream variable=v2809 depth=17
  #pragma HLS stream variable=v2810 depth=17
  #pragma HLS stream variable=v2811 depth=17
  #pragma HLS array_partition variable=v2812 complete dim=1
  #pragma HLS array_partition variable=v2812 complete dim=2

  int32_t v108;	// L3355
  v108 = 0;	// L3356
  l_reduction_k108: for (int k108 = 0; k108 < 16; k108++) {	// L3357
  #pragma HLS pipeline II=1
    int8_t v2817 = v2808.read(); // v2808[k108];	// L3358
    int8_t a108;	// L3359
    a108 = v2817;	// L3360
    int8_t v2819 = v2809.read(); // v2809[k108];	// L3361
    int8_t b108;	// L3362
    b108 = v2819;	// L3363
    int8_t v2821 = a108;	// L3364
    int8_t v2822 = b108;	// L3365
    int16_t v2823 = v2821;	// L3366
    int16_t v2824 = v2822;	// L3367
    int16_t v2825 = v2823 * v2824;	// L3368
    int32_t v2826 = v108;	// L3369
    ap_int<33> v2827 = v2826;	// L3370
    ap_int<33> v2828 = v2825;	// L3371
    ap_int<33> v2829 = v2827 + v2828;	// L3372
    int32_t v2830 = v2829;	// L3373
    v108 = v2830;	// L3374
    int8_t v2831 = a108;	// L3375
    v2810.write(v2831); // v2810[k108] = v2831;	// L3376
    int8_t v2832 = b108;	// L3377
    v2811.write(v2832); // v2811[k108] = v2832;	// L3378
  }
  int32_t v2833 = v108;	// L3380
  v2812[v2813][v2814] = v2833;	// L3381
}

void PE_kernel_gemm_13_6(
  hls::stream< int8_t > &v2834 /* v2834[16] */,
  hls::stream< int8_t > &v2835 /* v2835[16] */,
  hls::stream< int8_t > &v2836 /* v2836[16] */,
  hls::stream< int8_t > &v2837 /* v2837[16] */,
  int32_t v2838[16][16],
  int v2839,
  int v2840
) {	// L3384
  #pragma HLS stream variable=v2834 depth=17
  #pragma HLS stream variable=v2835 depth=17
  #pragma HLS stream variable=v2836 depth=17
  #pragma HLS stream variable=v2837 depth=17
  #pragma HLS array_partition variable=v2838 complete dim=1
  #pragma HLS array_partition variable=v2838 complete dim=2

  int32_t v109;	// L3386
  v109 = 0;	// L3387
  l_reduction_k109: for (int k109 = 0; k109 < 16; k109++) {	// L3388
  #pragma HLS pipeline II=1
    int8_t v2843 = v2834.read(); // v2834[k109];	// L3389
    int8_t a109;	// L3390
    a109 = v2843;	// L3391
    int8_t v2845 = v2835.read(); // v2835[k109];	// L3392
    int8_t b109;	// L3393
    b109 = v2845;	// L3394
    int8_t v2847 = a109;	// L3395
    int8_t v2848 = b109;	// L3396
    int16_t v2849 = v2847;	// L3397
    int16_t v2850 = v2848;	// L3398
    int16_t v2851 = v2849 * v2850;	// L3399
    int32_t v2852 = v109;	// L3400
    ap_int<33> v2853 = v2852;	// L3401
    ap_int<33> v2854 = v2851;	// L3402
    ap_int<33> v2855 = v2853 + v2854;	// L3403
    int32_t v2856 = v2855;	// L3404
    v109 = v2856;	// L3405
    int8_t v2857 = a109;	// L3406
    v2836.write(v2857); // v2836[k109] = v2857;	// L3407
    int8_t v2858 = b109;	// L3408
    v2837.write(v2858); // v2837[k109] = v2858;	// L3409
  }
  int32_t v2859 = v109;	// L3411
  v2838[v2839][v2840] = v2859;	// L3412
}

void PE_kernel_gemm_14_6(
  hls::stream< int8_t > &v2860 /* v2860[16] */,
  hls::stream< int8_t > &v2861 /* v2861[16] */,
  hls::stream< int8_t > &v2862 /* v2862[16] */,
  hls::stream< int8_t > &v2863 /* v2863[16] */,
  int32_t v2864[16][16],
  int v2865,
  int v2866
) {	// L3415
  #pragma HLS stream variable=v2860 depth=17
  #pragma HLS stream variable=v2861 depth=17
  #pragma HLS stream variable=v2862 depth=17
  #pragma HLS stream variable=v2863 depth=17
  #pragma HLS array_partition variable=v2864 complete dim=1
  #pragma HLS array_partition variable=v2864 complete dim=2

  int32_t v110;	// L3417
  v110 = 0;	// L3418
  l_reduction_k110: for (int k110 = 0; k110 < 16; k110++) {	// L3419
  #pragma HLS pipeline II=1
    int8_t v2869 = v2860.read(); // v2860[k110];	// L3420
    int8_t a110;	// L3421
    a110 = v2869;	// L3422
    int8_t v2871 = v2861.read(); // v2861[k110];	// L3423
    int8_t b110;	// L3424
    b110 = v2871;	// L3425
    int8_t v2873 = a110;	// L3426
    int8_t v2874 = b110;	// L3427
    int16_t v2875 = v2873;	// L3428
    int16_t v2876 = v2874;	// L3429
    int16_t v2877 = v2875 * v2876;	// L3430
    int32_t v2878 = v110;	// L3431
    ap_int<33> v2879 = v2878;	// L3432
    ap_int<33> v2880 = v2877;	// L3433
    ap_int<33> v2881 = v2879 + v2880;	// L3434
    int32_t v2882 = v2881;	// L3435
    v110 = v2882;	// L3436
    int8_t v2883 = a110;	// L3437
    v2862.write(v2883); // v2862[k110] = v2883;	// L3438
    int8_t v2884 = b110;	// L3439
    v2863.write(v2884); // v2863[k110] = v2884;	// L3440
  }
  int32_t v2885 = v110;	// L3442
  v2864[v2865][v2866] = v2885;	// L3443
}

void PE_kernel_gemm_15_6(
  hls::stream< int8_t > &v2886 /* v2886[16] */,
  hls::stream< int8_t > &v2887 /* v2887[16] */,
  hls::stream< int8_t > &v2888 /* v2888[16] */,
  hls::stream< int8_t > &v2889 /* v2889[16] */,
  int32_t v2890[16][16],
  int v2891,
  int v2892
) {	// L3446
  #pragma HLS stream variable=v2886 depth=17
  #pragma HLS stream variable=v2887 depth=17
  #pragma HLS stream variable=v2888 depth=17
  #pragma HLS stream variable=v2889 depth=17
  #pragma HLS array_partition variable=v2890 complete dim=1
  #pragma HLS array_partition variable=v2890 complete dim=2

  int32_t v111;	// L3448
  v111 = 0;	// L3449
  l_reduction_k111: for (int k111 = 0; k111 < 16; k111++) {	// L3450
  #pragma HLS pipeline II=1
    int8_t v2895 = v2886.read(); // v2886[k111];	// L3451
    int8_t a111;	// L3452
    a111 = v2895;	// L3453
    int8_t v2897 = v2887.read(); // v2887[k111];	// L3454
    int8_t b111;	// L3455
    b111 = v2897;	// L3456
    int8_t v2899 = a111;	// L3457
    int8_t v2900 = b111;	// L3458
    int16_t v2901 = v2899;	// L3459
    int16_t v2902 = v2900;	// L3460
    int16_t v2903 = v2901 * v2902;	// L3461
    int32_t v2904 = v111;	// L3462
    ap_int<33> v2905 = v2904;	// L3463
    ap_int<33> v2906 = v2903;	// L3464
    ap_int<33> v2907 = v2905 + v2906;	// L3465
    int32_t v2908 = v2907;	// L3466
    v111 = v2908;	// L3467
    int8_t v2909 = a111;	// L3468
    v2888.write(v2909); // v2888[k111] = v2909;	// L3469
    int8_t v2910 = b111;	// L3470
    v2889.write(v2910); // v2889[k111] = v2910;	// L3471
  }
  int32_t v2911 = v111;	// L3473
  v2890[v2891][v2892] = v2911;	// L3474
}

void PE_kernel_gemm_0_7(
  hls::stream< int8_t > &v2912 /* v2912[16] */,
  hls::stream< int8_t > &v2913 /* v2913[16] */,
  hls::stream< int8_t > &v2914 /* v2914[16] */,
  hls::stream< int8_t > &v2915 /* v2915[16] */,
  int32_t v2916[16][16],
  int v2917,
  int v2918
) {	// L3477
  #pragma HLS stream variable=v2912 depth=17
  #pragma HLS stream variable=v2913 depth=17
  #pragma HLS stream variable=v2914 depth=17
  #pragma HLS stream variable=v2915 depth=17
  #pragma HLS array_partition variable=v2916 complete dim=1
  #pragma HLS array_partition variable=v2916 complete dim=2

  int32_t v112;	// L3479
  v112 = 0;	// L3480
  l_reduction_k112: for (int k112 = 0; k112 < 16; k112++) {	// L3481
  #pragma HLS pipeline II=1
    int8_t v2921 = v2912.read(); // v2912[k112];	// L3482
    int8_t a112;	// L3483
    a112 = v2921;	// L3484
    int8_t v2923 = v2913.read(); // v2913[k112];	// L3485
    int8_t b112;	// L3486
    b112 = v2923;	// L3487
    int8_t v2925 = a112;	// L3488
    int8_t v2926 = b112;	// L3489
    int16_t v2927 = v2925;	// L3490
    int16_t v2928 = v2926;	// L3491
    int16_t v2929 = v2927 * v2928;	// L3492
    int32_t v2930 = v112;	// L3493
    ap_int<33> v2931 = v2930;	// L3494
    ap_int<33> v2932 = v2929;	// L3495
    ap_int<33> v2933 = v2931 + v2932;	// L3496
    int32_t v2934 = v2933;	// L3497
    v112 = v2934;	// L3498
    int8_t v2935 = a112;	// L3499
    v2914.write(v2935); // v2914[k112] = v2935;	// L3500
    int8_t v2936 = b112;	// L3501
    v2915.write(v2936); // v2915[k112] = v2936;	// L3502
  }
  int32_t v2937 = v112;	// L3504
  v2916[v2917][v2918] = v2937;	// L3505
}

void PE_kernel_gemm_1_7(
  hls::stream< int8_t > &v2938 /* v2938[16] */,
  hls::stream< int8_t > &v2939 /* v2939[16] */,
  hls::stream< int8_t > &v2940 /* v2940[16] */,
  hls::stream< int8_t > &v2941 /* v2941[16] */,
  int32_t v2942[16][16],
  int v2943,
  int v2944
) {	// L3508
  #pragma HLS stream variable=v2938 depth=17
  #pragma HLS stream variable=v2939 depth=17
  #pragma HLS stream variable=v2940 depth=17
  #pragma HLS stream variable=v2941 depth=17
  #pragma HLS array_partition variable=v2942 complete dim=1
  #pragma HLS array_partition variable=v2942 complete dim=2

  int32_t v113;	// L3510
  v113 = 0;	// L3511
  l_reduction_k113: for (int k113 = 0; k113 < 16; k113++) {	// L3512
  #pragma HLS pipeline II=1
    int8_t v2947 = v2938.read(); // v2938[k113];	// L3513
    int8_t a113;	// L3514
    a113 = v2947;	// L3515
    int8_t v2949 = v2939.read(); // v2939[k113];	// L3516
    int8_t b113;	// L3517
    b113 = v2949;	// L3518
    int8_t v2951 = a113;	// L3519
    int8_t v2952 = b113;	// L3520
    int16_t v2953 = v2951;	// L3521
    int16_t v2954 = v2952;	// L3522
    int16_t v2955 = v2953 * v2954;	// L3523
    int32_t v2956 = v113;	// L3524
    ap_int<33> v2957 = v2956;	// L3525
    ap_int<33> v2958 = v2955;	// L3526
    ap_int<33> v2959 = v2957 + v2958;	// L3527
    int32_t v2960 = v2959;	// L3528
    v113 = v2960;	// L3529
    int8_t v2961 = a113;	// L3530
    v2940.write(v2961); // v2940[k113] = v2961;	// L3531
    int8_t v2962 = b113;	// L3532
    v2941.write(v2962); // v2941[k113] = v2962;	// L3533
  }
  int32_t v2963 = v113;	// L3535
  v2942[v2943][v2944] = v2963;	// L3536
}

void PE_kernel_gemm_2_7(
  hls::stream< int8_t > &v2964 /* v2964[16] */,
  hls::stream< int8_t > &v2965 /* v2965[16] */,
  hls::stream< int8_t > &v2966 /* v2966[16] */,
  hls::stream< int8_t > &v2967 /* v2967[16] */,
  int32_t v2968[16][16],
  int v2969,
  int v2970
) {	// L3539
  #pragma HLS stream variable=v2964 depth=17
  #pragma HLS stream variable=v2965 depth=17
  #pragma HLS stream variable=v2966 depth=17
  #pragma HLS stream variable=v2967 depth=17
  #pragma HLS array_partition variable=v2968 complete dim=1
  #pragma HLS array_partition variable=v2968 complete dim=2

  int32_t v114;	// L3541
  v114 = 0;	// L3542
  l_reduction_k114: for (int k114 = 0; k114 < 16; k114++) {	// L3543
  #pragma HLS pipeline II=1
    int8_t v2973 = v2964.read(); // v2964[k114];	// L3544
    int8_t a114;	// L3545
    a114 = v2973;	// L3546
    int8_t v2975 = v2965.read(); // v2965[k114];	// L3547
    int8_t b114;	// L3548
    b114 = v2975;	// L3549
    int8_t v2977 = a114;	// L3550
    int8_t v2978 = b114;	// L3551
    int16_t v2979 = v2977;	// L3552
    int16_t v2980 = v2978;	// L3553
    int16_t v2981 = v2979 * v2980;	// L3554
    int32_t v2982 = v114;	// L3555
    ap_int<33> v2983 = v2982;	// L3556
    ap_int<33> v2984 = v2981;	// L3557
    ap_int<33> v2985 = v2983 + v2984;	// L3558
    int32_t v2986 = v2985;	// L3559
    v114 = v2986;	// L3560
    int8_t v2987 = a114;	// L3561
    v2966.write(v2987); // v2966[k114] = v2987;	// L3562
    int8_t v2988 = b114;	// L3563
    v2967.write(v2988); // v2967[k114] = v2988;	// L3564
  }
  int32_t v2989 = v114;	// L3566
  v2968[v2969][v2970] = v2989;	// L3567
}

void PE_kernel_gemm_3_7(
  hls::stream< int8_t > &v2990 /* v2990[16] */,
  hls::stream< int8_t > &v2991 /* v2991[16] */,
  hls::stream< int8_t > &v2992 /* v2992[16] */,
  hls::stream< int8_t > &v2993 /* v2993[16] */,
  int32_t v2994[16][16],
  int v2995,
  int v2996
) {	// L3570
  #pragma HLS stream variable=v2990 depth=17
  #pragma HLS stream variable=v2991 depth=17
  #pragma HLS stream variable=v2992 depth=17
  #pragma HLS stream variable=v2993 depth=17
  #pragma HLS array_partition variable=v2994 complete dim=1
  #pragma HLS array_partition variable=v2994 complete dim=2

  int32_t v115;	// L3572
  v115 = 0;	// L3573
  l_reduction_k115: for (int k115 = 0; k115 < 16; k115++) {	// L3574
  #pragma HLS pipeline II=1
    int8_t v2999 = v2990.read(); // v2990[k115];	// L3575
    int8_t a115;	// L3576
    a115 = v2999;	// L3577
    int8_t v3001 = v2991.read(); // v2991[k115];	// L3578
    int8_t b115;	// L3579
    b115 = v3001;	// L3580
    int8_t v3003 = a115;	// L3581
    int8_t v3004 = b115;	// L3582
    int16_t v3005 = v3003;	// L3583
    int16_t v3006 = v3004;	// L3584
    int16_t v3007 = v3005 * v3006;	// L3585
    int32_t v3008 = v115;	// L3586
    ap_int<33> v3009 = v3008;	// L3587
    ap_int<33> v3010 = v3007;	// L3588
    ap_int<33> v3011 = v3009 + v3010;	// L3589
    int32_t v3012 = v3011;	// L3590
    v115 = v3012;	// L3591
    int8_t v3013 = a115;	// L3592
    v2992.write(v3013); // v2992[k115] = v3013;	// L3593
    int8_t v3014 = b115;	// L3594
    v2993.write(v3014); // v2993[k115] = v3014;	// L3595
  }
  int32_t v3015 = v115;	// L3597
  v2994[v2995][v2996] = v3015;	// L3598
}

void PE_kernel_gemm_4_7(
  hls::stream< int8_t > &v3016 /* v3016[16] */,
  hls::stream< int8_t > &v3017 /* v3017[16] */,
  hls::stream< int8_t > &v3018 /* v3018[16] */,
  hls::stream< int8_t > &v3019 /* v3019[16] */,
  int32_t v3020[16][16],
  int v3021,
  int v3022
) {	// L3601
  #pragma HLS stream variable=v3016 depth=17
  #pragma HLS stream variable=v3017 depth=17
  #pragma HLS stream variable=v3018 depth=17
  #pragma HLS stream variable=v3019 depth=17
  #pragma HLS array_partition variable=v3020 complete dim=1
  #pragma HLS array_partition variable=v3020 complete dim=2

  int32_t v116;	// L3603
  v116 = 0;	// L3604
  l_reduction_k116: for (int k116 = 0; k116 < 16; k116++) {	// L3605
  #pragma HLS pipeline II=1
    int8_t v3025 = v3016.read(); // v3016[k116];	// L3606
    int8_t a116;	// L3607
    a116 = v3025;	// L3608
    int8_t v3027 = v3017.read(); // v3017[k116];	// L3609
    int8_t b116;	// L3610
    b116 = v3027;	// L3611
    int8_t v3029 = a116;	// L3612
    int8_t v3030 = b116;	// L3613
    int16_t v3031 = v3029;	// L3614
    int16_t v3032 = v3030;	// L3615
    int16_t v3033 = v3031 * v3032;	// L3616
    int32_t v3034 = v116;	// L3617
    ap_int<33> v3035 = v3034;	// L3618
    ap_int<33> v3036 = v3033;	// L3619
    ap_int<33> v3037 = v3035 + v3036;	// L3620
    int32_t v3038 = v3037;	// L3621
    v116 = v3038;	// L3622
    int8_t v3039 = a116;	// L3623
    v3018.write(v3039); // v3018[k116] = v3039;	// L3624
    int8_t v3040 = b116;	// L3625
    v3019.write(v3040); // v3019[k116] = v3040;	// L3626
  }
  int32_t v3041 = v116;	// L3628
  v3020[v3021][v3022] = v3041;	// L3629
}

void PE_kernel_gemm_5_7(
  hls::stream< int8_t > &v3042 /* v3042[16] */,
  hls::stream< int8_t > &v3043 /* v3043[16] */,
  hls::stream< int8_t > &v3044 /* v3044[16] */,
  hls::stream< int8_t > &v3045 /* v3045[16] */,
  int32_t v3046[16][16],
  int v3047,
  int v3048
) {	// L3632
  #pragma HLS stream variable=v3042 depth=17
  #pragma HLS stream variable=v3043 depth=17
  #pragma HLS stream variable=v3044 depth=17
  #pragma HLS stream variable=v3045 depth=17
  #pragma HLS array_partition variable=v3046 complete dim=1
  #pragma HLS array_partition variable=v3046 complete dim=2

  int32_t v117;	// L3634
  v117 = 0;	// L3635
  l_reduction_k117: for (int k117 = 0; k117 < 16; k117++) {	// L3636
  #pragma HLS pipeline II=1
    int8_t v3051 = v3042.read(); // v3042[k117];	// L3637
    int8_t a117;	// L3638
    a117 = v3051;	// L3639
    int8_t v3053 = v3043.read(); // v3043[k117];	// L3640
    int8_t b117;	// L3641
    b117 = v3053;	// L3642
    int8_t v3055 = a117;	// L3643
    int8_t v3056 = b117;	// L3644
    int16_t v3057 = v3055;	// L3645
    int16_t v3058 = v3056;	// L3646
    int16_t v3059 = v3057 * v3058;	// L3647
    int32_t v3060 = v117;	// L3648
    ap_int<33> v3061 = v3060;	// L3649
    ap_int<33> v3062 = v3059;	// L3650
    ap_int<33> v3063 = v3061 + v3062;	// L3651
    int32_t v3064 = v3063;	// L3652
    v117 = v3064;	// L3653
    int8_t v3065 = a117;	// L3654
    v3044.write(v3065); // v3044[k117] = v3065;	// L3655
    int8_t v3066 = b117;	// L3656
    v3045.write(v3066); // v3045[k117] = v3066;	// L3657
  }
  int32_t v3067 = v117;	// L3659
  v3046[v3047][v3048] = v3067;	// L3660
}

void PE_kernel_gemm_6_7(
  hls::stream< int8_t > &v3068 /* v3068[16] */,
  hls::stream< int8_t > &v3069 /* v3069[16] */,
  hls::stream< int8_t > &v3070 /* v3070[16] */,
  hls::stream< int8_t > &v3071 /* v3071[16] */,
  int32_t v3072[16][16],
  int v3073,
  int v3074
) {	// L3663
  #pragma HLS stream variable=v3068 depth=17
  #pragma HLS stream variable=v3069 depth=17
  #pragma HLS stream variable=v3070 depth=17
  #pragma HLS stream variable=v3071 depth=17
  #pragma HLS array_partition variable=v3072 complete dim=1
  #pragma HLS array_partition variable=v3072 complete dim=2

  int32_t v118;	// L3665
  v118 = 0;	// L3666
  l_reduction_k118: for (int k118 = 0; k118 < 16; k118++) {	// L3667
  #pragma HLS pipeline II=1
    int8_t v3077 = v3068.read(); // v3068[k118];	// L3668
    int8_t a118;	// L3669
    a118 = v3077;	// L3670
    int8_t v3079 = v3069.read(); // v3069[k118];	// L3671
    int8_t b118;	// L3672
    b118 = v3079;	// L3673
    int8_t v3081 = a118;	// L3674
    int8_t v3082 = b118;	// L3675
    int16_t v3083 = v3081;	// L3676
    int16_t v3084 = v3082;	// L3677
    int16_t v3085 = v3083 * v3084;	// L3678
    int32_t v3086 = v118;	// L3679
    ap_int<33> v3087 = v3086;	// L3680
    ap_int<33> v3088 = v3085;	// L3681
    ap_int<33> v3089 = v3087 + v3088;	// L3682
    int32_t v3090 = v3089;	// L3683
    v118 = v3090;	// L3684
    int8_t v3091 = a118;	// L3685
    v3070.write(v3091); // v3070[k118] = v3091;	// L3686
    int8_t v3092 = b118;	// L3687
    v3071.write(v3092); // v3071[k118] = v3092;	// L3688
  }
  int32_t v3093 = v118;	// L3690
  v3072[v3073][v3074] = v3093;	// L3691
}

void PE_kernel_gemm_7_7(
  hls::stream< int8_t > &v3094 /* v3094[16] */,
  hls::stream< int8_t > &v3095 /* v3095[16] */,
  hls::stream< int8_t > &v3096 /* v3096[16] */,
  hls::stream< int8_t > &v3097 /* v3097[16] */,
  int32_t v3098[16][16],
  int v3099,
  int v3100
) {	// L3694
  #pragma HLS stream variable=v3094 depth=17
  #pragma HLS stream variable=v3095 depth=17
  #pragma HLS stream variable=v3096 depth=17
  #pragma HLS stream variable=v3097 depth=17
  #pragma HLS array_partition variable=v3098 complete dim=1
  #pragma HLS array_partition variable=v3098 complete dim=2

  int32_t v119;	// L3696
  v119 = 0;	// L3697
  l_reduction_k119: for (int k119 = 0; k119 < 16; k119++) {	// L3698
  #pragma HLS pipeline II=1
    int8_t v3103 = v3094.read(); // v3094[k119];	// L3699
    int8_t a119;	// L3700
    a119 = v3103;	// L3701
    int8_t v3105 = v3095.read(); // v3095[k119];	// L3702
    int8_t b119;	// L3703
    b119 = v3105;	// L3704
    int8_t v3107 = a119;	// L3705
    int8_t v3108 = b119;	// L3706
    int16_t v3109 = v3107;	// L3707
    int16_t v3110 = v3108;	// L3708
    int16_t v3111 = v3109 * v3110;	// L3709
    int32_t v3112 = v119;	// L3710
    ap_int<33> v3113 = v3112;	// L3711
    ap_int<33> v3114 = v3111;	// L3712
    ap_int<33> v3115 = v3113 + v3114;	// L3713
    int32_t v3116 = v3115;	// L3714
    v119 = v3116;	// L3715
    int8_t v3117 = a119;	// L3716
    v3096.write(v3117); // v3096[k119] = v3117;	// L3717
    int8_t v3118 = b119;	// L3718
    v3097.write(v3118); // v3097[k119] = v3118;	// L3719
  }
  int32_t v3119 = v119;	// L3721
  v3098[v3099][v3100] = v3119;	// L3722
}

void PE_kernel_gemm_8_7(
  hls::stream< int8_t > &v3120 /* v3120[16] */,
  hls::stream< int8_t > &v3121 /* v3121[16] */,
  hls::stream< int8_t > &v3122 /* v3122[16] */,
  hls::stream< int8_t > &v3123 /* v3123[16] */,
  int32_t v3124[16][16],
  int v3125,
  int v3126
) {	// L3725
  #pragma HLS stream variable=v3120 depth=17
  #pragma HLS stream variable=v3121 depth=17
  #pragma HLS stream variable=v3122 depth=17
  #pragma HLS stream variable=v3123 depth=17
  #pragma HLS array_partition variable=v3124 complete dim=1
  #pragma HLS array_partition variable=v3124 complete dim=2

  int32_t v120;	// L3727
  v120 = 0;	// L3728
  l_reduction_k120: for (int k120 = 0; k120 < 16; k120++) {	// L3729
  #pragma HLS pipeline II=1
    int8_t v3129 = v3120.read(); // v3120[k120];	// L3730
    int8_t a120;	// L3731
    a120 = v3129;	// L3732
    int8_t v3131 = v3121.read(); // v3121[k120];	// L3733
    int8_t b120;	// L3734
    b120 = v3131;	// L3735
    int8_t v3133 = a120;	// L3736
    int8_t v3134 = b120;	// L3737
    int16_t v3135 = v3133;	// L3738
    int16_t v3136 = v3134;	// L3739
    int16_t v3137 = v3135 * v3136;	// L3740
    int32_t v3138 = v120;	// L3741
    ap_int<33> v3139 = v3138;	// L3742
    ap_int<33> v3140 = v3137;	// L3743
    ap_int<33> v3141 = v3139 + v3140;	// L3744
    int32_t v3142 = v3141;	// L3745
    v120 = v3142;	// L3746
    int8_t v3143 = a120;	// L3747
    v3122.write(v3143); // v3122[k120] = v3143;	// L3748
    int8_t v3144 = b120;	// L3749
    v3123.write(v3144); // v3123[k120] = v3144;	// L3750
  }
  int32_t v3145 = v120;	// L3752
  v3124[v3125][v3126] = v3145;	// L3753
}

void PE_kernel_gemm_9_7(
  hls::stream< int8_t > &v3146 /* v3146[16] */,
  hls::stream< int8_t > &v3147 /* v3147[16] */,
  hls::stream< int8_t > &v3148 /* v3148[16] */,
  hls::stream< int8_t > &v3149 /* v3149[16] */,
  int32_t v3150[16][16],
  int v3151,
  int v3152
) {	// L3756
  #pragma HLS stream variable=v3146 depth=17
  #pragma HLS stream variable=v3147 depth=17
  #pragma HLS stream variable=v3148 depth=17
  #pragma HLS stream variable=v3149 depth=17
  #pragma HLS array_partition variable=v3150 complete dim=1
  #pragma HLS array_partition variable=v3150 complete dim=2

  int32_t v121;	// L3758
  v121 = 0;	// L3759
  l_reduction_k121: for (int k121 = 0; k121 < 16; k121++) {	// L3760
  #pragma HLS pipeline II=1
    int8_t v3155 = v3146.read(); // v3146[k121];	// L3761
    int8_t a121;	// L3762
    a121 = v3155;	// L3763
    int8_t v3157 = v3147.read(); // v3147[k121];	// L3764
    int8_t b121;	// L3765
    b121 = v3157;	// L3766
    int8_t v3159 = a121;	// L3767
    int8_t v3160 = b121;	// L3768
    int16_t v3161 = v3159;	// L3769
    int16_t v3162 = v3160;	// L3770
    int16_t v3163 = v3161 * v3162;	// L3771
    int32_t v3164 = v121;	// L3772
    ap_int<33> v3165 = v3164;	// L3773
    ap_int<33> v3166 = v3163;	// L3774
    ap_int<33> v3167 = v3165 + v3166;	// L3775
    int32_t v3168 = v3167;	// L3776
    v121 = v3168;	// L3777
    int8_t v3169 = a121;	// L3778
    v3148.write(v3169); // v3148[k121] = v3169;	// L3779
    int8_t v3170 = b121;	// L3780
    v3149.write(v3170); // v3149[k121] = v3170;	// L3781
  }
  int32_t v3171 = v121;	// L3783
  v3150[v3151][v3152] = v3171;	// L3784
}

void PE_kernel_gemm_10_7(
  hls::stream< int8_t > &v3172 /* v3172[16] */,
  hls::stream< int8_t > &v3173 /* v3173[16] */,
  hls::stream< int8_t > &v3174 /* v3174[16] */,
  hls::stream< int8_t > &v3175 /* v3175[16] */,
  int32_t v3176[16][16],
  int v3177,
  int v3178
) {	// L3787
  #pragma HLS stream variable=v3172 depth=17
  #pragma HLS stream variable=v3173 depth=17
  #pragma HLS stream variable=v3174 depth=17
  #pragma HLS stream variable=v3175 depth=17
  #pragma HLS array_partition variable=v3176 complete dim=1
  #pragma HLS array_partition variable=v3176 complete dim=2

  int32_t v122;	// L3789
  v122 = 0;	// L3790
  l_reduction_k122: for (int k122 = 0; k122 < 16; k122++) {	// L3791
  #pragma HLS pipeline II=1
    int8_t v3181 = v3172.read(); // v3172[k122];	// L3792
    int8_t a122;	// L3793
    a122 = v3181;	// L3794
    int8_t v3183 = v3173.read(); // v3173[k122];	// L3795
    int8_t b122;	// L3796
    b122 = v3183;	// L3797
    int8_t v3185 = a122;	// L3798
    int8_t v3186 = b122;	// L3799
    int16_t v3187 = v3185;	// L3800
    int16_t v3188 = v3186;	// L3801
    int16_t v3189 = v3187 * v3188;	// L3802
    int32_t v3190 = v122;	// L3803
    ap_int<33> v3191 = v3190;	// L3804
    ap_int<33> v3192 = v3189;	// L3805
    ap_int<33> v3193 = v3191 + v3192;	// L3806
    int32_t v3194 = v3193;	// L3807
    v122 = v3194;	// L3808
    int8_t v3195 = a122;	// L3809
    v3174.write(v3195); // v3174[k122] = v3195;	// L3810
    int8_t v3196 = b122;	// L3811
    v3175.write(v3196); // v3175[k122] = v3196;	// L3812
  }
  int32_t v3197 = v122;	// L3814
  v3176[v3177][v3178] = v3197;	// L3815
}

void PE_kernel_gemm_11_7(
  hls::stream< int8_t > &v3198 /* v3198[16] */,
  hls::stream< int8_t > &v3199 /* v3199[16] */,
  hls::stream< int8_t > &v3200 /* v3200[16] */,
  hls::stream< int8_t > &v3201 /* v3201[16] */,
  int32_t v3202[16][16],
  int v3203,
  int v3204
) {	// L3818
  #pragma HLS stream variable=v3198 depth=17
  #pragma HLS stream variable=v3199 depth=17
  #pragma HLS stream variable=v3200 depth=17
  #pragma HLS stream variable=v3201 depth=17
  #pragma HLS array_partition variable=v3202 complete dim=1
  #pragma HLS array_partition variable=v3202 complete dim=2

  int32_t v123;	// L3820
  v123 = 0;	// L3821
  l_reduction_k123: for (int k123 = 0; k123 < 16; k123++) {	// L3822
  #pragma HLS pipeline II=1
    int8_t v3207 = v3198.read(); // v3198[k123];	// L3823
    int8_t a123;	// L3824
    a123 = v3207;	// L3825
    int8_t v3209 = v3199.read(); // v3199[k123];	// L3826
    int8_t b123;	// L3827
    b123 = v3209;	// L3828
    int8_t v3211 = a123;	// L3829
    int8_t v3212 = b123;	// L3830
    int16_t v3213 = v3211;	// L3831
    int16_t v3214 = v3212;	// L3832
    int16_t v3215 = v3213 * v3214;	// L3833
    int32_t v3216 = v123;	// L3834
    ap_int<33> v3217 = v3216;	// L3835
    ap_int<33> v3218 = v3215;	// L3836
    ap_int<33> v3219 = v3217 + v3218;	// L3837
    int32_t v3220 = v3219;	// L3838
    v123 = v3220;	// L3839
    int8_t v3221 = a123;	// L3840
    v3200.write(v3221); // v3200[k123] = v3221;	// L3841
    int8_t v3222 = b123;	// L3842
    v3201.write(v3222); // v3201[k123] = v3222;	// L3843
  }
  int32_t v3223 = v123;	// L3845
  v3202[v3203][v3204] = v3223;	// L3846
}

void PE_kernel_gemm_12_7(
  hls::stream< int8_t > &v3224 /* v3224[16] */,
  hls::stream< int8_t > &v3225 /* v3225[16] */,
  hls::stream< int8_t > &v3226 /* v3226[16] */,
  hls::stream< int8_t > &v3227 /* v3227[16] */,
  int32_t v3228[16][16],
  int v3229,
  int v3230
) {	// L3849
  #pragma HLS stream variable=v3224 depth=17
  #pragma HLS stream variable=v3225 depth=17
  #pragma HLS stream variable=v3226 depth=17
  #pragma HLS stream variable=v3227 depth=17
  #pragma HLS array_partition variable=v3228 complete dim=1
  #pragma HLS array_partition variable=v3228 complete dim=2

  int32_t v124;	// L3851
  v124 = 0;	// L3852
  l_reduction_k124: for (int k124 = 0; k124 < 16; k124++) {	// L3853
  #pragma HLS pipeline II=1
    int8_t v3233 = v3224.read(); // v3224[k124];	// L3854
    int8_t a124;	// L3855
    a124 = v3233;	// L3856
    int8_t v3235 = v3225.read(); // v3225[k124];	// L3857
    int8_t b124;	// L3858
    b124 = v3235;	// L3859
    int8_t v3237 = a124;	// L3860
    int8_t v3238 = b124;	// L3861
    int16_t v3239 = v3237;	// L3862
    int16_t v3240 = v3238;	// L3863
    int16_t v3241 = v3239 * v3240;	// L3864
    int32_t v3242 = v124;	// L3865
    ap_int<33> v3243 = v3242;	// L3866
    ap_int<33> v3244 = v3241;	// L3867
    ap_int<33> v3245 = v3243 + v3244;	// L3868
    int32_t v3246 = v3245;	// L3869
    v124 = v3246;	// L3870
    int8_t v3247 = a124;	// L3871
    v3226.write(v3247); // v3226[k124] = v3247;	// L3872
    int8_t v3248 = b124;	// L3873
    v3227.write(v3248); // v3227[k124] = v3248;	// L3874
  }
  int32_t v3249 = v124;	// L3876
  v3228[v3229][v3230] = v3249;	// L3877
}

void PE_kernel_gemm_13_7(
  hls::stream< int8_t > &v3250 /* v3250[16] */,
  hls::stream< int8_t > &v3251 /* v3251[16] */,
  hls::stream< int8_t > &v3252 /* v3252[16] */,
  hls::stream< int8_t > &v3253 /* v3253[16] */,
  int32_t v3254[16][16],
  int v3255,
  int v3256
) {	// L3880
  #pragma HLS stream variable=v3250 depth=17
  #pragma HLS stream variable=v3251 depth=17
  #pragma HLS stream variable=v3252 depth=17
  #pragma HLS stream variable=v3253 depth=17
  #pragma HLS array_partition variable=v3254 complete dim=1
  #pragma HLS array_partition variable=v3254 complete dim=2

  int32_t v125;	// L3882
  v125 = 0;	// L3883
  l_reduction_k125: for (int k125 = 0; k125 < 16; k125++) {	// L3884
  #pragma HLS pipeline II=1
    int8_t v3259 = v3250.read(); // v3250[k125];	// L3885
    int8_t a125;	// L3886
    a125 = v3259;	// L3887
    int8_t v3261 = v3251.read(); // v3251[k125];	// L3888
    int8_t b125;	// L3889
    b125 = v3261;	// L3890
    int8_t v3263 = a125;	// L3891
    int8_t v3264 = b125;	// L3892
    int16_t v3265 = v3263;	// L3893
    int16_t v3266 = v3264;	// L3894
    int16_t v3267 = v3265 * v3266;	// L3895
    int32_t v3268 = v125;	// L3896
    ap_int<33> v3269 = v3268;	// L3897
    ap_int<33> v3270 = v3267;	// L3898
    ap_int<33> v3271 = v3269 + v3270;	// L3899
    int32_t v3272 = v3271;	// L3900
    v125 = v3272;	// L3901
    int8_t v3273 = a125;	// L3902
    v3252.write(v3273); // v3252[k125] = v3273;	// L3903
    int8_t v3274 = b125;	// L3904
    v3253.write(v3274); // v3253[k125] = v3274;	// L3905
  }
  int32_t v3275 = v125;	// L3907
  v3254[v3255][v3256] = v3275;	// L3908
}

void PE_kernel_gemm_14_7(
  hls::stream< int8_t > &v3276 /* v3276[16] */,
  hls::stream< int8_t > &v3277 /* v3277[16] */,
  hls::stream< int8_t > &v3278 /* v3278[16] */,
  hls::stream< int8_t > &v3279 /* v3279[16] */,
  int32_t v3280[16][16],
  int v3281,
  int v3282
) {	// L3911
  #pragma HLS stream variable=v3276 depth=17
  #pragma HLS stream variable=v3277 depth=17
  #pragma HLS stream variable=v3278 depth=17
  #pragma HLS stream variable=v3279 depth=17
  #pragma HLS array_partition variable=v3280 complete dim=1
  #pragma HLS array_partition variable=v3280 complete dim=2

  int32_t v126;	// L3913
  v126 = 0;	// L3914
  l_reduction_k126: for (int k126 = 0; k126 < 16; k126++) {	// L3915
  #pragma HLS pipeline II=1
    int8_t v3285 = v3276.read(); // v3276[k126];	// L3916
    int8_t a126;	// L3917
    a126 = v3285;	// L3918
    int8_t v3287 = v3277.read(); // v3277[k126];	// L3919
    int8_t b126;	// L3920
    b126 = v3287;	// L3921
    int8_t v3289 = a126;	// L3922
    int8_t v3290 = b126;	// L3923
    int16_t v3291 = v3289;	// L3924
    int16_t v3292 = v3290;	// L3925
    int16_t v3293 = v3291 * v3292;	// L3926
    int32_t v3294 = v126;	// L3927
    ap_int<33> v3295 = v3294;	// L3928
    ap_int<33> v3296 = v3293;	// L3929
    ap_int<33> v3297 = v3295 + v3296;	// L3930
    int32_t v3298 = v3297;	// L3931
    v126 = v3298;	// L3932
    int8_t v3299 = a126;	// L3933
    v3278.write(v3299); // v3278[k126] = v3299;	// L3934
    int8_t v3300 = b126;	// L3935
    v3279.write(v3300); // v3279[k126] = v3300;	// L3936
  }
  int32_t v3301 = v126;	// L3938
  v3280[v3281][v3282] = v3301;	// L3939
}

void PE_kernel_gemm_15_7(
  hls::stream< int8_t > &v3302 /* v3302[16] */,
  hls::stream< int8_t > &v3303 /* v3303[16] */,
  hls::stream< int8_t > &v3304 /* v3304[16] */,
  hls::stream< int8_t > &v3305 /* v3305[16] */,
  int32_t v3306[16][16],
  int v3307,
  int v3308
) {	// L3942
  #pragma HLS stream variable=v3302 depth=17
  #pragma HLS stream variable=v3303 depth=17
  #pragma HLS stream variable=v3304 depth=17
  #pragma HLS stream variable=v3305 depth=17
  #pragma HLS array_partition variable=v3306 complete dim=1
  #pragma HLS array_partition variable=v3306 complete dim=2

  int32_t v127;	// L3944
  v127 = 0;	// L3945
  l_reduction_k127: for (int k127 = 0; k127 < 16; k127++) {	// L3946
  #pragma HLS pipeline II=1
    int8_t v3311 = v3302.read(); // v3302[k127];	// L3947
    int8_t a127;	// L3948
    a127 = v3311;	// L3949
    int8_t v3313 = v3303.read(); // v3303[k127];	// L3950
    int8_t b127;	// L3951
    b127 = v3313;	// L3952
    int8_t v3315 = a127;	// L3953
    int8_t v3316 = b127;	// L3954
    int16_t v3317 = v3315;	// L3955
    int16_t v3318 = v3316;	// L3956
    int16_t v3319 = v3317 * v3318;	// L3957
    int32_t v3320 = v127;	// L3958
    ap_int<33> v3321 = v3320;	// L3959
    ap_int<33> v3322 = v3319;	// L3960
    ap_int<33> v3323 = v3321 + v3322;	// L3961
    int32_t v3324 = v3323;	// L3962
    v127 = v3324;	// L3963
    int8_t v3325 = a127;	// L3964
    v3304.write(v3325); // v3304[k127] = v3325;	// L3965
    int8_t v3326 = b127;	// L3966
    v3305.write(v3326); // v3305[k127] = v3326;	// L3967
  }
  int32_t v3327 = v127;	// L3969
  v3306[v3307][v3308] = v3327;	// L3970
}

void PE_kernel_gemm_0_8(
  hls::stream< int8_t > &v3328 /* v3328[16] */,
  hls::stream< int8_t > &v3329 /* v3329[16] */,
  hls::stream< int8_t > &v3330 /* v3330[16] */,
  hls::stream< int8_t > &v3331 /* v3331[16] */,
  int32_t v3332[16][16],
  int v3333,
  int v3334
) {	// L3973
  #pragma HLS stream variable=v3328 depth=17
  #pragma HLS stream variable=v3329 depth=17
  #pragma HLS stream variable=v3330 depth=17
  #pragma HLS stream variable=v3331 depth=17
  #pragma HLS array_partition variable=v3332 complete dim=1
  #pragma HLS array_partition variable=v3332 complete dim=2

  int32_t v128;	// L3975
  v128 = 0;	// L3976
  l_reduction_k128: for (int k128 = 0; k128 < 16; k128++) {	// L3977
  #pragma HLS pipeline II=1
    int8_t v3337 = v3328.read(); // v3328[k128];	// L3978
    int8_t a128;	// L3979
    a128 = v3337;	// L3980
    int8_t v3339 = v3329.read(); // v3329[k128];	// L3981
    int8_t b128;	// L3982
    b128 = v3339;	// L3983
    int8_t v3341 = a128;	// L3984
    int8_t v3342 = b128;	// L3985
    int16_t v3343 = v3341;	// L3986
    int16_t v3344 = v3342;	// L3987
    int16_t v3345 = v3343 * v3344;	// L3988
    int32_t v3346 = v128;	// L3989
    ap_int<33> v3347 = v3346;	// L3990
    ap_int<33> v3348 = v3345;	// L3991
    ap_int<33> v3349 = v3347 + v3348;	// L3992
    int32_t v3350 = v3349;	// L3993
    v128 = v3350;	// L3994
    int8_t v3351 = a128;	// L3995
    v3330.write(v3351); // v3330[k128] = v3351;	// L3996
    int8_t v3352 = b128;	// L3997
    v3331.write(v3352); // v3331[k128] = v3352;	// L3998
  }
  int32_t v3353 = v128;	// L4000
  v3332[v3333][v3334] = v3353;	// L4001
}

void PE_kernel_gemm_1_8(
  hls::stream< int8_t > &v3354 /* v3354[16] */,
  hls::stream< int8_t > &v3355 /* v3355[16] */,
  hls::stream< int8_t > &v3356 /* v3356[16] */,
  hls::stream< int8_t > &v3357 /* v3357[16] */,
  int32_t v3358[16][16],
  int v3359,
  int v3360
) {	// L4004
  #pragma HLS stream variable=v3354 depth=17
  #pragma HLS stream variable=v3355 depth=17
  #pragma HLS stream variable=v3356 depth=17
  #pragma HLS stream variable=v3357 depth=17
  #pragma HLS array_partition variable=v3358 complete dim=1
  #pragma HLS array_partition variable=v3358 complete dim=2

  int32_t v129;	// L4006
  v129 = 0;	// L4007
  l_reduction_k129: for (int k129 = 0; k129 < 16; k129++) {	// L4008
  #pragma HLS pipeline II=1
    int8_t v3363 = v3354.read(); // v3354[k129];	// L4009
    int8_t a129;	// L4010
    a129 = v3363;	// L4011
    int8_t v3365 = v3355.read(); // v3355[k129];	// L4012
    int8_t b129;	// L4013
    b129 = v3365;	// L4014
    int8_t v3367 = a129;	// L4015
    int8_t v3368 = b129;	// L4016
    int16_t v3369 = v3367;	// L4017
    int16_t v3370 = v3368;	// L4018
    int16_t v3371 = v3369 * v3370;	// L4019
    int32_t v3372 = v129;	// L4020
    ap_int<33> v3373 = v3372;	// L4021
    ap_int<33> v3374 = v3371;	// L4022
    ap_int<33> v3375 = v3373 + v3374;	// L4023
    int32_t v3376 = v3375;	// L4024
    v129 = v3376;	// L4025
    int8_t v3377 = a129;	// L4026
    v3356.write(v3377); // v3356[k129] = v3377;	// L4027
    int8_t v3378 = b129;	// L4028
    v3357.write(v3378); // v3357[k129] = v3378;	// L4029
  }
  int32_t v3379 = v129;	// L4031
  v3358[v3359][v3360] = v3379;	// L4032
}

void PE_kernel_gemm_2_8(
  hls::stream< int8_t > &v3380 /* v3380[16] */,
  hls::stream< int8_t > &v3381 /* v3381[16] */,
  hls::stream< int8_t > &v3382 /* v3382[16] */,
  hls::stream< int8_t > &v3383 /* v3383[16] */,
  int32_t v3384[16][16],
  int v3385,
  int v3386
) {	// L4035
  #pragma HLS stream variable=v3380 depth=17
  #pragma HLS stream variable=v3381 depth=17
  #pragma HLS stream variable=v3382 depth=17
  #pragma HLS stream variable=v3383 depth=17
  #pragma HLS array_partition variable=v3384 complete dim=1
  #pragma HLS array_partition variable=v3384 complete dim=2

  int32_t v130;	// L4037
  v130 = 0;	// L4038
  l_reduction_k130: for (int k130 = 0; k130 < 16; k130++) {	// L4039
  #pragma HLS pipeline II=1
    int8_t v3389 = v3380.read(); // v3380[k130];	// L4040
    int8_t a130;	// L4041
    a130 = v3389;	// L4042
    int8_t v3391 = v3381.read(); // v3381[k130];	// L4043
    int8_t b130;	// L4044
    b130 = v3391;	// L4045
    int8_t v3393 = a130;	// L4046
    int8_t v3394 = b130;	// L4047
    int16_t v3395 = v3393;	// L4048
    int16_t v3396 = v3394;	// L4049
    int16_t v3397 = v3395 * v3396;	// L4050
    int32_t v3398 = v130;	// L4051
    ap_int<33> v3399 = v3398;	// L4052
    ap_int<33> v3400 = v3397;	// L4053
    ap_int<33> v3401 = v3399 + v3400;	// L4054
    int32_t v3402 = v3401;	// L4055
    v130 = v3402;	// L4056
    int8_t v3403 = a130;	// L4057
    v3382.write(v3403); // v3382[k130] = v3403;	// L4058
    int8_t v3404 = b130;	// L4059
    v3383.write(v3404); // v3383[k130] = v3404;	// L4060
  }
  int32_t v3405 = v130;	// L4062
  v3384[v3385][v3386] = v3405;	// L4063
}

void PE_kernel_gemm_3_8(
  hls::stream< int8_t > &v3406 /* v3406[16] */,
  hls::stream< int8_t > &v3407 /* v3407[16] */,
  hls::stream< int8_t > &v3408 /* v3408[16] */,
  hls::stream< int8_t > &v3409 /* v3409[16] */,
  int32_t v3410[16][16],
  int v3411,
  int v3412
) {	// L4066
  #pragma HLS stream variable=v3406 depth=17
  #pragma HLS stream variable=v3407 depth=17
  #pragma HLS stream variable=v3408 depth=17
  #pragma HLS stream variable=v3409 depth=17
  #pragma HLS array_partition variable=v3410 complete dim=1
  #pragma HLS array_partition variable=v3410 complete dim=2

  int32_t v131;	// L4068
  v131 = 0;	// L4069
  l_reduction_k131: for (int k131 = 0; k131 < 16; k131++) {	// L4070
  #pragma HLS pipeline II=1
    int8_t v3415 = v3406.read(); // v3406[k131];	// L4071
    int8_t a131;	// L4072
    a131 = v3415;	// L4073
    int8_t v3417 = v3407.read(); // v3407[k131];	// L4074
    int8_t b131;	// L4075
    b131 = v3417;	// L4076
    int8_t v3419 = a131;	// L4077
    int8_t v3420 = b131;	// L4078
    int16_t v3421 = v3419;	// L4079
    int16_t v3422 = v3420;	// L4080
    int16_t v3423 = v3421 * v3422;	// L4081
    int32_t v3424 = v131;	// L4082
    ap_int<33> v3425 = v3424;	// L4083
    ap_int<33> v3426 = v3423;	// L4084
    ap_int<33> v3427 = v3425 + v3426;	// L4085
    int32_t v3428 = v3427;	// L4086
    v131 = v3428;	// L4087
    int8_t v3429 = a131;	// L4088
    v3408.write(v3429); // v3408[k131] = v3429;	// L4089
    int8_t v3430 = b131;	// L4090
    v3409.write(v3430); // v3409[k131] = v3430;	// L4091
  }
  int32_t v3431 = v131;	// L4093
  v3410[v3411][v3412] = v3431;	// L4094
}

void PE_kernel_gemm_4_8(
  hls::stream< int8_t > &v3432 /* v3432[16] */,
  hls::stream< int8_t > &v3433 /* v3433[16] */,
  hls::stream< int8_t > &v3434 /* v3434[16] */,
  hls::stream< int8_t > &v3435 /* v3435[16] */,
  int32_t v3436[16][16],
  int v3437,
  int v3438
) {	// L4097
  #pragma HLS stream variable=v3432 depth=17
  #pragma HLS stream variable=v3433 depth=17
  #pragma HLS stream variable=v3434 depth=17
  #pragma HLS stream variable=v3435 depth=17
  #pragma HLS array_partition variable=v3436 complete dim=1
  #pragma HLS array_partition variable=v3436 complete dim=2

  int32_t v132;	// L4099
  v132 = 0;	// L4100
  l_reduction_k132: for (int k132 = 0; k132 < 16; k132++) {	// L4101
  #pragma HLS pipeline II=1
    int8_t v3441 = v3432.read(); // v3432[k132];	// L4102
    int8_t a132;	// L4103
    a132 = v3441;	// L4104
    int8_t v3443 = v3433.read(); // v3433[k132];	// L4105
    int8_t b132;	// L4106
    b132 = v3443;	// L4107
    int8_t v3445 = a132;	// L4108
    int8_t v3446 = b132;	// L4109
    int16_t v3447 = v3445;	// L4110
    int16_t v3448 = v3446;	// L4111
    int16_t v3449 = v3447 * v3448;	// L4112
    int32_t v3450 = v132;	// L4113
    ap_int<33> v3451 = v3450;	// L4114
    ap_int<33> v3452 = v3449;	// L4115
    ap_int<33> v3453 = v3451 + v3452;	// L4116
    int32_t v3454 = v3453;	// L4117
    v132 = v3454;	// L4118
    int8_t v3455 = a132;	// L4119
    v3434.write(v3455); // v3434[k132] = v3455;	// L4120
    int8_t v3456 = b132;	// L4121
    v3435.write(v3456); // v3435[k132] = v3456;	// L4122
  }
  int32_t v3457 = v132;	// L4124
  v3436[v3437][v3438] = v3457;	// L4125
}

void PE_kernel_gemm_5_8(
  hls::stream< int8_t > &v3458 /* v3458[16] */,
  hls::stream< int8_t > &v3459 /* v3459[16] */,
  hls::stream< int8_t > &v3460 /* v3460[16] */,
  hls::stream< int8_t > &v3461 /* v3461[16] */,
  int32_t v3462[16][16],
  int v3463,
  int v3464
) {	// L4128
  #pragma HLS stream variable=v3458 depth=17
  #pragma HLS stream variable=v3459 depth=17
  #pragma HLS stream variable=v3460 depth=17
  #pragma HLS stream variable=v3461 depth=17
  #pragma HLS array_partition variable=v3462 complete dim=1
  #pragma HLS array_partition variable=v3462 complete dim=2

  int32_t v133;	// L4130
  v133 = 0;	// L4131
  l_reduction_k133: for (int k133 = 0; k133 < 16; k133++) {	// L4132
  #pragma HLS pipeline II=1
    int8_t v3467 = v3458.read(); // v3458[k133];	// L4133
    int8_t a133;	// L4134
    a133 = v3467;	// L4135
    int8_t v3469 = v3459.read(); // v3459[k133];	// L4136
    int8_t b133;	// L4137
    b133 = v3469;	// L4138
    int8_t v3471 = a133;	// L4139
    int8_t v3472 = b133;	// L4140
    int16_t v3473 = v3471;	// L4141
    int16_t v3474 = v3472;	// L4142
    int16_t v3475 = v3473 * v3474;	// L4143
    int32_t v3476 = v133;	// L4144
    ap_int<33> v3477 = v3476;	// L4145
    ap_int<33> v3478 = v3475;	// L4146
    ap_int<33> v3479 = v3477 + v3478;	// L4147
    int32_t v3480 = v3479;	// L4148
    v133 = v3480;	// L4149
    int8_t v3481 = a133;	// L4150
    v3460.write(v3481); // v3460[k133] = v3481;	// L4151
    int8_t v3482 = b133;	// L4152
    v3461.write(v3482); // v3461[k133] = v3482;	// L4153
  }
  int32_t v3483 = v133;	// L4155
  v3462[v3463][v3464] = v3483;	// L4156
}

void PE_kernel_gemm_6_8(
  hls::stream< int8_t > &v3484 /* v3484[16] */,
  hls::stream< int8_t > &v3485 /* v3485[16] */,
  hls::stream< int8_t > &v3486 /* v3486[16] */,
  hls::stream< int8_t > &v3487 /* v3487[16] */,
  int32_t v3488[16][16],
  int v3489,
  int v3490
) {	// L4159
  #pragma HLS stream variable=v3484 depth=17
  #pragma HLS stream variable=v3485 depth=17
  #pragma HLS stream variable=v3486 depth=17
  #pragma HLS stream variable=v3487 depth=17
  #pragma HLS array_partition variable=v3488 complete dim=1
  #pragma HLS array_partition variable=v3488 complete dim=2

  int32_t v134;	// L4161
  v134 = 0;	// L4162
  l_reduction_k134: for (int k134 = 0; k134 < 16; k134++) {	// L4163
  #pragma HLS pipeline II=1
    int8_t v3493 = v3484.read(); // v3484[k134];	// L4164
    int8_t a134;	// L4165
    a134 = v3493;	// L4166
    int8_t v3495 = v3485.read(); // v3485[k134];	// L4167
    int8_t b134;	// L4168
    b134 = v3495;	// L4169
    int8_t v3497 = a134;	// L4170
    int8_t v3498 = b134;	// L4171
    int16_t v3499 = v3497;	// L4172
    int16_t v3500 = v3498;	// L4173
    int16_t v3501 = v3499 * v3500;	// L4174
    int32_t v3502 = v134;	// L4175
    ap_int<33> v3503 = v3502;	// L4176
    ap_int<33> v3504 = v3501;	// L4177
    ap_int<33> v3505 = v3503 + v3504;	// L4178
    int32_t v3506 = v3505;	// L4179
    v134 = v3506;	// L4180
    int8_t v3507 = a134;	// L4181
    v3486.write(v3507); // v3486[k134] = v3507;	// L4182
    int8_t v3508 = b134;	// L4183
    v3487.write(v3508); // v3487[k134] = v3508;	// L4184
  }
  int32_t v3509 = v134;	// L4186
  v3488[v3489][v3490] = v3509;	// L4187
}

void PE_kernel_gemm_7_8(
  hls::stream< int8_t > &v3510 /* v3510[16] */,
  hls::stream< int8_t > &v3511 /* v3511[16] */,
  hls::stream< int8_t > &v3512 /* v3512[16] */,
  hls::stream< int8_t > &v3513 /* v3513[16] */,
  int32_t v3514[16][16],
  int v3515,
  int v3516
) {	// L4190
  #pragma HLS stream variable=v3510 depth=17
  #pragma HLS stream variable=v3511 depth=17
  #pragma HLS stream variable=v3512 depth=17
  #pragma HLS stream variable=v3513 depth=17
  #pragma HLS array_partition variable=v3514 complete dim=1
  #pragma HLS array_partition variable=v3514 complete dim=2

  int32_t v135;	// L4192
  v135 = 0;	// L4193
  l_reduction_k135: for (int k135 = 0; k135 < 16; k135++) {	// L4194
  #pragma HLS pipeline II=1
    int8_t v3519 = v3510.read(); // v3510[k135];	// L4195
    int8_t a135;	// L4196
    a135 = v3519;	// L4197
    int8_t v3521 = v3511.read(); // v3511[k135];	// L4198
    int8_t b135;	// L4199
    b135 = v3521;	// L4200
    int8_t v3523 = a135;	// L4201
    int8_t v3524 = b135;	// L4202
    int16_t v3525 = v3523;	// L4203
    int16_t v3526 = v3524;	// L4204
    int16_t v3527 = v3525 * v3526;	// L4205
    int32_t v3528 = v135;	// L4206
    ap_int<33> v3529 = v3528;	// L4207
    ap_int<33> v3530 = v3527;	// L4208
    ap_int<33> v3531 = v3529 + v3530;	// L4209
    int32_t v3532 = v3531;	// L4210
    v135 = v3532;	// L4211
    int8_t v3533 = a135;	// L4212
    v3512.write(v3533); // v3512[k135] = v3533;	// L4213
    int8_t v3534 = b135;	// L4214
    v3513.write(v3534); // v3513[k135] = v3534;	// L4215
  }
  int32_t v3535 = v135;	// L4217
  v3514[v3515][v3516] = v3535;	// L4218
}

void PE_kernel_gemm_8_8(
  hls::stream< int8_t > &v3536 /* v3536[16] */,
  hls::stream< int8_t > &v3537 /* v3537[16] */,
  hls::stream< int8_t > &v3538 /* v3538[16] */,
  hls::stream< int8_t > &v3539 /* v3539[16] */,
  int32_t v3540[16][16],
  int v3541,
  int v3542
) {	// L4221
  #pragma HLS stream variable=v3536 depth=17
  #pragma HLS stream variable=v3537 depth=17
  #pragma HLS stream variable=v3538 depth=17
  #pragma HLS stream variable=v3539 depth=17
  #pragma HLS array_partition variable=v3540 complete dim=1
  #pragma HLS array_partition variable=v3540 complete dim=2

  int32_t v136;	// L4223
  v136 = 0;	// L4224
  l_reduction_k136: for (int k136 = 0; k136 < 16; k136++) {	// L4225
  #pragma HLS pipeline II=1
    int8_t v3545 = v3536.read(); // v3536[k136];	// L4226
    int8_t a136;	// L4227
    a136 = v3545;	// L4228
    int8_t v3547 = v3537.read(); // v3537[k136];	// L4229
    int8_t b136;	// L4230
    b136 = v3547;	// L4231
    int8_t v3549 = a136;	// L4232
    int8_t v3550 = b136;	// L4233
    int16_t v3551 = v3549;	// L4234
    int16_t v3552 = v3550;	// L4235
    int16_t v3553 = v3551 * v3552;	// L4236
    int32_t v3554 = v136;	// L4237
    ap_int<33> v3555 = v3554;	// L4238
    ap_int<33> v3556 = v3553;	// L4239
    ap_int<33> v3557 = v3555 + v3556;	// L4240
    int32_t v3558 = v3557;	// L4241
    v136 = v3558;	// L4242
    int8_t v3559 = a136;	// L4243
    v3538.write(v3559); // v3538[k136] = v3559;	// L4244
    int8_t v3560 = b136;	// L4245
    v3539.write(v3560); // v3539[k136] = v3560;	// L4246
  }
  int32_t v3561 = v136;	// L4248
  v3540[v3541][v3542] = v3561;	// L4249
}

void PE_kernel_gemm_9_8(
  hls::stream< int8_t > &v3562 /* v3562[16] */,
  hls::stream< int8_t > &v3563 /* v3563[16] */,
  hls::stream< int8_t > &v3564 /* v3564[16] */,
  hls::stream< int8_t > &v3565 /* v3565[16] */,
  int32_t v3566[16][16],
  int v3567,
  int v3568
) {	// L4252
  #pragma HLS stream variable=v3562 depth=17
  #pragma HLS stream variable=v3563 depth=17
  #pragma HLS stream variable=v3564 depth=17
  #pragma HLS stream variable=v3565 depth=17
  #pragma HLS array_partition variable=v3566 complete dim=1
  #pragma HLS array_partition variable=v3566 complete dim=2

  int32_t v137;	// L4254
  v137 = 0;	// L4255
  l_reduction_k137: for (int k137 = 0; k137 < 16; k137++) {	// L4256
  #pragma HLS pipeline II=1
    int8_t v3571 = v3562.read(); // v3562[k137];	// L4257
    int8_t a137;	// L4258
    a137 = v3571;	// L4259
    int8_t v3573 = v3563.read(); // v3563[k137];	// L4260
    int8_t b137;	// L4261
    b137 = v3573;	// L4262
    int8_t v3575 = a137;	// L4263
    int8_t v3576 = b137;	// L4264
    int16_t v3577 = v3575;	// L4265
    int16_t v3578 = v3576;	// L4266
    int16_t v3579 = v3577 * v3578;	// L4267
    int32_t v3580 = v137;	// L4268
    ap_int<33> v3581 = v3580;	// L4269
    ap_int<33> v3582 = v3579;	// L4270
    ap_int<33> v3583 = v3581 + v3582;	// L4271
    int32_t v3584 = v3583;	// L4272
    v137 = v3584;	// L4273
    int8_t v3585 = a137;	// L4274
    v3564.write(v3585); // v3564[k137] = v3585;	// L4275
    int8_t v3586 = b137;	// L4276
    v3565.write(v3586); // v3565[k137] = v3586;	// L4277
  }
  int32_t v3587 = v137;	// L4279
  v3566[v3567][v3568] = v3587;	// L4280
}

void PE_kernel_gemm_10_8(
  hls::stream< int8_t > &v3588 /* v3588[16] */,
  hls::stream< int8_t > &v3589 /* v3589[16] */,
  hls::stream< int8_t > &v3590 /* v3590[16] */,
  hls::stream< int8_t > &v3591 /* v3591[16] */,
  int32_t v3592[16][16],
  int v3593,
  int v3594
) {	// L4283
  #pragma HLS stream variable=v3588 depth=17
  #pragma HLS stream variable=v3589 depth=17
  #pragma HLS stream variable=v3590 depth=17
  #pragma HLS stream variable=v3591 depth=17
  #pragma HLS array_partition variable=v3592 complete dim=1
  #pragma HLS array_partition variable=v3592 complete dim=2

  int32_t v138;	// L4285
  v138 = 0;	// L4286
  l_reduction_k138: for (int k138 = 0; k138 < 16; k138++) {	// L4287
  #pragma HLS pipeline II=1
    int8_t v3597 = v3588.read(); // v3588[k138];	// L4288
    int8_t a138;	// L4289
    a138 = v3597;	// L4290
    int8_t v3599 = v3589.read(); // v3589[k138];	// L4291
    int8_t b138;	// L4292
    b138 = v3599;	// L4293
    int8_t v3601 = a138;	// L4294
    int8_t v3602 = b138;	// L4295
    int16_t v3603 = v3601;	// L4296
    int16_t v3604 = v3602;	// L4297
    int16_t v3605 = v3603 * v3604;	// L4298
    int32_t v3606 = v138;	// L4299
    ap_int<33> v3607 = v3606;	// L4300
    ap_int<33> v3608 = v3605;	// L4301
    ap_int<33> v3609 = v3607 + v3608;	// L4302
    int32_t v3610 = v3609;	// L4303
    v138 = v3610;	// L4304
    int8_t v3611 = a138;	// L4305
    v3590.write(v3611); // v3590[k138] = v3611;	// L4306
    int8_t v3612 = b138;	// L4307
    v3591.write(v3612); // v3591[k138] = v3612;	// L4308
  }
  int32_t v3613 = v138;	// L4310
  v3592[v3593][v3594] = v3613;	// L4311
}

void PE_kernel_gemm_11_8(
  hls::stream< int8_t > &v3614 /* v3614[16] */,
  hls::stream< int8_t > &v3615 /* v3615[16] */,
  hls::stream< int8_t > &v3616 /* v3616[16] */,
  hls::stream< int8_t > &v3617 /* v3617[16] */,
  int32_t v3618[16][16],
  int v3619,
  int v3620
) {	// L4314
  #pragma HLS stream variable=v3614 depth=17
  #pragma HLS stream variable=v3615 depth=17
  #pragma HLS stream variable=v3616 depth=17
  #pragma HLS stream variable=v3617 depth=17
  #pragma HLS array_partition variable=v3618 complete dim=1
  #pragma HLS array_partition variable=v3618 complete dim=2

  int32_t v139;	// L4316
  v139 = 0;	// L4317
  l_reduction_k139: for (int k139 = 0; k139 < 16; k139++) {	// L4318
  #pragma HLS pipeline II=1
    int8_t v3623 = v3614.read(); // v3614[k139];	// L4319
    int8_t a139;	// L4320
    a139 = v3623;	// L4321
    int8_t v3625 = v3615.read(); // v3615[k139];	// L4322
    int8_t b139;	// L4323
    b139 = v3625;	// L4324
    int8_t v3627 = a139;	// L4325
    int8_t v3628 = b139;	// L4326
    int16_t v3629 = v3627;	// L4327
    int16_t v3630 = v3628;	// L4328
    int16_t v3631 = v3629 * v3630;	// L4329
    int32_t v3632 = v139;	// L4330
    ap_int<33> v3633 = v3632;	// L4331
    ap_int<33> v3634 = v3631;	// L4332
    ap_int<33> v3635 = v3633 + v3634;	// L4333
    int32_t v3636 = v3635;	// L4334
    v139 = v3636;	// L4335
    int8_t v3637 = a139;	// L4336
    v3616.write(v3637); // v3616[k139] = v3637;	// L4337
    int8_t v3638 = b139;	// L4338
    v3617.write(v3638); // v3617[k139] = v3638;	// L4339
  }
  int32_t v3639 = v139;	// L4341
  v3618[v3619][v3620] = v3639;	// L4342
}

void PE_kernel_gemm_12_8(
  hls::stream< int8_t > &v3640 /* v3640[16] */,
  hls::stream< int8_t > &v3641 /* v3641[16] */,
  hls::stream< int8_t > &v3642 /* v3642[16] */,
  hls::stream< int8_t > &v3643 /* v3643[16] */,
  int32_t v3644[16][16],
  int v3645,
  int v3646
) {	// L4345
  #pragma HLS stream variable=v3640 depth=17
  #pragma HLS stream variable=v3641 depth=17
  #pragma HLS stream variable=v3642 depth=17
  #pragma HLS stream variable=v3643 depth=17
  #pragma HLS array_partition variable=v3644 complete dim=1
  #pragma HLS array_partition variable=v3644 complete dim=2

  int32_t v140;	// L4347
  v140 = 0;	// L4348
  l_reduction_k140: for (int k140 = 0; k140 < 16; k140++) {	// L4349
  #pragma HLS pipeline II=1
    int8_t v3649 = v3640.read(); // v3640[k140];	// L4350
    int8_t a140;	// L4351
    a140 = v3649;	// L4352
    int8_t v3651 = v3641.read(); // v3641[k140];	// L4353
    int8_t b140;	// L4354
    b140 = v3651;	// L4355
    int8_t v3653 = a140;	// L4356
    int8_t v3654 = b140;	// L4357
    int16_t v3655 = v3653;	// L4358
    int16_t v3656 = v3654;	// L4359
    int16_t v3657 = v3655 * v3656;	// L4360
    int32_t v3658 = v140;	// L4361
    ap_int<33> v3659 = v3658;	// L4362
    ap_int<33> v3660 = v3657;	// L4363
    ap_int<33> v3661 = v3659 + v3660;	// L4364
    int32_t v3662 = v3661;	// L4365
    v140 = v3662;	// L4366
    int8_t v3663 = a140;	// L4367
    v3642.write(v3663); // v3642[k140] = v3663;	// L4368
    int8_t v3664 = b140;	// L4369
    v3643.write(v3664); // v3643[k140] = v3664;	// L4370
  }
  int32_t v3665 = v140;	// L4372
  v3644[v3645][v3646] = v3665;	// L4373
}

void PE_kernel_gemm_13_8(
  hls::stream< int8_t > &v3666 /* v3666[16] */,
  hls::stream< int8_t > &v3667 /* v3667[16] */,
  hls::stream< int8_t > &v3668 /* v3668[16] */,
  hls::stream< int8_t > &v3669 /* v3669[16] */,
  int32_t v3670[16][16],
  int v3671,
  int v3672
) {	// L4376
  #pragma HLS stream variable=v3666 depth=17
  #pragma HLS stream variable=v3667 depth=17
  #pragma HLS stream variable=v3668 depth=17
  #pragma HLS stream variable=v3669 depth=17
  #pragma HLS array_partition variable=v3670 complete dim=1
  #pragma HLS array_partition variable=v3670 complete dim=2

  int32_t v141;	// L4378
  v141 = 0;	// L4379
  l_reduction_k141: for (int k141 = 0; k141 < 16; k141++) {	// L4380
  #pragma HLS pipeline II=1
    int8_t v3675 = v3666.read(); // v3666[k141];	// L4381
    int8_t a141;	// L4382
    a141 = v3675;	// L4383
    int8_t v3677 = v3667.read(); // v3667[k141];	// L4384
    int8_t b141;	// L4385
    b141 = v3677;	// L4386
    int8_t v3679 = a141;	// L4387
    int8_t v3680 = b141;	// L4388
    int16_t v3681 = v3679;	// L4389
    int16_t v3682 = v3680;	// L4390
    int16_t v3683 = v3681 * v3682;	// L4391
    int32_t v3684 = v141;	// L4392
    ap_int<33> v3685 = v3684;	// L4393
    ap_int<33> v3686 = v3683;	// L4394
    ap_int<33> v3687 = v3685 + v3686;	// L4395
    int32_t v3688 = v3687;	// L4396
    v141 = v3688;	// L4397
    int8_t v3689 = a141;	// L4398
    v3668.write(v3689); // v3668[k141] = v3689;	// L4399
    int8_t v3690 = b141;	// L4400
    v3669.write(v3690); // v3669[k141] = v3690;	// L4401
  }
  int32_t v3691 = v141;	// L4403
  v3670[v3671][v3672] = v3691;	// L4404
}

void PE_kernel_gemm_14_8(
  hls::stream< int8_t > &v3692 /* v3692[16] */,
  hls::stream< int8_t > &v3693 /* v3693[16] */,
  hls::stream< int8_t > &v3694 /* v3694[16] */,
  hls::stream< int8_t > &v3695 /* v3695[16] */,
  int32_t v3696[16][16],
  int v3697,
  int v3698
) {	// L4407
  #pragma HLS stream variable=v3692 depth=17
  #pragma HLS stream variable=v3693 depth=17
  #pragma HLS stream variable=v3694 depth=17
  #pragma HLS stream variable=v3695 depth=17
  #pragma HLS array_partition variable=v3696 complete dim=1
  #pragma HLS array_partition variable=v3696 complete dim=2

  int32_t v142;	// L4409
  v142 = 0;	// L4410
  l_reduction_k142: for (int k142 = 0; k142 < 16; k142++) {	// L4411
  #pragma HLS pipeline II=1
    int8_t v3701 = v3692.read(); // v3692[k142];	// L4412
    int8_t a142;	// L4413
    a142 = v3701;	// L4414
    int8_t v3703 = v3693.read(); // v3693[k142];	// L4415
    int8_t b142;	// L4416
    b142 = v3703;	// L4417
    int8_t v3705 = a142;	// L4418
    int8_t v3706 = b142;	// L4419
    int16_t v3707 = v3705;	// L4420
    int16_t v3708 = v3706;	// L4421
    int16_t v3709 = v3707 * v3708;	// L4422
    int32_t v3710 = v142;	// L4423
    ap_int<33> v3711 = v3710;	// L4424
    ap_int<33> v3712 = v3709;	// L4425
    ap_int<33> v3713 = v3711 + v3712;	// L4426
    int32_t v3714 = v3713;	// L4427
    v142 = v3714;	// L4428
    int8_t v3715 = a142;	// L4429
    v3694.write(v3715); // v3694[k142] = v3715;	// L4430
    int8_t v3716 = b142;	// L4431
    v3695.write(v3716); // v3695[k142] = v3716;	// L4432
  }
  int32_t v3717 = v142;	// L4434
  v3696[v3697][v3698] = v3717;	// L4435
}

void PE_kernel_gemm_15_8(
  hls::stream< int8_t > &v3718 /* v3718[16] */,
  hls::stream< int8_t > &v3719 /* v3719[16] */,
  hls::stream< int8_t > &v3720 /* v3720[16] */,
  hls::stream< int8_t > &v3721 /* v3721[16] */,
  int32_t v3722[16][16],
  int v3723,
  int v3724
) {	// L4438
  #pragma HLS stream variable=v3718 depth=17
  #pragma HLS stream variable=v3719 depth=17
  #pragma HLS stream variable=v3720 depth=17
  #pragma HLS stream variable=v3721 depth=17
  #pragma HLS array_partition variable=v3722 complete dim=1
  #pragma HLS array_partition variable=v3722 complete dim=2

  int32_t v143;	// L4440
  v143 = 0;	// L4441
  l_reduction_k143: for (int k143 = 0; k143 < 16; k143++) {	// L4442
  #pragma HLS pipeline II=1
    int8_t v3727 = v3718.read(); // v3718[k143];	// L4443
    int8_t a143;	// L4444
    a143 = v3727;	// L4445
    int8_t v3729 = v3719.read(); // v3719[k143];	// L4446
    int8_t b143;	// L4447
    b143 = v3729;	// L4448
    int8_t v3731 = a143;	// L4449
    int8_t v3732 = b143;	// L4450
    int16_t v3733 = v3731;	// L4451
    int16_t v3734 = v3732;	// L4452
    int16_t v3735 = v3733 * v3734;	// L4453
    int32_t v3736 = v143;	// L4454
    ap_int<33> v3737 = v3736;	// L4455
    ap_int<33> v3738 = v3735;	// L4456
    ap_int<33> v3739 = v3737 + v3738;	// L4457
    int32_t v3740 = v3739;	// L4458
    v143 = v3740;	// L4459
    int8_t v3741 = a143;	// L4460
    v3720.write(v3741); // v3720[k143] = v3741;	// L4461
    int8_t v3742 = b143;	// L4462
    v3721.write(v3742); // v3721[k143] = v3742;	// L4463
  }
  int32_t v3743 = v143;	// L4465
  v3722[v3723][v3724] = v3743;	// L4466
}

void PE_kernel_gemm_0_9(
  hls::stream< int8_t > &v3744 /* v3744[16] */,
  hls::stream< int8_t > &v3745 /* v3745[16] */,
  hls::stream< int8_t > &v3746 /* v3746[16] */,
  hls::stream< int8_t > &v3747 /* v3747[16] */,
  int32_t v3748[16][16],
  int v3749,
  int v3750
) {	// L4469
  #pragma HLS stream variable=v3744 depth=17
  #pragma HLS stream variable=v3745 depth=17
  #pragma HLS stream variable=v3746 depth=17
  #pragma HLS stream variable=v3747 depth=17
  #pragma HLS array_partition variable=v3748 complete dim=1
  #pragma HLS array_partition variable=v3748 complete dim=2

  int32_t v144;	// L4471
  v144 = 0;	// L4472
  l_reduction_k144: for (int k144 = 0; k144 < 16; k144++) {	// L4473
  #pragma HLS pipeline II=1
    int8_t v3753 = v3744.read(); // v3744[k144];	// L4474
    int8_t a144;	// L4475
    a144 = v3753;	// L4476
    int8_t v3755 = v3745.read(); // v3745[k144];	// L4477
    int8_t b144;	// L4478
    b144 = v3755;	// L4479
    int8_t v3757 = a144;	// L4480
    int8_t v3758 = b144;	// L4481
    int16_t v3759 = v3757;	// L4482
    int16_t v3760 = v3758;	// L4483
    int16_t v3761 = v3759 * v3760;	// L4484
    int32_t v3762 = v144;	// L4485
    ap_int<33> v3763 = v3762;	// L4486
    ap_int<33> v3764 = v3761;	// L4487
    ap_int<33> v3765 = v3763 + v3764;	// L4488
    int32_t v3766 = v3765;	// L4489
    v144 = v3766;	// L4490
    int8_t v3767 = a144;	// L4491
    v3746.write(v3767); // v3746[k144] = v3767;	// L4492
    int8_t v3768 = b144;	// L4493
    v3747.write(v3768); // v3747[k144] = v3768;	// L4494
  }
  int32_t v3769 = v144;	// L4496
  v3748[v3749][v3750] = v3769;	// L4497
}

void PE_kernel_gemm_1_9(
  hls::stream< int8_t > &v3770 /* v3770[16] */,
  hls::stream< int8_t > &v3771 /* v3771[16] */,
  hls::stream< int8_t > &v3772 /* v3772[16] */,
  hls::stream< int8_t > &v3773 /* v3773[16] */,
  int32_t v3774[16][16],
  int v3775,
  int v3776
) {	// L4500
  #pragma HLS stream variable=v3770 depth=17
  #pragma HLS stream variable=v3771 depth=17
  #pragma HLS stream variable=v3772 depth=17
  #pragma HLS stream variable=v3773 depth=17
  #pragma HLS array_partition variable=v3774 complete dim=1
  #pragma HLS array_partition variable=v3774 complete dim=2

  int32_t v145;	// L4502
  v145 = 0;	// L4503
  l_reduction_k145: for (int k145 = 0; k145 < 16; k145++) {	// L4504
  #pragma HLS pipeline II=1
    int8_t v3779 = v3770.read(); // v3770[k145];	// L4505
    int8_t a145;	// L4506
    a145 = v3779;	// L4507
    int8_t v3781 = v3771.read(); // v3771[k145];	// L4508
    int8_t b145;	// L4509
    b145 = v3781;	// L4510
    int8_t v3783 = a145;	// L4511
    int8_t v3784 = b145;	// L4512
    int16_t v3785 = v3783;	// L4513
    int16_t v3786 = v3784;	// L4514
    int16_t v3787 = v3785 * v3786;	// L4515
    int32_t v3788 = v145;	// L4516
    ap_int<33> v3789 = v3788;	// L4517
    ap_int<33> v3790 = v3787;	// L4518
    ap_int<33> v3791 = v3789 + v3790;	// L4519
    int32_t v3792 = v3791;	// L4520
    v145 = v3792;	// L4521
    int8_t v3793 = a145;	// L4522
    v3772.write(v3793); // v3772[k145] = v3793;	// L4523
    int8_t v3794 = b145;	// L4524
    v3773.write(v3794); // v3773[k145] = v3794;	// L4525
  }
  int32_t v3795 = v145;	// L4527
  v3774[v3775][v3776] = v3795;	// L4528
}

void PE_kernel_gemm_2_9(
  hls::stream< int8_t > &v3796 /* v3796[16] */,
  hls::stream< int8_t > &v3797 /* v3797[16] */,
  hls::stream< int8_t > &v3798 /* v3798[16] */,
  hls::stream< int8_t > &v3799 /* v3799[16] */,
  int32_t v3800[16][16],
  int v3801,
  int v3802
) {	// L4531
  #pragma HLS stream variable=v3796 depth=17
  #pragma HLS stream variable=v3797 depth=17
  #pragma HLS stream variable=v3798 depth=17
  #pragma HLS stream variable=v3799 depth=17
  #pragma HLS array_partition variable=v3800 complete dim=1
  #pragma HLS array_partition variable=v3800 complete dim=2

  int32_t v146;	// L4533
  v146 = 0;	// L4534
  l_reduction_k146: for (int k146 = 0; k146 < 16; k146++) {	// L4535
  #pragma HLS pipeline II=1
    int8_t v3805 = v3796.read(); // v3796[k146];	// L4536
    int8_t a146;	// L4537
    a146 = v3805;	// L4538
    int8_t v3807 = v3797.read(); // v3797[k146];	// L4539
    int8_t b146;	// L4540
    b146 = v3807;	// L4541
    int8_t v3809 = a146;	// L4542
    int8_t v3810 = b146;	// L4543
    int16_t v3811 = v3809;	// L4544
    int16_t v3812 = v3810;	// L4545
    int16_t v3813 = v3811 * v3812;	// L4546
    int32_t v3814 = v146;	// L4547
    ap_int<33> v3815 = v3814;	// L4548
    ap_int<33> v3816 = v3813;	// L4549
    ap_int<33> v3817 = v3815 + v3816;	// L4550
    int32_t v3818 = v3817;	// L4551
    v146 = v3818;	// L4552
    int8_t v3819 = a146;	// L4553
    v3798.write(v3819); // v3798[k146] = v3819;	// L4554
    int8_t v3820 = b146;	// L4555
    v3799.write(v3820); // v3799[k146] = v3820;	// L4556
  }
  int32_t v3821 = v146;	// L4558
  v3800[v3801][v3802] = v3821;	// L4559
}

void PE_kernel_gemm_3_9(
  hls::stream< int8_t > &v3822 /* v3822[16] */,
  hls::stream< int8_t > &v3823 /* v3823[16] */,
  hls::stream< int8_t > &v3824 /* v3824[16] */,
  hls::stream< int8_t > &v3825 /* v3825[16] */,
  int32_t v3826[16][16],
  int v3827,
  int v3828
) {	// L4562
  #pragma HLS stream variable=v3822 depth=17
  #pragma HLS stream variable=v3823 depth=17
  #pragma HLS stream variable=v3824 depth=17
  #pragma HLS stream variable=v3825 depth=17
  #pragma HLS array_partition variable=v3826 complete dim=1
  #pragma HLS array_partition variable=v3826 complete dim=2

  int32_t v147;	// L4564
  v147 = 0;	// L4565
  l_reduction_k147: for (int k147 = 0; k147 < 16; k147++) {	// L4566
  #pragma HLS pipeline II=1
    int8_t v3831 = v3822.read(); // v3822[k147];	// L4567
    int8_t a147;	// L4568
    a147 = v3831;	// L4569
    int8_t v3833 = v3823.read(); // v3823[k147];	// L4570
    int8_t b147;	// L4571
    b147 = v3833;	// L4572
    int8_t v3835 = a147;	// L4573
    int8_t v3836 = b147;	// L4574
    int16_t v3837 = v3835;	// L4575
    int16_t v3838 = v3836;	// L4576
    int16_t v3839 = v3837 * v3838;	// L4577
    int32_t v3840 = v147;	// L4578
    ap_int<33> v3841 = v3840;	// L4579
    ap_int<33> v3842 = v3839;	// L4580
    ap_int<33> v3843 = v3841 + v3842;	// L4581
    int32_t v3844 = v3843;	// L4582
    v147 = v3844;	// L4583
    int8_t v3845 = a147;	// L4584
    v3824.write(v3845); // v3824[k147] = v3845;	// L4585
    int8_t v3846 = b147;	// L4586
    v3825.write(v3846); // v3825[k147] = v3846;	// L4587
  }
  int32_t v3847 = v147;	// L4589
  v3826[v3827][v3828] = v3847;	// L4590
}

void PE_kernel_gemm_4_9(
  hls::stream< int8_t > &v3848 /* v3848[16] */,
  hls::stream< int8_t > &v3849 /* v3849[16] */,
  hls::stream< int8_t > &v3850 /* v3850[16] */,
  hls::stream< int8_t > &v3851 /* v3851[16] */,
  int32_t v3852[16][16],
  int v3853,
  int v3854
) {	// L4593
  #pragma HLS stream variable=v3848 depth=17
  #pragma HLS stream variable=v3849 depth=17
  #pragma HLS stream variable=v3850 depth=17
  #pragma HLS stream variable=v3851 depth=17
  #pragma HLS array_partition variable=v3852 complete dim=1
  #pragma HLS array_partition variable=v3852 complete dim=2

  int32_t v148;	// L4595
  v148 = 0;	// L4596
  l_reduction_k148: for (int k148 = 0; k148 < 16; k148++) {	// L4597
  #pragma HLS pipeline II=1
    int8_t v3857 = v3848.read(); // v3848[k148];	// L4598
    int8_t a148;	// L4599
    a148 = v3857;	// L4600
    int8_t v3859 = v3849.read(); // v3849[k148];	// L4601
    int8_t b148;	// L4602
    b148 = v3859;	// L4603
    int8_t v3861 = a148;	// L4604
    int8_t v3862 = b148;	// L4605
    int16_t v3863 = v3861;	// L4606
    int16_t v3864 = v3862;	// L4607
    int16_t v3865 = v3863 * v3864;	// L4608
    int32_t v3866 = v148;	// L4609
    ap_int<33> v3867 = v3866;	// L4610
    ap_int<33> v3868 = v3865;	// L4611
    ap_int<33> v3869 = v3867 + v3868;	// L4612
    int32_t v3870 = v3869;	// L4613
    v148 = v3870;	// L4614
    int8_t v3871 = a148;	// L4615
    v3850.write(v3871); // v3850[k148] = v3871;	// L4616
    int8_t v3872 = b148;	// L4617
    v3851.write(v3872); // v3851[k148] = v3872;	// L4618
  }
  int32_t v3873 = v148;	// L4620
  v3852[v3853][v3854] = v3873;	// L4621
}

void PE_kernel_gemm_5_9(
  hls::stream< int8_t > &v3874 /* v3874[16] */,
  hls::stream< int8_t > &v3875 /* v3875[16] */,
  hls::stream< int8_t > &v3876 /* v3876[16] */,
  hls::stream< int8_t > &v3877 /* v3877[16] */,
  int32_t v3878[16][16],
  int v3879,
  int v3880
) {	// L4624
  #pragma HLS stream variable=v3874 depth=17
  #pragma HLS stream variable=v3875 depth=17
  #pragma HLS stream variable=v3876 depth=17
  #pragma HLS stream variable=v3877 depth=17
  #pragma HLS array_partition variable=v3878 complete dim=1
  #pragma HLS array_partition variable=v3878 complete dim=2

  int32_t v149;	// L4626
  v149 = 0;	// L4627
  l_reduction_k149: for (int k149 = 0; k149 < 16; k149++) {	// L4628
  #pragma HLS pipeline II=1
    int8_t v3883 = v3874.read(); // v3874[k149];	// L4629
    int8_t a149;	// L4630
    a149 = v3883;	// L4631
    int8_t v3885 = v3875.read(); // v3875[k149];	// L4632
    int8_t b149;	// L4633
    b149 = v3885;	// L4634
    int8_t v3887 = a149;	// L4635
    int8_t v3888 = b149;	// L4636
    int16_t v3889 = v3887;	// L4637
    int16_t v3890 = v3888;	// L4638
    int16_t v3891 = v3889 * v3890;	// L4639
    int32_t v3892 = v149;	// L4640
    ap_int<33> v3893 = v3892;	// L4641
    ap_int<33> v3894 = v3891;	// L4642
    ap_int<33> v3895 = v3893 + v3894;	// L4643
    int32_t v3896 = v3895;	// L4644
    v149 = v3896;	// L4645
    int8_t v3897 = a149;	// L4646
    v3876.write(v3897); // v3876[k149] = v3897;	// L4647
    int8_t v3898 = b149;	// L4648
    v3877.write(v3898); // v3877[k149] = v3898;	// L4649
  }
  int32_t v3899 = v149;	// L4651
  v3878[v3879][v3880] = v3899;	// L4652
}

void PE_kernel_gemm_6_9(
  hls::stream< int8_t > &v3900 /* v3900[16] */,
  hls::stream< int8_t > &v3901 /* v3901[16] */,
  hls::stream< int8_t > &v3902 /* v3902[16] */,
  hls::stream< int8_t > &v3903 /* v3903[16] */,
  int32_t v3904[16][16],
  int v3905,
  int v3906
) {	// L4655
  #pragma HLS stream variable=v3900 depth=17
  #pragma HLS stream variable=v3901 depth=17
  #pragma HLS stream variable=v3902 depth=17
  #pragma HLS stream variable=v3903 depth=17
  #pragma HLS array_partition variable=v3904 complete dim=1
  #pragma HLS array_partition variable=v3904 complete dim=2

  int32_t v150;	// L4657
  v150 = 0;	// L4658
  l_reduction_k150: for (int k150 = 0; k150 < 16; k150++) {	// L4659
  #pragma HLS pipeline II=1
    int8_t v3909 = v3900.read(); // v3900[k150];	// L4660
    int8_t a150;	// L4661
    a150 = v3909;	// L4662
    int8_t v3911 = v3901.read(); // v3901[k150];	// L4663
    int8_t b150;	// L4664
    b150 = v3911;	// L4665
    int8_t v3913 = a150;	// L4666
    int8_t v3914 = b150;	// L4667
    int16_t v3915 = v3913;	// L4668
    int16_t v3916 = v3914;	// L4669
    int16_t v3917 = v3915 * v3916;	// L4670
    int32_t v3918 = v150;	// L4671
    ap_int<33> v3919 = v3918;	// L4672
    ap_int<33> v3920 = v3917;	// L4673
    ap_int<33> v3921 = v3919 + v3920;	// L4674
    int32_t v3922 = v3921;	// L4675
    v150 = v3922;	// L4676
    int8_t v3923 = a150;	// L4677
    v3902.write(v3923); // v3902[k150] = v3923;	// L4678
    int8_t v3924 = b150;	// L4679
    v3903.write(v3924); // v3903[k150] = v3924;	// L4680
  }
  int32_t v3925 = v150;	// L4682
  v3904[v3905][v3906] = v3925;	// L4683
}

void PE_kernel_gemm_7_9(
  hls::stream< int8_t > &v3926 /* v3926[16] */,
  hls::stream< int8_t > &v3927 /* v3927[16] */,
  hls::stream< int8_t > &v3928 /* v3928[16] */,
  hls::stream< int8_t > &v3929 /* v3929[16] */,
  int32_t v3930[16][16],
  int v3931,
  int v3932
) {	// L4686
  #pragma HLS stream variable=v3926 depth=17
  #pragma HLS stream variable=v3927 depth=17
  #pragma HLS stream variable=v3928 depth=17
  #pragma HLS stream variable=v3929 depth=17
  #pragma HLS array_partition variable=v3930 complete dim=1
  #pragma HLS array_partition variable=v3930 complete dim=2

  int32_t v151;	// L4688
  v151 = 0;	// L4689
  l_reduction_k151: for (int k151 = 0; k151 < 16; k151++) {	// L4690
  #pragma HLS pipeline II=1
    int8_t v3935 = v3926.read(); // v3926[k151];	// L4691
    int8_t a151;	// L4692
    a151 = v3935;	// L4693
    int8_t v3937 = v3927.read(); // v3927[k151];	// L4694
    int8_t b151;	// L4695
    b151 = v3937;	// L4696
    int8_t v3939 = a151;	// L4697
    int8_t v3940 = b151;	// L4698
    int16_t v3941 = v3939;	// L4699
    int16_t v3942 = v3940;	// L4700
    int16_t v3943 = v3941 * v3942;	// L4701
    int32_t v3944 = v151;	// L4702
    ap_int<33> v3945 = v3944;	// L4703
    ap_int<33> v3946 = v3943;	// L4704
    ap_int<33> v3947 = v3945 + v3946;	// L4705
    int32_t v3948 = v3947;	// L4706
    v151 = v3948;	// L4707
    int8_t v3949 = a151;	// L4708
    v3928.write(v3949); // v3928[k151] = v3949;	// L4709
    int8_t v3950 = b151;	// L4710
    v3929.write(v3950); // v3929[k151] = v3950;	// L4711
  }
  int32_t v3951 = v151;	// L4713
  v3930[v3931][v3932] = v3951;	// L4714
}

void PE_kernel_gemm_8_9(
  hls::stream< int8_t > &v3952 /* v3952[16] */,
  hls::stream< int8_t > &v3953 /* v3953[16] */,
  hls::stream< int8_t > &v3954 /* v3954[16] */,
  hls::stream< int8_t > &v3955 /* v3955[16] */,
  int32_t v3956[16][16],
  int v3957,
  int v3958
) {	// L4717
  #pragma HLS stream variable=v3952 depth=17
  #pragma HLS stream variable=v3953 depth=17
  #pragma HLS stream variable=v3954 depth=17
  #pragma HLS stream variable=v3955 depth=17
  #pragma HLS array_partition variable=v3956 complete dim=1
  #pragma HLS array_partition variable=v3956 complete dim=2

  int32_t v152;	// L4719
  v152 = 0;	// L4720
  l_reduction_k152: for (int k152 = 0; k152 < 16; k152++) {	// L4721
  #pragma HLS pipeline II=1
    int8_t v3961 = v3952.read(); // v3952[k152];	// L4722
    int8_t a152;	// L4723
    a152 = v3961;	// L4724
    int8_t v3963 = v3953.read(); // v3953[k152];	// L4725
    int8_t b152;	// L4726
    b152 = v3963;	// L4727
    int8_t v3965 = a152;	// L4728
    int8_t v3966 = b152;	// L4729
    int16_t v3967 = v3965;	// L4730
    int16_t v3968 = v3966;	// L4731
    int16_t v3969 = v3967 * v3968;	// L4732
    int32_t v3970 = v152;	// L4733
    ap_int<33> v3971 = v3970;	// L4734
    ap_int<33> v3972 = v3969;	// L4735
    ap_int<33> v3973 = v3971 + v3972;	// L4736
    int32_t v3974 = v3973;	// L4737
    v152 = v3974;	// L4738
    int8_t v3975 = a152;	// L4739
    v3954.write(v3975); // v3954[k152] = v3975;	// L4740
    int8_t v3976 = b152;	// L4741
    v3955.write(v3976); // v3955[k152] = v3976;	// L4742
  }
  int32_t v3977 = v152;	// L4744
  v3956[v3957][v3958] = v3977;	// L4745
}

void PE_kernel_gemm_9_9(
  hls::stream< int8_t > &v3978 /* v3978[16] */,
  hls::stream< int8_t > &v3979 /* v3979[16] */,
  hls::stream< int8_t > &v3980 /* v3980[16] */,
  hls::stream< int8_t > &v3981 /* v3981[16] */,
  int32_t v3982[16][16],
  int v3983,
  int v3984
) {	// L4748
  #pragma HLS stream variable=v3978 depth=17
  #pragma HLS stream variable=v3979 depth=17
  #pragma HLS stream variable=v3980 depth=17
  #pragma HLS stream variable=v3981 depth=17
  #pragma HLS array_partition variable=v3982 complete dim=1
  #pragma HLS array_partition variable=v3982 complete dim=2

  int32_t v153;	// L4750
  v153 = 0;	// L4751
  l_reduction_k153: for (int k153 = 0; k153 < 16; k153++) {	// L4752
  #pragma HLS pipeline II=1
    int8_t v3987 = v3978.read(); // v3978[k153];	// L4753
    int8_t a153;	// L4754
    a153 = v3987;	// L4755
    int8_t v3989 = v3979.read(); // v3979[k153];	// L4756
    int8_t b153;	// L4757
    b153 = v3989;	// L4758
    int8_t v3991 = a153;	// L4759
    int8_t v3992 = b153;	// L4760
    int16_t v3993 = v3991;	// L4761
    int16_t v3994 = v3992;	// L4762
    int16_t v3995 = v3993 * v3994;	// L4763
    int32_t v3996 = v153;	// L4764
    ap_int<33> v3997 = v3996;	// L4765
    ap_int<33> v3998 = v3995;	// L4766
    ap_int<33> v3999 = v3997 + v3998;	// L4767
    int32_t v4000 = v3999;	// L4768
    v153 = v4000;	// L4769
    int8_t v4001 = a153;	// L4770
    v3980.write(v4001); // v3980[k153] = v4001;	// L4771
    int8_t v4002 = b153;	// L4772
    v3981.write(v4002); // v3981[k153] = v4002;	// L4773
  }
  int32_t v4003 = v153;	// L4775
  v3982[v3983][v3984] = v4003;	// L4776
}

void PE_kernel_gemm_10_9(
  hls::stream< int8_t > &v4004 /* v4004[16] */,
  hls::stream< int8_t > &v4005 /* v4005[16] */,
  hls::stream< int8_t > &v4006 /* v4006[16] */,
  hls::stream< int8_t > &v4007 /* v4007[16] */,
  int32_t v4008[16][16],
  int v4009,
  int v4010
) {	// L4779
  #pragma HLS stream variable=v4004 depth=17
  #pragma HLS stream variable=v4005 depth=17
  #pragma HLS stream variable=v4006 depth=17
  #pragma HLS stream variable=v4007 depth=17
  #pragma HLS array_partition variable=v4008 complete dim=1
  #pragma HLS array_partition variable=v4008 complete dim=2

  int32_t v154;	// L4781
  v154 = 0;	// L4782
  l_reduction_k154: for (int k154 = 0; k154 < 16; k154++) {	// L4783
  #pragma HLS pipeline II=1
    int8_t v4013 = v4004.read(); // v4004[k154];	// L4784
    int8_t a154;	// L4785
    a154 = v4013;	// L4786
    int8_t v4015 = v4005.read(); // v4005[k154];	// L4787
    int8_t b154;	// L4788
    b154 = v4015;	// L4789
    int8_t v4017 = a154;	// L4790
    int8_t v4018 = b154;	// L4791
    int16_t v4019 = v4017;	// L4792
    int16_t v4020 = v4018;	// L4793
    int16_t v4021 = v4019 * v4020;	// L4794
    int32_t v4022 = v154;	// L4795
    ap_int<33> v4023 = v4022;	// L4796
    ap_int<33> v4024 = v4021;	// L4797
    ap_int<33> v4025 = v4023 + v4024;	// L4798
    int32_t v4026 = v4025;	// L4799
    v154 = v4026;	// L4800
    int8_t v4027 = a154;	// L4801
    v4006.write(v4027); // v4006[k154] = v4027;	// L4802
    int8_t v4028 = b154;	// L4803
    v4007.write(v4028); // v4007[k154] = v4028;	// L4804
  }
  int32_t v4029 = v154;	// L4806
  v4008[v4009][v4010] = v4029;	// L4807
}

void PE_kernel_gemm_11_9(
  hls::stream< int8_t > &v4030 /* v4030[16] */,
  hls::stream< int8_t > &v4031 /* v4031[16] */,
  hls::stream< int8_t > &v4032 /* v4032[16] */,
  hls::stream< int8_t > &v4033 /* v4033[16] */,
  int32_t v4034[16][16],
  int v4035,
  int v4036
) {	// L4810
  #pragma HLS stream variable=v4030 depth=17
  #pragma HLS stream variable=v4031 depth=17
  #pragma HLS stream variable=v4032 depth=17
  #pragma HLS stream variable=v4033 depth=17
  #pragma HLS array_partition variable=v4034 complete dim=1
  #pragma HLS array_partition variable=v4034 complete dim=2

  int32_t v155;	// L4812
  v155 = 0;	// L4813
  l_reduction_k155: for (int k155 = 0; k155 < 16; k155++) {	// L4814
  #pragma HLS pipeline II=1
    int8_t v4039 = v4030.read(); // v4030[k155];	// L4815
    int8_t a155;	// L4816
    a155 = v4039;	// L4817
    int8_t v4041 = v4031.read(); // v4031[k155];	// L4818
    int8_t b155;	// L4819
    b155 = v4041;	// L4820
    int8_t v4043 = a155;	// L4821
    int8_t v4044 = b155;	// L4822
    int16_t v4045 = v4043;	// L4823
    int16_t v4046 = v4044;	// L4824
    int16_t v4047 = v4045 * v4046;	// L4825
    int32_t v4048 = v155;	// L4826
    ap_int<33> v4049 = v4048;	// L4827
    ap_int<33> v4050 = v4047;	// L4828
    ap_int<33> v4051 = v4049 + v4050;	// L4829
    int32_t v4052 = v4051;	// L4830
    v155 = v4052;	// L4831
    int8_t v4053 = a155;	// L4832
    v4032.write(v4053); // v4032[k155] = v4053;	// L4833
    int8_t v4054 = b155;	// L4834
    v4033.write(v4054); // v4033[k155] = v4054;	// L4835
  }
  int32_t v4055 = v155;	// L4837
  v4034[v4035][v4036] = v4055;	// L4838
}

void PE_kernel_gemm_12_9(
  hls::stream< int8_t > &v4056 /* v4056[16] */,
  hls::stream< int8_t > &v4057 /* v4057[16] */,
  hls::stream< int8_t > &v4058 /* v4058[16] */,
  hls::stream< int8_t > &v4059 /* v4059[16] */,
  int32_t v4060[16][16],
  int v4061,
  int v4062
) {	// L4841
  #pragma HLS stream variable=v4056 depth=17
  #pragma HLS stream variable=v4057 depth=17
  #pragma HLS stream variable=v4058 depth=17
  #pragma HLS stream variable=v4059 depth=17
  #pragma HLS array_partition variable=v4060 complete dim=1
  #pragma HLS array_partition variable=v4060 complete dim=2

  int32_t v156;	// L4843
  v156 = 0;	// L4844
  l_reduction_k156: for (int k156 = 0; k156 < 16; k156++) {	// L4845
  #pragma HLS pipeline II=1
    int8_t v4065 = v4056.read(); // v4056[k156];	// L4846
    int8_t a156;	// L4847
    a156 = v4065;	// L4848
    int8_t v4067 = v4057.read(); // v4057[k156];	// L4849
    int8_t b156;	// L4850
    b156 = v4067;	// L4851
    int8_t v4069 = a156;	// L4852
    int8_t v4070 = b156;	// L4853
    int16_t v4071 = v4069;	// L4854
    int16_t v4072 = v4070;	// L4855
    int16_t v4073 = v4071 * v4072;	// L4856
    int32_t v4074 = v156;	// L4857
    ap_int<33> v4075 = v4074;	// L4858
    ap_int<33> v4076 = v4073;	// L4859
    ap_int<33> v4077 = v4075 + v4076;	// L4860
    int32_t v4078 = v4077;	// L4861
    v156 = v4078;	// L4862
    int8_t v4079 = a156;	// L4863
    v4058.write(v4079); // v4058[k156] = v4079;	// L4864
    int8_t v4080 = b156;	// L4865
    v4059.write(v4080); // v4059[k156] = v4080;	// L4866
  }
  int32_t v4081 = v156;	// L4868
  v4060[v4061][v4062] = v4081;	// L4869
}

void PE_kernel_gemm_13_9(
  hls::stream< int8_t > &v4082 /* v4082[16] */,
  hls::stream< int8_t > &v4083 /* v4083[16] */,
  hls::stream< int8_t > &v4084 /* v4084[16] */,
  hls::stream< int8_t > &v4085 /* v4085[16] */,
  int32_t v4086[16][16],
  int v4087,
  int v4088
) {	// L4872
  #pragma HLS stream variable=v4082 depth=17
  #pragma HLS stream variable=v4083 depth=17
  #pragma HLS stream variable=v4084 depth=17
  #pragma HLS stream variable=v4085 depth=17
  #pragma HLS array_partition variable=v4086 complete dim=1
  #pragma HLS array_partition variable=v4086 complete dim=2

  int32_t v157;	// L4874
  v157 = 0;	// L4875
  l_reduction_k157: for (int k157 = 0; k157 < 16; k157++) {	// L4876
  #pragma HLS pipeline II=1
    int8_t v4091 = v4082.read(); // v4082[k157];	// L4877
    int8_t a157;	// L4878
    a157 = v4091;	// L4879
    int8_t v4093 = v4083.read(); // v4083[k157];	// L4880
    int8_t b157;	// L4881
    b157 = v4093;	// L4882
    int8_t v4095 = a157;	// L4883
    int8_t v4096 = b157;	// L4884
    int16_t v4097 = v4095;	// L4885
    int16_t v4098 = v4096;	// L4886
    int16_t v4099 = v4097 * v4098;	// L4887
    int32_t v4100 = v157;	// L4888
    ap_int<33> v4101 = v4100;	// L4889
    ap_int<33> v4102 = v4099;	// L4890
    ap_int<33> v4103 = v4101 + v4102;	// L4891
    int32_t v4104 = v4103;	// L4892
    v157 = v4104;	// L4893
    int8_t v4105 = a157;	// L4894
    v4084.write(v4105); // v4084[k157] = v4105;	// L4895
    int8_t v4106 = b157;	// L4896
    v4085.write(v4106); // v4085[k157] = v4106;	// L4897
  }
  int32_t v4107 = v157;	// L4899
  v4086[v4087][v4088] = v4107;	// L4900
}

void PE_kernel_gemm_14_9(
  hls::stream< int8_t > &v4108 /* v4108[16] */,
  hls::stream< int8_t > &v4109 /* v4109[16] */,
  hls::stream< int8_t > &v4110 /* v4110[16] */,
  hls::stream< int8_t > &v4111 /* v4111[16] */,
  int32_t v4112[16][16],
  int v4113,
  int v4114
) {	// L4903
  #pragma HLS stream variable=v4108 depth=17
  #pragma HLS stream variable=v4109 depth=17
  #pragma HLS stream variable=v4110 depth=17
  #pragma HLS stream variable=v4111 depth=17
  #pragma HLS array_partition variable=v4112 complete dim=1
  #pragma HLS array_partition variable=v4112 complete dim=2

  int32_t v158;	// L4905
  v158 = 0;	// L4906
  l_reduction_k158: for (int k158 = 0; k158 < 16; k158++) {	// L4907
  #pragma HLS pipeline II=1
    int8_t v4117 = v4108.read(); // v4108[k158];	// L4908
    int8_t a158;	// L4909
    a158 = v4117;	// L4910
    int8_t v4119 = v4109.read(); // v4109[k158];	// L4911
    int8_t b158;	// L4912
    b158 = v4119;	// L4913
    int8_t v4121 = a158;	// L4914
    int8_t v4122 = b158;	// L4915
    int16_t v4123 = v4121;	// L4916
    int16_t v4124 = v4122;	// L4917
    int16_t v4125 = v4123 * v4124;	// L4918
    int32_t v4126 = v158;	// L4919
    ap_int<33> v4127 = v4126;	// L4920
    ap_int<33> v4128 = v4125;	// L4921
    ap_int<33> v4129 = v4127 + v4128;	// L4922
    int32_t v4130 = v4129;	// L4923
    v158 = v4130;	// L4924
    int8_t v4131 = a158;	// L4925
    v4110.write(v4131); // v4110[k158] = v4131;	// L4926
    int8_t v4132 = b158;	// L4927
    v4111.write(v4132); // v4111[k158] = v4132;	// L4928
  }
  int32_t v4133 = v158;	// L4930
  v4112[v4113][v4114] = v4133;	// L4931
}

void PE_kernel_gemm_15_9(
  hls::stream< int8_t > &v4134 /* v4134[16] */,
  hls::stream< int8_t > &v4135 /* v4135[16] */,
  hls::stream< int8_t > &v4136 /* v4136[16] */,
  hls::stream< int8_t > &v4137 /* v4137[16] */,
  int32_t v4138[16][16],
  int v4139,
  int v4140
) {	// L4934
  #pragma HLS stream variable=v4134 depth=17
  #pragma HLS stream variable=v4135 depth=17
  #pragma HLS stream variable=v4136 depth=17
  #pragma HLS stream variable=v4137 depth=17
  #pragma HLS array_partition variable=v4138 complete dim=1
  #pragma HLS array_partition variable=v4138 complete dim=2

  int32_t v159;	// L4936
  v159 = 0;	// L4937
  l_reduction_k159: for (int k159 = 0; k159 < 16; k159++) {	// L4938
  #pragma HLS pipeline II=1
    int8_t v4143 = v4134.read(); // v4134[k159];	// L4939
    int8_t a159;	// L4940
    a159 = v4143;	// L4941
    int8_t v4145 = v4135.read(); // v4135[k159];	// L4942
    int8_t b159;	// L4943
    b159 = v4145;	// L4944
    int8_t v4147 = a159;	// L4945
    int8_t v4148 = b159;	// L4946
    int16_t v4149 = v4147;	// L4947
    int16_t v4150 = v4148;	// L4948
    int16_t v4151 = v4149 * v4150;	// L4949
    int32_t v4152 = v159;	// L4950
    ap_int<33> v4153 = v4152;	// L4951
    ap_int<33> v4154 = v4151;	// L4952
    ap_int<33> v4155 = v4153 + v4154;	// L4953
    int32_t v4156 = v4155;	// L4954
    v159 = v4156;	// L4955
    int8_t v4157 = a159;	// L4956
    v4136.write(v4157); // v4136[k159] = v4157;	// L4957
    int8_t v4158 = b159;	// L4958
    v4137.write(v4158); // v4137[k159] = v4158;	// L4959
  }
  int32_t v4159 = v159;	// L4961
  v4138[v4139][v4140] = v4159;	// L4962
}

void PE_kernel_gemm_0_10(
  hls::stream< int8_t > &v4160 /* v4160[16] */,
  hls::stream< int8_t > &v4161 /* v4161[16] */,
  hls::stream< int8_t > &v4162 /* v4162[16] */,
  hls::stream< int8_t > &v4163 /* v4163[16] */,
  int32_t v4164[16][16],
  int v4165,
  int v4166
) {	// L4965
  #pragma HLS stream variable=v4160 depth=17
  #pragma HLS stream variable=v4161 depth=17
  #pragma HLS stream variable=v4162 depth=17
  #pragma HLS stream variable=v4163 depth=17
  #pragma HLS array_partition variable=v4164 complete dim=1
  #pragma HLS array_partition variable=v4164 complete dim=2

  int32_t v160;	// L4967
  v160 = 0;	// L4968
  l_reduction_k160: for (int k160 = 0; k160 < 16; k160++) {	// L4969
  #pragma HLS pipeline II=1
    int8_t v4169 = v4160.read(); // v4160[k160];	// L4970
    int8_t a160;	// L4971
    a160 = v4169;	// L4972
    int8_t v4171 = v4161.read(); // v4161[k160];	// L4973
    int8_t b160;	// L4974
    b160 = v4171;	// L4975
    int8_t v4173 = a160;	// L4976
    int8_t v4174 = b160;	// L4977
    int16_t v4175 = v4173;	// L4978
    int16_t v4176 = v4174;	// L4979
    int16_t v4177 = v4175 * v4176;	// L4980
    int32_t v4178 = v160;	// L4981
    ap_int<33> v4179 = v4178;	// L4982
    ap_int<33> v4180 = v4177;	// L4983
    ap_int<33> v4181 = v4179 + v4180;	// L4984
    int32_t v4182 = v4181;	// L4985
    v160 = v4182;	// L4986
    int8_t v4183 = a160;	// L4987
    v4162.write(v4183); // v4162[k160] = v4183;	// L4988
    int8_t v4184 = b160;	// L4989
    v4163.write(v4184); // v4163[k160] = v4184;	// L4990
  }
  int32_t v4185 = v160;	// L4992
  v4164[v4165][v4166] = v4185;	// L4993
}

void PE_kernel_gemm_1_10(
  hls::stream< int8_t > &v4186 /* v4186[16] */,
  hls::stream< int8_t > &v4187 /* v4187[16] */,
  hls::stream< int8_t > &v4188 /* v4188[16] */,
  hls::stream< int8_t > &v4189 /* v4189[16] */,
  int32_t v4190[16][16],
  int v4191,
  int v4192
) {	// L4996
  #pragma HLS stream variable=v4186 depth=17
  #pragma HLS stream variable=v4187 depth=17
  #pragma HLS stream variable=v4188 depth=17
  #pragma HLS stream variable=v4189 depth=17
  #pragma HLS array_partition variable=v4190 complete dim=1
  #pragma HLS array_partition variable=v4190 complete dim=2

  int32_t v161;	// L4998
  v161 = 0;	// L4999
  l_reduction_k161: for (int k161 = 0; k161 < 16; k161++) {	// L5000
  #pragma HLS pipeline II=1
    int8_t v4195 = v4186.read(); // v4186[k161];	// L5001
    int8_t a161;	// L5002
    a161 = v4195;	// L5003
    int8_t v4197 = v4187.read(); // v4187[k161];	// L5004
    int8_t b161;	// L5005
    b161 = v4197;	// L5006
    int8_t v4199 = a161;	// L5007
    int8_t v4200 = b161;	// L5008
    int16_t v4201 = v4199;	// L5009
    int16_t v4202 = v4200;	// L5010
    int16_t v4203 = v4201 * v4202;	// L5011
    int32_t v4204 = v161;	// L5012
    ap_int<33> v4205 = v4204;	// L5013
    ap_int<33> v4206 = v4203;	// L5014
    ap_int<33> v4207 = v4205 + v4206;	// L5015
    int32_t v4208 = v4207;	// L5016
    v161 = v4208;	// L5017
    int8_t v4209 = a161;	// L5018
    v4188.write(v4209); // v4188[k161] = v4209;	// L5019
    int8_t v4210 = b161;	// L5020
    v4189.write(v4210); // v4189[k161] = v4210;	// L5021
  }
  int32_t v4211 = v161;	// L5023
  v4190[v4191][v4192] = v4211;	// L5024
}

void PE_kernel_gemm_2_10(
  hls::stream< int8_t > &v4212 /* v4212[16] */,
  hls::stream< int8_t > &v4213 /* v4213[16] */,
  hls::stream< int8_t > &v4214 /* v4214[16] */,
  hls::stream< int8_t > &v4215 /* v4215[16] */,
  int32_t v4216[16][16],
  int v4217,
  int v4218
) {	// L5027
  #pragma HLS stream variable=v4212 depth=17
  #pragma HLS stream variable=v4213 depth=17
  #pragma HLS stream variable=v4214 depth=17
  #pragma HLS stream variable=v4215 depth=17
  #pragma HLS array_partition variable=v4216 complete dim=1
  #pragma HLS array_partition variable=v4216 complete dim=2

  int32_t v162;	// L5029
  v162 = 0;	// L5030
  l_reduction_k162: for (int k162 = 0; k162 < 16; k162++) {	// L5031
  #pragma HLS pipeline II=1
    int8_t v4221 = v4212.read(); // v4212[k162];	// L5032
    int8_t a162;	// L5033
    a162 = v4221;	// L5034
    int8_t v4223 = v4213.read(); // v4213[k162];	// L5035
    int8_t b162;	// L5036
    b162 = v4223;	// L5037
    int8_t v4225 = a162;	// L5038
    int8_t v4226 = b162;	// L5039
    int16_t v4227 = v4225;	// L5040
    int16_t v4228 = v4226;	// L5041
    int16_t v4229 = v4227 * v4228;	// L5042
    int32_t v4230 = v162;	// L5043
    ap_int<33> v4231 = v4230;	// L5044
    ap_int<33> v4232 = v4229;	// L5045
    ap_int<33> v4233 = v4231 + v4232;	// L5046
    int32_t v4234 = v4233;	// L5047
    v162 = v4234;	// L5048
    int8_t v4235 = a162;	// L5049
    v4214.write(v4235); // v4214[k162] = v4235;	// L5050
    int8_t v4236 = b162;	// L5051
    v4215.write(v4236); // v4215[k162] = v4236;	// L5052
  }
  int32_t v4237 = v162;	// L5054
  v4216[v4217][v4218] = v4237;	// L5055
}

void PE_kernel_gemm_3_10(
  hls::stream< int8_t > &v4238 /* v4238[16] */,
  hls::stream< int8_t > &v4239 /* v4239[16] */,
  hls::stream< int8_t > &v4240 /* v4240[16] */,
  hls::stream< int8_t > &v4241 /* v4241[16] */,
  int32_t v4242[16][16],
  int v4243,
  int v4244
) {	// L5058
  #pragma HLS stream variable=v4238 depth=17
  #pragma HLS stream variable=v4239 depth=17
  #pragma HLS stream variable=v4240 depth=17
  #pragma HLS stream variable=v4241 depth=17
  #pragma HLS array_partition variable=v4242 complete dim=1
  #pragma HLS array_partition variable=v4242 complete dim=2

  int32_t v163;	// L5060
  v163 = 0;	// L5061
  l_reduction_k163: for (int k163 = 0; k163 < 16; k163++) {	// L5062
  #pragma HLS pipeline II=1
    int8_t v4247 = v4238.read(); // v4238[k163];	// L5063
    int8_t a163;	// L5064
    a163 = v4247;	// L5065
    int8_t v4249 = v4239.read(); // v4239[k163];	// L5066
    int8_t b163;	// L5067
    b163 = v4249;	// L5068
    int8_t v4251 = a163;	// L5069
    int8_t v4252 = b163;	// L5070
    int16_t v4253 = v4251;	// L5071
    int16_t v4254 = v4252;	// L5072
    int16_t v4255 = v4253 * v4254;	// L5073
    int32_t v4256 = v163;	// L5074
    ap_int<33> v4257 = v4256;	// L5075
    ap_int<33> v4258 = v4255;	// L5076
    ap_int<33> v4259 = v4257 + v4258;	// L5077
    int32_t v4260 = v4259;	// L5078
    v163 = v4260;	// L5079
    int8_t v4261 = a163;	// L5080
    v4240.write(v4261); // v4240[k163] = v4261;	// L5081
    int8_t v4262 = b163;	// L5082
    v4241.write(v4262); // v4241[k163] = v4262;	// L5083
  }
  int32_t v4263 = v163;	// L5085
  v4242[v4243][v4244] = v4263;	// L5086
}

void PE_kernel_gemm_4_10(
  hls::stream< int8_t > &v4264 /* v4264[16] */,
  hls::stream< int8_t > &v4265 /* v4265[16] */,
  hls::stream< int8_t > &v4266 /* v4266[16] */,
  hls::stream< int8_t > &v4267 /* v4267[16] */,
  int32_t v4268[16][16],
  int v4269,
  int v4270
) {	// L5089
  #pragma HLS stream variable=v4264 depth=17
  #pragma HLS stream variable=v4265 depth=17
  #pragma HLS stream variable=v4266 depth=17
  #pragma HLS stream variable=v4267 depth=17
  #pragma HLS array_partition variable=v4268 complete dim=1
  #pragma HLS array_partition variable=v4268 complete dim=2

  int32_t v164;	// L5091
  v164 = 0;	// L5092
  l_reduction_k164: for (int k164 = 0; k164 < 16; k164++) {	// L5093
  #pragma HLS pipeline II=1
    int8_t v4273 = v4264.read(); // v4264[k164];	// L5094
    int8_t a164;	// L5095
    a164 = v4273;	// L5096
    int8_t v4275 = v4265.read(); // v4265[k164];	// L5097
    int8_t b164;	// L5098
    b164 = v4275;	// L5099
    int8_t v4277 = a164;	// L5100
    int8_t v4278 = b164;	// L5101
    int16_t v4279 = v4277;	// L5102
    int16_t v4280 = v4278;	// L5103
    int16_t v4281 = v4279 * v4280;	// L5104
    int32_t v4282 = v164;	// L5105
    ap_int<33> v4283 = v4282;	// L5106
    ap_int<33> v4284 = v4281;	// L5107
    ap_int<33> v4285 = v4283 + v4284;	// L5108
    int32_t v4286 = v4285;	// L5109
    v164 = v4286;	// L5110
    int8_t v4287 = a164;	// L5111
    v4266.write(v4287); // v4266[k164] = v4287;	// L5112
    int8_t v4288 = b164;	// L5113
    v4267.write(v4288); // v4267[k164] = v4288;	// L5114
  }
  int32_t v4289 = v164;	// L5116
  v4268[v4269][v4270] = v4289;	// L5117
}

void PE_kernel_gemm_5_10(
  hls::stream< int8_t > &v4290 /* v4290[16] */,
  hls::stream< int8_t > &v4291 /* v4291[16] */,
  hls::stream< int8_t > &v4292 /* v4292[16] */,
  hls::stream< int8_t > &v4293 /* v4293[16] */,
  int32_t v4294[16][16],
  int v4295,
  int v4296
) {	// L5120
  #pragma HLS stream variable=v4290 depth=17
  #pragma HLS stream variable=v4291 depth=17
  #pragma HLS stream variable=v4292 depth=17
  #pragma HLS stream variable=v4293 depth=17
  #pragma HLS array_partition variable=v4294 complete dim=1
  #pragma HLS array_partition variable=v4294 complete dim=2

  int32_t v165;	// L5122
  v165 = 0;	// L5123
  l_reduction_k165: for (int k165 = 0; k165 < 16; k165++) {	// L5124
  #pragma HLS pipeline II=1
    int8_t v4299 = v4290.read(); // v4290[k165];	// L5125
    int8_t a165;	// L5126
    a165 = v4299;	// L5127
    int8_t v4301 = v4291.read(); // v4291[k165];	// L5128
    int8_t b165;	// L5129
    b165 = v4301;	// L5130
    int8_t v4303 = a165;	// L5131
    int8_t v4304 = b165;	// L5132
    int16_t v4305 = v4303;	// L5133
    int16_t v4306 = v4304;	// L5134
    int16_t v4307 = v4305 * v4306;	// L5135
    int32_t v4308 = v165;	// L5136
    ap_int<33> v4309 = v4308;	// L5137
    ap_int<33> v4310 = v4307;	// L5138
    ap_int<33> v4311 = v4309 + v4310;	// L5139
    int32_t v4312 = v4311;	// L5140
    v165 = v4312;	// L5141
    int8_t v4313 = a165;	// L5142
    v4292.write(v4313); // v4292[k165] = v4313;	// L5143
    int8_t v4314 = b165;	// L5144
    v4293.write(v4314); // v4293[k165] = v4314;	// L5145
  }
  int32_t v4315 = v165;	// L5147
  v4294[v4295][v4296] = v4315;	// L5148
}

void PE_kernel_gemm_6_10(
  hls::stream< int8_t > &v4316 /* v4316[16] */,
  hls::stream< int8_t > &v4317 /* v4317[16] */,
  hls::stream< int8_t > &v4318 /* v4318[16] */,
  hls::stream< int8_t > &v4319 /* v4319[16] */,
  int32_t v4320[16][16],
  int v4321,
  int v4322
) {	// L5151
  #pragma HLS stream variable=v4316 depth=17
  #pragma HLS stream variable=v4317 depth=17
  #pragma HLS stream variable=v4318 depth=17
  #pragma HLS stream variable=v4319 depth=17
  #pragma HLS array_partition variable=v4320 complete dim=1
  #pragma HLS array_partition variable=v4320 complete dim=2

  int32_t v166;	// L5153
  v166 = 0;	// L5154
  l_reduction_k166: for (int k166 = 0; k166 < 16; k166++) {	// L5155
  #pragma HLS pipeline II=1
    int8_t v4325 = v4316.read(); // v4316[k166];	// L5156
    int8_t a166;	// L5157
    a166 = v4325;	// L5158
    int8_t v4327 = v4317.read(); // v4317[k166];	// L5159
    int8_t b166;	// L5160
    b166 = v4327;	// L5161
    int8_t v4329 = a166;	// L5162
    int8_t v4330 = b166;	// L5163
    int16_t v4331 = v4329;	// L5164
    int16_t v4332 = v4330;	// L5165
    int16_t v4333 = v4331 * v4332;	// L5166
    int32_t v4334 = v166;	// L5167
    ap_int<33> v4335 = v4334;	// L5168
    ap_int<33> v4336 = v4333;	// L5169
    ap_int<33> v4337 = v4335 + v4336;	// L5170
    int32_t v4338 = v4337;	// L5171
    v166 = v4338;	// L5172
    int8_t v4339 = a166;	// L5173
    v4318.write(v4339); // v4318[k166] = v4339;	// L5174
    int8_t v4340 = b166;	// L5175
    v4319.write(v4340); // v4319[k166] = v4340;	// L5176
  }
  int32_t v4341 = v166;	// L5178
  v4320[v4321][v4322] = v4341;	// L5179
}

void PE_kernel_gemm_7_10(
  hls::stream< int8_t > &v4342 /* v4342[16] */,
  hls::stream< int8_t > &v4343 /* v4343[16] */,
  hls::stream< int8_t > &v4344 /* v4344[16] */,
  hls::stream< int8_t > &v4345 /* v4345[16] */,
  int32_t v4346[16][16],
  int v4347,
  int v4348
) {	// L5182
  #pragma HLS stream variable=v4342 depth=17
  #pragma HLS stream variable=v4343 depth=17
  #pragma HLS stream variable=v4344 depth=17
  #pragma HLS stream variable=v4345 depth=17
  #pragma HLS array_partition variable=v4346 complete dim=1
  #pragma HLS array_partition variable=v4346 complete dim=2

  int32_t v167;	// L5184
  v167 = 0;	// L5185
  l_reduction_k167: for (int k167 = 0; k167 < 16; k167++) {	// L5186
  #pragma HLS pipeline II=1
    int8_t v4351 = v4342.read(); // v4342[k167];	// L5187
    int8_t a167;	// L5188
    a167 = v4351;	// L5189
    int8_t v4353 = v4343.read(); // v4343[k167];	// L5190
    int8_t b167;	// L5191
    b167 = v4353;	// L5192
    int8_t v4355 = a167;	// L5193
    int8_t v4356 = b167;	// L5194
    int16_t v4357 = v4355;	// L5195
    int16_t v4358 = v4356;	// L5196
    int16_t v4359 = v4357 * v4358;	// L5197
    int32_t v4360 = v167;	// L5198
    ap_int<33> v4361 = v4360;	// L5199
    ap_int<33> v4362 = v4359;	// L5200
    ap_int<33> v4363 = v4361 + v4362;	// L5201
    int32_t v4364 = v4363;	// L5202
    v167 = v4364;	// L5203
    int8_t v4365 = a167;	// L5204
    v4344.write(v4365); // v4344[k167] = v4365;	// L5205
    int8_t v4366 = b167;	// L5206
    v4345.write(v4366); // v4345[k167] = v4366;	// L5207
  }
  int32_t v4367 = v167;	// L5209
  v4346[v4347][v4348] = v4367;	// L5210
}

void PE_kernel_gemm_8_10(
  hls::stream< int8_t > &v4368 /* v4368[16] */,
  hls::stream< int8_t > &v4369 /* v4369[16] */,
  hls::stream< int8_t > &v4370 /* v4370[16] */,
  hls::stream< int8_t > &v4371 /* v4371[16] */,
  int32_t v4372[16][16],
  int v4373,
  int v4374
) {	// L5213
  #pragma HLS stream variable=v4368 depth=17
  #pragma HLS stream variable=v4369 depth=17
  #pragma HLS stream variable=v4370 depth=17
  #pragma HLS stream variable=v4371 depth=17
  #pragma HLS array_partition variable=v4372 complete dim=1
  #pragma HLS array_partition variable=v4372 complete dim=2

  int32_t v168;	// L5215
  v168 = 0;	// L5216
  l_reduction_k168: for (int k168 = 0; k168 < 16; k168++) {	// L5217
  #pragma HLS pipeline II=1
    int8_t v4377 = v4368.read(); // v4368[k168];	// L5218
    int8_t a168;	// L5219
    a168 = v4377;	// L5220
    int8_t v4379 = v4369.read(); // v4369[k168];	// L5221
    int8_t b168;	// L5222
    b168 = v4379;	// L5223
    int8_t v4381 = a168;	// L5224
    int8_t v4382 = b168;	// L5225
    int16_t v4383 = v4381;	// L5226
    int16_t v4384 = v4382;	// L5227
    int16_t v4385 = v4383 * v4384;	// L5228
    int32_t v4386 = v168;	// L5229
    ap_int<33> v4387 = v4386;	// L5230
    ap_int<33> v4388 = v4385;	// L5231
    ap_int<33> v4389 = v4387 + v4388;	// L5232
    int32_t v4390 = v4389;	// L5233
    v168 = v4390;	// L5234
    int8_t v4391 = a168;	// L5235
    v4370.write(v4391); // v4370[k168] = v4391;	// L5236
    int8_t v4392 = b168;	// L5237
    v4371.write(v4392); // v4371[k168] = v4392;	// L5238
  }
  int32_t v4393 = v168;	// L5240
  v4372[v4373][v4374] = v4393;	// L5241
}

void PE_kernel_gemm_9_10(
  hls::stream< int8_t > &v4394 /* v4394[16] */,
  hls::stream< int8_t > &v4395 /* v4395[16] */,
  hls::stream< int8_t > &v4396 /* v4396[16] */,
  hls::stream< int8_t > &v4397 /* v4397[16] */,
  int32_t v4398[16][16],
  int v4399,
  int v4400
) {	// L5244
  #pragma HLS stream variable=v4394 depth=17
  #pragma HLS stream variable=v4395 depth=17
  #pragma HLS stream variable=v4396 depth=17
  #pragma HLS stream variable=v4397 depth=17
  #pragma HLS array_partition variable=v4398 complete dim=1
  #pragma HLS array_partition variable=v4398 complete dim=2

  int32_t v169;	// L5246
  v169 = 0;	// L5247
  l_reduction_k169: for (int k169 = 0; k169 < 16; k169++) {	// L5248
  #pragma HLS pipeline II=1
    int8_t v4403 = v4394.read(); // v4394[k169];	// L5249
    int8_t a169;	// L5250
    a169 = v4403;	// L5251
    int8_t v4405 = v4395.read(); // v4395[k169];	// L5252
    int8_t b169;	// L5253
    b169 = v4405;	// L5254
    int8_t v4407 = a169;	// L5255
    int8_t v4408 = b169;	// L5256
    int16_t v4409 = v4407;	// L5257
    int16_t v4410 = v4408;	// L5258
    int16_t v4411 = v4409 * v4410;	// L5259
    int32_t v4412 = v169;	// L5260
    ap_int<33> v4413 = v4412;	// L5261
    ap_int<33> v4414 = v4411;	// L5262
    ap_int<33> v4415 = v4413 + v4414;	// L5263
    int32_t v4416 = v4415;	// L5264
    v169 = v4416;	// L5265
    int8_t v4417 = a169;	// L5266
    v4396.write(v4417); // v4396[k169] = v4417;	// L5267
    int8_t v4418 = b169;	// L5268
    v4397.write(v4418); // v4397[k169] = v4418;	// L5269
  }
  int32_t v4419 = v169;	// L5271
  v4398[v4399][v4400] = v4419;	// L5272
}

void PE_kernel_gemm_10_10(
  hls::stream< int8_t > &v4420 /* v4420[16] */,
  hls::stream< int8_t > &v4421 /* v4421[16] */,
  hls::stream< int8_t > &v4422 /* v4422[16] */,
  hls::stream< int8_t > &v4423 /* v4423[16] */,
  int32_t v4424[16][16],
  int v4425,
  int v4426
) {	// L5275
  #pragma HLS stream variable=v4420 depth=17
  #pragma HLS stream variable=v4421 depth=17
  #pragma HLS stream variable=v4422 depth=17
  #pragma HLS stream variable=v4423 depth=17
  #pragma HLS array_partition variable=v4424 complete dim=1
  #pragma HLS array_partition variable=v4424 complete dim=2

  int32_t v170;	// L5277
  v170 = 0;	// L5278
  l_reduction_k170: for (int k170 = 0; k170 < 16; k170++) {	// L5279
  #pragma HLS pipeline II=1
    int8_t v4429 = v4420.read(); // v4420[k170];	// L5280
    int8_t a170;	// L5281
    a170 = v4429;	// L5282
    int8_t v4431 = v4421.read(); // v4421[k170];	// L5283
    int8_t b170;	// L5284
    b170 = v4431;	// L5285
    int8_t v4433 = a170;	// L5286
    int8_t v4434 = b170;	// L5287
    int16_t v4435 = v4433;	// L5288
    int16_t v4436 = v4434;	// L5289
    int16_t v4437 = v4435 * v4436;	// L5290
    int32_t v4438 = v170;	// L5291
    ap_int<33> v4439 = v4438;	// L5292
    ap_int<33> v4440 = v4437;	// L5293
    ap_int<33> v4441 = v4439 + v4440;	// L5294
    int32_t v4442 = v4441;	// L5295
    v170 = v4442;	// L5296
    int8_t v4443 = a170;	// L5297
    v4422.write(v4443); // v4422[k170] = v4443;	// L5298
    int8_t v4444 = b170;	// L5299
    v4423.write(v4444); // v4423[k170] = v4444;	// L5300
  }
  int32_t v4445 = v170;	// L5302
  v4424[v4425][v4426] = v4445;	// L5303
}

void PE_kernel_gemm_11_10(
  hls::stream< int8_t > &v4446 /* v4446[16] */,
  hls::stream< int8_t > &v4447 /* v4447[16] */,
  hls::stream< int8_t > &v4448 /* v4448[16] */,
  hls::stream< int8_t > &v4449 /* v4449[16] */,
  int32_t v4450[16][16],
  int v4451,
  int v4452
) {	// L5306
  #pragma HLS stream variable=v4446 depth=17
  #pragma HLS stream variable=v4447 depth=17
  #pragma HLS stream variable=v4448 depth=17
  #pragma HLS stream variable=v4449 depth=17
  #pragma HLS array_partition variable=v4450 complete dim=1
  #pragma HLS array_partition variable=v4450 complete dim=2

  int32_t v171;	// L5308
  v171 = 0;	// L5309
  l_reduction_k171: for (int k171 = 0; k171 < 16; k171++) {	// L5310
  #pragma HLS pipeline II=1
    int8_t v4455 = v4446.read(); // v4446[k171];	// L5311
    int8_t a171;	// L5312
    a171 = v4455;	// L5313
    int8_t v4457 = v4447.read(); // v4447[k171];	// L5314
    int8_t b171;	// L5315
    b171 = v4457;	// L5316
    int8_t v4459 = a171;	// L5317
    int8_t v4460 = b171;	// L5318
    int16_t v4461 = v4459;	// L5319
    int16_t v4462 = v4460;	// L5320
    int16_t v4463 = v4461 * v4462;	// L5321
    int32_t v4464 = v171;	// L5322
    ap_int<33> v4465 = v4464;	// L5323
    ap_int<33> v4466 = v4463;	// L5324
    ap_int<33> v4467 = v4465 + v4466;	// L5325
    int32_t v4468 = v4467;	// L5326
    v171 = v4468;	// L5327
    int8_t v4469 = a171;	// L5328
    v4448.write(v4469); // v4448[k171] = v4469;	// L5329
    int8_t v4470 = b171;	// L5330
    v4449.write(v4470); // v4449[k171] = v4470;	// L5331
  }
  int32_t v4471 = v171;	// L5333
  v4450[v4451][v4452] = v4471;	// L5334
}

void PE_kernel_gemm_12_10(
  hls::stream< int8_t > &v4472 /* v4472[16] */,
  hls::stream< int8_t > &v4473 /* v4473[16] */,
  hls::stream< int8_t > &v4474 /* v4474[16] */,
  hls::stream< int8_t > &v4475 /* v4475[16] */,
  int32_t v4476[16][16],
  int v4477,
  int v4478
) {	// L5337
  #pragma HLS stream variable=v4472 depth=17
  #pragma HLS stream variable=v4473 depth=17
  #pragma HLS stream variable=v4474 depth=17
  #pragma HLS stream variable=v4475 depth=17
  #pragma HLS array_partition variable=v4476 complete dim=1
  #pragma HLS array_partition variable=v4476 complete dim=2

  int32_t v172;	// L5339
  v172 = 0;	// L5340
  l_reduction_k172: for (int k172 = 0; k172 < 16; k172++) {	// L5341
  #pragma HLS pipeline II=1
    int8_t v4481 = v4472.read(); // v4472[k172];	// L5342
    int8_t a172;	// L5343
    a172 = v4481;	// L5344
    int8_t v4483 = v4473.read(); // v4473[k172];	// L5345
    int8_t b172;	// L5346
    b172 = v4483;	// L5347
    int8_t v4485 = a172;	// L5348
    int8_t v4486 = b172;	// L5349
    int16_t v4487 = v4485;	// L5350
    int16_t v4488 = v4486;	// L5351
    int16_t v4489 = v4487 * v4488;	// L5352
    int32_t v4490 = v172;	// L5353
    ap_int<33> v4491 = v4490;	// L5354
    ap_int<33> v4492 = v4489;	// L5355
    ap_int<33> v4493 = v4491 + v4492;	// L5356
    int32_t v4494 = v4493;	// L5357
    v172 = v4494;	// L5358
    int8_t v4495 = a172;	// L5359
    v4474.write(v4495); // v4474[k172] = v4495;	// L5360
    int8_t v4496 = b172;	// L5361
    v4475.write(v4496); // v4475[k172] = v4496;	// L5362
  }
  int32_t v4497 = v172;	// L5364
  v4476[v4477][v4478] = v4497;	// L5365
}

void PE_kernel_gemm_13_10(
  hls::stream< int8_t > &v4498 /* v4498[16] */,
  hls::stream< int8_t > &v4499 /* v4499[16] */,
  hls::stream< int8_t > &v4500 /* v4500[16] */,
  hls::stream< int8_t > &v4501 /* v4501[16] */,
  int32_t v4502[16][16],
  int v4503,
  int v4504
) {	// L5368
  #pragma HLS stream variable=v4498 depth=17
  #pragma HLS stream variable=v4499 depth=17
  #pragma HLS stream variable=v4500 depth=17
  #pragma HLS stream variable=v4501 depth=17
  #pragma HLS array_partition variable=v4502 complete dim=1
  #pragma HLS array_partition variable=v4502 complete dim=2

  int32_t v173;	// L5370
  v173 = 0;	// L5371
  l_reduction_k173: for (int k173 = 0; k173 < 16; k173++) {	// L5372
  #pragma HLS pipeline II=1
    int8_t v4507 = v4498.read(); // v4498[k173];	// L5373
    int8_t a173;	// L5374
    a173 = v4507;	// L5375
    int8_t v4509 = v4499.read(); // v4499[k173];	// L5376
    int8_t b173;	// L5377
    b173 = v4509;	// L5378
    int8_t v4511 = a173;	// L5379
    int8_t v4512 = b173;	// L5380
    int16_t v4513 = v4511;	// L5381
    int16_t v4514 = v4512;	// L5382
    int16_t v4515 = v4513 * v4514;	// L5383
    int32_t v4516 = v173;	// L5384
    ap_int<33> v4517 = v4516;	// L5385
    ap_int<33> v4518 = v4515;	// L5386
    ap_int<33> v4519 = v4517 + v4518;	// L5387
    int32_t v4520 = v4519;	// L5388
    v173 = v4520;	// L5389
    int8_t v4521 = a173;	// L5390
    v4500.write(v4521); // v4500[k173] = v4521;	// L5391
    int8_t v4522 = b173;	// L5392
    v4501.write(v4522); // v4501[k173] = v4522;	// L5393
  }
  int32_t v4523 = v173;	// L5395
  v4502[v4503][v4504] = v4523;	// L5396
}

void PE_kernel_gemm_14_10(
  hls::stream< int8_t > &v4524 /* v4524[16] */,
  hls::stream< int8_t > &v4525 /* v4525[16] */,
  hls::stream< int8_t > &v4526 /* v4526[16] */,
  hls::stream< int8_t > &v4527 /* v4527[16] */,
  int32_t v4528[16][16],
  int v4529,
  int v4530
) {	// L5399
  #pragma HLS stream variable=v4524 depth=17
  #pragma HLS stream variable=v4525 depth=17
  #pragma HLS stream variable=v4526 depth=17
  #pragma HLS stream variable=v4527 depth=17
  #pragma HLS array_partition variable=v4528 complete dim=1
  #pragma HLS array_partition variable=v4528 complete dim=2

  int32_t v174;	// L5401
  v174 = 0;	// L5402
  l_reduction_k174: for (int k174 = 0; k174 < 16; k174++) {	// L5403
  #pragma HLS pipeline II=1
    int8_t v4533 = v4524.read(); // v4524[k174];	// L5404
    int8_t a174;	// L5405
    a174 = v4533;	// L5406
    int8_t v4535 = v4525.read(); // v4525[k174];	// L5407
    int8_t b174;	// L5408
    b174 = v4535;	// L5409
    int8_t v4537 = a174;	// L5410
    int8_t v4538 = b174;	// L5411
    int16_t v4539 = v4537;	// L5412
    int16_t v4540 = v4538;	// L5413
    int16_t v4541 = v4539 * v4540;	// L5414
    int32_t v4542 = v174;	// L5415
    ap_int<33> v4543 = v4542;	// L5416
    ap_int<33> v4544 = v4541;	// L5417
    ap_int<33> v4545 = v4543 + v4544;	// L5418
    int32_t v4546 = v4545;	// L5419
    v174 = v4546;	// L5420
    int8_t v4547 = a174;	// L5421
    v4526.write(v4547); // v4526[k174] = v4547;	// L5422
    int8_t v4548 = b174;	// L5423
    v4527.write(v4548); // v4527[k174] = v4548;	// L5424
  }
  int32_t v4549 = v174;	// L5426
  v4528[v4529][v4530] = v4549;	// L5427
}

void PE_kernel_gemm_15_10(
  hls::stream< int8_t > &v4550 /* v4550[16] */,
  hls::stream< int8_t > &v4551 /* v4551[16] */,
  hls::stream< int8_t > &v4552 /* v4552[16] */,
  hls::stream< int8_t > &v4553 /* v4553[16] */,
  int32_t v4554[16][16],
  int v4555,
  int v4556
) {	// L5430
  #pragma HLS stream variable=v4550 depth=17
  #pragma HLS stream variable=v4551 depth=17
  #pragma HLS stream variable=v4552 depth=17
  #pragma HLS stream variable=v4553 depth=17
  #pragma HLS array_partition variable=v4554 complete dim=1
  #pragma HLS array_partition variable=v4554 complete dim=2

  int32_t v175;	// L5432
  v175 = 0;	// L5433
  l_reduction_k175: for (int k175 = 0; k175 < 16; k175++) {	// L5434
  #pragma HLS pipeline II=1
    int8_t v4559 = v4550.read(); // v4550[k175];	// L5435
    int8_t a175;	// L5436
    a175 = v4559;	// L5437
    int8_t v4561 = v4551.read(); // v4551[k175];	// L5438
    int8_t b175;	// L5439
    b175 = v4561;	// L5440
    int8_t v4563 = a175;	// L5441
    int8_t v4564 = b175;	// L5442
    int16_t v4565 = v4563;	// L5443
    int16_t v4566 = v4564;	// L5444
    int16_t v4567 = v4565 * v4566;	// L5445
    int32_t v4568 = v175;	// L5446
    ap_int<33> v4569 = v4568;	// L5447
    ap_int<33> v4570 = v4567;	// L5448
    ap_int<33> v4571 = v4569 + v4570;	// L5449
    int32_t v4572 = v4571;	// L5450
    v175 = v4572;	// L5451
    int8_t v4573 = a175;	// L5452
    v4552.write(v4573); // v4552[k175] = v4573;	// L5453
    int8_t v4574 = b175;	// L5454
    v4553.write(v4574); // v4553[k175] = v4574;	// L5455
  }
  int32_t v4575 = v175;	// L5457
  v4554[v4555][v4556] = v4575;	// L5458
}

void PE_kernel_gemm_0_11(
  hls::stream< int8_t > &v4576 /* v4576[16] */,
  hls::stream< int8_t > &v4577 /* v4577[16] */,
  hls::stream< int8_t > &v4578 /* v4578[16] */,
  hls::stream< int8_t > &v4579 /* v4579[16] */,
  int32_t v4580[16][16],
  int v4581,
  int v4582
) {	// L5461
  #pragma HLS stream variable=v4576 depth=17
  #pragma HLS stream variable=v4577 depth=17
  #pragma HLS stream variable=v4578 depth=17
  #pragma HLS stream variable=v4579 depth=17
  #pragma HLS array_partition variable=v4580 complete dim=1
  #pragma HLS array_partition variable=v4580 complete dim=2

  int32_t v176;	// L5463
  v176 = 0;	// L5464
  l_reduction_k176: for (int k176 = 0; k176 < 16; k176++) {	// L5465
  #pragma HLS pipeline II=1
    int8_t v4585 = v4576.read(); // v4576[k176];	// L5466
    int8_t a176;	// L5467
    a176 = v4585;	// L5468
    int8_t v4587 = v4577.read(); // v4577[k176];	// L5469
    int8_t b176;	// L5470
    b176 = v4587;	// L5471
    int8_t v4589 = a176;	// L5472
    int8_t v4590 = b176;	// L5473
    int16_t v4591 = v4589;	// L5474
    int16_t v4592 = v4590;	// L5475
    int16_t v4593 = v4591 * v4592;	// L5476
    int32_t v4594 = v176;	// L5477
    ap_int<33> v4595 = v4594;	// L5478
    ap_int<33> v4596 = v4593;	// L5479
    ap_int<33> v4597 = v4595 + v4596;	// L5480
    int32_t v4598 = v4597;	// L5481
    v176 = v4598;	// L5482
    int8_t v4599 = a176;	// L5483
    v4578.write(v4599); // v4578[k176] = v4599;	// L5484
    int8_t v4600 = b176;	// L5485
    v4579.write(v4600); // v4579[k176] = v4600;	// L5486
  }
  int32_t v4601 = v176;	// L5488
  v4580[v4581][v4582] = v4601;	// L5489
}

void PE_kernel_gemm_1_11(
  hls::stream< int8_t > &v4602 /* v4602[16] */,
  hls::stream< int8_t > &v4603 /* v4603[16] */,
  hls::stream< int8_t > &v4604 /* v4604[16] */,
  hls::stream< int8_t > &v4605 /* v4605[16] */,
  int32_t v4606[16][16],
  int v4607,
  int v4608
) {	// L5492
  #pragma HLS stream variable=v4602 depth=17
  #pragma HLS stream variable=v4603 depth=17
  #pragma HLS stream variable=v4604 depth=17
  #pragma HLS stream variable=v4605 depth=17
  #pragma HLS array_partition variable=v4606 complete dim=1
  #pragma HLS array_partition variable=v4606 complete dim=2

  int32_t v177;	// L5494
  v177 = 0;	// L5495
  l_reduction_k177: for (int k177 = 0; k177 < 16; k177++) {	// L5496
  #pragma HLS pipeline II=1
    int8_t v4611 = v4602.read(); // v4602[k177];	// L5497
    int8_t a177;	// L5498
    a177 = v4611;	// L5499
    int8_t v4613 = v4603.read(); // v4603[k177];	// L5500
    int8_t b177;	// L5501
    b177 = v4613;	// L5502
    int8_t v4615 = a177;	// L5503
    int8_t v4616 = b177;	// L5504
    int16_t v4617 = v4615;	// L5505
    int16_t v4618 = v4616;	// L5506
    int16_t v4619 = v4617 * v4618;	// L5507
    int32_t v4620 = v177;	// L5508
    ap_int<33> v4621 = v4620;	// L5509
    ap_int<33> v4622 = v4619;	// L5510
    ap_int<33> v4623 = v4621 + v4622;	// L5511
    int32_t v4624 = v4623;	// L5512
    v177 = v4624;	// L5513
    int8_t v4625 = a177;	// L5514
    v4604.write(v4625); // v4604[k177] = v4625;	// L5515
    int8_t v4626 = b177;	// L5516
    v4605.write(v4626); // v4605[k177] = v4626;	// L5517
  }
  int32_t v4627 = v177;	// L5519
  v4606[v4607][v4608] = v4627;	// L5520
}

void PE_kernel_gemm_2_11(
  hls::stream< int8_t > &v4628 /* v4628[16] */,
  hls::stream< int8_t > &v4629 /* v4629[16] */,
  hls::stream< int8_t > &v4630 /* v4630[16] */,
  hls::stream< int8_t > &v4631 /* v4631[16] */,
  int32_t v4632[16][16],
  int v4633,
  int v4634
) {	// L5523
  #pragma HLS stream variable=v4628 depth=17
  #pragma HLS stream variable=v4629 depth=17
  #pragma HLS stream variable=v4630 depth=17
  #pragma HLS stream variable=v4631 depth=17
  #pragma HLS array_partition variable=v4632 complete dim=1
  #pragma HLS array_partition variable=v4632 complete dim=2

  int32_t v178;	// L5525
  v178 = 0;	// L5526
  l_reduction_k178: for (int k178 = 0; k178 < 16; k178++) {	// L5527
  #pragma HLS pipeline II=1
    int8_t v4637 = v4628.read(); // v4628[k178];	// L5528
    int8_t a178;	// L5529
    a178 = v4637;	// L5530
    int8_t v4639 = v4629.read(); // v4629[k178];	// L5531
    int8_t b178;	// L5532
    b178 = v4639;	// L5533
    int8_t v4641 = a178;	// L5534
    int8_t v4642 = b178;	// L5535
    int16_t v4643 = v4641;	// L5536
    int16_t v4644 = v4642;	// L5537
    int16_t v4645 = v4643 * v4644;	// L5538
    int32_t v4646 = v178;	// L5539
    ap_int<33> v4647 = v4646;	// L5540
    ap_int<33> v4648 = v4645;	// L5541
    ap_int<33> v4649 = v4647 + v4648;	// L5542
    int32_t v4650 = v4649;	// L5543
    v178 = v4650;	// L5544
    int8_t v4651 = a178;	// L5545
    v4630.write(v4651); // v4630[k178] = v4651;	// L5546
    int8_t v4652 = b178;	// L5547
    v4631.write(v4652); // v4631[k178] = v4652;	// L5548
  }
  int32_t v4653 = v178;	// L5550
  v4632[v4633][v4634] = v4653;	// L5551
}

void PE_kernel_gemm_3_11(
  hls::stream< int8_t > &v4654 /* v4654[16] */,
  hls::stream< int8_t > &v4655 /* v4655[16] */,
  hls::stream< int8_t > &v4656 /* v4656[16] */,
  hls::stream< int8_t > &v4657 /* v4657[16] */,
  int32_t v4658[16][16],
  int v4659,
  int v4660
) {	// L5554
  #pragma HLS stream variable=v4654 depth=17
  #pragma HLS stream variable=v4655 depth=17
  #pragma HLS stream variable=v4656 depth=17
  #pragma HLS stream variable=v4657 depth=17
  #pragma HLS array_partition variable=v4658 complete dim=1
  #pragma HLS array_partition variable=v4658 complete dim=2

  int32_t v179;	// L5556
  v179 = 0;	// L5557
  l_reduction_k179: for (int k179 = 0; k179 < 16; k179++) {	// L5558
  #pragma HLS pipeline II=1
    int8_t v4663 = v4654.read(); // v4654[k179];	// L5559
    int8_t a179;	// L5560
    a179 = v4663;	// L5561
    int8_t v4665 = v4655.read(); // v4655[k179];	// L5562
    int8_t b179;	// L5563
    b179 = v4665;	// L5564
    int8_t v4667 = a179;	// L5565
    int8_t v4668 = b179;	// L5566
    int16_t v4669 = v4667;	// L5567
    int16_t v4670 = v4668;	// L5568
    int16_t v4671 = v4669 * v4670;	// L5569
    int32_t v4672 = v179;	// L5570
    ap_int<33> v4673 = v4672;	// L5571
    ap_int<33> v4674 = v4671;	// L5572
    ap_int<33> v4675 = v4673 + v4674;	// L5573
    int32_t v4676 = v4675;	// L5574
    v179 = v4676;	// L5575
    int8_t v4677 = a179;	// L5576
    v4656.write(v4677); // v4656[k179] = v4677;	// L5577
    int8_t v4678 = b179;	// L5578
    v4657.write(v4678); // v4657[k179] = v4678;	// L5579
  }
  int32_t v4679 = v179;	// L5581
  v4658[v4659][v4660] = v4679;	// L5582
}

void PE_kernel_gemm_4_11(
  hls::stream< int8_t > &v4680 /* v4680[16] */,
  hls::stream< int8_t > &v4681 /* v4681[16] */,
  hls::stream< int8_t > &v4682 /* v4682[16] */,
  hls::stream< int8_t > &v4683 /* v4683[16] */,
  int32_t v4684[16][16],
  int v4685,
  int v4686
) {	// L5585
  #pragma HLS stream variable=v4680 depth=17
  #pragma HLS stream variable=v4681 depth=17
  #pragma HLS stream variable=v4682 depth=17
  #pragma HLS stream variable=v4683 depth=17
  #pragma HLS array_partition variable=v4684 complete dim=1
  #pragma HLS array_partition variable=v4684 complete dim=2

  int32_t v180;	// L5587
  v180 = 0;	// L5588
  l_reduction_k180: for (int k180 = 0; k180 < 16; k180++) {	// L5589
  #pragma HLS pipeline II=1
    int8_t v4689 = v4680.read(); // v4680[k180];	// L5590
    int8_t a180;	// L5591
    a180 = v4689;	// L5592
    int8_t v4691 = v4681.read(); // v4681[k180];	// L5593
    int8_t b180;	// L5594
    b180 = v4691;	// L5595
    int8_t v4693 = a180;	// L5596
    int8_t v4694 = b180;	// L5597
    int16_t v4695 = v4693;	// L5598
    int16_t v4696 = v4694;	// L5599
    int16_t v4697 = v4695 * v4696;	// L5600
    int32_t v4698 = v180;	// L5601
    ap_int<33> v4699 = v4698;	// L5602
    ap_int<33> v4700 = v4697;	// L5603
    ap_int<33> v4701 = v4699 + v4700;	// L5604
    int32_t v4702 = v4701;	// L5605
    v180 = v4702;	// L5606
    int8_t v4703 = a180;	// L5607
    v4682.write(v4703); // v4682[k180] = v4703;	// L5608
    int8_t v4704 = b180;	// L5609
    v4683.write(v4704); // v4683[k180] = v4704;	// L5610
  }
  int32_t v4705 = v180;	// L5612
  v4684[v4685][v4686] = v4705;	// L5613
}

void PE_kernel_gemm_5_11(
  hls::stream< int8_t > &v4706 /* v4706[16] */,
  hls::stream< int8_t > &v4707 /* v4707[16] */,
  hls::stream< int8_t > &v4708 /* v4708[16] */,
  hls::stream< int8_t > &v4709 /* v4709[16] */,
  int32_t v4710[16][16],
  int v4711,
  int v4712
) {	// L5616
  #pragma HLS stream variable=v4706 depth=17
  #pragma HLS stream variable=v4707 depth=17
  #pragma HLS stream variable=v4708 depth=17
  #pragma HLS stream variable=v4709 depth=17
  #pragma HLS array_partition variable=v4710 complete dim=1
  #pragma HLS array_partition variable=v4710 complete dim=2

  int32_t v181;	// L5618
  v181 = 0;	// L5619
  l_reduction_k181: for (int k181 = 0; k181 < 16; k181++) {	// L5620
  #pragma HLS pipeline II=1
    int8_t v4715 = v4706.read(); // v4706[k181];	// L5621
    int8_t a181;	// L5622
    a181 = v4715;	// L5623
    int8_t v4717 = v4707.read(); // v4707[k181];	// L5624
    int8_t b181;	// L5625
    b181 = v4717;	// L5626
    int8_t v4719 = a181;	// L5627
    int8_t v4720 = b181;	// L5628
    int16_t v4721 = v4719;	// L5629
    int16_t v4722 = v4720;	// L5630
    int16_t v4723 = v4721 * v4722;	// L5631
    int32_t v4724 = v181;	// L5632
    ap_int<33> v4725 = v4724;	// L5633
    ap_int<33> v4726 = v4723;	// L5634
    ap_int<33> v4727 = v4725 + v4726;	// L5635
    int32_t v4728 = v4727;	// L5636
    v181 = v4728;	// L5637
    int8_t v4729 = a181;	// L5638
    v4708.write(v4729); // v4708[k181] = v4729;	// L5639
    int8_t v4730 = b181;	// L5640
    v4709.write(v4730); // v4709[k181] = v4730;	// L5641
  }
  int32_t v4731 = v181;	// L5643
  v4710[v4711][v4712] = v4731;	// L5644
}

void PE_kernel_gemm_6_11(
  hls::stream< int8_t > &v4732 /* v4732[16] */,
  hls::stream< int8_t > &v4733 /* v4733[16] */,
  hls::stream< int8_t > &v4734 /* v4734[16] */,
  hls::stream< int8_t > &v4735 /* v4735[16] */,
  int32_t v4736[16][16],
  int v4737,
  int v4738
) {	// L5647
  #pragma HLS stream variable=v4732 depth=17
  #pragma HLS stream variable=v4733 depth=17
  #pragma HLS stream variable=v4734 depth=17
  #pragma HLS stream variable=v4735 depth=17
  #pragma HLS array_partition variable=v4736 complete dim=1
  #pragma HLS array_partition variable=v4736 complete dim=2

  int32_t v182;	// L5649
  v182 = 0;	// L5650
  l_reduction_k182: for (int k182 = 0; k182 < 16; k182++) {	// L5651
  #pragma HLS pipeline II=1
    int8_t v4741 = v4732.read(); // v4732[k182];	// L5652
    int8_t a182;	// L5653
    a182 = v4741;	// L5654
    int8_t v4743 = v4733.read(); // v4733[k182];	// L5655
    int8_t b182;	// L5656
    b182 = v4743;	// L5657
    int8_t v4745 = a182;	// L5658
    int8_t v4746 = b182;	// L5659
    int16_t v4747 = v4745;	// L5660
    int16_t v4748 = v4746;	// L5661
    int16_t v4749 = v4747 * v4748;	// L5662
    int32_t v4750 = v182;	// L5663
    ap_int<33> v4751 = v4750;	// L5664
    ap_int<33> v4752 = v4749;	// L5665
    ap_int<33> v4753 = v4751 + v4752;	// L5666
    int32_t v4754 = v4753;	// L5667
    v182 = v4754;	// L5668
    int8_t v4755 = a182;	// L5669
    v4734.write(v4755); // v4734[k182] = v4755;	// L5670
    int8_t v4756 = b182;	// L5671
    v4735.write(v4756); // v4735[k182] = v4756;	// L5672
  }
  int32_t v4757 = v182;	// L5674
  v4736[v4737][v4738] = v4757;	// L5675
}

void PE_kernel_gemm_7_11(
  hls::stream< int8_t > &v4758 /* v4758[16] */,
  hls::stream< int8_t > &v4759 /* v4759[16] */,
  hls::stream< int8_t > &v4760 /* v4760[16] */,
  hls::stream< int8_t > &v4761 /* v4761[16] */,
  int32_t v4762[16][16],
  int v4763,
  int v4764
) {	// L5678
  #pragma HLS stream variable=v4758 depth=17
  #pragma HLS stream variable=v4759 depth=17
  #pragma HLS stream variable=v4760 depth=17
  #pragma HLS stream variable=v4761 depth=17
  #pragma HLS array_partition variable=v4762 complete dim=1
  #pragma HLS array_partition variable=v4762 complete dim=2

  int32_t v183;	// L5680
  v183 = 0;	// L5681
  l_reduction_k183: for (int k183 = 0; k183 < 16; k183++) {	// L5682
  #pragma HLS pipeline II=1
    int8_t v4767 = v4758.read(); // v4758[k183];	// L5683
    int8_t a183;	// L5684
    a183 = v4767;	// L5685
    int8_t v4769 = v4759.read(); // v4759[k183];	// L5686
    int8_t b183;	// L5687
    b183 = v4769;	// L5688
    int8_t v4771 = a183;	// L5689
    int8_t v4772 = b183;	// L5690
    int16_t v4773 = v4771;	// L5691
    int16_t v4774 = v4772;	// L5692
    int16_t v4775 = v4773 * v4774;	// L5693
    int32_t v4776 = v183;	// L5694
    ap_int<33> v4777 = v4776;	// L5695
    ap_int<33> v4778 = v4775;	// L5696
    ap_int<33> v4779 = v4777 + v4778;	// L5697
    int32_t v4780 = v4779;	// L5698
    v183 = v4780;	// L5699
    int8_t v4781 = a183;	// L5700
    v4760.write(v4781); // v4760[k183] = v4781;	// L5701
    int8_t v4782 = b183;	// L5702
    v4761.write(v4782); // v4761[k183] = v4782;	// L5703
  }
  int32_t v4783 = v183;	// L5705
  v4762[v4763][v4764] = v4783;	// L5706
}

void PE_kernel_gemm_8_11(
  hls::stream< int8_t > &v4784 /* v4784[16] */,
  hls::stream< int8_t > &v4785 /* v4785[16] */,
  hls::stream< int8_t > &v4786 /* v4786[16] */,
  hls::stream< int8_t > &v4787 /* v4787[16] */,
  int32_t v4788[16][16],
  int v4789,
  int v4790
) {	// L5709
  #pragma HLS stream variable=v4784 depth=17
  #pragma HLS stream variable=v4785 depth=17
  #pragma HLS stream variable=v4786 depth=17
  #pragma HLS stream variable=v4787 depth=17
  #pragma HLS array_partition variable=v4788 complete dim=1
  #pragma HLS array_partition variable=v4788 complete dim=2

  int32_t v184;	// L5711
  v184 = 0;	// L5712
  l_reduction_k184: for (int k184 = 0; k184 < 16; k184++) {	// L5713
  #pragma HLS pipeline II=1
    int8_t v4793 = v4784.read(); // v4784[k184];	// L5714
    int8_t a184;	// L5715
    a184 = v4793;	// L5716
    int8_t v4795 = v4785.read(); // v4785[k184];	// L5717
    int8_t b184;	// L5718
    b184 = v4795;	// L5719
    int8_t v4797 = a184;	// L5720
    int8_t v4798 = b184;	// L5721
    int16_t v4799 = v4797;	// L5722
    int16_t v4800 = v4798;	// L5723
    int16_t v4801 = v4799 * v4800;	// L5724
    int32_t v4802 = v184;	// L5725
    ap_int<33> v4803 = v4802;	// L5726
    ap_int<33> v4804 = v4801;	// L5727
    ap_int<33> v4805 = v4803 + v4804;	// L5728
    int32_t v4806 = v4805;	// L5729
    v184 = v4806;	// L5730
    int8_t v4807 = a184;	// L5731
    v4786.write(v4807); // v4786[k184] = v4807;	// L5732
    int8_t v4808 = b184;	// L5733
    v4787.write(v4808); // v4787[k184] = v4808;	// L5734
  }
  int32_t v4809 = v184;	// L5736
  v4788[v4789][v4790] = v4809;	// L5737
}

void PE_kernel_gemm_9_11(
  hls::stream< int8_t > &v4810 /* v4810[16] */,
  hls::stream< int8_t > &v4811 /* v4811[16] */,
  hls::stream< int8_t > &v4812 /* v4812[16] */,
  hls::stream< int8_t > &v4813 /* v4813[16] */,
  int32_t v4814[16][16],
  int v4815,
  int v4816
) {	// L5740
  #pragma HLS stream variable=v4810 depth=17
  #pragma HLS stream variable=v4811 depth=17
  #pragma HLS stream variable=v4812 depth=17
  #pragma HLS stream variable=v4813 depth=17
  #pragma HLS array_partition variable=v4814 complete dim=1
  #pragma HLS array_partition variable=v4814 complete dim=2

  int32_t v185;	// L5742
  v185 = 0;	// L5743
  l_reduction_k185: for (int k185 = 0; k185 < 16; k185++) {	// L5744
  #pragma HLS pipeline II=1
    int8_t v4819 = v4810.read(); // v4810[k185];	// L5745
    int8_t a185;	// L5746
    a185 = v4819;	// L5747
    int8_t v4821 = v4811.read(); // v4811[k185];	// L5748
    int8_t b185;	// L5749
    b185 = v4821;	// L5750
    int8_t v4823 = a185;	// L5751
    int8_t v4824 = b185;	// L5752
    int16_t v4825 = v4823;	// L5753
    int16_t v4826 = v4824;	// L5754
    int16_t v4827 = v4825 * v4826;	// L5755
    int32_t v4828 = v185;	// L5756
    ap_int<33> v4829 = v4828;	// L5757
    ap_int<33> v4830 = v4827;	// L5758
    ap_int<33> v4831 = v4829 + v4830;	// L5759
    int32_t v4832 = v4831;	// L5760
    v185 = v4832;	// L5761
    int8_t v4833 = a185;	// L5762
    v4812.write(v4833); // v4812[k185] = v4833;	// L5763
    int8_t v4834 = b185;	// L5764
    v4813.write(v4834); // v4813[k185] = v4834;	// L5765
  }
  int32_t v4835 = v185;	// L5767
  v4814[v4815][v4816] = v4835;	// L5768
}

void PE_kernel_gemm_10_11(
  hls::stream< int8_t > &v4836 /* v4836[16] */,
  hls::stream< int8_t > &v4837 /* v4837[16] */,
  hls::stream< int8_t > &v4838 /* v4838[16] */,
  hls::stream< int8_t > &v4839 /* v4839[16] */,
  int32_t v4840[16][16],
  int v4841,
  int v4842
) {	// L5771
  #pragma HLS stream variable=v4836 depth=17
  #pragma HLS stream variable=v4837 depth=17
  #pragma HLS stream variable=v4838 depth=17
  #pragma HLS stream variable=v4839 depth=17
  #pragma HLS array_partition variable=v4840 complete dim=1
  #pragma HLS array_partition variable=v4840 complete dim=2

  int32_t v186;	// L5773
  v186 = 0;	// L5774
  l_reduction_k186: for (int k186 = 0; k186 < 16; k186++) {	// L5775
  #pragma HLS pipeline II=1
    int8_t v4845 = v4836.read(); // v4836[k186];	// L5776
    int8_t a186;	// L5777
    a186 = v4845;	// L5778
    int8_t v4847 = v4837.read(); // v4837[k186];	// L5779
    int8_t b186;	// L5780
    b186 = v4847;	// L5781
    int8_t v4849 = a186;	// L5782
    int8_t v4850 = b186;	// L5783
    int16_t v4851 = v4849;	// L5784
    int16_t v4852 = v4850;	// L5785
    int16_t v4853 = v4851 * v4852;	// L5786
    int32_t v4854 = v186;	// L5787
    ap_int<33> v4855 = v4854;	// L5788
    ap_int<33> v4856 = v4853;	// L5789
    ap_int<33> v4857 = v4855 + v4856;	// L5790
    int32_t v4858 = v4857;	// L5791
    v186 = v4858;	// L5792
    int8_t v4859 = a186;	// L5793
    v4838.write(v4859); // v4838[k186] = v4859;	// L5794
    int8_t v4860 = b186;	// L5795
    v4839.write(v4860); // v4839[k186] = v4860;	// L5796
  }
  int32_t v4861 = v186;	// L5798
  v4840[v4841][v4842] = v4861;	// L5799
}

void PE_kernel_gemm_11_11(
  hls::stream< int8_t > &v4862 /* v4862[16] */,
  hls::stream< int8_t > &v4863 /* v4863[16] */,
  hls::stream< int8_t > &v4864 /* v4864[16] */,
  hls::stream< int8_t > &v4865 /* v4865[16] */,
  int32_t v4866[16][16],
  int v4867,
  int v4868
) {	// L5802
  #pragma HLS stream variable=v4862 depth=17
  #pragma HLS stream variable=v4863 depth=17
  #pragma HLS stream variable=v4864 depth=17
  #pragma HLS stream variable=v4865 depth=17
  #pragma HLS array_partition variable=v4866 complete dim=1
  #pragma HLS array_partition variable=v4866 complete dim=2

  int32_t v187;	// L5804
  v187 = 0;	// L5805
  l_reduction_k187: for (int k187 = 0; k187 < 16; k187++) {	// L5806
  #pragma HLS pipeline II=1
    int8_t v4871 = v4862.read(); // v4862[k187];	// L5807
    int8_t a187;	// L5808
    a187 = v4871;	// L5809
    int8_t v4873 = v4863.read(); // v4863[k187];	// L5810
    int8_t b187;	// L5811
    b187 = v4873;	// L5812
    int8_t v4875 = a187;	// L5813
    int8_t v4876 = b187;	// L5814
    int16_t v4877 = v4875;	// L5815
    int16_t v4878 = v4876;	// L5816
    int16_t v4879 = v4877 * v4878;	// L5817
    int32_t v4880 = v187;	// L5818
    ap_int<33> v4881 = v4880;	// L5819
    ap_int<33> v4882 = v4879;	// L5820
    ap_int<33> v4883 = v4881 + v4882;	// L5821
    int32_t v4884 = v4883;	// L5822
    v187 = v4884;	// L5823
    int8_t v4885 = a187;	// L5824
    v4864.write(v4885); // v4864[k187] = v4885;	// L5825
    int8_t v4886 = b187;	// L5826
    v4865.write(v4886); // v4865[k187] = v4886;	// L5827
  }
  int32_t v4887 = v187;	// L5829
  v4866[v4867][v4868] = v4887;	// L5830
}

void PE_kernel_gemm_12_11(
  hls::stream< int8_t > &v4888 /* v4888[16] */,
  hls::stream< int8_t > &v4889 /* v4889[16] */,
  hls::stream< int8_t > &v4890 /* v4890[16] */,
  hls::stream< int8_t > &v4891 /* v4891[16] */,
  int32_t v4892[16][16],
  int v4893,
  int v4894
) {	// L5833
  #pragma HLS stream variable=v4888 depth=17
  #pragma HLS stream variable=v4889 depth=17
  #pragma HLS stream variable=v4890 depth=17
  #pragma HLS stream variable=v4891 depth=17
  #pragma HLS array_partition variable=v4892 complete dim=1
  #pragma HLS array_partition variable=v4892 complete dim=2

  int32_t v188;	// L5835
  v188 = 0;	// L5836
  l_reduction_k188: for (int k188 = 0; k188 < 16; k188++) {	// L5837
  #pragma HLS pipeline II=1
    int8_t v4897 = v4888.read(); // v4888[k188];	// L5838
    int8_t a188;	// L5839
    a188 = v4897;	// L5840
    int8_t v4899 = v4889.read(); // v4889[k188];	// L5841
    int8_t b188;	// L5842
    b188 = v4899;	// L5843
    int8_t v4901 = a188;	// L5844
    int8_t v4902 = b188;	// L5845
    int16_t v4903 = v4901;	// L5846
    int16_t v4904 = v4902;	// L5847
    int16_t v4905 = v4903 * v4904;	// L5848
    int32_t v4906 = v188;	// L5849
    ap_int<33> v4907 = v4906;	// L5850
    ap_int<33> v4908 = v4905;	// L5851
    ap_int<33> v4909 = v4907 + v4908;	// L5852
    int32_t v4910 = v4909;	// L5853
    v188 = v4910;	// L5854
    int8_t v4911 = a188;	// L5855
    v4890.write(v4911); // v4890[k188] = v4911;	// L5856
    int8_t v4912 = b188;	// L5857
    v4891.write(v4912); // v4891[k188] = v4912;	// L5858
  }
  int32_t v4913 = v188;	// L5860
  v4892[v4893][v4894] = v4913;	// L5861
}

void PE_kernel_gemm_13_11(
  hls::stream< int8_t > &v4914 /* v4914[16] */,
  hls::stream< int8_t > &v4915 /* v4915[16] */,
  hls::stream< int8_t > &v4916 /* v4916[16] */,
  hls::stream< int8_t > &v4917 /* v4917[16] */,
  int32_t v4918[16][16],
  int v4919,
  int v4920
) {	// L5864
  #pragma HLS stream variable=v4914 depth=17
  #pragma HLS stream variable=v4915 depth=17
  #pragma HLS stream variable=v4916 depth=17
  #pragma HLS stream variable=v4917 depth=17
  #pragma HLS array_partition variable=v4918 complete dim=1
  #pragma HLS array_partition variable=v4918 complete dim=2

  int32_t v189;	// L5866
  v189 = 0;	// L5867
  l_reduction_k189: for (int k189 = 0; k189 < 16; k189++) {	// L5868
  #pragma HLS pipeline II=1
    int8_t v4923 = v4914.read(); // v4914[k189];	// L5869
    int8_t a189;	// L5870
    a189 = v4923;	// L5871
    int8_t v4925 = v4915.read(); // v4915[k189];	// L5872
    int8_t b189;	// L5873
    b189 = v4925;	// L5874
    int8_t v4927 = a189;	// L5875
    int8_t v4928 = b189;	// L5876
    int16_t v4929 = v4927;	// L5877
    int16_t v4930 = v4928;	// L5878
    int16_t v4931 = v4929 * v4930;	// L5879
    int32_t v4932 = v189;	// L5880
    ap_int<33> v4933 = v4932;	// L5881
    ap_int<33> v4934 = v4931;	// L5882
    ap_int<33> v4935 = v4933 + v4934;	// L5883
    int32_t v4936 = v4935;	// L5884
    v189 = v4936;	// L5885
    int8_t v4937 = a189;	// L5886
    v4916.write(v4937); // v4916[k189] = v4937;	// L5887
    int8_t v4938 = b189;	// L5888
    v4917.write(v4938); // v4917[k189] = v4938;	// L5889
  }
  int32_t v4939 = v189;	// L5891
  v4918[v4919][v4920] = v4939;	// L5892
}

void PE_kernel_gemm_14_11(
  hls::stream< int8_t > &v4940 /* v4940[16] */,
  hls::stream< int8_t > &v4941 /* v4941[16] */,
  hls::stream< int8_t > &v4942 /* v4942[16] */,
  hls::stream< int8_t > &v4943 /* v4943[16] */,
  int32_t v4944[16][16],
  int v4945,
  int v4946
) {	// L5895
  #pragma HLS stream variable=v4940 depth=17
  #pragma HLS stream variable=v4941 depth=17
  #pragma HLS stream variable=v4942 depth=17
  #pragma HLS stream variable=v4943 depth=17
  #pragma HLS array_partition variable=v4944 complete dim=1
  #pragma HLS array_partition variable=v4944 complete dim=2

  int32_t v190;	// L5897
  v190 = 0;	// L5898
  l_reduction_k190: for (int k190 = 0; k190 < 16; k190++) {	// L5899
  #pragma HLS pipeline II=1
    int8_t v4949 = v4940.read(); // v4940[k190];	// L5900
    int8_t a190;	// L5901
    a190 = v4949;	// L5902
    int8_t v4951 = v4941.read(); // v4941[k190];	// L5903
    int8_t b190;	// L5904
    b190 = v4951;	// L5905
    int8_t v4953 = a190;	// L5906
    int8_t v4954 = b190;	// L5907
    int16_t v4955 = v4953;	// L5908
    int16_t v4956 = v4954;	// L5909
    int16_t v4957 = v4955 * v4956;	// L5910
    int32_t v4958 = v190;	// L5911
    ap_int<33> v4959 = v4958;	// L5912
    ap_int<33> v4960 = v4957;	// L5913
    ap_int<33> v4961 = v4959 + v4960;	// L5914
    int32_t v4962 = v4961;	// L5915
    v190 = v4962;	// L5916
    int8_t v4963 = a190;	// L5917
    v4942.write(v4963); // v4942[k190] = v4963;	// L5918
    int8_t v4964 = b190;	// L5919
    v4943.write(v4964); // v4943[k190] = v4964;	// L5920
  }
  int32_t v4965 = v190;	// L5922
  v4944[v4945][v4946] = v4965;	// L5923
}

void PE_kernel_gemm_15_11(
  hls::stream< int8_t > &v4966 /* v4966[16] */,
  hls::stream< int8_t > &v4967 /* v4967[16] */,
  hls::stream< int8_t > &v4968 /* v4968[16] */,
  hls::stream< int8_t > &v4969 /* v4969[16] */,
  int32_t v4970[16][16],
  int v4971,
  int v4972
) {	// L5926
  #pragma HLS stream variable=v4966 depth=17
  #pragma HLS stream variable=v4967 depth=17
  #pragma HLS stream variable=v4968 depth=17
  #pragma HLS stream variable=v4969 depth=17
  #pragma HLS array_partition variable=v4970 complete dim=1
  #pragma HLS array_partition variable=v4970 complete dim=2

  int32_t v191;	// L5928
  v191 = 0;	// L5929
  l_reduction_k191: for (int k191 = 0; k191 < 16; k191++) {	// L5930
  #pragma HLS pipeline II=1
    int8_t v4975 = v4966.read(); // v4966[k191];	// L5931
    int8_t a191;	// L5932
    a191 = v4975;	// L5933
    int8_t v4977 = v4967.read(); // v4967[k191];	// L5934
    int8_t b191;	// L5935
    b191 = v4977;	// L5936
    int8_t v4979 = a191;	// L5937
    int8_t v4980 = b191;	// L5938
    int16_t v4981 = v4979;	// L5939
    int16_t v4982 = v4980;	// L5940
    int16_t v4983 = v4981 * v4982;	// L5941
    int32_t v4984 = v191;	// L5942
    ap_int<33> v4985 = v4984;	// L5943
    ap_int<33> v4986 = v4983;	// L5944
    ap_int<33> v4987 = v4985 + v4986;	// L5945
    int32_t v4988 = v4987;	// L5946
    v191 = v4988;	// L5947
    int8_t v4989 = a191;	// L5948
    v4968.write(v4989); // v4968[k191] = v4989;	// L5949
    int8_t v4990 = b191;	// L5950
    v4969.write(v4990); // v4969[k191] = v4990;	// L5951
  }
  int32_t v4991 = v191;	// L5953
  v4970[v4971][v4972] = v4991;	// L5954
}

void PE_kernel_gemm_0_12(
  hls::stream< int8_t > &v4992 /* v4992[16] */,
  hls::stream< int8_t > &v4993 /* v4993[16] */,
  hls::stream< int8_t > &v4994 /* v4994[16] */,
  hls::stream< int8_t > &v4995 /* v4995[16] */,
  int32_t v4996[16][16],
  int v4997,
  int v4998
) {	// L5957
  #pragma HLS stream variable=v4992 depth=17
  #pragma HLS stream variable=v4993 depth=17
  #pragma HLS stream variable=v4994 depth=17
  #pragma HLS stream variable=v4995 depth=17
  #pragma HLS array_partition variable=v4996 complete dim=1
  #pragma HLS array_partition variable=v4996 complete dim=2

  int32_t v192;	// L5959
  v192 = 0;	// L5960
  l_reduction_k192: for (int k192 = 0; k192 < 16; k192++) {	// L5961
  #pragma HLS pipeline II=1
    int8_t v5001 = v4992.read(); // v4992[k192];	// L5962
    int8_t a192;	// L5963
    a192 = v5001;	// L5964
    int8_t v5003 = v4993.read(); // v4993[k192];	// L5965
    int8_t b192;	// L5966
    b192 = v5003;	// L5967
    int8_t v5005 = a192;	// L5968
    int8_t v5006 = b192;	// L5969
    int16_t v5007 = v5005;	// L5970
    int16_t v5008 = v5006;	// L5971
    int16_t v5009 = v5007 * v5008;	// L5972
    int32_t v5010 = v192;	// L5973
    ap_int<33> v5011 = v5010;	// L5974
    ap_int<33> v5012 = v5009;	// L5975
    ap_int<33> v5013 = v5011 + v5012;	// L5976
    int32_t v5014 = v5013;	// L5977
    v192 = v5014;	// L5978
    int8_t v5015 = a192;	// L5979
    v4994.write(v5015); // v4994[k192] = v5015;	// L5980
    int8_t v5016 = b192;	// L5981
    v4995.write(v5016); // v4995[k192] = v5016;	// L5982
  }
  int32_t v5017 = v192;	// L5984
  v4996[v4997][v4998] = v5017;	// L5985
}

void PE_kernel_gemm_1_12(
  hls::stream< int8_t > &v5018 /* v5018[16] */,
  hls::stream< int8_t > &v5019 /* v5019[16] */,
  hls::stream< int8_t > &v5020 /* v5020[16] */,
  hls::stream< int8_t > &v5021 /* v5021[16] */,
  int32_t v5022[16][16],
  int v5023,
  int v5024
) {	// L5988
  #pragma HLS stream variable=v5018 depth=17
  #pragma HLS stream variable=v5019 depth=17
  #pragma HLS stream variable=v5020 depth=17
  #pragma HLS stream variable=v5021 depth=17
  #pragma HLS array_partition variable=v5022 complete dim=1
  #pragma HLS array_partition variable=v5022 complete dim=2

  int32_t v193;	// L5990
  v193 = 0;	// L5991
  l_reduction_k193: for (int k193 = 0; k193 < 16; k193++) {	// L5992
  #pragma HLS pipeline II=1
    int8_t v5027 = v5018.read(); // v5018[k193];	// L5993
    int8_t a193;	// L5994
    a193 = v5027;	// L5995
    int8_t v5029 = v5019.read(); // v5019[k193];	// L5996
    int8_t b193;	// L5997
    b193 = v5029;	// L5998
    int8_t v5031 = a193;	// L5999
    int8_t v5032 = b193;	// L6000
    int16_t v5033 = v5031;	// L6001
    int16_t v5034 = v5032;	// L6002
    int16_t v5035 = v5033 * v5034;	// L6003
    int32_t v5036 = v193;	// L6004
    ap_int<33> v5037 = v5036;	// L6005
    ap_int<33> v5038 = v5035;	// L6006
    ap_int<33> v5039 = v5037 + v5038;	// L6007
    int32_t v5040 = v5039;	// L6008
    v193 = v5040;	// L6009
    int8_t v5041 = a193;	// L6010
    v5020.write(v5041); // v5020[k193] = v5041;	// L6011
    int8_t v5042 = b193;	// L6012
    v5021.write(v5042); // v5021[k193] = v5042;	// L6013
  }
  int32_t v5043 = v193;	// L6015
  v5022[v5023][v5024] = v5043;	// L6016
}

void PE_kernel_gemm_2_12(
  hls::stream< int8_t > &v5044 /* v5044[16] */,
  hls::stream< int8_t > &v5045 /* v5045[16] */,
  hls::stream< int8_t > &v5046 /* v5046[16] */,
  hls::stream< int8_t > &v5047 /* v5047[16] */,
  int32_t v5048[16][16],
  int v5049,
  int v5050
) {	// L6019
  #pragma HLS stream variable=v5044 depth=17
  #pragma HLS stream variable=v5045 depth=17
  #pragma HLS stream variable=v5046 depth=17
  #pragma HLS stream variable=v5047 depth=17
  #pragma HLS array_partition variable=v5048 complete dim=1
  #pragma HLS array_partition variable=v5048 complete dim=2

  int32_t v194;	// L6021
  v194 = 0;	// L6022
  l_reduction_k194: for (int k194 = 0; k194 < 16; k194++) {	// L6023
  #pragma HLS pipeline II=1
    int8_t v5053 = v5044.read(); // v5044[k194];	// L6024
    int8_t a194;	// L6025
    a194 = v5053;	// L6026
    int8_t v5055 = v5045.read(); // v5045[k194];	// L6027
    int8_t b194;	// L6028
    b194 = v5055;	// L6029
    int8_t v5057 = a194;	// L6030
    int8_t v5058 = b194;	// L6031
    int16_t v5059 = v5057;	// L6032
    int16_t v5060 = v5058;	// L6033
    int16_t v5061 = v5059 * v5060;	// L6034
    int32_t v5062 = v194;	// L6035
    ap_int<33> v5063 = v5062;	// L6036
    ap_int<33> v5064 = v5061;	// L6037
    ap_int<33> v5065 = v5063 + v5064;	// L6038
    int32_t v5066 = v5065;	// L6039
    v194 = v5066;	// L6040
    int8_t v5067 = a194;	// L6041
    v5046.write(v5067); // v5046[k194] = v5067;	// L6042
    int8_t v5068 = b194;	// L6043
    v5047.write(v5068); // v5047[k194] = v5068;	// L6044
  }
  int32_t v5069 = v194;	// L6046
  v5048[v5049][v5050] = v5069;	// L6047
}

void PE_kernel_gemm_3_12(
  hls::stream< int8_t > &v5070 /* v5070[16] */,
  hls::stream< int8_t > &v5071 /* v5071[16] */,
  hls::stream< int8_t > &v5072 /* v5072[16] */,
  hls::stream< int8_t > &v5073 /* v5073[16] */,
  int32_t v5074[16][16],
  int v5075,
  int v5076
) {	// L6050
  #pragma HLS stream variable=v5070 depth=17
  #pragma HLS stream variable=v5071 depth=17
  #pragma HLS stream variable=v5072 depth=17
  #pragma HLS stream variable=v5073 depth=17
  #pragma HLS array_partition variable=v5074 complete dim=1
  #pragma HLS array_partition variable=v5074 complete dim=2

  int32_t v195;	// L6052
  v195 = 0;	// L6053
  l_reduction_k195: for (int k195 = 0; k195 < 16; k195++) {	// L6054
  #pragma HLS pipeline II=1
    int8_t v5079 = v5070.read(); // v5070[k195];	// L6055
    int8_t a195;	// L6056
    a195 = v5079;	// L6057
    int8_t v5081 = v5071.read(); // v5071[k195];	// L6058
    int8_t b195;	// L6059
    b195 = v5081;	// L6060
    int8_t v5083 = a195;	// L6061
    int8_t v5084 = b195;	// L6062
    int16_t v5085 = v5083;	// L6063
    int16_t v5086 = v5084;	// L6064
    int16_t v5087 = v5085 * v5086;	// L6065
    int32_t v5088 = v195;	// L6066
    ap_int<33> v5089 = v5088;	// L6067
    ap_int<33> v5090 = v5087;	// L6068
    ap_int<33> v5091 = v5089 + v5090;	// L6069
    int32_t v5092 = v5091;	// L6070
    v195 = v5092;	// L6071
    int8_t v5093 = a195;	// L6072
    v5072.write(v5093); // v5072[k195] = v5093;	// L6073
    int8_t v5094 = b195;	// L6074
    v5073.write(v5094); // v5073[k195] = v5094;	// L6075
  }
  int32_t v5095 = v195;	// L6077
  v5074[v5075][v5076] = v5095;	// L6078
}

void PE_kernel_gemm_4_12(
  hls::stream< int8_t > &v5096 /* v5096[16] */,
  hls::stream< int8_t > &v5097 /* v5097[16] */,
  hls::stream< int8_t > &v5098 /* v5098[16] */,
  hls::stream< int8_t > &v5099 /* v5099[16] */,
  int32_t v5100[16][16],
  int v5101,
  int v5102
) {	// L6081
  #pragma HLS stream variable=v5096 depth=17
  #pragma HLS stream variable=v5097 depth=17
  #pragma HLS stream variable=v5098 depth=17
  #pragma HLS stream variable=v5099 depth=17
  #pragma HLS array_partition variable=v5100 complete dim=1
  #pragma HLS array_partition variable=v5100 complete dim=2

  int32_t v196;	// L6083
  v196 = 0;	// L6084
  l_reduction_k196: for (int k196 = 0; k196 < 16; k196++) {	// L6085
  #pragma HLS pipeline II=1
    int8_t v5105 = v5096.read(); // v5096[k196];	// L6086
    int8_t a196;	// L6087
    a196 = v5105;	// L6088
    int8_t v5107 = v5097.read(); // v5097[k196];	// L6089
    int8_t b196;	// L6090
    b196 = v5107;	// L6091
    int8_t v5109 = a196;	// L6092
    int8_t v5110 = b196;	// L6093
    int16_t v5111 = v5109;	// L6094
    int16_t v5112 = v5110;	// L6095
    int16_t v5113 = v5111 * v5112;	// L6096
    int32_t v5114 = v196;	// L6097
    ap_int<33> v5115 = v5114;	// L6098
    ap_int<33> v5116 = v5113;	// L6099
    ap_int<33> v5117 = v5115 + v5116;	// L6100
    int32_t v5118 = v5117;	// L6101
    v196 = v5118;	// L6102
    int8_t v5119 = a196;	// L6103
    v5098.write(v5119); // v5098[k196] = v5119;	// L6104
    int8_t v5120 = b196;	// L6105
    v5099.write(v5120); // v5099[k196] = v5120;	// L6106
  }
  int32_t v5121 = v196;	// L6108
  v5100[v5101][v5102] = v5121;	// L6109
}

void PE_kernel_gemm_5_12(
  hls::stream< int8_t > &v5122 /* v5122[16] */,
  hls::stream< int8_t > &v5123 /* v5123[16] */,
  hls::stream< int8_t > &v5124 /* v5124[16] */,
  hls::stream< int8_t > &v5125 /* v5125[16] */,
  int32_t v5126[16][16],
  int v5127,
  int v5128
) {	// L6112
  #pragma HLS stream variable=v5122 depth=17
  #pragma HLS stream variable=v5123 depth=17
  #pragma HLS stream variable=v5124 depth=17
  #pragma HLS stream variable=v5125 depth=17
  #pragma HLS array_partition variable=v5126 complete dim=1
  #pragma HLS array_partition variable=v5126 complete dim=2

  int32_t v197;	// L6114
  v197 = 0;	// L6115
  l_reduction_k197: for (int k197 = 0; k197 < 16; k197++) {	// L6116
  #pragma HLS pipeline II=1
    int8_t v5131 = v5122.read(); // v5122[k197];	// L6117
    int8_t a197;	// L6118
    a197 = v5131;	// L6119
    int8_t v5133 = v5123.read(); // v5123[k197];	// L6120
    int8_t b197;	// L6121
    b197 = v5133;	// L6122
    int8_t v5135 = a197;	// L6123
    int8_t v5136 = b197;	// L6124
    int16_t v5137 = v5135;	// L6125
    int16_t v5138 = v5136;	// L6126
    int16_t v5139 = v5137 * v5138;	// L6127
    int32_t v5140 = v197;	// L6128
    ap_int<33> v5141 = v5140;	// L6129
    ap_int<33> v5142 = v5139;	// L6130
    ap_int<33> v5143 = v5141 + v5142;	// L6131
    int32_t v5144 = v5143;	// L6132
    v197 = v5144;	// L6133
    int8_t v5145 = a197;	// L6134
    v5124.write(v5145); // v5124[k197] = v5145;	// L6135
    int8_t v5146 = b197;	// L6136
    v5125.write(v5146); // v5125[k197] = v5146;	// L6137
  }
  int32_t v5147 = v197;	// L6139
  v5126[v5127][v5128] = v5147;	// L6140
}

void PE_kernel_gemm_6_12(
  hls::stream< int8_t > &v5148 /* v5148[16] */,
  hls::stream< int8_t > &v5149 /* v5149[16] */,
  hls::stream< int8_t > &v5150 /* v5150[16] */,
  hls::stream< int8_t > &v5151 /* v5151[16] */,
  int32_t v5152[16][16],
  int v5153,
  int v5154
) {	// L6143
  #pragma HLS stream variable=v5148 depth=17
  #pragma HLS stream variable=v5149 depth=17
  #pragma HLS stream variable=v5150 depth=17
  #pragma HLS stream variable=v5151 depth=17
  #pragma HLS array_partition variable=v5152 complete dim=1
  #pragma HLS array_partition variable=v5152 complete dim=2

  int32_t v198;	// L6145
  v198 = 0;	// L6146
  l_reduction_k198: for (int k198 = 0; k198 < 16; k198++) {	// L6147
  #pragma HLS pipeline II=1
    int8_t v5157 = v5148.read(); // v5148[k198];	// L6148
    int8_t a198;	// L6149
    a198 = v5157;	// L6150
    int8_t v5159 = v5149.read(); // v5149[k198];	// L6151
    int8_t b198;	// L6152
    b198 = v5159;	// L6153
    int8_t v5161 = a198;	// L6154
    int8_t v5162 = b198;	// L6155
    int16_t v5163 = v5161;	// L6156
    int16_t v5164 = v5162;	// L6157
    int16_t v5165 = v5163 * v5164;	// L6158
    int32_t v5166 = v198;	// L6159
    ap_int<33> v5167 = v5166;	// L6160
    ap_int<33> v5168 = v5165;	// L6161
    ap_int<33> v5169 = v5167 + v5168;	// L6162
    int32_t v5170 = v5169;	// L6163
    v198 = v5170;	// L6164
    int8_t v5171 = a198;	// L6165
    v5150.write(v5171); // v5150[k198] = v5171;	// L6166
    int8_t v5172 = b198;	// L6167
    v5151.write(v5172); // v5151[k198] = v5172;	// L6168
  }
  int32_t v5173 = v198;	// L6170
  v5152[v5153][v5154] = v5173;	// L6171
}

void PE_kernel_gemm_7_12(
  hls::stream< int8_t > &v5174 /* v5174[16] */,
  hls::stream< int8_t > &v5175 /* v5175[16] */,
  hls::stream< int8_t > &v5176 /* v5176[16] */,
  hls::stream< int8_t > &v5177 /* v5177[16] */,
  int32_t v5178[16][16],
  int v5179,
  int v5180
) {	// L6174
  #pragma HLS stream variable=v5174 depth=17
  #pragma HLS stream variable=v5175 depth=17
  #pragma HLS stream variable=v5176 depth=17
  #pragma HLS stream variable=v5177 depth=17
  #pragma HLS array_partition variable=v5178 complete dim=1
  #pragma HLS array_partition variable=v5178 complete dim=2

  int32_t v199;	// L6176
  v199 = 0;	// L6177
  l_reduction_k199: for (int k199 = 0; k199 < 16; k199++) {	// L6178
  #pragma HLS pipeline II=1
    int8_t v5183 = v5174.read(); // v5174[k199];	// L6179
    int8_t a199;	// L6180
    a199 = v5183;	// L6181
    int8_t v5185 = v5175.read(); // v5175[k199];	// L6182
    int8_t b199;	// L6183
    b199 = v5185;	// L6184
    int8_t v5187 = a199;	// L6185
    int8_t v5188 = b199;	// L6186
    int16_t v5189 = v5187;	// L6187
    int16_t v5190 = v5188;	// L6188
    int16_t v5191 = v5189 * v5190;	// L6189
    int32_t v5192 = v199;	// L6190
    ap_int<33> v5193 = v5192;	// L6191
    ap_int<33> v5194 = v5191;	// L6192
    ap_int<33> v5195 = v5193 + v5194;	// L6193
    int32_t v5196 = v5195;	// L6194
    v199 = v5196;	// L6195
    int8_t v5197 = a199;	// L6196
    v5176.write(v5197); // v5176[k199] = v5197;	// L6197
    int8_t v5198 = b199;	// L6198
    v5177.write(v5198); // v5177[k199] = v5198;	// L6199
  }
  int32_t v5199 = v199;	// L6201
  v5178[v5179][v5180] = v5199;	// L6202
}

void PE_kernel_gemm_8_12(
  hls::stream< int8_t > &v5200 /* v5200[16] */,
  hls::stream< int8_t > &v5201 /* v5201[16] */,
  hls::stream< int8_t > &v5202 /* v5202[16] */,
  hls::stream< int8_t > &v5203 /* v5203[16] */,
  int32_t v5204[16][16],
  int v5205,
  int v5206
) {	// L6205
  #pragma HLS stream variable=v5200 depth=17
  #pragma HLS stream variable=v5201 depth=17
  #pragma HLS stream variable=v5202 depth=17
  #pragma HLS stream variable=v5203 depth=17
  #pragma HLS array_partition variable=v5204 complete dim=1
  #pragma HLS array_partition variable=v5204 complete dim=2

  int32_t v200;	// L6207
  v200 = 0;	// L6208
  l_reduction_k200: for (int k200 = 0; k200 < 16; k200++) {	// L6209
  #pragma HLS pipeline II=1
    int8_t v5209 = v5200.read(); // v5200[k200];	// L6210
    int8_t a200;	// L6211
    a200 = v5209;	// L6212
    int8_t v5211 = v5201.read(); // v5201[k200];	// L6213
    int8_t b200;	// L6214
    b200 = v5211;	// L6215
    int8_t v5213 = a200;	// L6216
    int8_t v5214 = b200;	// L6217
    int16_t v5215 = v5213;	// L6218
    int16_t v5216 = v5214;	// L6219
    int16_t v5217 = v5215 * v5216;	// L6220
    int32_t v5218 = v200;	// L6221
    ap_int<33> v5219 = v5218;	// L6222
    ap_int<33> v5220 = v5217;	// L6223
    ap_int<33> v5221 = v5219 + v5220;	// L6224
    int32_t v5222 = v5221;	// L6225
    v200 = v5222;	// L6226
    int8_t v5223 = a200;	// L6227
    v5202.write(v5223); // v5202[k200] = v5223;	// L6228
    int8_t v5224 = b200;	// L6229
    v5203.write(v5224); // v5203[k200] = v5224;	// L6230
  }
  int32_t v5225 = v200;	// L6232
  v5204[v5205][v5206] = v5225;	// L6233
}

void PE_kernel_gemm_9_12(
  hls::stream< int8_t > &v5226 /* v5226[16] */,
  hls::stream< int8_t > &v5227 /* v5227[16] */,
  hls::stream< int8_t > &v5228 /* v5228[16] */,
  hls::stream< int8_t > &v5229 /* v5229[16] */,
  int32_t v5230[16][16],
  int v5231,
  int v5232
) {	// L6236
  #pragma HLS stream variable=v5226 depth=17
  #pragma HLS stream variable=v5227 depth=17
  #pragma HLS stream variable=v5228 depth=17
  #pragma HLS stream variable=v5229 depth=17
  #pragma HLS array_partition variable=v5230 complete dim=1
  #pragma HLS array_partition variable=v5230 complete dim=2

  int32_t v201;	// L6238
  v201 = 0;	// L6239
  l_reduction_k201: for (int k201 = 0; k201 < 16; k201++) {	// L6240
  #pragma HLS pipeline II=1
    int8_t v5235 = v5226.read(); // v5226[k201];	// L6241
    int8_t a201;	// L6242
    a201 = v5235;	// L6243
    int8_t v5237 = v5227.read(); // v5227[k201];	// L6244
    int8_t b201;	// L6245
    b201 = v5237;	// L6246
    int8_t v5239 = a201;	// L6247
    int8_t v5240 = b201;	// L6248
    int16_t v5241 = v5239;	// L6249
    int16_t v5242 = v5240;	// L6250
    int16_t v5243 = v5241 * v5242;	// L6251
    int32_t v5244 = v201;	// L6252
    ap_int<33> v5245 = v5244;	// L6253
    ap_int<33> v5246 = v5243;	// L6254
    ap_int<33> v5247 = v5245 + v5246;	// L6255
    int32_t v5248 = v5247;	// L6256
    v201 = v5248;	// L6257
    int8_t v5249 = a201;	// L6258
    v5228.write(v5249); // v5228[k201] = v5249;	// L6259
    int8_t v5250 = b201;	// L6260
    v5229.write(v5250); // v5229[k201] = v5250;	// L6261
  }
  int32_t v5251 = v201;	// L6263
  v5230[v5231][v5232] = v5251;	// L6264
}

void PE_kernel_gemm_10_12(
  hls::stream< int8_t > &v5252 /* v5252[16] */,
  hls::stream< int8_t > &v5253 /* v5253[16] */,
  hls::stream< int8_t > &v5254 /* v5254[16] */,
  hls::stream< int8_t > &v5255 /* v5255[16] */,
  int32_t v5256[16][16],
  int v5257,
  int v5258
) {	// L6267
  #pragma HLS stream variable=v5252 depth=17
  #pragma HLS stream variable=v5253 depth=17
  #pragma HLS stream variable=v5254 depth=17
  #pragma HLS stream variable=v5255 depth=17
  #pragma HLS array_partition variable=v5256 complete dim=1
  #pragma HLS array_partition variable=v5256 complete dim=2

  int32_t v202;	// L6269
  v202 = 0;	// L6270
  l_reduction_k202: for (int k202 = 0; k202 < 16; k202++) {	// L6271
  #pragma HLS pipeline II=1
    int8_t v5261 = v5252.read(); // v5252[k202];	// L6272
    int8_t a202;	// L6273
    a202 = v5261;	// L6274
    int8_t v5263 = v5253.read(); // v5253[k202];	// L6275
    int8_t b202;	// L6276
    b202 = v5263;	// L6277
    int8_t v5265 = a202;	// L6278
    int8_t v5266 = b202;	// L6279
    int16_t v5267 = v5265;	// L6280
    int16_t v5268 = v5266;	// L6281
    int16_t v5269 = v5267 * v5268;	// L6282
    int32_t v5270 = v202;	// L6283
    ap_int<33> v5271 = v5270;	// L6284
    ap_int<33> v5272 = v5269;	// L6285
    ap_int<33> v5273 = v5271 + v5272;	// L6286
    int32_t v5274 = v5273;	// L6287
    v202 = v5274;	// L6288
    int8_t v5275 = a202;	// L6289
    v5254.write(v5275); // v5254[k202] = v5275;	// L6290
    int8_t v5276 = b202;	// L6291
    v5255.write(v5276); // v5255[k202] = v5276;	// L6292
  }
  int32_t v5277 = v202;	// L6294
  v5256[v5257][v5258] = v5277;	// L6295
}

void PE_kernel_gemm_11_12(
  hls::stream< int8_t > &v5278 /* v5278[16] */,
  hls::stream< int8_t > &v5279 /* v5279[16] */,
  hls::stream< int8_t > &v5280 /* v5280[16] */,
  hls::stream< int8_t > &v5281 /* v5281[16] */,
  int32_t v5282[16][16],
  int v5283,
  int v5284
) {	// L6298
  #pragma HLS stream variable=v5278 depth=17
  #pragma HLS stream variable=v5279 depth=17
  #pragma HLS stream variable=v5280 depth=17
  #pragma HLS stream variable=v5281 depth=17
  #pragma HLS array_partition variable=v5282 complete dim=1
  #pragma HLS array_partition variable=v5282 complete dim=2

  int32_t v203;	// L6300
  v203 = 0;	// L6301
  l_reduction_k203: for (int k203 = 0; k203 < 16; k203++) {	// L6302
  #pragma HLS pipeline II=1
    int8_t v5287 = v5278.read(); // v5278[k203];	// L6303
    int8_t a203;	// L6304
    a203 = v5287;	// L6305
    int8_t v5289 = v5279.read(); // v5279[k203];	// L6306
    int8_t b203;	// L6307
    b203 = v5289;	// L6308
    int8_t v5291 = a203;	// L6309
    int8_t v5292 = b203;	// L6310
    int16_t v5293 = v5291;	// L6311
    int16_t v5294 = v5292;	// L6312
    int16_t v5295 = v5293 * v5294;	// L6313
    int32_t v5296 = v203;	// L6314
    ap_int<33> v5297 = v5296;	// L6315
    ap_int<33> v5298 = v5295;	// L6316
    ap_int<33> v5299 = v5297 + v5298;	// L6317
    int32_t v5300 = v5299;	// L6318
    v203 = v5300;	// L6319
    int8_t v5301 = a203;	// L6320
    v5280.write(v5301); // v5280[k203] = v5301;	// L6321
    int8_t v5302 = b203;	// L6322
    v5281.write(v5302); // v5281[k203] = v5302;	// L6323
  }
  int32_t v5303 = v203;	// L6325
  v5282[v5283][v5284] = v5303;	// L6326
}

void PE_kernel_gemm_12_12(
  hls::stream< int8_t > &v5304 /* v5304[16] */,
  hls::stream< int8_t > &v5305 /* v5305[16] */,
  hls::stream< int8_t > &v5306 /* v5306[16] */,
  hls::stream< int8_t > &v5307 /* v5307[16] */,
  int32_t v5308[16][16],
  int v5309,
  int v5310
) {	// L6329
  #pragma HLS stream variable=v5304 depth=17
  #pragma HLS stream variable=v5305 depth=17
  #pragma HLS stream variable=v5306 depth=17
  #pragma HLS stream variable=v5307 depth=17
  #pragma HLS array_partition variable=v5308 complete dim=1
  #pragma HLS array_partition variable=v5308 complete dim=2

  int32_t v204;	// L6331
  v204 = 0;	// L6332
  l_reduction_k204: for (int k204 = 0; k204 < 16; k204++) {	// L6333
  #pragma HLS pipeline II=1
    int8_t v5313 = v5304.read(); // v5304[k204];	// L6334
    int8_t a204;	// L6335
    a204 = v5313;	// L6336
    int8_t v5315 = v5305.read(); // v5305[k204];	// L6337
    int8_t b204;	// L6338
    b204 = v5315;	// L6339
    int8_t v5317 = a204;	// L6340
    int8_t v5318 = b204;	// L6341
    int16_t v5319 = v5317;	// L6342
    int16_t v5320 = v5318;	// L6343
    int16_t v5321 = v5319 * v5320;	// L6344
    int32_t v5322 = v204;	// L6345
    ap_int<33> v5323 = v5322;	// L6346
    ap_int<33> v5324 = v5321;	// L6347
    ap_int<33> v5325 = v5323 + v5324;	// L6348
    int32_t v5326 = v5325;	// L6349
    v204 = v5326;	// L6350
    int8_t v5327 = a204;	// L6351
    v5306.write(v5327); // v5306[k204] = v5327;	// L6352
    int8_t v5328 = b204;	// L6353
    v5307.write(v5328); // v5307[k204] = v5328;	// L6354
  }
  int32_t v5329 = v204;	// L6356
  v5308[v5309][v5310] = v5329;	// L6357
}

void PE_kernel_gemm_13_12(
  hls::stream< int8_t > &v5330 /* v5330[16] */,
  hls::stream< int8_t > &v5331 /* v5331[16] */,
  hls::stream< int8_t > &v5332 /* v5332[16] */,
  hls::stream< int8_t > &v5333 /* v5333[16] */,
  int32_t v5334[16][16],
  int v5335,
  int v5336
) {	// L6360
  #pragma HLS stream variable=v5330 depth=17
  #pragma HLS stream variable=v5331 depth=17
  #pragma HLS stream variable=v5332 depth=17
  #pragma HLS stream variable=v5333 depth=17
  #pragma HLS array_partition variable=v5334 complete dim=1
  #pragma HLS array_partition variable=v5334 complete dim=2

  int32_t v205;	// L6362
  v205 = 0;	// L6363
  l_reduction_k205: for (int k205 = 0; k205 < 16; k205++) {	// L6364
  #pragma HLS pipeline II=1
    int8_t v5339 = v5330.read(); // v5330[k205];	// L6365
    int8_t a205;	// L6366
    a205 = v5339;	// L6367
    int8_t v5341 = v5331.read(); // v5331[k205];	// L6368
    int8_t b205;	// L6369
    b205 = v5341;	// L6370
    int8_t v5343 = a205;	// L6371
    int8_t v5344 = b205;	// L6372
    int16_t v5345 = v5343;	// L6373
    int16_t v5346 = v5344;	// L6374
    int16_t v5347 = v5345 * v5346;	// L6375
    int32_t v5348 = v205;	// L6376
    ap_int<33> v5349 = v5348;	// L6377
    ap_int<33> v5350 = v5347;	// L6378
    ap_int<33> v5351 = v5349 + v5350;	// L6379
    int32_t v5352 = v5351;	// L6380
    v205 = v5352;	// L6381
    int8_t v5353 = a205;	// L6382
    v5332.write(v5353); // v5332[k205] = v5353;	// L6383
    int8_t v5354 = b205;	// L6384
    v5333.write(v5354); // v5333[k205] = v5354;	// L6385
  }
  int32_t v5355 = v205;	// L6387
  v5334[v5335][v5336] = v5355;	// L6388
}

void PE_kernel_gemm_14_12(
  hls::stream< int8_t > &v5356 /* v5356[16] */,
  hls::stream< int8_t > &v5357 /* v5357[16] */,
  hls::stream< int8_t > &v5358 /* v5358[16] */,
  hls::stream< int8_t > &v5359 /* v5359[16] */,
  int32_t v5360[16][16],
  int v5361,
  int v5362
) {	// L6391
  #pragma HLS stream variable=v5356 depth=17
  #pragma HLS stream variable=v5357 depth=17
  #pragma HLS stream variable=v5358 depth=17
  #pragma HLS stream variable=v5359 depth=17
  #pragma HLS array_partition variable=v5360 complete dim=1
  #pragma HLS array_partition variable=v5360 complete dim=2

  int32_t v206;	// L6393
  v206 = 0;	// L6394
  l_reduction_k206: for (int k206 = 0; k206 < 16; k206++) {	// L6395
  #pragma HLS pipeline II=1
    int8_t v5365 = v5356.read(); // v5356[k206];	// L6396
    int8_t a206;	// L6397
    a206 = v5365;	// L6398
    int8_t v5367 = v5357.read(); // v5357[k206];	// L6399
    int8_t b206;	// L6400
    b206 = v5367;	// L6401
    int8_t v5369 = a206;	// L6402
    int8_t v5370 = b206;	// L6403
    int16_t v5371 = v5369;	// L6404
    int16_t v5372 = v5370;	// L6405
    int16_t v5373 = v5371 * v5372;	// L6406
    int32_t v5374 = v206;	// L6407
    ap_int<33> v5375 = v5374;	// L6408
    ap_int<33> v5376 = v5373;	// L6409
    ap_int<33> v5377 = v5375 + v5376;	// L6410
    int32_t v5378 = v5377;	// L6411
    v206 = v5378;	// L6412
    int8_t v5379 = a206;	// L6413
    v5358.write(v5379); // v5358[k206] = v5379;	// L6414
    int8_t v5380 = b206;	// L6415
    v5359.write(v5380); // v5359[k206] = v5380;	// L6416
  }
  int32_t v5381 = v206;	// L6418
  v5360[v5361][v5362] = v5381;	// L6419
}

void PE_kernel_gemm_15_12(
  hls::stream< int8_t > &v5382 /* v5382[16] */,
  hls::stream< int8_t > &v5383 /* v5383[16] */,
  hls::stream< int8_t > &v5384 /* v5384[16] */,
  hls::stream< int8_t > &v5385 /* v5385[16] */,
  int32_t v5386[16][16],
  int v5387,
  int v5388
) {	// L6422
  #pragma HLS stream variable=v5382 depth=17
  #pragma HLS stream variable=v5383 depth=17
  #pragma HLS stream variable=v5384 depth=17
  #pragma HLS stream variable=v5385 depth=17
  #pragma HLS array_partition variable=v5386 complete dim=1
  #pragma HLS array_partition variable=v5386 complete dim=2

  int32_t v207;	// L6424
  v207 = 0;	// L6425
  l_reduction_k207: for (int k207 = 0; k207 < 16; k207++) {	// L6426
  #pragma HLS pipeline II=1
    int8_t v5391 = v5382.read(); // v5382[k207];	// L6427
    int8_t a207;	// L6428
    a207 = v5391;	// L6429
    int8_t v5393 = v5383.read(); // v5383[k207];	// L6430
    int8_t b207;	// L6431
    b207 = v5393;	// L6432
    int8_t v5395 = a207;	// L6433
    int8_t v5396 = b207;	// L6434
    int16_t v5397 = v5395;	// L6435
    int16_t v5398 = v5396;	// L6436
    int16_t v5399 = v5397 * v5398;	// L6437
    int32_t v5400 = v207;	// L6438
    ap_int<33> v5401 = v5400;	// L6439
    ap_int<33> v5402 = v5399;	// L6440
    ap_int<33> v5403 = v5401 + v5402;	// L6441
    int32_t v5404 = v5403;	// L6442
    v207 = v5404;	// L6443
    int8_t v5405 = a207;	// L6444
    v5384.write(v5405); // v5384[k207] = v5405;	// L6445
    int8_t v5406 = b207;	// L6446
    v5385.write(v5406); // v5385[k207] = v5406;	// L6447
  }
  int32_t v5407 = v207;	// L6449
  v5386[v5387][v5388] = v5407;	// L6450
}

void PE_kernel_gemm_0_13(
  hls::stream< int8_t > &v5408 /* v5408[16] */,
  hls::stream< int8_t > &v5409 /* v5409[16] */,
  hls::stream< int8_t > &v5410 /* v5410[16] */,
  hls::stream< int8_t > &v5411 /* v5411[16] */,
  int32_t v5412[16][16],
  int v5413,
  int v5414
) {	// L6453
  #pragma HLS stream variable=v5408 depth=17
  #pragma HLS stream variable=v5409 depth=17
  #pragma HLS stream variable=v5410 depth=17
  #pragma HLS stream variable=v5411 depth=17
  #pragma HLS array_partition variable=v5412 complete dim=1
  #pragma HLS array_partition variable=v5412 complete dim=2

  int32_t v208;	// L6455
  v208 = 0;	// L6456
  l_reduction_k208: for (int k208 = 0; k208 < 16; k208++) {	// L6457
  #pragma HLS pipeline II=1
    int8_t v5417 = v5408.read(); // v5408[k208];	// L6458
    int8_t a208;	// L6459
    a208 = v5417;	// L6460
    int8_t v5419 = v5409.read(); // v5409[k208];	// L6461
    int8_t b208;	// L6462
    b208 = v5419;	// L6463
    int8_t v5421 = a208;	// L6464
    int8_t v5422 = b208;	// L6465
    int16_t v5423 = v5421;	// L6466
    int16_t v5424 = v5422;	// L6467
    int16_t v5425 = v5423 * v5424;	// L6468
    int32_t v5426 = v208;	// L6469
    ap_int<33> v5427 = v5426;	// L6470
    ap_int<33> v5428 = v5425;	// L6471
    ap_int<33> v5429 = v5427 + v5428;	// L6472
    int32_t v5430 = v5429;	// L6473
    v208 = v5430;	// L6474
    int8_t v5431 = a208;	// L6475
    v5410.write(v5431); // v5410[k208] = v5431;	// L6476
    int8_t v5432 = b208;	// L6477
    v5411.write(v5432); // v5411[k208] = v5432;	// L6478
  }
  int32_t v5433 = v208;	// L6480
  v5412[v5413][v5414] = v5433;	// L6481
}

void PE_kernel_gemm_1_13(
  hls::stream< int8_t > &v5434 /* v5434[16] */,
  hls::stream< int8_t > &v5435 /* v5435[16] */,
  hls::stream< int8_t > &v5436 /* v5436[16] */,
  hls::stream< int8_t > &v5437 /* v5437[16] */,
  int32_t v5438[16][16],
  int v5439,
  int v5440
) {	// L6484
  #pragma HLS stream variable=v5434 depth=17
  #pragma HLS stream variable=v5435 depth=17
  #pragma HLS stream variable=v5436 depth=17
  #pragma HLS stream variable=v5437 depth=17
  #pragma HLS array_partition variable=v5438 complete dim=1
  #pragma HLS array_partition variable=v5438 complete dim=2

  int32_t v209;	// L6486
  v209 = 0;	// L6487
  l_reduction_k209: for (int k209 = 0; k209 < 16; k209++) {	// L6488
  #pragma HLS pipeline II=1
    int8_t v5443 = v5434.read(); // v5434[k209];	// L6489
    int8_t a209;	// L6490
    a209 = v5443;	// L6491
    int8_t v5445 = v5435.read(); // v5435[k209];	// L6492
    int8_t b209;	// L6493
    b209 = v5445;	// L6494
    int8_t v5447 = a209;	// L6495
    int8_t v5448 = b209;	// L6496
    int16_t v5449 = v5447;	// L6497
    int16_t v5450 = v5448;	// L6498
    int16_t v5451 = v5449 * v5450;	// L6499
    int32_t v5452 = v209;	// L6500
    ap_int<33> v5453 = v5452;	// L6501
    ap_int<33> v5454 = v5451;	// L6502
    ap_int<33> v5455 = v5453 + v5454;	// L6503
    int32_t v5456 = v5455;	// L6504
    v209 = v5456;	// L6505
    int8_t v5457 = a209;	// L6506
    v5436.write(v5457); // v5436[k209] = v5457;	// L6507
    int8_t v5458 = b209;	// L6508
    v5437.write(v5458); // v5437[k209] = v5458;	// L6509
  }
  int32_t v5459 = v209;	// L6511
  v5438[v5439][v5440] = v5459;	// L6512
}

void PE_kernel_gemm_2_13(
  hls::stream< int8_t > &v5460 /* v5460[16] */,
  hls::stream< int8_t > &v5461 /* v5461[16] */,
  hls::stream< int8_t > &v5462 /* v5462[16] */,
  hls::stream< int8_t > &v5463 /* v5463[16] */,
  int32_t v5464[16][16],
  int v5465,
  int v5466
) {	// L6515
  #pragma HLS stream variable=v5460 depth=17
  #pragma HLS stream variable=v5461 depth=17
  #pragma HLS stream variable=v5462 depth=17
  #pragma HLS stream variable=v5463 depth=17
  #pragma HLS array_partition variable=v5464 complete dim=1
  #pragma HLS array_partition variable=v5464 complete dim=2

  int32_t v210;	// L6517
  v210 = 0;	// L6518
  l_reduction_k210: for (int k210 = 0; k210 < 16; k210++) {	// L6519
  #pragma HLS pipeline II=1
    int8_t v5469 = v5460.read(); // v5460[k210];	// L6520
    int8_t a210;	// L6521
    a210 = v5469;	// L6522
    int8_t v5471 = v5461.read(); // v5461[k210];	// L6523
    int8_t b210;	// L6524
    b210 = v5471;	// L6525
    int8_t v5473 = a210;	// L6526
    int8_t v5474 = b210;	// L6527
    int16_t v5475 = v5473;	// L6528
    int16_t v5476 = v5474;	// L6529
    int16_t v5477 = v5475 * v5476;	// L6530
    int32_t v5478 = v210;	// L6531
    ap_int<33> v5479 = v5478;	// L6532
    ap_int<33> v5480 = v5477;	// L6533
    ap_int<33> v5481 = v5479 + v5480;	// L6534
    int32_t v5482 = v5481;	// L6535
    v210 = v5482;	// L6536
    int8_t v5483 = a210;	// L6537
    v5462.write(v5483); // v5462[k210] = v5483;	// L6538
    int8_t v5484 = b210;	// L6539
    v5463.write(v5484); // v5463[k210] = v5484;	// L6540
  }
  int32_t v5485 = v210;	// L6542
  v5464[v5465][v5466] = v5485;	// L6543
}

void PE_kernel_gemm_3_13(
  hls::stream< int8_t > &v5486 /* v5486[16] */,
  hls::stream< int8_t > &v5487 /* v5487[16] */,
  hls::stream< int8_t > &v5488 /* v5488[16] */,
  hls::stream< int8_t > &v5489 /* v5489[16] */,
  int32_t v5490[16][16],
  int v5491,
  int v5492
) {	// L6546
  #pragma HLS stream variable=v5486 depth=17
  #pragma HLS stream variable=v5487 depth=17
  #pragma HLS stream variable=v5488 depth=17
  #pragma HLS stream variable=v5489 depth=17
  #pragma HLS array_partition variable=v5490 complete dim=1
  #pragma HLS array_partition variable=v5490 complete dim=2

  int32_t v211;	// L6548
  v211 = 0;	// L6549
  l_reduction_k211: for (int k211 = 0; k211 < 16; k211++) {	// L6550
  #pragma HLS pipeline II=1
    int8_t v5495 = v5486.read(); // v5486[k211];	// L6551
    int8_t a211;	// L6552
    a211 = v5495;	// L6553
    int8_t v5497 = v5487.read(); // v5487[k211];	// L6554
    int8_t b211;	// L6555
    b211 = v5497;	// L6556
    int8_t v5499 = a211;	// L6557
    int8_t v5500 = b211;	// L6558
    int16_t v5501 = v5499;	// L6559
    int16_t v5502 = v5500;	// L6560
    int16_t v5503 = v5501 * v5502;	// L6561
    int32_t v5504 = v211;	// L6562
    ap_int<33> v5505 = v5504;	// L6563
    ap_int<33> v5506 = v5503;	// L6564
    ap_int<33> v5507 = v5505 + v5506;	// L6565
    int32_t v5508 = v5507;	// L6566
    v211 = v5508;	// L6567
    int8_t v5509 = a211;	// L6568
    v5488.write(v5509); // v5488[k211] = v5509;	// L6569
    int8_t v5510 = b211;	// L6570
    v5489.write(v5510); // v5489[k211] = v5510;	// L6571
  }
  int32_t v5511 = v211;	// L6573
  v5490[v5491][v5492] = v5511;	// L6574
}

void PE_kernel_gemm_4_13(
  hls::stream< int8_t > &v5512 /* v5512[16] */,
  hls::stream< int8_t > &v5513 /* v5513[16] */,
  hls::stream< int8_t > &v5514 /* v5514[16] */,
  hls::stream< int8_t > &v5515 /* v5515[16] */,
  int32_t v5516[16][16],
  int v5517,
  int v5518
) {	// L6577
  #pragma HLS stream variable=v5512 depth=17
  #pragma HLS stream variable=v5513 depth=17
  #pragma HLS stream variable=v5514 depth=17
  #pragma HLS stream variable=v5515 depth=17
  #pragma HLS array_partition variable=v5516 complete dim=1
  #pragma HLS array_partition variable=v5516 complete dim=2

  int32_t v212;	// L6579
  v212 = 0;	// L6580
  l_reduction_k212: for (int k212 = 0; k212 < 16; k212++) {	// L6581
  #pragma HLS pipeline II=1
    int8_t v5521 = v5512.read(); // v5512[k212];	// L6582
    int8_t a212;	// L6583
    a212 = v5521;	// L6584
    int8_t v5523 = v5513.read(); // v5513[k212];	// L6585
    int8_t b212;	// L6586
    b212 = v5523;	// L6587
    int8_t v5525 = a212;	// L6588
    int8_t v5526 = b212;	// L6589
    int16_t v5527 = v5525;	// L6590
    int16_t v5528 = v5526;	// L6591
    int16_t v5529 = v5527 * v5528;	// L6592
    int32_t v5530 = v212;	// L6593
    ap_int<33> v5531 = v5530;	// L6594
    ap_int<33> v5532 = v5529;	// L6595
    ap_int<33> v5533 = v5531 + v5532;	// L6596
    int32_t v5534 = v5533;	// L6597
    v212 = v5534;	// L6598
    int8_t v5535 = a212;	// L6599
    v5514.write(v5535); // v5514[k212] = v5535;	// L6600
    int8_t v5536 = b212;	// L6601
    v5515.write(v5536); // v5515[k212] = v5536;	// L6602
  }
  int32_t v5537 = v212;	// L6604
  v5516[v5517][v5518] = v5537;	// L6605
}

void PE_kernel_gemm_5_13(
  hls::stream< int8_t > &v5538 /* v5538[16] */,
  hls::stream< int8_t > &v5539 /* v5539[16] */,
  hls::stream< int8_t > &v5540 /* v5540[16] */,
  hls::stream< int8_t > &v5541 /* v5541[16] */,
  int32_t v5542[16][16],
  int v5543,
  int v5544
) {	// L6608
  #pragma HLS stream variable=v5538 depth=17
  #pragma HLS stream variable=v5539 depth=17
  #pragma HLS stream variable=v5540 depth=17
  #pragma HLS stream variable=v5541 depth=17
  #pragma HLS array_partition variable=v5542 complete dim=1
  #pragma HLS array_partition variable=v5542 complete dim=2

  int32_t v213;	// L6610
  v213 = 0;	// L6611
  l_reduction_k213: for (int k213 = 0; k213 < 16; k213++) {	// L6612
  #pragma HLS pipeline II=1
    int8_t v5547 = v5538.read(); // v5538[k213];	// L6613
    int8_t a213;	// L6614
    a213 = v5547;	// L6615
    int8_t v5549 = v5539.read(); // v5539[k213];	// L6616
    int8_t b213;	// L6617
    b213 = v5549;	// L6618
    int8_t v5551 = a213;	// L6619
    int8_t v5552 = b213;	// L6620
    int16_t v5553 = v5551;	// L6621
    int16_t v5554 = v5552;	// L6622
    int16_t v5555 = v5553 * v5554;	// L6623
    int32_t v5556 = v213;	// L6624
    ap_int<33> v5557 = v5556;	// L6625
    ap_int<33> v5558 = v5555;	// L6626
    ap_int<33> v5559 = v5557 + v5558;	// L6627
    int32_t v5560 = v5559;	// L6628
    v213 = v5560;	// L6629
    int8_t v5561 = a213;	// L6630
    v5540.write(v5561); // v5540[k213] = v5561;	// L6631
    int8_t v5562 = b213;	// L6632
    v5541.write(v5562); // v5541[k213] = v5562;	// L6633
  }
  int32_t v5563 = v213;	// L6635
  v5542[v5543][v5544] = v5563;	// L6636
}

void PE_kernel_gemm_6_13(
  hls::stream< int8_t > &v5564 /* v5564[16] */,
  hls::stream< int8_t > &v5565 /* v5565[16] */,
  hls::stream< int8_t > &v5566 /* v5566[16] */,
  hls::stream< int8_t > &v5567 /* v5567[16] */,
  int32_t v5568[16][16],
  int v5569,
  int v5570
) {	// L6639
  #pragma HLS stream variable=v5564 depth=17
  #pragma HLS stream variable=v5565 depth=17
  #pragma HLS stream variable=v5566 depth=17
  #pragma HLS stream variable=v5567 depth=17
  #pragma HLS array_partition variable=v5568 complete dim=1
  #pragma HLS array_partition variable=v5568 complete dim=2

  int32_t v214;	// L6641
  v214 = 0;	// L6642
  l_reduction_k214: for (int k214 = 0; k214 < 16; k214++) {	// L6643
  #pragma HLS pipeline II=1
    int8_t v5573 = v5564.read(); // v5564[k214];	// L6644
    int8_t a214;	// L6645
    a214 = v5573;	// L6646
    int8_t v5575 = v5565.read(); // v5565[k214];	// L6647
    int8_t b214;	// L6648
    b214 = v5575;	// L6649
    int8_t v5577 = a214;	// L6650
    int8_t v5578 = b214;	// L6651
    int16_t v5579 = v5577;	// L6652
    int16_t v5580 = v5578;	// L6653
    int16_t v5581 = v5579 * v5580;	// L6654
    int32_t v5582 = v214;	// L6655
    ap_int<33> v5583 = v5582;	// L6656
    ap_int<33> v5584 = v5581;	// L6657
    ap_int<33> v5585 = v5583 + v5584;	// L6658
    int32_t v5586 = v5585;	// L6659
    v214 = v5586;	// L6660
    int8_t v5587 = a214;	// L6661
    v5566.write(v5587); // v5566[k214] = v5587;	// L6662
    int8_t v5588 = b214;	// L6663
    v5567.write(v5588); // v5567[k214] = v5588;	// L6664
  }
  int32_t v5589 = v214;	// L6666
  v5568[v5569][v5570] = v5589;	// L6667
}

void PE_kernel_gemm_7_13(
  hls::stream< int8_t > &v5590 /* v5590[16] */,
  hls::stream< int8_t > &v5591 /* v5591[16] */,
  hls::stream< int8_t > &v5592 /* v5592[16] */,
  hls::stream< int8_t > &v5593 /* v5593[16] */,
  int32_t v5594[16][16],
  int v5595,
  int v5596
) {	// L6670
  #pragma HLS stream variable=v5590 depth=17
  #pragma HLS stream variable=v5591 depth=17
  #pragma HLS stream variable=v5592 depth=17
  #pragma HLS stream variable=v5593 depth=17
  #pragma HLS array_partition variable=v5594 complete dim=1
  #pragma HLS array_partition variable=v5594 complete dim=2

  int32_t v215;	// L6672
  v215 = 0;	// L6673
  l_reduction_k215: for (int k215 = 0; k215 < 16; k215++) {	// L6674
  #pragma HLS pipeline II=1
    int8_t v5599 = v5590.read(); // v5590[k215];	// L6675
    int8_t a215;	// L6676
    a215 = v5599;	// L6677
    int8_t v5601 = v5591.read(); // v5591[k215];	// L6678
    int8_t b215;	// L6679
    b215 = v5601;	// L6680
    int8_t v5603 = a215;	// L6681
    int8_t v5604 = b215;	// L6682
    int16_t v5605 = v5603;	// L6683
    int16_t v5606 = v5604;	// L6684
    int16_t v5607 = v5605 * v5606;	// L6685
    int32_t v5608 = v215;	// L6686
    ap_int<33> v5609 = v5608;	// L6687
    ap_int<33> v5610 = v5607;	// L6688
    ap_int<33> v5611 = v5609 + v5610;	// L6689
    int32_t v5612 = v5611;	// L6690
    v215 = v5612;	// L6691
    int8_t v5613 = a215;	// L6692
    v5592.write(v5613); // v5592[k215] = v5613;	// L6693
    int8_t v5614 = b215;	// L6694
    v5593.write(v5614); // v5593[k215] = v5614;	// L6695
  }
  int32_t v5615 = v215;	// L6697
  v5594[v5595][v5596] = v5615;	// L6698
}

void PE_kernel_gemm_8_13(
  hls::stream< int8_t > &v5616 /* v5616[16] */,
  hls::stream< int8_t > &v5617 /* v5617[16] */,
  hls::stream< int8_t > &v5618 /* v5618[16] */,
  hls::stream< int8_t > &v5619 /* v5619[16] */,
  int32_t v5620[16][16],
  int v5621,
  int v5622
) {	// L6701
  #pragma HLS stream variable=v5616 depth=17
  #pragma HLS stream variable=v5617 depth=17
  #pragma HLS stream variable=v5618 depth=17
  #pragma HLS stream variable=v5619 depth=17
  #pragma HLS array_partition variable=v5620 complete dim=1
  #pragma HLS array_partition variable=v5620 complete dim=2

  int32_t v216;	// L6703
  v216 = 0;	// L6704
  l_reduction_k216: for (int k216 = 0; k216 < 16; k216++) {	// L6705
  #pragma HLS pipeline II=1
    int8_t v5625 = v5616.read(); // v5616[k216];	// L6706
    int8_t a216;	// L6707
    a216 = v5625;	// L6708
    int8_t v5627 = v5617.read(); // v5617[k216];	// L6709
    int8_t b216;	// L6710
    b216 = v5627;	// L6711
    int8_t v5629 = a216;	// L6712
    int8_t v5630 = b216;	// L6713
    int16_t v5631 = v5629;	// L6714
    int16_t v5632 = v5630;	// L6715
    int16_t v5633 = v5631 * v5632;	// L6716
    int32_t v5634 = v216;	// L6717
    ap_int<33> v5635 = v5634;	// L6718
    ap_int<33> v5636 = v5633;	// L6719
    ap_int<33> v5637 = v5635 + v5636;	// L6720
    int32_t v5638 = v5637;	// L6721
    v216 = v5638;	// L6722
    int8_t v5639 = a216;	// L6723
    v5618.write(v5639); // v5618[k216] = v5639;	// L6724
    int8_t v5640 = b216;	// L6725
    v5619.write(v5640); // v5619[k216] = v5640;	// L6726
  }
  int32_t v5641 = v216;	// L6728
  v5620[v5621][v5622] = v5641;	// L6729
}

void PE_kernel_gemm_9_13(
  hls::stream< int8_t > &v5642 /* v5642[16] */,
  hls::stream< int8_t > &v5643 /* v5643[16] */,
  hls::stream< int8_t > &v5644 /* v5644[16] */,
  hls::stream< int8_t > &v5645 /* v5645[16] */,
  int32_t v5646[16][16],
  int v5647,
  int v5648
) {	// L6732
  #pragma HLS stream variable=v5642 depth=17
  #pragma HLS stream variable=v5643 depth=17
  #pragma HLS stream variable=v5644 depth=17
  #pragma HLS stream variable=v5645 depth=17
  #pragma HLS array_partition variable=v5646 complete dim=1
  #pragma HLS array_partition variable=v5646 complete dim=2

  int32_t v217;	// L6734
  v217 = 0;	// L6735
  l_reduction_k217: for (int k217 = 0; k217 < 16; k217++) {	// L6736
  #pragma HLS pipeline II=1
    int8_t v5651 = v5642.read(); // v5642[k217];	// L6737
    int8_t a217;	// L6738
    a217 = v5651;	// L6739
    int8_t v5653 = v5643.read(); // v5643[k217];	// L6740
    int8_t b217;	// L6741
    b217 = v5653;	// L6742
    int8_t v5655 = a217;	// L6743
    int8_t v5656 = b217;	// L6744
    int16_t v5657 = v5655;	// L6745
    int16_t v5658 = v5656;	// L6746
    int16_t v5659 = v5657 * v5658;	// L6747
    int32_t v5660 = v217;	// L6748
    ap_int<33> v5661 = v5660;	// L6749
    ap_int<33> v5662 = v5659;	// L6750
    ap_int<33> v5663 = v5661 + v5662;	// L6751
    int32_t v5664 = v5663;	// L6752
    v217 = v5664;	// L6753
    int8_t v5665 = a217;	// L6754
    v5644.write(v5665); // v5644[k217] = v5665;	// L6755
    int8_t v5666 = b217;	// L6756
    v5645.write(v5666); // v5645[k217] = v5666;	// L6757
  }
  int32_t v5667 = v217;	// L6759
  v5646[v5647][v5648] = v5667;	// L6760
}

void PE_kernel_gemm_10_13(
  hls::stream< int8_t > &v5668 /* v5668[16] */,
  hls::stream< int8_t > &v5669 /* v5669[16] */,
  hls::stream< int8_t > &v5670 /* v5670[16] */,
  hls::stream< int8_t > &v5671 /* v5671[16] */,
  int32_t v5672[16][16],
  int v5673,
  int v5674
) {	// L6763
  #pragma HLS stream variable=v5668 depth=17
  #pragma HLS stream variable=v5669 depth=17
  #pragma HLS stream variable=v5670 depth=17
  #pragma HLS stream variable=v5671 depth=17
  #pragma HLS array_partition variable=v5672 complete dim=1
  #pragma HLS array_partition variable=v5672 complete dim=2

  int32_t v218;	// L6765
  v218 = 0;	// L6766
  l_reduction_k218: for (int k218 = 0; k218 < 16; k218++) {	// L6767
  #pragma HLS pipeline II=1
    int8_t v5677 = v5668.read(); // v5668[k218];	// L6768
    int8_t a218;	// L6769
    a218 = v5677;	// L6770
    int8_t v5679 = v5669.read(); // v5669[k218];	// L6771
    int8_t b218;	// L6772
    b218 = v5679;	// L6773
    int8_t v5681 = a218;	// L6774
    int8_t v5682 = b218;	// L6775
    int16_t v5683 = v5681;	// L6776
    int16_t v5684 = v5682;	// L6777
    int16_t v5685 = v5683 * v5684;	// L6778
    int32_t v5686 = v218;	// L6779
    ap_int<33> v5687 = v5686;	// L6780
    ap_int<33> v5688 = v5685;	// L6781
    ap_int<33> v5689 = v5687 + v5688;	// L6782
    int32_t v5690 = v5689;	// L6783
    v218 = v5690;	// L6784
    int8_t v5691 = a218;	// L6785
    v5670.write(v5691); // v5670[k218] = v5691;	// L6786
    int8_t v5692 = b218;	// L6787
    v5671.write(v5692); // v5671[k218] = v5692;	// L6788
  }
  int32_t v5693 = v218;	// L6790
  v5672[v5673][v5674] = v5693;	// L6791
}

void PE_kernel_gemm_11_13(
  hls::stream< int8_t > &v5694 /* v5694[16] */,
  hls::stream< int8_t > &v5695 /* v5695[16] */,
  hls::stream< int8_t > &v5696 /* v5696[16] */,
  hls::stream< int8_t > &v5697 /* v5697[16] */,
  int32_t v5698[16][16],
  int v5699,
  int v5700
) {	// L6794
  #pragma HLS stream variable=v5694 depth=17
  #pragma HLS stream variable=v5695 depth=17
  #pragma HLS stream variable=v5696 depth=17
  #pragma HLS stream variable=v5697 depth=17
  #pragma HLS array_partition variable=v5698 complete dim=1
  #pragma HLS array_partition variable=v5698 complete dim=2

  int32_t v219;	// L6796
  v219 = 0;	// L6797
  l_reduction_k219: for (int k219 = 0; k219 < 16; k219++) {	// L6798
  #pragma HLS pipeline II=1
    int8_t v5703 = v5694.read(); // v5694[k219];	// L6799
    int8_t a219;	// L6800
    a219 = v5703;	// L6801
    int8_t v5705 = v5695.read(); // v5695[k219];	// L6802
    int8_t b219;	// L6803
    b219 = v5705;	// L6804
    int8_t v5707 = a219;	// L6805
    int8_t v5708 = b219;	// L6806
    int16_t v5709 = v5707;	// L6807
    int16_t v5710 = v5708;	// L6808
    int16_t v5711 = v5709 * v5710;	// L6809
    int32_t v5712 = v219;	// L6810
    ap_int<33> v5713 = v5712;	// L6811
    ap_int<33> v5714 = v5711;	// L6812
    ap_int<33> v5715 = v5713 + v5714;	// L6813
    int32_t v5716 = v5715;	// L6814
    v219 = v5716;	// L6815
    int8_t v5717 = a219;	// L6816
    v5696.write(v5717); // v5696[k219] = v5717;	// L6817
    int8_t v5718 = b219;	// L6818
    v5697.write(v5718); // v5697[k219] = v5718;	// L6819
  }
  int32_t v5719 = v219;	// L6821
  v5698[v5699][v5700] = v5719;	// L6822
}

void PE_kernel_gemm_12_13(
  hls::stream< int8_t > &v5720 /* v5720[16] */,
  hls::stream< int8_t > &v5721 /* v5721[16] */,
  hls::stream< int8_t > &v5722 /* v5722[16] */,
  hls::stream< int8_t > &v5723 /* v5723[16] */,
  int32_t v5724[16][16],
  int v5725,
  int v5726
) {	// L6825
  #pragma HLS stream variable=v5720 depth=17
  #pragma HLS stream variable=v5721 depth=17
  #pragma HLS stream variable=v5722 depth=17
  #pragma HLS stream variable=v5723 depth=17
  #pragma HLS array_partition variable=v5724 complete dim=1
  #pragma HLS array_partition variable=v5724 complete dim=2

  int32_t v220;	// L6827
  v220 = 0;	// L6828
  l_reduction_k220: for (int k220 = 0; k220 < 16; k220++) {	// L6829
  #pragma HLS pipeline II=1
    int8_t v5729 = v5720.read(); // v5720[k220];	// L6830
    int8_t a220;	// L6831
    a220 = v5729;	// L6832
    int8_t v5731 = v5721.read(); // v5721[k220];	// L6833
    int8_t b220;	// L6834
    b220 = v5731;	// L6835
    int8_t v5733 = a220;	// L6836
    int8_t v5734 = b220;	// L6837
    int16_t v5735 = v5733;	// L6838
    int16_t v5736 = v5734;	// L6839
    int16_t v5737 = v5735 * v5736;	// L6840
    int32_t v5738 = v220;	// L6841
    ap_int<33> v5739 = v5738;	// L6842
    ap_int<33> v5740 = v5737;	// L6843
    ap_int<33> v5741 = v5739 + v5740;	// L6844
    int32_t v5742 = v5741;	// L6845
    v220 = v5742;	// L6846
    int8_t v5743 = a220;	// L6847
    v5722.write(v5743); // v5722[k220] = v5743;	// L6848
    int8_t v5744 = b220;	// L6849
    v5723.write(v5744); // v5723[k220] = v5744;	// L6850
  }
  int32_t v5745 = v220;	// L6852
  v5724[v5725][v5726] = v5745;	// L6853
}

void PE_kernel_gemm_13_13(
  hls::stream< int8_t > &v5746 /* v5746[16] */,
  hls::stream< int8_t > &v5747 /* v5747[16] */,
  hls::stream< int8_t > &v5748 /* v5748[16] */,
  hls::stream< int8_t > &v5749 /* v5749[16] */,
  int32_t v5750[16][16],
  int v5751,
  int v5752
) {	// L6856
  #pragma HLS stream variable=v5746 depth=17
  #pragma HLS stream variable=v5747 depth=17
  #pragma HLS stream variable=v5748 depth=17
  #pragma HLS stream variable=v5749 depth=17
  #pragma HLS array_partition variable=v5750 complete dim=1
  #pragma HLS array_partition variable=v5750 complete dim=2

  int32_t v221;	// L6858
  v221 = 0;	// L6859
  l_reduction_k221: for (int k221 = 0; k221 < 16; k221++) {	// L6860
  #pragma HLS pipeline II=1
    int8_t v5755 = v5746.read(); // v5746[k221];	// L6861
    int8_t a221;	// L6862
    a221 = v5755;	// L6863
    int8_t v5757 = v5747.read(); // v5747[k221];	// L6864
    int8_t b221;	// L6865
    b221 = v5757;	// L6866
    int8_t v5759 = a221;	// L6867
    int8_t v5760 = b221;	// L6868
    int16_t v5761 = v5759;	// L6869
    int16_t v5762 = v5760;	// L6870
    int16_t v5763 = v5761 * v5762;	// L6871
    int32_t v5764 = v221;	// L6872
    ap_int<33> v5765 = v5764;	// L6873
    ap_int<33> v5766 = v5763;	// L6874
    ap_int<33> v5767 = v5765 + v5766;	// L6875
    int32_t v5768 = v5767;	// L6876
    v221 = v5768;	// L6877
    int8_t v5769 = a221;	// L6878
    v5748.write(v5769); // v5748[k221] = v5769;	// L6879
    int8_t v5770 = b221;	// L6880
    v5749.write(v5770); // v5749[k221] = v5770;	// L6881
  }
  int32_t v5771 = v221;	// L6883
  v5750[v5751][v5752] = v5771;	// L6884
}

void PE_kernel_gemm_14_13(
  hls::stream< int8_t > &v5772 /* v5772[16] */,
  hls::stream< int8_t > &v5773 /* v5773[16] */,
  hls::stream< int8_t > &v5774 /* v5774[16] */,
  hls::stream< int8_t > &v5775 /* v5775[16] */,
  int32_t v5776[16][16],
  int v5777,
  int v5778
) {	// L6887
  #pragma HLS stream variable=v5772 depth=17
  #pragma HLS stream variable=v5773 depth=17
  #pragma HLS stream variable=v5774 depth=17
  #pragma HLS stream variable=v5775 depth=17
  #pragma HLS array_partition variable=v5776 complete dim=1
  #pragma HLS array_partition variable=v5776 complete dim=2

  int32_t v222;	// L6889
  v222 = 0;	// L6890
  l_reduction_k222: for (int k222 = 0; k222 < 16; k222++) {	// L6891
  #pragma HLS pipeline II=1
    int8_t v5781 = v5772.read(); // v5772[k222];	// L6892
    int8_t a222;	// L6893
    a222 = v5781;	// L6894
    int8_t v5783 = v5773.read(); // v5773[k222];	// L6895
    int8_t b222;	// L6896
    b222 = v5783;	// L6897
    int8_t v5785 = a222;	// L6898
    int8_t v5786 = b222;	// L6899
    int16_t v5787 = v5785;	// L6900
    int16_t v5788 = v5786;	// L6901
    int16_t v5789 = v5787 * v5788;	// L6902
    int32_t v5790 = v222;	// L6903
    ap_int<33> v5791 = v5790;	// L6904
    ap_int<33> v5792 = v5789;	// L6905
    ap_int<33> v5793 = v5791 + v5792;	// L6906
    int32_t v5794 = v5793;	// L6907
    v222 = v5794;	// L6908
    int8_t v5795 = a222;	// L6909
    v5774.write(v5795); // v5774[k222] = v5795;	// L6910
    int8_t v5796 = b222;	// L6911
    v5775.write(v5796); // v5775[k222] = v5796;	// L6912
  }
  int32_t v5797 = v222;	// L6914
  v5776[v5777][v5778] = v5797;	// L6915
}

void PE_kernel_gemm_15_13(
  hls::stream< int8_t > &v5798 /* v5798[16] */,
  hls::stream< int8_t > &v5799 /* v5799[16] */,
  hls::stream< int8_t > &v5800 /* v5800[16] */,
  hls::stream< int8_t > &v5801 /* v5801[16] */,
  int32_t v5802[16][16],
  int v5803,
  int v5804
) {	// L6918
  #pragma HLS stream variable=v5798 depth=17
  #pragma HLS stream variable=v5799 depth=17
  #pragma HLS stream variable=v5800 depth=17
  #pragma HLS stream variable=v5801 depth=17
  #pragma HLS array_partition variable=v5802 complete dim=1
  #pragma HLS array_partition variable=v5802 complete dim=2

  int32_t v223;	// L6920
  v223 = 0;	// L6921
  l_reduction_k223: for (int k223 = 0; k223 < 16; k223++) {	// L6922
  #pragma HLS pipeline II=1
    int8_t v5807 = v5798.read(); // v5798[k223];	// L6923
    int8_t a223;	// L6924
    a223 = v5807;	// L6925
    int8_t v5809 = v5799.read(); // v5799[k223];	// L6926
    int8_t b223;	// L6927
    b223 = v5809;	// L6928
    int8_t v5811 = a223;	// L6929
    int8_t v5812 = b223;	// L6930
    int16_t v5813 = v5811;	// L6931
    int16_t v5814 = v5812;	// L6932
    int16_t v5815 = v5813 * v5814;	// L6933
    int32_t v5816 = v223;	// L6934
    ap_int<33> v5817 = v5816;	// L6935
    ap_int<33> v5818 = v5815;	// L6936
    ap_int<33> v5819 = v5817 + v5818;	// L6937
    int32_t v5820 = v5819;	// L6938
    v223 = v5820;	// L6939
    int8_t v5821 = a223;	// L6940
    v5800.write(v5821); // v5800[k223] = v5821;	// L6941
    int8_t v5822 = b223;	// L6942
    v5801.write(v5822); // v5801[k223] = v5822;	// L6943
  }
  int32_t v5823 = v223;	// L6945
  v5802[v5803][v5804] = v5823;	// L6946
}

void PE_kernel_gemm_0_14(
  hls::stream< int8_t > &v5824 /* v5824[16] */,
  hls::stream< int8_t > &v5825 /* v5825[16] */,
  hls::stream< int8_t > &v5826 /* v5826[16] */,
  hls::stream< int8_t > &v5827 /* v5827[16] */,
  int32_t v5828[16][16],
  int v5829,
  int v5830
) {	// L6949
  #pragma HLS stream variable=v5824 depth=17
  #pragma HLS stream variable=v5825 depth=17
  #pragma HLS stream variable=v5826 depth=17
  #pragma HLS stream variable=v5827 depth=17
  #pragma HLS array_partition variable=v5828 complete dim=1
  #pragma HLS array_partition variable=v5828 complete dim=2

  int32_t v224;	// L6951
  v224 = 0;	// L6952
  l_reduction_k224: for (int k224 = 0; k224 < 16; k224++) {	// L6953
  #pragma HLS pipeline II=1
    int8_t v5833 = v5824.read(); // v5824[k224];	// L6954
    int8_t a224;	// L6955
    a224 = v5833;	// L6956
    int8_t v5835 = v5825.read(); // v5825[k224];	// L6957
    int8_t b224;	// L6958
    b224 = v5835;	// L6959
    int8_t v5837 = a224;	// L6960
    int8_t v5838 = b224;	// L6961
    int16_t v5839 = v5837;	// L6962
    int16_t v5840 = v5838;	// L6963
    int16_t v5841 = v5839 * v5840;	// L6964
    int32_t v5842 = v224;	// L6965
    ap_int<33> v5843 = v5842;	// L6966
    ap_int<33> v5844 = v5841;	// L6967
    ap_int<33> v5845 = v5843 + v5844;	// L6968
    int32_t v5846 = v5845;	// L6969
    v224 = v5846;	// L6970
    int8_t v5847 = a224;	// L6971
    v5826.write(v5847); // v5826[k224] = v5847;	// L6972
    int8_t v5848 = b224;	// L6973
    v5827.write(v5848); // v5827[k224] = v5848;	// L6974
  }
  int32_t v5849 = v224;	// L6976
  v5828[v5829][v5830] = v5849;	// L6977
}

void PE_kernel_gemm_1_14(
  hls::stream< int8_t > &v5850 /* v5850[16] */,
  hls::stream< int8_t > &v5851 /* v5851[16] */,
  hls::stream< int8_t > &v5852 /* v5852[16] */,
  hls::stream< int8_t > &v5853 /* v5853[16] */,
  int32_t v5854[16][16],
  int v5855,
  int v5856
) {	// L6980
  #pragma HLS stream variable=v5850 depth=17
  #pragma HLS stream variable=v5851 depth=17
  #pragma HLS stream variable=v5852 depth=17
  #pragma HLS stream variable=v5853 depth=17
  #pragma HLS array_partition variable=v5854 complete dim=1
  #pragma HLS array_partition variable=v5854 complete dim=2

  int32_t v225;	// L6982
  v225 = 0;	// L6983
  l_reduction_k225: for (int k225 = 0; k225 < 16; k225++) {	// L6984
  #pragma HLS pipeline II=1
    int8_t v5859 = v5850.read(); // v5850[k225];	// L6985
    int8_t a225;	// L6986
    a225 = v5859;	// L6987
    int8_t v5861 = v5851.read(); // v5851[k225];	// L6988
    int8_t b225;	// L6989
    b225 = v5861;	// L6990
    int8_t v5863 = a225;	// L6991
    int8_t v5864 = b225;	// L6992
    int16_t v5865 = v5863;	// L6993
    int16_t v5866 = v5864;	// L6994
    int16_t v5867 = v5865 * v5866;	// L6995
    int32_t v5868 = v225;	// L6996
    ap_int<33> v5869 = v5868;	// L6997
    ap_int<33> v5870 = v5867;	// L6998
    ap_int<33> v5871 = v5869 + v5870;	// L6999
    int32_t v5872 = v5871;	// L7000
    v225 = v5872;	// L7001
    int8_t v5873 = a225;	// L7002
    v5852.write(v5873); // v5852[k225] = v5873;	// L7003
    int8_t v5874 = b225;	// L7004
    v5853.write(v5874); // v5853[k225] = v5874;	// L7005
  }
  int32_t v5875 = v225;	// L7007
  v5854[v5855][v5856] = v5875;	// L7008
}

void PE_kernel_gemm_2_14(
  hls::stream< int8_t > &v5876 /* v5876[16] */,
  hls::stream< int8_t > &v5877 /* v5877[16] */,
  hls::stream< int8_t > &v5878 /* v5878[16] */,
  hls::stream< int8_t > &v5879 /* v5879[16] */,
  int32_t v5880[16][16],
  int v5881,
  int v5882
) {	// L7011
  #pragma HLS stream variable=v5876 depth=17
  #pragma HLS stream variable=v5877 depth=17
  #pragma HLS stream variable=v5878 depth=17
  #pragma HLS stream variable=v5879 depth=17
  #pragma HLS array_partition variable=v5880 complete dim=1
  #pragma HLS array_partition variable=v5880 complete dim=2

  int32_t v226;	// L7013
  v226 = 0;	// L7014
  l_reduction_k226: for (int k226 = 0; k226 < 16; k226++) {	// L7015
  #pragma HLS pipeline II=1
    int8_t v5885 = v5876.read(); // v5876[k226];	// L7016
    int8_t a226;	// L7017
    a226 = v5885;	// L7018
    int8_t v5887 = v5877.read(); // v5877[k226];	// L7019
    int8_t b226;	// L7020
    b226 = v5887;	// L7021
    int8_t v5889 = a226;	// L7022
    int8_t v5890 = b226;	// L7023
    int16_t v5891 = v5889;	// L7024
    int16_t v5892 = v5890;	// L7025
    int16_t v5893 = v5891 * v5892;	// L7026
    int32_t v5894 = v226;	// L7027
    ap_int<33> v5895 = v5894;	// L7028
    ap_int<33> v5896 = v5893;	// L7029
    ap_int<33> v5897 = v5895 + v5896;	// L7030
    int32_t v5898 = v5897;	// L7031
    v226 = v5898;	// L7032
    int8_t v5899 = a226;	// L7033
    v5878.write(v5899); // v5878[k226] = v5899;	// L7034
    int8_t v5900 = b226;	// L7035
    v5879.write(v5900); // v5879[k226] = v5900;	// L7036
  }
  int32_t v5901 = v226;	// L7038
  v5880[v5881][v5882] = v5901;	// L7039
}

void PE_kernel_gemm_3_14(
  hls::stream< int8_t > &v5902 /* v5902[16] */,
  hls::stream< int8_t > &v5903 /* v5903[16] */,
  hls::stream< int8_t > &v5904 /* v5904[16] */,
  hls::stream< int8_t > &v5905 /* v5905[16] */,
  int32_t v5906[16][16],
  int v5907,
  int v5908
) {	// L7042
  #pragma HLS stream variable=v5902 depth=17
  #pragma HLS stream variable=v5903 depth=17
  #pragma HLS stream variable=v5904 depth=17
  #pragma HLS stream variable=v5905 depth=17
  #pragma HLS array_partition variable=v5906 complete dim=1
  #pragma HLS array_partition variable=v5906 complete dim=2

  int32_t v227;	// L7044
  v227 = 0;	// L7045
  l_reduction_k227: for (int k227 = 0; k227 < 16; k227++) {	// L7046
  #pragma HLS pipeline II=1
    int8_t v5911 = v5902.read(); // v5902[k227];	// L7047
    int8_t a227;	// L7048
    a227 = v5911;	// L7049
    int8_t v5913 = v5903.read(); // v5903[k227];	// L7050
    int8_t b227;	// L7051
    b227 = v5913;	// L7052
    int8_t v5915 = a227;	// L7053
    int8_t v5916 = b227;	// L7054
    int16_t v5917 = v5915;	// L7055
    int16_t v5918 = v5916;	// L7056
    int16_t v5919 = v5917 * v5918;	// L7057
    int32_t v5920 = v227;	// L7058
    ap_int<33> v5921 = v5920;	// L7059
    ap_int<33> v5922 = v5919;	// L7060
    ap_int<33> v5923 = v5921 + v5922;	// L7061
    int32_t v5924 = v5923;	// L7062
    v227 = v5924;	// L7063
    int8_t v5925 = a227;	// L7064
    v5904.write(v5925); // v5904[k227] = v5925;	// L7065
    int8_t v5926 = b227;	// L7066
    v5905.write(v5926); // v5905[k227] = v5926;	// L7067
  }
  int32_t v5927 = v227;	// L7069
  v5906[v5907][v5908] = v5927;	// L7070
}

void PE_kernel_gemm_4_14(
  hls::stream< int8_t > &v5928 /* v5928[16] */,
  hls::stream< int8_t > &v5929 /* v5929[16] */,
  hls::stream< int8_t > &v5930 /* v5930[16] */,
  hls::stream< int8_t > &v5931 /* v5931[16] */,
  int32_t v5932[16][16],
  int v5933,
  int v5934
) {	// L7073
  #pragma HLS stream variable=v5928 depth=17
  #pragma HLS stream variable=v5929 depth=17
  #pragma HLS stream variable=v5930 depth=17
  #pragma HLS stream variable=v5931 depth=17
  #pragma HLS array_partition variable=v5932 complete dim=1
  #pragma HLS array_partition variable=v5932 complete dim=2

  int32_t v228;	// L7075
  v228 = 0;	// L7076
  l_reduction_k228: for (int k228 = 0; k228 < 16; k228++) {	// L7077
  #pragma HLS pipeline II=1
    int8_t v5937 = v5928.read(); // v5928[k228];	// L7078
    int8_t a228;	// L7079
    a228 = v5937;	// L7080
    int8_t v5939 = v5929.read(); // v5929[k228];	// L7081
    int8_t b228;	// L7082
    b228 = v5939;	// L7083
    int8_t v5941 = a228;	// L7084
    int8_t v5942 = b228;	// L7085
    int16_t v5943 = v5941;	// L7086
    int16_t v5944 = v5942;	// L7087
    int16_t v5945 = v5943 * v5944;	// L7088
    int32_t v5946 = v228;	// L7089
    ap_int<33> v5947 = v5946;	// L7090
    ap_int<33> v5948 = v5945;	// L7091
    ap_int<33> v5949 = v5947 + v5948;	// L7092
    int32_t v5950 = v5949;	// L7093
    v228 = v5950;	// L7094
    int8_t v5951 = a228;	// L7095
    v5930.write(v5951); // v5930[k228] = v5951;	// L7096
    int8_t v5952 = b228;	// L7097
    v5931.write(v5952); // v5931[k228] = v5952;	// L7098
  }
  int32_t v5953 = v228;	// L7100
  v5932[v5933][v5934] = v5953;	// L7101
}

void PE_kernel_gemm_5_14(
  hls::stream< int8_t > &v5954 /* v5954[16] */,
  hls::stream< int8_t > &v5955 /* v5955[16] */,
  hls::stream< int8_t > &v5956 /* v5956[16] */,
  hls::stream< int8_t > &v5957 /* v5957[16] */,
  int32_t v5958[16][16],
  int v5959,
  int v5960
) {	// L7104
  #pragma HLS stream variable=v5954 depth=17
  #pragma HLS stream variable=v5955 depth=17
  #pragma HLS stream variable=v5956 depth=17
  #pragma HLS stream variable=v5957 depth=17
  #pragma HLS array_partition variable=v5958 complete dim=1
  #pragma HLS array_partition variable=v5958 complete dim=2

  int32_t v229;	// L7106
  v229 = 0;	// L7107
  l_reduction_k229: for (int k229 = 0; k229 < 16; k229++) {	// L7108
  #pragma HLS pipeline II=1
    int8_t v5963 = v5954.read(); // v5954[k229];	// L7109
    int8_t a229;	// L7110
    a229 = v5963;	// L7111
    int8_t v5965 = v5955.read(); // v5955[k229];	// L7112
    int8_t b229;	// L7113
    b229 = v5965;	// L7114
    int8_t v5967 = a229;	// L7115
    int8_t v5968 = b229;	// L7116
    int16_t v5969 = v5967;	// L7117
    int16_t v5970 = v5968;	// L7118
    int16_t v5971 = v5969 * v5970;	// L7119
    int32_t v5972 = v229;	// L7120
    ap_int<33> v5973 = v5972;	// L7121
    ap_int<33> v5974 = v5971;	// L7122
    ap_int<33> v5975 = v5973 + v5974;	// L7123
    int32_t v5976 = v5975;	// L7124
    v229 = v5976;	// L7125
    int8_t v5977 = a229;	// L7126
    v5956.write(v5977); // v5956[k229] = v5977;	// L7127
    int8_t v5978 = b229;	// L7128
    v5957.write(v5978); // v5957[k229] = v5978;	// L7129
  }
  int32_t v5979 = v229;	// L7131
  v5958[v5959][v5960] = v5979;	// L7132
}

void PE_kernel_gemm_6_14(
  hls::stream< int8_t > &v5980 /* v5980[16] */,
  hls::stream< int8_t > &v5981 /* v5981[16] */,
  hls::stream< int8_t > &v5982 /* v5982[16] */,
  hls::stream< int8_t > &v5983 /* v5983[16] */,
  int32_t v5984[16][16],
  int v5985,
  int v5986
) {	// L7135
  #pragma HLS stream variable=v5980 depth=17
  #pragma HLS stream variable=v5981 depth=17
  #pragma HLS stream variable=v5982 depth=17
  #pragma HLS stream variable=v5983 depth=17
  #pragma HLS array_partition variable=v5984 complete dim=1
  #pragma HLS array_partition variable=v5984 complete dim=2

  int32_t v230;	// L7137
  v230 = 0;	// L7138
  l_reduction_k230: for (int k230 = 0; k230 < 16; k230++) {	// L7139
  #pragma HLS pipeline II=1
    int8_t v5989 = v5980.read(); // v5980[k230];	// L7140
    int8_t a230;	// L7141
    a230 = v5989;	// L7142
    int8_t v5991 = v5981.read(); // v5981[k230];	// L7143
    int8_t b230;	// L7144
    b230 = v5991;	// L7145
    int8_t v5993 = a230;	// L7146
    int8_t v5994 = b230;	// L7147
    int16_t v5995 = v5993;	// L7148
    int16_t v5996 = v5994;	// L7149
    int16_t v5997 = v5995 * v5996;	// L7150
    int32_t v5998 = v230;	// L7151
    ap_int<33> v5999 = v5998;	// L7152
    ap_int<33> v6000 = v5997;	// L7153
    ap_int<33> v6001 = v5999 + v6000;	// L7154
    int32_t v6002 = v6001;	// L7155
    v230 = v6002;	// L7156
    int8_t v6003 = a230;	// L7157
    v5982.write(v6003); // v5982[k230] = v6003;	// L7158
    int8_t v6004 = b230;	// L7159
    v5983.write(v6004); // v5983[k230] = v6004;	// L7160
  }
  int32_t v6005 = v230;	// L7162
  v5984[v5985][v5986] = v6005;	// L7163
}

void PE_kernel_gemm_7_14(
  hls::stream< int8_t > &v6006 /* v6006[16] */,
  hls::stream< int8_t > &v6007 /* v6007[16] */,
  hls::stream< int8_t > &v6008 /* v6008[16] */,
  hls::stream< int8_t > &v6009 /* v6009[16] */,
  int32_t v6010[16][16],
  int v6011,
  int v6012
) {	// L7166
  #pragma HLS stream variable=v6006 depth=17
  #pragma HLS stream variable=v6007 depth=17
  #pragma HLS stream variable=v6008 depth=17
  #pragma HLS stream variable=v6009 depth=17
  #pragma HLS array_partition variable=v6010 complete dim=1
  #pragma HLS array_partition variable=v6010 complete dim=2

  int32_t v231;	// L7168
  v231 = 0;	// L7169
  l_reduction_k231: for (int k231 = 0; k231 < 16; k231++) {	// L7170
  #pragma HLS pipeline II=1
    int8_t v6015 = v6006.read(); // v6006[k231];	// L7171
    int8_t a231;	// L7172
    a231 = v6015;	// L7173
    int8_t v6017 = v6007.read(); // v6007[k231];	// L7174
    int8_t b231;	// L7175
    b231 = v6017;	// L7176
    int8_t v6019 = a231;	// L7177
    int8_t v6020 = b231;	// L7178
    int16_t v6021 = v6019;	// L7179
    int16_t v6022 = v6020;	// L7180
    int16_t v6023 = v6021 * v6022;	// L7181
    int32_t v6024 = v231;	// L7182
    ap_int<33> v6025 = v6024;	// L7183
    ap_int<33> v6026 = v6023;	// L7184
    ap_int<33> v6027 = v6025 + v6026;	// L7185
    int32_t v6028 = v6027;	// L7186
    v231 = v6028;	// L7187
    int8_t v6029 = a231;	// L7188
    v6008.write(v6029); // v6008[k231] = v6029;	// L7189
    int8_t v6030 = b231;	// L7190
    v6009.write(v6030); // v6009[k231] = v6030;	// L7191
  }
  int32_t v6031 = v231;	// L7193
  v6010[v6011][v6012] = v6031;	// L7194
}

void PE_kernel_gemm_8_14(
  hls::stream< int8_t > &v6032 /* v6032[16] */,
  hls::stream< int8_t > &v6033 /* v6033[16] */,
  hls::stream< int8_t > &v6034 /* v6034[16] */,
  hls::stream< int8_t > &v6035 /* v6035[16] */,
  int32_t v6036[16][16],
  int v6037,
  int v6038
) {	// L7197
  #pragma HLS stream variable=v6032 depth=17
  #pragma HLS stream variable=v6033 depth=17
  #pragma HLS stream variable=v6034 depth=17
  #pragma HLS stream variable=v6035 depth=17
  #pragma HLS array_partition variable=v6036 complete dim=1
  #pragma HLS array_partition variable=v6036 complete dim=2

  int32_t v232;	// L7199
  v232 = 0;	// L7200
  l_reduction_k232: for (int k232 = 0; k232 < 16; k232++) {	// L7201
  #pragma HLS pipeline II=1
    int8_t v6041 = v6032.read(); // v6032[k232];	// L7202
    int8_t a232;	// L7203
    a232 = v6041;	// L7204
    int8_t v6043 = v6033.read(); // v6033[k232];	// L7205
    int8_t b232;	// L7206
    b232 = v6043;	// L7207
    int8_t v6045 = a232;	// L7208
    int8_t v6046 = b232;	// L7209
    int16_t v6047 = v6045;	// L7210
    int16_t v6048 = v6046;	// L7211
    int16_t v6049 = v6047 * v6048;	// L7212
    int32_t v6050 = v232;	// L7213
    ap_int<33> v6051 = v6050;	// L7214
    ap_int<33> v6052 = v6049;	// L7215
    ap_int<33> v6053 = v6051 + v6052;	// L7216
    int32_t v6054 = v6053;	// L7217
    v232 = v6054;	// L7218
    int8_t v6055 = a232;	// L7219
    v6034.write(v6055); // v6034[k232] = v6055;	// L7220
    int8_t v6056 = b232;	// L7221
    v6035.write(v6056); // v6035[k232] = v6056;	// L7222
  }
  int32_t v6057 = v232;	// L7224
  v6036[v6037][v6038] = v6057;	// L7225
}

void PE_kernel_gemm_9_14(
  hls::stream< int8_t > &v6058 /* v6058[16] */,
  hls::stream< int8_t > &v6059 /* v6059[16] */,
  hls::stream< int8_t > &v6060 /* v6060[16] */,
  hls::stream< int8_t > &v6061 /* v6061[16] */,
  int32_t v6062[16][16],
  int v6063,
  int v6064
) {	// L7228
  #pragma HLS stream variable=v6058 depth=17
  #pragma HLS stream variable=v6059 depth=17
  #pragma HLS stream variable=v6060 depth=17
  #pragma HLS stream variable=v6061 depth=17
  #pragma HLS array_partition variable=v6062 complete dim=1
  #pragma HLS array_partition variable=v6062 complete dim=2

  int32_t v233;	// L7230
  v233 = 0;	// L7231
  l_reduction_k233: for (int k233 = 0; k233 < 16; k233++) {	// L7232
  #pragma HLS pipeline II=1
    int8_t v6067 = v6058.read(); // v6058[k233];	// L7233
    int8_t a233;	// L7234
    a233 = v6067;	// L7235
    int8_t v6069 = v6059.read(); // v6059[k233];	// L7236
    int8_t b233;	// L7237
    b233 = v6069;	// L7238
    int8_t v6071 = a233;	// L7239
    int8_t v6072 = b233;	// L7240
    int16_t v6073 = v6071;	// L7241
    int16_t v6074 = v6072;	// L7242
    int16_t v6075 = v6073 * v6074;	// L7243
    int32_t v6076 = v233;	// L7244
    ap_int<33> v6077 = v6076;	// L7245
    ap_int<33> v6078 = v6075;	// L7246
    ap_int<33> v6079 = v6077 + v6078;	// L7247
    int32_t v6080 = v6079;	// L7248
    v233 = v6080;	// L7249
    int8_t v6081 = a233;	// L7250
    v6060.write(v6081); // v6060[k233] = v6081;	// L7251
    int8_t v6082 = b233;	// L7252
    v6061.write(v6082); // v6061[k233] = v6082;	// L7253
  }
  int32_t v6083 = v233;	// L7255
  v6062[v6063][v6064] = v6083;	// L7256
}

void PE_kernel_gemm_10_14(
  hls::stream< int8_t > &v6084 /* v6084[16] */,
  hls::stream< int8_t > &v6085 /* v6085[16] */,
  hls::stream< int8_t > &v6086 /* v6086[16] */,
  hls::stream< int8_t > &v6087 /* v6087[16] */,
  int32_t v6088[16][16],
  int v6089,
  int v6090
) {	// L7259
  #pragma HLS stream variable=v6084 depth=17
  #pragma HLS stream variable=v6085 depth=17
  #pragma HLS stream variable=v6086 depth=17
  #pragma HLS stream variable=v6087 depth=17
  #pragma HLS array_partition variable=v6088 complete dim=1
  #pragma HLS array_partition variable=v6088 complete dim=2

  int32_t v234;	// L7261
  v234 = 0;	// L7262
  l_reduction_k234: for (int k234 = 0; k234 < 16; k234++) {	// L7263
  #pragma HLS pipeline II=1
    int8_t v6093 = v6084.read(); // v6084[k234];	// L7264
    int8_t a234;	// L7265
    a234 = v6093;	// L7266
    int8_t v6095 = v6085.read(); // v6085[k234];	// L7267
    int8_t b234;	// L7268
    b234 = v6095;	// L7269
    int8_t v6097 = a234;	// L7270
    int8_t v6098 = b234;	// L7271
    int16_t v6099 = v6097;	// L7272
    int16_t v6100 = v6098;	// L7273
    int16_t v6101 = v6099 * v6100;	// L7274
    int32_t v6102 = v234;	// L7275
    ap_int<33> v6103 = v6102;	// L7276
    ap_int<33> v6104 = v6101;	// L7277
    ap_int<33> v6105 = v6103 + v6104;	// L7278
    int32_t v6106 = v6105;	// L7279
    v234 = v6106;	// L7280
    int8_t v6107 = a234;	// L7281
    v6086.write(v6107); // v6086[k234] = v6107;	// L7282
    int8_t v6108 = b234;	// L7283
    v6087.write(v6108); // v6087[k234] = v6108;	// L7284
  }
  int32_t v6109 = v234;	// L7286
  v6088[v6089][v6090] = v6109;	// L7287
}

void PE_kernel_gemm_11_14(
  hls::stream< int8_t > &v6110 /* v6110[16] */,
  hls::stream< int8_t > &v6111 /* v6111[16] */,
  hls::stream< int8_t > &v6112 /* v6112[16] */,
  hls::stream< int8_t > &v6113 /* v6113[16] */,
  int32_t v6114[16][16],
  int v6115,
  int v6116
) {	// L7290
  #pragma HLS stream variable=v6110 depth=17
  #pragma HLS stream variable=v6111 depth=17
  #pragma HLS stream variable=v6112 depth=17
  #pragma HLS stream variable=v6113 depth=17
  #pragma HLS array_partition variable=v6114 complete dim=1
  #pragma HLS array_partition variable=v6114 complete dim=2

  int32_t v235;	// L7292
  v235 = 0;	// L7293
  l_reduction_k235: for (int k235 = 0; k235 < 16; k235++) {	// L7294
  #pragma HLS pipeline II=1
    int8_t v6119 = v6110.read(); // v6110[k235];	// L7295
    int8_t a235;	// L7296
    a235 = v6119;	// L7297
    int8_t v6121 = v6111.read(); // v6111[k235];	// L7298
    int8_t b235;	// L7299
    b235 = v6121;	// L7300
    int8_t v6123 = a235;	// L7301
    int8_t v6124 = b235;	// L7302
    int16_t v6125 = v6123;	// L7303
    int16_t v6126 = v6124;	// L7304
    int16_t v6127 = v6125 * v6126;	// L7305
    int32_t v6128 = v235;	// L7306
    ap_int<33> v6129 = v6128;	// L7307
    ap_int<33> v6130 = v6127;	// L7308
    ap_int<33> v6131 = v6129 + v6130;	// L7309
    int32_t v6132 = v6131;	// L7310
    v235 = v6132;	// L7311
    int8_t v6133 = a235;	// L7312
    v6112.write(v6133); // v6112[k235] = v6133;	// L7313
    int8_t v6134 = b235;	// L7314
    v6113.write(v6134); // v6113[k235] = v6134;	// L7315
  }
  int32_t v6135 = v235;	// L7317
  v6114[v6115][v6116] = v6135;	// L7318
}

void PE_kernel_gemm_12_14(
  hls::stream< int8_t > &v6136 /* v6136[16] */,
  hls::stream< int8_t > &v6137 /* v6137[16] */,
  hls::stream< int8_t > &v6138 /* v6138[16] */,
  hls::stream< int8_t > &v6139 /* v6139[16] */,
  int32_t v6140[16][16],
  int v6141,
  int v6142
) {	// L7321
  #pragma HLS stream variable=v6136 depth=17
  #pragma HLS stream variable=v6137 depth=17
  #pragma HLS stream variable=v6138 depth=17
  #pragma HLS stream variable=v6139 depth=17
  #pragma HLS array_partition variable=v6140 complete dim=1
  #pragma HLS array_partition variable=v6140 complete dim=2

  int32_t v236;	// L7323
  v236 = 0;	// L7324
  l_reduction_k236: for (int k236 = 0; k236 < 16; k236++) {	// L7325
  #pragma HLS pipeline II=1
    int8_t v6145 = v6136.read(); // v6136[k236];	// L7326
    int8_t a236;	// L7327
    a236 = v6145;	// L7328
    int8_t v6147 = v6137.read(); // v6137[k236];	// L7329
    int8_t b236;	// L7330
    b236 = v6147;	// L7331
    int8_t v6149 = a236;	// L7332
    int8_t v6150 = b236;	// L7333
    int16_t v6151 = v6149;	// L7334
    int16_t v6152 = v6150;	// L7335
    int16_t v6153 = v6151 * v6152;	// L7336
    int32_t v6154 = v236;	// L7337
    ap_int<33> v6155 = v6154;	// L7338
    ap_int<33> v6156 = v6153;	// L7339
    ap_int<33> v6157 = v6155 + v6156;	// L7340
    int32_t v6158 = v6157;	// L7341
    v236 = v6158;	// L7342
    int8_t v6159 = a236;	// L7343
    v6138.write(v6159); // v6138[k236] = v6159;	// L7344
    int8_t v6160 = b236;	// L7345
    v6139.write(v6160); // v6139[k236] = v6160;	// L7346
  }
  int32_t v6161 = v236;	// L7348
  v6140[v6141][v6142] = v6161;	// L7349
}

void PE_kernel_gemm_13_14(
  hls::stream< int8_t > &v6162 /* v6162[16] */,
  hls::stream< int8_t > &v6163 /* v6163[16] */,
  hls::stream< int8_t > &v6164 /* v6164[16] */,
  hls::stream< int8_t > &v6165 /* v6165[16] */,
  int32_t v6166[16][16],
  int v6167,
  int v6168
) {	// L7352
  #pragma HLS stream variable=v6162 depth=17
  #pragma HLS stream variable=v6163 depth=17
  #pragma HLS stream variable=v6164 depth=17
  #pragma HLS stream variable=v6165 depth=17
  #pragma HLS array_partition variable=v6166 complete dim=1
  #pragma HLS array_partition variable=v6166 complete dim=2

  int32_t v237;	// L7354
  v237 = 0;	// L7355
  l_reduction_k237: for (int k237 = 0; k237 < 16; k237++) {	// L7356
  #pragma HLS pipeline II=1
    int8_t v6171 = v6162.read(); // v6162[k237];	// L7357
    int8_t a237;	// L7358
    a237 = v6171;	// L7359
    int8_t v6173 = v6163.read(); // v6163[k237];	// L7360
    int8_t b237;	// L7361
    b237 = v6173;	// L7362
    int8_t v6175 = a237;	// L7363
    int8_t v6176 = b237;	// L7364
    int16_t v6177 = v6175;	// L7365
    int16_t v6178 = v6176;	// L7366
    int16_t v6179 = v6177 * v6178;	// L7367
    int32_t v6180 = v237;	// L7368
    ap_int<33> v6181 = v6180;	// L7369
    ap_int<33> v6182 = v6179;	// L7370
    ap_int<33> v6183 = v6181 + v6182;	// L7371
    int32_t v6184 = v6183;	// L7372
    v237 = v6184;	// L7373
    int8_t v6185 = a237;	// L7374
    v6164.write(v6185); // v6164[k237] = v6185;	// L7375
    int8_t v6186 = b237;	// L7376
    v6165.write(v6186); // v6165[k237] = v6186;	// L7377
  }
  int32_t v6187 = v237;	// L7379
  v6166[v6167][v6168] = v6187;	// L7380
}

void PE_kernel_gemm_14_14(
  hls::stream< int8_t > &v6188 /* v6188[16] */,
  hls::stream< int8_t > &v6189 /* v6189[16] */,
  hls::stream< int8_t > &v6190 /* v6190[16] */,
  hls::stream< int8_t > &v6191 /* v6191[16] */,
  int32_t v6192[16][16],
  int v6193,
  int v6194
) {	// L7383
  #pragma HLS stream variable=v6188 depth=17
  #pragma HLS stream variable=v6189 depth=17
  #pragma HLS stream variable=v6190 depth=17
  #pragma HLS stream variable=v6191 depth=17
  #pragma HLS array_partition variable=v6192 complete dim=1
  #pragma HLS array_partition variable=v6192 complete dim=2

  int32_t v238;	// L7385
  v238 = 0;	// L7386
  l_reduction_k238: for (int k238 = 0; k238 < 16; k238++) {	// L7387
  #pragma HLS pipeline II=1
    int8_t v6197 = v6188.read(); // v6188[k238];	// L7388
    int8_t a238;	// L7389
    a238 = v6197;	// L7390
    int8_t v6199 = v6189.read(); // v6189[k238];	// L7391
    int8_t b238;	// L7392
    b238 = v6199;	// L7393
    int8_t v6201 = a238;	// L7394
    int8_t v6202 = b238;	// L7395
    int16_t v6203 = v6201;	// L7396
    int16_t v6204 = v6202;	// L7397
    int16_t v6205 = v6203 * v6204;	// L7398
    int32_t v6206 = v238;	// L7399
    ap_int<33> v6207 = v6206;	// L7400
    ap_int<33> v6208 = v6205;	// L7401
    ap_int<33> v6209 = v6207 + v6208;	// L7402
    int32_t v6210 = v6209;	// L7403
    v238 = v6210;	// L7404
    int8_t v6211 = a238;	// L7405
    v6190.write(v6211); // v6190[k238] = v6211;	// L7406
    int8_t v6212 = b238;	// L7407
    v6191.write(v6212); // v6191[k238] = v6212;	// L7408
  }
  int32_t v6213 = v238;	// L7410
  v6192[v6193][v6194] = v6213;	// L7411
}

void PE_kernel_gemm_15_14(
  hls::stream< int8_t > &v6214 /* v6214[16] */,
  hls::stream< int8_t > &v6215 /* v6215[16] */,
  hls::stream< int8_t > &v6216 /* v6216[16] */,
  hls::stream< int8_t > &v6217 /* v6217[16] */,
  int32_t v6218[16][16],
  int v6219,
  int v6220
) {	// L7414
  #pragma HLS stream variable=v6214 depth=17
  #pragma HLS stream variable=v6215 depth=17
  #pragma HLS stream variable=v6216 depth=17
  #pragma HLS stream variable=v6217 depth=17
  #pragma HLS array_partition variable=v6218 complete dim=1
  #pragma HLS array_partition variable=v6218 complete dim=2

  int32_t v239;	// L7416
  v239 = 0;	// L7417
  l_reduction_k239: for (int k239 = 0; k239 < 16; k239++) {	// L7418
  #pragma HLS pipeline II=1
    int8_t v6223 = v6214.read(); // v6214[k239];	// L7419
    int8_t a239;	// L7420
    a239 = v6223;	// L7421
    int8_t v6225 = v6215.read(); // v6215[k239];	// L7422
    int8_t b239;	// L7423
    b239 = v6225;	// L7424
    int8_t v6227 = a239;	// L7425
    int8_t v6228 = b239;	// L7426
    int16_t v6229 = v6227;	// L7427
    int16_t v6230 = v6228;	// L7428
    int16_t v6231 = v6229 * v6230;	// L7429
    int32_t v6232 = v239;	// L7430
    ap_int<33> v6233 = v6232;	// L7431
    ap_int<33> v6234 = v6231;	// L7432
    ap_int<33> v6235 = v6233 + v6234;	// L7433
    int32_t v6236 = v6235;	// L7434
    v239 = v6236;	// L7435
    int8_t v6237 = a239;	// L7436
    v6216.write(v6237); // v6216[k239] = v6237;	// L7437
    int8_t v6238 = b239;	// L7438
    v6217.write(v6238); // v6217[k239] = v6238;	// L7439
  }
  int32_t v6239 = v239;	// L7441
  v6218[v6219][v6220] = v6239;	// L7442
}

void PE_kernel_gemm_0_15(
  hls::stream< int8_t > &v6240 /* v6240[16] */,
  hls::stream< int8_t > &v6241 /* v6241[16] */,
  hls::stream< int8_t > &v6242 /* v6242[16] */,
  hls::stream< int8_t > &v6243 /* v6243[16] */,
  int32_t v6244[16][16],
  int v6245,
  int v6246
) {	// L7445
  #pragma HLS stream variable=v6240 depth=17
  #pragma HLS stream variable=v6241 depth=17
  #pragma HLS stream variable=v6242 depth=17
  #pragma HLS stream variable=v6243 depth=17
  #pragma HLS array_partition variable=v6244 complete dim=1
  #pragma HLS array_partition variable=v6244 complete dim=2

  int32_t v240;	// L7447
  v240 = 0;	// L7448
  l_reduction_k240: for (int k240 = 0; k240 < 16; k240++) {	// L7449
  #pragma HLS pipeline II=1
    int8_t v6249 = v6240.read(); // v6240[k240];	// L7450
    int8_t a240;	// L7451
    a240 = v6249;	// L7452
    int8_t v6251 = v6241.read(); // v6241[k240];	// L7453
    int8_t b240;	// L7454
    b240 = v6251;	// L7455
    int8_t v6253 = a240;	// L7456
    int8_t v6254 = b240;	// L7457
    int16_t v6255 = v6253;	// L7458
    int16_t v6256 = v6254;	// L7459
    int16_t v6257 = v6255 * v6256;	// L7460
    int32_t v6258 = v240;	// L7461
    ap_int<33> v6259 = v6258;	// L7462
    ap_int<33> v6260 = v6257;	// L7463
    ap_int<33> v6261 = v6259 + v6260;	// L7464
    int32_t v6262 = v6261;	// L7465
    v240 = v6262;	// L7466
    int8_t v6263 = a240;	// L7467
    v6242.write(v6263); // v6242[k240] = v6263;	// L7468
    int8_t v6264 = b240;	// L7469
    v6243.write(v6264); // v6243[k240] = v6264;	// L7470
  }
  int32_t v6265 = v240;	// L7472
  v6244[v6245][v6246] = v6265;	// L7473
}

void PE_kernel_gemm_1_15(
  hls::stream< int8_t > &v6266 /* v6266[16] */,
  hls::stream< int8_t > &v6267 /* v6267[16] */,
  hls::stream< int8_t > &v6268 /* v6268[16] */,
  hls::stream< int8_t > &v6269 /* v6269[16] */,
  int32_t v6270[16][16],
  int v6271,
  int v6272
) {	// L7476
  #pragma HLS stream variable=v6266 depth=17
  #pragma HLS stream variable=v6267 depth=17
  #pragma HLS stream variable=v6268 depth=17
  #pragma HLS stream variable=v6269 depth=17
  #pragma HLS array_partition variable=v6270 complete dim=1
  #pragma HLS array_partition variable=v6270 complete dim=2

  int32_t v241;	// L7478
  v241 = 0;	// L7479
  l_reduction_k241: for (int k241 = 0; k241 < 16; k241++) {	// L7480
  #pragma HLS pipeline II=1
    int8_t v6275 = v6266.read(); // v6266[k241];	// L7481
    int8_t a241;	// L7482
    a241 = v6275;	// L7483
    int8_t v6277 = v6267.read(); // v6267[k241];	// L7484
    int8_t b241;	// L7485
    b241 = v6277;	// L7486
    int8_t v6279 = a241;	// L7487
    int8_t v6280 = b241;	// L7488
    int16_t v6281 = v6279;	// L7489
    int16_t v6282 = v6280;	// L7490
    int16_t v6283 = v6281 * v6282;	// L7491
    int32_t v6284 = v241;	// L7492
    ap_int<33> v6285 = v6284;	// L7493
    ap_int<33> v6286 = v6283;	// L7494
    ap_int<33> v6287 = v6285 + v6286;	// L7495
    int32_t v6288 = v6287;	// L7496
    v241 = v6288;	// L7497
    int8_t v6289 = a241;	// L7498
    v6268.write(v6289); // v6268[k241] = v6289;	// L7499
    int8_t v6290 = b241;	// L7500
    v6269.write(v6290); // v6269[k241] = v6290;	// L7501
  }
  int32_t v6291 = v241;	// L7503
  v6270[v6271][v6272] = v6291;	// L7504
}

void PE_kernel_gemm_2_15(
  hls::stream< int8_t > &v6292 /* v6292[16] */,
  hls::stream< int8_t > &v6293 /* v6293[16] */,
  hls::stream< int8_t > &v6294 /* v6294[16] */,
  hls::stream< int8_t > &v6295 /* v6295[16] */,
  int32_t v6296[16][16],
  int v6297,
  int v6298
) {	// L7507
  #pragma HLS stream variable=v6292 depth=17
  #pragma HLS stream variable=v6293 depth=17
  #pragma HLS stream variable=v6294 depth=17
  #pragma HLS stream variable=v6295 depth=17
  #pragma HLS array_partition variable=v6296 complete dim=1
  #pragma HLS array_partition variable=v6296 complete dim=2

  int32_t v242;	// L7509
  v242 = 0;	// L7510
  l_reduction_k242: for (int k242 = 0; k242 < 16; k242++) {	// L7511
  #pragma HLS pipeline II=1
    int8_t v6301 = v6292.read(); // v6292[k242];	// L7512
    int8_t a242;	// L7513
    a242 = v6301;	// L7514
    int8_t v6303 = v6293.read(); // v6293[k242];	// L7515
    int8_t b242;	// L7516
    b242 = v6303;	// L7517
    int8_t v6305 = a242;	// L7518
    int8_t v6306 = b242;	// L7519
    int16_t v6307 = v6305;	// L7520
    int16_t v6308 = v6306;	// L7521
    int16_t v6309 = v6307 * v6308;	// L7522
    int32_t v6310 = v242;	// L7523
    ap_int<33> v6311 = v6310;	// L7524
    ap_int<33> v6312 = v6309;	// L7525
    ap_int<33> v6313 = v6311 + v6312;	// L7526
    int32_t v6314 = v6313;	// L7527
    v242 = v6314;	// L7528
    int8_t v6315 = a242;	// L7529
    v6294.write(v6315); // v6294[k242] = v6315;	// L7530
    int8_t v6316 = b242;	// L7531
    v6295.write(v6316); // v6295[k242] = v6316;	// L7532
  }
  int32_t v6317 = v242;	// L7534
  v6296[v6297][v6298] = v6317;	// L7535
}

void PE_kernel_gemm_3_15(
  hls::stream< int8_t > &v6318 /* v6318[16] */,
  hls::stream< int8_t > &v6319 /* v6319[16] */,
  hls::stream< int8_t > &v6320 /* v6320[16] */,
  hls::stream< int8_t > &v6321 /* v6321[16] */,
  int32_t v6322[16][16],
  int v6323,
  int v6324
) {	// L7538
  #pragma HLS stream variable=v6318 depth=17
  #pragma HLS stream variable=v6319 depth=17
  #pragma HLS stream variable=v6320 depth=17
  #pragma HLS stream variable=v6321 depth=17
  #pragma HLS array_partition variable=v6322 complete dim=1
  #pragma HLS array_partition variable=v6322 complete dim=2

  int32_t v243;	// L7540
  v243 = 0;	// L7541
  l_reduction_k243: for (int k243 = 0; k243 < 16; k243++) {	// L7542
  #pragma HLS pipeline II=1
    int8_t v6327 = v6318.read(); // v6318[k243];	// L7543
    int8_t a243;	// L7544
    a243 = v6327;	// L7545
    int8_t v6329 = v6319.read(); // v6319[k243];	// L7546
    int8_t b243;	// L7547
    b243 = v6329;	// L7548
    int8_t v6331 = a243;	// L7549
    int8_t v6332 = b243;	// L7550
    int16_t v6333 = v6331;	// L7551
    int16_t v6334 = v6332;	// L7552
    int16_t v6335 = v6333 * v6334;	// L7553
    int32_t v6336 = v243;	// L7554
    ap_int<33> v6337 = v6336;	// L7555
    ap_int<33> v6338 = v6335;	// L7556
    ap_int<33> v6339 = v6337 + v6338;	// L7557
    int32_t v6340 = v6339;	// L7558
    v243 = v6340;	// L7559
    int8_t v6341 = a243;	// L7560
    v6320.write(v6341); // v6320[k243] = v6341;	// L7561
    int8_t v6342 = b243;	// L7562
    v6321.write(v6342); // v6321[k243] = v6342;	// L7563
  }
  int32_t v6343 = v243;	// L7565
  v6322[v6323][v6324] = v6343;	// L7566
}

void PE_kernel_gemm_4_15(
  hls::stream< int8_t > &v6344 /* v6344[16] */,
  hls::stream< int8_t > &v6345 /* v6345[16] */,
  hls::stream< int8_t > &v6346 /* v6346[16] */,
  hls::stream< int8_t > &v6347 /* v6347[16] */,
  int32_t v6348[16][16],
  int v6349,
  int v6350
) {	// L7569
  #pragma HLS stream variable=v6344 depth=17
  #pragma HLS stream variable=v6345 depth=17
  #pragma HLS stream variable=v6346 depth=17
  #pragma HLS stream variable=v6347 depth=17
  #pragma HLS array_partition variable=v6348 complete dim=1
  #pragma HLS array_partition variable=v6348 complete dim=2

  int32_t v244;	// L7571
  v244 = 0;	// L7572
  l_reduction_k244: for (int k244 = 0; k244 < 16; k244++) {	// L7573
  #pragma HLS pipeline II=1
    int8_t v6353 = v6344.read(); // v6344[k244];	// L7574
    int8_t a244;	// L7575
    a244 = v6353;	// L7576
    int8_t v6355 = v6345.read(); // v6345[k244];	// L7577
    int8_t b244;	// L7578
    b244 = v6355;	// L7579
    int8_t v6357 = a244;	// L7580
    int8_t v6358 = b244;	// L7581
    int16_t v6359 = v6357;	// L7582
    int16_t v6360 = v6358;	// L7583
    int16_t v6361 = v6359 * v6360;	// L7584
    int32_t v6362 = v244;	// L7585
    ap_int<33> v6363 = v6362;	// L7586
    ap_int<33> v6364 = v6361;	// L7587
    ap_int<33> v6365 = v6363 + v6364;	// L7588
    int32_t v6366 = v6365;	// L7589
    v244 = v6366;	// L7590
    int8_t v6367 = a244;	// L7591
    v6346.write(v6367); // v6346[k244] = v6367;	// L7592
    int8_t v6368 = b244;	// L7593
    v6347.write(v6368); // v6347[k244] = v6368;	// L7594
  }
  int32_t v6369 = v244;	// L7596
  v6348[v6349][v6350] = v6369;	// L7597
}

void PE_kernel_gemm_5_15(
  hls::stream< int8_t > &v6370 /* v6370[16] */,
  hls::stream< int8_t > &v6371 /* v6371[16] */,
  hls::stream< int8_t > &v6372 /* v6372[16] */,
  hls::stream< int8_t > &v6373 /* v6373[16] */,
  int32_t v6374[16][16],
  int v6375,
  int v6376
) {	// L7600
  #pragma HLS stream variable=v6370 depth=17
  #pragma HLS stream variable=v6371 depth=17
  #pragma HLS stream variable=v6372 depth=17
  #pragma HLS stream variable=v6373 depth=17
  #pragma HLS array_partition variable=v6374 complete dim=1
  #pragma HLS array_partition variable=v6374 complete dim=2

  int32_t v245;	// L7602
  v245 = 0;	// L7603
  l_reduction_k245: for (int k245 = 0; k245 < 16; k245++) {	// L7604
  #pragma HLS pipeline II=1
    int8_t v6379 = v6370.read(); // v6370[k245];	// L7605
    int8_t a245;	// L7606
    a245 = v6379;	// L7607
    int8_t v6381 = v6371.read(); // v6371[k245];	// L7608
    int8_t b245;	// L7609
    b245 = v6381;	// L7610
    int8_t v6383 = a245;	// L7611
    int8_t v6384 = b245;	// L7612
    int16_t v6385 = v6383;	// L7613
    int16_t v6386 = v6384;	// L7614
    int16_t v6387 = v6385 * v6386;	// L7615
    int32_t v6388 = v245;	// L7616
    ap_int<33> v6389 = v6388;	// L7617
    ap_int<33> v6390 = v6387;	// L7618
    ap_int<33> v6391 = v6389 + v6390;	// L7619
    int32_t v6392 = v6391;	// L7620
    v245 = v6392;	// L7621
    int8_t v6393 = a245;	// L7622
    v6372.write(v6393); // v6372[k245] = v6393;	// L7623
    int8_t v6394 = b245;	// L7624
    v6373.write(v6394); // v6373[k245] = v6394;	// L7625
  }
  int32_t v6395 = v245;	// L7627
  v6374[v6375][v6376] = v6395;	// L7628
}

void PE_kernel_gemm_6_15(
  hls::stream< int8_t > &v6396 /* v6396[16] */,
  hls::stream< int8_t > &v6397 /* v6397[16] */,
  hls::stream< int8_t > &v6398 /* v6398[16] */,
  hls::stream< int8_t > &v6399 /* v6399[16] */,
  int32_t v6400[16][16],
  int v6401,
  int v6402
) {	// L7631
  #pragma HLS stream variable=v6396 depth=17
  #pragma HLS stream variable=v6397 depth=17
  #pragma HLS stream variable=v6398 depth=17
  #pragma HLS stream variable=v6399 depth=17
  #pragma HLS array_partition variable=v6400 complete dim=1
  #pragma HLS array_partition variable=v6400 complete dim=2

  int32_t v246;	// L7633
  v246 = 0;	// L7634
  l_reduction_k246: for (int k246 = 0; k246 < 16; k246++) {	// L7635
  #pragma HLS pipeline II=1
    int8_t v6405 = v6396.read(); // v6396[k246];	// L7636
    int8_t a246;	// L7637
    a246 = v6405;	// L7638
    int8_t v6407 = v6397.read(); // v6397[k246];	// L7639
    int8_t b246;	// L7640
    b246 = v6407;	// L7641
    int8_t v6409 = a246;	// L7642
    int8_t v6410 = b246;	// L7643
    int16_t v6411 = v6409;	// L7644
    int16_t v6412 = v6410;	// L7645
    int16_t v6413 = v6411 * v6412;	// L7646
    int32_t v6414 = v246;	// L7647
    ap_int<33> v6415 = v6414;	// L7648
    ap_int<33> v6416 = v6413;	// L7649
    ap_int<33> v6417 = v6415 + v6416;	// L7650
    int32_t v6418 = v6417;	// L7651
    v246 = v6418;	// L7652
    int8_t v6419 = a246;	// L7653
    v6398.write(v6419); // v6398[k246] = v6419;	// L7654
    int8_t v6420 = b246;	// L7655
    v6399.write(v6420); // v6399[k246] = v6420;	// L7656
  }
  int32_t v6421 = v246;	// L7658
  v6400[v6401][v6402] = v6421;	// L7659
}

void PE_kernel_gemm_7_15(
  hls::stream< int8_t > &v6422 /* v6422[16] */,
  hls::stream< int8_t > &v6423 /* v6423[16] */,
  hls::stream< int8_t > &v6424 /* v6424[16] */,
  hls::stream< int8_t > &v6425 /* v6425[16] */,
  int32_t v6426[16][16],
  int v6427,
  int v6428
) {	// L7662
  #pragma HLS stream variable=v6422 depth=17
  #pragma HLS stream variable=v6423 depth=17
  #pragma HLS stream variable=v6424 depth=17
  #pragma HLS stream variable=v6425 depth=17
  #pragma HLS array_partition variable=v6426 complete dim=1
  #pragma HLS array_partition variable=v6426 complete dim=2

  int32_t v247;	// L7664
  v247 = 0;	// L7665
  l_reduction_k247: for (int k247 = 0; k247 < 16; k247++) {	// L7666
  #pragma HLS pipeline II=1
    int8_t v6431 = v6422.read(); // v6422[k247];	// L7667
    int8_t a247;	// L7668
    a247 = v6431;	// L7669
    int8_t v6433 = v6423.read(); // v6423[k247];	// L7670
    int8_t b247;	// L7671
    b247 = v6433;	// L7672
    int8_t v6435 = a247;	// L7673
    int8_t v6436 = b247;	// L7674
    int16_t v6437 = v6435;	// L7675
    int16_t v6438 = v6436;	// L7676
    int16_t v6439 = v6437 * v6438;	// L7677
    int32_t v6440 = v247;	// L7678
    ap_int<33> v6441 = v6440;	// L7679
    ap_int<33> v6442 = v6439;	// L7680
    ap_int<33> v6443 = v6441 + v6442;	// L7681
    int32_t v6444 = v6443;	// L7682
    v247 = v6444;	// L7683
    int8_t v6445 = a247;	// L7684
    v6424.write(v6445); // v6424[k247] = v6445;	// L7685
    int8_t v6446 = b247;	// L7686
    v6425.write(v6446); // v6425[k247] = v6446;	// L7687
  }
  int32_t v6447 = v247;	// L7689
  v6426[v6427][v6428] = v6447;	// L7690
}

void PE_kernel_gemm_8_15(
  hls::stream< int8_t > &v6448 /* v6448[16] */,
  hls::stream< int8_t > &v6449 /* v6449[16] */,
  hls::stream< int8_t > &v6450 /* v6450[16] */,
  hls::stream< int8_t > &v6451 /* v6451[16] */,
  int32_t v6452[16][16],
  int v6453,
  int v6454
) {	// L7693
  #pragma HLS stream variable=v6448 depth=17
  #pragma HLS stream variable=v6449 depth=17
  #pragma HLS stream variable=v6450 depth=17
  #pragma HLS stream variable=v6451 depth=17
  #pragma HLS array_partition variable=v6452 complete dim=1
  #pragma HLS array_partition variable=v6452 complete dim=2

  int32_t v248;	// L7695
  v248 = 0;	// L7696
  l_reduction_k248: for (int k248 = 0; k248 < 16; k248++) {	// L7697
  #pragma HLS pipeline II=1
    int8_t v6457 = v6448.read(); // v6448[k248];	// L7698
    int8_t a248;	// L7699
    a248 = v6457;	// L7700
    int8_t v6459 = v6449.read(); // v6449[k248];	// L7701
    int8_t b248;	// L7702
    b248 = v6459;	// L7703
    int8_t v6461 = a248;	// L7704
    int8_t v6462 = b248;	// L7705
    int16_t v6463 = v6461;	// L7706
    int16_t v6464 = v6462;	// L7707
    int16_t v6465 = v6463 * v6464;	// L7708
    int32_t v6466 = v248;	// L7709
    ap_int<33> v6467 = v6466;	// L7710
    ap_int<33> v6468 = v6465;	// L7711
    ap_int<33> v6469 = v6467 + v6468;	// L7712
    int32_t v6470 = v6469;	// L7713
    v248 = v6470;	// L7714
    int8_t v6471 = a248;	// L7715
    v6450.write(v6471); // v6450[k248] = v6471;	// L7716
    int8_t v6472 = b248;	// L7717
    v6451.write(v6472); // v6451[k248] = v6472;	// L7718
  }
  int32_t v6473 = v248;	// L7720
  v6452[v6453][v6454] = v6473;	// L7721
}

void PE_kernel_gemm_9_15(
  hls::stream< int8_t > &v6474 /* v6474[16] */,
  hls::stream< int8_t > &v6475 /* v6475[16] */,
  hls::stream< int8_t > &v6476 /* v6476[16] */,
  hls::stream< int8_t > &v6477 /* v6477[16] */,
  int32_t v6478[16][16],
  int v6479,
  int v6480
) {	// L7724
  #pragma HLS stream variable=v6474 depth=17
  #pragma HLS stream variable=v6475 depth=17
  #pragma HLS stream variable=v6476 depth=17
  #pragma HLS stream variable=v6477 depth=17
  #pragma HLS array_partition variable=v6478 complete dim=1
  #pragma HLS array_partition variable=v6478 complete dim=2

  int32_t v249;	// L7726
  v249 = 0;	// L7727
  l_reduction_k249: for (int k249 = 0; k249 < 16; k249++) {	// L7728
  #pragma HLS pipeline II=1
    int8_t v6483 = v6474.read(); // v6474[k249];	// L7729
    int8_t a249;	// L7730
    a249 = v6483;	// L7731
    int8_t v6485 = v6475.read(); // v6475[k249];	// L7732
    int8_t b249;	// L7733
    b249 = v6485;	// L7734
    int8_t v6487 = a249;	// L7735
    int8_t v6488 = b249;	// L7736
    int16_t v6489 = v6487;	// L7737
    int16_t v6490 = v6488;	// L7738
    int16_t v6491 = v6489 * v6490;	// L7739
    int32_t v6492 = v249;	// L7740
    ap_int<33> v6493 = v6492;	// L7741
    ap_int<33> v6494 = v6491;	// L7742
    ap_int<33> v6495 = v6493 + v6494;	// L7743
    int32_t v6496 = v6495;	// L7744
    v249 = v6496;	// L7745
    int8_t v6497 = a249;	// L7746
    v6476.write(v6497); // v6476[k249] = v6497;	// L7747
    int8_t v6498 = b249;	// L7748
    v6477.write(v6498); // v6477[k249] = v6498;	// L7749
  }
  int32_t v6499 = v249;	// L7751
  v6478[v6479][v6480] = v6499;	// L7752
}

void PE_kernel_gemm_10_15(
  hls::stream< int8_t > &v6500 /* v6500[16] */,
  hls::stream< int8_t > &v6501 /* v6501[16] */,
  hls::stream< int8_t > &v6502 /* v6502[16] */,
  hls::stream< int8_t > &v6503 /* v6503[16] */,
  int32_t v6504[16][16],
  int v6505,
  int v6506
) {	// L7755
  #pragma HLS stream variable=v6500 depth=17
  #pragma HLS stream variable=v6501 depth=17
  #pragma HLS stream variable=v6502 depth=17
  #pragma HLS stream variable=v6503 depth=17
  #pragma HLS array_partition variable=v6504 complete dim=1
  #pragma HLS array_partition variable=v6504 complete dim=2

  int32_t v250;	// L7757
  v250 = 0;	// L7758
  l_reduction_k250: for (int k250 = 0; k250 < 16; k250++) {	// L7759
  #pragma HLS pipeline II=1
    int8_t v6509 = v6500.read(); // v6500[k250];	// L7760
    int8_t a250;	// L7761
    a250 = v6509;	// L7762
    int8_t v6511 = v6501.read(); // v6501[k250];	// L7763
    int8_t b250;	// L7764
    b250 = v6511;	// L7765
    int8_t v6513 = a250;	// L7766
    int8_t v6514 = b250;	// L7767
    int16_t v6515 = v6513;	// L7768
    int16_t v6516 = v6514;	// L7769
    int16_t v6517 = v6515 * v6516;	// L7770
    int32_t v6518 = v250;	// L7771
    ap_int<33> v6519 = v6518;	// L7772
    ap_int<33> v6520 = v6517;	// L7773
    ap_int<33> v6521 = v6519 + v6520;	// L7774
    int32_t v6522 = v6521;	// L7775
    v250 = v6522;	// L7776
    int8_t v6523 = a250;	// L7777
    v6502.write(v6523); // v6502[k250] = v6523;	// L7778
    int8_t v6524 = b250;	// L7779
    v6503.write(v6524); // v6503[k250] = v6524;	// L7780
  }
  int32_t v6525 = v250;	// L7782
  v6504[v6505][v6506] = v6525;	// L7783
}

void PE_kernel_gemm_11_15(
  hls::stream< int8_t > &v6526 /* v6526[16] */,
  hls::stream< int8_t > &v6527 /* v6527[16] */,
  hls::stream< int8_t > &v6528 /* v6528[16] */,
  hls::stream< int8_t > &v6529 /* v6529[16] */,
  int32_t v6530[16][16],
  int v6531,
  int v6532
) {	// L7786
  #pragma HLS stream variable=v6526 depth=17
  #pragma HLS stream variable=v6527 depth=17
  #pragma HLS stream variable=v6528 depth=17
  #pragma HLS stream variable=v6529 depth=17
  #pragma HLS array_partition variable=v6530 complete dim=1
  #pragma HLS array_partition variable=v6530 complete dim=2

  int32_t v251;	// L7788
  v251 = 0;	// L7789
  l_reduction_k251: for (int k251 = 0; k251 < 16; k251++) {	// L7790
  #pragma HLS pipeline II=1
    int8_t v6535 = v6526.read(); // v6526[k251];	// L7791
    int8_t a251;	// L7792
    a251 = v6535;	// L7793
    int8_t v6537 = v6527.read(); // v6527[k251];	// L7794
    int8_t b251;	// L7795
    b251 = v6537;	// L7796
    int8_t v6539 = a251;	// L7797
    int8_t v6540 = b251;	// L7798
    int16_t v6541 = v6539;	// L7799
    int16_t v6542 = v6540;	// L7800
    int16_t v6543 = v6541 * v6542;	// L7801
    int32_t v6544 = v251;	// L7802
    ap_int<33> v6545 = v6544;	// L7803
    ap_int<33> v6546 = v6543;	// L7804
    ap_int<33> v6547 = v6545 + v6546;	// L7805
    int32_t v6548 = v6547;	// L7806
    v251 = v6548;	// L7807
    int8_t v6549 = a251;	// L7808
    v6528.write(v6549); // v6528[k251] = v6549;	// L7809
    int8_t v6550 = b251;	// L7810
    v6529.write(v6550); // v6529[k251] = v6550;	// L7811
  }
  int32_t v6551 = v251;	// L7813
  v6530[v6531][v6532] = v6551;	// L7814
}

void PE_kernel_gemm_12_15(
  hls::stream< int8_t > &v6552 /* v6552[16] */,
  hls::stream< int8_t > &v6553 /* v6553[16] */,
  hls::stream< int8_t > &v6554 /* v6554[16] */,
  hls::stream< int8_t > &v6555 /* v6555[16] */,
  int32_t v6556[16][16],
  int v6557,
  int v6558
) {	// L7817
  #pragma HLS stream variable=v6552 depth=17
  #pragma HLS stream variable=v6553 depth=17
  #pragma HLS stream variable=v6554 depth=17
  #pragma HLS stream variable=v6555 depth=17
  #pragma HLS array_partition variable=v6556 complete dim=1
  #pragma HLS array_partition variable=v6556 complete dim=2

  int32_t v252;	// L7819
  v252 = 0;	// L7820
  l_reduction_k252: for (int k252 = 0; k252 < 16; k252++) {	// L7821
  #pragma HLS pipeline II=1
    int8_t v6561 = v6552.read(); // v6552[k252];	// L7822
    int8_t a252;	// L7823
    a252 = v6561;	// L7824
    int8_t v6563 = v6553.read(); // v6553[k252];	// L7825
    int8_t b252;	// L7826
    b252 = v6563;	// L7827
    int8_t v6565 = a252;	// L7828
    int8_t v6566 = b252;	// L7829
    int16_t v6567 = v6565;	// L7830
    int16_t v6568 = v6566;	// L7831
    int16_t v6569 = v6567 * v6568;	// L7832
    int32_t v6570 = v252;	// L7833
    ap_int<33> v6571 = v6570;	// L7834
    ap_int<33> v6572 = v6569;	// L7835
    ap_int<33> v6573 = v6571 + v6572;	// L7836
    int32_t v6574 = v6573;	// L7837
    v252 = v6574;	// L7838
    int8_t v6575 = a252;	// L7839
    v6554.write(v6575); // v6554[k252] = v6575;	// L7840
    int8_t v6576 = b252;	// L7841
    v6555.write(v6576); // v6555[k252] = v6576;	// L7842
  }
  int32_t v6577 = v252;	// L7844
  v6556[v6557][v6558] = v6577;	// L7845
}

void PE_kernel_gemm_13_15(
  hls::stream< int8_t > &v6578 /* v6578[16] */,
  hls::stream< int8_t > &v6579 /* v6579[16] */,
  hls::stream< int8_t > &v6580 /* v6580[16] */,
  hls::stream< int8_t > &v6581 /* v6581[16] */,
  int32_t v6582[16][16],
  int v6583,
  int v6584
) {	// L7848
  #pragma HLS stream variable=v6578 depth=17
  #pragma HLS stream variable=v6579 depth=17
  #pragma HLS stream variable=v6580 depth=17
  #pragma HLS stream variable=v6581 depth=17
  #pragma HLS array_partition variable=v6582 complete dim=1
  #pragma HLS array_partition variable=v6582 complete dim=2

  int32_t v253;	// L7850
  v253 = 0;	// L7851
  l_reduction_k253: for (int k253 = 0; k253 < 16; k253++) {	// L7852
  #pragma HLS pipeline II=1
    int8_t v6587 = v6578.read(); // v6578[k253];	// L7853
    int8_t a253;	// L7854
    a253 = v6587;	// L7855
    int8_t v6589 = v6579.read(); // v6579[k253];	// L7856
    int8_t b253;	// L7857
    b253 = v6589;	// L7858
    int8_t v6591 = a253;	// L7859
    int8_t v6592 = b253;	// L7860
    int16_t v6593 = v6591;	// L7861
    int16_t v6594 = v6592;	// L7862
    int16_t v6595 = v6593 * v6594;	// L7863
    int32_t v6596 = v253;	// L7864
    ap_int<33> v6597 = v6596;	// L7865
    ap_int<33> v6598 = v6595;	// L7866
    ap_int<33> v6599 = v6597 + v6598;	// L7867
    int32_t v6600 = v6599;	// L7868
    v253 = v6600;	// L7869
    int8_t v6601 = a253;	// L7870
    v6580.write(v6601); // v6580[k253] = v6601;	// L7871
    int8_t v6602 = b253;	// L7872
    v6581.write(v6602); // v6581[k253] = v6602;	// L7873
  }
  int32_t v6603 = v253;	// L7875
  v6582[v6583][v6584] = v6603;	// L7876
}

void PE_kernel_gemm_14_15(
  hls::stream< int8_t > &v6604 /* v6604[16] */,
  hls::stream< int8_t > &v6605 /* v6605[16] */,
  hls::stream< int8_t > &v6606 /* v6606[16] */,
  hls::stream< int8_t > &v6607 /* v6607[16] */,
  int32_t v6608[16][16],
  int v6609,
  int v6610
) {	// L7879
  #pragma HLS stream variable=v6604 depth=17
  #pragma HLS stream variable=v6605 depth=17
  #pragma HLS stream variable=v6606 depth=17
  #pragma HLS stream variable=v6607 depth=17
  #pragma HLS array_partition variable=v6608 complete dim=1
  #pragma HLS array_partition variable=v6608 complete dim=2

  int32_t v254;	// L7881
  v254 = 0;	// L7882
  l_reduction_k254: for (int k254 = 0; k254 < 16; k254++) {	// L7883
  #pragma HLS pipeline II=1
    int8_t v6613 = v6604.read(); // v6604[k254];	// L7884
    int8_t a254;	// L7885
    a254 = v6613;	// L7886
    int8_t v6615 = v6605.read(); // v6605[k254];	// L7887
    int8_t b254;	// L7888
    b254 = v6615;	// L7889
    int8_t v6617 = a254;	// L7890
    int8_t v6618 = b254;	// L7891
    int16_t v6619 = v6617;	// L7892
    int16_t v6620 = v6618;	// L7893
    int16_t v6621 = v6619 * v6620;	// L7894
    int32_t v6622 = v254;	// L7895
    ap_int<33> v6623 = v6622;	// L7896
    ap_int<33> v6624 = v6621;	// L7897
    ap_int<33> v6625 = v6623 + v6624;	// L7898
    int32_t v6626 = v6625;	// L7899
    v254 = v6626;	// L7900
    int8_t v6627 = a254;	// L7901
    v6606.write(v6627); // v6606[k254] = v6627;	// L7902
    int8_t v6628 = b254;	// L7903
    v6607.write(v6628); // v6607[k254] = v6628;	// L7904
  }
  int32_t v6629 = v254;	// L7906
  v6608[v6609][v6610] = v6629;	// L7907
}

void PE_kernel_gemm_15_15(
  hls::stream< int8_t > &v6630 /* v6630[16] */,
  hls::stream< int8_t > &v6631 /* v6631[16] */,
  hls::stream< int8_t > &v6632 /* v6632[16] */,
  hls::stream< int8_t > &v6633 /* v6633[16] */,
  int32_t v6634[16][16],
  int v6635,
  int v6636
) {	// L7910
  #pragma HLS stream variable=v6630 depth=17
  #pragma HLS stream variable=v6631 depth=17
  #pragma HLS stream variable=v6632 depth=17
  #pragma HLS stream variable=v6633 depth=17
  #pragma HLS array_partition variable=v6634 complete dim=1
  #pragma HLS array_partition variable=v6634 complete dim=2

  int32_t v255;	// L7912
  v255 = 0;	// L7913
  l_reduction_k255: for (int k255 = 0; k255 < 16; k255++) {	// L7914
  #pragma HLS pipeline II=1
    int8_t v6639 = v6630.read(); // v6630[k255];	// L7915
    int8_t a255;	// L7916
    a255 = v6639;	// L7917
    int8_t v6641 = v6631.read(); // v6631[k255];	// L7918
    int8_t b255;	// L7919
    b255 = v6641;	// L7920
    int8_t v6643 = a255;	// L7921
    int8_t v6644 = b255;	// L7922
    int16_t v6645 = v6643;	// L7923
    int16_t v6646 = v6644;	// L7924
    int16_t v6647 = v6645 * v6646;	// L7925
    int32_t v6648 = v255;	// L7926
    ap_int<33> v6649 = v6648;	// L7927
    ap_int<33> v6650 = v6647;	// L7928
    ap_int<33> v6651 = v6649 + v6650;	// L7929
    int32_t v6652 = v6651;	// L7930
    v255 = v6652;	// L7931
    int8_t v6653 = a255;	// L7932
    v6632.write(v6653); // v6632[k255] = v6653;	// L7933
    int8_t v6654 = b255;	// L7934
    v6633.write(v6654); // v6633[k255] = v6654;	// L7935
  }
  int32_t v6655 = v255;	// L7937
  v6634[v6635][v6636] = v6655;	// L7938
}

void systolic_tile_gemm(
  int8_t v6656[16][16],
  int8_t v6657[16][16],
  int32_t v6658[16][16]
) {	// L7941
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v6656 complete dim=1

  #pragma HLS array_partition variable=v6657 complete dim=2

  #pragma HLS array_partition variable=v6658 complete dim=1
  #pragma HLS array_partition variable=v6658 complete dim=2

  hls::stream< int8_t > A_fifo[16][17] /* A_fifo[16][17][16] */;	// L7942
  #pragma HLS stream variable=A_fifo depth=17
  hls::stream< int8_t > B_fifo[16][17] /* B_fifo[16][17][16] */;	// L7943
  #pragma HLS stream variable=B_fifo depth=17
  int8_t A_drain[16];	// L7944
  int8_t B_drain[16];	// L7945
  l_data_load_k256: for (int k256 = 0; k256 < 16; k256++) {	// L7946
    l_S_m_0_m: for (int m = 0; m < 16; m++) {	// L7947
      int8_t v6665 = v6656[m][k256];	// L7948
      A_fifo[m][0].write(v6665); // A_fifo[m][0][k256] = v6665;	// L7949
    }
    l_S_n_1_n: for (int n = 0; n < 16; n++) {	// L7951
      int8_t v6667 = v6657[k256][n];	// L7952
      B_fifo[n][0].write(v6667); // B_fifo[n][0][k256] = v6667;	// L7953
    }
  }
  hls::stream< int8_t > &v6668 /* v6668[16] */ = A_fifo[0][0];	// L7957
  hls::stream< int8_t > &v6669 /* v6669[16] */ = B_fifo[0][0];	// L7958
  hls::stream< int8_t > &v6670 /* v6670[16] */ = A_fifo[0][1];	// L7964
  hls::stream< int8_t > &v6671 /* v6671[16] */ = B_fifo[0][1];	// L7965
  PE_kernel_gemm_0_0(v6668, v6669, v6670, v6671, v6658, 0, 0);	// L7966
  hls::stream< int8_t > &v6672 /* v6672[16] */ = A_fifo[0][1];	// L7968
  hls::stream< int8_t > &v6673 /* v6673[16] */ = B_fifo[1][0];	// L7969
  hls::stream< int8_t > &v6674 /* v6674[16] */ = A_fifo[0][2];	// L7973
  hls::stream< int8_t > &v6675 /* v6675[16] */ = B_fifo[1][1];	// L7974
  PE_kernel_gemm_1_0(v6672, v6673, v6674, v6675, v6658, 0, 1);	// L7975
  hls::stream< int8_t > &v6676 /* v6676[16] */ = A_fifo[0][2];	// L7977
  hls::stream< int8_t > &v6677 /* v6677[16] */ = B_fifo[2][0];	// L7978
  hls::stream< int8_t > &v6678 /* v6678[16] */ = A_fifo[0][3];	// L7982
  hls::stream< int8_t > &v6679 /* v6679[16] */ = B_fifo[2][1];	// L7983
  PE_kernel_gemm_2_0(v6676, v6677, v6678, v6679, v6658, 0, 2);	// L7984
  hls::stream< int8_t > &v6680 /* v6680[16] */ = A_fifo[0][3];	// L7986
  hls::stream< int8_t > &v6681 /* v6681[16] */ = B_fifo[3][0];	// L7987
  hls::stream< int8_t > &v6682 /* v6682[16] */ = A_fifo[0][4];	// L7991
  hls::stream< int8_t > &v6683 /* v6683[16] */ = B_fifo[3][1];	// L7992
  PE_kernel_gemm_3_0(v6680, v6681, v6682, v6683, v6658, 0, 3);	// L7993
  hls::stream< int8_t > &v6684 /* v6684[16] */ = A_fifo[0][4];	// L7995
  hls::stream< int8_t > &v6685 /* v6685[16] */ = B_fifo[4][0];	// L7996
  hls::stream< int8_t > &v6686 /* v6686[16] */ = A_fifo[0][5];	// L8000
  hls::stream< int8_t > &v6687 /* v6687[16] */ = B_fifo[4][1];	// L8001
  PE_kernel_gemm_4_0(v6684, v6685, v6686, v6687, v6658, 0, 4);	// L8002
  hls::stream< int8_t > &v6688 /* v6688[16] */ = A_fifo[0][5];	// L8004
  hls::stream< int8_t > &v6689 /* v6689[16] */ = B_fifo[5][0];	// L8005
  hls::stream< int8_t > &v6690 /* v6690[16] */ = A_fifo[0][6];	// L8009
  hls::stream< int8_t > &v6691 /* v6691[16] */ = B_fifo[5][1];	// L8010
  PE_kernel_gemm_5_0(v6688, v6689, v6690, v6691, v6658, 0, 5);	// L8011
  hls::stream< int8_t > &v6692 /* v6692[16] */ = A_fifo[0][6];	// L8013
  hls::stream< int8_t > &v6693 /* v6693[16] */ = B_fifo[6][0];	// L8014
  hls::stream< int8_t > &v6694 /* v6694[16] */ = A_fifo[0][7];	// L8018
  hls::stream< int8_t > &v6695 /* v6695[16] */ = B_fifo[6][1];	// L8019
  PE_kernel_gemm_6_0(v6692, v6693, v6694, v6695, v6658, 0, 6);	// L8020
  hls::stream< int8_t > &v6696 /* v6696[16] */ = A_fifo[0][7];	// L8022
  hls::stream< int8_t > &v6697 /* v6697[16] */ = B_fifo[7][0];	// L8023
  hls::stream< int8_t > &v6698 /* v6698[16] */ = A_fifo[0][8];	// L8027
  hls::stream< int8_t > &v6699 /* v6699[16] */ = B_fifo[7][1];	// L8028
  PE_kernel_gemm_7_0(v6696, v6697, v6698, v6699, v6658, 0, 7);	// L8029
  hls::stream< int8_t > &v6700 /* v6700[16] */ = A_fifo[0][8];	// L8031
  hls::stream< int8_t > &v6701 /* v6701[16] */ = B_fifo[8][0];	// L8032
  hls::stream< int8_t > &v6702 /* v6702[16] */ = A_fifo[0][9];	// L8036
  hls::stream< int8_t > &v6703 /* v6703[16] */ = B_fifo[8][1];	// L8037
  PE_kernel_gemm_8_0(v6700, v6701, v6702, v6703, v6658, 0, 8);	// L8038
  hls::stream< int8_t > &v6704 /* v6704[16] */ = A_fifo[0][9];	// L8040
  hls::stream< int8_t > &v6705 /* v6705[16] */ = B_fifo[9][0];	// L8041
  hls::stream< int8_t > &v6706 /* v6706[16] */ = A_fifo[0][10];	// L8045
  hls::stream< int8_t > &v6707 /* v6707[16] */ = B_fifo[9][1];	// L8046
  PE_kernel_gemm_9_0(v6704, v6705, v6706, v6707, v6658, 0, 9);	// L8047
  hls::stream< int8_t > &v6708 /* v6708[16] */ = A_fifo[0][10];	// L8049
  hls::stream< int8_t > &v6709 /* v6709[16] */ = B_fifo[10][0];	// L8050
  hls::stream< int8_t > &v6710 /* v6710[16] */ = A_fifo[0][11];	// L8054
  hls::stream< int8_t > &v6711 /* v6711[16] */ = B_fifo[10][1];	// L8055
  PE_kernel_gemm_10_0(v6708, v6709, v6710, v6711, v6658, 0, 10);	// L8056
  hls::stream< int8_t > &v6712 /* v6712[16] */ = A_fifo[0][11];	// L8058
  hls::stream< int8_t > &v6713 /* v6713[16] */ = B_fifo[11][0];	// L8059
  hls::stream< int8_t > &v6714 /* v6714[16] */ = A_fifo[0][12];	// L8063
  hls::stream< int8_t > &v6715 /* v6715[16] */ = B_fifo[11][1];	// L8064
  PE_kernel_gemm_11_0(v6712, v6713, v6714, v6715, v6658, 0, 11);	// L8065
  hls::stream< int8_t > &v6716 /* v6716[16] */ = A_fifo[0][12];	// L8067
  hls::stream< int8_t > &v6717 /* v6717[16] */ = B_fifo[12][0];	// L8068
  hls::stream< int8_t > &v6718 /* v6718[16] */ = A_fifo[0][13];	// L8072
  hls::stream< int8_t > &v6719 /* v6719[16] */ = B_fifo[12][1];	// L8073
  PE_kernel_gemm_12_0(v6716, v6717, v6718, v6719, v6658, 0, 12);	// L8074
  hls::stream< int8_t > &v6720 /* v6720[16] */ = A_fifo[0][13];	// L8076
  hls::stream< int8_t > &v6721 /* v6721[16] */ = B_fifo[13][0];	// L8077
  hls::stream< int8_t > &v6722 /* v6722[16] */ = A_fifo[0][14];	// L8081
  hls::stream< int8_t > &v6723 /* v6723[16] */ = B_fifo[13][1];	// L8082
  PE_kernel_gemm_13_0(v6720, v6721, v6722, v6723, v6658, 0, 13);	// L8083
  hls::stream< int8_t > &v6724 /* v6724[16] */ = A_fifo[0][14];	// L8085
  hls::stream< int8_t > &v6725 /* v6725[16] */ = B_fifo[14][0];	// L8086
  hls::stream< int8_t > &v6726 /* v6726[16] */ = A_fifo[0][15];	// L8090
  hls::stream< int8_t > &v6727 /* v6727[16] */ = B_fifo[14][1];	// L8091
  PE_kernel_gemm_14_0(v6724, v6725, v6726, v6727, v6658, 0, 14);	// L8092
  hls::stream< int8_t > &v6728 /* v6728[16] */ = A_fifo[0][15];	// L8094
  hls::stream< int8_t > &v6729 /* v6729[16] */ = B_fifo[15][0];	// L8095
  hls::stream< int8_t > &v6730 /* v6730[16] */ = A_fifo[0][16];	// L8099
  hls::stream< int8_t > &v6731 /* v6731[16] */ = B_fifo[15][1];	// L8100
  PE_kernel_gemm_15_0(v6728, v6729, v6730, v6731, v6658, 0, 15);	// L8101
  hls::stream< int8_t > &v6732 /* v6732[16] */ = A_fifo[1][0];	// L8102
  hls::stream< int8_t > &v6733 /* v6733[16] */ = B_fifo[0][1];	// L8103
  hls::stream< int8_t > &v6734 /* v6734[16] */ = A_fifo[1][1];	// L8104
  hls::stream< int8_t > &v6735 /* v6735[16] */ = B_fifo[0][2];	// L8105
  PE_kernel_gemm_0_1(v6732, v6733, v6734, v6735, v6658, 1, 0);	// L8106
  hls::stream< int8_t > &v6736 /* v6736[16] */ = A_fifo[1][1];	// L8107
  hls::stream< int8_t > &v6737 /* v6737[16] */ = B_fifo[1][1];	// L8108
  hls::stream< int8_t > &v6738 /* v6738[16] */ = A_fifo[1][2];	// L8109
  hls::stream< int8_t > &v6739 /* v6739[16] */ = B_fifo[1][2];	// L8110
  PE_kernel_gemm_1_1(v6736, v6737, v6738, v6739, v6658, 1, 1);	// L8111
  hls::stream< int8_t > &v6740 /* v6740[16] */ = A_fifo[1][2];	// L8112
  hls::stream< int8_t > &v6741 /* v6741[16] */ = B_fifo[2][1];	// L8113
  hls::stream< int8_t > &v6742 /* v6742[16] */ = A_fifo[1][3];	// L8114
  hls::stream< int8_t > &v6743 /* v6743[16] */ = B_fifo[2][2];	// L8115
  PE_kernel_gemm_2_1(v6740, v6741, v6742, v6743, v6658, 1, 2);	// L8116
  hls::stream< int8_t > &v6744 /* v6744[16] */ = A_fifo[1][3];	// L8117
  hls::stream< int8_t > &v6745 /* v6745[16] */ = B_fifo[3][1];	// L8118
  hls::stream< int8_t > &v6746 /* v6746[16] */ = A_fifo[1][4];	// L8119
  hls::stream< int8_t > &v6747 /* v6747[16] */ = B_fifo[3][2];	// L8120
  PE_kernel_gemm_3_1(v6744, v6745, v6746, v6747, v6658, 1, 3);	// L8121
  hls::stream< int8_t > &v6748 /* v6748[16] */ = A_fifo[1][4];	// L8122
  hls::stream< int8_t > &v6749 /* v6749[16] */ = B_fifo[4][1];	// L8123
  hls::stream< int8_t > &v6750 /* v6750[16] */ = A_fifo[1][5];	// L8124
  hls::stream< int8_t > &v6751 /* v6751[16] */ = B_fifo[4][2];	// L8125
  PE_kernel_gemm_4_1(v6748, v6749, v6750, v6751, v6658, 1, 4);	// L8126
  hls::stream< int8_t > &v6752 /* v6752[16] */ = A_fifo[1][5];	// L8127
  hls::stream< int8_t > &v6753 /* v6753[16] */ = B_fifo[5][1];	// L8128
  hls::stream< int8_t > &v6754 /* v6754[16] */ = A_fifo[1][6];	// L8129
  hls::stream< int8_t > &v6755 /* v6755[16] */ = B_fifo[5][2];	// L8130
  PE_kernel_gemm_5_1(v6752, v6753, v6754, v6755, v6658, 1, 5);	// L8131
  hls::stream< int8_t > &v6756 /* v6756[16] */ = A_fifo[1][6];	// L8132
  hls::stream< int8_t > &v6757 /* v6757[16] */ = B_fifo[6][1];	// L8133
  hls::stream< int8_t > &v6758 /* v6758[16] */ = A_fifo[1][7];	// L8134
  hls::stream< int8_t > &v6759 /* v6759[16] */ = B_fifo[6][2];	// L8135
  PE_kernel_gemm_6_1(v6756, v6757, v6758, v6759, v6658, 1, 6);	// L8136
  hls::stream< int8_t > &v6760 /* v6760[16] */ = A_fifo[1][7];	// L8137
  hls::stream< int8_t > &v6761 /* v6761[16] */ = B_fifo[7][1];	// L8138
  hls::stream< int8_t > &v6762 /* v6762[16] */ = A_fifo[1][8];	// L8139
  hls::stream< int8_t > &v6763 /* v6763[16] */ = B_fifo[7][2];	// L8140
  PE_kernel_gemm_7_1(v6760, v6761, v6762, v6763, v6658, 1, 7);	// L8141
  hls::stream< int8_t > &v6764 /* v6764[16] */ = A_fifo[1][8];	// L8142
  hls::stream< int8_t > &v6765 /* v6765[16] */ = B_fifo[8][1];	// L8143
  hls::stream< int8_t > &v6766 /* v6766[16] */ = A_fifo[1][9];	// L8144
  hls::stream< int8_t > &v6767 /* v6767[16] */ = B_fifo[8][2];	// L8145
  PE_kernel_gemm_8_1(v6764, v6765, v6766, v6767, v6658, 1, 8);	// L8146
  hls::stream< int8_t > &v6768 /* v6768[16] */ = A_fifo[1][9];	// L8147
  hls::stream< int8_t > &v6769 /* v6769[16] */ = B_fifo[9][1];	// L8148
  hls::stream< int8_t > &v6770 /* v6770[16] */ = A_fifo[1][10];	// L8149
  hls::stream< int8_t > &v6771 /* v6771[16] */ = B_fifo[9][2];	// L8150
  PE_kernel_gemm_9_1(v6768, v6769, v6770, v6771, v6658, 1, 9);	// L8151
  hls::stream< int8_t > &v6772 /* v6772[16] */ = A_fifo[1][10];	// L8152
  hls::stream< int8_t > &v6773 /* v6773[16] */ = B_fifo[10][1];	// L8153
  hls::stream< int8_t > &v6774 /* v6774[16] */ = A_fifo[1][11];	// L8154
  hls::stream< int8_t > &v6775 /* v6775[16] */ = B_fifo[10][2];	// L8155
  PE_kernel_gemm_10_1(v6772, v6773, v6774, v6775, v6658, 1, 10);	// L8156
  hls::stream< int8_t > &v6776 /* v6776[16] */ = A_fifo[1][11];	// L8157
  hls::stream< int8_t > &v6777 /* v6777[16] */ = B_fifo[11][1];	// L8158
  hls::stream< int8_t > &v6778 /* v6778[16] */ = A_fifo[1][12];	// L8159
  hls::stream< int8_t > &v6779 /* v6779[16] */ = B_fifo[11][2];	// L8160
  PE_kernel_gemm_11_1(v6776, v6777, v6778, v6779, v6658, 1, 11);	// L8161
  hls::stream< int8_t > &v6780 /* v6780[16] */ = A_fifo[1][12];	// L8162
  hls::stream< int8_t > &v6781 /* v6781[16] */ = B_fifo[12][1];	// L8163
  hls::stream< int8_t > &v6782 /* v6782[16] */ = A_fifo[1][13];	// L8164
  hls::stream< int8_t > &v6783 /* v6783[16] */ = B_fifo[12][2];	// L8165
  PE_kernel_gemm_12_1(v6780, v6781, v6782, v6783, v6658, 1, 12);	// L8166
  hls::stream< int8_t > &v6784 /* v6784[16] */ = A_fifo[1][13];	// L8167
  hls::stream< int8_t > &v6785 /* v6785[16] */ = B_fifo[13][1];	// L8168
  hls::stream< int8_t > &v6786 /* v6786[16] */ = A_fifo[1][14];	// L8169
  hls::stream< int8_t > &v6787 /* v6787[16] */ = B_fifo[13][2];	// L8170
  PE_kernel_gemm_13_1(v6784, v6785, v6786, v6787, v6658, 1, 13);	// L8171
  hls::stream< int8_t > &v6788 /* v6788[16] */ = A_fifo[1][14];	// L8172
  hls::stream< int8_t > &v6789 /* v6789[16] */ = B_fifo[14][1];	// L8173
  hls::stream< int8_t > &v6790 /* v6790[16] */ = A_fifo[1][15];	// L8174
  hls::stream< int8_t > &v6791 /* v6791[16] */ = B_fifo[14][2];	// L8175
  PE_kernel_gemm_14_1(v6788, v6789, v6790, v6791, v6658, 1, 14);	// L8176
  hls::stream< int8_t > &v6792 /* v6792[16] */ = A_fifo[1][15];	// L8177
  hls::stream< int8_t > &v6793 /* v6793[16] */ = B_fifo[15][1];	// L8178
  hls::stream< int8_t > &v6794 /* v6794[16] */ = A_fifo[1][16];	// L8179
  hls::stream< int8_t > &v6795 /* v6795[16] */ = B_fifo[15][2];	// L8180
  PE_kernel_gemm_15_1(v6792, v6793, v6794, v6795, v6658, 1, 15);	// L8181
  hls::stream< int8_t > &v6796 /* v6796[16] */ = A_fifo[2][0];	// L8182
  hls::stream< int8_t > &v6797 /* v6797[16] */ = B_fifo[0][2];	// L8183
  hls::stream< int8_t > &v6798 /* v6798[16] */ = A_fifo[2][1];	// L8184
  hls::stream< int8_t > &v6799 /* v6799[16] */ = B_fifo[0][3];	// L8185
  PE_kernel_gemm_0_2(v6796, v6797, v6798, v6799, v6658, 2, 0);	// L8186
  hls::stream< int8_t > &v6800 /* v6800[16] */ = A_fifo[2][1];	// L8187
  hls::stream< int8_t > &v6801 /* v6801[16] */ = B_fifo[1][2];	// L8188
  hls::stream< int8_t > &v6802 /* v6802[16] */ = A_fifo[2][2];	// L8189
  hls::stream< int8_t > &v6803 /* v6803[16] */ = B_fifo[1][3];	// L8190
  PE_kernel_gemm_1_2(v6800, v6801, v6802, v6803, v6658, 2, 1);	// L8191
  hls::stream< int8_t > &v6804 /* v6804[16] */ = A_fifo[2][2];	// L8192
  hls::stream< int8_t > &v6805 /* v6805[16] */ = B_fifo[2][2];	// L8193
  hls::stream< int8_t > &v6806 /* v6806[16] */ = A_fifo[2][3];	// L8194
  hls::stream< int8_t > &v6807 /* v6807[16] */ = B_fifo[2][3];	// L8195
  PE_kernel_gemm_2_2(v6804, v6805, v6806, v6807, v6658, 2, 2);	// L8196
  hls::stream< int8_t > &v6808 /* v6808[16] */ = A_fifo[2][3];	// L8197
  hls::stream< int8_t > &v6809 /* v6809[16] */ = B_fifo[3][2];	// L8198
  hls::stream< int8_t > &v6810 /* v6810[16] */ = A_fifo[2][4];	// L8199
  hls::stream< int8_t > &v6811 /* v6811[16] */ = B_fifo[3][3];	// L8200
  PE_kernel_gemm_3_2(v6808, v6809, v6810, v6811, v6658, 2, 3);	// L8201
  hls::stream< int8_t > &v6812 /* v6812[16] */ = A_fifo[2][4];	// L8202
  hls::stream< int8_t > &v6813 /* v6813[16] */ = B_fifo[4][2];	// L8203
  hls::stream< int8_t > &v6814 /* v6814[16] */ = A_fifo[2][5];	// L8204
  hls::stream< int8_t > &v6815 /* v6815[16] */ = B_fifo[4][3];	// L8205
  PE_kernel_gemm_4_2(v6812, v6813, v6814, v6815, v6658, 2, 4);	// L8206
  hls::stream< int8_t > &v6816 /* v6816[16] */ = A_fifo[2][5];	// L8207
  hls::stream< int8_t > &v6817 /* v6817[16] */ = B_fifo[5][2];	// L8208
  hls::stream< int8_t > &v6818 /* v6818[16] */ = A_fifo[2][6];	// L8209
  hls::stream< int8_t > &v6819 /* v6819[16] */ = B_fifo[5][3];	// L8210
  PE_kernel_gemm_5_2(v6816, v6817, v6818, v6819, v6658, 2, 5);	// L8211
  hls::stream< int8_t > &v6820 /* v6820[16] */ = A_fifo[2][6];	// L8212
  hls::stream< int8_t > &v6821 /* v6821[16] */ = B_fifo[6][2];	// L8213
  hls::stream< int8_t > &v6822 /* v6822[16] */ = A_fifo[2][7];	// L8214
  hls::stream< int8_t > &v6823 /* v6823[16] */ = B_fifo[6][3];	// L8215
  PE_kernel_gemm_6_2(v6820, v6821, v6822, v6823, v6658, 2, 6);	// L8216
  hls::stream< int8_t > &v6824 /* v6824[16] */ = A_fifo[2][7];	// L8217
  hls::stream< int8_t > &v6825 /* v6825[16] */ = B_fifo[7][2];	// L8218
  hls::stream< int8_t > &v6826 /* v6826[16] */ = A_fifo[2][8];	// L8219
  hls::stream< int8_t > &v6827 /* v6827[16] */ = B_fifo[7][3];	// L8220
  PE_kernel_gemm_7_2(v6824, v6825, v6826, v6827, v6658, 2, 7);	// L8221
  hls::stream< int8_t > &v6828 /* v6828[16] */ = A_fifo[2][8];	// L8222
  hls::stream< int8_t > &v6829 /* v6829[16] */ = B_fifo[8][2];	// L8223
  hls::stream< int8_t > &v6830 /* v6830[16] */ = A_fifo[2][9];	// L8224
  hls::stream< int8_t > &v6831 /* v6831[16] */ = B_fifo[8][3];	// L8225
  PE_kernel_gemm_8_2(v6828, v6829, v6830, v6831, v6658, 2, 8);	// L8226
  hls::stream< int8_t > &v6832 /* v6832[16] */ = A_fifo[2][9];	// L8227
  hls::stream< int8_t > &v6833 /* v6833[16] */ = B_fifo[9][2];	// L8228
  hls::stream< int8_t > &v6834 /* v6834[16] */ = A_fifo[2][10];	// L8229
  hls::stream< int8_t > &v6835 /* v6835[16] */ = B_fifo[9][3];	// L8230
  PE_kernel_gemm_9_2(v6832, v6833, v6834, v6835, v6658, 2, 9);	// L8231
  hls::stream< int8_t > &v6836 /* v6836[16] */ = A_fifo[2][10];	// L8232
  hls::stream< int8_t > &v6837 /* v6837[16] */ = B_fifo[10][2];	// L8233
  hls::stream< int8_t > &v6838 /* v6838[16] */ = A_fifo[2][11];	// L8234
  hls::stream< int8_t > &v6839 /* v6839[16] */ = B_fifo[10][3];	// L8235
  PE_kernel_gemm_10_2(v6836, v6837, v6838, v6839, v6658, 2, 10);	// L8236
  hls::stream< int8_t > &v6840 /* v6840[16] */ = A_fifo[2][11];	// L8237
  hls::stream< int8_t > &v6841 /* v6841[16] */ = B_fifo[11][2];	// L8238
  hls::stream< int8_t > &v6842 /* v6842[16] */ = A_fifo[2][12];	// L8239
  hls::stream< int8_t > &v6843 /* v6843[16] */ = B_fifo[11][3];	// L8240
  PE_kernel_gemm_11_2(v6840, v6841, v6842, v6843, v6658, 2, 11);	// L8241
  hls::stream< int8_t > &v6844 /* v6844[16] */ = A_fifo[2][12];	// L8242
  hls::stream< int8_t > &v6845 /* v6845[16] */ = B_fifo[12][2];	// L8243
  hls::stream< int8_t > &v6846 /* v6846[16] */ = A_fifo[2][13];	// L8244
  hls::stream< int8_t > &v6847 /* v6847[16] */ = B_fifo[12][3];	// L8245
  PE_kernel_gemm_12_2(v6844, v6845, v6846, v6847, v6658, 2, 12);	// L8246
  hls::stream< int8_t > &v6848 /* v6848[16] */ = A_fifo[2][13];	// L8247
  hls::stream< int8_t > &v6849 /* v6849[16] */ = B_fifo[13][2];	// L8248
  hls::stream< int8_t > &v6850 /* v6850[16] */ = A_fifo[2][14];	// L8249
  hls::stream< int8_t > &v6851 /* v6851[16] */ = B_fifo[13][3];	// L8250
  PE_kernel_gemm_13_2(v6848, v6849, v6850, v6851, v6658, 2, 13);	// L8251
  hls::stream< int8_t > &v6852 /* v6852[16] */ = A_fifo[2][14];	// L8252
  hls::stream< int8_t > &v6853 /* v6853[16] */ = B_fifo[14][2];	// L8253
  hls::stream< int8_t > &v6854 /* v6854[16] */ = A_fifo[2][15];	// L8254
  hls::stream< int8_t > &v6855 /* v6855[16] */ = B_fifo[14][3];	// L8255
  PE_kernel_gemm_14_2(v6852, v6853, v6854, v6855, v6658, 2, 14);	// L8256
  hls::stream< int8_t > &v6856 /* v6856[16] */ = A_fifo[2][15];	// L8257
  hls::stream< int8_t > &v6857 /* v6857[16] */ = B_fifo[15][2];	// L8258
  hls::stream< int8_t > &v6858 /* v6858[16] */ = A_fifo[2][16];	// L8259
  hls::stream< int8_t > &v6859 /* v6859[16] */ = B_fifo[15][3];	// L8260
  PE_kernel_gemm_15_2(v6856, v6857, v6858, v6859, v6658, 2, 15);	// L8261
  hls::stream< int8_t > &v6860 /* v6860[16] */ = A_fifo[3][0];	// L8262
  hls::stream< int8_t > &v6861 /* v6861[16] */ = B_fifo[0][3];	// L8263
  hls::stream< int8_t > &v6862 /* v6862[16] */ = A_fifo[3][1];	// L8264
  hls::stream< int8_t > &v6863 /* v6863[16] */ = B_fifo[0][4];	// L8265
  PE_kernel_gemm_0_3(v6860, v6861, v6862, v6863, v6658, 3, 0);	// L8266
  hls::stream< int8_t > &v6864 /* v6864[16] */ = A_fifo[3][1];	// L8267
  hls::stream< int8_t > &v6865 /* v6865[16] */ = B_fifo[1][3];	// L8268
  hls::stream< int8_t > &v6866 /* v6866[16] */ = A_fifo[3][2];	// L8269
  hls::stream< int8_t > &v6867 /* v6867[16] */ = B_fifo[1][4];	// L8270
  PE_kernel_gemm_1_3(v6864, v6865, v6866, v6867, v6658, 3, 1);	// L8271
  hls::stream< int8_t > &v6868 /* v6868[16] */ = A_fifo[3][2];	// L8272
  hls::stream< int8_t > &v6869 /* v6869[16] */ = B_fifo[2][3];	// L8273
  hls::stream< int8_t > &v6870 /* v6870[16] */ = A_fifo[3][3];	// L8274
  hls::stream< int8_t > &v6871 /* v6871[16] */ = B_fifo[2][4];	// L8275
  PE_kernel_gemm_2_3(v6868, v6869, v6870, v6871, v6658, 3, 2);	// L8276
  hls::stream< int8_t > &v6872 /* v6872[16] */ = A_fifo[3][3];	// L8277
  hls::stream< int8_t > &v6873 /* v6873[16] */ = B_fifo[3][3];	// L8278
  hls::stream< int8_t > &v6874 /* v6874[16] */ = A_fifo[3][4];	// L8279
  hls::stream< int8_t > &v6875 /* v6875[16] */ = B_fifo[3][4];	// L8280
  PE_kernel_gemm_3_3(v6872, v6873, v6874, v6875, v6658, 3, 3);	// L8281
  hls::stream< int8_t > &v6876 /* v6876[16] */ = A_fifo[3][4];	// L8282
  hls::stream< int8_t > &v6877 /* v6877[16] */ = B_fifo[4][3];	// L8283
  hls::stream< int8_t > &v6878 /* v6878[16] */ = A_fifo[3][5];	// L8284
  hls::stream< int8_t > &v6879 /* v6879[16] */ = B_fifo[4][4];	// L8285
  PE_kernel_gemm_4_3(v6876, v6877, v6878, v6879, v6658, 3, 4);	// L8286
  hls::stream< int8_t > &v6880 /* v6880[16] */ = A_fifo[3][5];	// L8287
  hls::stream< int8_t > &v6881 /* v6881[16] */ = B_fifo[5][3];	// L8288
  hls::stream< int8_t > &v6882 /* v6882[16] */ = A_fifo[3][6];	// L8289
  hls::stream< int8_t > &v6883 /* v6883[16] */ = B_fifo[5][4];	// L8290
  PE_kernel_gemm_5_3(v6880, v6881, v6882, v6883, v6658, 3, 5);	// L8291
  hls::stream< int8_t > &v6884 /* v6884[16] */ = A_fifo[3][6];	// L8292
  hls::stream< int8_t > &v6885 /* v6885[16] */ = B_fifo[6][3];	// L8293
  hls::stream< int8_t > &v6886 /* v6886[16] */ = A_fifo[3][7];	// L8294
  hls::stream< int8_t > &v6887 /* v6887[16] */ = B_fifo[6][4];	// L8295
  PE_kernel_gemm_6_3(v6884, v6885, v6886, v6887, v6658, 3, 6);	// L8296
  hls::stream< int8_t > &v6888 /* v6888[16] */ = A_fifo[3][7];	// L8297
  hls::stream< int8_t > &v6889 /* v6889[16] */ = B_fifo[7][3];	// L8298
  hls::stream< int8_t > &v6890 /* v6890[16] */ = A_fifo[3][8];	// L8299
  hls::stream< int8_t > &v6891 /* v6891[16] */ = B_fifo[7][4];	// L8300
  PE_kernel_gemm_7_3(v6888, v6889, v6890, v6891, v6658, 3, 7);	// L8301
  hls::stream< int8_t > &v6892 /* v6892[16] */ = A_fifo[3][8];	// L8302
  hls::stream< int8_t > &v6893 /* v6893[16] */ = B_fifo[8][3];	// L8303
  hls::stream< int8_t > &v6894 /* v6894[16] */ = A_fifo[3][9];	// L8304
  hls::stream< int8_t > &v6895 /* v6895[16] */ = B_fifo[8][4];	// L8305
  PE_kernel_gemm_8_3(v6892, v6893, v6894, v6895, v6658, 3, 8);	// L8306
  hls::stream< int8_t > &v6896 /* v6896[16] */ = A_fifo[3][9];	// L8307
  hls::stream< int8_t > &v6897 /* v6897[16] */ = B_fifo[9][3];	// L8308
  hls::stream< int8_t > &v6898 /* v6898[16] */ = A_fifo[3][10];	// L8309
  hls::stream< int8_t > &v6899 /* v6899[16] */ = B_fifo[9][4];	// L8310
  PE_kernel_gemm_9_3(v6896, v6897, v6898, v6899, v6658, 3, 9);	// L8311
  hls::stream< int8_t > &v6900 /* v6900[16] */ = A_fifo[3][10];	// L8312
  hls::stream< int8_t > &v6901 /* v6901[16] */ = B_fifo[10][3];	// L8313
  hls::stream< int8_t > &v6902 /* v6902[16] */ = A_fifo[3][11];	// L8314
  hls::stream< int8_t > &v6903 /* v6903[16] */ = B_fifo[10][4];	// L8315
  PE_kernel_gemm_10_3(v6900, v6901, v6902, v6903, v6658, 3, 10);	// L8316
  hls::stream< int8_t > &v6904 /* v6904[16] */ = A_fifo[3][11];	// L8317
  hls::stream< int8_t > &v6905 /* v6905[16] */ = B_fifo[11][3];	// L8318
  hls::stream< int8_t > &v6906 /* v6906[16] */ = A_fifo[3][12];	// L8319
  hls::stream< int8_t > &v6907 /* v6907[16] */ = B_fifo[11][4];	// L8320
  PE_kernel_gemm_11_3(v6904, v6905, v6906, v6907, v6658, 3, 11);	// L8321
  hls::stream< int8_t > &v6908 /* v6908[16] */ = A_fifo[3][12];	// L8322
  hls::stream< int8_t > &v6909 /* v6909[16] */ = B_fifo[12][3];	// L8323
  hls::stream< int8_t > &v6910 /* v6910[16] */ = A_fifo[3][13];	// L8324
  hls::stream< int8_t > &v6911 /* v6911[16] */ = B_fifo[12][4];	// L8325
  PE_kernel_gemm_12_3(v6908, v6909, v6910, v6911, v6658, 3, 12);	// L8326
  hls::stream< int8_t > &v6912 /* v6912[16] */ = A_fifo[3][13];	// L8327
  hls::stream< int8_t > &v6913 /* v6913[16] */ = B_fifo[13][3];	// L8328
  hls::stream< int8_t > &v6914 /* v6914[16] */ = A_fifo[3][14];	// L8329
  hls::stream< int8_t > &v6915 /* v6915[16] */ = B_fifo[13][4];	// L8330
  PE_kernel_gemm_13_3(v6912, v6913, v6914, v6915, v6658, 3, 13);	// L8331
  hls::stream< int8_t > &v6916 /* v6916[16] */ = A_fifo[3][14];	// L8332
  hls::stream< int8_t > &v6917 /* v6917[16] */ = B_fifo[14][3];	// L8333
  hls::stream< int8_t > &v6918 /* v6918[16] */ = A_fifo[3][15];	// L8334
  hls::stream< int8_t > &v6919 /* v6919[16] */ = B_fifo[14][4];	// L8335
  PE_kernel_gemm_14_3(v6916, v6917, v6918, v6919, v6658, 3, 14);	// L8336
  hls::stream< int8_t > &v6920 /* v6920[16] */ = A_fifo[3][15];	// L8337
  hls::stream< int8_t > &v6921 /* v6921[16] */ = B_fifo[15][3];	// L8338
  hls::stream< int8_t > &v6922 /* v6922[16] */ = A_fifo[3][16];	// L8339
  hls::stream< int8_t > &v6923 /* v6923[16] */ = B_fifo[15][4];	// L8340
  PE_kernel_gemm_15_3(v6920, v6921, v6922, v6923, v6658, 3, 15);	// L8341
  hls::stream< int8_t > &v6924 /* v6924[16] */ = A_fifo[4][0];	// L8342
  hls::stream< int8_t > &v6925 /* v6925[16] */ = B_fifo[0][4];	// L8343
  hls::stream< int8_t > &v6926 /* v6926[16] */ = A_fifo[4][1];	// L8344
  hls::stream< int8_t > &v6927 /* v6927[16] */ = B_fifo[0][5];	// L8345
  PE_kernel_gemm_0_4(v6924, v6925, v6926, v6927, v6658, 4, 0);	// L8346
  hls::stream< int8_t > &v6928 /* v6928[16] */ = A_fifo[4][1];	// L8347
  hls::stream< int8_t > &v6929 /* v6929[16] */ = B_fifo[1][4];	// L8348
  hls::stream< int8_t > &v6930 /* v6930[16] */ = A_fifo[4][2];	// L8349
  hls::stream< int8_t > &v6931 /* v6931[16] */ = B_fifo[1][5];	// L8350
  PE_kernel_gemm_1_4(v6928, v6929, v6930, v6931, v6658, 4, 1);	// L8351
  hls::stream< int8_t > &v6932 /* v6932[16] */ = A_fifo[4][2];	// L8352
  hls::stream< int8_t > &v6933 /* v6933[16] */ = B_fifo[2][4];	// L8353
  hls::stream< int8_t > &v6934 /* v6934[16] */ = A_fifo[4][3];	// L8354
  hls::stream< int8_t > &v6935 /* v6935[16] */ = B_fifo[2][5];	// L8355
  PE_kernel_gemm_2_4(v6932, v6933, v6934, v6935, v6658, 4, 2);	// L8356
  hls::stream< int8_t > &v6936 /* v6936[16] */ = A_fifo[4][3];	// L8357
  hls::stream< int8_t > &v6937 /* v6937[16] */ = B_fifo[3][4];	// L8358
  hls::stream< int8_t > &v6938 /* v6938[16] */ = A_fifo[4][4];	// L8359
  hls::stream< int8_t > &v6939 /* v6939[16] */ = B_fifo[3][5];	// L8360
  PE_kernel_gemm_3_4(v6936, v6937, v6938, v6939, v6658, 4, 3);	// L8361
  hls::stream< int8_t > &v6940 /* v6940[16] */ = A_fifo[4][4];	// L8362
  hls::stream< int8_t > &v6941 /* v6941[16] */ = B_fifo[4][4];	// L8363
  hls::stream< int8_t > &v6942 /* v6942[16] */ = A_fifo[4][5];	// L8364
  hls::stream< int8_t > &v6943 /* v6943[16] */ = B_fifo[4][5];	// L8365
  PE_kernel_gemm_4_4(v6940, v6941, v6942, v6943, v6658, 4, 4);	// L8366
  hls::stream< int8_t > &v6944 /* v6944[16] */ = A_fifo[4][5];	// L8367
  hls::stream< int8_t > &v6945 /* v6945[16] */ = B_fifo[5][4];	// L8368
  hls::stream< int8_t > &v6946 /* v6946[16] */ = A_fifo[4][6];	// L8369
  hls::stream< int8_t > &v6947 /* v6947[16] */ = B_fifo[5][5];	// L8370
  PE_kernel_gemm_5_4(v6944, v6945, v6946, v6947, v6658, 4, 5);	// L8371
  hls::stream< int8_t > &v6948 /* v6948[16] */ = A_fifo[4][6];	// L8372
  hls::stream< int8_t > &v6949 /* v6949[16] */ = B_fifo[6][4];	// L8373
  hls::stream< int8_t > &v6950 /* v6950[16] */ = A_fifo[4][7];	// L8374
  hls::stream< int8_t > &v6951 /* v6951[16] */ = B_fifo[6][5];	// L8375
  PE_kernel_gemm_6_4(v6948, v6949, v6950, v6951, v6658, 4, 6);	// L8376
  hls::stream< int8_t > &v6952 /* v6952[16] */ = A_fifo[4][7];	// L8377
  hls::stream< int8_t > &v6953 /* v6953[16] */ = B_fifo[7][4];	// L8378
  hls::stream< int8_t > &v6954 /* v6954[16] */ = A_fifo[4][8];	// L8379
  hls::stream< int8_t > &v6955 /* v6955[16] */ = B_fifo[7][5];	// L8380
  PE_kernel_gemm_7_4(v6952, v6953, v6954, v6955, v6658, 4, 7);	// L8381
  hls::stream< int8_t > &v6956 /* v6956[16] */ = A_fifo[4][8];	// L8382
  hls::stream< int8_t > &v6957 /* v6957[16] */ = B_fifo[8][4];	// L8383
  hls::stream< int8_t > &v6958 /* v6958[16] */ = A_fifo[4][9];	// L8384
  hls::stream< int8_t > &v6959 /* v6959[16] */ = B_fifo[8][5];	// L8385
  PE_kernel_gemm_8_4(v6956, v6957, v6958, v6959, v6658, 4, 8);	// L8386
  hls::stream< int8_t > &v6960 /* v6960[16] */ = A_fifo[4][9];	// L8387
  hls::stream< int8_t > &v6961 /* v6961[16] */ = B_fifo[9][4];	// L8388
  hls::stream< int8_t > &v6962 /* v6962[16] */ = A_fifo[4][10];	// L8389
  hls::stream< int8_t > &v6963 /* v6963[16] */ = B_fifo[9][5];	// L8390
  PE_kernel_gemm_9_4(v6960, v6961, v6962, v6963, v6658, 4, 9);	// L8391
  hls::stream< int8_t > &v6964 /* v6964[16] */ = A_fifo[4][10];	// L8392
  hls::stream< int8_t > &v6965 /* v6965[16] */ = B_fifo[10][4];	// L8393
  hls::stream< int8_t > &v6966 /* v6966[16] */ = A_fifo[4][11];	// L8394
  hls::stream< int8_t > &v6967 /* v6967[16] */ = B_fifo[10][5];	// L8395
  PE_kernel_gemm_10_4(v6964, v6965, v6966, v6967, v6658, 4, 10);	// L8396
  hls::stream< int8_t > &v6968 /* v6968[16] */ = A_fifo[4][11];	// L8397
  hls::stream< int8_t > &v6969 /* v6969[16] */ = B_fifo[11][4];	// L8398
  hls::stream< int8_t > &v6970 /* v6970[16] */ = A_fifo[4][12];	// L8399
  hls::stream< int8_t > &v6971 /* v6971[16] */ = B_fifo[11][5];	// L8400
  PE_kernel_gemm_11_4(v6968, v6969, v6970, v6971, v6658, 4, 11);	// L8401
  hls::stream< int8_t > &v6972 /* v6972[16] */ = A_fifo[4][12];	// L8402
  hls::stream< int8_t > &v6973 /* v6973[16] */ = B_fifo[12][4];	// L8403
  hls::stream< int8_t > &v6974 /* v6974[16] */ = A_fifo[4][13];	// L8404
  hls::stream< int8_t > &v6975 /* v6975[16] */ = B_fifo[12][5];	// L8405
  PE_kernel_gemm_12_4(v6972, v6973, v6974, v6975, v6658, 4, 12);	// L8406
  hls::stream< int8_t > &v6976 /* v6976[16] */ = A_fifo[4][13];	// L8407
  hls::stream< int8_t > &v6977 /* v6977[16] */ = B_fifo[13][4];	// L8408
  hls::stream< int8_t > &v6978 /* v6978[16] */ = A_fifo[4][14];	// L8409
  hls::stream< int8_t > &v6979 /* v6979[16] */ = B_fifo[13][5];	// L8410
  PE_kernel_gemm_13_4(v6976, v6977, v6978, v6979, v6658, 4, 13);	// L8411
  hls::stream< int8_t > &v6980 /* v6980[16] */ = A_fifo[4][14];	// L8412
  hls::stream< int8_t > &v6981 /* v6981[16] */ = B_fifo[14][4];	// L8413
  hls::stream< int8_t > &v6982 /* v6982[16] */ = A_fifo[4][15];	// L8414
  hls::stream< int8_t > &v6983 /* v6983[16] */ = B_fifo[14][5];	// L8415
  PE_kernel_gemm_14_4(v6980, v6981, v6982, v6983, v6658, 4, 14);	// L8416
  hls::stream< int8_t > &v6984 /* v6984[16] */ = A_fifo[4][15];	// L8417
  hls::stream< int8_t > &v6985 /* v6985[16] */ = B_fifo[15][4];	// L8418
  hls::stream< int8_t > &v6986 /* v6986[16] */ = A_fifo[4][16];	// L8419
  hls::stream< int8_t > &v6987 /* v6987[16] */ = B_fifo[15][5];	// L8420
  PE_kernel_gemm_15_4(v6984, v6985, v6986, v6987, v6658, 4, 15);	// L8421
  hls::stream< int8_t > &v6988 /* v6988[16] */ = A_fifo[5][0];	// L8422
  hls::stream< int8_t > &v6989 /* v6989[16] */ = B_fifo[0][5];	// L8423
  hls::stream< int8_t > &v6990 /* v6990[16] */ = A_fifo[5][1];	// L8424
  hls::stream< int8_t > &v6991 /* v6991[16] */ = B_fifo[0][6];	// L8425
  PE_kernel_gemm_0_5(v6988, v6989, v6990, v6991, v6658, 5, 0);	// L8426
  hls::stream< int8_t > &v6992 /* v6992[16] */ = A_fifo[5][1];	// L8427
  hls::stream< int8_t > &v6993 /* v6993[16] */ = B_fifo[1][5];	// L8428
  hls::stream< int8_t > &v6994 /* v6994[16] */ = A_fifo[5][2];	// L8429
  hls::stream< int8_t > &v6995 /* v6995[16] */ = B_fifo[1][6];	// L8430
  PE_kernel_gemm_1_5(v6992, v6993, v6994, v6995, v6658, 5, 1);	// L8431
  hls::stream< int8_t > &v6996 /* v6996[16] */ = A_fifo[5][2];	// L8432
  hls::stream< int8_t > &v6997 /* v6997[16] */ = B_fifo[2][5];	// L8433
  hls::stream< int8_t > &v6998 /* v6998[16] */ = A_fifo[5][3];	// L8434
  hls::stream< int8_t > &v6999 /* v6999[16] */ = B_fifo[2][6];	// L8435
  PE_kernel_gemm_2_5(v6996, v6997, v6998, v6999, v6658, 5, 2);	// L8436
  hls::stream< int8_t > &v7000 /* v7000[16] */ = A_fifo[5][3];	// L8437
  hls::stream< int8_t > &v7001 /* v7001[16] */ = B_fifo[3][5];	// L8438
  hls::stream< int8_t > &v7002 /* v7002[16] */ = A_fifo[5][4];	// L8439
  hls::stream< int8_t > &v7003 /* v7003[16] */ = B_fifo[3][6];	// L8440
  PE_kernel_gemm_3_5(v7000, v7001, v7002, v7003, v6658, 5, 3);	// L8441
  hls::stream< int8_t > &v7004 /* v7004[16] */ = A_fifo[5][4];	// L8442
  hls::stream< int8_t > &v7005 /* v7005[16] */ = B_fifo[4][5];	// L8443
  hls::stream< int8_t > &v7006 /* v7006[16] */ = A_fifo[5][5];	// L8444
  hls::stream< int8_t > &v7007 /* v7007[16] */ = B_fifo[4][6];	// L8445
  PE_kernel_gemm_4_5(v7004, v7005, v7006, v7007, v6658, 5, 4);	// L8446
  hls::stream< int8_t > &v7008 /* v7008[16] */ = A_fifo[5][5];	// L8447
  hls::stream< int8_t > &v7009 /* v7009[16] */ = B_fifo[5][5];	// L8448
  hls::stream< int8_t > &v7010 /* v7010[16] */ = A_fifo[5][6];	// L8449
  hls::stream< int8_t > &v7011 /* v7011[16] */ = B_fifo[5][6];	// L8450
  PE_kernel_gemm_5_5(v7008, v7009, v7010, v7011, v6658, 5, 5);	// L8451
  hls::stream< int8_t > &v7012 /* v7012[16] */ = A_fifo[5][6];	// L8452
  hls::stream< int8_t > &v7013 /* v7013[16] */ = B_fifo[6][5];	// L8453
  hls::stream< int8_t > &v7014 /* v7014[16] */ = A_fifo[5][7];	// L8454
  hls::stream< int8_t > &v7015 /* v7015[16] */ = B_fifo[6][6];	// L8455
  PE_kernel_gemm_6_5(v7012, v7013, v7014, v7015, v6658, 5, 6);	// L8456
  hls::stream< int8_t > &v7016 /* v7016[16] */ = A_fifo[5][7];	// L8457
  hls::stream< int8_t > &v7017 /* v7017[16] */ = B_fifo[7][5];	// L8458
  hls::stream< int8_t > &v7018 /* v7018[16] */ = A_fifo[5][8];	// L8459
  hls::stream< int8_t > &v7019 /* v7019[16] */ = B_fifo[7][6];	// L8460
  PE_kernel_gemm_7_5(v7016, v7017, v7018, v7019, v6658, 5, 7);	// L8461
  hls::stream< int8_t > &v7020 /* v7020[16] */ = A_fifo[5][8];	// L8462
  hls::stream< int8_t > &v7021 /* v7021[16] */ = B_fifo[8][5];	// L8463
  hls::stream< int8_t > &v7022 /* v7022[16] */ = A_fifo[5][9];	// L8464
  hls::stream< int8_t > &v7023 /* v7023[16] */ = B_fifo[8][6];	// L8465
  PE_kernel_gemm_8_5(v7020, v7021, v7022, v7023, v6658, 5, 8);	// L8466
  hls::stream< int8_t > &v7024 /* v7024[16] */ = A_fifo[5][9];	// L8467
  hls::stream< int8_t > &v7025 /* v7025[16] */ = B_fifo[9][5];	// L8468
  hls::stream< int8_t > &v7026 /* v7026[16] */ = A_fifo[5][10];	// L8469
  hls::stream< int8_t > &v7027 /* v7027[16] */ = B_fifo[9][6];	// L8470
  PE_kernel_gemm_9_5(v7024, v7025, v7026, v7027, v6658, 5, 9);	// L8471
  hls::stream< int8_t > &v7028 /* v7028[16] */ = A_fifo[5][10];	// L8472
  hls::stream< int8_t > &v7029 /* v7029[16] */ = B_fifo[10][5];	// L8473
  hls::stream< int8_t > &v7030 /* v7030[16] */ = A_fifo[5][11];	// L8474
  hls::stream< int8_t > &v7031 /* v7031[16] */ = B_fifo[10][6];	// L8475
  PE_kernel_gemm_10_5(v7028, v7029, v7030, v7031, v6658, 5, 10);	// L8476
  hls::stream< int8_t > &v7032 /* v7032[16] */ = A_fifo[5][11];	// L8477
  hls::stream< int8_t > &v7033 /* v7033[16] */ = B_fifo[11][5];	// L8478
  hls::stream< int8_t > &v7034 /* v7034[16] */ = A_fifo[5][12];	// L8479
  hls::stream< int8_t > &v7035 /* v7035[16] */ = B_fifo[11][6];	// L8480
  PE_kernel_gemm_11_5(v7032, v7033, v7034, v7035, v6658, 5, 11);	// L8481
  hls::stream< int8_t > &v7036 /* v7036[16] */ = A_fifo[5][12];	// L8482
  hls::stream< int8_t > &v7037 /* v7037[16] */ = B_fifo[12][5];	// L8483
  hls::stream< int8_t > &v7038 /* v7038[16] */ = A_fifo[5][13];	// L8484
  hls::stream< int8_t > &v7039 /* v7039[16] */ = B_fifo[12][6];	// L8485
  PE_kernel_gemm_12_5(v7036, v7037, v7038, v7039, v6658, 5, 12);	// L8486
  hls::stream< int8_t > &v7040 /* v7040[16] */ = A_fifo[5][13];	// L8487
  hls::stream< int8_t > &v7041 /* v7041[16] */ = B_fifo[13][5];	// L8488
  hls::stream< int8_t > &v7042 /* v7042[16] */ = A_fifo[5][14];	// L8489
  hls::stream< int8_t > &v7043 /* v7043[16] */ = B_fifo[13][6];	// L8490
  PE_kernel_gemm_13_5(v7040, v7041, v7042, v7043, v6658, 5, 13);	// L8491
  hls::stream< int8_t > &v7044 /* v7044[16] */ = A_fifo[5][14];	// L8492
  hls::stream< int8_t > &v7045 /* v7045[16] */ = B_fifo[14][5];	// L8493
  hls::stream< int8_t > &v7046 /* v7046[16] */ = A_fifo[5][15];	// L8494
  hls::stream< int8_t > &v7047 /* v7047[16] */ = B_fifo[14][6];	// L8495
  PE_kernel_gemm_14_5(v7044, v7045, v7046, v7047, v6658, 5, 14);	// L8496
  hls::stream< int8_t > &v7048 /* v7048[16] */ = A_fifo[5][15];	// L8497
  hls::stream< int8_t > &v7049 /* v7049[16] */ = B_fifo[15][5];	// L8498
  hls::stream< int8_t > &v7050 /* v7050[16] */ = A_fifo[5][16];	// L8499
  hls::stream< int8_t > &v7051 /* v7051[16] */ = B_fifo[15][6];	// L8500
  PE_kernel_gemm_15_5(v7048, v7049, v7050, v7051, v6658, 5, 15);	// L8501
  hls::stream< int8_t > &v7052 /* v7052[16] */ = A_fifo[6][0];	// L8502
  hls::stream< int8_t > &v7053 /* v7053[16] */ = B_fifo[0][6];	// L8503
  hls::stream< int8_t > &v7054 /* v7054[16] */ = A_fifo[6][1];	// L8504
  hls::stream< int8_t > &v7055 /* v7055[16] */ = B_fifo[0][7];	// L8505
  PE_kernel_gemm_0_6(v7052, v7053, v7054, v7055, v6658, 6, 0);	// L8506
  hls::stream< int8_t > &v7056 /* v7056[16] */ = A_fifo[6][1];	// L8507
  hls::stream< int8_t > &v7057 /* v7057[16] */ = B_fifo[1][6];	// L8508
  hls::stream< int8_t > &v7058 /* v7058[16] */ = A_fifo[6][2];	// L8509
  hls::stream< int8_t > &v7059 /* v7059[16] */ = B_fifo[1][7];	// L8510
  PE_kernel_gemm_1_6(v7056, v7057, v7058, v7059, v6658, 6, 1);	// L8511
  hls::stream< int8_t > &v7060 /* v7060[16] */ = A_fifo[6][2];	// L8512
  hls::stream< int8_t > &v7061 /* v7061[16] */ = B_fifo[2][6];	// L8513
  hls::stream< int8_t > &v7062 /* v7062[16] */ = A_fifo[6][3];	// L8514
  hls::stream< int8_t > &v7063 /* v7063[16] */ = B_fifo[2][7];	// L8515
  PE_kernel_gemm_2_6(v7060, v7061, v7062, v7063, v6658, 6, 2);	// L8516
  hls::stream< int8_t > &v7064 /* v7064[16] */ = A_fifo[6][3];	// L8517
  hls::stream< int8_t > &v7065 /* v7065[16] */ = B_fifo[3][6];	// L8518
  hls::stream< int8_t > &v7066 /* v7066[16] */ = A_fifo[6][4];	// L8519
  hls::stream< int8_t > &v7067 /* v7067[16] */ = B_fifo[3][7];	// L8520
  PE_kernel_gemm_3_6(v7064, v7065, v7066, v7067, v6658, 6, 3);	// L8521
  hls::stream< int8_t > &v7068 /* v7068[16] */ = A_fifo[6][4];	// L8522
  hls::stream< int8_t > &v7069 /* v7069[16] */ = B_fifo[4][6];	// L8523
  hls::stream< int8_t > &v7070 /* v7070[16] */ = A_fifo[6][5];	// L8524
  hls::stream< int8_t > &v7071 /* v7071[16] */ = B_fifo[4][7];	// L8525
  PE_kernel_gemm_4_6(v7068, v7069, v7070, v7071, v6658, 6, 4);	// L8526
  hls::stream< int8_t > &v7072 /* v7072[16] */ = A_fifo[6][5];	// L8527
  hls::stream< int8_t > &v7073 /* v7073[16] */ = B_fifo[5][6];	// L8528
  hls::stream< int8_t > &v7074 /* v7074[16] */ = A_fifo[6][6];	// L8529
  hls::stream< int8_t > &v7075 /* v7075[16] */ = B_fifo[5][7];	// L8530
  PE_kernel_gemm_5_6(v7072, v7073, v7074, v7075, v6658, 6, 5);	// L8531
  hls::stream< int8_t > &v7076 /* v7076[16] */ = A_fifo[6][6];	// L8532
  hls::stream< int8_t > &v7077 /* v7077[16] */ = B_fifo[6][6];	// L8533
  hls::stream< int8_t > &v7078 /* v7078[16] */ = A_fifo[6][7];	// L8534
  hls::stream< int8_t > &v7079 /* v7079[16] */ = B_fifo[6][7];	// L8535
  PE_kernel_gemm_6_6(v7076, v7077, v7078, v7079, v6658, 6, 6);	// L8536
  hls::stream< int8_t > &v7080 /* v7080[16] */ = A_fifo[6][7];	// L8537
  hls::stream< int8_t > &v7081 /* v7081[16] */ = B_fifo[7][6];	// L8538
  hls::stream< int8_t > &v7082 /* v7082[16] */ = A_fifo[6][8];	// L8539
  hls::stream< int8_t > &v7083 /* v7083[16] */ = B_fifo[7][7];	// L8540
  PE_kernel_gemm_7_6(v7080, v7081, v7082, v7083, v6658, 6, 7);	// L8541
  hls::stream< int8_t > &v7084 /* v7084[16] */ = A_fifo[6][8];	// L8542
  hls::stream< int8_t > &v7085 /* v7085[16] */ = B_fifo[8][6];	// L8543
  hls::stream< int8_t > &v7086 /* v7086[16] */ = A_fifo[6][9];	// L8544
  hls::stream< int8_t > &v7087 /* v7087[16] */ = B_fifo[8][7];	// L8545
  PE_kernel_gemm_8_6(v7084, v7085, v7086, v7087, v6658, 6, 8);	// L8546
  hls::stream< int8_t > &v7088 /* v7088[16] */ = A_fifo[6][9];	// L8547
  hls::stream< int8_t > &v7089 /* v7089[16] */ = B_fifo[9][6];	// L8548
  hls::stream< int8_t > &v7090 /* v7090[16] */ = A_fifo[6][10];	// L8549
  hls::stream< int8_t > &v7091 /* v7091[16] */ = B_fifo[9][7];	// L8550
  PE_kernel_gemm_9_6(v7088, v7089, v7090, v7091, v6658, 6, 9);	// L8551
  hls::stream< int8_t > &v7092 /* v7092[16] */ = A_fifo[6][10];	// L8552
  hls::stream< int8_t > &v7093 /* v7093[16] */ = B_fifo[10][6];	// L8553
  hls::stream< int8_t > &v7094 /* v7094[16] */ = A_fifo[6][11];	// L8554
  hls::stream< int8_t > &v7095 /* v7095[16] */ = B_fifo[10][7];	// L8555
  PE_kernel_gemm_10_6(v7092, v7093, v7094, v7095, v6658, 6, 10);	// L8556
  hls::stream< int8_t > &v7096 /* v7096[16] */ = A_fifo[6][11];	// L8557
  hls::stream< int8_t > &v7097 /* v7097[16] */ = B_fifo[11][6];	// L8558
  hls::stream< int8_t > &v7098 /* v7098[16] */ = A_fifo[6][12];	// L8559
  hls::stream< int8_t > &v7099 /* v7099[16] */ = B_fifo[11][7];	// L8560
  PE_kernel_gemm_11_6(v7096, v7097, v7098, v7099, v6658, 6, 11);	// L8561
  hls::stream< int8_t > &v7100 /* v7100[16] */ = A_fifo[6][12];	// L8562
  hls::stream< int8_t > &v7101 /* v7101[16] */ = B_fifo[12][6];	// L8563
  hls::stream< int8_t > &v7102 /* v7102[16] */ = A_fifo[6][13];	// L8564
  hls::stream< int8_t > &v7103 /* v7103[16] */ = B_fifo[12][7];	// L8565
  PE_kernel_gemm_12_6(v7100, v7101, v7102, v7103, v6658, 6, 12);	// L8566
  hls::stream< int8_t > &v7104 /* v7104[16] */ = A_fifo[6][13];	// L8567
  hls::stream< int8_t > &v7105 /* v7105[16] */ = B_fifo[13][6];	// L8568
  hls::stream< int8_t > &v7106 /* v7106[16] */ = A_fifo[6][14];	// L8569
  hls::stream< int8_t > &v7107 /* v7107[16] */ = B_fifo[13][7];	// L8570
  PE_kernel_gemm_13_6(v7104, v7105, v7106, v7107, v6658, 6, 13);	// L8571
  hls::stream< int8_t > &v7108 /* v7108[16] */ = A_fifo[6][14];	// L8572
  hls::stream< int8_t > &v7109 /* v7109[16] */ = B_fifo[14][6];	// L8573
  hls::stream< int8_t > &v7110 /* v7110[16] */ = A_fifo[6][15];	// L8574
  hls::stream< int8_t > &v7111 /* v7111[16] */ = B_fifo[14][7];	// L8575
  PE_kernel_gemm_14_6(v7108, v7109, v7110, v7111, v6658, 6, 14);	// L8576
  hls::stream< int8_t > &v7112 /* v7112[16] */ = A_fifo[6][15];	// L8577
  hls::stream< int8_t > &v7113 /* v7113[16] */ = B_fifo[15][6];	// L8578
  hls::stream< int8_t > &v7114 /* v7114[16] */ = A_fifo[6][16];	// L8579
  hls::stream< int8_t > &v7115 /* v7115[16] */ = B_fifo[15][7];	// L8580
  PE_kernel_gemm_15_6(v7112, v7113, v7114, v7115, v6658, 6, 15);	// L8581
  hls::stream< int8_t > &v7116 /* v7116[16] */ = A_fifo[7][0];	// L8582
  hls::stream< int8_t > &v7117 /* v7117[16] */ = B_fifo[0][7];	// L8583
  hls::stream< int8_t > &v7118 /* v7118[16] */ = A_fifo[7][1];	// L8584
  hls::stream< int8_t > &v7119 /* v7119[16] */ = B_fifo[0][8];	// L8585
  PE_kernel_gemm_0_7(v7116, v7117, v7118, v7119, v6658, 7, 0);	// L8586
  hls::stream< int8_t > &v7120 /* v7120[16] */ = A_fifo[7][1];	// L8587
  hls::stream< int8_t > &v7121 /* v7121[16] */ = B_fifo[1][7];	// L8588
  hls::stream< int8_t > &v7122 /* v7122[16] */ = A_fifo[7][2];	// L8589
  hls::stream< int8_t > &v7123 /* v7123[16] */ = B_fifo[1][8];	// L8590
  PE_kernel_gemm_1_7(v7120, v7121, v7122, v7123, v6658, 7, 1);	// L8591
  hls::stream< int8_t > &v7124 /* v7124[16] */ = A_fifo[7][2];	// L8592
  hls::stream< int8_t > &v7125 /* v7125[16] */ = B_fifo[2][7];	// L8593
  hls::stream< int8_t > &v7126 /* v7126[16] */ = A_fifo[7][3];	// L8594
  hls::stream< int8_t > &v7127 /* v7127[16] */ = B_fifo[2][8];	// L8595
  PE_kernel_gemm_2_7(v7124, v7125, v7126, v7127, v6658, 7, 2);	// L8596
  hls::stream< int8_t > &v7128 /* v7128[16] */ = A_fifo[7][3];	// L8597
  hls::stream< int8_t > &v7129 /* v7129[16] */ = B_fifo[3][7];	// L8598
  hls::stream< int8_t > &v7130 /* v7130[16] */ = A_fifo[7][4];	// L8599
  hls::stream< int8_t > &v7131 /* v7131[16] */ = B_fifo[3][8];	// L8600
  PE_kernel_gemm_3_7(v7128, v7129, v7130, v7131, v6658, 7, 3);	// L8601
  hls::stream< int8_t > &v7132 /* v7132[16] */ = A_fifo[7][4];	// L8602
  hls::stream< int8_t > &v7133 /* v7133[16] */ = B_fifo[4][7];	// L8603
  hls::stream< int8_t > &v7134 /* v7134[16] */ = A_fifo[7][5];	// L8604
  hls::stream< int8_t > &v7135 /* v7135[16] */ = B_fifo[4][8];	// L8605
  PE_kernel_gemm_4_7(v7132, v7133, v7134, v7135, v6658, 7, 4);	// L8606
  hls::stream< int8_t > &v7136 /* v7136[16] */ = A_fifo[7][5];	// L8607
  hls::stream< int8_t > &v7137 /* v7137[16] */ = B_fifo[5][7];	// L8608
  hls::stream< int8_t > &v7138 /* v7138[16] */ = A_fifo[7][6];	// L8609
  hls::stream< int8_t > &v7139 /* v7139[16] */ = B_fifo[5][8];	// L8610
  PE_kernel_gemm_5_7(v7136, v7137, v7138, v7139, v6658, 7, 5);	// L8611
  hls::stream< int8_t > &v7140 /* v7140[16] */ = A_fifo[7][6];	// L8612
  hls::stream< int8_t > &v7141 /* v7141[16] */ = B_fifo[6][7];	// L8613
  hls::stream< int8_t > &v7142 /* v7142[16] */ = A_fifo[7][7];	// L8614
  hls::stream< int8_t > &v7143 /* v7143[16] */ = B_fifo[6][8];	// L8615
  PE_kernel_gemm_6_7(v7140, v7141, v7142, v7143, v6658, 7, 6);	// L8616
  hls::stream< int8_t > &v7144 /* v7144[16] */ = A_fifo[7][7];	// L8617
  hls::stream< int8_t > &v7145 /* v7145[16] */ = B_fifo[7][7];	// L8618
  hls::stream< int8_t > &v7146 /* v7146[16] */ = A_fifo[7][8];	// L8619
  hls::stream< int8_t > &v7147 /* v7147[16] */ = B_fifo[7][8];	// L8620
  PE_kernel_gemm_7_7(v7144, v7145, v7146, v7147, v6658, 7, 7);	// L8621
  hls::stream< int8_t > &v7148 /* v7148[16] */ = A_fifo[7][8];	// L8622
  hls::stream< int8_t > &v7149 /* v7149[16] */ = B_fifo[8][7];	// L8623
  hls::stream< int8_t > &v7150 /* v7150[16] */ = A_fifo[7][9];	// L8624
  hls::stream< int8_t > &v7151 /* v7151[16] */ = B_fifo[8][8];	// L8625
  PE_kernel_gemm_8_7(v7148, v7149, v7150, v7151, v6658, 7, 8);	// L8626
  hls::stream< int8_t > &v7152 /* v7152[16] */ = A_fifo[7][9];	// L8627
  hls::stream< int8_t > &v7153 /* v7153[16] */ = B_fifo[9][7];	// L8628
  hls::stream< int8_t > &v7154 /* v7154[16] */ = A_fifo[7][10];	// L8629
  hls::stream< int8_t > &v7155 /* v7155[16] */ = B_fifo[9][8];	// L8630
  PE_kernel_gemm_9_7(v7152, v7153, v7154, v7155, v6658, 7, 9);	// L8631
  hls::stream< int8_t > &v7156 /* v7156[16] */ = A_fifo[7][10];	// L8632
  hls::stream< int8_t > &v7157 /* v7157[16] */ = B_fifo[10][7];	// L8633
  hls::stream< int8_t > &v7158 /* v7158[16] */ = A_fifo[7][11];	// L8634
  hls::stream< int8_t > &v7159 /* v7159[16] */ = B_fifo[10][8];	// L8635
  PE_kernel_gemm_10_7(v7156, v7157, v7158, v7159, v6658, 7, 10);	// L8636
  hls::stream< int8_t > &v7160 /* v7160[16] */ = A_fifo[7][11];	// L8637
  hls::stream< int8_t > &v7161 /* v7161[16] */ = B_fifo[11][7];	// L8638
  hls::stream< int8_t > &v7162 /* v7162[16] */ = A_fifo[7][12];	// L8639
  hls::stream< int8_t > &v7163 /* v7163[16] */ = B_fifo[11][8];	// L8640
  PE_kernel_gemm_11_7(v7160, v7161, v7162, v7163, v6658, 7, 11);	// L8641
  hls::stream< int8_t > &v7164 /* v7164[16] */ = A_fifo[7][12];	// L8642
  hls::stream< int8_t > &v7165 /* v7165[16] */ = B_fifo[12][7];	// L8643
  hls::stream< int8_t > &v7166 /* v7166[16] */ = A_fifo[7][13];	// L8644
  hls::stream< int8_t > &v7167 /* v7167[16] */ = B_fifo[12][8];	// L8645
  PE_kernel_gemm_12_7(v7164, v7165, v7166, v7167, v6658, 7, 12);	// L8646
  hls::stream< int8_t > &v7168 /* v7168[16] */ = A_fifo[7][13];	// L8647
  hls::stream< int8_t > &v7169 /* v7169[16] */ = B_fifo[13][7];	// L8648
  hls::stream< int8_t > &v7170 /* v7170[16] */ = A_fifo[7][14];	// L8649
  hls::stream< int8_t > &v7171 /* v7171[16] */ = B_fifo[13][8];	// L8650
  PE_kernel_gemm_13_7(v7168, v7169, v7170, v7171, v6658, 7, 13);	// L8651
  hls::stream< int8_t > &v7172 /* v7172[16] */ = A_fifo[7][14];	// L8652
  hls::stream< int8_t > &v7173 /* v7173[16] */ = B_fifo[14][7];	// L8653
  hls::stream< int8_t > &v7174 /* v7174[16] */ = A_fifo[7][15];	// L8654
  hls::stream< int8_t > &v7175 /* v7175[16] */ = B_fifo[14][8];	// L8655
  PE_kernel_gemm_14_7(v7172, v7173, v7174, v7175, v6658, 7, 14);	// L8656
  hls::stream< int8_t > &v7176 /* v7176[16] */ = A_fifo[7][15];	// L8657
  hls::stream< int8_t > &v7177 /* v7177[16] */ = B_fifo[15][7];	// L8658
  hls::stream< int8_t > &v7178 /* v7178[16] */ = A_fifo[7][16];	// L8659
  hls::stream< int8_t > &v7179 /* v7179[16] */ = B_fifo[15][8];	// L8660
  PE_kernel_gemm_15_7(v7176, v7177, v7178, v7179, v6658, 7, 15);	// L8661
  hls::stream< int8_t > &v7180 /* v7180[16] */ = A_fifo[8][0];	// L8662
  hls::stream< int8_t > &v7181 /* v7181[16] */ = B_fifo[0][8];	// L8663
  hls::stream< int8_t > &v7182 /* v7182[16] */ = A_fifo[8][1];	// L8664
  hls::stream< int8_t > &v7183 /* v7183[16] */ = B_fifo[0][9];	// L8665
  PE_kernel_gemm_0_8(v7180, v7181, v7182, v7183, v6658, 8, 0);	// L8666
  hls::stream< int8_t > &v7184 /* v7184[16] */ = A_fifo[8][1];	// L8667
  hls::stream< int8_t > &v7185 /* v7185[16] */ = B_fifo[1][8];	// L8668
  hls::stream< int8_t > &v7186 /* v7186[16] */ = A_fifo[8][2];	// L8669
  hls::stream< int8_t > &v7187 /* v7187[16] */ = B_fifo[1][9];	// L8670
  PE_kernel_gemm_1_8(v7184, v7185, v7186, v7187, v6658, 8, 1);	// L8671
  hls::stream< int8_t > &v7188 /* v7188[16] */ = A_fifo[8][2];	// L8672
  hls::stream< int8_t > &v7189 /* v7189[16] */ = B_fifo[2][8];	// L8673
  hls::stream< int8_t > &v7190 /* v7190[16] */ = A_fifo[8][3];	// L8674
  hls::stream< int8_t > &v7191 /* v7191[16] */ = B_fifo[2][9];	// L8675
  PE_kernel_gemm_2_8(v7188, v7189, v7190, v7191, v6658, 8, 2);	// L8676
  hls::stream< int8_t > &v7192 /* v7192[16] */ = A_fifo[8][3];	// L8677
  hls::stream< int8_t > &v7193 /* v7193[16] */ = B_fifo[3][8];	// L8678
  hls::stream< int8_t > &v7194 /* v7194[16] */ = A_fifo[8][4];	// L8679
  hls::stream< int8_t > &v7195 /* v7195[16] */ = B_fifo[3][9];	// L8680
  PE_kernel_gemm_3_8(v7192, v7193, v7194, v7195, v6658, 8, 3);	// L8681
  hls::stream< int8_t > &v7196 /* v7196[16] */ = A_fifo[8][4];	// L8682
  hls::stream< int8_t > &v7197 /* v7197[16] */ = B_fifo[4][8];	// L8683
  hls::stream< int8_t > &v7198 /* v7198[16] */ = A_fifo[8][5];	// L8684
  hls::stream< int8_t > &v7199 /* v7199[16] */ = B_fifo[4][9];	// L8685
  PE_kernel_gemm_4_8(v7196, v7197, v7198, v7199, v6658, 8, 4);	// L8686
  hls::stream< int8_t > &v7200 /* v7200[16] */ = A_fifo[8][5];	// L8687
  hls::stream< int8_t > &v7201 /* v7201[16] */ = B_fifo[5][8];	// L8688
  hls::stream< int8_t > &v7202 /* v7202[16] */ = A_fifo[8][6];	// L8689
  hls::stream< int8_t > &v7203 /* v7203[16] */ = B_fifo[5][9];	// L8690
  PE_kernel_gemm_5_8(v7200, v7201, v7202, v7203, v6658, 8, 5);	// L8691
  hls::stream< int8_t > &v7204 /* v7204[16] */ = A_fifo[8][6];	// L8692
  hls::stream< int8_t > &v7205 /* v7205[16] */ = B_fifo[6][8];	// L8693
  hls::stream< int8_t > &v7206 /* v7206[16] */ = A_fifo[8][7];	// L8694
  hls::stream< int8_t > &v7207 /* v7207[16] */ = B_fifo[6][9];	// L8695
  PE_kernel_gemm_6_8(v7204, v7205, v7206, v7207, v6658, 8, 6);	// L8696
  hls::stream< int8_t > &v7208 /* v7208[16] */ = A_fifo[8][7];	// L8697
  hls::stream< int8_t > &v7209 /* v7209[16] */ = B_fifo[7][8];	// L8698
  hls::stream< int8_t > &v7210 /* v7210[16] */ = A_fifo[8][8];	// L8699
  hls::stream< int8_t > &v7211 /* v7211[16] */ = B_fifo[7][9];	// L8700
  PE_kernel_gemm_7_8(v7208, v7209, v7210, v7211, v6658, 8, 7);	// L8701
  hls::stream< int8_t > &v7212 /* v7212[16] */ = A_fifo[8][8];	// L8702
  hls::stream< int8_t > &v7213 /* v7213[16] */ = B_fifo[8][8];	// L8703
  hls::stream< int8_t > &v7214 /* v7214[16] */ = A_fifo[8][9];	// L8704
  hls::stream< int8_t > &v7215 /* v7215[16] */ = B_fifo[8][9];	// L8705
  PE_kernel_gemm_8_8(v7212, v7213, v7214, v7215, v6658, 8, 8);	// L8706
  hls::stream< int8_t > &v7216 /* v7216[16] */ = A_fifo[8][9];	// L8707
  hls::stream< int8_t > &v7217 /* v7217[16] */ = B_fifo[9][8];	// L8708
  hls::stream< int8_t > &v7218 /* v7218[16] */ = A_fifo[8][10];	// L8709
  hls::stream< int8_t > &v7219 /* v7219[16] */ = B_fifo[9][9];	// L8710
  PE_kernel_gemm_9_8(v7216, v7217, v7218, v7219, v6658, 8, 9);	// L8711
  hls::stream< int8_t > &v7220 /* v7220[16] */ = A_fifo[8][10];	// L8712
  hls::stream< int8_t > &v7221 /* v7221[16] */ = B_fifo[10][8];	// L8713
  hls::stream< int8_t > &v7222 /* v7222[16] */ = A_fifo[8][11];	// L8714
  hls::stream< int8_t > &v7223 /* v7223[16] */ = B_fifo[10][9];	// L8715
  PE_kernel_gemm_10_8(v7220, v7221, v7222, v7223, v6658, 8, 10);	// L8716
  hls::stream< int8_t > &v7224 /* v7224[16] */ = A_fifo[8][11];	// L8717
  hls::stream< int8_t > &v7225 /* v7225[16] */ = B_fifo[11][8];	// L8718
  hls::stream< int8_t > &v7226 /* v7226[16] */ = A_fifo[8][12];	// L8719
  hls::stream< int8_t > &v7227 /* v7227[16] */ = B_fifo[11][9];	// L8720
  PE_kernel_gemm_11_8(v7224, v7225, v7226, v7227, v6658, 8, 11);	// L8721
  hls::stream< int8_t > &v7228 /* v7228[16] */ = A_fifo[8][12];	// L8722
  hls::stream< int8_t > &v7229 /* v7229[16] */ = B_fifo[12][8];	// L8723
  hls::stream< int8_t > &v7230 /* v7230[16] */ = A_fifo[8][13];	// L8724
  hls::stream< int8_t > &v7231 /* v7231[16] */ = B_fifo[12][9];	// L8725
  PE_kernel_gemm_12_8(v7228, v7229, v7230, v7231, v6658, 8, 12);	// L8726
  hls::stream< int8_t > &v7232 /* v7232[16] */ = A_fifo[8][13];	// L8727
  hls::stream< int8_t > &v7233 /* v7233[16] */ = B_fifo[13][8];	// L8728
  hls::stream< int8_t > &v7234 /* v7234[16] */ = A_fifo[8][14];	// L8729
  hls::stream< int8_t > &v7235 /* v7235[16] */ = B_fifo[13][9];	// L8730
  PE_kernel_gemm_13_8(v7232, v7233, v7234, v7235, v6658, 8, 13);	// L8731
  hls::stream< int8_t > &v7236 /* v7236[16] */ = A_fifo[8][14];	// L8732
  hls::stream< int8_t > &v7237 /* v7237[16] */ = B_fifo[14][8];	// L8733
  hls::stream< int8_t > &v7238 /* v7238[16] */ = A_fifo[8][15];	// L8734
  hls::stream< int8_t > &v7239 /* v7239[16] */ = B_fifo[14][9];	// L8735
  PE_kernel_gemm_14_8(v7236, v7237, v7238, v7239, v6658, 8, 14);	// L8736
  hls::stream< int8_t > &v7240 /* v7240[16] */ = A_fifo[8][15];	// L8737
  hls::stream< int8_t > &v7241 /* v7241[16] */ = B_fifo[15][8];	// L8738
  hls::stream< int8_t > &v7242 /* v7242[16] */ = A_fifo[8][16];	// L8739
  hls::stream< int8_t > &v7243 /* v7243[16] */ = B_fifo[15][9];	// L8740
  PE_kernel_gemm_15_8(v7240, v7241, v7242, v7243, v6658, 8, 15);	// L8741
  hls::stream< int8_t > &v7244 /* v7244[16] */ = A_fifo[9][0];	// L8742
  hls::stream< int8_t > &v7245 /* v7245[16] */ = B_fifo[0][9];	// L8743
  hls::stream< int8_t > &v7246 /* v7246[16] */ = A_fifo[9][1];	// L8744
  hls::stream< int8_t > &v7247 /* v7247[16] */ = B_fifo[0][10];	// L8745
  PE_kernel_gemm_0_9(v7244, v7245, v7246, v7247, v6658, 9, 0);	// L8746
  hls::stream< int8_t > &v7248 /* v7248[16] */ = A_fifo[9][1];	// L8747
  hls::stream< int8_t > &v7249 /* v7249[16] */ = B_fifo[1][9];	// L8748
  hls::stream< int8_t > &v7250 /* v7250[16] */ = A_fifo[9][2];	// L8749
  hls::stream< int8_t > &v7251 /* v7251[16] */ = B_fifo[1][10];	// L8750
  PE_kernel_gemm_1_9(v7248, v7249, v7250, v7251, v6658, 9, 1);	// L8751
  hls::stream< int8_t > &v7252 /* v7252[16] */ = A_fifo[9][2];	// L8752
  hls::stream< int8_t > &v7253 /* v7253[16] */ = B_fifo[2][9];	// L8753
  hls::stream< int8_t > &v7254 /* v7254[16] */ = A_fifo[9][3];	// L8754
  hls::stream< int8_t > &v7255 /* v7255[16] */ = B_fifo[2][10];	// L8755
  PE_kernel_gemm_2_9(v7252, v7253, v7254, v7255, v6658, 9, 2);	// L8756
  hls::stream< int8_t > &v7256 /* v7256[16] */ = A_fifo[9][3];	// L8757
  hls::stream< int8_t > &v7257 /* v7257[16] */ = B_fifo[3][9];	// L8758
  hls::stream< int8_t > &v7258 /* v7258[16] */ = A_fifo[9][4];	// L8759
  hls::stream< int8_t > &v7259 /* v7259[16] */ = B_fifo[3][10];	// L8760
  PE_kernel_gemm_3_9(v7256, v7257, v7258, v7259, v6658, 9, 3);	// L8761
  hls::stream< int8_t > &v7260 /* v7260[16] */ = A_fifo[9][4];	// L8762
  hls::stream< int8_t > &v7261 /* v7261[16] */ = B_fifo[4][9];	// L8763
  hls::stream< int8_t > &v7262 /* v7262[16] */ = A_fifo[9][5];	// L8764
  hls::stream< int8_t > &v7263 /* v7263[16] */ = B_fifo[4][10];	// L8765
  PE_kernel_gemm_4_9(v7260, v7261, v7262, v7263, v6658, 9, 4);	// L8766
  hls::stream< int8_t > &v7264 /* v7264[16] */ = A_fifo[9][5];	// L8767
  hls::stream< int8_t > &v7265 /* v7265[16] */ = B_fifo[5][9];	// L8768
  hls::stream< int8_t > &v7266 /* v7266[16] */ = A_fifo[9][6];	// L8769
  hls::stream< int8_t > &v7267 /* v7267[16] */ = B_fifo[5][10];	// L8770
  PE_kernel_gemm_5_9(v7264, v7265, v7266, v7267, v6658, 9, 5);	// L8771
  hls::stream< int8_t > &v7268 /* v7268[16] */ = A_fifo[9][6];	// L8772
  hls::stream< int8_t > &v7269 /* v7269[16] */ = B_fifo[6][9];	// L8773
  hls::stream< int8_t > &v7270 /* v7270[16] */ = A_fifo[9][7];	// L8774
  hls::stream< int8_t > &v7271 /* v7271[16] */ = B_fifo[6][10];	// L8775
  PE_kernel_gemm_6_9(v7268, v7269, v7270, v7271, v6658, 9, 6);	// L8776
  hls::stream< int8_t > &v7272 /* v7272[16] */ = A_fifo[9][7];	// L8777
  hls::stream< int8_t > &v7273 /* v7273[16] */ = B_fifo[7][9];	// L8778
  hls::stream< int8_t > &v7274 /* v7274[16] */ = A_fifo[9][8];	// L8779
  hls::stream< int8_t > &v7275 /* v7275[16] */ = B_fifo[7][10];	// L8780
  PE_kernel_gemm_7_9(v7272, v7273, v7274, v7275, v6658, 9, 7);	// L8781
  hls::stream< int8_t > &v7276 /* v7276[16] */ = A_fifo[9][8];	// L8782
  hls::stream< int8_t > &v7277 /* v7277[16] */ = B_fifo[8][9];	// L8783
  hls::stream< int8_t > &v7278 /* v7278[16] */ = A_fifo[9][9];	// L8784
  hls::stream< int8_t > &v7279 /* v7279[16] */ = B_fifo[8][10];	// L8785
  PE_kernel_gemm_8_9(v7276, v7277, v7278, v7279, v6658, 9, 8);	// L8786
  hls::stream< int8_t > &v7280 /* v7280[16] */ = A_fifo[9][9];	// L8787
  hls::stream< int8_t > &v7281 /* v7281[16] */ = B_fifo[9][9];	// L8788
  hls::stream< int8_t > &v7282 /* v7282[16] */ = A_fifo[9][10];	// L8789
  hls::stream< int8_t > &v7283 /* v7283[16] */ = B_fifo[9][10];	// L8790
  PE_kernel_gemm_9_9(v7280, v7281, v7282, v7283, v6658, 9, 9);	// L8791
  hls::stream< int8_t > &v7284 /* v7284[16] */ = A_fifo[9][10];	// L8792
  hls::stream< int8_t > &v7285 /* v7285[16] */ = B_fifo[10][9];	// L8793
  hls::stream< int8_t > &v7286 /* v7286[16] */ = A_fifo[9][11];	// L8794
  hls::stream< int8_t > &v7287 /* v7287[16] */ = B_fifo[10][10];	// L8795
  PE_kernel_gemm_10_9(v7284, v7285, v7286, v7287, v6658, 9, 10);	// L8796
  hls::stream< int8_t > &v7288 /* v7288[16] */ = A_fifo[9][11];	// L8797
  hls::stream< int8_t > &v7289 /* v7289[16] */ = B_fifo[11][9];	// L8798
  hls::stream< int8_t > &v7290 /* v7290[16] */ = A_fifo[9][12];	// L8799
  hls::stream< int8_t > &v7291 /* v7291[16] */ = B_fifo[11][10];	// L8800
  PE_kernel_gemm_11_9(v7288, v7289, v7290, v7291, v6658, 9, 11);	// L8801
  hls::stream< int8_t > &v7292 /* v7292[16] */ = A_fifo[9][12];	// L8802
  hls::stream< int8_t > &v7293 /* v7293[16] */ = B_fifo[12][9];	// L8803
  hls::stream< int8_t > &v7294 /* v7294[16] */ = A_fifo[9][13];	// L8804
  hls::stream< int8_t > &v7295 /* v7295[16] */ = B_fifo[12][10];	// L8805
  PE_kernel_gemm_12_9(v7292, v7293, v7294, v7295, v6658, 9, 12);	// L8806
  hls::stream< int8_t > &v7296 /* v7296[16] */ = A_fifo[9][13];	// L8807
  hls::stream< int8_t > &v7297 /* v7297[16] */ = B_fifo[13][9];	// L8808
  hls::stream< int8_t > &v7298 /* v7298[16] */ = A_fifo[9][14];	// L8809
  hls::stream< int8_t > &v7299 /* v7299[16] */ = B_fifo[13][10];	// L8810
  PE_kernel_gemm_13_9(v7296, v7297, v7298, v7299, v6658, 9, 13);	// L8811
  hls::stream< int8_t > &v7300 /* v7300[16] */ = A_fifo[9][14];	// L8812
  hls::stream< int8_t > &v7301 /* v7301[16] */ = B_fifo[14][9];	// L8813
  hls::stream< int8_t > &v7302 /* v7302[16] */ = A_fifo[9][15];	// L8814
  hls::stream< int8_t > &v7303 /* v7303[16] */ = B_fifo[14][10];	// L8815
  PE_kernel_gemm_14_9(v7300, v7301, v7302, v7303, v6658, 9, 14);	// L8816
  hls::stream< int8_t > &v7304 /* v7304[16] */ = A_fifo[9][15];	// L8817
  hls::stream< int8_t > &v7305 /* v7305[16] */ = B_fifo[15][9];	// L8818
  hls::stream< int8_t > &v7306 /* v7306[16] */ = A_fifo[9][16];	// L8819
  hls::stream< int8_t > &v7307 /* v7307[16] */ = B_fifo[15][10];	// L8820
  PE_kernel_gemm_15_9(v7304, v7305, v7306, v7307, v6658, 9, 15);	// L8821
  hls::stream< int8_t > &v7308 /* v7308[16] */ = A_fifo[10][0];	// L8822
  hls::stream< int8_t > &v7309 /* v7309[16] */ = B_fifo[0][10];	// L8823
  hls::stream< int8_t > &v7310 /* v7310[16] */ = A_fifo[10][1];	// L8824
  hls::stream< int8_t > &v7311 /* v7311[16] */ = B_fifo[0][11];	// L8825
  PE_kernel_gemm_0_10(v7308, v7309, v7310, v7311, v6658, 10, 0);	// L8826
  hls::stream< int8_t > &v7312 /* v7312[16] */ = A_fifo[10][1];	// L8827
  hls::stream< int8_t > &v7313 /* v7313[16] */ = B_fifo[1][10];	// L8828
  hls::stream< int8_t > &v7314 /* v7314[16] */ = A_fifo[10][2];	// L8829
  hls::stream< int8_t > &v7315 /* v7315[16] */ = B_fifo[1][11];	// L8830
  PE_kernel_gemm_1_10(v7312, v7313, v7314, v7315, v6658, 10, 1);	// L8831
  hls::stream< int8_t > &v7316 /* v7316[16] */ = A_fifo[10][2];	// L8832
  hls::stream< int8_t > &v7317 /* v7317[16] */ = B_fifo[2][10];	// L8833
  hls::stream< int8_t > &v7318 /* v7318[16] */ = A_fifo[10][3];	// L8834
  hls::stream< int8_t > &v7319 /* v7319[16] */ = B_fifo[2][11];	// L8835
  PE_kernel_gemm_2_10(v7316, v7317, v7318, v7319, v6658, 10, 2);	// L8836
  hls::stream< int8_t > &v7320 /* v7320[16] */ = A_fifo[10][3];	// L8837
  hls::stream< int8_t > &v7321 /* v7321[16] */ = B_fifo[3][10];	// L8838
  hls::stream< int8_t > &v7322 /* v7322[16] */ = A_fifo[10][4];	// L8839
  hls::stream< int8_t > &v7323 /* v7323[16] */ = B_fifo[3][11];	// L8840
  PE_kernel_gemm_3_10(v7320, v7321, v7322, v7323, v6658, 10, 3);	// L8841
  hls::stream< int8_t > &v7324 /* v7324[16] */ = A_fifo[10][4];	// L8842
  hls::stream< int8_t > &v7325 /* v7325[16] */ = B_fifo[4][10];	// L8843
  hls::stream< int8_t > &v7326 /* v7326[16] */ = A_fifo[10][5];	// L8844
  hls::stream< int8_t > &v7327 /* v7327[16] */ = B_fifo[4][11];	// L8845
  PE_kernel_gemm_4_10(v7324, v7325, v7326, v7327, v6658, 10, 4);	// L8846
  hls::stream< int8_t > &v7328 /* v7328[16] */ = A_fifo[10][5];	// L8847
  hls::stream< int8_t > &v7329 /* v7329[16] */ = B_fifo[5][10];	// L8848
  hls::stream< int8_t > &v7330 /* v7330[16] */ = A_fifo[10][6];	// L8849
  hls::stream< int8_t > &v7331 /* v7331[16] */ = B_fifo[5][11];	// L8850
  PE_kernel_gemm_5_10(v7328, v7329, v7330, v7331, v6658, 10, 5);	// L8851
  hls::stream< int8_t > &v7332 /* v7332[16] */ = A_fifo[10][6];	// L8852
  hls::stream< int8_t > &v7333 /* v7333[16] */ = B_fifo[6][10];	// L8853
  hls::stream< int8_t > &v7334 /* v7334[16] */ = A_fifo[10][7];	// L8854
  hls::stream< int8_t > &v7335 /* v7335[16] */ = B_fifo[6][11];	// L8855
  PE_kernel_gemm_6_10(v7332, v7333, v7334, v7335, v6658, 10, 6);	// L8856
  hls::stream< int8_t > &v7336 /* v7336[16] */ = A_fifo[10][7];	// L8857
  hls::stream< int8_t > &v7337 /* v7337[16] */ = B_fifo[7][10];	// L8858
  hls::stream< int8_t > &v7338 /* v7338[16] */ = A_fifo[10][8];	// L8859
  hls::stream< int8_t > &v7339 /* v7339[16] */ = B_fifo[7][11];	// L8860
  PE_kernel_gemm_7_10(v7336, v7337, v7338, v7339, v6658, 10, 7);	// L8861
  hls::stream< int8_t > &v7340 /* v7340[16] */ = A_fifo[10][8];	// L8862
  hls::stream< int8_t > &v7341 /* v7341[16] */ = B_fifo[8][10];	// L8863
  hls::stream< int8_t > &v7342 /* v7342[16] */ = A_fifo[10][9];	// L8864
  hls::stream< int8_t > &v7343 /* v7343[16] */ = B_fifo[8][11];	// L8865
  PE_kernel_gemm_8_10(v7340, v7341, v7342, v7343, v6658, 10, 8);	// L8866
  hls::stream< int8_t > &v7344 /* v7344[16] */ = A_fifo[10][9];	// L8867
  hls::stream< int8_t > &v7345 /* v7345[16] */ = B_fifo[9][10];	// L8868
  hls::stream< int8_t > &v7346 /* v7346[16] */ = A_fifo[10][10];	// L8869
  hls::stream< int8_t > &v7347 /* v7347[16] */ = B_fifo[9][11];	// L8870
  PE_kernel_gemm_9_10(v7344, v7345, v7346, v7347, v6658, 10, 9);	// L8871
  hls::stream< int8_t > &v7348 /* v7348[16] */ = A_fifo[10][10];	// L8872
  hls::stream< int8_t > &v7349 /* v7349[16] */ = B_fifo[10][10];	// L8873
  hls::stream< int8_t > &v7350 /* v7350[16] */ = A_fifo[10][11];	// L8874
  hls::stream< int8_t > &v7351 /* v7351[16] */ = B_fifo[10][11];	// L8875
  PE_kernel_gemm_10_10(v7348, v7349, v7350, v7351, v6658, 10, 10);	// L8876
  hls::stream< int8_t > &v7352 /* v7352[16] */ = A_fifo[10][11];	// L8877
  hls::stream< int8_t > &v7353 /* v7353[16] */ = B_fifo[11][10];	// L8878
  hls::stream< int8_t > &v7354 /* v7354[16] */ = A_fifo[10][12];	// L8879
  hls::stream< int8_t > &v7355 /* v7355[16] */ = B_fifo[11][11];	// L8880
  PE_kernel_gemm_11_10(v7352, v7353, v7354, v7355, v6658, 10, 11);	// L8881
  hls::stream< int8_t > &v7356 /* v7356[16] */ = A_fifo[10][12];	// L8882
  hls::stream< int8_t > &v7357 /* v7357[16] */ = B_fifo[12][10];	// L8883
  hls::stream< int8_t > &v7358 /* v7358[16] */ = A_fifo[10][13];	// L8884
  hls::stream< int8_t > &v7359 /* v7359[16] */ = B_fifo[12][11];	// L8885
  PE_kernel_gemm_12_10(v7356, v7357, v7358, v7359, v6658, 10, 12);	// L8886
  hls::stream< int8_t > &v7360 /* v7360[16] */ = A_fifo[10][13];	// L8887
  hls::stream< int8_t > &v7361 /* v7361[16] */ = B_fifo[13][10];	// L8888
  hls::stream< int8_t > &v7362 /* v7362[16] */ = A_fifo[10][14];	// L8889
  hls::stream< int8_t > &v7363 /* v7363[16] */ = B_fifo[13][11];	// L8890
  PE_kernel_gemm_13_10(v7360, v7361, v7362, v7363, v6658, 10, 13);	// L8891
  hls::stream< int8_t > &v7364 /* v7364[16] */ = A_fifo[10][14];	// L8892
  hls::stream< int8_t > &v7365 /* v7365[16] */ = B_fifo[14][10];	// L8893
  hls::stream< int8_t > &v7366 /* v7366[16] */ = A_fifo[10][15];	// L8894
  hls::stream< int8_t > &v7367 /* v7367[16] */ = B_fifo[14][11];	// L8895
  PE_kernel_gemm_14_10(v7364, v7365, v7366, v7367, v6658, 10, 14);	// L8896
  hls::stream< int8_t > &v7368 /* v7368[16] */ = A_fifo[10][15];	// L8897
  hls::stream< int8_t > &v7369 /* v7369[16] */ = B_fifo[15][10];	// L8898
  hls::stream< int8_t > &v7370 /* v7370[16] */ = A_fifo[10][16];	// L8899
  hls::stream< int8_t > &v7371 /* v7371[16] */ = B_fifo[15][11];	// L8900
  PE_kernel_gemm_15_10(v7368, v7369, v7370, v7371, v6658, 10, 15);	// L8901
  hls::stream< int8_t > &v7372 /* v7372[16] */ = A_fifo[11][0];	// L8902
  hls::stream< int8_t > &v7373 /* v7373[16] */ = B_fifo[0][11];	// L8903
  hls::stream< int8_t > &v7374 /* v7374[16] */ = A_fifo[11][1];	// L8904
  hls::stream< int8_t > &v7375 /* v7375[16] */ = B_fifo[0][12];	// L8905
  PE_kernel_gemm_0_11(v7372, v7373, v7374, v7375, v6658, 11, 0);	// L8906
  hls::stream< int8_t > &v7376 /* v7376[16] */ = A_fifo[11][1];	// L8907
  hls::stream< int8_t > &v7377 /* v7377[16] */ = B_fifo[1][11];	// L8908
  hls::stream< int8_t > &v7378 /* v7378[16] */ = A_fifo[11][2];	// L8909
  hls::stream< int8_t > &v7379 /* v7379[16] */ = B_fifo[1][12];	// L8910
  PE_kernel_gemm_1_11(v7376, v7377, v7378, v7379, v6658, 11, 1);	// L8911
  hls::stream< int8_t > &v7380 /* v7380[16] */ = A_fifo[11][2];	// L8912
  hls::stream< int8_t > &v7381 /* v7381[16] */ = B_fifo[2][11];	// L8913
  hls::stream< int8_t > &v7382 /* v7382[16] */ = A_fifo[11][3];	// L8914
  hls::stream< int8_t > &v7383 /* v7383[16] */ = B_fifo[2][12];	// L8915
  PE_kernel_gemm_2_11(v7380, v7381, v7382, v7383, v6658, 11, 2);	// L8916
  hls::stream< int8_t > &v7384 /* v7384[16] */ = A_fifo[11][3];	// L8917
  hls::stream< int8_t > &v7385 /* v7385[16] */ = B_fifo[3][11];	// L8918
  hls::stream< int8_t > &v7386 /* v7386[16] */ = A_fifo[11][4];	// L8919
  hls::stream< int8_t > &v7387 /* v7387[16] */ = B_fifo[3][12];	// L8920
  PE_kernel_gemm_3_11(v7384, v7385, v7386, v7387, v6658, 11, 3);	// L8921
  hls::stream< int8_t > &v7388 /* v7388[16] */ = A_fifo[11][4];	// L8922
  hls::stream< int8_t > &v7389 /* v7389[16] */ = B_fifo[4][11];	// L8923
  hls::stream< int8_t > &v7390 /* v7390[16] */ = A_fifo[11][5];	// L8924
  hls::stream< int8_t > &v7391 /* v7391[16] */ = B_fifo[4][12];	// L8925
  PE_kernel_gemm_4_11(v7388, v7389, v7390, v7391, v6658, 11, 4);	// L8926
  hls::stream< int8_t > &v7392 /* v7392[16] */ = A_fifo[11][5];	// L8927
  hls::stream< int8_t > &v7393 /* v7393[16] */ = B_fifo[5][11];	// L8928
  hls::stream< int8_t > &v7394 /* v7394[16] */ = A_fifo[11][6];	// L8929
  hls::stream< int8_t > &v7395 /* v7395[16] */ = B_fifo[5][12];	// L8930
  PE_kernel_gemm_5_11(v7392, v7393, v7394, v7395, v6658, 11, 5);	// L8931
  hls::stream< int8_t > &v7396 /* v7396[16] */ = A_fifo[11][6];	// L8932
  hls::stream< int8_t > &v7397 /* v7397[16] */ = B_fifo[6][11];	// L8933
  hls::stream< int8_t > &v7398 /* v7398[16] */ = A_fifo[11][7];	// L8934
  hls::stream< int8_t > &v7399 /* v7399[16] */ = B_fifo[6][12];	// L8935
  PE_kernel_gemm_6_11(v7396, v7397, v7398, v7399, v6658, 11, 6);	// L8936
  hls::stream< int8_t > &v7400 /* v7400[16] */ = A_fifo[11][7];	// L8937
  hls::stream< int8_t > &v7401 /* v7401[16] */ = B_fifo[7][11];	// L8938
  hls::stream< int8_t > &v7402 /* v7402[16] */ = A_fifo[11][8];	// L8939
  hls::stream< int8_t > &v7403 /* v7403[16] */ = B_fifo[7][12];	// L8940
  PE_kernel_gemm_7_11(v7400, v7401, v7402, v7403, v6658, 11, 7);	// L8941
  hls::stream< int8_t > &v7404 /* v7404[16] */ = A_fifo[11][8];	// L8942
  hls::stream< int8_t > &v7405 /* v7405[16] */ = B_fifo[8][11];	// L8943
  hls::stream< int8_t > &v7406 /* v7406[16] */ = A_fifo[11][9];	// L8944
  hls::stream< int8_t > &v7407 /* v7407[16] */ = B_fifo[8][12];	// L8945
  PE_kernel_gemm_8_11(v7404, v7405, v7406, v7407, v6658, 11, 8);	// L8946
  hls::stream< int8_t > &v7408 /* v7408[16] */ = A_fifo[11][9];	// L8947
  hls::stream< int8_t > &v7409 /* v7409[16] */ = B_fifo[9][11];	// L8948
  hls::stream< int8_t > &v7410 /* v7410[16] */ = A_fifo[11][10];	// L8949
  hls::stream< int8_t > &v7411 /* v7411[16] */ = B_fifo[9][12];	// L8950
  PE_kernel_gemm_9_11(v7408, v7409, v7410, v7411, v6658, 11, 9);	// L8951
  hls::stream< int8_t > &v7412 /* v7412[16] */ = A_fifo[11][10];	// L8952
  hls::stream< int8_t > &v7413 /* v7413[16] */ = B_fifo[10][11];	// L8953
  hls::stream< int8_t > &v7414 /* v7414[16] */ = A_fifo[11][11];	// L8954
  hls::stream< int8_t > &v7415 /* v7415[16] */ = B_fifo[10][12];	// L8955
  PE_kernel_gemm_10_11(v7412, v7413, v7414, v7415, v6658, 11, 10);	// L8956
  hls::stream< int8_t > &v7416 /* v7416[16] */ = A_fifo[11][11];	// L8957
  hls::stream< int8_t > &v7417 /* v7417[16] */ = B_fifo[11][11];	// L8958
  hls::stream< int8_t > &v7418 /* v7418[16] */ = A_fifo[11][12];	// L8959
  hls::stream< int8_t > &v7419 /* v7419[16] */ = B_fifo[11][12];	// L8960
  PE_kernel_gemm_11_11(v7416, v7417, v7418, v7419, v6658, 11, 11);	// L8961
  hls::stream< int8_t > &v7420 /* v7420[16] */ = A_fifo[11][12];	// L8962
  hls::stream< int8_t > &v7421 /* v7421[16] */ = B_fifo[12][11];	// L8963
  hls::stream< int8_t > &v7422 /* v7422[16] */ = A_fifo[11][13];	// L8964
  hls::stream< int8_t > &v7423 /* v7423[16] */ = B_fifo[12][12];	// L8965
  PE_kernel_gemm_12_11(v7420, v7421, v7422, v7423, v6658, 11, 12);	// L8966
  hls::stream< int8_t > &v7424 /* v7424[16] */ = A_fifo[11][13];	// L8967
  hls::stream< int8_t > &v7425 /* v7425[16] */ = B_fifo[13][11];	// L8968
  hls::stream< int8_t > &v7426 /* v7426[16] */ = A_fifo[11][14];	// L8969
  hls::stream< int8_t > &v7427 /* v7427[16] */ = B_fifo[13][12];	// L8970
  PE_kernel_gemm_13_11(v7424, v7425, v7426, v7427, v6658, 11, 13);	// L8971
  hls::stream< int8_t > &v7428 /* v7428[16] */ = A_fifo[11][14];	// L8972
  hls::stream< int8_t > &v7429 /* v7429[16] */ = B_fifo[14][11];	// L8973
  hls::stream< int8_t > &v7430 /* v7430[16] */ = A_fifo[11][15];	// L8974
  hls::stream< int8_t > &v7431 /* v7431[16] */ = B_fifo[14][12];	// L8975
  PE_kernel_gemm_14_11(v7428, v7429, v7430, v7431, v6658, 11, 14);	// L8976
  hls::stream< int8_t > &v7432 /* v7432[16] */ = A_fifo[11][15];	// L8977
  hls::stream< int8_t > &v7433 /* v7433[16] */ = B_fifo[15][11];	// L8978
  hls::stream< int8_t > &v7434 /* v7434[16] */ = A_fifo[11][16];	// L8979
  hls::stream< int8_t > &v7435 /* v7435[16] */ = B_fifo[15][12];	// L8980
  PE_kernel_gemm_15_11(v7432, v7433, v7434, v7435, v6658, 11, 15);	// L8981
  hls::stream< int8_t > &v7436 /* v7436[16] */ = A_fifo[12][0];	// L8982
  hls::stream< int8_t > &v7437 /* v7437[16] */ = B_fifo[0][12];	// L8983
  hls::stream< int8_t > &v7438 /* v7438[16] */ = A_fifo[12][1];	// L8984
  hls::stream< int8_t > &v7439 /* v7439[16] */ = B_fifo[0][13];	// L8985
  PE_kernel_gemm_0_12(v7436, v7437, v7438, v7439, v6658, 12, 0);	// L8986
  hls::stream< int8_t > &v7440 /* v7440[16] */ = A_fifo[12][1];	// L8987
  hls::stream< int8_t > &v7441 /* v7441[16] */ = B_fifo[1][12];	// L8988
  hls::stream< int8_t > &v7442 /* v7442[16] */ = A_fifo[12][2];	// L8989
  hls::stream< int8_t > &v7443 /* v7443[16] */ = B_fifo[1][13];	// L8990
  PE_kernel_gemm_1_12(v7440, v7441, v7442, v7443, v6658, 12, 1);	// L8991
  hls::stream< int8_t > &v7444 /* v7444[16] */ = A_fifo[12][2];	// L8992
  hls::stream< int8_t > &v7445 /* v7445[16] */ = B_fifo[2][12];	// L8993
  hls::stream< int8_t > &v7446 /* v7446[16] */ = A_fifo[12][3];	// L8994
  hls::stream< int8_t > &v7447 /* v7447[16] */ = B_fifo[2][13];	// L8995
  PE_kernel_gemm_2_12(v7444, v7445, v7446, v7447, v6658, 12, 2);	// L8996
  hls::stream< int8_t > &v7448 /* v7448[16] */ = A_fifo[12][3];	// L8997
  hls::stream< int8_t > &v7449 /* v7449[16] */ = B_fifo[3][12];	// L8998
  hls::stream< int8_t > &v7450 /* v7450[16] */ = A_fifo[12][4];	// L8999
  hls::stream< int8_t > &v7451 /* v7451[16] */ = B_fifo[3][13];	// L9000
  PE_kernel_gemm_3_12(v7448, v7449, v7450, v7451, v6658, 12, 3);	// L9001
  hls::stream< int8_t > &v7452 /* v7452[16] */ = A_fifo[12][4];	// L9002
  hls::stream< int8_t > &v7453 /* v7453[16] */ = B_fifo[4][12];	// L9003
  hls::stream< int8_t > &v7454 /* v7454[16] */ = A_fifo[12][5];	// L9004
  hls::stream< int8_t > &v7455 /* v7455[16] */ = B_fifo[4][13];	// L9005
  PE_kernel_gemm_4_12(v7452, v7453, v7454, v7455, v6658, 12, 4);	// L9006
  hls::stream< int8_t > &v7456 /* v7456[16] */ = A_fifo[12][5];	// L9007
  hls::stream< int8_t > &v7457 /* v7457[16] */ = B_fifo[5][12];	// L9008
  hls::stream< int8_t > &v7458 /* v7458[16] */ = A_fifo[12][6];	// L9009
  hls::stream< int8_t > &v7459 /* v7459[16] */ = B_fifo[5][13];	// L9010
  PE_kernel_gemm_5_12(v7456, v7457, v7458, v7459, v6658, 12, 5);	// L9011
  hls::stream< int8_t > &v7460 /* v7460[16] */ = A_fifo[12][6];	// L9012
  hls::stream< int8_t > &v7461 /* v7461[16] */ = B_fifo[6][12];	// L9013
  hls::stream< int8_t > &v7462 /* v7462[16] */ = A_fifo[12][7];	// L9014
  hls::stream< int8_t > &v7463 /* v7463[16] */ = B_fifo[6][13];	// L9015
  PE_kernel_gemm_6_12(v7460, v7461, v7462, v7463, v6658, 12, 6);	// L9016
  hls::stream< int8_t > &v7464 /* v7464[16] */ = A_fifo[12][7];	// L9017
  hls::stream< int8_t > &v7465 /* v7465[16] */ = B_fifo[7][12];	// L9018
  hls::stream< int8_t > &v7466 /* v7466[16] */ = A_fifo[12][8];	// L9019
  hls::stream< int8_t > &v7467 /* v7467[16] */ = B_fifo[7][13];	// L9020
  PE_kernel_gemm_7_12(v7464, v7465, v7466, v7467, v6658, 12, 7);	// L9021
  hls::stream< int8_t > &v7468 /* v7468[16] */ = A_fifo[12][8];	// L9022
  hls::stream< int8_t > &v7469 /* v7469[16] */ = B_fifo[8][12];	// L9023
  hls::stream< int8_t > &v7470 /* v7470[16] */ = A_fifo[12][9];	// L9024
  hls::stream< int8_t > &v7471 /* v7471[16] */ = B_fifo[8][13];	// L9025
  PE_kernel_gemm_8_12(v7468, v7469, v7470, v7471, v6658, 12, 8);	// L9026
  hls::stream< int8_t > &v7472 /* v7472[16] */ = A_fifo[12][9];	// L9027
  hls::stream< int8_t > &v7473 /* v7473[16] */ = B_fifo[9][12];	// L9028
  hls::stream< int8_t > &v7474 /* v7474[16] */ = A_fifo[12][10];	// L9029
  hls::stream< int8_t > &v7475 /* v7475[16] */ = B_fifo[9][13];	// L9030
  PE_kernel_gemm_9_12(v7472, v7473, v7474, v7475, v6658, 12, 9);	// L9031
  hls::stream< int8_t > &v7476 /* v7476[16] */ = A_fifo[12][10];	// L9032
  hls::stream< int8_t > &v7477 /* v7477[16] */ = B_fifo[10][12];	// L9033
  hls::stream< int8_t > &v7478 /* v7478[16] */ = A_fifo[12][11];	// L9034
  hls::stream< int8_t > &v7479 /* v7479[16] */ = B_fifo[10][13];	// L9035
  PE_kernel_gemm_10_12(v7476, v7477, v7478, v7479, v6658, 12, 10);	// L9036
  hls::stream< int8_t > &v7480 /* v7480[16] */ = A_fifo[12][11];	// L9037
  hls::stream< int8_t > &v7481 /* v7481[16] */ = B_fifo[11][12];	// L9038
  hls::stream< int8_t > &v7482 /* v7482[16] */ = A_fifo[12][12];	// L9039
  hls::stream< int8_t > &v7483 /* v7483[16] */ = B_fifo[11][13];	// L9040
  PE_kernel_gemm_11_12(v7480, v7481, v7482, v7483, v6658, 12, 11);	// L9041
  hls::stream< int8_t > &v7484 /* v7484[16] */ = A_fifo[12][12];	// L9042
  hls::stream< int8_t > &v7485 /* v7485[16] */ = B_fifo[12][12];	// L9043
  hls::stream< int8_t > &v7486 /* v7486[16] */ = A_fifo[12][13];	// L9044
  hls::stream< int8_t > &v7487 /* v7487[16] */ = B_fifo[12][13];	// L9045
  PE_kernel_gemm_12_12(v7484, v7485, v7486, v7487, v6658, 12, 12);	// L9046
  hls::stream< int8_t > &v7488 /* v7488[16] */ = A_fifo[12][13];	// L9047
  hls::stream< int8_t > &v7489 /* v7489[16] */ = B_fifo[13][12];	// L9048
  hls::stream< int8_t > &v7490 /* v7490[16] */ = A_fifo[12][14];	// L9049
  hls::stream< int8_t > &v7491 /* v7491[16] */ = B_fifo[13][13];	// L9050
  PE_kernel_gemm_13_12(v7488, v7489, v7490, v7491, v6658, 12, 13);	// L9051
  hls::stream< int8_t > &v7492 /* v7492[16] */ = A_fifo[12][14];	// L9052
  hls::stream< int8_t > &v7493 /* v7493[16] */ = B_fifo[14][12];	// L9053
  hls::stream< int8_t > &v7494 /* v7494[16] */ = A_fifo[12][15];	// L9054
  hls::stream< int8_t > &v7495 /* v7495[16] */ = B_fifo[14][13];	// L9055
  PE_kernel_gemm_14_12(v7492, v7493, v7494, v7495, v6658, 12, 14);	// L9056
  hls::stream< int8_t > &v7496 /* v7496[16] */ = A_fifo[12][15];	// L9057
  hls::stream< int8_t > &v7497 /* v7497[16] */ = B_fifo[15][12];	// L9058
  hls::stream< int8_t > &v7498 /* v7498[16] */ = A_fifo[12][16];	// L9059
  hls::stream< int8_t > &v7499 /* v7499[16] */ = B_fifo[15][13];	// L9060
  PE_kernel_gemm_15_12(v7496, v7497, v7498, v7499, v6658, 12, 15);	// L9061
  hls::stream< int8_t > &v7500 /* v7500[16] */ = A_fifo[13][0];	// L9062
  hls::stream< int8_t > &v7501 /* v7501[16] */ = B_fifo[0][13];	// L9063
  hls::stream< int8_t > &v7502 /* v7502[16] */ = A_fifo[13][1];	// L9064
  hls::stream< int8_t > &v7503 /* v7503[16] */ = B_fifo[0][14];	// L9065
  PE_kernel_gemm_0_13(v7500, v7501, v7502, v7503, v6658, 13, 0);	// L9066
  hls::stream< int8_t > &v7504 /* v7504[16] */ = A_fifo[13][1];	// L9067
  hls::stream< int8_t > &v7505 /* v7505[16] */ = B_fifo[1][13];	// L9068
  hls::stream< int8_t > &v7506 /* v7506[16] */ = A_fifo[13][2];	// L9069
  hls::stream< int8_t > &v7507 /* v7507[16] */ = B_fifo[1][14];	// L9070
  PE_kernel_gemm_1_13(v7504, v7505, v7506, v7507, v6658, 13, 1);	// L9071
  hls::stream< int8_t > &v7508 /* v7508[16] */ = A_fifo[13][2];	// L9072
  hls::stream< int8_t > &v7509 /* v7509[16] */ = B_fifo[2][13];	// L9073
  hls::stream< int8_t > &v7510 /* v7510[16] */ = A_fifo[13][3];	// L9074
  hls::stream< int8_t > &v7511 /* v7511[16] */ = B_fifo[2][14];	// L9075
  PE_kernel_gemm_2_13(v7508, v7509, v7510, v7511, v6658, 13, 2);	// L9076
  hls::stream< int8_t > &v7512 /* v7512[16] */ = A_fifo[13][3];	// L9077
  hls::stream< int8_t > &v7513 /* v7513[16] */ = B_fifo[3][13];	// L9078
  hls::stream< int8_t > &v7514 /* v7514[16] */ = A_fifo[13][4];	// L9079
  hls::stream< int8_t > &v7515 /* v7515[16] */ = B_fifo[3][14];	// L9080
  PE_kernel_gemm_3_13(v7512, v7513, v7514, v7515, v6658, 13, 3);	// L9081
  hls::stream< int8_t > &v7516 /* v7516[16] */ = A_fifo[13][4];	// L9082
  hls::stream< int8_t > &v7517 /* v7517[16] */ = B_fifo[4][13];	// L9083
  hls::stream< int8_t > &v7518 /* v7518[16] */ = A_fifo[13][5];	// L9084
  hls::stream< int8_t > &v7519 /* v7519[16] */ = B_fifo[4][14];	// L9085
  PE_kernel_gemm_4_13(v7516, v7517, v7518, v7519, v6658, 13, 4);	// L9086
  hls::stream< int8_t > &v7520 /* v7520[16] */ = A_fifo[13][5];	// L9087
  hls::stream< int8_t > &v7521 /* v7521[16] */ = B_fifo[5][13];	// L9088
  hls::stream< int8_t > &v7522 /* v7522[16] */ = A_fifo[13][6];	// L9089
  hls::stream< int8_t > &v7523 /* v7523[16] */ = B_fifo[5][14];	// L9090
  PE_kernel_gemm_5_13(v7520, v7521, v7522, v7523, v6658, 13, 5);	// L9091
  hls::stream< int8_t > &v7524 /* v7524[16] */ = A_fifo[13][6];	// L9092
  hls::stream< int8_t > &v7525 /* v7525[16] */ = B_fifo[6][13];	// L9093
  hls::stream< int8_t > &v7526 /* v7526[16] */ = A_fifo[13][7];	// L9094
  hls::stream< int8_t > &v7527 /* v7527[16] */ = B_fifo[6][14];	// L9095
  PE_kernel_gemm_6_13(v7524, v7525, v7526, v7527, v6658, 13, 6);	// L9096
  hls::stream< int8_t > &v7528 /* v7528[16] */ = A_fifo[13][7];	// L9097
  hls::stream< int8_t > &v7529 /* v7529[16] */ = B_fifo[7][13];	// L9098
  hls::stream< int8_t > &v7530 /* v7530[16] */ = A_fifo[13][8];	// L9099
  hls::stream< int8_t > &v7531 /* v7531[16] */ = B_fifo[7][14];	// L9100
  PE_kernel_gemm_7_13(v7528, v7529, v7530, v7531, v6658, 13, 7);	// L9101
  hls::stream< int8_t > &v7532 /* v7532[16] */ = A_fifo[13][8];	// L9102
  hls::stream< int8_t > &v7533 /* v7533[16] */ = B_fifo[8][13];	// L9103
  hls::stream< int8_t > &v7534 /* v7534[16] */ = A_fifo[13][9];	// L9104
  hls::stream< int8_t > &v7535 /* v7535[16] */ = B_fifo[8][14];	// L9105
  PE_kernel_gemm_8_13(v7532, v7533, v7534, v7535, v6658, 13, 8);	// L9106
  hls::stream< int8_t > &v7536 /* v7536[16] */ = A_fifo[13][9];	// L9107
  hls::stream< int8_t > &v7537 /* v7537[16] */ = B_fifo[9][13];	// L9108
  hls::stream< int8_t > &v7538 /* v7538[16] */ = A_fifo[13][10];	// L9109
  hls::stream< int8_t > &v7539 /* v7539[16] */ = B_fifo[9][14];	// L9110
  PE_kernel_gemm_9_13(v7536, v7537, v7538, v7539, v6658, 13, 9);	// L9111
  hls::stream< int8_t > &v7540 /* v7540[16] */ = A_fifo[13][10];	// L9112
  hls::stream< int8_t > &v7541 /* v7541[16] */ = B_fifo[10][13];	// L9113
  hls::stream< int8_t > &v7542 /* v7542[16] */ = A_fifo[13][11];	// L9114
  hls::stream< int8_t > &v7543 /* v7543[16] */ = B_fifo[10][14];	// L9115
  PE_kernel_gemm_10_13(v7540, v7541, v7542, v7543, v6658, 13, 10);	// L9116
  hls::stream< int8_t > &v7544 /* v7544[16] */ = A_fifo[13][11];	// L9117
  hls::stream< int8_t > &v7545 /* v7545[16] */ = B_fifo[11][13];	// L9118
  hls::stream< int8_t > &v7546 /* v7546[16] */ = A_fifo[13][12];	// L9119
  hls::stream< int8_t > &v7547 /* v7547[16] */ = B_fifo[11][14];	// L9120
  PE_kernel_gemm_11_13(v7544, v7545, v7546, v7547, v6658, 13, 11);	// L9121
  hls::stream< int8_t > &v7548 /* v7548[16] */ = A_fifo[13][12];	// L9122
  hls::stream< int8_t > &v7549 /* v7549[16] */ = B_fifo[12][13];	// L9123
  hls::stream< int8_t > &v7550 /* v7550[16] */ = A_fifo[13][13];	// L9124
  hls::stream< int8_t > &v7551 /* v7551[16] */ = B_fifo[12][14];	// L9125
  PE_kernel_gemm_12_13(v7548, v7549, v7550, v7551, v6658, 13, 12);	// L9126
  hls::stream< int8_t > &v7552 /* v7552[16] */ = A_fifo[13][13];	// L9127
  hls::stream< int8_t > &v7553 /* v7553[16] */ = B_fifo[13][13];	// L9128
  hls::stream< int8_t > &v7554 /* v7554[16] */ = A_fifo[13][14];	// L9129
  hls::stream< int8_t > &v7555 /* v7555[16] */ = B_fifo[13][14];	// L9130
  PE_kernel_gemm_13_13(v7552, v7553, v7554, v7555, v6658, 13, 13);	// L9131
  hls::stream< int8_t > &v7556 /* v7556[16] */ = A_fifo[13][14];	// L9132
  hls::stream< int8_t > &v7557 /* v7557[16] */ = B_fifo[14][13];	// L9133
  hls::stream< int8_t > &v7558 /* v7558[16] */ = A_fifo[13][15];	// L9134
  hls::stream< int8_t > &v7559 /* v7559[16] */ = B_fifo[14][14];	// L9135
  PE_kernel_gemm_14_13(v7556, v7557, v7558, v7559, v6658, 13, 14);	// L9136
  hls::stream< int8_t > &v7560 /* v7560[16] */ = A_fifo[13][15];	// L9137
  hls::stream< int8_t > &v7561 /* v7561[16] */ = B_fifo[15][13];	// L9138
  hls::stream< int8_t > &v7562 /* v7562[16] */ = A_fifo[13][16];	// L9139
  hls::stream< int8_t > &v7563 /* v7563[16] */ = B_fifo[15][14];	// L9140
  PE_kernel_gemm_15_13(v7560, v7561, v7562, v7563, v6658, 13, 15);	// L9141
  hls::stream< int8_t > &v7564 /* v7564[16] */ = A_fifo[14][0];	// L9142
  hls::stream< int8_t > &v7565 /* v7565[16] */ = B_fifo[0][14];	// L9143
  hls::stream< int8_t > &v7566 /* v7566[16] */ = A_fifo[14][1];	// L9144
  hls::stream< int8_t > &v7567 /* v7567[16] */ = B_fifo[0][15];	// L9145
  PE_kernel_gemm_0_14(v7564, v7565, v7566, v7567, v6658, 14, 0);	// L9146
  hls::stream< int8_t > &v7568 /* v7568[16] */ = A_fifo[14][1];	// L9147
  hls::stream< int8_t > &v7569 /* v7569[16] */ = B_fifo[1][14];	// L9148
  hls::stream< int8_t > &v7570 /* v7570[16] */ = A_fifo[14][2];	// L9149
  hls::stream< int8_t > &v7571 /* v7571[16] */ = B_fifo[1][15];	// L9150
  PE_kernel_gemm_1_14(v7568, v7569, v7570, v7571, v6658, 14, 1);	// L9151
  hls::stream< int8_t > &v7572 /* v7572[16] */ = A_fifo[14][2];	// L9152
  hls::stream< int8_t > &v7573 /* v7573[16] */ = B_fifo[2][14];	// L9153
  hls::stream< int8_t > &v7574 /* v7574[16] */ = A_fifo[14][3];	// L9154
  hls::stream< int8_t > &v7575 /* v7575[16] */ = B_fifo[2][15];	// L9155
  PE_kernel_gemm_2_14(v7572, v7573, v7574, v7575, v6658, 14, 2);	// L9156
  hls::stream< int8_t > &v7576 /* v7576[16] */ = A_fifo[14][3];	// L9157
  hls::stream< int8_t > &v7577 /* v7577[16] */ = B_fifo[3][14];	// L9158
  hls::stream< int8_t > &v7578 /* v7578[16] */ = A_fifo[14][4];	// L9159
  hls::stream< int8_t > &v7579 /* v7579[16] */ = B_fifo[3][15];	// L9160
  PE_kernel_gemm_3_14(v7576, v7577, v7578, v7579, v6658, 14, 3);	// L9161
  hls::stream< int8_t > &v7580 /* v7580[16] */ = A_fifo[14][4];	// L9162
  hls::stream< int8_t > &v7581 /* v7581[16] */ = B_fifo[4][14];	// L9163
  hls::stream< int8_t > &v7582 /* v7582[16] */ = A_fifo[14][5];	// L9164
  hls::stream< int8_t > &v7583 /* v7583[16] */ = B_fifo[4][15];	// L9165
  PE_kernel_gemm_4_14(v7580, v7581, v7582, v7583, v6658, 14, 4);	// L9166
  hls::stream< int8_t > &v7584 /* v7584[16] */ = A_fifo[14][5];	// L9167
  hls::stream< int8_t > &v7585 /* v7585[16] */ = B_fifo[5][14];	// L9168
  hls::stream< int8_t > &v7586 /* v7586[16] */ = A_fifo[14][6];	// L9169
  hls::stream< int8_t > &v7587 /* v7587[16] */ = B_fifo[5][15];	// L9170
  PE_kernel_gemm_5_14(v7584, v7585, v7586, v7587, v6658, 14, 5);	// L9171
  hls::stream< int8_t > &v7588 /* v7588[16] */ = A_fifo[14][6];	// L9172
  hls::stream< int8_t > &v7589 /* v7589[16] */ = B_fifo[6][14];	// L9173
  hls::stream< int8_t > &v7590 /* v7590[16] */ = A_fifo[14][7];	// L9174
  hls::stream< int8_t > &v7591 /* v7591[16] */ = B_fifo[6][15];	// L9175
  PE_kernel_gemm_6_14(v7588, v7589, v7590, v7591, v6658, 14, 6);	// L9176
  hls::stream< int8_t > &v7592 /* v7592[16] */ = A_fifo[14][7];	// L9177
  hls::stream< int8_t > &v7593 /* v7593[16] */ = B_fifo[7][14];	// L9178
  hls::stream< int8_t > &v7594 /* v7594[16] */ = A_fifo[14][8];	// L9179
  hls::stream< int8_t > &v7595 /* v7595[16] */ = B_fifo[7][15];	// L9180
  PE_kernel_gemm_7_14(v7592, v7593, v7594, v7595, v6658, 14, 7);	// L9181
  hls::stream< int8_t > &v7596 /* v7596[16] */ = A_fifo[14][8];	// L9182
  hls::stream< int8_t > &v7597 /* v7597[16] */ = B_fifo[8][14];	// L9183
  hls::stream< int8_t > &v7598 /* v7598[16] */ = A_fifo[14][9];	// L9184
  hls::stream< int8_t > &v7599 /* v7599[16] */ = B_fifo[8][15];	// L9185
  PE_kernel_gemm_8_14(v7596, v7597, v7598, v7599, v6658, 14, 8);	// L9186
  hls::stream< int8_t > &v7600 /* v7600[16] */ = A_fifo[14][9];	// L9187
  hls::stream< int8_t > &v7601 /* v7601[16] */ = B_fifo[9][14];	// L9188
  hls::stream< int8_t > &v7602 /* v7602[16] */ = A_fifo[14][10];	// L9189
  hls::stream< int8_t > &v7603 /* v7603[16] */ = B_fifo[9][15];	// L9190
  PE_kernel_gemm_9_14(v7600, v7601, v7602, v7603, v6658, 14, 9);	// L9191
  hls::stream< int8_t > &v7604 /* v7604[16] */ = A_fifo[14][10];	// L9192
  hls::stream< int8_t > &v7605 /* v7605[16] */ = B_fifo[10][14];	// L9193
  hls::stream< int8_t > &v7606 /* v7606[16] */ = A_fifo[14][11];	// L9194
  hls::stream< int8_t > &v7607 /* v7607[16] */ = B_fifo[10][15];	// L9195
  PE_kernel_gemm_10_14(v7604, v7605, v7606, v7607, v6658, 14, 10);	// L9196
  hls::stream< int8_t > &v7608 /* v7608[16] */ = A_fifo[14][11];	// L9197
  hls::stream< int8_t > &v7609 /* v7609[16] */ = B_fifo[11][14];	// L9198
  hls::stream< int8_t > &v7610 /* v7610[16] */ = A_fifo[14][12];	// L9199
  hls::stream< int8_t > &v7611 /* v7611[16] */ = B_fifo[11][15];	// L9200
  PE_kernel_gemm_11_14(v7608, v7609, v7610, v7611, v6658, 14, 11);	// L9201
  hls::stream< int8_t > &v7612 /* v7612[16] */ = A_fifo[14][12];	// L9202
  hls::stream< int8_t > &v7613 /* v7613[16] */ = B_fifo[12][14];	// L9203
  hls::stream< int8_t > &v7614 /* v7614[16] */ = A_fifo[14][13];	// L9204
  hls::stream< int8_t > &v7615 /* v7615[16] */ = B_fifo[12][15];	// L9205
  PE_kernel_gemm_12_14(v7612, v7613, v7614, v7615, v6658, 14, 12);	// L9206
  hls::stream< int8_t > &v7616 /* v7616[16] */ = A_fifo[14][13];	// L9207
  hls::stream< int8_t > &v7617 /* v7617[16] */ = B_fifo[13][14];	// L9208
  hls::stream< int8_t > &v7618 /* v7618[16] */ = A_fifo[14][14];	// L9209
  hls::stream< int8_t > &v7619 /* v7619[16] */ = B_fifo[13][15];	// L9210
  PE_kernel_gemm_13_14(v7616, v7617, v7618, v7619, v6658, 14, 13);	// L9211
  hls::stream< int8_t > &v7620 /* v7620[16] */ = A_fifo[14][14];	// L9212
  hls::stream< int8_t > &v7621 /* v7621[16] */ = B_fifo[14][14];	// L9213
  hls::stream< int8_t > &v7622 /* v7622[16] */ = A_fifo[14][15];	// L9214
  hls::stream< int8_t > &v7623 /* v7623[16] */ = B_fifo[14][15];	// L9215
  PE_kernel_gemm_14_14(v7620, v7621, v7622, v7623, v6658, 14, 14);	// L9216
  hls::stream< int8_t > &v7624 /* v7624[16] */ = A_fifo[14][15];	// L9217
  hls::stream< int8_t > &v7625 /* v7625[16] */ = B_fifo[15][14];	// L9218
  hls::stream< int8_t > &v7626 /* v7626[16] */ = A_fifo[14][16];	// L9219
  hls::stream< int8_t > &v7627 /* v7627[16] */ = B_fifo[15][15];	// L9220
  PE_kernel_gemm_15_14(v7624, v7625, v7626, v7627, v6658, 14, 15);	// L9221
  hls::stream< int8_t > &v7628 /* v7628[16] */ = A_fifo[15][0];	// L9222
  hls::stream< int8_t > &v7629 /* v7629[16] */ = B_fifo[0][15];	// L9223
  hls::stream< int8_t > &v7630 /* v7630[16] */ = A_fifo[15][1];	// L9224
  hls::stream< int8_t > &v7631 /* v7631[16] */ = B_fifo[0][16];	// L9225
  PE_kernel_gemm_0_15(v7628, v7629, v7630, v7631, v6658, 15, 0);	// L9226
  hls::stream< int8_t > &v7632 /* v7632[16] */ = A_fifo[15][1];	// L9227
  hls::stream< int8_t > &v7633 /* v7633[16] */ = B_fifo[1][15];	// L9228
  hls::stream< int8_t > &v7634 /* v7634[16] */ = A_fifo[15][2];	// L9229
  hls::stream< int8_t > &v7635 /* v7635[16] */ = B_fifo[1][16];	// L9230
  PE_kernel_gemm_1_15(v7632, v7633, v7634, v7635, v6658, 15, 1);	// L9231
  hls::stream< int8_t > &v7636 /* v7636[16] */ = A_fifo[15][2];	// L9232
  hls::stream< int8_t > &v7637 /* v7637[16] */ = B_fifo[2][15];	// L9233
  hls::stream< int8_t > &v7638 /* v7638[16] */ = A_fifo[15][3];	// L9234
  hls::stream< int8_t > &v7639 /* v7639[16] */ = B_fifo[2][16];	// L9235
  PE_kernel_gemm_2_15(v7636, v7637, v7638, v7639, v6658, 15, 2);	// L9236
  hls::stream< int8_t > &v7640 /* v7640[16] */ = A_fifo[15][3];	// L9237
  hls::stream< int8_t > &v7641 /* v7641[16] */ = B_fifo[3][15];	// L9238
  hls::stream< int8_t > &v7642 /* v7642[16] */ = A_fifo[15][4];	// L9239
  hls::stream< int8_t > &v7643 /* v7643[16] */ = B_fifo[3][16];	// L9240
  PE_kernel_gemm_3_15(v7640, v7641, v7642, v7643, v6658, 15, 3);	// L9241
  hls::stream< int8_t > &v7644 /* v7644[16] */ = A_fifo[15][4];	// L9242
  hls::stream< int8_t > &v7645 /* v7645[16] */ = B_fifo[4][15];	// L9243
  hls::stream< int8_t > &v7646 /* v7646[16] */ = A_fifo[15][5];	// L9244
  hls::stream< int8_t > &v7647 /* v7647[16] */ = B_fifo[4][16];	// L9245
  PE_kernel_gemm_4_15(v7644, v7645, v7646, v7647, v6658, 15, 4);	// L9246
  hls::stream< int8_t > &v7648 /* v7648[16] */ = A_fifo[15][5];	// L9247
  hls::stream< int8_t > &v7649 /* v7649[16] */ = B_fifo[5][15];	// L9248
  hls::stream< int8_t > &v7650 /* v7650[16] */ = A_fifo[15][6];	// L9249
  hls::stream< int8_t > &v7651 /* v7651[16] */ = B_fifo[5][16];	// L9250
  PE_kernel_gemm_5_15(v7648, v7649, v7650, v7651, v6658, 15, 5);	// L9251
  hls::stream< int8_t > &v7652 /* v7652[16] */ = A_fifo[15][6];	// L9252
  hls::stream< int8_t > &v7653 /* v7653[16] */ = B_fifo[6][15];	// L9253
  hls::stream< int8_t > &v7654 /* v7654[16] */ = A_fifo[15][7];	// L9254
  hls::stream< int8_t > &v7655 /* v7655[16] */ = B_fifo[6][16];	// L9255
  PE_kernel_gemm_6_15(v7652, v7653, v7654, v7655, v6658, 15, 6);	// L9256
  hls::stream< int8_t > &v7656 /* v7656[16] */ = A_fifo[15][7];	// L9257
  hls::stream< int8_t > &v7657 /* v7657[16] */ = B_fifo[7][15];	// L9258
  hls::stream< int8_t > &v7658 /* v7658[16] */ = A_fifo[15][8];	// L9259
  hls::stream< int8_t > &v7659 /* v7659[16] */ = B_fifo[7][16];	// L9260
  PE_kernel_gemm_7_15(v7656, v7657, v7658, v7659, v6658, 15, 7);	// L9261
  hls::stream< int8_t > &v7660 /* v7660[16] */ = A_fifo[15][8];	// L9262
  hls::stream< int8_t > &v7661 /* v7661[16] */ = B_fifo[8][15];	// L9263
  hls::stream< int8_t > &v7662 /* v7662[16] */ = A_fifo[15][9];	// L9264
  hls::stream< int8_t > &v7663 /* v7663[16] */ = B_fifo[8][16];	// L9265
  PE_kernel_gemm_8_15(v7660, v7661, v7662, v7663, v6658, 15, 8);	// L9266
  hls::stream< int8_t > &v7664 /* v7664[16] */ = A_fifo[15][9];	// L9267
  hls::stream< int8_t > &v7665 /* v7665[16] */ = B_fifo[9][15];	// L9268
  hls::stream< int8_t > &v7666 /* v7666[16] */ = A_fifo[15][10];	// L9269
  hls::stream< int8_t > &v7667 /* v7667[16] */ = B_fifo[9][16];	// L9270
  PE_kernel_gemm_9_15(v7664, v7665, v7666, v7667, v6658, 15, 9);	// L9271
  hls::stream< int8_t > &v7668 /* v7668[16] */ = A_fifo[15][10];	// L9272
  hls::stream< int8_t > &v7669 /* v7669[16] */ = B_fifo[10][15];	// L9273
  hls::stream< int8_t > &v7670 /* v7670[16] */ = A_fifo[15][11];	// L9274
  hls::stream< int8_t > &v7671 /* v7671[16] */ = B_fifo[10][16];	// L9275
  PE_kernel_gemm_10_15(v7668, v7669, v7670, v7671, v6658, 15, 10);	// L9276
  hls::stream< int8_t > &v7672 /* v7672[16] */ = A_fifo[15][11];	// L9277
  hls::stream< int8_t > &v7673 /* v7673[16] */ = B_fifo[11][15];	// L9278
  hls::stream< int8_t > &v7674 /* v7674[16] */ = A_fifo[15][12];	// L9279
  hls::stream< int8_t > &v7675 /* v7675[16] */ = B_fifo[11][16];	// L9280
  PE_kernel_gemm_11_15(v7672, v7673, v7674, v7675, v6658, 15, 11);	// L9281
  hls::stream< int8_t > &v7676 /* v7676[16] */ = A_fifo[15][12];	// L9282
  hls::stream< int8_t > &v7677 /* v7677[16] */ = B_fifo[12][15];	// L9283
  hls::stream< int8_t > &v7678 /* v7678[16] */ = A_fifo[15][13];	// L9284
  hls::stream< int8_t > &v7679 /* v7679[16] */ = B_fifo[12][16];	// L9285
  PE_kernel_gemm_12_15(v7676, v7677, v7678, v7679, v6658, 15, 12);	// L9286
  hls::stream< int8_t > &v7680 /* v7680[16] */ = A_fifo[15][13];	// L9287
  hls::stream< int8_t > &v7681 /* v7681[16] */ = B_fifo[13][15];	// L9288
  hls::stream< int8_t > &v7682 /* v7682[16] */ = A_fifo[15][14];	// L9289
  hls::stream< int8_t > &v7683 /* v7683[16] */ = B_fifo[13][16];	// L9290
  PE_kernel_gemm_13_15(v7680, v7681, v7682, v7683, v6658, 15, 13);	// L9291
  hls::stream< int8_t > &v7684 /* v7684[16] */ = A_fifo[15][14];	// L9292
  hls::stream< int8_t > &v7685 /* v7685[16] */ = B_fifo[14][15];	// L9293
  hls::stream< int8_t > &v7686 /* v7686[16] */ = A_fifo[15][15];	// L9294
  hls::stream< int8_t > &v7687 /* v7687[16] */ = B_fifo[14][16];	// L9295
  PE_kernel_gemm_14_15(v7684, v7685, v7686, v7687, v6658, 15, 14);	// L9296
  hls::stream< int8_t > &v7688 /* v7688[16] */ = A_fifo[15][15];	// L9297
  hls::stream< int8_t > &v7689 /* v7689[16] */ = B_fifo[15][15];	// L9298
  hls::stream< int8_t > &v7690 /* v7690[16] */ = A_fifo[15][16];	// L9299
  hls::stream< int8_t > &v7691 /* v7691[16] */ = B_fifo[15][16];	// L9300
  PE_kernel_gemm_15_15(v7688, v7689, v7690, v7691, v6658, 15, 15);	// L9301
  l_data_drain_k257: for (int k257 = 0; k257 < 16; k257++) {	// L9302
    l_S_m_4_m1: for (int m1 = 0; m1 < 16; m1++) {	// L9303
      int8_t v7694 = A_fifo[m1][16].read(); // A_fifo[m1][16][k257];	// L9304
      A_drain[m1] = v7694;	// L9305
    }
    l_S_n_5_n1: for (int n1 = 0; n1 < 16; n1++) {	// L9307
      int8_t v7696 = B_fifo[n1][16].read(); // B_fifo[n1][16][k257];	// L9308
      B_drain[n1] = v7696;	// L9309
    }
  }
}

void systolic_gemm(
  int8_t v7697[16][16],
  int8_t v7698[16][16],
  int32_t v7699[16][16]
) {	// L9314
  int8_t local_A[16][16];	// L9315
  #pragma HLS array_partition variable=local_A complete dim=1

  int8_t local_B[16][16];	// L9316
  #pragma HLS array_partition variable=local_B complete dim=2

  int32_t local_C[16][16];	// L9317
  #pragma HLS array_partition variable=local_C complete dim=1
  #pragma HLS array_partition variable=local_C complete dim=2

  l_outer_tile_mi_ni_fused: for (int mi_ni_fused = 0; mi_ni_fused < 1; mi_ni_fused++) {	// L9318
    l_load_A_tile_ak: for (int ak = 0; ak < 16; ak++) {	// L9320
    #pragma HLS pipeline II=1
      l_ai: for (int ai = 0; ai < 16; ai++) {	// L9321
        if (1) {	// L9326
          int8_t v7706 = v7697[((mi_ni_fused * 16) + ai)][ak];	// L9327
          local_A[ai][ak] = v7706;	// L9328
        }
      }
    }
    l_load_B_tile_bk: for (int bk = 0; bk < 16; bk++) {	// L9332
    #pragma HLS pipeline II=1
      l_bj: for (int bj = 0; bj < 16; bj++) {	// L9333
        int8_t v7709 = v7698[bk][((0 * 16) + bj)];	// L9334
        local_B[bk][bj] = v7709;	// L9335
      }
    }
    systolic_tile_gemm(local_A, local_B, local_C);	// L9338
    l_store_C_tile_sj: for (int sj = 0; sj < 16; sj++) {	// L9339
    #pragma HLS pipeline II=1
      l_si: for (int si = 0; si < 16; si++) {	// L9340
        int32_t v7712 = local_C[si][sj];	// L9341
        v7699[((mi_ni_fused * 16) + si)][((0 * 16) + sj)] = v7712;	// L9342
      }
    }
  }
}

void load_buf0(
  int8_t v7713[256],
  int8_t v7714[16][16]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 16; load_buf0_l_0++) {	//
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 16; load_buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v7717 = v7713[((load_buf0_l_0 * 16) + load_buf0_l_1)];	//
      v7714[load_buf0_l_0][load_buf0_l_1] = v7717;	//
    }
  }
}

void load_buf1(
  int8_t v7718[256],
  int8_t v7719[16][16]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 16; load_buf1_l_0++) {	//
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 16; load_buf1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v7722 = v7718[((load_buf1_l_0 * 16) + load_buf1_l_1)];	//
      v7719[load_buf1_l_0][load_buf1_l_1] = v7722;	//
    }
  }
}

void store_res2(
  int32_t v7723[16][16],
  int32_t v7724[256]
) {	//
  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 16; store_res2_l_0++) {	//
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 16; store_res2_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int32_t v7727 = v7723[store_res2_l_0][store_res2_l_1];	//
      v7724[((store_res2_l_0 * 16) + store_res2_l_1)] = v7727;	//
    }
  }
}

/// This is top function.
void gemm(
  int8_t *v7728,
  int8_t *v7729,
  int32_t *v7730
) {	// L9348
  #pragma HLS interface m_axi port=v7728 offset=slave bundle=gmem0 depth=256
  #pragma HLS interface m_axi port=v7729 offset=slave bundle=gmem1 depth=256
  #pragma HLS interface m_axi port=v7730 offset=slave bundle=gmem2 depth=256
  int8_t buf0[16][16];	//
  load_buf0(v7728, buf0);	//
  int8_t buf1[16][16];	//
  load_buf1(v7729, buf1);	//
  int32_t buf2[16][16];	//
  systolic_gemm(buf0, buf1, buf2);	// L9349
  store_res2(buf2, v7730);	//
}


} // extern "C"


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
  hls::stream< int8_t > &v0 /* v0[128] */,
  hls::stream< int8_t > &v1 /* v1[128] */,
  hls::stream< int8_t > &v2 /* v2[128] */,
  hls::stream< int8_t > &v3 /* v3[128] */,
  int32_t v4[8][8],
  int v5,
  int v6
) {	// L7
  #pragma HLS stream variable=v0 depth=9
  #pragma HLS stream variable=v1 depth=9
  #pragma HLS stream variable=v2 depth=9
  #pragma HLS stream variable=v3 depth=9
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  int32_t v;	// L9
  v = 0;	// L10
  l_reduction_k: for (int k = 0; k < 128; k++) {	// L11
  #pragma HLS pipeline II=1
    int8_t v9 = v0.read(); // v0[k];	// L12
    int8_t a;	// L13
    a = v9;	// L14
    int8_t v11 = v1.read(); // v1[k];	// L15
    int8_t b;	// L16
    b = v11;	// L17
    int8_t v13 = a;	// L18
    int8_t v14 = b;	// L19
    int16_t v15 = v13;	// L20
    int16_t v16 = v14;	// L21
    int16_t v17 = v15 * v16;	// L22
    int32_t v18 = v;	// L23
    ap_int<33> v19 = v18;	// L24
    ap_int<33> v20 = v17;	// L25
    ap_int<33> v21 = v19 + v20;	// L26
    int32_t v22 = v21;	// L27
    v = v22;	// L28
    int8_t v23 = a;	// L29
    v2.write(v23); // v2[k] = v23;	// L30
    int8_t v24 = b;	// L31
    v3.write(v24); // v3[k] = v24;	// L32
  }
  int32_t v25 = v;	// L34
  v4[v5][v6] = v25;	// L35
}

void PE_kernel_gemm_1_0(
  hls::stream< int8_t > &v26 /* v26[128] */,
  hls::stream< int8_t > &v27 /* v27[128] */,
  hls::stream< int8_t > &v28 /* v28[128] */,
  hls::stream< int8_t > &v29 /* v29[128] */,
  int32_t v30[8][8],
  int v31,
  int v32
) {	// L38
  #pragma HLS stream variable=v26 depth=9
  #pragma HLS stream variable=v27 depth=9
  #pragma HLS stream variable=v28 depth=9
  #pragma HLS stream variable=v29 depth=9
  #pragma HLS array_partition variable=v30 complete dim=1
  #pragma HLS array_partition variable=v30 complete dim=2

  int32_t v1;	// L40
  v1 = 0;	// L41
  l_reduction_k1: for (int k1 = 0; k1 < 128; k1++) {	// L42
  #pragma HLS pipeline II=1
    int8_t v35 = v26.read(); // v26[k1];	// L43
    int8_t a1;	// L44
    a1 = v35;	// L45
    int8_t v37 = v27.read(); // v27[k1];	// L46
    int8_t b1;	// L47
    b1 = v37;	// L48
    int8_t v39 = a1;	// L49
    int8_t v40 = b1;	// L50
    int16_t v41 = v39;	// L51
    int16_t v42 = v40;	// L52
    int16_t v43 = v41 * v42;	// L53
    int32_t v44 = v1;	// L54
    ap_int<33> v45 = v44;	// L55
    ap_int<33> v46 = v43;	// L56
    ap_int<33> v47 = v45 + v46;	// L57
    int32_t v48 = v47;	// L58
    v1 = v48;	// L59
    int8_t v49 = a1;	// L60
    v28.write(v49); // v28[k1] = v49;	// L61
    int8_t v50 = b1;	// L62
    v29.write(v50); // v29[k1] = v50;	// L63
  }
  int32_t v51 = v1;	// L65
  v30[v31][v32] = v51;	// L66
}

void PE_kernel_gemm_2_0(
  hls::stream< int8_t > &v52 /* v52[128] */,
  hls::stream< int8_t > &v53 /* v53[128] */,
  hls::stream< int8_t > &v54 /* v54[128] */,
  hls::stream< int8_t > &v55 /* v55[128] */,
  int32_t v56[8][8],
  int v57,
  int v58
) {	// L69
  #pragma HLS stream variable=v52 depth=9
  #pragma HLS stream variable=v53 depth=9
  #pragma HLS stream variable=v54 depth=9
  #pragma HLS stream variable=v55 depth=9
  #pragma HLS array_partition variable=v56 complete dim=1
  #pragma HLS array_partition variable=v56 complete dim=2

  int32_t v2;	// L71
  v2 = 0;	// L72
  l_reduction_k2: for (int k2 = 0; k2 < 128; k2++) {	// L73
  #pragma HLS pipeline II=1
    int8_t v61 = v52.read(); // v52[k2];	// L74
    int8_t a2;	// L75
    a2 = v61;	// L76
    int8_t v63 = v53.read(); // v53[k2];	// L77
    int8_t b2;	// L78
    b2 = v63;	// L79
    int8_t v65 = a2;	// L80
    int8_t v66 = b2;	// L81
    int16_t v67 = v65;	// L82
    int16_t v68 = v66;	// L83
    int16_t v69 = v67 * v68;	// L84
    int32_t v70 = v2;	// L85
    ap_int<33> v71 = v70;	// L86
    ap_int<33> v72 = v69;	// L87
    ap_int<33> v73 = v71 + v72;	// L88
    int32_t v74 = v73;	// L89
    v2 = v74;	// L90
    int8_t v75 = a2;	// L91
    v54.write(v75); // v54[k2] = v75;	// L92
    int8_t v76 = b2;	// L93
    v55.write(v76); // v55[k2] = v76;	// L94
  }
  int32_t v77 = v2;	// L96
  v56[v57][v58] = v77;	// L97
}

void PE_kernel_gemm_3_0(
  hls::stream< int8_t > &v78 /* v78[128] */,
  hls::stream< int8_t > &v79 /* v79[128] */,
  hls::stream< int8_t > &v80 /* v80[128] */,
  hls::stream< int8_t > &v81 /* v81[128] */,
  int32_t v82[8][8],
  int v83,
  int v84
) {	// L100
  #pragma HLS stream variable=v78 depth=9
  #pragma HLS stream variable=v79 depth=9
  #pragma HLS stream variable=v80 depth=9
  #pragma HLS stream variable=v81 depth=9
  #pragma HLS array_partition variable=v82 complete dim=1
  #pragma HLS array_partition variable=v82 complete dim=2

  int32_t v3;	// L102
  v3 = 0;	// L103
  l_reduction_k3: for (int k3 = 0; k3 < 128; k3++) {	// L104
  #pragma HLS pipeline II=1
    int8_t v87 = v78.read(); // v78[k3];	// L105
    int8_t a3;	// L106
    a3 = v87;	// L107
    int8_t v89 = v79.read(); // v79[k3];	// L108
    int8_t b3;	// L109
    b3 = v89;	// L110
    int8_t v91 = a3;	// L111
    int8_t v92 = b3;	// L112
    int16_t v93 = v91;	// L113
    int16_t v94 = v92;	// L114
    int16_t v95 = v93 * v94;	// L115
    int32_t v96 = v3;	// L116
    ap_int<33> v97 = v96;	// L117
    ap_int<33> v98 = v95;	// L118
    ap_int<33> v99 = v97 + v98;	// L119
    int32_t v100 = v99;	// L120
    v3 = v100;	// L121
    int8_t v101 = a3;	// L122
    v80.write(v101); // v80[k3] = v101;	// L123
    int8_t v102 = b3;	// L124
    v81.write(v102); // v81[k3] = v102;	// L125
  }
  int32_t v103 = v3;	// L127
  v82[v83][v84] = v103;	// L128
}

void PE_kernel_gemm_4_0(
  hls::stream< int8_t > &v104 /* v104[128] */,
  hls::stream< int8_t > &v105 /* v105[128] */,
  hls::stream< int8_t > &v106 /* v106[128] */,
  hls::stream< int8_t > &v107 /* v107[128] */,
  int32_t v108[8][8],
  int v109,
  int v110
) {	// L131
  #pragma HLS stream variable=v104 depth=9
  #pragma HLS stream variable=v105 depth=9
  #pragma HLS stream variable=v106 depth=9
  #pragma HLS stream variable=v107 depth=9
  #pragma HLS array_partition variable=v108 complete dim=1
  #pragma HLS array_partition variable=v108 complete dim=2

  int32_t v4;	// L133
  v4 = 0;	// L134
  l_reduction_k4: for (int k4 = 0; k4 < 128; k4++) {	// L135
  #pragma HLS pipeline II=1
    int8_t v113 = v104.read(); // v104[k4];	// L136
    int8_t a4;	// L137
    a4 = v113;	// L138
    int8_t v115 = v105.read(); // v105[k4];	// L139
    int8_t b4;	// L140
    b4 = v115;	// L141
    int8_t v117 = a4;	// L142
    int8_t v118 = b4;	// L143
    int16_t v119 = v117;	// L144
    int16_t v120 = v118;	// L145
    int16_t v121 = v119 * v120;	// L146
    int32_t v122 = v4;	// L147
    ap_int<33> v123 = v122;	// L148
    ap_int<33> v124 = v121;	// L149
    ap_int<33> v125 = v123 + v124;	// L150
    int32_t v126 = v125;	// L151
    v4 = v126;	// L152
    int8_t v127 = a4;	// L153
    v106.write(v127); // v106[k4] = v127;	// L154
    int8_t v128 = b4;	// L155
    v107.write(v128); // v107[k4] = v128;	// L156
  }
  int32_t v129 = v4;	// L158
  v108[v109][v110] = v129;	// L159
}

void PE_kernel_gemm_5_0(
  hls::stream< int8_t > &v130 /* v130[128] */,
  hls::stream< int8_t > &v131 /* v131[128] */,
  hls::stream< int8_t > &v132 /* v132[128] */,
  hls::stream< int8_t > &v133 /* v133[128] */,
  int32_t v134[8][8],
  int v135,
  int v136
) {	// L162
  #pragma HLS stream variable=v130 depth=9
  #pragma HLS stream variable=v131 depth=9
  #pragma HLS stream variable=v132 depth=9
  #pragma HLS stream variable=v133 depth=9
  #pragma HLS array_partition variable=v134 complete dim=1
  #pragma HLS array_partition variable=v134 complete dim=2

  int32_t v5;	// L164
  v5 = 0;	// L165
  l_reduction_k5: for (int k5 = 0; k5 < 128; k5++) {	// L166
  #pragma HLS pipeline II=1
    int8_t v139 = v130.read(); // v130[k5];	// L167
    int8_t a5;	// L168
    a5 = v139;	// L169
    int8_t v141 = v131.read(); // v131[k5];	// L170
    int8_t b5;	// L171
    b5 = v141;	// L172
    int8_t v143 = a5;	// L173
    int8_t v144 = b5;	// L174
    int16_t v145 = v143;	// L175
    int16_t v146 = v144;	// L176
    int16_t v147 = v145 * v146;	// L177
    int32_t v148 = v5;	// L178
    ap_int<33> v149 = v148;	// L179
    ap_int<33> v150 = v147;	// L180
    ap_int<33> v151 = v149 + v150;	// L181
    int32_t v152 = v151;	// L182
    v5 = v152;	// L183
    int8_t v153 = a5;	// L184
    v132.write(v153); // v132[k5] = v153;	// L185
    int8_t v154 = b5;	// L186
    v133.write(v154); // v133[k5] = v154;	// L187
  }
  int32_t v155 = v5;	// L189
  v134[v135][v136] = v155;	// L190
}

void PE_kernel_gemm_6_0(
  hls::stream< int8_t > &v156 /* v156[128] */,
  hls::stream< int8_t > &v157 /* v157[128] */,
  hls::stream< int8_t > &v158 /* v158[128] */,
  hls::stream< int8_t > &v159 /* v159[128] */,
  int32_t v160[8][8],
  int v161,
  int v162
) {	// L193
  #pragma HLS stream variable=v156 depth=9
  #pragma HLS stream variable=v157 depth=9
  #pragma HLS stream variable=v158 depth=9
  #pragma HLS stream variable=v159 depth=9
  #pragma HLS array_partition variable=v160 complete dim=1
  #pragma HLS array_partition variable=v160 complete dim=2

  int32_t v6;	// L195
  v6 = 0;	// L196
  l_reduction_k6: for (int k6 = 0; k6 < 128; k6++) {	// L197
  #pragma HLS pipeline II=1
    int8_t v165 = v156.read(); // v156[k6];	// L198
    int8_t a6;	// L199
    a6 = v165;	// L200
    int8_t v167 = v157.read(); // v157[k6];	// L201
    int8_t b6;	// L202
    b6 = v167;	// L203
    int8_t v169 = a6;	// L204
    int8_t v170 = b6;	// L205
    int16_t v171 = v169;	// L206
    int16_t v172 = v170;	// L207
    int16_t v173 = v171 * v172;	// L208
    int32_t v174 = v6;	// L209
    ap_int<33> v175 = v174;	// L210
    ap_int<33> v176 = v173;	// L211
    ap_int<33> v177 = v175 + v176;	// L212
    int32_t v178 = v177;	// L213
    v6 = v178;	// L214
    int8_t v179 = a6;	// L215
    v158.write(v179); // v158[k6] = v179;	// L216
    int8_t v180 = b6;	// L217
    v159.write(v180); // v159[k6] = v180;	// L218
  }
  int32_t v181 = v6;	// L220
  v160[v161][v162] = v181;	// L221
}

void PE_kernel_gemm_7_0(
  hls::stream< int8_t > &v182 /* v182[128] */,
  hls::stream< int8_t > &v183 /* v183[128] */,
  hls::stream< int8_t > &v184 /* v184[128] */,
  hls::stream< int8_t > &v185 /* v185[128] */,
  int32_t v186[8][8],
  int v187,
  int v188
) {	// L224
  #pragma HLS stream variable=v182 depth=9
  #pragma HLS stream variable=v183 depth=9
  #pragma HLS stream variable=v184 depth=9
  #pragma HLS stream variable=v185 depth=9
  #pragma HLS array_partition variable=v186 complete dim=1
  #pragma HLS array_partition variable=v186 complete dim=2

  int32_t v7;	// L226
  v7 = 0;	// L227
  l_reduction_k7: for (int k7 = 0; k7 < 128; k7++) {	// L228
  #pragma HLS pipeline II=1
    int8_t v191 = v182.read(); // v182[k7];	// L229
    int8_t a7;	// L230
    a7 = v191;	// L231
    int8_t v193 = v183.read(); // v183[k7];	// L232
    int8_t b7;	// L233
    b7 = v193;	// L234
    int8_t v195 = a7;	// L235
    int8_t v196 = b7;	// L236
    int16_t v197 = v195;	// L237
    int16_t v198 = v196;	// L238
    int16_t v199 = v197 * v198;	// L239
    int32_t v200 = v7;	// L240
    ap_int<33> v201 = v200;	// L241
    ap_int<33> v202 = v199;	// L242
    ap_int<33> v203 = v201 + v202;	// L243
    int32_t v204 = v203;	// L244
    v7 = v204;	// L245
    int8_t v205 = a7;	// L246
    v184.write(v205); // v184[k7] = v205;	// L247
    int8_t v206 = b7;	// L248
    v185.write(v206); // v185[k7] = v206;	// L249
  }
  int32_t v207 = v7;	// L251
  v186[v187][v188] = v207;	// L252
}

void PE_kernel_gemm_0_1(
  hls::stream< int8_t > &v208 /* v208[128] */,
  hls::stream< int8_t > &v209 /* v209[128] */,
  hls::stream< int8_t > &v210 /* v210[128] */,
  hls::stream< int8_t > &v211 /* v211[128] */,
  int32_t v212[8][8],
  int v213,
  int v214
) {	// L255
  #pragma HLS stream variable=v208 depth=9
  #pragma HLS stream variable=v209 depth=9
  #pragma HLS stream variable=v210 depth=9
  #pragma HLS stream variable=v211 depth=9
  #pragma HLS array_partition variable=v212 complete dim=1
  #pragma HLS array_partition variable=v212 complete dim=2

  int32_t v8;	// L257
  v8 = 0;	// L258
  l_reduction_k8: for (int k8 = 0; k8 < 128; k8++) {	// L259
  #pragma HLS pipeline II=1
    int8_t v217 = v208.read(); // v208[k8];	// L260
    int8_t a8;	// L261
    a8 = v217;	// L262
    int8_t v219 = v209.read(); // v209[k8];	// L263
    int8_t b8;	// L264
    b8 = v219;	// L265
    int8_t v221 = a8;	// L266
    int8_t v222 = b8;	// L267
    int16_t v223 = v221;	// L268
    int16_t v224 = v222;	// L269
    int16_t v225 = v223 * v224;	// L270
    int32_t v226 = v8;	// L271
    ap_int<33> v227 = v226;	// L272
    ap_int<33> v228 = v225;	// L273
    ap_int<33> v229 = v227 + v228;	// L274
    int32_t v230 = v229;	// L275
    v8 = v230;	// L276
    int8_t v231 = a8;	// L277
    v210.write(v231); // v210[k8] = v231;	// L278
    int8_t v232 = b8;	// L279
    v211.write(v232); // v211[k8] = v232;	// L280
  }
  int32_t v233 = v8;	// L282
  v212[v213][v214] = v233;	// L283
}

void PE_kernel_gemm_1_1(
  hls::stream< int8_t > &v234 /* v234[128] */,
  hls::stream< int8_t > &v235 /* v235[128] */,
  hls::stream< int8_t > &v236 /* v236[128] */,
  hls::stream< int8_t > &v237 /* v237[128] */,
  int32_t v238[8][8],
  int v239,
  int v240
) {	// L286
  #pragma HLS stream variable=v234 depth=9
  #pragma HLS stream variable=v235 depth=9
  #pragma HLS stream variable=v236 depth=9
  #pragma HLS stream variable=v237 depth=9
  #pragma HLS array_partition variable=v238 complete dim=1
  #pragma HLS array_partition variable=v238 complete dim=2

  int32_t v9;	// L288
  v9 = 0;	// L289
  l_reduction_k9: for (int k9 = 0; k9 < 128; k9++) {	// L290
  #pragma HLS pipeline II=1
    int8_t v243 = v234.read(); // v234[k9];	// L291
    int8_t a9;	// L292
    a9 = v243;	// L293
    int8_t v245 = v235.read(); // v235[k9];	// L294
    int8_t b9;	// L295
    b9 = v245;	// L296
    int8_t v247 = a9;	// L297
    int8_t v248 = b9;	// L298
    int16_t v249 = v247;	// L299
    int16_t v250 = v248;	// L300
    int16_t v251 = v249 * v250;	// L301
    int32_t v252 = v9;	// L302
    ap_int<33> v253 = v252;	// L303
    ap_int<33> v254 = v251;	// L304
    ap_int<33> v255 = v253 + v254;	// L305
    int32_t v256 = v255;	// L306
    v9 = v256;	// L307
    int8_t v257 = a9;	// L308
    v236.write(v257); // v236[k9] = v257;	// L309
    int8_t v258 = b9;	// L310
    v237.write(v258); // v237[k9] = v258;	// L311
  }
  int32_t v259 = v9;	// L313
  v238[v239][v240] = v259;	// L314
}

void PE_kernel_gemm_2_1(
  hls::stream< int8_t > &v260 /* v260[128] */,
  hls::stream< int8_t > &v261 /* v261[128] */,
  hls::stream< int8_t > &v262 /* v262[128] */,
  hls::stream< int8_t > &v263 /* v263[128] */,
  int32_t v264[8][8],
  int v265,
  int v266
) {	// L317
  #pragma HLS stream variable=v260 depth=9
  #pragma HLS stream variable=v261 depth=9
  #pragma HLS stream variable=v262 depth=9
  #pragma HLS stream variable=v263 depth=9
  #pragma HLS array_partition variable=v264 complete dim=1
  #pragma HLS array_partition variable=v264 complete dim=2

  int32_t v10;	// L319
  v10 = 0;	// L320
  l_reduction_k10: for (int k10 = 0; k10 < 128; k10++) {	// L321
  #pragma HLS pipeline II=1
    int8_t v269 = v260.read(); // v260[k10];	// L322
    int8_t a10;	// L323
    a10 = v269;	// L324
    int8_t v271 = v261.read(); // v261[k10];	// L325
    int8_t b10;	// L326
    b10 = v271;	// L327
    int8_t v273 = a10;	// L328
    int8_t v274 = b10;	// L329
    int16_t v275 = v273;	// L330
    int16_t v276 = v274;	// L331
    int16_t v277 = v275 * v276;	// L332
    int32_t v278 = v10;	// L333
    ap_int<33> v279 = v278;	// L334
    ap_int<33> v280 = v277;	// L335
    ap_int<33> v281 = v279 + v280;	// L336
    int32_t v282 = v281;	// L337
    v10 = v282;	// L338
    int8_t v283 = a10;	// L339
    v262.write(v283); // v262[k10] = v283;	// L340
    int8_t v284 = b10;	// L341
    v263.write(v284); // v263[k10] = v284;	// L342
  }
  int32_t v285 = v10;	// L344
  v264[v265][v266] = v285;	// L345
}

void PE_kernel_gemm_3_1(
  hls::stream< int8_t > &v286 /* v286[128] */,
  hls::stream< int8_t > &v287 /* v287[128] */,
  hls::stream< int8_t > &v288 /* v288[128] */,
  hls::stream< int8_t > &v289 /* v289[128] */,
  int32_t v290[8][8],
  int v291,
  int v292
) {	// L348
  #pragma HLS stream variable=v286 depth=9
  #pragma HLS stream variable=v287 depth=9
  #pragma HLS stream variable=v288 depth=9
  #pragma HLS stream variable=v289 depth=9
  #pragma HLS array_partition variable=v290 complete dim=1
  #pragma HLS array_partition variable=v290 complete dim=2

  int32_t v11;	// L350
  v11 = 0;	// L351
  l_reduction_k11: for (int k11 = 0; k11 < 128; k11++) {	// L352
  #pragma HLS pipeline II=1
    int8_t v295 = v286.read(); // v286[k11];	// L353
    int8_t a11;	// L354
    a11 = v295;	// L355
    int8_t v297 = v287.read(); // v287[k11];	// L356
    int8_t b11;	// L357
    b11 = v297;	// L358
    int8_t v299 = a11;	// L359
    int8_t v300 = b11;	// L360
    int16_t v301 = v299;	// L361
    int16_t v302 = v300;	// L362
    int16_t v303 = v301 * v302;	// L363
    int32_t v304 = v11;	// L364
    ap_int<33> v305 = v304;	// L365
    ap_int<33> v306 = v303;	// L366
    ap_int<33> v307 = v305 + v306;	// L367
    int32_t v308 = v307;	// L368
    v11 = v308;	// L369
    int8_t v309 = a11;	// L370
    v288.write(v309); // v288[k11] = v309;	// L371
    int8_t v310 = b11;	// L372
    v289.write(v310); // v289[k11] = v310;	// L373
  }
  int32_t v311 = v11;	// L375
  v290[v291][v292] = v311;	// L376
}

void PE_kernel_gemm_4_1(
  hls::stream< int8_t > &v312 /* v312[128] */,
  hls::stream< int8_t > &v313 /* v313[128] */,
  hls::stream< int8_t > &v314 /* v314[128] */,
  hls::stream< int8_t > &v315 /* v315[128] */,
  int32_t v316[8][8],
  int v317,
  int v318
) {	// L379
  #pragma HLS stream variable=v312 depth=9
  #pragma HLS stream variable=v313 depth=9
  #pragma HLS stream variable=v314 depth=9
  #pragma HLS stream variable=v315 depth=9
  #pragma HLS array_partition variable=v316 complete dim=1
  #pragma HLS array_partition variable=v316 complete dim=2

  int32_t v12;	// L381
  v12 = 0;	// L382
  l_reduction_k12: for (int k12 = 0; k12 < 128; k12++) {	// L383
  #pragma HLS pipeline II=1
    int8_t v321 = v312.read(); // v312[k12];	// L384
    int8_t a12;	// L385
    a12 = v321;	// L386
    int8_t v323 = v313.read(); // v313[k12];	// L387
    int8_t b12;	// L388
    b12 = v323;	// L389
    int8_t v325 = a12;	// L390
    int8_t v326 = b12;	// L391
    int16_t v327 = v325;	// L392
    int16_t v328 = v326;	// L393
    int16_t v329 = v327 * v328;	// L394
    int32_t v330 = v12;	// L395
    ap_int<33> v331 = v330;	// L396
    ap_int<33> v332 = v329;	// L397
    ap_int<33> v333 = v331 + v332;	// L398
    int32_t v334 = v333;	// L399
    v12 = v334;	// L400
    int8_t v335 = a12;	// L401
    v314.write(v335); // v314[k12] = v335;	// L402
    int8_t v336 = b12;	// L403
    v315.write(v336); // v315[k12] = v336;	// L404
  }
  int32_t v337 = v12;	// L406
  v316[v317][v318] = v337;	// L407
}

void PE_kernel_gemm_5_1(
  hls::stream< int8_t > &v338 /* v338[128] */,
  hls::stream< int8_t > &v339 /* v339[128] */,
  hls::stream< int8_t > &v340 /* v340[128] */,
  hls::stream< int8_t > &v341 /* v341[128] */,
  int32_t v342[8][8],
  int v343,
  int v344
) {	// L410
  #pragma HLS stream variable=v338 depth=9
  #pragma HLS stream variable=v339 depth=9
  #pragma HLS stream variable=v340 depth=9
  #pragma HLS stream variable=v341 depth=9
  #pragma HLS array_partition variable=v342 complete dim=1
  #pragma HLS array_partition variable=v342 complete dim=2

  int32_t v13;	// L412
  v13 = 0;	// L413
  l_reduction_k13: for (int k13 = 0; k13 < 128; k13++) {	// L414
  #pragma HLS pipeline II=1
    int8_t v347 = v338.read(); // v338[k13];	// L415
    int8_t a13;	// L416
    a13 = v347;	// L417
    int8_t v349 = v339.read(); // v339[k13];	// L418
    int8_t b13;	// L419
    b13 = v349;	// L420
    int8_t v351 = a13;	// L421
    int8_t v352 = b13;	// L422
    int16_t v353 = v351;	// L423
    int16_t v354 = v352;	// L424
    int16_t v355 = v353 * v354;	// L425
    int32_t v356 = v13;	// L426
    ap_int<33> v357 = v356;	// L427
    ap_int<33> v358 = v355;	// L428
    ap_int<33> v359 = v357 + v358;	// L429
    int32_t v360 = v359;	// L430
    v13 = v360;	// L431
    int8_t v361 = a13;	// L432
    v340.write(v361); // v340[k13] = v361;	// L433
    int8_t v362 = b13;	// L434
    v341.write(v362); // v341[k13] = v362;	// L435
  }
  int32_t v363 = v13;	// L437
  v342[v343][v344] = v363;	// L438
}

void PE_kernel_gemm_6_1(
  hls::stream< int8_t > &v364 /* v364[128] */,
  hls::stream< int8_t > &v365 /* v365[128] */,
  hls::stream< int8_t > &v366 /* v366[128] */,
  hls::stream< int8_t > &v367 /* v367[128] */,
  int32_t v368[8][8],
  int v369,
  int v370
) {	// L441
  #pragma HLS stream variable=v364 depth=9
  #pragma HLS stream variable=v365 depth=9
  #pragma HLS stream variable=v366 depth=9
  #pragma HLS stream variable=v367 depth=9
  #pragma HLS array_partition variable=v368 complete dim=1
  #pragma HLS array_partition variable=v368 complete dim=2

  int32_t v14;	// L443
  v14 = 0;	// L444
  l_reduction_k14: for (int k14 = 0; k14 < 128; k14++) {	// L445
  #pragma HLS pipeline II=1
    int8_t v373 = v364.read(); // v364[k14];	// L446
    int8_t a14;	// L447
    a14 = v373;	// L448
    int8_t v375 = v365.read(); // v365[k14];	// L449
    int8_t b14;	// L450
    b14 = v375;	// L451
    int8_t v377 = a14;	// L452
    int8_t v378 = b14;	// L453
    int16_t v379 = v377;	// L454
    int16_t v380 = v378;	// L455
    int16_t v381 = v379 * v380;	// L456
    int32_t v382 = v14;	// L457
    ap_int<33> v383 = v382;	// L458
    ap_int<33> v384 = v381;	// L459
    ap_int<33> v385 = v383 + v384;	// L460
    int32_t v386 = v385;	// L461
    v14 = v386;	// L462
    int8_t v387 = a14;	// L463
    v366.write(v387); // v366[k14] = v387;	// L464
    int8_t v388 = b14;	// L465
    v367.write(v388); // v367[k14] = v388;	// L466
  }
  int32_t v389 = v14;	// L468
  v368[v369][v370] = v389;	// L469
}

void PE_kernel_gemm_7_1(
  hls::stream< int8_t > &v390 /* v390[128] */,
  hls::stream< int8_t > &v391 /* v391[128] */,
  hls::stream< int8_t > &v392 /* v392[128] */,
  hls::stream< int8_t > &v393 /* v393[128] */,
  int32_t v394[8][8],
  int v395,
  int v396
) {	// L472
  #pragma HLS stream variable=v390 depth=9
  #pragma HLS stream variable=v391 depth=9
  #pragma HLS stream variable=v392 depth=9
  #pragma HLS stream variable=v393 depth=9
  #pragma HLS array_partition variable=v394 complete dim=1
  #pragma HLS array_partition variable=v394 complete dim=2

  int32_t v15;	// L474
  v15 = 0;	// L475
  l_reduction_k15: for (int k15 = 0; k15 < 128; k15++) {	// L476
  #pragma HLS pipeline II=1
    int8_t v399 = v390.read(); // v390[k15];	// L477
    int8_t a15;	// L478
    a15 = v399;	// L479
    int8_t v401 = v391.read(); // v391[k15];	// L480
    int8_t b15;	// L481
    b15 = v401;	// L482
    int8_t v403 = a15;	// L483
    int8_t v404 = b15;	// L484
    int16_t v405 = v403;	// L485
    int16_t v406 = v404;	// L486
    int16_t v407 = v405 * v406;	// L487
    int32_t v408 = v15;	// L488
    ap_int<33> v409 = v408;	// L489
    ap_int<33> v410 = v407;	// L490
    ap_int<33> v411 = v409 + v410;	// L491
    int32_t v412 = v411;	// L492
    v15 = v412;	// L493
    int8_t v413 = a15;	// L494
    v392.write(v413); // v392[k15] = v413;	// L495
    int8_t v414 = b15;	// L496
    v393.write(v414); // v393[k15] = v414;	// L497
  }
  int32_t v415 = v15;	// L499
  v394[v395][v396] = v415;	// L500
}

void PE_kernel_gemm_0_2(
  hls::stream< int8_t > &v416 /* v416[128] */,
  hls::stream< int8_t > &v417 /* v417[128] */,
  hls::stream< int8_t > &v418 /* v418[128] */,
  hls::stream< int8_t > &v419 /* v419[128] */,
  int32_t v420[8][8],
  int v421,
  int v422
) {	// L503
  #pragma HLS stream variable=v416 depth=9
  #pragma HLS stream variable=v417 depth=9
  #pragma HLS stream variable=v418 depth=9
  #pragma HLS stream variable=v419 depth=9
  #pragma HLS array_partition variable=v420 complete dim=1
  #pragma HLS array_partition variable=v420 complete dim=2

  int32_t v16;	// L505
  v16 = 0;	// L506
  l_reduction_k16: for (int k16 = 0; k16 < 128; k16++) {	// L507
  #pragma HLS pipeline II=1
    int8_t v425 = v416.read(); // v416[k16];	// L508
    int8_t a16;	// L509
    a16 = v425;	// L510
    int8_t v427 = v417.read(); // v417[k16];	// L511
    int8_t b16;	// L512
    b16 = v427;	// L513
    int8_t v429 = a16;	// L514
    int8_t v430 = b16;	// L515
    int16_t v431 = v429;	// L516
    int16_t v432 = v430;	// L517
    int16_t v433 = v431 * v432;	// L518
    int32_t v434 = v16;	// L519
    ap_int<33> v435 = v434;	// L520
    ap_int<33> v436 = v433;	// L521
    ap_int<33> v437 = v435 + v436;	// L522
    int32_t v438 = v437;	// L523
    v16 = v438;	// L524
    int8_t v439 = a16;	// L525
    v418.write(v439); // v418[k16] = v439;	// L526
    int8_t v440 = b16;	// L527
    v419.write(v440); // v419[k16] = v440;	// L528
  }
  int32_t v441 = v16;	// L530
  v420[v421][v422] = v441;	// L531
}

void PE_kernel_gemm_1_2(
  hls::stream< int8_t > &v442 /* v442[128] */,
  hls::stream< int8_t > &v443 /* v443[128] */,
  hls::stream< int8_t > &v444 /* v444[128] */,
  hls::stream< int8_t > &v445 /* v445[128] */,
  int32_t v446[8][8],
  int v447,
  int v448
) {	// L534
  #pragma HLS stream variable=v442 depth=9
  #pragma HLS stream variable=v443 depth=9
  #pragma HLS stream variable=v444 depth=9
  #pragma HLS stream variable=v445 depth=9
  #pragma HLS array_partition variable=v446 complete dim=1
  #pragma HLS array_partition variable=v446 complete dim=2

  int32_t v17;	// L536
  v17 = 0;	// L537
  l_reduction_k17: for (int k17 = 0; k17 < 128; k17++) {	// L538
  #pragma HLS pipeline II=1
    int8_t v451 = v442.read(); // v442[k17];	// L539
    int8_t a17;	// L540
    a17 = v451;	// L541
    int8_t v453 = v443.read(); // v443[k17];	// L542
    int8_t b17;	// L543
    b17 = v453;	// L544
    int8_t v455 = a17;	// L545
    int8_t v456 = b17;	// L546
    int16_t v457 = v455;	// L547
    int16_t v458 = v456;	// L548
    int16_t v459 = v457 * v458;	// L549
    int32_t v460 = v17;	// L550
    ap_int<33> v461 = v460;	// L551
    ap_int<33> v462 = v459;	// L552
    ap_int<33> v463 = v461 + v462;	// L553
    int32_t v464 = v463;	// L554
    v17 = v464;	// L555
    int8_t v465 = a17;	// L556
    v444.write(v465); // v444[k17] = v465;	// L557
    int8_t v466 = b17;	// L558
    v445.write(v466); // v445[k17] = v466;	// L559
  }
  int32_t v467 = v17;	// L561
  v446[v447][v448] = v467;	// L562
}

void PE_kernel_gemm_2_2(
  hls::stream< int8_t > &v468 /* v468[128] */,
  hls::stream< int8_t > &v469 /* v469[128] */,
  hls::stream< int8_t > &v470 /* v470[128] */,
  hls::stream< int8_t > &v471 /* v471[128] */,
  int32_t v472[8][8],
  int v473,
  int v474
) {	// L565
  #pragma HLS stream variable=v468 depth=9
  #pragma HLS stream variable=v469 depth=9
  #pragma HLS stream variable=v470 depth=9
  #pragma HLS stream variable=v471 depth=9
  #pragma HLS array_partition variable=v472 complete dim=1
  #pragma HLS array_partition variable=v472 complete dim=2

  int32_t v18;	// L567
  v18 = 0;	// L568
  l_reduction_k18: for (int k18 = 0; k18 < 128; k18++) {	// L569
  #pragma HLS pipeline II=1
    int8_t v477 = v468.read(); // v468[k18];	// L570
    int8_t a18;	// L571
    a18 = v477;	// L572
    int8_t v479 = v469.read(); // v469[k18];	// L573
    int8_t b18;	// L574
    b18 = v479;	// L575
    int8_t v481 = a18;	// L576
    int8_t v482 = b18;	// L577
    int16_t v483 = v481;	// L578
    int16_t v484 = v482;	// L579
    int16_t v485 = v483 * v484;	// L580
    int32_t v486 = v18;	// L581
    ap_int<33> v487 = v486;	// L582
    ap_int<33> v488 = v485;	// L583
    ap_int<33> v489 = v487 + v488;	// L584
    int32_t v490 = v489;	// L585
    v18 = v490;	// L586
    int8_t v491 = a18;	// L587
    v470.write(v491); // v470[k18] = v491;	// L588
    int8_t v492 = b18;	// L589
    v471.write(v492); // v471[k18] = v492;	// L590
  }
  int32_t v493 = v18;	// L592
  v472[v473][v474] = v493;	// L593
}

void PE_kernel_gemm_3_2(
  hls::stream< int8_t > &v494 /* v494[128] */,
  hls::stream< int8_t > &v495 /* v495[128] */,
  hls::stream< int8_t > &v496 /* v496[128] */,
  hls::stream< int8_t > &v497 /* v497[128] */,
  int32_t v498[8][8],
  int v499,
  int v500
) {	// L596
  #pragma HLS stream variable=v494 depth=9
  #pragma HLS stream variable=v495 depth=9
  #pragma HLS stream variable=v496 depth=9
  #pragma HLS stream variable=v497 depth=9
  #pragma HLS array_partition variable=v498 complete dim=1
  #pragma HLS array_partition variable=v498 complete dim=2

  int32_t v19;	// L598
  v19 = 0;	// L599
  l_reduction_k19: for (int k19 = 0; k19 < 128; k19++) {	// L600
  #pragma HLS pipeline II=1
    int8_t v503 = v494.read(); // v494[k19];	// L601
    int8_t a19;	// L602
    a19 = v503;	// L603
    int8_t v505 = v495.read(); // v495[k19];	// L604
    int8_t b19;	// L605
    b19 = v505;	// L606
    int8_t v507 = a19;	// L607
    int8_t v508 = b19;	// L608
    int16_t v509 = v507;	// L609
    int16_t v510 = v508;	// L610
    int16_t v511 = v509 * v510;	// L611
    int32_t v512 = v19;	// L612
    ap_int<33> v513 = v512;	// L613
    ap_int<33> v514 = v511;	// L614
    ap_int<33> v515 = v513 + v514;	// L615
    int32_t v516 = v515;	// L616
    v19 = v516;	// L617
    int8_t v517 = a19;	// L618
    v496.write(v517); // v496[k19] = v517;	// L619
    int8_t v518 = b19;	// L620
    v497.write(v518); // v497[k19] = v518;	// L621
  }
  int32_t v519 = v19;	// L623
  v498[v499][v500] = v519;	// L624
}

void PE_kernel_gemm_4_2(
  hls::stream< int8_t > &v520 /* v520[128] */,
  hls::stream< int8_t > &v521 /* v521[128] */,
  hls::stream< int8_t > &v522 /* v522[128] */,
  hls::stream< int8_t > &v523 /* v523[128] */,
  int32_t v524[8][8],
  int v525,
  int v526
) {	// L627
  #pragma HLS stream variable=v520 depth=9
  #pragma HLS stream variable=v521 depth=9
  #pragma HLS stream variable=v522 depth=9
  #pragma HLS stream variable=v523 depth=9
  #pragma HLS array_partition variable=v524 complete dim=1
  #pragma HLS array_partition variable=v524 complete dim=2

  int32_t v20;	// L629
  v20 = 0;	// L630
  l_reduction_k20: for (int k20 = 0; k20 < 128; k20++) {	// L631
  #pragma HLS pipeline II=1
    int8_t v529 = v520.read(); // v520[k20];	// L632
    int8_t a20;	// L633
    a20 = v529;	// L634
    int8_t v531 = v521.read(); // v521[k20];	// L635
    int8_t b20;	// L636
    b20 = v531;	// L637
    int8_t v533 = a20;	// L638
    int8_t v534 = b20;	// L639
    int16_t v535 = v533;	// L640
    int16_t v536 = v534;	// L641
    int16_t v537 = v535 * v536;	// L642
    int32_t v538 = v20;	// L643
    ap_int<33> v539 = v538;	// L644
    ap_int<33> v540 = v537;	// L645
    ap_int<33> v541 = v539 + v540;	// L646
    int32_t v542 = v541;	// L647
    v20 = v542;	// L648
    int8_t v543 = a20;	// L649
    v522.write(v543); // v522[k20] = v543;	// L650
    int8_t v544 = b20;	// L651
    v523.write(v544); // v523[k20] = v544;	// L652
  }
  int32_t v545 = v20;	// L654
  v524[v525][v526] = v545;	// L655
}

void PE_kernel_gemm_5_2(
  hls::stream< int8_t > &v546 /* v546[128] */,
  hls::stream< int8_t > &v547 /* v547[128] */,
  hls::stream< int8_t > &v548 /* v548[128] */,
  hls::stream< int8_t > &v549 /* v549[128] */,
  int32_t v550[8][8],
  int v551,
  int v552
) {	// L658
  #pragma HLS stream variable=v546 depth=9
  #pragma HLS stream variable=v547 depth=9
  #pragma HLS stream variable=v548 depth=9
  #pragma HLS stream variable=v549 depth=9
  #pragma HLS array_partition variable=v550 complete dim=1
  #pragma HLS array_partition variable=v550 complete dim=2

  int32_t v21;	// L660
  v21 = 0;	// L661
  l_reduction_k21: for (int k21 = 0; k21 < 128; k21++) {	// L662
  #pragma HLS pipeline II=1
    int8_t v555 = v546.read(); // v546[k21];	// L663
    int8_t a21;	// L664
    a21 = v555;	// L665
    int8_t v557 = v547.read(); // v547[k21];	// L666
    int8_t b21;	// L667
    b21 = v557;	// L668
    int8_t v559 = a21;	// L669
    int8_t v560 = b21;	// L670
    int16_t v561 = v559;	// L671
    int16_t v562 = v560;	// L672
    int16_t v563 = v561 * v562;	// L673
    int32_t v564 = v21;	// L674
    ap_int<33> v565 = v564;	// L675
    ap_int<33> v566 = v563;	// L676
    ap_int<33> v567 = v565 + v566;	// L677
    int32_t v568 = v567;	// L678
    v21 = v568;	// L679
    int8_t v569 = a21;	// L680
    v548.write(v569); // v548[k21] = v569;	// L681
    int8_t v570 = b21;	// L682
    v549.write(v570); // v549[k21] = v570;	// L683
  }
  int32_t v571 = v21;	// L685
  v550[v551][v552] = v571;	// L686
}

void PE_kernel_gemm_6_2(
  hls::stream< int8_t > &v572 /* v572[128] */,
  hls::stream< int8_t > &v573 /* v573[128] */,
  hls::stream< int8_t > &v574 /* v574[128] */,
  hls::stream< int8_t > &v575 /* v575[128] */,
  int32_t v576[8][8],
  int v577,
  int v578
) {	// L689
  #pragma HLS stream variable=v572 depth=9
  #pragma HLS stream variable=v573 depth=9
  #pragma HLS stream variable=v574 depth=9
  #pragma HLS stream variable=v575 depth=9
  #pragma HLS array_partition variable=v576 complete dim=1
  #pragma HLS array_partition variable=v576 complete dim=2

  int32_t v22;	// L691
  v22 = 0;	// L692
  l_reduction_k22: for (int k22 = 0; k22 < 128; k22++) {	// L693
  #pragma HLS pipeline II=1
    int8_t v581 = v572.read(); // v572[k22];	// L694
    int8_t a22;	// L695
    a22 = v581;	// L696
    int8_t v583 = v573.read(); // v573[k22];	// L697
    int8_t b22;	// L698
    b22 = v583;	// L699
    int8_t v585 = a22;	// L700
    int8_t v586 = b22;	// L701
    int16_t v587 = v585;	// L702
    int16_t v588 = v586;	// L703
    int16_t v589 = v587 * v588;	// L704
    int32_t v590 = v22;	// L705
    ap_int<33> v591 = v590;	// L706
    ap_int<33> v592 = v589;	// L707
    ap_int<33> v593 = v591 + v592;	// L708
    int32_t v594 = v593;	// L709
    v22 = v594;	// L710
    int8_t v595 = a22;	// L711
    v574.write(v595); // v574[k22] = v595;	// L712
    int8_t v596 = b22;	// L713
    v575.write(v596); // v575[k22] = v596;	// L714
  }
  int32_t v597 = v22;	// L716
  v576[v577][v578] = v597;	// L717
}

void PE_kernel_gemm_7_2(
  hls::stream< int8_t > &v598 /* v598[128] */,
  hls::stream< int8_t > &v599 /* v599[128] */,
  hls::stream< int8_t > &v600 /* v600[128] */,
  hls::stream< int8_t > &v601 /* v601[128] */,
  int32_t v602[8][8],
  int v603,
  int v604
) {	// L720
  #pragma HLS stream variable=v598 depth=9
  #pragma HLS stream variable=v599 depth=9
  #pragma HLS stream variable=v600 depth=9
  #pragma HLS stream variable=v601 depth=9
  #pragma HLS array_partition variable=v602 complete dim=1
  #pragma HLS array_partition variable=v602 complete dim=2

  int32_t v23;	// L722
  v23 = 0;	// L723
  l_reduction_k23: for (int k23 = 0; k23 < 128; k23++) {	// L724
  #pragma HLS pipeline II=1
    int8_t v607 = v598.read(); // v598[k23];	// L725
    int8_t a23;	// L726
    a23 = v607;	// L727
    int8_t v609 = v599.read(); // v599[k23];	// L728
    int8_t b23;	// L729
    b23 = v609;	// L730
    int8_t v611 = a23;	// L731
    int8_t v612 = b23;	// L732
    int16_t v613 = v611;	// L733
    int16_t v614 = v612;	// L734
    int16_t v615 = v613 * v614;	// L735
    int32_t v616 = v23;	// L736
    ap_int<33> v617 = v616;	// L737
    ap_int<33> v618 = v615;	// L738
    ap_int<33> v619 = v617 + v618;	// L739
    int32_t v620 = v619;	// L740
    v23 = v620;	// L741
    int8_t v621 = a23;	// L742
    v600.write(v621); // v600[k23] = v621;	// L743
    int8_t v622 = b23;	// L744
    v601.write(v622); // v601[k23] = v622;	// L745
  }
  int32_t v623 = v23;	// L747
  v602[v603][v604] = v623;	// L748
}

void PE_kernel_gemm_0_3(
  hls::stream< int8_t > &v624 /* v624[128] */,
  hls::stream< int8_t > &v625 /* v625[128] */,
  hls::stream< int8_t > &v626 /* v626[128] */,
  hls::stream< int8_t > &v627 /* v627[128] */,
  int32_t v628[8][8],
  int v629,
  int v630
) {	// L751
  #pragma HLS stream variable=v624 depth=9
  #pragma HLS stream variable=v625 depth=9
  #pragma HLS stream variable=v626 depth=9
  #pragma HLS stream variable=v627 depth=9
  #pragma HLS array_partition variable=v628 complete dim=1
  #pragma HLS array_partition variable=v628 complete dim=2

  int32_t v24;	// L753
  v24 = 0;	// L754
  l_reduction_k24: for (int k24 = 0; k24 < 128; k24++) {	// L755
  #pragma HLS pipeline II=1
    int8_t v633 = v624.read(); // v624[k24];	// L756
    int8_t a24;	// L757
    a24 = v633;	// L758
    int8_t v635 = v625.read(); // v625[k24];	// L759
    int8_t b24;	// L760
    b24 = v635;	// L761
    int8_t v637 = a24;	// L762
    int8_t v638 = b24;	// L763
    int16_t v639 = v637;	// L764
    int16_t v640 = v638;	// L765
    int16_t v641 = v639 * v640;	// L766
    int32_t v642 = v24;	// L767
    ap_int<33> v643 = v642;	// L768
    ap_int<33> v644 = v641;	// L769
    ap_int<33> v645 = v643 + v644;	// L770
    int32_t v646 = v645;	// L771
    v24 = v646;	// L772
    int8_t v647 = a24;	// L773
    v626.write(v647); // v626[k24] = v647;	// L774
    int8_t v648 = b24;	// L775
    v627.write(v648); // v627[k24] = v648;	// L776
  }
  int32_t v649 = v24;	// L778
  v628[v629][v630] = v649;	// L779
}

void PE_kernel_gemm_1_3(
  hls::stream< int8_t > &v650 /* v650[128] */,
  hls::stream< int8_t > &v651 /* v651[128] */,
  hls::stream< int8_t > &v652 /* v652[128] */,
  hls::stream< int8_t > &v653 /* v653[128] */,
  int32_t v654[8][8],
  int v655,
  int v656
) {	// L782
  #pragma HLS stream variable=v650 depth=9
  #pragma HLS stream variable=v651 depth=9
  #pragma HLS stream variable=v652 depth=9
  #pragma HLS stream variable=v653 depth=9
  #pragma HLS array_partition variable=v654 complete dim=1
  #pragma HLS array_partition variable=v654 complete dim=2

  int32_t v25;	// L784
  v25 = 0;	// L785
  l_reduction_k25: for (int k25 = 0; k25 < 128; k25++) {	// L786
  #pragma HLS pipeline II=1
    int8_t v659 = v650.read(); // v650[k25];	// L787
    int8_t a25;	// L788
    a25 = v659;	// L789
    int8_t v661 = v651.read(); // v651[k25];	// L790
    int8_t b25;	// L791
    b25 = v661;	// L792
    int8_t v663 = a25;	// L793
    int8_t v664 = b25;	// L794
    int16_t v665 = v663;	// L795
    int16_t v666 = v664;	// L796
    int16_t v667 = v665 * v666;	// L797
    int32_t v668 = v25;	// L798
    ap_int<33> v669 = v668;	// L799
    ap_int<33> v670 = v667;	// L800
    ap_int<33> v671 = v669 + v670;	// L801
    int32_t v672 = v671;	// L802
    v25 = v672;	// L803
    int8_t v673 = a25;	// L804
    v652.write(v673); // v652[k25] = v673;	// L805
    int8_t v674 = b25;	// L806
    v653.write(v674); // v653[k25] = v674;	// L807
  }
  int32_t v675 = v25;	// L809
  v654[v655][v656] = v675;	// L810
}

void PE_kernel_gemm_2_3(
  hls::stream< int8_t > &v676 /* v676[128] */,
  hls::stream< int8_t > &v677 /* v677[128] */,
  hls::stream< int8_t > &v678 /* v678[128] */,
  hls::stream< int8_t > &v679 /* v679[128] */,
  int32_t v680[8][8],
  int v681,
  int v682
) {	// L813
  #pragma HLS stream variable=v676 depth=9
  #pragma HLS stream variable=v677 depth=9
  #pragma HLS stream variable=v678 depth=9
  #pragma HLS stream variable=v679 depth=9
  #pragma HLS array_partition variable=v680 complete dim=1
  #pragma HLS array_partition variable=v680 complete dim=2

  int32_t v26;	// L815
  v26 = 0;	// L816
  l_reduction_k26: for (int k26 = 0; k26 < 128; k26++) {	// L817
  #pragma HLS pipeline II=1
    int8_t v685 = v676.read(); // v676[k26];	// L818
    int8_t a26;	// L819
    a26 = v685;	// L820
    int8_t v687 = v677.read(); // v677[k26];	// L821
    int8_t b26;	// L822
    b26 = v687;	// L823
    int8_t v689 = a26;	// L824
    int8_t v690 = b26;	// L825
    int16_t v691 = v689;	// L826
    int16_t v692 = v690;	// L827
    int16_t v693 = v691 * v692;	// L828
    int32_t v694 = v26;	// L829
    ap_int<33> v695 = v694;	// L830
    ap_int<33> v696 = v693;	// L831
    ap_int<33> v697 = v695 + v696;	// L832
    int32_t v698 = v697;	// L833
    v26 = v698;	// L834
    int8_t v699 = a26;	// L835
    v678.write(v699); // v678[k26] = v699;	// L836
    int8_t v700 = b26;	// L837
    v679.write(v700); // v679[k26] = v700;	// L838
  }
  int32_t v701 = v26;	// L840
  v680[v681][v682] = v701;	// L841
}

void PE_kernel_gemm_3_3(
  hls::stream< int8_t > &v702 /* v702[128] */,
  hls::stream< int8_t > &v703 /* v703[128] */,
  hls::stream< int8_t > &v704 /* v704[128] */,
  hls::stream< int8_t > &v705 /* v705[128] */,
  int32_t v706[8][8],
  int v707,
  int v708
) {	// L844
  #pragma HLS stream variable=v702 depth=9
  #pragma HLS stream variable=v703 depth=9
  #pragma HLS stream variable=v704 depth=9
  #pragma HLS stream variable=v705 depth=9
  #pragma HLS array_partition variable=v706 complete dim=1
  #pragma HLS array_partition variable=v706 complete dim=2

  int32_t v27;	// L846
  v27 = 0;	// L847
  l_reduction_k27: for (int k27 = 0; k27 < 128; k27++) {	// L848
  #pragma HLS pipeline II=1
    int8_t v711 = v702.read(); // v702[k27];	// L849
    int8_t a27;	// L850
    a27 = v711;	// L851
    int8_t v713 = v703.read(); // v703[k27];	// L852
    int8_t b27;	// L853
    b27 = v713;	// L854
    int8_t v715 = a27;	// L855
    int8_t v716 = b27;	// L856
    int16_t v717 = v715;	// L857
    int16_t v718 = v716;	// L858
    int16_t v719 = v717 * v718;	// L859
    int32_t v720 = v27;	// L860
    ap_int<33> v721 = v720;	// L861
    ap_int<33> v722 = v719;	// L862
    ap_int<33> v723 = v721 + v722;	// L863
    int32_t v724 = v723;	// L864
    v27 = v724;	// L865
    int8_t v725 = a27;	// L866
    v704.write(v725); // v704[k27] = v725;	// L867
    int8_t v726 = b27;	// L868
    v705.write(v726); // v705[k27] = v726;	// L869
  }
  int32_t v727 = v27;	// L871
  v706[v707][v708] = v727;	// L872
}

void PE_kernel_gemm_4_3(
  hls::stream< int8_t > &v728 /* v728[128] */,
  hls::stream< int8_t > &v729 /* v729[128] */,
  hls::stream< int8_t > &v730 /* v730[128] */,
  hls::stream< int8_t > &v731 /* v731[128] */,
  int32_t v732[8][8],
  int v733,
  int v734
) {	// L875
  #pragma HLS stream variable=v728 depth=9
  #pragma HLS stream variable=v729 depth=9
  #pragma HLS stream variable=v730 depth=9
  #pragma HLS stream variable=v731 depth=9
  #pragma HLS array_partition variable=v732 complete dim=1
  #pragma HLS array_partition variable=v732 complete dim=2

  int32_t v28;	// L877
  v28 = 0;	// L878
  l_reduction_k28: for (int k28 = 0; k28 < 128; k28++) {	// L879
  #pragma HLS pipeline II=1
    int8_t v737 = v728.read(); // v728[k28];	// L880
    int8_t a28;	// L881
    a28 = v737;	// L882
    int8_t v739 = v729.read(); // v729[k28];	// L883
    int8_t b28;	// L884
    b28 = v739;	// L885
    int8_t v741 = a28;	// L886
    int8_t v742 = b28;	// L887
    int16_t v743 = v741;	// L888
    int16_t v744 = v742;	// L889
    int16_t v745 = v743 * v744;	// L890
    int32_t v746 = v28;	// L891
    ap_int<33> v747 = v746;	// L892
    ap_int<33> v748 = v745;	// L893
    ap_int<33> v749 = v747 + v748;	// L894
    int32_t v750 = v749;	// L895
    v28 = v750;	// L896
    int8_t v751 = a28;	// L897
    v730.write(v751); // v730[k28] = v751;	// L898
    int8_t v752 = b28;	// L899
    v731.write(v752); // v731[k28] = v752;	// L900
  }
  int32_t v753 = v28;	// L902
  v732[v733][v734] = v753;	// L903
}

void PE_kernel_gemm_5_3(
  hls::stream< int8_t > &v754 /* v754[128] */,
  hls::stream< int8_t > &v755 /* v755[128] */,
  hls::stream< int8_t > &v756 /* v756[128] */,
  hls::stream< int8_t > &v757 /* v757[128] */,
  int32_t v758[8][8],
  int v759,
  int v760
) {	// L906
  #pragma HLS stream variable=v754 depth=9
  #pragma HLS stream variable=v755 depth=9
  #pragma HLS stream variable=v756 depth=9
  #pragma HLS stream variable=v757 depth=9
  #pragma HLS array_partition variable=v758 complete dim=1
  #pragma HLS array_partition variable=v758 complete dim=2

  int32_t v29;	// L908
  v29 = 0;	// L909
  l_reduction_k29: for (int k29 = 0; k29 < 128; k29++) {	// L910
  #pragma HLS pipeline II=1
    int8_t v763 = v754.read(); // v754[k29];	// L911
    int8_t a29;	// L912
    a29 = v763;	// L913
    int8_t v765 = v755.read(); // v755[k29];	// L914
    int8_t b29;	// L915
    b29 = v765;	// L916
    int8_t v767 = a29;	// L917
    int8_t v768 = b29;	// L918
    int16_t v769 = v767;	// L919
    int16_t v770 = v768;	// L920
    int16_t v771 = v769 * v770;	// L921
    int32_t v772 = v29;	// L922
    ap_int<33> v773 = v772;	// L923
    ap_int<33> v774 = v771;	// L924
    ap_int<33> v775 = v773 + v774;	// L925
    int32_t v776 = v775;	// L926
    v29 = v776;	// L927
    int8_t v777 = a29;	// L928
    v756.write(v777); // v756[k29] = v777;	// L929
    int8_t v778 = b29;	// L930
    v757.write(v778); // v757[k29] = v778;	// L931
  }
  int32_t v779 = v29;	// L933
  v758[v759][v760] = v779;	// L934
}

void PE_kernel_gemm_6_3(
  hls::stream< int8_t > &v780 /* v780[128] */,
  hls::stream< int8_t > &v781 /* v781[128] */,
  hls::stream< int8_t > &v782 /* v782[128] */,
  hls::stream< int8_t > &v783 /* v783[128] */,
  int32_t v784[8][8],
  int v785,
  int v786
) {	// L937
  #pragma HLS stream variable=v780 depth=9
  #pragma HLS stream variable=v781 depth=9
  #pragma HLS stream variable=v782 depth=9
  #pragma HLS stream variable=v783 depth=9
  #pragma HLS array_partition variable=v784 complete dim=1
  #pragma HLS array_partition variable=v784 complete dim=2

  int32_t v30;	// L939
  v30 = 0;	// L940
  l_reduction_k30: for (int k30 = 0; k30 < 128; k30++) {	// L941
  #pragma HLS pipeline II=1
    int8_t v789 = v780.read(); // v780[k30];	// L942
    int8_t a30;	// L943
    a30 = v789;	// L944
    int8_t v791 = v781.read(); // v781[k30];	// L945
    int8_t b30;	// L946
    b30 = v791;	// L947
    int8_t v793 = a30;	// L948
    int8_t v794 = b30;	// L949
    int16_t v795 = v793;	// L950
    int16_t v796 = v794;	// L951
    int16_t v797 = v795 * v796;	// L952
    int32_t v798 = v30;	// L953
    ap_int<33> v799 = v798;	// L954
    ap_int<33> v800 = v797;	// L955
    ap_int<33> v801 = v799 + v800;	// L956
    int32_t v802 = v801;	// L957
    v30 = v802;	// L958
    int8_t v803 = a30;	// L959
    v782.write(v803); // v782[k30] = v803;	// L960
    int8_t v804 = b30;	// L961
    v783.write(v804); // v783[k30] = v804;	// L962
  }
  int32_t v805 = v30;	// L964
  v784[v785][v786] = v805;	// L965
}

void PE_kernel_gemm_7_3(
  hls::stream< int8_t > &v806 /* v806[128] */,
  hls::stream< int8_t > &v807 /* v807[128] */,
  hls::stream< int8_t > &v808 /* v808[128] */,
  hls::stream< int8_t > &v809 /* v809[128] */,
  int32_t v810[8][8],
  int v811,
  int v812
) {	// L968
  #pragma HLS stream variable=v806 depth=9
  #pragma HLS stream variable=v807 depth=9
  #pragma HLS stream variable=v808 depth=9
  #pragma HLS stream variable=v809 depth=9
  #pragma HLS array_partition variable=v810 complete dim=1
  #pragma HLS array_partition variable=v810 complete dim=2

  int32_t v31;	// L970
  v31 = 0;	// L971
  l_reduction_k31: for (int k31 = 0; k31 < 128; k31++) {	// L972
  #pragma HLS pipeline II=1
    int8_t v815 = v806.read(); // v806[k31];	// L973
    int8_t a31;	// L974
    a31 = v815;	// L975
    int8_t v817 = v807.read(); // v807[k31];	// L976
    int8_t b31;	// L977
    b31 = v817;	// L978
    int8_t v819 = a31;	// L979
    int8_t v820 = b31;	// L980
    int16_t v821 = v819;	// L981
    int16_t v822 = v820;	// L982
    int16_t v823 = v821 * v822;	// L983
    int32_t v824 = v31;	// L984
    ap_int<33> v825 = v824;	// L985
    ap_int<33> v826 = v823;	// L986
    ap_int<33> v827 = v825 + v826;	// L987
    int32_t v828 = v827;	// L988
    v31 = v828;	// L989
    int8_t v829 = a31;	// L990
    v808.write(v829); // v808[k31] = v829;	// L991
    int8_t v830 = b31;	// L992
    v809.write(v830); // v809[k31] = v830;	// L993
  }
  int32_t v831 = v31;	// L995
  v810[v811][v812] = v831;	// L996
}

void PE_kernel_gemm_0_4(
  hls::stream< int8_t > &v832 /* v832[128] */,
  hls::stream< int8_t > &v833 /* v833[128] */,
  hls::stream< int8_t > &v834 /* v834[128] */,
  hls::stream< int8_t > &v835 /* v835[128] */,
  int32_t v836[8][8],
  int v837,
  int v838
) {	// L999
  #pragma HLS stream variable=v832 depth=9
  #pragma HLS stream variable=v833 depth=9
  #pragma HLS stream variable=v834 depth=9
  #pragma HLS stream variable=v835 depth=9
  #pragma HLS array_partition variable=v836 complete dim=1
  #pragma HLS array_partition variable=v836 complete dim=2

  int32_t v32;	// L1001
  v32 = 0;	// L1002
  l_reduction_k32: for (int k32 = 0; k32 < 128; k32++) {	// L1003
  #pragma HLS pipeline II=1
    int8_t v841 = v832.read(); // v832[k32];	// L1004
    int8_t a32;	// L1005
    a32 = v841;	// L1006
    int8_t v843 = v833.read(); // v833[k32];	// L1007
    int8_t b32;	// L1008
    b32 = v843;	// L1009
    int8_t v845 = a32;	// L1010
    int8_t v846 = b32;	// L1011
    int16_t v847 = v845;	// L1012
    int16_t v848 = v846;	// L1013
    int16_t v849 = v847 * v848;	// L1014
    int32_t v850 = v32;	// L1015
    ap_int<33> v851 = v850;	// L1016
    ap_int<33> v852 = v849;	// L1017
    ap_int<33> v853 = v851 + v852;	// L1018
    int32_t v854 = v853;	// L1019
    v32 = v854;	// L1020
    int8_t v855 = a32;	// L1021
    v834.write(v855); // v834[k32] = v855;	// L1022
    int8_t v856 = b32;	// L1023
    v835.write(v856); // v835[k32] = v856;	// L1024
  }
  int32_t v857 = v32;	// L1026
  v836[v837][v838] = v857;	// L1027
}

void PE_kernel_gemm_1_4(
  hls::stream< int8_t > &v858 /* v858[128] */,
  hls::stream< int8_t > &v859 /* v859[128] */,
  hls::stream< int8_t > &v860 /* v860[128] */,
  hls::stream< int8_t > &v861 /* v861[128] */,
  int32_t v862[8][8],
  int v863,
  int v864
) {	// L1030
  #pragma HLS stream variable=v858 depth=9
  #pragma HLS stream variable=v859 depth=9
  #pragma HLS stream variable=v860 depth=9
  #pragma HLS stream variable=v861 depth=9
  #pragma HLS array_partition variable=v862 complete dim=1
  #pragma HLS array_partition variable=v862 complete dim=2

  int32_t v33;	// L1032
  v33 = 0;	// L1033
  l_reduction_k33: for (int k33 = 0; k33 < 128; k33++) {	// L1034
  #pragma HLS pipeline II=1
    int8_t v867 = v858.read(); // v858[k33];	// L1035
    int8_t a33;	// L1036
    a33 = v867;	// L1037
    int8_t v869 = v859.read(); // v859[k33];	// L1038
    int8_t b33;	// L1039
    b33 = v869;	// L1040
    int8_t v871 = a33;	// L1041
    int8_t v872 = b33;	// L1042
    int16_t v873 = v871;	// L1043
    int16_t v874 = v872;	// L1044
    int16_t v875 = v873 * v874;	// L1045
    int32_t v876 = v33;	// L1046
    ap_int<33> v877 = v876;	// L1047
    ap_int<33> v878 = v875;	// L1048
    ap_int<33> v879 = v877 + v878;	// L1049
    int32_t v880 = v879;	// L1050
    v33 = v880;	// L1051
    int8_t v881 = a33;	// L1052
    v860.write(v881); // v860[k33] = v881;	// L1053
    int8_t v882 = b33;	// L1054
    v861.write(v882); // v861[k33] = v882;	// L1055
  }
  int32_t v883 = v33;	// L1057
  v862[v863][v864] = v883;	// L1058
}

void PE_kernel_gemm_2_4(
  hls::stream< int8_t > &v884 /* v884[128] */,
  hls::stream< int8_t > &v885 /* v885[128] */,
  hls::stream< int8_t > &v886 /* v886[128] */,
  hls::stream< int8_t > &v887 /* v887[128] */,
  int32_t v888[8][8],
  int v889,
  int v890
) {	// L1061
  #pragma HLS stream variable=v884 depth=9
  #pragma HLS stream variable=v885 depth=9
  #pragma HLS stream variable=v886 depth=9
  #pragma HLS stream variable=v887 depth=9
  #pragma HLS array_partition variable=v888 complete dim=1
  #pragma HLS array_partition variable=v888 complete dim=2

  int32_t v34;	// L1063
  v34 = 0;	// L1064
  l_reduction_k34: for (int k34 = 0; k34 < 128; k34++) {	// L1065
  #pragma HLS pipeline II=1
    int8_t v893 = v884.read(); // v884[k34];	// L1066
    int8_t a34;	// L1067
    a34 = v893;	// L1068
    int8_t v895 = v885.read(); // v885[k34];	// L1069
    int8_t b34;	// L1070
    b34 = v895;	// L1071
    int8_t v897 = a34;	// L1072
    int8_t v898 = b34;	// L1073
    int16_t v899 = v897;	// L1074
    int16_t v900 = v898;	// L1075
    int16_t v901 = v899 * v900;	// L1076
    int32_t v902 = v34;	// L1077
    ap_int<33> v903 = v902;	// L1078
    ap_int<33> v904 = v901;	// L1079
    ap_int<33> v905 = v903 + v904;	// L1080
    int32_t v906 = v905;	// L1081
    v34 = v906;	// L1082
    int8_t v907 = a34;	// L1083
    v886.write(v907); // v886[k34] = v907;	// L1084
    int8_t v908 = b34;	// L1085
    v887.write(v908); // v887[k34] = v908;	// L1086
  }
  int32_t v909 = v34;	// L1088
  v888[v889][v890] = v909;	// L1089
}

void PE_kernel_gemm_3_4(
  hls::stream< int8_t > &v910 /* v910[128] */,
  hls::stream< int8_t > &v911 /* v911[128] */,
  hls::stream< int8_t > &v912 /* v912[128] */,
  hls::stream< int8_t > &v913 /* v913[128] */,
  int32_t v914[8][8],
  int v915,
  int v916
) {	// L1092
  #pragma HLS stream variable=v910 depth=9
  #pragma HLS stream variable=v911 depth=9
  #pragma HLS stream variable=v912 depth=9
  #pragma HLS stream variable=v913 depth=9
  #pragma HLS array_partition variable=v914 complete dim=1
  #pragma HLS array_partition variable=v914 complete dim=2

  int32_t v35;	// L1094
  v35 = 0;	// L1095
  l_reduction_k35: for (int k35 = 0; k35 < 128; k35++) {	// L1096
  #pragma HLS pipeline II=1
    int8_t v919 = v910.read(); // v910[k35];	// L1097
    int8_t a35;	// L1098
    a35 = v919;	// L1099
    int8_t v921 = v911.read(); // v911[k35];	// L1100
    int8_t b35;	// L1101
    b35 = v921;	// L1102
    int8_t v923 = a35;	// L1103
    int8_t v924 = b35;	// L1104
    int16_t v925 = v923;	// L1105
    int16_t v926 = v924;	// L1106
    int16_t v927 = v925 * v926;	// L1107
    int32_t v928 = v35;	// L1108
    ap_int<33> v929 = v928;	// L1109
    ap_int<33> v930 = v927;	// L1110
    ap_int<33> v931 = v929 + v930;	// L1111
    int32_t v932 = v931;	// L1112
    v35 = v932;	// L1113
    int8_t v933 = a35;	// L1114
    v912.write(v933); // v912[k35] = v933;	// L1115
    int8_t v934 = b35;	// L1116
    v913.write(v934); // v913[k35] = v934;	// L1117
  }
  int32_t v935 = v35;	// L1119
  v914[v915][v916] = v935;	// L1120
}

void PE_kernel_gemm_4_4(
  hls::stream< int8_t > &v936 /* v936[128] */,
  hls::stream< int8_t > &v937 /* v937[128] */,
  hls::stream< int8_t > &v938 /* v938[128] */,
  hls::stream< int8_t > &v939 /* v939[128] */,
  int32_t v940[8][8],
  int v941,
  int v942
) {	// L1123
  #pragma HLS stream variable=v936 depth=9
  #pragma HLS stream variable=v937 depth=9
  #pragma HLS stream variable=v938 depth=9
  #pragma HLS stream variable=v939 depth=9
  #pragma HLS array_partition variable=v940 complete dim=1
  #pragma HLS array_partition variable=v940 complete dim=2

  int32_t v36;	// L1125
  v36 = 0;	// L1126
  l_reduction_k36: for (int k36 = 0; k36 < 128; k36++) {	// L1127
  #pragma HLS pipeline II=1
    int8_t v945 = v936.read(); // v936[k36];	// L1128
    int8_t a36;	// L1129
    a36 = v945;	// L1130
    int8_t v947 = v937.read(); // v937[k36];	// L1131
    int8_t b36;	// L1132
    b36 = v947;	// L1133
    int8_t v949 = a36;	// L1134
    int8_t v950 = b36;	// L1135
    int16_t v951 = v949;	// L1136
    int16_t v952 = v950;	// L1137
    int16_t v953 = v951 * v952;	// L1138
    int32_t v954 = v36;	// L1139
    ap_int<33> v955 = v954;	// L1140
    ap_int<33> v956 = v953;	// L1141
    ap_int<33> v957 = v955 + v956;	// L1142
    int32_t v958 = v957;	// L1143
    v36 = v958;	// L1144
    int8_t v959 = a36;	// L1145
    v938.write(v959); // v938[k36] = v959;	// L1146
    int8_t v960 = b36;	// L1147
    v939.write(v960); // v939[k36] = v960;	// L1148
  }
  int32_t v961 = v36;	// L1150
  v940[v941][v942] = v961;	// L1151
}

void PE_kernel_gemm_5_4(
  hls::stream< int8_t > &v962 /* v962[128] */,
  hls::stream< int8_t > &v963 /* v963[128] */,
  hls::stream< int8_t > &v964 /* v964[128] */,
  hls::stream< int8_t > &v965 /* v965[128] */,
  int32_t v966[8][8],
  int v967,
  int v968
) {	// L1154
  #pragma HLS stream variable=v962 depth=9
  #pragma HLS stream variable=v963 depth=9
  #pragma HLS stream variable=v964 depth=9
  #pragma HLS stream variable=v965 depth=9
  #pragma HLS array_partition variable=v966 complete dim=1
  #pragma HLS array_partition variable=v966 complete dim=2

  int32_t v37;	// L1156
  v37 = 0;	// L1157
  l_reduction_k37: for (int k37 = 0; k37 < 128; k37++) {	// L1158
  #pragma HLS pipeline II=1
    int8_t v971 = v962.read(); // v962[k37];	// L1159
    int8_t a37;	// L1160
    a37 = v971;	// L1161
    int8_t v973 = v963.read(); // v963[k37];	// L1162
    int8_t b37;	// L1163
    b37 = v973;	// L1164
    int8_t v975 = a37;	// L1165
    int8_t v976 = b37;	// L1166
    int16_t v977 = v975;	// L1167
    int16_t v978 = v976;	// L1168
    int16_t v979 = v977 * v978;	// L1169
    int32_t v980 = v37;	// L1170
    ap_int<33> v981 = v980;	// L1171
    ap_int<33> v982 = v979;	// L1172
    ap_int<33> v983 = v981 + v982;	// L1173
    int32_t v984 = v983;	// L1174
    v37 = v984;	// L1175
    int8_t v985 = a37;	// L1176
    v964.write(v985); // v964[k37] = v985;	// L1177
    int8_t v986 = b37;	// L1178
    v965.write(v986); // v965[k37] = v986;	// L1179
  }
  int32_t v987 = v37;	// L1181
  v966[v967][v968] = v987;	// L1182
}

void PE_kernel_gemm_6_4(
  hls::stream< int8_t > &v988 /* v988[128] */,
  hls::stream< int8_t > &v989 /* v989[128] */,
  hls::stream< int8_t > &v990 /* v990[128] */,
  hls::stream< int8_t > &v991 /* v991[128] */,
  int32_t v992[8][8],
  int v993,
  int v994
) {	// L1185
  #pragma HLS stream variable=v988 depth=9
  #pragma HLS stream variable=v989 depth=9
  #pragma HLS stream variable=v990 depth=9
  #pragma HLS stream variable=v991 depth=9
  #pragma HLS array_partition variable=v992 complete dim=1
  #pragma HLS array_partition variable=v992 complete dim=2

  int32_t v38;	// L1187
  v38 = 0;	// L1188
  l_reduction_k38: for (int k38 = 0; k38 < 128; k38++) {	// L1189
  #pragma HLS pipeline II=1
    int8_t v997 = v988.read(); // v988[k38];	// L1190
    int8_t a38;	// L1191
    a38 = v997;	// L1192
    int8_t v999 = v989.read(); // v989[k38];	// L1193
    int8_t b38;	// L1194
    b38 = v999;	// L1195
    int8_t v1001 = a38;	// L1196
    int8_t v1002 = b38;	// L1197
    int16_t v1003 = v1001;	// L1198
    int16_t v1004 = v1002;	// L1199
    int16_t v1005 = v1003 * v1004;	// L1200
    int32_t v1006 = v38;	// L1201
    ap_int<33> v1007 = v1006;	// L1202
    ap_int<33> v1008 = v1005;	// L1203
    ap_int<33> v1009 = v1007 + v1008;	// L1204
    int32_t v1010 = v1009;	// L1205
    v38 = v1010;	// L1206
    int8_t v1011 = a38;	// L1207
    v990.write(v1011); // v990[k38] = v1011;	// L1208
    int8_t v1012 = b38;	// L1209
    v991.write(v1012); // v991[k38] = v1012;	// L1210
  }
  int32_t v1013 = v38;	// L1212
  v992[v993][v994] = v1013;	// L1213
}

void PE_kernel_gemm_7_4(
  hls::stream< int8_t > &v1014 /* v1014[128] */,
  hls::stream< int8_t > &v1015 /* v1015[128] */,
  hls::stream< int8_t > &v1016 /* v1016[128] */,
  hls::stream< int8_t > &v1017 /* v1017[128] */,
  int32_t v1018[8][8],
  int v1019,
  int v1020
) {	// L1216
  #pragma HLS stream variable=v1014 depth=9
  #pragma HLS stream variable=v1015 depth=9
  #pragma HLS stream variable=v1016 depth=9
  #pragma HLS stream variable=v1017 depth=9
  #pragma HLS array_partition variable=v1018 complete dim=1
  #pragma HLS array_partition variable=v1018 complete dim=2

  int32_t v39;	// L1218
  v39 = 0;	// L1219
  l_reduction_k39: for (int k39 = 0; k39 < 128; k39++) {	// L1220
  #pragma HLS pipeline II=1
    int8_t v1023 = v1014.read(); // v1014[k39];	// L1221
    int8_t a39;	// L1222
    a39 = v1023;	// L1223
    int8_t v1025 = v1015.read(); // v1015[k39];	// L1224
    int8_t b39;	// L1225
    b39 = v1025;	// L1226
    int8_t v1027 = a39;	// L1227
    int8_t v1028 = b39;	// L1228
    int16_t v1029 = v1027;	// L1229
    int16_t v1030 = v1028;	// L1230
    int16_t v1031 = v1029 * v1030;	// L1231
    int32_t v1032 = v39;	// L1232
    ap_int<33> v1033 = v1032;	// L1233
    ap_int<33> v1034 = v1031;	// L1234
    ap_int<33> v1035 = v1033 + v1034;	// L1235
    int32_t v1036 = v1035;	// L1236
    v39 = v1036;	// L1237
    int8_t v1037 = a39;	// L1238
    v1016.write(v1037); // v1016[k39] = v1037;	// L1239
    int8_t v1038 = b39;	// L1240
    v1017.write(v1038); // v1017[k39] = v1038;	// L1241
  }
  int32_t v1039 = v39;	// L1243
  v1018[v1019][v1020] = v1039;	// L1244
}

void PE_kernel_gemm_0_5(
  hls::stream< int8_t > &v1040 /* v1040[128] */,
  hls::stream< int8_t > &v1041 /* v1041[128] */,
  hls::stream< int8_t > &v1042 /* v1042[128] */,
  hls::stream< int8_t > &v1043 /* v1043[128] */,
  int32_t v1044[8][8],
  int v1045,
  int v1046
) {	// L1247
  #pragma HLS stream variable=v1040 depth=9
  #pragma HLS stream variable=v1041 depth=9
  #pragma HLS stream variable=v1042 depth=9
  #pragma HLS stream variable=v1043 depth=9
  #pragma HLS array_partition variable=v1044 complete dim=1
  #pragma HLS array_partition variable=v1044 complete dim=2

  int32_t v40;	// L1249
  v40 = 0;	// L1250
  l_reduction_k40: for (int k40 = 0; k40 < 128; k40++) {	// L1251
  #pragma HLS pipeline II=1
    int8_t v1049 = v1040.read(); // v1040[k40];	// L1252
    int8_t a40;	// L1253
    a40 = v1049;	// L1254
    int8_t v1051 = v1041.read(); // v1041[k40];	// L1255
    int8_t b40;	// L1256
    b40 = v1051;	// L1257
    int8_t v1053 = a40;	// L1258
    int8_t v1054 = b40;	// L1259
    int16_t v1055 = v1053;	// L1260
    int16_t v1056 = v1054;	// L1261
    int16_t v1057 = v1055 * v1056;	// L1262
    int32_t v1058 = v40;	// L1263
    ap_int<33> v1059 = v1058;	// L1264
    ap_int<33> v1060 = v1057;	// L1265
    ap_int<33> v1061 = v1059 + v1060;	// L1266
    int32_t v1062 = v1061;	// L1267
    v40 = v1062;	// L1268
    int8_t v1063 = a40;	// L1269
    v1042.write(v1063); // v1042[k40] = v1063;	// L1270
    int8_t v1064 = b40;	// L1271
    v1043.write(v1064); // v1043[k40] = v1064;	// L1272
  }
  int32_t v1065 = v40;	// L1274
  v1044[v1045][v1046] = v1065;	// L1275
}

void PE_kernel_gemm_1_5(
  hls::stream< int8_t > &v1066 /* v1066[128] */,
  hls::stream< int8_t > &v1067 /* v1067[128] */,
  hls::stream< int8_t > &v1068 /* v1068[128] */,
  hls::stream< int8_t > &v1069 /* v1069[128] */,
  int32_t v1070[8][8],
  int v1071,
  int v1072
) {	// L1278
  #pragma HLS stream variable=v1066 depth=9
  #pragma HLS stream variable=v1067 depth=9
  #pragma HLS stream variable=v1068 depth=9
  #pragma HLS stream variable=v1069 depth=9
  #pragma HLS array_partition variable=v1070 complete dim=1
  #pragma HLS array_partition variable=v1070 complete dim=2

  int32_t v41;	// L1280
  v41 = 0;	// L1281
  l_reduction_k41: for (int k41 = 0; k41 < 128; k41++) {	// L1282
  #pragma HLS pipeline II=1
    int8_t v1075 = v1066.read(); // v1066[k41];	// L1283
    int8_t a41;	// L1284
    a41 = v1075;	// L1285
    int8_t v1077 = v1067.read(); // v1067[k41];	// L1286
    int8_t b41;	// L1287
    b41 = v1077;	// L1288
    int8_t v1079 = a41;	// L1289
    int8_t v1080 = b41;	// L1290
    int16_t v1081 = v1079;	// L1291
    int16_t v1082 = v1080;	// L1292
    int16_t v1083 = v1081 * v1082;	// L1293
    int32_t v1084 = v41;	// L1294
    ap_int<33> v1085 = v1084;	// L1295
    ap_int<33> v1086 = v1083;	// L1296
    ap_int<33> v1087 = v1085 + v1086;	// L1297
    int32_t v1088 = v1087;	// L1298
    v41 = v1088;	// L1299
    int8_t v1089 = a41;	// L1300
    v1068.write(v1089); // v1068[k41] = v1089;	// L1301
    int8_t v1090 = b41;	// L1302
    v1069.write(v1090); // v1069[k41] = v1090;	// L1303
  }
  int32_t v1091 = v41;	// L1305
  v1070[v1071][v1072] = v1091;	// L1306
}

void PE_kernel_gemm_2_5(
  hls::stream< int8_t > &v1092 /* v1092[128] */,
  hls::stream< int8_t > &v1093 /* v1093[128] */,
  hls::stream< int8_t > &v1094 /* v1094[128] */,
  hls::stream< int8_t > &v1095 /* v1095[128] */,
  int32_t v1096[8][8],
  int v1097,
  int v1098
) {	// L1309
  #pragma HLS stream variable=v1092 depth=9
  #pragma HLS stream variable=v1093 depth=9
  #pragma HLS stream variable=v1094 depth=9
  #pragma HLS stream variable=v1095 depth=9
  #pragma HLS array_partition variable=v1096 complete dim=1
  #pragma HLS array_partition variable=v1096 complete dim=2

  int32_t v42;	// L1311
  v42 = 0;	// L1312
  l_reduction_k42: for (int k42 = 0; k42 < 128; k42++) {	// L1313
  #pragma HLS pipeline II=1
    int8_t v1101 = v1092.read(); // v1092[k42];	// L1314
    int8_t a42;	// L1315
    a42 = v1101;	// L1316
    int8_t v1103 = v1093.read(); // v1093[k42];	// L1317
    int8_t b42;	// L1318
    b42 = v1103;	// L1319
    int8_t v1105 = a42;	// L1320
    int8_t v1106 = b42;	// L1321
    int16_t v1107 = v1105;	// L1322
    int16_t v1108 = v1106;	// L1323
    int16_t v1109 = v1107 * v1108;	// L1324
    int32_t v1110 = v42;	// L1325
    ap_int<33> v1111 = v1110;	// L1326
    ap_int<33> v1112 = v1109;	// L1327
    ap_int<33> v1113 = v1111 + v1112;	// L1328
    int32_t v1114 = v1113;	// L1329
    v42 = v1114;	// L1330
    int8_t v1115 = a42;	// L1331
    v1094.write(v1115); // v1094[k42] = v1115;	// L1332
    int8_t v1116 = b42;	// L1333
    v1095.write(v1116); // v1095[k42] = v1116;	// L1334
  }
  int32_t v1117 = v42;	// L1336
  v1096[v1097][v1098] = v1117;	// L1337
}

void PE_kernel_gemm_3_5(
  hls::stream< int8_t > &v1118 /* v1118[128] */,
  hls::stream< int8_t > &v1119 /* v1119[128] */,
  hls::stream< int8_t > &v1120 /* v1120[128] */,
  hls::stream< int8_t > &v1121 /* v1121[128] */,
  int32_t v1122[8][8],
  int v1123,
  int v1124
) {	// L1340
  #pragma HLS stream variable=v1118 depth=9
  #pragma HLS stream variable=v1119 depth=9
  #pragma HLS stream variable=v1120 depth=9
  #pragma HLS stream variable=v1121 depth=9
  #pragma HLS array_partition variable=v1122 complete dim=1
  #pragma HLS array_partition variable=v1122 complete dim=2

  int32_t v43;	// L1342
  v43 = 0;	// L1343
  l_reduction_k43: for (int k43 = 0; k43 < 128; k43++) {	// L1344
  #pragma HLS pipeline II=1
    int8_t v1127 = v1118.read(); // v1118[k43];	// L1345
    int8_t a43;	// L1346
    a43 = v1127;	// L1347
    int8_t v1129 = v1119.read(); // v1119[k43];	// L1348
    int8_t b43;	// L1349
    b43 = v1129;	// L1350
    int8_t v1131 = a43;	// L1351
    int8_t v1132 = b43;	// L1352
    int16_t v1133 = v1131;	// L1353
    int16_t v1134 = v1132;	// L1354
    int16_t v1135 = v1133 * v1134;	// L1355
    int32_t v1136 = v43;	// L1356
    ap_int<33> v1137 = v1136;	// L1357
    ap_int<33> v1138 = v1135;	// L1358
    ap_int<33> v1139 = v1137 + v1138;	// L1359
    int32_t v1140 = v1139;	// L1360
    v43 = v1140;	// L1361
    int8_t v1141 = a43;	// L1362
    v1120.write(v1141); // v1120[k43] = v1141;	// L1363
    int8_t v1142 = b43;	// L1364
    v1121.write(v1142); // v1121[k43] = v1142;	// L1365
  }
  int32_t v1143 = v43;	// L1367
  v1122[v1123][v1124] = v1143;	// L1368
}

void PE_kernel_gemm_4_5(
  hls::stream< int8_t > &v1144 /* v1144[128] */,
  hls::stream< int8_t > &v1145 /* v1145[128] */,
  hls::stream< int8_t > &v1146 /* v1146[128] */,
  hls::stream< int8_t > &v1147 /* v1147[128] */,
  int32_t v1148[8][8],
  int v1149,
  int v1150
) {	// L1371
  #pragma HLS stream variable=v1144 depth=9
  #pragma HLS stream variable=v1145 depth=9
  #pragma HLS stream variable=v1146 depth=9
  #pragma HLS stream variable=v1147 depth=9
  #pragma HLS array_partition variable=v1148 complete dim=1
  #pragma HLS array_partition variable=v1148 complete dim=2

  int32_t v44;	// L1373
  v44 = 0;	// L1374
  l_reduction_k44: for (int k44 = 0; k44 < 128; k44++) {	// L1375
  #pragma HLS pipeline II=1
    int8_t v1153 = v1144.read(); // v1144[k44];	// L1376
    int8_t a44;	// L1377
    a44 = v1153;	// L1378
    int8_t v1155 = v1145.read(); // v1145[k44];	// L1379
    int8_t b44;	// L1380
    b44 = v1155;	// L1381
    int8_t v1157 = a44;	// L1382
    int8_t v1158 = b44;	// L1383
    int16_t v1159 = v1157;	// L1384
    int16_t v1160 = v1158;	// L1385
    int16_t v1161 = v1159 * v1160;	// L1386
    int32_t v1162 = v44;	// L1387
    ap_int<33> v1163 = v1162;	// L1388
    ap_int<33> v1164 = v1161;	// L1389
    ap_int<33> v1165 = v1163 + v1164;	// L1390
    int32_t v1166 = v1165;	// L1391
    v44 = v1166;	// L1392
    int8_t v1167 = a44;	// L1393
    v1146.write(v1167); // v1146[k44] = v1167;	// L1394
    int8_t v1168 = b44;	// L1395
    v1147.write(v1168); // v1147[k44] = v1168;	// L1396
  }
  int32_t v1169 = v44;	// L1398
  v1148[v1149][v1150] = v1169;	// L1399
}

void PE_kernel_gemm_5_5(
  hls::stream< int8_t > &v1170 /* v1170[128] */,
  hls::stream< int8_t > &v1171 /* v1171[128] */,
  hls::stream< int8_t > &v1172 /* v1172[128] */,
  hls::stream< int8_t > &v1173 /* v1173[128] */,
  int32_t v1174[8][8],
  int v1175,
  int v1176
) {	// L1402
  #pragma HLS stream variable=v1170 depth=9
  #pragma HLS stream variable=v1171 depth=9
  #pragma HLS stream variable=v1172 depth=9
  #pragma HLS stream variable=v1173 depth=9
  #pragma HLS array_partition variable=v1174 complete dim=1
  #pragma HLS array_partition variable=v1174 complete dim=2

  int32_t v45;	// L1404
  v45 = 0;	// L1405
  l_reduction_k45: for (int k45 = 0; k45 < 128; k45++) {	// L1406
  #pragma HLS pipeline II=1
    int8_t v1179 = v1170.read(); // v1170[k45];	// L1407
    int8_t a45;	// L1408
    a45 = v1179;	// L1409
    int8_t v1181 = v1171.read(); // v1171[k45];	// L1410
    int8_t b45;	// L1411
    b45 = v1181;	// L1412
    int8_t v1183 = a45;	// L1413
    int8_t v1184 = b45;	// L1414
    int16_t v1185 = v1183;	// L1415
    int16_t v1186 = v1184;	// L1416
    int16_t v1187 = v1185 * v1186;	// L1417
    int32_t v1188 = v45;	// L1418
    ap_int<33> v1189 = v1188;	// L1419
    ap_int<33> v1190 = v1187;	// L1420
    ap_int<33> v1191 = v1189 + v1190;	// L1421
    int32_t v1192 = v1191;	// L1422
    v45 = v1192;	// L1423
    int8_t v1193 = a45;	// L1424
    v1172.write(v1193); // v1172[k45] = v1193;	// L1425
    int8_t v1194 = b45;	// L1426
    v1173.write(v1194); // v1173[k45] = v1194;	// L1427
  }
  int32_t v1195 = v45;	// L1429
  v1174[v1175][v1176] = v1195;	// L1430
}

void PE_kernel_gemm_6_5(
  hls::stream< int8_t > &v1196 /* v1196[128] */,
  hls::stream< int8_t > &v1197 /* v1197[128] */,
  hls::stream< int8_t > &v1198 /* v1198[128] */,
  hls::stream< int8_t > &v1199 /* v1199[128] */,
  int32_t v1200[8][8],
  int v1201,
  int v1202
) {	// L1433
  #pragma HLS stream variable=v1196 depth=9
  #pragma HLS stream variable=v1197 depth=9
  #pragma HLS stream variable=v1198 depth=9
  #pragma HLS stream variable=v1199 depth=9
  #pragma HLS array_partition variable=v1200 complete dim=1
  #pragma HLS array_partition variable=v1200 complete dim=2

  int32_t v46;	// L1435
  v46 = 0;	// L1436
  l_reduction_k46: for (int k46 = 0; k46 < 128; k46++) {	// L1437
  #pragma HLS pipeline II=1
    int8_t v1205 = v1196.read(); // v1196[k46];	// L1438
    int8_t a46;	// L1439
    a46 = v1205;	// L1440
    int8_t v1207 = v1197.read(); // v1197[k46];	// L1441
    int8_t b46;	// L1442
    b46 = v1207;	// L1443
    int8_t v1209 = a46;	// L1444
    int8_t v1210 = b46;	// L1445
    int16_t v1211 = v1209;	// L1446
    int16_t v1212 = v1210;	// L1447
    int16_t v1213 = v1211 * v1212;	// L1448
    int32_t v1214 = v46;	// L1449
    ap_int<33> v1215 = v1214;	// L1450
    ap_int<33> v1216 = v1213;	// L1451
    ap_int<33> v1217 = v1215 + v1216;	// L1452
    int32_t v1218 = v1217;	// L1453
    v46 = v1218;	// L1454
    int8_t v1219 = a46;	// L1455
    v1198.write(v1219); // v1198[k46] = v1219;	// L1456
    int8_t v1220 = b46;	// L1457
    v1199.write(v1220); // v1199[k46] = v1220;	// L1458
  }
  int32_t v1221 = v46;	// L1460
  v1200[v1201][v1202] = v1221;	// L1461
}

void PE_kernel_gemm_7_5(
  hls::stream< int8_t > &v1222 /* v1222[128] */,
  hls::stream< int8_t > &v1223 /* v1223[128] */,
  hls::stream< int8_t > &v1224 /* v1224[128] */,
  hls::stream< int8_t > &v1225 /* v1225[128] */,
  int32_t v1226[8][8],
  int v1227,
  int v1228
) {	// L1464
  #pragma HLS stream variable=v1222 depth=9
  #pragma HLS stream variable=v1223 depth=9
  #pragma HLS stream variable=v1224 depth=9
  #pragma HLS stream variable=v1225 depth=9
  #pragma HLS array_partition variable=v1226 complete dim=1
  #pragma HLS array_partition variable=v1226 complete dim=2

  int32_t v47;	// L1466
  v47 = 0;	// L1467
  l_reduction_k47: for (int k47 = 0; k47 < 128; k47++) {	// L1468
  #pragma HLS pipeline II=1
    int8_t v1231 = v1222.read(); // v1222[k47];	// L1469
    int8_t a47;	// L1470
    a47 = v1231;	// L1471
    int8_t v1233 = v1223.read(); // v1223[k47];	// L1472
    int8_t b47;	// L1473
    b47 = v1233;	// L1474
    int8_t v1235 = a47;	// L1475
    int8_t v1236 = b47;	// L1476
    int16_t v1237 = v1235;	// L1477
    int16_t v1238 = v1236;	// L1478
    int16_t v1239 = v1237 * v1238;	// L1479
    int32_t v1240 = v47;	// L1480
    ap_int<33> v1241 = v1240;	// L1481
    ap_int<33> v1242 = v1239;	// L1482
    ap_int<33> v1243 = v1241 + v1242;	// L1483
    int32_t v1244 = v1243;	// L1484
    v47 = v1244;	// L1485
    int8_t v1245 = a47;	// L1486
    v1224.write(v1245); // v1224[k47] = v1245;	// L1487
    int8_t v1246 = b47;	// L1488
    v1225.write(v1246); // v1225[k47] = v1246;	// L1489
  }
  int32_t v1247 = v47;	// L1491
  v1226[v1227][v1228] = v1247;	// L1492
}

void PE_kernel_gemm_0_6(
  hls::stream< int8_t > &v1248 /* v1248[128] */,
  hls::stream< int8_t > &v1249 /* v1249[128] */,
  hls::stream< int8_t > &v1250 /* v1250[128] */,
  hls::stream< int8_t > &v1251 /* v1251[128] */,
  int32_t v1252[8][8],
  int v1253,
  int v1254
) {	// L1495
  #pragma HLS stream variable=v1248 depth=9
  #pragma HLS stream variable=v1249 depth=9
  #pragma HLS stream variable=v1250 depth=9
  #pragma HLS stream variable=v1251 depth=9
  #pragma HLS array_partition variable=v1252 complete dim=1
  #pragma HLS array_partition variable=v1252 complete dim=2

  int32_t v48;	// L1497
  v48 = 0;	// L1498
  l_reduction_k48: for (int k48 = 0; k48 < 128; k48++) {	// L1499
  #pragma HLS pipeline II=1
    int8_t v1257 = v1248.read(); // v1248[k48];	// L1500
    int8_t a48;	// L1501
    a48 = v1257;	// L1502
    int8_t v1259 = v1249.read(); // v1249[k48];	// L1503
    int8_t b48;	// L1504
    b48 = v1259;	// L1505
    int8_t v1261 = a48;	// L1506
    int8_t v1262 = b48;	// L1507
    int16_t v1263 = v1261;	// L1508
    int16_t v1264 = v1262;	// L1509
    int16_t v1265 = v1263 * v1264;	// L1510
    int32_t v1266 = v48;	// L1511
    ap_int<33> v1267 = v1266;	// L1512
    ap_int<33> v1268 = v1265;	// L1513
    ap_int<33> v1269 = v1267 + v1268;	// L1514
    int32_t v1270 = v1269;	// L1515
    v48 = v1270;	// L1516
    int8_t v1271 = a48;	// L1517
    v1250.write(v1271); // v1250[k48] = v1271;	// L1518
    int8_t v1272 = b48;	// L1519
    v1251.write(v1272); // v1251[k48] = v1272;	// L1520
  }
  int32_t v1273 = v48;	// L1522
  v1252[v1253][v1254] = v1273;	// L1523
}

void PE_kernel_gemm_1_6(
  hls::stream< int8_t > &v1274 /* v1274[128] */,
  hls::stream< int8_t > &v1275 /* v1275[128] */,
  hls::stream< int8_t > &v1276 /* v1276[128] */,
  hls::stream< int8_t > &v1277 /* v1277[128] */,
  int32_t v1278[8][8],
  int v1279,
  int v1280
) {	// L1526
  #pragma HLS stream variable=v1274 depth=9
  #pragma HLS stream variable=v1275 depth=9
  #pragma HLS stream variable=v1276 depth=9
  #pragma HLS stream variable=v1277 depth=9
  #pragma HLS array_partition variable=v1278 complete dim=1
  #pragma HLS array_partition variable=v1278 complete dim=2

  int32_t v49;	// L1528
  v49 = 0;	// L1529
  l_reduction_k49: for (int k49 = 0; k49 < 128; k49++) {	// L1530
  #pragma HLS pipeline II=1
    int8_t v1283 = v1274.read(); // v1274[k49];	// L1531
    int8_t a49;	// L1532
    a49 = v1283;	// L1533
    int8_t v1285 = v1275.read(); // v1275[k49];	// L1534
    int8_t b49;	// L1535
    b49 = v1285;	// L1536
    int8_t v1287 = a49;	// L1537
    int8_t v1288 = b49;	// L1538
    int16_t v1289 = v1287;	// L1539
    int16_t v1290 = v1288;	// L1540
    int16_t v1291 = v1289 * v1290;	// L1541
    int32_t v1292 = v49;	// L1542
    ap_int<33> v1293 = v1292;	// L1543
    ap_int<33> v1294 = v1291;	// L1544
    ap_int<33> v1295 = v1293 + v1294;	// L1545
    int32_t v1296 = v1295;	// L1546
    v49 = v1296;	// L1547
    int8_t v1297 = a49;	// L1548
    v1276.write(v1297); // v1276[k49] = v1297;	// L1549
    int8_t v1298 = b49;	// L1550
    v1277.write(v1298); // v1277[k49] = v1298;	// L1551
  }
  int32_t v1299 = v49;	// L1553
  v1278[v1279][v1280] = v1299;	// L1554
}

void PE_kernel_gemm_2_6(
  hls::stream< int8_t > &v1300 /* v1300[128] */,
  hls::stream< int8_t > &v1301 /* v1301[128] */,
  hls::stream< int8_t > &v1302 /* v1302[128] */,
  hls::stream< int8_t > &v1303 /* v1303[128] */,
  int32_t v1304[8][8],
  int v1305,
  int v1306
) {	// L1557
  #pragma HLS stream variable=v1300 depth=9
  #pragma HLS stream variable=v1301 depth=9
  #pragma HLS stream variable=v1302 depth=9
  #pragma HLS stream variable=v1303 depth=9
  #pragma HLS array_partition variable=v1304 complete dim=1
  #pragma HLS array_partition variable=v1304 complete dim=2

  int32_t v50;	// L1559
  v50 = 0;	// L1560
  l_reduction_k50: for (int k50 = 0; k50 < 128; k50++) {	// L1561
  #pragma HLS pipeline II=1
    int8_t v1309 = v1300.read(); // v1300[k50];	// L1562
    int8_t a50;	// L1563
    a50 = v1309;	// L1564
    int8_t v1311 = v1301.read(); // v1301[k50];	// L1565
    int8_t b50;	// L1566
    b50 = v1311;	// L1567
    int8_t v1313 = a50;	// L1568
    int8_t v1314 = b50;	// L1569
    int16_t v1315 = v1313;	// L1570
    int16_t v1316 = v1314;	// L1571
    int16_t v1317 = v1315 * v1316;	// L1572
    int32_t v1318 = v50;	// L1573
    ap_int<33> v1319 = v1318;	// L1574
    ap_int<33> v1320 = v1317;	// L1575
    ap_int<33> v1321 = v1319 + v1320;	// L1576
    int32_t v1322 = v1321;	// L1577
    v50 = v1322;	// L1578
    int8_t v1323 = a50;	// L1579
    v1302.write(v1323); // v1302[k50] = v1323;	// L1580
    int8_t v1324 = b50;	// L1581
    v1303.write(v1324); // v1303[k50] = v1324;	// L1582
  }
  int32_t v1325 = v50;	// L1584
  v1304[v1305][v1306] = v1325;	// L1585
}

void PE_kernel_gemm_3_6(
  hls::stream< int8_t > &v1326 /* v1326[128] */,
  hls::stream< int8_t > &v1327 /* v1327[128] */,
  hls::stream< int8_t > &v1328 /* v1328[128] */,
  hls::stream< int8_t > &v1329 /* v1329[128] */,
  int32_t v1330[8][8],
  int v1331,
  int v1332
) {	// L1588
  #pragma HLS stream variable=v1326 depth=9
  #pragma HLS stream variable=v1327 depth=9
  #pragma HLS stream variable=v1328 depth=9
  #pragma HLS stream variable=v1329 depth=9
  #pragma HLS array_partition variable=v1330 complete dim=1
  #pragma HLS array_partition variable=v1330 complete dim=2

  int32_t v51;	// L1590
  v51 = 0;	// L1591
  l_reduction_k51: for (int k51 = 0; k51 < 128; k51++) {	// L1592
  #pragma HLS pipeline II=1
    int8_t v1335 = v1326.read(); // v1326[k51];	// L1593
    int8_t a51;	// L1594
    a51 = v1335;	// L1595
    int8_t v1337 = v1327.read(); // v1327[k51];	// L1596
    int8_t b51;	// L1597
    b51 = v1337;	// L1598
    int8_t v1339 = a51;	// L1599
    int8_t v1340 = b51;	// L1600
    int16_t v1341 = v1339;	// L1601
    int16_t v1342 = v1340;	// L1602
    int16_t v1343 = v1341 * v1342;	// L1603
    int32_t v1344 = v51;	// L1604
    ap_int<33> v1345 = v1344;	// L1605
    ap_int<33> v1346 = v1343;	// L1606
    ap_int<33> v1347 = v1345 + v1346;	// L1607
    int32_t v1348 = v1347;	// L1608
    v51 = v1348;	// L1609
    int8_t v1349 = a51;	// L1610
    v1328.write(v1349); // v1328[k51] = v1349;	// L1611
    int8_t v1350 = b51;	// L1612
    v1329.write(v1350); // v1329[k51] = v1350;	// L1613
  }
  int32_t v1351 = v51;	// L1615
  v1330[v1331][v1332] = v1351;	// L1616
}

void PE_kernel_gemm_4_6(
  hls::stream< int8_t > &v1352 /* v1352[128] */,
  hls::stream< int8_t > &v1353 /* v1353[128] */,
  hls::stream< int8_t > &v1354 /* v1354[128] */,
  hls::stream< int8_t > &v1355 /* v1355[128] */,
  int32_t v1356[8][8],
  int v1357,
  int v1358
) {	// L1619
  #pragma HLS stream variable=v1352 depth=9
  #pragma HLS stream variable=v1353 depth=9
  #pragma HLS stream variable=v1354 depth=9
  #pragma HLS stream variable=v1355 depth=9
  #pragma HLS array_partition variable=v1356 complete dim=1
  #pragma HLS array_partition variable=v1356 complete dim=2

  int32_t v52;	// L1621
  v52 = 0;	// L1622
  l_reduction_k52: for (int k52 = 0; k52 < 128; k52++) {	// L1623
  #pragma HLS pipeline II=1
    int8_t v1361 = v1352.read(); // v1352[k52];	// L1624
    int8_t a52;	// L1625
    a52 = v1361;	// L1626
    int8_t v1363 = v1353.read(); // v1353[k52];	// L1627
    int8_t b52;	// L1628
    b52 = v1363;	// L1629
    int8_t v1365 = a52;	// L1630
    int8_t v1366 = b52;	// L1631
    int16_t v1367 = v1365;	// L1632
    int16_t v1368 = v1366;	// L1633
    int16_t v1369 = v1367 * v1368;	// L1634
    int32_t v1370 = v52;	// L1635
    ap_int<33> v1371 = v1370;	// L1636
    ap_int<33> v1372 = v1369;	// L1637
    ap_int<33> v1373 = v1371 + v1372;	// L1638
    int32_t v1374 = v1373;	// L1639
    v52 = v1374;	// L1640
    int8_t v1375 = a52;	// L1641
    v1354.write(v1375); // v1354[k52] = v1375;	// L1642
    int8_t v1376 = b52;	// L1643
    v1355.write(v1376); // v1355[k52] = v1376;	// L1644
  }
  int32_t v1377 = v52;	// L1646
  v1356[v1357][v1358] = v1377;	// L1647
}

void PE_kernel_gemm_5_6(
  hls::stream< int8_t > &v1378 /* v1378[128] */,
  hls::stream< int8_t > &v1379 /* v1379[128] */,
  hls::stream< int8_t > &v1380 /* v1380[128] */,
  hls::stream< int8_t > &v1381 /* v1381[128] */,
  int32_t v1382[8][8],
  int v1383,
  int v1384
) {	// L1650
  #pragma HLS stream variable=v1378 depth=9
  #pragma HLS stream variable=v1379 depth=9
  #pragma HLS stream variable=v1380 depth=9
  #pragma HLS stream variable=v1381 depth=9
  #pragma HLS array_partition variable=v1382 complete dim=1
  #pragma HLS array_partition variable=v1382 complete dim=2

  int32_t v53;	// L1652
  v53 = 0;	// L1653
  l_reduction_k53: for (int k53 = 0; k53 < 128; k53++) {	// L1654
  #pragma HLS pipeline II=1
    int8_t v1387 = v1378.read(); // v1378[k53];	// L1655
    int8_t a53;	// L1656
    a53 = v1387;	// L1657
    int8_t v1389 = v1379.read(); // v1379[k53];	// L1658
    int8_t b53;	// L1659
    b53 = v1389;	// L1660
    int8_t v1391 = a53;	// L1661
    int8_t v1392 = b53;	// L1662
    int16_t v1393 = v1391;	// L1663
    int16_t v1394 = v1392;	// L1664
    int16_t v1395 = v1393 * v1394;	// L1665
    int32_t v1396 = v53;	// L1666
    ap_int<33> v1397 = v1396;	// L1667
    ap_int<33> v1398 = v1395;	// L1668
    ap_int<33> v1399 = v1397 + v1398;	// L1669
    int32_t v1400 = v1399;	// L1670
    v53 = v1400;	// L1671
    int8_t v1401 = a53;	// L1672
    v1380.write(v1401); // v1380[k53] = v1401;	// L1673
    int8_t v1402 = b53;	// L1674
    v1381.write(v1402); // v1381[k53] = v1402;	// L1675
  }
  int32_t v1403 = v53;	// L1677
  v1382[v1383][v1384] = v1403;	// L1678
}

void PE_kernel_gemm_6_6(
  hls::stream< int8_t > &v1404 /* v1404[128] */,
  hls::stream< int8_t > &v1405 /* v1405[128] */,
  hls::stream< int8_t > &v1406 /* v1406[128] */,
  hls::stream< int8_t > &v1407 /* v1407[128] */,
  int32_t v1408[8][8],
  int v1409,
  int v1410
) {	// L1681
  #pragma HLS stream variable=v1404 depth=9
  #pragma HLS stream variable=v1405 depth=9
  #pragma HLS stream variable=v1406 depth=9
  #pragma HLS stream variable=v1407 depth=9
  #pragma HLS array_partition variable=v1408 complete dim=1
  #pragma HLS array_partition variable=v1408 complete dim=2

  int32_t v54;	// L1683
  v54 = 0;	// L1684
  l_reduction_k54: for (int k54 = 0; k54 < 128; k54++) {	// L1685
  #pragma HLS pipeline II=1
    int8_t v1413 = v1404.read(); // v1404[k54];	// L1686
    int8_t a54;	// L1687
    a54 = v1413;	// L1688
    int8_t v1415 = v1405.read(); // v1405[k54];	// L1689
    int8_t b54;	// L1690
    b54 = v1415;	// L1691
    int8_t v1417 = a54;	// L1692
    int8_t v1418 = b54;	// L1693
    int16_t v1419 = v1417;	// L1694
    int16_t v1420 = v1418;	// L1695
    int16_t v1421 = v1419 * v1420;	// L1696
    int32_t v1422 = v54;	// L1697
    ap_int<33> v1423 = v1422;	// L1698
    ap_int<33> v1424 = v1421;	// L1699
    ap_int<33> v1425 = v1423 + v1424;	// L1700
    int32_t v1426 = v1425;	// L1701
    v54 = v1426;	// L1702
    int8_t v1427 = a54;	// L1703
    v1406.write(v1427); // v1406[k54] = v1427;	// L1704
    int8_t v1428 = b54;	// L1705
    v1407.write(v1428); // v1407[k54] = v1428;	// L1706
  }
  int32_t v1429 = v54;	// L1708
  v1408[v1409][v1410] = v1429;	// L1709
}

void PE_kernel_gemm_7_6(
  hls::stream< int8_t > &v1430 /* v1430[128] */,
  hls::stream< int8_t > &v1431 /* v1431[128] */,
  hls::stream< int8_t > &v1432 /* v1432[128] */,
  hls::stream< int8_t > &v1433 /* v1433[128] */,
  int32_t v1434[8][8],
  int v1435,
  int v1436
) {	// L1712
  #pragma HLS stream variable=v1430 depth=9
  #pragma HLS stream variable=v1431 depth=9
  #pragma HLS stream variable=v1432 depth=9
  #pragma HLS stream variable=v1433 depth=9
  #pragma HLS array_partition variable=v1434 complete dim=1
  #pragma HLS array_partition variable=v1434 complete dim=2

  int32_t v55;	// L1714
  v55 = 0;	// L1715
  l_reduction_k55: for (int k55 = 0; k55 < 128; k55++) {	// L1716
  #pragma HLS pipeline II=1
    int8_t v1439 = v1430.read(); // v1430[k55];	// L1717
    int8_t a55;	// L1718
    a55 = v1439;	// L1719
    int8_t v1441 = v1431.read(); // v1431[k55];	// L1720
    int8_t b55;	// L1721
    b55 = v1441;	// L1722
    int8_t v1443 = a55;	// L1723
    int8_t v1444 = b55;	// L1724
    int16_t v1445 = v1443;	// L1725
    int16_t v1446 = v1444;	// L1726
    int16_t v1447 = v1445 * v1446;	// L1727
    int32_t v1448 = v55;	// L1728
    ap_int<33> v1449 = v1448;	// L1729
    ap_int<33> v1450 = v1447;	// L1730
    ap_int<33> v1451 = v1449 + v1450;	// L1731
    int32_t v1452 = v1451;	// L1732
    v55 = v1452;	// L1733
    int8_t v1453 = a55;	// L1734
    v1432.write(v1453); // v1432[k55] = v1453;	// L1735
    int8_t v1454 = b55;	// L1736
    v1433.write(v1454); // v1433[k55] = v1454;	// L1737
  }
  int32_t v1455 = v55;	// L1739
  v1434[v1435][v1436] = v1455;	// L1740
}

void PE_kernel_gemm_0_7(
  hls::stream< int8_t > &v1456 /* v1456[128] */,
  hls::stream< int8_t > &v1457 /* v1457[128] */,
  hls::stream< int8_t > &v1458 /* v1458[128] */,
  hls::stream< int8_t > &v1459 /* v1459[128] */,
  int32_t v1460[8][8],
  int v1461,
  int v1462
) {	// L1743
  #pragma HLS stream variable=v1456 depth=9
  #pragma HLS stream variable=v1457 depth=9
  #pragma HLS stream variable=v1458 depth=9
  #pragma HLS stream variable=v1459 depth=9
  #pragma HLS array_partition variable=v1460 complete dim=1
  #pragma HLS array_partition variable=v1460 complete dim=2

  int32_t v56;	// L1745
  v56 = 0;	// L1746
  l_reduction_k56: for (int k56 = 0; k56 < 128; k56++) {	// L1747
  #pragma HLS pipeline II=1
    int8_t v1465 = v1456.read(); // v1456[k56];	// L1748
    int8_t a56;	// L1749
    a56 = v1465;	// L1750
    int8_t v1467 = v1457.read(); // v1457[k56];	// L1751
    int8_t b56;	// L1752
    b56 = v1467;	// L1753
    int8_t v1469 = a56;	// L1754
    int8_t v1470 = b56;	// L1755
    int16_t v1471 = v1469;	// L1756
    int16_t v1472 = v1470;	// L1757
    int16_t v1473 = v1471 * v1472;	// L1758
    int32_t v1474 = v56;	// L1759
    ap_int<33> v1475 = v1474;	// L1760
    ap_int<33> v1476 = v1473;	// L1761
    ap_int<33> v1477 = v1475 + v1476;	// L1762
    int32_t v1478 = v1477;	// L1763
    v56 = v1478;	// L1764
    int8_t v1479 = a56;	// L1765
    v1458.write(v1479); // v1458[k56] = v1479;	// L1766
    int8_t v1480 = b56;	// L1767
    v1459.write(v1480); // v1459[k56] = v1480;	// L1768
  }
  int32_t v1481 = v56;	// L1770
  v1460[v1461][v1462] = v1481;	// L1771
}

void PE_kernel_gemm_1_7(
  hls::stream< int8_t > &v1482 /* v1482[128] */,
  hls::stream< int8_t > &v1483 /* v1483[128] */,
  hls::stream< int8_t > &v1484 /* v1484[128] */,
  hls::stream< int8_t > &v1485 /* v1485[128] */,
  int32_t v1486[8][8],
  int v1487,
  int v1488
) {	// L1774
  #pragma HLS stream variable=v1482 depth=9
  #pragma HLS stream variable=v1483 depth=9
  #pragma HLS stream variable=v1484 depth=9
  #pragma HLS stream variable=v1485 depth=9
  #pragma HLS array_partition variable=v1486 complete dim=1
  #pragma HLS array_partition variable=v1486 complete dim=2

  int32_t v57;	// L1776
  v57 = 0;	// L1777
  l_reduction_k57: for (int k57 = 0; k57 < 128; k57++) {	// L1778
  #pragma HLS pipeline II=1
    int8_t v1491 = v1482.read(); // v1482[k57];	// L1779
    int8_t a57;	// L1780
    a57 = v1491;	// L1781
    int8_t v1493 = v1483.read(); // v1483[k57];	// L1782
    int8_t b57;	// L1783
    b57 = v1493;	// L1784
    int8_t v1495 = a57;	// L1785
    int8_t v1496 = b57;	// L1786
    int16_t v1497 = v1495;	// L1787
    int16_t v1498 = v1496;	// L1788
    int16_t v1499 = v1497 * v1498;	// L1789
    int32_t v1500 = v57;	// L1790
    ap_int<33> v1501 = v1500;	// L1791
    ap_int<33> v1502 = v1499;	// L1792
    ap_int<33> v1503 = v1501 + v1502;	// L1793
    int32_t v1504 = v1503;	// L1794
    v57 = v1504;	// L1795
    int8_t v1505 = a57;	// L1796
    v1484.write(v1505); // v1484[k57] = v1505;	// L1797
    int8_t v1506 = b57;	// L1798
    v1485.write(v1506); // v1485[k57] = v1506;	// L1799
  }
  int32_t v1507 = v57;	// L1801
  v1486[v1487][v1488] = v1507;	// L1802
}

void PE_kernel_gemm_2_7(
  hls::stream< int8_t > &v1508 /* v1508[128] */,
  hls::stream< int8_t > &v1509 /* v1509[128] */,
  hls::stream< int8_t > &v1510 /* v1510[128] */,
  hls::stream< int8_t > &v1511 /* v1511[128] */,
  int32_t v1512[8][8],
  int v1513,
  int v1514
) {	// L1805
  #pragma HLS stream variable=v1508 depth=9
  #pragma HLS stream variable=v1509 depth=9
  #pragma HLS stream variable=v1510 depth=9
  #pragma HLS stream variable=v1511 depth=9
  #pragma HLS array_partition variable=v1512 complete dim=1
  #pragma HLS array_partition variable=v1512 complete dim=2

  int32_t v58;	// L1807
  v58 = 0;	// L1808
  l_reduction_k58: for (int k58 = 0; k58 < 128; k58++) {	// L1809
  #pragma HLS pipeline II=1
    int8_t v1517 = v1508.read(); // v1508[k58];	// L1810
    int8_t a58;	// L1811
    a58 = v1517;	// L1812
    int8_t v1519 = v1509.read(); // v1509[k58];	// L1813
    int8_t b58;	// L1814
    b58 = v1519;	// L1815
    int8_t v1521 = a58;	// L1816
    int8_t v1522 = b58;	// L1817
    int16_t v1523 = v1521;	// L1818
    int16_t v1524 = v1522;	// L1819
    int16_t v1525 = v1523 * v1524;	// L1820
    int32_t v1526 = v58;	// L1821
    ap_int<33> v1527 = v1526;	// L1822
    ap_int<33> v1528 = v1525;	// L1823
    ap_int<33> v1529 = v1527 + v1528;	// L1824
    int32_t v1530 = v1529;	// L1825
    v58 = v1530;	// L1826
    int8_t v1531 = a58;	// L1827
    v1510.write(v1531); // v1510[k58] = v1531;	// L1828
    int8_t v1532 = b58;	// L1829
    v1511.write(v1532); // v1511[k58] = v1532;	// L1830
  }
  int32_t v1533 = v58;	// L1832
  v1512[v1513][v1514] = v1533;	// L1833
}

void PE_kernel_gemm_3_7(
  hls::stream< int8_t > &v1534 /* v1534[128] */,
  hls::stream< int8_t > &v1535 /* v1535[128] */,
  hls::stream< int8_t > &v1536 /* v1536[128] */,
  hls::stream< int8_t > &v1537 /* v1537[128] */,
  int32_t v1538[8][8],
  int v1539,
  int v1540
) {	// L1836
  #pragma HLS stream variable=v1534 depth=9
  #pragma HLS stream variable=v1535 depth=9
  #pragma HLS stream variable=v1536 depth=9
  #pragma HLS stream variable=v1537 depth=9
  #pragma HLS array_partition variable=v1538 complete dim=1
  #pragma HLS array_partition variable=v1538 complete dim=2

  int32_t v59;	// L1838
  v59 = 0;	// L1839
  l_reduction_k59: for (int k59 = 0; k59 < 128; k59++) {	// L1840
  #pragma HLS pipeline II=1
    int8_t v1543 = v1534.read(); // v1534[k59];	// L1841
    int8_t a59;	// L1842
    a59 = v1543;	// L1843
    int8_t v1545 = v1535.read(); // v1535[k59];	// L1844
    int8_t b59;	// L1845
    b59 = v1545;	// L1846
    int8_t v1547 = a59;	// L1847
    int8_t v1548 = b59;	// L1848
    int16_t v1549 = v1547;	// L1849
    int16_t v1550 = v1548;	// L1850
    int16_t v1551 = v1549 * v1550;	// L1851
    int32_t v1552 = v59;	// L1852
    ap_int<33> v1553 = v1552;	// L1853
    ap_int<33> v1554 = v1551;	// L1854
    ap_int<33> v1555 = v1553 + v1554;	// L1855
    int32_t v1556 = v1555;	// L1856
    v59 = v1556;	// L1857
    int8_t v1557 = a59;	// L1858
    v1536.write(v1557); // v1536[k59] = v1557;	// L1859
    int8_t v1558 = b59;	// L1860
    v1537.write(v1558); // v1537[k59] = v1558;	// L1861
  }
  int32_t v1559 = v59;	// L1863
  v1538[v1539][v1540] = v1559;	// L1864
}

void PE_kernel_gemm_4_7(
  hls::stream< int8_t > &v1560 /* v1560[128] */,
  hls::stream< int8_t > &v1561 /* v1561[128] */,
  hls::stream< int8_t > &v1562 /* v1562[128] */,
  hls::stream< int8_t > &v1563 /* v1563[128] */,
  int32_t v1564[8][8],
  int v1565,
  int v1566
) {	// L1867
  #pragma HLS stream variable=v1560 depth=9
  #pragma HLS stream variable=v1561 depth=9
  #pragma HLS stream variable=v1562 depth=9
  #pragma HLS stream variable=v1563 depth=9
  #pragma HLS array_partition variable=v1564 complete dim=1
  #pragma HLS array_partition variable=v1564 complete dim=2

  int32_t v60;	// L1869
  v60 = 0;	// L1870
  l_reduction_k60: for (int k60 = 0; k60 < 128; k60++) {	// L1871
  #pragma HLS pipeline II=1
    int8_t v1569 = v1560.read(); // v1560[k60];	// L1872
    int8_t a60;	// L1873
    a60 = v1569;	// L1874
    int8_t v1571 = v1561.read(); // v1561[k60];	// L1875
    int8_t b60;	// L1876
    b60 = v1571;	// L1877
    int8_t v1573 = a60;	// L1878
    int8_t v1574 = b60;	// L1879
    int16_t v1575 = v1573;	// L1880
    int16_t v1576 = v1574;	// L1881
    int16_t v1577 = v1575 * v1576;	// L1882
    int32_t v1578 = v60;	// L1883
    ap_int<33> v1579 = v1578;	// L1884
    ap_int<33> v1580 = v1577;	// L1885
    ap_int<33> v1581 = v1579 + v1580;	// L1886
    int32_t v1582 = v1581;	// L1887
    v60 = v1582;	// L1888
    int8_t v1583 = a60;	// L1889
    v1562.write(v1583); // v1562[k60] = v1583;	// L1890
    int8_t v1584 = b60;	// L1891
    v1563.write(v1584); // v1563[k60] = v1584;	// L1892
  }
  int32_t v1585 = v60;	// L1894
  v1564[v1565][v1566] = v1585;	// L1895
}

void PE_kernel_gemm_5_7(
  hls::stream< int8_t > &v1586 /* v1586[128] */,
  hls::stream< int8_t > &v1587 /* v1587[128] */,
  hls::stream< int8_t > &v1588 /* v1588[128] */,
  hls::stream< int8_t > &v1589 /* v1589[128] */,
  int32_t v1590[8][8],
  int v1591,
  int v1592
) {	// L1898
  #pragma HLS stream variable=v1586 depth=9
  #pragma HLS stream variable=v1587 depth=9
  #pragma HLS stream variable=v1588 depth=9
  #pragma HLS stream variable=v1589 depth=9
  #pragma HLS array_partition variable=v1590 complete dim=1
  #pragma HLS array_partition variable=v1590 complete dim=2

  int32_t v61;	// L1900
  v61 = 0;	// L1901
  l_reduction_k61: for (int k61 = 0; k61 < 128; k61++) {	// L1902
  #pragma HLS pipeline II=1
    int8_t v1595 = v1586.read(); // v1586[k61];	// L1903
    int8_t a61;	// L1904
    a61 = v1595;	// L1905
    int8_t v1597 = v1587.read(); // v1587[k61];	// L1906
    int8_t b61;	// L1907
    b61 = v1597;	// L1908
    int8_t v1599 = a61;	// L1909
    int8_t v1600 = b61;	// L1910
    int16_t v1601 = v1599;	// L1911
    int16_t v1602 = v1600;	// L1912
    int16_t v1603 = v1601 * v1602;	// L1913
    int32_t v1604 = v61;	// L1914
    ap_int<33> v1605 = v1604;	// L1915
    ap_int<33> v1606 = v1603;	// L1916
    ap_int<33> v1607 = v1605 + v1606;	// L1917
    int32_t v1608 = v1607;	// L1918
    v61 = v1608;	// L1919
    int8_t v1609 = a61;	// L1920
    v1588.write(v1609); // v1588[k61] = v1609;	// L1921
    int8_t v1610 = b61;	// L1922
    v1589.write(v1610); // v1589[k61] = v1610;	// L1923
  }
  int32_t v1611 = v61;	// L1925
  v1590[v1591][v1592] = v1611;	// L1926
}

void PE_kernel_gemm_6_7(
  hls::stream< int8_t > &v1612 /* v1612[128] */,
  hls::stream< int8_t > &v1613 /* v1613[128] */,
  hls::stream< int8_t > &v1614 /* v1614[128] */,
  hls::stream< int8_t > &v1615 /* v1615[128] */,
  int32_t v1616[8][8],
  int v1617,
  int v1618
) {	// L1929
  #pragma HLS stream variable=v1612 depth=9
  #pragma HLS stream variable=v1613 depth=9
  #pragma HLS stream variable=v1614 depth=9
  #pragma HLS stream variable=v1615 depth=9
  #pragma HLS array_partition variable=v1616 complete dim=1
  #pragma HLS array_partition variable=v1616 complete dim=2

  int32_t v62;	// L1931
  v62 = 0;	// L1932
  l_reduction_k62: for (int k62 = 0; k62 < 128; k62++) {	// L1933
  #pragma HLS pipeline II=1
    int8_t v1621 = v1612.read(); // v1612[k62];	// L1934
    int8_t a62;	// L1935
    a62 = v1621;	// L1936
    int8_t v1623 = v1613.read(); // v1613[k62];	// L1937
    int8_t b62;	// L1938
    b62 = v1623;	// L1939
    int8_t v1625 = a62;	// L1940
    int8_t v1626 = b62;	// L1941
    int16_t v1627 = v1625;	// L1942
    int16_t v1628 = v1626;	// L1943
    int16_t v1629 = v1627 * v1628;	// L1944
    int32_t v1630 = v62;	// L1945
    ap_int<33> v1631 = v1630;	// L1946
    ap_int<33> v1632 = v1629;	// L1947
    ap_int<33> v1633 = v1631 + v1632;	// L1948
    int32_t v1634 = v1633;	// L1949
    v62 = v1634;	// L1950
    int8_t v1635 = a62;	// L1951
    v1614.write(v1635); // v1614[k62] = v1635;	// L1952
    int8_t v1636 = b62;	// L1953
    v1615.write(v1636); // v1615[k62] = v1636;	// L1954
  }
  int32_t v1637 = v62;	// L1956
  v1616[v1617][v1618] = v1637;	// L1957
}

void PE_kernel_gemm_7_7(
  hls::stream< int8_t > &v1638 /* v1638[128] */,
  hls::stream< int8_t > &v1639 /* v1639[128] */,
  hls::stream< int8_t > &v1640 /* v1640[128] */,
  hls::stream< int8_t > &v1641 /* v1641[128] */,
  int32_t v1642[8][8],
  int v1643,
  int v1644
) {	// L1960
  #pragma HLS stream variable=v1638 depth=9
  #pragma HLS stream variable=v1639 depth=9
  #pragma HLS stream variable=v1640 depth=9
  #pragma HLS stream variable=v1641 depth=9
  #pragma HLS array_partition variable=v1642 complete dim=1
  #pragma HLS array_partition variable=v1642 complete dim=2

  int32_t v63;	// L1962
  v63 = 0;	// L1963
  l_reduction_k63: for (int k63 = 0; k63 < 128; k63++) {	// L1964
  #pragma HLS pipeline II=1
    int8_t v1647 = v1638.read(); // v1638[k63];	// L1965
    int8_t a63;	// L1966
    a63 = v1647;	// L1967
    int8_t v1649 = v1639.read(); // v1639[k63];	// L1968
    int8_t b63;	// L1969
    b63 = v1649;	// L1970
    int8_t v1651 = a63;	// L1971
    int8_t v1652 = b63;	// L1972
    int16_t v1653 = v1651;	// L1973
    int16_t v1654 = v1652;	// L1974
    int16_t v1655 = v1653 * v1654;	// L1975
    int32_t v1656 = v63;	// L1976
    ap_int<33> v1657 = v1656;	// L1977
    ap_int<33> v1658 = v1655;	// L1978
    ap_int<33> v1659 = v1657 + v1658;	// L1979
    int32_t v1660 = v1659;	// L1980
    v63 = v1660;	// L1981
    int8_t v1661 = a63;	// L1982
    v1640.write(v1661); // v1640[k63] = v1661;	// L1983
    int8_t v1662 = b63;	// L1984
    v1641.write(v1662); // v1641[k63] = v1662;	// L1985
  }
  int32_t v1663 = v63;	// L1987
  v1642[v1643][v1644] = v1663;	// L1988
}

void systolic_tile_gemm(
  int8_t v1664[8][128],
  int8_t v1665[128][8],
  int32_t v1666[8][8]
) {	// L1991
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v1664 complete dim=1

  #pragma HLS array_partition variable=v1665 complete dim=2

  #pragma HLS array_partition variable=v1666 complete dim=1
  #pragma HLS array_partition variable=v1666 complete dim=2

  hls::stream< int8_t > A_fifo[8][9] /* A_fifo[8][9][128] */;	// L1992
  #pragma HLS stream variable=A_fifo depth=9
  hls::stream< int8_t > B_fifo[8][9] /* B_fifo[8][9][128] */;	// L1993
  #pragma HLS stream variable=B_fifo depth=9
  int8_t A_drain[8];	// L1994
  int8_t B_drain[8];	// L1995
  l_data_load_k64: for (int k64 = 0; k64 < 128; k64++) {	// L1996
    l_S_m_0_m: for (int m = 0; m < 8; m++) {	// L1997
      int8_t v1673 = v1664[m][k64];	// L1998
      A_fifo[m][0].write(v1673); // A_fifo[m][0][k64] = v1673;	// L1999
    }
    l_S_n_1_n: for (int n = 0; n < 8; n++) {	// L2001
      int8_t v1675 = v1665[k64][n];	// L2002
      B_fifo[n][0].write(v1675); // B_fifo[n][0][k64] = v1675;	// L2003
    }
  }
  hls::stream< int8_t > &v1676 /* v1676[128] */ = A_fifo[0][0];	// L2007
  hls::stream< int8_t > &v1677 /* v1677[128] */ = B_fifo[0][0];	// L2008
  hls::stream< int8_t > &v1678 /* v1678[128] */ = A_fifo[0][1];	// L2014
  hls::stream< int8_t > &v1679 /* v1679[128] */ = B_fifo[0][1];	// L2015
  PE_kernel_gemm_0_0(v1676, v1677, v1678, v1679, v1666, 0, 0);	// L2016
  hls::stream< int8_t > &v1680 /* v1680[128] */ = A_fifo[0][1];	// L2018
  hls::stream< int8_t > &v1681 /* v1681[128] */ = B_fifo[1][0];	// L2019
  hls::stream< int8_t > &v1682 /* v1682[128] */ = A_fifo[0][2];	// L2023
  hls::stream< int8_t > &v1683 /* v1683[128] */ = B_fifo[1][1];	// L2024
  PE_kernel_gemm_1_0(v1680, v1681, v1682, v1683, v1666, 0, 1);	// L2025
  hls::stream< int8_t > &v1684 /* v1684[128] */ = A_fifo[0][2];	// L2027
  hls::stream< int8_t > &v1685 /* v1685[128] */ = B_fifo[2][0];	// L2028
  hls::stream< int8_t > &v1686 /* v1686[128] */ = A_fifo[0][3];	// L2032
  hls::stream< int8_t > &v1687 /* v1687[128] */ = B_fifo[2][1];	// L2033
  PE_kernel_gemm_2_0(v1684, v1685, v1686, v1687, v1666, 0, 2);	// L2034
  hls::stream< int8_t > &v1688 /* v1688[128] */ = A_fifo[0][3];	// L2036
  hls::stream< int8_t > &v1689 /* v1689[128] */ = B_fifo[3][0];	// L2037
  hls::stream< int8_t > &v1690 /* v1690[128] */ = A_fifo[0][4];	// L2041
  hls::stream< int8_t > &v1691 /* v1691[128] */ = B_fifo[3][1];	// L2042
  PE_kernel_gemm_3_0(v1688, v1689, v1690, v1691, v1666, 0, 3);	// L2043
  hls::stream< int8_t > &v1692 /* v1692[128] */ = A_fifo[0][4];	// L2045
  hls::stream< int8_t > &v1693 /* v1693[128] */ = B_fifo[4][0];	// L2046
  hls::stream< int8_t > &v1694 /* v1694[128] */ = A_fifo[0][5];	// L2050
  hls::stream< int8_t > &v1695 /* v1695[128] */ = B_fifo[4][1];	// L2051
  PE_kernel_gemm_4_0(v1692, v1693, v1694, v1695, v1666, 0, 4);	// L2052
  hls::stream< int8_t > &v1696 /* v1696[128] */ = A_fifo[0][5];	// L2054
  hls::stream< int8_t > &v1697 /* v1697[128] */ = B_fifo[5][0];	// L2055
  hls::stream< int8_t > &v1698 /* v1698[128] */ = A_fifo[0][6];	// L2059
  hls::stream< int8_t > &v1699 /* v1699[128] */ = B_fifo[5][1];	// L2060
  PE_kernel_gemm_5_0(v1696, v1697, v1698, v1699, v1666, 0, 5);	// L2061
  hls::stream< int8_t > &v1700 /* v1700[128] */ = A_fifo[0][6];	// L2063
  hls::stream< int8_t > &v1701 /* v1701[128] */ = B_fifo[6][0];	// L2064
  hls::stream< int8_t > &v1702 /* v1702[128] */ = A_fifo[0][7];	// L2068
  hls::stream< int8_t > &v1703 /* v1703[128] */ = B_fifo[6][1];	// L2069
  PE_kernel_gemm_6_0(v1700, v1701, v1702, v1703, v1666, 0, 6);	// L2070
  hls::stream< int8_t > &v1704 /* v1704[128] */ = A_fifo[0][7];	// L2072
  hls::stream< int8_t > &v1705 /* v1705[128] */ = B_fifo[7][0];	// L2073
  hls::stream< int8_t > &v1706 /* v1706[128] */ = A_fifo[0][8];	// L2077
  hls::stream< int8_t > &v1707 /* v1707[128] */ = B_fifo[7][1];	// L2078
  PE_kernel_gemm_7_0(v1704, v1705, v1706, v1707, v1666, 0, 7);	// L2079
  hls::stream< int8_t > &v1708 /* v1708[128] */ = A_fifo[1][0];	// L2080
  hls::stream< int8_t > &v1709 /* v1709[128] */ = B_fifo[0][1];	// L2081
  hls::stream< int8_t > &v1710 /* v1710[128] */ = A_fifo[1][1];	// L2082
  hls::stream< int8_t > &v1711 /* v1711[128] */ = B_fifo[0][2];	// L2083
  PE_kernel_gemm_0_1(v1708, v1709, v1710, v1711, v1666, 1, 0);	// L2084
  hls::stream< int8_t > &v1712 /* v1712[128] */ = A_fifo[1][1];	// L2085
  hls::stream< int8_t > &v1713 /* v1713[128] */ = B_fifo[1][1];	// L2086
  hls::stream< int8_t > &v1714 /* v1714[128] */ = A_fifo[1][2];	// L2087
  hls::stream< int8_t > &v1715 /* v1715[128] */ = B_fifo[1][2];	// L2088
  PE_kernel_gemm_1_1(v1712, v1713, v1714, v1715, v1666, 1, 1);	// L2089
  hls::stream< int8_t > &v1716 /* v1716[128] */ = A_fifo[1][2];	// L2090
  hls::stream< int8_t > &v1717 /* v1717[128] */ = B_fifo[2][1];	// L2091
  hls::stream< int8_t > &v1718 /* v1718[128] */ = A_fifo[1][3];	// L2092
  hls::stream< int8_t > &v1719 /* v1719[128] */ = B_fifo[2][2];	// L2093
  PE_kernel_gemm_2_1(v1716, v1717, v1718, v1719, v1666, 1, 2);	// L2094
  hls::stream< int8_t > &v1720 /* v1720[128] */ = A_fifo[1][3];	// L2095
  hls::stream< int8_t > &v1721 /* v1721[128] */ = B_fifo[3][1];	// L2096
  hls::stream< int8_t > &v1722 /* v1722[128] */ = A_fifo[1][4];	// L2097
  hls::stream< int8_t > &v1723 /* v1723[128] */ = B_fifo[3][2];	// L2098
  PE_kernel_gemm_3_1(v1720, v1721, v1722, v1723, v1666, 1, 3);	// L2099
  hls::stream< int8_t > &v1724 /* v1724[128] */ = A_fifo[1][4];	// L2100
  hls::stream< int8_t > &v1725 /* v1725[128] */ = B_fifo[4][1];	// L2101
  hls::stream< int8_t > &v1726 /* v1726[128] */ = A_fifo[1][5];	// L2102
  hls::stream< int8_t > &v1727 /* v1727[128] */ = B_fifo[4][2];	// L2103
  PE_kernel_gemm_4_1(v1724, v1725, v1726, v1727, v1666, 1, 4);	// L2104
  hls::stream< int8_t > &v1728 /* v1728[128] */ = A_fifo[1][5];	// L2105
  hls::stream< int8_t > &v1729 /* v1729[128] */ = B_fifo[5][1];	// L2106
  hls::stream< int8_t > &v1730 /* v1730[128] */ = A_fifo[1][6];	// L2107
  hls::stream< int8_t > &v1731 /* v1731[128] */ = B_fifo[5][2];	// L2108
  PE_kernel_gemm_5_1(v1728, v1729, v1730, v1731, v1666, 1, 5);	// L2109
  hls::stream< int8_t > &v1732 /* v1732[128] */ = A_fifo[1][6];	// L2110
  hls::stream< int8_t > &v1733 /* v1733[128] */ = B_fifo[6][1];	// L2111
  hls::stream< int8_t > &v1734 /* v1734[128] */ = A_fifo[1][7];	// L2112
  hls::stream< int8_t > &v1735 /* v1735[128] */ = B_fifo[6][2];	// L2113
  PE_kernel_gemm_6_1(v1732, v1733, v1734, v1735, v1666, 1, 6);	// L2114
  hls::stream< int8_t > &v1736 /* v1736[128] */ = A_fifo[1][7];	// L2115
  hls::stream< int8_t > &v1737 /* v1737[128] */ = B_fifo[7][1];	// L2116
  hls::stream< int8_t > &v1738 /* v1738[128] */ = A_fifo[1][8];	// L2117
  hls::stream< int8_t > &v1739 /* v1739[128] */ = B_fifo[7][2];	// L2118
  PE_kernel_gemm_7_1(v1736, v1737, v1738, v1739, v1666, 1, 7);	// L2119
  hls::stream< int8_t > &v1740 /* v1740[128] */ = A_fifo[2][0];	// L2120
  hls::stream< int8_t > &v1741 /* v1741[128] */ = B_fifo[0][2];	// L2121
  hls::stream< int8_t > &v1742 /* v1742[128] */ = A_fifo[2][1];	// L2122
  hls::stream< int8_t > &v1743 /* v1743[128] */ = B_fifo[0][3];	// L2123
  PE_kernel_gemm_0_2(v1740, v1741, v1742, v1743, v1666, 2, 0);	// L2124
  hls::stream< int8_t > &v1744 /* v1744[128] */ = A_fifo[2][1];	// L2125
  hls::stream< int8_t > &v1745 /* v1745[128] */ = B_fifo[1][2];	// L2126
  hls::stream< int8_t > &v1746 /* v1746[128] */ = A_fifo[2][2];	// L2127
  hls::stream< int8_t > &v1747 /* v1747[128] */ = B_fifo[1][3];	// L2128
  PE_kernel_gemm_1_2(v1744, v1745, v1746, v1747, v1666, 2, 1);	// L2129
  hls::stream< int8_t > &v1748 /* v1748[128] */ = A_fifo[2][2];	// L2130
  hls::stream< int8_t > &v1749 /* v1749[128] */ = B_fifo[2][2];	// L2131
  hls::stream< int8_t > &v1750 /* v1750[128] */ = A_fifo[2][3];	// L2132
  hls::stream< int8_t > &v1751 /* v1751[128] */ = B_fifo[2][3];	// L2133
  PE_kernel_gemm_2_2(v1748, v1749, v1750, v1751, v1666, 2, 2);	// L2134
  hls::stream< int8_t > &v1752 /* v1752[128] */ = A_fifo[2][3];	// L2135
  hls::stream< int8_t > &v1753 /* v1753[128] */ = B_fifo[3][2];	// L2136
  hls::stream< int8_t > &v1754 /* v1754[128] */ = A_fifo[2][4];	// L2137
  hls::stream< int8_t > &v1755 /* v1755[128] */ = B_fifo[3][3];	// L2138
  PE_kernel_gemm_3_2(v1752, v1753, v1754, v1755, v1666, 2, 3);	// L2139
  hls::stream< int8_t > &v1756 /* v1756[128] */ = A_fifo[2][4];	// L2140
  hls::stream< int8_t > &v1757 /* v1757[128] */ = B_fifo[4][2];	// L2141
  hls::stream< int8_t > &v1758 /* v1758[128] */ = A_fifo[2][5];	// L2142
  hls::stream< int8_t > &v1759 /* v1759[128] */ = B_fifo[4][3];	// L2143
  PE_kernel_gemm_4_2(v1756, v1757, v1758, v1759, v1666, 2, 4);	// L2144
  hls::stream< int8_t > &v1760 /* v1760[128] */ = A_fifo[2][5];	// L2145
  hls::stream< int8_t > &v1761 /* v1761[128] */ = B_fifo[5][2];	// L2146
  hls::stream< int8_t > &v1762 /* v1762[128] */ = A_fifo[2][6];	// L2147
  hls::stream< int8_t > &v1763 /* v1763[128] */ = B_fifo[5][3];	// L2148
  PE_kernel_gemm_5_2(v1760, v1761, v1762, v1763, v1666, 2, 5);	// L2149
  hls::stream< int8_t > &v1764 /* v1764[128] */ = A_fifo[2][6];	// L2150
  hls::stream< int8_t > &v1765 /* v1765[128] */ = B_fifo[6][2];	// L2151
  hls::stream< int8_t > &v1766 /* v1766[128] */ = A_fifo[2][7];	// L2152
  hls::stream< int8_t > &v1767 /* v1767[128] */ = B_fifo[6][3];	// L2153
  PE_kernel_gemm_6_2(v1764, v1765, v1766, v1767, v1666, 2, 6);	// L2154
  hls::stream< int8_t > &v1768 /* v1768[128] */ = A_fifo[2][7];	// L2155
  hls::stream< int8_t > &v1769 /* v1769[128] */ = B_fifo[7][2];	// L2156
  hls::stream< int8_t > &v1770 /* v1770[128] */ = A_fifo[2][8];	// L2157
  hls::stream< int8_t > &v1771 /* v1771[128] */ = B_fifo[7][3];	// L2158
  PE_kernel_gemm_7_2(v1768, v1769, v1770, v1771, v1666, 2, 7);	// L2159
  hls::stream< int8_t > &v1772 /* v1772[128] */ = A_fifo[3][0];	// L2160
  hls::stream< int8_t > &v1773 /* v1773[128] */ = B_fifo[0][3];	// L2161
  hls::stream< int8_t > &v1774 /* v1774[128] */ = A_fifo[3][1];	// L2162
  hls::stream< int8_t > &v1775 /* v1775[128] */ = B_fifo[0][4];	// L2163
  PE_kernel_gemm_0_3(v1772, v1773, v1774, v1775, v1666, 3, 0);	// L2164
  hls::stream< int8_t > &v1776 /* v1776[128] */ = A_fifo[3][1];	// L2165
  hls::stream< int8_t > &v1777 /* v1777[128] */ = B_fifo[1][3];	// L2166
  hls::stream< int8_t > &v1778 /* v1778[128] */ = A_fifo[3][2];	// L2167
  hls::stream< int8_t > &v1779 /* v1779[128] */ = B_fifo[1][4];	// L2168
  PE_kernel_gemm_1_3(v1776, v1777, v1778, v1779, v1666, 3, 1);	// L2169
  hls::stream< int8_t > &v1780 /* v1780[128] */ = A_fifo[3][2];	// L2170
  hls::stream< int8_t > &v1781 /* v1781[128] */ = B_fifo[2][3];	// L2171
  hls::stream< int8_t > &v1782 /* v1782[128] */ = A_fifo[3][3];	// L2172
  hls::stream< int8_t > &v1783 /* v1783[128] */ = B_fifo[2][4];	// L2173
  PE_kernel_gemm_2_3(v1780, v1781, v1782, v1783, v1666, 3, 2);	// L2174
  hls::stream< int8_t > &v1784 /* v1784[128] */ = A_fifo[3][3];	// L2175
  hls::stream< int8_t > &v1785 /* v1785[128] */ = B_fifo[3][3];	// L2176
  hls::stream< int8_t > &v1786 /* v1786[128] */ = A_fifo[3][4];	// L2177
  hls::stream< int8_t > &v1787 /* v1787[128] */ = B_fifo[3][4];	// L2178
  PE_kernel_gemm_3_3(v1784, v1785, v1786, v1787, v1666, 3, 3);	// L2179
  hls::stream< int8_t > &v1788 /* v1788[128] */ = A_fifo[3][4];	// L2180
  hls::stream< int8_t > &v1789 /* v1789[128] */ = B_fifo[4][3];	// L2181
  hls::stream< int8_t > &v1790 /* v1790[128] */ = A_fifo[3][5];	// L2182
  hls::stream< int8_t > &v1791 /* v1791[128] */ = B_fifo[4][4];	// L2183
  PE_kernel_gemm_4_3(v1788, v1789, v1790, v1791, v1666, 3, 4);	// L2184
  hls::stream< int8_t > &v1792 /* v1792[128] */ = A_fifo[3][5];	// L2185
  hls::stream< int8_t > &v1793 /* v1793[128] */ = B_fifo[5][3];	// L2186
  hls::stream< int8_t > &v1794 /* v1794[128] */ = A_fifo[3][6];	// L2187
  hls::stream< int8_t > &v1795 /* v1795[128] */ = B_fifo[5][4];	// L2188
  PE_kernel_gemm_5_3(v1792, v1793, v1794, v1795, v1666, 3, 5);	// L2189
  hls::stream< int8_t > &v1796 /* v1796[128] */ = A_fifo[3][6];	// L2190
  hls::stream< int8_t > &v1797 /* v1797[128] */ = B_fifo[6][3];	// L2191
  hls::stream< int8_t > &v1798 /* v1798[128] */ = A_fifo[3][7];	// L2192
  hls::stream< int8_t > &v1799 /* v1799[128] */ = B_fifo[6][4];	// L2193
  PE_kernel_gemm_6_3(v1796, v1797, v1798, v1799, v1666, 3, 6);	// L2194
  hls::stream< int8_t > &v1800 /* v1800[128] */ = A_fifo[3][7];	// L2195
  hls::stream< int8_t > &v1801 /* v1801[128] */ = B_fifo[7][3];	// L2196
  hls::stream< int8_t > &v1802 /* v1802[128] */ = A_fifo[3][8];	// L2197
  hls::stream< int8_t > &v1803 /* v1803[128] */ = B_fifo[7][4];	// L2198
  PE_kernel_gemm_7_3(v1800, v1801, v1802, v1803, v1666, 3, 7);	// L2199
  hls::stream< int8_t > &v1804 /* v1804[128] */ = A_fifo[4][0];	// L2200
  hls::stream< int8_t > &v1805 /* v1805[128] */ = B_fifo[0][4];	// L2201
  hls::stream< int8_t > &v1806 /* v1806[128] */ = A_fifo[4][1];	// L2202
  hls::stream< int8_t > &v1807 /* v1807[128] */ = B_fifo[0][5];	// L2203
  PE_kernel_gemm_0_4(v1804, v1805, v1806, v1807, v1666, 4, 0);	// L2204
  hls::stream< int8_t > &v1808 /* v1808[128] */ = A_fifo[4][1];	// L2205
  hls::stream< int8_t > &v1809 /* v1809[128] */ = B_fifo[1][4];	// L2206
  hls::stream< int8_t > &v1810 /* v1810[128] */ = A_fifo[4][2];	// L2207
  hls::stream< int8_t > &v1811 /* v1811[128] */ = B_fifo[1][5];	// L2208
  PE_kernel_gemm_1_4(v1808, v1809, v1810, v1811, v1666, 4, 1);	// L2209
  hls::stream< int8_t > &v1812 /* v1812[128] */ = A_fifo[4][2];	// L2210
  hls::stream< int8_t > &v1813 /* v1813[128] */ = B_fifo[2][4];	// L2211
  hls::stream< int8_t > &v1814 /* v1814[128] */ = A_fifo[4][3];	// L2212
  hls::stream< int8_t > &v1815 /* v1815[128] */ = B_fifo[2][5];	// L2213
  PE_kernel_gemm_2_4(v1812, v1813, v1814, v1815, v1666, 4, 2);	// L2214
  hls::stream< int8_t > &v1816 /* v1816[128] */ = A_fifo[4][3];	// L2215
  hls::stream< int8_t > &v1817 /* v1817[128] */ = B_fifo[3][4];	// L2216
  hls::stream< int8_t > &v1818 /* v1818[128] */ = A_fifo[4][4];	// L2217
  hls::stream< int8_t > &v1819 /* v1819[128] */ = B_fifo[3][5];	// L2218
  PE_kernel_gemm_3_4(v1816, v1817, v1818, v1819, v1666, 4, 3);	// L2219
  hls::stream< int8_t > &v1820 /* v1820[128] */ = A_fifo[4][4];	// L2220
  hls::stream< int8_t > &v1821 /* v1821[128] */ = B_fifo[4][4];	// L2221
  hls::stream< int8_t > &v1822 /* v1822[128] */ = A_fifo[4][5];	// L2222
  hls::stream< int8_t > &v1823 /* v1823[128] */ = B_fifo[4][5];	// L2223
  PE_kernel_gemm_4_4(v1820, v1821, v1822, v1823, v1666, 4, 4);	// L2224
  hls::stream< int8_t > &v1824 /* v1824[128] */ = A_fifo[4][5];	// L2225
  hls::stream< int8_t > &v1825 /* v1825[128] */ = B_fifo[5][4];	// L2226
  hls::stream< int8_t > &v1826 /* v1826[128] */ = A_fifo[4][6];	// L2227
  hls::stream< int8_t > &v1827 /* v1827[128] */ = B_fifo[5][5];	// L2228
  PE_kernel_gemm_5_4(v1824, v1825, v1826, v1827, v1666, 4, 5);	// L2229
  hls::stream< int8_t > &v1828 /* v1828[128] */ = A_fifo[4][6];	// L2230
  hls::stream< int8_t > &v1829 /* v1829[128] */ = B_fifo[6][4];	// L2231
  hls::stream< int8_t > &v1830 /* v1830[128] */ = A_fifo[4][7];	// L2232
  hls::stream< int8_t > &v1831 /* v1831[128] */ = B_fifo[6][5];	// L2233
  PE_kernel_gemm_6_4(v1828, v1829, v1830, v1831, v1666, 4, 6);	// L2234
  hls::stream< int8_t > &v1832 /* v1832[128] */ = A_fifo[4][7];	// L2235
  hls::stream< int8_t > &v1833 /* v1833[128] */ = B_fifo[7][4];	// L2236
  hls::stream< int8_t > &v1834 /* v1834[128] */ = A_fifo[4][8];	// L2237
  hls::stream< int8_t > &v1835 /* v1835[128] */ = B_fifo[7][5];	// L2238
  PE_kernel_gemm_7_4(v1832, v1833, v1834, v1835, v1666, 4, 7);	// L2239
  hls::stream< int8_t > &v1836 /* v1836[128] */ = A_fifo[5][0];	// L2240
  hls::stream< int8_t > &v1837 /* v1837[128] */ = B_fifo[0][5];	// L2241
  hls::stream< int8_t > &v1838 /* v1838[128] */ = A_fifo[5][1];	// L2242
  hls::stream< int8_t > &v1839 /* v1839[128] */ = B_fifo[0][6];	// L2243
  PE_kernel_gemm_0_5(v1836, v1837, v1838, v1839, v1666, 5, 0);	// L2244
  hls::stream< int8_t > &v1840 /* v1840[128] */ = A_fifo[5][1];	// L2245
  hls::stream< int8_t > &v1841 /* v1841[128] */ = B_fifo[1][5];	// L2246
  hls::stream< int8_t > &v1842 /* v1842[128] */ = A_fifo[5][2];	// L2247
  hls::stream< int8_t > &v1843 /* v1843[128] */ = B_fifo[1][6];	// L2248
  PE_kernel_gemm_1_5(v1840, v1841, v1842, v1843, v1666, 5, 1);	// L2249
  hls::stream< int8_t > &v1844 /* v1844[128] */ = A_fifo[5][2];	// L2250
  hls::stream< int8_t > &v1845 /* v1845[128] */ = B_fifo[2][5];	// L2251
  hls::stream< int8_t > &v1846 /* v1846[128] */ = A_fifo[5][3];	// L2252
  hls::stream< int8_t > &v1847 /* v1847[128] */ = B_fifo[2][6];	// L2253
  PE_kernel_gemm_2_5(v1844, v1845, v1846, v1847, v1666, 5, 2);	// L2254
  hls::stream< int8_t > &v1848 /* v1848[128] */ = A_fifo[5][3];	// L2255
  hls::stream< int8_t > &v1849 /* v1849[128] */ = B_fifo[3][5];	// L2256
  hls::stream< int8_t > &v1850 /* v1850[128] */ = A_fifo[5][4];	// L2257
  hls::stream< int8_t > &v1851 /* v1851[128] */ = B_fifo[3][6];	// L2258
  PE_kernel_gemm_3_5(v1848, v1849, v1850, v1851, v1666, 5, 3);	// L2259
  hls::stream< int8_t > &v1852 /* v1852[128] */ = A_fifo[5][4];	// L2260
  hls::stream< int8_t > &v1853 /* v1853[128] */ = B_fifo[4][5];	// L2261
  hls::stream< int8_t > &v1854 /* v1854[128] */ = A_fifo[5][5];	// L2262
  hls::stream< int8_t > &v1855 /* v1855[128] */ = B_fifo[4][6];	// L2263
  PE_kernel_gemm_4_5(v1852, v1853, v1854, v1855, v1666, 5, 4);	// L2264
  hls::stream< int8_t > &v1856 /* v1856[128] */ = A_fifo[5][5];	// L2265
  hls::stream< int8_t > &v1857 /* v1857[128] */ = B_fifo[5][5];	// L2266
  hls::stream< int8_t > &v1858 /* v1858[128] */ = A_fifo[5][6];	// L2267
  hls::stream< int8_t > &v1859 /* v1859[128] */ = B_fifo[5][6];	// L2268
  PE_kernel_gemm_5_5(v1856, v1857, v1858, v1859, v1666, 5, 5);	// L2269
  hls::stream< int8_t > &v1860 /* v1860[128] */ = A_fifo[5][6];	// L2270
  hls::stream< int8_t > &v1861 /* v1861[128] */ = B_fifo[6][5];	// L2271
  hls::stream< int8_t > &v1862 /* v1862[128] */ = A_fifo[5][7];	// L2272
  hls::stream< int8_t > &v1863 /* v1863[128] */ = B_fifo[6][6];	// L2273
  PE_kernel_gemm_6_5(v1860, v1861, v1862, v1863, v1666, 5, 6);	// L2274
  hls::stream< int8_t > &v1864 /* v1864[128] */ = A_fifo[5][7];	// L2275
  hls::stream< int8_t > &v1865 /* v1865[128] */ = B_fifo[7][5];	// L2276
  hls::stream< int8_t > &v1866 /* v1866[128] */ = A_fifo[5][8];	// L2277
  hls::stream< int8_t > &v1867 /* v1867[128] */ = B_fifo[7][6];	// L2278
  PE_kernel_gemm_7_5(v1864, v1865, v1866, v1867, v1666, 5, 7);	// L2279
  hls::stream< int8_t > &v1868 /* v1868[128] */ = A_fifo[6][0];	// L2280
  hls::stream< int8_t > &v1869 /* v1869[128] */ = B_fifo[0][6];	// L2281
  hls::stream< int8_t > &v1870 /* v1870[128] */ = A_fifo[6][1];	// L2282
  hls::stream< int8_t > &v1871 /* v1871[128] */ = B_fifo[0][7];	// L2283
  PE_kernel_gemm_0_6(v1868, v1869, v1870, v1871, v1666, 6, 0);	// L2284
  hls::stream< int8_t > &v1872 /* v1872[128] */ = A_fifo[6][1];	// L2285
  hls::stream< int8_t > &v1873 /* v1873[128] */ = B_fifo[1][6];	// L2286
  hls::stream< int8_t > &v1874 /* v1874[128] */ = A_fifo[6][2];	// L2287
  hls::stream< int8_t > &v1875 /* v1875[128] */ = B_fifo[1][7];	// L2288
  PE_kernel_gemm_1_6(v1872, v1873, v1874, v1875, v1666, 6, 1);	// L2289
  hls::stream< int8_t > &v1876 /* v1876[128] */ = A_fifo[6][2];	// L2290
  hls::stream< int8_t > &v1877 /* v1877[128] */ = B_fifo[2][6];	// L2291
  hls::stream< int8_t > &v1878 /* v1878[128] */ = A_fifo[6][3];	// L2292
  hls::stream< int8_t > &v1879 /* v1879[128] */ = B_fifo[2][7];	// L2293
  PE_kernel_gemm_2_6(v1876, v1877, v1878, v1879, v1666, 6, 2);	// L2294
  hls::stream< int8_t > &v1880 /* v1880[128] */ = A_fifo[6][3];	// L2295
  hls::stream< int8_t > &v1881 /* v1881[128] */ = B_fifo[3][6];	// L2296
  hls::stream< int8_t > &v1882 /* v1882[128] */ = A_fifo[6][4];	// L2297
  hls::stream< int8_t > &v1883 /* v1883[128] */ = B_fifo[3][7];	// L2298
  PE_kernel_gemm_3_6(v1880, v1881, v1882, v1883, v1666, 6, 3);	// L2299
  hls::stream< int8_t > &v1884 /* v1884[128] */ = A_fifo[6][4];	// L2300
  hls::stream< int8_t > &v1885 /* v1885[128] */ = B_fifo[4][6];	// L2301
  hls::stream< int8_t > &v1886 /* v1886[128] */ = A_fifo[6][5];	// L2302
  hls::stream< int8_t > &v1887 /* v1887[128] */ = B_fifo[4][7];	// L2303
  PE_kernel_gemm_4_6(v1884, v1885, v1886, v1887, v1666, 6, 4);	// L2304
  hls::stream< int8_t > &v1888 /* v1888[128] */ = A_fifo[6][5];	// L2305
  hls::stream< int8_t > &v1889 /* v1889[128] */ = B_fifo[5][6];	// L2306
  hls::stream< int8_t > &v1890 /* v1890[128] */ = A_fifo[6][6];	// L2307
  hls::stream< int8_t > &v1891 /* v1891[128] */ = B_fifo[5][7];	// L2308
  PE_kernel_gemm_5_6(v1888, v1889, v1890, v1891, v1666, 6, 5);	// L2309
  hls::stream< int8_t > &v1892 /* v1892[128] */ = A_fifo[6][6];	// L2310
  hls::stream< int8_t > &v1893 /* v1893[128] */ = B_fifo[6][6];	// L2311
  hls::stream< int8_t > &v1894 /* v1894[128] */ = A_fifo[6][7];	// L2312
  hls::stream< int8_t > &v1895 /* v1895[128] */ = B_fifo[6][7];	// L2313
  PE_kernel_gemm_6_6(v1892, v1893, v1894, v1895, v1666, 6, 6);	// L2314
  hls::stream< int8_t > &v1896 /* v1896[128] */ = A_fifo[6][7];	// L2315
  hls::stream< int8_t > &v1897 /* v1897[128] */ = B_fifo[7][6];	// L2316
  hls::stream< int8_t > &v1898 /* v1898[128] */ = A_fifo[6][8];	// L2317
  hls::stream< int8_t > &v1899 /* v1899[128] */ = B_fifo[7][7];	// L2318
  PE_kernel_gemm_7_6(v1896, v1897, v1898, v1899, v1666, 6, 7);	// L2319
  hls::stream< int8_t > &v1900 /* v1900[128] */ = A_fifo[7][0];	// L2320
  hls::stream< int8_t > &v1901 /* v1901[128] */ = B_fifo[0][7];	// L2321
  hls::stream< int8_t > &v1902 /* v1902[128] */ = A_fifo[7][1];	// L2322
  hls::stream< int8_t > &v1903 /* v1903[128] */ = B_fifo[0][8];	// L2323
  PE_kernel_gemm_0_7(v1900, v1901, v1902, v1903, v1666, 7, 0);	// L2324
  hls::stream< int8_t > &v1904 /* v1904[128] */ = A_fifo[7][1];	// L2325
  hls::stream< int8_t > &v1905 /* v1905[128] */ = B_fifo[1][7];	// L2326
  hls::stream< int8_t > &v1906 /* v1906[128] */ = A_fifo[7][2];	// L2327
  hls::stream< int8_t > &v1907 /* v1907[128] */ = B_fifo[1][8];	// L2328
  PE_kernel_gemm_1_7(v1904, v1905, v1906, v1907, v1666, 7, 1);	// L2329
  hls::stream< int8_t > &v1908 /* v1908[128] */ = A_fifo[7][2];	// L2330
  hls::stream< int8_t > &v1909 /* v1909[128] */ = B_fifo[2][7];	// L2331
  hls::stream< int8_t > &v1910 /* v1910[128] */ = A_fifo[7][3];	// L2332
  hls::stream< int8_t > &v1911 /* v1911[128] */ = B_fifo[2][8];	// L2333
  PE_kernel_gemm_2_7(v1908, v1909, v1910, v1911, v1666, 7, 2);	// L2334
  hls::stream< int8_t > &v1912 /* v1912[128] */ = A_fifo[7][3];	// L2335
  hls::stream< int8_t > &v1913 /* v1913[128] */ = B_fifo[3][7];	// L2336
  hls::stream< int8_t > &v1914 /* v1914[128] */ = A_fifo[7][4];	// L2337
  hls::stream< int8_t > &v1915 /* v1915[128] */ = B_fifo[3][8];	// L2338
  PE_kernel_gemm_3_7(v1912, v1913, v1914, v1915, v1666, 7, 3);	// L2339
  hls::stream< int8_t > &v1916 /* v1916[128] */ = A_fifo[7][4];	// L2340
  hls::stream< int8_t > &v1917 /* v1917[128] */ = B_fifo[4][7];	// L2341
  hls::stream< int8_t > &v1918 /* v1918[128] */ = A_fifo[7][5];	// L2342
  hls::stream< int8_t > &v1919 /* v1919[128] */ = B_fifo[4][8];	// L2343
  PE_kernel_gemm_4_7(v1916, v1917, v1918, v1919, v1666, 7, 4);	// L2344
  hls::stream< int8_t > &v1920 /* v1920[128] */ = A_fifo[7][5];	// L2345
  hls::stream< int8_t > &v1921 /* v1921[128] */ = B_fifo[5][7];	// L2346
  hls::stream< int8_t > &v1922 /* v1922[128] */ = A_fifo[7][6];	// L2347
  hls::stream< int8_t > &v1923 /* v1923[128] */ = B_fifo[5][8];	// L2348
  PE_kernel_gemm_5_7(v1920, v1921, v1922, v1923, v1666, 7, 5);	// L2349
  hls::stream< int8_t > &v1924 /* v1924[128] */ = A_fifo[7][6];	// L2350
  hls::stream< int8_t > &v1925 /* v1925[128] */ = B_fifo[6][7];	// L2351
  hls::stream< int8_t > &v1926 /* v1926[128] */ = A_fifo[7][7];	// L2352
  hls::stream< int8_t > &v1927 /* v1927[128] */ = B_fifo[6][8];	// L2353
  PE_kernel_gemm_6_7(v1924, v1925, v1926, v1927, v1666, 7, 6);	// L2354
  hls::stream< int8_t > &v1928 /* v1928[128] */ = A_fifo[7][7];	// L2355
  hls::stream< int8_t > &v1929 /* v1929[128] */ = B_fifo[7][7];	// L2356
  hls::stream< int8_t > &v1930 /* v1930[128] */ = A_fifo[7][8];	// L2357
  hls::stream< int8_t > &v1931 /* v1931[128] */ = B_fifo[7][8];	// L2358
  PE_kernel_gemm_7_7(v1928, v1929, v1930, v1931, v1666, 7, 7);	// L2359
  l_data_drain_k65: for (int k65 = 0; k65 < 128; k65++) {	// L2360
    l_S_m_4_m1: for (int m1 = 0; m1 < 8; m1++) {	// L2361
      int8_t v1934 = A_fifo[m1][8].read(); // A_fifo[m1][8][k65];	// L2362
      A_drain[m1] = v1934;	// L2363
    }
    l_S_n_5_n1: for (int n1 = 0; n1 < 8; n1++) {	// L2365
      int8_t v1936 = B_fifo[n1][8].read(); // B_fifo[n1][8][k65];	// L2366
      B_drain[n1] = v1936;	// L2367
    }
  }
}

void systolic_gemm(
  int8_t v1937[128][128],
  int8_t v1938[128][128],
  int32_t v1939[128][128]
) {	// L2372
  int8_t local_A[8][128];	// L2373
  #pragma HLS array_partition variable=local_A complete dim=1

  int8_t local_B[128][8];	// L2374
  #pragma HLS array_partition variable=local_B complete dim=2

  int32_t local_C[8][8];	// L2375
  #pragma HLS array_partition variable=local_C complete dim=1
  #pragma HLS array_partition variable=local_C complete dim=2

  l_outer_tile_mi_ni_fused: for (int mi_ni_fused = 0; mi_ni_fused < 256; mi_ni_fused++) {	// L2376
    int v1944 = (mi_ni_fused % 16);	// L2377
    int v1945 = (mi_ni_fused / 16);	// L2378
    l_load_A_tile_ak: for (int ak = 0; ak < 128; ak++) {	// L2379
    #pragma HLS pipeline II=1
      l_ai: for (int ai = 0; ai < 8; ai++) {	// L2380
        ap_int<33> v1948 = v1944;	// L2381
        bool v1949 = v1948 == 0;	// L2384
        if (v1949) {	// L2385
          int8_t v1950 = v1937[((v1945 * 8) + ai)][ak];	// L2386
          local_A[ai][ak] = v1950;	// L2387
        }
      }
    }
    l_load_B_tile_bk: for (int bk = 0; bk < 128; bk++) {	// L2391
    #pragma HLS pipeline II=1
      l_bj: for (int bj = 0; bj < 8; bj++) {	// L2392
        int8_t v1953 = v1938[bk][((v1944 * 8) + bj)];	// L2393
        local_B[bk][bj] = v1953;	// L2394
      }
    }
    systolic_tile_gemm(local_A, local_B, local_C);	// L2397
    l_store_C_tile_sj: for (int sj = 0; sj < 8; sj++) {	// L2398
    #pragma HLS pipeline II=1
      l_si: for (int si = 0; si < 8; si++) {	// L2399
        int32_t v1956 = local_C[si][sj];	// L2400
        v1939[((v1945 * 8) + si)][((v1944 * 8) + sj)] = v1956;	// L2401
      }
    }
  }
}

void load_buf0(
  int8_t v1957[16384],
  int8_t v1958[128][128]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 128; load_buf0_l_0++) {	//
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 128; load_buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v1961 = v1957[((load_buf0_l_0 * 128) + load_buf0_l_1)];	//
      v1958[load_buf0_l_0][load_buf0_l_1] = v1961;	//
    }
  }
}

void load_buf1(
  int8_t v1962[16384],
  int8_t v1963[128][128]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 128; load_buf1_l_0++) {	//
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 128; load_buf1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v1966 = v1962[((load_buf1_l_0 * 128) + load_buf1_l_1)];	//
      v1963[load_buf1_l_0][load_buf1_l_1] = v1966;	//
    }
  }
}

void store_res2(
  int32_t v1967[128][128],
  int32_t v1968[16384]
) {	//
  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 128; store_res2_l_0++) {	//
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 128; store_res2_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int32_t v1971 = v1967[store_res2_l_0][store_res2_l_1];	//
      v1968[((store_res2_l_0 * 128) + store_res2_l_1)] = v1971;	//
    }
  }
}

/// This is top function.
void gemm(
  int8_t *v1972,
  int8_t *v1973,
  int32_t *v1974
) {	// L2407
  #pragma HLS interface m_axi port=v1972 offset=slave bundle=gmem0 depth=16384
  #pragma HLS interface m_axi port=v1973 offset=slave bundle=gmem1 depth=16384
  #pragma HLS interface m_axi port=v1974 offset=slave bundle=gmem2 depth=16384
  int8_t buf0[128][128];	//
  load_buf0(v1972, buf0);	//
  int8_t buf1[128][128];	//
  load_buf1(v1973, buf1);	//
  int32_t buf2[128][128];	//
  systolic_gemm(buf0, buf1, buf2);	// L2408
  store_res2(buf2, v1974);	//
}


} // extern "C"

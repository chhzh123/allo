
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
  int32_t v4[4][4],
  int v5,
  int v6
) {	// L7
  #pragma HLS stream variable=v0 depth=5
  #pragma HLS stream variable=v1 depth=5
  #pragma HLS stream variable=v2 depth=5
  #pragma HLS stream variable=v3 depth=5
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
  int32_t v30[4][4],
  int v31,
  int v32
) {	// L38
  #pragma HLS stream variable=v26 depth=5
  #pragma HLS stream variable=v27 depth=5
  #pragma HLS stream variable=v28 depth=5
  #pragma HLS stream variable=v29 depth=5
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
  int32_t v56[4][4],
  int v57,
  int v58
) {	// L69
  #pragma HLS stream variable=v52 depth=5
  #pragma HLS stream variable=v53 depth=5
  #pragma HLS stream variable=v54 depth=5
  #pragma HLS stream variable=v55 depth=5
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
  int32_t v82[4][4],
  int v83,
  int v84
) {	// L100
  #pragma HLS stream variable=v78 depth=5
  #pragma HLS stream variable=v79 depth=5
  #pragma HLS stream variable=v80 depth=5
  #pragma HLS stream variable=v81 depth=5
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

void PE_kernel_gemm_0_1(
  hls::stream< int8_t > &v104 /* v104[128] */,
  hls::stream< int8_t > &v105 /* v105[128] */,
  hls::stream< int8_t > &v106 /* v106[128] */,
  hls::stream< int8_t > &v107 /* v107[128] */,
  int32_t v108[4][4],
  int v109,
  int v110
) {	// L131
  #pragma HLS stream variable=v104 depth=5
  #pragma HLS stream variable=v105 depth=5
  #pragma HLS stream variable=v106 depth=5
  #pragma HLS stream variable=v107 depth=5
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

void PE_kernel_gemm_1_1(
  hls::stream< int8_t > &v130 /* v130[128] */,
  hls::stream< int8_t > &v131 /* v131[128] */,
  hls::stream< int8_t > &v132 /* v132[128] */,
  hls::stream< int8_t > &v133 /* v133[128] */,
  int32_t v134[4][4],
  int v135,
  int v136
) {	// L162
  #pragma HLS stream variable=v130 depth=5
  #pragma HLS stream variable=v131 depth=5
  #pragma HLS stream variable=v132 depth=5
  #pragma HLS stream variable=v133 depth=5
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

void PE_kernel_gemm_2_1(
  hls::stream< int8_t > &v156 /* v156[128] */,
  hls::stream< int8_t > &v157 /* v157[128] */,
  hls::stream< int8_t > &v158 /* v158[128] */,
  hls::stream< int8_t > &v159 /* v159[128] */,
  int32_t v160[4][4],
  int v161,
  int v162
) {	// L193
  #pragma HLS stream variable=v156 depth=5
  #pragma HLS stream variable=v157 depth=5
  #pragma HLS stream variable=v158 depth=5
  #pragma HLS stream variable=v159 depth=5
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

void PE_kernel_gemm_3_1(
  hls::stream< int8_t > &v182 /* v182[128] */,
  hls::stream< int8_t > &v183 /* v183[128] */,
  hls::stream< int8_t > &v184 /* v184[128] */,
  hls::stream< int8_t > &v185 /* v185[128] */,
  int32_t v186[4][4],
  int v187,
  int v188
) {	// L224
  #pragma HLS stream variable=v182 depth=5
  #pragma HLS stream variable=v183 depth=5
  #pragma HLS stream variable=v184 depth=5
  #pragma HLS stream variable=v185 depth=5
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

void PE_kernel_gemm_0_2(
  hls::stream< int8_t > &v208 /* v208[128] */,
  hls::stream< int8_t > &v209 /* v209[128] */,
  hls::stream< int8_t > &v210 /* v210[128] */,
  hls::stream< int8_t > &v211 /* v211[128] */,
  int32_t v212[4][4],
  int v213,
  int v214
) {	// L255
  #pragma HLS stream variable=v208 depth=5
  #pragma HLS stream variable=v209 depth=5
  #pragma HLS stream variable=v210 depth=5
  #pragma HLS stream variable=v211 depth=5
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

void PE_kernel_gemm_1_2(
  hls::stream< int8_t > &v234 /* v234[128] */,
  hls::stream< int8_t > &v235 /* v235[128] */,
  hls::stream< int8_t > &v236 /* v236[128] */,
  hls::stream< int8_t > &v237 /* v237[128] */,
  int32_t v238[4][4],
  int v239,
  int v240
) {	// L286
  #pragma HLS stream variable=v234 depth=5
  #pragma HLS stream variable=v235 depth=5
  #pragma HLS stream variable=v236 depth=5
  #pragma HLS stream variable=v237 depth=5
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

void PE_kernel_gemm_2_2(
  hls::stream< int8_t > &v260 /* v260[128] */,
  hls::stream< int8_t > &v261 /* v261[128] */,
  hls::stream< int8_t > &v262 /* v262[128] */,
  hls::stream< int8_t > &v263 /* v263[128] */,
  int32_t v264[4][4],
  int v265,
  int v266
) {	// L317
  #pragma HLS stream variable=v260 depth=5
  #pragma HLS stream variable=v261 depth=5
  #pragma HLS stream variable=v262 depth=5
  #pragma HLS stream variable=v263 depth=5
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

void PE_kernel_gemm_3_2(
  hls::stream< int8_t > &v286 /* v286[128] */,
  hls::stream< int8_t > &v287 /* v287[128] */,
  hls::stream< int8_t > &v288 /* v288[128] */,
  hls::stream< int8_t > &v289 /* v289[128] */,
  int32_t v290[4][4],
  int v291,
  int v292
) {	// L348
  #pragma HLS stream variable=v286 depth=5
  #pragma HLS stream variable=v287 depth=5
  #pragma HLS stream variable=v288 depth=5
  #pragma HLS stream variable=v289 depth=5
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

void PE_kernel_gemm_0_3(
  hls::stream< int8_t > &v312 /* v312[128] */,
  hls::stream< int8_t > &v313 /* v313[128] */,
  hls::stream< int8_t > &v314 /* v314[128] */,
  hls::stream< int8_t > &v315 /* v315[128] */,
  int32_t v316[4][4],
  int v317,
  int v318
) {	// L379
  #pragma HLS stream variable=v312 depth=5
  #pragma HLS stream variable=v313 depth=5
  #pragma HLS stream variable=v314 depth=5
  #pragma HLS stream variable=v315 depth=5
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

void PE_kernel_gemm_1_3(
  hls::stream< int8_t > &v338 /* v338[128] */,
  hls::stream< int8_t > &v339 /* v339[128] */,
  hls::stream< int8_t > &v340 /* v340[128] */,
  hls::stream< int8_t > &v341 /* v341[128] */,
  int32_t v342[4][4],
  int v343,
  int v344
) {	// L410
  #pragma HLS stream variable=v338 depth=5
  #pragma HLS stream variable=v339 depth=5
  #pragma HLS stream variable=v340 depth=5
  #pragma HLS stream variable=v341 depth=5
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

void PE_kernel_gemm_2_3(
  hls::stream< int8_t > &v364 /* v364[128] */,
  hls::stream< int8_t > &v365 /* v365[128] */,
  hls::stream< int8_t > &v366 /* v366[128] */,
  hls::stream< int8_t > &v367 /* v367[128] */,
  int32_t v368[4][4],
  int v369,
  int v370
) {	// L441
  #pragma HLS stream variable=v364 depth=5
  #pragma HLS stream variable=v365 depth=5
  #pragma HLS stream variable=v366 depth=5
  #pragma HLS stream variable=v367 depth=5
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

void PE_kernel_gemm_3_3(
  hls::stream< int8_t > &v390 /* v390[128] */,
  hls::stream< int8_t > &v391 /* v391[128] */,
  hls::stream< int8_t > &v392 /* v392[128] */,
  hls::stream< int8_t > &v393 /* v393[128] */,
  int32_t v394[4][4],
  int v395,
  int v396
) {	// L472
  #pragma HLS stream variable=v390 depth=5
  #pragma HLS stream variable=v391 depth=5
  #pragma HLS stream variable=v392 depth=5
  #pragma HLS stream variable=v393 depth=5
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

void systolic_tile_gemm(
  int8_t v416[4][128],
  int8_t v417[128][4],
  int32_t v418[4][4]
) {	// L503
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v416 complete dim=1

  #pragma HLS array_partition variable=v417 complete dim=2

  #pragma HLS array_partition variable=v418 complete dim=1
  #pragma HLS array_partition variable=v418 complete dim=2

  hls::stream< int8_t > A_fifo[4][5] /* A_fifo[4][5][128] */;	// L504
  #pragma HLS stream variable=A_fifo depth=5
  hls::stream< int8_t > B_fifo[4][5] /* B_fifo[4][5][128] */;	// L505
  #pragma HLS stream variable=B_fifo depth=5
  int8_t A_drain[4];	// L506
  int8_t B_drain[4];	// L507
  l_data_load_k16: for (int k16 = 0; k16 < 128; k16++) {	// L508
    l_S_m_0_m: for (int m = 0; m < 4; m++) {	// L509
      int8_t v425 = v416[m][k16];	// L510
      A_fifo[m][0].write(v425); // A_fifo[m][0][k16] = v425;	// L511
    }
    l_S_n_1_n: for (int n = 0; n < 4; n++) {	// L513
      int8_t v427 = v417[k16][n];	// L514
      B_fifo[n][0].write(v427); // B_fifo[n][0][k16] = v427;	// L515
    }
  }
  hls::stream< int8_t > &v428 /* v428[128] */ = A_fifo[0][0];	// L519
  hls::stream< int8_t > &v429 /* v429[128] */ = B_fifo[0][0];	// L520
  hls::stream< int8_t > &v430 /* v430[128] */ = A_fifo[0][1];	// L526
  hls::stream< int8_t > &v431 /* v431[128] */ = B_fifo[0][1];	// L527
  PE_kernel_gemm_0_0(v428, v429, v430, v431, v418, 0, 0);	// L528
  hls::stream< int8_t > &v432 /* v432[128] */ = A_fifo[0][1];	// L530
  hls::stream< int8_t > &v433 /* v433[128] */ = B_fifo[1][0];	// L531
  hls::stream< int8_t > &v434 /* v434[128] */ = A_fifo[0][2];	// L535
  hls::stream< int8_t > &v435 /* v435[128] */ = B_fifo[1][1];	// L536
  PE_kernel_gemm_1_0(v432, v433, v434, v435, v418, 0, 1);	// L537
  hls::stream< int8_t > &v436 /* v436[128] */ = A_fifo[0][2];	// L539
  hls::stream< int8_t > &v437 /* v437[128] */ = B_fifo[2][0];	// L540
  hls::stream< int8_t > &v438 /* v438[128] */ = A_fifo[0][3];	// L544
  hls::stream< int8_t > &v439 /* v439[128] */ = B_fifo[2][1];	// L545
  PE_kernel_gemm_2_0(v436, v437, v438, v439, v418, 0, 2);	// L546
  hls::stream< int8_t > &v440 /* v440[128] */ = A_fifo[0][3];	// L548
  hls::stream< int8_t > &v441 /* v441[128] */ = B_fifo[3][0];	// L549
  hls::stream< int8_t > &v442 /* v442[128] */ = A_fifo[0][4];	// L553
  hls::stream< int8_t > &v443 /* v443[128] */ = B_fifo[3][1];	// L554
  PE_kernel_gemm_3_0(v440, v441, v442, v443, v418, 0, 3);	// L555
  hls::stream< int8_t > &v444 /* v444[128] */ = A_fifo[1][0];	// L556
  hls::stream< int8_t > &v445 /* v445[128] */ = B_fifo[0][1];	// L557
  hls::stream< int8_t > &v446 /* v446[128] */ = A_fifo[1][1];	// L558
  hls::stream< int8_t > &v447 /* v447[128] */ = B_fifo[0][2];	// L559
  PE_kernel_gemm_0_1(v444, v445, v446, v447, v418, 1, 0);	// L560
  hls::stream< int8_t > &v448 /* v448[128] */ = A_fifo[1][1];	// L561
  hls::stream< int8_t > &v449 /* v449[128] */ = B_fifo[1][1];	// L562
  hls::stream< int8_t > &v450 /* v450[128] */ = A_fifo[1][2];	// L563
  hls::stream< int8_t > &v451 /* v451[128] */ = B_fifo[1][2];	// L564
  PE_kernel_gemm_1_1(v448, v449, v450, v451, v418, 1, 1);	// L565
  hls::stream< int8_t > &v452 /* v452[128] */ = A_fifo[1][2];	// L566
  hls::stream< int8_t > &v453 /* v453[128] */ = B_fifo[2][1];	// L567
  hls::stream< int8_t > &v454 /* v454[128] */ = A_fifo[1][3];	// L568
  hls::stream< int8_t > &v455 /* v455[128] */ = B_fifo[2][2];	// L569
  PE_kernel_gemm_2_1(v452, v453, v454, v455, v418, 1, 2);	// L570
  hls::stream< int8_t > &v456 /* v456[128] */ = A_fifo[1][3];	// L571
  hls::stream< int8_t > &v457 /* v457[128] */ = B_fifo[3][1];	// L572
  hls::stream< int8_t > &v458 /* v458[128] */ = A_fifo[1][4];	// L573
  hls::stream< int8_t > &v459 /* v459[128] */ = B_fifo[3][2];	// L574
  PE_kernel_gemm_3_1(v456, v457, v458, v459, v418, 1, 3);	// L575
  hls::stream< int8_t > &v460 /* v460[128] */ = A_fifo[2][0];	// L576
  hls::stream< int8_t > &v461 /* v461[128] */ = B_fifo[0][2];	// L577
  hls::stream< int8_t > &v462 /* v462[128] */ = A_fifo[2][1];	// L578
  hls::stream< int8_t > &v463 /* v463[128] */ = B_fifo[0][3];	// L579
  PE_kernel_gemm_0_2(v460, v461, v462, v463, v418, 2, 0);	// L580
  hls::stream< int8_t > &v464 /* v464[128] */ = A_fifo[2][1];	// L581
  hls::stream< int8_t > &v465 /* v465[128] */ = B_fifo[1][2];	// L582
  hls::stream< int8_t > &v466 /* v466[128] */ = A_fifo[2][2];	// L583
  hls::stream< int8_t > &v467 /* v467[128] */ = B_fifo[1][3];	// L584
  PE_kernel_gemm_1_2(v464, v465, v466, v467, v418, 2, 1);	// L585
  hls::stream< int8_t > &v468 /* v468[128] */ = A_fifo[2][2];	// L586
  hls::stream< int8_t > &v469 /* v469[128] */ = B_fifo[2][2];	// L587
  hls::stream< int8_t > &v470 /* v470[128] */ = A_fifo[2][3];	// L588
  hls::stream< int8_t > &v471 /* v471[128] */ = B_fifo[2][3];	// L589
  PE_kernel_gemm_2_2(v468, v469, v470, v471, v418, 2, 2);	// L590
  hls::stream< int8_t > &v472 /* v472[128] */ = A_fifo[2][3];	// L591
  hls::stream< int8_t > &v473 /* v473[128] */ = B_fifo[3][2];	// L592
  hls::stream< int8_t > &v474 /* v474[128] */ = A_fifo[2][4];	// L593
  hls::stream< int8_t > &v475 /* v475[128] */ = B_fifo[3][3];	// L594
  PE_kernel_gemm_3_2(v472, v473, v474, v475, v418, 2, 3);	// L595
  hls::stream< int8_t > &v476 /* v476[128] */ = A_fifo[3][0];	// L596
  hls::stream< int8_t > &v477 /* v477[128] */ = B_fifo[0][3];	// L597
  hls::stream< int8_t > &v478 /* v478[128] */ = A_fifo[3][1];	// L598
  hls::stream< int8_t > &v479 /* v479[128] */ = B_fifo[0][4];	// L599
  PE_kernel_gemm_0_3(v476, v477, v478, v479, v418, 3, 0);	// L600
  hls::stream< int8_t > &v480 /* v480[128] */ = A_fifo[3][1];	// L601
  hls::stream< int8_t > &v481 /* v481[128] */ = B_fifo[1][3];	// L602
  hls::stream< int8_t > &v482 /* v482[128] */ = A_fifo[3][2];	// L603
  hls::stream< int8_t > &v483 /* v483[128] */ = B_fifo[1][4];	// L604
  PE_kernel_gemm_1_3(v480, v481, v482, v483, v418, 3, 1);	// L605
  hls::stream< int8_t > &v484 /* v484[128] */ = A_fifo[3][2];	// L606
  hls::stream< int8_t > &v485 /* v485[128] */ = B_fifo[2][3];	// L607
  hls::stream< int8_t > &v486 /* v486[128] */ = A_fifo[3][3];	// L608
  hls::stream< int8_t > &v487 /* v487[128] */ = B_fifo[2][4];	// L609
  PE_kernel_gemm_2_3(v484, v485, v486, v487, v418, 3, 2);	// L610
  hls::stream< int8_t > &v488 /* v488[128] */ = A_fifo[3][3];	// L611
  hls::stream< int8_t > &v489 /* v489[128] */ = B_fifo[3][3];	// L612
  hls::stream< int8_t > &v490 /* v490[128] */ = A_fifo[3][4];	// L613
  hls::stream< int8_t > &v491 /* v491[128] */ = B_fifo[3][4];	// L614
  PE_kernel_gemm_3_3(v488, v489, v490, v491, v418, 3, 3);	// L615
  l_data_drain_k17: for (int k17 = 0; k17 < 128; k17++) {	// L616
    l_S_m_4_m1: for (int m1 = 0; m1 < 4; m1++) {	// L617
      int8_t v494 = A_fifo[m1][4].read(); // A_fifo[m1][4][k17];	// L618
      A_drain[m1] = v494;	// L619
    }
    l_S_n_5_n1: for (int n1 = 0; n1 < 4; n1++) {	// L621
      int8_t v496 = B_fifo[n1][4].read(); // B_fifo[n1][4][k17];	// L622
      B_drain[n1] = v496;	// L623
    }
  }
}

void systolic_gemm(
  int8_t v497[128][128],
  int8_t v498[128][128],
  int32_t v499[128][128]
) {	// L628
  int8_t local_A[4][128];	// L629
  #pragma HLS array_partition variable=local_A complete dim=1

  int8_t local_B[128][4];	// L630
  #pragma HLS array_partition variable=local_B complete dim=2

  int32_t local_C[4][4];	// L631
  #pragma HLS array_partition variable=local_C complete dim=1
  #pragma HLS array_partition variable=local_C complete dim=2

  l_outer_tile_mi_ni_fused: for (int mi_ni_fused = 0; mi_ni_fused < 1024; mi_ni_fused++) {	// L632
    int v504 = (mi_ni_fused % 32);	// L633
    int v505 = (mi_ni_fused / 32);	// L634
    l_load_A_tile_ak: for (int ak = 0; ak < 128; ak++) {	// L635
    #pragma HLS pipeline II=1
      l_ai: for (int ai = 0; ai < 4; ai++) {	// L636
        ap_int<33> v508 = v504;	// L637
        bool v509 = v508 == 0;	// L640
        if (v509) {	// L641
          int8_t v510 = v497[((v505 * 4) + ai)][ak];	// L642
          local_A[ai][ak] = v510;	// L643
        }
      }
    }
    l_load_B_tile_bk: for (int bk = 0; bk < 128; bk++) {	// L647
    #pragma HLS pipeline II=1
      l_bj: for (int bj = 0; bj < 4; bj++) {	// L648
        int8_t v513 = v498[bk][((v504 * 4) + bj)];	// L649
        local_B[bk][bj] = v513;	// L650
      }
    }
    systolic_tile_gemm(local_A, local_B, local_C);	// L653
    l_store_C_tile_sj: for (int sj = 0; sj < 4; sj++) {	// L654
    #pragma HLS pipeline II=1
      l_si: for (int si = 0; si < 4; si++) {	// L655
        int32_t v516 = local_C[si][sj];	// L656
        v499[((v505 * 4) + si)][((v504 * 4) + sj)] = v516;	// L657
      }
    }
  }
}

void load_buf0(
  int8_t v517[16384],
  int8_t v518[128][128]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 128; load_buf0_l_0++) {	//
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 128; load_buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v521 = v517[((load_buf0_l_0 * 128) + load_buf0_l_1)];	//
      v518[load_buf0_l_0][load_buf0_l_1] = v521;	//
    }
  }
}

void load_buf1(
  int8_t v522[16384],
  int8_t v523[128][128]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 128; load_buf1_l_0++) {	//
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 128; load_buf1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int8_t v526 = v522[((load_buf1_l_0 * 128) + load_buf1_l_1)];	//
      v523[load_buf1_l_0][load_buf1_l_1] = v526;	//
    }
  }
}

void store_res2(
  int32_t v527[128][128],
  int32_t v528[16384]
) {	//
  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 128; store_res2_l_0++) {	//
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 128; store_res2_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      int32_t v531 = v527[store_res2_l_0][store_res2_l_1];	//
      v528[((store_res2_l_0 * 128) + store_res2_l_1)] = v531;	//
    }
  }
}

/// This is top function.
void gemm(
  int8_t *v532,
  int8_t *v533,
  int32_t *v534
) {	// L663
  #pragma HLS interface m_axi port=v532 offset=slave bundle=gmem0 depth=16384
  #pragma HLS interface m_axi port=v533 offset=slave bundle=gmem1 depth=16384
  #pragma HLS interface m_axi port=v534 offset=slave bundle=gmem2 depth=16384
  int8_t buf0[128][128];	//
  load_buf0(v532, buf0);	//
  int8_t buf1[128][128];	//
  load_buf1(v533, buf1);	//
  int32_t buf2[128][128];	//
  systolic_gemm(buf0, buf1, buf2);	// L664
  store_res2(buf2, v534);	//
}


} // extern "C"

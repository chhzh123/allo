
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
void pe_west_load_0(
  int8_t v0[3][3],
  hls::stream< int8_t >& v1
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 3; _t++) {	// L4
  #pragma HLS pipeline II=1
    int8_t v3 = v0[0][_t];	// L5
    v1.write(v3);	// L6
  }
}

void pe_west_load_1(
  int8_t v4[3][3],
  hls::stream< int8_t >& v5
) {	// L10
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 3; _t1++) {	// L11
  #pragma HLS pipeline II=1
    int8_t v7 = v4[1][_t1];	// L12
    v5.write(v7);	// L13
  }
}

void pe_west_load_2(
  int8_t v8[3][3],
  hls::stream< int8_t >& v9
) {	// L17
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 3; _t2++) {	// L18
  #pragma HLS pipeline II=1
    int8_t v11 = v8[2][_t2];	// L19
    v9.write(v11);	// L20
  }
}

void pe_north_load_0(
  int8_t v12[3][3],
  hls::stream< int8_t >& v13
) {	// L24
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 3; _t3++) {	// L25
  #pragma HLS pipeline II=1
    int8_t v15 = v12[_t3][0];	// L26
    v13.write(v15);	// L27
  }
}

void pe_north_load_1(
  int8_t v16[3][3],
  hls::stream< int8_t >& v17
) {	// L31
  #pragma HLS array_partition variable=v16 complete dim=1
  #pragma HLS array_partition variable=v16 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 3; _t4++) {	// L32
  #pragma HLS pipeline II=1
    int8_t v19 = v16[_t4][1];	// L33
    v17.write(v19);	// L34
  }
}

void pe_north_load_2(
  int8_t v20[3][3],
  hls::stream< int8_t >& v21
) {	// L38
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 3; _t5++) {	// L39
  #pragma HLS pipeline II=1
    int8_t v23 = v20[_t5][2];	// L40
    v21.write(v23);	// L41
  }
}

void pe_0_0(
  int32_t v24[3][3],
  hls::stream< int8_t >& v25,
  hls::stream< int8_t >& v26,
  hls::stream< int8_t >& v27,
  hls::stream< int8_t >& v28
) {	// L45
  #pragma HLS array_partition variable=v24 complete dim=1
  #pragma HLS array_partition variable=v24 complete dim=2

  int32_t acc;	// L47
  acc = 0;	// L48
  l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L49
  #pragma HLS pipeline II=1
    int8_t v31 = v25.read();	// L50
    int8_t a;	// L51
    a = v31;	// L52
    int8_t v33 = v26.read();	// L53
    int8_t b;	// L54
    b = v33;	// L55
    int8_t v35 = a;	// L56
    int8_t v36 = b;	// L57
    int16_t v37 = v35;	// L58
    int16_t v38 = v36;	// L59
    int16_t v39 = v37 * v38;	// L60
    int32_t v40 = acc;	// L61
    ap_int<33> v41 = v40;	// L62
    ap_int<33> v42 = v39;	// L63
    ap_int<33> v43 = v41 + v42;	// L64
    int32_t v44 = v43;	// L65
    acc = v44;	// L66
    int8_t v45 = a;	// L67
    v27.write(v45);	// L68
    int8_t v46 = b;	// L69
    v28.write(v46);	// L70
  }
  int32_t v47 = acc;	// L72
  v24[0][0] = v47;	// L73
}

void pe_0_1(
  int32_t v48[3][3],
  hls::stream< int8_t >& v49,
  hls::stream< int8_t >& v50,
  hls::stream< int8_t >& v51,
  hls::stream< int8_t >& v52
) {	// L76
  #pragma HLS array_partition variable=v48 complete dim=1
  #pragma HLS array_partition variable=v48 complete dim=2

  int32_t acc1;	// L78
  acc1 = 0;	// L79
  l_S_k_0_k1: for (int k1 = 0; k1 < 3; k1++) {	// L80
  #pragma HLS pipeline II=1
    int8_t v55 = v49.read();	// L81
    int8_t a1;	// L82
    a1 = v55;	// L83
    int8_t v57 = v50.read();	// L84
    int8_t b1;	// L85
    b1 = v57;	// L86
    int8_t v59 = a1;	// L87
    int8_t v60 = b1;	// L88
    int16_t v61 = v59;	// L89
    int16_t v62 = v60;	// L90
    int16_t v63 = v61 * v62;	// L91
    int32_t v64 = acc1;	// L92
    ap_int<33> v65 = v64;	// L93
    ap_int<33> v66 = v63;	// L94
    ap_int<33> v67 = v65 + v66;	// L95
    int32_t v68 = v67;	// L96
    acc1 = v68;	// L97
    int8_t v69 = a1;	// L98
    v51.write(v69);	// L99
    int8_t v70 = b1;	// L100
    v52.write(v70);	// L101
  }
  int32_t v71 = acc1;	// L103
  v48[0][1] = v71;	// L104
}

void pe_0_2(
  int32_t v72[3][3],
  hls::stream< int8_t >& v73,
  hls::stream< int8_t >& v74,
  hls::stream< int8_t >& v75
) {	// L107
  #pragma HLS array_partition variable=v72 complete dim=1
  #pragma HLS array_partition variable=v72 complete dim=2

  int32_t acc2;	// L109
  acc2 = 0;	// L110
  l_S_k_0_k2: for (int k2 = 0; k2 < 3; k2++) {	// L111
  #pragma HLS pipeline II=1
    int8_t v78 = v73.read();	// L112
    int8_t a2;	// L113
    a2 = v78;	// L114
    int8_t v80 = v74.read();	// L115
    int8_t b2;	// L116
    b2 = v80;	// L117
    int8_t v82 = a2;	// L118
    int8_t v83 = b2;	// L119
    int16_t v84 = v82;	// L120
    int16_t v85 = v83;	// L121
    int16_t v86 = v84 * v85;	// L122
    int32_t v87 = acc2;	// L123
    ap_int<33> v88 = v87;	// L124
    ap_int<33> v89 = v86;	// L125
    ap_int<33> v90 = v88 + v89;	// L126
    int32_t v91 = v90;	// L127
    acc2 = v91;	// L128
    int8_t v92 = b2;	// L129
    v75.write(v92);	// L130
  }
  int32_t v93 = acc2;	// L132
  v72[0][2] = v93;	// L133
}

void pe_1_0(
  int32_t v94[3][3],
  hls::stream< int8_t >& v95,
  hls::stream< int8_t >& v96,
  hls::stream< int8_t >& v97,
  hls::stream< int8_t >& v98
) {	// L136
  #pragma HLS array_partition variable=v94 complete dim=1
  #pragma HLS array_partition variable=v94 complete dim=2

  int32_t acc3;	// L138
  acc3 = 0;	// L139
  l_S_k_0_k3: for (int k3 = 0; k3 < 3; k3++) {	// L140
  #pragma HLS pipeline II=1
    int8_t v101 = v95.read();	// L141
    int8_t a3;	// L142
    a3 = v101;	// L143
    int8_t v103 = v96.read();	// L144
    int8_t b3;	// L145
    b3 = v103;	// L146
    int8_t v105 = a3;	// L147
    int8_t v106 = b3;	// L148
    int16_t v107 = v105;	// L149
    int16_t v108 = v106;	// L150
    int16_t v109 = v107 * v108;	// L151
    int32_t v110 = acc3;	// L152
    ap_int<33> v111 = v110;	// L153
    ap_int<33> v112 = v109;	// L154
    ap_int<33> v113 = v111 + v112;	// L155
    int32_t v114 = v113;	// L156
    acc3 = v114;	// L157
    int8_t v115 = a3;	// L158
    v97.write(v115);	// L159
    int8_t v116 = b3;	// L160
    v98.write(v116);	// L161
  }
  int32_t v117 = acc3;	// L163
  v94[1][0] = v117;	// L164
}

void pe_1_1(
  int32_t v118[3][3],
  hls::stream< int8_t >& v119,
  hls::stream< int8_t >& v120,
  hls::stream< int8_t >& v121,
  hls::stream< int8_t >& v122
) {	// L167
  #pragma HLS array_partition variable=v118 complete dim=1
  #pragma HLS array_partition variable=v118 complete dim=2

  int32_t acc4;	// L169
  acc4 = 0;	// L170
  l_S_k_0_k4: for (int k4 = 0; k4 < 3; k4++) {	// L171
  #pragma HLS pipeline II=1
    int8_t v125 = v119.read();	// L172
    int8_t a4;	// L173
    a4 = v125;	// L174
    int8_t v127 = v120.read();	// L175
    int8_t b4;	// L176
    b4 = v127;	// L177
    int8_t v129 = a4;	// L178
    int8_t v130 = b4;	// L179
    int16_t v131 = v129;	// L180
    int16_t v132 = v130;	// L181
    int16_t v133 = v131 * v132;	// L182
    int32_t v134 = acc4;	// L183
    ap_int<33> v135 = v134;	// L184
    ap_int<33> v136 = v133;	// L185
    ap_int<33> v137 = v135 + v136;	// L186
    int32_t v138 = v137;	// L187
    acc4 = v138;	// L188
    int8_t v139 = a4;	// L189
    v121.write(v139);	// L190
    int8_t v140 = b4;	// L191
    v122.write(v140);	// L192
  }
  int32_t v141 = acc4;	// L194
  v118[1][1] = v141;	// L195
}

void pe_1_2(
  int32_t v142[3][3],
  hls::stream< int8_t >& v143,
  hls::stream< int8_t >& v144,
  hls::stream< int8_t >& v145
) {	// L198
  #pragma HLS array_partition variable=v142 complete dim=1
  #pragma HLS array_partition variable=v142 complete dim=2

  int32_t acc5;	// L200
  acc5 = 0;	// L201
  l_S_k_0_k5: for (int k5 = 0; k5 < 3; k5++) {	// L202
  #pragma HLS pipeline II=1
    int8_t v148 = v143.read();	// L203
    int8_t a5;	// L204
    a5 = v148;	// L205
    int8_t v150 = v144.read();	// L206
    int8_t b5;	// L207
    b5 = v150;	// L208
    int8_t v152 = a5;	// L209
    int8_t v153 = b5;	// L210
    int16_t v154 = v152;	// L211
    int16_t v155 = v153;	// L212
    int16_t v156 = v154 * v155;	// L213
    int32_t v157 = acc5;	// L214
    ap_int<33> v158 = v157;	// L215
    ap_int<33> v159 = v156;	// L216
    ap_int<33> v160 = v158 + v159;	// L217
    int32_t v161 = v160;	// L218
    acc5 = v161;	// L219
    int8_t v162 = b5;	// L220
    v145.write(v162);	// L221
  }
  int32_t v163 = acc5;	// L223
  v142[1][2] = v163;	// L224
}

void pe_2_0(
  int32_t v164[3][3],
  hls::stream< int8_t >& v165,
  hls::stream< int8_t >& v166,
  hls::stream< int8_t >& v167
) {	// L227
  #pragma HLS array_partition variable=v164 complete dim=1
  #pragma HLS array_partition variable=v164 complete dim=2

  int32_t acc6;	// L229
  acc6 = 0;	// L230
  l_S_k_0_k6: for (int k6 = 0; k6 < 3; k6++) {	// L231
  #pragma HLS pipeline II=1
    int8_t v170 = v165.read();	// L232
    int8_t a6;	// L233
    a6 = v170;	// L234
    int8_t v172 = v166.read();	// L235
    int8_t b6;	// L236
    b6 = v172;	// L237
    int8_t v174 = a6;	// L238
    int8_t v175 = b6;	// L239
    int16_t v176 = v174;	// L240
    int16_t v177 = v175;	// L241
    int16_t v178 = v176 * v177;	// L242
    int32_t v179 = acc6;	// L243
    ap_int<33> v180 = v179;	// L244
    ap_int<33> v181 = v178;	// L245
    ap_int<33> v182 = v180 + v181;	// L246
    int32_t v183 = v182;	// L247
    acc6 = v183;	// L248
    int8_t v184 = a6;	// L249
    v167.write(v184);	// L250
  }
  int32_t v185 = acc6;	// L252
  v164[2][0] = v185;	// L253
}

void pe_2_1(
  int32_t v186[3][3],
  hls::stream< int8_t >& v187,
  hls::stream< int8_t >& v188,
  hls::stream< int8_t >& v189
) {	// L256
  #pragma HLS array_partition variable=v186 complete dim=1
  #pragma HLS array_partition variable=v186 complete dim=2

  int32_t acc7;	// L258
  acc7 = 0;	// L259
  l_S_k_0_k7: for (int k7 = 0; k7 < 3; k7++) {	// L260
  #pragma HLS pipeline II=1
    int8_t v192 = v187.read();	// L261
    int8_t a7;	// L262
    a7 = v192;	// L263
    int8_t v194 = v188.read();	// L264
    int8_t b7;	// L265
    b7 = v194;	// L266
    int8_t v196 = a7;	// L267
    int8_t v197 = b7;	// L268
    int16_t v198 = v196;	// L269
    int16_t v199 = v197;	// L270
    int16_t v200 = v198 * v199;	// L271
    int32_t v201 = acc7;	// L272
    ap_int<33> v202 = v201;	// L273
    ap_int<33> v203 = v200;	// L274
    ap_int<33> v204 = v202 + v203;	// L275
    int32_t v205 = v204;	// L276
    acc7 = v205;	// L277
    int8_t v206 = a7;	// L278
    v189.write(v206);	// L279
  }
  int32_t v207 = acc7;	// L281
  v186[2][1] = v207;	// L282
}

void pe_2_2(
  int32_t v208[3][3],
  hls::stream< int8_t >& v209,
  hls::stream< int8_t >& v210
) {	// L285
  #pragma HLS array_partition variable=v208 complete dim=1
  #pragma HLS array_partition variable=v208 complete dim=2

  int32_t acc8;	// L287
  acc8 = 0;	// L288
  l_S_k_0_k8: for (int k8 = 0; k8 < 3; k8++) {	// L289
  #pragma HLS pipeline II=1
    int8_t v213 = v209.read();	// L290
    int8_t a8;	// L291
    a8 = v213;	// L292
    int8_t v215 = v210.read();	// L293
    int8_t b8;	// L294
    b8 = v215;	// L295
    int8_t v217 = a8;	// L296
    int8_t v218 = b8;	// L297
    int16_t v219 = v217;	// L298
    int16_t v220 = v218;	// L299
    int16_t v221 = v219 * v220;	// L300
    int32_t v222 = acc8;	// L301
    ap_int<33> v223 = v222;	// L302
    ap_int<33> v224 = v221;	// L303
    ap_int<33> v225 = v223 + v224;	// L304
    int32_t v226 = v225;	// L305
    acc8 = v226;	// L306
  }
  int32_t v227 = acc8;	// L308
  v208[2][2] = v227;	// L309
}

/// This is top function.
void top(
  int8_t v228[3][3],
  int8_t v229[3][3],
  int32_t v230[3][3]
) {	// L312
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v228 complete dim=1
  #pragma HLS array_partition variable=v228 complete dim=2

  #pragma HLS array_partition variable=v229 complete dim=1
  #pragma HLS array_partition variable=v229 complete dim=2

  #pragma HLS array_partition variable=v230 complete dim=1
  #pragma HLS array_partition variable=v230 complete dim=2

  hls::stream< int8_t > v231;
  #pragma HLS stream variable=v231 depth=2	// L313
  hls::stream< int8_t > v232;
  #pragma HLS stream variable=v232 depth=2	// L314
  hls::stream< int8_t > v233;
  #pragma HLS stream variable=v233 depth=2	// L315
  hls::stream< int8_t > v234;
  #pragma HLS stream variable=v234 depth=2	// L316
  hls::stream< int8_t > v235;
  #pragma HLS stream variable=v235 depth=2	// L317
  hls::stream< int8_t > v236;
  #pragma HLS stream variable=v236 depth=2	// L318
  hls::stream< int8_t > v237;
  #pragma HLS stream variable=v237 depth=2	// L319
  hls::stream< int8_t > v238;
  #pragma HLS stream variable=v238 depth=2	// L320
  hls::stream< int8_t > v239;
  #pragma HLS stream variable=v239 depth=2	// L321
  hls::stream< int8_t > v240;
  #pragma HLS stream variable=v240 depth=2	// L322
  hls::stream< int8_t > v241;
  #pragma HLS stream variable=v241 depth=2	// L323
  hls::stream< int8_t > v242;
  #pragma HLS stream variable=v242 depth=2	// L324
  hls::stream< int8_t > v243;
  #pragma HLS stream variable=v243 depth=2	// L325
  hls::stream< int8_t > v244;
  #pragma HLS stream variable=v244 depth=2	// L326
  hls::stream< int8_t > v245;
  #pragma HLS stream variable=v245 depth=2	// L327
  hls::stream< int8_t > v246;
  #pragma HLS stream variable=v246 depth=2	// L328
  hls::stream< int8_t > v247;
  #pragma HLS stream variable=v247 depth=2	// L329
  hls::stream< int8_t > v248;
  #pragma HLS stream variable=v248 depth=2	// L330
  hls::stream< int8_t > v249;
  #pragma HLS stream variable=v249 depth=2	// L331
  hls::stream< int8_t > v250;
  #pragma HLS stream variable=v250 depth=2	// L332
  hls::stream< int8_t > v251;
  #pragma HLS stream variable=v251 depth=2	// L333
  hls::stream< int8_t > v252;
  #pragma HLS stream variable=v252 depth=2	// L334
  hls::stream< int8_t > v253;
  #pragma HLS stream variable=v253 depth=2	// L335
  hls::stream< int8_t > v254;
  #pragma HLS stream variable=v254 depth=2	// L336
  pe_west_load_0(v228, v249);	// L337
  pe_west_load_1(v228, v250);	// L338
  pe_west_load_2(v228, v251);	// L339
  pe_north_load_0(v229, v252);	// L340
  pe_north_load_1(v229, v253);	// L341
  pe_north_load_2(v229, v254);	// L342
  pe_0_0(v230, v249, v252, v232, v243);	// L343
  pe_0_1(v230, v232, v253, v233, v244);	// L344
  pe_0_2(v230, v233, v254, v245);	// L345
  pe_1_0(v230, v250, v243, v235, v246);	// L346
  pe_1_1(v230, v235, v244, v236, v247);	// L347
  pe_1_2(v230, v236, v245, v248);	// L348
  pe_2_0(v230, v251, v246, v238);	// L349
  pe_2_1(v230, v238, v247, v239);	// L350
  pe_2_2(v230, v239, v248);	// L351
}


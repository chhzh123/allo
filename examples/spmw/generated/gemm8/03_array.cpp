
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
void pe_west_load(
  int8_t v0[3][3],
  int v1,
  hls::stream< int8_t >& v2
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 3; _t++) {	// L4
  #pragma HLS pipeline II=1
    int8_t v4 = v0[v1][_t];	// L5
    v2.write(v4);	// L6
  }
}

void pe_north_load(
  int8_t v5[3][3],
  int v6,
  hls::stream< int8_t >& v7
) {	// L10
  #pragma HLS array_partition variable=v5 complete dim=1
  #pragma HLS array_partition variable=v5 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 3; _t1++) {	// L11
  #pragma HLS pipeline II=1
    int8_t v9 = v5[_t1][v6];	// L12
    v7.write(v9);	// L13
  }
}

void pe_r0(
  int32_t v10[3][3],
  int v11,
  int v12,
  hls::stream< int8_t >& v13,
  hls::stream< int8_t >& v14,
  hls::stream< int8_t >& v15
) {	// L17
  #pragma HLS array_partition variable=v10 complete dim=1
  #pragma HLS array_partition variable=v10 complete dim=2

  int32_t acc;	// L19
  acc = 0;	// L20
  l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L21
  #pragma HLS pipeline II=1
    int8_t v18 = v15.read();	// L22
    int8_t a;	// L23
    a = v18;	// L24
    int8_t v20 = v14.read();	// L25
    int8_t b;	// L26
    b = v20;	// L27
    int8_t v22 = a;	// L28
    int8_t v23 = b;	// L29
    int16_t v24 = v22;	// L30
    int16_t v25 = v23;	// L31
    int16_t v26 = v24 * v25;	// L32
    int32_t v27 = acc;	// L33
    ap_int<33> v28 = v27;	// L34
    ap_int<33> v29 = v26;	// L35
    ap_int<33> v30 = v28 + v29;	// L36
    int32_t v31 = v30;	// L37
    acc = v31;	// L38
    int8_t v32 = a;	// L39
    v13.write(v32);	// L40
  }
  int32_t v33 = acc;	// L42
  v10[v11][v12] = v33;	// L43
}

void pe_r1(
  int32_t v34[3][3],
  int v35,
  int v36,
  hls::stream< int8_t >& v37,
  hls::stream< int8_t >& v38,
  hls::stream< int8_t >& v39,
  hls::stream< int8_t >& v40
) {	// L46
  #pragma HLS array_partition variable=v34 complete dim=1
  #pragma HLS array_partition variable=v34 complete dim=2

  int32_t acc1;	// L48
  acc1 = 0;	// L49
  l_S_k_0_k1: for (int k1 = 0; k1 < 3; k1++) {	// L50
  #pragma HLS pipeline II=1
    int8_t v43 = v40.read();	// L51
    int8_t a1;	// L52
    a1 = v43;	// L53
    int8_t v45 = v38.read();	// L54
    int8_t b1;	// L55
    b1 = v45;	// L56
    int8_t v47 = a1;	// L57
    int8_t v48 = b1;	// L58
    int16_t v49 = v47;	// L59
    int16_t v50 = v48;	// L60
    int16_t v51 = v49 * v50;	// L61
    int32_t v52 = acc1;	// L62
    ap_int<33> v53 = v52;	// L63
    ap_int<33> v54 = v51;	// L64
    ap_int<33> v55 = v53 + v54;	// L65
    int32_t v56 = v55;	// L66
    acc1 = v56;	// L67
    int8_t v57 = a1;	// L68
    v37.write(v57);	// L69
    int8_t v58 = b1;	// L70
    v39.write(v58);	// L71
  }
  int32_t v59 = acc1;	// L73
  v34[v35][v36] = v59;	// L74
}

void pe_r2(
  int32_t v60[3][3],
  int v61,
  int v62,
  hls::stream< int8_t >& v63,
  hls::stream< int8_t >& v64,
  hls::stream< int8_t >& v65,
  hls::stream< int8_t >& v66
) {	// L77
  #pragma HLS array_partition variable=v60 complete dim=1
  #pragma HLS array_partition variable=v60 complete dim=2

  int32_t acc2;	// L79
  acc2 = 0;	// L80
  l_S_k_0_k2: for (int k2 = 0; k2 < 3; k2++) {	// L81
  #pragma HLS pipeline II=1
    int8_t v69 = v66.read();	// L82
    int8_t a2;	// L83
    a2 = v69;	// L84
    int8_t v71 = v64.read();	// L85
    int8_t b2;	// L86
    b2 = v71;	// L87
    int8_t v73 = a2;	// L88
    int8_t v74 = b2;	// L89
    int16_t v75 = v73;	// L90
    int16_t v76 = v74;	// L91
    int16_t v77 = v75 * v76;	// L92
    int32_t v78 = acc2;	// L93
    ap_int<33> v79 = v78;	// L94
    ap_int<33> v80 = v77;	// L95
    ap_int<33> v81 = v79 + v80;	// L96
    int32_t v82 = v81;	// L97
    acc2 = v82;	// L98
    int8_t v83 = a2;	// L99
    v63.write(v83);	// L100
    int8_t v84 = b2;	// L101
    v65.write(v84);	// L102
  }
  int32_t v85 = acc2;	// L104
  v60[v61][v62] = v85;	// L105
}

void pe_r3(
  int32_t v86[3][3],
  int v87,
  int v88,
  hls::stream< int8_t >& v89,
  hls::stream< int8_t >& v90,
  hls::stream< int8_t >& v91
) {	// L108
  #pragma HLS array_partition variable=v86 complete dim=1
  #pragma HLS array_partition variable=v86 complete dim=2

  int32_t acc3;	// L110
  acc3 = 0;	// L111
  l_S_k_0_k3: for (int k3 = 0; k3 < 3; k3++) {	// L112
  #pragma HLS pipeline II=1
    int8_t v94 = v91.read();	// L113
    int8_t a3;	// L114
    a3 = v94;	// L115
    int8_t v96 = v90.read();	// L116
    int8_t b3;	// L117
    b3 = v96;	// L118
    int8_t v98 = a3;	// L119
    int8_t v99 = b3;	// L120
    int16_t v100 = v98;	// L121
    int16_t v101 = v99;	// L122
    int16_t v102 = v100 * v101;	// L123
    int32_t v103 = acc3;	// L124
    ap_int<33> v104 = v103;	// L125
    ap_int<33> v105 = v102;	// L126
    ap_int<33> v106 = v104 + v105;	// L127
    int32_t v107 = v106;	// L128
    acc3 = v107;	// L129
    int8_t v108 = a3;	// L130
    v89.write(v108);	// L131
  }
  int32_t v109 = acc3;	// L133
  v86[v87][v88] = v109;	// L134
}

void pe_r4(
  int32_t v110[3][3],
  int v111,
  int v112,
  hls::stream< int8_t >& v113,
  hls::stream< int8_t >& v114,
  hls::stream< int8_t >& v115,
  hls::stream< int8_t >& v116
) {	// L137
  #pragma HLS array_partition variable=v110 complete dim=1
  #pragma HLS array_partition variable=v110 complete dim=2

  int32_t acc4;	// L139
  acc4 = 0;	// L140
  l_S_k_0_k4: for (int k4 = 0; k4 < 3; k4++) {	// L141
  #pragma HLS pipeline II=1
    int8_t v119 = v116.read();	// L142
    int8_t a4;	// L143
    a4 = v119;	// L144
    int8_t v121 = v114.read();	// L145
    int8_t b4;	// L146
    b4 = v121;	// L147
    int8_t v123 = a4;	// L148
    int8_t v124 = b4;	// L149
    int16_t v125 = v123;	// L150
    int16_t v126 = v124;	// L151
    int16_t v127 = v125 * v126;	// L152
    int32_t v128 = acc4;	// L153
    ap_int<33> v129 = v128;	// L154
    ap_int<33> v130 = v127;	// L155
    ap_int<33> v131 = v129 + v130;	// L156
    int32_t v132 = v131;	// L157
    acc4 = v132;	// L158
    int8_t v133 = a4;	// L159
    v113.write(v133);	// L160
    int8_t v134 = b4;	// L161
    v115.write(v134);	// L162
  }
  int32_t v135 = acc4;	// L164
  v110[v111][v112] = v135;	// L165
}

void pe_r5(
  int32_t v136[3][3],
  int v137,
  int v138,
  hls::stream< int8_t >& v139,
  hls::stream< int8_t >& v140,
  hls::stream< int8_t >& v141,
  hls::stream< int8_t >& v142
) {	// L168
  #pragma HLS array_partition variable=v136 complete dim=1
  #pragma HLS array_partition variable=v136 complete dim=2

  int32_t acc5;	// L170
  acc5 = 0;	// L171
  l_S_k_0_k5: for (int k5 = 0; k5 < 3; k5++) {	// L172
  #pragma HLS pipeline II=1
    int8_t v145 = v142.read();	// L173
    int8_t a5;	// L174
    a5 = v145;	// L175
    int8_t v147 = v140.read();	// L176
    int8_t b5;	// L177
    b5 = v147;	// L178
    int8_t v149 = a5;	// L179
    int8_t v150 = b5;	// L180
    int16_t v151 = v149;	// L181
    int16_t v152 = v150;	// L182
    int16_t v153 = v151 * v152;	// L183
    int32_t v154 = acc5;	// L184
    ap_int<33> v155 = v154;	// L185
    ap_int<33> v156 = v153;	// L186
    ap_int<33> v157 = v155 + v156;	// L187
    int32_t v158 = v157;	// L188
    acc5 = v158;	// L189
    int8_t v159 = a5;	// L190
    v139.write(v159);	// L191
    int8_t v160 = b5;	// L192
    v141.write(v160);	// L193
  }
  int32_t v161 = acc5;	// L195
  v136[v137][v138] = v161;	// L196
}

void pe_r6(
  int32_t v162[3][3],
  int v163,
  int v164,
  hls::stream< int8_t >& v165,
  hls::stream< int8_t >& v166,
  hls::stream< int8_t >& v167
) {	// L199
  #pragma HLS array_partition variable=v162 complete dim=1
  #pragma HLS array_partition variable=v162 complete dim=2

  int32_t acc6;	// L201
  acc6 = 0;	// L202
  l_S_k_0_k6: for (int k6 = 0; k6 < 3; k6++) {	// L203
  #pragma HLS pipeline II=1
    int8_t v170 = v167.read();	// L204
    int8_t a6;	// L205
    a6 = v170;	// L206
    int8_t v172 = v165.read();	// L207
    int8_t b6;	// L208
    b6 = v172;	// L209
    int8_t v174 = a6;	// L210
    int8_t v175 = b6;	// L211
    int16_t v176 = v174;	// L212
    int16_t v177 = v175;	// L213
    int16_t v178 = v176 * v177;	// L214
    int32_t v179 = acc6;	// L215
    ap_int<33> v180 = v179;	// L216
    ap_int<33> v181 = v178;	// L217
    ap_int<33> v182 = v180 + v181;	// L218
    int32_t v183 = v182;	// L219
    acc6 = v183;	// L220
    int8_t v184 = b6;	// L221
    v166.write(v184);	// L222
  }
  int32_t v185 = acc6;	// L224
  v162[v163][v164] = v185;	// L225
}

void pe_r7(
  int32_t v186[3][3],
  int v187,
  int v188,
  hls::stream< int8_t >& v189,
  hls::stream< int8_t >& v190
) {	// L228
  #pragma HLS array_partition variable=v186 complete dim=1
  #pragma HLS array_partition variable=v186 complete dim=2

  int32_t acc7;	// L230
  acc7 = 0;	// L231
  l_S_k_0_k7: for (int k7 = 0; k7 < 3; k7++) {	// L232
  #pragma HLS pipeline II=1
    int8_t v193 = v190.read();	// L233
    int8_t a7;	// L234
    a7 = v193;	// L235
    int8_t v195 = v189.read();	// L236
    int8_t b7;	// L237
    b7 = v195;	// L238
    int8_t v197 = a7;	// L239
    int8_t v198 = b7;	// L240
    int16_t v199 = v197;	// L241
    int16_t v200 = v198;	// L242
    int16_t v201 = v199 * v200;	// L243
    int32_t v202 = acc7;	// L244
    ap_int<33> v203 = v202;	// L245
    ap_int<33> v204 = v201;	// L246
    ap_int<33> v205 = v203 + v204;	// L247
    int32_t v206 = v205;	// L248
    acc7 = v206;	// L249
  }
  int32_t v207 = acc7;	// L251
  v186[v187][v188] = v207;	// L252
}

void pe_r8(
  int32_t v208[3][3],
  int v209,
  int v210,
  hls::stream< int8_t >& v211,
  hls::stream< int8_t >& v212,
  hls::stream< int8_t >& v213
) {	// L255
  #pragma HLS array_partition variable=v208 complete dim=1
  #pragma HLS array_partition variable=v208 complete dim=2

  int32_t acc8;	// L257
  acc8 = 0;	// L258
  l_S_k_0_k8: for (int k8 = 0; k8 < 3; k8++) {	// L259
  #pragma HLS pipeline II=1
    int8_t v216 = v213.read();	// L260
    int8_t a8;	// L261
    a8 = v216;	// L262
    int8_t v218 = v211.read();	// L263
    int8_t b8;	// L264
    b8 = v218;	// L265
    int8_t v220 = a8;	// L266
    int8_t v221 = b8;	// L267
    int16_t v222 = v220;	// L268
    int16_t v223 = v221;	// L269
    int16_t v224 = v222 * v223;	// L270
    int32_t v225 = acc8;	// L271
    ap_int<33> v226 = v225;	// L272
    ap_int<33> v227 = v224;	// L273
    ap_int<33> v228 = v226 + v227;	// L274
    int32_t v229 = v228;	// L275
    acc8 = v229;	// L276
    int8_t v230 = b8;	// L277
    v212.write(v230);	// L278
  }
  int32_t v231 = acc8;	// L280
  v208[v209][v210] = v231;	// L281
}

/// This is top function.
void top(
  int8_t v232[3][3],
  int8_t v233[3][3],
  int32_t v234[3][3]
) {	// L284
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v232 complete dim=1
  #pragma HLS array_partition variable=v232 complete dim=2

  #pragma HLS array_partition variable=v233 complete dim=1
  #pragma HLS array_partition variable=v233 complete dim=2

  #pragma HLS array_partition variable=v234 complete dim=1
  #pragma HLS array_partition variable=v234 complete dim=2

  hls::stream< int8_t > v235;
  #pragma HLS stream variable=v235 depth=2	// L285
  hls::stream< int8_t > v236;
  #pragma HLS stream variable=v236 depth=2	// L286
  hls::stream< int8_t > v237;
  #pragma HLS stream variable=v237 depth=2	// L287
  hls::stream< int8_t > v238;
  #pragma HLS stream variable=v238 depth=2	// L288
  hls::stream< int8_t > v239;
  #pragma HLS stream variable=v239 depth=2	// L289
  hls::stream< int8_t > v240;
  #pragma HLS stream variable=v240 depth=2	// L290
  hls::stream< int8_t > v241;
  #pragma HLS stream variable=v241 depth=2	// L291
  hls::stream< int8_t > v242;
  #pragma HLS stream variable=v242 depth=2	// L292
  hls::stream< int8_t > v243;
  #pragma HLS stream variable=v243 depth=2	// L293
  hls::stream< int8_t > v244;
  #pragma HLS stream variable=v244 depth=2	// L294
  hls::stream< int8_t > v245;
  #pragma HLS stream variable=v245 depth=2	// L295
  hls::stream< int8_t > v246;
  #pragma HLS stream variable=v246 depth=2	// L296
  hls::stream< int8_t > v247;
  #pragma HLS stream variable=v247 depth=3	// L297
  hls::stream< int8_t > v248;
  #pragma HLS stream variable=v248 depth=3	// L298
  hls::stream< int8_t > v249;
  #pragma HLS stream variable=v249 depth=3	// L299
  hls::stream< int8_t > v250;
  #pragma HLS stream variable=v250 depth=3	// L300
  hls::stream< int8_t > v251;
  #pragma HLS stream variable=v251 depth=3	// L302
  hls::stream< int8_t > v252;
  #pragma HLS stream variable=v252 depth=3	// L304
  pe_west_load(v232, 0, v252);	// L306
  pe_west_load(v232, 1, v251);	// L307
  pe_west_load(v232, 2, v250);	// L308
  pe_north_load(v233, 0, v249);	// L309
  pe_north_load(v233, 1, v248);	// L310
  pe_north_load(v233, 2, v247);	// L311
  pe_r4(v234, 0, 0, v246, v249, v245, v252);	// L312
  pe_r5(v234, 0, 1, v244, v248, v243, v246);	// L313
  pe_r8(v234, 0, 2, v247, v242, v244);	// L314
  pe_r1(v234, 1, 0, v241, v245, v240, v251);	// L315
  pe_r2(v234, 1, 1, v239, v243, v238, v241);	// L316
  pe_r6(v234, 1, 2, v242, v237, v239);	// L317
  pe_r0(v234, 2, 0, v236, v240, v250);	// L318
  pe_r3(v234, 2, 1, v235, v238, v236);	// L319
  pe_r7(v234, 2, 2, v237, v235);	// L320
}


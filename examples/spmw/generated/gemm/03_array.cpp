
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
  float v0[3][3],
  hls::stream< float >& v1
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 3; _t++) {	// L4
    float v3 = v0[0][_t];	// L5
    v1.write(v3);	// L6
  }
}

void pe_west_load_1(
  float v4[3][3],
  hls::stream< float >& v5
) {	// L10
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 3; _t1++) {	// L11
    float v7 = v4[1][_t1];	// L12
    v5.write(v7);	// L13
  }
}

void pe_west_load_2(
  float v8[3][3],
  hls::stream< float >& v9
) {	// L17
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 3; _t2++) {	// L18
    float v11 = v8[2][_t2];	// L19
    v9.write(v11);	// L20
  }
}

void pe_north_load_0(
  float v12[3][3],
  hls::stream< float >& v13
) {	// L24
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 3; _t3++) {	// L25
    float v15 = v12[_t3][0];	// L26
    v13.write(v15);	// L27
  }
}

void pe_north_load_1(
  float v16[3][3],
  hls::stream< float >& v17
) {	// L31
  #pragma HLS array_partition variable=v16 complete dim=1
  #pragma HLS array_partition variable=v16 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 3; _t4++) {	// L32
    float v19 = v16[_t4][1];	// L33
    v17.write(v19);	// L34
  }
}

void pe_north_load_2(
  float v20[3][3],
  hls::stream< float >& v21
) {	// L38
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 3; _t5++) {	// L39
    float v23 = v20[_t5][2];	// L40
    v21.write(v23);	// L41
  }
}

void pe_0_0(
  float v24[3][3],
  hls::stream< float >& v25,
  hls::stream< float >& v26,
  hls::stream< float >& v27,
  hls::stream< float >& v28
) {	// L45
  #pragma HLS array_partition variable=v24 complete dim=1
  #pragma HLS array_partition variable=v24 complete dim=2

  float acc;	// L47
  acc = (float)0.000000;	// L48
  l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L49
    float v31 = v25.read();	// L50
    float a;	// L51
    a = v31;	// L52
    float v33 = v26.read();	// L53
    float b;	// L54
    b = v33;	// L55
    float v35 = a;	// L56
    float v36 = b;	// L57
    float v37 = v35 * v36;	// L58
    float v38 = acc;	// L59
    float v39 = v38 + v37;	// L60
    acc = v39;	// L61
    float v40 = a;	// L62
    v27.write(v40);	// L63
    float v41 = b;	// L64
    v28.write(v41);	// L65
  }
  float v42 = acc;	// L67
  v24[0][0] = v42;	// L68
}

void pe_0_1(
  float v43[3][3],
  hls::stream< float >& v44,
  hls::stream< float >& v45,
  hls::stream< float >& v46,
  hls::stream< float >& v47
) {	// L71
  #pragma HLS array_partition variable=v43 complete dim=1
  #pragma HLS array_partition variable=v43 complete dim=2

  float acc1;	// L73
  acc1 = (float)0.000000;	// L74
  l_S_k_0_k1: for (int k1 = 0; k1 < 3; k1++) {	// L75
    float v50 = v44.read();	// L76
    float a1;	// L77
    a1 = v50;	// L78
    float v52 = v45.read();	// L79
    float b1;	// L80
    b1 = v52;	// L81
    float v54 = a1;	// L82
    float v55 = b1;	// L83
    float v56 = v54 * v55;	// L84
    float v57 = acc1;	// L85
    float v58 = v57 + v56;	// L86
    acc1 = v58;	// L87
    float v59 = a1;	// L88
    v46.write(v59);	// L89
    float v60 = b1;	// L90
    v47.write(v60);	// L91
  }
  float v61 = acc1;	// L93
  v43[0][1] = v61;	// L94
}

void pe_0_2(
  float v62[3][3],
  hls::stream< float >& v63,
  hls::stream< float >& v64,
  hls::stream< float >& v65
) {	// L97
  #pragma HLS array_partition variable=v62 complete dim=1
  #pragma HLS array_partition variable=v62 complete dim=2

  float acc2;	// L99
  acc2 = (float)0.000000;	// L100
  l_S_k_0_k2: for (int k2 = 0; k2 < 3; k2++) {	// L101
    float v68 = v63.read();	// L102
    float a2;	// L103
    a2 = v68;	// L104
    float v70 = v64.read();	// L105
    float b2;	// L106
    b2 = v70;	// L107
    float v72 = a2;	// L108
    float v73 = b2;	// L109
    float v74 = v72 * v73;	// L110
    float v75 = acc2;	// L111
    float v76 = v75 + v74;	// L112
    acc2 = v76;	// L113
    float v77 = b2;	// L114
    v65.write(v77);	// L115
  }
  float v78 = acc2;	// L117
  v62[0][2] = v78;	// L118
}

void pe_1_0(
  float v79[3][3],
  hls::stream< float >& v80,
  hls::stream< float >& v81,
  hls::stream< float >& v82,
  hls::stream< float >& v83
) {	// L121
  #pragma HLS array_partition variable=v79 complete dim=1
  #pragma HLS array_partition variable=v79 complete dim=2

  float acc3;	// L123
  acc3 = (float)0.000000;	// L124
  l_S_k_0_k3: for (int k3 = 0; k3 < 3; k3++) {	// L125
    float v86 = v80.read();	// L126
    float a3;	// L127
    a3 = v86;	// L128
    float v88 = v81.read();	// L129
    float b3;	// L130
    b3 = v88;	// L131
    float v90 = a3;	// L132
    float v91 = b3;	// L133
    float v92 = v90 * v91;	// L134
    float v93 = acc3;	// L135
    float v94 = v93 + v92;	// L136
    acc3 = v94;	// L137
    float v95 = a3;	// L138
    v82.write(v95);	// L139
    float v96 = b3;	// L140
    v83.write(v96);	// L141
  }
  float v97 = acc3;	// L143
  v79[1][0] = v97;	// L144
}

void pe_1_1(
  float v98[3][3],
  hls::stream< float >& v99,
  hls::stream< float >& v100,
  hls::stream< float >& v101,
  hls::stream< float >& v102
) {	// L147
  #pragma HLS array_partition variable=v98 complete dim=1
  #pragma HLS array_partition variable=v98 complete dim=2

  float acc4;	// L149
  acc4 = (float)0.000000;	// L150
  l_S_k_0_k4: for (int k4 = 0; k4 < 3; k4++) {	// L151
    float v105 = v99.read();	// L152
    float a4;	// L153
    a4 = v105;	// L154
    float v107 = v100.read();	// L155
    float b4;	// L156
    b4 = v107;	// L157
    float v109 = a4;	// L158
    float v110 = b4;	// L159
    float v111 = v109 * v110;	// L160
    float v112 = acc4;	// L161
    float v113 = v112 + v111;	// L162
    acc4 = v113;	// L163
    float v114 = a4;	// L164
    v101.write(v114);	// L165
    float v115 = b4;	// L166
    v102.write(v115);	// L167
  }
  float v116 = acc4;	// L169
  v98[1][1] = v116;	// L170
}

void pe_1_2(
  float v117[3][3],
  hls::stream< float >& v118,
  hls::stream< float >& v119,
  hls::stream< float >& v120
) {	// L173
  #pragma HLS array_partition variable=v117 complete dim=1
  #pragma HLS array_partition variable=v117 complete dim=2

  float acc5;	// L175
  acc5 = (float)0.000000;	// L176
  l_S_k_0_k5: for (int k5 = 0; k5 < 3; k5++) {	// L177
    float v123 = v118.read();	// L178
    float a5;	// L179
    a5 = v123;	// L180
    float v125 = v119.read();	// L181
    float b5;	// L182
    b5 = v125;	// L183
    float v127 = a5;	// L184
    float v128 = b5;	// L185
    float v129 = v127 * v128;	// L186
    float v130 = acc5;	// L187
    float v131 = v130 + v129;	// L188
    acc5 = v131;	// L189
    float v132 = b5;	// L190
    v120.write(v132);	// L191
  }
  float v133 = acc5;	// L193
  v117[1][2] = v133;	// L194
}

void pe_2_0(
  float v134[3][3],
  hls::stream< float >& v135,
  hls::stream< float >& v136,
  hls::stream< float >& v137
) {	// L197
  #pragma HLS array_partition variable=v134 complete dim=1
  #pragma HLS array_partition variable=v134 complete dim=2

  float acc6;	// L199
  acc6 = (float)0.000000;	// L200
  l_S_k_0_k6: for (int k6 = 0; k6 < 3; k6++) {	// L201
    float v140 = v135.read();	// L202
    float a6;	// L203
    a6 = v140;	// L204
    float v142 = v136.read();	// L205
    float b6;	// L206
    b6 = v142;	// L207
    float v144 = a6;	// L208
    float v145 = b6;	// L209
    float v146 = v144 * v145;	// L210
    float v147 = acc6;	// L211
    float v148 = v147 + v146;	// L212
    acc6 = v148;	// L213
    float v149 = a6;	// L214
    v137.write(v149);	// L215
  }
  float v150 = acc6;	// L217
  v134[2][0] = v150;	// L218
}

void pe_2_1(
  float v151[3][3],
  hls::stream< float >& v152,
  hls::stream< float >& v153,
  hls::stream< float >& v154
) {	// L221
  #pragma HLS array_partition variable=v151 complete dim=1
  #pragma HLS array_partition variable=v151 complete dim=2

  float acc7;	// L223
  acc7 = (float)0.000000;	// L224
  l_S_k_0_k7: for (int k7 = 0; k7 < 3; k7++) {	// L225
    float v157 = v152.read();	// L226
    float a7;	// L227
    a7 = v157;	// L228
    float v159 = v153.read();	// L229
    float b7;	// L230
    b7 = v159;	// L231
    float v161 = a7;	// L232
    float v162 = b7;	// L233
    float v163 = v161 * v162;	// L234
    float v164 = acc7;	// L235
    float v165 = v164 + v163;	// L236
    acc7 = v165;	// L237
    float v166 = a7;	// L238
    v154.write(v166);	// L239
  }
  float v167 = acc7;	// L241
  v151[2][1] = v167;	// L242
}

void pe_2_2(
  float v168[3][3],
  hls::stream< float >& v169,
  hls::stream< float >& v170
) {	// L245
  #pragma HLS array_partition variable=v168 complete dim=1
  #pragma HLS array_partition variable=v168 complete dim=2

  float acc8;	// L247
  acc8 = (float)0.000000;	// L248
  l_S_k_0_k8: for (int k8 = 0; k8 < 3; k8++) {	// L249
    float v173 = v169.read();	// L250
    float a8;	// L251
    a8 = v173;	// L252
    float v175 = v170.read();	// L253
    float b8;	// L254
    b8 = v175;	// L255
    float v177 = a8;	// L256
    float v178 = b8;	// L257
    float v179 = v177 * v178;	// L258
    float v180 = acc8;	// L259
    float v181 = v180 + v179;	// L260
    acc8 = v181;	// L261
  }
  float v182 = acc8;	// L263
  v168[2][2] = v182;	// L264
}

/// This is top function.
void top(
  float v183[3][3],
  float v184[3][3],
  float v185[3][3]
) {	// L267
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v183 complete dim=1
  #pragma HLS array_partition variable=v183 complete dim=2

  #pragma HLS array_partition variable=v184 complete dim=1
  #pragma HLS array_partition variable=v184 complete dim=2

  #pragma HLS array_partition variable=v185 complete dim=1
  #pragma HLS array_partition variable=v185 complete dim=2

  hls::stream< float > v186;
  #pragma HLS stream variable=v186 depth=2	// L268
  hls::stream< float > v187;
  #pragma HLS stream variable=v187 depth=2	// L269
  hls::stream< float > v188;
  #pragma HLS stream variable=v188 depth=2	// L270
  hls::stream< float > v189;
  #pragma HLS stream variable=v189 depth=2	// L271
  hls::stream< float > v190;
  #pragma HLS stream variable=v190 depth=2	// L272
  hls::stream< float > v191;
  #pragma HLS stream variable=v191 depth=2	// L273
  hls::stream< float > v192;
  #pragma HLS stream variable=v192 depth=2	// L274
  hls::stream< float > v193;
  #pragma HLS stream variable=v193 depth=2	// L275
  hls::stream< float > v194;
  #pragma HLS stream variable=v194 depth=2	// L276
  hls::stream< float > v195;
  #pragma HLS stream variable=v195 depth=2	// L277
  hls::stream< float > v196;
  #pragma HLS stream variable=v196 depth=2	// L278
  hls::stream< float > v197;
  #pragma HLS stream variable=v197 depth=2	// L279
  hls::stream< float > v198;
  #pragma HLS stream variable=v198 depth=2	// L280
  hls::stream< float > v199;
  #pragma HLS stream variable=v199 depth=2	// L281
  hls::stream< float > v200;
  #pragma HLS stream variable=v200 depth=2	// L282
  hls::stream< float > v201;
  #pragma HLS stream variable=v201 depth=2	// L283
  hls::stream< float > v202;
  #pragma HLS stream variable=v202 depth=2	// L284
  hls::stream< float > v203;
  #pragma HLS stream variable=v203 depth=2	// L285
  hls::stream< float > v204;
  #pragma HLS stream variable=v204 depth=2	// L286
  hls::stream< float > v205;
  #pragma HLS stream variable=v205 depth=2	// L287
  hls::stream< float > v206;
  #pragma HLS stream variable=v206 depth=2	// L288
  hls::stream< float > v207;
  #pragma HLS stream variable=v207 depth=2	// L289
  hls::stream< float > v208;
  #pragma HLS stream variable=v208 depth=2	// L290
  hls::stream< float > v209;
  #pragma HLS stream variable=v209 depth=2	// L291
  pe_west_load_0(v183, v204);	// L292
  pe_west_load_1(v183, v205);	// L293
  pe_west_load_2(v183, v206);	// L294
  pe_north_load_0(v184, v207);	// L295
  pe_north_load_1(v184, v208);	// L296
  pe_north_load_2(v184, v209);	// L297
  pe_0_0(v185, v204, v207, v187, v198);	// L298
  pe_0_1(v185, v187, v208, v188, v199);	// L299
  pe_0_2(v185, v188, v209, v200);	// L300
  pe_1_0(v185, v205, v198, v190, v201);	// L301
  pe_1_1(v185, v190, v199, v191, v202);	// L302
  pe_1_2(v185, v191, v200, v203);	// L303
  pe_2_0(v185, v206, v201, v193);	// L304
  pe_2_1(v185, v193, v202, v194);	// L305
  pe_2_2(v185, v194, v203);	// L306
}


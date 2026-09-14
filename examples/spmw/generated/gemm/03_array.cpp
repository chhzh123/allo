
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
  float v0[3][3],
  int v1,
  hls::stream< float >& v2
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 3; _t++) {	// L4
  #pragma HLS pipeline II=1
    float v4 = v0[v1][_t];	// L5
    v2.write(v4);	// L6
  }
}

void pe_north_load(
  float v5[3][3],
  int v6,
  hls::stream< float >& v7
) {	// L10
  #pragma HLS array_partition variable=v5 complete dim=1
  #pragma HLS array_partition variable=v5 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 3; _t1++) {	// L11
  #pragma HLS pipeline II=1
    float v9 = v5[_t1][v6];	// L12
    v7.write(v9);	// L13
  }
}

void pe_r0(
  float v10[3][3],
  int v11,
  int v12,
  hls::stream< float >& v13,
  hls::stream< float >& v14,
  hls::stream< float >& v15
) {	// L17
  #pragma HLS array_partition variable=v10 complete dim=1
  #pragma HLS array_partition variable=v10 complete dim=2

  float acc;	// L20
  acc = (float)0.000000;	// L21
  l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L22
  #pragma HLS pipeline II=1
    float v18 = v15.read();	// L23
    float a;	// L24
    a = v18;	// L25
    float v20 = v14.read();	// L26
    float b;	// L27
    b = v20;	// L28
    float v22 = a;	// L29
    float v23 = b;	// L30
    float v24 = v22 * v23;	// L31
    float v25 = acc;	// L32
    float v26 = v25 + v24;	// L33
    acc = v26;	// L34
    float v27 = a;	// L35
    v13.write(v27);	// L36
  }
  float v28 = acc;	// L38
  v10[v11][v12] = v28;	// L39
}

void pe_r1(
  float v29[3][3],
  int v30,
  int v31,
  hls::stream< float >& v32,
  hls::stream< float >& v33,
  hls::stream< float >& v34,
  hls::stream< float >& v35
) {	// L42
  #pragma HLS array_partition variable=v29 complete dim=1
  #pragma HLS array_partition variable=v29 complete dim=2

  float acc1;	// L45
  acc1 = (float)0.000000;	// L46
  l_S_k_0_k1: for (int k1 = 0; k1 < 3; k1++) {	// L47
  #pragma HLS pipeline II=1
    float v38 = v35.read();	// L48
    float a1;	// L49
    a1 = v38;	// L50
    float v40 = v33.read();	// L51
    float b1;	// L52
    b1 = v40;	// L53
    float v42 = a1;	// L54
    float v43 = b1;	// L55
    float v44 = v42 * v43;	// L56
    float v45 = acc1;	// L57
    float v46 = v45 + v44;	// L58
    acc1 = v46;	// L59
    float v47 = a1;	// L60
    v32.write(v47);	// L61
    float v48 = b1;	// L62
    v34.write(v48);	// L63
  }
  float v49 = acc1;	// L65
  v29[v30][v31] = v49;	// L66
}

void pe_r2(
  float v50[3][3],
  int v51,
  int v52,
  hls::stream< float >& v53,
  hls::stream< float >& v54,
  hls::stream< float >& v55,
  hls::stream< float >& v56
) {	// L69
  #pragma HLS array_partition variable=v50 complete dim=1
  #pragma HLS array_partition variable=v50 complete dim=2

  float acc2;	// L72
  acc2 = (float)0.000000;	// L73
  l_S_k_0_k2: for (int k2 = 0; k2 < 3; k2++) {	// L74
  #pragma HLS pipeline II=1
    float v59 = v56.read();	// L75
    float a2;	// L76
    a2 = v59;	// L77
    float v61 = v54.read();	// L78
    float b2;	// L79
    b2 = v61;	// L80
    float v63 = a2;	// L81
    float v64 = b2;	// L82
    float v65 = v63 * v64;	// L83
    float v66 = acc2;	// L84
    float v67 = v66 + v65;	// L85
    acc2 = v67;	// L86
    float v68 = a2;	// L87
    v53.write(v68);	// L88
    float v69 = b2;	// L89
    v55.write(v69);	// L90
  }
  float v70 = acc2;	// L92
  v50[v51][v52] = v70;	// L93
}

void pe_r3(
  float v71[3][3],
  int v72,
  int v73,
  hls::stream< float >& v74,
  hls::stream< float >& v75,
  hls::stream< float >& v76
) {	// L96
  #pragma HLS array_partition variable=v71 complete dim=1
  #pragma HLS array_partition variable=v71 complete dim=2

  float acc3;	// L99
  acc3 = (float)0.000000;	// L100
  l_S_k_0_k3: for (int k3 = 0; k3 < 3; k3++) {	// L101
  #pragma HLS pipeline II=1
    float v79 = v76.read();	// L102
    float a3;	// L103
    a3 = v79;	// L104
    float v81 = v75.read();	// L105
    float b3;	// L106
    b3 = v81;	// L107
    float v83 = a3;	// L108
    float v84 = b3;	// L109
    float v85 = v83 * v84;	// L110
    float v86 = acc3;	// L111
    float v87 = v86 + v85;	// L112
    acc3 = v87;	// L113
    float v88 = a3;	// L114
    v74.write(v88);	// L115
  }
  float v89 = acc3;	// L117
  v71[v72][v73] = v89;	// L118
}

void pe_r4(
  float v90[3][3],
  int v91,
  int v92,
  hls::stream< float >& v93,
  hls::stream< float >& v94,
  hls::stream< float >& v95,
  hls::stream< float >& v96
) {	// L121
  #pragma HLS array_partition variable=v90 complete dim=1
  #pragma HLS array_partition variable=v90 complete dim=2

  float acc4;	// L124
  acc4 = (float)0.000000;	// L125
  l_S_k_0_k4: for (int k4 = 0; k4 < 3; k4++) {	// L126
  #pragma HLS pipeline II=1
    float v99 = v96.read();	// L127
    float a4;	// L128
    a4 = v99;	// L129
    float v101 = v94.read();	// L130
    float b4;	// L131
    b4 = v101;	// L132
    float v103 = a4;	// L133
    float v104 = b4;	// L134
    float v105 = v103 * v104;	// L135
    float v106 = acc4;	// L136
    float v107 = v106 + v105;	// L137
    acc4 = v107;	// L138
    float v108 = a4;	// L139
    v93.write(v108);	// L140
    float v109 = b4;	// L141
    v95.write(v109);	// L142
  }
  float v110 = acc4;	// L144
  v90[v91][v92] = v110;	// L145
}

void pe_r5(
  float v111[3][3],
  int v112,
  int v113,
  hls::stream< float >& v114,
  hls::stream< float >& v115,
  hls::stream< float >& v116,
  hls::stream< float >& v117
) {	// L148
  #pragma HLS array_partition variable=v111 complete dim=1
  #pragma HLS array_partition variable=v111 complete dim=2

  float acc5;	// L151
  acc5 = (float)0.000000;	// L152
  l_S_k_0_k5: for (int k5 = 0; k5 < 3; k5++) {	// L153
  #pragma HLS pipeline II=1
    float v120 = v117.read();	// L154
    float a5;	// L155
    a5 = v120;	// L156
    float v122 = v115.read();	// L157
    float b5;	// L158
    b5 = v122;	// L159
    float v124 = a5;	// L160
    float v125 = b5;	// L161
    float v126 = v124 * v125;	// L162
    float v127 = acc5;	// L163
    float v128 = v127 + v126;	// L164
    acc5 = v128;	// L165
    float v129 = a5;	// L166
    v114.write(v129);	// L167
    float v130 = b5;	// L168
    v116.write(v130);	// L169
  }
  float v131 = acc5;	// L171
  v111[v112][v113] = v131;	// L172
}

void pe_r6(
  float v132[3][3],
  int v133,
  int v134,
  hls::stream< float >& v135,
  hls::stream< float >& v136,
  hls::stream< float >& v137
) {	// L175
  #pragma HLS array_partition variable=v132 complete dim=1
  #pragma HLS array_partition variable=v132 complete dim=2

  float acc6;	// L178
  acc6 = (float)0.000000;	// L179
  l_S_k_0_k6: for (int k6 = 0; k6 < 3; k6++) {	// L180
  #pragma HLS pipeline II=1
    float v140 = v137.read();	// L181
    float a6;	// L182
    a6 = v140;	// L183
    float v142 = v135.read();	// L184
    float b6;	// L185
    b6 = v142;	// L186
    float v144 = a6;	// L187
    float v145 = b6;	// L188
    float v146 = v144 * v145;	// L189
    float v147 = acc6;	// L190
    float v148 = v147 + v146;	// L191
    acc6 = v148;	// L192
    float v149 = b6;	// L193
    v136.write(v149);	// L194
  }
  float v150 = acc6;	// L196
  v132[v133][v134] = v150;	// L197
}

void pe_r7(
  float v151[3][3],
  int v152,
  int v153,
  hls::stream< float >& v154,
  hls::stream< float >& v155
) {	// L200
  #pragma HLS array_partition variable=v151 complete dim=1
  #pragma HLS array_partition variable=v151 complete dim=2

  float acc7;	// L203
  acc7 = (float)0.000000;	// L204
  l_S_k_0_k7: for (int k7 = 0; k7 < 3; k7++) {	// L205
  #pragma HLS pipeline II=1
    float v158 = v155.read();	// L206
    float a7;	// L207
    a7 = v158;	// L208
    float v160 = v154.read();	// L209
    float b7;	// L210
    b7 = v160;	// L211
    float v162 = a7;	// L212
    float v163 = b7;	// L213
    float v164 = v162 * v163;	// L214
    float v165 = acc7;	// L215
    float v166 = v165 + v164;	// L216
    acc7 = v166;	// L217
  }
  float v167 = acc7;	// L219
  v151[v152][v153] = v167;	// L220
}

void pe_r8(
  float v168[3][3],
  int v169,
  int v170,
  hls::stream< float >& v171,
  hls::stream< float >& v172,
  hls::stream< float >& v173
) {	// L223
  #pragma HLS array_partition variable=v168 complete dim=1
  #pragma HLS array_partition variable=v168 complete dim=2

  float acc8;	// L226
  acc8 = (float)0.000000;	// L227
  l_S_k_0_k8: for (int k8 = 0; k8 < 3; k8++) {	// L228
  #pragma HLS pipeline II=1
    float v176 = v173.read();	// L229
    float a8;	// L230
    a8 = v176;	// L231
    float v178 = v171.read();	// L232
    float b8;	// L233
    b8 = v178;	// L234
    float v180 = a8;	// L235
    float v181 = b8;	// L236
    float v182 = v180 * v181;	// L237
    float v183 = acc8;	// L238
    float v184 = v183 + v182;	// L239
    acc8 = v184;	// L240
    float v185 = b8;	// L241
    v172.write(v185);	// L242
  }
  float v186 = acc8;	// L244
  v168[v169][v170] = v186;	// L245
}

/// This is top function.
void top(
  float v187[3][3],
  float v188[3][3],
  float v189[3][3]
) {	// L248
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v187 complete dim=1
  #pragma HLS array_partition variable=v187 complete dim=2

  #pragma HLS array_partition variable=v188 complete dim=1
  #pragma HLS array_partition variable=v188 complete dim=2

  #pragma HLS array_partition variable=v189 complete dim=1
  #pragma HLS array_partition variable=v189 complete dim=2

  hls::stream< float > v190;
  #pragma HLS stream variable=v190 depth=2	// L249
  hls::stream< float > v191;
  #pragma HLS stream variable=v191 depth=2	// L250
  hls::stream< float > v192;
  #pragma HLS stream variable=v192 depth=2	// L251
  hls::stream< float > v193;
  #pragma HLS stream variable=v193 depth=2	// L252
  hls::stream< float > v194;
  #pragma HLS stream variable=v194 depth=2	// L253
  hls::stream< float > v195;
  #pragma HLS stream variable=v195 depth=2	// L254
  hls::stream< float > v196;
  #pragma HLS stream variable=v196 depth=2	// L255
  hls::stream< float > v197;
  #pragma HLS stream variable=v197 depth=2	// L256
  hls::stream< float > v198;
  #pragma HLS stream variable=v198 depth=2	// L257
  hls::stream< float > v199;
  #pragma HLS stream variable=v199 depth=2	// L258
  hls::stream< float > v200;
  #pragma HLS stream variable=v200 depth=2	// L259
  hls::stream< float > v201;
  #pragma HLS stream variable=v201 depth=2	// L260
  hls::stream< float > v202;
  #pragma HLS stream variable=v202 depth=3	// L261
  hls::stream< float > v203;
  #pragma HLS stream variable=v203 depth=3	// L262
  hls::stream< float > v204;
  #pragma HLS stream variable=v204 depth=3	// L263
  hls::stream< float > v205;
  #pragma HLS stream variable=v205 depth=3	// L264
  hls::stream< float > v206;
  #pragma HLS stream variable=v206 depth=3	// L266
  hls::stream< float > v207;
  #pragma HLS stream variable=v207 depth=3	// L268
  pe_west_load(v187, 0, v207);	// L270
  pe_west_load(v187, 1, v206);	// L271
  pe_west_load(v187, 2, v205);	// L272
  pe_north_load(v188, 0, v204);	// L273
  pe_north_load(v188, 1, v203);	// L274
  pe_north_load(v188, 2, v202);	// L275
  pe_r4(v189, 0, 0, v201, v204, v200, v207);	// L276
  pe_r5(v189, 0, 1, v199, v203, v198, v201);	// L277
  pe_r8(v189, 0, 2, v202, v197, v199);	// L278
  pe_r1(v189, 1, 0, v196, v200, v195, v206);	// L279
  pe_r2(v189, 1, 1, v194, v198, v193, v196);	// L280
  pe_r6(v189, 1, 2, v197, v192, v194);	// L281
  pe_r0(v189, 2, 0, v191, v195, v205);	// L282
  pe_r3(v189, 2, 1, v190, v193, v191);	// L283
  pe_r7(v189, 2, 2, v192, v190);	// L284
}


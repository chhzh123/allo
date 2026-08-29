
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
void tiled_mac_a_in_load_0(
  int8_t v0[12][4],
  hls::stream< int8_t >& v1
) {	// L5
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 12; _t++) {	// L6
  #pragma HLS pipeline II=1
    int8_t v3 = v0[_t][0];	// L7
    v1.write(v3);	// L8
  }
}

void tiled_mac_a_in_load_1(
  int8_t v4[12][4],
  hls::stream< int8_t >& v5
) {	// L12
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 12; _t1++) {	// L13
  #pragma HLS pipeline II=1
    int8_t v7 = v4[_t1][1];	// L14
    v5.write(v7);	// L15
  }
}

void tiled_mac_a_in_load_2(
  int8_t v8[12][4],
  hls::stream< int8_t >& v9
) {	// L19
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 12; _t2++) {	// L20
  #pragma HLS pipeline II=1
    int8_t v11 = v8[_t2][2];	// L21
    v9.write(v11);	// L22
  }
}

void tiled_mac_a_in_load_3(
  int8_t v12[12][4],
  hls::stream< int8_t >& v13
) {	// L26
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 12; _t3++) {	// L27
  #pragma HLS pipeline II=1
    int8_t v15 = v12[_t3][3];	// L28
    v13.write(v15);	// L29
  }
}

void tiled_vpu_op_in_load_0(
  int32_t v16[12],
  hls::stream< int32_t >& v17
) {	// L33
  #pragma HLS array_partition variable=v16 complete dim=1

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 12; _t4++) {	// L34
  #pragma HLS pipeline II=1
    int32_t v19 = v16[_t4];	// L35
    v17.write(v19);	// L36
  }
}

void tiled_mac_0_0(
  int8_t v20[4][4][2],
  hls::stream< int8_t >& v21,
  hls::stream< int32_t >& v22,
  hls::stream< int8_t >& v23
) {	// L40
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2
  #pragma HLS array_partition variable=v20 complete dim=3

  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L42
    l_S_t_0_t: for (int t = 0; t < 2; t++) {	// L43
    #pragma HLS pipeline II=1
      int8_t v26 = v21.read();	// L44
      int8_t a;	// L45
      a = v26;	// L46
      int32_t p;	// L47
      p = 0;	// L48
      int8_t v29 = v20[0][0][t];	// L49
      int32_t v30 = v29;	// L50
      int32_t wt;	// L51
      wt = v30;	// L52
      int32_t v32 = p;	// L53
      int8_t v33 = a;	// L54
      int32_t v34 = wt;	// L55
      ap_int<40> v35 = v33;	// L56
      ap_int<40> v36 = v34;	// L57
      ap_int<40> v37 = v35 * v36;	// L58
      ap_int<41> v38 = v32;	// L59
      ap_int<41> v39 = v37;	// L60
      ap_int<41> v40 = v38 + v39;	// L61
      v22.write(v40);	// L62
      int8_t v41 = a;	// L63
      v23.write(v41);	// L64
    }
  }
}

void tiled_mac_0_1(
  int8_t v42[4][4][2],
  hls::stream< int8_t >& v43,
  hls::stream< int32_t >& v44,
  hls::stream< int8_t >& v45
) {	// L69
  #pragma HLS array_partition variable=v42 complete dim=1
  #pragma HLS array_partition variable=v42 complete dim=2
  #pragma HLS array_partition variable=v42 complete dim=3

  l_S_m_0_m1: for (int m1 = 0; m1 < 6; m1++) {	// L71
    l_S_t_0_t1: for (int t1 = 0; t1 < 2; t1++) {	// L72
    #pragma HLS pipeline II=1
      int8_t v48 = v43.read();	// L73
      int8_t a1;	// L74
      a1 = v48;	// L75
      int32_t p1;	// L76
      p1 = 0;	// L77
      int8_t v51 = v42[0][1][t1];	// L78
      int32_t v52 = v51;	// L79
      int32_t wt1;	// L80
      wt1 = v52;	// L81
      int32_t v54 = p1;	// L82
      int8_t v55 = a1;	// L83
      int32_t v56 = wt1;	// L84
      ap_int<40> v57 = v55;	// L85
      ap_int<40> v58 = v56;	// L86
      ap_int<40> v59 = v57 * v58;	// L87
      ap_int<41> v60 = v54;	// L88
      ap_int<41> v61 = v59;	// L89
      ap_int<41> v62 = v60 + v61;	// L90
      v44.write(v62);	// L91
      int8_t v63 = a1;	// L92
      v45.write(v63);	// L93
    }
  }
}

void tiled_mac_0_2(
  int8_t v64[4][4][2],
  hls::stream< int8_t >& v65,
  hls::stream< int32_t >& v66,
  hls::stream< int8_t >& v67
) {	// L98
  #pragma HLS array_partition variable=v64 complete dim=1
  #pragma HLS array_partition variable=v64 complete dim=2
  #pragma HLS array_partition variable=v64 complete dim=3

  l_S_m_0_m2: for (int m2 = 0; m2 < 6; m2++) {	// L100
    l_S_t_0_t2: for (int t2 = 0; t2 < 2; t2++) {	// L101
    #pragma HLS pipeline II=1
      int8_t v70 = v65.read();	// L102
      int8_t a2;	// L103
      a2 = v70;	// L104
      int32_t p2;	// L105
      p2 = 0;	// L106
      int8_t v73 = v64[0][2][t2];	// L107
      int32_t v74 = v73;	// L108
      int32_t wt2;	// L109
      wt2 = v74;	// L110
      int32_t v76 = p2;	// L111
      int8_t v77 = a2;	// L112
      int32_t v78 = wt2;	// L113
      ap_int<40> v79 = v77;	// L114
      ap_int<40> v80 = v78;	// L115
      ap_int<40> v81 = v79 * v80;	// L116
      ap_int<41> v82 = v76;	// L117
      ap_int<41> v83 = v81;	// L118
      ap_int<41> v84 = v82 + v83;	// L119
      v66.write(v84);	// L120
      int8_t v85 = a2;	// L121
      v67.write(v85);	// L122
    }
  }
}

void tiled_mac_0_3(
  int8_t v86[4][4][2],
  hls::stream< int8_t >& v87,
  hls::stream< int32_t >& v88
) {	// L127
  #pragma HLS array_partition variable=v86 complete dim=1
  #pragma HLS array_partition variable=v86 complete dim=2
  #pragma HLS array_partition variable=v86 complete dim=3

  l_S_m_0_m3: for (int m3 = 0; m3 < 6; m3++) {	// L129
    l_S_t_0_t3: for (int t3 = 0; t3 < 2; t3++) {	// L130
    #pragma HLS pipeline II=1
      int8_t v91 = v87.read();	// L131
      int8_t a3;	// L132
      a3 = v91;	// L133
      int32_t p3;	// L134
      p3 = 0;	// L135
      int8_t v94 = v86[0][3][t3];	// L136
      int32_t v95 = v94;	// L137
      int32_t wt3;	// L138
      wt3 = v95;	// L139
      int32_t v97 = p3;	// L140
      int8_t v98 = a3;	// L141
      int32_t v99 = wt3;	// L142
      ap_int<40> v100 = v98;	// L143
      ap_int<40> v101 = v99;	// L144
      ap_int<40> v102 = v100 * v101;	// L145
      ap_int<41> v103 = v97;	// L146
      ap_int<41> v104 = v102;	// L147
      ap_int<41> v105 = v103 + v104;	// L148
      v88.write(v105);	// L149
    }
  }
}

void tiled_mac_1_0(
  int8_t v106[4][4][2],
  hls::stream< int8_t >& v107,
  hls::stream< int32_t >& v108,
  hls::stream< int32_t >& v109,
  hls::stream< int8_t >& v110
) {	// L154
  #pragma HLS array_partition variable=v106 complete dim=1
  #pragma HLS array_partition variable=v106 complete dim=2
  #pragma HLS array_partition variable=v106 complete dim=3

  l_S_m_0_m4: for (int m4 = 0; m4 < 6; m4++) {	// L155
    l_S_t_0_t4: for (int t4 = 0; t4 < 2; t4++) {	// L156
    #pragma HLS pipeline II=1
      int8_t v113 = v107.read();	// L157
      int8_t a4;	// L158
      a4 = v113;	// L159
      int32_t v115 = v108.read();	// L160
      int32_t p4;	// L161
      p4 = v115;	// L162
      int8_t v117 = v106[1][0][t4];	// L163
      int32_t v118 = v117;	// L164
      int32_t wt4;	// L165
      wt4 = v118;	// L166
      int32_t v120 = p4;	// L167
      int8_t v121 = a4;	// L168
      int32_t v122 = wt4;	// L169
      ap_int<40> v123 = v121;	// L170
      ap_int<40> v124 = v122;	// L171
      ap_int<40> v125 = v123 * v124;	// L172
      ap_int<41> v126 = v120;	// L173
      ap_int<41> v127 = v125;	// L174
      ap_int<41> v128 = v126 + v127;	// L175
      v109.write(v128);	// L176
      int8_t v129 = a4;	// L177
      v110.write(v129);	// L178
    }
  }
}

void tiled_mac_1_1(
  int8_t v130[4][4][2],
  hls::stream< int8_t >& v131,
  hls::stream< int32_t >& v132,
  hls::stream< int32_t >& v133,
  hls::stream< int8_t >& v134
) {	// L183
  #pragma HLS array_partition variable=v130 complete dim=1
  #pragma HLS array_partition variable=v130 complete dim=2
  #pragma HLS array_partition variable=v130 complete dim=3

  l_S_m_0_m5: for (int m5 = 0; m5 < 6; m5++) {	// L184
    l_S_t_0_t5: for (int t5 = 0; t5 < 2; t5++) {	// L185
    #pragma HLS pipeline II=1
      int8_t v137 = v131.read();	// L186
      int8_t a5;	// L187
      a5 = v137;	// L188
      int32_t v139 = v132.read();	// L189
      int32_t p5;	// L190
      p5 = v139;	// L191
      int8_t v141 = v130[1][1][t5];	// L192
      int32_t v142 = v141;	// L193
      int32_t wt5;	// L194
      wt5 = v142;	// L195
      int32_t v144 = p5;	// L196
      int8_t v145 = a5;	// L197
      int32_t v146 = wt5;	// L198
      ap_int<40> v147 = v145;	// L199
      ap_int<40> v148 = v146;	// L200
      ap_int<40> v149 = v147 * v148;	// L201
      ap_int<41> v150 = v144;	// L202
      ap_int<41> v151 = v149;	// L203
      ap_int<41> v152 = v150 + v151;	// L204
      v133.write(v152);	// L205
      int8_t v153 = a5;	// L206
      v134.write(v153);	// L207
    }
  }
}

void tiled_mac_1_2(
  int8_t v154[4][4][2],
  hls::stream< int8_t >& v155,
  hls::stream< int32_t >& v156,
  hls::stream< int32_t >& v157,
  hls::stream< int8_t >& v158
) {	// L212
  #pragma HLS array_partition variable=v154 complete dim=1
  #pragma HLS array_partition variable=v154 complete dim=2
  #pragma HLS array_partition variable=v154 complete dim=3

  l_S_m_0_m6: for (int m6 = 0; m6 < 6; m6++) {	// L213
    l_S_t_0_t6: for (int t6 = 0; t6 < 2; t6++) {	// L214
    #pragma HLS pipeline II=1
      int8_t v161 = v155.read();	// L215
      int8_t a6;	// L216
      a6 = v161;	// L217
      int32_t v163 = v156.read();	// L218
      int32_t p6;	// L219
      p6 = v163;	// L220
      int8_t v165 = v154[1][2][t6];	// L221
      int32_t v166 = v165;	// L222
      int32_t wt6;	// L223
      wt6 = v166;	// L224
      int32_t v168 = p6;	// L225
      int8_t v169 = a6;	// L226
      int32_t v170 = wt6;	// L227
      ap_int<40> v171 = v169;	// L228
      ap_int<40> v172 = v170;	// L229
      ap_int<40> v173 = v171 * v172;	// L230
      ap_int<41> v174 = v168;	// L231
      ap_int<41> v175 = v173;	// L232
      ap_int<41> v176 = v174 + v175;	// L233
      v157.write(v176);	// L234
      int8_t v177 = a6;	// L235
      v158.write(v177);	// L236
    }
  }
}

void tiled_mac_1_3(
  int8_t v178[4][4][2],
  hls::stream< int8_t >& v179,
  hls::stream< int32_t >& v180,
  hls::stream< int32_t >& v181
) {	// L241
  #pragma HLS array_partition variable=v178 complete dim=1
  #pragma HLS array_partition variable=v178 complete dim=2
  #pragma HLS array_partition variable=v178 complete dim=3

  l_S_m_0_m7: for (int m7 = 0; m7 < 6; m7++) {	// L242
    l_S_t_0_t7: for (int t7 = 0; t7 < 2; t7++) {	// L243
    #pragma HLS pipeline II=1
      int8_t v184 = v179.read();	// L244
      int8_t a7;	// L245
      a7 = v184;	// L246
      int32_t v186 = v180.read();	// L247
      int32_t p7;	// L248
      p7 = v186;	// L249
      int8_t v188 = v178[1][3][t7];	// L250
      int32_t v189 = v188;	// L251
      int32_t wt7;	// L252
      wt7 = v189;	// L253
      int32_t v191 = p7;	// L254
      int8_t v192 = a7;	// L255
      int32_t v193 = wt7;	// L256
      ap_int<40> v194 = v192;	// L257
      ap_int<40> v195 = v193;	// L258
      ap_int<40> v196 = v194 * v195;	// L259
      ap_int<41> v197 = v191;	// L260
      ap_int<41> v198 = v196;	// L261
      ap_int<41> v199 = v197 + v198;	// L262
      v181.write(v199);	// L263
    }
  }
}

void tiled_mac_2_0(
  int8_t v200[4][4][2],
  hls::stream< int8_t >& v201,
  hls::stream< int32_t >& v202,
  hls::stream< int32_t >& v203,
  hls::stream< int8_t >& v204
) {	// L268
  #pragma HLS array_partition variable=v200 complete dim=1
  #pragma HLS array_partition variable=v200 complete dim=2
  #pragma HLS array_partition variable=v200 complete dim=3

  l_S_m_0_m8: for (int m8 = 0; m8 < 6; m8++) {	// L269
    l_S_t_0_t8: for (int t8 = 0; t8 < 2; t8++) {	// L270
    #pragma HLS pipeline II=1
      int8_t v207 = v201.read();	// L271
      int8_t a8;	// L272
      a8 = v207;	// L273
      int32_t v209 = v202.read();	// L274
      int32_t p8;	// L275
      p8 = v209;	// L276
      int8_t v211 = v200[2][0][t8];	// L277
      int32_t v212 = v211;	// L278
      int32_t wt8;	// L279
      wt8 = v212;	// L280
      int32_t v214 = p8;	// L281
      int8_t v215 = a8;	// L282
      int32_t v216 = wt8;	// L283
      ap_int<40> v217 = v215;	// L284
      ap_int<40> v218 = v216;	// L285
      ap_int<40> v219 = v217 * v218;	// L286
      ap_int<41> v220 = v214;	// L287
      ap_int<41> v221 = v219;	// L288
      ap_int<41> v222 = v220 + v221;	// L289
      v203.write(v222);	// L290
      int8_t v223 = a8;	// L291
      v204.write(v223);	// L292
    }
  }
}

void tiled_mac_2_1(
  int8_t v224[4][4][2],
  hls::stream< int8_t >& v225,
  hls::stream< int32_t >& v226,
  hls::stream< int32_t >& v227,
  hls::stream< int8_t >& v228
) {	// L297
  #pragma HLS array_partition variable=v224 complete dim=1
  #pragma HLS array_partition variable=v224 complete dim=2
  #pragma HLS array_partition variable=v224 complete dim=3

  l_S_m_0_m9: for (int m9 = 0; m9 < 6; m9++) {	// L298
    l_S_t_0_t9: for (int t9 = 0; t9 < 2; t9++) {	// L299
    #pragma HLS pipeline II=1
      int8_t v231 = v225.read();	// L300
      int8_t a9;	// L301
      a9 = v231;	// L302
      int32_t v233 = v226.read();	// L303
      int32_t p9;	// L304
      p9 = v233;	// L305
      int8_t v235 = v224[2][1][t9];	// L306
      int32_t v236 = v235;	// L307
      int32_t wt9;	// L308
      wt9 = v236;	// L309
      int32_t v238 = p9;	// L310
      int8_t v239 = a9;	// L311
      int32_t v240 = wt9;	// L312
      ap_int<40> v241 = v239;	// L313
      ap_int<40> v242 = v240;	// L314
      ap_int<40> v243 = v241 * v242;	// L315
      ap_int<41> v244 = v238;	// L316
      ap_int<41> v245 = v243;	// L317
      ap_int<41> v246 = v244 + v245;	// L318
      v227.write(v246);	// L319
      int8_t v247 = a9;	// L320
      v228.write(v247);	// L321
    }
  }
}

void tiled_mac_2_2(
  int8_t v248[4][4][2],
  hls::stream< int8_t >& v249,
  hls::stream< int32_t >& v250,
  hls::stream< int32_t >& v251,
  hls::stream< int8_t >& v252
) {	// L326
  #pragma HLS array_partition variable=v248 complete dim=1
  #pragma HLS array_partition variable=v248 complete dim=2
  #pragma HLS array_partition variable=v248 complete dim=3

  l_S_m_0_m10: for (int m10 = 0; m10 < 6; m10++) {	// L327
    l_S_t_0_t10: for (int t10 = 0; t10 < 2; t10++) {	// L328
    #pragma HLS pipeline II=1
      int8_t v255 = v249.read();	// L329
      int8_t a10;	// L330
      a10 = v255;	// L331
      int32_t v257 = v250.read();	// L332
      int32_t p10;	// L333
      p10 = v257;	// L334
      int8_t v259 = v248[2][2][t10];	// L335
      int32_t v260 = v259;	// L336
      int32_t wt10;	// L337
      wt10 = v260;	// L338
      int32_t v262 = p10;	// L339
      int8_t v263 = a10;	// L340
      int32_t v264 = wt10;	// L341
      ap_int<40> v265 = v263;	// L342
      ap_int<40> v266 = v264;	// L343
      ap_int<40> v267 = v265 * v266;	// L344
      ap_int<41> v268 = v262;	// L345
      ap_int<41> v269 = v267;	// L346
      ap_int<41> v270 = v268 + v269;	// L347
      v251.write(v270);	// L348
      int8_t v271 = a10;	// L349
      v252.write(v271);	// L350
    }
  }
}

void tiled_mac_2_3(
  int8_t v272[4][4][2],
  hls::stream< int8_t >& v273,
  hls::stream< int32_t >& v274,
  hls::stream< int32_t >& v275
) {	// L355
  #pragma HLS array_partition variable=v272 complete dim=1
  #pragma HLS array_partition variable=v272 complete dim=2
  #pragma HLS array_partition variable=v272 complete dim=3

  l_S_m_0_m11: for (int m11 = 0; m11 < 6; m11++) {	// L356
    l_S_t_0_t11: for (int t11 = 0; t11 < 2; t11++) {	// L357
    #pragma HLS pipeline II=1
      int8_t v278 = v273.read();	// L358
      int8_t a11;	// L359
      a11 = v278;	// L360
      int32_t v280 = v274.read();	// L361
      int32_t p11;	// L362
      p11 = v280;	// L363
      int8_t v282 = v272[2][3][t11];	// L364
      int32_t v283 = v282;	// L365
      int32_t wt11;	// L366
      wt11 = v283;	// L367
      int32_t v285 = p11;	// L368
      int8_t v286 = a11;	// L369
      int32_t v287 = wt11;	// L370
      ap_int<40> v288 = v286;	// L371
      ap_int<40> v289 = v287;	// L372
      ap_int<40> v290 = v288 * v289;	// L373
      ap_int<41> v291 = v285;	// L374
      ap_int<41> v292 = v290;	// L375
      ap_int<41> v293 = v291 + v292;	// L376
      v275.write(v293);	// L377
    }
  }
}

void tiled_mac_3_0(
  int8_t v294[4][4][2],
  hls::stream< int8_t >& v295,
  hls::stream< int32_t >& v296,
  hls::stream< int32_t >& v297,
  hls::stream< int8_t >& v298
) {	// L382
  #pragma HLS array_partition variable=v294 complete dim=1
  #pragma HLS array_partition variable=v294 complete dim=2
  #pragma HLS array_partition variable=v294 complete dim=3

  l_S_m_0_m12: for (int m12 = 0; m12 < 6; m12++) {	// L383
    l_S_t_0_t12: for (int t12 = 0; t12 < 2; t12++) {	// L384
    #pragma HLS pipeline II=1
      int8_t v301 = v295.read();	// L385
      int8_t a12;	// L386
      a12 = v301;	// L387
      int32_t v303 = v296.read();	// L388
      int32_t p12;	// L389
      p12 = v303;	// L390
      int8_t v305 = v294[3][0][t12];	// L391
      int32_t v306 = v305;	// L392
      int32_t wt12;	// L393
      wt12 = v306;	// L394
      int32_t v308 = p12;	// L395
      int8_t v309 = a12;	// L396
      int32_t v310 = wt12;	// L397
      ap_int<40> v311 = v309;	// L398
      ap_int<40> v312 = v310;	// L399
      ap_int<40> v313 = v311 * v312;	// L400
      ap_int<41> v314 = v308;	// L401
      ap_int<41> v315 = v313;	// L402
      ap_int<41> v316 = v314 + v315;	// L403
      v297.write(v316);	// L404
      int8_t v317 = a12;	// L405
      v298.write(v317);	// L406
    }
  }
}

void tiled_mac_3_1(
  int8_t v318[4][4][2],
  hls::stream< int8_t >& v319,
  hls::stream< int32_t >& v320,
  hls::stream< int32_t >& v321,
  hls::stream< int8_t >& v322
) {	// L411
  #pragma HLS array_partition variable=v318 complete dim=1
  #pragma HLS array_partition variable=v318 complete dim=2
  #pragma HLS array_partition variable=v318 complete dim=3

  l_S_m_0_m13: for (int m13 = 0; m13 < 6; m13++) {	// L412
    l_S_t_0_t13: for (int t13 = 0; t13 < 2; t13++) {	// L413
    #pragma HLS pipeline II=1
      int8_t v325 = v319.read();	// L414
      int8_t a13;	// L415
      a13 = v325;	// L416
      int32_t v327 = v320.read();	// L417
      int32_t p13;	// L418
      p13 = v327;	// L419
      int8_t v329 = v318[3][1][t13];	// L420
      int32_t v330 = v329;	// L421
      int32_t wt13;	// L422
      wt13 = v330;	// L423
      int32_t v332 = p13;	// L424
      int8_t v333 = a13;	// L425
      int32_t v334 = wt13;	// L426
      ap_int<40> v335 = v333;	// L427
      ap_int<40> v336 = v334;	// L428
      ap_int<40> v337 = v335 * v336;	// L429
      ap_int<41> v338 = v332;	// L430
      ap_int<41> v339 = v337;	// L431
      ap_int<41> v340 = v338 + v339;	// L432
      v321.write(v340);	// L433
      int8_t v341 = a13;	// L434
      v322.write(v341);	// L435
    }
  }
}

void tiled_mac_3_2(
  int8_t v342[4][4][2],
  hls::stream< int8_t >& v343,
  hls::stream< int32_t >& v344,
  hls::stream< int32_t >& v345,
  hls::stream< int8_t >& v346
) {	// L440
  #pragma HLS array_partition variable=v342 complete dim=1
  #pragma HLS array_partition variable=v342 complete dim=2
  #pragma HLS array_partition variable=v342 complete dim=3

  l_S_m_0_m14: for (int m14 = 0; m14 < 6; m14++) {	// L441
    l_S_t_0_t14: for (int t14 = 0; t14 < 2; t14++) {	// L442
    #pragma HLS pipeline II=1
      int8_t v349 = v343.read();	// L443
      int8_t a14;	// L444
      a14 = v349;	// L445
      int32_t v351 = v344.read();	// L446
      int32_t p14;	// L447
      p14 = v351;	// L448
      int8_t v353 = v342[3][2][t14];	// L449
      int32_t v354 = v353;	// L450
      int32_t wt14;	// L451
      wt14 = v354;	// L452
      int32_t v356 = p14;	// L453
      int8_t v357 = a14;	// L454
      int32_t v358 = wt14;	// L455
      ap_int<40> v359 = v357;	// L456
      ap_int<40> v360 = v358;	// L457
      ap_int<40> v361 = v359 * v360;	// L458
      ap_int<41> v362 = v356;	// L459
      ap_int<41> v363 = v361;	// L460
      ap_int<41> v364 = v362 + v363;	// L461
      v345.write(v364);	// L462
      int8_t v365 = a14;	// L463
      v346.write(v365);	// L464
    }
  }
}

void tiled_mac_3_3(
  int8_t v366[4][4][2],
  hls::stream< int8_t >& v367,
  hls::stream< int32_t >& v368,
  hls::stream< int32_t >& v369
) {	// L469
  #pragma HLS array_partition variable=v366 complete dim=1
  #pragma HLS array_partition variable=v366 complete dim=2
  #pragma HLS array_partition variable=v366 complete dim=3

  l_S_m_0_m15: for (int m15 = 0; m15 < 6; m15++) {	// L470
    l_S_t_0_t15: for (int t15 = 0; t15 < 2; t15++) {	// L471
    #pragma HLS pipeline II=1
      int8_t v372 = v367.read();	// L472
      int8_t a15;	// L473
      a15 = v372;	// L474
      int32_t v374 = v368.read();	// L475
      int32_t p15;	// L476
      p15 = v374;	// L477
      int8_t v376 = v366[3][3][t15];	// L478
      int32_t v377 = v376;	// L479
      int32_t wt15;	// L480
      wt15 = v377;	// L481
      int32_t v379 = p15;	// L482
      int8_t v380 = a15;	// L483
      int32_t v381 = wt15;	// L484
      ap_int<40> v382 = v380;	// L485
      ap_int<40> v383 = v381;	// L486
      ap_int<40> v384 = v382 * v383;	// L487
      ap_int<41> v385 = v379;	// L488
      ap_int<41> v386 = v384;	// L489
      ap_int<41> v387 = v385 + v386;	// L490
      v369.write(v387);	// L491
    }
  }
}

void tiled_vpu_0(
  int32_t v388[4],
  hls::stream< int32_t >& v389,
  hls::stream< int32_t >& v390,
  hls::stream< int32_t >& v391,
  hls::stream< int32_t >& v392
) {	// L496
  #pragma HLS array_partition variable=v388 complete dim=1

  int32_t prog[12];	// L512
  for (int v394 = 0; v394 < 12; v394++) {	// L513
    prog[v394] = 0;	// L513
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 12; pc++) {	// L514
  #pragma HLS pipeline II=1
    int32_t v396 = v389.read();	// L515
    int32_t word;	// L516
    word = v396;	// L517
    int32_t v398 = word;	// L518
    prog[pc] = v398;	// L519
    int32_t v399 = word;	// L520
    v390.write(v399);	// L521
  }
  l_S_m_1_m16: for (int m16 = 0; m16 < 6; m16++) {	// L523
    int32_t reg[4];	// L524
    for (int v402 = 0; v402 < 4; v402++) {	// L525
      reg[v402] = 0;	// L525
    }
    l_S_step_1_step: for (int step = 0; step < 12; step++) {	// L526
    #pragma HLS pipeline II=1
      int32_t v404 = prog[step];	// L527
      int32_t word2;	// L528
      word2 = v404;	// L529
      int32_t v406 = word2;	// L530
      int32_t v407 = v406 >> 24;	// L531
      int32_t v408 = v407 & 255;	// L532
      int32_t opcode;	// L533
      opcode = v408;	// L534
      int32_t v410 = word2;	// L535
      int32_t v411 = v410 >> 20;	// L536
      int32_t v412 = v411 & 15;	// L537
      int32_t dst;	// L538
      dst = v412;	// L539
      int32_t v414 = word2;	// L540
      int32_t v415 = v414 >> 16;	// L541
      int32_t v416 = v415 & 15;	// L542
      int32_t src;	// L543
      src = v416;	// L544
      int32_t v418 = word2;	// L545
      int32_t v419 = v418 & 65535;	// L546
      int32_t imm;	// L547
      imm = v419;	// L548
      int32_t v421 = opcode;	// L549
      bool v422 = v421 == 9;	// L550
      if (v422) {	// L551
        int32_t v423 = v391.read();	// L552
        int32_t zz;	// L553
        zz = v423;	// L554
        int32_t v425 = dst;	// L555
        int v426 = v425;	// L556
        int32_t v427 = reg[v426];	// L557
        int32_t v428 = zz;	// L558
        ap_int<33> v429 = v427;	// L559
        ap_int<33> v430 = v428;	// L560
        ap_int<33> v431 = v429 + v430;	// L561
        int32_t v432 = v431;	// L562
        reg[v426] = v432;	// L563
      } else {
        int32_t v433 = opcode;	// L565
        bool v434 = v433 == 2;	// L566
        if (v434) {	// L567
          int32_t v435 = v388[0];	// L568
          int32_t v436 = dst;	// L569
          int v437 = v436;	// L570
          reg[v437] = v435;	// L571
        } else {
          int32_t v438 = opcode;	// L573
          bool v439 = v438 == 3;	// L574
          if (v439) {	// L575
            int32_t v440 = imm;	// L576
            int32_t v441 = dst;	// L577
            int v442 = v441;	// L578
            reg[v442] = v440;	// L579
          } else {
            int32_t v443 = opcode;	// L581
            bool v444 = v443 == 4;	// L582
            if (v444) {	// L583
              int32_t v445 = dst;	// L584
              int v446 = v445;	// L585
              int32_t v447 = reg[v446];	// L586
              int32_t v448 = src;	// L587
              int v449 = v448;	// L588
              int32_t v450 = reg[v449];	// L589
              ap_int<33> v451 = v447;	// L590
              ap_int<33> v452 = v450;	// L591
              ap_int<33> v453 = v451 + v452;	// L592
              int32_t v454 = v453;	// L593
              reg[v446] = v454;	// L594
            } else {
              int32_t v455 = opcode;	// L596
              bool v456 = v455 == 5;	// L597
              if (v456) {	// L598
                int32_t v457 = dst;	// L599
                int v458 = v457;	// L600
                int32_t v459 = reg[v458];	// L601
                int32_t v460 = src;	// L602
                int v461 = v460;	// L603
                int32_t v462 = reg[v461];	// L604
                int64_t v463 = v459;	// L605
                int64_t v464 = v462;	// L606
                int64_t v465 = v463 * v464;	// L607
                int32_t v466 = v465;	// L608
                reg[v458] = v466;	// L609
              } else {
                int32_t v467 = opcode;	// L611
                bool v468 = v467 == 6;	// L612
                if (v468) {	// L613
                  int32_t v469 = src;	// L614
                  int v470 = v469;	// L615
                  int32_t v471 = reg[v470];	// L616
                  int32_t v472 = dst;	// L617
                  int v473 = v472;	// L618
                  int32_t v474 = reg[v473];	// L619
                  bool v475 = v471 > v474;	// L620
                  if (v475) {	// L621
                    int32_t v476 = src;	// L622
                    int v477 = v476;	// L623
                    int32_t v478 = reg[v477];	// L624
                    int32_t v479 = dst;	// L625
                    int v480 = v479;	// L626
                    reg[v480] = v478;	// L627
                  }
                } else {
                  int32_t v481 = opcode;	// L630
                  bool v482 = v481 == 7;	// L631
                  if (v482) {	// L632
                    int32_t v483 = dst;	// L633
                    int v484 = v483;	// L634
                    int32_t v485 = reg[v484];	// L635
                    int32_t v486 = imm;	// L636
                    int32_t v487 = v485 >> v486;	// L637
                    reg[v484] = v487;	// L638
                  } else {
                    int32_t v488 = opcode;	// L640
                    bool v489 = v488 == 8;	// L641
                    if (v489) {	// L642
                      int32_t v490 = dst;	// L643
                      int v491 = v490;	// L644
                      int32_t v492 = reg[v491];	// L645
                      v392.write(v492);	// L646
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void tiled_vpu_1(
  int32_t v493[4],
  hls::stream< int32_t >& v494,
  hls::stream< int32_t >& v495,
  hls::stream< int32_t >& v496,
  hls::stream< int32_t >& v497
) {	// L659
  #pragma HLS array_partition variable=v493 complete dim=1

  int32_t prog1[12];	// L675
  for (int v499 = 0; v499 < 12; v499++) {	// L676
    prog1[v499] = 0;	// L676
  }
  l_S_pc_0_pc1: for (int pc1 = 0; pc1 < 12; pc1++) {	// L677
  #pragma HLS pipeline II=1
    int32_t v501 = v494.read();	// L678
    int32_t word1;	// L679
    word1 = v501;	// L680
    int32_t v503 = word1;	// L681
    prog1[pc1] = v503;	// L682
    int32_t v504 = word1;	// L683
    v495.write(v504);	// L684
  }
  l_S_m_1_m17: for (int m17 = 0; m17 < 6; m17++) {	// L686
    int32_t reg1[4];	// L687
    for (int v507 = 0; v507 < 4; v507++) {	// L688
      reg1[v507] = 0;	// L688
    }
    l_S_step_1_step1: for (int step1 = 0; step1 < 12; step1++) {	// L689
    #pragma HLS pipeline II=1
      int32_t v509 = prog1[step1];	// L690
      int32_t word21;	// L691
      word21 = v509;	// L692
      int32_t v511 = word21;	// L693
      int32_t v512 = v511 >> 24;	// L694
      int32_t v513 = v512 & 255;	// L695
      int32_t opcode1;	// L696
      opcode1 = v513;	// L697
      int32_t v515 = word21;	// L698
      int32_t v516 = v515 >> 20;	// L699
      int32_t v517 = v516 & 15;	// L700
      int32_t dst1;	// L701
      dst1 = v517;	// L702
      int32_t v519 = word21;	// L703
      int32_t v520 = v519 >> 16;	// L704
      int32_t v521 = v520 & 15;	// L705
      int32_t src1;	// L706
      src1 = v521;	// L707
      int32_t v523 = word21;	// L708
      int32_t v524 = v523 & 65535;	// L709
      int32_t imm1;	// L710
      imm1 = v524;	// L711
      int32_t v526 = opcode1;	// L712
      bool v527 = v526 == 9;	// L713
      if (v527) {	// L714
        int32_t v528 = v496.read();	// L715
        int32_t zz1;	// L716
        zz1 = v528;	// L717
        int32_t v530 = dst1;	// L718
        int v531 = v530;	// L719
        int32_t v532 = reg1[v531];	// L720
        int32_t v533 = zz1;	// L721
        ap_int<33> v534 = v532;	// L722
        ap_int<33> v535 = v533;	// L723
        ap_int<33> v536 = v534 + v535;	// L724
        int32_t v537 = v536;	// L725
        reg1[v531] = v537;	// L726
      } else {
        int32_t v538 = opcode1;	// L728
        bool v539 = v538 == 2;	// L729
        if (v539) {	// L730
          int32_t v540 = v493[1];	// L731
          int32_t v541 = dst1;	// L732
          int v542 = v541;	// L733
          reg1[v542] = v540;	// L734
        } else {
          int32_t v543 = opcode1;	// L736
          bool v544 = v543 == 3;	// L737
          if (v544) {	// L738
            int32_t v545 = imm1;	// L739
            int32_t v546 = dst1;	// L740
            int v547 = v546;	// L741
            reg1[v547] = v545;	// L742
          } else {
            int32_t v548 = opcode1;	// L744
            bool v549 = v548 == 4;	// L745
            if (v549) {	// L746
              int32_t v550 = dst1;	// L747
              int v551 = v550;	// L748
              int32_t v552 = reg1[v551];	// L749
              int32_t v553 = src1;	// L750
              int v554 = v553;	// L751
              int32_t v555 = reg1[v554];	// L752
              ap_int<33> v556 = v552;	// L753
              ap_int<33> v557 = v555;	// L754
              ap_int<33> v558 = v556 + v557;	// L755
              int32_t v559 = v558;	// L756
              reg1[v551] = v559;	// L757
            } else {
              int32_t v560 = opcode1;	// L759
              bool v561 = v560 == 5;	// L760
              if (v561) {	// L761
                int32_t v562 = dst1;	// L762
                int v563 = v562;	// L763
                int32_t v564 = reg1[v563];	// L764
                int32_t v565 = src1;	// L765
                int v566 = v565;	// L766
                int32_t v567 = reg1[v566];	// L767
                int64_t v568 = v564;	// L768
                int64_t v569 = v567;	// L769
                int64_t v570 = v568 * v569;	// L770
                int32_t v571 = v570;	// L771
                reg1[v563] = v571;	// L772
              } else {
                int32_t v572 = opcode1;	// L774
                bool v573 = v572 == 6;	// L775
                if (v573) {	// L776
                  int32_t v574 = src1;	// L777
                  int v575 = v574;	// L778
                  int32_t v576 = reg1[v575];	// L779
                  int32_t v577 = dst1;	// L780
                  int v578 = v577;	// L781
                  int32_t v579 = reg1[v578];	// L782
                  bool v580 = v576 > v579;	// L783
                  if (v580) {	// L784
                    int32_t v581 = src1;	// L785
                    int v582 = v581;	// L786
                    int32_t v583 = reg1[v582];	// L787
                    int32_t v584 = dst1;	// L788
                    int v585 = v584;	// L789
                    reg1[v585] = v583;	// L790
                  }
                } else {
                  int32_t v586 = opcode1;	// L793
                  bool v587 = v586 == 7;	// L794
                  if (v587) {	// L795
                    int32_t v588 = dst1;	// L796
                    int v589 = v588;	// L797
                    int32_t v590 = reg1[v589];	// L798
                    int32_t v591 = imm1;	// L799
                    int32_t v592 = v590 >> v591;	// L800
                    reg1[v589] = v592;	// L801
                  } else {
                    int32_t v593 = opcode1;	// L803
                    bool v594 = v593 == 8;	// L804
                    if (v594) {	// L805
                      int32_t v595 = dst1;	// L806
                      int v596 = v595;	// L807
                      int32_t v597 = reg1[v596];	// L808
                      v497.write(v597);	// L809
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void tiled_vpu_2(
  int32_t v598[4],
  hls::stream< int32_t >& v599,
  hls::stream< int32_t >& v600,
  hls::stream< int32_t >& v601,
  hls::stream< int32_t >& v602
) {	// L822
  #pragma HLS array_partition variable=v598 complete dim=1

  int32_t prog2[12];	// L838
  for (int v604 = 0; v604 < 12; v604++) {	// L839
    prog2[v604] = 0;	// L839
  }
  l_S_pc_0_pc2: for (int pc2 = 0; pc2 < 12; pc2++) {	// L840
  #pragma HLS pipeline II=1
    int32_t v606 = v599.read();	// L841
    int32_t word2;	// L842
    word2 = v606;	// L843
    int32_t v608 = word2;	// L844
    prog2[pc2] = v608;	// L845
    int32_t v609 = word2;	// L846
    v600.write(v609);	// L847
  }
  l_S_m_1_m18: for (int m18 = 0; m18 < 6; m18++) {	// L849
    int32_t reg2[4];	// L850
    for (int v612 = 0; v612 < 4; v612++) {	// L851
      reg2[v612] = 0;	// L851
    }
    l_S_step_1_step2: for (int step2 = 0; step2 < 12; step2++) {	// L852
    #pragma HLS pipeline II=1
      int32_t v614 = prog2[step2];	// L853
      int32_t word22;	// L854
      word22 = v614;	// L855
      int32_t v616 = word22;	// L856
      int32_t v617 = v616 >> 24;	// L857
      int32_t v618 = v617 & 255;	// L858
      int32_t opcode2;	// L859
      opcode2 = v618;	// L860
      int32_t v620 = word22;	// L861
      int32_t v621 = v620 >> 20;	// L862
      int32_t v622 = v621 & 15;	// L863
      int32_t dst2;	// L864
      dst2 = v622;	// L865
      int32_t v624 = word22;	// L866
      int32_t v625 = v624 >> 16;	// L867
      int32_t v626 = v625 & 15;	// L868
      int32_t src2;	// L869
      src2 = v626;	// L870
      int32_t v628 = word22;	// L871
      int32_t v629 = v628 & 65535;	// L872
      int32_t imm2;	// L873
      imm2 = v629;	// L874
      int32_t v631 = opcode2;	// L875
      bool v632 = v631 == 9;	// L876
      if (v632) {	// L877
        int32_t v633 = v601.read();	// L878
        int32_t zz2;	// L879
        zz2 = v633;	// L880
        int32_t v635 = dst2;	// L881
        int v636 = v635;	// L882
        int32_t v637 = reg2[v636];	// L883
        int32_t v638 = zz2;	// L884
        ap_int<33> v639 = v637;	// L885
        ap_int<33> v640 = v638;	// L886
        ap_int<33> v641 = v639 + v640;	// L887
        int32_t v642 = v641;	// L888
        reg2[v636] = v642;	// L889
      } else {
        int32_t v643 = opcode2;	// L891
        bool v644 = v643 == 2;	// L892
        if (v644) {	// L893
          int32_t v645 = v598[2];	// L894
          int32_t v646 = dst2;	// L895
          int v647 = v646;	// L896
          reg2[v647] = v645;	// L897
        } else {
          int32_t v648 = opcode2;	// L899
          bool v649 = v648 == 3;	// L900
          if (v649) {	// L901
            int32_t v650 = imm2;	// L902
            int32_t v651 = dst2;	// L903
            int v652 = v651;	// L904
            reg2[v652] = v650;	// L905
          } else {
            int32_t v653 = opcode2;	// L907
            bool v654 = v653 == 4;	// L908
            if (v654) {	// L909
              int32_t v655 = dst2;	// L910
              int v656 = v655;	// L911
              int32_t v657 = reg2[v656];	// L912
              int32_t v658 = src2;	// L913
              int v659 = v658;	// L914
              int32_t v660 = reg2[v659];	// L915
              ap_int<33> v661 = v657;	// L916
              ap_int<33> v662 = v660;	// L917
              ap_int<33> v663 = v661 + v662;	// L918
              int32_t v664 = v663;	// L919
              reg2[v656] = v664;	// L920
            } else {
              int32_t v665 = opcode2;	// L922
              bool v666 = v665 == 5;	// L923
              if (v666) {	// L924
                int32_t v667 = dst2;	// L925
                int v668 = v667;	// L926
                int32_t v669 = reg2[v668];	// L927
                int32_t v670 = src2;	// L928
                int v671 = v670;	// L929
                int32_t v672 = reg2[v671];	// L930
                int64_t v673 = v669;	// L931
                int64_t v674 = v672;	// L932
                int64_t v675 = v673 * v674;	// L933
                int32_t v676 = v675;	// L934
                reg2[v668] = v676;	// L935
              } else {
                int32_t v677 = opcode2;	// L937
                bool v678 = v677 == 6;	// L938
                if (v678) {	// L939
                  int32_t v679 = src2;	// L940
                  int v680 = v679;	// L941
                  int32_t v681 = reg2[v680];	// L942
                  int32_t v682 = dst2;	// L943
                  int v683 = v682;	// L944
                  int32_t v684 = reg2[v683];	// L945
                  bool v685 = v681 > v684;	// L946
                  if (v685) {	// L947
                    int32_t v686 = src2;	// L948
                    int v687 = v686;	// L949
                    int32_t v688 = reg2[v687];	// L950
                    int32_t v689 = dst2;	// L951
                    int v690 = v689;	// L952
                    reg2[v690] = v688;	// L953
                  }
                } else {
                  int32_t v691 = opcode2;	// L956
                  bool v692 = v691 == 7;	// L957
                  if (v692) {	// L958
                    int32_t v693 = dst2;	// L959
                    int v694 = v693;	// L960
                    int32_t v695 = reg2[v694];	// L961
                    int32_t v696 = imm2;	// L962
                    int32_t v697 = v695 >> v696;	// L963
                    reg2[v694] = v697;	// L964
                  } else {
                    int32_t v698 = opcode2;	// L966
                    bool v699 = v698 == 8;	// L967
                    if (v699) {	// L968
                      int32_t v700 = dst2;	// L969
                      int v701 = v700;	// L970
                      int32_t v702 = reg2[v701];	// L971
                      v602.write(v702);	// L972
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void tiled_vpu_3(
  int32_t v703[4],
  hls::stream< int32_t >& v704,
  hls::stream< int32_t >& v705,
  hls::stream< int32_t >& v706
) {	// L985
  #pragma HLS array_partition variable=v703 complete dim=1

  int32_t prog3[12];	// L1001
  for (int v708 = 0; v708 < 12; v708++) {	// L1002
    prog3[v708] = 0;	// L1002
  }
  l_S_pc_0_pc3: for (int pc3 = 0; pc3 < 12; pc3++) {	// L1003
  #pragma HLS pipeline II=1
    int32_t v710 = v704.read();	// L1004
    int32_t word3;	// L1005
    word3 = v710;	// L1006
    int32_t v712 = word3;	// L1007
    prog3[pc3] = v712;	// L1008
  }
  l_S_m_1_m19: for (int m19 = 0; m19 < 6; m19++) {	// L1010
    int32_t reg3[4];	// L1011
    for (int v715 = 0; v715 < 4; v715++) {	// L1012
      reg3[v715] = 0;	// L1012
    }
    l_S_step_1_step3: for (int step3 = 0; step3 < 12; step3++) {	// L1013
    #pragma HLS pipeline II=1
      int32_t v717 = prog3[step3];	// L1014
      int32_t word23;	// L1015
      word23 = v717;	// L1016
      int32_t v719 = word23;	// L1017
      int32_t v720 = v719 >> 24;	// L1018
      int32_t v721 = v720 & 255;	// L1019
      int32_t opcode3;	// L1020
      opcode3 = v721;	// L1021
      int32_t v723 = word23;	// L1022
      int32_t v724 = v723 >> 20;	// L1023
      int32_t v725 = v724 & 15;	// L1024
      int32_t dst3;	// L1025
      dst3 = v725;	// L1026
      int32_t v727 = word23;	// L1027
      int32_t v728 = v727 >> 16;	// L1028
      int32_t v729 = v728 & 15;	// L1029
      int32_t src3;	// L1030
      src3 = v729;	// L1031
      int32_t v731 = word23;	// L1032
      int32_t v732 = v731 & 65535;	// L1033
      int32_t imm3;	// L1034
      imm3 = v732;	// L1035
      int32_t v734 = opcode3;	// L1036
      bool v735 = v734 == 9;	// L1037
      if (v735) {	// L1038
        int32_t v736 = v705.read();	// L1039
        int32_t zz3;	// L1040
        zz3 = v736;	// L1041
        int32_t v738 = dst3;	// L1042
        int v739 = v738;	// L1043
        int32_t v740 = reg3[v739];	// L1044
        int32_t v741 = zz3;	// L1045
        ap_int<33> v742 = v740;	// L1046
        ap_int<33> v743 = v741;	// L1047
        ap_int<33> v744 = v742 + v743;	// L1048
        int32_t v745 = v744;	// L1049
        reg3[v739] = v745;	// L1050
      } else {
        int32_t v746 = opcode3;	// L1052
        bool v747 = v746 == 2;	// L1053
        if (v747) {	// L1054
          int32_t v748 = v703[3];	// L1055
          int32_t v749 = dst3;	// L1056
          int v750 = v749;	// L1057
          reg3[v750] = v748;	// L1058
        } else {
          int32_t v751 = opcode3;	// L1060
          bool v752 = v751 == 3;	// L1061
          if (v752) {	// L1062
            int32_t v753 = imm3;	// L1063
            int32_t v754 = dst3;	// L1064
            int v755 = v754;	// L1065
            reg3[v755] = v753;	// L1066
          } else {
            int32_t v756 = opcode3;	// L1068
            bool v757 = v756 == 4;	// L1069
            if (v757) {	// L1070
              int32_t v758 = dst3;	// L1071
              int v759 = v758;	// L1072
              int32_t v760 = reg3[v759];	// L1073
              int32_t v761 = src3;	// L1074
              int v762 = v761;	// L1075
              int32_t v763 = reg3[v762];	// L1076
              ap_int<33> v764 = v760;	// L1077
              ap_int<33> v765 = v763;	// L1078
              ap_int<33> v766 = v764 + v765;	// L1079
              int32_t v767 = v766;	// L1080
              reg3[v759] = v767;	// L1081
            } else {
              int32_t v768 = opcode3;	// L1083
              bool v769 = v768 == 5;	// L1084
              if (v769) {	// L1085
                int32_t v770 = dst3;	// L1086
                int v771 = v770;	// L1087
                int32_t v772 = reg3[v771];	// L1088
                int32_t v773 = src3;	// L1089
                int v774 = v773;	// L1090
                int32_t v775 = reg3[v774];	// L1091
                int64_t v776 = v772;	// L1092
                int64_t v777 = v775;	// L1093
                int64_t v778 = v776 * v777;	// L1094
                int32_t v779 = v778;	// L1095
                reg3[v771] = v779;	// L1096
              } else {
                int32_t v780 = opcode3;	// L1098
                bool v781 = v780 == 6;	// L1099
                if (v781) {	// L1100
                  int32_t v782 = src3;	// L1101
                  int v783 = v782;	// L1102
                  int32_t v784 = reg3[v783];	// L1103
                  int32_t v785 = dst3;	// L1104
                  int v786 = v785;	// L1105
                  int32_t v787 = reg3[v786];	// L1106
                  bool v788 = v784 > v787;	// L1107
                  if (v788) {	// L1108
                    int32_t v789 = src3;	// L1109
                    int v790 = v789;	// L1110
                    int32_t v791 = reg3[v790];	// L1111
                    int32_t v792 = dst3;	// L1112
                    int v793 = v792;	// L1113
                    reg3[v793] = v791;	// L1114
                  }
                } else {
                  int32_t v794 = opcode3;	// L1117
                  bool v795 = v794 == 7;	// L1118
                  if (v795) {	// L1119
                    int32_t v796 = dst3;	// L1120
                    int v797 = v796;	// L1121
                    int32_t v798 = reg3[v797];	// L1122
                    int32_t v799 = imm3;	// L1123
                    int32_t v800 = v798 >> v799;	// L1124
                    reg3[v797] = v800;	// L1125
                  } else {
                    int32_t v801 = opcode3;	// L1127
                    bool v802 = v801 == 8;	// L1128
                    if (v802) {	// L1129
                      int32_t v803 = dst3;	// L1130
                      int v804 = v803;	// L1131
                      int32_t v805 = reg3[v804];	// L1132
                      v706.write(v805);	// L1133
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void tiled_vpu_y_out_drain_0(
  int32_t v806[6][4],
  hls::stream< int32_t >& v807
) {	// L1146
  #pragma HLS array_partition variable=v806 complete dim=1
  #pragma HLS array_partition variable=v806 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 6; _t5++) {	// L1147
  #pragma HLS pipeline II=1
    int32_t v809 = v807.read();	// L1148
    v806[_t5][0] = v809;	// L1149
  }
}

void tiled_vpu_y_out_drain_1(
  int32_t v810[6][4],
  hls::stream< int32_t >& v811
) {	// L1153
  #pragma HLS array_partition variable=v810 complete dim=1
  #pragma HLS array_partition variable=v810 complete dim=2

  l_S__t_0__t6: for (int _t6 = 0; _t6 < 6; _t6++) {	// L1154
  #pragma HLS pipeline II=1
    int32_t v813 = v811.read();	// L1155
    v810[_t6][1] = v813;	// L1156
  }
}

void tiled_vpu_y_out_drain_2(
  int32_t v814[6][4],
  hls::stream< int32_t >& v815
) {	// L1160
  #pragma HLS array_partition variable=v814 complete dim=1
  #pragma HLS array_partition variable=v814 complete dim=2

  l_S__t_0__t7: for (int _t7 = 0; _t7 < 6; _t7++) {	// L1161
  #pragma HLS pipeline II=1
    int32_t v817 = v815.read();	// L1162
    v814[_t7][2] = v817;	// L1163
  }
}

void tiled_vpu_y_out_drain_3(
  int32_t v818[6][4],
  hls::stream< int32_t >& v819
) {	// L1167
  #pragma HLS array_partition variable=v818 complete dim=1
  #pragma HLS array_partition variable=v818 complete dim=2

  l_S__t_0__t8: for (int _t8 = 0; _t8 < 6; _t8++) {	// L1168
  #pragma HLS pipeline II=1
    int32_t v821 = v819.read();	// L1169
    v818[_t8][3] = v821;	// L1170
  }
}

/// This is top function.
void top(
  int8_t v822[12][4],
  int32_t v823[12],
  int8_t v824[4][4][2],
  int32_t v825[4],
  int32_t v826[6][4]
) {	// L1174
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v822 complete dim=1
  #pragma HLS array_partition variable=v822 complete dim=2

  #pragma HLS array_partition variable=v823 complete dim=1

  #pragma HLS array_partition variable=v824 complete dim=1
  #pragma HLS array_partition variable=v824 complete dim=2
  #pragma HLS array_partition variable=v824 complete dim=3

  #pragma HLS array_partition variable=v825 complete dim=1

  #pragma HLS array_partition variable=v826 complete dim=1
  #pragma HLS array_partition variable=v826 complete dim=2

  hls::stream< int8_t > v827;
  #pragma HLS stream variable=v827 depth=2	// L1175
  hls::stream< int8_t > v828;
  #pragma HLS stream variable=v828 depth=2	// L1176
  hls::stream< int8_t > v829;
  #pragma HLS stream variable=v829 depth=2	// L1177
  hls::stream< int8_t > v830;
  #pragma HLS stream variable=v830 depth=2	// L1178
  hls::stream< int8_t > v831;
  #pragma HLS stream variable=v831 depth=2	// L1179
  hls::stream< int8_t > v832;
  #pragma HLS stream variable=v832 depth=2	// L1180
  hls::stream< int8_t > v833;
  #pragma HLS stream variable=v833 depth=2	// L1181
  hls::stream< int8_t > v834;
  #pragma HLS stream variable=v834 depth=2	// L1182
  hls::stream< int8_t > v835;
  #pragma HLS stream variable=v835 depth=2	// L1183
  hls::stream< int8_t > v836;
  #pragma HLS stream variable=v836 depth=2	// L1184
  hls::stream< int8_t > v837;
  #pragma HLS stream variable=v837 depth=2	// L1185
  hls::stream< int8_t > v838;
  #pragma HLS stream variable=v838 depth=2	// L1186
  hls::stream< int8_t > v839;
  #pragma HLS stream variable=v839 depth=2	// L1187
  hls::stream< int8_t > v840;
  #pragma HLS stream variable=v840 depth=2	// L1188
  hls::stream< int8_t > v841;
  #pragma HLS stream variable=v841 depth=2	// L1189
  hls::stream< int8_t > v842;
  #pragma HLS stream variable=v842 depth=2	// L1190
  hls::stream< int32_t > v843;
  #pragma HLS stream variable=v843 depth=2	// L1191
  hls::stream< int32_t > v844;
  #pragma HLS stream variable=v844 depth=2	// L1192
  hls::stream< int32_t > v845;
  #pragma HLS stream variable=v845 depth=2	// L1193
  hls::stream< int32_t > v846;
  #pragma HLS stream variable=v846 depth=2	// L1194
  hls::stream< int32_t > v847;
  #pragma HLS stream variable=v847 depth=2	// L1195
  hls::stream< int32_t > v848;
  #pragma HLS stream variable=v848 depth=2	// L1196
  hls::stream< int32_t > v849;
  #pragma HLS stream variable=v849 depth=2	// L1197
  hls::stream< int32_t > v850;
  #pragma HLS stream variable=v850 depth=2	// L1198
  hls::stream< int32_t > v851;
  #pragma HLS stream variable=v851 depth=2	// L1199
  hls::stream< int32_t > v852;
  #pragma HLS stream variable=v852 depth=2	// L1200
  hls::stream< int32_t > v853;
  #pragma HLS stream variable=v853 depth=2	// L1201
  hls::stream< int32_t > v854;
  #pragma HLS stream variable=v854 depth=2	// L1202
  hls::stream< int32_t > v855;
  #pragma HLS stream variable=v855 depth=2	// L1203
  hls::stream< int32_t > v856;
  #pragma HLS stream variable=v856 depth=2	// L1204
  hls::stream< int32_t > v857;
  #pragma HLS stream variable=v857 depth=2	// L1205
  hls::stream< int32_t > v858;
  #pragma HLS stream variable=v858 depth=2	// L1206
  hls::stream< int32_t > v859;
  #pragma HLS stream variable=v859 depth=2	// L1207
  hls::stream< int32_t > v860;
  #pragma HLS stream variable=v860 depth=2	// L1208
  hls::stream< int32_t > v861;
  #pragma HLS stream variable=v861 depth=2	// L1209
  hls::stream< int32_t > v862;
  #pragma HLS stream variable=v862 depth=2	// L1210
  hls::stream< int8_t > v863;
  #pragma HLS stream variable=v863 depth=2	// L1211
  hls::stream< int8_t > v864;
  #pragma HLS stream variable=v864 depth=2	// L1212
  hls::stream< int8_t > v865;
  #pragma HLS stream variable=v865 depth=2	// L1213
  hls::stream< int8_t > v866;
  #pragma HLS stream variable=v866 depth=2	// L1214
  hls::stream< int32_t > v867;
  #pragma HLS stream variable=v867 depth=2	// L1215
  hls::stream< int32_t > v868;
  #pragma HLS stream variable=v868 depth=2	// L1216
  hls::stream< int32_t > v869;
  #pragma HLS stream variable=v869 depth=2	// L1217
  hls::stream< int32_t > v870;
  #pragma HLS stream variable=v870 depth=2	// L1218
  hls::stream< int32_t > v871;
  #pragma HLS stream variable=v871 depth=2	// L1219
  hls::stream< int32_t > v872;
  #pragma HLS stream variable=v872 depth=2	// L1220
  hls::stream< int32_t > v873;
  #pragma HLS stream variable=v873 depth=2	// L1221
  hls::stream< int32_t > v874;
  #pragma HLS stream variable=v874 depth=2	// L1222
  hls::stream< int32_t > v875;
  #pragma HLS stream variable=v875 depth=2	// L1223
  tiled_mac_a_in_load_0(v822, v863);	// L1224
  tiled_mac_a_in_load_1(v822, v864);	// L1225
  tiled_mac_a_in_load_2(v822, v865);	// L1226
  tiled_mac_a_in_load_3(v822, v866);	// L1227
  tiled_vpu_op_in_load_0(v823, v871);	// L1228
  tiled_mac_0_0(v824, v863, v847, v828);	// L1229
  tiled_mac_0_1(v824, v828, v848, v829);	// L1230
  tiled_mac_0_2(v824, v829, v849, v830);	// L1231
  tiled_mac_0_3(v824, v830, v850);	// L1232
  tiled_mac_1_0(v824, v864, v847, v851, v832);	// L1233
  tiled_mac_1_1(v824, v832, v848, v852, v833);	// L1234
  tiled_mac_1_2(v824, v833, v849, v853, v834);	// L1235
  tiled_mac_1_3(v824, v834, v850, v854);	// L1236
  tiled_mac_2_0(v824, v865, v851, v855, v836);	// L1237
  tiled_mac_2_1(v824, v836, v852, v856, v837);	// L1238
  tiled_mac_2_2(v824, v837, v853, v857, v838);	// L1239
  tiled_mac_2_3(v824, v838, v854, v858);	// L1240
  tiled_mac_3_0(v824, v866, v855, v867, v840);	// L1241
  tiled_mac_3_1(v824, v840, v856, v868, v841);	// L1242
  tiled_mac_3_2(v824, v841, v857, v869, v842);	// L1243
  tiled_mac_3_3(v824, v842, v858, v870);	// L1244
  tiled_vpu_0(v825, v871, v860, v867, v872);	// L1245
  tiled_vpu_1(v825, v860, v861, v868, v873);	// L1246
  tiled_vpu_2(v825, v861, v862, v869, v874);	// L1247
  tiled_vpu_3(v825, v862, v870, v875);	// L1248
  tiled_vpu_y_out_drain_0(v826, v872);	// L1249
  tiled_vpu_y_out_drain_1(v826, v873);	// L1250
  tiled_vpu_y_out_drain_2(v826, v874);	// L1251
  tiled_vpu_y_out_drain_3(v826, v875);	// L1252
}


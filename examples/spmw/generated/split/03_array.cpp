
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
void feed_up_load(
  int8_t v0[4][4],
  int v1,
  hls::stream< hls::vector< int8_t, 4 > >& v2
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L4
    int8_t _blk[4];	// L7
    for (int v5 = 0; v5 < 4; v5++) {	// L8
      _blk[v5] = 0;	// L8
    }
    l_S__b0_0__b0: for (int _b0 = 0; _b0 < 4; _b0++) {	// L9
    #pragma HLS pipeline II=1
      int8_t v7 = v0[_t][_b0];	// L10
      _blk[_b0] = v7;	// L11
    }
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = _blk[_iv0];
      }
      v2.write(_vec);
    }	// L13
  }
}

void feed_3_up_load(
  int8_t v8[4][4],
  int v9,
  hls::stream< hls::vector< int8_t, 4 > >& v10
) {	// L17
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 4; _t1++) {	// L18
    int8_t _blk1[4];	// L21
    for (int v13 = 0; v13 < 4; v13++) {	// L22
      _blk1[v13] = 0;	// L22
    }
    l_S__b0_0__b01: for (int _b01 = 0; _b01 < 4; _b01++) {	// L23
    #pragma HLS pipeline II=1
      int8_t v15 = v8[_t1][_b01];	// L24
      _blk1[_b01] = v15;	// L25
    }
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = _blk1[_iv0];
      }
      v10.write(_vec);
    }	// L27
  }
}

void pe_r0(
  int v16,
  int v17,
  hls::stream< int32_t >& v18,
  hls::stream< int8_t >& v19,
  hls::stream< int8_t >& v20,
  hls::stream< int8_t >& v21,
  hls::stream< int8_t >& v22
) {	// L31
  int32_t acc;	// L33
  acc = 0;	// L34
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L35
  #pragma HLS pipeline II=1
    int8_t v25 = v22.read();	// L36
    int8_t a;	// L37
    a = v25;	// L38
    int8_t v27 = v20.read();	// L39
    int8_t b;	// L40
    b = v27;	// L41
    int8_t v29 = a;	// L42
    int8_t v30 = b;	// L43
    int16_t v31 = v29;	// L44
    int16_t v32 = v30;	// L45
    int16_t v33 = v31 * v32;	// L46
    int32_t v34 = acc;	// L47
    ap_int<33> v35 = v34;	// L48
    ap_int<33> v36 = v33;	// L49
    ap_int<33> v37 = v35 + v36;	// L50
    int32_t v38 = v37;	// L51
    acc = v38;	// L52
    int8_t v39 = a;	// L53
    v19.write(v39);	// L54
    int8_t v40 = b;	// L55
    v21.write(v40);	// L56
  }
  int32_t v41 = acc;	// L58
  v18.write(v41);	// L59
}

void pe_r1(
  int v42,
  int v43,
  hls::stream< int32_t >& v44,
  hls::stream< int8_t >& v45,
  hls::stream< int8_t >& v46,
  hls::stream< int8_t >& v47,
  hls::stream< int8_t >& v48
) {	// L62
  int32_t acc1;	// L64
  acc1 = 0;	// L65
  l_S_k_0_k1: for (int k1 = 0; k1 < 4; k1++) {	// L66
  #pragma HLS pipeline II=1
    int8_t v51 = v48.read();	// L67
    int8_t a1;	// L68
    a1 = v51;	// L69
    int8_t v53 = v46.read();	// L70
    int8_t b1;	// L71
    b1 = v53;	// L72
    int8_t v55 = a1;	// L73
    int8_t v56 = b1;	// L74
    int16_t v57 = v55;	// L75
    int16_t v58 = v56;	// L76
    int16_t v59 = v57 * v58;	// L77
    int32_t v60 = acc1;	// L78
    ap_int<33> v61 = v60;	// L79
    ap_int<33> v62 = v59;	// L80
    ap_int<33> v63 = v61 + v62;	// L81
    int32_t v64 = v63;	// L82
    acc1 = v64;	// L83
    int8_t v65 = a1;	// L84
    v45.write(v65);	// L85
    int8_t v66 = b1;	// L86
    v47.write(v66);	// L87
  }
  int32_t v67 = acc1;	// L89
  v44.write(v67);	// L90
}

void pe_r2(
  int v68,
  int v69,
  hls::stream< int32_t >& v70,
  hls::stream< int8_t >& v71,
  hls::stream< int8_t >& v72,
  hls::stream< int8_t >& v73
) {	// L93
  int32_t acc2;	// L95
  acc2 = 0;	// L96
  l_S_k_0_k2: for (int k2 = 0; k2 < 4; k2++) {	// L97
  #pragma HLS pipeline II=1
    int8_t v76 = v73.read();	// L98
    int8_t a2;	// L99
    a2 = v76;	// L100
    int8_t v78 = v72.read();	// L101
    int8_t b2;	// L102
    b2 = v78;	// L103
    int8_t v80 = a2;	// L104
    int8_t v81 = b2;	// L105
    int16_t v82 = v80;	// L106
    int16_t v83 = v81;	// L107
    int16_t v84 = v82 * v83;	// L108
    int32_t v85 = acc2;	// L109
    ap_int<33> v86 = v85;	// L110
    ap_int<33> v87 = v84;	// L111
    ap_int<33> v88 = v86 + v87;	// L112
    int32_t v89 = v88;	// L113
    acc2 = v89;	// L114
    int8_t v90 = a2;	// L115
    v71.write(v90);	// L116
  }
  int32_t v91 = acc2;	// L118
  v70.write(v91);	// L119
}

void pe_r3(
  int v92,
  int v93,
  hls::stream< int32_t >& v94,
  hls::stream< int8_t >& v95,
  hls::stream< int8_t >& v96,
  hls::stream< int8_t >& v97,
  hls::stream< int8_t >& v98
) {	// L122
  int32_t acc3;	// L124
  acc3 = 0;	// L125
  l_S_k_0_k3: for (int k3 = 0; k3 < 4; k3++) {	// L126
  #pragma HLS pipeline II=1
    int8_t v101 = v98.read();	// L127
    int8_t a3;	// L128
    a3 = v101;	// L129
    int8_t v103 = v96.read();	// L130
    int8_t b3;	// L131
    b3 = v103;	// L132
    int8_t v105 = a3;	// L133
    int8_t v106 = b3;	// L134
    int16_t v107 = v105;	// L135
    int16_t v108 = v106;	// L136
    int16_t v109 = v107 * v108;	// L137
    int32_t v110 = acc3;	// L138
    ap_int<33> v111 = v110;	// L139
    ap_int<33> v112 = v109;	// L140
    ap_int<33> v113 = v111 + v112;	// L141
    int32_t v114 = v113;	// L142
    acc3 = v114;	// L143
    int8_t v115 = a3;	// L144
    v95.write(v115);	// L145
    int8_t v116 = b3;	// L146
    v97.write(v116);	// L147
  }
  int32_t v117 = acc3;	// L149
  v94.write(v117);	// L150
}

void pe_r4(
  int v118,
  int v119,
  hls::stream< int32_t >& v120,
  hls::stream< int8_t >& v121,
  hls::stream< int8_t >& v122,
  hls::stream< int8_t >& v123
) {	// L153
  int32_t acc4;	// L155
  acc4 = 0;	// L156
  l_S_k_0_k4: for (int k4 = 0; k4 < 4; k4++) {	// L157
  #pragma HLS pipeline II=1
    int8_t v126 = v123.read();	// L158
    int8_t a4;	// L159
    a4 = v126;	// L160
    int8_t v128 = v121.read();	// L161
    int8_t b4;	// L162
    b4 = v128;	// L163
    int8_t v130 = a4;	// L164
    int8_t v131 = b4;	// L165
    int16_t v132 = v130;	// L166
    int16_t v133 = v131;	// L167
    int16_t v134 = v132 * v133;	// L168
    int32_t v135 = acc4;	// L169
    ap_int<33> v136 = v135;	// L170
    ap_int<33> v137 = v134;	// L171
    ap_int<33> v138 = v136 + v137;	// L172
    int32_t v139 = v138;	// L173
    acc4 = v139;	// L174
    int8_t v140 = b4;	// L175
    v122.write(v140);	// L176
  }
  int32_t v141 = acc4;	// L178
  v120.write(v141);	// L179
}

void pe_r5(
  int v142,
  int v143,
  hls::stream< int32_t >& v144,
  hls::stream< int8_t >& v145,
  hls::stream< int8_t >& v146,
  hls::stream< int8_t >& v147
) {	// L182
  int32_t acc5;	// L184
  acc5 = 0;	// L185
  l_S_k_0_k5: for (int k5 = 0; k5 < 4; k5++) {	// L186
  #pragma HLS pipeline II=1
    int8_t v150 = v147.read();	// L187
    int8_t a5;	// L188
    a5 = v150;	// L189
    int8_t v152 = v146.read();	// L190
    int8_t b5;	// L191
    b5 = v152;	// L192
    int8_t v154 = a5;	// L193
    int8_t v155 = b5;	// L194
    int16_t v156 = v154;	// L195
    int16_t v157 = v155;	// L196
    int16_t v158 = v156 * v157;	// L197
    int32_t v159 = acc5;	// L198
    ap_int<33> v160 = v159;	// L199
    ap_int<33> v161 = v158;	// L200
    ap_int<33> v162 = v160 + v161;	// L201
    int32_t v163 = v162;	// L202
    acc5 = v163;	// L203
    int8_t v164 = a5;	// L204
    v145.write(v164);	// L205
  }
  int32_t v165 = acc5;	// L207
  v144.write(v165);	// L208
}

void pe_r6(
  int v166,
  int v167,
  hls::stream< int32_t >& v168,
  hls::stream< int8_t >& v169,
  hls::stream< int8_t >& v170,
  hls::stream< int8_t >& v171,
  hls::stream< int8_t >& v172
) {	// L211
  int32_t acc6;	// L213
  acc6 = 0;	// L214
  l_S_k_0_k6: for (int k6 = 0; k6 < 4; k6++) {	// L215
  #pragma HLS pipeline II=1
    int8_t v175 = v172.read();	// L216
    int8_t a6;	// L217
    a6 = v175;	// L218
    int8_t v177 = v170.read();	// L219
    int8_t b6;	// L220
    b6 = v177;	// L221
    int8_t v179 = a6;	// L222
    int8_t v180 = b6;	// L223
    int16_t v181 = v179;	// L224
    int16_t v182 = v180;	// L225
    int16_t v183 = v181 * v182;	// L226
    int32_t v184 = acc6;	// L227
    ap_int<33> v185 = v184;	// L228
    ap_int<33> v186 = v183;	// L229
    ap_int<33> v187 = v185 + v186;	// L230
    int32_t v188 = v187;	// L231
    acc6 = v188;	// L232
    int8_t v189 = a6;	// L233
    v169.write(v189);	// L234
    int8_t v190 = b6;	// L235
    v171.write(v190);	// L236
  }
  int32_t v191 = acc6;	// L238
  v168.write(v191);	// L239
}

void pe_r7(
  int v192,
  int v193,
  hls::stream< int32_t >& v194,
  hls::stream< int8_t >& v195,
  hls::stream< int8_t >& v196
) {	// L242
  int32_t acc7;	// L244
  acc7 = 0;	// L245
  l_S_k_0_k7: for (int k7 = 0; k7 < 4; k7++) {	// L246
  #pragma HLS pipeline II=1
    int8_t v199 = v196.read();	// L247
    int8_t a7;	// L248
    a7 = v199;	// L249
    int8_t v201 = v195.read();	// L250
    int8_t b7;	// L251
    b7 = v201;	// L252
    int8_t v203 = a7;	// L253
    int8_t v204 = b7;	// L254
    int16_t v205 = v203;	// L255
    int16_t v206 = v204;	// L256
    int16_t v207 = v205 * v206;	// L257
    int32_t v208 = acc7;	// L258
    ap_int<33> v209 = v208;	// L259
    ap_int<33> v210 = v207;	// L260
    ap_int<33> v211 = v209 + v210;	// L261
    int32_t v212 = v211;	// L262
    acc7 = v212;	// L263
  }
  int32_t v213 = acc7;	// L265
  v194.write(v213);	// L266
}

void pe_r8(
  int v214,
  int v215,
  hls::stream< int32_t >& v216,
  hls::stream< int8_t >& v217,
  hls::stream< int8_t >& v218,
  hls::stream< int8_t >& v219
) {	// L269
  int32_t acc8;	// L271
  acc8 = 0;	// L272
  l_S_k_0_k8: for (int k8 = 0; k8 < 4; k8++) {	// L273
  #pragma HLS pipeline II=1
    int8_t v222 = v219.read();	// L274
    int8_t a8;	// L275
    a8 = v222;	// L276
    int8_t v224 = v217.read();	// L277
    int8_t b8;	// L278
    b8 = v224;	// L279
    int8_t v226 = a8;	// L280
    int8_t v227 = b8;	// L281
    int16_t v228 = v226;	// L282
    int16_t v229 = v227;	// L283
    int16_t v230 = v228 * v229;	// L284
    int32_t v231 = acc8;	// L285
    ap_int<33> v232 = v231;	// L286
    ap_int<33> v233 = v230;	// L287
    ap_int<33> v234 = v232 + v233;	// L288
    int32_t v235 = v234;	// L289
    acc8 = v235;	// L290
    int8_t v236 = b8;	// L291
    v218.write(v236);	// L292
  }
  int32_t v237 = acc8;	// L294
  v216.write(v237);	// L295
}

void drain_r0(
  int v238,
  int v239,
  hls::stream< int32_t >& v240,
  hls::stream< int32_t >& v241,
  hls::stream< int32_t >& v242
) {	// L298
  int32_t v243 = v241.read();	// L299
  v240.write(v243);	// L300
  l_S__i_0__i: for (int _i = 0; _i < v238; _i++) {	// L301
  #pragma HLS pipeline II=1
    int32_t v245 = v242.read();	// L302
    v240.write(v245);	// L303
  }
}

void drain_r1(
  int v246,
  int v247,
  hls::stream< int32_t >& v248,
  hls::stream< int32_t >& v249
) {	// L307
  int32_t v250 = v249.read();	// L308
  v248.write(v250);	// L309
  l_S__i_0__i1: for (int _i1 = 0; _i1 < v246; _i1++) {	// L310
  #pragma HLS pipeline II=1
    v248.write(0);	// L312
  }
}

void drain_r2(
  int v252,
  int v253,
  hls::stream< int32_t >& v254,
  hls::stream< int32_t >& v255,
  hls::stream< int32_t >& v256
) {	// L316
  int32_t v257 = v255.read();	// L317
  v254.write(v257);	// L318
  l_S__i_0__i2: for (int _i2 = 0; _i2 < v252; _i2++) {	// L319
  #pragma HLS pipeline II=1
    int32_t v259 = v256.read();	// L320
    v254.write(v259);	// L321
  }
}

void feed_r0(
  int v260,
  hls::stream< hls::vector< int8_t, 4 > >& v261,
  hls::stream< int8_t >& v262,
  hls::stream< hls::vector< int8_t, 4 > >& v263
) {	// L325
  l_S_k_0_k9: for (int k9 = 0; k9 < 4; k9++) {	// L326
  #pragma HLS pipeline II=1
    int8_t v265[4];
    {
      hls::vector< int8_t, 4 > _vec = v263.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v265[_iv0] = _vec[_iv0];
      }
    }	// L327
    int8_t v266 = v265[v260];	// L328
    v262.write(v266);	// L329
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v265[_iv0];
      }
      v261.write(_vec);
    }	// L330
  }
}

void feed_r1(
  int v267,
  hls::stream< hls::vector< int8_t, 4 > >& v268,
  hls::stream< int8_t >& v269,
  hls::stream< hls::vector< int8_t, 4 > >& v270
) {	// L334
  l_S_k_0_k10: for (int k10 = 0; k10 < 4; k10++) {	// L335
  #pragma HLS pipeline II=1
    int8_t v272[4];
    {
      hls::vector< int8_t, 4 > _vec = v270.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v272[_iv0] = _vec[_iv0];
      }
    }	// L336
    int8_t v273 = v272[v267];	// L337
    v269.write(v273);	// L338
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v272[_iv0];
      }
      v268.write(_vec);
    }	// L339
  }
}

void feed_r2(
  int v274,
  hls::stream< int8_t >& v275,
  hls::stream< hls::vector< int8_t, 4 > >& v276
) {	// L343
  l_S_k_0_k11: for (int k11 = 0; k11 < 4; k11++) {	// L344
  #pragma HLS pipeline II=1
    int8_t v278[4];
    {
      hls::vector< int8_t, 4 > _vec = v276.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v278[_iv0] = _vec[_iv0];
      }
    }	// L345
    int8_t v279 = v278[v274];	// L346
    v275.write(v279);	// L347
  }
}

void feed_3_r0(
  int v280,
  hls::stream< hls::vector< int8_t, 4 > >& v281,
  hls::stream< int8_t >& v282,
  hls::stream< hls::vector< int8_t, 4 > >& v283
) {	// L351
  l_S_k_0_k12: for (int k12 = 0; k12 < 4; k12++) {	// L352
  #pragma HLS pipeline II=1
    int8_t v285[4];
    {
      hls::vector< int8_t, 4 > _vec = v283.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v285[_iv0] = _vec[_iv0];
      }
    }	// L353
    int8_t v286 = v285[v280];	// L354
    v282.write(v286);	// L355
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v285[_iv0];
      }
      v281.write(_vec);
    }	// L356
  }
}

void feed_3_r1(
  int v287,
  hls::stream< hls::vector< int8_t, 4 > >& v288,
  hls::stream< int8_t >& v289,
  hls::stream< hls::vector< int8_t, 4 > >& v290
) {	// L360
  l_S_k_0_k13: for (int k13 = 0; k13 < 4; k13++) {	// L361
  #pragma HLS pipeline II=1
    int8_t v292[4];
    {
      hls::vector< int8_t, 4 > _vec = v290.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v292[_iv0] = _vec[_iv0];
      }
    }	// L362
    int8_t v293 = v292[v287];	// L363
    v289.write(v293);	// L364
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v292[_iv0];
      }
      v288.write(_vec);
    }	// L365
  }
}

void feed_3_r2(
  int v294,
  hls::stream< int8_t >& v295,
  hls::stream< hls::vector< int8_t, 4 > >& v296
) {	// L369
  l_S_k_0_k14: for (int k14 = 0; k14 < 4; k14++) {	// L370
  #pragma HLS pipeline II=1
    int8_t v298[4];
    {
      hls::vector< int8_t, 4 > _vec = v296.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v298[_iv0] = _vec[_iv0];
      }
    }	// L371
    int8_t v299 = v298[v294];	// L372
    v295.write(v299);	// L373
  }
}

void drain_down_drain(
  int32_t v300[4][4],
  int v301,
  hls::stream< int32_t >& v302
) {	// L377
  #pragma HLS array_partition variable=v300 complete dim=1
  #pragma HLS array_partition variable=v300 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L378
  #pragma HLS pipeline II=1
    int32_t v304 = v302.read();	// L379
    v300[v301][_t2] = v304;	// L380
  }
}

/// This is top function.
void top(
  int8_t v305[4][4],
  int8_t v306[4][4],
  int32_t v307[4][4]
) {	// L384
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v305 complete dim=1
  #pragma HLS array_partition variable=v305 complete dim=2

  #pragma HLS array_partition variable=v306 complete dim=1
  #pragma HLS array_partition variable=v306 complete dim=2

  #pragma HLS array_partition variable=v307 complete dim=1
  #pragma HLS array_partition variable=v307 complete dim=2

  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v308;
  #pragma HLS stream variable=v308 depth=2	// L385
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v309;
  #pragma HLS stream variable=v309 depth=2	// L386
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v310;
  #pragma HLS stream variable=v310 depth=2	// L387
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v311;
  #pragma HLS stream variable=v311 depth=2	// L388
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v312;
  #pragma HLS stream variable=v312 depth=2	// L389
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v313;
  #pragma HLS stream variable=v313 depth=2	// L390
  hls::stream< int32_t > v314;
  #pragma HLS stream variable=v314 depth=4	// L391
  hls::stream< int32_t > v315;
  #pragma HLS stream variable=v315 depth=4	// L392
  hls::stream< int32_t > v316;
  #pragma HLS stream variable=v316 depth=4	// L393
  hls::stream< int32_t > v317;
  #pragma HLS stream variable=v317 depth=4	// L394
  hls::stream< int32_t > v318;
  #pragma HLS stream variable=v318 depth=2	// L395
  hls::stream< int32_t > v319;
  #pragma HLS stream variable=v319 depth=2	// L396
  hls::stream< int32_t > v320;
  #pragma HLS stream variable=v320 depth=2	// L397
  hls::stream< int32_t > v321;
  #pragma HLS stream variable=v321 depth=2	// L398
  hls::stream< int32_t > v322;
  #pragma HLS stream variable=v322 depth=2	// L399
  hls::stream< int32_t > v323;
  #pragma HLS stream variable=v323 depth=2	// L400
  hls::stream< int32_t > v324;
  #pragma HLS stream variable=v324 depth=2	// L401
  hls::stream< int32_t > v325;
  #pragma HLS stream variable=v325 depth=2	// L402
  hls::stream< int32_t > v326;
  #pragma HLS stream variable=v326 depth=2	// L403
  hls::stream< int32_t > v327;
  #pragma HLS stream variable=v327 depth=2	// L404
  hls::stream< int32_t > v328;
  #pragma HLS stream variable=v328 depth=2	// L405
  hls::stream< int32_t > v329;
  #pragma HLS stream variable=v329 depth=2	// L406
  hls::stream< int32_t > v330;
  #pragma HLS stream variable=v330 depth=2	// L407
  hls::stream< int8_t > v331;
  #pragma HLS stream variable=v331 depth=2	// L408
  hls::stream< int32_t > v332;
  #pragma HLS stream variable=v332 depth=2	// L409
  hls::stream< int8_t > v333;
  #pragma HLS stream variable=v333 depth=2	// L410
  hls::stream< int32_t > v334;
  #pragma HLS stream variable=v334 depth=2	// L411
  hls::stream< int8_t > v335;
  #pragma HLS stream variable=v335 depth=2	// L412
  hls::stream< int8_t > v336;
  #pragma HLS stream variable=v336 depth=2	// L413
  hls::stream< int32_t > v337;
  #pragma HLS stream variable=v337 depth=2	// L414
  hls::stream< int8_t > v338;
  #pragma HLS stream variable=v338 depth=2	// L415
  hls::stream< int32_t > v339;
  #pragma HLS stream variable=v339 depth=2	// L416
  hls::stream< int8_t > v340;
  #pragma HLS stream variable=v340 depth=2	// L417
  hls::stream< int8_t > v341;
  #pragma HLS stream variable=v341 depth=2	// L418
  hls::stream< int32_t > v342;
  #pragma HLS stream variable=v342 depth=2	// L419
  hls::stream< int8_t > v343;
  #pragma HLS stream variable=v343 depth=2	// L420
  hls::stream< int8_t > v344;
  #pragma HLS stream variable=v344 depth=2	// L421
  hls::stream< int32_t > v345;
  #pragma HLS stream variable=v345 depth=2	// L422
  hls::stream< int8_t > v346;
  #pragma HLS stream variable=v346 depth=2	// L423
  hls::stream< int8_t > v347;
  #pragma HLS stream variable=v347 depth=2	// L424
  hls::stream< int8_t > v348;
  #pragma HLS stream variable=v348 depth=2	// L425
  hls::stream< int32_t > v349;
  #pragma HLS stream variable=v349 depth=2	// L426
  hls::stream< int8_t > v350;
  #pragma HLS stream variable=v350 depth=2	// L427
  hls::stream< int32_t > v351;
  #pragma HLS stream variable=v351 depth=2	// L428
  hls::stream< int8_t > v352;
  #pragma HLS stream variable=v352 depth=2	// L429
  hls::stream< int8_t > v353;
  #pragma HLS stream variable=v353 depth=2	// L430
  hls::stream< int32_t > v354;
  #pragma HLS stream variable=v354 depth=2	// L431
  hls::stream< int8_t > v355;
  #pragma HLS stream variable=v355 depth=2	// L432
  hls::stream< int8_t > v356;
  #pragma HLS stream variable=v356 depth=2	// L433
  hls::stream< int32_t > v357;
  #pragma HLS stream variable=v357 depth=2	// L434
  hls::stream< int8_t > v358;
  #pragma HLS stream variable=v358 depth=2	// L435
  hls::stream< int8_t > v359;
  #pragma HLS stream variable=v359 depth=2	// L436
  hls::stream< int8_t > v360;
  #pragma HLS stream variable=v360 depth=2	// L437
  hls::stream< int32_t > v361;
  #pragma HLS stream variable=v361 depth=2	// L438
  hls::stream< int8_t > v362;
  #pragma HLS stream variable=v362 depth=2	// L439
  hls::stream< int8_t > v363;
  #pragma HLS stream variable=v363 depth=2	// L440
  hls::stream< int32_t > v364;
  #pragma HLS stream variable=v364 depth=2	// L441
  hls::stream< int8_t > v365;
  #pragma HLS stream variable=v365 depth=2	// L443
  hls::stream< int8_t > v366;
  #pragma HLS stream variable=v366 depth=2	// L444
  hls::stream< int8_t > v367;
  #pragma HLS stream variable=v367 depth=2	// L445
  hls::stream< int32_t > v368;
  #pragma HLS stream variable=v368 depth=2	// L446
  hls::stream< int8_t > v369;
  #pragma HLS stream variable=v369 depth=2	// L448
  hls::stream< int8_t > v370;
  #pragma HLS stream variable=v370 depth=2	// L449
  hls::stream< int8_t > v371;
  #pragma HLS stream variable=v371 depth=2	// L450
  hls::stream< int32_t > v372;
  #pragma HLS stream variable=v372 depth=2	// L451
  hls::stream< int8_t > v373;
  #pragma HLS stream variable=v373 depth=2	// L453
  hls::stream< int8_t > v374;
  #pragma HLS stream variable=v374 depth=2	// L454
  hls::stream< int8_t > v375;
  #pragma HLS stream variable=v375 depth=2	// L455
  hls::stream< int8_t > v376;
  #pragma HLS stream variable=v376 depth=2	// L456
  hls::stream< int32_t > v377;
  #pragma HLS stream variable=v377 depth=2	// L457
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v378;
  #pragma HLS stream variable=v378 depth=16	// L458
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v379;
  #pragma HLS stream variable=v379 depth=16	// L459
  feed_up_load(v305, 0, v379);	// L461
  feed_3_up_load(v306, 0, v378);	// L462
  pe_r6(0, 0, v377, v376, v375, v374, v373);	// L463
  pe_r3(0, 1, v372, v371, v370, v369, v376);	// L464
  pe_r3(0, 2, v368, v367, v366, v365, v371);	// L465
  pe_r8(0, 3, v364, v363, v362, v367);	// L466
  pe_r1(1, 0, v361, v360, v374, v359, v358);	// L467
  pe_r0(1, 1, v357, v356, v369, v355, v360);	// L468
  pe_r0(1, 2, v354, v353, v365, v352, v356);	// L469
  pe_r4(1, 3, v351, v362, v350, v353);	// L470
  pe_r1(2, 0, v349, v348, v359, v347, v346);	// L471
  pe_r0(2, 1, v345, v344, v355, v343, v348);	// L472
  pe_r0(2, 2, v342, v341, v352, v340, v344);	// L473
  pe_r4(2, 3, v339, v350, v338, v341);	// L474
  pe_r5(3, 0, v337, v336, v347, v335);	// L475
  pe_r2(3, 1, v334, v333, v343, v336);	// L476
  pe_r2(3, 2, v332, v331, v340, v333);	// L477
  pe_r7(3, 3, v330, v338, v331);	// L478
  drain_r1(0, 0, v329, v377);	// L479
  drain_r1(0, 1, v328, v372);	// L480
  drain_r1(0, 2, v327, v368);	// L481
  drain_r1(0, 3, v326, v364);	// L482
  drain_r0(1, 0, v325, v361, v329);	// L483
  drain_r0(1, 1, v324, v357, v328);	// L484
  drain_r0(1, 2, v323, v354, v327);	// L485
  drain_r0(1, 3, v322, v351, v326);	// L486
  drain_r0(2, 0, v321, v349, v325);	// L487
  drain_r0(2, 1, v320, v345, v324);	// L488
  drain_r0(2, 2, v319, v342, v323);	// L489
  drain_r0(2, 3, v318, v339, v322);	// L490
  drain_r2(3, 0, v317, v337, v321);	// L491
  drain_r2(3, 1, v316, v334, v320);	// L492
  drain_r2(3, 2, v315, v332, v319);	// L493
  drain_r2(3, 3, v314, v330, v318);	// L494
  feed_r1(0, v313, v373, v379);	// L495
  feed_r0(1, v312, v358, v313);	// L496
  feed_r0(2, v311, v346, v312);	// L497
  feed_r2(3, v335, v311);	// L498
  feed_3_r1(0, v310, v375, v378);	// L499
  feed_3_r0(1, v309, v370, v310);	// L500
  feed_3_r0(2, v308, v366, v309);	// L501
  feed_3_r2(3, v363, v308);	// L502
  drain_down_drain(v307, 0, v317);	// L503
  drain_down_drain(v307, 1, v316);	// L504
  drain_down_drain(v307, 2, v315);	// L505
  drain_down_drain(v307, 3, v314);	// L506
}


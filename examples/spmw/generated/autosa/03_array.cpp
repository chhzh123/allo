
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

void feed_2_up_load(
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
  hls::stream< int32_t >& v19,
  hls::stream< int8_t >& v20,
  hls::stream< int8_t >& v21,
  hls::stream< int8_t >& v22,
  hls::stream< int8_t >& v23
) {	// L31
  int32_t acc;	// L33
  acc = 0;	// L34
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L35
  #pragma HLS pipeline II=1
    int8_t v26 = v23.read();	// L36
    int8_t a;	// L37
    a = v26;	// L38
    int8_t v28 = v21.read();	// L39
    int8_t b;	// L40
    b = v28;	// L41
    int8_t v30 = a;	// L42
    int8_t v31 = b;	// L43
    int16_t v32 = v30;	// L44
    int16_t v33 = v31;	// L45
    int16_t v34 = v32 * v33;	// L46
    int32_t v35 = acc;	// L47
    ap_int<33> v36 = v35;	// L48
    ap_int<33> v37 = v34;	// L49
    ap_int<33> v38 = v36 + v37;	// L50
    int32_t v39 = v38;	// L51
    acc = v39;	// L52
    int8_t v40 = a;	// L53
    v20.write(v40);	// L54
    int8_t v41 = b;	// L55
    v22.write(v41);	// L56
  }
  int32_t v42 = acc;	// L58
  v19.write(v42);	// L59
  l_S__i_1__i: for (int _i = 0; _i < v16; _i++) {	// L60
  #pragma HLS pipeline II=1
    int32_t v44 = v18.read();	// L61
    v19.write(v44);	// L62
  }
}

void pe_r1(
  int v45,
  int v46,
  hls::stream< int32_t >& v47,
  hls::stream< int32_t >& v48,
  hls::stream< int8_t >& v49,
  hls::stream< int8_t >& v50,
  hls::stream< int8_t >& v51,
  hls::stream< int8_t >& v52
) {	// L66
  int32_t acc1;	// L68
  acc1 = 0;	// L69
  l_S_k_0_k1: for (int k1 = 0; k1 < 4; k1++) {	// L70
  #pragma HLS pipeline II=1
    int8_t v55 = v52.read();	// L71
    int8_t a1;	// L72
    a1 = v55;	// L73
    int8_t v57 = v50.read();	// L74
    int8_t b1;	// L75
    b1 = v57;	// L76
    int8_t v59 = a1;	// L77
    int8_t v60 = b1;	// L78
    int16_t v61 = v59;	// L79
    int16_t v62 = v60;	// L80
    int16_t v63 = v61 * v62;	// L81
    int32_t v64 = acc1;	// L82
    ap_int<33> v65 = v64;	// L83
    ap_int<33> v66 = v63;	// L84
    ap_int<33> v67 = v65 + v66;	// L85
    int32_t v68 = v67;	// L86
    acc1 = v68;	// L87
    int8_t v69 = a1;	// L88
    v49.write(v69);	// L89
    int8_t v70 = b1;	// L90
    v51.write(v70);	// L91
  }
  int32_t v71 = acc1;	// L93
  v48.write(v71);	// L94
  l_S__i_1__i1: for (int _i1 = 0; _i1 < v45; _i1++) {	// L95
  #pragma HLS pipeline II=1
    int32_t v73 = v47.read();	// L96
    v48.write(v73);	// L97
  }
}

void pe_r2(
  int v74,
  int v75,
  hls::stream< int32_t >& v76,
  hls::stream< int32_t >& v77,
  hls::stream< int8_t >& v78,
  hls::stream< int8_t >& v79,
  hls::stream< int8_t >& v80
) {	// L101
  int32_t acc2;	// L103
  acc2 = 0;	// L104
  l_S_k_0_k2: for (int k2 = 0; k2 < 4; k2++) {	// L105
  #pragma HLS pipeline II=1
    int8_t v83 = v80.read();	// L106
    int8_t a2;	// L107
    a2 = v83;	// L108
    int8_t v85 = v78.read();	// L109
    int8_t b2;	// L110
    b2 = v85;	// L111
    int8_t v87 = a2;	// L112
    int8_t v88 = b2;	// L113
    int16_t v89 = v87;	// L114
    int16_t v90 = v88;	// L115
    int16_t v91 = v89 * v90;	// L116
    int32_t v92 = acc2;	// L117
    ap_int<33> v93 = v92;	// L118
    ap_int<33> v94 = v91;	// L119
    ap_int<33> v95 = v93 + v94;	// L120
    int32_t v96 = v95;	// L121
    acc2 = v96;	// L122
    int8_t v97 = b2;	// L123
    v79.write(v97);	// L124
  }
  int32_t v98 = acc2;	// L126
  v77.write(v98);	// L127
  l_S__i_1__i2: for (int _i2 = 0; _i2 < v74; _i2++) {	// L128
  #pragma HLS pipeline II=1
    int32_t v100 = v76.read();	// L129
    v77.write(v100);	// L130
  }
}

void pe_r3(
  int v101,
  int v102,
  hls::stream< int32_t >& v103,
  hls::stream< int32_t >& v104,
  hls::stream< int8_t >& v105,
  hls::stream< int8_t >& v106,
  hls::stream< int8_t >& v107
) {	// L134
  int32_t acc3;	// L136
  acc3 = 0;	// L137
  l_S_k_0_k3: for (int k3 = 0; k3 < 4; k3++) {	// L138
  #pragma HLS pipeline II=1
    int8_t v110 = v107.read();	// L139
    int8_t a3;	// L140
    a3 = v110;	// L141
    int8_t v112 = v106.read();	// L142
    int8_t b3;	// L143
    b3 = v112;	// L144
    int8_t v114 = a3;	// L145
    int8_t v115 = b3;	// L146
    int16_t v116 = v114;	// L147
    int16_t v117 = v115;	// L148
    int16_t v118 = v116 * v117;	// L149
    int32_t v119 = acc3;	// L150
    ap_int<33> v120 = v119;	// L151
    ap_int<33> v121 = v118;	// L152
    ap_int<33> v122 = v120 + v121;	// L153
    int32_t v123 = v122;	// L154
    acc3 = v123;	// L155
    int8_t v124 = a3;	// L156
    v105.write(v124);	// L157
  }
  int32_t v125 = acc3;	// L159
  v104.write(v125);	// L160
  l_S__i_1__i3: for (int _i3 = 0; _i3 < v101; _i3++) {	// L161
  #pragma HLS pipeline II=1
    int32_t v127 = v103.read();	// L162
    v104.write(v127);	// L163
  }
}

void pe_r4(
  int v128,
  int v129,
  hls::stream< int32_t >& v130,
  hls::stream< int8_t >& v131,
  hls::stream< int8_t >& v132,
  hls::stream< int8_t >& v133,
  hls::stream< int8_t >& v134
) {	// L167
  int32_t acc4;	// L169
  acc4 = 0;	// L170
  l_S_k_0_k4: for (int k4 = 0; k4 < 4; k4++) {	// L171
  #pragma HLS pipeline II=1
    int8_t v137 = v134.read();	// L172
    int8_t a4;	// L173
    a4 = v137;	// L174
    int8_t v139 = v132.read();	// L175
    int8_t b4;	// L176
    b4 = v139;	// L177
    int8_t v141 = a4;	// L178
    int8_t v142 = b4;	// L179
    int16_t v143 = v141;	// L180
    int16_t v144 = v142;	// L181
    int16_t v145 = v143 * v144;	// L182
    int32_t v146 = acc4;	// L183
    ap_int<33> v147 = v146;	// L184
    ap_int<33> v148 = v145;	// L185
    ap_int<33> v149 = v147 + v148;	// L186
    int32_t v150 = v149;	// L187
    acc4 = v150;	// L188
    int8_t v151 = a4;	// L189
    v131.write(v151);	// L190
    int8_t v152 = b4;	// L191
    v133.write(v152);	// L192
  }
  int32_t v153 = acc4;	// L194
  v130.write(v153);	// L195
  l_S__i_1__i4: for (int _i4 = 0; _i4 < v128; _i4++) {	// L196
  #pragma HLS pipeline II=1
    v130.write(0);	// L197
  }
}

void pe_r5(
  int v155,
  int v156,
  hls::stream< int32_t >& v157,
  hls::stream< int32_t >& v158,
  hls::stream< int8_t >& v159,
  hls::stream< int8_t >& v160,
  hls::stream< int8_t >& v161
) {	// L201
  int32_t acc5;	// L203
  acc5 = 0;	// L204
  l_S_k_0_k5: for (int k5 = 0; k5 < 4; k5++) {	// L205
  #pragma HLS pipeline II=1
    int8_t v164 = v161.read();	// L206
    int8_t a5;	// L207
    a5 = v164;	// L208
    int8_t v166 = v160.read();	// L209
    int8_t b5;	// L210
    b5 = v166;	// L211
    int8_t v168 = a5;	// L212
    int8_t v169 = b5;	// L213
    int16_t v170 = v168;	// L214
    int16_t v171 = v169;	// L215
    int16_t v172 = v170 * v171;	// L216
    int32_t v173 = acc5;	// L217
    ap_int<33> v174 = v173;	// L218
    ap_int<33> v175 = v172;	// L219
    ap_int<33> v176 = v174 + v175;	// L220
    int32_t v177 = v176;	// L221
    acc5 = v177;	// L222
    int8_t v178 = a5;	// L223
    v159.write(v178);	// L224
  }
  int32_t v179 = acc5;	// L226
  v158.write(v179);	// L227
  l_S__i_1__i5: for (int _i5 = 0; _i5 < v155; _i5++) {	// L228
  #pragma HLS pipeline II=1
    int32_t v181 = v157.read();	// L229
    v158.write(v181);	// L230
  }
}

void pe_r6(
  int v182,
  int v183,
  hls::stream< int32_t >& v184,
  hls::stream< int32_t >& v185,
  hls::stream< int8_t >& v186,
  hls::stream< int8_t >& v187
) {	// L234
  int32_t acc6;	// L236
  acc6 = 0;	// L237
  l_S_k_0_k6: for (int k6 = 0; k6 < 4; k6++) {	// L238
  #pragma HLS pipeline II=1
    int8_t v190 = v187.read();	// L239
    int8_t a6;	// L240
    a6 = v190;	// L241
    int8_t v192 = v186.read();	// L242
    int8_t b6;	// L243
    b6 = v192;	// L244
    int8_t v194 = a6;	// L245
    int8_t v195 = b6;	// L246
    int16_t v196 = v194;	// L247
    int16_t v197 = v195;	// L248
    int16_t v198 = v196 * v197;	// L249
    int32_t v199 = acc6;	// L250
    ap_int<33> v200 = v199;	// L251
    ap_int<33> v201 = v198;	// L252
    ap_int<33> v202 = v200 + v201;	// L253
    int32_t v203 = v202;	// L254
    acc6 = v203;	// L255
  }
  int32_t v204 = acc6;	// L257
  v185.write(v204);	// L258
  l_S__i_1__i6: for (int _i6 = 0; _i6 < v182; _i6++) {	// L259
  #pragma HLS pipeline II=1
    int32_t v206 = v184.read();	// L260
    v185.write(v206);	// L261
  }
}

void pe_r7(
  int v207,
  int v208,
  hls::stream< int32_t >& v209,
  hls::stream< int8_t >& v210,
  hls::stream< int8_t >& v211,
  hls::stream< int8_t >& v212,
  hls::stream< int8_t >& v213
) {	// L265
  int32_t acc7;	// L267
  acc7 = 0;	// L268
  l_S_k_0_k7: for (int k7 = 0; k7 < 4; k7++) {	// L269
  #pragma HLS pipeline II=1
    int8_t v216 = v213.read();	// L270
    int8_t a7;	// L271
    a7 = v216;	// L272
    int8_t v218 = v211.read();	// L273
    int8_t b7;	// L274
    b7 = v218;	// L275
    int8_t v220 = a7;	// L276
    int8_t v221 = b7;	// L277
    int16_t v222 = v220;	// L278
    int16_t v223 = v221;	// L279
    int16_t v224 = v222 * v223;	// L280
    int32_t v225 = acc7;	// L281
    ap_int<33> v226 = v225;	// L282
    ap_int<33> v227 = v224;	// L283
    ap_int<33> v228 = v226 + v227;	// L284
    int32_t v229 = v228;	// L285
    acc7 = v229;	// L286
    int8_t v230 = a7;	// L287
    v210.write(v230);	// L288
    int8_t v231 = b7;	// L289
    v212.write(v231);	// L290
  }
  int32_t v232 = acc7;	// L292
  v209.write(v232);	// L293
  l_S__i_1__i7: for (int _i7 = 0; _i7 < v207; _i7++) {	// L294
  #pragma HLS pipeline II=1
    v209.write(0);	// L295
  }
}

void pe_r8(
  int v234,
  int v235,
  hls::stream< int32_t >& v236,
  hls::stream< int8_t >& v237,
  hls::stream< int8_t >& v238,
  hls::stream< int8_t >& v239
) {	// L299
  int32_t acc8;	// L301
  acc8 = 0;	// L302
  l_S_k_0_k8: for (int k8 = 0; k8 < 4; k8++) {	// L303
  #pragma HLS pipeline II=1
    int8_t v242 = v239.read();	// L304
    int8_t a8;	// L305
    a8 = v242;	// L306
    int8_t v244 = v237.read();	// L307
    int8_t b8;	// L308
    b8 = v244;	// L309
    int8_t v246 = a8;	// L310
    int8_t v247 = b8;	// L311
    int16_t v248 = v246;	// L312
    int16_t v249 = v247;	// L313
    int16_t v250 = v248 * v249;	// L314
    int32_t v251 = acc8;	// L315
    ap_int<33> v252 = v251;	// L316
    ap_int<33> v253 = v250;	// L317
    ap_int<33> v254 = v252 + v253;	// L318
    int32_t v255 = v254;	// L319
    acc8 = v255;	// L320
    int8_t v256 = b8;	// L321
    v238.write(v256);	// L322
  }
  int32_t v257 = acc8;	// L324
  v236.write(v257);	// L325
  l_S__i_1__i8: for (int _i8 = 0; _i8 < v234; _i8++) {	// L326
  #pragma HLS pipeline II=1
    v236.write(0);	// L327
  }
}

void feed_r0(
  int v259,
  hls::stream< hls::vector< int8_t, 4 > >& v260,
  hls::stream< int8_t >& v261,
  hls::stream< hls::vector< int8_t, 4 > >& v262
) {	// L331
  l_S_k_0_k9: for (int k9 = 0; k9 < 4; k9++) {	// L332
  #pragma HLS pipeline II=1
    int8_t v264[4];
    {
      hls::vector< int8_t, 4 > _vec = v262.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v264[_iv0] = _vec[_iv0];
      }
    }	// L333
    int8_t v265 = v264[v259];	// L334
    v261.write(v265);	// L335
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v264[_iv0];
      }
      v260.write(_vec);
    }	// L336
  }
}

void feed_r1(
  int v266,
  hls::stream< hls::vector< int8_t, 4 > >& v267,
  hls::stream< int8_t >& v268,
  hls::stream< hls::vector< int8_t, 4 > >& v269
) {	// L340
  l_S_k_0_k10: for (int k10 = 0; k10 < 4; k10++) {	// L341
  #pragma HLS pipeline II=1
    int8_t v271[4];
    {
      hls::vector< int8_t, 4 > _vec = v269.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v271[_iv0] = _vec[_iv0];
      }
    }	// L342
    int8_t v272 = v271[v266];	// L343
    v268.write(v272);	// L344
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v271[_iv0];
      }
      v267.write(_vec);
    }	// L345
  }
}

void feed_r2(
  int v273,
  hls::stream< int8_t >& v274,
  hls::stream< hls::vector< int8_t, 4 > >& v275
) {	// L349
  l_S_k_0_k11: for (int k11 = 0; k11 < 4; k11++) {	// L350
  #pragma HLS pipeline II=1
    int8_t v277[4];
    {
      hls::vector< int8_t, 4 > _vec = v275.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v277[_iv0] = _vec[_iv0];
      }
    }	// L351
    int8_t v278 = v277[v273];	// L352
    v274.write(v278);	// L353
  }
}

void feed_2_r0(
  int v279,
  hls::stream< hls::vector< int8_t, 4 > >& v280,
  hls::stream< int8_t >& v281,
  hls::stream< hls::vector< int8_t, 4 > >& v282
) {	// L357
  l_S_k_0_k12: for (int k12 = 0; k12 < 4; k12++) {	// L358
  #pragma HLS pipeline II=1
    int8_t v284[4];
    {
      hls::vector< int8_t, 4 > _vec = v282.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v284[_iv0] = _vec[_iv0];
      }
    }	// L359
    int8_t v285 = v284[v279];	// L360
    v281.write(v285);	// L361
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v284[_iv0];
      }
      v280.write(_vec);
    }	// L362
  }
}

void feed_2_r1(
  int v286,
  hls::stream< hls::vector< int8_t, 4 > >& v287,
  hls::stream< int8_t >& v288,
  hls::stream< hls::vector< int8_t, 4 > >& v289
) {	// L366
  l_S_k_0_k13: for (int k13 = 0; k13 < 4; k13++) {	// L367
  #pragma HLS pipeline II=1
    int8_t v291[4];
    {
      hls::vector< int8_t, 4 > _vec = v289.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v291[_iv0] = _vec[_iv0];
      }
    }	// L368
    int8_t v292 = v291[v286];	// L369
    v288.write(v292);	// L370
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = v291[_iv0];
      }
      v287.write(_vec);
    }	// L371
  }
}

void feed_2_r2(
  int v293,
  hls::stream< int8_t >& v294,
  hls::stream< hls::vector< int8_t, 4 > >& v295
) {	// L375
  l_S_k_0_k14: for (int k14 = 0; k14 < 4; k14++) {	// L376
  #pragma HLS pipeline II=1
    int8_t v297[4];
    {
      hls::vector< int8_t, 4 > _vec = v295.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v297[_iv0] = _vec[_iv0];
      }
    }	// L377
    int8_t v298 = v297[v293];	// L378
    v294.write(v298);	// L379
  }
}

void pe_c_out_drain(
  int32_t v299[4][4],
  int v300,
  hls::stream< int32_t >& v301
) {	// L383
  #pragma HLS array_partition variable=v299 complete dim=1
  #pragma HLS array_partition variable=v299 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L384
  #pragma HLS pipeline II=1
    int32_t v303 = v301.read();	// L385
    v299[v300][_t2] = v303;	// L386
  }
}

/// This is top function.
void top(
  int8_t v304[4][4],
  int8_t v305[4][4],
  int32_t v306[4][4]
) {	// L390
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v304 complete dim=1
  #pragma HLS array_partition variable=v304 complete dim=2

  #pragma HLS array_partition variable=v305 complete dim=1
  #pragma HLS array_partition variable=v305 complete dim=2

  #pragma HLS array_partition variable=v306 complete dim=1
  #pragma HLS array_partition variable=v306 complete dim=2

  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v307;
  #pragma HLS stream variable=v307 depth=2	// L391
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v308;
  #pragma HLS stream variable=v308 depth=2	// L392
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v309;
  #pragma HLS stream variable=v309 depth=2	// L393
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v310;
  #pragma HLS stream variable=v310 depth=2	// L394
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v311;
  #pragma HLS stream variable=v311 depth=2	// L395
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v312;
  #pragma HLS stream variable=v312 depth=2	// L396
  hls::stream< int32_t > v313;
  #pragma HLS stream variable=v313 depth=4	// L397
  hls::stream< int8_t > v314;
  #pragma HLS stream variable=v314 depth=2	// L398
  hls::stream< int32_t > v315;
  #pragma HLS stream variable=v315 depth=4	// L399
  hls::stream< int8_t > v316;
  #pragma HLS stream variable=v316 depth=2	// L400
  hls::stream< int32_t > v317;
  #pragma HLS stream variable=v317 depth=4	// L401
  hls::stream< int8_t > v318;
  #pragma HLS stream variable=v318 depth=2	// L402
  hls::stream< int8_t > v319;
  #pragma HLS stream variable=v319 depth=2	// L403
  hls::stream< int32_t > v320;
  #pragma HLS stream variable=v320 depth=4	// L404
  hls::stream< int8_t > v321;
  #pragma HLS stream variable=v321 depth=2	// L405
  hls::stream< int32_t > v322;
  #pragma HLS stream variable=v322 depth=2	// L406
  hls::stream< int8_t > v323;
  #pragma HLS stream variable=v323 depth=2	// L407
  hls::stream< int8_t > v324;
  #pragma HLS stream variable=v324 depth=2	// L408
  hls::stream< int32_t > v325;
  #pragma HLS stream variable=v325 depth=2	// L409
  hls::stream< int8_t > v326;
  #pragma HLS stream variable=v326 depth=2	// L410
  hls::stream< int8_t > v327;
  #pragma HLS stream variable=v327 depth=2	// L411
  hls::stream< int32_t > v328;
  #pragma HLS stream variable=v328 depth=2	// L412
  hls::stream< int8_t > v329;
  #pragma HLS stream variable=v329 depth=2	// L413
  hls::stream< int8_t > v330;
  #pragma HLS stream variable=v330 depth=2	// L414
  hls::stream< int8_t > v331;
  #pragma HLS stream variable=v331 depth=2	// L415
  hls::stream< int32_t > v332;
  #pragma HLS stream variable=v332 depth=2	// L416
  hls::stream< int8_t > v333;
  #pragma HLS stream variable=v333 depth=2	// L417
  hls::stream< int32_t > v334;
  #pragma HLS stream variable=v334 depth=2	// L418
  hls::stream< int8_t > v335;
  #pragma HLS stream variable=v335 depth=2	// L419
  hls::stream< int8_t > v336;
  #pragma HLS stream variable=v336 depth=2	// L420
  hls::stream< int32_t > v337;
  #pragma HLS stream variable=v337 depth=2	// L421
  hls::stream< int8_t > v338;
  #pragma HLS stream variable=v338 depth=2	// L422
  hls::stream< int8_t > v339;
  #pragma HLS stream variable=v339 depth=2	// L423
  hls::stream< int32_t > v340;
  #pragma HLS stream variable=v340 depth=2	// L424
  hls::stream< int8_t > v341;
  #pragma HLS stream variable=v341 depth=2	// L425
  hls::stream< int8_t > v342;
  #pragma HLS stream variable=v342 depth=2	// L426
  hls::stream< int8_t > v343;
  #pragma HLS stream variable=v343 depth=2	// L427
  hls::stream< int32_t > v344;
  #pragma HLS stream variable=v344 depth=2	// L428
  hls::stream< int8_t > v345;
  #pragma HLS stream variable=v345 depth=2	// L429
  hls::stream< int8_t > v346;
  #pragma HLS stream variable=v346 depth=2	// L430
  hls::stream< int32_t > v347;
  #pragma HLS stream variable=v347 depth=2	// L431
  hls::stream< int8_t > v348;
  #pragma HLS stream variable=v348 depth=2	// L433
  hls::stream< int8_t > v349;
  #pragma HLS stream variable=v349 depth=2	// L434
  hls::stream< int8_t > v350;
  #pragma HLS stream variable=v350 depth=2	// L435
  hls::stream< int32_t > v351;
  #pragma HLS stream variable=v351 depth=2	// L436
  hls::stream< int8_t > v352;
  #pragma HLS stream variable=v352 depth=2	// L438
  hls::stream< int8_t > v353;
  #pragma HLS stream variable=v353 depth=2	// L439
  hls::stream< int8_t > v354;
  #pragma HLS stream variable=v354 depth=2	// L440
  hls::stream< int32_t > v355;
  #pragma HLS stream variable=v355 depth=2	// L441
  hls::stream< int8_t > v356;
  #pragma HLS stream variable=v356 depth=2	// L443
  hls::stream< int8_t > v357;
  #pragma HLS stream variable=v357 depth=2	// L444
  hls::stream< int8_t > v358;
  #pragma HLS stream variable=v358 depth=2	// L445
  hls::stream< int8_t > v359;
  #pragma HLS stream variable=v359 depth=2	// L446
  hls::stream< int32_t > v360;
  #pragma HLS stream variable=v360 depth=2	// L447
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v361;
  #pragma HLS stream variable=v361 depth=16	// L448
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v362;
  #pragma HLS stream variable=v362 depth=16	// L449
  feed_up_load(v304, 0, v362);	// L451
  feed_2_up_load(v305, 0, v361);	// L452
  pe_r7(0, 0, v360, v359, v358, v357, v356);	// L453
  pe_r4(0, 1, v355, v354, v353, v352, v359);	// L454
  pe_r4(0, 2, v351, v350, v349, v348, v354);	// L455
  pe_r8(0, 3, v347, v346, v345, v350);	// L456
  pe_r1(1, 0, v360, v344, v343, v357, v342, v341);	// L457
  pe_r0(1, 1, v355, v340, v339, v352, v338, v343);	// L458
  pe_r0(1, 2, v351, v337, v336, v348, v335, v339);	// L459
  pe_r2(1, 3, v347, v334, v345, v333, v336);	// L460
  pe_r1(2, 0, v344, v332, v331, v342, v330, v329);	// L461
  pe_r0(2, 1, v340, v328, v327, v338, v326, v331);	// L462
  pe_r0(2, 2, v337, v325, v324, v335, v323, v327);	// L463
  pe_r2(2, 3, v334, v322, v333, v321, v324);	// L464
  pe_r5(3, 0, v332, v320, v319, v330, v318);	// L465
  pe_r3(3, 1, v328, v317, v316, v326, v319);	// L466
  pe_r3(3, 2, v325, v315, v314, v323, v316);	// L467
  pe_r6(3, 3, v322, v313, v321, v314);	// L468
  feed_r1(0, v312, v356, v362);	// L469
  feed_r0(1, v311, v341, v312);	// L470
  feed_r0(2, v310, v329, v311);	// L471
  feed_r2(3, v318, v310);	// L472
  feed_2_r1(0, v309, v358, v361);	// L473
  feed_2_r0(1, v308, v353, v309);	// L474
  feed_2_r0(2, v307, v349, v308);	// L475
  feed_2_r2(3, v346, v307);	// L476
  pe_c_out_drain(v306, 0, v320);	// L477
  pe_c_out_drain(v306, 1, v317);	// L478
  pe_c_out_drain(v306, 2, v315);	// L479
  pe_c_out_drain(v306, 3, v313);	// L480
}


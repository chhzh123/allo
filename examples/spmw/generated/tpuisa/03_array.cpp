
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
void mac_a_in_load_0(
  int8_t v0[4][4],
  hls::stream< int8_t >& v1
) {	// L5
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L6
  #pragma HLS pipeline II=1
    int8_t v3 = v0[_t][0];	// L7
    v1.write(v3);	// L8
  }
}

void mac_a_in_load_1(
  int8_t v4[4][4],
  hls::stream< int8_t >& v5
) {	// L12
  #pragma HLS array_partition variable=v4 complete dim=1
  #pragma HLS array_partition variable=v4 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 4; _t1++) {	// L13
  #pragma HLS pipeline II=1
    int8_t v7 = v4[_t1][1];	// L14
    v5.write(v7);	// L15
  }
}

void mac_a_in_load_2(
  int8_t v8[4][4],
  hls::stream< int8_t >& v9
) {	// L19
  #pragma HLS array_partition variable=v8 complete dim=1
  #pragma HLS array_partition variable=v8 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 4; _t2++) {	// L20
  #pragma HLS pipeline II=1
    int8_t v11 = v8[_t2][2];	// L21
    v9.write(v11);	// L22
  }
}

void mac_a_in_load_3(
  int8_t v12[4][4],
  hls::stream< int8_t >& v13
) {	// L26
  #pragma HLS array_partition variable=v12 complete dim=1
  #pragma HLS array_partition variable=v12 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 4; _t3++) {	// L27
  #pragma HLS pipeline II=1
    int8_t v15 = v12[_t3][3];	// L28
    v13.write(v15);	// L29
  }
}

void mac_op_in_load_0(
  int32_t v16[4][4],
  hls::stream< int32_t >& v17
) {	// L33
  #pragma HLS array_partition variable=v16 complete dim=1
  #pragma HLS array_partition variable=v16 complete dim=2

  l_S__t_0__t4: for (int _t4 = 0; _t4 < 4; _t4++) {	// L34
  #pragma HLS pipeline II=1
    int32_t v19 = v16[_t4][0];	// L35
    v17.write(v19);	// L36
  }
}

void mac_op_in_load_1(
  int32_t v20[4][4],
  hls::stream< int32_t >& v21
) {	// L40
  #pragma HLS array_partition variable=v20 complete dim=1
  #pragma HLS array_partition variable=v20 complete dim=2

  l_S__t_0__t5: for (int _t5 = 0; _t5 < 4; _t5++) {	// L41
  #pragma HLS pipeline II=1
    int32_t v23 = v20[_t5][1];	// L42
    v21.write(v23);	// L43
  }
}

void mac_op_in_load_2(
  int32_t v24[4][4],
  hls::stream< int32_t >& v25
) {	// L47
  #pragma HLS array_partition variable=v24 complete dim=1
  #pragma HLS array_partition variable=v24 complete dim=2

  l_S__t_0__t6: for (int _t6 = 0; _t6 < 4; _t6++) {	// L48
  #pragma HLS pipeline II=1
    int32_t v27 = v24[_t6][2];	// L49
    v25.write(v27);	// L50
  }
}

void mac_op_in_load_3(
  int32_t v28[4][4],
  hls::stream< int32_t >& v29
) {	// L54
  #pragma HLS array_partition variable=v28 complete dim=1
  #pragma HLS array_partition variable=v28 complete dim=2

  l_S__t_0__t7: for (int _t7 = 0; _t7 < 4; _t7++) {	// L55
  #pragma HLS pipeline II=1
    int32_t v31 = v28[_t7][3];	// L56
    v29.write(v31);	// L57
  }
}

void vpu_op_in_load_0(
  int32_t v32[16],
  hls::stream< int32_t >& v33
) {	// L61
  #pragma HLS array_partition variable=v32 complete dim=1

  l_S__t_0__t8: for (int _t8 = 0; _t8 < 16; _t8++) {	// L62
  #pragma HLS pipeline II=1
    int32_t v35 = v32[_t8];	// L63
    v33.write(v35);	// L64
  }
}

void mac_0_0(
  int8_t v36[4][4][4],
  hls::stream< int32_t >& v37,
  hls::stream< int32_t >& v38,
  hls::stream< int8_t >& v39,
  hls::stream< int8_t >& v40,
  hls::stream< int32_t >& v41
) {	// L68
  #pragma HLS array_partition variable=v36 complete dim=1
  #pragma HLS array_partition variable=v36 complete dim=2
  #pragma HLS array_partition variable=v36 complete dim=3

  l_S_step_0_step: for (int step = 0; step < 4; step++) {	// L76
  #pragma HLS pipeline II=1
    int32_t v43 = v37.read();	// L77
    int32_t word;	// L78
    word = v43;	// L79
    int32_t v45 = word;	// L80
    v38.write(v45);	// L81
    int32_t v46 = word;	// L82
    int32_t v47 = v46 >> 24;	// L83
    int32_t v48 = v47 & 255;	// L84
    int32_t opcode;	// L85
    opcode = v48;	// L86
    int32_t v50 = word;	// L87
    int32_t v51 = v50 >> 16;	// L88
    int32_t v52 = v51 & 255;	// L89
    int32_t tile;	// L90
    tile = v52;	// L91
    int8_t v54 = v39.read();	// L92
    int8_t a;	// L93
    a = v54;	// L94
    int32_t p;	// L95
    p = 0;	// L96
    int8_t v57 = a;	// L97
    v40.write(v57);	// L98
    int32_t v58 = tile;	// L99
    int v59 = v58;	// L100
    int8_t v60 = v36[0][0][v59];	// L101
    int32_t v61 = v60;	// L102
    int32_t wt;	// L103
    wt = v61;	// L104
    int32_t v63 = opcode;	// L105
    bool v64 = v63 == 1;	// L106
    if (v64) {	// L107
      int32_t v65 = p;	// L108
      int8_t v66 = a;	// L109
      int32_t v67 = wt;	// L110
      ap_int<40> v68 = v66;	// L111
      ap_int<40> v69 = v67;	// L112
      ap_int<40> v70 = v68 * v69;	// L113
      ap_int<41> v71 = v65;	// L114
      ap_int<41> v72 = v70;	// L115
      ap_int<41> v73 = v71 + v72;	// L116
      v41.write(v73);	// L117
    } else {
      int32_t v74 = opcode;	// L119
      bool v75 = v74 == 2;	// L120
      if (v75) {	// L121
        int8_t v76 = a;	// L122
        int32_t v77 = wt;	// L123
        ap_int<40> v78 = v76;	// L124
        ap_int<40> v79 = v77;	// L125
        ap_int<40> v80 = v78 * v79;	// L126
        v41.write(v80);	// L127
      } else {
        int32_t v81 = p;	// L129
        v41.write(v81);	// L130
      }
    }
  }
}

void mac_0_1(
  int8_t v82[4][4][4],
  hls::stream< int32_t >& v83,
  hls::stream< int32_t >& v84,
  hls::stream< int8_t >& v85,
  hls::stream< int8_t >& v86,
  hls::stream< int32_t >& v87
) {	// L136
  #pragma HLS array_partition variable=v82 complete dim=1
  #pragma HLS array_partition variable=v82 complete dim=2
  #pragma HLS array_partition variable=v82 complete dim=3

  l_S_step_0_step1: for (int step1 = 0; step1 < 4; step1++) {	// L145
  #pragma HLS pipeline II=1
    int32_t v89 = v83.read();	// L146
    int32_t word1;	// L147
    word1 = v89;	// L148
    int32_t v91 = word1;	// L149
    v84.write(v91);	// L150
    int32_t v92 = word1;	// L151
    int32_t v93 = v92 >> 24;	// L152
    int32_t v94 = v93 & 255;	// L153
    int32_t opcode1;	// L154
    opcode1 = v94;	// L155
    int32_t v96 = word1;	// L156
    int32_t v97 = v96 >> 16;	// L157
    int32_t v98 = v97 & 255;	// L158
    int32_t tile1;	// L159
    tile1 = v98;	// L160
    int8_t v100 = v85.read();	// L161
    int8_t a1;	// L162
    a1 = v100;	// L163
    int32_t p1;	// L164
    p1 = 0;	// L165
    int8_t v103 = a1;	// L166
    v86.write(v103);	// L167
    int32_t v104 = tile1;	// L168
    int v105 = v104;	// L169
    int8_t v106 = v82[0][1][v105];	// L170
    int32_t v107 = v106;	// L171
    int32_t wt1;	// L172
    wt1 = v107;	// L173
    int32_t v109 = opcode1;	// L174
    bool v110 = v109 == 1;	// L175
    if (v110) {	// L176
      int32_t v111 = p1;	// L177
      int8_t v112 = a1;	// L178
      int32_t v113 = wt1;	// L179
      ap_int<40> v114 = v112;	// L180
      ap_int<40> v115 = v113;	// L181
      ap_int<40> v116 = v114 * v115;	// L182
      ap_int<41> v117 = v111;	// L183
      ap_int<41> v118 = v116;	// L184
      ap_int<41> v119 = v117 + v118;	// L185
      v87.write(v119);	// L186
    } else {
      int32_t v120 = opcode1;	// L188
      bool v121 = v120 == 2;	// L189
      if (v121) {	// L190
        int8_t v122 = a1;	// L191
        int32_t v123 = wt1;	// L192
        ap_int<40> v124 = v122;	// L193
        ap_int<40> v125 = v123;	// L194
        ap_int<40> v126 = v124 * v125;	// L195
        v87.write(v126);	// L196
      } else {
        int32_t v127 = p1;	// L198
        v87.write(v127);	// L199
      }
    }
  }
}

void mac_0_2(
  int8_t v128[4][4][4],
  hls::stream< int32_t >& v129,
  hls::stream< int32_t >& v130,
  hls::stream< int8_t >& v131,
  hls::stream< int8_t >& v132,
  hls::stream< int32_t >& v133
) {	// L205
  #pragma HLS array_partition variable=v128 complete dim=1
  #pragma HLS array_partition variable=v128 complete dim=2
  #pragma HLS array_partition variable=v128 complete dim=3

  l_S_step_0_step2: for (int step2 = 0; step2 < 4; step2++) {	// L214
  #pragma HLS pipeline II=1
    int32_t v135 = v129.read();	// L215
    int32_t word2;	// L216
    word2 = v135;	// L217
    int32_t v137 = word2;	// L218
    v130.write(v137);	// L219
    int32_t v138 = word2;	// L220
    int32_t v139 = v138 >> 24;	// L221
    int32_t v140 = v139 & 255;	// L222
    int32_t opcode2;	// L223
    opcode2 = v140;	// L224
    int32_t v142 = word2;	// L225
    int32_t v143 = v142 >> 16;	// L226
    int32_t v144 = v143 & 255;	// L227
    int32_t tile2;	// L228
    tile2 = v144;	// L229
    int8_t v146 = v131.read();	// L230
    int8_t a2;	// L231
    a2 = v146;	// L232
    int32_t p2;	// L233
    p2 = 0;	// L234
    int8_t v149 = a2;	// L235
    v132.write(v149);	// L236
    int32_t v150 = tile2;	// L237
    int v151 = v150;	// L238
    int8_t v152 = v128[0][2][v151];	// L239
    int32_t v153 = v152;	// L240
    int32_t wt2;	// L241
    wt2 = v153;	// L242
    int32_t v155 = opcode2;	// L243
    bool v156 = v155 == 1;	// L244
    if (v156) {	// L245
      int32_t v157 = p2;	// L246
      int8_t v158 = a2;	// L247
      int32_t v159 = wt2;	// L248
      ap_int<40> v160 = v158;	// L249
      ap_int<40> v161 = v159;	// L250
      ap_int<40> v162 = v160 * v161;	// L251
      ap_int<41> v163 = v157;	// L252
      ap_int<41> v164 = v162;	// L253
      ap_int<41> v165 = v163 + v164;	// L254
      v133.write(v165);	// L255
    } else {
      int32_t v166 = opcode2;	// L257
      bool v167 = v166 == 2;	// L258
      if (v167) {	// L259
        int8_t v168 = a2;	// L260
        int32_t v169 = wt2;	// L261
        ap_int<40> v170 = v168;	// L262
        ap_int<40> v171 = v169;	// L263
        ap_int<40> v172 = v170 * v171;	// L264
        v133.write(v172);	// L265
      } else {
        int32_t v173 = p2;	// L267
        v133.write(v173);	// L268
      }
    }
  }
}

void mac_0_3(
  int8_t v174[4][4][4],
  hls::stream< int32_t >& v175,
  hls::stream< int8_t >& v176,
  hls::stream< int32_t >& v177
) {	// L274
  #pragma HLS array_partition variable=v174 complete dim=1
  #pragma HLS array_partition variable=v174 complete dim=2
  #pragma HLS array_partition variable=v174 complete dim=3

  l_S_step_0_step3: for (int step3 = 0; step3 < 4; step3++) {	// L283
  #pragma HLS pipeline II=1
    int32_t v179 = v175.read();	// L284
    int32_t word3;	// L285
    word3 = v179;	// L286
    int32_t v181 = word3;	// L287
    int32_t v182 = v181 >> 24;	// L288
    int32_t v183 = v182 & 255;	// L289
    int32_t opcode3;	// L290
    opcode3 = v183;	// L291
    int32_t v185 = word3;	// L292
    int32_t v186 = v185 >> 16;	// L293
    int32_t v187 = v186 & 255;	// L294
    int32_t tile3;	// L295
    tile3 = v187;	// L296
    int8_t v189 = v176.read();	// L297
    int8_t a3;	// L298
    a3 = v189;	// L299
    int32_t p3;	// L300
    p3 = 0;	// L301
    int32_t v192 = tile3;	// L302
    int v193 = v192;	// L303
    int8_t v194 = v174[0][3][v193];	// L304
    int32_t v195 = v194;	// L305
    int32_t wt3;	// L306
    wt3 = v195;	// L307
    int32_t v197 = opcode3;	// L308
    bool v198 = v197 == 1;	// L309
    if (v198) {	// L310
      int32_t v199 = p3;	// L311
      int8_t v200 = a3;	// L312
      int32_t v201 = wt3;	// L313
      ap_int<40> v202 = v200;	// L314
      ap_int<40> v203 = v201;	// L315
      ap_int<40> v204 = v202 * v203;	// L316
      ap_int<41> v205 = v199;	// L317
      ap_int<41> v206 = v204;	// L318
      ap_int<41> v207 = v205 + v206;	// L319
      v177.write(v207);	// L320
    } else {
      int32_t v208 = opcode3;	// L322
      bool v209 = v208 == 2;	// L323
      if (v209) {	// L324
        int8_t v210 = a3;	// L325
        int32_t v211 = wt3;	// L326
        ap_int<40> v212 = v210;	// L327
        ap_int<40> v213 = v211;	// L328
        ap_int<40> v214 = v212 * v213;	// L329
        v177.write(v214);	// L330
      } else {
        int32_t v215 = p3;	// L332
        v177.write(v215);	// L333
      }
    }
  }
}

void mac_1_0(
  int8_t v216[4][4][4],
  hls::stream< int32_t >& v217,
  hls::stream< int32_t >& v218,
  hls::stream< int8_t >& v219,
  hls::stream< int32_t >& v220,
  hls::stream< int8_t >& v221,
  hls::stream< int32_t >& v222
) {	// L339
  #pragma HLS array_partition variable=v216 complete dim=1
  #pragma HLS array_partition variable=v216 complete dim=2
  #pragma HLS array_partition variable=v216 complete dim=3

  l_S_step_0_step4: for (int step4 = 0; step4 < 4; step4++) {	// L347
  #pragma HLS pipeline II=1
    int32_t v224 = v217.read();	// L348
    int32_t word4;	// L349
    word4 = v224;	// L350
    int32_t v226 = word4;	// L351
    v218.write(v226);	// L352
    int32_t v227 = word4;	// L353
    int32_t v228 = v227 >> 24;	// L354
    int32_t v229 = v228 & 255;	// L355
    int32_t opcode4;	// L356
    opcode4 = v229;	// L357
    int32_t v231 = word4;	// L358
    int32_t v232 = v231 >> 16;	// L359
    int32_t v233 = v232 & 255;	// L360
    int32_t tile4;	// L361
    tile4 = v233;	// L362
    int8_t v235 = v219.read();	// L363
    int8_t a4;	// L364
    a4 = v235;	// L365
    int32_t v237 = v220.read();	// L366
    int32_t p4;	// L367
    p4 = v237;	// L368
    int8_t v239 = a4;	// L369
    v221.write(v239);	// L370
    int32_t v240 = tile4;	// L371
    int v241 = v240;	// L372
    int8_t v242 = v216[1][0][v241];	// L373
    int32_t v243 = v242;	// L374
    int32_t wt4;	// L375
    wt4 = v243;	// L376
    int32_t v245 = opcode4;	// L377
    bool v246 = v245 == 1;	// L378
    if (v246) {	// L379
      int32_t v247 = p4;	// L380
      int8_t v248 = a4;	// L381
      int32_t v249 = wt4;	// L382
      ap_int<40> v250 = v248;	// L383
      ap_int<40> v251 = v249;	// L384
      ap_int<40> v252 = v250 * v251;	// L385
      ap_int<41> v253 = v247;	// L386
      ap_int<41> v254 = v252;	// L387
      ap_int<41> v255 = v253 + v254;	// L388
      v222.write(v255);	// L389
    } else {
      int32_t v256 = opcode4;	// L391
      bool v257 = v256 == 2;	// L392
      if (v257) {	// L393
        int8_t v258 = a4;	// L394
        int32_t v259 = wt4;	// L395
        ap_int<40> v260 = v258;	// L396
        ap_int<40> v261 = v259;	// L397
        ap_int<40> v262 = v260 * v261;	// L398
        v222.write(v262);	// L399
      } else {
        int32_t v263 = p4;	// L401
        v222.write(v263);	// L402
      }
    }
  }
}

void mac_1_1(
  int8_t v264[4][4][4],
  hls::stream< int32_t >& v265,
  hls::stream< int32_t >& v266,
  hls::stream< int8_t >& v267,
  hls::stream< int32_t >& v268,
  hls::stream< int8_t >& v269,
  hls::stream< int32_t >& v270
) {	// L408
  #pragma HLS array_partition variable=v264 complete dim=1
  #pragma HLS array_partition variable=v264 complete dim=2
  #pragma HLS array_partition variable=v264 complete dim=3

  l_S_step_0_step5: for (int step5 = 0; step5 < 4; step5++) {	// L415
  #pragma HLS pipeline II=1
    int32_t v272 = v265.read();	// L416
    int32_t word5;	// L417
    word5 = v272;	// L418
    int32_t v274 = word5;	// L419
    v266.write(v274);	// L420
    int32_t v275 = word5;	// L421
    int32_t v276 = v275 >> 24;	// L422
    int32_t v277 = v276 & 255;	// L423
    int32_t opcode5;	// L424
    opcode5 = v277;	// L425
    int32_t v279 = word5;	// L426
    int32_t v280 = v279 >> 16;	// L427
    int32_t v281 = v280 & 255;	// L428
    int32_t tile5;	// L429
    tile5 = v281;	// L430
    int8_t v283 = v267.read();	// L431
    int8_t a5;	// L432
    a5 = v283;	// L433
    int32_t v285 = v268.read();	// L434
    int32_t p5;	// L435
    p5 = v285;	// L436
    int8_t v287 = a5;	// L437
    v269.write(v287);	// L438
    int32_t v288 = tile5;	// L439
    int v289 = v288;	// L440
    int8_t v290 = v264[1][1][v289];	// L441
    int32_t v291 = v290;	// L442
    int32_t wt5;	// L443
    wt5 = v291;	// L444
    int32_t v293 = opcode5;	// L445
    bool v294 = v293 == 1;	// L446
    if (v294) {	// L447
      int32_t v295 = p5;	// L448
      int8_t v296 = a5;	// L449
      int32_t v297 = wt5;	// L450
      ap_int<40> v298 = v296;	// L451
      ap_int<40> v299 = v297;	// L452
      ap_int<40> v300 = v298 * v299;	// L453
      ap_int<41> v301 = v295;	// L454
      ap_int<41> v302 = v300;	// L455
      ap_int<41> v303 = v301 + v302;	// L456
      v270.write(v303);	// L457
    } else {
      int32_t v304 = opcode5;	// L459
      bool v305 = v304 == 2;	// L460
      if (v305) {	// L461
        int8_t v306 = a5;	// L462
        int32_t v307 = wt5;	// L463
        ap_int<40> v308 = v306;	// L464
        ap_int<40> v309 = v307;	// L465
        ap_int<40> v310 = v308 * v309;	// L466
        v270.write(v310);	// L467
      } else {
        int32_t v311 = p5;	// L469
        v270.write(v311);	// L470
      }
    }
  }
}

void mac_1_2(
  int8_t v312[4][4][4],
  hls::stream< int32_t >& v313,
  hls::stream< int32_t >& v314,
  hls::stream< int8_t >& v315,
  hls::stream< int32_t >& v316,
  hls::stream< int8_t >& v317,
  hls::stream< int32_t >& v318
) {	// L476
  #pragma HLS array_partition variable=v312 complete dim=1
  #pragma HLS array_partition variable=v312 complete dim=2
  #pragma HLS array_partition variable=v312 complete dim=3

  l_S_step_0_step6: for (int step6 = 0; step6 < 4; step6++) {	// L484
  #pragma HLS pipeline II=1
    int32_t v320 = v313.read();	// L485
    int32_t word6;	// L486
    word6 = v320;	// L487
    int32_t v322 = word6;	// L488
    v314.write(v322);	// L489
    int32_t v323 = word6;	// L490
    int32_t v324 = v323 >> 24;	// L491
    int32_t v325 = v324 & 255;	// L492
    int32_t opcode6;	// L493
    opcode6 = v325;	// L494
    int32_t v327 = word6;	// L495
    int32_t v328 = v327 >> 16;	// L496
    int32_t v329 = v328 & 255;	// L497
    int32_t tile6;	// L498
    tile6 = v329;	// L499
    int8_t v331 = v315.read();	// L500
    int8_t a6;	// L501
    a6 = v331;	// L502
    int32_t v333 = v316.read();	// L503
    int32_t p6;	// L504
    p6 = v333;	// L505
    int8_t v335 = a6;	// L506
    v317.write(v335);	// L507
    int32_t v336 = tile6;	// L508
    int v337 = v336;	// L509
    int8_t v338 = v312[1][2][v337];	// L510
    int32_t v339 = v338;	// L511
    int32_t wt6;	// L512
    wt6 = v339;	// L513
    int32_t v341 = opcode6;	// L514
    bool v342 = v341 == 1;	// L515
    if (v342) {	// L516
      int32_t v343 = p6;	// L517
      int8_t v344 = a6;	// L518
      int32_t v345 = wt6;	// L519
      ap_int<40> v346 = v344;	// L520
      ap_int<40> v347 = v345;	// L521
      ap_int<40> v348 = v346 * v347;	// L522
      ap_int<41> v349 = v343;	// L523
      ap_int<41> v350 = v348;	// L524
      ap_int<41> v351 = v349 + v350;	// L525
      v318.write(v351);	// L526
    } else {
      int32_t v352 = opcode6;	// L528
      bool v353 = v352 == 2;	// L529
      if (v353) {	// L530
        int8_t v354 = a6;	// L531
        int32_t v355 = wt6;	// L532
        ap_int<40> v356 = v354;	// L533
        ap_int<40> v357 = v355;	// L534
        ap_int<40> v358 = v356 * v357;	// L535
        v318.write(v358);	// L536
      } else {
        int32_t v359 = p6;	// L538
        v318.write(v359);	// L539
      }
    }
  }
}

void mac_1_3(
  int8_t v360[4][4][4],
  hls::stream< int32_t >& v361,
  hls::stream< int8_t >& v362,
  hls::stream< int32_t >& v363,
  hls::stream< int32_t >& v364
) {	// L545
  #pragma HLS array_partition variable=v360 complete dim=1
  #pragma HLS array_partition variable=v360 complete dim=2
  #pragma HLS array_partition variable=v360 complete dim=3

  l_S_step_0_step7: for (int step7 = 0; step7 < 4; step7++) {	// L553
  #pragma HLS pipeline II=1
    int32_t v366 = v361.read();	// L554
    int32_t word7;	// L555
    word7 = v366;	// L556
    int32_t v368 = word7;	// L557
    int32_t v369 = v368 >> 24;	// L558
    int32_t v370 = v369 & 255;	// L559
    int32_t opcode7;	// L560
    opcode7 = v370;	// L561
    int32_t v372 = word7;	// L562
    int32_t v373 = v372 >> 16;	// L563
    int32_t v374 = v373 & 255;	// L564
    int32_t tile7;	// L565
    tile7 = v374;	// L566
    int8_t v376 = v362.read();	// L567
    int8_t a7;	// L568
    a7 = v376;	// L569
    int32_t v378 = v363.read();	// L570
    int32_t p7;	// L571
    p7 = v378;	// L572
    int32_t v380 = tile7;	// L573
    int v381 = v380;	// L574
    int8_t v382 = v360[1][3][v381];	// L575
    int32_t v383 = v382;	// L576
    int32_t wt7;	// L577
    wt7 = v383;	// L578
    int32_t v385 = opcode7;	// L579
    bool v386 = v385 == 1;	// L580
    if (v386) {	// L581
      int32_t v387 = p7;	// L582
      int8_t v388 = a7;	// L583
      int32_t v389 = wt7;	// L584
      ap_int<40> v390 = v388;	// L585
      ap_int<40> v391 = v389;	// L586
      ap_int<40> v392 = v390 * v391;	// L587
      ap_int<41> v393 = v387;	// L588
      ap_int<41> v394 = v392;	// L589
      ap_int<41> v395 = v393 + v394;	// L590
      v364.write(v395);	// L591
    } else {
      int32_t v396 = opcode7;	// L593
      bool v397 = v396 == 2;	// L594
      if (v397) {	// L595
        int8_t v398 = a7;	// L596
        int32_t v399 = wt7;	// L597
        ap_int<40> v400 = v398;	// L598
        ap_int<40> v401 = v399;	// L599
        ap_int<40> v402 = v400 * v401;	// L600
        v364.write(v402);	// L601
      } else {
        int32_t v403 = p7;	// L603
        v364.write(v403);	// L604
      }
    }
  }
}

void mac_2_0(
  int8_t v404[4][4][4],
  hls::stream< int32_t >& v405,
  hls::stream< int32_t >& v406,
  hls::stream< int8_t >& v407,
  hls::stream< int32_t >& v408,
  hls::stream< int8_t >& v409,
  hls::stream< int32_t >& v410
) {	// L610
  #pragma HLS array_partition variable=v404 complete dim=1
  #pragma HLS array_partition variable=v404 complete dim=2
  #pragma HLS array_partition variable=v404 complete dim=3

  l_S_step_0_step8: for (int step8 = 0; step8 < 4; step8++) {	// L618
  #pragma HLS pipeline II=1
    int32_t v412 = v405.read();	// L619
    int32_t word8;	// L620
    word8 = v412;	// L621
    int32_t v414 = word8;	// L622
    v406.write(v414);	// L623
    int32_t v415 = word8;	// L624
    int32_t v416 = v415 >> 24;	// L625
    int32_t v417 = v416 & 255;	// L626
    int32_t opcode8;	// L627
    opcode8 = v417;	// L628
    int32_t v419 = word8;	// L629
    int32_t v420 = v419 >> 16;	// L630
    int32_t v421 = v420 & 255;	// L631
    int32_t tile8;	// L632
    tile8 = v421;	// L633
    int8_t v423 = v407.read();	// L634
    int8_t a8;	// L635
    a8 = v423;	// L636
    int32_t v425 = v408.read();	// L637
    int32_t p8;	// L638
    p8 = v425;	// L639
    int8_t v427 = a8;	// L640
    v409.write(v427);	// L641
    int32_t v428 = tile8;	// L642
    int v429 = v428;	// L643
    int8_t v430 = v404[2][0][v429];	// L644
    int32_t v431 = v430;	// L645
    int32_t wt8;	// L646
    wt8 = v431;	// L647
    int32_t v433 = opcode8;	// L648
    bool v434 = v433 == 1;	// L649
    if (v434) {	// L650
      int32_t v435 = p8;	// L651
      int8_t v436 = a8;	// L652
      int32_t v437 = wt8;	// L653
      ap_int<40> v438 = v436;	// L654
      ap_int<40> v439 = v437;	// L655
      ap_int<40> v440 = v438 * v439;	// L656
      ap_int<41> v441 = v435;	// L657
      ap_int<41> v442 = v440;	// L658
      ap_int<41> v443 = v441 + v442;	// L659
      v410.write(v443);	// L660
    } else {
      int32_t v444 = opcode8;	// L662
      bool v445 = v444 == 2;	// L663
      if (v445) {	// L664
        int8_t v446 = a8;	// L665
        int32_t v447 = wt8;	// L666
        ap_int<40> v448 = v446;	// L667
        ap_int<40> v449 = v447;	// L668
        ap_int<40> v450 = v448 * v449;	// L669
        v410.write(v450);	// L670
      } else {
        int32_t v451 = p8;	// L672
        v410.write(v451);	// L673
      }
    }
  }
}

void mac_2_1(
  int8_t v452[4][4][4],
  hls::stream< int32_t >& v453,
  hls::stream< int32_t >& v454,
  hls::stream< int8_t >& v455,
  hls::stream< int32_t >& v456,
  hls::stream< int8_t >& v457,
  hls::stream< int32_t >& v458
) {	// L679
  #pragma HLS array_partition variable=v452 complete dim=1
  #pragma HLS array_partition variable=v452 complete dim=2
  #pragma HLS array_partition variable=v452 complete dim=3

  l_S_step_0_step9: for (int step9 = 0; step9 < 4; step9++) {	// L687
  #pragma HLS pipeline II=1
    int32_t v460 = v453.read();	// L688
    int32_t word9;	// L689
    word9 = v460;	// L690
    int32_t v462 = word9;	// L691
    v454.write(v462);	// L692
    int32_t v463 = word9;	// L693
    int32_t v464 = v463 >> 24;	// L694
    int32_t v465 = v464 & 255;	// L695
    int32_t opcode9;	// L696
    opcode9 = v465;	// L697
    int32_t v467 = word9;	// L698
    int32_t v468 = v467 >> 16;	// L699
    int32_t v469 = v468 & 255;	// L700
    int32_t tile9;	// L701
    tile9 = v469;	// L702
    int8_t v471 = v455.read();	// L703
    int8_t a9;	// L704
    a9 = v471;	// L705
    int32_t v473 = v456.read();	// L706
    int32_t p9;	// L707
    p9 = v473;	// L708
    int8_t v475 = a9;	// L709
    v457.write(v475);	// L710
    int32_t v476 = tile9;	// L711
    int v477 = v476;	// L712
    int8_t v478 = v452[2][1][v477];	// L713
    int32_t v479 = v478;	// L714
    int32_t wt9;	// L715
    wt9 = v479;	// L716
    int32_t v481 = opcode9;	// L717
    bool v482 = v481 == 1;	// L718
    if (v482) {	// L719
      int32_t v483 = p9;	// L720
      int8_t v484 = a9;	// L721
      int32_t v485 = wt9;	// L722
      ap_int<40> v486 = v484;	// L723
      ap_int<40> v487 = v485;	// L724
      ap_int<40> v488 = v486 * v487;	// L725
      ap_int<41> v489 = v483;	// L726
      ap_int<41> v490 = v488;	// L727
      ap_int<41> v491 = v489 + v490;	// L728
      v458.write(v491);	// L729
    } else {
      int32_t v492 = opcode9;	// L731
      bool v493 = v492 == 2;	// L732
      if (v493) {	// L733
        int8_t v494 = a9;	// L734
        int32_t v495 = wt9;	// L735
        ap_int<40> v496 = v494;	// L736
        ap_int<40> v497 = v495;	// L737
        ap_int<40> v498 = v496 * v497;	// L738
        v458.write(v498);	// L739
      } else {
        int32_t v499 = p9;	// L741
        v458.write(v499);	// L742
      }
    }
  }
}

void mac_2_2(
  int8_t v500[4][4][4],
  hls::stream< int32_t >& v501,
  hls::stream< int32_t >& v502,
  hls::stream< int8_t >& v503,
  hls::stream< int32_t >& v504,
  hls::stream< int8_t >& v505,
  hls::stream< int32_t >& v506
) {	// L748
  #pragma HLS array_partition variable=v500 complete dim=1
  #pragma HLS array_partition variable=v500 complete dim=2
  #pragma HLS array_partition variable=v500 complete dim=3

  l_S_step_0_step10: for (int step10 = 0; step10 < 4; step10++) {	// L755
  #pragma HLS pipeline II=1
    int32_t v508 = v501.read();	// L756
    int32_t word10;	// L757
    word10 = v508;	// L758
    int32_t v510 = word10;	// L759
    v502.write(v510);	// L760
    int32_t v511 = word10;	// L761
    int32_t v512 = v511 >> 24;	// L762
    int32_t v513 = v512 & 255;	// L763
    int32_t opcode10;	// L764
    opcode10 = v513;	// L765
    int32_t v515 = word10;	// L766
    int32_t v516 = v515 >> 16;	// L767
    int32_t v517 = v516 & 255;	// L768
    int32_t tile10;	// L769
    tile10 = v517;	// L770
    int8_t v519 = v503.read();	// L771
    int8_t a10;	// L772
    a10 = v519;	// L773
    int32_t v521 = v504.read();	// L774
    int32_t p10;	// L775
    p10 = v521;	// L776
    int8_t v523 = a10;	// L777
    v505.write(v523);	// L778
    int32_t v524 = tile10;	// L779
    int v525 = v524;	// L780
    int8_t v526 = v500[2][2][v525];	// L781
    int32_t v527 = v526;	// L782
    int32_t wt10;	// L783
    wt10 = v527;	// L784
    int32_t v529 = opcode10;	// L785
    bool v530 = v529 == 1;	// L786
    if (v530) {	// L787
      int32_t v531 = p10;	// L788
      int8_t v532 = a10;	// L789
      int32_t v533 = wt10;	// L790
      ap_int<40> v534 = v532;	// L791
      ap_int<40> v535 = v533;	// L792
      ap_int<40> v536 = v534 * v535;	// L793
      ap_int<41> v537 = v531;	// L794
      ap_int<41> v538 = v536;	// L795
      ap_int<41> v539 = v537 + v538;	// L796
      v506.write(v539);	// L797
    } else {
      int32_t v540 = opcode10;	// L799
      bool v541 = v540 == 2;	// L800
      if (v541) {	// L801
        int8_t v542 = a10;	// L802
        int32_t v543 = wt10;	// L803
        ap_int<40> v544 = v542;	// L804
        ap_int<40> v545 = v543;	// L805
        ap_int<40> v546 = v544 * v545;	// L806
        v506.write(v546);	// L807
      } else {
        int32_t v547 = p10;	// L809
        v506.write(v547);	// L810
      }
    }
  }
}

void mac_2_3(
  int8_t v548[4][4][4],
  hls::stream< int32_t >& v549,
  hls::stream< int8_t >& v550,
  hls::stream< int32_t >& v551,
  hls::stream< int32_t >& v552
) {	// L816
  #pragma HLS array_partition variable=v548 complete dim=1
  #pragma HLS array_partition variable=v548 complete dim=2
  #pragma HLS array_partition variable=v548 complete dim=3

  l_S_step_0_step11: for (int step11 = 0; step11 < 4; step11++) {	// L824
  #pragma HLS pipeline II=1
    int32_t v554 = v549.read();	// L825
    int32_t word11;	// L826
    word11 = v554;	// L827
    int32_t v556 = word11;	// L828
    int32_t v557 = v556 >> 24;	// L829
    int32_t v558 = v557 & 255;	// L830
    int32_t opcode11;	// L831
    opcode11 = v558;	// L832
    int32_t v560 = word11;	// L833
    int32_t v561 = v560 >> 16;	// L834
    int32_t v562 = v561 & 255;	// L835
    int32_t tile11;	// L836
    tile11 = v562;	// L837
    int8_t v564 = v550.read();	// L838
    int8_t a11;	// L839
    a11 = v564;	// L840
    int32_t v566 = v551.read();	// L841
    int32_t p11;	// L842
    p11 = v566;	// L843
    int32_t v568 = tile11;	// L844
    int v569 = v568;	// L845
    int8_t v570 = v548[2][3][v569];	// L846
    int32_t v571 = v570;	// L847
    int32_t wt11;	// L848
    wt11 = v571;	// L849
    int32_t v573 = opcode11;	// L850
    bool v574 = v573 == 1;	// L851
    if (v574) {	// L852
      int32_t v575 = p11;	// L853
      int8_t v576 = a11;	// L854
      int32_t v577 = wt11;	// L855
      ap_int<40> v578 = v576;	// L856
      ap_int<40> v579 = v577;	// L857
      ap_int<40> v580 = v578 * v579;	// L858
      ap_int<41> v581 = v575;	// L859
      ap_int<41> v582 = v580;	// L860
      ap_int<41> v583 = v581 + v582;	// L861
      v552.write(v583);	// L862
    } else {
      int32_t v584 = opcode11;	// L864
      bool v585 = v584 == 2;	// L865
      if (v585) {	// L866
        int8_t v586 = a11;	// L867
        int32_t v587 = wt11;	// L868
        ap_int<40> v588 = v586;	// L869
        ap_int<40> v589 = v587;	// L870
        ap_int<40> v590 = v588 * v589;	// L871
        v552.write(v590);	// L872
      } else {
        int32_t v591 = p11;	// L874
        v552.write(v591);	// L875
      }
    }
  }
}

void mac_3_0(
  int8_t v592[4][4][4],
  hls::stream< int32_t >& v593,
  hls::stream< int32_t >& v594,
  hls::stream< int8_t >& v595,
  hls::stream< int32_t >& v596,
  hls::stream< int8_t >& v597,
  hls::stream< int32_t >& v598
) {	// L881
  #pragma HLS array_partition variable=v592 complete dim=1
  #pragma HLS array_partition variable=v592 complete dim=2
  #pragma HLS array_partition variable=v592 complete dim=3

  l_S_step_0_step12: for (int step12 = 0; step12 < 4; step12++) {	// L889
  #pragma HLS pipeline II=1
    int32_t v600 = v593.read();	// L890
    int32_t word12;	// L891
    word12 = v600;	// L892
    int32_t v602 = word12;	// L893
    v594.write(v602);	// L894
    int32_t v603 = word12;	// L895
    int32_t v604 = v603 >> 24;	// L896
    int32_t v605 = v604 & 255;	// L897
    int32_t opcode12;	// L898
    opcode12 = v605;	// L899
    int32_t v607 = word12;	// L900
    int32_t v608 = v607 >> 16;	// L901
    int32_t v609 = v608 & 255;	// L902
    int32_t tile12;	// L903
    tile12 = v609;	// L904
    int8_t v611 = v595.read();	// L905
    int8_t a12;	// L906
    a12 = v611;	// L907
    int32_t v613 = v596.read();	// L908
    int32_t p12;	// L909
    p12 = v613;	// L910
    int8_t v615 = a12;	// L911
    v597.write(v615);	// L912
    int32_t v616 = tile12;	// L913
    int v617 = v616;	// L914
    int8_t v618 = v592[3][0][v617];	// L915
    int32_t v619 = v618;	// L916
    int32_t wt12;	// L917
    wt12 = v619;	// L918
    int32_t v621 = opcode12;	// L919
    bool v622 = v621 == 1;	// L920
    if (v622) {	// L921
      int32_t v623 = p12;	// L922
      int8_t v624 = a12;	// L923
      int32_t v625 = wt12;	// L924
      ap_int<40> v626 = v624;	// L925
      ap_int<40> v627 = v625;	// L926
      ap_int<40> v628 = v626 * v627;	// L927
      ap_int<41> v629 = v623;	// L928
      ap_int<41> v630 = v628;	// L929
      ap_int<41> v631 = v629 + v630;	// L930
      v598.write(v631);	// L931
    } else {
      int32_t v632 = opcode12;	// L933
      bool v633 = v632 == 2;	// L934
      if (v633) {	// L935
        int8_t v634 = a12;	// L936
        int32_t v635 = wt12;	// L937
        ap_int<40> v636 = v634;	// L938
        ap_int<40> v637 = v635;	// L939
        ap_int<40> v638 = v636 * v637;	// L940
        v598.write(v638);	// L941
      } else {
        int32_t v639 = p12;	// L943
        v598.write(v639);	// L944
      }
    }
  }
}

void mac_3_1(
  int8_t v640[4][4][4],
  hls::stream< int32_t >& v641,
  hls::stream< int32_t >& v642,
  hls::stream< int8_t >& v643,
  hls::stream< int32_t >& v644,
  hls::stream< int8_t >& v645,
  hls::stream< int32_t >& v646
) {	// L950
  #pragma HLS array_partition variable=v640 complete dim=1
  #pragma HLS array_partition variable=v640 complete dim=2
  #pragma HLS array_partition variable=v640 complete dim=3

  l_S_step_0_step13: for (int step13 = 0; step13 < 4; step13++) {	// L958
  #pragma HLS pipeline II=1
    int32_t v648 = v641.read();	// L959
    int32_t word13;	// L960
    word13 = v648;	// L961
    int32_t v650 = word13;	// L962
    v642.write(v650);	// L963
    int32_t v651 = word13;	// L964
    int32_t v652 = v651 >> 24;	// L965
    int32_t v653 = v652 & 255;	// L966
    int32_t opcode13;	// L967
    opcode13 = v653;	// L968
    int32_t v655 = word13;	// L969
    int32_t v656 = v655 >> 16;	// L970
    int32_t v657 = v656 & 255;	// L971
    int32_t tile13;	// L972
    tile13 = v657;	// L973
    int8_t v659 = v643.read();	// L974
    int8_t a13;	// L975
    a13 = v659;	// L976
    int32_t v661 = v644.read();	// L977
    int32_t p13;	// L978
    p13 = v661;	// L979
    int8_t v663 = a13;	// L980
    v645.write(v663);	// L981
    int32_t v664 = tile13;	// L982
    int v665 = v664;	// L983
    int8_t v666 = v640[3][1][v665];	// L984
    int32_t v667 = v666;	// L985
    int32_t wt13;	// L986
    wt13 = v667;	// L987
    int32_t v669 = opcode13;	// L988
    bool v670 = v669 == 1;	// L989
    if (v670) {	// L990
      int32_t v671 = p13;	// L991
      int8_t v672 = a13;	// L992
      int32_t v673 = wt13;	// L993
      ap_int<40> v674 = v672;	// L994
      ap_int<40> v675 = v673;	// L995
      ap_int<40> v676 = v674 * v675;	// L996
      ap_int<41> v677 = v671;	// L997
      ap_int<41> v678 = v676;	// L998
      ap_int<41> v679 = v677 + v678;	// L999
      v646.write(v679);	// L1000
    } else {
      int32_t v680 = opcode13;	// L1002
      bool v681 = v680 == 2;	// L1003
      if (v681) {	// L1004
        int8_t v682 = a13;	// L1005
        int32_t v683 = wt13;	// L1006
        ap_int<40> v684 = v682;	// L1007
        ap_int<40> v685 = v683;	// L1008
        ap_int<40> v686 = v684 * v685;	// L1009
        v646.write(v686);	// L1010
      } else {
        int32_t v687 = p13;	// L1012
        v646.write(v687);	// L1013
      }
    }
  }
}

void mac_3_2(
  int8_t v688[4][4][4],
  hls::stream< int32_t >& v689,
  hls::stream< int32_t >& v690,
  hls::stream< int8_t >& v691,
  hls::stream< int32_t >& v692,
  hls::stream< int8_t >& v693,
  hls::stream< int32_t >& v694
) {	// L1019
  #pragma HLS array_partition variable=v688 complete dim=1
  #pragma HLS array_partition variable=v688 complete dim=2
  #pragma HLS array_partition variable=v688 complete dim=3

  l_S_step_0_step14: for (int step14 = 0; step14 < 4; step14++) {	// L1027
  #pragma HLS pipeline II=1
    int32_t v696 = v689.read();	// L1028
    int32_t word14;	// L1029
    word14 = v696;	// L1030
    int32_t v698 = word14;	// L1031
    v690.write(v698);	// L1032
    int32_t v699 = word14;	// L1033
    int32_t v700 = v699 >> 24;	// L1034
    int32_t v701 = v700 & 255;	// L1035
    int32_t opcode14;	// L1036
    opcode14 = v701;	// L1037
    int32_t v703 = word14;	// L1038
    int32_t v704 = v703 >> 16;	// L1039
    int32_t v705 = v704 & 255;	// L1040
    int32_t tile14;	// L1041
    tile14 = v705;	// L1042
    int8_t v707 = v691.read();	// L1043
    int8_t a14;	// L1044
    a14 = v707;	// L1045
    int32_t v709 = v692.read();	// L1046
    int32_t p14;	// L1047
    p14 = v709;	// L1048
    int8_t v711 = a14;	// L1049
    v693.write(v711);	// L1050
    int32_t v712 = tile14;	// L1051
    int v713 = v712;	// L1052
    int8_t v714 = v688[3][2][v713];	// L1053
    int32_t v715 = v714;	// L1054
    int32_t wt14;	// L1055
    wt14 = v715;	// L1056
    int32_t v717 = opcode14;	// L1057
    bool v718 = v717 == 1;	// L1058
    if (v718) {	// L1059
      int32_t v719 = p14;	// L1060
      int8_t v720 = a14;	// L1061
      int32_t v721 = wt14;	// L1062
      ap_int<40> v722 = v720;	// L1063
      ap_int<40> v723 = v721;	// L1064
      ap_int<40> v724 = v722 * v723;	// L1065
      ap_int<41> v725 = v719;	// L1066
      ap_int<41> v726 = v724;	// L1067
      ap_int<41> v727 = v725 + v726;	// L1068
      v694.write(v727);	// L1069
    } else {
      int32_t v728 = opcode14;	// L1071
      bool v729 = v728 == 2;	// L1072
      if (v729) {	// L1073
        int8_t v730 = a14;	// L1074
        int32_t v731 = wt14;	// L1075
        ap_int<40> v732 = v730;	// L1076
        ap_int<40> v733 = v731;	// L1077
        ap_int<40> v734 = v732 * v733;	// L1078
        v694.write(v734);	// L1079
      } else {
        int32_t v735 = p14;	// L1081
        v694.write(v735);	// L1082
      }
    }
  }
}

void mac_3_3(
  int8_t v736[4][4][4],
  hls::stream< int32_t >& v737,
  hls::stream< int8_t >& v738,
  hls::stream< int32_t >& v739,
  hls::stream< int32_t >& v740
) {	// L1088
  #pragma HLS array_partition variable=v736 complete dim=1
  #pragma HLS array_partition variable=v736 complete dim=2
  #pragma HLS array_partition variable=v736 complete dim=3

  l_S_step_0_step15: for (int step15 = 0; step15 < 4; step15++) {	// L1095
  #pragma HLS pipeline II=1
    int32_t v742 = v737.read();	// L1096
    int32_t word15;	// L1097
    word15 = v742;	// L1098
    int32_t v744 = word15;	// L1099
    int32_t v745 = v744 >> 24;	// L1100
    int32_t v746 = v745 & 255;	// L1101
    int32_t opcode15;	// L1102
    opcode15 = v746;	// L1103
    int32_t v748 = word15;	// L1104
    int32_t v749 = v748 >> 16;	// L1105
    int32_t v750 = v749 & 255;	// L1106
    int32_t tile15;	// L1107
    tile15 = v750;	// L1108
    int8_t v752 = v738.read();	// L1109
    int8_t a15;	// L1110
    a15 = v752;	// L1111
    int32_t v754 = v739.read();	// L1112
    int32_t p15;	// L1113
    p15 = v754;	// L1114
    int32_t v756 = tile15;	// L1115
    int v757 = v756;	// L1116
    int8_t v758 = v736[3][3][v757];	// L1117
    int32_t v759 = v758;	// L1118
    int32_t wt15;	// L1119
    wt15 = v759;	// L1120
    int32_t v761 = opcode15;	// L1121
    bool v762 = v761 == 1;	// L1122
    if (v762) {	// L1123
      int32_t v763 = p15;	// L1124
      int8_t v764 = a15;	// L1125
      int32_t v765 = wt15;	// L1126
      ap_int<40> v766 = v764;	// L1127
      ap_int<40> v767 = v765;	// L1128
      ap_int<40> v768 = v766 * v767;	// L1129
      ap_int<41> v769 = v763;	// L1130
      ap_int<41> v770 = v768;	// L1131
      ap_int<41> v771 = v769 + v770;	// L1132
      v740.write(v771);	// L1133
    } else {
      int32_t v772 = opcode15;	// L1135
      bool v773 = v772 == 2;	// L1136
      if (v773) {	// L1137
        int8_t v774 = a15;	// L1138
        int32_t v775 = wt15;	// L1139
        ap_int<40> v776 = v774;	// L1140
        ap_int<40> v777 = v775;	// L1141
        ap_int<40> v778 = v776 * v777;	// L1142
        v740.write(v778);	// L1143
      } else {
        int32_t v779 = p15;	// L1145
        v740.write(v779);	// L1146
      }
    }
  }
}

void vpu_0(
  int32_t v780[4][2],
  hls::stream< int32_t >& v781,
  hls::stream< int32_t >& v782,
  hls::stream< int32_t >& v783,
  hls::stream< int32_t >& v784
) {	// L1152
  #pragma HLS array_partition variable=v780 complete dim=1
  #pragma HLS array_partition variable=v780 complete dim=2

  int32_t prog[16];	// L1174
  for (int v786 = 0; v786 < 16; v786++) {	// L1175
    prog[v786] = 0;	// L1175
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 16; pc++) {	// L1176
  #pragma HLS pipeline II=1
    int32_t v788 = v781.read();	// L1177
    int32_t word16;	// L1178
    word16 = v788;	// L1179
    int32_t v790 = word16;	// L1180
    prog[pc] = v790;	// L1181
    int32_t v791 = word16;	// L1182
    v782.write(v791);	// L1183
  }
  int32_t reg[4];	// L1185
  for (int v793 = 0; v793 < 4; v793++) {	// L1186
    reg[v793] = 0;	// L1186
  }
  l_S_m_1_m: for (int m = 0; m < 4; m++) {	// L1187
    l_S_pc2_1_pc2: for (int pc2 = 0; pc2 < 16; pc2++) {	// L1188
    #pragma HLS pipeline II=1
      int32_t v796 = prog[pc2];	// L1189
      int32_t word2;	// L1190
      word2 = v796;	// L1191
      int32_t v798 = word2;	// L1192
      int32_t v799 = v798 >> 24;	// L1193
      int32_t v800 = v799 & 255;	// L1194
      int32_t opcode16;	// L1195
      opcode16 = v800;	// L1196
      int32_t v802 = word2;	// L1197
      int32_t v803 = v802 >> 20;	// L1198
      int32_t v804 = v803 & 15;	// L1199
      int32_t dst;	// L1200
      dst = v804;	// L1201
      int32_t v806 = word2;	// L1202
      int32_t v807 = v806 >> 16;	// L1203
      int32_t v808 = v807 & 15;	// L1204
      int32_t src;	// L1205
      src = v808;	// L1206
      int32_t v810 = word2;	// L1207
      int32_t v811 = v810 & 65535;	// L1208
      int32_t imm;	// L1209
      imm = v811;	// L1210
      int32_t v813 = opcode16;	// L1211
      bool v814 = v813 == 9;	// L1212
      if (v814) {	// L1213
        int32_t v815 = v783.read();	// L1214
        int32_t zz;	// L1215
        zz = v815;	// L1216
        int32_t v817 = dst;	// L1217
        int v818 = v817;	// L1218
        int32_t v819 = reg[v818];	// L1219
        int32_t v820 = zz;	// L1220
        ap_int<33> v821 = v819;	// L1221
        ap_int<33> v822 = v820;	// L1222
        ap_int<33> v823 = v821 + v822;	// L1223
        int32_t v824 = v823;	// L1224
        reg[v818] = v824;	// L1225
      } else {
        int32_t v825 = opcode16;	// L1227
        bool v826 = v825 == 1;	// L1228
        if (v826) {	// L1229
          int32_t v827 = v783.read();	// L1230
          int32_t z2;	// L1231
          z2 = v827;	// L1232
          int32_t v829 = z2;	// L1233
          int32_t v830 = dst;	// L1234
          int v831 = v830;	// L1235
          reg[v831] = v829;	// L1236
        } else {
          int32_t v832 = opcode16;	// L1238
          bool v833 = v832 == 2;	// L1239
          if (v833) {	// L1240
            int32_t v834 = src;	// L1241
            int v835 = v834;	// L1242
            int32_t v836 = v780[0][v835];	// L1243
            int32_t v837 = dst;	// L1244
            int v838 = v837;	// L1245
            reg[v838] = v836;	// L1246
          } else {
            int32_t v839 = opcode16;	// L1248
            bool v840 = v839 == 3;	// L1249
            if (v840) {	// L1250
              int32_t v841 = imm;	// L1251
              int32_t v842 = dst;	// L1252
              int v843 = v842;	// L1253
              reg[v843] = v841;	// L1254
            } else {
              int32_t v844 = opcode16;	// L1256
              bool v845 = v844 == 4;	// L1257
              if (v845) {	// L1258
                int32_t v846 = dst;	// L1259
                int v847 = v846;	// L1260
                int32_t v848 = reg[v847];	// L1261
                int32_t v849 = src;	// L1262
                int v850 = v849;	// L1263
                int32_t v851 = reg[v850];	// L1264
                ap_int<33> v852 = v848;	// L1265
                ap_int<33> v853 = v851;	// L1266
                ap_int<33> v854 = v852 + v853;	// L1267
                int32_t v855 = v854;	// L1268
                reg[v847] = v855;	// L1269
              } else {
                int32_t v856 = opcode16;	// L1271
                bool v857 = v856 == 5;	// L1272
                if (v857) {	// L1273
                  int32_t v858 = dst;	// L1274
                  int v859 = v858;	// L1275
                  int32_t v860 = reg[v859];	// L1276
                  int32_t v861 = src;	// L1277
                  int v862 = v861;	// L1278
                  int32_t v863 = reg[v862];	// L1279
                  int64_t v864 = v860;	// L1280
                  int64_t v865 = v863;	// L1281
                  int64_t v866 = v864 * v865;	// L1282
                  int32_t v867 = v866;	// L1283
                  reg[v859] = v867;	// L1284
                } else {
                  int32_t v868 = opcode16;	// L1286
                  bool v869 = v868 == 6;	// L1287
                  if (v869) {	// L1288
                    int32_t v870 = src;	// L1289
                    int v871 = v870;	// L1290
                    int32_t v872 = reg[v871];	// L1291
                    int32_t v873 = dst;	// L1292
                    int v874 = v873;	// L1293
                    int32_t v875 = reg[v874];	// L1294
                    bool v876 = v872 > v875;	// L1295
                    if (v876) {	// L1296
                      int32_t v877 = src;	// L1297
                      int v878 = v877;	// L1298
                      int32_t v879 = reg[v878];	// L1299
                      int32_t v880 = dst;	// L1300
                      int v881 = v880;	// L1301
                      reg[v881] = v879;	// L1302
                    }
                  } else {
                    int32_t v882 = opcode16;	// L1305
                    bool v883 = v882 == 7;	// L1306
                    if (v883) {	// L1307
                      int32_t v884 = dst;	// L1308
                      int v885 = v884;	// L1309
                      int32_t v886 = reg[v885];	// L1310
                      int32_t v887 = imm;	// L1311
                      int32_t v888 = v886 >> v887;	// L1312
                      reg[v885] = v888;	// L1313
                    } else {
                      int32_t v889 = opcode16;	// L1315
                      bool v890 = v889 == 10;	// L1316
                      if (v890) {	// L1317
                        int32_t v891 = dst;	// L1318
                        int v892 = v891;	// L1319
                        int32_t v893 = reg[v892];	// L1320
                        int32_t v894 = src;	// L1321
                        int v895 = v894;	// L1322
                        int32_t v896 = reg[v895];	// L1323
                        ap_int<33> v897 = v893;	// L1324
                        ap_int<33> v898 = v896;	// L1325
                        ap_int<33> v899 = v897 - v898;	// L1326
                        int32_t v900 = v899;	// L1327
                        reg[v892] = v900;	// L1328
                      } else {
                        int32_t v901 = opcode16;	// L1330
                        bool v902 = v901 == 11;	// L1331
                        if (v902) {	// L1332
                          int32_t v903 = dst;	// L1333
                          int v904 = v903;	// L1334
                          int32_t v905 = reg[v904];	// L1335
                          int32_t e;	// L1336
                          e = v905;	// L1337
                          int32_t v907 = e;	// L1338
                          bool v908 = v907 < 0;	// L1339
                          if (v908) {	// L1340
                            e = 0;	// L1341
                          }
                          int32_t v909 = e;	// L1343
                          bool v910 = v909 > 30;	// L1344
                          if (v910) {	// L1345
                            e = 30;	// L1346
                          }
                          int32_t v911 = e;	// L1348
                          int32_t v912 = 1 << v911;	// L1349
                          int32_t v913 = dst;	// L1350
                          int v914 = v913;	// L1351
                          reg[v914] = v912;	// L1352
                        } else {
                          int32_t v915 = opcode16;	// L1354
                          bool v916 = v915 == 12;	// L1355
                          if (v916) {	// L1356
                            int32_t v917 = dst;	// L1357
                            int v918 = v917;	// L1358
                            int32_t v919 = reg[v918];	// L1359
                            int32_t d;	// L1360
                            d = v919;	// L1361
                            int32_t v921 = d;	// L1362
                            bool v922 = v921 > 0;	// L1363
                            if (v922) {	// L1364
                              int32_t v923 = imm;	// L1365
                              int32_t v924 = 1 << v923;	// L1366
                              int32_t v925 = d;	// L1367
                              int32_t v926 = v924 / v925;	// L1368
                              int32_t v927 = dst;	// L1369
                              int v928 = v927;	// L1370
                              reg[v928] = v926;	// L1371
                            } else {
                              int32_t v929 = dst;	// L1373
                              int v930 = v929;	// L1374
                              reg[v930] = 0;	// L1375
                            }
                          } else {
                            int32_t v931 = opcode16;	// L1378
                            bool v932 = v931 == 8;	// L1379
                            if (v932) {	// L1380
                              int32_t v933 = dst;	// L1381
                              int v934 = v933;	// L1382
                              int32_t v935 = reg[v934];	// L1383
                              v784.write(v935);	// L1384
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
      }
    }
  }
}

void vpu_1(
  int32_t v936[4][2],
  hls::stream< int32_t >& v937,
  hls::stream< int32_t >& v938,
  hls::stream< int32_t >& v939,
  hls::stream< int32_t >& v940
) {	// L1401
  #pragma HLS array_partition variable=v936 complete dim=1
  #pragma HLS array_partition variable=v936 complete dim=2

  int32_t prog1[16];	// L1423
  for (int v942 = 0; v942 < 16; v942++) {	// L1424
    prog1[v942] = 0;	// L1424
  }
  l_S_pc_0_pc1: for (int pc1 = 0; pc1 < 16; pc1++) {	// L1425
  #pragma HLS pipeline II=1
    int32_t v944 = v937.read();	// L1426
    int32_t word17;	// L1427
    word17 = v944;	// L1428
    int32_t v946 = word17;	// L1429
    prog1[pc1] = v946;	// L1430
    int32_t v947 = word17;	// L1431
    v938.write(v947);	// L1432
  }
  int32_t reg1[4];	// L1434
  for (int v949 = 0; v949 < 4; v949++) {	// L1435
    reg1[v949] = 0;	// L1435
  }
  l_S_m_1_m1: for (int m1 = 0; m1 < 4; m1++) {	// L1436
    l_S_pc2_1_pc21: for (int pc21 = 0; pc21 < 16; pc21++) {	// L1437
    #pragma HLS pipeline II=1
      int32_t v952 = prog1[pc21];	// L1438
      int32_t word21;	// L1439
      word21 = v952;	// L1440
      int32_t v954 = word21;	// L1441
      int32_t v955 = v954 >> 24;	// L1442
      int32_t v956 = v955 & 255;	// L1443
      int32_t opcode17;	// L1444
      opcode17 = v956;	// L1445
      int32_t v958 = word21;	// L1446
      int32_t v959 = v958 >> 20;	// L1447
      int32_t v960 = v959 & 15;	// L1448
      int32_t dst1;	// L1449
      dst1 = v960;	// L1450
      int32_t v962 = word21;	// L1451
      int32_t v963 = v962 >> 16;	// L1452
      int32_t v964 = v963 & 15;	// L1453
      int32_t src1;	// L1454
      src1 = v964;	// L1455
      int32_t v966 = word21;	// L1456
      int32_t v967 = v966 & 65535;	// L1457
      int32_t imm1;	// L1458
      imm1 = v967;	// L1459
      int32_t v969 = opcode17;	// L1460
      bool v970 = v969 == 9;	// L1461
      if (v970) {	// L1462
        int32_t v971 = v939.read();	// L1463
        int32_t zz1;	// L1464
        zz1 = v971;	// L1465
        int32_t v973 = dst1;	// L1466
        int v974 = v973;	// L1467
        int32_t v975 = reg1[v974];	// L1468
        int32_t v976 = zz1;	// L1469
        ap_int<33> v977 = v975;	// L1470
        ap_int<33> v978 = v976;	// L1471
        ap_int<33> v979 = v977 + v978;	// L1472
        int32_t v980 = v979;	// L1473
        reg1[v974] = v980;	// L1474
      } else {
        int32_t v981 = opcode17;	// L1476
        bool v982 = v981 == 1;	// L1477
        if (v982) {	// L1478
          int32_t v983 = v939.read();	// L1479
          int32_t z21;	// L1480
          z21 = v983;	// L1481
          int32_t v985 = z21;	// L1482
          int32_t v986 = dst1;	// L1483
          int v987 = v986;	// L1484
          reg1[v987] = v985;	// L1485
        } else {
          int32_t v988 = opcode17;	// L1487
          bool v989 = v988 == 2;	// L1488
          if (v989) {	// L1489
            int32_t v990 = src1;	// L1490
            int v991 = v990;	// L1491
            int32_t v992 = v936[1][v991];	// L1492
            int32_t v993 = dst1;	// L1493
            int v994 = v993;	// L1494
            reg1[v994] = v992;	// L1495
          } else {
            int32_t v995 = opcode17;	// L1497
            bool v996 = v995 == 3;	// L1498
            if (v996) {	// L1499
              int32_t v997 = imm1;	// L1500
              int32_t v998 = dst1;	// L1501
              int v999 = v998;	// L1502
              reg1[v999] = v997;	// L1503
            } else {
              int32_t v1000 = opcode17;	// L1505
              bool v1001 = v1000 == 4;	// L1506
              if (v1001) {	// L1507
                int32_t v1002 = dst1;	// L1508
                int v1003 = v1002;	// L1509
                int32_t v1004 = reg1[v1003];	// L1510
                int32_t v1005 = src1;	// L1511
                int v1006 = v1005;	// L1512
                int32_t v1007 = reg1[v1006];	// L1513
                ap_int<33> v1008 = v1004;	// L1514
                ap_int<33> v1009 = v1007;	// L1515
                ap_int<33> v1010 = v1008 + v1009;	// L1516
                int32_t v1011 = v1010;	// L1517
                reg1[v1003] = v1011;	// L1518
              } else {
                int32_t v1012 = opcode17;	// L1520
                bool v1013 = v1012 == 5;	// L1521
                if (v1013) {	// L1522
                  int32_t v1014 = dst1;	// L1523
                  int v1015 = v1014;	// L1524
                  int32_t v1016 = reg1[v1015];	// L1525
                  int32_t v1017 = src1;	// L1526
                  int v1018 = v1017;	// L1527
                  int32_t v1019 = reg1[v1018];	// L1528
                  int64_t v1020 = v1016;	// L1529
                  int64_t v1021 = v1019;	// L1530
                  int64_t v1022 = v1020 * v1021;	// L1531
                  int32_t v1023 = v1022;	// L1532
                  reg1[v1015] = v1023;	// L1533
                } else {
                  int32_t v1024 = opcode17;	// L1535
                  bool v1025 = v1024 == 6;	// L1536
                  if (v1025) {	// L1537
                    int32_t v1026 = src1;	// L1538
                    int v1027 = v1026;	// L1539
                    int32_t v1028 = reg1[v1027];	// L1540
                    int32_t v1029 = dst1;	// L1541
                    int v1030 = v1029;	// L1542
                    int32_t v1031 = reg1[v1030];	// L1543
                    bool v1032 = v1028 > v1031;	// L1544
                    if (v1032) {	// L1545
                      int32_t v1033 = src1;	// L1546
                      int v1034 = v1033;	// L1547
                      int32_t v1035 = reg1[v1034];	// L1548
                      int32_t v1036 = dst1;	// L1549
                      int v1037 = v1036;	// L1550
                      reg1[v1037] = v1035;	// L1551
                    }
                  } else {
                    int32_t v1038 = opcode17;	// L1554
                    bool v1039 = v1038 == 7;	// L1555
                    if (v1039) {	// L1556
                      int32_t v1040 = dst1;	// L1557
                      int v1041 = v1040;	// L1558
                      int32_t v1042 = reg1[v1041];	// L1559
                      int32_t v1043 = imm1;	// L1560
                      int32_t v1044 = v1042 >> v1043;	// L1561
                      reg1[v1041] = v1044;	// L1562
                    } else {
                      int32_t v1045 = opcode17;	// L1564
                      bool v1046 = v1045 == 10;	// L1565
                      if (v1046) {	// L1566
                        int32_t v1047 = dst1;	// L1567
                        int v1048 = v1047;	// L1568
                        int32_t v1049 = reg1[v1048];	// L1569
                        int32_t v1050 = src1;	// L1570
                        int v1051 = v1050;	// L1571
                        int32_t v1052 = reg1[v1051];	// L1572
                        ap_int<33> v1053 = v1049;	// L1573
                        ap_int<33> v1054 = v1052;	// L1574
                        ap_int<33> v1055 = v1053 - v1054;	// L1575
                        int32_t v1056 = v1055;	// L1576
                        reg1[v1048] = v1056;	// L1577
                      } else {
                        int32_t v1057 = opcode17;	// L1579
                        bool v1058 = v1057 == 11;	// L1580
                        if (v1058) {	// L1581
                          int32_t v1059 = dst1;	// L1582
                          int v1060 = v1059;	// L1583
                          int32_t v1061 = reg1[v1060];	// L1584
                          int32_t e1;	// L1585
                          e1 = v1061;	// L1586
                          int32_t v1063 = e1;	// L1587
                          bool v1064 = v1063 < 0;	// L1588
                          if (v1064) {	// L1589
                            e1 = 0;	// L1590
                          }
                          int32_t v1065 = e1;	// L1592
                          bool v1066 = v1065 > 30;	// L1593
                          if (v1066) {	// L1594
                            e1 = 30;	// L1595
                          }
                          int32_t v1067 = e1;	// L1597
                          int32_t v1068 = 1 << v1067;	// L1598
                          int32_t v1069 = dst1;	// L1599
                          int v1070 = v1069;	// L1600
                          reg1[v1070] = v1068;	// L1601
                        } else {
                          int32_t v1071 = opcode17;	// L1603
                          bool v1072 = v1071 == 12;	// L1604
                          if (v1072) {	// L1605
                            int32_t v1073 = dst1;	// L1606
                            int v1074 = v1073;	// L1607
                            int32_t v1075 = reg1[v1074];	// L1608
                            int32_t d1;	// L1609
                            d1 = v1075;	// L1610
                            int32_t v1077 = d1;	// L1611
                            bool v1078 = v1077 > 0;	// L1612
                            if (v1078) {	// L1613
                              int32_t v1079 = imm1;	// L1614
                              int32_t v1080 = 1 << v1079;	// L1615
                              int32_t v1081 = d1;	// L1616
                              int32_t v1082 = v1080 / v1081;	// L1617
                              int32_t v1083 = dst1;	// L1618
                              int v1084 = v1083;	// L1619
                              reg1[v1084] = v1082;	// L1620
                            } else {
                              int32_t v1085 = dst1;	// L1622
                              int v1086 = v1085;	// L1623
                              reg1[v1086] = 0;	// L1624
                            }
                          } else {
                            int32_t v1087 = opcode17;	// L1627
                            bool v1088 = v1087 == 8;	// L1628
                            if (v1088) {	// L1629
                              int32_t v1089 = dst1;	// L1630
                              int v1090 = v1089;	// L1631
                              int32_t v1091 = reg1[v1090];	// L1632
                              v940.write(v1091);	// L1633
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
      }
    }
  }
}

void vpu_2(
  int32_t v1092[4][2],
  hls::stream< int32_t >& v1093,
  hls::stream< int32_t >& v1094,
  hls::stream< int32_t >& v1095,
  hls::stream< int32_t >& v1096
) {	// L1650
  #pragma HLS array_partition variable=v1092 complete dim=1
  #pragma HLS array_partition variable=v1092 complete dim=2

  int32_t prog2[16];	// L1672
  for (int v1098 = 0; v1098 < 16; v1098++) {	// L1673
    prog2[v1098] = 0;	// L1673
  }
  l_S_pc_0_pc2: for (int pc2 = 0; pc2 < 16; pc2++) {	// L1674
  #pragma HLS pipeline II=1
    int32_t v1100 = v1093.read();	// L1675
    int32_t word18;	// L1676
    word18 = v1100;	// L1677
    int32_t v1102 = word18;	// L1678
    prog2[pc2] = v1102;	// L1679
    int32_t v1103 = word18;	// L1680
    v1094.write(v1103);	// L1681
  }
  int32_t reg2[4];	// L1683
  for (int v1105 = 0; v1105 < 4; v1105++) {	// L1684
    reg2[v1105] = 0;	// L1684
  }
  l_S_m_1_m2: for (int m2 = 0; m2 < 4; m2++) {	// L1685
    l_S_pc2_1_pc22: for (int pc22 = 0; pc22 < 16; pc22++) {	// L1686
    #pragma HLS pipeline II=1
      int32_t v1108 = prog2[pc22];	// L1687
      int32_t word22;	// L1688
      word22 = v1108;	// L1689
      int32_t v1110 = word22;	// L1690
      int32_t v1111 = v1110 >> 24;	// L1691
      int32_t v1112 = v1111 & 255;	// L1692
      int32_t opcode18;	// L1693
      opcode18 = v1112;	// L1694
      int32_t v1114 = word22;	// L1695
      int32_t v1115 = v1114 >> 20;	// L1696
      int32_t v1116 = v1115 & 15;	// L1697
      int32_t dst2;	// L1698
      dst2 = v1116;	// L1699
      int32_t v1118 = word22;	// L1700
      int32_t v1119 = v1118 >> 16;	// L1701
      int32_t v1120 = v1119 & 15;	// L1702
      int32_t src2;	// L1703
      src2 = v1120;	// L1704
      int32_t v1122 = word22;	// L1705
      int32_t v1123 = v1122 & 65535;	// L1706
      int32_t imm2;	// L1707
      imm2 = v1123;	// L1708
      int32_t v1125 = opcode18;	// L1709
      bool v1126 = v1125 == 9;	// L1710
      if (v1126) {	// L1711
        int32_t v1127 = v1095.read();	// L1712
        int32_t zz2;	// L1713
        zz2 = v1127;	// L1714
        int32_t v1129 = dst2;	// L1715
        int v1130 = v1129;	// L1716
        int32_t v1131 = reg2[v1130];	// L1717
        int32_t v1132 = zz2;	// L1718
        ap_int<33> v1133 = v1131;	// L1719
        ap_int<33> v1134 = v1132;	// L1720
        ap_int<33> v1135 = v1133 + v1134;	// L1721
        int32_t v1136 = v1135;	// L1722
        reg2[v1130] = v1136;	// L1723
      } else {
        int32_t v1137 = opcode18;	// L1725
        bool v1138 = v1137 == 1;	// L1726
        if (v1138) {	// L1727
          int32_t v1139 = v1095.read();	// L1728
          int32_t z22;	// L1729
          z22 = v1139;	// L1730
          int32_t v1141 = z22;	// L1731
          int32_t v1142 = dst2;	// L1732
          int v1143 = v1142;	// L1733
          reg2[v1143] = v1141;	// L1734
        } else {
          int32_t v1144 = opcode18;	// L1736
          bool v1145 = v1144 == 2;	// L1737
          if (v1145) {	// L1738
            int32_t v1146 = src2;	// L1739
            int v1147 = v1146;	// L1740
            int32_t v1148 = v1092[2][v1147];	// L1741
            int32_t v1149 = dst2;	// L1742
            int v1150 = v1149;	// L1743
            reg2[v1150] = v1148;	// L1744
          } else {
            int32_t v1151 = opcode18;	// L1746
            bool v1152 = v1151 == 3;	// L1747
            if (v1152) {	// L1748
              int32_t v1153 = imm2;	// L1749
              int32_t v1154 = dst2;	// L1750
              int v1155 = v1154;	// L1751
              reg2[v1155] = v1153;	// L1752
            } else {
              int32_t v1156 = opcode18;	// L1754
              bool v1157 = v1156 == 4;	// L1755
              if (v1157) {	// L1756
                int32_t v1158 = dst2;	// L1757
                int v1159 = v1158;	// L1758
                int32_t v1160 = reg2[v1159];	// L1759
                int32_t v1161 = src2;	// L1760
                int v1162 = v1161;	// L1761
                int32_t v1163 = reg2[v1162];	// L1762
                ap_int<33> v1164 = v1160;	// L1763
                ap_int<33> v1165 = v1163;	// L1764
                ap_int<33> v1166 = v1164 + v1165;	// L1765
                int32_t v1167 = v1166;	// L1766
                reg2[v1159] = v1167;	// L1767
              } else {
                int32_t v1168 = opcode18;	// L1769
                bool v1169 = v1168 == 5;	// L1770
                if (v1169) {	// L1771
                  int32_t v1170 = dst2;	// L1772
                  int v1171 = v1170;	// L1773
                  int32_t v1172 = reg2[v1171];	// L1774
                  int32_t v1173 = src2;	// L1775
                  int v1174 = v1173;	// L1776
                  int32_t v1175 = reg2[v1174];	// L1777
                  int64_t v1176 = v1172;	// L1778
                  int64_t v1177 = v1175;	// L1779
                  int64_t v1178 = v1176 * v1177;	// L1780
                  int32_t v1179 = v1178;	// L1781
                  reg2[v1171] = v1179;	// L1782
                } else {
                  int32_t v1180 = opcode18;	// L1784
                  bool v1181 = v1180 == 6;	// L1785
                  if (v1181) {	// L1786
                    int32_t v1182 = src2;	// L1787
                    int v1183 = v1182;	// L1788
                    int32_t v1184 = reg2[v1183];	// L1789
                    int32_t v1185 = dst2;	// L1790
                    int v1186 = v1185;	// L1791
                    int32_t v1187 = reg2[v1186];	// L1792
                    bool v1188 = v1184 > v1187;	// L1793
                    if (v1188) {	// L1794
                      int32_t v1189 = src2;	// L1795
                      int v1190 = v1189;	// L1796
                      int32_t v1191 = reg2[v1190];	// L1797
                      int32_t v1192 = dst2;	// L1798
                      int v1193 = v1192;	// L1799
                      reg2[v1193] = v1191;	// L1800
                    }
                  } else {
                    int32_t v1194 = opcode18;	// L1803
                    bool v1195 = v1194 == 7;	// L1804
                    if (v1195) {	// L1805
                      int32_t v1196 = dst2;	// L1806
                      int v1197 = v1196;	// L1807
                      int32_t v1198 = reg2[v1197];	// L1808
                      int32_t v1199 = imm2;	// L1809
                      int32_t v1200 = v1198 >> v1199;	// L1810
                      reg2[v1197] = v1200;	// L1811
                    } else {
                      int32_t v1201 = opcode18;	// L1813
                      bool v1202 = v1201 == 10;	// L1814
                      if (v1202) {	// L1815
                        int32_t v1203 = dst2;	// L1816
                        int v1204 = v1203;	// L1817
                        int32_t v1205 = reg2[v1204];	// L1818
                        int32_t v1206 = src2;	// L1819
                        int v1207 = v1206;	// L1820
                        int32_t v1208 = reg2[v1207];	// L1821
                        ap_int<33> v1209 = v1205;	// L1822
                        ap_int<33> v1210 = v1208;	// L1823
                        ap_int<33> v1211 = v1209 - v1210;	// L1824
                        int32_t v1212 = v1211;	// L1825
                        reg2[v1204] = v1212;	// L1826
                      } else {
                        int32_t v1213 = opcode18;	// L1828
                        bool v1214 = v1213 == 11;	// L1829
                        if (v1214) {	// L1830
                          int32_t v1215 = dst2;	// L1831
                          int v1216 = v1215;	// L1832
                          int32_t v1217 = reg2[v1216];	// L1833
                          int32_t e2;	// L1834
                          e2 = v1217;	// L1835
                          int32_t v1219 = e2;	// L1836
                          bool v1220 = v1219 < 0;	// L1837
                          if (v1220) {	// L1838
                            e2 = 0;	// L1839
                          }
                          int32_t v1221 = e2;	// L1841
                          bool v1222 = v1221 > 30;	// L1842
                          if (v1222) {	// L1843
                            e2 = 30;	// L1844
                          }
                          int32_t v1223 = e2;	// L1846
                          int32_t v1224 = 1 << v1223;	// L1847
                          int32_t v1225 = dst2;	// L1848
                          int v1226 = v1225;	// L1849
                          reg2[v1226] = v1224;	// L1850
                        } else {
                          int32_t v1227 = opcode18;	// L1852
                          bool v1228 = v1227 == 12;	// L1853
                          if (v1228) {	// L1854
                            int32_t v1229 = dst2;	// L1855
                            int v1230 = v1229;	// L1856
                            int32_t v1231 = reg2[v1230];	// L1857
                            int32_t d2;	// L1858
                            d2 = v1231;	// L1859
                            int32_t v1233 = d2;	// L1860
                            bool v1234 = v1233 > 0;	// L1861
                            if (v1234) {	// L1862
                              int32_t v1235 = imm2;	// L1863
                              int32_t v1236 = 1 << v1235;	// L1864
                              int32_t v1237 = d2;	// L1865
                              int32_t v1238 = v1236 / v1237;	// L1866
                              int32_t v1239 = dst2;	// L1867
                              int v1240 = v1239;	// L1868
                              reg2[v1240] = v1238;	// L1869
                            } else {
                              int32_t v1241 = dst2;	// L1871
                              int v1242 = v1241;	// L1872
                              reg2[v1242] = 0;	// L1873
                            }
                          } else {
                            int32_t v1243 = opcode18;	// L1876
                            bool v1244 = v1243 == 8;	// L1877
                            if (v1244) {	// L1878
                              int32_t v1245 = dst2;	// L1879
                              int v1246 = v1245;	// L1880
                              int32_t v1247 = reg2[v1246];	// L1881
                              v1096.write(v1247);	// L1882
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
      }
    }
  }
}

void vpu_3(
  int32_t v1248[4][2],
  hls::stream< int32_t >& v1249,
  hls::stream< int32_t >& v1250,
  hls::stream< int32_t >& v1251
) {	// L1899
  #pragma HLS array_partition variable=v1248 complete dim=1
  #pragma HLS array_partition variable=v1248 complete dim=2

  int32_t prog3[16];	// L1921
  for (int v1253 = 0; v1253 < 16; v1253++) {	// L1922
    prog3[v1253] = 0;	// L1922
  }
  l_S_pc_0_pc3: for (int pc3 = 0; pc3 < 16; pc3++) {	// L1923
  #pragma HLS pipeline II=1
    int32_t v1255 = v1249.read();	// L1924
    int32_t word19;	// L1925
    word19 = v1255;	// L1926
    int32_t v1257 = word19;	// L1927
    prog3[pc3] = v1257;	// L1928
  }
  int32_t reg3[4];	// L1930
  for (int v1259 = 0; v1259 < 4; v1259++) {	// L1931
    reg3[v1259] = 0;	// L1931
  }
  l_S_m_1_m3: for (int m3 = 0; m3 < 4; m3++) {	// L1932
    l_S_pc2_1_pc23: for (int pc23 = 0; pc23 < 16; pc23++) {	// L1933
    #pragma HLS pipeline II=1
      int32_t v1262 = prog3[pc23];	// L1934
      int32_t word23;	// L1935
      word23 = v1262;	// L1936
      int32_t v1264 = word23;	// L1937
      int32_t v1265 = v1264 >> 24;	// L1938
      int32_t v1266 = v1265 & 255;	// L1939
      int32_t opcode19;	// L1940
      opcode19 = v1266;	// L1941
      int32_t v1268 = word23;	// L1942
      int32_t v1269 = v1268 >> 20;	// L1943
      int32_t v1270 = v1269 & 15;	// L1944
      int32_t dst3;	// L1945
      dst3 = v1270;	// L1946
      int32_t v1272 = word23;	// L1947
      int32_t v1273 = v1272 >> 16;	// L1948
      int32_t v1274 = v1273 & 15;	// L1949
      int32_t src3;	// L1950
      src3 = v1274;	// L1951
      int32_t v1276 = word23;	// L1952
      int32_t v1277 = v1276 & 65535;	// L1953
      int32_t imm3;	// L1954
      imm3 = v1277;	// L1955
      int32_t v1279 = opcode19;	// L1956
      bool v1280 = v1279 == 9;	// L1957
      if (v1280) {	// L1958
        int32_t v1281 = v1250.read();	// L1959
        int32_t zz3;	// L1960
        zz3 = v1281;	// L1961
        int32_t v1283 = dst3;	// L1962
        int v1284 = v1283;	// L1963
        int32_t v1285 = reg3[v1284];	// L1964
        int32_t v1286 = zz3;	// L1965
        ap_int<33> v1287 = v1285;	// L1966
        ap_int<33> v1288 = v1286;	// L1967
        ap_int<33> v1289 = v1287 + v1288;	// L1968
        int32_t v1290 = v1289;	// L1969
        reg3[v1284] = v1290;	// L1970
      } else {
        int32_t v1291 = opcode19;	// L1972
        bool v1292 = v1291 == 1;	// L1973
        if (v1292) {	// L1974
          int32_t v1293 = v1250.read();	// L1975
          int32_t z23;	// L1976
          z23 = v1293;	// L1977
          int32_t v1295 = z23;	// L1978
          int32_t v1296 = dst3;	// L1979
          int v1297 = v1296;	// L1980
          reg3[v1297] = v1295;	// L1981
        } else {
          int32_t v1298 = opcode19;	// L1983
          bool v1299 = v1298 == 2;	// L1984
          if (v1299) {	// L1985
            int32_t v1300 = src3;	// L1986
            int v1301 = v1300;	// L1987
            int32_t v1302 = v1248[3][v1301];	// L1988
            int32_t v1303 = dst3;	// L1989
            int v1304 = v1303;	// L1990
            reg3[v1304] = v1302;	// L1991
          } else {
            int32_t v1305 = opcode19;	// L1993
            bool v1306 = v1305 == 3;	// L1994
            if (v1306) {	// L1995
              int32_t v1307 = imm3;	// L1996
              int32_t v1308 = dst3;	// L1997
              int v1309 = v1308;	// L1998
              reg3[v1309] = v1307;	// L1999
            } else {
              int32_t v1310 = opcode19;	// L2001
              bool v1311 = v1310 == 4;	// L2002
              if (v1311) {	// L2003
                int32_t v1312 = dst3;	// L2004
                int v1313 = v1312;	// L2005
                int32_t v1314 = reg3[v1313];	// L2006
                int32_t v1315 = src3;	// L2007
                int v1316 = v1315;	// L2008
                int32_t v1317 = reg3[v1316];	// L2009
                ap_int<33> v1318 = v1314;	// L2010
                ap_int<33> v1319 = v1317;	// L2011
                ap_int<33> v1320 = v1318 + v1319;	// L2012
                int32_t v1321 = v1320;	// L2013
                reg3[v1313] = v1321;	// L2014
              } else {
                int32_t v1322 = opcode19;	// L2016
                bool v1323 = v1322 == 5;	// L2017
                if (v1323) {	// L2018
                  int32_t v1324 = dst3;	// L2019
                  int v1325 = v1324;	// L2020
                  int32_t v1326 = reg3[v1325];	// L2021
                  int32_t v1327 = src3;	// L2022
                  int v1328 = v1327;	// L2023
                  int32_t v1329 = reg3[v1328];	// L2024
                  int64_t v1330 = v1326;	// L2025
                  int64_t v1331 = v1329;	// L2026
                  int64_t v1332 = v1330 * v1331;	// L2027
                  int32_t v1333 = v1332;	// L2028
                  reg3[v1325] = v1333;	// L2029
                } else {
                  int32_t v1334 = opcode19;	// L2031
                  bool v1335 = v1334 == 6;	// L2032
                  if (v1335) {	// L2033
                    int32_t v1336 = src3;	// L2034
                    int v1337 = v1336;	// L2035
                    int32_t v1338 = reg3[v1337];	// L2036
                    int32_t v1339 = dst3;	// L2037
                    int v1340 = v1339;	// L2038
                    int32_t v1341 = reg3[v1340];	// L2039
                    bool v1342 = v1338 > v1341;	// L2040
                    if (v1342) {	// L2041
                      int32_t v1343 = src3;	// L2042
                      int v1344 = v1343;	// L2043
                      int32_t v1345 = reg3[v1344];	// L2044
                      int32_t v1346 = dst3;	// L2045
                      int v1347 = v1346;	// L2046
                      reg3[v1347] = v1345;	// L2047
                    }
                  } else {
                    int32_t v1348 = opcode19;	// L2050
                    bool v1349 = v1348 == 7;	// L2051
                    if (v1349) {	// L2052
                      int32_t v1350 = dst3;	// L2053
                      int v1351 = v1350;	// L2054
                      int32_t v1352 = reg3[v1351];	// L2055
                      int32_t v1353 = imm3;	// L2056
                      int32_t v1354 = v1352 >> v1353;	// L2057
                      reg3[v1351] = v1354;	// L2058
                    } else {
                      int32_t v1355 = opcode19;	// L2060
                      bool v1356 = v1355 == 10;	// L2061
                      if (v1356) {	// L2062
                        int32_t v1357 = dst3;	// L2063
                        int v1358 = v1357;	// L2064
                        int32_t v1359 = reg3[v1358];	// L2065
                        int32_t v1360 = src3;	// L2066
                        int v1361 = v1360;	// L2067
                        int32_t v1362 = reg3[v1361];	// L2068
                        ap_int<33> v1363 = v1359;	// L2069
                        ap_int<33> v1364 = v1362;	// L2070
                        ap_int<33> v1365 = v1363 - v1364;	// L2071
                        int32_t v1366 = v1365;	// L2072
                        reg3[v1358] = v1366;	// L2073
                      } else {
                        int32_t v1367 = opcode19;	// L2075
                        bool v1368 = v1367 == 11;	// L2076
                        if (v1368) {	// L2077
                          int32_t v1369 = dst3;	// L2078
                          int v1370 = v1369;	// L2079
                          int32_t v1371 = reg3[v1370];	// L2080
                          int32_t e3;	// L2081
                          e3 = v1371;	// L2082
                          int32_t v1373 = e3;	// L2083
                          bool v1374 = v1373 < 0;	// L2084
                          if (v1374) {	// L2085
                            e3 = 0;	// L2086
                          }
                          int32_t v1375 = e3;	// L2088
                          bool v1376 = v1375 > 30;	// L2089
                          if (v1376) {	// L2090
                            e3 = 30;	// L2091
                          }
                          int32_t v1377 = e3;	// L2093
                          int32_t v1378 = 1 << v1377;	// L2094
                          int32_t v1379 = dst3;	// L2095
                          int v1380 = v1379;	// L2096
                          reg3[v1380] = v1378;	// L2097
                        } else {
                          int32_t v1381 = opcode19;	// L2099
                          bool v1382 = v1381 == 12;	// L2100
                          if (v1382) {	// L2101
                            int32_t v1383 = dst3;	// L2102
                            int v1384 = v1383;	// L2103
                            int32_t v1385 = reg3[v1384];	// L2104
                            int32_t d3;	// L2105
                            d3 = v1385;	// L2106
                            int32_t v1387 = d3;	// L2107
                            bool v1388 = v1387 > 0;	// L2108
                            if (v1388) {	// L2109
                              int32_t v1389 = imm3;	// L2110
                              int32_t v1390 = 1 << v1389;	// L2111
                              int32_t v1391 = d3;	// L2112
                              int32_t v1392 = v1390 / v1391;	// L2113
                              int32_t v1393 = dst3;	// L2114
                              int v1394 = v1393;	// L2115
                              reg3[v1394] = v1392;	// L2116
                            } else {
                              int32_t v1395 = dst3;	// L2118
                              int v1396 = v1395;	// L2119
                              reg3[v1396] = 0;	// L2120
                            }
                          } else {
                            int32_t v1397 = opcode19;	// L2123
                            bool v1398 = v1397 == 8;	// L2124
                            if (v1398) {	// L2125
                              int32_t v1399 = dst3;	// L2126
                              int v1400 = v1399;	// L2127
                              int32_t v1401 = reg3[v1400];	// L2128
                              v1251.write(v1401);	// L2129
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
      }
    }
  }
}

void vpu_y_out_drain_0(
  int32_t v1402[4][4],
  hls::stream< int32_t >& v1403
) {	// L2146
  #pragma HLS array_partition variable=v1402 complete dim=1
  #pragma HLS array_partition variable=v1402 complete dim=2

  l_S__t_0__t9: for (int _t9 = 0; _t9 < 4; _t9++) {	// L2147
  #pragma HLS pipeline II=1
    int32_t v1405 = v1403.read();	// L2148
    v1402[_t9][0] = v1405;	// L2149
  }
}

void vpu_y_out_drain_1(
  int32_t v1406[4][4],
  hls::stream< int32_t >& v1407
) {	// L2153
  #pragma HLS array_partition variable=v1406 complete dim=1
  #pragma HLS array_partition variable=v1406 complete dim=2

  l_S__t_0__t10: for (int _t10 = 0; _t10 < 4; _t10++) {	// L2154
  #pragma HLS pipeline II=1
    int32_t v1409 = v1407.read();	// L2155
    v1406[_t10][1] = v1409;	// L2156
  }
}

void vpu_y_out_drain_2(
  int32_t v1410[4][4],
  hls::stream< int32_t >& v1411
) {	// L2160
  #pragma HLS array_partition variable=v1410 complete dim=1
  #pragma HLS array_partition variable=v1410 complete dim=2

  l_S__t_0__t11: for (int _t11 = 0; _t11 < 4; _t11++) {	// L2161
  #pragma HLS pipeline II=1
    int32_t v1413 = v1411.read();	// L2162
    v1410[_t11][2] = v1413;	// L2163
  }
}

void vpu_y_out_drain_3(
  int32_t v1414[4][4],
  hls::stream< int32_t >& v1415
) {	// L2167
  #pragma HLS array_partition variable=v1414 complete dim=1
  #pragma HLS array_partition variable=v1414 complete dim=2

  l_S__t_0__t12: for (int _t12 = 0; _t12 < 4; _t12++) {	// L2168
  #pragma HLS pipeline II=1
    int32_t v1417 = v1415.read();	// L2169
    v1414[_t12][3] = v1417;	// L2170
  }
}

/// This is top function.
void top(
  int8_t v1418[4][4],
  int32_t v1419[4][4],
  int32_t v1420[16],
  int8_t v1421[4][4][4],
  int32_t v1422[4][2],
  int32_t v1423[4][4]
) {	// L2174
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v1418 complete dim=1
  #pragma HLS array_partition variable=v1418 complete dim=2

  #pragma HLS array_partition variable=v1419 complete dim=1
  #pragma HLS array_partition variable=v1419 complete dim=2

  #pragma HLS array_partition variable=v1420 complete dim=1

  #pragma HLS array_partition variable=v1421 complete dim=1
  #pragma HLS array_partition variable=v1421 complete dim=2
  #pragma HLS array_partition variable=v1421 complete dim=3

  #pragma HLS array_partition variable=v1422 complete dim=1
  #pragma HLS array_partition variable=v1422 complete dim=2

  #pragma HLS array_partition variable=v1423 complete dim=1
  #pragma HLS array_partition variable=v1423 complete dim=2

  hls::stream< int8_t > v1424;
  #pragma HLS stream variable=v1424 depth=2	// L2175
  hls::stream< int8_t > v1425;
  #pragma HLS stream variable=v1425 depth=2	// L2176
  hls::stream< int8_t > v1426;
  #pragma HLS stream variable=v1426 depth=2	// L2177
  hls::stream< int8_t > v1427;
  #pragma HLS stream variable=v1427 depth=2	// L2178
  hls::stream< int8_t > v1428;
  #pragma HLS stream variable=v1428 depth=2	// L2179
  hls::stream< int8_t > v1429;
  #pragma HLS stream variable=v1429 depth=2	// L2180
  hls::stream< int8_t > v1430;
  #pragma HLS stream variable=v1430 depth=2	// L2181
  hls::stream< int8_t > v1431;
  #pragma HLS stream variable=v1431 depth=2	// L2182
  hls::stream< int8_t > v1432;
  #pragma HLS stream variable=v1432 depth=2	// L2183
  hls::stream< int8_t > v1433;
  #pragma HLS stream variable=v1433 depth=2	// L2184
  hls::stream< int8_t > v1434;
  #pragma HLS stream variable=v1434 depth=2	// L2185
  hls::stream< int8_t > v1435;
  #pragma HLS stream variable=v1435 depth=2	// L2186
  hls::stream< int8_t > v1436;
  #pragma HLS stream variable=v1436 depth=2	// L2187
  hls::stream< int8_t > v1437;
  #pragma HLS stream variable=v1437 depth=2	// L2188
  hls::stream< int8_t > v1438;
  #pragma HLS stream variable=v1438 depth=2	// L2189
  hls::stream< int8_t > v1439;
  #pragma HLS stream variable=v1439 depth=2	// L2190
  hls::stream< int32_t > v1440;
  #pragma HLS stream variable=v1440 depth=2	// L2191
  hls::stream< int32_t > v1441;
  #pragma HLS stream variable=v1441 depth=2	// L2192
  hls::stream< int32_t > v1442;
  #pragma HLS stream variable=v1442 depth=2	// L2193
  hls::stream< int32_t > v1443;
  #pragma HLS stream variable=v1443 depth=2	// L2194
  hls::stream< int32_t > v1444;
  #pragma HLS stream variable=v1444 depth=2	// L2195
  hls::stream< int32_t > v1445;
  #pragma HLS stream variable=v1445 depth=2	// L2196
  hls::stream< int32_t > v1446;
  #pragma HLS stream variable=v1446 depth=2	// L2197
  hls::stream< int32_t > v1447;
  #pragma HLS stream variable=v1447 depth=2	// L2198
  hls::stream< int32_t > v1448;
  #pragma HLS stream variable=v1448 depth=2	// L2199
  hls::stream< int32_t > v1449;
  #pragma HLS stream variable=v1449 depth=2	// L2200
  hls::stream< int32_t > v1450;
  #pragma HLS stream variable=v1450 depth=2	// L2201
  hls::stream< int32_t > v1451;
  #pragma HLS stream variable=v1451 depth=2	// L2202
  hls::stream< int32_t > v1452;
  #pragma HLS stream variable=v1452 depth=2	// L2203
  hls::stream< int32_t > v1453;
  #pragma HLS stream variable=v1453 depth=2	// L2204
  hls::stream< int32_t > v1454;
  #pragma HLS stream variable=v1454 depth=2	// L2205
  hls::stream< int32_t > v1455;
  #pragma HLS stream variable=v1455 depth=2	// L2206
  hls::stream< int32_t > v1456;
  #pragma HLS stream variable=v1456 depth=2	// L2207
  hls::stream< int32_t > v1457;
  #pragma HLS stream variable=v1457 depth=2	// L2208
  hls::stream< int32_t > v1458;
  #pragma HLS stream variable=v1458 depth=2	// L2209
  hls::stream< int32_t > v1459;
  #pragma HLS stream variable=v1459 depth=2	// L2210
  hls::stream< int32_t > v1460;
  #pragma HLS stream variable=v1460 depth=2	// L2211
  hls::stream< int32_t > v1461;
  #pragma HLS stream variable=v1461 depth=2	// L2212
  hls::stream< int32_t > v1462;
  #pragma HLS stream variable=v1462 depth=2	// L2213
  hls::stream< int32_t > v1463;
  #pragma HLS stream variable=v1463 depth=2	// L2214
  hls::stream< int32_t > v1464;
  #pragma HLS stream variable=v1464 depth=2	// L2215
  hls::stream< int32_t > v1465;
  #pragma HLS stream variable=v1465 depth=2	// L2216
  hls::stream< int32_t > v1466;
  #pragma HLS stream variable=v1466 depth=2	// L2217
  hls::stream< int32_t > v1467;
  #pragma HLS stream variable=v1467 depth=2	// L2218
  hls::stream< int32_t > v1468;
  #pragma HLS stream variable=v1468 depth=2	// L2219
  hls::stream< int32_t > v1469;
  #pragma HLS stream variable=v1469 depth=2	// L2220
  hls::stream< int32_t > v1470;
  #pragma HLS stream variable=v1470 depth=2	// L2221
  hls::stream< int32_t > v1471;
  #pragma HLS stream variable=v1471 depth=2	// L2222
  hls::stream< int32_t > v1472;
  #pragma HLS stream variable=v1472 depth=2	// L2223
  hls::stream< int32_t > v1473;
  #pragma HLS stream variable=v1473 depth=2	// L2224
  hls::stream< int32_t > v1474;
  #pragma HLS stream variable=v1474 depth=2	// L2225
  hls::stream< int32_t > v1475;
  #pragma HLS stream variable=v1475 depth=2	// L2226
  hls::stream< int8_t > v1476;
  #pragma HLS stream variable=v1476 depth=2	// L2227
  hls::stream< int8_t > v1477;
  #pragma HLS stream variable=v1477 depth=2	// L2228
  hls::stream< int8_t > v1478;
  #pragma HLS stream variable=v1478 depth=2	// L2229
  hls::stream< int8_t > v1479;
  #pragma HLS stream variable=v1479 depth=2	// L2230
  hls::stream< int32_t > v1480;
  #pragma HLS stream variable=v1480 depth=2	// L2231
  hls::stream< int32_t > v1481;
  #pragma HLS stream variable=v1481 depth=2	// L2232
  hls::stream< int32_t > v1482;
  #pragma HLS stream variable=v1482 depth=2	// L2233
  hls::stream< int32_t > v1483;
  #pragma HLS stream variable=v1483 depth=2	// L2234
  hls::stream< int32_t > v1484;
  #pragma HLS stream variable=v1484 depth=2	// L2235
  hls::stream< int32_t > v1485;
  #pragma HLS stream variable=v1485 depth=2	// L2236
  hls::stream< int32_t > v1486;
  #pragma HLS stream variable=v1486 depth=2	// L2237
  hls::stream< int32_t > v1487;
  #pragma HLS stream variable=v1487 depth=2	// L2238
  hls::stream< int32_t > v1488;
  #pragma HLS stream variable=v1488 depth=2	// L2239
  hls::stream< int32_t > v1489;
  #pragma HLS stream variable=v1489 depth=2	// L2240
  hls::stream< int32_t > v1490;
  #pragma HLS stream variable=v1490 depth=2	// L2241
  hls::stream< int32_t > v1491;
  #pragma HLS stream variable=v1491 depth=2	// L2242
  hls::stream< int32_t > v1492;
  #pragma HLS stream variable=v1492 depth=2	// L2243
  mac_a_in_load_0(v1418, v1476);	// L2244
  mac_a_in_load_1(v1418, v1477);	// L2245
  mac_a_in_load_2(v1418, v1478);	// L2246
  mac_a_in_load_3(v1418, v1479);	// L2247
  mac_op_in_load_0(v1419, v1480);	// L2248
  mac_op_in_load_1(v1419, v1481);	// L2249
  mac_op_in_load_2(v1419, v1482);	// L2250
  mac_op_in_load_3(v1419, v1483);	// L2251
  vpu_op_in_load_0(v1420, v1488);	// L2252
  mac_0_0(v1421, v1480, v1441, v1476, v1425, v1460);	// L2253
  mac_0_1(v1421, v1441, v1442, v1425, v1426, v1461);	// L2254
  mac_0_2(v1421, v1442, v1443, v1426, v1427, v1462);	// L2255
  mac_0_3(v1421, v1443, v1427, v1463);	// L2256
  mac_1_0(v1421, v1481, v1445, v1477, v1460, v1429, v1464);	// L2257
  mac_1_1(v1421, v1445, v1446, v1429, v1461, v1430, v1465);	// L2258
  mac_1_2(v1421, v1446, v1447, v1430, v1462, v1431, v1466);	// L2259
  mac_1_3(v1421, v1447, v1431, v1463, v1467);	// L2260
  mac_2_0(v1421, v1482, v1449, v1478, v1464, v1433, v1468);	// L2261
  mac_2_1(v1421, v1449, v1450, v1433, v1465, v1434, v1469);	// L2262
  mac_2_2(v1421, v1450, v1451, v1434, v1466, v1435, v1470);	// L2263
  mac_2_3(v1421, v1451, v1435, v1467, v1471);	// L2264
  mac_3_0(v1421, v1483, v1453, v1479, v1468, v1437, v1484);	// L2265
  mac_3_1(v1421, v1453, v1454, v1437, v1469, v1438, v1485);	// L2266
  mac_3_2(v1421, v1454, v1455, v1438, v1470, v1439, v1486);	// L2267
  mac_3_3(v1421, v1455, v1439, v1471, v1487);	// L2268
  vpu_0(v1422, v1488, v1473, v1484, v1489);	// L2269
  vpu_1(v1422, v1473, v1474, v1485, v1490);	// L2270
  vpu_2(v1422, v1474, v1475, v1486, v1491);	// L2271
  vpu_3(v1422, v1475, v1487, v1492);	// L2272
  vpu_y_out_drain_0(v1423, v1489);	// L2273
  vpu_y_out_drain_1(v1423, v1490);	// L2274
  vpu_y_out_drain_2(v1423, v1491);	// L2275
  vpu_y_out_drain_3(v1423, v1492);	// L2276
}



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
void mac_a_in_load(
  int8_t v0[4][4],
  int v1,
  hls::stream< int8_t >& v2
) {	// L5
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L6
  #pragma HLS pipeline II=1
    int8_t v4 = v0[_t][v1];	// L7
    v2.write(v4);	// L8
  }
}

void mac_op_in_load(
  int32_t v5[5][4],
  int v6,
  hls::stream< int32_t >& v7
) {	// L12
  #pragma HLS array_partition variable=v5 complete dim=1
  #pragma HLS array_partition variable=v5 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 5; _t1++) {	// L13
  #pragma HLS pipeline II=1
    int32_t v9 = v5[_t1][v6];	// L14
    v7.write(v9);	// L15
  }
}

void vpu_op_in_load(
  int32_t v10[17],
  int v11,
  hls::stream< int32_t >& v12
) {	// L19
  #pragma HLS array_partition variable=v10 complete dim=1

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 17; _t2++) {	// L20
  #pragma HLS pipeline II=1
    int32_t v14 = v10[_t2];	// L21
    v12.write(v14);	// L22
  }
}

void mac_r0(
  int8_t v15[4][4][4],
  int v16,
  int v17,
  hls::stream< int8_t >& v18,
  hls::stream< int8_t >& v19,
  hls::stream< int32_t >& v20,
  hls::stream< int32_t >& v21,
  hls::stream< int32_t >& v22,
  hls::stream< int32_t >& v23
) {	// L26
  #pragma HLS array_partition variable=v15 complete dim=1
  #pragma HLS array_partition variable=v15 complete dim=2
  #pragma HLS array_partition variable=v15 complete dim=3

  int32_t v24 = v20.read();	// L27
  int32_t count;	// L28
  count = v24;	// L29
  int32_t v26 = count;	// L30
  v21.write(v26);	// L31
  int32_t v27 = count;	// L32
  int v28 = v27;	// L35
  for (int v29 = 0; v29 < v28; v29 += 1) {	// L38
  #pragma HLS pipeline II=1
    int32_t v30 = v20.read();	// L39
    int32_t word;	// L40
    word = v30;	// L41
    int32_t v32 = word;	// L42
    v21.write(v32);	// L43
    int32_t v33 = word;	// L44
    int32_t v34 = v33 >> 24;	// L46
    int32_t v35 = v34 & 255;	// L48
    int32_t opcode;	// L49
    opcode = v35;	// L50
    int32_t v37 = word;	// L51
    int32_t v38 = v37 >> 16;	// L53
    int32_t v39 = v38 & 255;	// L54
    int32_t tile;	// L55
    tile = v39;	// L56
    int8_t v41 = v18.read();	// L57
    int8_t a;	// L58
    a = v41;	// L59
    int32_t v43 = v22.read();	// L60
    int32_t p;	// L61
    p = v43;	// L62
    int8_t v45 = a;	// L63
    v19.write(v45);	// L64
    int32_t v46 = tile;	// L65
    int v47 = v46;	// L66
    int8_t v48 = v15[v16][v17][v47];	// L67
    int32_t v49 = v48;	// L68
    int32_t wt;	// L69
    wt = v49;	// L70
    int32_t v51 = opcode;	// L71
    bool v52 = v51 == 1;	// L72
    if (v52) {	// L73
      int32_t v53 = p;	// L74
      int8_t v54 = a;	// L75
      int32_t v55 = wt;	// L76
      ap_int<40> v56 = v54;	// L77
      ap_int<40> v57 = v55;	// L78
      ap_int<40> v58 = v56 * v57;	// L79
      ap_int<41> v59 = v53;	// L80
      ap_int<41> v60 = v58;	// L81
      ap_int<41> v61 = v59 + v60;	// L82
      v23.write(v61);	// L83
    } else {
      int32_t v62 = opcode;	// L85
      bool v63 = v62 == 2;	// L87
      if (v63) {	// L88
        int8_t v64 = a;	// L89
        int32_t v65 = wt;	// L90
        ap_int<40> v66 = v64;	// L91
        ap_int<40> v67 = v65;	// L92
        ap_int<40> v68 = v66 * v67;	// L93
        v23.write(v68);	// L94
      } else {
        int32_t v69 = p;	// L96
        v23.write(v69);	// L97
      }
    }
  }
}

void mac_r1(
  int8_t v70[4][4][4],
  int v71,
  int v72,
  hls::stream< int8_t >& v73,
  hls::stream< int8_t >& v74,
  hls::stream< int32_t >& v75,
  hls::stream< int32_t >& v76,
  hls::stream< int32_t >& v77,
  hls::stream< int32_t >& v78
) {	// L103
  #pragma HLS array_partition variable=v70 complete dim=1
  #pragma HLS array_partition variable=v70 complete dim=2
  #pragma HLS array_partition variable=v70 complete dim=3

  int32_t v79 = v75.read();	// L104
  int32_t count1;	// L105
  count1 = v79;	// L106
  int32_t v81 = count1;	// L107
  v76.write(v81);	// L108
  int32_t v82 = count1;	// L109
  int v83 = v82;	// L112
  for (int v84 = 0; v84 < v83; v84 += 1) {	// L115
  #pragma HLS pipeline II=1
    int32_t v85 = v75.read();	// L116
    int32_t word1;	// L117
    word1 = v85;	// L118
    int32_t v87 = word1;	// L119
    v76.write(v87);	// L120
    int32_t v88 = word1;	// L121
    int32_t v89 = v88 >> 24;	// L123
    int32_t v90 = v89 & 255;	// L125
    int32_t opcode1;	// L126
    opcode1 = v90;	// L127
    int32_t v92 = word1;	// L128
    int32_t v93 = v92 >> 16;	// L130
    int32_t v94 = v93 & 255;	// L131
    int32_t tile1;	// L132
    tile1 = v94;	// L133
    int8_t v96 = v73.read();	// L134
    int8_t a1;	// L135
    a1 = v96;	// L136
    int32_t v98 = v77.read();	// L137
    int32_t p1;	// L138
    p1 = v98;	// L139
    int8_t v100 = a1;	// L140
    v74.write(v100);	// L141
    int32_t v101 = tile1;	// L142
    int v102 = v101;	// L143
    int8_t v103 = v70[v71][v72][v102];	// L144
    int32_t v104 = v103;	// L145
    int32_t wt1;	// L146
    wt1 = v104;	// L147
    int32_t v106 = opcode1;	// L148
    bool v107 = v106 == 1;	// L149
    if (v107) {	// L150
      int32_t v108 = p1;	// L151
      int8_t v109 = a1;	// L152
      int32_t v110 = wt1;	// L153
      ap_int<40> v111 = v109;	// L154
      ap_int<40> v112 = v110;	// L155
      ap_int<40> v113 = v111 * v112;	// L156
      ap_int<41> v114 = v108;	// L157
      ap_int<41> v115 = v113;	// L158
      ap_int<41> v116 = v114 + v115;	// L159
      v78.write(v116);	// L160
    } else {
      int32_t v117 = opcode1;	// L162
      bool v118 = v117 == 2;	// L164
      if (v118) {	// L165
        int8_t v119 = a1;	// L166
        int32_t v120 = wt1;	// L167
        ap_int<40> v121 = v119;	// L168
        ap_int<40> v122 = v120;	// L169
        ap_int<40> v123 = v121 * v122;	// L170
        v78.write(v123);	// L171
      } else {
        int32_t v124 = p1;	// L173
        v78.write(v124);	// L174
      }
    }
  }
}

void mac_r2(
  int8_t v125[4][4][4],
  int v126,
  int v127,
  hls::stream< int8_t >& v128,
  hls::stream< int8_t >& v129,
  hls::stream< int32_t >& v130,
  hls::stream< int32_t >& v131,
  hls::stream< int32_t >& v132
) {	// L180
  #pragma HLS array_partition variable=v125 complete dim=1
  #pragma HLS array_partition variable=v125 complete dim=2
  #pragma HLS array_partition variable=v125 complete dim=3

  int32_t v133 = v130.read();	// L181
  int32_t count2;	// L182
  count2 = v133;	// L183
  int32_t v135 = count2;	// L184
  v131.write(v135);	// L185
  int32_t v136 = count2;	// L186
  int v137 = v136;	// L189
  for (int v138 = 0; v138 < v137; v138 += 1) {	// L192
  #pragma HLS pipeline II=1
    int32_t v139 = v130.read();	// L193
    int32_t word2;	// L194
    word2 = v139;	// L195
    int32_t v141 = word2;	// L196
    v131.write(v141);	// L197
    int32_t v142 = word2;	// L198
    int32_t v143 = v142 >> 24;	// L200
    int32_t v144 = v143 & 255;	// L202
    int32_t opcode2;	// L203
    opcode2 = v144;	// L204
    int32_t v146 = word2;	// L205
    int32_t v147 = v146 >> 16;	// L207
    int32_t v148 = v147 & 255;	// L208
    int32_t tile2;	// L209
    tile2 = v148;	// L210
    int8_t v150 = v128.read();	// L211
    int8_t a2;	// L212
    a2 = v150;	// L213
    int32_t p2;	// L214
    p2 = 0;	// L215
    int8_t v153 = a2;	// L216
    v129.write(v153);	// L217
    int32_t v154 = tile2;	// L218
    int v155 = v154;	// L219
    int8_t v156 = v125[v126][v127][v155];	// L220
    int32_t v157 = v156;	// L221
    int32_t wt2;	// L222
    wt2 = v157;	// L223
    int32_t v159 = opcode2;	// L224
    bool v160 = v159 == 1;	// L225
    if (v160) {	// L226
      int32_t v161 = p2;	// L227
      int8_t v162 = a2;	// L228
      int32_t v163 = wt2;	// L229
      ap_int<40> v164 = v162;	// L230
      ap_int<40> v165 = v163;	// L231
      ap_int<40> v166 = v164 * v165;	// L232
      ap_int<41> v167 = v161;	// L233
      ap_int<41> v168 = v166;	// L234
      ap_int<41> v169 = v167 + v168;	// L235
      v132.write(v169);	// L236
    } else {
      int32_t v170 = opcode2;	// L238
      bool v171 = v170 == 2;	// L240
      if (v171) {	// L241
        int8_t v172 = a2;	// L242
        int32_t v173 = wt2;	// L243
        ap_int<40> v174 = v172;	// L244
        ap_int<40> v175 = v173;	// L245
        ap_int<40> v176 = v174 * v175;	// L246
        v132.write(v176);	// L247
      } else {
        int32_t v177 = p2;	// L249
        v132.write(v177);	// L250
      }
    }
  }
}

void mac_r3(
  int8_t v178[4][4][4],
  int v179,
  int v180,
  hls::stream< int8_t >& v181,
  hls::stream< int32_t >& v182,
  hls::stream< int32_t >& v183,
  hls::stream< int32_t >& v184
) {	// L256
  #pragma HLS array_partition variable=v178 complete dim=1
  #pragma HLS array_partition variable=v178 complete dim=2
  #pragma HLS array_partition variable=v178 complete dim=3

  int32_t v185 = v182.read();	// L257
  int32_t count3;	// L258
  count3 = v185;	// L259
  int32_t v187 = count3;	// L260
  int v188 = v187;	// L263
  for (int v189 = 0; v189 < v188; v189 += 1) {	// L266
  #pragma HLS pipeline II=1
    int32_t v190 = v182.read();	// L267
    int32_t word3;	// L268
    word3 = v190;	// L269
    int32_t v192 = word3;	// L270
    int32_t v193 = v192 >> 24;	// L272
    int32_t v194 = v193 & 255;	// L274
    int32_t opcode3;	// L275
    opcode3 = v194;	// L276
    int32_t v196 = word3;	// L277
    int32_t v197 = v196 >> 16;	// L279
    int32_t v198 = v197 & 255;	// L280
    int32_t tile3;	// L281
    tile3 = v198;	// L282
    int8_t v200 = v181.read();	// L283
    int8_t a3;	// L284
    a3 = v200;	// L285
    int32_t v202 = v183.read();	// L286
    int32_t p3;	// L287
    p3 = v202;	// L288
    int32_t v204 = tile3;	// L289
    int v205 = v204;	// L290
    int8_t v206 = v178[v179][v180][v205];	// L291
    int32_t v207 = v206;	// L292
    int32_t wt3;	// L293
    wt3 = v207;	// L294
    int32_t v209 = opcode3;	// L295
    bool v210 = v209 == 1;	// L296
    if (v210) {	// L297
      int32_t v211 = p3;	// L298
      int8_t v212 = a3;	// L299
      int32_t v213 = wt3;	// L300
      ap_int<40> v214 = v212;	// L301
      ap_int<40> v215 = v213;	// L302
      ap_int<40> v216 = v214 * v215;	// L303
      ap_int<41> v217 = v211;	// L304
      ap_int<41> v218 = v216;	// L305
      ap_int<41> v219 = v217 + v218;	// L306
      v184.write(v219);	// L307
    } else {
      int32_t v220 = opcode3;	// L309
      bool v221 = v220 == 2;	// L311
      if (v221) {	// L312
        int8_t v222 = a3;	// L313
        int32_t v223 = wt3;	// L314
        ap_int<40> v224 = v222;	// L315
        ap_int<40> v225 = v223;	// L316
        ap_int<40> v226 = v224 * v225;	// L317
        v184.write(v226);	// L318
      } else {
        int32_t v227 = p3;	// L320
        v184.write(v227);	// L321
      }
    }
  }
}

void mac_r4(
  int8_t v228[4][4][4],
  int v229,
  int v230,
  hls::stream< int8_t >& v231,
  hls::stream< int8_t >& v232,
  hls::stream< int32_t >& v233,
  hls::stream< int32_t >& v234,
  hls::stream< int32_t >& v235,
  hls::stream< int32_t >& v236
) {	// L327
  #pragma HLS array_partition variable=v228 complete dim=1
  #pragma HLS array_partition variable=v228 complete dim=2
  #pragma HLS array_partition variable=v228 complete dim=3

  int32_t v237 = v233.read();	// L328
  int32_t count4;	// L329
  count4 = v237;	// L330
  int32_t v239 = count4;	// L331
  v234.write(v239);	// L332
  int32_t v240 = count4;	// L333
  int v241 = v240;	// L336
  for (int v242 = 0; v242 < v241; v242 += 1) {	// L339
  #pragma HLS pipeline II=1
    int32_t v243 = v233.read();	// L340
    int32_t word4;	// L341
    word4 = v243;	// L342
    int32_t v245 = word4;	// L343
    v234.write(v245);	// L344
    int32_t v246 = word4;	// L345
    int32_t v247 = v246 >> 24;	// L347
    int32_t v248 = v247 & 255;	// L349
    int32_t opcode4;	// L350
    opcode4 = v248;	// L351
    int32_t v250 = word4;	// L352
    int32_t v251 = v250 >> 16;	// L354
    int32_t v252 = v251 & 255;	// L355
    int32_t tile4;	// L356
    tile4 = v252;	// L357
    int8_t v254 = v231.read();	// L358
    int8_t a4;	// L359
    a4 = v254;	// L360
    int32_t v256 = v235.read();	// L361
    int32_t p4;	// L362
    p4 = v256;	// L363
    int8_t v258 = a4;	// L364
    v232.write(v258);	// L365
    int32_t v259 = tile4;	// L366
    int v260 = v259;	// L367
    int8_t v261 = v228[v229][v230][v260];	// L368
    int32_t v262 = v261;	// L369
    int32_t wt4;	// L370
    wt4 = v262;	// L371
    int32_t v264 = opcode4;	// L372
    bool v265 = v264 == 1;	// L373
    if (v265) {	// L374
      int32_t v266 = p4;	// L375
      int8_t v267 = a4;	// L376
      int32_t v268 = wt4;	// L377
      ap_int<40> v269 = v267;	// L378
      ap_int<40> v270 = v268;	// L379
      ap_int<40> v271 = v269 * v270;	// L380
      ap_int<41> v272 = v266;	// L381
      ap_int<41> v273 = v271;	// L382
      ap_int<41> v274 = v272 + v273;	// L383
      v236.write(v274);	// L384
    } else {
      int32_t v275 = opcode4;	// L386
      bool v276 = v275 == 2;	// L388
      if (v276) {	// L389
        int8_t v277 = a4;	// L390
        int32_t v278 = wt4;	// L391
        ap_int<40> v279 = v277;	// L392
        ap_int<40> v280 = v278;	// L393
        ap_int<40> v281 = v279 * v280;	// L394
        v236.write(v281);	// L395
      } else {
        int32_t v282 = p4;	// L397
        v236.write(v282);	// L398
      }
    }
  }
}

void mac_r5(
  int8_t v283[4][4][4],
  int v284,
  int v285,
  hls::stream< int8_t >& v286,
  hls::stream< int32_t >& v287,
  hls::stream< int32_t >& v288,
  hls::stream< int32_t >& v289
) {	// L404
  #pragma HLS array_partition variable=v283 complete dim=1
  #pragma HLS array_partition variable=v283 complete dim=2
  #pragma HLS array_partition variable=v283 complete dim=3

  int32_t v290 = v287.read();	// L405
  int32_t count5;	// L406
  count5 = v290;	// L407
  int32_t v292 = count5;	// L408
  int v293 = v292;	// L411
  for (int v294 = 0; v294 < v293; v294 += 1) {	// L414
  #pragma HLS pipeline II=1
    int32_t v295 = v287.read();	// L415
    int32_t word5;	// L416
    word5 = v295;	// L417
    int32_t v297 = word5;	// L418
    int32_t v298 = v297 >> 24;	// L420
    int32_t v299 = v298 & 255;	// L422
    int32_t opcode5;	// L423
    opcode5 = v299;	// L424
    int32_t v301 = word5;	// L425
    int32_t v302 = v301 >> 16;	// L427
    int32_t v303 = v302 & 255;	// L428
    int32_t tile5;	// L429
    tile5 = v303;	// L430
    int8_t v305 = v286.read();	// L431
    int8_t a5;	// L432
    a5 = v305;	// L433
    int32_t v307 = v288.read();	// L434
    int32_t p5;	// L435
    p5 = v307;	// L436
    int32_t v309 = tile5;	// L437
    int v310 = v309;	// L438
    int8_t v311 = v283[v284][v285][v310];	// L439
    int32_t v312 = v311;	// L440
    int32_t wt5;	// L441
    wt5 = v312;	// L442
    int32_t v314 = opcode5;	// L443
    bool v315 = v314 == 1;	// L444
    if (v315) {	// L445
      int32_t v316 = p5;	// L446
      int8_t v317 = a5;	// L447
      int32_t v318 = wt5;	// L448
      ap_int<40> v319 = v317;	// L449
      ap_int<40> v320 = v318;	// L450
      ap_int<40> v321 = v319 * v320;	// L451
      ap_int<41> v322 = v316;	// L452
      ap_int<41> v323 = v321;	// L453
      ap_int<41> v324 = v322 + v323;	// L454
      v289.write(v324);	// L455
    } else {
      int32_t v325 = opcode5;	// L457
      bool v326 = v325 == 2;	// L459
      if (v326) {	// L460
        int8_t v327 = a5;	// L461
        int32_t v328 = wt5;	// L462
        ap_int<40> v329 = v327;	// L463
        ap_int<40> v330 = v328;	// L464
        ap_int<40> v331 = v329 * v330;	// L465
        v289.write(v331);	// L466
      } else {
        int32_t v332 = p5;	// L468
        v289.write(v332);	// L469
      }
    }
  }
}

void mac_r6(
  int8_t v333[4][4][4],
  int v334,
  int v335,
  hls::stream< int8_t >& v336,
  hls::stream< int32_t >& v337,
  hls::stream< int32_t >& v338
) {	// L475
  #pragma HLS array_partition variable=v333 complete dim=1
  #pragma HLS array_partition variable=v333 complete dim=2
  #pragma HLS array_partition variable=v333 complete dim=3

  int32_t v339 = v337.read();	// L476
  int32_t count6;	// L477
  count6 = v339;	// L478
  int32_t v341 = count6;	// L479
  int v342 = v341;	// L482
  for (int v343 = 0; v343 < v342; v343 += 1) {	// L485
  #pragma HLS pipeline II=1
    int32_t v344 = v337.read();	// L486
    int32_t word6;	// L487
    word6 = v344;	// L488
    int32_t v346 = word6;	// L489
    int32_t v347 = v346 >> 24;	// L491
    int32_t v348 = v347 & 255;	// L493
    int32_t opcode6;	// L494
    opcode6 = v348;	// L495
    int32_t v350 = word6;	// L496
    int32_t v351 = v350 >> 16;	// L498
    int32_t v352 = v351 & 255;	// L499
    int32_t tile6;	// L500
    tile6 = v352;	// L501
    int8_t v354 = v336.read();	// L502
    int8_t a6;	// L503
    a6 = v354;	// L504
    int32_t p6;	// L505
    p6 = 0;	// L506
    int32_t v357 = tile6;	// L507
    int v358 = v357;	// L508
    int8_t v359 = v333[v334][v335][v358];	// L509
    int32_t v360 = v359;	// L510
    int32_t wt6;	// L511
    wt6 = v360;	// L512
    int32_t v362 = opcode6;	// L513
    bool v363 = v362 == 1;	// L514
    if (v363) {	// L515
      int32_t v364 = p6;	// L516
      int8_t v365 = a6;	// L517
      int32_t v366 = wt6;	// L518
      ap_int<40> v367 = v365;	// L519
      ap_int<40> v368 = v366;	// L520
      ap_int<40> v369 = v367 * v368;	// L521
      ap_int<41> v370 = v364;	// L522
      ap_int<41> v371 = v369;	// L523
      ap_int<41> v372 = v370 + v371;	// L524
      v338.write(v372);	// L525
    } else {
      int32_t v373 = opcode6;	// L527
      bool v374 = v373 == 2;	// L529
      if (v374) {	// L530
        int8_t v375 = a6;	// L531
        int32_t v376 = wt6;	// L532
        ap_int<40> v377 = v375;	// L533
        ap_int<40> v378 = v376;	// L534
        ap_int<40> v379 = v377 * v378;	// L535
        v338.write(v379);	// L536
      } else {
        int32_t v380 = p6;	// L538
        v338.write(v380);	// L539
      }
    }
  }
}

void mac_r7(
  int8_t v381[4][4][4],
  int v382,
  int v383,
  hls::stream< int8_t >& v384,
  hls::stream< int8_t >& v385,
  hls::stream< int32_t >& v386,
  hls::stream< int32_t >& v387,
  hls::stream< int32_t >& v388,
  hls::stream< int32_t >& v389
) {	// L545
  #pragma HLS array_partition variable=v381 complete dim=1
  #pragma HLS array_partition variable=v381 complete dim=2
  #pragma HLS array_partition variable=v381 complete dim=3

  int32_t v390 = v386.read();	// L546
  int32_t count7;	// L547
  count7 = v390;	// L548
  int32_t v392 = count7;	// L549
  v387.write(v392);	// L550
  int32_t v393 = count7;	// L551
  int v394 = v393;	// L554
  for (int v395 = 0; v395 < v394; v395 += 1) {	// L557
  #pragma HLS pipeline II=1
    int32_t v396 = v386.read();	// L558
    int32_t word7;	// L559
    word7 = v396;	// L560
    int32_t v398 = word7;	// L561
    v387.write(v398);	// L562
    int32_t v399 = word7;	// L563
    int32_t v400 = v399 >> 24;	// L565
    int32_t v401 = v400 & 255;	// L567
    int32_t opcode7;	// L568
    opcode7 = v401;	// L569
    int32_t v403 = word7;	// L570
    int32_t v404 = v403 >> 16;	// L572
    int32_t v405 = v404 & 255;	// L573
    int32_t tile7;	// L574
    tile7 = v405;	// L575
    int8_t v407 = v384.read();	// L576
    int8_t a7;	// L577
    a7 = v407;	// L578
    int32_t v409 = v388.read();	// L579
    int32_t p7;	// L580
    p7 = v409;	// L581
    int8_t v411 = a7;	// L582
    v385.write(v411);	// L583
    int32_t v412 = tile7;	// L584
    int v413 = v412;	// L585
    int8_t v414 = v381[v382][v383][v413];	// L586
    int32_t v415 = v414;	// L587
    int32_t wt7;	// L588
    wt7 = v415;	// L589
    int32_t v417 = opcode7;	// L590
    bool v418 = v417 == 1;	// L591
    if (v418) {	// L592
      int32_t v419 = p7;	// L593
      int8_t v420 = a7;	// L594
      int32_t v421 = wt7;	// L595
      ap_int<40> v422 = v420;	// L596
      ap_int<40> v423 = v421;	// L597
      ap_int<40> v424 = v422 * v423;	// L598
      ap_int<41> v425 = v419;	// L599
      ap_int<41> v426 = v424;	// L600
      ap_int<41> v427 = v425 + v426;	// L601
      v389.write(v427);	// L602
    } else {
      int32_t v428 = opcode7;	// L604
      bool v429 = v428 == 2;	// L606
      if (v429) {	// L607
        int8_t v430 = a7;	// L608
        int32_t v431 = wt7;	// L609
        ap_int<40> v432 = v430;	// L610
        ap_int<40> v433 = v431;	// L611
        ap_int<40> v434 = v432 * v433;	// L612
        v389.write(v434);	// L613
      } else {
        int32_t v435 = p7;	// L615
        v389.write(v435);	// L616
      }
    }
  }
}

void mac_r8(
  int8_t v436[4][4][4],
  int v437,
  int v438,
  hls::stream< int8_t >& v439,
  hls::stream< int8_t >& v440,
  hls::stream< int32_t >& v441,
  hls::stream< int32_t >& v442,
  hls::stream< int32_t >& v443
) {	// L622
  #pragma HLS array_partition variable=v436 complete dim=1
  #pragma HLS array_partition variable=v436 complete dim=2
  #pragma HLS array_partition variable=v436 complete dim=3

  int32_t v444 = v441.read();	// L623
  int32_t count8;	// L624
  count8 = v444;	// L625
  int32_t v446 = count8;	// L626
  v442.write(v446);	// L627
  int32_t v447 = count8;	// L628
  int v448 = v447;	// L631
  for (int v449 = 0; v449 < v448; v449 += 1) {	// L634
  #pragma HLS pipeline II=1
    int32_t v450 = v441.read();	// L635
    int32_t word8;	// L636
    word8 = v450;	// L637
    int32_t v452 = word8;	// L638
    v442.write(v452);	// L639
    int32_t v453 = word8;	// L640
    int32_t v454 = v453 >> 24;	// L642
    int32_t v455 = v454 & 255;	// L644
    int32_t opcode8;	// L645
    opcode8 = v455;	// L646
    int32_t v457 = word8;	// L647
    int32_t v458 = v457 >> 16;	// L649
    int32_t v459 = v458 & 255;	// L650
    int32_t tile8;	// L651
    tile8 = v459;	// L652
    int8_t v461 = v439.read();	// L653
    int8_t a8;	// L654
    a8 = v461;	// L655
    int32_t p8;	// L656
    p8 = 0;	// L657
    int8_t v464 = a8;	// L658
    v440.write(v464);	// L659
    int32_t v465 = tile8;	// L660
    int v466 = v465;	// L661
    int8_t v467 = v436[v437][v438][v466];	// L662
    int32_t v468 = v467;	// L663
    int32_t wt8;	// L664
    wt8 = v468;	// L665
    int32_t v470 = opcode8;	// L666
    bool v471 = v470 == 1;	// L667
    if (v471) {	// L668
      int32_t v472 = p8;	// L669
      int8_t v473 = a8;	// L670
      int32_t v474 = wt8;	// L671
      ap_int<40> v475 = v473;	// L672
      ap_int<40> v476 = v474;	// L673
      ap_int<40> v477 = v475 * v476;	// L674
      ap_int<41> v478 = v472;	// L675
      ap_int<41> v479 = v477;	// L676
      ap_int<41> v480 = v478 + v479;	// L677
      v443.write(v480);	// L678
    } else {
      int32_t v481 = opcode8;	// L680
      bool v482 = v481 == 2;	// L682
      if (v482) {	// L683
        int8_t v483 = a8;	// L684
        int32_t v484 = wt8;	// L685
        ap_int<40> v485 = v483;	// L686
        ap_int<40> v486 = v484;	// L687
        ap_int<40> v487 = v485 * v486;	// L688
        v443.write(v487);	// L689
      } else {
        int32_t v488 = p8;	// L691
        v443.write(v488);	// L692
      }
    }
  }
}

void vpu_r0(
  int32_t v489[4][2],
  int v490,
  hls::stream< int32_t >& v491,
  hls::stream< int32_t >& v492,
  hls::stream< int32_t >& v493,
  hls::stream< int32_t >& v494
) {	// L698
  #pragma HLS array_partition variable=v489 complete dim=1
  #pragma HLS array_partition variable=v489 complete dim=2

  int32_t v495 = v491.read();	// L699
  int32_t header;	// L700
  header = v495;	// L701
  int32_t v497 = header;	// L702
  v492.write(v497);	// L703
  int32_t v498 = header;	// L704
  int32_t v499 = v498 & 65535;	// L706
  int32_t plen;	// L707
  plen = v499;	// L708
  int32_t v501 = header;	// L709
  int32_t v502 = v501 >> 16;	// L711
  int32_t v503 = v502 & 65535;	// L712
  int32_t nouts;	// L713
  nouts = v503;	// L714
  int32_t prog[16];	// L715
  for (int v506 = 0; v506 < 16; v506++) {	// L717
    prog[v506] = 0;	// L717
  }
  int32_t v507 = plen;	// L718
  int v508 = v507;	// L720
  for (int v509 = 0; v509 < v508; v509 += 1) {	// L723
  #pragma HLS pipeline II=1
    int32_t v510 = v491.read();	// L724
    int32_t word9;	// L725
    word9 = v510;	// L726
    int32_t v512 = word9;	// L727
    prog[v509] = v512;	// L728
    int32_t v513 = word9;	// L729
    v492.write(v513);	// L730
  }
  int32_t v514 = plen;	// L732
  ap_int<33> v515 = v514;	// L734
  ap_int<33> v516 = 16 - v515;	// L735
  int v517 = v516;	// L736
  for (int v518 = 0; v518 < v517; v518 += 1) {	// L737
  #pragma HLS pipeline II=1
    int32_t v519 = v491.read();	// L738
    int32_t spare;	// L739
    spare = v519;	// L740
    int32_t v521 = spare;	// L741
    v492.write(v521);	// L742
  }
  int32_t v522 = v489[v490][1];	// L744
  int32_t denom;	// L745
  denom = v522;	// L746
  int32_t rcp;	// L747
  rcp = 0;	// L748
  int32_t v525 = denom;	// L749
  bool v526 = v525 > 0;	// L750
  if (v526) {	// L751
    int32_t v527 = denom;	// L754
    int32_t v528 = 16384 / v527;	// L755
    rcp = v528;	// L756
  }
  int32_t r0;	// L758
  r0 = 0;	// L759
  int32_t r1;	// L760
  r1 = 0;	// L761
  int32_t r2;	// L762
  r2 = 0;	// L763
  int32_t r3;	// L764
  r3 = 0;	// L765
  int32_t pc2;	// L766
  pc2 = 0;	// L767
  int32_t v534 = nouts;	// L768
  int32_t v535 = plen;	// L769
  int64_t v536 = v534;	// L770
  int64_t v537 = v535;	// L771
  int64_t v538 = v536 * v537;	// L772
  int v539 = v538;	// L773
  for (int v540 = 0; v540 < v539; v540 += 1) {	// L774
  #pragma HLS pipeline II=1
    int32_t v541 = pc2;	// L775
    int v542 = v541;	// L776
    int32_t v543 = prog[v542];	// L777
    int32_t word2;	// L778
    word2 = v543;	// L779
    int32_t v545 = word2;	// L780
    int32_t v546 = v545 >> 24;	// L782
    int32_t v547 = v546 & 255;	// L784
    int32_t opcode9;	// L785
    opcode9 = v547;	// L786
    int32_t v549 = word2;	// L787
    int32_t v550 = v549 >> 20;	// L789
    int32_t v551 = v550 & 15;	// L791
    int32_t dst;	// L792
    dst = v551;	// L793
    int32_t v553 = word2;	// L794
    int32_t v554 = v553 >> 16;	// L795
    int32_t v555 = v554 & 15;	// L796
    int32_t src;	// L797
    src = v555;	// L798
    int32_t v557 = word2;	// L799
    int32_t v558 = v557 & 65535;	// L800
    int32_t imm;	// L801
    imm = v558;	// L802
    int32_t v560 = r0;	// L803
    int32_t d;	// L804
    d = v560;	// L805
    int32_t v562 = dst;	// L806
    bool v563 = v562 == 1;	// L807
    if (v563) {	// L808
      int32_t v564 = r1;	// L809
      d = v564;	// L810
    } else {
      int32_t v565 = dst;	// L812
      bool v566 = v565 == 2;	// L814
      if (v566) {	// L815
        int32_t v567 = r2;	// L816
        d = v567;	// L817
      } else {
        int32_t v568 = dst;	// L819
        bool v569 = v568 == 3;	// L821
        if (v569) {	// L822
          int32_t v570 = r3;	// L823
          d = v570;	// L824
        }
      }
    }
    int32_t v571 = r0;	// L828
    int32_t a9;	// L829
    a9 = v571;	// L830
    int32_t v573 = src;	// L831
    bool v574 = v573 == 1;	// L832
    if (v574) {	// L833
      int32_t v575 = r1;	// L834
      a9 = v575;	// L835
    } else {
      int32_t v576 = src;	// L837
      bool v577 = v576 == 2;	// L839
      if (v577) {	// L840
        int32_t v578 = r2;	// L841
        a9 = v578;	// L842
      } else {
        int32_t v579 = src;	// L844
        bool v580 = v579 == 3;	// L846
        if (v580) {	// L847
          int32_t v581 = r3;	// L848
          a9 = v581;	// L849
        }
      }
    }
    int32_t wr;	// L853
    wr = 1;	// L854
    int32_t v583 = opcode9;	// L855
    bool v584 = v583 == 9;	// L857
    if (v584) {	// L858
      int32_t v585 = v494.read();	// L859
      int32_t zz;	// L860
      zz = v585;	// L861
      int32_t v587 = d;	// L862
      int32_t v588 = zz;	// L863
      ap_int<33> v589 = v587;	// L864
      ap_int<33> v590 = v588;	// L865
      ap_int<33> v591 = v589 + v590;	// L866
      int32_t v592 = v591;	// L867
      d = v592;	// L868
    } else {
      int32_t v593 = opcode9;	// L870
      bool v594 = v593 == 1;	// L871
      if (v594) {	// L872
        int32_t v595 = v494.read();	// L873
        int32_t z2;	// L874
        z2 = v595;	// L875
        int32_t v597 = z2;	// L876
        d = v597;	// L877
      } else {
        int32_t v598 = opcode9;	// L879
        bool v599 = v598 == 2;	// L881
        if (v599) {	// L882
          int32_t v600 = src;	// L883
          int v601 = v600;	// L884
          int32_t v602 = v489[v490][v601];	// L885
          d = v602;	// L886
        } else {
          int32_t v603 = opcode9;	// L888
          bool v604 = v603 == 3;	// L890
          if (v604) {	// L891
            int32_t v605 = imm;	// L892
            d = v605;	// L893
          } else {
            int32_t v606 = opcode9;	// L895
            bool v607 = v606 == 4;	// L897
            if (v607) {	// L898
              int32_t v608 = d;	// L899
              int32_t v609 = a9;	// L900
              ap_int<33> v610 = v608;	// L901
              ap_int<33> v611 = v609;	// L902
              ap_int<33> v612 = v610 + v611;	// L903
              int32_t v613 = v612;	// L904
              d = v613;	// L905
            } else {
              int32_t v614 = opcode9;	// L907
              bool v615 = v614 == 5;	// L909
              if (v615) {	// L910
                int32_t v616 = d;	// L911
                int32_t v617 = a9;	// L912
                int64_t v618 = v616;	// L913
                int64_t v619 = v617;	// L914
                int64_t v620 = v618 * v619;	// L915
                int32_t v621 = v620;	// L916
                d = v621;	// L917
              } else {
                int32_t v622 = opcode9;	// L919
                bool v623 = v622 == 6;	// L921
                if (v623) {	// L922
                  int32_t v624 = a9;	// L923
                  int32_t v625 = d;	// L924
                  bool v626 = v624 > v625;	// L925
                  if (v626) {	// L926
                    int32_t v627 = a9;	// L927
                    d = v627;	// L928
                  }
                } else {
                  int32_t v628 = opcode9;	// L931
                  bool v629 = v628 == 7;	// L933
                  if (v629) {	// L934
                    int32_t v630 = d;	// L935
                    int32_t v631 = imm;	// L936
                    int32_t v632 = v630 >> v631;	// L937
                    d = v632;	// L938
                  } else {
                    int32_t v633 = opcode9;	// L940
                    bool v634 = v633 == 10;	// L942
                    if (v634) {	// L943
                      int32_t v635 = d;	// L944
                      int32_t v636 = a9;	// L945
                      ap_int<33> v637 = v635;	// L946
                      ap_int<33> v638 = v636;	// L947
                      ap_int<33> v639 = v637 - v638;	// L948
                      int32_t v640 = v639;	// L949
                      d = v640;	// L950
                    } else {
                      int32_t v641 = opcode9;	// L952
                      bool v642 = v641 == 11;	// L954
                      if (v642) {	// L955
                        int32_t v643 = d;	// L956
                        int32_t e;	// L957
                        e = v643;	// L958
                        int32_t v645 = e;	// L959
                        bool v646 = v645 < 0;	// L960
                        if (v646) {	// L961
                          e = 0;	// L962
                        }
                        int32_t v647 = e;	// L964
                        bool v648 = v647 > 30;	// L966
                        if (v648) {	// L967
                          e = 30;	// L968
                        }
                        int32_t v649 = e;	// L970
                        int32_t v650 = 1 << v649;	// L971
                        d = v650;	// L972
                      } else {
                        int32_t v651 = opcode9;	// L974
                        bool v652 = v651 == 12;	// L976
                        if (v652) {	// L977
                          int32_t v653 = rcp;	// L978
                          d = v653;	// L979
                        } else {
                          int32_t v654 = opcode9;	// L981
                          bool v655 = v654 == 8;	// L983
                          if (v655) {	// L984
                            int32_t v656 = d;	// L985
                            v493.write(v656);	// L986
                            wr = 0;	// L987
                          } else {
                            wr = 0;	// L989
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
    int32_t v657 = wr;	// L1002
    bool v658 = v657 == 1;	// L1003
    if (v658) {	// L1004
      int32_t v659 = dst;	// L1005
      bool v660 = v659 == 0;	// L1006
      if (v660) {	// L1007
        int32_t v661 = d;	// L1008
        r0 = v661;	// L1009
      } else {
        int32_t v662 = dst;	// L1011
        bool v663 = v662 == 1;	// L1012
        if (v663) {	// L1013
          int32_t v664 = d;	// L1014
          r1 = v664;	// L1015
        } else {
          int32_t v665 = dst;	// L1017
          bool v666 = v665 == 2;	// L1019
          if (v666) {	// L1020
            int32_t v667 = d;	// L1021
            r2 = v667;	// L1022
          } else {
            int32_t v668 = d;	// L1024
            r3 = v668;	// L1025
          }
        }
      }
    }
    int32_t v669 = pc2;	// L1030
    ap_int<33> v670 = v669;	// L1031
    ap_int<33> v671 = v670 + 1;	// L1033
    int32_t v672 = v671;	// L1034
    pc2 = v672;	// L1035
    int32_t v673 = pc2;	// L1036
    int32_t v674 = plen;	// L1037
    bool v675 = v673 == v674;	// L1038
    if (v675) {	// L1039
      pc2 = 0;	// L1040
    }
  }
}

void vpu_r1(
  int32_t v676[4][2],
  int v677,
  hls::stream< int32_t >& v678,
  hls::stream< int32_t >& v679,
  hls::stream< int32_t >& v680
) {	// L1045
  #pragma HLS array_partition variable=v676 complete dim=1
  #pragma HLS array_partition variable=v676 complete dim=2

  int32_t v681 = v678.read();	// L1046
  int32_t header1;	// L1047
  header1 = v681;	// L1048
  int32_t v683 = header1;	// L1049
  int32_t v684 = v683 & 65535;	// L1051
  int32_t plen1;	// L1052
  plen1 = v684;	// L1053
  int32_t v686 = header1;	// L1054
  int32_t v687 = v686 >> 16;	// L1056
  int32_t v688 = v687 & 65535;	// L1057
  int32_t nouts1;	// L1058
  nouts1 = v688;	// L1059
  int32_t prog1[16];	// L1060
  for (int v691 = 0; v691 < 16; v691++) {	// L1062
    prog1[v691] = 0;	// L1062
  }
  int32_t v692 = plen1;	// L1063
  int v693 = v692;	// L1065
  for (int v694 = 0; v694 < v693; v694 += 1) {	// L1068
  #pragma HLS pipeline II=1
    int32_t v695 = v678.read();	// L1069
    int32_t word10;	// L1070
    word10 = v695;	// L1071
    int32_t v697 = word10;	// L1072
    prog1[v694] = v697;	// L1073
  }
  int32_t v698 = plen1;	// L1075
  ap_int<33> v699 = v698;	// L1077
  ap_int<33> v700 = 16 - v699;	// L1078
  int v701 = v700;	// L1079
  for (int v702 = 0; v702 < v701; v702 += 1) {	// L1080
  #pragma HLS pipeline II=1
    int32_t v703 = v678.read();	// L1081
    int32_t spare1;	// L1082
    spare1 = v703;	// L1083
  }
  int32_t v705 = v676[v677][1];	// L1085
  int32_t denom1;	// L1086
  denom1 = v705;	// L1087
  int32_t rcp1;	// L1088
  rcp1 = 0;	// L1089
  int32_t v708 = denom1;	// L1090
  bool v709 = v708 > 0;	// L1091
  if (v709) {	// L1092
    int32_t v710 = denom1;	// L1095
    int32_t v711 = 16384 / v710;	// L1096
    rcp1 = v711;	// L1097
  }
  int32_t r01;	// L1099
  r01 = 0;	// L1100
  int32_t r11;	// L1101
  r11 = 0;	// L1102
  int32_t r21;	// L1103
  r21 = 0;	// L1104
  int32_t r31;	// L1105
  r31 = 0;	// L1106
  int32_t pc21;	// L1107
  pc21 = 0;	// L1108
  int32_t v717 = nouts1;	// L1109
  int32_t v718 = plen1;	// L1110
  int64_t v719 = v717;	// L1111
  int64_t v720 = v718;	// L1112
  int64_t v721 = v719 * v720;	// L1113
  int v722 = v721;	// L1114
  for (int v723 = 0; v723 < v722; v723 += 1) {	// L1115
  #pragma HLS pipeline II=1
    int32_t v724 = pc21;	// L1116
    int v725 = v724;	// L1117
    int32_t v726 = prog1[v725];	// L1118
    int32_t word21;	// L1119
    word21 = v726;	// L1120
    int32_t v728 = word21;	// L1121
    int32_t v729 = v728 >> 24;	// L1123
    int32_t v730 = v729 & 255;	// L1125
    int32_t opcode10;	// L1126
    opcode10 = v730;	// L1127
    int32_t v732 = word21;	// L1128
    int32_t v733 = v732 >> 20;	// L1130
    int32_t v734 = v733 & 15;	// L1132
    int32_t dst1;	// L1133
    dst1 = v734;	// L1134
    int32_t v736 = word21;	// L1135
    int32_t v737 = v736 >> 16;	// L1136
    int32_t v738 = v737 & 15;	// L1137
    int32_t src1;	// L1138
    src1 = v738;	// L1139
    int32_t v740 = word21;	// L1140
    int32_t v741 = v740 & 65535;	// L1141
    int32_t imm1;	// L1142
    imm1 = v741;	// L1143
    int32_t v743 = r01;	// L1144
    int32_t d1;	// L1145
    d1 = v743;	// L1146
    int32_t v745 = dst1;	// L1147
    bool v746 = v745 == 1;	// L1148
    if (v746) {	// L1149
      int32_t v747 = r11;	// L1150
      d1 = v747;	// L1151
    } else {
      int32_t v748 = dst1;	// L1153
      bool v749 = v748 == 2;	// L1155
      if (v749) {	// L1156
        int32_t v750 = r21;	// L1157
        d1 = v750;	// L1158
      } else {
        int32_t v751 = dst1;	// L1160
        bool v752 = v751 == 3;	// L1162
        if (v752) {	// L1163
          int32_t v753 = r31;	// L1164
          d1 = v753;	// L1165
        }
      }
    }
    int32_t v754 = r01;	// L1169
    int32_t a10;	// L1170
    a10 = v754;	// L1171
    int32_t v756 = src1;	// L1172
    bool v757 = v756 == 1;	// L1173
    if (v757) {	// L1174
      int32_t v758 = r11;	// L1175
      a10 = v758;	// L1176
    } else {
      int32_t v759 = src1;	// L1178
      bool v760 = v759 == 2;	// L1180
      if (v760) {	// L1181
        int32_t v761 = r21;	// L1182
        a10 = v761;	// L1183
      } else {
        int32_t v762 = src1;	// L1185
        bool v763 = v762 == 3;	// L1187
        if (v763) {	// L1188
          int32_t v764 = r31;	// L1189
          a10 = v764;	// L1190
        }
      }
    }
    int32_t wr1;	// L1194
    wr1 = 1;	// L1195
    int32_t v766 = opcode10;	// L1196
    bool v767 = v766 == 9;	// L1198
    if (v767) {	// L1199
      int32_t v768 = v680.read();	// L1200
      int32_t zz1;	// L1201
      zz1 = v768;	// L1202
      int32_t v770 = d1;	// L1203
      int32_t v771 = zz1;	// L1204
      ap_int<33> v772 = v770;	// L1205
      ap_int<33> v773 = v771;	// L1206
      ap_int<33> v774 = v772 + v773;	// L1207
      int32_t v775 = v774;	// L1208
      d1 = v775;	// L1209
    } else {
      int32_t v776 = opcode10;	// L1211
      bool v777 = v776 == 1;	// L1212
      if (v777) {	// L1213
        int32_t v778 = v680.read();	// L1214
        int32_t z21;	// L1215
        z21 = v778;	// L1216
        int32_t v780 = z21;	// L1217
        d1 = v780;	// L1218
      } else {
        int32_t v781 = opcode10;	// L1220
        bool v782 = v781 == 2;	// L1222
        if (v782) {	// L1223
          int32_t v783 = src1;	// L1224
          int v784 = v783;	// L1225
          int32_t v785 = v676[v677][v784];	// L1226
          d1 = v785;	// L1227
        } else {
          int32_t v786 = opcode10;	// L1229
          bool v787 = v786 == 3;	// L1231
          if (v787) {	// L1232
            int32_t v788 = imm1;	// L1233
            d1 = v788;	// L1234
          } else {
            int32_t v789 = opcode10;	// L1236
            bool v790 = v789 == 4;	// L1238
            if (v790) {	// L1239
              int32_t v791 = d1;	// L1240
              int32_t v792 = a10;	// L1241
              ap_int<33> v793 = v791;	// L1242
              ap_int<33> v794 = v792;	// L1243
              ap_int<33> v795 = v793 + v794;	// L1244
              int32_t v796 = v795;	// L1245
              d1 = v796;	// L1246
            } else {
              int32_t v797 = opcode10;	// L1248
              bool v798 = v797 == 5;	// L1250
              if (v798) {	// L1251
                int32_t v799 = d1;	// L1252
                int32_t v800 = a10;	// L1253
                int64_t v801 = v799;	// L1254
                int64_t v802 = v800;	// L1255
                int64_t v803 = v801 * v802;	// L1256
                int32_t v804 = v803;	// L1257
                d1 = v804;	// L1258
              } else {
                int32_t v805 = opcode10;	// L1260
                bool v806 = v805 == 6;	// L1262
                if (v806) {	// L1263
                  int32_t v807 = a10;	// L1264
                  int32_t v808 = d1;	// L1265
                  bool v809 = v807 > v808;	// L1266
                  if (v809) {	// L1267
                    int32_t v810 = a10;	// L1268
                    d1 = v810;	// L1269
                  }
                } else {
                  int32_t v811 = opcode10;	// L1272
                  bool v812 = v811 == 7;	// L1274
                  if (v812) {	// L1275
                    int32_t v813 = d1;	// L1276
                    int32_t v814 = imm1;	// L1277
                    int32_t v815 = v813 >> v814;	// L1278
                    d1 = v815;	// L1279
                  } else {
                    int32_t v816 = opcode10;	// L1281
                    bool v817 = v816 == 10;	// L1283
                    if (v817) {	// L1284
                      int32_t v818 = d1;	// L1285
                      int32_t v819 = a10;	// L1286
                      ap_int<33> v820 = v818;	// L1287
                      ap_int<33> v821 = v819;	// L1288
                      ap_int<33> v822 = v820 - v821;	// L1289
                      int32_t v823 = v822;	// L1290
                      d1 = v823;	// L1291
                    } else {
                      int32_t v824 = opcode10;	// L1293
                      bool v825 = v824 == 11;	// L1295
                      if (v825) {	// L1296
                        int32_t v826 = d1;	// L1297
                        int32_t e1;	// L1298
                        e1 = v826;	// L1299
                        int32_t v828 = e1;	// L1300
                        bool v829 = v828 < 0;	// L1301
                        if (v829) {	// L1302
                          e1 = 0;	// L1303
                        }
                        int32_t v830 = e1;	// L1305
                        bool v831 = v830 > 30;	// L1307
                        if (v831) {	// L1308
                          e1 = 30;	// L1309
                        }
                        int32_t v832 = e1;	// L1311
                        int32_t v833 = 1 << v832;	// L1312
                        d1 = v833;	// L1313
                      } else {
                        int32_t v834 = opcode10;	// L1315
                        bool v835 = v834 == 12;	// L1317
                        if (v835) {	// L1318
                          int32_t v836 = rcp1;	// L1319
                          d1 = v836;	// L1320
                        } else {
                          int32_t v837 = opcode10;	// L1322
                          bool v838 = v837 == 8;	// L1324
                          if (v838) {	// L1325
                            int32_t v839 = d1;	// L1326
                            v679.write(v839);	// L1327
                            wr1 = 0;	// L1328
                          } else {
                            wr1 = 0;	// L1330
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
    int32_t v840 = wr1;	// L1343
    bool v841 = v840 == 1;	// L1344
    if (v841) {	// L1345
      int32_t v842 = dst1;	// L1346
      bool v843 = v842 == 0;	// L1347
      if (v843) {	// L1348
        int32_t v844 = d1;	// L1349
        r01 = v844;	// L1350
      } else {
        int32_t v845 = dst1;	// L1352
        bool v846 = v845 == 1;	// L1353
        if (v846) {	// L1354
          int32_t v847 = d1;	// L1355
          r11 = v847;	// L1356
        } else {
          int32_t v848 = dst1;	// L1358
          bool v849 = v848 == 2;	// L1360
          if (v849) {	// L1361
            int32_t v850 = d1;	// L1362
            r21 = v850;	// L1363
          } else {
            int32_t v851 = d1;	// L1365
            r31 = v851;	// L1366
          }
        }
      }
    }
    int32_t v852 = pc21;	// L1371
    ap_int<33> v853 = v852;	// L1372
    ap_int<33> v854 = v853 + 1;	// L1374
    int32_t v855 = v854;	// L1375
    pc21 = v855;	// L1376
    int32_t v856 = pc21;	// L1377
    int32_t v857 = plen1;	// L1378
    bool v858 = v856 == v857;	// L1379
    if (v858) {	// L1380
      pc21 = 0;	// L1381
    }
  }
}

void vpu_r2(
  int32_t v859[4][2],
  int v860,
  hls::stream< int32_t >& v861,
  hls::stream< int32_t >& v862,
  hls::stream< int32_t >& v863,
  hls::stream< int32_t >& v864
) {	// L1386
  #pragma HLS array_partition variable=v859 complete dim=1
  #pragma HLS array_partition variable=v859 complete dim=2

  int32_t v865 = v861.read();	// L1387
  int32_t header2;	// L1388
  header2 = v865;	// L1389
  int32_t v867 = header2;	// L1390
  v862.write(v867);	// L1391
  int32_t v868 = header2;	// L1392
  int32_t v869 = v868 & 65535;	// L1394
  int32_t plen2;	// L1395
  plen2 = v869;	// L1396
  int32_t v871 = header2;	// L1397
  int32_t v872 = v871 >> 16;	// L1399
  int32_t v873 = v872 & 65535;	// L1400
  int32_t nouts2;	// L1401
  nouts2 = v873;	// L1402
  int32_t prog2[16];	// L1403
  for (int v876 = 0; v876 < 16; v876++) {	// L1405
    prog2[v876] = 0;	// L1405
  }
  int32_t v877 = plen2;	// L1406
  int v878 = v877;	// L1408
  for (int v879 = 0; v879 < v878; v879 += 1) {	// L1411
  #pragma HLS pipeline II=1
    int32_t v880 = v861.read();	// L1412
    int32_t word11;	// L1413
    word11 = v880;	// L1414
    int32_t v882 = word11;	// L1415
    prog2[v879] = v882;	// L1416
    int32_t v883 = word11;	// L1417
    v862.write(v883);	// L1418
  }
  int32_t v884 = plen2;	// L1420
  ap_int<33> v885 = v884;	// L1422
  ap_int<33> v886 = 16 - v885;	// L1423
  int v887 = v886;	// L1424
  for (int v888 = 0; v888 < v887; v888 += 1) {	// L1425
  #pragma HLS pipeline II=1
    int32_t v889 = v861.read();	// L1426
    int32_t spare2;	// L1427
    spare2 = v889;	// L1428
    int32_t v891 = spare2;	// L1429
    v862.write(v891);	// L1430
  }
  int32_t v892 = v859[v860][1];	// L1432
  int32_t denom2;	// L1433
  denom2 = v892;	// L1434
  int32_t rcp2;	// L1435
  rcp2 = 0;	// L1436
  int32_t v895 = denom2;	// L1437
  bool v896 = v895 > 0;	// L1438
  if (v896) {	// L1439
    int32_t v897 = denom2;	// L1442
    int32_t v898 = 16384 / v897;	// L1443
    rcp2 = v898;	// L1444
  }
  int32_t r02;	// L1446
  r02 = 0;	// L1447
  int32_t r12;	// L1448
  r12 = 0;	// L1449
  int32_t r22;	// L1450
  r22 = 0;	// L1451
  int32_t r32;	// L1452
  r32 = 0;	// L1453
  int32_t pc22;	// L1454
  pc22 = 0;	// L1455
  int32_t v904 = nouts2;	// L1456
  int32_t v905 = plen2;	// L1457
  int64_t v906 = v904;	// L1458
  int64_t v907 = v905;	// L1459
  int64_t v908 = v906 * v907;	// L1460
  int v909 = v908;	// L1461
  for (int v910 = 0; v910 < v909; v910 += 1) {	// L1462
  #pragma HLS pipeline II=1
    int32_t v911 = pc22;	// L1463
    int v912 = v911;	// L1464
    int32_t v913 = prog2[v912];	// L1465
    int32_t word22;	// L1466
    word22 = v913;	// L1467
    int32_t v915 = word22;	// L1468
    int32_t v916 = v915 >> 24;	// L1470
    int32_t v917 = v916 & 255;	// L1472
    int32_t opcode11;	// L1473
    opcode11 = v917;	// L1474
    int32_t v919 = word22;	// L1475
    int32_t v920 = v919 >> 20;	// L1477
    int32_t v921 = v920 & 15;	// L1479
    int32_t dst2;	// L1480
    dst2 = v921;	// L1481
    int32_t v923 = word22;	// L1482
    int32_t v924 = v923 >> 16;	// L1483
    int32_t v925 = v924 & 15;	// L1484
    int32_t src2;	// L1485
    src2 = v925;	// L1486
    int32_t v927 = word22;	// L1487
    int32_t v928 = v927 & 65535;	// L1488
    int32_t imm2;	// L1489
    imm2 = v928;	// L1490
    int32_t v930 = r02;	// L1491
    int32_t d2;	// L1492
    d2 = v930;	// L1493
    int32_t v932 = dst2;	// L1494
    bool v933 = v932 == 1;	// L1495
    if (v933) {	// L1496
      int32_t v934 = r12;	// L1497
      d2 = v934;	// L1498
    } else {
      int32_t v935 = dst2;	// L1500
      bool v936 = v935 == 2;	// L1502
      if (v936) {	// L1503
        int32_t v937 = r22;	// L1504
        d2 = v937;	// L1505
      } else {
        int32_t v938 = dst2;	// L1507
        bool v939 = v938 == 3;	// L1509
        if (v939) {	// L1510
          int32_t v940 = r32;	// L1511
          d2 = v940;	// L1512
        }
      }
    }
    int32_t v941 = r02;	// L1516
    int32_t a11;	// L1517
    a11 = v941;	// L1518
    int32_t v943 = src2;	// L1519
    bool v944 = v943 == 1;	// L1520
    if (v944) {	// L1521
      int32_t v945 = r12;	// L1522
      a11 = v945;	// L1523
    } else {
      int32_t v946 = src2;	// L1525
      bool v947 = v946 == 2;	// L1527
      if (v947) {	// L1528
        int32_t v948 = r22;	// L1529
        a11 = v948;	// L1530
      } else {
        int32_t v949 = src2;	// L1532
        bool v950 = v949 == 3;	// L1534
        if (v950) {	// L1535
          int32_t v951 = r32;	// L1536
          a11 = v951;	// L1537
        }
      }
    }
    int32_t wr2;	// L1541
    wr2 = 1;	// L1542
    int32_t v953 = opcode11;	// L1543
    bool v954 = v953 == 9;	// L1545
    if (v954) {	// L1546
      int32_t v955 = v864.read();	// L1547
      int32_t zz2;	// L1548
      zz2 = v955;	// L1549
      int32_t v957 = d2;	// L1550
      int32_t v958 = zz2;	// L1551
      ap_int<33> v959 = v957;	// L1552
      ap_int<33> v960 = v958;	// L1553
      ap_int<33> v961 = v959 + v960;	// L1554
      int32_t v962 = v961;	// L1555
      d2 = v962;	// L1556
    } else {
      int32_t v963 = opcode11;	// L1558
      bool v964 = v963 == 1;	// L1559
      if (v964) {	// L1560
        int32_t v965 = v864.read();	// L1561
        int32_t z22;	// L1562
        z22 = v965;	// L1563
        int32_t v967 = z22;	// L1564
        d2 = v967;	// L1565
      } else {
        int32_t v968 = opcode11;	// L1567
        bool v969 = v968 == 2;	// L1569
        if (v969) {	// L1570
          int32_t v970 = src2;	// L1571
          int v971 = v970;	// L1572
          int32_t v972 = v859[v860][v971];	// L1573
          d2 = v972;	// L1574
        } else {
          int32_t v973 = opcode11;	// L1576
          bool v974 = v973 == 3;	// L1578
          if (v974) {	// L1579
            int32_t v975 = imm2;	// L1580
            d2 = v975;	// L1581
          } else {
            int32_t v976 = opcode11;	// L1583
            bool v977 = v976 == 4;	// L1585
            if (v977) {	// L1586
              int32_t v978 = d2;	// L1587
              int32_t v979 = a11;	// L1588
              ap_int<33> v980 = v978;	// L1589
              ap_int<33> v981 = v979;	// L1590
              ap_int<33> v982 = v980 + v981;	// L1591
              int32_t v983 = v982;	// L1592
              d2 = v983;	// L1593
            } else {
              int32_t v984 = opcode11;	// L1595
              bool v985 = v984 == 5;	// L1597
              if (v985) {	// L1598
                int32_t v986 = d2;	// L1599
                int32_t v987 = a11;	// L1600
                int64_t v988 = v986;	// L1601
                int64_t v989 = v987;	// L1602
                int64_t v990 = v988 * v989;	// L1603
                int32_t v991 = v990;	// L1604
                d2 = v991;	// L1605
              } else {
                int32_t v992 = opcode11;	// L1607
                bool v993 = v992 == 6;	// L1609
                if (v993) {	// L1610
                  int32_t v994 = a11;	// L1611
                  int32_t v995 = d2;	// L1612
                  bool v996 = v994 > v995;	// L1613
                  if (v996) {	// L1614
                    int32_t v997 = a11;	// L1615
                    d2 = v997;	// L1616
                  }
                } else {
                  int32_t v998 = opcode11;	// L1619
                  bool v999 = v998 == 7;	// L1621
                  if (v999) {	// L1622
                    int32_t v1000 = d2;	// L1623
                    int32_t v1001 = imm2;	// L1624
                    int32_t v1002 = v1000 >> v1001;	// L1625
                    d2 = v1002;	// L1626
                  } else {
                    int32_t v1003 = opcode11;	// L1628
                    bool v1004 = v1003 == 10;	// L1630
                    if (v1004) {	// L1631
                      int32_t v1005 = d2;	// L1632
                      int32_t v1006 = a11;	// L1633
                      ap_int<33> v1007 = v1005;	// L1634
                      ap_int<33> v1008 = v1006;	// L1635
                      ap_int<33> v1009 = v1007 - v1008;	// L1636
                      int32_t v1010 = v1009;	// L1637
                      d2 = v1010;	// L1638
                    } else {
                      int32_t v1011 = opcode11;	// L1640
                      bool v1012 = v1011 == 11;	// L1642
                      if (v1012) {	// L1643
                        int32_t v1013 = d2;	// L1644
                        int32_t e2;	// L1645
                        e2 = v1013;	// L1646
                        int32_t v1015 = e2;	// L1647
                        bool v1016 = v1015 < 0;	// L1648
                        if (v1016) {	// L1649
                          e2 = 0;	// L1650
                        }
                        int32_t v1017 = e2;	// L1652
                        bool v1018 = v1017 > 30;	// L1654
                        if (v1018) {	// L1655
                          e2 = 30;	// L1656
                        }
                        int32_t v1019 = e2;	// L1658
                        int32_t v1020 = 1 << v1019;	// L1659
                        d2 = v1020;	// L1660
                      } else {
                        int32_t v1021 = opcode11;	// L1662
                        bool v1022 = v1021 == 12;	// L1664
                        if (v1022) {	// L1665
                          int32_t v1023 = rcp2;	// L1666
                          d2 = v1023;	// L1667
                        } else {
                          int32_t v1024 = opcode11;	// L1669
                          bool v1025 = v1024 == 8;	// L1671
                          if (v1025) {	// L1672
                            int32_t v1026 = d2;	// L1673
                            v863.write(v1026);	// L1674
                            wr2 = 0;	// L1675
                          } else {
                            wr2 = 0;	// L1677
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
    int32_t v1027 = wr2;	// L1690
    bool v1028 = v1027 == 1;	// L1691
    if (v1028) {	// L1692
      int32_t v1029 = dst2;	// L1693
      bool v1030 = v1029 == 0;	// L1694
      if (v1030) {	// L1695
        int32_t v1031 = d2;	// L1696
        r02 = v1031;	// L1697
      } else {
        int32_t v1032 = dst2;	// L1699
        bool v1033 = v1032 == 1;	// L1700
        if (v1033) {	// L1701
          int32_t v1034 = d2;	// L1702
          r12 = v1034;	// L1703
        } else {
          int32_t v1035 = dst2;	// L1705
          bool v1036 = v1035 == 2;	// L1707
          if (v1036) {	// L1708
            int32_t v1037 = d2;	// L1709
            r22 = v1037;	// L1710
          } else {
            int32_t v1038 = d2;	// L1712
            r32 = v1038;	// L1713
          }
        }
      }
    }
    int32_t v1039 = pc22;	// L1718
    ap_int<33> v1040 = v1039;	// L1719
    ap_int<33> v1041 = v1040 + 1;	// L1721
    int32_t v1042 = v1041;	// L1722
    pc22 = v1042;	// L1723
    int32_t v1043 = pc22;	// L1724
    int32_t v1044 = plen2;	// L1725
    bool v1045 = v1043 == v1044;	// L1726
    if (v1045) {	// L1727
      pc22 = 0;	// L1728
    }
  }
}

void vpu_y_out_drain(
  int32_t v1046[4][4],
  int v1047,
  hls::stream< int32_t >& v1048
) {	// L1733
  #pragma HLS array_partition variable=v1046 complete dim=1
  #pragma HLS array_partition variable=v1046 complete dim=2

  l_S__t_0__t3: for (int _t3 = 0; _t3 < 4; _t3++) {	// L1734
  #pragma HLS pipeline II=1
    int32_t v1050 = v1048.read();	// L1735
    v1046[_t3][v1047] = v1050;	// L1736
  }
}

/// This is top function.
void top(
  int8_t v1051[4][4],
  int32_t v1052[5][4],
  int32_t v1053[17],
  int8_t v1054[4][4][4],
  int32_t v1055[4][2],
  int32_t v1056[4][4]
) {	// L1740
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v1051 complete dim=1
  #pragma HLS array_partition variable=v1051 complete dim=2

  #pragma HLS array_partition variable=v1052 complete dim=1
  #pragma HLS array_partition variable=v1052 complete dim=2

  #pragma HLS array_partition variable=v1053 complete dim=1

  #pragma HLS array_partition variable=v1054 complete dim=1
  #pragma HLS array_partition variable=v1054 complete dim=2
  #pragma HLS array_partition variable=v1054 complete dim=3

  #pragma HLS array_partition variable=v1055 complete dim=1
  #pragma HLS array_partition variable=v1055 complete dim=2

  #pragma HLS array_partition variable=v1056 complete dim=1
  #pragma HLS array_partition variable=v1056 complete dim=2

  hls::stream< int32_t > v1057;
  #pragma HLS stream variable=v1057 depth=4	// L1741
  hls::stream< int32_t > v1058;
  #pragma HLS stream variable=v1058 depth=4	// L1742
  hls::stream< int32_t > v1059;
  #pragma HLS stream variable=v1059 depth=2	// L1743
  hls::stream< int32_t > v1060;
  #pragma HLS stream variable=v1060 depth=4	// L1744
  hls::stream< int32_t > v1061;
  #pragma HLS stream variable=v1061 depth=2	// L1745
  hls::stream< int32_t > v1062;
  #pragma HLS stream variable=v1062 depth=4	// L1746
  hls::stream< int32_t > v1063;
  #pragma HLS stream variable=v1063 depth=2	// L1747
  hls::stream< int32_t > v1064;
  #pragma HLS stream variable=v1064 depth=2	// L1748
  hls::stream< int32_t > v1065;
  #pragma HLS stream variable=v1065 depth=2	// L1749
  hls::stream< int32_t > v1066;
  #pragma HLS stream variable=v1066 depth=2	// L1750
  hls::stream< int8_t > v1067;
  #pragma HLS stream variable=v1067 depth=2	// L1751
  hls::stream< int32_t > v1068;
  #pragma HLS stream variable=v1068 depth=2	// L1752
  hls::stream< int32_t > v1069;
  #pragma HLS stream variable=v1069 depth=2	// L1753
  hls::stream< int8_t > v1070;
  #pragma HLS stream variable=v1070 depth=2	// L1754
  hls::stream< int32_t > v1071;
  #pragma HLS stream variable=v1071 depth=2	// L1755
  hls::stream< int32_t > v1072;
  #pragma HLS stream variable=v1072 depth=2	// L1756
  hls::stream< int8_t > v1073;
  #pragma HLS stream variable=v1073 depth=2	// L1757
  hls::stream< int32_t > v1074;
  #pragma HLS stream variable=v1074 depth=2	// L1758
  hls::stream< int32_t > v1075;
  #pragma HLS stream variable=v1075 depth=2	// L1759
  hls::stream< int32_t > v1076;
  #pragma HLS stream variable=v1076 depth=2	// L1760
  hls::stream< int8_t > v1077;
  #pragma HLS stream variable=v1077 depth=2	// L1761
  hls::stream< int32_t > v1078;
  #pragma HLS stream variable=v1078 depth=2	// L1762
  hls::stream< int32_t > v1079;
  #pragma HLS stream variable=v1079 depth=2	// L1763
  hls::stream< int8_t > v1080;
  #pragma HLS stream variable=v1080 depth=2	// L1764
  hls::stream< int32_t > v1081;
  #pragma HLS stream variable=v1081 depth=2	// L1765
  hls::stream< int32_t > v1082;
  #pragma HLS stream variable=v1082 depth=2	// L1766
  hls::stream< int8_t > v1083;
  #pragma HLS stream variable=v1083 depth=2	// L1767
  hls::stream< int32_t > v1084;
  #pragma HLS stream variable=v1084 depth=2	// L1768
  hls::stream< int32_t > v1085;
  #pragma HLS stream variable=v1085 depth=2	// L1769
  hls::stream< int32_t > v1086;
  #pragma HLS stream variable=v1086 depth=2	// L1770
  hls::stream< int8_t > v1087;
  #pragma HLS stream variable=v1087 depth=2	// L1771
  hls::stream< int32_t > v1088;
  #pragma HLS stream variable=v1088 depth=2	// L1772
  hls::stream< int32_t > v1089;
  #pragma HLS stream variable=v1089 depth=2	// L1773
  hls::stream< int8_t > v1090;
  #pragma HLS stream variable=v1090 depth=2	// L1774
  hls::stream< int32_t > v1091;
  #pragma HLS stream variable=v1091 depth=2	// L1775
  hls::stream< int32_t > v1092;
  #pragma HLS stream variable=v1092 depth=2	// L1776
  hls::stream< int8_t > v1093;
  #pragma HLS stream variable=v1093 depth=2	// L1777
  hls::stream< int32_t > v1094;
  #pragma HLS stream variable=v1094 depth=2	// L1778
  hls::stream< int32_t > v1095;
  #pragma HLS stream variable=v1095 depth=2	// L1779
  hls::stream< int32_t > v1096;
  #pragma HLS stream variable=v1096 depth=2	// L1780
  hls::stream< int8_t > v1097;
  #pragma HLS stream variable=v1097 depth=2	// L1781
  hls::stream< int32_t > v1098;
  #pragma HLS stream variable=v1098 depth=2	// L1782
  hls::stream< int32_t > v1099;
  #pragma HLS stream variable=v1099 depth=2	// L1783
  hls::stream< int8_t > v1100;
  #pragma HLS stream variable=v1100 depth=2	// L1784
  hls::stream< int32_t > v1101;
  #pragma HLS stream variable=v1101 depth=2	// L1785
  hls::stream< int32_t > v1102;
  #pragma HLS stream variable=v1102 depth=2	// L1786
  hls::stream< int8_t > v1103;
  #pragma HLS stream variable=v1103 depth=2	// L1787
  hls::stream< int32_t > v1104;
  #pragma HLS stream variable=v1104 depth=17	// L1788
  hls::stream< int32_t > v1105;
  #pragma HLS stream variable=v1105 depth=5	// L1789
  hls::stream< int32_t > v1106;
  #pragma HLS stream variable=v1106 depth=5	// L1790
  hls::stream< int32_t > v1107;
  #pragma HLS stream variable=v1107 depth=5	// L1791
  hls::stream< int32_t > v1108;
  #pragma HLS stream variable=v1108 depth=5	// L1792
  hls::stream< int8_t > v1109;
  #pragma HLS stream variable=v1109 depth=4	// L1793
  hls::stream< int8_t > v1110;
  #pragma HLS stream variable=v1110 depth=4	// L1795
  hls::stream< int8_t > v1111;
  #pragma HLS stream variable=v1111 depth=4	// L1797
  hls::stream< int8_t > v1112;
  #pragma HLS stream variable=v1112 depth=4	// L1799
  mac_a_in_load(v1051, 0, v1112);	// L1801
  mac_a_in_load(v1051, 1, v1111);	// L1802
  mac_a_in_load(v1051, 2, v1110);	// L1803
  mac_a_in_load(v1051, 3, v1109);	// L1804
  mac_op_in_load(v1052, 0, v1108);	// L1805
  mac_op_in_load(v1052, 1, v1107);	// L1806
  mac_op_in_load(v1052, 2, v1106);	// L1807
  mac_op_in_load(v1052, 3, v1105);	// L1808
  vpu_op_in_load(v1053, 0, v1104);	// L1809
  mac_r8(v1054, 0, 0, v1112, v1103, v1108, v1102, v1101);	// L1810
  mac_r2(v1054, 0, 1, v1103, v1100, v1102, v1099, v1098);	// L1811
  mac_r2(v1054, 0, 2, v1100, v1097, v1099, v1096, v1095);	// L1812
  mac_r6(v1054, 0, 3, v1097, v1096, v1094);	// L1813
  mac_r4(v1054, 1, 0, v1111, v1093, v1107, v1092, v1101, v1091);	// L1814
  mac_r0(v1054, 1, 1, v1093, v1090, v1092, v1089, v1098, v1088);	// L1815
  mac_r0(v1054, 1, 2, v1090, v1087, v1089, v1086, v1095, v1085);	// L1816
  mac_r3(v1054, 1, 3, v1087, v1086, v1094, v1084);	// L1817
  mac_r4(v1054, 2, 0, v1110, v1083, v1106, v1082, v1091, v1081);	// L1818
  mac_r0(v1054, 2, 1, v1083, v1080, v1082, v1079, v1088, v1078);	// L1819
  mac_r0(v1054, 2, 2, v1080, v1077, v1079, v1076, v1085, v1075);	// L1820
  mac_r3(v1054, 2, 3, v1077, v1076, v1084, v1074);	// L1821
  mac_r7(v1054, 3, 0, v1109, v1073, v1105, v1072, v1081, v1071);	// L1822
  mac_r1(v1054, 3, 1, v1073, v1070, v1072, v1069, v1078, v1068);	// L1823
  mac_r1(v1054, 3, 2, v1070, v1067, v1069, v1066, v1075, v1065);	// L1824
  mac_r5(v1054, 3, 3, v1067, v1066, v1074, v1064);	// L1825
  vpu_r2(v1055, 0, v1104, v1063, v1062, v1071);	// L1826
  vpu_r0(v1055, 1, v1063, v1061, v1060, v1068);	// L1827
  vpu_r0(v1055, 2, v1061, v1059, v1058, v1065);	// L1828
  vpu_r1(v1055, 3, v1059, v1057, v1064);	// L1829
  vpu_y_out_drain(v1056, 0, v1062);	// L1830
  vpu_y_out_drain(v1056, 1, v1060);	// L1831
  vpu_y_out_drain(v1056, 2, v1058);	// L1832
  vpu_y_out_drain(v1056, 3, v1057);	// L1833
}



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
void tiled_mac_a_in_load(
  int8_t v0[12][4],
  int v1,
  hls::stream< int8_t >& v2
) {	// L5
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 12; _t++) {	// L6
  #pragma HLS pipeline II=1
    int8_t v4 = v0[_t][v1];	// L7
    v2.write(v4);	// L8
  }
}

void tiled_vpu_op_in_load(
  int32_t v5[12],
  int v6,
  hls::stream< int32_t >& v7
) {	// L12
  #pragma HLS array_partition variable=v5 complete dim=1

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 12; _t1++) {	// L13
  #pragma HLS pipeline II=1
    int32_t v9 = v5[_t1];	// L14
    v7.write(v9);	// L15
  }
}

void tiled_mac_r0(
  int8_t v10[4][4][2],
  int v11,
  int v12,
  hls::stream< int8_t >& v13,
  hls::stream< int8_t >& v14,
  hls::stream< int32_t >& v15,
  hls::stream< int32_t >& v16
) {	// L19
  #pragma HLS array_partition variable=v10 complete dim=1
  #pragma HLS array_partition variable=v10 complete dim=2
  #pragma HLS array_partition variable=v10 complete dim=3

  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L20
    l_S_t_0_t: for (int t = 0; t < 2; t++) {	// L21
    #pragma HLS pipeline II=1
      int8_t v19 = v13.read();	// L22
      int8_t a;	// L23
      a = v19;	// L24
      int32_t v21 = v15.read();	// L25
      int32_t p;	// L26
      p = v21;	// L27
      int8_t v23 = v10[v11][v12][t];	// L28
      int32_t v24 = v23;	// L29
      int32_t wt;	// L30
      wt = v24;	// L31
      int32_t v26 = p;	// L32
      int8_t v27 = a;	// L33
      int32_t v28 = wt;	// L34
      ap_int<40> v29 = v27;	// L35
      ap_int<40> v30 = v28;	// L36
      ap_int<40> v31 = v29 * v30;	// L37
      ap_int<41> v32 = v26;	// L38
      ap_int<41> v33 = v31;	// L39
      ap_int<41> v34 = v32 + v33;	// L40
      v16.write(v34);	// L41
      int8_t v35 = a;	// L42
      v14.write(v35);	// L43
    }
  }
}

void tiled_mac_r1(
  int8_t v36[4][4][2],
  int v37,
  int v38,
  hls::stream< int8_t >& v39,
  hls::stream< int8_t >& v40,
  hls::stream< int32_t >& v41,
  hls::stream< int32_t >& v42
) {	// L48
  #pragma HLS array_partition variable=v36 complete dim=1
  #pragma HLS array_partition variable=v36 complete dim=2
  #pragma HLS array_partition variable=v36 complete dim=3

  l_S_m_0_m1: for (int m1 = 0; m1 < 6; m1++) {	// L49
    l_S_t_0_t1: for (int t1 = 0; t1 < 2; t1++) {	// L50
    #pragma HLS pipeline II=1
      int8_t v45 = v39.read();	// L51
      int8_t a1;	// L52
      a1 = v45;	// L53
      int32_t v47 = v41.read();	// L54
      int32_t p1;	// L55
      p1 = v47;	// L56
      int8_t v49 = v36[v37][v38][t1];	// L57
      int32_t v50 = v49;	// L58
      int32_t wt1;	// L59
      wt1 = v50;	// L60
      int32_t v52 = p1;	// L61
      int8_t v53 = a1;	// L62
      int32_t v54 = wt1;	// L63
      ap_int<40> v55 = v53;	// L64
      ap_int<40> v56 = v54;	// L65
      ap_int<40> v57 = v55 * v56;	// L66
      ap_int<41> v58 = v52;	// L67
      ap_int<41> v59 = v57;	// L68
      ap_int<41> v60 = v58 + v59;	// L69
      v42.write(v60);	// L70
      int8_t v61 = a1;	// L71
      v40.write(v61);	// L72
    }
  }
}

void tiled_mac_r2(
  int8_t v62[4][4][2],
  int v63,
  int v64,
  hls::stream< int8_t >& v65,
  hls::stream< int8_t >& v66,
  hls::stream< int32_t >& v67
) {	// L77
  #pragma HLS array_partition variable=v62 complete dim=1
  #pragma HLS array_partition variable=v62 complete dim=2
  #pragma HLS array_partition variable=v62 complete dim=3

  l_S_m_0_m2: for (int m2 = 0; m2 < 6; m2++) {	// L78
    l_S_t_0_t2: for (int t2 = 0; t2 < 2; t2++) {	// L79
    #pragma HLS pipeline II=1
      int8_t v70 = v65.read();	// L80
      int8_t a2;	// L81
      a2 = v70;	// L82
      int32_t p2;	// L84
      p2 = 0;	// L85
      int8_t v73 = v62[v63][v64][t2];	// L86
      int32_t v74 = v73;	// L87
      int32_t wt2;	// L88
      wt2 = v74;	// L89
      int32_t v76 = p2;	// L90
      int8_t v77 = a2;	// L91
      int32_t v78 = wt2;	// L92
      ap_int<40> v79 = v77;	// L93
      ap_int<40> v80 = v78;	// L94
      ap_int<40> v81 = v79 * v80;	// L95
      ap_int<41> v82 = v76;	// L96
      ap_int<41> v83 = v81;	// L97
      ap_int<41> v84 = v82 + v83;	// L98
      v67.write(v84);	// L99
      int8_t v85 = a2;	// L100
      v66.write(v85);	// L101
    }
  }
}

void tiled_mac_r3(
  int8_t v86[4][4][2],
  int v87,
  int v88,
  hls::stream< int8_t >& v89,
  hls::stream< int32_t >& v90,
  hls::stream< int32_t >& v91
) {	// L106
  #pragma HLS array_partition variable=v86 complete dim=1
  #pragma HLS array_partition variable=v86 complete dim=2
  #pragma HLS array_partition variable=v86 complete dim=3

  l_S_m_0_m3: for (int m3 = 0; m3 < 6; m3++) {	// L107
    l_S_t_0_t3: for (int t3 = 0; t3 < 2; t3++) {	// L108
    #pragma HLS pipeline II=1
      int8_t v94 = v89.read();	// L109
      int8_t a3;	// L110
      a3 = v94;	// L111
      int32_t v96 = v90.read();	// L112
      int32_t p3;	// L113
      p3 = v96;	// L114
      int8_t v98 = v86[v87][v88][t3];	// L115
      int32_t v99 = v98;	// L116
      int32_t wt3;	// L117
      wt3 = v99;	// L118
      int32_t v101 = p3;	// L119
      int8_t v102 = a3;	// L120
      int32_t v103 = wt3;	// L121
      ap_int<40> v104 = v102;	// L122
      ap_int<40> v105 = v103;	// L123
      ap_int<40> v106 = v104 * v105;	// L124
      ap_int<41> v107 = v101;	// L125
      ap_int<41> v108 = v106;	// L126
      ap_int<41> v109 = v107 + v108;	// L127
      v91.write(v109);	// L128
    }
  }
}

void tiled_mac_r4(
  int8_t v110[4][4][2],
  int v111,
  int v112,
  hls::stream< int8_t >& v113,
  hls::stream< int8_t >& v114,
  hls::stream< int32_t >& v115,
  hls::stream< int32_t >& v116
) {	// L133
  #pragma HLS array_partition variable=v110 complete dim=1
  #pragma HLS array_partition variable=v110 complete dim=2
  #pragma HLS array_partition variable=v110 complete dim=3

  l_S_m_0_m4: for (int m4 = 0; m4 < 6; m4++) {	// L134
    l_S_t_0_t4: for (int t4 = 0; t4 < 2; t4++) {	// L135
    #pragma HLS pipeline II=1
      int8_t v119 = v113.read();	// L136
      int8_t a4;	// L137
      a4 = v119;	// L138
      int32_t v121 = v115.read();	// L139
      int32_t p4;	// L140
      p4 = v121;	// L141
      int8_t v123 = v110[v111][v112][t4];	// L142
      int32_t v124 = v123;	// L143
      int32_t wt4;	// L144
      wt4 = v124;	// L145
      int32_t v126 = p4;	// L146
      int8_t v127 = a4;	// L147
      int32_t v128 = wt4;	// L148
      ap_int<40> v129 = v127;	// L149
      ap_int<40> v130 = v128;	// L150
      ap_int<40> v131 = v129 * v130;	// L151
      ap_int<41> v132 = v126;	// L152
      ap_int<41> v133 = v131;	// L153
      ap_int<41> v134 = v132 + v133;	// L154
      v116.write(v134);	// L155
      int8_t v135 = a4;	// L156
      v114.write(v135);	// L157
    }
  }
}

void tiled_mac_r5(
  int8_t v136[4][4][2],
  int v137,
  int v138,
  hls::stream< int8_t >& v139,
  hls::stream< int32_t >& v140,
  hls::stream< int32_t >& v141
) {	// L162
  #pragma HLS array_partition variable=v136 complete dim=1
  #pragma HLS array_partition variable=v136 complete dim=2
  #pragma HLS array_partition variable=v136 complete dim=3

  l_S_m_0_m5: for (int m5 = 0; m5 < 6; m5++) {	// L163
    l_S_t_0_t5: for (int t5 = 0; t5 < 2; t5++) {	// L164
    #pragma HLS pipeline II=1
      int8_t v144 = v139.read();	// L165
      int8_t a5;	// L166
      a5 = v144;	// L167
      int32_t v146 = v140.read();	// L168
      int32_t p5;	// L169
      p5 = v146;	// L170
      int8_t v148 = v136[v137][v138][t5];	// L171
      int32_t v149 = v148;	// L172
      int32_t wt5;	// L173
      wt5 = v149;	// L174
      int32_t v151 = p5;	// L175
      int8_t v152 = a5;	// L176
      int32_t v153 = wt5;	// L177
      ap_int<40> v154 = v152;	// L178
      ap_int<40> v155 = v153;	// L179
      ap_int<40> v156 = v154 * v155;	// L180
      ap_int<41> v157 = v151;	// L181
      ap_int<41> v158 = v156;	// L182
      ap_int<41> v159 = v157 + v158;	// L183
      v141.write(v159);	// L184
    }
  }
}

void tiled_mac_r6(
  int8_t v160[4][4][2],
  int v161,
  int v162,
  hls::stream< int8_t >& v163,
  hls::stream< int32_t >& v164
) {	// L189
  #pragma HLS array_partition variable=v160 complete dim=1
  #pragma HLS array_partition variable=v160 complete dim=2
  #pragma HLS array_partition variable=v160 complete dim=3

  l_S_m_0_m6: for (int m6 = 0; m6 < 6; m6++) {	// L190
    l_S_t_0_t6: for (int t6 = 0; t6 < 2; t6++) {	// L191
    #pragma HLS pipeline II=1
      int8_t v167 = v163.read();	// L192
      int8_t a6;	// L193
      a6 = v167;	// L194
      int32_t p6;	// L196
      p6 = 0;	// L197
      int8_t v170 = v160[v161][v162][t6];	// L198
      int32_t v171 = v170;	// L199
      int32_t wt6;	// L200
      wt6 = v171;	// L201
      int32_t v173 = p6;	// L202
      int8_t v174 = a6;	// L203
      int32_t v175 = wt6;	// L204
      ap_int<40> v176 = v174;	// L205
      ap_int<40> v177 = v175;	// L206
      ap_int<40> v178 = v176 * v177;	// L207
      ap_int<41> v179 = v173;	// L208
      ap_int<41> v180 = v178;	// L209
      ap_int<41> v181 = v179 + v180;	// L210
      v164.write(v181);	// L211
    }
  }
}

void tiled_mac_r7(
  int8_t v182[4][4][2],
  int v183,
  int v184,
  hls::stream< int8_t >& v185,
  hls::stream< int8_t >& v186,
  hls::stream< int32_t >& v187,
  hls::stream< int32_t >& v188
) {	// L216
  #pragma HLS array_partition variable=v182 complete dim=1
  #pragma HLS array_partition variable=v182 complete dim=2
  #pragma HLS array_partition variable=v182 complete dim=3

  l_S_m_0_m7: for (int m7 = 0; m7 < 6; m7++) {	// L217
    l_S_t_0_t7: for (int t7 = 0; t7 < 2; t7++) {	// L218
    #pragma HLS pipeline II=1
      int8_t v191 = v185.read();	// L219
      int8_t a7;	// L220
      a7 = v191;	// L221
      int32_t v193 = v187.read();	// L222
      int32_t p7;	// L223
      p7 = v193;	// L224
      int8_t v195 = v182[v183][v184][t7];	// L225
      int32_t v196 = v195;	// L226
      int32_t wt7;	// L227
      wt7 = v196;	// L228
      int32_t v198 = p7;	// L229
      int8_t v199 = a7;	// L230
      int32_t v200 = wt7;	// L231
      ap_int<40> v201 = v199;	// L232
      ap_int<40> v202 = v200;	// L233
      ap_int<40> v203 = v201 * v202;	// L234
      ap_int<41> v204 = v198;	// L235
      ap_int<41> v205 = v203;	// L236
      ap_int<41> v206 = v204 + v205;	// L237
      v188.write(v206);	// L238
      int8_t v207 = a7;	// L239
      v186.write(v207);	// L240
    }
  }
}

void tiled_mac_r8(
  int8_t v208[4][4][2],
  int v209,
  int v210,
  hls::stream< int8_t >& v211,
  hls::stream< int8_t >& v212,
  hls::stream< int32_t >& v213
) {	// L245
  #pragma HLS array_partition variable=v208 complete dim=1
  #pragma HLS array_partition variable=v208 complete dim=2
  #pragma HLS array_partition variable=v208 complete dim=3

  l_S_m_0_m8: for (int m8 = 0; m8 < 6; m8++) {	// L246
    l_S_t_0_t8: for (int t8 = 0; t8 < 2; t8++) {	// L247
    #pragma HLS pipeline II=1
      int8_t v216 = v211.read();	// L248
      int8_t a8;	// L249
      a8 = v216;	// L250
      int32_t p8;	// L252
      p8 = 0;	// L253
      int8_t v219 = v208[v209][v210][t8];	// L254
      int32_t v220 = v219;	// L255
      int32_t wt8;	// L256
      wt8 = v220;	// L257
      int32_t v222 = p8;	// L258
      int8_t v223 = a8;	// L259
      int32_t v224 = wt8;	// L260
      ap_int<40> v225 = v223;	// L261
      ap_int<40> v226 = v224;	// L262
      ap_int<40> v227 = v225 * v226;	// L263
      ap_int<41> v228 = v222;	// L264
      ap_int<41> v229 = v227;	// L265
      ap_int<41> v230 = v228 + v229;	// L266
      v213.write(v230);	// L267
      int8_t v231 = a8;	// L268
      v212.write(v231);	// L269
    }
  }
}

void tiled_vpu_r0(
  int32_t v232[4],
  int v233,
  hls::stream< int32_t >& v234,
  hls::stream< int32_t >& v235,
  hls::stream< int32_t >& v236,
  hls::stream< int32_t >& v237
) {	// L274
  #pragma HLS array_partition variable=v232 complete dim=1

  int32_t prog[12];	// L275
  for (int v239 = 0; v239 < 12; v239++) {	// L277
    prog[v239] = 0;	// L277
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 12; pc++) {	// L278
  #pragma HLS pipeline II=1
    int32_t v241 = v234.read();	// L279
    int32_t word;	// L280
    word = v241;	// L281
    int32_t v243 = word;	// L282
    prog[pc] = v243;	// L283
    int32_t v244 = word;	// L284
    v235.write(v244);	// L285
  }
  l_S_m_1_m9: for (int m9 = 0; m9 < 6; m9++) {	// L287
    int32_t reg[4];	// L288
    for (int v247 = 0; v247 < 4; v247++) {	// L289
      reg[v247] = 0;	// L289
    }
    l_S_step_1_step: for (int step = 0; step < 12; step++) {	// L290
    #pragma HLS pipeline II=1
      int32_t v249 = prog[step];	// L291
      int32_t word2;	// L292
      word2 = v249;	// L293
      int32_t v251 = word2;	// L294
      int32_t v252 = v251 >> 24;	// L296
      int32_t v253 = v252 & 255;	// L298
      int32_t opcode;	// L299
      opcode = v253;	// L300
      int32_t v255 = word2;	// L301
      int32_t v256 = v255 >> 20;	// L303
      int32_t v257 = v256 & 15;	// L305
      int32_t dst;	// L306
      dst = v257;	// L307
      int32_t v259 = word2;	// L308
      int32_t v260 = v259 >> 16;	// L310
      int32_t v261 = v260 & 15;	// L311
      int32_t src;	// L312
      src = v261;	// L313
      int32_t v263 = word2;	// L314
      int32_t v264 = v263 & 65535;	// L316
      int32_t imm;	// L317
      imm = v264;	// L318
      int32_t v266 = opcode;	// L319
      bool v267 = v266 == 9;	// L321
      if (v267) {	// L322
        int32_t v268 = v237.read();	// L323
        int32_t zz;	// L324
        zz = v268;	// L325
        int32_t v270 = dst;	// L326
        int v271 = v270;	// L327
        int32_t v272 = reg[v271];	// L328
        int32_t v273 = zz;	// L329
        ap_int<33> v274 = v272;	// L330
        ap_int<33> v275 = v273;	// L331
        ap_int<33> v276 = v274 + v275;	// L332
        int32_t v277 = v276;	// L333
        reg[v271] = v277;	// L334
      } else {
        int32_t v278 = opcode;	// L336
        bool v279 = v278 == 2;	// L338
        if (v279) {	// L339
          int32_t v280 = v232[v233];	// L340
          int32_t v281 = dst;	// L341
          int v282 = v281;	// L342
          reg[v282] = v280;	// L343
        } else {
          int32_t v283 = opcode;	// L345
          bool v284 = v283 == 3;	// L347
          if (v284) {	// L348
            int32_t v285 = imm;	// L349
            int32_t v286 = dst;	// L350
            int v287 = v286;	// L351
            reg[v287] = v285;	// L352
          } else {
            int32_t v288 = opcode;	// L354
            bool v289 = v288 == 4;	// L356
            if (v289) {	// L357
              int32_t v290 = dst;	// L358
              int v291 = v290;	// L359
              int32_t v292 = reg[v291];	// L360
              int32_t v293 = src;	// L361
              int v294 = v293;	// L362
              int32_t v295 = reg[v294];	// L363
              ap_int<33> v296 = v292;	// L364
              ap_int<33> v297 = v295;	// L365
              ap_int<33> v298 = v296 + v297;	// L366
              int32_t v299 = v298;	// L367
              reg[v291] = v299;	// L368
            } else {
              int32_t v300 = opcode;	// L370
              bool v301 = v300 == 5;	// L372
              if (v301) {	// L373
                int32_t v302 = dst;	// L374
                int v303 = v302;	// L375
                int32_t v304 = reg[v303];	// L376
                int32_t v305 = src;	// L377
                int v306 = v305;	// L378
                int32_t v307 = reg[v306];	// L379
                int64_t v308 = v304;	// L380
                int64_t v309 = v307;	// L381
                int64_t v310 = v308 * v309;	// L382
                int32_t v311 = v310;	// L383
                reg[v303] = v311;	// L384
              } else {
                int32_t v312 = opcode;	// L386
                bool v313 = v312 == 6;	// L388
                if (v313) {	// L389
                  int32_t v314 = src;	// L390
                  int v315 = v314;	// L391
                  int32_t v316 = reg[v315];	// L392
                  int32_t v317 = dst;	// L393
                  int v318 = v317;	// L394
                  int32_t v319 = reg[v318];	// L395
                  bool v320 = v316 > v319;	// L396
                  if (v320) {	// L397
                    int32_t v321 = src;	// L398
                    int v322 = v321;	// L399
                    int32_t v323 = reg[v322];	// L400
                    int32_t v324 = dst;	// L401
                    int v325 = v324;	// L402
                    reg[v325] = v323;	// L403
                  }
                } else {
                  int32_t v326 = opcode;	// L406
                  bool v327 = v326 == 7;	// L408
                  if (v327) {	// L409
                    int32_t v328 = dst;	// L410
                    int v329 = v328;	// L411
                    int32_t v330 = reg[v329];	// L412
                    int32_t v331 = imm;	// L413
                    int32_t v332 = v330 >> v331;	// L414
                    reg[v329] = v332;	// L415
                  } else {
                    int32_t v333 = opcode;	// L417
                    bool v334 = v333 == 8;	// L419
                    if (v334) {	// L420
                      int32_t v335 = dst;	// L421
                      int v336 = v335;	// L422
                      int32_t v337 = reg[v336];	// L423
                      v236.write(v337);	// L424
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

void tiled_vpu_r1(
  int32_t v338[4],
  int v339,
  hls::stream< int32_t >& v340,
  hls::stream< int32_t >& v341,
  hls::stream< int32_t >& v342
) {	// L437
  #pragma HLS array_partition variable=v338 complete dim=1

  int32_t prog1[12];	// L438
  for (int v344 = 0; v344 < 12; v344++) {	// L440
    prog1[v344] = 0;	// L440
  }
  l_S_pc_0_pc1: for (int pc1 = 0; pc1 < 12; pc1++) {	// L441
  #pragma HLS pipeline II=1
    int32_t v346 = v340.read();	// L442
    int32_t word1;	// L443
    word1 = v346;	// L444
    int32_t v348 = word1;	// L445
    prog1[pc1] = v348;	// L446
  }
  l_S_m_1_m10: for (int m10 = 0; m10 < 6; m10++) {	// L448
    int32_t reg1[4];	// L449
    for (int v351 = 0; v351 < 4; v351++) {	// L450
      reg1[v351] = 0;	// L450
    }
    l_S_step_1_step1: for (int step1 = 0; step1 < 12; step1++) {	// L451
    #pragma HLS pipeline II=1
      int32_t v353 = prog1[step1];	// L452
      int32_t word21;	// L453
      word21 = v353;	// L454
      int32_t v355 = word21;	// L455
      int32_t v356 = v355 >> 24;	// L457
      int32_t v357 = v356 & 255;	// L459
      int32_t opcode1;	// L460
      opcode1 = v357;	// L461
      int32_t v359 = word21;	// L462
      int32_t v360 = v359 >> 20;	// L464
      int32_t v361 = v360 & 15;	// L466
      int32_t dst1;	// L467
      dst1 = v361;	// L468
      int32_t v363 = word21;	// L469
      int32_t v364 = v363 >> 16;	// L471
      int32_t v365 = v364 & 15;	// L472
      int32_t src1;	// L473
      src1 = v365;	// L474
      int32_t v367 = word21;	// L475
      int32_t v368 = v367 & 65535;	// L477
      int32_t imm1;	// L478
      imm1 = v368;	// L479
      int32_t v370 = opcode1;	// L480
      bool v371 = v370 == 9;	// L482
      if (v371) {	// L483
        int32_t v372 = v342.read();	// L484
        int32_t zz1;	// L485
        zz1 = v372;	// L486
        int32_t v374 = dst1;	// L487
        int v375 = v374;	// L488
        int32_t v376 = reg1[v375];	// L489
        int32_t v377 = zz1;	// L490
        ap_int<33> v378 = v376;	// L491
        ap_int<33> v379 = v377;	// L492
        ap_int<33> v380 = v378 + v379;	// L493
        int32_t v381 = v380;	// L494
        reg1[v375] = v381;	// L495
      } else {
        int32_t v382 = opcode1;	// L497
        bool v383 = v382 == 2;	// L499
        if (v383) {	// L500
          int32_t v384 = v338[v339];	// L501
          int32_t v385 = dst1;	// L502
          int v386 = v385;	// L503
          reg1[v386] = v384;	// L504
        } else {
          int32_t v387 = opcode1;	// L506
          bool v388 = v387 == 3;	// L508
          if (v388) {	// L509
            int32_t v389 = imm1;	// L510
            int32_t v390 = dst1;	// L511
            int v391 = v390;	// L512
            reg1[v391] = v389;	// L513
          } else {
            int32_t v392 = opcode1;	// L515
            bool v393 = v392 == 4;	// L517
            if (v393) {	// L518
              int32_t v394 = dst1;	// L519
              int v395 = v394;	// L520
              int32_t v396 = reg1[v395];	// L521
              int32_t v397 = src1;	// L522
              int v398 = v397;	// L523
              int32_t v399 = reg1[v398];	// L524
              ap_int<33> v400 = v396;	// L525
              ap_int<33> v401 = v399;	// L526
              ap_int<33> v402 = v400 + v401;	// L527
              int32_t v403 = v402;	// L528
              reg1[v395] = v403;	// L529
            } else {
              int32_t v404 = opcode1;	// L531
              bool v405 = v404 == 5;	// L533
              if (v405) {	// L534
                int32_t v406 = dst1;	// L535
                int v407 = v406;	// L536
                int32_t v408 = reg1[v407];	// L537
                int32_t v409 = src1;	// L538
                int v410 = v409;	// L539
                int32_t v411 = reg1[v410];	// L540
                int64_t v412 = v408;	// L541
                int64_t v413 = v411;	// L542
                int64_t v414 = v412 * v413;	// L543
                int32_t v415 = v414;	// L544
                reg1[v407] = v415;	// L545
              } else {
                int32_t v416 = opcode1;	// L547
                bool v417 = v416 == 6;	// L549
                if (v417) {	// L550
                  int32_t v418 = src1;	// L551
                  int v419 = v418;	// L552
                  int32_t v420 = reg1[v419];	// L553
                  int32_t v421 = dst1;	// L554
                  int v422 = v421;	// L555
                  int32_t v423 = reg1[v422];	// L556
                  bool v424 = v420 > v423;	// L557
                  if (v424) {	// L558
                    int32_t v425 = src1;	// L559
                    int v426 = v425;	// L560
                    int32_t v427 = reg1[v426];	// L561
                    int32_t v428 = dst1;	// L562
                    int v429 = v428;	// L563
                    reg1[v429] = v427;	// L564
                  }
                } else {
                  int32_t v430 = opcode1;	// L567
                  bool v431 = v430 == 7;	// L569
                  if (v431) {	// L570
                    int32_t v432 = dst1;	// L571
                    int v433 = v432;	// L572
                    int32_t v434 = reg1[v433];	// L573
                    int32_t v435 = imm1;	// L574
                    int32_t v436 = v434 >> v435;	// L575
                    reg1[v433] = v436;	// L576
                  } else {
                    int32_t v437 = opcode1;	// L578
                    bool v438 = v437 == 8;	// L580
                    if (v438) {	// L581
                      int32_t v439 = dst1;	// L582
                      int v440 = v439;	// L583
                      int32_t v441 = reg1[v440];	// L584
                      v341.write(v441);	// L585
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

void tiled_vpu_r2(
  int32_t v442[4],
  int v443,
  hls::stream< int32_t >& v444,
  hls::stream< int32_t >& v445,
  hls::stream< int32_t >& v446,
  hls::stream< int32_t >& v447
) {	// L598
  #pragma HLS array_partition variable=v442 complete dim=1

  int32_t prog2[12];	// L599
  for (int v449 = 0; v449 < 12; v449++) {	// L601
    prog2[v449] = 0;	// L601
  }
  l_S_pc_0_pc2: for (int pc2 = 0; pc2 < 12; pc2++) {	// L602
  #pragma HLS pipeline II=1
    int32_t v451 = v444.read();	// L603
    int32_t word2;	// L604
    word2 = v451;	// L605
    int32_t v453 = word2;	// L606
    prog2[pc2] = v453;	// L607
    int32_t v454 = word2;	// L608
    v445.write(v454);	// L609
  }
  l_S_m_1_m11: for (int m11 = 0; m11 < 6; m11++) {	// L611
    int32_t reg2[4];	// L612
    for (int v457 = 0; v457 < 4; v457++) {	// L613
      reg2[v457] = 0;	// L613
    }
    l_S_step_1_step2: for (int step2 = 0; step2 < 12; step2++) {	// L614
    #pragma HLS pipeline II=1
      int32_t v459 = prog2[step2];	// L615
      int32_t word22;	// L616
      word22 = v459;	// L617
      int32_t v461 = word22;	// L618
      int32_t v462 = v461 >> 24;	// L620
      int32_t v463 = v462 & 255;	// L622
      int32_t opcode2;	// L623
      opcode2 = v463;	// L624
      int32_t v465 = word22;	// L625
      int32_t v466 = v465 >> 20;	// L627
      int32_t v467 = v466 & 15;	// L629
      int32_t dst2;	// L630
      dst2 = v467;	// L631
      int32_t v469 = word22;	// L632
      int32_t v470 = v469 >> 16;	// L634
      int32_t v471 = v470 & 15;	// L635
      int32_t src2;	// L636
      src2 = v471;	// L637
      int32_t v473 = word22;	// L638
      int32_t v474 = v473 & 65535;	// L640
      int32_t imm2;	// L641
      imm2 = v474;	// L642
      int32_t v476 = opcode2;	// L643
      bool v477 = v476 == 9;	// L645
      if (v477) {	// L646
        int32_t v478 = v447.read();	// L647
        int32_t zz2;	// L648
        zz2 = v478;	// L649
        int32_t v480 = dst2;	// L650
        int v481 = v480;	// L651
        int32_t v482 = reg2[v481];	// L652
        int32_t v483 = zz2;	// L653
        ap_int<33> v484 = v482;	// L654
        ap_int<33> v485 = v483;	// L655
        ap_int<33> v486 = v484 + v485;	// L656
        int32_t v487 = v486;	// L657
        reg2[v481] = v487;	// L658
      } else {
        int32_t v488 = opcode2;	// L660
        bool v489 = v488 == 2;	// L662
        if (v489) {	// L663
          int32_t v490 = v442[v443];	// L664
          int32_t v491 = dst2;	// L665
          int v492 = v491;	// L666
          reg2[v492] = v490;	// L667
        } else {
          int32_t v493 = opcode2;	// L669
          bool v494 = v493 == 3;	// L671
          if (v494) {	// L672
            int32_t v495 = imm2;	// L673
            int32_t v496 = dst2;	// L674
            int v497 = v496;	// L675
            reg2[v497] = v495;	// L676
          } else {
            int32_t v498 = opcode2;	// L678
            bool v499 = v498 == 4;	// L680
            if (v499) {	// L681
              int32_t v500 = dst2;	// L682
              int v501 = v500;	// L683
              int32_t v502 = reg2[v501];	// L684
              int32_t v503 = src2;	// L685
              int v504 = v503;	// L686
              int32_t v505 = reg2[v504];	// L687
              ap_int<33> v506 = v502;	// L688
              ap_int<33> v507 = v505;	// L689
              ap_int<33> v508 = v506 + v507;	// L690
              int32_t v509 = v508;	// L691
              reg2[v501] = v509;	// L692
            } else {
              int32_t v510 = opcode2;	// L694
              bool v511 = v510 == 5;	// L696
              if (v511) {	// L697
                int32_t v512 = dst2;	// L698
                int v513 = v512;	// L699
                int32_t v514 = reg2[v513];	// L700
                int32_t v515 = src2;	// L701
                int v516 = v515;	// L702
                int32_t v517 = reg2[v516];	// L703
                int64_t v518 = v514;	// L704
                int64_t v519 = v517;	// L705
                int64_t v520 = v518 * v519;	// L706
                int32_t v521 = v520;	// L707
                reg2[v513] = v521;	// L708
              } else {
                int32_t v522 = opcode2;	// L710
                bool v523 = v522 == 6;	// L712
                if (v523) {	// L713
                  int32_t v524 = src2;	// L714
                  int v525 = v524;	// L715
                  int32_t v526 = reg2[v525];	// L716
                  int32_t v527 = dst2;	// L717
                  int v528 = v527;	// L718
                  int32_t v529 = reg2[v528];	// L719
                  bool v530 = v526 > v529;	// L720
                  if (v530) {	// L721
                    int32_t v531 = src2;	// L722
                    int v532 = v531;	// L723
                    int32_t v533 = reg2[v532];	// L724
                    int32_t v534 = dst2;	// L725
                    int v535 = v534;	// L726
                    reg2[v535] = v533;	// L727
                  }
                } else {
                  int32_t v536 = opcode2;	// L730
                  bool v537 = v536 == 7;	// L732
                  if (v537) {	// L733
                    int32_t v538 = dst2;	// L734
                    int v539 = v538;	// L735
                    int32_t v540 = reg2[v539];	// L736
                    int32_t v541 = imm2;	// L737
                    int32_t v542 = v540 >> v541;	// L738
                    reg2[v539] = v542;	// L739
                  } else {
                    int32_t v543 = opcode2;	// L741
                    bool v544 = v543 == 8;	// L743
                    if (v544) {	// L744
                      int32_t v545 = dst2;	// L745
                      int v546 = v545;	// L746
                      int32_t v547 = reg2[v546];	// L747
                      v446.write(v547);	// L748
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

void tiled_vpu_y_out_drain(
  int32_t v548[6][4],
  int v549,
  hls::stream< int32_t >& v550
) {	// L761
  #pragma HLS array_partition variable=v548 complete dim=1
  #pragma HLS array_partition variable=v548 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 6; _t2++) {	// L762
  #pragma HLS pipeline II=1
    int32_t v552 = v550.read();	// L763
    v548[_t2][v549] = v552;	// L764
  }
}

/// This is top function.
void top(
  int8_t v553[12][4],
  int32_t v554[12],
  int8_t v555[4][4][2],
  int32_t v556[4],
  int32_t v557[6][4]
) {	// L768
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v553 complete dim=1
  #pragma HLS array_partition variable=v553 complete dim=2

  #pragma HLS array_partition variable=v554 complete dim=1

  #pragma HLS array_partition variable=v555 complete dim=1
  #pragma HLS array_partition variable=v555 complete dim=2
  #pragma HLS array_partition variable=v555 complete dim=3

  #pragma HLS array_partition variable=v556 complete dim=1

  #pragma HLS array_partition variable=v557 complete dim=1
  #pragma HLS array_partition variable=v557 complete dim=2

  hls::stream< int32_t > v558;
  #pragma HLS stream variable=v558 depth=6	// L769
  hls::stream< int32_t > v559;
  #pragma HLS stream variable=v559 depth=6	// L770
  hls::stream< int32_t > v560;
  #pragma HLS stream variable=v560 depth=2	// L771
  hls::stream< int32_t > v561;
  #pragma HLS stream variable=v561 depth=6	// L772
  hls::stream< int32_t > v562;
  #pragma HLS stream variable=v562 depth=2	// L773
  hls::stream< int32_t > v563;
  #pragma HLS stream variable=v563 depth=6	// L774
  hls::stream< int32_t > v564;
  #pragma HLS stream variable=v564 depth=2	// L775
  hls::stream< int32_t > v565;
  #pragma HLS stream variable=v565 depth=2	// L776
  hls::stream< int32_t > v566;
  #pragma HLS stream variable=v566 depth=2	// L777
  hls::stream< int8_t > v567;
  #pragma HLS stream variable=v567 depth=2	// L778
  hls::stream< int32_t > v568;
  #pragma HLS stream variable=v568 depth=2	// L779
  hls::stream< int8_t > v569;
  #pragma HLS stream variable=v569 depth=2	// L780
  hls::stream< int32_t > v570;
  #pragma HLS stream variable=v570 depth=2	// L781
  hls::stream< int8_t > v571;
  #pragma HLS stream variable=v571 depth=2	// L782
  hls::stream< int32_t > v572;
  #pragma HLS stream variable=v572 depth=2	// L783
  hls::stream< int32_t > v573;
  #pragma HLS stream variable=v573 depth=2	// L784
  hls::stream< int8_t > v574;
  #pragma HLS stream variable=v574 depth=2	// L785
  hls::stream< int32_t > v575;
  #pragma HLS stream variable=v575 depth=2	// L786
  hls::stream< int8_t > v576;
  #pragma HLS stream variable=v576 depth=2	// L787
  hls::stream< int32_t > v577;
  #pragma HLS stream variable=v577 depth=2	// L788
  hls::stream< int8_t > v578;
  #pragma HLS stream variable=v578 depth=2	// L789
  hls::stream< int32_t > v579;
  #pragma HLS stream variable=v579 depth=2	// L790
  hls::stream< int32_t > v580;
  #pragma HLS stream variable=v580 depth=2	// L791
  hls::stream< int8_t > v581;
  #pragma HLS stream variable=v581 depth=2	// L792
  hls::stream< int32_t > v582;
  #pragma HLS stream variable=v582 depth=2	// L793
  hls::stream< int8_t > v583;
  #pragma HLS stream variable=v583 depth=2	// L794
  hls::stream< int32_t > v584;
  #pragma HLS stream variable=v584 depth=2	// L795
  hls::stream< int8_t > v585;
  #pragma HLS stream variable=v585 depth=2	// L796
  hls::stream< int32_t > v586;
  #pragma HLS stream variable=v586 depth=2	// L797
  hls::stream< int32_t > v587;
  #pragma HLS stream variable=v587 depth=2	// L798
  hls::stream< int8_t > v588;
  #pragma HLS stream variable=v588 depth=2	// L799
  hls::stream< int32_t > v589;
  #pragma HLS stream variable=v589 depth=2	// L800
  hls::stream< int8_t > v590;
  #pragma HLS stream variable=v590 depth=2	// L801
  hls::stream< int32_t > v591;
  #pragma HLS stream variable=v591 depth=2	// L802
  hls::stream< int8_t > v592;
  #pragma HLS stream variable=v592 depth=2	// L803
  hls::stream< int32_t > v593;
  #pragma HLS stream variable=v593 depth=12	// L804
  hls::stream< int8_t > v594;
  #pragma HLS stream variable=v594 depth=12	// L805
  hls::stream< int8_t > v595;
  #pragma HLS stream variable=v595 depth=12	// L807
  hls::stream< int8_t > v596;
  #pragma HLS stream variable=v596 depth=12	// L809
  hls::stream< int8_t > v597;
  #pragma HLS stream variable=v597 depth=12	// L811
  tiled_mac_a_in_load(v553, 0, v597);	// L813
  tiled_mac_a_in_load(v553, 1, v596);	// L814
  tiled_mac_a_in_load(v553, 2, v595);	// L815
  tiled_mac_a_in_load(v553, 3, v594);	// L816
  tiled_vpu_op_in_load(v554, 0, v593);	// L817
  tiled_mac_r8(v555, 0, 0, v597, v592, v591);	// L818
  tiled_mac_r2(v555, 0, 1, v592, v590, v589);	// L819
  tiled_mac_r2(v555, 0, 2, v590, v588, v587);	// L820
  tiled_mac_r6(v555, 0, 3, v588, v586);	// L821
  tiled_mac_r4(v555, 1, 0, v596, v585, v591, v584);	// L822
  tiled_mac_r0(v555, 1, 1, v585, v583, v589, v582);	// L823
  tiled_mac_r0(v555, 1, 2, v583, v581, v587, v580);	// L824
  tiled_mac_r3(v555, 1, 3, v581, v586, v579);	// L825
  tiled_mac_r4(v555, 2, 0, v595, v578, v584, v577);	// L826
  tiled_mac_r0(v555, 2, 1, v578, v576, v582, v575);	// L827
  tiled_mac_r0(v555, 2, 2, v576, v574, v580, v573);	// L828
  tiled_mac_r3(v555, 2, 3, v574, v579, v572);	// L829
  tiled_mac_r7(v555, 3, 0, v594, v571, v577, v570);	// L830
  tiled_mac_r1(v555, 3, 1, v571, v569, v575, v568);	// L831
  tiled_mac_r1(v555, 3, 2, v569, v567, v573, v566);	// L832
  tiled_mac_r5(v555, 3, 3, v567, v572, v565);	// L833
  tiled_vpu_r2(v556, 0, v593, v564, v563, v570);	// L834
  tiled_vpu_r0(v556, 1, v564, v562, v561, v568);	// L835
  tiled_vpu_r0(v556, 2, v562, v560, v559, v566);	// L836
  tiled_vpu_r1(v556, 3, v560, v558, v565);	// L837
  tiled_vpu_y_out_drain(v557, 0, v563);	// L838
  tiled_vpu_y_out_drain(v557, 1, v561);	// L839
  tiled_vpu_y_out_drain(v557, 2, v559);	// L840
  tiled_vpu_y_out_drain(v557, 3, v558);	// L841
}


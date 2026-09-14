
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
  int8_t v0[6][4],
  int v1,
  hls::stream< int8_t >& v2
) {	// L4
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 6; _t++) {	// L5
  #pragma HLS pipeline II=1
    int8_t v4 = v0[_t][v1];	// L6
    v2.write(v4);	// L7
  }
}

void vpu_op_in_load(
  int32_t v5[8],
  int v6,
  hls::stream< int32_t >& v7
) {	// L11
  #pragma HLS array_partition variable=v5 complete dim=1

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 8; _t1++) {	// L12
  #pragma HLS pipeline II=1
    int32_t v9 = v5[_t1];	// L13
    v7.write(v9);	// L14
  }
}

void mac_r0(
  int8_t v10[4][4],
  int v11,
  int v12,
  hls::stream< int8_t >& v13,
  hls::stream< int8_t >& v14,
  hls::stream< int32_t >& v15,
  hls::stream< int32_t >& v16
) {	// L18
  #pragma HLS array_partition variable=v10 complete dim=1
  #pragma HLS array_partition variable=v10 complete dim=2

  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L19
  #pragma HLS pipeline II=1
    int8_t v18 = v13.read();	// L20
    int8_t a;	// L21
    a = v18;	// L22
    int32_t v20 = v15.read();	// L23
    int32_t p;	// L24
    p = v20;	// L25
    int32_t v22 = p;	// L26
    int8_t v23 = a;	// L27
    int8_t v24 = v10[v11][v12];	// L28
    int16_t v25 = v23;	// L29
    int16_t v26 = v24;	// L30
    int16_t v27 = v25 * v26;	// L31
    ap_int<33> v28 = v22;	// L32
    ap_int<33> v29 = v27;	// L33
    ap_int<33> v30 = v28 + v29;	// L34
    v16.write(v30);	// L35
    int8_t v31 = a;	// L36
    v14.write(v31);	// L37
  }
}

void mac_r1(
  int8_t v32[4][4],
  int v33,
  int v34,
  hls::stream< int8_t >& v35,
  hls::stream< int8_t >& v36,
  hls::stream< int32_t >& v37,
  hls::stream< int32_t >& v38
) {	// L41
  #pragma HLS array_partition variable=v32 complete dim=1
  #pragma HLS array_partition variable=v32 complete dim=2

  l_S_m_0_m1: for (int m1 = 0; m1 < 6; m1++) {	// L42
  #pragma HLS pipeline II=1
    int8_t v40 = v35.read();	// L43
    int8_t a1;	// L44
    a1 = v40;	// L45
    int32_t v42 = v37.read();	// L46
    int32_t p1;	// L47
    p1 = v42;	// L48
    int32_t v44 = p1;	// L49
    int8_t v45 = a1;	// L50
    int8_t v46 = v32[v33][v34];	// L51
    int16_t v47 = v45;	// L52
    int16_t v48 = v46;	// L53
    int16_t v49 = v47 * v48;	// L54
    ap_int<33> v50 = v44;	// L55
    ap_int<33> v51 = v49;	// L56
    ap_int<33> v52 = v50 + v51;	// L57
    v38.write(v52);	// L58
    int8_t v53 = a1;	// L59
    v36.write(v53);	// L60
  }
}

void mac_r2(
  int8_t v54[4][4],
  int v55,
  int v56,
  hls::stream< int8_t >& v57,
  hls::stream< int8_t >& v58,
  hls::stream< int32_t >& v59
) {	// L64
  #pragma HLS array_partition variable=v54 complete dim=1
  #pragma HLS array_partition variable=v54 complete dim=2

  l_S_m_0_m2: for (int m2 = 0; m2 < 6; m2++) {	// L65
  #pragma HLS pipeline II=1
    int8_t v61 = v57.read();	// L66
    int8_t a2;	// L67
    a2 = v61;	// L68
    int32_t p2;	// L70
    p2 = 0;	// L71
    int32_t v64 = p2;	// L72
    int8_t v65 = a2;	// L73
    int8_t v66 = v54[v55][v56];	// L74
    int16_t v67 = v65;	// L75
    int16_t v68 = v66;	// L76
    int16_t v69 = v67 * v68;	// L77
    ap_int<33> v70 = v64;	// L78
    ap_int<33> v71 = v69;	// L79
    ap_int<33> v72 = v70 + v71;	// L80
    v59.write(v72);	// L81
    int8_t v73 = a2;	// L82
    v58.write(v73);	// L83
  }
}

void mac_r3(
  int8_t v74[4][4],
  int v75,
  int v76,
  hls::stream< int8_t >& v77,
  hls::stream< int32_t >& v78,
  hls::stream< int32_t >& v79
) {	// L87
  #pragma HLS array_partition variable=v74 complete dim=1
  #pragma HLS array_partition variable=v74 complete dim=2

  l_S_m_0_m3: for (int m3 = 0; m3 < 6; m3++) {	// L88
  #pragma HLS pipeline II=1
    int8_t v81 = v77.read();	// L89
    int8_t a3;	// L90
    a3 = v81;	// L91
    int32_t v83 = v78.read();	// L92
    int32_t p3;	// L93
    p3 = v83;	// L94
    int32_t v85 = p3;	// L95
    int8_t v86 = a3;	// L96
    int8_t v87 = v74[v75][v76];	// L97
    int16_t v88 = v86;	// L98
    int16_t v89 = v87;	// L99
    int16_t v90 = v88 * v89;	// L100
    ap_int<33> v91 = v85;	// L101
    ap_int<33> v92 = v90;	// L102
    ap_int<33> v93 = v91 + v92;	// L103
    v79.write(v93);	// L104
  }
}

void mac_r4(
  int8_t v94[4][4],
  int v95,
  int v96,
  hls::stream< int8_t >& v97,
  hls::stream< int8_t >& v98,
  hls::stream< int32_t >& v99,
  hls::stream< int32_t >& v100
) {	// L108
  #pragma HLS array_partition variable=v94 complete dim=1
  #pragma HLS array_partition variable=v94 complete dim=2

  l_S_m_0_m4: for (int m4 = 0; m4 < 6; m4++) {	// L109
  #pragma HLS pipeline II=1
    int8_t v102 = v97.read();	// L110
    int8_t a4;	// L111
    a4 = v102;	// L112
    int32_t v104 = v99.read();	// L113
    int32_t p4;	// L114
    p4 = v104;	// L115
    int32_t v106 = p4;	// L116
    int8_t v107 = a4;	// L117
    int8_t v108 = v94[v95][v96];	// L118
    int16_t v109 = v107;	// L119
    int16_t v110 = v108;	// L120
    int16_t v111 = v109 * v110;	// L121
    ap_int<33> v112 = v106;	// L122
    ap_int<33> v113 = v111;	// L123
    ap_int<33> v114 = v112 + v113;	// L124
    v100.write(v114);	// L125
    int8_t v115 = a4;	// L126
    v98.write(v115);	// L127
  }
}

void mac_r5(
  int8_t v116[4][4],
  int v117,
  int v118,
  hls::stream< int8_t >& v119,
  hls::stream< int32_t >& v120,
  hls::stream< int32_t >& v121
) {	// L131
  #pragma HLS array_partition variable=v116 complete dim=1
  #pragma HLS array_partition variable=v116 complete dim=2

  l_S_m_0_m5: for (int m5 = 0; m5 < 6; m5++) {	// L132
  #pragma HLS pipeline II=1
    int8_t v123 = v119.read();	// L133
    int8_t a5;	// L134
    a5 = v123;	// L135
    int32_t v125 = v120.read();	// L136
    int32_t p5;	// L137
    p5 = v125;	// L138
    int32_t v127 = p5;	// L139
    int8_t v128 = a5;	// L140
    int8_t v129 = v116[v117][v118];	// L141
    int16_t v130 = v128;	// L142
    int16_t v131 = v129;	// L143
    int16_t v132 = v130 * v131;	// L144
    ap_int<33> v133 = v127;	// L145
    ap_int<33> v134 = v132;	// L146
    ap_int<33> v135 = v133 + v134;	// L147
    v121.write(v135);	// L148
  }
}

void mac_r6(
  int8_t v136[4][4],
  int v137,
  int v138,
  hls::stream< int8_t >& v139,
  hls::stream< int32_t >& v140
) {	// L152
  #pragma HLS array_partition variable=v136 complete dim=1
  #pragma HLS array_partition variable=v136 complete dim=2

  l_S_m_0_m6: for (int m6 = 0; m6 < 6; m6++) {	// L153
  #pragma HLS pipeline II=1
    int8_t v142 = v139.read();	// L154
    int8_t a6;	// L155
    a6 = v142;	// L156
    int32_t p6;	// L158
    p6 = 0;	// L159
    int32_t v145 = p6;	// L160
    int8_t v146 = a6;	// L161
    int8_t v147 = v136[v137][v138];	// L162
    int16_t v148 = v146;	// L163
    int16_t v149 = v147;	// L164
    int16_t v150 = v148 * v149;	// L165
    ap_int<33> v151 = v145;	// L166
    ap_int<33> v152 = v150;	// L167
    ap_int<33> v153 = v151 + v152;	// L168
    v140.write(v153);	// L169
  }
}

void mac_r7(
  int8_t v154[4][4],
  int v155,
  int v156,
  hls::stream< int8_t >& v157,
  hls::stream< int8_t >& v158,
  hls::stream< int32_t >& v159,
  hls::stream< int32_t >& v160
) {	// L173
  #pragma HLS array_partition variable=v154 complete dim=1
  #pragma HLS array_partition variable=v154 complete dim=2

  l_S_m_0_m7: for (int m7 = 0; m7 < 6; m7++) {	// L174
  #pragma HLS pipeline II=1
    int8_t v162 = v157.read();	// L175
    int8_t a7;	// L176
    a7 = v162;	// L177
    int32_t v164 = v159.read();	// L178
    int32_t p7;	// L179
    p7 = v164;	// L180
    int32_t v166 = p7;	// L181
    int8_t v167 = a7;	// L182
    int8_t v168 = v154[v155][v156];	// L183
    int16_t v169 = v167;	// L184
    int16_t v170 = v168;	// L185
    int16_t v171 = v169 * v170;	// L186
    ap_int<33> v172 = v166;	// L187
    ap_int<33> v173 = v171;	// L188
    ap_int<33> v174 = v172 + v173;	// L189
    v160.write(v174);	// L190
    int8_t v175 = a7;	// L191
    v158.write(v175);	// L192
  }
}

void mac_r8(
  int8_t v176[4][4],
  int v177,
  int v178,
  hls::stream< int8_t >& v179,
  hls::stream< int8_t >& v180,
  hls::stream< int32_t >& v181
) {	// L196
  #pragma HLS array_partition variable=v176 complete dim=1
  #pragma HLS array_partition variable=v176 complete dim=2

  l_S_m_0_m8: for (int m8 = 0; m8 < 6; m8++) {	// L197
  #pragma HLS pipeline II=1
    int8_t v183 = v179.read();	// L198
    int8_t a8;	// L199
    a8 = v183;	// L200
    int32_t p8;	// L202
    p8 = 0;	// L203
    int32_t v186 = p8;	// L204
    int8_t v187 = a8;	// L205
    int8_t v188 = v176[v177][v178];	// L206
    int16_t v189 = v187;	// L207
    int16_t v190 = v188;	// L208
    int16_t v191 = v189 * v190;	// L209
    ap_int<33> v192 = v186;	// L210
    ap_int<33> v193 = v191;	// L211
    ap_int<33> v194 = v192 + v193;	// L212
    v181.write(v194);	// L213
    int8_t v195 = a8;	// L214
    v180.write(v195);	// L215
  }
}

void vpu_r0(
  int32_t v196[4],
  int v197,
  hls::stream< int32_t >& v198,
  hls::stream< int32_t >& v199,
  hls::stream< int32_t >& v200,
  hls::stream< int32_t >& v201
) {	// L219
  #pragma HLS array_partition variable=v196 complete dim=1

  int32_t prog[8];	// L220
  for (int v203 = 0; v203 < 8; v203++) {	// L222
    prog[v203] = 0;	// L222
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 8; pc++) {	// L223
  #pragma HLS pipeline II=1
    int32_t v205 = v198.read();	// L224
    int32_t word;	// L225
    word = v205;	// L226
    int32_t v207 = word;	// L227
    prog[pc] = v207;	// L228
    int32_t v208 = word;	// L229
    v199.write(v208);	// L230
  }
  l_S_m_1_m9: for (int m9 = 0; m9 < 6; m9++) {	// L232
    int32_t v210 = v201.read();	// L233
    int32_t z;	// L234
    z = v210;	// L235
    int32_t reg[4];	// L236
    for (int v213 = 0; v213 < 4; v213++) {	// L237
      reg[v213] = 0;	// L237
    }
    l_S_step_1_step: for (int step = 0; step < 8; step++) {	// L238
    #pragma HLS pipeline II=1
      int32_t v215 = prog[step];	// L239
      int32_t word2;	// L240
      word2 = v215;	// L241
      int32_t v217 = word2;	// L242
      int32_t v218 = v217 >> 24;	// L244
      int32_t v219 = v218 & 255;	// L246
      int32_t opcode;	// L247
      opcode = v219;	// L248
      int32_t v221 = word2;	// L249
      int32_t v222 = v221 >> 20;	// L251
      int32_t v223 = v222 & 15;	// L253
      int32_t dst;	// L254
      dst = v223;	// L255
      int32_t v225 = word2;	// L256
      int32_t v226 = v225 >> 16;	// L258
      int32_t v227 = v226 & 15;	// L259
      int32_t src;	// L260
      src = v227;	// L261
      int32_t v229 = word2;	// L262
      int32_t v230 = v229 & 65535;	// L264
      int32_t imm;	// L265
      imm = v230;	// L266
      int32_t v232 = opcode;	// L267
      bool v233 = v232 == 1;	// L269
      if (v233) {	// L270
        int32_t v234 = z;	// L271
        int32_t v235 = dst;	// L272
        int v236 = v235;	// L273
        reg[v236] = v234;	// L274
      } else {
        int32_t v237 = opcode;	// L276
        bool v238 = v237 == 2;	// L278
        if (v238) {	// L279
          int32_t v239 = v196[v197];	// L280
          int32_t v240 = dst;	// L281
          int v241 = v240;	// L282
          reg[v241] = v239;	// L283
        } else {
          int32_t v242 = opcode;	// L285
          bool v243 = v242 == 3;	// L287
          if (v243) {	// L288
            int32_t v244 = imm;	// L289
            int32_t v245 = dst;	// L290
            int v246 = v245;	// L291
            reg[v246] = v244;	// L292
          } else {
            int32_t v247 = opcode;	// L294
            bool v248 = v247 == 4;	// L296
            if (v248) {	// L297
              int32_t v249 = dst;	// L298
              int v250 = v249;	// L299
              int32_t v251 = reg[v250];	// L300
              int32_t v252 = src;	// L301
              int v253 = v252;	// L302
              int32_t v254 = reg[v253];	// L303
              ap_int<33> v255 = v251;	// L304
              ap_int<33> v256 = v254;	// L305
              ap_int<33> v257 = v255 + v256;	// L306
              int32_t v258 = v257;	// L307
              reg[v250] = v258;	// L308
            } else {
              int32_t v259 = opcode;	// L310
              bool v260 = v259 == 5;	// L312
              if (v260) {	// L313
                int32_t v261 = dst;	// L314
                int v262 = v261;	// L315
                int32_t v263 = reg[v262];	// L316
                int32_t v264 = src;	// L317
                int v265 = v264;	// L318
                int32_t v266 = reg[v265];	// L319
                int64_t v267 = v263;	// L320
                int64_t v268 = v266;	// L321
                int64_t v269 = v267 * v268;	// L322
                int32_t v270 = v269;	// L323
                reg[v262] = v270;	// L324
              } else {
                int32_t v271 = opcode;	// L326
                bool v272 = v271 == 6;	// L328
                if (v272) {	// L329
                  int32_t v273 = src;	// L330
                  int v274 = v273;	// L331
                  int32_t v275 = reg[v274];	// L332
                  int32_t v276 = dst;	// L333
                  int v277 = v276;	// L334
                  int32_t v278 = reg[v277];	// L335
                  bool v279 = v275 > v278;	// L336
                  if (v279) {	// L337
                    int32_t v280 = src;	// L338
                    int v281 = v280;	// L339
                    int32_t v282 = reg[v281];	// L340
                    int32_t v283 = dst;	// L341
                    int v284 = v283;	// L342
                    reg[v284] = v282;	// L343
                  }
                } else {
                  int32_t v285 = opcode;	// L346
                  bool v286 = v285 == 7;	// L348
                  if (v286) {	// L349
                    int32_t v287 = dst;	// L350
                    int v288 = v287;	// L351
                    int32_t v289 = reg[v288];	// L352
                    int32_t v290 = imm;	// L353
                    int32_t v291 = v289 >> v290;	// L354
                    reg[v288] = v291;	// L355
                  } else {
                    int32_t v292 = opcode;	// L357
                    bool v293 = v292 == 8;	// L359
                    if (v293) {	// L360
                      int32_t v294 = dst;	// L361
                      int v295 = v294;	// L362
                      int32_t v296 = reg[v295];	// L363
                      v200.write(v296);	// L364
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

void vpu_r1(
  int32_t v297[4],
  int v298,
  hls::stream< int32_t >& v299,
  hls::stream< int32_t >& v300,
  hls::stream< int32_t >& v301
) {	// L377
  #pragma HLS array_partition variable=v297 complete dim=1

  int32_t prog1[8];	// L378
  for (int v303 = 0; v303 < 8; v303++) {	// L380
    prog1[v303] = 0;	// L380
  }
  l_S_pc_0_pc1: for (int pc1 = 0; pc1 < 8; pc1++) {	// L381
  #pragma HLS pipeline II=1
    int32_t v305 = v299.read();	// L382
    int32_t word1;	// L383
    word1 = v305;	// L384
    int32_t v307 = word1;	// L385
    prog1[pc1] = v307;	// L386
  }
  l_S_m_1_m10: for (int m10 = 0; m10 < 6; m10++) {	// L388
    int32_t v309 = v301.read();	// L389
    int32_t z1;	// L390
    z1 = v309;	// L391
    int32_t reg1[4];	// L392
    for (int v312 = 0; v312 < 4; v312++) {	// L393
      reg1[v312] = 0;	// L393
    }
    l_S_step_1_step1: for (int step1 = 0; step1 < 8; step1++) {	// L394
    #pragma HLS pipeline II=1
      int32_t v314 = prog1[step1];	// L395
      int32_t word21;	// L396
      word21 = v314;	// L397
      int32_t v316 = word21;	// L398
      int32_t v317 = v316 >> 24;	// L400
      int32_t v318 = v317 & 255;	// L402
      int32_t opcode1;	// L403
      opcode1 = v318;	// L404
      int32_t v320 = word21;	// L405
      int32_t v321 = v320 >> 20;	// L407
      int32_t v322 = v321 & 15;	// L409
      int32_t dst1;	// L410
      dst1 = v322;	// L411
      int32_t v324 = word21;	// L412
      int32_t v325 = v324 >> 16;	// L414
      int32_t v326 = v325 & 15;	// L415
      int32_t src1;	// L416
      src1 = v326;	// L417
      int32_t v328 = word21;	// L418
      int32_t v329 = v328 & 65535;	// L420
      int32_t imm1;	// L421
      imm1 = v329;	// L422
      int32_t v331 = opcode1;	// L423
      bool v332 = v331 == 1;	// L425
      if (v332) {	// L426
        int32_t v333 = z1;	// L427
        int32_t v334 = dst1;	// L428
        int v335 = v334;	// L429
        reg1[v335] = v333;	// L430
      } else {
        int32_t v336 = opcode1;	// L432
        bool v337 = v336 == 2;	// L434
        if (v337) {	// L435
          int32_t v338 = v297[v298];	// L436
          int32_t v339 = dst1;	// L437
          int v340 = v339;	// L438
          reg1[v340] = v338;	// L439
        } else {
          int32_t v341 = opcode1;	// L441
          bool v342 = v341 == 3;	// L443
          if (v342) {	// L444
            int32_t v343 = imm1;	// L445
            int32_t v344 = dst1;	// L446
            int v345 = v344;	// L447
            reg1[v345] = v343;	// L448
          } else {
            int32_t v346 = opcode1;	// L450
            bool v347 = v346 == 4;	// L452
            if (v347) {	// L453
              int32_t v348 = dst1;	// L454
              int v349 = v348;	// L455
              int32_t v350 = reg1[v349];	// L456
              int32_t v351 = src1;	// L457
              int v352 = v351;	// L458
              int32_t v353 = reg1[v352];	// L459
              ap_int<33> v354 = v350;	// L460
              ap_int<33> v355 = v353;	// L461
              ap_int<33> v356 = v354 + v355;	// L462
              int32_t v357 = v356;	// L463
              reg1[v349] = v357;	// L464
            } else {
              int32_t v358 = opcode1;	// L466
              bool v359 = v358 == 5;	// L468
              if (v359) {	// L469
                int32_t v360 = dst1;	// L470
                int v361 = v360;	// L471
                int32_t v362 = reg1[v361];	// L472
                int32_t v363 = src1;	// L473
                int v364 = v363;	// L474
                int32_t v365 = reg1[v364];	// L475
                int64_t v366 = v362;	// L476
                int64_t v367 = v365;	// L477
                int64_t v368 = v366 * v367;	// L478
                int32_t v369 = v368;	// L479
                reg1[v361] = v369;	// L480
              } else {
                int32_t v370 = opcode1;	// L482
                bool v371 = v370 == 6;	// L484
                if (v371) {	// L485
                  int32_t v372 = src1;	// L486
                  int v373 = v372;	// L487
                  int32_t v374 = reg1[v373];	// L488
                  int32_t v375 = dst1;	// L489
                  int v376 = v375;	// L490
                  int32_t v377 = reg1[v376];	// L491
                  bool v378 = v374 > v377;	// L492
                  if (v378) {	// L493
                    int32_t v379 = src1;	// L494
                    int v380 = v379;	// L495
                    int32_t v381 = reg1[v380];	// L496
                    int32_t v382 = dst1;	// L497
                    int v383 = v382;	// L498
                    reg1[v383] = v381;	// L499
                  }
                } else {
                  int32_t v384 = opcode1;	// L502
                  bool v385 = v384 == 7;	// L504
                  if (v385) {	// L505
                    int32_t v386 = dst1;	// L506
                    int v387 = v386;	// L507
                    int32_t v388 = reg1[v387];	// L508
                    int32_t v389 = imm1;	// L509
                    int32_t v390 = v388 >> v389;	// L510
                    reg1[v387] = v390;	// L511
                  } else {
                    int32_t v391 = opcode1;	// L513
                    bool v392 = v391 == 8;	// L515
                    if (v392) {	// L516
                      int32_t v393 = dst1;	// L517
                      int v394 = v393;	// L518
                      int32_t v395 = reg1[v394];	// L519
                      v300.write(v395);	// L520
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

void vpu_r2(
  int32_t v396[4],
  int v397,
  hls::stream< int32_t >& v398,
  hls::stream< int32_t >& v399,
  hls::stream< int32_t >& v400,
  hls::stream< int32_t >& v401
) {	// L533
  #pragma HLS array_partition variable=v396 complete dim=1

  int32_t prog2[8];	// L534
  for (int v403 = 0; v403 < 8; v403++) {	// L536
    prog2[v403] = 0;	// L536
  }
  l_S_pc_0_pc2: for (int pc2 = 0; pc2 < 8; pc2++) {	// L537
  #pragma HLS pipeline II=1
    int32_t v405 = v398.read();	// L538
    int32_t word2;	// L539
    word2 = v405;	// L540
    int32_t v407 = word2;	// L541
    prog2[pc2] = v407;	// L542
    int32_t v408 = word2;	// L543
    v399.write(v408);	// L544
  }
  l_S_m_1_m11: for (int m11 = 0; m11 < 6; m11++) {	// L546
    int32_t v410 = v401.read();	// L547
    int32_t z2;	// L548
    z2 = v410;	// L549
    int32_t reg2[4];	// L550
    for (int v413 = 0; v413 < 4; v413++) {	// L551
      reg2[v413] = 0;	// L551
    }
    l_S_step_1_step2: for (int step2 = 0; step2 < 8; step2++) {	// L552
    #pragma HLS pipeline II=1
      int32_t v415 = prog2[step2];	// L553
      int32_t word22;	// L554
      word22 = v415;	// L555
      int32_t v417 = word22;	// L556
      int32_t v418 = v417 >> 24;	// L558
      int32_t v419 = v418 & 255;	// L560
      int32_t opcode2;	// L561
      opcode2 = v419;	// L562
      int32_t v421 = word22;	// L563
      int32_t v422 = v421 >> 20;	// L565
      int32_t v423 = v422 & 15;	// L567
      int32_t dst2;	// L568
      dst2 = v423;	// L569
      int32_t v425 = word22;	// L570
      int32_t v426 = v425 >> 16;	// L572
      int32_t v427 = v426 & 15;	// L573
      int32_t src2;	// L574
      src2 = v427;	// L575
      int32_t v429 = word22;	// L576
      int32_t v430 = v429 & 65535;	// L578
      int32_t imm2;	// L579
      imm2 = v430;	// L580
      int32_t v432 = opcode2;	// L581
      bool v433 = v432 == 1;	// L583
      if (v433) {	// L584
        int32_t v434 = z2;	// L585
        int32_t v435 = dst2;	// L586
        int v436 = v435;	// L587
        reg2[v436] = v434;	// L588
      } else {
        int32_t v437 = opcode2;	// L590
        bool v438 = v437 == 2;	// L592
        if (v438) {	// L593
          int32_t v439 = v396[v397];	// L594
          int32_t v440 = dst2;	// L595
          int v441 = v440;	// L596
          reg2[v441] = v439;	// L597
        } else {
          int32_t v442 = opcode2;	// L599
          bool v443 = v442 == 3;	// L601
          if (v443) {	// L602
            int32_t v444 = imm2;	// L603
            int32_t v445 = dst2;	// L604
            int v446 = v445;	// L605
            reg2[v446] = v444;	// L606
          } else {
            int32_t v447 = opcode2;	// L608
            bool v448 = v447 == 4;	// L610
            if (v448) {	// L611
              int32_t v449 = dst2;	// L612
              int v450 = v449;	// L613
              int32_t v451 = reg2[v450];	// L614
              int32_t v452 = src2;	// L615
              int v453 = v452;	// L616
              int32_t v454 = reg2[v453];	// L617
              ap_int<33> v455 = v451;	// L618
              ap_int<33> v456 = v454;	// L619
              ap_int<33> v457 = v455 + v456;	// L620
              int32_t v458 = v457;	// L621
              reg2[v450] = v458;	// L622
            } else {
              int32_t v459 = opcode2;	// L624
              bool v460 = v459 == 5;	// L626
              if (v460) {	// L627
                int32_t v461 = dst2;	// L628
                int v462 = v461;	// L629
                int32_t v463 = reg2[v462];	// L630
                int32_t v464 = src2;	// L631
                int v465 = v464;	// L632
                int32_t v466 = reg2[v465];	// L633
                int64_t v467 = v463;	// L634
                int64_t v468 = v466;	// L635
                int64_t v469 = v467 * v468;	// L636
                int32_t v470 = v469;	// L637
                reg2[v462] = v470;	// L638
              } else {
                int32_t v471 = opcode2;	// L640
                bool v472 = v471 == 6;	// L642
                if (v472) {	// L643
                  int32_t v473 = src2;	// L644
                  int v474 = v473;	// L645
                  int32_t v475 = reg2[v474];	// L646
                  int32_t v476 = dst2;	// L647
                  int v477 = v476;	// L648
                  int32_t v478 = reg2[v477];	// L649
                  bool v479 = v475 > v478;	// L650
                  if (v479) {	// L651
                    int32_t v480 = src2;	// L652
                    int v481 = v480;	// L653
                    int32_t v482 = reg2[v481];	// L654
                    int32_t v483 = dst2;	// L655
                    int v484 = v483;	// L656
                    reg2[v484] = v482;	// L657
                  }
                } else {
                  int32_t v485 = opcode2;	// L660
                  bool v486 = v485 == 7;	// L662
                  if (v486) {	// L663
                    int32_t v487 = dst2;	// L664
                    int v488 = v487;	// L665
                    int32_t v489 = reg2[v488];	// L666
                    int32_t v490 = imm2;	// L667
                    int32_t v491 = v489 >> v490;	// L668
                    reg2[v488] = v491;	// L669
                  } else {
                    int32_t v492 = opcode2;	// L671
                    bool v493 = v492 == 8;	// L673
                    if (v493) {	// L674
                      int32_t v494 = dst2;	// L675
                      int v495 = v494;	// L676
                      int32_t v496 = reg2[v495];	// L677
                      v400.write(v496);	// L678
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

void vpu_y_out_drain(
  int32_t v497[6][4],
  int v498,
  hls::stream< int32_t >& v499
) {	// L691
  #pragma HLS array_partition variable=v497 complete dim=1
  #pragma HLS array_partition variable=v497 complete dim=2

  l_S__t_0__t2: for (int _t2 = 0; _t2 < 6; _t2++) {	// L692
  #pragma HLS pipeline II=1
    int32_t v501 = v499.read();	// L693
    v497[_t2][v498] = v501;	// L694
  }
}

/// This is top function.
void top(
  int8_t v502[6][4],
  int32_t v503[8],
  int8_t v504[4][4],
  int32_t v505[4],
  int32_t v506[6][4]
) {	// L698
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v502 complete dim=1
  #pragma HLS array_partition variable=v502 complete dim=2

  #pragma HLS array_partition variable=v503 complete dim=1

  #pragma HLS array_partition variable=v504 complete dim=1
  #pragma HLS array_partition variable=v504 complete dim=2

  #pragma HLS array_partition variable=v505 complete dim=1

  #pragma HLS array_partition variable=v506 complete dim=1
  #pragma HLS array_partition variable=v506 complete dim=2

  hls::stream< int32_t > v507;
  #pragma HLS stream variable=v507 depth=6	// L699
  hls::stream< int32_t > v508;
  #pragma HLS stream variable=v508 depth=6	// L700
  hls::stream< int32_t > v509;
  #pragma HLS stream variable=v509 depth=2	// L701
  hls::stream< int32_t > v510;
  #pragma HLS stream variable=v510 depth=6	// L702
  hls::stream< int32_t > v511;
  #pragma HLS stream variable=v511 depth=2	// L703
  hls::stream< int32_t > v512;
  #pragma HLS stream variable=v512 depth=6	// L704
  hls::stream< int32_t > v513;
  #pragma HLS stream variable=v513 depth=2	// L705
  hls::stream< int32_t > v514;
  #pragma HLS stream variable=v514 depth=2	// L706
  hls::stream< int32_t > v515;
  #pragma HLS stream variable=v515 depth=2	// L707
  hls::stream< int8_t > v516;
  #pragma HLS stream variable=v516 depth=2	// L708
  hls::stream< int32_t > v517;
  #pragma HLS stream variable=v517 depth=2	// L709
  hls::stream< int8_t > v518;
  #pragma HLS stream variable=v518 depth=2	// L710
  hls::stream< int32_t > v519;
  #pragma HLS stream variable=v519 depth=2	// L711
  hls::stream< int8_t > v520;
  #pragma HLS stream variable=v520 depth=2	// L712
  hls::stream< int32_t > v521;
  #pragma HLS stream variable=v521 depth=2	// L713
  hls::stream< int32_t > v522;
  #pragma HLS stream variable=v522 depth=2	// L714
  hls::stream< int8_t > v523;
  #pragma HLS stream variable=v523 depth=2	// L715
  hls::stream< int32_t > v524;
  #pragma HLS stream variable=v524 depth=2	// L716
  hls::stream< int8_t > v525;
  #pragma HLS stream variable=v525 depth=2	// L717
  hls::stream< int32_t > v526;
  #pragma HLS stream variable=v526 depth=2	// L718
  hls::stream< int8_t > v527;
  #pragma HLS stream variable=v527 depth=2	// L719
  hls::stream< int32_t > v528;
  #pragma HLS stream variable=v528 depth=2	// L720
  hls::stream< int32_t > v529;
  #pragma HLS stream variable=v529 depth=2	// L721
  hls::stream< int8_t > v530;
  #pragma HLS stream variable=v530 depth=2	// L722
  hls::stream< int32_t > v531;
  #pragma HLS stream variable=v531 depth=2	// L723
  hls::stream< int8_t > v532;
  #pragma HLS stream variable=v532 depth=2	// L724
  hls::stream< int32_t > v533;
  #pragma HLS stream variable=v533 depth=2	// L725
  hls::stream< int8_t > v534;
  #pragma HLS stream variable=v534 depth=2	// L726
  hls::stream< int32_t > v535;
  #pragma HLS stream variable=v535 depth=2	// L727
  hls::stream< int32_t > v536;
  #pragma HLS stream variable=v536 depth=2	// L728
  hls::stream< int8_t > v537;
  #pragma HLS stream variable=v537 depth=2	// L729
  hls::stream< int32_t > v538;
  #pragma HLS stream variable=v538 depth=2	// L730
  hls::stream< int8_t > v539;
  #pragma HLS stream variable=v539 depth=2	// L731
  hls::stream< int32_t > v540;
  #pragma HLS stream variable=v540 depth=2	// L732
  hls::stream< int8_t > v541;
  #pragma HLS stream variable=v541 depth=2	// L733
  hls::stream< int32_t > v542;
  #pragma HLS stream variable=v542 depth=8	// L734
  hls::stream< int8_t > v543;
  #pragma HLS stream variable=v543 depth=6	// L735
  hls::stream< int8_t > v544;
  #pragma HLS stream variable=v544 depth=6	// L737
  hls::stream< int8_t > v545;
  #pragma HLS stream variable=v545 depth=6	// L739
  hls::stream< int8_t > v546;
  #pragma HLS stream variable=v546 depth=6	// L741
  mac_a_in_load(v502, 0, v546);	// L743
  mac_a_in_load(v502, 1, v545);	// L744
  mac_a_in_load(v502, 2, v544);	// L745
  mac_a_in_load(v502, 3, v543);	// L746
  vpu_op_in_load(v503, 0, v542);	// L747
  mac_r8(v504, 0, 0, v546, v541, v540);	// L748
  mac_r2(v504, 0, 1, v541, v539, v538);	// L749
  mac_r2(v504, 0, 2, v539, v537, v536);	// L750
  mac_r6(v504, 0, 3, v537, v535);	// L751
  mac_r4(v504, 1, 0, v545, v534, v540, v533);	// L752
  mac_r0(v504, 1, 1, v534, v532, v538, v531);	// L753
  mac_r0(v504, 1, 2, v532, v530, v536, v529);	// L754
  mac_r3(v504, 1, 3, v530, v535, v528);	// L755
  mac_r4(v504, 2, 0, v544, v527, v533, v526);	// L756
  mac_r0(v504, 2, 1, v527, v525, v531, v524);	// L757
  mac_r0(v504, 2, 2, v525, v523, v529, v522);	// L758
  mac_r3(v504, 2, 3, v523, v528, v521);	// L759
  mac_r7(v504, 3, 0, v543, v520, v526, v519);	// L760
  mac_r1(v504, 3, 1, v520, v518, v524, v517);	// L761
  mac_r1(v504, 3, 2, v518, v516, v522, v515);	// L762
  mac_r5(v504, 3, 3, v516, v521, v514);	// L763
  vpu_r2(v505, 0, v542, v513, v512, v519);	// L764
  vpu_r0(v505, 1, v513, v511, v510, v517);	// L765
  vpu_r0(v505, 2, v511, v509, v508, v515);	// L766
  vpu_r1(v505, 3, v509, v507, v514);	// L767
  vpu_y_out_drain(v506, 0, v512);	// L768
  vpu_y_out_drain(v506, 1, v510);	// L769
  vpu_y_out_drain(v506, 2, v508);	// L770
  vpu_y_out_drain(v506, 3, v507);	// L771
}


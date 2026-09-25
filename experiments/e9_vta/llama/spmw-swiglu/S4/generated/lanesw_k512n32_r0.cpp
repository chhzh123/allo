
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void lanesw_k512n32_r0_0(
  hls::stream< hls::vector< int32_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t v3[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v3[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v4 = v3[0];	// L4
  int32_t v5 = v4 & 31;	// L6
  int32_t sh;	// L7
  sh = v5;	// L8
  int32_t v7 = v3[1];	// L9
  int32_t v8 = v7 & 31;	// L10
  int32_t sh2;	// L11
  sh2 = v8;	// L12
  int32_t r0;	// L14
  r0 = 0;	// L15
  int32_t r1;	// L16
  r1 = 0;	// L17
  int32_t r2;	// L18
  r2 = 0;	// L19
  int32_t r3;	// L20
  r3 = 0;	// L21
  int32_t r4;	// L22
  r4 = 0;	// L23
  int32_t r5;	// L24
  r5 = 0;	// L25
  int32_t r6;	// L26
  r6 = 0;	// L27
  int32_t r7;	// L28
  r7 = 0;	// L29
  int32_t r8;	// L30
  r8 = 0;	// L31
  int32_t r9;	// L32
  r9 = 0;	// L33
  int32_t r10;	// L34
  r10 = 0;	// L35
  int32_t r11;	// L36
  r11 = 0;	// L37
  int32_t r12;	// L38
  r12 = 0;	// L39
  int32_t r13;	// L40
  r13 = 0;	// L41
  int32_t r14;	// L42
  r14 = 0;	// L43
  int32_t r15;	// L44
  r15 = 0;	// L45
  int32_t r16;	// L46
  r16 = 0;	// L47
  int32_t r17;	// L48
  r17 = 0;	// L49
  int32_t r18;	// L50
  r18 = 0;	// L51
  int32_t r19;	// L52
  r19 = 0;	// L53
  int32_t r20;	// L54
  r20 = 0;	// L55
  int32_t r21;	// L56
  r21 = 0;	// L57
  int32_t r22;	// L58
  r22 = 0;	// L59
  int32_t r23;	// L60
  r23 = 0;	// L61
  int32_t r24;	// L62
  r24 = 0;	// L63
  int32_t r25;	// L64
  r25 = 0;	// L65
  int32_t r26;	// L66
  r26 = 0;	// L67
  int32_t r27;	// L68
  r27 = 0;	// L69
  int32_t r28;	// L70
  r28 = 0;	// L71
  int32_t r29;	// L72
  r29 = 0;	// L73
  int32_t r30;	// L74
  r30 = 0;	// L75
  int32_t r31;	// L76
  r31 = 0;	// L77
  int32_t r32;	// L78
  r32 = 0;	// L79
  int32_t r33;	// L80
  r33 = 0;	// L81
  int32_t r34;	// L82
  r34 = 0;	// L83
  int32_t r35;	// L84
  r35 = 0;	// L85
  int32_t r36;	// L86
  r36 = 0;	// L87
  int32_t r37;	// L88
  r37 = 0;	// L89
  int32_t r38;	// L90
  r38 = 0;	// L91
  int32_t r39;	// L92
  r39 = 0;	// L93
  int32_t r40;	// L94
  r40 = 0;	// L95
  int32_t r41;	// L96
  r41 = 0;	// L97
  int32_t r42;	// L98
  r42 = 0;	// L99
  int32_t r43;	// L100
  r43 = 0;	// L101
  int32_t r44;	// L102
  r44 = 0;	// L103
  int32_t r45;	// L104
  r45 = 0;	// L105
  int32_t r46;	// L106
  r46 = 0;	// L107
  int32_t r47;	// L108
  r47 = 0;	// L109
  int32_t r48;	// L110
  r48 = 0;	// L111
  int32_t r49;	// L112
  r49 = 0;	// L113
  int32_t r50;	// L114
  r50 = 0;	// L115
  int32_t r51;	// L116
  r51 = 0;	// L117
  int32_t r52;	// L118
  r52 = 0;	// L119
  int32_t r53;	// L120
  r53 = 0;	// L121
  int32_t r54;	// L122
  r54 = 0;	// L123
  int32_t r55;	// L124
  r55 = 0;	// L125
  int32_t r56;	// L126
  r56 = 0;	// L127
  int32_t r57;	// L128
  r57 = 0;	// L129
  int32_t r58;	// L130
  r58 = 0;	// L131
  int32_t r59;	// L132
  r59 = 0;	// L133
  int32_t r60;	// L134
  r60 = 0;	// L135
  int32_t r61;	// L136
  r61 = 0;	// L137
  int32_t r62;	// L138
  r62 = 0;	// L139
  int32_t r63;	// L140
  r63 = 0;	// L141
  int32_t r64;	// L142
  r64 = 0;	// L143
  int32_t r65;	// L144
  r65 = 0;	// L145
  int32_t r66;	// L146
  r66 = 0;	// L147
  int32_t r67;	// L148
  r67 = 0;	// L149
  int32_t r68;	// L150
  r68 = 0;	// L151
  int32_t r69;	// L152
  r69 = 0;	// L153
  int32_t r70;	// L154
  r70 = 0;	// L155
  int32_t r71;	// L156
  r71 = 0;	// L157
  int32_t r72;	// L158
  r72 = 0;	// L159
  int32_t r73;	// L160
  r73 = 0;	// L161
  int32_t r74;	// L162
  r74 = 0;	// L163
  int32_t r75;	// L164
  r75 = 0;	// L165
  int32_t r76;	// L166
  r76 = 0;	// L167
  int32_t r77;	// L168
  r77 = 0;	// L169
  int32_t r78;	// L170
  r78 = 0;	// L171
  int32_t r79;	// L172
  r79 = 0;	// L173
  int32_t r80;	// L174
  r80 = 0;	// L175
  int32_t r81;	// L176
  r81 = 0;	// L177
  int32_t r82;	// L178
  r82 = 0;	// L179
  int32_t r83;	// L180
  r83 = 0;	// L181
  int32_t r84;	// L182
  r84 = 0;	// L183
  int32_t r85;	// L184
  r85 = 0;	// L185
  int32_t r86;	// L186
  r86 = 0;	// L187
  int32_t r87;	// L188
  r87 = 0;	// L189
  int32_t r88;	// L190
  r88 = 0;	// L191
  int32_t r89;	// L192
  r89 = 0;	// L193
  int32_t r90;	// L194
  r90 = 0;	// L195
  int32_t r91;	// L196
  r91 = 0;	// L197
  int32_t r92;	// L198
  r92 = 0;	// L199
  int32_t r93;	// L200
  r93 = 0;	// L201
  int32_t r94;	// L202
  r94 = 0;	// L203
  int32_t r95;	// L204
  r95 = 0;	// L205
  int32_t r96;	// L206
  r96 = 0;	// L207
  int32_t r97;	// L208
  r97 = 0;	// L209
  int32_t r98;	// L210
  r98 = 0;	// L211
  int32_t r99;	// L212
  r99 = 0;	// L213
  int32_t r100;	// L214
  r100 = 0;	// L215
  int32_t r101;	// L216
  r101 = 0;	// L217
  int32_t r102;	// L218
  r102 = 0;	// L219
  int32_t r103;	// L220
  r103 = 0;	// L221
  int32_t r104;	// L222
  r104 = 0;	// L223
  int32_t r105;	// L224
  r105 = 0;	// L225
  int32_t r106;	// L226
  r106 = 0;	// L227
  int32_t r107;	// L228
  r107 = 0;	// L229
  int32_t r108;	// L230
  r108 = 0;	// L231
  int32_t r109;	// L232
  r109 = 0;	// L233
  int32_t r110;	// L234
  r110 = 0;	// L235
  int32_t r111;	// L236
  r111 = 0;	// L237
  int32_t r112;	// L238
  r112 = 0;	// L239
  int32_t r113;	// L240
  r113 = 0;	// L241
  int32_t r114;	// L242
  r114 = 0;	// L243
  int32_t r115;	// L244
  r115 = 0;	// L245
  int32_t r116;	// L246
  r116 = 0;	// L247
  int32_t r117;	// L248
  r117 = 0;	// L249
  int32_t r118;	// L250
  r118 = 0;	// L251
  int32_t r119;	// L252
  r119 = 0;	// L253
  int32_t r120;	// L254
  r120 = 0;	// L255
  int32_t r121;	// L256
  r121 = 0;	// L257
  int32_t r122;	// L258
  r122 = 0;	// L259
  int32_t r123;	// L260
  r123 = 0;	// L261
  int32_t r124;	// L262
  r124 = 0;	// L263
  int32_t r125;	// L264
  r125 = 0;	// L265
  int32_t r126;	// L266
  r126 = 0;	// L267
  int32_t r127;	// L268
  r127 = 0;	// L269
  l_S_s_0_s: for (int s = 0; s < 1048576; s++) {	// L270
  #pragma HLS pipeline II=1
    int v139 = s >> 6;	// L273
    int32_t v140 = v139;	// L274
    int32_t blk;	// L275
    blk = v140;	// L276
    int32_t v142 = blk;	// L277
    int32_t v143 = v142 & 1;	// L279
    int32_t p;	// L280
    p = v143;	// L281
    int32_t v145 = blk;	// L282
    int32_t v146 = v145 >> 1;	// L283
    ap_int<33> v147 = v146;	// L288
    ap_int<33> v148 = v147 & 511;	// L289
    int32_t v149 = v148;	// L290
    int32_t kb;	// L291
    kb = v149;	// L292
    int32_t v151 = v2.read();	// L293
    int32_t z;	// L294
    z = v151;	// L295
    int32_t v153 = z;	// L296
    int32_t v;	// L297
    v = v153;	// L298
    int32_t v155 = kb;	// L299
    bool v156 = v155 != 0;	// L300
    if (v156) {	// L301
      int32_t v157 = r0;	// L302
      int32_t v158 = z;	// L303
      ap_int<33> v159 = v157;	// L304
      ap_int<33> v160 = v158;	// L305
      ap_int<33> v161 = v159 + v160;	// L306
      int32_t v162 = v161;	// L307
      v = v162;	// L308
    }
    int32_t v163 = r64;	// L310
    int32_t g;	// L311
    g = v163;	// L312
    int32_t v165 = r1;	// L313
    r0 = v165;	// L314
    int32_t v166 = r2;	// L315
    r1 = v166;	// L316
    int32_t v167 = r3;	// L317
    r2 = v167;	// L318
    int32_t v168 = r4;	// L319
    r3 = v168;	// L320
    int32_t v169 = r5;	// L321
    r4 = v169;	// L322
    int32_t v170 = r6;	// L323
    r5 = v170;	// L324
    int32_t v171 = r7;	// L325
    r6 = v171;	// L326
    int32_t v172 = r8;	// L327
    r7 = v172;	// L328
    int32_t v173 = r9;	// L329
    r8 = v173;	// L330
    int32_t v174 = r10;	// L331
    r9 = v174;	// L332
    int32_t v175 = r11;	// L333
    r10 = v175;	// L334
    int32_t v176 = r12;	// L335
    r11 = v176;	// L336
    int32_t v177 = r13;	// L337
    r12 = v177;	// L338
    int32_t v178 = r14;	// L339
    r13 = v178;	// L340
    int32_t v179 = r15;	// L341
    r14 = v179;	// L342
    int32_t v180 = r16;	// L343
    r15 = v180;	// L344
    int32_t v181 = r17;	// L345
    r16 = v181;	// L346
    int32_t v182 = r18;	// L347
    r17 = v182;	// L348
    int32_t v183 = r19;	// L349
    r18 = v183;	// L350
    int32_t v184 = r20;	// L351
    r19 = v184;	// L352
    int32_t v185 = r21;	// L353
    r20 = v185;	// L354
    int32_t v186 = r22;	// L355
    r21 = v186;	// L356
    int32_t v187 = r23;	// L357
    r22 = v187;	// L358
    int32_t v188 = r24;	// L359
    r23 = v188;	// L360
    int32_t v189 = r25;	// L361
    r24 = v189;	// L362
    int32_t v190 = r26;	// L363
    r25 = v190;	// L364
    int32_t v191 = r27;	// L365
    r26 = v191;	// L366
    int32_t v192 = r28;	// L367
    r27 = v192;	// L368
    int32_t v193 = r29;	// L369
    r28 = v193;	// L370
    int32_t v194 = r30;	// L371
    r29 = v194;	// L372
    int32_t v195 = r31;	// L373
    r30 = v195;	// L374
    int32_t v196 = r32;	// L375
    r31 = v196;	// L376
    int32_t v197 = r33;	// L377
    r32 = v197;	// L378
    int32_t v198 = r34;	// L379
    r33 = v198;	// L380
    int32_t v199 = r35;	// L381
    r34 = v199;	// L382
    int32_t v200 = r36;	// L383
    r35 = v200;	// L384
    int32_t v201 = r37;	// L385
    r36 = v201;	// L386
    int32_t v202 = r38;	// L387
    r37 = v202;	// L388
    int32_t v203 = r39;	// L389
    r38 = v203;	// L390
    int32_t v204 = r40;	// L391
    r39 = v204;	// L392
    int32_t v205 = r41;	// L393
    r40 = v205;	// L394
    int32_t v206 = r42;	// L395
    r41 = v206;	// L396
    int32_t v207 = r43;	// L397
    r42 = v207;	// L398
    int32_t v208 = r44;	// L399
    r43 = v208;	// L400
    int32_t v209 = r45;	// L401
    r44 = v209;	// L402
    int32_t v210 = r46;	// L403
    r45 = v210;	// L404
    int32_t v211 = r47;	// L405
    r46 = v211;	// L406
    int32_t v212 = r48;	// L407
    r47 = v212;	// L408
    int32_t v213 = r49;	// L409
    r48 = v213;	// L410
    int32_t v214 = r50;	// L411
    r49 = v214;	// L412
    int32_t v215 = r51;	// L413
    r50 = v215;	// L414
    int32_t v216 = r52;	// L415
    r51 = v216;	// L416
    int32_t v217 = r53;	// L417
    r52 = v217;	// L418
    int32_t v218 = r54;	// L419
    r53 = v218;	// L420
    int32_t v219 = r55;	// L421
    r54 = v219;	// L422
    int32_t v220 = r56;	// L423
    r55 = v220;	// L424
    int32_t v221 = r57;	// L425
    r56 = v221;	// L426
    int32_t v222 = r58;	// L427
    r57 = v222;	// L428
    int32_t v223 = r59;	// L429
    r58 = v223;	// L430
    int32_t v224 = r60;	// L431
    r59 = v224;	// L432
    int32_t v225 = r61;	// L433
    r60 = v225;	// L434
    int32_t v226 = r62;	// L435
    r61 = v226;	// L436
    int32_t v227 = r63;	// L437
    r62 = v227;	// L438
    int32_t v228 = r64;	// L439
    r63 = v228;	// L440
    int32_t v229 = r65;	// L441
    r64 = v229;	// L442
    int32_t v230 = r66;	// L443
    r65 = v230;	// L444
    int32_t v231 = r67;	// L445
    r66 = v231;	// L446
    int32_t v232 = r68;	// L447
    r67 = v232;	// L448
    int32_t v233 = r69;	// L449
    r68 = v233;	// L450
    int32_t v234 = r70;	// L451
    r69 = v234;	// L452
    int32_t v235 = r71;	// L453
    r70 = v235;	// L454
    int32_t v236 = r72;	// L455
    r71 = v236;	// L456
    int32_t v237 = r73;	// L457
    r72 = v237;	// L458
    int32_t v238 = r74;	// L459
    r73 = v238;	// L460
    int32_t v239 = r75;	// L461
    r74 = v239;	// L462
    int32_t v240 = r76;	// L463
    r75 = v240;	// L464
    int32_t v241 = r77;	// L465
    r76 = v241;	// L466
    int32_t v242 = r78;	// L467
    r77 = v242;	// L468
    int32_t v243 = r79;	// L469
    r78 = v243;	// L470
    int32_t v244 = r80;	// L471
    r79 = v244;	// L472
    int32_t v245 = r81;	// L473
    r80 = v245;	// L474
    int32_t v246 = r82;	// L475
    r81 = v246;	// L476
    int32_t v247 = r83;	// L477
    r82 = v247;	// L478
    int32_t v248 = r84;	// L479
    r83 = v248;	// L480
    int32_t v249 = r85;	// L481
    r84 = v249;	// L482
    int32_t v250 = r86;	// L483
    r85 = v250;	// L484
    int32_t v251 = r87;	// L485
    r86 = v251;	// L486
    int32_t v252 = r88;	// L487
    r87 = v252;	// L488
    int32_t v253 = r89;	// L489
    r88 = v253;	// L490
    int32_t v254 = r90;	// L491
    r89 = v254;	// L492
    int32_t v255 = r91;	// L493
    r90 = v255;	// L494
    int32_t v256 = r92;	// L495
    r91 = v256;	// L496
    int32_t v257 = r93;	// L497
    r92 = v257;	// L498
    int32_t v258 = r94;	// L499
    r93 = v258;	// L500
    int32_t v259 = r95;	// L501
    r94 = v259;	// L502
    int32_t v260 = r96;	// L503
    r95 = v260;	// L504
    int32_t v261 = r97;	// L505
    r96 = v261;	// L506
    int32_t v262 = r98;	// L507
    r97 = v262;	// L508
    int32_t v263 = r99;	// L509
    r98 = v263;	// L510
    int32_t v264 = r100;	// L511
    r99 = v264;	// L512
    int32_t v265 = r101;	// L513
    r100 = v265;	// L514
    int32_t v266 = r102;	// L515
    r101 = v266;	// L516
    int32_t v267 = r103;	// L517
    r102 = v267;	// L518
    int32_t v268 = r104;	// L519
    r103 = v268;	// L520
    int32_t v269 = r105;	// L521
    r104 = v269;	// L522
    int32_t v270 = r106;	// L523
    r105 = v270;	// L524
    int32_t v271 = r107;	// L525
    r106 = v271;	// L526
    int32_t v272 = r108;	// L527
    r107 = v272;	// L528
    int32_t v273 = r109;	// L529
    r108 = v273;	// L530
    int32_t v274 = r110;	// L531
    r109 = v274;	// L532
    int32_t v275 = r111;	// L533
    r110 = v275;	// L534
    int32_t v276 = r112;	// L535
    r111 = v276;	// L536
    int32_t v277 = r113;	// L537
    r112 = v277;	// L538
    int32_t v278 = r114;	// L539
    r113 = v278;	// L540
    int32_t v279 = r115;	// L541
    r114 = v279;	// L542
    int32_t v280 = r116;	// L543
    r115 = v280;	// L544
    int32_t v281 = r117;	// L545
    r116 = v281;	// L546
    int32_t v282 = r118;	// L547
    r117 = v282;	// L548
    int32_t v283 = r119;	// L549
    r118 = v283;	// L550
    int32_t v284 = r120;	// L551
    r119 = v284;	// L552
    int32_t v285 = r121;	// L553
    r120 = v285;	// L554
    int32_t v286 = r122;	// L555
    r121 = v286;	// L556
    int32_t v287 = r123;	// L557
    r122 = v287;	// L558
    int32_t v288 = r124;	// L559
    r123 = v288;	// L560
    int32_t v289 = r125;	// L561
    r124 = v289;	// L562
    int32_t v290 = r126;	// L563
    r125 = v290;	// L564
    int32_t v291 = r127;	// L565
    r126 = v291;	// L566
    int32_t v292 = v;	// L567
    r127 = v292;	// L568
    int32_t v293 = kb;	// L569
    ap_int<33> v294 = v293;	// L570
    bool v295 = v294 == 511;	// L571
    if (v295) {	// L572
      int32_t v296 = p;	// L573
      bool v297 = v296 == 1;	// L574
      if (v297) {	// L575
        int32_t v298 = g;	// L576
        int32_t v299 = sh;	// L577
        int32_t v300 = v298 >> v299;	// L578
        int32_t gq;	// L579
        gq = v300;	// L580
        int32_t v302 = gq;	// L581
        bool v303 = v302 < -128;	// L584
        if (v303) {	// L585
          gq = -128;	// L586
        }
        int32_t v304 = gq;	// L588
        bool v305 = v304 > 127;	// L590
        if (v305) {	// L591
          gq = 127;	// L592
        }
        int32_t v306 = v;	// L594
        int32_t v307 = sh;	// L595
        int32_t v308 = v306 >> v307;	// L596
        int32_t uq;	// L597
        uq = v308;	// L598
        int32_t v310 = uq;	// L599
        bool v311 = v310 < -128;	// L600
        if (v311) {	// L601
          uq = -128;	// L602
        }
        int32_t v312 = uq;	// L604
        bool v313 = v312 > 127;	// L605
        if (v313) {	// L606
          uq = 127;	// L607
        }
        int32_t v314 = gq;	// L609
        int32_t a;	// L610
        a = v314;	// L611
        int32_t v316 = a;	// L612
        bool v317 = v316 < 0;	// L613
        if (v317) {	// L614
          int32_t v318 = a;	// L615
          int32_t v319 = 0 - v318;	// L616
          a = v319;	// L617
        }
        int32_t v320 = a;	// L619
        bool v321 = v320 > 64;	// L621
        if (v321) {	// L622
          a = 64;	// L623
        }
        int32_t v322 = a;	// L625
        ap_int<33> v323 = v322;	// L627
        ap_int<33> v324 = 64 - v323;	// L628
        int8_t v325 = v324;	// L629
        int8_t w;	// L630
        w = v325;	// L631
        int8_t v327 = w;	// L632
        int16_t v328 = v327;	// L633
        int16_t v329 = v328 * v328;	// L634
        #pragma HLS bind_op variable=v329 op=mul impl=fabric
        int16_t v330 = v329 >> 4;	// L637
        ap_int<33> v331 = v330;	// L640
        ap_int<33> v332 = 256 - v331;	// L641
        int16_t v333 = v332;	// L642
        int16_t th;	// L643
        th = v333;	// L644
        int16_t v335 = th;	// L645
        ap_int<33> v336 = v335;	// L646
        ap_int<33> v337 = v336 + 256;	// L647
        int16_t v338 = v337;	// L648
        int16_t sig;	// L649
        sig = v338;	// L650
        int32_t v340 = gq;	// L651
        bool v341 = v340 < 0;	// L652
        if (v341) {	// L653
          int16_t v342 = th;	// L654
          ap_int<33> v343 = v342;	// L655
          ap_int<33> v344 = 256 - v343;	// L656
          int16_t v345 = v344;	// L657
          sig = v345;	// L658
        }
        int16_t v346 = sig;	// L660
        int16_t v347 = v346 >> 1;	// L662
        sig = v347;	// L663
        int32_t v348 = gq;	// L664
        int8_t v349 = v348;	// L665
        int8_t g8;	// L666
        g8 = v349;	// L667
        int32_t v351 = uq;	// L668
        int8_t v352 = v351;	// L669
        int8_t u8;	// L670
        u8 = v352;	// L671
        int8_t v354 = g8;	// L672
        int16_t v355 = sig;	// L673
        ap_int<24> v356 = v354;	// L674
        ap_int<24> v357 = v355;	// L675
        ap_int<24> v358 = v356 * v357;	// L676
        #pragma HLS bind_op variable=v358 op=mul impl=fabric
        int16_t v359 = v358;	// L677
        int16_t gs;	// L678
        gs = v359;	// L679
        int16_t v361 = gs;	// L680
        int8_t v362 = u8;	// L681
        ap_int<24> v363 = v361;	// L682
        ap_int<24> v364 = v362;	// L683
        ap_int<24> v365 = v363 * v364;	// L684
        #pragma HLS bind_op variable=v365 op=mul impl=fabric
        int32_t v366 = v365;	// L685
        int32_t h;	// L686
        h = v366;	// L687
        int32_t v368 = h;	// L688
        int32_t v369 = sh2;	// L689
        int32_t v370 = v368 >> v369;	// L690
        h = v370;	// L691
        int32_t v371 = h;	// L692
        bool v372 = v371 < -128;	// L693
        if (v372) {	// L694
          h = -128;	// L695
        }
        int32_t v373 = h;	// L697
        bool v374 = v373 > 127;	// L698
        if (v374) {	// L699
          h = 127;	// L700
        }
        int32_t v375 = h;	// L702
        v1.write(v375);	// L703
      }
    }
  }
}


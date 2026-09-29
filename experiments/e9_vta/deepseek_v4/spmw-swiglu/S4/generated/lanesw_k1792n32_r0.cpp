
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
void lanesw_k1792n32_r0_0(
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
  int32_t p;	// L270
  p = 0;	// L271
  int32_t kb;	// L272
  kb = 0;	// L273
  l_S_s_0_s: for (int s = 0; s < 3670016; s++) {	// L274
  #pragma HLS pipeline II=1
    int32_t v141 = v2.read();	// L275
    int32_t z;	// L276
    z = v141;	// L277
    int32_t v143 = z;	// L278
    int32_t v;	// L279
    v = v143;	// L280
    int32_t v145 = kb;	// L281
    bool v146 = v145 != 0;	// L282
    if (v146) {	// L283
      int32_t v147 = r0;	// L284
      int32_t v148 = z;	// L285
      ap_int<33> v149 = v147;	// L286
      ap_int<33> v150 = v148;	// L287
      ap_int<33> v151 = v149 + v150;	// L288
      int32_t v152 = v151;	// L289
      v = v152;	// L290
    }
    int32_t v153 = r64;	// L292
    int32_t g;	// L293
    g = v153;	// L294
    int32_t v155 = r1;	// L295
    r0 = v155;	// L296
    int32_t v156 = r2;	// L297
    r1 = v156;	// L298
    int32_t v157 = r3;	// L299
    r2 = v157;	// L300
    int32_t v158 = r4;	// L301
    r3 = v158;	// L302
    int32_t v159 = r5;	// L303
    r4 = v159;	// L304
    int32_t v160 = r6;	// L305
    r5 = v160;	// L306
    int32_t v161 = r7;	// L307
    r6 = v161;	// L308
    int32_t v162 = r8;	// L309
    r7 = v162;	// L310
    int32_t v163 = r9;	// L311
    r8 = v163;	// L312
    int32_t v164 = r10;	// L313
    r9 = v164;	// L314
    int32_t v165 = r11;	// L315
    r10 = v165;	// L316
    int32_t v166 = r12;	// L317
    r11 = v166;	// L318
    int32_t v167 = r13;	// L319
    r12 = v167;	// L320
    int32_t v168 = r14;	// L321
    r13 = v168;	// L322
    int32_t v169 = r15;	// L323
    r14 = v169;	// L324
    int32_t v170 = r16;	// L325
    r15 = v170;	// L326
    int32_t v171 = r17;	// L327
    r16 = v171;	// L328
    int32_t v172 = r18;	// L329
    r17 = v172;	// L330
    int32_t v173 = r19;	// L331
    r18 = v173;	// L332
    int32_t v174 = r20;	// L333
    r19 = v174;	// L334
    int32_t v175 = r21;	// L335
    r20 = v175;	// L336
    int32_t v176 = r22;	// L337
    r21 = v176;	// L338
    int32_t v177 = r23;	// L339
    r22 = v177;	// L340
    int32_t v178 = r24;	// L341
    r23 = v178;	// L342
    int32_t v179 = r25;	// L343
    r24 = v179;	// L344
    int32_t v180 = r26;	// L345
    r25 = v180;	// L346
    int32_t v181 = r27;	// L347
    r26 = v181;	// L348
    int32_t v182 = r28;	// L349
    r27 = v182;	// L350
    int32_t v183 = r29;	// L351
    r28 = v183;	// L352
    int32_t v184 = r30;	// L353
    r29 = v184;	// L354
    int32_t v185 = r31;	// L355
    r30 = v185;	// L356
    int32_t v186 = r32;	// L357
    r31 = v186;	// L358
    int32_t v187 = r33;	// L359
    r32 = v187;	// L360
    int32_t v188 = r34;	// L361
    r33 = v188;	// L362
    int32_t v189 = r35;	// L363
    r34 = v189;	// L364
    int32_t v190 = r36;	// L365
    r35 = v190;	// L366
    int32_t v191 = r37;	// L367
    r36 = v191;	// L368
    int32_t v192 = r38;	// L369
    r37 = v192;	// L370
    int32_t v193 = r39;	// L371
    r38 = v193;	// L372
    int32_t v194 = r40;	// L373
    r39 = v194;	// L374
    int32_t v195 = r41;	// L375
    r40 = v195;	// L376
    int32_t v196 = r42;	// L377
    r41 = v196;	// L378
    int32_t v197 = r43;	// L379
    r42 = v197;	// L380
    int32_t v198 = r44;	// L381
    r43 = v198;	// L382
    int32_t v199 = r45;	// L383
    r44 = v199;	// L384
    int32_t v200 = r46;	// L385
    r45 = v200;	// L386
    int32_t v201 = r47;	// L387
    r46 = v201;	// L388
    int32_t v202 = r48;	// L389
    r47 = v202;	// L390
    int32_t v203 = r49;	// L391
    r48 = v203;	// L392
    int32_t v204 = r50;	// L393
    r49 = v204;	// L394
    int32_t v205 = r51;	// L395
    r50 = v205;	// L396
    int32_t v206 = r52;	// L397
    r51 = v206;	// L398
    int32_t v207 = r53;	// L399
    r52 = v207;	// L400
    int32_t v208 = r54;	// L401
    r53 = v208;	// L402
    int32_t v209 = r55;	// L403
    r54 = v209;	// L404
    int32_t v210 = r56;	// L405
    r55 = v210;	// L406
    int32_t v211 = r57;	// L407
    r56 = v211;	// L408
    int32_t v212 = r58;	// L409
    r57 = v212;	// L410
    int32_t v213 = r59;	// L411
    r58 = v213;	// L412
    int32_t v214 = r60;	// L413
    r59 = v214;	// L414
    int32_t v215 = r61;	// L415
    r60 = v215;	// L416
    int32_t v216 = r62;	// L417
    r61 = v216;	// L418
    int32_t v217 = r63;	// L419
    r62 = v217;	// L420
    int32_t v218 = r64;	// L421
    r63 = v218;	// L422
    int32_t v219 = r65;	// L423
    r64 = v219;	// L424
    int32_t v220 = r66;	// L425
    r65 = v220;	// L426
    int32_t v221 = r67;	// L427
    r66 = v221;	// L428
    int32_t v222 = r68;	// L429
    r67 = v222;	// L430
    int32_t v223 = r69;	// L431
    r68 = v223;	// L432
    int32_t v224 = r70;	// L433
    r69 = v224;	// L434
    int32_t v225 = r71;	// L435
    r70 = v225;	// L436
    int32_t v226 = r72;	// L437
    r71 = v226;	// L438
    int32_t v227 = r73;	// L439
    r72 = v227;	// L440
    int32_t v228 = r74;	// L441
    r73 = v228;	// L442
    int32_t v229 = r75;	// L443
    r74 = v229;	// L444
    int32_t v230 = r76;	// L445
    r75 = v230;	// L446
    int32_t v231 = r77;	// L447
    r76 = v231;	// L448
    int32_t v232 = r78;	// L449
    r77 = v232;	// L450
    int32_t v233 = r79;	// L451
    r78 = v233;	// L452
    int32_t v234 = r80;	// L453
    r79 = v234;	// L454
    int32_t v235 = r81;	// L455
    r80 = v235;	// L456
    int32_t v236 = r82;	// L457
    r81 = v236;	// L458
    int32_t v237 = r83;	// L459
    r82 = v237;	// L460
    int32_t v238 = r84;	// L461
    r83 = v238;	// L462
    int32_t v239 = r85;	// L463
    r84 = v239;	// L464
    int32_t v240 = r86;	// L465
    r85 = v240;	// L466
    int32_t v241 = r87;	// L467
    r86 = v241;	// L468
    int32_t v242 = r88;	// L469
    r87 = v242;	// L470
    int32_t v243 = r89;	// L471
    r88 = v243;	// L472
    int32_t v244 = r90;	// L473
    r89 = v244;	// L474
    int32_t v245 = r91;	// L475
    r90 = v245;	// L476
    int32_t v246 = r92;	// L477
    r91 = v246;	// L478
    int32_t v247 = r93;	// L479
    r92 = v247;	// L480
    int32_t v248 = r94;	// L481
    r93 = v248;	// L482
    int32_t v249 = r95;	// L483
    r94 = v249;	// L484
    int32_t v250 = r96;	// L485
    r95 = v250;	// L486
    int32_t v251 = r97;	// L487
    r96 = v251;	// L488
    int32_t v252 = r98;	// L489
    r97 = v252;	// L490
    int32_t v253 = r99;	// L491
    r98 = v253;	// L492
    int32_t v254 = r100;	// L493
    r99 = v254;	// L494
    int32_t v255 = r101;	// L495
    r100 = v255;	// L496
    int32_t v256 = r102;	// L497
    r101 = v256;	// L498
    int32_t v257 = r103;	// L499
    r102 = v257;	// L500
    int32_t v258 = r104;	// L501
    r103 = v258;	// L502
    int32_t v259 = r105;	// L503
    r104 = v259;	// L504
    int32_t v260 = r106;	// L505
    r105 = v260;	// L506
    int32_t v261 = r107;	// L507
    r106 = v261;	// L508
    int32_t v262 = r108;	// L509
    r107 = v262;	// L510
    int32_t v263 = r109;	// L511
    r108 = v263;	// L512
    int32_t v264 = r110;	// L513
    r109 = v264;	// L514
    int32_t v265 = r111;	// L515
    r110 = v265;	// L516
    int32_t v266 = r112;	// L517
    r111 = v266;	// L518
    int32_t v267 = r113;	// L519
    r112 = v267;	// L520
    int32_t v268 = r114;	// L521
    r113 = v268;	// L522
    int32_t v269 = r115;	// L523
    r114 = v269;	// L524
    int32_t v270 = r116;	// L525
    r115 = v270;	// L526
    int32_t v271 = r117;	// L527
    r116 = v271;	// L528
    int32_t v272 = r118;	// L529
    r117 = v272;	// L530
    int32_t v273 = r119;	// L531
    r118 = v273;	// L532
    int32_t v274 = r120;	// L533
    r119 = v274;	// L534
    int32_t v275 = r121;	// L535
    r120 = v275;	// L536
    int32_t v276 = r122;	// L537
    r121 = v276;	// L538
    int32_t v277 = r123;	// L539
    r122 = v277;	// L540
    int32_t v278 = r124;	// L541
    r123 = v278;	// L542
    int32_t v279 = r125;	// L543
    r124 = v279;	// L544
    int32_t v280 = r126;	// L545
    r125 = v280;	// L546
    int32_t v281 = r127;	// L547
    r126 = v281;	// L548
    int32_t v282 = v;	// L549
    r127 = v282;	// L550
    int32_t v283 = kb;	// L551
    ap_int<33> v284 = v283;	// L557
    bool v285 = v284 == 1791;	// L558
    if (v285) {	// L559
      int32_t v286 = p;	// L560
      bool v287 = v286 == 1;	// L561
      if (v287) {	// L562
        int32_t v288 = g;	// L563
        int32_t v289 = sh;	// L564
        int32_t v290 = v288 >> v289;	// L565
        int32_t gq;	// L566
        gq = v290;	// L567
        int32_t v292 = gq;	// L568
        bool v293 = v292 < -128;	// L571
        if (v293) {	// L572
          gq = -128;	// L573
        }
        int32_t v294 = gq;	// L575
        bool v295 = v294 > 127;	// L577
        if (v295) {	// L578
          gq = 127;	// L579
        }
        int32_t v296 = v;	// L581
        int32_t v297 = sh;	// L582
        int32_t v298 = v296 >> v297;	// L583
        int32_t uq;	// L584
        uq = v298;	// L585
        int32_t v300 = uq;	// L586
        bool v301 = v300 < -128;	// L587
        if (v301) {	// L588
          uq = -128;	// L589
        }
        int32_t v302 = uq;	// L591
        bool v303 = v302 > 127;	// L592
        if (v303) {	// L593
          uq = 127;	// L594
        }
        int32_t v304 = gq;	// L596
        int32_t a;	// L597
        a = v304;	// L598
        int32_t v306 = a;	// L599
        bool v307 = v306 < 0;	// L600
        if (v307) {	// L601
          int32_t v308 = a;	// L602
          int32_t v309 = 0 - v308;	// L603
          a = v309;	// L604
        }
        int32_t v310 = a;	// L606
        bool v311 = v310 > 64;	// L608
        if (v311) {	// L609
          a = 64;	// L610
        }
        int32_t v312 = a;	// L612
        ap_int<33> v313 = v312;	// L614
        ap_int<33> v314 = 64 - v313;	// L615
        int8_t v315 = v314;	// L616
        int8_t w;	// L617
        w = v315;	// L618
        int8_t v317 = w;	// L619
        int16_t v318 = v317;	// L620
        int16_t v319 = v318 * v318;	// L621
        #pragma HLS bind_op variable=v319 op=mul impl=fabric
        int16_t v320 = v319 >> 4;	// L624
        ap_int<33> v321 = v320;	// L627
        ap_int<33> v322 = 256 - v321;	// L628
        int16_t v323 = v322;	// L629
        int16_t th;	// L630
        th = v323;	// L631
        int16_t v325 = th;	// L632
        ap_int<33> v326 = v325;	// L633
        ap_int<33> v327 = v326 + 256;	// L634
        int16_t v328 = v327;	// L635
        int16_t sig;	// L636
        sig = v328;	// L637
        int32_t v330 = gq;	// L638
        bool v331 = v330 < 0;	// L639
        if (v331) {	// L640
          int16_t v332 = th;	// L641
          ap_int<33> v333 = v332;	// L642
          ap_int<33> v334 = 256 - v333;	// L643
          int16_t v335 = v334;	// L644
          sig = v335;	// L645
        }
        int16_t v336 = sig;	// L647
        int16_t v337 = v336 >> 1;	// L649
        sig = v337;	// L650
        int32_t v338 = gq;	// L651
        int8_t v339 = v338;	// L652
        int8_t g8;	// L653
        g8 = v339;	// L654
        int32_t v341 = uq;	// L655
        int8_t v342 = v341;	// L656
        int8_t u8;	// L657
        u8 = v342;	// L658
        int8_t v344 = g8;	// L659
        int16_t v345 = sig;	// L660
        ap_int<24> v346 = v344;	// L661
        ap_int<24> v347 = v345;	// L662
        ap_int<24> v348 = v346 * v347;	// L663
        #pragma HLS bind_op variable=v348 op=mul impl=fabric
        int16_t v349 = v348;	// L664
        int16_t gs;	// L665
        gs = v349;	// L666
        int16_t v351 = gs;	// L667
        int8_t v352 = u8;	// L668
        ap_int<24> v353 = v351;	// L669
        ap_int<24> v354 = v352;	// L670
        ap_int<24> v355 = v353 * v354;	// L671
        #pragma HLS bind_op variable=v355 op=mul impl=fabric
        int32_t v356 = v355;	// L672
        int32_t h;	// L673
        h = v356;	// L674
        int32_t v358 = h;	// L675
        int32_t v359 = sh2;	// L676
        int32_t v360 = v358 >> v359;	// L677
        h = v360;	// L678
        int32_t v361 = h;	// L679
        bool v362 = v361 < -128;	// L680
        if (v362) {	// L681
          h = -128;	// L682
        }
        int32_t v363 = h;	// L684
        bool v364 = v363 > 127;	// L685
        if (v364) {	// L686
          h = 127;	// L687
        }
        int32_t v365 = h;	// L689
        v1.write(v365);	// L690
      }
    }
    int32_t v366 = s;	// L693
    int32_t v367 = v366 & 63;	// L695
    bool v368 = v367 == 63;	// L696
    if (v368) {	// L697
      int32_t v369 = p;	// L698
      ap_int<33> v370 = v369;	// L699
      ap_int<33> v371 = 1 - v370;	// L700
      int32_t v372 = v371;	// L701
      p = v372;	// L702
      int32_t v373 = p;	// L703
      bool v374 = v373 == 0;	// L704
      if (v374) {	// L705
        int32_t v375 = kb;	// L706
        ap_int<33> v376 = v375;	// L707
        ap_int<33> v377 = v376 + 1;	// L708
        int32_t v378 = v377;	// L709
        kb = v378;	// L710
        int32_t v379 = kb;	// L711
        bool v380 = v379 == 1792;	// L712
        if (v380) {	// L713
          kb = 0;	// L714
        }
      }
    }
  }
}


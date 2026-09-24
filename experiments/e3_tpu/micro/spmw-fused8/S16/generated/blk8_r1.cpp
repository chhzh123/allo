
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void blk8_r1_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int8_t >& v5,
  hls::stream< int8_t >& v6,
  hls::stream< int8_t >& v7,
  hls::stream< int32_t >& v8,
  hls::stream< int32_t >& v9,
  hls::stream< int32_t >& v10,
  hls::stream< int32_t >& v11,
  hls::stream< int32_t >& v12,
  hls::stream< int32_t >& v13,
  hls::stream< int32_t >& v14,
  hls::stream< int32_t >& v15,
  hls::stream< int32_t >& v16,
  hls::stream< int32_t >& v17,
  hls::stream< int32_t >& v18,
  hls::stream< int32_t >& v19,
  hls::stream< int32_t >& v20,
  hls::stream< int32_t >& v21,
  hls::stream< int32_t >& v22,
  hls::stream< int32_t >& v23,
  hls::stream< int8_t >& v24,
  hls::stream< int8_t >& v25,
  hls::stream< int8_t >& v26,
  hls::stream< int8_t >& v27,
  hls::stream< int8_t >& v28,
  hls::stream< int8_t >& v29,
  hls::stream< int8_t >& v30,
  hls::stream< int8_t >& v31
) {	// L2
  int8_t n0_0;	// L5
  n0_0 = 0;	// L6
  int8_t n0_1;	// L7
  n0_1 = 0;	// L8
  int8_t n0_2;	// L9
  n0_2 = 0;	// L10
  int8_t n0_3;	// L11
  n0_3 = 0;	// L12
  int8_t n0_4;	// L13
  n0_4 = 0;	// L14
  int8_t n0_5;	// L15
  n0_5 = 0;	// L16
  int8_t n0_6;	// L17
  n0_6 = 0;	// L18
  int8_t n0_7;	// L19
  n0_7 = 0;	// L20
  int8_t n1_0;	// L21
  n1_0 = 0;	// L22
  int8_t n1_1;	// L23
  n1_1 = 0;	// L24
  int8_t n1_2;	// L25
  n1_2 = 0;	// L26
  int8_t n1_3;	// L27
  n1_3 = 0;	// L28
  int8_t n1_4;	// L29
  n1_4 = 0;	// L30
  int8_t n1_5;	// L31
  n1_5 = 0;	// L32
  int8_t n1_6;	// L33
  n1_6 = 0;	// L34
  int8_t n1_7;	// L35
  n1_7 = 0;	// L36
  int8_t n2_0;	// L37
  n2_0 = 0;	// L38
  int8_t n2_1;	// L39
  n2_1 = 0;	// L40
  int8_t n2_2;	// L41
  n2_2 = 0;	// L42
  int8_t n2_3;	// L43
  n2_3 = 0;	// L44
  int8_t n2_4;	// L45
  n2_4 = 0;	// L46
  int8_t n2_5;	// L47
  n2_5 = 0;	// L48
  int8_t n2_6;	// L49
  n2_6 = 0;	// L50
  int8_t n2_7;	// L51
  n2_7 = 0;	// L52
  int8_t n3_0;	// L53
  n3_0 = 0;	// L54
  int8_t n3_1;	// L55
  n3_1 = 0;	// L56
  int8_t n3_2;	// L57
  n3_2 = 0;	// L58
  int8_t n3_3;	// L59
  n3_3 = 0;	// L60
  int8_t n3_4;	// L61
  n3_4 = 0;	// L62
  int8_t n3_5;	// L63
  n3_5 = 0;	// L64
  int8_t n3_6;	// L65
  n3_6 = 0;	// L66
  int8_t n3_7;	// L67
  n3_7 = 0;	// L68
  int8_t n4_0;	// L69
  n4_0 = 0;	// L70
  int8_t n4_1;	// L71
  n4_1 = 0;	// L72
  int8_t n4_2;	// L73
  n4_2 = 0;	// L74
  int8_t n4_3;	// L75
  n4_3 = 0;	// L76
  int8_t n4_4;	// L77
  n4_4 = 0;	// L78
  int8_t n4_5;	// L79
  n4_5 = 0;	// L80
  int8_t n4_6;	// L81
  n4_6 = 0;	// L82
  int8_t n4_7;	// L83
  n4_7 = 0;	// L84
  int8_t n5_0;	// L85
  n5_0 = 0;	// L86
  int8_t n5_1;	// L87
  n5_1 = 0;	// L88
  int8_t n5_2;	// L89
  n5_2 = 0;	// L90
  int8_t n5_3;	// L91
  n5_3 = 0;	// L92
  int8_t n5_4;	// L93
  n5_4 = 0;	// L94
  int8_t n5_5;	// L95
  n5_5 = 0;	// L96
  int8_t n5_6;	// L97
  n5_6 = 0;	// L98
  int8_t n5_7;	// L99
  n5_7 = 0;	// L100
  int8_t n6_0;	// L101
  n6_0 = 0;	// L102
  int8_t n6_1;	// L103
  n6_1 = 0;	// L104
  int8_t n6_2;	// L105
  n6_2 = 0;	// L106
  int8_t n6_3;	// L107
  n6_3 = 0;	// L108
  int8_t n6_4;	// L109
  n6_4 = 0;	// L110
  int8_t n6_5;	// L111
  n6_5 = 0;	// L112
  int8_t n6_6;	// L113
  n6_6 = 0;	// L114
  int8_t n6_7;	// L115
  n6_7 = 0;	// L116
  int8_t n7_0;	// L117
  n7_0 = 0;	// L118
  int8_t n7_1;	// L119
  n7_1 = 0;	// L120
  int8_t n7_2;	// L121
  n7_2 = 0;	// L122
  int8_t n7_3;	// L123
  n7_3 = 0;	// L124
  int8_t n7_4;	// L125
  n7_4 = 0;	// L126
  int8_t n7_5;	// L127
  n7_5 = 0;	// L128
  int8_t n7_6;	// L129
  n7_6 = 0;	// L130
  int8_t n7_7;	// L131
  n7_7 = 0;	// L132
  l_S__k_0__k: for (int _k = 0; _k < 16; _k++) {	// L133
  #pragma HLS pipeline II=1
    int8_t v97 = n0_6;	// L134
    n0_7 = v97;	// L135
    int8_t v98 = n0_5;	// L136
    n0_6 = v98;	// L137
    int8_t v99 = n0_4;	// L138
    n0_5 = v99;	// L139
    int8_t v100 = n0_3;	// L140
    n0_4 = v100;	// L141
    int8_t v101 = n0_2;	// L142
    n0_3 = v101;	// L143
    int8_t v102 = n0_1;	// L144
    n0_2 = v102;	// L145
    int8_t v103 = n0_0;	// L146
    n0_1 = v103;	// L147
    int8_t v104 = v24.read();	// L148
    n0_0 = v104;	// L149
    int8_t v105 = n1_6;	// L150
    n1_7 = v105;	// L151
    int8_t v106 = n1_5;	// L152
    n1_6 = v106;	// L153
    int8_t v107 = n1_4;	// L154
    n1_5 = v107;	// L155
    int8_t v108 = n1_3;	// L156
    n1_4 = v108;	// L157
    int8_t v109 = n1_2;	// L158
    n1_3 = v109;	// L159
    int8_t v110 = n1_1;	// L160
    n1_2 = v110;	// L161
    int8_t v111 = n1_0;	// L162
    n1_1 = v111;	// L163
    int8_t v112 = v25.read();	// L164
    n1_0 = v112;	// L165
    int8_t v113 = n2_6;	// L166
    n2_7 = v113;	// L167
    int8_t v114 = n2_5;	// L168
    n2_6 = v114;	// L169
    int8_t v115 = n2_4;	// L170
    n2_5 = v115;	// L171
    int8_t v116 = n2_3;	// L172
    n2_4 = v116;	// L173
    int8_t v117 = n2_2;	// L174
    n2_3 = v117;	// L175
    int8_t v118 = n2_1;	// L176
    n2_2 = v118;	// L177
    int8_t v119 = n2_0;	// L178
    n2_1 = v119;	// L179
    int8_t v120 = v26.read();	// L180
    n2_0 = v120;	// L181
    int8_t v121 = n3_6;	// L182
    n3_7 = v121;	// L183
    int8_t v122 = n3_5;	// L184
    n3_6 = v122;	// L185
    int8_t v123 = n3_4;	// L186
    n3_5 = v123;	// L187
    int8_t v124 = n3_3;	// L188
    n3_4 = v124;	// L189
    int8_t v125 = n3_2;	// L190
    n3_3 = v125;	// L191
    int8_t v126 = n3_1;	// L192
    n3_2 = v126;	// L193
    int8_t v127 = n3_0;	// L194
    n3_1 = v127;	// L195
    int8_t v128 = v27.read();	// L196
    n3_0 = v128;	// L197
    int8_t v129 = n4_6;	// L198
    n4_7 = v129;	// L199
    int8_t v130 = n4_5;	// L200
    n4_6 = v130;	// L201
    int8_t v131 = n4_4;	// L202
    n4_5 = v131;	// L203
    int8_t v132 = n4_3;	// L204
    n4_4 = v132;	// L205
    int8_t v133 = n4_2;	// L206
    n4_3 = v133;	// L207
    int8_t v134 = n4_1;	// L208
    n4_2 = v134;	// L209
    int8_t v135 = n4_0;	// L210
    n4_1 = v135;	// L211
    int8_t v136 = v28.read();	// L212
    n4_0 = v136;	// L213
    int8_t v137 = n5_6;	// L214
    n5_7 = v137;	// L215
    int8_t v138 = n5_5;	// L216
    n5_6 = v138;	// L217
    int8_t v139 = n5_4;	// L218
    n5_5 = v139;	// L219
    int8_t v140 = n5_3;	// L220
    n5_4 = v140;	// L221
    int8_t v141 = n5_2;	// L222
    n5_3 = v141;	// L223
    int8_t v142 = n5_1;	// L224
    n5_2 = v142;	// L225
    int8_t v143 = n5_0;	// L226
    n5_1 = v143;	// L227
    int8_t v144 = v29.read();	// L228
    n5_0 = v144;	// L229
    int8_t v145 = n6_6;	// L230
    n6_7 = v145;	// L231
    int8_t v146 = n6_5;	// L232
    n6_6 = v146;	// L233
    int8_t v147 = n6_4;	// L234
    n6_5 = v147;	// L235
    int8_t v148 = n6_3;	// L236
    n6_4 = v148;	// L237
    int8_t v149 = n6_2;	// L238
    n6_3 = v149;	// L239
    int8_t v150 = n6_1;	// L240
    n6_2 = v150;	// L241
    int8_t v151 = n6_0;	// L242
    n6_1 = v151;	// L243
    int8_t v152 = v30.read();	// L244
    n6_0 = v152;	// L245
    int8_t v153 = n7_6;	// L246
    n7_7 = v153;	// L247
    int8_t v154 = n7_5;	// L248
    n7_6 = v154;	// L249
    int8_t v155 = n7_4;	// L250
    n7_5 = v155;	// L251
    int8_t v156 = n7_3;	// L252
    n7_4 = v156;	// L253
    int8_t v157 = n7_2;	// L254
    n7_3 = v157;	// L255
    int8_t v158 = n7_1;	// L256
    n7_2 = v158;	// L257
    int8_t v159 = n7_0;	// L258
    n7_1 = v159;	// L259
    int8_t v160 = v31.read();	// L260
    n7_0 = v160;	// L261
  }
  int8_t v161 = n0_0;	// L263
  int8_t c0_0;	// L264
  c0_0 = v161;	// L265
  int8_t v163 = n0_1;	// L266
  int8_t c0_1;	// L267
  c0_1 = v163;	// L268
  int8_t v165 = n0_2;	// L269
  int8_t c0_2;	// L270
  c0_2 = v165;	// L271
  int8_t v167 = n0_3;	// L272
  int8_t c0_3;	// L273
  c0_3 = v167;	// L274
  int8_t v169 = n0_4;	// L275
  int8_t c0_4;	// L276
  c0_4 = v169;	// L277
  int8_t v171 = n0_5;	// L278
  int8_t c0_5;	// L279
  c0_5 = v171;	// L280
  int8_t v173 = n0_6;	// L281
  int8_t c0_6;	// L282
  c0_6 = v173;	// L283
  int8_t v175 = n0_7;	// L284
  int8_t c0_7;	// L285
  c0_7 = v175;	// L286
  int8_t v177 = n1_0;	// L287
  int8_t c1_0;	// L288
  c1_0 = v177;	// L289
  int8_t v179 = n1_1;	// L290
  int8_t c1_1;	// L291
  c1_1 = v179;	// L292
  int8_t v181 = n1_2;	// L293
  int8_t c1_2;	// L294
  c1_2 = v181;	// L295
  int8_t v183 = n1_3;	// L296
  int8_t c1_3;	// L297
  c1_3 = v183;	// L298
  int8_t v185 = n1_4;	// L299
  int8_t c1_4;	// L300
  c1_4 = v185;	// L301
  int8_t v187 = n1_5;	// L302
  int8_t c1_5;	// L303
  c1_5 = v187;	// L304
  int8_t v189 = n1_6;	// L305
  int8_t c1_6;	// L306
  c1_6 = v189;	// L307
  int8_t v191 = n1_7;	// L308
  int8_t c1_7;	// L309
  c1_7 = v191;	// L310
  int8_t v193 = n2_0;	// L311
  int8_t c2_0;	// L312
  c2_0 = v193;	// L313
  int8_t v195 = n2_1;	// L314
  int8_t c2_1;	// L315
  c2_1 = v195;	// L316
  int8_t v197 = n2_2;	// L317
  int8_t c2_2;	// L318
  c2_2 = v197;	// L319
  int8_t v199 = n2_3;	// L320
  int8_t c2_3;	// L321
  c2_3 = v199;	// L322
  int8_t v201 = n2_4;	// L323
  int8_t c2_4;	// L324
  c2_4 = v201;	// L325
  int8_t v203 = n2_5;	// L326
  int8_t c2_5;	// L327
  c2_5 = v203;	// L328
  int8_t v205 = n2_6;	// L329
  int8_t c2_6;	// L330
  c2_6 = v205;	// L331
  int8_t v207 = n2_7;	// L332
  int8_t c2_7;	// L333
  c2_7 = v207;	// L334
  int8_t v209 = n3_0;	// L335
  int8_t c3_0;	// L336
  c3_0 = v209;	// L337
  int8_t v211 = n3_1;	// L338
  int8_t c3_1;	// L339
  c3_1 = v211;	// L340
  int8_t v213 = n3_2;	// L341
  int8_t c3_2;	// L342
  c3_2 = v213;	// L343
  int8_t v215 = n3_3;	// L344
  int8_t c3_3;	// L345
  c3_3 = v215;	// L346
  int8_t v217 = n3_4;	// L347
  int8_t c3_4;	// L348
  c3_4 = v217;	// L349
  int8_t v219 = n3_5;	// L350
  int8_t c3_5;	// L351
  c3_5 = v219;	// L352
  int8_t v221 = n3_6;	// L353
  int8_t c3_6;	// L354
  c3_6 = v221;	// L355
  int8_t v223 = n3_7;	// L356
  int8_t c3_7;	// L357
  c3_7 = v223;	// L358
  int8_t v225 = n4_0;	// L359
  int8_t c4_0;	// L360
  c4_0 = v225;	// L361
  int8_t v227 = n4_1;	// L362
  int8_t c4_1;	// L363
  c4_1 = v227;	// L364
  int8_t v229 = n4_2;	// L365
  int8_t c4_2;	// L366
  c4_2 = v229;	// L367
  int8_t v231 = n4_3;	// L368
  int8_t c4_3;	// L369
  c4_3 = v231;	// L370
  int8_t v233 = n4_4;	// L371
  int8_t c4_4;	// L372
  c4_4 = v233;	// L373
  int8_t v235 = n4_5;	// L374
  int8_t c4_5;	// L375
  c4_5 = v235;	// L376
  int8_t v237 = n4_6;	// L377
  int8_t c4_6;	// L378
  c4_6 = v237;	// L379
  int8_t v239 = n4_7;	// L380
  int8_t c4_7;	// L381
  c4_7 = v239;	// L382
  int8_t v241 = n5_0;	// L383
  int8_t c5_0;	// L384
  c5_0 = v241;	// L385
  int8_t v243 = n5_1;	// L386
  int8_t c5_1;	// L387
  c5_1 = v243;	// L388
  int8_t v245 = n5_2;	// L389
  int8_t c5_2;	// L390
  c5_2 = v245;	// L391
  int8_t v247 = n5_3;	// L392
  int8_t c5_3;	// L393
  c5_3 = v247;	// L394
  int8_t v249 = n5_4;	// L395
  int8_t c5_4;	// L396
  c5_4 = v249;	// L397
  int8_t v251 = n5_5;	// L398
  int8_t c5_5;	// L399
  c5_5 = v251;	// L400
  int8_t v253 = n5_6;	// L401
  int8_t c5_6;	// L402
  c5_6 = v253;	// L403
  int8_t v255 = n5_7;	// L404
  int8_t c5_7;	// L405
  c5_7 = v255;	// L406
  int8_t v257 = n6_0;	// L407
  int8_t c6_0;	// L408
  c6_0 = v257;	// L409
  int8_t v259 = n6_1;	// L410
  int8_t c6_1;	// L411
  c6_1 = v259;	// L412
  int8_t v261 = n6_2;	// L413
  int8_t c6_2;	// L414
  c6_2 = v261;	// L415
  int8_t v263 = n6_3;	// L416
  int8_t c6_3;	// L417
  c6_3 = v263;	// L418
  int8_t v265 = n6_4;	// L419
  int8_t c6_4;	// L420
  c6_4 = v265;	// L421
  int8_t v267 = n6_5;	// L422
  int8_t c6_5;	// L423
  c6_5 = v267;	// L424
  int8_t v269 = n6_6;	// L425
  int8_t c6_6;	// L426
  c6_6 = v269;	// L427
  int8_t v271 = n6_7;	// L428
  int8_t c6_7;	// L429
  c6_7 = v271;	// L430
  int8_t v273 = n7_0;	// L431
  int8_t c7_0;	// L432
  c7_0 = v273;	// L433
  int8_t v275 = n7_1;	// L434
  int8_t c7_1;	// L435
  c7_1 = v275;	// L436
  int8_t v277 = n7_2;	// L437
  int8_t c7_2;	// L438
  c7_2 = v277;	// L439
  int8_t v279 = n7_3;	// L440
  int8_t c7_3;	// L441
  c7_3 = v279;	// L442
  int8_t v281 = n7_4;	// L443
  int8_t c7_4;	// L444
  c7_4 = v281;	// L445
  int8_t v283 = n7_5;	// L446
  int8_t c7_5;	// L447
  c7_5 = v283;	// L448
  int8_t v285 = n7_6;	// L449
  int8_t c7_6;	// L450
  c7_6 = v285;	// L451
  int8_t v287 = n7_7;	// L452
  int8_t c7_7;	// L453
  c7_7 = v287;	// L454
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L455
  #pragma HLS pipeline II=1
    int8_t v290 = v0.read();	// L456
    int8_t a0;	// L457
    a0 = v290;	// L458
    int8_t v292 = v1.read();	// L459
    int8_t a1;	// L460
    a1 = v292;	// L461
    int8_t v294 = v2.read();	// L462
    int8_t a2;	// L463
    a2 = v294;	// L464
    int8_t v296 = v3.read();	// L465
    int8_t a3;	// L466
    a3 = v296;	// L467
    int8_t v298 = v4.read();	// L468
    int8_t a4;	// L469
    a4 = v298;	// L470
    int8_t v300 = v5.read();	// L471
    int8_t a5;	// L472
    a5 = v300;	// L473
    int8_t v302 = v6.read();	// L474
    int8_t a6;	// L475
    a6 = v302;	// L476
    int8_t v304 = v7.read();	// L477
    int8_t a7;	// L478
    a7 = v304;	// L479
    int32_t v306 = v8.read();	// L480
    int32_t p0;	// L481
    p0 = v306;	// L482
    int32_t v308 = v10.read();	// L483
    int32_t p1;	// L484
    p1 = v308;	// L485
    int32_t v310 = v12.read();	// L486
    int32_t p2;	// L487
    p2 = v310;	// L488
    int32_t v312 = v14.read();	// L489
    int32_t p3;	// L490
    p3 = v312;	// L491
    int32_t v314 = v16.read();	// L492
    int32_t p4;	// L493
    p4 = v314;	// L494
    int32_t v316 = v18.read();	// L495
    int32_t p5;	// L496
    p5 = v316;	// L497
    int32_t v318 = v20.read();	// L498
    int32_t p6;	// L499
    p6 = v318;	// L500
    int32_t v320 = v22.read();	// L501
    int32_t p7;	// L502
    p7 = v320;	// L503
    int32_t v322 = p0;	// L504
    int8_t v323 = a0;	// L505
    int8_t v324 = c0_0;	// L506
    int16_t v325 = v323;	// L507
    int16_t v326 = v324;	// L508
    int16_t v327 = v325 * v326;	// L509
    #pragma HLS bind_op variable=v327 op=mul impl=fabric
    int8_t v328 = a1;	// L510
    int8_t v329 = c1_0;	// L511
    int16_t v330 = v328;	// L512
    int16_t v331 = v329;	// L513
    int16_t v332 = v330 * v331;	// L514
    #pragma HLS bind_op variable=v332 op=mul impl=fabric
    ap_int<17> v333 = v327;	// L515
    ap_int<17> v334 = v332;	// L516
    ap_int<17> v335 = v333 + v334;	// L517
    int8_t v336 = a2;	// L518
    int8_t v337 = c2_0;	// L519
    int16_t v338 = v336;	// L520
    int16_t v339 = v337;	// L521
    int16_t v340 = v338 * v339;	// L522
    #pragma HLS bind_op variable=v340 op=mul impl=fabric
    int8_t v341 = a3;	// L523
    int8_t v342 = c3_0;	// L524
    int16_t v343 = v341;	// L525
    int16_t v344 = v342;	// L526
    int16_t v345 = v343 * v344;	// L527
    #pragma HLS bind_op variable=v345 op=mul impl=fabric
    ap_int<17> v346 = v340;	// L528
    ap_int<17> v347 = v345;	// L529
    ap_int<17> v348 = v346 + v347;	// L530
    ap_int<18> v349 = v335;	// L531
    ap_int<18> v350 = v348;	// L532
    ap_int<18> v351 = v349 + v350;	// L533
    int8_t v352 = a4;	// L534
    int8_t v353 = c4_0;	// L535
    int16_t v354 = v352;	// L536
    int16_t v355 = v353;	// L537
    int16_t v356 = v354 * v355;	// L538
    #pragma HLS bind_op variable=v356 op=mul impl=fabric
    int8_t v357 = a5;	// L539
    int8_t v358 = c5_0;	// L540
    int16_t v359 = v357;	// L541
    int16_t v360 = v358;	// L542
    int16_t v361 = v359 * v360;	// L543
    #pragma HLS bind_op variable=v361 op=mul impl=fabric
    ap_int<17> v362 = v356;	// L544
    ap_int<17> v363 = v361;	// L545
    ap_int<17> v364 = v362 + v363;	// L546
    int8_t v365 = a6;	// L547
    int8_t v366 = c6_0;	// L548
    int16_t v367 = v365;	// L549
    int16_t v368 = v366;	// L550
    int16_t v369 = v367 * v368;	// L551
    #pragma HLS bind_op variable=v369 op=mul impl=fabric
    int8_t v370 = a7;	// L552
    int8_t v371 = c7_0;	// L553
    int16_t v372 = v370;	// L554
    int16_t v373 = v371;	// L555
    int16_t v374 = v372 * v373;	// L556
    #pragma HLS bind_op variable=v374 op=mul impl=fabric
    ap_int<17> v375 = v369;	// L557
    ap_int<17> v376 = v374;	// L558
    ap_int<17> v377 = v375 + v376;	// L559
    ap_int<18> v378 = v364;	// L560
    ap_int<18> v379 = v377;	// L561
    ap_int<18> v380 = v378 + v379;	// L562
    ap_int<19> v381 = v351;	// L563
    ap_int<19> v382 = v380;	// L564
    ap_int<19> v383 = v381 + v382;	// L565
    ap_int<33> v384 = v322;	// L566
    ap_int<33> v385 = v383;	// L567
    ap_int<33> v386 = v384 + v385;	// L568
    v9.write(v386);	// L569
    int32_t v387 = p1;	// L570
    int8_t v388 = a0;	// L571
    int8_t v389 = c0_1;	// L572
    int16_t v390 = v388;	// L573
    int16_t v391 = v389;	// L574
    int16_t v392 = v390 * v391;	// L575
    #pragma HLS bind_op variable=v392 op=mul impl=fabric
    int8_t v393 = a1;	// L576
    int8_t v394 = c1_1;	// L577
    int16_t v395 = v393;	// L578
    int16_t v396 = v394;	// L579
    int16_t v397 = v395 * v396;	// L580
    #pragma HLS bind_op variable=v397 op=mul impl=fabric
    ap_int<17> v398 = v392;	// L581
    ap_int<17> v399 = v397;	// L582
    ap_int<17> v400 = v398 + v399;	// L583
    int8_t v401 = a2;	// L584
    int8_t v402 = c2_1;	// L585
    int16_t v403 = v401;	// L586
    int16_t v404 = v402;	// L587
    int16_t v405 = v403 * v404;	// L588
    #pragma HLS bind_op variable=v405 op=mul impl=fabric
    int8_t v406 = a3;	// L589
    int8_t v407 = c3_1;	// L590
    int16_t v408 = v406;	// L591
    int16_t v409 = v407;	// L592
    int16_t v410 = v408 * v409;	// L593
    #pragma HLS bind_op variable=v410 op=mul impl=fabric
    ap_int<17> v411 = v405;	// L594
    ap_int<17> v412 = v410;	// L595
    ap_int<17> v413 = v411 + v412;	// L596
    ap_int<18> v414 = v400;	// L597
    ap_int<18> v415 = v413;	// L598
    ap_int<18> v416 = v414 + v415;	// L599
    int8_t v417 = a4;	// L600
    int8_t v418 = c4_1;	// L601
    int16_t v419 = v417;	// L602
    int16_t v420 = v418;	// L603
    int16_t v421 = v419 * v420;	// L604
    #pragma HLS bind_op variable=v421 op=mul impl=fabric
    int8_t v422 = a5;	// L605
    int8_t v423 = c5_1;	// L606
    int16_t v424 = v422;	// L607
    int16_t v425 = v423;	// L608
    int16_t v426 = v424 * v425;	// L609
    #pragma HLS bind_op variable=v426 op=mul impl=fabric
    ap_int<17> v427 = v421;	// L610
    ap_int<17> v428 = v426;	// L611
    ap_int<17> v429 = v427 + v428;	// L612
    int8_t v430 = a6;	// L613
    int8_t v431 = c6_1;	// L614
    int16_t v432 = v430;	// L615
    int16_t v433 = v431;	// L616
    int16_t v434 = v432 * v433;	// L617
    #pragma HLS bind_op variable=v434 op=mul impl=fabric
    int8_t v435 = a7;	// L618
    int8_t v436 = c7_1;	// L619
    int16_t v437 = v435;	// L620
    int16_t v438 = v436;	// L621
    int16_t v439 = v437 * v438;	// L622
    #pragma HLS bind_op variable=v439 op=mul impl=fabric
    ap_int<17> v440 = v434;	// L623
    ap_int<17> v441 = v439;	// L624
    ap_int<17> v442 = v440 + v441;	// L625
    ap_int<18> v443 = v429;	// L626
    ap_int<18> v444 = v442;	// L627
    ap_int<18> v445 = v443 + v444;	// L628
    ap_int<19> v446 = v416;	// L629
    ap_int<19> v447 = v445;	// L630
    ap_int<19> v448 = v446 + v447;	// L631
    ap_int<33> v449 = v387;	// L632
    ap_int<33> v450 = v448;	// L633
    ap_int<33> v451 = v449 + v450;	// L634
    v11.write(v451);	// L635
    int32_t v452 = p2;	// L636
    int8_t v453 = a0;	// L637
    int8_t v454 = c0_2;	// L638
    int16_t v455 = v453;	// L639
    int16_t v456 = v454;	// L640
    int16_t v457 = v455 * v456;	// L641
    #pragma HLS bind_op variable=v457 op=mul impl=fabric
    int8_t v458 = a1;	// L642
    int8_t v459 = c1_2;	// L643
    int16_t v460 = v458;	// L644
    int16_t v461 = v459;	// L645
    int16_t v462 = v460 * v461;	// L646
    #pragma HLS bind_op variable=v462 op=mul impl=fabric
    ap_int<17> v463 = v457;	// L647
    ap_int<17> v464 = v462;	// L648
    ap_int<17> v465 = v463 + v464;	// L649
    int8_t v466 = a2;	// L650
    int8_t v467 = c2_2;	// L651
    int16_t v468 = v466;	// L652
    int16_t v469 = v467;	// L653
    int16_t v470 = v468 * v469;	// L654
    #pragma HLS bind_op variable=v470 op=mul impl=fabric
    int8_t v471 = a3;	// L655
    int8_t v472 = c3_2;	// L656
    int16_t v473 = v471;	// L657
    int16_t v474 = v472;	// L658
    int16_t v475 = v473 * v474;	// L659
    #pragma HLS bind_op variable=v475 op=mul impl=fabric
    ap_int<17> v476 = v470;	// L660
    ap_int<17> v477 = v475;	// L661
    ap_int<17> v478 = v476 + v477;	// L662
    ap_int<18> v479 = v465;	// L663
    ap_int<18> v480 = v478;	// L664
    ap_int<18> v481 = v479 + v480;	// L665
    int8_t v482 = a4;	// L666
    int8_t v483 = c4_2;	// L667
    int16_t v484 = v482;	// L668
    int16_t v485 = v483;	// L669
    int16_t v486 = v484 * v485;	// L670
    #pragma HLS bind_op variable=v486 op=mul impl=fabric
    int8_t v487 = a5;	// L671
    int8_t v488 = c5_2;	// L672
    int16_t v489 = v487;	// L673
    int16_t v490 = v488;	// L674
    int16_t v491 = v489 * v490;	// L675
    #pragma HLS bind_op variable=v491 op=mul impl=fabric
    ap_int<17> v492 = v486;	// L676
    ap_int<17> v493 = v491;	// L677
    ap_int<17> v494 = v492 + v493;	// L678
    int8_t v495 = a6;	// L679
    int8_t v496 = c6_2;	// L680
    int16_t v497 = v495;	// L681
    int16_t v498 = v496;	// L682
    int16_t v499 = v497 * v498;	// L683
    #pragma HLS bind_op variable=v499 op=mul impl=fabric
    int8_t v500 = a7;	// L684
    int8_t v501 = c7_2;	// L685
    int16_t v502 = v500;	// L686
    int16_t v503 = v501;	// L687
    int16_t v504 = v502 * v503;	// L688
    #pragma HLS bind_op variable=v504 op=mul impl=fabric
    ap_int<17> v505 = v499;	// L689
    ap_int<17> v506 = v504;	// L690
    ap_int<17> v507 = v505 + v506;	// L691
    ap_int<18> v508 = v494;	// L692
    ap_int<18> v509 = v507;	// L693
    ap_int<18> v510 = v508 + v509;	// L694
    ap_int<19> v511 = v481;	// L695
    ap_int<19> v512 = v510;	// L696
    ap_int<19> v513 = v511 + v512;	// L697
    ap_int<33> v514 = v452;	// L698
    ap_int<33> v515 = v513;	// L699
    ap_int<33> v516 = v514 + v515;	// L700
    v13.write(v516);	// L701
    int32_t v517 = p3;	// L702
    int8_t v518 = a0;	// L703
    int8_t v519 = c0_3;	// L704
    int16_t v520 = v518;	// L705
    int16_t v521 = v519;	// L706
    int16_t v522 = v520 * v521;	// L707
    #pragma HLS bind_op variable=v522 op=mul impl=fabric
    int8_t v523 = a1;	// L708
    int8_t v524 = c1_3;	// L709
    int16_t v525 = v523;	// L710
    int16_t v526 = v524;	// L711
    int16_t v527 = v525 * v526;	// L712
    #pragma HLS bind_op variable=v527 op=mul impl=fabric
    ap_int<17> v528 = v522;	// L713
    ap_int<17> v529 = v527;	// L714
    ap_int<17> v530 = v528 + v529;	// L715
    int8_t v531 = a2;	// L716
    int8_t v532 = c2_3;	// L717
    int16_t v533 = v531;	// L718
    int16_t v534 = v532;	// L719
    int16_t v535 = v533 * v534;	// L720
    #pragma HLS bind_op variable=v535 op=mul impl=fabric
    int8_t v536 = a3;	// L721
    int8_t v537 = c3_3;	// L722
    int16_t v538 = v536;	// L723
    int16_t v539 = v537;	// L724
    int16_t v540 = v538 * v539;	// L725
    #pragma HLS bind_op variable=v540 op=mul impl=fabric
    ap_int<17> v541 = v535;	// L726
    ap_int<17> v542 = v540;	// L727
    ap_int<17> v543 = v541 + v542;	// L728
    ap_int<18> v544 = v530;	// L729
    ap_int<18> v545 = v543;	// L730
    ap_int<18> v546 = v544 + v545;	// L731
    int8_t v547 = a4;	// L732
    int8_t v548 = c4_3;	// L733
    int16_t v549 = v547;	// L734
    int16_t v550 = v548;	// L735
    int16_t v551 = v549 * v550;	// L736
    #pragma HLS bind_op variable=v551 op=mul impl=fabric
    int8_t v552 = a5;	// L737
    int8_t v553 = c5_3;	// L738
    int16_t v554 = v552;	// L739
    int16_t v555 = v553;	// L740
    int16_t v556 = v554 * v555;	// L741
    #pragma HLS bind_op variable=v556 op=mul impl=fabric
    ap_int<17> v557 = v551;	// L742
    ap_int<17> v558 = v556;	// L743
    ap_int<17> v559 = v557 + v558;	// L744
    int8_t v560 = a6;	// L745
    int8_t v561 = c6_3;	// L746
    int16_t v562 = v560;	// L747
    int16_t v563 = v561;	// L748
    int16_t v564 = v562 * v563;	// L749
    #pragma HLS bind_op variable=v564 op=mul impl=fabric
    int8_t v565 = a7;	// L750
    int8_t v566 = c7_3;	// L751
    int16_t v567 = v565;	// L752
    int16_t v568 = v566;	// L753
    int16_t v569 = v567 * v568;	// L754
    #pragma HLS bind_op variable=v569 op=mul impl=fabric
    ap_int<17> v570 = v564;	// L755
    ap_int<17> v571 = v569;	// L756
    ap_int<17> v572 = v570 + v571;	// L757
    ap_int<18> v573 = v559;	// L758
    ap_int<18> v574 = v572;	// L759
    ap_int<18> v575 = v573 + v574;	// L760
    ap_int<19> v576 = v546;	// L761
    ap_int<19> v577 = v575;	// L762
    ap_int<19> v578 = v576 + v577;	// L763
    ap_int<33> v579 = v517;	// L764
    ap_int<33> v580 = v578;	// L765
    ap_int<33> v581 = v579 + v580;	// L766
    v15.write(v581);	// L767
    int32_t v582 = p4;	// L768
    int8_t v583 = a0;	// L769
    int8_t v584 = c0_4;	// L770
    int16_t v585 = v583;	// L771
    int16_t v586 = v584;	// L772
    int16_t v587 = v585 * v586;	// L773
    #pragma HLS bind_op variable=v587 op=mul impl=fabric
    int8_t v588 = a1;	// L774
    int8_t v589 = c1_4;	// L775
    int16_t v590 = v588;	// L776
    int16_t v591 = v589;	// L777
    int16_t v592 = v590 * v591;	// L778
    #pragma HLS bind_op variable=v592 op=mul impl=fabric
    ap_int<17> v593 = v587;	// L779
    ap_int<17> v594 = v592;	// L780
    ap_int<17> v595 = v593 + v594;	// L781
    int8_t v596 = a2;	// L782
    int8_t v597 = c2_4;	// L783
    int16_t v598 = v596;	// L784
    int16_t v599 = v597;	// L785
    int16_t v600 = v598 * v599;	// L786
    #pragma HLS bind_op variable=v600 op=mul impl=fabric
    int8_t v601 = a3;	// L787
    int8_t v602 = c3_4;	// L788
    int16_t v603 = v601;	// L789
    int16_t v604 = v602;	// L790
    int16_t v605 = v603 * v604;	// L791
    #pragma HLS bind_op variable=v605 op=mul impl=fabric
    ap_int<17> v606 = v600;	// L792
    ap_int<17> v607 = v605;	// L793
    ap_int<17> v608 = v606 + v607;	// L794
    ap_int<18> v609 = v595;	// L795
    ap_int<18> v610 = v608;	// L796
    ap_int<18> v611 = v609 + v610;	// L797
    int8_t v612 = a4;	// L798
    int8_t v613 = c4_4;	// L799
    int16_t v614 = v612;	// L800
    int16_t v615 = v613;	// L801
    int16_t v616 = v614 * v615;	// L802
    #pragma HLS bind_op variable=v616 op=mul impl=fabric
    int8_t v617 = a5;	// L803
    int8_t v618 = c5_4;	// L804
    int16_t v619 = v617;	// L805
    int16_t v620 = v618;	// L806
    int16_t v621 = v619 * v620;	// L807
    #pragma HLS bind_op variable=v621 op=mul impl=fabric
    ap_int<17> v622 = v616;	// L808
    ap_int<17> v623 = v621;	// L809
    ap_int<17> v624 = v622 + v623;	// L810
    int8_t v625 = a6;	// L811
    int8_t v626 = c6_4;	// L812
    int16_t v627 = v625;	// L813
    int16_t v628 = v626;	// L814
    int16_t v629 = v627 * v628;	// L815
    #pragma HLS bind_op variable=v629 op=mul impl=fabric
    int8_t v630 = a7;	// L816
    int8_t v631 = c7_4;	// L817
    int16_t v632 = v630;	// L818
    int16_t v633 = v631;	// L819
    int16_t v634 = v632 * v633;	// L820
    #pragma HLS bind_op variable=v634 op=mul impl=fabric
    ap_int<17> v635 = v629;	// L821
    ap_int<17> v636 = v634;	// L822
    ap_int<17> v637 = v635 + v636;	// L823
    ap_int<18> v638 = v624;	// L824
    ap_int<18> v639 = v637;	// L825
    ap_int<18> v640 = v638 + v639;	// L826
    ap_int<19> v641 = v611;	// L827
    ap_int<19> v642 = v640;	// L828
    ap_int<19> v643 = v641 + v642;	// L829
    ap_int<33> v644 = v582;	// L830
    ap_int<33> v645 = v643;	// L831
    ap_int<33> v646 = v644 + v645;	// L832
    v17.write(v646);	// L833
    int32_t v647 = p5;	// L834
    int8_t v648 = a0;	// L835
    int8_t v649 = c0_5;	// L836
    int16_t v650 = v648;	// L837
    int16_t v651 = v649;	// L838
    int16_t v652 = v650 * v651;	// L839
    #pragma HLS bind_op variable=v652 op=mul impl=fabric
    int8_t v653 = a1;	// L840
    int8_t v654 = c1_5;	// L841
    int16_t v655 = v653;	// L842
    int16_t v656 = v654;	// L843
    int16_t v657 = v655 * v656;	// L844
    #pragma HLS bind_op variable=v657 op=mul impl=fabric
    ap_int<17> v658 = v652;	// L845
    ap_int<17> v659 = v657;	// L846
    ap_int<17> v660 = v658 + v659;	// L847
    int8_t v661 = a2;	// L848
    int8_t v662 = c2_5;	// L849
    int16_t v663 = v661;	// L850
    int16_t v664 = v662;	// L851
    int16_t v665 = v663 * v664;	// L852
    #pragma HLS bind_op variable=v665 op=mul impl=fabric
    int8_t v666 = a3;	// L853
    int8_t v667 = c3_5;	// L854
    int16_t v668 = v666;	// L855
    int16_t v669 = v667;	// L856
    int16_t v670 = v668 * v669;	// L857
    #pragma HLS bind_op variable=v670 op=mul impl=fabric
    ap_int<17> v671 = v665;	// L858
    ap_int<17> v672 = v670;	// L859
    ap_int<17> v673 = v671 + v672;	// L860
    ap_int<18> v674 = v660;	// L861
    ap_int<18> v675 = v673;	// L862
    ap_int<18> v676 = v674 + v675;	// L863
    int8_t v677 = a4;	// L864
    int8_t v678 = c4_5;	// L865
    int16_t v679 = v677;	// L866
    int16_t v680 = v678;	// L867
    int16_t v681 = v679 * v680;	// L868
    #pragma HLS bind_op variable=v681 op=mul impl=fabric
    int8_t v682 = a5;	// L869
    int8_t v683 = c5_5;	// L870
    int16_t v684 = v682;	// L871
    int16_t v685 = v683;	// L872
    int16_t v686 = v684 * v685;	// L873
    #pragma HLS bind_op variable=v686 op=mul impl=fabric
    ap_int<17> v687 = v681;	// L874
    ap_int<17> v688 = v686;	// L875
    ap_int<17> v689 = v687 + v688;	// L876
    int8_t v690 = a6;	// L877
    int8_t v691 = c6_5;	// L878
    int16_t v692 = v690;	// L879
    int16_t v693 = v691;	// L880
    int16_t v694 = v692 * v693;	// L881
    #pragma HLS bind_op variable=v694 op=mul impl=fabric
    int8_t v695 = a7;	// L882
    int8_t v696 = c7_5;	// L883
    int16_t v697 = v695;	// L884
    int16_t v698 = v696;	// L885
    int16_t v699 = v697 * v698;	// L886
    #pragma HLS bind_op variable=v699 op=mul impl=fabric
    ap_int<17> v700 = v694;	// L887
    ap_int<17> v701 = v699;	// L888
    ap_int<17> v702 = v700 + v701;	// L889
    ap_int<18> v703 = v689;	// L890
    ap_int<18> v704 = v702;	// L891
    ap_int<18> v705 = v703 + v704;	// L892
    ap_int<19> v706 = v676;	// L893
    ap_int<19> v707 = v705;	// L894
    ap_int<19> v708 = v706 + v707;	// L895
    ap_int<33> v709 = v647;	// L896
    ap_int<33> v710 = v708;	// L897
    ap_int<33> v711 = v709 + v710;	// L898
    v19.write(v711);	// L899
    int32_t v712 = p6;	// L900
    int8_t v713 = a0;	// L901
    int8_t v714 = c0_6;	// L902
    int16_t v715 = v713;	// L903
    int16_t v716 = v714;	// L904
    int16_t v717 = v715 * v716;	// L905
    #pragma HLS bind_op variable=v717 op=mul impl=fabric
    int8_t v718 = a1;	// L906
    int8_t v719 = c1_6;	// L907
    int16_t v720 = v718;	// L908
    int16_t v721 = v719;	// L909
    int16_t v722 = v720 * v721;	// L910
    #pragma HLS bind_op variable=v722 op=mul impl=fabric
    ap_int<17> v723 = v717;	// L911
    ap_int<17> v724 = v722;	// L912
    ap_int<17> v725 = v723 + v724;	// L913
    int8_t v726 = a2;	// L914
    int8_t v727 = c2_6;	// L915
    int16_t v728 = v726;	// L916
    int16_t v729 = v727;	// L917
    int16_t v730 = v728 * v729;	// L918
    #pragma HLS bind_op variable=v730 op=mul impl=fabric
    int8_t v731 = a3;	// L919
    int8_t v732 = c3_6;	// L920
    int16_t v733 = v731;	// L921
    int16_t v734 = v732;	// L922
    int16_t v735 = v733 * v734;	// L923
    #pragma HLS bind_op variable=v735 op=mul impl=fabric
    ap_int<17> v736 = v730;	// L924
    ap_int<17> v737 = v735;	// L925
    ap_int<17> v738 = v736 + v737;	// L926
    ap_int<18> v739 = v725;	// L927
    ap_int<18> v740 = v738;	// L928
    ap_int<18> v741 = v739 + v740;	// L929
    int8_t v742 = a4;	// L930
    int8_t v743 = c4_6;	// L931
    int16_t v744 = v742;	// L932
    int16_t v745 = v743;	// L933
    int16_t v746 = v744 * v745;	// L934
    #pragma HLS bind_op variable=v746 op=mul impl=fabric
    int8_t v747 = a5;	// L935
    int8_t v748 = c5_6;	// L936
    int16_t v749 = v747;	// L937
    int16_t v750 = v748;	// L938
    int16_t v751 = v749 * v750;	// L939
    #pragma HLS bind_op variable=v751 op=mul impl=fabric
    ap_int<17> v752 = v746;	// L940
    ap_int<17> v753 = v751;	// L941
    ap_int<17> v754 = v752 + v753;	// L942
    int8_t v755 = a6;	// L943
    int8_t v756 = c6_6;	// L944
    int16_t v757 = v755;	// L945
    int16_t v758 = v756;	// L946
    int16_t v759 = v757 * v758;	// L947
    #pragma HLS bind_op variable=v759 op=mul impl=fabric
    int8_t v760 = a7;	// L948
    int8_t v761 = c7_6;	// L949
    int16_t v762 = v760;	// L950
    int16_t v763 = v761;	// L951
    int16_t v764 = v762 * v763;	// L952
    #pragma HLS bind_op variable=v764 op=mul impl=fabric
    ap_int<17> v765 = v759;	// L953
    ap_int<17> v766 = v764;	// L954
    ap_int<17> v767 = v765 + v766;	// L955
    ap_int<18> v768 = v754;	// L956
    ap_int<18> v769 = v767;	// L957
    ap_int<18> v770 = v768 + v769;	// L958
    ap_int<19> v771 = v741;	// L959
    ap_int<19> v772 = v770;	// L960
    ap_int<19> v773 = v771 + v772;	// L961
    ap_int<33> v774 = v712;	// L962
    ap_int<33> v775 = v773;	// L963
    ap_int<33> v776 = v774 + v775;	// L964
    v21.write(v776);	// L965
    int32_t v777 = p7;	// L966
    int8_t v778 = a0;	// L967
    int8_t v779 = c0_7;	// L968
    int16_t v780 = v778;	// L969
    int16_t v781 = v779;	// L970
    int16_t v782 = v780 * v781;	// L971
    #pragma HLS bind_op variable=v782 op=mul impl=fabric
    int8_t v783 = a1;	// L972
    int8_t v784 = c1_7;	// L973
    int16_t v785 = v783;	// L974
    int16_t v786 = v784;	// L975
    int16_t v787 = v785 * v786;	// L976
    #pragma HLS bind_op variable=v787 op=mul impl=fabric
    ap_int<17> v788 = v782;	// L977
    ap_int<17> v789 = v787;	// L978
    ap_int<17> v790 = v788 + v789;	// L979
    int8_t v791 = a2;	// L980
    int8_t v792 = c2_7;	// L981
    int16_t v793 = v791;	// L982
    int16_t v794 = v792;	// L983
    int16_t v795 = v793 * v794;	// L984
    #pragma HLS bind_op variable=v795 op=mul impl=fabric
    int8_t v796 = a3;	// L985
    int8_t v797 = c3_7;	// L986
    int16_t v798 = v796;	// L987
    int16_t v799 = v797;	// L988
    int16_t v800 = v798 * v799;	// L989
    #pragma HLS bind_op variable=v800 op=mul impl=fabric
    ap_int<17> v801 = v795;	// L990
    ap_int<17> v802 = v800;	// L991
    ap_int<17> v803 = v801 + v802;	// L992
    ap_int<18> v804 = v790;	// L993
    ap_int<18> v805 = v803;	// L994
    ap_int<18> v806 = v804 + v805;	// L995
    int8_t v807 = a4;	// L996
    int8_t v808 = c4_7;	// L997
    int16_t v809 = v807;	// L998
    int16_t v810 = v808;	// L999
    int16_t v811 = v809 * v810;	// L1000
    #pragma HLS bind_op variable=v811 op=mul impl=fabric
    int8_t v812 = a5;	// L1001
    int8_t v813 = c5_7;	// L1002
    int16_t v814 = v812;	// L1003
    int16_t v815 = v813;	// L1004
    int16_t v816 = v814 * v815;	// L1005
    #pragma HLS bind_op variable=v816 op=mul impl=fabric
    ap_int<17> v817 = v811;	// L1006
    ap_int<17> v818 = v816;	// L1007
    ap_int<17> v819 = v817 + v818;	// L1008
    int8_t v820 = a6;	// L1009
    int8_t v821 = c6_7;	// L1010
    int16_t v822 = v820;	// L1011
    int16_t v823 = v821;	// L1012
    int16_t v824 = v822 * v823;	// L1013
    #pragma HLS bind_op variable=v824 op=mul impl=fabric
    int8_t v825 = a7;	// L1014
    int8_t v826 = c7_7;	// L1015
    int16_t v827 = v825;	// L1016
    int16_t v828 = v826;	// L1017
    int16_t v829 = v827 * v828;	// L1018
    #pragma HLS bind_op variable=v829 op=mul impl=fabric
    ap_int<17> v830 = v824;	// L1019
    ap_int<17> v831 = v829;	// L1020
    ap_int<17> v832 = v830 + v831;	// L1021
    ap_int<18> v833 = v819;	// L1022
    ap_int<18> v834 = v832;	// L1023
    ap_int<18> v835 = v833 + v834;	// L1024
    ap_int<19> v836 = v806;	// L1025
    ap_int<19> v837 = v835;	// L1026
    ap_int<19> v838 = v836 + v837;	// L1027
    ap_int<33> v839 = v777;	// L1028
    ap_int<33> v840 = v838;	// L1029
    ap_int<33> v841 = v839 + v840;	// L1030
    v23.write(v841);	// L1031
    int8_t v842 = n0_6;	// L1032
    n0_7 = v842;	// L1033
    int8_t v843 = n0_5;	// L1034
    n0_6 = v843;	// L1035
    int8_t v844 = n0_4;	// L1036
    n0_5 = v844;	// L1037
    int8_t v845 = n0_3;	// L1038
    n0_4 = v845;	// L1039
    int8_t v846 = n0_2;	// L1040
    n0_3 = v846;	// L1041
    int8_t v847 = n0_1;	// L1042
    n0_2 = v847;	// L1043
    int8_t v848 = n0_0;	// L1044
    n0_1 = v848;	// L1045
    int8_t v849 = v24.read();	// L1046
    n0_0 = v849;	// L1047
    int8_t v850 = n1_6;	// L1048
    n1_7 = v850;	// L1049
    int8_t v851 = n1_5;	// L1050
    n1_6 = v851;	// L1051
    int8_t v852 = n1_4;	// L1052
    n1_5 = v852;	// L1053
    int8_t v853 = n1_3;	// L1054
    n1_4 = v853;	// L1055
    int8_t v854 = n1_2;	// L1056
    n1_3 = v854;	// L1057
    int8_t v855 = n1_1;	// L1058
    n1_2 = v855;	// L1059
    int8_t v856 = n1_0;	// L1060
    n1_1 = v856;	// L1061
    int8_t v857 = v25.read();	// L1062
    n1_0 = v857;	// L1063
    int8_t v858 = n2_6;	// L1064
    n2_7 = v858;	// L1065
    int8_t v859 = n2_5;	// L1066
    n2_6 = v859;	// L1067
    int8_t v860 = n2_4;	// L1068
    n2_5 = v860;	// L1069
    int8_t v861 = n2_3;	// L1070
    n2_4 = v861;	// L1071
    int8_t v862 = n2_2;	// L1072
    n2_3 = v862;	// L1073
    int8_t v863 = n2_1;	// L1074
    n2_2 = v863;	// L1075
    int8_t v864 = n2_0;	// L1076
    n2_1 = v864;	// L1077
    int8_t v865 = v26.read();	// L1078
    n2_0 = v865;	// L1079
    int8_t v866 = n3_6;	// L1080
    n3_7 = v866;	// L1081
    int8_t v867 = n3_5;	// L1082
    n3_6 = v867;	// L1083
    int8_t v868 = n3_4;	// L1084
    n3_5 = v868;	// L1085
    int8_t v869 = n3_3;	// L1086
    n3_4 = v869;	// L1087
    int8_t v870 = n3_2;	// L1088
    n3_3 = v870;	// L1089
    int8_t v871 = n3_1;	// L1090
    n3_2 = v871;	// L1091
    int8_t v872 = n3_0;	// L1092
    n3_1 = v872;	// L1093
    int8_t v873 = v27.read();	// L1094
    n3_0 = v873;	// L1095
    int8_t v874 = n4_6;	// L1096
    n4_7 = v874;	// L1097
    int8_t v875 = n4_5;	// L1098
    n4_6 = v875;	// L1099
    int8_t v876 = n4_4;	// L1100
    n4_5 = v876;	// L1101
    int8_t v877 = n4_3;	// L1102
    n4_4 = v877;	// L1103
    int8_t v878 = n4_2;	// L1104
    n4_3 = v878;	// L1105
    int8_t v879 = n4_1;	// L1106
    n4_2 = v879;	// L1107
    int8_t v880 = n4_0;	// L1108
    n4_1 = v880;	// L1109
    int8_t v881 = v28.read();	// L1110
    n4_0 = v881;	// L1111
    int8_t v882 = n5_6;	// L1112
    n5_7 = v882;	// L1113
    int8_t v883 = n5_5;	// L1114
    n5_6 = v883;	// L1115
    int8_t v884 = n5_4;	// L1116
    n5_5 = v884;	// L1117
    int8_t v885 = n5_3;	// L1118
    n5_4 = v885;	// L1119
    int8_t v886 = n5_2;	// L1120
    n5_3 = v886;	// L1121
    int8_t v887 = n5_1;	// L1122
    n5_2 = v887;	// L1123
    int8_t v888 = n5_0;	// L1124
    n5_1 = v888;	// L1125
    int8_t v889 = v29.read();	// L1126
    n5_0 = v889;	// L1127
    int8_t v890 = n6_6;	// L1128
    n6_7 = v890;	// L1129
    int8_t v891 = n6_5;	// L1130
    n6_6 = v891;	// L1131
    int8_t v892 = n6_4;	// L1132
    n6_5 = v892;	// L1133
    int8_t v893 = n6_3;	// L1134
    n6_4 = v893;	// L1135
    int8_t v894 = n6_2;	// L1136
    n6_3 = v894;	// L1137
    int8_t v895 = n6_1;	// L1138
    n6_2 = v895;	// L1139
    int8_t v896 = n6_0;	// L1140
    n6_1 = v896;	// L1141
    int8_t v897 = v30.read();	// L1142
    n6_0 = v897;	// L1143
    int8_t v898 = n7_6;	// L1144
    n7_7 = v898;	// L1145
    int8_t v899 = n7_5;	// L1146
    n7_6 = v899;	// L1147
    int8_t v900 = n7_4;	// L1148
    n7_5 = v900;	// L1149
    int8_t v901 = n7_3;	// L1150
    n7_4 = v901;	// L1151
    int8_t v902 = n7_2;	// L1152
    n7_3 = v902;	// L1153
    int8_t v903 = n7_1;	// L1154
    n7_2 = v903;	// L1155
    int8_t v904 = n7_0;	// L1156
    n7_1 = v904;	// L1157
    int8_t v905 = v31.read();	// L1158
    n7_0 = v905;	// L1159
    int32_t v906 = s;	// L1160
    int32_t v907 = v906 & 15;	// L1162
    bool v908 = v907 == 15;	// L1163
    if (v908) {	// L1164
      int8_t v909 = n0_0;	// L1165
      c0_0 = v909;	// L1166
      int8_t v910 = n0_1;	// L1167
      c0_1 = v910;	// L1168
      int8_t v911 = n0_2;	// L1169
      c0_2 = v911;	// L1170
      int8_t v912 = n0_3;	// L1171
      c0_3 = v912;	// L1172
      int8_t v913 = n0_4;	// L1173
      c0_4 = v913;	// L1174
      int8_t v914 = n0_5;	// L1175
      c0_5 = v914;	// L1176
      int8_t v915 = n0_6;	// L1177
      c0_6 = v915;	// L1178
      int8_t v916 = n0_7;	// L1179
      c0_7 = v916;	// L1180
      int8_t v917 = n1_0;	// L1181
      c1_0 = v917;	// L1182
      int8_t v918 = n1_1;	// L1183
      c1_1 = v918;	// L1184
      int8_t v919 = n1_2;	// L1185
      c1_2 = v919;	// L1186
      int8_t v920 = n1_3;	// L1187
      c1_3 = v920;	// L1188
      int8_t v921 = n1_4;	// L1189
      c1_4 = v921;	// L1190
      int8_t v922 = n1_5;	// L1191
      c1_5 = v922;	// L1192
      int8_t v923 = n1_6;	// L1193
      c1_6 = v923;	// L1194
      int8_t v924 = n1_7;	// L1195
      c1_7 = v924;	// L1196
      int8_t v925 = n2_0;	// L1197
      c2_0 = v925;	// L1198
      int8_t v926 = n2_1;	// L1199
      c2_1 = v926;	// L1200
      int8_t v927 = n2_2;	// L1201
      c2_2 = v927;	// L1202
      int8_t v928 = n2_3;	// L1203
      c2_3 = v928;	// L1204
      int8_t v929 = n2_4;	// L1205
      c2_4 = v929;	// L1206
      int8_t v930 = n2_5;	// L1207
      c2_5 = v930;	// L1208
      int8_t v931 = n2_6;	// L1209
      c2_6 = v931;	// L1210
      int8_t v932 = n2_7;	// L1211
      c2_7 = v932;	// L1212
      int8_t v933 = n3_0;	// L1213
      c3_0 = v933;	// L1214
      int8_t v934 = n3_1;	// L1215
      c3_1 = v934;	// L1216
      int8_t v935 = n3_2;	// L1217
      c3_2 = v935;	// L1218
      int8_t v936 = n3_3;	// L1219
      c3_3 = v936;	// L1220
      int8_t v937 = n3_4;	// L1221
      c3_4 = v937;	// L1222
      int8_t v938 = n3_5;	// L1223
      c3_5 = v938;	// L1224
      int8_t v939 = n3_6;	// L1225
      c3_6 = v939;	// L1226
      int8_t v940 = n3_7;	// L1227
      c3_7 = v940;	// L1228
      int8_t v941 = n4_0;	// L1229
      c4_0 = v941;	// L1230
      int8_t v942 = n4_1;	// L1231
      c4_1 = v942;	// L1232
      int8_t v943 = n4_2;	// L1233
      c4_2 = v943;	// L1234
      int8_t v944 = n4_3;	// L1235
      c4_3 = v944;	// L1236
      int8_t v945 = n4_4;	// L1237
      c4_4 = v945;	// L1238
      int8_t v946 = n4_5;	// L1239
      c4_5 = v946;	// L1240
      int8_t v947 = n4_6;	// L1241
      c4_6 = v947;	// L1242
      int8_t v948 = n4_7;	// L1243
      c4_7 = v948;	// L1244
      int8_t v949 = n5_0;	// L1245
      c5_0 = v949;	// L1246
      int8_t v950 = n5_1;	// L1247
      c5_1 = v950;	// L1248
      int8_t v951 = n5_2;	// L1249
      c5_2 = v951;	// L1250
      int8_t v952 = n5_3;	// L1251
      c5_3 = v952;	// L1252
      int8_t v953 = n5_4;	// L1253
      c5_4 = v953;	// L1254
      int8_t v954 = n5_5;	// L1255
      c5_5 = v954;	// L1256
      int8_t v955 = n5_6;	// L1257
      c5_6 = v955;	// L1258
      int8_t v956 = n5_7;	// L1259
      c5_7 = v956;	// L1260
      int8_t v957 = n6_0;	// L1261
      c6_0 = v957;	// L1262
      int8_t v958 = n6_1;	// L1263
      c6_1 = v958;	// L1264
      int8_t v959 = n6_2;	// L1265
      c6_2 = v959;	// L1266
      int8_t v960 = n6_3;	// L1267
      c6_3 = v960;	// L1268
      int8_t v961 = n6_4;	// L1269
      c6_4 = v961;	// L1270
      int8_t v962 = n6_5;	// L1271
      c6_5 = v962;	// L1272
      int8_t v963 = n6_6;	// L1273
      c6_6 = v963;	// L1274
      int8_t v964 = n6_7;	// L1275
      c6_7 = v964;	// L1276
      int8_t v965 = n7_0;	// L1277
      c7_0 = v965;	// L1278
      int8_t v966 = n7_1;	// L1279
      c7_1 = v966;	// L1280
      int8_t v967 = n7_2;	// L1281
      c7_2 = v967;	// L1282
      int8_t v968 = n7_3;	// L1283
      c7_3 = v968;	// L1284
      int8_t v969 = n7_4;	// L1285
      c7_4 = v969;	// L1286
      int8_t v970 = n7_5;	// L1287
      c7_5 = v970;	// L1288
      int8_t v971 = n7_6;	// L1289
      c7_6 = v971;	// L1290
      int8_t v972 = n7_7;	// L1291
      c7_7 = v972;	// L1292
    }
  }
}


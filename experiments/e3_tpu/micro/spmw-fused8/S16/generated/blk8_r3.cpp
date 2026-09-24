
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
void blk8_r3_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int8_t >& v5,
  hls::stream< int8_t >& v6,
  hls::stream< int8_t >& v7,
  hls::stream< int8_t >& v8,
  hls::stream< int8_t >& v9,
  hls::stream< int8_t >& v10,
  hls::stream< int8_t >& v11,
  hls::stream< int8_t >& v12,
  hls::stream< int8_t >& v13,
  hls::stream< int8_t >& v14,
  hls::stream< int8_t >& v15,
  hls::stream< int32_t >& v16,
  hls::stream< int32_t >& v17,
  hls::stream< int32_t >& v18,
  hls::stream< int32_t >& v19,
  hls::stream< int32_t >& v20,
  hls::stream< int32_t >& v21,
  hls::stream< int32_t >& v22,
  hls::stream< int32_t >& v23,
  hls::stream< int32_t >& v24,
  hls::stream< int32_t >& v25,
  hls::stream< int32_t >& v26,
  hls::stream< int32_t >& v27,
  hls::stream< int32_t >& v28,
  hls::stream< int32_t >& v29,
  hls::stream< int32_t >& v30,
  hls::stream< int32_t >& v31,
  hls::stream< int8_t >& v32,
  hls::stream< int8_t >& v33,
  hls::stream< int8_t >& v34,
  hls::stream< int8_t >& v35,
  hls::stream< int8_t >& v36,
  hls::stream< int8_t >& v37,
  hls::stream< int8_t >& v38,
  hls::stream< int8_t >& v39,
  hls::stream< int8_t >& v40,
  hls::stream< int8_t >& v41,
  hls::stream< int8_t >& v42,
  hls::stream< int8_t >& v43,
  hls::stream< int8_t >& v44,
  hls::stream< int8_t >& v45,
  hls::stream< int8_t >& v46,
  hls::stream< int8_t >& v47
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
    int8_t v113 = n0_7;	// L134
    v33.write(v113);	// L135
    int8_t v114 = n0_6;	// L136
    n0_7 = v114;	// L137
    int8_t v115 = n0_5;	// L138
    n0_6 = v115;	// L139
    int8_t v116 = n0_4;	// L140
    n0_5 = v116;	// L141
    int8_t v117 = n0_3;	// L142
    n0_4 = v117;	// L143
    int8_t v118 = n0_2;	// L144
    n0_3 = v118;	// L145
    int8_t v119 = n0_1;	// L146
    n0_2 = v119;	// L147
    int8_t v120 = n0_0;	// L148
    n0_1 = v120;	// L149
    int8_t v121 = v32.read();	// L150
    n0_0 = v121;	// L151
    int8_t v122 = n1_7;	// L152
    v35.write(v122);	// L153
    int8_t v123 = n1_6;	// L154
    n1_7 = v123;	// L155
    int8_t v124 = n1_5;	// L156
    n1_6 = v124;	// L157
    int8_t v125 = n1_4;	// L158
    n1_5 = v125;	// L159
    int8_t v126 = n1_3;	// L160
    n1_4 = v126;	// L161
    int8_t v127 = n1_2;	// L162
    n1_3 = v127;	// L163
    int8_t v128 = n1_1;	// L164
    n1_2 = v128;	// L165
    int8_t v129 = n1_0;	// L166
    n1_1 = v129;	// L167
    int8_t v130 = v34.read();	// L168
    n1_0 = v130;	// L169
    int8_t v131 = n2_7;	// L170
    v37.write(v131);	// L171
    int8_t v132 = n2_6;	// L172
    n2_7 = v132;	// L173
    int8_t v133 = n2_5;	// L174
    n2_6 = v133;	// L175
    int8_t v134 = n2_4;	// L176
    n2_5 = v134;	// L177
    int8_t v135 = n2_3;	// L178
    n2_4 = v135;	// L179
    int8_t v136 = n2_2;	// L180
    n2_3 = v136;	// L181
    int8_t v137 = n2_1;	// L182
    n2_2 = v137;	// L183
    int8_t v138 = n2_0;	// L184
    n2_1 = v138;	// L185
    int8_t v139 = v36.read();	// L186
    n2_0 = v139;	// L187
    int8_t v140 = n3_7;	// L188
    v39.write(v140);	// L189
    int8_t v141 = n3_6;	// L190
    n3_7 = v141;	// L191
    int8_t v142 = n3_5;	// L192
    n3_6 = v142;	// L193
    int8_t v143 = n3_4;	// L194
    n3_5 = v143;	// L195
    int8_t v144 = n3_3;	// L196
    n3_4 = v144;	// L197
    int8_t v145 = n3_2;	// L198
    n3_3 = v145;	// L199
    int8_t v146 = n3_1;	// L200
    n3_2 = v146;	// L201
    int8_t v147 = n3_0;	// L202
    n3_1 = v147;	// L203
    int8_t v148 = v38.read();	// L204
    n3_0 = v148;	// L205
    int8_t v149 = n4_7;	// L206
    v41.write(v149);	// L207
    int8_t v150 = n4_6;	// L208
    n4_7 = v150;	// L209
    int8_t v151 = n4_5;	// L210
    n4_6 = v151;	// L211
    int8_t v152 = n4_4;	// L212
    n4_5 = v152;	// L213
    int8_t v153 = n4_3;	// L214
    n4_4 = v153;	// L215
    int8_t v154 = n4_2;	// L216
    n4_3 = v154;	// L217
    int8_t v155 = n4_1;	// L218
    n4_2 = v155;	// L219
    int8_t v156 = n4_0;	// L220
    n4_1 = v156;	// L221
    int8_t v157 = v40.read();	// L222
    n4_0 = v157;	// L223
    int8_t v158 = n5_7;	// L224
    v43.write(v158);	// L225
    int8_t v159 = n5_6;	// L226
    n5_7 = v159;	// L227
    int8_t v160 = n5_5;	// L228
    n5_6 = v160;	// L229
    int8_t v161 = n5_4;	// L230
    n5_5 = v161;	// L231
    int8_t v162 = n5_3;	// L232
    n5_4 = v162;	// L233
    int8_t v163 = n5_2;	// L234
    n5_3 = v163;	// L235
    int8_t v164 = n5_1;	// L236
    n5_2 = v164;	// L237
    int8_t v165 = n5_0;	// L238
    n5_1 = v165;	// L239
    int8_t v166 = v42.read();	// L240
    n5_0 = v166;	// L241
    int8_t v167 = n6_7;	// L242
    v45.write(v167);	// L243
    int8_t v168 = n6_6;	// L244
    n6_7 = v168;	// L245
    int8_t v169 = n6_5;	// L246
    n6_6 = v169;	// L247
    int8_t v170 = n6_4;	// L248
    n6_5 = v170;	// L249
    int8_t v171 = n6_3;	// L250
    n6_4 = v171;	// L251
    int8_t v172 = n6_2;	// L252
    n6_3 = v172;	// L253
    int8_t v173 = n6_1;	// L254
    n6_2 = v173;	// L255
    int8_t v174 = n6_0;	// L256
    n6_1 = v174;	// L257
    int8_t v175 = v44.read();	// L258
    n6_0 = v175;	// L259
    int8_t v176 = n7_7;	// L260
    v47.write(v176);	// L261
    int8_t v177 = n7_6;	// L262
    n7_7 = v177;	// L263
    int8_t v178 = n7_5;	// L264
    n7_6 = v178;	// L265
    int8_t v179 = n7_4;	// L266
    n7_5 = v179;	// L267
    int8_t v180 = n7_3;	// L268
    n7_4 = v180;	// L269
    int8_t v181 = n7_2;	// L270
    n7_3 = v181;	// L271
    int8_t v182 = n7_1;	// L272
    n7_2 = v182;	// L273
    int8_t v183 = n7_0;	// L274
    n7_1 = v183;	// L275
    int8_t v184 = v46.read();	// L276
    n7_0 = v184;	// L277
  }
  int8_t v185 = n0_0;	// L279
  int8_t c0_0;	// L280
  c0_0 = v185;	// L281
  int8_t v187 = n0_1;	// L282
  int8_t c0_1;	// L283
  c0_1 = v187;	// L284
  int8_t v189 = n0_2;	// L285
  int8_t c0_2;	// L286
  c0_2 = v189;	// L287
  int8_t v191 = n0_3;	// L288
  int8_t c0_3;	// L289
  c0_3 = v191;	// L290
  int8_t v193 = n0_4;	// L291
  int8_t c0_4;	// L292
  c0_4 = v193;	// L293
  int8_t v195 = n0_5;	// L294
  int8_t c0_5;	// L295
  c0_5 = v195;	// L296
  int8_t v197 = n0_6;	// L297
  int8_t c0_6;	// L298
  c0_6 = v197;	// L299
  int8_t v199 = n0_7;	// L300
  int8_t c0_7;	// L301
  c0_7 = v199;	// L302
  int8_t v201 = n1_0;	// L303
  int8_t c1_0;	// L304
  c1_0 = v201;	// L305
  int8_t v203 = n1_1;	// L306
  int8_t c1_1;	// L307
  c1_1 = v203;	// L308
  int8_t v205 = n1_2;	// L309
  int8_t c1_2;	// L310
  c1_2 = v205;	// L311
  int8_t v207 = n1_3;	// L312
  int8_t c1_3;	// L313
  c1_3 = v207;	// L314
  int8_t v209 = n1_4;	// L315
  int8_t c1_4;	// L316
  c1_4 = v209;	// L317
  int8_t v211 = n1_5;	// L318
  int8_t c1_5;	// L319
  c1_5 = v211;	// L320
  int8_t v213 = n1_6;	// L321
  int8_t c1_6;	// L322
  c1_6 = v213;	// L323
  int8_t v215 = n1_7;	// L324
  int8_t c1_7;	// L325
  c1_7 = v215;	// L326
  int8_t v217 = n2_0;	// L327
  int8_t c2_0;	// L328
  c2_0 = v217;	// L329
  int8_t v219 = n2_1;	// L330
  int8_t c2_1;	// L331
  c2_1 = v219;	// L332
  int8_t v221 = n2_2;	// L333
  int8_t c2_2;	// L334
  c2_2 = v221;	// L335
  int8_t v223 = n2_3;	// L336
  int8_t c2_3;	// L337
  c2_3 = v223;	// L338
  int8_t v225 = n2_4;	// L339
  int8_t c2_4;	// L340
  c2_4 = v225;	// L341
  int8_t v227 = n2_5;	// L342
  int8_t c2_5;	// L343
  c2_5 = v227;	// L344
  int8_t v229 = n2_6;	// L345
  int8_t c2_6;	// L346
  c2_6 = v229;	// L347
  int8_t v231 = n2_7;	// L348
  int8_t c2_7;	// L349
  c2_7 = v231;	// L350
  int8_t v233 = n3_0;	// L351
  int8_t c3_0;	// L352
  c3_0 = v233;	// L353
  int8_t v235 = n3_1;	// L354
  int8_t c3_1;	// L355
  c3_1 = v235;	// L356
  int8_t v237 = n3_2;	// L357
  int8_t c3_2;	// L358
  c3_2 = v237;	// L359
  int8_t v239 = n3_3;	// L360
  int8_t c3_3;	// L361
  c3_3 = v239;	// L362
  int8_t v241 = n3_4;	// L363
  int8_t c3_4;	// L364
  c3_4 = v241;	// L365
  int8_t v243 = n3_5;	// L366
  int8_t c3_5;	// L367
  c3_5 = v243;	// L368
  int8_t v245 = n3_6;	// L369
  int8_t c3_6;	// L370
  c3_6 = v245;	// L371
  int8_t v247 = n3_7;	// L372
  int8_t c3_7;	// L373
  c3_7 = v247;	// L374
  int8_t v249 = n4_0;	// L375
  int8_t c4_0;	// L376
  c4_0 = v249;	// L377
  int8_t v251 = n4_1;	// L378
  int8_t c4_1;	// L379
  c4_1 = v251;	// L380
  int8_t v253 = n4_2;	// L381
  int8_t c4_2;	// L382
  c4_2 = v253;	// L383
  int8_t v255 = n4_3;	// L384
  int8_t c4_3;	// L385
  c4_3 = v255;	// L386
  int8_t v257 = n4_4;	// L387
  int8_t c4_4;	// L388
  c4_4 = v257;	// L389
  int8_t v259 = n4_5;	// L390
  int8_t c4_5;	// L391
  c4_5 = v259;	// L392
  int8_t v261 = n4_6;	// L393
  int8_t c4_6;	// L394
  c4_6 = v261;	// L395
  int8_t v263 = n4_7;	// L396
  int8_t c4_7;	// L397
  c4_7 = v263;	// L398
  int8_t v265 = n5_0;	// L399
  int8_t c5_0;	// L400
  c5_0 = v265;	// L401
  int8_t v267 = n5_1;	// L402
  int8_t c5_1;	// L403
  c5_1 = v267;	// L404
  int8_t v269 = n5_2;	// L405
  int8_t c5_2;	// L406
  c5_2 = v269;	// L407
  int8_t v271 = n5_3;	// L408
  int8_t c5_3;	// L409
  c5_3 = v271;	// L410
  int8_t v273 = n5_4;	// L411
  int8_t c5_4;	// L412
  c5_4 = v273;	// L413
  int8_t v275 = n5_5;	// L414
  int8_t c5_5;	// L415
  c5_5 = v275;	// L416
  int8_t v277 = n5_6;	// L417
  int8_t c5_6;	// L418
  c5_6 = v277;	// L419
  int8_t v279 = n5_7;	// L420
  int8_t c5_7;	// L421
  c5_7 = v279;	// L422
  int8_t v281 = n6_0;	// L423
  int8_t c6_0;	// L424
  c6_0 = v281;	// L425
  int8_t v283 = n6_1;	// L426
  int8_t c6_1;	// L427
  c6_1 = v283;	// L428
  int8_t v285 = n6_2;	// L429
  int8_t c6_2;	// L430
  c6_2 = v285;	// L431
  int8_t v287 = n6_3;	// L432
  int8_t c6_3;	// L433
  c6_3 = v287;	// L434
  int8_t v289 = n6_4;	// L435
  int8_t c6_4;	// L436
  c6_4 = v289;	// L437
  int8_t v291 = n6_5;	// L438
  int8_t c6_5;	// L439
  c6_5 = v291;	// L440
  int8_t v293 = n6_6;	// L441
  int8_t c6_6;	// L442
  c6_6 = v293;	// L443
  int8_t v295 = n6_7;	// L444
  int8_t c6_7;	// L445
  c6_7 = v295;	// L446
  int8_t v297 = n7_0;	// L447
  int8_t c7_0;	// L448
  c7_0 = v297;	// L449
  int8_t v299 = n7_1;	// L450
  int8_t c7_1;	// L451
  c7_1 = v299;	// L452
  int8_t v301 = n7_2;	// L453
  int8_t c7_2;	// L454
  c7_2 = v301;	// L455
  int8_t v303 = n7_3;	// L456
  int8_t c7_3;	// L457
  c7_3 = v303;	// L458
  int8_t v305 = n7_4;	// L459
  int8_t c7_4;	// L460
  c7_4 = v305;	// L461
  int8_t v307 = n7_5;	// L462
  int8_t c7_5;	// L463
  c7_5 = v307;	// L464
  int8_t v309 = n7_6;	// L465
  int8_t c7_6;	// L466
  c7_6 = v309;	// L467
  int8_t v311 = n7_7;	// L468
  int8_t c7_7;	// L469
  c7_7 = v311;	// L470
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L471
  #pragma HLS pipeline II=1
    int8_t v314 = v0.read();	// L472
    int8_t a0;	// L473
    a0 = v314;	// L474
    int8_t v316 = v2.read();	// L475
    int8_t a1;	// L476
    a1 = v316;	// L477
    int8_t v318 = v4.read();	// L478
    int8_t a2;	// L479
    a2 = v318;	// L480
    int8_t v320 = v6.read();	// L481
    int8_t a3;	// L482
    a3 = v320;	// L483
    int8_t v322 = v8.read();	// L484
    int8_t a4;	// L485
    a4 = v322;	// L486
    int8_t v324 = v10.read();	// L487
    int8_t a5;	// L488
    a5 = v324;	// L489
    int8_t v326 = v12.read();	// L490
    int8_t a6;	// L491
    a6 = v326;	// L492
    int8_t v328 = v14.read();	// L493
    int8_t a7;	// L494
    a7 = v328;	// L495
    int32_t v330 = v16.read();	// L496
    int32_t p0;	// L497
    p0 = v330;	// L498
    int32_t v332 = v18.read();	// L499
    int32_t p1;	// L500
    p1 = v332;	// L501
    int32_t v334 = v20.read();	// L502
    int32_t p2;	// L503
    p2 = v334;	// L504
    int32_t v336 = v22.read();	// L505
    int32_t p3;	// L506
    p3 = v336;	// L507
    int32_t v338 = v24.read();	// L508
    int32_t p4;	// L509
    p4 = v338;	// L510
    int32_t v340 = v26.read();	// L511
    int32_t p5;	// L512
    p5 = v340;	// L513
    int32_t v342 = v28.read();	// L514
    int32_t p6;	// L515
    p6 = v342;	// L516
    int32_t v344 = v30.read();	// L517
    int32_t p7;	// L518
    p7 = v344;	// L519
    int8_t v346 = a0;	// L520
    v1.write(v346);	// L521
    int8_t v347 = a1;	// L522
    v3.write(v347);	// L523
    int8_t v348 = a2;	// L524
    v5.write(v348);	// L525
    int8_t v349 = a3;	// L526
    v7.write(v349);	// L527
    int8_t v350 = a4;	// L528
    v9.write(v350);	// L529
    int8_t v351 = a5;	// L530
    v11.write(v351);	// L531
    int8_t v352 = a6;	// L532
    v13.write(v352);	// L533
    int8_t v353 = a7;	// L534
    v15.write(v353);	// L535
    int32_t v354 = p0;	// L536
    int8_t v355 = a0;	// L537
    int8_t v356 = c0_0;	// L538
    int16_t v357 = v355;	// L539
    int16_t v358 = v356;	// L540
    int16_t v359 = v357 * v358;	// L541
    #pragma HLS bind_op variable=v359 op=mul impl=fabric
    int8_t v360 = a1;	// L542
    int8_t v361 = c1_0;	// L543
    int16_t v362 = v360;	// L544
    int16_t v363 = v361;	// L545
    int16_t v364 = v362 * v363;	// L546
    #pragma HLS bind_op variable=v364 op=mul impl=fabric
    ap_int<17> v365 = v359;	// L547
    ap_int<17> v366 = v364;	// L548
    ap_int<17> v367 = v365 + v366;	// L549
    int8_t v368 = a2;	// L550
    int8_t v369 = c2_0;	// L551
    int16_t v370 = v368;	// L552
    int16_t v371 = v369;	// L553
    int16_t v372 = v370 * v371;	// L554
    #pragma HLS bind_op variable=v372 op=mul impl=fabric
    int8_t v373 = a3;	// L555
    int8_t v374 = c3_0;	// L556
    int16_t v375 = v373;	// L557
    int16_t v376 = v374;	// L558
    int16_t v377 = v375 * v376;	// L559
    #pragma HLS bind_op variable=v377 op=mul impl=fabric
    ap_int<17> v378 = v372;	// L560
    ap_int<17> v379 = v377;	// L561
    ap_int<17> v380 = v378 + v379;	// L562
    ap_int<18> v381 = v367;	// L563
    ap_int<18> v382 = v380;	// L564
    ap_int<18> v383 = v381 + v382;	// L565
    int8_t v384 = a4;	// L566
    int8_t v385 = c4_0;	// L567
    int16_t v386 = v384;	// L568
    int16_t v387 = v385;	// L569
    int16_t v388 = v386 * v387;	// L570
    #pragma HLS bind_op variable=v388 op=mul impl=fabric
    int8_t v389 = a5;	// L571
    int8_t v390 = c5_0;	// L572
    int16_t v391 = v389;	// L573
    int16_t v392 = v390;	// L574
    int16_t v393 = v391 * v392;	// L575
    #pragma HLS bind_op variable=v393 op=mul impl=fabric
    ap_int<17> v394 = v388;	// L576
    ap_int<17> v395 = v393;	// L577
    ap_int<17> v396 = v394 + v395;	// L578
    int8_t v397 = a6;	// L579
    int8_t v398 = c6_0;	// L580
    int16_t v399 = v397;	// L581
    int16_t v400 = v398;	// L582
    int16_t v401 = v399 * v400;	// L583
    #pragma HLS bind_op variable=v401 op=mul impl=fabric
    int8_t v402 = a7;	// L584
    int8_t v403 = c7_0;	// L585
    int16_t v404 = v402;	// L586
    int16_t v405 = v403;	// L587
    int16_t v406 = v404 * v405;	// L588
    #pragma HLS bind_op variable=v406 op=mul impl=fabric
    ap_int<17> v407 = v401;	// L589
    ap_int<17> v408 = v406;	// L590
    ap_int<17> v409 = v407 + v408;	// L591
    ap_int<18> v410 = v396;	// L592
    ap_int<18> v411 = v409;	// L593
    ap_int<18> v412 = v410 + v411;	// L594
    ap_int<19> v413 = v383;	// L595
    ap_int<19> v414 = v412;	// L596
    ap_int<19> v415 = v413 + v414;	// L597
    ap_int<33> v416 = v354;	// L598
    ap_int<33> v417 = v415;	// L599
    ap_int<33> v418 = v416 + v417;	// L600
    v17.write(v418);	// L601
    int32_t v419 = p1;	// L602
    int8_t v420 = a0;	// L603
    int8_t v421 = c0_1;	// L604
    int16_t v422 = v420;	// L605
    int16_t v423 = v421;	// L606
    int16_t v424 = v422 * v423;	// L607
    #pragma HLS bind_op variable=v424 op=mul impl=fabric
    int8_t v425 = a1;	// L608
    int8_t v426 = c1_1;	// L609
    int16_t v427 = v425;	// L610
    int16_t v428 = v426;	// L611
    int16_t v429 = v427 * v428;	// L612
    #pragma HLS bind_op variable=v429 op=mul impl=fabric
    ap_int<17> v430 = v424;	// L613
    ap_int<17> v431 = v429;	// L614
    ap_int<17> v432 = v430 + v431;	// L615
    int8_t v433 = a2;	// L616
    int8_t v434 = c2_1;	// L617
    int16_t v435 = v433;	// L618
    int16_t v436 = v434;	// L619
    int16_t v437 = v435 * v436;	// L620
    #pragma HLS bind_op variable=v437 op=mul impl=fabric
    int8_t v438 = a3;	// L621
    int8_t v439 = c3_1;	// L622
    int16_t v440 = v438;	// L623
    int16_t v441 = v439;	// L624
    int16_t v442 = v440 * v441;	// L625
    #pragma HLS bind_op variable=v442 op=mul impl=fabric
    ap_int<17> v443 = v437;	// L626
    ap_int<17> v444 = v442;	// L627
    ap_int<17> v445 = v443 + v444;	// L628
    ap_int<18> v446 = v432;	// L629
    ap_int<18> v447 = v445;	// L630
    ap_int<18> v448 = v446 + v447;	// L631
    int8_t v449 = a4;	// L632
    int8_t v450 = c4_1;	// L633
    int16_t v451 = v449;	// L634
    int16_t v452 = v450;	// L635
    int16_t v453 = v451 * v452;	// L636
    #pragma HLS bind_op variable=v453 op=mul impl=fabric
    int8_t v454 = a5;	// L637
    int8_t v455 = c5_1;	// L638
    int16_t v456 = v454;	// L639
    int16_t v457 = v455;	// L640
    int16_t v458 = v456 * v457;	// L641
    #pragma HLS bind_op variable=v458 op=mul impl=fabric
    ap_int<17> v459 = v453;	// L642
    ap_int<17> v460 = v458;	// L643
    ap_int<17> v461 = v459 + v460;	// L644
    int8_t v462 = a6;	// L645
    int8_t v463 = c6_1;	// L646
    int16_t v464 = v462;	// L647
    int16_t v465 = v463;	// L648
    int16_t v466 = v464 * v465;	// L649
    #pragma HLS bind_op variable=v466 op=mul impl=fabric
    int8_t v467 = a7;	// L650
    int8_t v468 = c7_1;	// L651
    int16_t v469 = v467;	// L652
    int16_t v470 = v468;	// L653
    int16_t v471 = v469 * v470;	// L654
    #pragma HLS bind_op variable=v471 op=mul impl=fabric
    ap_int<17> v472 = v466;	// L655
    ap_int<17> v473 = v471;	// L656
    ap_int<17> v474 = v472 + v473;	// L657
    ap_int<18> v475 = v461;	// L658
    ap_int<18> v476 = v474;	// L659
    ap_int<18> v477 = v475 + v476;	// L660
    ap_int<19> v478 = v448;	// L661
    ap_int<19> v479 = v477;	// L662
    ap_int<19> v480 = v478 + v479;	// L663
    ap_int<33> v481 = v419;	// L664
    ap_int<33> v482 = v480;	// L665
    ap_int<33> v483 = v481 + v482;	// L666
    v19.write(v483);	// L667
    int32_t v484 = p2;	// L668
    int8_t v485 = a0;	// L669
    int8_t v486 = c0_2;	// L670
    int16_t v487 = v485;	// L671
    int16_t v488 = v486;	// L672
    int16_t v489 = v487 * v488;	// L673
    #pragma HLS bind_op variable=v489 op=mul impl=fabric
    int8_t v490 = a1;	// L674
    int8_t v491 = c1_2;	// L675
    int16_t v492 = v490;	// L676
    int16_t v493 = v491;	// L677
    int16_t v494 = v492 * v493;	// L678
    #pragma HLS bind_op variable=v494 op=mul impl=fabric
    ap_int<17> v495 = v489;	// L679
    ap_int<17> v496 = v494;	// L680
    ap_int<17> v497 = v495 + v496;	// L681
    int8_t v498 = a2;	// L682
    int8_t v499 = c2_2;	// L683
    int16_t v500 = v498;	// L684
    int16_t v501 = v499;	// L685
    int16_t v502 = v500 * v501;	// L686
    #pragma HLS bind_op variable=v502 op=mul impl=fabric
    int8_t v503 = a3;	// L687
    int8_t v504 = c3_2;	// L688
    int16_t v505 = v503;	// L689
    int16_t v506 = v504;	// L690
    int16_t v507 = v505 * v506;	// L691
    #pragma HLS bind_op variable=v507 op=mul impl=fabric
    ap_int<17> v508 = v502;	// L692
    ap_int<17> v509 = v507;	// L693
    ap_int<17> v510 = v508 + v509;	// L694
    ap_int<18> v511 = v497;	// L695
    ap_int<18> v512 = v510;	// L696
    ap_int<18> v513 = v511 + v512;	// L697
    int8_t v514 = a4;	// L698
    int8_t v515 = c4_2;	// L699
    int16_t v516 = v514;	// L700
    int16_t v517 = v515;	// L701
    int16_t v518 = v516 * v517;	// L702
    #pragma HLS bind_op variable=v518 op=mul impl=fabric
    int8_t v519 = a5;	// L703
    int8_t v520 = c5_2;	// L704
    int16_t v521 = v519;	// L705
    int16_t v522 = v520;	// L706
    int16_t v523 = v521 * v522;	// L707
    #pragma HLS bind_op variable=v523 op=mul impl=fabric
    ap_int<17> v524 = v518;	// L708
    ap_int<17> v525 = v523;	// L709
    ap_int<17> v526 = v524 + v525;	// L710
    int8_t v527 = a6;	// L711
    int8_t v528 = c6_2;	// L712
    int16_t v529 = v527;	// L713
    int16_t v530 = v528;	// L714
    int16_t v531 = v529 * v530;	// L715
    #pragma HLS bind_op variable=v531 op=mul impl=fabric
    int8_t v532 = a7;	// L716
    int8_t v533 = c7_2;	// L717
    int16_t v534 = v532;	// L718
    int16_t v535 = v533;	// L719
    int16_t v536 = v534 * v535;	// L720
    #pragma HLS bind_op variable=v536 op=mul impl=fabric
    ap_int<17> v537 = v531;	// L721
    ap_int<17> v538 = v536;	// L722
    ap_int<17> v539 = v537 + v538;	// L723
    ap_int<18> v540 = v526;	// L724
    ap_int<18> v541 = v539;	// L725
    ap_int<18> v542 = v540 + v541;	// L726
    ap_int<19> v543 = v513;	// L727
    ap_int<19> v544 = v542;	// L728
    ap_int<19> v545 = v543 + v544;	// L729
    ap_int<33> v546 = v484;	// L730
    ap_int<33> v547 = v545;	// L731
    ap_int<33> v548 = v546 + v547;	// L732
    v21.write(v548);	// L733
    int32_t v549 = p3;	// L734
    int8_t v550 = a0;	// L735
    int8_t v551 = c0_3;	// L736
    int16_t v552 = v550;	// L737
    int16_t v553 = v551;	// L738
    int16_t v554 = v552 * v553;	// L739
    #pragma HLS bind_op variable=v554 op=mul impl=fabric
    int8_t v555 = a1;	// L740
    int8_t v556 = c1_3;	// L741
    int16_t v557 = v555;	// L742
    int16_t v558 = v556;	// L743
    int16_t v559 = v557 * v558;	// L744
    #pragma HLS bind_op variable=v559 op=mul impl=fabric
    ap_int<17> v560 = v554;	// L745
    ap_int<17> v561 = v559;	// L746
    ap_int<17> v562 = v560 + v561;	// L747
    int8_t v563 = a2;	// L748
    int8_t v564 = c2_3;	// L749
    int16_t v565 = v563;	// L750
    int16_t v566 = v564;	// L751
    int16_t v567 = v565 * v566;	// L752
    #pragma HLS bind_op variable=v567 op=mul impl=fabric
    int8_t v568 = a3;	// L753
    int8_t v569 = c3_3;	// L754
    int16_t v570 = v568;	// L755
    int16_t v571 = v569;	// L756
    int16_t v572 = v570 * v571;	// L757
    #pragma HLS bind_op variable=v572 op=mul impl=fabric
    ap_int<17> v573 = v567;	// L758
    ap_int<17> v574 = v572;	// L759
    ap_int<17> v575 = v573 + v574;	// L760
    ap_int<18> v576 = v562;	// L761
    ap_int<18> v577 = v575;	// L762
    ap_int<18> v578 = v576 + v577;	// L763
    int8_t v579 = a4;	// L764
    int8_t v580 = c4_3;	// L765
    int16_t v581 = v579;	// L766
    int16_t v582 = v580;	// L767
    int16_t v583 = v581 * v582;	// L768
    #pragma HLS bind_op variable=v583 op=mul impl=fabric
    int8_t v584 = a5;	// L769
    int8_t v585 = c5_3;	// L770
    int16_t v586 = v584;	// L771
    int16_t v587 = v585;	// L772
    int16_t v588 = v586 * v587;	// L773
    #pragma HLS bind_op variable=v588 op=mul impl=fabric
    ap_int<17> v589 = v583;	// L774
    ap_int<17> v590 = v588;	// L775
    ap_int<17> v591 = v589 + v590;	// L776
    int8_t v592 = a6;	// L777
    int8_t v593 = c6_3;	// L778
    int16_t v594 = v592;	// L779
    int16_t v595 = v593;	// L780
    int16_t v596 = v594 * v595;	// L781
    #pragma HLS bind_op variable=v596 op=mul impl=fabric
    int8_t v597 = a7;	// L782
    int8_t v598 = c7_3;	// L783
    int16_t v599 = v597;	// L784
    int16_t v600 = v598;	// L785
    int16_t v601 = v599 * v600;	// L786
    #pragma HLS bind_op variable=v601 op=mul impl=fabric
    ap_int<17> v602 = v596;	// L787
    ap_int<17> v603 = v601;	// L788
    ap_int<17> v604 = v602 + v603;	// L789
    ap_int<18> v605 = v591;	// L790
    ap_int<18> v606 = v604;	// L791
    ap_int<18> v607 = v605 + v606;	// L792
    ap_int<19> v608 = v578;	// L793
    ap_int<19> v609 = v607;	// L794
    ap_int<19> v610 = v608 + v609;	// L795
    ap_int<33> v611 = v549;	// L796
    ap_int<33> v612 = v610;	// L797
    ap_int<33> v613 = v611 + v612;	// L798
    v23.write(v613);	// L799
    int32_t v614 = p4;	// L800
    int8_t v615 = a0;	// L801
    int8_t v616 = c0_4;	// L802
    int16_t v617 = v615;	// L803
    int16_t v618 = v616;	// L804
    int16_t v619 = v617 * v618;	// L805
    #pragma HLS bind_op variable=v619 op=mul impl=fabric
    int8_t v620 = a1;	// L806
    int8_t v621 = c1_4;	// L807
    int16_t v622 = v620;	// L808
    int16_t v623 = v621;	// L809
    int16_t v624 = v622 * v623;	// L810
    #pragma HLS bind_op variable=v624 op=mul impl=fabric
    ap_int<17> v625 = v619;	// L811
    ap_int<17> v626 = v624;	// L812
    ap_int<17> v627 = v625 + v626;	// L813
    int8_t v628 = a2;	// L814
    int8_t v629 = c2_4;	// L815
    int16_t v630 = v628;	// L816
    int16_t v631 = v629;	// L817
    int16_t v632 = v630 * v631;	// L818
    #pragma HLS bind_op variable=v632 op=mul impl=fabric
    int8_t v633 = a3;	// L819
    int8_t v634 = c3_4;	// L820
    int16_t v635 = v633;	// L821
    int16_t v636 = v634;	// L822
    int16_t v637 = v635 * v636;	// L823
    #pragma HLS bind_op variable=v637 op=mul impl=fabric
    ap_int<17> v638 = v632;	// L824
    ap_int<17> v639 = v637;	// L825
    ap_int<17> v640 = v638 + v639;	// L826
    ap_int<18> v641 = v627;	// L827
    ap_int<18> v642 = v640;	// L828
    ap_int<18> v643 = v641 + v642;	// L829
    int8_t v644 = a4;	// L830
    int8_t v645 = c4_4;	// L831
    int16_t v646 = v644;	// L832
    int16_t v647 = v645;	// L833
    int16_t v648 = v646 * v647;	// L834
    #pragma HLS bind_op variable=v648 op=mul impl=fabric
    int8_t v649 = a5;	// L835
    int8_t v650 = c5_4;	// L836
    int16_t v651 = v649;	// L837
    int16_t v652 = v650;	// L838
    int16_t v653 = v651 * v652;	// L839
    #pragma HLS bind_op variable=v653 op=mul impl=fabric
    ap_int<17> v654 = v648;	// L840
    ap_int<17> v655 = v653;	// L841
    ap_int<17> v656 = v654 + v655;	// L842
    int8_t v657 = a6;	// L843
    int8_t v658 = c6_4;	// L844
    int16_t v659 = v657;	// L845
    int16_t v660 = v658;	// L846
    int16_t v661 = v659 * v660;	// L847
    #pragma HLS bind_op variable=v661 op=mul impl=fabric
    int8_t v662 = a7;	// L848
    int8_t v663 = c7_4;	// L849
    int16_t v664 = v662;	// L850
    int16_t v665 = v663;	// L851
    int16_t v666 = v664 * v665;	// L852
    #pragma HLS bind_op variable=v666 op=mul impl=fabric
    ap_int<17> v667 = v661;	// L853
    ap_int<17> v668 = v666;	// L854
    ap_int<17> v669 = v667 + v668;	// L855
    ap_int<18> v670 = v656;	// L856
    ap_int<18> v671 = v669;	// L857
    ap_int<18> v672 = v670 + v671;	// L858
    ap_int<19> v673 = v643;	// L859
    ap_int<19> v674 = v672;	// L860
    ap_int<19> v675 = v673 + v674;	// L861
    ap_int<33> v676 = v614;	// L862
    ap_int<33> v677 = v675;	// L863
    ap_int<33> v678 = v676 + v677;	// L864
    v25.write(v678);	// L865
    int32_t v679 = p5;	// L866
    int8_t v680 = a0;	// L867
    int8_t v681 = c0_5;	// L868
    int16_t v682 = v680;	// L869
    int16_t v683 = v681;	// L870
    int16_t v684 = v682 * v683;	// L871
    #pragma HLS bind_op variable=v684 op=mul impl=fabric
    int8_t v685 = a1;	// L872
    int8_t v686 = c1_5;	// L873
    int16_t v687 = v685;	// L874
    int16_t v688 = v686;	// L875
    int16_t v689 = v687 * v688;	// L876
    #pragma HLS bind_op variable=v689 op=mul impl=fabric
    ap_int<17> v690 = v684;	// L877
    ap_int<17> v691 = v689;	// L878
    ap_int<17> v692 = v690 + v691;	// L879
    int8_t v693 = a2;	// L880
    int8_t v694 = c2_5;	// L881
    int16_t v695 = v693;	// L882
    int16_t v696 = v694;	// L883
    int16_t v697 = v695 * v696;	// L884
    #pragma HLS bind_op variable=v697 op=mul impl=fabric
    int8_t v698 = a3;	// L885
    int8_t v699 = c3_5;	// L886
    int16_t v700 = v698;	// L887
    int16_t v701 = v699;	// L888
    int16_t v702 = v700 * v701;	// L889
    #pragma HLS bind_op variable=v702 op=mul impl=fabric
    ap_int<17> v703 = v697;	// L890
    ap_int<17> v704 = v702;	// L891
    ap_int<17> v705 = v703 + v704;	// L892
    ap_int<18> v706 = v692;	// L893
    ap_int<18> v707 = v705;	// L894
    ap_int<18> v708 = v706 + v707;	// L895
    int8_t v709 = a4;	// L896
    int8_t v710 = c4_5;	// L897
    int16_t v711 = v709;	// L898
    int16_t v712 = v710;	// L899
    int16_t v713 = v711 * v712;	// L900
    #pragma HLS bind_op variable=v713 op=mul impl=fabric
    int8_t v714 = a5;	// L901
    int8_t v715 = c5_5;	// L902
    int16_t v716 = v714;	// L903
    int16_t v717 = v715;	// L904
    int16_t v718 = v716 * v717;	// L905
    #pragma HLS bind_op variable=v718 op=mul impl=fabric
    ap_int<17> v719 = v713;	// L906
    ap_int<17> v720 = v718;	// L907
    ap_int<17> v721 = v719 + v720;	// L908
    int8_t v722 = a6;	// L909
    int8_t v723 = c6_5;	// L910
    int16_t v724 = v722;	// L911
    int16_t v725 = v723;	// L912
    int16_t v726 = v724 * v725;	// L913
    #pragma HLS bind_op variable=v726 op=mul impl=fabric
    int8_t v727 = a7;	// L914
    int8_t v728 = c7_5;	// L915
    int16_t v729 = v727;	// L916
    int16_t v730 = v728;	// L917
    int16_t v731 = v729 * v730;	// L918
    #pragma HLS bind_op variable=v731 op=mul impl=fabric
    ap_int<17> v732 = v726;	// L919
    ap_int<17> v733 = v731;	// L920
    ap_int<17> v734 = v732 + v733;	// L921
    ap_int<18> v735 = v721;	// L922
    ap_int<18> v736 = v734;	// L923
    ap_int<18> v737 = v735 + v736;	// L924
    ap_int<19> v738 = v708;	// L925
    ap_int<19> v739 = v737;	// L926
    ap_int<19> v740 = v738 + v739;	// L927
    ap_int<33> v741 = v679;	// L928
    ap_int<33> v742 = v740;	// L929
    ap_int<33> v743 = v741 + v742;	// L930
    v27.write(v743);	// L931
    int32_t v744 = p6;	// L932
    int8_t v745 = a0;	// L933
    int8_t v746 = c0_6;	// L934
    int16_t v747 = v745;	// L935
    int16_t v748 = v746;	// L936
    int16_t v749 = v747 * v748;	// L937
    #pragma HLS bind_op variable=v749 op=mul impl=fabric
    int8_t v750 = a1;	// L938
    int8_t v751 = c1_6;	// L939
    int16_t v752 = v750;	// L940
    int16_t v753 = v751;	// L941
    int16_t v754 = v752 * v753;	// L942
    #pragma HLS bind_op variable=v754 op=mul impl=fabric
    ap_int<17> v755 = v749;	// L943
    ap_int<17> v756 = v754;	// L944
    ap_int<17> v757 = v755 + v756;	// L945
    int8_t v758 = a2;	// L946
    int8_t v759 = c2_6;	// L947
    int16_t v760 = v758;	// L948
    int16_t v761 = v759;	// L949
    int16_t v762 = v760 * v761;	// L950
    #pragma HLS bind_op variable=v762 op=mul impl=fabric
    int8_t v763 = a3;	// L951
    int8_t v764 = c3_6;	// L952
    int16_t v765 = v763;	// L953
    int16_t v766 = v764;	// L954
    int16_t v767 = v765 * v766;	// L955
    #pragma HLS bind_op variable=v767 op=mul impl=fabric
    ap_int<17> v768 = v762;	// L956
    ap_int<17> v769 = v767;	// L957
    ap_int<17> v770 = v768 + v769;	// L958
    ap_int<18> v771 = v757;	// L959
    ap_int<18> v772 = v770;	// L960
    ap_int<18> v773 = v771 + v772;	// L961
    int8_t v774 = a4;	// L962
    int8_t v775 = c4_6;	// L963
    int16_t v776 = v774;	// L964
    int16_t v777 = v775;	// L965
    int16_t v778 = v776 * v777;	// L966
    #pragma HLS bind_op variable=v778 op=mul impl=fabric
    int8_t v779 = a5;	// L967
    int8_t v780 = c5_6;	// L968
    int16_t v781 = v779;	// L969
    int16_t v782 = v780;	// L970
    int16_t v783 = v781 * v782;	// L971
    #pragma HLS bind_op variable=v783 op=mul impl=fabric
    ap_int<17> v784 = v778;	// L972
    ap_int<17> v785 = v783;	// L973
    ap_int<17> v786 = v784 + v785;	// L974
    int8_t v787 = a6;	// L975
    int8_t v788 = c6_6;	// L976
    int16_t v789 = v787;	// L977
    int16_t v790 = v788;	// L978
    int16_t v791 = v789 * v790;	// L979
    #pragma HLS bind_op variable=v791 op=mul impl=fabric
    int8_t v792 = a7;	// L980
    int8_t v793 = c7_6;	// L981
    int16_t v794 = v792;	// L982
    int16_t v795 = v793;	// L983
    int16_t v796 = v794 * v795;	// L984
    #pragma HLS bind_op variable=v796 op=mul impl=fabric
    ap_int<17> v797 = v791;	// L985
    ap_int<17> v798 = v796;	// L986
    ap_int<17> v799 = v797 + v798;	// L987
    ap_int<18> v800 = v786;	// L988
    ap_int<18> v801 = v799;	// L989
    ap_int<18> v802 = v800 + v801;	// L990
    ap_int<19> v803 = v773;	// L991
    ap_int<19> v804 = v802;	// L992
    ap_int<19> v805 = v803 + v804;	// L993
    ap_int<33> v806 = v744;	// L994
    ap_int<33> v807 = v805;	// L995
    ap_int<33> v808 = v806 + v807;	// L996
    v29.write(v808);	// L997
    int32_t v809 = p7;	// L998
    int8_t v810 = a0;	// L999
    int8_t v811 = c0_7;	// L1000
    int16_t v812 = v810;	// L1001
    int16_t v813 = v811;	// L1002
    int16_t v814 = v812 * v813;	// L1003
    #pragma HLS bind_op variable=v814 op=mul impl=fabric
    int8_t v815 = a1;	// L1004
    int8_t v816 = c1_7;	// L1005
    int16_t v817 = v815;	// L1006
    int16_t v818 = v816;	// L1007
    int16_t v819 = v817 * v818;	// L1008
    #pragma HLS bind_op variable=v819 op=mul impl=fabric
    ap_int<17> v820 = v814;	// L1009
    ap_int<17> v821 = v819;	// L1010
    ap_int<17> v822 = v820 + v821;	// L1011
    int8_t v823 = a2;	// L1012
    int8_t v824 = c2_7;	// L1013
    int16_t v825 = v823;	// L1014
    int16_t v826 = v824;	// L1015
    int16_t v827 = v825 * v826;	// L1016
    #pragma HLS bind_op variable=v827 op=mul impl=fabric
    int8_t v828 = a3;	// L1017
    int8_t v829 = c3_7;	// L1018
    int16_t v830 = v828;	// L1019
    int16_t v831 = v829;	// L1020
    int16_t v832 = v830 * v831;	// L1021
    #pragma HLS bind_op variable=v832 op=mul impl=fabric
    ap_int<17> v833 = v827;	// L1022
    ap_int<17> v834 = v832;	// L1023
    ap_int<17> v835 = v833 + v834;	// L1024
    ap_int<18> v836 = v822;	// L1025
    ap_int<18> v837 = v835;	// L1026
    ap_int<18> v838 = v836 + v837;	// L1027
    int8_t v839 = a4;	// L1028
    int8_t v840 = c4_7;	// L1029
    int16_t v841 = v839;	// L1030
    int16_t v842 = v840;	// L1031
    int16_t v843 = v841 * v842;	// L1032
    #pragma HLS bind_op variable=v843 op=mul impl=fabric
    int8_t v844 = a5;	// L1033
    int8_t v845 = c5_7;	// L1034
    int16_t v846 = v844;	// L1035
    int16_t v847 = v845;	// L1036
    int16_t v848 = v846 * v847;	// L1037
    #pragma HLS bind_op variable=v848 op=mul impl=fabric
    ap_int<17> v849 = v843;	// L1038
    ap_int<17> v850 = v848;	// L1039
    ap_int<17> v851 = v849 + v850;	// L1040
    int8_t v852 = a6;	// L1041
    int8_t v853 = c6_7;	// L1042
    int16_t v854 = v852;	// L1043
    int16_t v855 = v853;	// L1044
    int16_t v856 = v854 * v855;	// L1045
    #pragma HLS bind_op variable=v856 op=mul impl=fabric
    int8_t v857 = a7;	// L1046
    int8_t v858 = c7_7;	// L1047
    int16_t v859 = v857;	// L1048
    int16_t v860 = v858;	// L1049
    int16_t v861 = v859 * v860;	// L1050
    #pragma HLS bind_op variable=v861 op=mul impl=fabric
    ap_int<17> v862 = v856;	// L1051
    ap_int<17> v863 = v861;	// L1052
    ap_int<17> v864 = v862 + v863;	// L1053
    ap_int<18> v865 = v851;	// L1054
    ap_int<18> v866 = v864;	// L1055
    ap_int<18> v867 = v865 + v866;	// L1056
    ap_int<19> v868 = v838;	// L1057
    ap_int<19> v869 = v867;	// L1058
    ap_int<19> v870 = v868 + v869;	// L1059
    ap_int<33> v871 = v809;	// L1060
    ap_int<33> v872 = v870;	// L1061
    ap_int<33> v873 = v871 + v872;	// L1062
    v31.write(v873);	// L1063
    int8_t v874 = n0_7;	// L1064
    v33.write(v874);	// L1065
    int8_t v875 = n0_6;	// L1066
    n0_7 = v875;	// L1067
    int8_t v876 = n0_5;	// L1068
    n0_6 = v876;	// L1069
    int8_t v877 = n0_4;	// L1070
    n0_5 = v877;	// L1071
    int8_t v878 = n0_3;	// L1072
    n0_4 = v878;	// L1073
    int8_t v879 = n0_2;	// L1074
    n0_3 = v879;	// L1075
    int8_t v880 = n0_1;	// L1076
    n0_2 = v880;	// L1077
    int8_t v881 = n0_0;	// L1078
    n0_1 = v881;	// L1079
    int8_t v882 = v32.read();	// L1080
    n0_0 = v882;	// L1081
    int8_t v883 = n1_7;	// L1082
    v35.write(v883);	// L1083
    int8_t v884 = n1_6;	// L1084
    n1_7 = v884;	// L1085
    int8_t v885 = n1_5;	// L1086
    n1_6 = v885;	// L1087
    int8_t v886 = n1_4;	// L1088
    n1_5 = v886;	// L1089
    int8_t v887 = n1_3;	// L1090
    n1_4 = v887;	// L1091
    int8_t v888 = n1_2;	// L1092
    n1_3 = v888;	// L1093
    int8_t v889 = n1_1;	// L1094
    n1_2 = v889;	// L1095
    int8_t v890 = n1_0;	// L1096
    n1_1 = v890;	// L1097
    int8_t v891 = v34.read();	// L1098
    n1_0 = v891;	// L1099
    int8_t v892 = n2_7;	// L1100
    v37.write(v892);	// L1101
    int8_t v893 = n2_6;	// L1102
    n2_7 = v893;	// L1103
    int8_t v894 = n2_5;	// L1104
    n2_6 = v894;	// L1105
    int8_t v895 = n2_4;	// L1106
    n2_5 = v895;	// L1107
    int8_t v896 = n2_3;	// L1108
    n2_4 = v896;	// L1109
    int8_t v897 = n2_2;	// L1110
    n2_3 = v897;	// L1111
    int8_t v898 = n2_1;	// L1112
    n2_2 = v898;	// L1113
    int8_t v899 = n2_0;	// L1114
    n2_1 = v899;	// L1115
    int8_t v900 = v36.read();	// L1116
    n2_0 = v900;	// L1117
    int8_t v901 = n3_7;	// L1118
    v39.write(v901);	// L1119
    int8_t v902 = n3_6;	// L1120
    n3_7 = v902;	// L1121
    int8_t v903 = n3_5;	// L1122
    n3_6 = v903;	// L1123
    int8_t v904 = n3_4;	// L1124
    n3_5 = v904;	// L1125
    int8_t v905 = n3_3;	// L1126
    n3_4 = v905;	// L1127
    int8_t v906 = n3_2;	// L1128
    n3_3 = v906;	// L1129
    int8_t v907 = n3_1;	// L1130
    n3_2 = v907;	// L1131
    int8_t v908 = n3_0;	// L1132
    n3_1 = v908;	// L1133
    int8_t v909 = v38.read();	// L1134
    n3_0 = v909;	// L1135
    int8_t v910 = n4_7;	// L1136
    v41.write(v910);	// L1137
    int8_t v911 = n4_6;	// L1138
    n4_7 = v911;	// L1139
    int8_t v912 = n4_5;	// L1140
    n4_6 = v912;	// L1141
    int8_t v913 = n4_4;	// L1142
    n4_5 = v913;	// L1143
    int8_t v914 = n4_3;	// L1144
    n4_4 = v914;	// L1145
    int8_t v915 = n4_2;	// L1146
    n4_3 = v915;	// L1147
    int8_t v916 = n4_1;	// L1148
    n4_2 = v916;	// L1149
    int8_t v917 = n4_0;	// L1150
    n4_1 = v917;	// L1151
    int8_t v918 = v40.read();	// L1152
    n4_0 = v918;	// L1153
    int8_t v919 = n5_7;	// L1154
    v43.write(v919);	// L1155
    int8_t v920 = n5_6;	// L1156
    n5_7 = v920;	// L1157
    int8_t v921 = n5_5;	// L1158
    n5_6 = v921;	// L1159
    int8_t v922 = n5_4;	// L1160
    n5_5 = v922;	// L1161
    int8_t v923 = n5_3;	// L1162
    n5_4 = v923;	// L1163
    int8_t v924 = n5_2;	// L1164
    n5_3 = v924;	// L1165
    int8_t v925 = n5_1;	// L1166
    n5_2 = v925;	// L1167
    int8_t v926 = n5_0;	// L1168
    n5_1 = v926;	// L1169
    int8_t v927 = v42.read();	// L1170
    n5_0 = v927;	// L1171
    int8_t v928 = n6_7;	// L1172
    v45.write(v928);	// L1173
    int8_t v929 = n6_6;	// L1174
    n6_7 = v929;	// L1175
    int8_t v930 = n6_5;	// L1176
    n6_6 = v930;	// L1177
    int8_t v931 = n6_4;	// L1178
    n6_5 = v931;	// L1179
    int8_t v932 = n6_3;	// L1180
    n6_4 = v932;	// L1181
    int8_t v933 = n6_2;	// L1182
    n6_3 = v933;	// L1183
    int8_t v934 = n6_1;	// L1184
    n6_2 = v934;	// L1185
    int8_t v935 = n6_0;	// L1186
    n6_1 = v935;	// L1187
    int8_t v936 = v44.read();	// L1188
    n6_0 = v936;	// L1189
    int8_t v937 = n7_7;	// L1190
    v47.write(v937);	// L1191
    int8_t v938 = n7_6;	// L1192
    n7_7 = v938;	// L1193
    int8_t v939 = n7_5;	// L1194
    n7_6 = v939;	// L1195
    int8_t v940 = n7_4;	// L1196
    n7_5 = v940;	// L1197
    int8_t v941 = n7_3;	// L1198
    n7_4 = v941;	// L1199
    int8_t v942 = n7_2;	// L1200
    n7_3 = v942;	// L1201
    int8_t v943 = n7_1;	// L1202
    n7_2 = v943;	// L1203
    int8_t v944 = n7_0;	// L1204
    n7_1 = v944;	// L1205
    int8_t v945 = v46.read();	// L1206
    n7_0 = v945;	// L1207
    int32_t v946 = s;	// L1208
    int32_t v947 = v946 & 15;	// L1210
    bool v948 = v947 == 15;	// L1211
    if (v948) {	// L1212
      int8_t v949 = n0_0;	// L1213
      c0_0 = v949;	// L1214
      int8_t v950 = n0_1;	// L1215
      c0_1 = v950;	// L1216
      int8_t v951 = n0_2;	// L1217
      c0_2 = v951;	// L1218
      int8_t v952 = n0_3;	// L1219
      c0_3 = v952;	// L1220
      int8_t v953 = n0_4;	// L1221
      c0_4 = v953;	// L1222
      int8_t v954 = n0_5;	// L1223
      c0_5 = v954;	// L1224
      int8_t v955 = n0_6;	// L1225
      c0_6 = v955;	// L1226
      int8_t v956 = n0_7;	// L1227
      c0_7 = v956;	// L1228
      int8_t v957 = n1_0;	// L1229
      c1_0 = v957;	// L1230
      int8_t v958 = n1_1;	// L1231
      c1_1 = v958;	// L1232
      int8_t v959 = n1_2;	// L1233
      c1_2 = v959;	// L1234
      int8_t v960 = n1_3;	// L1235
      c1_3 = v960;	// L1236
      int8_t v961 = n1_4;	// L1237
      c1_4 = v961;	// L1238
      int8_t v962 = n1_5;	// L1239
      c1_5 = v962;	// L1240
      int8_t v963 = n1_6;	// L1241
      c1_6 = v963;	// L1242
      int8_t v964 = n1_7;	// L1243
      c1_7 = v964;	// L1244
      int8_t v965 = n2_0;	// L1245
      c2_0 = v965;	// L1246
      int8_t v966 = n2_1;	// L1247
      c2_1 = v966;	// L1248
      int8_t v967 = n2_2;	// L1249
      c2_2 = v967;	// L1250
      int8_t v968 = n2_3;	// L1251
      c2_3 = v968;	// L1252
      int8_t v969 = n2_4;	// L1253
      c2_4 = v969;	// L1254
      int8_t v970 = n2_5;	// L1255
      c2_5 = v970;	// L1256
      int8_t v971 = n2_6;	// L1257
      c2_6 = v971;	// L1258
      int8_t v972 = n2_7;	// L1259
      c2_7 = v972;	// L1260
      int8_t v973 = n3_0;	// L1261
      c3_0 = v973;	// L1262
      int8_t v974 = n3_1;	// L1263
      c3_1 = v974;	// L1264
      int8_t v975 = n3_2;	// L1265
      c3_2 = v975;	// L1266
      int8_t v976 = n3_3;	// L1267
      c3_3 = v976;	// L1268
      int8_t v977 = n3_4;	// L1269
      c3_4 = v977;	// L1270
      int8_t v978 = n3_5;	// L1271
      c3_5 = v978;	// L1272
      int8_t v979 = n3_6;	// L1273
      c3_6 = v979;	// L1274
      int8_t v980 = n3_7;	// L1275
      c3_7 = v980;	// L1276
      int8_t v981 = n4_0;	// L1277
      c4_0 = v981;	// L1278
      int8_t v982 = n4_1;	// L1279
      c4_1 = v982;	// L1280
      int8_t v983 = n4_2;	// L1281
      c4_2 = v983;	// L1282
      int8_t v984 = n4_3;	// L1283
      c4_3 = v984;	// L1284
      int8_t v985 = n4_4;	// L1285
      c4_4 = v985;	// L1286
      int8_t v986 = n4_5;	// L1287
      c4_5 = v986;	// L1288
      int8_t v987 = n4_6;	// L1289
      c4_6 = v987;	// L1290
      int8_t v988 = n4_7;	// L1291
      c4_7 = v988;	// L1292
      int8_t v989 = n5_0;	// L1293
      c5_0 = v989;	// L1294
      int8_t v990 = n5_1;	// L1295
      c5_1 = v990;	// L1296
      int8_t v991 = n5_2;	// L1297
      c5_2 = v991;	// L1298
      int8_t v992 = n5_3;	// L1299
      c5_3 = v992;	// L1300
      int8_t v993 = n5_4;	// L1301
      c5_4 = v993;	// L1302
      int8_t v994 = n5_5;	// L1303
      c5_5 = v994;	// L1304
      int8_t v995 = n5_6;	// L1305
      c5_6 = v995;	// L1306
      int8_t v996 = n5_7;	// L1307
      c5_7 = v996;	// L1308
      int8_t v997 = n6_0;	// L1309
      c6_0 = v997;	// L1310
      int8_t v998 = n6_1;	// L1311
      c6_1 = v998;	// L1312
      int8_t v999 = n6_2;	// L1313
      c6_2 = v999;	// L1314
      int8_t v1000 = n6_3;	// L1315
      c6_3 = v1000;	// L1316
      int8_t v1001 = n6_4;	// L1317
      c6_4 = v1001;	// L1318
      int8_t v1002 = n6_5;	// L1319
      c6_5 = v1002;	// L1320
      int8_t v1003 = n6_6;	// L1321
      c6_6 = v1003;	// L1322
      int8_t v1004 = n6_7;	// L1323
      c6_7 = v1004;	// L1324
      int8_t v1005 = n7_0;	// L1325
      c7_0 = v1005;	// L1326
      int8_t v1006 = n7_1;	// L1327
      c7_1 = v1006;	// L1328
      int8_t v1007 = n7_2;	// L1329
      c7_2 = v1007;	// L1330
      int8_t v1008 = n7_3;	// L1331
      c7_3 = v1008;	// L1332
      int8_t v1009 = n7_4;	// L1333
      c7_4 = v1009;	// L1334
      int8_t v1010 = n7_5;	// L1335
      c7_5 = v1010;	// L1336
      int8_t v1011 = n7_6;	// L1337
      c7_6 = v1011;	// L1338
      int8_t v1012 = n7_7;	// L1339
      c7_7 = v1012;	// L1340
    }
  }
}


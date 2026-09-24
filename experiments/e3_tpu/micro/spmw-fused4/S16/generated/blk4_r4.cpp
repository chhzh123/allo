
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
void blk4_r4_0(
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
  hls::stream< int8_t >& v16,
  hls::stream< int8_t >& v17,
  hls::stream< int8_t >& v18,
  hls::stream< int8_t >& v19,
  hls::stream< int8_t >& v20,
  hls::stream< int8_t >& v21,
  hls::stream< int8_t >& v22,
  hls::stream< int8_t >& v23
) {	// L2
  int8_t n0_0;	// L5
  n0_0 = 0;	// L6
  int8_t n0_1;	// L7
  n0_1 = 0;	// L8
  int8_t n0_2;	// L9
  n0_2 = 0;	// L10
  int8_t n0_3;	// L11
  n0_3 = 0;	// L12
  int8_t n1_0;	// L13
  n1_0 = 0;	// L14
  int8_t n1_1;	// L15
  n1_1 = 0;	// L16
  int8_t n1_2;	// L17
  n1_2 = 0;	// L18
  int8_t n1_3;	// L19
  n1_3 = 0;	// L20
  int8_t n2_0;	// L21
  n2_0 = 0;	// L22
  int8_t n2_1;	// L23
  n2_1 = 0;	// L24
  int8_t n2_2;	// L25
  n2_2 = 0;	// L26
  int8_t n2_3;	// L27
  n2_3 = 0;	// L28
  int8_t n3_0;	// L29
  n3_0 = 0;	// L30
  int8_t n3_1;	// L31
  n3_1 = 0;	// L32
  int8_t n3_2;	// L33
  n3_2 = 0;	// L34
  int8_t n3_3;	// L35
  n3_3 = 0;	// L36
  l_S__k_0__k: for (int _k = 0; _k < 16; _k++) {	// L37
  #pragma HLS pipeline II=1
    int8_t v41 = n0_3;	// L38
    v17.write(v41);	// L39
    int8_t v42 = n0_2;	// L40
    n0_3 = v42;	// L41
    int8_t v43 = n0_1;	// L42
    n0_2 = v43;	// L43
    int8_t v44 = n0_0;	// L44
    n0_1 = v44;	// L45
    int8_t v45 = v16.read();	// L46
    n0_0 = v45;	// L47
    int8_t v46 = n1_3;	// L48
    v19.write(v46);	// L49
    int8_t v47 = n1_2;	// L50
    n1_3 = v47;	// L51
    int8_t v48 = n1_1;	// L52
    n1_2 = v48;	// L53
    int8_t v49 = n1_0;	// L54
    n1_1 = v49;	// L55
    int8_t v50 = v18.read();	// L56
    n1_0 = v50;	// L57
    int8_t v51 = n2_3;	// L58
    v21.write(v51);	// L59
    int8_t v52 = n2_2;	// L60
    n2_3 = v52;	// L61
    int8_t v53 = n2_1;	// L62
    n2_2 = v53;	// L63
    int8_t v54 = n2_0;	// L64
    n2_1 = v54;	// L65
    int8_t v55 = v20.read();	// L66
    n2_0 = v55;	// L67
    int8_t v56 = n3_3;	// L68
    v23.write(v56);	// L69
    int8_t v57 = n3_2;	// L70
    n3_3 = v57;	// L71
    int8_t v58 = n3_1;	// L72
    n3_2 = v58;	// L73
    int8_t v59 = n3_0;	// L74
    n3_1 = v59;	// L75
    int8_t v60 = v22.read();	// L76
    n3_0 = v60;	// L77
  }
  int8_t v61 = n0_0;	// L79
  int8_t c0_0;	// L80
  c0_0 = v61;	// L81
  int8_t v63 = n0_1;	// L82
  int8_t c0_1;	// L83
  c0_1 = v63;	// L84
  int8_t v65 = n0_2;	// L85
  int8_t c0_2;	// L86
  c0_2 = v65;	// L87
  int8_t v67 = n0_3;	// L88
  int8_t c0_3;	// L89
  c0_3 = v67;	// L90
  int8_t v69 = n1_0;	// L91
  int8_t c1_0;	// L92
  c1_0 = v69;	// L93
  int8_t v71 = n1_1;	// L94
  int8_t c1_1;	// L95
  c1_1 = v71;	// L96
  int8_t v73 = n1_2;	// L97
  int8_t c1_2;	// L98
  c1_2 = v73;	// L99
  int8_t v75 = n1_3;	// L100
  int8_t c1_3;	// L101
  c1_3 = v75;	// L102
  int8_t v77 = n2_0;	// L103
  int8_t c2_0;	// L104
  c2_0 = v77;	// L105
  int8_t v79 = n2_1;	// L106
  int8_t c2_1;	// L107
  c2_1 = v79;	// L108
  int8_t v81 = n2_2;	// L109
  int8_t c2_2;	// L110
  c2_2 = v81;	// L111
  int8_t v83 = n2_3;	// L112
  int8_t c2_3;	// L113
  c2_3 = v83;	// L114
  int8_t v85 = n3_0;	// L115
  int8_t c3_0;	// L116
  c3_0 = v85;	// L117
  int8_t v87 = n3_1;	// L118
  int8_t c3_1;	// L119
  c3_1 = v87;	// L120
  int8_t v89 = n3_2;	// L121
  int8_t c3_2;	// L122
  c3_2 = v89;	// L123
  int8_t v91 = n3_3;	// L124
  int8_t c3_3;	// L125
  c3_3 = v91;	// L126
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L127
  #pragma HLS pipeline II=1
    int8_t v94 = v0.read();	// L128
    int8_t a0;	// L129
    a0 = v94;	// L130
    int8_t v96 = v2.read();	// L131
    int8_t a1;	// L132
    a1 = v96;	// L133
    int8_t v98 = v4.read();	// L134
    int8_t a2;	// L135
    a2 = v98;	// L136
    int8_t v100 = v6.read();	// L137
    int8_t a3;	// L138
    a3 = v100;	// L139
    int32_t v102 = v8.read();	// L140
    int32_t p0;	// L141
    p0 = v102;	// L142
    int32_t v104 = v10.read();	// L143
    int32_t p1;	// L144
    p1 = v104;	// L145
    int32_t v106 = v12.read();	// L146
    int32_t p2;	// L147
    p2 = v106;	// L148
    int32_t v108 = v14.read();	// L149
    int32_t p3;	// L150
    p3 = v108;	// L151
    int8_t v110 = a0;	// L152
    v1.write(v110);	// L153
    int8_t v111 = a1;	// L154
    v3.write(v111);	// L155
    int8_t v112 = a2;	// L156
    v5.write(v112);	// L157
    int8_t v113 = a3;	// L158
    v7.write(v113);	// L159
    int32_t v114 = p0;	// L160
    int8_t v115 = a0;	// L161
    int8_t v116 = c0_0;	// L162
    int16_t v117 = v115;	// L163
    int16_t v118 = v116;	// L164
    int16_t v119 = v117 * v118;	// L165
    #pragma HLS bind_op variable=v119 op=mul impl=fabric
    int8_t v120 = a1;	// L166
    int8_t v121 = c1_0;	// L167
    int16_t v122 = v120;	// L168
    int16_t v123 = v121;	// L169
    int16_t v124 = v122 * v123;	// L170
    #pragma HLS bind_op variable=v124 op=mul impl=fabric
    ap_int<17> v125 = v119;	// L171
    ap_int<17> v126 = v124;	// L172
    ap_int<17> v127 = v125 + v126;	// L173
    int8_t v128 = a2;	// L174
    int8_t v129 = c2_0;	// L175
    int16_t v130 = v128;	// L176
    int16_t v131 = v129;	// L177
    int16_t v132 = v130 * v131;	// L178
    #pragma HLS bind_op variable=v132 op=mul impl=fabric
    int8_t v133 = a3;	// L179
    int8_t v134 = c3_0;	// L180
    int16_t v135 = v133;	// L181
    int16_t v136 = v134;	// L182
    int16_t v137 = v135 * v136;	// L183
    #pragma HLS bind_op variable=v137 op=mul impl=fabric
    ap_int<17> v138 = v132;	// L184
    ap_int<17> v139 = v137;	// L185
    ap_int<17> v140 = v138 + v139;	// L186
    ap_int<18> v141 = v127;	// L187
    ap_int<18> v142 = v140;	// L188
    ap_int<18> v143 = v141 + v142;	// L189
    ap_int<33> v144 = v114;	// L190
    ap_int<33> v145 = v143;	// L191
    ap_int<33> v146 = v144 + v145;	// L192
    v9.write(v146);	// L193
    int32_t v147 = p1;	// L194
    int8_t v148 = a0;	// L195
    int8_t v149 = c0_1;	// L196
    int16_t v150 = v148;	// L197
    int16_t v151 = v149;	// L198
    int16_t v152 = v150 * v151;	// L199
    #pragma HLS bind_op variable=v152 op=mul impl=fabric
    int8_t v153 = a1;	// L200
    int8_t v154 = c1_1;	// L201
    int16_t v155 = v153;	// L202
    int16_t v156 = v154;	// L203
    int16_t v157 = v155 * v156;	// L204
    #pragma HLS bind_op variable=v157 op=mul impl=fabric
    ap_int<17> v158 = v152;	// L205
    ap_int<17> v159 = v157;	// L206
    ap_int<17> v160 = v158 + v159;	// L207
    int8_t v161 = a2;	// L208
    int8_t v162 = c2_1;	// L209
    int16_t v163 = v161;	// L210
    int16_t v164 = v162;	// L211
    int16_t v165 = v163 * v164;	// L212
    #pragma HLS bind_op variable=v165 op=mul impl=fabric
    int8_t v166 = a3;	// L213
    int8_t v167 = c3_1;	// L214
    int16_t v168 = v166;	// L215
    int16_t v169 = v167;	// L216
    int16_t v170 = v168 * v169;	// L217
    #pragma HLS bind_op variable=v170 op=mul impl=fabric
    ap_int<17> v171 = v165;	// L218
    ap_int<17> v172 = v170;	// L219
    ap_int<17> v173 = v171 + v172;	// L220
    ap_int<18> v174 = v160;	// L221
    ap_int<18> v175 = v173;	// L222
    ap_int<18> v176 = v174 + v175;	// L223
    ap_int<33> v177 = v147;	// L224
    ap_int<33> v178 = v176;	// L225
    ap_int<33> v179 = v177 + v178;	// L226
    v11.write(v179);	// L227
    int32_t v180 = p2;	// L228
    int8_t v181 = a0;	// L229
    int8_t v182 = c0_2;	// L230
    int16_t v183 = v181;	// L231
    int16_t v184 = v182;	// L232
    int16_t v185 = v183 * v184;	// L233
    #pragma HLS bind_op variable=v185 op=mul impl=fabric
    int8_t v186 = a1;	// L234
    int8_t v187 = c1_2;	// L235
    int16_t v188 = v186;	// L236
    int16_t v189 = v187;	// L237
    int16_t v190 = v188 * v189;	// L238
    #pragma HLS bind_op variable=v190 op=mul impl=fabric
    ap_int<17> v191 = v185;	// L239
    ap_int<17> v192 = v190;	// L240
    ap_int<17> v193 = v191 + v192;	// L241
    int8_t v194 = a2;	// L242
    int8_t v195 = c2_2;	// L243
    int16_t v196 = v194;	// L244
    int16_t v197 = v195;	// L245
    int16_t v198 = v196 * v197;	// L246
    #pragma HLS bind_op variable=v198 op=mul impl=fabric
    int8_t v199 = a3;	// L247
    int8_t v200 = c3_2;	// L248
    int16_t v201 = v199;	// L249
    int16_t v202 = v200;	// L250
    int16_t v203 = v201 * v202;	// L251
    #pragma HLS bind_op variable=v203 op=mul impl=fabric
    ap_int<17> v204 = v198;	// L252
    ap_int<17> v205 = v203;	// L253
    ap_int<17> v206 = v204 + v205;	// L254
    ap_int<18> v207 = v193;	// L255
    ap_int<18> v208 = v206;	// L256
    ap_int<18> v209 = v207 + v208;	// L257
    ap_int<33> v210 = v180;	// L258
    ap_int<33> v211 = v209;	// L259
    ap_int<33> v212 = v210 + v211;	// L260
    v13.write(v212);	// L261
    int32_t v213 = p3;	// L262
    int8_t v214 = a0;	// L263
    int8_t v215 = c0_3;	// L264
    int16_t v216 = v214;	// L265
    int16_t v217 = v215;	// L266
    int16_t v218 = v216 * v217;	// L267
    #pragma HLS bind_op variable=v218 op=mul impl=fabric
    int8_t v219 = a1;	// L268
    int8_t v220 = c1_3;	// L269
    int16_t v221 = v219;	// L270
    int16_t v222 = v220;	// L271
    int16_t v223 = v221 * v222;	// L272
    #pragma HLS bind_op variable=v223 op=mul impl=fabric
    ap_int<17> v224 = v218;	// L273
    ap_int<17> v225 = v223;	// L274
    ap_int<17> v226 = v224 + v225;	// L275
    int8_t v227 = a2;	// L276
    int8_t v228 = c2_3;	// L277
    int16_t v229 = v227;	// L278
    int16_t v230 = v228;	// L279
    int16_t v231 = v229 * v230;	// L280
    #pragma HLS bind_op variable=v231 op=mul impl=fabric
    int8_t v232 = a3;	// L281
    int8_t v233 = c3_3;	// L282
    int16_t v234 = v232;	// L283
    int16_t v235 = v233;	// L284
    int16_t v236 = v234 * v235;	// L285
    #pragma HLS bind_op variable=v236 op=mul impl=fabric
    ap_int<17> v237 = v231;	// L286
    ap_int<17> v238 = v236;	// L287
    ap_int<17> v239 = v237 + v238;	// L288
    ap_int<18> v240 = v226;	// L289
    ap_int<18> v241 = v239;	// L290
    ap_int<18> v242 = v240 + v241;	// L291
    ap_int<33> v243 = v213;	// L292
    ap_int<33> v244 = v242;	// L293
    ap_int<33> v245 = v243 + v244;	// L294
    v15.write(v245);	// L295
    int8_t v246 = n0_3;	// L296
    v17.write(v246);	// L297
    int8_t v247 = n0_2;	// L298
    n0_3 = v247;	// L299
    int8_t v248 = n0_1;	// L300
    n0_2 = v248;	// L301
    int8_t v249 = n0_0;	// L302
    n0_1 = v249;	// L303
    int8_t v250 = v16.read();	// L304
    n0_0 = v250;	// L305
    int8_t v251 = n1_3;	// L306
    v19.write(v251);	// L307
    int8_t v252 = n1_2;	// L308
    n1_3 = v252;	// L309
    int8_t v253 = n1_1;	// L310
    n1_2 = v253;	// L311
    int8_t v254 = n1_0;	// L312
    n1_1 = v254;	// L313
    int8_t v255 = v18.read();	// L314
    n1_0 = v255;	// L315
    int8_t v256 = n2_3;	// L316
    v21.write(v256);	// L317
    int8_t v257 = n2_2;	// L318
    n2_3 = v257;	// L319
    int8_t v258 = n2_1;	// L320
    n2_2 = v258;	// L321
    int8_t v259 = n2_0;	// L322
    n2_1 = v259;	// L323
    int8_t v260 = v20.read();	// L324
    n2_0 = v260;	// L325
    int8_t v261 = n3_3;	// L326
    v23.write(v261);	// L327
    int8_t v262 = n3_2;	// L328
    n3_3 = v262;	// L329
    int8_t v263 = n3_1;	// L330
    n3_2 = v263;	// L331
    int8_t v264 = n3_0;	// L332
    n3_1 = v264;	// L333
    int8_t v265 = v22.read();	// L334
    n3_0 = v265;	// L335
    int32_t v266 = s;	// L336
    int32_t v267 = v266 & 15;	// L338
    bool v268 = v267 == 15;	// L339
    if (v268) {	// L340
      int8_t v269 = n0_0;	// L341
      c0_0 = v269;	// L342
      int8_t v270 = n0_1;	// L343
      c0_1 = v270;	// L344
      int8_t v271 = n0_2;	// L345
      c0_2 = v271;	// L346
      int8_t v272 = n0_3;	// L347
      c0_3 = v272;	// L348
      int8_t v273 = n1_0;	// L349
      c1_0 = v273;	// L350
      int8_t v274 = n1_1;	// L351
      c1_1 = v274;	// L352
      int8_t v275 = n1_2;	// L353
      c1_2 = v275;	// L354
      int8_t v276 = n1_3;	// L355
      c1_3 = v276;	// L356
      int8_t v277 = n2_0;	// L357
      c2_0 = v277;	// L358
      int8_t v278 = n2_1;	// L359
      c2_1 = v278;	// L360
      int8_t v279 = n2_2;	// L361
      c2_2 = v279;	// L362
      int8_t v280 = n2_3;	// L363
      c2_3 = v280;	// L364
      int8_t v281 = n3_0;	// L365
      c3_0 = v281;	// L366
      int8_t v282 = n3_1;	// L367
      c3_1 = v282;	// L368
      int8_t v283 = n3_2;	// L369
      c3_2 = v283;	// L370
      int8_t v284 = n3_3;	// L371
      c3_3 = v284;	// L372
    }
  }
}


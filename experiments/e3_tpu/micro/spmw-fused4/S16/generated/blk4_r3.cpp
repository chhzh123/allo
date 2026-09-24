
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
void blk4_r3_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int32_t >& v6,
  hls::stream< int32_t >& v7,
  hls::stream< int32_t >& v8,
  hls::stream< int32_t >& v9,
  hls::stream< int32_t >& v10,
  hls::stream< int32_t >& v11,
  hls::stream< int8_t >& v12,
  hls::stream< int8_t >& v13,
  hls::stream< int8_t >& v14,
  hls::stream< int8_t >& v15
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
    int8_t v33 = n0_2;	// L38
    n0_3 = v33;	// L39
    int8_t v34 = n0_1;	// L40
    n0_2 = v34;	// L41
    int8_t v35 = n0_0;	// L42
    n0_1 = v35;	// L43
    int8_t v36 = v12.read();	// L44
    n0_0 = v36;	// L45
    int8_t v37 = n1_2;	// L46
    n1_3 = v37;	// L47
    int8_t v38 = n1_1;	// L48
    n1_2 = v38;	// L49
    int8_t v39 = n1_0;	// L50
    n1_1 = v39;	// L51
    int8_t v40 = v13.read();	// L52
    n1_0 = v40;	// L53
    int8_t v41 = n2_2;	// L54
    n2_3 = v41;	// L55
    int8_t v42 = n2_1;	// L56
    n2_2 = v42;	// L57
    int8_t v43 = n2_0;	// L58
    n2_1 = v43;	// L59
    int8_t v44 = v14.read();	// L60
    n2_0 = v44;	// L61
    int8_t v45 = n3_2;	// L62
    n3_3 = v45;	// L63
    int8_t v46 = n3_1;	// L64
    n3_2 = v46;	// L65
    int8_t v47 = n3_0;	// L66
    n3_1 = v47;	// L67
    int8_t v48 = v15.read();	// L68
    n3_0 = v48;	// L69
  }
  int8_t v49 = n0_0;	// L71
  int8_t c0_0;	// L72
  c0_0 = v49;	// L73
  int8_t v51 = n0_1;	// L74
  int8_t c0_1;	// L75
  c0_1 = v51;	// L76
  int8_t v53 = n0_2;	// L77
  int8_t c0_2;	// L78
  c0_2 = v53;	// L79
  int8_t v55 = n0_3;	// L80
  int8_t c0_3;	// L81
  c0_3 = v55;	// L82
  int8_t v57 = n1_0;	// L83
  int8_t c1_0;	// L84
  c1_0 = v57;	// L85
  int8_t v59 = n1_1;	// L86
  int8_t c1_1;	// L87
  c1_1 = v59;	// L88
  int8_t v61 = n1_2;	// L89
  int8_t c1_2;	// L90
  c1_2 = v61;	// L91
  int8_t v63 = n1_3;	// L92
  int8_t c1_3;	// L93
  c1_3 = v63;	// L94
  int8_t v65 = n2_0;	// L95
  int8_t c2_0;	// L96
  c2_0 = v65;	// L97
  int8_t v67 = n2_1;	// L98
  int8_t c2_1;	// L99
  c2_1 = v67;	// L100
  int8_t v69 = n2_2;	// L101
  int8_t c2_2;	// L102
  c2_2 = v69;	// L103
  int8_t v71 = n2_3;	// L104
  int8_t c2_3;	// L105
  c2_3 = v71;	// L106
  int8_t v73 = n3_0;	// L107
  int8_t c3_0;	// L108
  c3_0 = v73;	// L109
  int8_t v75 = n3_1;	// L110
  int8_t c3_1;	// L111
  c3_1 = v75;	// L112
  int8_t v77 = n3_2;	// L113
  int8_t c3_2;	// L114
  c3_2 = v77;	// L115
  int8_t v79 = n3_3;	// L116
  int8_t c3_3;	// L117
  c3_3 = v79;	// L118
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L119
  #pragma HLS pipeline II=1
    int8_t v82 = v0.read();	// L120
    int8_t a0;	// L121
    a0 = v82;	// L122
    int8_t v84 = v1.read();	// L123
    int8_t a1;	// L124
    a1 = v84;	// L125
    int8_t v86 = v2.read();	// L126
    int8_t a2;	// L127
    a2 = v86;	// L128
    int8_t v88 = v3.read();	// L129
    int8_t a3;	// L130
    a3 = v88;	// L131
    int32_t v90 = v4.read();	// L132
    int32_t p0;	// L133
    p0 = v90;	// L134
    int32_t v92 = v6.read();	// L135
    int32_t p1;	// L136
    p1 = v92;	// L137
    int32_t v94 = v8.read();	// L138
    int32_t p2;	// L139
    p2 = v94;	// L140
    int32_t v96 = v10.read();	// L141
    int32_t p3;	// L142
    p3 = v96;	// L143
    int32_t v98 = p0;	// L144
    int8_t v99 = a0;	// L145
    int8_t v100 = c0_0;	// L146
    int16_t v101 = v99;	// L147
    int16_t v102 = v100;	// L148
    int16_t v103 = v101 * v102;	// L149
    #pragma HLS bind_op variable=v103 op=mul impl=fabric
    int8_t v104 = a1;	// L150
    int8_t v105 = c1_0;	// L151
    int16_t v106 = v104;	// L152
    int16_t v107 = v105;	// L153
    int16_t v108 = v106 * v107;	// L154
    #pragma HLS bind_op variable=v108 op=mul impl=fabric
    ap_int<17> v109 = v103;	// L155
    ap_int<17> v110 = v108;	// L156
    ap_int<17> v111 = v109 + v110;	// L157
    int8_t v112 = a2;	// L158
    int8_t v113 = c2_0;	// L159
    int16_t v114 = v112;	// L160
    int16_t v115 = v113;	// L161
    int16_t v116 = v114 * v115;	// L162
    #pragma HLS bind_op variable=v116 op=mul impl=fabric
    int8_t v117 = a3;	// L163
    int8_t v118 = c3_0;	// L164
    int16_t v119 = v117;	// L165
    int16_t v120 = v118;	// L166
    int16_t v121 = v119 * v120;	// L167
    #pragma HLS bind_op variable=v121 op=mul impl=fabric
    ap_int<17> v122 = v116;	// L168
    ap_int<17> v123 = v121;	// L169
    ap_int<17> v124 = v122 + v123;	// L170
    ap_int<18> v125 = v111;	// L171
    ap_int<18> v126 = v124;	// L172
    ap_int<18> v127 = v125 + v126;	// L173
    ap_int<33> v128 = v98;	// L174
    ap_int<33> v129 = v127;	// L175
    ap_int<33> v130 = v128 + v129;	// L176
    v5.write(v130);	// L177
    int32_t v131 = p1;	// L178
    int8_t v132 = a0;	// L179
    int8_t v133 = c0_1;	// L180
    int16_t v134 = v132;	// L181
    int16_t v135 = v133;	// L182
    int16_t v136 = v134 * v135;	// L183
    #pragma HLS bind_op variable=v136 op=mul impl=fabric
    int8_t v137 = a1;	// L184
    int8_t v138 = c1_1;	// L185
    int16_t v139 = v137;	// L186
    int16_t v140 = v138;	// L187
    int16_t v141 = v139 * v140;	// L188
    #pragma HLS bind_op variable=v141 op=mul impl=fabric
    ap_int<17> v142 = v136;	// L189
    ap_int<17> v143 = v141;	// L190
    ap_int<17> v144 = v142 + v143;	// L191
    int8_t v145 = a2;	// L192
    int8_t v146 = c2_1;	// L193
    int16_t v147 = v145;	// L194
    int16_t v148 = v146;	// L195
    int16_t v149 = v147 * v148;	// L196
    #pragma HLS bind_op variable=v149 op=mul impl=fabric
    int8_t v150 = a3;	// L197
    int8_t v151 = c3_1;	// L198
    int16_t v152 = v150;	// L199
    int16_t v153 = v151;	// L200
    int16_t v154 = v152 * v153;	// L201
    #pragma HLS bind_op variable=v154 op=mul impl=fabric
    ap_int<17> v155 = v149;	// L202
    ap_int<17> v156 = v154;	// L203
    ap_int<17> v157 = v155 + v156;	// L204
    ap_int<18> v158 = v144;	// L205
    ap_int<18> v159 = v157;	// L206
    ap_int<18> v160 = v158 + v159;	// L207
    ap_int<33> v161 = v131;	// L208
    ap_int<33> v162 = v160;	// L209
    ap_int<33> v163 = v161 + v162;	// L210
    v7.write(v163);	// L211
    int32_t v164 = p2;	// L212
    int8_t v165 = a0;	// L213
    int8_t v166 = c0_2;	// L214
    int16_t v167 = v165;	// L215
    int16_t v168 = v166;	// L216
    int16_t v169 = v167 * v168;	// L217
    #pragma HLS bind_op variable=v169 op=mul impl=fabric
    int8_t v170 = a1;	// L218
    int8_t v171 = c1_2;	// L219
    int16_t v172 = v170;	// L220
    int16_t v173 = v171;	// L221
    int16_t v174 = v172 * v173;	// L222
    #pragma HLS bind_op variable=v174 op=mul impl=fabric
    ap_int<17> v175 = v169;	// L223
    ap_int<17> v176 = v174;	// L224
    ap_int<17> v177 = v175 + v176;	// L225
    int8_t v178 = a2;	// L226
    int8_t v179 = c2_2;	// L227
    int16_t v180 = v178;	// L228
    int16_t v181 = v179;	// L229
    int16_t v182 = v180 * v181;	// L230
    #pragma HLS bind_op variable=v182 op=mul impl=fabric
    int8_t v183 = a3;	// L231
    int8_t v184 = c3_2;	// L232
    int16_t v185 = v183;	// L233
    int16_t v186 = v184;	// L234
    int16_t v187 = v185 * v186;	// L235
    #pragma HLS bind_op variable=v187 op=mul impl=fabric
    ap_int<17> v188 = v182;	// L236
    ap_int<17> v189 = v187;	// L237
    ap_int<17> v190 = v188 + v189;	// L238
    ap_int<18> v191 = v177;	// L239
    ap_int<18> v192 = v190;	// L240
    ap_int<18> v193 = v191 + v192;	// L241
    ap_int<33> v194 = v164;	// L242
    ap_int<33> v195 = v193;	// L243
    ap_int<33> v196 = v194 + v195;	// L244
    v9.write(v196);	// L245
    int32_t v197 = p3;	// L246
    int8_t v198 = a0;	// L247
    int8_t v199 = c0_3;	// L248
    int16_t v200 = v198;	// L249
    int16_t v201 = v199;	// L250
    int16_t v202 = v200 * v201;	// L251
    #pragma HLS bind_op variable=v202 op=mul impl=fabric
    int8_t v203 = a1;	// L252
    int8_t v204 = c1_3;	// L253
    int16_t v205 = v203;	// L254
    int16_t v206 = v204;	// L255
    int16_t v207 = v205 * v206;	// L256
    #pragma HLS bind_op variable=v207 op=mul impl=fabric
    ap_int<17> v208 = v202;	// L257
    ap_int<17> v209 = v207;	// L258
    ap_int<17> v210 = v208 + v209;	// L259
    int8_t v211 = a2;	// L260
    int8_t v212 = c2_3;	// L261
    int16_t v213 = v211;	// L262
    int16_t v214 = v212;	// L263
    int16_t v215 = v213 * v214;	// L264
    #pragma HLS bind_op variable=v215 op=mul impl=fabric
    int8_t v216 = a3;	// L265
    int8_t v217 = c3_3;	// L266
    int16_t v218 = v216;	// L267
    int16_t v219 = v217;	// L268
    int16_t v220 = v218 * v219;	// L269
    #pragma HLS bind_op variable=v220 op=mul impl=fabric
    ap_int<17> v221 = v215;	// L270
    ap_int<17> v222 = v220;	// L271
    ap_int<17> v223 = v221 + v222;	// L272
    ap_int<18> v224 = v210;	// L273
    ap_int<18> v225 = v223;	// L274
    ap_int<18> v226 = v224 + v225;	// L275
    ap_int<33> v227 = v197;	// L276
    ap_int<33> v228 = v226;	// L277
    ap_int<33> v229 = v227 + v228;	// L278
    v11.write(v229);	// L279
    int8_t v230 = n0_2;	// L280
    n0_3 = v230;	// L281
    int8_t v231 = n0_1;	// L282
    n0_2 = v231;	// L283
    int8_t v232 = n0_0;	// L284
    n0_1 = v232;	// L285
    int8_t v233 = v12.read();	// L286
    n0_0 = v233;	// L287
    int8_t v234 = n1_2;	// L288
    n1_3 = v234;	// L289
    int8_t v235 = n1_1;	// L290
    n1_2 = v235;	// L291
    int8_t v236 = n1_0;	// L292
    n1_1 = v236;	// L293
    int8_t v237 = v13.read();	// L294
    n1_0 = v237;	// L295
    int8_t v238 = n2_2;	// L296
    n2_3 = v238;	// L297
    int8_t v239 = n2_1;	// L298
    n2_2 = v239;	// L299
    int8_t v240 = n2_0;	// L300
    n2_1 = v240;	// L301
    int8_t v241 = v14.read();	// L302
    n2_0 = v241;	// L303
    int8_t v242 = n3_2;	// L304
    n3_3 = v242;	// L305
    int8_t v243 = n3_1;	// L306
    n3_2 = v243;	// L307
    int8_t v244 = n3_0;	// L308
    n3_1 = v244;	// L309
    int8_t v245 = v15.read();	// L310
    n3_0 = v245;	// L311
    int32_t v246 = s;	// L312
    int32_t v247 = v246 & 15;	// L314
    bool v248 = v247 == 15;	// L315
    if (v248) {	// L316
      int8_t v249 = n0_0;	// L317
      c0_0 = v249;	// L318
      int8_t v250 = n0_1;	// L319
      c0_1 = v250;	// L320
      int8_t v251 = n0_2;	// L321
      c0_2 = v251;	// L322
      int8_t v252 = n0_3;	// L323
      c0_3 = v252;	// L324
      int8_t v253 = n1_0;	// L325
      c1_0 = v253;	// L326
      int8_t v254 = n1_1;	// L327
      c1_1 = v254;	// L328
      int8_t v255 = n1_2;	// L329
      c1_2 = v255;	// L330
      int8_t v256 = n1_3;	// L331
      c1_3 = v256;	// L332
      int8_t v257 = n2_0;	// L333
      c2_0 = v257;	// L334
      int8_t v258 = n2_1;	// L335
      c2_1 = v258;	// L336
      int8_t v259 = n2_2;	// L337
      c2_2 = v259;	// L338
      int8_t v260 = n2_3;	// L339
      c2_3 = v260;	// L340
      int8_t v261 = n3_0;	// L341
      c3_0 = v261;	// L342
      int8_t v262 = n3_1;	// L343
      c3_1 = v262;	// L344
      int8_t v263 = n3_2;	// L345
      c3_2 = v263;	// L346
      int8_t v264 = n3_3;	// L347
      c3_3 = v264;	// L348
    }
  }
}


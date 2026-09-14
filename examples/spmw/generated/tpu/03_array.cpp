
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
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 6; _t++) {	// L4
  #pragma HLS pipeline II=1
    int8_t v4 = v0[_t][v1];	// L5
    v2.write(v4);	// L6
  }
}

void mac_r0(
  int8_t v5[4][4],
  int v6,
  int v7,
  hls::stream< int8_t >& v8,
  hls::stream< int8_t >& v9,
  hls::stream< int32_t >& v10,
  hls::stream< int32_t >& v11
) {	// L10
  #pragma HLS array_partition variable=v5 complete dim=1
  #pragma HLS array_partition variable=v5 complete dim=2

  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L11
  #pragma HLS pipeline II=1
    int8_t v13 = v8.read();	// L12
    int8_t a;	// L13
    a = v13;	// L14
    int32_t v15 = v10.read();	// L15
    int32_t p;	// L16
    p = v15;	// L17
    int32_t v17 = p;	// L18
    int8_t v18 = a;	// L19
    int8_t v19 = v5[v6][v7];	// L20
    int16_t v20 = v18;	// L21
    int16_t v21 = v19;	// L22
    int16_t v22 = v20 * v21;	// L23
    ap_int<33> v23 = v17;	// L24
    ap_int<33> v24 = v22;	// L25
    ap_int<33> v25 = v23 + v24;	// L26
    v11.write(v25);	// L27
    int8_t v26 = a;	// L28
    v9.write(v26);	// L29
  }
}

void mac_r1(
  int8_t v27[4][4],
  int v28,
  int v29,
  hls::stream< int8_t >& v30,
  hls::stream< int8_t >& v31,
  hls::stream< int32_t >& v32,
  hls::stream< int32_t >& v33
) {	// L33
  #pragma HLS array_partition variable=v27 complete dim=1
  #pragma HLS array_partition variable=v27 complete dim=2

  l_S_m_0_m1: for (int m1 = 0; m1 < 6; m1++) {	// L34
  #pragma HLS pipeline II=1
    int8_t v35 = v30.read();	// L35
    int8_t a1;	// L36
    a1 = v35;	// L37
    int32_t v37 = v32.read();	// L38
    int32_t p1;	// L39
    p1 = v37;	// L40
    int32_t v39 = p1;	// L41
    int8_t v40 = a1;	// L42
    int8_t v41 = v27[v28][v29];	// L43
    int16_t v42 = v40;	// L44
    int16_t v43 = v41;	// L45
    int16_t v44 = v42 * v43;	// L46
    ap_int<33> v45 = v39;	// L47
    ap_int<33> v46 = v44;	// L48
    ap_int<33> v47 = v45 + v46;	// L49
    v33.write(v47);	// L50
    int8_t v48 = a1;	// L51
    v31.write(v48);	// L52
  }
}

void mac_r2(
  int8_t v49[4][4],
  int v50,
  int v51,
  hls::stream< int8_t >& v52,
  hls::stream< int8_t >& v53,
  hls::stream< int32_t >& v54
) {	// L56
  #pragma HLS array_partition variable=v49 complete dim=1
  #pragma HLS array_partition variable=v49 complete dim=2

  l_S_m_0_m2: for (int m2 = 0; m2 < 6; m2++) {	// L57
  #pragma HLS pipeline II=1
    int8_t v56 = v52.read();	// L58
    int8_t a2;	// L59
    a2 = v56;	// L60
    int32_t p2;	// L62
    p2 = 0;	// L63
    int32_t v59 = p2;	// L64
    int8_t v60 = a2;	// L65
    int8_t v61 = v49[v50][v51];	// L66
    int16_t v62 = v60;	// L67
    int16_t v63 = v61;	// L68
    int16_t v64 = v62 * v63;	// L69
    ap_int<33> v65 = v59;	// L70
    ap_int<33> v66 = v64;	// L71
    ap_int<33> v67 = v65 + v66;	// L72
    v54.write(v67);	// L73
    int8_t v68 = a2;	// L74
    v53.write(v68);	// L75
  }
}

void mac_r3(
  int8_t v69[4][4],
  int v70,
  int v71,
  hls::stream< int8_t >& v72,
  hls::stream< int32_t >& v73,
  hls::stream< int32_t >& v74
) {	// L79
  #pragma HLS array_partition variable=v69 complete dim=1
  #pragma HLS array_partition variable=v69 complete dim=2

  l_S_m_0_m3: for (int m3 = 0; m3 < 6; m3++) {	// L80
  #pragma HLS pipeline II=1
    int8_t v76 = v72.read();	// L81
    int8_t a3;	// L82
    a3 = v76;	// L83
    int32_t v78 = v73.read();	// L84
    int32_t p3;	// L85
    p3 = v78;	// L86
    int32_t v80 = p3;	// L87
    int8_t v81 = a3;	// L88
    int8_t v82 = v69[v70][v71];	// L89
    int16_t v83 = v81;	// L90
    int16_t v84 = v82;	// L91
    int16_t v85 = v83 * v84;	// L92
    ap_int<33> v86 = v80;	// L93
    ap_int<33> v87 = v85;	// L94
    ap_int<33> v88 = v86 + v87;	// L95
    v74.write(v88);	// L96
  }
}

void mac_r4(
  int8_t v89[4][4],
  int v90,
  int v91,
  hls::stream< int8_t >& v92,
  hls::stream< int8_t >& v93,
  hls::stream< int32_t >& v94,
  hls::stream< int32_t >& v95
) {	// L100
  #pragma HLS array_partition variable=v89 complete dim=1
  #pragma HLS array_partition variable=v89 complete dim=2

  l_S_m_0_m4: for (int m4 = 0; m4 < 6; m4++) {	// L101
  #pragma HLS pipeline II=1
    int8_t v97 = v92.read();	// L102
    int8_t a4;	// L103
    a4 = v97;	// L104
    int32_t v99 = v94.read();	// L105
    int32_t p4;	// L106
    p4 = v99;	// L107
    int32_t v101 = p4;	// L108
    int8_t v102 = a4;	// L109
    int8_t v103 = v89[v90][v91];	// L110
    int16_t v104 = v102;	// L111
    int16_t v105 = v103;	// L112
    int16_t v106 = v104 * v105;	// L113
    ap_int<33> v107 = v101;	// L114
    ap_int<33> v108 = v106;	// L115
    ap_int<33> v109 = v107 + v108;	// L116
    v95.write(v109);	// L117
    int8_t v110 = a4;	// L118
    v93.write(v110);	// L119
  }
}

void mac_r5(
  int8_t v111[4][4],
  int v112,
  int v113,
  hls::stream< int8_t >& v114,
  hls::stream< int32_t >& v115,
  hls::stream< int32_t >& v116
) {	// L123
  #pragma HLS array_partition variable=v111 complete dim=1
  #pragma HLS array_partition variable=v111 complete dim=2

  l_S_m_0_m5: for (int m5 = 0; m5 < 6; m5++) {	// L124
  #pragma HLS pipeline II=1
    int8_t v118 = v114.read();	// L125
    int8_t a5;	// L126
    a5 = v118;	// L127
    int32_t v120 = v115.read();	// L128
    int32_t p5;	// L129
    p5 = v120;	// L130
    int32_t v122 = p5;	// L131
    int8_t v123 = a5;	// L132
    int8_t v124 = v111[v112][v113];	// L133
    int16_t v125 = v123;	// L134
    int16_t v126 = v124;	// L135
    int16_t v127 = v125 * v126;	// L136
    ap_int<33> v128 = v122;	// L137
    ap_int<33> v129 = v127;	// L138
    ap_int<33> v130 = v128 + v129;	// L139
    v116.write(v130);	// L140
  }
}

void mac_r6(
  int8_t v131[4][4],
  int v132,
  int v133,
  hls::stream< int8_t >& v134,
  hls::stream< int32_t >& v135
) {	// L144
  #pragma HLS array_partition variable=v131 complete dim=1
  #pragma HLS array_partition variable=v131 complete dim=2

  l_S_m_0_m6: for (int m6 = 0; m6 < 6; m6++) {	// L145
  #pragma HLS pipeline II=1
    int8_t v137 = v134.read();	// L146
    int8_t a6;	// L147
    a6 = v137;	// L148
    int32_t p6;	// L150
    p6 = 0;	// L151
    int32_t v140 = p6;	// L152
    int8_t v141 = a6;	// L153
    int8_t v142 = v131[v132][v133];	// L154
    int16_t v143 = v141;	// L155
    int16_t v144 = v142;	// L156
    int16_t v145 = v143 * v144;	// L157
    ap_int<33> v146 = v140;	// L158
    ap_int<33> v147 = v145;	// L159
    ap_int<33> v148 = v146 + v147;	// L160
    v135.write(v148);	// L161
  }
}

void mac_r7(
  int8_t v149[4][4],
  int v150,
  int v151,
  hls::stream< int8_t >& v152,
  hls::stream< int8_t >& v153,
  hls::stream< int32_t >& v154,
  hls::stream< int32_t >& v155
) {	// L165
  #pragma HLS array_partition variable=v149 complete dim=1
  #pragma HLS array_partition variable=v149 complete dim=2

  l_S_m_0_m7: for (int m7 = 0; m7 < 6; m7++) {	// L166
  #pragma HLS pipeline II=1
    int8_t v157 = v152.read();	// L167
    int8_t a7;	// L168
    a7 = v157;	// L169
    int32_t v159 = v154.read();	// L170
    int32_t p7;	// L171
    p7 = v159;	// L172
    int32_t v161 = p7;	// L173
    int8_t v162 = a7;	// L174
    int8_t v163 = v149[v150][v151];	// L175
    int16_t v164 = v162;	// L176
    int16_t v165 = v163;	// L177
    int16_t v166 = v164 * v165;	// L178
    ap_int<33> v167 = v161;	// L179
    ap_int<33> v168 = v166;	// L180
    ap_int<33> v169 = v167 + v168;	// L181
    v155.write(v169);	// L182
    int8_t v170 = a7;	// L183
    v153.write(v170);	// L184
  }
}

void mac_r8(
  int8_t v171[4][4],
  int v172,
  int v173,
  hls::stream< int8_t >& v174,
  hls::stream< int8_t >& v175,
  hls::stream< int32_t >& v176
) {	// L188
  #pragma HLS array_partition variable=v171 complete dim=1
  #pragma HLS array_partition variable=v171 complete dim=2

  l_S_m_0_m8: for (int m8 = 0; m8 < 6; m8++) {	// L189
  #pragma HLS pipeline II=1
    int8_t v178 = v174.read();	// L190
    int8_t a8;	// L191
    a8 = v178;	// L192
    int32_t p8;	// L194
    p8 = 0;	// L195
    int32_t v181 = p8;	// L196
    int8_t v182 = a8;	// L197
    int8_t v183 = v171[v172][v173];	// L198
    int16_t v184 = v182;	// L199
    int16_t v185 = v183;	// L200
    int16_t v186 = v184 * v185;	// L201
    ap_int<33> v187 = v181;	// L202
    ap_int<33> v188 = v186;	// L203
    ap_int<33> v189 = v187 + v188;	// L204
    v176.write(v189);	// L205
    int8_t v190 = a8;	// L206
    v175.write(v190);	// L207
  }
}

void act_r0(
  int v191,
  hls::stream< int8_t >& v192,
  hls::stream< int32_t >& v193
) {	// L211
  l_S_m_0_m9: for (int m9 = 0; m9 < 6; m9++) {	// L212
  #pragma HLS pipeline II=1
    int32_t v195 = v193.read();	// L213
    int32_t z;	// L214
    z = v195;	// L215
    int32_t v197 = z;	// L216
    bool v198 = v197 < 0;	// L218
    if (v198) {	// L219
      z = 0;	// L220
    }
    int32_t v199 = z;	// L222
    int32_t v200 = v199 >> 4;	// L224
    int8_t v201 = v200;	// L225
    int8_t y;	// L226
    y = v201;	// L227
    int8_t v203 = y;	// L228
    v192.write(v203);	// L229
  }
}

void act_y_out_drain(
  int8_t v204[6][4],
  int v205,
  hls::stream< int8_t >& v206
) {	// L233
  #pragma HLS array_partition variable=v204 complete dim=1
  #pragma HLS array_partition variable=v204 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 6; _t1++) {	// L234
  #pragma HLS pipeline II=1
    int8_t v208 = v206.read();	// L235
    v204[_t1][v205] = v208;	// L236
  }
}

/// This is top function.
void top(
  int8_t v209[6][4],
  int8_t v210[4][4],
  int8_t v211[6][4]
) {	// L240
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v209 complete dim=1
  #pragma HLS array_partition variable=v209 complete dim=2

  #pragma HLS array_partition variable=v210 complete dim=1
  #pragma HLS array_partition variable=v210 complete dim=2

  #pragma HLS array_partition variable=v211 complete dim=1
  #pragma HLS array_partition variable=v211 complete dim=2

  hls::stream< int8_t > v212;
  #pragma HLS stream variable=v212 depth=6	// L241
  hls::stream< int8_t > v213;
  #pragma HLS stream variable=v213 depth=6	// L242
  hls::stream< int8_t > v214;
  #pragma HLS stream variable=v214 depth=6	// L243
  hls::stream< int8_t > v215;
  #pragma HLS stream variable=v215 depth=6	// L244
  hls::stream< int32_t > v216;
  #pragma HLS stream variable=v216 depth=2	// L245
  hls::stream< int32_t > v217;
  #pragma HLS stream variable=v217 depth=2	// L246
  hls::stream< int8_t > v218;
  #pragma HLS stream variable=v218 depth=2	// L247
  hls::stream< int32_t > v219;
  #pragma HLS stream variable=v219 depth=2	// L248
  hls::stream< int8_t > v220;
  #pragma HLS stream variable=v220 depth=2	// L249
  hls::stream< int32_t > v221;
  #pragma HLS stream variable=v221 depth=2	// L250
  hls::stream< int8_t > v222;
  #pragma HLS stream variable=v222 depth=2	// L251
  hls::stream< int32_t > v223;
  #pragma HLS stream variable=v223 depth=2	// L252
  hls::stream< int32_t > v224;
  #pragma HLS stream variable=v224 depth=2	// L253
  hls::stream< int8_t > v225;
  #pragma HLS stream variable=v225 depth=2	// L254
  hls::stream< int32_t > v226;
  #pragma HLS stream variable=v226 depth=2	// L255
  hls::stream< int8_t > v227;
  #pragma HLS stream variable=v227 depth=2	// L256
  hls::stream< int32_t > v228;
  #pragma HLS stream variable=v228 depth=2	// L257
  hls::stream< int8_t > v229;
  #pragma HLS stream variable=v229 depth=2	// L258
  hls::stream< int32_t > v230;
  #pragma HLS stream variable=v230 depth=2	// L259
  hls::stream< int32_t > v231;
  #pragma HLS stream variable=v231 depth=2	// L260
  hls::stream< int8_t > v232;
  #pragma HLS stream variable=v232 depth=2	// L261
  hls::stream< int32_t > v233;
  #pragma HLS stream variable=v233 depth=2	// L262
  hls::stream< int8_t > v234;
  #pragma HLS stream variable=v234 depth=2	// L263
  hls::stream< int32_t > v235;
  #pragma HLS stream variable=v235 depth=2	// L264
  hls::stream< int8_t > v236;
  #pragma HLS stream variable=v236 depth=2	// L265
  hls::stream< int32_t > v237;
  #pragma HLS stream variable=v237 depth=2	// L266
  hls::stream< int32_t > v238;
  #pragma HLS stream variable=v238 depth=2	// L267
  hls::stream< int8_t > v239;
  #pragma HLS stream variable=v239 depth=2	// L268
  hls::stream< int32_t > v240;
  #pragma HLS stream variable=v240 depth=2	// L269
  hls::stream< int8_t > v241;
  #pragma HLS stream variable=v241 depth=2	// L270
  hls::stream< int32_t > v242;
  #pragma HLS stream variable=v242 depth=2	// L271
  hls::stream< int8_t > v243;
  #pragma HLS stream variable=v243 depth=2	// L272
  hls::stream< int8_t > v244;
  #pragma HLS stream variable=v244 depth=6	// L273
  hls::stream< int8_t > v245;
  #pragma HLS stream variable=v245 depth=6	// L275
  hls::stream< int8_t > v246;
  #pragma HLS stream variable=v246 depth=6	// L277
  hls::stream< int8_t > v247;
  #pragma HLS stream variable=v247 depth=6	// L279
  mac_a_in_load(v209, 0, v247);	// L281
  mac_a_in_load(v209, 1, v246);	// L282
  mac_a_in_load(v209, 2, v245);	// L283
  mac_a_in_load(v209, 3, v244);	// L284
  mac_r8(v210, 0, 0, v247, v243, v242);	// L285
  mac_r2(v210, 0, 1, v243, v241, v240);	// L286
  mac_r2(v210, 0, 2, v241, v239, v238);	// L287
  mac_r6(v210, 0, 3, v239, v237);	// L288
  mac_r4(v210, 1, 0, v246, v236, v242, v235);	// L289
  mac_r0(v210, 1, 1, v236, v234, v240, v233);	// L290
  mac_r0(v210, 1, 2, v234, v232, v238, v231);	// L291
  mac_r3(v210, 1, 3, v232, v237, v230);	// L292
  mac_r4(v210, 2, 0, v245, v229, v235, v228);	// L293
  mac_r0(v210, 2, 1, v229, v227, v233, v226);	// L294
  mac_r0(v210, 2, 2, v227, v225, v231, v224);	// L295
  mac_r3(v210, 2, 3, v225, v230, v223);	// L296
  mac_r7(v210, 3, 0, v244, v222, v228, v221);	// L297
  mac_r1(v210, 3, 1, v222, v220, v226, v219);	// L298
  mac_r1(v210, 3, 2, v220, v218, v224, v217);	// L299
  mac_r5(v210, 3, 3, v218, v223, v216);	// L300
  act_r0(0, v215, v221);	// L301
  act_r0(1, v214, v219);	// L302
  act_r0(2, v213, v217);	// L303
  act_r0(3, v212, v216);	// L304
  act_y_out_drain(v211, 0, v215);	// L305
  act_y_out_drain(v211, 1, v214);	// L306
  act_y_out_drain(v211, 2, v213);	// L307
  act_y_out_drain(v211, 3, v212);	// L308
}



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
  int8_t v0[6][8],
  int v1,
  int v2,
  hls::stream< int8_t >& v3
) {	// L3
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  l_S__t_0__t: for (int _t = 0; _t < 6; _t++) {	// L4
  #pragma HLS pipeline II=1
    int8_t v5 = v0[_t][((v2 * 4) + v1)];	// L5
    v3.write(v5);	// L6
  }
}

void mac_r0(
  int8_t v6[8][2],
  int v7,
  int v8,
  hls::stream< int8_t >& v9,
  hls::stream< int32_t >& v10,
  hls::stream< int32_t >& v11
) {	// L10
  #pragma HLS array_partition variable=v6 complete dim=1
  #pragma HLS array_partition variable=v6 complete dim=2

  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L11
  #pragma HLS pipeline II=1
    int8_t v13 = v9.read();	// L12
    int8_t a;	// L13
    a = v13;	// L14
    int32_t v15 = v10.read();	// L15
    int32_t p;	// L16
    p = v15;	// L17
    int32_t v17 = p;	// L18
    int8_t v18 = a;	// L19
    int8_t v19 = v6[(v7 + ((v8 / 2) * 4))][(v8 % 2)];	// L20
    int16_t v20 = v18;	// L21
    int16_t v21 = v19;	// L22
    int16_t v22 = v20 * v21;	// L23
    ap_int<33> v23 = v17;	// L24
    ap_int<33> v24 = v22;	// L25
    ap_int<33> v25 = v23 + v24;	// L26
    v11.write(v25);	// L27
  }
}

void mac_r1(
  int8_t v26[8][2],
  int v27,
  int v28,
  hls::stream< int8_t >& v29,
  hls::stream< int8_t >& v30,
  hls::stream< int32_t >& v31,
  hls::stream< int32_t >& v32
) {	// L31
  #pragma HLS array_partition variable=v26 complete dim=1
  #pragma HLS array_partition variable=v26 complete dim=2

  l_S_m_0_m1: for (int m1 = 0; m1 < 6; m1++) {	// L32
  #pragma HLS pipeline II=1
    int8_t v34 = v29.read();	// L33
    int8_t a1;	// L34
    a1 = v34;	// L35
    int32_t v36 = v31.read();	// L36
    int32_t p1;	// L37
    p1 = v36;	// L38
    int32_t v38 = p1;	// L39
    int8_t v39 = a1;	// L40
    int8_t v40 = v26[(v27 + ((v28 / 2) * 4))][(v28 % 2)];	// L41
    int16_t v41 = v39;	// L42
    int16_t v42 = v40;	// L43
    int16_t v43 = v41 * v42;	// L44
    ap_int<33> v44 = v38;	// L45
    ap_int<33> v45 = v43;	// L46
    ap_int<33> v46 = v44 + v45;	// L47
    v32.write(v46);	// L48
    int8_t v47 = a1;	// L49
    v30.write(v47);	// L50
  }
}

void mac_r2(
  int8_t v48[8][2],
  int v49,
  int v50,
  hls::stream< int8_t >& v51,
  hls::stream< int32_t >& v52,
  hls::stream< int32_t >& v53
) {	// L54
  #pragma HLS array_partition variable=v48 complete dim=1
  #pragma HLS array_partition variable=v48 complete dim=2

  l_S_m_0_m2: for (int m2 = 0; m2 < 6; m2++) {	// L55
  #pragma HLS pipeline II=1
    int8_t v55 = v51.read();	// L56
    int8_t a2;	// L57
    a2 = v55;	// L58
    int32_t v57 = v52.read();	// L59
    int32_t p2;	// L60
    p2 = v57;	// L61
    int32_t v59 = p2;	// L62
    int8_t v60 = a2;	// L63
    int8_t v61 = v48[(v49 + ((v50 / 2) * 4))][(v50 % 2)];	// L64
    int16_t v62 = v60;	// L65
    int16_t v63 = v61;	// L66
    int16_t v64 = v62 * v63;	// L67
    ap_int<33> v65 = v59;	// L68
    ap_int<33> v66 = v64;	// L69
    ap_int<33> v67 = v65 + v66;	// L70
    v53.write(v67);	// L71
  }
}

void mac_r3(
  int8_t v68[8][2],
  int v69,
  int v70,
  hls::stream< int8_t >& v71,
  hls::stream< int32_t >& v72
) {	// L75
  #pragma HLS array_partition variable=v68 complete dim=1
  #pragma HLS array_partition variable=v68 complete dim=2

  l_S_m_0_m3: for (int m3 = 0; m3 < 6; m3++) {	// L76
  #pragma HLS pipeline II=1
    int8_t v74 = v71.read();	// L77
    int8_t a3;	// L78
    a3 = v74;	// L79
    int32_t p3;	// L81
    p3 = 0;	// L82
    int32_t v77 = p3;	// L83
    int8_t v78 = a3;	// L84
    int8_t v79 = v68[(v69 + ((v70 / 2) * 4))][(v70 % 2)];	// L85
    int16_t v80 = v78;	// L86
    int16_t v81 = v79;	// L87
    int16_t v82 = v80 * v81;	// L88
    ap_int<33> v83 = v77;	// L89
    ap_int<33> v84 = v82;	// L90
    ap_int<33> v85 = v83 + v84;	// L91
    v72.write(v85);	// L92
  }
}

void mac_r4(
  int8_t v86[8][2],
  int v87,
  int v88,
  hls::stream< int8_t >& v89,
  hls::stream< int8_t >& v90,
  hls::stream< int32_t >& v91,
  hls::stream< int32_t >& v92
) {	// L96
  #pragma HLS array_partition variable=v86 complete dim=1
  #pragma HLS array_partition variable=v86 complete dim=2

  l_S_m_0_m4: for (int m4 = 0; m4 < 6; m4++) {	// L97
  #pragma HLS pipeline II=1
    int8_t v94 = v89.read();	// L98
    int8_t a4;	// L99
    a4 = v94;	// L100
    int32_t v96 = v91.read();	// L101
    int32_t p4;	// L102
    p4 = v96;	// L103
    int32_t v98 = p4;	// L104
    int8_t v99 = a4;	// L105
    int8_t v100 = v86[(v87 + ((v88 / 2) * 4))][(v88 % 2)];	// L106
    int16_t v101 = v99;	// L107
    int16_t v102 = v100;	// L108
    int16_t v103 = v101 * v102;	// L109
    ap_int<33> v104 = v98;	// L110
    ap_int<33> v105 = v103;	// L111
    ap_int<33> v106 = v104 + v105;	// L112
    v92.write(v106);	// L113
    int8_t v107 = a4;	// L114
    v90.write(v107);	// L115
  }
}

void mac_r5(
  int8_t v108[8][2],
  int v109,
  int v110,
  hls::stream< int8_t >& v111,
  hls::stream< int8_t >& v112,
  hls::stream< int32_t >& v113
) {	// L119
  #pragma HLS array_partition variable=v108 complete dim=1
  #pragma HLS array_partition variable=v108 complete dim=2

  l_S_m_0_m5: for (int m5 = 0; m5 < 6; m5++) {	// L120
  #pragma HLS pipeline II=1
    int8_t v115 = v111.read();	// L121
    int8_t a5;	// L122
    a5 = v115;	// L123
    int32_t p5;	// L125
    p5 = 0;	// L126
    int32_t v118 = p5;	// L127
    int8_t v119 = a5;	// L128
    int8_t v120 = v108[(v109 + ((v110 / 2) * 4))][(v110 % 2)];	// L129
    int16_t v121 = v119;	// L130
    int16_t v122 = v120;	// L131
    int16_t v123 = v121 * v122;	// L132
    ap_int<33> v124 = v118;	// L133
    ap_int<33> v125 = v123;	// L134
    ap_int<33> v126 = v124 + v125;	// L135
    v113.write(v126);	// L136
    int8_t v127 = a5;	// L137
    v112.write(v127);	// L138
  }
}

void act_r0(
  int v128,
  hls::stream< int8_t >& v129,
  hls::stream< int32_t >& v130
) {	// L142
  l_S_m_0_m6: for (int m6 = 0; m6 < 6; m6++) {	// L143
  #pragma HLS pipeline II=1
    int32_t v132 = v130.read();	// L144
    int32_t z;	// L145
    z = v132;	// L146
    int32_t v134 = z;	// L147
    bool v135 = v134 < 0;	// L149
    if (v135) {	// L150
      z = 0;	// L151
    }
    int32_t v136 = z;	// L153
    int32_t v137 = v136 >> 2;	// L155
    int8_t v138 = v137;	// L156
    int8_t y;	// L157
    y = v138;	// L158
    int8_t v140 = y;	// L159
    v129.write(v140);	// L160
  }
}

void act_y_out_drain(
  int8_t v141[6][2],
  int v142,
  hls::stream< int8_t >& v143
) {	// L164
  #pragma HLS array_partition variable=v141 complete dim=1
  #pragma HLS array_partition variable=v141 complete dim=2

  l_S__t_0__t1: for (int _t1 = 0; _t1 < 6; _t1++) {	// L165
  #pragma HLS pipeline II=1
    int8_t v145 = v143.read();	// L166
    v141[_t1][v142] = v145;	// L167
  }
}

/// This is top function.
void top(
  int8_t v146[6][8],
  int8_t v147[8][2],
  int8_t v148[6][2]
) {	// L171
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v146 complete dim=1
  #pragma HLS array_partition variable=v146 complete dim=2

  #pragma HLS array_partition variable=v147 complete dim=1
  #pragma HLS array_partition variable=v147 complete dim=2

  #pragma HLS array_partition variable=v148 complete dim=1
  #pragma HLS array_partition variable=v148 complete dim=2

  hls::stream< int8_t > v149;
  #pragma HLS stream variable=v149 depth=6	// L172
  hls::stream< int8_t > v150;
  #pragma HLS stream variable=v150 depth=6	// L173
  hls::stream< int32_t > v151;
  #pragma HLS stream variable=v151 depth=2	// L174
  hls::stream< int32_t > v152;
  #pragma HLS stream variable=v152 depth=2	// L175
  hls::stream< int8_t > v153;
  #pragma HLS stream variable=v153 depth=2	// L176
  hls::stream< int8_t > v154;
  #pragma HLS stream variable=v154 depth=2	// L177
  hls::stream< int32_t > v155;
  #pragma HLS stream variable=v155 depth=2	// L178
  hls::stream< int32_t > v156;
  #pragma HLS stream variable=v156 depth=2	// L179
  hls::stream< int8_t > v157;
  #pragma HLS stream variable=v157 depth=2	// L180
  hls::stream< int32_t > v158;
  #pragma HLS stream variable=v158 depth=2	// L181
  hls::stream< int32_t > v159;
  #pragma HLS stream variable=v159 depth=2	// L182
  hls::stream< int8_t > v160;
  #pragma HLS stream variable=v160 depth=2	// L183
  hls::stream< int32_t > v161;
  #pragma HLS stream variable=v161 depth=2	// L184
  hls::stream< int32_t > v162;
  #pragma HLS stream variable=v162 depth=2	// L185
  hls::stream< int8_t > v163;
  #pragma HLS stream variable=v163 depth=2	// L186
  hls::stream< int32_t > v164;
  #pragma HLS stream variable=v164 depth=2	// L187
  hls::stream< int32_t > v165;
  #pragma HLS stream variable=v165 depth=2	// L188
  hls::stream< int8_t > v166;
  #pragma HLS stream variable=v166 depth=2	// L189
  hls::stream< int32_t > v167;
  #pragma HLS stream variable=v167 depth=2	// L190
  hls::stream< int32_t > v168;
  #pragma HLS stream variable=v168 depth=2	// L191
  hls::stream< int32_t > v169;
  #pragma HLS stream variable=v169 depth=2	// L192
  hls::stream< int32_t > v170;
  #pragma HLS stream variable=v170 depth=2	// L193
  hls::stream< int8_t > v171;
  #pragma HLS stream variable=v171 depth=2	// L194
  hls::stream< int32_t > v172;
  #pragma HLS stream variable=v172 depth=2	// L195
  hls::stream< int32_t > v173;
  #pragma HLS stream variable=v173 depth=2	// L196
  hls::stream< int8_t > v174;
  #pragma HLS stream variable=v174 depth=2	// L197
  hls::stream< int8_t > v175;
  #pragma HLS stream variable=v175 depth=6	// L198
  hls::stream< int8_t > v176;
  #pragma HLS stream variable=v176 depth=6	// L199
  hls::stream< int8_t > v177;
  #pragma HLS stream variable=v177 depth=6	// L201
  hls::stream< int8_t > v178;
  #pragma HLS stream variable=v178 depth=6	// L202
  hls::stream< int8_t > v179;
  #pragma HLS stream variable=v179 depth=6	// L204
  hls::stream< int8_t > v180;
  #pragma HLS stream variable=v180 depth=6	// L205
  hls::stream< int8_t > v181;
  #pragma HLS stream variable=v181 depth=6	// L206
  hls::stream< int8_t > v182;
  #pragma HLS stream variable=v182 depth=6	// L208
  mac_a_in_load(v146, 0, 0, v182);	// L210
  mac_a_in_load(v146, 0, 1, v181);	// L211
  mac_a_in_load(v146, 1, 0, v180);	// L212
  mac_a_in_load(v146, 1, 1, v179);	// L213
  mac_a_in_load(v146, 2, 0, v178);	// L214
  mac_a_in_load(v146, 2, 1, v177);	// L215
  mac_a_in_load(v146, 3, 0, v176);	// L216
  mac_a_in_load(v146, 3, 1, v175);	// L217
  mac_r5(v147, 0, 0, v182, v174, v173);	// L218
  mac_r3(v147, 0, 1, v174, v172);	// L219
  mac_r1(v147, 0, 2, v181, v171, v170, v169);	// L220
  mac_r0(v147, 0, 3, v171, v168, v167);	// L221
  mac_r1(v147, 1, 0, v180, v166, v173, v165);	// L222
  mac_r0(v147, 1, 1, v166, v172, v164);	// L223
  mac_r1(v147, 1, 2, v179, v163, v169, v162);	// L224
  mac_r0(v147, 1, 3, v163, v167, v161);	// L225
  mac_r1(v147, 2, 0, v178, v160, v165, v159);	// L226
  mac_r0(v147, 2, 1, v160, v164, v158);	// L227
  mac_r1(v147, 2, 2, v177, v157, v162, v156);	// L228
  mac_r0(v147, 2, 3, v157, v161, v155);	// L229
  mac_r1(v147, 3, 0, v176, v154, v159, v170);	// L230
  mac_r0(v147, 3, 1, v154, v158, v168);	// L231
  mac_r4(v147, 3, 2, v175, v153, v156, v152);	// L232
  mac_r2(v147, 3, 3, v153, v155, v151);	// L233
  act_r0(0, v150, v152);	// L234
  act_r0(1, v149, v151);	// L235
  act_y_out_drain(v148, 0, v150);	// L236
  act_y_out_drain(v148, 1, v149);	// L237
}


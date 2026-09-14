
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
int32_t _tab0[4][1][1] = {0, 2, 1, 3};	// L3
void bfly_up_in_load(
  float v0[8][2],
  int v1,
  hls::stream< hls::vector< float, 2 > >& v2
) {	// L4
  #pragma HLS array_partition variable=v0 complete dim=1
  #pragma HLS array_partition variable=v0 complete dim=2

  // placeholder for const int32_t _tab0	// L5
  l_S__t_0__t: for (int _t = 0; _t < 1; _t++) {	// L6
    float _blk[2];	// L9
    for (int v6 = 0; v6 < 2; v6++) {	// L10
      _blk[v6] = (float)0.000000;	// L10
    }
    l_S__b0_0__b0: for (int _b0 = 0; _b0 < 2; _b0++) {	// L11
    #pragma HLS pipeline II=1
      int32_t v8 = _tab0[v1][_t][0];	// L12
      int v9 = v8;	// L13
      float v10 = v0[v9][_b0];	// L14
      _blk[_b0] = v10;	// L15
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk[_iv0];
      }
      v2.write(_vec);
    }	// L17
  }
}

int32_t _tab0_0[4][1][1] = {4, 6, 5, 7};	// L21
void bfly_lo_in_load(
  float v11[8][2],
  int v12,
  hls::stream< hls::vector< float, 2 > >& v13
) {	// L22
  #pragma HLS array_partition variable=v11 complete dim=1
  #pragma HLS array_partition variable=v11 complete dim=2

  // placeholder for const int32_t _tab0_0	// L23
  l_S__t_0__t1: for (int _t1 = 0; _t1 < 1; _t1++) {	// L24
    float _blk1[2];	// L27
    for (int v17 = 0; v17 < 2; v17++) {	// L28
      _blk1[v17] = (float)0.000000;	// L28
    }
    l_S__b0_0__b01: for (int _b01 = 0; _b01 < 2; _b01++) {	// L29
    #pragma HLS pipeline II=1
      int32_t v19 = _tab0_0[v12][_t1][0];	// L30
      int v20 = v19;	// L31
      float v21 = v11[v20][_b01];	// L32
      _blk1[_b01] = v21;	// L33
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = _blk1[_iv0];
      }
      v13.write(_vec);
    }	// L35
  }
}

float _st_tw[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L39
void bfly_r0(
  int v22,
  int v23,
  hls::stream< hls::vector< float, 2 > >& v24,
  hls::stream< hls::vector< float, 2 > >& v25,
  hls::stream< hls::vector< float, 2 > >& v26,
  hls::stream< hls::vector< float, 2 > >& v27
) {	// L40
  // placeholder for const float _st_tw	// L41
  int32_t v29 = v22;	// L42
  int32_t v30 = 1 << v29;	// L44
  int32_t span;	// L45
  span = v30;	// L46
  int32_t v32 = span;	// L47
  ap_int<33> v33 = v23;	// L48
  ap_int<33> v34 = v32;	// L49
  ap_int<33> v35 = v33 % v34;	// L50
  int32_t v36 = 4 / v32;	// L52
  ap_int<65> v37 = v35;	// L53
  ap_int<65> v38 = v36;	// L54
  ap_int<65> v39 = v37 * v38;	// L55
  ap_int<65> k;	// L56
  k = v39;	// L57
  ap_int<65> v41 = k;	// L58
  int v42 = v41;	// L59
  float v43 = _st_tw[v42][0];	// L62
  float wr;	// L63
  wr = v43;	// L64
  ap_int<65> v45 = k;	// L65
  int v46 = v45;	// L66
  float v47 = _st_tw[v46][1];	// L68
  float wi;	// L69
  wi = v47;	// L70
  float v49[2];
  {
    hls::vector< float, 2 > _vec = v26.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v49[_iv0] = _vec[_iv0];
    }
  }	// L71
  float v50[2];
  {
    hls::vector< float, 2 > _vec = v24.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v50[_iv0] = _vec[_iv0];
    }
  }	// L72
  float v51 = wr;	// L73
  float v52 = v50[0];	// L74
  float v53 = v51 * v52;	// L75
  float v54 = wi;	// L76
  float v55 = v50[1];	// L77
  float v56 = v54 * v55;	// L78
  float v57 = v53 - v56;	// L79
  float tr;	// L80
  tr = v57;	// L81
  float v59 = wr;	// L82
  float v60 = v50[1];	// L83
  float v61 = v59 * v60;	// L84
  float v62 = wi;	// L85
  float v63 = v50[0];	// L86
  float v64 = v62 * v63;	// L87
  float v65 = v61 + v64;	// L88
  float ti;	// L89
  ti = v65;	// L90
  float u[2];	// L92
  for (int v68 = 0; v68 < 2; v68++) {	// L93
    u[v68] = (float)0.000000;	// L93
  }
  float l[2];	// L94
  for (int v70 = 0; v70 < 2; v70++) {	// L95
    l[v70] = (float)0.000000;	// L95
  }
  float v71 = v49[0];	// L96
  float v72 = tr;	// L97
  float v73 = v71 + v72;	// L98
  u[0] = v73;	// L99
  float v74 = v49[1];	// L100
  float v75 = ti;	// L101
  float v76 = v74 + v75;	// L102
  u[1] = v76;	// L103
  float v77 = v49[0];	// L104
  float v78 = tr;	// L105
  float v79 = v77 - v78;	// L106
  l[0] = v79;	// L107
  float v80 = v49[1];	// L108
  float v81 = ti;	// L109
  float v82 = v80 - v81;	// L110
  l[1] = v82;	// L111
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u[_iv0];
    }
    v27.write(_vec);
  }	// L112
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l[_iv0];
    }
    v25.write(_vec);
  }	// L113
}

float _st_tw_0[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L116
void bfly_r1(
  int v83,
  int v84,
  hls::stream< hls::vector< float, 2 > >& v85,
  hls::stream< hls::vector< float, 2 > >& v86,
  hls::stream< hls::vector< float, 2 > >& v87,
  hls::stream< hls::vector< float, 2 > >& v88
) {	// L117
  // placeholder for const float _st_tw_0	// L118
  int32_t v90 = v83;	// L119
  int32_t v91 = 1 << v90;	// L121
  int32_t span1;	// L122
  span1 = v91;	// L123
  int32_t v93 = span1;	// L124
  ap_int<33> v94 = v84;	// L125
  ap_int<33> v95 = v93;	// L126
  ap_int<33> v96 = v94 % v95;	// L127
  int32_t v97 = 4 / v93;	// L129
  ap_int<65> v98 = v96;	// L130
  ap_int<65> v99 = v97;	// L131
  ap_int<65> v100 = v98 * v99;	// L132
  ap_int<65> k1;	// L133
  k1 = v100;	// L134
  ap_int<65> v102 = k1;	// L135
  int v103 = v102;	// L136
  float v104 = _st_tw_0[v103][0];	// L139
  float wr1;	// L140
  wr1 = v104;	// L141
  ap_int<65> v106 = k1;	// L142
  int v107 = v106;	// L143
  float v108 = _st_tw_0[v107][1];	// L145
  float wi1;	// L146
  wi1 = v108;	// L147
  float v110[2];
  {
    hls::vector< float, 2 > _vec = v87.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v110[_iv0] = _vec[_iv0];
    }
  }	// L148
  float v111[2];
  {
    hls::vector< float, 2 > _vec = v85.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v111[_iv0] = _vec[_iv0];
    }
  }	// L149
  float v112 = wr1;	// L150
  float v113 = v111[0];	// L151
  float v114 = v112 * v113;	// L152
  float v115 = wi1;	// L153
  float v116 = v111[1];	// L154
  float v117 = v115 * v116;	// L155
  float v118 = v114 - v117;	// L156
  float tr1;	// L157
  tr1 = v118;	// L158
  float v120 = wr1;	// L159
  float v121 = v111[1];	// L160
  float v122 = v120 * v121;	// L161
  float v123 = wi1;	// L162
  float v124 = v111[0];	// L163
  float v125 = v123 * v124;	// L164
  float v126 = v122 + v125;	// L165
  float ti1;	// L166
  ti1 = v126;	// L167
  float u1[2];	// L169
  for (int v129 = 0; v129 < 2; v129++) {	// L170
    u1[v129] = (float)0.000000;	// L170
  }
  float l1[2];	// L171
  for (int v131 = 0; v131 < 2; v131++) {	// L172
    l1[v131] = (float)0.000000;	// L172
  }
  float v132 = v110[0];	// L173
  float v133 = tr1;	// L174
  float v134 = v132 + v133;	// L175
  u1[0] = v134;	// L176
  float v135 = v110[1];	// L177
  float v136 = ti1;	// L178
  float v137 = v135 + v136;	// L179
  u1[1] = v137;	// L180
  float v138 = v110[0];	// L181
  float v139 = tr1;	// L182
  float v140 = v138 - v139;	// L183
  l1[0] = v140;	// L184
  float v141 = v110[1];	// L185
  float v142 = ti1;	// L186
  float v143 = v141 - v142;	// L187
  l1[1] = v143;	// L188
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u1[_iv0];
    }
    v88.write(_vec);
  }	// L189
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l1[_iv0];
    }
    v86.write(_vec);
  }	// L190
}

float _st_tw_1[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L193
void bfly_r2(
  int v144,
  int v145,
  hls::stream< hls::vector< float, 2 > >& v146,
  hls::stream< hls::vector< float, 2 > >& v147,
  hls::stream< hls::vector< float, 2 > >& v148,
  hls::stream< hls::vector< float, 2 > >& v149
) {	// L194
  // placeholder for const float _st_tw_1	// L195
  int32_t v151 = v144;	// L196
  int32_t v152 = 1 << v151;	// L198
  int32_t span2;	// L199
  span2 = v152;	// L200
  int32_t v154 = span2;	// L201
  ap_int<33> v155 = v145;	// L202
  ap_int<33> v156 = v154;	// L203
  ap_int<33> v157 = v155 % v156;	// L204
  int32_t v158 = 4 / v154;	// L206
  ap_int<65> v159 = v157;	// L207
  ap_int<65> v160 = v158;	// L208
  ap_int<65> v161 = v159 * v160;	// L209
  ap_int<65> k2;	// L210
  k2 = v161;	// L211
  ap_int<65> v163 = k2;	// L212
  int v164 = v163;	// L213
  float v165 = _st_tw_1[v164][0];	// L216
  float wr2;	// L217
  wr2 = v165;	// L218
  ap_int<65> v167 = k2;	// L219
  int v168 = v167;	// L220
  float v169 = _st_tw_1[v168][1];	// L222
  float wi2;	// L223
  wi2 = v169;	// L224
  float v171[2];
  {
    hls::vector< float, 2 > _vec = v148.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v171[_iv0] = _vec[_iv0];
    }
  }	// L225
  float v172[2];
  {
    hls::vector< float, 2 > _vec = v146.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v172[_iv0] = _vec[_iv0];
    }
  }	// L226
  float v173 = wr2;	// L227
  float v174 = v172[0];	// L228
  float v175 = v173 * v174;	// L229
  float v176 = wi2;	// L230
  float v177 = v172[1];	// L231
  float v178 = v176 * v177;	// L232
  float v179 = v175 - v178;	// L233
  float tr2;	// L234
  tr2 = v179;	// L235
  float v181 = wr2;	// L236
  float v182 = v172[1];	// L237
  float v183 = v181 * v182;	// L238
  float v184 = wi2;	// L239
  float v185 = v172[0];	// L240
  float v186 = v184 * v185;	// L241
  float v187 = v183 + v186;	// L242
  float ti2;	// L243
  ti2 = v187;	// L244
  float u2[2];	// L246
  for (int v190 = 0; v190 < 2; v190++) {	// L247
    u2[v190] = (float)0.000000;	// L247
  }
  float l2[2];	// L248
  for (int v192 = 0; v192 < 2; v192++) {	// L249
    l2[v192] = (float)0.000000;	// L249
  }
  float v193 = v171[0];	// L250
  float v194 = tr2;	// L251
  float v195 = v193 + v194;	// L252
  u2[0] = v195;	// L253
  float v196 = v171[1];	// L254
  float v197 = ti2;	// L255
  float v198 = v196 + v197;	// L256
  u2[1] = v198;	// L257
  float v199 = v171[0];	// L258
  float v200 = tr2;	// L259
  float v201 = v199 - v200;	// L260
  l2[0] = v201;	// L261
  float v202 = v171[1];	// L262
  float v203 = ti2;	// L263
  float v204 = v202 - v203;	// L264
  l2[1] = v204;	// L265
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u2[_iv0];
    }
    v149.write(_vec);
  }	// L266
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l2[_iv0];
    }
    v147.write(_vec);
  }	// L267
}

int32_t _tab0_1[4][1][1] = {0, 1, 2, 3};	// L270
void bfly_up_out_drain(
  float v205[8][2],
  int v206,
  hls::stream< hls::vector< float, 2 > >& v207
) {	// L271
  #pragma HLS array_partition variable=v205 complete dim=1
  #pragma HLS array_partition variable=v205 complete dim=2

  // placeholder for const int32_t _tab0_1	// L272
  l_S__t_0__t2: for (int _t2 = 0; _t2 < 1; _t2++) {	// L273
    float v210[2];
    {
      hls::vector< float, 2 > _vec = v207.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v210[_iv0] = _vec[_iv0];
      }
    }	// L274
    l_S__b0_0__b02: for (int _b02 = 0; _b02 < 2; _b02++) {	// L275
    #pragma HLS pipeline II=1
      float v212 = v210[_b02];	// L276
      int32_t v213 = _tab0_1[v206][_t2][0];	// L277
      int v214 = v213;	// L278
      v205[v214][_b02] = v212;	// L279
    }
  }
}

int32_t _tab0_2[4][1][1] = {4, 5, 6, 7};	// L284
void bfly_lo_out_drain(
  float v215[8][2],
  int v216,
  hls::stream< hls::vector< float, 2 > >& v217
) {	// L285
  #pragma HLS array_partition variable=v215 complete dim=1
  #pragma HLS array_partition variable=v215 complete dim=2

  // placeholder for const int32_t _tab0_2	// L286
  l_S__t_0__t3: for (int _t3 = 0; _t3 < 1; _t3++) {	// L287
    float v220[2];
    {
      hls::vector< float, 2 > _vec = v217.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v220[_iv0] = _vec[_iv0];
      }
    }	// L288
    l_S__b0_0__b03: for (int _b03 = 0; _b03 < 2; _b03++) {	// L289
    #pragma HLS pipeline II=1
      float v222 = v220[_b03];	// L290
      int32_t v223 = _tab0_2[v216][_t3][0];	// L291
      int v224 = v223;	// L292
      v215[v224][_b03] = v222;	// L293
    }
  }
}

/// This is top function.
void top(
  float v225[8][2],
  float v226[8][2]
) {	// L298
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v225 complete dim=1
  #pragma HLS array_partition variable=v225 complete dim=2

  #pragma HLS array_partition variable=v226 complete dim=1
  #pragma HLS array_partition variable=v226 complete dim=2

  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v227;
  #pragma HLS stream variable=v227 depth=4	// L299
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v228;
  #pragma HLS stream variable=v228 depth=4	// L300
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v229;
  #pragma HLS stream variable=v229 depth=4	// L301
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v230;
  #pragma HLS stream variable=v230 depth=4	// L302
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v231;
  #pragma HLS stream variable=v231 depth=4	// L303
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v232;
  #pragma HLS stream variable=v232 depth=4	// L304
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v233;
  #pragma HLS stream variable=v233 depth=4	// L305
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v234;
  #pragma HLS stream variable=v234 depth=4	// L306
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v235;
  #pragma HLS stream variable=v235 depth=2	// L307
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v236;
  #pragma HLS stream variable=v236 depth=2	// L308
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v237;
  #pragma HLS stream variable=v237 depth=2	// L309
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v238;
  #pragma HLS stream variable=v238 depth=2	// L310
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v239;
  #pragma HLS stream variable=v239 depth=2	// L311
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v240;
  #pragma HLS stream variable=v240 depth=2	// L312
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v241;
  #pragma HLS stream variable=v241 depth=2	// L313
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v242;
  #pragma HLS stream variable=v242 depth=2	// L314
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v243;
  #pragma HLS stream variable=v243 depth=2	// L315
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v244;
  #pragma HLS stream variable=v244 depth=2	// L316
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v245;
  #pragma HLS stream variable=v245 depth=2	// L317
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v246;
  #pragma HLS stream variable=v246 depth=2	// L318
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v247;
  #pragma HLS stream variable=v247 depth=2	// L319
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v248;
  #pragma HLS stream variable=v248 depth=2	// L320
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v249;
  #pragma HLS stream variable=v249 depth=2	// L321
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v250;
  #pragma HLS stream variable=v250 depth=2	// L322
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v251;
  #pragma HLS stream variable=v251 depth=4	// L323
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v252;
  #pragma HLS stream variable=v252 depth=4	// L324
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v253;
  #pragma HLS stream variable=v253 depth=4	// L325
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v254;
  #pragma HLS stream variable=v254 depth=4	// L326
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v255;
  #pragma HLS stream variable=v255 depth=4	// L327
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v256;
  #pragma HLS stream variable=v256 depth=4	// L329
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v257;
  #pragma HLS stream variable=v257 depth=4	// L331
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v258;
  #pragma HLS stream variable=v258 depth=4	// L333
  bfly_up_in_load(v225, 0, v258);	// L335
  bfly_up_in_load(v225, 1, v257);	// L336
  bfly_up_in_load(v225, 2, v256);	// L337
  bfly_up_in_load(v225, 3, v255);	// L338
  bfly_lo_in_load(v225, 0, v254);	// L339
  bfly_lo_in_load(v225, 1, v253);	// L340
  bfly_lo_in_load(v225, 2, v252);	// L341
  bfly_lo_in_load(v225, 3, v251);	// L342
  bfly_r2(0, 0, v254, v250, v258, v249);	// L343
  bfly_r2(0, 1, v253, v248, v257, v247);	// L344
  bfly_r2(0, 2, v252, v246, v256, v245);	// L345
  bfly_r2(0, 3, v251, v244, v255, v243);	// L346
  bfly_r0(1, 0, v247, v242, v249, v241);	// L347
  bfly_r0(1, 1, v248, v240, v250, v239);	// L348
  bfly_r0(1, 2, v243, v238, v245, v237);	// L349
  bfly_r0(1, 3, v244, v236, v246, v235);	// L350
  bfly_r1(2, 0, v237, v234, v241, v233);	// L351
  bfly_r1(2, 1, v235, v232, v239, v231);	// L352
  bfly_r1(2, 2, v238, v230, v242, v229);	// L353
  bfly_r1(2, 3, v236, v228, v240, v227);	// L354
  bfly_up_out_drain(v226, 0, v233);	// L355
  bfly_up_out_drain(v226, 1, v231);	// L356
  bfly_up_out_drain(v226, 2, v229);	// L357
  bfly_up_out_drain(v226, 3, v227);	// L358
  bfly_lo_out_drain(v226, 0, v234);	// L359
  bfly_lo_out_drain(v226, 1, v232);	// L360
  bfly_lo_out_drain(v226, 2, v230);	// L361
  bfly_lo_out_drain(v226, 3, v228);	// L362
}


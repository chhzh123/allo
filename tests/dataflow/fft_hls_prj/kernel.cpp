
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
void bit_rev_stage_0(
  float v0[8][32],
  float v1[8][32],
  hls::stream< hls::vector< float, 32 > >& v2,
  hls::stream< hls::vector< float, 32 > >& v3
) {	// L4
  #pragma HLS dataflow disable_start_propagation
  #pragma HLS array_partition variable=v0 complete dim=2

  #pragma HLS array_partition variable=v1 complete dim=2

  float buf_re[32][8];	// L13
  #pragma HLS array_partition variable=buf_re complete dim=1

  #pragma HLS bind_storage variable=buf_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=buf_re inter false
  float buf_im[32][8];	// L14
  #pragma HLS array_partition variable=buf_im complete dim=1

  #pragma HLS bind_storage variable=buf_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=buf_im inter false
  l_S_ii_0_ii: for (int ii = 0; ii < 8; ii++) {	// L15
  #pragma HLS pipeline II=1
    l_S_kk_0_kk: for (int kk = 0; kk < 32; kk++) {	// L16
    #pragma HLS unroll
      int32_t v8 = kk;	// L17
      int32_t v9 = v8 & 1;	// L18
      int32_t v10 = v9 << 4;	// L19
      int32_t v11 = v8 & 2;	// L20
      int32_t v12 = v11 << 2;	// L21
      int32_t v13 = v10 | v12;	// L22
      int32_t v14 = v8 & 4;	// L23
      int32_t v15 = v13 | v14;	// L24
      int32_t v16 = v8 & 8;	// L25
      int32_t v17 = v16 >> 2;	// L26
      int32_t v18 = v15 | v17;	// L27
      int32_t v19 = v8 & 16;	// L28
      int32_t v20 = v19 >> 4;	// L29
      int32_t v21 = v18 | v20;	// L30
      int32_t bank;	// L31
      bank = v21;	// L32
      int32_t v23 = ii;	// L33
      int32_t v24 = v23 & 4;	// L34
      int32_t v25 = v24 >> 2;	// L35
      int32_t v26 = v23 & 2;	// L36
      int32_t v27 = v25 | v26;	// L37
      int32_t v28 = v23 & 1;	// L38
      int32_t v29 = v28 << 2;	// L39
      int32_t v30 = v27 | v29;	// L40
      int32_t offset;	// L41
      offset = v30;	// L42
      float v32 = v0[ii][kk];	// L43
      int32_t v33 = bank;	// L44
      int v34 = v33;	// L45
      int32_t v35 = offset;	// L46
      int v36 = v35;	// L47
      buf_re[v34][v36] = v32;	// L48
      float v37 = v1[ii][kk];	// L49
      int32_t v38 = bank;	// L50
      int v39 = v38;	// L51
      int32_t v40 = offset;	// L52
      int v41 = v40;	// L53
      buf_im[v39][v41] = v37;	// L54
    }
  }
  l_S_jj_2_jj: for (int jj = 0; jj < 8; jj++) {	// L57
  #pragma HLS pipeline II=1
    float chunk_re[32];	// L58
    #pragma HLS array_partition variable=chunk_re complete
    float chunk_im[32];	// L59
    #pragma HLS array_partition variable=chunk_im complete
    l_S_mm_2_mm: for (int mm = 0; mm < 32; mm++) {	// L60
    #pragma HLS unroll
      int v46 = jj << 2;	// L61
      int v47 = mm >> 3;	// L62
      int v48 = v46 | v47;	// L63
      int32_t v49 = v48;	// L64
      int32_t rd_bank;	// L65
      rd_bank = v49;	// L66
      int32_t v51 = mm;	// L67
      int32_t v52 = v51 & 7;	// L68
      int32_t rd_off;	// L69
      rd_off = v52;	// L70
      int32_t v54 = rd_bank;	// L71
      int v55 = v54;	// L72
      int32_t v56 = rd_off;	// L73
      int v57 = v56;	// L74
      float v58 = buf_re[v55][v57];	// L75
      chunk_re[mm] = v58;	// L76
      int32_t v59 = rd_bank;	// L77
      int v60 = v59;	// L78
      int32_t v61 = rd_off;	// L79
      int v62 = v61;	// L80
      float v63 = buf_im[v60][v62];	// L81
      chunk_im[mm] = v63;	// L82
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re[_iv0];
      }
      v2.write(_vec);
    }	// L84
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im[_iv0];
      }
      v3.write(_vec);
    }	// L85
  }
}

void intra_0_0(
  hls::stream< hls::vector< float, 32 > >& v64,
  hls::stream< hls::vector< float, 32 > >& v65,
  hls::stream< hls::vector< float, 32 > >& v66,
  hls::stream< hls::vector< float, 32 > >& v67
) {	// L89
  l_S__i_0__i: for (int _i = 0; _i < 8; _i++) {	// L90
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v69 = v64.read();
    hls::vector< float, 32 > v70 = v65.read();
    float o_re[32];	// L93
    #pragma HLS array_partition variable=o_re complete
    float o_im[32];	// L94
    #pragma HLS array_partition variable=o_im complete
    l_S_k_0_k: for (int k = 0; k < 16; k++) {	// L95
    #pragma HLS unroll
      float v74 = v69[(k * 2)];	// L96
      float a_re;	// L97
      a_re = v74;	// L98
      float v76 = v70[(k * 2)];	// L99
      float a_im;	// L100
      a_im = v76;	// L101
      float v78 = v69[((k * 2) + 1)];	// L102
      float b_re;	// L103
      b_re = v78;	// L104
      float v80 = v70[((k * 2) + 1)];	// L105
      float b_im;	// L106
      b_im = v80;	// L107
      float v82 = a_re;	// L108
      float v83 = b_re;	// L109
      float v84 = v82 + v83;	// L110
      #pragma HLS bind_op variable=v84 op=fadd impl=fabric
      o_re[(k * 2)] = v84;	// L111
      float v85 = a_im;	// L112
      float v86 = b_im;	// L113
      float v87 = v85 + v86;	// L114
      #pragma HLS bind_op variable=v87 op=fadd impl=fabric
      o_im[(k * 2)] = v87;	// L115
      float v88 = a_re;	// L116
      float v89 = b_re;	// L117
      float v90 = v88 - v89;	// L118
      #pragma HLS bind_op variable=v90 op=fsub impl=fabric
      o_re[((k * 2) + 1)] = v90;	// L119
      float v91 = a_im;	// L120
      float v92 = b_im;	// L121
      float v93 = v91 - v92;	// L122
      #pragma HLS bind_op variable=v93 op=fsub impl=fabric
      o_im[((k * 2) + 1)] = v93;	// L123
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re[_iv0];
      }
      v66.write(_vec);
    }	// L125
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im[_iv0];
      }
      v67.write(_vec);
    }	// L126
  }
}

void intra_1_0(
  hls::stream< hls::vector< float, 32 > >& v94,
  hls::stream< hls::vector< float, 32 > >& v95,
  hls::stream< hls::vector< float, 32 > >& v96,
  hls::stream< hls::vector< float, 32 > >& v97
) {	// L130
  l_S__i_0__i1: for (int _i1 = 0; _i1 < 8; _i1++) {	// L131
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v99 = v94.read();
    hls::vector< float, 32 > v100 = v95.read();
    float o_re1[32];	// L134
    #pragma HLS array_partition variable=o_re1 complete
    float o_im1[32];	// L135
    #pragma HLS array_partition variable=o_im1 complete
    l_S_k_0_k1: for (int k1 = 0; k1 < 8; k1++) {	// L136
    #pragma HLS unroll
      float v104 = v99[(k1 * 4)];	// L137
      float a_re1;	// L138
      a_re1 = v104;	// L139
      float v106 = v100[(k1 * 4)];	// L140
      float a_im1;	// L141
      a_im1 = v106;	// L142
      float v108 = v99[((k1 * 4) + 2)];	// L143
      float b_re1;	// L144
      b_re1 = v108;	// L145
      float v110 = v100[((k1 * 4) + 2)];	// L146
      float b_im1;	// L147
      b_im1 = v110;	// L148
      float v112 = a_re1;	// L149
      float v113 = b_re1;	// L150
      float v114 = v112 + v113;	// L151
      #pragma HLS bind_op variable=v114 op=fadd impl=fabric
      o_re1[(k1 * 4)] = v114;	// L152
      float v115 = a_im1;	// L153
      float v116 = b_im1;	// L154
      float v117 = v115 + v116;	// L155
      #pragma HLS bind_op variable=v117 op=fadd impl=fabric
      o_im1[(k1 * 4)] = v117;	// L156
      float v118 = a_re1;	// L157
      float v119 = b_re1;	// L158
      float v120 = v118 - v119;	// L159
      #pragma HLS bind_op variable=v120 op=fsub impl=fabric
      o_re1[((k1 * 4) + 2)] = v120;	// L160
      float v121 = a_im1;	// L161
      float v122 = b_im1;	// L162
      float v123 = v121 - v122;	// L163
      #pragma HLS bind_op variable=v123 op=fsub impl=fabric
      o_im1[((k1 * 4) + 2)] = v123;	// L164
      float v124 = v99[((k1 * 4) + 1)];	// L165
      float v125 = v100[((k1 * 4) + 3)];	// L166
      float v126 = v124 + v125;	// L167
      #pragma HLS bind_op variable=v126 op=fadd impl=fabric
      o_re1[((k1 * 4) + 1)] = v126;	// L168
      float v127 = v100[((k1 * 4) + 1)];	// L169
      float v128 = v99[((k1 * 4) + 3)];	// L170
      float v129 = v127 - v128;	// L171
      #pragma HLS bind_op variable=v129 op=fsub impl=fabric
      o_im1[((k1 * 4) + 1)] = v129;	// L172
      float v130 = v99[((k1 * 4) + 1)];	// L173
      float v131 = v100[((k1 * 4) + 3)];	// L174
      float v132 = v130 - v131;	// L175
      #pragma HLS bind_op variable=v132 op=fsub impl=fabric
      o_re1[((k1 * 4) + 3)] = v132;	// L176
      float v133 = v100[((k1 * 4) + 1)];	// L177
      float v134 = v99[((k1 * 4) + 3)];	// L178
      float v135 = v133 + v134;	// L179
      #pragma HLS bind_op variable=v135 op=fadd impl=fabric
      o_im1[((k1 * 4) + 3)] = v135;	// L180
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re1[_iv0];
      }
      v96.write(_vec);
    }	// L182
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im1[_iv0];
      }
      v97.write(_vec);
    }	// L183
  }
}

const float twr[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L187
const float twi[128] = {0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L188
void intra_2_0(
  hls::stream< hls::vector< float, 32 > >& v136,
  hls::stream< hls::vector< float, 32 > >& v137,
  hls::stream< hls::vector< float, 32 > >& v138,
  hls::stream< hls::vector< float, 32 > >& v139
) {	// L189
  // placeholder for const float twr	// L195
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L196
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i2: for (int _i2 = 0; _i2 < 8; _i2++) {	// L197
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v143 = v136.read();
    hls::vector< float, 32 > v144 = v137.read();
    float o_re2[32];	// L200
    #pragma HLS array_partition variable=o_re2 complete
    float o_im2[32];	// L201
    #pragma HLS array_partition variable=o_im2 complete
    l_S_k_0_k2: for (int k2 = 0; k2 < 16; k2++) {	// L202
    #pragma HLS unroll
      int v148 = k2 >> 2;	// L203
      int v149 = v148 << 3;	// L204
      int32_t v150 = k2;	// L205
      int32_t v151 = v150 & 3;	// L206
      int32_t v152 = v149;	// L207
      int32_t v153 = v152 | v151;	// L208
      int32_t il;	// L209
      il = v153;	// L210
      int32_t v155 = il;	// L211
      int32_t v156 = v155 | 4;	// L212
      int32_t iu;	// L213
      iu = v156;	// L214
      int32_t v158 = v151 << 5;	// L215
      int32_t tw_k;	// L216
      tw_k = v158;	// L217
      int32_t v160 = il;	// L218
      int v161 = v160;	// L219
      float v162 = v143[v161];	// L220
      float a_re2;	// L221
      a_re2 = v162;	// L222
      int32_t v164 = il;	// L223
      int v165 = v164;	// L224
      float v166 = v144[v165];	// L225
      float a_im2;	// L226
      a_im2 = v166;	// L227
      int32_t v168 = iu;	// L228
      int v169 = v168;	// L229
      float v170 = v143[v169];	// L230
      float b_re2;	// L231
      b_re2 = v170;	// L232
      int32_t v172 = iu;	// L233
      int v173 = v172;	// L234
      float v174 = v144[v173];	// L235
      float b_im2;	// L236
      b_im2 = v174;	// L237
      int32_t v176 = tw_k;	// L238
      int v177 = v176;	// L239
      float v178 = twr[v177];	// L240
      float tr;	// L241
      tr = v178;	// L242
      int32_t v180 = tw_k;	// L243
      int v181 = v180;	// L244
      float v182 = twi[v181];	// L245
      float ti;	// L246
      ti = v182;	// L247
      float v184 = b_re2;	// L248
      float v185 = tr;	// L249
      float v186 = v184 * v185;	// L250
      float v187 = b_im2;	// L251
      float v188 = ti;	// L252
      float v189 = v187 * v188;	// L253
      float v190 = v186 - v189;	// L254
      float bw_re;	// L255
      bw_re = v190;	// L256
      float v192 = b_re2;	// L257
      float v193 = ti;	// L258
      float v194 = v192 * v193;	// L259
      float v195 = b_im2;	// L260
      float v196 = tr;	// L261
      float v197 = v195 * v196;	// L262
      float v198 = v194 + v197;	// L263
      float bw_im;	// L264
      bw_im = v198;	// L265
      float v200 = a_re2;	// L266
      float v201 = bw_re;	// L267
      float v202 = v200 + v201;	// L268
      #pragma HLS bind_op variable=v202 op=fadd impl=fabric
      int32_t v203 = il;	// L269
      int v204 = v203;	// L270
      o_re2[v204] = v202;	// L271
      float v205 = a_im2;	// L272
      float v206 = bw_im;	// L273
      float v207 = v205 + v206;	// L274
      #pragma HLS bind_op variable=v207 op=fadd impl=fabric
      int32_t v208 = il;	// L275
      int v209 = v208;	// L276
      o_im2[v209] = v207;	// L277
      float v210 = a_re2;	// L278
      float v211 = bw_re;	// L279
      float v212 = v210 - v211;	// L280
      #pragma HLS bind_op variable=v212 op=fsub impl=fabric
      int32_t v213 = iu;	// L281
      int v214 = v213;	// L282
      o_re2[v214] = v212;	// L283
      float v215 = a_im2;	// L284
      float v216 = bw_im;	// L285
      float v217 = v215 - v216;	// L286
      #pragma HLS bind_op variable=v217 op=fsub impl=fabric
      int32_t v218 = iu;	// L287
      int v219 = v218;	// L288
      o_im2[v219] = v217;	// L289
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re2[_iv0];
      }
      v138.write(_vec);
    }	// L291
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im2[_iv0];
      }
      v139.write(_vec);
    }	// L292
  }
}

void intra_3_0(
  hls::stream< hls::vector< float, 32 > >& v220,
  hls::stream< hls::vector< float, 32 > >& v221,
  hls::stream< hls::vector< float, 32 > >& v222,
  hls::stream< hls::vector< float, 32 > >& v223
) {	// L296
  // placeholder for const float twr	// L302
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L303
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i3: for (int _i3 = 0; _i3 < 8; _i3++) {	// L304
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v227 = v220.read();
    hls::vector< float, 32 > v228 = v221.read();
    float o_re3[32];	// L307
    #pragma HLS array_partition variable=o_re3 complete
    float o_im3[32];	// L308
    #pragma HLS array_partition variable=o_im3 complete
    l_S_k_0_k3: for (int k3 = 0; k3 < 16; k3++) {	// L309
    #pragma HLS unroll
      int v232 = k3 >> 3;	// L310
      int v233 = v232 << 4;	// L311
      int32_t v234 = k3;	// L312
      int32_t v235 = v234 & 7;	// L313
      int32_t v236 = v233;	// L314
      int32_t v237 = v236 | v235;	// L315
      int32_t il1;	// L316
      il1 = v237;	// L317
      int32_t v239 = il1;	// L318
      int32_t v240 = v239 | 8;	// L319
      int32_t iu1;	// L320
      iu1 = v240;	// L321
      int32_t v242 = v235 << 4;	// L322
      int32_t tw_k1;	// L323
      tw_k1 = v242;	// L324
      int32_t v244 = il1;	// L325
      int v245 = v244;	// L326
      float v246 = v227[v245];	// L327
      float a_re3;	// L328
      a_re3 = v246;	// L329
      int32_t v248 = il1;	// L330
      int v249 = v248;	// L331
      float v250 = v228[v249];	// L332
      float a_im3;	// L333
      a_im3 = v250;	// L334
      int32_t v252 = iu1;	// L335
      int v253 = v252;	// L336
      float v254 = v227[v253];	// L337
      float b_re3;	// L338
      b_re3 = v254;	// L339
      int32_t v256 = iu1;	// L340
      int v257 = v256;	// L341
      float v258 = v228[v257];	// L342
      float b_im3;	// L343
      b_im3 = v258;	// L344
      int32_t v260 = tw_k1;	// L345
      int v261 = v260;	// L346
      float v262 = twr[v261];	// L347
      float tr1;	// L348
      tr1 = v262;	// L349
      int32_t v264 = tw_k1;	// L350
      int v265 = v264;	// L351
      float v266 = twi[v265];	// L352
      float ti1;	// L353
      ti1 = v266;	// L354
      float v268 = b_re3;	// L355
      float v269 = tr1;	// L356
      float v270 = v268 * v269;	// L357
      float v271 = b_im3;	// L358
      float v272 = ti1;	// L359
      float v273 = v271 * v272;	// L360
      float v274 = v270 - v273;	// L361
      float bw_re1;	// L362
      bw_re1 = v274;	// L363
      float v276 = b_re3;	// L364
      float v277 = ti1;	// L365
      float v278 = v276 * v277;	// L366
      float v279 = b_im3;	// L367
      float v280 = tr1;	// L368
      float v281 = v279 * v280;	// L369
      float v282 = v278 + v281;	// L370
      float bw_im1;	// L371
      bw_im1 = v282;	// L372
      float v284 = a_re3;	// L373
      float v285 = bw_re1;	// L374
      float v286 = v284 + v285;	// L375
      #pragma HLS bind_op variable=v286 op=fadd impl=fabric
      int32_t v287 = il1;	// L376
      int v288 = v287;	// L377
      o_re3[v288] = v286;	// L378
      float v289 = a_im3;	// L379
      float v290 = bw_im1;	// L380
      float v291 = v289 + v290;	// L381
      #pragma HLS bind_op variable=v291 op=fadd impl=fabric
      int32_t v292 = il1;	// L382
      int v293 = v292;	// L383
      o_im3[v293] = v291;	// L384
      float v294 = a_re3;	// L385
      float v295 = bw_re1;	// L386
      float v296 = v294 - v295;	// L387
      #pragma HLS bind_op variable=v296 op=fsub impl=fabric
      int32_t v297 = iu1;	// L388
      int v298 = v297;	// L389
      o_re3[v298] = v296;	// L390
      float v299 = a_im3;	// L391
      float v300 = bw_im1;	// L392
      float v301 = v299 - v300;	// L393
      #pragma HLS bind_op variable=v301 op=fsub impl=fabric
      int32_t v302 = iu1;	// L394
      int v303 = v302;	// L395
      o_im3[v303] = v301;	// L396
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re3[_iv0];
      }
      v222.write(_vec);
    }	// L398
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im3[_iv0];
      }
      v223.write(_vec);
    }	// L399
  }
}

void intra_4_0(
  hls::stream< hls::vector< float, 32 > >& v304,
  hls::stream< hls::vector< float, 32 > >& v305,
  hls::stream< hls::vector< float, 32 > >& v306,
  hls::stream< hls::vector< float, 32 > >& v307
) {	// L403
  // placeholder for const float twr	// L406
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L407
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i4: for (int _i4 = 0; _i4 < 8; _i4++) {	// L408
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v311 = v304.read();
    hls::vector< float, 32 > v312 = v305.read();
    float o_re4[32];	// L411
    #pragma HLS array_partition variable=o_re4 complete
    float o_im4[32];	// L412
    #pragma HLS array_partition variable=o_im4 complete
    l_S_k_0_k4: for (int k4 = 0; k4 < 16; k4++) {	// L413
    #pragma HLS unroll
      int32_t v316 = k4;	// L414
      int32_t il2;	// L415
      il2 = v316;	// L416
      int32_t v318 = v316 | 16;	// L417
      int32_t iu2;	// L418
      iu2 = v318;	// L419
      int v320 = k4 << 3;	// L420
      int32_t v321 = v320;	// L421
      int32_t tw_k2;	// L422
      tw_k2 = v321;	// L423
      int32_t v323 = il2;	// L424
      int v324 = v323;	// L425
      float v325 = v311[v324];	// L426
      float a_re4;	// L427
      a_re4 = v325;	// L428
      int32_t v327 = il2;	// L429
      int v328 = v327;	// L430
      float v329 = v312[v328];	// L431
      float a_im4;	// L432
      a_im4 = v329;	// L433
      int32_t v331 = iu2;	// L434
      int v332 = v331;	// L435
      float v333 = v311[v332];	// L436
      float b_re4;	// L437
      b_re4 = v333;	// L438
      int32_t v335 = iu2;	// L439
      int v336 = v335;	// L440
      float v337 = v312[v336];	// L441
      float b_im4;	// L442
      b_im4 = v337;	// L443
      int32_t v339 = tw_k2;	// L444
      int v340 = v339;	// L445
      float v341 = twr[v340];	// L446
      float tr2;	// L447
      tr2 = v341;	// L448
      int32_t v343 = tw_k2;	// L449
      int v344 = v343;	// L450
      float v345 = twi[v344];	// L451
      float ti2;	// L452
      ti2 = v345;	// L453
      float v347 = b_re4;	// L454
      float v348 = tr2;	// L455
      float v349 = v347 * v348;	// L456
      float v350 = b_im4;	// L457
      float v351 = ti2;	// L458
      float v352 = v350 * v351;	// L459
      float v353 = v349 - v352;	// L460
      float bw_re2;	// L461
      bw_re2 = v353;	// L462
      float v355 = b_re4;	// L463
      float v356 = ti2;	// L464
      float v357 = v355 * v356;	// L465
      float v358 = b_im4;	// L466
      float v359 = tr2;	// L467
      float v360 = v358 * v359;	// L468
      float v361 = v357 + v360;	// L469
      float bw_im2;	// L470
      bw_im2 = v361;	// L471
      float v363 = a_re4;	// L472
      float v364 = bw_re2;	// L473
      float v365 = v363 + v364;	// L474
      #pragma HLS bind_op variable=v365 op=fadd impl=fabric
      int32_t v366 = il2;	// L475
      int v367 = v366;	// L476
      o_re4[v367] = v365;	// L477
      float v368 = a_im4;	// L478
      float v369 = bw_im2;	// L479
      float v370 = v368 + v369;	// L480
      #pragma HLS bind_op variable=v370 op=fadd impl=fabric
      int32_t v371 = il2;	// L481
      int v372 = v371;	// L482
      o_im4[v372] = v370;	// L483
      float v373 = a_re4;	// L484
      float v374 = bw_re2;	// L485
      float v375 = v373 - v374;	// L486
      #pragma HLS bind_op variable=v375 op=fsub impl=fabric
      int32_t v376 = iu2;	// L487
      int v377 = v376;	// L488
      o_re4[v377] = v375;	// L489
      float v378 = a_im4;	// L490
      float v379 = bw_im2;	// L491
      float v380 = v378 - v379;	// L492
      #pragma HLS bind_op variable=v380 op=fsub impl=fabric
      int32_t v381 = iu2;	// L493
      int v382 = v381;	// L494
      o_im4[v382] = v380;	// L495
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re4[_iv0];
      }
      v306.write(_vec);
    }	// L497
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im4[_iv0];
      }
      v307.write(_vec);
    }	// L498
  }
}

void inter_5_0(
  hls::stream< hls::vector< float, 32 > >& v383,
  hls::stream< hls::vector< float, 32 > >& v384,
  hls::stream< hls::vector< float, 32 > >& v385,
  hls::stream< hls::vector< float, 32 > >& v386
) {	// L502
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L510
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L511
  #pragma HLS array_partition variable=twi complete
  float in_re[32][8];	// L512
  #pragma HLS array_partition variable=in_re complete dim=1

  #pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L513
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L514
  #pragma HLS array_partition variable=out_re_b complete dim=1

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L515
  #pragma HLS array_partition variable=out_im_b complete dim=1

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L516
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v394 = v383.read();
    hls::vector< float, 32 > v395 = v384.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L519
    #pragma HLS unroll
      int32_t v397 = i;	// L520
      int32_t v398 = v397 & 1;	// L521
      int32_t v399 = v398 << 4;	// L522
      int32_t v400 = k5;	// L523
      int32_t v401 = v400 ^ v399;	// L524
      int32_t bank1;	// L525
      bank1 = v401;	// L526
      float v403 = v394[k5];	// L527
      int32_t v404 = bank1;	// L528
      int v405 = v404;	// L529
      in_re[v405][i] = v403;	// L530
      float v406 = v395[k5];	// L531
      int32_t v407 = bank1;	// L532
      int v408 = v407;	// L533
      in_im[v408][i] = v406;	// L534
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L537
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L538
    #pragma HLS unroll
      int v411 = i1 << 4;	// L539
      int v412 = v411 | k6;	// L540
      int32_t v413 = v412;	// L541
      int32_t bg;	// L542
      bg = v413;	// L543
      int32_t v415 = bg;	// L544
      int32_t v416 = v415 >> 5;	// L545
      int32_t grp;	// L546
      grp = v416;	// L547
      int32_t v418 = bg;	// L548
      int32_t v419 = v418 & 31;	// L549
      int32_t within;	// L550
      within = v419;	// L551
      int32_t v421 = within;	// L552
      int32_t v422 = v421 << 2;	// L553
      int32_t tw_k3;	// L554
      tw_k3 = v422;	// L555
      int32_t v424 = grp;	// L556
      int32_t v425 = v424 << 1;	// L557
      int32_t off_l;	// L558
      off_l = v425;	// L559
      int32_t v427 = off_l;	// L560
      int32_t v428 = v427 | 1;	// L561
      int32_t off_u;	// L562
      off_u = v428;	// L563
      int32_t v430 = within;	// L564
      int v431 = v430;	// L565
      int32_t v432 = off_l;	// L566
      int v433 = v432;	// L567
      float v434 = in_re[v431][v433];	// L568
      float a_re5;	// L569
      a_re5 = v434;	// L570
      int32_t v436 = within;	// L571
      int v437 = v436;	// L572
      int32_t v438 = off_l;	// L573
      int v439 = v438;	// L574
      float v440 = in_im[v437][v439];	// L575
      float a_im5;	// L576
      a_im5 = v440;	// L577
      int32_t v442 = within;	// L578
      int32_t v443 = v442 ^ 16;	// L579
      int v444 = v443;	// L580
      int32_t v445 = off_u;	// L581
      int v446 = v445;	// L582
      float v447 = in_re[v444][v446];	// L583
      float b_re5;	// L584
      b_re5 = v447;	// L585
      int32_t v449 = within;	// L586
      int32_t v450 = v449 ^ 16;	// L587
      int v451 = v450;	// L588
      int32_t v452 = off_u;	// L589
      int v453 = v452;	// L590
      float v454 = in_im[v451][v453];	// L591
      float b_im5;	// L592
      b_im5 = v454;	// L593
      int32_t v456 = tw_k3;	// L594
      int v457 = v456;	// L595
      float v458 = twr[v457];	// L596
      float tr3;	// L597
      tr3 = v458;	// L598
      int32_t v460 = tw_k3;	// L599
      int v461 = v460;	// L600
      float v462 = twi[v461];	// L601
      float ti3;	// L602
      ti3 = v462;	// L603
      float v464 = b_re5;	// L604
      float v465 = tr3;	// L605
      float v466 = v464 * v465;	// L606
      float v467 = b_im5;	// L607
      float v468 = ti3;	// L608
      float v469 = v467 * v468;	// L609
      float v470 = v466 - v469;	// L610
      float bw_re3;	// L611
      bw_re3 = v470;	// L612
      float v472 = b_re5;	// L613
      float v473 = ti3;	// L614
      float v474 = v472 * v473;	// L615
      float v475 = b_im5;	// L616
      float v476 = tr3;	// L617
      float v477 = v475 * v476;	// L618
      float v478 = v474 + v477;	// L619
      float bw_im3;	// L620
      bw_im3 = v478;	// L621
      float v480 = a_re5;	// L622
      float v481 = bw_re3;	// L623
      float v482 = v480 + v481;	// L624
      #pragma HLS bind_op variable=v482 op=fadd impl=fabric
      int32_t v483 = within;	// L625
      int v484 = v483;	// L626
      int32_t v485 = off_l;	// L627
      int v486 = v485;	// L628
      out_re_b[v484][v486] = v482;	// L629
      float v487 = a_im5;	// L630
      float v488 = bw_im3;	// L631
      float v489 = v487 + v488;	// L632
      #pragma HLS bind_op variable=v489 op=fadd impl=fabric
      int32_t v490 = within;	// L633
      int v491 = v490;	// L634
      int32_t v492 = off_l;	// L635
      int v493 = v492;	// L636
      out_im_b[v491][v493] = v489;	// L637
      float v494 = a_re5;	// L638
      float v495 = bw_re3;	// L639
      float v496 = v494 - v495;	// L640
      #pragma HLS bind_op variable=v496 op=fsub impl=fabric
      int32_t v497 = within;	// L641
      int32_t v498 = v497 ^ 16;	// L642
      int v499 = v498;	// L643
      int32_t v500 = off_u;	// L644
      int v501 = v500;	// L645
      out_re_b[v499][v501] = v496;	// L646
      float v502 = a_im5;	// L647
      float v503 = bw_im3;	// L648
      float v504 = v502 - v503;	// L649
      #pragma HLS bind_op variable=v504 op=fsub impl=fabric
      int32_t v505 = within;	// L650
      int32_t v506 = v505 ^ 16;	// L651
      int v507 = v506;	// L652
      int32_t v508 = off_u;	// L653
      int v509 = v508;	// L654
      out_im_b[v507][v509] = v504;	// L655
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L658
  #pragma HLS pipeline II=1
    float chunk_re1[32];	// L659
    #pragma HLS array_partition variable=chunk_re1 complete
    float chunk_im1[32];	// L660
    #pragma HLS array_partition variable=chunk_im1 complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L661
    #pragma HLS unroll
      int32_t v514 = i2;	// L662
      int32_t v515 = v514 & 1;	// L663
      int32_t v516 = v515 << 4;	// L664
      int32_t v517 = k7;	// L665
      int32_t v518 = v517 ^ v516;	// L666
      int32_t bank2;	// L667
      bank2 = v518;	// L668
      int32_t v520 = bank2;	// L669
      int v521 = v520;	// L670
      float v522 = out_re_b[v521][i2];	// L671
      chunk_re1[k7] = v522;	// L672
      int32_t v523 = bank2;	// L673
      int v524 = v523;	// L674
      float v525 = out_im_b[v524][i2];	// L675
      chunk_im1[k7] = v525;	// L676
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re1[_iv0];
      }
      v385.write(_vec);
    }	// L678
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im1[_iv0];
      }
      v386.write(_vec);
    }	// L679
  }
}

void inter_6_0(
  hls::stream< hls::vector< float, 32 > >& v526,
  hls::stream< hls::vector< float, 32 > >& v527,
  hls::stream< hls::vector< float, 32 > >& v528,
  hls::stream< hls::vector< float, 32 > >& v529
) {	// L683
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L692
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L693
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L694
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L695
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L696
  #pragma HLS array_partition variable=out_re_b1 complete dim=1

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L697
  #pragma HLS array_partition variable=out_im_b1 complete dim=1

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L698
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v537 = v526.read();
    hls::vector< float, 32 > v538 = v527.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L701
    #pragma HLS unroll
      int v540 = i3 >> 1;	// L702
      int32_t v541 = v540;	// L703
      int32_t v542 = v541 & 1;	// L704
      int32_t v543 = v542 << 4;	// L705
      int32_t v544 = k8;	// L706
      int32_t v545 = v544 ^ v543;	// L707
      int32_t bank3;	// L708
      bank3 = v545;	// L709
      float v547 = v537[k8];	// L710
      int32_t v548 = bank3;	// L711
      int v549 = v548;	// L712
      in_re1[v549][i3] = v547;	// L713
      float v550 = v538[k8];	// L714
      int32_t v551 = bank3;	// L715
      int v552 = v551;	// L716
      in_im1[v552][i3] = v550;	// L717
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L720
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L721
    #pragma HLS unroll
      int v555 = i4 << 4;	// L722
      int v556 = v555 | k9;	// L723
      uint32_t v557 = v556;	// L724
      uint32_t bg1;	// L725
      bg1 = v557;	// L726
      int32_t v559 = bg1;	// L727
      int32_t v560 = v559 & 63;	// L728
      int32_t v561 = v560 << 1;	// L729
      uint32_t tw_k4;	// L730
      tw_k4 = v561;	// L731
      int32_t v563 = i4;	// L732
      int32_t v564 = v563 & 1;	// L733
      int32_t v565 = v564 << 4;	// L734
      int32_t v566 = k9;	// L735
      int32_t v567 = v566 | v565;	// L736
      uint32_t bank_il;	// L737
      bank_il = v567;	// L738
      int32_t v569 = bank_il;	// L739
      int32_t v570 = v569 ^ 16;	// L740
      uint32_t bank_iu;	// L741
      bank_iu = v570;	// L742
      int v572 = i4 >> 2;	// L743
      int v573 = v572 << 2;	// L744
      int v574 = i4 >> 1;	// L745
      int32_t v575 = v574;	// L746
      int32_t v576 = v575 & 1;	// L747
      int32_t v577 = v573;	// L748
      int32_t v578 = v577 | v576;	// L749
      uint32_t off_il;	// L750
      off_il = v578;	// L751
      int32_t v580 = off_il;	// L752
      int32_t v581 = v580 | 2;	// L753
      uint32_t off_iu;	// L754
      off_iu = v581;	// L755
      int32_t v583 = bank_il;	// L756
      int v584 = v583;	// L757
      int32_t v585 = off_il;	// L758
      int v586 = v585;	// L759
      float v587 = in_re1[v584][v586];	// L760
      float a_re6;	// L761
      a_re6 = v587;	// L762
      int32_t v589 = bank_il;	// L763
      int v590 = v589;	// L764
      int32_t v591 = off_il;	// L765
      int v592 = v591;	// L766
      float v593 = in_im1[v590][v592];	// L767
      float a_im6;	// L768
      a_im6 = v593;	// L769
      int32_t v595 = bank_iu;	// L770
      int v596 = v595;	// L771
      int32_t v597 = off_iu;	// L772
      int v598 = v597;	// L773
      float v599 = in_re1[v596][v598];	// L774
      float b_re6;	// L775
      b_re6 = v599;	// L776
      int32_t v601 = bank_iu;	// L777
      int v602 = v601;	// L778
      int32_t v603 = off_iu;	// L779
      int v604 = v603;	// L780
      float v605 = in_im1[v602][v604];	// L781
      float b_im6;	// L782
      b_im6 = v605;	// L783
      int32_t v607 = tw_k4;	// L784
      int v608 = v607;	// L785
      float v609 = twr[v608];	// L786
      float tr4;	// L787
      tr4 = v609;	// L788
      int32_t v611 = tw_k4;	// L789
      int v612 = v611;	// L790
      float v613 = twi[v612];	// L791
      float ti4;	// L792
      ti4 = v613;	// L793
      float v615 = b_re6;	// L794
      float v616 = tr4;	// L795
      float v617 = v615 * v616;	// L796
      float v618 = b_im6;	// L797
      float v619 = ti4;	// L798
      float v620 = v618 * v619;	// L799
      float v621 = v617 - v620;	// L800
      float bw_re4;	// L801
      bw_re4 = v621;	// L802
      float v623 = b_re6;	// L803
      float v624 = ti4;	// L804
      float v625 = v623 * v624;	// L805
      float v626 = b_im6;	// L806
      float v627 = tr4;	// L807
      float v628 = v626 * v627;	// L808
      float v629 = v625 + v628;	// L809
      float bw_im4;	// L810
      bw_im4 = v629;	// L811
      float v631 = a_re6;	// L812
      float v632 = bw_re4;	// L813
      float v633 = v631 + v632;	// L814
      #pragma HLS bind_op variable=v633 op=fadd impl=fabric
      int32_t v634 = bank_il;	// L815
      int v635 = v634;	// L816
      int32_t v636 = off_il;	// L817
      int v637 = v636;	// L818
      out_re_b1[v635][v637] = v633;	// L819
      float v638 = a_im6;	// L820
      float v639 = bw_im4;	// L821
      float v640 = v638 + v639;	// L822
      #pragma HLS bind_op variable=v640 op=fadd impl=fabric
      int32_t v641 = bank_il;	// L823
      int v642 = v641;	// L824
      int32_t v643 = off_il;	// L825
      int v644 = v643;	// L826
      out_im_b1[v642][v644] = v640;	// L827
      float v645 = a_re6;	// L828
      float v646 = bw_re4;	// L829
      float v647 = v645 - v646;	// L830
      #pragma HLS bind_op variable=v647 op=fsub impl=fabric
      int32_t v648 = bank_iu;	// L831
      int v649 = v648;	// L832
      int32_t v650 = off_iu;	// L833
      int v651 = v650;	// L834
      out_re_b1[v649][v651] = v647;	// L835
      float v652 = a_im6;	// L836
      float v653 = bw_im4;	// L837
      float v654 = v652 - v653;	// L838
      #pragma HLS bind_op variable=v654 op=fsub impl=fabric
      int32_t v655 = bank_iu;	// L839
      int v656 = v655;	// L840
      int32_t v657 = off_iu;	// L841
      int v658 = v657;	// L842
      out_im_b1[v656][v658] = v654;	// L843
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L846
  #pragma HLS pipeline II=1
    float chunk_re2[32];	// L847
    #pragma HLS array_partition variable=chunk_re2 complete
    float chunk_im2[32];	// L848
    #pragma HLS array_partition variable=chunk_im2 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L849
    #pragma HLS unroll
      int v663 = i5 >> 1;	// L850
      int32_t v664 = v663;	// L851
      int32_t v665 = v664 & 1;	// L852
      int32_t v666 = v665 << 4;	// L853
      int32_t v667 = k10;	// L854
      int32_t v668 = v667 ^ v666;	// L855
      int32_t bank4;	// L856
      bank4 = v668;	// L857
      int32_t v670 = bank4;	// L858
      int v671 = v670;	// L859
      float v672 = out_re_b1[v671][i5];	// L860
      chunk_re2[k10] = v672;	// L861
      int32_t v673 = bank4;	// L862
      int v674 = v673;	// L863
      float v675 = out_im_b1[v674][i5];	// L864
      chunk_im2[k10] = v675;	// L865
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re2[_iv0];
      }
      v528.write(_vec);
    }	// L867
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im2[_iv0];
      }
      v529.write(_vec);
    }	// L868
  }
}

void inter_7_0(
  hls::stream< hls::vector< float, 32 > >& v676,
  hls::stream< hls::vector< float, 32 > >& v677,
  hls::stream< hls::vector< float, 32 > >& v678,
  hls::stream< hls::vector< float, 32 > >& v679
) {	// L872
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L879
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L880
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L881
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L882
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L883
  #pragma HLS array_partition variable=out_re_b2 complete dim=1

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L884
  #pragma HLS array_partition variable=out_im_b2 complete dim=1

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L885
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v687 = v676.read();
    hls::vector< float, 32 > v688 = v677.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L888
    #pragma HLS unroll
      int v690 = i6 >> 2;	// L889
      int32_t v691 = v690;	// L890
      int32_t v692 = v691 & 1;	// L891
      int32_t v693 = v692 << 4;	// L892
      int32_t v694 = k11;	// L893
      int32_t v695 = v694 ^ v693;	// L894
      int32_t bank5;	// L895
      bank5 = v695;	// L896
      float v697 = v687[k11];	// L897
      int32_t v698 = bank5;	// L898
      int v699 = v698;	// L899
      in_re2[v699][i6] = v697;	// L900
      float v700 = v688[k11];	// L901
      int32_t v701 = bank5;	// L902
      int v702 = v701;	// L903
      in_im2[v702][i6] = v700;	// L904
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L907
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L908
    #pragma HLS unroll
      int32_t v705 = i7;	// L909
      int32_t v706 = v705 & 1;	// L910
      int32_t v707 = v706 << 4;	// L911
      int32_t v708 = k12;	// L912
      int32_t v709 = v708 | v707;	// L913
      uint32_t bank_il1;	// L914
      bank_il1 = v709;	// L915
      int32_t v711 = bank_il1;	// L916
      int32_t v712 = v711 ^ 16;	// L917
      uint32_t bank_iu1;	// L918
      bank_iu1 = v712;	// L919
      int v714 = i7 >> 1;	// L920
      uint32_t v715 = v714;	// L921
      uint32_t off_il1;	// L922
      off_il1 = v715;	// L923
      int32_t v717 = off_il1;	// L924
      int32_t v718 = v717 | 4;	// L925
      uint32_t off_iu1;	// L926
      off_iu1 = v718;	// L927
      int v720 = i7 << 4;	// L928
      int v721 = v720 | k12;	// L929
      uint32_t v722 = v721;	// L930
      uint32_t tw_k5;	// L931
      tw_k5 = v722;	// L932
      int32_t v724 = bank_il1;	// L933
      int v725 = v724;	// L934
      int32_t v726 = off_il1;	// L935
      int v727 = v726;	// L936
      float v728 = in_re2[v725][v727];	// L937
      float a_re7;	// L938
      a_re7 = v728;	// L939
      int32_t v730 = bank_il1;	// L940
      int v731 = v730;	// L941
      int32_t v732 = off_il1;	// L942
      int v733 = v732;	// L943
      float v734 = in_im2[v731][v733];	// L944
      float a_im7;	// L945
      a_im7 = v734;	// L946
      int32_t v736 = bank_iu1;	// L947
      int v737 = v736;	// L948
      int32_t v738 = off_iu1;	// L949
      int v739 = v738;	// L950
      float v740 = in_re2[v737][v739];	// L951
      float b_re7;	// L952
      b_re7 = v740;	// L953
      int32_t v742 = bank_iu1;	// L954
      int v743 = v742;	// L955
      int32_t v744 = off_iu1;	// L956
      int v745 = v744;	// L957
      float v746 = in_im2[v743][v745];	// L958
      float b_im7;	// L959
      b_im7 = v746;	// L960
      int32_t v748 = tw_k5;	// L961
      int v749 = v748;	// L962
      float v750 = twr[v749];	// L963
      float tr5;	// L964
      tr5 = v750;	// L965
      int32_t v752 = tw_k5;	// L966
      int v753 = v752;	// L967
      float v754 = twi[v753];	// L968
      float ti5;	// L969
      ti5 = v754;	// L970
      float v756 = b_re7;	// L971
      float v757 = tr5;	// L972
      float v758 = v756 * v757;	// L973
      float v759 = b_im7;	// L974
      float v760 = ti5;	// L975
      float v761 = v759 * v760;	// L976
      float v762 = v758 - v761;	// L977
      float bw_re5;	// L978
      bw_re5 = v762;	// L979
      float v764 = b_re7;	// L980
      float v765 = ti5;	// L981
      float v766 = v764 * v765;	// L982
      float v767 = b_im7;	// L983
      float v768 = tr5;	// L984
      float v769 = v767 * v768;	// L985
      float v770 = v766 + v769;	// L986
      float bw_im5;	// L987
      bw_im5 = v770;	// L988
      float v772 = a_re7;	// L989
      float v773 = bw_re5;	// L990
      float v774 = v772 + v773;	// L991
      #pragma HLS bind_op variable=v774 op=fadd impl=fabric
      int32_t v775 = bank_il1;	// L992
      int v776 = v775;	// L993
      int32_t v777 = off_il1;	// L994
      int v778 = v777;	// L995
      out_re_b2[v776][v778] = v774;	// L996
      float v779 = a_im7;	// L997
      float v780 = bw_im5;	// L998
      float v781 = v779 + v780;	// L999
      #pragma HLS bind_op variable=v781 op=fadd impl=fabric
      int32_t v782 = bank_il1;	// L1000
      int v783 = v782;	// L1001
      int32_t v784 = off_il1;	// L1002
      int v785 = v784;	// L1003
      out_im_b2[v783][v785] = v781;	// L1004
      float v786 = a_re7;	// L1005
      float v787 = bw_re5;	// L1006
      float v788 = v786 - v787;	// L1007
      #pragma HLS bind_op variable=v788 op=fsub impl=fabric
      int32_t v789 = bank_iu1;	// L1008
      int v790 = v789;	// L1009
      int32_t v791 = off_iu1;	// L1010
      int v792 = v791;	// L1011
      out_re_b2[v790][v792] = v788;	// L1012
      float v793 = a_im7;	// L1013
      float v794 = bw_im5;	// L1014
      float v795 = v793 - v794;	// L1015
      #pragma HLS bind_op variable=v795 op=fsub impl=fabric
      int32_t v796 = bank_iu1;	// L1016
      int v797 = v796;	// L1017
      int32_t v798 = off_iu1;	// L1018
      int v799 = v798;	// L1019
      out_im_b2[v797][v799] = v795;	// L1020
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1023
  #pragma HLS pipeline II=1
    float chunk_re3[32];	// L1024
    #pragma HLS array_partition variable=chunk_re3 complete
    float chunk_im3[32];	// L1025
    #pragma HLS array_partition variable=chunk_im3 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1026
    #pragma HLS unroll
      int v804 = i8 >> 2;	// L1027
      int32_t v805 = v804;	// L1028
      int32_t v806 = v805 & 1;	// L1029
      int32_t v807 = v806 << 4;	// L1030
      int32_t v808 = k13;	// L1031
      int32_t v809 = v808 ^ v807;	// L1032
      int32_t bank6;	// L1033
      bank6 = v809;	// L1034
      int32_t v811 = bank6;	// L1035
      int v812 = v811;	// L1036
      float v813 = out_re_b2[v812][i8];	// L1037
      chunk_re3[k13] = v813;	// L1038
      int32_t v814 = bank6;	// L1039
      int v815 = v814;	// L1040
      float v816 = out_im_b2[v815][i8];	// L1041
      chunk_im3[k13] = v816;	// L1042
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re3[_iv0];
      }
      v678.write(_vec);
    }	// L1044
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im3[_iv0];
      }
      v679.write(_vec);
    }	// L1045
  }
}

void output_stage_0(
  float v817[8][32],
  float v818[8][32],
  hls::stream< hls::vector< float, 32 > >& v819,
  hls::stream< hls::vector< float, 32 > >& v820
) {	// L1049
  #pragma HLS array_partition variable=v817 complete dim=2

  #pragma HLS array_partition variable=v818 complete dim=2

  l_S_i_0_i9: for (int i9 = 0; i9 < 8; i9++) {	// L1050
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v822 = v819.read();
    hls::vector< float, 32 > v823 = v820.read();
    l_S_k_0_k14: for (int k14 = 0; k14 < 32; k14++) {	// L1053
    #pragma HLS unroll
      float v825 = v822[k14];	// L1054
      v817[i9][k14] = v825;	// L1055
      float v826 = v823[k14];	// L1056
      v818[i9][k14] = v826;	// L1057
    }
  }
}

/// This is top function.
void fft_256(
  float v827[8][32],
  float v828[8][32],
  float v829[8][32],
  float v830[8][32]
) {	// L1062
  #pragma HLS dataflow disable_start_propagation
  #pragma HLS array_partition variable=v827 complete dim=2

  #pragma HLS array_partition variable=v828 complete dim=2

  #pragma HLS array_partition variable=v829 complete dim=2

  #pragma HLS array_partition variable=v830 complete dim=2

  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v831;
  #pragma HLS stream variable=v831 depth=2	// L1063
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v832;
  #pragma HLS stream variable=v832 depth=2	// L1064
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v833;
  #pragma HLS stream variable=v833 depth=2	// L1065
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v834;
  #pragma HLS stream variable=v834 depth=2	// L1066
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v835;
  #pragma HLS stream variable=v835 depth=2	// L1067
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v836;
  #pragma HLS stream variable=v836 depth=2	// L1068
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v837;
  #pragma HLS stream variable=v837 depth=2	// L1069
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v838;
  #pragma HLS stream variable=v838 depth=2	// L1070
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v839;
  #pragma HLS stream variable=v839 depth=2	// L1071
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v840;
  #pragma HLS stream variable=v840 depth=2	// L1072
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v841;
  #pragma HLS stream variable=v841 depth=2	// L1073
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v842;
  #pragma HLS stream variable=v842 depth=2	// L1074
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v843;
  #pragma HLS stream variable=v843 depth=2	// L1075
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v844;
  #pragma HLS stream variable=v844 depth=2	// L1076
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v845;
  #pragma HLS stream variable=v845 depth=2	// L1077
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v846;
  #pragma HLS stream variable=v846 depth=2	// L1078
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v847;
  #pragma HLS stream variable=v847 depth=2	// L1079
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v848;
  #pragma HLS stream variable=v848 depth=2	// L1080
  bit_rev_stage_0(v827, v828, v831, v840);	// L1081
  intra_0_0(v831, v840, v832, v841);	// L1082
  intra_1_0(v832, v841, v833, v842);	// L1083
  intra_2_0(v833, v842, v834, v843);	// L1084
  intra_3_0(v834, v843, v835, v844);	// L1085
  intra_4_0(v835, v844, v836, v845);	// L1086
  inter_5_0(v836, v845, v837, v846);	// L1087
  inter_6_0(v837, v846, v838, v847);	// L1088
  inter_7_0(v838, v847, v839, v848);	// L1089
  output_stage_0(v829, v830, v839, v848);	// L1090
}


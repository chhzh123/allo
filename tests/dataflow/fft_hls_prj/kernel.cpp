
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
  #pragma HLS dependence variable=buf_re intra false
  float buf_im[32][8];	// L14
  #pragma HLS array_partition variable=buf_im complete dim=1

  #pragma HLS bind_storage variable=buf_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=buf_im inter false
  #pragma HLS dependence variable=buf_im intra false
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
  #pragma HLS dependence variable=in_re intra false
  float in_im[32][8];	// L513
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  #pragma HLS dependence variable=in_im intra false
  float out_re_b[32][8];	// L514
  #pragma HLS array_partition variable=out_re_b complete dim=1
  #pragma HLS array_partition variable=out_re_b cyclic factor=2 dim=2

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
  float out_im_b[32][8];	// L515
  #pragma HLS array_partition variable=out_im_b complete dim=1
  #pragma HLS array_partition variable=out_im_b cyclic factor=2 dim=2

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
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
  // placeholder for const float twr	// L694
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L695
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L696
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  #pragma HLS dependence variable=in_re1 intra false
  float in_im1[32][8];	// L697
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  #pragma HLS dependence variable=in_im1 intra false
  float out_re_b1[32][8];	// L698
  #pragma HLS array_partition variable=out_re_b1 complete dim=1
  #pragma HLS array_partition variable=out_re_b1 cyclic factor=4 dim=2

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
  float out_im_b1[32][8];	// L699
  #pragma HLS array_partition variable=out_im_b1 complete dim=1
  #pragma HLS array_partition variable=out_im_b1 cyclic factor=4 dim=2

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L700
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v537 = v526.read();
    hls::vector< float, 32 > v538 = v527.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L703
    #pragma HLS unroll
      int v540 = i3 >> 1;	// L704
      int32_t v541 = v540;	// L705
      int32_t v542 = v541 & 1;	// L706
      int32_t v543 = v542 << 4;	// L707
      int32_t v544 = k8;	// L708
      int32_t v545 = v544 ^ v543;	// L709
      int32_t bank3;	// L710
      bank3 = v545;	// L711
      float v547 = v537[k8];	// L712
      int32_t v548 = bank3;	// L713
      int v549 = v548;	// L714
      in_re1[v549][i3] = v547;	// L715
      float v550 = v538[k8];	// L716
      int32_t v551 = bank3;	// L717
      int v552 = v551;	// L718
      in_im1[v552][i3] = v550;	// L719
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L722
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L723
    #pragma HLS unroll
      int v555 = i4 << 4;	// L724
      int v556 = v555 | k9;	// L725
      int32_t v557 = v556;	// L726
      int32_t bg1;	// L727
      bg1 = v557;	// L728
      int32_t v559 = bg1;	// L729
      int32_t v560 = v559 >> 6;	// L730
      int32_t grp1;	// L731
      grp1 = v560;	// L732
      int32_t v562 = bg1;	// L733
      int32_t v563 = v562 & 63;	// L734
      int32_t within1;	// L735
      within1 = v563;	// L736
      int32_t v565 = within1;	// L737
      int32_t v566 = v565 << 1;	// L738
      int32_t tw_k4;	// L739
      tw_k4 = v566;	// L740
      int32_t v568 = grp1;	// L741
      int32_t v569 = v568 << 7;	// L742
      int32_t v570 = within1;	// L743
      int32_t v571 = v569 | v570;	// L744
      int32_t il3;	// L745
      il3 = v571;	// L746
      int32_t v573 = il3;	// L747
      int32_t v574 = v573 | 64;	// L748
      int32_t iu3;	// L749
      iu3 = v574;	// L750
      int32_t v576 = il3;	// L751
      int32_t v577 = v576 & 31;	// L752
      int32_t v578 = v576 >> 6;	// L753
      int32_t v579 = v578 & 1;	// L754
      int32_t v580 = v579 << 4;	// L755
      int32_t v581 = v577 ^ v580;	// L756
      int32_t bank_il;	// L757
      bank_il = v581;	// L758
      int32_t v583 = il3;	// L759
      int32_t v584 = v583 >> 5;	// L760
      int32_t off_il;	// L761
      off_il = v584;	// L762
      int32_t v586 = iu3;	// L763
      int32_t v587 = v586 & 31;	// L764
      int32_t v588 = v586 >> 6;	// L765
      int32_t v589 = v588 & 1;	// L766
      int32_t v590 = v589 << 4;	// L767
      int32_t v591 = v587 ^ v590;	// L768
      int32_t bank_iu;	// L769
      bank_iu = v591;	// L770
      int32_t v593 = iu3;	// L771
      int32_t v594 = v593 >> 5;	// L772
      int32_t off_iu;	// L773
      off_iu = v594;	// L774
      int32_t v596 = bank_il;	// L775
      int v597 = v596;	// L776
      int32_t v598 = off_il;	// L777
      int v599 = v598;	// L778
      float v600 = in_re1[v597][v599];	// L779
      float a_re6;	// L780
      a_re6 = v600;	// L781
      int32_t v602 = bank_il;	// L782
      int v603 = v602;	// L783
      int32_t v604 = off_il;	// L784
      int v605 = v604;	// L785
      float v606 = in_im1[v603][v605];	// L786
      float a_im6;	// L787
      a_im6 = v606;	// L788
      int32_t v608 = bank_iu;	// L789
      int v609 = v608;	// L790
      int32_t v610 = off_iu;	// L791
      int v611 = v610;	// L792
      float v612 = in_re1[v609][v611];	// L793
      float b_re6;	// L794
      b_re6 = v612;	// L795
      int32_t v614 = bank_iu;	// L796
      int v615 = v614;	// L797
      int32_t v616 = off_iu;	// L798
      int v617 = v616;	// L799
      float v618 = in_im1[v615][v617];	// L800
      float b_im6;	// L801
      b_im6 = v618;	// L802
      int32_t v620 = tw_k4;	// L803
      int v621 = v620;	// L804
      float v622 = twr[v621];	// L805
      float tr4;	// L806
      tr4 = v622;	// L807
      int32_t v624 = tw_k4;	// L808
      int v625 = v624;	// L809
      float v626 = twi[v625];	// L810
      float ti4;	// L811
      ti4 = v626;	// L812
      float v628 = b_re6;	// L813
      float v629 = tr4;	// L814
      float v630 = v628 * v629;	// L815
      float v631 = b_im6;	// L816
      float v632 = ti4;	// L817
      float v633 = v631 * v632;	// L818
      float v634 = v630 - v633;	// L819
      float bw_re4;	// L820
      bw_re4 = v634;	// L821
      float v636 = b_re6;	// L822
      float v637 = ti4;	// L823
      float v638 = v636 * v637;	// L824
      float v639 = b_im6;	// L825
      float v640 = tr4;	// L826
      float v641 = v639 * v640;	// L827
      float v642 = v638 + v641;	// L828
      float bw_im4;	// L829
      bw_im4 = v642;	// L830
      float v644 = a_re6;	// L831
      float v645 = bw_re4;	// L832
      float v646 = v644 + v645;	// L833
      #pragma HLS bind_op variable=v646 op=fadd impl=fabric
      int32_t v647 = bank_il;	// L834
      int v648 = v647;	// L835
      int32_t v649 = off_il;	// L836
      int v650 = v649;	// L837
      out_re_b1[v648][v650] = v646;	// L838
      float v651 = a_im6;	// L839
      float v652 = bw_im4;	// L840
      float v653 = v651 + v652;	// L841
      #pragma HLS bind_op variable=v653 op=fadd impl=fabric
      int32_t v654 = bank_il;	// L842
      int v655 = v654;	// L843
      int32_t v656 = off_il;	// L844
      int v657 = v656;	// L845
      out_im_b1[v655][v657] = v653;	// L846
      float v658 = a_re6;	// L847
      float v659 = bw_re4;	// L848
      float v660 = v658 - v659;	// L849
      #pragma HLS bind_op variable=v660 op=fsub impl=fabric
      int32_t v661 = bank_iu;	// L850
      int v662 = v661;	// L851
      int32_t v663 = off_iu;	// L852
      int v664 = v663;	// L853
      out_re_b1[v662][v664] = v660;	// L854
      float v665 = a_im6;	// L855
      float v666 = bw_im4;	// L856
      float v667 = v665 - v666;	// L857
      #pragma HLS bind_op variable=v667 op=fsub impl=fabric
      int32_t v668 = bank_iu;	// L858
      int v669 = v668;	// L859
      int32_t v670 = off_iu;	// L860
      int v671 = v670;	// L861
      out_im_b1[v669][v671] = v667;	// L862
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L865
  #pragma HLS pipeline II=1
    float chunk_re2[32];	// L866
    #pragma HLS array_partition variable=chunk_re2 complete
    float chunk_im2[32];	// L867
    #pragma HLS array_partition variable=chunk_im2 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L868
    #pragma HLS unroll
      int v676 = i5 >> 1;	// L869
      int32_t v677 = v676;	// L870
      int32_t v678 = v677 & 1;	// L871
      int32_t v679 = v678 << 4;	// L872
      int32_t v680 = k10;	// L873
      int32_t v681 = v680 ^ v679;	// L874
      int32_t bank4;	// L875
      bank4 = v681;	// L876
      int32_t v683 = bank4;	// L877
      int v684 = v683;	// L878
      float v685 = out_re_b1[v684][i5];	// L879
      chunk_re2[k10] = v685;	// L880
      int32_t v686 = bank4;	// L881
      int v687 = v686;	// L882
      float v688 = out_im_b1[v687][i5];	// L883
      chunk_im2[k10] = v688;	// L884
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re2[_iv0];
      }
      v528.write(_vec);
    }	// L886
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im2[_iv0];
      }
      v529.write(_vec);
    }	// L887
  }
}

void inter_7_0(
  hls::stream< hls::vector< float, 32 > >& v689,
  hls::stream< hls::vector< float, 32 > >& v690,
  hls::stream< hls::vector< float, 32 > >& v691,
  hls::stream< hls::vector< float, 32 > >& v692
) {	// L891
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L900
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L901
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L902
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  #pragma HLS dependence variable=in_re2 intra false
  float in_im2[32][8];	// L903
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  #pragma HLS dependence variable=in_im2 intra false
  float out_re_b2[32][8];	// L904
  #pragma HLS array_partition variable=out_re_b2 complete dim=1
  #pragma HLS array_partition variable=out_re_b2 block factor=2 dim=2

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
  float out_im_b2[32][8];	// L905
  #pragma HLS array_partition variable=out_im_b2 complete dim=1
  #pragma HLS array_partition variable=out_im_b2 block factor=2 dim=2

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L906
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v700 = v689.read();
    hls::vector< float, 32 > v701 = v690.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L909
    #pragma HLS unroll
      int v703 = i6 >> 2;	// L910
      int32_t v704 = v703;	// L911
      int32_t v705 = v704 & 1;	// L912
      int32_t v706 = v705 << 4;	// L913
      int32_t v707 = k11;	// L914
      int32_t v708 = v707 ^ v706;	// L915
      int32_t bank5;	// L916
      bank5 = v708;	// L917
      float v710 = v700[k11];	// L918
      int32_t v711 = bank5;	// L919
      int v712 = v711;	// L920
      in_re2[v712][i6] = v710;	// L921
      float v713 = v701[k11];	// L922
      int32_t v714 = bank5;	// L923
      int v715 = v714;	// L924
      in_im2[v715][i6] = v713;	// L925
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L928
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L929
    #pragma HLS unroll
      int v718 = i7 << 4;	// L930
      int v719 = v718 | k12;	// L931
      int32_t v720 = v719;	// L932
      int32_t bg2;	// L933
      bg2 = v720;	// L934
      int32_t v722 = bg2;	// L935
      int32_t within2;	// L936
      within2 = v722;	// L937
      int32_t v724 = within2;	// L938
      int32_t tw_k5;	// L939
      tw_k5 = v724;	// L940
      int32_t v726 = within2;	// L941
      int32_t il4;	// L942
      il4 = v726;	// L943
      int32_t v728 = within2;	// L944
      int32_t v729 = v728 | 128;	// L945
      int32_t iu4;	// L946
      iu4 = v729;	// L947
      int32_t v731 = il4;	// L948
      int32_t v732 = v731 & 31;	// L949
      int32_t v733 = v731 >> 7;	// L950
      int32_t v734 = v733 & 1;	// L951
      int32_t v735 = v734 << 4;	// L952
      int32_t v736 = v732 ^ v735;	// L953
      int32_t bank_il1;	// L954
      bank_il1 = v736;	// L955
      int32_t v738 = il4;	// L956
      int32_t v739 = v738 >> 5;	// L957
      int32_t off_il1;	// L958
      off_il1 = v739;	// L959
      int32_t v741 = iu4;	// L960
      int32_t v742 = v741 & 31;	// L961
      int32_t v743 = v741 >> 7;	// L962
      int32_t v744 = v743 & 1;	// L963
      int32_t v745 = v744 << 4;	// L964
      int32_t v746 = v742 ^ v745;	// L965
      int32_t bank_iu1;	// L966
      bank_iu1 = v746;	// L967
      int32_t v748 = iu4;	// L968
      int32_t v749 = v748 >> 5;	// L969
      int32_t off_iu1;	// L970
      off_iu1 = v749;	// L971
      int32_t v751 = bank_il1;	// L972
      int v752 = v751;	// L973
      int32_t v753 = off_il1;	// L974
      int v754 = v753;	// L975
      float v755 = in_re2[v752][v754];	// L976
      float a_re7;	// L977
      a_re7 = v755;	// L978
      int32_t v757 = bank_il1;	// L979
      int v758 = v757;	// L980
      int32_t v759 = off_il1;	// L981
      int v760 = v759;	// L982
      float v761 = in_im2[v758][v760];	// L983
      float a_im7;	// L984
      a_im7 = v761;	// L985
      int32_t v763 = bank_iu1;	// L986
      int v764 = v763;	// L987
      int32_t v765 = off_iu1;	// L988
      int v766 = v765;	// L989
      float v767 = in_re2[v764][v766];	// L990
      float b_re7;	// L991
      b_re7 = v767;	// L992
      int32_t v769 = bank_iu1;	// L993
      int v770 = v769;	// L994
      int32_t v771 = off_iu1;	// L995
      int v772 = v771;	// L996
      float v773 = in_im2[v770][v772];	// L997
      float b_im7;	// L998
      b_im7 = v773;	// L999
      int32_t v775 = tw_k5;	// L1000
      int v776 = v775;	// L1001
      float v777 = twr[v776];	// L1002
      float tr5;	// L1003
      tr5 = v777;	// L1004
      int32_t v779 = tw_k5;	// L1005
      int v780 = v779;	// L1006
      float v781 = twi[v780];	// L1007
      float ti5;	// L1008
      ti5 = v781;	// L1009
      float v783 = b_re7;	// L1010
      float v784 = tr5;	// L1011
      float v785 = v783 * v784;	// L1012
      float v786 = b_im7;	// L1013
      float v787 = ti5;	// L1014
      float v788 = v786 * v787;	// L1015
      float v789 = v785 - v788;	// L1016
      float bw_re5;	// L1017
      bw_re5 = v789;	// L1018
      float v791 = b_re7;	// L1019
      float v792 = ti5;	// L1020
      float v793 = v791 * v792;	// L1021
      float v794 = b_im7;	// L1022
      float v795 = tr5;	// L1023
      float v796 = v794 * v795;	// L1024
      float v797 = v793 + v796;	// L1025
      float bw_im5;	// L1026
      bw_im5 = v797;	// L1027
      float v799 = a_re7;	// L1028
      float v800 = bw_re5;	// L1029
      float v801 = v799 + v800;	// L1030
      #pragma HLS bind_op variable=v801 op=fadd impl=fabric
      int32_t v802 = bank_il1;	// L1031
      int v803 = v802;	// L1032
      int32_t v804 = off_il1;	// L1033
      int v805 = v804;	// L1034
      out_re_b2[v803][v805] = v801;	// L1035
      float v806 = a_im7;	// L1036
      float v807 = bw_im5;	// L1037
      float v808 = v806 + v807;	// L1038
      #pragma HLS bind_op variable=v808 op=fadd impl=fabric
      int32_t v809 = bank_il1;	// L1039
      int v810 = v809;	// L1040
      int32_t v811 = off_il1;	// L1041
      int v812 = v811;	// L1042
      out_im_b2[v810][v812] = v808;	// L1043
      float v813 = a_re7;	// L1044
      float v814 = bw_re5;	// L1045
      float v815 = v813 - v814;	// L1046
      #pragma HLS bind_op variable=v815 op=fsub impl=fabric
      int32_t v816 = bank_iu1;	// L1047
      int v817 = v816;	// L1048
      int32_t v818 = off_iu1;	// L1049
      int v819 = v818;	// L1050
      out_re_b2[v817][v819] = v815;	// L1051
      float v820 = a_im7;	// L1052
      float v821 = bw_im5;	// L1053
      float v822 = v820 - v821;	// L1054
      #pragma HLS bind_op variable=v822 op=fsub impl=fabric
      int32_t v823 = bank_iu1;	// L1055
      int v824 = v823;	// L1056
      int32_t v825 = off_iu1;	// L1057
      int v826 = v825;	// L1058
      out_im_b2[v824][v826] = v822;	// L1059
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1062
  #pragma HLS pipeline II=1
    float chunk_re3[32];	// L1063
    #pragma HLS array_partition variable=chunk_re3 complete
    float chunk_im3[32];	// L1064
    #pragma HLS array_partition variable=chunk_im3 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1065
    #pragma HLS unroll
      int v831 = i8 >> 2;	// L1066
      int32_t v832 = v831;	// L1067
      int32_t v833 = v832 & 1;	// L1068
      int32_t v834 = v833 << 4;	// L1069
      int32_t v835 = k13;	// L1070
      int32_t v836 = v835 ^ v834;	// L1071
      int32_t bank6;	// L1072
      bank6 = v836;	// L1073
      int32_t v838 = bank6;	// L1074
      int v839 = v838;	// L1075
      float v840 = out_re_b2[v839][i8];	// L1076
      chunk_re3[k13] = v840;	// L1077
      int32_t v841 = bank6;	// L1078
      int v842 = v841;	// L1079
      float v843 = out_im_b2[v842][i8];	// L1080
      chunk_im3[k13] = v843;	// L1081
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re3[_iv0];
      }
      v691.write(_vec);
    }	// L1083
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im3[_iv0];
      }
      v692.write(_vec);
    }	// L1084
  }
}

void output_stage_0(
  float v844[8][32],
  float v845[8][32],
  hls::stream< hls::vector< float, 32 > >& v846,
  hls::stream< hls::vector< float, 32 > >& v847
) {	// L1088
  #pragma HLS array_partition variable=v844 complete dim=2

  #pragma HLS array_partition variable=v845 complete dim=2

  l_S_i_0_i9: for (int i9 = 0; i9 < 8; i9++) {	// L1089
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v849 = v846.read();
    hls::vector< float, 32 > v850 = v847.read();
    l_S_k_0_k14: for (int k14 = 0; k14 < 32; k14++) {	// L1092
    #pragma HLS unroll
      float v852 = v849[k14];	// L1093
      v844[i9][k14] = v852;	// L1094
      float v853 = v850[k14];	// L1095
      v845[i9][k14] = v853;	// L1096
    }
  }
}

/// This is top function.
void fft_256(
  float v854[8][32],
  float v855[8][32],
  float v856[8][32],
  float v857[8][32]
) {	// L1101
  #pragma HLS dataflow disable_start_propagation
  #pragma HLS array_partition variable=v854 complete dim=2

  #pragma HLS array_partition variable=v855 complete dim=2

  #pragma HLS array_partition variable=v856 complete dim=2

  #pragma HLS array_partition variable=v857 complete dim=2

  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v858;
  #pragma HLS stream variable=v858 depth=2	// L1102
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v859;
  #pragma HLS stream variable=v859 depth=2	// L1103
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v860;
  #pragma HLS stream variable=v860 depth=2	// L1104
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v861;
  #pragma HLS stream variable=v861 depth=2	// L1105
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v862;
  #pragma HLS stream variable=v862 depth=2	// L1106
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v863;
  #pragma HLS stream variable=v863 depth=2	// L1107
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v864;
  #pragma HLS stream variable=v864 depth=2	// L1108
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v865;
  #pragma HLS stream variable=v865 depth=2	// L1109
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v866;
  #pragma HLS stream variable=v866 depth=2	// L1110
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v867;
  #pragma HLS stream variable=v867 depth=2	// L1111
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v868;
  #pragma HLS stream variable=v868 depth=2	// L1112
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v869;
  #pragma HLS stream variable=v869 depth=2	// L1113
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v870;
  #pragma HLS stream variable=v870 depth=2	// L1114
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v871;
  #pragma HLS stream variable=v871 depth=2	// L1115
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v872;
  #pragma HLS stream variable=v872 depth=2	// L1116
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v873;
  #pragma HLS stream variable=v873 depth=2	// L1117
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v874;
  #pragma HLS stream variable=v874 depth=2	// L1118
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v875;
  #pragma HLS stream variable=v875 depth=2	// L1119
  bit_rev_stage_0(v854, v855, v858, v867);	// L1120
  intra_0_0(v858, v867, v859, v868);	// L1121
  intra_1_0(v859, v868, v860, v869);	// L1122
  intra_2_0(v860, v869, v861, v870);	// L1123
  intra_3_0(v861, v870, v862, v871);	// L1124
  intra_4_0(v862, v871, v863, v872);	// L1125
  inter_5_0(v863, v872, v864, v873);	// L1126
  inter_6_0(v864, v873, v865, v874);	// L1127
  inter_7_0(v865, v874, v866, v875);	// L1128
  output_stage_0(v856, v857, v866, v875);	// L1129
}


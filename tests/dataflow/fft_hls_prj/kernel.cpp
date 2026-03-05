
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
  #pragma HLS array_partition variable=buf_re complete

    #pragma HLS dependence variable=buf_re inter false
  float buf_im[32][8];	// L14
  #pragma HLS array_partition variable=buf_im complete

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

float twr[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 6.123234e-17, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L130
float twi[128] = {-0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L131
void intra_1_0(
  hls::stream< hls::vector< float, 32 > >& v94,
  hls::stream< hls::vector< float, 32 > >& v95,
  hls::stream< hls::vector< float, 32 > >& v96,
  hls::stream< hls::vector< float, 32 > >& v97
) {	// L132
  // placeholder for const float twr	// L138
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L139
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i1: for (int _i1 = 0; _i1 < 8; _i1++) {	// L140
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v101 = v94.read();
    hls::vector< float, 32 > v102 = v95.read();
    float o_re1[32];	// L143
    #pragma HLS array_partition variable=o_re1 complete
    float o_im1[32];	// L144
    #pragma HLS array_partition variable=o_im1 complete
    l_S_k_0_k1: for (int k1 = 0; k1 < 16; k1++) {	// L145
    #pragma HLS unroll
      int v106 = k1 >> 1;	// L146
      int v107 = v106 << 2;	// L147
      int32_t v108 = k1;	// L148
      int32_t v109 = v108 & 1;	// L149
      int32_t v110 = v107;	// L150
      int32_t v111 = v110 | v109;	// L151
      int32_t il;	// L152
      il = v111;	// L153
      int32_t v113 = il;	// L154
      int32_t v114 = v113 | 2;	// L155
      int32_t iu;	// L156
      iu = v114;	// L157
      int32_t v116 = v109 << 6;	// L158
      int32_t tw_k;	// L159
      tw_k = v116;	// L160
      int32_t v118 = il;	// L161
      int v119 = v118;	// L162
      float v120 = v101[v119];	// L163
      float a_re1;	// L164
      a_re1 = v120;	// L165
      int32_t v122 = il;	// L166
      int v123 = v122;	// L167
      float v124 = v102[v123];	// L168
      float a_im1;	// L169
      a_im1 = v124;	// L170
      int32_t v126 = iu;	// L171
      int v127 = v126;	// L172
      float v128 = v101[v127];	// L173
      float b_re1;	// L174
      b_re1 = v128;	// L175
      int32_t v130 = iu;	// L176
      int v131 = v130;	// L177
      float v132 = v102[v131];	// L178
      float b_im1;	// L179
      b_im1 = v132;	// L180
      int32_t v134 = tw_k;	// L181
      int v135 = v134;	// L182
      float v136 = twr[v135];	// L183
      float tr;	// L184
      tr = v136;	// L185
      int32_t v138 = tw_k;	// L186
      int v139 = v138;	// L187
      float v140 = twi[v139];	// L188
      float ti;	// L189
      ti = v140;	// L190
      float v142 = b_re1;	// L191
      float v143 = tr;	// L192
      float v144 = v142 * v143;	// L193
      float v145 = b_im1;	// L194
      float v146 = ti;	// L195
      float v147 = v145 * v146;	// L196
      float v148 = v144 - v147;	// L197
      #pragma HLS bind_op variable=v148 op=fsub impl=fabric
      float bw_re;	// L198
      bw_re = v148;	// L199
      float v150 = b_re1;	// L200
      float v151 = ti;	// L201
      float v152 = v150 * v151;	// L202
      float v153 = b_im1;	// L203
      float v154 = tr;	// L204
      float v155 = v153 * v154;	// L205
      float v156 = v152 + v155;	// L206
      #pragma HLS bind_op variable=v156 op=fadd impl=fabric
      float bw_im;	// L207
      bw_im = v156;	// L208
      float v158 = a_re1;	// L209
      float v159 = bw_re;	// L210
      float v160 = v158 + v159;	// L211
      #pragma HLS bind_op variable=v160 op=fadd impl=fabric
      int32_t v161 = il;	// L212
      int v162 = v161;	// L213
      o_re1[v162] = v160;	// L214
      float v163 = a_im1;	// L215
      float v164 = bw_im;	// L216
      float v165 = v163 + v164;	// L217
      #pragma HLS bind_op variable=v165 op=fadd impl=fabric
      int32_t v166 = il;	// L218
      int v167 = v166;	// L219
      o_im1[v167] = v165;	// L220
      float v168 = a_re1;	// L221
      float v169 = bw_re;	// L222
      float v170 = v168 - v169;	// L223
      #pragma HLS bind_op variable=v170 op=fsub impl=fabric
      int32_t v171 = iu;	// L224
      int v172 = v171;	// L225
      o_re1[v172] = v170;	// L226
      float v173 = a_im1;	// L227
      float v174 = bw_im;	// L228
      float v175 = v173 - v174;	// L229
      #pragma HLS bind_op variable=v175 op=fsub impl=fabric
      int32_t v176 = iu;	// L230
      int v177 = v176;	// L231
      o_im1[v177] = v175;	// L232
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re1[_iv0];
      }
      v96.write(_vec);
    }	// L234
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im1[_iv0];
      }
      v97.write(_vec);
    }	// L235
  }
}

float twr_0[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 6.123234e-17, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L239
float twi_0[128] = {-0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L240
void intra_2_0(
  hls::stream< hls::vector< float, 32 > >& v178,
  hls::stream< hls::vector< float, 32 > >& v179,
  hls::stream< hls::vector< float, 32 > >& v180,
  hls::stream< hls::vector< float, 32 > >& v181
) {	// L241
  // placeholder for const float twr_0	// L247
  #pragma HLS array_partition variable=twr_0 complete
  // placeholder for const float twi_0	// L248
  #pragma HLS array_partition variable=twi_0 complete
  l_S__i_0__i2: for (int _i2 = 0; _i2 < 8; _i2++) {	// L249
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v185 = v178.read();
    hls::vector< float, 32 > v186 = v179.read();
    float o_re2[32];	// L252
    #pragma HLS array_partition variable=o_re2 complete
    float o_im2[32];	// L253
    #pragma HLS array_partition variable=o_im2 complete
    l_S_k_0_k2: for (int k2 = 0; k2 < 16; k2++) {	// L254
    #pragma HLS unroll
      int v190 = k2 >> 2;	// L255
      int v191 = v190 << 3;	// L256
      int32_t v192 = k2;	// L257
      int32_t v193 = v192 & 3;	// L258
      int32_t v194 = v191;	// L259
      int32_t v195 = v194 | v193;	// L260
      int32_t il1;	// L261
      il1 = v195;	// L262
      int32_t v197 = il1;	// L263
      int32_t v198 = v197 | 4;	// L264
      int32_t iu1;	// L265
      iu1 = v198;	// L266
      int32_t v200 = v193 << 5;	// L267
      int32_t tw_k1;	// L268
      tw_k1 = v200;	// L269
      int32_t v202 = il1;	// L270
      int v203 = v202;	// L271
      float v204 = v185[v203];	// L272
      float a_re2;	// L273
      a_re2 = v204;	// L274
      int32_t v206 = il1;	// L275
      int v207 = v206;	// L276
      float v208 = v186[v207];	// L277
      float a_im2;	// L278
      a_im2 = v208;	// L279
      int32_t v210 = iu1;	// L280
      int v211 = v210;	// L281
      float v212 = v185[v211];	// L282
      float b_re2;	// L283
      b_re2 = v212;	// L284
      int32_t v214 = iu1;	// L285
      int v215 = v214;	// L286
      float v216 = v186[v215];	// L287
      float b_im2;	// L288
      b_im2 = v216;	// L289
      int32_t v218 = tw_k1;	// L290
      int v219 = v218;	// L291
      float v220 = twr_0[v219];	// L292
      float tr1;	// L293
      tr1 = v220;	// L294
      int32_t v222 = tw_k1;	// L295
      int v223 = v222;	// L296
      float v224 = twi_0[v223];	// L297
      float ti1;	// L298
      ti1 = v224;	// L299
      float v226 = b_re2;	// L300
      float v227 = tr1;	// L301
      float v228 = v226 * v227;	// L302
      float v229 = b_im2;	// L303
      float v230 = ti1;	// L304
      float v231 = v229 * v230;	// L305
      float v232 = v228 - v231;	// L306
      #pragma HLS bind_op variable=v232 op=fsub impl=fabric
      float bw_re1;	// L307
      bw_re1 = v232;	// L308
      float v234 = b_re2;	// L309
      float v235 = ti1;	// L310
      float v236 = v234 * v235;	// L311
      float v237 = b_im2;	// L312
      float v238 = tr1;	// L313
      float v239 = v237 * v238;	// L314
      float v240 = v236 + v239;	// L315
      #pragma HLS bind_op variable=v240 op=fadd impl=fabric
      float bw_im1;	// L316
      bw_im1 = v240;	// L317
      float v242 = a_re2;	// L318
      float v243 = bw_re1;	// L319
      float v244 = v242 + v243;	// L320
      #pragma HLS bind_op variable=v244 op=fadd impl=fabric
      int32_t v245 = il1;	// L321
      int v246 = v245;	// L322
      o_re2[v246] = v244;	// L323
      float v247 = a_im2;	// L324
      float v248 = bw_im1;	// L325
      float v249 = v247 + v248;	// L326
      #pragma HLS bind_op variable=v249 op=fadd impl=fabric
      int32_t v250 = il1;	// L327
      int v251 = v250;	// L328
      o_im2[v251] = v249;	// L329
      float v252 = a_re2;	// L330
      float v253 = bw_re1;	// L331
      float v254 = v252 - v253;	// L332
      #pragma HLS bind_op variable=v254 op=fsub impl=fabric
      int32_t v255 = iu1;	// L333
      int v256 = v255;	// L334
      o_re2[v256] = v254;	// L335
      float v257 = a_im2;	// L336
      float v258 = bw_im1;	// L337
      float v259 = v257 - v258;	// L338
      #pragma HLS bind_op variable=v259 op=fsub impl=fabric
      int32_t v260 = iu1;	// L339
      int v261 = v260;	// L340
      o_im2[v261] = v259;	// L341
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re2[_iv0];
      }
      v180.write(_vec);
    }	// L343
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im2[_iv0];
      }
      v181.write(_vec);
    }	// L344
  }
}

float twr_1[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 6.123234e-17, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L348
float twi_1[128] = {-0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L349
void intra_3_0(
  hls::stream< hls::vector< float, 32 > >& v262,
  hls::stream< hls::vector< float, 32 > >& v263,
  hls::stream< hls::vector< float, 32 > >& v264,
  hls::stream< hls::vector< float, 32 > >& v265
) {	// L350
  // placeholder for const float twr_1	// L356
  #pragma HLS array_partition variable=twr_1 complete
  // placeholder for const float twi_1	// L357
  #pragma HLS array_partition variable=twi_1 complete
  l_S__i_0__i3: for (int _i3 = 0; _i3 < 8; _i3++) {	// L358
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v269 = v262.read();
    hls::vector< float, 32 > v270 = v263.read();
    float o_re3[32];	// L361
    #pragma HLS array_partition variable=o_re3 complete
    float o_im3[32];	// L362
    #pragma HLS array_partition variable=o_im3 complete
    l_S_k_0_k3: for (int k3 = 0; k3 < 16; k3++) {	// L363
    #pragma HLS unroll
      int v274 = k3 >> 3;	// L364
      int v275 = v274 << 4;	// L365
      int32_t v276 = k3;	// L366
      int32_t v277 = v276 & 7;	// L367
      int32_t v278 = v275;	// L368
      int32_t v279 = v278 | v277;	// L369
      int32_t il2;	// L370
      il2 = v279;	// L371
      int32_t v281 = il2;	// L372
      int32_t v282 = v281 | 8;	// L373
      int32_t iu2;	// L374
      iu2 = v282;	// L375
      int32_t v284 = v277 << 4;	// L376
      int32_t tw_k2;	// L377
      tw_k2 = v284;	// L378
      int32_t v286 = il2;	// L379
      int v287 = v286;	// L380
      float v288 = v269[v287];	// L381
      float a_re3;	// L382
      a_re3 = v288;	// L383
      int32_t v290 = il2;	// L384
      int v291 = v290;	// L385
      float v292 = v270[v291];	// L386
      float a_im3;	// L387
      a_im3 = v292;	// L388
      int32_t v294 = iu2;	// L389
      int v295 = v294;	// L390
      float v296 = v269[v295];	// L391
      float b_re3;	// L392
      b_re3 = v296;	// L393
      int32_t v298 = iu2;	// L394
      int v299 = v298;	// L395
      float v300 = v270[v299];	// L396
      float b_im3;	// L397
      b_im3 = v300;	// L398
      int32_t v302 = tw_k2;	// L399
      int v303 = v302;	// L400
      float v304 = twr_1[v303];	// L401
      float tr2;	// L402
      tr2 = v304;	// L403
      int32_t v306 = tw_k2;	// L404
      int v307 = v306;	// L405
      float v308 = twi_1[v307];	// L406
      float ti2;	// L407
      ti2 = v308;	// L408
      float v310 = b_re3;	// L409
      float v311 = tr2;	// L410
      float v312 = v310 * v311;	// L411
      float v313 = b_im3;	// L412
      float v314 = ti2;	// L413
      float v315 = v313 * v314;	// L414
      float v316 = v312 - v315;	// L415
      #pragma HLS bind_op variable=v316 op=fsub impl=fabric
      float bw_re2;	// L416
      bw_re2 = v316;	// L417
      float v318 = b_re3;	// L418
      float v319 = ti2;	// L419
      float v320 = v318 * v319;	// L420
      float v321 = b_im3;	// L421
      float v322 = tr2;	// L422
      float v323 = v321 * v322;	// L423
      float v324 = v320 + v323;	// L424
      #pragma HLS bind_op variable=v324 op=fadd impl=fabric
      float bw_im2;	// L425
      bw_im2 = v324;	// L426
      float v326 = a_re3;	// L427
      float v327 = bw_re2;	// L428
      float v328 = v326 + v327;	// L429
      #pragma HLS bind_op variable=v328 op=fadd impl=fabric
      int32_t v329 = il2;	// L430
      int v330 = v329;	// L431
      o_re3[v330] = v328;	// L432
      float v331 = a_im3;	// L433
      float v332 = bw_im2;	// L434
      float v333 = v331 + v332;	// L435
      #pragma HLS bind_op variable=v333 op=fadd impl=fabric
      int32_t v334 = il2;	// L436
      int v335 = v334;	// L437
      o_im3[v335] = v333;	// L438
      float v336 = a_re3;	// L439
      float v337 = bw_re2;	// L440
      float v338 = v336 - v337;	// L441
      #pragma HLS bind_op variable=v338 op=fsub impl=fabric
      int32_t v339 = iu2;	// L442
      int v340 = v339;	// L443
      o_re3[v340] = v338;	// L444
      float v341 = a_im3;	// L445
      float v342 = bw_im2;	// L446
      float v343 = v341 - v342;	// L447
      #pragma HLS bind_op variable=v343 op=fsub impl=fabric
      int32_t v344 = iu2;	// L448
      int v345 = v344;	// L449
      o_im3[v345] = v343;	// L450
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re3[_iv0];
      }
      v264.write(_vec);
    }	// L452
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im3[_iv0];
      }
      v265.write(_vec);
    }	// L453
  }
}

float twr_2[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 6.123234e-17, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L457
float twi_2[128] = {-0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L458
void intra_4_0(
  hls::stream< hls::vector< float, 32 > >& v346,
  hls::stream< hls::vector< float, 32 > >& v347,
  hls::stream< hls::vector< float, 32 > >& v348,
  hls::stream< hls::vector< float, 32 > >& v349
) {	// L459
  // placeholder for const float twr_2	// L462
  #pragma HLS array_partition variable=twr_2 complete
  // placeholder for const float twi_2	// L463
  #pragma HLS array_partition variable=twi_2 complete
  l_S__i_0__i4: for (int _i4 = 0; _i4 < 8; _i4++) {	// L464
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v353 = v346.read();
    hls::vector< float, 32 > v354 = v347.read();
    float o_re4[32];	// L467
    #pragma HLS array_partition variable=o_re4 complete
    float o_im4[32];	// L468
    #pragma HLS array_partition variable=o_im4 complete
    l_S_k_0_k4: for (int k4 = 0; k4 < 16; k4++) {	// L469
    #pragma HLS unroll
      int32_t v358 = k4;	// L470
      int32_t il3;	// L471
      il3 = v358;	// L472
      int32_t v360 = v358 | 16;	// L473
      int32_t iu3;	// L474
      iu3 = v360;	// L475
      int v362 = k4 << 3;	// L476
      int32_t v363 = v362;	// L477
      int32_t tw_k3;	// L478
      tw_k3 = v363;	// L479
      int32_t v365 = il3;	// L480
      int v366 = v365;	// L481
      float v367 = v353[v366];	// L482
      float a_re4;	// L483
      a_re4 = v367;	// L484
      int32_t v369 = il3;	// L485
      int v370 = v369;	// L486
      float v371 = v354[v370];	// L487
      float a_im4;	// L488
      a_im4 = v371;	// L489
      int32_t v373 = iu3;	// L490
      int v374 = v373;	// L491
      float v375 = v353[v374];	// L492
      float b_re4;	// L493
      b_re4 = v375;	// L494
      int32_t v377 = iu3;	// L495
      int v378 = v377;	// L496
      float v379 = v354[v378];	// L497
      float b_im4;	// L498
      b_im4 = v379;	// L499
      int32_t v381 = tw_k3;	// L500
      int v382 = v381;	// L501
      float v383 = twr_2[v382];	// L502
      float tr3;	// L503
      tr3 = v383;	// L504
      int32_t v385 = tw_k3;	// L505
      int v386 = v385;	// L506
      float v387 = twi_2[v386];	// L507
      float ti3;	// L508
      ti3 = v387;	// L509
      float v389 = b_re4;	// L510
      float v390 = tr3;	// L511
      float v391 = v389 * v390;	// L512
      float v392 = b_im4;	// L513
      float v393 = ti3;	// L514
      float v394 = v392 * v393;	// L515
      float v395 = v391 - v394;	// L516
      #pragma HLS bind_op variable=v395 op=fsub impl=fabric
      float bw_re3;	// L517
      bw_re3 = v395;	// L518
      float v397 = b_re4;	// L519
      float v398 = ti3;	// L520
      float v399 = v397 * v398;	// L521
      float v400 = b_im4;	// L522
      float v401 = tr3;	// L523
      float v402 = v400 * v401;	// L524
      float v403 = v399 + v402;	// L525
      #pragma HLS bind_op variable=v403 op=fadd impl=fabric
      float bw_im3;	// L526
      bw_im3 = v403;	// L527
      float v405 = a_re4;	// L528
      float v406 = bw_re3;	// L529
      float v407 = v405 + v406;	// L530
      #pragma HLS bind_op variable=v407 op=fadd impl=fabric
      int32_t v408 = il3;	// L531
      int v409 = v408;	// L532
      o_re4[v409] = v407;	// L533
      float v410 = a_im4;	// L534
      float v411 = bw_im3;	// L535
      float v412 = v410 + v411;	// L536
      #pragma HLS bind_op variable=v412 op=fadd impl=fabric
      int32_t v413 = il3;	// L537
      int v414 = v413;	// L538
      o_im4[v414] = v412;	// L539
      float v415 = a_re4;	// L540
      float v416 = bw_re3;	// L541
      float v417 = v415 - v416;	// L542
      #pragma HLS bind_op variable=v417 op=fsub impl=fabric
      int32_t v418 = iu3;	// L543
      int v419 = v418;	// L544
      o_re4[v419] = v417;	// L545
      float v420 = a_im4;	// L546
      float v421 = bw_im3;	// L547
      float v422 = v420 - v421;	// L548
      #pragma HLS bind_op variable=v422 op=fsub impl=fabric
      int32_t v423 = iu3;	// L549
      int v424 = v423;	// L550
      o_im4[v424] = v422;	// L551
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re4[_iv0];
      }
      v348.write(_vec);
    }	// L553
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im4[_iv0];
      }
      v349.write(_vec);
    }	// L554
  }
}

float twr_3[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 6.123234e-17, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L558
float twi_3[128] = {-0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L559
void inter_5_0(
  hls::stream< hls::vector< float, 32 > >& v425,
  hls::stream< hls::vector< float, 32 > >& v426,
  hls::stream< hls::vector< float, 32 > >& v427,
  hls::stream< hls::vector< float, 32 > >& v428
) {	// L560
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr_3	// L568
  #pragma HLS array_partition variable=twr_3 complete
  // placeholder for const float twi_3	// L569
  #pragma HLS array_partition variable=twi_3 complete
  float in_re[32][8];	// L570
  #pragma HLS array_partition variable=in_re complete

    #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L571
  #pragma HLS array_partition variable=in_im complete

    #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L572
  #pragma HLS array_partition variable=out_re_b complete

    #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L573
  #pragma HLS array_partition variable=out_im_b complete

    #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L574
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v436 = v425.read();
    hls::vector< float, 32 > v437 = v426.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L577
    #pragma HLS unroll
      int32_t v439 = i;	// L578
      int32_t v440 = v439 & 1;	// L579
      int32_t v441 = v440 << 4;	// L580
      int32_t v442 = k5;	// L581
      int32_t v443 = v442 ^ v441;	// L582
      int32_t bank1;	// L583
      bank1 = v443;	// L584
      float v445 = v436[k5];	// L585
      int32_t v446 = bank1;	// L586
      int v447 = v446;	// L587
      in_re[v447][i] = v445;	// L588
      float v448 = v437[k5];	// L589
      int32_t v449 = bank1;	// L590
      int v450 = v449;	// L591
      in_im[v450][i] = v448;	// L592
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L595
  #pragma HLS pipeline II=1
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L596
    #pragma HLS unroll
      int v453 = i1 << 4;	// L597
      int v454 = v453 | k6;	// L598
      int32_t v455 = v454;	// L599
      int32_t bg;	// L600
      bg = v455;	// L601
      int32_t v457 = bg;	// L602
      int32_t v458 = v457 >> 5;	// L603
      int32_t grp;	// L604
      grp = v458;	// L605
      int32_t v460 = bg;	// L606
      int32_t v461 = v460 & 31;	// L607
      int32_t within;	// L608
      within = v461;	// L609
      int32_t v463 = within;	// L610
      int32_t v464 = v463 << 2;	// L611
      int32_t tw_k4;	// L612
      tw_k4 = v464;	// L613
      int32_t v466 = grp;	// L614
      int32_t v467 = v466 << 1;	// L615
      int32_t off_l;	// L616
      off_l = v467;	// L617
      int32_t v469 = off_l;	// L618
      int32_t v470 = v469 | 1;	// L619
      int32_t off_u;	// L620
      off_u = v470;	// L621
      int32_t v472 = within;	// L622
      int v473 = v472;	// L623
      int32_t v474 = off_l;	// L624
      int v475 = v474;	// L625
      float v476 = in_re[v473][v475];	// L626
      float a_re5;	// L627
      a_re5 = v476;	// L628
      int32_t v478 = within;	// L629
      int v479 = v478;	// L630
      int32_t v480 = off_l;	// L631
      int v481 = v480;	// L632
      float v482 = in_im[v479][v481];	// L633
      float a_im5;	// L634
      a_im5 = v482;	// L635
      int32_t v484 = within;	// L636
      int32_t v485 = v484 ^ 16;	// L637
      int v486 = v485;	// L638
      int32_t v487 = off_u;	// L639
      int v488 = v487;	// L640
      float v489 = in_re[v486][v488];	// L641
      float b_re5;	// L642
      b_re5 = v489;	// L643
      int32_t v491 = within;	// L644
      int32_t v492 = v491 ^ 16;	// L645
      int v493 = v492;	// L646
      int32_t v494 = off_u;	// L647
      int v495 = v494;	// L648
      float v496 = in_im[v493][v495];	// L649
      float b_im5;	// L650
      b_im5 = v496;	// L651
      int32_t v498 = tw_k4;	// L652
      int v499 = v498;	// L653
      float v500 = twr_3[v499];	// L654
      float tr4;	// L655
      tr4 = v500;	// L656
      int32_t v502 = tw_k4;	// L657
      int v503 = v502;	// L658
      float v504 = twi_3[v503];	// L659
      float ti4;	// L660
      ti4 = v504;	// L661
      float v506 = b_re5;	// L662
      float v507 = tr4;	// L663
      float v508 = v506 * v507;	// L664
      float v509 = b_im5;	// L665
      float v510 = ti4;	// L666
      float v511 = v509 * v510;	// L667
      float v512 = v508 - v511;	// L668
      #pragma HLS bind_op variable=v512 op=fsub impl=fabric
      float bw_re4;	// L669
      bw_re4 = v512;	// L670
      float v514 = b_re5;	// L671
      float v515 = ti4;	// L672
      float v516 = v514 * v515;	// L673
      float v517 = b_im5;	// L674
      float v518 = tr4;	// L675
      float v519 = v517 * v518;	// L676
      float v520 = v516 + v519;	// L677
      #pragma HLS bind_op variable=v520 op=fadd impl=fabric
      float bw_im4;	// L678
      bw_im4 = v520;	// L679
      float v522 = a_re5;	// L680
      float v523 = bw_re4;	// L681
      float v524 = v522 + v523;	// L682
      #pragma HLS bind_op variable=v524 op=fadd impl=fabric
      int32_t v525 = within;	// L683
      int v526 = v525;	// L684
      int32_t v527 = off_l;	// L685
      int v528 = v527;	// L686
      out_re_b[v526][v528] = v524;	// L687
      float v529 = a_im5;	// L688
      float v530 = bw_im4;	// L689
      float v531 = v529 + v530;	// L690
      #pragma HLS bind_op variable=v531 op=fadd impl=fabric
      int32_t v532 = within;	// L691
      int v533 = v532;	// L692
      int32_t v534 = off_l;	// L693
      int v535 = v534;	// L694
      out_im_b[v533][v535] = v531;	// L695
      float v536 = a_re5;	// L696
      float v537 = bw_re4;	// L697
      float v538 = v536 - v537;	// L698
      #pragma HLS bind_op variable=v538 op=fsub impl=fabric
      int32_t v539 = within;	// L699
      int32_t v540 = v539 ^ 16;	// L700
      int v541 = v540;	// L701
      int32_t v542 = off_u;	// L702
      int v543 = v542;	// L703
      out_re_b[v541][v543] = v538;	// L704
      float v544 = a_im5;	// L705
      float v545 = bw_im4;	// L706
      float v546 = v544 - v545;	// L707
      #pragma HLS bind_op variable=v546 op=fsub impl=fabric
      int32_t v547 = within;	// L708
      int32_t v548 = v547 ^ 16;	// L709
      int v549 = v548;	// L710
      int32_t v550 = off_u;	// L711
      int v551 = v550;	// L712
      out_im_b[v549][v551] = v546;	// L713
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L716
  #pragma HLS pipeline II=1
    float chunk_re1[32];	// L717
    #pragma HLS array_partition variable=chunk_re1 complete
    float chunk_im1[32];	// L718
    #pragma HLS array_partition variable=chunk_im1 complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L719
    #pragma HLS unroll
      int32_t v556 = i2;	// L720
      int32_t v557 = v556 & 1;	// L721
      int32_t v558 = v557 << 4;	// L722
      int32_t v559 = k7;	// L723
      int32_t v560 = v559 ^ v558;	// L724
      int32_t bank2;	// L725
      bank2 = v560;	// L726
      int32_t v562 = bank2;	// L727
      int v563 = v562;	// L728
      float v564 = out_re_b[v563][i2];	// L729
      chunk_re1[k7] = v564;	// L730
      int32_t v565 = bank2;	// L731
      int v566 = v565;	// L732
      float v567 = out_im_b[v566][i2];	// L733
      chunk_im1[k7] = v567;	// L734
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re1[_iv0];
      }
      v427.write(_vec);
    }	// L736
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im1[_iv0];
      }
      v428.write(_vec);
    }	// L737
  }
}

float twr_4[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 6.123234e-17, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L741
float twi_4[128] = {-0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L742
void inter_6_0(
  hls::stream< hls::vector< float, 32 > >& v568,
  hls::stream< hls::vector< float, 32 > >& v569,
  hls::stream< hls::vector< float, 32 > >& v570,
  hls::stream< hls::vector< float, 32 > >& v571
) {	// L743
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr_4	// L754
  #pragma HLS array_partition variable=twr_4 complete
  // placeholder for const float twi_4	// L755
  #pragma HLS array_partition variable=twi_4 complete
  float in_re1[32][8];	// L756
  #pragma HLS array_partition variable=in_re1 complete

    #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L757
  #pragma HLS array_partition variable=in_im1 complete

    #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L758
  #pragma HLS array_partition variable=out_re_b1 complete

    #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L759
  #pragma HLS array_partition variable=out_im_b1 complete

    #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L760
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v579 = v568.read();
    hls::vector< float, 32 > v580 = v569.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L763
    #pragma HLS unroll
      int v582 = i3 >> 1;	// L764
      int32_t v583 = v582;	// L765
      int32_t v584 = v583 & 1;	// L766
      int32_t v585 = v584 << 4;	// L767
      int32_t v586 = k8;	// L768
      int32_t v587 = v586 ^ v585;	// L769
      int32_t bank3;	// L770
      bank3 = v587;	// L771
      float v589 = v579[k8];	// L772
      int32_t v590 = bank3;	// L773
      int v591 = v590;	// L774
      in_re1[v591][i3] = v589;	// L775
      float v592 = v580[k8];	// L776
      int32_t v593 = bank3;	// L777
      int v594 = v593;	// L778
      in_im1[v594][i3] = v592;	// L779
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L782
  #pragma HLS pipeline II=1
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L783
    #pragma HLS unroll
      int v597 = i4 << 4;	// L784
      int v598 = v597 | k9;	// L785
      int32_t v599 = v598;	// L786
      int32_t bg1;	// L787
      bg1 = v599;	// L788
      int32_t v601 = bg1;	// L789
      int32_t v602 = v601 >> 6;	// L790
      int32_t grp1;	// L791
      grp1 = v602;	// L792
      int32_t v604 = bg1;	// L793
      int32_t v605 = v604 & 63;	// L794
      int32_t within1;	// L795
      within1 = v605;	// L796
      int32_t v607 = within1;	// L797
      int32_t v608 = v607 << 1;	// L798
      int32_t tw_k5;	// L799
      tw_k5 = v608;	// L800
      int32_t v610 = grp1;	// L801
      int32_t v611 = v610 << 7;	// L802
      int32_t v612 = within1;	// L803
      int32_t v613 = v611 | v612;	// L804
      int32_t il4;	// L805
      il4 = v613;	// L806
      int32_t v615 = il4;	// L807
      int32_t v616 = v615 | 64;	// L808
      int32_t iu4;	// L809
      iu4 = v616;	// L810
      int32_t v618 = il4;	// L811
      int32_t v619 = v618 & 31;	// L812
      int32_t v620 = v618 >> 6;	// L813
      int32_t v621 = v620 & 1;	// L814
      int32_t v622 = v621 << 4;	// L815
      int32_t v623 = v619 ^ v622;	// L816
      int32_t bank_il;	// L817
      bank_il = v623;	// L818
      int32_t v625 = il4;	// L819
      int32_t v626 = v625 >> 5;	// L820
      int32_t off_il;	// L821
      off_il = v626;	// L822
      int32_t v628 = iu4;	// L823
      int32_t v629 = v628 & 31;	// L824
      int32_t v630 = v628 >> 6;	// L825
      int32_t v631 = v630 & 1;	// L826
      int32_t v632 = v631 << 4;	// L827
      int32_t v633 = v629 ^ v632;	// L828
      int32_t bank_iu;	// L829
      bank_iu = v633;	// L830
      int32_t v635 = iu4;	// L831
      int32_t v636 = v635 >> 5;	// L832
      int32_t off_iu;	// L833
      off_iu = v636;	// L834
      int32_t v638 = bank_il;	// L835
      int v639 = v638;	// L836
      int32_t v640 = off_il;	// L837
      int v641 = v640;	// L838
      float v642 = in_re1[v639][v641];	// L839
      float a_re6;	// L840
      a_re6 = v642;	// L841
      int32_t v644 = bank_il;	// L842
      int v645 = v644;	// L843
      int32_t v646 = off_il;	// L844
      int v647 = v646;	// L845
      float v648 = in_im1[v645][v647];	// L846
      float a_im6;	// L847
      a_im6 = v648;	// L848
      int32_t v650 = bank_iu;	// L849
      int v651 = v650;	// L850
      int32_t v652 = off_iu;	// L851
      int v653 = v652;	// L852
      float v654 = in_re1[v651][v653];	// L853
      float b_re6;	// L854
      b_re6 = v654;	// L855
      int32_t v656 = bank_iu;	// L856
      int v657 = v656;	// L857
      int32_t v658 = off_iu;	// L858
      int v659 = v658;	// L859
      float v660 = in_im1[v657][v659];	// L860
      float b_im6;	// L861
      b_im6 = v660;	// L862
      int32_t v662 = tw_k5;	// L863
      int v663 = v662;	// L864
      float v664 = twr_4[v663];	// L865
      float tr5;	// L866
      tr5 = v664;	// L867
      int32_t v666 = tw_k5;	// L868
      int v667 = v666;	// L869
      float v668 = twi_4[v667];	// L870
      float ti5;	// L871
      ti5 = v668;	// L872
      float v670 = b_re6;	// L873
      float v671 = tr5;	// L874
      float v672 = v670 * v671;	// L875
      float v673 = b_im6;	// L876
      float v674 = ti5;	// L877
      float v675 = v673 * v674;	// L878
      float v676 = v672 - v675;	// L879
      #pragma HLS bind_op variable=v676 op=fsub impl=fabric
      float bw_re5;	// L880
      bw_re5 = v676;	// L881
      float v678 = b_re6;	// L882
      float v679 = ti5;	// L883
      float v680 = v678 * v679;	// L884
      float v681 = b_im6;	// L885
      float v682 = tr5;	// L886
      float v683 = v681 * v682;	// L887
      float v684 = v680 + v683;	// L888
      #pragma HLS bind_op variable=v684 op=fadd impl=fabric
      float bw_im5;	// L889
      bw_im5 = v684;	// L890
      float v686 = a_re6;	// L891
      float v687 = bw_re5;	// L892
      float v688 = v686 + v687;	// L893
      #pragma HLS bind_op variable=v688 op=fadd impl=fabric
      int32_t v689 = bank_il;	// L894
      int v690 = v689;	// L895
      int32_t v691 = off_il;	// L896
      int v692 = v691;	// L897
      out_re_b1[v690][v692] = v688;	// L898
      float v693 = a_im6;	// L899
      float v694 = bw_im5;	// L900
      float v695 = v693 + v694;	// L901
      #pragma HLS bind_op variable=v695 op=fadd impl=fabric
      int32_t v696 = bank_il;	// L902
      int v697 = v696;	// L903
      int32_t v698 = off_il;	// L904
      int v699 = v698;	// L905
      out_im_b1[v697][v699] = v695;	// L906
      float v700 = a_re6;	// L907
      float v701 = bw_re5;	// L908
      float v702 = v700 - v701;	// L909
      #pragma HLS bind_op variable=v702 op=fsub impl=fabric
      int32_t v703 = bank_iu;	// L910
      int v704 = v703;	// L911
      int32_t v705 = off_iu;	// L912
      int v706 = v705;	// L913
      out_re_b1[v704][v706] = v702;	// L914
      float v707 = a_im6;	// L915
      float v708 = bw_im5;	// L916
      float v709 = v707 - v708;	// L917
      #pragma HLS bind_op variable=v709 op=fsub impl=fabric
      int32_t v710 = bank_iu;	// L918
      int v711 = v710;	// L919
      int32_t v712 = off_iu;	// L920
      int v713 = v712;	// L921
      out_im_b1[v711][v713] = v709;	// L922
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L925
  #pragma HLS pipeline II=1
    float chunk_re2[32];	// L926
    #pragma HLS array_partition variable=chunk_re2 complete
    float chunk_im2[32];	// L927
    #pragma HLS array_partition variable=chunk_im2 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L928
    #pragma HLS unroll
      int v718 = i5 >> 1;	// L929
      int32_t v719 = v718;	// L930
      int32_t v720 = v719 & 1;	// L931
      int32_t v721 = v720 << 4;	// L932
      int32_t v722 = k10;	// L933
      int32_t v723 = v722 ^ v721;	// L934
      int32_t bank4;	// L935
      bank4 = v723;	// L936
      int32_t v725 = bank4;	// L937
      int v726 = v725;	// L938
      float v727 = out_re_b1[v726][i5];	// L939
      chunk_re2[k10] = v727;	// L940
      int32_t v728 = bank4;	// L941
      int v729 = v728;	// L942
      float v730 = out_im_b1[v729][i5];	// L943
      chunk_im2[k10] = v730;	// L944
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re2[_iv0];
      }
      v570.write(_vec);
    }	// L946
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im2[_iv0];
      }
      v571.write(_vec);
    }	// L947
  }
}

float twr_5[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 6.123234e-17, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L951
float twi_5[128] = {-0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L952
void inter_7_0(
  hls::stream< hls::vector< float, 32 > >& v731,
  hls::stream< hls::vector< float, 32 > >& v732,
  hls::stream< hls::vector< float, 32 > >& v733,
  hls::stream< hls::vector< float, 32 > >& v734
) {	// L953
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr_5	// L962
  #pragma HLS array_partition variable=twr_5 complete
  // placeholder for const float twi_5	// L963
  #pragma HLS array_partition variable=twi_5 complete
  float in_re2[32][8];	// L964
  #pragma HLS array_partition variable=in_re2 complete

    #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L965
  #pragma HLS array_partition variable=in_im2 complete

    #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L966
  #pragma HLS array_partition variable=out_re_b2 complete

    #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L967
  #pragma HLS array_partition variable=out_im_b2 complete

    #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L968
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v742 = v731.read();
    hls::vector< float, 32 > v743 = v732.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L971
    #pragma HLS unroll
      int v745 = i6 >> 2;	// L972
      int32_t v746 = v745;	// L973
      int32_t v747 = v746 & 1;	// L974
      int32_t v748 = v747 << 4;	// L975
      int32_t v749 = k11;	// L976
      int32_t v750 = v749 ^ v748;	// L977
      int32_t bank5;	// L978
      bank5 = v750;	// L979
      float v752 = v742[k11];	// L980
      int32_t v753 = bank5;	// L981
      int v754 = v753;	// L982
      in_re2[v754][i6] = v752;	// L983
      float v755 = v743[k11];	// L984
      int32_t v756 = bank5;	// L985
      int v757 = v756;	// L986
      in_im2[v757][i6] = v755;	// L987
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L990
  #pragma HLS pipeline II=1
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L991
    #pragma HLS unroll
      int v760 = i7 << 4;	// L992
      int v761 = v760 | k12;	// L993
      int32_t v762 = v761;	// L994
      int32_t bg2;	// L995
      bg2 = v762;	// L996
      int32_t v764 = bg2;	// L997
      int32_t within2;	// L998
      within2 = v764;	// L999
      int32_t v766 = within2;	// L1000
      int32_t tw_k6;	// L1001
      tw_k6 = v766;	// L1002
      int32_t v768 = within2;	// L1003
      int32_t il5;	// L1004
      il5 = v768;	// L1005
      int32_t v770 = within2;	// L1006
      int32_t v771 = v770 | 128;	// L1007
      int32_t iu5;	// L1008
      iu5 = v771;	// L1009
      int32_t v773 = il5;	// L1010
      int32_t v774 = v773 & 31;	// L1011
      int32_t v775 = v773 >> 7;	// L1012
      int32_t v776 = v775 & 1;	// L1013
      int32_t v777 = v776 << 4;	// L1014
      int32_t v778 = v774 ^ v777;	// L1015
      int32_t bank_il1;	// L1016
      bank_il1 = v778;	// L1017
      int32_t v780 = il5;	// L1018
      int32_t v781 = v780 >> 5;	// L1019
      int32_t off_il1;	// L1020
      off_il1 = v781;	// L1021
      int32_t v783 = iu5;	// L1022
      int32_t v784 = v783 & 31;	// L1023
      int32_t v785 = v783 >> 7;	// L1024
      int32_t v786 = v785 & 1;	// L1025
      int32_t v787 = v786 << 4;	// L1026
      int32_t v788 = v784 ^ v787;	// L1027
      int32_t bank_iu1;	// L1028
      bank_iu1 = v788;	// L1029
      int32_t v790 = iu5;	// L1030
      int32_t v791 = v790 >> 5;	// L1031
      int32_t off_iu1;	// L1032
      off_iu1 = v791;	// L1033
      int32_t v793 = bank_il1;	// L1034
      int v794 = v793;	// L1035
      int32_t v795 = off_il1;	// L1036
      int v796 = v795;	// L1037
      float v797 = in_re2[v794][v796];	// L1038
      float a_re7;	// L1039
      a_re7 = v797;	// L1040
      int32_t v799 = bank_il1;	// L1041
      int v800 = v799;	// L1042
      int32_t v801 = off_il1;	// L1043
      int v802 = v801;	// L1044
      float v803 = in_im2[v800][v802];	// L1045
      float a_im7;	// L1046
      a_im7 = v803;	// L1047
      int32_t v805 = bank_iu1;	// L1048
      int v806 = v805;	// L1049
      int32_t v807 = off_iu1;	// L1050
      int v808 = v807;	// L1051
      float v809 = in_re2[v806][v808];	// L1052
      float b_re7;	// L1053
      b_re7 = v809;	// L1054
      int32_t v811 = bank_iu1;	// L1055
      int v812 = v811;	// L1056
      int32_t v813 = off_iu1;	// L1057
      int v814 = v813;	// L1058
      float v815 = in_im2[v812][v814];	// L1059
      float b_im7;	// L1060
      b_im7 = v815;	// L1061
      int32_t v817 = tw_k6;	// L1062
      int v818 = v817;	// L1063
      float v819 = twr_5[v818];	// L1064
      float tr6;	// L1065
      tr6 = v819;	// L1066
      int32_t v821 = tw_k6;	// L1067
      int v822 = v821;	// L1068
      float v823 = twi_5[v822];	// L1069
      float ti6;	// L1070
      ti6 = v823;	// L1071
      float v825 = b_re7;	// L1072
      float v826 = tr6;	// L1073
      float v827 = v825 * v826;	// L1074
      float v828 = b_im7;	// L1075
      float v829 = ti6;	// L1076
      float v830 = v828 * v829;	// L1077
      float v831 = v827 - v830;	// L1078
      #pragma HLS bind_op variable=v831 op=fsub impl=fabric
      float bw_re6;	// L1079
      bw_re6 = v831;	// L1080
      float v833 = b_re7;	// L1081
      float v834 = ti6;	// L1082
      float v835 = v833 * v834;	// L1083
      float v836 = b_im7;	// L1084
      float v837 = tr6;	// L1085
      float v838 = v836 * v837;	// L1086
      float v839 = v835 + v838;	// L1087
      #pragma HLS bind_op variable=v839 op=fadd impl=fabric
      float bw_im6;	// L1088
      bw_im6 = v839;	// L1089
      float v841 = a_re7;	// L1090
      float v842 = bw_re6;	// L1091
      float v843 = v841 + v842;	// L1092
      #pragma HLS bind_op variable=v843 op=fadd impl=fabric
      int32_t v844 = bank_il1;	// L1093
      int v845 = v844;	// L1094
      int32_t v846 = off_il1;	// L1095
      int v847 = v846;	// L1096
      out_re_b2[v845][v847] = v843;	// L1097
      float v848 = a_im7;	// L1098
      float v849 = bw_im6;	// L1099
      float v850 = v848 + v849;	// L1100
      #pragma HLS bind_op variable=v850 op=fadd impl=fabric
      int32_t v851 = bank_il1;	// L1101
      int v852 = v851;	// L1102
      int32_t v853 = off_il1;	// L1103
      int v854 = v853;	// L1104
      out_im_b2[v852][v854] = v850;	// L1105
      float v855 = a_re7;	// L1106
      float v856 = bw_re6;	// L1107
      float v857 = v855 - v856;	// L1108
      #pragma HLS bind_op variable=v857 op=fsub impl=fabric
      int32_t v858 = bank_iu1;	// L1109
      int v859 = v858;	// L1110
      int32_t v860 = off_iu1;	// L1111
      int v861 = v860;	// L1112
      out_re_b2[v859][v861] = v857;	// L1113
      float v862 = a_im7;	// L1114
      float v863 = bw_im6;	// L1115
      float v864 = v862 - v863;	// L1116
      #pragma HLS bind_op variable=v864 op=fsub impl=fabric
      int32_t v865 = bank_iu1;	// L1117
      int v866 = v865;	// L1118
      int32_t v867 = off_iu1;	// L1119
      int v868 = v867;	// L1120
      out_im_b2[v866][v868] = v864;	// L1121
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1124
  #pragma HLS pipeline II=1
    float chunk_re3[32];	// L1125
    #pragma HLS array_partition variable=chunk_re3 complete
    float chunk_im3[32];	// L1126
    #pragma HLS array_partition variable=chunk_im3 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1127
    #pragma HLS unroll
      int v873 = i8 >> 2;	// L1128
      int32_t v874 = v873;	// L1129
      int32_t v875 = v874 & 1;	// L1130
      int32_t v876 = v875 << 4;	// L1131
      int32_t v877 = k13;	// L1132
      int32_t v878 = v877 ^ v876;	// L1133
      int32_t bank6;	// L1134
      bank6 = v878;	// L1135
      int32_t v880 = bank6;	// L1136
      int v881 = v880;	// L1137
      float v882 = out_re_b2[v881][i8];	// L1138
      chunk_re3[k13] = v882;	// L1139
      int32_t v883 = bank6;	// L1140
      int v884 = v883;	// L1141
      float v885 = out_im_b2[v884][i8];	// L1142
      chunk_im3[k13] = v885;	// L1143
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re3[_iv0];
      }
      v733.write(_vec);
    }	// L1145
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im3[_iv0];
      }
      v734.write(_vec);
    }	// L1146
  }
}

void output_stage_0(
  float v886[8][32],
  float v887[8][32],
  hls::stream< hls::vector< float, 32 > >& v888,
  hls::stream< hls::vector< float, 32 > >& v889
) {	// L1150
  #pragma HLS array_partition variable=v886 complete dim=2

  #pragma HLS array_partition variable=v887 complete dim=2

  l_S_i_0_i9: for (int i9 = 0; i9 < 8; i9++) {	// L1151
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v891 = v888.read();
    hls::vector< float, 32 > v892 = v889.read();
    l_S_k_0_k14: for (int k14 = 0; k14 < 32; k14++) {	// L1154
    #pragma HLS unroll
      float v894 = v891[k14];	// L1155
      v886[i9][k14] = v894;	// L1156
      float v895 = v892[k14];	// L1157
      v887[i9][k14] = v895;	// L1158
    }
  }
}

void load_buf0(
  float v896[8][32],
  float v897[8][32]
) {	//
  #pragma HLS array_partition variable=v896 complete dim=2

  #pragma HLS array_partition variable=v897 complete dim=2

  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 8; load_buf0_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    l_load_buf0_l_1: for (int load_buf0_l_1 = 0; load_buf0_l_1 < 32; load_buf0_l_1++) {	//
    #pragma HLS unroll
      float v900 = v896[load_buf0_l_0][load_buf0_l_1];	//
      v897[load_buf0_l_0][load_buf0_l_1] = v900;	//
    }
  }
}

void load_buf1(
  float v901[8][32],
  float v902[8][32]
) {	//
  #pragma HLS array_partition variable=v901 complete dim=2

  #pragma HLS array_partition variable=v902 complete dim=2

  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 8; load_buf1_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    l_load_buf1_l_1: for (int load_buf1_l_1 = 0; load_buf1_l_1 < 32; load_buf1_l_1++) {	//
    #pragma HLS unroll
      float v905 = v901[load_buf1_l_0][load_buf1_l_1];	//
      v902[load_buf1_l_0][load_buf1_l_1] = v905;	//
    }
  }
}

void store_res2(
  float v906[8][32],
  float v907[8][32]
) {	//
  #pragma HLS array_partition variable=v906 complete dim=2

  #pragma HLS array_partition variable=v907 complete dim=2

  l_S_store_res2_store_res2_l_0: for (int store_res2_l_0 = 0; store_res2_l_0 < 8; store_res2_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    l_store_res2_l_1: for (int store_res2_l_1 = 0; store_res2_l_1 < 32; store_res2_l_1++) {	//
    #pragma HLS unroll
      float v910 = v906[store_res2_l_0][store_res2_l_1];	//
      v907[store_res2_l_0][store_res2_l_1] = v910;	//
    }
  }
}

void store_res3(
  float v911[8][32],
  float v912[8][32]
) {	//
  #pragma HLS array_partition variable=v911 complete dim=2

  #pragma HLS array_partition variable=v912 complete dim=2

  l_S_store_res3_store_res3_l_0: for (int store_res3_l_0 = 0; store_res3_l_0 < 8; store_res3_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    l_store_res3_l_1: for (int store_res3_l_1 = 0; store_res3_l_1 < 32; store_res3_l_1++) {	//
    #pragma HLS unroll
      float v915 = v911[store_res3_l_0][store_res3_l_1];	//
      v912[store_res3_l_0][store_res3_l_1] = v915;	//
    }
  }
}

/// This is top function.
void fft_256(
  float v916[8][32],
  float v917[8][32],
  float v918[8][32],
  float v919[8][32]
) {	// L1163
  #pragma HLS dataflow disable_start_propagation
  #pragma HLS array_partition variable=v916 complete dim=2

  #pragma HLS array_partition variable=v917 complete dim=2

  #pragma HLS array_partition variable=v918 complete dim=2

  #pragma HLS array_partition variable=v919 complete dim=2

  float buf0[8][32];	//
  #pragma HLS array_partition variable=buf0 complete dim=2

  load_buf0(v916, buf0);	//
  float buf1[8][32];	//
  #pragma HLS array_partition variable=buf1 complete dim=2

  load_buf1(v917, buf1);	//
  float buf2[8][32];	//
  #pragma HLS array_partition variable=buf2 complete dim=2

  float buf3[8][32];	//
  #pragma HLS array_partition variable=buf3 complete dim=2

  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v924;
  #pragma HLS stream variable=v924 depth=16	// L1164
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v925;
  #pragma HLS stream variable=v925 depth=16	// L1165
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v926;
  #pragma HLS stream variable=v926 depth=16	// L1166
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v927;
  #pragma HLS stream variable=v927 depth=16	// L1167
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v928;
  #pragma HLS stream variable=v928 depth=16	// L1168
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v929;
  #pragma HLS stream variable=v929 depth=16	// L1169
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v930;
  #pragma HLS stream variable=v930 depth=16	// L1170
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v931;
  #pragma HLS stream variable=v931 depth=16	// L1171
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v932;
  #pragma HLS stream variable=v932 depth=16	// L1172
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v933;
  #pragma HLS stream variable=v933 depth=16	// L1173
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v934;
  #pragma HLS stream variable=v934 depth=16	// L1174
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v935;
  #pragma HLS stream variable=v935 depth=16	// L1175
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v936;
  #pragma HLS stream variable=v936 depth=16	// L1176
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v937;
  #pragma HLS stream variable=v937 depth=16	// L1177
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v938;
  #pragma HLS stream variable=v938 depth=16	// L1178
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v939;
  #pragma HLS stream variable=v939 depth=16	// L1179
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v940;
  #pragma HLS stream variable=v940 depth=16	// L1180
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v941;
  #pragma HLS stream variable=v941 depth=16	// L1181
  bit_rev_stage_0(buf0, buf1, v924, v933);	// L1182
  intra_0_0(v924, v933, v925, v934);	// L1183
  intra_1_0(v925, v934, v926, v935);	// L1184
  intra_2_0(v926, v935, v927, v936);	// L1185
  intra_3_0(v927, v936, v928, v937);	// L1186
  intra_4_0(v928, v937, v929, v938);	// L1187
  inter_5_0(v929, v938, v930, v939);	// L1188
  inter_6_0(v930, v939, v931, v940);	// L1189
  inter_7_0(v931, v940, v932, v941);	// L1190
  output_stage_0(buf2, buf3, v932, v941);	// L1191
  store_res2(buf2, v918);	//
  store_res3(buf3, v919);	//
}


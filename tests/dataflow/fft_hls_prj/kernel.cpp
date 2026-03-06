
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
  hls::stream< hls::vector< float, 32 > >& v0,
  hls::stream< hls::vector< float, 32 > >& v1,
  hls::stream< hls::vector< float, 32 > >& v2,
  hls::stream< hls::vector< float, 32 > >& v3
) {	// L3
  #pragma HLS dataflow disable_start_propagation
  float buf_re[32][8];	// L12
  #pragma HLS array_partition variable=buf_re complete dim=1

  #pragma HLS bind_storage variable=buf_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=buf_re inter false
  float buf_im[32][8];	// L13
  #pragma HLS array_partition variable=buf_im complete dim=1

  #pragma HLS bind_storage variable=buf_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=buf_im inter false
  l_S_ii_0_ii: for (int ii = 0; ii < 8; ii++) {	// L14
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v7 = v0.read();
    hls::vector< float, 32 > v8 = v1.read();
    l_S_kk_0_kk: for (int kk = 0; kk < 32; kk++) {	// L17
    #pragma HLS unroll
      int32_t v10 = kk;	// L18
      int32_t v11 = v10 & 1;	// L19
      int32_t v12 = v11 << 4;	// L20
      int32_t v13 = v10 & 2;	// L21
      int32_t v14 = v13 << 2;	// L22
      int32_t v15 = v12 | v14;	// L23
      int32_t v16 = v10 & 4;	// L24
      int32_t v17 = v15 | v16;	// L25
      int32_t v18 = v10 & 8;	// L26
      int32_t v19 = v18 >> 2;	// L27
      int32_t v20 = v17 | v19;	// L28
      int32_t v21 = v10 & 16;	// L29
      int32_t v22 = v21 >> 4;	// L30
      int32_t v23 = v20 | v22;	// L31
      int32_t bank;	// L32
      bank = v23;	// L33
      int32_t v25 = ii;	// L34
      int32_t v26 = v25 & 4;	// L35
      int32_t v27 = v26 >> 2;	// L36
      int32_t v28 = v25 & 2;	// L37
      int32_t v29 = v27 | v28;	// L38
      int32_t v30 = v25 & 1;	// L39
      int32_t v31 = v30 << 2;	// L40
      int32_t v32 = v29 | v31;	// L41
      int32_t offset;	// L42
      offset = v32;	// L43
      float v34 = v7[kk];	// L44
      int32_t v35 = bank;	// L45
      int v36 = v35;	// L46
      int32_t v37 = offset;	// L47
      int v38 = v37;	// L48
      buf_re[v36][v38] = v34;	// L49
      float v39 = v8[kk];	// L50
      int32_t v40 = bank;	// L51
      int v41 = v40;	// L52
      int32_t v42 = offset;	// L53
      int v43 = v42;	// L54
      buf_im[v41][v43] = v39;	// L55
    }
  }
  l_S_jj_2_jj: for (int jj = 0; jj < 8; jj++) {	// L58
  #pragma HLS pipeline II=1
    float chunk_re[32];	// L59
    #pragma HLS array_partition variable=chunk_re complete
    float chunk_im[32];	// L60
    #pragma HLS array_partition variable=chunk_im complete
    l_S_mm_2_mm: for (int mm = 0; mm < 32; mm++) {	// L61
    #pragma HLS unroll
      int v48 = jj << 2;	// L62
      int v49 = mm >> 3;	// L63
      int v50 = v48 | v49;	// L64
      int32_t v51 = v50;	// L65
      int32_t rd_bank;	// L66
      rd_bank = v51;	// L67
      int32_t v53 = mm;	// L68
      int32_t v54 = v53 & 7;	// L69
      int32_t rd_off;	// L70
      rd_off = v54;	// L71
      int32_t v56 = rd_bank;	// L72
      int v57 = v56;	// L73
      int32_t v58 = rd_off;	// L74
      int v59 = v58;	// L75
      float v60 = buf_re[v57][v59];	// L76
      chunk_re[mm] = v60;	// L77
      int32_t v61 = rd_bank;	// L78
      int v62 = v61;	// L79
      int32_t v63 = rd_off;	// L80
      int v64 = v63;	// L81
      float v65 = buf_im[v62][v64];	// L82
      chunk_im[mm] = v65;	// L83
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re[_iv0];
      }
      v2.write(_vec);
    }	// L85
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im[_iv0];
      }
      v3.write(_vec);
    }	// L86
  }
}

void intra_0_0(
  hls::stream< hls::vector< float, 32 > >& v66,
  hls::stream< hls::vector< float, 32 > >& v67,
  hls::stream< hls::vector< float, 32 > >& v68,
  hls::stream< hls::vector< float, 32 > >& v69
) {	// L90
  l_S__i_0__i: for (int _i = 0; _i < 8; _i++) {	// L91
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v71 = v66.read();
    hls::vector< float, 32 > v72 = v67.read();
    float o_re[32];	// L94
    #pragma HLS array_partition variable=o_re complete
    float o_im[32];	// L95
    #pragma HLS array_partition variable=o_im complete
    l_S_k_0_k: for (int k = 0; k < 16; k++) {	// L96
    #pragma HLS unroll
      float v76 = v71[(k * 2)];	// L97
      float a_re;	// L98
      a_re = v76;	// L99
      float v78 = v72[(k * 2)];	// L100
      float a_im;	// L101
      a_im = v78;	// L102
      float v80 = v71[((k * 2) + 1)];	// L103
      float b_re;	// L104
      b_re = v80;	// L105
      float v82 = v72[((k * 2) + 1)];	// L106
      float b_im;	// L107
      b_im = v82;	// L108
      float v84 = a_re;	// L109
      float v85 = b_re;	// L110
      float v86 = v84 + v85;	// L111
      #pragma HLS bind_op variable=v86 op=fadd impl=fabric
      o_re[(k * 2)] = v86;	// L112
      float v87 = a_im;	// L113
      float v88 = b_im;	// L114
      float v89 = v87 + v88;	// L115
      #pragma HLS bind_op variable=v89 op=fadd impl=fabric
      o_im[(k * 2)] = v89;	// L116
      float v90 = a_re;	// L117
      float v91 = b_re;	// L118
      float v92 = v90 - v91;	// L119
      #pragma HLS bind_op variable=v92 op=fsub impl=fabric
      o_re[((k * 2) + 1)] = v92;	// L120
      float v93 = a_im;	// L121
      float v94 = b_im;	// L122
      float v95 = v93 - v94;	// L123
      #pragma HLS bind_op variable=v95 op=fsub impl=fabric
      o_im[((k * 2) + 1)] = v95;	// L124
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re[_iv0];
      }
      v68.write(_vec);
    }	// L126
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im[_iv0];
      }
      v69.write(_vec);
    }	// L127
  }
}

void intra_1_0(
  hls::stream< hls::vector< float, 32 > >& v96,
  hls::stream< hls::vector< float, 32 > >& v97,
  hls::stream< hls::vector< float, 32 > >& v98,
  hls::stream< hls::vector< float, 32 > >& v99
) {	// L131
  l_S__i_0__i1: for (int _i1 = 0; _i1 < 8; _i1++) {	// L132
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v101 = v96.read();
    hls::vector< float, 32 > v102 = v97.read();
    float o_re1[32];	// L135
    #pragma HLS array_partition variable=o_re1 complete
    float o_im1[32];	// L136
    #pragma HLS array_partition variable=o_im1 complete
    l_S_k_0_k1: for (int k1 = 0; k1 < 8; k1++) {	// L137
    #pragma HLS unroll
      float v106 = v101[(k1 * 4)];	// L138
      float a_re1;	// L139
      a_re1 = v106;	// L140
      float v108 = v102[(k1 * 4)];	// L141
      float a_im1;	// L142
      a_im1 = v108;	// L143
      float v110 = v101[((k1 * 4) + 2)];	// L144
      float b_re1;	// L145
      b_re1 = v110;	// L146
      float v112 = v102[((k1 * 4) + 2)];	// L147
      float b_im1;	// L148
      b_im1 = v112;	// L149
      float v114 = a_re1;	// L150
      float v115 = b_re1;	// L151
      float v116 = v114 + v115;	// L152
      #pragma HLS bind_op variable=v116 op=fadd impl=fabric
      o_re1[(k1 * 4)] = v116;	// L153
      float v117 = a_im1;	// L154
      float v118 = b_im1;	// L155
      float v119 = v117 + v118;	// L156
      #pragma HLS bind_op variable=v119 op=fadd impl=fabric
      o_im1[(k1 * 4)] = v119;	// L157
      float v120 = a_re1;	// L158
      float v121 = b_re1;	// L159
      float v122 = v120 - v121;	// L160
      #pragma HLS bind_op variable=v122 op=fsub impl=fabric
      o_re1[((k1 * 4) + 2)] = v122;	// L161
      float v123 = a_im1;	// L162
      float v124 = b_im1;	// L163
      float v125 = v123 - v124;	// L164
      #pragma HLS bind_op variable=v125 op=fsub impl=fabric
      o_im1[((k1 * 4) + 2)] = v125;	// L165
      float v126 = v101[((k1 * 4) + 1)];	// L166
      float v127 = v102[((k1 * 4) + 3)];	// L167
      float v128 = v126 + v127;	// L168
      #pragma HLS bind_op variable=v128 op=fadd impl=fabric
      o_re1[((k1 * 4) + 1)] = v128;	// L169
      float v129 = v102[((k1 * 4) + 1)];	// L170
      float v130 = v101[((k1 * 4) + 3)];	// L171
      float v131 = v129 - v130;	// L172
      #pragma HLS bind_op variable=v131 op=fsub impl=fabric
      o_im1[((k1 * 4) + 1)] = v131;	// L173
      float v132 = v101[((k1 * 4) + 1)];	// L174
      float v133 = v102[((k1 * 4) + 3)];	// L175
      float v134 = v132 - v133;	// L176
      #pragma HLS bind_op variable=v134 op=fsub impl=fabric
      o_re1[((k1 * 4) + 3)] = v134;	// L177
      float v135 = v102[((k1 * 4) + 1)];	// L178
      float v136 = v101[((k1 * 4) + 3)];	// L179
      float v137 = v135 + v136;	// L180
      #pragma HLS bind_op variable=v137 op=fadd impl=fabric
      o_im1[((k1 * 4) + 3)] = v137;	// L181
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re1[_iv0];
      }
      v98.write(_vec);
    }	// L183
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im1[_iv0];
      }
      v99.write(_vec);
    }	// L184
  }
}

const float twr[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L188
const float twi[128] = {0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L189
void intra_2_0(
  hls::stream< hls::vector< float, 32 > >& v138,
  hls::stream< hls::vector< float, 32 > >& v139,
  hls::stream< hls::vector< float, 32 > >& v140,
  hls::stream< hls::vector< float, 32 > >& v141
) {	// L190
  // placeholder for const float twr	// L198
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L199
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i2: for (int _i2 = 0; _i2 < 8; _i2++) {	// L200
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v145 = v138.read();
    hls::vector< float, 32 > v146 = v139.read();
    float o_re2[32];	// L203
    #pragma HLS array_partition variable=o_re2 complete
    float o_im2[32];	// L204
    #pragma HLS array_partition variable=o_im2 complete
    l_S_k_0_k2: for (int k2 = 0; k2 < 16; k2++) {	// L205
    #pragma HLS unroll
      int v150 = k2 >> 2;	// L206
      int v151 = v150 << 3;	// L207
      int32_t v152 = k2;	// L208
      int32_t v153 = v152 & 3;	// L209
      int32_t v154 = v151;	// L210
      int32_t v155 = v154 | v153;	// L211
      int32_t il;	// L212
      il = v155;	// L213
      int32_t v157 = il;	// L214
      int32_t v158 = v157 | 4;	// L215
      int32_t iu;	// L216
      iu = v158;	// L217
      int32_t v160 = v153 << 5;	// L218
      int32_t tw_k;	// L219
      tw_k = v160;	// L220
      int32_t v162 = il;	// L221
      int v163 = v162;	// L222
      float v164 = v145[v163];	// L223
      float a_re2;	// L224
      a_re2 = v164;	// L225
      int32_t v166 = il;	// L226
      int v167 = v166;	// L227
      float v168 = v146[v167];	// L228
      float a_im2;	// L229
      a_im2 = v168;	// L230
      int32_t v170 = iu;	// L231
      int v171 = v170;	// L232
      float v172 = v145[v171];	// L233
      float b_re2;	// L234
      b_re2 = v172;	// L235
      int32_t v174 = iu;	// L236
      int v175 = v174;	// L237
      float v176 = v146[v175];	// L238
      float b_im2;	// L239
      b_im2 = v176;	// L240
      int32_t v178 = tw_k;	// L241
      bool v179 = v178 == 0;	// L242
      if (v179) {	// L243
        float v180 = a_re2;	// L244
        float v181 = b_re2;	// L245
        float v182 = v180 + v181;	// L246
        #pragma HLS bind_op variable=v182 op=fadd impl=fabric
        int32_t v183 = il;	// L247
        int v184 = v183;	// L248
        o_re2[v184] = v182;	// L249
        float v185 = a_im2;	// L250
        float v186 = b_im2;	// L251
        float v187 = v185 + v186;	// L252
        #pragma HLS bind_op variable=v187 op=fadd impl=fabric
        int32_t v188 = il;	// L253
        int v189 = v188;	// L254
        o_im2[v189] = v187;	// L255
        float v190 = a_re2;	// L256
        float v191 = b_re2;	// L257
        float v192 = v190 - v191;	// L258
        #pragma HLS bind_op variable=v192 op=fsub impl=fabric
        int32_t v193 = iu;	// L259
        int v194 = v193;	// L260
        o_re2[v194] = v192;	// L261
        float v195 = a_im2;	// L262
        float v196 = b_im2;	// L263
        float v197 = v195 - v196;	// L264
        #pragma HLS bind_op variable=v197 op=fsub impl=fabric
        int32_t v198 = iu;	// L265
        int v199 = v198;	// L266
        o_im2[v199] = v197;	// L267
      } else {
        int32_t v200 = tw_k;	// L269
        bool v201 = v200 == 64;	// L270
        if (v201) {	// L271
          float v202 = a_re2;	// L272
          float v203 = b_im2;	// L273
          float v204 = v202 + v203;	// L274
          #pragma HLS bind_op variable=v204 op=fadd impl=fabric
          int32_t v205 = il;	// L275
          int v206 = v205;	// L276
          o_re2[v206] = v204;	// L277
          float v207 = a_im2;	// L278
          float v208 = b_re2;	// L279
          float v209 = v207 - v208;	// L280
          #pragma HLS bind_op variable=v209 op=fsub impl=fabric
          int32_t v210 = il;	// L281
          int v211 = v210;	// L282
          o_im2[v211] = v209;	// L283
          float v212 = a_re2;	// L284
          float v213 = b_im2;	// L285
          float v214 = v212 - v213;	// L286
          #pragma HLS bind_op variable=v214 op=fsub impl=fabric
          int32_t v215 = iu;	// L287
          int v216 = v215;	// L288
          o_re2[v216] = v214;	// L289
          float v217 = a_im2;	// L290
          float v218 = b_re2;	// L291
          float v219 = v217 + v218;	// L292
          #pragma HLS bind_op variable=v219 op=fadd impl=fabric
          int32_t v220 = iu;	// L293
          int v221 = v220;	// L294
          o_im2[v221] = v219;	// L295
        } else {
          int32_t v222 = tw_k;	// L297
          int v223 = v222;	// L298
          float v224 = twr[v223];	// L299
          float tr;	// L300
          tr = v224;	// L301
          int32_t v226 = tw_k;	// L302
          int v227 = v226;	// L303
          float v228 = twi[v227];	// L304
          float ti;	// L305
          ti = v228;	// L306
          float v230 = b_re2;	// L307
          float v231 = tr;	// L308
          float v232 = v230 * v231;	// L309
          float v233 = b_im2;	// L310
          float v234 = ti;	// L311
          float v235 = v233 * v234;	// L312
          float v236 = v232 - v235;	// L313
          float bw_re;	// L314
          bw_re = v236;	// L315
          float v238 = b_re2;	// L316
          float v239 = ti;	// L317
          float v240 = v238 * v239;	// L318
          float v241 = b_im2;	// L319
          float v242 = tr;	// L320
          float v243 = v241 * v242;	// L321
          float v244 = v240 + v243;	// L322
          float bw_im;	// L323
          bw_im = v244;	// L324
          float v246 = a_re2;	// L325
          float v247 = bw_re;	// L326
          float v248 = v246 + v247;	// L327
          #pragma HLS bind_op variable=v248 op=fadd impl=fabric
          int32_t v249 = il;	// L328
          int v250 = v249;	// L329
          o_re2[v250] = v248;	// L330
          float v251 = a_im2;	// L331
          float v252 = bw_im;	// L332
          float v253 = v251 + v252;	// L333
          #pragma HLS bind_op variable=v253 op=fadd impl=fabric
          int32_t v254 = il;	// L334
          int v255 = v254;	// L335
          o_im2[v255] = v253;	// L336
          float v256 = a_re2;	// L337
          float v257 = bw_re;	// L338
          float v258 = v256 - v257;	// L339
          #pragma HLS bind_op variable=v258 op=fsub impl=fabric
          int32_t v259 = iu;	// L340
          int v260 = v259;	// L341
          o_re2[v260] = v258;	// L342
          float v261 = a_im2;	// L343
          float v262 = bw_im;	// L344
          float v263 = v261 - v262;	// L345
          #pragma HLS bind_op variable=v263 op=fsub impl=fabric
          int32_t v264 = iu;	// L346
          int v265 = v264;	// L347
          o_im2[v265] = v263;	// L348
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re2[_iv0];
      }
      v140.write(_vec);
    }	// L352
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im2[_iv0];
      }
      v141.write(_vec);
    }	// L353
  }
}

void intra_3_0(
  hls::stream< hls::vector< float, 32 > >& v266,
  hls::stream< hls::vector< float, 32 > >& v267,
  hls::stream< hls::vector< float, 32 > >& v268,
  hls::stream< hls::vector< float, 32 > >& v269
) {	// L357
  // placeholder for const float twr	// L365
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L366
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i3: for (int _i3 = 0; _i3 < 8; _i3++) {	// L367
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v273 = v266.read();
    hls::vector< float, 32 > v274 = v267.read();
    float o_re3[32];	// L370
    #pragma HLS array_partition variable=o_re3 complete
    float o_im3[32];	// L371
    #pragma HLS array_partition variable=o_im3 complete
    l_S_k_0_k3: for (int k3 = 0; k3 < 16; k3++) {	// L372
    #pragma HLS unroll
      int v278 = k3 >> 3;	// L373
      int v279 = v278 << 4;	// L374
      int32_t v280 = k3;	// L375
      int32_t v281 = v280 & 7;	// L376
      int32_t v282 = v279;	// L377
      int32_t v283 = v282 | v281;	// L378
      int32_t il1;	// L379
      il1 = v283;	// L380
      int32_t v285 = il1;	// L381
      int32_t v286 = v285 | 8;	// L382
      int32_t iu1;	// L383
      iu1 = v286;	// L384
      int32_t v288 = v281 << 4;	// L385
      int32_t tw_k1;	// L386
      tw_k1 = v288;	// L387
      int32_t v290 = il1;	// L388
      int v291 = v290;	// L389
      float v292 = v273[v291];	// L390
      float a_re3;	// L391
      a_re3 = v292;	// L392
      int32_t v294 = il1;	// L393
      int v295 = v294;	// L394
      float v296 = v274[v295];	// L395
      float a_im3;	// L396
      a_im3 = v296;	// L397
      int32_t v298 = iu1;	// L398
      int v299 = v298;	// L399
      float v300 = v273[v299];	// L400
      float b_re3;	// L401
      b_re3 = v300;	// L402
      int32_t v302 = iu1;	// L403
      int v303 = v302;	// L404
      float v304 = v274[v303];	// L405
      float b_im3;	// L406
      b_im3 = v304;	// L407
      int32_t v306 = tw_k1;	// L408
      bool v307 = v306 == 0;	// L409
      if (v307) {	// L410
        float v308 = a_re3;	// L411
        float v309 = b_re3;	// L412
        float v310 = v308 + v309;	// L413
        #pragma HLS bind_op variable=v310 op=fadd impl=fabric
        int32_t v311 = il1;	// L414
        int v312 = v311;	// L415
        o_re3[v312] = v310;	// L416
        float v313 = a_im3;	// L417
        float v314 = b_im3;	// L418
        float v315 = v313 + v314;	// L419
        #pragma HLS bind_op variable=v315 op=fadd impl=fabric
        int32_t v316 = il1;	// L420
        int v317 = v316;	// L421
        o_im3[v317] = v315;	// L422
        float v318 = a_re3;	// L423
        float v319 = b_re3;	// L424
        float v320 = v318 - v319;	// L425
        #pragma HLS bind_op variable=v320 op=fsub impl=fabric
        int32_t v321 = iu1;	// L426
        int v322 = v321;	// L427
        o_re3[v322] = v320;	// L428
        float v323 = a_im3;	// L429
        float v324 = b_im3;	// L430
        float v325 = v323 - v324;	// L431
        #pragma HLS bind_op variable=v325 op=fsub impl=fabric
        int32_t v326 = iu1;	// L432
        int v327 = v326;	// L433
        o_im3[v327] = v325;	// L434
      } else {
        int32_t v328 = tw_k1;	// L436
        bool v329 = v328 == 64;	// L437
        if (v329) {	// L438
          float v330 = a_re3;	// L439
          float v331 = b_im3;	// L440
          float v332 = v330 + v331;	// L441
          #pragma HLS bind_op variable=v332 op=fadd impl=fabric
          int32_t v333 = il1;	// L442
          int v334 = v333;	// L443
          o_re3[v334] = v332;	// L444
          float v335 = a_im3;	// L445
          float v336 = b_re3;	// L446
          float v337 = v335 - v336;	// L447
          #pragma HLS bind_op variable=v337 op=fsub impl=fabric
          int32_t v338 = il1;	// L448
          int v339 = v338;	// L449
          o_im3[v339] = v337;	// L450
          float v340 = a_re3;	// L451
          float v341 = b_im3;	// L452
          float v342 = v340 - v341;	// L453
          #pragma HLS bind_op variable=v342 op=fsub impl=fabric
          int32_t v343 = iu1;	// L454
          int v344 = v343;	// L455
          o_re3[v344] = v342;	// L456
          float v345 = a_im3;	// L457
          float v346 = b_re3;	// L458
          float v347 = v345 + v346;	// L459
          #pragma HLS bind_op variable=v347 op=fadd impl=fabric
          int32_t v348 = iu1;	// L460
          int v349 = v348;	// L461
          o_im3[v349] = v347;	// L462
        } else {
          int32_t v350 = tw_k1;	// L464
          int v351 = v350;	// L465
          float v352 = twr[v351];	// L466
          float tr1;	// L467
          tr1 = v352;	// L468
          int32_t v354 = tw_k1;	// L469
          int v355 = v354;	// L470
          float v356 = twi[v355];	// L471
          float ti1;	// L472
          ti1 = v356;	// L473
          float v358 = b_re3;	// L474
          float v359 = tr1;	// L475
          float v360 = v358 * v359;	// L476
          float v361 = b_im3;	// L477
          float v362 = ti1;	// L478
          float v363 = v361 * v362;	// L479
          float v364 = v360 - v363;	// L480
          float bw_re1;	// L481
          bw_re1 = v364;	// L482
          float v366 = b_re3;	// L483
          float v367 = ti1;	// L484
          float v368 = v366 * v367;	// L485
          float v369 = b_im3;	// L486
          float v370 = tr1;	// L487
          float v371 = v369 * v370;	// L488
          float v372 = v368 + v371;	// L489
          float bw_im1;	// L490
          bw_im1 = v372;	// L491
          float v374 = a_re3;	// L492
          float v375 = bw_re1;	// L493
          float v376 = v374 + v375;	// L494
          #pragma HLS bind_op variable=v376 op=fadd impl=fabric
          int32_t v377 = il1;	// L495
          int v378 = v377;	// L496
          o_re3[v378] = v376;	// L497
          float v379 = a_im3;	// L498
          float v380 = bw_im1;	// L499
          float v381 = v379 + v380;	// L500
          #pragma HLS bind_op variable=v381 op=fadd impl=fabric
          int32_t v382 = il1;	// L501
          int v383 = v382;	// L502
          o_im3[v383] = v381;	// L503
          float v384 = a_re3;	// L504
          float v385 = bw_re1;	// L505
          float v386 = v384 - v385;	// L506
          #pragma HLS bind_op variable=v386 op=fsub impl=fabric
          int32_t v387 = iu1;	// L507
          int v388 = v387;	// L508
          o_re3[v388] = v386;	// L509
          float v389 = a_im3;	// L510
          float v390 = bw_im1;	// L511
          float v391 = v389 - v390;	// L512
          #pragma HLS bind_op variable=v391 op=fsub impl=fabric
          int32_t v392 = iu1;	// L513
          int v393 = v392;	// L514
          o_im3[v393] = v391;	// L515
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re3[_iv0];
      }
      v268.write(_vec);
    }	// L519
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im3[_iv0];
      }
      v269.write(_vec);
    }	// L520
  }
}

void intra_4_0(
  hls::stream< hls::vector< float, 32 > >& v394,
  hls::stream< hls::vector< float, 32 > >& v395,
  hls::stream< hls::vector< float, 32 > >& v396,
  hls::stream< hls::vector< float, 32 > >& v397
) {	// L524
  // placeholder for const float twr	// L529
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L530
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i4: for (int _i4 = 0; _i4 < 8; _i4++) {	// L531
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v401 = v394.read();
    hls::vector< float, 32 > v402 = v395.read();
    float o_re4[32];	// L534
    #pragma HLS array_partition variable=o_re4 complete
    float o_im4[32];	// L535
    #pragma HLS array_partition variable=o_im4 complete
    l_S_k_0_k4: for (int k4 = 0; k4 < 16; k4++) {	// L536
    #pragma HLS unroll
      int32_t v406 = k4;	// L537
      int32_t il2;	// L538
      il2 = v406;	// L539
      int32_t v408 = v406 | 16;	// L540
      int32_t iu2;	// L541
      iu2 = v408;	// L542
      int v410 = k4 << 3;	// L543
      int32_t v411 = v410;	// L544
      int32_t tw_k2;	// L545
      tw_k2 = v411;	// L546
      int32_t v413 = il2;	// L547
      int v414 = v413;	// L548
      float v415 = v401[v414];	// L549
      float a_re4;	// L550
      a_re4 = v415;	// L551
      int32_t v417 = il2;	// L552
      int v418 = v417;	// L553
      float v419 = v402[v418];	// L554
      float a_im4;	// L555
      a_im4 = v419;	// L556
      int32_t v421 = iu2;	// L557
      int v422 = v421;	// L558
      float v423 = v401[v422];	// L559
      float b_re4;	// L560
      b_re4 = v423;	// L561
      int32_t v425 = iu2;	// L562
      int v426 = v425;	// L563
      float v427 = v402[v426];	// L564
      float b_im4;	// L565
      b_im4 = v427;	// L566
      int32_t v429 = tw_k2;	// L567
      bool v430 = v429 == 0;	// L568
      if (v430) {	// L569
        float v431 = a_re4;	// L570
        float v432 = b_re4;	// L571
        float v433 = v431 + v432;	// L572
        #pragma HLS bind_op variable=v433 op=fadd impl=fabric
        int32_t v434 = il2;	// L573
        int v435 = v434;	// L574
        o_re4[v435] = v433;	// L575
        float v436 = a_im4;	// L576
        float v437 = b_im4;	// L577
        float v438 = v436 + v437;	// L578
        #pragma HLS bind_op variable=v438 op=fadd impl=fabric
        int32_t v439 = il2;	// L579
        int v440 = v439;	// L580
        o_im4[v440] = v438;	// L581
        float v441 = a_re4;	// L582
        float v442 = b_re4;	// L583
        float v443 = v441 - v442;	// L584
        #pragma HLS bind_op variable=v443 op=fsub impl=fabric
        int32_t v444 = iu2;	// L585
        int v445 = v444;	// L586
        o_re4[v445] = v443;	// L587
        float v446 = a_im4;	// L588
        float v447 = b_im4;	// L589
        float v448 = v446 - v447;	// L590
        #pragma HLS bind_op variable=v448 op=fsub impl=fabric
        int32_t v449 = iu2;	// L591
        int v450 = v449;	// L592
        o_im4[v450] = v448;	// L593
      } else {
        int32_t v451 = tw_k2;	// L595
        bool v452 = v451 == 64;	// L596
        if (v452) {	// L597
          float v453 = a_re4;	// L598
          float v454 = b_im4;	// L599
          float v455 = v453 + v454;	// L600
          #pragma HLS bind_op variable=v455 op=fadd impl=fabric
          int32_t v456 = il2;	// L601
          int v457 = v456;	// L602
          o_re4[v457] = v455;	// L603
          float v458 = a_im4;	// L604
          float v459 = b_re4;	// L605
          float v460 = v458 - v459;	// L606
          #pragma HLS bind_op variable=v460 op=fsub impl=fabric
          int32_t v461 = il2;	// L607
          int v462 = v461;	// L608
          o_im4[v462] = v460;	// L609
          float v463 = a_re4;	// L610
          float v464 = b_im4;	// L611
          float v465 = v463 - v464;	// L612
          #pragma HLS bind_op variable=v465 op=fsub impl=fabric
          int32_t v466 = iu2;	// L613
          int v467 = v466;	// L614
          o_re4[v467] = v465;	// L615
          float v468 = a_im4;	// L616
          float v469 = b_re4;	// L617
          float v470 = v468 + v469;	// L618
          #pragma HLS bind_op variable=v470 op=fadd impl=fabric
          int32_t v471 = iu2;	// L619
          int v472 = v471;	// L620
          o_im4[v472] = v470;	// L621
        } else {
          int32_t v473 = tw_k2;	// L623
          int v474 = v473;	// L624
          float v475 = twr[v474];	// L625
          float tr2;	// L626
          tr2 = v475;	// L627
          int32_t v477 = tw_k2;	// L628
          int v478 = v477;	// L629
          float v479 = twi[v478];	// L630
          float ti2;	// L631
          ti2 = v479;	// L632
          float v481 = b_re4;	// L633
          float v482 = tr2;	// L634
          float v483 = v481 * v482;	// L635
          float v484 = b_im4;	// L636
          float v485 = ti2;	// L637
          float v486 = v484 * v485;	// L638
          float v487 = v483 - v486;	// L639
          float bw_re2;	// L640
          bw_re2 = v487;	// L641
          float v489 = b_re4;	// L642
          float v490 = ti2;	// L643
          float v491 = v489 * v490;	// L644
          float v492 = b_im4;	// L645
          float v493 = tr2;	// L646
          float v494 = v492 * v493;	// L647
          float v495 = v491 + v494;	// L648
          float bw_im2;	// L649
          bw_im2 = v495;	// L650
          float v497 = a_re4;	// L651
          float v498 = bw_re2;	// L652
          float v499 = v497 + v498;	// L653
          #pragma HLS bind_op variable=v499 op=fadd impl=fabric
          int32_t v500 = il2;	// L654
          int v501 = v500;	// L655
          o_re4[v501] = v499;	// L656
          float v502 = a_im4;	// L657
          float v503 = bw_im2;	// L658
          float v504 = v502 + v503;	// L659
          #pragma HLS bind_op variable=v504 op=fadd impl=fabric
          int32_t v505 = il2;	// L660
          int v506 = v505;	// L661
          o_im4[v506] = v504;	// L662
          float v507 = a_re4;	// L663
          float v508 = bw_re2;	// L664
          float v509 = v507 - v508;	// L665
          #pragma HLS bind_op variable=v509 op=fsub impl=fabric
          int32_t v510 = iu2;	// L666
          int v511 = v510;	// L667
          o_re4[v511] = v509;	// L668
          float v512 = a_im4;	// L669
          float v513 = bw_im2;	// L670
          float v514 = v512 - v513;	// L671
          #pragma HLS bind_op variable=v514 op=fsub impl=fabric
          int32_t v515 = iu2;	// L672
          int v516 = v515;	// L673
          o_im4[v516] = v514;	// L674
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re4[_iv0];
      }
      v396.write(_vec);
    }	// L678
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im4[_iv0];
      }
      v397.write(_vec);
    }	// L679
  }
}

void inter_5_0(
  hls::stream< hls::vector< float, 32 > >& v517,
  hls::stream< hls::vector< float, 32 > >& v518,
  hls::stream< hls::vector< float, 32 > >& v519,
  hls::stream< hls::vector< float, 32 > >& v520
) {	// L683
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L693
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L694
  #pragma HLS array_partition variable=twi complete
  float in_re[32][8];	// L695
  #pragma HLS array_partition variable=in_re complete dim=1

  #pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L696
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L697
  #pragma HLS array_partition variable=out_re_b complete dim=1

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L698
  #pragma HLS array_partition variable=out_im_b complete dim=1

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L699
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v528 = v517.read();
    hls::vector< float, 32 > v529 = v518.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L702
    #pragma HLS unroll
      int32_t v531 = i;	// L703
      int32_t v532 = v531 & 1;	// L704
      int32_t v533 = v532 << 4;	// L705
      int32_t v534 = k5;	// L706
      int32_t v535 = v534 ^ v533;	// L707
      int32_t bank1;	// L708
      bank1 = v535;	// L709
      float v537 = v528[k5];	// L710
      int32_t v538 = bank1;	// L711
      int v539 = v538;	// L712
      in_re[v539][i] = v537;	// L713
      float v540 = v529[k5];	// L714
      int32_t v541 = bank1;	// L715
      int v542 = v541;	// L716
      in_im[v542][i] = v540;	// L717
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L720
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L721
    #pragma HLS unroll
      int v545 = i1 << 4;	// L722
      int v546 = v545 | k6;	// L723
      int32_t v547 = v546;	// L724
      int32_t bg;	// L725
      bg = v547;	// L726
      int32_t v549 = bg;	// L727
      int32_t v550 = v549 >> 5;	// L728
      int32_t grp;	// L729
      grp = v550;	// L730
      int32_t v552 = bg;	// L731
      int32_t v553 = v552 & 31;	// L732
      int32_t within;	// L733
      within = v553;	// L734
      int32_t v555 = within;	// L735
      int32_t v556 = v555 << 2;	// L736
      int32_t tw_k3;	// L737
      tw_k3 = v556;	// L738
      int32_t v558 = grp;	// L739
      int32_t v559 = v558 << 1;	// L740
      int32_t off_l;	// L741
      off_l = v559;	// L742
      int32_t v561 = off_l;	// L743
      int32_t v562 = v561 | 1;	// L744
      int32_t off_u;	// L745
      off_u = v562;	// L746
      int32_t v564 = within;	// L747
      int v565 = v564;	// L748
      int32_t v566 = off_l;	// L749
      int v567 = v566;	// L750
      float v568 = in_re[v565][v567];	// L751
      float a_re5;	// L752
      a_re5 = v568;	// L753
      int32_t v570 = within;	// L754
      int v571 = v570;	// L755
      int32_t v572 = off_l;	// L756
      int v573 = v572;	// L757
      float v574 = in_im[v571][v573];	// L758
      float a_im5;	// L759
      a_im5 = v574;	// L760
      int32_t v576 = within;	// L761
      int32_t v577 = v576 ^ 16;	// L762
      int v578 = v577;	// L763
      int32_t v579 = off_u;	// L764
      int v580 = v579;	// L765
      float v581 = in_re[v578][v580];	// L766
      float b_re5;	// L767
      b_re5 = v581;	// L768
      int32_t v583 = within;	// L769
      int32_t v584 = v583 ^ 16;	// L770
      int v585 = v584;	// L771
      int32_t v586 = off_u;	// L772
      int v587 = v586;	// L773
      float v588 = in_im[v585][v587];	// L774
      float b_im5;	// L775
      b_im5 = v588;	// L776
      int32_t v590 = tw_k3;	// L777
      bool v591 = v590 == 0;	// L778
      if (v591) {	// L779
        float v592 = a_re5;	// L780
        float v593 = b_re5;	// L781
        float v594 = v592 + v593;	// L782
        #pragma HLS bind_op variable=v594 op=fadd impl=fabric
        int32_t v595 = within;	// L783
        int v596 = v595;	// L784
        int32_t v597 = off_l;	// L785
        int v598 = v597;	// L786
        out_re_b[v596][v598] = v594;	// L787
        float v599 = a_im5;	// L788
        float v600 = b_im5;	// L789
        float v601 = v599 + v600;	// L790
        #pragma HLS bind_op variable=v601 op=fadd impl=fabric
        int32_t v602 = within;	// L791
        int v603 = v602;	// L792
        int32_t v604 = off_l;	// L793
        int v605 = v604;	// L794
        out_im_b[v603][v605] = v601;	// L795
        float v606 = a_re5;	// L796
        float v607 = b_re5;	// L797
        float v608 = v606 - v607;	// L798
        #pragma HLS bind_op variable=v608 op=fsub impl=fabric
        int32_t v609 = within;	// L799
        int32_t v610 = v609 ^ 16;	// L800
        int v611 = v610;	// L801
        int32_t v612 = off_u;	// L802
        int v613 = v612;	// L803
        out_re_b[v611][v613] = v608;	// L804
        float v614 = a_im5;	// L805
        float v615 = b_im5;	// L806
        float v616 = v614 - v615;	// L807
        #pragma HLS bind_op variable=v616 op=fsub impl=fabric
        int32_t v617 = within;	// L808
        int32_t v618 = v617 ^ 16;	// L809
        int v619 = v618;	// L810
        int32_t v620 = off_u;	// L811
        int v621 = v620;	// L812
        out_im_b[v619][v621] = v616;	// L813
      } else {
        int32_t v622 = tw_k3;	// L815
        bool v623 = v622 == 64;	// L816
        if (v623) {	// L817
          float v624 = a_re5;	// L818
          float v625 = b_im5;	// L819
          float v626 = v624 + v625;	// L820
          #pragma HLS bind_op variable=v626 op=fadd impl=fabric
          int32_t v627 = within;	// L821
          int v628 = v627;	// L822
          int32_t v629 = off_l;	// L823
          int v630 = v629;	// L824
          out_re_b[v628][v630] = v626;	// L825
          float v631 = a_im5;	// L826
          float v632 = b_re5;	// L827
          float v633 = v631 - v632;	// L828
          #pragma HLS bind_op variable=v633 op=fsub impl=fabric
          int32_t v634 = within;	// L829
          int v635 = v634;	// L830
          int32_t v636 = off_l;	// L831
          int v637 = v636;	// L832
          out_im_b[v635][v637] = v633;	// L833
          float v638 = a_re5;	// L834
          float v639 = b_im5;	// L835
          float v640 = v638 - v639;	// L836
          #pragma HLS bind_op variable=v640 op=fsub impl=fabric
          int32_t v641 = within;	// L837
          int32_t v642 = v641 ^ 16;	// L838
          int v643 = v642;	// L839
          int32_t v644 = off_u;	// L840
          int v645 = v644;	// L841
          out_re_b[v643][v645] = v640;	// L842
          float v646 = a_im5;	// L843
          float v647 = b_re5;	// L844
          float v648 = v646 + v647;	// L845
          #pragma HLS bind_op variable=v648 op=fadd impl=fabric
          int32_t v649 = within;	// L846
          int32_t v650 = v649 ^ 16;	// L847
          int v651 = v650;	// L848
          int32_t v652 = off_u;	// L849
          int v653 = v652;	// L850
          out_im_b[v651][v653] = v648;	// L851
        } else {
          int32_t v654 = tw_k3;	// L853
          int v655 = v654;	// L854
          float v656 = twr[v655];	// L855
          float tr3;	// L856
          tr3 = v656;	// L857
          int32_t v658 = tw_k3;	// L858
          int v659 = v658;	// L859
          float v660 = twi[v659];	// L860
          float ti3;	// L861
          ti3 = v660;	// L862
          float v662 = b_re5;	// L863
          float v663 = tr3;	// L864
          float v664 = v662 * v663;	// L865
          float v665 = b_im5;	// L866
          float v666 = ti3;	// L867
          float v667 = v665 * v666;	// L868
          float v668 = v664 - v667;	// L869
          float bw_re3;	// L870
          bw_re3 = v668;	// L871
          float v670 = b_re5;	// L872
          float v671 = ti3;	// L873
          float v672 = v670 * v671;	// L874
          float v673 = b_im5;	// L875
          float v674 = tr3;	// L876
          float v675 = v673 * v674;	// L877
          float v676 = v672 + v675;	// L878
          float bw_im3;	// L879
          bw_im3 = v676;	// L880
          float v678 = a_re5;	// L881
          float v679 = bw_re3;	// L882
          float v680 = v678 + v679;	// L883
          #pragma HLS bind_op variable=v680 op=fadd impl=fabric
          int32_t v681 = within;	// L884
          int v682 = v681;	// L885
          int32_t v683 = off_l;	// L886
          int v684 = v683;	// L887
          out_re_b[v682][v684] = v680;	// L888
          float v685 = a_im5;	// L889
          float v686 = bw_im3;	// L890
          float v687 = v685 + v686;	// L891
          #pragma HLS bind_op variable=v687 op=fadd impl=fabric
          int32_t v688 = within;	// L892
          int v689 = v688;	// L893
          int32_t v690 = off_l;	// L894
          int v691 = v690;	// L895
          out_im_b[v689][v691] = v687;	// L896
          float v692 = a_re5;	// L897
          float v693 = bw_re3;	// L898
          float v694 = v692 - v693;	// L899
          #pragma HLS bind_op variable=v694 op=fsub impl=fabric
          int32_t v695 = within;	// L900
          int32_t v696 = v695 ^ 16;	// L901
          int v697 = v696;	// L902
          int32_t v698 = off_u;	// L903
          int v699 = v698;	// L904
          out_re_b[v697][v699] = v694;	// L905
          float v700 = a_im5;	// L906
          float v701 = bw_im3;	// L907
          float v702 = v700 - v701;	// L908
          #pragma HLS bind_op variable=v702 op=fsub impl=fabric
          int32_t v703 = within;	// L909
          int32_t v704 = v703 ^ 16;	// L910
          int v705 = v704;	// L911
          int32_t v706 = off_u;	// L912
          int v707 = v706;	// L913
          out_im_b[v705][v707] = v702;	// L914
        }
      }
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L919
  #pragma HLS pipeline II=1
    float chunk_re1[32];	// L920
    #pragma HLS array_partition variable=chunk_re1 complete
    float chunk_im1[32];	// L921
    #pragma HLS array_partition variable=chunk_im1 complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L922
    #pragma HLS unroll
      int32_t v712 = i2;	// L923
      int32_t v713 = v712 & 1;	// L924
      int32_t v714 = v713 << 4;	// L925
      int32_t v715 = k7;	// L926
      int32_t v716 = v715 ^ v714;	// L927
      int32_t bank2;	// L928
      bank2 = v716;	// L929
      int32_t v718 = bank2;	// L930
      int v719 = v718;	// L931
      float v720 = out_re_b[v719][i2];	// L932
      chunk_re1[k7] = v720;	// L933
      int32_t v721 = bank2;	// L934
      int v722 = v721;	// L935
      float v723 = out_im_b[v722][i2];	// L936
      chunk_im1[k7] = v723;	// L937
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re1[_iv0];
      }
      v519.write(_vec);
    }	// L939
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im1[_iv0];
      }
      v520.write(_vec);
    }	// L940
  }
}

void inter_6_0(
  hls::stream< hls::vector< float, 32 > >& v724,
  hls::stream< hls::vector< float, 32 > >& v725,
  hls::stream< hls::vector< float, 32 > >& v726,
  hls::stream< hls::vector< float, 32 > >& v727
) {	// L944
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L953
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L954
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L955
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L956
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L957
  #pragma HLS array_partition variable=out_re_b1 complete dim=1

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L958
  #pragma HLS array_partition variable=out_im_b1 complete dim=1

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L959
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v735 = v724.read();
    hls::vector< float, 32 > v736 = v725.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L962
    #pragma HLS unroll
      int v738 = i3 >> 1;	// L963
      int32_t v739 = v738;	// L964
      int32_t v740 = v739 & 1;	// L965
      int32_t v741 = v740 << 4;	// L966
      int32_t v742 = k8;	// L967
      int32_t v743 = v742 ^ v741;	// L968
      int32_t bank3;	// L969
      bank3 = v743;	// L970
      float v745 = v735[k8];	// L971
      int32_t v746 = bank3;	// L972
      int v747 = v746;	// L973
      in_re1[v747][i3] = v745;	// L974
      float v748 = v736[k8];	// L975
      int32_t v749 = bank3;	// L976
      int v750 = v749;	// L977
      in_im1[v750][i3] = v748;	// L978
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L981
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L982
    #pragma HLS unroll
      int v753 = i4 << 4;	// L983
      int v754 = v753 | k9;	// L984
      uint32_t v755 = v754;	// L985
      uint32_t bg1;	// L986
      bg1 = v755;	// L987
      int32_t v757 = bg1;	// L988
      int32_t v758 = v757 & 63;	// L989
      int32_t v759 = v758 << 1;	// L990
      uint32_t tw_k4;	// L991
      tw_k4 = v759;	// L992
      int32_t v761 = i4;	// L993
      int32_t v762 = v761 & 1;	// L994
      int32_t v763 = v762 << 4;	// L995
      int32_t v764 = k9;	// L996
      int32_t v765 = v764 | v763;	// L997
      uint32_t bank_il;	// L998
      bank_il = v765;	// L999
      int32_t v767 = bank_il;	// L1000
      int32_t v768 = v767 ^ 16;	// L1001
      uint32_t bank_iu;	// L1002
      bank_iu = v768;	// L1003
      int v770 = i4 >> 2;	// L1004
      int v771 = v770 << 2;	// L1005
      int v772 = i4 >> 1;	// L1006
      int32_t v773 = v772;	// L1007
      int32_t v774 = v773 & 1;	// L1008
      int32_t v775 = v771;	// L1009
      int32_t v776 = v775 | v774;	// L1010
      uint32_t off_il;	// L1011
      off_il = v776;	// L1012
      int32_t v778 = off_il;	// L1013
      int32_t v779 = v778 | 2;	// L1014
      uint32_t off_iu;	// L1015
      off_iu = v779;	// L1016
      int32_t v781 = bank_il;	// L1017
      int v782 = v781;	// L1018
      int32_t v783 = off_il;	// L1019
      int v784 = v783;	// L1020
      float v785 = in_re1[v782][v784];	// L1021
      float a_re6;	// L1022
      a_re6 = v785;	// L1023
      int32_t v787 = bank_il;	// L1024
      int v788 = v787;	// L1025
      int32_t v789 = off_il;	// L1026
      int v790 = v789;	// L1027
      float v791 = in_im1[v788][v790];	// L1028
      float a_im6;	// L1029
      a_im6 = v791;	// L1030
      int32_t v793 = bank_iu;	// L1031
      int v794 = v793;	// L1032
      int32_t v795 = off_iu;	// L1033
      int v796 = v795;	// L1034
      float v797 = in_re1[v794][v796];	// L1035
      float b_re6;	// L1036
      b_re6 = v797;	// L1037
      int32_t v799 = bank_iu;	// L1038
      int v800 = v799;	// L1039
      int32_t v801 = off_iu;	// L1040
      int v802 = v801;	// L1041
      float v803 = in_im1[v800][v802];	// L1042
      float b_im6;	// L1043
      b_im6 = v803;	// L1044
      int32_t v805 = tw_k4;	// L1045
      int v806 = v805;	// L1046
      float v807 = twr[v806];	// L1047
      float tr4;	// L1048
      tr4 = v807;	// L1049
      int32_t v809 = tw_k4;	// L1050
      int v810 = v809;	// L1051
      float v811 = twi[v810];	// L1052
      float ti4;	// L1053
      ti4 = v811;	// L1054
      float v813 = b_re6;	// L1055
      float v814 = tr4;	// L1056
      float v815 = v813 * v814;	// L1057
      float v816 = b_im6;	// L1058
      float v817 = ti4;	// L1059
      float v818 = v816 * v817;	// L1060
      float v819 = v815 - v818;	// L1061
      float bw_re4;	// L1062
      bw_re4 = v819;	// L1063
      float v821 = b_re6;	// L1064
      float v822 = ti4;	// L1065
      float v823 = v821 * v822;	// L1066
      float v824 = b_im6;	// L1067
      float v825 = tr4;	// L1068
      float v826 = v824 * v825;	// L1069
      float v827 = v823 + v826;	// L1070
      float bw_im4;	// L1071
      bw_im4 = v827;	// L1072
      float v829 = a_re6;	// L1073
      float v830 = bw_re4;	// L1074
      float v831 = v829 + v830;	// L1075
      #pragma HLS bind_op variable=v831 op=fadd impl=fabric
      int32_t v832 = bank_il;	// L1076
      int v833 = v832;	// L1077
      int32_t v834 = off_il;	// L1078
      int v835 = v834;	// L1079
      out_re_b1[v833][v835] = v831;	// L1080
      float v836 = a_im6;	// L1081
      float v837 = bw_im4;	// L1082
      float v838 = v836 + v837;	// L1083
      #pragma HLS bind_op variable=v838 op=fadd impl=fabric
      int32_t v839 = bank_il;	// L1084
      int v840 = v839;	// L1085
      int32_t v841 = off_il;	// L1086
      int v842 = v841;	// L1087
      out_im_b1[v840][v842] = v838;	// L1088
      float v843 = a_re6;	// L1089
      float v844 = bw_re4;	// L1090
      float v845 = v843 - v844;	// L1091
      #pragma HLS bind_op variable=v845 op=fsub impl=fabric
      int32_t v846 = bank_iu;	// L1092
      int v847 = v846;	// L1093
      int32_t v848 = off_iu;	// L1094
      int v849 = v848;	// L1095
      out_re_b1[v847][v849] = v845;	// L1096
      float v850 = a_im6;	// L1097
      float v851 = bw_im4;	// L1098
      float v852 = v850 - v851;	// L1099
      #pragma HLS bind_op variable=v852 op=fsub impl=fabric
      int32_t v853 = bank_iu;	// L1100
      int v854 = v853;	// L1101
      int32_t v855 = off_iu;	// L1102
      int v856 = v855;	// L1103
      out_im_b1[v854][v856] = v852;	// L1104
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L1107
  #pragma HLS pipeline II=1
    float chunk_re2[32];	// L1108
    #pragma HLS array_partition variable=chunk_re2 complete
    float chunk_im2[32];	// L1109
    #pragma HLS array_partition variable=chunk_im2 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L1110
    #pragma HLS unroll
      int v861 = i5 >> 1;	// L1111
      int32_t v862 = v861;	// L1112
      int32_t v863 = v862 & 1;	// L1113
      int32_t v864 = v863 << 4;	// L1114
      int32_t v865 = k10;	// L1115
      int32_t v866 = v865 ^ v864;	// L1116
      int32_t bank4;	// L1117
      bank4 = v866;	// L1118
      int32_t v868 = bank4;	// L1119
      int v869 = v868;	// L1120
      float v870 = out_re_b1[v869][i5];	// L1121
      chunk_re2[k10] = v870;	// L1122
      int32_t v871 = bank4;	// L1123
      int v872 = v871;	// L1124
      float v873 = out_im_b1[v872][i5];	// L1125
      chunk_im2[k10] = v873;	// L1126
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re2[_iv0];
      }
      v726.write(_vec);
    }	// L1128
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im2[_iv0];
      }
      v727.write(_vec);
    }	// L1129
  }
}

void inter_7_0(
  hls::stream< hls::vector< float, 32 > >& v874,
  hls::stream< hls::vector< float, 32 > >& v875,
  hls::stream< hls::vector< float, 32 > >& v876,
  hls::stream< hls::vector< float, 32 > >& v877
) {	// L1133
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1140
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1141
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L1142
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L1143
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L1144
  #pragma HLS array_partition variable=out_re_b2 complete dim=1

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L1145
  #pragma HLS array_partition variable=out_im_b2 complete dim=1

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L1146
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v885 = v874.read();
    hls::vector< float, 32 > v886 = v875.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L1149
    #pragma HLS unroll
      int v888 = i6 >> 2;	// L1150
      int32_t v889 = v888;	// L1151
      int32_t v890 = v889 & 1;	// L1152
      int32_t v891 = v890 << 4;	// L1153
      int32_t v892 = k11;	// L1154
      int32_t v893 = v892 ^ v891;	// L1155
      int32_t bank5;	// L1156
      bank5 = v893;	// L1157
      float v895 = v885[k11];	// L1158
      int32_t v896 = bank5;	// L1159
      int v897 = v896;	// L1160
      in_re2[v897][i6] = v895;	// L1161
      float v898 = v886[k11];	// L1162
      int32_t v899 = bank5;	// L1163
      int v900 = v899;	// L1164
      in_im2[v900][i6] = v898;	// L1165
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L1168
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L1169
    #pragma HLS unroll
      int32_t v903 = i7;	// L1170
      int32_t v904 = v903 & 1;	// L1171
      int32_t v905 = v904 << 4;	// L1172
      int32_t v906 = k12;	// L1173
      int32_t v907 = v906 | v905;	// L1174
      uint32_t bank_il1;	// L1175
      bank_il1 = v907;	// L1176
      int32_t v909 = bank_il1;	// L1177
      int32_t v910 = v909 ^ 16;	// L1178
      uint32_t bank_iu1;	// L1179
      bank_iu1 = v910;	// L1180
      int v912 = i7 >> 1;	// L1181
      uint32_t v913 = v912;	// L1182
      uint32_t off_il1;	// L1183
      off_il1 = v913;	// L1184
      int32_t v915 = off_il1;	// L1185
      int32_t v916 = v915 | 4;	// L1186
      uint32_t off_iu1;	// L1187
      off_iu1 = v916;	// L1188
      int v918 = i7 << 4;	// L1189
      int v919 = v918 | k12;	// L1190
      uint32_t v920 = v919;	// L1191
      uint32_t tw_k5;	// L1192
      tw_k5 = v920;	// L1193
      int32_t v922 = bank_il1;	// L1194
      int v923 = v922;	// L1195
      int32_t v924 = off_il1;	// L1196
      int v925 = v924;	// L1197
      float v926 = in_re2[v923][v925];	// L1198
      float a_re7;	// L1199
      a_re7 = v926;	// L1200
      int32_t v928 = bank_il1;	// L1201
      int v929 = v928;	// L1202
      int32_t v930 = off_il1;	// L1203
      int v931 = v930;	// L1204
      float v932 = in_im2[v929][v931];	// L1205
      float a_im7;	// L1206
      a_im7 = v932;	// L1207
      int32_t v934 = bank_iu1;	// L1208
      int v935 = v934;	// L1209
      int32_t v936 = off_iu1;	// L1210
      int v937 = v936;	// L1211
      float v938 = in_re2[v935][v937];	// L1212
      float b_re7;	// L1213
      b_re7 = v938;	// L1214
      int32_t v940 = bank_iu1;	// L1215
      int v941 = v940;	// L1216
      int32_t v942 = off_iu1;	// L1217
      int v943 = v942;	// L1218
      float v944 = in_im2[v941][v943];	// L1219
      float b_im7;	// L1220
      b_im7 = v944;	// L1221
      int32_t v946 = tw_k5;	// L1222
      int v947 = v946;	// L1223
      float v948 = twr[v947];	// L1224
      float tr5;	// L1225
      tr5 = v948;	// L1226
      int32_t v950 = tw_k5;	// L1227
      int v951 = v950;	// L1228
      float v952 = twi[v951];	// L1229
      float ti5;	// L1230
      ti5 = v952;	// L1231
      float v954 = b_re7;	// L1232
      float v955 = tr5;	// L1233
      float v956 = v954 * v955;	// L1234
      float v957 = b_im7;	// L1235
      float v958 = ti5;	// L1236
      float v959 = v957 * v958;	// L1237
      float v960 = v956 - v959;	// L1238
      float bw_re5;	// L1239
      bw_re5 = v960;	// L1240
      float v962 = b_re7;	// L1241
      float v963 = ti5;	// L1242
      float v964 = v962 * v963;	// L1243
      float v965 = b_im7;	// L1244
      float v966 = tr5;	// L1245
      float v967 = v965 * v966;	// L1246
      float v968 = v964 + v967;	// L1247
      float bw_im5;	// L1248
      bw_im5 = v968;	// L1249
      float v970 = a_re7;	// L1250
      float v971 = bw_re5;	// L1251
      float v972 = v970 + v971;	// L1252
      #pragma HLS bind_op variable=v972 op=fadd impl=fabric
      int32_t v973 = bank_il1;	// L1253
      int v974 = v973;	// L1254
      int32_t v975 = off_il1;	// L1255
      int v976 = v975;	// L1256
      out_re_b2[v974][v976] = v972;	// L1257
      float v977 = a_im7;	// L1258
      float v978 = bw_im5;	// L1259
      float v979 = v977 + v978;	// L1260
      #pragma HLS bind_op variable=v979 op=fadd impl=fabric
      int32_t v980 = bank_il1;	// L1261
      int v981 = v980;	// L1262
      int32_t v982 = off_il1;	// L1263
      int v983 = v982;	// L1264
      out_im_b2[v981][v983] = v979;	// L1265
      float v984 = a_re7;	// L1266
      float v985 = bw_re5;	// L1267
      float v986 = v984 - v985;	// L1268
      #pragma HLS bind_op variable=v986 op=fsub impl=fabric
      int32_t v987 = bank_iu1;	// L1269
      int v988 = v987;	// L1270
      int32_t v989 = off_iu1;	// L1271
      int v990 = v989;	// L1272
      out_re_b2[v988][v990] = v986;	// L1273
      float v991 = a_im7;	// L1274
      float v992 = bw_im5;	// L1275
      float v993 = v991 - v992;	// L1276
      #pragma HLS bind_op variable=v993 op=fsub impl=fabric
      int32_t v994 = bank_iu1;	// L1277
      int v995 = v994;	// L1278
      int32_t v996 = off_iu1;	// L1279
      int v997 = v996;	// L1280
      out_im_b2[v995][v997] = v993;	// L1281
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1284
  #pragma HLS pipeline II=1
    float chunk_re3[32];	// L1285
    #pragma HLS array_partition variable=chunk_re3 complete
    float chunk_im3[32];	// L1286
    #pragma HLS array_partition variable=chunk_im3 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1287
    #pragma HLS unroll
      int v1002 = i8 >> 2;	// L1288
      int32_t v1003 = v1002;	// L1289
      int32_t v1004 = v1003 & 1;	// L1290
      int32_t v1005 = v1004 << 4;	// L1291
      int32_t v1006 = k13;	// L1292
      int32_t v1007 = v1006 ^ v1005;	// L1293
      int32_t bank6;	// L1294
      bank6 = v1007;	// L1295
      int32_t v1009 = bank6;	// L1296
      int v1010 = v1009;	// L1297
      float v1011 = out_re_b2[v1010][i8];	// L1298
      chunk_re3[k13] = v1011;	// L1299
      int32_t v1012 = bank6;	// L1300
      int v1013 = v1012;	// L1301
      float v1014 = out_im_b2[v1013][i8];	// L1302
      chunk_im3[k13] = v1014;	// L1303
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re3[_iv0];
      }
      v876.write(_vec);
    }	// L1305
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im3[_iv0];
      }
      v877.write(_vec);
    }	// L1306
  }
}

/// This is top function.
void fft_256(
  hls::stream< hls::vector< float, 32 > >& v1015,
  hls::stream< hls::vector< float, 32 > >& v1016,
  hls::stream< hls::vector< float, 32 > >& v1017,
  hls::stream< hls::vector< float, 32 > >& v1018
) {	// L1310
  #pragma HLS dataflow disable_start_propagation
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1019;
  #pragma HLS stream variable=v1019 depth=2	// L1311
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1020;
  #pragma HLS stream variable=v1020 depth=2	// L1312
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1021;
  #pragma HLS stream variable=v1021 depth=2	// L1313
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1022;
  #pragma HLS stream variable=v1022 depth=2	// L1314
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1023;
  #pragma HLS stream variable=v1023 depth=2	// L1315
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1024;
  #pragma HLS stream variable=v1024 depth=2	// L1316
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1025;
  #pragma HLS stream variable=v1025 depth=2	// L1317
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1026;
  #pragma HLS stream variable=v1026 depth=2	// L1318
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1027;
  #pragma HLS stream variable=v1027 depth=2	// L1319
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1028;
  #pragma HLS stream variable=v1028 depth=2	// L1320
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1029;
  #pragma HLS stream variable=v1029 depth=2	// L1321
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1030;
  #pragma HLS stream variable=v1030 depth=2	// L1322
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1031;
  #pragma HLS stream variable=v1031 depth=2	// L1323
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1032;
  #pragma HLS stream variable=v1032 depth=2	// L1324
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1033;
  #pragma HLS stream variable=v1033 depth=2	// L1325
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1034;
  #pragma HLS stream variable=v1034 depth=2	// L1326
  bit_rev_stage_0(v1015, v1016, v1019, v1027);	// L1327
  intra_0_0(v1019, v1027, v1020, v1028);	// L1328
  intra_1_0(v1020, v1028, v1021, v1029);	// L1329
  intra_2_0(v1021, v1029, v1022, v1030);	// L1330
  intra_3_0(v1022, v1030, v1023, v1031);	// L1331
  intra_4_0(v1023, v1031, v1024, v1032);	// L1332
  inter_5_0(v1024, v1032, v1025, v1033);	// L1333
  inter_6_0(v1025, v1033, v1026, v1034);	// L1334
  inter_7_0(v1026, v1034, v1017, v1018);	// L1335
}



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
  // placeholder for const float twr	// L694
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L695
  #pragma HLS array_partition variable=twi complete
  float in_re[32][8];	// L696
  #pragma HLS array_partition variable=in_re complete dim=1

  #pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L697
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L698
  #pragma HLS array_partition variable=out_re_b complete dim=1

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L699
  #pragma HLS array_partition variable=out_im_b complete dim=1

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L700
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v528 = v517.read();
    hls::vector< float, 32 > v529 = v518.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L703
    #pragma HLS unroll
      int32_t v531 = k5;	// L704
      int32_t v532 = v531 & 15;	// L705
      int v533 = k5 >> 4;	// L706
      int32_t v534 = i;	// L707
      int32_t v535 = v534 & 1;	// L708
      int32_t v536 = v533;	// L709
      int32_t v537 = v536 ^ v535;	// L710
      int32_t v538 = v537 << 4;	// L711
      int32_t v539 = v532 | v538;	// L712
      int32_t bank1;	// L713
      bank1 = v539;	// L714
      float v541 = v528[k5];	// L715
      int32_t v542 = bank1;	// L716
      int v543 = v542;	// L717
      in_re[v543][i] = v541;	// L718
      float v544 = v529[k5];	// L719
      int32_t v545 = bank1;	// L720
      int v546 = v545;	// L721
      in_im[v546][i] = v544;	// L722
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L725
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im inter false
  #pragma HLS dependence variable=in_im intra false
  #pragma HLS dependence variable=in_re inter false
  #pragma HLS dependence variable=in_re intra false
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L726
    #pragma HLS unroll
      int v549 = i1 << 4;	// L727
      int v550 = v549 | k6;	// L728
      int32_t v551 = v550;	// L729
      int32_t bg;	// L730
      bg = v551;	// L731
      int32_t v553 = bg;	// L732
      int32_t v554 = v553 >> 5;	// L733
      int32_t grp;	// L734
      grp = v554;	// L735
      int32_t v556 = bg;	// L736
      int32_t v557 = v556 & 31;	// L737
      int32_t within;	// L738
      within = v557;	// L739
      int32_t v559 = within;	// L740
      int32_t v560 = v559 << 2;	// L741
      int32_t tw_k3;	// L742
      tw_k3 = v560;	// L743
      int32_t v562 = grp;	// L744
      int32_t v563 = v562 << 1;	// L745
      int32_t off_l;	// L746
      off_l = v563;	// L747
      int32_t v565 = off_l;	// L748
      int32_t v566 = v565 | 1;	// L749
      int32_t off_u;	// L750
      off_u = v566;	// L751
      int32_t v568 = within;	// L752
      int v569 = v568;	// L753
      int32_t v570 = off_l;	// L754
      int v571 = v570;	// L755
      float v572 = in_re[v569][v571];	// L756
      float a_re5;	// L757
      a_re5 = v572;	// L758
      int32_t v574 = within;	// L759
      int v575 = v574;	// L760
      int32_t v576 = off_l;	// L761
      int v577 = v576;	// L762
      float v578 = in_im[v575][v577];	// L763
      float a_im5;	// L764
      a_im5 = v578;	// L765
      int32_t v580 = within;	// L766
      int32_t v581 = v580 ^ 16;	// L767
      int v582 = v581;	// L768
      int32_t v583 = off_u;	// L769
      int v584 = v583;	// L770
      float v585 = in_re[v582][v584];	// L771
      float b_re5;	// L772
      b_re5 = v585;	// L773
      int32_t v587 = within;	// L774
      int32_t v588 = v587 ^ 16;	// L775
      int v589 = v588;	// L776
      int32_t v590 = off_u;	// L777
      int v591 = v590;	// L778
      float v592 = in_im[v589][v591];	// L779
      float b_im5;	// L780
      b_im5 = v592;	// L781
      int32_t v594 = tw_k3;	// L782
      bool v595 = v594 == 0;	// L783
      if (v595) {	// L784
        float v596 = a_re5;	// L785
        float v597 = b_re5;	// L786
        float v598 = v596 + v597;	// L787
        #pragma HLS bind_op variable=v598 op=fadd impl=fabric
        int32_t v599 = within;	// L788
        int v600 = v599;	// L789
        int32_t v601 = off_l;	// L790
        int v602 = v601;	// L791
        out_re_b[v600][v602] = v598;	// L792
        float v603 = a_im5;	// L793
        float v604 = b_im5;	// L794
        float v605 = v603 + v604;	// L795
        #pragma HLS bind_op variable=v605 op=fadd impl=fabric
        int32_t v606 = within;	// L796
        int v607 = v606;	// L797
        int32_t v608 = off_l;	// L798
        int v609 = v608;	// L799
        out_im_b[v607][v609] = v605;	// L800
        float v610 = a_re5;	// L801
        float v611 = b_re5;	// L802
        float v612 = v610 - v611;	// L803
        #pragma HLS bind_op variable=v612 op=fsub impl=fabric
        int32_t v613 = within;	// L804
        int32_t v614 = v613 ^ 16;	// L805
        int v615 = v614;	// L806
        int32_t v616 = off_u;	// L807
        int v617 = v616;	// L808
        out_re_b[v615][v617] = v612;	// L809
        float v618 = a_im5;	// L810
        float v619 = b_im5;	// L811
        float v620 = v618 - v619;	// L812
        #pragma HLS bind_op variable=v620 op=fsub impl=fabric
        int32_t v621 = within;	// L813
        int32_t v622 = v621 ^ 16;	// L814
        int v623 = v622;	// L815
        int32_t v624 = off_u;	// L816
        int v625 = v624;	// L817
        out_im_b[v623][v625] = v620;	// L818
      } else {
        int32_t v626 = tw_k3;	// L820
        bool v627 = v626 == 64;	// L821
        if (v627) {	// L822
          float v628 = a_re5;	// L823
          float v629 = b_im5;	// L824
          float v630 = v628 + v629;	// L825
          #pragma HLS bind_op variable=v630 op=fadd impl=fabric
          int32_t v631 = within;	// L826
          int v632 = v631;	// L827
          int32_t v633 = off_l;	// L828
          int v634 = v633;	// L829
          out_re_b[v632][v634] = v630;	// L830
          float v635 = a_im5;	// L831
          float v636 = b_re5;	// L832
          float v637 = v635 - v636;	// L833
          #pragma HLS bind_op variable=v637 op=fsub impl=fabric
          int32_t v638 = within;	// L834
          int v639 = v638;	// L835
          int32_t v640 = off_l;	// L836
          int v641 = v640;	// L837
          out_im_b[v639][v641] = v637;	// L838
          float v642 = a_re5;	// L839
          float v643 = b_im5;	// L840
          float v644 = v642 - v643;	// L841
          #pragma HLS bind_op variable=v644 op=fsub impl=fabric
          int32_t v645 = within;	// L842
          int32_t v646 = v645 ^ 16;	// L843
          int v647 = v646;	// L844
          int32_t v648 = off_u;	// L845
          int v649 = v648;	// L846
          out_re_b[v647][v649] = v644;	// L847
          float v650 = a_im5;	// L848
          float v651 = b_re5;	// L849
          float v652 = v650 + v651;	// L850
          #pragma HLS bind_op variable=v652 op=fadd impl=fabric
          int32_t v653 = within;	// L851
          int32_t v654 = v653 ^ 16;	// L852
          int v655 = v654;	// L853
          int32_t v656 = off_u;	// L854
          int v657 = v656;	// L855
          out_im_b[v655][v657] = v652;	// L856
        } else {
          int32_t v658 = tw_k3;	// L858
          int v659 = v658;	// L859
          float v660 = twr[v659];	// L860
          float tr3;	// L861
          tr3 = v660;	// L862
          int32_t v662 = tw_k3;	// L863
          int v663 = v662;	// L864
          float v664 = twi[v663];	// L865
          float ti3;	// L866
          ti3 = v664;	// L867
          float v666 = b_re5;	// L868
          float v667 = tr3;	// L869
          float v668 = v666 * v667;	// L870
          float v669 = b_im5;	// L871
          float v670 = ti3;	// L872
          float v671 = v669 * v670;	// L873
          float v672 = v668 - v671;	// L874
          float bw_re3;	// L875
          bw_re3 = v672;	// L876
          float v674 = b_re5;	// L877
          float v675 = ti3;	// L878
          float v676 = v674 * v675;	// L879
          float v677 = b_im5;	// L880
          float v678 = tr3;	// L881
          float v679 = v677 * v678;	// L882
          float v680 = v676 + v679;	// L883
          float bw_im3;	// L884
          bw_im3 = v680;	// L885
          float v682 = a_re5;	// L886
          float v683 = bw_re3;	// L887
          float v684 = v682 + v683;	// L888
          #pragma HLS bind_op variable=v684 op=fadd impl=fabric
          int32_t v685 = within;	// L889
          int v686 = v685;	// L890
          int32_t v687 = off_l;	// L891
          int v688 = v687;	// L892
          out_re_b[v686][v688] = v684;	// L893
          float v689 = a_im5;	// L894
          float v690 = bw_im3;	// L895
          float v691 = v689 + v690;	// L896
          #pragma HLS bind_op variable=v691 op=fadd impl=fabric
          int32_t v692 = within;	// L897
          int v693 = v692;	// L898
          int32_t v694 = off_l;	// L899
          int v695 = v694;	// L900
          out_im_b[v693][v695] = v691;	// L901
          float v696 = a_re5;	// L902
          float v697 = bw_re3;	// L903
          float v698 = v696 - v697;	// L904
          #pragma HLS bind_op variable=v698 op=fsub impl=fabric
          int32_t v699 = within;	// L905
          int32_t v700 = v699 ^ 16;	// L906
          int v701 = v700;	// L907
          int32_t v702 = off_u;	// L908
          int v703 = v702;	// L909
          out_re_b[v701][v703] = v698;	// L910
          float v704 = a_im5;	// L911
          float v705 = bw_im3;	// L912
          float v706 = v704 - v705;	// L913
          #pragma HLS bind_op variable=v706 op=fsub impl=fabric
          int32_t v707 = within;	// L914
          int32_t v708 = v707 ^ 16;	// L915
          int v709 = v708;	// L916
          int32_t v710 = off_u;	// L917
          int v711 = v710;	// L918
          out_im_b[v709][v711] = v706;	// L919
        }
      }
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L924
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    float chunk_re1[32];	// L925
    #pragma HLS array_partition variable=chunk_re1 complete
    float chunk_im1[32];	// L926
    #pragma HLS array_partition variable=chunk_im1 complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L927
    #pragma HLS unroll
      int32_t v716 = k7;	// L928
      int32_t v717 = v716 & 15;	// L929
      int v718 = k7 >> 4;	// L930
      int32_t v719 = i2;	// L931
      int32_t v720 = v719 & 1;	// L932
      int32_t v721 = v718;	// L933
      int32_t v722 = v721 ^ v720;	// L934
      int32_t v723 = v722 << 4;	// L935
      int32_t v724 = v717 | v723;	// L936
      int32_t bank2;	// L937
      bank2 = v724;	// L938
      int32_t v726 = bank2;	// L939
      int v727 = v726;	// L940
      float v728 = out_re_b[v727][i2];	// L941
      chunk_re1[k7] = v728;	// L942
      int32_t v729 = bank2;	// L943
      int v730 = v729;	// L944
      float v731 = out_im_b[v730][i2];	// L945
      chunk_im1[k7] = v731;	// L946
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re1[_iv0];
      }
      v519.write(_vec);
    }	// L948
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im1[_iv0];
      }
      v520.write(_vec);
    }	// L949
  }
}

void inter_6_0(
  hls::stream< hls::vector< float, 32 > >& v732,
  hls::stream< hls::vector< float, 32 > >& v733,
  hls::stream< hls::vector< float, 32 > >& v734,
  hls::stream< hls::vector< float, 32 > >& v735
) {	// L953
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L963
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L964
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L965
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L966
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L967
  #pragma HLS array_partition variable=out_re_b1 complete dim=1

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L968
  #pragma HLS array_partition variable=out_im_b1 complete dim=1

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L969
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v743 = v732.read();
    hls::vector< float, 32 > v744 = v733.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L972
    #pragma HLS unroll
      int32_t v746 = k8;	// L973
      int32_t v747 = v746 & 15;	// L974
      int v748 = k8 >> 4;	// L975
      int v749 = i3 >> 1;	// L976
      int32_t v750 = v749;	// L977
      int32_t v751 = v750 & 1;	// L978
      int32_t v752 = v748;	// L979
      int32_t v753 = v752 ^ v751;	// L980
      int32_t v754 = v753 << 4;	// L981
      int32_t v755 = v747 | v754;	// L982
      int32_t bank3;	// L983
      bank3 = v755;	// L984
      float v757 = v743[k8];	// L985
      int32_t v758 = bank3;	// L986
      int v759 = v758;	// L987
      in_re1[v759][i3] = v757;	// L988
      float v760 = v744[k8];	// L989
      int32_t v761 = bank3;	// L990
      int v762 = v761;	// L991
      in_im1[v762][i3] = v760;	// L992
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L995
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im1 inter false
  #pragma HLS dependence variable=in_im1 intra false
  #pragma HLS dependence variable=in_re1 inter false
  #pragma HLS dependence variable=in_re1 intra false
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L996
    #pragma HLS unroll
      int v765 = i4 << 4;	// L997
      int v766 = v765 | k9;	// L998
      uint32_t v767 = v766;	// L999
      uint32_t bg1;	// L1000
      bg1 = v767;	// L1001
      int32_t v769 = bg1;	// L1002
      int32_t v770 = v769 & 63;	// L1003
      int32_t v771 = v770 << 1;	// L1004
      uint32_t tw_k4;	// L1005
      tw_k4 = v771;	// L1006
      int32_t v773 = i4;	// L1007
      int32_t v774 = v773 & 1;	// L1008
      int32_t v775 = v774 << 4;	// L1009
      int32_t v776 = k9;	// L1010
      int32_t v777 = v776 | v775;	// L1011
      uint32_t bank_il;	// L1012
      bank_il = v777;	// L1013
      int32_t v779 = bank_il;	// L1014
      int32_t v780 = v779 ^ 16;	// L1015
      uint32_t bank_iu;	// L1016
      bank_iu = v780;	// L1017
      int v782 = i4 >> 2;	// L1018
      int v783 = v782 << 2;	// L1019
      int v784 = i4 >> 1;	// L1020
      int32_t v785 = v784;	// L1021
      int32_t v786 = v785 & 1;	// L1022
      int32_t v787 = v783;	// L1023
      int32_t v788 = v787 | v786;	// L1024
      uint32_t off_il;	// L1025
      off_il = v788;	// L1026
      int32_t v790 = off_il;	// L1027
      int32_t v791 = v790 | 2;	// L1028
      uint32_t off_iu;	// L1029
      off_iu = v791;	// L1030
      int32_t v793 = bank_il;	// L1031
      int v794 = v793;	// L1032
      int32_t v795 = off_il;	// L1033
      int v796 = v795;	// L1034
      float v797 = in_re1[v794][v796];	// L1035
      float a_re6;	// L1036
      a_re6 = v797;	// L1037
      int32_t v799 = bank_il;	// L1038
      int v800 = v799;	// L1039
      int32_t v801 = off_il;	// L1040
      int v802 = v801;	// L1041
      float v803 = in_im1[v800][v802];	// L1042
      float a_im6;	// L1043
      a_im6 = v803;	// L1044
      int32_t v805 = bank_iu;	// L1045
      int v806 = v805;	// L1046
      int32_t v807 = off_iu;	// L1047
      int v808 = v807;	// L1048
      float v809 = in_re1[v806][v808];	// L1049
      float b_re6;	// L1050
      b_re6 = v809;	// L1051
      int32_t v811 = bank_iu;	// L1052
      int v812 = v811;	// L1053
      int32_t v813 = off_iu;	// L1054
      int v814 = v813;	// L1055
      float v815 = in_im1[v812][v814];	// L1056
      float b_im6;	// L1057
      b_im6 = v815;	// L1058
      int32_t v817 = tw_k4;	// L1059
      int v818 = v817;	// L1060
      float v819 = twr[v818];	// L1061
      float tr4;	// L1062
      tr4 = v819;	// L1063
      int32_t v821 = tw_k4;	// L1064
      int v822 = v821;	// L1065
      float v823 = twi[v822];	// L1066
      float ti4;	// L1067
      ti4 = v823;	// L1068
      float v825 = b_re6;	// L1069
      float v826 = tr4;	// L1070
      float v827 = v825 * v826;	// L1071
      float v828 = b_im6;	// L1072
      float v829 = ti4;	// L1073
      float v830 = v828 * v829;	// L1074
      float v831 = v827 - v830;	// L1075
      float bw_re4;	// L1076
      bw_re4 = v831;	// L1077
      float v833 = b_re6;	// L1078
      float v834 = ti4;	// L1079
      float v835 = v833 * v834;	// L1080
      float v836 = b_im6;	// L1081
      float v837 = tr4;	// L1082
      float v838 = v836 * v837;	// L1083
      float v839 = v835 + v838;	// L1084
      float bw_im4;	// L1085
      bw_im4 = v839;	// L1086
      float v841 = a_re6;	// L1087
      float v842 = bw_re4;	// L1088
      float v843 = v841 + v842;	// L1089
      #pragma HLS bind_op variable=v843 op=fadd impl=fabric
      int32_t v844 = bank_il;	// L1090
      int v845 = v844;	// L1091
      int32_t v846 = off_il;	// L1092
      int v847 = v846;	// L1093
      out_re_b1[v845][v847] = v843;	// L1094
      float v848 = a_im6;	// L1095
      float v849 = bw_im4;	// L1096
      float v850 = v848 + v849;	// L1097
      #pragma HLS bind_op variable=v850 op=fadd impl=fabric
      int32_t v851 = bank_il;	// L1098
      int v852 = v851;	// L1099
      int32_t v853 = off_il;	// L1100
      int v854 = v853;	// L1101
      out_im_b1[v852][v854] = v850;	// L1102
      float v855 = a_re6;	// L1103
      float v856 = bw_re4;	// L1104
      float v857 = v855 - v856;	// L1105
      #pragma HLS bind_op variable=v857 op=fsub impl=fabric
      int32_t v858 = bank_iu;	// L1106
      int v859 = v858;	// L1107
      int32_t v860 = off_iu;	// L1108
      int v861 = v860;	// L1109
      out_re_b1[v859][v861] = v857;	// L1110
      float v862 = a_im6;	// L1111
      float v863 = bw_im4;	// L1112
      float v864 = v862 - v863;	// L1113
      #pragma HLS bind_op variable=v864 op=fsub impl=fabric
      int32_t v865 = bank_iu;	// L1114
      int v866 = v865;	// L1115
      int32_t v867 = off_iu;	// L1116
      int v868 = v867;	// L1117
      out_im_b1[v866][v868] = v864;	// L1118
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L1121
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    float chunk_re2[32];	// L1122
    #pragma HLS array_partition variable=chunk_re2 complete
    float chunk_im2[32];	// L1123
    #pragma HLS array_partition variable=chunk_im2 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L1124
    #pragma HLS unroll
      int32_t v873 = k10;	// L1125
      int32_t v874 = v873 & 15;	// L1126
      int v875 = k10 >> 4;	// L1127
      int v876 = i5 >> 1;	// L1128
      int32_t v877 = v876;	// L1129
      int32_t v878 = v877 & 1;	// L1130
      int32_t v879 = v875;	// L1131
      int32_t v880 = v879 ^ v878;	// L1132
      int32_t v881 = v880 << 4;	// L1133
      int32_t v882 = v874 | v881;	// L1134
      int32_t bank4;	// L1135
      bank4 = v882;	// L1136
      int32_t v884 = bank4;	// L1137
      int v885 = v884;	// L1138
      float v886 = out_re_b1[v885][i5];	// L1139
      chunk_re2[k10] = v886;	// L1140
      int32_t v887 = bank4;	// L1141
      int v888 = v887;	// L1142
      float v889 = out_im_b1[v888][i5];	// L1143
      chunk_im2[k10] = v889;	// L1144
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re2[_iv0];
      }
      v734.write(_vec);
    }	// L1146
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im2[_iv0];
      }
      v735.write(_vec);
    }	// L1147
  }
}

void inter_7_0(
  hls::stream< hls::vector< float, 32 > >& v890,
  hls::stream< hls::vector< float, 32 > >& v891,
  hls::stream< hls::vector< float, 32 > >& v892,
  hls::stream< hls::vector< float, 32 > >& v893
) {	// L1151
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1159
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1160
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L1161
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L1162
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L1163
  #pragma HLS array_partition variable=out_re_b2 complete dim=1

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L1164
  #pragma HLS array_partition variable=out_im_b2 complete dim=1

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L1165
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v901 = v890.read();
    hls::vector< float, 32 > v902 = v891.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L1168
    #pragma HLS unroll
      int32_t v904 = k11;	// L1169
      int32_t v905 = v904 & 15;	// L1170
      int v906 = k11 >> 4;	// L1171
      int v907 = i6 >> 2;	// L1172
      int32_t v908 = v907;	// L1173
      int32_t v909 = v908 & 1;	// L1174
      int32_t v910 = v906;	// L1175
      int32_t v911 = v910 ^ v909;	// L1176
      int32_t v912 = v911 << 4;	// L1177
      int32_t v913 = v905 | v912;	// L1178
      int32_t bank5;	// L1179
      bank5 = v913;	// L1180
      float v915 = v901[k11];	// L1181
      int32_t v916 = bank5;	// L1182
      int v917 = v916;	// L1183
      in_re2[v917][i6] = v915;	// L1184
      float v918 = v902[k11];	// L1185
      int32_t v919 = bank5;	// L1186
      int v920 = v919;	// L1187
      in_im2[v920][i6] = v918;	// L1188
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L1191
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im2 inter false
  #pragma HLS dependence variable=in_im2 intra false
  #pragma HLS dependence variable=in_re2 inter false
  #pragma HLS dependence variable=in_re2 intra false
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L1192
    #pragma HLS unroll
      int32_t v923 = i7;	// L1193
      int32_t v924 = v923 & 1;	// L1194
      int32_t v925 = v924 << 4;	// L1195
      int32_t v926 = k12;	// L1196
      int32_t v927 = v926 | v925;	// L1197
      uint32_t bank_il1;	// L1198
      bank_il1 = v927;	// L1199
      int32_t v929 = bank_il1;	// L1200
      int32_t v930 = v929 ^ 16;	// L1201
      uint32_t bank_iu1;	// L1202
      bank_iu1 = v930;	// L1203
      int v932 = i7 >> 1;	// L1204
      uint32_t v933 = v932;	// L1205
      uint32_t off_il1;	// L1206
      off_il1 = v933;	// L1207
      int32_t v935 = off_il1;	// L1208
      int32_t v936 = v935 | 4;	// L1209
      uint32_t off_iu1;	// L1210
      off_iu1 = v936;	// L1211
      int v938 = i7 << 4;	// L1212
      int v939 = v938 | k12;	// L1213
      uint32_t v940 = v939;	// L1214
      uint32_t tw_k5;	// L1215
      tw_k5 = v940;	// L1216
      int32_t v942 = bank_il1;	// L1217
      int v943 = v942;	// L1218
      int32_t v944 = off_il1;	// L1219
      int v945 = v944;	// L1220
      float v946 = in_re2[v943][v945];	// L1221
      float a_re7;	// L1222
      a_re7 = v946;	// L1223
      int32_t v948 = bank_il1;	// L1224
      int v949 = v948;	// L1225
      int32_t v950 = off_il1;	// L1226
      int v951 = v950;	// L1227
      float v952 = in_im2[v949][v951];	// L1228
      float a_im7;	// L1229
      a_im7 = v952;	// L1230
      int32_t v954 = bank_iu1;	// L1231
      int v955 = v954;	// L1232
      int32_t v956 = off_iu1;	// L1233
      int v957 = v956;	// L1234
      float v958 = in_re2[v955][v957];	// L1235
      float b_re7;	// L1236
      b_re7 = v958;	// L1237
      int32_t v960 = bank_iu1;	// L1238
      int v961 = v960;	// L1239
      int32_t v962 = off_iu1;	// L1240
      int v963 = v962;	// L1241
      float v964 = in_im2[v961][v963];	// L1242
      float b_im7;	// L1243
      b_im7 = v964;	// L1244
      int32_t v966 = tw_k5;	// L1245
      int v967 = v966;	// L1246
      float v968 = twr[v967];	// L1247
      float tr5;	// L1248
      tr5 = v968;	// L1249
      int32_t v970 = tw_k5;	// L1250
      int v971 = v970;	// L1251
      float v972 = twi[v971];	// L1252
      float ti5;	// L1253
      ti5 = v972;	// L1254
      float v974 = b_re7;	// L1255
      float v975 = tr5;	// L1256
      float v976 = v974 * v975;	// L1257
      float v977 = b_im7;	// L1258
      float v978 = ti5;	// L1259
      float v979 = v977 * v978;	// L1260
      float v980 = v976 - v979;	// L1261
      float bw_re5;	// L1262
      bw_re5 = v980;	// L1263
      float v982 = b_re7;	// L1264
      float v983 = ti5;	// L1265
      float v984 = v982 * v983;	// L1266
      float v985 = b_im7;	// L1267
      float v986 = tr5;	// L1268
      float v987 = v985 * v986;	// L1269
      float v988 = v984 + v987;	// L1270
      float bw_im5;	// L1271
      bw_im5 = v988;	// L1272
      float v990 = a_re7;	// L1273
      float v991 = bw_re5;	// L1274
      float v992 = v990 + v991;	// L1275
      #pragma HLS bind_op variable=v992 op=fadd impl=fabric
      int32_t v993 = bank_il1;	// L1276
      int v994 = v993;	// L1277
      int32_t v995 = off_il1;	// L1278
      int v996 = v995;	// L1279
      out_re_b2[v994][v996] = v992;	// L1280
      float v997 = a_im7;	// L1281
      float v998 = bw_im5;	// L1282
      float v999 = v997 + v998;	// L1283
      #pragma HLS bind_op variable=v999 op=fadd impl=fabric
      int32_t v1000 = bank_il1;	// L1284
      int v1001 = v1000;	// L1285
      int32_t v1002 = off_il1;	// L1286
      int v1003 = v1002;	// L1287
      out_im_b2[v1001][v1003] = v999;	// L1288
      float v1004 = a_re7;	// L1289
      float v1005 = bw_re5;	// L1290
      float v1006 = v1004 - v1005;	// L1291
      #pragma HLS bind_op variable=v1006 op=fsub impl=fabric
      int32_t v1007 = bank_iu1;	// L1292
      int v1008 = v1007;	// L1293
      int32_t v1009 = off_iu1;	// L1294
      int v1010 = v1009;	// L1295
      out_re_b2[v1008][v1010] = v1006;	// L1296
      float v1011 = a_im7;	// L1297
      float v1012 = bw_im5;	// L1298
      float v1013 = v1011 - v1012;	// L1299
      #pragma HLS bind_op variable=v1013 op=fsub impl=fabric
      int32_t v1014 = bank_iu1;	// L1300
      int v1015 = v1014;	// L1301
      int32_t v1016 = off_iu1;	// L1302
      int v1017 = v1016;	// L1303
      out_im_b2[v1015][v1017] = v1013;	// L1304
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1307
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    float chunk_re3[32];	// L1308
    #pragma HLS array_partition variable=chunk_re3 complete
    float chunk_im3[32];	// L1309
    #pragma HLS array_partition variable=chunk_im3 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1310
    #pragma HLS unroll
      int32_t v1022 = k13;	// L1311
      int32_t v1023 = v1022 & 15;	// L1312
      int v1024 = k13 >> 4;	// L1313
      int v1025 = i8 >> 2;	// L1314
      int32_t v1026 = v1025;	// L1315
      int32_t v1027 = v1026 & 1;	// L1316
      int32_t v1028 = v1024;	// L1317
      int32_t v1029 = v1028 ^ v1027;	// L1318
      int32_t v1030 = v1029 << 4;	// L1319
      int32_t v1031 = v1023 | v1030;	// L1320
      int32_t bank6;	// L1321
      bank6 = v1031;	// L1322
      int32_t v1033 = bank6;	// L1323
      int v1034 = v1033;	// L1324
      float v1035 = out_re_b2[v1034][i8];	// L1325
      chunk_re3[k13] = v1035;	// L1326
      int32_t v1036 = bank6;	// L1327
      int v1037 = v1036;	// L1328
      float v1038 = out_im_b2[v1037][i8];	// L1329
      chunk_im3[k13] = v1038;	// L1330
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re3[_iv0];
      }
      v892.write(_vec);
    }	// L1332
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im3[_iv0];
      }
      v893.write(_vec);
    }	// L1333
  }
}

/// This is top function.
void fft_256(
  hls::stream< hls::vector< float, 32 > >& v1039,
  hls::stream< hls::vector< float, 32 > >& v1040,
  hls::stream< hls::vector< float, 32 > >& v1041,
  hls::stream< hls::vector< float, 32 > >& v1042
) {	// L1337
  #pragma HLS dataflow disable_start_propagation
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1043;
  #pragma HLS stream variable=v1043 depth=2	// L1338
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1044;
  #pragma HLS stream variable=v1044 depth=2	// L1339
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1045;
  #pragma HLS stream variable=v1045 depth=2	// L1340
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1046;
  #pragma HLS stream variable=v1046 depth=2	// L1341
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1047;
  #pragma HLS stream variable=v1047 depth=2	// L1342
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1048;
  #pragma HLS stream variable=v1048 depth=2	// L1343
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1049;
  #pragma HLS stream variable=v1049 depth=2	// L1344
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1050;
  #pragma HLS stream variable=v1050 depth=2	// L1345
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1051;
  #pragma HLS stream variable=v1051 depth=2	// L1346
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1052;
  #pragma HLS stream variable=v1052 depth=2	// L1347
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1053;
  #pragma HLS stream variable=v1053 depth=2	// L1348
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1054;
  #pragma HLS stream variable=v1054 depth=2	// L1349
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1055;
  #pragma HLS stream variable=v1055 depth=2	// L1350
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1056;
  #pragma HLS stream variable=v1056 depth=2	// L1351
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1057;
  #pragma HLS stream variable=v1057 depth=2	// L1352
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1058;
  #pragma HLS stream variable=v1058 depth=2	// L1353
  bit_rev_stage_0(v1039, v1040, v1043, v1051);	// L1354
  intra_0_0(v1043, v1051, v1044, v1052);	// L1355
  intra_1_0(v1044, v1052, v1045, v1053);	// L1356
  intra_2_0(v1045, v1053, v1046, v1054);	// L1357
  intra_3_0(v1046, v1054, v1047, v1055);	// L1358
  intra_4_0(v1047, v1055, v1048, v1056);	// L1359
  inter_5_0(v1048, v1056, v1049, v1057);	// L1360
  inter_6_0(v1049, v1057, v1050, v1058);	// L1361
  inter_7_0(v1050, v1058, v1041, v1042);	// L1362
}


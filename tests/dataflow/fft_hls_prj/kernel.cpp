
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
  // placeholder for const float twr	// L197
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L198
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i2: for (int _i2 = 0; _i2 < 8; _i2++) {	// L199
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v143 = v136.read();
    hls::vector< float, 32 > v144 = v137.read();
    float o_re2[32];	// L202
    #pragma HLS array_partition variable=o_re2 complete
    float o_im2[32];	// L203
    #pragma HLS array_partition variable=o_im2 complete
    l_S_k_0_k2: for (int k2 = 0; k2 < 16; k2++) {	// L204
    #pragma HLS unroll
      int v148 = k2 >> 2;	// L205
      int v149 = v148 << 3;	// L206
      int32_t v150 = k2;	// L207
      int32_t v151 = v150 & 3;	// L208
      int32_t v152 = v149;	// L209
      int32_t v153 = v152 | v151;	// L210
      int32_t il;	// L211
      il = v153;	// L212
      int32_t v155 = il;	// L213
      int32_t v156 = v155 | 4;	// L214
      int32_t iu;	// L215
      iu = v156;	// L216
      int32_t v158 = v151 << 5;	// L217
      int32_t tw_k;	// L218
      tw_k = v158;	// L219
      int32_t v160 = il;	// L220
      int v161 = v160;	// L221
      float v162 = v143[v161];	// L222
      float a_re2;	// L223
      a_re2 = v162;	// L224
      int32_t v164 = il;	// L225
      int v165 = v164;	// L226
      float v166 = v144[v165];	// L227
      float a_im2;	// L228
      a_im2 = v166;	// L229
      int32_t v168 = iu;	// L230
      int v169 = v168;	// L231
      float v170 = v143[v169];	// L232
      float b_re2;	// L233
      b_re2 = v170;	// L234
      int32_t v172 = iu;	// L235
      int v173 = v172;	// L236
      float v174 = v144[v173];	// L237
      float b_im2;	// L238
      b_im2 = v174;	// L239
      int32_t v176 = tw_k;	// L240
      bool v177 = v176 == 0;	// L241
      if (v177) {	// L242
        float v178 = a_re2;	// L243
        float v179 = b_re2;	// L244
        float v180 = v178 + v179;	// L245
        #pragma HLS bind_op variable=v180 op=fadd impl=fabric
        int32_t v181 = il;	// L246
        int v182 = v181;	// L247
        o_re2[v182] = v180;	// L248
        float v183 = a_im2;	// L249
        float v184 = b_im2;	// L250
        float v185 = v183 + v184;	// L251
        #pragma HLS bind_op variable=v185 op=fadd impl=fabric
        int32_t v186 = il;	// L252
        int v187 = v186;	// L253
        o_im2[v187] = v185;	// L254
        float v188 = a_re2;	// L255
        float v189 = b_re2;	// L256
        float v190 = v188 - v189;	// L257
        #pragma HLS bind_op variable=v190 op=fsub impl=fabric
        int32_t v191 = iu;	// L258
        int v192 = v191;	// L259
        o_re2[v192] = v190;	// L260
        float v193 = a_im2;	// L261
        float v194 = b_im2;	// L262
        float v195 = v193 - v194;	// L263
        #pragma HLS bind_op variable=v195 op=fsub impl=fabric
        int32_t v196 = iu;	// L264
        int v197 = v196;	// L265
        o_im2[v197] = v195;	// L266
      } else {
        int32_t v198 = tw_k;	// L268
        bool v199 = v198 == 64;	// L269
        if (v199) {	// L270
          float v200 = a_re2;	// L271
          float v201 = b_im2;	// L272
          float v202 = v200 + v201;	// L273
          #pragma HLS bind_op variable=v202 op=fadd impl=fabric
          int32_t v203 = il;	// L274
          int v204 = v203;	// L275
          o_re2[v204] = v202;	// L276
          float v205 = a_im2;	// L277
          float v206 = b_re2;	// L278
          float v207 = v205 - v206;	// L279
          #pragma HLS bind_op variable=v207 op=fsub impl=fabric
          int32_t v208 = il;	// L280
          int v209 = v208;	// L281
          o_im2[v209] = v207;	// L282
          float v210 = a_re2;	// L283
          float v211 = b_im2;	// L284
          float v212 = v210 - v211;	// L285
          #pragma HLS bind_op variable=v212 op=fsub impl=fabric
          int32_t v213 = iu;	// L286
          int v214 = v213;	// L287
          o_re2[v214] = v212;	// L288
          float v215 = a_im2;	// L289
          float v216 = b_re2;	// L290
          float v217 = v215 + v216;	// L291
          #pragma HLS bind_op variable=v217 op=fadd impl=fabric
          int32_t v218 = iu;	// L292
          int v219 = v218;	// L293
          o_im2[v219] = v217;	// L294
        } else {
          int32_t v220 = tw_k;	// L296
          int v221 = v220;	// L297
          float v222 = twr[v221];	// L298
          float tr;	// L299
          tr = v222;	// L300
          int32_t v224 = tw_k;	// L301
          int v225 = v224;	// L302
          float v226 = twi[v225];	// L303
          float ti;	// L304
          ti = v226;	// L305
          float v228 = b_re2;	// L306
          float v229 = tr;	// L307
          float v230 = v228 * v229;	// L308
          float v231 = b_im2;	// L309
          float v232 = ti;	// L310
          float v233 = v231 * v232;	// L311
          float v234 = v230 - v233;	// L312
          float bw_re;	// L313
          bw_re = v234;	// L314
          float v236 = b_re2;	// L315
          float v237 = ti;	// L316
          float v238 = v236 * v237;	// L317
          float v239 = b_im2;	// L318
          float v240 = tr;	// L319
          float v241 = v239 * v240;	// L320
          float v242 = v238 + v241;	// L321
          float bw_im;	// L322
          bw_im = v242;	// L323
          float v244 = a_re2;	// L324
          float v245 = bw_re;	// L325
          float v246 = v244 + v245;	// L326
          #pragma HLS bind_op variable=v246 op=fadd impl=fabric
          int32_t v247 = il;	// L327
          int v248 = v247;	// L328
          o_re2[v248] = v246;	// L329
          float v249 = a_im2;	// L330
          float v250 = bw_im;	// L331
          float v251 = v249 + v250;	// L332
          #pragma HLS bind_op variable=v251 op=fadd impl=fabric
          int32_t v252 = il;	// L333
          int v253 = v252;	// L334
          o_im2[v253] = v251;	// L335
          float v254 = a_re2;	// L336
          float v255 = bw_re;	// L337
          float v256 = v254 - v255;	// L338
          #pragma HLS bind_op variable=v256 op=fsub impl=fabric
          int32_t v257 = iu;	// L339
          int v258 = v257;	// L340
          o_re2[v258] = v256;	// L341
          float v259 = a_im2;	// L342
          float v260 = bw_im;	// L343
          float v261 = v259 - v260;	// L344
          #pragma HLS bind_op variable=v261 op=fsub impl=fabric
          int32_t v262 = iu;	// L345
          int v263 = v262;	// L346
          o_im2[v263] = v261;	// L347
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re2[_iv0];
      }
      v138.write(_vec);
    }	// L351
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im2[_iv0];
      }
      v139.write(_vec);
    }	// L352
  }
}

void intra_3_0(
  hls::stream< hls::vector< float, 32 > >& v264,
  hls::stream< hls::vector< float, 32 > >& v265,
  hls::stream< hls::vector< float, 32 > >& v266,
  hls::stream< hls::vector< float, 32 > >& v267
) {	// L356
  // placeholder for const float twr	// L364
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L365
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i3: for (int _i3 = 0; _i3 < 8; _i3++) {	// L366
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v271 = v264.read();
    hls::vector< float, 32 > v272 = v265.read();
    float o_re3[32];	// L369
    #pragma HLS array_partition variable=o_re3 complete
    float o_im3[32];	// L370
    #pragma HLS array_partition variable=o_im3 complete
    l_S_k_0_k3: for (int k3 = 0; k3 < 16; k3++) {	// L371
    #pragma HLS unroll
      int v276 = k3 >> 3;	// L372
      int v277 = v276 << 4;	// L373
      int32_t v278 = k3;	// L374
      int32_t v279 = v278 & 7;	// L375
      int32_t v280 = v277;	// L376
      int32_t v281 = v280 | v279;	// L377
      int32_t il1;	// L378
      il1 = v281;	// L379
      int32_t v283 = il1;	// L380
      int32_t v284 = v283 | 8;	// L381
      int32_t iu1;	// L382
      iu1 = v284;	// L383
      int32_t v286 = v279 << 4;	// L384
      int32_t tw_k1;	// L385
      tw_k1 = v286;	// L386
      int32_t v288 = il1;	// L387
      int v289 = v288;	// L388
      float v290 = v271[v289];	// L389
      float a_re3;	// L390
      a_re3 = v290;	// L391
      int32_t v292 = il1;	// L392
      int v293 = v292;	// L393
      float v294 = v272[v293];	// L394
      float a_im3;	// L395
      a_im3 = v294;	// L396
      int32_t v296 = iu1;	// L397
      int v297 = v296;	// L398
      float v298 = v271[v297];	// L399
      float b_re3;	// L400
      b_re3 = v298;	// L401
      int32_t v300 = iu1;	// L402
      int v301 = v300;	// L403
      float v302 = v272[v301];	// L404
      float b_im3;	// L405
      b_im3 = v302;	// L406
      int32_t v304 = tw_k1;	// L407
      bool v305 = v304 == 0;	// L408
      if (v305) {	// L409
        float v306 = a_re3;	// L410
        float v307 = b_re3;	// L411
        float v308 = v306 + v307;	// L412
        #pragma HLS bind_op variable=v308 op=fadd impl=fabric
        int32_t v309 = il1;	// L413
        int v310 = v309;	// L414
        o_re3[v310] = v308;	// L415
        float v311 = a_im3;	// L416
        float v312 = b_im3;	// L417
        float v313 = v311 + v312;	// L418
        #pragma HLS bind_op variable=v313 op=fadd impl=fabric
        int32_t v314 = il1;	// L419
        int v315 = v314;	// L420
        o_im3[v315] = v313;	// L421
        float v316 = a_re3;	// L422
        float v317 = b_re3;	// L423
        float v318 = v316 - v317;	// L424
        #pragma HLS bind_op variable=v318 op=fsub impl=fabric
        int32_t v319 = iu1;	// L425
        int v320 = v319;	// L426
        o_re3[v320] = v318;	// L427
        float v321 = a_im3;	// L428
        float v322 = b_im3;	// L429
        float v323 = v321 - v322;	// L430
        #pragma HLS bind_op variable=v323 op=fsub impl=fabric
        int32_t v324 = iu1;	// L431
        int v325 = v324;	// L432
        o_im3[v325] = v323;	// L433
      } else {
        int32_t v326 = tw_k1;	// L435
        bool v327 = v326 == 64;	// L436
        if (v327) {	// L437
          float v328 = a_re3;	// L438
          float v329 = b_im3;	// L439
          float v330 = v328 + v329;	// L440
          #pragma HLS bind_op variable=v330 op=fadd impl=fabric
          int32_t v331 = il1;	// L441
          int v332 = v331;	// L442
          o_re3[v332] = v330;	// L443
          float v333 = a_im3;	// L444
          float v334 = b_re3;	// L445
          float v335 = v333 - v334;	// L446
          #pragma HLS bind_op variable=v335 op=fsub impl=fabric
          int32_t v336 = il1;	// L447
          int v337 = v336;	// L448
          o_im3[v337] = v335;	// L449
          float v338 = a_re3;	// L450
          float v339 = b_im3;	// L451
          float v340 = v338 - v339;	// L452
          #pragma HLS bind_op variable=v340 op=fsub impl=fabric
          int32_t v341 = iu1;	// L453
          int v342 = v341;	// L454
          o_re3[v342] = v340;	// L455
          float v343 = a_im3;	// L456
          float v344 = b_re3;	// L457
          float v345 = v343 + v344;	// L458
          #pragma HLS bind_op variable=v345 op=fadd impl=fabric
          int32_t v346 = iu1;	// L459
          int v347 = v346;	// L460
          o_im3[v347] = v345;	// L461
        } else {
          int32_t v348 = tw_k1;	// L463
          int v349 = v348;	// L464
          float v350 = twr[v349];	// L465
          float tr1;	// L466
          tr1 = v350;	// L467
          int32_t v352 = tw_k1;	// L468
          int v353 = v352;	// L469
          float v354 = twi[v353];	// L470
          float ti1;	// L471
          ti1 = v354;	// L472
          float v356 = b_re3;	// L473
          float v357 = tr1;	// L474
          float v358 = v356 * v357;	// L475
          float v359 = b_im3;	// L476
          float v360 = ti1;	// L477
          float v361 = v359 * v360;	// L478
          float v362 = v358 - v361;	// L479
          float bw_re1;	// L480
          bw_re1 = v362;	// L481
          float v364 = b_re3;	// L482
          float v365 = ti1;	// L483
          float v366 = v364 * v365;	// L484
          float v367 = b_im3;	// L485
          float v368 = tr1;	// L486
          float v369 = v367 * v368;	// L487
          float v370 = v366 + v369;	// L488
          float bw_im1;	// L489
          bw_im1 = v370;	// L490
          float v372 = a_re3;	// L491
          float v373 = bw_re1;	// L492
          float v374 = v372 + v373;	// L493
          #pragma HLS bind_op variable=v374 op=fadd impl=fabric
          int32_t v375 = il1;	// L494
          int v376 = v375;	// L495
          o_re3[v376] = v374;	// L496
          float v377 = a_im3;	// L497
          float v378 = bw_im1;	// L498
          float v379 = v377 + v378;	// L499
          #pragma HLS bind_op variable=v379 op=fadd impl=fabric
          int32_t v380 = il1;	// L500
          int v381 = v380;	// L501
          o_im3[v381] = v379;	// L502
          float v382 = a_re3;	// L503
          float v383 = bw_re1;	// L504
          float v384 = v382 - v383;	// L505
          #pragma HLS bind_op variable=v384 op=fsub impl=fabric
          int32_t v385 = iu1;	// L506
          int v386 = v385;	// L507
          o_re3[v386] = v384;	// L508
          float v387 = a_im3;	// L509
          float v388 = bw_im1;	// L510
          float v389 = v387 - v388;	// L511
          #pragma HLS bind_op variable=v389 op=fsub impl=fabric
          int32_t v390 = iu1;	// L512
          int v391 = v390;	// L513
          o_im3[v391] = v389;	// L514
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re3[_iv0];
      }
      v266.write(_vec);
    }	// L518
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im3[_iv0];
      }
      v267.write(_vec);
    }	// L519
  }
}

void intra_4_0(
  hls::stream< hls::vector< float, 32 > >& v392,
  hls::stream< hls::vector< float, 32 > >& v393,
  hls::stream< hls::vector< float, 32 > >& v394,
  hls::stream< hls::vector< float, 32 > >& v395
) {	// L523
  // placeholder for const float twr	// L528
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L529
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i4: for (int _i4 = 0; _i4 < 8; _i4++) {	// L530
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v399 = v392.read();
    hls::vector< float, 32 > v400 = v393.read();
    float o_re4[32];	// L533
    #pragma HLS array_partition variable=o_re4 complete
    float o_im4[32];	// L534
    #pragma HLS array_partition variable=o_im4 complete
    l_S_k_0_k4: for (int k4 = 0; k4 < 16; k4++) {	// L535
    #pragma HLS unroll
      int32_t v404 = k4;	// L536
      int32_t il2;	// L537
      il2 = v404;	// L538
      int32_t v406 = v404 | 16;	// L539
      int32_t iu2;	// L540
      iu2 = v406;	// L541
      int v408 = k4 << 3;	// L542
      int32_t v409 = v408;	// L543
      int32_t tw_k2;	// L544
      tw_k2 = v409;	// L545
      int32_t v411 = il2;	// L546
      int v412 = v411;	// L547
      float v413 = v399[v412];	// L548
      float a_re4;	// L549
      a_re4 = v413;	// L550
      int32_t v415 = il2;	// L551
      int v416 = v415;	// L552
      float v417 = v400[v416];	// L553
      float a_im4;	// L554
      a_im4 = v417;	// L555
      int32_t v419 = iu2;	// L556
      int v420 = v419;	// L557
      float v421 = v399[v420];	// L558
      float b_re4;	// L559
      b_re4 = v421;	// L560
      int32_t v423 = iu2;	// L561
      int v424 = v423;	// L562
      float v425 = v400[v424];	// L563
      float b_im4;	// L564
      b_im4 = v425;	// L565
      int32_t v427 = tw_k2;	// L566
      bool v428 = v427 == 0;	// L567
      if (v428) {	// L568
        float v429 = a_re4;	// L569
        float v430 = b_re4;	// L570
        float v431 = v429 + v430;	// L571
        #pragma HLS bind_op variable=v431 op=fadd impl=fabric
        int32_t v432 = il2;	// L572
        int v433 = v432;	// L573
        o_re4[v433] = v431;	// L574
        float v434 = a_im4;	// L575
        float v435 = b_im4;	// L576
        float v436 = v434 + v435;	// L577
        #pragma HLS bind_op variable=v436 op=fadd impl=fabric
        int32_t v437 = il2;	// L578
        int v438 = v437;	// L579
        o_im4[v438] = v436;	// L580
        float v439 = a_re4;	// L581
        float v440 = b_re4;	// L582
        float v441 = v439 - v440;	// L583
        #pragma HLS bind_op variable=v441 op=fsub impl=fabric
        int32_t v442 = iu2;	// L584
        int v443 = v442;	// L585
        o_re4[v443] = v441;	// L586
        float v444 = a_im4;	// L587
        float v445 = b_im4;	// L588
        float v446 = v444 - v445;	// L589
        #pragma HLS bind_op variable=v446 op=fsub impl=fabric
        int32_t v447 = iu2;	// L590
        int v448 = v447;	// L591
        o_im4[v448] = v446;	// L592
      } else {
        int32_t v449 = tw_k2;	// L594
        bool v450 = v449 == 64;	// L595
        if (v450) {	// L596
          float v451 = a_re4;	// L597
          float v452 = b_im4;	// L598
          float v453 = v451 + v452;	// L599
          #pragma HLS bind_op variable=v453 op=fadd impl=fabric
          int32_t v454 = il2;	// L600
          int v455 = v454;	// L601
          o_re4[v455] = v453;	// L602
          float v456 = a_im4;	// L603
          float v457 = b_re4;	// L604
          float v458 = v456 - v457;	// L605
          #pragma HLS bind_op variable=v458 op=fsub impl=fabric
          int32_t v459 = il2;	// L606
          int v460 = v459;	// L607
          o_im4[v460] = v458;	// L608
          float v461 = a_re4;	// L609
          float v462 = b_im4;	// L610
          float v463 = v461 - v462;	// L611
          #pragma HLS bind_op variable=v463 op=fsub impl=fabric
          int32_t v464 = iu2;	// L612
          int v465 = v464;	// L613
          o_re4[v465] = v463;	// L614
          float v466 = a_im4;	// L615
          float v467 = b_re4;	// L616
          float v468 = v466 + v467;	// L617
          #pragma HLS bind_op variable=v468 op=fadd impl=fabric
          int32_t v469 = iu2;	// L618
          int v470 = v469;	// L619
          o_im4[v470] = v468;	// L620
        } else {
          int32_t v471 = tw_k2;	// L622
          int v472 = v471;	// L623
          float v473 = twr[v472];	// L624
          float tr2;	// L625
          tr2 = v473;	// L626
          int32_t v475 = tw_k2;	// L627
          int v476 = v475;	// L628
          float v477 = twi[v476];	// L629
          float ti2;	// L630
          ti2 = v477;	// L631
          float v479 = b_re4;	// L632
          float v480 = tr2;	// L633
          float v481 = v479 * v480;	// L634
          float v482 = b_im4;	// L635
          float v483 = ti2;	// L636
          float v484 = v482 * v483;	// L637
          float v485 = v481 - v484;	// L638
          float bw_re2;	// L639
          bw_re2 = v485;	// L640
          float v487 = b_re4;	// L641
          float v488 = ti2;	// L642
          float v489 = v487 * v488;	// L643
          float v490 = b_im4;	// L644
          float v491 = tr2;	// L645
          float v492 = v490 * v491;	// L646
          float v493 = v489 + v492;	// L647
          float bw_im2;	// L648
          bw_im2 = v493;	// L649
          float v495 = a_re4;	// L650
          float v496 = bw_re2;	// L651
          float v497 = v495 + v496;	// L652
          #pragma HLS bind_op variable=v497 op=fadd impl=fabric
          int32_t v498 = il2;	// L653
          int v499 = v498;	// L654
          o_re4[v499] = v497;	// L655
          float v500 = a_im4;	// L656
          float v501 = bw_im2;	// L657
          float v502 = v500 + v501;	// L658
          #pragma HLS bind_op variable=v502 op=fadd impl=fabric
          int32_t v503 = il2;	// L659
          int v504 = v503;	// L660
          o_im4[v504] = v502;	// L661
          float v505 = a_re4;	// L662
          float v506 = bw_re2;	// L663
          float v507 = v505 - v506;	// L664
          #pragma HLS bind_op variable=v507 op=fsub impl=fabric
          int32_t v508 = iu2;	// L665
          int v509 = v508;	// L666
          o_re4[v509] = v507;	// L667
          float v510 = a_im4;	// L668
          float v511 = bw_im2;	// L669
          float v512 = v510 - v511;	// L670
          #pragma HLS bind_op variable=v512 op=fsub impl=fabric
          int32_t v513 = iu2;	// L671
          int v514 = v513;	// L672
          o_im4[v514] = v512;	// L673
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re4[_iv0];
      }
      v394.write(_vec);
    }	// L677
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im4[_iv0];
      }
      v395.write(_vec);
    }	// L678
  }
}

void inter_5_0(
  hls::stream< hls::vector< float, 32 > >& v515,
  hls::stream< hls::vector< float, 32 > >& v516,
  hls::stream< hls::vector< float, 32 > >& v517,
  hls::stream< hls::vector< float, 32 > >& v518
) {	// L682
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L692
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L693
  #pragma HLS array_partition variable=twi complete
  float in_re[32][8];	// L694
  #pragma HLS array_partition variable=in_re complete dim=1

  #pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L695
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L696
  #pragma HLS array_partition variable=out_re_b complete dim=1

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L697
  #pragma HLS array_partition variable=out_im_b complete dim=1

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L698
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v526 = v515.read();
    hls::vector< float, 32 > v527 = v516.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L701
    #pragma HLS unroll
      int32_t v529 = i;	// L702
      int32_t v530 = v529 & 1;	// L703
      int32_t v531 = v530 << 4;	// L704
      int32_t v532 = k5;	// L705
      int32_t v533 = v532 ^ v531;	// L706
      int32_t bank1;	// L707
      bank1 = v533;	// L708
      float v535 = v526[k5];	// L709
      int32_t v536 = bank1;	// L710
      int v537 = v536;	// L711
      in_re[v537][i] = v535;	// L712
      float v538 = v527[k5];	// L713
      int32_t v539 = bank1;	// L714
      int v540 = v539;	// L715
      in_im[v540][i] = v538;	// L716
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L719
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L720
    #pragma HLS unroll
      int v543 = i1 << 4;	// L721
      int v544 = v543 | k6;	// L722
      int32_t v545 = v544;	// L723
      int32_t bg;	// L724
      bg = v545;	// L725
      int32_t v547 = bg;	// L726
      int32_t v548 = v547 >> 5;	// L727
      int32_t grp;	// L728
      grp = v548;	// L729
      int32_t v550 = bg;	// L730
      int32_t v551 = v550 & 31;	// L731
      int32_t within;	// L732
      within = v551;	// L733
      int32_t v553 = within;	// L734
      int32_t v554 = v553 << 2;	// L735
      int32_t tw_k3;	// L736
      tw_k3 = v554;	// L737
      int32_t v556 = grp;	// L738
      int32_t v557 = v556 << 1;	// L739
      int32_t off_l;	// L740
      off_l = v557;	// L741
      int32_t v559 = off_l;	// L742
      int32_t v560 = v559 | 1;	// L743
      int32_t off_u;	// L744
      off_u = v560;	// L745
      int32_t v562 = within;	// L746
      int v563 = v562;	// L747
      int32_t v564 = off_l;	// L748
      int v565 = v564;	// L749
      float v566 = in_re[v563][v565];	// L750
      float a_re5;	// L751
      a_re5 = v566;	// L752
      int32_t v568 = within;	// L753
      int v569 = v568;	// L754
      int32_t v570 = off_l;	// L755
      int v571 = v570;	// L756
      float v572 = in_im[v569][v571];	// L757
      float a_im5;	// L758
      a_im5 = v572;	// L759
      int32_t v574 = within;	// L760
      int32_t v575 = v574 ^ 16;	// L761
      int v576 = v575;	// L762
      int32_t v577 = off_u;	// L763
      int v578 = v577;	// L764
      float v579 = in_re[v576][v578];	// L765
      float b_re5;	// L766
      b_re5 = v579;	// L767
      int32_t v581 = within;	// L768
      int32_t v582 = v581 ^ 16;	// L769
      int v583 = v582;	// L770
      int32_t v584 = off_u;	// L771
      int v585 = v584;	// L772
      float v586 = in_im[v583][v585];	// L773
      float b_im5;	// L774
      b_im5 = v586;	// L775
      int32_t v588 = tw_k3;	// L776
      bool v589 = v588 == 0;	// L777
      if (v589) {	// L778
        float v590 = a_re5;	// L779
        float v591 = b_re5;	// L780
        float v592 = v590 + v591;	// L781
        #pragma HLS bind_op variable=v592 op=fadd impl=fabric
        int32_t v593 = within;	// L782
        int v594 = v593;	// L783
        int32_t v595 = off_l;	// L784
        int v596 = v595;	// L785
        out_re_b[v594][v596] = v592;	// L786
        float v597 = a_im5;	// L787
        float v598 = b_im5;	// L788
        float v599 = v597 + v598;	// L789
        #pragma HLS bind_op variable=v599 op=fadd impl=fabric
        int32_t v600 = within;	// L790
        int v601 = v600;	// L791
        int32_t v602 = off_l;	// L792
        int v603 = v602;	// L793
        out_im_b[v601][v603] = v599;	// L794
        float v604 = a_re5;	// L795
        float v605 = b_re5;	// L796
        float v606 = v604 - v605;	// L797
        #pragma HLS bind_op variable=v606 op=fsub impl=fabric
        int32_t v607 = within;	// L798
        int32_t v608 = v607 ^ 16;	// L799
        int v609 = v608;	// L800
        int32_t v610 = off_u;	// L801
        int v611 = v610;	// L802
        out_re_b[v609][v611] = v606;	// L803
        float v612 = a_im5;	// L804
        float v613 = b_im5;	// L805
        float v614 = v612 - v613;	// L806
        #pragma HLS bind_op variable=v614 op=fsub impl=fabric
        int32_t v615 = within;	// L807
        int32_t v616 = v615 ^ 16;	// L808
        int v617 = v616;	// L809
        int32_t v618 = off_u;	// L810
        int v619 = v618;	// L811
        out_im_b[v617][v619] = v614;	// L812
      } else {
        int32_t v620 = tw_k3;	// L814
        bool v621 = v620 == 64;	// L815
        if (v621) {	// L816
          float v622 = a_re5;	// L817
          float v623 = b_im5;	// L818
          float v624 = v622 + v623;	// L819
          #pragma HLS bind_op variable=v624 op=fadd impl=fabric
          int32_t v625 = within;	// L820
          int v626 = v625;	// L821
          int32_t v627 = off_l;	// L822
          int v628 = v627;	// L823
          out_re_b[v626][v628] = v624;	// L824
          float v629 = a_im5;	// L825
          float v630 = b_re5;	// L826
          float v631 = v629 - v630;	// L827
          #pragma HLS bind_op variable=v631 op=fsub impl=fabric
          int32_t v632 = within;	// L828
          int v633 = v632;	// L829
          int32_t v634 = off_l;	// L830
          int v635 = v634;	// L831
          out_im_b[v633][v635] = v631;	// L832
          float v636 = a_re5;	// L833
          float v637 = b_im5;	// L834
          float v638 = v636 - v637;	// L835
          #pragma HLS bind_op variable=v638 op=fsub impl=fabric
          int32_t v639 = within;	// L836
          int32_t v640 = v639 ^ 16;	// L837
          int v641 = v640;	// L838
          int32_t v642 = off_u;	// L839
          int v643 = v642;	// L840
          out_re_b[v641][v643] = v638;	// L841
          float v644 = a_im5;	// L842
          float v645 = b_re5;	// L843
          float v646 = v644 + v645;	// L844
          #pragma HLS bind_op variable=v646 op=fadd impl=fabric
          int32_t v647 = within;	// L845
          int32_t v648 = v647 ^ 16;	// L846
          int v649 = v648;	// L847
          int32_t v650 = off_u;	// L848
          int v651 = v650;	// L849
          out_im_b[v649][v651] = v646;	// L850
        } else {
          int32_t v652 = tw_k3;	// L852
          int v653 = v652;	// L853
          float v654 = twr[v653];	// L854
          float tr3;	// L855
          tr3 = v654;	// L856
          int32_t v656 = tw_k3;	// L857
          int v657 = v656;	// L858
          float v658 = twi[v657];	// L859
          float ti3;	// L860
          ti3 = v658;	// L861
          float v660 = b_re5;	// L862
          float v661 = tr3;	// L863
          float v662 = v660 * v661;	// L864
          float v663 = b_im5;	// L865
          float v664 = ti3;	// L866
          float v665 = v663 * v664;	// L867
          float v666 = v662 - v665;	// L868
          float bw_re3;	// L869
          bw_re3 = v666;	// L870
          float v668 = b_re5;	// L871
          float v669 = ti3;	// L872
          float v670 = v668 * v669;	// L873
          float v671 = b_im5;	// L874
          float v672 = tr3;	// L875
          float v673 = v671 * v672;	// L876
          float v674 = v670 + v673;	// L877
          float bw_im3;	// L878
          bw_im3 = v674;	// L879
          float v676 = a_re5;	// L880
          float v677 = bw_re3;	// L881
          float v678 = v676 + v677;	// L882
          #pragma HLS bind_op variable=v678 op=fadd impl=fabric
          int32_t v679 = within;	// L883
          int v680 = v679;	// L884
          int32_t v681 = off_l;	// L885
          int v682 = v681;	// L886
          out_re_b[v680][v682] = v678;	// L887
          float v683 = a_im5;	// L888
          float v684 = bw_im3;	// L889
          float v685 = v683 + v684;	// L890
          #pragma HLS bind_op variable=v685 op=fadd impl=fabric
          int32_t v686 = within;	// L891
          int v687 = v686;	// L892
          int32_t v688 = off_l;	// L893
          int v689 = v688;	// L894
          out_im_b[v687][v689] = v685;	// L895
          float v690 = a_re5;	// L896
          float v691 = bw_re3;	// L897
          float v692 = v690 - v691;	// L898
          #pragma HLS bind_op variable=v692 op=fsub impl=fabric
          int32_t v693 = within;	// L899
          int32_t v694 = v693 ^ 16;	// L900
          int v695 = v694;	// L901
          int32_t v696 = off_u;	// L902
          int v697 = v696;	// L903
          out_re_b[v695][v697] = v692;	// L904
          float v698 = a_im5;	// L905
          float v699 = bw_im3;	// L906
          float v700 = v698 - v699;	// L907
          #pragma HLS bind_op variable=v700 op=fsub impl=fabric
          int32_t v701 = within;	// L908
          int32_t v702 = v701 ^ 16;	// L909
          int v703 = v702;	// L910
          int32_t v704 = off_u;	// L911
          int v705 = v704;	// L912
          out_im_b[v703][v705] = v700;	// L913
        }
      }
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L918
  #pragma HLS pipeline II=1
    float chunk_re1[32];	// L919
    #pragma HLS array_partition variable=chunk_re1 complete
    float chunk_im1[32];	// L920
    #pragma HLS array_partition variable=chunk_im1 complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L921
    #pragma HLS unroll
      int32_t v710 = i2;	// L922
      int32_t v711 = v710 & 1;	// L923
      int32_t v712 = v711 << 4;	// L924
      int32_t v713 = k7;	// L925
      int32_t v714 = v713 ^ v712;	// L926
      int32_t bank2;	// L927
      bank2 = v714;	// L928
      int32_t v716 = bank2;	// L929
      int v717 = v716;	// L930
      float v718 = out_re_b[v717][i2];	// L931
      chunk_re1[k7] = v718;	// L932
      int32_t v719 = bank2;	// L933
      int v720 = v719;	// L934
      float v721 = out_im_b[v720][i2];	// L935
      chunk_im1[k7] = v721;	// L936
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re1[_iv0];
      }
      v517.write(_vec);
    }	// L938
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im1[_iv0];
      }
      v518.write(_vec);
    }	// L939
  }
}

void inter_6_0(
  hls::stream< hls::vector< float, 32 > >& v722,
  hls::stream< hls::vector< float, 32 > >& v723,
  hls::stream< hls::vector< float, 32 > >& v724,
  hls::stream< hls::vector< float, 32 > >& v725
) {	// L943
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L952
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L953
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L954
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L955
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L956
  #pragma HLS array_partition variable=out_re_b1 complete dim=1

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L957
  #pragma HLS array_partition variable=out_im_b1 complete dim=1

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L958
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v733 = v722.read();
    hls::vector< float, 32 > v734 = v723.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L961
    #pragma HLS unroll
      int v736 = i3 >> 1;	// L962
      int32_t v737 = v736;	// L963
      int32_t v738 = v737 & 1;	// L964
      int32_t v739 = v738 << 4;	// L965
      int32_t v740 = k8;	// L966
      int32_t v741 = v740 ^ v739;	// L967
      int32_t bank3;	// L968
      bank3 = v741;	// L969
      float v743 = v733[k8];	// L970
      int32_t v744 = bank3;	// L971
      int v745 = v744;	// L972
      in_re1[v745][i3] = v743;	// L973
      float v746 = v734[k8];	// L974
      int32_t v747 = bank3;	// L975
      int v748 = v747;	// L976
      in_im1[v748][i3] = v746;	// L977
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L980
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L981
    #pragma HLS unroll
      int v751 = i4 << 4;	// L982
      int v752 = v751 | k9;	// L983
      uint32_t v753 = v752;	// L984
      uint32_t bg1;	// L985
      bg1 = v753;	// L986
      int32_t v755 = bg1;	// L987
      int32_t v756 = v755 & 63;	// L988
      int32_t v757 = v756 << 1;	// L989
      uint32_t tw_k4;	// L990
      tw_k4 = v757;	// L991
      int32_t v759 = i4;	// L992
      int32_t v760 = v759 & 1;	// L993
      int32_t v761 = v760 << 4;	// L994
      int32_t v762 = k9;	// L995
      int32_t v763 = v762 | v761;	// L996
      uint32_t bank_il;	// L997
      bank_il = v763;	// L998
      int32_t v765 = bank_il;	// L999
      int32_t v766 = v765 ^ 16;	// L1000
      uint32_t bank_iu;	// L1001
      bank_iu = v766;	// L1002
      int v768 = i4 >> 2;	// L1003
      int v769 = v768 << 2;	// L1004
      int v770 = i4 >> 1;	// L1005
      int32_t v771 = v770;	// L1006
      int32_t v772 = v771 & 1;	// L1007
      int32_t v773 = v769;	// L1008
      int32_t v774 = v773 | v772;	// L1009
      uint32_t off_il;	// L1010
      off_il = v774;	// L1011
      int32_t v776 = off_il;	// L1012
      int32_t v777 = v776 | 2;	// L1013
      uint32_t off_iu;	// L1014
      off_iu = v777;	// L1015
      int32_t v779 = bank_il;	// L1016
      int v780 = v779;	// L1017
      int32_t v781 = off_il;	// L1018
      int v782 = v781;	// L1019
      float v783 = in_re1[v780][v782];	// L1020
      float a_re6;	// L1021
      a_re6 = v783;	// L1022
      int32_t v785 = bank_il;	// L1023
      int v786 = v785;	// L1024
      int32_t v787 = off_il;	// L1025
      int v788 = v787;	// L1026
      float v789 = in_im1[v786][v788];	// L1027
      float a_im6;	// L1028
      a_im6 = v789;	// L1029
      int32_t v791 = bank_iu;	// L1030
      int v792 = v791;	// L1031
      int32_t v793 = off_iu;	// L1032
      int v794 = v793;	// L1033
      float v795 = in_re1[v792][v794];	// L1034
      float b_re6;	// L1035
      b_re6 = v795;	// L1036
      int32_t v797 = bank_iu;	// L1037
      int v798 = v797;	// L1038
      int32_t v799 = off_iu;	// L1039
      int v800 = v799;	// L1040
      float v801 = in_im1[v798][v800];	// L1041
      float b_im6;	// L1042
      b_im6 = v801;	// L1043
      int32_t v803 = tw_k4;	// L1044
      int v804 = v803;	// L1045
      float v805 = twr[v804];	// L1046
      float tr4;	// L1047
      tr4 = v805;	// L1048
      int32_t v807 = tw_k4;	// L1049
      int v808 = v807;	// L1050
      float v809 = twi[v808];	// L1051
      float ti4;	// L1052
      ti4 = v809;	// L1053
      float v811 = b_re6;	// L1054
      float v812 = tr4;	// L1055
      float v813 = v811 * v812;	// L1056
      float v814 = b_im6;	// L1057
      float v815 = ti4;	// L1058
      float v816 = v814 * v815;	// L1059
      float v817 = v813 - v816;	// L1060
      float bw_re4;	// L1061
      bw_re4 = v817;	// L1062
      float v819 = b_re6;	// L1063
      float v820 = ti4;	// L1064
      float v821 = v819 * v820;	// L1065
      float v822 = b_im6;	// L1066
      float v823 = tr4;	// L1067
      float v824 = v822 * v823;	// L1068
      float v825 = v821 + v824;	// L1069
      float bw_im4;	// L1070
      bw_im4 = v825;	// L1071
      float v827 = a_re6;	// L1072
      float v828 = bw_re4;	// L1073
      float v829 = v827 + v828;	// L1074
      #pragma HLS bind_op variable=v829 op=fadd impl=fabric
      int32_t v830 = bank_il;	// L1075
      int v831 = v830;	// L1076
      int32_t v832 = off_il;	// L1077
      int v833 = v832;	// L1078
      out_re_b1[v831][v833] = v829;	// L1079
      float v834 = a_im6;	// L1080
      float v835 = bw_im4;	// L1081
      float v836 = v834 + v835;	// L1082
      #pragma HLS bind_op variable=v836 op=fadd impl=fabric
      int32_t v837 = bank_il;	// L1083
      int v838 = v837;	// L1084
      int32_t v839 = off_il;	// L1085
      int v840 = v839;	// L1086
      out_im_b1[v838][v840] = v836;	// L1087
      float v841 = a_re6;	// L1088
      float v842 = bw_re4;	// L1089
      float v843 = v841 - v842;	// L1090
      #pragma HLS bind_op variable=v843 op=fsub impl=fabric
      int32_t v844 = bank_iu;	// L1091
      int v845 = v844;	// L1092
      int32_t v846 = off_iu;	// L1093
      int v847 = v846;	// L1094
      out_re_b1[v845][v847] = v843;	// L1095
      float v848 = a_im6;	// L1096
      float v849 = bw_im4;	// L1097
      float v850 = v848 - v849;	// L1098
      #pragma HLS bind_op variable=v850 op=fsub impl=fabric
      int32_t v851 = bank_iu;	// L1099
      int v852 = v851;	// L1100
      int32_t v853 = off_iu;	// L1101
      int v854 = v853;	// L1102
      out_im_b1[v852][v854] = v850;	// L1103
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L1106
  #pragma HLS pipeline II=1
    float chunk_re2[32];	// L1107
    #pragma HLS array_partition variable=chunk_re2 complete
    float chunk_im2[32];	// L1108
    #pragma HLS array_partition variable=chunk_im2 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L1109
    #pragma HLS unroll
      int v859 = i5 >> 1;	// L1110
      int32_t v860 = v859;	// L1111
      int32_t v861 = v860 & 1;	// L1112
      int32_t v862 = v861 << 4;	// L1113
      int32_t v863 = k10;	// L1114
      int32_t v864 = v863 ^ v862;	// L1115
      int32_t bank4;	// L1116
      bank4 = v864;	// L1117
      int32_t v866 = bank4;	// L1118
      int v867 = v866;	// L1119
      float v868 = out_re_b1[v867][i5];	// L1120
      chunk_re2[k10] = v868;	// L1121
      int32_t v869 = bank4;	// L1122
      int v870 = v869;	// L1123
      float v871 = out_im_b1[v870][i5];	// L1124
      chunk_im2[k10] = v871;	// L1125
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re2[_iv0];
      }
      v724.write(_vec);
    }	// L1127
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im2[_iv0];
      }
      v725.write(_vec);
    }	// L1128
  }
}

void inter_7_0(
  hls::stream< hls::vector< float, 32 > >& v872,
  hls::stream< hls::vector< float, 32 > >& v873,
  hls::stream< hls::vector< float, 32 > >& v874,
  hls::stream< hls::vector< float, 32 > >& v875
) {	// L1132
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1139
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1140
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L1141
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L1142
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L1143
  #pragma HLS array_partition variable=out_re_b2 complete dim=1

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L1144
  #pragma HLS array_partition variable=out_im_b2 complete dim=1

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L1145
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v883 = v872.read();
    hls::vector< float, 32 > v884 = v873.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L1148
    #pragma HLS unroll
      int v886 = i6 >> 2;	// L1149
      int32_t v887 = v886;	// L1150
      int32_t v888 = v887 & 1;	// L1151
      int32_t v889 = v888 << 4;	// L1152
      int32_t v890 = k11;	// L1153
      int32_t v891 = v890 ^ v889;	// L1154
      int32_t bank5;	// L1155
      bank5 = v891;	// L1156
      float v893 = v883[k11];	// L1157
      int32_t v894 = bank5;	// L1158
      int v895 = v894;	// L1159
      in_re2[v895][i6] = v893;	// L1160
      float v896 = v884[k11];	// L1161
      int32_t v897 = bank5;	// L1162
      int v898 = v897;	// L1163
      in_im2[v898][i6] = v896;	// L1164
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L1167
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L1168
    #pragma HLS unroll
      int32_t v901 = i7;	// L1169
      int32_t v902 = v901 & 1;	// L1170
      int32_t v903 = v902 << 4;	// L1171
      int32_t v904 = k12;	// L1172
      int32_t v905 = v904 | v903;	// L1173
      uint32_t bank_il1;	// L1174
      bank_il1 = v905;	// L1175
      int32_t v907 = bank_il1;	// L1176
      int32_t v908 = v907 ^ 16;	// L1177
      uint32_t bank_iu1;	// L1178
      bank_iu1 = v908;	// L1179
      int v910 = i7 >> 1;	// L1180
      uint32_t v911 = v910;	// L1181
      uint32_t off_il1;	// L1182
      off_il1 = v911;	// L1183
      int32_t v913 = off_il1;	// L1184
      int32_t v914 = v913 | 4;	// L1185
      uint32_t off_iu1;	// L1186
      off_iu1 = v914;	// L1187
      int v916 = i7 << 4;	// L1188
      int v917 = v916 | k12;	// L1189
      uint32_t v918 = v917;	// L1190
      uint32_t tw_k5;	// L1191
      tw_k5 = v918;	// L1192
      int32_t v920 = bank_il1;	// L1193
      int v921 = v920;	// L1194
      int32_t v922 = off_il1;	// L1195
      int v923 = v922;	// L1196
      float v924 = in_re2[v921][v923];	// L1197
      float a_re7;	// L1198
      a_re7 = v924;	// L1199
      int32_t v926 = bank_il1;	// L1200
      int v927 = v926;	// L1201
      int32_t v928 = off_il1;	// L1202
      int v929 = v928;	// L1203
      float v930 = in_im2[v927][v929];	// L1204
      float a_im7;	// L1205
      a_im7 = v930;	// L1206
      int32_t v932 = bank_iu1;	// L1207
      int v933 = v932;	// L1208
      int32_t v934 = off_iu1;	// L1209
      int v935 = v934;	// L1210
      float v936 = in_re2[v933][v935];	// L1211
      float b_re7;	// L1212
      b_re7 = v936;	// L1213
      int32_t v938 = bank_iu1;	// L1214
      int v939 = v938;	// L1215
      int32_t v940 = off_iu1;	// L1216
      int v941 = v940;	// L1217
      float v942 = in_im2[v939][v941];	// L1218
      float b_im7;	// L1219
      b_im7 = v942;	// L1220
      int32_t v944 = tw_k5;	// L1221
      int v945 = v944;	// L1222
      float v946 = twr[v945];	// L1223
      float tr5;	// L1224
      tr5 = v946;	// L1225
      int32_t v948 = tw_k5;	// L1226
      int v949 = v948;	// L1227
      float v950 = twi[v949];	// L1228
      float ti5;	// L1229
      ti5 = v950;	// L1230
      float v952 = b_re7;	// L1231
      float v953 = tr5;	// L1232
      float v954 = v952 * v953;	// L1233
      float v955 = b_im7;	// L1234
      float v956 = ti5;	// L1235
      float v957 = v955 * v956;	// L1236
      float v958 = v954 - v957;	// L1237
      float bw_re5;	// L1238
      bw_re5 = v958;	// L1239
      float v960 = b_re7;	// L1240
      float v961 = ti5;	// L1241
      float v962 = v960 * v961;	// L1242
      float v963 = b_im7;	// L1243
      float v964 = tr5;	// L1244
      float v965 = v963 * v964;	// L1245
      float v966 = v962 + v965;	// L1246
      float bw_im5;	// L1247
      bw_im5 = v966;	// L1248
      float v968 = a_re7;	// L1249
      float v969 = bw_re5;	// L1250
      float v970 = v968 + v969;	// L1251
      #pragma HLS bind_op variable=v970 op=fadd impl=fabric
      int32_t v971 = bank_il1;	// L1252
      int v972 = v971;	// L1253
      int32_t v973 = off_il1;	// L1254
      int v974 = v973;	// L1255
      out_re_b2[v972][v974] = v970;	// L1256
      float v975 = a_im7;	// L1257
      float v976 = bw_im5;	// L1258
      float v977 = v975 + v976;	// L1259
      #pragma HLS bind_op variable=v977 op=fadd impl=fabric
      int32_t v978 = bank_il1;	// L1260
      int v979 = v978;	// L1261
      int32_t v980 = off_il1;	// L1262
      int v981 = v980;	// L1263
      out_im_b2[v979][v981] = v977;	// L1264
      float v982 = a_re7;	// L1265
      float v983 = bw_re5;	// L1266
      float v984 = v982 - v983;	// L1267
      #pragma HLS bind_op variable=v984 op=fsub impl=fabric
      int32_t v985 = bank_iu1;	// L1268
      int v986 = v985;	// L1269
      int32_t v987 = off_iu1;	// L1270
      int v988 = v987;	// L1271
      out_re_b2[v986][v988] = v984;	// L1272
      float v989 = a_im7;	// L1273
      float v990 = bw_im5;	// L1274
      float v991 = v989 - v990;	// L1275
      #pragma HLS bind_op variable=v991 op=fsub impl=fabric
      int32_t v992 = bank_iu1;	// L1276
      int v993 = v992;	// L1277
      int32_t v994 = off_iu1;	// L1278
      int v995 = v994;	// L1279
      out_im_b2[v993][v995] = v991;	// L1280
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1283
  #pragma HLS pipeline II=1
    float chunk_re3[32];	// L1284
    #pragma HLS array_partition variable=chunk_re3 complete
    float chunk_im3[32];	// L1285
    #pragma HLS array_partition variable=chunk_im3 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1286
    #pragma HLS unroll
      int v1000 = i8 >> 2;	// L1287
      int32_t v1001 = v1000;	// L1288
      int32_t v1002 = v1001 & 1;	// L1289
      int32_t v1003 = v1002 << 4;	// L1290
      int32_t v1004 = k13;	// L1291
      int32_t v1005 = v1004 ^ v1003;	// L1292
      int32_t bank6;	// L1293
      bank6 = v1005;	// L1294
      int32_t v1007 = bank6;	// L1295
      int v1008 = v1007;	// L1296
      float v1009 = out_re_b2[v1008][i8];	// L1297
      chunk_re3[k13] = v1009;	// L1298
      int32_t v1010 = bank6;	// L1299
      int v1011 = v1010;	// L1300
      float v1012 = out_im_b2[v1011][i8];	// L1301
      chunk_im3[k13] = v1012;	// L1302
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re3[_iv0];
      }
      v874.write(_vec);
    }	// L1304
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im3[_iv0];
      }
      v875.write(_vec);
    }	// L1305
  }
}

void output_stage_0(
  float v1013[8][32],
  float v1014[8][32],
  hls::stream< hls::vector< float, 32 > >& v1015,
  hls::stream< hls::vector< float, 32 > >& v1016
) {	// L1309
  #pragma HLS array_partition variable=v1013 complete dim=2

  #pragma HLS array_partition variable=v1014 complete dim=2

  l_S_i_0_i9: for (int i9 = 0; i9 < 8; i9++) {	// L1310
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1018 = v1015.read();
    hls::vector< float, 32 > v1019 = v1016.read();
    l_S_k_0_k14: for (int k14 = 0; k14 < 32; k14++) {	// L1313
    #pragma HLS unroll
      float v1021 = v1018[k14];	// L1314
      v1013[i9][k14] = v1021;	// L1315
      float v1022 = v1019[k14];	// L1316
      v1014[i9][k14] = v1022;	// L1317
    }
  }
}

/// This is top function.
void fft_256(
  float v1023[8][32],
  float v1024[8][32],
  float v1025[8][32],
  float v1026[8][32]
) {	// L1322
  #pragma HLS dataflow disable_start_propagation
  #pragma HLS array_partition variable=v1023 complete dim=2

  #pragma HLS array_partition variable=v1024 complete dim=2

  #pragma HLS array_partition variable=v1025 complete dim=2

  #pragma HLS array_partition variable=v1026 complete dim=2

  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1027;
  #pragma HLS stream variable=v1027 depth=2	// L1323
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1028;
  #pragma HLS stream variable=v1028 depth=2	// L1324
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1029;
  #pragma HLS stream variable=v1029 depth=2	// L1325
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1030;
  #pragma HLS stream variable=v1030 depth=2	// L1326
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1031;
  #pragma HLS stream variable=v1031 depth=2	// L1327
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1032;
  #pragma HLS stream variable=v1032 depth=2	// L1328
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1033;
  #pragma HLS stream variable=v1033 depth=2	// L1329
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1034;
  #pragma HLS stream variable=v1034 depth=2	// L1330
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1035;
  #pragma HLS stream variable=v1035 depth=2	// L1331
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1036;
  #pragma HLS stream variable=v1036 depth=2	// L1332
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1037;
  #pragma HLS stream variable=v1037 depth=2	// L1333
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1038;
  #pragma HLS stream variable=v1038 depth=2	// L1334
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1039;
  #pragma HLS stream variable=v1039 depth=2	// L1335
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1040;
  #pragma HLS stream variable=v1040 depth=2	// L1336
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1041;
  #pragma HLS stream variable=v1041 depth=2	// L1337
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1042;
  #pragma HLS stream variable=v1042 depth=2	// L1338
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1043;
  #pragma HLS stream variable=v1043 depth=2	// L1339
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1044;
  #pragma HLS stream variable=v1044 depth=2	// L1340
  bit_rev_stage_0(v1023, v1024, v1027, v1036);	// L1341
  intra_0_0(v1027, v1036, v1028, v1037);	// L1342
  intra_1_0(v1028, v1037, v1029, v1038);	// L1343
  intra_2_0(v1029, v1038, v1030, v1039);	// L1344
  intra_3_0(v1030, v1039, v1031, v1040);	// L1345
  intra_4_0(v1031, v1040, v1032, v1041);	// L1346
  inter_5_0(v1032, v1041, v1033, v1042);	// L1347
  inter_6_0(v1033, v1042, v1034, v1043);	// L1348
  inter_7_0(v1034, v1043, v1035, v1044);	// L1349
  output_stage_0(v1025, v1026, v1035, v1044);	// L1350
}


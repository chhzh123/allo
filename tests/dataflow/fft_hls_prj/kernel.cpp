
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
) {	// L4
  #pragma HLS dataflow disable_start_propagation
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
    hls::vector< float, 32 > v7 = v0.read();
    hls::vector< float, 32 > v8 = v1.read();
    l_S_kk_0_kk: for (int kk = 0; kk < 32; kk++) {	// L18
    #pragma HLS unroll
      int32_t v10 = kk;	// L19
      int32_t v11 = v10 & 1;	// L20
      int32_t v12 = v11 << 4;	// L21
      int32_t v13 = v10 & 2;	// L22
      int32_t v14 = v13 << 2;	// L23
      int32_t v15 = v12 | v14;	// L24
      int32_t v16 = v10 & 4;	// L25
      int32_t v17 = v15 | v16;	// L26
      int32_t v18 = v10 & 8;	// L27
      int32_t v19 = v18 >> 2;	// L28
      int32_t v20 = v17 | v19;	// L29
      int32_t v21 = v10 & 16;	// L30
      int32_t v22 = v21 >> 4;	// L31
      int32_t v23 = v20 | v22;	// L32
      int32_t bank;	// L33
      bank = v23;	// L34
      int32_t v25 = ii;	// L35
      int32_t v26 = v25 & 4;	// L36
      int32_t v27 = v26 >> 2;	// L37
      int32_t v28 = v25 & 2;	// L38
      int32_t v29 = v27 | v28;	// L39
      int32_t v30 = v25 & 1;	// L40
      int32_t v31 = v30 << 2;	// L41
      int32_t v32 = v29 | v31;	// L42
      int32_t offset;	// L43
      offset = v32;	// L44
      float v34 = v7[kk];	// L45
      int32_t v35 = bank;	// L46
      int v36 = v35;	// L47
      int32_t v37 = offset;	// L48
      int v38 = v37;	// L49
      buf_re[v36][v38] = v34;	// L50
      float v39 = v8[kk];	// L51
      int32_t v40 = bank;	// L52
      int v41 = v40;	// L53
      int32_t v42 = offset;	// L54
      int v43 = v42;	// L55
      buf_im[v41][v43] = v39;	// L56
    }
  }
  l_S_jj_2_jj: for (int jj = 0; jj < 8; jj++) {	// L59
  #pragma HLS pipeline II=1
    float chunk_re[32];	// L60
    #pragma HLS array_partition variable=chunk_re complete
    float chunk_im[32];	// L61
    #pragma HLS array_partition variable=chunk_im complete
    l_S_mm_2_mm: for (int mm = 0; mm < 32; mm++) {	// L62
    #pragma HLS unroll
      int v48 = jj << 2;	// L63
      int v49 = mm >> 3;	// L64
      int v50 = v48 | v49;	// L65
      int32_t v51 = v50;	// L66
      int32_t rd_bank;	// L67
      rd_bank = v51;	// L68
      int32_t v53 = mm;	// L69
      int32_t v54 = v53 & 7;	// L70
      int32_t rd_off;	// L71
      rd_off = v54;	// L72
      int32_t v56 = rd_bank;	// L73
      int v57 = v56;	// L74
      int32_t v58 = rd_off;	// L75
      int v59 = v58;	// L76
      float v60 = buf_re[v57][v59];	// L77
      chunk_re[mm] = v60;	// L78
      int32_t v61 = rd_bank;	// L79
      int v62 = v61;	// L80
      int32_t v63 = rd_off;	// L81
      int v64 = v63;	// L82
      float v65 = buf_im[v62][v64];	// L83
      chunk_im[mm] = v65;	// L84
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re[_iv0];
      }
      v2.write(_vec);
    }	// L86
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im[_iv0];
      }
      v3.write(_vec);
    }	// L87
  }
}

const float twr[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L91
const float twi[128] = {0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L92
void intra_0(
  hls::stream< hls::vector< float, 32 > >& v66,
  hls::stream< hls::vector< float, 32 > >& v67,
  hls::stream< hls::vector< float, 32 > >& v68,
  hls::stream< hls::vector< float, 32 > >& v69
) {	// L93
  // placeholder for const float twr	// L100
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L101
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i: for (int _i = 0; _i < 8; _i++) {	// L102
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v73 = v66.read();
    hls::vector< float, 32 > v74 = v67.read();
    float o_re[32];	// L105
    #pragma HLS array_partition variable=o_re complete
    float o_im[32];	// L106
    #pragma HLS array_partition variable=o_im complete
    int32_t stride;	// L107
    stride = 1;	// L108
    l_S_k_0_k: for (int k = 0; k < 16; k++) {	// L109
    #pragma HLS unroll
      int v79 = k << 1;	// L110
      int32_t v80 = stride;	// L111
      int64_t v81 = v80;	// L112
      int64_t v82 = v81 - 1;	// L113
      int64_t v83 = k;	// L114
      int64_t v84 = v83 & v82;	// L115
      int64_t v85 = v79;	// L116
      int64_t v86 = v85 | v84;	// L117
      int32_t v87 = v86;	// L118
      int32_t il;	// L119
      il = v87;	// L120
      int32_t v89 = il;	// L121
      int32_t v90 = stride;	// L122
      int32_t v91 = v89 | v90;	// L123
      int32_t iu;	// L124
      iu = v91;	// L125
      int32_t v93 = stride;	// L126
      int64_t v94 = v93;	// L127
      int64_t v95 = v94 - 1;	// L128
      int64_t v96 = v83 & v95;	// L129
      int64_t v97 = v96 << 7;	// L130
      int32_t v98 = v97;	// L131
      int32_t tw_k;	// L132
      tw_k = v98;	// L133
      int32_t v100 = il;	// L134
      int v101 = v100;	// L135
      float v102 = v73[v101];	// L136
      float a_re;	// L137
      a_re = v102;	// L138
      int32_t v104 = il;	// L139
      int v105 = v104;	// L140
      float v106 = v74[v105];	// L141
      float a_im;	// L142
      a_im = v106;	// L143
      int32_t v108 = iu;	// L144
      int v109 = v108;	// L145
      float v110 = v73[v109];	// L146
      float b_re;	// L147
      b_re = v110;	// L148
      int32_t v112 = iu;	// L149
      int v113 = v112;	// L150
      float v114 = v74[v113];	// L151
      float b_im;	// L152
      b_im = v114;	// L153
      int32_t v116 = tw_k;	// L154
      bool v117 = v116 == 0;	// L155
      if (v117) {	// L156
        float v118 = a_re;	// L157
        float v119 = b_re;	// L158
        float v120 = v118 + v119;	// L159
        #pragma HLS bind_op variable=v120 op=fadd impl=fabric
        int32_t v121 = il;	// L160
        int v122 = v121;	// L161
        o_re[v122] = v120;	// L162
        float v123 = a_im;	// L163
        float v124 = b_im;	// L164
        float v125 = v123 + v124;	// L165
        #pragma HLS bind_op variable=v125 op=fadd impl=fabric
        int32_t v126 = il;	// L166
        int v127 = v126;	// L167
        o_im[v127] = v125;	// L168
        float v128 = a_re;	// L169
        float v129 = b_re;	// L170
        float v130 = v128 - v129;	// L171
        #pragma HLS bind_op variable=v130 op=fsub impl=fabric
        int32_t v131 = iu;	// L172
        int v132 = v131;	// L173
        o_re[v132] = v130;	// L174
        float v133 = a_im;	// L175
        float v134 = b_im;	// L176
        float v135 = v133 - v134;	// L177
        #pragma HLS bind_op variable=v135 op=fsub impl=fabric
        int32_t v136 = iu;	// L178
        int v137 = v136;	// L179
        o_im[v137] = v135;	// L180
      } else {
        int32_t v138 = tw_k;	// L182
        bool v139 = v138 == 64;	// L183
        if (v139) {	// L184
          float v140 = a_re;	// L185
          float v141 = b_im;	// L186
          float v142 = v140 + v141;	// L187
          #pragma HLS bind_op variable=v142 op=fadd impl=fabric
          int32_t v143 = il;	// L188
          int v144 = v143;	// L189
          o_re[v144] = v142;	// L190
          float v145 = a_im;	// L191
          float v146 = b_re;	// L192
          float v147 = v145 - v146;	// L193
          #pragma HLS bind_op variable=v147 op=fsub impl=fabric
          int32_t v148 = il;	// L194
          int v149 = v148;	// L195
          o_im[v149] = v147;	// L196
          float v150 = a_re;	// L197
          float v151 = b_im;	// L198
          float v152 = v150 - v151;	// L199
          #pragma HLS bind_op variable=v152 op=fsub impl=fabric
          int32_t v153 = iu;	// L200
          int v154 = v153;	// L201
          o_re[v154] = v152;	// L202
          float v155 = a_im;	// L203
          float v156 = b_re;	// L204
          float v157 = v155 + v156;	// L205
          #pragma HLS bind_op variable=v157 op=fadd impl=fabric
          int32_t v158 = iu;	// L206
          int v159 = v158;	// L207
          o_im[v159] = v157;	// L208
        } else {
          int32_t v160 = tw_k;	// L210
          int v161 = v160;	// L211
          float v162 = twr[v161];	// L212
          float tr;	// L213
          tr = v162;	// L214
          int32_t v164 = tw_k;	// L215
          int v165 = v164;	// L216
          float v166 = twi[v165];	// L217
          float ti;	// L218
          ti = v166;	// L219
          float v168 = b_re;	// L220
          float v169 = tr;	// L221
          float v170 = v168 * v169;	// L222
          float v171 = b_im;	// L223
          float v172 = ti;	// L224
          float v173 = v171 * v172;	// L225
          float v174 = v170 - v173;	// L226
          float bw_re;	// L227
          bw_re = v174;	// L228
          float v176 = b_re;	// L229
          float v177 = ti;	// L230
          float v178 = v176 * v177;	// L231
          float v179 = b_im;	// L232
          float v180 = tr;	// L233
          float v181 = v179 * v180;	// L234
          float v182 = v178 + v181;	// L235
          float bw_im;	// L236
          bw_im = v182;	// L237
          float v184 = a_re;	// L238
          float v185 = bw_re;	// L239
          float v186 = v184 + v185;	// L240
          #pragma HLS bind_op variable=v186 op=fadd impl=fabric
          int32_t v187 = il;	// L241
          int v188 = v187;	// L242
          o_re[v188] = v186;	// L243
          float v189 = a_im;	// L244
          float v190 = bw_im;	// L245
          float v191 = v189 + v190;	// L246
          #pragma HLS bind_op variable=v191 op=fadd impl=fabric
          int32_t v192 = il;	// L247
          int v193 = v192;	// L248
          o_im[v193] = v191;	// L249
          float v194 = a_re;	// L250
          float v195 = bw_re;	// L251
          float v196 = v194 - v195;	// L252
          #pragma HLS bind_op variable=v196 op=fsub impl=fabric
          int32_t v197 = iu;	// L253
          int v198 = v197;	// L254
          o_re[v198] = v196;	// L255
          float v199 = a_im;	// L256
          float v200 = bw_im;	// L257
          float v201 = v199 - v200;	// L258
          #pragma HLS bind_op variable=v201 op=fsub impl=fabric
          int32_t v202 = iu;	// L259
          int v203 = v202;	// L260
          o_im[v203] = v201;	// L261
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re[_iv0];
      }
      v68.write(_vec);
    }	// L265
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im[_iv0];
      }
      v69.write(_vec);
    }	// L266
  }
}

void intra_1(
  hls::stream< hls::vector< float, 32 > >& v204,
  hls::stream< hls::vector< float, 32 > >& v205,
  hls::stream< hls::vector< float, 32 > >& v206,
  hls::stream< hls::vector< float, 32 > >& v207
) {	// L270
  // placeholder for const float twr	// L278
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L279
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i1: for (int _i1 = 0; _i1 < 8; _i1++) {	// L280
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v211 = v204.read();
    hls::vector< float, 32 > v212 = v205.read();
    float o_re1[32];	// L283
    #pragma HLS array_partition variable=o_re1 complete
    float o_im1[32];	// L284
    #pragma HLS array_partition variable=o_im1 complete
    int32_t stride1;	// L285
    stride1 = 2;	// L286
    l_S_k_0_k1: for (int k1 = 0; k1 < 16; k1++) {	// L287
    #pragma HLS unroll
      int v217 = k1 >> 1;	// L288
      int v218 = v217 << 2;	// L289
      int32_t v219 = stride1;	// L290
      int64_t v220 = v219;	// L291
      int64_t v221 = v220 - 1;	// L292
      int64_t v222 = k1;	// L293
      int64_t v223 = v222 & v221;	// L294
      int64_t v224 = v218;	// L295
      int64_t v225 = v224 | v223;	// L296
      int32_t v226 = v225;	// L297
      int32_t il1;	// L298
      il1 = v226;	// L299
      int32_t v228 = il1;	// L300
      int32_t v229 = stride1;	// L301
      int32_t v230 = v228 | v229;	// L302
      int32_t iu1;	// L303
      iu1 = v230;	// L304
      int32_t v232 = stride1;	// L305
      int64_t v233 = v232;	// L306
      int64_t v234 = v233 - 1;	// L307
      int64_t v235 = v222 & v234;	// L308
      int64_t v236 = v235 << 6;	// L309
      int32_t v237 = v236;	// L310
      int32_t tw_k1;	// L311
      tw_k1 = v237;	// L312
      int32_t v239 = il1;	// L313
      int v240 = v239;	// L314
      float v241 = v211[v240];	// L315
      float a_re1;	// L316
      a_re1 = v241;	// L317
      int32_t v243 = il1;	// L318
      int v244 = v243;	// L319
      float v245 = v212[v244];	// L320
      float a_im1;	// L321
      a_im1 = v245;	// L322
      int32_t v247 = iu1;	// L323
      int v248 = v247;	// L324
      float v249 = v211[v248];	// L325
      float b_re1;	// L326
      b_re1 = v249;	// L327
      int32_t v251 = iu1;	// L328
      int v252 = v251;	// L329
      float v253 = v212[v252];	// L330
      float b_im1;	// L331
      b_im1 = v253;	// L332
      int32_t v255 = tw_k1;	// L333
      bool v256 = v255 == 0;	// L334
      if (v256) {	// L335
        float v257 = a_re1;	// L336
        float v258 = b_re1;	// L337
        float v259 = v257 + v258;	// L338
        #pragma HLS bind_op variable=v259 op=fadd impl=fabric
        int32_t v260 = il1;	// L339
        int v261 = v260;	// L340
        o_re1[v261] = v259;	// L341
        float v262 = a_im1;	// L342
        float v263 = b_im1;	// L343
        float v264 = v262 + v263;	// L344
        #pragma HLS bind_op variable=v264 op=fadd impl=fabric
        int32_t v265 = il1;	// L345
        int v266 = v265;	// L346
        o_im1[v266] = v264;	// L347
        float v267 = a_re1;	// L348
        float v268 = b_re1;	// L349
        float v269 = v267 - v268;	// L350
        #pragma HLS bind_op variable=v269 op=fsub impl=fabric
        int32_t v270 = iu1;	// L351
        int v271 = v270;	// L352
        o_re1[v271] = v269;	// L353
        float v272 = a_im1;	// L354
        float v273 = b_im1;	// L355
        float v274 = v272 - v273;	// L356
        #pragma HLS bind_op variable=v274 op=fsub impl=fabric
        int32_t v275 = iu1;	// L357
        int v276 = v275;	// L358
        o_im1[v276] = v274;	// L359
      } else {
        int32_t v277 = tw_k1;	// L361
        bool v278 = v277 == 64;	// L362
        if (v278) {	// L363
          float v279 = a_re1;	// L364
          float v280 = b_im1;	// L365
          float v281 = v279 + v280;	// L366
          #pragma HLS bind_op variable=v281 op=fadd impl=fabric
          int32_t v282 = il1;	// L367
          int v283 = v282;	// L368
          o_re1[v283] = v281;	// L369
          float v284 = a_im1;	// L370
          float v285 = b_re1;	// L371
          float v286 = v284 - v285;	// L372
          #pragma HLS bind_op variable=v286 op=fsub impl=fabric
          int32_t v287 = il1;	// L373
          int v288 = v287;	// L374
          o_im1[v288] = v286;	// L375
          float v289 = a_re1;	// L376
          float v290 = b_im1;	// L377
          float v291 = v289 - v290;	// L378
          #pragma HLS bind_op variable=v291 op=fsub impl=fabric
          int32_t v292 = iu1;	// L379
          int v293 = v292;	// L380
          o_re1[v293] = v291;	// L381
          float v294 = a_im1;	// L382
          float v295 = b_re1;	// L383
          float v296 = v294 + v295;	// L384
          #pragma HLS bind_op variable=v296 op=fadd impl=fabric
          int32_t v297 = iu1;	// L385
          int v298 = v297;	// L386
          o_im1[v298] = v296;	// L387
        } else {
          int32_t v299 = tw_k1;	// L389
          int v300 = v299;	// L390
          float v301 = twr[v300];	// L391
          float tr1;	// L392
          tr1 = v301;	// L393
          int32_t v303 = tw_k1;	// L394
          int v304 = v303;	// L395
          float v305 = twi[v304];	// L396
          float ti1;	// L397
          ti1 = v305;	// L398
          float v307 = b_re1;	// L399
          float v308 = tr1;	// L400
          float v309 = v307 * v308;	// L401
          float v310 = b_im1;	// L402
          float v311 = ti1;	// L403
          float v312 = v310 * v311;	// L404
          float v313 = v309 - v312;	// L405
          float bw_re1;	// L406
          bw_re1 = v313;	// L407
          float v315 = b_re1;	// L408
          float v316 = ti1;	// L409
          float v317 = v315 * v316;	// L410
          float v318 = b_im1;	// L411
          float v319 = tr1;	// L412
          float v320 = v318 * v319;	// L413
          float v321 = v317 + v320;	// L414
          float bw_im1;	// L415
          bw_im1 = v321;	// L416
          float v323 = a_re1;	// L417
          float v324 = bw_re1;	// L418
          float v325 = v323 + v324;	// L419
          #pragma HLS bind_op variable=v325 op=fadd impl=fabric
          int32_t v326 = il1;	// L420
          int v327 = v326;	// L421
          o_re1[v327] = v325;	// L422
          float v328 = a_im1;	// L423
          float v329 = bw_im1;	// L424
          float v330 = v328 + v329;	// L425
          #pragma HLS bind_op variable=v330 op=fadd impl=fabric
          int32_t v331 = il1;	// L426
          int v332 = v331;	// L427
          o_im1[v332] = v330;	// L428
          float v333 = a_re1;	// L429
          float v334 = bw_re1;	// L430
          float v335 = v333 - v334;	// L431
          #pragma HLS bind_op variable=v335 op=fsub impl=fabric
          int32_t v336 = iu1;	// L432
          int v337 = v336;	// L433
          o_re1[v337] = v335;	// L434
          float v338 = a_im1;	// L435
          float v339 = bw_im1;	// L436
          float v340 = v338 - v339;	// L437
          #pragma HLS bind_op variable=v340 op=fsub impl=fabric
          int32_t v341 = iu1;	// L438
          int v342 = v341;	// L439
          o_im1[v342] = v340;	// L440
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re1[_iv0];
      }
      v206.write(_vec);
    }	// L444
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im1[_iv0];
      }
      v207.write(_vec);
    }	// L445
  }
}

void intra_2(
  hls::stream< hls::vector< float, 32 > >& v343,
  hls::stream< hls::vector< float, 32 > >& v344,
  hls::stream< hls::vector< float, 32 > >& v345,
  hls::stream< hls::vector< float, 32 > >& v346
) {	// L449
  // placeholder for const float twr	// L457
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L458
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i2: for (int _i2 = 0; _i2 < 8; _i2++) {	// L459
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v350 = v343.read();
    hls::vector< float, 32 > v351 = v344.read();
    float o_re2[32];	// L462
    #pragma HLS array_partition variable=o_re2 complete
    float o_im2[32];	// L463
    #pragma HLS array_partition variable=o_im2 complete
    int32_t stride2;	// L464
    stride2 = 4;	// L465
    l_S_k_0_k2: for (int k2 = 0; k2 < 16; k2++) {	// L466
    #pragma HLS unroll
      int v356 = k2 >> 2;	// L467
      int v357 = v356 << 3;	// L468
      int32_t v358 = stride2;	// L469
      int64_t v359 = v358;	// L470
      int64_t v360 = v359 - 1;	// L471
      int64_t v361 = k2;	// L472
      int64_t v362 = v361 & v360;	// L473
      int64_t v363 = v357;	// L474
      int64_t v364 = v363 | v362;	// L475
      int32_t v365 = v364;	// L476
      int32_t il2;	// L477
      il2 = v365;	// L478
      int32_t v367 = il2;	// L479
      int32_t v368 = stride2;	// L480
      int32_t v369 = v367 | v368;	// L481
      int32_t iu2;	// L482
      iu2 = v369;	// L483
      int32_t v371 = stride2;	// L484
      int64_t v372 = v371;	// L485
      int64_t v373 = v372 - 1;	// L486
      int64_t v374 = v361 & v373;	// L487
      int64_t v375 = v374 << 5;	// L488
      int32_t v376 = v375;	// L489
      int32_t tw_k2;	// L490
      tw_k2 = v376;	// L491
      int32_t v378 = il2;	// L492
      int v379 = v378;	// L493
      float v380 = v350[v379];	// L494
      float a_re2;	// L495
      a_re2 = v380;	// L496
      int32_t v382 = il2;	// L497
      int v383 = v382;	// L498
      float v384 = v351[v383];	// L499
      float a_im2;	// L500
      a_im2 = v384;	// L501
      int32_t v386 = iu2;	// L502
      int v387 = v386;	// L503
      float v388 = v350[v387];	// L504
      float b_re2;	// L505
      b_re2 = v388;	// L506
      int32_t v390 = iu2;	// L507
      int v391 = v390;	// L508
      float v392 = v351[v391];	// L509
      float b_im2;	// L510
      b_im2 = v392;	// L511
      int32_t v394 = tw_k2;	// L512
      bool v395 = v394 == 0;	// L513
      if (v395) {	// L514
        float v396 = a_re2;	// L515
        float v397 = b_re2;	// L516
        float v398 = v396 + v397;	// L517
        #pragma HLS bind_op variable=v398 op=fadd impl=fabric
        int32_t v399 = il2;	// L518
        int v400 = v399;	// L519
        o_re2[v400] = v398;	// L520
        float v401 = a_im2;	// L521
        float v402 = b_im2;	// L522
        float v403 = v401 + v402;	// L523
        #pragma HLS bind_op variable=v403 op=fadd impl=fabric
        int32_t v404 = il2;	// L524
        int v405 = v404;	// L525
        o_im2[v405] = v403;	// L526
        float v406 = a_re2;	// L527
        float v407 = b_re2;	// L528
        float v408 = v406 - v407;	// L529
        #pragma HLS bind_op variable=v408 op=fsub impl=fabric
        int32_t v409 = iu2;	// L530
        int v410 = v409;	// L531
        o_re2[v410] = v408;	// L532
        float v411 = a_im2;	// L533
        float v412 = b_im2;	// L534
        float v413 = v411 - v412;	// L535
        #pragma HLS bind_op variable=v413 op=fsub impl=fabric
        int32_t v414 = iu2;	// L536
        int v415 = v414;	// L537
        o_im2[v415] = v413;	// L538
      } else {
        int32_t v416 = tw_k2;	// L540
        bool v417 = v416 == 64;	// L541
        if (v417) {	// L542
          float v418 = a_re2;	// L543
          float v419 = b_im2;	// L544
          float v420 = v418 + v419;	// L545
          #pragma HLS bind_op variable=v420 op=fadd impl=fabric
          int32_t v421 = il2;	// L546
          int v422 = v421;	// L547
          o_re2[v422] = v420;	// L548
          float v423 = a_im2;	// L549
          float v424 = b_re2;	// L550
          float v425 = v423 - v424;	// L551
          #pragma HLS bind_op variable=v425 op=fsub impl=fabric
          int32_t v426 = il2;	// L552
          int v427 = v426;	// L553
          o_im2[v427] = v425;	// L554
          float v428 = a_re2;	// L555
          float v429 = b_im2;	// L556
          float v430 = v428 - v429;	// L557
          #pragma HLS bind_op variable=v430 op=fsub impl=fabric
          int32_t v431 = iu2;	// L558
          int v432 = v431;	// L559
          o_re2[v432] = v430;	// L560
          float v433 = a_im2;	// L561
          float v434 = b_re2;	// L562
          float v435 = v433 + v434;	// L563
          #pragma HLS bind_op variable=v435 op=fadd impl=fabric
          int32_t v436 = iu2;	// L564
          int v437 = v436;	// L565
          o_im2[v437] = v435;	// L566
        } else {
          int32_t v438 = tw_k2;	// L568
          int v439 = v438;	// L569
          float v440 = twr[v439];	// L570
          float tr2;	// L571
          tr2 = v440;	// L572
          int32_t v442 = tw_k2;	// L573
          int v443 = v442;	// L574
          float v444 = twi[v443];	// L575
          float ti2;	// L576
          ti2 = v444;	// L577
          float v446 = b_re2;	// L578
          float v447 = tr2;	// L579
          float v448 = v446 * v447;	// L580
          float v449 = b_im2;	// L581
          float v450 = ti2;	// L582
          float v451 = v449 * v450;	// L583
          float v452 = v448 - v451;	// L584
          float bw_re2;	// L585
          bw_re2 = v452;	// L586
          float v454 = b_re2;	// L587
          float v455 = ti2;	// L588
          float v456 = v454 * v455;	// L589
          float v457 = b_im2;	// L590
          float v458 = tr2;	// L591
          float v459 = v457 * v458;	// L592
          float v460 = v456 + v459;	// L593
          float bw_im2;	// L594
          bw_im2 = v460;	// L595
          float v462 = a_re2;	// L596
          float v463 = bw_re2;	// L597
          float v464 = v462 + v463;	// L598
          #pragma HLS bind_op variable=v464 op=fadd impl=fabric
          int32_t v465 = il2;	// L599
          int v466 = v465;	// L600
          o_re2[v466] = v464;	// L601
          float v467 = a_im2;	// L602
          float v468 = bw_im2;	// L603
          float v469 = v467 + v468;	// L604
          #pragma HLS bind_op variable=v469 op=fadd impl=fabric
          int32_t v470 = il2;	// L605
          int v471 = v470;	// L606
          o_im2[v471] = v469;	// L607
          float v472 = a_re2;	// L608
          float v473 = bw_re2;	// L609
          float v474 = v472 - v473;	// L610
          #pragma HLS bind_op variable=v474 op=fsub impl=fabric
          int32_t v475 = iu2;	// L611
          int v476 = v475;	// L612
          o_re2[v476] = v474;	// L613
          float v477 = a_im2;	// L614
          float v478 = bw_im2;	// L615
          float v479 = v477 - v478;	// L616
          #pragma HLS bind_op variable=v479 op=fsub impl=fabric
          int32_t v480 = iu2;	// L617
          int v481 = v480;	// L618
          o_im2[v481] = v479;	// L619
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re2[_iv0];
      }
      v345.write(_vec);
    }	// L623
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im2[_iv0];
      }
      v346.write(_vec);
    }	// L624
  }
}

void intra_3(
  hls::stream< hls::vector< float, 32 > >& v482,
  hls::stream< hls::vector< float, 32 > >& v483,
  hls::stream< hls::vector< float, 32 > >& v484,
  hls::stream< hls::vector< float, 32 > >& v485
) {	// L628
  // placeholder for const float twr	// L636
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L637
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i3: for (int _i3 = 0; _i3 < 8; _i3++) {	// L638
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v489 = v482.read();
    hls::vector< float, 32 > v490 = v483.read();
    float o_re3[32];	// L641
    #pragma HLS array_partition variable=o_re3 complete
    float o_im3[32];	// L642
    #pragma HLS array_partition variable=o_im3 complete
    int32_t stride3;	// L643
    stride3 = 8;	// L644
    l_S_k_0_k3: for (int k3 = 0; k3 < 16; k3++) {	// L645
    #pragma HLS unroll
      int v495 = k3 >> 3;	// L646
      int v496 = v495 << 4;	// L647
      int32_t v497 = stride3;	// L648
      int64_t v498 = v497;	// L649
      int64_t v499 = v498 - 1;	// L650
      int64_t v500 = k3;	// L651
      int64_t v501 = v500 & v499;	// L652
      int64_t v502 = v496;	// L653
      int64_t v503 = v502 | v501;	// L654
      int32_t v504 = v503;	// L655
      int32_t il3;	// L656
      il3 = v504;	// L657
      int32_t v506 = il3;	// L658
      int32_t v507 = stride3;	// L659
      int32_t v508 = v506 | v507;	// L660
      int32_t iu3;	// L661
      iu3 = v508;	// L662
      int32_t v510 = stride3;	// L663
      int64_t v511 = v510;	// L664
      int64_t v512 = v511 - 1;	// L665
      int64_t v513 = v500 & v512;	// L666
      int64_t v514 = v513 << 4;	// L667
      int32_t v515 = v514;	// L668
      int32_t tw_k3;	// L669
      tw_k3 = v515;	// L670
      int32_t v517 = il3;	// L671
      int v518 = v517;	// L672
      float v519 = v489[v518];	// L673
      float a_re3;	// L674
      a_re3 = v519;	// L675
      int32_t v521 = il3;	// L676
      int v522 = v521;	// L677
      float v523 = v490[v522];	// L678
      float a_im3;	// L679
      a_im3 = v523;	// L680
      int32_t v525 = iu3;	// L681
      int v526 = v525;	// L682
      float v527 = v489[v526];	// L683
      float b_re3;	// L684
      b_re3 = v527;	// L685
      int32_t v529 = iu3;	// L686
      int v530 = v529;	// L687
      float v531 = v490[v530];	// L688
      float b_im3;	// L689
      b_im3 = v531;	// L690
      int32_t v533 = tw_k3;	// L691
      bool v534 = v533 == 0;	// L692
      if (v534) {	// L693
        float v535 = a_re3;	// L694
        float v536 = b_re3;	// L695
        float v537 = v535 + v536;	// L696
        #pragma HLS bind_op variable=v537 op=fadd impl=fabric
        int32_t v538 = il3;	// L697
        int v539 = v538;	// L698
        o_re3[v539] = v537;	// L699
        float v540 = a_im3;	// L700
        float v541 = b_im3;	// L701
        float v542 = v540 + v541;	// L702
        #pragma HLS bind_op variable=v542 op=fadd impl=fabric
        int32_t v543 = il3;	// L703
        int v544 = v543;	// L704
        o_im3[v544] = v542;	// L705
        float v545 = a_re3;	// L706
        float v546 = b_re3;	// L707
        float v547 = v545 - v546;	// L708
        #pragma HLS bind_op variable=v547 op=fsub impl=fabric
        int32_t v548 = iu3;	// L709
        int v549 = v548;	// L710
        o_re3[v549] = v547;	// L711
        float v550 = a_im3;	// L712
        float v551 = b_im3;	// L713
        float v552 = v550 - v551;	// L714
        #pragma HLS bind_op variable=v552 op=fsub impl=fabric
        int32_t v553 = iu3;	// L715
        int v554 = v553;	// L716
        o_im3[v554] = v552;	// L717
      } else {
        int32_t v555 = tw_k3;	// L719
        bool v556 = v555 == 64;	// L720
        if (v556) {	// L721
          float v557 = a_re3;	// L722
          float v558 = b_im3;	// L723
          float v559 = v557 + v558;	// L724
          #pragma HLS bind_op variable=v559 op=fadd impl=fabric
          int32_t v560 = il3;	// L725
          int v561 = v560;	// L726
          o_re3[v561] = v559;	// L727
          float v562 = a_im3;	// L728
          float v563 = b_re3;	// L729
          float v564 = v562 - v563;	// L730
          #pragma HLS bind_op variable=v564 op=fsub impl=fabric
          int32_t v565 = il3;	// L731
          int v566 = v565;	// L732
          o_im3[v566] = v564;	// L733
          float v567 = a_re3;	// L734
          float v568 = b_im3;	// L735
          float v569 = v567 - v568;	// L736
          #pragma HLS bind_op variable=v569 op=fsub impl=fabric
          int32_t v570 = iu3;	// L737
          int v571 = v570;	// L738
          o_re3[v571] = v569;	// L739
          float v572 = a_im3;	// L740
          float v573 = b_re3;	// L741
          float v574 = v572 + v573;	// L742
          #pragma HLS bind_op variable=v574 op=fadd impl=fabric
          int32_t v575 = iu3;	// L743
          int v576 = v575;	// L744
          o_im3[v576] = v574;	// L745
        } else {
          int32_t v577 = tw_k3;	// L747
          int v578 = v577;	// L748
          float v579 = twr[v578];	// L749
          float tr3;	// L750
          tr3 = v579;	// L751
          int32_t v581 = tw_k3;	// L752
          int v582 = v581;	// L753
          float v583 = twi[v582];	// L754
          float ti3;	// L755
          ti3 = v583;	// L756
          float v585 = b_re3;	// L757
          float v586 = tr3;	// L758
          float v587 = v585 * v586;	// L759
          float v588 = b_im3;	// L760
          float v589 = ti3;	// L761
          float v590 = v588 * v589;	// L762
          float v591 = v587 - v590;	// L763
          float bw_re3;	// L764
          bw_re3 = v591;	// L765
          float v593 = b_re3;	// L766
          float v594 = ti3;	// L767
          float v595 = v593 * v594;	// L768
          float v596 = b_im3;	// L769
          float v597 = tr3;	// L770
          float v598 = v596 * v597;	// L771
          float v599 = v595 + v598;	// L772
          float bw_im3;	// L773
          bw_im3 = v599;	// L774
          float v601 = a_re3;	// L775
          float v602 = bw_re3;	// L776
          float v603 = v601 + v602;	// L777
          #pragma HLS bind_op variable=v603 op=fadd impl=fabric
          int32_t v604 = il3;	// L778
          int v605 = v604;	// L779
          o_re3[v605] = v603;	// L780
          float v606 = a_im3;	// L781
          float v607 = bw_im3;	// L782
          float v608 = v606 + v607;	// L783
          #pragma HLS bind_op variable=v608 op=fadd impl=fabric
          int32_t v609 = il3;	// L784
          int v610 = v609;	// L785
          o_im3[v610] = v608;	// L786
          float v611 = a_re3;	// L787
          float v612 = bw_re3;	// L788
          float v613 = v611 - v612;	// L789
          #pragma HLS bind_op variable=v613 op=fsub impl=fabric
          int32_t v614 = iu3;	// L790
          int v615 = v614;	// L791
          o_re3[v615] = v613;	// L792
          float v616 = a_im3;	// L793
          float v617 = bw_im3;	// L794
          float v618 = v616 - v617;	// L795
          #pragma HLS bind_op variable=v618 op=fsub impl=fabric
          int32_t v619 = iu3;	// L796
          int v620 = v619;	// L797
          o_im3[v620] = v618;	// L798
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re3[_iv0];
      }
      v484.write(_vec);
    }	// L802
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im3[_iv0];
      }
      v485.write(_vec);
    }	// L803
  }
}

void intra_4(
  hls::stream< hls::vector< float, 32 > >& v621,
  hls::stream< hls::vector< float, 32 > >& v622,
  hls::stream< hls::vector< float, 32 > >& v623,
  hls::stream< hls::vector< float, 32 > >& v624
) {	// L807
  // placeholder for const float twr	// L815
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L816
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i4: for (int _i4 = 0; _i4 < 8; _i4++) {	// L817
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v628 = v621.read();
    hls::vector< float, 32 > v629 = v622.read();
    float o_re4[32];	// L820
    #pragma HLS array_partition variable=o_re4 complete
    float o_im4[32];	// L821
    #pragma HLS array_partition variable=o_im4 complete
    int32_t stride4;	// L822
    stride4 = 16;	// L823
    l_S_k_0_k4: for (int k4 = 0; k4 < 16; k4++) {	// L824
    #pragma HLS unroll
      int v634 = k4 >> 4;	// L825
      int v635 = v634 << 5;	// L826
      int32_t v636 = stride4;	// L827
      int64_t v637 = v636;	// L828
      int64_t v638 = v637 - 1;	// L829
      int64_t v639 = k4;	// L830
      int64_t v640 = v639 & v638;	// L831
      int64_t v641 = v635;	// L832
      int64_t v642 = v641 | v640;	// L833
      int32_t v643 = v642;	// L834
      int32_t il4;	// L835
      il4 = v643;	// L836
      int32_t v645 = il4;	// L837
      int32_t v646 = stride4;	// L838
      int32_t v647 = v645 | v646;	// L839
      int32_t iu4;	// L840
      iu4 = v647;	// L841
      int32_t v649 = stride4;	// L842
      int64_t v650 = v649;	// L843
      int64_t v651 = v650 - 1;	// L844
      int64_t v652 = v639 & v651;	// L845
      int64_t v653 = v652 << 3;	// L846
      int32_t v654 = v653;	// L847
      int32_t tw_k4;	// L848
      tw_k4 = v654;	// L849
      int32_t v656 = il4;	// L850
      int v657 = v656;	// L851
      float v658 = v628[v657];	// L852
      float a_re4;	// L853
      a_re4 = v658;	// L854
      int32_t v660 = il4;	// L855
      int v661 = v660;	// L856
      float v662 = v629[v661];	// L857
      float a_im4;	// L858
      a_im4 = v662;	// L859
      int32_t v664 = iu4;	// L860
      int v665 = v664;	// L861
      float v666 = v628[v665];	// L862
      float b_re4;	// L863
      b_re4 = v666;	// L864
      int32_t v668 = iu4;	// L865
      int v669 = v668;	// L866
      float v670 = v629[v669];	// L867
      float b_im4;	// L868
      b_im4 = v670;	// L869
      int32_t v672 = tw_k4;	// L870
      bool v673 = v672 == 0;	// L871
      if (v673) {	// L872
        float v674 = a_re4;	// L873
        float v675 = b_re4;	// L874
        float v676 = v674 + v675;	// L875
        #pragma HLS bind_op variable=v676 op=fadd impl=fabric
        int32_t v677 = il4;	// L876
        int v678 = v677;	// L877
        o_re4[v678] = v676;	// L878
        float v679 = a_im4;	// L879
        float v680 = b_im4;	// L880
        float v681 = v679 + v680;	// L881
        #pragma HLS bind_op variable=v681 op=fadd impl=fabric
        int32_t v682 = il4;	// L882
        int v683 = v682;	// L883
        o_im4[v683] = v681;	// L884
        float v684 = a_re4;	// L885
        float v685 = b_re4;	// L886
        float v686 = v684 - v685;	// L887
        #pragma HLS bind_op variable=v686 op=fsub impl=fabric
        int32_t v687 = iu4;	// L888
        int v688 = v687;	// L889
        o_re4[v688] = v686;	// L890
        float v689 = a_im4;	// L891
        float v690 = b_im4;	// L892
        float v691 = v689 - v690;	// L893
        #pragma HLS bind_op variable=v691 op=fsub impl=fabric
        int32_t v692 = iu4;	// L894
        int v693 = v692;	// L895
        o_im4[v693] = v691;	// L896
      } else {
        int32_t v694 = tw_k4;	// L898
        bool v695 = v694 == 64;	// L899
        if (v695) {	// L900
          float v696 = a_re4;	// L901
          float v697 = b_im4;	// L902
          float v698 = v696 + v697;	// L903
          #pragma HLS bind_op variable=v698 op=fadd impl=fabric
          int32_t v699 = il4;	// L904
          int v700 = v699;	// L905
          o_re4[v700] = v698;	// L906
          float v701 = a_im4;	// L907
          float v702 = b_re4;	// L908
          float v703 = v701 - v702;	// L909
          #pragma HLS bind_op variable=v703 op=fsub impl=fabric
          int32_t v704 = il4;	// L910
          int v705 = v704;	// L911
          o_im4[v705] = v703;	// L912
          float v706 = a_re4;	// L913
          float v707 = b_im4;	// L914
          float v708 = v706 - v707;	// L915
          #pragma HLS bind_op variable=v708 op=fsub impl=fabric
          int32_t v709 = iu4;	// L916
          int v710 = v709;	// L917
          o_re4[v710] = v708;	// L918
          float v711 = a_im4;	// L919
          float v712 = b_re4;	// L920
          float v713 = v711 + v712;	// L921
          #pragma HLS bind_op variable=v713 op=fadd impl=fabric
          int32_t v714 = iu4;	// L922
          int v715 = v714;	// L923
          o_im4[v715] = v713;	// L924
        } else {
          int32_t v716 = tw_k4;	// L926
          int v717 = v716;	// L927
          float v718 = twr[v717];	// L928
          float tr4;	// L929
          tr4 = v718;	// L930
          int32_t v720 = tw_k4;	// L931
          int v721 = v720;	// L932
          float v722 = twi[v721];	// L933
          float ti4;	// L934
          ti4 = v722;	// L935
          float v724 = b_re4;	// L936
          float v725 = tr4;	// L937
          float v726 = v724 * v725;	// L938
          float v727 = b_im4;	// L939
          float v728 = ti4;	// L940
          float v729 = v727 * v728;	// L941
          float v730 = v726 - v729;	// L942
          float bw_re4;	// L943
          bw_re4 = v730;	// L944
          float v732 = b_re4;	// L945
          float v733 = ti4;	// L946
          float v734 = v732 * v733;	// L947
          float v735 = b_im4;	// L948
          float v736 = tr4;	// L949
          float v737 = v735 * v736;	// L950
          float v738 = v734 + v737;	// L951
          float bw_im4;	// L952
          bw_im4 = v738;	// L953
          float v740 = a_re4;	// L954
          float v741 = bw_re4;	// L955
          float v742 = v740 + v741;	// L956
          #pragma HLS bind_op variable=v742 op=fadd impl=fabric
          int32_t v743 = il4;	// L957
          int v744 = v743;	// L958
          o_re4[v744] = v742;	// L959
          float v745 = a_im4;	// L960
          float v746 = bw_im4;	// L961
          float v747 = v745 + v746;	// L962
          #pragma HLS bind_op variable=v747 op=fadd impl=fabric
          int32_t v748 = il4;	// L963
          int v749 = v748;	// L964
          o_im4[v749] = v747;	// L965
          float v750 = a_re4;	// L966
          float v751 = bw_re4;	// L967
          float v752 = v750 - v751;	// L968
          #pragma HLS bind_op variable=v752 op=fsub impl=fabric
          int32_t v753 = iu4;	// L969
          int v754 = v753;	// L970
          o_re4[v754] = v752;	// L971
          float v755 = a_im4;	// L972
          float v756 = bw_im4;	// L973
          float v757 = v755 - v756;	// L974
          #pragma HLS bind_op variable=v757 op=fsub impl=fabric
          int32_t v758 = iu4;	// L975
          int v759 = v758;	// L976
          o_im4[v759] = v757;	// L977
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re4[_iv0];
      }
      v623.write(_vec);
    }	// L981
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im4[_iv0];
      }
      v624.write(_vec);
    }	// L982
  }
}

void inter_0(
  hls::stream< hls::vector< float, 32 > >& v760,
  hls::stream< hls::vector< float, 32 > >& v761,
  hls::stream< hls::vector< float, 32 > >& v762,
  hls::stream< hls::vector< float, 32 > >& v763
) {	// L986
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L994
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L995
  #pragma HLS array_partition variable=twi complete
  float in_re[32][8];	// L996
  #pragma HLS array_partition variable=in_re complete dim=1

  #pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L997
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L998
  #pragma HLS array_partition variable=out_re_b complete dim=1

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L999
  #pragma HLS array_partition variable=out_im_b complete dim=1

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L1000
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v771 = v760.read();
    hls::vector< float, 32 > v772 = v761.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L1003
    #pragma HLS unroll
      float v774 = v771[k5];	// L1004
      int v775 = ((i * 32) + k5);	// L1005
      int32_t v776 = v775;	// L1006
      int32_t v777 = v776 & 31;	// L1008
      int32_t v778 = v776 >> 5;	// L1010
      int32_t v779 = v778 & 1;	// L1012
      int32_t v780 = v779 << 4;	// L1014
      int32_t v781 = v777 ^ v780;	// L1015
      int v782 = v781;	// L1016
      int v783 = v778;	// L1017
      in_re[v782][v783] = v774;	// L1018
      float v784 = v772[k5];	// L1019
      in_im[v782][v783] = v784;	// L1020
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L1023
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im inter false
  #pragma HLS dependence variable=in_im intra false
  #pragma HLS dependence variable=in_re inter false
  #pragma HLS dependence variable=in_re intra false
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L1024
    #pragma HLS unroll
      int v787 = i1 << 4;	// L1025
      int v788 = v787 | k6;	// L1026
      uint32_t v789 = v788;	// L1027
      uint32_t bg;	// L1028
      bg = v789;	// L1029
      uint32_t stride5;	// L1030
      stride5 = 32;	// L1031
      int32_t v792 = bg;	// L1032
      uint32_t v793 = v792 >> 5;	// L1033
      uint32_t v794 = v793 << 6;	// L1034
      int32_t v795 = stride5;	// L1035
      int64_t v796 = v795;	// L1036
      int64_t v797 = v796 - 1;	// L1037
      int64_t v798 = v792;	// L1038
      int64_t v799 = v798 & v797;	// L1039
      int64_t v800 = v794;	// L1040
      int64_t v801 = v800 | v799;	// L1041
      uint32_t v802 = v801;	// L1042
      uint32_t il5;	// L1043
      il5 = v802;	// L1044
      int32_t v804 = il5;	// L1045
      int32_t v805 = stride5;	// L1046
      uint64_t v806 = v804;	// L1047
      uint64_t v807 = v805;	// L1048
      uint64_t v808 = v806 + v807;	// L1049
      uint32_t v809 = v808;	// L1050
      uint32_t iu5;	// L1051
      iu5 = v809;	// L1052
      int32_t v811 = il5;	// L1053
      int v812 = v811;	// L1054
      int32_t v813 = v812;	// L1055
      int32_t v814 = v813 & 31;	// L1057
      int32_t v815 = v813 >> 5;	// L1059
      int32_t v816 = v815 & 1;	// L1061
      int32_t v817 = v816 << 4;	// L1063
      int32_t v818 = v814 ^ v817;	// L1064
      int v819 = v818;	// L1065
      int v820 = v815;	// L1066
      float v821 = in_re[v819][v820];	// L1067
      float a_re5;	// L1068
      a_re5 = v821;	// L1069
      int32_t v823 = il5;	// L1070
      int v824 = v823;	// L1071
      int32_t v825 = v824;	// L1072
      int32_t v826 = v825 & 31;	// L1073
      int32_t v827 = v825 >> 5;	// L1074
      int32_t v828 = v827 & 1;	// L1075
      int32_t v829 = v828 << 4;	// L1076
      int32_t v830 = v826 ^ v829;	// L1077
      int v831 = v830;	// L1078
      int v832 = v827;	// L1079
      float v833 = in_im[v831][v832];	// L1080
      float a_im5;	// L1081
      a_im5 = v833;	// L1082
      int32_t v835 = iu5;	// L1083
      int v836 = v835;	// L1084
      int32_t v837 = v836;	// L1085
      int32_t v838 = v837 & 31;	// L1086
      int32_t v839 = v837 >> 5;	// L1087
      int32_t v840 = v839 & 1;	// L1088
      int32_t v841 = v840 << 4;	// L1089
      int32_t v842 = v838 ^ v841;	// L1090
      int v843 = v842;	// L1091
      int v844 = v839;	// L1092
      float v845 = in_re[v843][v844];	// L1093
      float b_re5;	// L1094
      b_re5 = v845;	// L1095
      int32_t v847 = iu5;	// L1096
      int v848 = v847;	// L1097
      int32_t v849 = v848;	// L1098
      int32_t v850 = v849 & 31;	// L1099
      int32_t v851 = v849 >> 5;	// L1100
      int32_t v852 = v851 & 1;	// L1101
      int32_t v853 = v852 << 4;	// L1102
      int32_t v854 = v850 ^ v853;	// L1103
      int v855 = v854;	// L1104
      int v856 = v851;	// L1105
      float v857 = in_im[v855][v856];	// L1106
      float b_im5;	// L1107
      b_im5 = v857;	// L1108
      int32_t v859 = bg;	// L1109
      int64_t v860 = v859;	// L1110
      int64_t v861 = v860 & 31;	// L1111
      int64_t v862 = v861 << 2;	// L1112
      uint32_t v863 = v862;	// L1113
      uint32_t tw_k5;	// L1114
      tw_k5 = v863;	// L1115
      int32_t v865 = tw_k5;	// L1116
      int v866 = v865;	// L1117
      float v867 = twr[v866];	// L1118
      float tr5;	// L1119
      tr5 = v867;	// L1120
      int32_t v869 = tw_k5;	// L1121
      int v870 = v869;	// L1122
      float v871 = twi[v870];	// L1123
      float ti5;	// L1124
      ti5 = v871;	// L1125
      float v873 = b_re5;	// L1126
      float v874 = tr5;	// L1127
      float v875 = v873 * v874;	// L1128
      float v876 = b_im5;	// L1129
      float v877 = ti5;	// L1130
      float v878 = v876 * v877;	// L1131
      float v879 = v875 - v878;	// L1132
      float bw_re5;	// L1133
      bw_re5 = v879;	// L1134
      float v881 = b_re5;	// L1135
      float v882 = ti5;	// L1136
      float v883 = v881 * v882;	// L1137
      float v884 = b_im5;	// L1138
      float v885 = tr5;	// L1139
      float v886 = v884 * v885;	// L1140
      float v887 = v883 + v886;	// L1141
      float bw_im5;	// L1142
      bw_im5 = v887;	// L1143
      float v889 = a_re5;	// L1144
      float v890 = bw_re5;	// L1145
      float v891 = v889 + v890;	// L1146
      #pragma HLS bind_op variable=v891 op=fadd impl=fabric
      int32_t v892 = il5;	// L1147
      int v893 = v892;	// L1148
      int32_t v894 = v893;	// L1149
      int32_t v895 = v894 & 31;	// L1150
      int32_t v896 = v894 >> 5;	// L1151
      int32_t v897 = v896 & 1;	// L1152
      int32_t v898 = v897 << 4;	// L1153
      int32_t v899 = v895 ^ v898;	// L1154
      int v900 = v899;	// L1155
      int v901 = v896;	// L1156
      out_re_b[v900][v901] = v891;	// L1157
      float v902 = a_im5;	// L1158
      float v903 = bw_im5;	// L1159
      float v904 = v902 + v903;	// L1160
      #pragma HLS bind_op variable=v904 op=fadd impl=fabric
      int32_t v905 = il5;	// L1161
      int v906 = v905;	// L1162
      int32_t v907 = v906;	// L1163
      int32_t v908 = v907 & 31;	// L1164
      int32_t v909 = v907 >> 5;	// L1165
      int32_t v910 = v909 & 1;	// L1166
      int32_t v911 = v910 << 4;	// L1167
      int32_t v912 = v908 ^ v911;	// L1168
      int v913 = v912;	// L1169
      int v914 = v909;	// L1170
      out_im_b[v913][v914] = v904;	// L1171
      float v915 = a_re5;	// L1172
      float v916 = bw_re5;	// L1173
      float v917 = v915 - v916;	// L1174
      #pragma HLS bind_op variable=v917 op=fsub impl=fabric
      int32_t v918 = iu5;	// L1175
      int v919 = v918;	// L1176
      int32_t v920 = v919;	// L1177
      int32_t v921 = v920 & 31;	// L1178
      int32_t v922 = v920 >> 5;	// L1179
      int32_t v923 = v922 & 1;	// L1180
      int32_t v924 = v923 << 4;	// L1181
      int32_t v925 = v921 ^ v924;	// L1182
      int v926 = v925;	// L1183
      int v927 = v922;	// L1184
      out_re_b[v926][v927] = v917;	// L1185
      float v928 = a_im5;	// L1186
      float v929 = bw_im5;	// L1187
      float v930 = v928 - v929;	// L1188
      #pragma HLS bind_op variable=v930 op=fsub impl=fabric
      int32_t v931 = iu5;	// L1189
      int v932 = v931;	// L1190
      int32_t v933 = v932;	// L1191
      int32_t v934 = v933 & 31;	// L1192
      int32_t v935 = v933 >> 5;	// L1193
      int32_t v936 = v935 & 1;	// L1194
      int32_t v937 = v936 << 4;	// L1195
      int32_t v938 = v934 ^ v937;	// L1196
      int v939 = v938;	// L1197
      int v940 = v935;	// L1198
      out_im_b[v939][v940] = v930;	// L1199
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L1202
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    float chunk_re_out[32];	// L1203
    #pragma HLS array_partition variable=chunk_re_out complete
    float chunk_im_out[32];	// L1204
    #pragma HLS array_partition variable=chunk_im_out complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L1205
    #pragma HLS unroll
      int v945 = ((i2 * 32) + k7);	// L1206
      int32_t v946 = v945;	// L1207
      int32_t v947 = v946 & 31;	// L1209
      int32_t v948 = v946 >> 5;	// L1211
      int32_t v949 = v948 & 1;	// L1213
      int32_t v950 = v949 << 4;	// L1215
      int32_t v951 = v947 ^ v950;	// L1216
      int v952 = v951;	// L1217
      int v953 = v948;	// L1218
      float v954 = out_re_b[v952][v953];	// L1219
      chunk_re_out[k7] = v954;	// L1220
      float v955 = out_im_b[v952][v953];	// L1221
      chunk_im_out[k7] = v955;	// L1222
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out[_iv0];
      }
      v762.write(_vec);
    }	// L1224
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out[_iv0];
      }
      v763.write(_vec);
    }	// L1225
  }
}

void inter_1(
  hls::stream< hls::vector< float, 32 > >& v956,
  hls::stream< hls::vector< float, 32 > >& v957,
  hls::stream< hls::vector< float, 32 > >& v958,
  hls::stream< hls::vector< float, 32 > >& v959
) {	// L1229
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1237
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1238
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L1239
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L1240
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L1241
  #pragma HLS array_partition variable=out_re_b1 complete dim=1

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L1242
  #pragma HLS array_partition variable=out_im_b1 complete dim=1

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L1243
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v967 = v956.read();
    hls::vector< float, 32 > v968 = v957.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L1246
    #pragma HLS unroll
      float v970 = v967[k8];	// L1247
      int v971 = ((i3 * 32) + k8);	// L1248
      int32_t v972 = v971;	// L1249
      int32_t v973 = v972 & 31;	// L1251
      int32_t v974 = v972 >> 6;	// L1253
      int32_t v975 = v974 & 1;	// L1255
      int32_t v976 = v975 << 4;	// L1257
      int32_t v977 = v973 ^ v976;	// L1258
      int32_t v978 = v972 >> 5;	// L1260
      int v979 = v977;	// L1261
      int v980 = v978;	// L1262
      in_re1[v979][v980] = v970;	// L1263
      float v981 = v968[k8];	// L1264
      in_im1[v979][v980] = v981;	// L1265
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L1268
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im1 inter false
  #pragma HLS dependence variable=in_im1 intra false
  #pragma HLS dependence variable=in_re1 inter false
  #pragma HLS dependence variable=in_re1 intra false
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L1269
    #pragma HLS unroll
      int v984 = i4 << 4;	// L1270
      int v985 = v984 | k9;	// L1271
      uint32_t v986 = v985;	// L1272
      uint32_t bg1;	// L1273
      bg1 = v986;	// L1274
      uint32_t stride6;	// L1275
      stride6 = 64;	// L1276
      int32_t v989 = bg1;	// L1277
      uint32_t v990 = v989 >> 6;	// L1278
      uint32_t v991 = v990 << 7;	// L1279
      int32_t v992 = stride6;	// L1280
      int64_t v993 = v992;	// L1281
      int64_t v994 = v993 - 1;	// L1282
      int64_t v995 = v989;	// L1283
      int64_t v996 = v995 & v994;	// L1284
      int64_t v997 = v991;	// L1285
      int64_t v998 = v997 | v996;	// L1286
      uint32_t v999 = v998;	// L1287
      uint32_t il6;	// L1288
      il6 = v999;	// L1289
      int32_t v1001 = il6;	// L1290
      int32_t v1002 = stride6;	// L1291
      uint64_t v1003 = v1001;	// L1292
      uint64_t v1004 = v1002;	// L1293
      uint64_t v1005 = v1003 + v1004;	// L1294
      uint32_t v1006 = v1005;	// L1295
      uint32_t iu6;	// L1296
      iu6 = v1006;	// L1297
      int32_t v1008 = il6;	// L1298
      int v1009 = v1008;	// L1299
      int32_t v1010 = v1009;	// L1300
      int32_t v1011 = v1010 & 31;	// L1302
      int32_t v1012 = v1010 >> 6;	// L1304
      int32_t v1013 = v1012 & 1;	// L1306
      int32_t v1014 = v1013 << 4;	// L1308
      int32_t v1015 = v1011 ^ v1014;	// L1309
      int32_t v1016 = v1010 >> 5;	// L1311
      int v1017 = v1015;	// L1312
      int v1018 = v1016;	// L1313
      float v1019 = in_re1[v1017][v1018];	// L1314
      float a_re6;	// L1315
      a_re6 = v1019;	// L1316
      int32_t v1021 = il6;	// L1317
      int v1022 = v1021;	// L1318
      int32_t v1023 = v1022;	// L1319
      int32_t v1024 = v1023 & 31;	// L1320
      int32_t v1025 = v1023 >> 6;	// L1321
      int32_t v1026 = v1025 & 1;	// L1322
      int32_t v1027 = v1026 << 4;	// L1323
      int32_t v1028 = v1024 ^ v1027;	// L1324
      int32_t v1029 = v1023 >> 5;	// L1325
      int v1030 = v1028;	// L1326
      int v1031 = v1029;	// L1327
      float v1032 = in_im1[v1030][v1031];	// L1328
      float a_im6;	// L1329
      a_im6 = v1032;	// L1330
      int32_t v1034 = iu6;	// L1331
      int v1035 = v1034;	// L1332
      int32_t v1036 = v1035;	// L1333
      int32_t v1037 = v1036 & 31;	// L1334
      int32_t v1038 = v1036 >> 6;	// L1335
      int32_t v1039 = v1038 & 1;	// L1336
      int32_t v1040 = v1039 << 4;	// L1337
      int32_t v1041 = v1037 ^ v1040;	// L1338
      int32_t v1042 = v1036 >> 5;	// L1339
      int v1043 = v1041;	// L1340
      int v1044 = v1042;	// L1341
      float v1045 = in_re1[v1043][v1044];	// L1342
      float b_re6;	// L1343
      b_re6 = v1045;	// L1344
      int32_t v1047 = iu6;	// L1345
      int v1048 = v1047;	// L1346
      int32_t v1049 = v1048;	// L1347
      int32_t v1050 = v1049 & 31;	// L1348
      int32_t v1051 = v1049 >> 6;	// L1349
      int32_t v1052 = v1051 & 1;	// L1350
      int32_t v1053 = v1052 << 4;	// L1351
      int32_t v1054 = v1050 ^ v1053;	// L1352
      int32_t v1055 = v1049 >> 5;	// L1353
      int v1056 = v1054;	// L1354
      int v1057 = v1055;	// L1355
      float v1058 = in_im1[v1056][v1057];	// L1356
      float b_im6;	// L1357
      b_im6 = v1058;	// L1358
      int32_t v1060 = bg1;	// L1359
      int64_t v1061 = v1060;	// L1360
      int64_t v1062 = v1061 & 63;	// L1361
      int64_t v1063 = v1062 << 1;	// L1362
      uint32_t v1064 = v1063;	// L1363
      uint32_t tw_k6;	// L1364
      tw_k6 = v1064;	// L1365
      int32_t v1066 = tw_k6;	// L1366
      int v1067 = v1066;	// L1367
      float v1068 = twr[v1067];	// L1368
      float tr6;	// L1369
      tr6 = v1068;	// L1370
      int32_t v1070 = tw_k6;	// L1371
      int v1071 = v1070;	// L1372
      float v1072 = twi[v1071];	// L1373
      float ti6;	// L1374
      ti6 = v1072;	// L1375
      float v1074 = b_re6;	// L1376
      float v1075 = tr6;	// L1377
      float v1076 = v1074 * v1075;	// L1378
      float v1077 = b_im6;	// L1379
      float v1078 = ti6;	// L1380
      float v1079 = v1077 * v1078;	// L1381
      float v1080 = v1076 - v1079;	// L1382
      float bw_re6;	// L1383
      bw_re6 = v1080;	// L1384
      float v1082 = b_re6;	// L1385
      float v1083 = ti6;	// L1386
      float v1084 = v1082 * v1083;	// L1387
      float v1085 = b_im6;	// L1388
      float v1086 = tr6;	// L1389
      float v1087 = v1085 * v1086;	// L1390
      float v1088 = v1084 + v1087;	// L1391
      float bw_im6;	// L1392
      bw_im6 = v1088;	// L1393
      float v1090 = a_re6;	// L1394
      float v1091 = bw_re6;	// L1395
      float v1092 = v1090 + v1091;	// L1396
      #pragma HLS bind_op variable=v1092 op=fadd impl=fabric
      int32_t v1093 = il6;	// L1397
      int v1094 = v1093;	// L1398
      int32_t v1095 = v1094;	// L1399
      int32_t v1096 = v1095 & 31;	// L1400
      int32_t v1097 = v1095 >> 6;	// L1401
      int32_t v1098 = v1097 & 1;	// L1402
      int32_t v1099 = v1098 << 4;	// L1403
      int32_t v1100 = v1096 ^ v1099;	// L1404
      int32_t v1101 = v1095 >> 5;	// L1405
      int v1102 = v1100;	// L1406
      int v1103 = v1101;	// L1407
      out_re_b1[v1102][v1103] = v1092;	// L1408
      float v1104 = a_im6;	// L1409
      float v1105 = bw_im6;	// L1410
      float v1106 = v1104 + v1105;	// L1411
      #pragma HLS bind_op variable=v1106 op=fadd impl=fabric
      int32_t v1107 = il6;	// L1412
      int v1108 = v1107;	// L1413
      int32_t v1109 = v1108;	// L1414
      int32_t v1110 = v1109 & 31;	// L1415
      int32_t v1111 = v1109 >> 6;	// L1416
      int32_t v1112 = v1111 & 1;	// L1417
      int32_t v1113 = v1112 << 4;	// L1418
      int32_t v1114 = v1110 ^ v1113;	// L1419
      int32_t v1115 = v1109 >> 5;	// L1420
      int v1116 = v1114;	// L1421
      int v1117 = v1115;	// L1422
      out_im_b1[v1116][v1117] = v1106;	// L1423
      float v1118 = a_re6;	// L1424
      float v1119 = bw_re6;	// L1425
      float v1120 = v1118 - v1119;	// L1426
      #pragma HLS bind_op variable=v1120 op=fsub impl=fabric
      int32_t v1121 = iu6;	// L1427
      int v1122 = v1121;	// L1428
      int32_t v1123 = v1122;	// L1429
      int32_t v1124 = v1123 & 31;	// L1430
      int32_t v1125 = v1123 >> 6;	// L1431
      int32_t v1126 = v1125 & 1;	// L1432
      int32_t v1127 = v1126 << 4;	// L1433
      int32_t v1128 = v1124 ^ v1127;	// L1434
      int32_t v1129 = v1123 >> 5;	// L1435
      int v1130 = v1128;	// L1436
      int v1131 = v1129;	// L1437
      out_re_b1[v1130][v1131] = v1120;	// L1438
      float v1132 = a_im6;	// L1439
      float v1133 = bw_im6;	// L1440
      float v1134 = v1132 - v1133;	// L1441
      #pragma HLS bind_op variable=v1134 op=fsub impl=fabric
      int32_t v1135 = iu6;	// L1442
      int v1136 = v1135;	// L1443
      int32_t v1137 = v1136;	// L1444
      int32_t v1138 = v1137 & 31;	// L1445
      int32_t v1139 = v1137 >> 6;	// L1446
      int32_t v1140 = v1139 & 1;	// L1447
      int32_t v1141 = v1140 << 4;	// L1448
      int32_t v1142 = v1138 ^ v1141;	// L1449
      int32_t v1143 = v1137 >> 5;	// L1450
      int v1144 = v1142;	// L1451
      int v1145 = v1143;	// L1452
      out_im_b1[v1144][v1145] = v1134;	// L1453
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L1456
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    float chunk_re_out1[32];	// L1457
    #pragma HLS array_partition variable=chunk_re_out1 complete
    float chunk_im_out1[32];	// L1458
    #pragma HLS array_partition variable=chunk_im_out1 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L1459
    #pragma HLS unroll
      int v1150 = ((i5 * 32) + k10);	// L1460
      int32_t v1151 = v1150;	// L1461
      int32_t v1152 = v1151 & 31;	// L1463
      int32_t v1153 = v1151 >> 6;	// L1465
      int32_t v1154 = v1153 & 1;	// L1467
      int32_t v1155 = v1154 << 4;	// L1469
      int32_t v1156 = v1152 ^ v1155;	// L1470
      int32_t v1157 = v1151 >> 5;	// L1472
      int v1158 = v1156;	// L1473
      int v1159 = v1157;	// L1474
      float v1160 = out_re_b1[v1158][v1159];	// L1475
      chunk_re_out1[k10] = v1160;	// L1476
      float v1161 = out_im_b1[v1158][v1159];	// L1477
      chunk_im_out1[k10] = v1161;	// L1478
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out1[_iv0];
      }
      v958.write(_vec);
    }	// L1480
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out1[_iv0];
      }
      v959.write(_vec);
    }	// L1481
  }
}

void inter_2(
  hls::stream< hls::vector< float, 32 > >& v1162,
  hls::stream< hls::vector< float, 32 > >& v1163,
  hls::stream< hls::vector< float, 32 > >& v1164,
  hls::stream< hls::vector< float, 32 > >& v1165
) {	// L1485
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1492
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1493
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L1494
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L1495
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L1496
  #pragma HLS array_partition variable=out_re_b2 complete dim=1

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L1497
  #pragma HLS array_partition variable=out_im_b2 complete dim=1

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L1498
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1173 = v1162.read();
    hls::vector< float, 32 > v1174 = v1163.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L1501
    #pragma HLS unroll
      float v1176 = v1173[k11];	// L1502
      int v1177 = ((i6 * 32) + k11);	// L1503
      int32_t v1178 = v1177;	// L1504
      int32_t v1179 = v1178 & 31;	// L1506
      int32_t v1180 = v1178 >> 7;	// L1508
      int32_t v1181 = v1180 & 1;	// L1510
      int32_t v1182 = v1181 << 4;	// L1512
      int32_t v1183 = v1179 ^ v1182;	// L1513
      int32_t v1184 = v1178 >> 5;	// L1515
      int v1185 = v1183;	// L1516
      int v1186 = v1184;	// L1517
      in_re2[v1185][v1186] = v1176;	// L1518
      float v1187 = v1174[k11];	// L1519
      in_im2[v1185][v1186] = v1187;	// L1520
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L1523
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im2 inter false
  #pragma HLS dependence variable=in_im2 intra false
  #pragma HLS dependence variable=in_re2 inter false
  #pragma HLS dependence variable=in_re2 intra false
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L1524
    #pragma HLS unroll
      int v1190 = i7 << 4;	// L1525
      int v1191 = v1190 | k12;	// L1526
      uint32_t v1192 = v1191;	// L1527
      uint32_t bg2;	// L1528
      bg2 = v1192;	// L1529
      uint32_t stride7;	// L1530
      stride7 = 128;	// L1531
      int32_t v1195 = bg2;	// L1532
      uint32_t v1196 = v1195 >> 7;	// L1533
      uint32_t v1197 = v1196 << 8;	// L1534
      int32_t v1198 = stride7;	// L1535
      int64_t v1199 = v1198;	// L1536
      int64_t v1200 = v1199 - 1;	// L1537
      int64_t v1201 = v1195;	// L1538
      int64_t v1202 = v1201 & v1200;	// L1539
      int64_t v1203 = v1197;	// L1540
      int64_t v1204 = v1203 | v1202;	// L1541
      uint32_t v1205 = v1204;	// L1542
      uint32_t il7;	// L1543
      il7 = v1205;	// L1544
      int32_t v1207 = il7;	// L1545
      int32_t v1208 = stride7;	// L1546
      uint64_t v1209 = v1207;	// L1547
      uint64_t v1210 = v1208;	// L1548
      uint64_t v1211 = v1209 + v1210;	// L1549
      uint32_t v1212 = v1211;	// L1550
      uint32_t iu7;	// L1551
      iu7 = v1212;	// L1552
      int32_t v1214 = il7;	// L1553
      int v1215 = v1214;	// L1554
      int32_t v1216 = v1215;	// L1555
      int32_t v1217 = v1216 & 31;	// L1557
      int32_t v1218 = v1216 >> 7;	// L1559
      int32_t v1219 = v1218 & 1;	// L1561
      int32_t v1220 = v1219 << 4;	// L1563
      int32_t v1221 = v1217 ^ v1220;	// L1564
      int32_t v1222 = v1216 >> 5;	// L1566
      int v1223 = v1221;	// L1567
      int v1224 = v1222;	// L1568
      float v1225 = in_re2[v1223][v1224];	// L1569
      float a_re7;	// L1570
      a_re7 = v1225;	// L1571
      int32_t v1227 = il7;	// L1572
      int v1228 = v1227;	// L1573
      int32_t v1229 = v1228;	// L1574
      int32_t v1230 = v1229 & 31;	// L1575
      int32_t v1231 = v1229 >> 7;	// L1576
      int32_t v1232 = v1231 & 1;	// L1577
      int32_t v1233 = v1232 << 4;	// L1578
      int32_t v1234 = v1230 ^ v1233;	// L1579
      int32_t v1235 = v1229 >> 5;	// L1580
      int v1236 = v1234;	// L1581
      int v1237 = v1235;	// L1582
      float v1238 = in_im2[v1236][v1237];	// L1583
      float a_im7;	// L1584
      a_im7 = v1238;	// L1585
      int32_t v1240 = iu7;	// L1586
      int v1241 = v1240;	// L1587
      int32_t v1242 = v1241;	// L1588
      int32_t v1243 = v1242 & 31;	// L1589
      int32_t v1244 = v1242 >> 7;	// L1590
      int32_t v1245 = v1244 & 1;	// L1591
      int32_t v1246 = v1245 << 4;	// L1592
      int32_t v1247 = v1243 ^ v1246;	// L1593
      int32_t v1248 = v1242 >> 5;	// L1594
      int v1249 = v1247;	// L1595
      int v1250 = v1248;	// L1596
      float v1251 = in_re2[v1249][v1250];	// L1597
      float b_re7;	// L1598
      b_re7 = v1251;	// L1599
      int32_t v1253 = iu7;	// L1600
      int v1254 = v1253;	// L1601
      int32_t v1255 = v1254;	// L1602
      int32_t v1256 = v1255 & 31;	// L1603
      int32_t v1257 = v1255 >> 7;	// L1604
      int32_t v1258 = v1257 & 1;	// L1605
      int32_t v1259 = v1258 << 4;	// L1606
      int32_t v1260 = v1256 ^ v1259;	// L1607
      int32_t v1261 = v1255 >> 5;	// L1608
      int v1262 = v1260;	// L1609
      int v1263 = v1261;	// L1610
      float v1264 = in_im2[v1262][v1263];	// L1611
      float b_im7;	// L1612
      b_im7 = v1264;	// L1613
      int32_t v1266 = bg2;	// L1614
      int64_t v1267 = v1266;	// L1615
      int64_t v1268 = v1267 & 127;	// L1616
      uint32_t v1269 = v1268;	// L1617
      uint32_t tw_k7;	// L1618
      tw_k7 = v1269;	// L1619
      int32_t v1271 = tw_k7;	// L1620
      int v1272 = v1271;	// L1621
      float v1273 = twr[v1272];	// L1622
      float tr7;	// L1623
      tr7 = v1273;	// L1624
      int32_t v1275 = tw_k7;	// L1625
      int v1276 = v1275;	// L1626
      float v1277 = twi[v1276];	// L1627
      float ti7;	// L1628
      ti7 = v1277;	// L1629
      float v1279 = b_re7;	// L1630
      float v1280 = tr7;	// L1631
      float v1281 = v1279 * v1280;	// L1632
      float v1282 = b_im7;	// L1633
      float v1283 = ti7;	// L1634
      float v1284 = v1282 * v1283;	// L1635
      float v1285 = v1281 - v1284;	// L1636
      float bw_re7;	// L1637
      bw_re7 = v1285;	// L1638
      float v1287 = b_re7;	// L1639
      float v1288 = ti7;	// L1640
      float v1289 = v1287 * v1288;	// L1641
      float v1290 = b_im7;	// L1642
      float v1291 = tr7;	// L1643
      float v1292 = v1290 * v1291;	// L1644
      float v1293 = v1289 + v1292;	// L1645
      float bw_im7;	// L1646
      bw_im7 = v1293;	// L1647
      float v1295 = a_re7;	// L1648
      float v1296 = bw_re7;	// L1649
      float v1297 = v1295 + v1296;	// L1650
      #pragma HLS bind_op variable=v1297 op=fadd impl=fabric
      int32_t v1298 = il7;	// L1651
      int v1299 = v1298;	// L1652
      int32_t v1300 = v1299;	// L1653
      int32_t v1301 = v1300 & 31;	// L1654
      int32_t v1302 = v1300 >> 7;	// L1655
      int32_t v1303 = v1302 & 1;	// L1656
      int32_t v1304 = v1303 << 4;	// L1657
      int32_t v1305 = v1301 ^ v1304;	// L1658
      int32_t v1306 = v1300 >> 5;	// L1659
      int v1307 = v1305;	// L1660
      int v1308 = v1306;	// L1661
      out_re_b2[v1307][v1308] = v1297;	// L1662
      float v1309 = a_im7;	// L1663
      float v1310 = bw_im7;	// L1664
      float v1311 = v1309 + v1310;	// L1665
      #pragma HLS bind_op variable=v1311 op=fadd impl=fabric
      int32_t v1312 = il7;	// L1666
      int v1313 = v1312;	// L1667
      int32_t v1314 = v1313;	// L1668
      int32_t v1315 = v1314 & 31;	// L1669
      int32_t v1316 = v1314 >> 7;	// L1670
      int32_t v1317 = v1316 & 1;	// L1671
      int32_t v1318 = v1317 << 4;	// L1672
      int32_t v1319 = v1315 ^ v1318;	// L1673
      int32_t v1320 = v1314 >> 5;	// L1674
      int v1321 = v1319;	// L1675
      int v1322 = v1320;	// L1676
      out_im_b2[v1321][v1322] = v1311;	// L1677
      float v1323 = a_re7;	// L1678
      float v1324 = bw_re7;	// L1679
      float v1325 = v1323 - v1324;	// L1680
      #pragma HLS bind_op variable=v1325 op=fsub impl=fabric
      int32_t v1326 = iu7;	// L1681
      int v1327 = v1326;	// L1682
      int32_t v1328 = v1327;	// L1683
      int32_t v1329 = v1328 & 31;	// L1684
      int32_t v1330 = v1328 >> 7;	// L1685
      int32_t v1331 = v1330 & 1;	// L1686
      int32_t v1332 = v1331 << 4;	// L1687
      int32_t v1333 = v1329 ^ v1332;	// L1688
      int32_t v1334 = v1328 >> 5;	// L1689
      int v1335 = v1333;	// L1690
      int v1336 = v1334;	// L1691
      out_re_b2[v1335][v1336] = v1325;	// L1692
      float v1337 = a_im7;	// L1693
      float v1338 = bw_im7;	// L1694
      float v1339 = v1337 - v1338;	// L1695
      #pragma HLS bind_op variable=v1339 op=fsub impl=fabric
      int32_t v1340 = iu7;	// L1696
      int v1341 = v1340;	// L1697
      int32_t v1342 = v1341;	// L1698
      int32_t v1343 = v1342 & 31;	// L1699
      int32_t v1344 = v1342 >> 7;	// L1700
      int32_t v1345 = v1344 & 1;	// L1701
      int32_t v1346 = v1345 << 4;	// L1702
      int32_t v1347 = v1343 ^ v1346;	// L1703
      int32_t v1348 = v1342 >> 5;	// L1704
      int v1349 = v1347;	// L1705
      int v1350 = v1348;	// L1706
      out_im_b2[v1349][v1350] = v1339;	// L1707
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1710
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    float chunk_re_out2[32];	// L1711
    #pragma HLS array_partition variable=chunk_re_out2 complete
    float chunk_im_out2[32];	// L1712
    #pragma HLS array_partition variable=chunk_im_out2 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1713
    #pragma HLS unroll
      int v1355 = ((i8 * 32) + k13);	// L1714
      int32_t v1356 = v1355;	// L1715
      int32_t v1357 = v1356 & 31;	// L1717
      int32_t v1358 = v1356 >> 7;	// L1719
      int32_t v1359 = v1358 & 1;	// L1721
      int32_t v1360 = v1359 << 4;	// L1723
      int32_t v1361 = v1357 ^ v1360;	// L1724
      int32_t v1362 = v1356 >> 5;	// L1726
      int v1363 = v1361;	// L1727
      int v1364 = v1362;	// L1728
      float v1365 = out_re_b2[v1363][v1364];	// L1729
      chunk_re_out2[k13] = v1365;	// L1730
      float v1366 = out_im_b2[v1363][v1364];	// L1731
      chunk_im_out2[k13] = v1366;	// L1732
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out2[_iv0];
      }
      v1164.write(_vec);
    }	// L1734
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out2[_iv0];
      }
      v1165.write(_vec);
    }	// L1735
  }
}

void output_stage_0(
  hls::stream< hls::vector< float, 32 > >& v1367,
  hls::stream< hls::vector< float, 32 > >& v1368,
  hls::stream< hls::vector< float, 32 > >& v1369,
  hls::stream< hls::vector< float, 32 > >& v1370
) {	// L1739
  l_S_i_0_i9: for (int i9 = 0; i9 < 8; i9++) {	// L1740
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1372 = v1367.read();
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = v1372[_iv0];
      }
      v1368.write(_vec);
    }	// L1742
    hls::vector< float, 32 > v1373 = v1369.read();
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = v1373[_iv0];
      }
      v1370.write(_vec);
    }	// L1744
  }
}

/// This is top function.
void fft_256(
  hls::stream< hls::vector< float, 32 > >& v1374,
  hls::stream< hls::vector< float, 32 > >& v1375,
  hls::stream< hls::vector< float, 32 > >& v1376,
  hls::stream< hls::vector< float, 32 > >& v1377
) {	// L1748
  #pragma HLS dataflow disable_start_propagation
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1378;
  #pragma HLS stream variable=v1378 depth=2	// L1749
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1379;
  #pragma HLS stream variable=v1379 depth=2	// L1750
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1380;
  #pragma HLS stream variable=v1380 depth=2	// L1751
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1381;
  #pragma HLS stream variable=v1381 depth=2	// L1752
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1382;
  #pragma HLS stream variable=v1382 depth=2	// L1753
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1383;
  #pragma HLS stream variable=v1383 depth=2	// L1754
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1384;
  #pragma HLS stream variable=v1384 depth=2	// L1755
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1385;
  #pragma HLS stream variable=v1385 depth=2	// L1756
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1386;
  #pragma HLS stream variable=v1386 depth=2	// L1757
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1387;
  #pragma HLS stream variable=v1387 depth=2	// L1758
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1388;
  #pragma HLS stream variable=v1388 depth=2	// L1759
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1389;
  #pragma HLS stream variable=v1389 depth=2	// L1760
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1390;
  #pragma HLS stream variable=v1390 depth=2	// L1761
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1391;
  #pragma HLS stream variable=v1391 depth=2	// L1762
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1392;
  #pragma HLS stream variable=v1392 depth=2	// L1763
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1393;
  #pragma HLS stream variable=v1393 depth=2	// L1764
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1394;
  #pragma HLS stream variable=v1394 depth=2	// L1765
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1395;
  #pragma HLS stream variable=v1395 depth=2	// L1766
  bit_rev_stage_0(v1374, v1375, v1378, v1387);	// L1767
  intra_0(v1378, v1387, v1379, v1388);	// L1768
  intra_1(v1379, v1388, v1380, v1389);	// L1769
  intra_2(v1380, v1389, v1381, v1390);	// L1770
  intra_3(v1381, v1390, v1382, v1391);	// L1771
  intra_4(v1382, v1391, v1383, v1392);	// L1772
  inter_0(v1383, v1392, v1384, v1393);	// L1773
  inter_1(v1384, v1393, v1385, v1394);	// L1774
  inter_2(v1385, v1394, v1386, v1395);	// L1775
  output_stage_0(v1386, v1376, v1395, v1377);	// L1776
}


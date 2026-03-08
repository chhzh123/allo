
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

const float twr[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L90
const float twi[128] = {0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L91
void intra_0(
  hls::stream< hls::vector< float, 32 > >& v66,
  hls::stream< hls::vector< float, 32 > >& v67,
  hls::stream< hls::vector< float, 32 > >& v68,
  hls::stream< hls::vector< float, 32 > >& v69
) {	// L92
  // placeholder for const float twr	// L99
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L100
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i: for (int _i = 0; _i < 8; _i++) {	// L101
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v73 = v66.read();
    hls::vector< float, 32 > v74 = v67.read();
    float o_re[32];	// L104
    #pragma HLS array_partition variable=o_re complete
    float o_im[32];	// L105
    #pragma HLS array_partition variable=o_im complete
    int32_t stride;	// L106
    stride = 1;	// L107
    l_S_k_0_k: for (int k = 0; k < 16; k++) {	// L108
    #pragma HLS unroll
      int v79 = k << 1;	// L109
      int32_t v80 = stride;	// L110
      int64_t v81 = v80;	// L111
      int64_t v82 = v81 - 1;	// L112
      int64_t v83 = k;	// L113
      int64_t v84 = v83 & v82;	// L114
      int64_t v85 = v79;	// L115
      int64_t v86 = v85 | v84;	// L116
      int32_t v87 = v86;	// L117
      int32_t il;	// L118
      il = v87;	// L119
      int32_t v89 = il;	// L120
      int32_t v90 = stride;	// L121
      int32_t v91 = v89 | v90;	// L122
      int32_t iu;	// L123
      iu = v91;	// L124
      int32_t v93 = stride;	// L125
      int64_t v94 = v93;	// L126
      int64_t v95 = v94 - 1;	// L127
      int64_t v96 = v83 & v95;	// L128
      int64_t v97 = v96 << 7;	// L129
      int32_t v98 = v97;	// L130
      int32_t tw_k;	// L131
      tw_k = v98;	// L132
      int32_t v100 = il;	// L133
      int v101 = v100;	// L134
      float v102 = v73[v101];	// L135
      float a_re;	// L136
      a_re = v102;	// L137
      int32_t v104 = il;	// L138
      int v105 = v104;	// L139
      float v106 = v74[v105];	// L140
      float a_im;	// L141
      a_im = v106;	// L142
      int32_t v108 = iu;	// L143
      int v109 = v108;	// L144
      float v110 = v73[v109];	// L145
      float b_re;	// L146
      b_re = v110;	// L147
      int32_t v112 = iu;	// L148
      int v113 = v112;	// L149
      float v114 = v74[v113];	// L150
      float b_im;	// L151
      b_im = v114;	// L152
      int32_t v116 = tw_k;	// L153
      bool v117 = v116 == 0;	// L154
      if (v117) {	// L155
        float v118 = a_re;	// L156
        float v119 = b_re;	// L157
        float v120 = v118 + v119;	// L158
        #pragma HLS bind_op variable=v120 op=fadd impl=fabric
        int32_t v121 = il;	// L159
        int v122 = v121;	// L160
        o_re[v122] = v120;	// L161
        float v123 = a_im;	// L162
        float v124 = b_im;	// L163
        float v125 = v123 + v124;	// L164
        #pragma HLS bind_op variable=v125 op=fadd impl=fabric
        int32_t v126 = il;	// L165
        int v127 = v126;	// L166
        o_im[v127] = v125;	// L167
        float v128 = a_re;	// L168
        float v129 = b_re;	// L169
        float v130 = v128 - v129;	// L170
        #pragma HLS bind_op variable=v130 op=fsub impl=fabric
        int32_t v131 = iu;	// L171
        int v132 = v131;	// L172
        o_re[v132] = v130;	// L173
        float v133 = a_im;	// L174
        float v134 = b_im;	// L175
        float v135 = v133 - v134;	// L176
        #pragma HLS bind_op variable=v135 op=fsub impl=fabric
        int32_t v136 = iu;	// L177
        int v137 = v136;	// L178
        o_im[v137] = v135;	// L179
      } else {
        int32_t v138 = tw_k;	// L181
        bool v139 = v138 == 64;	// L182
        if (v139) {	// L183
          float v140 = a_re;	// L184
          float v141 = b_im;	// L185
          float v142 = v140 + v141;	// L186
          #pragma HLS bind_op variable=v142 op=fadd impl=fabric
          int32_t v143 = il;	// L187
          int v144 = v143;	// L188
          o_re[v144] = v142;	// L189
          float v145 = a_im;	// L190
          float v146 = b_re;	// L191
          float v147 = v145 - v146;	// L192
          #pragma HLS bind_op variable=v147 op=fsub impl=fabric
          int32_t v148 = il;	// L193
          int v149 = v148;	// L194
          o_im[v149] = v147;	// L195
          float v150 = a_re;	// L196
          float v151 = b_im;	// L197
          float v152 = v150 - v151;	// L198
          #pragma HLS bind_op variable=v152 op=fsub impl=fabric
          int32_t v153 = iu;	// L199
          int v154 = v153;	// L200
          o_re[v154] = v152;	// L201
          float v155 = a_im;	// L202
          float v156 = b_re;	// L203
          float v157 = v155 + v156;	// L204
          #pragma HLS bind_op variable=v157 op=fadd impl=fabric
          int32_t v158 = iu;	// L205
          int v159 = v158;	// L206
          o_im[v159] = v157;	// L207
        } else {
          int32_t v160 = tw_k;	// L209
          int v161 = v160;	// L210
          float v162 = twr[v161];	// L211
          float tr;	// L212
          tr = v162;	// L213
          int32_t v164 = tw_k;	// L214
          int v165 = v164;	// L215
          float v166 = twi[v165];	// L216
          float ti;	// L217
          ti = v166;	// L218
          float v168 = b_re;	// L219
          float v169 = tr;	// L220
          float v170 = v168 * v169;	// L221
          float v171 = b_im;	// L222
          float v172 = ti;	// L223
          float v173 = v171 * v172;	// L224
          float v174 = v170 - v173;	// L225
          float bw_re;	// L226
          bw_re = v174;	// L227
          float v176 = b_re;	// L228
          float v177 = ti;	// L229
          float v178 = v176 * v177;	// L230
          float v179 = b_im;	// L231
          float v180 = tr;	// L232
          float v181 = v179 * v180;	// L233
          float v182 = v178 + v181;	// L234
          float bw_im;	// L235
          bw_im = v182;	// L236
          float v184 = a_re;	// L237
          float v185 = bw_re;	// L238
          float v186 = v184 + v185;	// L239
          #pragma HLS bind_op variable=v186 op=fadd impl=fabric
          int32_t v187 = il;	// L240
          int v188 = v187;	// L241
          o_re[v188] = v186;	// L242
          float v189 = a_im;	// L243
          float v190 = bw_im;	// L244
          float v191 = v189 + v190;	// L245
          #pragma HLS bind_op variable=v191 op=fadd impl=fabric
          int32_t v192 = il;	// L246
          int v193 = v192;	// L247
          o_im[v193] = v191;	// L248
          float v194 = a_re;	// L249
          float v195 = bw_re;	// L250
          float v196 = v194 - v195;	// L251
          #pragma HLS bind_op variable=v196 op=fsub impl=fabric
          int32_t v197 = iu;	// L252
          int v198 = v197;	// L253
          o_re[v198] = v196;	// L254
          float v199 = a_im;	// L255
          float v200 = bw_im;	// L256
          float v201 = v199 - v200;	// L257
          #pragma HLS bind_op variable=v201 op=fsub impl=fabric
          int32_t v202 = iu;	// L258
          int v203 = v202;	// L259
          o_im[v203] = v201;	// L260
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
    }	// L264
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im[_iv0];
      }
      v69.write(_vec);
    }	// L265
  }
}

void intra_1(
  hls::stream< hls::vector< float, 32 > >& v204,
  hls::stream< hls::vector< float, 32 > >& v205,
  hls::stream< hls::vector< float, 32 > >& v206,
  hls::stream< hls::vector< float, 32 > >& v207
) {	// L269
  // placeholder for const float twr	// L277
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L278
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i1: for (int _i1 = 0; _i1 < 8; _i1++) {	// L279
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v211 = v204.read();
    hls::vector< float, 32 > v212 = v205.read();
    float o_re1[32];	// L282
    #pragma HLS array_partition variable=o_re1 complete
    float o_im1[32];	// L283
    #pragma HLS array_partition variable=o_im1 complete
    int32_t stride1;	// L284
    stride1 = 2;	// L285
    l_S_k_0_k1: for (int k1 = 0; k1 < 16; k1++) {	// L286
    #pragma HLS unroll
      int v217 = k1 >> 1;	// L287
      int v218 = v217 << 2;	// L288
      int32_t v219 = stride1;	// L289
      int64_t v220 = v219;	// L290
      int64_t v221 = v220 - 1;	// L291
      int64_t v222 = k1;	// L292
      int64_t v223 = v222 & v221;	// L293
      int64_t v224 = v218;	// L294
      int64_t v225 = v224 | v223;	// L295
      int32_t v226 = v225;	// L296
      int32_t il1;	// L297
      il1 = v226;	// L298
      int32_t v228 = il1;	// L299
      int32_t v229 = stride1;	// L300
      int32_t v230 = v228 | v229;	// L301
      int32_t iu1;	// L302
      iu1 = v230;	// L303
      int32_t v232 = stride1;	// L304
      int64_t v233 = v232;	// L305
      int64_t v234 = v233 - 1;	// L306
      int64_t v235 = v222 & v234;	// L307
      int64_t v236 = v235 << 6;	// L308
      int32_t v237 = v236;	// L309
      int32_t tw_k1;	// L310
      tw_k1 = v237;	// L311
      int32_t v239 = il1;	// L312
      int v240 = v239;	// L313
      float v241 = v211[v240];	// L314
      float a_re1;	// L315
      a_re1 = v241;	// L316
      int32_t v243 = il1;	// L317
      int v244 = v243;	// L318
      float v245 = v212[v244];	// L319
      float a_im1;	// L320
      a_im1 = v245;	// L321
      int32_t v247 = iu1;	// L322
      int v248 = v247;	// L323
      float v249 = v211[v248];	// L324
      float b_re1;	// L325
      b_re1 = v249;	// L326
      int32_t v251 = iu1;	// L327
      int v252 = v251;	// L328
      float v253 = v212[v252];	// L329
      float b_im1;	// L330
      b_im1 = v253;	// L331
      int32_t v255 = tw_k1;	// L332
      bool v256 = v255 == 0;	// L333
      if (v256) {	// L334
        float v257 = a_re1;	// L335
        float v258 = b_re1;	// L336
        float v259 = v257 + v258;	// L337
        #pragma HLS bind_op variable=v259 op=fadd impl=fabric
        int32_t v260 = il1;	// L338
        int v261 = v260;	// L339
        o_re1[v261] = v259;	// L340
        float v262 = a_im1;	// L341
        float v263 = b_im1;	// L342
        float v264 = v262 + v263;	// L343
        #pragma HLS bind_op variable=v264 op=fadd impl=fabric
        int32_t v265 = il1;	// L344
        int v266 = v265;	// L345
        o_im1[v266] = v264;	// L346
        float v267 = a_re1;	// L347
        float v268 = b_re1;	// L348
        float v269 = v267 - v268;	// L349
        #pragma HLS bind_op variable=v269 op=fsub impl=fabric
        int32_t v270 = iu1;	// L350
        int v271 = v270;	// L351
        o_re1[v271] = v269;	// L352
        float v272 = a_im1;	// L353
        float v273 = b_im1;	// L354
        float v274 = v272 - v273;	// L355
        #pragma HLS bind_op variable=v274 op=fsub impl=fabric
        int32_t v275 = iu1;	// L356
        int v276 = v275;	// L357
        o_im1[v276] = v274;	// L358
      } else {
        int32_t v277 = tw_k1;	// L360
        bool v278 = v277 == 64;	// L361
        if (v278) {	// L362
          float v279 = a_re1;	// L363
          float v280 = b_im1;	// L364
          float v281 = v279 + v280;	// L365
          #pragma HLS bind_op variable=v281 op=fadd impl=fabric
          int32_t v282 = il1;	// L366
          int v283 = v282;	// L367
          o_re1[v283] = v281;	// L368
          float v284 = a_im1;	// L369
          float v285 = b_re1;	// L370
          float v286 = v284 - v285;	// L371
          #pragma HLS bind_op variable=v286 op=fsub impl=fabric
          int32_t v287 = il1;	// L372
          int v288 = v287;	// L373
          o_im1[v288] = v286;	// L374
          float v289 = a_re1;	// L375
          float v290 = b_im1;	// L376
          float v291 = v289 - v290;	// L377
          #pragma HLS bind_op variable=v291 op=fsub impl=fabric
          int32_t v292 = iu1;	// L378
          int v293 = v292;	// L379
          o_re1[v293] = v291;	// L380
          float v294 = a_im1;	// L381
          float v295 = b_re1;	// L382
          float v296 = v294 + v295;	// L383
          #pragma HLS bind_op variable=v296 op=fadd impl=fabric
          int32_t v297 = iu1;	// L384
          int v298 = v297;	// L385
          o_im1[v298] = v296;	// L386
        } else {
          int32_t v299 = tw_k1;	// L388
          int v300 = v299;	// L389
          float v301 = twr[v300];	// L390
          float tr1;	// L391
          tr1 = v301;	// L392
          int32_t v303 = tw_k1;	// L393
          int v304 = v303;	// L394
          float v305 = twi[v304];	// L395
          float ti1;	// L396
          ti1 = v305;	// L397
          float v307 = b_re1;	// L398
          float v308 = tr1;	// L399
          float v309 = v307 * v308;	// L400
          float v310 = b_im1;	// L401
          float v311 = ti1;	// L402
          float v312 = v310 * v311;	// L403
          float v313 = v309 - v312;	// L404
          float bw_re1;	// L405
          bw_re1 = v313;	// L406
          float v315 = b_re1;	// L407
          float v316 = ti1;	// L408
          float v317 = v315 * v316;	// L409
          float v318 = b_im1;	// L410
          float v319 = tr1;	// L411
          float v320 = v318 * v319;	// L412
          float v321 = v317 + v320;	// L413
          float bw_im1;	// L414
          bw_im1 = v321;	// L415
          float v323 = a_re1;	// L416
          float v324 = bw_re1;	// L417
          float v325 = v323 + v324;	// L418
          #pragma HLS bind_op variable=v325 op=fadd impl=fabric
          int32_t v326 = il1;	// L419
          int v327 = v326;	// L420
          o_re1[v327] = v325;	// L421
          float v328 = a_im1;	// L422
          float v329 = bw_im1;	// L423
          float v330 = v328 + v329;	// L424
          #pragma HLS bind_op variable=v330 op=fadd impl=fabric
          int32_t v331 = il1;	// L425
          int v332 = v331;	// L426
          o_im1[v332] = v330;	// L427
          float v333 = a_re1;	// L428
          float v334 = bw_re1;	// L429
          float v335 = v333 - v334;	// L430
          #pragma HLS bind_op variable=v335 op=fsub impl=fabric
          int32_t v336 = iu1;	// L431
          int v337 = v336;	// L432
          o_re1[v337] = v335;	// L433
          float v338 = a_im1;	// L434
          float v339 = bw_im1;	// L435
          float v340 = v338 - v339;	// L436
          #pragma HLS bind_op variable=v340 op=fsub impl=fabric
          int32_t v341 = iu1;	// L437
          int v342 = v341;	// L438
          o_im1[v342] = v340;	// L439
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
    }	// L443
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im1[_iv0];
      }
      v207.write(_vec);
    }	// L444
  }
}

void intra_2(
  hls::stream< hls::vector< float, 32 > >& v343,
  hls::stream< hls::vector< float, 32 > >& v344,
  hls::stream< hls::vector< float, 32 > >& v345,
  hls::stream< hls::vector< float, 32 > >& v346
) {	// L448
  // placeholder for const float twr	// L456
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L457
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i2: for (int _i2 = 0; _i2 < 8; _i2++) {	// L458
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v350 = v343.read();
    hls::vector< float, 32 > v351 = v344.read();
    float o_re2[32];	// L461
    #pragma HLS array_partition variable=o_re2 complete
    float o_im2[32];	// L462
    #pragma HLS array_partition variable=o_im2 complete
    int32_t stride2;	// L463
    stride2 = 4;	// L464
    l_S_k_0_k2: for (int k2 = 0; k2 < 16; k2++) {	// L465
    #pragma HLS unroll
      int v356 = k2 >> 2;	// L466
      int v357 = v356 << 3;	// L467
      int32_t v358 = stride2;	// L468
      int64_t v359 = v358;	// L469
      int64_t v360 = v359 - 1;	// L470
      int64_t v361 = k2;	// L471
      int64_t v362 = v361 & v360;	// L472
      int64_t v363 = v357;	// L473
      int64_t v364 = v363 | v362;	// L474
      int32_t v365 = v364;	// L475
      int32_t il2;	// L476
      il2 = v365;	// L477
      int32_t v367 = il2;	// L478
      int32_t v368 = stride2;	// L479
      int32_t v369 = v367 | v368;	// L480
      int32_t iu2;	// L481
      iu2 = v369;	// L482
      int32_t v371 = stride2;	// L483
      int64_t v372 = v371;	// L484
      int64_t v373 = v372 - 1;	// L485
      int64_t v374 = v361 & v373;	// L486
      int64_t v375 = v374 << 5;	// L487
      int32_t v376 = v375;	// L488
      int32_t tw_k2;	// L489
      tw_k2 = v376;	// L490
      int32_t v378 = il2;	// L491
      int v379 = v378;	// L492
      float v380 = v350[v379];	// L493
      float a_re2;	// L494
      a_re2 = v380;	// L495
      int32_t v382 = il2;	// L496
      int v383 = v382;	// L497
      float v384 = v351[v383];	// L498
      float a_im2;	// L499
      a_im2 = v384;	// L500
      int32_t v386 = iu2;	// L501
      int v387 = v386;	// L502
      float v388 = v350[v387];	// L503
      float b_re2;	// L504
      b_re2 = v388;	// L505
      int32_t v390 = iu2;	// L506
      int v391 = v390;	// L507
      float v392 = v351[v391];	// L508
      float b_im2;	// L509
      b_im2 = v392;	// L510
      int32_t v394 = tw_k2;	// L511
      bool v395 = v394 == 0;	// L512
      if (v395) {	// L513
        float v396 = a_re2;	// L514
        float v397 = b_re2;	// L515
        float v398 = v396 + v397;	// L516
        #pragma HLS bind_op variable=v398 op=fadd impl=fabric
        int32_t v399 = il2;	// L517
        int v400 = v399;	// L518
        o_re2[v400] = v398;	// L519
        float v401 = a_im2;	// L520
        float v402 = b_im2;	// L521
        float v403 = v401 + v402;	// L522
        #pragma HLS bind_op variable=v403 op=fadd impl=fabric
        int32_t v404 = il2;	// L523
        int v405 = v404;	// L524
        o_im2[v405] = v403;	// L525
        float v406 = a_re2;	// L526
        float v407 = b_re2;	// L527
        float v408 = v406 - v407;	// L528
        #pragma HLS bind_op variable=v408 op=fsub impl=fabric
        int32_t v409 = iu2;	// L529
        int v410 = v409;	// L530
        o_re2[v410] = v408;	// L531
        float v411 = a_im2;	// L532
        float v412 = b_im2;	// L533
        float v413 = v411 - v412;	// L534
        #pragma HLS bind_op variable=v413 op=fsub impl=fabric
        int32_t v414 = iu2;	// L535
        int v415 = v414;	// L536
        o_im2[v415] = v413;	// L537
      } else {
        int32_t v416 = tw_k2;	// L539
        bool v417 = v416 == 64;	// L540
        if (v417) {	// L541
          float v418 = a_re2;	// L542
          float v419 = b_im2;	// L543
          float v420 = v418 + v419;	// L544
          #pragma HLS bind_op variable=v420 op=fadd impl=fabric
          int32_t v421 = il2;	// L545
          int v422 = v421;	// L546
          o_re2[v422] = v420;	// L547
          float v423 = a_im2;	// L548
          float v424 = b_re2;	// L549
          float v425 = v423 - v424;	// L550
          #pragma HLS bind_op variable=v425 op=fsub impl=fabric
          int32_t v426 = il2;	// L551
          int v427 = v426;	// L552
          o_im2[v427] = v425;	// L553
          float v428 = a_re2;	// L554
          float v429 = b_im2;	// L555
          float v430 = v428 - v429;	// L556
          #pragma HLS bind_op variable=v430 op=fsub impl=fabric
          int32_t v431 = iu2;	// L557
          int v432 = v431;	// L558
          o_re2[v432] = v430;	// L559
          float v433 = a_im2;	// L560
          float v434 = b_re2;	// L561
          float v435 = v433 + v434;	// L562
          #pragma HLS bind_op variable=v435 op=fadd impl=fabric
          int32_t v436 = iu2;	// L563
          int v437 = v436;	// L564
          o_im2[v437] = v435;	// L565
        } else {
          int32_t v438 = tw_k2;	// L567
          int v439 = v438;	// L568
          float v440 = twr[v439];	// L569
          float tr2;	// L570
          tr2 = v440;	// L571
          int32_t v442 = tw_k2;	// L572
          int v443 = v442;	// L573
          float v444 = twi[v443];	// L574
          float ti2;	// L575
          ti2 = v444;	// L576
          float v446 = b_re2;	// L577
          float v447 = tr2;	// L578
          float v448 = v446 * v447;	// L579
          float v449 = b_im2;	// L580
          float v450 = ti2;	// L581
          float v451 = v449 * v450;	// L582
          float v452 = v448 - v451;	// L583
          float bw_re2;	// L584
          bw_re2 = v452;	// L585
          float v454 = b_re2;	// L586
          float v455 = ti2;	// L587
          float v456 = v454 * v455;	// L588
          float v457 = b_im2;	// L589
          float v458 = tr2;	// L590
          float v459 = v457 * v458;	// L591
          float v460 = v456 + v459;	// L592
          float bw_im2;	// L593
          bw_im2 = v460;	// L594
          float v462 = a_re2;	// L595
          float v463 = bw_re2;	// L596
          float v464 = v462 + v463;	// L597
          #pragma HLS bind_op variable=v464 op=fadd impl=fabric
          int32_t v465 = il2;	// L598
          int v466 = v465;	// L599
          o_re2[v466] = v464;	// L600
          float v467 = a_im2;	// L601
          float v468 = bw_im2;	// L602
          float v469 = v467 + v468;	// L603
          #pragma HLS bind_op variable=v469 op=fadd impl=fabric
          int32_t v470 = il2;	// L604
          int v471 = v470;	// L605
          o_im2[v471] = v469;	// L606
          float v472 = a_re2;	// L607
          float v473 = bw_re2;	// L608
          float v474 = v472 - v473;	// L609
          #pragma HLS bind_op variable=v474 op=fsub impl=fabric
          int32_t v475 = iu2;	// L610
          int v476 = v475;	// L611
          o_re2[v476] = v474;	// L612
          float v477 = a_im2;	// L613
          float v478 = bw_im2;	// L614
          float v479 = v477 - v478;	// L615
          #pragma HLS bind_op variable=v479 op=fsub impl=fabric
          int32_t v480 = iu2;	// L616
          int v481 = v480;	// L617
          o_im2[v481] = v479;	// L618
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
    }	// L622
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im2[_iv0];
      }
      v346.write(_vec);
    }	// L623
  }
}

void intra_3(
  hls::stream< hls::vector< float, 32 > >& v482,
  hls::stream< hls::vector< float, 32 > >& v483,
  hls::stream< hls::vector< float, 32 > >& v484,
  hls::stream< hls::vector< float, 32 > >& v485
) {	// L627
  // placeholder for const float twr	// L635
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L636
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i3: for (int _i3 = 0; _i3 < 8; _i3++) {	// L637
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v489 = v482.read();
    hls::vector< float, 32 > v490 = v483.read();
    float o_re3[32];	// L640
    #pragma HLS array_partition variable=o_re3 complete
    float o_im3[32];	// L641
    #pragma HLS array_partition variable=o_im3 complete
    int32_t stride3;	// L642
    stride3 = 8;	// L643
    l_S_k_0_k3: for (int k3 = 0; k3 < 16; k3++) {	// L644
    #pragma HLS unroll
      int v495 = k3 >> 3;	// L645
      int v496 = v495 << 4;	// L646
      int32_t v497 = stride3;	// L647
      int64_t v498 = v497;	// L648
      int64_t v499 = v498 - 1;	// L649
      int64_t v500 = k3;	// L650
      int64_t v501 = v500 & v499;	// L651
      int64_t v502 = v496;	// L652
      int64_t v503 = v502 | v501;	// L653
      int32_t v504 = v503;	// L654
      int32_t il3;	// L655
      il3 = v504;	// L656
      int32_t v506 = il3;	// L657
      int32_t v507 = stride3;	// L658
      int32_t v508 = v506 | v507;	// L659
      int32_t iu3;	// L660
      iu3 = v508;	// L661
      int32_t v510 = stride3;	// L662
      int64_t v511 = v510;	// L663
      int64_t v512 = v511 - 1;	// L664
      int64_t v513 = v500 & v512;	// L665
      int64_t v514 = v513 << 4;	// L666
      int32_t v515 = v514;	// L667
      int32_t tw_k3;	// L668
      tw_k3 = v515;	// L669
      int32_t v517 = il3;	// L670
      int v518 = v517;	// L671
      float v519 = v489[v518];	// L672
      float a_re3;	// L673
      a_re3 = v519;	// L674
      int32_t v521 = il3;	// L675
      int v522 = v521;	// L676
      float v523 = v490[v522];	// L677
      float a_im3;	// L678
      a_im3 = v523;	// L679
      int32_t v525 = iu3;	// L680
      int v526 = v525;	// L681
      float v527 = v489[v526];	// L682
      float b_re3;	// L683
      b_re3 = v527;	// L684
      int32_t v529 = iu3;	// L685
      int v530 = v529;	// L686
      float v531 = v490[v530];	// L687
      float b_im3;	// L688
      b_im3 = v531;	// L689
      int32_t v533 = tw_k3;	// L690
      bool v534 = v533 == 0;	// L691
      if (v534) {	// L692
        float v535 = a_re3;	// L693
        float v536 = b_re3;	// L694
        float v537 = v535 + v536;	// L695
        #pragma HLS bind_op variable=v537 op=fadd impl=fabric
        int32_t v538 = il3;	// L696
        int v539 = v538;	// L697
        o_re3[v539] = v537;	// L698
        float v540 = a_im3;	// L699
        float v541 = b_im3;	// L700
        float v542 = v540 + v541;	// L701
        #pragma HLS bind_op variable=v542 op=fadd impl=fabric
        int32_t v543 = il3;	// L702
        int v544 = v543;	// L703
        o_im3[v544] = v542;	// L704
        float v545 = a_re3;	// L705
        float v546 = b_re3;	// L706
        float v547 = v545 - v546;	// L707
        #pragma HLS bind_op variable=v547 op=fsub impl=fabric
        int32_t v548 = iu3;	// L708
        int v549 = v548;	// L709
        o_re3[v549] = v547;	// L710
        float v550 = a_im3;	// L711
        float v551 = b_im3;	// L712
        float v552 = v550 - v551;	// L713
        #pragma HLS bind_op variable=v552 op=fsub impl=fabric
        int32_t v553 = iu3;	// L714
        int v554 = v553;	// L715
        o_im3[v554] = v552;	// L716
      } else {
        int32_t v555 = tw_k3;	// L718
        bool v556 = v555 == 64;	// L719
        if (v556) {	// L720
          float v557 = a_re3;	// L721
          float v558 = b_im3;	// L722
          float v559 = v557 + v558;	// L723
          #pragma HLS bind_op variable=v559 op=fadd impl=fabric
          int32_t v560 = il3;	// L724
          int v561 = v560;	// L725
          o_re3[v561] = v559;	// L726
          float v562 = a_im3;	// L727
          float v563 = b_re3;	// L728
          float v564 = v562 - v563;	// L729
          #pragma HLS bind_op variable=v564 op=fsub impl=fabric
          int32_t v565 = il3;	// L730
          int v566 = v565;	// L731
          o_im3[v566] = v564;	// L732
          float v567 = a_re3;	// L733
          float v568 = b_im3;	// L734
          float v569 = v567 - v568;	// L735
          #pragma HLS bind_op variable=v569 op=fsub impl=fabric
          int32_t v570 = iu3;	// L736
          int v571 = v570;	// L737
          o_re3[v571] = v569;	// L738
          float v572 = a_im3;	// L739
          float v573 = b_re3;	// L740
          float v574 = v572 + v573;	// L741
          #pragma HLS bind_op variable=v574 op=fadd impl=fabric
          int32_t v575 = iu3;	// L742
          int v576 = v575;	// L743
          o_im3[v576] = v574;	// L744
        } else {
          int32_t v577 = tw_k3;	// L746
          int v578 = v577;	// L747
          float v579 = twr[v578];	// L748
          float tr3;	// L749
          tr3 = v579;	// L750
          int32_t v581 = tw_k3;	// L751
          int v582 = v581;	// L752
          float v583 = twi[v582];	// L753
          float ti3;	// L754
          ti3 = v583;	// L755
          float v585 = b_re3;	// L756
          float v586 = tr3;	// L757
          float v587 = v585 * v586;	// L758
          float v588 = b_im3;	// L759
          float v589 = ti3;	// L760
          float v590 = v588 * v589;	// L761
          float v591 = v587 - v590;	// L762
          float bw_re3;	// L763
          bw_re3 = v591;	// L764
          float v593 = b_re3;	// L765
          float v594 = ti3;	// L766
          float v595 = v593 * v594;	// L767
          float v596 = b_im3;	// L768
          float v597 = tr3;	// L769
          float v598 = v596 * v597;	// L770
          float v599 = v595 + v598;	// L771
          float bw_im3;	// L772
          bw_im3 = v599;	// L773
          float v601 = a_re3;	// L774
          float v602 = bw_re3;	// L775
          float v603 = v601 + v602;	// L776
          #pragma HLS bind_op variable=v603 op=fadd impl=fabric
          int32_t v604 = il3;	// L777
          int v605 = v604;	// L778
          o_re3[v605] = v603;	// L779
          float v606 = a_im3;	// L780
          float v607 = bw_im3;	// L781
          float v608 = v606 + v607;	// L782
          #pragma HLS bind_op variable=v608 op=fadd impl=fabric
          int32_t v609 = il3;	// L783
          int v610 = v609;	// L784
          o_im3[v610] = v608;	// L785
          float v611 = a_re3;	// L786
          float v612 = bw_re3;	// L787
          float v613 = v611 - v612;	// L788
          #pragma HLS bind_op variable=v613 op=fsub impl=fabric
          int32_t v614 = iu3;	// L789
          int v615 = v614;	// L790
          o_re3[v615] = v613;	// L791
          float v616 = a_im3;	// L792
          float v617 = bw_im3;	// L793
          float v618 = v616 - v617;	// L794
          #pragma HLS bind_op variable=v618 op=fsub impl=fabric
          int32_t v619 = iu3;	// L795
          int v620 = v619;	// L796
          o_im3[v620] = v618;	// L797
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
    }	// L801
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im3[_iv0];
      }
      v485.write(_vec);
    }	// L802
  }
}

void intra_4(
  hls::stream< hls::vector< float, 32 > >& v621,
  hls::stream< hls::vector< float, 32 > >& v622,
  hls::stream< hls::vector< float, 32 > >& v623,
  hls::stream< hls::vector< float, 32 > >& v624
) {	// L806
  // placeholder for const float twr	// L814
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L815
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i4: for (int _i4 = 0; _i4 < 8; _i4++) {	// L816
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v628 = v621.read();
    hls::vector< float, 32 > v629 = v622.read();
    float o_re4[32];	// L819
    #pragma HLS array_partition variable=o_re4 complete
    float o_im4[32];	// L820
    #pragma HLS array_partition variable=o_im4 complete
    int32_t stride4;	// L821
    stride4 = 16;	// L822
    l_S_k_0_k4: for (int k4 = 0; k4 < 16; k4++) {	// L823
    #pragma HLS unroll
      int v634 = k4 >> 4;	// L824
      int v635 = v634 << 5;	// L825
      int32_t v636 = stride4;	// L826
      int64_t v637 = v636;	// L827
      int64_t v638 = v637 - 1;	// L828
      int64_t v639 = k4;	// L829
      int64_t v640 = v639 & v638;	// L830
      int64_t v641 = v635;	// L831
      int64_t v642 = v641 | v640;	// L832
      int32_t v643 = v642;	// L833
      int32_t il4;	// L834
      il4 = v643;	// L835
      int32_t v645 = il4;	// L836
      int32_t v646 = stride4;	// L837
      int32_t v647 = v645 | v646;	// L838
      int32_t iu4;	// L839
      iu4 = v647;	// L840
      int32_t v649 = stride4;	// L841
      int64_t v650 = v649;	// L842
      int64_t v651 = v650 - 1;	// L843
      int64_t v652 = v639 & v651;	// L844
      int64_t v653 = v652 << 3;	// L845
      int32_t v654 = v653;	// L846
      int32_t tw_k4;	// L847
      tw_k4 = v654;	// L848
      int32_t v656 = il4;	// L849
      int v657 = v656;	// L850
      float v658 = v628[v657];	// L851
      float a_re4;	// L852
      a_re4 = v658;	// L853
      int32_t v660 = il4;	// L854
      int v661 = v660;	// L855
      float v662 = v629[v661];	// L856
      float a_im4;	// L857
      a_im4 = v662;	// L858
      int32_t v664 = iu4;	// L859
      int v665 = v664;	// L860
      float v666 = v628[v665];	// L861
      float b_re4;	// L862
      b_re4 = v666;	// L863
      int32_t v668 = iu4;	// L864
      int v669 = v668;	// L865
      float v670 = v629[v669];	// L866
      float b_im4;	// L867
      b_im4 = v670;	// L868
      int32_t v672 = tw_k4;	// L869
      bool v673 = v672 == 0;	// L870
      if (v673) {	// L871
        float v674 = a_re4;	// L872
        float v675 = b_re4;	// L873
        float v676 = v674 + v675;	// L874
        #pragma HLS bind_op variable=v676 op=fadd impl=fabric
        int32_t v677 = il4;	// L875
        int v678 = v677;	// L876
        o_re4[v678] = v676;	// L877
        float v679 = a_im4;	// L878
        float v680 = b_im4;	// L879
        float v681 = v679 + v680;	// L880
        #pragma HLS bind_op variable=v681 op=fadd impl=fabric
        int32_t v682 = il4;	// L881
        int v683 = v682;	// L882
        o_im4[v683] = v681;	// L883
        float v684 = a_re4;	// L884
        float v685 = b_re4;	// L885
        float v686 = v684 - v685;	// L886
        #pragma HLS bind_op variable=v686 op=fsub impl=fabric
        int32_t v687 = iu4;	// L887
        int v688 = v687;	// L888
        o_re4[v688] = v686;	// L889
        float v689 = a_im4;	// L890
        float v690 = b_im4;	// L891
        float v691 = v689 - v690;	// L892
        #pragma HLS bind_op variable=v691 op=fsub impl=fabric
        int32_t v692 = iu4;	// L893
        int v693 = v692;	// L894
        o_im4[v693] = v691;	// L895
      } else {
        int32_t v694 = tw_k4;	// L897
        bool v695 = v694 == 64;	// L898
        if (v695) {	// L899
          float v696 = a_re4;	// L900
          float v697 = b_im4;	// L901
          float v698 = v696 + v697;	// L902
          #pragma HLS bind_op variable=v698 op=fadd impl=fabric
          int32_t v699 = il4;	// L903
          int v700 = v699;	// L904
          o_re4[v700] = v698;	// L905
          float v701 = a_im4;	// L906
          float v702 = b_re4;	// L907
          float v703 = v701 - v702;	// L908
          #pragma HLS bind_op variable=v703 op=fsub impl=fabric
          int32_t v704 = il4;	// L909
          int v705 = v704;	// L910
          o_im4[v705] = v703;	// L911
          float v706 = a_re4;	// L912
          float v707 = b_im4;	// L913
          float v708 = v706 - v707;	// L914
          #pragma HLS bind_op variable=v708 op=fsub impl=fabric
          int32_t v709 = iu4;	// L915
          int v710 = v709;	// L916
          o_re4[v710] = v708;	// L917
          float v711 = a_im4;	// L918
          float v712 = b_re4;	// L919
          float v713 = v711 + v712;	// L920
          #pragma HLS bind_op variable=v713 op=fadd impl=fabric
          int32_t v714 = iu4;	// L921
          int v715 = v714;	// L922
          o_im4[v715] = v713;	// L923
        } else {
          int32_t v716 = tw_k4;	// L925
          int v717 = v716;	// L926
          float v718 = twr[v717];	// L927
          float tr4;	// L928
          tr4 = v718;	// L929
          int32_t v720 = tw_k4;	// L930
          int v721 = v720;	// L931
          float v722 = twi[v721];	// L932
          float ti4;	// L933
          ti4 = v722;	// L934
          float v724 = b_re4;	// L935
          float v725 = tr4;	// L936
          float v726 = v724 * v725;	// L937
          float v727 = b_im4;	// L938
          float v728 = ti4;	// L939
          float v729 = v727 * v728;	// L940
          float v730 = v726 - v729;	// L941
          float bw_re4;	// L942
          bw_re4 = v730;	// L943
          float v732 = b_re4;	// L944
          float v733 = ti4;	// L945
          float v734 = v732 * v733;	// L946
          float v735 = b_im4;	// L947
          float v736 = tr4;	// L948
          float v737 = v735 * v736;	// L949
          float v738 = v734 + v737;	// L950
          float bw_im4;	// L951
          bw_im4 = v738;	// L952
          float v740 = a_re4;	// L953
          float v741 = bw_re4;	// L954
          float v742 = v740 + v741;	// L955
          #pragma HLS bind_op variable=v742 op=fadd impl=fabric
          int32_t v743 = il4;	// L956
          int v744 = v743;	// L957
          o_re4[v744] = v742;	// L958
          float v745 = a_im4;	// L959
          float v746 = bw_im4;	// L960
          float v747 = v745 + v746;	// L961
          #pragma HLS bind_op variable=v747 op=fadd impl=fabric
          int32_t v748 = il4;	// L962
          int v749 = v748;	// L963
          o_im4[v749] = v747;	// L964
          float v750 = a_re4;	// L965
          float v751 = bw_re4;	// L966
          float v752 = v750 - v751;	// L967
          #pragma HLS bind_op variable=v752 op=fsub impl=fabric
          int32_t v753 = iu4;	// L968
          int v754 = v753;	// L969
          o_re4[v754] = v752;	// L970
          float v755 = a_im4;	// L971
          float v756 = bw_im4;	// L972
          float v757 = v755 - v756;	// L973
          #pragma HLS bind_op variable=v757 op=fsub impl=fabric
          int32_t v758 = iu4;	// L974
          int v759 = v758;	// L975
          o_im4[v759] = v757;	// L976
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
    }	// L980
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im4[_iv0];
      }
      v624.write(_vec);
    }	// L981
  }
}

void inter_0(
  hls::stream< hls::vector< float, 32 > >& v760,
  hls::stream< hls::vector< float, 32 > >& v761,
  hls::stream< hls::vector< float, 32 > >& v762,
  hls::stream< hls::vector< float, 32 > >& v763
) {	// L985
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L997
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L998
  #pragma HLS array_partition variable=twi complete
  float in_re[32][8];	// L999
  #pragma HLS array_partition variable=in_re complete dim=1

  #pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L1000
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L1001
  #pragma HLS array_partition variable=out_re_b complete dim=1

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L1002
  #pragma HLS array_partition variable=out_im_b complete dim=1

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L1003
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v771 = v760.read();
    hls::vector< float, 32 > v772 = v761.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L1006
    #pragma HLS unroll
      int32_t v774 = k5;	// L1007
      int32_t v775 = v774 & 15;	// L1008
      int v776 = k5 >> 4;	// L1009
      int32_t v777 = i;	// L1010
      int32_t v778 = v777 & 1;	// L1011
      int32_t v779 = v776;	// L1012
      int32_t v780 = v779 ^ v778;	// L1013
      int32_t v781 = v780 << 4;	// L1014
      int32_t v782 = v775 | v781;	// L1015
      int32_t bank1;	// L1016
      bank1 = v782;	// L1017
      float v784 = v771[k5];	// L1018
      int32_t v785 = bank1;	// L1019
      int v786 = v785;	// L1020
      in_re[v786][i] = v784;	// L1021
      float v787 = v772[k5];	// L1022
      int32_t v788 = bank1;	// L1023
      int v789 = v788;	// L1024
      in_im[v789][i] = v787;	// L1025
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L1028
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im inter false
  #pragma HLS dependence variable=in_im intra false
  #pragma HLS dependence variable=in_re inter false
  #pragma HLS dependence variable=in_re intra false
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L1029
    #pragma HLS unroll
      int v792 = i1 << 4;	// L1030
      int v793 = v792 | k6;	// L1031
      uint32_t v794 = v793;	// L1032
      uint32_t bg;	// L1033
      bg = v794;	// L1034
      int32_t v796 = bg;	// L1035
      int64_t v797 = v796;	// L1036
      int64_t v798 = v797 & 31;	// L1037
      int64_t v799 = v798 << 2;	// L1038
      uint32_t v800 = v799;	// L1039
      uint32_t tw_k5;	// L1040
      tw_k5 = v800;	// L1041
      int32_t v802 = i1;	// L1042
      int32_t v803 = v802 & 1;	// L1043
      int32_t v804 = v803 << 4;	// L1044
      int32_t v805 = k6;	// L1045
      int32_t v806 = v805 | v804;	// L1046
      uint32_t bank_il;	// L1047
      bank_il = v806;	// L1048
      int32_t v808 = bank_il;	// L1049
      int32_t v809 = v808 ^ 16;	// L1050
      uint32_t bank_iu;	// L1051
      bank_iu = v809;	// L1052
      int v811 = i1 >> 1;	// L1053
      uint32_t v812 = v811;	// L1054
      uint32_t i_shr;	// L1055
      i_shr = v812;	// L1056
      uint32_t low_mask;	// L1057
      low_mask = 0;	// L1058
      int32_t v815 = i_shr;	// L1059
      int32_t v816 = low_mask;	// L1060
      uint32_t v817 = v815 & v816;	// L1061
      uint32_t low_bits;	// L1062
      low_bits = v817;	// L1063
      int32_t v819 = i_shr;	// L1064
      uint32_t high_bits;	// L1065
      high_bits = v819;	// L1066
      int32_t v821 = high_bits;	// L1067
      uint32_t v822 = v821 << 1;	// L1068
      int32_t v823 = low_bits;	// L1069
      uint32_t v824 = v822 | v823;	// L1070
      uint32_t off_il;	// L1071
      off_il = v824;	// L1072
      uint32_t stride_off;	// L1073
      stride_off = 1;	// L1074
      int32_t v827 = off_il;	// L1075
      int32_t v828 = stride_off;	// L1076
      uint32_t v829 = v827 | v828;	// L1077
      uint32_t off_iu;	// L1078
      off_iu = v829;	// L1079
      int32_t v831 = bank_il;	// L1080
      int v832 = v831;	// L1081
      int32_t v833 = off_il;	// L1082
      int v834 = v833;	// L1083
      float v835 = in_re[v832][v834];	// L1084
      float a_re5;	// L1085
      a_re5 = v835;	// L1086
      int32_t v837 = bank_il;	// L1087
      int v838 = v837;	// L1088
      int32_t v839 = off_il;	// L1089
      int v840 = v839;	// L1090
      float v841 = in_im[v838][v840];	// L1091
      float a_im5;	// L1092
      a_im5 = v841;	// L1093
      int32_t v843 = bank_iu;	// L1094
      int v844 = v843;	// L1095
      int32_t v845 = off_iu;	// L1096
      int v846 = v845;	// L1097
      float v847 = in_re[v844][v846];	// L1098
      float b_re5;	// L1099
      b_re5 = v847;	// L1100
      int32_t v849 = bank_iu;	// L1101
      int v850 = v849;	// L1102
      int32_t v851 = off_iu;	// L1103
      int v852 = v851;	// L1104
      float v853 = in_im[v850][v852];	// L1105
      float b_im5;	// L1106
      b_im5 = v853;	// L1107
      int32_t v855 = tw_k5;	// L1108
      int64_t v856 = v855;	// L1109
      bool v857 = v856 == 0;	// L1110
      if (v857) {	// L1111
        float v858 = a_re5;	// L1112
        float v859 = b_re5;	// L1113
        float v860 = v858 + v859;	// L1114
        #pragma HLS bind_op variable=v860 op=fadd impl=fabric
        int32_t v861 = bank_il;	// L1115
        int v862 = v861;	// L1116
        int32_t v863 = off_il;	// L1117
        int v864 = v863;	// L1118
        out_re_b[v862][v864] = v860;	// L1119
        float v865 = a_im5;	// L1120
        float v866 = b_im5;	// L1121
        float v867 = v865 + v866;	// L1122
        #pragma HLS bind_op variable=v867 op=fadd impl=fabric
        int32_t v868 = bank_il;	// L1123
        int v869 = v868;	// L1124
        int32_t v870 = off_il;	// L1125
        int v871 = v870;	// L1126
        out_im_b[v869][v871] = v867;	// L1127
        float v872 = a_re5;	// L1128
        float v873 = b_re5;	// L1129
        float v874 = v872 - v873;	// L1130
        #pragma HLS bind_op variable=v874 op=fsub impl=fabric
        int32_t v875 = bank_iu;	// L1131
        int v876 = v875;	// L1132
        int32_t v877 = off_iu;	// L1133
        int v878 = v877;	// L1134
        out_re_b[v876][v878] = v874;	// L1135
        float v879 = a_im5;	// L1136
        float v880 = b_im5;	// L1137
        float v881 = v879 - v880;	// L1138
        #pragma HLS bind_op variable=v881 op=fsub impl=fabric
        int32_t v882 = bank_iu;	// L1139
        int v883 = v882;	// L1140
        int32_t v884 = off_iu;	// L1141
        int v885 = v884;	// L1142
        out_im_b[v883][v885] = v881;	// L1143
      } else {
        int32_t v886 = tw_k5;	// L1145
        int64_t v887 = v886;	// L1146
        bool v888 = v887 == 64;	// L1147
        if (v888) {	// L1148
          float v889 = a_re5;	// L1149
          float v890 = b_im5;	// L1150
          float v891 = v889 + v890;	// L1151
          #pragma HLS bind_op variable=v891 op=fadd impl=fabric
          int32_t v892 = bank_il;	// L1152
          int v893 = v892;	// L1153
          int32_t v894 = off_il;	// L1154
          int v895 = v894;	// L1155
          out_re_b[v893][v895] = v891;	// L1156
          float v896 = a_im5;	// L1157
          float v897 = b_re5;	// L1158
          float v898 = v896 - v897;	// L1159
          #pragma HLS bind_op variable=v898 op=fsub impl=fabric
          int32_t v899 = bank_il;	// L1160
          int v900 = v899;	// L1161
          int32_t v901 = off_il;	// L1162
          int v902 = v901;	// L1163
          out_im_b[v900][v902] = v898;	// L1164
          float v903 = a_re5;	// L1165
          float v904 = b_im5;	// L1166
          float v905 = v903 - v904;	// L1167
          #pragma HLS bind_op variable=v905 op=fsub impl=fabric
          int32_t v906 = bank_iu;	// L1168
          int v907 = v906;	// L1169
          int32_t v908 = off_iu;	// L1170
          int v909 = v908;	// L1171
          out_re_b[v907][v909] = v905;	// L1172
          float v910 = a_im5;	// L1173
          float v911 = b_re5;	// L1174
          float v912 = v910 + v911;	// L1175
          #pragma HLS bind_op variable=v912 op=fadd impl=fabric
          int32_t v913 = bank_iu;	// L1176
          int v914 = v913;	// L1177
          int32_t v915 = off_iu;	// L1178
          int v916 = v915;	// L1179
          out_im_b[v914][v916] = v912;	// L1180
        } else {
          int32_t v917 = tw_k5;	// L1182
          int v918 = v917;	// L1183
          float v919 = twr[v918];	// L1184
          float tr5;	// L1185
          tr5 = v919;	// L1186
          int32_t v921 = tw_k5;	// L1187
          int v922 = v921;	// L1188
          float v923 = twi[v922];	// L1189
          float ti5;	// L1190
          ti5 = v923;	// L1191
          float v925 = b_re5;	// L1192
          float v926 = tr5;	// L1193
          float v927 = v925 * v926;	// L1194
          float v928 = b_im5;	// L1195
          float v929 = ti5;	// L1196
          float v930 = v928 * v929;	// L1197
          float v931 = v927 - v930;	// L1198
          float bw_re5;	// L1199
          bw_re5 = v931;	// L1200
          float v933 = b_re5;	// L1201
          float v934 = ti5;	// L1202
          float v935 = v933 * v934;	// L1203
          float v936 = b_im5;	// L1204
          float v937 = tr5;	// L1205
          float v938 = v936 * v937;	// L1206
          float v939 = v935 + v938;	// L1207
          float bw_im5;	// L1208
          bw_im5 = v939;	// L1209
          float v941 = a_re5;	// L1210
          float v942 = bw_re5;	// L1211
          float v943 = v941 + v942;	// L1212
          #pragma HLS bind_op variable=v943 op=fadd impl=fabric
          int32_t v944 = bank_il;	// L1213
          int v945 = v944;	// L1214
          int32_t v946 = off_il;	// L1215
          int v947 = v946;	// L1216
          out_re_b[v945][v947] = v943;	// L1217
          float v948 = a_im5;	// L1218
          float v949 = bw_im5;	// L1219
          float v950 = v948 + v949;	// L1220
          #pragma HLS bind_op variable=v950 op=fadd impl=fabric
          int32_t v951 = bank_il;	// L1221
          int v952 = v951;	// L1222
          int32_t v953 = off_il;	// L1223
          int v954 = v953;	// L1224
          out_im_b[v952][v954] = v950;	// L1225
          float v955 = a_re5;	// L1226
          float v956 = bw_re5;	// L1227
          float v957 = v955 - v956;	// L1228
          #pragma HLS bind_op variable=v957 op=fsub impl=fabric
          int32_t v958 = bank_iu;	// L1229
          int v959 = v958;	// L1230
          int32_t v960 = off_iu;	// L1231
          int v961 = v960;	// L1232
          out_re_b[v959][v961] = v957;	// L1233
          float v962 = a_im5;	// L1234
          float v963 = bw_im5;	// L1235
          float v964 = v962 - v963;	// L1236
          #pragma HLS bind_op variable=v964 op=fsub impl=fabric
          int32_t v965 = bank_iu;	// L1237
          int v966 = v965;	// L1238
          int32_t v967 = off_iu;	// L1239
          int v968 = v967;	// L1240
          out_im_b[v966][v968] = v964;	// L1241
        }
      }
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L1246
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    float chunk_re_out[32];	// L1247
    #pragma HLS array_partition variable=chunk_re_out complete
    float chunk_im_out[32];	// L1248
    #pragma HLS array_partition variable=chunk_im_out complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L1249
    #pragma HLS unroll
      int32_t v973 = k7;	// L1250
      int32_t v974 = v973 & 15;	// L1251
      int v975 = k7 >> 4;	// L1252
      int32_t v976 = i2;	// L1253
      int32_t v977 = v976 & 1;	// L1254
      int32_t v978 = v975;	// L1255
      int32_t v979 = v978 ^ v977;	// L1256
      int32_t v980 = v979 << 4;	// L1257
      int32_t v981 = v974 | v980;	// L1258
      int32_t bank2;	// L1259
      bank2 = v981;	// L1260
      int32_t v983 = bank2;	// L1261
      int v984 = v983;	// L1262
      float v985 = out_re_b[v984][i2];	// L1263
      chunk_re_out[k7] = v985;	// L1264
      int32_t v986 = bank2;	// L1265
      int v987 = v986;	// L1266
      float v988 = out_im_b[v987][i2];	// L1267
      chunk_im_out[k7] = v988;	// L1268
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out[_iv0];
      }
      v762.write(_vec);
    }	// L1270
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out[_iv0];
      }
      v763.write(_vec);
    }	// L1271
  }
}

void inter_1(
  hls::stream< hls::vector< float, 32 > >& v989,
  hls::stream< hls::vector< float, 32 > >& v990,
  hls::stream< hls::vector< float, 32 > >& v991,
  hls::stream< hls::vector< float, 32 > >& v992
) {	// L1275
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1287
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1288
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L1289
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L1290
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L1291
  #pragma HLS array_partition variable=out_re_b1 complete dim=1

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L1292
  #pragma HLS array_partition variable=out_im_b1 complete dim=1

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L1293
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1000 = v989.read();
    hls::vector< float, 32 > v1001 = v990.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L1296
    #pragma HLS unroll
      int32_t v1003 = k8;	// L1297
      int32_t v1004 = v1003 & 15;	// L1298
      int v1005 = k8 >> 4;	// L1299
      int v1006 = i3 >> 1;	// L1300
      int32_t v1007 = v1006;	// L1301
      int32_t v1008 = v1007 & 1;	// L1302
      int32_t v1009 = v1005;	// L1303
      int32_t v1010 = v1009 ^ v1008;	// L1304
      int32_t v1011 = v1010 << 4;	// L1305
      int32_t v1012 = v1004 | v1011;	// L1306
      int32_t bank3;	// L1307
      bank3 = v1012;	// L1308
      float v1014 = v1000[k8];	// L1309
      int32_t v1015 = bank3;	// L1310
      int v1016 = v1015;	// L1311
      in_re1[v1016][i3] = v1014;	// L1312
      float v1017 = v1001[k8];	// L1313
      int32_t v1018 = bank3;	// L1314
      int v1019 = v1018;	// L1315
      in_im1[v1019][i3] = v1017;	// L1316
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L1319
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im1 inter false
  #pragma HLS dependence variable=in_im1 intra false
  #pragma HLS dependence variable=in_re1 inter false
  #pragma HLS dependence variable=in_re1 intra false
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L1320
    #pragma HLS unroll
      int v1022 = i4 << 4;	// L1321
      int v1023 = v1022 | k9;	// L1322
      uint32_t v1024 = v1023;	// L1323
      uint32_t bg1;	// L1324
      bg1 = v1024;	// L1325
      int32_t v1026 = bg1;	// L1326
      int64_t v1027 = v1026;	// L1327
      int64_t v1028 = v1027 & 63;	// L1328
      int64_t v1029 = v1028 << 1;	// L1329
      uint32_t v1030 = v1029;	// L1330
      uint32_t tw_k6;	// L1331
      tw_k6 = v1030;	// L1332
      int32_t v1032 = i4;	// L1333
      int32_t v1033 = v1032 & 1;	// L1334
      int32_t v1034 = v1033 << 4;	// L1335
      int32_t v1035 = k9;	// L1336
      int32_t v1036 = v1035 | v1034;	// L1337
      uint32_t bank_il1;	// L1338
      bank_il1 = v1036;	// L1339
      int32_t v1038 = bank_il1;	// L1340
      int32_t v1039 = v1038 ^ 16;	// L1341
      uint32_t bank_iu1;	// L1342
      bank_iu1 = v1039;	// L1343
      int v1041 = i4 >> 1;	// L1344
      uint32_t v1042 = v1041;	// L1345
      uint32_t i_shr1;	// L1346
      i_shr1 = v1042;	// L1347
      uint32_t low_mask1;	// L1348
      low_mask1 = 1;	// L1349
      int32_t v1045 = i_shr1;	// L1350
      int32_t v1046 = low_mask1;	// L1351
      uint32_t v1047 = v1045 & v1046;	// L1352
      uint32_t low_bits1;	// L1353
      low_bits1 = v1047;	// L1354
      int32_t v1049 = i_shr1;	// L1355
      uint32_t v1050 = v1049 >> 1;	// L1356
      uint32_t high_bits1;	// L1357
      high_bits1 = v1050;	// L1358
      int32_t v1052 = high_bits1;	// L1359
      uint32_t v1053 = v1052 << 2;	// L1360
      int32_t v1054 = low_bits1;	// L1361
      uint32_t v1055 = v1053 | v1054;	// L1362
      uint32_t off_il1;	// L1363
      off_il1 = v1055;	// L1364
      uint32_t stride_off1;	// L1365
      stride_off1 = 2;	// L1366
      int32_t v1058 = off_il1;	// L1367
      int32_t v1059 = stride_off1;	// L1368
      uint32_t v1060 = v1058 | v1059;	// L1369
      uint32_t off_iu1;	// L1370
      off_iu1 = v1060;	// L1371
      int32_t v1062 = bank_il1;	// L1372
      int v1063 = v1062;	// L1373
      int32_t v1064 = off_il1;	// L1374
      int v1065 = v1064;	// L1375
      float v1066 = in_re1[v1063][v1065];	// L1376
      float a_re6;	// L1377
      a_re6 = v1066;	// L1378
      int32_t v1068 = bank_il1;	// L1379
      int v1069 = v1068;	// L1380
      int32_t v1070 = off_il1;	// L1381
      int v1071 = v1070;	// L1382
      float v1072 = in_im1[v1069][v1071];	// L1383
      float a_im6;	// L1384
      a_im6 = v1072;	// L1385
      int32_t v1074 = bank_iu1;	// L1386
      int v1075 = v1074;	// L1387
      int32_t v1076 = off_iu1;	// L1388
      int v1077 = v1076;	// L1389
      float v1078 = in_re1[v1075][v1077];	// L1390
      float b_re6;	// L1391
      b_re6 = v1078;	// L1392
      int32_t v1080 = bank_iu1;	// L1393
      int v1081 = v1080;	// L1394
      int32_t v1082 = off_iu1;	// L1395
      int v1083 = v1082;	// L1396
      float v1084 = in_im1[v1081][v1083];	// L1397
      float b_im6;	// L1398
      b_im6 = v1084;	// L1399
      int32_t v1086 = tw_k6;	// L1400
      int64_t v1087 = v1086;	// L1401
      bool v1088 = v1087 == 0;	// L1402
      if (v1088) {	// L1403
        float v1089 = a_re6;	// L1404
        float v1090 = b_re6;	// L1405
        float v1091 = v1089 + v1090;	// L1406
        #pragma HLS bind_op variable=v1091 op=fadd impl=fabric
        int32_t v1092 = bank_il1;	// L1407
        int v1093 = v1092;	// L1408
        int32_t v1094 = off_il1;	// L1409
        int v1095 = v1094;	// L1410
        out_re_b1[v1093][v1095] = v1091;	// L1411
        float v1096 = a_im6;	// L1412
        float v1097 = b_im6;	// L1413
        float v1098 = v1096 + v1097;	// L1414
        #pragma HLS bind_op variable=v1098 op=fadd impl=fabric
        int32_t v1099 = bank_il1;	// L1415
        int v1100 = v1099;	// L1416
        int32_t v1101 = off_il1;	// L1417
        int v1102 = v1101;	// L1418
        out_im_b1[v1100][v1102] = v1098;	// L1419
        float v1103 = a_re6;	// L1420
        float v1104 = b_re6;	// L1421
        float v1105 = v1103 - v1104;	// L1422
        #pragma HLS bind_op variable=v1105 op=fsub impl=fabric
        int32_t v1106 = bank_iu1;	// L1423
        int v1107 = v1106;	// L1424
        int32_t v1108 = off_iu1;	// L1425
        int v1109 = v1108;	// L1426
        out_re_b1[v1107][v1109] = v1105;	// L1427
        float v1110 = a_im6;	// L1428
        float v1111 = b_im6;	// L1429
        float v1112 = v1110 - v1111;	// L1430
        #pragma HLS bind_op variable=v1112 op=fsub impl=fabric
        int32_t v1113 = bank_iu1;	// L1431
        int v1114 = v1113;	// L1432
        int32_t v1115 = off_iu1;	// L1433
        int v1116 = v1115;	// L1434
        out_im_b1[v1114][v1116] = v1112;	// L1435
      } else {
        int32_t v1117 = tw_k6;	// L1437
        int64_t v1118 = v1117;	// L1438
        bool v1119 = v1118 == 64;	// L1439
        if (v1119) {	// L1440
          float v1120 = a_re6;	// L1441
          float v1121 = b_im6;	// L1442
          float v1122 = v1120 + v1121;	// L1443
          #pragma HLS bind_op variable=v1122 op=fadd impl=fabric
          int32_t v1123 = bank_il1;	// L1444
          int v1124 = v1123;	// L1445
          int32_t v1125 = off_il1;	// L1446
          int v1126 = v1125;	// L1447
          out_re_b1[v1124][v1126] = v1122;	// L1448
          float v1127 = a_im6;	// L1449
          float v1128 = b_re6;	// L1450
          float v1129 = v1127 - v1128;	// L1451
          #pragma HLS bind_op variable=v1129 op=fsub impl=fabric
          int32_t v1130 = bank_il1;	// L1452
          int v1131 = v1130;	// L1453
          int32_t v1132 = off_il1;	// L1454
          int v1133 = v1132;	// L1455
          out_im_b1[v1131][v1133] = v1129;	// L1456
          float v1134 = a_re6;	// L1457
          float v1135 = b_im6;	// L1458
          float v1136 = v1134 - v1135;	// L1459
          #pragma HLS bind_op variable=v1136 op=fsub impl=fabric
          int32_t v1137 = bank_iu1;	// L1460
          int v1138 = v1137;	// L1461
          int32_t v1139 = off_iu1;	// L1462
          int v1140 = v1139;	// L1463
          out_re_b1[v1138][v1140] = v1136;	// L1464
          float v1141 = a_im6;	// L1465
          float v1142 = b_re6;	// L1466
          float v1143 = v1141 + v1142;	// L1467
          #pragma HLS bind_op variable=v1143 op=fadd impl=fabric
          int32_t v1144 = bank_iu1;	// L1468
          int v1145 = v1144;	// L1469
          int32_t v1146 = off_iu1;	// L1470
          int v1147 = v1146;	// L1471
          out_im_b1[v1145][v1147] = v1143;	// L1472
        } else {
          int32_t v1148 = tw_k6;	// L1474
          int v1149 = v1148;	// L1475
          float v1150 = twr[v1149];	// L1476
          float tr6;	// L1477
          tr6 = v1150;	// L1478
          int32_t v1152 = tw_k6;	// L1479
          int v1153 = v1152;	// L1480
          float v1154 = twi[v1153];	// L1481
          float ti6;	// L1482
          ti6 = v1154;	// L1483
          float v1156 = b_re6;	// L1484
          float v1157 = tr6;	// L1485
          float v1158 = v1156 * v1157;	// L1486
          float v1159 = b_im6;	// L1487
          float v1160 = ti6;	// L1488
          float v1161 = v1159 * v1160;	// L1489
          float v1162 = v1158 - v1161;	// L1490
          float bw_re6;	// L1491
          bw_re6 = v1162;	// L1492
          float v1164 = b_re6;	// L1493
          float v1165 = ti6;	// L1494
          float v1166 = v1164 * v1165;	// L1495
          float v1167 = b_im6;	// L1496
          float v1168 = tr6;	// L1497
          float v1169 = v1167 * v1168;	// L1498
          float v1170 = v1166 + v1169;	// L1499
          float bw_im6;	// L1500
          bw_im6 = v1170;	// L1501
          float v1172 = a_re6;	// L1502
          float v1173 = bw_re6;	// L1503
          float v1174 = v1172 + v1173;	// L1504
          #pragma HLS bind_op variable=v1174 op=fadd impl=fabric
          int32_t v1175 = bank_il1;	// L1505
          int v1176 = v1175;	// L1506
          int32_t v1177 = off_il1;	// L1507
          int v1178 = v1177;	// L1508
          out_re_b1[v1176][v1178] = v1174;	// L1509
          float v1179 = a_im6;	// L1510
          float v1180 = bw_im6;	// L1511
          float v1181 = v1179 + v1180;	// L1512
          #pragma HLS bind_op variable=v1181 op=fadd impl=fabric
          int32_t v1182 = bank_il1;	// L1513
          int v1183 = v1182;	// L1514
          int32_t v1184 = off_il1;	// L1515
          int v1185 = v1184;	// L1516
          out_im_b1[v1183][v1185] = v1181;	// L1517
          float v1186 = a_re6;	// L1518
          float v1187 = bw_re6;	// L1519
          float v1188 = v1186 - v1187;	// L1520
          #pragma HLS bind_op variable=v1188 op=fsub impl=fabric
          int32_t v1189 = bank_iu1;	// L1521
          int v1190 = v1189;	// L1522
          int32_t v1191 = off_iu1;	// L1523
          int v1192 = v1191;	// L1524
          out_re_b1[v1190][v1192] = v1188;	// L1525
          float v1193 = a_im6;	// L1526
          float v1194 = bw_im6;	// L1527
          float v1195 = v1193 - v1194;	// L1528
          #pragma HLS bind_op variable=v1195 op=fsub impl=fabric
          int32_t v1196 = bank_iu1;	// L1529
          int v1197 = v1196;	// L1530
          int32_t v1198 = off_iu1;	// L1531
          int v1199 = v1198;	// L1532
          out_im_b1[v1197][v1199] = v1195;	// L1533
        }
      }
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L1538
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    float chunk_re_out1[32];	// L1539
    #pragma HLS array_partition variable=chunk_re_out1 complete
    float chunk_im_out1[32];	// L1540
    #pragma HLS array_partition variable=chunk_im_out1 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L1541
    #pragma HLS unroll
      int32_t v1204 = k10;	// L1542
      int32_t v1205 = v1204 & 15;	// L1543
      int v1206 = k10 >> 4;	// L1544
      int v1207 = i5 >> 1;	// L1545
      int32_t v1208 = v1207;	// L1546
      int32_t v1209 = v1208 & 1;	// L1547
      int32_t v1210 = v1206;	// L1548
      int32_t v1211 = v1210 ^ v1209;	// L1549
      int32_t v1212 = v1211 << 4;	// L1550
      int32_t v1213 = v1205 | v1212;	// L1551
      int32_t bank4;	// L1552
      bank4 = v1213;	// L1553
      int32_t v1215 = bank4;	// L1554
      int v1216 = v1215;	// L1555
      float v1217 = out_re_b1[v1216][i5];	// L1556
      chunk_re_out1[k10] = v1217;	// L1557
      int32_t v1218 = bank4;	// L1558
      int v1219 = v1218;	// L1559
      float v1220 = out_im_b1[v1219][i5];	// L1560
      chunk_im_out1[k10] = v1220;	// L1561
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out1[_iv0];
      }
      v991.write(_vec);
    }	// L1563
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out1[_iv0];
      }
      v992.write(_vec);
    }	// L1564
  }
}

void inter_2(
  hls::stream< hls::vector< float, 32 > >& v1221,
  hls::stream< hls::vector< float, 32 > >& v1222,
  hls::stream< hls::vector< float, 32 > >& v1223,
  hls::stream< hls::vector< float, 32 > >& v1224
) {	// L1568
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1581
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1582
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L1583
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L1584
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L1585
  #pragma HLS array_partition variable=out_re_b2 complete dim=1

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L1586
  #pragma HLS array_partition variable=out_im_b2 complete dim=1

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L1587
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1232 = v1221.read();
    hls::vector< float, 32 > v1233 = v1222.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L1590
    #pragma HLS unroll
      int32_t v1235 = k11;	// L1591
      int32_t v1236 = v1235 & 15;	// L1592
      int v1237 = k11 >> 4;	// L1593
      int v1238 = i6 >> 2;	// L1594
      int32_t v1239 = v1238;	// L1595
      int32_t v1240 = v1239 & 1;	// L1596
      int32_t v1241 = v1237;	// L1597
      int32_t v1242 = v1241 ^ v1240;	// L1598
      int32_t v1243 = v1242 << 4;	// L1599
      int32_t v1244 = v1236 | v1243;	// L1600
      int32_t bank5;	// L1601
      bank5 = v1244;	// L1602
      float v1246 = v1232[k11];	// L1603
      int32_t v1247 = bank5;	// L1604
      int v1248 = v1247;	// L1605
      in_re2[v1248][i6] = v1246;	// L1606
      float v1249 = v1233[k11];	// L1607
      int32_t v1250 = bank5;	// L1608
      int v1251 = v1250;	// L1609
      in_im2[v1251][i6] = v1249;	// L1610
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L1613
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im2 inter false
  #pragma HLS dependence variable=in_im2 intra false
  #pragma HLS dependence variable=in_re2 inter false
  #pragma HLS dependence variable=in_re2 intra false
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L1614
    #pragma HLS unroll
      int v1254 = i7 << 4;	// L1615
      int v1255 = v1254 | k12;	// L1616
      uint32_t v1256 = v1255;	// L1617
      uint32_t bg2;	// L1618
      bg2 = v1256;	// L1619
      int32_t v1258 = bg2;	// L1620
      int64_t v1259 = v1258;	// L1621
      int64_t v1260 = v1259 & 127;	// L1622
      uint32_t v1261 = v1260;	// L1623
      uint32_t tw_k7;	// L1624
      tw_k7 = v1261;	// L1625
      int32_t v1263 = i7;	// L1626
      int32_t v1264 = v1263 & 1;	// L1627
      int32_t v1265 = v1264 << 4;	// L1628
      int32_t v1266 = k12;	// L1629
      int32_t v1267 = v1266 | v1265;	// L1630
      uint32_t bank_il2;	// L1631
      bank_il2 = v1267;	// L1632
      int32_t v1269 = bank_il2;	// L1633
      int32_t v1270 = v1269 ^ 16;	// L1634
      uint32_t bank_iu2;	// L1635
      bank_iu2 = v1270;	// L1636
      int v1272 = i7 >> 1;	// L1637
      uint32_t v1273 = v1272;	// L1638
      uint32_t i_shr2;	// L1639
      i_shr2 = v1273;	// L1640
      uint32_t low_mask2;	// L1641
      low_mask2 = 3;	// L1642
      int32_t v1276 = i_shr2;	// L1643
      int32_t v1277 = low_mask2;	// L1644
      uint32_t v1278 = v1276 & v1277;	// L1645
      uint32_t low_bits2;	// L1646
      low_bits2 = v1278;	// L1647
      int32_t v1280 = i_shr2;	// L1648
      uint32_t v1281 = v1280 >> 2;	// L1649
      uint32_t high_bits2;	// L1650
      high_bits2 = v1281;	// L1651
      int32_t v1283 = high_bits2;	// L1652
      uint32_t v1284 = v1283 << 3;	// L1653
      int32_t v1285 = low_bits2;	// L1654
      uint32_t v1286 = v1284 | v1285;	// L1655
      uint32_t off_il2;	// L1656
      off_il2 = v1286;	// L1657
      uint32_t stride_off2;	// L1658
      stride_off2 = 4;	// L1659
      int32_t v1289 = off_il2;	// L1660
      int32_t v1290 = stride_off2;	// L1661
      uint32_t v1291 = v1289 | v1290;	// L1662
      uint32_t off_iu2;	// L1663
      off_iu2 = v1291;	// L1664
      int32_t v1293 = bank_il2;	// L1665
      int v1294 = v1293;	// L1666
      int32_t v1295 = off_il2;	// L1667
      int v1296 = v1295;	// L1668
      float v1297 = in_re2[v1294][v1296];	// L1669
      float a_re7;	// L1670
      a_re7 = v1297;	// L1671
      int32_t v1299 = bank_il2;	// L1672
      int v1300 = v1299;	// L1673
      int32_t v1301 = off_il2;	// L1674
      int v1302 = v1301;	// L1675
      float v1303 = in_im2[v1300][v1302];	// L1676
      float a_im7;	// L1677
      a_im7 = v1303;	// L1678
      int32_t v1305 = bank_iu2;	// L1679
      int v1306 = v1305;	// L1680
      int32_t v1307 = off_iu2;	// L1681
      int v1308 = v1307;	// L1682
      float v1309 = in_re2[v1306][v1308];	// L1683
      float b_re7;	// L1684
      b_re7 = v1309;	// L1685
      int32_t v1311 = bank_iu2;	// L1686
      int v1312 = v1311;	// L1687
      int32_t v1313 = off_iu2;	// L1688
      int v1314 = v1313;	// L1689
      float v1315 = in_im2[v1312][v1314];	// L1690
      float b_im7;	// L1691
      b_im7 = v1315;	// L1692
      int32_t v1317 = tw_k7;	// L1693
      int64_t v1318 = v1317;	// L1694
      bool v1319 = v1318 == 0;	// L1695
      if (v1319) {	// L1696
        float v1320 = a_re7;	// L1697
        float v1321 = b_re7;	// L1698
        float v1322 = v1320 + v1321;	// L1699
        #pragma HLS bind_op variable=v1322 op=fadd impl=fabric
        int32_t v1323 = bank_il2;	// L1700
        int v1324 = v1323;	// L1701
        int32_t v1325 = off_il2;	// L1702
        int v1326 = v1325;	// L1703
        out_re_b2[v1324][v1326] = v1322;	// L1704
        float v1327 = a_im7;	// L1705
        float v1328 = b_im7;	// L1706
        float v1329 = v1327 + v1328;	// L1707
        #pragma HLS bind_op variable=v1329 op=fadd impl=fabric
        int32_t v1330 = bank_il2;	// L1708
        int v1331 = v1330;	// L1709
        int32_t v1332 = off_il2;	// L1710
        int v1333 = v1332;	// L1711
        out_im_b2[v1331][v1333] = v1329;	// L1712
        float v1334 = a_re7;	// L1713
        float v1335 = b_re7;	// L1714
        float v1336 = v1334 - v1335;	// L1715
        #pragma HLS bind_op variable=v1336 op=fsub impl=fabric
        int32_t v1337 = bank_iu2;	// L1716
        int v1338 = v1337;	// L1717
        int32_t v1339 = off_iu2;	// L1718
        int v1340 = v1339;	// L1719
        out_re_b2[v1338][v1340] = v1336;	// L1720
        float v1341 = a_im7;	// L1721
        float v1342 = b_im7;	// L1722
        float v1343 = v1341 - v1342;	// L1723
        #pragma HLS bind_op variable=v1343 op=fsub impl=fabric
        int32_t v1344 = bank_iu2;	// L1724
        int v1345 = v1344;	// L1725
        int32_t v1346 = off_iu2;	// L1726
        int v1347 = v1346;	// L1727
        out_im_b2[v1345][v1347] = v1343;	// L1728
      } else {
        int32_t v1348 = tw_k7;	// L1730
        int64_t v1349 = v1348;	// L1731
        bool v1350 = v1349 == 64;	// L1732
        if (v1350) {	// L1733
          float v1351 = a_re7;	// L1734
          float v1352 = b_im7;	// L1735
          float v1353 = v1351 + v1352;	// L1736
          #pragma HLS bind_op variable=v1353 op=fadd impl=fabric
          int32_t v1354 = bank_il2;	// L1737
          int v1355 = v1354;	// L1738
          int32_t v1356 = off_il2;	// L1739
          int v1357 = v1356;	// L1740
          out_re_b2[v1355][v1357] = v1353;	// L1741
          float v1358 = a_im7;	// L1742
          float v1359 = b_re7;	// L1743
          float v1360 = v1358 - v1359;	// L1744
          #pragma HLS bind_op variable=v1360 op=fsub impl=fabric
          int32_t v1361 = bank_il2;	// L1745
          int v1362 = v1361;	// L1746
          int32_t v1363 = off_il2;	// L1747
          int v1364 = v1363;	// L1748
          out_im_b2[v1362][v1364] = v1360;	// L1749
          float v1365 = a_re7;	// L1750
          float v1366 = b_im7;	// L1751
          float v1367 = v1365 - v1366;	// L1752
          #pragma HLS bind_op variable=v1367 op=fsub impl=fabric
          int32_t v1368 = bank_iu2;	// L1753
          int v1369 = v1368;	// L1754
          int32_t v1370 = off_iu2;	// L1755
          int v1371 = v1370;	// L1756
          out_re_b2[v1369][v1371] = v1367;	// L1757
          float v1372 = a_im7;	// L1758
          float v1373 = b_re7;	// L1759
          float v1374 = v1372 + v1373;	// L1760
          #pragma HLS bind_op variable=v1374 op=fadd impl=fabric
          int32_t v1375 = bank_iu2;	// L1761
          int v1376 = v1375;	// L1762
          int32_t v1377 = off_iu2;	// L1763
          int v1378 = v1377;	// L1764
          out_im_b2[v1376][v1378] = v1374;	// L1765
        } else {
          int32_t v1379 = tw_k7;	// L1767
          int v1380 = v1379;	// L1768
          float v1381 = twr[v1380];	// L1769
          float tr7;	// L1770
          tr7 = v1381;	// L1771
          int32_t v1383 = tw_k7;	// L1772
          int v1384 = v1383;	// L1773
          float v1385 = twi[v1384];	// L1774
          float ti7;	// L1775
          ti7 = v1385;	// L1776
          float v1387 = b_re7;	// L1777
          float v1388 = tr7;	// L1778
          float v1389 = v1387 * v1388;	// L1779
          float v1390 = b_im7;	// L1780
          float v1391 = ti7;	// L1781
          float v1392 = v1390 * v1391;	// L1782
          float v1393 = v1389 - v1392;	// L1783
          float bw_re7;	// L1784
          bw_re7 = v1393;	// L1785
          float v1395 = b_re7;	// L1786
          float v1396 = ti7;	// L1787
          float v1397 = v1395 * v1396;	// L1788
          float v1398 = b_im7;	// L1789
          float v1399 = tr7;	// L1790
          float v1400 = v1398 * v1399;	// L1791
          float v1401 = v1397 + v1400;	// L1792
          float bw_im7;	// L1793
          bw_im7 = v1401;	// L1794
          float v1403 = a_re7;	// L1795
          float v1404 = bw_re7;	// L1796
          float v1405 = v1403 + v1404;	// L1797
          #pragma HLS bind_op variable=v1405 op=fadd impl=fabric
          int32_t v1406 = bank_il2;	// L1798
          int v1407 = v1406;	// L1799
          int32_t v1408 = off_il2;	// L1800
          int v1409 = v1408;	// L1801
          out_re_b2[v1407][v1409] = v1405;	// L1802
          float v1410 = a_im7;	// L1803
          float v1411 = bw_im7;	// L1804
          float v1412 = v1410 + v1411;	// L1805
          #pragma HLS bind_op variable=v1412 op=fadd impl=fabric
          int32_t v1413 = bank_il2;	// L1806
          int v1414 = v1413;	// L1807
          int32_t v1415 = off_il2;	// L1808
          int v1416 = v1415;	// L1809
          out_im_b2[v1414][v1416] = v1412;	// L1810
          float v1417 = a_re7;	// L1811
          float v1418 = bw_re7;	// L1812
          float v1419 = v1417 - v1418;	// L1813
          #pragma HLS bind_op variable=v1419 op=fsub impl=fabric
          int32_t v1420 = bank_iu2;	// L1814
          int v1421 = v1420;	// L1815
          int32_t v1422 = off_iu2;	// L1816
          int v1423 = v1422;	// L1817
          out_re_b2[v1421][v1423] = v1419;	// L1818
          float v1424 = a_im7;	// L1819
          float v1425 = bw_im7;	// L1820
          float v1426 = v1424 - v1425;	// L1821
          #pragma HLS bind_op variable=v1426 op=fsub impl=fabric
          int32_t v1427 = bank_iu2;	// L1822
          int v1428 = v1427;	// L1823
          int32_t v1429 = off_iu2;	// L1824
          int v1430 = v1429;	// L1825
          out_im_b2[v1428][v1430] = v1426;	// L1826
        }
      }
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1831
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    float chunk_re_out2[32];	// L1832
    #pragma HLS array_partition variable=chunk_re_out2 complete
    float chunk_im_out2[32];	// L1833
    #pragma HLS array_partition variable=chunk_im_out2 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1834
    #pragma HLS unroll
      int32_t v1435 = k13;	// L1835
      int32_t v1436 = v1435 & 15;	// L1836
      int v1437 = k13 >> 4;	// L1837
      int v1438 = i8 >> 2;	// L1838
      int32_t v1439 = v1438;	// L1839
      int32_t v1440 = v1439 & 1;	// L1840
      int32_t v1441 = v1437;	// L1841
      int32_t v1442 = v1441 ^ v1440;	// L1842
      int32_t v1443 = v1442 << 4;	// L1843
      int32_t v1444 = v1436 | v1443;	// L1844
      int32_t bank6;	// L1845
      bank6 = v1444;	// L1846
      int32_t v1446 = bank6;	// L1847
      int v1447 = v1446;	// L1848
      float v1448 = out_re_b2[v1447][i8];	// L1849
      chunk_re_out2[k13] = v1448;	// L1850
      int32_t v1449 = bank6;	// L1851
      int v1450 = v1449;	// L1852
      float v1451 = out_im_b2[v1450][i8];	// L1853
      chunk_im_out2[k13] = v1451;	// L1854
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out2[_iv0];
      }
      v1223.write(_vec);
    }	// L1856
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out2[_iv0];
      }
      v1224.write(_vec);
    }	// L1857
  }
}

void output_stage_0(
  hls::stream< hls::vector< float, 32 > >& v1452,
  hls::stream< hls::vector< float, 32 > >& v1453,
  hls::stream< hls::vector< float, 32 > >& v1454,
  hls::stream< hls::vector< float, 32 > >& v1455
) {	// L1861
  l_S_i_0_i9: for (int i9 = 0; i9 < 8; i9++) {	// L1862
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1457 = v1452.read();
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = v1457[_iv0];
      }
      v1453.write(_vec);
    }	// L1864
    hls::vector< float, 32 > v1458 = v1454.read();
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = v1458[_iv0];
      }
      v1455.write(_vec);
    }	// L1866
  }
}

/// This is top function.
void fft_256(
  hls::stream< hls::vector< float, 32 > >& v1459,
  hls::stream< hls::vector< float, 32 > >& v1460,
  hls::stream< hls::vector< float, 32 > >& v1461,
  hls::stream< hls::vector< float, 32 > >& v1462
) {	// L1870
  #pragma HLS dataflow disable_start_propagation
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1463;
  #pragma HLS stream variable=v1463 depth=2	// L1871
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1464;
  #pragma HLS stream variable=v1464 depth=2	// L1872
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1465;
  #pragma HLS stream variable=v1465 depth=2	// L1873
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1466;
  #pragma HLS stream variable=v1466 depth=2	// L1874
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1467;
  #pragma HLS stream variable=v1467 depth=2	// L1875
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1468;
  #pragma HLS stream variable=v1468 depth=2	// L1876
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1469;
  #pragma HLS stream variable=v1469 depth=2	// L1877
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1470;
  #pragma HLS stream variable=v1470 depth=2	// L1878
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1471;
  #pragma HLS stream variable=v1471 depth=2	// L1879
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1472;
  #pragma HLS stream variable=v1472 depth=2	// L1880
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1473;
  #pragma HLS stream variable=v1473 depth=2	// L1881
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1474;
  #pragma HLS stream variable=v1474 depth=2	// L1882
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1475;
  #pragma HLS stream variable=v1475 depth=2	// L1883
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1476;
  #pragma HLS stream variable=v1476 depth=2	// L1884
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1477;
  #pragma HLS stream variable=v1477 depth=2	// L1885
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1478;
  #pragma HLS stream variable=v1478 depth=2	// L1886
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1479;
  #pragma HLS stream variable=v1479 depth=2	// L1887
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1480;
  #pragma HLS stream variable=v1480 depth=2	// L1888
  bit_rev_stage_0(v1459, v1460, v1463, v1472);	// L1889
  intra_0(v1463, v1472, v1464, v1473);	// L1890
  intra_1(v1464, v1473, v1465, v1474);	// L1891
  intra_2(v1465, v1474, v1466, v1475);	// L1892
  intra_3(v1466, v1475, v1467, v1476);	// L1893
  intra_4(v1467, v1476, v1468, v1477);	// L1894
  inter_0(v1468, v1477, v1469, v1478);	// L1895
  inter_1(v1469, v1478, v1470, v1479);	// L1896
  inter_2(v1470, v1479, v1471, v1480);	// L1897
  output_stage_0(v1471, v1461, v1480, v1462);	// L1898
}


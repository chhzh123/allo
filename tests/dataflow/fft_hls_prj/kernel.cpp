
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
  // placeholder for const float twr	// L995
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L996
  #pragma HLS array_partition variable=twi complete
  float in_re[32][8];	// L997
  #pragma HLS array_partition variable=in_re complete dim=1

  #pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L998
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L999
  #pragma HLS array_partition variable=out_re_b complete dim=1

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L1000
  #pragma HLS array_partition variable=out_im_b complete dim=1

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L1001
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v771 = v760.read();
    hls::vector< float, 32 > v772 = v761.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L1004
    #pragma HLS unroll
      int32_t v774 = k5;	// L1005
      int32_t v775 = v774 & 15;	// L1006
      int v776 = k5 >> 4;	// L1007
      int32_t v777 = i;	// L1008
      int32_t v778 = v777 & 1;	// L1009
      int32_t v779 = v776;	// L1010
      int32_t v780 = v779 ^ v778;	// L1011
      int32_t v781 = v780 << 4;	// L1012
      int32_t v782 = v775 | v781;	// L1013
      int32_t bank1;	// L1014
      bank1 = v782;	// L1015
      float v784 = v771[k5];	// L1016
      int32_t v785 = bank1;	// L1017
      int v786 = v785;	// L1018
      in_re[v786][i] = v784;	// L1019
      float v787 = v772[k5];	// L1020
      int32_t v788 = bank1;	// L1021
      int v789 = v788;	// L1022
      in_im[v789][i] = v787;	// L1023
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L1026
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im inter false
  #pragma HLS dependence variable=in_im intra false
  #pragma HLS dependence variable=in_re inter false
  #pragma HLS dependence variable=in_re intra false
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L1027
    #pragma HLS unroll
      int v792 = i1 << 4;	// L1028
      int v793 = v792 | k6;	// L1029
      uint32_t v794 = v793;	// L1030
      uint32_t bg;	// L1031
      bg = v794;	// L1032
      int32_t v796 = i1;	// L1033
      int32_t v797 = v796 & 1;	// L1034
      int32_t v798 = v797 << 4;	// L1035
      int32_t v799 = k6;	// L1036
      int32_t v800 = v799 | v798;	// L1037
      uint32_t bank_il;	// L1038
      bank_il = v800;	// L1039
      int32_t v802 = bank_il;	// L1040
      int32_t v803 = v802 ^ 16;	// L1041
      uint32_t bank_iu;	// L1042
      bank_iu = v803;	// L1043
      int v805 = i1 >> 1;	// L1044
      uint32_t v806 = v805;	// L1045
      uint32_t i_shr;	// L1046
      i_shr = v806;	// L1047
      uint32_t low_mask;	// L1048
      low_mask = 0;	// L1049
      int32_t v809 = i_shr;	// L1050
      int32_t v810 = low_mask;	// L1051
      uint32_t v811 = v809 & v810;	// L1052
      uint32_t low_bits;	// L1053
      low_bits = v811;	// L1054
      int32_t v813 = i_shr;	// L1055
      uint32_t high_bits;	// L1056
      high_bits = v813;	// L1057
      int32_t v815 = high_bits;	// L1058
      uint32_t v816 = v815 << 1;	// L1059
      int32_t v817 = low_bits;	// L1060
      uint32_t v818 = v816 | v817;	// L1061
      uint32_t off_il;	// L1062
      off_il = v818;	// L1063
      uint32_t stride_off;	// L1064
      stride_off = 1;	// L1065
      int32_t v821 = off_il;	// L1066
      int32_t v822 = stride_off;	// L1067
      uint32_t v823 = v821 | v822;	// L1068
      uint32_t off_iu;	// L1069
      off_iu = v823;	// L1070
      int32_t v825 = bank_il;	// L1071
      int v826 = v825;	// L1072
      int32_t v827 = off_il;	// L1073
      int v828 = v827;	// L1074
      float v829 = in_re[v826][v828];	// L1075
      float a_re5;	// L1076
      a_re5 = v829;	// L1077
      int32_t v831 = bank_il;	// L1078
      int v832 = v831;	// L1079
      int32_t v833 = off_il;	// L1080
      int v834 = v833;	// L1081
      float v835 = in_im[v832][v834];	// L1082
      float a_im5;	// L1083
      a_im5 = v835;	// L1084
      int32_t v837 = bank_iu;	// L1085
      int v838 = v837;	// L1086
      int32_t v839 = off_iu;	// L1087
      int v840 = v839;	// L1088
      float v841 = in_re[v838][v840];	// L1089
      float b_re5;	// L1090
      b_re5 = v841;	// L1091
      int32_t v843 = bank_iu;	// L1092
      int v844 = v843;	// L1093
      int32_t v845 = off_iu;	// L1094
      int v846 = v845;	// L1095
      float v847 = in_im[v844][v846];	// L1096
      float b_im5;	// L1097
      b_im5 = v847;	// L1098
      int32_t v849 = bg;	// L1099
      int64_t v850 = v849;	// L1100
      int64_t v851 = v850 & 31;	// L1101
      int64_t v852 = v851 << 2;	// L1102
      uint32_t v853 = v852;	// L1103
      uint32_t tw_k5;	// L1104
      tw_k5 = v853;	// L1105
      int32_t v855 = tw_k5;	// L1106
      int v856 = v855;	// L1107
      float v857 = twr[v856];	// L1108
      float tr5;	// L1109
      tr5 = v857;	// L1110
      int32_t v859 = tw_k5;	// L1111
      int v860 = v859;	// L1112
      float v861 = twi[v860];	// L1113
      float ti5;	// L1114
      ti5 = v861;	// L1115
      float v863 = b_re5;	// L1116
      float v864 = tr5;	// L1117
      float v865 = v863 * v864;	// L1118
      float v866 = b_im5;	// L1119
      float v867 = ti5;	// L1120
      float v868 = v866 * v867;	// L1121
      float v869 = v865 - v868;	// L1122
      float bw_re5;	// L1123
      bw_re5 = v869;	// L1124
      float v871 = b_re5;	// L1125
      float v872 = ti5;	// L1126
      float v873 = v871 * v872;	// L1127
      float v874 = b_im5;	// L1128
      float v875 = tr5;	// L1129
      float v876 = v874 * v875;	// L1130
      float v877 = v873 + v876;	// L1131
      float bw_im5;	// L1132
      bw_im5 = v877;	// L1133
      float v879 = a_re5;	// L1134
      float v880 = bw_re5;	// L1135
      float v881 = v879 + v880;	// L1136
      #pragma HLS bind_op variable=v881 op=fadd impl=fabric
      int32_t v882 = bank_il;	// L1137
      int v883 = v882;	// L1138
      int32_t v884 = off_il;	// L1139
      int v885 = v884;	// L1140
      out_re_b[v883][v885] = v881;	// L1141
      float v886 = a_im5;	// L1142
      float v887 = bw_im5;	// L1143
      float v888 = v886 + v887;	// L1144
      #pragma HLS bind_op variable=v888 op=fadd impl=fabric
      int32_t v889 = bank_il;	// L1145
      int v890 = v889;	// L1146
      int32_t v891 = off_il;	// L1147
      int v892 = v891;	// L1148
      out_im_b[v890][v892] = v888;	// L1149
      float v893 = a_re5;	// L1150
      float v894 = bw_re5;	// L1151
      float v895 = v893 - v894;	// L1152
      #pragma HLS bind_op variable=v895 op=fsub impl=fabric
      int32_t v896 = bank_iu;	// L1153
      int v897 = v896;	// L1154
      int32_t v898 = off_iu;	// L1155
      int v899 = v898;	// L1156
      out_re_b[v897][v899] = v895;	// L1157
      float v900 = a_im5;	// L1158
      float v901 = bw_im5;	// L1159
      float v902 = v900 - v901;	// L1160
      #pragma HLS bind_op variable=v902 op=fsub impl=fabric
      int32_t v903 = bank_iu;	// L1161
      int v904 = v903;	// L1162
      int32_t v905 = off_iu;	// L1163
      int v906 = v905;	// L1164
      out_im_b[v904][v906] = v902;	// L1165
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L1168
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    float chunk_re_out[32];	// L1169
    #pragma HLS array_partition variable=chunk_re_out complete
    float chunk_im_out[32];	// L1170
    #pragma HLS array_partition variable=chunk_im_out complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L1171
    #pragma HLS unroll
      int32_t v911 = k7;	// L1172
      int32_t v912 = v911 & 15;	// L1173
      int v913 = k7 >> 4;	// L1174
      int32_t v914 = i2;	// L1175
      int32_t v915 = v914 & 1;	// L1176
      int32_t v916 = v913;	// L1177
      int32_t v917 = v916 ^ v915;	// L1178
      int32_t v918 = v917 << 4;	// L1179
      int32_t v919 = v912 | v918;	// L1180
      int32_t bank2;	// L1181
      bank2 = v919;	// L1182
      int32_t v921 = bank2;	// L1183
      int v922 = v921;	// L1184
      float v923 = out_re_b[v922][i2];	// L1185
      chunk_re_out[k7] = v923;	// L1186
      int32_t v924 = bank2;	// L1187
      int v925 = v924;	// L1188
      float v926 = out_im_b[v925][i2];	// L1189
      chunk_im_out[k7] = v926;	// L1190
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out[_iv0];
      }
      v762.write(_vec);
    }	// L1192
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out[_iv0];
      }
      v763.write(_vec);
    }	// L1193
  }
}

void inter_1(
  hls::stream< hls::vector< float, 32 > >& v927,
  hls::stream< hls::vector< float, 32 > >& v928,
  hls::stream< hls::vector< float, 32 > >& v929,
  hls::stream< hls::vector< float, 32 > >& v930
) {	// L1197
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1207
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1208
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L1209
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L1210
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L1211
  #pragma HLS array_partition variable=out_re_b1 complete dim=1

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L1212
  #pragma HLS array_partition variable=out_im_b1 complete dim=1

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L1213
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v938 = v927.read();
    hls::vector< float, 32 > v939 = v928.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L1216
    #pragma HLS unroll
      int32_t v941 = k8;	// L1217
      int32_t v942 = v941 & 15;	// L1218
      int v943 = k8 >> 4;	// L1219
      int v944 = i3 >> 1;	// L1220
      int32_t v945 = v944;	// L1221
      int32_t v946 = v945 & 1;	// L1222
      int32_t v947 = v943;	// L1223
      int32_t v948 = v947 ^ v946;	// L1224
      int32_t v949 = v948 << 4;	// L1225
      int32_t v950 = v942 | v949;	// L1226
      int32_t bank3;	// L1227
      bank3 = v950;	// L1228
      float v952 = v938[k8];	// L1229
      int32_t v953 = bank3;	// L1230
      int v954 = v953;	// L1231
      in_re1[v954][i3] = v952;	// L1232
      float v955 = v939[k8];	// L1233
      int32_t v956 = bank3;	// L1234
      int v957 = v956;	// L1235
      in_im1[v957][i3] = v955;	// L1236
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L1239
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im1 inter false
  #pragma HLS dependence variable=in_im1 intra false
  #pragma HLS dependence variable=in_re1 inter false
  #pragma HLS dependence variable=in_re1 intra false
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L1240
    #pragma HLS unroll
      int v960 = i4 << 4;	// L1241
      int v961 = v960 | k9;	// L1242
      uint32_t v962 = v961;	// L1243
      uint32_t bg1;	// L1244
      bg1 = v962;	// L1245
      int32_t v964 = i4;	// L1246
      int32_t v965 = v964 & 1;	// L1247
      int32_t v966 = v965 << 4;	// L1248
      int32_t v967 = k9;	// L1249
      int32_t v968 = v967 | v966;	// L1250
      uint32_t bank_il1;	// L1251
      bank_il1 = v968;	// L1252
      int32_t v970 = bank_il1;	// L1253
      int32_t v971 = v970 ^ 16;	// L1254
      uint32_t bank_iu1;	// L1255
      bank_iu1 = v971;	// L1256
      int v973 = i4 >> 1;	// L1257
      uint32_t v974 = v973;	// L1258
      uint32_t i_shr1;	// L1259
      i_shr1 = v974;	// L1260
      uint32_t low_mask1;	// L1261
      low_mask1 = 1;	// L1262
      int32_t v977 = i_shr1;	// L1263
      int32_t v978 = low_mask1;	// L1264
      uint32_t v979 = v977 & v978;	// L1265
      uint32_t low_bits1;	// L1266
      low_bits1 = v979;	// L1267
      int32_t v981 = i_shr1;	// L1268
      uint32_t v982 = v981 >> 1;	// L1269
      uint32_t high_bits1;	// L1270
      high_bits1 = v982;	// L1271
      int32_t v984 = high_bits1;	// L1272
      uint32_t v985 = v984 << 2;	// L1273
      int32_t v986 = low_bits1;	// L1274
      uint32_t v987 = v985 | v986;	// L1275
      uint32_t off_il1;	// L1276
      off_il1 = v987;	// L1277
      uint32_t stride_off1;	// L1278
      stride_off1 = 2;	// L1279
      int32_t v990 = off_il1;	// L1280
      int32_t v991 = stride_off1;	// L1281
      uint32_t v992 = v990 | v991;	// L1282
      uint32_t off_iu1;	// L1283
      off_iu1 = v992;	// L1284
      int32_t v994 = bank_il1;	// L1285
      int v995 = v994;	// L1286
      int32_t v996 = off_il1;	// L1287
      int v997 = v996;	// L1288
      float v998 = in_re1[v995][v997];	// L1289
      float a_re6;	// L1290
      a_re6 = v998;	// L1291
      int32_t v1000 = bank_il1;	// L1292
      int v1001 = v1000;	// L1293
      int32_t v1002 = off_il1;	// L1294
      int v1003 = v1002;	// L1295
      float v1004 = in_im1[v1001][v1003];	// L1296
      float a_im6;	// L1297
      a_im6 = v1004;	// L1298
      int32_t v1006 = bank_iu1;	// L1299
      int v1007 = v1006;	// L1300
      int32_t v1008 = off_iu1;	// L1301
      int v1009 = v1008;	// L1302
      float v1010 = in_re1[v1007][v1009];	// L1303
      float b_re6;	// L1304
      b_re6 = v1010;	// L1305
      int32_t v1012 = bank_iu1;	// L1306
      int v1013 = v1012;	// L1307
      int32_t v1014 = off_iu1;	// L1308
      int v1015 = v1014;	// L1309
      float v1016 = in_im1[v1013][v1015];	// L1310
      float b_im6;	// L1311
      b_im6 = v1016;	// L1312
      int32_t v1018 = bg1;	// L1313
      int64_t v1019 = v1018;	// L1314
      int64_t v1020 = v1019 & 63;	// L1315
      int64_t v1021 = v1020 << 1;	// L1316
      uint32_t v1022 = v1021;	// L1317
      uint32_t tw_k6;	// L1318
      tw_k6 = v1022;	// L1319
      int32_t v1024 = tw_k6;	// L1320
      int v1025 = v1024;	// L1321
      float v1026 = twr[v1025];	// L1322
      float tr6;	// L1323
      tr6 = v1026;	// L1324
      int32_t v1028 = tw_k6;	// L1325
      int v1029 = v1028;	// L1326
      float v1030 = twi[v1029];	// L1327
      float ti6;	// L1328
      ti6 = v1030;	// L1329
      float v1032 = b_re6;	// L1330
      float v1033 = tr6;	// L1331
      float v1034 = v1032 * v1033;	// L1332
      float v1035 = b_im6;	// L1333
      float v1036 = ti6;	// L1334
      float v1037 = v1035 * v1036;	// L1335
      float v1038 = v1034 - v1037;	// L1336
      float bw_re6;	// L1337
      bw_re6 = v1038;	// L1338
      float v1040 = b_re6;	// L1339
      float v1041 = ti6;	// L1340
      float v1042 = v1040 * v1041;	// L1341
      float v1043 = b_im6;	// L1342
      float v1044 = tr6;	// L1343
      float v1045 = v1043 * v1044;	// L1344
      float v1046 = v1042 + v1045;	// L1345
      float bw_im6;	// L1346
      bw_im6 = v1046;	// L1347
      float v1048 = a_re6;	// L1348
      float v1049 = bw_re6;	// L1349
      float v1050 = v1048 + v1049;	// L1350
      #pragma HLS bind_op variable=v1050 op=fadd impl=fabric
      int32_t v1051 = bank_il1;	// L1351
      int v1052 = v1051;	// L1352
      int32_t v1053 = off_il1;	// L1353
      int v1054 = v1053;	// L1354
      out_re_b1[v1052][v1054] = v1050;	// L1355
      float v1055 = a_im6;	// L1356
      float v1056 = bw_im6;	// L1357
      float v1057 = v1055 + v1056;	// L1358
      #pragma HLS bind_op variable=v1057 op=fadd impl=fabric
      int32_t v1058 = bank_il1;	// L1359
      int v1059 = v1058;	// L1360
      int32_t v1060 = off_il1;	// L1361
      int v1061 = v1060;	// L1362
      out_im_b1[v1059][v1061] = v1057;	// L1363
      float v1062 = a_re6;	// L1364
      float v1063 = bw_re6;	// L1365
      float v1064 = v1062 - v1063;	// L1366
      #pragma HLS bind_op variable=v1064 op=fsub impl=fabric
      int32_t v1065 = bank_iu1;	// L1367
      int v1066 = v1065;	// L1368
      int32_t v1067 = off_iu1;	// L1369
      int v1068 = v1067;	// L1370
      out_re_b1[v1066][v1068] = v1064;	// L1371
      float v1069 = a_im6;	// L1372
      float v1070 = bw_im6;	// L1373
      float v1071 = v1069 - v1070;	// L1374
      #pragma HLS bind_op variable=v1071 op=fsub impl=fabric
      int32_t v1072 = bank_iu1;	// L1375
      int v1073 = v1072;	// L1376
      int32_t v1074 = off_iu1;	// L1377
      int v1075 = v1074;	// L1378
      out_im_b1[v1073][v1075] = v1071;	// L1379
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L1382
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    float chunk_re_out1[32];	// L1383
    #pragma HLS array_partition variable=chunk_re_out1 complete
    float chunk_im_out1[32];	// L1384
    #pragma HLS array_partition variable=chunk_im_out1 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L1385
    #pragma HLS unroll
      int32_t v1080 = k10;	// L1386
      int32_t v1081 = v1080 & 15;	// L1387
      int v1082 = k10 >> 4;	// L1388
      int v1083 = i5 >> 1;	// L1389
      int32_t v1084 = v1083;	// L1390
      int32_t v1085 = v1084 & 1;	// L1391
      int32_t v1086 = v1082;	// L1392
      int32_t v1087 = v1086 ^ v1085;	// L1393
      int32_t v1088 = v1087 << 4;	// L1394
      int32_t v1089 = v1081 | v1088;	// L1395
      int32_t bank4;	// L1396
      bank4 = v1089;	// L1397
      int32_t v1091 = bank4;	// L1398
      int v1092 = v1091;	// L1399
      float v1093 = out_re_b1[v1092][i5];	// L1400
      chunk_re_out1[k10] = v1093;	// L1401
      int32_t v1094 = bank4;	// L1402
      int v1095 = v1094;	// L1403
      float v1096 = out_im_b1[v1095][i5];	// L1404
      chunk_im_out1[k10] = v1096;	// L1405
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out1[_iv0];
      }
      v929.write(_vec);
    }	// L1407
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out1[_iv0];
      }
      v930.write(_vec);
    }	// L1408
  }
}

void inter_2(
  hls::stream< hls::vector< float, 32 > >& v1097,
  hls::stream< hls::vector< float, 32 > >& v1098,
  hls::stream< hls::vector< float, 32 > >& v1099,
  hls::stream< hls::vector< float, 32 > >& v1100
) {	// L1412
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1423
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1424
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L1425
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L1426
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L1427
  #pragma HLS array_partition variable=out_re_b2 complete dim=1

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L1428
  #pragma HLS array_partition variable=out_im_b2 complete dim=1

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L1429
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1108 = v1097.read();
    hls::vector< float, 32 > v1109 = v1098.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L1432
    #pragma HLS unroll
      int32_t v1111 = k11;	// L1433
      int32_t v1112 = v1111 & 15;	// L1434
      int v1113 = k11 >> 4;	// L1435
      int v1114 = i6 >> 2;	// L1436
      int32_t v1115 = v1114;	// L1437
      int32_t v1116 = v1115 & 1;	// L1438
      int32_t v1117 = v1113;	// L1439
      int32_t v1118 = v1117 ^ v1116;	// L1440
      int32_t v1119 = v1118 << 4;	// L1441
      int32_t v1120 = v1112 | v1119;	// L1442
      int32_t bank5;	// L1443
      bank5 = v1120;	// L1444
      float v1122 = v1108[k11];	// L1445
      int32_t v1123 = bank5;	// L1446
      int v1124 = v1123;	// L1447
      in_re2[v1124][i6] = v1122;	// L1448
      float v1125 = v1109[k11];	// L1449
      int32_t v1126 = bank5;	// L1450
      int v1127 = v1126;	// L1451
      in_im2[v1127][i6] = v1125;	// L1452
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L1455
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im2 inter false
  #pragma HLS dependence variable=in_im2 intra false
  #pragma HLS dependence variable=in_re2 inter false
  #pragma HLS dependence variable=in_re2 intra false
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L1456
    #pragma HLS unroll
      int v1130 = i7 << 4;	// L1457
      int v1131 = v1130 | k12;	// L1458
      uint32_t v1132 = v1131;	// L1459
      uint32_t bg2;	// L1460
      bg2 = v1132;	// L1461
      int32_t v1134 = i7;	// L1462
      int32_t v1135 = v1134 & 1;	// L1463
      int32_t v1136 = v1135 << 4;	// L1464
      int32_t v1137 = k12;	// L1465
      int32_t v1138 = v1137 | v1136;	// L1466
      uint32_t bank_il2;	// L1467
      bank_il2 = v1138;	// L1468
      int32_t v1140 = bank_il2;	// L1469
      int32_t v1141 = v1140 ^ 16;	// L1470
      uint32_t bank_iu2;	// L1471
      bank_iu2 = v1141;	// L1472
      int v1143 = i7 >> 1;	// L1473
      uint32_t v1144 = v1143;	// L1474
      uint32_t i_shr2;	// L1475
      i_shr2 = v1144;	// L1476
      uint32_t low_mask2;	// L1477
      low_mask2 = 3;	// L1478
      int32_t v1147 = i_shr2;	// L1479
      int32_t v1148 = low_mask2;	// L1480
      uint32_t v1149 = v1147 & v1148;	// L1481
      uint32_t low_bits2;	// L1482
      low_bits2 = v1149;	// L1483
      int32_t v1151 = i_shr2;	// L1484
      uint32_t v1152 = v1151 >> 2;	// L1485
      uint32_t high_bits2;	// L1486
      high_bits2 = v1152;	// L1487
      int32_t v1154 = high_bits2;	// L1488
      uint32_t v1155 = v1154 << 3;	// L1489
      int32_t v1156 = low_bits2;	// L1490
      uint32_t v1157 = v1155 | v1156;	// L1491
      uint32_t off_il2;	// L1492
      off_il2 = v1157;	// L1493
      uint32_t stride_off2;	// L1494
      stride_off2 = 4;	// L1495
      int32_t v1160 = off_il2;	// L1496
      int32_t v1161 = stride_off2;	// L1497
      uint32_t v1162 = v1160 | v1161;	// L1498
      uint32_t off_iu2;	// L1499
      off_iu2 = v1162;	// L1500
      int32_t v1164 = bank_il2;	// L1501
      int v1165 = v1164;	// L1502
      int32_t v1166 = off_il2;	// L1503
      int v1167 = v1166;	// L1504
      float v1168 = in_re2[v1165][v1167];	// L1505
      float a_re7;	// L1506
      a_re7 = v1168;	// L1507
      int32_t v1170 = bank_il2;	// L1508
      int v1171 = v1170;	// L1509
      int32_t v1172 = off_il2;	// L1510
      int v1173 = v1172;	// L1511
      float v1174 = in_im2[v1171][v1173];	// L1512
      float a_im7;	// L1513
      a_im7 = v1174;	// L1514
      int32_t v1176 = bank_iu2;	// L1515
      int v1177 = v1176;	// L1516
      int32_t v1178 = off_iu2;	// L1517
      int v1179 = v1178;	// L1518
      float v1180 = in_re2[v1177][v1179];	// L1519
      float b_re7;	// L1520
      b_re7 = v1180;	// L1521
      int32_t v1182 = bank_iu2;	// L1522
      int v1183 = v1182;	// L1523
      int32_t v1184 = off_iu2;	// L1524
      int v1185 = v1184;	// L1525
      float v1186 = in_im2[v1183][v1185];	// L1526
      float b_im7;	// L1527
      b_im7 = v1186;	// L1528
      int32_t v1188 = bg2;	// L1529
      int64_t v1189 = v1188;	// L1530
      int64_t v1190 = v1189 & 127;	// L1531
      uint32_t v1191 = v1190;	// L1532
      uint32_t tw_k7;	// L1533
      tw_k7 = v1191;	// L1534
      int32_t v1193 = tw_k7;	// L1535
      int v1194 = v1193;	// L1536
      float v1195 = twr[v1194];	// L1537
      float tr7;	// L1538
      tr7 = v1195;	// L1539
      int32_t v1197 = tw_k7;	// L1540
      int v1198 = v1197;	// L1541
      float v1199 = twi[v1198];	// L1542
      float ti7;	// L1543
      ti7 = v1199;	// L1544
      float v1201 = b_re7;	// L1545
      float v1202 = tr7;	// L1546
      float v1203 = v1201 * v1202;	// L1547
      float v1204 = b_im7;	// L1548
      float v1205 = ti7;	// L1549
      float v1206 = v1204 * v1205;	// L1550
      float v1207 = v1203 - v1206;	// L1551
      float bw_re7;	// L1552
      bw_re7 = v1207;	// L1553
      float v1209 = b_re7;	// L1554
      float v1210 = ti7;	// L1555
      float v1211 = v1209 * v1210;	// L1556
      float v1212 = b_im7;	// L1557
      float v1213 = tr7;	// L1558
      float v1214 = v1212 * v1213;	// L1559
      float v1215 = v1211 + v1214;	// L1560
      float bw_im7;	// L1561
      bw_im7 = v1215;	// L1562
      float v1217 = a_re7;	// L1563
      float v1218 = bw_re7;	// L1564
      float v1219 = v1217 + v1218;	// L1565
      #pragma HLS bind_op variable=v1219 op=fadd impl=fabric
      int32_t v1220 = bank_il2;	// L1566
      int v1221 = v1220;	// L1567
      int32_t v1222 = off_il2;	// L1568
      int v1223 = v1222;	// L1569
      out_re_b2[v1221][v1223] = v1219;	// L1570
      float v1224 = a_im7;	// L1571
      float v1225 = bw_im7;	// L1572
      float v1226 = v1224 + v1225;	// L1573
      #pragma HLS bind_op variable=v1226 op=fadd impl=fabric
      int32_t v1227 = bank_il2;	// L1574
      int v1228 = v1227;	// L1575
      int32_t v1229 = off_il2;	// L1576
      int v1230 = v1229;	// L1577
      out_im_b2[v1228][v1230] = v1226;	// L1578
      float v1231 = a_re7;	// L1579
      float v1232 = bw_re7;	// L1580
      float v1233 = v1231 - v1232;	// L1581
      #pragma HLS bind_op variable=v1233 op=fsub impl=fabric
      int32_t v1234 = bank_iu2;	// L1582
      int v1235 = v1234;	// L1583
      int32_t v1236 = off_iu2;	// L1584
      int v1237 = v1236;	// L1585
      out_re_b2[v1235][v1237] = v1233;	// L1586
      float v1238 = a_im7;	// L1587
      float v1239 = bw_im7;	// L1588
      float v1240 = v1238 - v1239;	// L1589
      #pragma HLS bind_op variable=v1240 op=fsub impl=fabric
      int32_t v1241 = bank_iu2;	// L1590
      int v1242 = v1241;	// L1591
      int32_t v1243 = off_iu2;	// L1592
      int v1244 = v1243;	// L1593
      out_im_b2[v1242][v1244] = v1240;	// L1594
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1597
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    float chunk_re_out2[32];	// L1598
    #pragma HLS array_partition variable=chunk_re_out2 complete
    float chunk_im_out2[32];	// L1599
    #pragma HLS array_partition variable=chunk_im_out2 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1600
    #pragma HLS unroll
      int32_t v1249 = k13;	// L1601
      int32_t v1250 = v1249 & 15;	// L1602
      int v1251 = k13 >> 4;	// L1603
      int v1252 = i8 >> 2;	// L1604
      int32_t v1253 = v1252;	// L1605
      int32_t v1254 = v1253 & 1;	// L1606
      int32_t v1255 = v1251;	// L1607
      int32_t v1256 = v1255 ^ v1254;	// L1608
      int32_t v1257 = v1256 << 4;	// L1609
      int32_t v1258 = v1250 | v1257;	// L1610
      int32_t bank6;	// L1611
      bank6 = v1258;	// L1612
      int32_t v1260 = bank6;	// L1613
      int v1261 = v1260;	// L1614
      float v1262 = out_re_b2[v1261][i8];	// L1615
      chunk_re_out2[k13] = v1262;	// L1616
      int32_t v1263 = bank6;	// L1617
      int v1264 = v1263;	// L1618
      float v1265 = out_im_b2[v1264][i8];	// L1619
      chunk_im_out2[k13] = v1265;	// L1620
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out2[_iv0];
      }
      v1099.write(_vec);
    }	// L1622
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out2[_iv0];
      }
      v1100.write(_vec);
    }	// L1623
  }
}

void output_stage_0(
  hls::stream< hls::vector< float, 32 > >& v1266,
  hls::stream< hls::vector< float, 32 > >& v1267,
  hls::stream< hls::vector< float, 32 > >& v1268,
  hls::stream< hls::vector< float, 32 > >& v1269
) {	// L1627
  l_S_i_0_i9: for (int i9 = 0; i9 < 8; i9++) {	// L1628
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1271 = v1266.read();
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = v1271[_iv0];
      }
      v1267.write(_vec);
    }	// L1630
    hls::vector< float, 32 > v1272 = v1268.read();
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = v1272[_iv0];
      }
      v1269.write(_vec);
    }	// L1632
  }
}

/// This is top function.
void fft_256(
  hls::stream< hls::vector< float, 32 > >& v1273,
  hls::stream< hls::vector< float, 32 > >& v1274,
  hls::stream< hls::vector< float, 32 > >& v1275,
  hls::stream< hls::vector< float, 32 > >& v1276
) {	// L1636
  #pragma HLS dataflow disable_start_propagation
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1277;
  #pragma HLS stream variable=v1277 depth=2	// L1637
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1278;
  #pragma HLS stream variable=v1278 depth=2	// L1638
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1279;
  #pragma HLS stream variable=v1279 depth=2	// L1639
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1280;
  #pragma HLS stream variable=v1280 depth=2	// L1640
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1281;
  #pragma HLS stream variable=v1281 depth=2	// L1641
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1282;
  #pragma HLS stream variable=v1282 depth=2	// L1642
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1283;
  #pragma HLS stream variable=v1283 depth=2	// L1643
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1284;
  #pragma HLS stream variable=v1284 depth=2	// L1644
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1285;
  #pragma HLS stream variable=v1285 depth=2	// L1645
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1286;
  #pragma HLS stream variable=v1286 depth=2	// L1646
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1287;
  #pragma HLS stream variable=v1287 depth=2	// L1647
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1288;
  #pragma HLS stream variable=v1288 depth=2	// L1648
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1289;
  #pragma HLS stream variable=v1289 depth=2	// L1649
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1290;
  #pragma HLS stream variable=v1290 depth=2	// L1650
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1291;
  #pragma HLS stream variable=v1291 depth=2	// L1651
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1292;
  #pragma HLS stream variable=v1292 depth=2	// L1652
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1293;
  #pragma HLS stream variable=v1293 depth=2	// L1653
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1294;
  #pragma HLS stream variable=v1294 depth=2	// L1654
  bit_rev_stage_0(v1273, v1274, v1277, v1286);	// L1655
  intra_0(v1277, v1286, v1278, v1287);	// L1656
  intra_1(v1278, v1287, v1279, v1288);	// L1657
  intra_2(v1279, v1288, v1280, v1289);	// L1658
  intra_3(v1280, v1289, v1281, v1290);	// L1659
  intra_4(v1281, v1290, v1282, v1291);	// L1660
  inter_0(v1282, v1291, v1283, v1292);	// L1661
  inter_1(v1283, v1292, v1284, v1293);	// L1662
  inter_2(v1284, v1293, v1285, v1294);	// L1663
  output_stage_0(v1285, v1275, v1294, v1276);	// L1664
}



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
  float buf_re[32][8];	// L17
  #pragma HLS array_partition variable=buf_re complete dim=1

  #pragma HLS bind_storage variable=buf_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=buf_re inter false
  float buf_im[32][8];	// L18
  #pragma HLS array_partition variable=buf_im complete dim=1

  #pragma HLS bind_storage variable=buf_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=buf_im inter false
  l_S_ii_0_ii: for (int ii = 0; ii < 8; ii++) {	// L19
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v7 = v0.read();
    hls::vector< float, 32 > v8 = v1.read();
    l_S_kk_0_kk: for (int kk = 0; kk < 32; kk++) {	// L22
    #pragma HLS unroll
      int v10 = ii << 5;	// L23
      int v11 = v10 | kk;	// L24
      uint32_t v12 = v11;	// L25
      uint32_t idx;	// L26
      idx = v12;	// L27
      int32_t v14 = idx;	// L28
      int32_t v15 = v14 & 1;	// L29
      int32_t v16 = v15 << 7;	// L30
      int32_t v17 = v14 & 2;	// L31
      int32_t v18 = v17 << 5;	// L32
      int32_t v19 = v16 | v18;	// L33
      int32_t v20 = v14 & 4;	// L34
      int32_t v21 = v20 << 3;	// L35
      int32_t v22 = v19 | v21;	// L36
      int32_t v23 = v14 & 8;	// L37
      int32_t v24 = v23 << 1;	// L38
      int32_t v25 = v22 | v24;	// L39
      int32_t v26 = v14 & 16;	// L40
      int32_t v27 = v26 >> 1;	// L41
      int32_t v28 = v25 | v27;	// L42
      int32_t v29 = v14 & 32;	// L43
      int32_t v30 = v29 >> 3;	// L44
      int32_t v31 = v28 | v30;	// L45
      int32_t v32 = v14 & 64;	// L46
      int32_t v33 = v32 >> 5;	// L47
      int32_t v34 = v31 | v33;	// L48
      int32_t v35 = v14 & 128;	// L49
      int32_t v36 = v35 >> 7;	// L50
      int32_t v37 = v34 | v36;	// L51
      uint32_t rev;	// L52
      rev = v37;	// L53
      float v39 = v7[kk];	// L54
      int32_t v40 = rev;	// L55
      int32_t v41 = v40 >> 3;	// L57
      int32_t v42 = v40 & 7;	// L59
      int v43 = v41;	// L60
      int v44 = v42;	// L61
      buf_re[v43][v44] = v39;	// L62
      float v45 = v8[kk];	// L63
      int32_t v46 = rev;	// L64
      int32_t v47 = v46 >> 3;	// L65
      int32_t v48 = v46 & 7;	// L66
      int v49 = v47;	// L67
      int v50 = v48;	// L68
      buf_im[v49][v50] = v45;	// L69
    }
  }
  l_S_jj_2_jj: for (int jj = 0; jj < 8; jj++) {	// L72
  #pragma HLS pipeline II=1
    float chunk_re[32];	// L73
    #pragma HLS array_partition variable=chunk_re complete
    float chunk_im[32];	// L74
    #pragma HLS array_partition variable=chunk_im complete
    l_S_mm_2_mm: for (int mm = 0; mm < 32; mm++) {	// L75
    #pragma HLS unroll
      int v55 = ((jj * 32) + mm);	// L76
      int32_t v56 = v55;	// L77
      int32_t v57 = v56 >> 3;	// L79
      int32_t v58 = v56 & 7;	// L81
      int v59 = v57;	// L82
      int v60 = v58;	// L83
      float v61 = buf_re[v59][v60];	// L84
      chunk_re[mm] = v61;	// L85
      float v62 = buf_im[v59][v60];	// L86
      chunk_im[mm] = v62;	// L87
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re[_iv0];
      }
      v2.write(_vec);
    }	// L89
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im[_iv0];
      }
      v3.write(_vec);
    }	// L90
  }
}

const float twr[128] = {1.000000e+00, 9.996988e-01, 9.987954e-01, 9.972904e-01, 9.951847e-01, 9.924796e-01, 9.891765e-01, 9.852777e-01, 9.807853e-01, 9.757021e-01, 9.700313e-01, 9.637761e-01, 9.569404e-01, 9.495282e-01, 9.415441e-01, 9.329928e-01, 9.238795e-01, 9.142098e-01, 9.039893e-01, 8.932243e-01, 8.819213e-01, 8.700870e-01, 8.577286e-01, 8.448536e-01, 8.314696e-01, 8.175848e-01, 8.032075e-01, 7.883464e-01, 7.730104e-01, 7.572088e-01, 7.409511e-01, 7.242471e-01, 7.071068e-01, 6.895406e-01, 6.715590e-01, 6.531729e-01, 6.343933e-01, 6.152316e-01, 5.956993e-01, 5.758082e-01, 5.555702e-01, 5.349976e-01, 5.141028e-01, 4.928982e-01, 4.713967e-01, 4.496113e-01, 4.275551e-01, 4.052413e-01, 3.826834e-01, 3.598951e-01, 3.368899e-01, 3.136818e-01, 2.902847e-01, 2.667128e-01, 2.429802e-01, 2.191012e-01, 1.950903e-01, 1.709619e-01, 1.467305e-01, 1.224107e-01, 9.801714e-02, 7.356457e-02, 4.906768e-02, 2.454123e-02, 0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01};	// L94
const float twi[128] = {0.000000e+00, -2.454123e-02, -4.906768e-02, -7.356457e-02, -9.801714e-02, -1.224107e-01, -1.467305e-01, -1.709619e-01, -1.950903e-01, -2.191012e-01, -2.429802e-01, -2.667128e-01, -2.902847e-01, -3.136818e-01, -3.368899e-01, -3.598951e-01, -3.826834e-01, -4.052413e-01, -4.275551e-01, -4.496113e-01, -4.713967e-01, -4.928982e-01, -5.141028e-01, -5.349976e-01, -5.555702e-01, -5.758082e-01, -5.956993e-01, -6.152316e-01, -6.343933e-01, -6.531729e-01, -6.715590e-01, -6.895406e-01, -7.071068e-01, -7.242471e-01, -7.409511e-01, -7.572088e-01, -7.730104e-01, -7.883464e-01, -8.032075e-01, -8.175848e-01, -8.314696e-01, -8.448536e-01, -8.577286e-01, -8.700870e-01, -8.819213e-01, -8.932243e-01, -9.039893e-01, -9.142098e-01, -9.238795e-01, -9.329928e-01, -9.415441e-01, -9.495282e-01, -9.569404e-01, -9.637761e-01, -9.700313e-01, -9.757021e-01, -9.807853e-01, -9.852777e-01, -9.891765e-01, -9.924796e-01, -9.951847e-01, -9.972904e-01, -9.987954e-01, -9.996988e-01, -1.000000e+00, -9.996988e-01, -9.987954e-01, -9.972904e-01, -9.951847e-01, -9.924796e-01, -9.891765e-01, -9.852777e-01, -9.807853e-01, -9.757021e-01, -9.700313e-01, -9.637761e-01, -9.569404e-01, -9.495282e-01, -9.415441e-01, -9.329928e-01, -9.238795e-01, -9.142098e-01, -9.039893e-01, -8.932243e-01, -8.819213e-01, -8.700870e-01, -8.577286e-01, -8.448536e-01, -8.314696e-01, -8.175848e-01, -8.032075e-01, -7.883464e-01, -7.730104e-01, -7.572088e-01, -7.409511e-01, -7.242471e-01, -7.071068e-01, -6.895406e-01, -6.715590e-01, -6.531729e-01, -6.343933e-01, -6.152316e-01, -5.956993e-01, -5.758082e-01, -5.555702e-01, -5.349976e-01, -5.141028e-01, -4.928982e-01, -4.713967e-01, -4.496113e-01, -4.275551e-01, -4.052413e-01, -3.826834e-01, -3.598951e-01, -3.368899e-01, -3.136818e-01, -2.902847e-01, -2.667128e-01, -2.429802e-01, -2.191012e-01, -1.950903e-01, -1.709619e-01, -1.467305e-01, -1.224107e-01, -9.801714e-02, -7.356457e-02, -4.906768e-02, -2.454123e-02};	// L95
void intra_0(
  hls::stream< hls::vector< float, 32 > >& v63,
  hls::stream< hls::vector< float, 32 > >& v64,
  hls::stream< hls::vector< float, 32 > >& v65,
  hls::stream< hls::vector< float, 32 > >& v66
) {	// L96
  // placeholder for const float twr	// L103
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L104
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i: for (int _i = 0; _i < 8; _i++) {	// L105
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v70 = v63.read();
    hls::vector< float, 32 > v71 = v64.read();
    float o_re[32];	// L108
    #pragma HLS array_partition variable=o_re complete
    float o_im[32];	// L109
    #pragma HLS array_partition variable=o_im complete
    int32_t stride;	// L110
    stride = 1;	// L111
    l_S_k_0_k: for (int k = 0; k < 16; k++) {	// L112
    #pragma HLS unroll
      int v76 = k << 1;	// L113
      int32_t v77 = stride;	// L114
      int64_t v78 = v77;	// L115
      int64_t v79 = v78 - 1;	// L116
      int64_t v80 = k;	// L117
      int64_t v81 = v80 & v79;	// L118
      int64_t v82 = v76;	// L119
      int64_t v83 = v82 | v81;	// L120
      int32_t v84 = v83;	// L121
      int32_t il;	// L122
      il = v84;	// L123
      int32_t v86 = il;	// L124
      int32_t v87 = stride;	// L125
      int32_t v88 = v86 | v87;	// L126
      int32_t iu;	// L127
      iu = v88;	// L128
      int32_t v90 = stride;	// L129
      int64_t v91 = v90;	// L130
      int64_t v92 = v91 - 1;	// L131
      int64_t v93 = v80 & v92;	// L132
      int64_t v94 = v93 << 7;	// L133
      int32_t v95 = v94;	// L134
      int32_t tw_k;	// L135
      tw_k = v95;	// L136
      int32_t v97 = il;	// L137
      int v98 = v97;	// L138
      float v99 = v70[v98];	// L139
      float a_re;	// L140
      a_re = v99;	// L141
      int32_t v101 = il;	// L142
      int v102 = v101;	// L143
      float v103 = v71[v102];	// L144
      float a_im;	// L145
      a_im = v103;	// L146
      int32_t v105 = iu;	// L147
      int v106 = v105;	// L148
      float v107 = v70[v106];	// L149
      float b_re;	// L150
      b_re = v107;	// L151
      int32_t v109 = iu;	// L152
      int v110 = v109;	// L153
      float v111 = v71[v110];	// L154
      float b_im;	// L155
      b_im = v111;	// L156
      int32_t v113 = tw_k;	// L157
      bool v114 = v113 == 0;	// L158
      if (v114) {	// L159
        float v115 = a_re;	// L160
        float v116 = b_re;	// L161
        float v117 = v115 + v116;	// L162
        #pragma HLS bind_op variable=v117 op=fadd impl=fabric
        int32_t v118 = il;	// L163
        int v119 = v118;	// L164
        o_re[v119] = v117;	// L165
        float v120 = a_im;	// L166
        float v121 = b_im;	// L167
        float v122 = v120 + v121;	// L168
        #pragma HLS bind_op variable=v122 op=fadd impl=fabric
        int32_t v123 = il;	// L169
        int v124 = v123;	// L170
        o_im[v124] = v122;	// L171
        float v125 = a_re;	// L172
        float v126 = b_re;	// L173
        float v127 = v125 - v126;	// L174
        #pragma HLS bind_op variable=v127 op=fsub impl=fabric
        int32_t v128 = iu;	// L175
        int v129 = v128;	// L176
        o_re[v129] = v127;	// L177
        float v130 = a_im;	// L178
        float v131 = b_im;	// L179
        float v132 = v130 - v131;	// L180
        #pragma HLS bind_op variable=v132 op=fsub impl=fabric
        int32_t v133 = iu;	// L181
        int v134 = v133;	// L182
        o_im[v134] = v132;	// L183
      } else {
        int32_t v135 = tw_k;	// L185
        bool v136 = v135 == 64;	// L186
        if (v136) {	// L187
          float v137 = a_re;	// L188
          float v138 = b_im;	// L189
          float v139 = v137 + v138;	// L190
          #pragma HLS bind_op variable=v139 op=fadd impl=fabric
          int32_t v140 = il;	// L191
          int v141 = v140;	// L192
          o_re[v141] = v139;	// L193
          float v142 = a_im;	// L194
          float v143 = b_re;	// L195
          float v144 = v142 - v143;	// L196
          #pragma HLS bind_op variable=v144 op=fsub impl=fabric
          int32_t v145 = il;	// L197
          int v146 = v145;	// L198
          o_im[v146] = v144;	// L199
          float v147 = a_re;	// L200
          float v148 = b_im;	// L201
          float v149 = v147 - v148;	// L202
          #pragma HLS bind_op variable=v149 op=fsub impl=fabric
          int32_t v150 = iu;	// L203
          int v151 = v150;	// L204
          o_re[v151] = v149;	// L205
          float v152 = a_im;	// L206
          float v153 = b_re;	// L207
          float v154 = v152 + v153;	// L208
          #pragma HLS bind_op variable=v154 op=fadd impl=fabric
          int32_t v155 = iu;	// L209
          int v156 = v155;	// L210
          o_im[v156] = v154;	// L211
        } else {
          int32_t v157 = tw_k;	// L213
          int v158 = v157;	// L214
          float v159 = twr[v158];	// L215
          float tr;	// L216
          tr = v159;	// L217
          int32_t v161 = tw_k;	// L218
          int v162 = v161;	// L219
          float v163 = twi[v162];	// L220
          float ti;	// L221
          ti = v163;	// L222
          float v165 = b_re;	// L223
          float v166 = tr;	// L224
          float v167 = v165 * v166;	// L225
          float v168 = b_im;	// L226
          float v169 = ti;	// L227
          float v170 = v168 * v169;	// L228
          float v171 = v167 - v170;	// L229
          float bw_re;	// L230
          bw_re = v171;	// L231
          float v173 = b_re;	// L232
          float v174 = ti;	// L233
          float v175 = v173 * v174;	// L234
          float v176 = b_im;	// L235
          float v177 = tr;	// L236
          float v178 = v176 * v177;	// L237
          float v179 = v175 + v178;	// L238
          float bw_im;	// L239
          bw_im = v179;	// L240
          float v181 = a_re;	// L241
          float v182 = bw_re;	// L242
          float v183 = v181 + v182;	// L243
          #pragma HLS bind_op variable=v183 op=fadd impl=fabric
          int32_t v184 = il;	// L244
          int v185 = v184;	// L245
          o_re[v185] = v183;	// L246
          float v186 = a_im;	// L247
          float v187 = bw_im;	// L248
          float v188 = v186 + v187;	// L249
          #pragma HLS bind_op variable=v188 op=fadd impl=fabric
          int32_t v189 = il;	// L250
          int v190 = v189;	// L251
          o_im[v190] = v188;	// L252
          float v191 = a_re;	// L253
          float v192 = bw_re;	// L254
          float v193 = v191 - v192;	// L255
          #pragma HLS bind_op variable=v193 op=fsub impl=fabric
          int32_t v194 = iu;	// L256
          int v195 = v194;	// L257
          o_re[v195] = v193;	// L258
          float v196 = a_im;	// L259
          float v197 = bw_im;	// L260
          float v198 = v196 - v197;	// L261
          #pragma HLS bind_op variable=v198 op=fsub impl=fabric
          int32_t v199 = iu;	// L262
          int v200 = v199;	// L263
          o_im[v200] = v198;	// L264
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re[_iv0];
      }
      v65.write(_vec);
    }	// L268
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im[_iv0];
      }
      v66.write(_vec);
    }	// L269
  }
}

void intra_1(
  hls::stream< hls::vector< float, 32 > >& v201,
  hls::stream< hls::vector< float, 32 > >& v202,
  hls::stream< hls::vector< float, 32 > >& v203,
  hls::stream< hls::vector< float, 32 > >& v204
) {	// L273
  // placeholder for const float twr	// L281
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L282
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i1: for (int _i1 = 0; _i1 < 8; _i1++) {	// L283
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v208 = v201.read();
    hls::vector< float, 32 > v209 = v202.read();
    float o_re1[32];	// L286
    #pragma HLS array_partition variable=o_re1 complete
    float o_im1[32];	// L287
    #pragma HLS array_partition variable=o_im1 complete
    int32_t stride1;	// L288
    stride1 = 2;	// L289
    l_S_k_0_k1: for (int k1 = 0; k1 < 16; k1++) {	// L290
    #pragma HLS unroll
      int v214 = k1 >> 1;	// L291
      int v215 = v214 << 2;	// L292
      int32_t v216 = stride1;	// L293
      int64_t v217 = v216;	// L294
      int64_t v218 = v217 - 1;	// L295
      int64_t v219 = k1;	// L296
      int64_t v220 = v219 & v218;	// L297
      int64_t v221 = v215;	// L298
      int64_t v222 = v221 | v220;	// L299
      int32_t v223 = v222;	// L300
      int32_t il1;	// L301
      il1 = v223;	// L302
      int32_t v225 = il1;	// L303
      int32_t v226 = stride1;	// L304
      int32_t v227 = v225 | v226;	// L305
      int32_t iu1;	// L306
      iu1 = v227;	// L307
      int32_t v229 = stride1;	// L308
      int64_t v230 = v229;	// L309
      int64_t v231 = v230 - 1;	// L310
      int64_t v232 = v219 & v231;	// L311
      int64_t v233 = v232 << 6;	// L312
      int32_t v234 = v233;	// L313
      int32_t tw_k1;	// L314
      tw_k1 = v234;	// L315
      int32_t v236 = il1;	// L316
      int v237 = v236;	// L317
      float v238 = v208[v237];	// L318
      float a_re1;	// L319
      a_re1 = v238;	// L320
      int32_t v240 = il1;	// L321
      int v241 = v240;	// L322
      float v242 = v209[v241];	// L323
      float a_im1;	// L324
      a_im1 = v242;	// L325
      int32_t v244 = iu1;	// L326
      int v245 = v244;	// L327
      float v246 = v208[v245];	// L328
      float b_re1;	// L329
      b_re1 = v246;	// L330
      int32_t v248 = iu1;	// L331
      int v249 = v248;	// L332
      float v250 = v209[v249];	// L333
      float b_im1;	// L334
      b_im1 = v250;	// L335
      int32_t v252 = tw_k1;	// L336
      bool v253 = v252 == 0;	// L337
      if (v253) {	// L338
        float v254 = a_re1;	// L339
        float v255 = b_re1;	// L340
        float v256 = v254 + v255;	// L341
        #pragma HLS bind_op variable=v256 op=fadd impl=fabric
        int32_t v257 = il1;	// L342
        int v258 = v257;	// L343
        o_re1[v258] = v256;	// L344
        float v259 = a_im1;	// L345
        float v260 = b_im1;	// L346
        float v261 = v259 + v260;	// L347
        #pragma HLS bind_op variable=v261 op=fadd impl=fabric
        int32_t v262 = il1;	// L348
        int v263 = v262;	// L349
        o_im1[v263] = v261;	// L350
        float v264 = a_re1;	// L351
        float v265 = b_re1;	// L352
        float v266 = v264 - v265;	// L353
        #pragma HLS bind_op variable=v266 op=fsub impl=fabric
        int32_t v267 = iu1;	// L354
        int v268 = v267;	// L355
        o_re1[v268] = v266;	// L356
        float v269 = a_im1;	// L357
        float v270 = b_im1;	// L358
        float v271 = v269 - v270;	// L359
        #pragma HLS bind_op variable=v271 op=fsub impl=fabric
        int32_t v272 = iu1;	// L360
        int v273 = v272;	// L361
        o_im1[v273] = v271;	// L362
      } else {
        int32_t v274 = tw_k1;	// L364
        bool v275 = v274 == 64;	// L365
        if (v275) {	// L366
          float v276 = a_re1;	// L367
          float v277 = b_im1;	// L368
          float v278 = v276 + v277;	// L369
          #pragma HLS bind_op variable=v278 op=fadd impl=fabric
          int32_t v279 = il1;	// L370
          int v280 = v279;	// L371
          o_re1[v280] = v278;	// L372
          float v281 = a_im1;	// L373
          float v282 = b_re1;	// L374
          float v283 = v281 - v282;	// L375
          #pragma HLS bind_op variable=v283 op=fsub impl=fabric
          int32_t v284 = il1;	// L376
          int v285 = v284;	// L377
          o_im1[v285] = v283;	// L378
          float v286 = a_re1;	// L379
          float v287 = b_im1;	// L380
          float v288 = v286 - v287;	// L381
          #pragma HLS bind_op variable=v288 op=fsub impl=fabric
          int32_t v289 = iu1;	// L382
          int v290 = v289;	// L383
          o_re1[v290] = v288;	// L384
          float v291 = a_im1;	// L385
          float v292 = b_re1;	// L386
          float v293 = v291 + v292;	// L387
          #pragma HLS bind_op variable=v293 op=fadd impl=fabric
          int32_t v294 = iu1;	// L388
          int v295 = v294;	// L389
          o_im1[v295] = v293;	// L390
        } else {
          int32_t v296 = tw_k1;	// L392
          int v297 = v296;	// L393
          float v298 = twr[v297];	// L394
          float tr1;	// L395
          tr1 = v298;	// L396
          int32_t v300 = tw_k1;	// L397
          int v301 = v300;	// L398
          float v302 = twi[v301];	// L399
          float ti1;	// L400
          ti1 = v302;	// L401
          float v304 = b_re1;	// L402
          float v305 = tr1;	// L403
          float v306 = v304 * v305;	// L404
          float v307 = b_im1;	// L405
          float v308 = ti1;	// L406
          float v309 = v307 * v308;	// L407
          float v310 = v306 - v309;	// L408
          float bw_re1;	// L409
          bw_re1 = v310;	// L410
          float v312 = b_re1;	// L411
          float v313 = ti1;	// L412
          float v314 = v312 * v313;	// L413
          float v315 = b_im1;	// L414
          float v316 = tr1;	// L415
          float v317 = v315 * v316;	// L416
          float v318 = v314 + v317;	// L417
          float bw_im1;	// L418
          bw_im1 = v318;	// L419
          float v320 = a_re1;	// L420
          float v321 = bw_re1;	// L421
          float v322 = v320 + v321;	// L422
          #pragma HLS bind_op variable=v322 op=fadd impl=fabric
          int32_t v323 = il1;	// L423
          int v324 = v323;	// L424
          o_re1[v324] = v322;	// L425
          float v325 = a_im1;	// L426
          float v326 = bw_im1;	// L427
          float v327 = v325 + v326;	// L428
          #pragma HLS bind_op variable=v327 op=fadd impl=fabric
          int32_t v328 = il1;	// L429
          int v329 = v328;	// L430
          o_im1[v329] = v327;	// L431
          float v330 = a_re1;	// L432
          float v331 = bw_re1;	// L433
          float v332 = v330 - v331;	// L434
          #pragma HLS bind_op variable=v332 op=fsub impl=fabric
          int32_t v333 = iu1;	// L435
          int v334 = v333;	// L436
          o_re1[v334] = v332;	// L437
          float v335 = a_im1;	// L438
          float v336 = bw_im1;	// L439
          float v337 = v335 - v336;	// L440
          #pragma HLS bind_op variable=v337 op=fsub impl=fabric
          int32_t v338 = iu1;	// L441
          int v339 = v338;	// L442
          o_im1[v339] = v337;	// L443
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re1[_iv0];
      }
      v203.write(_vec);
    }	// L447
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im1[_iv0];
      }
      v204.write(_vec);
    }	// L448
  }
}

void intra_2(
  hls::stream< hls::vector< float, 32 > >& v340,
  hls::stream< hls::vector< float, 32 > >& v341,
  hls::stream< hls::vector< float, 32 > >& v342,
  hls::stream< hls::vector< float, 32 > >& v343
) {	// L452
  // placeholder for const float twr	// L460
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L461
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i2: for (int _i2 = 0; _i2 < 8; _i2++) {	// L462
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v347 = v340.read();
    hls::vector< float, 32 > v348 = v341.read();
    float o_re2[32];	// L465
    #pragma HLS array_partition variable=o_re2 complete
    float o_im2[32];	// L466
    #pragma HLS array_partition variable=o_im2 complete
    int32_t stride2;	// L467
    stride2 = 4;	// L468
    l_S_k_0_k2: for (int k2 = 0; k2 < 16; k2++) {	// L469
    #pragma HLS unroll
      int v353 = k2 >> 2;	// L470
      int v354 = v353 << 3;	// L471
      int32_t v355 = stride2;	// L472
      int64_t v356 = v355;	// L473
      int64_t v357 = v356 - 1;	// L474
      int64_t v358 = k2;	// L475
      int64_t v359 = v358 & v357;	// L476
      int64_t v360 = v354;	// L477
      int64_t v361 = v360 | v359;	// L478
      int32_t v362 = v361;	// L479
      int32_t il2;	// L480
      il2 = v362;	// L481
      int32_t v364 = il2;	// L482
      int32_t v365 = stride2;	// L483
      int32_t v366 = v364 | v365;	// L484
      int32_t iu2;	// L485
      iu2 = v366;	// L486
      int32_t v368 = stride2;	// L487
      int64_t v369 = v368;	// L488
      int64_t v370 = v369 - 1;	// L489
      int64_t v371 = v358 & v370;	// L490
      int64_t v372 = v371 << 5;	// L491
      int32_t v373 = v372;	// L492
      int32_t tw_k2;	// L493
      tw_k2 = v373;	// L494
      int32_t v375 = il2;	// L495
      int v376 = v375;	// L496
      float v377 = v347[v376];	// L497
      float a_re2;	// L498
      a_re2 = v377;	// L499
      int32_t v379 = il2;	// L500
      int v380 = v379;	// L501
      float v381 = v348[v380];	// L502
      float a_im2;	// L503
      a_im2 = v381;	// L504
      int32_t v383 = iu2;	// L505
      int v384 = v383;	// L506
      float v385 = v347[v384];	// L507
      float b_re2;	// L508
      b_re2 = v385;	// L509
      int32_t v387 = iu2;	// L510
      int v388 = v387;	// L511
      float v389 = v348[v388];	// L512
      float b_im2;	// L513
      b_im2 = v389;	// L514
      int32_t v391 = tw_k2;	// L515
      bool v392 = v391 == 0;	// L516
      if (v392) {	// L517
        float v393 = a_re2;	// L518
        float v394 = b_re2;	// L519
        float v395 = v393 + v394;	// L520
        #pragma HLS bind_op variable=v395 op=fadd impl=fabric
        int32_t v396 = il2;	// L521
        int v397 = v396;	// L522
        o_re2[v397] = v395;	// L523
        float v398 = a_im2;	// L524
        float v399 = b_im2;	// L525
        float v400 = v398 + v399;	// L526
        #pragma HLS bind_op variable=v400 op=fadd impl=fabric
        int32_t v401 = il2;	// L527
        int v402 = v401;	// L528
        o_im2[v402] = v400;	// L529
        float v403 = a_re2;	// L530
        float v404 = b_re2;	// L531
        float v405 = v403 - v404;	// L532
        #pragma HLS bind_op variable=v405 op=fsub impl=fabric
        int32_t v406 = iu2;	// L533
        int v407 = v406;	// L534
        o_re2[v407] = v405;	// L535
        float v408 = a_im2;	// L536
        float v409 = b_im2;	// L537
        float v410 = v408 - v409;	// L538
        #pragma HLS bind_op variable=v410 op=fsub impl=fabric
        int32_t v411 = iu2;	// L539
        int v412 = v411;	// L540
        o_im2[v412] = v410;	// L541
      } else {
        int32_t v413 = tw_k2;	// L543
        bool v414 = v413 == 64;	// L544
        if (v414) {	// L545
          float v415 = a_re2;	// L546
          float v416 = b_im2;	// L547
          float v417 = v415 + v416;	// L548
          #pragma HLS bind_op variable=v417 op=fadd impl=fabric
          int32_t v418 = il2;	// L549
          int v419 = v418;	// L550
          o_re2[v419] = v417;	// L551
          float v420 = a_im2;	// L552
          float v421 = b_re2;	// L553
          float v422 = v420 - v421;	// L554
          #pragma HLS bind_op variable=v422 op=fsub impl=fabric
          int32_t v423 = il2;	// L555
          int v424 = v423;	// L556
          o_im2[v424] = v422;	// L557
          float v425 = a_re2;	// L558
          float v426 = b_im2;	// L559
          float v427 = v425 - v426;	// L560
          #pragma HLS bind_op variable=v427 op=fsub impl=fabric
          int32_t v428 = iu2;	// L561
          int v429 = v428;	// L562
          o_re2[v429] = v427;	// L563
          float v430 = a_im2;	// L564
          float v431 = b_re2;	// L565
          float v432 = v430 + v431;	// L566
          #pragma HLS bind_op variable=v432 op=fadd impl=fabric
          int32_t v433 = iu2;	// L567
          int v434 = v433;	// L568
          o_im2[v434] = v432;	// L569
        } else {
          int32_t v435 = tw_k2;	// L571
          int v436 = v435;	// L572
          float v437 = twr[v436];	// L573
          float tr2;	// L574
          tr2 = v437;	// L575
          int32_t v439 = tw_k2;	// L576
          int v440 = v439;	// L577
          float v441 = twi[v440];	// L578
          float ti2;	// L579
          ti2 = v441;	// L580
          float v443 = b_re2;	// L581
          float v444 = tr2;	// L582
          float v445 = v443 * v444;	// L583
          float v446 = b_im2;	// L584
          float v447 = ti2;	// L585
          float v448 = v446 * v447;	// L586
          float v449 = v445 - v448;	// L587
          float bw_re2;	// L588
          bw_re2 = v449;	// L589
          float v451 = b_re2;	// L590
          float v452 = ti2;	// L591
          float v453 = v451 * v452;	// L592
          float v454 = b_im2;	// L593
          float v455 = tr2;	// L594
          float v456 = v454 * v455;	// L595
          float v457 = v453 + v456;	// L596
          float bw_im2;	// L597
          bw_im2 = v457;	// L598
          float v459 = a_re2;	// L599
          float v460 = bw_re2;	// L600
          float v461 = v459 + v460;	// L601
          #pragma HLS bind_op variable=v461 op=fadd impl=fabric
          int32_t v462 = il2;	// L602
          int v463 = v462;	// L603
          o_re2[v463] = v461;	// L604
          float v464 = a_im2;	// L605
          float v465 = bw_im2;	// L606
          float v466 = v464 + v465;	// L607
          #pragma HLS bind_op variable=v466 op=fadd impl=fabric
          int32_t v467 = il2;	// L608
          int v468 = v467;	// L609
          o_im2[v468] = v466;	// L610
          float v469 = a_re2;	// L611
          float v470 = bw_re2;	// L612
          float v471 = v469 - v470;	// L613
          #pragma HLS bind_op variable=v471 op=fsub impl=fabric
          int32_t v472 = iu2;	// L614
          int v473 = v472;	// L615
          o_re2[v473] = v471;	// L616
          float v474 = a_im2;	// L617
          float v475 = bw_im2;	// L618
          float v476 = v474 - v475;	// L619
          #pragma HLS bind_op variable=v476 op=fsub impl=fabric
          int32_t v477 = iu2;	// L620
          int v478 = v477;	// L621
          o_im2[v478] = v476;	// L622
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re2[_iv0];
      }
      v342.write(_vec);
    }	// L626
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im2[_iv0];
      }
      v343.write(_vec);
    }	// L627
  }
}

void intra_3(
  hls::stream< hls::vector< float, 32 > >& v479,
  hls::stream< hls::vector< float, 32 > >& v480,
  hls::stream< hls::vector< float, 32 > >& v481,
  hls::stream< hls::vector< float, 32 > >& v482
) {	// L631
  // placeholder for const float twr	// L639
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L640
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i3: for (int _i3 = 0; _i3 < 8; _i3++) {	// L641
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v486 = v479.read();
    hls::vector< float, 32 > v487 = v480.read();
    float o_re3[32];	// L644
    #pragma HLS array_partition variable=o_re3 complete
    float o_im3[32];	// L645
    #pragma HLS array_partition variable=o_im3 complete
    int32_t stride3;	// L646
    stride3 = 8;	// L647
    l_S_k_0_k3: for (int k3 = 0; k3 < 16; k3++) {	// L648
    #pragma HLS unroll
      int v492 = k3 >> 3;	// L649
      int v493 = v492 << 4;	// L650
      int32_t v494 = stride3;	// L651
      int64_t v495 = v494;	// L652
      int64_t v496 = v495 - 1;	// L653
      int64_t v497 = k3;	// L654
      int64_t v498 = v497 & v496;	// L655
      int64_t v499 = v493;	// L656
      int64_t v500 = v499 | v498;	// L657
      int32_t v501 = v500;	// L658
      int32_t il3;	// L659
      il3 = v501;	// L660
      int32_t v503 = il3;	// L661
      int32_t v504 = stride3;	// L662
      int32_t v505 = v503 | v504;	// L663
      int32_t iu3;	// L664
      iu3 = v505;	// L665
      int32_t v507 = stride3;	// L666
      int64_t v508 = v507;	// L667
      int64_t v509 = v508 - 1;	// L668
      int64_t v510 = v497 & v509;	// L669
      int64_t v511 = v510 << 4;	// L670
      int32_t v512 = v511;	// L671
      int32_t tw_k3;	// L672
      tw_k3 = v512;	// L673
      int32_t v514 = il3;	// L674
      int v515 = v514;	// L675
      float v516 = v486[v515];	// L676
      float a_re3;	// L677
      a_re3 = v516;	// L678
      int32_t v518 = il3;	// L679
      int v519 = v518;	// L680
      float v520 = v487[v519];	// L681
      float a_im3;	// L682
      a_im3 = v520;	// L683
      int32_t v522 = iu3;	// L684
      int v523 = v522;	// L685
      float v524 = v486[v523];	// L686
      float b_re3;	// L687
      b_re3 = v524;	// L688
      int32_t v526 = iu3;	// L689
      int v527 = v526;	// L690
      float v528 = v487[v527];	// L691
      float b_im3;	// L692
      b_im3 = v528;	// L693
      int32_t v530 = tw_k3;	// L694
      bool v531 = v530 == 0;	// L695
      if (v531) {	// L696
        float v532 = a_re3;	// L697
        float v533 = b_re3;	// L698
        float v534 = v532 + v533;	// L699
        #pragma HLS bind_op variable=v534 op=fadd impl=fabric
        int32_t v535 = il3;	// L700
        int v536 = v535;	// L701
        o_re3[v536] = v534;	// L702
        float v537 = a_im3;	// L703
        float v538 = b_im3;	// L704
        float v539 = v537 + v538;	// L705
        #pragma HLS bind_op variable=v539 op=fadd impl=fabric
        int32_t v540 = il3;	// L706
        int v541 = v540;	// L707
        o_im3[v541] = v539;	// L708
        float v542 = a_re3;	// L709
        float v543 = b_re3;	// L710
        float v544 = v542 - v543;	// L711
        #pragma HLS bind_op variable=v544 op=fsub impl=fabric
        int32_t v545 = iu3;	// L712
        int v546 = v545;	// L713
        o_re3[v546] = v544;	// L714
        float v547 = a_im3;	// L715
        float v548 = b_im3;	// L716
        float v549 = v547 - v548;	// L717
        #pragma HLS bind_op variable=v549 op=fsub impl=fabric
        int32_t v550 = iu3;	// L718
        int v551 = v550;	// L719
        o_im3[v551] = v549;	// L720
      } else {
        int32_t v552 = tw_k3;	// L722
        bool v553 = v552 == 64;	// L723
        if (v553) {	// L724
          float v554 = a_re3;	// L725
          float v555 = b_im3;	// L726
          float v556 = v554 + v555;	// L727
          #pragma HLS bind_op variable=v556 op=fadd impl=fabric
          int32_t v557 = il3;	// L728
          int v558 = v557;	// L729
          o_re3[v558] = v556;	// L730
          float v559 = a_im3;	// L731
          float v560 = b_re3;	// L732
          float v561 = v559 - v560;	// L733
          #pragma HLS bind_op variable=v561 op=fsub impl=fabric
          int32_t v562 = il3;	// L734
          int v563 = v562;	// L735
          o_im3[v563] = v561;	// L736
          float v564 = a_re3;	// L737
          float v565 = b_im3;	// L738
          float v566 = v564 - v565;	// L739
          #pragma HLS bind_op variable=v566 op=fsub impl=fabric
          int32_t v567 = iu3;	// L740
          int v568 = v567;	// L741
          o_re3[v568] = v566;	// L742
          float v569 = a_im3;	// L743
          float v570 = b_re3;	// L744
          float v571 = v569 + v570;	// L745
          #pragma HLS bind_op variable=v571 op=fadd impl=fabric
          int32_t v572 = iu3;	// L746
          int v573 = v572;	// L747
          o_im3[v573] = v571;	// L748
        } else {
          int32_t v574 = tw_k3;	// L750
          int v575 = v574;	// L751
          float v576 = twr[v575];	// L752
          float tr3;	// L753
          tr3 = v576;	// L754
          int32_t v578 = tw_k3;	// L755
          int v579 = v578;	// L756
          float v580 = twi[v579];	// L757
          float ti3;	// L758
          ti3 = v580;	// L759
          float v582 = b_re3;	// L760
          float v583 = tr3;	// L761
          float v584 = v582 * v583;	// L762
          float v585 = b_im3;	// L763
          float v586 = ti3;	// L764
          float v587 = v585 * v586;	// L765
          float v588 = v584 - v587;	// L766
          float bw_re3;	// L767
          bw_re3 = v588;	// L768
          float v590 = b_re3;	// L769
          float v591 = ti3;	// L770
          float v592 = v590 * v591;	// L771
          float v593 = b_im3;	// L772
          float v594 = tr3;	// L773
          float v595 = v593 * v594;	// L774
          float v596 = v592 + v595;	// L775
          float bw_im3;	// L776
          bw_im3 = v596;	// L777
          float v598 = a_re3;	// L778
          float v599 = bw_re3;	// L779
          float v600 = v598 + v599;	// L780
          #pragma HLS bind_op variable=v600 op=fadd impl=fabric
          int32_t v601 = il3;	// L781
          int v602 = v601;	// L782
          o_re3[v602] = v600;	// L783
          float v603 = a_im3;	// L784
          float v604 = bw_im3;	// L785
          float v605 = v603 + v604;	// L786
          #pragma HLS bind_op variable=v605 op=fadd impl=fabric
          int32_t v606 = il3;	// L787
          int v607 = v606;	// L788
          o_im3[v607] = v605;	// L789
          float v608 = a_re3;	// L790
          float v609 = bw_re3;	// L791
          float v610 = v608 - v609;	// L792
          #pragma HLS bind_op variable=v610 op=fsub impl=fabric
          int32_t v611 = iu3;	// L793
          int v612 = v611;	// L794
          o_re3[v612] = v610;	// L795
          float v613 = a_im3;	// L796
          float v614 = bw_im3;	// L797
          float v615 = v613 - v614;	// L798
          #pragma HLS bind_op variable=v615 op=fsub impl=fabric
          int32_t v616 = iu3;	// L799
          int v617 = v616;	// L800
          o_im3[v617] = v615;	// L801
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re3[_iv0];
      }
      v481.write(_vec);
    }	// L805
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im3[_iv0];
      }
      v482.write(_vec);
    }	// L806
  }
}

void intra_4(
  hls::stream< hls::vector< float, 32 > >& v618,
  hls::stream< hls::vector< float, 32 > >& v619,
  hls::stream< hls::vector< float, 32 > >& v620,
  hls::stream< hls::vector< float, 32 > >& v621
) {	// L810
  // placeholder for const float twr	// L818
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L819
  #pragma HLS array_partition variable=twi complete
  l_S__i_0__i4: for (int _i4 = 0; _i4 < 8; _i4++) {	// L820
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v625 = v618.read();
    hls::vector< float, 32 > v626 = v619.read();
    float o_re4[32];	// L823
    #pragma HLS array_partition variable=o_re4 complete
    float o_im4[32];	// L824
    #pragma HLS array_partition variable=o_im4 complete
    int32_t stride4;	// L825
    stride4 = 16;	// L826
    l_S_k_0_k4: for (int k4 = 0; k4 < 16; k4++) {	// L827
    #pragma HLS unroll
      int v631 = k4 >> 4;	// L828
      int v632 = v631 << 5;	// L829
      int32_t v633 = stride4;	// L830
      int64_t v634 = v633;	// L831
      int64_t v635 = v634 - 1;	// L832
      int64_t v636 = k4;	// L833
      int64_t v637 = v636 & v635;	// L834
      int64_t v638 = v632;	// L835
      int64_t v639 = v638 | v637;	// L836
      int32_t v640 = v639;	// L837
      int32_t il4;	// L838
      il4 = v640;	// L839
      int32_t v642 = il4;	// L840
      int32_t v643 = stride4;	// L841
      int32_t v644 = v642 | v643;	// L842
      int32_t iu4;	// L843
      iu4 = v644;	// L844
      int32_t v646 = stride4;	// L845
      int64_t v647 = v646;	// L846
      int64_t v648 = v647 - 1;	// L847
      int64_t v649 = v636 & v648;	// L848
      int64_t v650 = v649 << 3;	// L849
      int32_t v651 = v650;	// L850
      int32_t tw_k4;	// L851
      tw_k4 = v651;	// L852
      int32_t v653 = il4;	// L853
      int v654 = v653;	// L854
      float v655 = v625[v654];	// L855
      float a_re4;	// L856
      a_re4 = v655;	// L857
      int32_t v657 = il4;	// L858
      int v658 = v657;	// L859
      float v659 = v626[v658];	// L860
      float a_im4;	// L861
      a_im4 = v659;	// L862
      int32_t v661 = iu4;	// L863
      int v662 = v661;	// L864
      float v663 = v625[v662];	// L865
      float b_re4;	// L866
      b_re4 = v663;	// L867
      int32_t v665 = iu4;	// L868
      int v666 = v665;	// L869
      float v667 = v626[v666];	// L870
      float b_im4;	// L871
      b_im4 = v667;	// L872
      int32_t v669 = tw_k4;	// L873
      bool v670 = v669 == 0;	// L874
      if (v670) {	// L875
        float v671 = a_re4;	// L876
        float v672 = b_re4;	// L877
        float v673 = v671 + v672;	// L878
        #pragma HLS bind_op variable=v673 op=fadd impl=fabric
        int32_t v674 = il4;	// L879
        int v675 = v674;	// L880
        o_re4[v675] = v673;	// L881
        float v676 = a_im4;	// L882
        float v677 = b_im4;	// L883
        float v678 = v676 + v677;	// L884
        #pragma HLS bind_op variable=v678 op=fadd impl=fabric
        int32_t v679 = il4;	// L885
        int v680 = v679;	// L886
        o_im4[v680] = v678;	// L887
        float v681 = a_re4;	// L888
        float v682 = b_re4;	// L889
        float v683 = v681 - v682;	// L890
        #pragma HLS bind_op variable=v683 op=fsub impl=fabric
        int32_t v684 = iu4;	// L891
        int v685 = v684;	// L892
        o_re4[v685] = v683;	// L893
        float v686 = a_im4;	// L894
        float v687 = b_im4;	// L895
        float v688 = v686 - v687;	// L896
        #pragma HLS bind_op variable=v688 op=fsub impl=fabric
        int32_t v689 = iu4;	// L897
        int v690 = v689;	// L898
        o_im4[v690] = v688;	// L899
      } else {
        int32_t v691 = tw_k4;	// L901
        bool v692 = v691 == 64;	// L902
        if (v692) {	// L903
          float v693 = a_re4;	// L904
          float v694 = b_im4;	// L905
          float v695 = v693 + v694;	// L906
          #pragma HLS bind_op variable=v695 op=fadd impl=fabric
          int32_t v696 = il4;	// L907
          int v697 = v696;	// L908
          o_re4[v697] = v695;	// L909
          float v698 = a_im4;	// L910
          float v699 = b_re4;	// L911
          float v700 = v698 - v699;	// L912
          #pragma HLS bind_op variable=v700 op=fsub impl=fabric
          int32_t v701 = il4;	// L913
          int v702 = v701;	// L914
          o_im4[v702] = v700;	// L915
          float v703 = a_re4;	// L916
          float v704 = b_im4;	// L917
          float v705 = v703 - v704;	// L918
          #pragma HLS bind_op variable=v705 op=fsub impl=fabric
          int32_t v706 = iu4;	// L919
          int v707 = v706;	// L920
          o_re4[v707] = v705;	// L921
          float v708 = a_im4;	// L922
          float v709 = b_re4;	// L923
          float v710 = v708 + v709;	// L924
          #pragma HLS bind_op variable=v710 op=fadd impl=fabric
          int32_t v711 = iu4;	// L925
          int v712 = v711;	// L926
          o_im4[v712] = v710;	// L927
        } else {
          int32_t v713 = tw_k4;	// L929
          int v714 = v713;	// L930
          float v715 = twr[v714];	// L931
          float tr4;	// L932
          tr4 = v715;	// L933
          int32_t v717 = tw_k4;	// L934
          int v718 = v717;	// L935
          float v719 = twi[v718];	// L936
          float ti4;	// L937
          ti4 = v719;	// L938
          float v721 = b_re4;	// L939
          float v722 = tr4;	// L940
          float v723 = v721 * v722;	// L941
          float v724 = b_im4;	// L942
          float v725 = ti4;	// L943
          float v726 = v724 * v725;	// L944
          float v727 = v723 - v726;	// L945
          float bw_re4;	// L946
          bw_re4 = v727;	// L947
          float v729 = b_re4;	// L948
          float v730 = ti4;	// L949
          float v731 = v729 * v730;	// L950
          float v732 = b_im4;	// L951
          float v733 = tr4;	// L952
          float v734 = v732 * v733;	// L953
          float v735 = v731 + v734;	// L954
          float bw_im4;	// L955
          bw_im4 = v735;	// L956
          float v737 = a_re4;	// L957
          float v738 = bw_re4;	// L958
          float v739 = v737 + v738;	// L959
          #pragma HLS bind_op variable=v739 op=fadd impl=fabric
          int32_t v740 = il4;	// L960
          int v741 = v740;	// L961
          o_re4[v741] = v739;	// L962
          float v742 = a_im4;	// L963
          float v743 = bw_im4;	// L964
          float v744 = v742 + v743;	// L965
          #pragma HLS bind_op variable=v744 op=fadd impl=fabric
          int32_t v745 = il4;	// L966
          int v746 = v745;	// L967
          o_im4[v746] = v744;	// L968
          float v747 = a_re4;	// L969
          float v748 = bw_re4;	// L970
          float v749 = v747 - v748;	// L971
          #pragma HLS bind_op variable=v749 op=fsub impl=fabric
          int32_t v750 = iu4;	// L972
          int v751 = v750;	// L973
          o_re4[v751] = v749;	// L974
          float v752 = a_im4;	// L975
          float v753 = bw_im4;	// L976
          float v754 = v752 - v753;	// L977
          #pragma HLS bind_op variable=v754 op=fsub impl=fabric
          int32_t v755 = iu4;	// L978
          int v756 = v755;	// L979
          o_im4[v756] = v754;	// L980
        }
      }
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_re4[_iv0];
      }
      v620.write(_vec);
    }	// L984
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = o_im4[_iv0];
      }
      v621.write(_vec);
    }	// L985
  }
}

void inter_0(
  hls::stream< hls::vector< float, 32 > >& v757,
  hls::stream< hls::vector< float, 32 > >& v758,
  hls::stream< hls::vector< float, 32 > >& v759,
  hls::stream< hls::vector< float, 32 > >& v760
) {	// L989
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L998
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L999
  #pragma HLS array_partition variable=twi complete
  float in_re[32][8];	// L1000
  #pragma HLS array_partition variable=in_re complete dim=1

  #pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re inter false
  float in_im[32][8];	// L1001
  #pragma HLS array_partition variable=in_im complete dim=1

  #pragma HLS bind_storage variable=in_im type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im inter false
  float out_re_b[32][8];	// L1002
  #pragma HLS array_partition variable=out_re_b complete dim=1

  #pragma HLS bind_storage variable=out_re_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b inter false
  float out_im_b[32][8];	// L1003
  #pragma HLS array_partition variable=out_im_b complete dim=1

  #pragma HLS bind_storage variable=out_im_b type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b inter false
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L1004
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v768 = v757.read();
    hls::vector< float, 32 > v769 = v758.read();
    l_S_k_0_k5: for (int k5 = 0; k5 < 32; k5++) {	// L1007
    #pragma HLS unroll
      float v771 = v768[k5];	// L1008
      int v772 = ((i * 32) + k5);	// L1009
      int32_t v773 = v772;	// L1010
      int32_t v774 = v773 >> 5;	// L1012
      int32_t v775 = v773 & 31;	// L1014
      int32_t v776 = v774 & 1;	// L1016
      int32_t v777 = v776 << 4;	// L1018
      int32_t v778 = v775 ^ v777;	// L1019
      int v779 = v778;	// L1020
      int v780 = v774;	// L1021
      in_re[v779][v780] = v771;	// L1022
      float v781 = v769[k5];	// L1023
      in_im[v779][v780] = v781;	// L1024
    }
  }
  l_S_i_2_i1: for (int i1 = 0; i1 < 8; i1++) {	// L1027
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im inter false
  #pragma HLS dependence variable=in_im intra false
  #pragma HLS dependence variable=in_re inter false
  #pragma HLS dependence variable=in_re intra false
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    l_S_k_2_k6: for (int k6 = 0; k6 < 16; k6++) {	// L1028
    #pragma HLS unroll
      int v784 = i1 << 4;	// L1029
      int v785 = v784 | k6;	// L1030
      uint32_t v786 = v785;	// L1031
      uint32_t bg;	// L1032
      bg = v786;	// L1033
      int32_t v788 = i1;	// L1034
      int32_t v789 = v788 & 1;	// L1035
      int32_t v790 = v789 << 4;	// L1036
      int32_t v791 = k6;	// L1037
      int32_t v792 = v791 | v790;	// L1038
      uint32_t raw_bank;	// L1039
      raw_bank = v792;	// L1040
      int v794 = i1 >> 1;	// L1041
      uint32_t v795 = v794;	// L1042
      uint32_t i_shr;	// L1043
      i_shr = v795;	// L1044
      uint32_t low_mask;	// L1045
      low_mask = 0;	// L1046
      int32_t v798 = i_shr;	// L1047
      int32_t v799 = low_mask;	// L1048
      uint32_t v800 = v798 & v799;	// L1049
      uint32_t low_bits;	// L1050
      low_bits = v800;	// L1051
      int32_t v802 = i_shr;	// L1052
      uint32_t high_bits;	// L1053
      high_bits = v802;	// L1054
      int32_t v804 = high_bits;	// L1055
      uint32_t v805 = v804 << 1;	// L1056
      int32_t v806 = low_bits;	// L1057
      uint32_t v807 = v805 | v806;	// L1058
      uint32_t off_il;	// L1059
      off_il = v807;	// L1060
      uint32_t stride_off;	// L1061
      stride_off = 1;	// L1062
      int32_t v810 = off_il;	// L1063
      int32_t v811 = stride_off;	// L1064
      uint32_t v812 = v810 | v811;	// L1065
      uint32_t off_iu;	// L1066
      off_iu = v812;	// L1067
      int32_t v814 = off_il;	// L1068
      uint32_t v815 = v814 << 5;	// L1069
      int32_t v816 = raw_bank;	// L1070
      uint32_t v817 = v815 | v816;	// L1071
      uint32_t il5;	// L1072
      il5 = v817;	// L1073
      int32_t v819 = off_iu;	// L1074
      uint32_t v820 = v819 << 5;	// L1075
      int32_t v821 = raw_bank;	// L1076
      uint32_t v822 = v820 | v821;	// L1077
      uint32_t iu5;	// L1078
      iu5 = v822;	// L1079
      int32_t v824 = il5;	// L1080
      int32_t v825 = v824 >> 5;	// L1082
      int32_t v826 = v824 & 31;	// L1084
      int32_t v827 = v825 & 1;	// L1086
      int32_t v828 = v827 << 4;	// L1088
      int32_t v829 = v826 ^ v828;	// L1089
      int v830 = v829;	// L1090
      int v831 = v825;	// L1091
      float v832 = in_re[v830][v831];	// L1092
      float a_re5;	// L1093
      a_re5 = v832;	// L1094
      int32_t v834 = il5;	// L1095
      int32_t v835 = v834 >> 5;	// L1096
      int32_t v836 = v834 & 31;	// L1097
      int32_t v837 = v835 & 1;	// L1098
      int32_t v838 = v837 << 4;	// L1099
      int32_t v839 = v836 ^ v838;	// L1100
      int v840 = v839;	// L1101
      int v841 = v835;	// L1102
      float v842 = in_im[v840][v841];	// L1103
      float a_im5;	// L1104
      a_im5 = v842;	// L1105
      int32_t v844 = iu5;	// L1106
      int32_t v845 = v844 >> 5;	// L1107
      int32_t v846 = v844 & 31;	// L1108
      int32_t v847 = v845 & 1;	// L1109
      int32_t v848 = v847 << 4;	// L1110
      int32_t v849 = v846 ^ v848;	// L1111
      int v850 = v849;	// L1112
      int v851 = v845;	// L1113
      float v852 = in_re[v850][v851];	// L1114
      float b_re5;	// L1115
      b_re5 = v852;	// L1116
      int32_t v854 = iu5;	// L1117
      int32_t v855 = v854 >> 5;	// L1118
      int32_t v856 = v854 & 31;	// L1119
      int32_t v857 = v855 & 1;	// L1120
      int32_t v858 = v857 << 4;	// L1121
      int32_t v859 = v856 ^ v858;	// L1122
      int v860 = v859;	// L1123
      int v861 = v855;	// L1124
      float v862 = in_im[v860][v861];	// L1125
      float b_im5;	// L1126
      b_im5 = v862;	// L1127
      int32_t v864 = bg;	// L1128
      int64_t v865 = v864;	// L1129
      int64_t v866 = v865 & 31;	// L1130
      int64_t v867 = v866 << 2;	// L1131
      uint32_t v868 = v867;	// L1132
      uint32_t tw_k5;	// L1133
      tw_k5 = v868;	// L1134
      int32_t v870 = tw_k5;	// L1135
      int v871 = v870;	// L1136
      float v872 = twr[v871];	// L1137
      float tr5;	// L1138
      tr5 = v872;	// L1139
      int32_t v874 = tw_k5;	// L1140
      int v875 = v874;	// L1141
      float v876 = twi[v875];	// L1142
      float ti5;	// L1143
      ti5 = v876;	// L1144
      float v878 = b_re5;	// L1145
      float v879 = tr5;	// L1146
      float v880 = v878 * v879;	// L1147
      float v881 = b_im5;	// L1148
      float v882 = ti5;	// L1149
      float v883 = v881 * v882;	// L1150
      float v884 = v880 - v883;	// L1151
      float bw_re5;	// L1152
      bw_re5 = v884;	// L1153
      float v886 = b_re5;	// L1154
      float v887 = ti5;	// L1155
      float v888 = v886 * v887;	// L1156
      float v889 = b_im5;	// L1157
      float v890 = tr5;	// L1158
      float v891 = v889 * v890;	// L1159
      float v892 = v888 + v891;	// L1160
      float bw_im5;	// L1161
      bw_im5 = v892;	// L1162
      float v894 = a_re5;	// L1163
      float v895 = bw_re5;	// L1164
      float v896 = v894 + v895;	// L1165
      #pragma HLS bind_op variable=v896 op=fadd impl=fabric
      int32_t v897 = il5;	// L1166
      int32_t v898 = v897 >> 5;	// L1167
      int32_t v899 = v897 & 31;	// L1168
      int32_t v900 = v898 & 1;	// L1169
      int32_t v901 = v900 << 4;	// L1170
      int32_t v902 = v899 ^ v901;	// L1171
      int v903 = v902;	// L1172
      int v904 = v898;	// L1173
      out_re_b[v903][v904] = v896;	// L1174
      float v905 = a_im5;	// L1175
      float v906 = bw_im5;	// L1176
      float v907 = v905 + v906;	// L1177
      #pragma HLS bind_op variable=v907 op=fadd impl=fabric
      int32_t v908 = il5;	// L1178
      int32_t v909 = v908 >> 5;	// L1179
      int32_t v910 = v908 & 31;	// L1180
      int32_t v911 = v909 & 1;	// L1181
      int32_t v912 = v911 << 4;	// L1182
      int32_t v913 = v910 ^ v912;	// L1183
      int v914 = v913;	// L1184
      int v915 = v909;	// L1185
      out_im_b[v914][v915] = v907;	// L1186
      float v916 = a_re5;	// L1187
      float v917 = bw_re5;	// L1188
      float v918 = v916 - v917;	// L1189
      #pragma HLS bind_op variable=v918 op=fsub impl=fabric
      int32_t v919 = iu5;	// L1190
      int32_t v920 = v919 >> 5;	// L1191
      int32_t v921 = v919 & 31;	// L1192
      int32_t v922 = v920 & 1;	// L1193
      int32_t v923 = v922 << 4;	// L1194
      int32_t v924 = v921 ^ v923;	// L1195
      int v925 = v924;	// L1196
      int v926 = v920;	// L1197
      out_re_b[v925][v926] = v918;	// L1198
      float v927 = a_im5;	// L1199
      float v928 = bw_im5;	// L1200
      float v929 = v927 - v928;	// L1201
      #pragma HLS bind_op variable=v929 op=fsub impl=fabric
      int32_t v930 = iu5;	// L1202
      int32_t v931 = v930 >> 5;	// L1203
      int32_t v932 = v930 & 31;	// L1204
      int32_t v933 = v931 & 1;	// L1205
      int32_t v934 = v933 << 4;	// L1206
      int32_t v935 = v932 ^ v934;	// L1207
      int v936 = v935;	// L1208
      int v937 = v931;	// L1209
      out_im_b[v936][v937] = v929;	// L1210
    }
  }
  l_S_i_4_i2: for (int i2 = 0; i2 < 8; i2++) {	// L1213
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b inter false
  #pragma HLS dependence variable=out_im_b intra false
  #pragma HLS dependence variable=out_re_b inter false
  #pragma HLS dependence variable=out_re_b intra false
    float chunk_re_out[32];	// L1214
    #pragma HLS array_partition variable=chunk_re_out complete
    float chunk_im_out[32];	// L1215
    #pragma HLS array_partition variable=chunk_im_out complete
    l_S_k_4_k7: for (int k7 = 0; k7 < 32; k7++) {	// L1216
    #pragma HLS unroll
      int v942 = ((i2 * 32) + k7);	// L1217
      int32_t v943 = v942;	// L1218
      int32_t v944 = v943 >> 5;	// L1220
      int32_t v945 = v943 & 31;	// L1222
      int32_t v946 = v944 & 1;	// L1224
      int32_t v947 = v946 << 4;	// L1226
      int32_t v948 = v945 ^ v947;	// L1227
      int v949 = v948;	// L1228
      int v950 = v944;	// L1229
      float v951 = out_re_b[v949][v950];	// L1230
      chunk_re_out[k7] = v951;	// L1231
      float v952 = out_im_b[v949][v950];	// L1232
      chunk_im_out[k7] = v952;	// L1233
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out[_iv0];
      }
      v759.write(_vec);
    }	// L1235
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out[_iv0];
      }
      v760.write(_vec);
    }	// L1236
  }
}

void inter_1(
  hls::stream< hls::vector< float, 32 > >& v953,
  hls::stream< hls::vector< float, 32 > >& v954,
  hls::stream< hls::vector< float, 32 > >& v955,
  hls::stream< hls::vector< float, 32 > >& v956
) {	// L1240
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1249
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1250
  #pragma HLS array_partition variable=twi complete
  float in_re1[32][8];	// L1251
  #pragma HLS array_partition variable=in_re1 complete dim=1

  #pragma HLS bind_storage variable=in_re1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re1 inter false
  float in_im1[32][8];	// L1252
  #pragma HLS array_partition variable=in_im1 complete dim=1

  #pragma HLS bind_storage variable=in_im1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im1 inter false
  float out_re_b1[32][8];	// L1253
  #pragma HLS array_partition variable=out_re_b1 complete dim=1

  #pragma HLS bind_storage variable=out_re_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b1 inter false
  float out_im_b1[32][8];	// L1254
  #pragma HLS array_partition variable=out_im_b1 complete dim=1

  #pragma HLS bind_storage variable=out_im_b1 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b1 inter false
  l_S_i_0_i3: for (int i3 = 0; i3 < 8; i3++) {	// L1255
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v964 = v953.read();
    hls::vector< float, 32 > v965 = v954.read();
    l_S_k_0_k8: for (int k8 = 0; k8 < 32; k8++) {	// L1258
    #pragma HLS unroll
      float v967 = v964[k8];	// L1259
      int v968 = ((i3 * 32) + k8);	// L1260
      int32_t v969 = v968;	// L1261
      int32_t v970 = v969 >> 5;	// L1263
      int32_t v971 = v969 & 31;	// L1265
      int32_t v972 = v970 >> 1;	// L1267
      int32_t v973 = v972 & 1;	// L1268
      int32_t v974 = v973 << 4;	// L1270
      int32_t v975 = v971 ^ v974;	// L1271
      int v976 = v975;	// L1272
      int v977 = v970;	// L1273
      in_re1[v976][v977] = v967;	// L1274
      float v978 = v965[k8];	// L1275
      in_im1[v976][v977] = v978;	// L1276
    }
  }
  l_S_i_2_i4: for (int i4 = 0; i4 < 8; i4++) {	// L1279
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im1 inter false
  #pragma HLS dependence variable=in_im1 intra false
  #pragma HLS dependence variable=in_re1 inter false
  #pragma HLS dependence variable=in_re1 intra false
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    l_S_k_2_k9: for (int k9 = 0; k9 < 16; k9++) {	// L1280
    #pragma HLS unroll
      int v981 = i4 << 4;	// L1281
      int v982 = v981 | k9;	// L1282
      uint32_t v983 = v982;	// L1283
      uint32_t bg1;	// L1284
      bg1 = v983;	// L1285
      int32_t v985 = i4;	// L1286
      int32_t v986 = v985 & 1;	// L1287
      int32_t v987 = v986 << 4;	// L1288
      int32_t v988 = k9;	// L1289
      int32_t v989 = v988 | v987;	// L1290
      uint32_t raw_bank1;	// L1291
      raw_bank1 = v989;	// L1292
      int v991 = i4 >> 1;	// L1293
      uint32_t v992 = v991;	// L1294
      uint32_t i_shr1;	// L1295
      i_shr1 = v992;	// L1296
      uint32_t low_mask1;	// L1297
      low_mask1 = 1;	// L1298
      int32_t v995 = i_shr1;	// L1299
      int32_t v996 = low_mask1;	// L1300
      uint32_t v997 = v995 & v996;	// L1301
      uint32_t low_bits1;	// L1302
      low_bits1 = v997;	// L1303
      int32_t v999 = i_shr1;	// L1304
      uint32_t v1000 = v999 >> 1;	// L1305
      uint32_t high_bits1;	// L1306
      high_bits1 = v1000;	// L1307
      int32_t v1002 = high_bits1;	// L1308
      uint32_t v1003 = v1002 << 2;	// L1309
      int32_t v1004 = low_bits1;	// L1310
      uint32_t v1005 = v1003 | v1004;	// L1311
      uint32_t off_il1;	// L1312
      off_il1 = v1005;	// L1313
      uint32_t stride_off1;	// L1314
      stride_off1 = 2;	// L1315
      int32_t v1008 = off_il1;	// L1316
      int32_t v1009 = stride_off1;	// L1317
      uint32_t v1010 = v1008 | v1009;	// L1318
      uint32_t off_iu1;	// L1319
      off_iu1 = v1010;	// L1320
      int32_t v1012 = off_il1;	// L1321
      uint32_t v1013 = v1012 << 5;	// L1322
      int32_t v1014 = raw_bank1;	// L1323
      uint32_t v1015 = v1013 | v1014;	// L1324
      uint32_t il6;	// L1325
      il6 = v1015;	// L1326
      int32_t v1017 = off_iu1;	// L1327
      uint32_t v1018 = v1017 << 5;	// L1328
      int32_t v1019 = raw_bank1;	// L1329
      uint32_t v1020 = v1018 | v1019;	// L1330
      uint32_t iu6;	// L1331
      iu6 = v1020;	// L1332
      int32_t v1022 = il6;	// L1333
      int32_t v1023 = v1022 >> 5;	// L1335
      int32_t v1024 = v1022 & 31;	// L1337
      int32_t v1025 = v1023 >> 1;	// L1339
      int32_t v1026 = v1025 & 1;	// L1340
      int32_t v1027 = v1026 << 4;	// L1342
      int32_t v1028 = v1024 ^ v1027;	// L1343
      int v1029 = v1028;	// L1344
      int v1030 = v1023;	// L1345
      float v1031 = in_re1[v1029][v1030];	// L1346
      float a_re6;	// L1347
      a_re6 = v1031;	// L1348
      int32_t v1033 = il6;	// L1349
      int32_t v1034 = v1033 >> 5;	// L1350
      int32_t v1035 = v1033 & 31;	// L1351
      int32_t v1036 = v1034 >> 1;	// L1352
      int32_t v1037 = v1036 & 1;	// L1353
      int32_t v1038 = v1037 << 4;	// L1354
      int32_t v1039 = v1035 ^ v1038;	// L1355
      int v1040 = v1039;	// L1356
      int v1041 = v1034;	// L1357
      float v1042 = in_im1[v1040][v1041];	// L1358
      float a_im6;	// L1359
      a_im6 = v1042;	// L1360
      int32_t v1044 = iu6;	// L1361
      int32_t v1045 = v1044 >> 5;	// L1362
      int32_t v1046 = v1044 & 31;	// L1363
      int32_t v1047 = v1045 >> 1;	// L1364
      int32_t v1048 = v1047 & 1;	// L1365
      int32_t v1049 = v1048 << 4;	// L1366
      int32_t v1050 = v1046 ^ v1049;	// L1367
      int v1051 = v1050;	// L1368
      int v1052 = v1045;	// L1369
      float v1053 = in_re1[v1051][v1052];	// L1370
      float b_re6;	// L1371
      b_re6 = v1053;	// L1372
      int32_t v1055 = iu6;	// L1373
      int32_t v1056 = v1055 >> 5;	// L1374
      int32_t v1057 = v1055 & 31;	// L1375
      int32_t v1058 = v1056 >> 1;	// L1376
      int32_t v1059 = v1058 & 1;	// L1377
      int32_t v1060 = v1059 << 4;	// L1378
      int32_t v1061 = v1057 ^ v1060;	// L1379
      int v1062 = v1061;	// L1380
      int v1063 = v1056;	// L1381
      float v1064 = in_im1[v1062][v1063];	// L1382
      float b_im6;	// L1383
      b_im6 = v1064;	// L1384
      int32_t v1066 = bg1;	// L1385
      int64_t v1067 = v1066;	// L1386
      int64_t v1068 = v1067 & 63;	// L1387
      int64_t v1069 = v1068 << 1;	// L1388
      uint32_t v1070 = v1069;	// L1389
      uint32_t tw_k6;	// L1390
      tw_k6 = v1070;	// L1391
      int32_t v1072 = tw_k6;	// L1392
      int v1073 = v1072;	// L1393
      float v1074 = twr[v1073];	// L1394
      float tr6;	// L1395
      tr6 = v1074;	// L1396
      int32_t v1076 = tw_k6;	// L1397
      int v1077 = v1076;	// L1398
      float v1078 = twi[v1077];	// L1399
      float ti6;	// L1400
      ti6 = v1078;	// L1401
      float v1080 = b_re6;	// L1402
      float v1081 = tr6;	// L1403
      float v1082 = v1080 * v1081;	// L1404
      float v1083 = b_im6;	// L1405
      float v1084 = ti6;	// L1406
      float v1085 = v1083 * v1084;	// L1407
      float v1086 = v1082 - v1085;	// L1408
      float bw_re6;	// L1409
      bw_re6 = v1086;	// L1410
      float v1088 = b_re6;	// L1411
      float v1089 = ti6;	// L1412
      float v1090 = v1088 * v1089;	// L1413
      float v1091 = b_im6;	// L1414
      float v1092 = tr6;	// L1415
      float v1093 = v1091 * v1092;	// L1416
      float v1094 = v1090 + v1093;	// L1417
      float bw_im6;	// L1418
      bw_im6 = v1094;	// L1419
      float v1096 = a_re6;	// L1420
      float v1097 = bw_re6;	// L1421
      float v1098 = v1096 + v1097;	// L1422
      #pragma HLS bind_op variable=v1098 op=fadd impl=fabric
      int32_t v1099 = il6;	// L1423
      int32_t v1100 = v1099 >> 5;	// L1424
      int32_t v1101 = v1099 & 31;	// L1425
      int32_t v1102 = v1100 >> 1;	// L1426
      int32_t v1103 = v1102 & 1;	// L1427
      int32_t v1104 = v1103 << 4;	// L1428
      int32_t v1105 = v1101 ^ v1104;	// L1429
      int v1106 = v1105;	// L1430
      int v1107 = v1100;	// L1431
      out_re_b1[v1106][v1107] = v1098;	// L1432
      float v1108 = a_im6;	// L1433
      float v1109 = bw_im6;	// L1434
      float v1110 = v1108 + v1109;	// L1435
      #pragma HLS bind_op variable=v1110 op=fadd impl=fabric
      int32_t v1111 = il6;	// L1436
      int32_t v1112 = v1111 >> 5;	// L1437
      int32_t v1113 = v1111 & 31;	// L1438
      int32_t v1114 = v1112 >> 1;	// L1439
      int32_t v1115 = v1114 & 1;	// L1440
      int32_t v1116 = v1115 << 4;	// L1441
      int32_t v1117 = v1113 ^ v1116;	// L1442
      int v1118 = v1117;	// L1443
      int v1119 = v1112;	// L1444
      out_im_b1[v1118][v1119] = v1110;	// L1445
      float v1120 = a_re6;	// L1446
      float v1121 = bw_re6;	// L1447
      float v1122 = v1120 - v1121;	// L1448
      #pragma HLS bind_op variable=v1122 op=fsub impl=fabric
      int32_t v1123 = iu6;	// L1449
      int32_t v1124 = v1123 >> 5;	// L1450
      int32_t v1125 = v1123 & 31;	// L1451
      int32_t v1126 = v1124 >> 1;	// L1452
      int32_t v1127 = v1126 & 1;	// L1453
      int32_t v1128 = v1127 << 4;	// L1454
      int32_t v1129 = v1125 ^ v1128;	// L1455
      int v1130 = v1129;	// L1456
      int v1131 = v1124;	// L1457
      out_re_b1[v1130][v1131] = v1122;	// L1458
      float v1132 = a_im6;	// L1459
      float v1133 = bw_im6;	// L1460
      float v1134 = v1132 - v1133;	// L1461
      #pragma HLS bind_op variable=v1134 op=fsub impl=fabric
      int32_t v1135 = iu6;	// L1462
      int32_t v1136 = v1135 >> 5;	// L1463
      int32_t v1137 = v1135 & 31;	// L1464
      int32_t v1138 = v1136 >> 1;	// L1465
      int32_t v1139 = v1138 & 1;	// L1466
      int32_t v1140 = v1139 << 4;	// L1467
      int32_t v1141 = v1137 ^ v1140;	// L1468
      int v1142 = v1141;	// L1469
      int v1143 = v1136;	// L1470
      out_im_b1[v1142][v1143] = v1134;	// L1471
    }
  }
  l_S_i_4_i5: for (int i5 = 0; i5 < 8; i5++) {	// L1474
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b1 inter false
  #pragma HLS dependence variable=out_im_b1 intra false
  #pragma HLS dependence variable=out_re_b1 inter false
  #pragma HLS dependence variable=out_re_b1 intra false
    float chunk_re_out1[32];	// L1475
    #pragma HLS array_partition variable=chunk_re_out1 complete
    float chunk_im_out1[32];	// L1476
    #pragma HLS array_partition variable=chunk_im_out1 complete
    l_S_k_4_k10: for (int k10 = 0; k10 < 32; k10++) {	// L1477
    #pragma HLS unroll
      int v1148 = ((i5 * 32) + k10);	// L1478
      int32_t v1149 = v1148;	// L1479
      int32_t v1150 = v1149 >> 5;	// L1481
      int32_t v1151 = v1149 & 31;	// L1483
      int32_t v1152 = v1150 >> 1;	// L1485
      int32_t v1153 = v1152 & 1;	// L1486
      int32_t v1154 = v1153 << 4;	// L1488
      int32_t v1155 = v1151 ^ v1154;	// L1489
      int v1156 = v1155;	// L1490
      int v1157 = v1150;	// L1491
      float v1158 = out_re_b1[v1156][v1157];	// L1492
      chunk_re_out1[k10] = v1158;	// L1493
      float v1159 = out_im_b1[v1156][v1157];	// L1494
      chunk_im_out1[k10] = v1159;	// L1495
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out1[_iv0];
      }
      v955.write(_vec);
    }	// L1497
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out1[_iv0];
      }
      v956.write(_vec);
    }	// L1498
  }
}

void inter_2(
  hls::stream< hls::vector< float, 32 > >& v1160,
  hls::stream< hls::vector< float, 32 > >& v1161,
  hls::stream< hls::vector< float, 32 > >& v1162,
  hls::stream< hls::vector< float, 32 > >& v1163
) {	// L1502
  #pragma HLS dataflow disable_start_propagation
  // placeholder for const float twr	// L1511
  #pragma HLS array_partition variable=twr complete
  // placeholder for const float twi	// L1512
  #pragma HLS array_partition variable=twi complete
  float in_re2[32][8];	// L1513
  #pragma HLS array_partition variable=in_re2 complete dim=1

  #pragma HLS bind_storage variable=in_re2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_re2 inter false
  float in_im2[32][8];	// L1514
  #pragma HLS array_partition variable=in_im2 complete dim=1

  #pragma HLS bind_storage variable=in_im2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=in_im2 inter false
  float out_re_b2[32][8];	// L1515
  #pragma HLS array_partition variable=out_re_b2 complete dim=1

  #pragma HLS bind_storage variable=out_re_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_re_b2 inter false
  float out_im_b2[32][8];	// L1516
  #pragma HLS array_partition variable=out_im_b2 complete dim=1

  #pragma HLS bind_storage variable=out_im_b2 type=ram_2p impl=lutram
  #pragma HLS dependence variable=out_im_b2 inter false
  l_S_i_0_i6: for (int i6 = 0; i6 < 8; i6++) {	// L1517
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1171 = v1160.read();
    hls::vector< float, 32 > v1172 = v1161.read();
    l_S_k_0_k11: for (int k11 = 0; k11 < 32; k11++) {	// L1520
    #pragma HLS unroll
      float v1174 = v1171[k11];	// L1521
      int v1175 = ((i6 * 32) + k11);	// L1522
      int32_t v1176 = v1175;	// L1523
      int32_t v1177 = v1176 >> 5;	// L1525
      int32_t v1178 = v1176 & 31;	// L1527
      int32_t v1179 = v1177 >> 2;	// L1530
      int32_t v1180 = v1179 & 1;	// L1531
      int32_t v1181 = v1180 << 4;	// L1533
      int32_t v1182 = v1178 ^ v1181;	// L1534
      int v1183 = v1182;	// L1535
      int v1184 = v1177;	// L1536
      in_re2[v1183][v1184] = v1174;	// L1537
      float v1185 = v1172[k11];	// L1538
      in_im2[v1183][v1184] = v1185;	// L1539
    }
  }
  l_S_i_2_i7: for (int i7 = 0; i7 < 8; i7++) {	// L1542
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=in_im2 inter false
  #pragma HLS dependence variable=in_im2 intra false
  #pragma HLS dependence variable=in_re2 inter false
  #pragma HLS dependence variable=in_re2 intra false
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    l_S_k_2_k12: for (int k12 = 0; k12 < 16; k12++) {	// L1543
    #pragma HLS unroll
      int v1188 = i7 << 4;	// L1544
      int v1189 = v1188 | k12;	// L1545
      uint32_t v1190 = v1189;	// L1546
      uint32_t bg2;	// L1547
      bg2 = v1190;	// L1548
      int32_t v1192 = i7;	// L1549
      int32_t v1193 = v1192 & 1;	// L1550
      int32_t v1194 = v1193 << 4;	// L1551
      int32_t v1195 = k12;	// L1552
      int32_t v1196 = v1195 | v1194;	// L1553
      uint32_t raw_bank2;	// L1554
      raw_bank2 = v1196;	// L1555
      int v1198 = i7 >> 1;	// L1556
      uint32_t v1199 = v1198;	// L1557
      uint32_t i_shr2;	// L1558
      i_shr2 = v1199;	// L1559
      uint32_t low_mask2;	// L1560
      low_mask2 = 3;	// L1561
      int32_t v1202 = i_shr2;	// L1562
      int32_t v1203 = low_mask2;	// L1563
      uint32_t v1204 = v1202 & v1203;	// L1564
      uint32_t low_bits2;	// L1565
      low_bits2 = v1204;	// L1566
      int32_t v1206 = i_shr2;	// L1567
      uint32_t v1207 = v1206 >> 2;	// L1568
      uint32_t high_bits2;	// L1569
      high_bits2 = v1207;	// L1570
      int32_t v1209 = high_bits2;	// L1571
      uint32_t v1210 = v1209 << 3;	// L1572
      int32_t v1211 = low_bits2;	// L1573
      uint32_t v1212 = v1210 | v1211;	// L1574
      uint32_t off_il2;	// L1575
      off_il2 = v1212;	// L1576
      uint32_t stride_off2;	// L1577
      stride_off2 = 4;	// L1578
      int32_t v1215 = off_il2;	// L1579
      int32_t v1216 = stride_off2;	// L1580
      uint32_t v1217 = v1215 | v1216;	// L1581
      uint32_t off_iu2;	// L1582
      off_iu2 = v1217;	// L1583
      int32_t v1219 = off_il2;	// L1584
      uint32_t v1220 = v1219 << 5;	// L1585
      int32_t v1221 = raw_bank2;	// L1586
      uint32_t v1222 = v1220 | v1221;	// L1587
      uint32_t il7;	// L1588
      il7 = v1222;	// L1589
      int32_t v1224 = off_iu2;	// L1590
      uint32_t v1225 = v1224 << 5;	// L1591
      int32_t v1226 = raw_bank2;	// L1592
      uint32_t v1227 = v1225 | v1226;	// L1593
      uint32_t iu7;	// L1594
      iu7 = v1227;	// L1595
      int32_t v1229 = il7;	// L1596
      int32_t v1230 = v1229 >> 5;	// L1598
      int32_t v1231 = v1229 & 31;	// L1600
      int32_t v1232 = v1230 >> 2;	// L1603
      int32_t v1233 = v1232 & 1;	// L1604
      int32_t v1234 = v1233 << 4;	// L1606
      int32_t v1235 = v1231 ^ v1234;	// L1607
      int v1236 = v1235;	// L1608
      int v1237 = v1230;	// L1609
      float v1238 = in_re2[v1236][v1237];	// L1610
      float a_re7;	// L1611
      a_re7 = v1238;	// L1612
      int32_t v1240 = il7;	// L1613
      int32_t v1241 = v1240 >> 5;	// L1614
      int32_t v1242 = v1240 & 31;	// L1615
      int32_t v1243 = v1241 >> 2;	// L1616
      int32_t v1244 = v1243 & 1;	// L1617
      int32_t v1245 = v1244 << 4;	// L1618
      int32_t v1246 = v1242 ^ v1245;	// L1619
      int v1247 = v1246;	// L1620
      int v1248 = v1241;	// L1621
      float v1249 = in_im2[v1247][v1248];	// L1622
      float a_im7;	// L1623
      a_im7 = v1249;	// L1624
      int32_t v1251 = iu7;	// L1625
      int32_t v1252 = v1251 >> 5;	// L1626
      int32_t v1253 = v1251 & 31;	// L1627
      int32_t v1254 = v1252 >> 2;	// L1628
      int32_t v1255 = v1254 & 1;	// L1629
      int32_t v1256 = v1255 << 4;	// L1630
      int32_t v1257 = v1253 ^ v1256;	// L1631
      int v1258 = v1257;	// L1632
      int v1259 = v1252;	// L1633
      float v1260 = in_re2[v1258][v1259];	// L1634
      float b_re7;	// L1635
      b_re7 = v1260;	// L1636
      int32_t v1262 = iu7;	// L1637
      int32_t v1263 = v1262 >> 5;	// L1638
      int32_t v1264 = v1262 & 31;	// L1639
      int32_t v1265 = v1263 >> 2;	// L1640
      int32_t v1266 = v1265 & 1;	// L1641
      int32_t v1267 = v1266 << 4;	// L1642
      int32_t v1268 = v1264 ^ v1267;	// L1643
      int v1269 = v1268;	// L1644
      int v1270 = v1263;	// L1645
      float v1271 = in_im2[v1269][v1270];	// L1646
      float b_im7;	// L1647
      b_im7 = v1271;	// L1648
      int32_t v1273 = bg2;	// L1649
      int64_t v1274 = v1273;	// L1650
      int64_t v1275 = v1274 & 127;	// L1651
      uint32_t v1276 = v1275;	// L1652
      uint32_t tw_k7;	// L1653
      tw_k7 = v1276;	// L1654
      int32_t v1278 = tw_k7;	// L1655
      int v1279 = v1278;	// L1656
      float v1280 = twr[v1279];	// L1657
      float tr7;	// L1658
      tr7 = v1280;	// L1659
      int32_t v1282 = tw_k7;	// L1660
      int v1283 = v1282;	// L1661
      float v1284 = twi[v1283];	// L1662
      float ti7;	// L1663
      ti7 = v1284;	// L1664
      float v1286 = b_re7;	// L1665
      float v1287 = tr7;	// L1666
      float v1288 = v1286 * v1287;	// L1667
      float v1289 = b_im7;	// L1668
      float v1290 = ti7;	// L1669
      float v1291 = v1289 * v1290;	// L1670
      float v1292 = v1288 - v1291;	// L1671
      float bw_re7;	// L1672
      bw_re7 = v1292;	// L1673
      float v1294 = b_re7;	// L1674
      float v1295 = ti7;	// L1675
      float v1296 = v1294 * v1295;	// L1676
      float v1297 = b_im7;	// L1677
      float v1298 = tr7;	// L1678
      float v1299 = v1297 * v1298;	// L1679
      float v1300 = v1296 + v1299;	// L1680
      float bw_im7;	// L1681
      bw_im7 = v1300;	// L1682
      float v1302 = a_re7;	// L1683
      float v1303 = bw_re7;	// L1684
      float v1304 = v1302 + v1303;	// L1685
      #pragma HLS bind_op variable=v1304 op=fadd impl=fabric
      int32_t v1305 = il7;	// L1686
      int32_t v1306 = v1305 >> 5;	// L1687
      int32_t v1307 = v1305 & 31;	// L1688
      int32_t v1308 = v1306 >> 2;	// L1689
      int32_t v1309 = v1308 & 1;	// L1690
      int32_t v1310 = v1309 << 4;	// L1691
      int32_t v1311 = v1307 ^ v1310;	// L1692
      int v1312 = v1311;	// L1693
      int v1313 = v1306;	// L1694
      out_re_b2[v1312][v1313] = v1304;	// L1695
      float v1314 = a_im7;	// L1696
      float v1315 = bw_im7;	// L1697
      float v1316 = v1314 + v1315;	// L1698
      #pragma HLS bind_op variable=v1316 op=fadd impl=fabric
      int32_t v1317 = il7;	// L1699
      int32_t v1318 = v1317 >> 5;	// L1700
      int32_t v1319 = v1317 & 31;	// L1701
      int32_t v1320 = v1318 >> 2;	// L1702
      int32_t v1321 = v1320 & 1;	// L1703
      int32_t v1322 = v1321 << 4;	// L1704
      int32_t v1323 = v1319 ^ v1322;	// L1705
      int v1324 = v1323;	// L1706
      int v1325 = v1318;	// L1707
      out_im_b2[v1324][v1325] = v1316;	// L1708
      float v1326 = a_re7;	// L1709
      float v1327 = bw_re7;	// L1710
      float v1328 = v1326 - v1327;	// L1711
      #pragma HLS bind_op variable=v1328 op=fsub impl=fabric
      int32_t v1329 = iu7;	// L1712
      int32_t v1330 = v1329 >> 5;	// L1713
      int32_t v1331 = v1329 & 31;	// L1714
      int32_t v1332 = v1330 >> 2;	// L1715
      int32_t v1333 = v1332 & 1;	// L1716
      int32_t v1334 = v1333 << 4;	// L1717
      int32_t v1335 = v1331 ^ v1334;	// L1718
      int v1336 = v1335;	// L1719
      int v1337 = v1330;	// L1720
      out_re_b2[v1336][v1337] = v1328;	// L1721
      float v1338 = a_im7;	// L1722
      float v1339 = bw_im7;	// L1723
      float v1340 = v1338 - v1339;	// L1724
      #pragma HLS bind_op variable=v1340 op=fsub impl=fabric
      int32_t v1341 = iu7;	// L1725
      int32_t v1342 = v1341 >> 5;	// L1726
      int32_t v1343 = v1341 & 31;	// L1727
      int32_t v1344 = v1342 >> 2;	// L1728
      int32_t v1345 = v1344 & 1;	// L1729
      int32_t v1346 = v1345 << 4;	// L1730
      int32_t v1347 = v1343 ^ v1346;	// L1731
      int v1348 = v1347;	// L1732
      int v1349 = v1342;	// L1733
      out_im_b2[v1348][v1349] = v1340;	// L1734
    }
  }
  l_S_i_4_i8: for (int i8 = 0; i8 < 8; i8++) {	// L1737
  #pragma HLS pipeline II=1
  #pragma HLS dependence variable=out_im_b2 inter false
  #pragma HLS dependence variable=out_im_b2 intra false
  #pragma HLS dependence variable=out_re_b2 inter false
  #pragma HLS dependence variable=out_re_b2 intra false
    float chunk_re_out2[32];	// L1738
    #pragma HLS array_partition variable=chunk_re_out2 complete
    float chunk_im_out2[32];	// L1739
    #pragma HLS array_partition variable=chunk_im_out2 complete
    l_S_k_4_k13: for (int k13 = 0; k13 < 32; k13++) {	// L1740
    #pragma HLS unroll
      int v1354 = ((i8 * 32) + k13);	// L1741
      int32_t v1355 = v1354;	// L1742
      int32_t v1356 = v1355 >> 5;	// L1744
      int32_t v1357 = v1355 & 31;	// L1746
      int32_t v1358 = v1356 >> 2;	// L1749
      int32_t v1359 = v1358 & 1;	// L1750
      int32_t v1360 = v1359 << 4;	// L1752
      int32_t v1361 = v1357 ^ v1360;	// L1753
      int v1362 = v1361;	// L1754
      int v1363 = v1356;	// L1755
      float v1364 = out_re_b2[v1362][v1363];	// L1756
      chunk_re_out2[k13] = v1364;	// L1757
      float v1365 = out_im_b2[v1362][v1363];	// L1758
      chunk_im_out2[k13] = v1365;	// L1759
    }
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_re_out2[_iv0];
      }
      v1162.write(_vec);
    }	// L1761
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = chunk_im_out2[_iv0];
      }
      v1163.write(_vec);
    }	// L1762
  }
}

void output_stage_0(
  hls::stream< hls::vector< float, 32 > >& v1366,
  hls::stream< hls::vector< float, 32 > >& v1367,
  hls::stream< hls::vector< float, 32 > >& v1368,
  hls::stream< hls::vector< float, 32 > >& v1369
) {	// L1766
  l_S_i_0_i9: for (int i9 = 0; i9 < 8; i9++) {	// L1767
  #pragma HLS pipeline II=1
    hls::vector< float, 32 > v1371 = v1366.read();
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = v1371[_iv0];
      }
      v1367.write(_vec);
    }	// L1769
    hls::vector< float, 32 > v1372 = v1368.read();
    {
      hls::vector< float, 32 > _vec;
      for (int _iv0 = 0; _iv0 < 32; ++_iv0) {
        #pragma HLS unroll
        _vec[_iv0] = v1372[_iv0];
      }
      v1369.write(_vec);
    }	// L1771
  }
}

/// This is top function.
void fft_256(
  hls::stream< hls::vector< float, 32 > >& v1373,
  hls::stream< hls::vector< float, 32 > >& v1374,
  hls::stream< hls::vector< float, 32 > >& v1375,
  hls::stream< hls::vector< float, 32 > >& v1376
) {	// L1775
  #pragma HLS dataflow disable_start_propagation
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1377;
  #pragma HLS stream variable=v1377 depth=2	// L1776
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1378;
  #pragma HLS stream variable=v1378 depth=2	// L1777
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1379;
  #pragma HLS stream variable=v1379 depth=2	// L1778
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1380;
  #pragma HLS stream variable=v1380 depth=2	// L1779
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1381;
  #pragma HLS stream variable=v1381 depth=2	// L1780
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1382;
  #pragma HLS stream variable=v1382 depth=2	// L1781
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1383;
  #pragma HLS stream variable=v1383 depth=2	// L1782
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1384;
  #pragma HLS stream variable=v1384 depth=2	// L1783
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1385;
  #pragma HLS stream variable=v1385 depth=2	// L1784
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1386;
  #pragma HLS stream variable=v1386 depth=2	// L1785
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1387;
  #pragma HLS stream variable=v1387 depth=2	// L1786
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1388;
  #pragma HLS stream variable=v1388 depth=2	// L1787
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1389;
  #pragma HLS stream variable=v1389 depth=2	// L1788
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1390;
  #pragma HLS stream variable=v1390 depth=2	// L1789
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1391;
  #pragma HLS stream variable=v1391 depth=2	// L1790
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1392;
  #pragma HLS stream variable=v1392 depth=2	// L1791
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1393;
  #pragma HLS stream variable=v1393 depth=2	// L1792
  // Stream of vectors: each vector packs float array[32] into hls::vector<float, 32>
  hls::stream< hls::vector< float, 32 > > v1394;
  #pragma HLS stream variable=v1394 depth=2	// L1793
  bit_rev_stage_0(v1373, v1374, v1377, v1386);	// L1794
  intra_0(v1377, v1386, v1378, v1387);	// L1795
  intra_1(v1378, v1387, v1379, v1388);	// L1796
  intra_2(v1379, v1388, v1380, v1389);	// L1797
  intra_3(v1380, v1389, v1381, v1390);	// L1798
  intra_4(v1381, v1390, v1382, v1391);	// L1799
  inter_0(v1382, v1391, v1383, v1392);	// L1800
  inter_1(v1383, v1392, v1384, v1393);	// L1801
  inter_2(v1384, v1393, v1385, v1394);	// L1802
  output_stage_0(v1385, v1375, v1394, v1376);	// L1803
}



//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
void vpu_r2_0(
  hls::stream< hls::vector< int32_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t v5[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v5[_iv0] = _vec[_iv0];
    }
  }	// L28
  int32_t v6 = v1.read();	// L29
  int32_t header;	// L30
  header = v6;	// L31
  int32_t v8 = header;	// L32
  v2.write(v8);	// L33
  int32_t v9 = header;	// L34
  int32_t v10 = v9 & 65535;	// L35
  int32_t plen;	// L36
  plen = v10;	// L37
  int32_t v12 = header;	// L38
  int32_t v13 = v12 >> 16;	// L39
  int32_t v14 = v13 & 65535;	// L40
  int32_t nouts;	// L41
  nouts = v14;	// L42
  int32_t prog[16];	// L43
  for (int v17 = 0; v17 < 16; v17++) {	// L44
    prog[v17] = 0;	// L44
  }
  int32_t v18 = plen;	// L45
  int v19 = v18;	// L46
  for (int v20 = 0; v20 < v19; v20 += 1) {	// L47
    int32_t v21 = v1.read();	// L48
    int32_t word;	// L49
    word = v21;	// L50
    int32_t v23 = word;	// L51
    prog[v20] = v23;	// L52
    int32_t v24 = word;	// L53
    v2.write(v24);	// L54
  }
  int32_t v25 = plen;	// L56
  ap_int<33> v26 = v25;	// L57
  ap_int<33> v27 = 16 - v26;	// L58
  int v28 = v27;	// L59
  for (int v29 = 0; v29 < v28; v29 += 1) {	// L60
    int32_t v30 = v1.read();	// L61
    int32_t spare;	// L62
    spare = v30;	// L63
    int32_t v32 = spare;	// L64
    v2.write(v32);	// L65
  }
  int32_t v33 = v5[1];	// L67
  int32_t denom;	// L68
  denom = v33;	// L69
  int32_t rcp;	// L70
  rcp = 0;	// L71
  int32_t v36 = denom;	// L72
  bool v37 = v36 > 0;	// L73
  if (v37) {	// L74
    int32_t v38 = denom;	// L75
    int32_t v39 = 16384 / v38;	// L76
    rcp = v39;	// L77
  }
  int32_t reg[4];	// L79
  for (int v41 = 0; v41 < 4; v41++) {	// L80
    reg[v41] = 0;	// L80
  }
  l_S_r0_2_r0: for (int r0 = 0; r0 < 4; r0++) {	// L81
    reg[r0] = 0;	// L82
  }
  int32_t v43 = nouts;	// L84
  int v44 = v43;	// L85
  for (int v45 = 0; v45 < v44; v45 += 1) {	// L86
    int32_t v46 = plen;	// L87
    int v47 = v46;	// L88
    for (int v48 = 0; v48 < v47; v48 += 1) {	// L89
      int32_t v49 = prog[v48];	// L90
      int32_t word2;	// L91
      word2 = v49;	// L92
      int32_t v51 = word2;	// L93
      int32_t v52 = v51 >> 24;	// L94
      int32_t v53 = v52 & 255;	// L95
      int32_t opcode;	// L96
      opcode = v53;	// L97
      int32_t v55 = word2;	// L98
      int32_t v56 = v55 >> 20;	// L99
      int32_t v57 = v56 & 15;	// L100
      int32_t dst;	// L101
      dst = v57;	// L102
      int32_t v59 = word2;	// L103
      int32_t v60 = v59 >> 16;	// L104
      int32_t v61 = v60 & 15;	// L105
      int32_t src;	// L106
      src = v61;	// L107
      int32_t v63 = word2;	// L108
      int32_t v64 = v63 & 65535;	// L109
      int32_t imm;	// L110
      imm = v64;	// L111
      int32_t v66 = opcode;	// L112
      bool v67 = v66 == 13;	// L113
      if (v67) {	// L114
        int32_t v68 = dst;	// L115
        int v69 = v68;	// L116
        int32_t v70 = reg[v69];	// L117
        int32_t acc;	// L118
        acc = v70;	// L119
        int32_t v72 = imm;	// L120
        int v73 = v72;	// L121
        for (int v74 = 0; v74 < v73; v74 += 1) {	// L122
          int32_t v75 = v3.read();	// L123
          int32_t zz;	// L124
          zz = v75;	// L125
          int32_t v77 = acc;	// L126
          int32_t v78 = zz;	// L127
          ap_int<33> v79 = v77;	// L128
          ap_int<33> v80 = v78;	// L129
          ap_int<33> v81 = v79 + v80;	// L130
          int32_t v82 = v81;	// L131
          acc = v82;	// L132
        }
        int32_t v83 = acc;	// L134
        int32_t v84 = dst;	// L135
        int v85 = v84;	// L136
        reg[v85] = v83;	// L137
      } else {
        int32_t v86 = opcode;	// L139
        bool v87 = v86 == 9;	// L140
        if (v87) {	// L141
          int32_t v88 = v3.read();	// L142
          int32_t z1;	// L143
          z1 = v88;	// L144
          int32_t v90 = dst;	// L145
          int v91 = v90;	// L146
          int32_t v92 = reg[v91];	// L147
          int32_t v93 = z1;	// L148
          ap_int<33> v94 = v92;	// L149
          ap_int<33> v95 = v93;	// L150
          ap_int<33> v96 = v94 + v95;	// L151
          int32_t v97 = v96;	// L152
          reg[v91] = v97;	// L155
        } else {
          int32_t v98 = opcode;	// L157
          bool v99 = v98 == 1;	// L158
          if (v99) {	// L159
            int32_t v100 = v3.read();	// L160
            int32_t z2;	// L161
            z2 = v100;	// L162
            int32_t v102 = z2;	// L163
            int32_t v103 = dst;	// L164
            int v104 = v103;	// L165
            reg[v104] = v102;	// L166
          } else {
            int32_t v105 = opcode;	// L168
            bool v106 = v105 == 2;	// L169
            if (v106) {	// L170
              int32_t v107 = src;	// L171
              int v108 = v107;	// L172
              int32_t v109 = v5[v108];	// L173
              int32_t v110 = dst;	// L174
              int v111 = v110;	// L175
              reg[v111] = v109;	// L176
            } else {
              int32_t v112 = opcode;	// L178
              bool v113 = v112 == 3;	// L179
              if (v113) {	// L180
                int32_t v114 = imm;	// L181
                int32_t v115 = dst;	// L182
                int v116 = v115;	// L183
                reg[v116] = v114;	// L184
              } else {
                int32_t v117 = opcode;	// L186
                bool v118 = v117 == 4;	// L187
                if (v118) {	// L188
                  int32_t v119 = dst;	// L189
                  int v120 = v119;	// L190
                  int32_t v121 = reg[v120];	// L191
                  int32_t v122 = src;	// L192
                  int v123 = v122;	// L193
                  int32_t v124 = reg[v123];	// L194
                  ap_int<33> v125 = v121;	// L195
                  ap_int<33> v126 = v124;	// L196
                  ap_int<33> v127 = v125 + v126;	// L197
                  int32_t v128 = v127;	// L198
                  reg[v120] = v128;	// L201
                } else {
                  int32_t v129 = opcode;	// L203
                  bool v130 = v129 == 10;	// L204
                  if (v130) {	// L205
                    int32_t v131 = dst;	// L206
                    int v132 = v131;	// L207
                    int32_t v133 = reg[v132];	// L208
                    int32_t v134 = src;	// L209
                    int v135 = v134;	// L210
                    int32_t v136 = reg[v135];	// L211
                    ap_int<33> v137 = v133;	// L212
                    ap_int<33> v138 = v136;	// L213
                    ap_int<33> v139 = v137 - v138;	// L214
                    int32_t v140 = v139;	// L215
                    reg[v132] = v140;	// L218
                  } else {
                    int32_t v141 = opcode;	// L220
                    bool v142 = v141 == 5;	// L221
                    if (v142) {	// L222
                      int32_t v143 = dst;	// L223
                      int v144 = v143;	// L224
                      int32_t v145 = reg[v144];	// L225
                      int32_t v146 = src;	// L226
                      int v147 = v146;	// L227
                      int32_t v148 = reg[v147];	// L228
                      int64_t v149 = v145;	// L229
                      int64_t v150 = v148;	// L230
                      int64_t v151 = v149 * v150;	// L231
                      int32_t v152 = v151;	// L232
                      reg[v144] = v152;	// L235
                    } else {
                      int32_t v153 = opcode;	// L237
                      bool v154 = v153 == 6;	// L238
                      if (v154) {	// L239
                        int32_t v155 = src;	// L240
                        int v156 = v155;	// L241
                        int32_t v157 = reg[v156];	// L242
                        int32_t v158 = dst;	// L243
                        int v159 = v158;	// L244
                        int32_t v160 = reg[v159];	// L245
                        bool v161 = v157 > v160;	// L246
                        if (v161) {	// L247
                          int32_t v162 = src;	// L248
                          int v163 = v162;	// L249
                          int32_t v164 = reg[v163];	// L250
                          int32_t v165 = dst;	// L251
                          int v166 = v165;	// L252
                          reg[v166] = v164;	// L253
                        }
                      } else {
                        int32_t v167 = opcode;	// L256
                        bool v168 = v167 == 7;	// L257
                        if (v168) {	// L258
                          int32_t v169 = dst;	// L259
                          int v170 = v169;	// L260
                          int32_t v171 = reg[v170];	// L261
                          int32_t v172 = imm;	// L262
                          int32_t v173 = v171 >> v172;	// L263
                          reg[v170] = v173;	// L266
                        } else {
                          int32_t v174 = opcode;	// L268
                          bool v175 = v174 == 11;	// L269
                          if (v175) {	// L270
                            int32_t v176 = dst;	// L271
                            int v177 = v176;	// L272
                            int32_t v178 = reg[v177];	// L273
                            int32_t e;	// L274
                            e = v178;	// L275
                            int32_t v180 = e;	// L276
                            bool v181 = v180 < 0;	// L277
                            if (v181) {	// L278
                              e = 0;	// L279
                            }
                            int32_t v182 = e;	// L281
                            bool v183 = v182 > 30;	// L282
                            if (v183) {	// L283
                              e = 30;	// L284
                            }
                            int32_t v184 = e;	// L286
                            int32_t v185 = 1 << v184;	// L287
                            int32_t v186 = dst;	// L288
                            int v187 = v186;	// L289
                            reg[v187] = v185;	// L290
                          } else {
                            int32_t v188 = opcode;	// L292
                            bool v189 = v188 == 12;	// L293
                            if (v189) {	// L294
                              int32_t v190 = rcp;	// L295
                              int32_t v191 = dst;	// L296
                              int v192 = v191;	// L297
                              reg[v192] = v190;	// L298
                            } else {
                              int32_t v193 = opcode;	// L300
                              bool v194 = v193 == 8;	// L301
                              if (v194) {	// L302
                                int32_t v195 = dst;	// L303
                                int v196 = v195;	// L304
                                int32_t v197 = reg[v196];	// L305
                                v4.write(v197);	// L306
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/// This is top function.
void top(

) {	// L324
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int32_t array[2] into hls::vector<int32_t, 2>
  hls::stream< hls::vector< int32_t, 2 > > v198;
  #pragma HLS stream variable=v198 depth=2	// L325
  hls::stream< int32_t > v199;
  #pragma HLS stream variable=v199 depth=2	// L326
  hls::stream< int32_t > v200;
  #pragma HLS stream variable=v200 depth=2	// L327
  hls::stream< int32_t > v201;
  #pragma HLS stream variable=v201 depth=2	// L328
  hls::stream< int32_t > v202;
  #pragma HLS stream variable=v202 depth=2	// L329
  vpu_r2_0(v198, v199, v200, v202, v201);	// L330
}


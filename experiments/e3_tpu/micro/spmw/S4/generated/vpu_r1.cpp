
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
void vpu_r1_0(
  hls::stream< hls::vector< int32_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int32_t v4[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v4[_iv0] = _vec[_iv0];
    }
  }	// L28
  int32_t v5 = v1.read();	// L29
  int32_t header;	// L30
  header = v5;	// L31
  int32_t v7 = header;	// L32
  int32_t v8 = v7 & 65535;	// L33
  int32_t plen;	// L34
  plen = v8;	// L35
  int32_t v10 = header;	// L36
  int32_t v11 = v10 >> 16;	// L37
  int32_t v12 = v11 & 65535;	// L38
  int32_t nouts;	// L39
  nouts = v12;	// L40
  int32_t prog[16];	// L41
  for (int v15 = 0; v15 < 16; v15++) {	// L42
    prog[v15] = 0;	// L42
  }
  int32_t v16 = plen;	// L43
  int v17 = v16;	// L44
  for (int v18 = 0; v18 < v17; v18 += 1) {	// L45
    int32_t v19 = v1.read();	// L46
    int32_t word;	// L47
    word = v19;	// L48
    int32_t v21 = word;	// L49
    prog[v18] = v21;	// L50
  }
  int32_t v22 = plen;	// L52
  ap_int<33> v23 = v22;	// L53
  ap_int<33> v24 = 16 - v23;	// L54
  int v25 = v24;	// L55
  for (int v26 = 0; v26 < v25; v26 += 1) {	// L56
    int32_t v27 = v1.read();	// L57
    int32_t spare;	// L58
    spare = v27;	// L59
  }
  int32_t v29 = v4[1];	// L61
  int32_t denom;	// L62
  denom = v29;	// L63
  int32_t rcp;	// L64
  rcp = 0;	// L65
  int32_t v32 = denom;	// L66
  bool v33 = v32 > 0;	// L67
  if (v33) {	// L68
    int32_t v34 = denom;	// L69
    int32_t v35 = 16384 / v34;	// L70
    rcp = v35;	// L71
  }
  int32_t reg[4];	// L73
  for (int v37 = 0; v37 < 4; v37++) {	// L74
    reg[v37] = 0;	// L74
  }
  l_S_r0_2_r0: for (int r0 = 0; r0 < 4; r0++) {	// L75
    reg[r0] = 0;	// L76
  }
  int32_t v39 = nouts;	// L78
  int v40 = v39;	// L79
  for (int v41 = 0; v41 < v40; v41 += 1) {	// L80
    int32_t v42 = plen;	// L81
    int v43 = v42;	// L82
    for (int v44 = 0; v44 < v43; v44 += 1) {	// L83
      int32_t v45 = prog[v44];	// L84
      int32_t word2;	// L85
      word2 = v45;	// L86
      int32_t v47 = word2;	// L87
      int32_t v48 = v47 >> 24;	// L88
      int32_t v49 = v48 & 255;	// L89
      int32_t opcode;	// L90
      opcode = v49;	// L91
      int32_t v51 = word2;	// L92
      int32_t v52 = v51 >> 20;	// L93
      int32_t v53 = v52 & 15;	// L94
      int32_t dst;	// L95
      dst = v53;	// L96
      int32_t v55 = word2;	// L97
      int32_t v56 = v55 >> 16;	// L98
      int32_t v57 = v56 & 15;	// L99
      int32_t src;	// L100
      src = v57;	// L101
      int32_t v59 = word2;	// L102
      int32_t v60 = v59 & 65535;	// L103
      int32_t imm;	// L104
      imm = v60;	// L105
      int32_t v62 = opcode;	// L106
      bool v63 = v62 == 13;	// L107
      if (v63) {	// L108
        int32_t v64 = dst;	// L109
        int v65 = v64;	// L110
        int32_t v66 = reg[v65];	// L111
        int32_t acc;	// L112
        acc = v66;	// L113
        int32_t v68 = imm;	// L114
        int v69 = v68;	// L115
        for (int v70 = 0; v70 < v69; v70 += 1) {	// L116
          int32_t v71 = v2.read();	// L117
          int32_t zz;	// L118
          zz = v71;	// L119
          int32_t v73 = acc;	// L120
          int32_t v74 = zz;	// L121
          ap_int<33> v75 = v73;	// L122
          ap_int<33> v76 = v74;	// L123
          ap_int<33> v77 = v75 + v76;	// L124
          int32_t v78 = v77;	// L125
          acc = v78;	// L126
        }
        int32_t v79 = acc;	// L128
        int32_t v80 = dst;	// L129
        int v81 = v80;	// L130
        reg[v81] = v79;	// L131
      } else {
        int32_t v82 = opcode;	// L133
        bool v83 = v82 == 9;	// L134
        if (v83) {	// L135
          int32_t v84 = v2.read();	// L136
          int32_t z1;	// L137
          z1 = v84;	// L138
          int32_t v86 = dst;	// L139
          int v87 = v86;	// L140
          int32_t v88 = reg[v87];	// L141
          int32_t v89 = z1;	// L142
          ap_int<33> v90 = v88;	// L143
          ap_int<33> v91 = v89;	// L144
          ap_int<33> v92 = v90 + v91;	// L145
          int32_t v93 = v92;	// L146
          reg[v87] = v93;	// L149
        } else {
          int32_t v94 = opcode;	// L151
          bool v95 = v94 == 1;	// L152
          if (v95) {	// L153
            int32_t v96 = v2.read();	// L154
            int32_t z2;	// L155
            z2 = v96;	// L156
            int32_t v98 = z2;	// L157
            int32_t v99 = dst;	// L158
            int v100 = v99;	// L159
            reg[v100] = v98;	// L160
          } else {
            int32_t v101 = opcode;	// L162
            bool v102 = v101 == 2;	// L163
            if (v102) {	// L164
              int32_t v103 = src;	// L165
              int v104 = v103;	// L166
              int32_t v105 = v4[v104];	// L167
              int32_t v106 = dst;	// L168
              int v107 = v106;	// L169
              reg[v107] = v105;	// L170
            } else {
              int32_t v108 = opcode;	// L172
              bool v109 = v108 == 3;	// L173
              if (v109) {	// L174
                int32_t v110 = imm;	// L175
                int32_t v111 = dst;	// L176
                int v112 = v111;	// L177
                reg[v112] = v110;	// L178
              } else {
                int32_t v113 = opcode;	// L180
                bool v114 = v113 == 4;	// L181
                if (v114) {	// L182
                  int32_t v115 = dst;	// L183
                  int v116 = v115;	// L184
                  int32_t v117 = reg[v116];	// L185
                  int32_t v118 = src;	// L186
                  int v119 = v118;	// L187
                  int32_t v120 = reg[v119];	// L188
                  ap_int<33> v121 = v117;	// L189
                  ap_int<33> v122 = v120;	// L190
                  ap_int<33> v123 = v121 + v122;	// L191
                  int32_t v124 = v123;	// L192
                  reg[v116] = v124;	// L195
                } else {
                  int32_t v125 = opcode;	// L197
                  bool v126 = v125 == 10;	// L198
                  if (v126) {	// L199
                    int32_t v127 = dst;	// L200
                    int v128 = v127;	// L201
                    int32_t v129 = reg[v128];	// L202
                    int32_t v130 = src;	// L203
                    int v131 = v130;	// L204
                    int32_t v132 = reg[v131];	// L205
                    ap_int<33> v133 = v129;	// L206
                    ap_int<33> v134 = v132;	// L207
                    ap_int<33> v135 = v133 - v134;	// L208
                    int32_t v136 = v135;	// L209
                    reg[v128] = v136;	// L212
                  } else {
                    int32_t v137 = opcode;	// L214
                    bool v138 = v137 == 5;	// L215
                    if (v138) {	// L216
                      int32_t v139 = dst;	// L217
                      int v140 = v139;	// L218
                      int32_t v141 = reg[v140];	// L219
                      int32_t v142 = src;	// L220
                      int v143 = v142;	// L221
                      int32_t v144 = reg[v143];	// L222
                      int64_t v145 = v141;	// L223
                      int64_t v146 = v144;	// L224
                      int64_t v147 = v145 * v146;	// L225
                      int32_t v148 = v147;	// L226
                      reg[v140] = v148;	// L229
                    } else {
                      int32_t v149 = opcode;	// L231
                      bool v150 = v149 == 6;	// L232
                      if (v150) {	// L233
                        int32_t v151 = src;	// L234
                        int v152 = v151;	// L235
                        int32_t v153 = reg[v152];	// L236
                        int32_t v154 = dst;	// L237
                        int v155 = v154;	// L238
                        int32_t v156 = reg[v155];	// L239
                        bool v157 = v153 > v156;	// L240
                        if (v157) {	// L241
                          int32_t v158 = src;	// L242
                          int v159 = v158;	// L243
                          int32_t v160 = reg[v159];	// L244
                          int32_t v161 = dst;	// L245
                          int v162 = v161;	// L246
                          reg[v162] = v160;	// L247
                        }
                      } else {
                        int32_t v163 = opcode;	// L250
                        bool v164 = v163 == 7;	// L251
                        if (v164) {	// L252
                          int32_t v165 = dst;	// L253
                          int v166 = v165;	// L254
                          int32_t v167 = reg[v166];	// L255
                          int32_t v168 = imm;	// L256
                          int32_t v169 = v167 >> v168;	// L257
                          reg[v166] = v169;	// L260
                        } else {
                          int32_t v170 = opcode;	// L262
                          bool v171 = v170 == 11;	// L263
                          if (v171) {	// L264
                            int32_t v172 = dst;	// L265
                            int v173 = v172;	// L266
                            int32_t v174 = reg[v173];	// L267
                            int32_t e;	// L268
                            e = v174;	// L269
                            int32_t v176 = e;	// L270
                            bool v177 = v176 < 0;	// L271
                            if (v177) {	// L272
                              e = 0;	// L273
                            }
                            int32_t v178 = e;	// L275
                            bool v179 = v178 > 30;	// L276
                            if (v179) {	// L277
                              e = 30;	// L278
                            }
                            int32_t v180 = e;	// L280
                            int32_t v181 = 1 << v180;	// L281
                            int32_t v182 = dst;	// L282
                            int v183 = v182;	// L283
                            reg[v183] = v181;	// L284
                          } else {
                            int32_t v184 = opcode;	// L286
                            bool v185 = v184 == 12;	// L287
                            if (v185) {	// L288
                              int32_t v186 = rcp;	// L289
                              int32_t v187 = dst;	// L290
                              int v188 = v187;	// L291
                              reg[v188] = v186;	// L292
                            } else {
                              int32_t v189 = opcode;	// L294
                              bool v190 = v189 == 8;	// L295
                              if (v190) {	// L296
                                int32_t v191 = dst;	// L297
                                int v192 = v191;	// L298
                                int32_t v193 = reg[v192];	// L299
                                v3.write(v193);	// L300
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

) {	// L318
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int32_t array[2] into hls::vector<int32_t, 2>
  hls::stream< hls::vector< int32_t, 2 > > v194;
  #pragma HLS stream variable=v194 depth=2	// L319
  hls::stream< int32_t > v195;
  #pragma HLS stream variable=v195 depth=2	// L320
  hls::stream< int32_t > v196;
  #pragma HLS stream variable=v196 depth=2	// L321
  hls::stream< int32_t > v197;
  #pragma HLS stream variable=v197 depth=2	// L322
  vpu_r1_0(v194, v195, v197, v196);	// L323
}


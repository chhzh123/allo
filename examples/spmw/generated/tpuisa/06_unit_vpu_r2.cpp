
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
  }	// L23
  int32_t prog[16];	// L24
  for (int v7 = 0; v7 < 16; v7++) {	// L25
    prog[v7] = 0;	// L25
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 16; pc++) {	// L26
    int32_t v9 = v1.read();	// L27
    int32_t word;	// L28
    word = v9;	// L29
    int32_t v11 = word;	// L30
    prog[pc] = v11;	// L31
    int32_t v12 = word;	// L32
    v2.write(v12);	// L33
  }
  int32_t reg[4];	// L35
  for (int v14 = 0; v14 < 4; v14++) {	// L36
    reg[v14] = 0;	// L36
  }
  l_S_m_1_m: for (int m = 0; m < 4; m++) {	// L37
    l_S_pc2_1_pc2: for (int pc2 = 0; pc2 < 16; pc2++) {	// L38
      int32_t v17 = prog[pc2];	// L39
      int32_t word2;	// L40
      word2 = v17;	// L41
      int32_t v19 = word2;	// L42
      int32_t v20 = v19 >> 24;	// L43
      int32_t v21 = v20 & 255;	// L44
      int32_t opcode;	// L45
      opcode = v21;	// L46
      int32_t v23 = word2;	// L47
      int32_t v24 = v23 >> 20;	// L48
      int32_t v25 = v24 & 15;	// L49
      int32_t dst;	// L50
      dst = v25;	// L51
      int32_t v27 = word2;	// L52
      int32_t v28 = v27 >> 16;	// L53
      int32_t v29 = v28 & 15;	// L54
      int32_t src;	// L55
      src = v29;	// L56
      int32_t v31 = word2;	// L57
      int32_t v32 = v31 & 65535;	// L58
      int32_t imm;	// L59
      imm = v32;	// L60
      int32_t v34 = opcode;	// L61
      bool v35 = v34 == 9;	// L62
      if (v35) {	// L63
        int32_t v36 = v3.read();	// L64
        int32_t zz;	// L65
        zz = v36;	// L66
        int32_t v38 = dst;	// L67
        int v39 = v38;	// L68
        int32_t v40 = reg[v39];	// L69
        int32_t v41 = zz;	// L70
        ap_int<33> v42 = v40;	// L71
        ap_int<33> v43 = v41;	// L72
        ap_int<33> v44 = v42 + v43;	// L73
        int32_t v45 = v44;	// L74
        reg[v39] = v45;	// L77
      } else {
        int32_t v46 = opcode;	// L79
        bool v47 = v46 == 1;	// L80
        if (v47) {	// L81
          int32_t v48 = v3.read();	// L82
          int32_t z2;	// L83
          z2 = v48;	// L84
          int32_t v50 = z2;	// L85
          int32_t v51 = dst;	// L86
          int v52 = v51;	// L87
          reg[v52] = v50;	// L88
        } else {
          int32_t v53 = opcode;	// L90
          bool v54 = v53 == 2;	// L91
          if (v54) {	// L92
            int32_t v55 = src;	// L93
            int v56 = v55;	// L94
            int32_t v57 = v5[v56];	// L95
            int32_t v58 = dst;	// L96
            int v59 = v58;	// L97
            reg[v59] = v57;	// L98
          } else {
            int32_t v60 = opcode;	// L100
            bool v61 = v60 == 3;	// L101
            if (v61) {	// L102
              int32_t v62 = imm;	// L103
              int32_t v63 = dst;	// L104
              int v64 = v63;	// L105
              reg[v64] = v62;	// L106
            } else {
              int32_t v65 = opcode;	// L108
              bool v66 = v65 == 4;	// L109
              if (v66) {	// L110
                int32_t v67 = dst;	// L111
                int v68 = v67;	// L112
                int32_t v69 = reg[v68];	// L113
                int32_t v70 = src;	// L114
                int v71 = v70;	// L115
                int32_t v72 = reg[v71];	// L116
                ap_int<33> v73 = v69;	// L117
                ap_int<33> v74 = v72;	// L118
                ap_int<33> v75 = v73 + v74;	// L119
                int32_t v76 = v75;	// L120
                reg[v68] = v76;	// L123
              } else {
                int32_t v77 = opcode;	// L125
                bool v78 = v77 == 5;	// L126
                if (v78) {	// L127
                  int32_t v79 = dst;	// L128
                  int v80 = v79;	// L129
                  int32_t v81 = reg[v80];	// L130
                  int32_t v82 = src;	// L131
                  int v83 = v82;	// L132
                  int32_t v84 = reg[v83];	// L133
                  int64_t v85 = v81;	// L134
                  int64_t v86 = v84;	// L135
                  int64_t v87 = v85 * v86;	// L136
                  int32_t v88 = v87;	// L137
                  reg[v80] = v88;	// L140
                } else {
                  int32_t v89 = opcode;	// L142
                  bool v90 = v89 == 6;	// L143
                  if (v90) {	// L144
                    int32_t v91 = src;	// L145
                    int v92 = v91;	// L146
                    int32_t v93 = reg[v92];	// L147
                    int32_t v94 = dst;	// L148
                    int v95 = v94;	// L149
                    int32_t v96 = reg[v95];	// L150
                    bool v97 = v93 > v96;	// L151
                    if (v97) {	// L152
                      int32_t v98 = src;	// L153
                      int v99 = v98;	// L154
                      int32_t v100 = reg[v99];	// L155
                      int32_t v101 = dst;	// L156
                      int v102 = v101;	// L157
                      reg[v102] = v100;	// L158
                    }
                  } else {
                    int32_t v103 = opcode;	// L161
                    bool v104 = v103 == 7;	// L162
                    if (v104) {	// L163
                      int32_t v105 = dst;	// L164
                      int v106 = v105;	// L165
                      int32_t v107 = reg[v106];	// L166
                      int32_t v108 = imm;	// L167
                      int32_t v109 = v107 >> v108;	// L168
                      reg[v106] = v109;	// L171
                    } else {
                      int32_t v110 = opcode;	// L173
                      bool v111 = v110 == 10;	// L174
                      if (v111) {	// L175
                        int32_t v112 = dst;	// L176
                        int v113 = v112;	// L177
                        int32_t v114 = reg[v113];	// L178
                        int32_t v115 = src;	// L179
                        int v116 = v115;	// L180
                        int32_t v117 = reg[v116];	// L181
                        ap_int<33> v118 = v114;	// L182
                        ap_int<33> v119 = v117;	// L183
                        ap_int<33> v120 = v118 - v119;	// L184
                        int32_t v121 = v120;	// L185
                        reg[v113] = v121;	// L188
                      } else {
                        int32_t v122 = opcode;	// L190
                        bool v123 = v122 == 11;	// L191
                        if (v123) {	// L192
                          int32_t v124 = dst;	// L193
                          int v125 = v124;	// L194
                          int32_t v126 = reg[v125];	// L195
                          int32_t e;	// L196
                          e = v126;	// L197
                          int32_t v128 = e;	// L198
                          bool v129 = v128 < 0;	// L199
                          if (v129) {	// L200
                            e = 0;	// L201
                          }
                          int32_t v130 = e;	// L203
                          bool v131 = v130 > 30;	// L204
                          if (v131) {	// L205
                            e = 30;	// L206
                          }
                          int32_t v132 = e;	// L208
                          int32_t v133 = 1 << v132;	// L209
                          int32_t v134 = dst;	// L210
                          int v135 = v134;	// L211
                          reg[v135] = v133;	// L212
                        } else {
                          int32_t v136 = opcode;	// L214
                          bool v137 = v136 == 12;	// L215
                          if (v137) {	// L216
                            int32_t v138 = dst;	// L217
                            int v139 = v138;	// L218
                            int32_t v140 = reg[v139];	// L219
                            int32_t d;	// L220
                            d = v140;	// L221
                            int32_t v142 = d;	// L222
                            bool v143 = v142 > 0;	// L223
                            if (v143) {	// L224
                              int32_t v144 = imm;	// L225
                              int32_t v145 = 1 << v144;	// L226
                              int32_t v146 = d;	// L227
                              int32_t v147 = v145 / v146;	// L228
                              int32_t v148 = dst;	// L229
                              int v149 = v148;	// L230
                              reg[v149] = v147;	// L231
                            } else {
                              int32_t v150 = dst;	// L233
                              int v151 = v150;	// L234
                              reg[v151] = 0;	// L235
                            }
                          } else {
                            int32_t v152 = opcode;	// L238
                            bool v153 = v152 == 8;	// L239
                            if (v153) {	// L240
                              int32_t v154 = dst;	// L241
                              int v155 = v154;	// L242
                              int32_t v156 = reg[v155];	// L243
                              v4.write(v156);	// L244
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

) {	// L261
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int32_t array[2] into hls::vector<int32_t, 2>
  hls::stream< hls::vector< int32_t, 2 > > v157;
  #pragma HLS stream variable=v157 depth=2	// L262
  hls::stream< int32_t > v158;
  #pragma HLS stream variable=v158 depth=2	// L263
  hls::stream< int32_t > v159;
  #pragma HLS stream variable=v159 depth=2	// L264
  hls::stream< int32_t > v160;
  #pragma HLS stream variable=v160 depth=2	// L265
  hls::stream< int32_t > v161;
  #pragma HLS stream variable=v161 depth=2	// L266
  vpu_r2_0(v157, v158, v159, v161, v160);	// L267
}



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
/// This is top function.
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
  }	// L3
  int32_t v6 = v1.read();	// L4
  int32_t header;	// L5
  header = v6;	// L6
  int32_t v8 = header;	// L7
  v2.write(v8);	// L8
  int32_t v9 = header;	// L9
  int32_t v10 = v9 & 65535;	// L12
  int32_t plen;	// L13
  plen = v10;	// L14
  int32_t v12 = header;	// L15
  int32_t v13 = v12 >> 16;	// L18
  int32_t v14 = v13 & 65535;	// L21
  int32_t nouts;	// L22
  nouts = v14;	// L23
  int32_t prog[16];	// L24
  for (int v17 = 0; v17 < 16; v17++) {	// L26
    prog[v17] = 0;	// L26
  }
  int32_t v18 = plen;	// L27
  int v19 = v18;	// L31
  for (int v20 = 0; v20 < v19; v20 += 1) {	// L35
    int32_t v21 = v1.read();	// L36
    int32_t word;	// L37
    word = v21;	// L38
    int32_t v23 = word;	// L39
    prog[v20] = v23;	// L40
    int32_t v24 = word;	// L41
    v2.write(v24);	// L42
  }
  int32_t v25 = plen;	// L44
  ap_int<33> v26 = v25;	// L48
  ap_int<33> v27 = 16 - v26;	// L49
  int v28 = v27;	// L53
  for (int v29 = 0; v29 < v28; v29 += 1) {	// L57
    int32_t v30 = v1.read();	// L58
    int32_t spare;	// L59
    spare = v30;	// L60
    int32_t v32 = spare;	// L61
    v2.write(v32);	// L62
  }
  int32_t v33 = v5[1];	// L64
  int32_t denom;	// L65
  denom = v33;	// L66
  int32_t rcp;	// L69
  rcp = 0;	// L70
  int32_t v36 = denom;	// L71
  bool v37 = v36 > 0;	// L74
  if (v37) {	// L75
    int32_t v38 = denom;	// L81
    int32_t v39 = 16384 / v38;	// L82
    rcp = v39;	// L83
  }
  int32_t r0;	// L87
  r0 = 0;	// L88
  int32_t r1;	// L91
  r1 = 0;	// L92
  int32_t r2;	// L95
  r2 = 0;	// L96
  int32_t r3;	// L99
  r3 = 0;	// L100
  int32_t pc2;	// L103
  pc2 = 0;	// L104
  int32_t v45 = nouts;	// L105
  int32_t v46 = plen;	// L106
  int64_t v47 = v45;	// L107
  int64_t v48 = v46;	// L108
  int64_t v49 = v47 * v48;	// L109
  int v50 = v49;	// L113
  for (int v51 = 0; v51 < v50; v51 += 1) {	// L117
    int32_t v52 = pc2;	// L118
    int v53 = v52;	// L119
    int32_t v54 = prog[v53];	// L120
    int32_t word2;	// L121
    word2 = v54;	// L122
    int32_t v56 = word2;	// L123
    int32_t v57 = v56 >> 24;	// L126
    int32_t v58 = v57 & 255;	// L129
    int32_t opcode;	// L130
    opcode = v58;	// L131
    int32_t v60 = word2;	// L132
    int32_t v61 = v60 >> 20;	// L135
    int32_t v62 = v61 & 15;	// L138
    int32_t dst;	// L139
    dst = v62;	// L140
    int32_t v64 = word2;	// L141
    int32_t v65 = v64 >> 16;	// L144
    int32_t v66 = v65 & 15;	// L147
    int32_t src;	// L148
    src = v66;	// L149
    int32_t v68 = word2;	// L150
    int32_t v69 = v68 & 65535;	// L153
    int32_t imm;	// L154
    imm = v69;	// L155
    int32_t v71 = r0;	// L156
    int32_t d;	// L157
    d = v71;	// L158
    int32_t v73 = dst;	// L159
    bool v74 = v73 == 1;	// L162
    if (v74) {	// L163
      int32_t v75 = r1;	// L164
      d = v75;	// L165
    } else {
      int32_t v76 = dst;	// L167
      bool v77 = v76 == 2;	// L170
      if (v77) {	// L171
        int32_t v78 = r2;	// L172
        d = v78;	// L173
      } else {
        int32_t v79 = dst;	// L175
        bool v80 = v79 == 3;	// L178
        if (v80) {	// L179
          int32_t v81 = r3;	// L180
          d = v81;	// L181
        }
      }
    }
    int32_t v82 = r0;	// L185
    int32_t a;	// L186
    a = v82;	// L187
    int32_t v84 = src;	// L188
    bool v85 = v84 == 1;	// L191
    if (v85) {	// L192
      int32_t v86 = r1;	// L193
      a = v86;	// L194
    } else {
      int32_t v87 = src;	// L196
      bool v88 = v87 == 2;	// L199
      if (v88) {	// L200
        int32_t v89 = r2;	// L201
        a = v89;	// L202
      } else {
        int32_t v90 = src;	// L204
        bool v91 = v90 == 3;	// L207
        if (v91) {	// L208
          int32_t v92 = r3;	// L209
          a = v92;	// L210
        }
      }
    }
    int32_t wr;	// L216
    wr = 1;	// L217
    int32_t v94 = opcode;	// L218
    bool v95 = v94 == 9;	// L221
    if (v95) {	// L222
      int32_t v96 = v4.read();	// L223
      int32_t zz;	// L224
      zz = v96;	// L225
      int32_t v98 = d;	// L226
      int32_t v99 = zz;	// L227
      ap_int<33> v100 = v98;	// L228
      ap_int<33> v101 = v99;	// L229
      ap_int<33> v102 = v100 + v101;	// L230
      int32_t v103 = v102;	// L231
      d = v103;	// L232
    } else {
      int32_t v104 = opcode;	// L234
      bool v105 = v104 == 1;	// L237
      if (v105) {	// L238
        int32_t v106 = v4.read();	// L239
        int32_t z2;	// L240
        z2 = v106;	// L241
        int32_t v108 = z2;	// L242
        d = v108;	// L243
      } else {
        int32_t v109 = opcode;	// L245
        bool v110 = v109 == 2;	// L248
        if (v110) {	// L249
          int32_t v111 = src;	// L250
          int v112 = v111;	// L251
          int32_t v113 = v5[v112];	// L252
          d = v113;	// L253
        } else {
          int32_t v114 = opcode;	// L255
          bool v115 = v114 == 3;	// L258
          if (v115) {	// L259
            int32_t v116 = imm;	// L260
            d = v116;	// L261
          } else {
            int32_t v117 = opcode;	// L263
            bool v118 = v117 == 4;	// L266
            if (v118) {	// L267
              int32_t v119 = d;	// L268
              int32_t v120 = a;	// L269
              ap_int<33> v121 = v119;	// L270
              ap_int<33> v122 = v120;	// L271
              ap_int<33> v123 = v121 + v122;	// L272
              int32_t v124 = v123;	// L273
              d = v124;	// L274
            } else {
              int32_t v125 = opcode;	// L276
              bool v126 = v125 == 5;	// L279
              if (v126) {	// L280
                int32_t v127 = d;	// L281
                int32_t v128 = a;	// L282
                int64_t v129 = v127;	// L283
                int64_t v130 = v128;	// L284
                int64_t v131 = v129 * v130;	// L285
                int32_t v132 = v131;	// L286
                d = v132;	// L287
              } else {
                int32_t v133 = opcode;	// L289
                bool v134 = v133 == 6;	// L292
                if (v134) {	// L293
                  int32_t v135 = a;	// L294
                  int32_t v136 = d;	// L295
                  bool v137 = v135 > v136;	// L296
                  if (v137) {	// L297
                    int32_t v138 = a;	// L298
                    d = v138;	// L299
                  }
                } else {
                  int32_t v139 = opcode;	// L302
                  bool v140 = v139 == 7;	// L305
                  if (v140) {	// L306
                    int32_t v141 = d;	// L307
                    int32_t v142 = imm;	// L308
                    int32_t v143 = v141 >> v142;	// L309
                    d = v143;	// L310
                  } else {
                    int32_t v144 = opcode;	// L312
                    bool v145 = v144 == 10;	// L315
                    if (v145) {	// L316
                      int32_t v146 = d;	// L317
                      int32_t v147 = a;	// L318
                      ap_int<33> v148 = v146;	// L319
                      ap_int<33> v149 = v147;	// L320
                      ap_int<33> v150 = v148 - v149;	// L321
                      int32_t v151 = v150;	// L322
                      d = v151;	// L323
                    } else {
                      int32_t v152 = opcode;	// L325
                      bool v153 = v152 == 11;	// L328
                      if (v153) {	// L329
                        int32_t v154 = d;	// L330
                        int32_t e;	// L331
                        e = v154;	// L332
                        int32_t v156 = e;	// L333
                        bool v157 = v156 < 0;	// L336
                        if (v157) {	// L337
                          e = 0;	// L340
                        }
                        int32_t v158 = e;	// L342
                        bool v159 = v158 > 30;	// L345
                        if (v159) {	// L346
                          e = 30;	// L349
                        }
                        int32_t v160 = e;	// L351
                        int32_t v161 = 1 << v160;	// L354
                        d = v161;	// L355
                      } else {
                        int32_t v162 = opcode;	// L357
                        bool v163 = v162 == 12;	// L360
                        if (v163) {	// L361
                          int32_t v164 = rcp;	// L362
                          d = v164;	// L363
                        } else {
                          int32_t v165 = opcode;	// L365
                          bool v166 = v165 == 8;	// L368
                          if (v166) {	// L369
                            int32_t v167 = d;	// L370
                            v3.write(v167);	// L371
                            wr = 0;	// L374
                          } else {
                            wr = 0;	// L378
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
    int32_t v168 = wr;	// L391
    bool v169 = v168 == 1;	// L394
    if (v169) {	// L395
      int32_t v170 = dst;	// L396
      bool v171 = v170 == 0;	// L399
      if (v171) {	// L400
        int32_t v172 = d;	// L401
        r0 = v172;	// L402
      } else {
        int32_t v173 = dst;	// L404
        bool v174 = v173 == 1;	// L407
        if (v174) {	// L408
          int32_t v175 = d;	// L409
          r1 = v175;	// L410
        } else {
          int32_t v176 = dst;	// L412
          bool v177 = v176 == 2;	// L415
          if (v177) {	// L416
            int32_t v178 = d;	// L417
            r2 = v178;	// L418
          } else {
            int32_t v179 = d;	// L420
            r3 = v179;	// L421
          }
        }
      }
    }
    int32_t v180 = pc2;	// L426
    ap_int<33> v181 = v180;	// L427
    ap_int<33> v182 = v181 + 1;	// L431
    int32_t v183 = v182;	// L432
    pc2 = v183;	// L433
    int32_t v184 = pc2;	// L434
    int32_t v185 = plen;	// L435
    bool v186 = v184 == v185;	// L436
    if (v186) {	// L437
      pc2 = 0;	// L440
    }
  }
}


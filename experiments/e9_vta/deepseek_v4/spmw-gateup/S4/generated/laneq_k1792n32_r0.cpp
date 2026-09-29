
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
void laneq_k1792n32_r0_0(
  hls::stream< hls::vector< int32_t, 1 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t v3[1];
  {
    hls::vector< int32_t, 1 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 1; ++_iv0) {
      v3[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v4 = v3[0];	// L4
  int32_t v5 = v4 & 31;	// L6
  int32_t sh;	// L7
  sh = v5;	// L8
  int32_t r0;	// L10
  r0 = 0;	// L11
  int32_t r1;	// L12
  r1 = 0;	// L13
  int32_t r2;	// L14
  r2 = 0;	// L15
  int32_t r3;	// L16
  r3 = 0;	// L17
  int32_t r4;	// L18
  r4 = 0;	// L19
  int32_t r5;	// L20
  r5 = 0;	// L21
  int32_t r6;	// L22
  r6 = 0;	// L23
  int32_t r7;	// L24
  r7 = 0;	// L25
  int32_t r8;	// L26
  r8 = 0;	// L27
  int32_t r9;	// L28
  r9 = 0;	// L29
  int32_t r10;	// L30
  r10 = 0;	// L31
  int32_t r11;	// L32
  r11 = 0;	// L33
  int32_t r12;	// L34
  r12 = 0;	// L35
  int32_t r13;	// L36
  r13 = 0;	// L37
  int32_t r14;	// L38
  r14 = 0;	// L39
  int32_t r15;	// L40
  r15 = 0;	// L41
  int32_t r16;	// L42
  r16 = 0;	// L43
  int32_t r17;	// L44
  r17 = 0;	// L45
  int32_t r18;	// L46
  r18 = 0;	// L47
  int32_t r19;	// L48
  r19 = 0;	// L49
  int32_t r20;	// L50
  r20 = 0;	// L51
  int32_t r21;	// L52
  r21 = 0;	// L53
  int32_t r22;	// L54
  r22 = 0;	// L55
  int32_t r23;	// L56
  r23 = 0;	// L57
  int32_t r24;	// L58
  r24 = 0;	// L59
  int32_t r25;	// L60
  r25 = 0;	// L61
  int32_t r26;	// L62
  r26 = 0;	// L63
  int32_t r27;	// L64
  r27 = 0;	// L65
  int32_t r28;	// L66
  r28 = 0;	// L67
  int32_t r29;	// L68
  r29 = 0;	// L69
  int32_t r30;	// L70
  r30 = 0;	// L71
  int32_t r31;	// L72
  r31 = 0;	// L73
  int32_t r32;	// L74
  r32 = 0;	// L75
  int32_t r33;	// L76
  r33 = 0;	// L77
  int32_t r34;	// L78
  r34 = 0;	// L79
  int32_t r35;	// L80
  r35 = 0;	// L81
  int32_t r36;	// L82
  r36 = 0;	// L83
  int32_t r37;	// L84
  r37 = 0;	// L85
  int32_t r38;	// L86
  r38 = 0;	// L87
  int32_t r39;	// L88
  r39 = 0;	// L89
  int32_t r40;	// L90
  r40 = 0;	// L91
  int32_t r41;	// L92
  r41 = 0;	// L93
  int32_t r42;	// L94
  r42 = 0;	// L95
  int32_t r43;	// L96
  r43 = 0;	// L97
  int32_t r44;	// L98
  r44 = 0;	// L99
  int32_t r45;	// L100
  r45 = 0;	// L101
  int32_t r46;	// L102
  r46 = 0;	// L103
  int32_t r47;	// L104
  r47 = 0;	// L105
  int32_t r48;	// L106
  r48 = 0;	// L107
  int32_t r49;	// L108
  r49 = 0;	// L109
  int32_t r50;	// L110
  r50 = 0;	// L111
  int32_t r51;	// L112
  r51 = 0;	// L113
  int32_t r52;	// L114
  r52 = 0;	// L115
  int32_t r53;	// L116
  r53 = 0;	// L117
  int32_t r54;	// L118
  r54 = 0;	// L119
  int32_t r55;	// L120
  r55 = 0;	// L121
  int32_t r56;	// L122
  r56 = 0;	// L123
  int32_t r57;	// L124
  r57 = 0;	// L125
  int32_t r58;	// L126
  r58 = 0;	// L127
  int32_t r59;	// L128
  r59 = 0;	// L129
  int32_t r60;	// L130
  r60 = 0;	// L131
  int32_t r61;	// L132
  r61 = 0;	// L133
  int32_t r62;	// L134
  r62 = 0;	// L135
  int32_t r63;	// L136
  r63 = 0;	// L137
  int32_t kb;	// L138
  kb = 0;	// L139
  l_S_s_0_s: for (int s = 0; s < 3670016; s++) {	// L140
  #pragma HLS pipeline II=1
    int32_t v73 = v2.read();	// L141
    int32_t z;	// L142
    z = v73;	// L143
    int32_t v75 = z;	// L144
    int32_t v;	// L145
    v = v75;	// L146
    int32_t v77 = kb;	// L147
    bool v78 = v77 != 0;	// L148
    if (v78) {	// L149
      int32_t v79 = r0;	// L150
      int32_t v80 = z;	// L151
      ap_int<33> v81 = v79;	// L152
      ap_int<33> v82 = v80;	// L153
      ap_int<33> v83 = v81 + v82;	// L154
      int32_t v84 = v83;	// L155
      v = v84;	// L156
    }
    int32_t v85 = r1;	// L158
    r0 = v85;	// L159
    int32_t v86 = r2;	// L160
    r1 = v86;	// L161
    int32_t v87 = r3;	// L162
    r2 = v87;	// L163
    int32_t v88 = r4;	// L164
    r3 = v88;	// L165
    int32_t v89 = r5;	// L166
    r4 = v89;	// L167
    int32_t v90 = r6;	// L168
    r5 = v90;	// L169
    int32_t v91 = r7;	// L170
    r6 = v91;	// L171
    int32_t v92 = r8;	// L172
    r7 = v92;	// L173
    int32_t v93 = r9;	// L174
    r8 = v93;	// L175
    int32_t v94 = r10;	// L176
    r9 = v94;	// L177
    int32_t v95 = r11;	// L178
    r10 = v95;	// L179
    int32_t v96 = r12;	// L180
    r11 = v96;	// L181
    int32_t v97 = r13;	// L182
    r12 = v97;	// L183
    int32_t v98 = r14;	// L184
    r13 = v98;	// L185
    int32_t v99 = r15;	// L186
    r14 = v99;	// L187
    int32_t v100 = r16;	// L188
    r15 = v100;	// L189
    int32_t v101 = r17;	// L190
    r16 = v101;	// L191
    int32_t v102 = r18;	// L192
    r17 = v102;	// L193
    int32_t v103 = r19;	// L194
    r18 = v103;	// L195
    int32_t v104 = r20;	// L196
    r19 = v104;	// L197
    int32_t v105 = r21;	// L198
    r20 = v105;	// L199
    int32_t v106 = r22;	// L200
    r21 = v106;	// L201
    int32_t v107 = r23;	// L202
    r22 = v107;	// L203
    int32_t v108 = r24;	// L204
    r23 = v108;	// L205
    int32_t v109 = r25;	// L206
    r24 = v109;	// L207
    int32_t v110 = r26;	// L208
    r25 = v110;	// L209
    int32_t v111 = r27;	// L210
    r26 = v111;	// L211
    int32_t v112 = r28;	// L212
    r27 = v112;	// L213
    int32_t v113 = r29;	// L214
    r28 = v113;	// L215
    int32_t v114 = r30;	// L216
    r29 = v114;	// L217
    int32_t v115 = r31;	// L218
    r30 = v115;	// L219
    int32_t v116 = r32;	// L220
    r31 = v116;	// L221
    int32_t v117 = r33;	// L222
    r32 = v117;	// L223
    int32_t v118 = r34;	// L224
    r33 = v118;	// L225
    int32_t v119 = r35;	// L226
    r34 = v119;	// L227
    int32_t v120 = r36;	// L228
    r35 = v120;	// L229
    int32_t v121 = r37;	// L230
    r36 = v121;	// L231
    int32_t v122 = r38;	// L232
    r37 = v122;	// L233
    int32_t v123 = r39;	// L234
    r38 = v123;	// L235
    int32_t v124 = r40;	// L236
    r39 = v124;	// L237
    int32_t v125 = r41;	// L238
    r40 = v125;	// L239
    int32_t v126 = r42;	// L240
    r41 = v126;	// L241
    int32_t v127 = r43;	// L242
    r42 = v127;	// L243
    int32_t v128 = r44;	// L244
    r43 = v128;	// L245
    int32_t v129 = r45;	// L246
    r44 = v129;	// L247
    int32_t v130 = r46;	// L248
    r45 = v130;	// L249
    int32_t v131 = r47;	// L250
    r46 = v131;	// L251
    int32_t v132 = r48;	// L252
    r47 = v132;	// L253
    int32_t v133 = r49;	// L254
    r48 = v133;	// L255
    int32_t v134 = r50;	// L256
    r49 = v134;	// L257
    int32_t v135 = r51;	// L258
    r50 = v135;	// L259
    int32_t v136 = r52;	// L260
    r51 = v136;	// L261
    int32_t v137 = r53;	// L262
    r52 = v137;	// L263
    int32_t v138 = r54;	// L264
    r53 = v138;	// L265
    int32_t v139 = r55;	// L266
    r54 = v139;	// L267
    int32_t v140 = r56;	// L268
    r55 = v140;	// L269
    int32_t v141 = r57;	// L270
    r56 = v141;	// L271
    int32_t v142 = r58;	// L272
    r57 = v142;	// L273
    int32_t v143 = r59;	// L274
    r58 = v143;	// L275
    int32_t v144 = r60;	// L276
    r59 = v144;	// L277
    int32_t v145 = r61;	// L278
    r60 = v145;	// L279
    int32_t v146 = r62;	// L280
    r61 = v146;	// L281
    int32_t v147 = r63;	// L282
    r62 = v147;	// L283
    int32_t v148 = v;	// L284
    r63 = v148;	// L285
    int32_t v149 = kb;	// L286
    ap_int<33> v150 = v149;	// L292
    bool v151 = v150 == 1791;	// L293
    if (v151) {	// L294
      int32_t v152 = v;	// L295
      int32_t v153 = sh;	// L296
      int32_t v154 = v152 >> v153;	// L297
      v = v154;	// L298
      int32_t v155 = v;	// L299
      bool v156 = v155 < -128;	// L302
      if (v156) {	// L303
        v = -128;	// L304
      }
      int32_t v157 = v;	// L306
      bool v158 = v157 > 127;	// L308
      if (v158) {	// L309
        v = 127;	// L310
      }
      int32_t v159 = v;	// L312
      v1.write(v159);	// L313
    }
    int32_t v160 = s;	// L315
    int32_t v161 = v160 & 63;	// L317
    bool v162 = v161 == 63;	// L318
    if (v162) {	// L319
      int32_t v163 = kb;	// L320
      ap_int<33> v164 = v163;	// L321
      ap_int<33> v165 = v164 + 1;	// L322
      int32_t v166 = v165;	// L323
      kb = v166;	// L324
      int32_t v167 = kb;	// L325
      bool v168 = v167 == 1792;	// L326
      if (v168) {	// L327
        kb = 0;	// L328
      }
    }
  }
}


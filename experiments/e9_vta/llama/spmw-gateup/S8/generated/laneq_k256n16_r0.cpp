
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
void laneq_k256n16_r0_0(
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
  l_S_s_0_s: for (int s = 0; s < 262144; s++) {	// L138
  #pragma HLS pipeline II=1
    int v72 = s >> 6;	// L141
    ap_int<33> v73 = v72;	// L147
    ap_int<33> v74 = v73 & 255;	// L148
    int32_t v75 = v74;	// L149
    int32_t kb;	// L150
    kb = v75;	// L151
    int32_t v77 = v2.read();	// L152
    int32_t z;	// L153
    z = v77;	// L154
    int32_t v79 = z;	// L155
    int32_t v;	// L156
    v = v79;	// L157
    int32_t v81 = kb;	// L158
    bool v82 = v81 != 0;	// L159
    if (v82) {	// L160
      int32_t v83 = r0;	// L161
      int32_t v84 = z;	// L162
      ap_int<33> v85 = v83;	// L163
      ap_int<33> v86 = v84;	// L164
      ap_int<33> v87 = v85 + v86;	// L165
      int32_t v88 = v87;	// L166
      v = v88;	// L167
    }
    int32_t v89 = r1;	// L169
    r0 = v89;	// L170
    int32_t v90 = r2;	// L171
    r1 = v90;	// L172
    int32_t v91 = r3;	// L173
    r2 = v91;	// L174
    int32_t v92 = r4;	// L175
    r3 = v92;	// L176
    int32_t v93 = r5;	// L177
    r4 = v93;	// L178
    int32_t v94 = r6;	// L179
    r5 = v94;	// L180
    int32_t v95 = r7;	// L181
    r6 = v95;	// L182
    int32_t v96 = r8;	// L183
    r7 = v96;	// L184
    int32_t v97 = r9;	// L185
    r8 = v97;	// L186
    int32_t v98 = r10;	// L187
    r9 = v98;	// L188
    int32_t v99 = r11;	// L189
    r10 = v99;	// L190
    int32_t v100 = r12;	// L191
    r11 = v100;	// L192
    int32_t v101 = r13;	// L193
    r12 = v101;	// L194
    int32_t v102 = r14;	// L195
    r13 = v102;	// L196
    int32_t v103 = r15;	// L197
    r14 = v103;	// L198
    int32_t v104 = r16;	// L199
    r15 = v104;	// L200
    int32_t v105 = r17;	// L201
    r16 = v105;	// L202
    int32_t v106 = r18;	// L203
    r17 = v106;	// L204
    int32_t v107 = r19;	// L205
    r18 = v107;	// L206
    int32_t v108 = r20;	// L207
    r19 = v108;	// L208
    int32_t v109 = r21;	// L209
    r20 = v109;	// L210
    int32_t v110 = r22;	// L211
    r21 = v110;	// L212
    int32_t v111 = r23;	// L213
    r22 = v111;	// L214
    int32_t v112 = r24;	// L215
    r23 = v112;	// L216
    int32_t v113 = r25;	// L217
    r24 = v113;	// L218
    int32_t v114 = r26;	// L219
    r25 = v114;	// L220
    int32_t v115 = r27;	// L221
    r26 = v115;	// L222
    int32_t v116 = r28;	// L223
    r27 = v116;	// L224
    int32_t v117 = r29;	// L225
    r28 = v117;	// L226
    int32_t v118 = r30;	// L227
    r29 = v118;	// L228
    int32_t v119 = r31;	// L229
    r30 = v119;	// L230
    int32_t v120 = r32;	// L231
    r31 = v120;	// L232
    int32_t v121 = r33;	// L233
    r32 = v121;	// L234
    int32_t v122 = r34;	// L235
    r33 = v122;	// L236
    int32_t v123 = r35;	// L237
    r34 = v123;	// L238
    int32_t v124 = r36;	// L239
    r35 = v124;	// L240
    int32_t v125 = r37;	// L241
    r36 = v125;	// L242
    int32_t v126 = r38;	// L243
    r37 = v126;	// L244
    int32_t v127 = r39;	// L245
    r38 = v127;	// L246
    int32_t v128 = r40;	// L247
    r39 = v128;	// L248
    int32_t v129 = r41;	// L249
    r40 = v129;	// L250
    int32_t v130 = r42;	// L251
    r41 = v130;	// L252
    int32_t v131 = r43;	// L253
    r42 = v131;	// L254
    int32_t v132 = r44;	// L255
    r43 = v132;	// L256
    int32_t v133 = r45;	// L257
    r44 = v133;	// L258
    int32_t v134 = r46;	// L259
    r45 = v134;	// L260
    int32_t v135 = r47;	// L261
    r46 = v135;	// L262
    int32_t v136 = r48;	// L263
    r47 = v136;	// L264
    int32_t v137 = r49;	// L265
    r48 = v137;	// L266
    int32_t v138 = r50;	// L267
    r49 = v138;	// L268
    int32_t v139 = r51;	// L269
    r50 = v139;	// L270
    int32_t v140 = r52;	// L271
    r51 = v140;	// L272
    int32_t v141 = r53;	// L273
    r52 = v141;	// L274
    int32_t v142 = r54;	// L275
    r53 = v142;	// L276
    int32_t v143 = r55;	// L277
    r54 = v143;	// L278
    int32_t v144 = r56;	// L279
    r55 = v144;	// L280
    int32_t v145 = r57;	// L281
    r56 = v145;	// L282
    int32_t v146 = r58;	// L283
    r57 = v146;	// L284
    int32_t v147 = r59;	// L285
    r58 = v147;	// L286
    int32_t v148 = r60;	// L287
    r59 = v148;	// L288
    int32_t v149 = r61;	// L289
    r60 = v149;	// L290
    int32_t v150 = r62;	// L291
    r61 = v150;	// L292
    int32_t v151 = r63;	// L293
    r62 = v151;	// L294
    int32_t v152 = v;	// L295
    r63 = v152;	// L296
    int32_t v153 = kb;	// L297
    ap_int<33> v154 = v153;	// L298
    bool v155 = v154 == 255;	// L299
    if (v155) {	// L300
      int32_t v156 = v;	// L301
      int32_t v157 = sh;	// L302
      int32_t v158 = v156 >> v157;	// L303
      v = v158;	// L304
      int32_t v159 = v;	// L305
      bool v160 = v159 < -128;	// L308
      if (v160) {	// L309
        v = -128;	// L310
      }
      int32_t v161 = v;	// L312
      bool v162 = v161 > 127;	// L314
      if (v162) {	// L315
        v = 127;	// L316
      }
      int32_t v163 = v;	// L318
      v1.write(v163);	// L319
    }
  }
}


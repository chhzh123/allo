
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void pack16_r0_0(
  hls::stream< int8_t >& v0,
  hls::stream< hls::vector< int8_t, 17 > >& v1,
  hls::stream< int64_t >& v2
) {	// L2
  int64_t hi;	// L6
  hi = 0;	// L7
  int32_t fl;	// L10
  fl = 0;	// L11
  bool st;	// L15
  st = 0;	// L16
  bool go;	// L20
  go = 1;	// L21
  while (true) {	// L22
    #pragma HLS pipeline II=1 style=flp
    bool v7 = go;	// L23
    if (!(v7)) break;
    bool v8 = st;	// L30
    int32_t v9 = v8;	// L31
    bool v10 = v9 == 0;	// L34
    if (v10) {	// L35
      int8_t v11[17];
      {
        hls::vector< int8_t, 17 > _vec = v1.read();
        for (int _iv0 = 0; _iv0 < 17; ++_iv0) {
          v11[_iv0] = _vec[_iv0];
        }
      }	// L36
      v0.write(1);	// L39
      int8_t v12 = v11[0];	// L40
      int32_t v13 = v12;	// L41
      int32_t v14 = v13 & 255;	// L44
      int64_t v15 = v14;	// L45
      int64_t w0_0;	// L46
      w0_0 = v15;	// L47
      int8_t v17 = v11[1];	// L48
      int32_t v18 = v17;	// L49
      int32_t v19 = v18 & 255;	// L52
      int64_t v20 = v19;	// L53
      int64_t w0_1;	// L54
      w0_1 = v20;	// L55
      int8_t v22 = v11[2];	// L56
      int32_t v23 = v22;	// L57
      int32_t v24 = v23 & 255;	// L60
      int64_t v25 = v24;	// L61
      int64_t w0_2;	// L62
      w0_2 = v25;	// L63
      int8_t v27 = v11[3];	// L64
      int32_t v28 = v27;	// L65
      int32_t v29 = v28 & 255;	// L68
      int64_t v30 = v29;	// L69
      int64_t w0_3;	// L70
      w0_3 = v30;	// L71
      int8_t v32 = v11[4];	// L72
      int32_t v33 = v32;	// L73
      int32_t v34 = v33 & 255;	// L76
      int64_t v35 = v34;	// L77
      int64_t w0_4;	// L78
      w0_4 = v35;	// L79
      int8_t v37 = v11[5];	// L80
      int32_t v38 = v37;	// L81
      int32_t v39 = v38 & 255;	// L84
      int64_t v40 = v39;	// L85
      int64_t w0_5;	// L86
      w0_5 = v40;	// L87
      int8_t v42 = v11[6];	// L88
      int32_t v43 = v42;	// L89
      int32_t v44 = v43 & 255;	// L92
      int64_t v45 = v44;	// L93
      int64_t w0_6;	// L94
      w0_6 = v45;	// L95
      int8_t v47 = v11[7];	// L96
      int32_t v48 = v47;	// L97
      int32_t v49 = v48 & 255;	// L100
      int64_t v50 = v49;	// L101
      int64_t w0_7;	// L102
      w0_7 = v50;	// L103
      int64_t w0;	// L107
      w0 = 0;	// L108
      int64_t v53 = w0;	// L109
      int64_t v54 = w0_0;	// L110
      int64_t v55 = v53 | v54;	// L115
      w0 = v55;	// L116
      int64_t v56 = w0;	// L117
      int64_t v57 = w0_1;	// L118
      int64_t v58 = v57 << 8;	// L122
      int64_t v59 = v56 | v58;	// L123
      w0 = v59;	// L124
      int64_t v60 = w0;	// L125
      int64_t v61 = w0_2;	// L126
      int64_t v62 = v61 << 16;	// L130
      int64_t v63 = v60 | v62;	// L131
      w0 = v63;	// L132
      int64_t v64 = w0;	// L133
      int64_t v65 = w0_3;	// L134
      int64_t v66 = v65 << 24;	// L138
      int64_t v67 = v64 | v66;	// L139
      w0 = v67;	// L140
      int64_t v68 = w0;	// L141
      int64_t v69 = w0_4;	// L142
      int64_t v70 = v69 << 32;	// L146
      int64_t v71 = v68 | v70;	// L147
      w0 = v71;	// L148
      int64_t v72 = w0;	// L149
      int64_t v73 = w0_5;	// L150
      int64_t v74 = v73 << 40;	// L154
      int64_t v75 = v72 | v74;	// L155
      w0 = v75;	// L156
      int64_t v76 = w0;	// L157
      int64_t v77 = w0_6;	// L158
      int64_t v78 = v77 << 48;	// L162
      int64_t v79 = v76 | v78;	// L163
      w0 = v79;	// L164
      int64_t v80 = w0;	// L165
      int64_t v81 = w0_7;	// L166
      int64_t v82 = v81 << 56;	// L170
      int64_t v83 = v80 | v82;	// L171
      w0 = v83;	// L172
      int8_t v84 = v11[8];	// L173
      int32_t v85 = v84;	// L174
      int32_t v86 = v85 & 255;	// L177
      int64_t v87 = v86;	// L178
      int64_t w1_0;	// L179
      w1_0 = v87;	// L180
      int8_t v89 = v11[9];	// L181
      int32_t v90 = v89;	// L182
      int32_t v91 = v90 & 255;	// L185
      int64_t v92 = v91;	// L186
      int64_t w1_1;	// L187
      w1_1 = v92;	// L188
      int8_t v94 = v11[10];	// L189
      int32_t v95 = v94;	// L190
      int32_t v96 = v95 & 255;	// L193
      int64_t v97 = v96;	// L194
      int64_t w1_2;	// L195
      w1_2 = v97;	// L196
      int8_t v99 = v11[11];	// L197
      int32_t v100 = v99;	// L198
      int32_t v101 = v100 & 255;	// L201
      int64_t v102 = v101;	// L202
      int64_t w1_3;	// L203
      w1_3 = v102;	// L204
      int8_t v104 = v11[12];	// L205
      int32_t v105 = v104;	// L206
      int32_t v106 = v105 & 255;	// L209
      int64_t v107 = v106;	// L210
      int64_t w1_4;	// L211
      w1_4 = v107;	// L212
      int8_t v109 = v11[13];	// L213
      int32_t v110 = v109;	// L214
      int32_t v111 = v110 & 255;	// L217
      int64_t v112 = v111;	// L218
      int64_t w1_5;	// L219
      w1_5 = v112;	// L220
      int8_t v114 = v11[14];	// L221
      int32_t v115 = v114;	// L222
      int32_t v116 = v115 & 255;	// L225
      int64_t v117 = v116;	// L226
      int64_t w1_6;	// L227
      w1_6 = v117;	// L228
      int8_t v119 = v11[15];	// L229
      int32_t v120 = v119;	// L230
      int32_t v121 = v120 & 255;	// L233
      int64_t v122 = v121;	// L234
      int64_t w1_7;	// L235
      w1_7 = v122;	// L236
      int64_t w1;	// L240
      w1 = 0;	// L241
      int64_t v125 = w1;	// L242
      int64_t v126 = w1_0;	// L243
      int64_t v127 = v125 | v126;	// L248
      w1 = v127;	// L249
      int64_t v128 = w1;	// L250
      int64_t v129 = w1_1;	// L251
      int64_t v130 = v129 << 8;	// L255
      int64_t v131 = v128 | v130;	// L256
      w1 = v131;	// L257
      int64_t v132 = w1;	// L258
      int64_t v133 = w1_2;	// L259
      int64_t v134 = v133 << 16;	// L263
      int64_t v135 = v132 | v134;	// L264
      w1 = v135;	// L265
      int64_t v136 = w1;	// L266
      int64_t v137 = w1_3;	// L267
      int64_t v138 = v137 << 24;	// L271
      int64_t v139 = v136 | v138;	// L272
      w1 = v139;	// L273
      int64_t v140 = w1;	// L274
      int64_t v141 = w1_4;	// L275
      int64_t v142 = v141 << 32;	// L279
      int64_t v143 = v140 | v142;	// L280
      w1 = v143;	// L281
      int64_t v144 = w1;	// L282
      int64_t v145 = w1_5;	// L283
      int64_t v146 = v145 << 40;	// L287
      int64_t v147 = v144 | v146;	// L288
      w1 = v147;	// L289
      int64_t v148 = w1;	// L290
      int64_t v149 = w1_6;	// L291
      int64_t v150 = v149 << 48;	// L295
      int64_t v151 = v148 | v150;	// L296
      w1 = v151;	// L297
      int64_t v152 = w1;	// L298
      int64_t v153 = w1_7;	// L299
      int64_t v154 = v153 << 56;	// L303
      int64_t v155 = v152 | v154;	// L304
      w1 = v155;	// L305
      int64_t v156 = w0;	// L306
      v2.write(v156);	// L307
      int64_t v157 = w1;	// L308
      hi = v157;	// L309
      int8_t v158 = v11[16];	// L310
      int32_t v159 = v158;	// L311
      fl = v159;	// L312
      st = 1;	// L316
    } else {
      int64_t v160 = hi;	// L318
      v2.write(v160);	// L319
      st = 0;	// L323
      int32_t v161 = fl;	// L324
      bool v162 = v161 != 0;	// L327
      if (v162) {	// L328
        go = 0;	// L332
      }
    }
  }
}


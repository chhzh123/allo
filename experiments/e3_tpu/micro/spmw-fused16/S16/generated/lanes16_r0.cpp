
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
void lanes16_r0_0(
  hls::stream< hls::vector< int32_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int32_t >& v6,
  hls::stream< int32_t >& v7,
  hls::stream< int32_t >& v8,
  hls::stream< int32_t >& v9,
  hls::stream< int32_t >& v10,
  hls::stream< int32_t >& v11,
  hls::stream< int32_t >& v12,
  hls::stream< int32_t >& v13,
  hls::stream< int32_t >& v14,
  hls::stream< int32_t >& v15,
  hls::stream< int32_t >& v16,
  hls::stream< int32_t >& v17,
  hls::stream< int32_t >& v18,
  hls::stream< int32_t >& v19,
  hls::stream< int32_t >& v20,
  hls::stream< int32_t >& v21,
  hls::stream< int32_t >& v22,
  hls::stream< int32_t >& v23,
  hls::stream< int32_t >& v24,
  hls::stream< int32_t >& v25,
  hls::stream< int32_t >& v26,
  hls::stream< int32_t >& v27,
  hls::stream< int32_t >& v28,
  hls::stream< int32_t >& v29,
  hls::stream< int32_t >& v30,
  hls::stream< int32_t >& v31,
  hls::stream< int32_t >& v32
) {	// L2
  int32_t v33[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v33[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v34 = v33[1];	// L4
  int32_t v35 = v34 & 31;	// L6
  int32_t sh;	// L7
  sh = v35;	// L8
  l_S__m_0__m: for (int _m = 0; _m < 256; _m++) {	// L9
  #pragma HLS pipeline II=1
    int32_t v38 = v17.read();	// L10
    int32_t y0;	// L11
    y0 = v38;	// L12
    int32_t v40 = y0;	// L13
    bool v41 = v40 < 0;	// L15
    if (v41) {	// L16
      y0 = 0;	// L17
    }
    int32_t v42 = y0;	// L19
    int32_t v43 = sh;	// L20
    int32_t v44 = v42 >> v43;	// L21
    y0 = v44;	// L22
    int32_t v45 = y0;	// L23
    bool v46 = v45 > 127;	// L25
    if (v46) {	// L26
      y0 = 127;	// L27
    }
    int32_t v47 = y0;	// L29
    v1.write(v47);	// L30
    int32_t v48 = v24.read();	// L31
    int32_t y1;	// L32
    y1 = v48;	// L33
    int32_t v50 = y1;	// L34
    bool v51 = v50 < 0;	// L35
    if (v51) {	// L36
      y1 = 0;	// L37
    }
    int32_t v52 = y1;	// L39
    int32_t v53 = sh;	// L40
    int32_t v54 = v52 >> v53;	// L41
    y1 = v54;	// L42
    int32_t v55 = y1;	// L43
    bool v56 = v55 > 127;	// L44
    if (v56) {	// L45
      y1 = 127;	// L46
    }
    int32_t v57 = y1;	// L48
    v8.write(v57);	// L49
    int32_t v58 = v25.read();	// L50
    int32_t y2;	// L51
    y2 = v58;	// L52
    int32_t v60 = y2;	// L53
    bool v61 = v60 < 0;	// L54
    if (v61) {	// L55
      y2 = 0;	// L56
    }
    int32_t v62 = y2;	// L58
    int32_t v63 = sh;	// L59
    int32_t v64 = v62 >> v63;	// L60
    y2 = v64;	// L61
    int32_t v65 = y2;	// L62
    bool v66 = v65 > 127;	// L63
    if (v66) {	// L64
      y2 = 127;	// L65
    }
    int32_t v67 = y2;	// L67
    v9.write(v67);	// L68
    int32_t v68 = v26.read();	// L69
    int32_t y3;	// L70
    y3 = v68;	// L71
    int32_t v70 = y3;	// L72
    bool v71 = v70 < 0;	// L73
    if (v71) {	// L74
      y3 = 0;	// L75
    }
    int32_t v72 = y3;	// L77
    int32_t v73 = sh;	// L78
    int32_t v74 = v72 >> v73;	// L79
    y3 = v74;	// L80
    int32_t v75 = y3;	// L81
    bool v76 = v75 > 127;	// L82
    if (v76) {	// L83
      y3 = 127;	// L84
    }
    int32_t v77 = y3;	// L86
    v10.write(v77);	// L87
    int32_t v78 = v27.read();	// L88
    int32_t y4;	// L89
    y4 = v78;	// L90
    int32_t v80 = y4;	// L91
    bool v81 = v80 < 0;	// L92
    if (v81) {	// L93
      y4 = 0;	// L94
    }
    int32_t v82 = y4;	// L96
    int32_t v83 = sh;	// L97
    int32_t v84 = v82 >> v83;	// L98
    y4 = v84;	// L99
    int32_t v85 = y4;	// L100
    bool v86 = v85 > 127;	// L101
    if (v86) {	// L102
      y4 = 127;	// L103
    }
    int32_t v87 = y4;	// L105
    v11.write(v87);	// L106
    int32_t v88 = v28.read();	// L107
    int32_t y5;	// L108
    y5 = v88;	// L109
    int32_t v90 = y5;	// L110
    bool v91 = v90 < 0;	// L111
    if (v91) {	// L112
      y5 = 0;	// L113
    }
    int32_t v92 = y5;	// L115
    int32_t v93 = sh;	// L116
    int32_t v94 = v92 >> v93;	// L117
    y5 = v94;	// L118
    int32_t v95 = y5;	// L119
    bool v96 = v95 > 127;	// L120
    if (v96) {	// L121
      y5 = 127;	// L122
    }
    int32_t v97 = y5;	// L124
    v12.write(v97);	// L125
    int32_t v98 = v29.read();	// L126
    int32_t y6;	// L127
    y6 = v98;	// L128
    int32_t v100 = y6;	// L129
    bool v101 = v100 < 0;	// L130
    if (v101) {	// L131
      y6 = 0;	// L132
    }
    int32_t v102 = y6;	// L134
    int32_t v103 = sh;	// L135
    int32_t v104 = v102 >> v103;	// L136
    y6 = v104;	// L137
    int32_t v105 = y6;	// L138
    bool v106 = v105 > 127;	// L139
    if (v106) {	// L140
      y6 = 127;	// L141
    }
    int32_t v107 = y6;	// L143
    v13.write(v107);	// L144
    int32_t v108 = v30.read();	// L145
    int32_t y7;	// L146
    y7 = v108;	// L147
    int32_t v110 = y7;	// L148
    bool v111 = v110 < 0;	// L149
    if (v111) {	// L150
      y7 = 0;	// L151
    }
    int32_t v112 = y7;	// L153
    int32_t v113 = sh;	// L154
    int32_t v114 = v112 >> v113;	// L155
    y7 = v114;	// L156
    int32_t v115 = y7;	// L157
    bool v116 = v115 > 127;	// L158
    if (v116) {	// L159
      y7 = 127;	// L160
    }
    int32_t v117 = y7;	// L162
    v14.write(v117);	// L163
    int32_t v118 = v31.read();	// L164
    int32_t y8;	// L165
    y8 = v118;	// L166
    int32_t v120 = y8;	// L167
    bool v121 = v120 < 0;	// L168
    if (v121) {	// L169
      y8 = 0;	// L170
    }
    int32_t v122 = y8;	// L172
    int32_t v123 = sh;	// L173
    int32_t v124 = v122 >> v123;	// L174
    y8 = v124;	// L175
    int32_t v125 = y8;	// L176
    bool v126 = v125 > 127;	// L177
    if (v126) {	// L178
      y8 = 127;	// L179
    }
    int32_t v127 = y8;	// L181
    v15.write(v127);	// L182
    int32_t v128 = v32.read();	// L183
    int32_t y9;	// L184
    y9 = v128;	// L185
    int32_t v130 = y9;	// L186
    bool v131 = v130 < 0;	// L187
    if (v131) {	// L188
      y9 = 0;	// L189
    }
    int32_t v132 = y9;	// L191
    int32_t v133 = sh;	// L192
    int32_t v134 = v132 >> v133;	// L193
    y9 = v134;	// L194
    int32_t v135 = y9;	// L195
    bool v136 = v135 > 127;	// L196
    if (v136) {	// L197
      y9 = 127;	// L198
    }
    int32_t v137 = y9;	// L200
    v16.write(v137);	// L201
    int32_t v138 = v18.read();	// L202
    int32_t y10;	// L203
    y10 = v138;	// L204
    int32_t v140 = y10;	// L205
    bool v141 = v140 < 0;	// L206
    if (v141) {	// L207
      y10 = 0;	// L208
    }
    int32_t v142 = y10;	// L210
    int32_t v143 = sh;	// L211
    int32_t v144 = v142 >> v143;	// L212
    y10 = v144;	// L213
    int32_t v145 = y10;	// L214
    bool v146 = v145 > 127;	// L215
    if (v146) {	// L216
      y10 = 127;	// L217
    }
    int32_t v147 = y10;	// L219
    v2.write(v147);	// L220
    int32_t v148 = v19.read();	// L221
    int32_t y11;	// L222
    y11 = v148;	// L223
    int32_t v150 = y11;	// L224
    bool v151 = v150 < 0;	// L225
    if (v151) {	// L226
      y11 = 0;	// L227
    }
    int32_t v152 = y11;	// L229
    int32_t v153 = sh;	// L230
    int32_t v154 = v152 >> v153;	// L231
    y11 = v154;	// L232
    int32_t v155 = y11;	// L233
    bool v156 = v155 > 127;	// L234
    if (v156) {	// L235
      y11 = 127;	// L236
    }
    int32_t v157 = y11;	// L238
    v3.write(v157);	// L239
    int32_t v158 = v20.read();	// L240
    int32_t y12;	// L241
    y12 = v158;	// L242
    int32_t v160 = y12;	// L243
    bool v161 = v160 < 0;	// L244
    if (v161) {	// L245
      y12 = 0;	// L246
    }
    int32_t v162 = y12;	// L248
    int32_t v163 = sh;	// L249
    int32_t v164 = v162 >> v163;	// L250
    y12 = v164;	// L251
    int32_t v165 = y12;	// L252
    bool v166 = v165 > 127;	// L253
    if (v166) {	// L254
      y12 = 127;	// L255
    }
    int32_t v167 = y12;	// L257
    v4.write(v167);	// L258
    int32_t v168 = v21.read();	// L259
    int32_t y13;	// L260
    y13 = v168;	// L261
    int32_t v170 = y13;	// L262
    bool v171 = v170 < 0;	// L263
    if (v171) {	// L264
      y13 = 0;	// L265
    }
    int32_t v172 = y13;	// L267
    int32_t v173 = sh;	// L268
    int32_t v174 = v172 >> v173;	// L269
    y13 = v174;	// L270
    int32_t v175 = y13;	// L271
    bool v176 = v175 > 127;	// L272
    if (v176) {	// L273
      y13 = 127;	// L274
    }
    int32_t v177 = y13;	// L276
    v5.write(v177);	// L277
    int32_t v178 = v22.read();	// L278
    int32_t y14;	// L279
    y14 = v178;	// L280
    int32_t v180 = y14;	// L281
    bool v181 = v180 < 0;	// L282
    if (v181) {	// L283
      y14 = 0;	// L284
    }
    int32_t v182 = y14;	// L286
    int32_t v183 = sh;	// L287
    int32_t v184 = v182 >> v183;	// L288
    y14 = v184;	// L289
    int32_t v185 = y14;	// L290
    bool v186 = v185 > 127;	// L291
    if (v186) {	// L292
      y14 = 127;	// L293
    }
    int32_t v187 = y14;	// L295
    v6.write(v187);	// L296
    int32_t v188 = v23.read();	// L297
    int32_t y15;	// L298
    y15 = v188;	// L299
    int32_t v190 = y15;	// L300
    bool v191 = v190 < 0;	// L301
    if (v191) {	// L302
      y15 = 0;	// L303
    }
    int32_t v192 = y15;	// L305
    int32_t v193 = sh;	// L306
    int32_t v194 = v192 >> v193;	// L307
    y15 = v194;	// L308
    int32_t v195 = y15;	// L309
    bool v196 = v195 > 127;	// L310
    if (v196) {	// L311
      y15 = 127;	// L312
    }
    int32_t v197 = y15;	// L314
    v7.write(v197);	// L315
  }
}


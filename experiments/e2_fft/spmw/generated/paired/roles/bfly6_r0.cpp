
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
int32_t _st_sel[128] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};	// L2
void bfly6_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L3
  // placeholder for const int32_t _st_sel	// L17
  float z0r[256];	// L18
  for (int v6 = 0; v6 < 256; v6++) {	// L19
    z0r[v6] = (float)0.000000;	// L19
  }
  float z0i[256];	// L20
  for (int v8 = 0; v8 < 256; v8++) {	// L21
    z0i[v8] = (float)0.000000;	// L21
  }
  float z1r[256];	// L22
  for (int v10 = 0; v10 < 256; v10++) {	// L23
    z1r[v10] = (float)0.000000;	// L23
  }
  float z1i[256];	// L24
  for (int v12 = 0; v12 < 256; v12++) {	// L25
    z1i[v12] = (float)0.000000;	// L25
  }
  l_S_t_0_t: for (int t = 0; t < 4744; t++) {	// L26
    float v14[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v14[_iv0] = _vec[_iv0];
      }
    }	// L27
    float v15[2];
    {
      hls::vector< float, 2 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v15[_iv0] = _vec[_iv0];
      }
    }	// L28
    int64_t v16 = t;	// L29
    int64_t v17 = v16 * 2;	// L30
    int64_t v18 = v17 & 511;	// L31
    int32_t v19 = v18;	// L32
    int32_t w0;	// L33
    w0 = v19;	// L34
    int32_t v21 = w0;	// L35
    int32_t v22 = v21 >> 7;	// L36
    int32_t v23 = v22 & 1;	// L37
    int32_t wsel;	// L38
    wsel = v23;	// L39
    int32_t v25 = w0;	// L40
    int32_t v26 = v25 >> 1;	// L41
    int32_t rw;	// L42
    rw = v26;	// L43
    ap_int<34> v28 = t;	// L44
    ap_int<34> v29 = v28 + 57;	// L45
    int32_t v30 = v29;	// L46
    int32_t u;	// L47
    u = v30;	// L48
    int32_t v32 = u;	// L49
    ap_int<33> v33 = v32;	// L50
    ap_int<33> v34 = v33 & 127;	// L51
    int32_t v35 = v34;	// L52
    int32_t j;	// L53
    j = v35;	// L54
    int32_t v37 = u;	// L55
    int32_t v38 = v37 >> 7;	// L56
    int32_t v39 = v38 & 1;	// L57
    int32_t blk;	// L58
    blk = v39;	// L59
    int32_t v41 = blk;	// L60
    int32_t v42 = v41 << 8;	// L61
    ap_int<33> v43 = v42;	// L62
    ap_int<33> v44 = v43 + 268;	// L63
    int32_t v45 = j;	// L64
    ap_int<34> v46 = v44;	// L65
    ap_int<34> v47 = v45;	// L66
    ap_int<34> v48 = v46 + v47;	// L67
    ap_int<34> v49 = v48 & 511;	// L68
    int32_t v50 = v49;	// L69
    int32_t pa;	// L70
    pa = v50;	// L71
    int32_t v52 = pa;	// L72
    ap_int<33> v53 = v52;	// L73
    ap_int<33> v54 = v53 + 128;	// L74
    ap_int<33> v55 = v54 & 511;	// L75
    int32_t v56 = v55;	// L76
    int32_t pb;	// L77
    pb = v56;	// L78
    int32_t v58 = pa;	// L79
    int32_t v59 = v58 & 1;	// L80
    int32_t v60 = v58 >> 7;	// L82
    int32_t v61 = v60 & 1;	// L83
    int32_t v62 = v59 ^ v61;	// L84
    int32_t sel;	// L85
    sel = v62;	// L86
    int32_t v64 = pa;	// L87
    int32_t v65 = v64 >> 1;	// L88
    int32_t ra;	// L89
    ra = v65;	// L90
    int32_t v67 = pb;	// L91
    int32_t v68 = v67 >> 1;	// L92
    int32_t rb;	// L93
    rb = v68;	// L94
    int32_t v70 = ra;	// L95
    int32_t r0;	// L96
    r0 = v70;	// L97
    int32_t v72 = rb;	// L98
    int32_t r1;	// L99
    r1 = v72;	// L100
    int32_t v74 = sel;	// L101
    bool v75 = v74 == 1;	// L102
    if (v75) {	// L103
      int32_t v76 = rb;	// L104
      r0 = v76;	// L105
      int32_t v77 = ra;	// L106
      r1 = v77;	// L107
    }
    int32_t v78 = r0;	// L109
    int v79 = v78;	// L110
    float v80 = z0r[v79];	// L111
    float g0r;	// L112
    g0r = v80;	// L113
    int32_t v82 = r0;	// L114
    int v83 = v82;	// L115
    float v84 = z0i[v83];	// L116
    float g0i;	// L117
    g0i = v84;	// L118
    int32_t v86 = r1;	// L119
    int v87 = v86;	// L120
    float v88 = z1r[v87];	// L121
    float g1r;	// L122
    g1r = v88;	// L123
    int32_t v90 = r1;	// L124
    int v91 = v90;	// L125
    float v92 = z1i[v91];	// L126
    float g1i;	// L127
    g1i = v92;	// L128
    float v94 = g0r;	// L129
    float ar;	// L130
    ar = v94;	// L131
    float v96 = g0i;	// L132
    float ai;	// L133
    ai = v96;	// L134
    float v98 = g1r;	// L135
    float br;	// L136
    br = v98;	// L137
    float v100 = g1i;	// L138
    float bi;	// L139
    bi = v100;	// L140
    int32_t v102 = sel;	// L141
    bool v103 = v102 == 1;	// L142
    if (v103) {	// L143
      float v104 = g1r;	// L144
      ar = v104;	// L145
      float v105 = g1i;	// L146
      ai = v105;	// L147
      float v106 = g0r;	// L148
      br = v106;	// L149
      float v107 = g0i;	// L150
      bi = v107;	// L151
    }
    int32_t v108 = wsel;	// L153
    bool v109 = v108 == 0;	// L154
    if (v109) {	// L155
      float v110 = v14[0];	// L156
      int32_t v111 = rw;	// L157
      int v112 = v111;	// L158
      z0r[v112] = v110;	// L159
      float v113 = v14[1];	// L160
      int32_t v114 = rw;	// L161
      int v115 = v114;	// L162
      z0i[v115] = v113;	// L163
      float v116 = v15[0];	// L164
      int32_t v117 = rw;	// L165
      int v118 = v117;	// L166
      z1r[v118] = v116;	// L167
      float v119 = v15[1];	// L168
      int32_t v120 = rw;	// L169
      int v121 = v120;	// L170
      z1i[v121] = v119;	// L171
    } else {
      float v122 = v14[0];	// L173
      int32_t v123 = rw;	// L174
      int v124 = v123;	// L175
      z1r[v124] = v122;	// L176
      float v125 = v14[1];	// L177
      int32_t v126 = rw;	// L178
      int v127 = v126;	// L179
      z1i[v127] = v125;	// L180
      float v128 = v15[0];	// L181
      int32_t v129 = rw;	// L182
      int v130 = v129;	// L183
      z0r[v130] = v128;	// L184
      float v131 = v15[1];	// L185
      int32_t v132 = rw;	// L186
      int v133 = v132;	// L187
      z0i[v133] = v131;	// L188
    }
    float v134 = ar;	// L190
    float v135 = br;	// L191
    float v136 = v134 - v135;	// L192
    #pragma HLS bind_op variable=v136 op=fsub impl=fabric
    float dr;	// L193
    dr = v136;	// L194
    float v138 = ai;	// L195
    float v139 = bi;	// L196
    float v140 = v138 - v139;	// L197
    #pragma HLS bind_op variable=v140 op=fsub impl=fabric
    float di;	// L198
    di = v140;	// L199
    int32_t v142 = j;	// L200
    int v143 = v142;	// L201
    int32_t v144 = _st_sel[v143];	// L202
    int32_t rot;	// L203
    rot = v144;	// L204
    float p[2];	// L205
    for (int v147 = 0; v147 < 2; v147++) {	// L206
      p[v147] = (float)0.000000;	// L206
    }
    float q[2];	// L207
    for (int v149 = 0; v149 < 2; v149++) {	// L208
      q[v149] = (float)0.000000;	// L208
    }
    float v150 = ar;	// L209
    float v151 = br;	// L210
    float v152 = v150 + v151;	// L211
    #pragma HLS bind_op variable=v152 op=fadd impl=fabric
    p[0] = v152;	// L212
    float v153 = ai;	// L213
    float v154 = bi;	// L214
    float v155 = v153 + v154;	// L215
    #pragma HLS bind_op variable=v155 op=fadd impl=fabric
    p[1] = v155;	// L216
    float v156 = dr;	// L217
    q[0] = v156;	// L218
    float v157 = di;	// L219
    q[1] = v157;	// L220
    int32_t v158 = rot;	// L221
    bool v159 = v158 == 1;	// L222
    if (v159) {	// L223
      float v160 = di;	// L224
      q[0] = v160;	// L225
      float v161 = dr;	// L226
      float v162 = (float)0.000000 - v161;	// L227
      #pragma HLS bind_op variable=v162 op=fsub impl=fabric
      q[1] = v162;	// L228
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v2.write(_vec);
    }	// L230
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L231
  }
}

/// This is top function.
void top(

) {	// L235
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v163;
  #pragma HLS stream variable=v163 depth=8	// L236
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v164;
  #pragma HLS stream variable=v164 depth=8	// L237
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v165;
  #pragma HLS stream variable=v165 depth=8	// L238
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v166;
  #pragma HLS stream variable=v166 depth=8	// L239
  bfly6_r0_0(v163, v165, v164, v166);	// L240
}


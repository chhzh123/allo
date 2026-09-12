
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
float _st_tw[128][2] = {1.000000e+00, 0.000000e+00, 9.996988e-01, -2.454123e-02, 9.987954e-01, -4.906768e-02, 9.972904e-01, -7.356457e-02, 9.951847e-01, -9.801714e-02, 9.924796e-01, -1.224107e-01, 9.891765e-01, -1.467305e-01, 9.852777e-01, -1.709619e-01, 9.807853e-01, -1.950903e-01, 9.757021e-01, -2.191012e-01, 9.700313e-01, -2.429802e-01, 9.637761e-01, -2.667128e-01, 9.569404e-01, -2.902847e-01, 9.495282e-01, -3.136818e-01, 9.415441e-01, -3.368899e-01, 9.329928e-01, -3.598951e-01, 9.238795e-01, -3.826834e-01, 9.142098e-01, -4.052413e-01, 9.039893e-01, -4.275551e-01, 8.932243e-01, -4.496113e-01, 8.819213e-01, -4.713967e-01, 8.700870e-01, -4.928982e-01, 8.577286e-01, -5.141028e-01, 8.448536e-01, -5.349976e-01, 8.314696e-01, -5.555702e-01, 8.175848e-01, -5.758082e-01, 8.032075e-01, -5.956993e-01, 7.883464e-01, -6.152316e-01, 7.730104e-01, -6.343933e-01, 7.572088e-01, -6.531729e-01, 7.409511e-01, -6.715590e-01, 7.242471e-01, -6.895406e-01, 7.071068e-01, -7.071068e-01, 6.895406e-01, -7.242471e-01, 6.715590e-01, -7.409511e-01, 6.531729e-01, -7.572088e-01, 6.343933e-01, -7.730104e-01, 6.152316e-01, -7.883464e-01, 5.956993e-01, -8.032075e-01, 5.758082e-01, -8.175848e-01, 5.555702e-01, -8.314696e-01, 5.349976e-01, -8.448536e-01, 5.141028e-01, -8.577286e-01, 4.928982e-01, -8.700870e-01, 4.713967e-01, -8.819213e-01, 4.496113e-01, -8.932243e-01, 4.275551e-01, -9.039893e-01, 4.052413e-01, -9.142098e-01, 3.826834e-01, -9.238795e-01, 3.598951e-01, -9.329928e-01, 3.368899e-01, -9.415441e-01, 3.136818e-01, -9.495282e-01, 2.902847e-01, -9.569404e-01, 2.667128e-01, -9.637761e-01, 2.429802e-01, -9.700313e-01, 2.191012e-01, -9.757021e-01, 1.950903e-01, -9.807853e-01, 1.709619e-01, -9.852777e-01, 1.467305e-01, -9.891765e-01, 1.224107e-01, -9.924796e-01, 9.801714e-02, -9.951847e-01, 7.356457e-02, -9.972904e-01, 4.906768e-02, -9.987954e-01, 2.454123e-02, -9.996988e-01, 6.123234e-17, -1.000000e+00, -2.454123e-02, -9.996988e-01, -4.906768e-02, -9.987954e-01, -7.356457e-02, -9.972904e-01, -9.801714e-02, -9.951847e-01, -1.224107e-01, -9.924796e-01, -1.467305e-01, -9.891765e-01, -1.709619e-01, -9.852777e-01, -1.950903e-01, -9.807853e-01, -2.191012e-01, -9.757021e-01, -2.429802e-01, -9.700313e-01, -2.667128e-01, -9.637761e-01, -2.902847e-01, -9.569404e-01, -3.136818e-01, -9.495282e-01, -3.368899e-01, -9.415441e-01, -3.598951e-01, -9.329928e-01, -3.826834e-01, -9.238795e-01, -4.052413e-01, -9.142098e-01, -4.275551e-01, -9.039893e-01, -4.496113e-01, -8.932243e-01, -4.713967e-01, -8.819213e-01, -4.928982e-01, -8.700870e-01, -5.141028e-01, -8.577286e-01, -5.349976e-01, -8.448536e-01, -5.555702e-01, -8.314696e-01, -5.758082e-01, -8.175848e-01, -5.956993e-01, -8.032075e-01, -6.152316e-01, -7.883464e-01, -6.343933e-01, -7.730104e-01, -6.531729e-01, -7.572088e-01, -6.715590e-01, -7.409511e-01, -6.895406e-01, -7.242471e-01, -7.071068e-01, -7.071068e-01, -7.242471e-01, -6.895406e-01, -7.409511e-01, -6.715590e-01, -7.572088e-01, -6.531729e-01, -7.730104e-01, -6.343933e-01, -7.883464e-01, -6.152316e-01, -8.032075e-01, -5.956993e-01, -8.175848e-01, -5.758082e-01, -8.314696e-01, -5.555702e-01, -8.448536e-01, -5.349976e-01, -8.577286e-01, -5.141028e-01, -8.700870e-01, -4.928982e-01, -8.819213e-01, -4.713967e-01, -8.932243e-01, -4.496113e-01, -9.039893e-01, -4.275551e-01, -9.142098e-01, -4.052413e-01, -9.238795e-01, -3.826834e-01, -9.329928e-01, -3.598951e-01, -9.415441e-01, -3.368899e-01, -9.495282e-01, -3.136818e-01, -9.569404e-01, -2.902847e-01, -9.637761e-01, -2.667128e-01, -9.700313e-01, -2.429802e-01, -9.757021e-01, -2.191012e-01, -9.807853e-01, -1.950903e-01, -9.852777e-01, -1.709619e-01, -9.891765e-01, -1.467305e-01, -9.924796e-01, -1.224107e-01, -9.951847e-01, -9.801714e-02, -9.972904e-01, -7.356457e-02, -9.987954e-01, -4.906768e-02, -9.996988e-01, -2.454123e-02};	// L2
void bfly0_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L3
  // placeholder for const float _st_tw	// L18
  float z0r[256];	// L19
  for (int v6 = 0; v6 < 256; v6++) {	// L20
    z0r[v6] = (float)0.000000;	// L20
  }
  float z0i[256];	// L21
  for (int v8 = 0; v8 < 256; v8++) {	// L22
    z0i[v8] = (float)0.000000;	// L22
  }
  float z1r[256];	// L23
  for (int v10 = 0; v10 < 256; v10++) {	// L24
    z1r[v10] = (float)0.000000;	// L24
  }
  float z1i[256];	// L25
  for (int v12 = 0; v12 < 256; v12++) {	// L26
    z1i[v12] = (float)0.000000;	// L26
  }
  l_S_t_0_t: for (int t = 0; t < 4744; t++) {	// L27
    float v14[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v14[_iv0] = _vec[_iv0];
      }
    }	// L28
    float v15[2];
    {
      hls::vector< float, 2 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v15[_iv0] = _vec[_iv0];
      }
    }	// L29
    int64_t v16 = t;	// L30
    int64_t v17 = v16 * 2;	// L31
    int64_t v18 = v17 & 511;	// L32
    int32_t v19 = v18;	// L33
    int32_t w0;	// L34
    w0 = v19;	// L35
    int32_t v21 = w0;	// L36
    int32_t v22 = v21 >> 7;	// L37
    int32_t v23 = v22 & 1;	// L38
    int32_t wsel;	// L39
    wsel = v23;	// L40
    int32_t v25 = w0;	// L41
    int32_t v26 = v25 >> 1;	// L42
    int32_t rw;	// L43
    rw = v26;	// L44
    ap_int<34> v28 = t;	// L45
    ap_int<34> v29 = v28 + 191;	// L46
    int32_t v30 = v29;	// L47
    int32_t u;	// L48
    u = v30;	// L49
    int32_t v32 = u;	// L50
    ap_int<33> v33 = v32;	// L51
    ap_int<33> v34 = v33 & 127;	// L52
    int32_t v35 = v34;	// L53
    int32_t j;	// L54
    j = v35;	// L55
    int32_t v37 = u;	// L56
    int32_t v38 = v37 >> 7;	// L57
    int32_t v39 = v38 & 1;	// L58
    int32_t blk;	// L59
    blk = v39;	// L60
    int32_t v41 = blk;	// L61
    int32_t v42 = v41 << 8;	// L62
    int32_t v43 = j;	// L63
    ap_int<34> v44 = v42;	// L64
    ap_int<34> v45 = v43;	// L65
    ap_int<34> v46 = v44 + v45;	// L66
    ap_int<34> v47 = v46 & 511;	// L67
    int32_t v48 = v47;	// L68
    int32_t pa;	// L69
    pa = v48;	// L70
    int32_t v50 = pa;	// L71
    ap_int<33> v51 = v50;	// L72
    ap_int<33> v52 = v51 + 128;	// L73
    ap_int<33> v53 = v52 & 511;	// L74
    int32_t v54 = v53;	// L75
    int32_t pb;	// L76
    pb = v54;	// L77
    int32_t v56 = pa;	// L78
    int32_t v57 = v56 & 1;	// L79
    int32_t v58 = v56 >> 7;	// L81
    int32_t v59 = v58 & 1;	// L82
    int32_t v60 = v57 ^ v59;	// L83
    int32_t sel;	// L84
    sel = v60;	// L85
    int32_t v62 = pa;	// L86
    int32_t v63 = v62 >> 1;	// L87
    int32_t ra;	// L88
    ra = v63;	// L89
    int32_t v65 = pb;	// L90
    int32_t v66 = v65 >> 1;	// L91
    int32_t rb;	// L92
    rb = v66;	// L93
    int32_t v68 = ra;	// L94
    int32_t r0;	// L95
    r0 = v68;	// L96
    int32_t v70 = rb;	// L97
    int32_t r1;	// L98
    r1 = v70;	// L99
    int32_t v72 = sel;	// L100
    bool v73 = v72 == 1;	// L101
    if (v73) {	// L102
      int32_t v74 = rb;	// L103
      r0 = v74;	// L104
      int32_t v75 = ra;	// L105
      r1 = v75;	// L106
    }
    int32_t v76 = r0;	// L108
    int v77 = v76;	// L109
    float v78 = z0r[v77];	// L110
    float g0r;	// L111
    g0r = v78;	// L112
    int32_t v80 = r0;	// L113
    int v81 = v80;	// L114
    float v82 = z0i[v81];	// L115
    float g0i;	// L116
    g0i = v82;	// L117
    int32_t v84 = r1;	// L118
    int v85 = v84;	// L119
    float v86 = z1r[v85];	// L120
    float g1r;	// L121
    g1r = v86;	// L122
    int32_t v88 = r1;	// L123
    int v89 = v88;	// L124
    float v90 = z1i[v89];	// L125
    float g1i;	// L126
    g1i = v90;	// L127
    float v92 = g0r;	// L128
    float ar;	// L129
    ar = v92;	// L130
    float v94 = g0i;	// L131
    float ai;	// L132
    ai = v94;	// L133
    float v96 = g1r;	// L134
    float br;	// L135
    br = v96;	// L136
    float v98 = g1i;	// L137
    float bi;	// L138
    bi = v98;	// L139
    int32_t v100 = sel;	// L140
    bool v101 = v100 == 1;	// L141
    if (v101) {	// L142
      float v102 = g1r;	// L143
      ar = v102;	// L144
      float v103 = g1i;	// L145
      ai = v103;	// L146
      float v104 = g0r;	// L147
      br = v104;	// L148
      float v105 = g0i;	// L149
      bi = v105;	// L150
    }
    int32_t v106 = wsel;	// L152
    bool v107 = v106 == 0;	// L153
    if (v107) {	// L154
      float v108 = v14[0];	// L155
      int32_t v109 = rw;	// L156
      int v110 = v109;	// L157
      z0r[v110] = v108;	// L158
      float v111 = v14[1];	// L159
      int32_t v112 = rw;	// L160
      int v113 = v112;	// L161
      z0i[v113] = v111;	// L162
      float v114 = v15[0];	// L163
      int32_t v115 = rw;	// L164
      int v116 = v115;	// L165
      z1r[v116] = v114;	// L166
      float v117 = v15[1];	// L167
      int32_t v118 = rw;	// L168
      int v119 = v118;	// L169
      z1i[v119] = v117;	// L170
    } else {
      float v120 = v14[0];	// L172
      int32_t v121 = rw;	// L173
      int v122 = v121;	// L174
      z1r[v122] = v120;	// L175
      float v123 = v14[1];	// L176
      int32_t v124 = rw;	// L177
      int v125 = v124;	// L178
      z1i[v125] = v123;	// L179
      float v126 = v15[0];	// L180
      int32_t v127 = rw;	// L181
      int v128 = v127;	// L182
      z0r[v128] = v126;	// L183
      float v129 = v15[1];	// L184
      int32_t v130 = rw;	// L185
      int v131 = v130;	// L186
      z0i[v131] = v129;	// L187
    }
    int32_t v132 = j;	// L189
    int v133 = v132;	// L190
    float v134 = _st_tw[v133][0];	// L191
    float wr;	// L192
    wr = v134;	// L193
    int32_t v136 = j;	// L194
    int v137 = v136;	// L195
    float v138 = _st_tw[v137][1];	// L196
    float wi;	// L197
    wi = v138;	// L198
    float v140 = ar;	// L199
    float v141 = br;	// L200
    float v142 = v140 - v141;	// L201
    #pragma HLS bind_op variable=v142 op=fsub impl=fabric
    float dr;	// L202
    dr = v142;	// L203
    float v144 = ai;	// L204
    float v145 = bi;	// L205
    float v146 = v144 - v145;	// L206
    #pragma HLS bind_op variable=v146 op=fsub impl=fabric
    float di;	// L207
    di = v146;	// L208
    float p[2];	// L209
    for (int v149 = 0; v149 < 2; v149++) {	// L210
      p[v149] = (float)0.000000;	// L210
    }
    float q[2];	// L211
    for (int v151 = 0; v151 < 2; v151++) {	// L212
      q[v151] = (float)0.000000;	// L212
    }
    float v152 = ar;	// L213
    float v153 = br;	// L214
    float v154 = v152 + v153;	// L215
    #pragma HLS bind_op variable=v154 op=fadd impl=fabric
    p[0] = v154;	// L216
    float v155 = ai;	// L217
    float v156 = bi;	// L218
    float v157 = v155 + v156;	// L219
    #pragma HLS bind_op variable=v157 op=fadd impl=fabric
    p[1] = v157;	// L220
    float v158 = dr;	// L221
    float v159 = wr;	// L222
    float v160 = v158 * v159;	// L223
    float v161 = di;	// L224
    float v162 = wi;	// L225
    float v163 = v161 * v162;	// L226
    float v164 = v160 - v163;	// L227
    #pragma HLS bind_op variable=v164 op=fsub impl=fabric
    q[0] = v164;	// L228
    float v165 = dr;	// L229
    float v166 = wi;	// L230
    float v167 = v165 * v166;	// L231
    float v168 = di;	// L232
    float v169 = wr;	// L233
    float v170 = v168 * v169;	// L234
    float v171 = v167 + v170;	// L235
    #pragma HLS bind_op variable=v171 op=fadd impl=fabric
    q[1] = v171;	// L236
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v2.write(_vec);
    }	// L237
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L238
  }
}

/// This is top function.
void top(

) {	// L242
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v172;
  #pragma HLS stream variable=v172 depth=8	// L243
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v173;
  #pragma HLS stream variable=v173 depth=8	// L244
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v174;
  #pragma HLS stream variable=v174 depth=8	// L245
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v175;
  #pragma HLS stream variable=v175 depth=8	// L246
  bfly0_r0_0(v172, v174, v173, v175);	// L247
}


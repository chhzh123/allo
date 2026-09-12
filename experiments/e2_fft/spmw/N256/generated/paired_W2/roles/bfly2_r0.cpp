
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
float _st_tw[128][2] = {1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 9.951847e-01, -9.801714e-02, 9.951847e-01, -9.801714e-02, 9.951847e-01, -9.801714e-02, 9.951847e-01, -9.801714e-02, 9.807853e-01, -1.950903e-01, 9.807853e-01, -1.950903e-01, 9.807853e-01, -1.950903e-01, 9.807853e-01, -1.950903e-01, 9.569404e-01, -2.902847e-01, 9.569404e-01, -2.902847e-01, 9.569404e-01, -2.902847e-01, 9.569404e-01, -2.902847e-01, 9.238795e-01, -3.826834e-01, 9.238795e-01, -3.826834e-01, 9.238795e-01, -3.826834e-01, 9.238795e-01, -3.826834e-01, 8.819213e-01, -4.713967e-01, 8.819213e-01, -4.713967e-01, 8.819213e-01, -4.713967e-01, 8.819213e-01, -4.713967e-01, 8.314696e-01, -5.555702e-01, 8.314696e-01, -5.555702e-01, 8.314696e-01, -5.555702e-01, 8.314696e-01, -5.555702e-01, 7.730104e-01, -6.343933e-01, 7.730104e-01, -6.343933e-01, 7.730104e-01, -6.343933e-01, 7.730104e-01, -6.343933e-01, 7.071068e-01, -7.071068e-01, 7.071068e-01, -7.071068e-01, 7.071068e-01, -7.071068e-01, 7.071068e-01, -7.071068e-01, 6.343933e-01, -7.730104e-01, 6.343933e-01, -7.730104e-01, 6.343933e-01, -7.730104e-01, 6.343933e-01, -7.730104e-01, 5.555702e-01, -8.314696e-01, 5.555702e-01, -8.314696e-01, 5.555702e-01, -8.314696e-01, 5.555702e-01, -8.314696e-01, 4.713967e-01, -8.819213e-01, 4.713967e-01, -8.819213e-01, 4.713967e-01, -8.819213e-01, 4.713967e-01, -8.819213e-01, 3.826834e-01, -9.238795e-01, 3.826834e-01, -9.238795e-01, 3.826834e-01, -9.238795e-01, 3.826834e-01, -9.238795e-01, 2.902847e-01, -9.569404e-01, 2.902847e-01, -9.569404e-01, 2.902847e-01, -9.569404e-01, 2.902847e-01, -9.569404e-01, 1.950903e-01, -9.807853e-01, 1.950903e-01, -9.807853e-01, 1.950903e-01, -9.807853e-01, 1.950903e-01, -9.807853e-01, 9.801714e-02, -9.951847e-01, 9.801714e-02, -9.951847e-01, 9.801714e-02, -9.951847e-01, 9.801714e-02, -9.951847e-01, 6.123234e-17, -1.000000e+00, 6.123234e-17, -1.000000e+00, 6.123234e-17, -1.000000e+00, 6.123234e-17, -1.000000e+00, -9.801714e-02, -9.951847e-01, -9.801714e-02, -9.951847e-01, -9.801714e-02, -9.951847e-01, -9.801714e-02, -9.951847e-01, -1.950903e-01, -9.807853e-01, -1.950903e-01, -9.807853e-01, -1.950903e-01, -9.807853e-01, -1.950903e-01, -9.807853e-01, -2.902847e-01, -9.569404e-01, -2.902847e-01, -9.569404e-01, -2.902847e-01, -9.569404e-01, -2.902847e-01, -9.569404e-01, -3.826834e-01, -9.238795e-01, -3.826834e-01, -9.238795e-01, -3.826834e-01, -9.238795e-01, -3.826834e-01, -9.238795e-01, -4.713967e-01, -8.819213e-01, -4.713967e-01, -8.819213e-01, -4.713967e-01, -8.819213e-01, -4.713967e-01, -8.819213e-01, -5.555702e-01, -8.314696e-01, -5.555702e-01, -8.314696e-01, -5.555702e-01, -8.314696e-01, -5.555702e-01, -8.314696e-01, -6.343933e-01, -7.730104e-01, -6.343933e-01, -7.730104e-01, -6.343933e-01, -7.730104e-01, -6.343933e-01, -7.730104e-01, -7.071068e-01, -7.071068e-01, -7.071068e-01, -7.071068e-01, -7.071068e-01, -7.071068e-01, -7.071068e-01, -7.071068e-01, -7.730104e-01, -6.343933e-01, -7.730104e-01, -6.343933e-01, -7.730104e-01, -6.343933e-01, -7.730104e-01, -6.343933e-01, -8.314696e-01, -5.555702e-01, -8.314696e-01, -5.555702e-01, -8.314696e-01, -5.555702e-01, -8.314696e-01, -5.555702e-01, -8.819213e-01, -4.713967e-01, -8.819213e-01, -4.713967e-01, -8.819213e-01, -4.713967e-01, -8.819213e-01, -4.713967e-01, -9.238795e-01, -3.826834e-01, -9.238795e-01, -3.826834e-01, -9.238795e-01, -3.826834e-01, -9.238795e-01, -3.826834e-01, -9.569404e-01, -2.902847e-01, -9.569404e-01, -2.902847e-01, -9.569404e-01, -2.902847e-01, -9.569404e-01, -2.902847e-01, -9.807853e-01, -1.950903e-01, -9.807853e-01, -1.950903e-01, -9.807853e-01, -1.950903e-01, -9.807853e-01, -1.950903e-01, -9.951847e-01, -9.801714e-02, -9.951847e-01, -9.801714e-02, -9.951847e-01, -9.801714e-02, -9.951847e-01, -9.801714e-02};	// L2
void bfly2_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L3
  // placeholder for const float _st_tw	// L19
  float z0r[256];	// L20
  for (int v6 = 0; v6 < 256; v6++) {	// L21
    z0r[v6] = (float)0.000000;	// L21
  }
  float z0i[256];	// L22
  for (int v8 = 0; v8 < 256; v8++) {	// L23
    z0i[v8] = (float)0.000000;	// L23
  }
  float z1r[256];	// L24
  for (int v10 = 0; v10 < 256; v10++) {	// L25
    z1r[v10] = (float)0.000000;	// L25
  }
  float z1i[256];	// L26
  for (int v12 = 0; v12 < 256; v12++) {	// L27
    z1i[v12] = (float)0.000000;	// L27
  }
  l_S_t_0_t: for (int t = 0; t < 4744; t++) {	// L28
    float v14[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v14[_iv0] = _vec[_iv0];
      }
    }	// L29
    float v15[2];
    {
      hls::vector< float, 2 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v15[_iv0] = _vec[_iv0];
      }
    }	// L30
    int64_t v16 = t;	// L31
    int64_t v17 = v16 * 2;	// L32
    int64_t v18 = v17 & 511;	// L33
    int32_t v19 = v18;	// L34
    int32_t w0;	// L35
    w0 = v19;	// L36
    int32_t v21 = w0;	// L37
    int32_t v22 = v21 >> 7;	// L38
    int32_t v23 = v22 & 1;	// L39
    int32_t wsel;	// L40
    wsel = v23;	// L41
    int32_t v25 = w0;	// L42
    int32_t v26 = v25 >> 1;	// L43
    int32_t rw;	// L44
    rw = v26;	// L45
    ap_int<34> v28 = t;	// L46
    ap_int<34> v29 = v28 + 61;	// L47
    int32_t v30 = v29;	// L48
    int32_t u;	// L49
    u = v30;	// L50
    int32_t v32 = u;	// L51
    ap_int<33> v33 = v32;	// L52
    ap_int<33> v34 = v33 & 127;	// L53
    int32_t v35 = v34;	// L54
    int32_t j;	// L55
    j = v35;	// L56
    int32_t v37 = u;	// L57
    int32_t v38 = v37 >> 7;	// L58
    int32_t v39 = v38 & 1;	// L59
    int32_t blk;	// L60
    blk = v39;	// L61
    int32_t v41 = blk;	// L62
    int32_t v42 = v41 << 8;	// L63
    ap_int<33> v43 = v42;	// L64
    ap_int<33> v44 = v43 + 260;	// L65
    int32_t v45 = j;	// L66
    ap_int<34> v46 = v44;	// L67
    ap_int<34> v47 = v45;	// L68
    ap_int<34> v48 = v46 + v47;	// L69
    ap_int<34> v49 = v48 & 511;	// L70
    int32_t v50 = v49;	// L71
    int32_t pa;	// L72
    pa = v50;	// L73
    int32_t v52 = pa;	// L74
    ap_int<33> v53 = v52;	// L75
    ap_int<33> v54 = v53 + 128;	// L76
    ap_int<33> v55 = v54 & 511;	// L77
    int32_t v56 = v55;	// L78
    int32_t pb;	// L79
    pb = v56;	// L80
    int32_t v58 = pa;	// L81
    int32_t v59 = v58 & 1;	// L82
    int32_t v60 = v58 >> 7;	// L84
    int32_t v61 = v60 & 1;	// L85
    int32_t v62 = v59 ^ v61;	// L86
    int32_t sel;	// L87
    sel = v62;	// L88
    int32_t v64 = pa;	// L89
    int32_t v65 = v64 >> 1;	// L90
    int32_t ra;	// L91
    ra = v65;	// L92
    int32_t v67 = pb;	// L93
    int32_t v68 = v67 >> 1;	// L94
    int32_t rb;	// L95
    rb = v68;	// L96
    int32_t v70 = ra;	// L97
    int32_t r0;	// L98
    r0 = v70;	// L99
    int32_t v72 = rb;	// L100
    int32_t r1;	// L101
    r1 = v72;	// L102
    int32_t v74 = sel;	// L103
    bool v75 = v74 == 1;	// L104
    if (v75) {	// L105
      int32_t v76 = rb;	// L106
      r0 = v76;	// L107
      int32_t v77 = ra;	// L108
      r1 = v77;	// L109
    }
    int32_t v78 = r0;	// L111
    int v79 = v78;	// L112
    float v80 = z0r[v79];	// L113
    float g0r;	// L114
    g0r = v80;	// L115
    int32_t v82 = r0;	// L116
    int v83 = v82;	// L117
    float v84 = z0i[v83];	// L118
    float g0i;	// L119
    g0i = v84;	// L120
    int32_t v86 = r1;	// L121
    int v87 = v86;	// L122
    float v88 = z1r[v87];	// L123
    float g1r;	// L124
    g1r = v88;	// L125
    int32_t v90 = r1;	// L126
    int v91 = v90;	// L127
    float v92 = z1i[v91];	// L128
    float g1i;	// L129
    g1i = v92;	// L130
    float v94 = g0r;	// L131
    float ar;	// L132
    ar = v94;	// L133
    float v96 = g0i;	// L134
    float ai;	// L135
    ai = v96;	// L136
    float v98 = g1r;	// L137
    float br;	// L138
    br = v98;	// L139
    float v100 = g1i;	// L140
    float bi;	// L141
    bi = v100;	// L142
    int32_t v102 = sel;	// L143
    bool v103 = v102 == 1;	// L144
    if (v103) {	// L145
      float v104 = g1r;	// L146
      ar = v104;	// L147
      float v105 = g1i;	// L148
      ai = v105;	// L149
      float v106 = g0r;	// L150
      br = v106;	// L151
      float v107 = g0i;	// L152
      bi = v107;	// L153
    }
    int32_t v108 = wsel;	// L155
    bool v109 = v108 == 0;	// L156
    if (v109) {	// L157
      float v110 = v14[0];	// L158
      int32_t v111 = rw;	// L159
      int v112 = v111;	// L160
      z0r[v112] = v110;	// L161
      float v113 = v14[1];	// L162
      int32_t v114 = rw;	// L163
      int v115 = v114;	// L164
      z0i[v115] = v113;	// L165
      float v116 = v15[0];	// L166
      int32_t v117 = rw;	// L167
      int v118 = v117;	// L168
      z1r[v118] = v116;	// L169
      float v119 = v15[1];	// L170
      int32_t v120 = rw;	// L171
      int v121 = v120;	// L172
      z1i[v121] = v119;	// L173
    } else {
      float v122 = v14[0];	// L175
      int32_t v123 = rw;	// L176
      int v124 = v123;	// L177
      z1r[v124] = v122;	// L178
      float v125 = v14[1];	// L179
      int32_t v126 = rw;	// L180
      int v127 = v126;	// L181
      z1i[v127] = v125;	// L182
      float v128 = v15[0];	// L183
      int32_t v129 = rw;	// L184
      int v130 = v129;	// L185
      z0r[v130] = v128;	// L186
      float v131 = v15[1];	// L187
      int32_t v132 = rw;	// L188
      int v133 = v132;	// L189
      z0i[v133] = v131;	// L190
    }
    int32_t v134 = j;	// L192
    int v135 = v134;	// L193
    float v136 = _st_tw[v135][0];	// L194
    float wr;	// L195
    wr = v136;	// L196
    int32_t v138 = j;	// L197
    int v139 = v138;	// L198
    float v140 = _st_tw[v139][1];	// L199
    float wi;	// L200
    wi = v140;	// L201
    float v142 = ar;	// L202
    float v143 = br;	// L203
    float v144 = v142 - v143;	// L204
    #pragma HLS bind_op variable=v144 op=fsub impl=fabric
    float dr;	// L205
    dr = v144;	// L206
    float v146 = ai;	// L207
    float v147 = bi;	// L208
    float v148 = v146 - v147;	// L209
    #pragma HLS bind_op variable=v148 op=fsub impl=fabric
    float di;	// L210
    di = v148;	// L211
    float p[2];	// L212
    for (int v151 = 0; v151 < 2; v151++) {	// L213
      p[v151] = (float)0.000000;	// L213
    }
    float q[2];	// L214
    for (int v153 = 0; v153 < 2; v153++) {	// L215
      q[v153] = (float)0.000000;	// L215
    }
    float v154 = ar;	// L216
    float v155 = br;	// L217
    float v156 = v154 + v155;	// L218
    #pragma HLS bind_op variable=v156 op=fadd impl=fabric
    p[0] = v156;	// L219
    float v157 = ai;	// L220
    float v158 = bi;	// L221
    float v159 = v157 + v158;	// L222
    #pragma HLS bind_op variable=v159 op=fadd impl=fabric
    p[1] = v159;	// L223
    float v160 = dr;	// L224
    float v161 = wr;	// L225
    float v162 = v160 * v161;	// L226
    float v163 = di;	// L227
    float v164 = wi;	// L228
    float v165 = v163 * v164;	// L229
    float v166 = v162 - v165;	// L230
    #pragma HLS bind_op variable=v166 op=fsub impl=fabric
    q[0] = v166;	// L231
    float v167 = dr;	// L232
    float v168 = wi;	// L233
    float v169 = v167 * v168;	// L234
    float v170 = di;	// L235
    float v171 = wr;	// L236
    float v172 = v170 * v171;	// L237
    float v173 = v169 + v172;	// L238
    #pragma HLS bind_op variable=v173 op=fadd impl=fabric
    q[1] = v173;	// L239
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v2.write(_vec);
    }	// L240
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L241
  }
}

/// This is top function.
void top(

) {	// L245
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v174;
  #pragma HLS stream variable=v174 depth=8	// L246
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v175;
  #pragma HLS stream variable=v175 depth=8	// L247
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v176;
  #pragma HLS stream variable=v176 depth=8	// L248
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v177;
  #pragma HLS stream variable=v177 depth=8	// L249
  bfly2_r0_0(v174, v176, v175, v177);	// L250
}


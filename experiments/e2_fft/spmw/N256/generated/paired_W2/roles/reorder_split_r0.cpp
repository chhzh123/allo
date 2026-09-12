
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
int32_t _st_rd[128] = {0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120, 4, 68, 36, 100, 20, 84, 52, 116, 12, 76, 44, 108, 28, 92, 60, 124, 2, 66, 34, 98, 18, 82, 50, 114, 10, 74, 42, 106, 26, 90, 58, 122, 6, 70, 38, 102, 22, 86, 54, 118, 14, 78, 46, 110, 30, 94, 62, 126, 1, 65, 33, 97, 17, 81, 49, 113, 9, 73, 41, 105, 25, 89, 57, 121, 5, 69, 37, 101, 21, 85, 53, 117, 13, 77, 45, 109, 29, 93, 61, 125, 3, 67, 35, 99, 19, 83, 51, 115, 11, 75, 43, 107, 27, 91, 59, 123, 7, 71, 39, 103, 23, 87, 55, 119, 15, 79, 47, 111, 31, 95, 63, 127};	// L2
void reorder_split_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L3
  // placeholder for const int32_t _st_rd	// L18
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
    ap_int<34> v29 = v28 + 120;	// L46
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
    ap_int<33> v43 = v42;	// L63
    ap_int<33> v44 = v43 + 16;	// L64
    int32_t v45 = j;	// L65
    int v46 = v45;	// L66
    int32_t v47 = _st_rd[v46];	// L67
    ap_int<34> v48 = v44;	// L68
    ap_int<34> v49 = v47;	// L69
    ap_int<34> v50 = v48 + v49;	// L70
    ap_int<34> v51 = v50 & 511;	// L71
    int32_t v52 = v51;	// L72
    int32_t pa;	// L73
    pa = v52;	// L74
    int32_t v54 = pa;	// L75
    ap_int<33> v55 = v54;	// L76
    ap_int<33> v56 = v55 + 128;	// L77
    ap_int<33> v57 = v56 & 511;	// L78
    int32_t v58 = v57;	// L79
    int32_t pb;	// L80
    pb = v58;	// L81
    int32_t v60 = pa;	// L82
    int32_t v61 = v60 & 1;	// L83
    int32_t v62 = v60 >> 7;	// L85
    int32_t v63 = v62 & 1;	// L86
    int32_t v64 = v61 ^ v63;	// L87
    int32_t sel;	// L88
    sel = v64;	// L89
    int32_t v66 = pa;	// L90
    int32_t v67 = v66 >> 1;	// L91
    int32_t ra;	// L92
    ra = v67;	// L93
    int32_t v69 = pb;	// L94
    int32_t v70 = v69 >> 1;	// L95
    int32_t rb;	// L96
    rb = v70;	// L97
    int32_t v72 = ra;	// L98
    int32_t r0;	// L99
    r0 = v72;	// L100
    int32_t v74 = rb;	// L101
    int32_t r1;	// L102
    r1 = v74;	// L103
    int32_t v76 = sel;	// L104
    bool v77 = v76 == 1;	// L105
    if (v77) {	// L106
      int32_t v78 = rb;	// L107
      r0 = v78;	// L108
      int32_t v79 = ra;	// L109
      r1 = v79;	// L110
    }
    int32_t v80 = r0;	// L112
    int v81 = v80;	// L113
    float v82 = z0r[v81];	// L114
    float g0r;	// L115
    g0r = v82;	// L116
    int32_t v84 = r0;	// L117
    int v85 = v84;	// L118
    float v86 = z0i[v85];	// L119
    float g0i;	// L120
    g0i = v86;	// L121
    int32_t v88 = r1;	// L122
    int v89 = v88;	// L123
    float v90 = z1r[v89];	// L124
    float g1r;	// L125
    g1r = v90;	// L126
    int32_t v92 = r1;	// L127
    int v93 = v92;	// L128
    float v94 = z1i[v93];	// L129
    float g1i;	// L130
    g1i = v94;	// L131
    float p[2];	// L132
    for (int v97 = 0; v97 < 2; v97++) {	// L133
      p[v97] = (float)0.000000;	// L133
    }
    float q[2];	// L134
    for (int v99 = 0; v99 < 2; v99++) {	// L135
      q[v99] = (float)0.000000;	// L135
    }
    float v100 = g0r;	// L136
    p[0] = v100;	// L137
    float v101 = g0i;	// L138
    p[1] = v101;	// L139
    float v102 = g1r;	// L140
    q[0] = v102;	// L141
    float v103 = g1i;	// L142
    q[1] = v103;	// L143
    int32_t v104 = sel;	// L144
    bool v105 = v104 == 1;	// L145
    if (v105) {	// L146
      float v106 = g1r;	// L147
      p[0] = v106;	// L148
      float v107 = g1i;	// L149
      p[1] = v107;	// L150
      float v108 = g0r;	// L151
      q[0] = v108;	// L152
      float v109 = g0i;	// L153
      q[1] = v109;	// L154
    }
    int32_t v110 = wsel;	// L156
    bool v111 = v110 == 0;	// L157
    if (v111) {	// L158
      float v112 = v14[0];	// L159
      int32_t v113 = rw;	// L160
      int v114 = v113;	// L161
      z0r[v114] = v112;	// L162
      float v115 = v14[1];	// L163
      int32_t v116 = rw;	// L164
      int v117 = v116;	// L165
      z0i[v117] = v115;	// L166
      float v118 = v15[0];	// L167
      int32_t v119 = rw;	// L168
      int v120 = v119;	// L169
      z1r[v120] = v118;	// L170
      float v121 = v15[1];	// L171
      int32_t v122 = rw;	// L172
      int v123 = v122;	// L173
      z1i[v123] = v121;	// L174
    } else {
      float v124 = v14[0];	// L176
      int32_t v125 = rw;	// L177
      int v126 = v125;	// L178
      z1r[v126] = v124;	// L179
      float v127 = v14[1];	// L180
      int32_t v128 = rw;	// L181
      int v129 = v128;	// L182
      z1i[v129] = v127;	// L183
      float v130 = v15[0];	// L184
      int32_t v131 = rw;	// L185
      int v132 = v131;	// L186
      z0r[v132] = v130;	// L187
      float v133 = v15[1];	// L188
      int32_t v134 = rw;	// L189
      int v135 = v134;	// L190
      z0i[v135] = v133;	// L191
    }
    ap_int<33> v136 = t;	// L193
    bool v137 = v136 >= 648;	// L194
    if (v137) {	// L195
      {
        hls::vector< float, 2 > _vec;
        for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
          _vec[_iv0] = p[_iv0];
        }
        v2.write(_vec);
      }	// L196
      {
        hls::vector< float, 2 > _vec;
        for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
          _vec[_iv0] = q[_iv0];
        }
        v3.write(_vec);
      }	// L197
    }
  }
}

/// This is top function.
void top(

) {	// L202
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v138;
  #pragma HLS stream variable=v138 depth=8	// L203
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v139;
  #pragma HLS stream variable=v139 depth=8	// L204
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v140;
  #pragma HLS stream variable=v140 depth=8	// L205
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v141;
  #pragma HLS stream variable=v141 depth=8	// L206
  reorder_split_r0_0(v138, v140, v139, v141);	// L207
}


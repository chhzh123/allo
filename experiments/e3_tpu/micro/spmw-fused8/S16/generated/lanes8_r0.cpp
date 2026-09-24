
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
void lanes8_r0_0(
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
  hls::stream< int32_t >& v16
) {	// L2
  int32_t v17[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v17[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v18 = v17[1];	// L4
  int32_t v19 = v18 & 31;	// L6
  int32_t sh;	// L7
  sh = v19;	// L8
  l_S__m_0__m: for (int _m = 0; _m < 256; _m++) {	// L9
  #pragma HLS pipeline II=1
    int32_t v22 = v9.read();	// L10
    int32_t y0;	// L11
    y0 = v22;	// L12
    int32_t v24 = y0;	// L13
    bool v25 = v24 < 0;	// L15
    if (v25) {	// L16
      y0 = 0;	// L17
    }
    int32_t v26 = y0;	// L19
    int32_t v27 = sh;	// L20
    int32_t v28 = v26 >> v27;	// L21
    y0 = v28;	// L22
    int32_t v29 = y0;	// L23
    bool v30 = v29 > 127;	// L25
    if (v30) {	// L26
      y0 = 127;	// L27
    }
    int32_t v31 = y0;	// L29
    v1.write(v31);	// L30
    int32_t v32 = v10.read();	// L31
    int32_t y1;	// L32
    y1 = v32;	// L33
    int32_t v34 = y1;	// L34
    bool v35 = v34 < 0;	// L35
    if (v35) {	// L36
      y1 = 0;	// L37
    }
    int32_t v36 = y1;	// L39
    int32_t v37 = sh;	// L40
    int32_t v38 = v36 >> v37;	// L41
    y1 = v38;	// L42
    int32_t v39 = y1;	// L43
    bool v40 = v39 > 127;	// L44
    if (v40) {	// L45
      y1 = 127;	// L46
    }
    int32_t v41 = y1;	// L48
    v2.write(v41);	// L49
    int32_t v42 = v11.read();	// L50
    int32_t y2;	// L51
    y2 = v42;	// L52
    int32_t v44 = y2;	// L53
    bool v45 = v44 < 0;	// L54
    if (v45) {	// L55
      y2 = 0;	// L56
    }
    int32_t v46 = y2;	// L58
    int32_t v47 = sh;	// L59
    int32_t v48 = v46 >> v47;	// L60
    y2 = v48;	// L61
    int32_t v49 = y2;	// L62
    bool v50 = v49 > 127;	// L63
    if (v50) {	// L64
      y2 = 127;	// L65
    }
    int32_t v51 = y2;	// L67
    v3.write(v51);	// L68
    int32_t v52 = v12.read();	// L69
    int32_t y3;	// L70
    y3 = v52;	// L71
    int32_t v54 = y3;	// L72
    bool v55 = v54 < 0;	// L73
    if (v55) {	// L74
      y3 = 0;	// L75
    }
    int32_t v56 = y3;	// L77
    int32_t v57 = sh;	// L78
    int32_t v58 = v56 >> v57;	// L79
    y3 = v58;	// L80
    int32_t v59 = y3;	// L81
    bool v60 = v59 > 127;	// L82
    if (v60) {	// L83
      y3 = 127;	// L84
    }
    int32_t v61 = y3;	// L86
    v4.write(v61);	// L87
    int32_t v62 = v13.read();	// L88
    int32_t y4;	// L89
    y4 = v62;	// L90
    int32_t v64 = y4;	// L91
    bool v65 = v64 < 0;	// L92
    if (v65) {	// L93
      y4 = 0;	// L94
    }
    int32_t v66 = y4;	// L96
    int32_t v67 = sh;	// L97
    int32_t v68 = v66 >> v67;	// L98
    y4 = v68;	// L99
    int32_t v69 = y4;	// L100
    bool v70 = v69 > 127;	// L101
    if (v70) {	// L102
      y4 = 127;	// L103
    }
    int32_t v71 = y4;	// L105
    v5.write(v71);	// L106
    int32_t v72 = v14.read();	// L107
    int32_t y5;	// L108
    y5 = v72;	// L109
    int32_t v74 = y5;	// L110
    bool v75 = v74 < 0;	// L111
    if (v75) {	// L112
      y5 = 0;	// L113
    }
    int32_t v76 = y5;	// L115
    int32_t v77 = sh;	// L116
    int32_t v78 = v76 >> v77;	// L117
    y5 = v78;	// L118
    int32_t v79 = y5;	// L119
    bool v80 = v79 > 127;	// L120
    if (v80) {	// L121
      y5 = 127;	// L122
    }
    int32_t v81 = y5;	// L124
    v6.write(v81);	// L125
    int32_t v82 = v15.read();	// L126
    int32_t y6;	// L127
    y6 = v82;	// L128
    int32_t v84 = y6;	// L129
    bool v85 = v84 < 0;	// L130
    if (v85) {	// L131
      y6 = 0;	// L132
    }
    int32_t v86 = y6;	// L134
    int32_t v87 = sh;	// L135
    int32_t v88 = v86 >> v87;	// L136
    y6 = v88;	// L137
    int32_t v89 = y6;	// L138
    bool v90 = v89 > 127;	// L139
    if (v90) {	// L140
      y6 = 127;	// L141
    }
    int32_t v91 = y6;	// L143
    v7.write(v91);	// L144
    int32_t v92 = v16.read();	// L145
    int32_t y7;	// L146
    y7 = v92;	// L147
    int32_t v94 = y7;	// L148
    bool v95 = v94 < 0;	// L149
    if (v95) {	// L150
      y7 = 0;	// L151
    }
    int32_t v96 = y7;	// L153
    int32_t v97 = sh;	// L154
    int32_t v98 = v96 >> v97;	// L155
    y7 = v98;	// L156
    int32_t v99 = y7;	// L157
    bool v100 = v99 > 127;	// L158
    if (v100) {	// L159
      y7 = 127;	// L160
    }
    int32_t v101 = y7;	// L162
    v8.write(v101);	// L163
  }
}


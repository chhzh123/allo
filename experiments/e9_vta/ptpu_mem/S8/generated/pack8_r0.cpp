
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
void pack8_r0_0(
  hls::stream< int8_t >& v0,
  hls::stream< hls::vector< int8_t, 9 > >& v1,
  hls::stream< int64_t >& v2
) {	// L2
  bool go;	// L6
  go = 1;	// L7
  while (true) {	// L8
    #pragma HLS pipeline II=1 style=flp
    bool v4 = go;	// L9
    if (!(v4)) break;
    int8_t v5[9];
    {
      hls::vector< int8_t, 9 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 9; ++_iv0) {
        v5[_iv0] = _vec[_iv0];
      }
    }	// L16
    v0.write(1);	// L19
    int8_t v6 = v5[0];	// L20
    int32_t v7 = v6;	// L21
    int32_t v8 = v7 & 255;	// L24
    int64_t v9 = v8;	// L25
    int64_t w0_0;	// L26
    w0_0 = v9;	// L27
    int8_t v11 = v5[1];	// L28
    int32_t v12 = v11;	// L29
    int32_t v13 = v12 & 255;	// L32
    int64_t v14 = v13;	// L33
    int64_t w0_1;	// L34
    w0_1 = v14;	// L35
    int8_t v16 = v5[2];	// L36
    int32_t v17 = v16;	// L37
    int32_t v18 = v17 & 255;	// L40
    int64_t v19 = v18;	// L41
    int64_t w0_2;	// L42
    w0_2 = v19;	// L43
    int8_t v21 = v5[3];	// L44
    int32_t v22 = v21;	// L45
    int32_t v23 = v22 & 255;	// L48
    int64_t v24 = v23;	// L49
    int64_t w0_3;	// L50
    w0_3 = v24;	// L51
    int8_t v26 = v5[4];	// L52
    int32_t v27 = v26;	// L53
    int32_t v28 = v27 & 255;	// L56
    int64_t v29 = v28;	// L57
    int64_t w0_4;	// L58
    w0_4 = v29;	// L59
    int8_t v31 = v5[5];	// L60
    int32_t v32 = v31;	// L61
    int32_t v33 = v32 & 255;	// L64
    int64_t v34 = v33;	// L65
    int64_t w0_5;	// L66
    w0_5 = v34;	// L67
    int8_t v36 = v5[6];	// L68
    int32_t v37 = v36;	// L69
    int32_t v38 = v37 & 255;	// L72
    int64_t v39 = v38;	// L73
    int64_t w0_6;	// L74
    w0_6 = v39;	// L75
    int8_t v41 = v5[7];	// L76
    int32_t v42 = v41;	// L77
    int32_t v43 = v42 & 255;	// L80
    int64_t v44 = v43;	// L81
    int64_t w0_7;	// L82
    w0_7 = v44;	// L83
    int64_t w0;	// L87
    w0 = 0;	// L88
    int64_t v47 = w0;	// L89
    int64_t v48 = w0_0;	// L90
    int64_t v49 = v47 | v48;	// L95
    w0 = v49;	// L96
    int64_t v50 = w0;	// L97
    int64_t v51 = w0_1;	// L98
    int64_t v52 = v51 << 8;	// L102
    int64_t v53 = v50 | v52;	// L103
    w0 = v53;	// L104
    int64_t v54 = w0;	// L105
    int64_t v55 = w0_2;	// L106
    int64_t v56 = v55 << 16;	// L110
    int64_t v57 = v54 | v56;	// L111
    w0 = v57;	// L112
    int64_t v58 = w0;	// L113
    int64_t v59 = w0_3;	// L114
    int64_t v60 = v59 << 24;	// L118
    int64_t v61 = v58 | v60;	// L119
    w0 = v61;	// L120
    int64_t v62 = w0;	// L121
    int64_t v63 = w0_4;	// L122
    int64_t v64 = v63 << 32;	// L126
    int64_t v65 = v62 | v64;	// L127
    w0 = v65;	// L128
    int64_t v66 = w0;	// L129
    int64_t v67 = w0_5;	// L130
    int64_t v68 = v67 << 40;	// L134
    int64_t v69 = v66 | v68;	// L135
    w0 = v69;	// L136
    int64_t v70 = w0;	// L137
    int64_t v71 = w0_6;	// L138
    int64_t v72 = v71 << 48;	// L142
    int64_t v73 = v70 | v72;	// L143
    w0 = v73;	// L144
    int64_t v74 = w0;	// L145
    int64_t v75 = w0_7;	// L146
    int64_t v76 = v75 << 56;	// L150
    int64_t v77 = v74 | v76;	// L151
    w0 = v77;	// L152
    int64_t v78 = w0;	// L153
    v2.write(v78);	// L154
    int8_t v79 = v5[8];	// L155
    int32_t v80 = v79;	// L156
    int32_t fl;	// L157
    fl = v80;	// L158
    int32_t v82 = fl;	// L159
    bool v83 = v82 != 0;	// L162
    if (v83) {	// L163
      go = 0;	// L167
    }
  }
}


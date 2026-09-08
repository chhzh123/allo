
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
using namespace std;

extern "C" {

void load_buf0(
  float v0[1024],
  float v1[1024]
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 1024; load_buf0_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    float v3 = v0[load_buf0_l_0];	//
    v1[load_buf0_l_0] = v3;	//
  }
}

void load_buf1(
  float v4[1024],
  float v5[1024]
) {	//
  l_S_load_buf1_load_buf1_l_0: for (int load_buf1_l_0 = 0; load_buf1_l_0 < 1024; load_buf1_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    float v7 = v4[load_buf1_l_0];	//
    v5[load_buf1_l_0] = v7;	//
  }
}

void load_buf2(
  float v8[512],
  float v9[512]
) {	//
  l_S_load_buf2_load_buf2_l_0: for (int load_buf2_l_0 = 0; load_buf2_l_0 < 512; load_buf2_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    float v11 = v8[load_buf2_l_0];	//
    v9[load_buf2_l_0] = v11;	//
  }
}

void load_buf3(
  float v12[512],
  float v13[512]
) {	//
  l_S_load_buf3_load_buf3_l_0: for (int load_buf3_l_0 = 0; load_buf3_l_0 < 512; load_buf3_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    float v15 = v12[load_buf3_l_0];	//
    v13[load_buf3_l_0] = v15;	//
  }
}

void store_res0(
  float v16[1024],
  float v17[1024]
) {	//
  l_S_store_res0_store_res0_l_0: for (int store_res0_l_0 = 0; store_res0_l_0 < 1024; store_res0_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    float v19 = v16[store_res0_l_0];	//
    v17[store_res0_l_0] = v19;	//
  }
}

void store_res1(
  float v20[1024],
  float v21[1024]
) {	//
  l_S_store_res1_store_res1_l_0: for (int store_res1_l_0 = 0; store_res1_l_0 < 1024; store_res1_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    float v23 = v20[store_res1_l_0];	//
    v21[store_res1_l_0] = v23;	//
  }
}

/// This is top function.
void fft(
  float *v24,
  float *v25,
  float *v26,
  float *v27
) {	// L2
  #pragma HLS interface m_axi port=v24 offset=slave bundle=gmem0 depth=1024
  #pragma HLS interface m_axi port=v25 offset=slave bundle=gmem1 depth=1024
  #pragma HLS interface m_axi port=v26 offset=slave bundle=gmem2 depth=512
  #pragma HLS interface m_axi port=v27 offset=slave bundle=gmem3 depth=512
  float buf0[1024];	//
  load_buf0(v24, buf0);	//
  float buf1[1024];	//
  load_buf1(v25, buf1);	//
  float buf2[512];	//
  load_buf2(v26, buf2);	//
  float buf3[512];	//
  load_buf3(v27, buf3);	//
  int32_t span;	// L8
  span = 512;	// L9
  int32_t log;	// L12
  log = 0;	// L13
  int32_t even;	// L16
  even = 0;	// L17
  int32_t odd;	// L20
  odd = 0;	// L21
  int32_t rootindex;	// L24
  rootindex = 0;	// L25
  float temp;	// L28
  temp = (float)0.000000;	// L29
  while (true) {	// L30
    int32_t v38 = span;	// L31
    bool v39 = v38 > 0;	// L34
    if (!(v39)) break;
    int32_t v40 = span;	// L37
    odd = v40;	// L38
    while (true) {	// L39
      int32_t v41 = odd;	// L40
      bool v42 = v41 < 1024;	// L43
      if (!(v42)) break;
      int32_t v43 = span;	// L46
      int32_t v44 = odd;	// L47
      int32_t v45 = v44 | v43;	// L48
      odd = v45;	// L49
      int32_t v46 = odd;	// L50
      int32_t v47 = span;	// L51
      int32_t v48 = v46 ^ v47;	// L52
      even = v48;	// L53
      int32_t v49 = even;	// L54
      int v50 = v49;	// L55
      float v51 = buf0[v50];	// L56
      int32_t v52 = odd;	// L57
      int v53 = v52;	// L58
      float v54 = buf0[v53];	// L59
      float v55 = v51 + v54;	// L60
      temp = v55;	// L61
      int32_t v56 = even;	// L62
      int v57 = v56;	// L63
      float v58 = buf0[v57];	// L64
      int32_t v59 = odd;	// L65
      int v60 = v59;	// L66
      float v61 = buf0[v60];	// L67
      float v62 = v58 - v61;	// L68
      buf0[v60] = v62;	// L71
      float v63 = temp;	// L72
      int32_t v64 = even;	// L73
      int v65 = v64;	// L74
      buf0[v65] = v63;	// L75
      int32_t v66 = even;	// L76
      int v67 = v66;	// L77
      float v68 = buf1[v67];	// L78
      int32_t v69 = odd;	// L79
      int v70 = v69;	// L80
      float v71 = buf1[v70];	// L81
      float v72 = v68 + v71;	// L82
      temp = v72;	// L83
      int32_t v73 = even;	// L84
      int v74 = v73;	// L85
      float v75 = buf1[v74];	// L86
      int32_t v76 = odd;	// L87
      int v77 = v76;	// L88
      float v78 = buf1[v77];	// L89
      float v79 = v75 - v78;	// L90
      buf1[v77] = v79;	// L93
      float v80 = temp;	// L94
      int32_t v81 = even;	// L95
      int v82 = v81;	// L96
      buf1[v82] = v80;	// L97
      int32_t v83 = even;	// L98
      int32_t v84 = log;	// L99
      int32_t v85 = v83 << v84;	// L100
      ap_int<33> v86 = v85;	// L108
      ap_int<33> v87 = v86 & 1023;	// L109
      int32_t v88 = v87;	// L110
      rootindex = v88;	// L111
      int32_t v89 = rootindex;	// L112
      bool v90 = v89 > 0;	// L115
      if (v90) {	// L116
        int32_t v91 = rootindex;	// L117
        int v92 = v91;	// L118
        float v93 = buf2[v92];	// L119
        int32_t v94 = odd;	// L120
        int v95 = v94;	// L121
        float v96 = buf0[v95];	// L122
        float v97 = v93 * v96;	// L123
        float v98 = buf3[v92];	// L126
        float v99 = buf1[v95];	// L129
        float v100 = v98 * v99;	// L130
        float v101 = v97 - v100;	// L131
        temp = v101;	// L132
        int32_t v102 = rootindex;	// L133
        int v103 = v102;	// L134
        float v104 = buf2[v103];	// L135
        int32_t v105 = odd;	// L136
        int v106 = v105;	// L137
        float v107 = buf1[v106];	// L138
        float v108 = v104 * v107;	// L139
        float v109 = buf3[v103];	// L142
        float v110 = buf0[v106];	// L145
        float v111 = v109 * v110;	// L146
        float v112 = v108 + v111;	// L147
        buf1[v106] = v112;	// L150
        float v113 = temp;	// L151
        int32_t v114 = odd;	// L152
        int v115 = v114;	// L153
        buf0[v115] = v113;	// L154
      }
      int32_t v116 = odd;	// L156
      ap_int<33> v117 = v116;	// L157
      ap_int<33> v118 = v117 + 1;	// L161
      int32_t v119 = v118;	// L162
      odd = v119;	// L163
    }
    int32_t v120 = span;	// L166
    int32_t v121 = v120 >> 1;	// L169
    span = v121;	// L170
    int32_t v122 = log;	// L171
    ap_int<33> v123 = v122;	// L172
    ap_int<33> v124 = v123 + 1;	// L176
    int32_t v125 = v124;	// L177
    log = v125;	// L178
  }
  store_res0(buf0, v24);	//
  store_res1(buf1, v25);	//
}


} // extern "C"

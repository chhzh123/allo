
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

void offchip_loadA_0(
  ap_uint<128> v0[256],
  hls::stream< ap_uint<128> >& v1
) {	// L2
  l_S_mt_nt_0_mt: for (int mt = 0; mt < 4; mt++) {	// L3
    l_nt: for (int nt = 0; nt < 4; nt++) {	// L4
      l_S_k_0_k: for (int k = 0; k < 64; k++) {	// L5
        ap_uint<128> v5 = v0[((mt * 64) + k)];	// L6
        v1.write(v5);	// L7
      }
    }
  }
}

void offchip_loadB_0(
  ap_uint<128> v6[256],
  hls::stream< ap_uint<128> >& v7
) {	// L13
  l_S_mt_nt_0_mt1: for (int mt1 = 0; mt1 < 4; mt1++) {	// L14
    l_nt1: for (int nt1 = 0; nt1 < 4; nt1++) {	// L15
      l_S_k_0_k1: for (int k1 = 0; k1 < 64; k1++) {	// L16
        ap_uint<128> v11 = v6[((nt1 * 64) + k1)];	// L17
        v7.write(v11);	// L18
      }
    }
  }
}

void gemm_0_0(
  hls::stream< ap_uint<128> >& v12,
  hls::stream< ap_uint<128> >& v13,
  hls::stream< ap_uint<128> >& v14,
  hls::stream< ap_uint<128> >& v15
) {	// L24
  l_S_mt_nt_0_mt2: for (int mt2 = 0; mt2 < 4; mt2++) {	// L25
    l_nt2: for (int nt2 = 0; nt2 < 4; nt2++) {	// L26
      l_S_k_0_k2: for (int k2 = 0; k2 < 64; k2++) {	// L27
        ap_uint<128> v19 = v12.read();	// L28
        v13.write(v19);	// L29
        ap_uint<128> v20 = v14.read();	// L30
        v15.write(v20);	// L31
      }
    }
  }
}

void gemm_0_1(
  hls::stream< ap_uint<128> >& v21,
  hls::stream< int8_t >& v22,
  hls::stream< ap_uint<128> >& v23
) {	// L37
  l_S_mt_nt_0_mt3: for (int mt3 = 0; mt3 < 4; mt3++) {	// L40
    l_nt3: for (int nt3 = 0; nt3 < 4; nt3++) {	// L41
      l_S_k_0_k3: for (int k3 = 0; k3 < 64; k3++) {	// L42
        ap_uint<128> v27 = v21.read();	// L43
        ap_uint<128> b;	// L44
        b = v27;	// L45
        ap_int<128> v29 = b;	// L46
        int8_t v30;
        ap_int<128> v30_tmp = v29;
        v30 = v30_tmp(7, 0);	// L47
        v22.write(v30);	// L48
        ap_int<128> v31 = b;	// L49
        v23.write(v31);	// L50
      }
    }
  }
}

void gemm_0_2(
  hls::stream< ap_uint<128> >& v32,
  hls::stream< int8_t >& v33,
  hls::stream< ap_uint<128> >& v34
) {	// L56
  l_S_mt_nt_0_mt4: for (int mt4 = 0; mt4 < 4; mt4++) {	// L59
    l_nt4: for (int nt4 = 0; nt4 < 4; nt4++) {	// L60
      l_S_k_0_k4: for (int k4 = 0; k4 < 64; k4++) {	// L61
        ap_uint<128> v38 = v32.read();	// L62
        ap_uint<128> b1;	// L63
        b1 = v38;	// L64
        ap_int<128> v40 = b1;	// L65
        int8_t v41;
        ap_int<128> v41_tmp = v40;
        v41 = v41_tmp(15, 8);	// L66
        v33.write(v41);	// L67
        ap_int<128> v42 = b1;	// L68
        v34.write(v42);	// L69
      }
    }
  }
}

void gemm_0_3(
  hls::stream< ap_uint<128> >& v43,
  hls::stream< int8_t >& v44,
  hls::stream< ap_uint<128> >& v45
) {	// L75
  l_S_mt_nt_0_mt5: for (int mt5 = 0; mt5 < 4; mt5++) {	// L78
    l_nt5: for (int nt5 = 0; nt5 < 4; nt5++) {	// L79
      l_S_k_0_k5: for (int k5 = 0; k5 < 64; k5++) {	// L80
        ap_uint<128> v49 = v43.read();	// L81
        ap_uint<128> b2;	// L82
        b2 = v49;	// L83
        ap_int<128> v51 = b2;	// L84
        int8_t v52;
        ap_int<128> v52_tmp = v51;
        v52 = v52_tmp(23, 16);	// L85
        v44.write(v52);	// L86
        ap_int<128> v53 = b2;	// L87
        v45.write(v53);	// L88
      }
    }
  }
}

void gemm_0_4(
  hls::stream< ap_uint<128> >& v54,
  hls::stream< int8_t >& v55,
  hls::stream< ap_uint<128> >& v56
) {	// L94
  l_S_mt_nt_0_mt6: for (int mt6 = 0; mt6 < 4; mt6++) {	// L97
    l_nt6: for (int nt6 = 0; nt6 < 4; nt6++) {	// L98
      l_S_k_0_k6: for (int k6 = 0; k6 < 64; k6++) {	// L99
        ap_uint<128> v60 = v54.read();	// L100
        ap_uint<128> b3;	// L101
        b3 = v60;	// L102
        ap_int<128> v62 = b3;	// L103
        int8_t v63;
        ap_int<128> v63_tmp = v62;
        v63 = v63_tmp(31, 24);	// L104
        v55.write(v63);	// L105
        ap_int<128> v64 = b3;	// L106
        v56.write(v64);	// L107
      }
    }
  }
}

void gemm_0_5(
  hls::stream< ap_uint<128> >& v65,
  hls::stream< int8_t >& v66,
  hls::stream< ap_uint<128> >& v67
) {	// L113
  l_S_mt_nt_0_mt7: for (int mt7 = 0; mt7 < 4; mt7++) {	// L116
    l_nt7: for (int nt7 = 0; nt7 < 4; nt7++) {	// L117
      l_S_k_0_k7: for (int k7 = 0; k7 < 64; k7++) {	// L118
        ap_uint<128> v71 = v65.read();	// L119
        ap_uint<128> b4;	// L120
        b4 = v71;	// L121
        ap_int<128> v73 = b4;	// L122
        int8_t v74;
        ap_int<128> v74_tmp = v73;
        v74 = v74_tmp(39, 32);	// L123
        v66.write(v74);	// L124
        ap_int<128> v75 = b4;	// L125
        v67.write(v75);	// L126
      }
    }
  }
}

void gemm_0_6(
  hls::stream< ap_uint<128> >& v76,
  hls::stream< int8_t >& v77,
  hls::stream< ap_uint<128> >& v78
) {	// L132
  l_S_mt_nt_0_mt8: for (int mt8 = 0; mt8 < 4; mt8++) {	// L135
    l_nt8: for (int nt8 = 0; nt8 < 4; nt8++) {	// L136
      l_S_k_0_k8: for (int k8 = 0; k8 < 64; k8++) {	// L137
        ap_uint<128> v82 = v76.read();	// L138
        ap_uint<128> b5;	// L139
        b5 = v82;	// L140
        ap_int<128> v84 = b5;	// L141
        int8_t v85;
        ap_int<128> v85_tmp = v84;
        v85 = v85_tmp(47, 40);	// L142
        v77.write(v85);	// L143
        ap_int<128> v86 = b5;	// L144
        v78.write(v86);	// L145
      }
    }
  }
}

void gemm_0_7(
  hls::stream< ap_uint<128> >& v87,
  hls::stream< int8_t >& v88,
  hls::stream< ap_uint<128> >& v89
) {	// L151
  l_S_mt_nt_0_mt9: for (int mt9 = 0; mt9 < 4; mt9++) {	// L154
    l_nt9: for (int nt9 = 0; nt9 < 4; nt9++) {	// L155
      l_S_k_0_k9: for (int k9 = 0; k9 < 64; k9++) {	// L156
        ap_uint<128> v93 = v87.read();	// L157
        ap_uint<128> b6;	// L158
        b6 = v93;	// L159
        ap_int<128> v95 = b6;	// L160
        int8_t v96;
        ap_int<128> v96_tmp = v95;
        v96 = v96_tmp(55, 48);	// L161
        v88.write(v96);	// L162
        ap_int<128> v97 = b6;	// L163
        v89.write(v97);	// L164
      }
    }
  }
}

void gemm_0_8(
  hls::stream< ap_uint<128> >& v98,
  hls::stream< int8_t >& v99,
  hls::stream< ap_uint<128> >& v100
) {	// L170
  l_S_mt_nt_0_mt10: for (int mt10 = 0; mt10 < 4; mt10++) {	// L173
    l_nt10: for (int nt10 = 0; nt10 < 4; nt10++) {	// L174
      l_S_k_0_k10: for (int k10 = 0; k10 < 64; k10++) {	// L175
        ap_uint<128> v104 = v98.read();	// L176
        ap_uint<128> b7;	// L177
        b7 = v104;	// L178
        ap_int<128> v106 = b7;	// L179
        int8_t v107;
        ap_int<128> v107_tmp = v106;
        v107 = v107_tmp(63, 56);	// L180
        v99.write(v107);	// L181
        ap_int<128> v108 = b7;	// L182
        v100.write(v108);	// L183
      }
    }
  }
}

void gemm_0_9(
  hls::stream< ap_uint<128> >& v109,
  hls::stream< int8_t >& v110,
  hls::stream< ap_uint<128> >& v111
) {	// L189
  l_S_mt_nt_0_mt11: for (int mt11 = 0; mt11 < 4; mt11++) {	// L192
    l_nt11: for (int nt11 = 0; nt11 < 4; nt11++) {	// L193
      l_S_k_0_k11: for (int k11 = 0; k11 < 64; k11++) {	// L194
        ap_uint<128> v115 = v109.read();	// L195
        ap_uint<128> b8;	// L196
        b8 = v115;	// L197
        ap_int<128> v117 = b8;	// L198
        int8_t v118;
        ap_int<128> v118_tmp = v117;
        v118 = v118_tmp(71, 64);	// L199
        v110.write(v118);	// L200
        ap_int<128> v119 = b8;	// L201
        v111.write(v119);	// L202
      }
    }
  }
}

void gemm_0_10(
  hls::stream< ap_uint<128> >& v120,
  hls::stream< int8_t >& v121,
  hls::stream< ap_uint<128> >& v122
) {	// L208
  l_S_mt_nt_0_mt12: for (int mt12 = 0; mt12 < 4; mt12++) {	// L211
    l_nt12: for (int nt12 = 0; nt12 < 4; nt12++) {	// L212
      l_S_k_0_k12: for (int k12 = 0; k12 < 64; k12++) {	// L213
        ap_uint<128> v126 = v120.read();	// L214
        ap_uint<128> b9;	// L215
        b9 = v126;	// L216
        ap_int<128> v128 = b9;	// L217
        int8_t v129;
        ap_int<128> v129_tmp = v128;
        v129 = v129_tmp(79, 72);	// L218
        v121.write(v129);	// L219
        ap_int<128> v130 = b9;	// L220
        v122.write(v130);	// L221
      }
    }
  }
}

void gemm_0_11(
  hls::stream< ap_uint<128> >& v131,
  hls::stream< int8_t >& v132,
  hls::stream< ap_uint<128> >& v133
) {	// L227
  l_S_mt_nt_0_mt13: for (int mt13 = 0; mt13 < 4; mt13++) {	// L230
    l_nt13: for (int nt13 = 0; nt13 < 4; nt13++) {	// L231
      l_S_k_0_k13: for (int k13 = 0; k13 < 64; k13++) {	// L232
        ap_uint<128> v137 = v131.read();	// L233
        ap_uint<128> b10;	// L234
        b10 = v137;	// L235
        ap_int<128> v139 = b10;	// L236
        int8_t v140;
        ap_int<128> v140_tmp = v139;
        v140 = v140_tmp(87, 80);	// L237
        v132.write(v140);	// L238
        ap_int<128> v141 = b10;	// L239
        v133.write(v141);	// L240
      }
    }
  }
}

void gemm_0_12(
  hls::stream< ap_uint<128> >& v142,
  hls::stream< int8_t >& v143,
  hls::stream< ap_uint<128> >& v144
) {	// L246
  l_S_mt_nt_0_mt14: for (int mt14 = 0; mt14 < 4; mt14++) {	// L249
    l_nt14: for (int nt14 = 0; nt14 < 4; nt14++) {	// L250
      l_S_k_0_k14: for (int k14 = 0; k14 < 64; k14++) {	// L251
        ap_uint<128> v148 = v142.read();	// L252
        ap_uint<128> b11;	// L253
        b11 = v148;	// L254
        ap_int<128> v150 = b11;	// L255
        int8_t v151;
        ap_int<128> v151_tmp = v150;
        v151 = v151_tmp(95, 88);	// L256
        v143.write(v151);	// L257
        ap_int<128> v152 = b11;	// L258
        v144.write(v152);	// L259
      }
    }
  }
}

void gemm_0_13(
  hls::stream< ap_uint<128> >& v153,
  hls::stream< int8_t >& v154,
  hls::stream< ap_uint<128> >& v155
) {	// L265
  l_S_mt_nt_0_mt15: for (int mt15 = 0; mt15 < 4; mt15++) {	// L268
    l_nt15: for (int nt15 = 0; nt15 < 4; nt15++) {	// L269
      l_S_k_0_k15: for (int k15 = 0; k15 < 64; k15++) {	// L270
        ap_uint<128> v159 = v153.read();	// L271
        ap_uint<128> b12;	// L272
        b12 = v159;	// L273
        ap_int<128> v161 = b12;	// L274
        int8_t v162;
        ap_int<128> v162_tmp = v161;
        v162 = v162_tmp(103, 96);	// L275
        v154.write(v162);	// L276
        ap_int<128> v163 = b12;	// L277
        v155.write(v163);	// L278
      }
    }
  }
}

void gemm_0_14(
  hls::stream< ap_uint<128> >& v164,
  hls::stream< int8_t >& v165,
  hls::stream< ap_uint<128> >& v166
) {	// L284
  l_S_mt_nt_0_mt16: for (int mt16 = 0; mt16 < 4; mt16++) {	// L287
    l_nt16: for (int nt16 = 0; nt16 < 4; nt16++) {	// L288
      l_S_k_0_k16: for (int k16 = 0; k16 < 64; k16++) {	// L289
        ap_uint<128> v170 = v164.read();	// L290
        ap_uint<128> b13;	// L291
        b13 = v170;	// L292
        ap_int<128> v172 = b13;	// L293
        int8_t v173;
        ap_int<128> v173_tmp = v172;
        v173 = v173_tmp(111, 104);	// L294
        v165.write(v173);	// L295
        ap_int<128> v174 = b13;	// L296
        v166.write(v174);	// L297
      }
    }
  }
}

void gemm_0_15(
  hls::stream< ap_uint<128> >& v175,
  hls::stream< int8_t >& v176,
  hls::stream< ap_uint<128> >& v177
) {	// L303
  l_S_mt_nt_0_mt17: for (int mt17 = 0; mt17 < 4; mt17++) {	// L306
    l_nt17: for (int nt17 = 0; nt17 < 4; nt17++) {	// L307
      l_S_k_0_k17: for (int k17 = 0; k17 < 64; k17++) {	// L308
        ap_uint<128> v181 = v175.read();	// L309
        ap_uint<128> b14;	// L310
        b14 = v181;	// L311
        ap_int<128> v183 = b14;	// L312
        int8_t v184;
        ap_int<128> v184_tmp = v183;
        v184 = v184_tmp(119, 112);	// L313
        v176.write(v184);	// L314
        ap_int<128> v185 = b14;	// L315
        v177.write(v185);	// L316
      }
    }
  }
}

void gemm_0_16(
  hls::stream< ap_uint<128> >& v186,
  hls::stream< int8_t >& v187
) {	// L322
  l_S_mt_nt_0_mt18: for (int mt18 = 0; mt18 < 4; mt18++) {	// L325
    l_nt18: for (int nt18 = 0; nt18 < 4; nt18++) {	// L326
      l_S_k_0_k18: for (int k18 = 0; k18 < 64; k18++) {	// L327
        ap_uint<128> v191 = v186.read();	// L328
        ap_uint<128> b15;	// L329
        b15 = v191;	// L330
        ap_int<128> v193 = b15;	// L331
        int8_t v194;
        ap_int<128> v194_tmp = v193;
        v194 = v194_tmp(127, 120);	// L332
        v187.write(v194);	// L333
      }
    }
  }
}

void gemm_1_0(
  hls::stream< ap_uint<128> >& v195,
  hls::stream< int8_t >& v196,
  hls::stream< ap_uint<128> >& v197
) {	// L339
  l_S_mt_nt_0_mt19: for (int mt19 = 0; mt19 < 4; mt19++) {	// L342
    l_nt19: for (int nt19 = 0; nt19 < 4; nt19++) {	// L343
      l_S_k_0_k19: for (int k19 = 0; k19 < 64; k19++) {	// L344
        ap_uint<128> v201 = v195.read();	// L345
        ap_uint<128> a;	// L346
        a = v201;	// L347
        ap_int<128> v203 = a;	// L348
        int8_t v204;
        ap_int<128> v204_tmp = v203;
        v204 = v204_tmp(7, 0);	// L349
        v196.write(v204);	// L350
        ap_int<128> v205 = a;	// L351
        v197.write(v205);	// L352
      }
    }
  }
}

void gemm_1_1(
  hls::stream< int8_t >& v206,
  hls::stream< int8_t >& v207,
  hls::stream< int8_t >& v208,
  hls::stream< int8_t >& v209,
  hls::stream< ap_uint<128> >& v210
) {	// L358
  l_S_mt_nt_0_mt20: for (int mt20 = 0; mt20 < 4; mt20++) {	// L366
    l_nt20: for (int nt20 = 0; nt20 < 4; nt20++) {	// L367
      int8_t c;	// L368
      c = 0;	// L369
      l_S_k_0_k20: for (int k20 = 0; k20 < 64; k20++) {	// L370
        int8_t v215 = v206.read();	// L371
        int8_t a1;	// L372
        a1 = v215;	// L373
        int8_t v217 = v207.read();	// L374
        int8_t b16;	// L375
        b16 = v217;	// L376
        int8_t v219 = a1;	// L377
        int8_t v220 = b16;	// L378
        int16_t v221 = v219;	// L379
        int16_t v222 = v220;	// L380
        int16_t v223 = v221 * v222;	// L381
        int8_t v224 = c;	// L382
        ap_int<17> v225 = v224;	// L383
        ap_int<17> v226 = v223;	// L384
        ap_int<17> v227 = v225 + v226;	// L385
        int8_t v228 = v227;	// L386
        c = v228;	// L387
        int8_t v229 = a1;	// L388
        v208.write(v229);	// L389
        int8_t v230 = b16;	// L390
        v209.write(v230);	// L391
      }
      ap_uint<128> packed_tmp;	// L393
      packed_tmp = 0;	// L394
      ap_uint<128> packed_c;	// L395
      packed_c = 0;	// L396
      l_S_m_1_m: for (int m = 0; m < 16; m++) {	// L397
        ap_int<33> v234 = m;	// L398
        bool v235 = v234 == 0;	// L399
        if (v235) {	// L400
          int8_t v236 = c;	// L401
          ap_int<128> v237 = packed_c;	// L402
          int64_t v238 = m;	// L403
          int64_t v239 = v238 * 8;	// L404
          ap_int<34> v240 = m;	// L405
          ap_int<34> v241 = v240 + 1;	// L406
          ap_int<66> v242 = v241;	// L407
          ap_int<66> v243 = v242 * 8;	// L408
          ap_int<66> v244 = v243 - 1;	// L409
          int v245 = v239;	// L410
          int v246 = v244;	// L411
          ap_int<128> v247;
          ap_int<128> v247_tmp = v237;
          v247_tmp(v246, v245) = v236;
          v247 = v247_tmp;	// L412
          packed_c = v247;	// L413
        } else {
          ap_int<128> v248 = packed_tmp;	// L415
          int64_t v249 = m;	// L416
          int64_t v250 = v249 * 8;	// L417
          ap_int<34> v251 = m;	// L418
          ap_int<34> v252 = v251 + 1;	// L419
          ap_int<66> v253 = v252;	// L420
          ap_int<66> v254 = v253 * 8;	// L421
          ap_int<66> v255 = v254 - 1;	// L422
          int v256 = v250;	// L423
          int v257 = v255;	// L424
          int8_t v258;
          ap_int<128> v258_tmp = v248;
          v258 = v258_tmp(v257, v256);	// L425
          ap_int<128> v259 = packed_c;	// L426
          ap_int<128> v260;
          ap_int<128> v260_tmp = v259;
          v260_tmp(v257, v256) = v258;
          v260 = v260_tmp;	// L436
          packed_c = v260;	// L437
        }
      }
      ap_int<128> v261 = packed_c;	// L440
      v210.write(v261);	// L441
    }
  }
}

void gemm_1_2(
  hls::stream< int8_t >& v262,
  hls::stream< int8_t >& v263,
  hls::stream< int8_t >& v264,
  hls::stream< int8_t >& v265,
  hls::stream< ap_uint<128> >& v266
) {	// L446
  l_S_mt_nt_0_mt21: for (int mt21 = 0; mt21 < 4; mt21++) {	// L454
    l_nt21: for (int nt21 = 0; nt21 < 4; nt21++) {	// L455
      int8_t c1;	// L456
      c1 = 0;	// L457
      l_S_k_0_k21: for (int k21 = 0; k21 < 64; k21++) {	// L458
        int8_t v271 = v262.read();	// L459
        int8_t a2;	// L460
        a2 = v271;	// L461
        int8_t v273 = v263.read();	// L462
        int8_t b17;	// L463
        b17 = v273;	// L464
        int8_t v275 = a2;	// L465
        int8_t v276 = b17;	// L466
        int16_t v277 = v275;	// L467
        int16_t v278 = v276;	// L468
        int16_t v279 = v277 * v278;	// L469
        int8_t v280 = c1;	// L470
        ap_int<17> v281 = v280;	// L471
        ap_int<17> v282 = v279;	// L472
        ap_int<17> v283 = v281 + v282;	// L473
        int8_t v284 = v283;	// L474
        c1 = v284;	// L475
        int8_t v285 = a2;	// L476
        v264.write(v285);	// L477
        int8_t v286 = b17;	// L478
        v265.write(v286);	// L479
      }
      ap_uint<128> packed_tmp1;	// L481
      packed_tmp1 = 0;	// L482
      ap_uint<128> packed_c1;	// L483
      packed_c1 = 0;	// L484
      l_S_m_1_m1: for (int m1 = 0; m1 < 16; m1++) {	// L485
        ap_int<33> v290 = m1;	// L486
        bool v291 = v290 == 0;	// L487
        if (v291) {	// L488
          int8_t v292 = c1;	// L489
          ap_int<128> v293 = packed_c1;	// L490
          int64_t v294 = m1;	// L491
          int64_t v295 = v294 * 8;	// L492
          ap_int<34> v296 = m1;	// L493
          ap_int<34> v297 = v296 + 1;	// L494
          ap_int<66> v298 = v297;	// L495
          ap_int<66> v299 = v298 * 8;	// L496
          ap_int<66> v300 = v299 - 1;	// L497
          int v301 = v295;	// L498
          int v302 = v300;	// L499
          ap_int<128> v303;
          ap_int<128> v303_tmp = v293;
          v303_tmp(v302, v301) = v292;
          v303 = v303_tmp;	// L500
          packed_c1 = v303;	// L501
        } else {
          ap_int<128> v304 = packed_tmp1;	// L503
          int64_t v305 = m1;	// L504
          int64_t v306 = v305 * 8;	// L505
          ap_int<34> v307 = m1;	// L506
          ap_int<34> v308 = v307 + 1;	// L507
          ap_int<66> v309 = v308;	// L508
          ap_int<66> v310 = v309 * 8;	// L509
          ap_int<66> v311 = v310 - 1;	// L510
          int v312 = v306;	// L511
          int v313 = v311;	// L512
          int8_t v314;
          ap_int<128> v314_tmp = v304;
          v314 = v314_tmp(v313, v312);	// L513
          ap_int<128> v315 = packed_c1;	// L514
          ap_int<128> v316;
          ap_int<128> v316_tmp = v315;
          v316_tmp(v313, v312) = v314;
          v316 = v316_tmp;	// L524
          packed_c1 = v316;	// L525
        }
      }
      ap_int<128> v317 = packed_c1;	// L528
      v266.write(v317);	// L529
    }
  }
}

void gemm_1_3(
  hls::stream< int8_t >& v318,
  hls::stream< int8_t >& v319,
  hls::stream< int8_t >& v320,
  hls::stream< int8_t >& v321,
  hls::stream< ap_uint<128> >& v322
) {	// L534
  l_S_mt_nt_0_mt22: for (int mt22 = 0; mt22 < 4; mt22++) {	// L542
    l_nt22: for (int nt22 = 0; nt22 < 4; nt22++) {	// L543
      int8_t c2;	// L544
      c2 = 0;	// L545
      l_S_k_0_k22: for (int k22 = 0; k22 < 64; k22++) {	// L546
        int8_t v327 = v318.read();	// L547
        int8_t a3;	// L548
        a3 = v327;	// L549
        int8_t v329 = v319.read();	// L550
        int8_t b18;	// L551
        b18 = v329;	// L552
        int8_t v331 = a3;	// L553
        int8_t v332 = b18;	// L554
        int16_t v333 = v331;	// L555
        int16_t v334 = v332;	// L556
        int16_t v335 = v333 * v334;	// L557
        int8_t v336 = c2;	// L558
        ap_int<17> v337 = v336;	// L559
        ap_int<17> v338 = v335;	// L560
        ap_int<17> v339 = v337 + v338;	// L561
        int8_t v340 = v339;	// L562
        c2 = v340;	// L563
        int8_t v341 = a3;	// L564
        v320.write(v341);	// L565
        int8_t v342 = b18;	// L566
        v321.write(v342);	// L567
      }
      ap_uint<128> packed_tmp2;	// L569
      packed_tmp2 = 0;	// L570
      ap_uint<128> packed_c2;	// L571
      packed_c2 = 0;	// L572
      l_S_m_1_m2: for (int m2 = 0; m2 < 16; m2++) {	// L573
        ap_int<33> v346 = m2;	// L574
        bool v347 = v346 == 0;	// L575
        if (v347) {	// L576
          int8_t v348 = c2;	// L577
          ap_int<128> v349 = packed_c2;	// L578
          int64_t v350 = m2;	// L579
          int64_t v351 = v350 * 8;	// L580
          ap_int<34> v352 = m2;	// L581
          ap_int<34> v353 = v352 + 1;	// L582
          ap_int<66> v354 = v353;	// L583
          ap_int<66> v355 = v354 * 8;	// L584
          ap_int<66> v356 = v355 - 1;	// L585
          int v357 = v351;	// L586
          int v358 = v356;	// L587
          ap_int<128> v359;
          ap_int<128> v359_tmp = v349;
          v359_tmp(v358, v357) = v348;
          v359 = v359_tmp;	// L588
          packed_c2 = v359;	// L589
        } else {
          ap_int<128> v360 = packed_tmp2;	// L591
          int64_t v361 = m2;	// L592
          int64_t v362 = v361 * 8;	// L593
          ap_int<34> v363 = m2;	// L594
          ap_int<34> v364 = v363 + 1;	// L595
          ap_int<66> v365 = v364;	// L596
          ap_int<66> v366 = v365 * 8;	// L597
          ap_int<66> v367 = v366 - 1;	// L598
          int v368 = v362;	// L599
          int v369 = v367;	// L600
          int8_t v370;
          ap_int<128> v370_tmp = v360;
          v370 = v370_tmp(v369, v368);	// L601
          ap_int<128> v371 = packed_c2;	// L602
          ap_int<128> v372;
          ap_int<128> v372_tmp = v371;
          v372_tmp(v369, v368) = v370;
          v372 = v372_tmp;	// L612
          packed_c2 = v372;	// L613
        }
      }
      ap_int<128> v373 = packed_c2;	// L616
      v322.write(v373);	// L617
    }
  }
}

void gemm_1_4(
  hls::stream< int8_t >& v374,
  hls::stream< int8_t >& v375,
  hls::stream< int8_t >& v376,
  hls::stream< int8_t >& v377,
  hls::stream< ap_uint<128> >& v378
) {	// L622
  l_S_mt_nt_0_mt23: for (int mt23 = 0; mt23 < 4; mt23++) {	// L630
    l_nt23: for (int nt23 = 0; nt23 < 4; nt23++) {	// L631
      int8_t c3;	// L632
      c3 = 0;	// L633
      l_S_k_0_k23: for (int k23 = 0; k23 < 64; k23++) {	// L634
        int8_t v383 = v374.read();	// L635
        int8_t a4;	// L636
        a4 = v383;	// L637
        int8_t v385 = v375.read();	// L638
        int8_t b19;	// L639
        b19 = v385;	// L640
        int8_t v387 = a4;	// L641
        int8_t v388 = b19;	// L642
        int16_t v389 = v387;	// L643
        int16_t v390 = v388;	// L644
        int16_t v391 = v389 * v390;	// L645
        int8_t v392 = c3;	// L646
        ap_int<17> v393 = v392;	// L647
        ap_int<17> v394 = v391;	// L648
        ap_int<17> v395 = v393 + v394;	// L649
        int8_t v396 = v395;	// L650
        c3 = v396;	// L651
        int8_t v397 = a4;	// L652
        v376.write(v397);	// L653
        int8_t v398 = b19;	// L654
        v377.write(v398);	// L655
      }
      ap_uint<128> packed_tmp3;	// L657
      packed_tmp3 = 0;	// L658
      ap_uint<128> packed_c3;	// L659
      packed_c3 = 0;	// L660
      l_S_m_1_m3: for (int m3 = 0; m3 < 16; m3++) {	// L661
        ap_int<33> v402 = m3;	// L662
        bool v403 = v402 == 0;	// L663
        if (v403) {	// L664
          int8_t v404 = c3;	// L665
          ap_int<128> v405 = packed_c3;	// L666
          int64_t v406 = m3;	// L667
          int64_t v407 = v406 * 8;	// L668
          ap_int<34> v408 = m3;	// L669
          ap_int<34> v409 = v408 + 1;	// L670
          ap_int<66> v410 = v409;	// L671
          ap_int<66> v411 = v410 * 8;	// L672
          ap_int<66> v412 = v411 - 1;	// L673
          int v413 = v407;	// L674
          int v414 = v412;	// L675
          ap_int<128> v415;
          ap_int<128> v415_tmp = v405;
          v415_tmp(v414, v413) = v404;
          v415 = v415_tmp;	// L676
          packed_c3 = v415;	// L677
        } else {
          ap_int<128> v416 = packed_tmp3;	// L679
          int64_t v417 = m3;	// L680
          int64_t v418 = v417 * 8;	// L681
          ap_int<34> v419 = m3;	// L682
          ap_int<34> v420 = v419 + 1;	// L683
          ap_int<66> v421 = v420;	// L684
          ap_int<66> v422 = v421 * 8;	// L685
          ap_int<66> v423 = v422 - 1;	// L686
          int v424 = v418;	// L687
          int v425 = v423;	// L688
          int8_t v426;
          ap_int<128> v426_tmp = v416;
          v426 = v426_tmp(v425, v424);	// L689
          ap_int<128> v427 = packed_c3;	// L690
          ap_int<128> v428;
          ap_int<128> v428_tmp = v427;
          v428_tmp(v425, v424) = v426;
          v428 = v428_tmp;	// L700
          packed_c3 = v428;	// L701
        }
      }
      ap_int<128> v429 = packed_c3;	// L704
      v378.write(v429);	// L705
    }
  }
}

void gemm_1_5(
  hls::stream< int8_t >& v430,
  hls::stream< int8_t >& v431,
  hls::stream< int8_t >& v432,
  hls::stream< int8_t >& v433,
  hls::stream< ap_uint<128> >& v434
) {	// L710
  l_S_mt_nt_0_mt24: for (int mt24 = 0; mt24 < 4; mt24++) {	// L718
    l_nt24: for (int nt24 = 0; nt24 < 4; nt24++) {	// L719
      int8_t c4;	// L720
      c4 = 0;	// L721
      l_S_k_0_k24: for (int k24 = 0; k24 < 64; k24++) {	// L722
        int8_t v439 = v430.read();	// L723
        int8_t a5;	// L724
        a5 = v439;	// L725
        int8_t v441 = v431.read();	// L726
        int8_t b20;	// L727
        b20 = v441;	// L728
        int8_t v443 = a5;	// L729
        int8_t v444 = b20;	// L730
        int16_t v445 = v443;	// L731
        int16_t v446 = v444;	// L732
        int16_t v447 = v445 * v446;	// L733
        int8_t v448 = c4;	// L734
        ap_int<17> v449 = v448;	// L735
        ap_int<17> v450 = v447;	// L736
        ap_int<17> v451 = v449 + v450;	// L737
        int8_t v452 = v451;	// L738
        c4 = v452;	// L739
        int8_t v453 = a5;	// L740
        v432.write(v453);	// L741
        int8_t v454 = b20;	// L742
        v433.write(v454);	// L743
      }
      ap_uint<128> packed_tmp4;	// L745
      packed_tmp4 = 0;	// L746
      ap_uint<128> packed_c4;	// L747
      packed_c4 = 0;	// L748
      l_S_m_1_m4: for (int m4 = 0; m4 < 16; m4++) {	// L749
        ap_int<33> v458 = m4;	// L750
        bool v459 = v458 == 0;	// L751
        if (v459) {	// L752
          int8_t v460 = c4;	// L753
          ap_int<128> v461 = packed_c4;	// L754
          int64_t v462 = m4;	// L755
          int64_t v463 = v462 * 8;	// L756
          ap_int<34> v464 = m4;	// L757
          ap_int<34> v465 = v464 + 1;	// L758
          ap_int<66> v466 = v465;	// L759
          ap_int<66> v467 = v466 * 8;	// L760
          ap_int<66> v468 = v467 - 1;	// L761
          int v469 = v463;	// L762
          int v470 = v468;	// L763
          ap_int<128> v471;
          ap_int<128> v471_tmp = v461;
          v471_tmp(v470, v469) = v460;
          v471 = v471_tmp;	// L764
          packed_c4 = v471;	// L765
        } else {
          ap_int<128> v472 = packed_tmp4;	// L767
          int64_t v473 = m4;	// L768
          int64_t v474 = v473 * 8;	// L769
          ap_int<34> v475 = m4;	// L770
          ap_int<34> v476 = v475 + 1;	// L771
          ap_int<66> v477 = v476;	// L772
          ap_int<66> v478 = v477 * 8;	// L773
          ap_int<66> v479 = v478 - 1;	// L774
          int v480 = v474;	// L775
          int v481 = v479;	// L776
          int8_t v482;
          ap_int<128> v482_tmp = v472;
          v482 = v482_tmp(v481, v480);	// L777
          ap_int<128> v483 = packed_c4;	// L778
          ap_int<128> v484;
          ap_int<128> v484_tmp = v483;
          v484_tmp(v481, v480) = v482;
          v484 = v484_tmp;	// L788
          packed_c4 = v484;	// L789
        }
      }
      ap_int<128> v485 = packed_c4;	// L792
      v434.write(v485);	// L793
    }
  }
}

void gemm_1_6(
  hls::stream< int8_t >& v486,
  hls::stream< int8_t >& v487,
  hls::stream< int8_t >& v488,
  hls::stream< int8_t >& v489,
  hls::stream< ap_uint<128> >& v490
) {	// L798
  l_S_mt_nt_0_mt25: for (int mt25 = 0; mt25 < 4; mt25++) {	// L806
    l_nt25: for (int nt25 = 0; nt25 < 4; nt25++) {	// L807
      int8_t c5;	// L808
      c5 = 0;	// L809
      l_S_k_0_k25: for (int k25 = 0; k25 < 64; k25++) {	// L810
        int8_t v495 = v486.read();	// L811
        int8_t a6;	// L812
        a6 = v495;	// L813
        int8_t v497 = v487.read();	// L814
        int8_t b21;	// L815
        b21 = v497;	// L816
        int8_t v499 = a6;	// L817
        int8_t v500 = b21;	// L818
        int16_t v501 = v499;	// L819
        int16_t v502 = v500;	// L820
        int16_t v503 = v501 * v502;	// L821
        int8_t v504 = c5;	// L822
        ap_int<17> v505 = v504;	// L823
        ap_int<17> v506 = v503;	// L824
        ap_int<17> v507 = v505 + v506;	// L825
        int8_t v508 = v507;	// L826
        c5 = v508;	// L827
        int8_t v509 = a6;	// L828
        v488.write(v509);	// L829
        int8_t v510 = b21;	// L830
        v489.write(v510);	// L831
      }
      ap_uint<128> packed_tmp5;	// L833
      packed_tmp5 = 0;	// L834
      ap_uint<128> packed_c5;	// L835
      packed_c5 = 0;	// L836
      l_S_m_1_m5: for (int m5 = 0; m5 < 16; m5++) {	// L837
        ap_int<33> v514 = m5;	// L838
        bool v515 = v514 == 0;	// L839
        if (v515) {	// L840
          int8_t v516 = c5;	// L841
          ap_int<128> v517 = packed_c5;	// L842
          int64_t v518 = m5;	// L843
          int64_t v519 = v518 * 8;	// L844
          ap_int<34> v520 = m5;	// L845
          ap_int<34> v521 = v520 + 1;	// L846
          ap_int<66> v522 = v521;	// L847
          ap_int<66> v523 = v522 * 8;	// L848
          ap_int<66> v524 = v523 - 1;	// L849
          int v525 = v519;	// L850
          int v526 = v524;	// L851
          ap_int<128> v527;
          ap_int<128> v527_tmp = v517;
          v527_tmp(v526, v525) = v516;
          v527 = v527_tmp;	// L852
          packed_c5 = v527;	// L853
        } else {
          ap_int<128> v528 = packed_tmp5;	// L855
          int64_t v529 = m5;	// L856
          int64_t v530 = v529 * 8;	// L857
          ap_int<34> v531 = m5;	// L858
          ap_int<34> v532 = v531 + 1;	// L859
          ap_int<66> v533 = v532;	// L860
          ap_int<66> v534 = v533 * 8;	// L861
          ap_int<66> v535 = v534 - 1;	// L862
          int v536 = v530;	// L863
          int v537 = v535;	// L864
          int8_t v538;
          ap_int<128> v538_tmp = v528;
          v538 = v538_tmp(v537, v536);	// L865
          ap_int<128> v539 = packed_c5;	// L866
          ap_int<128> v540;
          ap_int<128> v540_tmp = v539;
          v540_tmp(v537, v536) = v538;
          v540 = v540_tmp;	// L876
          packed_c5 = v540;	// L877
        }
      }
      ap_int<128> v541 = packed_c5;	// L880
      v490.write(v541);	// L881
    }
  }
}

void gemm_1_7(
  hls::stream< int8_t >& v542,
  hls::stream< int8_t >& v543,
  hls::stream< int8_t >& v544,
  hls::stream< int8_t >& v545,
  hls::stream< ap_uint<128> >& v546
) {	// L886
  l_S_mt_nt_0_mt26: for (int mt26 = 0; mt26 < 4; mt26++) {	// L894
    l_nt26: for (int nt26 = 0; nt26 < 4; nt26++) {	// L895
      int8_t c6;	// L896
      c6 = 0;	// L897
      l_S_k_0_k26: for (int k26 = 0; k26 < 64; k26++) {	// L898
        int8_t v551 = v542.read();	// L899
        int8_t a7;	// L900
        a7 = v551;	// L901
        int8_t v553 = v543.read();	// L902
        int8_t b22;	// L903
        b22 = v553;	// L904
        int8_t v555 = a7;	// L905
        int8_t v556 = b22;	// L906
        int16_t v557 = v555;	// L907
        int16_t v558 = v556;	// L908
        int16_t v559 = v557 * v558;	// L909
        int8_t v560 = c6;	// L910
        ap_int<17> v561 = v560;	// L911
        ap_int<17> v562 = v559;	// L912
        ap_int<17> v563 = v561 + v562;	// L913
        int8_t v564 = v563;	// L914
        c6 = v564;	// L915
        int8_t v565 = a7;	// L916
        v544.write(v565);	// L917
        int8_t v566 = b22;	// L918
        v545.write(v566);	// L919
      }
      ap_uint<128> packed_tmp6;	// L921
      packed_tmp6 = 0;	// L922
      ap_uint<128> packed_c6;	// L923
      packed_c6 = 0;	// L924
      l_S_m_1_m6: for (int m6 = 0; m6 < 16; m6++) {	// L925
        ap_int<33> v570 = m6;	// L926
        bool v571 = v570 == 0;	// L927
        if (v571) {	// L928
          int8_t v572 = c6;	// L929
          ap_int<128> v573 = packed_c6;	// L930
          int64_t v574 = m6;	// L931
          int64_t v575 = v574 * 8;	// L932
          ap_int<34> v576 = m6;	// L933
          ap_int<34> v577 = v576 + 1;	// L934
          ap_int<66> v578 = v577;	// L935
          ap_int<66> v579 = v578 * 8;	// L936
          ap_int<66> v580 = v579 - 1;	// L937
          int v581 = v575;	// L938
          int v582 = v580;	// L939
          ap_int<128> v583;
          ap_int<128> v583_tmp = v573;
          v583_tmp(v582, v581) = v572;
          v583 = v583_tmp;	// L940
          packed_c6 = v583;	// L941
        } else {
          ap_int<128> v584 = packed_tmp6;	// L943
          int64_t v585 = m6;	// L944
          int64_t v586 = v585 * 8;	// L945
          ap_int<34> v587 = m6;	// L946
          ap_int<34> v588 = v587 + 1;	// L947
          ap_int<66> v589 = v588;	// L948
          ap_int<66> v590 = v589 * 8;	// L949
          ap_int<66> v591 = v590 - 1;	// L950
          int v592 = v586;	// L951
          int v593 = v591;	// L952
          int8_t v594;
          ap_int<128> v594_tmp = v584;
          v594 = v594_tmp(v593, v592);	// L953
          ap_int<128> v595 = packed_c6;	// L954
          ap_int<128> v596;
          ap_int<128> v596_tmp = v595;
          v596_tmp(v593, v592) = v594;
          v596 = v596_tmp;	// L964
          packed_c6 = v596;	// L965
        }
      }
      ap_int<128> v597 = packed_c6;	// L968
      v546.write(v597);	// L969
    }
  }
}

void gemm_1_8(
  hls::stream< int8_t >& v598,
  hls::stream< int8_t >& v599,
  hls::stream< int8_t >& v600,
  hls::stream< int8_t >& v601,
  hls::stream< ap_uint<128> >& v602
) {	// L974
  l_S_mt_nt_0_mt27: for (int mt27 = 0; mt27 < 4; mt27++) {	// L982
    l_nt27: for (int nt27 = 0; nt27 < 4; nt27++) {	// L983
      int8_t c7;	// L984
      c7 = 0;	// L985
      l_S_k_0_k27: for (int k27 = 0; k27 < 64; k27++) {	// L986
        int8_t v607 = v598.read();	// L987
        int8_t a8;	// L988
        a8 = v607;	// L989
        int8_t v609 = v599.read();	// L990
        int8_t b23;	// L991
        b23 = v609;	// L992
        int8_t v611 = a8;	// L993
        int8_t v612 = b23;	// L994
        int16_t v613 = v611;	// L995
        int16_t v614 = v612;	// L996
        int16_t v615 = v613 * v614;	// L997
        int8_t v616 = c7;	// L998
        ap_int<17> v617 = v616;	// L999
        ap_int<17> v618 = v615;	// L1000
        ap_int<17> v619 = v617 + v618;	// L1001
        int8_t v620 = v619;	// L1002
        c7 = v620;	// L1003
        int8_t v621 = a8;	// L1004
        v600.write(v621);	// L1005
        int8_t v622 = b23;	// L1006
        v601.write(v622);	// L1007
      }
      ap_uint<128> packed_tmp7;	// L1009
      packed_tmp7 = 0;	// L1010
      ap_uint<128> packed_c7;	// L1011
      packed_c7 = 0;	// L1012
      l_S_m_1_m7: for (int m7 = 0; m7 < 16; m7++) {	// L1013
        ap_int<33> v626 = m7;	// L1014
        bool v627 = v626 == 0;	// L1015
        if (v627) {	// L1016
          int8_t v628 = c7;	// L1017
          ap_int<128> v629 = packed_c7;	// L1018
          int64_t v630 = m7;	// L1019
          int64_t v631 = v630 * 8;	// L1020
          ap_int<34> v632 = m7;	// L1021
          ap_int<34> v633 = v632 + 1;	// L1022
          ap_int<66> v634 = v633;	// L1023
          ap_int<66> v635 = v634 * 8;	// L1024
          ap_int<66> v636 = v635 - 1;	// L1025
          int v637 = v631;	// L1026
          int v638 = v636;	// L1027
          ap_int<128> v639;
          ap_int<128> v639_tmp = v629;
          v639_tmp(v638, v637) = v628;
          v639 = v639_tmp;	// L1028
          packed_c7 = v639;	// L1029
        } else {
          ap_int<128> v640 = packed_tmp7;	// L1031
          int64_t v641 = m7;	// L1032
          int64_t v642 = v641 * 8;	// L1033
          ap_int<34> v643 = m7;	// L1034
          ap_int<34> v644 = v643 + 1;	// L1035
          ap_int<66> v645 = v644;	// L1036
          ap_int<66> v646 = v645 * 8;	// L1037
          ap_int<66> v647 = v646 - 1;	// L1038
          int v648 = v642;	// L1039
          int v649 = v647;	// L1040
          int8_t v650;
          ap_int<128> v650_tmp = v640;
          v650 = v650_tmp(v649, v648);	// L1041
          ap_int<128> v651 = packed_c7;	// L1042
          ap_int<128> v652;
          ap_int<128> v652_tmp = v651;
          v652_tmp(v649, v648) = v650;
          v652 = v652_tmp;	// L1052
          packed_c7 = v652;	// L1053
        }
      }
      ap_int<128> v653 = packed_c7;	// L1056
      v602.write(v653);	// L1057
    }
  }
}

void gemm_1_9(
  hls::stream< int8_t >& v654,
  hls::stream< int8_t >& v655,
  hls::stream< int8_t >& v656,
  hls::stream< int8_t >& v657,
  hls::stream< ap_uint<128> >& v658
) {	// L1062
  l_S_mt_nt_0_mt28: for (int mt28 = 0; mt28 < 4; mt28++) {	// L1070
    l_nt28: for (int nt28 = 0; nt28 < 4; nt28++) {	// L1071
      int8_t c8;	// L1072
      c8 = 0;	// L1073
      l_S_k_0_k28: for (int k28 = 0; k28 < 64; k28++) {	// L1074
        int8_t v663 = v654.read();	// L1075
        int8_t a9;	// L1076
        a9 = v663;	// L1077
        int8_t v665 = v655.read();	// L1078
        int8_t b24;	// L1079
        b24 = v665;	// L1080
        int8_t v667 = a9;	// L1081
        int8_t v668 = b24;	// L1082
        int16_t v669 = v667;	// L1083
        int16_t v670 = v668;	// L1084
        int16_t v671 = v669 * v670;	// L1085
        int8_t v672 = c8;	// L1086
        ap_int<17> v673 = v672;	// L1087
        ap_int<17> v674 = v671;	// L1088
        ap_int<17> v675 = v673 + v674;	// L1089
        int8_t v676 = v675;	// L1090
        c8 = v676;	// L1091
        int8_t v677 = a9;	// L1092
        v656.write(v677);	// L1093
        int8_t v678 = b24;	// L1094
        v657.write(v678);	// L1095
      }
      ap_uint<128> packed_tmp8;	// L1097
      packed_tmp8 = 0;	// L1098
      ap_uint<128> packed_c8;	// L1099
      packed_c8 = 0;	// L1100
      l_S_m_1_m8: for (int m8 = 0; m8 < 16; m8++) {	// L1101
        ap_int<33> v682 = m8;	// L1102
        bool v683 = v682 == 0;	// L1103
        if (v683) {	// L1104
          int8_t v684 = c8;	// L1105
          ap_int<128> v685 = packed_c8;	// L1106
          int64_t v686 = m8;	// L1107
          int64_t v687 = v686 * 8;	// L1108
          ap_int<34> v688 = m8;	// L1109
          ap_int<34> v689 = v688 + 1;	// L1110
          ap_int<66> v690 = v689;	// L1111
          ap_int<66> v691 = v690 * 8;	// L1112
          ap_int<66> v692 = v691 - 1;	// L1113
          int v693 = v687;	// L1114
          int v694 = v692;	// L1115
          ap_int<128> v695;
          ap_int<128> v695_tmp = v685;
          v695_tmp(v694, v693) = v684;
          v695 = v695_tmp;	// L1116
          packed_c8 = v695;	// L1117
        } else {
          ap_int<128> v696 = packed_tmp8;	// L1119
          int64_t v697 = m8;	// L1120
          int64_t v698 = v697 * 8;	// L1121
          ap_int<34> v699 = m8;	// L1122
          ap_int<34> v700 = v699 + 1;	// L1123
          ap_int<66> v701 = v700;	// L1124
          ap_int<66> v702 = v701 * 8;	// L1125
          ap_int<66> v703 = v702 - 1;	// L1126
          int v704 = v698;	// L1127
          int v705 = v703;	// L1128
          int8_t v706;
          ap_int<128> v706_tmp = v696;
          v706 = v706_tmp(v705, v704);	// L1129
          ap_int<128> v707 = packed_c8;	// L1130
          ap_int<128> v708;
          ap_int<128> v708_tmp = v707;
          v708_tmp(v705, v704) = v706;
          v708 = v708_tmp;	// L1140
          packed_c8 = v708;	// L1141
        }
      }
      ap_int<128> v709 = packed_c8;	// L1144
      v658.write(v709);	// L1145
    }
  }
}

void gemm_1_10(
  hls::stream< int8_t >& v710,
  hls::stream< int8_t >& v711,
  hls::stream< int8_t >& v712,
  hls::stream< int8_t >& v713,
  hls::stream< ap_uint<128> >& v714
) {	// L1150
  l_S_mt_nt_0_mt29: for (int mt29 = 0; mt29 < 4; mt29++) {	// L1158
    l_nt29: for (int nt29 = 0; nt29 < 4; nt29++) {	// L1159
      int8_t c9;	// L1160
      c9 = 0;	// L1161
      l_S_k_0_k29: for (int k29 = 0; k29 < 64; k29++) {	// L1162
        int8_t v719 = v710.read();	// L1163
        int8_t a10;	// L1164
        a10 = v719;	// L1165
        int8_t v721 = v711.read();	// L1166
        int8_t b25;	// L1167
        b25 = v721;	// L1168
        int8_t v723 = a10;	// L1169
        int8_t v724 = b25;	// L1170
        int16_t v725 = v723;	// L1171
        int16_t v726 = v724;	// L1172
        int16_t v727 = v725 * v726;	// L1173
        int8_t v728 = c9;	// L1174
        ap_int<17> v729 = v728;	// L1175
        ap_int<17> v730 = v727;	// L1176
        ap_int<17> v731 = v729 + v730;	// L1177
        int8_t v732 = v731;	// L1178
        c9 = v732;	// L1179
        int8_t v733 = a10;	// L1180
        v712.write(v733);	// L1181
        int8_t v734 = b25;	// L1182
        v713.write(v734);	// L1183
      }
      ap_uint<128> packed_tmp9;	// L1185
      packed_tmp9 = 0;	// L1186
      ap_uint<128> packed_c9;	// L1187
      packed_c9 = 0;	// L1188
      l_S_m_1_m9: for (int m9 = 0; m9 < 16; m9++) {	// L1189
        ap_int<33> v738 = m9;	// L1190
        bool v739 = v738 == 0;	// L1191
        if (v739) {	// L1192
          int8_t v740 = c9;	// L1193
          ap_int<128> v741 = packed_c9;	// L1194
          int64_t v742 = m9;	// L1195
          int64_t v743 = v742 * 8;	// L1196
          ap_int<34> v744 = m9;	// L1197
          ap_int<34> v745 = v744 + 1;	// L1198
          ap_int<66> v746 = v745;	// L1199
          ap_int<66> v747 = v746 * 8;	// L1200
          ap_int<66> v748 = v747 - 1;	// L1201
          int v749 = v743;	// L1202
          int v750 = v748;	// L1203
          ap_int<128> v751;
          ap_int<128> v751_tmp = v741;
          v751_tmp(v750, v749) = v740;
          v751 = v751_tmp;	// L1204
          packed_c9 = v751;	// L1205
        } else {
          ap_int<128> v752 = packed_tmp9;	// L1207
          int64_t v753 = m9;	// L1208
          int64_t v754 = v753 * 8;	// L1209
          ap_int<34> v755 = m9;	// L1210
          ap_int<34> v756 = v755 + 1;	// L1211
          ap_int<66> v757 = v756;	// L1212
          ap_int<66> v758 = v757 * 8;	// L1213
          ap_int<66> v759 = v758 - 1;	// L1214
          int v760 = v754;	// L1215
          int v761 = v759;	// L1216
          int8_t v762;
          ap_int<128> v762_tmp = v752;
          v762 = v762_tmp(v761, v760);	// L1217
          ap_int<128> v763 = packed_c9;	// L1218
          ap_int<128> v764;
          ap_int<128> v764_tmp = v763;
          v764_tmp(v761, v760) = v762;
          v764 = v764_tmp;	// L1228
          packed_c9 = v764;	// L1229
        }
      }
      ap_int<128> v765 = packed_c9;	// L1232
      v714.write(v765);	// L1233
    }
  }
}

void gemm_1_11(
  hls::stream< int8_t >& v766,
  hls::stream< int8_t >& v767,
  hls::stream< int8_t >& v768,
  hls::stream< int8_t >& v769,
  hls::stream< ap_uint<128> >& v770
) {	// L1238
  l_S_mt_nt_0_mt30: for (int mt30 = 0; mt30 < 4; mt30++) {	// L1246
    l_nt30: for (int nt30 = 0; nt30 < 4; nt30++) {	// L1247
      int8_t c10;	// L1248
      c10 = 0;	// L1249
      l_S_k_0_k30: for (int k30 = 0; k30 < 64; k30++) {	// L1250
        int8_t v775 = v766.read();	// L1251
        int8_t a11;	// L1252
        a11 = v775;	// L1253
        int8_t v777 = v767.read();	// L1254
        int8_t b26;	// L1255
        b26 = v777;	// L1256
        int8_t v779 = a11;	// L1257
        int8_t v780 = b26;	// L1258
        int16_t v781 = v779;	// L1259
        int16_t v782 = v780;	// L1260
        int16_t v783 = v781 * v782;	// L1261
        int8_t v784 = c10;	// L1262
        ap_int<17> v785 = v784;	// L1263
        ap_int<17> v786 = v783;	// L1264
        ap_int<17> v787 = v785 + v786;	// L1265
        int8_t v788 = v787;	// L1266
        c10 = v788;	// L1267
        int8_t v789 = a11;	// L1268
        v768.write(v789);	// L1269
        int8_t v790 = b26;	// L1270
        v769.write(v790);	// L1271
      }
      ap_uint<128> packed_tmp10;	// L1273
      packed_tmp10 = 0;	// L1274
      ap_uint<128> packed_c10;	// L1275
      packed_c10 = 0;	// L1276
      l_S_m_1_m10: for (int m10 = 0; m10 < 16; m10++) {	// L1277
        ap_int<33> v794 = m10;	// L1278
        bool v795 = v794 == 0;	// L1279
        if (v795) {	// L1280
          int8_t v796 = c10;	// L1281
          ap_int<128> v797 = packed_c10;	// L1282
          int64_t v798 = m10;	// L1283
          int64_t v799 = v798 * 8;	// L1284
          ap_int<34> v800 = m10;	// L1285
          ap_int<34> v801 = v800 + 1;	// L1286
          ap_int<66> v802 = v801;	// L1287
          ap_int<66> v803 = v802 * 8;	// L1288
          ap_int<66> v804 = v803 - 1;	// L1289
          int v805 = v799;	// L1290
          int v806 = v804;	// L1291
          ap_int<128> v807;
          ap_int<128> v807_tmp = v797;
          v807_tmp(v806, v805) = v796;
          v807 = v807_tmp;	// L1292
          packed_c10 = v807;	// L1293
        } else {
          ap_int<128> v808 = packed_tmp10;	// L1295
          int64_t v809 = m10;	// L1296
          int64_t v810 = v809 * 8;	// L1297
          ap_int<34> v811 = m10;	// L1298
          ap_int<34> v812 = v811 + 1;	// L1299
          ap_int<66> v813 = v812;	// L1300
          ap_int<66> v814 = v813 * 8;	// L1301
          ap_int<66> v815 = v814 - 1;	// L1302
          int v816 = v810;	// L1303
          int v817 = v815;	// L1304
          int8_t v818;
          ap_int<128> v818_tmp = v808;
          v818 = v818_tmp(v817, v816);	// L1305
          ap_int<128> v819 = packed_c10;	// L1306
          ap_int<128> v820;
          ap_int<128> v820_tmp = v819;
          v820_tmp(v817, v816) = v818;
          v820 = v820_tmp;	// L1316
          packed_c10 = v820;	// L1317
        }
      }
      ap_int<128> v821 = packed_c10;	// L1320
      v770.write(v821);	// L1321
    }
  }
}

void gemm_1_12(
  hls::stream< int8_t >& v822,
  hls::stream< int8_t >& v823,
  hls::stream< int8_t >& v824,
  hls::stream< int8_t >& v825,
  hls::stream< ap_uint<128> >& v826
) {	// L1326
  l_S_mt_nt_0_mt31: for (int mt31 = 0; mt31 < 4; mt31++) {	// L1334
    l_nt31: for (int nt31 = 0; nt31 < 4; nt31++) {	// L1335
      int8_t c11;	// L1336
      c11 = 0;	// L1337
      l_S_k_0_k31: for (int k31 = 0; k31 < 64; k31++) {	// L1338
        int8_t v831 = v822.read();	// L1339
        int8_t a12;	// L1340
        a12 = v831;	// L1341
        int8_t v833 = v823.read();	// L1342
        int8_t b27;	// L1343
        b27 = v833;	// L1344
        int8_t v835 = a12;	// L1345
        int8_t v836 = b27;	// L1346
        int16_t v837 = v835;	// L1347
        int16_t v838 = v836;	// L1348
        int16_t v839 = v837 * v838;	// L1349
        int8_t v840 = c11;	// L1350
        ap_int<17> v841 = v840;	// L1351
        ap_int<17> v842 = v839;	// L1352
        ap_int<17> v843 = v841 + v842;	// L1353
        int8_t v844 = v843;	// L1354
        c11 = v844;	// L1355
        int8_t v845 = a12;	// L1356
        v824.write(v845);	// L1357
        int8_t v846 = b27;	// L1358
        v825.write(v846);	// L1359
      }
      ap_uint<128> packed_tmp11;	// L1361
      packed_tmp11 = 0;	// L1362
      ap_uint<128> packed_c11;	// L1363
      packed_c11 = 0;	// L1364
      l_S_m_1_m11: for (int m11 = 0; m11 < 16; m11++) {	// L1365
        ap_int<33> v850 = m11;	// L1366
        bool v851 = v850 == 0;	// L1367
        if (v851) {	// L1368
          int8_t v852 = c11;	// L1369
          ap_int<128> v853 = packed_c11;	// L1370
          int64_t v854 = m11;	// L1371
          int64_t v855 = v854 * 8;	// L1372
          ap_int<34> v856 = m11;	// L1373
          ap_int<34> v857 = v856 + 1;	// L1374
          ap_int<66> v858 = v857;	// L1375
          ap_int<66> v859 = v858 * 8;	// L1376
          ap_int<66> v860 = v859 - 1;	// L1377
          int v861 = v855;	// L1378
          int v862 = v860;	// L1379
          ap_int<128> v863;
          ap_int<128> v863_tmp = v853;
          v863_tmp(v862, v861) = v852;
          v863 = v863_tmp;	// L1380
          packed_c11 = v863;	// L1381
        } else {
          ap_int<128> v864 = packed_tmp11;	// L1383
          int64_t v865 = m11;	// L1384
          int64_t v866 = v865 * 8;	// L1385
          ap_int<34> v867 = m11;	// L1386
          ap_int<34> v868 = v867 + 1;	// L1387
          ap_int<66> v869 = v868;	// L1388
          ap_int<66> v870 = v869 * 8;	// L1389
          ap_int<66> v871 = v870 - 1;	// L1390
          int v872 = v866;	// L1391
          int v873 = v871;	// L1392
          int8_t v874;
          ap_int<128> v874_tmp = v864;
          v874 = v874_tmp(v873, v872);	// L1393
          ap_int<128> v875 = packed_c11;	// L1394
          ap_int<128> v876;
          ap_int<128> v876_tmp = v875;
          v876_tmp(v873, v872) = v874;
          v876 = v876_tmp;	// L1404
          packed_c11 = v876;	// L1405
        }
      }
      ap_int<128> v877 = packed_c11;	// L1408
      v826.write(v877);	// L1409
    }
  }
}

void gemm_1_13(
  hls::stream< int8_t >& v878,
  hls::stream< int8_t >& v879,
  hls::stream< int8_t >& v880,
  hls::stream< int8_t >& v881,
  hls::stream< ap_uint<128> >& v882
) {	// L1414
  l_S_mt_nt_0_mt32: for (int mt32 = 0; mt32 < 4; mt32++) {	// L1422
    l_nt32: for (int nt32 = 0; nt32 < 4; nt32++) {	// L1423
      int8_t c12;	// L1424
      c12 = 0;	// L1425
      l_S_k_0_k32: for (int k32 = 0; k32 < 64; k32++) {	// L1426
        int8_t v887 = v878.read();	// L1427
        int8_t a13;	// L1428
        a13 = v887;	// L1429
        int8_t v889 = v879.read();	// L1430
        int8_t b28;	// L1431
        b28 = v889;	// L1432
        int8_t v891 = a13;	// L1433
        int8_t v892 = b28;	// L1434
        int16_t v893 = v891;	// L1435
        int16_t v894 = v892;	// L1436
        int16_t v895 = v893 * v894;	// L1437
        int8_t v896 = c12;	// L1438
        ap_int<17> v897 = v896;	// L1439
        ap_int<17> v898 = v895;	// L1440
        ap_int<17> v899 = v897 + v898;	// L1441
        int8_t v900 = v899;	// L1442
        c12 = v900;	// L1443
        int8_t v901 = a13;	// L1444
        v880.write(v901);	// L1445
        int8_t v902 = b28;	// L1446
        v881.write(v902);	// L1447
      }
      ap_uint<128> packed_tmp12;	// L1449
      packed_tmp12 = 0;	// L1450
      ap_uint<128> packed_c12;	// L1451
      packed_c12 = 0;	// L1452
      l_S_m_1_m12: for (int m12 = 0; m12 < 16; m12++) {	// L1453
        ap_int<33> v906 = m12;	// L1454
        bool v907 = v906 == 0;	// L1455
        if (v907) {	// L1456
          int8_t v908 = c12;	// L1457
          ap_int<128> v909 = packed_c12;	// L1458
          int64_t v910 = m12;	// L1459
          int64_t v911 = v910 * 8;	// L1460
          ap_int<34> v912 = m12;	// L1461
          ap_int<34> v913 = v912 + 1;	// L1462
          ap_int<66> v914 = v913;	// L1463
          ap_int<66> v915 = v914 * 8;	// L1464
          ap_int<66> v916 = v915 - 1;	// L1465
          int v917 = v911;	// L1466
          int v918 = v916;	// L1467
          ap_int<128> v919;
          ap_int<128> v919_tmp = v909;
          v919_tmp(v918, v917) = v908;
          v919 = v919_tmp;	// L1468
          packed_c12 = v919;	// L1469
        } else {
          ap_int<128> v920 = packed_tmp12;	// L1471
          int64_t v921 = m12;	// L1472
          int64_t v922 = v921 * 8;	// L1473
          ap_int<34> v923 = m12;	// L1474
          ap_int<34> v924 = v923 + 1;	// L1475
          ap_int<66> v925 = v924;	// L1476
          ap_int<66> v926 = v925 * 8;	// L1477
          ap_int<66> v927 = v926 - 1;	// L1478
          int v928 = v922;	// L1479
          int v929 = v927;	// L1480
          int8_t v930;
          ap_int<128> v930_tmp = v920;
          v930 = v930_tmp(v929, v928);	// L1481
          ap_int<128> v931 = packed_c12;	// L1482
          ap_int<128> v932;
          ap_int<128> v932_tmp = v931;
          v932_tmp(v929, v928) = v930;
          v932 = v932_tmp;	// L1492
          packed_c12 = v932;	// L1493
        }
      }
      ap_int<128> v933 = packed_c12;	// L1496
      v882.write(v933);	// L1497
    }
  }
}

void gemm_1_14(
  hls::stream< int8_t >& v934,
  hls::stream< int8_t >& v935,
  hls::stream< int8_t >& v936,
  hls::stream< int8_t >& v937,
  hls::stream< ap_uint<128> >& v938
) {	// L1502
  l_S_mt_nt_0_mt33: for (int mt33 = 0; mt33 < 4; mt33++) {	// L1510
    l_nt33: for (int nt33 = 0; nt33 < 4; nt33++) {	// L1511
      int8_t c13;	// L1512
      c13 = 0;	// L1513
      l_S_k_0_k33: for (int k33 = 0; k33 < 64; k33++) {	// L1514
        int8_t v943 = v934.read();	// L1515
        int8_t a14;	// L1516
        a14 = v943;	// L1517
        int8_t v945 = v935.read();	// L1518
        int8_t b29;	// L1519
        b29 = v945;	// L1520
        int8_t v947 = a14;	// L1521
        int8_t v948 = b29;	// L1522
        int16_t v949 = v947;	// L1523
        int16_t v950 = v948;	// L1524
        int16_t v951 = v949 * v950;	// L1525
        int8_t v952 = c13;	// L1526
        ap_int<17> v953 = v952;	// L1527
        ap_int<17> v954 = v951;	// L1528
        ap_int<17> v955 = v953 + v954;	// L1529
        int8_t v956 = v955;	// L1530
        c13 = v956;	// L1531
        int8_t v957 = a14;	// L1532
        v936.write(v957);	// L1533
        int8_t v958 = b29;	// L1534
        v937.write(v958);	// L1535
      }
      ap_uint<128> packed_tmp13;	// L1537
      packed_tmp13 = 0;	// L1538
      ap_uint<128> packed_c13;	// L1539
      packed_c13 = 0;	// L1540
      l_S_m_1_m13: for (int m13 = 0; m13 < 16; m13++) {	// L1541
        ap_int<33> v962 = m13;	// L1542
        bool v963 = v962 == 0;	// L1543
        if (v963) {	// L1544
          int8_t v964 = c13;	// L1545
          ap_int<128> v965 = packed_c13;	// L1546
          int64_t v966 = m13;	// L1547
          int64_t v967 = v966 * 8;	// L1548
          ap_int<34> v968 = m13;	// L1549
          ap_int<34> v969 = v968 + 1;	// L1550
          ap_int<66> v970 = v969;	// L1551
          ap_int<66> v971 = v970 * 8;	// L1552
          ap_int<66> v972 = v971 - 1;	// L1553
          int v973 = v967;	// L1554
          int v974 = v972;	// L1555
          ap_int<128> v975;
          ap_int<128> v975_tmp = v965;
          v975_tmp(v974, v973) = v964;
          v975 = v975_tmp;	// L1556
          packed_c13 = v975;	// L1557
        } else {
          ap_int<128> v976 = packed_tmp13;	// L1559
          int64_t v977 = m13;	// L1560
          int64_t v978 = v977 * 8;	// L1561
          ap_int<34> v979 = m13;	// L1562
          ap_int<34> v980 = v979 + 1;	// L1563
          ap_int<66> v981 = v980;	// L1564
          ap_int<66> v982 = v981 * 8;	// L1565
          ap_int<66> v983 = v982 - 1;	// L1566
          int v984 = v978;	// L1567
          int v985 = v983;	// L1568
          int8_t v986;
          ap_int<128> v986_tmp = v976;
          v986 = v986_tmp(v985, v984);	// L1569
          ap_int<128> v987 = packed_c13;	// L1570
          ap_int<128> v988;
          ap_int<128> v988_tmp = v987;
          v988_tmp(v985, v984) = v986;
          v988 = v988_tmp;	// L1580
          packed_c13 = v988;	// L1581
        }
      }
      ap_int<128> v989 = packed_c13;	// L1584
      v938.write(v989);	// L1585
    }
  }
}

void gemm_1_15(
  hls::stream< int8_t >& v990,
  hls::stream< int8_t >& v991,
  hls::stream< int8_t >& v992,
  hls::stream< int8_t >& v993,
  hls::stream< ap_uint<128> >& v994
) {	// L1590
  l_S_mt_nt_0_mt34: for (int mt34 = 0; mt34 < 4; mt34++) {	// L1598
    l_nt34: for (int nt34 = 0; nt34 < 4; nt34++) {	// L1599
      int8_t c14;	// L1600
      c14 = 0;	// L1601
      l_S_k_0_k34: for (int k34 = 0; k34 < 64; k34++) {	// L1602
        int8_t v999 = v990.read();	// L1603
        int8_t a15;	// L1604
        a15 = v999;	// L1605
        int8_t v1001 = v991.read();	// L1606
        int8_t b30;	// L1607
        b30 = v1001;	// L1608
        int8_t v1003 = a15;	// L1609
        int8_t v1004 = b30;	// L1610
        int16_t v1005 = v1003;	// L1611
        int16_t v1006 = v1004;	// L1612
        int16_t v1007 = v1005 * v1006;	// L1613
        int8_t v1008 = c14;	// L1614
        ap_int<17> v1009 = v1008;	// L1615
        ap_int<17> v1010 = v1007;	// L1616
        ap_int<17> v1011 = v1009 + v1010;	// L1617
        int8_t v1012 = v1011;	// L1618
        c14 = v1012;	// L1619
        int8_t v1013 = a15;	// L1620
        v992.write(v1013);	// L1621
        int8_t v1014 = b30;	// L1622
        v993.write(v1014);	// L1623
      }
      ap_uint<128> packed_tmp14;	// L1625
      packed_tmp14 = 0;	// L1626
      ap_uint<128> packed_c14;	// L1627
      packed_c14 = 0;	// L1628
      l_S_m_1_m14: for (int m14 = 0; m14 < 16; m14++) {	// L1629
        ap_int<33> v1018 = m14;	// L1630
        bool v1019 = v1018 == 0;	// L1631
        if (v1019) {	// L1632
          int8_t v1020 = c14;	// L1633
          ap_int<128> v1021 = packed_c14;	// L1634
          int64_t v1022 = m14;	// L1635
          int64_t v1023 = v1022 * 8;	// L1636
          ap_int<34> v1024 = m14;	// L1637
          ap_int<34> v1025 = v1024 + 1;	// L1638
          ap_int<66> v1026 = v1025;	// L1639
          ap_int<66> v1027 = v1026 * 8;	// L1640
          ap_int<66> v1028 = v1027 - 1;	// L1641
          int v1029 = v1023;	// L1642
          int v1030 = v1028;	// L1643
          ap_int<128> v1031;
          ap_int<128> v1031_tmp = v1021;
          v1031_tmp(v1030, v1029) = v1020;
          v1031 = v1031_tmp;	// L1644
          packed_c14 = v1031;	// L1645
        } else {
          ap_int<128> v1032 = packed_tmp14;	// L1647
          int64_t v1033 = m14;	// L1648
          int64_t v1034 = v1033 * 8;	// L1649
          ap_int<34> v1035 = m14;	// L1650
          ap_int<34> v1036 = v1035 + 1;	// L1651
          ap_int<66> v1037 = v1036;	// L1652
          ap_int<66> v1038 = v1037 * 8;	// L1653
          ap_int<66> v1039 = v1038 - 1;	// L1654
          int v1040 = v1034;	// L1655
          int v1041 = v1039;	// L1656
          int8_t v1042;
          ap_int<128> v1042_tmp = v1032;
          v1042 = v1042_tmp(v1041, v1040);	// L1657
          ap_int<128> v1043 = packed_c14;	// L1658
          ap_int<128> v1044;
          ap_int<128> v1044_tmp = v1043;
          v1044_tmp(v1041, v1040) = v1042;
          v1044 = v1044_tmp;	// L1668
          packed_c14 = v1044;	// L1669
        }
      }
      ap_int<128> v1045 = packed_c14;	// L1672
      v994.write(v1045);	// L1673
    }
  }
}

void gemm_1_16(
  hls::stream< int8_t >& v1046,
  hls::stream< int8_t >& v1047,
  hls::stream< int8_t >& v1048,
  hls::stream< ap_uint<128> >& v1049
) {	// L1678
  l_S_mt_nt_0_mt35: for (int mt35 = 0; mt35 < 4; mt35++) {	// L1686
    l_nt35: for (int nt35 = 0; nt35 < 4; nt35++) {	// L1687
      int8_t c15;	// L1688
      c15 = 0;	// L1689
      l_S_k_0_k35: for (int k35 = 0; k35 < 64; k35++) {	// L1690
        int8_t v1054 = v1046.read();	// L1691
        int8_t a16;	// L1692
        a16 = v1054;	// L1693
        int8_t v1056 = v1047.read();	// L1694
        int8_t b31;	// L1695
        b31 = v1056;	// L1696
        int8_t v1058 = a16;	// L1697
        int8_t v1059 = b31;	// L1698
        int16_t v1060 = v1058;	// L1699
        int16_t v1061 = v1059;	// L1700
        int16_t v1062 = v1060 * v1061;	// L1701
        int8_t v1063 = c15;	// L1702
        ap_int<17> v1064 = v1063;	// L1703
        ap_int<17> v1065 = v1062;	// L1704
        ap_int<17> v1066 = v1064 + v1065;	// L1705
        int8_t v1067 = v1066;	// L1706
        c15 = v1067;	// L1707
        int8_t v1068 = b31;	// L1708
        v1048.write(v1068);	// L1709
      }
      ap_uint<128> packed_tmp15;	// L1711
      packed_tmp15 = 0;	// L1712
      ap_uint<128> packed_c15;	// L1713
      packed_c15 = 0;	// L1714
      l_S_m_1_m15: for (int m15 = 0; m15 < 16; m15++) {	// L1715
        ap_int<33> v1072 = m15;	// L1716
        bool v1073 = v1072 == 0;	// L1717
        if (v1073) {	// L1718
          int8_t v1074 = c15;	// L1719
          ap_int<128> v1075 = packed_c15;	// L1720
          int64_t v1076 = m15;	// L1721
          int64_t v1077 = v1076 * 8;	// L1722
          ap_int<34> v1078 = m15;	// L1723
          ap_int<34> v1079 = v1078 + 1;	// L1724
          ap_int<66> v1080 = v1079;	// L1725
          ap_int<66> v1081 = v1080 * 8;	// L1726
          ap_int<66> v1082 = v1081 - 1;	// L1727
          int v1083 = v1077;	// L1728
          int v1084 = v1082;	// L1729
          ap_int<128> v1085;
          ap_int<128> v1085_tmp = v1075;
          v1085_tmp(v1084, v1083) = v1074;
          v1085 = v1085_tmp;	// L1730
          packed_c15 = v1085;	// L1731
        } else {
          ap_int<128> v1086 = packed_tmp15;	// L1733
          int64_t v1087 = m15;	// L1734
          int64_t v1088 = v1087 * 8;	// L1735
          ap_int<34> v1089 = m15;	// L1736
          ap_int<34> v1090 = v1089 + 1;	// L1737
          ap_int<66> v1091 = v1090;	// L1738
          ap_int<66> v1092 = v1091 * 8;	// L1739
          ap_int<66> v1093 = v1092 - 1;	// L1740
          int v1094 = v1088;	// L1741
          int v1095 = v1093;	// L1742
          int8_t v1096;
          ap_int<128> v1096_tmp = v1086;
          v1096 = v1096_tmp(v1095, v1094);	// L1743
          ap_int<128> v1097 = packed_c15;	// L1744
          ap_int<128> v1098;
          ap_int<128> v1098_tmp = v1097;
          v1098_tmp(v1095, v1094) = v1096;
          v1098 = v1098_tmp;	// L1754
          packed_c15 = v1098;	// L1755
        }
      }
      ap_int<128> v1099 = packed_c15;	// L1758
      v1049.write(v1099);	// L1759
    }
  }
}

void gemm_2_0(
  hls::stream< ap_uint<128> >& v1100,
  hls::stream< int8_t >& v1101,
  hls::stream< ap_uint<128> >& v1102
) {	// L1764
  l_S_mt_nt_0_mt36: for (int mt36 = 0; mt36 < 4; mt36++) {	// L1767
    l_nt36: for (int nt36 = 0; nt36 < 4; nt36++) {	// L1768
      l_S_k_0_k36: for (int k36 = 0; k36 < 64; k36++) {	// L1769
        ap_uint<128> v1106 = v1100.read();	// L1770
        ap_uint<128> a17;	// L1771
        a17 = v1106;	// L1772
        ap_int<128> v1108 = a17;	// L1773
        int8_t v1109;
        ap_int<128> v1109_tmp = v1108;
        v1109 = v1109_tmp(15, 8);	// L1774
        v1101.write(v1109);	// L1775
        ap_int<128> v1110 = a17;	// L1776
        v1102.write(v1110);	// L1777
      }
    }
  }
}

void gemm_2_1(
  hls::stream< int8_t >& v1111,
  hls::stream< int8_t >& v1112,
  hls::stream< int8_t >& v1113,
  hls::stream< int8_t >& v1114,
  hls::stream< ap_uint<128> >& v1115,
  hls::stream< ap_uint<128> >& v1116
) {	// L1783
  l_S_mt_nt_0_mt37: for (int mt37 = 0; mt37 < 4; mt37++) {	// L1791
    l_nt37: for (int nt37 = 0; nt37 < 4; nt37++) {	// L1792
      int8_t c16;	// L1793
      c16 = 0;	// L1794
      l_S_k_0_k37: for (int k37 = 0; k37 < 64; k37++) {	// L1795
        int8_t v1121 = v1111.read();	// L1796
        int8_t a18;	// L1797
        a18 = v1121;	// L1798
        int8_t v1123 = v1112.read();	// L1799
        int8_t b32;	// L1800
        b32 = v1123;	// L1801
        int8_t v1125 = a18;	// L1802
        int8_t v1126 = b32;	// L1803
        int16_t v1127 = v1125;	// L1804
        int16_t v1128 = v1126;	// L1805
        int16_t v1129 = v1127 * v1128;	// L1806
        int8_t v1130 = c16;	// L1807
        ap_int<17> v1131 = v1130;	// L1808
        ap_int<17> v1132 = v1129;	// L1809
        ap_int<17> v1133 = v1131 + v1132;	// L1810
        int8_t v1134 = v1133;	// L1811
        c16 = v1134;	// L1812
        int8_t v1135 = a18;	// L1813
        v1113.write(v1135);	// L1814
        int8_t v1136 = b32;	// L1815
        v1114.write(v1136);	// L1816
      }
      ap_uint<128> packed_tmp16;	// L1818
      ap_uint<128> v1138 = v1115.read();	// L1819
      packed_tmp16 = v1138;	// L1820
      ap_uint<128> packed_c16;	// L1821
      packed_c16 = 0;	// L1822
      l_S_m_1_m16: for (int m16 = 0; m16 < 16; m16++) {	// L1823
        ap_int<33> v1141 = m16;	// L1824
        bool v1142 = v1141 == 1;	// L1825
        if (v1142) {	// L1826
          int8_t v1143 = c16;	// L1827
          ap_int<128> v1144 = packed_c16;	// L1828
          int64_t v1145 = m16;	// L1829
          int64_t v1146 = v1145 * 8;	// L1830
          ap_int<34> v1147 = m16;	// L1831
          ap_int<34> v1148 = v1147 + 1;	// L1832
          ap_int<66> v1149 = v1148;	// L1833
          ap_int<66> v1150 = v1149 * 8;	// L1834
          ap_int<66> v1151 = v1150 - 1;	// L1835
          int v1152 = v1146;	// L1836
          int v1153 = v1151;	// L1837
          ap_int<128> v1154;
          ap_int<128> v1154_tmp = v1144;
          v1154_tmp(v1153, v1152) = v1143;
          v1154 = v1154_tmp;	// L1838
          packed_c16 = v1154;	// L1839
        } else {
          ap_int<128> v1155 = packed_tmp16;	// L1841
          int64_t v1156 = m16;	// L1842
          int64_t v1157 = v1156 * 8;	// L1843
          ap_int<34> v1158 = m16;	// L1844
          ap_int<34> v1159 = v1158 + 1;	// L1845
          ap_int<66> v1160 = v1159;	// L1846
          ap_int<66> v1161 = v1160 * 8;	// L1847
          ap_int<66> v1162 = v1161 - 1;	// L1848
          int v1163 = v1157;	// L1849
          int v1164 = v1162;	// L1850
          int8_t v1165;
          ap_int<128> v1165_tmp = v1155;
          v1165 = v1165_tmp(v1164, v1163);	// L1851
          ap_int<128> v1166 = packed_c16;	// L1852
          ap_int<128> v1167;
          ap_int<128> v1167_tmp = v1166;
          v1167_tmp(v1164, v1163) = v1165;
          v1167 = v1167_tmp;	// L1862
          packed_c16 = v1167;	// L1863
        }
      }
      ap_int<128> v1168 = packed_c16;	// L1866
      v1116.write(v1168);	// L1867
    }
  }
}

void gemm_2_2(
  hls::stream< int8_t >& v1169,
  hls::stream< int8_t >& v1170,
  hls::stream< int8_t >& v1171,
  hls::stream< int8_t >& v1172,
  hls::stream< ap_uint<128> >& v1173,
  hls::stream< ap_uint<128> >& v1174
) {	// L1872
  l_S_mt_nt_0_mt38: for (int mt38 = 0; mt38 < 4; mt38++) {	// L1880
    l_nt38: for (int nt38 = 0; nt38 < 4; nt38++) {	// L1881
      int8_t c17;	// L1882
      c17 = 0;	// L1883
      l_S_k_0_k38: for (int k38 = 0; k38 < 64; k38++) {	// L1884
        int8_t v1179 = v1169.read();	// L1885
        int8_t a19;	// L1886
        a19 = v1179;	// L1887
        int8_t v1181 = v1170.read();	// L1888
        int8_t b33;	// L1889
        b33 = v1181;	// L1890
        int8_t v1183 = a19;	// L1891
        int8_t v1184 = b33;	// L1892
        int16_t v1185 = v1183;	// L1893
        int16_t v1186 = v1184;	// L1894
        int16_t v1187 = v1185 * v1186;	// L1895
        int8_t v1188 = c17;	// L1896
        ap_int<17> v1189 = v1188;	// L1897
        ap_int<17> v1190 = v1187;	// L1898
        ap_int<17> v1191 = v1189 + v1190;	// L1899
        int8_t v1192 = v1191;	// L1900
        c17 = v1192;	// L1901
        int8_t v1193 = a19;	// L1902
        v1171.write(v1193);	// L1903
        int8_t v1194 = b33;	// L1904
        v1172.write(v1194);	// L1905
      }
      ap_uint<128> packed_tmp17;	// L1907
      ap_uint<128> v1196 = v1173.read();	// L1908
      packed_tmp17 = v1196;	// L1909
      ap_uint<128> packed_c17;	// L1910
      packed_c17 = 0;	// L1911
      l_S_m_1_m17: for (int m17 = 0; m17 < 16; m17++) {	// L1912
        ap_int<33> v1199 = m17;	// L1913
        bool v1200 = v1199 == 1;	// L1914
        if (v1200) {	// L1915
          int8_t v1201 = c17;	// L1916
          ap_int<128> v1202 = packed_c17;	// L1917
          int64_t v1203 = m17;	// L1918
          int64_t v1204 = v1203 * 8;	// L1919
          ap_int<34> v1205 = m17;	// L1920
          ap_int<34> v1206 = v1205 + 1;	// L1921
          ap_int<66> v1207 = v1206;	// L1922
          ap_int<66> v1208 = v1207 * 8;	// L1923
          ap_int<66> v1209 = v1208 - 1;	// L1924
          int v1210 = v1204;	// L1925
          int v1211 = v1209;	// L1926
          ap_int<128> v1212;
          ap_int<128> v1212_tmp = v1202;
          v1212_tmp(v1211, v1210) = v1201;
          v1212 = v1212_tmp;	// L1927
          packed_c17 = v1212;	// L1928
        } else {
          ap_int<128> v1213 = packed_tmp17;	// L1930
          int64_t v1214 = m17;	// L1931
          int64_t v1215 = v1214 * 8;	// L1932
          ap_int<34> v1216 = m17;	// L1933
          ap_int<34> v1217 = v1216 + 1;	// L1934
          ap_int<66> v1218 = v1217;	// L1935
          ap_int<66> v1219 = v1218 * 8;	// L1936
          ap_int<66> v1220 = v1219 - 1;	// L1937
          int v1221 = v1215;	// L1938
          int v1222 = v1220;	// L1939
          int8_t v1223;
          ap_int<128> v1223_tmp = v1213;
          v1223 = v1223_tmp(v1222, v1221);	// L1940
          ap_int<128> v1224 = packed_c17;	// L1941
          ap_int<128> v1225;
          ap_int<128> v1225_tmp = v1224;
          v1225_tmp(v1222, v1221) = v1223;
          v1225 = v1225_tmp;	// L1951
          packed_c17 = v1225;	// L1952
        }
      }
      ap_int<128> v1226 = packed_c17;	// L1955
      v1174.write(v1226);	// L1956
    }
  }
}

void gemm_2_3(
  hls::stream< int8_t >& v1227,
  hls::stream< int8_t >& v1228,
  hls::stream< int8_t >& v1229,
  hls::stream< int8_t >& v1230,
  hls::stream< ap_uint<128> >& v1231,
  hls::stream< ap_uint<128> >& v1232
) {	// L1961
  l_S_mt_nt_0_mt39: for (int mt39 = 0; mt39 < 4; mt39++) {	// L1969
    l_nt39: for (int nt39 = 0; nt39 < 4; nt39++) {	// L1970
      int8_t c18;	// L1971
      c18 = 0;	// L1972
      l_S_k_0_k39: for (int k39 = 0; k39 < 64; k39++) {	// L1973
        int8_t v1237 = v1227.read();	// L1974
        int8_t a20;	// L1975
        a20 = v1237;	// L1976
        int8_t v1239 = v1228.read();	// L1977
        int8_t b34;	// L1978
        b34 = v1239;	// L1979
        int8_t v1241 = a20;	// L1980
        int8_t v1242 = b34;	// L1981
        int16_t v1243 = v1241;	// L1982
        int16_t v1244 = v1242;	// L1983
        int16_t v1245 = v1243 * v1244;	// L1984
        int8_t v1246 = c18;	// L1985
        ap_int<17> v1247 = v1246;	// L1986
        ap_int<17> v1248 = v1245;	// L1987
        ap_int<17> v1249 = v1247 + v1248;	// L1988
        int8_t v1250 = v1249;	// L1989
        c18 = v1250;	// L1990
        int8_t v1251 = a20;	// L1991
        v1229.write(v1251);	// L1992
        int8_t v1252 = b34;	// L1993
        v1230.write(v1252);	// L1994
      }
      ap_uint<128> packed_tmp18;	// L1996
      ap_uint<128> v1254 = v1231.read();	// L1997
      packed_tmp18 = v1254;	// L1998
      ap_uint<128> packed_c18;	// L1999
      packed_c18 = 0;	// L2000
      l_S_m_1_m18: for (int m18 = 0; m18 < 16; m18++) {	// L2001
        ap_int<33> v1257 = m18;	// L2002
        bool v1258 = v1257 == 1;	// L2003
        if (v1258) {	// L2004
          int8_t v1259 = c18;	// L2005
          ap_int<128> v1260 = packed_c18;	// L2006
          int64_t v1261 = m18;	// L2007
          int64_t v1262 = v1261 * 8;	// L2008
          ap_int<34> v1263 = m18;	// L2009
          ap_int<34> v1264 = v1263 + 1;	// L2010
          ap_int<66> v1265 = v1264;	// L2011
          ap_int<66> v1266 = v1265 * 8;	// L2012
          ap_int<66> v1267 = v1266 - 1;	// L2013
          int v1268 = v1262;	// L2014
          int v1269 = v1267;	// L2015
          ap_int<128> v1270;
          ap_int<128> v1270_tmp = v1260;
          v1270_tmp(v1269, v1268) = v1259;
          v1270 = v1270_tmp;	// L2016
          packed_c18 = v1270;	// L2017
        } else {
          ap_int<128> v1271 = packed_tmp18;	// L2019
          int64_t v1272 = m18;	// L2020
          int64_t v1273 = v1272 * 8;	// L2021
          ap_int<34> v1274 = m18;	// L2022
          ap_int<34> v1275 = v1274 + 1;	// L2023
          ap_int<66> v1276 = v1275;	// L2024
          ap_int<66> v1277 = v1276 * 8;	// L2025
          ap_int<66> v1278 = v1277 - 1;	// L2026
          int v1279 = v1273;	// L2027
          int v1280 = v1278;	// L2028
          int8_t v1281;
          ap_int<128> v1281_tmp = v1271;
          v1281 = v1281_tmp(v1280, v1279);	// L2029
          ap_int<128> v1282 = packed_c18;	// L2030
          ap_int<128> v1283;
          ap_int<128> v1283_tmp = v1282;
          v1283_tmp(v1280, v1279) = v1281;
          v1283 = v1283_tmp;	// L2040
          packed_c18 = v1283;	// L2041
        }
      }
      ap_int<128> v1284 = packed_c18;	// L2044
      v1232.write(v1284);	// L2045
    }
  }
}

void gemm_2_4(
  hls::stream< int8_t >& v1285,
  hls::stream< int8_t >& v1286,
  hls::stream< int8_t >& v1287,
  hls::stream< int8_t >& v1288,
  hls::stream< ap_uint<128> >& v1289,
  hls::stream< ap_uint<128> >& v1290
) {	// L2050
  l_S_mt_nt_0_mt40: for (int mt40 = 0; mt40 < 4; mt40++) {	// L2058
    l_nt40: for (int nt40 = 0; nt40 < 4; nt40++) {	// L2059
      int8_t c19;	// L2060
      c19 = 0;	// L2061
      l_S_k_0_k40: for (int k40 = 0; k40 < 64; k40++) {	// L2062
        int8_t v1295 = v1285.read();	// L2063
        int8_t a21;	// L2064
        a21 = v1295;	// L2065
        int8_t v1297 = v1286.read();	// L2066
        int8_t b35;	// L2067
        b35 = v1297;	// L2068
        int8_t v1299 = a21;	// L2069
        int8_t v1300 = b35;	// L2070
        int16_t v1301 = v1299;	// L2071
        int16_t v1302 = v1300;	// L2072
        int16_t v1303 = v1301 * v1302;	// L2073
        int8_t v1304 = c19;	// L2074
        ap_int<17> v1305 = v1304;	// L2075
        ap_int<17> v1306 = v1303;	// L2076
        ap_int<17> v1307 = v1305 + v1306;	// L2077
        int8_t v1308 = v1307;	// L2078
        c19 = v1308;	// L2079
        int8_t v1309 = a21;	// L2080
        v1287.write(v1309);	// L2081
        int8_t v1310 = b35;	// L2082
        v1288.write(v1310);	// L2083
      }
      ap_uint<128> packed_tmp19;	// L2085
      ap_uint<128> v1312 = v1289.read();	// L2086
      packed_tmp19 = v1312;	// L2087
      ap_uint<128> packed_c19;	// L2088
      packed_c19 = 0;	// L2089
      l_S_m_1_m19: for (int m19 = 0; m19 < 16; m19++) {	// L2090
        ap_int<33> v1315 = m19;	// L2091
        bool v1316 = v1315 == 1;	// L2092
        if (v1316) {	// L2093
          int8_t v1317 = c19;	// L2094
          ap_int<128> v1318 = packed_c19;	// L2095
          int64_t v1319 = m19;	// L2096
          int64_t v1320 = v1319 * 8;	// L2097
          ap_int<34> v1321 = m19;	// L2098
          ap_int<34> v1322 = v1321 + 1;	// L2099
          ap_int<66> v1323 = v1322;	// L2100
          ap_int<66> v1324 = v1323 * 8;	// L2101
          ap_int<66> v1325 = v1324 - 1;	// L2102
          int v1326 = v1320;	// L2103
          int v1327 = v1325;	// L2104
          ap_int<128> v1328;
          ap_int<128> v1328_tmp = v1318;
          v1328_tmp(v1327, v1326) = v1317;
          v1328 = v1328_tmp;	// L2105
          packed_c19 = v1328;	// L2106
        } else {
          ap_int<128> v1329 = packed_tmp19;	// L2108
          int64_t v1330 = m19;	// L2109
          int64_t v1331 = v1330 * 8;	// L2110
          ap_int<34> v1332 = m19;	// L2111
          ap_int<34> v1333 = v1332 + 1;	// L2112
          ap_int<66> v1334 = v1333;	// L2113
          ap_int<66> v1335 = v1334 * 8;	// L2114
          ap_int<66> v1336 = v1335 - 1;	// L2115
          int v1337 = v1331;	// L2116
          int v1338 = v1336;	// L2117
          int8_t v1339;
          ap_int<128> v1339_tmp = v1329;
          v1339 = v1339_tmp(v1338, v1337);	// L2118
          ap_int<128> v1340 = packed_c19;	// L2119
          ap_int<128> v1341;
          ap_int<128> v1341_tmp = v1340;
          v1341_tmp(v1338, v1337) = v1339;
          v1341 = v1341_tmp;	// L2129
          packed_c19 = v1341;	// L2130
        }
      }
      ap_int<128> v1342 = packed_c19;	// L2133
      v1290.write(v1342);	// L2134
    }
  }
}

void gemm_2_5(
  hls::stream< int8_t >& v1343,
  hls::stream< int8_t >& v1344,
  hls::stream< int8_t >& v1345,
  hls::stream< int8_t >& v1346,
  hls::stream< ap_uint<128> >& v1347,
  hls::stream< ap_uint<128> >& v1348
) {	// L2139
  l_S_mt_nt_0_mt41: for (int mt41 = 0; mt41 < 4; mt41++) {	// L2147
    l_nt41: for (int nt41 = 0; nt41 < 4; nt41++) {	// L2148
      int8_t c20;	// L2149
      c20 = 0;	// L2150
      l_S_k_0_k41: for (int k41 = 0; k41 < 64; k41++) {	// L2151
        int8_t v1353 = v1343.read();	// L2152
        int8_t a22;	// L2153
        a22 = v1353;	// L2154
        int8_t v1355 = v1344.read();	// L2155
        int8_t b36;	// L2156
        b36 = v1355;	// L2157
        int8_t v1357 = a22;	// L2158
        int8_t v1358 = b36;	// L2159
        int16_t v1359 = v1357;	// L2160
        int16_t v1360 = v1358;	// L2161
        int16_t v1361 = v1359 * v1360;	// L2162
        int8_t v1362 = c20;	// L2163
        ap_int<17> v1363 = v1362;	// L2164
        ap_int<17> v1364 = v1361;	// L2165
        ap_int<17> v1365 = v1363 + v1364;	// L2166
        int8_t v1366 = v1365;	// L2167
        c20 = v1366;	// L2168
        int8_t v1367 = a22;	// L2169
        v1345.write(v1367);	// L2170
        int8_t v1368 = b36;	// L2171
        v1346.write(v1368);	// L2172
      }
      ap_uint<128> packed_tmp20;	// L2174
      ap_uint<128> v1370 = v1347.read();	// L2175
      packed_tmp20 = v1370;	// L2176
      ap_uint<128> packed_c20;	// L2177
      packed_c20 = 0;	// L2178
      l_S_m_1_m20: for (int m20 = 0; m20 < 16; m20++) {	// L2179
        ap_int<33> v1373 = m20;	// L2180
        bool v1374 = v1373 == 1;	// L2181
        if (v1374) {	// L2182
          int8_t v1375 = c20;	// L2183
          ap_int<128> v1376 = packed_c20;	// L2184
          int64_t v1377 = m20;	// L2185
          int64_t v1378 = v1377 * 8;	// L2186
          ap_int<34> v1379 = m20;	// L2187
          ap_int<34> v1380 = v1379 + 1;	// L2188
          ap_int<66> v1381 = v1380;	// L2189
          ap_int<66> v1382 = v1381 * 8;	// L2190
          ap_int<66> v1383 = v1382 - 1;	// L2191
          int v1384 = v1378;	// L2192
          int v1385 = v1383;	// L2193
          ap_int<128> v1386;
          ap_int<128> v1386_tmp = v1376;
          v1386_tmp(v1385, v1384) = v1375;
          v1386 = v1386_tmp;	// L2194
          packed_c20 = v1386;	// L2195
        } else {
          ap_int<128> v1387 = packed_tmp20;	// L2197
          int64_t v1388 = m20;	// L2198
          int64_t v1389 = v1388 * 8;	// L2199
          ap_int<34> v1390 = m20;	// L2200
          ap_int<34> v1391 = v1390 + 1;	// L2201
          ap_int<66> v1392 = v1391;	// L2202
          ap_int<66> v1393 = v1392 * 8;	// L2203
          ap_int<66> v1394 = v1393 - 1;	// L2204
          int v1395 = v1389;	// L2205
          int v1396 = v1394;	// L2206
          int8_t v1397;
          ap_int<128> v1397_tmp = v1387;
          v1397 = v1397_tmp(v1396, v1395);	// L2207
          ap_int<128> v1398 = packed_c20;	// L2208
          ap_int<128> v1399;
          ap_int<128> v1399_tmp = v1398;
          v1399_tmp(v1396, v1395) = v1397;
          v1399 = v1399_tmp;	// L2218
          packed_c20 = v1399;	// L2219
        }
      }
      ap_int<128> v1400 = packed_c20;	// L2222
      v1348.write(v1400);	// L2223
    }
  }
}

void gemm_2_6(
  hls::stream< int8_t >& v1401,
  hls::stream< int8_t >& v1402,
  hls::stream< int8_t >& v1403,
  hls::stream< int8_t >& v1404,
  hls::stream< ap_uint<128> >& v1405,
  hls::stream< ap_uint<128> >& v1406
) {	// L2228
  l_S_mt_nt_0_mt42: for (int mt42 = 0; mt42 < 4; mt42++) {	// L2236
    l_nt42: for (int nt42 = 0; nt42 < 4; nt42++) {	// L2237
      int8_t c21;	// L2238
      c21 = 0;	// L2239
      l_S_k_0_k42: for (int k42 = 0; k42 < 64; k42++) {	// L2240
        int8_t v1411 = v1401.read();	// L2241
        int8_t a23;	// L2242
        a23 = v1411;	// L2243
        int8_t v1413 = v1402.read();	// L2244
        int8_t b37;	// L2245
        b37 = v1413;	// L2246
        int8_t v1415 = a23;	// L2247
        int8_t v1416 = b37;	// L2248
        int16_t v1417 = v1415;	// L2249
        int16_t v1418 = v1416;	// L2250
        int16_t v1419 = v1417 * v1418;	// L2251
        int8_t v1420 = c21;	// L2252
        ap_int<17> v1421 = v1420;	// L2253
        ap_int<17> v1422 = v1419;	// L2254
        ap_int<17> v1423 = v1421 + v1422;	// L2255
        int8_t v1424 = v1423;	// L2256
        c21 = v1424;	// L2257
        int8_t v1425 = a23;	// L2258
        v1403.write(v1425);	// L2259
        int8_t v1426 = b37;	// L2260
        v1404.write(v1426);	// L2261
      }
      ap_uint<128> packed_tmp21;	// L2263
      ap_uint<128> v1428 = v1405.read();	// L2264
      packed_tmp21 = v1428;	// L2265
      ap_uint<128> packed_c21;	// L2266
      packed_c21 = 0;	// L2267
      l_S_m_1_m21: for (int m21 = 0; m21 < 16; m21++) {	// L2268
        ap_int<33> v1431 = m21;	// L2269
        bool v1432 = v1431 == 1;	// L2270
        if (v1432) {	// L2271
          int8_t v1433 = c21;	// L2272
          ap_int<128> v1434 = packed_c21;	// L2273
          int64_t v1435 = m21;	// L2274
          int64_t v1436 = v1435 * 8;	// L2275
          ap_int<34> v1437 = m21;	// L2276
          ap_int<34> v1438 = v1437 + 1;	// L2277
          ap_int<66> v1439 = v1438;	// L2278
          ap_int<66> v1440 = v1439 * 8;	// L2279
          ap_int<66> v1441 = v1440 - 1;	// L2280
          int v1442 = v1436;	// L2281
          int v1443 = v1441;	// L2282
          ap_int<128> v1444;
          ap_int<128> v1444_tmp = v1434;
          v1444_tmp(v1443, v1442) = v1433;
          v1444 = v1444_tmp;	// L2283
          packed_c21 = v1444;	// L2284
        } else {
          ap_int<128> v1445 = packed_tmp21;	// L2286
          int64_t v1446 = m21;	// L2287
          int64_t v1447 = v1446 * 8;	// L2288
          ap_int<34> v1448 = m21;	// L2289
          ap_int<34> v1449 = v1448 + 1;	// L2290
          ap_int<66> v1450 = v1449;	// L2291
          ap_int<66> v1451 = v1450 * 8;	// L2292
          ap_int<66> v1452 = v1451 - 1;	// L2293
          int v1453 = v1447;	// L2294
          int v1454 = v1452;	// L2295
          int8_t v1455;
          ap_int<128> v1455_tmp = v1445;
          v1455 = v1455_tmp(v1454, v1453);	// L2296
          ap_int<128> v1456 = packed_c21;	// L2297
          ap_int<128> v1457;
          ap_int<128> v1457_tmp = v1456;
          v1457_tmp(v1454, v1453) = v1455;
          v1457 = v1457_tmp;	// L2307
          packed_c21 = v1457;	// L2308
        }
      }
      ap_int<128> v1458 = packed_c21;	// L2311
      v1406.write(v1458);	// L2312
    }
  }
}

void gemm_2_7(
  hls::stream< int8_t >& v1459,
  hls::stream< int8_t >& v1460,
  hls::stream< int8_t >& v1461,
  hls::stream< int8_t >& v1462,
  hls::stream< ap_uint<128> >& v1463,
  hls::stream< ap_uint<128> >& v1464
) {	// L2317
  l_S_mt_nt_0_mt43: for (int mt43 = 0; mt43 < 4; mt43++) {	// L2325
    l_nt43: for (int nt43 = 0; nt43 < 4; nt43++) {	// L2326
      int8_t c22;	// L2327
      c22 = 0;	// L2328
      l_S_k_0_k43: for (int k43 = 0; k43 < 64; k43++) {	// L2329
        int8_t v1469 = v1459.read();	// L2330
        int8_t a24;	// L2331
        a24 = v1469;	// L2332
        int8_t v1471 = v1460.read();	// L2333
        int8_t b38;	// L2334
        b38 = v1471;	// L2335
        int8_t v1473 = a24;	// L2336
        int8_t v1474 = b38;	// L2337
        int16_t v1475 = v1473;	// L2338
        int16_t v1476 = v1474;	// L2339
        int16_t v1477 = v1475 * v1476;	// L2340
        int8_t v1478 = c22;	// L2341
        ap_int<17> v1479 = v1478;	// L2342
        ap_int<17> v1480 = v1477;	// L2343
        ap_int<17> v1481 = v1479 + v1480;	// L2344
        int8_t v1482 = v1481;	// L2345
        c22 = v1482;	// L2346
        int8_t v1483 = a24;	// L2347
        v1461.write(v1483);	// L2348
        int8_t v1484 = b38;	// L2349
        v1462.write(v1484);	// L2350
      }
      ap_uint<128> packed_tmp22;	// L2352
      ap_uint<128> v1486 = v1463.read();	// L2353
      packed_tmp22 = v1486;	// L2354
      ap_uint<128> packed_c22;	// L2355
      packed_c22 = 0;	// L2356
      l_S_m_1_m22: for (int m22 = 0; m22 < 16; m22++) {	// L2357
        ap_int<33> v1489 = m22;	// L2358
        bool v1490 = v1489 == 1;	// L2359
        if (v1490) {	// L2360
          int8_t v1491 = c22;	// L2361
          ap_int<128> v1492 = packed_c22;	// L2362
          int64_t v1493 = m22;	// L2363
          int64_t v1494 = v1493 * 8;	// L2364
          ap_int<34> v1495 = m22;	// L2365
          ap_int<34> v1496 = v1495 + 1;	// L2366
          ap_int<66> v1497 = v1496;	// L2367
          ap_int<66> v1498 = v1497 * 8;	// L2368
          ap_int<66> v1499 = v1498 - 1;	// L2369
          int v1500 = v1494;	// L2370
          int v1501 = v1499;	// L2371
          ap_int<128> v1502;
          ap_int<128> v1502_tmp = v1492;
          v1502_tmp(v1501, v1500) = v1491;
          v1502 = v1502_tmp;	// L2372
          packed_c22 = v1502;	// L2373
        } else {
          ap_int<128> v1503 = packed_tmp22;	// L2375
          int64_t v1504 = m22;	// L2376
          int64_t v1505 = v1504 * 8;	// L2377
          ap_int<34> v1506 = m22;	// L2378
          ap_int<34> v1507 = v1506 + 1;	// L2379
          ap_int<66> v1508 = v1507;	// L2380
          ap_int<66> v1509 = v1508 * 8;	// L2381
          ap_int<66> v1510 = v1509 - 1;	// L2382
          int v1511 = v1505;	// L2383
          int v1512 = v1510;	// L2384
          int8_t v1513;
          ap_int<128> v1513_tmp = v1503;
          v1513 = v1513_tmp(v1512, v1511);	// L2385
          ap_int<128> v1514 = packed_c22;	// L2386
          ap_int<128> v1515;
          ap_int<128> v1515_tmp = v1514;
          v1515_tmp(v1512, v1511) = v1513;
          v1515 = v1515_tmp;	// L2396
          packed_c22 = v1515;	// L2397
        }
      }
      ap_int<128> v1516 = packed_c22;	// L2400
      v1464.write(v1516);	// L2401
    }
  }
}

void gemm_2_8(
  hls::stream< int8_t >& v1517,
  hls::stream< int8_t >& v1518,
  hls::stream< int8_t >& v1519,
  hls::stream< int8_t >& v1520,
  hls::stream< ap_uint<128> >& v1521,
  hls::stream< ap_uint<128> >& v1522
) {	// L2406
  l_S_mt_nt_0_mt44: for (int mt44 = 0; mt44 < 4; mt44++) {	// L2414
    l_nt44: for (int nt44 = 0; nt44 < 4; nt44++) {	// L2415
      int8_t c23;	// L2416
      c23 = 0;	// L2417
      l_S_k_0_k44: for (int k44 = 0; k44 < 64; k44++) {	// L2418
        int8_t v1527 = v1517.read();	// L2419
        int8_t a25;	// L2420
        a25 = v1527;	// L2421
        int8_t v1529 = v1518.read();	// L2422
        int8_t b39;	// L2423
        b39 = v1529;	// L2424
        int8_t v1531 = a25;	// L2425
        int8_t v1532 = b39;	// L2426
        int16_t v1533 = v1531;	// L2427
        int16_t v1534 = v1532;	// L2428
        int16_t v1535 = v1533 * v1534;	// L2429
        int8_t v1536 = c23;	// L2430
        ap_int<17> v1537 = v1536;	// L2431
        ap_int<17> v1538 = v1535;	// L2432
        ap_int<17> v1539 = v1537 + v1538;	// L2433
        int8_t v1540 = v1539;	// L2434
        c23 = v1540;	// L2435
        int8_t v1541 = a25;	// L2436
        v1519.write(v1541);	// L2437
        int8_t v1542 = b39;	// L2438
        v1520.write(v1542);	// L2439
      }
      ap_uint<128> packed_tmp23;	// L2441
      ap_uint<128> v1544 = v1521.read();	// L2442
      packed_tmp23 = v1544;	// L2443
      ap_uint<128> packed_c23;	// L2444
      packed_c23 = 0;	// L2445
      l_S_m_1_m23: for (int m23 = 0; m23 < 16; m23++) {	// L2446
        ap_int<33> v1547 = m23;	// L2447
        bool v1548 = v1547 == 1;	// L2448
        if (v1548) {	// L2449
          int8_t v1549 = c23;	// L2450
          ap_int<128> v1550 = packed_c23;	// L2451
          int64_t v1551 = m23;	// L2452
          int64_t v1552 = v1551 * 8;	// L2453
          ap_int<34> v1553 = m23;	// L2454
          ap_int<34> v1554 = v1553 + 1;	// L2455
          ap_int<66> v1555 = v1554;	// L2456
          ap_int<66> v1556 = v1555 * 8;	// L2457
          ap_int<66> v1557 = v1556 - 1;	// L2458
          int v1558 = v1552;	// L2459
          int v1559 = v1557;	// L2460
          ap_int<128> v1560;
          ap_int<128> v1560_tmp = v1550;
          v1560_tmp(v1559, v1558) = v1549;
          v1560 = v1560_tmp;	// L2461
          packed_c23 = v1560;	// L2462
        } else {
          ap_int<128> v1561 = packed_tmp23;	// L2464
          int64_t v1562 = m23;	// L2465
          int64_t v1563 = v1562 * 8;	// L2466
          ap_int<34> v1564 = m23;	// L2467
          ap_int<34> v1565 = v1564 + 1;	// L2468
          ap_int<66> v1566 = v1565;	// L2469
          ap_int<66> v1567 = v1566 * 8;	// L2470
          ap_int<66> v1568 = v1567 - 1;	// L2471
          int v1569 = v1563;	// L2472
          int v1570 = v1568;	// L2473
          int8_t v1571;
          ap_int<128> v1571_tmp = v1561;
          v1571 = v1571_tmp(v1570, v1569);	// L2474
          ap_int<128> v1572 = packed_c23;	// L2475
          ap_int<128> v1573;
          ap_int<128> v1573_tmp = v1572;
          v1573_tmp(v1570, v1569) = v1571;
          v1573 = v1573_tmp;	// L2485
          packed_c23 = v1573;	// L2486
        }
      }
      ap_int<128> v1574 = packed_c23;	// L2489
      v1522.write(v1574);	// L2490
    }
  }
}

void gemm_2_9(
  hls::stream< int8_t >& v1575,
  hls::stream< int8_t >& v1576,
  hls::stream< int8_t >& v1577,
  hls::stream< int8_t >& v1578,
  hls::stream< ap_uint<128> >& v1579,
  hls::stream< ap_uint<128> >& v1580
) {	// L2495
  l_S_mt_nt_0_mt45: for (int mt45 = 0; mt45 < 4; mt45++) {	// L2503
    l_nt45: for (int nt45 = 0; nt45 < 4; nt45++) {	// L2504
      int8_t c24;	// L2505
      c24 = 0;	// L2506
      l_S_k_0_k45: for (int k45 = 0; k45 < 64; k45++) {	// L2507
        int8_t v1585 = v1575.read();	// L2508
        int8_t a26;	// L2509
        a26 = v1585;	// L2510
        int8_t v1587 = v1576.read();	// L2511
        int8_t b40;	// L2512
        b40 = v1587;	// L2513
        int8_t v1589 = a26;	// L2514
        int8_t v1590 = b40;	// L2515
        int16_t v1591 = v1589;	// L2516
        int16_t v1592 = v1590;	// L2517
        int16_t v1593 = v1591 * v1592;	// L2518
        int8_t v1594 = c24;	// L2519
        ap_int<17> v1595 = v1594;	// L2520
        ap_int<17> v1596 = v1593;	// L2521
        ap_int<17> v1597 = v1595 + v1596;	// L2522
        int8_t v1598 = v1597;	// L2523
        c24 = v1598;	// L2524
        int8_t v1599 = a26;	// L2525
        v1577.write(v1599);	// L2526
        int8_t v1600 = b40;	// L2527
        v1578.write(v1600);	// L2528
      }
      ap_uint<128> packed_tmp24;	// L2530
      ap_uint<128> v1602 = v1579.read();	// L2531
      packed_tmp24 = v1602;	// L2532
      ap_uint<128> packed_c24;	// L2533
      packed_c24 = 0;	// L2534
      l_S_m_1_m24: for (int m24 = 0; m24 < 16; m24++) {	// L2535
        ap_int<33> v1605 = m24;	// L2536
        bool v1606 = v1605 == 1;	// L2537
        if (v1606) {	// L2538
          int8_t v1607 = c24;	// L2539
          ap_int<128> v1608 = packed_c24;	// L2540
          int64_t v1609 = m24;	// L2541
          int64_t v1610 = v1609 * 8;	// L2542
          ap_int<34> v1611 = m24;	// L2543
          ap_int<34> v1612 = v1611 + 1;	// L2544
          ap_int<66> v1613 = v1612;	// L2545
          ap_int<66> v1614 = v1613 * 8;	// L2546
          ap_int<66> v1615 = v1614 - 1;	// L2547
          int v1616 = v1610;	// L2548
          int v1617 = v1615;	// L2549
          ap_int<128> v1618;
          ap_int<128> v1618_tmp = v1608;
          v1618_tmp(v1617, v1616) = v1607;
          v1618 = v1618_tmp;	// L2550
          packed_c24 = v1618;	// L2551
        } else {
          ap_int<128> v1619 = packed_tmp24;	// L2553
          int64_t v1620 = m24;	// L2554
          int64_t v1621 = v1620 * 8;	// L2555
          ap_int<34> v1622 = m24;	// L2556
          ap_int<34> v1623 = v1622 + 1;	// L2557
          ap_int<66> v1624 = v1623;	// L2558
          ap_int<66> v1625 = v1624 * 8;	// L2559
          ap_int<66> v1626 = v1625 - 1;	// L2560
          int v1627 = v1621;	// L2561
          int v1628 = v1626;	// L2562
          int8_t v1629;
          ap_int<128> v1629_tmp = v1619;
          v1629 = v1629_tmp(v1628, v1627);	// L2563
          ap_int<128> v1630 = packed_c24;	// L2564
          ap_int<128> v1631;
          ap_int<128> v1631_tmp = v1630;
          v1631_tmp(v1628, v1627) = v1629;
          v1631 = v1631_tmp;	// L2574
          packed_c24 = v1631;	// L2575
        }
      }
      ap_int<128> v1632 = packed_c24;	// L2578
      v1580.write(v1632);	// L2579
    }
  }
}

void gemm_2_10(
  hls::stream< int8_t >& v1633,
  hls::stream< int8_t >& v1634,
  hls::stream< int8_t >& v1635,
  hls::stream< int8_t >& v1636,
  hls::stream< ap_uint<128> >& v1637,
  hls::stream< ap_uint<128> >& v1638
) {	// L2584
  l_S_mt_nt_0_mt46: for (int mt46 = 0; mt46 < 4; mt46++) {	// L2592
    l_nt46: for (int nt46 = 0; nt46 < 4; nt46++) {	// L2593
      int8_t c25;	// L2594
      c25 = 0;	// L2595
      l_S_k_0_k46: for (int k46 = 0; k46 < 64; k46++) {	// L2596
        int8_t v1643 = v1633.read();	// L2597
        int8_t a27;	// L2598
        a27 = v1643;	// L2599
        int8_t v1645 = v1634.read();	// L2600
        int8_t b41;	// L2601
        b41 = v1645;	// L2602
        int8_t v1647 = a27;	// L2603
        int8_t v1648 = b41;	// L2604
        int16_t v1649 = v1647;	// L2605
        int16_t v1650 = v1648;	// L2606
        int16_t v1651 = v1649 * v1650;	// L2607
        int8_t v1652 = c25;	// L2608
        ap_int<17> v1653 = v1652;	// L2609
        ap_int<17> v1654 = v1651;	// L2610
        ap_int<17> v1655 = v1653 + v1654;	// L2611
        int8_t v1656 = v1655;	// L2612
        c25 = v1656;	// L2613
        int8_t v1657 = a27;	// L2614
        v1635.write(v1657);	// L2615
        int8_t v1658 = b41;	// L2616
        v1636.write(v1658);	// L2617
      }
      ap_uint<128> packed_tmp25;	// L2619
      ap_uint<128> v1660 = v1637.read();	// L2620
      packed_tmp25 = v1660;	// L2621
      ap_uint<128> packed_c25;	// L2622
      packed_c25 = 0;	// L2623
      l_S_m_1_m25: for (int m25 = 0; m25 < 16; m25++) {	// L2624
        ap_int<33> v1663 = m25;	// L2625
        bool v1664 = v1663 == 1;	// L2626
        if (v1664) {	// L2627
          int8_t v1665 = c25;	// L2628
          ap_int<128> v1666 = packed_c25;	// L2629
          int64_t v1667 = m25;	// L2630
          int64_t v1668 = v1667 * 8;	// L2631
          ap_int<34> v1669 = m25;	// L2632
          ap_int<34> v1670 = v1669 + 1;	// L2633
          ap_int<66> v1671 = v1670;	// L2634
          ap_int<66> v1672 = v1671 * 8;	// L2635
          ap_int<66> v1673 = v1672 - 1;	// L2636
          int v1674 = v1668;	// L2637
          int v1675 = v1673;	// L2638
          ap_int<128> v1676;
          ap_int<128> v1676_tmp = v1666;
          v1676_tmp(v1675, v1674) = v1665;
          v1676 = v1676_tmp;	// L2639
          packed_c25 = v1676;	// L2640
        } else {
          ap_int<128> v1677 = packed_tmp25;	// L2642
          int64_t v1678 = m25;	// L2643
          int64_t v1679 = v1678 * 8;	// L2644
          ap_int<34> v1680 = m25;	// L2645
          ap_int<34> v1681 = v1680 + 1;	// L2646
          ap_int<66> v1682 = v1681;	// L2647
          ap_int<66> v1683 = v1682 * 8;	// L2648
          ap_int<66> v1684 = v1683 - 1;	// L2649
          int v1685 = v1679;	// L2650
          int v1686 = v1684;	// L2651
          int8_t v1687;
          ap_int<128> v1687_tmp = v1677;
          v1687 = v1687_tmp(v1686, v1685);	// L2652
          ap_int<128> v1688 = packed_c25;	// L2653
          ap_int<128> v1689;
          ap_int<128> v1689_tmp = v1688;
          v1689_tmp(v1686, v1685) = v1687;
          v1689 = v1689_tmp;	// L2663
          packed_c25 = v1689;	// L2664
        }
      }
      ap_int<128> v1690 = packed_c25;	// L2667
      v1638.write(v1690);	// L2668
    }
  }
}

void gemm_2_11(
  hls::stream< int8_t >& v1691,
  hls::stream< int8_t >& v1692,
  hls::stream< int8_t >& v1693,
  hls::stream< int8_t >& v1694,
  hls::stream< ap_uint<128> >& v1695,
  hls::stream< ap_uint<128> >& v1696
) {	// L2673
  l_S_mt_nt_0_mt47: for (int mt47 = 0; mt47 < 4; mt47++) {	// L2681
    l_nt47: for (int nt47 = 0; nt47 < 4; nt47++) {	// L2682
      int8_t c26;	// L2683
      c26 = 0;	// L2684
      l_S_k_0_k47: for (int k47 = 0; k47 < 64; k47++) {	// L2685
        int8_t v1701 = v1691.read();	// L2686
        int8_t a28;	// L2687
        a28 = v1701;	// L2688
        int8_t v1703 = v1692.read();	// L2689
        int8_t b42;	// L2690
        b42 = v1703;	// L2691
        int8_t v1705 = a28;	// L2692
        int8_t v1706 = b42;	// L2693
        int16_t v1707 = v1705;	// L2694
        int16_t v1708 = v1706;	// L2695
        int16_t v1709 = v1707 * v1708;	// L2696
        int8_t v1710 = c26;	// L2697
        ap_int<17> v1711 = v1710;	// L2698
        ap_int<17> v1712 = v1709;	// L2699
        ap_int<17> v1713 = v1711 + v1712;	// L2700
        int8_t v1714 = v1713;	// L2701
        c26 = v1714;	// L2702
        int8_t v1715 = a28;	// L2703
        v1693.write(v1715);	// L2704
        int8_t v1716 = b42;	// L2705
        v1694.write(v1716);	// L2706
      }
      ap_uint<128> packed_tmp26;	// L2708
      ap_uint<128> v1718 = v1695.read();	// L2709
      packed_tmp26 = v1718;	// L2710
      ap_uint<128> packed_c26;	// L2711
      packed_c26 = 0;	// L2712
      l_S_m_1_m26: for (int m26 = 0; m26 < 16; m26++) {	// L2713
        ap_int<33> v1721 = m26;	// L2714
        bool v1722 = v1721 == 1;	// L2715
        if (v1722) {	// L2716
          int8_t v1723 = c26;	// L2717
          ap_int<128> v1724 = packed_c26;	// L2718
          int64_t v1725 = m26;	// L2719
          int64_t v1726 = v1725 * 8;	// L2720
          ap_int<34> v1727 = m26;	// L2721
          ap_int<34> v1728 = v1727 + 1;	// L2722
          ap_int<66> v1729 = v1728;	// L2723
          ap_int<66> v1730 = v1729 * 8;	// L2724
          ap_int<66> v1731 = v1730 - 1;	// L2725
          int v1732 = v1726;	// L2726
          int v1733 = v1731;	// L2727
          ap_int<128> v1734;
          ap_int<128> v1734_tmp = v1724;
          v1734_tmp(v1733, v1732) = v1723;
          v1734 = v1734_tmp;	// L2728
          packed_c26 = v1734;	// L2729
        } else {
          ap_int<128> v1735 = packed_tmp26;	// L2731
          int64_t v1736 = m26;	// L2732
          int64_t v1737 = v1736 * 8;	// L2733
          ap_int<34> v1738 = m26;	// L2734
          ap_int<34> v1739 = v1738 + 1;	// L2735
          ap_int<66> v1740 = v1739;	// L2736
          ap_int<66> v1741 = v1740 * 8;	// L2737
          ap_int<66> v1742 = v1741 - 1;	// L2738
          int v1743 = v1737;	// L2739
          int v1744 = v1742;	// L2740
          int8_t v1745;
          ap_int<128> v1745_tmp = v1735;
          v1745 = v1745_tmp(v1744, v1743);	// L2741
          ap_int<128> v1746 = packed_c26;	// L2742
          ap_int<128> v1747;
          ap_int<128> v1747_tmp = v1746;
          v1747_tmp(v1744, v1743) = v1745;
          v1747 = v1747_tmp;	// L2752
          packed_c26 = v1747;	// L2753
        }
      }
      ap_int<128> v1748 = packed_c26;	// L2756
      v1696.write(v1748);	// L2757
    }
  }
}

void gemm_2_12(
  hls::stream< int8_t >& v1749,
  hls::stream< int8_t >& v1750,
  hls::stream< int8_t >& v1751,
  hls::stream< int8_t >& v1752,
  hls::stream< ap_uint<128> >& v1753,
  hls::stream< ap_uint<128> >& v1754
) {	// L2762
  l_S_mt_nt_0_mt48: for (int mt48 = 0; mt48 < 4; mt48++) {	// L2770
    l_nt48: for (int nt48 = 0; nt48 < 4; nt48++) {	// L2771
      int8_t c27;	// L2772
      c27 = 0;	// L2773
      l_S_k_0_k48: for (int k48 = 0; k48 < 64; k48++) {	// L2774
        int8_t v1759 = v1749.read();	// L2775
        int8_t a29;	// L2776
        a29 = v1759;	// L2777
        int8_t v1761 = v1750.read();	// L2778
        int8_t b43;	// L2779
        b43 = v1761;	// L2780
        int8_t v1763 = a29;	// L2781
        int8_t v1764 = b43;	// L2782
        int16_t v1765 = v1763;	// L2783
        int16_t v1766 = v1764;	// L2784
        int16_t v1767 = v1765 * v1766;	// L2785
        int8_t v1768 = c27;	// L2786
        ap_int<17> v1769 = v1768;	// L2787
        ap_int<17> v1770 = v1767;	// L2788
        ap_int<17> v1771 = v1769 + v1770;	// L2789
        int8_t v1772 = v1771;	// L2790
        c27 = v1772;	// L2791
        int8_t v1773 = a29;	// L2792
        v1751.write(v1773);	// L2793
        int8_t v1774 = b43;	// L2794
        v1752.write(v1774);	// L2795
      }
      ap_uint<128> packed_tmp27;	// L2797
      ap_uint<128> v1776 = v1753.read();	// L2798
      packed_tmp27 = v1776;	// L2799
      ap_uint<128> packed_c27;	// L2800
      packed_c27 = 0;	// L2801
      l_S_m_1_m27: for (int m27 = 0; m27 < 16; m27++) {	// L2802
        ap_int<33> v1779 = m27;	// L2803
        bool v1780 = v1779 == 1;	// L2804
        if (v1780) {	// L2805
          int8_t v1781 = c27;	// L2806
          ap_int<128> v1782 = packed_c27;	// L2807
          int64_t v1783 = m27;	// L2808
          int64_t v1784 = v1783 * 8;	// L2809
          ap_int<34> v1785 = m27;	// L2810
          ap_int<34> v1786 = v1785 + 1;	// L2811
          ap_int<66> v1787 = v1786;	// L2812
          ap_int<66> v1788 = v1787 * 8;	// L2813
          ap_int<66> v1789 = v1788 - 1;	// L2814
          int v1790 = v1784;	// L2815
          int v1791 = v1789;	// L2816
          ap_int<128> v1792;
          ap_int<128> v1792_tmp = v1782;
          v1792_tmp(v1791, v1790) = v1781;
          v1792 = v1792_tmp;	// L2817
          packed_c27 = v1792;	// L2818
        } else {
          ap_int<128> v1793 = packed_tmp27;	// L2820
          int64_t v1794 = m27;	// L2821
          int64_t v1795 = v1794 * 8;	// L2822
          ap_int<34> v1796 = m27;	// L2823
          ap_int<34> v1797 = v1796 + 1;	// L2824
          ap_int<66> v1798 = v1797;	// L2825
          ap_int<66> v1799 = v1798 * 8;	// L2826
          ap_int<66> v1800 = v1799 - 1;	// L2827
          int v1801 = v1795;	// L2828
          int v1802 = v1800;	// L2829
          int8_t v1803;
          ap_int<128> v1803_tmp = v1793;
          v1803 = v1803_tmp(v1802, v1801);	// L2830
          ap_int<128> v1804 = packed_c27;	// L2831
          ap_int<128> v1805;
          ap_int<128> v1805_tmp = v1804;
          v1805_tmp(v1802, v1801) = v1803;
          v1805 = v1805_tmp;	// L2841
          packed_c27 = v1805;	// L2842
        }
      }
      ap_int<128> v1806 = packed_c27;	// L2845
      v1754.write(v1806);	// L2846
    }
  }
}

void gemm_2_13(
  hls::stream< int8_t >& v1807,
  hls::stream< int8_t >& v1808,
  hls::stream< int8_t >& v1809,
  hls::stream< int8_t >& v1810,
  hls::stream< ap_uint<128> >& v1811,
  hls::stream< ap_uint<128> >& v1812
) {	// L2851
  l_S_mt_nt_0_mt49: for (int mt49 = 0; mt49 < 4; mt49++) {	// L2859
    l_nt49: for (int nt49 = 0; nt49 < 4; nt49++) {	// L2860
      int8_t c28;	// L2861
      c28 = 0;	// L2862
      l_S_k_0_k49: for (int k49 = 0; k49 < 64; k49++) {	// L2863
        int8_t v1817 = v1807.read();	// L2864
        int8_t a30;	// L2865
        a30 = v1817;	// L2866
        int8_t v1819 = v1808.read();	// L2867
        int8_t b44;	// L2868
        b44 = v1819;	// L2869
        int8_t v1821 = a30;	// L2870
        int8_t v1822 = b44;	// L2871
        int16_t v1823 = v1821;	// L2872
        int16_t v1824 = v1822;	// L2873
        int16_t v1825 = v1823 * v1824;	// L2874
        int8_t v1826 = c28;	// L2875
        ap_int<17> v1827 = v1826;	// L2876
        ap_int<17> v1828 = v1825;	// L2877
        ap_int<17> v1829 = v1827 + v1828;	// L2878
        int8_t v1830 = v1829;	// L2879
        c28 = v1830;	// L2880
        int8_t v1831 = a30;	// L2881
        v1809.write(v1831);	// L2882
        int8_t v1832 = b44;	// L2883
        v1810.write(v1832);	// L2884
      }
      ap_uint<128> packed_tmp28;	// L2886
      ap_uint<128> v1834 = v1811.read();	// L2887
      packed_tmp28 = v1834;	// L2888
      ap_uint<128> packed_c28;	// L2889
      packed_c28 = 0;	// L2890
      l_S_m_1_m28: for (int m28 = 0; m28 < 16; m28++) {	// L2891
        ap_int<33> v1837 = m28;	// L2892
        bool v1838 = v1837 == 1;	// L2893
        if (v1838) {	// L2894
          int8_t v1839 = c28;	// L2895
          ap_int<128> v1840 = packed_c28;	// L2896
          int64_t v1841 = m28;	// L2897
          int64_t v1842 = v1841 * 8;	// L2898
          ap_int<34> v1843 = m28;	// L2899
          ap_int<34> v1844 = v1843 + 1;	// L2900
          ap_int<66> v1845 = v1844;	// L2901
          ap_int<66> v1846 = v1845 * 8;	// L2902
          ap_int<66> v1847 = v1846 - 1;	// L2903
          int v1848 = v1842;	// L2904
          int v1849 = v1847;	// L2905
          ap_int<128> v1850;
          ap_int<128> v1850_tmp = v1840;
          v1850_tmp(v1849, v1848) = v1839;
          v1850 = v1850_tmp;	// L2906
          packed_c28 = v1850;	// L2907
        } else {
          ap_int<128> v1851 = packed_tmp28;	// L2909
          int64_t v1852 = m28;	// L2910
          int64_t v1853 = v1852 * 8;	// L2911
          ap_int<34> v1854 = m28;	// L2912
          ap_int<34> v1855 = v1854 + 1;	// L2913
          ap_int<66> v1856 = v1855;	// L2914
          ap_int<66> v1857 = v1856 * 8;	// L2915
          ap_int<66> v1858 = v1857 - 1;	// L2916
          int v1859 = v1853;	// L2917
          int v1860 = v1858;	// L2918
          int8_t v1861;
          ap_int<128> v1861_tmp = v1851;
          v1861 = v1861_tmp(v1860, v1859);	// L2919
          ap_int<128> v1862 = packed_c28;	// L2920
          ap_int<128> v1863;
          ap_int<128> v1863_tmp = v1862;
          v1863_tmp(v1860, v1859) = v1861;
          v1863 = v1863_tmp;	// L2930
          packed_c28 = v1863;	// L2931
        }
      }
      ap_int<128> v1864 = packed_c28;	// L2934
      v1812.write(v1864);	// L2935
    }
  }
}

void gemm_2_14(
  hls::stream< int8_t >& v1865,
  hls::stream< int8_t >& v1866,
  hls::stream< int8_t >& v1867,
  hls::stream< int8_t >& v1868,
  hls::stream< ap_uint<128> >& v1869,
  hls::stream< ap_uint<128> >& v1870
) {	// L2940
  l_S_mt_nt_0_mt50: for (int mt50 = 0; mt50 < 4; mt50++) {	// L2948
    l_nt50: for (int nt50 = 0; nt50 < 4; nt50++) {	// L2949
      int8_t c29;	// L2950
      c29 = 0;	// L2951
      l_S_k_0_k50: for (int k50 = 0; k50 < 64; k50++) {	// L2952
        int8_t v1875 = v1865.read();	// L2953
        int8_t a31;	// L2954
        a31 = v1875;	// L2955
        int8_t v1877 = v1866.read();	// L2956
        int8_t b45;	// L2957
        b45 = v1877;	// L2958
        int8_t v1879 = a31;	// L2959
        int8_t v1880 = b45;	// L2960
        int16_t v1881 = v1879;	// L2961
        int16_t v1882 = v1880;	// L2962
        int16_t v1883 = v1881 * v1882;	// L2963
        int8_t v1884 = c29;	// L2964
        ap_int<17> v1885 = v1884;	// L2965
        ap_int<17> v1886 = v1883;	// L2966
        ap_int<17> v1887 = v1885 + v1886;	// L2967
        int8_t v1888 = v1887;	// L2968
        c29 = v1888;	// L2969
        int8_t v1889 = a31;	// L2970
        v1867.write(v1889);	// L2971
        int8_t v1890 = b45;	// L2972
        v1868.write(v1890);	// L2973
      }
      ap_uint<128> packed_tmp29;	// L2975
      ap_uint<128> v1892 = v1869.read();	// L2976
      packed_tmp29 = v1892;	// L2977
      ap_uint<128> packed_c29;	// L2978
      packed_c29 = 0;	// L2979
      l_S_m_1_m29: for (int m29 = 0; m29 < 16; m29++) {	// L2980
        ap_int<33> v1895 = m29;	// L2981
        bool v1896 = v1895 == 1;	// L2982
        if (v1896) {	// L2983
          int8_t v1897 = c29;	// L2984
          ap_int<128> v1898 = packed_c29;	// L2985
          int64_t v1899 = m29;	// L2986
          int64_t v1900 = v1899 * 8;	// L2987
          ap_int<34> v1901 = m29;	// L2988
          ap_int<34> v1902 = v1901 + 1;	// L2989
          ap_int<66> v1903 = v1902;	// L2990
          ap_int<66> v1904 = v1903 * 8;	// L2991
          ap_int<66> v1905 = v1904 - 1;	// L2992
          int v1906 = v1900;	// L2993
          int v1907 = v1905;	// L2994
          ap_int<128> v1908;
          ap_int<128> v1908_tmp = v1898;
          v1908_tmp(v1907, v1906) = v1897;
          v1908 = v1908_tmp;	// L2995
          packed_c29 = v1908;	// L2996
        } else {
          ap_int<128> v1909 = packed_tmp29;	// L2998
          int64_t v1910 = m29;	// L2999
          int64_t v1911 = v1910 * 8;	// L3000
          ap_int<34> v1912 = m29;	// L3001
          ap_int<34> v1913 = v1912 + 1;	// L3002
          ap_int<66> v1914 = v1913;	// L3003
          ap_int<66> v1915 = v1914 * 8;	// L3004
          ap_int<66> v1916 = v1915 - 1;	// L3005
          int v1917 = v1911;	// L3006
          int v1918 = v1916;	// L3007
          int8_t v1919;
          ap_int<128> v1919_tmp = v1909;
          v1919 = v1919_tmp(v1918, v1917);	// L3008
          ap_int<128> v1920 = packed_c29;	// L3009
          ap_int<128> v1921;
          ap_int<128> v1921_tmp = v1920;
          v1921_tmp(v1918, v1917) = v1919;
          v1921 = v1921_tmp;	// L3019
          packed_c29 = v1921;	// L3020
        }
      }
      ap_int<128> v1922 = packed_c29;	// L3023
      v1870.write(v1922);	// L3024
    }
  }
}

void gemm_2_15(
  hls::stream< int8_t >& v1923,
  hls::stream< int8_t >& v1924,
  hls::stream< int8_t >& v1925,
  hls::stream< int8_t >& v1926,
  hls::stream< ap_uint<128> >& v1927,
  hls::stream< ap_uint<128> >& v1928
) {	// L3029
  l_S_mt_nt_0_mt51: for (int mt51 = 0; mt51 < 4; mt51++) {	// L3037
    l_nt51: for (int nt51 = 0; nt51 < 4; nt51++) {	// L3038
      int8_t c30;	// L3039
      c30 = 0;	// L3040
      l_S_k_0_k51: for (int k51 = 0; k51 < 64; k51++) {	// L3041
        int8_t v1933 = v1923.read();	// L3042
        int8_t a32;	// L3043
        a32 = v1933;	// L3044
        int8_t v1935 = v1924.read();	// L3045
        int8_t b46;	// L3046
        b46 = v1935;	// L3047
        int8_t v1937 = a32;	// L3048
        int8_t v1938 = b46;	// L3049
        int16_t v1939 = v1937;	// L3050
        int16_t v1940 = v1938;	// L3051
        int16_t v1941 = v1939 * v1940;	// L3052
        int8_t v1942 = c30;	// L3053
        ap_int<17> v1943 = v1942;	// L3054
        ap_int<17> v1944 = v1941;	// L3055
        ap_int<17> v1945 = v1943 + v1944;	// L3056
        int8_t v1946 = v1945;	// L3057
        c30 = v1946;	// L3058
        int8_t v1947 = a32;	// L3059
        v1925.write(v1947);	// L3060
        int8_t v1948 = b46;	// L3061
        v1926.write(v1948);	// L3062
      }
      ap_uint<128> packed_tmp30;	// L3064
      ap_uint<128> v1950 = v1927.read();	// L3065
      packed_tmp30 = v1950;	// L3066
      ap_uint<128> packed_c30;	// L3067
      packed_c30 = 0;	// L3068
      l_S_m_1_m30: for (int m30 = 0; m30 < 16; m30++) {	// L3069
        ap_int<33> v1953 = m30;	// L3070
        bool v1954 = v1953 == 1;	// L3071
        if (v1954) {	// L3072
          int8_t v1955 = c30;	// L3073
          ap_int<128> v1956 = packed_c30;	// L3074
          int64_t v1957 = m30;	// L3075
          int64_t v1958 = v1957 * 8;	// L3076
          ap_int<34> v1959 = m30;	// L3077
          ap_int<34> v1960 = v1959 + 1;	// L3078
          ap_int<66> v1961 = v1960;	// L3079
          ap_int<66> v1962 = v1961 * 8;	// L3080
          ap_int<66> v1963 = v1962 - 1;	// L3081
          int v1964 = v1958;	// L3082
          int v1965 = v1963;	// L3083
          ap_int<128> v1966;
          ap_int<128> v1966_tmp = v1956;
          v1966_tmp(v1965, v1964) = v1955;
          v1966 = v1966_tmp;	// L3084
          packed_c30 = v1966;	// L3085
        } else {
          ap_int<128> v1967 = packed_tmp30;	// L3087
          int64_t v1968 = m30;	// L3088
          int64_t v1969 = v1968 * 8;	// L3089
          ap_int<34> v1970 = m30;	// L3090
          ap_int<34> v1971 = v1970 + 1;	// L3091
          ap_int<66> v1972 = v1971;	// L3092
          ap_int<66> v1973 = v1972 * 8;	// L3093
          ap_int<66> v1974 = v1973 - 1;	// L3094
          int v1975 = v1969;	// L3095
          int v1976 = v1974;	// L3096
          int8_t v1977;
          ap_int<128> v1977_tmp = v1967;
          v1977 = v1977_tmp(v1976, v1975);	// L3097
          ap_int<128> v1978 = packed_c30;	// L3098
          ap_int<128> v1979;
          ap_int<128> v1979_tmp = v1978;
          v1979_tmp(v1976, v1975) = v1977;
          v1979 = v1979_tmp;	// L3108
          packed_c30 = v1979;	// L3109
        }
      }
      ap_int<128> v1980 = packed_c30;	// L3112
      v1928.write(v1980);	// L3113
    }
  }
}

void gemm_2_16(
  hls::stream< int8_t >& v1981,
  hls::stream< int8_t >& v1982,
  hls::stream< int8_t >& v1983,
  hls::stream< ap_uint<128> >& v1984,
  hls::stream< ap_uint<128> >& v1985
) {	// L3118
  l_S_mt_nt_0_mt52: for (int mt52 = 0; mt52 < 4; mt52++) {	// L3126
    l_nt52: for (int nt52 = 0; nt52 < 4; nt52++) {	// L3127
      int8_t c31;	// L3128
      c31 = 0;	// L3129
      l_S_k_0_k52: for (int k52 = 0; k52 < 64; k52++) {	// L3130
        int8_t v1990 = v1981.read();	// L3131
        int8_t a33;	// L3132
        a33 = v1990;	// L3133
        int8_t v1992 = v1982.read();	// L3134
        int8_t b47;	// L3135
        b47 = v1992;	// L3136
        int8_t v1994 = a33;	// L3137
        int8_t v1995 = b47;	// L3138
        int16_t v1996 = v1994;	// L3139
        int16_t v1997 = v1995;	// L3140
        int16_t v1998 = v1996 * v1997;	// L3141
        int8_t v1999 = c31;	// L3142
        ap_int<17> v2000 = v1999;	// L3143
        ap_int<17> v2001 = v1998;	// L3144
        ap_int<17> v2002 = v2000 + v2001;	// L3145
        int8_t v2003 = v2002;	// L3146
        c31 = v2003;	// L3147
        int8_t v2004 = b47;	// L3148
        v1983.write(v2004);	// L3149
      }
      ap_uint<128> packed_tmp31;	// L3151
      ap_uint<128> v2006 = v1984.read();	// L3152
      packed_tmp31 = v2006;	// L3153
      ap_uint<128> packed_c31;	// L3154
      packed_c31 = 0;	// L3155
      l_S_m_1_m31: for (int m31 = 0; m31 < 16; m31++) {	// L3156
        ap_int<33> v2009 = m31;	// L3157
        bool v2010 = v2009 == 1;	// L3158
        if (v2010) {	// L3159
          int8_t v2011 = c31;	// L3160
          ap_int<128> v2012 = packed_c31;	// L3161
          int64_t v2013 = m31;	// L3162
          int64_t v2014 = v2013 * 8;	// L3163
          ap_int<34> v2015 = m31;	// L3164
          ap_int<34> v2016 = v2015 + 1;	// L3165
          ap_int<66> v2017 = v2016;	// L3166
          ap_int<66> v2018 = v2017 * 8;	// L3167
          ap_int<66> v2019 = v2018 - 1;	// L3168
          int v2020 = v2014;	// L3169
          int v2021 = v2019;	// L3170
          ap_int<128> v2022;
          ap_int<128> v2022_tmp = v2012;
          v2022_tmp(v2021, v2020) = v2011;
          v2022 = v2022_tmp;	// L3171
          packed_c31 = v2022;	// L3172
        } else {
          ap_int<128> v2023 = packed_tmp31;	// L3174
          int64_t v2024 = m31;	// L3175
          int64_t v2025 = v2024 * 8;	// L3176
          ap_int<34> v2026 = m31;	// L3177
          ap_int<34> v2027 = v2026 + 1;	// L3178
          ap_int<66> v2028 = v2027;	// L3179
          ap_int<66> v2029 = v2028 * 8;	// L3180
          ap_int<66> v2030 = v2029 - 1;	// L3181
          int v2031 = v2025;	// L3182
          int v2032 = v2030;	// L3183
          int8_t v2033;
          ap_int<128> v2033_tmp = v2023;
          v2033 = v2033_tmp(v2032, v2031);	// L3184
          ap_int<128> v2034 = packed_c31;	// L3185
          ap_int<128> v2035;
          ap_int<128> v2035_tmp = v2034;
          v2035_tmp(v2032, v2031) = v2033;
          v2035 = v2035_tmp;	// L3195
          packed_c31 = v2035;	// L3196
        }
      }
      ap_int<128> v2036 = packed_c31;	// L3199
      v1985.write(v2036);	// L3200
    }
  }
}

void gemm_3_0(
  hls::stream< ap_uint<128> >& v2037,
  hls::stream< int8_t >& v2038,
  hls::stream< ap_uint<128> >& v2039
) {	// L3205
  l_S_mt_nt_0_mt53: for (int mt53 = 0; mt53 < 4; mt53++) {	// L3208
    l_nt53: for (int nt53 = 0; nt53 < 4; nt53++) {	// L3209
      l_S_k_0_k53: for (int k53 = 0; k53 < 64; k53++) {	// L3210
        ap_uint<128> v2043 = v2037.read();	// L3211
        ap_uint<128> a34;	// L3212
        a34 = v2043;	// L3213
        ap_int<128> v2045 = a34;	// L3214
        int8_t v2046;
        ap_int<128> v2046_tmp = v2045;
        v2046 = v2046_tmp(23, 16);	// L3215
        v2038.write(v2046);	// L3216
        ap_int<128> v2047 = a34;	// L3217
        v2039.write(v2047);	// L3218
      }
    }
  }
}

void gemm_3_1(
  hls::stream< int8_t >& v2048,
  hls::stream< int8_t >& v2049,
  hls::stream< int8_t >& v2050,
  hls::stream< int8_t >& v2051,
  hls::stream< ap_uint<128> >& v2052,
  hls::stream< ap_uint<128> >& v2053
) {	// L3224
  l_S_mt_nt_0_mt54: for (int mt54 = 0; mt54 < 4; mt54++) {	// L3232
    l_nt54: for (int nt54 = 0; nt54 < 4; nt54++) {	// L3233
      int8_t c32;	// L3234
      c32 = 0;	// L3235
      l_S_k_0_k54: for (int k54 = 0; k54 < 64; k54++) {	// L3236
        int8_t v2058 = v2048.read();	// L3237
        int8_t a35;	// L3238
        a35 = v2058;	// L3239
        int8_t v2060 = v2049.read();	// L3240
        int8_t b48;	// L3241
        b48 = v2060;	// L3242
        int8_t v2062 = a35;	// L3243
        int8_t v2063 = b48;	// L3244
        int16_t v2064 = v2062;	// L3245
        int16_t v2065 = v2063;	// L3246
        int16_t v2066 = v2064 * v2065;	// L3247
        int8_t v2067 = c32;	// L3248
        ap_int<17> v2068 = v2067;	// L3249
        ap_int<17> v2069 = v2066;	// L3250
        ap_int<17> v2070 = v2068 + v2069;	// L3251
        int8_t v2071 = v2070;	// L3252
        c32 = v2071;	// L3253
        int8_t v2072 = a35;	// L3254
        v2050.write(v2072);	// L3255
        int8_t v2073 = b48;	// L3256
        v2051.write(v2073);	// L3257
      }
      ap_uint<128> packed_tmp32;	// L3259
      ap_uint<128> v2075 = v2052.read();	// L3260
      packed_tmp32 = v2075;	// L3261
      ap_uint<128> packed_c32;	// L3262
      packed_c32 = 0;	// L3263
      l_S_m_1_m32: for (int m32 = 0; m32 < 16; m32++) {	// L3264
        ap_int<33> v2078 = m32;	// L3265
        bool v2079 = v2078 == 2;	// L3266
        if (v2079) {	// L3267
          int8_t v2080 = c32;	// L3268
          ap_int<128> v2081 = packed_c32;	// L3269
          int64_t v2082 = m32;	// L3270
          int64_t v2083 = v2082 * 8;	// L3271
          ap_int<34> v2084 = m32;	// L3272
          ap_int<34> v2085 = v2084 + 1;	// L3273
          ap_int<66> v2086 = v2085;	// L3274
          ap_int<66> v2087 = v2086 * 8;	// L3275
          ap_int<66> v2088 = v2087 - 1;	// L3276
          int v2089 = v2083;	// L3277
          int v2090 = v2088;	// L3278
          ap_int<128> v2091;
          ap_int<128> v2091_tmp = v2081;
          v2091_tmp(v2090, v2089) = v2080;
          v2091 = v2091_tmp;	// L3279
          packed_c32 = v2091;	// L3280
        } else {
          ap_int<128> v2092 = packed_tmp32;	// L3282
          int64_t v2093 = m32;	// L3283
          int64_t v2094 = v2093 * 8;	// L3284
          ap_int<34> v2095 = m32;	// L3285
          ap_int<34> v2096 = v2095 + 1;	// L3286
          ap_int<66> v2097 = v2096;	// L3287
          ap_int<66> v2098 = v2097 * 8;	// L3288
          ap_int<66> v2099 = v2098 - 1;	// L3289
          int v2100 = v2094;	// L3290
          int v2101 = v2099;	// L3291
          int8_t v2102;
          ap_int<128> v2102_tmp = v2092;
          v2102 = v2102_tmp(v2101, v2100);	// L3292
          ap_int<128> v2103 = packed_c32;	// L3293
          ap_int<128> v2104;
          ap_int<128> v2104_tmp = v2103;
          v2104_tmp(v2101, v2100) = v2102;
          v2104 = v2104_tmp;	// L3303
          packed_c32 = v2104;	// L3304
        }
      }
      ap_int<128> v2105 = packed_c32;	// L3307
      v2053.write(v2105);	// L3308
    }
  }
}

void gemm_3_2(
  hls::stream< int8_t >& v2106,
  hls::stream< int8_t >& v2107,
  hls::stream< int8_t >& v2108,
  hls::stream< int8_t >& v2109,
  hls::stream< ap_uint<128> >& v2110,
  hls::stream< ap_uint<128> >& v2111
) {	// L3313
  l_S_mt_nt_0_mt55: for (int mt55 = 0; mt55 < 4; mt55++) {	// L3321
    l_nt55: for (int nt55 = 0; nt55 < 4; nt55++) {	// L3322
      int8_t c33;	// L3323
      c33 = 0;	// L3324
      l_S_k_0_k55: for (int k55 = 0; k55 < 64; k55++) {	// L3325
        int8_t v2116 = v2106.read();	// L3326
        int8_t a36;	// L3327
        a36 = v2116;	// L3328
        int8_t v2118 = v2107.read();	// L3329
        int8_t b49;	// L3330
        b49 = v2118;	// L3331
        int8_t v2120 = a36;	// L3332
        int8_t v2121 = b49;	// L3333
        int16_t v2122 = v2120;	// L3334
        int16_t v2123 = v2121;	// L3335
        int16_t v2124 = v2122 * v2123;	// L3336
        int8_t v2125 = c33;	// L3337
        ap_int<17> v2126 = v2125;	// L3338
        ap_int<17> v2127 = v2124;	// L3339
        ap_int<17> v2128 = v2126 + v2127;	// L3340
        int8_t v2129 = v2128;	// L3341
        c33 = v2129;	// L3342
        int8_t v2130 = a36;	// L3343
        v2108.write(v2130);	// L3344
        int8_t v2131 = b49;	// L3345
        v2109.write(v2131);	// L3346
      }
      ap_uint<128> packed_tmp33;	// L3348
      ap_uint<128> v2133 = v2110.read();	// L3349
      packed_tmp33 = v2133;	// L3350
      ap_uint<128> packed_c33;	// L3351
      packed_c33 = 0;	// L3352
      l_S_m_1_m33: for (int m33 = 0; m33 < 16; m33++) {	// L3353
        ap_int<33> v2136 = m33;	// L3354
        bool v2137 = v2136 == 2;	// L3355
        if (v2137) {	// L3356
          int8_t v2138 = c33;	// L3357
          ap_int<128> v2139 = packed_c33;	// L3358
          int64_t v2140 = m33;	// L3359
          int64_t v2141 = v2140 * 8;	// L3360
          ap_int<34> v2142 = m33;	// L3361
          ap_int<34> v2143 = v2142 + 1;	// L3362
          ap_int<66> v2144 = v2143;	// L3363
          ap_int<66> v2145 = v2144 * 8;	// L3364
          ap_int<66> v2146 = v2145 - 1;	// L3365
          int v2147 = v2141;	// L3366
          int v2148 = v2146;	// L3367
          ap_int<128> v2149;
          ap_int<128> v2149_tmp = v2139;
          v2149_tmp(v2148, v2147) = v2138;
          v2149 = v2149_tmp;	// L3368
          packed_c33 = v2149;	// L3369
        } else {
          ap_int<128> v2150 = packed_tmp33;	// L3371
          int64_t v2151 = m33;	// L3372
          int64_t v2152 = v2151 * 8;	// L3373
          ap_int<34> v2153 = m33;	// L3374
          ap_int<34> v2154 = v2153 + 1;	// L3375
          ap_int<66> v2155 = v2154;	// L3376
          ap_int<66> v2156 = v2155 * 8;	// L3377
          ap_int<66> v2157 = v2156 - 1;	// L3378
          int v2158 = v2152;	// L3379
          int v2159 = v2157;	// L3380
          int8_t v2160;
          ap_int<128> v2160_tmp = v2150;
          v2160 = v2160_tmp(v2159, v2158);	// L3381
          ap_int<128> v2161 = packed_c33;	// L3382
          ap_int<128> v2162;
          ap_int<128> v2162_tmp = v2161;
          v2162_tmp(v2159, v2158) = v2160;
          v2162 = v2162_tmp;	// L3392
          packed_c33 = v2162;	// L3393
        }
      }
      ap_int<128> v2163 = packed_c33;	// L3396
      v2111.write(v2163);	// L3397
    }
  }
}

void gemm_3_3(
  hls::stream< int8_t >& v2164,
  hls::stream< int8_t >& v2165,
  hls::stream< int8_t >& v2166,
  hls::stream< int8_t >& v2167,
  hls::stream< ap_uint<128> >& v2168,
  hls::stream< ap_uint<128> >& v2169
) {	// L3402
  l_S_mt_nt_0_mt56: for (int mt56 = 0; mt56 < 4; mt56++) {	// L3410
    l_nt56: for (int nt56 = 0; nt56 < 4; nt56++) {	// L3411
      int8_t c34;	// L3412
      c34 = 0;	// L3413
      l_S_k_0_k56: for (int k56 = 0; k56 < 64; k56++) {	// L3414
        int8_t v2174 = v2164.read();	// L3415
        int8_t a37;	// L3416
        a37 = v2174;	// L3417
        int8_t v2176 = v2165.read();	// L3418
        int8_t b50;	// L3419
        b50 = v2176;	// L3420
        int8_t v2178 = a37;	// L3421
        int8_t v2179 = b50;	// L3422
        int16_t v2180 = v2178;	// L3423
        int16_t v2181 = v2179;	// L3424
        int16_t v2182 = v2180 * v2181;	// L3425
        int8_t v2183 = c34;	// L3426
        ap_int<17> v2184 = v2183;	// L3427
        ap_int<17> v2185 = v2182;	// L3428
        ap_int<17> v2186 = v2184 + v2185;	// L3429
        int8_t v2187 = v2186;	// L3430
        c34 = v2187;	// L3431
        int8_t v2188 = a37;	// L3432
        v2166.write(v2188);	// L3433
        int8_t v2189 = b50;	// L3434
        v2167.write(v2189);	// L3435
      }
      ap_uint<128> packed_tmp34;	// L3437
      ap_uint<128> v2191 = v2168.read();	// L3438
      packed_tmp34 = v2191;	// L3439
      ap_uint<128> packed_c34;	// L3440
      packed_c34 = 0;	// L3441
      l_S_m_1_m34: for (int m34 = 0; m34 < 16; m34++) {	// L3442
        ap_int<33> v2194 = m34;	// L3443
        bool v2195 = v2194 == 2;	// L3444
        if (v2195) {	// L3445
          int8_t v2196 = c34;	// L3446
          ap_int<128> v2197 = packed_c34;	// L3447
          int64_t v2198 = m34;	// L3448
          int64_t v2199 = v2198 * 8;	// L3449
          ap_int<34> v2200 = m34;	// L3450
          ap_int<34> v2201 = v2200 + 1;	// L3451
          ap_int<66> v2202 = v2201;	// L3452
          ap_int<66> v2203 = v2202 * 8;	// L3453
          ap_int<66> v2204 = v2203 - 1;	// L3454
          int v2205 = v2199;	// L3455
          int v2206 = v2204;	// L3456
          ap_int<128> v2207;
          ap_int<128> v2207_tmp = v2197;
          v2207_tmp(v2206, v2205) = v2196;
          v2207 = v2207_tmp;	// L3457
          packed_c34 = v2207;	// L3458
        } else {
          ap_int<128> v2208 = packed_tmp34;	// L3460
          int64_t v2209 = m34;	// L3461
          int64_t v2210 = v2209 * 8;	// L3462
          ap_int<34> v2211 = m34;	// L3463
          ap_int<34> v2212 = v2211 + 1;	// L3464
          ap_int<66> v2213 = v2212;	// L3465
          ap_int<66> v2214 = v2213 * 8;	// L3466
          ap_int<66> v2215 = v2214 - 1;	// L3467
          int v2216 = v2210;	// L3468
          int v2217 = v2215;	// L3469
          int8_t v2218;
          ap_int<128> v2218_tmp = v2208;
          v2218 = v2218_tmp(v2217, v2216);	// L3470
          ap_int<128> v2219 = packed_c34;	// L3471
          ap_int<128> v2220;
          ap_int<128> v2220_tmp = v2219;
          v2220_tmp(v2217, v2216) = v2218;
          v2220 = v2220_tmp;	// L3481
          packed_c34 = v2220;	// L3482
        }
      }
      ap_int<128> v2221 = packed_c34;	// L3485
      v2169.write(v2221);	// L3486
    }
  }
}

void gemm_3_4(
  hls::stream< int8_t >& v2222,
  hls::stream< int8_t >& v2223,
  hls::stream< int8_t >& v2224,
  hls::stream< int8_t >& v2225,
  hls::stream< ap_uint<128> >& v2226,
  hls::stream< ap_uint<128> >& v2227
) {	// L3491
  l_S_mt_nt_0_mt57: for (int mt57 = 0; mt57 < 4; mt57++) {	// L3499
    l_nt57: for (int nt57 = 0; nt57 < 4; nt57++) {	// L3500
      int8_t c35;	// L3501
      c35 = 0;	// L3502
      l_S_k_0_k57: for (int k57 = 0; k57 < 64; k57++) {	// L3503
        int8_t v2232 = v2222.read();	// L3504
        int8_t a38;	// L3505
        a38 = v2232;	// L3506
        int8_t v2234 = v2223.read();	// L3507
        int8_t b51;	// L3508
        b51 = v2234;	// L3509
        int8_t v2236 = a38;	// L3510
        int8_t v2237 = b51;	// L3511
        int16_t v2238 = v2236;	// L3512
        int16_t v2239 = v2237;	// L3513
        int16_t v2240 = v2238 * v2239;	// L3514
        int8_t v2241 = c35;	// L3515
        ap_int<17> v2242 = v2241;	// L3516
        ap_int<17> v2243 = v2240;	// L3517
        ap_int<17> v2244 = v2242 + v2243;	// L3518
        int8_t v2245 = v2244;	// L3519
        c35 = v2245;	// L3520
        int8_t v2246 = a38;	// L3521
        v2224.write(v2246);	// L3522
        int8_t v2247 = b51;	// L3523
        v2225.write(v2247);	// L3524
      }
      ap_uint<128> packed_tmp35;	// L3526
      ap_uint<128> v2249 = v2226.read();	// L3527
      packed_tmp35 = v2249;	// L3528
      ap_uint<128> packed_c35;	// L3529
      packed_c35 = 0;	// L3530
      l_S_m_1_m35: for (int m35 = 0; m35 < 16; m35++) {	// L3531
        ap_int<33> v2252 = m35;	// L3532
        bool v2253 = v2252 == 2;	// L3533
        if (v2253) {	// L3534
          int8_t v2254 = c35;	// L3535
          ap_int<128> v2255 = packed_c35;	// L3536
          int64_t v2256 = m35;	// L3537
          int64_t v2257 = v2256 * 8;	// L3538
          ap_int<34> v2258 = m35;	// L3539
          ap_int<34> v2259 = v2258 + 1;	// L3540
          ap_int<66> v2260 = v2259;	// L3541
          ap_int<66> v2261 = v2260 * 8;	// L3542
          ap_int<66> v2262 = v2261 - 1;	// L3543
          int v2263 = v2257;	// L3544
          int v2264 = v2262;	// L3545
          ap_int<128> v2265;
          ap_int<128> v2265_tmp = v2255;
          v2265_tmp(v2264, v2263) = v2254;
          v2265 = v2265_tmp;	// L3546
          packed_c35 = v2265;	// L3547
        } else {
          ap_int<128> v2266 = packed_tmp35;	// L3549
          int64_t v2267 = m35;	// L3550
          int64_t v2268 = v2267 * 8;	// L3551
          ap_int<34> v2269 = m35;	// L3552
          ap_int<34> v2270 = v2269 + 1;	// L3553
          ap_int<66> v2271 = v2270;	// L3554
          ap_int<66> v2272 = v2271 * 8;	// L3555
          ap_int<66> v2273 = v2272 - 1;	// L3556
          int v2274 = v2268;	// L3557
          int v2275 = v2273;	// L3558
          int8_t v2276;
          ap_int<128> v2276_tmp = v2266;
          v2276 = v2276_tmp(v2275, v2274);	// L3559
          ap_int<128> v2277 = packed_c35;	// L3560
          ap_int<128> v2278;
          ap_int<128> v2278_tmp = v2277;
          v2278_tmp(v2275, v2274) = v2276;
          v2278 = v2278_tmp;	// L3570
          packed_c35 = v2278;	// L3571
        }
      }
      ap_int<128> v2279 = packed_c35;	// L3574
      v2227.write(v2279);	// L3575
    }
  }
}

void gemm_3_5(
  hls::stream< int8_t >& v2280,
  hls::stream< int8_t >& v2281,
  hls::stream< int8_t >& v2282,
  hls::stream< int8_t >& v2283,
  hls::stream< ap_uint<128> >& v2284,
  hls::stream< ap_uint<128> >& v2285
) {	// L3580
  l_S_mt_nt_0_mt58: for (int mt58 = 0; mt58 < 4; mt58++) {	// L3588
    l_nt58: for (int nt58 = 0; nt58 < 4; nt58++) {	// L3589
      int8_t c36;	// L3590
      c36 = 0;	// L3591
      l_S_k_0_k58: for (int k58 = 0; k58 < 64; k58++) {	// L3592
        int8_t v2290 = v2280.read();	// L3593
        int8_t a39;	// L3594
        a39 = v2290;	// L3595
        int8_t v2292 = v2281.read();	// L3596
        int8_t b52;	// L3597
        b52 = v2292;	// L3598
        int8_t v2294 = a39;	// L3599
        int8_t v2295 = b52;	// L3600
        int16_t v2296 = v2294;	// L3601
        int16_t v2297 = v2295;	// L3602
        int16_t v2298 = v2296 * v2297;	// L3603
        int8_t v2299 = c36;	// L3604
        ap_int<17> v2300 = v2299;	// L3605
        ap_int<17> v2301 = v2298;	// L3606
        ap_int<17> v2302 = v2300 + v2301;	// L3607
        int8_t v2303 = v2302;	// L3608
        c36 = v2303;	// L3609
        int8_t v2304 = a39;	// L3610
        v2282.write(v2304);	// L3611
        int8_t v2305 = b52;	// L3612
        v2283.write(v2305);	// L3613
      }
      ap_uint<128> packed_tmp36;	// L3615
      ap_uint<128> v2307 = v2284.read();	// L3616
      packed_tmp36 = v2307;	// L3617
      ap_uint<128> packed_c36;	// L3618
      packed_c36 = 0;	// L3619
      l_S_m_1_m36: for (int m36 = 0; m36 < 16; m36++) {	// L3620
        ap_int<33> v2310 = m36;	// L3621
        bool v2311 = v2310 == 2;	// L3622
        if (v2311) {	// L3623
          int8_t v2312 = c36;	// L3624
          ap_int<128> v2313 = packed_c36;	// L3625
          int64_t v2314 = m36;	// L3626
          int64_t v2315 = v2314 * 8;	// L3627
          ap_int<34> v2316 = m36;	// L3628
          ap_int<34> v2317 = v2316 + 1;	// L3629
          ap_int<66> v2318 = v2317;	// L3630
          ap_int<66> v2319 = v2318 * 8;	// L3631
          ap_int<66> v2320 = v2319 - 1;	// L3632
          int v2321 = v2315;	// L3633
          int v2322 = v2320;	// L3634
          ap_int<128> v2323;
          ap_int<128> v2323_tmp = v2313;
          v2323_tmp(v2322, v2321) = v2312;
          v2323 = v2323_tmp;	// L3635
          packed_c36 = v2323;	// L3636
        } else {
          ap_int<128> v2324 = packed_tmp36;	// L3638
          int64_t v2325 = m36;	// L3639
          int64_t v2326 = v2325 * 8;	// L3640
          ap_int<34> v2327 = m36;	// L3641
          ap_int<34> v2328 = v2327 + 1;	// L3642
          ap_int<66> v2329 = v2328;	// L3643
          ap_int<66> v2330 = v2329 * 8;	// L3644
          ap_int<66> v2331 = v2330 - 1;	// L3645
          int v2332 = v2326;	// L3646
          int v2333 = v2331;	// L3647
          int8_t v2334;
          ap_int<128> v2334_tmp = v2324;
          v2334 = v2334_tmp(v2333, v2332);	// L3648
          ap_int<128> v2335 = packed_c36;	// L3649
          ap_int<128> v2336;
          ap_int<128> v2336_tmp = v2335;
          v2336_tmp(v2333, v2332) = v2334;
          v2336 = v2336_tmp;	// L3659
          packed_c36 = v2336;	// L3660
        }
      }
      ap_int<128> v2337 = packed_c36;	// L3663
      v2285.write(v2337);	// L3664
    }
  }
}

void gemm_3_6(
  hls::stream< int8_t >& v2338,
  hls::stream< int8_t >& v2339,
  hls::stream< int8_t >& v2340,
  hls::stream< int8_t >& v2341,
  hls::stream< ap_uint<128> >& v2342,
  hls::stream< ap_uint<128> >& v2343
) {	// L3669
  l_S_mt_nt_0_mt59: for (int mt59 = 0; mt59 < 4; mt59++) {	// L3677
    l_nt59: for (int nt59 = 0; nt59 < 4; nt59++) {	// L3678
      int8_t c37;	// L3679
      c37 = 0;	// L3680
      l_S_k_0_k59: for (int k59 = 0; k59 < 64; k59++) {	// L3681
        int8_t v2348 = v2338.read();	// L3682
        int8_t a40;	// L3683
        a40 = v2348;	// L3684
        int8_t v2350 = v2339.read();	// L3685
        int8_t b53;	// L3686
        b53 = v2350;	// L3687
        int8_t v2352 = a40;	// L3688
        int8_t v2353 = b53;	// L3689
        int16_t v2354 = v2352;	// L3690
        int16_t v2355 = v2353;	// L3691
        int16_t v2356 = v2354 * v2355;	// L3692
        int8_t v2357 = c37;	// L3693
        ap_int<17> v2358 = v2357;	// L3694
        ap_int<17> v2359 = v2356;	// L3695
        ap_int<17> v2360 = v2358 + v2359;	// L3696
        int8_t v2361 = v2360;	// L3697
        c37 = v2361;	// L3698
        int8_t v2362 = a40;	// L3699
        v2340.write(v2362);	// L3700
        int8_t v2363 = b53;	// L3701
        v2341.write(v2363);	// L3702
      }
      ap_uint<128> packed_tmp37;	// L3704
      ap_uint<128> v2365 = v2342.read();	// L3705
      packed_tmp37 = v2365;	// L3706
      ap_uint<128> packed_c37;	// L3707
      packed_c37 = 0;	// L3708
      l_S_m_1_m37: for (int m37 = 0; m37 < 16; m37++) {	// L3709
        ap_int<33> v2368 = m37;	// L3710
        bool v2369 = v2368 == 2;	// L3711
        if (v2369) {	// L3712
          int8_t v2370 = c37;	// L3713
          ap_int<128> v2371 = packed_c37;	// L3714
          int64_t v2372 = m37;	// L3715
          int64_t v2373 = v2372 * 8;	// L3716
          ap_int<34> v2374 = m37;	// L3717
          ap_int<34> v2375 = v2374 + 1;	// L3718
          ap_int<66> v2376 = v2375;	// L3719
          ap_int<66> v2377 = v2376 * 8;	// L3720
          ap_int<66> v2378 = v2377 - 1;	// L3721
          int v2379 = v2373;	// L3722
          int v2380 = v2378;	// L3723
          ap_int<128> v2381;
          ap_int<128> v2381_tmp = v2371;
          v2381_tmp(v2380, v2379) = v2370;
          v2381 = v2381_tmp;	// L3724
          packed_c37 = v2381;	// L3725
        } else {
          ap_int<128> v2382 = packed_tmp37;	// L3727
          int64_t v2383 = m37;	// L3728
          int64_t v2384 = v2383 * 8;	// L3729
          ap_int<34> v2385 = m37;	// L3730
          ap_int<34> v2386 = v2385 + 1;	// L3731
          ap_int<66> v2387 = v2386;	// L3732
          ap_int<66> v2388 = v2387 * 8;	// L3733
          ap_int<66> v2389 = v2388 - 1;	// L3734
          int v2390 = v2384;	// L3735
          int v2391 = v2389;	// L3736
          int8_t v2392;
          ap_int<128> v2392_tmp = v2382;
          v2392 = v2392_tmp(v2391, v2390);	// L3737
          ap_int<128> v2393 = packed_c37;	// L3738
          ap_int<128> v2394;
          ap_int<128> v2394_tmp = v2393;
          v2394_tmp(v2391, v2390) = v2392;
          v2394 = v2394_tmp;	// L3748
          packed_c37 = v2394;	// L3749
        }
      }
      ap_int<128> v2395 = packed_c37;	// L3752
      v2343.write(v2395);	// L3753
    }
  }
}

void gemm_3_7(
  hls::stream< int8_t >& v2396,
  hls::stream< int8_t >& v2397,
  hls::stream< int8_t >& v2398,
  hls::stream< int8_t >& v2399,
  hls::stream< ap_uint<128> >& v2400,
  hls::stream< ap_uint<128> >& v2401
) {	// L3758
  l_S_mt_nt_0_mt60: for (int mt60 = 0; mt60 < 4; mt60++) {	// L3766
    l_nt60: for (int nt60 = 0; nt60 < 4; nt60++) {	// L3767
      int8_t c38;	// L3768
      c38 = 0;	// L3769
      l_S_k_0_k60: for (int k60 = 0; k60 < 64; k60++) {	// L3770
        int8_t v2406 = v2396.read();	// L3771
        int8_t a41;	// L3772
        a41 = v2406;	// L3773
        int8_t v2408 = v2397.read();	// L3774
        int8_t b54;	// L3775
        b54 = v2408;	// L3776
        int8_t v2410 = a41;	// L3777
        int8_t v2411 = b54;	// L3778
        int16_t v2412 = v2410;	// L3779
        int16_t v2413 = v2411;	// L3780
        int16_t v2414 = v2412 * v2413;	// L3781
        int8_t v2415 = c38;	// L3782
        ap_int<17> v2416 = v2415;	// L3783
        ap_int<17> v2417 = v2414;	// L3784
        ap_int<17> v2418 = v2416 + v2417;	// L3785
        int8_t v2419 = v2418;	// L3786
        c38 = v2419;	// L3787
        int8_t v2420 = a41;	// L3788
        v2398.write(v2420);	// L3789
        int8_t v2421 = b54;	// L3790
        v2399.write(v2421);	// L3791
      }
      ap_uint<128> packed_tmp38;	// L3793
      ap_uint<128> v2423 = v2400.read();	// L3794
      packed_tmp38 = v2423;	// L3795
      ap_uint<128> packed_c38;	// L3796
      packed_c38 = 0;	// L3797
      l_S_m_1_m38: for (int m38 = 0; m38 < 16; m38++) {	// L3798
        ap_int<33> v2426 = m38;	// L3799
        bool v2427 = v2426 == 2;	// L3800
        if (v2427) {	// L3801
          int8_t v2428 = c38;	// L3802
          ap_int<128> v2429 = packed_c38;	// L3803
          int64_t v2430 = m38;	// L3804
          int64_t v2431 = v2430 * 8;	// L3805
          ap_int<34> v2432 = m38;	// L3806
          ap_int<34> v2433 = v2432 + 1;	// L3807
          ap_int<66> v2434 = v2433;	// L3808
          ap_int<66> v2435 = v2434 * 8;	// L3809
          ap_int<66> v2436 = v2435 - 1;	// L3810
          int v2437 = v2431;	// L3811
          int v2438 = v2436;	// L3812
          ap_int<128> v2439;
          ap_int<128> v2439_tmp = v2429;
          v2439_tmp(v2438, v2437) = v2428;
          v2439 = v2439_tmp;	// L3813
          packed_c38 = v2439;	// L3814
        } else {
          ap_int<128> v2440 = packed_tmp38;	// L3816
          int64_t v2441 = m38;	// L3817
          int64_t v2442 = v2441 * 8;	// L3818
          ap_int<34> v2443 = m38;	// L3819
          ap_int<34> v2444 = v2443 + 1;	// L3820
          ap_int<66> v2445 = v2444;	// L3821
          ap_int<66> v2446 = v2445 * 8;	// L3822
          ap_int<66> v2447 = v2446 - 1;	// L3823
          int v2448 = v2442;	// L3824
          int v2449 = v2447;	// L3825
          int8_t v2450;
          ap_int<128> v2450_tmp = v2440;
          v2450 = v2450_tmp(v2449, v2448);	// L3826
          ap_int<128> v2451 = packed_c38;	// L3827
          ap_int<128> v2452;
          ap_int<128> v2452_tmp = v2451;
          v2452_tmp(v2449, v2448) = v2450;
          v2452 = v2452_tmp;	// L3837
          packed_c38 = v2452;	// L3838
        }
      }
      ap_int<128> v2453 = packed_c38;	// L3841
      v2401.write(v2453);	// L3842
    }
  }
}

void gemm_3_8(
  hls::stream< int8_t >& v2454,
  hls::stream< int8_t >& v2455,
  hls::stream< int8_t >& v2456,
  hls::stream< int8_t >& v2457,
  hls::stream< ap_uint<128> >& v2458,
  hls::stream< ap_uint<128> >& v2459
) {	// L3847
  l_S_mt_nt_0_mt61: for (int mt61 = 0; mt61 < 4; mt61++) {	// L3855
    l_nt61: for (int nt61 = 0; nt61 < 4; nt61++) {	// L3856
      int8_t c39;	// L3857
      c39 = 0;	// L3858
      l_S_k_0_k61: for (int k61 = 0; k61 < 64; k61++) {	// L3859
        int8_t v2464 = v2454.read();	// L3860
        int8_t a42;	// L3861
        a42 = v2464;	// L3862
        int8_t v2466 = v2455.read();	// L3863
        int8_t b55;	// L3864
        b55 = v2466;	// L3865
        int8_t v2468 = a42;	// L3866
        int8_t v2469 = b55;	// L3867
        int16_t v2470 = v2468;	// L3868
        int16_t v2471 = v2469;	// L3869
        int16_t v2472 = v2470 * v2471;	// L3870
        int8_t v2473 = c39;	// L3871
        ap_int<17> v2474 = v2473;	// L3872
        ap_int<17> v2475 = v2472;	// L3873
        ap_int<17> v2476 = v2474 + v2475;	// L3874
        int8_t v2477 = v2476;	// L3875
        c39 = v2477;	// L3876
        int8_t v2478 = a42;	// L3877
        v2456.write(v2478);	// L3878
        int8_t v2479 = b55;	// L3879
        v2457.write(v2479);	// L3880
      }
      ap_uint<128> packed_tmp39;	// L3882
      ap_uint<128> v2481 = v2458.read();	// L3883
      packed_tmp39 = v2481;	// L3884
      ap_uint<128> packed_c39;	// L3885
      packed_c39 = 0;	// L3886
      l_S_m_1_m39: for (int m39 = 0; m39 < 16; m39++) {	// L3887
        ap_int<33> v2484 = m39;	// L3888
        bool v2485 = v2484 == 2;	// L3889
        if (v2485) {	// L3890
          int8_t v2486 = c39;	// L3891
          ap_int<128> v2487 = packed_c39;	// L3892
          int64_t v2488 = m39;	// L3893
          int64_t v2489 = v2488 * 8;	// L3894
          ap_int<34> v2490 = m39;	// L3895
          ap_int<34> v2491 = v2490 + 1;	// L3896
          ap_int<66> v2492 = v2491;	// L3897
          ap_int<66> v2493 = v2492 * 8;	// L3898
          ap_int<66> v2494 = v2493 - 1;	// L3899
          int v2495 = v2489;	// L3900
          int v2496 = v2494;	// L3901
          ap_int<128> v2497;
          ap_int<128> v2497_tmp = v2487;
          v2497_tmp(v2496, v2495) = v2486;
          v2497 = v2497_tmp;	// L3902
          packed_c39 = v2497;	// L3903
        } else {
          ap_int<128> v2498 = packed_tmp39;	// L3905
          int64_t v2499 = m39;	// L3906
          int64_t v2500 = v2499 * 8;	// L3907
          ap_int<34> v2501 = m39;	// L3908
          ap_int<34> v2502 = v2501 + 1;	// L3909
          ap_int<66> v2503 = v2502;	// L3910
          ap_int<66> v2504 = v2503 * 8;	// L3911
          ap_int<66> v2505 = v2504 - 1;	// L3912
          int v2506 = v2500;	// L3913
          int v2507 = v2505;	// L3914
          int8_t v2508;
          ap_int<128> v2508_tmp = v2498;
          v2508 = v2508_tmp(v2507, v2506);	// L3915
          ap_int<128> v2509 = packed_c39;	// L3916
          ap_int<128> v2510;
          ap_int<128> v2510_tmp = v2509;
          v2510_tmp(v2507, v2506) = v2508;
          v2510 = v2510_tmp;	// L3926
          packed_c39 = v2510;	// L3927
        }
      }
      ap_int<128> v2511 = packed_c39;	// L3930
      v2459.write(v2511);	// L3931
    }
  }
}

void gemm_3_9(
  hls::stream< int8_t >& v2512,
  hls::stream< int8_t >& v2513,
  hls::stream< int8_t >& v2514,
  hls::stream< int8_t >& v2515,
  hls::stream< ap_uint<128> >& v2516,
  hls::stream< ap_uint<128> >& v2517
) {	// L3936
  l_S_mt_nt_0_mt62: for (int mt62 = 0; mt62 < 4; mt62++) {	// L3944
    l_nt62: for (int nt62 = 0; nt62 < 4; nt62++) {	// L3945
      int8_t c40;	// L3946
      c40 = 0;	// L3947
      l_S_k_0_k62: for (int k62 = 0; k62 < 64; k62++) {	// L3948
        int8_t v2522 = v2512.read();	// L3949
        int8_t a43;	// L3950
        a43 = v2522;	// L3951
        int8_t v2524 = v2513.read();	// L3952
        int8_t b56;	// L3953
        b56 = v2524;	// L3954
        int8_t v2526 = a43;	// L3955
        int8_t v2527 = b56;	// L3956
        int16_t v2528 = v2526;	// L3957
        int16_t v2529 = v2527;	// L3958
        int16_t v2530 = v2528 * v2529;	// L3959
        int8_t v2531 = c40;	// L3960
        ap_int<17> v2532 = v2531;	// L3961
        ap_int<17> v2533 = v2530;	// L3962
        ap_int<17> v2534 = v2532 + v2533;	// L3963
        int8_t v2535 = v2534;	// L3964
        c40 = v2535;	// L3965
        int8_t v2536 = a43;	// L3966
        v2514.write(v2536);	// L3967
        int8_t v2537 = b56;	// L3968
        v2515.write(v2537);	// L3969
      }
      ap_uint<128> packed_tmp40;	// L3971
      ap_uint<128> v2539 = v2516.read();	// L3972
      packed_tmp40 = v2539;	// L3973
      ap_uint<128> packed_c40;	// L3974
      packed_c40 = 0;	// L3975
      l_S_m_1_m40: for (int m40 = 0; m40 < 16; m40++) {	// L3976
        ap_int<33> v2542 = m40;	// L3977
        bool v2543 = v2542 == 2;	// L3978
        if (v2543) {	// L3979
          int8_t v2544 = c40;	// L3980
          ap_int<128> v2545 = packed_c40;	// L3981
          int64_t v2546 = m40;	// L3982
          int64_t v2547 = v2546 * 8;	// L3983
          ap_int<34> v2548 = m40;	// L3984
          ap_int<34> v2549 = v2548 + 1;	// L3985
          ap_int<66> v2550 = v2549;	// L3986
          ap_int<66> v2551 = v2550 * 8;	// L3987
          ap_int<66> v2552 = v2551 - 1;	// L3988
          int v2553 = v2547;	// L3989
          int v2554 = v2552;	// L3990
          ap_int<128> v2555;
          ap_int<128> v2555_tmp = v2545;
          v2555_tmp(v2554, v2553) = v2544;
          v2555 = v2555_tmp;	// L3991
          packed_c40 = v2555;	// L3992
        } else {
          ap_int<128> v2556 = packed_tmp40;	// L3994
          int64_t v2557 = m40;	// L3995
          int64_t v2558 = v2557 * 8;	// L3996
          ap_int<34> v2559 = m40;	// L3997
          ap_int<34> v2560 = v2559 + 1;	// L3998
          ap_int<66> v2561 = v2560;	// L3999
          ap_int<66> v2562 = v2561 * 8;	// L4000
          ap_int<66> v2563 = v2562 - 1;	// L4001
          int v2564 = v2558;	// L4002
          int v2565 = v2563;	// L4003
          int8_t v2566;
          ap_int<128> v2566_tmp = v2556;
          v2566 = v2566_tmp(v2565, v2564);	// L4004
          ap_int<128> v2567 = packed_c40;	// L4005
          ap_int<128> v2568;
          ap_int<128> v2568_tmp = v2567;
          v2568_tmp(v2565, v2564) = v2566;
          v2568 = v2568_tmp;	// L4015
          packed_c40 = v2568;	// L4016
        }
      }
      ap_int<128> v2569 = packed_c40;	// L4019
      v2517.write(v2569);	// L4020
    }
  }
}

void gemm_3_10(
  hls::stream< int8_t >& v2570,
  hls::stream< int8_t >& v2571,
  hls::stream< int8_t >& v2572,
  hls::stream< int8_t >& v2573,
  hls::stream< ap_uint<128> >& v2574,
  hls::stream< ap_uint<128> >& v2575
) {	// L4025
  l_S_mt_nt_0_mt63: for (int mt63 = 0; mt63 < 4; mt63++) {	// L4033
    l_nt63: for (int nt63 = 0; nt63 < 4; nt63++) {	// L4034
      int8_t c41;	// L4035
      c41 = 0;	// L4036
      l_S_k_0_k63: for (int k63 = 0; k63 < 64; k63++) {	// L4037
        int8_t v2580 = v2570.read();	// L4038
        int8_t a44;	// L4039
        a44 = v2580;	// L4040
        int8_t v2582 = v2571.read();	// L4041
        int8_t b57;	// L4042
        b57 = v2582;	// L4043
        int8_t v2584 = a44;	// L4044
        int8_t v2585 = b57;	// L4045
        int16_t v2586 = v2584;	// L4046
        int16_t v2587 = v2585;	// L4047
        int16_t v2588 = v2586 * v2587;	// L4048
        int8_t v2589 = c41;	// L4049
        ap_int<17> v2590 = v2589;	// L4050
        ap_int<17> v2591 = v2588;	// L4051
        ap_int<17> v2592 = v2590 + v2591;	// L4052
        int8_t v2593 = v2592;	// L4053
        c41 = v2593;	// L4054
        int8_t v2594 = a44;	// L4055
        v2572.write(v2594);	// L4056
        int8_t v2595 = b57;	// L4057
        v2573.write(v2595);	// L4058
      }
      ap_uint<128> packed_tmp41;	// L4060
      ap_uint<128> v2597 = v2574.read();	// L4061
      packed_tmp41 = v2597;	// L4062
      ap_uint<128> packed_c41;	// L4063
      packed_c41 = 0;	// L4064
      l_S_m_1_m41: for (int m41 = 0; m41 < 16; m41++) {	// L4065
        ap_int<33> v2600 = m41;	// L4066
        bool v2601 = v2600 == 2;	// L4067
        if (v2601) {	// L4068
          int8_t v2602 = c41;	// L4069
          ap_int<128> v2603 = packed_c41;	// L4070
          int64_t v2604 = m41;	// L4071
          int64_t v2605 = v2604 * 8;	// L4072
          ap_int<34> v2606 = m41;	// L4073
          ap_int<34> v2607 = v2606 + 1;	// L4074
          ap_int<66> v2608 = v2607;	// L4075
          ap_int<66> v2609 = v2608 * 8;	// L4076
          ap_int<66> v2610 = v2609 - 1;	// L4077
          int v2611 = v2605;	// L4078
          int v2612 = v2610;	// L4079
          ap_int<128> v2613;
          ap_int<128> v2613_tmp = v2603;
          v2613_tmp(v2612, v2611) = v2602;
          v2613 = v2613_tmp;	// L4080
          packed_c41 = v2613;	// L4081
        } else {
          ap_int<128> v2614 = packed_tmp41;	// L4083
          int64_t v2615 = m41;	// L4084
          int64_t v2616 = v2615 * 8;	// L4085
          ap_int<34> v2617 = m41;	// L4086
          ap_int<34> v2618 = v2617 + 1;	// L4087
          ap_int<66> v2619 = v2618;	// L4088
          ap_int<66> v2620 = v2619 * 8;	// L4089
          ap_int<66> v2621 = v2620 - 1;	// L4090
          int v2622 = v2616;	// L4091
          int v2623 = v2621;	// L4092
          int8_t v2624;
          ap_int<128> v2624_tmp = v2614;
          v2624 = v2624_tmp(v2623, v2622);	// L4093
          ap_int<128> v2625 = packed_c41;	// L4094
          ap_int<128> v2626;
          ap_int<128> v2626_tmp = v2625;
          v2626_tmp(v2623, v2622) = v2624;
          v2626 = v2626_tmp;	// L4104
          packed_c41 = v2626;	// L4105
        }
      }
      ap_int<128> v2627 = packed_c41;	// L4108
      v2575.write(v2627);	// L4109
    }
  }
}

void gemm_3_11(
  hls::stream< int8_t >& v2628,
  hls::stream< int8_t >& v2629,
  hls::stream< int8_t >& v2630,
  hls::stream< int8_t >& v2631,
  hls::stream< ap_uint<128> >& v2632,
  hls::stream< ap_uint<128> >& v2633
) {	// L4114
  l_S_mt_nt_0_mt64: for (int mt64 = 0; mt64 < 4; mt64++) {	// L4122
    l_nt64: for (int nt64 = 0; nt64 < 4; nt64++) {	// L4123
      int8_t c42;	// L4124
      c42 = 0;	// L4125
      l_S_k_0_k64: for (int k64 = 0; k64 < 64; k64++) {	// L4126
        int8_t v2638 = v2628.read();	// L4127
        int8_t a45;	// L4128
        a45 = v2638;	// L4129
        int8_t v2640 = v2629.read();	// L4130
        int8_t b58;	// L4131
        b58 = v2640;	// L4132
        int8_t v2642 = a45;	// L4133
        int8_t v2643 = b58;	// L4134
        int16_t v2644 = v2642;	// L4135
        int16_t v2645 = v2643;	// L4136
        int16_t v2646 = v2644 * v2645;	// L4137
        int8_t v2647 = c42;	// L4138
        ap_int<17> v2648 = v2647;	// L4139
        ap_int<17> v2649 = v2646;	// L4140
        ap_int<17> v2650 = v2648 + v2649;	// L4141
        int8_t v2651 = v2650;	// L4142
        c42 = v2651;	// L4143
        int8_t v2652 = a45;	// L4144
        v2630.write(v2652);	// L4145
        int8_t v2653 = b58;	// L4146
        v2631.write(v2653);	// L4147
      }
      ap_uint<128> packed_tmp42;	// L4149
      ap_uint<128> v2655 = v2632.read();	// L4150
      packed_tmp42 = v2655;	// L4151
      ap_uint<128> packed_c42;	// L4152
      packed_c42 = 0;	// L4153
      l_S_m_1_m42: for (int m42 = 0; m42 < 16; m42++) {	// L4154
        ap_int<33> v2658 = m42;	// L4155
        bool v2659 = v2658 == 2;	// L4156
        if (v2659) {	// L4157
          int8_t v2660 = c42;	// L4158
          ap_int<128> v2661 = packed_c42;	// L4159
          int64_t v2662 = m42;	// L4160
          int64_t v2663 = v2662 * 8;	// L4161
          ap_int<34> v2664 = m42;	// L4162
          ap_int<34> v2665 = v2664 + 1;	// L4163
          ap_int<66> v2666 = v2665;	// L4164
          ap_int<66> v2667 = v2666 * 8;	// L4165
          ap_int<66> v2668 = v2667 - 1;	// L4166
          int v2669 = v2663;	// L4167
          int v2670 = v2668;	// L4168
          ap_int<128> v2671;
          ap_int<128> v2671_tmp = v2661;
          v2671_tmp(v2670, v2669) = v2660;
          v2671 = v2671_tmp;	// L4169
          packed_c42 = v2671;	// L4170
        } else {
          ap_int<128> v2672 = packed_tmp42;	// L4172
          int64_t v2673 = m42;	// L4173
          int64_t v2674 = v2673 * 8;	// L4174
          ap_int<34> v2675 = m42;	// L4175
          ap_int<34> v2676 = v2675 + 1;	// L4176
          ap_int<66> v2677 = v2676;	// L4177
          ap_int<66> v2678 = v2677 * 8;	// L4178
          ap_int<66> v2679 = v2678 - 1;	// L4179
          int v2680 = v2674;	// L4180
          int v2681 = v2679;	// L4181
          int8_t v2682;
          ap_int<128> v2682_tmp = v2672;
          v2682 = v2682_tmp(v2681, v2680);	// L4182
          ap_int<128> v2683 = packed_c42;	// L4183
          ap_int<128> v2684;
          ap_int<128> v2684_tmp = v2683;
          v2684_tmp(v2681, v2680) = v2682;
          v2684 = v2684_tmp;	// L4193
          packed_c42 = v2684;	// L4194
        }
      }
      ap_int<128> v2685 = packed_c42;	// L4197
      v2633.write(v2685);	// L4198
    }
  }
}

void gemm_3_12(
  hls::stream< int8_t >& v2686,
  hls::stream< int8_t >& v2687,
  hls::stream< int8_t >& v2688,
  hls::stream< int8_t >& v2689,
  hls::stream< ap_uint<128> >& v2690,
  hls::stream< ap_uint<128> >& v2691
) {	// L4203
  l_S_mt_nt_0_mt65: for (int mt65 = 0; mt65 < 4; mt65++) {	// L4211
    l_nt65: for (int nt65 = 0; nt65 < 4; nt65++) {	// L4212
      int8_t c43;	// L4213
      c43 = 0;	// L4214
      l_S_k_0_k65: for (int k65 = 0; k65 < 64; k65++) {	// L4215
        int8_t v2696 = v2686.read();	// L4216
        int8_t a46;	// L4217
        a46 = v2696;	// L4218
        int8_t v2698 = v2687.read();	// L4219
        int8_t b59;	// L4220
        b59 = v2698;	// L4221
        int8_t v2700 = a46;	// L4222
        int8_t v2701 = b59;	// L4223
        int16_t v2702 = v2700;	// L4224
        int16_t v2703 = v2701;	// L4225
        int16_t v2704 = v2702 * v2703;	// L4226
        int8_t v2705 = c43;	// L4227
        ap_int<17> v2706 = v2705;	// L4228
        ap_int<17> v2707 = v2704;	// L4229
        ap_int<17> v2708 = v2706 + v2707;	// L4230
        int8_t v2709 = v2708;	// L4231
        c43 = v2709;	// L4232
        int8_t v2710 = a46;	// L4233
        v2688.write(v2710);	// L4234
        int8_t v2711 = b59;	// L4235
        v2689.write(v2711);	// L4236
      }
      ap_uint<128> packed_tmp43;	// L4238
      ap_uint<128> v2713 = v2690.read();	// L4239
      packed_tmp43 = v2713;	// L4240
      ap_uint<128> packed_c43;	// L4241
      packed_c43 = 0;	// L4242
      l_S_m_1_m43: for (int m43 = 0; m43 < 16; m43++) {	// L4243
        ap_int<33> v2716 = m43;	// L4244
        bool v2717 = v2716 == 2;	// L4245
        if (v2717) {	// L4246
          int8_t v2718 = c43;	// L4247
          ap_int<128> v2719 = packed_c43;	// L4248
          int64_t v2720 = m43;	// L4249
          int64_t v2721 = v2720 * 8;	// L4250
          ap_int<34> v2722 = m43;	// L4251
          ap_int<34> v2723 = v2722 + 1;	// L4252
          ap_int<66> v2724 = v2723;	// L4253
          ap_int<66> v2725 = v2724 * 8;	// L4254
          ap_int<66> v2726 = v2725 - 1;	// L4255
          int v2727 = v2721;	// L4256
          int v2728 = v2726;	// L4257
          ap_int<128> v2729;
          ap_int<128> v2729_tmp = v2719;
          v2729_tmp(v2728, v2727) = v2718;
          v2729 = v2729_tmp;	// L4258
          packed_c43 = v2729;	// L4259
        } else {
          ap_int<128> v2730 = packed_tmp43;	// L4261
          int64_t v2731 = m43;	// L4262
          int64_t v2732 = v2731 * 8;	// L4263
          ap_int<34> v2733 = m43;	// L4264
          ap_int<34> v2734 = v2733 + 1;	// L4265
          ap_int<66> v2735 = v2734;	// L4266
          ap_int<66> v2736 = v2735 * 8;	// L4267
          ap_int<66> v2737 = v2736 - 1;	// L4268
          int v2738 = v2732;	// L4269
          int v2739 = v2737;	// L4270
          int8_t v2740;
          ap_int<128> v2740_tmp = v2730;
          v2740 = v2740_tmp(v2739, v2738);	// L4271
          ap_int<128> v2741 = packed_c43;	// L4272
          ap_int<128> v2742;
          ap_int<128> v2742_tmp = v2741;
          v2742_tmp(v2739, v2738) = v2740;
          v2742 = v2742_tmp;	// L4282
          packed_c43 = v2742;	// L4283
        }
      }
      ap_int<128> v2743 = packed_c43;	// L4286
      v2691.write(v2743);	// L4287
    }
  }
}

void gemm_3_13(
  hls::stream< int8_t >& v2744,
  hls::stream< int8_t >& v2745,
  hls::stream< int8_t >& v2746,
  hls::stream< int8_t >& v2747,
  hls::stream< ap_uint<128> >& v2748,
  hls::stream< ap_uint<128> >& v2749
) {	// L4292
  l_S_mt_nt_0_mt66: for (int mt66 = 0; mt66 < 4; mt66++) {	// L4300
    l_nt66: for (int nt66 = 0; nt66 < 4; nt66++) {	// L4301
      int8_t c44;	// L4302
      c44 = 0;	// L4303
      l_S_k_0_k66: for (int k66 = 0; k66 < 64; k66++) {	// L4304
        int8_t v2754 = v2744.read();	// L4305
        int8_t a47;	// L4306
        a47 = v2754;	// L4307
        int8_t v2756 = v2745.read();	// L4308
        int8_t b60;	// L4309
        b60 = v2756;	// L4310
        int8_t v2758 = a47;	// L4311
        int8_t v2759 = b60;	// L4312
        int16_t v2760 = v2758;	// L4313
        int16_t v2761 = v2759;	// L4314
        int16_t v2762 = v2760 * v2761;	// L4315
        int8_t v2763 = c44;	// L4316
        ap_int<17> v2764 = v2763;	// L4317
        ap_int<17> v2765 = v2762;	// L4318
        ap_int<17> v2766 = v2764 + v2765;	// L4319
        int8_t v2767 = v2766;	// L4320
        c44 = v2767;	// L4321
        int8_t v2768 = a47;	// L4322
        v2746.write(v2768);	// L4323
        int8_t v2769 = b60;	// L4324
        v2747.write(v2769);	// L4325
      }
      ap_uint<128> packed_tmp44;	// L4327
      ap_uint<128> v2771 = v2748.read();	// L4328
      packed_tmp44 = v2771;	// L4329
      ap_uint<128> packed_c44;	// L4330
      packed_c44 = 0;	// L4331
      l_S_m_1_m44: for (int m44 = 0; m44 < 16; m44++) {	// L4332
        ap_int<33> v2774 = m44;	// L4333
        bool v2775 = v2774 == 2;	// L4334
        if (v2775) {	// L4335
          int8_t v2776 = c44;	// L4336
          ap_int<128> v2777 = packed_c44;	// L4337
          int64_t v2778 = m44;	// L4338
          int64_t v2779 = v2778 * 8;	// L4339
          ap_int<34> v2780 = m44;	// L4340
          ap_int<34> v2781 = v2780 + 1;	// L4341
          ap_int<66> v2782 = v2781;	// L4342
          ap_int<66> v2783 = v2782 * 8;	// L4343
          ap_int<66> v2784 = v2783 - 1;	// L4344
          int v2785 = v2779;	// L4345
          int v2786 = v2784;	// L4346
          ap_int<128> v2787;
          ap_int<128> v2787_tmp = v2777;
          v2787_tmp(v2786, v2785) = v2776;
          v2787 = v2787_tmp;	// L4347
          packed_c44 = v2787;	// L4348
        } else {
          ap_int<128> v2788 = packed_tmp44;	// L4350
          int64_t v2789 = m44;	// L4351
          int64_t v2790 = v2789 * 8;	// L4352
          ap_int<34> v2791 = m44;	// L4353
          ap_int<34> v2792 = v2791 + 1;	// L4354
          ap_int<66> v2793 = v2792;	// L4355
          ap_int<66> v2794 = v2793 * 8;	// L4356
          ap_int<66> v2795 = v2794 - 1;	// L4357
          int v2796 = v2790;	// L4358
          int v2797 = v2795;	// L4359
          int8_t v2798;
          ap_int<128> v2798_tmp = v2788;
          v2798 = v2798_tmp(v2797, v2796);	// L4360
          ap_int<128> v2799 = packed_c44;	// L4361
          ap_int<128> v2800;
          ap_int<128> v2800_tmp = v2799;
          v2800_tmp(v2797, v2796) = v2798;
          v2800 = v2800_tmp;	// L4371
          packed_c44 = v2800;	// L4372
        }
      }
      ap_int<128> v2801 = packed_c44;	// L4375
      v2749.write(v2801);	// L4376
    }
  }
}

void gemm_3_14(
  hls::stream< int8_t >& v2802,
  hls::stream< int8_t >& v2803,
  hls::stream< int8_t >& v2804,
  hls::stream< int8_t >& v2805,
  hls::stream< ap_uint<128> >& v2806,
  hls::stream< ap_uint<128> >& v2807
) {	// L4381
  l_S_mt_nt_0_mt67: for (int mt67 = 0; mt67 < 4; mt67++) {	// L4389
    l_nt67: for (int nt67 = 0; nt67 < 4; nt67++) {	// L4390
      int8_t c45;	// L4391
      c45 = 0;	// L4392
      l_S_k_0_k67: for (int k67 = 0; k67 < 64; k67++) {	// L4393
        int8_t v2812 = v2802.read();	// L4394
        int8_t a48;	// L4395
        a48 = v2812;	// L4396
        int8_t v2814 = v2803.read();	// L4397
        int8_t b61;	// L4398
        b61 = v2814;	// L4399
        int8_t v2816 = a48;	// L4400
        int8_t v2817 = b61;	// L4401
        int16_t v2818 = v2816;	// L4402
        int16_t v2819 = v2817;	// L4403
        int16_t v2820 = v2818 * v2819;	// L4404
        int8_t v2821 = c45;	// L4405
        ap_int<17> v2822 = v2821;	// L4406
        ap_int<17> v2823 = v2820;	// L4407
        ap_int<17> v2824 = v2822 + v2823;	// L4408
        int8_t v2825 = v2824;	// L4409
        c45 = v2825;	// L4410
        int8_t v2826 = a48;	// L4411
        v2804.write(v2826);	// L4412
        int8_t v2827 = b61;	// L4413
        v2805.write(v2827);	// L4414
      }
      ap_uint<128> packed_tmp45;	// L4416
      ap_uint<128> v2829 = v2806.read();	// L4417
      packed_tmp45 = v2829;	// L4418
      ap_uint<128> packed_c45;	// L4419
      packed_c45 = 0;	// L4420
      l_S_m_1_m45: for (int m45 = 0; m45 < 16; m45++) {	// L4421
        ap_int<33> v2832 = m45;	// L4422
        bool v2833 = v2832 == 2;	// L4423
        if (v2833) {	// L4424
          int8_t v2834 = c45;	// L4425
          ap_int<128> v2835 = packed_c45;	// L4426
          int64_t v2836 = m45;	// L4427
          int64_t v2837 = v2836 * 8;	// L4428
          ap_int<34> v2838 = m45;	// L4429
          ap_int<34> v2839 = v2838 + 1;	// L4430
          ap_int<66> v2840 = v2839;	// L4431
          ap_int<66> v2841 = v2840 * 8;	// L4432
          ap_int<66> v2842 = v2841 - 1;	// L4433
          int v2843 = v2837;	// L4434
          int v2844 = v2842;	// L4435
          ap_int<128> v2845;
          ap_int<128> v2845_tmp = v2835;
          v2845_tmp(v2844, v2843) = v2834;
          v2845 = v2845_tmp;	// L4436
          packed_c45 = v2845;	// L4437
        } else {
          ap_int<128> v2846 = packed_tmp45;	// L4439
          int64_t v2847 = m45;	// L4440
          int64_t v2848 = v2847 * 8;	// L4441
          ap_int<34> v2849 = m45;	// L4442
          ap_int<34> v2850 = v2849 + 1;	// L4443
          ap_int<66> v2851 = v2850;	// L4444
          ap_int<66> v2852 = v2851 * 8;	// L4445
          ap_int<66> v2853 = v2852 - 1;	// L4446
          int v2854 = v2848;	// L4447
          int v2855 = v2853;	// L4448
          int8_t v2856;
          ap_int<128> v2856_tmp = v2846;
          v2856 = v2856_tmp(v2855, v2854);	// L4449
          ap_int<128> v2857 = packed_c45;	// L4450
          ap_int<128> v2858;
          ap_int<128> v2858_tmp = v2857;
          v2858_tmp(v2855, v2854) = v2856;
          v2858 = v2858_tmp;	// L4460
          packed_c45 = v2858;	// L4461
        }
      }
      ap_int<128> v2859 = packed_c45;	// L4464
      v2807.write(v2859);	// L4465
    }
  }
}

void gemm_3_15(
  hls::stream< int8_t >& v2860,
  hls::stream< int8_t >& v2861,
  hls::stream< int8_t >& v2862,
  hls::stream< int8_t >& v2863,
  hls::stream< ap_uint<128> >& v2864,
  hls::stream< ap_uint<128> >& v2865
) {	// L4470
  l_S_mt_nt_0_mt68: for (int mt68 = 0; mt68 < 4; mt68++) {	// L4478
    l_nt68: for (int nt68 = 0; nt68 < 4; nt68++) {	// L4479
      int8_t c46;	// L4480
      c46 = 0;	// L4481
      l_S_k_0_k68: for (int k68 = 0; k68 < 64; k68++) {	// L4482
        int8_t v2870 = v2860.read();	// L4483
        int8_t a49;	// L4484
        a49 = v2870;	// L4485
        int8_t v2872 = v2861.read();	// L4486
        int8_t b62;	// L4487
        b62 = v2872;	// L4488
        int8_t v2874 = a49;	// L4489
        int8_t v2875 = b62;	// L4490
        int16_t v2876 = v2874;	// L4491
        int16_t v2877 = v2875;	// L4492
        int16_t v2878 = v2876 * v2877;	// L4493
        int8_t v2879 = c46;	// L4494
        ap_int<17> v2880 = v2879;	// L4495
        ap_int<17> v2881 = v2878;	// L4496
        ap_int<17> v2882 = v2880 + v2881;	// L4497
        int8_t v2883 = v2882;	// L4498
        c46 = v2883;	// L4499
        int8_t v2884 = a49;	// L4500
        v2862.write(v2884);	// L4501
        int8_t v2885 = b62;	// L4502
        v2863.write(v2885);	// L4503
      }
      ap_uint<128> packed_tmp46;	// L4505
      ap_uint<128> v2887 = v2864.read();	// L4506
      packed_tmp46 = v2887;	// L4507
      ap_uint<128> packed_c46;	// L4508
      packed_c46 = 0;	// L4509
      l_S_m_1_m46: for (int m46 = 0; m46 < 16; m46++) {	// L4510
        ap_int<33> v2890 = m46;	// L4511
        bool v2891 = v2890 == 2;	// L4512
        if (v2891) {	// L4513
          int8_t v2892 = c46;	// L4514
          ap_int<128> v2893 = packed_c46;	// L4515
          int64_t v2894 = m46;	// L4516
          int64_t v2895 = v2894 * 8;	// L4517
          ap_int<34> v2896 = m46;	// L4518
          ap_int<34> v2897 = v2896 + 1;	// L4519
          ap_int<66> v2898 = v2897;	// L4520
          ap_int<66> v2899 = v2898 * 8;	// L4521
          ap_int<66> v2900 = v2899 - 1;	// L4522
          int v2901 = v2895;	// L4523
          int v2902 = v2900;	// L4524
          ap_int<128> v2903;
          ap_int<128> v2903_tmp = v2893;
          v2903_tmp(v2902, v2901) = v2892;
          v2903 = v2903_tmp;	// L4525
          packed_c46 = v2903;	// L4526
        } else {
          ap_int<128> v2904 = packed_tmp46;	// L4528
          int64_t v2905 = m46;	// L4529
          int64_t v2906 = v2905 * 8;	// L4530
          ap_int<34> v2907 = m46;	// L4531
          ap_int<34> v2908 = v2907 + 1;	// L4532
          ap_int<66> v2909 = v2908;	// L4533
          ap_int<66> v2910 = v2909 * 8;	// L4534
          ap_int<66> v2911 = v2910 - 1;	// L4535
          int v2912 = v2906;	// L4536
          int v2913 = v2911;	// L4537
          int8_t v2914;
          ap_int<128> v2914_tmp = v2904;
          v2914 = v2914_tmp(v2913, v2912);	// L4538
          ap_int<128> v2915 = packed_c46;	// L4539
          ap_int<128> v2916;
          ap_int<128> v2916_tmp = v2915;
          v2916_tmp(v2913, v2912) = v2914;
          v2916 = v2916_tmp;	// L4549
          packed_c46 = v2916;	// L4550
        }
      }
      ap_int<128> v2917 = packed_c46;	// L4553
      v2865.write(v2917);	// L4554
    }
  }
}

void gemm_3_16(
  hls::stream< int8_t >& v2918,
  hls::stream< int8_t >& v2919,
  hls::stream< int8_t >& v2920,
  hls::stream< ap_uint<128> >& v2921,
  hls::stream< ap_uint<128> >& v2922
) {	// L4559
  l_S_mt_nt_0_mt69: for (int mt69 = 0; mt69 < 4; mt69++) {	// L4567
    l_nt69: for (int nt69 = 0; nt69 < 4; nt69++) {	// L4568
      int8_t c47;	// L4569
      c47 = 0;	// L4570
      l_S_k_0_k69: for (int k69 = 0; k69 < 64; k69++) {	// L4571
        int8_t v2927 = v2918.read();	// L4572
        int8_t a50;	// L4573
        a50 = v2927;	// L4574
        int8_t v2929 = v2919.read();	// L4575
        int8_t b63;	// L4576
        b63 = v2929;	// L4577
        int8_t v2931 = a50;	// L4578
        int8_t v2932 = b63;	// L4579
        int16_t v2933 = v2931;	// L4580
        int16_t v2934 = v2932;	// L4581
        int16_t v2935 = v2933 * v2934;	// L4582
        int8_t v2936 = c47;	// L4583
        ap_int<17> v2937 = v2936;	// L4584
        ap_int<17> v2938 = v2935;	// L4585
        ap_int<17> v2939 = v2937 + v2938;	// L4586
        int8_t v2940 = v2939;	// L4587
        c47 = v2940;	// L4588
        int8_t v2941 = b63;	// L4589
        v2920.write(v2941);	// L4590
      }
      ap_uint<128> packed_tmp47;	// L4592
      ap_uint<128> v2943 = v2921.read();	// L4593
      packed_tmp47 = v2943;	// L4594
      ap_uint<128> packed_c47;	// L4595
      packed_c47 = 0;	// L4596
      l_S_m_1_m47: for (int m47 = 0; m47 < 16; m47++) {	// L4597
        ap_int<33> v2946 = m47;	// L4598
        bool v2947 = v2946 == 2;	// L4599
        if (v2947) {	// L4600
          int8_t v2948 = c47;	// L4601
          ap_int<128> v2949 = packed_c47;	// L4602
          int64_t v2950 = m47;	// L4603
          int64_t v2951 = v2950 * 8;	// L4604
          ap_int<34> v2952 = m47;	// L4605
          ap_int<34> v2953 = v2952 + 1;	// L4606
          ap_int<66> v2954 = v2953;	// L4607
          ap_int<66> v2955 = v2954 * 8;	// L4608
          ap_int<66> v2956 = v2955 - 1;	// L4609
          int v2957 = v2951;	// L4610
          int v2958 = v2956;	// L4611
          ap_int<128> v2959;
          ap_int<128> v2959_tmp = v2949;
          v2959_tmp(v2958, v2957) = v2948;
          v2959 = v2959_tmp;	// L4612
          packed_c47 = v2959;	// L4613
        } else {
          ap_int<128> v2960 = packed_tmp47;	// L4615
          int64_t v2961 = m47;	// L4616
          int64_t v2962 = v2961 * 8;	// L4617
          ap_int<34> v2963 = m47;	// L4618
          ap_int<34> v2964 = v2963 + 1;	// L4619
          ap_int<66> v2965 = v2964;	// L4620
          ap_int<66> v2966 = v2965 * 8;	// L4621
          ap_int<66> v2967 = v2966 - 1;	// L4622
          int v2968 = v2962;	// L4623
          int v2969 = v2967;	// L4624
          int8_t v2970;
          ap_int<128> v2970_tmp = v2960;
          v2970 = v2970_tmp(v2969, v2968);	// L4625
          ap_int<128> v2971 = packed_c47;	// L4626
          ap_int<128> v2972;
          ap_int<128> v2972_tmp = v2971;
          v2972_tmp(v2969, v2968) = v2970;
          v2972 = v2972_tmp;	// L4636
          packed_c47 = v2972;	// L4637
        }
      }
      ap_int<128> v2973 = packed_c47;	// L4640
      v2922.write(v2973);	// L4641
    }
  }
}

void gemm_4_0(
  hls::stream< ap_uint<128> >& v2974,
  hls::stream< int8_t >& v2975,
  hls::stream< ap_uint<128> >& v2976
) {	// L4646
  l_S_mt_nt_0_mt70: for (int mt70 = 0; mt70 < 4; mt70++) {	// L4649
    l_nt70: for (int nt70 = 0; nt70 < 4; nt70++) {	// L4650
      l_S_k_0_k70: for (int k70 = 0; k70 < 64; k70++) {	// L4651
        ap_uint<128> v2980 = v2974.read();	// L4652
        ap_uint<128> a51;	// L4653
        a51 = v2980;	// L4654
        ap_int<128> v2982 = a51;	// L4655
        int8_t v2983;
        ap_int<128> v2983_tmp = v2982;
        v2983 = v2983_tmp(31, 24);	// L4656
        v2975.write(v2983);	// L4657
        ap_int<128> v2984 = a51;	// L4658
        v2976.write(v2984);	// L4659
      }
    }
  }
}

void gemm_4_1(
  hls::stream< int8_t >& v2985,
  hls::stream< int8_t >& v2986,
  hls::stream< int8_t >& v2987,
  hls::stream< int8_t >& v2988,
  hls::stream< ap_uint<128> >& v2989,
  hls::stream< ap_uint<128> >& v2990
) {	// L4665
  l_S_mt_nt_0_mt71: for (int mt71 = 0; mt71 < 4; mt71++) {	// L4673
    l_nt71: for (int nt71 = 0; nt71 < 4; nt71++) {	// L4674
      int8_t c48;	// L4675
      c48 = 0;	// L4676
      l_S_k_0_k71: for (int k71 = 0; k71 < 64; k71++) {	// L4677
        int8_t v2995 = v2985.read();	// L4678
        int8_t a52;	// L4679
        a52 = v2995;	// L4680
        int8_t v2997 = v2986.read();	// L4681
        int8_t b64;	// L4682
        b64 = v2997;	// L4683
        int8_t v2999 = a52;	// L4684
        int8_t v3000 = b64;	// L4685
        int16_t v3001 = v2999;	// L4686
        int16_t v3002 = v3000;	// L4687
        int16_t v3003 = v3001 * v3002;	// L4688
        int8_t v3004 = c48;	// L4689
        ap_int<17> v3005 = v3004;	// L4690
        ap_int<17> v3006 = v3003;	// L4691
        ap_int<17> v3007 = v3005 + v3006;	// L4692
        int8_t v3008 = v3007;	// L4693
        c48 = v3008;	// L4694
        int8_t v3009 = a52;	// L4695
        v2987.write(v3009);	// L4696
        int8_t v3010 = b64;	// L4697
        v2988.write(v3010);	// L4698
      }
      ap_uint<128> packed_tmp48;	// L4700
      ap_uint<128> v3012 = v2989.read();	// L4701
      packed_tmp48 = v3012;	// L4702
      ap_uint<128> packed_c48;	// L4703
      packed_c48 = 0;	// L4704
      l_S_m_1_m48: for (int m48 = 0; m48 < 16; m48++) {	// L4705
        ap_int<33> v3015 = m48;	// L4706
        bool v3016 = v3015 == 3;	// L4707
        if (v3016) {	// L4708
          int8_t v3017 = c48;	// L4709
          ap_int<128> v3018 = packed_c48;	// L4710
          int64_t v3019 = m48;	// L4711
          int64_t v3020 = v3019 * 8;	// L4712
          ap_int<34> v3021 = m48;	// L4713
          ap_int<34> v3022 = v3021 + 1;	// L4714
          ap_int<66> v3023 = v3022;	// L4715
          ap_int<66> v3024 = v3023 * 8;	// L4716
          ap_int<66> v3025 = v3024 - 1;	// L4717
          int v3026 = v3020;	// L4718
          int v3027 = v3025;	// L4719
          ap_int<128> v3028;
          ap_int<128> v3028_tmp = v3018;
          v3028_tmp(v3027, v3026) = v3017;
          v3028 = v3028_tmp;	// L4720
          packed_c48 = v3028;	// L4721
        } else {
          ap_int<128> v3029 = packed_tmp48;	// L4723
          int64_t v3030 = m48;	// L4724
          int64_t v3031 = v3030 * 8;	// L4725
          ap_int<34> v3032 = m48;	// L4726
          ap_int<34> v3033 = v3032 + 1;	// L4727
          ap_int<66> v3034 = v3033;	// L4728
          ap_int<66> v3035 = v3034 * 8;	// L4729
          ap_int<66> v3036 = v3035 - 1;	// L4730
          int v3037 = v3031;	// L4731
          int v3038 = v3036;	// L4732
          int8_t v3039;
          ap_int<128> v3039_tmp = v3029;
          v3039 = v3039_tmp(v3038, v3037);	// L4733
          ap_int<128> v3040 = packed_c48;	// L4734
          ap_int<128> v3041;
          ap_int<128> v3041_tmp = v3040;
          v3041_tmp(v3038, v3037) = v3039;
          v3041 = v3041_tmp;	// L4744
          packed_c48 = v3041;	// L4745
        }
      }
      ap_int<128> v3042 = packed_c48;	// L4748
      v2990.write(v3042);	// L4749
    }
  }
}

void gemm_4_2(
  hls::stream< int8_t >& v3043,
  hls::stream< int8_t >& v3044,
  hls::stream< int8_t >& v3045,
  hls::stream< int8_t >& v3046,
  hls::stream< ap_uint<128> >& v3047,
  hls::stream< ap_uint<128> >& v3048
) {	// L4754
  l_S_mt_nt_0_mt72: for (int mt72 = 0; mt72 < 4; mt72++) {	// L4762
    l_nt72: for (int nt72 = 0; nt72 < 4; nt72++) {	// L4763
      int8_t c49;	// L4764
      c49 = 0;	// L4765
      l_S_k_0_k72: for (int k72 = 0; k72 < 64; k72++) {	// L4766
        int8_t v3053 = v3043.read();	// L4767
        int8_t a53;	// L4768
        a53 = v3053;	// L4769
        int8_t v3055 = v3044.read();	// L4770
        int8_t b65;	// L4771
        b65 = v3055;	// L4772
        int8_t v3057 = a53;	// L4773
        int8_t v3058 = b65;	// L4774
        int16_t v3059 = v3057;	// L4775
        int16_t v3060 = v3058;	// L4776
        int16_t v3061 = v3059 * v3060;	// L4777
        int8_t v3062 = c49;	// L4778
        ap_int<17> v3063 = v3062;	// L4779
        ap_int<17> v3064 = v3061;	// L4780
        ap_int<17> v3065 = v3063 + v3064;	// L4781
        int8_t v3066 = v3065;	// L4782
        c49 = v3066;	// L4783
        int8_t v3067 = a53;	// L4784
        v3045.write(v3067);	// L4785
        int8_t v3068 = b65;	// L4786
        v3046.write(v3068);	// L4787
      }
      ap_uint<128> packed_tmp49;	// L4789
      ap_uint<128> v3070 = v3047.read();	// L4790
      packed_tmp49 = v3070;	// L4791
      ap_uint<128> packed_c49;	// L4792
      packed_c49 = 0;	// L4793
      l_S_m_1_m49: for (int m49 = 0; m49 < 16; m49++) {	// L4794
        ap_int<33> v3073 = m49;	// L4795
        bool v3074 = v3073 == 3;	// L4796
        if (v3074) {	// L4797
          int8_t v3075 = c49;	// L4798
          ap_int<128> v3076 = packed_c49;	// L4799
          int64_t v3077 = m49;	// L4800
          int64_t v3078 = v3077 * 8;	// L4801
          ap_int<34> v3079 = m49;	// L4802
          ap_int<34> v3080 = v3079 + 1;	// L4803
          ap_int<66> v3081 = v3080;	// L4804
          ap_int<66> v3082 = v3081 * 8;	// L4805
          ap_int<66> v3083 = v3082 - 1;	// L4806
          int v3084 = v3078;	// L4807
          int v3085 = v3083;	// L4808
          ap_int<128> v3086;
          ap_int<128> v3086_tmp = v3076;
          v3086_tmp(v3085, v3084) = v3075;
          v3086 = v3086_tmp;	// L4809
          packed_c49 = v3086;	// L4810
        } else {
          ap_int<128> v3087 = packed_tmp49;	// L4812
          int64_t v3088 = m49;	// L4813
          int64_t v3089 = v3088 * 8;	// L4814
          ap_int<34> v3090 = m49;	// L4815
          ap_int<34> v3091 = v3090 + 1;	// L4816
          ap_int<66> v3092 = v3091;	// L4817
          ap_int<66> v3093 = v3092 * 8;	// L4818
          ap_int<66> v3094 = v3093 - 1;	// L4819
          int v3095 = v3089;	// L4820
          int v3096 = v3094;	// L4821
          int8_t v3097;
          ap_int<128> v3097_tmp = v3087;
          v3097 = v3097_tmp(v3096, v3095);	// L4822
          ap_int<128> v3098 = packed_c49;	// L4823
          ap_int<128> v3099;
          ap_int<128> v3099_tmp = v3098;
          v3099_tmp(v3096, v3095) = v3097;
          v3099 = v3099_tmp;	// L4833
          packed_c49 = v3099;	// L4834
        }
      }
      ap_int<128> v3100 = packed_c49;	// L4837
      v3048.write(v3100);	// L4838
    }
  }
}

void gemm_4_3(
  hls::stream< int8_t >& v3101,
  hls::stream< int8_t >& v3102,
  hls::stream< int8_t >& v3103,
  hls::stream< int8_t >& v3104,
  hls::stream< ap_uint<128> >& v3105,
  hls::stream< ap_uint<128> >& v3106
) {	// L4843
  l_S_mt_nt_0_mt73: for (int mt73 = 0; mt73 < 4; mt73++) {	// L4851
    l_nt73: for (int nt73 = 0; nt73 < 4; nt73++) {	// L4852
      int8_t c50;	// L4853
      c50 = 0;	// L4854
      l_S_k_0_k73: for (int k73 = 0; k73 < 64; k73++) {	// L4855
        int8_t v3111 = v3101.read();	// L4856
        int8_t a54;	// L4857
        a54 = v3111;	// L4858
        int8_t v3113 = v3102.read();	// L4859
        int8_t b66;	// L4860
        b66 = v3113;	// L4861
        int8_t v3115 = a54;	// L4862
        int8_t v3116 = b66;	// L4863
        int16_t v3117 = v3115;	// L4864
        int16_t v3118 = v3116;	// L4865
        int16_t v3119 = v3117 * v3118;	// L4866
        int8_t v3120 = c50;	// L4867
        ap_int<17> v3121 = v3120;	// L4868
        ap_int<17> v3122 = v3119;	// L4869
        ap_int<17> v3123 = v3121 + v3122;	// L4870
        int8_t v3124 = v3123;	// L4871
        c50 = v3124;	// L4872
        int8_t v3125 = a54;	// L4873
        v3103.write(v3125);	// L4874
        int8_t v3126 = b66;	// L4875
        v3104.write(v3126);	// L4876
      }
      ap_uint<128> packed_tmp50;	// L4878
      ap_uint<128> v3128 = v3105.read();	// L4879
      packed_tmp50 = v3128;	// L4880
      ap_uint<128> packed_c50;	// L4881
      packed_c50 = 0;	// L4882
      l_S_m_1_m50: for (int m50 = 0; m50 < 16; m50++) {	// L4883
        ap_int<33> v3131 = m50;	// L4884
        bool v3132 = v3131 == 3;	// L4885
        if (v3132) {	// L4886
          int8_t v3133 = c50;	// L4887
          ap_int<128> v3134 = packed_c50;	// L4888
          int64_t v3135 = m50;	// L4889
          int64_t v3136 = v3135 * 8;	// L4890
          ap_int<34> v3137 = m50;	// L4891
          ap_int<34> v3138 = v3137 + 1;	// L4892
          ap_int<66> v3139 = v3138;	// L4893
          ap_int<66> v3140 = v3139 * 8;	// L4894
          ap_int<66> v3141 = v3140 - 1;	// L4895
          int v3142 = v3136;	// L4896
          int v3143 = v3141;	// L4897
          ap_int<128> v3144;
          ap_int<128> v3144_tmp = v3134;
          v3144_tmp(v3143, v3142) = v3133;
          v3144 = v3144_tmp;	// L4898
          packed_c50 = v3144;	// L4899
        } else {
          ap_int<128> v3145 = packed_tmp50;	// L4901
          int64_t v3146 = m50;	// L4902
          int64_t v3147 = v3146 * 8;	// L4903
          ap_int<34> v3148 = m50;	// L4904
          ap_int<34> v3149 = v3148 + 1;	// L4905
          ap_int<66> v3150 = v3149;	// L4906
          ap_int<66> v3151 = v3150 * 8;	// L4907
          ap_int<66> v3152 = v3151 - 1;	// L4908
          int v3153 = v3147;	// L4909
          int v3154 = v3152;	// L4910
          int8_t v3155;
          ap_int<128> v3155_tmp = v3145;
          v3155 = v3155_tmp(v3154, v3153);	// L4911
          ap_int<128> v3156 = packed_c50;	// L4912
          ap_int<128> v3157;
          ap_int<128> v3157_tmp = v3156;
          v3157_tmp(v3154, v3153) = v3155;
          v3157 = v3157_tmp;	// L4922
          packed_c50 = v3157;	// L4923
        }
      }
      ap_int<128> v3158 = packed_c50;	// L4926
      v3106.write(v3158);	// L4927
    }
  }
}

void gemm_4_4(
  hls::stream< int8_t >& v3159,
  hls::stream< int8_t >& v3160,
  hls::stream< int8_t >& v3161,
  hls::stream< int8_t >& v3162,
  hls::stream< ap_uint<128> >& v3163,
  hls::stream< ap_uint<128> >& v3164
) {	// L4932
  l_S_mt_nt_0_mt74: for (int mt74 = 0; mt74 < 4; mt74++) {	// L4940
    l_nt74: for (int nt74 = 0; nt74 < 4; nt74++) {	// L4941
      int8_t c51;	// L4942
      c51 = 0;	// L4943
      l_S_k_0_k74: for (int k74 = 0; k74 < 64; k74++) {	// L4944
        int8_t v3169 = v3159.read();	// L4945
        int8_t a55;	// L4946
        a55 = v3169;	// L4947
        int8_t v3171 = v3160.read();	// L4948
        int8_t b67;	// L4949
        b67 = v3171;	// L4950
        int8_t v3173 = a55;	// L4951
        int8_t v3174 = b67;	// L4952
        int16_t v3175 = v3173;	// L4953
        int16_t v3176 = v3174;	// L4954
        int16_t v3177 = v3175 * v3176;	// L4955
        int8_t v3178 = c51;	// L4956
        ap_int<17> v3179 = v3178;	// L4957
        ap_int<17> v3180 = v3177;	// L4958
        ap_int<17> v3181 = v3179 + v3180;	// L4959
        int8_t v3182 = v3181;	// L4960
        c51 = v3182;	// L4961
        int8_t v3183 = a55;	// L4962
        v3161.write(v3183);	// L4963
        int8_t v3184 = b67;	// L4964
        v3162.write(v3184);	// L4965
      }
      ap_uint<128> packed_tmp51;	// L4967
      ap_uint<128> v3186 = v3163.read();	// L4968
      packed_tmp51 = v3186;	// L4969
      ap_uint<128> packed_c51;	// L4970
      packed_c51 = 0;	// L4971
      l_S_m_1_m51: for (int m51 = 0; m51 < 16; m51++) {	// L4972
        ap_int<33> v3189 = m51;	// L4973
        bool v3190 = v3189 == 3;	// L4974
        if (v3190) {	// L4975
          int8_t v3191 = c51;	// L4976
          ap_int<128> v3192 = packed_c51;	// L4977
          int64_t v3193 = m51;	// L4978
          int64_t v3194 = v3193 * 8;	// L4979
          ap_int<34> v3195 = m51;	// L4980
          ap_int<34> v3196 = v3195 + 1;	// L4981
          ap_int<66> v3197 = v3196;	// L4982
          ap_int<66> v3198 = v3197 * 8;	// L4983
          ap_int<66> v3199 = v3198 - 1;	// L4984
          int v3200 = v3194;	// L4985
          int v3201 = v3199;	// L4986
          ap_int<128> v3202;
          ap_int<128> v3202_tmp = v3192;
          v3202_tmp(v3201, v3200) = v3191;
          v3202 = v3202_tmp;	// L4987
          packed_c51 = v3202;	// L4988
        } else {
          ap_int<128> v3203 = packed_tmp51;	// L4990
          int64_t v3204 = m51;	// L4991
          int64_t v3205 = v3204 * 8;	// L4992
          ap_int<34> v3206 = m51;	// L4993
          ap_int<34> v3207 = v3206 + 1;	// L4994
          ap_int<66> v3208 = v3207;	// L4995
          ap_int<66> v3209 = v3208 * 8;	// L4996
          ap_int<66> v3210 = v3209 - 1;	// L4997
          int v3211 = v3205;	// L4998
          int v3212 = v3210;	// L4999
          int8_t v3213;
          ap_int<128> v3213_tmp = v3203;
          v3213 = v3213_tmp(v3212, v3211);	// L5000
          ap_int<128> v3214 = packed_c51;	// L5001
          ap_int<128> v3215;
          ap_int<128> v3215_tmp = v3214;
          v3215_tmp(v3212, v3211) = v3213;
          v3215 = v3215_tmp;	// L5011
          packed_c51 = v3215;	// L5012
        }
      }
      ap_int<128> v3216 = packed_c51;	// L5015
      v3164.write(v3216);	// L5016
    }
  }
}

void gemm_4_5(
  hls::stream< int8_t >& v3217,
  hls::stream< int8_t >& v3218,
  hls::stream< int8_t >& v3219,
  hls::stream< int8_t >& v3220,
  hls::stream< ap_uint<128> >& v3221,
  hls::stream< ap_uint<128> >& v3222
) {	// L5021
  l_S_mt_nt_0_mt75: for (int mt75 = 0; mt75 < 4; mt75++) {	// L5029
    l_nt75: for (int nt75 = 0; nt75 < 4; nt75++) {	// L5030
      int8_t c52;	// L5031
      c52 = 0;	// L5032
      l_S_k_0_k75: for (int k75 = 0; k75 < 64; k75++) {	// L5033
        int8_t v3227 = v3217.read();	// L5034
        int8_t a56;	// L5035
        a56 = v3227;	// L5036
        int8_t v3229 = v3218.read();	// L5037
        int8_t b68;	// L5038
        b68 = v3229;	// L5039
        int8_t v3231 = a56;	// L5040
        int8_t v3232 = b68;	// L5041
        int16_t v3233 = v3231;	// L5042
        int16_t v3234 = v3232;	// L5043
        int16_t v3235 = v3233 * v3234;	// L5044
        int8_t v3236 = c52;	// L5045
        ap_int<17> v3237 = v3236;	// L5046
        ap_int<17> v3238 = v3235;	// L5047
        ap_int<17> v3239 = v3237 + v3238;	// L5048
        int8_t v3240 = v3239;	// L5049
        c52 = v3240;	// L5050
        int8_t v3241 = a56;	// L5051
        v3219.write(v3241);	// L5052
        int8_t v3242 = b68;	// L5053
        v3220.write(v3242);	// L5054
      }
      ap_uint<128> packed_tmp52;	// L5056
      ap_uint<128> v3244 = v3221.read();	// L5057
      packed_tmp52 = v3244;	// L5058
      ap_uint<128> packed_c52;	// L5059
      packed_c52 = 0;	// L5060
      l_S_m_1_m52: for (int m52 = 0; m52 < 16; m52++) {	// L5061
        ap_int<33> v3247 = m52;	// L5062
        bool v3248 = v3247 == 3;	// L5063
        if (v3248) {	// L5064
          int8_t v3249 = c52;	// L5065
          ap_int<128> v3250 = packed_c52;	// L5066
          int64_t v3251 = m52;	// L5067
          int64_t v3252 = v3251 * 8;	// L5068
          ap_int<34> v3253 = m52;	// L5069
          ap_int<34> v3254 = v3253 + 1;	// L5070
          ap_int<66> v3255 = v3254;	// L5071
          ap_int<66> v3256 = v3255 * 8;	// L5072
          ap_int<66> v3257 = v3256 - 1;	// L5073
          int v3258 = v3252;	// L5074
          int v3259 = v3257;	// L5075
          ap_int<128> v3260;
          ap_int<128> v3260_tmp = v3250;
          v3260_tmp(v3259, v3258) = v3249;
          v3260 = v3260_tmp;	// L5076
          packed_c52 = v3260;	// L5077
        } else {
          ap_int<128> v3261 = packed_tmp52;	// L5079
          int64_t v3262 = m52;	// L5080
          int64_t v3263 = v3262 * 8;	// L5081
          ap_int<34> v3264 = m52;	// L5082
          ap_int<34> v3265 = v3264 + 1;	// L5083
          ap_int<66> v3266 = v3265;	// L5084
          ap_int<66> v3267 = v3266 * 8;	// L5085
          ap_int<66> v3268 = v3267 - 1;	// L5086
          int v3269 = v3263;	// L5087
          int v3270 = v3268;	// L5088
          int8_t v3271;
          ap_int<128> v3271_tmp = v3261;
          v3271 = v3271_tmp(v3270, v3269);	// L5089
          ap_int<128> v3272 = packed_c52;	// L5090
          ap_int<128> v3273;
          ap_int<128> v3273_tmp = v3272;
          v3273_tmp(v3270, v3269) = v3271;
          v3273 = v3273_tmp;	// L5100
          packed_c52 = v3273;	// L5101
        }
      }
      ap_int<128> v3274 = packed_c52;	// L5104
      v3222.write(v3274);	// L5105
    }
  }
}

void gemm_4_6(
  hls::stream< int8_t >& v3275,
  hls::stream< int8_t >& v3276,
  hls::stream< int8_t >& v3277,
  hls::stream< int8_t >& v3278,
  hls::stream< ap_uint<128> >& v3279,
  hls::stream< ap_uint<128> >& v3280
) {	// L5110
  l_S_mt_nt_0_mt76: for (int mt76 = 0; mt76 < 4; mt76++) {	// L5118
    l_nt76: for (int nt76 = 0; nt76 < 4; nt76++) {	// L5119
      int8_t c53;	// L5120
      c53 = 0;	// L5121
      l_S_k_0_k76: for (int k76 = 0; k76 < 64; k76++) {	// L5122
        int8_t v3285 = v3275.read();	// L5123
        int8_t a57;	// L5124
        a57 = v3285;	// L5125
        int8_t v3287 = v3276.read();	// L5126
        int8_t b69;	// L5127
        b69 = v3287;	// L5128
        int8_t v3289 = a57;	// L5129
        int8_t v3290 = b69;	// L5130
        int16_t v3291 = v3289;	// L5131
        int16_t v3292 = v3290;	// L5132
        int16_t v3293 = v3291 * v3292;	// L5133
        int8_t v3294 = c53;	// L5134
        ap_int<17> v3295 = v3294;	// L5135
        ap_int<17> v3296 = v3293;	// L5136
        ap_int<17> v3297 = v3295 + v3296;	// L5137
        int8_t v3298 = v3297;	// L5138
        c53 = v3298;	// L5139
        int8_t v3299 = a57;	// L5140
        v3277.write(v3299);	// L5141
        int8_t v3300 = b69;	// L5142
        v3278.write(v3300);	// L5143
      }
      ap_uint<128> packed_tmp53;	// L5145
      ap_uint<128> v3302 = v3279.read();	// L5146
      packed_tmp53 = v3302;	// L5147
      ap_uint<128> packed_c53;	// L5148
      packed_c53 = 0;	// L5149
      l_S_m_1_m53: for (int m53 = 0; m53 < 16; m53++) {	// L5150
        ap_int<33> v3305 = m53;	// L5151
        bool v3306 = v3305 == 3;	// L5152
        if (v3306) {	// L5153
          int8_t v3307 = c53;	// L5154
          ap_int<128> v3308 = packed_c53;	// L5155
          int64_t v3309 = m53;	// L5156
          int64_t v3310 = v3309 * 8;	// L5157
          ap_int<34> v3311 = m53;	// L5158
          ap_int<34> v3312 = v3311 + 1;	// L5159
          ap_int<66> v3313 = v3312;	// L5160
          ap_int<66> v3314 = v3313 * 8;	// L5161
          ap_int<66> v3315 = v3314 - 1;	// L5162
          int v3316 = v3310;	// L5163
          int v3317 = v3315;	// L5164
          ap_int<128> v3318;
          ap_int<128> v3318_tmp = v3308;
          v3318_tmp(v3317, v3316) = v3307;
          v3318 = v3318_tmp;	// L5165
          packed_c53 = v3318;	// L5166
        } else {
          ap_int<128> v3319 = packed_tmp53;	// L5168
          int64_t v3320 = m53;	// L5169
          int64_t v3321 = v3320 * 8;	// L5170
          ap_int<34> v3322 = m53;	// L5171
          ap_int<34> v3323 = v3322 + 1;	// L5172
          ap_int<66> v3324 = v3323;	// L5173
          ap_int<66> v3325 = v3324 * 8;	// L5174
          ap_int<66> v3326 = v3325 - 1;	// L5175
          int v3327 = v3321;	// L5176
          int v3328 = v3326;	// L5177
          int8_t v3329;
          ap_int<128> v3329_tmp = v3319;
          v3329 = v3329_tmp(v3328, v3327);	// L5178
          ap_int<128> v3330 = packed_c53;	// L5179
          ap_int<128> v3331;
          ap_int<128> v3331_tmp = v3330;
          v3331_tmp(v3328, v3327) = v3329;
          v3331 = v3331_tmp;	// L5189
          packed_c53 = v3331;	// L5190
        }
      }
      ap_int<128> v3332 = packed_c53;	// L5193
      v3280.write(v3332);	// L5194
    }
  }
}

void gemm_4_7(
  hls::stream< int8_t >& v3333,
  hls::stream< int8_t >& v3334,
  hls::stream< int8_t >& v3335,
  hls::stream< int8_t >& v3336,
  hls::stream< ap_uint<128> >& v3337,
  hls::stream< ap_uint<128> >& v3338
) {	// L5199
  l_S_mt_nt_0_mt77: for (int mt77 = 0; mt77 < 4; mt77++) {	// L5207
    l_nt77: for (int nt77 = 0; nt77 < 4; nt77++) {	// L5208
      int8_t c54;	// L5209
      c54 = 0;	// L5210
      l_S_k_0_k77: for (int k77 = 0; k77 < 64; k77++) {	// L5211
        int8_t v3343 = v3333.read();	// L5212
        int8_t a58;	// L5213
        a58 = v3343;	// L5214
        int8_t v3345 = v3334.read();	// L5215
        int8_t b70;	// L5216
        b70 = v3345;	// L5217
        int8_t v3347 = a58;	// L5218
        int8_t v3348 = b70;	// L5219
        int16_t v3349 = v3347;	// L5220
        int16_t v3350 = v3348;	// L5221
        int16_t v3351 = v3349 * v3350;	// L5222
        int8_t v3352 = c54;	// L5223
        ap_int<17> v3353 = v3352;	// L5224
        ap_int<17> v3354 = v3351;	// L5225
        ap_int<17> v3355 = v3353 + v3354;	// L5226
        int8_t v3356 = v3355;	// L5227
        c54 = v3356;	// L5228
        int8_t v3357 = a58;	// L5229
        v3335.write(v3357);	// L5230
        int8_t v3358 = b70;	// L5231
        v3336.write(v3358);	// L5232
      }
      ap_uint<128> packed_tmp54;	// L5234
      ap_uint<128> v3360 = v3337.read();	// L5235
      packed_tmp54 = v3360;	// L5236
      ap_uint<128> packed_c54;	// L5237
      packed_c54 = 0;	// L5238
      l_S_m_1_m54: for (int m54 = 0; m54 < 16; m54++) {	// L5239
        ap_int<33> v3363 = m54;	// L5240
        bool v3364 = v3363 == 3;	// L5241
        if (v3364) {	// L5242
          int8_t v3365 = c54;	// L5243
          ap_int<128> v3366 = packed_c54;	// L5244
          int64_t v3367 = m54;	// L5245
          int64_t v3368 = v3367 * 8;	// L5246
          ap_int<34> v3369 = m54;	// L5247
          ap_int<34> v3370 = v3369 + 1;	// L5248
          ap_int<66> v3371 = v3370;	// L5249
          ap_int<66> v3372 = v3371 * 8;	// L5250
          ap_int<66> v3373 = v3372 - 1;	// L5251
          int v3374 = v3368;	// L5252
          int v3375 = v3373;	// L5253
          ap_int<128> v3376;
          ap_int<128> v3376_tmp = v3366;
          v3376_tmp(v3375, v3374) = v3365;
          v3376 = v3376_tmp;	// L5254
          packed_c54 = v3376;	// L5255
        } else {
          ap_int<128> v3377 = packed_tmp54;	// L5257
          int64_t v3378 = m54;	// L5258
          int64_t v3379 = v3378 * 8;	// L5259
          ap_int<34> v3380 = m54;	// L5260
          ap_int<34> v3381 = v3380 + 1;	// L5261
          ap_int<66> v3382 = v3381;	// L5262
          ap_int<66> v3383 = v3382 * 8;	// L5263
          ap_int<66> v3384 = v3383 - 1;	// L5264
          int v3385 = v3379;	// L5265
          int v3386 = v3384;	// L5266
          int8_t v3387;
          ap_int<128> v3387_tmp = v3377;
          v3387 = v3387_tmp(v3386, v3385);	// L5267
          ap_int<128> v3388 = packed_c54;	// L5268
          ap_int<128> v3389;
          ap_int<128> v3389_tmp = v3388;
          v3389_tmp(v3386, v3385) = v3387;
          v3389 = v3389_tmp;	// L5278
          packed_c54 = v3389;	// L5279
        }
      }
      ap_int<128> v3390 = packed_c54;	// L5282
      v3338.write(v3390);	// L5283
    }
  }
}

void gemm_4_8(
  hls::stream< int8_t >& v3391,
  hls::stream< int8_t >& v3392,
  hls::stream< int8_t >& v3393,
  hls::stream< int8_t >& v3394,
  hls::stream< ap_uint<128> >& v3395,
  hls::stream< ap_uint<128> >& v3396
) {	// L5288
  l_S_mt_nt_0_mt78: for (int mt78 = 0; mt78 < 4; mt78++) {	// L5296
    l_nt78: for (int nt78 = 0; nt78 < 4; nt78++) {	// L5297
      int8_t c55;	// L5298
      c55 = 0;	// L5299
      l_S_k_0_k78: for (int k78 = 0; k78 < 64; k78++) {	// L5300
        int8_t v3401 = v3391.read();	// L5301
        int8_t a59;	// L5302
        a59 = v3401;	// L5303
        int8_t v3403 = v3392.read();	// L5304
        int8_t b71;	// L5305
        b71 = v3403;	// L5306
        int8_t v3405 = a59;	// L5307
        int8_t v3406 = b71;	// L5308
        int16_t v3407 = v3405;	// L5309
        int16_t v3408 = v3406;	// L5310
        int16_t v3409 = v3407 * v3408;	// L5311
        int8_t v3410 = c55;	// L5312
        ap_int<17> v3411 = v3410;	// L5313
        ap_int<17> v3412 = v3409;	// L5314
        ap_int<17> v3413 = v3411 + v3412;	// L5315
        int8_t v3414 = v3413;	// L5316
        c55 = v3414;	// L5317
        int8_t v3415 = a59;	// L5318
        v3393.write(v3415);	// L5319
        int8_t v3416 = b71;	// L5320
        v3394.write(v3416);	// L5321
      }
      ap_uint<128> packed_tmp55;	// L5323
      ap_uint<128> v3418 = v3395.read();	// L5324
      packed_tmp55 = v3418;	// L5325
      ap_uint<128> packed_c55;	// L5326
      packed_c55 = 0;	// L5327
      l_S_m_1_m55: for (int m55 = 0; m55 < 16; m55++) {	// L5328
        ap_int<33> v3421 = m55;	// L5329
        bool v3422 = v3421 == 3;	// L5330
        if (v3422) {	// L5331
          int8_t v3423 = c55;	// L5332
          ap_int<128> v3424 = packed_c55;	// L5333
          int64_t v3425 = m55;	// L5334
          int64_t v3426 = v3425 * 8;	// L5335
          ap_int<34> v3427 = m55;	// L5336
          ap_int<34> v3428 = v3427 + 1;	// L5337
          ap_int<66> v3429 = v3428;	// L5338
          ap_int<66> v3430 = v3429 * 8;	// L5339
          ap_int<66> v3431 = v3430 - 1;	// L5340
          int v3432 = v3426;	// L5341
          int v3433 = v3431;	// L5342
          ap_int<128> v3434;
          ap_int<128> v3434_tmp = v3424;
          v3434_tmp(v3433, v3432) = v3423;
          v3434 = v3434_tmp;	// L5343
          packed_c55 = v3434;	// L5344
        } else {
          ap_int<128> v3435 = packed_tmp55;	// L5346
          int64_t v3436 = m55;	// L5347
          int64_t v3437 = v3436 * 8;	// L5348
          ap_int<34> v3438 = m55;	// L5349
          ap_int<34> v3439 = v3438 + 1;	// L5350
          ap_int<66> v3440 = v3439;	// L5351
          ap_int<66> v3441 = v3440 * 8;	// L5352
          ap_int<66> v3442 = v3441 - 1;	// L5353
          int v3443 = v3437;	// L5354
          int v3444 = v3442;	// L5355
          int8_t v3445;
          ap_int<128> v3445_tmp = v3435;
          v3445 = v3445_tmp(v3444, v3443);	// L5356
          ap_int<128> v3446 = packed_c55;	// L5357
          ap_int<128> v3447;
          ap_int<128> v3447_tmp = v3446;
          v3447_tmp(v3444, v3443) = v3445;
          v3447 = v3447_tmp;	// L5367
          packed_c55 = v3447;	// L5368
        }
      }
      ap_int<128> v3448 = packed_c55;	// L5371
      v3396.write(v3448);	// L5372
    }
  }
}

void gemm_4_9(
  hls::stream< int8_t >& v3449,
  hls::stream< int8_t >& v3450,
  hls::stream< int8_t >& v3451,
  hls::stream< int8_t >& v3452,
  hls::stream< ap_uint<128> >& v3453,
  hls::stream< ap_uint<128> >& v3454
) {	// L5377
  l_S_mt_nt_0_mt79: for (int mt79 = 0; mt79 < 4; mt79++) {	// L5385
    l_nt79: for (int nt79 = 0; nt79 < 4; nt79++) {	// L5386
      int8_t c56;	// L5387
      c56 = 0;	// L5388
      l_S_k_0_k79: for (int k79 = 0; k79 < 64; k79++) {	// L5389
        int8_t v3459 = v3449.read();	// L5390
        int8_t a60;	// L5391
        a60 = v3459;	// L5392
        int8_t v3461 = v3450.read();	// L5393
        int8_t b72;	// L5394
        b72 = v3461;	// L5395
        int8_t v3463 = a60;	// L5396
        int8_t v3464 = b72;	// L5397
        int16_t v3465 = v3463;	// L5398
        int16_t v3466 = v3464;	// L5399
        int16_t v3467 = v3465 * v3466;	// L5400
        int8_t v3468 = c56;	// L5401
        ap_int<17> v3469 = v3468;	// L5402
        ap_int<17> v3470 = v3467;	// L5403
        ap_int<17> v3471 = v3469 + v3470;	// L5404
        int8_t v3472 = v3471;	// L5405
        c56 = v3472;	// L5406
        int8_t v3473 = a60;	// L5407
        v3451.write(v3473);	// L5408
        int8_t v3474 = b72;	// L5409
        v3452.write(v3474);	// L5410
      }
      ap_uint<128> packed_tmp56;	// L5412
      ap_uint<128> v3476 = v3453.read();	// L5413
      packed_tmp56 = v3476;	// L5414
      ap_uint<128> packed_c56;	// L5415
      packed_c56 = 0;	// L5416
      l_S_m_1_m56: for (int m56 = 0; m56 < 16; m56++) {	// L5417
        ap_int<33> v3479 = m56;	// L5418
        bool v3480 = v3479 == 3;	// L5419
        if (v3480) {	// L5420
          int8_t v3481 = c56;	// L5421
          ap_int<128> v3482 = packed_c56;	// L5422
          int64_t v3483 = m56;	// L5423
          int64_t v3484 = v3483 * 8;	// L5424
          ap_int<34> v3485 = m56;	// L5425
          ap_int<34> v3486 = v3485 + 1;	// L5426
          ap_int<66> v3487 = v3486;	// L5427
          ap_int<66> v3488 = v3487 * 8;	// L5428
          ap_int<66> v3489 = v3488 - 1;	// L5429
          int v3490 = v3484;	// L5430
          int v3491 = v3489;	// L5431
          ap_int<128> v3492;
          ap_int<128> v3492_tmp = v3482;
          v3492_tmp(v3491, v3490) = v3481;
          v3492 = v3492_tmp;	// L5432
          packed_c56 = v3492;	// L5433
        } else {
          ap_int<128> v3493 = packed_tmp56;	// L5435
          int64_t v3494 = m56;	// L5436
          int64_t v3495 = v3494 * 8;	// L5437
          ap_int<34> v3496 = m56;	// L5438
          ap_int<34> v3497 = v3496 + 1;	// L5439
          ap_int<66> v3498 = v3497;	// L5440
          ap_int<66> v3499 = v3498 * 8;	// L5441
          ap_int<66> v3500 = v3499 - 1;	// L5442
          int v3501 = v3495;	// L5443
          int v3502 = v3500;	// L5444
          int8_t v3503;
          ap_int<128> v3503_tmp = v3493;
          v3503 = v3503_tmp(v3502, v3501);	// L5445
          ap_int<128> v3504 = packed_c56;	// L5446
          ap_int<128> v3505;
          ap_int<128> v3505_tmp = v3504;
          v3505_tmp(v3502, v3501) = v3503;
          v3505 = v3505_tmp;	// L5456
          packed_c56 = v3505;	// L5457
        }
      }
      ap_int<128> v3506 = packed_c56;	// L5460
      v3454.write(v3506);	// L5461
    }
  }
}

void gemm_4_10(
  hls::stream< int8_t >& v3507,
  hls::stream< int8_t >& v3508,
  hls::stream< int8_t >& v3509,
  hls::stream< int8_t >& v3510,
  hls::stream< ap_uint<128> >& v3511,
  hls::stream< ap_uint<128> >& v3512
) {	// L5466
  l_S_mt_nt_0_mt80: for (int mt80 = 0; mt80 < 4; mt80++) {	// L5474
    l_nt80: for (int nt80 = 0; nt80 < 4; nt80++) {	// L5475
      int8_t c57;	// L5476
      c57 = 0;	// L5477
      l_S_k_0_k80: for (int k80 = 0; k80 < 64; k80++) {	// L5478
        int8_t v3517 = v3507.read();	// L5479
        int8_t a61;	// L5480
        a61 = v3517;	// L5481
        int8_t v3519 = v3508.read();	// L5482
        int8_t b73;	// L5483
        b73 = v3519;	// L5484
        int8_t v3521 = a61;	// L5485
        int8_t v3522 = b73;	// L5486
        int16_t v3523 = v3521;	// L5487
        int16_t v3524 = v3522;	// L5488
        int16_t v3525 = v3523 * v3524;	// L5489
        int8_t v3526 = c57;	// L5490
        ap_int<17> v3527 = v3526;	// L5491
        ap_int<17> v3528 = v3525;	// L5492
        ap_int<17> v3529 = v3527 + v3528;	// L5493
        int8_t v3530 = v3529;	// L5494
        c57 = v3530;	// L5495
        int8_t v3531 = a61;	// L5496
        v3509.write(v3531);	// L5497
        int8_t v3532 = b73;	// L5498
        v3510.write(v3532);	// L5499
      }
      ap_uint<128> packed_tmp57;	// L5501
      ap_uint<128> v3534 = v3511.read();	// L5502
      packed_tmp57 = v3534;	// L5503
      ap_uint<128> packed_c57;	// L5504
      packed_c57 = 0;	// L5505
      l_S_m_1_m57: for (int m57 = 0; m57 < 16; m57++) {	// L5506
        ap_int<33> v3537 = m57;	// L5507
        bool v3538 = v3537 == 3;	// L5508
        if (v3538) {	// L5509
          int8_t v3539 = c57;	// L5510
          ap_int<128> v3540 = packed_c57;	// L5511
          int64_t v3541 = m57;	// L5512
          int64_t v3542 = v3541 * 8;	// L5513
          ap_int<34> v3543 = m57;	// L5514
          ap_int<34> v3544 = v3543 + 1;	// L5515
          ap_int<66> v3545 = v3544;	// L5516
          ap_int<66> v3546 = v3545 * 8;	// L5517
          ap_int<66> v3547 = v3546 - 1;	// L5518
          int v3548 = v3542;	// L5519
          int v3549 = v3547;	// L5520
          ap_int<128> v3550;
          ap_int<128> v3550_tmp = v3540;
          v3550_tmp(v3549, v3548) = v3539;
          v3550 = v3550_tmp;	// L5521
          packed_c57 = v3550;	// L5522
        } else {
          ap_int<128> v3551 = packed_tmp57;	// L5524
          int64_t v3552 = m57;	// L5525
          int64_t v3553 = v3552 * 8;	// L5526
          ap_int<34> v3554 = m57;	// L5527
          ap_int<34> v3555 = v3554 + 1;	// L5528
          ap_int<66> v3556 = v3555;	// L5529
          ap_int<66> v3557 = v3556 * 8;	// L5530
          ap_int<66> v3558 = v3557 - 1;	// L5531
          int v3559 = v3553;	// L5532
          int v3560 = v3558;	// L5533
          int8_t v3561;
          ap_int<128> v3561_tmp = v3551;
          v3561 = v3561_tmp(v3560, v3559);	// L5534
          ap_int<128> v3562 = packed_c57;	// L5535
          ap_int<128> v3563;
          ap_int<128> v3563_tmp = v3562;
          v3563_tmp(v3560, v3559) = v3561;
          v3563 = v3563_tmp;	// L5545
          packed_c57 = v3563;	// L5546
        }
      }
      ap_int<128> v3564 = packed_c57;	// L5549
      v3512.write(v3564);	// L5550
    }
  }
}

void gemm_4_11(
  hls::stream< int8_t >& v3565,
  hls::stream< int8_t >& v3566,
  hls::stream< int8_t >& v3567,
  hls::stream< int8_t >& v3568,
  hls::stream< ap_uint<128> >& v3569,
  hls::stream< ap_uint<128> >& v3570
) {	// L5555
  l_S_mt_nt_0_mt81: for (int mt81 = 0; mt81 < 4; mt81++) {	// L5563
    l_nt81: for (int nt81 = 0; nt81 < 4; nt81++) {	// L5564
      int8_t c58;	// L5565
      c58 = 0;	// L5566
      l_S_k_0_k81: for (int k81 = 0; k81 < 64; k81++) {	// L5567
        int8_t v3575 = v3565.read();	// L5568
        int8_t a62;	// L5569
        a62 = v3575;	// L5570
        int8_t v3577 = v3566.read();	// L5571
        int8_t b74;	// L5572
        b74 = v3577;	// L5573
        int8_t v3579 = a62;	// L5574
        int8_t v3580 = b74;	// L5575
        int16_t v3581 = v3579;	// L5576
        int16_t v3582 = v3580;	// L5577
        int16_t v3583 = v3581 * v3582;	// L5578
        int8_t v3584 = c58;	// L5579
        ap_int<17> v3585 = v3584;	// L5580
        ap_int<17> v3586 = v3583;	// L5581
        ap_int<17> v3587 = v3585 + v3586;	// L5582
        int8_t v3588 = v3587;	// L5583
        c58 = v3588;	// L5584
        int8_t v3589 = a62;	// L5585
        v3567.write(v3589);	// L5586
        int8_t v3590 = b74;	// L5587
        v3568.write(v3590);	// L5588
      }
      ap_uint<128> packed_tmp58;	// L5590
      ap_uint<128> v3592 = v3569.read();	// L5591
      packed_tmp58 = v3592;	// L5592
      ap_uint<128> packed_c58;	// L5593
      packed_c58 = 0;	// L5594
      l_S_m_1_m58: for (int m58 = 0; m58 < 16; m58++) {	// L5595
        ap_int<33> v3595 = m58;	// L5596
        bool v3596 = v3595 == 3;	// L5597
        if (v3596) {	// L5598
          int8_t v3597 = c58;	// L5599
          ap_int<128> v3598 = packed_c58;	// L5600
          int64_t v3599 = m58;	// L5601
          int64_t v3600 = v3599 * 8;	// L5602
          ap_int<34> v3601 = m58;	// L5603
          ap_int<34> v3602 = v3601 + 1;	// L5604
          ap_int<66> v3603 = v3602;	// L5605
          ap_int<66> v3604 = v3603 * 8;	// L5606
          ap_int<66> v3605 = v3604 - 1;	// L5607
          int v3606 = v3600;	// L5608
          int v3607 = v3605;	// L5609
          ap_int<128> v3608;
          ap_int<128> v3608_tmp = v3598;
          v3608_tmp(v3607, v3606) = v3597;
          v3608 = v3608_tmp;	// L5610
          packed_c58 = v3608;	// L5611
        } else {
          ap_int<128> v3609 = packed_tmp58;	// L5613
          int64_t v3610 = m58;	// L5614
          int64_t v3611 = v3610 * 8;	// L5615
          ap_int<34> v3612 = m58;	// L5616
          ap_int<34> v3613 = v3612 + 1;	// L5617
          ap_int<66> v3614 = v3613;	// L5618
          ap_int<66> v3615 = v3614 * 8;	// L5619
          ap_int<66> v3616 = v3615 - 1;	// L5620
          int v3617 = v3611;	// L5621
          int v3618 = v3616;	// L5622
          int8_t v3619;
          ap_int<128> v3619_tmp = v3609;
          v3619 = v3619_tmp(v3618, v3617);	// L5623
          ap_int<128> v3620 = packed_c58;	// L5624
          ap_int<128> v3621;
          ap_int<128> v3621_tmp = v3620;
          v3621_tmp(v3618, v3617) = v3619;
          v3621 = v3621_tmp;	// L5634
          packed_c58 = v3621;	// L5635
        }
      }
      ap_int<128> v3622 = packed_c58;	// L5638
      v3570.write(v3622);	// L5639
    }
  }
}

void gemm_4_12(
  hls::stream< int8_t >& v3623,
  hls::stream< int8_t >& v3624,
  hls::stream< int8_t >& v3625,
  hls::stream< int8_t >& v3626,
  hls::stream< ap_uint<128> >& v3627,
  hls::stream< ap_uint<128> >& v3628
) {	// L5644
  l_S_mt_nt_0_mt82: for (int mt82 = 0; mt82 < 4; mt82++) {	// L5652
    l_nt82: for (int nt82 = 0; nt82 < 4; nt82++) {	// L5653
      int8_t c59;	// L5654
      c59 = 0;	// L5655
      l_S_k_0_k82: for (int k82 = 0; k82 < 64; k82++) {	// L5656
        int8_t v3633 = v3623.read();	// L5657
        int8_t a63;	// L5658
        a63 = v3633;	// L5659
        int8_t v3635 = v3624.read();	// L5660
        int8_t b75;	// L5661
        b75 = v3635;	// L5662
        int8_t v3637 = a63;	// L5663
        int8_t v3638 = b75;	// L5664
        int16_t v3639 = v3637;	// L5665
        int16_t v3640 = v3638;	// L5666
        int16_t v3641 = v3639 * v3640;	// L5667
        int8_t v3642 = c59;	// L5668
        ap_int<17> v3643 = v3642;	// L5669
        ap_int<17> v3644 = v3641;	// L5670
        ap_int<17> v3645 = v3643 + v3644;	// L5671
        int8_t v3646 = v3645;	// L5672
        c59 = v3646;	// L5673
        int8_t v3647 = a63;	// L5674
        v3625.write(v3647);	// L5675
        int8_t v3648 = b75;	// L5676
        v3626.write(v3648);	// L5677
      }
      ap_uint<128> packed_tmp59;	// L5679
      ap_uint<128> v3650 = v3627.read();	// L5680
      packed_tmp59 = v3650;	// L5681
      ap_uint<128> packed_c59;	// L5682
      packed_c59 = 0;	// L5683
      l_S_m_1_m59: for (int m59 = 0; m59 < 16; m59++) {	// L5684
        ap_int<33> v3653 = m59;	// L5685
        bool v3654 = v3653 == 3;	// L5686
        if (v3654) {	// L5687
          int8_t v3655 = c59;	// L5688
          ap_int<128> v3656 = packed_c59;	// L5689
          int64_t v3657 = m59;	// L5690
          int64_t v3658 = v3657 * 8;	// L5691
          ap_int<34> v3659 = m59;	// L5692
          ap_int<34> v3660 = v3659 + 1;	// L5693
          ap_int<66> v3661 = v3660;	// L5694
          ap_int<66> v3662 = v3661 * 8;	// L5695
          ap_int<66> v3663 = v3662 - 1;	// L5696
          int v3664 = v3658;	// L5697
          int v3665 = v3663;	// L5698
          ap_int<128> v3666;
          ap_int<128> v3666_tmp = v3656;
          v3666_tmp(v3665, v3664) = v3655;
          v3666 = v3666_tmp;	// L5699
          packed_c59 = v3666;	// L5700
        } else {
          ap_int<128> v3667 = packed_tmp59;	// L5702
          int64_t v3668 = m59;	// L5703
          int64_t v3669 = v3668 * 8;	// L5704
          ap_int<34> v3670 = m59;	// L5705
          ap_int<34> v3671 = v3670 + 1;	// L5706
          ap_int<66> v3672 = v3671;	// L5707
          ap_int<66> v3673 = v3672 * 8;	// L5708
          ap_int<66> v3674 = v3673 - 1;	// L5709
          int v3675 = v3669;	// L5710
          int v3676 = v3674;	// L5711
          int8_t v3677;
          ap_int<128> v3677_tmp = v3667;
          v3677 = v3677_tmp(v3676, v3675);	// L5712
          ap_int<128> v3678 = packed_c59;	// L5713
          ap_int<128> v3679;
          ap_int<128> v3679_tmp = v3678;
          v3679_tmp(v3676, v3675) = v3677;
          v3679 = v3679_tmp;	// L5723
          packed_c59 = v3679;	// L5724
        }
      }
      ap_int<128> v3680 = packed_c59;	// L5727
      v3628.write(v3680);	// L5728
    }
  }
}

void gemm_4_13(
  hls::stream< int8_t >& v3681,
  hls::stream< int8_t >& v3682,
  hls::stream< int8_t >& v3683,
  hls::stream< int8_t >& v3684,
  hls::stream< ap_uint<128> >& v3685,
  hls::stream< ap_uint<128> >& v3686
) {	// L5733
  l_S_mt_nt_0_mt83: for (int mt83 = 0; mt83 < 4; mt83++) {	// L5741
    l_nt83: for (int nt83 = 0; nt83 < 4; nt83++) {	// L5742
      int8_t c60;	// L5743
      c60 = 0;	// L5744
      l_S_k_0_k83: for (int k83 = 0; k83 < 64; k83++) {	// L5745
        int8_t v3691 = v3681.read();	// L5746
        int8_t a64;	// L5747
        a64 = v3691;	// L5748
        int8_t v3693 = v3682.read();	// L5749
        int8_t b76;	// L5750
        b76 = v3693;	// L5751
        int8_t v3695 = a64;	// L5752
        int8_t v3696 = b76;	// L5753
        int16_t v3697 = v3695;	// L5754
        int16_t v3698 = v3696;	// L5755
        int16_t v3699 = v3697 * v3698;	// L5756
        int8_t v3700 = c60;	// L5757
        ap_int<17> v3701 = v3700;	// L5758
        ap_int<17> v3702 = v3699;	// L5759
        ap_int<17> v3703 = v3701 + v3702;	// L5760
        int8_t v3704 = v3703;	// L5761
        c60 = v3704;	// L5762
        int8_t v3705 = a64;	// L5763
        v3683.write(v3705);	// L5764
        int8_t v3706 = b76;	// L5765
        v3684.write(v3706);	// L5766
      }
      ap_uint<128> packed_tmp60;	// L5768
      ap_uint<128> v3708 = v3685.read();	// L5769
      packed_tmp60 = v3708;	// L5770
      ap_uint<128> packed_c60;	// L5771
      packed_c60 = 0;	// L5772
      l_S_m_1_m60: for (int m60 = 0; m60 < 16; m60++) {	// L5773
        ap_int<33> v3711 = m60;	// L5774
        bool v3712 = v3711 == 3;	// L5775
        if (v3712) {	// L5776
          int8_t v3713 = c60;	// L5777
          ap_int<128> v3714 = packed_c60;	// L5778
          int64_t v3715 = m60;	// L5779
          int64_t v3716 = v3715 * 8;	// L5780
          ap_int<34> v3717 = m60;	// L5781
          ap_int<34> v3718 = v3717 + 1;	// L5782
          ap_int<66> v3719 = v3718;	// L5783
          ap_int<66> v3720 = v3719 * 8;	// L5784
          ap_int<66> v3721 = v3720 - 1;	// L5785
          int v3722 = v3716;	// L5786
          int v3723 = v3721;	// L5787
          ap_int<128> v3724;
          ap_int<128> v3724_tmp = v3714;
          v3724_tmp(v3723, v3722) = v3713;
          v3724 = v3724_tmp;	// L5788
          packed_c60 = v3724;	// L5789
        } else {
          ap_int<128> v3725 = packed_tmp60;	// L5791
          int64_t v3726 = m60;	// L5792
          int64_t v3727 = v3726 * 8;	// L5793
          ap_int<34> v3728 = m60;	// L5794
          ap_int<34> v3729 = v3728 + 1;	// L5795
          ap_int<66> v3730 = v3729;	// L5796
          ap_int<66> v3731 = v3730 * 8;	// L5797
          ap_int<66> v3732 = v3731 - 1;	// L5798
          int v3733 = v3727;	// L5799
          int v3734 = v3732;	// L5800
          int8_t v3735;
          ap_int<128> v3735_tmp = v3725;
          v3735 = v3735_tmp(v3734, v3733);	// L5801
          ap_int<128> v3736 = packed_c60;	// L5802
          ap_int<128> v3737;
          ap_int<128> v3737_tmp = v3736;
          v3737_tmp(v3734, v3733) = v3735;
          v3737 = v3737_tmp;	// L5812
          packed_c60 = v3737;	// L5813
        }
      }
      ap_int<128> v3738 = packed_c60;	// L5816
      v3686.write(v3738);	// L5817
    }
  }
}

void gemm_4_14(
  hls::stream< int8_t >& v3739,
  hls::stream< int8_t >& v3740,
  hls::stream< int8_t >& v3741,
  hls::stream< int8_t >& v3742,
  hls::stream< ap_uint<128> >& v3743,
  hls::stream< ap_uint<128> >& v3744
) {	// L5822
  l_S_mt_nt_0_mt84: for (int mt84 = 0; mt84 < 4; mt84++) {	// L5830
    l_nt84: for (int nt84 = 0; nt84 < 4; nt84++) {	// L5831
      int8_t c61;	// L5832
      c61 = 0;	// L5833
      l_S_k_0_k84: for (int k84 = 0; k84 < 64; k84++) {	// L5834
        int8_t v3749 = v3739.read();	// L5835
        int8_t a65;	// L5836
        a65 = v3749;	// L5837
        int8_t v3751 = v3740.read();	// L5838
        int8_t b77;	// L5839
        b77 = v3751;	// L5840
        int8_t v3753 = a65;	// L5841
        int8_t v3754 = b77;	// L5842
        int16_t v3755 = v3753;	// L5843
        int16_t v3756 = v3754;	// L5844
        int16_t v3757 = v3755 * v3756;	// L5845
        int8_t v3758 = c61;	// L5846
        ap_int<17> v3759 = v3758;	// L5847
        ap_int<17> v3760 = v3757;	// L5848
        ap_int<17> v3761 = v3759 + v3760;	// L5849
        int8_t v3762 = v3761;	// L5850
        c61 = v3762;	// L5851
        int8_t v3763 = a65;	// L5852
        v3741.write(v3763);	// L5853
        int8_t v3764 = b77;	// L5854
        v3742.write(v3764);	// L5855
      }
      ap_uint<128> packed_tmp61;	// L5857
      ap_uint<128> v3766 = v3743.read();	// L5858
      packed_tmp61 = v3766;	// L5859
      ap_uint<128> packed_c61;	// L5860
      packed_c61 = 0;	// L5861
      l_S_m_1_m61: for (int m61 = 0; m61 < 16; m61++) {	// L5862
        ap_int<33> v3769 = m61;	// L5863
        bool v3770 = v3769 == 3;	// L5864
        if (v3770) {	// L5865
          int8_t v3771 = c61;	// L5866
          ap_int<128> v3772 = packed_c61;	// L5867
          int64_t v3773 = m61;	// L5868
          int64_t v3774 = v3773 * 8;	// L5869
          ap_int<34> v3775 = m61;	// L5870
          ap_int<34> v3776 = v3775 + 1;	// L5871
          ap_int<66> v3777 = v3776;	// L5872
          ap_int<66> v3778 = v3777 * 8;	// L5873
          ap_int<66> v3779 = v3778 - 1;	// L5874
          int v3780 = v3774;	// L5875
          int v3781 = v3779;	// L5876
          ap_int<128> v3782;
          ap_int<128> v3782_tmp = v3772;
          v3782_tmp(v3781, v3780) = v3771;
          v3782 = v3782_tmp;	// L5877
          packed_c61 = v3782;	// L5878
        } else {
          ap_int<128> v3783 = packed_tmp61;	// L5880
          int64_t v3784 = m61;	// L5881
          int64_t v3785 = v3784 * 8;	// L5882
          ap_int<34> v3786 = m61;	// L5883
          ap_int<34> v3787 = v3786 + 1;	// L5884
          ap_int<66> v3788 = v3787;	// L5885
          ap_int<66> v3789 = v3788 * 8;	// L5886
          ap_int<66> v3790 = v3789 - 1;	// L5887
          int v3791 = v3785;	// L5888
          int v3792 = v3790;	// L5889
          int8_t v3793;
          ap_int<128> v3793_tmp = v3783;
          v3793 = v3793_tmp(v3792, v3791);	// L5890
          ap_int<128> v3794 = packed_c61;	// L5891
          ap_int<128> v3795;
          ap_int<128> v3795_tmp = v3794;
          v3795_tmp(v3792, v3791) = v3793;
          v3795 = v3795_tmp;	// L5901
          packed_c61 = v3795;	// L5902
        }
      }
      ap_int<128> v3796 = packed_c61;	// L5905
      v3744.write(v3796);	// L5906
    }
  }
}

void gemm_4_15(
  hls::stream< int8_t >& v3797,
  hls::stream< int8_t >& v3798,
  hls::stream< int8_t >& v3799,
  hls::stream< int8_t >& v3800,
  hls::stream< ap_uint<128> >& v3801,
  hls::stream< ap_uint<128> >& v3802
) {	// L5911
  l_S_mt_nt_0_mt85: for (int mt85 = 0; mt85 < 4; mt85++) {	// L5919
    l_nt85: for (int nt85 = 0; nt85 < 4; nt85++) {	// L5920
      int8_t c62;	// L5921
      c62 = 0;	// L5922
      l_S_k_0_k85: for (int k85 = 0; k85 < 64; k85++) {	// L5923
        int8_t v3807 = v3797.read();	// L5924
        int8_t a66;	// L5925
        a66 = v3807;	// L5926
        int8_t v3809 = v3798.read();	// L5927
        int8_t b78;	// L5928
        b78 = v3809;	// L5929
        int8_t v3811 = a66;	// L5930
        int8_t v3812 = b78;	// L5931
        int16_t v3813 = v3811;	// L5932
        int16_t v3814 = v3812;	// L5933
        int16_t v3815 = v3813 * v3814;	// L5934
        int8_t v3816 = c62;	// L5935
        ap_int<17> v3817 = v3816;	// L5936
        ap_int<17> v3818 = v3815;	// L5937
        ap_int<17> v3819 = v3817 + v3818;	// L5938
        int8_t v3820 = v3819;	// L5939
        c62 = v3820;	// L5940
        int8_t v3821 = a66;	// L5941
        v3799.write(v3821);	// L5942
        int8_t v3822 = b78;	// L5943
        v3800.write(v3822);	// L5944
      }
      ap_uint<128> packed_tmp62;	// L5946
      ap_uint<128> v3824 = v3801.read();	// L5947
      packed_tmp62 = v3824;	// L5948
      ap_uint<128> packed_c62;	// L5949
      packed_c62 = 0;	// L5950
      l_S_m_1_m62: for (int m62 = 0; m62 < 16; m62++) {	// L5951
        ap_int<33> v3827 = m62;	// L5952
        bool v3828 = v3827 == 3;	// L5953
        if (v3828) {	// L5954
          int8_t v3829 = c62;	// L5955
          ap_int<128> v3830 = packed_c62;	// L5956
          int64_t v3831 = m62;	// L5957
          int64_t v3832 = v3831 * 8;	// L5958
          ap_int<34> v3833 = m62;	// L5959
          ap_int<34> v3834 = v3833 + 1;	// L5960
          ap_int<66> v3835 = v3834;	// L5961
          ap_int<66> v3836 = v3835 * 8;	// L5962
          ap_int<66> v3837 = v3836 - 1;	// L5963
          int v3838 = v3832;	// L5964
          int v3839 = v3837;	// L5965
          ap_int<128> v3840;
          ap_int<128> v3840_tmp = v3830;
          v3840_tmp(v3839, v3838) = v3829;
          v3840 = v3840_tmp;	// L5966
          packed_c62 = v3840;	// L5967
        } else {
          ap_int<128> v3841 = packed_tmp62;	// L5969
          int64_t v3842 = m62;	// L5970
          int64_t v3843 = v3842 * 8;	// L5971
          ap_int<34> v3844 = m62;	// L5972
          ap_int<34> v3845 = v3844 + 1;	// L5973
          ap_int<66> v3846 = v3845;	// L5974
          ap_int<66> v3847 = v3846 * 8;	// L5975
          ap_int<66> v3848 = v3847 - 1;	// L5976
          int v3849 = v3843;	// L5977
          int v3850 = v3848;	// L5978
          int8_t v3851;
          ap_int<128> v3851_tmp = v3841;
          v3851 = v3851_tmp(v3850, v3849);	// L5979
          ap_int<128> v3852 = packed_c62;	// L5980
          ap_int<128> v3853;
          ap_int<128> v3853_tmp = v3852;
          v3853_tmp(v3850, v3849) = v3851;
          v3853 = v3853_tmp;	// L5990
          packed_c62 = v3853;	// L5991
        }
      }
      ap_int<128> v3854 = packed_c62;	// L5994
      v3802.write(v3854);	// L5995
    }
  }
}

void gemm_4_16(
  hls::stream< int8_t >& v3855,
  hls::stream< int8_t >& v3856,
  hls::stream< int8_t >& v3857,
  hls::stream< ap_uint<128> >& v3858,
  hls::stream< ap_uint<128> >& v3859
) {	// L6000
  l_S_mt_nt_0_mt86: for (int mt86 = 0; mt86 < 4; mt86++) {	// L6008
    l_nt86: for (int nt86 = 0; nt86 < 4; nt86++) {	// L6009
      int8_t c63;	// L6010
      c63 = 0;	// L6011
      l_S_k_0_k86: for (int k86 = 0; k86 < 64; k86++) {	// L6012
        int8_t v3864 = v3855.read();	// L6013
        int8_t a67;	// L6014
        a67 = v3864;	// L6015
        int8_t v3866 = v3856.read();	// L6016
        int8_t b79;	// L6017
        b79 = v3866;	// L6018
        int8_t v3868 = a67;	// L6019
        int8_t v3869 = b79;	// L6020
        int16_t v3870 = v3868;	// L6021
        int16_t v3871 = v3869;	// L6022
        int16_t v3872 = v3870 * v3871;	// L6023
        int8_t v3873 = c63;	// L6024
        ap_int<17> v3874 = v3873;	// L6025
        ap_int<17> v3875 = v3872;	// L6026
        ap_int<17> v3876 = v3874 + v3875;	// L6027
        int8_t v3877 = v3876;	// L6028
        c63 = v3877;	// L6029
        int8_t v3878 = b79;	// L6030
        v3857.write(v3878);	// L6031
      }
      ap_uint<128> packed_tmp63;	// L6033
      ap_uint<128> v3880 = v3858.read();	// L6034
      packed_tmp63 = v3880;	// L6035
      ap_uint<128> packed_c63;	// L6036
      packed_c63 = 0;	// L6037
      l_S_m_1_m63: for (int m63 = 0; m63 < 16; m63++) {	// L6038
        ap_int<33> v3883 = m63;	// L6039
        bool v3884 = v3883 == 3;	// L6040
        if (v3884) {	// L6041
          int8_t v3885 = c63;	// L6042
          ap_int<128> v3886 = packed_c63;	// L6043
          int64_t v3887 = m63;	// L6044
          int64_t v3888 = v3887 * 8;	// L6045
          ap_int<34> v3889 = m63;	// L6046
          ap_int<34> v3890 = v3889 + 1;	// L6047
          ap_int<66> v3891 = v3890;	// L6048
          ap_int<66> v3892 = v3891 * 8;	// L6049
          ap_int<66> v3893 = v3892 - 1;	// L6050
          int v3894 = v3888;	// L6051
          int v3895 = v3893;	// L6052
          ap_int<128> v3896;
          ap_int<128> v3896_tmp = v3886;
          v3896_tmp(v3895, v3894) = v3885;
          v3896 = v3896_tmp;	// L6053
          packed_c63 = v3896;	// L6054
        } else {
          ap_int<128> v3897 = packed_tmp63;	// L6056
          int64_t v3898 = m63;	// L6057
          int64_t v3899 = v3898 * 8;	// L6058
          ap_int<34> v3900 = m63;	// L6059
          ap_int<34> v3901 = v3900 + 1;	// L6060
          ap_int<66> v3902 = v3901;	// L6061
          ap_int<66> v3903 = v3902 * 8;	// L6062
          ap_int<66> v3904 = v3903 - 1;	// L6063
          int v3905 = v3899;	// L6064
          int v3906 = v3904;	// L6065
          int8_t v3907;
          ap_int<128> v3907_tmp = v3897;
          v3907 = v3907_tmp(v3906, v3905);	// L6066
          ap_int<128> v3908 = packed_c63;	// L6067
          ap_int<128> v3909;
          ap_int<128> v3909_tmp = v3908;
          v3909_tmp(v3906, v3905) = v3907;
          v3909 = v3909_tmp;	// L6077
          packed_c63 = v3909;	// L6078
        }
      }
      ap_int<128> v3910 = packed_c63;	// L6081
      v3859.write(v3910);	// L6082
    }
  }
}

void gemm_5_0(
  hls::stream< ap_uint<128> >& v3911,
  hls::stream< int8_t >& v3912,
  hls::stream< ap_uint<128> >& v3913
) {	// L6087
  l_S_mt_nt_0_mt87: for (int mt87 = 0; mt87 < 4; mt87++) {	// L6090
    l_nt87: for (int nt87 = 0; nt87 < 4; nt87++) {	// L6091
      l_S_k_0_k87: for (int k87 = 0; k87 < 64; k87++) {	// L6092
        ap_uint<128> v3917 = v3911.read();	// L6093
        ap_uint<128> a68;	// L6094
        a68 = v3917;	// L6095
        ap_int<128> v3919 = a68;	// L6096
        int8_t v3920;
        ap_int<128> v3920_tmp = v3919;
        v3920 = v3920_tmp(39, 32);	// L6097
        v3912.write(v3920);	// L6098
        ap_int<128> v3921 = a68;	// L6099
        v3913.write(v3921);	// L6100
      }
    }
  }
}

void gemm_5_1(
  hls::stream< int8_t >& v3922,
  hls::stream< int8_t >& v3923,
  hls::stream< int8_t >& v3924,
  hls::stream< int8_t >& v3925,
  hls::stream< ap_uint<128> >& v3926,
  hls::stream< ap_uint<128> >& v3927
) {	// L6106
  l_S_mt_nt_0_mt88: for (int mt88 = 0; mt88 < 4; mt88++) {	// L6114
    l_nt88: for (int nt88 = 0; nt88 < 4; nt88++) {	// L6115
      int8_t c64;	// L6116
      c64 = 0;	// L6117
      l_S_k_0_k88: for (int k88 = 0; k88 < 64; k88++) {	// L6118
        int8_t v3932 = v3922.read();	// L6119
        int8_t a69;	// L6120
        a69 = v3932;	// L6121
        int8_t v3934 = v3923.read();	// L6122
        int8_t b80;	// L6123
        b80 = v3934;	// L6124
        int8_t v3936 = a69;	// L6125
        int8_t v3937 = b80;	// L6126
        int16_t v3938 = v3936;	// L6127
        int16_t v3939 = v3937;	// L6128
        int16_t v3940 = v3938 * v3939;	// L6129
        int8_t v3941 = c64;	// L6130
        ap_int<17> v3942 = v3941;	// L6131
        ap_int<17> v3943 = v3940;	// L6132
        ap_int<17> v3944 = v3942 + v3943;	// L6133
        int8_t v3945 = v3944;	// L6134
        c64 = v3945;	// L6135
        int8_t v3946 = a69;	// L6136
        v3924.write(v3946);	// L6137
        int8_t v3947 = b80;	// L6138
        v3925.write(v3947);	// L6139
      }
      ap_uint<128> packed_tmp64;	// L6141
      ap_uint<128> v3949 = v3926.read();	// L6142
      packed_tmp64 = v3949;	// L6143
      ap_uint<128> packed_c64;	// L6144
      packed_c64 = 0;	// L6145
      l_S_m_1_m64: for (int m64 = 0; m64 < 16; m64++) {	// L6146
        ap_int<33> v3952 = m64;	// L6147
        bool v3953 = v3952 == 4;	// L6148
        if (v3953) {	// L6149
          int8_t v3954 = c64;	// L6150
          ap_int<128> v3955 = packed_c64;	// L6151
          int64_t v3956 = m64;	// L6152
          int64_t v3957 = v3956 * 8;	// L6153
          ap_int<34> v3958 = m64;	// L6154
          ap_int<34> v3959 = v3958 + 1;	// L6155
          ap_int<66> v3960 = v3959;	// L6156
          ap_int<66> v3961 = v3960 * 8;	// L6157
          ap_int<66> v3962 = v3961 - 1;	// L6158
          int v3963 = v3957;	// L6159
          int v3964 = v3962;	// L6160
          ap_int<128> v3965;
          ap_int<128> v3965_tmp = v3955;
          v3965_tmp(v3964, v3963) = v3954;
          v3965 = v3965_tmp;	// L6161
          packed_c64 = v3965;	// L6162
        } else {
          ap_int<128> v3966 = packed_tmp64;	// L6164
          int64_t v3967 = m64;	// L6165
          int64_t v3968 = v3967 * 8;	// L6166
          ap_int<34> v3969 = m64;	// L6167
          ap_int<34> v3970 = v3969 + 1;	// L6168
          ap_int<66> v3971 = v3970;	// L6169
          ap_int<66> v3972 = v3971 * 8;	// L6170
          ap_int<66> v3973 = v3972 - 1;	// L6171
          int v3974 = v3968;	// L6172
          int v3975 = v3973;	// L6173
          int8_t v3976;
          ap_int<128> v3976_tmp = v3966;
          v3976 = v3976_tmp(v3975, v3974);	// L6174
          ap_int<128> v3977 = packed_c64;	// L6175
          ap_int<128> v3978;
          ap_int<128> v3978_tmp = v3977;
          v3978_tmp(v3975, v3974) = v3976;
          v3978 = v3978_tmp;	// L6185
          packed_c64 = v3978;	// L6186
        }
      }
      ap_int<128> v3979 = packed_c64;	// L6189
      v3927.write(v3979);	// L6190
    }
  }
}

void gemm_5_2(
  hls::stream< int8_t >& v3980,
  hls::stream< int8_t >& v3981,
  hls::stream< int8_t >& v3982,
  hls::stream< int8_t >& v3983,
  hls::stream< ap_uint<128> >& v3984,
  hls::stream< ap_uint<128> >& v3985
) {	// L6195
  l_S_mt_nt_0_mt89: for (int mt89 = 0; mt89 < 4; mt89++) {	// L6203
    l_nt89: for (int nt89 = 0; nt89 < 4; nt89++) {	// L6204
      int8_t c65;	// L6205
      c65 = 0;	// L6206
      l_S_k_0_k89: for (int k89 = 0; k89 < 64; k89++) {	// L6207
        int8_t v3990 = v3980.read();	// L6208
        int8_t a70;	// L6209
        a70 = v3990;	// L6210
        int8_t v3992 = v3981.read();	// L6211
        int8_t b81;	// L6212
        b81 = v3992;	// L6213
        int8_t v3994 = a70;	// L6214
        int8_t v3995 = b81;	// L6215
        int16_t v3996 = v3994;	// L6216
        int16_t v3997 = v3995;	// L6217
        int16_t v3998 = v3996 * v3997;	// L6218
        int8_t v3999 = c65;	// L6219
        ap_int<17> v4000 = v3999;	// L6220
        ap_int<17> v4001 = v3998;	// L6221
        ap_int<17> v4002 = v4000 + v4001;	// L6222
        int8_t v4003 = v4002;	// L6223
        c65 = v4003;	// L6224
        int8_t v4004 = a70;	// L6225
        v3982.write(v4004);	// L6226
        int8_t v4005 = b81;	// L6227
        v3983.write(v4005);	// L6228
      }
      ap_uint<128> packed_tmp65;	// L6230
      ap_uint<128> v4007 = v3984.read();	// L6231
      packed_tmp65 = v4007;	// L6232
      ap_uint<128> packed_c65;	// L6233
      packed_c65 = 0;	// L6234
      l_S_m_1_m65: for (int m65 = 0; m65 < 16; m65++) {	// L6235
        ap_int<33> v4010 = m65;	// L6236
        bool v4011 = v4010 == 4;	// L6237
        if (v4011) {	// L6238
          int8_t v4012 = c65;	// L6239
          ap_int<128> v4013 = packed_c65;	// L6240
          int64_t v4014 = m65;	// L6241
          int64_t v4015 = v4014 * 8;	// L6242
          ap_int<34> v4016 = m65;	// L6243
          ap_int<34> v4017 = v4016 + 1;	// L6244
          ap_int<66> v4018 = v4017;	// L6245
          ap_int<66> v4019 = v4018 * 8;	// L6246
          ap_int<66> v4020 = v4019 - 1;	// L6247
          int v4021 = v4015;	// L6248
          int v4022 = v4020;	// L6249
          ap_int<128> v4023;
          ap_int<128> v4023_tmp = v4013;
          v4023_tmp(v4022, v4021) = v4012;
          v4023 = v4023_tmp;	// L6250
          packed_c65 = v4023;	// L6251
        } else {
          ap_int<128> v4024 = packed_tmp65;	// L6253
          int64_t v4025 = m65;	// L6254
          int64_t v4026 = v4025 * 8;	// L6255
          ap_int<34> v4027 = m65;	// L6256
          ap_int<34> v4028 = v4027 + 1;	// L6257
          ap_int<66> v4029 = v4028;	// L6258
          ap_int<66> v4030 = v4029 * 8;	// L6259
          ap_int<66> v4031 = v4030 - 1;	// L6260
          int v4032 = v4026;	// L6261
          int v4033 = v4031;	// L6262
          int8_t v4034;
          ap_int<128> v4034_tmp = v4024;
          v4034 = v4034_tmp(v4033, v4032);	// L6263
          ap_int<128> v4035 = packed_c65;	// L6264
          ap_int<128> v4036;
          ap_int<128> v4036_tmp = v4035;
          v4036_tmp(v4033, v4032) = v4034;
          v4036 = v4036_tmp;	// L6274
          packed_c65 = v4036;	// L6275
        }
      }
      ap_int<128> v4037 = packed_c65;	// L6278
      v3985.write(v4037);	// L6279
    }
  }
}

void gemm_5_3(
  hls::stream< int8_t >& v4038,
  hls::stream< int8_t >& v4039,
  hls::stream< int8_t >& v4040,
  hls::stream< int8_t >& v4041,
  hls::stream< ap_uint<128> >& v4042,
  hls::stream< ap_uint<128> >& v4043
) {	// L6284
  l_S_mt_nt_0_mt90: for (int mt90 = 0; mt90 < 4; mt90++) {	// L6292
    l_nt90: for (int nt90 = 0; nt90 < 4; nt90++) {	// L6293
      int8_t c66;	// L6294
      c66 = 0;	// L6295
      l_S_k_0_k90: for (int k90 = 0; k90 < 64; k90++) {	// L6296
        int8_t v4048 = v4038.read();	// L6297
        int8_t a71;	// L6298
        a71 = v4048;	// L6299
        int8_t v4050 = v4039.read();	// L6300
        int8_t b82;	// L6301
        b82 = v4050;	// L6302
        int8_t v4052 = a71;	// L6303
        int8_t v4053 = b82;	// L6304
        int16_t v4054 = v4052;	// L6305
        int16_t v4055 = v4053;	// L6306
        int16_t v4056 = v4054 * v4055;	// L6307
        int8_t v4057 = c66;	// L6308
        ap_int<17> v4058 = v4057;	// L6309
        ap_int<17> v4059 = v4056;	// L6310
        ap_int<17> v4060 = v4058 + v4059;	// L6311
        int8_t v4061 = v4060;	// L6312
        c66 = v4061;	// L6313
        int8_t v4062 = a71;	// L6314
        v4040.write(v4062);	// L6315
        int8_t v4063 = b82;	// L6316
        v4041.write(v4063);	// L6317
      }
      ap_uint<128> packed_tmp66;	// L6319
      ap_uint<128> v4065 = v4042.read();	// L6320
      packed_tmp66 = v4065;	// L6321
      ap_uint<128> packed_c66;	// L6322
      packed_c66 = 0;	// L6323
      l_S_m_1_m66: for (int m66 = 0; m66 < 16; m66++) {	// L6324
        ap_int<33> v4068 = m66;	// L6325
        bool v4069 = v4068 == 4;	// L6326
        if (v4069) {	// L6327
          int8_t v4070 = c66;	// L6328
          ap_int<128> v4071 = packed_c66;	// L6329
          int64_t v4072 = m66;	// L6330
          int64_t v4073 = v4072 * 8;	// L6331
          ap_int<34> v4074 = m66;	// L6332
          ap_int<34> v4075 = v4074 + 1;	// L6333
          ap_int<66> v4076 = v4075;	// L6334
          ap_int<66> v4077 = v4076 * 8;	// L6335
          ap_int<66> v4078 = v4077 - 1;	// L6336
          int v4079 = v4073;	// L6337
          int v4080 = v4078;	// L6338
          ap_int<128> v4081;
          ap_int<128> v4081_tmp = v4071;
          v4081_tmp(v4080, v4079) = v4070;
          v4081 = v4081_tmp;	// L6339
          packed_c66 = v4081;	// L6340
        } else {
          ap_int<128> v4082 = packed_tmp66;	// L6342
          int64_t v4083 = m66;	// L6343
          int64_t v4084 = v4083 * 8;	// L6344
          ap_int<34> v4085 = m66;	// L6345
          ap_int<34> v4086 = v4085 + 1;	// L6346
          ap_int<66> v4087 = v4086;	// L6347
          ap_int<66> v4088 = v4087 * 8;	// L6348
          ap_int<66> v4089 = v4088 - 1;	// L6349
          int v4090 = v4084;	// L6350
          int v4091 = v4089;	// L6351
          int8_t v4092;
          ap_int<128> v4092_tmp = v4082;
          v4092 = v4092_tmp(v4091, v4090);	// L6352
          ap_int<128> v4093 = packed_c66;	// L6353
          ap_int<128> v4094;
          ap_int<128> v4094_tmp = v4093;
          v4094_tmp(v4091, v4090) = v4092;
          v4094 = v4094_tmp;	// L6363
          packed_c66 = v4094;	// L6364
        }
      }
      ap_int<128> v4095 = packed_c66;	// L6367
      v4043.write(v4095);	// L6368
    }
  }
}

void gemm_5_4(
  hls::stream< int8_t >& v4096,
  hls::stream< int8_t >& v4097,
  hls::stream< int8_t >& v4098,
  hls::stream< int8_t >& v4099,
  hls::stream< ap_uint<128> >& v4100,
  hls::stream< ap_uint<128> >& v4101
) {	// L6373
  l_S_mt_nt_0_mt91: for (int mt91 = 0; mt91 < 4; mt91++) {	// L6381
    l_nt91: for (int nt91 = 0; nt91 < 4; nt91++) {	// L6382
      int8_t c67;	// L6383
      c67 = 0;	// L6384
      l_S_k_0_k91: for (int k91 = 0; k91 < 64; k91++) {	// L6385
        int8_t v4106 = v4096.read();	// L6386
        int8_t a72;	// L6387
        a72 = v4106;	// L6388
        int8_t v4108 = v4097.read();	// L6389
        int8_t b83;	// L6390
        b83 = v4108;	// L6391
        int8_t v4110 = a72;	// L6392
        int8_t v4111 = b83;	// L6393
        int16_t v4112 = v4110;	// L6394
        int16_t v4113 = v4111;	// L6395
        int16_t v4114 = v4112 * v4113;	// L6396
        int8_t v4115 = c67;	// L6397
        ap_int<17> v4116 = v4115;	// L6398
        ap_int<17> v4117 = v4114;	// L6399
        ap_int<17> v4118 = v4116 + v4117;	// L6400
        int8_t v4119 = v4118;	// L6401
        c67 = v4119;	// L6402
        int8_t v4120 = a72;	// L6403
        v4098.write(v4120);	// L6404
        int8_t v4121 = b83;	// L6405
        v4099.write(v4121);	// L6406
      }
      ap_uint<128> packed_tmp67;	// L6408
      ap_uint<128> v4123 = v4100.read();	// L6409
      packed_tmp67 = v4123;	// L6410
      ap_uint<128> packed_c67;	// L6411
      packed_c67 = 0;	// L6412
      l_S_m_1_m67: for (int m67 = 0; m67 < 16; m67++) {	// L6413
        ap_int<33> v4126 = m67;	// L6414
        bool v4127 = v4126 == 4;	// L6415
        if (v4127) {	// L6416
          int8_t v4128 = c67;	// L6417
          ap_int<128> v4129 = packed_c67;	// L6418
          int64_t v4130 = m67;	// L6419
          int64_t v4131 = v4130 * 8;	// L6420
          ap_int<34> v4132 = m67;	// L6421
          ap_int<34> v4133 = v4132 + 1;	// L6422
          ap_int<66> v4134 = v4133;	// L6423
          ap_int<66> v4135 = v4134 * 8;	// L6424
          ap_int<66> v4136 = v4135 - 1;	// L6425
          int v4137 = v4131;	// L6426
          int v4138 = v4136;	// L6427
          ap_int<128> v4139;
          ap_int<128> v4139_tmp = v4129;
          v4139_tmp(v4138, v4137) = v4128;
          v4139 = v4139_tmp;	// L6428
          packed_c67 = v4139;	// L6429
        } else {
          ap_int<128> v4140 = packed_tmp67;	// L6431
          int64_t v4141 = m67;	// L6432
          int64_t v4142 = v4141 * 8;	// L6433
          ap_int<34> v4143 = m67;	// L6434
          ap_int<34> v4144 = v4143 + 1;	// L6435
          ap_int<66> v4145 = v4144;	// L6436
          ap_int<66> v4146 = v4145 * 8;	// L6437
          ap_int<66> v4147 = v4146 - 1;	// L6438
          int v4148 = v4142;	// L6439
          int v4149 = v4147;	// L6440
          int8_t v4150;
          ap_int<128> v4150_tmp = v4140;
          v4150 = v4150_tmp(v4149, v4148);	// L6441
          ap_int<128> v4151 = packed_c67;	// L6442
          ap_int<128> v4152;
          ap_int<128> v4152_tmp = v4151;
          v4152_tmp(v4149, v4148) = v4150;
          v4152 = v4152_tmp;	// L6452
          packed_c67 = v4152;	// L6453
        }
      }
      ap_int<128> v4153 = packed_c67;	// L6456
      v4101.write(v4153);	// L6457
    }
  }
}

void gemm_5_5(
  hls::stream< int8_t >& v4154,
  hls::stream< int8_t >& v4155,
  hls::stream< int8_t >& v4156,
  hls::stream< int8_t >& v4157,
  hls::stream< ap_uint<128> >& v4158,
  hls::stream< ap_uint<128> >& v4159
) {	// L6462
  l_S_mt_nt_0_mt92: for (int mt92 = 0; mt92 < 4; mt92++) {	// L6470
    l_nt92: for (int nt92 = 0; nt92 < 4; nt92++) {	// L6471
      int8_t c68;	// L6472
      c68 = 0;	// L6473
      l_S_k_0_k92: for (int k92 = 0; k92 < 64; k92++) {	// L6474
        int8_t v4164 = v4154.read();	// L6475
        int8_t a73;	// L6476
        a73 = v4164;	// L6477
        int8_t v4166 = v4155.read();	// L6478
        int8_t b84;	// L6479
        b84 = v4166;	// L6480
        int8_t v4168 = a73;	// L6481
        int8_t v4169 = b84;	// L6482
        int16_t v4170 = v4168;	// L6483
        int16_t v4171 = v4169;	// L6484
        int16_t v4172 = v4170 * v4171;	// L6485
        int8_t v4173 = c68;	// L6486
        ap_int<17> v4174 = v4173;	// L6487
        ap_int<17> v4175 = v4172;	// L6488
        ap_int<17> v4176 = v4174 + v4175;	// L6489
        int8_t v4177 = v4176;	// L6490
        c68 = v4177;	// L6491
        int8_t v4178 = a73;	// L6492
        v4156.write(v4178);	// L6493
        int8_t v4179 = b84;	// L6494
        v4157.write(v4179);	// L6495
      }
      ap_uint<128> packed_tmp68;	// L6497
      ap_uint<128> v4181 = v4158.read();	// L6498
      packed_tmp68 = v4181;	// L6499
      ap_uint<128> packed_c68;	// L6500
      packed_c68 = 0;	// L6501
      l_S_m_1_m68: for (int m68 = 0; m68 < 16; m68++) {	// L6502
        ap_int<33> v4184 = m68;	// L6503
        bool v4185 = v4184 == 4;	// L6504
        if (v4185) {	// L6505
          int8_t v4186 = c68;	// L6506
          ap_int<128> v4187 = packed_c68;	// L6507
          int64_t v4188 = m68;	// L6508
          int64_t v4189 = v4188 * 8;	// L6509
          ap_int<34> v4190 = m68;	// L6510
          ap_int<34> v4191 = v4190 + 1;	// L6511
          ap_int<66> v4192 = v4191;	// L6512
          ap_int<66> v4193 = v4192 * 8;	// L6513
          ap_int<66> v4194 = v4193 - 1;	// L6514
          int v4195 = v4189;	// L6515
          int v4196 = v4194;	// L6516
          ap_int<128> v4197;
          ap_int<128> v4197_tmp = v4187;
          v4197_tmp(v4196, v4195) = v4186;
          v4197 = v4197_tmp;	// L6517
          packed_c68 = v4197;	// L6518
        } else {
          ap_int<128> v4198 = packed_tmp68;	// L6520
          int64_t v4199 = m68;	// L6521
          int64_t v4200 = v4199 * 8;	// L6522
          ap_int<34> v4201 = m68;	// L6523
          ap_int<34> v4202 = v4201 + 1;	// L6524
          ap_int<66> v4203 = v4202;	// L6525
          ap_int<66> v4204 = v4203 * 8;	// L6526
          ap_int<66> v4205 = v4204 - 1;	// L6527
          int v4206 = v4200;	// L6528
          int v4207 = v4205;	// L6529
          int8_t v4208;
          ap_int<128> v4208_tmp = v4198;
          v4208 = v4208_tmp(v4207, v4206);	// L6530
          ap_int<128> v4209 = packed_c68;	// L6531
          ap_int<128> v4210;
          ap_int<128> v4210_tmp = v4209;
          v4210_tmp(v4207, v4206) = v4208;
          v4210 = v4210_tmp;	// L6541
          packed_c68 = v4210;	// L6542
        }
      }
      ap_int<128> v4211 = packed_c68;	// L6545
      v4159.write(v4211);	// L6546
    }
  }
}

void gemm_5_6(
  hls::stream< int8_t >& v4212,
  hls::stream< int8_t >& v4213,
  hls::stream< int8_t >& v4214,
  hls::stream< int8_t >& v4215,
  hls::stream< ap_uint<128> >& v4216,
  hls::stream< ap_uint<128> >& v4217
) {	// L6551
  l_S_mt_nt_0_mt93: for (int mt93 = 0; mt93 < 4; mt93++) {	// L6559
    l_nt93: for (int nt93 = 0; nt93 < 4; nt93++) {	// L6560
      int8_t c69;	// L6561
      c69 = 0;	// L6562
      l_S_k_0_k93: for (int k93 = 0; k93 < 64; k93++) {	// L6563
        int8_t v4222 = v4212.read();	// L6564
        int8_t a74;	// L6565
        a74 = v4222;	// L6566
        int8_t v4224 = v4213.read();	// L6567
        int8_t b85;	// L6568
        b85 = v4224;	// L6569
        int8_t v4226 = a74;	// L6570
        int8_t v4227 = b85;	// L6571
        int16_t v4228 = v4226;	// L6572
        int16_t v4229 = v4227;	// L6573
        int16_t v4230 = v4228 * v4229;	// L6574
        int8_t v4231 = c69;	// L6575
        ap_int<17> v4232 = v4231;	// L6576
        ap_int<17> v4233 = v4230;	// L6577
        ap_int<17> v4234 = v4232 + v4233;	// L6578
        int8_t v4235 = v4234;	// L6579
        c69 = v4235;	// L6580
        int8_t v4236 = a74;	// L6581
        v4214.write(v4236);	// L6582
        int8_t v4237 = b85;	// L6583
        v4215.write(v4237);	// L6584
      }
      ap_uint<128> packed_tmp69;	// L6586
      ap_uint<128> v4239 = v4216.read();	// L6587
      packed_tmp69 = v4239;	// L6588
      ap_uint<128> packed_c69;	// L6589
      packed_c69 = 0;	// L6590
      l_S_m_1_m69: for (int m69 = 0; m69 < 16; m69++) {	// L6591
        ap_int<33> v4242 = m69;	// L6592
        bool v4243 = v4242 == 4;	// L6593
        if (v4243) {	// L6594
          int8_t v4244 = c69;	// L6595
          ap_int<128> v4245 = packed_c69;	// L6596
          int64_t v4246 = m69;	// L6597
          int64_t v4247 = v4246 * 8;	// L6598
          ap_int<34> v4248 = m69;	// L6599
          ap_int<34> v4249 = v4248 + 1;	// L6600
          ap_int<66> v4250 = v4249;	// L6601
          ap_int<66> v4251 = v4250 * 8;	// L6602
          ap_int<66> v4252 = v4251 - 1;	// L6603
          int v4253 = v4247;	// L6604
          int v4254 = v4252;	// L6605
          ap_int<128> v4255;
          ap_int<128> v4255_tmp = v4245;
          v4255_tmp(v4254, v4253) = v4244;
          v4255 = v4255_tmp;	// L6606
          packed_c69 = v4255;	// L6607
        } else {
          ap_int<128> v4256 = packed_tmp69;	// L6609
          int64_t v4257 = m69;	// L6610
          int64_t v4258 = v4257 * 8;	// L6611
          ap_int<34> v4259 = m69;	// L6612
          ap_int<34> v4260 = v4259 + 1;	// L6613
          ap_int<66> v4261 = v4260;	// L6614
          ap_int<66> v4262 = v4261 * 8;	// L6615
          ap_int<66> v4263 = v4262 - 1;	// L6616
          int v4264 = v4258;	// L6617
          int v4265 = v4263;	// L6618
          int8_t v4266;
          ap_int<128> v4266_tmp = v4256;
          v4266 = v4266_tmp(v4265, v4264);	// L6619
          ap_int<128> v4267 = packed_c69;	// L6620
          ap_int<128> v4268;
          ap_int<128> v4268_tmp = v4267;
          v4268_tmp(v4265, v4264) = v4266;
          v4268 = v4268_tmp;	// L6630
          packed_c69 = v4268;	// L6631
        }
      }
      ap_int<128> v4269 = packed_c69;	// L6634
      v4217.write(v4269);	// L6635
    }
  }
}

void gemm_5_7(
  hls::stream< int8_t >& v4270,
  hls::stream< int8_t >& v4271,
  hls::stream< int8_t >& v4272,
  hls::stream< int8_t >& v4273,
  hls::stream< ap_uint<128> >& v4274,
  hls::stream< ap_uint<128> >& v4275
) {	// L6640
  l_S_mt_nt_0_mt94: for (int mt94 = 0; mt94 < 4; mt94++) {	// L6648
    l_nt94: for (int nt94 = 0; nt94 < 4; nt94++) {	// L6649
      int8_t c70;	// L6650
      c70 = 0;	// L6651
      l_S_k_0_k94: for (int k94 = 0; k94 < 64; k94++) {	// L6652
        int8_t v4280 = v4270.read();	// L6653
        int8_t a75;	// L6654
        a75 = v4280;	// L6655
        int8_t v4282 = v4271.read();	// L6656
        int8_t b86;	// L6657
        b86 = v4282;	// L6658
        int8_t v4284 = a75;	// L6659
        int8_t v4285 = b86;	// L6660
        int16_t v4286 = v4284;	// L6661
        int16_t v4287 = v4285;	// L6662
        int16_t v4288 = v4286 * v4287;	// L6663
        int8_t v4289 = c70;	// L6664
        ap_int<17> v4290 = v4289;	// L6665
        ap_int<17> v4291 = v4288;	// L6666
        ap_int<17> v4292 = v4290 + v4291;	// L6667
        int8_t v4293 = v4292;	// L6668
        c70 = v4293;	// L6669
        int8_t v4294 = a75;	// L6670
        v4272.write(v4294);	// L6671
        int8_t v4295 = b86;	// L6672
        v4273.write(v4295);	// L6673
      }
      ap_uint<128> packed_tmp70;	// L6675
      ap_uint<128> v4297 = v4274.read();	// L6676
      packed_tmp70 = v4297;	// L6677
      ap_uint<128> packed_c70;	// L6678
      packed_c70 = 0;	// L6679
      l_S_m_1_m70: for (int m70 = 0; m70 < 16; m70++) {	// L6680
        ap_int<33> v4300 = m70;	// L6681
        bool v4301 = v4300 == 4;	// L6682
        if (v4301) {	// L6683
          int8_t v4302 = c70;	// L6684
          ap_int<128> v4303 = packed_c70;	// L6685
          int64_t v4304 = m70;	// L6686
          int64_t v4305 = v4304 * 8;	// L6687
          ap_int<34> v4306 = m70;	// L6688
          ap_int<34> v4307 = v4306 + 1;	// L6689
          ap_int<66> v4308 = v4307;	// L6690
          ap_int<66> v4309 = v4308 * 8;	// L6691
          ap_int<66> v4310 = v4309 - 1;	// L6692
          int v4311 = v4305;	// L6693
          int v4312 = v4310;	// L6694
          ap_int<128> v4313;
          ap_int<128> v4313_tmp = v4303;
          v4313_tmp(v4312, v4311) = v4302;
          v4313 = v4313_tmp;	// L6695
          packed_c70 = v4313;	// L6696
        } else {
          ap_int<128> v4314 = packed_tmp70;	// L6698
          int64_t v4315 = m70;	// L6699
          int64_t v4316 = v4315 * 8;	// L6700
          ap_int<34> v4317 = m70;	// L6701
          ap_int<34> v4318 = v4317 + 1;	// L6702
          ap_int<66> v4319 = v4318;	// L6703
          ap_int<66> v4320 = v4319 * 8;	// L6704
          ap_int<66> v4321 = v4320 - 1;	// L6705
          int v4322 = v4316;	// L6706
          int v4323 = v4321;	// L6707
          int8_t v4324;
          ap_int<128> v4324_tmp = v4314;
          v4324 = v4324_tmp(v4323, v4322);	// L6708
          ap_int<128> v4325 = packed_c70;	// L6709
          ap_int<128> v4326;
          ap_int<128> v4326_tmp = v4325;
          v4326_tmp(v4323, v4322) = v4324;
          v4326 = v4326_tmp;	// L6719
          packed_c70 = v4326;	// L6720
        }
      }
      ap_int<128> v4327 = packed_c70;	// L6723
      v4275.write(v4327);	// L6724
    }
  }
}

void gemm_5_8(
  hls::stream< int8_t >& v4328,
  hls::stream< int8_t >& v4329,
  hls::stream< int8_t >& v4330,
  hls::stream< int8_t >& v4331,
  hls::stream< ap_uint<128> >& v4332,
  hls::stream< ap_uint<128> >& v4333
) {	// L6729
  l_S_mt_nt_0_mt95: for (int mt95 = 0; mt95 < 4; mt95++) {	// L6737
    l_nt95: for (int nt95 = 0; nt95 < 4; nt95++) {	// L6738
      int8_t c71;	// L6739
      c71 = 0;	// L6740
      l_S_k_0_k95: for (int k95 = 0; k95 < 64; k95++) {	// L6741
        int8_t v4338 = v4328.read();	// L6742
        int8_t a76;	// L6743
        a76 = v4338;	// L6744
        int8_t v4340 = v4329.read();	// L6745
        int8_t b87;	// L6746
        b87 = v4340;	// L6747
        int8_t v4342 = a76;	// L6748
        int8_t v4343 = b87;	// L6749
        int16_t v4344 = v4342;	// L6750
        int16_t v4345 = v4343;	// L6751
        int16_t v4346 = v4344 * v4345;	// L6752
        int8_t v4347 = c71;	// L6753
        ap_int<17> v4348 = v4347;	// L6754
        ap_int<17> v4349 = v4346;	// L6755
        ap_int<17> v4350 = v4348 + v4349;	// L6756
        int8_t v4351 = v4350;	// L6757
        c71 = v4351;	// L6758
        int8_t v4352 = a76;	// L6759
        v4330.write(v4352);	// L6760
        int8_t v4353 = b87;	// L6761
        v4331.write(v4353);	// L6762
      }
      ap_uint<128> packed_tmp71;	// L6764
      ap_uint<128> v4355 = v4332.read();	// L6765
      packed_tmp71 = v4355;	// L6766
      ap_uint<128> packed_c71;	// L6767
      packed_c71 = 0;	// L6768
      l_S_m_1_m71: for (int m71 = 0; m71 < 16; m71++) {	// L6769
        ap_int<33> v4358 = m71;	// L6770
        bool v4359 = v4358 == 4;	// L6771
        if (v4359) {	// L6772
          int8_t v4360 = c71;	// L6773
          ap_int<128> v4361 = packed_c71;	// L6774
          int64_t v4362 = m71;	// L6775
          int64_t v4363 = v4362 * 8;	// L6776
          ap_int<34> v4364 = m71;	// L6777
          ap_int<34> v4365 = v4364 + 1;	// L6778
          ap_int<66> v4366 = v4365;	// L6779
          ap_int<66> v4367 = v4366 * 8;	// L6780
          ap_int<66> v4368 = v4367 - 1;	// L6781
          int v4369 = v4363;	// L6782
          int v4370 = v4368;	// L6783
          ap_int<128> v4371;
          ap_int<128> v4371_tmp = v4361;
          v4371_tmp(v4370, v4369) = v4360;
          v4371 = v4371_tmp;	// L6784
          packed_c71 = v4371;	// L6785
        } else {
          ap_int<128> v4372 = packed_tmp71;	// L6787
          int64_t v4373 = m71;	// L6788
          int64_t v4374 = v4373 * 8;	// L6789
          ap_int<34> v4375 = m71;	// L6790
          ap_int<34> v4376 = v4375 + 1;	// L6791
          ap_int<66> v4377 = v4376;	// L6792
          ap_int<66> v4378 = v4377 * 8;	// L6793
          ap_int<66> v4379 = v4378 - 1;	// L6794
          int v4380 = v4374;	// L6795
          int v4381 = v4379;	// L6796
          int8_t v4382;
          ap_int<128> v4382_tmp = v4372;
          v4382 = v4382_tmp(v4381, v4380);	// L6797
          ap_int<128> v4383 = packed_c71;	// L6798
          ap_int<128> v4384;
          ap_int<128> v4384_tmp = v4383;
          v4384_tmp(v4381, v4380) = v4382;
          v4384 = v4384_tmp;	// L6808
          packed_c71 = v4384;	// L6809
        }
      }
      ap_int<128> v4385 = packed_c71;	// L6812
      v4333.write(v4385);	// L6813
    }
  }
}

void gemm_5_9(
  hls::stream< int8_t >& v4386,
  hls::stream< int8_t >& v4387,
  hls::stream< int8_t >& v4388,
  hls::stream< int8_t >& v4389,
  hls::stream< ap_uint<128> >& v4390,
  hls::stream< ap_uint<128> >& v4391
) {	// L6818
  l_S_mt_nt_0_mt96: for (int mt96 = 0; mt96 < 4; mt96++) {	// L6826
    l_nt96: for (int nt96 = 0; nt96 < 4; nt96++) {	// L6827
      int8_t c72;	// L6828
      c72 = 0;	// L6829
      l_S_k_0_k96: for (int k96 = 0; k96 < 64; k96++) {	// L6830
        int8_t v4396 = v4386.read();	// L6831
        int8_t a77;	// L6832
        a77 = v4396;	// L6833
        int8_t v4398 = v4387.read();	// L6834
        int8_t b88;	// L6835
        b88 = v4398;	// L6836
        int8_t v4400 = a77;	// L6837
        int8_t v4401 = b88;	// L6838
        int16_t v4402 = v4400;	// L6839
        int16_t v4403 = v4401;	// L6840
        int16_t v4404 = v4402 * v4403;	// L6841
        int8_t v4405 = c72;	// L6842
        ap_int<17> v4406 = v4405;	// L6843
        ap_int<17> v4407 = v4404;	// L6844
        ap_int<17> v4408 = v4406 + v4407;	// L6845
        int8_t v4409 = v4408;	// L6846
        c72 = v4409;	// L6847
        int8_t v4410 = a77;	// L6848
        v4388.write(v4410);	// L6849
        int8_t v4411 = b88;	// L6850
        v4389.write(v4411);	// L6851
      }
      ap_uint<128> packed_tmp72;	// L6853
      ap_uint<128> v4413 = v4390.read();	// L6854
      packed_tmp72 = v4413;	// L6855
      ap_uint<128> packed_c72;	// L6856
      packed_c72 = 0;	// L6857
      l_S_m_1_m72: for (int m72 = 0; m72 < 16; m72++) {	// L6858
        ap_int<33> v4416 = m72;	// L6859
        bool v4417 = v4416 == 4;	// L6860
        if (v4417) {	// L6861
          int8_t v4418 = c72;	// L6862
          ap_int<128> v4419 = packed_c72;	// L6863
          int64_t v4420 = m72;	// L6864
          int64_t v4421 = v4420 * 8;	// L6865
          ap_int<34> v4422 = m72;	// L6866
          ap_int<34> v4423 = v4422 + 1;	// L6867
          ap_int<66> v4424 = v4423;	// L6868
          ap_int<66> v4425 = v4424 * 8;	// L6869
          ap_int<66> v4426 = v4425 - 1;	// L6870
          int v4427 = v4421;	// L6871
          int v4428 = v4426;	// L6872
          ap_int<128> v4429;
          ap_int<128> v4429_tmp = v4419;
          v4429_tmp(v4428, v4427) = v4418;
          v4429 = v4429_tmp;	// L6873
          packed_c72 = v4429;	// L6874
        } else {
          ap_int<128> v4430 = packed_tmp72;	// L6876
          int64_t v4431 = m72;	// L6877
          int64_t v4432 = v4431 * 8;	// L6878
          ap_int<34> v4433 = m72;	// L6879
          ap_int<34> v4434 = v4433 + 1;	// L6880
          ap_int<66> v4435 = v4434;	// L6881
          ap_int<66> v4436 = v4435 * 8;	// L6882
          ap_int<66> v4437 = v4436 - 1;	// L6883
          int v4438 = v4432;	// L6884
          int v4439 = v4437;	// L6885
          int8_t v4440;
          ap_int<128> v4440_tmp = v4430;
          v4440 = v4440_tmp(v4439, v4438);	// L6886
          ap_int<128> v4441 = packed_c72;	// L6887
          ap_int<128> v4442;
          ap_int<128> v4442_tmp = v4441;
          v4442_tmp(v4439, v4438) = v4440;
          v4442 = v4442_tmp;	// L6897
          packed_c72 = v4442;	// L6898
        }
      }
      ap_int<128> v4443 = packed_c72;	// L6901
      v4391.write(v4443);	// L6902
    }
  }
}

void gemm_5_10(
  hls::stream< int8_t >& v4444,
  hls::stream< int8_t >& v4445,
  hls::stream< int8_t >& v4446,
  hls::stream< int8_t >& v4447,
  hls::stream< ap_uint<128> >& v4448,
  hls::stream< ap_uint<128> >& v4449
) {	// L6907
  l_S_mt_nt_0_mt97: for (int mt97 = 0; mt97 < 4; mt97++) {	// L6915
    l_nt97: for (int nt97 = 0; nt97 < 4; nt97++) {	// L6916
      int8_t c73;	// L6917
      c73 = 0;	// L6918
      l_S_k_0_k97: for (int k97 = 0; k97 < 64; k97++) {	// L6919
        int8_t v4454 = v4444.read();	// L6920
        int8_t a78;	// L6921
        a78 = v4454;	// L6922
        int8_t v4456 = v4445.read();	// L6923
        int8_t b89;	// L6924
        b89 = v4456;	// L6925
        int8_t v4458 = a78;	// L6926
        int8_t v4459 = b89;	// L6927
        int16_t v4460 = v4458;	// L6928
        int16_t v4461 = v4459;	// L6929
        int16_t v4462 = v4460 * v4461;	// L6930
        int8_t v4463 = c73;	// L6931
        ap_int<17> v4464 = v4463;	// L6932
        ap_int<17> v4465 = v4462;	// L6933
        ap_int<17> v4466 = v4464 + v4465;	// L6934
        int8_t v4467 = v4466;	// L6935
        c73 = v4467;	// L6936
        int8_t v4468 = a78;	// L6937
        v4446.write(v4468);	// L6938
        int8_t v4469 = b89;	// L6939
        v4447.write(v4469);	// L6940
      }
      ap_uint<128> packed_tmp73;	// L6942
      ap_uint<128> v4471 = v4448.read();	// L6943
      packed_tmp73 = v4471;	// L6944
      ap_uint<128> packed_c73;	// L6945
      packed_c73 = 0;	// L6946
      l_S_m_1_m73: for (int m73 = 0; m73 < 16; m73++) {	// L6947
        ap_int<33> v4474 = m73;	// L6948
        bool v4475 = v4474 == 4;	// L6949
        if (v4475) {	// L6950
          int8_t v4476 = c73;	// L6951
          ap_int<128> v4477 = packed_c73;	// L6952
          int64_t v4478 = m73;	// L6953
          int64_t v4479 = v4478 * 8;	// L6954
          ap_int<34> v4480 = m73;	// L6955
          ap_int<34> v4481 = v4480 + 1;	// L6956
          ap_int<66> v4482 = v4481;	// L6957
          ap_int<66> v4483 = v4482 * 8;	// L6958
          ap_int<66> v4484 = v4483 - 1;	// L6959
          int v4485 = v4479;	// L6960
          int v4486 = v4484;	// L6961
          ap_int<128> v4487;
          ap_int<128> v4487_tmp = v4477;
          v4487_tmp(v4486, v4485) = v4476;
          v4487 = v4487_tmp;	// L6962
          packed_c73 = v4487;	// L6963
        } else {
          ap_int<128> v4488 = packed_tmp73;	// L6965
          int64_t v4489 = m73;	// L6966
          int64_t v4490 = v4489 * 8;	// L6967
          ap_int<34> v4491 = m73;	// L6968
          ap_int<34> v4492 = v4491 + 1;	// L6969
          ap_int<66> v4493 = v4492;	// L6970
          ap_int<66> v4494 = v4493 * 8;	// L6971
          ap_int<66> v4495 = v4494 - 1;	// L6972
          int v4496 = v4490;	// L6973
          int v4497 = v4495;	// L6974
          int8_t v4498;
          ap_int<128> v4498_tmp = v4488;
          v4498 = v4498_tmp(v4497, v4496);	// L6975
          ap_int<128> v4499 = packed_c73;	// L6976
          ap_int<128> v4500;
          ap_int<128> v4500_tmp = v4499;
          v4500_tmp(v4497, v4496) = v4498;
          v4500 = v4500_tmp;	// L6986
          packed_c73 = v4500;	// L6987
        }
      }
      ap_int<128> v4501 = packed_c73;	// L6990
      v4449.write(v4501);	// L6991
    }
  }
}

void gemm_5_11(
  hls::stream< int8_t >& v4502,
  hls::stream< int8_t >& v4503,
  hls::stream< int8_t >& v4504,
  hls::stream< int8_t >& v4505,
  hls::stream< ap_uint<128> >& v4506,
  hls::stream< ap_uint<128> >& v4507
) {	// L6996
  l_S_mt_nt_0_mt98: for (int mt98 = 0; mt98 < 4; mt98++) {	// L7004
    l_nt98: for (int nt98 = 0; nt98 < 4; nt98++) {	// L7005
      int8_t c74;	// L7006
      c74 = 0;	// L7007
      l_S_k_0_k98: for (int k98 = 0; k98 < 64; k98++) {	// L7008
        int8_t v4512 = v4502.read();	// L7009
        int8_t a79;	// L7010
        a79 = v4512;	// L7011
        int8_t v4514 = v4503.read();	// L7012
        int8_t b90;	// L7013
        b90 = v4514;	// L7014
        int8_t v4516 = a79;	// L7015
        int8_t v4517 = b90;	// L7016
        int16_t v4518 = v4516;	// L7017
        int16_t v4519 = v4517;	// L7018
        int16_t v4520 = v4518 * v4519;	// L7019
        int8_t v4521 = c74;	// L7020
        ap_int<17> v4522 = v4521;	// L7021
        ap_int<17> v4523 = v4520;	// L7022
        ap_int<17> v4524 = v4522 + v4523;	// L7023
        int8_t v4525 = v4524;	// L7024
        c74 = v4525;	// L7025
        int8_t v4526 = a79;	// L7026
        v4504.write(v4526);	// L7027
        int8_t v4527 = b90;	// L7028
        v4505.write(v4527);	// L7029
      }
      ap_uint<128> packed_tmp74;	// L7031
      ap_uint<128> v4529 = v4506.read();	// L7032
      packed_tmp74 = v4529;	// L7033
      ap_uint<128> packed_c74;	// L7034
      packed_c74 = 0;	// L7035
      l_S_m_1_m74: for (int m74 = 0; m74 < 16; m74++) {	// L7036
        ap_int<33> v4532 = m74;	// L7037
        bool v4533 = v4532 == 4;	// L7038
        if (v4533) {	// L7039
          int8_t v4534 = c74;	// L7040
          ap_int<128> v4535 = packed_c74;	// L7041
          int64_t v4536 = m74;	// L7042
          int64_t v4537 = v4536 * 8;	// L7043
          ap_int<34> v4538 = m74;	// L7044
          ap_int<34> v4539 = v4538 + 1;	// L7045
          ap_int<66> v4540 = v4539;	// L7046
          ap_int<66> v4541 = v4540 * 8;	// L7047
          ap_int<66> v4542 = v4541 - 1;	// L7048
          int v4543 = v4537;	// L7049
          int v4544 = v4542;	// L7050
          ap_int<128> v4545;
          ap_int<128> v4545_tmp = v4535;
          v4545_tmp(v4544, v4543) = v4534;
          v4545 = v4545_tmp;	// L7051
          packed_c74 = v4545;	// L7052
        } else {
          ap_int<128> v4546 = packed_tmp74;	// L7054
          int64_t v4547 = m74;	// L7055
          int64_t v4548 = v4547 * 8;	// L7056
          ap_int<34> v4549 = m74;	// L7057
          ap_int<34> v4550 = v4549 + 1;	// L7058
          ap_int<66> v4551 = v4550;	// L7059
          ap_int<66> v4552 = v4551 * 8;	// L7060
          ap_int<66> v4553 = v4552 - 1;	// L7061
          int v4554 = v4548;	// L7062
          int v4555 = v4553;	// L7063
          int8_t v4556;
          ap_int<128> v4556_tmp = v4546;
          v4556 = v4556_tmp(v4555, v4554);	// L7064
          ap_int<128> v4557 = packed_c74;	// L7065
          ap_int<128> v4558;
          ap_int<128> v4558_tmp = v4557;
          v4558_tmp(v4555, v4554) = v4556;
          v4558 = v4558_tmp;	// L7075
          packed_c74 = v4558;	// L7076
        }
      }
      ap_int<128> v4559 = packed_c74;	// L7079
      v4507.write(v4559);	// L7080
    }
  }
}

void gemm_5_12(
  hls::stream< int8_t >& v4560,
  hls::stream< int8_t >& v4561,
  hls::stream< int8_t >& v4562,
  hls::stream< int8_t >& v4563,
  hls::stream< ap_uint<128> >& v4564,
  hls::stream< ap_uint<128> >& v4565
) {	// L7085
  l_S_mt_nt_0_mt99: for (int mt99 = 0; mt99 < 4; mt99++) {	// L7093
    l_nt99: for (int nt99 = 0; nt99 < 4; nt99++) {	// L7094
      int8_t c75;	// L7095
      c75 = 0;	// L7096
      l_S_k_0_k99: for (int k99 = 0; k99 < 64; k99++) {	// L7097
        int8_t v4570 = v4560.read();	// L7098
        int8_t a80;	// L7099
        a80 = v4570;	// L7100
        int8_t v4572 = v4561.read();	// L7101
        int8_t b91;	// L7102
        b91 = v4572;	// L7103
        int8_t v4574 = a80;	// L7104
        int8_t v4575 = b91;	// L7105
        int16_t v4576 = v4574;	// L7106
        int16_t v4577 = v4575;	// L7107
        int16_t v4578 = v4576 * v4577;	// L7108
        int8_t v4579 = c75;	// L7109
        ap_int<17> v4580 = v4579;	// L7110
        ap_int<17> v4581 = v4578;	// L7111
        ap_int<17> v4582 = v4580 + v4581;	// L7112
        int8_t v4583 = v4582;	// L7113
        c75 = v4583;	// L7114
        int8_t v4584 = a80;	// L7115
        v4562.write(v4584);	// L7116
        int8_t v4585 = b91;	// L7117
        v4563.write(v4585);	// L7118
      }
      ap_uint<128> packed_tmp75;	// L7120
      ap_uint<128> v4587 = v4564.read();	// L7121
      packed_tmp75 = v4587;	// L7122
      ap_uint<128> packed_c75;	// L7123
      packed_c75 = 0;	// L7124
      l_S_m_1_m75: for (int m75 = 0; m75 < 16; m75++) {	// L7125
        ap_int<33> v4590 = m75;	// L7126
        bool v4591 = v4590 == 4;	// L7127
        if (v4591) {	// L7128
          int8_t v4592 = c75;	// L7129
          ap_int<128> v4593 = packed_c75;	// L7130
          int64_t v4594 = m75;	// L7131
          int64_t v4595 = v4594 * 8;	// L7132
          ap_int<34> v4596 = m75;	// L7133
          ap_int<34> v4597 = v4596 + 1;	// L7134
          ap_int<66> v4598 = v4597;	// L7135
          ap_int<66> v4599 = v4598 * 8;	// L7136
          ap_int<66> v4600 = v4599 - 1;	// L7137
          int v4601 = v4595;	// L7138
          int v4602 = v4600;	// L7139
          ap_int<128> v4603;
          ap_int<128> v4603_tmp = v4593;
          v4603_tmp(v4602, v4601) = v4592;
          v4603 = v4603_tmp;	// L7140
          packed_c75 = v4603;	// L7141
        } else {
          ap_int<128> v4604 = packed_tmp75;	// L7143
          int64_t v4605 = m75;	// L7144
          int64_t v4606 = v4605 * 8;	// L7145
          ap_int<34> v4607 = m75;	// L7146
          ap_int<34> v4608 = v4607 + 1;	// L7147
          ap_int<66> v4609 = v4608;	// L7148
          ap_int<66> v4610 = v4609 * 8;	// L7149
          ap_int<66> v4611 = v4610 - 1;	// L7150
          int v4612 = v4606;	// L7151
          int v4613 = v4611;	// L7152
          int8_t v4614;
          ap_int<128> v4614_tmp = v4604;
          v4614 = v4614_tmp(v4613, v4612);	// L7153
          ap_int<128> v4615 = packed_c75;	// L7154
          ap_int<128> v4616;
          ap_int<128> v4616_tmp = v4615;
          v4616_tmp(v4613, v4612) = v4614;
          v4616 = v4616_tmp;	// L7164
          packed_c75 = v4616;	// L7165
        }
      }
      ap_int<128> v4617 = packed_c75;	// L7168
      v4565.write(v4617);	// L7169
    }
  }
}

void gemm_5_13(
  hls::stream< int8_t >& v4618,
  hls::stream< int8_t >& v4619,
  hls::stream< int8_t >& v4620,
  hls::stream< int8_t >& v4621,
  hls::stream< ap_uint<128> >& v4622,
  hls::stream< ap_uint<128> >& v4623
) {	// L7174
  l_S_mt_nt_0_mt100: for (int mt100 = 0; mt100 < 4; mt100++) {	// L7182
    l_nt100: for (int nt100 = 0; nt100 < 4; nt100++) {	// L7183
      int8_t c76;	// L7184
      c76 = 0;	// L7185
      l_S_k_0_k100: for (int k100 = 0; k100 < 64; k100++) {	// L7186
        int8_t v4628 = v4618.read();	// L7187
        int8_t a81;	// L7188
        a81 = v4628;	// L7189
        int8_t v4630 = v4619.read();	// L7190
        int8_t b92;	// L7191
        b92 = v4630;	// L7192
        int8_t v4632 = a81;	// L7193
        int8_t v4633 = b92;	// L7194
        int16_t v4634 = v4632;	// L7195
        int16_t v4635 = v4633;	// L7196
        int16_t v4636 = v4634 * v4635;	// L7197
        int8_t v4637 = c76;	// L7198
        ap_int<17> v4638 = v4637;	// L7199
        ap_int<17> v4639 = v4636;	// L7200
        ap_int<17> v4640 = v4638 + v4639;	// L7201
        int8_t v4641 = v4640;	// L7202
        c76 = v4641;	// L7203
        int8_t v4642 = a81;	// L7204
        v4620.write(v4642);	// L7205
        int8_t v4643 = b92;	// L7206
        v4621.write(v4643);	// L7207
      }
      ap_uint<128> packed_tmp76;	// L7209
      ap_uint<128> v4645 = v4622.read();	// L7210
      packed_tmp76 = v4645;	// L7211
      ap_uint<128> packed_c76;	// L7212
      packed_c76 = 0;	// L7213
      l_S_m_1_m76: for (int m76 = 0; m76 < 16; m76++) {	// L7214
        ap_int<33> v4648 = m76;	// L7215
        bool v4649 = v4648 == 4;	// L7216
        if (v4649) {	// L7217
          int8_t v4650 = c76;	// L7218
          ap_int<128> v4651 = packed_c76;	// L7219
          int64_t v4652 = m76;	// L7220
          int64_t v4653 = v4652 * 8;	// L7221
          ap_int<34> v4654 = m76;	// L7222
          ap_int<34> v4655 = v4654 + 1;	// L7223
          ap_int<66> v4656 = v4655;	// L7224
          ap_int<66> v4657 = v4656 * 8;	// L7225
          ap_int<66> v4658 = v4657 - 1;	// L7226
          int v4659 = v4653;	// L7227
          int v4660 = v4658;	// L7228
          ap_int<128> v4661;
          ap_int<128> v4661_tmp = v4651;
          v4661_tmp(v4660, v4659) = v4650;
          v4661 = v4661_tmp;	// L7229
          packed_c76 = v4661;	// L7230
        } else {
          ap_int<128> v4662 = packed_tmp76;	// L7232
          int64_t v4663 = m76;	// L7233
          int64_t v4664 = v4663 * 8;	// L7234
          ap_int<34> v4665 = m76;	// L7235
          ap_int<34> v4666 = v4665 + 1;	// L7236
          ap_int<66> v4667 = v4666;	// L7237
          ap_int<66> v4668 = v4667 * 8;	// L7238
          ap_int<66> v4669 = v4668 - 1;	// L7239
          int v4670 = v4664;	// L7240
          int v4671 = v4669;	// L7241
          int8_t v4672;
          ap_int<128> v4672_tmp = v4662;
          v4672 = v4672_tmp(v4671, v4670);	// L7242
          ap_int<128> v4673 = packed_c76;	// L7243
          ap_int<128> v4674;
          ap_int<128> v4674_tmp = v4673;
          v4674_tmp(v4671, v4670) = v4672;
          v4674 = v4674_tmp;	// L7253
          packed_c76 = v4674;	// L7254
        }
      }
      ap_int<128> v4675 = packed_c76;	// L7257
      v4623.write(v4675);	// L7258
    }
  }
}

void gemm_5_14(
  hls::stream< int8_t >& v4676,
  hls::stream< int8_t >& v4677,
  hls::stream< int8_t >& v4678,
  hls::stream< int8_t >& v4679,
  hls::stream< ap_uint<128> >& v4680,
  hls::stream< ap_uint<128> >& v4681
) {	// L7263
  l_S_mt_nt_0_mt101: for (int mt101 = 0; mt101 < 4; mt101++) {	// L7271
    l_nt101: for (int nt101 = 0; nt101 < 4; nt101++) {	// L7272
      int8_t c77;	// L7273
      c77 = 0;	// L7274
      l_S_k_0_k101: for (int k101 = 0; k101 < 64; k101++) {	// L7275
        int8_t v4686 = v4676.read();	// L7276
        int8_t a82;	// L7277
        a82 = v4686;	// L7278
        int8_t v4688 = v4677.read();	// L7279
        int8_t b93;	// L7280
        b93 = v4688;	// L7281
        int8_t v4690 = a82;	// L7282
        int8_t v4691 = b93;	// L7283
        int16_t v4692 = v4690;	// L7284
        int16_t v4693 = v4691;	// L7285
        int16_t v4694 = v4692 * v4693;	// L7286
        int8_t v4695 = c77;	// L7287
        ap_int<17> v4696 = v4695;	// L7288
        ap_int<17> v4697 = v4694;	// L7289
        ap_int<17> v4698 = v4696 + v4697;	// L7290
        int8_t v4699 = v4698;	// L7291
        c77 = v4699;	// L7292
        int8_t v4700 = a82;	// L7293
        v4678.write(v4700);	// L7294
        int8_t v4701 = b93;	// L7295
        v4679.write(v4701);	// L7296
      }
      ap_uint<128> packed_tmp77;	// L7298
      ap_uint<128> v4703 = v4680.read();	// L7299
      packed_tmp77 = v4703;	// L7300
      ap_uint<128> packed_c77;	// L7301
      packed_c77 = 0;	// L7302
      l_S_m_1_m77: for (int m77 = 0; m77 < 16; m77++) {	// L7303
        ap_int<33> v4706 = m77;	// L7304
        bool v4707 = v4706 == 4;	// L7305
        if (v4707) {	// L7306
          int8_t v4708 = c77;	// L7307
          ap_int<128> v4709 = packed_c77;	// L7308
          int64_t v4710 = m77;	// L7309
          int64_t v4711 = v4710 * 8;	// L7310
          ap_int<34> v4712 = m77;	// L7311
          ap_int<34> v4713 = v4712 + 1;	// L7312
          ap_int<66> v4714 = v4713;	// L7313
          ap_int<66> v4715 = v4714 * 8;	// L7314
          ap_int<66> v4716 = v4715 - 1;	// L7315
          int v4717 = v4711;	// L7316
          int v4718 = v4716;	// L7317
          ap_int<128> v4719;
          ap_int<128> v4719_tmp = v4709;
          v4719_tmp(v4718, v4717) = v4708;
          v4719 = v4719_tmp;	// L7318
          packed_c77 = v4719;	// L7319
        } else {
          ap_int<128> v4720 = packed_tmp77;	// L7321
          int64_t v4721 = m77;	// L7322
          int64_t v4722 = v4721 * 8;	// L7323
          ap_int<34> v4723 = m77;	// L7324
          ap_int<34> v4724 = v4723 + 1;	// L7325
          ap_int<66> v4725 = v4724;	// L7326
          ap_int<66> v4726 = v4725 * 8;	// L7327
          ap_int<66> v4727 = v4726 - 1;	// L7328
          int v4728 = v4722;	// L7329
          int v4729 = v4727;	// L7330
          int8_t v4730;
          ap_int<128> v4730_tmp = v4720;
          v4730 = v4730_tmp(v4729, v4728);	// L7331
          ap_int<128> v4731 = packed_c77;	// L7332
          ap_int<128> v4732;
          ap_int<128> v4732_tmp = v4731;
          v4732_tmp(v4729, v4728) = v4730;
          v4732 = v4732_tmp;	// L7342
          packed_c77 = v4732;	// L7343
        }
      }
      ap_int<128> v4733 = packed_c77;	// L7346
      v4681.write(v4733);	// L7347
    }
  }
}

void gemm_5_15(
  hls::stream< int8_t >& v4734,
  hls::stream< int8_t >& v4735,
  hls::stream< int8_t >& v4736,
  hls::stream< int8_t >& v4737,
  hls::stream< ap_uint<128> >& v4738,
  hls::stream< ap_uint<128> >& v4739
) {	// L7352
  l_S_mt_nt_0_mt102: for (int mt102 = 0; mt102 < 4; mt102++) {	// L7360
    l_nt102: for (int nt102 = 0; nt102 < 4; nt102++) {	// L7361
      int8_t c78;	// L7362
      c78 = 0;	// L7363
      l_S_k_0_k102: for (int k102 = 0; k102 < 64; k102++) {	// L7364
        int8_t v4744 = v4734.read();	// L7365
        int8_t a83;	// L7366
        a83 = v4744;	// L7367
        int8_t v4746 = v4735.read();	// L7368
        int8_t b94;	// L7369
        b94 = v4746;	// L7370
        int8_t v4748 = a83;	// L7371
        int8_t v4749 = b94;	// L7372
        int16_t v4750 = v4748;	// L7373
        int16_t v4751 = v4749;	// L7374
        int16_t v4752 = v4750 * v4751;	// L7375
        int8_t v4753 = c78;	// L7376
        ap_int<17> v4754 = v4753;	// L7377
        ap_int<17> v4755 = v4752;	// L7378
        ap_int<17> v4756 = v4754 + v4755;	// L7379
        int8_t v4757 = v4756;	// L7380
        c78 = v4757;	// L7381
        int8_t v4758 = a83;	// L7382
        v4736.write(v4758);	// L7383
        int8_t v4759 = b94;	// L7384
        v4737.write(v4759);	// L7385
      }
      ap_uint<128> packed_tmp78;	// L7387
      ap_uint<128> v4761 = v4738.read();	// L7388
      packed_tmp78 = v4761;	// L7389
      ap_uint<128> packed_c78;	// L7390
      packed_c78 = 0;	// L7391
      l_S_m_1_m78: for (int m78 = 0; m78 < 16; m78++) {	// L7392
        ap_int<33> v4764 = m78;	// L7393
        bool v4765 = v4764 == 4;	// L7394
        if (v4765) {	// L7395
          int8_t v4766 = c78;	// L7396
          ap_int<128> v4767 = packed_c78;	// L7397
          int64_t v4768 = m78;	// L7398
          int64_t v4769 = v4768 * 8;	// L7399
          ap_int<34> v4770 = m78;	// L7400
          ap_int<34> v4771 = v4770 + 1;	// L7401
          ap_int<66> v4772 = v4771;	// L7402
          ap_int<66> v4773 = v4772 * 8;	// L7403
          ap_int<66> v4774 = v4773 - 1;	// L7404
          int v4775 = v4769;	// L7405
          int v4776 = v4774;	// L7406
          ap_int<128> v4777;
          ap_int<128> v4777_tmp = v4767;
          v4777_tmp(v4776, v4775) = v4766;
          v4777 = v4777_tmp;	// L7407
          packed_c78 = v4777;	// L7408
        } else {
          ap_int<128> v4778 = packed_tmp78;	// L7410
          int64_t v4779 = m78;	// L7411
          int64_t v4780 = v4779 * 8;	// L7412
          ap_int<34> v4781 = m78;	// L7413
          ap_int<34> v4782 = v4781 + 1;	// L7414
          ap_int<66> v4783 = v4782;	// L7415
          ap_int<66> v4784 = v4783 * 8;	// L7416
          ap_int<66> v4785 = v4784 - 1;	// L7417
          int v4786 = v4780;	// L7418
          int v4787 = v4785;	// L7419
          int8_t v4788;
          ap_int<128> v4788_tmp = v4778;
          v4788 = v4788_tmp(v4787, v4786);	// L7420
          ap_int<128> v4789 = packed_c78;	// L7421
          ap_int<128> v4790;
          ap_int<128> v4790_tmp = v4789;
          v4790_tmp(v4787, v4786) = v4788;
          v4790 = v4790_tmp;	// L7431
          packed_c78 = v4790;	// L7432
        }
      }
      ap_int<128> v4791 = packed_c78;	// L7435
      v4739.write(v4791);	// L7436
    }
  }
}

void gemm_5_16(
  hls::stream< int8_t >& v4792,
  hls::stream< int8_t >& v4793,
  hls::stream< int8_t >& v4794,
  hls::stream< ap_uint<128> >& v4795,
  hls::stream< ap_uint<128> >& v4796
) {	// L7441
  l_S_mt_nt_0_mt103: for (int mt103 = 0; mt103 < 4; mt103++) {	// L7449
    l_nt103: for (int nt103 = 0; nt103 < 4; nt103++) {	// L7450
      int8_t c79;	// L7451
      c79 = 0;	// L7452
      l_S_k_0_k103: for (int k103 = 0; k103 < 64; k103++) {	// L7453
        int8_t v4801 = v4792.read();	// L7454
        int8_t a84;	// L7455
        a84 = v4801;	// L7456
        int8_t v4803 = v4793.read();	// L7457
        int8_t b95;	// L7458
        b95 = v4803;	// L7459
        int8_t v4805 = a84;	// L7460
        int8_t v4806 = b95;	// L7461
        int16_t v4807 = v4805;	// L7462
        int16_t v4808 = v4806;	// L7463
        int16_t v4809 = v4807 * v4808;	// L7464
        int8_t v4810 = c79;	// L7465
        ap_int<17> v4811 = v4810;	// L7466
        ap_int<17> v4812 = v4809;	// L7467
        ap_int<17> v4813 = v4811 + v4812;	// L7468
        int8_t v4814 = v4813;	// L7469
        c79 = v4814;	// L7470
        int8_t v4815 = b95;	// L7471
        v4794.write(v4815);	// L7472
      }
      ap_uint<128> packed_tmp79;	// L7474
      ap_uint<128> v4817 = v4795.read();	// L7475
      packed_tmp79 = v4817;	// L7476
      ap_uint<128> packed_c79;	// L7477
      packed_c79 = 0;	// L7478
      l_S_m_1_m79: for (int m79 = 0; m79 < 16; m79++) {	// L7479
        ap_int<33> v4820 = m79;	// L7480
        bool v4821 = v4820 == 4;	// L7481
        if (v4821) {	// L7482
          int8_t v4822 = c79;	// L7483
          ap_int<128> v4823 = packed_c79;	// L7484
          int64_t v4824 = m79;	// L7485
          int64_t v4825 = v4824 * 8;	// L7486
          ap_int<34> v4826 = m79;	// L7487
          ap_int<34> v4827 = v4826 + 1;	// L7488
          ap_int<66> v4828 = v4827;	// L7489
          ap_int<66> v4829 = v4828 * 8;	// L7490
          ap_int<66> v4830 = v4829 - 1;	// L7491
          int v4831 = v4825;	// L7492
          int v4832 = v4830;	// L7493
          ap_int<128> v4833;
          ap_int<128> v4833_tmp = v4823;
          v4833_tmp(v4832, v4831) = v4822;
          v4833 = v4833_tmp;	// L7494
          packed_c79 = v4833;	// L7495
        } else {
          ap_int<128> v4834 = packed_tmp79;	// L7497
          int64_t v4835 = m79;	// L7498
          int64_t v4836 = v4835 * 8;	// L7499
          ap_int<34> v4837 = m79;	// L7500
          ap_int<34> v4838 = v4837 + 1;	// L7501
          ap_int<66> v4839 = v4838;	// L7502
          ap_int<66> v4840 = v4839 * 8;	// L7503
          ap_int<66> v4841 = v4840 - 1;	// L7504
          int v4842 = v4836;	// L7505
          int v4843 = v4841;	// L7506
          int8_t v4844;
          ap_int<128> v4844_tmp = v4834;
          v4844 = v4844_tmp(v4843, v4842);	// L7507
          ap_int<128> v4845 = packed_c79;	// L7508
          ap_int<128> v4846;
          ap_int<128> v4846_tmp = v4845;
          v4846_tmp(v4843, v4842) = v4844;
          v4846 = v4846_tmp;	// L7518
          packed_c79 = v4846;	// L7519
        }
      }
      ap_int<128> v4847 = packed_c79;	// L7522
      v4796.write(v4847);	// L7523
    }
  }
}

void gemm_6_0(
  hls::stream< ap_uint<128> >& v4848,
  hls::stream< int8_t >& v4849,
  hls::stream< ap_uint<128> >& v4850
) {	// L7528
  l_S_mt_nt_0_mt104: for (int mt104 = 0; mt104 < 4; mt104++) {	// L7531
    l_nt104: for (int nt104 = 0; nt104 < 4; nt104++) {	// L7532
      l_S_k_0_k104: for (int k104 = 0; k104 < 64; k104++) {	// L7533
        ap_uint<128> v4854 = v4848.read();	// L7534
        ap_uint<128> a85;	// L7535
        a85 = v4854;	// L7536
        ap_int<128> v4856 = a85;	// L7537
        int8_t v4857;
        ap_int<128> v4857_tmp = v4856;
        v4857 = v4857_tmp(47, 40);	// L7538
        v4849.write(v4857);	// L7539
        ap_int<128> v4858 = a85;	// L7540
        v4850.write(v4858);	// L7541
      }
    }
  }
}

void gemm_6_1(
  hls::stream< int8_t >& v4859,
  hls::stream< int8_t >& v4860,
  hls::stream< int8_t >& v4861,
  hls::stream< int8_t >& v4862,
  hls::stream< ap_uint<128> >& v4863,
  hls::stream< ap_uint<128> >& v4864
) {	// L7547
  l_S_mt_nt_0_mt105: for (int mt105 = 0; mt105 < 4; mt105++) {	// L7555
    l_nt105: for (int nt105 = 0; nt105 < 4; nt105++) {	// L7556
      int8_t c80;	// L7557
      c80 = 0;	// L7558
      l_S_k_0_k105: for (int k105 = 0; k105 < 64; k105++) {	// L7559
        int8_t v4869 = v4859.read();	// L7560
        int8_t a86;	// L7561
        a86 = v4869;	// L7562
        int8_t v4871 = v4860.read();	// L7563
        int8_t b96;	// L7564
        b96 = v4871;	// L7565
        int8_t v4873 = a86;	// L7566
        int8_t v4874 = b96;	// L7567
        int16_t v4875 = v4873;	// L7568
        int16_t v4876 = v4874;	// L7569
        int16_t v4877 = v4875 * v4876;	// L7570
        int8_t v4878 = c80;	// L7571
        ap_int<17> v4879 = v4878;	// L7572
        ap_int<17> v4880 = v4877;	// L7573
        ap_int<17> v4881 = v4879 + v4880;	// L7574
        int8_t v4882 = v4881;	// L7575
        c80 = v4882;	// L7576
        int8_t v4883 = a86;	// L7577
        v4861.write(v4883);	// L7578
        int8_t v4884 = b96;	// L7579
        v4862.write(v4884);	// L7580
      }
      ap_uint<128> packed_tmp80;	// L7582
      ap_uint<128> v4886 = v4863.read();	// L7583
      packed_tmp80 = v4886;	// L7584
      ap_uint<128> packed_c80;	// L7585
      packed_c80 = 0;	// L7586
      l_S_m_1_m80: for (int m80 = 0; m80 < 16; m80++) {	// L7587
        ap_int<33> v4889 = m80;	// L7588
        bool v4890 = v4889 == 5;	// L7589
        if (v4890) {	// L7590
          int8_t v4891 = c80;	// L7591
          ap_int<128> v4892 = packed_c80;	// L7592
          int64_t v4893 = m80;	// L7593
          int64_t v4894 = v4893 * 8;	// L7594
          ap_int<34> v4895 = m80;	// L7595
          ap_int<34> v4896 = v4895 + 1;	// L7596
          ap_int<66> v4897 = v4896;	// L7597
          ap_int<66> v4898 = v4897 * 8;	// L7598
          ap_int<66> v4899 = v4898 - 1;	// L7599
          int v4900 = v4894;	// L7600
          int v4901 = v4899;	// L7601
          ap_int<128> v4902;
          ap_int<128> v4902_tmp = v4892;
          v4902_tmp(v4901, v4900) = v4891;
          v4902 = v4902_tmp;	// L7602
          packed_c80 = v4902;	// L7603
        } else {
          ap_int<128> v4903 = packed_tmp80;	// L7605
          int64_t v4904 = m80;	// L7606
          int64_t v4905 = v4904 * 8;	// L7607
          ap_int<34> v4906 = m80;	// L7608
          ap_int<34> v4907 = v4906 + 1;	// L7609
          ap_int<66> v4908 = v4907;	// L7610
          ap_int<66> v4909 = v4908 * 8;	// L7611
          ap_int<66> v4910 = v4909 - 1;	// L7612
          int v4911 = v4905;	// L7613
          int v4912 = v4910;	// L7614
          int8_t v4913;
          ap_int<128> v4913_tmp = v4903;
          v4913 = v4913_tmp(v4912, v4911);	// L7615
          ap_int<128> v4914 = packed_c80;	// L7616
          ap_int<128> v4915;
          ap_int<128> v4915_tmp = v4914;
          v4915_tmp(v4912, v4911) = v4913;
          v4915 = v4915_tmp;	// L7626
          packed_c80 = v4915;	// L7627
        }
      }
      ap_int<128> v4916 = packed_c80;	// L7630
      v4864.write(v4916);	// L7631
    }
  }
}

void gemm_6_2(
  hls::stream< int8_t >& v4917,
  hls::stream< int8_t >& v4918,
  hls::stream< int8_t >& v4919,
  hls::stream< int8_t >& v4920,
  hls::stream< ap_uint<128> >& v4921,
  hls::stream< ap_uint<128> >& v4922
) {	// L7636
  l_S_mt_nt_0_mt106: for (int mt106 = 0; mt106 < 4; mt106++) {	// L7644
    l_nt106: for (int nt106 = 0; nt106 < 4; nt106++) {	// L7645
      int8_t c81;	// L7646
      c81 = 0;	// L7647
      l_S_k_0_k106: for (int k106 = 0; k106 < 64; k106++) {	// L7648
        int8_t v4927 = v4917.read();	// L7649
        int8_t a87;	// L7650
        a87 = v4927;	// L7651
        int8_t v4929 = v4918.read();	// L7652
        int8_t b97;	// L7653
        b97 = v4929;	// L7654
        int8_t v4931 = a87;	// L7655
        int8_t v4932 = b97;	// L7656
        int16_t v4933 = v4931;	// L7657
        int16_t v4934 = v4932;	// L7658
        int16_t v4935 = v4933 * v4934;	// L7659
        int8_t v4936 = c81;	// L7660
        ap_int<17> v4937 = v4936;	// L7661
        ap_int<17> v4938 = v4935;	// L7662
        ap_int<17> v4939 = v4937 + v4938;	// L7663
        int8_t v4940 = v4939;	// L7664
        c81 = v4940;	// L7665
        int8_t v4941 = a87;	// L7666
        v4919.write(v4941);	// L7667
        int8_t v4942 = b97;	// L7668
        v4920.write(v4942);	// L7669
      }
      ap_uint<128> packed_tmp81;	// L7671
      ap_uint<128> v4944 = v4921.read();	// L7672
      packed_tmp81 = v4944;	// L7673
      ap_uint<128> packed_c81;	// L7674
      packed_c81 = 0;	// L7675
      l_S_m_1_m81: for (int m81 = 0; m81 < 16; m81++) {	// L7676
        ap_int<33> v4947 = m81;	// L7677
        bool v4948 = v4947 == 5;	// L7678
        if (v4948) {	// L7679
          int8_t v4949 = c81;	// L7680
          ap_int<128> v4950 = packed_c81;	// L7681
          int64_t v4951 = m81;	// L7682
          int64_t v4952 = v4951 * 8;	// L7683
          ap_int<34> v4953 = m81;	// L7684
          ap_int<34> v4954 = v4953 + 1;	// L7685
          ap_int<66> v4955 = v4954;	// L7686
          ap_int<66> v4956 = v4955 * 8;	// L7687
          ap_int<66> v4957 = v4956 - 1;	// L7688
          int v4958 = v4952;	// L7689
          int v4959 = v4957;	// L7690
          ap_int<128> v4960;
          ap_int<128> v4960_tmp = v4950;
          v4960_tmp(v4959, v4958) = v4949;
          v4960 = v4960_tmp;	// L7691
          packed_c81 = v4960;	// L7692
        } else {
          ap_int<128> v4961 = packed_tmp81;	// L7694
          int64_t v4962 = m81;	// L7695
          int64_t v4963 = v4962 * 8;	// L7696
          ap_int<34> v4964 = m81;	// L7697
          ap_int<34> v4965 = v4964 + 1;	// L7698
          ap_int<66> v4966 = v4965;	// L7699
          ap_int<66> v4967 = v4966 * 8;	// L7700
          ap_int<66> v4968 = v4967 - 1;	// L7701
          int v4969 = v4963;	// L7702
          int v4970 = v4968;	// L7703
          int8_t v4971;
          ap_int<128> v4971_tmp = v4961;
          v4971 = v4971_tmp(v4970, v4969);	// L7704
          ap_int<128> v4972 = packed_c81;	// L7705
          ap_int<128> v4973;
          ap_int<128> v4973_tmp = v4972;
          v4973_tmp(v4970, v4969) = v4971;
          v4973 = v4973_tmp;	// L7715
          packed_c81 = v4973;	// L7716
        }
      }
      ap_int<128> v4974 = packed_c81;	// L7719
      v4922.write(v4974);	// L7720
    }
  }
}

void gemm_6_3(
  hls::stream< int8_t >& v4975,
  hls::stream< int8_t >& v4976,
  hls::stream< int8_t >& v4977,
  hls::stream< int8_t >& v4978,
  hls::stream< ap_uint<128> >& v4979,
  hls::stream< ap_uint<128> >& v4980
) {	// L7725
  l_S_mt_nt_0_mt107: for (int mt107 = 0; mt107 < 4; mt107++) {	// L7733
    l_nt107: for (int nt107 = 0; nt107 < 4; nt107++) {	// L7734
      int8_t c82;	// L7735
      c82 = 0;	// L7736
      l_S_k_0_k107: for (int k107 = 0; k107 < 64; k107++) {	// L7737
        int8_t v4985 = v4975.read();	// L7738
        int8_t a88;	// L7739
        a88 = v4985;	// L7740
        int8_t v4987 = v4976.read();	// L7741
        int8_t b98;	// L7742
        b98 = v4987;	// L7743
        int8_t v4989 = a88;	// L7744
        int8_t v4990 = b98;	// L7745
        int16_t v4991 = v4989;	// L7746
        int16_t v4992 = v4990;	// L7747
        int16_t v4993 = v4991 * v4992;	// L7748
        int8_t v4994 = c82;	// L7749
        ap_int<17> v4995 = v4994;	// L7750
        ap_int<17> v4996 = v4993;	// L7751
        ap_int<17> v4997 = v4995 + v4996;	// L7752
        int8_t v4998 = v4997;	// L7753
        c82 = v4998;	// L7754
        int8_t v4999 = a88;	// L7755
        v4977.write(v4999);	// L7756
        int8_t v5000 = b98;	// L7757
        v4978.write(v5000);	// L7758
      }
      ap_uint<128> packed_tmp82;	// L7760
      ap_uint<128> v5002 = v4979.read();	// L7761
      packed_tmp82 = v5002;	// L7762
      ap_uint<128> packed_c82;	// L7763
      packed_c82 = 0;	// L7764
      l_S_m_1_m82: for (int m82 = 0; m82 < 16; m82++) {	// L7765
        ap_int<33> v5005 = m82;	// L7766
        bool v5006 = v5005 == 5;	// L7767
        if (v5006) {	// L7768
          int8_t v5007 = c82;	// L7769
          ap_int<128> v5008 = packed_c82;	// L7770
          int64_t v5009 = m82;	// L7771
          int64_t v5010 = v5009 * 8;	// L7772
          ap_int<34> v5011 = m82;	// L7773
          ap_int<34> v5012 = v5011 + 1;	// L7774
          ap_int<66> v5013 = v5012;	// L7775
          ap_int<66> v5014 = v5013 * 8;	// L7776
          ap_int<66> v5015 = v5014 - 1;	// L7777
          int v5016 = v5010;	// L7778
          int v5017 = v5015;	// L7779
          ap_int<128> v5018;
          ap_int<128> v5018_tmp = v5008;
          v5018_tmp(v5017, v5016) = v5007;
          v5018 = v5018_tmp;	// L7780
          packed_c82 = v5018;	// L7781
        } else {
          ap_int<128> v5019 = packed_tmp82;	// L7783
          int64_t v5020 = m82;	// L7784
          int64_t v5021 = v5020 * 8;	// L7785
          ap_int<34> v5022 = m82;	// L7786
          ap_int<34> v5023 = v5022 + 1;	// L7787
          ap_int<66> v5024 = v5023;	// L7788
          ap_int<66> v5025 = v5024 * 8;	// L7789
          ap_int<66> v5026 = v5025 - 1;	// L7790
          int v5027 = v5021;	// L7791
          int v5028 = v5026;	// L7792
          int8_t v5029;
          ap_int<128> v5029_tmp = v5019;
          v5029 = v5029_tmp(v5028, v5027);	// L7793
          ap_int<128> v5030 = packed_c82;	// L7794
          ap_int<128> v5031;
          ap_int<128> v5031_tmp = v5030;
          v5031_tmp(v5028, v5027) = v5029;
          v5031 = v5031_tmp;	// L7804
          packed_c82 = v5031;	// L7805
        }
      }
      ap_int<128> v5032 = packed_c82;	// L7808
      v4980.write(v5032);	// L7809
    }
  }
}

void gemm_6_4(
  hls::stream< int8_t >& v5033,
  hls::stream< int8_t >& v5034,
  hls::stream< int8_t >& v5035,
  hls::stream< int8_t >& v5036,
  hls::stream< ap_uint<128> >& v5037,
  hls::stream< ap_uint<128> >& v5038
) {	// L7814
  l_S_mt_nt_0_mt108: for (int mt108 = 0; mt108 < 4; mt108++) {	// L7822
    l_nt108: for (int nt108 = 0; nt108 < 4; nt108++) {	// L7823
      int8_t c83;	// L7824
      c83 = 0;	// L7825
      l_S_k_0_k108: for (int k108 = 0; k108 < 64; k108++) {	// L7826
        int8_t v5043 = v5033.read();	// L7827
        int8_t a89;	// L7828
        a89 = v5043;	// L7829
        int8_t v5045 = v5034.read();	// L7830
        int8_t b99;	// L7831
        b99 = v5045;	// L7832
        int8_t v5047 = a89;	// L7833
        int8_t v5048 = b99;	// L7834
        int16_t v5049 = v5047;	// L7835
        int16_t v5050 = v5048;	// L7836
        int16_t v5051 = v5049 * v5050;	// L7837
        int8_t v5052 = c83;	// L7838
        ap_int<17> v5053 = v5052;	// L7839
        ap_int<17> v5054 = v5051;	// L7840
        ap_int<17> v5055 = v5053 + v5054;	// L7841
        int8_t v5056 = v5055;	// L7842
        c83 = v5056;	// L7843
        int8_t v5057 = a89;	// L7844
        v5035.write(v5057);	// L7845
        int8_t v5058 = b99;	// L7846
        v5036.write(v5058);	// L7847
      }
      ap_uint<128> packed_tmp83;	// L7849
      ap_uint<128> v5060 = v5037.read();	// L7850
      packed_tmp83 = v5060;	// L7851
      ap_uint<128> packed_c83;	// L7852
      packed_c83 = 0;	// L7853
      l_S_m_1_m83: for (int m83 = 0; m83 < 16; m83++) {	// L7854
        ap_int<33> v5063 = m83;	// L7855
        bool v5064 = v5063 == 5;	// L7856
        if (v5064) {	// L7857
          int8_t v5065 = c83;	// L7858
          ap_int<128> v5066 = packed_c83;	// L7859
          int64_t v5067 = m83;	// L7860
          int64_t v5068 = v5067 * 8;	// L7861
          ap_int<34> v5069 = m83;	// L7862
          ap_int<34> v5070 = v5069 + 1;	// L7863
          ap_int<66> v5071 = v5070;	// L7864
          ap_int<66> v5072 = v5071 * 8;	// L7865
          ap_int<66> v5073 = v5072 - 1;	// L7866
          int v5074 = v5068;	// L7867
          int v5075 = v5073;	// L7868
          ap_int<128> v5076;
          ap_int<128> v5076_tmp = v5066;
          v5076_tmp(v5075, v5074) = v5065;
          v5076 = v5076_tmp;	// L7869
          packed_c83 = v5076;	// L7870
        } else {
          ap_int<128> v5077 = packed_tmp83;	// L7872
          int64_t v5078 = m83;	// L7873
          int64_t v5079 = v5078 * 8;	// L7874
          ap_int<34> v5080 = m83;	// L7875
          ap_int<34> v5081 = v5080 + 1;	// L7876
          ap_int<66> v5082 = v5081;	// L7877
          ap_int<66> v5083 = v5082 * 8;	// L7878
          ap_int<66> v5084 = v5083 - 1;	// L7879
          int v5085 = v5079;	// L7880
          int v5086 = v5084;	// L7881
          int8_t v5087;
          ap_int<128> v5087_tmp = v5077;
          v5087 = v5087_tmp(v5086, v5085);	// L7882
          ap_int<128> v5088 = packed_c83;	// L7883
          ap_int<128> v5089;
          ap_int<128> v5089_tmp = v5088;
          v5089_tmp(v5086, v5085) = v5087;
          v5089 = v5089_tmp;	// L7893
          packed_c83 = v5089;	// L7894
        }
      }
      ap_int<128> v5090 = packed_c83;	// L7897
      v5038.write(v5090);	// L7898
    }
  }
}

void gemm_6_5(
  hls::stream< int8_t >& v5091,
  hls::stream< int8_t >& v5092,
  hls::stream< int8_t >& v5093,
  hls::stream< int8_t >& v5094,
  hls::stream< ap_uint<128> >& v5095,
  hls::stream< ap_uint<128> >& v5096
) {	// L7903
  l_S_mt_nt_0_mt109: for (int mt109 = 0; mt109 < 4; mt109++) {	// L7911
    l_nt109: for (int nt109 = 0; nt109 < 4; nt109++) {	// L7912
      int8_t c84;	// L7913
      c84 = 0;	// L7914
      l_S_k_0_k109: for (int k109 = 0; k109 < 64; k109++) {	// L7915
        int8_t v5101 = v5091.read();	// L7916
        int8_t a90;	// L7917
        a90 = v5101;	// L7918
        int8_t v5103 = v5092.read();	// L7919
        int8_t b100;	// L7920
        b100 = v5103;	// L7921
        int8_t v5105 = a90;	// L7922
        int8_t v5106 = b100;	// L7923
        int16_t v5107 = v5105;	// L7924
        int16_t v5108 = v5106;	// L7925
        int16_t v5109 = v5107 * v5108;	// L7926
        int8_t v5110 = c84;	// L7927
        ap_int<17> v5111 = v5110;	// L7928
        ap_int<17> v5112 = v5109;	// L7929
        ap_int<17> v5113 = v5111 + v5112;	// L7930
        int8_t v5114 = v5113;	// L7931
        c84 = v5114;	// L7932
        int8_t v5115 = a90;	// L7933
        v5093.write(v5115);	// L7934
        int8_t v5116 = b100;	// L7935
        v5094.write(v5116);	// L7936
      }
      ap_uint<128> packed_tmp84;	// L7938
      ap_uint<128> v5118 = v5095.read();	// L7939
      packed_tmp84 = v5118;	// L7940
      ap_uint<128> packed_c84;	// L7941
      packed_c84 = 0;	// L7942
      l_S_m_1_m84: for (int m84 = 0; m84 < 16; m84++) {	// L7943
        ap_int<33> v5121 = m84;	// L7944
        bool v5122 = v5121 == 5;	// L7945
        if (v5122) {	// L7946
          int8_t v5123 = c84;	// L7947
          ap_int<128> v5124 = packed_c84;	// L7948
          int64_t v5125 = m84;	// L7949
          int64_t v5126 = v5125 * 8;	// L7950
          ap_int<34> v5127 = m84;	// L7951
          ap_int<34> v5128 = v5127 + 1;	// L7952
          ap_int<66> v5129 = v5128;	// L7953
          ap_int<66> v5130 = v5129 * 8;	// L7954
          ap_int<66> v5131 = v5130 - 1;	// L7955
          int v5132 = v5126;	// L7956
          int v5133 = v5131;	// L7957
          ap_int<128> v5134;
          ap_int<128> v5134_tmp = v5124;
          v5134_tmp(v5133, v5132) = v5123;
          v5134 = v5134_tmp;	// L7958
          packed_c84 = v5134;	// L7959
        } else {
          ap_int<128> v5135 = packed_tmp84;	// L7961
          int64_t v5136 = m84;	// L7962
          int64_t v5137 = v5136 * 8;	// L7963
          ap_int<34> v5138 = m84;	// L7964
          ap_int<34> v5139 = v5138 + 1;	// L7965
          ap_int<66> v5140 = v5139;	// L7966
          ap_int<66> v5141 = v5140 * 8;	// L7967
          ap_int<66> v5142 = v5141 - 1;	// L7968
          int v5143 = v5137;	// L7969
          int v5144 = v5142;	// L7970
          int8_t v5145;
          ap_int<128> v5145_tmp = v5135;
          v5145 = v5145_tmp(v5144, v5143);	// L7971
          ap_int<128> v5146 = packed_c84;	// L7972
          ap_int<128> v5147;
          ap_int<128> v5147_tmp = v5146;
          v5147_tmp(v5144, v5143) = v5145;
          v5147 = v5147_tmp;	// L7982
          packed_c84 = v5147;	// L7983
        }
      }
      ap_int<128> v5148 = packed_c84;	// L7986
      v5096.write(v5148);	// L7987
    }
  }
}

void gemm_6_6(
  hls::stream< int8_t >& v5149,
  hls::stream< int8_t >& v5150,
  hls::stream< int8_t >& v5151,
  hls::stream< int8_t >& v5152,
  hls::stream< ap_uint<128> >& v5153,
  hls::stream< ap_uint<128> >& v5154
) {	// L7992
  l_S_mt_nt_0_mt110: for (int mt110 = 0; mt110 < 4; mt110++) {	// L8000
    l_nt110: for (int nt110 = 0; nt110 < 4; nt110++) {	// L8001
      int8_t c85;	// L8002
      c85 = 0;	// L8003
      l_S_k_0_k110: for (int k110 = 0; k110 < 64; k110++) {	// L8004
        int8_t v5159 = v5149.read();	// L8005
        int8_t a91;	// L8006
        a91 = v5159;	// L8007
        int8_t v5161 = v5150.read();	// L8008
        int8_t b101;	// L8009
        b101 = v5161;	// L8010
        int8_t v5163 = a91;	// L8011
        int8_t v5164 = b101;	// L8012
        int16_t v5165 = v5163;	// L8013
        int16_t v5166 = v5164;	// L8014
        int16_t v5167 = v5165 * v5166;	// L8015
        int8_t v5168 = c85;	// L8016
        ap_int<17> v5169 = v5168;	// L8017
        ap_int<17> v5170 = v5167;	// L8018
        ap_int<17> v5171 = v5169 + v5170;	// L8019
        int8_t v5172 = v5171;	// L8020
        c85 = v5172;	// L8021
        int8_t v5173 = a91;	// L8022
        v5151.write(v5173);	// L8023
        int8_t v5174 = b101;	// L8024
        v5152.write(v5174);	// L8025
      }
      ap_uint<128> packed_tmp85;	// L8027
      ap_uint<128> v5176 = v5153.read();	// L8028
      packed_tmp85 = v5176;	// L8029
      ap_uint<128> packed_c85;	// L8030
      packed_c85 = 0;	// L8031
      l_S_m_1_m85: for (int m85 = 0; m85 < 16; m85++) {	// L8032
        ap_int<33> v5179 = m85;	// L8033
        bool v5180 = v5179 == 5;	// L8034
        if (v5180) {	// L8035
          int8_t v5181 = c85;	// L8036
          ap_int<128> v5182 = packed_c85;	// L8037
          int64_t v5183 = m85;	// L8038
          int64_t v5184 = v5183 * 8;	// L8039
          ap_int<34> v5185 = m85;	// L8040
          ap_int<34> v5186 = v5185 + 1;	// L8041
          ap_int<66> v5187 = v5186;	// L8042
          ap_int<66> v5188 = v5187 * 8;	// L8043
          ap_int<66> v5189 = v5188 - 1;	// L8044
          int v5190 = v5184;	// L8045
          int v5191 = v5189;	// L8046
          ap_int<128> v5192;
          ap_int<128> v5192_tmp = v5182;
          v5192_tmp(v5191, v5190) = v5181;
          v5192 = v5192_tmp;	// L8047
          packed_c85 = v5192;	// L8048
        } else {
          ap_int<128> v5193 = packed_tmp85;	// L8050
          int64_t v5194 = m85;	// L8051
          int64_t v5195 = v5194 * 8;	// L8052
          ap_int<34> v5196 = m85;	// L8053
          ap_int<34> v5197 = v5196 + 1;	// L8054
          ap_int<66> v5198 = v5197;	// L8055
          ap_int<66> v5199 = v5198 * 8;	// L8056
          ap_int<66> v5200 = v5199 - 1;	// L8057
          int v5201 = v5195;	// L8058
          int v5202 = v5200;	// L8059
          int8_t v5203;
          ap_int<128> v5203_tmp = v5193;
          v5203 = v5203_tmp(v5202, v5201);	// L8060
          ap_int<128> v5204 = packed_c85;	// L8061
          ap_int<128> v5205;
          ap_int<128> v5205_tmp = v5204;
          v5205_tmp(v5202, v5201) = v5203;
          v5205 = v5205_tmp;	// L8071
          packed_c85 = v5205;	// L8072
        }
      }
      ap_int<128> v5206 = packed_c85;	// L8075
      v5154.write(v5206);	// L8076
    }
  }
}

void gemm_6_7(
  hls::stream< int8_t >& v5207,
  hls::stream< int8_t >& v5208,
  hls::stream< int8_t >& v5209,
  hls::stream< int8_t >& v5210,
  hls::stream< ap_uint<128> >& v5211,
  hls::stream< ap_uint<128> >& v5212
) {	// L8081
  l_S_mt_nt_0_mt111: for (int mt111 = 0; mt111 < 4; mt111++) {	// L8089
    l_nt111: for (int nt111 = 0; nt111 < 4; nt111++) {	// L8090
      int8_t c86;	// L8091
      c86 = 0;	// L8092
      l_S_k_0_k111: for (int k111 = 0; k111 < 64; k111++) {	// L8093
        int8_t v5217 = v5207.read();	// L8094
        int8_t a92;	// L8095
        a92 = v5217;	// L8096
        int8_t v5219 = v5208.read();	// L8097
        int8_t b102;	// L8098
        b102 = v5219;	// L8099
        int8_t v5221 = a92;	// L8100
        int8_t v5222 = b102;	// L8101
        int16_t v5223 = v5221;	// L8102
        int16_t v5224 = v5222;	// L8103
        int16_t v5225 = v5223 * v5224;	// L8104
        int8_t v5226 = c86;	// L8105
        ap_int<17> v5227 = v5226;	// L8106
        ap_int<17> v5228 = v5225;	// L8107
        ap_int<17> v5229 = v5227 + v5228;	// L8108
        int8_t v5230 = v5229;	// L8109
        c86 = v5230;	// L8110
        int8_t v5231 = a92;	// L8111
        v5209.write(v5231);	// L8112
        int8_t v5232 = b102;	// L8113
        v5210.write(v5232);	// L8114
      }
      ap_uint<128> packed_tmp86;	// L8116
      ap_uint<128> v5234 = v5211.read();	// L8117
      packed_tmp86 = v5234;	// L8118
      ap_uint<128> packed_c86;	// L8119
      packed_c86 = 0;	// L8120
      l_S_m_1_m86: for (int m86 = 0; m86 < 16; m86++) {	// L8121
        ap_int<33> v5237 = m86;	// L8122
        bool v5238 = v5237 == 5;	// L8123
        if (v5238) {	// L8124
          int8_t v5239 = c86;	// L8125
          ap_int<128> v5240 = packed_c86;	// L8126
          int64_t v5241 = m86;	// L8127
          int64_t v5242 = v5241 * 8;	// L8128
          ap_int<34> v5243 = m86;	// L8129
          ap_int<34> v5244 = v5243 + 1;	// L8130
          ap_int<66> v5245 = v5244;	// L8131
          ap_int<66> v5246 = v5245 * 8;	// L8132
          ap_int<66> v5247 = v5246 - 1;	// L8133
          int v5248 = v5242;	// L8134
          int v5249 = v5247;	// L8135
          ap_int<128> v5250;
          ap_int<128> v5250_tmp = v5240;
          v5250_tmp(v5249, v5248) = v5239;
          v5250 = v5250_tmp;	// L8136
          packed_c86 = v5250;	// L8137
        } else {
          ap_int<128> v5251 = packed_tmp86;	// L8139
          int64_t v5252 = m86;	// L8140
          int64_t v5253 = v5252 * 8;	// L8141
          ap_int<34> v5254 = m86;	// L8142
          ap_int<34> v5255 = v5254 + 1;	// L8143
          ap_int<66> v5256 = v5255;	// L8144
          ap_int<66> v5257 = v5256 * 8;	// L8145
          ap_int<66> v5258 = v5257 - 1;	// L8146
          int v5259 = v5253;	// L8147
          int v5260 = v5258;	// L8148
          int8_t v5261;
          ap_int<128> v5261_tmp = v5251;
          v5261 = v5261_tmp(v5260, v5259);	// L8149
          ap_int<128> v5262 = packed_c86;	// L8150
          ap_int<128> v5263;
          ap_int<128> v5263_tmp = v5262;
          v5263_tmp(v5260, v5259) = v5261;
          v5263 = v5263_tmp;	// L8160
          packed_c86 = v5263;	// L8161
        }
      }
      ap_int<128> v5264 = packed_c86;	// L8164
      v5212.write(v5264);	// L8165
    }
  }
}

void gemm_6_8(
  hls::stream< int8_t >& v5265,
  hls::stream< int8_t >& v5266,
  hls::stream< int8_t >& v5267,
  hls::stream< int8_t >& v5268,
  hls::stream< ap_uint<128> >& v5269,
  hls::stream< ap_uint<128> >& v5270
) {	// L8170
  l_S_mt_nt_0_mt112: for (int mt112 = 0; mt112 < 4; mt112++) {	// L8178
    l_nt112: for (int nt112 = 0; nt112 < 4; nt112++) {	// L8179
      int8_t c87;	// L8180
      c87 = 0;	// L8181
      l_S_k_0_k112: for (int k112 = 0; k112 < 64; k112++) {	// L8182
        int8_t v5275 = v5265.read();	// L8183
        int8_t a93;	// L8184
        a93 = v5275;	// L8185
        int8_t v5277 = v5266.read();	// L8186
        int8_t b103;	// L8187
        b103 = v5277;	// L8188
        int8_t v5279 = a93;	// L8189
        int8_t v5280 = b103;	// L8190
        int16_t v5281 = v5279;	// L8191
        int16_t v5282 = v5280;	// L8192
        int16_t v5283 = v5281 * v5282;	// L8193
        int8_t v5284 = c87;	// L8194
        ap_int<17> v5285 = v5284;	// L8195
        ap_int<17> v5286 = v5283;	// L8196
        ap_int<17> v5287 = v5285 + v5286;	// L8197
        int8_t v5288 = v5287;	// L8198
        c87 = v5288;	// L8199
        int8_t v5289 = a93;	// L8200
        v5267.write(v5289);	// L8201
        int8_t v5290 = b103;	// L8202
        v5268.write(v5290);	// L8203
      }
      ap_uint<128> packed_tmp87;	// L8205
      ap_uint<128> v5292 = v5269.read();	// L8206
      packed_tmp87 = v5292;	// L8207
      ap_uint<128> packed_c87;	// L8208
      packed_c87 = 0;	// L8209
      l_S_m_1_m87: for (int m87 = 0; m87 < 16; m87++) {	// L8210
        ap_int<33> v5295 = m87;	// L8211
        bool v5296 = v5295 == 5;	// L8212
        if (v5296) {	// L8213
          int8_t v5297 = c87;	// L8214
          ap_int<128> v5298 = packed_c87;	// L8215
          int64_t v5299 = m87;	// L8216
          int64_t v5300 = v5299 * 8;	// L8217
          ap_int<34> v5301 = m87;	// L8218
          ap_int<34> v5302 = v5301 + 1;	// L8219
          ap_int<66> v5303 = v5302;	// L8220
          ap_int<66> v5304 = v5303 * 8;	// L8221
          ap_int<66> v5305 = v5304 - 1;	// L8222
          int v5306 = v5300;	// L8223
          int v5307 = v5305;	// L8224
          ap_int<128> v5308;
          ap_int<128> v5308_tmp = v5298;
          v5308_tmp(v5307, v5306) = v5297;
          v5308 = v5308_tmp;	// L8225
          packed_c87 = v5308;	// L8226
        } else {
          ap_int<128> v5309 = packed_tmp87;	// L8228
          int64_t v5310 = m87;	// L8229
          int64_t v5311 = v5310 * 8;	// L8230
          ap_int<34> v5312 = m87;	// L8231
          ap_int<34> v5313 = v5312 + 1;	// L8232
          ap_int<66> v5314 = v5313;	// L8233
          ap_int<66> v5315 = v5314 * 8;	// L8234
          ap_int<66> v5316 = v5315 - 1;	// L8235
          int v5317 = v5311;	// L8236
          int v5318 = v5316;	// L8237
          int8_t v5319;
          ap_int<128> v5319_tmp = v5309;
          v5319 = v5319_tmp(v5318, v5317);	// L8238
          ap_int<128> v5320 = packed_c87;	// L8239
          ap_int<128> v5321;
          ap_int<128> v5321_tmp = v5320;
          v5321_tmp(v5318, v5317) = v5319;
          v5321 = v5321_tmp;	// L8249
          packed_c87 = v5321;	// L8250
        }
      }
      ap_int<128> v5322 = packed_c87;	// L8253
      v5270.write(v5322);	// L8254
    }
  }
}

void gemm_6_9(
  hls::stream< int8_t >& v5323,
  hls::stream< int8_t >& v5324,
  hls::stream< int8_t >& v5325,
  hls::stream< int8_t >& v5326,
  hls::stream< ap_uint<128> >& v5327,
  hls::stream< ap_uint<128> >& v5328
) {	// L8259
  l_S_mt_nt_0_mt113: for (int mt113 = 0; mt113 < 4; mt113++) {	// L8267
    l_nt113: for (int nt113 = 0; nt113 < 4; nt113++) {	// L8268
      int8_t c88;	// L8269
      c88 = 0;	// L8270
      l_S_k_0_k113: for (int k113 = 0; k113 < 64; k113++) {	// L8271
        int8_t v5333 = v5323.read();	// L8272
        int8_t a94;	// L8273
        a94 = v5333;	// L8274
        int8_t v5335 = v5324.read();	// L8275
        int8_t b104;	// L8276
        b104 = v5335;	// L8277
        int8_t v5337 = a94;	// L8278
        int8_t v5338 = b104;	// L8279
        int16_t v5339 = v5337;	// L8280
        int16_t v5340 = v5338;	// L8281
        int16_t v5341 = v5339 * v5340;	// L8282
        int8_t v5342 = c88;	// L8283
        ap_int<17> v5343 = v5342;	// L8284
        ap_int<17> v5344 = v5341;	// L8285
        ap_int<17> v5345 = v5343 + v5344;	// L8286
        int8_t v5346 = v5345;	// L8287
        c88 = v5346;	// L8288
        int8_t v5347 = a94;	// L8289
        v5325.write(v5347);	// L8290
        int8_t v5348 = b104;	// L8291
        v5326.write(v5348);	// L8292
      }
      ap_uint<128> packed_tmp88;	// L8294
      ap_uint<128> v5350 = v5327.read();	// L8295
      packed_tmp88 = v5350;	// L8296
      ap_uint<128> packed_c88;	// L8297
      packed_c88 = 0;	// L8298
      l_S_m_1_m88: for (int m88 = 0; m88 < 16; m88++) {	// L8299
        ap_int<33> v5353 = m88;	// L8300
        bool v5354 = v5353 == 5;	// L8301
        if (v5354) {	// L8302
          int8_t v5355 = c88;	// L8303
          ap_int<128> v5356 = packed_c88;	// L8304
          int64_t v5357 = m88;	// L8305
          int64_t v5358 = v5357 * 8;	// L8306
          ap_int<34> v5359 = m88;	// L8307
          ap_int<34> v5360 = v5359 + 1;	// L8308
          ap_int<66> v5361 = v5360;	// L8309
          ap_int<66> v5362 = v5361 * 8;	// L8310
          ap_int<66> v5363 = v5362 - 1;	// L8311
          int v5364 = v5358;	// L8312
          int v5365 = v5363;	// L8313
          ap_int<128> v5366;
          ap_int<128> v5366_tmp = v5356;
          v5366_tmp(v5365, v5364) = v5355;
          v5366 = v5366_tmp;	// L8314
          packed_c88 = v5366;	// L8315
        } else {
          ap_int<128> v5367 = packed_tmp88;	// L8317
          int64_t v5368 = m88;	// L8318
          int64_t v5369 = v5368 * 8;	// L8319
          ap_int<34> v5370 = m88;	// L8320
          ap_int<34> v5371 = v5370 + 1;	// L8321
          ap_int<66> v5372 = v5371;	// L8322
          ap_int<66> v5373 = v5372 * 8;	// L8323
          ap_int<66> v5374 = v5373 - 1;	// L8324
          int v5375 = v5369;	// L8325
          int v5376 = v5374;	// L8326
          int8_t v5377;
          ap_int<128> v5377_tmp = v5367;
          v5377 = v5377_tmp(v5376, v5375);	// L8327
          ap_int<128> v5378 = packed_c88;	// L8328
          ap_int<128> v5379;
          ap_int<128> v5379_tmp = v5378;
          v5379_tmp(v5376, v5375) = v5377;
          v5379 = v5379_tmp;	// L8338
          packed_c88 = v5379;	// L8339
        }
      }
      ap_int<128> v5380 = packed_c88;	// L8342
      v5328.write(v5380);	// L8343
    }
  }
}

void gemm_6_10(
  hls::stream< int8_t >& v5381,
  hls::stream< int8_t >& v5382,
  hls::stream< int8_t >& v5383,
  hls::stream< int8_t >& v5384,
  hls::stream< ap_uint<128> >& v5385,
  hls::stream< ap_uint<128> >& v5386
) {	// L8348
  l_S_mt_nt_0_mt114: for (int mt114 = 0; mt114 < 4; mt114++) {	// L8356
    l_nt114: for (int nt114 = 0; nt114 < 4; nt114++) {	// L8357
      int8_t c89;	// L8358
      c89 = 0;	// L8359
      l_S_k_0_k114: for (int k114 = 0; k114 < 64; k114++) {	// L8360
        int8_t v5391 = v5381.read();	// L8361
        int8_t a95;	// L8362
        a95 = v5391;	// L8363
        int8_t v5393 = v5382.read();	// L8364
        int8_t b105;	// L8365
        b105 = v5393;	// L8366
        int8_t v5395 = a95;	// L8367
        int8_t v5396 = b105;	// L8368
        int16_t v5397 = v5395;	// L8369
        int16_t v5398 = v5396;	// L8370
        int16_t v5399 = v5397 * v5398;	// L8371
        int8_t v5400 = c89;	// L8372
        ap_int<17> v5401 = v5400;	// L8373
        ap_int<17> v5402 = v5399;	// L8374
        ap_int<17> v5403 = v5401 + v5402;	// L8375
        int8_t v5404 = v5403;	// L8376
        c89 = v5404;	// L8377
        int8_t v5405 = a95;	// L8378
        v5383.write(v5405);	// L8379
        int8_t v5406 = b105;	// L8380
        v5384.write(v5406);	// L8381
      }
      ap_uint<128> packed_tmp89;	// L8383
      ap_uint<128> v5408 = v5385.read();	// L8384
      packed_tmp89 = v5408;	// L8385
      ap_uint<128> packed_c89;	// L8386
      packed_c89 = 0;	// L8387
      l_S_m_1_m89: for (int m89 = 0; m89 < 16; m89++) {	// L8388
        ap_int<33> v5411 = m89;	// L8389
        bool v5412 = v5411 == 5;	// L8390
        if (v5412) {	// L8391
          int8_t v5413 = c89;	// L8392
          ap_int<128> v5414 = packed_c89;	// L8393
          int64_t v5415 = m89;	// L8394
          int64_t v5416 = v5415 * 8;	// L8395
          ap_int<34> v5417 = m89;	// L8396
          ap_int<34> v5418 = v5417 + 1;	// L8397
          ap_int<66> v5419 = v5418;	// L8398
          ap_int<66> v5420 = v5419 * 8;	// L8399
          ap_int<66> v5421 = v5420 - 1;	// L8400
          int v5422 = v5416;	// L8401
          int v5423 = v5421;	// L8402
          ap_int<128> v5424;
          ap_int<128> v5424_tmp = v5414;
          v5424_tmp(v5423, v5422) = v5413;
          v5424 = v5424_tmp;	// L8403
          packed_c89 = v5424;	// L8404
        } else {
          ap_int<128> v5425 = packed_tmp89;	// L8406
          int64_t v5426 = m89;	// L8407
          int64_t v5427 = v5426 * 8;	// L8408
          ap_int<34> v5428 = m89;	// L8409
          ap_int<34> v5429 = v5428 + 1;	// L8410
          ap_int<66> v5430 = v5429;	// L8411
          ap_int<66> v5431 = v5430 * 8;	// L8412
          ap_int<66> v5432 = v5431 - 1;	// L8413
          int v5433 = v5427;	// L8414
          int v5434 = v5432;	// L8415
          int8_t v5435;
          ap_int<128> v5435_tmp = v5425;
          v5435 = v5435_tmp(v5434, v5433);	// L8416
          ap_int<128> v5436 = packed_c89;	// L8417
          ap_int<128> v5437;
          ap_int<128> v5437_tmp = v5436;
          v5437_tmp(v5434, v5433) = v5435;
          v5437 = v5437_tmp;	// L8427
          packed_c89 = v5437;	// L8428
        }
      }
      ap_int<128> v5438 = packed_c89;	// L8431
      v5386.write(v5438);	// L8432
    }
  }
}

void gemm_6_11(
  hls::stream< int8_t >& v5439,
  hls::stream< int8_t >& v5440,
  hls::stream< int8_t >& v5441,
  hls::stream< int8_t >& v5442,
  hls::stream< ap_uint<128> >& v5443,
  hls::stream< ap_uint<128> >& v5444
) {	// L8437
  l_S_mt_nt_0_mt115: for (int mt115 = 0; mt115 < 4; mt115++) {	// L8445
    l_nt115: for (int nt115 = 0; nt115 < 4; nt115++) {	// L8446
      int8_t c90;	// L8447
      c90 = 0;	// L8448
      l_S_k_0_k115: for (int k115 = 0; k115 < 64; k115++) {	// L8449
        int8_t v5449 = v5439.read();	// L8450
        int8_t a96;	// L8451
        a96 = v5449;	// L8452
        int8_t v5451 = v5440.read();	// L8453
        int8_t b106;	// L8454
        b106 = v5451;	// L8455
        int8_t v5453 = a96;	// L8456
        int8_t v5454 = b106;	// L8457
        int16_t v5455 = v5453;	// L8458
        int16_t v5456 = v5454;	// L8459
        int16_t v5457 = v5455 * v5456;	// L8460
        int8_t v5458 = c90;	// L8461
        ap_int<17> v5459 = v5458;	// L8462
        ap_int<17> v5460 = v5457;	// L8463
        ap_int<17> v5461 = v5459 + v5460;	// L8464
        int8_t v5462 = v5461;	// L8465
        c90 = v5462;	// L8466
        int8_t v5463 = a96;	// L8467
        v5441.write(v5463);	// L8468
        int8_t v5464 = b106;	// L8469
        v5442.write(v5464);	// L8470
      }
      ap_uint<128> packed_tmp90;	// L8472
      ap_uint<128> v5466 = v5443.read();	// L8473
      packed_tmp90 = v5466;	// L8474
      ap_uint<128> packed_c90;	// L8475
      packed_c90 = 0;	// L8476
      l_S_m_1_m90: for (int m90 = 0; m90 < 16; m90++) {	// L8477
        ap_int<33> v5469 = m90;	// L8478
        bool v5470 = v5469 == 5;	// L8479
        if (v5470) {	// L8480
          int8_t v5471 = c90;	// L8481
          ap_int<128> v5472 = packed_c90;	// L8482
          int64_t v5473 = m90;	// L8483
          int64_t v5474 = v5473 * 8;	// L8484
          ap_int<34> v5475 = m90;	// L8485
          ap_int<34> v5476 = v5475 + 1;	// L8486
          ap_int<66> v5477 = v5476;	// L8487
          ap_int<66> v5478 = v5477 * 8;	// L8488
          ap_int<66> v5479 = v5478 - 1;	// L8489
          int v5480 = v5474;	// L8490
          int v5481 = v5479;	// L8491
          ap_int<128> v5482;
          ap_int<128> v5482_tmp = v5472;
          v5482_tmp(v5481, v5480) = v5471;
          v5482 = v5482_tmp;	// L8492
          packed_c90 = v5482;	// L8493
        } else {
          ap_int<128> v5483 = packed_tmp90;	// L8495
          int64_t v5484 = m90;	// L8496
          int64_t v5485 = v5484 * 8;	// L8497
          ap_int<34> v5486 = m90;	// L8498
          ap_int<34> v5487 = v5486 + 1;	// L8499
          ap_int<66> v5488 = v5487;	// L8500
          ap_int<66> v5489 = v5488 * 8;	// L8501
          ap_int<66> v5490 = v5489 - 1;	// L8502
          int v5491 = v5485;	// L8503
          int v5492 = v5490;	// L8504
          int8_t v5493;
          ap_int<128> v5493_tmp = v5483;
          v5493 = v5493_tmp(v5492, v5491);	// L8505
          ap_int<128> v5494 = packed_c90;	// L8506
          ap_int<128> v5495;
          ap_int<128> v5495_tmp = v5494;
          v5495_tmp(v5492, v5491) = v5493;
          v5495 = v5495_tmp;	// L8516
          packed_c90 = v5495;	// L8517
        }
      }
      ap_int<128> v5496 = packed_c90;	// L8520
      v5444.write(v5496);	// L8521
    }
  }
}

void gemm_6_12(
  hls::stream< int8_t >& v5497,
  hls::stream< int8_t >& v5498,
  hls::stream< int8_t >& v5499,
  hls::stream< int8_t >& v5500,
  hls::stream< ap_uint<128> >& v5501,
  hls::stream< ap_uint<128> >& v5502
) {	// L8526
  l_S_mt_nt_0_mt116: for (int mt116 = 0; mt116 < 4; mt116++) {	// L8534
    l_nt116: for (int nt116 = 0; nt116 < 4; nt116++) {	// L8535
      int8_t c91;	// L8536
      c91 = 0;	// L8537
      l_S_k_0_k116: for (int k116 = 0; k116 < 64; k116++) {	// L8538
        int8_t v5507 = v5497.read();	// L8539
        int8_t a97;	// L8540
        a97 = v5507;	// L8541
        int8_t v5509 = v5498.read();	// L8542
        int8_t b107;	// L8543
        b107 = v5509;	// L8544
        int8_t v5511 = a97;	// L8545
        int8_t v5512 = b107;	// L8546
        int16_t v5513 = v5511;	// L8547
        int16_t v5514 = v5512;	// L8548
        int16_t v5515 = v5513 * v5514;	// L8549
        int8_t v5516 = c91;	// L8550
        ap_int<17> v5517 = v5516;	// L8551
        ap_int<17> v5518 = v5515;	// L8552
        ap_int<17> v5519 = v5517 + v5518;	// L8553
        int8_t v5520 = v5519;	// L8554
        c91 = v5520;	// L8555
        int8_t v5521 = a97;	// L8556
        v5499.write(v5521);	// L8557
        int8_t v5522 = b107;	// L8558
        v5500.write(v5522);	// L8559
      }
      ap_uint<128> packed_tmp91;	// L8561
      ap_uint<128> v5524 = v5501.read();	// L8562
      packed_tmp91 = v5524;	// L8563
      ap_uint<128> packed_c91;	// L8564
      packed_c91 = 0;	// L8565
      l_S_m_1_m91: for (int m91 = 0; m91 < 16; m91++) {	// L8566
        ap_int<33> v5527 = m91;	// L8567
        bool v5528 = v5527 == 5;	// L8568
        if (v5528) {	// L8569
          int8_t v5529 = c91;	// L8570
          ap_int<128> v5530 = packed_c91;	// L8571
          int64_t v5531 = m91;	// L8572
          int64_t v5532 = v5531 * 8;	// L8573
          ap_int<34> v5533 = m91;	// L8574
          ap_int<34> v5534 = v5533 + 1;	// L8575
          ap_int<66> v5535 = v5534;	// L8576
          ap_int<66> v5536 = v5535 * 8;	// L8577
          ap_int<66> v5537 = v5536 - 1;	// L8578
          int v5538 = v5532;	// L8579
          int v5539 = v5537;	// L8580
          ap_int<128> v5540;
          ap_int<128> v5540_tmp = v5530;
          v5540_tmp(v5539, v5538) = v5529;
          v5540 = v5540_tmp;	// L8581
          packed_c91 = v5540;	// L8582
        } else {
          ap_int<128> v5541 = packed_tmp91;	// L8584
          int64_t v5542 = m91;	// L8585
          int64_t v5543 = v5542 * 8;	// L8586
          ap_int<34> v5544 = m91;	// L8587
          ap_int<34> v5545 = v5544 + 1;	// L8588
          ap_int<66> v5546 = v5545;	// L8589
          ap_int<66> v5547 = v5546 * 8;	// L8590
          ap_int<66> v5548 = v5547 - 1;	// L8591
          int v5549 = v5543;	// L8592
          int v5550 = v5548;	// L8593
          int8_t v5551;
          ap_int<128> v5551_tmp = v5541;
          v5551 = v5551_tmp(v5550, v5549);	// L8594
          ap_int<128> v5552 = packed_c91;	// L8595
          ap_int<128> v5553;
          ap_int<128> v5553_tmp = v5552;
          v5553_tmp(v5550, v5549) = v5551;
          v5553 = v5553_tmp;	// L8605
          packed_c91 = v5553;	// L8606
        }
      }
      ap_int<128> v5554 = packed_c91;	// L8609
      v5502.write(v5554);	// L8610
    }
  }
}

void gemm_6_13(
  hls::stream< int8_t >& v5555,
  hls::stream< int8_t >& v5556,
  hls::stream< int8_t >& v5557,
  hls::stream< int8_t >& v5558,
  hls::stream< ap_uint<128> >& v5559,
  hls::stream< ap_uint<128> >& v5560
) {	// L8615
  l_S_mt_nt_0_mt117: for (int mt117 = 0; mt117 < 4; mt117++) {	// L8623
    l_nt117: for (int nt117 = 0; nt117 < 4; nt117++) {	// L8624
      int8_t c92;	// L8625
      c92 = 0;	// L8626
      l_S_k_0_k117: for (int k117 = 0; k117 < 64; k117++) {	// L8627
        int8_t v5565 = v5555.read();	// L8628
        int8_t a98;	// L8629
        a98 = v5565;	// L8630
        int8_t v5567 = v5556.read();	// L8631
        int8_t b108;	// L8632
        b108 = v5567;	// L8633
        int8_t v5569 = a98;	// L8634
        int8_t v5570 = b108;	// L8635
        int16_t v5571 = v5569;	// L8636
        int16_t v5572 = v5570;	// L8637
        int16_t v5573 = v5571 * v5572;	// L8638
        int8_t v5574 = c92;	// L8639
        ap_int<17> v5575 = v5574;	// L8640
        ap_int<17> v5576 = v5573;	// L8641
        ap_int<17> v5577 = v5575 + v5576;	// L8642
        int8_t v5578 = v5577;	// L8643
        c92 = v5578;	// L8644
        int8_t v5579 = a98;	// L8645
        v5557.write(v5579);	// L8646
        int8_t v5580 = b108;	// L8647
        v5558.write(v5580);	// L8648
      }
      ap_uint<128> packed_tmp92;	// L8650
      ap_uint<128> v5582 = v5559.read();	// L8651
      packed_tmp92 = v5582;	// L8652
      ap_uint<128> packed_c92;	// L8653
      packed_c92 = 0;	// L8654
      l_S_m_1_m92: for (int m92 = 0; m92 < 16; m92++) {	// L8655
        ap_int<33> v5585 = m92;	// L8656
        bool v5586 = v5585 == 5;	// L8657
        if (v5586) {	// L8658
          int8_t v5587 = c92;	// L8659
          ap_int<128> v5588 = packed_c92;	// L8660
          int64_t v5589 = m92;	// L8661
          int64_t v5590 = v5589 * 8;	// L8662
          ap_int<34> v5591 = m92;	// L8663
          ap_int<34> v5592 = v5591 + 1;	// L8664
          ap_int<66> v5593 = v5592;	// L8665
          ap_int<66> v5594 = v5593 * 8;	// L8666
          ap_int<66> v5595 = v5594 - 1;	// L8667
          int v5596 = v5590;	// L8668
          int v5597 = v5595;	// L8669
          ap_int<128> v5598;
          ap_int<128> v5598_tmp = v5588;
          v5598_tmp(v5597, v5596) = v5587;
          v5598 = v5598_tmp;	// L8670
          packed_c92 = v5598;	// L8671
        } else {
          ap_int<128> v5599 = packed_tmp92;	// L8673
          int64_t v5600 = m92;	// L8674
          int64_t v5601 = v5600 * 8;	// L8675
          ap_int<34> v5602 = m92;	// L8676
          ap_int<34> v5603 = v5602 + 1;	// L8677
          ap_int<66> v5604 = v5603;	// L8678
          ap_int<66> v5605 = v5604 * 8;	// L8679
          ap_int<66> v5606 = v5605 - 1;	// L8680
          int v5607 = v5601;	// L8681
          int v5608 = v5606;	// L8682
          int8_t v5609;
          ap_int<128> v5609_tmp = v5599;
          v5609 = v5609_tmp(v5608, v5607);	// L8683
          ap_int<128> v5610 = packed_c92;	// L8684
          ap_int<128> v5611;
          ap_int<128> v5611_tmp = v5610;
          v5611_tmp(v5608, v5607) = v5609;
          v5611 = v5611_tmp;	// L8694
          packed_c92 = v5611;	// L8695
        }
      }
      ap_int<128> v5612 = packed_c92;	// L8698
      v5560.write(v5612);	// L8699
    }
  }
}

void gemm_6_14(
  hls::stream< int8_t >& v5613,
  hls::stream< int8_t >& v5614,
  hls::stream< int8_t >& v5615,
  hls::stream< int8_t >& v5616,
  hls::stream< ap_uint<128> >& v5617,
  hls::stream< ap_uint<128> >& v5618
) {	// L8704
  l_S_mt_nt_0_mt118: for (int mt118 = 0; mt118 < 4; mt118++) {	// L8712
    l_nt118: for (int nt118 = 0; nt118 < 4; nt118++) {	// L8713
      int8_t c93;	// L8714
      c93 = 0;	// L8715
      l_S_k_0_k118: for (int k118 = 0; k118 < 64; k118++) {	// L8716
        int8_t v5623 = v5613.read();	// L8717
        int8_t a99;	// L8718
        a99 = v5623;	// L8719
        int8_t v5625 = v5614.read();	// L8720
        int8_t b109;	// L8721
        b109 = v5625;	// L8722
        int8_t v5627 = a99;	// L8723
        int8_t v5628 = b109;	// L8724
        int16_t v5629 = v5627;	// L8725
        int16_t v5630 = v5628;	// L8726
        int16_t v5631 = v5629 * v5630;	// L8727
        int8_t v5632 = c93;	// L8728
        ap_int<17> v5633 = v5632;	// L8729
        ap_int<17> v5634 = v5631;	// L8730
        ap_int<17> v5635 = v5633 + v5634;	// L8731
        int8_t v5636 = v5635;	// L8732
        c93 = v5636;	// L8733
        int8_t v5637 = a99;	// L8734
        v5615.write(v5637);	// L8735
        int8_t v5638 = b109;	// L8736
        v5616.write(v5638);	// L8737
      }
      ap_uint<128> packed_tmp93;	// L8739
      ap_uint<128> v5640 = v5617.read();	// L8740
      packed_tmp93 = v5640;	// L8741
      ap_uint<128> packed_c93;	// L8742
      packed_c93 = 0;	// L8743
      l_S_m_1_m93: for (int m93 = 0; m93 < 16; m93++) {	// L8744
        ap_int<33> v5643 = m93;	// L8745
        bool v5644 = v5643 == 5;	// L8746
        if (v5644) {	// L8747
          int8_t v5645 = c93;	// L8748
          ap_int<128> v5646 = packed_c93;	// L8749
          int64_t v5647 = m93;	// L8750
          int64_t v5648 = v5647 * 8;	// L8751
          ap_int<34> v5649 = m93;	// L8752
          ap_int<34> v5650 = v5649 + 1;	// L8753
          ap_int<66> v5651 = v5650;	// L8754
          ap_int<66> v5652 = v5651 * 8;	// L8755
          ap_int<66> v5653 = v5652 - 1;	// L8756
          int v5654 = v5648;	// L8757
          int v5655 = v5653;	// L8758
          ap_int<128> v5656;
          ap_int<128> v5656_tmp = v5646;
          v5656_tmp(v5655, v5654) = v5645;
          v5656 = v5656_tmp;	// L8759
          packed_c93 = v5656;	// L8760
        } else {
          ap_int<128> v5657 = packed_tmp93;	// L8762
          int64_t v5658 = m93;	// L8763
          int64_t v5659 = v5658 * 8;	// L8764
          ap_int<34> v5660 = m93;	// L8765
          ap_int<34> v5661 = v5660 + 1;	// L8766
          ap_int<66> v5662 = v5661;	// L8767
          ap_int<66> v5663 = v5662 * 8;	// L8768
          ap_int<66> v5664 = v5663 - 1;	// L8769
          int v5665 = v5659;	// L8770
          int v5666 = v5664;	// L8771
          int8_t v5667;
          ap_int<128> v5667_tmp = v5657;
          v5667 = v5667_tmp(v5666, v5665);	// L8772
          ap_int<128> v5668 = packed_c93;	// L8773
          ap_int<128> v5669;
          ap_int<128> v5669_tmp = v5668;
          v5669_tmp(v5666, v5665) = v5667;
          v5669 = v5669_tmp;	// L8783
          packed_c93 = v5669;	// L8784
        }
      }
      ap_int<128> v5670 = packed_c93;	// L8787
      v5618.write(v5670);	// L8788
    }
  }
}

void gemm_6_15(
  hls::stream< int8_t >& v5671,
  hls::stream< int8_t >& v5672,
  hls::stream< int8_t >& v5673,
  hls::stream< int8_t >& v5674,
  hls::stream< ap_uint<128> >& v5675,
  hls::stream< ap_uint<128> >& v5676
) {	// L8793
  l_S_mt_nt_0_mt119: for (int mt119 = 0; mt119 < 4; mt119++) {	// L8801
    l_nt119: for (int nt119 = 0; nt119 < 4; nt119++) {	// L8802
      int8_t c94;	// L8803
      c94 = 0;	// L8804
      l_S_k_0_k119: for (int k119 = 0; k119 < 64; k119++) {	// L8805
        int8_t v5681 = v5671.read();	// L8806
        int8_t a100;	// L8807
        a100 = v5681;	// L8808
        int8_t v5683 = v5672.read();	// L8809
        int8_t b110;	// L8810
        b110 = v5683;	// L8811
        int8_t v5685 = a100;	// L8812
        int8_t v5686 = b110;	// L8813
        int16_t v5687 = v5685;	// L8814
        int16_t v5688 = v5686;	// L8815
        int16_t v5689 = v5687 * v5688;	// L8816
        int8_t v5690 = c94;	// L8817
        ap_int<17> v5691 = v5690;	// L8818
        ap_int<17> v5692 = v5689;	// L8819
        ap_int<17> v5693 = v5691 + v5692;	// L8820
        int8_t v5694 = v5693;	// L8821
        c94 = v5694;	// L8822
        int8_t v5695 = a100;	// L8823
        v5673.write(v5695);	// L8824
        int8_t v5696 = b110;	// L8825
        v5674.write(v5696);	// L8826
      }
      ap_uint<128> packed_tmp94;	// L8828
      ap_uint<128> v5698 = v5675.read();	// L8829
      packed_tmp94 = v5698;	// L8830
      ap_uint<128> packed_c94;	// L8831
      packed_c94 = 0;	// L8832
      l_S_m_1_m94: for (int m94 = 0; m94 < 16; m94++) {	// L8833
        ap_int<33> v5701 = m94;	// L8834
        bool v5702 = v5701 == 5;	// L8835
        if (v5702) {	// L8836
          int8_t v5703 = c94;	// L8837
          ap_int<128> v5704 = packed_c94;	// L8838
          int64_t v5705 = m94;	// L8839
          int64_t v5706 = v5705 * 8;	// L8840
          ap_int<34> v5707 = m94;	// L8841
          ap_int<34> v5708 = v5707 + 1;	// L8842
          ap_int<66> v5709 = v5708;	// L8843
          ap_int<66> v5710 = v5709 * 8;	// L8844
          ap_int<66> v5711 = v5710 - 1;	// L8845
          int v5712 = v5706;	// L8846
          int v5713 = v5711;	// L8847
          ap_int<128> v5714;
          ap_int<128> v5714_tmp = v5704;
          v5714_tmp(v5713, v5712) = v5703;
          v5714 = v5714_tmp;	// L8848
          packed_c94 = v5714;	// L8849
        } else {
          ap_int<128> v5715 = packed_tmp94;	// L8851
          int64_t v5716 = m94;	// L8852
          int64_t v5717 = v5716 * 8;	// L8853
          ap_int<34> v5718 = m94;	// L8854
          ap_int<34> v5719 = v5718 + 1;	// L8855
          ap_int<66> v5720 = v5719;	// L8856
          ap_int<66> v5721 = v5720 * 8;	// L8857
          ap_int<66> v5722 = v5721 - 1;	// L8858
          int v5723 = v5717;	// L8859
          int v5724 = v5722;	// L8860
          int8_t v5725;
          ap_int<128> v5725_tmp = v5715;
          v5725 = v5725_tmp(v5724, v5723);	// L8861
          ap_int<128> v5726 = packed_c94;	// L8862
          ap_int<128> v5727;
          ap_int<128> v5727_tmp = v5726;
          v5727_tmp(v5724, v5723) = v5725;
          v5727 = v5727_tmp;	// L8872
          packed_c94 = v5727;	// L8873
        }
      }
      ap_int<128> v5728 = packed_c94;	// L8876
      v5676.write(v5728);	// L8877
    }
  }
}

void gemm_6_16(
  hls::stream< int8_t >& v5729,
  hls::stream< int8_t >& v5730,
  hls::stream< int8_t >& v5731,
  hls::stream< ap_uint<128> >& v5732,
  hls::stream< ap_uint<128> >& v5733
) {	// L8882
  l_S_mt_nt_0_mt120: for (int mt120 = 0; mt120 < 4; mt120++) {	// L8890
    l_nt120: for (int nt120 = 0; nt120 < 4; nt120++) {	// L8891
      int8_t c95;	// L8892
      c95 = 0;	// L8893
      l_S_k_0_k120: for (int k120 = 0; k120 < 64; k120++) {	// L8894
        int8_t v5738 = v5729.read();	// L8895
        int8_t a101;	// L8896
        a101 = v5738;	// L8897
        int8_t v5740 = v5730.read();	// L8898
        int8_t b111;	// L8899
        b111 = v5740;	// L8900
        int8_t v5742 = a101;	// L8901
        int8_t v5743 = b111;	// L8902
        int16_t v5744 = v5742;	// L8903
        int16_t v5745 = v5743;	// L8904
        int16_t v5746 = v5744 * v5745;	// L8905
        int8_t v5747 = c95;	// L8906
        ap_int<17> v5748 = v5747;	// L8907
        ap_int<17> v5749 = v5746;	// L8908
        ap_int<17> v5750 = v5748 + v5749;	// L8909
        int8_t v5751 = v5750;	// L8910
        c95 = v5751;	// L8911
        int8_t v5752 = b111;	// L8912
        v5731.write(v5752);	// L8913
      }
      ap_uint<128> packed_tmp95;	// L8915
      ap_uint<128> v5754 = v5732.read();	// L8916
      packed_tmp95 = v5754;	// L8917
      ap_uint<128> packed_c95;	// L8918
      packed_c95 = 0;	// L8919
      l_S_m_1_m95: for (int m95 = 0; m95 < 16; m95++) {	// L8920
        ap_int<33> v5757 = m95;	// L8921
        bool v5758 = v5757 == 5;	// L8922
        if (v5758) {	// L8923
          int8_t v5759 = c95;	// L8924
          ap_int<128> v5760 = packed_c95;	// L8925
          int64_t v5761 = m95;	// L8926
          int64_t v5762 = v5761 * 8;	// L8927
          ap_int<34> v5763 = m95;	// L8928
          ap_int<34> v5764 = v5763 + 1;	// L8929
          ap_int<66> v5765 = v5764;	// L8930
          ap_int<66> v5766 = v5765 * 8;	// L8931
          ap_int<66> v5767 = v5766 - 1;	// L8932
          int v5768 = v5762;	// L8933
          int v5769 = v5767;	// L8934
          ap_int<128> v5770;
          ap_int<128> v5770_tmp = v5760;
          v5770_tmp(v5769, v5768) = v5759;
          v5770 = v5770_tmp;	// L8935
          packed_c95 = v5770;	// L8936
        } else {
          ap_int<128> v5771 = packed_tmp95;	// L8938
          int64_t v5772 = m95;	// L8939
          int64_t v5773 = v5772 * 8;	// L8940
          ap_int<34> v5774 = m95;	// L8941
          ap_int<34> v5775 = v5774 + 1;	// L8942
          ap_int<66> v5776 = v5775;	// L8943
          ap_int<66> v5777 = v5776 * 8;	// L8944
          ap_int<66> v5778 = v5777 - 1;	// L8945
          int v5779 = v5773;	// L8946
          int v5780 = v5778;	// L8947
          int8_t v5781;
          ap_int<128> v5781_tmp = v5771;
          v5781 = v5781_tmp(v5780, v5779);	// L8948
          ap_int<128> v5782 = packed_c95;	// L8949
          ap_int<128> v5783;
          ap_int<128> v5783_tmp = v5782;
          v5783_tmp(v5780, v5779) = v5781;
          v5783 = v5783_tmp;	// L8959
          packed_c95 = v5783;	// L8960
        }
      }
      ap_int<128> v5784 = packed_c95;	// L8963
      v5733.write(v5784);	// L8964
    }
  }
}

void gemm_7_0(
  hls::stream< ap_uint<128> >& v5785,
  hls::stream< int8_t >& v5786,
  hls::stream< ap_uint<128> >& v5787
) {	// L8969
  l_S_mt_nt_0_mt121: for (int mt121 = 0; mt121 < 4; mt121++) {	// L8972
    l_nt121: for (int nt121 = 0; nt121 < 4; nt121++) {	// L8973
      l_S_k_0_k121: for (int k121 = 0; k121 < 64; k121++) {	// L8974
        ap_uint<128> v5791 = v5785.read();	// L8975
        ap_uint<128> a102;	// L8976
        a102 = v5791;	// L8977
        ap_int<128> v5793 = a102;	// L8978
        int8_t v5794;
        ap_int<128> v5794_tmp = v5793;
        v5794 = v5794_tmp(55, 48);	// L8979
        v5786.write(v5794);	// L8980
        ap_int<128> v5795 = a102;	// L8981
        v5787.write(v5795);	// L8982
      }
    }
  }
}

void gemm_7_1(
  hls::stream< int8_t >& v5796,
  hls::stream< int8_t >& v5797,
  hls::stream< int8_t >& v5798,
  hls::stream< int8_t >& v5799,
  hls::stream< ap_uint<128> >& v5800,
  hls::stream< ap_uint<128> >& v5801
) {	// L8988
  l_S_mt_nt_0_mt122: for (int mt122 = 0; mt122 < 4; mt122++) {	// L8996
    l_nt122: for (int nt122 = 0; nt122 < 4; nt122++) {	// L8997
      int8_t c96;	// L8998
      c96 = 0;	// L8999
      l_S_k_0_k122: for (int k122 = 0; k122 < 64; k122++) {	// L9000
        int8_t v5806 = v5796.read();	// L9001
        int8_t a103;	// L9002
        a103 = v5806;	// L9003
        int8_t v5808 = v5797.read();	// L9004
        int8_t b112;	// L9005
        b112 = v5808;	// L9006
        int8_t v5810 = a103;	// L9007
        int8_t v5811 = b112;	// L9008
        int16_t v5812 = v5810;	// L9009
        int16_t v5813 = v5811;	// L9010
        int16_t v5814 = v5812 * v5813;	// L9011
        int8_t v5815 = c96;	// L9012
        ap_int<17> v5816 = v5815;	// L9013
        ap_int<17> v5817 = v5814;	// L9014
        ap_int<17> v5818 = v5816 + v5817;	// L9015
        int8_t v5819 = v5818;	// L9016
        c96 = v5819;	// L9017
        int8_t v5820 = a103;	// L9018
        v5798.write(v5820);	// L9019
        int8_t v5821 = b112;	// L9020
        v5799.write(v5821);	// L9021
      }
      ap_uint<128> packed_tmp96;	// L9023
      ap_uint<128> v5823 = v5800.read();	// L9024
      packed_tmp96 = v5823;	// L9025
      ap_uint<128> packed_c96;	// L9026
      packed_c96 = 0;	// L9027
      l_S_m_1_m96: for (int m96 = 0; m96 < 16; m96++) {	// L9028
        ap_int<33> v5826 = m96;	// L9029
        bool v5827 = v5826 == 6;	// L9030
        if (v5827) {	// L9031
          int8_t v5828 = c96;	// L9032
          ap_int<128> v5829 = packed_c96;	// L9033
          int64_t v5830 = m96;	// L9034
          int64_t v5831 = v5830 * 8;	// L9035
          ap_int<34> v5832 = m96;	// L9036
          ap_int<34> v5833 = v5832 + 1;	// L9037
          ap_int<66> v5834 = v5833;	// L9038
          ap_int<66> v5835 = v5834 * 8;	// L9039
          ap_int<66> v5836 = v5835 - 1;	// L9040
          int v5837 = v5831;	// L9041
          int v5838 = v5836;	// L9042
          ap_int<128> v5839;
          ap_int<128> v5839_tmp = v5829;
          v5839_tmp(v5838, v5837) = v5828;
          v5839 = v5839_tmp;	// L9043
          packed_c96 = v5839;	// L9044
        } else {
          ap_int<128> v5840 = packed_tmp96;	// L9046
          int64_t v5841 = m96;	// L9047
          int64_t v5842 = v5841 * 8;	// L9048
          ap_int<34> v5843 = m96;	// L9049
          ap_int<34> v5844 = v5843 + 1;	// L9050
          ap_int<66> v5845 = v5844;	// L9051
          ap_int<66> v5846 = v5845 * 8;	// L9052
          ap_int<66> v5847 = v5846 - 1;	// L9053
          int v5848 = v5842;	// L9054
          int v5849 = v5847;	// L9055
          int8_t v5850;
          ap_int<128> v5850_tmp = v5840;
          v5850 = v5850_tmp(v5849, v5848);	// L9056
          ap_int<128> v5851 = packed_c96;	// L9057
          ap_int<128> v5852;
          ap_int<128> v5852_tmp = v5851;
          v5852_tmp(v5849, v5848) = v5850;
          v5852 = v5852_tmp;	// L9067
          packed_c96 = v5852;	// L9068
        }
      }
      ap_int<128> v5853 = packed_c96;	// L9071
      v5801.write(v5853);	// L9072
    }
  }
}

void gemm_7_2(
  hls::stream< int8_t >& v5854,
  hls::stream< int8_t >& v5855,
  hls::stream< int8_t >& v5856,
  hls::stream< int8_t >& v5857,
  hls::stream< ap_uint<128> >& v5858,
  hls::stream< ap_uint<128> >& v5859
) {	// L9077
  l_S_mt_nt_0_mt123: for (int mt123 = 0; mt123 < 4; mt123++) {	// L9085
    l_nt123: for (int nt123 = 0; nt123 < 4; nt123++) {	// L9086
      int8_t c97;	// L9087
      c97 = 0;	// L9088
      l_S_k_0_k123: for (int k123 = 0; k123 < 64; k123++) {	// L9089
        int8_t v5864 = v5854.read();	// L9090
        int8_t a104;	// L9091
        a104 = v5864;	// L9092
        int8_t v5866 = v5855.read();	// L9093
        int8_t b113;	// L9094
        b113 = v5866;	// L9095
        int8_t v5868 = a104;	// L9096
        int8_t v5869 = b113;	// L9097
        int16_t v5870 = v5868;	// L9098
        int16_t v5871 = v5869;	// L9099
        int16_t v5872 = v5870 * v5871;	// L9100
        int8_t v5873 = c97;	// L9101
        ap_int<17> v5874 = v5873;	// L9102
        ap_int<17> v5875 = v5872;	// L9103
        ap_int<17> v5876 = v5874 + v5875;	// L9104
        int8_t v5877 = v5876;	// L9105
        c97 = v5877;	// L9106
        int8_t v5878 = a104;	// L9107
        v5856.write(v5878);	// L9108
        int8_t v5879 = b113;	// L9109
        v5857.write(v5879);	// L9110
      }
      ap_uint<128> packed_tmp97;	// L9112
      ap_uint<128> v5881 = v5858.read();	// L9113
      packed_tmp97 = v5881;	// L9114
      ap_uint<128> packed_c97;	// L9115
      packed_c97 = 0;	// L9116
      l_S_m_1_m97: for (int m97 = 0; m97 < 16; m97++) {	// L9117
        ap_int<33> v5884 = m97;	// L9118
        bool v5885 = v5884 == 6;	// L9119
        if (v5885) {	// L9120
          int8_t v5886 = c97;	// L9121
          ap_int<128> v5887 = packed_c97;	// L9122
          int64_t v5888 = m97;	// L9123
          int64_t v5889 = v5888 * 8;	// L9124
          ap_int<34> v5890 = m97;	// L9125
          ap_int<34> v5891 = v5890 + 1;	// L9126
          ap_int<66> v5892 = v5891;	// L9127
          ap_int<66> v5893 = v5892 * 8;	// L9128
          ap_int<66> v5894 = v5893 - 1;	// L9129
          int v5895 = v5889;	// L9130
          int v5896 = v5894;	// L9131
          ap_int<128> v5897;
          ap_int<128> v5897_tmp = v5887;
          v5897_tmp(v5896, v5895) = v5886;
          v5897 = v5897_tmp;	// L9132
          packed_c97 = v5897;	// L9133
        } else {
          ap_int<128> v5898 = packed_tmp97;	// L9135
          int64_t v5899 = m97;	// L9136
          int64_t v5900 = v5899 * 8;	// L9137
          ap_int<34> v5901 = m97;	// L9138
          ap_int<34> v5902 = v5901 + 1;	// L9139
          ap_int<66> v5903 = v5902;	// L9140
          ap_int<66> v5904 = v5903 * 8;	// L9141
          ap_int<66> v5905 = v5904 - 1;	// L9142
          int v5906 = v5900;	// L9143
          int v5907 = v5905;	// L9144
          int8_t v5908;
          ap_int<128> v5908_tmp = v5898;
          v5908 = v5908_tmp(v5907, v5906);	// L9145
          ap_int<128> v5909 = packed_c97;	// L9146
          ap_int<128> v5910;
          ap_int<128> v5910_tmp = v5909;
          v5910_tmp(v5907, v5906) = v5908;
          v5910 = v5910_tmp;	// L9156
          packed_c97 = v5910;	// L9157
        }
      }
      ap_int<128> v5911 = packed_c97;	// L9160
      v5859.write(v5911);	// L9161
    }
  }
}

void gemm_7_3(
  hls::stream< int8_t >& v5912,
  hls::stream< int8_t >& v5913,
  hls::stream< int8_t >& v5914,
  hls::stream< int8_t >& v5915,
  hls::stream< ap_uint<128> >& v5916,
  hls::stream< ap_uint<128> >& v5917
) {	// L9166
  l_S_mt_nt_0_mt124: for (int mt124 = 0; mt124 < 4; mt124++) {	// L9174
    l_nt124: for (int nt124 = 0; nt124 < 4; nt124++) {	// L9175
      int8_t c98;	// L9176
      c98 = 0;	// L9177
      l_S_k_0_k124: for (int k124 = 0; k124 < 64; k124++) {	// L9178
        int8_t v5922 = v5912.read();	// L9179
        int8_t a105;	// L9180
        a105 = v5922;	// L9181
        int8_t v5924 = v5913.read();	// L9182
        int8_t b114;	// L9183
        b114 = v5924;	// L9184
        int8_t v5926 = a105;	// L9185
        int8_t v5927 = b114;	// L9186
        int16_t v5928 = v5926;	// L9187
        int16_t v5929 = v5927;	// L9188
        int16_t v5930 = v5928 * v5929;	// L9189
        int8_t v5931 = c98;	// L9190
        ap_int<17> v5932 = v5931;	// L9191
        ap_int<17> v5933 = v5930;	// L9192
        ap_int<17> v5934 = v5932 + v5933;	// L9193
        int8_t v5935 = v5934;	// L9194
        c98 = v5935;	// L9195
        int8_t v5936 = a105;	// L9196
        v5914.write(v5936);	// L9197
        int8_t v5937 = b114;	// L9198
        v5915.write(v5937);	// L9199
      }
      ap_uint<128> packed_tmp98;	// L9201
      ap_uint<128> v5939 = v5916.read();	// L9202
      packed_tmp98 = v5939;	// L9203
      ap_uint<128> packed_c98;	// L9204
      packed_c98 = 0;	// L9205
      l_S_m_1_m98: for (int m98 = 0; m98 < 16; m98++) {	// L9206
        ap_int<33> v5942 = m98;	// L9207
        bool v5943 = v5942 == 6;	// L9208
        if (v5943) {	// L9209
          int8_t v5944 = c98;	// L9210
          ap_int<128> v5945 = packed_c98;	// L9211
          int64_t v5946 = m98;	// L9212
          int64_t v5947 = v5946 * 8;	// L9213
          ap_int<34> v5948 = m98;	// L9214
          ap_int<34> v5949 = v5948 + 1;	// L9215
          ap_int<66> v5950 = v5949;	// L9216
          ap_int<66> v5951 = v5950 * 8;	// L9217
          ap_int<66> v5952 = v5951 - 1;	// L9218
          int v5953 = v5947;	// L9219
          int v5954 = v5952;	// L9220
          ap_int<128> v5955;
          ap_int<128> v5955_tmp = v5945;
          v5955_tmp(v5954, v5953) = v5944;
          v5955 = v5955_tmp;	// L9221
          packed_c98 = v5955;	// L9222
        } else {
          ap_int<128> v5956 = packed_tmp98;	// L9224
          int64_t v5957 = m98;	// L9225
          int64_t v5958 = v5957 * 8;	// L9226
          ap_int<34> v5959 = m98;	// L9227
          ap_int<34> v5960 = v5959 + 1;	// L9228
          ap_int<66> v5961 = v5960;	// L9229
          ap_int<66> v5962 = v5961 * 8;	// L9230
          ap_int<66> v5963 = v5962 - 1;	// L9231
          int v5964 = v5958;	// L9232
          int v5965 = v5963;	// L9233
          int8_t v5966;
          ap_int<128> v5966_tmp = v5956;
          v5966 = v5966_tmp(v5965, v5964);	// L9234
          ap_int<128> v5967 = packed_c98;	// L9235
          ap_int<128> v5968;
          ap_int<128> v5968_tmp = v5967;
          v5968_tmp(v5965, v5964) = v5966;
          v5968 = v5968_tmp;	// L9245
          packed_c98 = v5968;	// L9246
        }
      }
      ap_int<128> v5969 = packed_c98;	// L9249
      v5917.write(v5969);	// L9250
    }
  }
}

void gemm_7_4(
  hls::stream< int8_t >& v5970,
  hls::stream< int8_t >& v5971,
  hls::stream< int8_t >& v5972,
  hls::stream< int8_t >& v5973,
  hls::stream< ap_uint<128> >& v5974,
  hls::stream< ap_uint<128> >& v5975
) {	// L9255
  l_S_mt_nt_0_mt125: for (int mt125 = 0; mt125 < 4; mt125++) {	// L9263
    l_nt125: for (int nt125 = 0; nt125 < 4; nt125++) {	// L9264
      int8_t c99;	// L9265
      c99 = 0;	// L9266
      l_S_k_0_k125: for (int k125 = 0; k125 < 64; k125++) {	// L9267
        int8_t v5980 = v5970.read();	// L9268
        int8_t a106;	// L9269
        a106 = v5980;	// L9270
        int8_t v5982 = v5971.read();	// L9271
        int8_t b115;	// L9272
        b115 = v5982;	// L9273
        int8_t v5984 = a106;	// L9274
        int8_t v5985 = b115;	// L9275
        int16_t v5986 = v5984;	// L9276
        int16_t v5987 = v5985;	// L9277
        int16_t v5988 = v5986 * v5987;	// L9278
        int8_t v5989 = c99;	// L9279
        ap_int<17> v5990 = v5989;	// L9280
        ap_int<17> v5991 = v5988;	// L9281
        ap_int<17> v5992 = v5990 + v5991;	// L9282
        int8_t v5993 = v5992;	// L9283
        c99 = v5993;	// L9284
        int8_t v5994 = a106;	// L9285
        v5972.write(v5994);	// L9286
        int8_t v5995 = b115;	// L9287
        v5973.write(v5995);	// L9288
      }
      ap_uint<128> packed_tmp99;	// L9290
      ap_uint<128> v5997 = v5974.read();	// L9291
      packed_tmp99 = v5997;	// L9292
      ap_uint<128> packed_c99;	// L9293
      packed_c99 = 0;	// L9294
      l_S_m_1_m99: for (int m99 = 0; m99 < 16; m99++) {	// L9295
        ap_int<33> v6000 = m99;	// L9296
        bool v6001 = v6000 == 6;	// L9297
        if (v6001) {	// L9298
          int8_t v6002 = c99;	// L9299
          ap_int<128> v6003 = packed_c99;	// L9300
          int64_t v6004 = m99;	// L9301
          int64_t v6005 = v6004 * 8;	// L9302
          ap_int<34> v6006 = m99;	// L9303
          ap_int<34> v6007 = v6006 + 1;	// L9304
          ap_int<66> v6008 = v6007;	// L9305
          ap_int<66> v6009 = v6008 * 8;	// L9306
          ap_int<66> v6010 = v6009 - 1;	// L9307
          int v6011 = v6005;	// L9308
          int v6012 = v6010;	// L9309
          ap_int<128> v6013;
          ap_int<128> v6013_tmp = v6003;
          v6013_tmp(v6012, v6011) = v6002;
          v6013 = v6013_tmp;	// L9310
          packed_c99 = v6013;	// L9311
        } else {
          ap_int<128> v6014 = packed_tmp99;	// L9313
          int64_t v6015 = m99;	// L9314
          int64_t v6016 = v6015 * 8;	// L9315
          ap_int<34> v6017 = m99;	// L9316
          ap_int<34> v6018 = v6017 + 1;	// L9317
          ap_int<66> v6019 = v6018;	// L9318
          ap_int<66> v6020 = v6019 * 8;	// L9319
          ap_int<66> v6021 = v6020 - 1;	// L9320
          int v6022 = v6016;	// L9321
          int v6023 = v6021;	// L9322
          int8_t v6024;
          ap_int<128> v6024_tmp = v6014;
          v6024 = v6024_tmp(v6023, v6022);	// L9323
          ap_int<128> v6025 = packed_c99;	// L9324
          ap_int<128> v6026;
          ap_int<128> v6026_tmp = v6025;
          v6026_tmp(v6023, v6022) = v6024;
          v6026 = v6026_tmp;	// L9334
          packed_c99 = v6026;	// L9335
        }
      }
      ap_int<128> v6027 = packed_c99;	// L9338
      v5975.write(v6027);	// L9339
    }
  }
}

void gemm_7_5(
  hls::stream< int8_t >& v6028,
  hls::stream< int8_t >& v6029,
  hls::stream< int8_t >& v6030,
  hls::stream< int8_t >& v6031,
  hls::stream< ap_uint<128> >& v6032,
  hls::stream< ap_uint<128> >& v6033
) {	// L9344
  l_S_mt_nt_0_mt126: for (int mt126 = 0; mt126 < 4; mt126++) {	// L9352
    l_nt126: for (int nt126 = 0; nt126 < 4; nt126++) {	// L9353
      int8_t c100;	// L9354
      c100 = 0;	// L9355
      l_S_k_0_k126: for (int k126 = 0; k126 < 64; k126++) {	// L9356
        int8_t v6038 = v6028.read();	// L9357
        int8_t a107;	// L9358
        a107 = v6038;	// L9359
        int8_t v6040 = v6029.read();	// L9360
        int8_t b116;	// L9361
        b116 = v6040;	// L9362
        int8_t v6042 = a107;	// L9363
        int8_t v6043 = b116;	// L9364
        int16_t v6044 = v6042;	// L9365
        int16_t v6045 = v6043;	// L9366
        int16_t v6046 = v6044 * v6045;	// L9367
        int8_t v6047 = c100;	// L9368
        ap_int<17> v6048 = v6047;	// L9369
        ap_int<17> v6049 = v6046;	// L9370
        ap_int<17> v6050 = v6048 + v6049;	// L9371
        int8_t v6051 = v6050;	// L9372
        c100 = v6051;	// L9373
        int8_t v6052 = a107;	// L9374
        v6030.write(v6052);	// L9375
        int8_t v6053 = b116;	// L9376
        v6031.write(v6053);	// L9377
      }
      ap_uint<128> packed_tmp100;	// L9379
      ap_uint<128> v6055 = v6032.read();	// L9380
      packed_tmp100 = v6055;	// L9381
      ap_uint<128> packed_c100;	// L9382
      packed_c100 = 0;	// L9383
      l_S_m_1_m100: for (int m100 = 0; m100 < 16; m100++) {	// L9384
        ap_int<33> v6058 = m100;	// L9385
        bool v6059 = v6058 == 6;	// L9386
        if (v6059) {	// L9387
          int8_t v6060 = c100;	// L9388
          ap_int<128> v6061 = packed_c100;	// L9389
          int64_t v6062 = m100;	// L9390
          int64_t v6063 = v6062 * 8;	// L9391
          ap_int<34> v6064 = m100;	// L9392
          ap_int<34> v6065 = v6064 + 1;	// L9393
          ap_int<66> v6066 = v6065;	// L9394
          ap_int<66> v6067 = v6066 * 8;	// L9395
          ap_int<66> v6068 = v6067 - 1;	// L9396
          int v6069 = v6063;	// L9397
          int v6070 = v6068;	// L9398
          ap_int<128> v6071;
          ap_int<128> v6071_tmp = v6061;
          v6071_tmp(v6070, v6069) = v6060;
          v6071 = v6071_tmp;	// L9399
          packed_c100 = v6071;	// L9400
        } else {
          ap_int<128> v6072 = packed_tmp100;	// L9402
          int64_t v6073 = m100;	// L9403
          int64_t v6074 = v6073 * 8;	// L9404
          ap_int<34> v6075 = m100;	// L9405
          ap_int<34> v6076 = v6075 + 1;	// L9406
          ap_int<66> v6077 = v6076;	// L9407
          ap_int<66> v6078 = v6077 * 8;	// L9408
          ap_int<66> v6079 = v6078 - 1;	// L9409
          int v6080 = v6074;	// L9410
          int v6081 = v6079;	// L9411
          int8_t v6082;
          ap_int<128> v6082_tmp = v6072;
          v6082 = v6082_tmp(v6081, v6080);	// L9412
          ap_int<128> v6083 = packed_c100;	// L9413
          ap_int<128> v6084;
          ap_int<128> v6084_tmp = v6083;
          v6084_tmp(v6081, v6080) = v6082;
          v6084 = v6084_tmp;	// L9423
          packed_c100 = v6084;	// L9424
        }
      }
      ap_int<128> v6085 = packed_c100;	// L9427
      v6033.write(v6085);	// L9428
    }
  }
}

void gemm_7_6(
  hls::stream< int8_t >& v6086,
  hls::stream< int8_t >& v6087,
  hls::stream< int8_t >& v6088,
  hls::stream< int8_t >& v6089,
  hls::stream< ap_uint<128> >& v6090,
  hls::stream< ap_uint<128> >& v6091
) {	// L9433
  l_S_mt_nt_0_mt127: for (int mt127 = 0; mt127 < 4; mt127++) {	// L9441
    l_nt127: for (int nt127 = 0; nt127 < 4; nt127++) {	// L9442
      int8_t c101;	// L9443
      c101 = 0;	// L9444
      l_S_k_0_k127: for (int k127 = 0; k127 < 64; k127++) {	// L9445
        int8_t v6096 = v6086.read();	// L9446
        int8_t a108;	// L9447
        a108 = v6096;	// L9448
        int8_t v6098 = v6087.read();	// L9449
        int8_t b117;	// L9450
        b117 = v6098;	// L9451
        int8_t v6100 = a108;	// L9452
        int8_t v6101 = b117;	// L9453
        int16_t v6102 = v6100;	// L9454
        int16_t v6103 = v6101;	// L9455
        int16_t v6104 = v6102 * v6103;	// L9456
        int8_t v6105 = c101;	// L9457
        ap_int<17> v6106 = v6105;	// L9458
        ap_int<17> v6107 = v6104;	// L9459
        ap_int<17> v6108 = v6106 + v6107;	// L9460
        int8_t v6109 = v6108;	// L9461
        c101 = v6109;	// L9462
        int8_t v6110 = a108;	// L9463
        v6088.write(v6110);	// L9464
        int8_t v6111 = b117;	// L9465
        v6089.write(v6111);	// L9466
      }
      ap_uint<128> packed_tmp101;	// L9468
      ap_uint<128> v6113 = v6090.read();	// L9469
      packed_tmp101 = v6113;	// L9470
      ap_uint<128> packed_c101;	// L9471
      packed_c101 = 0;	// L9472
      l_S_m_1_m101: for (int m101 = 0; m101 < 16; m101++) {	// L9473
        ap_int<33> v6116 = m101;	// L9474
        bool v6117 = v6116 == 6;	// L9475
        if (v6117) {	// L9476
          int8_t v6118 = c101;	// L9477
          ap_int<128> v6119 = packed_c101;	// L9478
          int64_t v6120 = m101;	// L9479
          int64_t v6121 = v6120 * 8;	// L9480
          ap_int<34> v6122 = m101;	// L9481
          ap_int<34> v6123 = v6122 + 1;	// L9482
          ap_int<66> v6124 = v6123;	// L9483
          ap_int<66> v6125 = v6124 * 8;	// L9484
          ap_int<66> v6126 = v6125 - 1;	// L9485
          int v6127 = v6121;	// L9486
          int v6128 = v6126;	// L9487
          ap_int<128> v6129;
          ap_int<128> v6129_tmp = v6119;
          v6129_tmp(v6128, v6127) = v6118;
          v6129 = v6129_tmp;	// L9488
          packed_c101 = v6129;	// L9489
        } else {
          ap_int<128> v6130 = packed_tmp101;	// L9491
          int64_t v6131 = m101;	// L9492
          int64_t v6132 = v6131 * 8;	// L9493
          ap_int<34> v6133 = m101;	// L9494
          ap_int<34> v6134 = v6133 + 1;	// L9495
          ap_int<66> v6135 = v6134;	// L9496
          ap_int<66> v6136 = v6135 * 8;	// L9497
          ap_int<66> v6137 = v6136 - 1;	// L9498
          int v6138 = v6132;	// L9499
          int v6139 = v6137;	// L9500
          int8_t v6140;
          ap_int<128> v6140_tmp = v6130;
          v6140 = v6140_tmp(v6139, v6138);	// L9501
          ap_int<128> v6141 = packed_c101;	// L9502
          ap_int<128> v6142;
          ap_int<128> v6142_tmp = v6141;
          v6142_tmp(v6139, v6138) = v6140;
          v6142 = v6142_tmp;	// L9512
          packed_c101 = v6142;	// L9513
        }
      }
      ap_int<128> v6143 = packed_c101;	// L9516
      v6091.write(v6143);	// L9517
    }
  }
}

void gemm_7_7(
  hls::stream< int8_t >& v6144,
  hls::stream< int8_t >& v6145,
  hls::stream< int8_t >& v6146,
  hls::stream< int8_t >& v6147,
  hls::stream< ap_uint<128> >& v6148,
  hls::stream< ap_uint<128> >& v6149
) {	// L9522
  l_S_mt_nt_0_mt128: for (int mt128 = 0; mt128 < 4; mt128++) {	// L9530
    l_nt128: for (int nt128 = 0; nt128 < 4; nt128++) {	// L9531
      int8_t c102;	// L9532
      c102 = 0;	// L9533
      l_S_k_0_k128: for (int k128 = 0; k128 < 64; k128++) {	// L9534
        int8_t v6154 = v6144.read();	// L9535
        int8_t a109;	// L9536
        a109 = v6154;	// L9537
        int8_t v6156 = v6145.read();	// L9538
        int8_t b118;	// L9539
        b118 = v6156;	// L9540
        int8_t v6158 = a109;	// L9541
        int8_t v6159 = b118;	// L9542
        int16_t v6160 = v6158;	// L9543
        int16_t v6161 = v6159;	// L9544
        int16_t v6162 = v6160 * v6161;	// L9545
        int8_t v6163 = c102;	// L9546
        ap_int<17> v6164 = v6163;	// L9547
        ap_int<17> v6165 = v6162;	// L9548
        ap_int<17> v6166 = v6164 + v6165;	// L9549
        int8_t v6167 = v6166;	// L9550
        c102 = v6167;	// L9551
        int8_t v6168 = a109;	// L9552
        v6146.write(v6168);	// L9553
        int8_t v6169 = b118;	// L9554
        v6147.write(v6169);	// L9555
      }
      ap_uint<128> packed_tmp102;	// L9557
      ap_uint<128> v6171 = v6148.read();	// L9558
      packed_tmp102 = v6171;	// L9559
      ap_uint<128> packed_c102;	// L9560
      packed_c102 = 0;	// L9561
      l_S_m_1_m102: for (int m102 = 0; m102 < 16; m102++) {	// L9562
        ap_int<33> v6174 = m102;	// L9563
        bool v6175 = v6174 == 6;	// L9564
        if (v6175) {	// L9565
          int8_t v6176 = c102;	// L9566
          ap_int<128> v6177 = packed_c102;	// L9567
          int64_t v6178 = m102;	// L9568
          int64_t v6179 = v6178 * 8;	// L9569
          ap_int<34> v6180 = m102;	// L9570
          ap_int<34> v6181 = v6180 + 1;	// L9571
          ap_int<66> v6182 = v6181;	// L9572
          ap_int<66> v6183 = v6182 * 8;	// L9573
          ap_int<66> v6184 = v6183 - 1;	// L9574
          int v6185 = v6179;	// L9575
          int v6186 = v6184;	// L9576
          ap_int<128> v6187;
          ap_int<128> v6187_tmp = v6177;
          v6187_tmp(v6186, v6185) = v6176;
          v6187 = v6187_tmp;	// L9577
          packed_c102 = v6187;	// L9578
        } else {
          ap_int<128> v6188 = packed_tmp102;	// L9580
          int64_t v6189 = m102;	// L9581
          int64_t v6190 = v6189 * 8;	// L9582
          ap_int<34> v6191 = m102;	// L9583
          ap_int<34> v6192 = v6191 + 1;	// L9584
          ap_int<66> v6193 = v6192;	// L9585
          ap_int<66> v6194 = v6193 * 8;	// L9586
          ap_int<66> v6195 = v6194 - 1;	// L9587
          int v6196 = v6190;	// L9588
          int v6197 = v6195;	// L9589
          int8_t v6198;
          ap_int<128> v6198_tmp = v6188;
          v6198 = v6198_tmp(v6197, v6196);	// L9590
          ap_int<128> v6199 = packed_c102;	// L9591
          ap_int<128> v6200;
          ap_int<128> v6200_tmp = v6199;
          v6200_tmp(v6197, v6196) = v6198;
          v6200 = v6200_tmp;	// L9601
          packed_c102 = v6200;	// L9602
        }
      }
      ap_int<128> v6201 = packed_c102;	// L9605
      v6149.write(v6201);	// L9606
    }
  }
}

void gemm_7_8(
  hls::stream< int8_t >& v6202,
  hls::stream< int8_t >& v6203,
  hls::stream< int8_t >& v6204,
  hls::stream< int8_t >& v6205,
  hls::stream< ap_uint<128> >& v6206,
  hls::stream< ap_uint<128> >& v6207
) {	// L9611
  l_S_mt_nt_0_mt129: for (int mt129 = 0; mt129 < 4; mt129++) {	// L9619
    l_nt129: for (int nt129 = 0; nt129 < 4; nt129++) {	// L9620
      int8_t c103;	// L9621
      c103 = 0;	// L9622
      l_S_k_0_k129: for (int k129 = 0; k129 < 64; k129++) {	// L9623
        int8_t v6212 = v6202.read();	// L9624
        int8_t a110;	// L9625
        a110 = v6212;	// L9626
        int8_t v6214 = v6203.read();	// L9627
        int8_t b119;	// L9628
        b119 = v6214;	// L9629
        int8_t v6216 = a110;	// L9630
        int8_t v6217 = b119;	// L9631
        int16_t v6218 = v6216;	// L9632
        int16_t v6219 = v6217;	// L9633
        int16_t v6220 = v6218 * v6219;	// L9634
        int8_t v6221 = c103;	// L9635
        ap_int<17> v6222 = v6221;	// L9636
        ap_int<17> v6223 = v6220;	// L9637
        ap_int<17> v6224 = v6222 + v6223;	// L9638
        int8_t v6225 = v6224;	// L9639
        c103 = v6225;	// L9640
        int8_t v6226 = a110;	// L9641
        v6204.write(v6226);	// L9642
        int8_t v6227 = b119;	// L9643
        v6205.write(v6227);	// L9644
      }
      ap_uint<128> packed_tmp103;	// L9646
      ap_uint<128> v6229 = v6206.read();	// L9647
      packed_tmp103 = v6229;	// L9648
      ap_uint<128> packed_c103;	// L9649
      packed_c103 = 0;	// L9650
      l_S_m_1_m103: for (int m103 = 0; m103 < 16; m103++) {	// L9651
        ap_int<33> v6232 = m103;	// L9652
        bool v6233 = v6232 == 6;	// L9653
        if (v6233) {	// L9654
          int8_t v6234 = c103;	// L9655
          ap_int<128> v6235 = packed_c103;	// L9656
          int64_t v6236 = m103;	// L9657
          int64_t v6237 = v6236 * 8;	// L9658
          ap_int<34> v6238 = m103;	// L9659
          ap_int<34> v6239 = v6238 + 1;	// L9660
          ap_int<66> v6240 = v6239;	// L9661
          ap_int<66> v6241 = v6240 * 8;	// L9662
          ap_int<66> v6242 = v6241 - 1;	// L9663
          int v6243 = v6237;	// L9664
          int v6244 = v6242;	// L9665
          ap_int<128> v6245;
          ap_int<128> v6245_tmp = v6235;
          v6245_tmp(v6244, v6243) = v6234;
          v6245 = v6245_tmp;	// L9666
          packed_c103 = v6245;	// L9667
        } else {
          ap_int<128> v6246 = packed_tmp103;	// L9669
          int64_t v6247 = m103;	// L9670
          int64_t v6248 = v6247 * 8;	// L9671
          ap_int<34> v6249 = m103;	// L9672
          ap_int<34> v6250 = v6249 + 1;	// L9673
          ap_int<66> v6251 = v6250;	// L9674
          ap_int<66> v6252 = v6251 * 8;	// L9675
          ap_int<66> v6253 = v6252 - 1;	// L9676
          int v6254 = v6248;	// L9677
          int v6255 = v6253;	// L9678
          int8_t v6256;
          ap_int<128> v6256_tmp = v6246;
          v6256 = v6256_tmp(v6255, v6254);	// L9679
          ap_int<128> v6257 = packed_c103;	// L9680
          ap_int<128> v6258;
          ap_int<128> v6258_tmp = v6257;
          v6258_tmp(v6255, v6254) = v6256;
          v6258 = v6258_tmp;	// L9690
          packed_c103 = v6258;	// L9691
        }
      }
      ap_int<128> v6259 = packed_c103;	// L9694
      v6207.write(v6259);	// L9695
    }
  }
}

void gemm_7_9(
  hls::stream< int8_t >& v6260,
  hls::stream< int8_t >& v6261,
  hls::stream< int8_t >& v6262,
  hls::stream< int8_t >& v6263,
  hls::stream< ap_uint<128> >& v6264,
  hls::stream< ap_uint<128> >& v6265
) {	// L9700
  l_S_mt_nt_0_mt130: for (int mt130 = 0; mt130 < 4; mt130++) {	// L9708
    l_nt130: for (int nt130 = 0; nt130 < 4; nt130++) {	// L9709
      int8_t c104;	// L9710
      c104 = 0;	// L9711
      l_S_k_0_k130: for (int k130 = 0; k130 < 64; k130++) {	// L9712
        int8_t v6270 = v6260.read();	// L9713
        int8_t a111;	// L9714
        a111 = v6270;	// L9715
        int8_t v6272 = v6261.read();	// L9716
        int8_t b120;	// L9717
        b120 = v6272;	// L9718
        int8_t v6274 = a111;	// L9719
        int8_t v6275 = b120;	// L9720
        int16_t v6276 = v6274;	// L9721
        int16_t v6277 = v6275;	// L9722
        int16_t v6278 = v6276 * v6277;	// L9723
        int8_t v6279 = c104;	// L9724
        ap_int<17> v6280 = v6279;	// L9725
        ap_int<17> v6281 = v6278;	// L9726
        ap_int<17> v6282 = v6280 + v6281;	// L9727
        int8_t v6283 = v6282;	// L9728
        c104 = v6283;	// L9729
        int8_t v6284 = a111;	// L9730
        v6262.write(v6284);	// L9731
        int8_t v6285 = b120;	// L9732
        v6263.write(v6285);	// L9733
      }
      ap_uint<128> packed_tmp104;	// L9735
      ap_uint<128> v6287 = v6264.read();	// L9736
      packed_tmp104 = v6287;	// L9737
      ap_uint<128> packed_c104;	// L9738
      packed_c104 = 0;	// L9739
      l_S_m_1_m104: for (int m104 = 0; m104 < 16; m104++) {	// L9740
        ap_int<33> v6290 = m104;	// L9741
        bool v6291 = v6290 == 6;	// L9742
        if (v6291) {	// L9743
          int8_t v6292 = c104;	// L9744
          ap_int<128> v6293 = packed_c104;	// L9745
          int64_t v6294 = m104;	// L9746
          int64_t v6295 = v6294 * 8;	// L9747
          ap_int<34> v6296 = m104;	// L9748
          ap_int<34> v6297 = v6296 + 1;	// L9749
          ap_int<66> v6298 = v6297;	// L9750
          ap_int<66> v6299 = v6298 * 8;	// L9751
          ap_int<66> v6300 = v6299 - 1;	// L9752
          int v6301 = v6295;	// L9753
          int v6302 = v6300;	// L9754
          ap_int<128> v6303;
          ap_int<128> v6303_tmp = v6293;
          v6303_tmp(v6302, v6301) = v6292;
          v6303 = v6303_tmp;	// L9755
          packed_c104 = v6303;	// L9756
        } else {
          ap_int<128> v6304 = packed_tmp104;	// L9758
          int64_t v6305 = m104;	// L9759
          int64_t v6306 = v6305 * 8;	// L9760
          ap_int<34> v6307 = m104;	// L9761
          ap_int<34> v6308 = v6307 + 1;	// L9762
          ap_int<66> v6309 = v6308;	// L9763
          ap_int<66> v6310 = v6309 * 8;	// L9764
          ap_int<66> v6311 = v6310 - 1;	// L9765
          int v6312 = v6306;	// L9766
          int v6313 = v6311;	// L9767
          int8_t v6314;
          ap_int<128> v6314_tmp = v6304;
          v6314 = v6314_tmp(v6313, v6312);	// L9768
          ap_int<128> v6315 = packed_c104;	// L9769
          ap_int<128> v6316;
          ap_int<128> v6316_tmp = v6315;
          v6316_tmp(v6313, v6312) = v6314;
          v6316 = v6316_tmp;	// L9779
          packed_c104 = v6316;	// L9780
        }
      }
      ap_int<128> v6317 = packed_c104;	// L9783
      v6265.write(v6317);	// L9784
    }
  }
}

void gemm_7_10(
  hls::stream< int8_t >& v6318,
  hls::stream< int8_t >& v6319,
  hls::stream< int8_t >& v6320,
  hls::stream< int8_t >& v6321,
  hls::stream< ap_uint<128> >& v6322,
  hls::stream< ap_uint<128> >& v6323
) {	// L9789
  l_S_mt_nt_0_mt131: for (int mt131 = 0; mt131 < 4; mt131++) {	// L9797
    l_nt131: for (int nt131 = 0; nt131 < 4; nt131++) {	// L9798
      int8_t c105;	// L9799
      c105 = 0;	// L9800
      l_S_k_0_k131: for (int k131 = 0; k131 < 64; k131++) {	// L9801
        int8_t v6328 = v6318.read();	// L9802
        int8_t a112;	// L9803
        a112 = v6328;	// L9804
        int8_t v6330 = v6319.read();	// L9805
        int8_t b121;	// L9806
        b121 = v6330;	// L9807
        int8_t v6332 = a112;	// L9808
        int8_t v6333 = b121;	// L9809
        int16_t v6334 = v6332;	// L9810
        int16_t v6335 = v6333;	// L9811
        int16_t v6336 = v6334 * v6335;	// L9812
        int8_t v6337 = c105;	// L9813
        ap_int<17> v6338 = v6337;	// L9814
        ap_int<17> v6339 = v6336;	// L9815
        ap_int<17> v6340 = v6338 + v6339;	// L9816
        int8_t v6341 = v6340;	// L9817
        c105 = v6341;	// L9818
        int8_t v6342 = a112;	// L9819
        v6320.write(v6342);	// L9820
        int8_t v6343 = b121;	// L9821
        v6321.write(v6343);	// L9822
      }
      ap_uint<128> packed_tmp105;	// L9824
      ap_uint<128> v6345 = v6322.read();	// L9825
      packed_tmp105 = v6345;	// L9826
      ap_uint<128> packed_c105;	// L9827
      packed_c105 = 0;	// L9828
      l_S_m_1_m105: for (int m105 = 0; m105 < 16; m105++) {	// L9829
        ap_int<33> v6348 = m105;	// L9830
        bool v6349 = v6348 == 6;	// L9831
        if (v6349) {	// L9832
          int8_t v6350 = c105;	// L9833
          ap_int<128> v6351 = packed_c105;	// L9834
          int64_t v6352 = m105;	// L9835
          int64_t v6353 = v6352 * 8;	// L9836
          ap_int<34> v6354 = m105;	// L9837
          ap_int<34> v6355 = v6354 + 1;	// L9838
          ap_int<66> v6356 = v6355;	// L9839
          ap_int<66> v6357 = v6356 * 8;	// L9840
          ap_int<66> v6358 = v6357 - 1;	// L9841
          int v6359 = v6353;	// L9842
          int v6360 = v6358;	// L9843
          ap_int<128> v6361;
          ap_int<128> v6361_tmp = v6351;
          v6361_tmp(v6360, v6359) = v6350;
          v6361 = v6361_tmp;	// L9844
          packed_c105 = v6361;	// L9845
        } else {
          ap_int<128> v6362 = packed_tmp105;	// L9847
          int64_t v6363 = m105;	// L9848
          int64_t v6364 = v6363 * 8;	// L9849
          ap_int<34> v6365 = m105;	// L9850
          ap_int<34> v6366 = v6365 + 1;	// L9851
          ap_int<66> v6367 = v6366;	// L9852
          ap_int<66> v6368 = v6367 * 8;	// L9853
          ap_int<66> v6369 = v6368 - 1;	// L9854
          int v6370 = v6364;	// L9855
          int v6371 = v6369;	// L9856
          int8_t v6372;
          ap_int<128> v6372_tmp = v6362;
          v6372 = v6372_tmp(v6371, v6370);	// L9857
          ap_int<128> v6373 = packed_c105;	// L9858
          ap_int<128> v6374;
          ap_int<128> v6374_tmp = v6373;
          v6374_tmp(v6371, v6370) = v6372;
          v6374 = v6374_tmp;	// L9868
          packed_c105 = v6374;	// L9869
        }
      }
      ap_int<128> v6375 = packed_c105;	// L9872
      v6323.write(v6375);	// L9873
    }
  }
}

void gemm_7_11(
  hls::stream< int8_t >& v6376,
  hls::stream< int8_t >& v6377,
  hls::stream< int8_t >& v6378,
  hls::stream< int8_t >& v6379,
  hls::stream< ap_uint<128> >& v6380,
  hls::stream< ap_uint<128> >& v6381
) {	// L9878
  l_S_mt_nt_0_mt132: for (int mt132 = 0; mt132 < 4; mt132++) {	// L9886
    l_nt132: for (int nt132 = 0; nt132 < 4; nt132++) {	// L9887
      int8_t c106;	// L9888
      c106 = 0;	// L9889
      l_S_k_0_k132: for (int k132 = 0; k132 < 64; k132++) {	// L9890
        int8_t v6386 = v6376.read();	// L9891
        int8_t a113;	// L9892
        a113 = v6386;	// L9893
        int8_t v6388 = v6377.read();	// L9894
        int8_t b122;	// L9895
        b122 = v6388;	// L9896
        int8_t v6390 = a113;	// L9897
        int8_t v6391 = b122;	// L9898
        int16_t v6392 = v6390;	// L9899
        int16_t v6393 = v6391;	// L9900
        int16_t v6394 = v6392 * v6393;	// L9901
        int8_t v6395 = c106;	// L9902
        ap_int<17> v6396 = v6395;	// L9903
        ap_int<17> v6397 = v6394;	// L9904
        ap_int<17> v6398 = v6396 + v6397;	// L9905
        int8_t v6399 = v6398;	// L9906
        c106 = v6399;	// L9907
        int8_t v6400 = a113;	// L9908
        v6378.write(v6400);	// L9909
        int8_t v6401 = b122;	// L9910
        v6379.write(v6401);	// L9911
      }
      ap_uint<128> packed_tmp106;	// L9913
      ap_uint<128> v6403 = v6380.read();	// L9914
      packed_tmp106 = v6403;	// L9915
      ap_uint<128> packed_c106;	// L9916
      packed_c106 = 0;	// L9917
      l_S_m_1_m106: for (int m106 = 0; m106 < 16; m106++) {	// L9918
        ap_int<33> v6406 = m106;	// L9919
        bool v6407 = v6406 == 6;	// L9920
        if (v6407) {	// L9921
          int8_t v6408 = c106;	// L9922
          ap_int<128> v6409 = packed_c106;	// L9923
          int64_t v6410 = m106;	// L9924
          int64_t v6411 = v6410 * 8;	// L9925
          ap_int<34> v6412 = m106;	// L9926
          ap_int<34> v6413 = v6412 + 1;	// L9927
          ap_int<66> v6414 = v6413;	// L9928
          ap_int<66> v6415 = v6414 * 8;	// L9929
          ap_int<66> v6416 = v6415 - 1;	// L9930
          int v6417 = v6411;	// L9931
          int v6418 = v6416;	// L9932
          ap_int<128> v6419;
          ap_int<128> v6419_tmp = v6409;
          v6419_tmp(v6418, v6417) = v6408;
          v6419 = v6419_tmp;	// L9933
          packed_c106 = v6419;	// L9934
        } else {
          ap_int<128> v6420 = packed_tmp106;	// L9936
          int64_t v6421 = m106;	// L9937
          int64_t v6422 = v6421 * 8;	// L9938
          ap_int<34> v6423 = m106;	// L9939
          ap_int<34> v6424 = v6423 + 1;	// L9940
          ap_int<66> v6425 = v6424;	// L9941
          ap_int<66> v6426 = v6425 * 8;	// L9942
          ap_int<66> v6427 = v6426 - 1;	// L9943
          int v6428 = v6422;	// L9944
          int v6429 = v6427;	// L9945
          int8_t v6430;
          ap_int<128> v6430_tmp = v6420;
          v6430 = v6430_tmp(v6429, v6428);	// L9946
          ap_int<128> v6431 = packed_c106;	// L9947
          ap_int<128> v6432;
          ap_int<128> v6432_tmp = v6431;
          v6432_tmp(v6429, v6428) = v6430;
          v6432 = v6432_tmp;	// L9957
          packed_c106 = v6432;	// L9958
        }
      }
      ap_int<128> v6433 = packed_c106;	// L9961
      v6381.write(v6433);	// L9962
    }
  }
}

void gemm_7_12(
  hls::stream< int8_t >& v6434,
  hls::stream< int8_t >& v6435,
  hls::stream< int8_t >& v6436,
  hls::stream< int8_t >& v6437,
  hls::stream< ap_uint<128> >& v6438,
  hls::stream< ap_uint<128> >& v6439
) {	// L9967
  l_S_mt_nt_0_mt133: for (int mt133 = 0; mt133 < 4; mt133++) {	// L9975
    l_nt133: for (int nt133 = 0; nt133 < 4; nt133++) {	// L9976
      int8_t c107;	// L9977
      c107 = 0;	// L9978
      l_S_k_0_k133: for (int k133 = 0; k133 < 64; k133++) {	// L9979
        int8_t v6444 = v6434.read();	// L9980
        int8_t a114;	// L9981
        a114 = v6444;	// L9982
        int8_t v6446 = v6435.read();	// L9983
        int8_t b123;	// L9984
        b123 = v6446;	// L9985
        int8_t v6448 = a114;	// L9986
        int8_t v6449 = b123;	// L9987
        int16_t v6450 = v6448;	// L9988
        int16_t v6451 = v6449;	// L9989
        int16_t v6452 = v6450 * v6451;	// L9990
        int8_t v6453 = c107;	// L9991
        ap_int<17> v6454 = v6453;	// L9992
        ap_int<17> v6455 = v6452;	// L9993
        ap_int<17> v6456 = v6454 + v6455;	// L9994
        int8_t v6457 = v6456;	// L9995
        c107 = v6457;	// L9996
        int8_t v6458 = a114;	// L9997
        v6436.write(v6458);	// L9998
        int8_t v6459 = b123;	// L9999
        v6437.write(v6459);	// L10000
      }
      ap_uint<128> packed_tmp107;	// L10002
      ap_uint<128> v6461 = v6438.read();	// L10003
      packed_tmp107 = v6461;	// L10004
      ap_uint<128> packed_c107;	// L10005
      packed_c107 = 0;	// L10006
      l_S_m_1_m107: for (int m107 = 0; m107 < 16; m107++) {	// L10007
        ap_int<33> v6464 = m107;	// L10008
        bool v6465 = v6464 == 6;	// L10009
        if (v6465) {	// L10010
          int8_t v6466 = c107;	// L10011
          ap_int<128> v6467 = packed_c107;	// L10012
          int64_t v6468 = m107;	// L10013
          int64_t v6469 = v6468 * 8;	// L10014
          ap_int<34> v6470 = m107;	// L10015
          ap_int<34> v6471 = v6470 + 1;	// L10016
          ap_int<66> v6472 = v6471;	// L10017
          ap_int<66> v6473 = v6472 * 8;	// L10018
          ap_int<66> v6474 = v6473 - 1;	// L10019
          int v6475 = v6469;	// L10020
          int v6476 = v6474;	// L10021
          ap_int<128> v6477;
          ap_int<128> v6477_tmp = v6467;
          v6477_tmp(v6476, v6475) = v6466;
          v6477 = v6477_tmp;	// L10022
          packed_c107 = v6477;	// L10023
        } else {
          ap_int<128> v6478 = packed_tmp107;	// L10025
          int64_t v6479 = m107;	// L10026
          int64_t v6480 = v6479 * 8;	// L10027
          ap_int<34> v6481 = m107;	// L10028
          ap_int<34> v6482 = v6481 + 1;	// L10029
          ap_int<66> v6483 = v6482;	// L10030
          ap_int<66> v6484 = v6483 * 8;	// L10031
          ap_int<66> v6485 = v6484 - 1;	// L10032
          int v6486 = v6480;	// L10033
          int v6487 = v6485;	// L10034
          int8_t v6488;
          ap_int<128> v6488_tmp = v6478;
          v6488 = v6488_tmp(v6487, v6486);	// L10035
          ap_int<128> v6489 = packed_c107;	// L10036
          ap_int<128> v6490;
          ap_int<128> v6490_tmp = v6489;
          v6490_tmp(v6487, v6486) = v6488;
          v6490 = v6490_tmp;	// L10046
          packed_c107 = v6490;	// L10047
        }
      }
      ap_int<128> v6491 = packed_c107;	// L10050
      v6439.write(v6491);	// L10051
    }
  }
}

void gemm_7_13(
  hls::stream< int8_t >& v6492,
  hls::stream< int8_t >& v6493,
  hls::stream< int8_t >& v6494,
  hls::stream< int8_t >& v6495,
  hls::stream< ap_uint<128> >& v6496,
  hls::stream< ap_uint<128> >& v6497
) {	// L10056
  l_S_mt_nt_0_mt134: for (int mt134 = 0; mt134 < 4; mt134++) {	// L10064
    l_nt134: for (int nt134 = 0; nt134 < 4; nt134++) {	// L10065
      int8_t c108;	// L10066
      c108 = 0;	// L10067
      l_S_k_0_k134: for (int k134 = 0; k134 < 64; k134++) {	// L10068
        int8_t v6502 = v6492.read();	// L10069
        int8_t a115;	// L10070
        a115 = v6502;	// L10071
        int8_t v6504 = v6493.read();	// L10072
        int8_t b124;	// L10073
        b124 = v6504;	// L10074
        int8_t v6506 = a115;	// L10075
        int8_t v6507 = b124;	// L10076
        int16_t v6508 = v6506;	// L10077
        int16_t v6509 = v6507;	// L10078
        int16_t v6510 = v6508 * v6509;	// L10079
        int8_t v6511 = c108;	// L10080
        ap_int<17> v6512 = v6511;	// L10081
        ap_int<17> v6513 = v6510;	// L10082
        ap_int<17> v6514 = v6512 + v6513;	// L10083
        int8_t v6515 = v6514;	// L10084
        c108 = v6515;	// L10085
        int8_t v6516 = a115;	// L10086
        v6494.write(v6516);	// L10087
        int8_t v6517 = b124;	// L10088
        v6495.write(v6517);	// L10089
      }
      ap_uint<128> packed_tmp108;	// L10091
      ap_uint<128> v6519 = v6496.read();	// L10092
      packed_tmp108 = v6519;	// L10093
      ap_uint<128> packed_c108;	// L10094
      packed_c108 = 0;	// L10095
      l_S_m_1_m108: for (int m108 = 0; m108 < 16; m108++) {	// L10096
        ap_int<33> v6522 = m108;	// L10097
        bool v6523 = v6522 == 6;	// L10098
        if (v6523) {	// L10099
          int8_t v6524 = c108;	// L10100
          ap_int<128> v6525 = packed_c108;	// L10101
          int64_t v6526 = m108;	// L10102
          int64_t v6527 = v6526 * 8;	// L10103
          ap_int<34> v6528 = m108;	// L10104
          ap_int<34> v6529 = v6528 + 1;	// L10105
          ap_int<66> v6530 = v6529;	// L10106
          ap_int<66> v6531 = v6530 * 8;	// L10107
          ap_int<66> v6532 = v6531 - 1;	// L10108
          int v6533 = v6527;	// L10109
          int v6534 = v6532;	// L10110
          ap_int<128> v6535;
          ap_int<128> v6535_tmp = v6525;
          v6535_tmp(v6534, v6533) = v6524;
          v6535 = v6535_tmp;	// L10111
          packed_c108 = v6535;	// L10112
        } else {
          ap_int<128> v6536 = packed_tmp108;	// L10114
          int64_t v6537 = m108;	// L10115
          int64_t v6538 = v6537 * 8;	// L10116
          ap_int<34> v6539 = m108;	// L10117
          ap_int<34> v6540 = v6539 + 1;	// L10118
          ap_int<66> v6541 = v6540;	// L10119
          ap_int<66> v6542 = v6541 * 8;	// L10120
          ap_int<66> v6543 = v6542 - 1;	// L10121
          int v6544 = v6538;	// L10122
          int v6545 = v6543;	// L10123
          int8_t v6546;
          ap_int<128> v6546_tmp = v6536;
          v6546 = v6546_tmp(v6545, v6544);	// L10124
          ap_int<128> v6547 = packed_c108;	// L10125
          ap_int<128> v6548;
          ap_int<128> v6548_tmp = v6547;
          v6548_tmp(v6545, v6544) = v6546;
          v6548 = v6548_tmp;	// L10135
          packed_c108 = v6548;	// L10136
        }
      }
      ap_int<128> v6549 = packed_c108;	// L10139
      v6497.write(v6549);	// L10140
    }
  }
}

void gemm_7_14(
  hls::stream< int8_t >& v6550,
  hls::stream< int8_t >& v6551,
  hls::stream< int8_t >& v6552,
  hls::stream< int8_t >& v6553,
  hls::stream< ap_uint<128> >& v6554,
  hls::stream< ap_uint<128> >& v6555
) {	// L10145
  l_S_mt_nt_0_mt135: for (int mt135 = 0; mt135 < 4; mt135++) {	// L10153
    l_nt135: for (int nt135 = 0; nt135 < 4; nt135++) {	// L10154
      int8_t c109;	// L10155
      c109 = 0;	// L10156
      l_S_k_0_k135: for (int k135 = 0; k135 < 64; k135++) {	// L10157
        int8_t v6560 = v6550.read();	// L10158
        int8_t a116;	// L10159
        a116 = v6560;	// L10160
        int8_t v6562 = v6551.read();	// L10161
        int8_t b125;	// L10162
        b125 = v6562;	// L10163
        int8_t v6564 = a116;	// L10164
        int8_t v6565 = b125;	// L10165
        int16_t v6566 = v6564;	// L10166
        int16_t v6567 = v6565;	// L10167
        int16_t v6568 = v6566 * v6567;	// L10168
        int8_t v6569 = c109;	// L10169
        ap_int<17> v6570 = v6569;	// L10170
        ap_int<17> v6571 = v6568;	// L10171
        ap_int<17> v6572 = v6570 + v6571;	// L10172
        int8_t v6573 = v6572;	// L10173
        c109 = v6573;	// L10174
        int8_t v6574 = a116;	// L10175
        v6552.write(v6574);	// L10176
        int8_t v6575 = b125;	// L10177
        v6553.write(v6575);	// L10178
      }
      ap_uint<128> packed_tmp109;	// L10180
      ap_uint<128> v6577 = v6554.read();	// L10181
      packed_tmp109 = v6577;	// L10182
      ap_uint<128> packed_c109;	// L10183
      packed_c109 = 0;	// L10184
      l_S_m_1_m109: for (int m109 = 0; m109 < 16; m109++) {	// L10185
        ap_int<33> v6580 = m109;	// L10186
        bool v6581 = v6580 == 6;	// L10187
        if (v6581) {	// L10188
          int8_t v6582 = c109;	// L10189
          ap_int<128> v6583 = packed_c109;	// L10190
          int64_t v6584 = m109;	// L10191
          int64_t v6585 = v6584 * 8;	// L10192
          ap_int<34> v6586 = m109;	// L10193
          ap_int<34> v6587 = v6586 + 1;	// L10194
          ap_int<66> v6588 = v6587;	// L10195
          ap_int<66> v6589 = v6588 * 8;	// L10196
          ap_int<66> v6590 = v6589 - 1;	// L10197
          int v6591 = v6585;	// L10198
          int v6592 = v6590;	// L10199
          ap_int<128> v6593;
          ap_int<128> v6593_tmp = v6583;
          v6593_tmp(v6592, v6591) = v6582;
          v6593 = v6593_tmp;	// L10200
          packed_c109 = v6593;	// L10201
        } else {
          ap_int<128> v6594 = packed_tmp109;	// L10203
          int64_t v6595 = m109;	// L10204
          int64_t v6596 = v6595 * 8;	// L10205
          ap_int<34> v6597 = m109;	// L10206
          ap_int<34> v6598 = v6597 + 1;	// L10207
          ap_int<66> v6599 = v6598;	// L10208
          ap_int<66> v6600 = v6599 * 8;	// L10209
          ap_int<66> v6601 = v6600 - 1;	// L10210
          int v6602 = v6596;	// L10211
          int v6603 = v6601;	// L10212
          int8_t v6604;
          ap_int<128> v6604_tmp = v6594;
          v6604 = v6604_tmp(v6603, v6602);	// L10213
          ap_int<128> v6605 = packed_c109;	// L10214
          ap_int<128> v6606;
          ap_int<128> v6606_tmp = v6605;
          v6606_tmp(v6603, v6602) = v6604;
          v6606 = v6606_tmp;	// L10224
          packed_c109 = v6606;	// L10225
        }
      }
      ap_int<128> v6607 = packed_c109;	// L10228
      v6555.write(v6607);	// L10229
    }
  }
}

void gemm_7_15(
  hls::stream< int8_t >& v6608,
  hls::stream< int8_t >& v6609,
  hls::stream< int8_t >& v6610,
  hls::stream< int8_t >& v6611,
  hls::stream< ap_uint<128> >& v6612,
  hls::stream< ap_uint<128> >& v6613
) {	// L10234
  l_S_mt_nt_0_mt136: for (int mt136 = 0; mt136 < 4; mt136++) {	// L10242
    l_nt136: for (int nt136 = 0; nt136 < 4; nt136++) {	// L10243
      int8_t c110;	// L10244
      c110 = 0;	// L10245
      l_S_k_0_k136: for (int k136 = 0; k136 < 64; k136++) {	// L10246
        int8_t v6618 = v6608.read();	// L10247
        int8_t a117;	// L10248
        a117 = v6618;	// L10249
        int8_t v6620 = v6609.read();	// L10250
        int8_t b126;	// L10251
        b126 = v6620;	// L10252
        int8_t v6622 = a117;	// L10253
        int8_t v6623 = b126;	// L10254
        int16_t v6624 = v6622;	// L10255
        int16_t v6625 = v6623;	// L10256
        int16_t v6626 = v6624 * v6625;	// L10257
        int8_t v6627 = c110;	// L10258
        ap_int<17> v6628 = v6627;	// L10259
        ap_int<17> v6629 = v6626;	// L10260
        ap_int<17> v6630 = v6628 + v6629;	// L10261
        int8_t v6631 = v6630;	// L10262
        c110 = v6631;	// L10263
        int8_t v6632 = a117;	// L10264
        v6610.write(v6632);	// L10265
        int8_t v6633 = b126;	// L10266
        v6611.write(v6633);	// L10267
      }
      ap_uint<128> packed_tmp110;	// L10269
      ap_uint<128> v6635 = v6612.read();	// L10270
      packed_tmp110 = v6635;	// L10271
      ap_uint<128> packed_c110;	// L10272
      packed_c110 = 0;	// L10273
      l_S_m_1_m110: for (int m110 = 0; m110 < 16; m110++) {	// L10274
        ap_int<33> v6638 = m110;	// L10275
        bool v6639 = v6638 == 6;	// L10276
        if (v6639) {	// L10277
          int8_t v6640 = c110;	// L10278
          ap_int<128> v6641 = packed_c110;	// L10279
          int64_t v6642 = m110;	// L10280
          int64_t v6643 = v6642 * 8;	// L10281
          ap_int<34> v6644 = m110;	// L10282
          ap_int<34> v6645 = v6644 + 1;	// L10283
          ap_int<66> v6646 = v6645;	// L10284
          ap_int<66> v6647 = v6646 * 8;	// L10285
          ap_int<66> v6648 = v6647 - 1;	// L10286
          int v6649 = v6643;	// L10287
          int v6650 = v6648;	// L10288
          ap_int<128> v6651;
          ap_int<128> v6651_tmp = v6641;
          v6651_tmp(v6650, v6649) = v6640;
          v6651 = v6651_tmp;	// L10289
          packed_c110 = v6651;	// L10290
        } else {
          ap_int<128> v6652 = packed_tmp110;	// L10292
          int64_t v6653 = m110;	// L10293
          int64_t v6654 = v6653 * 8;	// L10294
          ap_int<34> v6655 = m110;	// L10295
          ap_int<34> v6656 = v6655 + 1;	// L10296
          ap_int<66> v6657 = v6656;	// L10297
          ap_int<66> v6658 = v6657 * 8;	// L10298
          ap_int<66> v6659 = v6658 - 1;	// L10299
          int v6660 = v6654;	// L10300
          int v6661 = v6659;	// L10301
          int8_t v6662;
          ap_int<128> v6662_tmp = v6652;
          v6662 = v6662_tmp(v6661, v6660);	// L10302
          ap_int<128> v6663 = packed_c110;	// L10303
          ap_int<128> v6664;
          ap_int<128> v6664_tmp = v6663;
          v6664_tmp(v6661, v6660) = v6662;
          v6664 = v6664_tmp;	// L10313
          packed_c110 = v6664;	// L10314
        }
      }
      ap_int<128> v6665 = packed_c110;	// L10317
      v6613.write(v6665);	// L10318
    }
  }
}

void gemm_7_16(
  hls::stream< int8_t >& v6666,
  hls::stream< int8_t >& v6667,
  hls::stream< int8_t >& v6668,
  hls::stream< ap_uint<128> >& v6669,
  hls::stream< ap_uint<128> >& v6670
) {	// L10323
  l_S_mt_nt_0_mt137: for (int mt137 = 0; mt137 < 4; mt137++) {	// L10331
    l_nt137: for (int nt137 = 0; nt137 < 4; nt137++) {	// L10332
      int8_t c111;	// L10333
      c111 = 0;	// L10334
      l_S_k_0_k137: for (int k137 = 0; k137 < 64; k137++) {	// L10335
        int8_t v6675 = v6666.read();	// L10336
        int8_t a118;	// L10337
        a118 = v6675;	// L10338
        int8_t v6677 = v6667.read();	// L10339
        int8_t b127;	// L10340
        b127 = v6677;	// L10341
        int8_t v6679 = a118;	// L10342
        int8_t v6680 = b127;	// L10343
        int16_t v6681 = v6679;	// L10344
        int16_t v6682 = v6680;	// L10345
        int16_t v6683 = v6681 * v6682;	// L10346
        int8_t v6684 = c111;	// L10347
        ap_int<17> v6685 = v6684;	// L10348
        ap_int<17> v6686 = v6683;	// L10349
        ap_int<17> v6687 = v6685 + v6686;	// L10350
        int8_t v6688 = v6687;	// L10351
        c111 = v6688;	// L10352
        int8_t v6689 = b127;	// L10353
        v6668.write(v6689);	// L10354
      }
      ap_uint<128> packed_tmp111;	// L10356
      ap_uint<128> v6691 = v6669.read();	// L10357
      packed_tmp111 = v6691;	// L10358
      ap_uint<128> packed_c111;	// L10359
      packed_c111 = 0;	// L10360
      l_S_m_1_m111: for (int m111 = 0; m111 < 16; m111++) {	// L10361
        ap_int<33> v6694 = m111;	// L10362
        bool v6695 = v6694 == 6;	// L10363
        if (v6695) {	// L10364
          int8_t v6696 = c111;	// L10365
          ap_int<128> v6697 = packed_c111;	// L10366
          int64_t v6698 = m111;	// L10367
          int64_t v6699 = v6698 * 8;	// L10368
          ap_int<34> v6700 = m111;	// L10369
          ap_int<34> v6701 = v6700 + 1;	// L10370
          ap_int<66> v6702 = v6701;	// L10371
          ap_int<66> v6703 = v6702 * 8;	// L10372
          ap_int<66> v6704 = v6703 - 1;	// L10373
          int v6705 = v6699;	// L10374
          int v6706 = v6704;	// L10375
          ap_int<128> v6707;
          ap_int<128> v6707_tmp = v6697;
          v6707_tmp(v6706, v6705) = v6696;
          v6707 = v6707_tmp;	// L10376
          packed_c111 = v6707;	// L10377
        } else {
          ap_int<128> v6708 = packed_tmp111;	// L10379
          int64_t v6709 = m111;	// L10380
          int64_t v6710 = v6709 * 8;	// L10381
          ap_int<34> v6711 = m111;	// L10382
          ap_int<34> v6712 = v6711 + 1;	// L10383
          ap_int<66> v6713 = v6712;	// L10384
          ap_int<66> v6714 = v6713 * 8;	// L10385
          ap_int<66> v6715 = v6714 - 1;	// L10386
          int v6716 = v6710;	// L10387
          int v6717 = v6715;	// L10388
          int8_t v6718;
          ap_int<128> v6718_tmp = v6708;
          v6718 = v6718_tmp(v6717, v6716);	// L10389
          ap_int<128> v6719 = packed_c111;	// L10390
          ap_int<128> v6720;
          ap_int<128> v6720_tmp = v6719;
          v6720_tmp(v6717, v6716) = v6718;
          v6720 = v6720_tmp;	// L10400
          packed_c111 = v6720;	// L10401
        }
      }
      ap_int<128> v6721 = packed_c111;	// L10404
      v6670.write(v6721);	// L10405
    }
  }
}

void gemm_8_0(
  hls::stream< ap_uint<128> >& v6722,
  hls::stream< int8_t >& v6723,
  hls::stream< ap_uint<128> >& v6724
) {	// L10410
  l_S_mt_nt_0_mt138: for (int mt138 = 0; mt138 < 4; mt138++) {	// L10413
    l_nt138: for (int nt138 = 0; nt138 < 4; nt138++) {	// L10414
      l_S_k_0_k138: for (int k138 = 0; k138 < 64; k138++) {	// L10415
        ap_uint<128> v6728 = v6722.read();	// L10416
        ap_uint<128> a119;	// L10417
        a119 = v6728;	// L10418
        ap_int<128> v6730 = a119;	// L10419
        int8_t v6731;
        ap_int<128> v6731_tmp = v6730;
        v6731 = v6731_tmp(63, 56);	// L10420
        v6723.write(v6731);	// L10421
        ap_int<128> v6732 = a119;	// L10422
        v6724.write(v6732);	// L10423
      }
    }
  }
}

void gemm_8_1(
  hls::stream< int8_t >& v6733,
  hls::stream< int8_t >& v6734,
  hls::stream< int8_t >& v6735,
  hls::stream< int8_t >& v6736,
  hls::stream< ap_uint<128> >& v6737,
  hls::stream< ap_uint<128> >& v6738
) {	// L10429
  l_S_mt_nt_0_mt139: for (int mt139 = 0; mt139 < 4; mt139++) {	// L10437
    l_nt139: for (int nt139 = 0; nt139 < 4; nt139++) {	// L10438
      int8_t c112;	// L10439
      c112 = 0;	// L10440
      l_S_k_0_k139: for (int k139 = 0; k139 < 64; k139++) {	// L10441
        int8_t v6743 = v6733.read();	// L10442
        int8_t a120;	// L10443
        a120 = v6743;	// L10444
        int8_t v6745 = v6734.read();	// L10445
        int8_t b128;	// L10446
        b128 = v6745;	// L10447
        int8_t v6747 = a120;	// L10448
        int8_t v6748 = b128;	// L10449
        int16_t v6749 = v6747;	// L10450
        int16_t v6750 = v6748;	// L10451
        int16_t v6751 = v6749 * v6750;	// L10452
        int8_t v6752 = c112;	// L10453
        ap_int<17> v6753 = v6752;	// L10454
        ap_int<17> v6754 = v6751;	// L10455
        ap_int<17> v6755 = v6753 + v6754;	// L10456
        int8_t v6756 = v6755;	// L10457
        c112 = v6756;	// L10458
        int8_t v6757 = a120;	// L10459
        v6735.write(v6757);	// L10460
        int8_t v6758 = b128;	// L10461
        v6736.write(v6758);	// L10462
      }
      ap_uint<128> packed_tmp112;	// L10464
      ap_uint<128> v6760 = v6737.read();	// L10465
      packed_tmp112 = v6760;	// L10466
      ap_uint<128> packed_c112;	// L10467
      packed_c112 = 0;	// L10468
      l_S_m_1_m112: for (int m112 = 0; m112 < 16; m112++) {	// L10469
        ap_int<33> v6763 = m112;	// L10470
        bool v6764 = v6763 == 7;	// L10471
        if (v6764) {	// L10472
          int8_t v6765 = c112;	// L10473
          ap_int<128> v6766 = packed_c112;	// L10474
          int64_t v6767 = m112;	// L10475
          int64_t v6768 = v6767 * 8;	// L10476
          ap_int<34> v6769 = m112;	// L10477
          ap_int<34> v6770 = v6769 + 1;	// L10478
          ap_int<66> v6771 = v6770;	// L10479
          ap_int<66> v6772 = v6771 * 8;	// L10480
          ap_int<66> v6773 = v6772 - 1;	// L10481
          int v6774 = v6768;	// L10482
          int v6775 = v6773;	// L10483
          ap_int<128> v6776;
          ap_int<128> v6776_tmp = v6766;
          v6776_tmp(v6775, v6774) = v6765;
          v6776 = v6776_tmp;	// L10484
          packed_c112 = v6776;	// L10485
        } else {
          ap_int<128> v6777 = packed_tmp112;	// L10487
          int64_t v6778 = m112;	// L10488
          int64_t v6779 = v6778 * 8;	// L10489
          ap_int<34> v6780 = m112;	// L10490
          ap_int<34> v6781 = v6780 + 1;	// L10491
          ap_int<66> v6782 = v6781;	// L10492
          ap_int<66> v6783 = v6782 * 8;	// L10493
          ap_int<66> v6784 = v6783 - 1;	// L10494
          int v6785 = v6779;	// L10495
          int v6786 = v6784;	// L10496
          int8_t v6787;
          ap_int<128> v6787_tmp = v6777;
          v6787 = v6787_tmp(v6786, v6785);	// L10497
          ap_int<128> v6788 = packed_c112;	// L10498
          ap_int<128> v6789;
          ap_int<128> v6789_tmp = v6788;
          v6789_tmp(v6786, v6785) = v6787;
          v6789 = v6789_tmp;	// L10508
          packed_c112 = v6789;	// L10509
        }
      }
      ap_int<128> v6790 = packed_c112;	// L10512
      v6738.write(v6790);	// L10513
    }
  }
}

void gemm_8_2(
  hls::stream< int8_t >& v6791,
  hls::stream< int8_t >& v6792,
  hls::stream< int8_t >& v6793,
  hls::stream< int8_t >& v6794,
  hls::stream< ap_uint<128> >& v6795,
  hls::stream< ap_uint<128> >& v6796
) {	// L10518
  l_S_mt_nt_0_mt140: for (int mt140 = 0; mt140 < 4; mt140++) {	// L10526
    l_nt140: for (int nt140 = 0; nt140 < 4; nt140++) {	// L10527
      int8_t c113;	// L10528
      c113 = 0;	// L10529
      l_S_k_0_k140: for (int k140 = 0; k140 < 64; k140++) {	// L10530
        int8_t v6801 = v6791.read();	// L10531
        int8_t a121;	// L10532
        a121 = v6801;	// L10533
        int8_t v6803 = v6792.read();	// L10534
        int8_t b129;	// L10535
        b129 = v6803;	// L10536
        int8_t v6805 = a121;	// L10537
        int8_t v6806 = b129;	// L10538
        int16_t v6807 = v6805;	// L10539
        int16_t v6808 = v6806;	// L10540
        int16_t v6809 = v6807 * v6808;	// L10541
        int8_t v6810 = c113;	// L10542
        ap_int<17> v6811 = v6810;	// L10543
        ap_int<17> v6812 = v6809;	// L10544
        ap_int<17> v6813 = v6811 + v6812;	// L10545
        int8_t v6814 = v6813;	// L10546
        c113 = v6814;	// L10547
        int8_t v6815 = a121;	// L10548
        v6793.write(v6815);	// L10549
        int8_t v6816 = b129;	// L10550
        v6794.write(v6816);	// L10551
      }
      ap_uint<128> packed_tmp113;	// L10553
      ap_uint<128> v6818 = v6795.read();	// L10554
      packed_tmp113 = v6818;	// L10555
      ap_uint<128> packed_c113;	// L10556
      packed_c113 = 0;	// L10557
      l_S_m_1_m113: for (int m113 = 0; m113 < 16; m113++) {	// L10558
        ap_int<33> v6821 = m113;	// L10559
        bool v6822 = v6821 == 7;	// L10560
        if (v6822) {	// L10561
          int8_t v6823 = c113;	// L10562
          ap_int<128> v6824 = packed_c113;	// L10563
          int64_t v6825 = m113;	// L10564
          int64_t v6826 = v6825 * 8;	// L10565
          ap_int<34> v6827 = m113;	// L10566
          ap_int<34> v6828 = v6827 + 1;	// L10567
          ap_int<66> v6829 = v6828;	// L10568
          ap_int<66> v6830 = v6829 * 8;	// L10569
          ap_int<66> v6831 = v6830 - 1;	// L10570
          int v6832 = v6826;	// L10571
          int v6833 = v6831;	// L10572
          ap_int<128> v6834;
          ap_int<128> v6834_tmp = v6824;
          v6834_tmp(v6833, v6832) = v6823;
          v6834 = v6834_tmp;	// L10573
          packed_c113 = v6834;	// L10574
        } else {
          ap_int<128> v6835 = packed_tmp113;	// L10576
          int64_t v6836 = m113;	// L10577
          int64_t v6837 = v6836 * 8;	// L10578
          ap_int<34> v6838 = m113;	// L10579
          ap_int<34> v6839 = v6838 + 1;	// L10580
          ap_int<66> v6840 = v6839;	// L10581
          ap_int<66> v6841 = v6840 * 8;	// L10582
          ap_int<66> v6842 = v6841 - 1;	// L10583
          int v6843 = v6837;	// L10584
          int v6844 = v6842;	// L10585
          int8_t v6845;
          ap_int<128> v6845_tmp = v6835;
          v6845 = v6845_tmp(v6844, v6843);	// L10586
          ap_int<128> v6846 = packed_c113;	// L10587
          ap_int<128> v6847;
          ap_int<128> v6847_tmp = v6846;
          v6847_tmp(v6844, v6843) = v6845;
          v6847 = v6847_tmp;	// L10597
          packed_c113 = v6847;	// L10598
        }
      }
      ap_int<128> v6848 = packed_c113;	// L10601
      v6796.write(v6848);	// L10602
    }
  }
}

void gemm_8_3(
  hls::stream< int8_t >& v6849,
  hls::stream< int8_t >& v6850,
  hls::stream< int8_t >& v6851,
  hls::stream< int8_t >& v6852,
  hls::stream< ap_uint<128> >& v6853,
  hls::stream< ap_uint<128> >& v6854
) {	// L10607
  l_S_mt_nt_0_mt141: for (int mt141 = 0; mt141 < 4; mt141++) {	// L10615
    l_nt141: for (int nt141 = 0; nt141 < 4; nt141++) {	// L10616
      int8_t c114;	// L10617
      c114 = 0;	// L10618
      l_S_k_0_k141: for (int k141 = 0; k141 < 64; k141++) {	// L10619
        int8_t v6859 = v6849.read();	// L10620
        int8_t a122;	// L10621
        a122 = v6859;	// L10622
        int8_t v6861 = v6850.read();	// L10623
        int8_t b130;	// L10624
        b130 = v6861;	// L10625
        int8_t v6863 = a122;	// L10626
        int8_t v6864 = b130;	// L10627
        int16_t v6865 = v6863;	// L10628
        int16_t v6866 = v6864;	// L10629
        int16_t v6867 = v6865 * v6866;	// L10630
        int8_t v6868 = c114;	// L10631
        ap_int<17> v6869 = v6868;	// L10632
        ap_int<17> v6870 = v6867;	// L10633
        ap_int<17> v6871 = v6869 + v6870;	// L10634
        int8_t v6872 = v6871;	// L10635
        c114 = v6872;	// L10636
        int8_t v6873 = a122;	// L10637
        v6851.write(v6873);	// L10638
        int8_t v6874 = b130;	// L10639
        v6852.write(v6874);	// L10640
      }
      ap_uint<128> packed_tmp114;	// L10642
      ap_uint<128> v6876 = v6853.read();	// L10643
      packed_tmp114 = v6876;	// L10644
      ap_uint<128> packed_c114;	// L10645
      packed_c114 = 0;	// L10646
      l_S_m_1_m114: for (int m114 = 0; m114 < 16; m114++) {	// L10647
        ap_int<33> v6879 = m114;	// L10648
        bool v6880 = v6879 == 7;	// L10649
        if (v6880) {	// L10650
          int8_t v6881 = c114;	// L10651
          ap_int<128> v6882 = packed_c114;	// L10652
          int64_t v6883 = m114;	// L10653
          int64_t v6884 = v6883 * 8;	// L10654
          ap_int<34> v6885 = m114;	// L10655
          ap_int<34> v6886 = v6885 + 1;	// L10656
          ap_int<66> v6887 = v6886;	// L10657
          ap_int<66> v6888 = v6887 * 8;	// L10658
          ap_int<66> v6889 = v6888 - 1;	// L10659
          int v6890 = v6884;	// L10660
          int v6891 = v6889;	// L10661
          ap_int<128> v6892;
          ap_int<128> v6892_tmp = v6882;
          v6892_tmp(v6891, v6890) = v6881;
          v6892 = v6892_tmp;	// L10662
          packed_c114 = v6892;	// L10663
        } else {
          ap_int<128> v6893 = packed_tmp114;	// L10665
          int64_t v6894 = m114;	// L10666
          int64_t v6895 = v6894 * 8;	// L10667
          ap_int<34> v6896 = m114;	// L10668
          ap_int<34> v6897 = v6896 + 1;	// L10669
          ap_int<66> v6898 = v6897;	// L10670
          ap_int<66> v6899 = v6898 * 8;	// L10671
          ap_int<66> v6900 = v6899 - 1;	// L10672
          int v6901 = v6895;	// L10673
          int v6902 = v6900;	// L10674
          int8_t v6903;
          ap_int<128> v6903_tmp = v6893;
          v6903 = v6903_tmp(v6902, v6901);	// L10675
          ap_int<128> v6904 = packed_c114;	// L10676
          ap_int<128> v6905;
          ap_int<128> v6905_tmp = v6904;
          v6905_tmp(v6902, v6901) = v6903;
          v6905 = v6905_tmp;	// L10686
          packed_c114 = v6905;	// L10687
        }
      }
      ap_int<128> v6906 = packed_c114;	// L10690
      v6854.write(v6906);	// L10691
    }
  }
}

void gemm_8_4(
  hls::stream< int8_t >& v6907,
  hls::stream< int8_t >& v6908,
  hls::stream< int8_t >& v6909,
  hls::stream< int8_t >& v6910,
  hls::stream< ap_uint<128> >& v6911,
  hls::stream< ap_uint<128> >& v6912
) {	// L10696
  l_S_mt_nt_0_mt142: for (int mt142 = 0; mt142 < 4; mt142++) {	// L10704
    l_nt142: for (int nt142 = 0; nt142 < 4; nt142++) {	// L10705
      int8_t c115;	// L10706
      c115 = 0;	// L10707
      l_S_k_0_k142: for (int k142 = 0; k142 < 64; k142++) {	// L10708
        int8_t v6917 = v6907.read();	// L10709
        int8_t a123;	// L10710
        a123 = v6917;	// L10711
        int8_t v6919 = v6908.read();	// L10712
        int8_t b131;	// L10713
        b131 = v6919;	// L10714
        int8_t v6921 = a123;	// L10715
        int8_t v6922 = b131;	// L10716
        int16_t v6923 = v6921;	// L10717
        int16_t v6924 = v6922;	// L10718
        int16_t v6925 = v6923 * v6924;	// L10719
        int8_t v6926 = c115;	// L10720
        ap_int<17> v6927 = v6926;	// L10721
        ap_int<17> v6928 = v6925;	// L10722
        ap_int<17> v6929 = v6927 + v6928;	// L10723
        int8_t v6930 = v6929;	// L10724
        c115 = v6930;	// L10725
        int8_t v6931 = a123;	// L10726
        v6909.write(v6931);	// L10727
        int8_t v6932 = b131;	// L10728
        v6910.write(v6932);	// L10729
      }
      ap_uint<128> packed_tmp115;	// L10731
      ap_uint<128> v6934 = v6911.read();	// L10732
      packed_tmp115 = v6934;	// L10733
      ap_uint<128> packed_c115;	// L10734
      packed_c115 = 0;	// L10735
      l_S_m_1_m115: for (int m115 = 0; m115 < 16; m115++) {	// L10736
        ap_int<33> v6937 = m115;	// L10737
        bool v6938 = v6937 == 7;	// L10738
        if (v6938) {	// L10739
          int8_t v6939 = c115;	// L10740
          ap_int<128> v6940 = packed_c115;	// L10741
          int64_t v6941 = m115;	// L10742
          int64_t v6942 = v6941 * 8;	// L10743
          ap_int<34> v6943 = m115;	// L10744
          ap_int<34> v6944 = v6943 + 1;	// L10745
          ap_int<66> v6945 = v6944;	// L10746
          ap_int<66> v6946 = v6945 * 8;	// L10747
          ap_int<66> v6947 = v6946 - 1;	// L10748
          int v6948 = v6942;	// L10749
          int v6949 = v6947;	// L10750
          ap_int<128> v6950;
          ap_int<128> v6950_tmp = v6940;
          v6950_tmp(v6949, v6948) = v6939;
          v6950 = v6950_tmp;	// L10751
          packed_c115 = v6950;	// L10752
        } else {
          ap_int<128> v6951 = packed_tmp115;	// L10754
          int64_t v6952 = m115;	// L10755
          int64_t v6953 = v6952 * 8;	// L10756
          ap_int<34> v6954 = m115;	// L10757
          ap_int<34> v6955 = v6954 + 1;	// L10758
          ap_int<66> v6956 = v6955;	// L10759
          ap_int<66> v6957 = v6956 * 8;	// L10760
          ap_int<66> v6958 = v6957 - 1;	// L10761
          int v6959 = v6953;	// L10762
          int v6960 = v6958;	// L10763
          int8_t v6961;
          ap_int<128> v6961_tmp = v6951;
          v6961 = v6961_tmp(v6960, v6959);	// L10764
          ap_int<128> v6962 = packed_c115;	// L10765
          ap_int<128> v6963;
          ap_int<128> v6963_tmp = v6962;
          v6963_tmp(v6960, v6959) = v6961;
          v6963 = v6963_tmp;	// L10775
          packed_c115 = v6963;	// L10776
        }
      }
      ap_int<128> v6964 = packed_c115;	// L10779
      v6912.write(v6964);	// L10780
    }
  }
}

void gemm_8_5(
  hls::stream< int8_t >& v6965,
  hls::stream< int8_t >& v6966,
  hls::stream< int8_t >& v6967,
  hls::stream< int8_t >& v6968,
  hls::stream< ap_uint<128> >& v6969,
  hls::stream< ap_uint<128> >& v6970
) {	// L10785
  l_S_mt_nt_0_mt143: for (int mt143 = 0; mt143 < 4; mt143++) {	// L10793
    l_nt143: for (int nt143 = 0; nt143 < 4; nt143++) {	// L10794
      int8_t c116;	// L10795
      c116 = 0;	// L10796
      l_S_k_0_k143: for (int k143 = 0; k143 < 64; k143++) {	// L10797
        int8_t v6975 = v6965.read();	// L10798
        int8_t a124;	// L10799
        a124 = v6975;	// L10800
        int8_t v6977 = v6966.read();	// L10801
        int8_t b132;	// L10802
        b132 = v6977;	// L10803
        int8_t v6979 = a124;	// L10804
        int8_t v6980 = b132;	// L10805
        int16_t v6981 = v6979;	// L10806
        int16_t v6982 = v6980;	// L10807
        int16_t v6983 = v6981 * v6982;	// L10808
        int8_t v6984 = c116;	// L10809
        ap_int<17> v6985 = v6984;	// L10810
        ap_int<17> v6986 = v6983;	// L10811
        ap_int<17> v6987 = v6985 + v6986;	// L10812
        int8_t v6988 = v6987;	// L10813
        c116 = v6988;	// L10814
        int8_t v6989 = a124;	// L10815
        v6967.write(v6989);	// L10816
        int8_t v6990 = b132;	// L10817
        v6968.write(v6990);	// L10818
      }
      ap_uint<128> packed_tmp116;	// L10820
      ap_uint<128> v6992 = v6969.read();	// L10821
      packed_tmp116 = v6992;	// L10822
      ap_uint<128> packed_c116;	// L10823
      packed_c116 = 0;	// L10824
      l_S_m_1_m116: for (int m116 = 0; m116 < 16; m116++) {	// L10825
        ap_int<33> v6995 = m116;	// L10826
        bool v6996 = v6995 == 7;	// L10827
        if (v6996) {	// L10828
          int8_t v6997 = c116;	// L10829
          ap_int<128> v6998 = packed_c116;	// L10830
          int64_t v6999 = m116;	// L10831
          int64_t v7000 = v6999 * 8;	// L10832
          ap_int<34> v7001 = m116;	// L10833
          ap_int<34> v7002 = v7001 + 1;	// L10834
          ap_int<66> v7003 = v7002;	// L10835
          ap_int<66> v7004 = v7003 * 8;	// L10836
          ap_int<66> v7005 = v7004 - 1;	// L10837
          int v7006 = v7000;	// L10838
          int v7007 = v7005;	// L10839
          ap_int<128> v7008;
          ap_int<128> v7008_tmp = v6998;
          v7008_tmp(v7007, v7006) = v6997;
          v7008 = v7008_tmp;	// L10840
          packed_c116 = v7008;	// L10841
        } else {
          ap_int<128> v7009 = packed_tmp116;	// L10843
          int64_t v7010 = m116;	// L10844
          int64_t v7011 = v7010 * 8;	// L10845
          ap_int<34> v7012 = m116;	// L10846
          ap_int<34> v7013 = v7012 + 1;	// L10847
          ap_int<66> v7014 = v7013;	// L10848
          ap_int<66> v7015 = v7014 * 8;	// L10849
          ap_int<66> v7016 = v7015 - 1;	// L10850
          int v7017 = v7011;	// L10851
          int v7018 = v7016;	// L10852
          int8_t v7019;
          ap_int<128> v7019_tmp = v7009;
          v7019 = v7019_tmp(v7018, v7017);	// L10853
          ap_int<128> v7020 = packed_c116;	// L10854
          ap_int<128> v7021;
          ap_int<128> v7021_tmp = v7020;
          v7021_tmp(v7018, v7017) = v7019;
          v7021 = v7021_tmp;	// L10864
          packed_c116 = v7021;	// L10865
        }
      }
      ap_int<128> v7022 = packed_c116;	// L10868
      v6970.write(v7022);	// L10869
    }
  }
}

void gemm_8_6(
  hls::stream< int8_t >& v7023,
  hls::stream< int8_t >& v7024,
  hls::stream< int8_t >& v7025,
  hls::stream< int8_t >& v7026,
  hls::stream< ap_uint<128> >& v7027,
  hls::stream< ap_uint<128> >& v7028
) {	// L10874
  l_S_mt_nt_0_mt144: for (int mt144 = 0; mt144 < 4; mt144++) {	// L10882
    l_nt144: for (int nt144 = 0; nt144 < 4; nt144++) {	// L10883
      int8_t c117;	// L10884
      c117 = 0;	// L10885
      l_S_k_0_k144: for (int k144 = 0; k144 < 64; k144++) {	// L10886
        int8_t v7033 = v7023.read();	// L10887
        int8_t a125;	// L10888
        a125 = v7033;	// L10889
        int8_t v7035 = v7024.read();	// L10890
        int8_t b133;	// L10891
        b133 = v7035;	// L10892
        int8_t v7037 = a125;	// L10893
        int8_t v7038 = b133;	// L10894
        int16_t v7039 = v7037;	// L10895
        int16_t v7040 = v7038;	// L10896
        int16_t v7041 = v7039 * v7040;	// L10897
        int8_t v7042 = c117;	// L10898
        ap_int<17> v7043 = v7042;	// L10899
        ap_int<17> v7044 = v7041;	// L10900
        ap_int<17> v7045 = v7043 + v7044;	// L10901
        int8_t v7046 = v7045;	// L10902
        c117 = v7046;	// L10903
        int8_t v7047 = a125;	// L10904
        v7025.write(v7047);	// L10905
        int8_t v7048 = b133;	// L10906
        v7026.write(v7048);	// L10907
      }
      ap_uint<128> packed_tmp117;	// L10909
      ap_uint<128> v7050 = v7027.read();	// L10910
      packed_tmp117 = v7050;	// L10911
      ap_uint<128> packed_c117;	// L10912
      packed_c117 = 0;	// L10913
      l_S_m_1_m117: for (int m117 = 0; m117 < 16; m117++) {	// L10914
        ap_int<33> v7053 = m117;	// L10915
        bool v7054 = v7053 == 7;	// L10916
        if (v7054) {	// L10917
          int8_t v7055 = c117;	// L10918
          ap_int<128> v7056 = packed_c117;	// L10919
          int64_t v7057 = m117;	// L10920
          int64_t v7058 = v7057 * 8;	// L10921
          ap_int<34> v7059 = m117;	// L10922
          ap_int<34> v7060 = v7059 + 1;	// L10923
          ap_int<66> v7061 = v7060;	// L10924
          ap_int<66> v7062 = v7061 * 8;	// L10925
          ap_int<66> v7063 = v7062 - 1;	// L10926
          int v7064 = v7058;	// L10927
          int v7065 = v7063;	// L10928
          ap_int<128> v7066;
          ap_int<128> v7066_tmp = v7056;
          v7066_tmp(v7065, v7064) = v7055;
          v7066 = v7066_tmp;	// L10929
          packed_c117 = v7066;	// L10930
        } else {
          ap_int<128> v7067 = packed_tmp117;	// L10932
          int64_t v7068 = m117;	// L10933
          int64_t v7069 = v7068 * 8;	// L10934
          ap_int<34> v7070 = m117;	// L10935
          ap_int<34> v7071 = v7070 + 1;	// L10936
          ap_int<66> v7072 = v7071;	// L10937
          ap_int<66> v7073 = v7072 * 8;	// L10938
          ap_int<66> v7074 = v7073 - 1;	// L10939
          int v7075 = v7069;	// L10940
          int v7076 = v7074;	// L10941
          int8_t v7077;
          ap_int<128> v7077_tmp = v7067;
          v7077 = v7077_tmp(v7076, v7075);	// L10942
          ap_int<128> v7078 = packed_c117;	// L10943
          ap_int<128> v7079;
          ap_int<128> v7079_tmp = v7078;
          v7079_tmp(v7076, v7075) = v7077;
          v7079 = v7079_tmp;	// L10953
          packed_c117 = v7079;	// L10954
        }
      }
      ap_int<128> v7080 = packed_c117;	// L10957
      v7028.write(v7080);	// L10958
    }
  }
}

void gemm_8_7(
  hls::stream< int8_t >& v7081,
  hls::stream< int8_t >& v7082,
  hls::stream< int8_t >& v7083,
  hls::stream< int8_t >& v7084,
  hls::stream< ap_uint<128> >& v7085,
  hls::stream< ap_uint<128> >& v7086
) {	// L10963
  l_S_mt_nt_0_mt145: for (int mt145 = 0; mt145 < 4; mt145++) {	// L10971
    l_nt145: for (int nt145 = 0; nt145 < 4; nt145++) {	// L10972
      int8_t c118;	// L10973
      c118 = 0;	// L10974
      l_S_k_0_k145: for (int k145 = 0; k145 < 64; k145++) {	// L10975
        int8_t v7091 = v7081.read();	// L10976
        int8_t a126;	// L10977
        a126 = v7091;	// L10978
        int8_t v7093 = v7082.read();	// L10979
        int8_t b134;	// L10980
        b134 = v7093;	// L10981
        int8_t v7095 = a126;	// L10982
        int8_t v7096 = b134;	// L10983
        int16_t v7097 = v7095;	// L10984
        int16_t v7098 = v7096;	// L10985
        int16_t v7099 = v7097 * v7098;	// L10986
        int8_t v7100 = c118;	// L10987
        ap_int<17> v7101 = v7100;	// L10988
        ap_int<17> v7102 = v7099;	// L10989
        ap_int<17> v7103 = v7101 + v7102;	// L10990
        int8_t v7104 = v7103;	// L10991
        c118 = v7104;	// L10992
        int8_t v7105 = a126;	// L10993
        v7083.write(v7105);	// L10994
        int8_t v7106 = b134;	// L10995
        v7084.write(v7106);	// L10996
      }
      ap_uint<128> packed_tmp118;	// L10998
      ap_uint<128> v7108 = v7085.read();	// L10999
      packed_tmp118 = v7108;	// L11000
      ap_uint<128> packed_c118;	// L11001
      packed_c118 = 0;	// L11002
      l_S_m_1_m118: for (int m118 = 0; m118 < 16; m118++) {	// L11003
        ap_int<33> v7111 = m118;	// L11004
        bool v7112 = v7111 == 7;	// L11005
        if (v7112) {	// L11006
          int8_t v7113 = c118;	// L11007
          ap_int<128> v7114 = packed_c118;	// L11008
          int64_t v7115 = m118;	// L11009
          int64_t v7116 = v7115 * 8;	// L11010
          ap_int<34> v7117 = m118;	// L11011
          ap_int<34> v7118 = v7117 + 1;	// L11012
          ap_int<66> v7119 = v7118;	// L11013
          ap_int<66> v7120 = v7119 * 8;	// L11014
          ap_int<66> v7121 = v7120 - 1;	// L11015
          int v7122 = v7116;	// L11016
          int v7123 = v7121;	// L11017
          ap_int<128> v7124;
          ap_int<128> v7124_tmp = v7114;
          v7124_tmp(v7123, v7122) = v7113;
          v7124 = v7124_tmp;	// L11018
          packed_c118 = v7124;	// L11019
        } else {
          ap_int<128> v7125 = packed_tmp118;	// L11021
          int64_t v7126 = m118;	// L11022
          int64_t v7127 = v7126 * 8;	// L11023
          ap_int<34> v7128 = m118;	// L11024
          ap_int<34> v7129 = v7128 + 1;	// L11025
          ap_int<66> v7130 = v7129;	// L11026
          ap_int<66> v7131 = v7130 * 8;	// L11027
          ap_int<66> v7132 = v7131 - 1;	// L11028
          int v7133 = v7127;	// L11029
          int v7134 = v7132;	// L11030
          int8_t v7135;
          ap_int<128> v7135_tmp = v7125;
          v7135 = v7135_tmp(v7134, v7133);	// L11031
          ap_int<128> v7136 = packed_c118;	// L11032
          ap_int<128> v7137;
          ap_int<128> v7137_tmp = v7136;
          v7137_tmp(v7134, v7133) = v7135;
          v7137 = v7137_tmp;	// L11042
          packed_c118 = v7137;	// L11043
        }
      }
      ap_int<128> v7138 = packed_c118;	// L11046
      v7086.write(v7138);	// L11047
    }
  }
}

void gemm_8_8(
  hls::stream< int8_t >& v7139,
  hls::stream< int8_t >& v7140,
  hls::stream< int8_t >& v7141,
  hls::stream< int8_t >& v7142,
  hls::stream< ap_uint<128> >& v7143,
  hls::stream< ap_uint<128> >& v7144
) {	// L11052
  l_S_mt_nt_0_mt146: for (int mt146 = 0; mt146 < 4; mt146++) {	// L11060
    l_nt146: for (int nt146 = 0; nt146 < 4; nt146++) {	// L11061
      int8_t c119;	// L11062
      c119 = 0;	// L11063
      l_S_k_0_k146: for (int k146 = 0; k146 < 64; k146++) {	// L11064
        int8_t v7149 = v7139.read();	// L11065
        int8_t a127;	// L11066
        a127 = v7149;	// L11067
        int8_t v7151 = v7140.read();	// L11068
        int8_t b135;	// L11069
        b135 = v7151;	// L11070
        int8_t v7153 = a127;	// L11071
        int8_t v7154 = b135;	// L11072
        int16_t v7155 = v7153;	// L11073
        int16_t v7156 = v7154;	// L11074
        int16_t v7157 = v7155 * v7156;	// L11075
        int8_t v7158 = c119;	// L11076
        ap_int<17> v7159 = v7158;	// L11077
        ap_int<17> v7160 = v7157;	// L11078
        ap_int<17> v7161 = v7159 + v7160;	// L11079
        int8_t v7162 = v7161;	// L11080
        c119 = v7162;	// L11081
        int8_t v7163 = a127;	// L11082
        v7141.write(v7163);	// L11083
        int8_t v7164 = b135;	// L11084
        v7142.write(v7164);	// L11085
      }
      ap_uint<128> packed_tmp119;	// L11087
      ap_uint<128> v7166 = v7143.read();	// L11088
      packed_tmp119 = v7166;	// L11089
      ap_uint<128> packed_c119;	// L11090
      packed_c119 = 0;	// L11091
      l_S_m_1_m119: for (int m119 = 0; m119 < 16; m119++) {	// L11092
        ap_int<33> v7169 = m119;	// L11093
        bool v7170 = v7169 == 7;	// L11094
        if (v7170) {	// L11095
          int8_t v7171 = c119;	// L11096
          ap_int<128> v7172 = packed_c119;	// L11097
          int64_t v7173 = m119;	// L11098
          int64_t v7174 = v7173 * 8;	// L11099
          ap_int<34> v7175 = m119;	// L11100
          ap_int<34> v7176 = v7175 + 1;	// L11101
          ap_int<66> v7177 = v7176;	// L11102
          ap_int<66> v7178 = v7177 * 8;	// L11103
          ap_int<66> v7179 = v7178 - 1;	// L11104
          int v7180 = v7174;	// L11105
          int v7181 = v7179;	// L11106
          ap_int<128> v7182;
          ap_int<128> v7182_tmp = v7172;
          v7182_tmp(v7181, v7180) = v7171;
          v7182 = v7182_tmp;	// L11107
          packed_c119 = v7182;	// L11108
        } else {
          ap_int<128> v7183 = packed_tmp119;	// L11110
          int64_t v7184 = m119;	// L11111
          int64_t v7185 = v7184 * 8;	// L11112
          ap_int<34> v7186 = m119;	// L11113
          ap_int<34> v7187 = v7186 + 1;	// L11114
          ap_int<66> v7188 = v7187;	// L11115
          ap_int<66> v7189 = v7188 * 8;	// L11116
          ap_int<66> v7190 = v7189 - 1;	// L11117
          int v7191 = v7185;	// L11118
          int v7192 = v7190;	// L11119
          int8_t v7193;
          ap_int<128> v7193_tmp = v7183;
          v7193 = v7193_tmp(v7192, v7191);	// L11120
          ap_int<128> v7194 = packed_c119;	// L11121
          ap_int<128> v7195;
          ap_int<128> v7195_tmp = v7194;
          v7195_tmp(v7192, v7191) = v7193;
          v7195 = v7195_tmp;	// L11131
          packed_c119 = v7195;	// L11132
        }
      }
      ap_int<128> v7196 = packed_c119;	// L11135
      v7144.write(v7196);	// L11136
    }
  }
}

void gemm_8_9(
  hls::stream< int8_t >& v7197,
  hls::stream< int8_t >& v7198,
  hls::stream< int8_t >& v7199,
  hls::stream< int8_t >& v7200,
  hls::stream< ap_uint<128> >& v7201,
  hls::stream< ap_uint<128> >& v7202
) {	// L11141
  l_S_mt_nt_0_mt147: for (int mt147 = 0; mt147 < 4; mt147++) {	// L11149
    l_nt147: for (int nt147 = 0; nt147 < 4; nt147++) {	// L11150
      int8_t c120;	// L11151
      c120 = 0;	// L11152
      l_S_k_0_k147: for (int k147 = 0; k147 < 64; k147++) {	// L11153
        int8_t v7207 = v7197.read();	// L11154
        int8_t a128;	// L11155
        a128 = v7207;	// L11156
        int8_t v7209 = v7198.read();	// L11157
        int8_t b136;	// L11158
        b136 = v7209;	// L11159
        int8_t v7211 = a128;	// L11160
        int8_t v7212 = b136;	// L11161
        int16_t v7213 = v7211;	// L11162
        int16_t v7214 = v7212;	// L11163
        int16_t v7215 = v7213 * v7214;	// L11164
        int8_t v7216 = c120;	// L11165
        ap_int<17> v7217 = v7216;	// L11166
        ap_int<17> v7218 = v7215;	// L11167
        ap_int<17> v7219 = v7217 + v7218;	// L11168
        int8_t v7220 = v7219;	// L11169
        c120 = v7220;	// L11170
        int8_t v7221 = a128;	// L11171
        v7199.write(v7221);	// L11172
        int8_t v7222 = b136;	// L11173
        v7200.write(v7222);	// L11174
      }
      ap_uint<128> packed_tmp120;	// L11176
      ap_uint<128> v7224 = v7201.read();	// L11177
      packed_tmp120 = v7224;	// L11178
      ap_uint<128> packed_c120;	// L11179
      packed_c120 = 0;	// L11180
      l_S_m_1_m120: for (int m120 = 0; m120 < 16; m120++) {	// L11181
        ap_int<33> v7227 = m120;	// L11182
        bool v7228 = v7227 == 7;	// L11183
        if (v7228) {	// L11184
          int8_t v7229 = c120;	// L11185
          ap_int<128> v7230 = packed_c120;	// L11186
          int64_t v7231 = m120;	// L11187
          int64_t v7232 = v7231 * 8;	// L11188
          ap_int<34> v7233 = m120;	// L11189
          ap_int<34> v7234 = v7233 + 1;	// L11190
          ap_int<66> v7235 = v7234;	// L11191
          ap_int<66> v7236 = v7235 * 8;	// L11192
          ap_int<66> v7237 = v7236 - 1;	// L11193
          int v7238 = v7232;	// L11194
          int v7239 = v7237;	// L11195
          ap_int<128> v7240;
          ap_int<128> v7240_tmp = v7230;
          v7240_tmp(v7239, v7238) = v7229;
          v7240 = v7240_tmp;	// L11196
          packed_c120 = v7240;	// L11197
        } else {
          ap_int<128> v7241 = packed_tmp120;	// L11199
          int64_t v7242 = m120;	// L11200
          int64_t v7243 = v7242 * 8;	// L11201
          ap_int<34> v7244 = m120;	// L11202
          ap_int<34> v7245 = v7244 + 1;	// L11203
          ap_int<66> v7246 = v7245;	// L11204
          ap_int<66> v7247 = v7246 * 8;	// L11205
          ap_int<66> v7248 = v7247 - 1;	// L11206
          int v7249 = v7243;	// L11207
          int v7250 = v7248;	// L11208
          int8_t v7251;
          ap_int<128> v7251_tmp = v7241;
          v7251 = v7251_tmp(v7250, v7249);	// L11209
          ap_int<128> v7252 = packed_c120;	// L11210
          ap_int<128> v7253;
          ap_int<128> v7253_tmp = v7252;
          v7253_tmp(v7250, v7249) = v7251;
          v7253 = v7253_tmp;	// L11220
          packed_c120 = v7253;	// L11221
        }
      }
      ap_int<128> v7254 = packed_c120;	// L11224
      v7202.write(v7254);	// L11225
    }
  }
}

void gemm_8_10(
  hls::stream< int8_t >& v7255,
  hls::stream< int8_t >& v7256,
  hls::stream< int8_t >& v7257,
  hls::stream< int8_t >& v7258,
  hls::stream< ap_uint<128> >& v7259,
  hls::stream< ap_uint<128> >& v7260
) {	// L11230
  l_S_mt_nt_0_mt148: for (int mt148 = 0; mt148 < 4; mt148++) {	// L11238
    l_nt148: for (int nt148 = 0; nt148 < 4; nt148++) {	// L11239
      int8_t c121;	// L11240
      c121 = 0;	// L11241
      l_S_k_0_k148: for (int k148 = 0; k148 < 64; k148++) {	// L11242
        int8_t v7265 = v7255.read();	// L11243
        int8_t a129;	// L11244
        a129 = v7265;	// L11245
        int8_t v7267 = v7256.read();	// L11246
        int8_t b137;	// L11247
        b137 = v7267;	// L11248
        int8_t v7269 = a129;	// L11249
        int8_t v7270 = b137;	// L11250
        int16_t v7271 = v7269;	// L11251
        int16_t v7272 = v7270;	// L11252
        int16_t v7273 = v7271 * v7272;	// L11253
        int8_t v7274 = c121;	// L11254
        ap_int<17> v7275 = v7274;	// L11255
        ap_int<17> v7276 = v7273;	// L11256
        ap_int<17> v7277 = v7275 + v7276;	// L11257
        int8_t v7278 = v7277;	// L11258
        c121 = v7278;	// L11259
        int8_t v7279 = a129;	// L11260
        v7257.write(v7279);	// L11261
        int8_t v7280 = b137;	// L11262
        v7258.write(v7280);	// L11263
      }
      ap_uint<128> packed_tmp121;	// L11265
      ap_uint<128> v7282 = v7259.read();	// L11266
      packed_tmp121 = v7282;	// L11267
      ap_uint<128> packed_c121;	// L11268
      packed_c121 = 0;	// L11269
      l_S_m_1_m121: for (int m121 = 0; m121 < 16; m121++) {	// L11270
        ap_int<33> v7285 = m121;	// L11271
        bool v7286 = v7285 == 7;	// L11272
        if (v7286) {	// L11273
          int8_t v7287 = c121;	// L11274
          ap_int<128> v7288 = packed_c121;	// L11275
          int64_t v7289 = m121;	// L11276
          int64_t v7290 = v7289 * 8;	// L11277
          ap_int<34> v7291 = m121;	// L11278
          ap_int<34> v7292 = v7291 + 1;	// L11279
          ap_int<66> v7293 = v7292;	// L11280
          ap_int<66> v7294 = v7293 * 8;	// L11281
          ap_int<66> v7295 = v7294 - 1;	// L11282
          int v7296 = v7290;	// L11283
          int v7297 = v7295;	// L11284
          ap_int<128> v7298;
          ap_int<128> v7298_tmp = v7288;
          v7298_tmp(v7297, v7296) = v7287;
          v7298 = v7298_tmp;	// L11285
          packed_c121 = v7298;	// L11286
        } else {
          ap_int<128> v7299 = packed_tmp121;	// L11288
          int64_t v7300 = m121;	// L11289
          int64_t v7301 = v7300 * 8;	// L11290
          ap_int<34> v7302 = m121;	// L11291
          ap_int<34> v7303 = v7302 + 1;	// L11292
          ap_int<66> v7304 = v7303;	// L11293
          ap_int<66> v7305 = v7304 * 8;	// L11294
          ap_int<66> v7306 = v7305 - 1;	// L11295
          int v7307 = v7301;	// L11296
          int v7308 = v7306;	// L11297
          int8_t v7309;
          ap_int<128> v7309_tmp = v7299;
          v7309 = v7309_tmp(v7308, v7307);	// L11298
          ap_int<128> v7310 = packed_c121;	// L11299
          ap_int<128> v7311;
          ap_int<128> v7311_tmp = v7310;
          v7311_tmp(v7308, v7307) = v7309;
          v7311 = v7311_tmp;	// L11309
          packed_c121 = v7311;	// L11310
        }
      }
      ap_int<128> v7312 = packed_c121;	// L11313
      v7260.write(v7312);	// L11314
    }
  }
}

void gemm_8_11(
  hls::stream< int8_t >& v7313,
  hls::stream< int8_t >& v7314,
  hls::stream< int8_t >& v7315,
  hls::stream< int8_t >& v7316,
  hls::stream< ap_uint<128> >& v7317,
  hls::stream< ap_uint<128> >& v7318
) {	// L11319
  l_S_mt_nt_0_mt149: for (int mt149 = 0; mt149 < 4; mt149++) {	// L11327
    l_nt149: for (int nt149 = 0; nt149 < 4; nt149++) {	// L11328
      int8_t c122;	// L11329
      c122 = 0;	// L11330
      l_S_k_0_k149: for (int k149 = 0; k149 < 64; k149++) {	// L11331
        int8_t v7323 = v7313.read();	// L11332
        int8_t a130;	// L11333
        a130 = v7323;	// L11334
        int8_t v7325 = v7314.read();	// L11335
        int8_t b138;	// L11336
        b138 = v7325;	// L11337
        int8_t v7327 = a130;	// L11338
        int8_t v7328 = b138;	// L11339
        int16_t v7329 = v7327;	// L11340
        int16_t v7330 = v7328;	// L11341
        int16_t v7331 = v7329 * v7330;	// L11342
        int8_t v7332 = c122;	// L11343
        ap_int<17> v7333 = v7332;	// L11344
        ap_int<17> v7334 = v7331;	// L11345
        ap_int<17> v7335 = v7333 + v7334;	// L11346
        int8_t v7336 = v7335;	// L11347
        c122 = v7336;	// L11348
        int8_t v7337 = a130;	// L11349
        v7315.write(v7337);	// L11350
        int8_t v7338 = b138;	// L11351
        v7316.write(v7338);	// L11352
      }
      ap_uint<128> packed_tmp122;	// L11354
      ap_uint<128> v7340 = v7317.read();	// L11355
      packed_tmp122 = v7340;	// L11356
      ap_uint<128> packed_c122;	// L11357
      packed_c122 = 0;	// L11358
      l_S_m_1_m122: for (int m122 = 0; m122 < 16; m122++) {	// L11359
        ap_int<33> v7343 = m122;	// L11360
        bool v7344 = v7343 == 7;	// L11361
        if (v7344) {	// L11362
          int8_t v7345 = c122;	// L11363
          ap_int<128> v7346 = packed_c122;	// L11364
          int64_t v7347 = m122;	// L11365
          int64_t v7348 = v7347 * 8;	// L11366
          ap_int<34> v7349 = m122;	// L11367
          ap_int<34> v7350 = v7349 + 1;	// L11368
          ap_int<66> v7351 = v7350;	// L11369
          ap_int<66> v7352 = v7351 * 8;	// L11370
          ap_int<66> v7353 = v7352 - 1;	// L11371
          int v7354 = v7348;	// L11372
          int v7355 = v7353;	// L11373
          ap_int<128> v7356;
          ap_int<128> v7356_tmp = v7346;
          v7356_tmp(v7355, v7354) = v7345;
          v7356 = v7356_tmp;	// L11374
          packed_c122 = v7356;	// L11375
        } else {
          ap_int<128> v7357 = packed_tmp122;	// L11377
          int64_t v7358 = m122;	// L11378
          int64_t v7359 = v7358 * 8;	// L11379
          ap_int<34> v7360 = m122;	// L11380
          ap_int<34> v7361 = v7360 + 1;	// L11381
          ap_int<66> v7362 = v7361;	// L11382
          ap_int<66> v7363 = v7362 * 8;	// L11383
          ap_int<66> v7364 = v7363 - 1;	// L11384
          int v7365 = v7359;	// L11385
          int v7366 = v7364;	// L11386
          int8_t v7367;
          ap_int<128> v7367_tmp = v7357;
          v7367 = v7367_tmp(v7366, v7365);	// L11387
          ap_int<128> v7368 = packed_c122;	// L11388
          ap_int<128> v7369;
          ap_int<128> v7369_tmp = v7368;
          v7369_tmp(v7366, v7365) = v7367;
          v7369 = v7369_tmp;	// L11398
          packed_c122 = v7369;	// L11399
        }
      }
      ap_int<128> v7370 = packed_c122;	// L11402
      v7318.write(v7370);	// L11403
    }
  }
}

void gemm_8_12(
  hls::stream< int8_t >& v7371,
  hls::stream< int8_t >& v7372,
  hls::stream< int8_t >& v7373,
  hls::stream< int8_t >& v7374,
  hls::stream< ap_uint<128> >& v7375,
  hls::stream< ap_uint<128> >& v7376
) {	// L11408
  l_S_mt_nt_0_mt150: for (int mt150 = 0; mt150 < 4; mt150++) {	// L11416
    l_nt150: for (int nt150 = 0; nt150 < 4; nt150++) {	// L11417
      int8_t c123;	// L11418
      c123 = 0;	// L11419
      l_S_k_0_k150: for (int k150 = 0; k150 < 64; k150++) {	// L11420
        int8_t v7381 = v7371.read();	// L11421
        int8_t a131;	// L11422
        a131 = v7381;	// L11423
        int8_t v7383 = v7372.read();	// L11424
        int8_t b139;	// L11425
        b139 = v7383;	// L11426
        int8_t v7385 = a131;	// L11427
        int8_t v7386 = b139;	// L11428
        int16_t v7387 = v7385;	// L11429
        int16_t v7388 = v7386;	// L11430
        int16_t v7389 = v7387 * v7388;	// L11431
        int8_t v7390 = c123;	// L11432
        ap_int<17> v7391 = v7390;	// L11433
        ap_int<17> v7392 = v7389;	// L11434
        ap_int<17> v7393 = v7391 + v7392;	// L11435
        int8_t v7394 = v7393;	// L11436
        c123 = v7394;	// L11437
        int8_t v7395 = a131;	// L11438
        v7373.write(v7395);	// L11439
        int8_t v7396 = b139;	// L11440
        v7374.write(v7396);	// L11441
      }
      ap_uint<128> packed_tmp123;	// L11443
      ap_uint<128> v7398 = v7375.read();	// L11444
      packed_tmp123 = v7398;	// L11445
      ap_uint<128> packed_c123;	// L11446
      packed_c123 = 0;	// L11447
      l_S_m_1_m123: for (int m123 = 0; m123 < 16; m123++) {	// L11448
        ap_int<33> v7401 = m123;	// L11449
        bool v7402 = v7401 == 7;	// L11450
        if (v7402) {	// L11451
          int8_t v7403 = c123;	// L11452
          ap_int<128> v7404 = packed_c123;	// L11453
          int64_t v7405 = m123;	// L11454
          int64_t v7406 = v7405 * 8;	// L11455
          ap_int<34> v7407 = m123;	// L11456
          ap_int<34> v7408 = v7407 + 1;	// L11457
          ap_int<66> v7409 = v7408;	// L11458
          ap_int<66> v7410 = v7409 * 8;	// L11459
          ap_int<66> v7411 = v7410 - 1;	// L11460
          int v7412 = v7406;	// L11461
          int v7413 = v7411;	// L11462
          ap_int<128> v7414;
          ap_int<128> v7414_tmp = v7404;
          v7414_tmp(v7413, v7412) = v7403;
          v7414 = v7414_tmp;	// L11463
          packed_c123 = v7414;	// L11464
        } else {
          ap_int<128> v7415 = packed_tmp123;	// L11466
          int64_t v7416 = m123;	// L11467
          int64_t v7417 = v7416 * 8;	// L11468
          ap_int<34> v7418 = m123;	// L11469
          ap_int<34> v7419 = v7418 + 1;	// L11470
          ap_int<66> v7420 = v7419;	// L11471
          ap_int<66> v7421 = v7420 * 8;	// L11472
          ap_int<66> v7422 = v7421 - 1;	// L11473
          int v7423 = v7417;	// L11474
          int v7424 = v7422;	// L11475
          int8_t v7425;
          ap_int<128> v7425_tmp = v7415;
          v7425 = v7425_tmp(v7424, v7423);	// L11476
          ap_int<128> v7426 = packed_c123;	// L11477
          ap_int<128> v7427;
          ap_int<128> v7427_tmp = v7426;
          v7427_tmp(v7424, v7423) = v7425;
          v7427 = v7427_tmp;	// L11487
          packed_c123 = v7427;	// L11488
        }
      }
      ap_int<128> v7428 = packed_c123;	// L11491
      v7376.write(v7428);	// L11492
    }
  }
}

void gemm_8_13(
  hls::stream< int8_t >& v7429,
  hls::stream< int8_t >& v7430,
  hls::stream< int8_t >& v7431,
  hls::stream< int8_t >& v7432,
  hls::stream< ap_uint<128> >& v7433,
  hls::stream< ap_uint<128> >& v7434
) {	// L11497
  l_S_mt_nt_0_mt151: for (int mt151 = 0; mt151 < 4; mt151++) {	// L11505
    l_nt151: for (int nt151 = 0; nt151 < 4; nt151++) {	// L11506
      int8_t c124;	// L11507
      c124 = 0;	// L11508
      l_S_k_0_k151: for (int k151 = 0; k151 < 64; k151++) {	// L11509
        int8_t v7439 = v7429.read();	// L11510
        int8_t a132;	// L11511
        a132 = v7439;	// L11512
        int8_t v7441 = v7430.read();	// L11513
        int8_t b140;	// L11514
        b140 = v7441;	// L11515
        int8_t v7443 = a132;	// L11516
        int8_t v7444 = b140;	// L11517
        int16_t v7445 = v7443;	// L11518
        int16_t v7446 = v7444;	// L11519
        int16_t v7447 = v7445 * v7446;	// L11520
        int8_t v7448 = c124;	// L11521
        ap_int<17> v7449 = v7448;	// L11522
        ap_int<17> v7450 = v7447;	// L11523
        ap_int<17> v7451 = v7449 + v7450;	// L11524
        int8_t v7452 = v7451;	// L11525
        c124 = v7452;	// L11526
        int8_t v7453 = a132;	// L11527
        v7431.write(v7453);	// L11528
        int8_t v7454 = b140;	// L11529
        v7432.write(v7454);	// L11530
      }
      ap_uint<128> packed_tmp124;	// L11532
      ap_uint<128> v7456 = v7433.read();	// L11533
      packed_tmp124 = v7456;	// L11534
      ap_uint<128> packed_c124;	// L11535
      packed_c124 = 0;	// L11536
      l_S_m_1_m124: for (int m124 = 0; m124 < 16; m124++) {	// L11537
        ap_int<33> v7459 = m124;	// L11538
        bool v7460 = v7459 == 7;	// L11539
        if (v7460) {	// L11540
          int8_t v7461 = c124;	// L11541
          ap_int<128> v7462 = packed_c124;	// L11542
          int64_t v7463 = m124;	// L11543
          int64_t v7464 = v7463 * 8;	// L11544
          ap_int<34> v7465 = m124;	// L11545
          ap_int<34> v7466 = v7465 + 1;	// L11546
          ap_int<66> v7467 = v7466;	// L11547
          ap_int<66> v7468 = v7467 * 8;	// L11548
          ap_int<66> v7469 = v7468 - 1;	// L11549
          int v7470 = v7464;	// L11550
          int v7471 = v7469;	// L11551
          ap_int<128> v7472;
          ap_int<128> v7472_tmp = v7462;
          v7472_tmp(v7471, v7470) = v7461;
          v7472 = v7472_tmp;	// L11552
          packed_c124 = v7472;	// L11553
        } else {
          ap_int<128> v7473 = packed_tmp124;	// L11555
          int64_t v7474 = m124;	// L11556
          int64_t v7475 = v7474 * 8;	// L11557
          ap_int<34> v7476 = m124;	// L11558
          ap_int<34> v7477 = v7476 + 1;	// L11559
          ap_int<66> v7478 = v7477;	// L11560
          ap_int<66> v7479 = v7478 * 8;	// L11561
          ap_int<66> v7480 = v7479 - 1;	// L11562
          int v7481 = v7475;	// L11563
          int v7482 = v7480;	// L11564
          int8_t v7483;
          ap_int<128> v7483_tmp = v7473;
          v7483 = v7483_tmp(v7482, v7481);	// L11565
          ap_int<128> v7484 = packed_c124;	// L11566
          ap_int<128> v7485;
          ap_int<128> v7485_tmp = v7484;
          v7485_tmp(v7482, v7481) = v7483;
          v7485 = v7485_tmp;	// L11576
          packed_c124 = v7485;	// L11577
        }
      }
      ap_int<128> v7486 = packed_c124;	// L11580
      v7434.write(v7486);	// L11581
    }
  }
}

void gemm_8_14(
  hls::stream< int8_t >& v7487,
  hls::stream< int8_t >& v7488,
  hls::stream< int8_t >& v7489,
  hls::stream< int8_t >& v7490,
  hls::stream< ap_uint<128> >& v7491,
  hls::stream< ap_uint<128> >& v7492
) {	// L11586
  l_S_mt_nt_0_mt152: for (int mt152 = 0; mt152 < 4; mt152++) {	// L11594
    l_nt152: for (int nt152 = 0; nt152 < 4; nt152++) {	// L11595
      int8_t c125;	// L11596
      c125 = 0;	// L11597
      l_S_k_0_k152: for (int k152 = 0; k152 < 64; k152++) {	// L11598
        int8_t v7497 = v7487.read();	// L11599
        int8_t a133;	// L11600
        a133 = v7497;	// L11601
        int8_t v7499 = v7488.read();	// L11602
        int8_t b141;	// L11603
        b141 = v7499;	// L11604
        int8_t v7501 = a133;	// L11605
        int8_t v7502 = b141;	// L11606
        int16_t v7503 = v7501;	// L11607
        int16_t v7504 = v7502;	// L11608
        int16_t v7505 = v7503 * v7504;	// L11609
        int8_t v7506 = c125;	// L11610
        ap_int<17> v7507 = v7506;	// L11611
        ap_int<17> v7508 = v7505;	// L11612
        ap_int<17> v7509 = v7507 + v7508;	// L11613
        int8_t v7510 = v7509;	// L11614
        c125 = v7510;	// L11615
        int8_t v7511 = a133;	// L11616
        v7489.write(v7511);	// L11617
        int8_t v7512 = b141;	// L11618
        v7490.write(v7512);	// L11619
      }
      ap_uint<128> packed_tmp125;	// L11621
      ap_uint<128> v7514 = v7491.read();	// L11622
      packed_tmp125 = v7514;	// L11623
      ap_uint<128> packed_c125;	// L11624
      packed_c125 = 0;	// L11625
      l_S_m_1_m125: for (int m125 = 0; m125 < 16; m125++) {	// L11626
        ap_int<33> v7517 = m125;	// L11627
        bool v7518 = v7517 == 7;	// L11628
        if (v7518) {	// L11629
          int8_t v7519 = c125;	// L11630
          ap_int<128> v7520 = packed_c125;	// L11631
          int64_t v7521 = m125;	// L11632
          int64_t v7522 = v7521 * 8;	// L11633
          ap_int<34> v7523 = m125;	// L11634
          ap_int<34> v7524 = v7523 + 1;	// L11635
          ap_int<66> v7525 = v7524;	// L11636
          ap_int<66> v7526 = v7525 * 8;	// L11637
          ap_int<66> v7527 = v7526 - 1;	// L11638
          int v7528 = v7522;	// L11639
          int v7529 = v7527;	// L11640
          ap_int<128> v7530;
          ap_int<128> v7530_tmp = v7520;
          v7530_tmp(v7529, v7528) = v7519;
          v7530 = v7530_tmp;	// L11641
          packed_c125 = v7530;	// L11642
        } else {
          ap_int<128> v7531 = packed_tmp125;	// L11644
          int64_t v7532 = m125;	// L11645
          int64_t v7533 = v7532 * 8;	// L11646
          ap_int<34> v7534 = m125;	// L11647
          ap_int<34> v7535 = v7534 + 1;	// L11648
          ap_int<66> v7536 = v7535;	// L11649
          ap_int<66> v7537 = v7536 * 8;	// L11650
          ap_int<66> v7538 = v7537 - 1;	// L11651
          int v7539 = v7533;	// L11652
          int v7540 = v7538;	// L11653
          int8_t v7541;
          ap_int<128> v7541_tmp = v7531;
          v7541 = v7541_tmp(v7540, v7539);	// L11654
          ap_int<128> v7542 = packed_c125;	// L11655
          ap_int<128> v7543;
          ap_int<128> v7543_tmp = v7542;
          v7543_tmp(v7540, v7539) = v7541;
          v7543 = v7543_tmp;	// L11665
          packed_c125 = v7543;	// L11666
        }
      }
      ap_int<128> v7544 = packed_c125;	// L11669
      v7492.write(v7544);	// L11670
    }
  }
}

void gemm_8_15(
  hls::stream< int8_t >& v7545,
  hls::stream< int8_t >& v7546,
  hls::stream< int8_t >& v7547,
  hls::stream< int8_t >& v7548,
  hls::stream< ap_uint<128> >& v7549,
  hls::stream< ap_uint<128> >& v7550
) {	// L11675
  l_S_mt_nt_0_mt153: for (int mt153 = 0; mt153 < 4; mt153++) {	// L11683
    l_nt153: for (int nt153 = 0; nt153 < 4; nt153++) {	// L11684
      int8_t c126;	// L11685
      c126 = 0;	// L11686
      l_S_k_0_k153: for (int k153 = 0; k153 < 64; k153++) {	// L11687
        int8_t v7555 = v7545.read();	// L11688
        int8_t a134;	// L11689
        a134 = v7555;	// L11690
        int8_t v7557 = v7546.read();	// L11691
        int8_t b142;	// L11692
        b142 = v7557;	// L11693
        int8_t v7559 = a134;	// L11694
        int8_t v7560 = b142;	// L11695
        int16_t v7561 = v7559;	// L11696
        int16_t v7562 = v7560;	// L11697
        int16_t v7563 = v7561 * v7562;	// L11698
        int8_t v7564 = c126;	// L11699
        ap_int<17> v7565 = v7564;	// L11700
        ap_int<17> v7566 = v7563;	// L11701
        ap_int<17> v7567 = v7565 + v7566;	// L11702
        int8_t v7568 = v7567;	// L11703
        c126 = v7568;	// L11704
        int8_t v7569 = a134;	// L11705
        v7547.write(v7569);	// L11706
        int8_t v7570 = b142;	// L11707
        v7548.write(v7570);	// L11708
      }
      ap_uint<128> packed_tmp126;	// L11710
      ap_uint<128> v7572 = v7549.read();	// L11711
      packed_tmp126 = v7572;	// L11712
      ap_uint<128> packed_c126;	// L11713
      packed_c126 = 0;	// L11714
      l_S_m_1_m126: for (int m126 = 0; m126 < 16; m126++) {	// L11715
        ap_int<33> v7575 = m126;	// L11716
        bool v7576 = v7575 == 7;	// L11717
        if (v7576) {	// L11718
          int8_t v7577 = c126;	// L11719
          ap_int<128> v7578 = packed_c126;	// L11720
          int64_t v7579 = m126;	// L11721
          int64_t v7580 = v7579 * 8;	// L11722
          ap_int<34> v7581 = m126;	// L11723
          ap_int<34> v7582 = v7581 + 1;	// L11724
          ap_int<66> v7583 = v7582;	// L11725
          ap_int<66> v7584 = v7583 * 8;	// L11726
          ap_int<66> v7585 = v7584 - 1;	// L11727
          int v7586 = v7580;	// L11728
          int v7587 = v7585;	// L11729
          ap_int<128> v7588;
          ap_int<128> v7588_tmp = v7578;
          v7588_tmp(v7587, v7586) = v7577;
          v7588 = v7588_tmp;	// L11730
          packed_c126 = v7588;	// L11731
        } else {
          ap_int<128> v7589 = packed_tmp126;	// L11733
          int64_t v7590 = m126;	// L11734
          int64_t v7591 = v7590 * 8;	// L11735
          ap_int<34> v7592 = m126;	// L11736
          ap_int<34> v7593 = v7592 + 1;	// L11737
          ap_int<66> v7594 = v7593;	// L11738
          ap_int<66> v7595 = v7594 * 8;	// L11739
          ap_int<66> v7596 = v7595 - 1;	// L11740
          int v7597 = v7591;	// L11741
          int v7598 = v7596;	// L11742
          int8_t v7599;
          ap_int<128> v7599_tmp = v7589;
          v7599 = v7599_tmp(v7598, v7597);	// L11743
          ap_int<128> v7600 = packed_c126;	// L11744
          ap_int<128> v7601;
          ap_int<128> v7601_tmp = v7600;
          v7601_tmp(v7598, v7597) = v7599;
          v7601 = v7601_tmp;	// L11754
          packed_c126 = v7601;	// L11755
        }
      }
      ap_int<128> v7602 = packed_c126;	// L11758
      v7550.write(v7602);	// L11759
    }
  }
}

void gemm_8_16(
  hls::stream< int8_t >& v7603,
  hls::stream< int8_t >& v7604,
  hls::stream< int8_t >& v7605,
  hls::stream< ap_uint<128> >& v7606,
  hls::stream< ap_uint<128> >& v7607
) {	// L11764
  l_S_mt_nt_0_mt154: for (int mt154 = 0; mt154 < 4; mt154++) {	// L11772
    l_nt154: for (int nt154 = 0; nt154 < 4; nt154++) {	// L11773
      int8_t c127;	// L11774
      c127 = 0;	// L11775
      l_S_k_0_k154: for (int k154 = 0; k154 < 64; k154++) {	// L11776
        int8_t v7612 = v7603.read();	// L11777
        int8_t a135;	// L11778
        a135 = v7612;	// L11779
        int8_t v7614 = v7604.read();	// L11780
        int8_t b143;	// L11781
        b143 = v7614;	// L11782
        int8_t v7616 = a135;	// L11783
        int8_t v7617 = b143;	// L11784
        int16_t v7618 = v7616;	// L11785
        int16_t v7619 = v7617;	// L11786
        int16_t v7620 = v7618 * v7619;	// L11787
        int8_t v7621 = c127;	// L11788
        ap_int<17> v7622 = v7621;	// L11789
        ap_int<17> v7623 = v7620;	// L11790
        ap_int<17> v7624 = v7622 + v7623;	// L11791
        int8_t v7625 = v7624;	// L11792
        c127 = v7625;	// L11793
        int8_t v7626 = b143;	// L11794
        v7605.write(v7626);	// L11795
      }
      ap_uint<128> packed_tmp127;	// L11797
      ap_uint<128> v7628 = v7606.read();	// L11798
      packed_tmp127 = v7628;	// L11799
      ap_uint<128> packed_c127;	// L11800
      packed_c127 = 0;	// L11801
      l_S_m_1_m127: for (int m127 = 0; m127 < 16; m127++) {	// L11802
        ap_int<33> v7631 = m127;	// L11803
        bool v7632 = v7631 == 7;	// L11804
        if (v7632) {	// L11805
          int8_t v7633 = c127;	// L11806
          ap_int<128> v7634 = packed_c127;	// L11807
          int64_t v7635 = m127;	// L11808
          int64_t v7636 = v7635 * 8;	// L11809
          ap_int<34> v7637 = m127;	// L11810
          ap_int<34> v7638 = v7637 + 1;	// L11811
          ap_int<66> v7639 = v7638;	// L11812
          ap_int<66> v7640 = v7639 * 8;	// L11813
          ap_int<66> v7641 = v7640 - 1;	// L11814
          int v7642 = v7636;	// L11815
          int v7643 = v7641;	// L11816
          ap_int<128> v7644;
          ap_int<128> v7644_tmp = v7634;
          v7644_tmp(v7643, v7642) = v7633;
          v7644 = v7644_tmp;	// L11817
          packed_c127 = v7644;	// L11818
        } else {
          ap_int<128> v7645 = packed_tmp127;	// L11820
          int64_t v7646 = m127;	// L11821
          int64_t v7647 = v7646 * 8;	// L11822
          ap_int<34> v7648 = m127;	// L11823
          ap_int<34> v7649 = v7648 + 1;	// L11824
          ap_int<66> v7650 = v7649;	// L11825
          ap_int<66> v7651 = v7650 * 8;	// L11826
          ap_int<66> v7652 = v7651 - 1;	// L11827
          int v7653 = v7647;	// L11828
          int v7654 = v7652;	// L11829
          int8_t v7655;
          ap_int<128> v7655_tmp = v7645;
          v7655 = v7655_tmp(v7654, v7653);	// L11830
          ap_int<128> v7656 = packed_c127;	// L11831
          ap_int<128> v7657;
          ap_int<128> v7657_tmp = v7656;
          v7657_tmp(v7654, v7653) = v7655;
          v7657 = v7657_tmp;	// L11841
          packed_c127 = v7657;	// L11842
        }
      }
      ap_int<128> v7658 = packed_c127;	// L11845
      v7607.write(v7658);	// L11846
    }
  }
}

void gemm_9_0(
  hls::stream< ap_uint<128> >& v7659,
  hls::stream< int8_t >& v7660,
  hls::stream< ap_uint<128> >& v7661
) {	// L11851
  l_S_mt_nt_0_mt155: for (int mt155 = 0; mt155 < 4; mt155++) {	// L11854
    l_nt155: for (int nt155 = 0; nt155 < 4; nt155++) {	// L11855
      l_S_k_0_k155: for (int k155 = 0; k155 < 64; k155++) {	// L11856
        ap_uint<128> v7665 = v7659.read();	// L11857
        ap_uint<128> a136;	// L11858
        a136 = v7665;	// L11859
        ap_int<128> v7667 = a136;	// L11860
        int8_t v7668;
        ap_int<128> v7668_tmp = v7667;
        v7668 = v7668_tmp(71, 64);	// L11861
        v7660.write(v7668);	// L11862
        ap_int<128> v7669 = a136;	// L11863
        v7661.write(v7669);	// L11864
      }
    }
  }
}

void gemm_9_1(
  hls::stream< int8_t >& v7670,
  hls::stream< int8_t >& v7671,
  hls::stream< int8_t >& v7672,
  hls::stream< int8_t >& v7673,
  hls::stream< ap_uint<128> >& v7674,
  hls::stream< ap_uint<128> >& v7675
) {	// L11870
  l_S_mt_nt_0_mt156: for (int mt156 = 0; mt156 < 4; mt156++) {	// L11878
    l_nt156: for (int nt156 = 0; nt156 < 4; nt156++) {	// L11879
      int8_t c128;	// L11880
      c128 = 0;	// L11881
      l_S_k_0_k156: for (int k156 = 0; k156 < 64; k156++) {	// L11882
        int8_t v7680 = v7670.read();	// L11883
        int8_t a137;	// L11884
        a137 = v7680;	// L11885
        int8_t v7682 = v7671.read();	// L11886
        int8_t b144;	// L11887
        b144 = v7682;	// L11888
        int8_t v7684 = a137;	// L11889
        int8_t v7685 = b144;	// L11890
        int16_t v7686 = v7684;	// L11891
        int16_t v7687 = v7685;	// L11892
        int16_t v7688 = v7686 * v7687;	// L11893
        int8_t v7689 = c128;	// L11894
        ap_int<17> v7690 = v7689;	// L11895
        ap_int<17> v7691 = v7688;	// L11896
        ap_int<17> v7692 = v7690 + v7691;	// L11897
        int8_t v7693 = v7692;	// L11898
        c128 = v7693;	// L11899
        int8_t v7694 = a137;	// L11900
        v7672.write(v7694);	// L11901
        int8_t v7695 = b144;	// L11902
        v7673.write(v7695);	// L11903
      }
      ap_uint<128> packed_tmp128;	// L11905
      ap_uint<128> v7697 = v7674.read();	// L11906
      packed_tmp128 = v7697;	// L11907
      ap_uint<128> packed_c128;	// L11908
      packed_c128 = 0;	// L11909
      l_S_m_1_m128: for (int m128 = 0; m128 < 16; m128++) {	// L11910
        ap_int<33> v7700 = m128;	// L11911
        bool v7701 = v7700 == 8;	// L11912
        if (v7701) {	// L11913
          int8_t v7702 = c128;	// L11914
          ap_int<128> v7703 = packed_c128;	// L11915
          int64_t v7704 = m128;	// L11916
          int64_t v7705 = v7704 * 8;	// L11917
          ap_int<34> v7706 = m128;	// L11918
          ap_int<34> v7707 = v7706 + 1;	// L11919
          ap_int<66> v7708 = v7707;	// L11920
          ap_int<66> v7709 = v7708 * 8;	// L11921
          ap_int<66> v7710 = v7709 - 1;	// L11922
          int v7711 = v7705;	// L11923
          int v7712 = v7710;	// L11924
          ap_int<128> v7713;
          ap_int<128> v7713_tmp = v7703;
          v7713_tmp(v7712, v7711) = v7702;
          v7713 = v7713_tmp;	// L11925
          packed_c128 = v7713;	// L11926
        } else {
          ap_int<128> v7714 = packed_tmp128;	// L11928
          int64_t v7715 = m128;	// L11929
          int64_t v7716 = v7715 * 8;	// L11930
          ap_int<34> v7717 = m128;	// L11931
          ap_int<34> v7718 = v7717 + 1;	// L11932
          ap_int<66> v7719 = v7718;	// L11933
          ap_int<66> v7720 = v7719 * 8;	// L11934
          ap_int<66> v7721 = v7720 - 1;	// L11935
          int v7722 = v7716;	// L11936
          int v7723 = v7721;	// L11937
          int8_t v7724;
          ap_int<128> v7724_tmp = v7714;
          v7724 = v7724_tmp(v7723, v7722);	// L11938
          ap_int<128> v7725 = packed_c128;	// L11939
          ap_int<128> v7726;
          ap_int<128> v7726_tmp = v7725;
          v7726_tmp(v7723, v7722) = v7724;
          v7726 = v7726_tmp;	// L11949
          packed_c128 = v7726;	// L11950
        }
      }
      ap_int<128> v7727 = packed_c128;	// L11953
      v7675.write(v7727);	// L11954
    }
  }
}

void gemm_9_2(
  hls::stream< int8_t >& v7728,
  hls::stream< int8_t >& v7729,
  hls::stream< int8_t >& v7730,
  hls::stream< int8_t >& v7731,
  hls::stream< ap_uint<128> >& v7732,
  hls::stream< ap_uint<128> >& v7733
) {	// L11959
  l_S_mt_nt_0_mt157: for (int mt157 = 0; mt157 < 4; mt157++) {	// L11967
    l_nt157: for (int nt157 = 0; nt157 < 4; nt157++) {	// L11968
      int8_t c129;	// L11969
      c129 = 0;	// L11970
      l_S_k_0_k157: for (int k157 = 0; k157 < 64; k157++) {	// L11971
        int8_t v7738 = v7728.read();	// L11972
        int8_t a138;	// L11973
        a138 = v7738;	// L11974
        int8_t v7740 = v7729.read();	// L11975
        int8_t b145;	// L11976
        b145 = v7740;	// L11977
        int8_t v7742 = a138;	// L11978
        int8_t v7743 = b145;	// L11979
        int16_t v7744 = v7742;	// L11980
        int16_t v7745 = v7743;	// L11981
        int16_t v7746 = v7744 * v7745;	// L11982
        int8_t v7747 = c129;	// L11983
        ap_int<17> v7748 = v7747;	// L11984
        ap_int<17> v7749 = v7746;	// L11985
        ap_int<17> v7750 = v7748 + v7749;	// L11986
        int8_t v7751 = v7750;	// L11987
        c129 = v7751;	// L11988
        int8_t v7752 = a138;	// L11989
        v7730.write(v7752);	// L11990
        int8_t v7753 = b145;	// L11991
        v7731.write(v7753);	// L11992
      }
      ap_uint<128> packed_tmp129;	// L11994
      ap_uint<128> v7755 = v7732.read();	// L11995
      packed_tmp129 = v7755;	// L11996
      ap_uint<128> packed_c129;	// L11997
      packed_c129 = 0;	// L11998
      l_S_m_1_m129: for (int m129 = 0; m129 < 16; m129++) {	// L11999
        ap_int<33> v7758 = m129;	// L12000
        bool v7759 = v7758 == 8;	// L12001
        if (v7759) {	// L12002
          int8_t v7760 = c129;	// L12003
          ap_int<128> v7761 = packed_c129;	// L12004
          int64_t v7762 = m129;	// L12005
          int64_t v7763 = v7762 * 8;	// L12006
          ap_int<34> v7764 = m129;	// L12007
          ap_int<34> v7765 = v7764 + 1;	// L12008
          ap_int<66> v7766 = v7765;	// L12009
          ap_int<66> v7767 = v7766 * 8;	// L12010
          ap_int<66> v7768 = v7767 - 1;	// L12011
          int v7769 = v7763;	// L12012
          int v7770 = v7768;	// L12013
          ap_int<128> v7771;
          ap_int<128> v7771_tmp = v7761;
          v7771_tmp(v7770, v7769) = v7760;
          v7771 = v7771_tmp;	// L12014
          packed_c129 = v7771;	// L12015
        } else {
          ap_int<128> v7772 = packed_tmp129;	// L12017
          int64_t v7773 = m129;	// L12018
          int64_t v7774 = v7773 * 8;	// L12019
          ap_int<34> v7775 = m129;	// L12020
          ap_int<34> v7776 = v7775 + 1;	// L12021
          ap_int<66> v7777 = v7776;	// L12022
          ap_int<66> v7778 = v7777 * 8;	// L12023
          ap_int<66> v7779 = v7778 - 1;	// L12024
          int v7780 = v7774;	// L12025
          int v7781 = v7779;	// L12026
          int8_t v7782;
          ap_int<128> v7782_tmp = v7772;
          v7782 = v7782_tmp(v7781, v7780);	// L12027
          ap_int<128> v7783 = packed_c129;	// L12028
          ap_int<128> v7784;
          ap_int<128> v7784_tmp = v7783;
          v7784_tmp(v7781, v7780) = v7782;
          v7784 = v7784_tmp;	// L12038
          packed_c129 = v7784;	// L12039
        }
      }
      ap_int<128> v7785 = packed_c129;	// L12042
      v7733.write(v7785);	// L12043
    }
  }
}

void gemm_9_3(
  hls::stream< int8_t >& v7786,
  hls::stream< int8_t >& v7787,
  hls::stream< int8_t >& v7788,
  hls::stream< int8_t >& v7789,
  hls::stream< ap_uint<128> >& v7790,
  hls::stream< ap_uint<128> >& v7791
) {	// L12048
  l_S_mt_nt_0_mt158: for (int mt158 = 0; mt158 < 4; mt158++) {	// L12056
    l_nt158: for (int nt158 = 0; nt158 < 4; nt158++) {	// L12057
      int8_t c130;	// L12058
      c130 = 0;	// L12059
      l_S_k_0_k158: for (int k158 = 0; k158 < 64; k158++) {	// L12060
        int8_t v7796 = v7786.read();	// L12061
        int8_t a139;	// L12062
        a139 = v7796;	// L12063
        int8_t v7798 = v7787.read();	// L12064
        int8_t b146;	// L12065
        b146 = v7798;	// L12066
        int8_t v7800 = a139;	// L12067
        int8_t v7801 = b146;	// L12068
        int16_t v7802 = v7800;	// L12069
        int16_t v7803 = v7801;	// L12070
        int16_t v7804 = v7802 * v7803;	// L12071
        int8_t v7805 = c130;	// L12072
        ap_int<17> v7806 = v7805;	// L12073
        ap_int<17> v7807 = v7804;	// L12074
        ap_int<17> v7808 = v7806 + v7807;	// L12075
        int8_t v7809 = v7808;	// L12076
        c130 = v7809;	// L12077
        int8_t v7810 = a139;	// L12078
        v7788.write(v7810);	// L12079
        int8_t v7811 = b146;	// L12080
        v7789.write(v7811);	// L12081
      }
      ap_uint<128> packed_tmp130;	// L12083
      ap_uint<128> v7813 = v7790.read();	// L12084
      packed_tmp130 = v7813;	// L12085
      ap_uint<128> packed_c130;	// L12086
      packed_c130 = 0;	// L12087
      l_S_m_1_m130: for (int m130 = 0; m130 < 16; m130++) {	// L12088
        ap_int<33> v7816 = m130;	// L12089
        bool v7817 = v7816 == 8;	// L12090
        if (v7817) {	// L12091
          int8_t v7818 = c130;	// L12092
          ap_int<128> v7819 = packed_c130;	// L12093
          int64_t v7820 = m130;	// L12094
          int64_t v7821 = v7820 * 8;	// L12095
          ap_int<34> v7822 = m130;	// L12096
          ap_int<34> v7823 = v7822 + 1;	// L12097
          ap_int<66> v7824 = v7823;	// L12098
          ap_int<66> v7825 = v7824 * 8;	// L12099
          ap_int<66> v7826 = v7825 - 1;	// L12100
          int v7827 = v7821;	// L12101
          int v7828 = v7826;	// L12102
          ap_int<128> v7829;
          ap_int<128> v7829_tmp = v7819;
          v7829_tmp(v7828, v7827) = v7818;
          v7829 = v7829_tmp;	// L12103
          packed_c130 = v7829;	// L12104
        } else {
          ap_int<128> v7830 = packed_tmp130;	// L12106
          int64_t v7831 = m130;	// L12107
          int64_t v7832 = v7831 * 8;	// L12108
          ap_int<34> v7833 = m130;	// L12109
          ap_int<34> v7834 = v7833 + 1;	// L12110
          ap_int<66> v7835 = v7834;	// L12111
          ap_int<66> v7836 = v7835 * 8;	// L12112
          ap_int<66> v7837 = v7836 - 1;	// L12113
          int v7838 = v7832;	// L12114
          int v7839 = v7837;	// L12115
          int8_t v7840;
          ap_int<128> v7840_tmp = v7830;
          v7840 = v7840_tmp(v7839, v7838);	// L12116
          ap_int<128> v7841 = packed_c130;	// L12117
          ap_int<128> v7842;
          ap_int<128> v7842_tmp = v7841;
          v7842_tmp(v7839, v7838) = v7840;
          v7842 = v7842_tmp;	// L12127
          packed_c130 = v7842;	// L12128
        }
      }
      ap_int<128> v7843 = packed_c130;	// L12131
      v7791.write(v7843);	// L12132
    }
  }
}

void gemm_9_4(
  hls::stream< int8_t >& v7844,
  hls::stream< int8_t >& v7845,
  hls::stream< int8_t >& v7846,
  hls::stream< int8_t >& v7847,
  hls::stream< ap_uint<128> >& v7848,
  hls::stream< ap_uint<128> >& v7849
) {	// L12137
  l_S_mt_nt_0_mt159: for (int mt159 = 0; mt159 < 4; mt159++) {	// L12145
    l_nt159: for (int nt159 = 0; nt159 < 4; nt159++) {	// L12146
      int8_t c131;	// L12147
      c131 = 0;	// L12148
      l_S_k_0_k159: for (int k159 = 0; k159 < 64; k159++) {	// L12149
        int8_t v7854 = v7844.read();	// L12150
        int8_t a140;	// L12151
        a140 = v7854;	// L12152
        int8_t v7856 = v7845.read();	// L12153
        int8_t b147;	// L12154
        b147 = v7856;	// L12155
        int8_t v7858 = a140;	// L12156
        int8_t v7859 = b147;	// L12157
        int16_t v7860 = v7858;	// L12158
        int16_t v7861 = v7859;	// L12159
        int16_t v7862 = v7860 * v7861;	// L12160
        int8_t v7863 = c131;	// L12161
        ap_int<17> v7864 = v7863;	// L12162
        ap_int<17> v7865 = v7862;	// L12163
        ap_int<17> v7866 = v7864 + v7865;	// L12164
        int8_t v7867 = v7866;	// L12165
        c131 = v7867;	// L12166
        int8_t v7868 = a140;	// L12167
        v7846.write(v7868);	// L12168
        int8_t v7869 = b147;	// L12169
        v7847.write(v7869);	// L12170
      }
      ap_uint<128> packed_tmp131;	// L12172
      ap_uint<128> v7871 = v7848.read();	// L12173
      packed_tmp131 = v7871;	// L12174
      ap_uint<128> packed_c131;	// L12175
      packed_c131 = 0;	// L12176
      l_S_m_1_m131: for (int m131 = 0; m131 < 16; m131++) {	// L12177
        ap_int<33> v7874 = m131;	// L12178
        bool v7875 = v7874 == 8;	// L12179
        if (v7875) {	// L12180
          int8_t v7876 = c131;	// L12181
          ap_int<128> v7877 = packed_c131;	// L12182
          int64_t v7878 = m131;	// L12183
          int64_t v7879 = v7878 * 8;	// L12184
          ap_int<34> v7880 = m131;	// L12185
          ap_int<34> v7881 = v7880 + 1;	// L12186
          ap_int<66> v7882 = v7881;	// L12187
          ap_int<66> v7883 = v7882 * 8;	// L12188
          ap_int<66> v7884 = v7883 - 1;	// L12189
          int v7885 = v7879;	// L12190
          int v7886 = v7884;	// L12191
          ap_int<128> v7887;
          ap_int<128> v7887_tmp = v7877;
          v7887_tmp(v7886, v7885) = v7876;
          v7887 = v7887_tmp;	// L12192
          packed_c131 = v7887;	// L12193
        } else {
          ap_int<128> v7888 = packed_tmp131;	// L12195
          int64_t v7889 = m131;	// L12196
          int64_t v7890 = v7889 * 8;	// L12197
          ap_int<34> v7891 = m131;	// L12198
          ap_int<34> v7892 = v7891 + 1;	// L12199
          ap_int<66> v7893 = v7892;	// L12200
          ap_int<66> v7894 = v7893 * 8;	// L12201
          ap_int<66> v7895 = v7894 - 1;	// L12202
          int v7896 = v7890;	// L12203
          int v7897 = v7895;	// L12204
          int8_t v7898;
          ap_int<128> v7898_tmp = v7888;
          v7898 = v7898_tmp(v7897, v7896);	// L12205
          ap_int<128> v7899 = packed_c131;	// L12206
          ap_int<128> v7900;
          ap_int<128> v7900_tmp = v7899;
          v7900_tmp(v7897, v7896) = v7898;
          v7900 = v7900_tmp;	// L12216
          packed_c131 = v7900;	// L12217
        }
      }
      ap_int<128> v7901 = packed_c131;	// L12220
      v7849.write(v7901);	// L12221
    }
  }
}

void gemm_9_5(
  hls::stream< int8_t >& v7902,
  hls::stream< int8_t >& v7903,
  hls::stream< int8_t >& v7904,
  hls::stream< int8_t >& v7905,
  hls::stream< ap_uint<128> >& v7906,
  hls::stream< ap_uint<128> >& v7907
) {	// L12226
  l_S_mt_nt_0_mt160: for (int mt160 = 0; mt160 < 4; mt160++) {	// L12234
    l_nt160: for (int nt160 = 0; nt160 < 4; nt160++) {	// L12235
      int8_t c132;	// L12236
      c132 = 0;	// L12237
      l_S_k_0_k160: for (int k160 = 0; k160 < 64; k160++) {	// L12238
        int8_t v7912 = v7902.read();	// L12239
        int8_t a141;	// L12240
        a141 = v7912;	// L12241
        int8_t v7914 = v7903.read();	// L12242
        int8_t b148;	// L12243
        b148 = v7914;	// L12244
        int8_t v7916 = a141;	// L12245
        int8_t v7917 = b148;	// L12246
        int16_t v7918 = v7916;	// L12247
        int16_t v7919 = v7917;	// L12248
        int16_t v7920 = v7918 * v7919;	// L12249
        int8_t v7921 = c132;	// L12250
        ap_int<17> v7922 = v7921;	// L12251
        ap_int<17> v7923 = v7920;	// L12252
        ap_int<17> v7924 = v7922 + v7923;	// L12253
        int8_t v7925 = v7924;	// L12254
        c132 = v7925;	// L12255
        int8_t v7926 = a141;	// L12256
        v7904.write(v7926);	// L12257
        int8_t v7927 = b148;	// L12258
        v7905.write(v7927);	// L12259
      }
      ap_uint<128> packed_tmp132;	// L12261
      ap_uint<128> v7929 = v7906.read();	// L12262
      packed_tmp132 = v7929;	// L12263
      ap_uint<128> packed_c132;	// L12264
      packed_c132 = 0;	// L12265
      l_S_m_1_m132: for (int m132 = 0; m132 < 16; m132++) {	// L12266
        ap_int<33> v7932 = m132;	// L12267
        bool v7933 = v7932 == 8;	// L12268
        if (v7933) {	// L12269
          int8_t v7934 = c132;	// L12270
          ap_int<128> v7935 = packed_c132;	// L12271
          int64_t v7936 = m132;	// L12272
          int64_t v7937 = v7936 * 8;	// L12273
          ap_int<34> v7938 = m132;	// L12274
          ap_int<34> v7939 = v7938 + 1;	// L12275
          ap_int<66> v7940 = v7939;	// L12276
          ap_int<66> v7941 = v7940 * 8;	// L12277
          ap_int<66> v7942 = v7941 - 1;	// L12278
          int v7943 = v7937;	// L12279
          int v7944 = v7942;	// L12280
          ap_int<128> v7945;
          ap_int<128> v7945_tmp = v7935;
          v7945_tmp(v7944, v7943) = v7934;
          v7945 = v7945_tmp;	// L12281
          packed_c132 = v7945;	// L12282
        } else {
          ap_int<128> v7946 = packed_tmp132;	// L12284
          int64_t v7947 = m132;	// L12285
          int64_t v7948 = v7947 * 8;	// L12286
          ap_int<34> v7949 = m132;	// L12287
          ap_int<34> v7950 = v7949 + 1;	// L12288
          ap_int<66> v7951 = v7950;	// L12289
          ap_int<66> v7952 = v7951 * 8;	// L12290
          ap_int<66> v7953 = v7952 - 1;	// L12291
          int v7954 = v7948;	// L12292
          int v7955 = v7953;	// L12293
          int8_t v7956;
          ap_int<128> v7956_tmp = v7946;
          v7956 = v7956_tmp(v7955, v7954);	// L12294
          ap_int<128> v7957 = packed_c132;	// L12295
          ap_int<128> v7958;
          ap_int<128> v7958_tmp = v7957;
          v7958_tmp(v7955, v7954) = v7956;
          v7958 = v7958_tmp;	// L12305
          packed_c132 = v7958;	// L12306
        }
      }
      ap_int<128> v7959 = packed_c132;	// L12309
      v7907.write(v7959);	// L12310
    }
  }
}

void gemm_9_6(
  hls::stream< int8_t >& v7960,
  hls::stream< int8_t >& v7961,
  hls::stream< int8_t >& v7962,
  hls::stream< int8_t >& v7963,
  hls::stream< ap_uint<128> >& v7964,
  hls::stream< ap_uint<128> >& v7965
) {	// L12315
  l_S_mt_nt_0_mt161: for (int mt161 = 0; mt161 < 4; mt161++) {	// L12323
    l_nt161: for (int nt161 = 0; nt161 < 4; nt161++) {	// L12324
      int8_t c133;	// L12325
      c133 = 0;	// L12326
      l_S_k_0_k161: for (int k161 = 0; k161 < 64; k161++) {	// L12327
        int8_t v7970 = v7960.read();	// L12328
        int8_t a142;	// L12329
        a142 = v7970;	// L12330
        int8_t v7972 = v7961.read();	// L12331
        int8_t b149;	// L12332
        b149 = v7972;	// L12333
        int8_t v7974 = a142;	// L12334
        int8_t v7975 = b149;	// L12335
        int16_t v7976 = v7974;	// L12336
        int16_t v7977 = v7975;	// L12337
        int16_t v7978 = v7976 * v7977;	// L12338
        int8_t v7979 = c133;	// L12339
        ap_int<17> v7980 = v7979;	// L12340
        ap_int<17> v7981 = v7978;	// L12341
        ap_int<17> v7982 = v7980 + v7981;	// L12342
        int8_t v7983 = v7982;	// L12343
        c133 = v7983;	// L12344
        int8_t v7984 = a142;	// L12345
        v7962.write(v7984);	// L12346
        int8_t v7985 = b149;	// L12347
        v7963.write(v7985);	// L12348
      }
      ap_uint<128> packed_tmp133;	// L12350
      ap_uint<128> v7987 = v7964.read();	// L12351
      packed_tmp133 = v7987;	// L12352
      ap_uint<128> packed_c133;	// L12353
      packed_c133 = 0;	// L12354
      l_S_m_1_m133: for (int m133 = 0; m133 < 16; m133++) {	// L12355
        ap_int<33> v7990 = m133;	// L12356
        bool v7991 = v7990 == 8;	// L12357
        if (v7991) {	// L12358
          int8_t v7992 = c133;	// L12359
          ap_int<128> v7993 = packed_c133;	// L12360
          int64_t v7994 = m133;	// L12361
          int64_t v7995 = v7994 * 8;	// L12362
          ap_int<34> v7996 = m133;	// L12363
          ap_int<34> v7997 = v7996 + 1;	// L12364
          ap_int<66> v7998 = v7997;	// L12365
          ap_int<66> v7999 = v7998 * 8;	// L12366
          ap_int<66> v8000 = v7999 - 1;	// L12367
          int v8001 = v7995;	// L12368
          int v8002 = v8000;	// L12369
          ap_int<128> v8003;
          ap_int<128> v8003_tmp = v7993;
          v8003_tmp(v8002, v8001) = v7992;
          v8003 = v8003_tmp;	// L12370
          packed_c133 = v8003;	// L12371
        } else {
          ap_int<128> v8004 = packed_tmp133;	// L12373
          int64_t v8005 = m133;	// L12374
          int64_t v8006 = v8005 * 8;	// L12375
          ap_int<34> v8007 = m133;	// L12376
          ap_int<34> v8008 = v8007 + 1;	// L12377
          ap_int<66> v8009 = v8008;	// L12378
          ap_int<66> v8010 = v8009 * 8;	// L12379
          ap_int<66> v8011 = v8010 - 1;	// L12380
          int v8012 = v8006;	// L12381
          int v8013 = v8011;	// L12382
          int8_t v8014;
          ap_int<128> v8014_tmp = v8004;
          v8014 = v8014_tmp(v8013, v8012);	// L12383
          ap_int<128> v8015 = packed_c133;	// L12384
          ap_int<128> v8016;
          ap_int<128> v8016_tmp = v8015;
          v8016_tmp(v8013, v8012) = v8014;
          v8016 = v8016_tmp;	// L12394
          packed_c133 = v8016;	// L12395
        }
      }
      ap_int<128> v8017 = packed_c133;	// L12398
      v7965.write(v8017);	// L12399
    }
  }
}

void gemm_9_7(
  hls::stream< int8_t >& v8018,
  hls::stream< int8_t >& v8019,
  hls::stream< int8_t >& v8020,
  hls::stream< int8_t >& v8021,
  hls::stream< ap_uint<128> >& v8022,
  hls::stream< ap_uint<128> >& v8023
) {	// L12404
  l_S_mt_nt_0_mt162: for (int mt162 = 0; mt162 < 4; mt162++) {	// L12412
    l_nt162: for (int nt162 = 0; nt162 < 4; nt162++) {	// L12413
      int8_t c134;	// L12414
      c134 = 0;	// L12415
      l_S_k_0_k162: for (int k162 = 0; k162 < 64; k162++) {	// L12416
        int8_t v8028 = v8018.read();	// L12417
        int8_t a143;	// L12418
        a143 = v8028;	// L12419
        int8_t v8030 = v8019.read();	// L12420
        int8_t b150;	// L12421
        b150 = v8030;	// L12422
        int8_t v8032 = a143;	// L12423
        int8_t v8033 = b150;	// L12424
        int16_t v8034 = v8032;	// L12425
        int16_t v8035 = v8033;	// L12426
        int16_t v8036 = v8034 * v8035;	// L12427
        int8_t v8037 = c134;	// L12428
        ap_int<17> v8038 = v8037;	// L12429
        ap_int<17> v8039 = v8036;	// L12430
        ap_int<17> v8040 = v8038 + v8039;	// L12431
        int8_t v8041 = v8040;	// L12432
        c134 = v8041;	// L12433
        int8_t v8042 = a143;	// L12434
        v8020.write(v8042);	// L12435
        int8_t v8043 = b150;	// L12436
        v8021.write(v8043);	// L12437
      }
      ap_uint<128> packed_tmp134;	// L12439
      ap_uint<128> v8045 = v8022.read();	// L12440
      packed_tmp134 = v8045;	// L12441
      ap_uint<128> packed_c134;	// L12442
      packed_c134 = 0;	// L12443
      l_S_m_1_m134: for (int m134 = 0; m134 < 16; m134++) {	// L12444
        ap_int<33> v8048 = m134;	// L12445
        bool v8049 = v8048 == 8;	// L12446
        if (v8049) {	// L12447
          int8_t v8050 = c134;	// L12448
          ap_int<128> v8051 = packed_c134;	// L12449
          int64_t v8052 = m134;	// L12450
          int64_t v8053 = v8052 * 8;	// L12451
          ap_int<34> v8054 = m134;	// L12452
          ap_int<34> v8055 = v8054 + 1;	// L12453
          ap_int<66> v8056 = v8055;	// L12454
          ap_int<66> v8057 = v8056 * 8;	// L12455
          ap_int<66> v8058 = v8057 - 1;	// L12456
          int v8059 = v8053;	// L12457
          int v8060 = v8058;	// L12458
          ap_int<128> v8061;
          ap_int<128> v8061_tmp = v8051;
          v8061_tmp(v8060, v8059) = v8050;
          v8061 = v8061_tmp;	// L12459
          packed_c134 = v8061;	// L12460
        } else {
          ap_int<128> v8062 = packed_tmp134;	// L12462
          int64_t v8063 = m134;	// L12463
          int64_t v8064 = v8063 * 8;	// L12464
          ap_int<34> v8065 = m134;	// L12465
          ap_int<34> v8066 = v8065 + 1;	// L12466
          ap_int<66> v8067 = v8066;	// L12467
          ap_int<66> v8068 = v8067 * 8;	// L12468
          ap_int<66> v8069 = v8068 - 1;	// L12469
          int v8070 = v8064;	// L12470
          int v8071 = v8069;	// L12471
          int8_t v8072;
          ap_int<128> v8072_tmp = v8062;
          v8072 = v8072_tmp(v8071, v8070);	// L12472
          ap_int<128> v8073 = packed_c134;	// L12473
          ap_int<128> v8074;
          ap_int<128> v8074_tmp = v8073;
          v8074_tmp(v8071, v8070) = v8072;
          v8074 = v8074_tmp;	// L12483
          packed_c134 = v8074;	// L12484
        }
      }
      ap_int<128> v8075 = packed_c134;	// L12487
      v8023.write(v8075);	// L12488
    }
  }
}

void gemm_9_8(
  hls::stream< int8_t >& v8076,
  hls::stream< int8_t >& v8077,
  hls::stream< int8_t >& v8078,
  hls::stream< int8_t >& v8079,
  hls::stream< ap_uint<128> >& v8080,
  hls::stream< ap_uint<128> >& v8081
) {	// L12493
  l_S_mt_nt_0_mt163: for (int mt163 = 0; mt163 < 4; mt163++) {	// L12501
    l_nt163: for (int nt163 = 0; nt163 < 4; nt163++) {	// L12502
      int8_t c135;	// L12503
      c135 = 0;	// L12504
      l_S_k_0_k163: for (int k163 = 0; k163 < 64; k163++) {	// L12505
        int8_t v8086 = v8076.read();	// L12506
        int8_t a144;	// L12507
        a144 = v8086;	// L12508
        int8_t v8088 = v8077.read();	// L12509
        int8_t b151;	// L12510
        b151 = v8088;	// L12511
        int8_t v8090 = a144;	// L12512
        int8_t v8091 = b151;	// L12513
        int16_t v8092 = v8090;	// L12514
        int16_t v8093 = v8091;	// L12515
        int16_t v8094 = v8092 * v8093;	// L12516
        int8_t v8095 = c135;	// L12517
        ap_int<17> v8096 = v8095;	// L12518
        ap_int<17> v8097 = v8094;	// L12519
        ap_int<17> v8098 = v8096 + v8097;	// L12520
        int8_t v8099 = v8098;	// L12521
        c135 = v8099;	// L12522
        int8_t v8100 = a144;	// L12523
        v8078.write(v8100);	// L12524
        int8_t v8101 = b151;	// L12525
        v8079.write(v8101);	// L12526
      }
      ap_uint<128> packed_tmp135;	// L12528
      ap_uint<128> v8103 = v8080.read();	// L12529
      packed_tmp135 = v8103;	// L12530
      ap_uint<128> packed_c135;	// L12531
      packed_c135 = 0;	// L12532
      l_S_m_1_m135: for (int m135 = 0; m135 < 16; m135++) {	// L12533
        ap_int<33> v8106 = m135;	// L12534
        bool v8107 = v8106 == 8;	// L12535
        if (v8107) {	// L12536
          int8_t v8108 = c135;	// L12537
          ap_int<128> v8109 = packed_c135;	// L12538
          int64_t v8110 = m135;	// L12539
          int64_t v8111 = v8110 * 8;	// L12540
          ap_int<34> v8112 = m135;	// L12541
          ap_int<34> v8113 = v8112 + 1;	// L12542
          ap_int<66> v8114 = v8113;	// L12543
          ap_int<66> v8115 = v8114 * 8;	// L12544
          ap_int<66> v8116 = v8115 - 1;	// L12545
          int v8117 = v8111;	// L12546
          int v8118 = v8116;	// L12547
          ap_int<128> v8119;
          ap_int<128> v8119_tmp = v8109;
          v8119_tmp(v8118, v8117) = v8108;
          v8119 = v8119_tmp;	// L12548
          packed_c135 = v8119;	// L12549
        } else {
          ap_int<128> v8120 = packed_tmp135;	// L12551
          int64_t v8121 = m135;	// L12552
          int64_t v8122 = v8121 * 8;	// L12553
          ap_int<34> v8123 = m135;	// L12554
          ap_int<34> v8124 = v8123 + 1;	// L12555
          ap_int<66> v8125 = v8124;	// L12556
          ap_int<66> v8126 = v8125 * 8;	// L12557
          ap_int<66> v8127 = v8126 - 1;	// L12558
          int v8128 = v8122;	// L12559
          int v8129 = v8127;	// L12560
          int8_t v8130;
          ap_int<128> v8130_tmp = v8120;
          v8130 = v8130_tmp(v8129, v8128);	// L12561
          ap_int<128> v8131 = packed_c135;	// L12562
          ap_int<128> v8132;
          ap_int<128> v8132_tmp = v8131;
          v8132_tmp(v8129, v8128) = v8130;
          v8132 = v8132_tmp;	// L12572
          packed_c135 = v8132;	// L12573
        }
      }
      ap_int<128> v8133 = packed_c135;	// L12576
      v8081.write(v8133);	// L12577
    }
  }
}

void gemm_9_9(
  hls::stream< int8_t >& v8134,
  hls::stream< int8_t >& v8135,
  hls::stream< int8_t >& v8136,
  hls::stream< int8_t >& v8137,
  hls::stream< ap_uint<128> >& v8138,
  hls::stream< ap_uint<128> >& v8139
) {	// L12582
  l_S_mt_nt_0_mt164: for (int mt164 = 0; mt164 < 4; mt164++) {	// L12590
    l_nt164: for (int nt164 = 0; nt164 < 4; nt164++) {	// L12591
      int8_t c136;	// L12592
      c136 = 0;	// L12593
      l_S_k_0_k164: for (int k164 = 0; k164 < 64; k164++) {	// L12594
        int8_t v8144 = v8134.read();	// L12595
        int8_t a145;	// L12596
        a145 = v8144;	// L12597
        int8_t v8146 = v8135.read();	// L12598
        int8_t b152;	// L12599
        b152 = v8146;	// L12600
        int8_t v8148 = a145;	// L12601
        int8_t v8149 = b152;	// L12602
        int16_t v8150 = v8148;	// L12603
        int16_t v8151 = v8149;	// L12604
        int16_t v8152 = v8150 * v8151;	// L12605
        int8_t v8153 = c136;	// L12606
        ap_int<17> v8154 = v8153;	// L12607
        ap_int<17> v8155 = v8152;	// L12608
        ap_int<17> v8156 = v8154 + v8155;	// L12609
        int8_t v8157 = v8156;	// L12610
        c136 = v8157;	// L12611
        int8_t v8158 = a145;	// L12612
        v8136.write(v8158);	// L12613
        int8_t v8159 = b152;	// L12614
        v8137.write(v8159);	// L12615
      }
      ap_uint<128> packed_tmp136;	// L12617
      ap_uint<128> v8161 = v8138.read();	// L12618
      packed_tmp136 = v8161;	// L12619
      ap_uint<128> packed_c136;	// L12620
      packed_c136 = 0;	// L12621
      l_S_m_1_m136: for (int m136 = 0; m136 < 16; m136++) {	// L12622
        ap_int<33> v8164 = m136;	// L12623
        bool v8165 = v8164 == 8;	// L12624
        if (v8165) {	// L12625
          int8_t v8166 = c136;	// L12626
          ap_int<128> v8167 = packed_c136;	// L12627
          int64_t v8168 = m136;	// L12628
          int64_t v8169 = v8168 * 8;	// L12629
          ap_int<34> v8170 = m136;	// L12630
          ap_int<34> v8171 = v8170 + 1;	// L12631
          ap_int<66> v8172 = v8171;	// L12632
          ap_int<66> v8173 = v8172 * 8;	// L12633
          ap_int<66> v8174 = v8173 - 1;	// L12634
          int v8175 = v8169;	// L12635
          int v8176 = v8174;	// L12636
          ap_int<128> v8177;
          ap_int<128> v8177_tmp = v8167;
          v8177_tmp(v8176, v8175) = v8166;
          v8177 = v8177_tmp;	// L12637
          packed_c136 = v8177;	// L12638
        } else {
          ap_int<128> v8178 = packed_tmp136;	// L12640
          int64_t v8179 = m136;	// L12641
          int64_t v8180 = v8179 * 8;	// L12642
          ap_int<34> v8181 = m136;	// L12643
          ap_int<34> v8182 = v8181 + 1;	// L12644
          ap_int<66> v8183 = v8182;	// L12645
          ap_int<66> v8184 = v8183 * 8;	// L12646
          ap_int<66> v8185 = v8184 - 1;	// L12647
          int v8186 = v8180;	// L12648
          int v8187 = v8185;	// L12649
          int8_t v8188;
          ap_int<128> v8188_tmp = v8178;
          v8188 = v8188_tmp(v8187, v8186);	// L12650
          ap_int<128> v8189 = packed_c136;	// L12651
          ap_int<128> v8190;
          ap_int<128> v8190_tmp = v8189;
          v8190_tmp(v8187, v8186) = v8188;
          v8190 = v8190_tmp;	// L12661
          packed_c136 = v8190;	// L12662
        }
      }
      ap_int<128> v8191 = packed_c136;	// L12665
      v8139.write(v8191);	// L12666
    }
  }
}

void gemm_9_10(
  hls::stream< int8_t >& v8192,
  hls::stream< int8_t >& v8193,
  hls::stream< int8_t >& v8194,
  hls::stream< int8_t >& v8195,
  hls::stream< ap_uint<128> >& v8196,
  hls::stream< ap_uint<128> >& v8197
) {	// L12671
  l_S_mt_nt_0_mt165: for (int mt165 = 0; mt165 < 4; mt165++) {	// L12679
    l_nt165: for (int nt165 = 0; nt165 < 4; nt165++) {	// L12680
      int8_t c137;	// L12681
      c137 = 0;	// L12682
      l_S_k_0_k165: for (int k165 = 0; k165 < 64; k165++) {	// L12683
        int8_t v8202 = v8192.read();	// L12684
        int8_t a146;	// L12685
        a146 = v8202;	// L12686
        int8_t v8204 = v8193.read();	// L12687
        int8_t b153;	// L12688
        b153 = v8204;	// L12689
        int8_t v8206 = a146;	// L12690
        int8_t v8207 = b153;	// L12691
        int16_t v8208 = v8206;	// L12692
        int16_t v8209 = v8207;	// L12693
        int16_t v8210 = v8208 * v8209;	// L12694
        int8_t v8211 = c137;	// L12695
        ap_int<17> v8212 = v8211;	// L12696
        ap_int<17> v8213 = v8210;	// L12697
        ap_int<17> v8214 = v8212 + v8213;	// L12698
        int8_t v8215 = v8214;	// L12699
        c137 = v8215;	// L12700
        int8_t v8216 = a146;	// L12701
        v8194.write(v8216);	// L12702
        int8_t v8217 = b153;	// L12703
        v8195.write(v8217);	// L12704
      }
      ap_uint<128> packed_tmp137;	// L12706
      ap_uint<128> v8219 = v8196.read();	// L12707
      packed_tmp137 = v8219;	// L12708
      ap_uint<128> packed_c137;	// L12709
      packed_c137 = 0;	// L12710
      l_S_m_1_m137: for (int m137 = 0; m137 < 16; m137++) {	// L12711
        ap_int<33> v8222 = m137;	// L12712
        bool v8223 = v8222 == 8;	// L12713
        if (v8223) {	// L12714
          int8_t v8224 = c137;	// L12715
          ap_int<128> v8225 = packed_c137;	// L12716
          int64_t v8226 = m137;	// L12717
          int64_t v8227 = v8226 * 8;	// L12718
          ap_int<34> v8228 = m137;	// L12719
          ap_int<34> v8229 = v8228 + 1;	// L12720
          ap_int<66> v8230 = v8229;	// L12721
          ap_int<66> v8231 = v8230 * 8;	// L12722
          ap_int<66> v8232 = v8231 - 1;	// L12723
          int v8233 = v8227;	// L12724
          int v8234 = v8232;	// L12725
          ap_int<128> v8235;
          ap_int<128> v8235_tmp = v8225;
          v8235_tmp(v8234, v8233) = v8224;
          v8235 = v8235_tmp;	// L12726
          packed_c137 = v8235;	// L12727
        } else {
          ap_int<128> v8236 = packed_tmp137;	// L12729
          int64_t v8237 = m137;	// L12730
          int64_t v8238 = v8237 * 8;	// L12731
          ap_int<34> v8239 = m137;	// L12732
          ap_int<34> v8240 = v8239 + 1;	// L12733
          ap_int<66> v8241 = v8240;	// L12734
          ap_int<66> v8242 = v8241 * 8;	// L12735
          ap_int<66> v8243 = v8242 - 1;	// L12736
          int v8244 = v8238;	// L12737
          int v8245 = v8243;	// L12738
          int8_t v8246;
          ap_int<128> v8246_tmp = v8236;
          v8246 = v8246_tmp(v8245, v8244);	// L12739
          ap_int<128> v8247 = packed_c137;	// L12740
          ap_int<128> v8248;
          ap_int<128> v8248_tmp = v8247;
          v8248_tmp(v8245, v8244) = v8246;
          v8248 = v8248_tmp;	// L12750
          packed_c137 = v8248;	// L12751
        }
      }
      ap_int<128> v8249 = packed_c137;	// L12754
      v8197.write(v8249);	// L12755
    }
  }
}

void gemm_9_11(
  hls::stream< int8_t >& v8250,
  hls::stream< int8_t >& v8251,
  hls::stream< int8_t >& v8252,
  hls::stream< int8_t >& v8253,
  hls::stream< ap_uint<128> >& v8254,
  hls::stream< ap_uint<128> >& v8255
) {	// L12760
  l_S_mt_nt_0_mt166: for (int mt166 = 0; mt166 < 4; mt166++) {	// L12768
    l_nt166: for (int nt166 = 0; nt166 < 4; nt166++) {	// L12769
      int8_t c138;	// L12770
      c138 = 0;	// L12771
      l_S_k_0_k166: for (int k166 = 0; k166 < 64; k166++) {	// L12772
        int8_t v8260 = v8250.read();	// L12773
        int8_t a147;	// L12774
        a147 = v8260;	// L12775
        int8_t v8262 = v8251.read();	// L12776
        int8_t b154;	// L12777
        b154 = v8262;	// L12778
        int8_t v8264 = a147;	// L12779
        int8_t v8265 = b154;	// L12780
        int16_t v8266 = v8264;	// L12781
        int16_t v8267 = v8265;	// L12782
        int16_t v8268 = v8266 * v8267;	// L12783
        int8_t v8269 = c138;	// L12784
        ap_int<17> v8270 = v8269;	// L12785
        ap_int<17> v8271 = v8268;	// L12786
        ap_int<17> v8272 = v8270 + v8271;	// L12787
        int8_t v8273 = v8272;	// L12788
        c138 = v8273;	// L12789
        int8_t v8274 = a147;	// L12790
        v8252.write(v8274);	// L12791
        int8_t v8275 = b154;	// L12792
        v8253.write(v8275);	// L12793
      }
      ap_uint<128> packed_tmp138;	// L12795
      ap_uint<128> v8277 = v8254.read();	// L12796
      packed_tmp138 = v8277;	// L12797
      ap_uint<128> packed_c138;	// L12798
      packed_c138 = 0;	// L12799
      l_S_m_1_m138: for (int m138 = 0; m138 < 16; m138++) {	// L12800
        ap_int<33> v8280 = m138;	// L12801
        bool v8281 = v8280 == 8;	// L12802
        if (v8281) {	// L12803
          int8_t v8282 = c138;	// L12804
          ap_int<128> v8283 = packed_c138;	// L12805
          int64_t v8284 = m138;	// L12806
          int64_t v8285 = v8284 * 8;	// L12807
          ap_int<34> v8286 = m138;	// L12808
          ap_int<34> v8287 = v8286 + 1;	// L12809
          ap_int<66> v8288 = v8287;	// L12810
          ap_int<66> v8289 = v8288 * 8;	// L12811
          ap_int<66> v8290 = v8289 - 1;	// L12812
          int v8291 = v8285;	// L12813
          int v8292 = v8290;	// L12814
          ap_int<128> v8293;
          ap_int<128> v8293_tmp = v8283;
          v8293_tmp(v8292, v8291) = v8282;
          v8293 = v8293_tmp;	// L12815
          packed_c138 = v8293;	// L12816
        } else {
          ap_int<128> v8294 = packed_tmp138;	// L12818
          int64_t v8295 = m138;	// L12819
          int64_t v8296 = v8295 * 8;	// L12820
          ap_int<34> v8297 = m138;	// L12821
          ap_int<34> v8298 = v8297 + 1;	// L12822
          ap_int<66> v8299 = v8298;	// L12823
          ap_int<66> v8300 = v8299 * 8;	// L12824
          ap_int<66> v8301 = v8300 - 1;	// L12825
          int v8302 = v8296;	// L12826
          int v8303 = v8301;	// L12827
          int8_t v8304;
          ap_int<128> v8304_tmp = v8294;
          v8304 = v8304_tmp(v8303, v8302);	// L12828
          ap_int<128> v8305 = packed_c138;	// L12829
          ap_int<128> v8306;
          ap_int<128> v8306_tmp = v8305;
          v8306_tmp(v8303, v8302) = v8304;
          v8306 = v8306_tmp;	// L12839
          packed_c138 = v8306;	// L12840
        }
      }
      ap_int<128> v8307 = packed_c138;	// L12843
      v8255.write(v8307);	// L12844
    }
  }
}

void gemm_9_12(
  hls::stream< int8_t >& v8308,
  hls::stream< int8_t >& v8309,
  hls::stream< int8_t >& v8310,
  hls::stream< int8_t >& v8311,
  hls::stream< ap_uint<128> >& v8312,
  hls::stream< ap_uint<128> >& v8313
) {	// L12849
  l_S_mt_nt_0_mt167: for (int mt167 = 0; mt167 < 4; mt167++) {	// L12857
    l_nt167: for (int nt167 = 0; nt167 < 4; nt167++) {	// L12858
      int8_t c139;	// L12859
      c139 = 0;	// L12860
      l_S_k_0_k167: for (int k167 = 0; k167 < 64; k167++) {	// L12861
        int8_t v8318 = v8308.read();	// L12862
        int8_t a148;	// L12863
        a148 = v8318;	// L12864
        int8_t v8320 = v8309.read();	// L12865
        int8_t b155;	// L12866
        b155 = v8320;	// L12867
        int8_t v8322 = a148;	// L12868
        int8_t v8323 = b155;	// L12869
        int16_t v8324 = v8322;	// L12870
        int16_t v8325 = v8323;	// L12871
        int16_t v8326 = v8324 * v8325;	// L12872
        int8_t v8327 = c139;	// L12873
        ap_int<17> v8328 = v8327;	// L12874
        ap_int<17> v8329 = v8326;	// L12875
        ap_int<17> v8330 = v8328 + v8329;	// L12876
        int8_t v8331 = v8330;	// L12877
        c139 = v8331;	// L12878
        int8_t v8332 = a148;	// L12879
        v8310.write(v8332);	// L12880
        int8_t v8333 = b155;	// L12881
        v8311.write(v8333);	// L12882
      }
      ap_uint<128> packed_tmp139;	// L12884
      ap_uint<128> v8335 = v8312.read();	// L12885
      packed_tmp139 = v8335;	// L12886
      ap_uint<128> packed_c139;	// L12887
      packed_c139 = 0;	// L12888
      l_S_m_1_m139: for (int m139 = 0; m139 < 16; m139++) {	// L12889
        ap_int<33> v8338 = m139;	// L12890
        bool v8339 = v8338 == 8;	// L12891
        if (v8339) {	// L12892
          int8_t v8340 = c139;	// L12893
          ap_int<128> v8341 = packed_c139;	// L12894
          int64_t v8342 = m139;	// L12895
          int64_t v8343 = v8342 * 8;	// L12896
          ap_int<34> v8344 = m139;	// L12897
          ap_int<34> v8345 = v8344 + 1;	// L12898
          ap_int<66> v8346 = v8345;	// L12899
          ap_int<66> v8347 = v8346 * 8;	// L12900
          ap_int<66> v8348 = v8347 - 1;	// L12901
          int v8349 = v8343;	// L12902
          int v8350 = v8348;	// L12903
          ap_int<128> v8351;
          ap_int<128> v8351_tmp = v8341;
          v8351_tmp(v8350, v8349) = v8340;
          v8351 = v8351_tmp;	// L12904
          packed_c139 = v8351;	// L12905
        } else {
          ap_int<128> v8352 = packed_tmp139;	// L12907
          int64_t v8353 = m139;	// L12908
          int64_t v8354 = v8353 * 8;	// L12909
          ap_int<34> v8355 = m139;	// L12910
          ap_int<34> v8356 = v8355 + 1;	// L12911
          ap_int<66> v8357 = v8356;	// L12912
          ap_int<66> v8358 = v8357 * 8;	// L12913
          ap_int<66> v8359 = v8358 - 1;	// L12914
          int v8360 = v8354;	// L12915
          int v8361 = v8359;	// L12916
          int8_t v8362;
          ap_int<128> v8362_tmp = v8352;
          v8362 = v8362_tmp(v8361, v8360);	// L12917
          ap_int<128> v8363 = packed_c139;	// L12918
          ap_int<128> v8364;
          ap_int<128> v8364_tmp = v8363;
          v8364_tmp(v8361, v8360) = v8362;
          v8364 = v8364_tmp;	// L12928
          packed_c139 = v8364;	// L12929
        }
      }
      ap_int<128> v8365 = packed_c139;	// L12932
      v8313.write(v8365);	// L12933
    }
  }
}

void gemm_9_13(
  hls::stream< int8_t >& v8366,
  hls::stream< int8_t >& v8367,
  hls::stream< int8_t >& v8368,
  hls::stream< int8_t >& v8369,
  hls::stream< ap_uint<128> >& v8370,
  hls::stream< ap_uint<128> >& v8371
) {	// L12938
  l_S_mt_nt_0_mt168: for (int mt168 = 0; mt168 < 4; mt168++) {	// L12946
    l_nt168: for (int nt168 = 0; nt168 < 4; nt168++) {	// L12947
      int8_t c140;	// L12948
      c140 = 0;	// L12949
      l_S_k_0_k168: for (int k168 = 0; k168 < 64; k168++) {	// L12950
        int8_t v8376 = v8366.read();	// L12951
        int8_t a149;	// L12952
        a149 = v8376;	// L12953
        int8_t v8378 = v8367.read();	// L12954
        int8_t b156;	// L12955
        b156 = v8378;	// L12956
        int8_t v8380 = a149;	// L12957
        int8_t v8381 = b156;	// L12958
        int16_t v8382 = v8380;	// L12959
        int16_t v8383 = v8381;	// L12960
        int16_t v8384 = v8382 * v8383;	// L12961
        int8_t v8385 = c140;	// L12962
        ap_int<17> v8386 = v8385;	// L12963
        ap_int<17> v8387 = v8384;	// L12964
        ap_int<17> v8388 = v8386 + v8387;	// L12965
        int8_t v8389 = v8388;	// L12966
        c140 = v8389;	// L12967
        int8_t v8390 = a149;	// L12968
        v8368.write(v8390);	// L12969
        int8_t v8391 = b156;	// L12970
        v8369.write(v8391);	// L12971
      }
      ap_uint<128> packed_tmp140;	// L12973
      ap_uint<128> v8393 = v8370.read();	// L12974
      packed_tmp140 = v8393;	// L12975
      ap_uint<128> packed_c140;	// L12976
      packed_c140 = 0;	// L12977
      l_S_m_1_m140: for (int m140 = 0; m140 < 16; m140++) {	// L12978
        ap_int<33> v8396 = m140;	// L12979
        bool v8397 = v8396 == 8;	// L12980
        if (v8397) {	// L12981
          int8_t v8398 = c140;	// L12982
          ap_int<128> v8399 = packed_c140;	// L12983
          int64_t v8400 = m140;	// L12984
          int64_t v8401 = v8400 * 8;	// L12985
          ap_int<34> v8402 = m140;	// L12986
          ap_int<34> v8403 = v8402 + 1;	// L12987
          ap_int<66> v8404 = v8403;	// L12988
          ap_int<66> v8405 = v8404 * 8;	// L12989
          ap_int<66> v8406 = v8405 - 1;	// L12990
          int v8407 = v8401;	// L12991
          int v8408 = v8406;	// L12992
          ap_int<128> v8409;
          ap_int<128> v8409_tmp = v8399;
          v8409_tmp(v8408, v8407) = v8398;
          v8409 = v8409_tmp;	// L12993
          packed_c140 = v8409;	// L12994
        } else {
          ap_int<128> v8410 = packed_tmp140;	// L12996
          int64_t v8411 = m140;	// L12997
          int64_t v8412 = v8411 * 8;	// L12998
          ap_int<34> v8413 = m140;	// L12999
          ap_int<34> v8414 = v8413 + 1;	// L13000
          ap_int<66> v8415 = v8414;	// L13001
          ap_int<66> v8416 = v8415 * 8;	// L13002
          ap_int<66> v8417 = v8416 - 1;	// L13003
          int v8418 = v8412;	// L13004
          int v8419 = v8417;	// L13005
          int8_t v8420;
          ap_int<128> v8420_tmp = v8410;
          v8420 = v8420_tmp(v8419, v8418);	// L13006
          ap_int<128> v8421 = packed_c140;	// L13007
          ap_int<128> v8422;
          ap_int<128> v8422_tmp = v8421;
          v8422_tmp(v8419, v8418) = v8420;
          v8422 = v8422_tmp;	// L13017
          packed_c140 = v8422;	// L13018
        }
      }
      ap_int<128> v8423 = packed_c140;	// L13021
      v8371.write(v8423);	// L13022
    }
  }
}

void gemm_9_14(
  hls::stream< int8_t >& v8424,
  hls::stream< int8_t >& v8425,
  hls::stream< int8_t >& v8426,
  hls::stream< int8_t >& v8427,
  hls::stream< ap_uint<128> >& v8428,
  hls::stream< ap_uint<128> >& v8429
) {	// L13027
  l_S_mt_nt_0_mt169: for (int mt169 = 0; mt169 < 4; mt169++) {	// L13035
    l_nt169: for (int nt169 = 0; nt169 < 4; nt169++) {	// L13036
      int8_t c141;	// L13037
      c141 = 0;	// L13038
      l_S_k_0_k169: for (int k169 = 0; k169 < 64; k169++) {	// L13039
        int8_t v8434 = v8424.read();	// L13040
        int8_t a150;	// L13041
        a150 = v8434;	// L13042
        int8_t v8436 = v8425.read();	// L13043
        int8_t b157;	// L13044
        b157 = v8436;	// L13045
        int8_t v8438 = a150;	// L13046
        int8_t v8439 = b157;	// L13047
        int16_t v8440 = v8438;	// L13048
        int16_t v8441 = v8439;	// L13049
        int16_t v8442 = v8440 * v8441;	// L13050
        int8_t v8443 = c141;	// L13051
        ap_int<17> v8444 = v8443;	// L13052
        ap_int<17> v8445 = v8442;	// L13053
        ap_int<17> v8446 = v8444 + v8445;	// L13054
        int8_t v8447 = v8446;	// L13055
        c141 = v8447;	// L13056
        int8_t v8448 = a150;	// L13057
        v8426.write(v8448);	// L13058
        int8_t v8449 = b157;	// L13059
        v8427.write(v8449);	// L13060
      }
      ap_uint<128> packed_tmp141;	// L13062
      ap_uint<128> v8451 = v8428.read();	// L13063
      packed_tmp141 = v8451;	// L13064
      ap_uint<128> packed_c141;	// L13065
      packed_c141 = 0;	// L13066
      l_S_m_1_m141: for (int m141 = 0; m141 < 16; m141++) {	// L13067
        ap_int<33> v8454 = m141;	// L13068
        bool v8455 = v8454 == 8;	// L13069
        if (v8455) {	// L13070
          int8_t v8456 = c141;	// L13071
          ap_int<128> v8457 = packed_c141;	// L13072
          int64_t v8458 = m141;	// L13073
          int64_t v8459 = v8458 * 8;	// L13074
          ap_int<34> v8460 = m141;	// L13075
          ap_int<34> v8461 = v8460 + 1;	// L13076
          ap_int<66> v8462 = v8461;	// L13077
          ap_int<66> v8463 = v8462 * 8;	// L13078
          ap_int<66> v8464 = v8463 - 1;	// L13079
          int v8465 = v8459;	// L13080
          int v8466 = v8464;	// L13081
          ap_int<128> v8467;
          ap_int<128> v8467_tmp = v8457;
          v8467_tmp(v8466, v8465) = v8456;
          v8467 = v8467_tmp;	// L13082
          packed_c141 = v8467;	// L13083
        } else {
          ap_int<128> v8468 = packed_tmp141;	// L13085
          int64_t v8469 = m141;	// L13086
          int64_t v8470 = v8469 * 8;	// L13087
          ap_int<34> v8471 = m141;	// L13088
          ap_int<34> v8472 = v8471 + 1;	// L13089
          ap_int<66> v8473 = v8472;	// L13090
          ap_int<66> v8474 = v8473 * 8;	// L13091
          ap_int<66> v8475 = v8474 - 1;	// L13092
          int v8476 = v8470;	// L13093
          int v8477 = v8475;	// L13094
          int8_t v8478;
          ap_int<128> v8478_tmp = v8468;
          v8478 = v8478_tmp(v8477, v8476);	// L13095
          ap_int<128> v8479 = packed_c141;	// L13096
          ap_int<128> v8480;
          ap_int<128> v8480_tmp = v8479;
          v8480_tmp(v8477, v8476) = v8478;
          v8480 = v8480_tmp;	// L13106
          packed_c141 = v8480;	// L13107
        }
      }
      ap_int<128> v8481 = packed_c141;	// L13110
      v8429.write(v8481);	// L13111
    }
  }
}

void gemm_9_15(
  hls::stream< int8_t >& v8482,
  hls::stream< int8_t >& v8483,
  hls::stream< int8_t >& v8484,
  hls::stream< int8_t >& v8485,
  hls::stream< ap_uint<128> >& v8486,
  hls::stream< ap_uint<128> >& v8487
) {	// L13116
  l_S_mt_nt_0_mt170: for (int mt170 = 0; mt170 < 4; mt170++) {	// L13124
    l_nt170: for (int nt170 = 0; nt170 < 4; nt170++) {	// L13125
      int8_t c142;	// L13126
      c142 = 0;	// L13127
      l_S_k_0_k170: for (int k170 = 0; k170 < 64; k170++) {	// L13128
        int8_t v8492 = v8482.read();	// L13129
        int8_t a151;	// L13130
        a151 = v8492;	// L13131
        int8_t v8494 = v8483.read();	// L13132
        int8_t b158;	// L13133
        b158 = v8494;	// L13134
        int8_t v8496 = a151;	// L13135
        int8_t v8497 = b158;	// L13136
        int16_t v8498 = v8496;	// L13137
        int16_t v8499 = v8497;	// L13138
        int16_t v8500 = v8498 * v8499;	// L13139
        int8_t v8501 = c142;	// L13140
        ap_int<17> v8502 = v8501;	// L13141
        ap_int<17> v8503 = v8500;	// L13142
        ap_int<17> v8504 = v8502 + v8503;	// L13143
        int8_t v8505 = v8504;	// L13144
        c142 = v8505;	// L13145
        int8_t v8506 = a151;	// L13146
        v8484.write(v8506);	// L13147
        int8_t v8507 = b158;	// L13148
        v8485.write(v8507);	// L13149
      }
      ap_uint<128> packed_tmp142;	// L13151
      ap_uint<128> v8509 = v8486.read();	// L13152
      packed_tmp142 = v8509;	// L13153
      ap_uint<128> packed_c142;	// L13154
      packed_c142 = 0;	// L13155
      l_S_m_1_m142: for (int m142 = 0; m142 < 16; m142++) {	// L13156
        ap_int<33> v8512 = m142;	// L13157
        bool v8513 = v8512 == 8;	// L13158
        if (v8513) {	// L13159
          int8_t v8514 = c142;	// L13160
          ap_int<128> v8515 = packed_c142;	// L13161
          int64_t v8516 = m142;	// L13162
          int64_t v8517 = v8516 * 8;	// L13163
          ap_int<34> v8518 = m142;	// L13164
          ap_int<34> v8519 = v8518 + 1;	// L13165
          ap_int<66> v8520 = v8519;	// L13166
          ap_int<66> v8521 = v8520 * 8;	// L13167
          ap_int<66> v8522 = v8521 - 1;	// L13168
          int v8523 = v8517;	// L13169
          int v8524 = v8522;	// L13170
          ap_int<128> v8525;
          ap_int<128> v8525_tmp = v8515;
          v8525_tmp(v8524, v8523) = v8514;
          v8525 = v8525_tmp;	// L13171
          packed_c142 = v8525;	// L13172
        } else {
          ap_int<128> v8526 = packed_tmp142;	// L13174
          int64_t v8527 = m142;	// L13175
          int64_t v8528 = v8527 * 8;	// L13176
          ap_int<34> v8529 = m142;	// L13177
          ap_int<34> v8530 = v8529 + 1;	// L13178
          ap_int<66> v8531 = v8530;	// L13179
          ap_int<66> v8532 = v8531 * 8;	// L13180
          ap_int<66> v8533 = v8532 - 1;	// L13181
          int v8534 = v8528;	// L13182
          int v8535 = v8533;	// L13183
          int8_t v8536;
          ap_int<128> v8536_tmp = v8526;
          v8536 = v8536_tmp(v8535, v8534);	// L13184
          ap_int<128> v8537 = packed_c142;	// L13185
          ap_int<128> v8538;
          ap_int<128> v8538_tmp = v8537;
          v8538_tmp(v8535, v8534) = v8536;
          v8538 = v8538_tmp;	// L13195
          packed_c142 = v8538;	// L13196
        }
      }
      ap_int<128> v8539 = packed_c142;	// L13199
      v8487.write(v8539);	// L13200
    }
  }
}

void gemm_9_16(
  hls::stream< int8_t >& v8540,
  hls::stream< int8_t >& v8541,
  hls::stream< int8_t >& v8542,
  hls::stream< ap_uint<128> >& v8543,
  hls::stream< ap_uint<128> >& v8544
) {	// L13205
  l_S_mt_nt_0_mt171: for (int mt171 = 0; mt171 < 4; mt171++) {	// L13213
    l_nt171: for (int nt171 = 0; nt171 < 4; nt171++) {	// L13214
      int8_t c143;	// L13215
      c143 = 0;	// L13216
      l_S_k_0_k171: for (int k171 = 0; k171 < 64; k171++) {	// L13217
        int8_t v8549 = v8540.read();	// L13218
        int8_t a152;	// L13219
        a152 = v8549;	// L13220
        int8_t v8551 = v8541.read();	// L13221
        int8_t b159;	// L13222
        b159 = v8551;	// L13223
        int8_t v8553 = a152;	// L13224
        int8_t v8554 = b159;	// L13225
        int16_t v8555 = v8553;	// L13226
        int16_t v8556 = v8554;	// L13227
        int16_t v8557 = v8555 * v8556;	// L13228
        int8_t v8558 = c143;	// L13229
        ap_int<17> v8559 = v8558;	// L13230
        ap_int<17> v8560 = v8557;	// L13231
        ap_int<17> v8561 = v8559 + v8560;	// L13232
        int8_t v8562 = v8561;	// L13233
        c143 = v8562;	// L13234
        int8_t v8563 = b159;	// L13235
        v8542.write(v8563);	// L13236
      }
      ap_uint<128> packed_tmp143;	// L13238
      ap_uint<128> v8565 = v8543.read();	// L13239
      packed_tmp143 = v8565;	// L13240
      ap_uint<128> packed_c143;	// L13241
      packed_c143 = 0;	// L13242
      l_S_m_1_m143: for (int m143 = 0; m143 < 16; m143++) {	// L13243
        ap_int<33> v8568 = m143;	// L13244
        bool v8569 = v8568 == 8;	// L13245
        if (v8569) {	// L13246
          int8_t v8570 = c143;	// L13247
          ap_int<128> v8571 = packed_c143;	// L13248
          int64_t v8572 = m143;	// L13249
          int64_t v8573 = v8572 * 8;	// L13250
          ap_int<34> v8574 = m143;	// L13251
          ap_int<34> v8575 = v8574 + 1;	// L13252
          ap_int<66> v8576 = v8575;	// L13253
          ap_int<66> v8577 = v8576 * 8;	// L13254
          ap_int<66> v8578 = v8577 - 1;	// L13255
          int v8579 = v8573;	// L13256
          int v8580 = v8578;	// L13257
          ap_int<128> v8581;
          ap_int<128> v8581_tmp = v8571;
          v8581_tmp(v8580, v8579) = v8570;
          v8581 = v8581_tmp;	// L13258
          packed_c143 = v8581;	// L13259
        } else {
          ap_int<128> v8582 = packed_tmp143;	// L13261
          int64_t v8583 = m143;	// L13262
          int64_t v8584 = v8583 * 8;	// L13263
          ap_int<34> v8585 = m143;	// L13264
          ap_int<34> v8586 = v8585 + 1;	// L13265
          ap_int<66> v8587 = v8586;	// L13266
          ap_int<66> v8588 = v8587 * 8;	// L13267
          ap_int<66> v8589 = v8588 - 1;	// L13268
          int v8590 = v8584;	// L13269
          int v8591 = v8589;	// L13270
          int8_t v8592;
          ap_int<128> v8592_tmp = v8582;
          v8592 = v8592_tmp(v8591, v8590);	// L13271
          ap_int<128> v8593 = packed_c143;	// L13272
          ap_int<128> v8594;
          ap_int<128> v8594_tmp = v8593;
          v8594_tmp(v8591, v8590) = v8592;
          v8594 = v8594_tmp;	// L13282
          packed_c143 = v8594;	// L13283
        }
      }
      ap_int<128> v8595 = packed_c143;	// L13286
      v8544.write(v8595);	// L13287
    }
  }
}

void gemm_10_0(
  hls::stream< ap_uint<128> >& v8596,
  hls::stream< int8_t >& v8597,
  hls::stream< ap_uint<128> >& v8598
) {	// L13292
  l_S_mt_nt_0_mt172: for (int mt172 = 0; mt172 < 4; mt172++) {	// L13295
    l_nt172: for (int nt172 = 0; nt172 < 4; nt172++) {	// L13296
      l_S_k_0_k172: for (int k172 = 0; k172 < 64; k172++) {	// L13297
        ap_uint<128> v8602 = v8596.read();	// L13298
        ap_uint<128> a153;	// L13299
        a153 = v8602;	// L13300
        ap_int<128> v8604 = a153;	// L13301
        int8_t v8605;
        ap_int<128> v8605_tmp = v8604;
        v8605 = v8605_tmp(79, 72);	// L13302
        v8597.write(v8605);	// L13303
        ap_int<128> v8606 = a153;	// L13304
        v8598.write(v8606);	// L13305
      }
    }
  }
}

void gemm_10_1(
  hls::stream< int8_t >& v8607,
  hls::stream< int8_t >& v8608,
  hls::stream< int8_t >& v8609,
  hls::stream< int8_t >& v8610,
  hls::stream< ap_uint<128> >& v8611,
  hls::stream< ap_uint<128> >& v8612
) {	// L13311
  l_S_mt_nt_0_mt173: for (int mt173 = 0; mt173 < 4; mt173++) {	// L13319
    l_nt173: for (int nt173 = 0; nt173 < 4; nt173++) {	// L13320
      int8_t c144;	// L13321
      c144 = 0;	// L13322
      l_S_k_0_k173: for (int k173 = 0; k173 < 64; k173++) {	// L13323
        int8_t v8617 = v8607.read();	// L13324
        int8_t a154;	// L13325
        a154 = v8617;	// L13326
        int8_t v8619 = v8608.read();	// L13327
        int8_t b160;	// L13328
        b160 = v8619;	// L13329
        int8_t v8621 = a154;	// L13330
        int8_t v8622 = b160;	// L13331
        int16_t v8623 = v8621;	// L13332
        int16_t v8624 = v8622;	// L13333
        int16_t v8625 = v8623 * v8624;	// L13334
        int8_t v8626 = c144;	// L13335
        ap_int<17> v8627 = v8626;	// L13336
        ap_int<17> v8628 = v8625;	// L13337
        ap_int<17> v8629 = v8627 + v8628;	// L13338
        int8_t v8630 = v8629;	// L13339
        c144 = v8630;	// L13340
        int8_t v8631 = a154;	// L13341
        v8609.write(v8631);	// L13342
        int8_t v8632 = b160;	// L13343
        v8610.write(v8632);	// L13344
      }
      ap_uint<128> packed_tmp144;	// L13346
      ap_uint<128> v8634 = v8611.read();	// L13347
      packed_tmp144 = v8634;	// L13348
      ap_uint<128> packed_c144;	// L13349
      packed_c144 = 0;	// L13350
      l_S_m_1_m144: for (int m144 = 0; m144 < 16; m144++) {	// L13351
        ap_int<33> v8637 = m144;	// L13352
        bool v8638 = v8637 == 9;	// L13353
        if (v8638) {	// L13354
          int8_t v8639 = c144;	// L13355
          ap_int<128> v8640 = packed_c144;	// L13356
          int64_t v8641 = m144;	// L13357
          int64_t v8642 = v8641 * 8;	// L13358
          ap_int<34> v8643 = m144;	// L13359
          ap_int<34> v8644 = v8643 + 1;	// L13360
          ap_int<66> v8645 = v8644;	// L13361
          ap_int<66> v8646 = v8645 * 8;	// L13362
          ap_int<66> v8647 = v8646 - 1;	// L13363
          int v8648 = v8642;	// L13364
          int v8649 = v8647;	// L13365
          ap_int<128> v8650;
          ap_int<128> v8650_tmp = v8640;
          v8650_tmp(v8649, v8648) = v8639;
          v8650 = v8650_tmp;	// L13366
          packed_c144 = v8650;	// L13367
        } else {
          ap_int<128> v8651 = packed_tmp144;	// L13369
          int64_t v8652 = m144;	// L13370
          int64_t v8653 = v8652 * 8;	// L13371
          ap_int<34> v8654 = m144;	// L13372
          ap_int<34> v8655 = v8654 + 1;	// L13373
          ap_int<66> v8656 = v8655;	// L13374
          ap_int<66> v8657 = v8656 * 8;	// L13375
          ap_int<66> v8658 = v8657 - 1;	// L13376
          int v8659 = v8653;	// L13377
          int v8660 = v8658;	// L13378
          int8_t v8661;
          ap_int<128> v8661_tmp = v8651;
          v8661 = v8661_tmp(v8660, v8659);	// L13379
          ap_int<128> v8662 = packed_c144;	// L13380
          ap_int<128> v8663;
          ap_int<128> v8663_tmp = v8662;
          v8663_tmp(v8660, v8659) = v8661;
          v8663 = v8663_tmp;	// L13390
          packed_c144 = v8663;	// L13391
        }
      }
      ap_int<128> v8664 = packed_c144;	// L13394
      v8612.write(v8664);	// L13395
    }
  }
}

void gemm_10_2(
  hls::stream< int8_t >& v8665,
  hls::stream< int8_t >& v8666,
  hls::stream< int8_t >& v8667,
  hls::stream< int8_t >& v8668,
  hls::stream< ap_uint<128> >& v8669,
  hls::stream< ap_uint<128> >& v8670
) {	// L13400
  l_S_mt_nt_0_mt174: for (int mt174 = 0; mt174 < 4; mt174++) {	// L13408
    l_nt174: for (int nt174 = 0; nt174 < 4; nt174++) {	// L13409
      int8_t c145;	// L13410
      c145 = 0;	// L13411
      l_S_k_0_k174: for (int k174 = 0; k174 < 64; k174++) {	// L13412
        int8_t v8675 = v8665.read();	// L13413
        int8_t a155;	// L13414
        a155 = v8675;	// L13415
        int8_t v8677 = v8666.read();	// L13416
        int8_t b161;	// L13417
        b161 = v8677;	// L13418
        int8_t v8679 = a155;	// L13419
        int8_t v8680 = b161;	// L13420
        int16_t v8681 = v8679;	// L13421
        int16_t v8682 = v8680;	// L13422
        int16_t v8683 = v8681 * v8682;	// L13423
        int8_t v8684 = c145;	// L13424
        ap_int<17> v8685 = v8684;	// L13425
        ap_int<17> v8686 = v8683;	// L13426
        ap_int<17> v8687 = v8685 + v8686;	// L13427
        int8_t v8688 = v8687;	// L13428
        c145 = v8688;	// L13429
        int8_t v8689 = a155;	// L13430
        v8667.write(v8689);	// L13431
        int8_t v8690 = b161;	// L13432
        v8668.write(v8690);	// L13433
      }
      ap_uint<128> packed_tmp145;	// L13435
      ap_uint<128> v8692 = v8669.read();	// L13436
      packed_tmp145 = v8692;	// L13437
      ap_uint<128> packed_c145;	// L13438
      packed_c145 = 0;	// L13439
      l_S_m_1_m145: for (int m145 = 0; m145 < 16; m145++) {	// L13440
        ap_int<33> v8695 = m145;	// L13441
        bool v8696 = v8695 == 9;	// L13442
        if (v8696) {	// L13443
          int8_t v8697 = c145;	// L13444
          ap_int<128> v8698 = packed_c145;	// L13445
          int64_t v8699 = m145;	// L13446
          int64_t v8700 = v8699 * 8;	// L13447
          ap_int<34> v8701 = m145;	// L13448
          ap_int<34> v8702 = v8701 + 1;	// L13449
          ap_int<66> v8703 = v8702;	// L13450
          ap_int<66> v8704 = v8703 * 8;	// L13451
          ap_int<66> v8705 = v8704 - 1;	// L13452
          int v8706 = v8700;	// L13453
          int v8707 = v8705;	// L13454
          ap_int<128> v8708;
          ap_int<128> v8708_tmp = v8698;
          v8708_tmp(v8707, v8706) = v8697;
          v8708 = v8708_tmp;	// L13455
          packed_c145 = v8708;	// L13456
        } else {
          ap_int<128> v8709 = packed_tmp145;	// L13458
          int64_t v8710 = m145;	// L13459
          int64_t v8711 = v8710 * 8;	// L13460
          ap_int<34> v8712 = m145;	// L13461
          ap_int<34> v8713 = v8712 + 1;	// L13462
          ap_int<66> v8714 = v8713;	// L13463
          ap_int<66> v8715 = v8714 * 8;	// L13464
          ap_int<66> v8716 = v8715 - 1;	// L13465
          int v8717 = v8711;	// L13466
          int v8718 = v8716;	// L13467
          int8_t v8719;
          ap_int<128> v8719_tmp = v8709;
          v8719 = v8719_tmp(v8718, v8717);	// L13468
          ap_int<128> v8720 = packed_c145;	// L13469
          ap_int<128> v8721;
          ap_int<128> v8721_tmp = v8720;
          v8721_tmp(v8718, v8717) = v8719;
          v8721 = v8721_tmp;	// L13479
          packed_c145 = v8721;	// L13480
        }
      }
      ap_int<128> v8722 = packed_c145;	// L13483
      v8670.write(v8722);	// L13484
    }
  }
}

void gemm_10_3(
  hls::stream< int8_t >& v8723,
  hls::stream< int8_t >& v8724,
  hls::stream< int8_t >& v8725,
  hls::stream< int8_t >& v8726,
  hls::stream< ap_uint<128> >& v8727,
  hls::stream< ap_uint<128> >& v8728
) {	// L13489
  l_S_mt_nt_0_mt175: for (int mt175 = 0; mt175 < 4; mt175++) {	// L13497
    l_nt175: for (int nt175 = 0; nt175 < 4; nt175++) {	// L13498
      int8_t c146;	// L13499
      c146 = 0;	// L13500
      l_S_k_0_k175: for (int k175 = 0; k175 < 64; k175++) {	// L13501
        int8_t v8733 = v8723.read();	// L13502
        int8_t a156;	// L13503
        a156 = v8733;	// L13504
        int8_t v8735 = v8724.read();	// L13505
        int8_t b162;	// L13506
        b162 = v8735;	// L13507
        int8_t v8737 = a156;	// L13508
        int8_t v8738 = b162;	// L13509
        int16_t v8739 = v8737;	// L13510
        int16_t v8740 = v8738;	// L13511
        int16_t v8741 = v8739 * v8740;	// L13512
        int8_t v8742 = c146;	// L13513
        ap_int<17> v8743 = v8742;	// L13514
        ap_int<17> v8744 = v8741;	// L13515
        ap_int<17> v8745 = v8743 + v8744;	// L13516
        int8_t v8746 = v8745;	// L13517
        c146 = v8746;	// L13518
        int8_t v8747 = a156;	// L13519
        v8725.write(v8747);	// L13520
        int8_t v8748 = b162;	// L13521
        v8726.write(v8748);	// L13522
      }
      ap_uint<128> packed_tmp146;	// L13524
      ap_uint<128> v8750 = v8727.read();	// L13525
      packed_tmp146 = v8750;	// L13526
      ap_uint<128> packed_c146;	// L13527
      packed_c146 = 0;	// L13528
      l_S_m_1_m146: for (int m146 = 0; m146 < 16; m146++) {	// L13529
        ap_int<33> v8753 = m146;	// L13530
        bool v8754 = v8753 == 9;	// L13531
        if (v8754) {	// L13532
          int8_t v8755 = c146;	// L13533
          ap_int<128> v8756 = packed_c146;	// L13534
          int64_t v8757 = m146;	// L13535
          int64_t v8758 = v8757 * 8;	// L13536
          ap_int<34> v8759 = m146;	// L13537
          ap_int<34> v8760 = v8759 + 1;	// L13538
          ap_int<66> v8761 = v8760;	// L13539
          ap_int<66> v8762 = v8761 * 8;	// L13540
          ap_int<66> v8763 = v8762 - 1;	// L13541
          int v8764 = v8758;	// L13542
          int v8765 = v8763;	// L13543
          ap_int<128> v8766;
          ap_int<128> v8766_tmp = v8756;
          v8766_tmp(v8765, v8764) = v8755;
          v8766 = v8766_tmp;	// L13544
          packed_c146 = v8766;	// L13545
        } else {
          ap_int<128> v8767 = packed_tmp146;	// L13547
          int64_t v8768 = m146;	// L13548
          int64_t v8769 = v8768 * 8;	// L13549
          ap_int<34> v8770 = m146;	// L13550
          ap_int<34> v8771 = v8770 + 1;	// L13551
          ap_int<66> v8772 = v8771;	// L13552
          ap_int<66> v8773 = v8772 * 8;	// L13553
          ap_int<66> v8774 = v8773 - 1;	// L13554
          int v8775 = v8769;	// L13555
          int v8776 = v8774;	// L13556
          int8_t v8777;
          ap_int<128> v8777_tmp = v8767;
          v8777 = v8777_tmp(v8776, v8775);	// L13557
          ap_int<128> v8778 = packed_c146;	// L13558
          ap_int<128> v8779;
          ap_int<128> v8779_tmp = v8778;
          v8779_tmp(v8776, v8775) = v8777;
          v8779 = v8779_tmp;	// L13568
          packed_c146 = v8779;	// L13569
        }
      }
      ap_int<128> v8780 = packed_c146;	// L13572
      v8728.write(v8780);	// L13573
    }
  }
}

void gemm_10_4(
  hls::stream< int8_t >& v8781,
  hls::stream< int8_t >& v8782,
  hls::stream< int8_t >& v8783,
  hls::stream< int8_t >& v8784,
  hls::stream< ap_uint<128> >& v8785,
  hls::stream< ap_uint<128> >& v8786
) {	// L13578
  l_S_mt_nt_0_mt176: for (int mt176 = 0; mt176 < 4; mt176++) {	// L13586
    l_nt176: for (int nt176 = 0; nt176 < 4; nt176++) {	// L13587
      int8_t c147;	// L13588
      c147 = 0;	// L13589
      l_S_k_0_k176: for (int k176 = 0; k176 < 64; k176++) {	// L13590
        int8_t v8791 = v8781.read();	// L13591
        int8_t a157;	// L13592
        a157 = v8791;	// L13593
        int8_t v8793 = v8782.read();	// L13594
        int8_t b163;	// L13595
        b163 = v8793;	// L13596
        int8_t v8795 = a157;	// L13597
        int8_t v8796 = b163;	// L13598
        int16_t v8797 = v8795;	// L13599
        int16_t v8798 = v8796;	// L13600
        int16_t v8799 = v8797 * v8798;	// L13601
        int8_t v8800 = c147;	// L13602
        ap_int<17> v8801 = v8800;	// L13603
        ap_int<17> v8802 = v8799;	// L13604
        ap_int<17> v8803 = v8801 + v8802;	// L13605
        int8_t v8804 = v8803;	// L13606
        c147 = v8804;	// L13607
        int8_t v8805 = a157;	// L13608
        v8783.write(v8805);	// L13609
        int8_t v8806 = b163;	// L13610
        v8784.write(v8806);	// L13611
      }
      ap_uint<128> packed_tmp147;	// L13613
      ap_uint<128> v8808 = v8785.read();	// L13614
      packed_tmp147 = v8808;	// L13615
      ap_uint<128> packed_c147;	// L13616
      packed_c147 = 0;	// L13617
      l_S_m_1_m147: for (int m147 = 0; m147 < 16; m147++) {	// L13618
        ap_int<33> v8811 = m147;	// L13619
        bool v8812 = v8811 == 9;	// L13620
        if (v8812) {	// L13621
          int8_t v8813 = c147;	// L13622
          ap_int<128> v8814 = packed_c147;	// L13623
          int64_t v8815 = m147;	// L13624
          int64_t v8816 = v8815 * 8;	// L13625
          ap_int<34> v8817 = m147;	// L13626
          ap_int<34> v8818 = v8817 + 1;	// L13627
          ap_int<66> v8819 = v8818;	// L13628
          ap_int<66> v8820 = v8819 * 8;	// L13629
          ap_int<66> v8821 = v8820 - 1;	// L13630
          int v8822 = v8816;	// L13631
          int v8823 = v8821;	// L13632
          ap_int<128> v8824;
          ap_int<128> v8824_tmp = v8814;
          v8824_tmp(v8823, v8822) = v8813;
          v8824 = v8824_tmp;	// L13633
          packed_c147 = v8824;	// L13634
        } else {
          ap_int<128> v8825 = packed_tmp147;	// L13636
          int64_t v8826 = m147;	// L13637
          int64_t v8827 = v8826 * 8;	// L13638
          ap_int<34> v8828 = m147;	// L13639
          ap_int<34> v8829 = v8828 + 1;	// L13640
          ap_int<66> v8830 = v8829;	// L13641
          ap_int<66> v8831 = v8830 * 8;	// L13642
          ap_int<66> v8832 = v8831 - 1;	// L13643
          int v8833 = v8827;	// L13644
          int v8834 = v8832;	// L13645
          int8_t v8835;
          ap_int<128> v8835_tmp = v8825;
          v8835 = v8835_tmp(v8834, v8833);	// L13646
          ap_int<128> v8836 = packed_c147;	// L13647
          ap_int<128> v8837;
          ap_int<128> v8837_tmp = v8836;
          v8837_tmp(v8834, v8833) = v8835;
          v8837 = v8837_tmp;	// L13657
          packed_c147 = v8837;	// L13658
        }
      }
      ap_int<128> v8838 = packed_c147;	// L13661
      v8786.write(v8838);	// L13662
    }
  }
}

void gemm_10_5(
  hls::stream< int8_t >& v8839,
  hls::stream< int8_t >& v8840,
  hls::stream< int8_t >& v8841,
  hls::stream< int8_t >& v8842,
  hls::stream< ap_uint<128> >& v8843,
  hls::stream< ap_uint<128> >& v8844
) {	// L13667
  l_S_mt_nt_0_mt177: for (int mt177 = 0; mt177 < 4; mt177++) {	// L13675
    l_nt177: for (int nt177 = 0; nt177 < 4; nt177++) {	// L13676
      int8_t c148;	// L13677
      c148 = 0;	// L13678
      l_S_k_0_k177: for (int k177 = 0; k177 < 64; k177++) {	// L13679
        int8_t v8849 = v8839.read();	// L13680
        int8_t a158;	// L13681
        a158 = v8849;	// L13682
        int8_t v8851 = v8840.read();	// L13683
        int8_t b164;	// L13684
        b164 = v8851;	// L13685
        int8_t v8853 = a158;	// L13686
        int8_t v8854 = b164;	// L13687
        int16_t v8855 = v8853;	// L13688
        int16_t v8856 = v8854;	// L13689
        int16_t v8857 = v8855 * v8856;	// L13690
        int8_t v8858 = c148;	// L13691
        ap_int<17> v8859 = v8858;	// L13692
        ap_int<17> v8860 = v8857;	// L13693
        ap_int<17> v8861 = v8859 + v8860;	// L13694
        int8_t v8862 = v8861;	// L13695
        c148 = v8862;	// L13696
        int8_t v8863 = a158;	// L13697
        v8841.write(v8863);	// L13698
        int8_t v8864 = b164;	// L13699
        v8842.write(v8864);	// L13700
      }
      ap_uint<128> packed_tmp148;	// L13702
      ap_uint<128> v8866 = v8843.read();	// L13703
      packed_tmp148 = v8866;	// L13704
      ap_uint<128> packed_c148;	// L13705
      packed_c148 = 0;	// L13706
      l_S_m_1_m148: for (int m148 = 0; m148 < 16; m148++) {	// L13707
        ap_int<33> v8869 = m148;	// L13708
        bool v8870 = v8869 == 9;	// L13709
        if (v8870) {	// L13710
          int8_t v8871 = c148;	// L13711
          ap_int<128> v8872 = packed_c148;	// L13712
          int64_t v8873 = m148;	// L13713
          int64_t v8874 = v8873 * 8;	// L13714
          ap_int<34> v8875 = m148;	// L13715
          ap_int<34> v8876 = v8875 + 1;	// L13716
          ap_int<66> v8877 = v8876;	// L13717
          ap_int<66> v8878 = v8877 * 8;	// L13718
          ap_int<66> v8879 = v8878 - 1;	// L13719
          int v8880 = v8874;	// L13720
          int v8881 = v8879;	// L13721
          ap_int<128> v8882;
          ap_int<128> v8882_tmp = v8872;
          v8882_tmp(v8881, v8880) = v8871;
          v8882 = v8882_tmp;	// L13722
          packed_c148 = v8882;	// L13723
        } else {
          ap_int<128> v8883 = packed_tmp148;	// L13725
          int64_t v8884 = m148;	// L13726
          int64_t v8885 = v8884 * 8;	// L13727
          ap_int<34> v8886 = m148;	// L13728
          ap_int<34> v8887 = v8886 + 1;	// L13729
          ap_int<66> v8888 = v8887;	// L13730
          ap_int<66> v8889 = v8888 * 8;	// L13731
          ap_int<66> v8890 = v8889 - 1;	// L13732
          int v8891 = v8885;	// L13733
          int v8892 = v8890;	// L13734
          int8_t v8893;
          ap_int<128> v8893_tmp = v8883;
          v8893 = v8893_tmp(v8892, v8891);	// L13735
          ap_int<128> v8894 = packed_c148;	// L13736
          ap_int<128> v8895;
          ap_int<128> v8895_tmp = v8894;
          v8895_tmp(v8892, v8891) = v8893;
          v8895 = v8895_tmp;	// L13746
          packed_c148 = v8895;	// L13747
        }
      }
      ap_int<128> v8896 = packed_c148;	// L13750
      v8844.write(v8896);	// L13751
    }
  }
}

void gemm_10_6(
  hls::stream< int8_t >& v8897,
  hls::stream< int8_t >& v8898,
  hls::stream< int8_t >& v8899,
  hls::stream< int8_t >& v8900,
  hls::stream< ap_uint<128> >& v8901,
  hls::stream< ap_uint<128> >& v8902
) {	// L13756
  l_S_mt_nt_0_mt178: for (int mt178 = 0; mt178 < 4; mt178++) {	// L13764
    l_nt178: for (int nt178 = 0; nt178 < 4; nt178++) {	// L13765
      int8_t c149;	// L13766
      c149 = 0;	// L13767
      l_S_k_0_k178: for (int k178 = 0; k178 < 64; k178++) {	// L13768
        int8_t v8907 = v8897.read();	// L13769
        int8_t a159;	// L13770
        a159 = v8907;	// L13771
        int8_t v8909 = v8898.read();	// L13772
        int8_t b165;	// L13773
        b165 = v8909;	// L13774
        int8_t v8911 = a159;	// L13775
        int8_t v8912 = b165;	// L13776
        int16_t v8913 = v8911;	// L13777
        int16_t v8914 = v8912;	// L13778
        int16_t v8915 = v8913 * v8914;	// L13779
        int8_t v8916 = c149;	// L13780
        ap_int<17> v8917 = v8916;	// L13781
        ap_int<17> v8918 = v8915;	// L13782
        ap_int<17> v8919 = v8917 + v8918;	// L13783
        int8_t v8920 = v8919;	// L13784
        c149 = v8920;	// L13785
        int8_t v8921 = a159;	// L13786
        v8899.write(v8921);	// L13787
        int8_t v8922 = b165;	// L13788
        v8900.write(v8922);	// L13789
      }
      ap_uint<128> packed_tmp149;	// L13791
      ap_uint<128> v8924 = v8901.read();	// L13792
      packed_tmp149 = v8924;	// L13793
      ap_uint<128> packed_c149;	// L13794
      packed_c149 = 0;	// L13795
      l_S_m_1_m149: for (int m149 = 0; m149 < 16; m149++) {	// L13796
        ap_int<33> v8927 = m149;	// L13797
        bool v8928 = v8927 == 9;	// L13798
        if (v8928) {	// L13799
          int8_t v8929 = c149;	// L13800
          ap_int<128> v8930 = packed_c149;	// L13801
          int64_t v8931 = m149;	// L13802
          int64_t v8932 = v8931 * 8;	// L13803
          ap_int<34> v8933 = m149;	// L13804
          ap_int<34> v8934 = v8933 + 1;	// L13805
          ap_int<66> v8935 = v8934;	// L13806
          ap_int<66> v8936 = v8935 * 8;	// L13807
          ap_int<66> v8937 = v8936 - 1;	// L13808
          int v8938 = v8932;	// L13809
          int v8939 = v8937;	// L13810
          ap_int<128> v8940;
          ap_int<128> v8940_tmp = v8930;
          v8940_tmp(v8939, v8938) = v8929;
          v8940 = v8940_tmp;	// L13811
          packed_c149 = v8940;	// L13812
        } else {
          ap_int<128> v8941 = packed_tmp149;	// L13814
          int64_t v8942 = m149;	// L13815
          int64_t v8943 = v8942 * 8;	// L13816
          ap_int<34> v8944 = m149;	// L13817
          ap_int<34> v8945 = v8944 + 1;	// L13818
          ap_int<66> v8946 = v8945;	// L13819
          ap_int<66> v8947 = v8946 * 8;	// L13820
          ap_int<66> v8948 = v8947 - 1;	// L13821
          int v8949 = v8943;	// L13822
          int v8950 = v8948;	// L13823
          int8_t v8951;
          ap_int<128> v8951_tmp = v8941;
          v8951 = v8951_tmp(v8950, v8949);	// L13824
          ap_int<128> v8952 = packed_c149;	// L13825
          ap_int<128> v8953;
          ap_int<128> v8953_tmp = v8952;
          v8953_tmp(v8950, v8949) = v8951;
          v8953 = v8953_tmp;	// L13835
          packed_c149 = v8953;	// L13836
        }
      }
      ap_int<128> v8954 = packed_c149;	// L13839
      v8902.write(v8954);	// L13840
    }
  }
}

void gemm_10_7(
  hls::stream< int8_t >& v8955,
  hls::stream< int8_t >& v8956,
  hls::stream< int8_t >& v8957,
  hls::stream< int8_t >& v8958,
  hls::stream< ap_uint<128> >& v8959,
  hls::stream< ap_uint<128> >& v8960
) {	// L13845
  l_S_mt_nt_0_mt179: for (int mt179 = 0; mt179 < 4; mt179++) {	// L13853
    l_nt179: for (int nt179 = 0; nt179 < 4; nt179++) {	// L13854
      int8_t c150;	// L13855
      c150 = 0;	// L13856
      l_S_k_0_k179: for (int k179 = 0; k179 < 64; k179++) {	// L13857
        int8_t v8965 = v8955.read();	// L13858
        int8_t a160;	// L13859
        a160 = v8965;	// L13860
        int8_t v8967 = v8956.read();	// L13861
        int8_t b166;	// L13862
        b166 = v8967;	// L13863
        int8_t v8969 = a160;	// L13864
        int8_t v8970 = b166;	// L13865
        int16_t v8971 = v8969;	// L13866
        int16_t v8972 = v8970;	// L13867
        int16_t v8973 = v8971 * v8972;	// L13868
        int8_t v8974 = c150;	// L13869
        ap_int<17> v8975 = v8974;	// L13870
        ap_int<17> v8976 = v8973;	// L13871
        ap_int<17> v8977 = v8975 + v8976;	// L13872
        int8_t v8978 = v8977;	// L13873
        c150 = v8978;	// L13874
        int8_t v8979 = a160;	// L13875
        v8957.write(v8979);	// L13876
        int8_t v8980 = b166;	// L13877
        v8958.write(v8980);	// L13878
      }
      ap_uint<128> packed_tmp150;	// L13880
      ap_uint<128> v8982 = v8959.read();	// L13881
      packed_tmp150 = v8982;	// L13882
      ap_uint<128> packed_c150;	// L13883
      packed_c150 = 0;	// L13884
      l_S_m_1_m150: for (int m150 = 0; m150 < 16; m150++) {	// L13885
        ap_int<33> v8985 = m150;	// L13886
        bool v8986 = v8985 == 9;	// L13887
        if (v8986) {	// L13888
          int8_t v8987 = c150;	// L13889
          ap_int<128> v8988 = packed_c150;	// L13890
          int64_t v8989 = m150;	// L13891
          int64_t v8990 = v8989 * 8;	// L13892
          ap_int<34> v8991 = m150;	// L13893
          ap_int<34> v8992 = v8991 + 1;	// L13894
          ap_int<66> v8993 = v8992;	// L13895
          ap_int<66> v8994 = v8993 * 8;	// L13896
          ap_int<66> v8995 = v8994 - 1;	// L13897
          int v8996 = v8990;	// L13898
          int v8997 = v8995;	// L13899
          ap_int<128> v8998;
          ap_int<128> v8998_tmp = v8988;
          v8998_tmp(v8997, v8996) = v8987;
          v8998 = v8998_tmp;	// L13900
          packed_c150 = v8998;	// L13901
        } else {
          ap_int<128> v8999 = packed_tmp150;	// L13903
          int64_t v9000 = m150;	// L13904
          int64_t v9001 = v9000 * 8;	// L13905
          ap_int<34> v9002 = m150;	// L13906
          ap_int<34> v9003 = v9002 + 1;	// L13907
          ap_int<66> v9004 = v9003;	// L13908
          ap_int<66> v9005 = v9004 * 8;	// L13909
          ap_int<66> v9006 = v9005 - 1;	// L13910
          int v9007 = v9001;	// L13911
          int v9008 = v9006;	// L13912
          int8_t v9009;
          ap_int<128> v9009_tmp = v8999;
          v9009 = v9009_tmp(v9008, v9007);	// L13913
          ap_int<128> v9010 = packed_c150;	// L13914
          ap_int<128> v9011;
          ap_int<128> v9011_tmp = v9010;
          v9011_tmp(v9008, v9007) = v9009;
          v9011 = v9011_tmp;	// L13924
          packed_c150 = v9011;	// L13925
        }
      }
      ap_int<128> v9012 = packed_c150;	// L13928
      v8960.write(v9012);	// L13929
    }
  }
}

void gemm_10_8(
  hls::stream< int8_t >& v9013,
  hls::stream< int8_t >& v9014,
  hls::stream< int8_t >& v9015,
  hls::stream< int8_t >& v9016,
  hls::stream< ap_uint<128> >& v9017,
  hls::stream< ap_uint<128> >& v9018
) {	// L13934
  l_S_mt_nt_0_mt180: for (int mt180 = 0; mt180 < 4; mt180++) {	// L13942
    l_nt180: for (int nt180 = 0; nt180 < 4; nt180++) {	// L13943
      int8_t c151;	// L13944
      c151 = 0;	// L13945
      l_S_k_0_k180: for (int k180 = 0; k180 < 64; k180++) {	// L13946
        int8_t v9023 = v9013.read();	// L13947
        int8_t a161;	// L13948
        a161 = v9023;	// L13949
        int8_t v9025 = v9014.read();	// L13950
        int8_t b167;	// L13951
        b167 = v9025;	// L13952
        int8_t v9027 = a161;	// L13953
        int8_t v9028 = b167;	// L13954
        int16_t v9029 = v9027;	// L13955
        int16_t v9030 = v9028;	// L13956
        int16_t v9031 = v9029 * v9030;	// L13957
        int8_t v9032 = c151;	// L13958
        ap_int<17> v9033 = v9032;	// L13959
        ap_int<17> v9034 = v9031;	// L13960
        ap_int<17> v9035 = v9033 + v9034;	// L13961
        int8_t v9036 = v9035;	// L13962
        c151 = v9036;	// L13963
        int8_t v9037 = a161;	// L13964
        v9015.write(v9037);	// L13965
        int8_t v9038 = b167;	// L13966
        v9016.write(v9038);	// L13967
      }
      ap_uint<128> packed_tmp151;	// L13969
      ap_uint<128> v9040 = v9017.read();	// L13970
      packed_tmp151 = v9040;	// L13971
      ap_uint<128> packed_c151;	// L13972
      packed_c151 = 0;	// L13973
      l_S_m_1_m151: for (int m151 = 0; m151 < 16; m151++) {	// L13974
        ap_int<33> v9043 = m151;	// L13975
        bool v9044 = v9043 == 9;	// L13976
        if (v9044) {	// L13977
          int8_t v9045 = c151;	// L13978
          ap_int<128> v9046 = packed_c151;	// L13979
          int64_t v9047 = m151;	// L13980
          int64_t v9048 = v9047 * 8;	// L13981
          ap_int<34> v9049 = m151;	// L13982
          ap_int<34> v9050 = v9049 + 1;	// L13983
          ap_int<66> v9051 = v9050;	// L13984
          ap_int<66> v9052 = v9051 * 8;	// L13985
          ap_int<66> v9053 = v9052 - 1;	// L13986
          int v9054 = v9048;	// L13987
          int v9055 = v9053;	// L13988
          ap_int<128> v9056;
          ap_int<128> v9056_tmp = v9046;
          v9056_tmp(v9055, v9054) = v9045;
          v9056 = v9056_tmp;	// L13989
          packed_c151 = v9056;	// L13990
        } else {
          ap_int<128> v9057 = packed_tmp151;	// L13992
          int64_t v9058 = m151;	// L13993
          int64_t v9059 = v9058 * 8;	// L13994
          ap_int<34> v9060 = m151;	// L13995
          ap_int<34> v9061 = v9060 + 1;	// L13996
          ap_int<66> v9062 = v9061;	// L13997
          ap_int<66> v9063 = v9062 * 8;	// L13998
          ap_int<66> v9064 = v9063 - 1;	// L13999
          int v9065 = v9059;	// L14000
          int v9066 = v9064;	// L14001
          int8_t v9067;
          ap_int<128> v9067_tmp = v9057;
          v9067 = v9067_tmp(v9066, v9065);	// L14002
          ap_int<128> v9068 = packed_c151;	// L14003
          ap_int<128> v9069;
          ap_int<128> v9069_tmp = v9068;
          v9069_tmp(v9066, v9065) = v9067;
          v9069 = v9069_tmp;	// L14013
          packed_c151 = v9069;	// L14014
        }
      }
      ap_int<128> v9070 = packed_c151;	// L14017
      v9018.write(v9070);	// L14018
    }
  }
}

void gemm_10_9(
  hls::stream< int8_t >& v9071,
  hls::stream< int8_t >& v9072,
  hls::stream< int8_t >& v9073,
  hls::stream< int8_t >& v9074,
  hls::stream< ap_uint<128> >& v9075,
  hls::stream< ap_uint<128> >& v9076
) {	// L14023
  l_S_mt_nt_0_mt181: for (int mt181 = 0; mt181 < 4; mt181++) {	// L14031
    l_nt181: for (int nt181 = 0; nt181 < 4; nt181++) {	// L14032
      int8_t c152;	// L14033
      c152 = 0;	// L14034
      l_S_k_0_k181: for (int k181 = 0; k181 < 64; k181++) {	// L14035
        int8_t v9081 = v9071.read();	// L14036
        int8_t a162;	// L14037
        a162 = v9081;	// L14038
        int8_t v9083 = v9072.read();	// L14039
        int8_t b168;	// L14040
        b168 = v9083;	// L14041
        int8_t v9085 = a162;	// L14042
        int8_t v9086 = b168;	// L14043
        int16_t v9087 = v9085;	// L14044
        int16_t v9088 = v9086;	// L14045
        int16_t v9089 = v9087 * v9088;	// L14046
        int8_t v9090 = c152;	// L14047
        ap_int<17> v9091 = v9090;	// L14048
        ap_int<17> v9092 = v9089;	// L14049
        ap_int<17> v9093 = v9091 + v9092;	// L14050
        int8_t v9094 = v9093;	// L14051
        c152 = v9094;	// L14052
        int8_t v9095 = a162;	// L14053
        v9073.write(v9095);	// L14054
        int8_t v9096 = b168;	// L14055
        v9074.write(v9096);	// L14056
      }
      ap_uint<128> packed_tmp152;	// L14058
      ap_uint<128> v9098 = v9075.read();	// L14059
      packed_tmp152 = v9098;	// L14060
      ap_uint<128> packed_c152;	// L14061
      packed_c152 = 0;	// L14062
      l_S_m_1_m152: for (int m152 = 0; m152 < 16; m152++) {	// L14063
        ap_int<33> v9101 = m152;	// L14064
        bool v9102 = v9101 == 9;	// L14065
        if (v9102) {	// L14066
          int8_t v9103 = c152;	// L14067
          ap_int<128> v9104 = packed_c152;	// L14068
          int64_t v9105 = m152;	// L14069
          int64_t v9106 = v9105 * 8;	// L14070
          ap_int<34> v9107 = m152;	// L14071
          ap_int<34> v9108 = v9107 + 1;	// L14072
          ap_int<66> v9109 = v9108;	// L14073
          ap_int<66> v9110 = v9109 * 8;	// L14074
          ap_int<66> v9111 = v9110 - 1;	// L14075
          int v9112 = v9106;	// L14076
          int v9113 = v9111;	// L14077
          ap_int<128> v9114;
          ap_int<128> v9114_tmp = v9104;
          v9114_tmp(v9113, v9112) = v9103;
          v9114 = v9114_tmp;	// L14078
          packed_c152 = v9114;	// L14079
        } else {
          ap_int<128> v9115 = packed_tmp152;	// L14081
          int64_t v9116 = m152;	// L14082
          int64_t v9117 = v9116 * 8;	// L14083
          ap_int<34> v9118 = m152;	// L14084
          ap_int<34> v9119 = v9118 + 1;	// L14085
          ap_int<66> v9120 = v9119;	// L14086
          ap_int<66> v9121 = v9120 * 8;	// L14087
          ap_int<66> v9122 = v9121 - 1;	// L14088
          int v9123 = v9117;	// L14089
          int v9124 = v9122;	// L14090
          int8_t v9125;
          ap_int<128> v9125_tmp = v9115;
          v9125 = v9125_tmp(v9124, v9123);	// L14091
          ap_int<128> v9126 = packed_c152;	// L14092
          ap_int<128> v9127;
          ap_int<128> v9127_tmp = v9126;
          v9127_tmp(v9124, v9123) = v9125;
          v9127 = v9127_tmp;	// L14102
          packed_c152 = v9127;	// L14103
        }
      }
      ap_int<128> v9128 = packed_c152;	// L14106
      v9076.write(v9128);	// L14107
    }
  }
}

void gemm_10_10(
  hls::stream< int8_t >& v9129,
  hls::stream< int8_t >& v9130,
  hls::stream< int8_t >& v9131,
  hls::stream< int8_t >& v9132,
  hls::stream< ap_uint<128> >& v9133,
  hls::stream< ap_uint<128> >& v9134
) {	// L14112
  l_S_mt_nt_0_mt182: for (int mt182 = 0; mt182 < 4; mt182++) {	// L14120
    l_nt182: for (int nt182 = 0; nt182 < 4; nt182++) {	// L14121
      int8_t c153;	// L14122
      c153 = 0;	// L14123
      l_S_k_0_k182: for (int k182 = 0; k182 < 64; k182++) {	// L14124
        int8_t v9139 = v9129.read();	// L14125
        int8_t a163;	// L14126
        a163 = v9139;	// L14127
        int8_t v9141 = v9130.read();	// L14128
        int8_t b169;	// L14129
        b169 = v9141;	// L14130
        int8_t v9143 = a163;	// L14131
        int8_t v9144 = b169;	// L14132
        int16_t v9145 = v9143;	// L14133
        int16_t v9146 = v9144;	// L14134
        int16_t v9147 = v9145 * v9146;	// L14135
        int8_t v9148 = c153;	// L14136
        ap_int<17> v9149 = v9148;	// L14137
        ap_int<17> v9150 = v9147;	// L14138
        ap_int<17> v9151 = v9149 + v9150;	// L14139
        int8_t v9152 = v9151;	// L14140
        c153 = v9152;	// L14141
        int8_t v9153 = a163;	// L14142
        v9131.write(v9153);	// L14143
        int8_t v9154 = b169;	// L14144
        v9132.write(v9154);	// L14145
      }
      ap_uint<128> packed_tmp153;	// L14147
      ap_uint<128> v9156 = v9133.read();	// L14148
      packed_tmp153 = v9156;	// L14149
      ap_uint<128> packed_c153;	// L14150
      packed_c153 = 0;	// L14151
      l_S_m_1_m153: for (int m153 = 0; m153 < 16; m153++) {	// L14152
        ap_int<33> v9159 = m153;	// L14153
        bool v9160 = v9159 == 9;	// L14154
        if (v9160) {	// L14155
          int8_t v9161 = c153;	// L14156
          ap_int<128> v9162 = packed_c153;	// L14157
          int64_t v9163 = m153;	// L14158
          int64_t v9164 = v9163 * 8;	// L14159
          ap_int<34> v9165 = m153;	// L14160
          ap_int<34> v9166 = v9165 + 1;	// L14161
          ap_int<66> v9167 = v9166;	// L14162
          ap_int<66> v9168 = v9167 * 8;	// L14163
          ap_int<66> v9169 = v9168 - 1;	// L14164
          int v9170 = v9164;	// L14165
          int v9171 = v9169;	// L14166
          ap_int<128> v9172;
          ap_int<128> v9172_tmp = v9162;
          v9172_tmp(v9171, v9170) = v9161;
          v9172 = v9172_tmp;	// L14167
          packed_c153 = v9172;	// L14168
        } else {
          ap_int<128> v9173 = packed_tmp153;	// L14170
          int64_t v9174 = m153;	// L14171
          int64_t v9175 = v9174 * 8;	// L14172
          ap_int<34> v9176 = m153;	// L14173
          ap_int<34> v9177 = v9176 + 1;	// L14174
          ap_int<66> v9178 = v9177;	// L14175
          ap_int<66> v9179 = v9178 * 8;	// L14176
          ap_int<66> v9180 = v9179 - 1;	// L14177
          int v9181 = v9175;	// L14178
          int v9182 = v9180;	// L14179
          int8_t v9183;
          ap_int<128> v9183_tmp = v9173;
          v9183 = v9183_tmp(v9182, v9181);	// L14180
          ap_int<128> v9184 = packed_c153;	// L14181
          ap_int<128> v9185;
          ap_int<128> v9185_tmp = v9184;
          v9185_tmp(v9182, v9181) = v9183;
          v9185 = v9185_tmp;	// L14191
          packed_c153 = v9185;	// L14192
        }
      }
      ap_int<128> v9186 = packed_c153;	// L14195
      v9134.write(v9186);	// L14196
    }
  }
}

void gemm_10_11(
  hls::stream< int8_t >& v9187,
  hls::stream< int8_t >& v9188,
  hls::stream< int8_t >& v9189,
  hls::stream< int8_t >& v9190,
  hls::stream< ap_uint<128> >& v9191,
  hls::stream< ap_uint<128> >& v9192
) {	// L14201
  l_S_mt_nt_0_mt183: for (int mt183 = 0; mt183 < 4; mt183++) {	// L14209
    l_nt183: for (int nt183 = 0; nt183 < 4; nt183++) {	// L14210
      int8_t c154;	// L14211
      c154 = 0;	// L14212
      l_S_k_0_k183: for (int k183 = 0; k183 < 64; k183++) {	// L14213
        int8_t v9197 = v9187.read();	// L14214
        int8_t a164;	// L14215
        a164 = v9197;	// L14216
        int8_t v9199 = v9188.read();	// L14217
        int8_t b170;	// L14218
        b170 = v9199;	// L14219
        int8_t v9201 = a164;	// L14220
        int8_t v9202 = b170;	// L14221
        int16_t v9203 = v9201;	// L14222
        int16_t v9204 = v9202;	// L14223
        int16_t v9205 = v9203 * v9204;	// L14224
        int8_t v9206 = c154;	// L14225
        ap_int<17> v9207 = v9206;	// L14226
        ap_int<17> v9208 = v9205;	// L14227
        ap_int<17> v9209 = v9207 + v9208;	// L14228
        int8_t v9210 = v9209;	// L14229
        c154 = v9210;	// L14230
        int8_t v9211 = a164;	// L14231
        v9189.write(v9211);	// L14232
        int8_t v9212 = b170;	// L14233
        v9190.write(v9212);	// L14234
      }
      ap_uint<128> packed_tmp154;	// L14236
      ap_uint<128> v9214 = v9191.read();	// L14237
      packed_tmp154 = v9214;	// L14238
      ap_uint<128> packed_c154;	// L14239
      packed_c154 = 0;	// L14240
      l_S_m_1_m154: for (int m154 = 0; m154 < 16; m154++) {	// L14241
        ap_int<33> v9217 = m154;	// L14242
        bool v9218 = v9217 == 9;	// L14243
        if (v9218) {	// L14244
          int8_t v9219 = c154;	// L14245
          ap_int<128> v9220 = packed_c154;	// L14246
          int64_t v9221 = m154;	// L14247
          int64_t v9222 = v9221 * 8;	// L14248
          ap_int<34> v9223 = m154;	// L14249
          ap_int<34> v9224 = v9223 + 1;	// L14250
          ap_int<66> v9225 = v9224;	// L14251
          ap_int<66> v9226 = v9225 * 8;	// L14252
          ap_int<66> v9227 = v9226 - 1;	// L14253
          int v9228 = v9222;	// L14254
          int v9229 = v9227;	// L14255
          ap_int<128> v9230;
          ap_int<128> v9230_tmp = v9220;
          v9230_tmp(v9229, v9228) = v9219;
          v9230 = v9230_tmp;	// L14256
          packed_c154 = v9230;	// L14257
        } else {
          ap_int<128> v9231 = packed_tmp154;	// L14259
          int64_t v9232 = m154;	// L14260
          int64_t v9233 = v9232 * 8;	// L14261
          ap_int<34> v9234 = m154;	// L14262
          ap_int<34> v9235 = v9234 + 1;	// L14263
          ap_int<66> v9236 = v9235;	// L14264
          ap_int<66> v9237 = v9236 * 8;	// L14265
          ap_int<66> v9238 = v9237 - 1;	// L14266
          int v9239 = v9233;	// L14267
          int v9240 = v9238;	// L14268
          int8_t v9241;
          ap_int<128> v9241_tmp = v9231;
          v9241 = v9241_tmp(v9240, v9239);	// L14269
          ap_int<128> v9242 = packed_c154;	// L14270
          ap_int<128> v9243;
          ap_int<128> v9243_tmp = v9242;
          v9243_tmp(v9240, v9239) = v9241;
          v9243 = v9243_tmp;	// L14280
          packed_c154 = v9243;	// L14281
        }
      }
      ap_int<128> v9244 = packed_c154;	// L14284
      v9192.write(v9244);	// L14285
    }
  }
}

void gemm_10_12(
  hls::stream< int8_t >& v9245,
  hls::stream< int8_t >& v9246,
  hls::stream< int8_t >& v9247,
  hls::stream< int8_t >& v9248,
  hls::stream< ap_uint<128> >& v9249,
  hls::stream< ap_uint<128> >& v9250
) {	// L14290
  l_S_mt_nt_0_mt184: for (int mt184 = 0; mt184 < 4; mt184++) {	// L14298
    l_nt184: for (int nt184 = 0; nt184 < 4; nt184++) {	// L14299
      int8_t c155;	// L14300
      c155 = 0;	// L14301
      l_S_k_0_k184: for (int k184 = 0; k184 < 64; k184++) {	// L14302
        int8_t v9255 = v9245.read();	// L14303
        int8_t a165;	// L14304
        a165 = v9255;	// L14305
        int8_t v9257 = v9246.read();	// L14306
        int8_t b171;	// L14307
        b171 = v9257;	// L14308
        int8_t v9259 = a165;	// L14309
        int8_t v9260 = b171;	// L14310
        int16_t v9261 = v9259;	// L14311
        int16_t v9262 = v9260;	// L14312
        int16_t v9263 = v9261 * v9262;	// L14313
        int8_t v9264 = c155;	// L14314
        ap_int<17> v9265 = v9264;	// L14315
        ap_int<17> v9266 = v9263;	// L14316
        ap_int<17> v9267 = v9265 + v9266;	// L14317
        int8_t v9268 = v9267;	// L14318
        c155 = v9268;	// L14319
        int8_t v9269 = a165;	// L14320
        v9247.write(v9269);	// L14321
        int8_t v9270 = b171;	// L14322
        v9248.write(v9270);	// L14323
      }
      ap_uint<128> packed_tmp155;	// L14325
      ap_uint<128> v9272 = v9249.read();	// L14326
      packed_tmp155 = v9272;	// L14327
      ap_uint<128> packed_c155;	// L14328
      packed_c155 = 0;	// L14329
      l_S_m_1_m155: for (int m155 = 0; m155 < 16; m155++) {	// L14330
        ap_int<33> v9275 = m155;	// L14331
        bool v9276 = v9275 == 9;	// L14332
        if (v9276) {	// L14333
          int8_t v9277 = c155;	// L14334
          ap_int<128> v9278 = packed_c155;	// L14335
          int64_t v9279 = m155;	// L14336
          int64_t v9280 = v9279 * 8;	// L14337
          ap_int<34> v9281 = m155;	// L14338
          ap_int<34> v9282 = v9281 + 1;	// L14339
          ap_int<66> v9283 = v9282;	// L14340
          ap_int<66> v9284 = v9283 * 8;	// L14341
          ap_int<66> v9285 = v9284 - 1;	// L14342
          int v9286 = v9280;	// L14343
          int v9287 = v9285;	// L14344
          ap_int<128> v9288;
          ap_int<128> v9288_tmp = v9278;
          v9288_tmp(v9287, v9286) = v9277;
          v9288 = v9288_tmp;	// L14345
          packed_c155 = v9288;	// L14346
        } else {
          ap_int<128> v9289 = packed_tmp155;	// L14348
          int64_t v9290 = m155;	// L14349
          int64_t v9291 = v9290 * 8;	// L14350
          ap_int<34> v9292 = m155;	// L14351
          ap_int<34> v9293 = v9292 + 1;	// L14352
          ap_int<66> v9294 = v9293;	// L14353
          ap_int<66> v9295 = v9294 * 8;	// L14354
          ap_int<66> v9296 = v9295 - 1;	// L14355
          int v9297 = v9291;	// L14356
          int v9298 = v9296;	// L14357
          int8_t v9299;
          ap_int<128> v9299_tmp = v9289;
          v9299 = v9299_tmp(v9298, v9297);	// L14358
          ap_int<128> v9300 = packed_c155;	// L14359
          ap_int<128> v9301;
          ap_int<128> v9301_tmp = v9300;
          v9301_tmp(v9298, v9297) = v9299;
          v9301 = v9301_tmp;	// L14369
          packed_c155 = v9301;	// L14370
        }
      }
      ap_int<128> v9302 = packed_c155;	// L14373
      v9250.write(v9302);	// L14374
    }
  }
}

void gemm_10_13(
  hls::stream< int8_t >& v9303,
  hls::stream< int8_t >& v9304,
  hls::stream< int8_t >& v9305,
  hls::stream< int8_t >& v9306,
  hls::stream< ap_uint<128> >& v9307,
  hls::stream< ap_uint<128> >& v9308
) {	// L14379
  l_S_mt_nt_0_mt185: for (int mt185 = 0; mt185 < 4; mt185++) {	// L14387
    l_nt185: for (int nt185 = 0; nt185 < 4; nt185++) {	// L14388
      int8_t c156;	// L14389
      c156 = 0;	// L14390
      l_S_k_0_k185: for (int k185 = 0; k185 < 64; k185++) {	// L14391
        int8_t v9313 = v9303.read();	// L14392
        int8_t a166;	// L14393
        a166 = v9313;	// L14394
        int8_t v9315 = v9304.read();	// L14395
        int8_t b172;	// L14396
        b172 = v9315;	// L14397
        int8_t v9317 = a166;	// L14398
        int8_t v9318 = b172;	// L14399
        int16_t v9319 = v9317;	// L14400
        int16_t v9320 = v9318;	// L14401
        int16_t v9321 = v9319 * v9320;	// L14402
        int8_t v9322 = c156;	// L14403
        ap_int<17> v9323 = v9322;	// L14404
        ap_int<17> v9324 = v9321;	// L14405
        ap_int<17> v9325 = v9323 + v9324;	// L14406
        int8_t v9326 = v9325;	// L14407
        c156 = v9326;	// L14408
        int8_t v9327 = a166;	// L14409
        v9305.write(v9327);	// L14410
        int8_t v9328 = b172;	// L14411
        v9306.write(v9328);	// L14412
      }
      ap_uint<128> packed_tmp156;	// L14414
      ap_uint<128> v9330 = v9307.read();	// L14415
      packed_tmp156 = v9330;	// L14416
      ap_uint<128> packed_c156;	// L14417
      packed_c156 = 0;	// L14418
      l_S_m_1_m156: for (int m156 = 0; m156 < 16; m156++) {	// L14419
        ap_int<33> v9333 = m156;	// L14420
        bool v9334 = v9333 == 9;	// L14421
        if (v9334) {	// L14422
          int8_t v9335 = c156;	// L14423
          ap_int<128> v9336 = packed_c156;	// L14424
          int64_t v9337 = m156;	// L14425
          int64_t v9338 = v9337 * 8;	// L14426
          ap_int<34> v9339 = m156;	// L14427
          ap_int<34> v9340 = v9339 + 1;	// L14428
          ap_int<66> v9341 = v9340;	// L14429
          ap_int<66> v9342 = v9341 * 8;	// L14430
          ap_int<66> v9343 = v9342 - 1;	// L14431
          int v9344 = v9338;	// L14432
          int v9345 = v9343;	// L14433
          ap_int<128> v9346;
          ap_int<128> v9346_tmp = v9336;
          v9346_tmp(v9345, v9344) = v9335;
          v9346 = v9346_tmp;	// L14434
          packed_c156 = v9346;	// L14435
        } else {
          ap_int<128> v9347 = packed_tmp156;	// L14437
          int64_t v9348 = m156;	// L14438
          int64_t v9349 = v9348 * 8;	// L14439
          ap_int<34> v9350 = m156;	// L14440
          ap_int<34> v9351 = v9350 + 1;	// L14441
          ap_int<66> v9352 = v9351;	// L14442
          ap_int<66> v9353 = v9352 * 8;	// L14443
          ap_int<66> v9354 = v9353 - 1;	// L14444
          int v9355 = v9349;	// L14445
          int v9356 = v9354;	// L14446
          int8_t v9357;
          ap_int<128> v9357_tmp = v9347;
          v9357 = v9357_tmp(v9356, v9355);	// L14447
          ap_int<128> v9358 = packed_c156;	// L14448
          ap_int<128> v9359;
          ap_int<128> v9359_tmp = v9358;
          v9359_tmp(v9356, v9355) = v9357;
          v9359 = v9359_tmp;	// L14458
          packed_c156 = v9359;	// L14459
        }
      }
      ap_int<128> v9360 = packed_c156;	// L14462
      v9308.write(v9360);	// L14463
    }
  }
}

void gemm_10_14(
  hls::stream< int8_t >& v9361,
  hls::stream< int8_t >& v9362,
  hls::stream< int8_t >& v9363,
  hls::stream< int8_t >& v9364,
  hls::stream< ap_uint<128> >& v9365,
  hls::stream< ap_uint<128> >& v9366
) {	// L14468
  l_S_mt_nt_0_mt186: for (int mt186 = 0; mt186 < 4; mt186++) {	// L14476
    l_nt186: for (int nt186 = 0; nt186 < 4; nt186++) {	// L14477
      int8_t c157;	// L14478
      c157 = 0;	// L14479
      l_S_k_0_k186: for (int k186 = 0; k186 < 64; k186++) {	// L14480
        int8_t v9371 = v9361.read();	// L14481
        int8_t a167;	// L14482
        a167 = v9371;	// L14483
        int8_t v9373 = v9362.read();	// L14484
        int8_t b173;	// L14485
        b173 = v9373;	// L14486
        int8_t v9375 = a167;	// L14487
        int8_t v9376 = b173;	// L14488
        int16_t v9377 = v9375;	// L14489
        int16_t v9378 = v9376;	// L14490
        int16_t v9379 = v9377 * v9378;	// L14491
        int8_t v9380 = c157;	// L14492
        ap_int<17> v9381 = v9380;	// L14493
        ap_int<17> v9382 = v9379;	// L14494
        ap_int<17> v9383 = v9381 + v9382;	// L14495
        int8_t v9384 = v9383;	// L14496
        c157 = v9384;	// L14497
        int8_t v9385 = a167;	// L14498
        v9363.write(v9385);	// L14499
        int8_t v9386 = b173;	// L14500
        v9364.write(v9386);	// L14501
      }
      ap_uint<128> packed_tmp157;	// L14503
      ap_uint<128> v9388 = v9365.read();	// L14504
      packed_tmp157 = v9388;	// L14505
      ap_uint<128> packed_c157;	// L14506
      packed_c157 = 0;	// L14507
      l_S_m_1_m157: for (int m157 = 0; m157 < 16; m157++) {	// L14508
        ap_int<33> v9391 = m157;	// L14509
        bool v9392 = v9391 == 9;	// L14510
        if (v9392) {	// L14511
          int8_t v9393 = c157;	// L14512
          ap_int<128> v9394 = packed_c157;	// L14513
          int64_t v9395 = m157;	// L14514
          int64_t v9396 = v9395 * 8;	// L14515
          ap_int<34> v9397 = m157;	// L14516
          ap_int<34> v9398 = v9397 + 1;	// L14517
          ap_int<66> v9399 = v9398;	// L14518
          ap_int<66> v9400 = v9399 * 8;	// L14519
          ap_int<66> v9401 = v9400 - 1;	// L14520
          int v9402 = v9396;	// L14521
          int v9403 = v9401;	// L14522
          ap_int<128> v9404;
          ap_int<128> v9404_tmp = v9394;
          v9404_tmp(v9403, v9402) = v9393;
          v9404 = v9404_tmp;	// L14523
          packed_c157 = v9404;	// L14524
        } else {
          ap_int<128> v9405 = packed_tmp157;	// L14526
          int64_t v9406 = m157;	// L14527
          int64_t v9407 = v9406 * 8;	// L14528
          ap_int<34> v9408 = m157;	// L14529
          ap_int<34> v9409 = v9408 + 1;	// L14530
          ap_int<66> v9410 = v9409;	// L14531
          ap_int<66> v9411 = v9410 * 8;	// L14532
          ap_int<66> v9412 = v9411 - 1;	// L14533
          int v9413 = v9407;	// L14534
          int v9414 = v9412;	// L14535
          int8_t v9415;
          ap_int<128> v9415_tmp = v9405;
          v9415 = v9415_tmp(v9414, v9413);	// L14536
          ap_int<128> v9416 = packed_c157;	// L14537
          ap_int<128> v9417;
          ap_int<128> v9417_tmp = v9416;
          v9417_tmp(v9414, v9413) = v9415;
          v9417 = v9417_tmp;	// L14547
          packed_c157 = v9417;	// L14548
        }
      }
      ap_int<128> v9418 = packed_c157;	// L14551
      v9366.write(v9418);	// L14552
    }
  }
}

void gemm_10_15(
  hls::stream< int8_t >& v9419,
  hls::stream< int8_t >& v9420,
  hls::stream< int8_t >& v9421,
  hls::stream< int8_t >& v9422,
  hls::stream< ap_uint<128> >& v9423,
  hls::stream< ap_uint<128> >& v9424
) {	// L14557
  l_S_mt_nt_0_mt187: for (int mt187 = 0; mt187 < 4; mt187++) {	// L14565
    l_nt187: for (int nt187 = 0; nt187 < 4; nt187++) {	// L14566
      int8_t c158;	// L14567
      c158 = 0;	// L14568
      l_S_k_0_k187: for (int k187 = 0; k187 < 64; k187++) {	// L14569
        int8_t v9429 = v9419.read();	// L14570
        int8_t a168;	// L14571
        a168 = v9429;	// L14572
        int8_t v9431 = v9420.read();	// L14573
        int8_t b174;	// L14574
        b174 = v9431;	// L14575
        int8_t v9433 = a168;	// L14576
        int8_t v9434 = b174;	// L14577
        int16_t v9435 = v9433;	// L14578
        int16_t v9436 = v9434;	// L14579
        int16_t v9437 = v9435 * v9436;	// L14580
        int8_t v9438 = c158;	// L14581
        ap_int<17> v9439 = v9438;	// L14582
        ap_int<17> v9440 = v9437;	// L14583
        ap_int<17> v9441 = v9439 + v9440;	// L14584
        int8_t v9442 = v9441;	// L14585
        c158 = v9442;	// L14586
        int8_t v9443 = a168;	// L14587
        v9421.write(v9443);	// L14588
        int8_t v9444 = b174;	// L14589
        v9422.write(v9444);	// L14590
      }
      ap_uint<128> packed_tmp158;	// L14592
      ap_uint<128> v9446 = v9423.read();	// L14593
      packed_tmp158 = v9446;	// L14594
      ap_uint<128> packed_c158;	// L14595
      packed_c158 = 0;	// L14596
      l_S_m_1_m158: for (int m158 = 0; m158 < 16; m158++) {	// L14597
        ap_int<33> v9449 = m158;	// L14598
        bool v9450 = v9449 == 9;	// L14599
        if (v9450) {	// L14600
          int8_t v9451 = c158;	// L14601
          ap_int<128> v9452 = packed_c158;	// L14602
          int64_t v9453 = m158;	// L14603
          int64_t v9454 = v9453 * 8;	// L14604
          ap_int<34> v9455 = m158;	// L14605
          ap_int<34> v9456 = v9455 + 1;	// L14606
          ap_int<66> v9457 = v9456;	// L14607
          ap_int<66> v9458 = v9457 * 8;	// L14608
          ap_int<66> v9459 = v9458 - 1;	// L14609
          int v9460 = v9454;	// L14610
          int v9461 = v9459;	// L14611
          ap_int<128> v9462;
          ap_int<128> v9462_tmp = v9452;
          v9462_tmp(v9461, v9460) = v9451;
          v9462 = v9462_tmp;	// L14612
          packed_c158 = v9462;	// L14613
        } else {
          ap_int<128> v9463 = packed_tmp158;	// L14615
          int64_t v9464 = m158;	// L14616
          int64_t v9465 = v9464 * 8;	// L14617
          ap_int<34> v9466 = m158;	// L14618
          ap_int<34> v9467 = v9466 + 1;	// L14619
          ap_int<66> v9468 = v9467;	// L14620
          ap_int<66> v9469 = v9468 * 8;	// L14621
          ap_int<66> v9470 = v9469 - 1;	// L14622
          int v9471 = v9465;	// L14623
          int v9472 = v9470;	// L14624
          int8_t v9473;
          ap_int<128> v9473_tmp = v9463;
          v9473 = v9473_tmp(v9472, v9471);	// L14625
          ap_int<128> v9474 = packed_c158;	// L14626
          ap_int<128> v9475;
          ap_int<128> v9475_tmp = v9474;
          v9475_tmp(v9472, v9471) = v9473;
          v9475 = v9475_tmp;	// L14636
          packed_c158 = v9475;	// L14637
        }
      }
      ap_int<128> v9476 = packed_c158;	// L14640
      v9424.write(v9476);	// L14641
    }
  }
}

void gemm_10_16(
  hls::stream< int8_t >& v9477,
  hls::stream< int8_t >& v9478,
  hls::stream< int8_t >& v9479,
  hls::stream< ap_uint<128> >& v9480,
  hls::stream< ap_uint<128> >& v9481
) {	// L14646
  l_S_mt_nt_0_mt188: for (int mt188 = 0; mt188 < 4; mt188++) {	// L14654
    l_nt188: for (int nt188 = 0; nt188 < 4; nt188++) {	// L14655
      int8_t c159;	// L14656
      c159 = 0;	// L14657
      l_S_k_0_k188: for (int k188 = 0; k188 < 64; k188++) {	// L14658
        int8_t v9486 = v9477.read();	// L14659
        int8_t a169;	// L14660
        a169 = v9486;	// L14661
        int8_t v9488 = v9478.read();	// L14662
        int8_t b175;	// L14663
        b175 = v9488;	// L14664
        int8_t v9490 = a169;	// L14665
        int8_t v9491 = b175;	// L14666
        int16_t v9492 = v9490;	// L14667
        int16_t v9493 = v9491;	// L14668
        int16_t v9494 = v9492 * v9493;	// L14669
        int8_t v9495 = c159;	// L14670
        ap_int<17> v9496 = v9495;	// L14671
        ap_int<17> v9497 = v9494;	// L14672
        ap_int<17> v9498 = v9496 + v9497;	// L14673
        int8_t v9499 = v9498;	// L14674
        c159 = v9499;	// L14675
        int8_t v9500 = b175;	// L14676
        v9479.write(v9500);	// L14677
      }
      ap_uint<128> packed_tmp159;	// L14679
      ap_uint<128> v9502 = v9480.read();	// L14680
      packed_tmp159 = v9502;	// L14681
      ap_uint<128> packed_c159;	// L14682
      packed_c159 = 0;	// L14683
      l_S_m_1_m159: for (int m159 = 0; m159 < 16; m159++) {	// L14684
        ap_int<33> v9505 = m159;	// L14685
        bool v9506 = v9505 == 9;	// L14686
        if (v9506) {	// L14687
          int8_t v9507 = c159;	// L14688
          ap_int<128> v9508 = packed_c159;	// L14689
          int64_t v9509 = m159;	// L14690
          int64_t v9510 = v9509 * 8;	// L14691
          ap_int<34> v9511 = m159;	// L14692
          ap_int<34> v9512 = v9511 + 1;	// L14693
          ap_int<66> v9513 = v9512;	// L14694
          ap_int<66> v9514 = v9513 * 8;	// L14695
          ap_int<66> v9515 = v9514 - 1;	// L14696
          int v9516 = v9510;	// L14697
          int v9517 = v9515;	// L14698
          ap_int<128> v9518;
          ap_int<128> v9518_tmp = v9508;
          v9518_tmp(v9517, v9516) = v9507;
          v9518 = v9518_tmp;	// L14699
          packed_c159 = v9518;	// L14700
        } else {
          ap_int<128> v9519 = packed_tmp159;	// L14702
          int64_t v9520 = m159;	// L14703
          int64_t v9521 = v9520 * 8;	// L14704
          ap_int<34> v9522 = m159;	// L14705
          ap_int<34> v9523 = v9522 + 1;	// L14706
          ap_int<66> v9524 = v9523;	// L14707
          ap_int<66> v9525 = v9524 * 8;	// L14708
          ap_int<66> v9526 = v9525 - 1;	// L14709
          int v9527 = v9521;	// L14710
          int v9528 = v9526;	// L14711
          int8_t v9529;
          ap_int<128> v9529_tmp = v9519;
          v9529 = v9529_tmp(v9528, v9527);	// L14712
          ap_int<128> v9530 = packed_c159;	// L14713
          ap_int<128> v9531;
          ap_int<128> v9531_tmp = v9530;
          v9531_tmp(v9528, v9527) = v9529;
          v9531 = v9531_tmp;	// L14723
          packed_c159 = v9531;	// L14724
        }
      }
      ap_int<128> v9532 = packed_c159;	// L14727
      v9481.write(v9532);	// L14728
    }
  }
}

void gemm_11_0(
  hls::stream< ap_uint<128> >& v9533,
  hls::stream< int8_t >& v9534,
  hls::stream< ap_uint<128> >& v9535
) {	// L14733
  l_S_mt_nt_0_mt189: for (int mt189 = 0; mt189 < 4; mt189++) {	// L14736
    l_nt189: for (int nt189 = 0; nt189 < 4; nt189++) {	// L14737
      l_S_k_0_k189: for (int k189 = 0; k189 < 64; k189++) {	// L14738
        ap_uint<128> v9539 = v9533.read();	// L14739
        ap_uint<128> a170;	// L14740
        a170 = v9539;	// L14741
        ap_int<128> v9541 = a170;	// L14742
        int8_t v9542;
        ap_int<128> v9542_tmp = v9541;
        v9542 = v9542_tmp(87, 80);	// L14743
        v9534.write(v9542);	// L14744
        ap_int<128> v9543 = a170;	// L14745
        v9535.write(v9543);	// L14746
      }
    }
  }
}

void gemm_11_1(
  hls::stream< int8_t >& v9544,
  hls::stream< int8_t >& v9545,
  hls::stream< int8_t >& v9546,
  hls::stream< int8_t >& v9547,
  hls::stream< ap_uint<128> >& v9548,
  hls::stream< ap_uint<128> >& v9549
) {	// L14752
  l_S_mt_nt_0_mt190: for (int mt190 = 0; mt190 < 4; mt190++) {	// L14760
    l_nt190: for (int nt190 = 0; nt190 < 4; nt190++) {	// L14761
      int8_t c160;	// L14762
      c160 = 0;	// L14763
      l_S_k_0_k190: for (int k190 = 0; k190 < 64; k190++) {	// L14764
        int8_t v9554 = v9544.read();	// L14765
        int8_t a171;	// L14766
        a171 = v9554;	// L14767
        int8_t v9556 = v9545.read();	// L14768
        int8_t b176;	// L14769
        b176 = v9556;	// L14770
        int8_t v9558 = a171;	// L14771
        int8_t v9559 = b176;	// L14772
        int16_t v9560 = v9558;	// L14773
        int16_t v9561 = v9559;	// L14774
        int16_t v9562 = v9560 * v9561;	// L14775
        int8_t v9563 = c160;	// L14776
        ap_int<17> v9564 = v9563;	// L14777
        ap_int<17> v9565 = v9562;	// L14778
        ap_int<17> v9566 = v9564 + v9565;	// L14779
        int8_t v9567 = v9566;	// L14780
        c160 = v9567;	// L14781
        int8_t v9568 = a171;	// L14782
        v9546.write(v9568);	// L14783
        int8_t v9569 = b176;	// L14784
        v9547.write(v9569);	// L14785
      }
      ap_uint<128> packed_tmp160;	// L14787
      ap_uint<128> v9571 = v9548.read();	// L14788
      packed_tmp160 = v9571;	// L14789
      ap_uint<128> packed_c160;	// L14790
      packed_c160 = 0;	// L14791
      l_S_m_1_m160: for (int m160 = 0; m160 < 16; m160++) {	// L14792
        ap_int<33> v9574 = m160;	// L14793
        bool v9575 = v9574 == 10;	// L14794
        if (v9575) {	// L14795
          int8_t v9576 = c160;	// L14796
          ap_int<128> v9577 = packed_c160;	// L14797
          int64_t v9578 = m160;	// L14798
          int64_t v9579 = v9578 * 8;	// L14799
          ap_int<34> v9580 = m160;	// L14800
          ap_int<34> v9581 = v9580 + 1;	// L14801
          ap_int<66> v9582 = v9581;	// L14802
          ap_int<66> v9583 = v9582 * 8;	// L14803
          ap_int<66> v9584 = v9583 - 1;	// L14804
          int v9585 = v9579;	// L14805
          int v9586 = v9584;	// L14806
          ap_int<128> v9587;
          ap_int<128> v9587_tmp = v9577;
          v9587_tmp(v9586, v9585) = v9576;
          v9587 = v9587_tmp;	// L14807
          packed_c160 = v9587;	// L14808
        } else {
          ap_int<128> v9588 = packed_tmp160;	// L14810
          int64_t v9589 = m160;	// L14811
          int64_t v9590 = v9589 * 8;	// L14812
          ap_int<34> v9591 = m160;	// L14813
          ap_int<34> v9592 = v9591 + 1;	// L14814
          ap_int<66> v9593 = v9592;	// L14815
          ap_int<66> v9594 = v9593 * 8;	// L14816
          ap_int<66> v9595 = v9594 - 1;	// L14817
          int v9596 = v9590;	// L14818
          int v9597 = v9595;	// L14819
          int8_t v9598;
          ap_int<128> v9598_tmp = v9588;
          v9598 = v9598_tmp(v9597, v9596);	// L14820
          ap_int<128> v9599 = packed_c160;	// L14821
          ap_int<128> v9600;
          ap_int<128> v9600_tmp = v9599;
          v9600_tmp(v9597, v9596) = v9598;
          v9600 = v9600_tmp;	// L14831
          packed_c160 = v9600;	// L14832
        }
      }
      ap_int<128> v9601 = packed_c160;	// L14835
      v9549.write(v9601);	// L14836
    }
  }
}

void gemm_11_2(
  hls::stream< int8_t >& v9602,
  hls::stream< int8_t >& v9603,
  hls::stream< int8_t >& v9604,
  hls::stream< int8_t >& v9605,
  hls::stream< ap_uint<128> >& v9606,
  hls::stream< ap_uint<128> >& v9607
) {	// L14841
  l_S_mt_nt_0_mt191: for (int mt191 = 0; mt191 < 4; mt191++) {	// L14849
    l_nt191: for (int nt191 = 0; nt191 < 4; nt191++) {	// L14850
      int8_t c161;	// L14851
      c161 = 0;	// L14852
      l_S_k_0_k191: for (int k191 = 0; k191 < 64; k191++) {	// L14853
        int8_t v9612 = v9602.read();	// L14854
        int8_t a172;	// L14855
        a172 = v9612;	// L14856
        int8_t v9614 = v9603.read();	// L14857
        int8_t b177;	// L14858
        b177 = v9614;	// L14859
        int8_t v9616 = a172;	// L14860
        int8_t v9617 = b177;	// L14861
        int16_t v9618 = v9616;	// L14862
        int16_t v9619 = v9617;	// L14863
        int16_t v9620 = v9618 * v9619;	// L14864
        int8_t v9621 = c161;	// L14865
        ap_int<17> v9622 = v9621;	// L14866
        ap_int<17> v9623 = v9620;	// L14867
        ap_int<17> v9624 = v9622 + v9623;	// L14868
        int8_t v9625 = v9624;	// L14869
        c161 = v9625;	// L14870
        int8_t v9626 = a172;	// L14871
        v9604.write(v9626);	// L14872
        int8_t v9627 = b177;	// L14873
        v9605.write(v9627);	// L14874
      }
      ap_uint<128> packed_tmp161;	// L14876
      ap_uint<128> v9629 = v9606.read();	// L14877
      packed_tmp161 = v9629;	// L14878
      ap_uint<128> packed_c161;	// L14879
      packed_c161 = 0;	// L14880
      l_S_m_1_m161: for (int m161 = 0; m161 < 16; m161++) {	// L14881
        ap_int<33> v9632 = m161;	// L14882
        bool v9633 = v9632 == 10;	// L14883
        if (v9633) {	// L14884
          int8_t v9634 = c161;	// L14885
          ap_int<128> v9635 = packed_c161;	// L14886
          int64_t v9636 = m161;	// L14887
          int64_t v9637 = v9636 * 8;	// L14888
          ap_int<34> v9638 = m161;	// L14889
          ap_int<34> v9639 = v9638 + 1;	// L14890
          ap_int<66> v9640 = v9639;	// L14891
          ap_int<66> v9641 = v9640 * 8;	// L14892
          ap_int<66> v9642 = v9641 - 1;	// L14893
          int v9643 = v9637;	// L14894
          int v9644 = v9642;	// L14895
          ap_int<128> v9645;
          ap_int<128> v9645_tmp = v9635;
          v9645_tmp(v9644, v9643) = v9634;
          v9645 = v9645_tmp;	// L14896
          packed_c161 = v9645;	// L14897
        } else {
          ap_int<128> v9646 = packed_tmp161;	// L14899
          int64_t v9647 = m161;	// L14900
          int64_t v9648 = v9647 * 8;	// L14901
          ap_int<34> v9649 = m161;	// L14902
          ap_int<34> v9650 = v9649 + 1;	// L14903
          ap_int<66> v9651 = v9650;	// L14904
          ap_int<66> v9652 = v9651 * 8;	// L14905
          ap_int<66> v9653 = v9652 - 1;	// L14906
          int v9654 = v9648;	// L14907
          int v9655 = v9653;	// L14908
          int8_t v9656;
          ap_int<128> v9656_tmp = v9646;
          v9656 = v9656_tmp(v9655, v9654);	// L14909
          ap_int<128> v9657 = packed_c161;	// L14910
          ap_int<128> v9658;
          ap_int<128> v9658_tmp = v9657;
          v9658_tmp(v9655, v9654) = v9656;
          v9658 = v9658_tmp;	// L14920
          packed_c161 = v9658;	// L14921
        }
      }
      ap_int<128> v9659 = packed_c161;	// L14924
      v9607.write(v9659);	// L14925
    }
  }
}

void gemm_11_3(
  hls::stream< int8_t >& v9660,
  hls::stream< int8_t >& v9661,
  hls::stream< int8_t >& v9662,
  hls::stream< int8_t >& v9663,
  hls::stream< ap_uint<128> >& v9664,
  hls::stream< ap_uint<128> >& v9665
) {	// L14930
  l_S_mt_nt_0_mt192: for (int mt192 = 0; mt192 < 4; mt192++) {	// L14938
    l_nt192: for (int nt192 = 0; nt192 < 4; nt192++) {	// L14939
      int8_t c162;	// L14940
      c162 = 0;	// L14941
      l_S_k_0_k192: for (int k192 = 0; k192 < 64; k192++) {	// L14942
        int8_t v9670 = v9660.read();	// L14943
        int8_t a173;	// L14944
        a173 = v9670;	// L14945
        int8_t v9672 = v9661.read();	// L14946
        int8_t b178;	// L14947
        b178 = v9672;	// L14948
        int8_t v9674 = a173;	// L14949
        int8_t v9675 = b178;	// L14950
        int16_t v9676 = v9674;	// L14951
        int16_t v9677 = v9675;	// L14952
        int16_t v9678 = v9676 * v9677;	// L14953
        int8_t v9679 = c162;	// L14954
        ap_int<17> v9680 = v9679;	// L14955
        ap_int<17> v9681 = v9678;	// L14956
        ap_int<17> v9682 = v9680 + v9681;	// L14957
        int8_t v9683 = v9682;	// L14958
        c162 = v9683;	// L14959
        int8_t v9684 = a173;	// L14960
        v9662.write(v9684);	// L14961
        int8_t v9685 = b178;	// L14962
        v9663.write(v9685);	// L14963
      }
      ap_uint<128> packed_tmp162;	// L14965
      ap_uint<128> v9687 = v9664.read();	// L14966
      packed_tmp162 = v9687;	// L14967
      ap_uint<128> packed_c162;	// L14968
      packed_c162 = 0;	// L14969
      l_S_m_1_m162: for (int m162 = 0; m162 < 16; m162++) {	// L14970
        ap_int<33> v9690 = m162;	// L14971
        bool v9691 = v9690 == 10;	// L14972
        if (v9691) {	// L14973
          int8_t v9692 = c162;	// L14974
          ap_int<128> v9693 = packed_c162;	// L14975
          int64_t v9694 = m162;	// L14976
          int64_t v9695 = v9694 * 8;	// L14977
          ap_int<34> v9696 = m162;	// L14978
          ap_int<34> v9697 = v9696 + 1;	// L14979
          ap_int<66> v9698 = v9697;	// L14980
          ap_int<66> v9699 = v9698 * 8;	// L14981
          ap_int<66> v9700 = v9699 - 1;	// L14982
          int v9701 = v9695;	// L14983
          int v9702 = v9700;	// L14984
          ap_int<128> v9703;
          ap_int<128> v9703_tmp = v9693;
          v9703_tmp(v9702, v9701) = v9692;
          v9703 = v9703_tmp;	// L14985
          packed_c162 = v9703;	// L14986
        } else {
          ap_int<128> v9704 = packed_tmp162;	// L14988
          int64_t v9705 = m162;	// L14989
          int64_t v9706 = v9705 * 8;	// L14990
          ap_int<34> v9707 = m162;	// L14991
          ap_int<34> v9708 = v9707 + 1;	// L14992
          ap_int<66> v9709 = v9708;	// L14993
          ap_int<66> v9710 = v9709 * 8;	// L14994
          ap_int<66> v9711 = v9710 - 1;	// L14995
          int v9712 = v9706;	// L14996
          int v9713 = v9711;	// L14997
          int8_t v9714;
          ap_int<128> v9714_tmp = v9704;
          v9714 = v9714_tmp(v9713, v9712);	// L14998
          ap_int<128> v9715 = packed_c162;	// L14999
          ap_int<128> v9716;
          ap_int<128> v9716_tmp = v9715;
          v9716_tmp(v9713, v9712) = v9714;
          v9716 = v9716_tmp;	// L15009
          packed_c162 = v9716;	// L15010
        }
      }
      ap_int<128> v9717 = packed_c162;	// L15013
      v9665.write(v9717);	// L15014
    }
  }
}

void gemm_11_4(
  hls::stream< int8_t >& v9718,
  hls::stream< int8_t >& v9719,
  hls::stream< int8_t >& v9720,
  hls::stream< int8_t >& v9721,
  hls::stream< ap_uint<128> >& v9722,
  hls::stream< ap_uint<128> >& v9723
) {	// L15019
  l_S_mt_nt_0_mt193: for (int mt193 = 0; mt193 < 4; mt193++) {	// L15027
    l_nt193: for (int nt193 = 0; nt193 < 4; nt193++) {	// L15028
      int8_t c163;	// L15029
      c163 = 0;	// L15030
      l_S_k_0_k193: for (int k193 = 0; k193 < 64; k193++) {	// L15031
        int8_t v9728 = v9718.read();	// L15032
        int8_t a174;	// L15033
        a174 = v9728;	// L15034
        int8_t v9730 = v9719.read();	// L15035
        int8_t b179;	// L15036
        b179 = v9730;	// L15037
        int8_t v9732 = a174;	// L15038
        int8_t v9733 = b179;	// L15039
        int16_t v9734 = v9732;	// L15040
        int16_t v9735 = v9733;	// L15041
        int16_t v9736 = v9734 * v9735;	// L15042
        int8_t v9737 = c163;	// L15043
        ap_int<17> v9738 = v9737;	// L15044
        ap_int<17> v9739 = v9736;	// L15045
        ap_int<17> v9740 = v9738 + v9739;	// L15046
        int8_t v9741 = v9740;	// L15047
        c163 = v9741;	// L15048
        int8_t v9742 = a174;	// L15049
        v9720.write(v9742);	// L15050
        int8_t v9743 = b179;	// L15051
        v9721.write(v9743);	// L15052
      }
      ap_uint<128> packed_tmp163;	// L15054
      ap_uint<128> v9745 = v9722.read();	// L15055
      packed_tmp163 = v9745;	// L15056
      ap_uint<128> packed_c163;	// L15057
      packed_c163 = 0;	// L15058
      l_S_m_1_m163: for (int m163 = 0; m163 < 16; m163++) {	// L15059
        ap_int<33> v9748 = m163;	// L15060
        bool v9749 = v9748 == 10;	// L15061
        if (v9749) {	// L15062
          int8_t v9750 = c163;	// L15063
          ap_int<128> v9751 = packed_c163;	// L15064
          int64_t v9752 = m163;	// L15065
          int64_t v9753 = v9752 * 8;	// L15066
          ap_int<34> v9754 = m163;	// L15067
          ap_int<34> v9755 = v9754 + 1;	// L15068
          ap_int<66> v9756 = v9755;	// L15069
          ap_int<66> v9757 = v9756 * 8;	// L15070
          ap_int<66> v9758 = v9757 - 1;	// L15071
          int v9759 = v9753;	// L15072
          int v9760 = v9758;	// L15073
          ap_int<128> v9761;
          ap_int<128> v9761_tmp = v9751;
          v9761_tmp(v9760, v9759) = v9750;
          v9761 = v9761_tmp;	// L15074
          packed_c163 = v9761;	// L15075
        } else {
          ap_int<128> v9762 = packed_tmp163;	// L15077
          int64_t v9763 = m163;	// L15078
          int64_t v9764 = v9763 * 8;	// L15079
          ap_int<34> v9765 = m163;	// L15080
          ap_int<34> v9766 = v9765 + 1;	// L15081
          ap_int<66> v9767 = v9766;	// L15082
          ap_int<66> v9768 = v9767 * 8;	// L15083
          ap_int<66> v9769 = v9768 - 1;	// L15084
          int v9770 = v9764;	// L15085
          int v9771 = v9769;	// L15086
          int8_t v9772;
          ap_int<128> v9772_tmp = v9762;
          v9772 = v9772_tmp(v9771, v9770);	// L15087
          ap_int<128> v9773 = packed_c163;	// L15088
          ap_int<128> v9774;
          ap_int<128> v9774_tmp = v9773;
          v9774_tmp(v9771, v9770) = v9772;
          v9774 = v9774_tmp;	// L15098
          packed_c163 = v9774;	// L15099
        }
      }
      ap_int<128> v9775 = packed_c163;	// L15102
      v9723.write(v9775);	// L15103
    }
  }
}

void gemm_11_5(
  hls::stream< int8_t >& v9776,
  hls::stream< int8_t >& v9777,
  hls::stream< int8_t >& v9778,
  hls::stream< int8_t >& v9779,
  hls::stream< ap_uint<128> >& v9780,
  hls::stream< ap_uint<128> >& v9781
) {	// L15108
  l_S_mt_nt_0_mt194: for (int mt194 = 0; mt194 < 4; mt194++) {	// L15116
    l_nt194: for (int nt194 = 0; nt194 < 4; nt194++) {	// L15117
      int8_t c164;	// L15118
      c164 = 0;	// L15119
      l_S_k_0_k194: for (int k194 = 0; k194 < 64; k194++) {	// L15120
        int8_t v9786 = v9776.read();	// L15121
        int8_t a175;	// L15122
        a175 = v9786;	// L15123
        int8_t v9788 = v9777.read();	// L15124
        int8_t b180;	// L15125
        b180 = v9788;	// L15126
        int8_t v9790 = a175;	// L15127
        int8_t v9791 = b180;	// L15128
        int16_t v9792 = v9790;	// L15129
        int16_t v9793 = v9791;	// L15130
        int16_t v9794 = v9792 * v9793;	// L15131
        int8_t v9795 = c164;	// L15132
        ap_int<17> v9796 = v9795;	// L15133
        ap_int<17> v9797 = v9794;	// L15134
        ap_int<17> v9798 = v9796 + v9797;	// L15135
        int8_t v9799 = v9798;	// L15136
        c164 = v9799;	// L15137
        int8_t v9800 = a175;	// L15138
        v9778.write(v9800);	// L15139
        int8_t v9801 = b180;	// L15140
        v9779.write(v9801);	// L15141
      }
      ap_uint<128> packed_tmp164;	// L15143
      ap_uint<128> v9803 = v9780.read();	// L15144
      packed_tmp164 = v9803;	// L15145
      ap_uint<128> packed_c164;	// L15146
      packed_c164 = 0;	// L15147
      l_S_m_1_m164: for (int m164 = 0; m164 < 16; m164++) {	// L15148
        ap_int<33> v9806 = m164;	// L15149
        bool v9807 = v9806 == 10;	// L15150
        if (v9807) {	// L15151
          int8_t v9808 = c164;	// L15152
          ap_int<128> v9809 = packed_c164;	// L15153
          int64_t v9810 = m164;	// L15154
          int64_t v9811 = v9810 * 8;	// L15155
          ap_int<34> v9812 = m164;	// L15156
          ap_int<34> v9813 = v9812 + 1;	// L15157
          ap_int<66> v9814 = v9813;	// L15158
          ap_int<66> v9815 = v9814 * 8;	// L15159
          ap_int<66> v9816 = v9815 - 1;	// L15160
          int v9817 = v9811;	// L15161
          int v9818 = v9816;	// L15162
          ap_int<128> v9819;
          ap_int<128> v9819_tmp = v9809;
          v9819_tmp(v9818, v9817) = v9808;
          v9819 = v9819_tmp;	// L15163
          packed_c164 = v9819;	// L15164
        } else {
          ap_int<128> v9820 = packed_tmp164;	// L15166
          int64_t v9821 = m164;	// L15167
          int64_t v9822 = v9821 * 8;	// L15168
          ap_int<34> v9823 = m164;	// L15169
          ap_int<34> v9824 = v9823 + 1;	// L15170
          ap_int<66> v9825 = v9824;	// L15171
          ap_int<66> v9826 = v9825 * 8;	// L15172
          ap_int<66> v9827 = v9826 - 1;	// L15173
          int v9828 = v9822;	// L15174
          int v9829 = v9827;	// L15175
          int8_t v9830;
          ap_int<128> v9830_tmp = v9820;
          v9830 = v9830_tmp(v9829, v9828);	// L15176
          ap_int<128> v9831 = packed_c164;	// L15177
          ap_int<128> v9832;
          ap_int<128> v9832_tmp = v9831;
          v9832_tmp(v9829, v9828) = v9830;
          v9832 = v9832_tmp;	// L15187
          packed_c164 = v9832;	// L15188
        }
      }
      ap_int<128> v9833 = packed_c164;	// L15191
      v9781.write(v9833);	// L15192
    }
  }
}

void gemm_11_6(
  hls::stream< int8_t >& v9834,
  hls::stream< int8_t >& v9835,
  hls::stream< int8_t >& v9836,
  hls::stream< int8_t >& v9837,
  hls::stream< ap_uint<128> >& v9838,
  hls::stream< ap_uint<128> >& v9839
) {	// L15197
  l_S_mt_nt_0_mt195: for (int mt195 = 0; mt195 < 4; mt195++) {	// L15205
    l_nt195: for (int nt195 = 0; nt195 < 4; nt195++) {	// L15206
      int8_t c165;	// L15207
      c165 = 0;	// L15208
      l_S_k_0_k195: for (int k195 = 0; k195 < 64; k195++) {	// L15209
        int8_t v9844 = v9834.read();	// L15210
        int8_t a176;	// L15211
        a176 = v9844;	// L15212
        int8_t v9846 = v9835.read();	// L15213
        int8_t b181;	// L15214
        b181 = v9846;	// L15215
        int8_t v9848 = a176;	// L15216
        int8_t v9849 = b181;	// L15217
        int16_t v9850 = v9848;	// L15218
        int16_t v9851 = v9849;	// L15219
        int16_t v9852 = v9850 * v9851;	// L15220
        int8_t v9853 = c165;	// L15221
        ap_int<17> v9854 = v9853;	// L15222
        ap_int<17> v9855 = v9852;	// L15223
        ap_int<17> v9856 = v9854 + v9855;	// L15224
        int8_t v9857 = v9856;	// L15225
        c165 = v9857;	// L15226
        int8_t v9858 = a176;	// L15227
        v9836.write(v9858);	// L15228
        int8_t v9859 = b181;	// L15229
        v9837.write(v9859);	// L15230
      }
      ap_uint<128> packed_tmp165;	// L15232
      ap_uint<128> v9861 = v9838.read();	// L15233
      packed_tmp165 = v9861;	// L15234
      ap_uint<128> packed_c165;	// L15235
      packed_c165 = 0;	// L15236
      l_S_m_1_m165: for (int m165 = 0; m165 < 16; m165++) {	// L15237
        ap_int<33> v9864 = m165;	// L15238
        bool v9865 = v9864 == 10;	// L15239
        if (v9865) {	// L15240
          int8_t v9866 = c165;	// L15241
          ap_int<128> v9867 = packed_c165;	// L15242
          int64_t v9868 = m165;	// L15243
          int64_t v9869 = v9868 * 8;	// L15244
          ap_int<34> v9870 = m165;	// L15245
          ap_int<34> v9871 = v9870 + 1;	// L15246
          ap_int<66> v9872 = v9871;	// L15247
          ap_int<66> v9873 = v9872 * 8;	// L15248
          ap_int<66> v9874 = v9873 - 1;	// L15249
          int v9875 = v9869;	// L15250
          int v9876 = v9874;	// L15251
          ap_int<128> v9877;
          ap_int<128> v9877_tmp = v9867;
          v9877_tmp(v9876, v9875) = v9866;
          v9877 = v9877_tmp;	// L15252
          packed_c165 = v9877;	// L15253
        } else {
          ap_int<128> v9878 = packed_tmp165;	// L15255
          int64_t v9879 = m165;	// L15256
          int64_t v9880 = v9879 * 8;	// L15257
          ap_int<34> v9881 = m165;	// L15258
          ap_int<34> v9882 = v9881 + 1;	// L15259
          ap_int<66> v9883 = v9882;	// L15260
          ap_int<66> v9884 = v9883 * 8;	// L15261
          ap_int<66> v9885 = v9884 - 1;	// L15262
          int v9886 = v9880;	// L15263
          int v9887 = v9885;	// L15264
          int8_t v9888;
          ap_int<128> v9888_tmp = v9878;
          v9888 = v9888_tmp(v9887, v9886);	// L15265
          ap_int<128> v9889 = packed_c165;	// L15266
          ap_int<128> v9890;
          ap_int<128> v9890_tmp = v9889;
          v9890_tmp(v9887, v9886) = v9888;
          v9890 = v9890_tmp;	// L15276
          packed_c165 = v9890;	// L15277
        }
      }
      ap_int<128> v9891 = packed_c165;	// L15280
      v9839.write(v9891);	// L15281
    }
  }
}

void gemm_11_7(
  hls::stream< int8_t >& v9892,
  hls::stream< int8_t >& v9893,
  hls::stream< int8_t >& v9894,
  hls::stream< int8_t >& v9895,
  hls::stream< ap_uint<128> >& v9896,
  hls::stream< ap_uint<128> >& v9897
) {	// L15286
  l_S_mt_nt_0_mt196: for (int mt196 = 0; mt196 < 4; mt196++) {	// L15294
    l_nt196: for (int nt196 = 0; nt196 < 4; nt196++) {	// L15295
      int8_t c166;	// L15296
      c166 = 0;	// L15297
      l_S_k_0_k196: for (int k196 = 0; k196 < 64; k196++) {	// L15298
        int8_t v9902 = v9892.read();	// L15299
        int8_t a177;	// L15300
        a177 = v9902;	// L15301
        int8_t v9904 = v9893.read();	// L15302
        int8_t b182;	// L15303
        b182 = v9904;	// L15304
        int8_t v9906 = a177;	// L15305
        int8_t v9907 = b182;	// L15306
        int16_t v9908 = v9906;	// L15307
        int16_t v9909 = v9907;	// L15308
        int16_t v9910 = v9908 * v9909;	// L15309
        int8_t v9911 = c166;	// L15310
        ap_int<17> v9912 = v9911;	// L15311
        ap_int<17> v9913 = v9910;	// L15312
        ap_int<17> v9914 = v9912 + v9913;	// L15313
        int8_t v9915 = v9914;	// L15314
        c166 = v9915;	// L15315
        int8_t v9916 = a177;	// L15316
        v9894.write(v9916);	// L15317
        int8_t v9917 = b182;	// L15318
        v9895.write(v9917);	// L15319
      }
      ap_uint<128> packed_tmp166;	// L15321
      ap_uint<128> v9919 = v9896.read();	// L15322
      packed_tmp166 = v9919;	// L15323
      ap_uint<128> packed_c166;	// L15324
      packed_c166 = 0;	// L15325
      l_S_m_1_m166: for (int m166 = 0; m166 < 16; m166++) {	// L15326
        ap_int<33> v9922 = m166;	// L15327
        bool v9923 = v9922 == 10;	// L15328
        if (v9923) {	// L15329
          int8_t v9924 = c166;	// L15330
          ap_int<128> v9925 = packed_c166;	// L15331
          int64_t v9926 = m166;	// L15332
          int64_t v9927 = v9926 * 8;	// L15333
          ap_int<34> v9928 = m166;	// L15334
          ap_int<34> v9929 = v9928 + 1;	// L15335
          ap_int<66> v9930 = v9929;	// L15336
          ap_int<66> v9931 = v9930 * 8;	// L15337
          ap_int<66> v9932 = v9931 - 1;	// L15338
          int v9933 = v9927;	// L15339
          int v9934 = v9932;	// L15340
          ap_int<128> v9935;
          ap_int<128> v9935_tmp = v9925;
          v9935_tmp(v9934, v9933) = v9924;
          v9935 = v9935_tmp;	// L15341
          packed_c166 = v9935;	// L15342
        } else {
          ap_int<128> v9936 = packed_tmp166;	// L15344
          int64_t v9937 = m166;	// L15345
          int64_t v9938 = v9937 * 8;	// L15346
          ap_int<34> v9939 = m166;	// L15347
          ap_int<34> v9940 = v9939 + 1;	// L15348
          ap_int<66> v9941 = v9940;	// L15349
          ap_int<66> v9942 = v9941 * 8;	// L15350
          ap_int<66> v9943 = v9942 - 1;	// L15351
          int v9944 = v9938;	// L15352
          int v9945 = v9943;	// L15353
          int8_t v9946;
          ap_int<128> v9946_tmp = v9936;
          v9946 = v9946_tmp(v9945, v9944);	// L15354
          ap_int<128> v9947 = packed_c166;	// L15355
          ap_int<128> v9948;
          ap_int<128> v9948_tmp = v9947;
          v9948_tmp(v9945, v9944) = v9946;
          v9948 = v9948_tmp;	// L15365
          packed_c166 = v9948;	// L15366
        }
      }
      ap_int<128> v9949 = packed_c166;	// L15369
      v9897.write(v9949);	// L15370
    }
  }
}

void gemm_11_8(
  hls::stream< int8_t >& v9950,
  hls::stream< int8_t >& v9951,
  hls::stream< int8_t >& v9952,
  hls::stream< int8_t >& v9953,
  hls::stream< ap_uint<128> >& v9954,
  hls::stream< ap_uint<128> >& v9955
) {	// L15375
  l_S_mt_nt_0_mt197: for (int mt197 = 0; mt197 < 4; mt197++) {	// L15383
    l_nt197: for (int nt197 = 0; nt197 < 4; nt197++) {	// L15384
      int8_t c167;	// L15385
      c167 = 0;	// L15386
      l_S_k_0_k197: for (int k197 = 0; k197 < 64; k197++) {	// L15387
        int8_t v9960 = v9950.read();	// L15388
        int8_t a178;	// L15389
        a178 = v9960;	// L15390
        int8_t v9962 = v9951.read();	// L15391
        int8_t b183;	// L15392
        b183 = v9962;	// L15393
        int8_t v9964 = a178;	// L15394
        int8_t v9965 = b183;	// L15395
        int16_t v9966 = v9964;	// L15396
        int16_t v9967 = v9965;	// L15397
        int16_t v9968 = v9966 * v9967;	// L15398
        int8_t v9969 = c167;	// L15399
        ap_int<17> v9970 = v9969;	// L15400
        ap_int<17> v9971 = v9968;	// L15401
        ap_int<17> v9972 = v9970 + v9971;	// L15402
        int8_t v9973 = v9972;	// L15403
        c167 = v9973;	// L15404
        int8_t v9974 = a178;	// L15405
        v9952.write(v9974);	// L15406
        int8_t v9975 = b183;	// L15407
        v9953.write(v9975);	// L15408
      }
      ap_uint<128> packed_tmp167;	// L15410
      ap_uint<128> v9977 = v9954.read();	// L15411
      packed_tmp167 = v9977;	// L15412
      ap_uint<128> packed_c167;	// L15413
      packed_c167 = 0;	// L15414
      l_S_m_1_m167: for (int m167 = 0; m167 < 16; m167++) {	// L15415
        ap_int<33> v9980 = m167;	// L15416
        bool v9981 = v9980 == 10;	// L15417
        if (v9981) {	// L15418
          int8_t v9982 = c167;	// L15419
          ap_int<128> v9983 = packed_c167;	// L15420
          int64_t v9984 = m167;	// L15421
          int64_t v9985 = v9984 * 8;	// L15422
          ap_int<34> v9986 = m167;	// L15423
          ap_int<34> v9987 = v9986 + 1;	// L15424
          ap_int<66> v9988 = v9987;	// L15425
          ap_int<66> v9989 = v9988 * 8;	// L15426
          ap_int<66> v9990 = v9989 - 1;	// L15427
          int v9991 = v9985;	// L15428
          int v9992 = v9990;	// L15429
          ap_int<128> v9993;
          ap_int<128> v9993_tmp = v9983;
          v9993_tmp(v9992, v9991) = v9982;
          v9993 = v9993_tmp;	// L15430
          packed_c167 = v9993;	// L15431
        } else {
          ap_int<128> v9994 = packed_tmp167;	// L15433
          int64_t v9995 = m167;	// L15434
          int64_t v9996 = v9995 * 8;	// L15435
          ap_int<34> v9997 = m167;	// L15436
          ap_int<34> v9998 = v9997 + 1;	// L15437
          ap_int<66> v9999 = v9998;	// L15438
          ap_int<66> v10000 = v9999 * 8;	// L15439
          ap_int<66> v10001 = v10000 - 1;	// L15440
          int v10002 = v9996;	// L15441
          int v10003 = v10001;	// L15442
          int8_t v10004;
          ap_int<128> v10004_tmp = v9994;
          v10004 = v10004_tmp(v10003, v10002);	// L15443
          ap_int<128> v10005 = packed_c167;	// L15444
          ap_int<128> v10006;
          ap_int<128> v10006_tmp = v10005;
          v10006_tmp(v10003, v10002) = v10004;
          v10006 = v10006_tmp;	// L15454
          packed_c167 = v10006;	// L15455
        }
      }
      ap_int<128> v10007 = packed_c167;	// L15458
      v9955.write(v10007);	// L15459
    }
  }
}

void gemm_11_9(
  hls::stream< int8_t >& v10008,
  hls::stream< int8_t >& v10009,
  hls::stream< int8_t >& v10010,
  hls::stream< int8_t >& v10011,
  hls::stream< ap_uint<128> >& v10012,
  hls::stream< ap_uint<128> >& v10013
) {	// L15464
  l_S_mt_nt_0_mt198: for (int mt198 = 0; mt198 < 4; mt198++) {	// L15472
    l_nt198: for (int nt198 = 0; nt198 < 4; nt198++) {	// L15473
      int8_t c168;	// L15474
      c168 = 0;	// L15475
      l_S_k_0_k198: for (int k198 = 0; k198 < 64; k198++) {	// L15476
        int8_t v10018 = v10008.read();	// L15477
        int8_t a179;	// L15478
        a179 = v10018;	// L15479
        int8_t v10020 = v10009.read();	// L15480
        int8_t b184;	// L15481
        b184 = v10020;	// L15482
        int8_t v10022 = a179;	// L15483
        int8_t v10023 = b184;	// L15484
        int16_t v10024 = v10022;	// L15485
        int16_t v10025 = v10023;	// L15486
        int16_t v10026 = v10024 * v10025;	// L15487
        int8_t v10027 = c168;	// L15488
        ap_int<17> v10028 = v10027;	// L15489
        ap_int<17> v10029 = v10026;	// L15490
        ap_int<17> v10030 = v10028 + v10029;	// L15491
        int8_t v10031 = v10030;	// L15492
        c168 = v10031;	// L15493
        int8_t v10032 = a179;	// L15494
        v10010.write(v10032);	// L15495
        int8_t v10033 = b184;	// L15496
        v10011.write(v10033);	// L15497
      }
      ap_uint<128> packed_tmp168;	// L15499
      ap_uint<128> v10035 = v10012.read();	// L15500
      packed_tmp168 = v10035;	// L15501
      ap_uint<128> packed_c168;	// L15502
      packed_c168 = 0;	// L15503
      l_S_m_1_m168: for (int m168 = 0; m168 < 16; m168++) {	// L15504
        ap_int<33> v10038 = m168;	// L15505
        bool v10039 = v10038 == 10;	// L15506
        if (v10039) {	// L15507
          int8_t v10040 = c168;	// L15508
          ap_int<128> v10041 = packed_c168;	// L15509
          int64_t v10042 = m168;	// L15510
          int64_t v10043 = v10042 * 8;	// L15511
          ap_int<34> v10044 = m168;	// L15512
          ap_int<34> v10045 = v10044 + 1;	// L15513
          ap_int<66> v10046 = v10045;	// L15514
          ap_int<66> v10047 = v10046 * 8;	// L15515
          ap_int<66> v10048 = v10047 - 1;	// L15516
          int v10049 = v10043;	// L15517
          int v10050 = v10048;	// L15518
          ap_int<128> v10051;
          ap_int<128> v10051_tmp = v10041;
          v10051_tmp(v10050, v10049) = v10040;
          v10051 = v10051_tmp;	// L15519
          packed_c168 = v10051;	// L15520
        } else {
          ap_int<128> v10052 = packed_tmp168;	// L15522
          int64_t v10053 = m168;	// L15523
          int64_t v10054 = v10053 * 8;	// L15524
          ap_int<34> v10055 = m168;	// L15525
          ap_int<34> v10056 = v10055 + 1;	// L15526
          ap_int<66> v10057 = v10056;	// L15527
          ap_int<66> v10058 = v10057 * 8;	// L15528
          ap_int<66> v10059 = v10058 - 1;	// L15529
          int v10060 = v10054;	// L15530
          int v10061 = v10059;	// L15531
          int8_t v10062;
          ap_int<128> v10062_tmp = v10052;
          v10062 = v10062_tmp(v10061, v10060);	// L15532
          ap_int<128> v10063 = packed_c168;	// L15533
          ap_int<128> v10064;
          ap_int<128> v10064_tmp = v10063;
          v10064_tmp(v10061, v10060) = v10062;
          v10064 = v10064_tmp;	// L15543
          packed_c168 = v10064;	// L15544
        }
      }
      ap_int<128> v10065 = packed_c168;	// L15547
      v10013.write(v10065);	// L15548
    }
  }
}

void gemm_11_10(
  hls::stream< int8_t >& v10066,
  hls::stream< int8_t >& v10067,
  hls::stream< int8_t >& v10068,
  hls::stream< int8_t >& v10069,
  hls::stream< ap_uint<128> >& v10070,
  hls::stream< ap_uint<128> >& v10071
) {	// L15553
  l_S_mt_nt_0_mt199: for (int mt199 = 0; mt199 < 4; mt199++) {	// L15561
    l_nt199: for (int nt199 = 0; nt199 < 4; nt199++) {	// L15562
      int8_t c169;	// L15563
      c169 = 0;	// L15564
      l_S_k_0_k199: for (int k199 = 0; k199 < 64; k199++) {	// L15565
        int8_t v10076 = v10066.read();	// L15566
        int8_t a180;	// L15567
        a180 = v10076;	// L15568
        int8_t v10078 = v10067.read();	// L15569
        int8_t b185;	// L15570
        b185 = v10078;	// L15571
        int8_t v10080 = a180;	// L15572
        int8_t v10081 = b185;	// L15573
        int16_t v10082 = v10080;	// L15574
        int16_t v10083 = v10081;	// L15575
        int16_t v10084 = v10082 * v10083;	// L15576
        int8_t v10085 = c169;	// L15577
        ap_int<17> v10086 = v10085;	// L15578
        ap_int<17> v10087 = v10084;	// L15579
        ap_int<17> v10088 = v10086 + v10087;	// L15580
        int8_t v10089 = v10088;	// L15581
        c169 = v10089;	// L15582
        int8_t v10090 = a180;	// L15583
        v10068.write(v10090);	// L15584
        int8_t v10091 = b185;	// L15585
        v10069.write(v10091);	// L15586
      }
      ap_uint<128> packed_tmp169;	// L15588
      ap_uint<128> v10093 = v10070.read();	// L15589
      packed_tmp169 = v10093;	// L15590
      ap_uint<128> packed_c169;	// L15591
      packed_c169 = 0;	// L15592
      l_S_m_1_m169: for (int m169 = 0; m169 < 16; m169++) {	// L15593
        ap_int<33> v10096 = m169;	// L15594
        bool v10097 = v10096 == 10;	// L15595
        if (v10097) {	// L15596
          int8_t v10098 = c169;	// L15597
          ap_int<128> v10099 = packed_c169;	// L15598
          int64_t v10100 = m169;	// L15599
          int64_t v10101 = v10100 * 8;	// L15600
          ap_int<34> v10102 = m169;	// L15601
          ap_int<34> v10103 = v10102 + 1;	// L15602
          ap_int<66> v10104 = v10103;	// L15603
          ap_int<66> v10105 = v10104 * 8;	// L15604
          ap_int<66> v10106 = v10105 - 1;	// L15605
          int v10107 = v10101;	// L15606
          int v10108 = v10106;	// L15607
          ap_int<128> v10109;
          ap_int<128> v10109_tmp = v10099;
          v10109_tmp(v10108, v10107) = v10098;
          v10109 = v10109_tmp;	// L15608
          packed_c169 = v10109;	// L15609
        } else {
          ap_int<128> v10110 = packed_tmp169;	// L15611
          int64_t v10111 = m169;	// L15612
          int64_t v10112 = v10111 * 8;	// L15613
          ap_int<34> v10113 = m169;	// L15614
          ap_int<34> v10114 = v10113 + 1;	// L15615
          ap_int<66> v10115 = v10114;	// L15616
          ap_int<66> v10116 = v10115 * 8;	// L15617
          ap_int<66> v10117 = v10116 - 1;	// L15618
          int v10118 = v10112;	// L15619
          int v10119 = v10117;	// L15620
          int8_t v10120;
          ap_int<128> v10120_tmp = v10110;
          v10120 = v10120_tmp(v10119, v10118);	// L15621
          ap_int<128> v10121 = packed_c169;	// L15622
          ap_int<128> v10122;
          ap_int<128> v10122_tmp = v10121;
          v10122_tmp(v10119, v10118) = v10120;
          v10122 = v10122_tmp;	// L15632
          packed_c169 = v10122;	// L15633
        }
      }
      ap_int<128> v10123 = packed_c169;	// L15636
      v10071.write(v10123);	// L15637
    }
  }
}

void gemm_11_11(
  hls::stream< int8_t >& v10124,
  hls::stream< int8_t >& v10125,
  hls::stream< int8_t >& v10126,
  hls::stream< int8_t >& v10127,
  hls::stream< ap_uint<128> >& v10128,
  hls::stream< ap_uint<128> >& v10129
) {	// L15642
  l_S_mt_nt_0_mt200: for (int mt200 = 0; mt200 < 4; mt200++) {	// L15650
    l_nt200: for (int nt200 = 0; nt200 < 4; nt200++) {	// L15651
      int8_t c170;	// L15652
      c170 = 0;	// L15653
      l_S_k_0_k200: for (int k200 = 0; k200 < 64; k200++) {	// L15654
        int8_t v10134 = v10124.read();	// L15655
        int8_t a181;	// L15656
        a181 = v10134;	// L15657
        int8_t v10136 = v10125.read();	// L15658
        int8_t b186;	// L15659
        b186 = v10136;	// L15660
        int8_t v10138 = a181;	// L15661
        int8_t v10139 = b186;	// L15662
        int16_t v10140 = v10138;	// L15663
        int16_t v10141 = v10139;	// L15664
        int16_t v10142 = v10140 * v10141;	// L15665
        int8_t v10143 = c170;	// L15666
        ap_int<17> v10144 = v10143;	// L15667
        ap_int<17> v10145 = v10142;	// L15668
        ap_int<17> v10146 = v10144 + v10145;	// L15669
        int8_t v10147 = v10146;	// L15670
        c170 = v10147;	// L15671
        int8_t v10148 = a181;	// L15672
        v10126.write(v10148);	// L15673
        int8_t v10149 = b186;	// L15674
        v10127.write(v10149);	// L15675
      }
      ap_uint<128> packed_tmp170;	// L15677
      ap_uint<128> v10151 = v10128.read();	// L15678
      packed_tmp170 = v10151;	// L15679
      ap_uint<128> packed_c170;	// L15680
      packed_c170 = 0;	// L15681
      l_S_m_1_m170: for (int m170 = 0; m170 < 16; m170++) {	// L15682
        ap_int<33> v10154 = m170;	// L15683
        bool v10155 = v10154 == 10;	// L15684
        if (v10155) {	// L15685
          int8_t v10156 = c170;	// L15686
          ap_int<128> v10157 = packed_c170;	// L15687
          int64_t v10158 = m170;	// L15688
          int64_t v10159 = v10158 * 8;	// L15689
          ap_int<34> v10160 = m170;	// L15690
          ap_int<34> v10161 = v10160 + 1;	// L15691
          ap_int<66> v10162 = v10161;	// L15692
          ap_int<66> v10163 = v10162 * 8;	// L15693
          ap_int<66> v10164 = v10163 - 1;	// L15694
          int v10165 = v10159;	// L15695
          int v10166 = v10164;	// L15696
          ap_int<128> v10167;
          ap_int<128> v10167_tmp = v10157;
          v10167_tmp(v10166, v10165) = v10156;
          v10167 = v10167_tmp;	// L15697
          packed_c170 = v10167;	// L15698
        } else {
          ap_int<128> v10168 = packed_tmp170;	// L15700
          int64_t v10169 = m170;	// L15701
          int64_t v10170 = v10169 * 8;	// L15702
          ap_int<34> v10171 = m170;	// L15703
          ap_int<34> v10172 = v10171 + 1;	// L15704
          ap_int<66> v10173 = v10172;	// L15705
          ap_int<66> v10174 = v10173 * 8;	// L15706
          ap_int<66> v10175 = v10174 - 1;	// L15707
          int v10176 = v10170;	// L15708
          int v10177 = v10175;	// L15709
          int8_t v10178;
          ap_int<128> v10178_tmp = v10168;
          v10178 = v10178_tmp(v10177, v10176);	// L15710
          ap_int<128> v10179 = packed_c170;	// L15711
          ap_int<128> v10180;
          ap_int<128> v10180_tmp = v10179;
          v10180_tmp(v10177, v10176) = v10178;
          v10180 = v10180_tmp;	// L15721
          packed_c170 = v10180;	// L15722
        }
      }
      ap_int<128> v10181 = packed_c170;	// L15725
      v10129.write(v10181);	// L15726
    }
  }
}

void gemm_11_12(
  hls::stream< int8_t >& v10182,
  hls::stream< int8_t >& v10183,
  hls::stream< int8_t >& v10184,
  hls::stream< int8_t >& v10185,
  hls::stream< ap_uint<128> >& v10186,
  hls::stream< ap_uint<128> >& v10187
) {	// L15731
  l_S_mt_nt_0_mt201: for (int mt201 = 0; mt201 < 4; mt201++) {	// L15739
    l_nt201: for (int nt201 = 0; nt201 < 4; nt201++) {	// L15740
      int8_t c171;	// L15741
      c171 = 0;	// L15742
      l_S_k_0_k201: for (int k201 = 0; k201 < 64; k201++) {	// L15743
        int8_t v10192 = v10182.read();	// L15744
        int8_t a182;	// L15745
        a182 = v10192;	// L15746
        int8_t v10194 = v10183.read();	// L15747
        int8_t b187;	// L15748
        b187 = v10194;	// L15749
        int8_t v10196 = a182;	// L15750
        int8_t v10197 = b187;	// L15751
        int16_t v10198 = v10196;	// L15752
        int16_t v10199 = v10197;	// L15753
        int16_t v10200 = v10198 * v10199;	// L15754
        int8_t v10201 = c171;	// L15755
        ap_int<17> v10202 = v10201;	// L15756
        ap_int<17> v10203 = v10200;	// L15757
        ap_int<17> v10204 = v10202 + v10203;	// L15758
        int8_t v10205 = v10204;	// L15759
        c171 = v10205;	// L15760
        int8_t v10206 = a182;	// L15761
        v10184.write(v10206);	// L15762
        int8_t v10207 = b187;	// L15763
        v10185.write(v10207);	// L15764
      }
      ap_uint<128> packed_tmp171;	// L15766
      ap_uint<128> v10209 = v10186.read();	// L15767
      packed_tmp171 = v10209;	// L15768
      ap_uint<128> packed_c171;	// L15769
      packed_c171 = 0;	// L15770
      l_S_m_1_m171: for (int m171 = 0; m171 < 16; m171++) {	// L15771
        ap_int<33> v10212 = m171;	// L15772
        bool v10213 = v10212 == 10;	// L15773
        if (v10213) {	// L15774
          int8_t v10214 = c171;	// L15775
          ap_int<128> v10215 = packed_c171;	// L15776
          int64_t v10216 = m171;	// L15777
          int64_t v10217 = v10216 * 8;	// L15778
          ap_int<34> v10218 = m171;	// L15779
          ap_int<34> v10219 = v10218 + 1;	// L15780
          ap_int<66> v10220 = v10219;	// L15781
          ap_int<66> v10221 = v10220 * 8;	// L15782
          ap_int<66> v10222 = v10221 - 1;	// L15783
          int v10223 = v10217;	// L15784
          int v10224 = v10222;	// L15785
          ap_int<128> v10225;
          ap_int<128> v10225_tmp = v10215;
          v10225_tmp(v10224, v10223) = v10214;
          v10225 = v10225_tmp;	// L15786
          packed_c171 = v10225;	// L15787
        } else {
          ap_int<128> v10226 = packed_tmp171;	// L15789
          int64_t v10227 = m171;	// L15790
          int64_t v10228 = v10227 * 8;	// L15791
          ap_int<34> v10229 = m171;	// L15792
          ap_int<34> v10230 = v10229 + 1;	// L15793
          ap_int<66> v10231 = v10230;	// L15794
          ap_int<66> v10232 = v10231 * 8;	// L15795
          ap_int<66> v10233 = v10232 - 1;	// L15796
          int v10234 = v10228;	// L15797
          int v10235 = v10233;	// L15798
          int8_t v10236;
          ap_int<128> v10236_tmp = v10226;
          v10236 = v10236_tmp(v10235, v10234);	// L15799
          ap_int<128> v10237 = packed_c171;	// L15800
          ap_int<128> v10238;
          ap_int<128> v10238_tmp = v10237;
          v10238_tmp(v10235, v10234) = v10236;
          v10238 = v10238_tmp;	// L15810
          packed_c171 = v10238;	// L15811
        }
      }
      ap_int<128> v10239 = packed_c171;	// L15814
      v10187.write(v10239);	// L15815
    }
  }
}

void gemm_11_13(
  hls::stream< int8_t >& v10240,
  hls::stream< int8_t >& v10241,
  hls::stream< int8_t >& v10242,
  hls::stream< int8_t >& v10243,
  hls::stream< ap_uint<128> >& v10244,
  hls::stream< ap_uint<128> >& v10245
) {	// L15820
  l_S_mt_nt_0_mt202: for (int mt202 = 0; mt202 < 4; mt202++) {	// L15828
    l_nt202: for (int nt202 = 0; nt202 < 4; nt202++) {	// L15829
      int8_t c172;	// L15830
      c172 = 0;	// L15831
      l_S_k_0_k202: for (int k202 = 0; k202 < 64; k202++) {	// L15832
        int8_t v10250 = v10240.read();	// L15833
        int8_t a183;	// L15834
        a183 = v10250;	// L15835
        int8_t v10252 = v10241.read();	// L15836
        int8_t b188;	// L15837
        b188 = v10252;	// L15838
        int8_t v10254 = a183;	// L15839
        int8_t v10255 = b188;	// L15840
        int16_t v10256 = v10254;	// L15841
        int16_t v10257 = v10255;	// L15842
        int16_t v10258 = v10256 * v10257;	// L15843
        int8_t v10259 = c172;	// L15844
        ap_int<17> v10260 = v10259;	// L15845
        ap_int<17> v10261 = v10258;	// L15846
        ap_int<17> v10262 = v10260 + v10261;	// L15847
        int8_t v10263 = v10262;	// L15848
        c172 = v10263;	// L15849
        int8_t v10264 = a183;	// L15850
        v10242.write(v10264);	// L15851
        int8_t v10265 = b188;	// L15852
        v10243.write(v10265);	// L15853
      }
      ap_uint<128> packed_tmp172;	// L15855
      ap_uint<128> v10267 = v10244.read();	// L15856
      packed_tmp172 = v10267;	// L15857
      ap_uint<128> packed_c172;	// L15858
      packed_c172 = 0;	// L15859
      l_S_m_1_m172: for (int m172 = 0; m172 < 16; m172++) {	// L15860
        ap_int<33> v10270 = m172;	// L15861
        bool v10271 = v10270 == 10;	// L15862
        if (v10271) {	// L15863
          int8_t v10272 = c172;	// L15864
          ap_int<128> v10273 = packed_c172;	// L15865
          int64_t v10274 = m172;	// L15866
          int64_t v10275 = v10274 * 8;	// L15867
          ap_int<34> v10276 = m172;	// L15868
          ap_int<34> v10277 = v10276 + 1;	// L15869
          ap_int<66> v10278 = v10277;	// L15870
          ap_int<66> v10279 = v10278 * 8;	// L15871
          ap_int<66> v10280 = v10279 - 1;	// L15872
          int v10281 = v10275;	// L15873
          int v10282 = v10280;	// L15874
          ap_int<128> v10283;
          ap_int<128> v10283_tmp = v10273;
          v10283_tmp(v10282, v10281) = v10272;
          v10283 = v10283_tmp;	// L15875
          packed_c172 = v10283;	// L15876
        } else {
          ap_int<128> v10284 = packed_tmp172;	// L15878
          int64_t v10285 = m172;	// L15879
          int64_t v10286 = v10285 * 8;	// L15880
          ap_int<34> v10287 = m172;	// L15881
          ap_int<34> v10288 = v10287 + 1;	// L15882
          ap_int<66> v10289 = v10288;	// L15883
          ap_int<66> v10290 = v10289 * 8;	// L15884
          ap_int<66> v10291 = v10290 - 1;	// L15885
          int v10292 = v10286;	// L15886
          int v10293 = v10291;	// L15887
          int8_t v10294;
          ap_int<128> v10294_tmp = v10284;
          v10294 = v10294_tmp(v10293, v10292);	// L15888
          ap_int<128> v10295 = packed_c172;	// L15889
          ap_int<128> v10296;
          ap_int<128> v10296_tmp = v10295;
          v10296_tmp(v10293, v10292) = v10294;
          v10296 = v10296_tmp;	// L15899
          packed_c172 = v10296;	// L15900
        }
      }
      ap_int<128> v10297 = packed_c172;	// L15903
      v10245.write(v10297);	// L15904
    }
  }
}

void gemm_11_14(
  hls::stream< int8_t >& v10298,
  hls::stream< int8_t >& v10299,
  hls::stream< int8_t >& v10300,
  hls::stream< int8_t >& v10301,
  hls::stream< ap_uint<128> >& v10302,
  hls::stream< ap_uint<128> >& v10303
) {	// L15909
  l_S_mt_nt_0_mt203: for (int mt203 = 0; mt203 < 4; mt203++) {	// L15917
    l_nt203: for (int nt203 = 0; nt203 < 4; nt203++) {	// L15918
      int8_t c173;	// L15919
      c173 = 0;	// L15920
      l_S_k_0_k203: for (int k203 = 0; k203 < 64; k203++) {	// L15921
        int8_t v10308 = v10298.read();	// L15922
        int8_t a184;	// L15923
        a184 = v10308;	// L15924
        int8_t v10310 = v10299.read();	// L15925
        int8_t b189;	// L15926
        b189 = v10310;	// L15927
        int8_t v10312 = a184;	// L15928
        int8_t v10313 = b189;	// L15929
        int16_t v10314 = v10312;	// L15930
        int16_t v10315 = v10313;	// L15931
        int16_t v10316 = v10314 * v10315;	// L15932
        int8_t v10317 = c173;	// L15933
        ap_int<17> v10318 = v10317;	// L15934
        ap_int<17> v10319 = v10316;	// L15935
        ap_int<17> v10320 = v10318 + v10319;	// L15936
        int8_t v10321 = v10320;	// L15937
        c173 = v10321;	// L15938
        int8_t v10322 = a184;	// L15939
        v10300.write(v10322);	// L15940
        int8_t v10323 = b189;	// L15941
        v10301.write(v10323);	// L15942
      }
      ap_uint<128> packed_tmp173;	// L15944
      ap_uint<128> v10325 = v10302.read();	// L15945
      packed_tmp173 = v10325;	// L15946
      ap_uint<128> packed_c173;	// L15947
      packed_c173 = 0;	// L15948
      l_S_m_1_m173: for (int m173 = 0; m173 < 16; m173++) {	// L15949
        ap_int<33> v10328 = m173;	// L15950
        bool v10329 = v10328 == 10;	// L15951
        if (v10329) {	// L15952
          int8_t v10330 = c173;	// L15953
          ap_int<128> v10331 = packed_c173;	// L15954
          int64_t v10332 = m173;	// L15955
          int64_t v10333 = v10332 * 8;	// L15956
          ap_int<34> v10334 = m173;	// L15957
          ap_int<34> v10335 = v10334 + 1;	// L15958
          ap_int<66> v10336 = v10335;	// L15959
          ap_int<66> v10337 = v10336 * 8;	// L15960
          ap_int<66> v10338 = v10337 - 1;	// L15961
          int v10339 = v10333;	// L15962
          int v10340 = v10338;	// L15963
          ap_int<128> v10341;
          ap_int<128> v10341_tmp = v10331;
          v10341_tmp(v10340, v10339) = v10330;
          v10341 = v10341_tmp;	// L15964
          packed_c173 = v10341;	// L15965
        } else {
          ap_int<128> v10342 = packed_tmp173;	// L15967
          int64_t v10343 = m173;	// L15968
          int64_t v10344 = v10343 * 8;	// L15969
          ap_int<34> v10345 = m173;	// L15970
          ap_int<34> v10346 = v10345 + 1;	// L15971
          ap_int<66> v10347 = v10346;	// L15972
          ap_int<66> v10348 = v10347 * 8;	// L15973
          ap_int<66> v10349 = v10348 - 1;	// L15974
          int v10350 = v10344;	// L15975
          int v10351 = v10349;	// L15976
          int8_t v10352;
          ap_int<128> v10352_tmp = v10342;
          v10352 = v10352_tmp(v10351, v10350);	// L15977
          ap_int<128> v10353 = packed_c173;	// L15978
          ap_int<128> v10354;
          ap_int<128> v10354_tmp = v10353;
          v10354_tmp(v10351, v10350) = v10352;
          v10354 = v10354_tmp;	// L15988
          packed_c173 = v10354;	// L15989
        }
      }
      ap_int<128> v10355 = packed_c173;	// L15992
      v10303.write(v10355);	// L15993
    }
  }
}

void gemm_11_15(
  hls::stream< int8_t >& v10356,
  hls::stream< int8_t >& v10357,
  hls::stream< int8_t >& v10358,
  hls::stream< int8_t >& v10359,
  hls::stream< ap_uint<128> >& v10360,
  hls::stream< ap_uint<128> >& v10361
) {	// L15998
  l_S_mt_nt_0_mt204: for (int mt204 = 0; mt204 < 4; mt204++) {	// L16006
    l_nt204: for (int nt204 = 0; nt204 < 4; nt204++) {	// L16007
      int8_t c174;	// L16008
      c174 = 0;	// L16009
      l_S_k_0_k204: for (int k204 = 0; k204 < 64; k204++) {	// L16010
        int8_t v10366 = v10356.read();	// L16011
        int8_t a185;	// L16012
        a185 = v10366;	// L16013
        int8_t v10368 = v10357.read();	// L16014
        int8_t b190;	// L16015
        b190 = v10368;	// L16016
        int8_t v10370 = a185;	// L16017
        int8_t v10371 = b190;	// L16018
        int16_t v10372 = v10370;	// L16019
        int16_t v10373 = v10371;	// L16020
        int16_t v10374 = v10372 * v10373;	// L16021
        int8_t v10375 = c174;	// L16022
        ap_int<17> v10376 = v10375;	// L16023
        ap_int<17> v10377 = v10374;	// L16024
        ap_int<17> v10378 = v10376 + v10377;	// L16025
        int8_t v10379 = v10378;	// L16026
        c174 = v10379;	// L16027
        int8_t v10380 = a185;	// L16028
        v10358.write(v10380);	// L16029
        int8_t v10381 = b190;	// L16030
        v10359.write(v10381);	// L16031
      }
      ap_uint<128> packed_tmp174;	// L16033
      ap_uint<128> v10383 = v10360.read();	// L16034
      packed_tmp174 = v10383;	// L16035
      ap_uint<128> packed_c174;	// L16036
      packed_c174 = 0;	// L16037
      l_S_m_1_m174: for (int m174 = 0; m174 < 16; m174++) {	// L16038
        ap_int<33> v10386 = m174;	// L16039
        bool v10387 = v10386 == 10;	// L16040
        if (v10387) {	// L16041
          int8_t v10388 = c174;	// L16042
          ap_int<128> v10389 = packed_c174;	// L16043
          int64_t v10390 = m174;	// L16044
          int64_t v10391 = v10390 * 8;	// L16045
          ap_int<34> v10392 = m174;	// L16046
          ap_int<34> v10393 = v10392 + 1;	// L16047
          ap_int<66> v10394 = v10393;	// L16048
          ap_int<66> v10395 = v10394 * 8;	// L16049
          ap_int<66> v10396 = v10395 - 1;	// L16050
          int v10397 = v10391;	// L16051
          int v10398 = v10396;	// L16052
          ap_int<128> v10399;
          ap_int<128> v10399_tmp = v10389;
          v10399_tmp(v10398, v10397) = v10388;
          v10399 = v10399_tmp;	// L16053
          packed_c174 = v10399;	// L16054
        } else {
          ap_int<128> v10400 = packed_tmp174;	// L16056
          int64_t v10401 = m174;	// L16057
          int64_t v10402 = v10401 * 8;	// L16058
          ap_int<34> v10403 = m174;	// L16059
          ap_int<34> v10404 = v10403 + 1;	// L16060
          ap_int<66> v10405 = v10404;	// L16061
          ap_int<66> v10406 = v10405 * 8;	// L16062
          ap_int<66> v10407 = v10406 - 1;	// L16063
          int v10408 = v10402;	// L16064
          int v10409 = v10407;	// L16065
          int8_t v10410;
          ap_int<128> v10410_tmp = v10400;
          v10410 = v10410_tmp(v10409, v10408);	// L16066
          ap_int<128> v10411 = packed_c174;	// L16067
          ap_int<128> v10412;
          ap_int<128> v10412_tmp = v10411;
          v10412_tmp(v10409, v10408) = v10410;
          v10412 = v10412_tmp;	// L16077
          packed_c174 = v10412;	// L16078
        }
      }
      ap_int<128> v10413 = packed_c174;	// L16081
      v10361.write(v10413);	// L16082
    }
  }
}

void gemm_11_16(
  hls::stream< int8_t >& v10414,
  hls::stream< int8_t >& v10415,
  hls::stream< int8_t >& v10416,
  hls::stream< ap_uint<128> >& v10417,
  hls::stream< ap_uint<128> >& v10418
) {	// L16087
  l_S_mt_nt_0_mt205: for (int mt205 = 0; mt205 < 4; mt205++) {	// L16095
    l_nt205: for (int nt205 = 0; nt205 < 4; nt205++) {	// L16096
      int8_t c175;	// L16097
      c175 = 0;	// L16098
      l_S_k_0_k205: for (int k205 = 0; k205 < 64; k205++) {	// L16099
        int8_t v10423 = v10414.read();	// L16100
        int8_t a186;	// L16101
        a186 = v10423;	// L16102
        int8_t v10425 = v10415.read();	// L16103
        int8_t b191;	// L16104
        b191 = v10425;	// L16105
        int8_t v10427 = a186;	// L16106
        int8_t v10428 = b191;	// L16107
        int16_t v10429 = v10427;	// L16108
        int16_t v10430 = v10428;	// L16109
        int16_t v10431 = v10429 * v10430;	// L16110
        int8_t v10432 = c175;	// L16111
        ap_int<17> v10433 = v10432;	// L16112
        ap_int<17> v10434 = v10431;	// L16113
        ap_int<17> v10435 = v10433 + v10434;	// L16114
        int8_t v10436 = v10435;	// L16115
        c175 = v10436;	// L16116
        int8_t v10437 = b191;	// L16117
        v10416.write(v10437);	// L16118
      }
      ap_uint<128> packed_tmp175;	// L16120
      ap_uint<128> v10439 = v10417.read();	// L16121
      packed_tmp175 = v10439;	// L16122
      ap_uint<128> packed_c175;	// L16123
      packed_c175 = 0;	// L16124
      l_S_m_1_m175: for (int m175 = 0; m175 < 16; m175++) {	// L16125
        ap_int<33> v10442 = m175;	// L16126
        bool v10443 = v10442 == 10;	// L16127
        if (v10443) {	// L16128
          int8_t v10444 = c175;	// L16129
          ap_int<128> v10445 = packed_c175;	// L16130
          int64_t v10446 = m175;	// L16131
          int64_t v10447 = v10446 * 8;	// L16132
          ap_int<34> v10448 = m175;	// L16133
          ap_int<34> v10449 = v10448 + 1;	// L16134
          ap_int<66> v10450 = v10449;	// L16135
          ap_int<66> v10451 = v10450 * 8;	// L16136
          ap_int<66> v10452 = v10451 - 1;	// L16137
          int v10453 = v10447;	// L16138
          int v10454 = v10452;	// L16139
          ap_int<128> v10455;
          ap_int<128> v10455_tmp = v10445;
          v10455_tmp(v10454, v10453) = v10444;
          v10455 = v10455_tmp;	// L16140
          packed_c175 = v10455;	// L16141
        } else {
          ap_int<128> v10456 = packed_tmp175;	// L16143
          int64_t v10457 = m175;	// L16144
          int64_t v10458 = v10457 * 8;	// L16145
          ap_int<34> v10459 = m175;	// L16146
          ap_int<34> v10460 = v10459 + 1;	// L16147
          ap_int<66> v10461 = v10460;	// L16148
          ap_int<66> v10462 = v10461 * 8;	// L16149
          ap_int<66> v10463 = v10462 - 1;	// L16150
          int v10464 = v10458;	// L16151
          int v10465 = v10463;	// L16152
          int8_t v10466;
          ap_int<128> v10466_tmp = v10456;
          v10466 = v10466_tmp(v10465, v10464);	// L16153
          ap_int<128> v10467 = packed_c175;	// L16154
          ap_int<128> v10468;
          ap_int<128> v10468_tmp = v10467;
          v10468_tmp(v10465, v10464) = v10466;
          v10468 = v10468_tmp;	// L16164
          packed_c175 = v10468;	// L16165
        }
      }
      ap_int<128> v10469 = packed_c175;	// L16168
      v10418.write(v10469);	// L16169
    }
  }
}

void gemm_12_0(
  hls::stream< ap_uint<128> >& v10470,
  hls::stream< int8_t >& v10471,
  hls::stream< ap_uint<128> >& v10472
) {	// L16174
  l_S_mt_nt_0_mt206: for (int mt206 = 0; mt206 < 4; mt206++) {	// L16177
    l_nt206: for (int nt206 = 0; nt206 < 4; nt206++) {	// L16178
      l_S_k_0_k206: for (int k206 = 0; k206 < 64; k206++) {	// L16179
        ap_uint<128> v10476 = v10470.read();	// L16180
        ap_uint<128> a187;	// L16181
        a187 = v10476;	// L16182
        ap_int<128> v10478 = a187;	// L16183
        int8_t v10479;
        ap_int<128> v10479_tmp = v10478;
        v10479 = v10479_tmp(95, 88);	// L16184
        v10471.write(v10479);	// L16185
        ap_int<128> v10480 = a187;	// L16186
        v10472.write(v10480);	// L16187
      }
    }
  }
}

void gemm_12_1(
  hls::stream< int8_t >& v10481,
  hls::stream< int8_t >& v10482,
  hls::stream< int8_t >& v10483,
  hls::stream< int8_t >& v10484,
  hls::stream< ap_uint<128> >& v10485,
  hls::stream< ap_uint<128> >& v10486
) {	// L16193
  l_S_mt_nt_0_mt207: for (int mt207 = 0; mt207 < 4; mt207++) {	// L16201
    l_nt207: for (int nt207 = 0; nt207 < 4; nt207++) {	// L16202
      int8_t c176;	// L16203
      c176 = 0;	// L16204
      l_S_k_0_k207: for (int k207 = 0; k207 < 64; k207++) {	// L16205
        int8_t v10491 = v10481.read();	// L16206
        int8_t a188;	// L16207
        a188 = v10491;	// L16208
        int8_t v10493 = v10482.read();	// L16209
        int8_t b192;	// L16210
        b192 = v10493;	// L16211
        int8_t v10495 = a188;	// L16212
        int8_t v10496 = b192;	// L16213
        int16_t v10497 = v10495;	// L16214
        int16_t v10498 = v10496;	// L16215
        int16_t v10499 = v10497 * v10498;	// L16216
        int8_t v10500 = c176;	// L16217
        ap_int<17> v10501 = v10500;	// L16218
        ap_int<17> v10502 = v10499;	// L16219
        ap_int<17> v10503 = v10501 + v10502;	// L16220
        int8_t v10504 = v10503;	// L16221
        c176 = v10504;	// L16222
        int8_t v10505 = a188;	// L16223
        v10483.write(v10505);	// L16224
        int8_t v10506 = b192;	// L16225
        v10484.write(v10506);	// L16226
      }
      ap_uint<128> packed_tmp176;	// L16228
      ap_uint<128> v10508 = v10485.read();	// L16229
      packed_tmp176 = v10508;	// L16230
      ap_uint<128> packed_c176;	// L16231
      packed_c176 = 0;	// L16232
      l_S_m_1_m176: for (int m176 = 0; m176 < 16; m176++) {	// L16233
        ap_int<33> v10511 = m176;	// L16234
        bool v10512 = v10511 == 11;	// L16235
        if (v10512) {	// L16236
          int8_t v10513 = c176;	// L16237
          ap_int<128> v10514 = packed_c176;	// L16238
          int64_t v10515 = m176;	// L16239
          int64_t v10516 = v10515 * 8;	// L16240
          ap_int<34> v10517 = m176;	// L16241
          ap_int<34> v10518 = v10517 + 1;	// L16242
          ap_int<66> v10519 = v10518;	// L16243
          ap_int<66> v10520 = v10519 * 8;	// L16244
          ap_int<66> v10521 = v10520 - 1;	// L16245
          int v10522 = v10516;	// L16246
          int v10523 = v10521;	// L16247
          ap_int<128> v10524;
          ap_int<128> v10524_tmp = v10514;
          v10524_tmp(v10523, v10522) = v10513;
          v10524 = v10524_tmp;	// L16248
          packed_c176 = v10524;	// L16249
        } else {
          ap_int<128> v10525 = packed_tmp176;	// L16251
          int64_t v10526 = m176;	// L16252
          int64_t v10527 = v10526 * 8;	// L16253
          ap_int<34> v10528 = m176;	// L16254
          ap_int<34> v10529 = v10528 + 1;	// L16255
          ap_int<66> v10530 = v10529;	// L16256
          ap_int<66> v10531 = v10530 * 8;	// L16257
          ap_int<66> v10532 = v10531 - 1;	// L16258
          int v10533 = v10527;	// L16259
          int v10534 = v10532;	// L16260
          int8_t v10535;
          ap_int<128> v10535_tmp = v10525;
          v10535 = v10535_tmp(v10534, v10533);	// L16261
          ap_int<128> v10536 = packed_c176;	// L16262
          ap_int<128> v10537;
          ap_int<128> v10537_tmp = v10536;
          v10537_tmp(v10534, v10533) = v10535;
          v10537 = v10537_tmp;	// L16272
          packed_c176 = v10537;	// L16273
        }
      }
      ap_int<128> v10538 = packed_c176;	// L16276
      v10486.write(v10538);	// L16277
    }
  }
}

void gemm_12_2(
  hls::stream< int8_t >& v10539,
  hls::stream< int8_t >& v10540,
  hls::stream< int8_t >& v10541,
  hls::stream< int8_t >& v10542,
  hls::stream< ap_uint<128> >& v10543,
  hls::stream< ap_uint<128> >& v10544
) {	// L16282
  l_S_mt_nt_0_mt208: for (int mt208 = 0; mt208 < 4; mt208++) {	// L16290
    l_nt208: for (int nt208 = 0; nt208 < 4; nt208++) {	// L16291
      int8_t c177;	// L16292
      c177 = 0;	// L16293
      l_S_k_0_k208: for (int k208 = 0; k208 < 64; k208++) {	// L16294
        int8_t v10549 = v10539.read();	// L16295
        int8_t a189;	// L16296
        a189 = v10549;	// L16297
        int8_t v10551 = v10540.read();	// L16298
        int8_t b193;	// L16299
        b193 = v10551;	// L16300
        int8_t v10553 = a189;	// L16301
        int8_t v10554 = b193;	// L16302
        int16_t v10555 = v10553;	// L16303
        int16_t v10556 = v10554;	// L16304
        int16_t v10557 = v10555 * v10556;	// L16305
        int8_t v10558 = c177;	// L16306
        ap_int<17> v10559 = v10558;	// L16307
        ap_int<17> v10560 = v10557;	// L16308
        ap_int<17> v10561 = v10559 + v10560;	// L16309
        int8_t v10562 = v10561;	// L16310
        c177 = v10562;	// L16311
        int8_t v10563 = a189;	// L16312
        v10541.write(v10563);	// L16313
        int8_t v10564 = b193;	// L16314
        v10542.write(v10564);	// L16315
      }
      ap_uint<128> packed_tmp177;	// L16317
      ap_uint<128> v10566 = v10543.read();	// L16318
      packed_tmp177 = v10566;	// L16319
      ap_uint<128> packed_c177;	// L16320
      packed_c177 = 0;	// L16321
      l_S_m_1_m177: for (int m177 = 0; m177 < 16; m177++) {	// L16322
        ap_int<33> v10569 = m177;	// L16323
        bool v10570 = v10569 == 11;	// L16324
        if (v10570) {	// L16325
          int8_t v10571 = c177;	// L16326
          ap_int<128> v10572 = packed_c177;	// L16327
          int64_t v10573 = m177;	// L16328
          int64_t v10574 = v10573 * 8;	// L16329
          ap_int<34> v10575 = m177;	// L16330
          ap_int<34> v10576 = v10575 + 1;	// L16331
          ap_int<66> v10577 = v10576;	// L16332
          ap_int<66> v10578 = v10577 * 8;	// L16333
          ap_int<66> v10579 = v10578 - 1;	// L16334
          int v10580 = v10574;	// L16335
          int v10581 = v10579;	// L16336
          ap_int<128> v10582;
          ap_int<128> v10582_tmp = v10572;
          v10582_tmp(v10581, v10580) = v10571;
          v10582 = v10582_tmp;	// L16337
          packed_c177 = v10582;	// L16338
        } else {
          ap_int<128> v10583 = packed_tmp177;	// L16340
          int64_t v10584 = m177;	// L16341
          int64_t v10585 = v10584 * 8;	// L16342
          ap_int<34> v10586 = m177;	// L16343
          ap_int<34> v10587 = v10586 + 1;	// L16344
          ap_int<66> v10588 = v10587;	// L16345
          ap_int<66> v10589 = v10588 * 8;	// L16346
          ap_int<66> v10590 = v10589 - 1;	// L16347
          int v10591 = v10585;	// L16348
          int v10592 = v10590;	// L16349
          int8_t v10593;
          ap_int<128> v10593_tmp = v10583;
          v10593 = v10593_tmp(v10592, v10591);	// L16350
          ap_int<128> v10594 = packed_c177;	// L16351
          ap_int<128> v10595;
          ap_int<128> v10595_tmp = v10594;
          v10595_tmp(v10592, v10591) = v10593;
          v10595 = v10595_tmp;	// L16361
          packed_c177 = v10595;	// L16362
        }
      }
      ap_int<128> v10596 = packed_c177;	// L16365
      v10544.write(v10596);	// L16366
    }
  }
}

void gemm_12_3(
  hls::stream< int8_t >& v10597,
  hls::stream< int8_t >& v10598,
  hls::stream< int8_t >& v10599,
  hls::stream< int8_t >& v10600,
  hls::stream< ap_uint<128> >& v10601,
  hls::stream< ap_uint<128> >& v10602
) {	// L16371
  l_S_mt_nt_0_mt209: for (int mt209 = 0; mt209 < 4; mt209++) {	// L16379
    l_nt209: for (int nt209 = 0; nt209 < 4; nt209++) {	// L16380
      int8_t c178;	// L16381
      c178 = 0;	// L16382
      l_S_k_0_k209: for (int k209 = 0; k209 < 64; k209++) {	// L16383
        int8_t v10607 = v10597.read();	// L16384
        int8_t a190;	// L16385
        a190 = v10607;	// L16386
        int8_t v10609 = v10598.read();	// L16387
        int8_t b194;	// L16388
        b194 = v10609;	// L16389
        int8_t v10611 = a190;	// L16390
        int8_t v10612 = b194;	// L16391
        int16_t v10613 = v10611;	// L16392
        int16_t v10614 = v10612;	// L16393
        int16_t v10615 = v10613 * v10614;	// L16394
        int8_t v10616 = c178;	// L16395
        ap_int<17> v10617 = v10616;	// L16396
        ap_int<17> v10618 = v10615;	// L16397
        ap_int<17> v10619 = v10617 + v10618;	// L16398
        int8_t v10620 = v10619;	// L16399
        c178 = v10620;	// L16400
        int8_t v10621 = a190;	// L16401
        v10599.write(v10621);	// L16402
        int8_t v10622 = b194;	// L16403
        v10600.write(v10622);	// L16404
      }
      ap_uint<128> packed_tmp178;	// L16406
      ap_uint<128> v10624 = v10601.read();	// L16407
      packed_tmp178 = v10624;	// L16408
      ap_uint<128> packed_c178;	// L16409
      packed_c178 = 0;	// L16410
      l_S_m_1_m178: for (int m178 = 0; m178 < 16; m178++) {	// L16411
        ap_int<33> v10627 = m178;	// L16412
        bool v10628 = v10627 == 11;	// L16413
        if (v10628) {	// L16414
          int8_t v10629 = c178;	// L16415
          ap_int<128> v10630 = packed_c178;	// L16416
          int64_t v10631 = m178;	// L16417
          int64_t v10632 = v10631 * 8;	// L16418
          ap_int<34> v10633 = m178;	// L16419
          ap_int<34> v10634 = v10633 + 1;	// L16420
          ap_int<66> v10635 = v10634;	// L16421
          ap_int<66> v10636 = v10635 * 8;	// L16422
          ap_int<66> v10637 = v10636 - 1;	// L16423
          int v10638 = v10632;	// L16424
          int v10639 = v10637;	// L16425
          ap_int<128> v10640;
          ap_int<128> v10640_tmp = v10630;
          v10640_tmp(v10639, v10638) = v10629;
          v10640 = v10640_tmp;	// L16426
          packed_c178 = v10640;	// L16427
        } else {
          ap_int<128> v10641 = packed_tmp178;	// L16429
          int64_t v10642 = m178;	// L16430
          int64_t v10643 = v10642 * 8;	// L16431
          ap_int<34> v10644 = m178;	// L16432
          ap_int<34> v10645 = v10644 + 1;	// L16433
          ap_int<66> v10646 = v10645;	// L16434
          ap_int<66> v10647 = v10646 * 8;	// L16435
          ap_int<66> v10648 = v10647 - 1;	// L16436
          int v10649 = v10643;	// L16437
          int v10650 = v10648;	// L16438
          int8_t v10651;
          ap_int<128> v10651_tmp = v10641;
          v10651 = v10651_tmp(v10650, v10649);	// L16439
          ap_int<128> v10652 = packed_c178;	// L16440
          ap_int<128> v10653;
          ap_int<128> v10653_tmp = v10652;
          v10653_tmp(v10650, v10649) = v10651;
          v10653 = v10653_tmp;	// L16450
          packed_c178 = v10653;	// L16451
        }
      }
      ap_int<128> v10654 = packed_c178;	// L16454
      v10602.write(v10654);	// L16455
    }
  }
}

void gemm_12_4(
  hls::stream< int8_t >& v10655,
  hls::stream< int8_t >& v10656,
  hls::stream< int8_t >& v10657,
  hls::stream< int8_t >& v10658,
  hls::stream< ap_uint<128> >& v10659,
  hls::stream< ap_uint<128> >& v10660
) {	// L16460
  l_S_mt_nt_0_mt210: for (int mt210 = 0; mt210 < 4; mt210++) {	// L16468
    l_nt210: for (int nt210 = 0; nt210 < 4; nt210++) {	// L16469
      int8_t c179;	// L16470
      c179 = 0;	// L16471
      l_S_k_0_k210: for (int k210 = 0; k210 < 64; k210++) {	// L16472
        int8_t v10665 = v10655.read();	// L16473
        int8_t a191;	// L16474
        a191 = v10665;	// L16475
        int8_t v10667 = v10656.read();	// L16476
        int8_t b195;	// L16477
        b195 = v10667;	// L16478
        int8_t v10669 = a191;	// L16479
        int8_t v10670 = b195;	// L16480
        int16_t v10671 = v10669;	// L16481
        int16_t v10672 = v10670;	// L16482
        int16_t v10673 = v10671 * v10672;	// L16483
        int8_t v10674 = c179;	// L16484
        ap_int<17> v10675 = v10674;	// L16485
        ap_int<17> v10676 = v10673;	// L16486
        ap_int<17> v10677 = v10675 + v10676;	// L16487
        int8_t v10678 = v10677;	// L16488
        c179 = v10678;	// L16489
        int8_t v10679 = a191;	// L16490
        v10657.write(v10679);	// L16491
        int8_t v10680 = b195;	// L16492
        v10658.write(v10680);	// L16493
      }
      ap_uint<128> packed_tmp179;	// L16495
      ap_uint<128> v10682 = v10659.read();	// L16496
      packed_tmp179 = v10682;	// L16497
      ap_uint<128> packed_c179;	// L16498
      packed_c179 = 0;	// L16499
      l_S_m_1_m179: for (int m179 = 0; m179 < 16; m179++) {	// L16500
        ap_int<33> v10685 = m179;	// L16501
        bool v10686 = v10685 == 11;	// L16502
        if (v10686) {	// L16503
          int8_t v10687 = c179;	// L16504
          ap_int<128> v10688 = packed_c179;	// L16505
          int64_t v10689 = m179;	// L16506
          int64_t v10690 = v10689 * 8;	// L16507
          ap_int<34> v10691 = m179;	// L16508
          ap_int<34> v10692 = v10691 + 1;	// L16509
          ap_int<66> v10693 = v10692;	// L16510
          ap_int<66> v10694 = v10693 * 8;	// L16511
          ap_int<66> v10695 = v10694 - 1;	// L16512
          int v10696 = v10690;	// L16513
          int v10697 = v10695;	// L16514
          ap_int<128> v10698;
          ap_int<128> v10698_tmp = v10688;
          v10698_tmp(v10697, v10696) = v10687;
          v10698 = v10698_tmp;	// L16515
          packed_c179 = v10698;	// L16516
        } else {
          ap_int<128> v10699 = packed_tmp179;	// L16518
          int64_t v10700 = m179;	// L16519
          int64_t v10701 = v10700 * 8;	// L16520
          ap_int<34> v10702 = m179;	// L16521
          ap_int<34> v10703 = v10702 + 1;	// L16522
          ap_int<66> v10704 = v10703;	// L16523
          ap_int<66> v10705 = v10704 * 8;	// L16524
          ap_int<66> v10706 = v10705 - 1;	// L16525
          int v10707 = v10701;	// L16526
          int v10708 = v10706;	// L16527
          int8_t v10709;
          ap_int<128> v10709_tmp = v10699;
          v10709 = v10709_tmp(v10708, v10707);	// L16528
          ap_int<128> v10710 = packed_c179;	// L16529
          ap_int<128> v10711;
          ap_int<128> v10711_tmp = v10710;
          v10711_tmp(v10708, v10707) = v10709;
          v10711 = v10711_tmp;	// L16539
          packed_c179 = v10711;	// L16540
        }
      }
      ap_int<128> v10712 = packed_c179;	// L16543
      v10660.write(v10712);	// L16544
    }
  }
}

void gemm_12_5(
  hls::stream< int8_t >& v10713,
  hls::stream< int8_t >& v10714,
  hls::stream< int8_t >& v10715,
  hls::stream< int8_t >& v10716,
  hls::stream< ap_uint<128> >& v10717,
  hls::stream< ap_uint<128> >& v10718
) {	// L16549
  l_S_mt_nt_0_mt211: for (int mt211 = 0; mt211 < 4; mt211++) {	// L16557
    l_nt211: for (int nt211 = 0; nt211 < 4; nt211++) {	// L16558
      int8_t c180;	// L16559
      c180 = 0;	// L16560
      l_S_k_0_k211: for (int k211 = 0; k211 < 64; k211++) {	// L16561
        int8_t v10723 = v10713.read();	// L16562
        int8_t a192;	// L16563
        a192 = v10723;	// L16564
        int8_t v10725 = v10714.read();	// L16565
        int8_t b196;	// L16566
        b196 = v10725;	// L16567
        int8_t v10727 = a192;	// L16568
        int8_t v10728 = b196;	// L16569
        int16_t v10729 = v10727;	// L16570
        int16_t v10730 = v10728;	// L16571
        int16_t v10731 = v10729 * v10730;	// L16572
        int8_t v10732 = c180;	// L16573
        ap_int<17> v10733 = v10732;	// L16574
        ap_int<17> v10734 = v10731;	// L16575
        ap_int<17> v10735 = v10733 + v10734;	// L16576
        int8_t v10736 = v10735;	// L16577
        c180 = v10736;	// L16578
        int8_t v10737 = a192;	// L16579
        v10715.write(v10737);	// L16580
        int8_t v10738 = b196;	// L16581
        v10716.write(v10738);	// L16582
      }
      ap_uint<128> packed_tmp180;	// L16584
      ap_uint<128> v10740 = v10717.read();	// L16585
      packed_tmp180 = v10740;	// L16586
      ap_uint<128> packed_c180;	// L16587
      packed_c180 = 0;	// L16588
      l_S_m_1_m180: for (int m180 = 0; m180 < 16; m180++) {	// L16589
        ap_int<33> v10743 = m180;	// L16590
        bool v10744 = v10743 == 11;	// L16591
        if (v10744) {	// L16592
          int8_t v10745 = c180;	// L16593
          ap_int<128> v10746 = packed_c180;	// L16594
          int64_t v10747 = m180;	// L16595
          int64_t v10748 = v10747 * 8;	// L16596
          ap_int<34> v10749 = m180;	// L16597
          ap_int<34> v10750 = v10749 + 1;	// L16598
          ap_int<66> v10751 = v10750;	// L16599
          ap_int<66> v10752 = v10751 * 8;	// L16600
          ap_int<66> v10753 = v10752 - 1;	// L16601
          int v10754 = v10748;	// L16602
          int v10755 = v10753;	// L16603
          ap_int<128> v10756;
          ap_int<128> v10756_tmp = v10746;
          v10756_tmp(v10755, v10754) = v10745;
          v10756 = v10756_tmp;	// L16604
          packed_c180 = v10756;	// L16605
        } else {
          ap_int<128> v10757 = packed_tmp180;	// L16607
          int64_t v10758 = m180;	// L16608
          int64_t v10759 = v10758 * 8;	// L16609
          ap_int<34> v10760 = m180;	// L16610
          ap_int<34> v10761 = v10760 + 1;	// L16611
          ap_int<66> v10762 = v10761;	// L16612
          ap_int<66> v10763 = v10762 * 8;	// L16613
          ap_int<66> v10764 = v10763 - 1;	// L16614
          int v10765 = v10759;	// L16615
          int v10766 = v10764;	// L16616
          int8_t v10767;
          ap_int<128> v10767_tmp = v10757;
          v10767 = v10767_tmp(v10766, v10765);	// L16617
          ap_int<128> v10768 = packed_c180;	// L16618
          ap_int<128> v10769;
          ap_int<128> v10769_tmp = v10768;
          v10769_tmp(v10766, v10765) = v10767;
          v10769 = v10769_tmp;	// L16628
          packed_c180 = v10769;	// L16629
        }
      }
      ap_int<128> v10770 = packed_c180;	// L16632
      v10718.write(v10770);	// L16633
    }
  }
}

void gemm_12_6(
  hls::stream< int8_t >& v10771,
  hls::stream< int8_t >& v10772,
  hls::stream< int8_t >& v10773,
  hls::stream< int8_t >& v10774,
  hls::stream< ap_uint<128> >& v10775,
  hls::stream< ap_uint<128> >& v10776
) {	// L16638
  l_S_mt_nt_0_mt212: for (int mt212 = 0; mt212 < 4; mt212++) {	// L16646
    l_nt212: for (int nt212 = 0; nt212 < 4; nt212++) {	// L16647
      int8_t c181;	// L16648
      c181 = 0;	// L16649
      l_S_k_0_k212: for (int k212 = 0; k212 < 64; k212++) {	// L16650
        int8_t v10781 = v10771.read();	// L16651
        int8_t a193;	// L16652
        a193 = v10781;	// L16653
        int8_t v10783 = v10772.read();	// L16654
        int8_t b197;	// L16655
        b197 = v10783;	// L16656
        int8_t v10785 = a193;	// L16657
        int8_t v10786 = b197;	// L16658
        int16_t v10787 = v10785;	// L16659
        int16_t v10788 = v10786;	// L16660
        int16_t v10789 = v10787 * v10788;	// L16661
        int8_t v10790 = c181;	// L16662
        ap_int<17> v10791 = v10790;	// L16663
        ap_int<17> v10792 = v10789;	// L16664
        ap_int<17> v10793 = v10791 + v10792;	// L16665
        int8_t v10794 = v10793;	// L16666
        c181 = v10794;	// L16667
        int8_t v10795 = a193;	// L16668
        v10773.write(v10795);	// L16669
        int8_t v10796 = b197;	// L16670
        v10774.write(v10796);	// L16671
      }
      ap_uint<128> packed_tmp181;	// L16673
      ap_uint<128> v10798 = v10775.read();	// L16674
      packed_tmp181 = v10798;	// L16675
      ap_uint<128> packed_c181;	// L16676
      packed_c181 = 0;	// L16677
      l_S_m_1_m181: for (int m181 = 0; m181 < 16; m181++) {	// L16678
        ap_int<33> v10801 = m181;	// L16679
        bool v10802 = v10801 == 11;	// L16680
        if (v10802) {	// L16681
          int8_t v10803 = c181;	// L16682
          ap_int<128> v10804 = packed_c181;	// L16683
          int64_t v10805 = m181;	// L16684
          int64_t v10806 = v10805 * 8;	// L16685
          ap_int<34> v10807 = m181;	// L16686
          ap_int<34> v10808 = v10807 + 1;	// L16687
          ap_int<66> v10809 = v10808;	// L16688
          ap_int<66> v10810 = v10809 * 8;	// L16689
          ap_int<66> v10811 = v10810 - 1;	// L16690
          int v10812 = v10806;	// L16691
          int v10813 = v10811;	// L16692
          ap_int<128> v10814;
          ap_int<128> v10814_tmp = v10804;
          v10814_tmp(v10813, v10812) = v10803;
          v10814 = v10814_tmp;	// L16693
          packed_c181 = v10814;	// L16694
        } else {
          ap_int<128> v10815 = packed_tmp181;	// L16696
          int64_t v10816 = m181;	// L16697
          int64_t v10817 = v10816 * 8;	// L16698
          ap_int<34> v10818 = m181;	// L16699
          ap_int<34> v10819 = v10818 + 1;	// L16700
          ap_int<66> v10820 = v10819;	// L16701
          ap_int<66> v10821 = v10820 * 8;	// L16702
          ap_int<66> v10822 = v10821 - 1;	// L16703
          int v10823 = v10817;	// L16704
          int v10824 = v10822;	// L16705
          int8_t v10825;
          ap_int<128> v10825_tmp = v10815;
          v10825 = v10825_tmp(v10824, v10823);	// L16706
          ap_int<128> v10826 = packed_c181;	// L16707
          ap_int<128> v10827;
          ap_int<128> v10827_tmp = v10826;
          v10827_tmp(v10824, v10823) = v10825;
          v10827 = v10827_tmp;	// L16717
          packed_c181 = v10827;	// L16718
        }
      }
      ap_int<128> v10828 = packed_c181;	// L16721
      v10776.write(v10828);	// L16722
    }
  }
}

void gemm_12_7(
  hls::stream< int8_t >& v10829,
  hls::stream< int8_t >& v10830,
  hls::stream< int8_t >& v10831,
  hls::stream< int8_t >& v10832,
  hls::stream< ap_uint<128> >& v10833,
  hls::stream< ap_uint<128> >& v10834
) {	// L16727
  l_S_mt_nt_0_mt213: for (int mt213 = 0; mt213 < 4; mt213++) {	// L16735
    l_nt213: for (int nt213 = 0; nt213 < 4; nt213++) {	// L16736
      int8_t c182;	// L16737
      c182 = 0;	// L16738
      l_S_k_0_k213: for (int k213 = 0; k213 < 64; k213++) {	// L16739
        int8_t v10839 = v10829.read();	// L16740
        int8_t a194;	// L16741
        a194 = v10839;	// L16742
        int8_t v10841 = v10830.read();	// L16743
        int8_t b198;	// L16744
        b198 = v10841;	// L16745
        int8_t v10843 = a194;	// L16746
        int8_t v10844 = b198;	// L16747
        int16_t v10845 = v10843;	// L16748
        int16_t v10846 = v10844;	// L16749
        int16_t v10847 = v10845 * v10846;	// L16750
        int8_t v10848 = c182;	// L16751
        ap_int<17> v10849 = v10848;	// L16752
        ap_int<17> v10850 = v10847;	// L16753
        ap_int<17> v10851 = v10849 + v10850;	// L16754
        int8_t v10852 = v10851;	// L16755
        c182 = v10852;	// L16756
        int8_t v10853 = a194;	// L16757
        v10831.write(v10853);	// L16758
        int8_t v10854 = b198;	// L16759
        v10832.write(v10854);	// L16760
      }
      ap_uint<128> packed_tmp182;	// L16762
      ap_uint<128> v10856 = v10833.read();	// L16763
      packed_tmp182 = v10856;	// L16764
      ap_uint<128> packed_c182;	// L16765
      packed_c182 = 0;	// L16766
      l_S_m_1_m182: for (int m182 = 0; m182 < 16; m182++) {	// L16767
        ap_int<33> v10859 = m182;	// L16768
        bool v10860 = v10859 == 11;	// L16769
        if (v10860) {	// L16770
          int8_t v10861 = c182;	// L16771
          ap_int<128> v10862 = packed_c182;	// L16772
          int64_t v10863 = m182;	// L16773
          int64_t v10864 = v10863 * 8;	// L16774
          ap_int<34> v10865 = m182;	// L16775
          ap_int<34> v10866 = v10865 + 1;	// L16776
          ap_int<66> v10867 = v10866;	// L16777
          ap_int<66> v10868 = v10867 * 8;	// L16778
          ap_int<66> v10869 = v10868 - 1;	// L16779
          int v10870 = v10864;	// L16780
          int v10871 = v10869;	// L16781
          ap_int<128> v10872;
          ap_int<128> v10872_tmp = v10862;
          v10872_tmp(v10871, v10870) = v10861;
          v10872 = v10872_tmp;	// L16782
          packed_c182 = v10872;	// L16783
        } else {
          ap_int<128> v10873 = packed_tmp182;	// L16785
          int64_t v10874 = m182;	// L16786
          int64_t v10875 = v10874 * 8;	// L16787
          ap_int<34> v10876 = m182;	// L16788
          ap_int<34> v10877 = v10876 + 1;	// L16789
          ap_int<66> v10878 = v10877;	// L16790
          ap_int<66> v10879 = v10878 * 8;	// L16791
          ap_int<66> v10880 = v10879 - 1;	// L16792
          int v10881 = v10875;	// L16793
          int v10882 = v10880;	// L16794
          int8_t v10883;
          ap_int<128> v10883_tmp = v10873;
          v10883 = v10883_tmp(v10882, v10881);	// L16795
          ap_int<128> v10884 = packed_c182;	// L16796
          ap_int<128> v10885;
          ap_int<128> v10885_tmp = v10884;
          v10885_tmp(v10882, v10881) = v10883;
          v10885 = v10885_tmp;	// L16806
          packed_c182 = v10885;	// L16807
        }
      }
      ap_int<128> v10886 = packed_c182;	// L16810
      v10834.write(v10886);	// L16811
    }
  }
}

void gemm_12_8(
  hls::stream< int8_t >& v10887,
  hls::stream< int8_t >& v10888,
  hls::stream< int8_t >& v10889,
  hls::stream< int8_t >& v10890,
  hls::stream< ap_uint<128> >& v10891,
  hls::stream< ap_uint<128> >& v10892
) {	// L16816
  l_S_mt_nt_0_mt214: for (int mt214 = 0; mt214 < 4; mt214++) {	// L16824
    l_nt214: for (int nt214 = 0; nt214 < 4; nt214++) {	// L16825
      int8_t c183;	// L16826
      c183 = 0;	// L16827
      l_S_k_0_k214: for (int k214 = 0; k214 < 64; k214++) {	// L16828
        int8_t v10897 = v10887.read();	// L16829
        int8_t a195;	// L16830
        a195 = v10897;	// L16831
        int8_t v10899 = v10888.read();	// L16832
        int8_t b199;	// L16833
        b199 = v10899;	// L16834
        int8_t v10901 = a195;	// L16835
        int8_t v10902 = b199;	// L16836
        int16_t v10903 = v10901;	// L16837
        int16_t v10904 = v10902;	// L16838
        int16_t v10905 = v10903 * v10904;	// L16839
        int8_t v10906 = c183;	// L16840
        ap_int<17> v10907 = v10906;	// L16841
        ap_int<17> v10908 = v10905;	// L16842
        ap_int<17> v10909 = v10907 + v10908;	// L16843
        int8_t v10910 = v10909;	// L16844
        c183 = v10910;	// L16845
        int8_t v10911 = a195;	// L16846
        v10889.write(v10911);	// L16847
        int8_t v10912 = b199;	// L16848
        v10890.write(v10912);	// L16849
      }
      ap_uint<128> packed_tmp183;	// L16851
      ap_uint<128> v10914 = v10891.read();	// L16852
      packed_tmp183 = v10914;	// L16853
      ap_uint<128> packed_c183;	// L16854
      packed_c183 = 0;	// L16855
      l_S_m_1_m183: for (int m183 = 0; m183 < 16; m183++) {	// L16856
        ap_int<33> v10917 = m183;	// L16857
        bool v10918 = v10917 == 11;	// L16858
        if (v10918) {	// L16859
          int8_t v10919 = c183;	// L16860
          ap_int<128> v10920 = packed_c183;	// L16861
          int64_t v10921 = m183;	// L16862
          int64_t v10922 = v10921 * 8;	// L16863
          ap_int<34> v10923 = m183;	// L16864
          ap_int<34> v10924 = v10923 + 1;	// L16865
          ap_int<66> v10925 = v10924;	// L16866
          ap_int<66> v10926 = v10925 * 8;	// L16867
          ap_int<66> v10927 = v10926 - 1;	// L16868
          int v10928 = v10922;	// L16869
          int v10929 = v10927;	// L16870
          ap_int<128> v10930;
          ap_int<128> v10930_tmp = v10920;
          v10930_tmp(v10929, v10928) = v10919;
          v10930 = v10930_tmp;	// L16871
          packed_c183 = v10930;	// L16872
        } else {
          ap_int<128> v10931 = packed_tmp183;	// L16874
          int64_t v10932 = m183;	// L16875
          int64_t v10933 = v10932 * 8;	// L16876
          ap_int<34> v10934 = m183;	// L16877
          ap_int<34> v10935 = v10934 + 1;	// L16878
          ap_int<66> v10936 = v10935;	// L16879
          ap_int<66> v10937 = v10936 * 8;	// L16880
          ap_int<66> v10938 = v10937 - 1;	// L16881
          int v10939 = v10933;	// L16882
          int v10940 = v10938;	// L16883
          int8_t v10941;
          ap_int<128> v10941_tmp = v10931;
          v10941 = v10941_tmp(v10940, v10939);	// L16884
          ap_int<128> v10942 = packed_c183;	// L16885
          ap_int<128> v10943;
          ap_int<128> v10943_tmp = v10942;
          v10943_tmp(v10940, v10939) = v10941;
          v10943 = v10943_tmp;	// L16895
          packed_c183 = v10943;	// L16896
        }
      }
      ap_int<128> v10944 = packed_c183;	// L16899
      v10892.write(v10944);	// L16900
    }
  }
}

void gemm_12_9(
  hls::stream< int8_t >& v10945,
  hls::stream< int8_t >& v10946,
  hls::stream< int8_t >& v10947,
  hls::stream< int8_t >& v10948,
  hls::stream< ap_uint<128> >& v10949,
  hls::stream< ap_uint<128> >& v10950
) {	// L16905
  l_S_mt_nt_0_mt215: for (int mt215 = 0; mt215 < 4; mt215++) {	// L16913
    l_nt215: for (int nt215 = 0; nt215 < 4; nt215++) {	// L16914
      int8_t c184;	// L16915
      c184 = 0;	// L16916
      l_S_k_0_k215: for (int k215 = 0; k215 < 64; k215++) {	// L16917
        int8_t v10955 = v10945.read();	// L16918
        int8_t a196;	// L16919
        a196 = v10955;	// L16920
        int8_t v10957 = v10946.read();	// L16921
        int8_t b200;	// L16922
        b200 = v10957;	// L16923
        int8_t v10959 = a196;	// L16924
        int8_t v10960 = b200;	// L16925
        int16_t v10961 = v10959;	// L16926
        int16_t v10962 = v10960;	// L16927
        int16_t v10963 = v10961 * v10962;	// L16928
        int8_t v10964 = c184;	// L16929
        ap_int<17> v10965 = v10964;	// L16930
        ap_int<17> v10966 = v10963;	// L16931
        ap_int<17> v10967 = v10965 + v10966;	// L16932
        int8_t v10968 = v10967;	// L16933
        c184 = v10968;	// L16934
        int8_t v10969 = a196;	// L16935
        v10947.write(v10969);	// L16936
        int8_t v10970 = b200;	// L16937
        v10948.write(v10970);	// L16938
      }
      ap_uint<128> packed_tmp184;	// L16940
      ap_uint<128> v10972 = v10949.read();	// L16941
      packed_tmp184 = v10972;	// L16942
      ap_uint<128> packed_c184;	// L16943
      packed_c184 = 0;	// L16944
      l_S_m_1_m184: for (int m184 = 0; m184 < 16; m184++) {	// L16945
        ap_int<33> v10975 = m184;	// L16946
        bool v10976 = v10975 == 11;	// L16947
        if (v10976) {	// L16948
          int8_t v10977 = c184;	// L16949
          ap_int<128> v10978 = packed_c184;	// L16950
          int64_t v10979 = m184;	// L16951
          int64_t v10980 = v10979 * 8;	// L16952
          ap_int<34> v10981 = m184;	// L16953
          ap_int<34> v10982 = v10981 + 1;	// L16954
          ap_int<66> v10983 = v10982;	// L16955
          ap_int<66> v10984 = v10983 * 8;	// L16956
          ap_int<66> v10985 = v10984 - 1;	// L16957
          int v10986 = v10980;	// L16958
          int v10987 = v10985;	// L16959
          ap_int<128> v10988;
          ap_int<128> v10988_tmp = v10978;
          v10988_tmp(v10987, v10986) = v10977;
          v10988 = v10988_tmp;	// L16960
          packed_c184 = v10988;	// L16961
        } else {
          ap_int<128> v10989 = packed_tmp184;	// L16963
          int64_t v10990 = m184;	// L16964
          int64_t v10991 = v10990 * 8;	// L16965
          ap_int<34> v10992 = m184;	// L16966
          ap_int<34> v10993 = v10992 + 1;	// L16967
          ap_int<66> v10994 = v10993;	// L16968
          ap_int<66> v10995 = v10994 * 8;	// L16969
          ap_int<66> v10996 = v10995 - 1;	// L16970
          int v10997 = v10991;	// L16971
          int v10998 = v10996;	// L16972
          int8_t v10999;
          ap_int<128> v10999_tmp = v10989;
          v10999 = v10999_tmp(v10998, v10997);	// L16973
          ap_int<128> v11000 = packed_c184;	// L16974
          ap_int<128> v11001;
          ap_int<128> v11001_tmp = v11000;
          v11001_tmp(v10998, v10997) = v10999;
          v11001 = v11001_tmp;	// L16984
          packed_c184 = v11001;	// L16985
        }
      }
      ap_int<128> v11002 = packed_c184;	// L16988
      v10950.write(v11002);	// L16989
    }
  }
}

void gemm_12_10(
  hls::stream< int8_t >& v11003,
  hls::stream< int8_t >& v11004,
  hls::stream< int8_t >& v11005,
  hls::stream< int8_t >& v11006,
  hls::stream< ap_uint<128> >& v11007,
  hls::stream< ap_uint<128> >& v11008
) {	// L16994
  l_S_mt_nt_0_mt216: for (int mt216 = 0; mt216 < 4; mt216++) {	// L17002
    l_nt216: for (int nt216 = 0; nt216 < 4; nt216++) {	// L17003
      int8_t c185;	// L17004
      c185 = 0;	// L17005
      l_S_k_0_k216: for (int k216 = 0; k216 < 64; k216++) {	// L17006
        int8_t v11013 = v11003.read();	// L17007
        int8_t a197;	// L17008
        a197 = v11013;	// L17009
        int8_t v11015 = v11004.read();	// L17010
        int8_t b201;	// L17011
        b201 = v11015;	// L17012
        int8_t v11017 = a197;	// L17013
        int8_t v11018 = b201;	// L17014
        int16_t v11019 = v11017;	// L17015
        int16_t v11020 = v11018;	// L17016
        int16_t v11021 = v11019 * v11020;	// L17017
        int8_t v11022 = c185;	// L17018
        ap_int<17> v11023 = v11022;	// L17019
        ap_int<17> v11024 = v11021;	// L17020
        ap_int<17> v11025 = v11023 + v11024;	// L17021
        int8_t v11026 = v11025;	// L17022
        c185 = v11026;	// L17023
        int8_t v11027 = a197;	// L17024
        v11005.write(v11027);	// L17025
        int8_t v11028 = b201;	// L17026
        v11006.write(v11028);	// L17027
      }
      ap_uint<128> packed_tmp185;	// L17029
      ap_uint<128> v11030 = v11007.read();	// L17030
      packed_tmp185 = v11030;	// L17031
      ap_uint<128> packed_c185;	// L17032
      packed_c185 = 0;	// L17033
      l_S_m_1_m185: for (int m185 = 0; m185 < 16; m185++) {	// L17034
        ap_int<33> v11033 = m185;	// L17035
        bool v11034 = v11033 == 11;	// L17036
        if (v11034) {	// L17037
          int8_t v11035 = c185;	// L17038
          ap_int<128> v11036 = packed_c185;	// L17039
          int64_t v11037 = m185;	// L17040
          int64_t v11038 = v11037 * 8;	// L17041
          ap_int<34> v11039 = m185;	// L17042
          ap_int<34> v11040 = v11039 + 1;	// L17043
          ap_int<66> v11041 = v11040;	// L17044
          ap_int<66> v11042 = v11041 * 8;	// L17045
          ap_int<66> v11043 = v11042 - 1;	// L17046
          int v11044 = v11038;	// L17047
          int v11045 = v11043;	// L17048
          ap_int<128> v11046;
          ap_int<128> v11046_tmp = v11036;
          v11046_tmp(v11045, v11044) = v11035;
          v11046 = v11046_tmp;	// L17049
          packed_c185 = v11046;	// L17050
        } else {
          ap_int<128> v11047 = packed_tmp185;	// L17052
          int64_t v11048 = m185;	// L17053
          int64_t v11049 = v11048 * 8;	// L17054
          ap_int<34> v11050 = m185;	// L17055
          ap_int<34> v11051 = v11050 + 1;	// L17056
          ap_int<66> v11052 = v11051;	// L17057
          ap_int<66> v11053 = v11052 * 8;	// L17058
          ap_int<66> v11054 = v11053 - 1;	// L17059
          int v11055 = v11049;	// L17060
          int v11056 = v11054;	// L17061
          int8_t v11057;
          ap_int<128> v11057_tmp = v11047;
          v11057 = v11057_tmp(v11056, v11055);	// L17062
          ap_int<128> v11058 = packed_c185;	// L17063
          ap_int<128> v11059;
          ap_int<128> v11059_tmp = v11058;
          v11059_tmp(v11056, v11055) = v11057;
          v11059 = v11059_tmp;	// L17073
          packed_c185 = v11059;	// L17074
        }
      }
      ap_int<128> v11060 = packed_c185;	// L17077
      v11008.write(v11060);	// L17078
    }
  }
}

void gemm_12_11(
  hls::stream< int8_t >& v11061,
  hls::stream< int8_t >& v11062,
  hls::stream< int8_t >& v11063,
  hls::stream< int8_t >& v11064,
  hls::stream< ap_uint<128> >& v11065,
  hls::stream< ap_uint<128> >& v11066
) {	// L17083
  l_S_mt_nt_0_mt217: for (int mt217 = 0; mt217 < 4; mt217++) {	// L17091
    l_nt217: for (int nt217 = 0; nt217 < 4; nt217++) {	// L17092
      int8_t c186;	// L17093
      c186 = 0;	// L17094
      l_S_k_0_k217: for (int k217 = 0; k217 < 64; k217++) {	// L17095
        int8_t v11071 = v11061.read();	// L17096
        int8_t a198;	// L17097
        a198 = v11071;	// L17098
        int8_t v11073 = v11062.read();	// L17099
        int8_t b202;	// L17100
        b202 = v11073;	// L17101
        int8_t v11075 = a198;	// L17102
        int8_t v11076 = b202;	// L17103
        int16_t v11077 = v11075;	// L17104
        int16_t v11078 = v11076;	// L17105
        int16_t v11079 = v11077 * v11078;	// L17106
        int8_t v11080 = c186;	// L17107
        ap_int<17> v11081 = v11080;	// L17108
        ap_int<17> v11082 = v11079;	// L17109
        ap_int<17> v11083 = v11081 + v11082;	// L17110
        int8_t v11084 = v11083;	// L17111
        c186 = v11084;	// L17112
        int8_t v11085 = a198;	// L17113
        v11063.write(v11085);	// L17114
        int8_t v11086 = b202;	// L17115
        v11064.write(v11086);	// L17116
      }
      ap_uint<128> packed_tmp186;	// L17118
      ap_uint<128> v11088 = v11065.read();	// L17119
      packed_tmp186 = v11088;	// L17120
      ap_uint<128> packed_c186;	// L17121
      packed_c186 = 0;	// L17122
      l_S_m_1_m186: for (int m186 = 0; m186 < 16; m186++) {	// L17123
        ap_int<33> v11091 = m186;	// L17124
        bool v11092 = v11091 == 11;	// L17125
        if (v11092) {	// L17126
          int8_t v11093 = c186;	// L17127
          ap_int<128> v11094 = packed_c186;	// L17128
          int64_t v11095 = m186;	// L17129
          int64_t v11096 = v11095 * 8;	// L17130
          ap_int<34> v11097 = m186;	// L17131
          ap_int<34> v11098 = v11097 + 1;	// L17132
          ap_int<66> v11099 = v11098;	// L17133
          ap_int<66> v11100 = v11099 * 8;	// L17134
          ap_int<66> v11101 = v11100 - 1;	// L17135
          int v11102 = v11096;	// L17136
          int v11103 = v11101;	// L17137
          ap_int<128> v11104;
          ap_int<128> v11104_tmp = v11094;
          v11104_tmp(v11103, v11102) = v11093;
          v11104 = v11104_tmp;	// L17138
          packed_c186 = v11104;	// L17139
        } else {
          ap_int<128> v11105 = packed_tmp186;	// L17141
          int64_t v11106 = m186;	// L17142
          int64_t v11107 = v11106 * 8;	// L17143
          ap_int<34> v11108 = m186;	// L17144
          ap_int<34> v11109 = v11108 + 1;	// L17145
          ap_int<66> v11110 = v11109;	// L17146
          ap_int<66> v11111 = v11110 * 8;	// L17147
          ap_int<66> v11112 = v11111 - 1;	// L17148
          int v11113 = v11107;	// L17149
          int v11114 = v11112;	// L17150
          int8_t v11115;
          ap_int<128> v11115_tmp = v11105;
          v11115 = v11115_tmp(v11114, v11113);	// L17151
          ap_int<128> v11116 = packed_c186;	// L17152
          ap_int<128> v11117;
          ap_int<128> v11117_tmp = v11116;
          v11117_tmp(v11114, v11113) = v11115;
          v11117 = v11117_tmp;	// L17162
          packed_c186 = v11117;	// L17163
        }
      }
      ap_int<128> v11118 = packed_c186;	// L17166
      v11066.write(v11118);	// L17167
    }
  }
}

void gemm_12_12(
  hls::stream< int8_t >& v11119,
  hls::stream< int8_t >& v11120,
  hls::stream< int8_t >& v11121,
  hls::stream< int8_t >& v11122,
  hls::stream< ap_uint<128> >& v11123,
  hls::stream< ap_uint<128> >& v11124
) {	// L17172
  l_S_mt_nt_0_mt218: for (int mt218 = 0; mt218 < 4; mt218++) {	// L17180
    l_nt218: for (int nt218 = 0; nt218 < 4; nt218++) {	// L17181
      int8_t c187;	// L17182
      c187 = 0;	// L17183
      l_S_k_0_k218: for (int k218 = 0; k218 < 64; k218++) {	// L17184
        int8_t v11129 = v11119.read();	// L17185
        int8_t a199;	// L17186
        a199 = v11129;	// L17187
        int8_t v11131 = v11120.read();	// L17188
        int8_t b203;	// L17189
        b203 = v11131;	// L17190
        int8_t v11133 = a199;	// L17191
        int8_t v11134 = b203;	// L17192
        int16_t v11135 = v11133;	// L17193
        int16_t v11136 = v11134;	// L17194
        int16_t v11137 = v11135 * v11136;	// L17195
        int8_t v11138 = c187;	// L17196
        ap_int<17> v11139 = v11138;	// L17197
        ap_int<17> v11140 = v11137;	// L17198
        ap_int<17> v11141 = v11139 + v11140;	// L17199
        int8_t v11142 = v11141;	// L17200
        c187 = v11142;	// L17201
        int8_t v11143 = a199;	// L17202
        v11121.write(v11143);	// L17203
        int8_t v11144 = b203;	// L17204
        v11122.write(v11144);	// L17205
      }
      ap_uint<128> packed_tmp187;	// L17207
      ap_uint<128> v11146 = v11123.read();	// L17208
      packed_tmp187 = v11146;	// L17209
      ap_uint<128> packed_c187;	// L17210
      packed_c187 = 0;	// L17211
      l_S_m_1_m187: for (int m187 = 0; m187 < 16; m187++) {	// L17212
        ap_int<33> v11149 = m187;	// L17213
        bool v11150 = v11149 == 11;	// L17214
        if (v11150) {	// L17215
          int8_t v11151 = c187;	// L17216
          ap_int<128> v11152 = packed_c187;	// L17217
          int64_t v11153 = m187;	// L17218
          int64_t v11154 = v11153 * 8;	// L17219
          ap_int<34> v11155 = m187;	// L17220
          ap_int<34> v11156 = v11155 + 1;	// L17221
          ap_int<66> v11157 = v11156;	// L17222
          ap_int<66> v11158 = v11157 * 8;	// L17223
          ap_int<66> v11159 = v11158 - 1;	// L17224
          int v11160 = v11154;	// L17225
          int v11161 = v11159;	// L17226
          ap_int<128> v11162;
          ap_int<128> v11162_tmp = v11152;
          v11162_tmp(v11161, v11160) = v11151;
          v11162 = v11162_tmp;	// L17227
          packed_c187 = v11162;	// L17228
        } else {
          ap_int<128> v11163 = packed_tmp187;	// L17230
          int64_t v11164 = m187;	// L17231
          int64_t v11165 = v11164 * 8;	// L17232
          ap_int<34> v11166 = m187;	// L17233
          ap_int<34> v11167 = v11166 + 1;	// L17234
          ap_int<66> v11168 = v11167;	// L17235
          ap_int<66> v11169 = v11168 * 8;	// L17236
          ap_int<66> v11170 = v11169 - 1;	// L17237
          int v11171 = v11165;	// L17238
          int v11172 = v11170;	// L17239
          int8_t v11173;
          ap_int<128> v11173_tmp = v11163;
          v11173 = v11173_tmp(v11172, v11171);	// L17240
          ap_int<128> v11174 = packed_c187;	// L17241
          ap_int<128> v11175;
          ap_int<128> v11175_tmp = v11174;
          v11175_tmp(v11172, v11171) = v11173;
          v11175 = v11175_tmp;	// L17251
          packed_c187 = v11175;	// L17252
        }
      }
      ap_int<128> v11176 = packed_c187;	// L17255
      v11124.write(v11176);	// L17256
    }
  }
}

void gemm_12_13(
  hls::stream< int8_t >& v11177,
  hls::stream< int8_t >& v11178,
  hls::stream< int8_t >& v11179,
  hls::stream< int8_t >& v11180,
  hls::stream< ap_uint<128> >& v11181,
  hls::stream< ap_uint<128> >& v11182
) {	// L17261
  l_S_mt_nt_0_mt219: for (int mt219 = 0; mt219 < 4; mt219++) {	// L17269
    l_nt219: for (int nt219 = 0; nt219 < 4; nt219++) {	// L17270
      int8_t c188;	// L17271
      c188 = 0;	// L17272
      l_S_k_0_k219: for (int k219 = 0; k219 < 64; k219++) {	// L17273
        int8_t v11187 = v11177.read();	// L17274
        int8_t a200;	// L17275
        a200 = v11187;	// L17276
        int8_t v11189 = v11178.read();	// L17277
        int8_t b204;	// L17278
        b204 = v11189;	// L17279
        int8_t v11191 = a200;	// L17280
        int8_t v11192 = b204;	// L17281
        int16_t v11193 = v11191;	// L17282
        int16_t v11194 = v11192;	// L17283
        int16_t v11195 = v11193 * v11194;	// L17284
        int8_t v11196 = c188;	// L17285
        ap_int<17> v11197 = v11196;	// L17286
        ap_int<17> v11198 = v11195;	// L17287
        ap_int<17> v11199 = v11197 + v11198;	// L17288
        int8_t v11200 = v11199;	// L17289
        c188 = v11200;	// L17290
        int8_t v11201 = a200;	// L17291
        v11179.write(v11201);	// L17292
        int8_t v11202 = b204;	// L17293
        v11180.write(v11202);	// L17294
      }
      ap_uint<128> packed_tmp188;	// L17296
      ap_uint<128> v11204 = v11181.read();	// L17297
      packed_tmp188 = v11204;	// L17298
      ap_uint<128> packed_c188;	// L17299
      packed_c188 = 0;	// L17300
      l_S_m_1_m188: for (int m188 = 0; m188 < 16; m188++) {	// L17301
        ap_int<33> v11207 = m188;	// L17302
        bool v11208 = v11207 == 11;	// L17303
        if (v11208) {	// L17304
          int8_t v11209 = c188;	// L17305
          ap_int<128> v11210 = packed_c188;	// L17306
          int64_t v11211 = m188;	// L17307
          int64_t v11212 = v11211 * 8;	// L17308
          ap_int<34> v11213 = m188;	// L17309
          ap_int<34> v11214 = v11213 + 1;	// L17310
          ap_int<66> v11215 = v11214;	// L17311
          ap_int<66> v11216 = v11215 * 8;	// L17312
          ap_int<66> v11217 = v11216 - 1;	// L17313
          int v11218 = v11212;	// L17314
          int v11219 = v11217;	// L17315
          ap_int<128> v11220;
          ap_int<128> v11220_tmp = v11210;
          v11220_tmp(v11219, v11218) = v11209;
          v11220 = v11220_tmp;	// L17316
          packed_c188 = v11220;	// L17317
        } else {
          ap_int<128> v11221 = packed_tmp188;	// L17319
          int64_t v11222 = m188;	// L17320
          int64_t v11223 = v11222 * 8;	// L17321
          ap_int<34> v11224 = m188;	// L17322
          ap_int<34> v11225 = v11224 + 1;	// L17323
          ap_int<66> v11226 = v11225;	// L17324
          ap_int<66> v11227 = v11226 * 8;	// L17325
          ap_int<66> v11228 = v11227 - 1;	// L17326
          int v11229 = v11223;	// L17327
          int v11230 = v11228;	// L17328
          int8_t v11231;
          ap_int<128> v11231_tmp = v11221;
          v11231 = v11231_tmp(v11230, v11229);	// L17329
          ap_int<128> v11232 = packed_c188;	// L17330
          ap_int<128> v11233;
          ap_int<128> v11233_tmp = v11232;
          v11233_tmp(v11230, v11229) = v11231;
          v11233 = v11233_tmp;	// L17340
          packed_c188 = v11233;	// L17341
        }
      }
      ap_int<128> v11234 = packed_c188;	// L17344
      v11182.write(v11234);	// L17345
    }
  }
}

void gemm_12_14(
  hls::stream< int8_t >& v11235,
  hls::stream< int8_t >& v11236,
  hls::stream< int8_t >& v11237,
  hls::stream< int8_t >& v11238,
  hls::stream< ap_uint<128> >& v11239,
  hls::stream< ap_uint<128> >& v11240
) {	// L17350
  l_S_mt_nt_0_mt220: for (int mt220 = 0; mt220 < 4; mt220++) {	// L17358
    l_nt220: for (int nt220 = 0; nt220 < 4; nt220++) {	// L17359
      int8_t c189;	// L17360
      c189 = 0;	// L17361
      l_S_k_0_k220: for (int k220 = 0; k220 < 64; k220++) {	// L17362
        int8_t v11245 = v11235.read();	// L17363
        int8_t a201;	// L17364
        a201 = v11245;	// L17365
        int8_t v11247 = v11236.read();	// L17366
        int8_t b205;	// L17367
        b205 = v11247;	// L17368
        int8_t v11249 = a201;	// L17369
        int8_t v11250 = b205;	// L17370
        int16_t v11251 = v11249;	// L17371
        int16_t v11252 = v11250;	// L17372
        int16_t v11253 = v11251 * v11252;	// L17373
        int8_t v11254 = c189;	// L17374
        ap_int<17> v11255 = v11254;	// L17375
        ap_int<17> v11256 = v11253;	// L17376
        ap_int<17> v11257 = v11255 + v11256;	// L17377
        int8_t v11258 = v11257;	// L17378
        c189 = v11258;	// L17379
        int8_t v11259 = a201;	// L17380
        v11237.write(v11259);	// L17381
        int8_t v11260 = b205;	// L17382
        v11238.write(v11260);	// L17383
      }
      ap_uint<128> packed_tmp189;	// L17385
      ap_uint<128> v11262 = v11239.read();	// L17386
      packed_tmp189 = v11262;	// L17387
      ap_uint<128> packed_c189;	// L17388
      packed_c189 = 0;	// L17389
      l_S_m_1_m189: for (int m189 = 0; m189 < 16; m189++) {	// L17390
        ap_int<33> v11265 = m189;	// L17391
        bool v11266 = v11265 == 11;	// L17392
        if (v11266) {	// L17393
          int8_t v11267 = c189;	// L17394
          ap_int<128> v11268 = packed_c189;	// L17395
          int64_t v11269 = m189;	// L17396
          int64_t v11270 = v11269 * 8;	// L17397
          ap_int<34> v11271 = m189;	// L17398
          ap_int<34> v11272 = v11271 + 1;	// L17399
          ap_int<66> v11273 = v11272;	// L17400
          ap_int<66> v11274 = v11273 * 8;	// L17401
          ap_int<66> v11275 = v11274 - 1;	// L17402
          int v11276 = v11270;	// L17403
          int v11277 = v11275;	// L17404
          ap_int<128> v11278;
          ap_int<128> v11278_tmp = v11268;
          v11278_tmp(v11277, v11276) = v11267;
          v11278 = v11278_tmp;	// L17405
          packed_c189 = v11278;	// L17406
        } else {
          ap_int<128> v11279 = packed_tmp189;	// L17408
          int64_t v11280 = m189;	// L17409
          int64_t v11281 = v11280 * 8;	// L17410
          ap_int<34> v11282 = m189;	// L17411
          ap_int<34> v11283 = v11282 + 1;	// L17412
          ap_int<66> v11284 = v11283;	// L17413
          ap_int<66> v11285 = v11284 * 8;	// L17414
          ap_int<66> v11286 = v11285 - 1;	// L17415
          int v11287 = v11281;	// L17416
          int v11288 = v11286;	// L17417
          int8_t v11289;
          ap_int<128> v11289_tmp = v11279;
          v11289 = v11289_tmp(v11288, v11287);	// L17418
          ap_int<128> v11290 = packed_c189;	// L17419
          ap_int<128> v11291;
          ap_int<128> v11291_tmp = v11290;
          v11291_tmp(v11288, v11287) = v11289;
          v11291 = v11291_tmp;	// L17429
          packed_c189 = v11291;	// L17430
        }
      }
      ap_int<128> v11292 = packed_c189;	// L17433
      v11240.write(v11292);	// L17434
    }
  }
}

void gemm_12_15(
  hls::stream< int8_t >& v11293,
  hls::stream< int8_t >& v11294,
  hls::stream< int8_t >& v11295,
  hls::stream< int8_t >& v11296,
  hls::stream< ap_uint<128> >& v11297,
  hls::stream< ap_uint<128> >& v11298
) {	// L17439
  l_S_mt_nt_0_mt221: for (int mt221 = 0; mt221 < 4; mt221++) {	// L17447
    l_nt221: for (int nt221 = 0; nt221 < 4; nt221++) {	// L17448
      int8_t c190;	// L17449
      c190 = 0;	// L17450
      l_S_k_0_k221: for (int k221 = 0; k221 < 64; k221++) {	// L17451
        int8_t v11303 = v11293.read();	// L17452
        int8_t a202;	// L17453
        a202 = v11303;	// L17454
        int8_t v11305 = v11294.read();	// L17455
        int8_t b206;	// L17456
        b206 = v11305;	// L17457
        int8_t v11307 = a202;	// L17458
        int8_t v11308 = b206;	// L17459
        int16_t v11309 = v11307;	// L17460
        int16_t v11310 = v11308;	// L17461
        int16_t v11311 = v11309 * v11310;	// L17462
        int8_t v11312 = c190;	// L17463
        ap_int<17> v11313 = v11312;	// L17464
        ap_int<17> v11314 = v11311;	// L17465
        ap_int<17> v11315 = v11313 + v11314;	// L17466
        int8_t v11316 = v11315;	// L17467
        c190 = v11316;	// L17468
        int8_t v11317 = a202;	// L17469
        v11295.write(v11317);	// L17470
        int8_t v11318 = b206;	// L17471
        v11296.write(v11318);	// L17472
      }
      ap_uint<128> packed_tmp190;	// L17474
      ap_uint<128> v11320 = v11297.read();	// L17475
      packed_tmp190 = v11320;	// L17476
      ap_uint<128> packed_c190;	// L17477
      packed_c190 = 0;	// L17478
      l_S_m_1_m190: for (int m190 = 0; m190 < 16; m190++) {	// L17479
        ap_int<33> v11323 = m190;	// L17480
        bool v11324 = v11323 == 11;	// L17481
        if (v11324) {	// L17482
          int8_t v11325 = c190;	// L17483
          ap_int<128> v11326 = packed_c190;	// L17484
          int64_t v11327 = m190;	// L17485
          int64_t v11328 = v11327 * 8;	// L17486
          ap_int<34> v11329 = m190;	// L17487
          ap_int<34> v11330 = v11329 + 1;	// L17488
          ap_int<66> v11331 = v11330;	// L17489
          ap_int<66> v11332 = v11331 * 8;	// L17490
          ap_int<66> v11333 = v11332 - 1;	// L17491
          int v11334 = v11328;	// L17492
          int v11335 = v11333;	// L17493
          ap_int<128> v11336;
          ap_int<128> v11336_tmp = v11326;
          v11336_tmp(v11335, v11334) = v11325;
          v11336 = v11336_tmp;	// L17494
          packed_c190 = v11336;	// L17495
        } else {
          ap_int<128> v11337 = packed_tmp190;	// L17497
          int64_t v11338 = m190;	// L17498
          int64_t v11339 = v11338 * 8;	// L17499
          ap_int<34> v11340 = m190;	// L17500
          ap_int<34> v11341 = v11340 + 1;	// L17501
          ap_int<66> v11342 = v11341;	// L17502
          ap_int<66> v11343 = v11342 * 8;	// L17503
          ap_int<66> v11344 = v11343 - 1;	// L17504
          int v11345 = v11339;	// L17505
          int v11346 = v11344;	// L17506
          int8_t v11347;
          ap_int<128> v11347_tmp = v11337;
          v11347 = v11347_tmp(v11346, v11345);	// L17507
          ap_int<128> v11348 = packed_c190;	// L17508
          ap_int<128> v11349;
          ap_int<128> v11349_tmp = v11348;
          v11349_tmp(v11346, v11345) = v11347;
          v11349 = v11349_tmp;	// L17518
          packed_c190 = v11349;	// L17519
        }
      }
      ap_int<128> v11350 = packed_c190;	// L17522
      v11298.write(v11350);	// L17523
    }
  }
}

void gemm_12_16(
  hls::stream< int8_t >& v11351,
  hls::stream< int8_t >& v11352,
  hls::stream< int8_t >& v11353,
  hls::stream< ap_uint<128> >& v11354,
  hls::stream< ap_uint<128> >& v11355
) {	// L17528
  l_S_mt_nt_0_mt222: for (int mt222 = 0; mt222 < 4; mt222++) {	// L17536
    l_nt222: for (int nt222 = 0; nt222 < 4; nt222++) {	// L17537
      int8_t c191;	// L17538
      c191 = 0;	// L17539
      l_S_k_0_k222: for (int k222 = 0; k222 < 64; k222++) {	// L17540
        int8_t v11360 = v11351.read();	// L17541
        int8_t a203;	// L17542
        a203 = v11360;	// L17543
        int8_t v11362 = v11352.read();	// L17544
        int8_t b207;	// L17545
        b207 = v11362;	// L17546
        int8_t v11364 = a203;	// L17547
        int8_t v11365 = b207;	// L17548
        int16_t v11366 = v11364;	// L17549
        int16_t v11367 = v11365;	// L17550
        int16_t v11368 = v11366 * v11367;	// L17551
        int8_t v11369 = c191;	// L17552
        ap_int<17> v11370 = v11369;	// L17553
        ap_int<17> v11371 = v11368;	// L17554
        ap_int<17> v11372 = v11370 + v11371;	// L17555
        int8_t v11373 = v11372;	// L17556
        c191 = v11373;	// L17557
        int8_t v11374 = b207;	// L17558
        v11353.write(v11374);	// L17559
      }
      ap_uint<128> packed_tmp191;	// L17561
      ap_uint<128> v11376 = v11354.read();	// L17562
      packed_tmp191 = v11376;	// L17563
      ap_uint<128> packed_c191;	// L17564
      packed_c191 = 0;	// L17565
      l_S_m_1_m191: for (int m191 = 0; m191 < 16; m191++) {	// L17566
        ap_int<33> v11379 = m191;	// L17567
        bool v11380 = v11379 == 11;	// L17568
        if (v11380) {	// L17569
          int8_t v11381 = c191;	// L17570
          ap_int<128> v11382 = packed_c191;	// L17571
          int64_t v11383 = m191;	// L17572
          int64_t v11384 = v11383 * 8;	// L17573
          ap_int<34> v11385 = m191;	// L17574
          ap_int<34> v11386 = v11385 + 1;	// L17575
          ap_int<66> v11387 = v11386;	// L17576
          ap_int<66> v11388 = v11387 * 8;	// L17577
          ap_int<66> v11389 = v11388 - 1;	// L17578
          int v11390 = v11384;	// L17579
          int v11391 = v11389;	// L17580
          ap_int<128> v11392;
          ap_int<128> v11392_tmp = v11382;
          v11392_tmp(v11391, v11390) = v11381;
          v11392 = v11392_tmp;	// L17581
          packed_c191 = v11392;	// L17582
        } else {
          ap_int<128> v11393 = packed_tmp191;	// L17584
          int64_t v11394 = m191;	// L17585
          int64_t v11395 = v11394 * 8;	// L17586
          ap_int<34> v11396 = m191;	// L17587
          ap_int<34> v11397 = v11396 + 1;	// L17588
          ap_int<66> v11398 = v11397;	// L17589
          ap_int<66> v11399 = v11398 * 8;	// L17590
          ap_int<66> v11400 = v11399 - 1;	// L17591
          int v11401 = v11395;	// L17592
          int v11402 = v11400;	// L17593
          int8_t v11403;
          ap_int<128> v11403_tmp = v11393;
          v11403 = v11403_tmp(v11402, v11401);	// L17594
          ap_int<128> v11404 = packed_c191;	// L17595
          ap_int<128> v11405;
          ap_int<128> v11405_tmp = v11404;
          v11405_tmp(v11402, v11401) = v11403;
          v11405 = v11405_tmp;	// L17605
          packed_c191 = v11405;	// L17606
        }
      }
      ap_int<128> v11406 = packed_c191;	// L17609
      v11355.write(v11406);	// L17610
    }
  }
}

void gemm_13_0(
  hls::stream< ap_uint<128> >& v11407,
  hls::stream< int8_t >& v11408,
  hls::stream< ap_uint<128> >& v11409
) {	// L17615
  l_S_mt_nt_0_mt223: for (int mt223 = 0; mt223 < 4; mt223++) {	// L17618
    l_nt223: for (int nt223 = 0; nt223 < 4; nt223++) {	// L17619
      l_S_k_0_k223: for (int k223 = 0; k223 < 64; k223++) {	// L17620
        ap_uint<128> v11413 = v11407.read();	// L17621
        ap_uint<128> a204;	// L17622
        a204 = v11413;	// L17623
        ap_int<128> v11415 = a204;	// L17624
        int8_t v11416;
        ap_int<128> v11416_tmp = v11415;
        v11416 = v11416_tmp(103, 96);	// L17625
        v11408.write(v11416);	// L17626
        ap_int<128> v11417 = a204;	// L17627
        v11409.write(v11417);	// L17628
      }
    }
  }
}

void gemm_13_1(
  hls::stream< int8_t >& v11418,
  hls::stream< int8_t >& v11419,
  hls::stream< int8_t >& v11420,
  hls::stream< int8_t >& v11421,
  hls::stream< ap_uint<128> >& v11422,
  hls::stream< ap_uint<128> >& v11423
) {	// L17634
  l_S_mt_nt_0_mt224: for (int mt224 = 0; mt224 < 4; mt224++) {	// L17642
    l_nt224: for (int nt224 = 0; nt224 < 4; nt224++) {	// L17643
      int8_t c192;	// L17644
      c192 = 0;	// L17645
      l_S_k_0_k224: for (int k224 = 0; k224 < 64; k224++) {	// L17646
        int8_t v11428 = v11418.read();	// L17647
        int8_t a205;	// L17648
        a205 = v11428;	// L17649
        int8_t v11430 = v11419.read();	// L17650
        int8_t b208;	// L17651
        b208 = v11430;	// L17652
        int8_t v11432 = a205;	// L17653
        int8_t v11433 = b208;	// L17654
        int16_t v11434 = v11432;	// L17655
        int16_t v11435 = v11433;	// L17656
        int16_t v11436 = v11434 * v11435;	// L17657
        int8_t v11437 = c192;	// L17658
        ap_int<17> v11438 = v11437;	// L17659
        ap_int<17> v11439 = v11436;	// L17660
        ap_int<17> v11440 = v11438 + v11439;	// L17661
        int8_t v11441 = v11440;	// L17662
        c192 = v11441;	// L17663
        int8_t v11442 = a205;	// L17664
        v11420.write(v11442);	// L17665
        int8_t v11443 = b208;	// L17666
        v11421.write(v11443);	// L17667
      }
      ap_uint<128> packed_tmp192;	// L17669
      ap_uint<128> v11445 = v11422.read();	// L17670
      packed_tmp192 = v11445;	// L17671
      ap_uint<128> packed_c192;	// L17672
      packed_c192 = 0;	// L17673
      l_S_m_1_m192: for (int m192 = 0; m192 < 16; m192++) {	// L17674
        ap_int<33> v11448 = m192;	// L17675
        bool v11449 = v11448 == 12;	// L17676
        if (v11449) {	// L17677
          int8_t v11450 = c192;	// L17678
          ap_int<128> v11451 = packed_c192;	// L17679
          int64_t v11452 = m192;	// L17680
          int64_t v11453 = v11452 * 8;	// L17681
          ap_int<34> v11454 = m192;	// L17682
          ap_int<34> v11455 = v11454 + 1;	// L17683
          ap_int<66> v11456 = v11455;	// L17684
          ap_int<66> v11457 = v11456 * 8;	// L17685
          ap_int<66> v11458 = v11457 - 1;	// L17686
          int v11459 = v11453;	// L17687
          int v11460 = v11458;	// L17688
          ap_int<128> v11461;
          ap_int<128> v11461_tmp = v11451;
          v11461_tmp(v11460, v11459) = v11450;
          v11461 = v11461_tmp;	// L17689
          packed_c192 = v11461;	// L17690
        } else {
          ap_int<128> v11462 = packed_tmp192;	// L17692
          int64_t v11463 = m192;	// L17693
          int64_t v11464 = v11463 * 8;	// L17694
          ap_int<34> v11465 = m192;	// L17695
          ap_int<34> v11466 = v11465 + 1;	// L17696
          ap_int<66> v11467 = v11466;	// L17697
          ap_int<66> v11468 = v11467 * 8;	// L17698
          ap_int<66> v11469 = v11468 - 1;	// L17699
          int v11470 = v11464;	// L17700
          int v11471 = v11469;	// L17701
          int8_t v11472;
          ap_int<128> v11472_tmp = v11462;
          v11472 = v11472_tmp(v11471, v11470);	// L17702
          ap_int<128> v11473 = packed_c192;	// L17703
          ap_int<128> v11474;
          ap_int<128> v11474_tmp = v11473;
          v11474_tmp(v11471, v11470) = v11472;
          v11474 = v11474_tmp;	// L17713
          packed_c192 = v11474;	// L17714
        }
      }
      ap_int<128> v11475 = packed_c192;	// L17717
      v11423.write(v11475);	// L17718
    }
  }
}

void gemm_13_2(
  hls::stream< int8_t >& v11476,
  hls::stream< int8_t >& v11477,
  hls::stream< int8_t >& v11478,
  hls::stream< int8_t >& v11479,
  hls::stream< ap_uint<128> >& v11480,
  hls::stream< ap_uint<128> >& v11481
) {	// L17723
  l_S_mt_nt_0_mt225: for (int mt225 = 0; mt225 < 4; mt225++) {	// L17731
    l_nt225: for (int nt225 = 0; nt225 < 4; nt225++) {	// L17732
      int8_t c193;	// L17733
      c193 = 0;	// L17734
      l_S_k_0_k225: for (int k225 = 0; k225 < 64; k225++) {	// L17735
        int8_t v11486 = v11476.read();	// L17736
        int8_t a206;	// L17737
        a206 = v11486;	// L17738
        int8_t v11488 = v11477.read();	// L17739
        int8_t b209;	// L17740
        b209 = v11488;	// L17741
        int8_t v11490 = a206;	// L17742
        int8_t v11491 = b209;	// L17743
        int16_t v11492 = v11490;	// L17744
        int16_t v11493 = v11491;	// L17745
        int16_t v11494 = v11492 * v11493;	// L17746
        int8_t v11495 = c193;	// L17747
        ap_int<17> v11496 = v11495;	// L17748
        ap_int<17> v11497 = v11494;	// L17749
        ap_int<17> v11498 = v11496 + v11497;	// L17750
        int8_t v11499 = v11498;	// L17751
        c193 = v11499;	// L17752
        int8_t v11500 = a206;	// L17753
        v11478.write(v11500);	// L17754
        int8_t v11501 = b209;	// L17755
        v11479.write(v11501);	// L17756
      }
      ap_uint<128> packed_tmp193;	// L17758
      ap_uint<128> v11503 = v11480.read();	// L17759
      packed_tmp193 = v11503;	// L17760
      ap_uint<128> packed_c193;	// L17761
      packed_c193 = 0;	// L17762
      l_S_m_1_m193: for (int m193 = 0; m193 < 16; m193++) {	// L17763
        ap_int<33> v11506 = m193;	// L17764
        bool v11507 = v11506 == 12;	// L17765
        if (v11507) {	// L17766
          int8_t v11508 = c193;	// L17767
          ap_int<128> v11509 = packed_c193;	// L17768
          int64_t v11510 = m193;	// L17769
          int64_t v11511 = v11510 * 8;	// L17770
          ap_int<34> v11512 = m193;	// L17771
          ap_int<34> v11513 = v11512 + 1;	// L17772
          ap_int<66> v11514 = v11513;	// L17773
          ap_int<66> v11515 = v11514 * 8;	// L17774
          ap_int<66> v11516 = v11515 - 1;	// L17775
          int v11517 = v11511;	// L17776
          int v11518 = v11516;	// L17777
          ap_int<128> v11519;
          ap_int<128> v11519_tmp = v11509;
          v11519_tmp(v11518, v11517) = v11508;
          v11519 = v11519_tmp;	// L17778
          packed_c193 = v11519;	// L17779
        } else {
          ap_int<128> v11520 = packed_tmp193;	// L17781
          int64_t v11521 = m193;	// L17782
          int64_t v11522 = v11521 * 8;	// L17783
          ap_int<34> v11523 = m193;	// L17784
          ap_int<34> v11524 = v11523 + 1;	// L17785
          ap_int<66> v11525 = v11524;	// L17786
          ap_int<66> v11526 = v11525 * 8;	// L17787
          ap_int<66> v11527 = v11526 - 1;	// L17788
          int v11528 = v11522;	// L17789
          int v11529 = v11527;	// L17790
          int8_t v11530;
          ap_int<128> v11530_tmp = v11520;
          v11530 = v11530_tmp(v11529, v11528);	// L17791
          ap_int<128> v11531 = packed_c193;	// L17792
          ap_int<128> v11532;
          ap_int<128> v11532_tmp = v11531;
          v11532_tmp(v11529, v11528) = v11530;
          v11532 = v11532_tmp;	// L17802
          packed_c193 = v11532;	// L17803
        }
      }
      ap_int<128> v11533 = packed_c193;	// L17806
      v11481.write(v11533);	// L17807
    }
  }
}

void gemm_13_3(
  hls::stream< int8_t >& v11534,
  hls::stream< int8_t >& v11535,
  hls::stream< int8_t >& v11536,
  hls::stream< int8_t >& v11537,
  hls::stream< ap_uint<128> >& v11538,
  hls::stream< ap_uint<128> >& v11539
) {	// L17812
  l_S_mt_nt_0_mt226: for (int mt226 = 0; mt226 < 4; mt226++) {	// L17820
    l_nt226: for (int nt226 = 0; nt226 < 4; nt226++) {	// L17821
      int8_t c194;	// L17822
      c194 = 0;	// L17823
      l_S_k_0_k226: for (int k226 = 0; k226 < 64; k226++) {	// L17824
        int8_t v11544 = v11534.read();	// L17825
        int8_t a207;	// L17826
        a207 = v11544;	// L17827
        int8_t v11546 = v11535.read();	// L17828
        int8_t b210;	// L17829
        b210 = v11546;	// L17830
        int8_t v11548 = a207;	// L17831
        int8_t v11549 = b210;	// L17832
        int16_t v11550 = v11548;	// L17833
        int16_t v11551 = v11549;	// L17834
        int16_t v11552 = v11550 * v11551;	// L17835
        int8_t v11553 = c194;	// L17836
        ap_int<17> v11554 = v11553;	// L17837
        ap_int<17> v11555 = v11552;	// L17838
        ap_int<17> v11556 = v11554 + v11555;	// L17839
        int8_t v11557 = v11556;	// L17840
        c194 = v11557;	// L17841
        int8_t v11558 = a207;	// L17842
        v11536.write(v11558);	// L17843
        int8_t v11559 = b210;	// L17844
        v11537.write(v11559);	// L17845
      }
      ap_uint<128> packed_tmp194;	// L17847
      ap_uint<128> v11561 = v11538.read();	// L17848
      packed_tmp194 = v11561;	// L17849
      ap_uint<128> packed_c194;	// L17850
      packed_c194 = 0;	// L17851
      l_S_m_1_m194: for (int m194 = 0; m194 < 16; m194++) {	// L17852
        ap_int<33> v11564 = m194;	// L17853
        bool v11565 = v11564 == 12;	// L17854
        if (v11565) {	// L17855
          int8_t v11566 = c194;	// L17856
          ap_int<128> v11567 = packed_c194;	// L17857
          int64_t v11568 = m194;	// L17858
          int64_t v11569 = v11568 * 8;	// L17859
          ap_int<34> v11570 = m194;	// L17860
          ap_int<34> v11571 = v11570 + 1;	// L17861
          ap_int<66> v11572 = v11571;	// L17862
          ap_int<66> v11573 = v11572 * 8;	// L17863
          ap_int<66> v11574 = v11573 - 1;	// L17864
          int v11575 = v11569;	// L17865
          int v11576 = v11574;	// L17866
          ap_int<128> v11577;
          ap_int<128> v11577_tmp = v11567;
          v11577_tmp(v11576, v11575) = v11566;
          v11577 = v11577_tmp;	// L17867
          packed_c194 = v11577;	// L17868
        } else {
          ap_int<128> v11578 = packed_tmp194;	// L17870
          int64_t v11579 = m194;	// L17871
          int64_t v11580 = v11579 * 8;	// L17872
          ap_int<34> v11581 = m194;	// L17873
          ap_int<34> v11582 = v11581 + 1;	// L17874
          ap_int<66> v11583 = v11582;	// L17875
          ap_int<66> v11584 = v11583 * 8;	// L17876
          ap_int<66> v11585 = v11584 - 1;	// L17877
          int v11586 = v11580;	// L17878
          int v11587 = v11585;	// L17879
          int8_t v11588;
          ap_int<128> v11588_tmp = v11578;
          v11588 = v11588_tmp(v11587, v11586);	// L17880
          ap_int<128> v11589 = packed_c194;	// L17881
          ap_int<128> v11590;
          ap_int<128> v11590_tmp = v11589;
          v11590_tmp(v11587, v11586) = v11588;
          v11590 = v11590_tmp;	// L17891
          packed_c194 = v11590;	// L17892
        }
      }
      ap_int<128> v11591 = packed_c194;	// L17895
      v11539.write(v11591);	// L17896
    }
  }
}

void gemm_13_4(
  hls::stream< int8_t >& v11592,
  hls::stream< int8_t >& v11593,
  hls::stream< int8_t >& v11594,
  hls::stream< int8_t >& v11595,
  hls::stream< ap_uint<128> >& v11596,
  hls::stream< ap_uint<128> >& v11597
) {	// L17901
  l_S_mt_nt_0_mt227: for (int mt227 = 0; mt227 < 4; mt227++) {	// L17909
    l_nt227: for (int nt227 = 0; nt227 < 4; nt227++) {	// L17910
      int8_t c195;	// L17911
      c195 = 0;	// L17912
      l_S_k_0_k227: for (int k227 = 0; k227 < 64; k227++) {	// L17913
        int8_t v11602 = v11592.read();	// L17914
        int8_t a208;	// L17915
        a208 = v11602;	// L17916
        int8_t v11604 = v11593.read();	// L17917
        int8_t b211;	// L17918
        b211 = v11604;	// L17919
        int8_t v11606 = a208;	// L17920
        int8_t v11607 = b211;	// L17921
        int16_t v11608 = v11606;	// L17922
        int16_t v11609 = v11607;	// L17923
        int16_t v11610 = v11608 * v11609;	// L17924
        int8_t v11611 = c195;	// L17925
        ap_int<17> v11612 = v11611;	// L17926
        ap_int<17> v11613 = v11610;	// L17927
        ap_int<17> v11614 = v11612 + v11613;	// L17928
        int8_t v11615 = v11614;	// L17929
        c195 = v11615;	// L17930
        int8_t v11616 = a208;	// L17931
        v11594.write(v11616);	// L17932
        int8_t v11617 = b211;	// L17933
        v11595.write(v11617);	// L17934
      }
      ap_uint<128> packed_tmp195;	// L17936
      ap_uint<128> v11619 = v11596.read();	// L17937
      packed_tmp195 = v11619;	// L17938
      ap_uint<128> packed_c195;	// L17939
      packed_c195 = 0;	// L17940
      l_S_m_1_m195: for (int m195 = 0; m195 < 16; m195++) {	// L17941
        ap_int<33> v11622 = m195;	// L17942
        bool v11623 = v11622 == 12;	// L17943
        if (v11623) {	// L17944
          int8_t v11624 = c195;	// L17945
          ap_int<128> v11625 = packed_c195;	// L17946
          int64_t v11626 = m195;	// L17947
          int64_t v11627 = v11626 * 8;	// L17948
          ap_int<34> v11628 = m195;	// L17949
          ap_int<34> v11629 = v11628 + 1;	// L17950
          ap_int<66> v11630 = v11629;	// L17951
          ap_int<66> v11631 = v11630 * 8;	// L17952
          ap_int<66> v11632 = v11631 - 1;	// L17953
          int v11633 = v11627;	// L17954
          int v11634 = v11632;	// L17955
          ap_int<128> v11635;
          ap_int<128> v11635_tmp = v11625;
          v11635_tmp(v11634, v11633) = v11624;
          v11635 = v11635_tmp;	// L17956
          packed_c195 = v11635;	// L17957
        } else {
          ap_int<128> v11636 = packed_tmp195;	// L17959
          int64_t v11637 = m195;	// L17960
          int64_t v11638 = v11637 * 8;	// L17961
          ap_int<34> v11639 = m195;	// L17962
          ap_int<34> v11640 = v11639 + 1;	// L17963
          ap_int<66> v11641 = v11640;	// L17964
          ap_int<66> v11642 = v11641 * 8;	// L17965
          ap_int<66> v11643 = v11642 - 1;	// L17966
          int v11644 = v11638;	// L17967
          int v11645 = v11643;	// L17968
          int8_t v11646;
          ap_int<128> v11646_tmp = v11636;
          v11646 = v11646_tmp(v11645, v11644);	// L17969
          ap_int<128> v11647 = packed_c195;	// L17970
          ap_int<128> v11648;
          ap_int<128> v11648_tmp = v11647;
          v11648_tmp(v11645, v11644) = v11646;
          v11648 = v11648_tmp;	// L17980
          packed_c195 = v11648;	// L17981
        }
      }
      ap_int<128> v11649 = packed_c195;	// L17984
      v11597.write(v11649);	// L17985
    }
  }
}

void gemm_13_5(
  hls::stream< int8_t >& v11650,
  hls::stream< int8_t >& v11651,
  hls::stream< int8_t >& v11652,
  hls::stream< int8_t >& v11653,
  hls::stream< ap_uint<128> >& v11654,
  hls::stream< ap_uint<128> >& v11655
) {	// L17990
  l_S_mt_nt_0_mt228: for (int mt228 = 0; mt228 < 4; mt228++) {	// L17998
    l_nt228: for (int nt228 = 0; nt228 < 4; nt228++) {	// L17999
      int8_t c196;	// L18000
      c196 = 0;	// L18001
      l_S_k_0_k228: for (int k228 = 0; k228 < 64; k228++) {	// L18002
        int8_t v11660 = v11650.read();	// L18003
        int8_t a209;	// L18004
        a209 = v11660;	// L18005
        int8_t v11662 = v11651.read();	// L18006
        int8_t b212;	// L18007
        b212 = v11662;	// L18008
        int8_t v11664 = a209;	// L18009
        int8_t v11665 = b212;	// L18010
        int16_t v11666 = v11664;	// L18011
        int16_t v11667 = v11665;	// L18012
        int16_t v11668 = v11666 * v11667;	// L18013
        int8_t v11669 = c196;	// L18014
        ap_int<17> v11670 = v11669;	// L18015
        ap_int<17> v11671 = v11668;	// L18016
        ap_int<17> v11672 = v11670 + v11671;	// L18017
        int8_t v11673 = v11672;	// L18018
        c196 = v11673;	// L18019
        int8_t v11674 = a209;	// L18020
        v11652.write(v11674);	// L18021
        int8_t v11675 = b212;	// L18022
        v11653.write(v11675);	// L18023
      }
      ap_uint<128> packed_tmp196;	// L18025
      ap_uint<128> v11677 = v11654.read();	// L18026
      packed_tmp196 = v11677;	// L18027
      ap_uint<128> packed_c196;	// L18028
      packed_c196 = 0;	// L18029
      l_S_m_1_m196: for (int m196 = 0; m196 < 16; m196++) {	// L18030
        ap_int<33> v11680 = m196;	// L18031
        bool v11681 = v11680 == 12;	// L18032
        if (v11681) {	// L18033
          int8_t v11682 = c196;	// L18034
          ap_int<128> v11683 = packed_c196;	// L18035
          int64_t v11684 = m196;	// L18036
          int64_t v11685 = v11684 * 8;	// L18037
          ap_int<34> v11686 = m196;	// L18038
          ap_int<34> v11687 = v11686 + 1;	// L18039
          ap_int<66> v11688 = v11687;	// L18040
          ap_int<66> v11689 = v11688 * 8;	// L18041
          ap_int<66> v11690 = v11689 - 1;	// L18042
          int v11691 = v11685;	// L18043
          int v11692 = v11690;	// L18044
          ap_int<128> v11693;
          ap_int<128> v11693_tmp = v11683;
          v11693_tmp(v11692, v11691) = v11682;
          v11693 = v11693_tmp;	// L18045
          packed_c196 = v11693;	// L18046
        } else {
          ap_int<128> v11694 = packed_tmp196;	// L18048
          int64_t v11695 = m196;	// L18049
          int64_t v11696 = v11695 * 8;	// L18050
          ap_int<34> v11697 = m196;	// L18051
          ap_int<34> v11698 = v11697 + 1;	// L18052
          ap_int<66> v11699 = v11698;	// L18053
          ap_int<66> v11700 = v11699 * 8;	// L18054
          ap_int<66> v11701 = v11700 - 1;	// L18055
          int v11702 = v11696;	// L18056
          int v11703 = v11701;	// L18057
          int8_t v11704;
          ap_int<128> v11704_tmp = v11694;
          v11704 = v11704_tmp(v11703, v11702);	// L18058
          ap_int<128> v11705 = packed_c196;	// L18059
          ap_int<128> v11706;
          ap_int<128> v11706_tmp = v11705;
          v11706_tmp(v11703, v11702) = v11704;
          v11706 = v11706_tmp;	// L18069
          packed_c196 = v11706;	// L18070
        }
      }
      ap_int<128> v11707 = packed_c196;	// L18073
      v11655.write(v11707);	// L18074
    }
  }
}

void gemm_13_6(
  hls::stream< int8_t >& v11708,
  hls::stream< int8_t >& v11709,
  hls::stream< int8_t >& v11710,
  hls::stream< int8_t >& v11711,
  hls::stream< ap_uint<128> >& v11712,
  hls::stream< ap_uint<128> >& v11713
) {	// L18079
  l_S_mt_nt_0_mt229: for (int mt229 = 0; mt229 < 4; mt229++) {	// L18087
    l_nt229: for (int nt229 = 0; nt229 < 4; nt229++) {	// L18088
      int8_t c197;	// L18089
      c197 = 0;	// L18090
      l_S_k_0_k229: for (int k229 = 0; k229 < 64; k229++) {	// L18091
        int8_t v11718 = v11708.read();	// L18092
        int8_t a210;	// L18093
        a210 = v11718;	// L18094
        int8_t v11720 = v11709.read();	// L18095
        int8_t b213;	// L18096
        b213 = v11720;	// L18097
        int8_t v11722 = a210;	// L18098
        int8_t v11723 = b213;	// L18099
        int16_t v11724 = v11722;	// L18100
        int16_t v11725 = v11723;	// L18101
        int16_t v11726 = v11724 * v11725;	// L18102
        int8_t v11727 = c197;	// L18103
        ap_int<17> v11728 = v11727;	// L18104
        ap_int<17> v11729 = v11726;	// L18105
        ap_int<17> v11730 = v11728 + v11729;	// L18106
        int8_t v11731 = v11730;	// L18107
        c197 = v11731;	// L18108
        int8_t v11732 = a210;	// L18109
        v11710.write(v11732);	// L18110
        int8_t v11733 = b213;	// L18111
        v11711.write(v11733);	// L18112
      }
      ap_uint<128> packed_tmp197;	// L18114
      ap_uint<128> v11735 = v11712.read();	// L18115
      packed_tmp197 = v11735;	// L18116
      ap_uint<128> packed_c197;	// L18117
      packed_c197 = 0;	// L18118
      l_S_m_1_m197: for (int m197 = 0; m197 < 16; m197++) {	// L18119
        ap_int<33> v11738 = m197;	// L18120
        bool v11739 = v11738 == 12;	// L18121
        if (v11739) {	// L18122
          int8_t v11740 = c197;	// L18123
          ap_int<128> v11741 = packed_c197;	// L18124
          int64_t v11742 = m197;	// L18125
          int64_t v11743 = v11742 * 8;	// L18126
          ap_int<34> v11744 = m197;	// L18127
          ap_int<34> v11745 = v11744 + 1;	// L18128
          ap_int<66> v11746 = v11745;	// L18129
          ap_int<66> v11747 = v11746 * 8;	// L18130
          ap_int<66> v11748 = v11747 - 1;	// L18131
          int v11749 = v11743;	// L18132
          int v11750 = v11748;	// L18133
          ap_int<128> v11751;
          ap_int<128> v11751_tmp = v11741;
          v11751_tmp(v11750, v11749) = v11740;
          v11751 = v11751_tmp;	// L18134
          packed_c197 = v11751;	// L18135
        } else {
          ap_int<128> v11752 = packed_tmp197;	// L18137
          int64_t v11753 = m197;	// L18138
          int64_t v11754 = v11753 * 8;	// L18139
          ap_int<34> v11755 = m197;	// L18140
          ap_int<34> v11756 = v11755 + 1;	// L18141
          ap_int<66> v11757 = v11756;	// L18142
          ap_int<66> v11758 = v11757 * 8;	// L18143
          ap_int<66> v11759 = v11758 - 1;	// L18144
          int v11760 = v11754;	// L18145
          int v11761 = v11759;	// L18146
          int8_t v11762;
          ap_int<128> v11762_tmp = v11752;
          v11762 = v11762_tmp(v11761, v11760);	// L18147
          ap_int<128> v11763 = packed_c197;	// L18148
          ap_int<128> v11764;
          ap_int<128> v11764_tmp = v11763;
          v11764_tmp(v11761, v11760) = v11762;
          v11764 = v11764_tmp;	// L18158
          packed_c197 = v11764;	// L18159
        }
      }
      ap_int<128> v11765 = packed_c197;	// L18162
      v11713.write(v11765);	// L18163
    }
  }
}

void gemm_13_7(
  hls::stream< int8_t >& v11766,
  hls::stream< int8_t >& v11767,
  hls::stream< int8_t >& v11768,
  hls::stream< int8_t >& v11769,
  hls::stream< ap_uint<128> >& v11770,
  hls::stream< ap_uint<128> >& v11771
) {	// L18168
  l_S_mt_nt_0_mt230: for (int mt230 = 0; mt230 < 4; mt230++) {	// L18176
    l_nt230: for (int nt230 = 0; nt230 < 4; nt230++) {	// L18177
      int8_t c198;	// L18178
      c198 = 0;	// L18179
      l_S_k_0_k230: for (int k230 = 0; k230 < 64; k230++) {	// L18180
        int8_t v11776 = v11766.read();	// L18181
        int8_t a211;	// L18182
        a211 = v11776;	// L18183
        int8_t v11778 = v11767.read();	// L18184
        int8_t b214;	// L18185
        b214 = v11778;	// L18186
        int8_t v11780 = a211;	// L18187
        int8_t v11781 = b214;	// L18188
        int16_t v11782 = v11780;	// L18189
        int16_t v11783 = v11781;	// L18190
        int16_t v11784 = v11782 * v11783;	// L18191
        int8_t v11785 = c198;	// L18192
        ap_int<17> v11786 = v11785;	// L18193
        ap_int<17> v11787 = v11784;	// L18194
        ap_int<17> v11788 = v11786 + v11787;	// L18195
        int8_t v11789 = v11788;	// L18196
        c198 = v11789;	// L18197
        int8_t v11790 = a211;	// L18198
        v11768.write(v11790);	// L18199
        int8_t v11791 = b214;	// L18200
        v11769.write(v11791);	// L18201
      }
      ap_uint<128> packed_tmp198;	// L18203
      ap_uint<128> v11793 = v11770.read();	// L18204
      packed_tmp198 = v11793;	// L18205
      ap_uint<128> packed_c198;	// L18206
      packed_c198 = 0;	// L18207
      l_S_m_1_m198: for (int m198 = 0; m198 < 16; m198++) {	// L18208
        ap_int<33> v11796 = m198;	// L18209
        bool v11797 = v11796 == 12;	// L18210
        if (v11797) {	// L18211
          int8_t v11798 = c198;	// L18212
          ap_int<128> v11799 = packed_c198;	// L18213
          int64_t v11800 = m198;	// L18214
          int64_t v11801 = v11800 * 8;	// L18215
          ap_int<34> v11802 = m198;	// L18216
          ap_int<34> v11803 = v11802 + 1;	// L18217
          ap_int<66> v11804 = v11803;	// L18218
          ap_int<66> v11805 = v11804 * 8;	// L18219
          ap_int<66> v11806 = v11805 - 1;	// L18220
          int v11807 = v11801;	// L18221
          int v11808 = v11806;	// L18222
          ap_int<128> v11809;
          ap_int<128> v11809_tmp = v11799;
          v11809_tmp(v11808, v11807) = v11798;
          v11809 = v11809_tmp;	// L18223
          packed_c198 = v11809;	// L18224
        } else {
          ap_int<128> v11810 = packed_tmp198;	// L18226
          int64_t v11811 = m198;	// L18227
          int64_t v11812 = v11811 * 8;	// L18228
          ap_int<34> v11813 = m198;	// L18229
          ap_int<34> v11814 = v11813 + 1;	// L18230
          ap_int<66> v11815 = v11814;	// L18231
          ap_int<66> v11816 = v11815 * 8;	// L18232
          ap_int<66> v11817 = v11816 - 1;	// L18233
          int v11818 = v11812;	// L18234
          int v11819 = v11817;	// L18235
          int8_t v11820;
          ap_int<128> v11820_tmp = v11810;
          v11820 = v11820_tmp(v11819, v11818);	// L18236
          ap_int<128> v11821 = packed_c198;	// L18237
          ap_int<128> v11822;
          ap_int<128> v11822_tmp = v11821;
          v11822_tmp(v11819, v11818) = v11820;
          v11822 = v11822_tmp;	// L18247
          packed_c198 = v11822;	// L18248
        }
      }
      ap_int<128> v11823 = packed_c198;	// L18251
      v11771.write(v11823);	// L18252
    }
  }
}

void gemm_13_8(
  hls::stream< int8_t >& v11824,
  hls::stream< int8_t >& v11825,
  hls::stream< int8_t >& v11826,
  hls::stream< int8_t >& v11827,
  hls::stream< ap_uint<128> >& v11828,
  hls::stream< ap_uint<128> >& v11829
) {	// L18257
  l_S_mt_nt_0_mt231: for (int mt231 = 0; mt231 < 4; mt231++) {	// L18265
    l_nt231: for (int nt231 = 0; nt231 < 4; nt231++) {	// L18266
      int8_t c199;	// L18267
      c199 = 0;	// L18268
      l_S_k_0_k231: for (int k231 = 0; k231 < 64; k231++) {	// L18269
        int8_t v11834 = v11824.read();	// L18270
        int8_t a212;	// L18271
        a212 = v11834;	// L18272
        int8_t v11836 = v11825.read();	// L18273
        int8_t b215;	// L18274
        b215 = v11836;	// L18275
        int8_t v11838 = a212;	// L18276
        int8_t v11839 = b215;	// L18277
        int16_t v11840 = v11838;	// L18278
        int16_t v11841 = v11839;	// L18279
        int16_t v11842 = v11840 * v11841;	// L18280
        int8_t v11843 = c199;	// L18281
        ap_int<17> v11844 = v11843;	// L18282
        ap_int<17> v11845 = v11842;	// L18283
        ap_int<17> v11846 = v11844 + v11845;	// L18284
        int8_t v11847 = v11846;	// L18285
        c199 = v11847;	// L18286
        int8_t v11848 = a212;	// L18287
        v11826.write(v11848);	// L18288
        int8_t v11849 = b215;	// L18289
        v11827.write(v11849);	// L18290
      }
      ap_uint<128> packed_tmp199;	// L18292
      ap_uint<128> v11851 = v11828.read();	// L18293
      packed_tmp199 = v11851;	// L18294
      ap_uint<128> packed_c199;	// L18295
      packed_c199 = 0;	// L18296
      l_S_m_1_m199: for (int m199 = 0; m199 < 16; m199++) {	// L18297
        ap_int<33> v11854 = m199;	// L18298
        bool v11855 = v11854 == 12;	// L18299
        if (v11855) {	// L18300
          int8_t v11856 = c199;	// L18301
          ap_int<128> v11857 = packed_c199;	// L18302
          int64_t v11858 = m199;	// L18303
          int64_t v11859 = v11858 * 8;	// L18304
          ap_int<34> v11860 = m199;	// L18305
          ap_int<34> v11861 = v11860 + 1;	// L18306
          ap_int<66> v11862 = v11861;	// L18307
          ap_int<66> v11863 = v11862 * 8;	// L18308
          ap_int<66> v11864 = v11863 - 1;	// L18309
          int v11865 = v11859;	// L18310
          int v11866 = v11864;	// L18311
          ap_int<128> v11867;
          ap_int<128> v11867_tmp = v11857;
          v11867_tmp(v11866, v11865) = v11856;
          v11867 = v11867_tmp;	// L18312
          packed_c199 = v11867;	// L18313
        } else {
          ap_int<128> v11868 = packed_tmp199;	// L18315
          int64_t v11869 = m199;	// L18316
          int64_t v11870 = v11869 * 8;	// L18317
          ap_int<34> v11871 = m199;	// L18318
          ap_int<34> v11872 = v11871 + 1;	// L18319
          ap_int<66> v11873 = v11872;	// L18320
          ap_int<66> v11874 = v11873 * 8;	// L18321
          ap_int<66> v11875 = v11874 - 1;	// L18322
          int v11876 = v11870;	// L18323
          int v11877 = v11875;	// L18324
          int8_t v11878;
          ap_int<128> v11878_tmp = v11868;
          v11878 = v11878_tmp(v11877, v11876);	// L18325
          ap_int<128> v11879 = packed_c199;	// L18326
          ap_int<128> v11880;
          ap_int<128> v11880_tmp = v11879;
          v11880_tmp(v11877, v11876) = v11878;
          v11880 = v11880_tmp;	// L18336
          packed_c199 = v11880;	// L18337
        }
      }
      ap_int<128> v11881 = packed_c199;	// L18340
      v11829.write(v11881);	// L18341
    }
  }
}

void gemm_13_9(
  hls::stream< int8_t >& v11882,
  hls::stream< int8_t >& v11883,
  hls::stream< int8_t >& v11884,
  hls::stream< int8_t >& v11885,
  hls::stream< ap_uint<128> >& v11886,
  hls::stream< ap_uint<128> >& v11887
) {	// L18346
  l_S_mt_nt_0_mt232: for (int mt232 = 0; mt232 < 4; mt232++) {	// L18354
    l_nt232: for (int nt232 = 0; nt232 < 4; nt232++) {	// L18355
      int8_t c200;	// L18356
      c200 = 0;	// L18357
      l_S_k_0_k232: for (int k232 = 0; k232 < 64; k232++) {	// L18358
        int8_t v11892 = v11882.read();	// L18359
        int8_t a213;	// L18360
        a213 = v11892;	// L18361
        int8_t v11894 = v11883.read();	// L18362
        int8_t b216;	// L18363
        b216 = v11894;	// L18364
        int8_t v11896 = a213;	// L18365
        int8_t v11897 = b216;	// L18366
        int16_t v11898 = v11896;	// L18367
        int16_t v11899 = v11897;	// L18368
        int16_t v11900 = v11898 * v11899;	// L18369
        int8_t v11901 = c200;	// L18370
        ap_int<17> v11902 = v11901;	// L18371
        ap_int<17> v11903 = v11900;	// L18372
        ap_int<17> v11904 = v11902 + v11903;	// L18373
        int8_t v11905 = v11904;	// L18374
        c200 = v11905;	// L18375
        int8_t v11906 = a213;	// L18376
        v11884.write(v11906);	// L18377
        int8_t v11907 = b216;	// L18378
        v11885.write(v11907);	// L18379
      }
      ap_uint<128> packed_tmp200;	// L18381
      ap_uint<128> v11909 = v11886.read();	// L18382
      packed_tmp200 = v11909;	// L18383
      ap_uint<128> packed_c200;	// L18384
      packed_c200 = 0;	// L18385
      l_S_m_1_m200: for (int m200 = 0; m200 < 16; m200++) {	// L18386
        ap_int<33> v11912 = m200;	// L18387
        bool v11913 = v11912 == 12;	// L18388
        if (v11913) {	// L18389
          int8_t v11914 = c200;	// L18390
          ap_int<128> v11915 = packed_c200;	// L18391
          int64_t v11916 = m200;	// L18392
          int64_t v11917 = v11916 * 8;	// L18393
          ap_int<34> v11918 = m200;	// L18394
          ap_int<34> v11919 = v11918 + 1;	// L18395
          ap_int<66> v11920 = v11919;	// L18396
          ap_int<66> v11921 = v11920 * 8;	// L18397
          ap_int<66> v11922 = v11921 - 1;	// L18398
          int v11923 = v11917;	// L18399
          int v11924 = v11922;	// L18400
          ap_int<128> v11925;
          ap_int<128> v11925_tmp = v11915;
          v11925_tmp(v11924, v11923) = v11914;
          v11925 = v11925_tmp;	// L18401
          packed_c200 = v11925;	// L18402
        } else {
          ap_int<128> v11926 = packed_tmp200;	// L18404
          int64_t v11927 = m200;	// L18405
          int64_t v11928 = v11927 * 8;	// L18406
          ap_int<34> v11929 = m200;	// L18407
          ap_int<34> v11930 = v11929 + 1;	// L18408
          ap_int<66> v11931 = v11930;	// L18409
          ap_int<66> v11932 = v11931 * 8;	// L18410
          ap_int<66> v11933 = v11932 - 1;	// L18411
          int v11934 = v11928;	// L18412
          int v11935 = v11933;	// L18413
          int8_t v11936;
          ap_int<128> v11936_tmp = v11926;
          v11936 = v11936_tmp(v11935, v11934);	// L18414
          ap_int<128> v11937 = packed_c200;	// L18415
          ap_int<128> v11938;
          ap_int<128> v11938_tmp = v11937;
          v11938_tmp(v11935, v11934) = v11936;
          v11938 = v11938_tmp;	// L18425
          packed_c200 = v11938;	// L18426
        }
      }
      ap_int<128> v11939 = packed_c200;	// L18429
      v11887.write(v11939);	// L18430
    }
  }
}

void gemm_13_10(
  hls::stream< int8_t >& v11940,
  hls::stream< int8_t >& v11941,
  hls::stream< int8_t >& v11942,
  hls::stream< int8_t >& v11943,
  hls::stream< ap_uint<128> >& v11944,
  hls::stream< ap_uint<128> >& v11945
) {	// L18435
  l_S_mt_nt_0_mt233: for (int mt233 = 0; mt233 < 4; mt233++) {	// L18443
    l_nt233: for (int nt233 = 0; nt233 < 4; nt233++) {	// L18444
      int8_t c201;	// L18445
      c201 = 0;	// L18446
      l_S_k_0_k233: for (int k233 = 0; k233 < 64; k233++) {	// L18447
        int8_t v11950 = v11940.read();	// L18448
        int8_t a214;	// L18449
        a214 = v11950;	// L18450
        int8_t v11952 = v11941.read();	// L18451
        int8_t b217;	// L18452
        b217 = v11952;	// L18453
        int8_t v11954 = a214;	// L18454
        int8_t v11955 = b217;	// L18455
        int16_t v11956 = v11954;	// L18456
        int16_t v11957 = v11955;	// L18457
        int16_t v11958 = v11956 * v11957;	// L18458
        int8_t v11959 = c201;	// L18459
        ap_int<17> v11960 = v11959;	// L18460
        ap_int<17> v11961 = v11958;	// L18461
        ap_int<17> v11962 = v11960 + v11961;	// L18462
        int8_t v11963 = v11962;	// L18463
        c201 = v11963;	// L18464
        int8_t v11964 = a214;	// L18465
        v11942.write(v11964);	// L18466
        int8_t v11965 = b217;	// L18467
        v11943.write(v11965);	// L18468
      }
      ap_uint<128> packed_tmp201;	// L18470
      ap_uint<128> v11967 = v11944.read();	// L18471
      packed_tmp201 = v11967;	// L18472
      ap_uint<128> packed_c201;	// L18473
      packed_c201 = 0;	// L18474
      l_S_m_1_m201: for (int m201 = 0; m201 < 16; m201++) {	// L18475
        ap_int<33> v11970 = m201;	// L18476
        bool v11971 = v11970 == 12;	// L18477
        if (v11971) {	// L18478
          int8_t v11972 = c201;	// L18479
          ap_int<128> v11973 = packed_c201;	// L18480
          int64_t v11974 = m201;	// L18481
          int64_t v11975 = v11974 * 8;	// L18482
          ap_int<34> v11976 = m201;	// L18483
          ap_int<34> v11977 = v11976 + 1;	// L18484
          ap_int<66> v11978 = v11977;	// L18485
          ap_int<66> v11979 = v11978 * 8;	// L18486
          ap_int<66> v11980 = v11979 - 1;	// L18487
          int v11981 = v11975;	// L18488
          int v11982 = v11980;	// L18489
          ap_int<128> v11983;
          ap_int<128> v11983_tmp = v11973;
          v11983_tmp(v11982, v11981) = v11972;
          v11983 = v11983_tmp;	// L18490
          packed_c201 = v11983;	// L18491
        } else {
          ap_int<128> v11984 = packed_tmp201;	// L18493
          int64_t v11985 = m201;	// L18494
          int64_t v11986 = v11985 * 8;	// L18495
          ap_int<34> v11987 = m201;	// L18496
          ap_int<34> v11988 = v11987 + 1;	// L18497
          ap_int<66> v11989 = v11988;	// L18498
          ap_int<66> v11990 = v11989 * 8;	// L18499
          ap_int<66> v11991 = v11990 - 1;	// L18500
          int v11992 = v11986;	// L18501
          int v11993 = v11991;	// L18502
          int8_t v11994;
          ap_int<128> v11994_tmp = v11984;
          v11994 = v11994_tmp(v11993, v11992);	// L18503
          ap_int<128> v11995 = packed_c201;	// L18504
          ap_int<128> v11996;
          ap_int<128> v11996_tmp = v11995;
          v11996_tmp(v11993, v11992) = v11994;
          v11996 = v11996_tmp;	// L18514
          packed_c201 = v11996;	// L18515
        }
      }
      ap_int<128> v11997 = packed_c201;	// L18518
      v11945.write(v11997);	// L18519
    }
  }
}

void gemm_13_11(
  hls::stream< int8_t >& v11998,
  hls::stream< int8_t >& v11999,
  hls::stream< int8_t >& v12000,
  hls::stream< int8_t >& v12001,
  hls::stream< ap_uint<128> >& v12002,
  hls::stream< ap_uint<128> >& v12003
) {	// L18524
  l_S_mt_nt_0_mt234: for (int mt234 = 0; mt234 < 4; mt234++) {	// L18532
    l_nt234: for (int nt234 = 0; nt234 < 4; nt234++) {	// L18533
      int8_t c202;	// L18534
      c202 = 0;	// L18535
      l_S_k_0_k234: for (int k234 = 0; k234 < 64; k234++) {	// L18536
        int8_t v12008 = v11998.read();	// L18537
        int8_t a215;	// L18538
        a215 = v12008;	// L18539
        int8_t v12010 = v11999.read();	// L18540
        int8_t b218;	// L18541
        b218 = v12010;	// L18542
        int8_t v12012 = a215;	// L18543
        int8_t v12013 = b218;	// L18544
        int16_t v12014 = v12012;	// L18545
        int16_t v12015 = v12013;	// L18546
        int16_t v12016 = v12014 * v12015;	// L18547
        int8_t v12017 = c202;	// L18548
        ap_int<17> v12018 = v12017;	// L18549
        ap_int<17> v12019 = v12016;	// L18550
        ap_int<17> v12020 = v12018 + v12019;	// L18551
        int8_t v12021 = v12020;	// L18552
        c202 = v12021;	// L18553
        int8_t v12022 = a215;	// L18554
        v12000.write(v12022);	// L18555
        int8_t v12023 = b218;	// L18556
        v12001.write(v12023);	// L18557
      }
      ap_uint<128> packed_tmp202;	// L18559
      ap_uint<128> v12025 = v12002.read();	// L18560
      packed_tmp202 = v12025;	// L18561
      ap_uint<128> packed_c202;	// L18562
      packed_c202 = 0;	// L18563
      l_S_m_1_m202: for (int m202 = 0; m202 < 16; m202++) {	// L18564
        ap_int<33> v12028 = m202;	// L18565
        bool v12029 = v12028 == 12;	// L18566
        if (v12029) {	// L18567
          int8_t v12030 = c202;	// L18568
          ap_int<128> v12031 = packed_c202;	// L18569
          int64_t v12032 = m202;	// L18570
          int64_t v12033 = v12032 * 8;	// L18571
          ap_int<34> v12034 = m202;	// L18572
          ap_int<34> v12035 = v12034 + 1;	// L18573
          ap_int<66> v12036 = v12035;	// L18574
          ap_int<66> v12037 = v12036 * 8;	// L18575
          ap_int<66> v12038 = v12037 - 1;	// L18576
          int v12039 = v12033;	// L18577
          int v12040 = v12038;	// L18578
          ap_int<128> v12041;
          ap_int<128> v12041_tmp = v12031;
          v12041_tmp(v12040, v12039) = v12030;
          v12041 = v12041_tmp;	// L18579
          packed_c202 = v12041;	// L18580
        } else {
          ap_int<128> v12042 = packed_tmp202;	// L18582
          int64_t v12043 = m202;	// L18583
          int64_t v12044 = v12043 * 8;	// L18584
          ap_int<34> v12045 = m202;	// L18585
          ap_int<34> v12046 = v12045 + 1;	// L18586
          ap_int<66> v12047 = v12046;	// L18587
          ap_int<66> v12048 = v12047 * 8;	// L18588
          ap_int<66> v12049 = v12048 - 1;	// L18589
          int v12050 = v12044;	// L18590
          int v12051 = v12049;	// L18591
          int8_t v12052;
          ap_int<128> v12052_tmp = v12042;
          v12052 = v12052_tmp(v12051, v12050);	// L18592
          ap_int<128> v12053 = packed_c202;	// L18593
          ap_int<128> v12054;
          ap_int<128> v12054_tmp = v12053;
          v12054_tmp(v12051, v12050) = v12052;
          v12054 = v12054_tmp;	// L18603
          packed_c202 = v12054;	// L18604
        }
      }
      ap_int<128> v12055 = packed_c202;	// L18607
      v12003.write(v12055);	// L18608
    }
  }
}

void gemm_13_12(
  hls::stream< int8_t >& v12056,
  hls::stream< int8_t >& v12057,
  hls::stream< int8_t >& v12058,
  hls::stream< int8_t >& v12059,
  hls::stream< ap_uint<128> >& v12060,
  hls::stream< ap_uint<128> >& v12061
) {	// L18613
  l_S_mt_nt_0_mt235: for (int mt235 = 0; mt235 < 4; mt235++) {	// L18621
    l_nt235: for (int nt235 = 0; nt235 < 4; nt235++) {	// L18622
      int8_t c203;	// L18623
      c203 = 0;	// L18624
      l_S_k_0_k235: for (int k235 = 0; k235 < 64; k235++) {	// L18625
        int8_t v12066 = v12056.read();	// L18626
        int8_t a216;	// L18627
        a216 = v12066;	// L18628
        int8_t v12068 = v12057.read();	// L18629
        int8_t b219;	// L18630
        b219 = v12068;	// L18631
        int8_t v12070 = a216;	// L18632
        int8_t v12071 = b219;	// L18633
        int16_t v12072 = v12070;	// L18634
        int16_t v12073 = v12071;	// L18635
        int16_t v12074 = v12072 * v12073;	// L18636
        int8_t v12075 = c203;	// L18637
        ap_int<17> v12076 = v12075;	// L18638
        ap_int<17> v12077 = v12074;	// L18639
        ap_int<17> v12078 = v12076 + v12077;	// L18640
        int8_t v12079 = v12078;	// L18641
        c203 = v12079;	// L18642
        int8_t v12080 = a216;	// L18643
        v12058.write(v12080);	// L18644
        int8_t v12081 = b219;	// L18645
        v12059.write(v12081);	// L18646
      }
      ap_uint<128> packed_tmp203;	// L18648
      ap_uint<128> v12083 = v12060.read();	// L18649
      packed_tmp203 = v12083;	// L18650
      ap_uint<128> packed_c203;	// L18651
      packed_c203 = 0;	// L18652
      l_S_m_1_m203: for (int m203 = 0; m203 < 16; m203++) {	// L18653
        ap_int<33> v12086 = m203;	// L18654
        bool v12087 = v12086 == 12;	// L18655
        if (v12087) {	// L18656
          int8_t v12088 = c203;	// L18657
          ap_int<128> v12089 = packed_c203;	// L18658
          int64_t v12090 = m203;	// L18659
          int64_t v12091 = v12090 * 8;	// L18660
          ap_int<34> v12092 = m203;	// L18661
          ap_int<34> v12093 = v12092 + 1;	// L18662
          ap_int<66> v12094 = v12093;	// L18663
          ap_int<66> v12095 = v12094 * 8;	// L18664
          ap_int<66> v12096 = v12095 - 1;	// L18665
          int v12097 = v12091;	// L18666
          int v12098 = v12096;	// L18667
          ap_int<128> v12099;
          ap_int<128> v12099_tmp = v12089;
          v12099_tmp(v12098, v12097) = v12088;
          v12099 = v12099_tmp;	// L18668
          packed_c203 = v12099;	// L18669
        } else {
          ap_int<128> v12100 = packed_tmp203;	// L18671
          int64_t v12101 = m203;	// L18672
          int64_t v12102 = v12101 * 8;	// L18673
          ap_int<34> v12103 = m203;	// L18674
          ap_int<34> v12104 = v12103 + 1;	// L18675
          ap_int<66> v12105 = v12104;	// L18676
          ap_int<66> v12106 = v12105 * 8;	// L18677
          ap_int<66> v12107 = v12106 - 1;	// L18678
          int v12108 = v12102;	// L18679
          int v12109 = v12107;	// L18680
          int8_t v12110;
          ap_int<128> v12110_tmp = v12100;
          v12110 = v12110_tmp(v12109, v12108);	// L18681
          ap_int<128> v12111 = packed_c203;	// L18682
          ap_int<128> v12112;
          ap_int<128> v12112_tmp = v12111;
          v12112_tmp(v12109, v12108) = v12110;
          v12112 = v12112_tmp;	// L18692
          packed_c203 = v12112;	// L18693
        }
      }
      ap_int<128> v12113 = packed_c203;	// L18696
      v12061.write(v12113);	// L18697
    }
  }
}

void gemm_13_13(
  hls::stream< int8_t >& v12114,
  hls::stream< int8_t >& v12115,
  hls::stream< int8_t >& v12116,
  hls::stream< int8_t >& v12117,
  hls::stream< ap_uint<128> >& v12118,
  hls::stream< ap_uint<128> >& v12119
) {	// L18702
  l_S_mt_nt_0_mt236: for (int mt236 = 0; mt236 < 4; mt236++) {	// L18710
    l_nt236: for (int nt236 = 0; nt236 < 4; nt236++) {	// L18711
      int8_t c204;	// L18712
      c204 = 0;	// L18713
      l_S_k_0_k236: for (int k236 = 0; k236 < 64; k236++) {	// L18714
        int8_t v12124 = v12114.read();	// L18715
        int8_t a217;	// L18716
        a217 = v12124;	// L18717
        int8_t v12126 = v12115.read();	// L18718
        int8_t b220;	// L18719
        b220 = v12126;	// L18720
        int8_t v12128 = a217;	// L18721
        int8_t v12129 = b220;	// L18722
        int16_t v12130 = v12128;	// L18723
        int16_t v12131 = v12129;	// L18724
        int16_t v12132 = v12130 * v12131;	// L18725
        int8_t v12133 = c204;	// L18726
        ap_int<17> v12134 = v12133;	// L18727
        ap_int<17> v12135 = v12132;	// L18728
        ap_int<17> v12136 = v12134 + v12135;	// L18729
        int8_t v12137 = v12136;	// L18730
        c204 = v12137;	// L18731
        int8_t v12138 = a217;	// L18732
        v12116.write(v12138);	// L18733
        int8_t v12139 = b220;	// L18734
        v12117.write(v12139);	// L18735
      }
      ap_uint<128> packed_tmp204;	// L18737
      ap_uint<128> v12141 = v12118.read();	// L18738
      packed_tmp204 = v12141;	// L18739
      ap_uint<128> packed_c204;	// L18740
      packed_c204 = 0;	// L18741
      l_S_m_1_m204: for (int m204 = 0; m204 < 16; m204++) {	// L18742
        ap_int<33> v12144 = m204;	// L18743
        bool v12145 = v12144 == 12;	// L18744
        if (v12145) {	// L18745
          int8_t v12146 = c204;	// L18746
          ap_int<128> v12147 = packed_c204;	// L18747
          int64_t v12148 = m204;	// L18748
          int64_t v12149 = v12148 * 8;	// L18749
          ap_int<34> v12150 = m204;	// L18750
          ap_int<34> v12151 = v12150 + 1;	// L18751
          ap_int<66> v12152 = v12151;	// L18752
          ap_int<66> v12153 = v12152 * 8;	// L18753
          ap_int<66> v12154 = v12153 - 1;	// L18754
          int v12155 = v12149;	// L18755
          int v12156 = v12154;	// L18756
          ap_int<128> v12157;
          ap_int<128> v12157_tmp = v12147;
          v12157_tmp(v12156, v12155) = v12146;
          v12157 = v12157_tmp;	// L18757
          packed_c204 = v12157;	// L18758
        } else {
          ap_int<128> v12158 = packed_tmp204;	// L18760
          int64_t v12159 = m204;	// L18761
          int64_t v12160 = v12159 * 8;	// L18762
          ap_int<34> v12161 = m204;	// L18763
          ap_int<34> v12162 = v12161 + 1;	// L18764
          ap_int<66> v12163 = v12162;	// L18765
          ap_int<66> v12164 = v12163 * 8;	// L18766
          ap_int<66> v12165 = v12164 - 1;	// L18767
          int v12166 = v12160;	// L18768
          int v12167 = v12165;	// L18769
          int8_t v12168;
          ap_int<128> v12168_tmp = v12158;
          v12168 = v12168_tmp(v12167, v12166);	// L18770
          ap_int<128> v12169 = packed_c204;	// L18771
          ap_int<128> v12170;
          ap_int<128> v12170_tmp = v12169;
          v12170_tmp(v12167, v12166) = v12168;
          v12170 = v12170_tmp;	// L18781
          packed_c204 = v12170;	// L18782
        }
      }
      ap_int<128> v12171 = packed_c204;	// L18785
      v12119.write(v12171);	// L18786
    }
  }
}

void gemm_13_14(
  hls::stream< int8_t >& v12172,
  hls::stream< int8_t >& v12173,
  hls::stream< int8_t >& v12174,
  hls::stream< int8_t >& v12175,
  hls::stream< ap_uint<128> >& v12176,
  hls::stream< ap_uint<128> >& v12177
) {	// L18791
  l_S_mt_nt_0_mt237: for (int mt237 = 0; mt237 < 4; mt237++) {	// L18799
    l_nt237: for (int nt237 = 0; nt237 < 4; nt237++) {	// L18800
      int8_t c205;	// L18801
      c205 = 0;	// L18802
      l_S_k_0_k237: for (int k237 = 0; k237 < 64; k237++) {	// L18803
        int8_t v12182 = v12172.read();	// L18804
        int8_t a218;	// L18805
        a218 = v12182;	// L18806
        int8_t v12184 = v12173.read();	// L18807
        int8_t b221;	// L18808
        b221 = v12184;	// L18809
        int8_t v12186 = a218;	// L18810
        int8_t v12187 = b221;	// L18811
        int16_t v12188 = v12186;	// L18812
        int16_t v12189 = v12187;	// L18813
        int16_t v12190 = v12188 * v12189;	// L18814
        int8_t v12191 = c205;	// L18815
        ap_int<17> v12192 = v12191;	// L18816
        ap_int<17> v12193 = v12190;	// L18817
        ap_int<17> v12194 = v12192 + v12193;	// L18818
        int8_t v12195 = v12194;	// L18819
        c205 = v12195;	// L18820
        int8_t v12196 = a218;	// L18821
        v12174.write(v12196);	// L18822
        int8_t v12197 = b221;	// L18823
        v12175.write(v12197);	// L18824
      }
      ap_uint<128> packed_tmp205;	// L18826
      ap_uint<128> v12199 = v12176.read();	// L18827
      packed_tmp205 = v12199;	// L18828
      ap_uint<128> packed_c205;	// L18829
      packed_c205 = 0;	// L18830
      l_S_m_1_m205: for (int m205 = 0; m205 < 16; m205++) {	// L18831
        ap_int<33> v12202 = m205;	// L18832
        bool v12203 = v12202 == 12;	// L18833
        if (v12203) {	// L18834
          int8_t v12204 = c205;	// L18835
          ap_int<128> v12205 = packed_c205;	// L18836
          int64_t v12206 = m205;	// L18837
          int64_t v12207 = v12206 * 8;	// L18838
          ap_int<34> v12208 = m205;	// L18839
          ap_int<34> v12209 = v12208 + 1;	// L18840
          ap_int<66> v12210 = v12209;	// L18841
          ap_int<66> v12211 = v12210 * 8;	// L18842
          ap_int<66> v12212 = v12211 - 1;	// L18843
          int v12213 = v12207;	// L18844
          int v12214 = v12212;	// L18845
          ap_int<128> v12215;
          ap_int<128> v12215_tmp = v12205;
          v12215_tmp(v12214, v12213) = v12204;
          v12215 = v12215_tmp;	// L18846
          packed_c205 = v12215;	// L18847
        } else {
          ap_int<128> v12216 = packed_tmp205;	// L18849
          int64_t v12217 = m205;	// L18850
          int64_t v12218 = v12217 * 8;	// L18851
          ap_int<34> v12219 = m205;	// L18852
          ap_int<34> v12220 = v12219 + 1;	// L18853
          ap_int<66> v12221 = v12220;	// L18854
          ap_int<66> v12222 = v12221 * 8;	// L18855
          ap_int<66> v12223 = v12222 - 1;	// L18856
          int v12224 = v12218;	// L18857
          int v12225 = v12223;	// L18858
          int8_t v12226;
          ap_int<128> v12226_tmp = v12216;
          v12226 = v12226_tmp(v12225, v12224);	// L18859
          ap_int<128> v12227 = packed_c205;	// L18860
          ap_int<128> v12228;
          ap_int<128> v12228_tmp = v12227;
          v12228_tmp(v12225, v12224) = v12226;
          v12228 = v12228_tmp;	// L18870
          packed_c205 = v12228;	// L18871
        }
      }
      ap_int<128> v12229 = packed_c205;	// L18874
      v12177.write(v12229);	// L18875
    }
  }
}

void gemm_13_15(
  hls::stream< int8_t >& v12230,
  hls::stream< int8_t >& v12231,
  hls::stream< int8_t >& v12232,
  hls::stream< int8_t >& v12233,
  hls::stream< ap_uint<128> >& v12234,
  hls::stream< ap_uint<128> >& v12235
) {	// L18880
  l_S_mt_nt_0_mt238: for (int mt238 = 0; mt238 < 4; mt238++) {	// L18888
    l_nt238: for (int nt238 = 0; nt238 < 4; nt238++) {	// L18889
      int8_t c206;	// L18890
      c206 = 0;	// L18891
      l_S_k_0_k238: for (int k238 = 0; k238 < 64; k238++) {	// L18892
        int8_t v12240 = v12230.read();	// L18893
        int8_t a219;	// L18894
        a219 = v12240;	// L18895
        int8_t v12242 = v12231.read();	// L18896
        int8_t b222;	// L18897
        b222 = v12242;	// L18898
        int8_t v12244 = a219;	// L18899
        int8_t v12245 = b222;	// L18900
        int16_t v12246 = v12244;	// L18901
        int16_t v12247 = v12245;	// L18902
        int16_t v12248 = v12246 * v12247;	// L18903
        int8_t v12249 = c206;	// L18904
        ap_int<17> v12250 = v12249;	// L18905
        ap_int<17> v12251 = v12248;	// L18906
        ap_int<17> v12252 = v12250 + v12251;	// L18907
        int8_t v12253 = v12252;	// L18908
        c206 = v12253;	// L18909
        int8_t v12254 = a219;	// L18910
        v12232.write(v12254);	// L18911
        int8_t v12255 = b222;	// L18912
        v12233.write(v12255);	// L18913
      }
      ap_uint<128> packed_tmp206;	// L18915
      ap_uint<128> v12257 = v12234.read();	// L18916
      packed_tmp206 = v12257;	// L18917
      ap_uint<128> packed_c206;	// L18918
      packed_c206 = 0;	// L18919
      l_S_m_1_m206: for (int m206 = 0; m206 < 16; m206++) {	// L18920
        ap_int<33> v12260 = m206;	// L18921
        bool v12261 = v12260 == 12;	// L18922
        if (v12261) {	// L18923
          int8_t v12262 = c206;	// L18924
          ap_int<128> v12263 = packed_c206;	// L18925
          int64_t v12264 = m206;	// L18926
          int64_t v12265 = v12264 * 8;	// L18927
          ap_int<34> v12266 = m206;	// L18928
          ap_int<34> v12267 = v12266 + 1;	// L18929
          ap_int<66> v12268 = v12267;	// L18930
          ap_int<66> v12269 = v12268 * 8;	// L18931
          ap_int<66> v12270 = v12269 - 1;	// L18932
          int v12271 = v12265;	// L18933
          int v12272 = v12270;	// L18934
          ap_int<128> v12273;
          ap_int<128> v12273_tmp = v12263;
          v12273_tmp(v12272, v12271) = v12262;
          v12273 = v12273_tmp;	// L18935
          packed_c206 = v12273;	// L18936
        } else {
          ap_int<128> v12274 = packed_tmp206;	// L18938
          int64_t v12275 = m206;	// L18939
          int64_t v12276 = v12275 * 8;	// L18940
          ap_int<34> v12277 = m206;	// L18941
          ap_int<34> v12278 = v12277 + 1;	// L18942
          ap_int<66> v12279 = v12278;	// L18943
          ap_int<66> v12280 = v12279 * 8;	// L18944
          ap_int<66> v12281 = v12280 - 1;	// L18945
          int v12282 = v12276;	// L18946
          int v12283 = v12281;	// L18947
          int8_t v12284;
          ap_int<128> v12284_tmp = v12274;
          v12284 = v12284_tmp(v12283, v12282);	// L18948
          ap_int<128> v12285 = packed_c206;	// L18949
          ap_int<128> v12286;
          ap_int<128> v12286_tmp = v12285;
          v12286_tmp(v12283, v12282) = v12284;
          v12286 = v12286_tmp;	// L18959
          packed_c206 = v12286;	// L18960
        }
      }
      ap_int<128> v12287 = packed_c206;	// L18963
      v12235.write(v12287);	// L18964
    }
  }
}

void gemm_13_16(
  hls::stream< int8_t >& v12288,
  hls::stream< int8_t >& v12289,
  hls::stream< int8_t >& v12290,
  hls::stream< ap_uint<128> >& v12291,
  hls::stream< ap_uint<128> >& v12292
) {	// L18969
  l_S_mt_nt_0_mt239: for (int mt239 = 0; mt239 < 4; mt239++) {	// L18977
    l_nt239: for (int nt239 = 0; nt239 < 4; nt239++) {	// L18978
      int8_t c207;	// L18979
      c207 = 0;	// L18980
      l_S_k_0_k239: for (int k239 = 0; k239 < 64; k239++) {	// L18981
        int8_t v12297 = v12288.read();	// L18982
        int8_t a220;	// L18983
        a220 = v12297;	// L18984
        int8_t v12299 = v12289.read();	// L18985
        int8_t b223;	// L18986
        b223 = v12299;	// L18987
        int8_t v12301 = a220;	// L18988
        int8_t v12302 = b223;	// L18989
        int16_t v12303 = v12301;	// L18990
        int16_t v12304 = v12302;	// L18991
        int16_t v12305 = v12303 * v12304;	// L18992
        int8_t v12306 = c207;	// L18993
        ap_int<17> v12307 = v12306;	// L18994
        ap_int<17> v12308 = v12305;	// L18995
        ap_int<17> v12309 = v12307 + v12308;	// L18996
        int8_t v12310 = v12309;	// L18997
        c207 = v12310;	// L18998
        int8_t v12311 = b223;	// L18999
        v12290.write(v12311);	// L19000
      }
      ap_uint<128> packed_tmp207;	// L19002
      ap_uint<128> v12313 = v12291.read();	// L19003
      packed_tmp207 = v12313;	// L19004
      ap_uint<128> packed_c207;	// L19005
      packed_c207 = 0;	// L19006
      l_S_m_1_m207: for (int m207 = 0; m207 < 16; m207++) {	// L19007
        ap_int<33> v12316 = m207;	// L19008
        bool v12317 = v12316 == 12;	// L19009
        if (v12317) {	// L19010
          int8_t v12318 = c207;	// L19011
          ap_int<128> v12319 = packed_c207;	// L19012
          int64_t v12320 = m207;	// L19013
          int64_t v12321 = v12320 * 8;	// L19014
          ap_int<34> v12322 = m207;	// L19015
          ap_int<34> v12323 = v12322 + 1;	// L19016
          ap_int<66> v12324 = v12323;	// L19017
          ap_int<66> v12325 = v12324 * 8;	// L19018
          ap_int<66> v12326 = v12325 - 1;	// L19019
          int v12327 = v12321;	// L19020
          int v12328 = v12326;	// L19021
          ap_int<128> v12329;
          ap_int<128> v12329_tmp = v12319;
          v12329_tmp(v12328, v12327) = v12318;
          v12329 = v12329_tmp;	// L19022
          packed_c207 = v12329;	// L19023
        } else {
          ap_int<128> v12330 = packed_tmp207;	// L19025
          int64_t v12331 = m207;	// L19026
          int64_t v12332 = v12331 * 8;	// L19027
          ap_int<34> v12333 = m207;	// L19028
          ap_int<34> v12334 = v12333 + 1;	// L19029
          ap_int<66> v12335 = v12334;	// L19030
          ap_int<66> v12336 = v12335 * 8;	// L19031
          ap_int<66> v12337 = v12336 - 1;	// L19032
          int v12338 = v12332;	// L19033
          int v12339 = v12337;	// L19034
          int8_t v12340;
          ap_int<128> v12340_tmp = v12330;
          v12340 = v12340_tmp(v12339, v12338);	// L19035
          ap_int<128> v12341 = packed_c207;	// L19036
          ap_int<128> v12342;
          ap_int<128> v12342_tmp = v12341;
          v12342_tmp(v12339, v12338) = v12340;
          v12342 = v12342_tmp;	// L19046
          packed_c207 = v12342;	// L19047
        }
      }
      ap_int<128> v12343 = packed_c207;	// L19050
      v12292.write(v12343);	// L19051
    }
  }
}

void gemm_14_0(
  hls::stream< ap_uint<128> >& v12344,
  hls::stream< int8_t >& v12345,
  hls::stream< ap_uint<128> >& v12346
) {	// L19056
  l_S_mt_nt_0_mt240: for (int mt240 = 0; mt240 < 4; mt240++) {	// L19059
    l_nt240: for (int nt240 = 0; nt240 < 4; nt240++) {	// L19060
      l_S_k_0_k240: for (int k240 = 0; k240 < 64; k240++) {	// L19061
        ap_uint<128> v12350 = v12344.read();	// L19062
        ap_uint<128> a221;	// L19063
        a221 = v12350;	// L19064
        ap_int<128> v12352 = a221;	// L19065
        int8_t v12353;
        ap_int<128> v12353_tmp = v12352;
        v12353 = v12353_tmp(111, 104);	// L19066
        v12345.write(v12353);	// L19067
        ap_int<128> v12354 = a221;	// L19068
        v12346.write(v12354);	// L19069
      }
    }
  }
}

void gemm_14_1(
  hls::stream< int8_t >& v12355,
  hls::stream< int8_t >& v12356,
  hls::stream< int8_t >& v12357,
  hls::stream< int8_t >& v12358,
  hls::stream< ap_uint<128> >& v12359,
  hls::stream< ap_uint<128> >& v12360
) {	// L19075
  l_S_mt_nt_0_mt241: for (int mt241 = 0; mt241 < 4; mt241++) {	// L19083
    l_nt241: for (int nt241 = 0; nt241 < 4; nt241++) {	// L19084
      int8_t c208;	// L19085
      c208 = 0;	// L19086
      l_S_k_0_k241: for (int k241 = 0; k241 < 64; k241++) {	// L19087
        int8_t v12365 = v12355.read();	// L19088
        int8_t a222;	// L19089
        a222 = v12365;	// L19090
        int8_t v12367 = v12356.read();	// L19091
        int8_t b224;	// L19092
        b224 = v12367;	// L19093
        int8_t v12369 = a222;	// L19094
        int8_t v12370 = b224;	// L19095
        int16_t v12371 = v12369;	// L19096
        int16_t v12372 = v12370;	// L19097
        int16_t v12373 = v12371 * v12372;	// L19098
        int8_t v12374 = c208;	// L19099
        ap_int<17> v12375 = v12374;	// L19100
        ap_int<17> v12376 = v12373;	// L19101
        ap_int<17> v12377 = v12375 + v12376;	// L19102
        int8_t v12378 = v12377;	// L19103
        c208 = v12378;	// L19104
        int8_t v12379 = a222;	// L19105
        v12357.write(v12379);	// L19106
        int8_t v12380 = b224;	// L19107
        v12358.write(v12380);	// L19108
      }
      ap_uint<128> packed_tmp208;	// L19110
      ap_uint<128> v12382 = v12359.read();	// L19111
      packed_tmp208 = v12382;	// L19112
      ap_uint<128> packed_c208;	// L19113
      packed_c208 = 0;	// L19114
      l_S_m_1_m208: for (int m208 = 0; m208 < 16; m208++) {	// L19115
        ap_int<33> v12385 = m208;	// L19116
        bool v12386 = v12385 == 13;	// L19117
        if (v12386) {	// L19118
          int8_t v12387 = c208;	// L19119
          ap_int<128> v12388 = packed_c208;	// L19120
          int64_t v12389 = m208;	// L19121
          int64_t v12390 = v12389 * 8;	// L19122
          ap_int<34> v12391 = m208;	// L19123
          ap_int<34> v12392 = v12391 + 1;	// L19124
          ap_int<66> v12393 = v12392;	// L19125
          ap_int<66> v12394 = v12393 * 8;	// L19126
          ap_int<66> v12395 = v12394 - 1;	// L19127
          int v12396 = v12390;	// L19128
          int v12397 = v12395;	// L19129
          ap_int<128> v12398;
          ap_int<128> v12398_tmp = v12388;
          v12398_tmp(v12397, v12396) = v12387;
          v12398 = v12398_tmp;	// L19130
          packed_c208 = v12398;	// L19131
        } else {
          ap_int<128> v12399 = packed_tmp208;	// L19133
          int64_t v12400 = m208;	// L19134
          int64_t v12401 = v12400 * 8;	// L19135
          ap_int<34> v12402 = m208;	// L19136
          ap_int<34> v12403 = v12402 + 1;	// L19137
          ap_int<66> v12404 = v12403;	// L19138
          ap_int<66> v12405 = v12404 * 8;	// L19139
          ap_int<66> v12406 = v12405 - 1;	// L19140
          int v12407 = v12401;	// L19141
          int v12408 = v12406;	// L19142
          int8_t v12409;
          ap_int<128> v12409_tmp = v12399;
          v12409 = v12409_tmp(v12408, v12407);	// L19143
          ap_int<128> v12410 = packed_c208;	// L19144
          ap_int<128> v12411;
          ap_int<128> v12411_tmp = v12410;
          v12411_tmp(v12408, v12407) = v12409;
          v12411 = v12411_tmp;	// L19154
          packed_c208 = v12411;	// L19155
        }
      }
      ap_int<128> v12412 = packed_c208;	// L19158
      v12360.write(v12412);	// L19159
    }
  }
}

void gemm_14_2(
  hls::stream< int8_t >& v12413,
  hls::stream< int8_t >& v12414,
  hls::stream< int8_t >& v12415,
  hls::stream< int8_t >& v12416,
  hls::stream< ap_uint<128> >& v12417,
  hls::stream< ap_uint<128> >& v12418
) {	// L19164
  l_S_mt_nt_0_mt242: for (int mt242 = 0; mt242 < 4; mt242++) {	// L19172
    l_nt242: for (int nt242 = 0; nt242 < 4; nt242++) {	// L19173
      int8_t c209;	// L19174
      c209 = 0;	// L19175
      l_S_k_0_k242: for (int k242 = 0; k242 < 64; k242++) {	// L19176
        int8_t v12423 = v12413.read();	// L19177
        int8_t a223;	// L19178
        a223 = v12423;	// L19179
        int8_t v12425 = v12414.read();	// L19180
        int8_t b225;	// L19181
        b225 = v12425;	// L19182
        int8_t v12427 = a223;	// L19183
        int8_t v12428 = b225;	// L19184
        int16_t v12429 = v12427;	// L19185
        int16_t v12430 = v12428;	// L19186
        int16_t v12431 = v12429 * v12430;	// L19187
        int8_t v12432 = c209;	// L19188
        ap_int<17> v12433 = v12432;	// L19189
        ap_int<17> v12434 = v12431;	// L19190
        ap_int<17> v12435 = v12433 + v12434;	// L19191
        int8_t v12436 = v12435;	// L19192
        c209 = v12436;	// L19193
        int8_t v12437 = a223;	// L19194
        v12415.write(v12437);	// L19195
        int8_t v12438 = b225;	// L19196
        v12416.write(v12438);	// L19197
      }
      ap_uint<128> packed_tmp209;	// L19199
      ap_uint<128> v12440 = v12417.read();	// L19200
      packed_tmp209 = v12440;	// L19201
      ap_uint<128> packed_c209;	// L19202
      packed_c209 = 0;	// L19203
      l_S_m_1_m209: for (int m209 = 0; m209 < 16; m209++) {	// L19204
        ap_int<33> v12443 = m209;	// L19205
        bool v12444 = v12443 == 13;	// L19206
        if (v12444) {	// L19207
          int8_t v12445 = c209;	// L19208
          ap_int<128> v12446 = packed_c209;	// L19209
          int64_t v12447 = m209;	// L19210
          int64_t v12448 = v12447 * 8;	// L19211
          ap_int<34> v12449 = m209;	// L19212
          ap_int<34> v12450 = v12449 + 1;	// L19213
          ap_int<66> v12451 = v12450;	// L19214
          ap_int<66> v12452 = v12451 * 8;	// L19215
          ap_int<66> v12453 = v12452 - 1;	// L19216
          int v12454 = v12448;	// L19217
          int v12455 = v12453;	// L19218
          ap_int<128> v12456;
          ap_int<128> v12456_tmp = v12446;
          v12456_tmp(v12455, v12454) = v12445;
          v12456 = v12456_tmp;	// L19219
          packed_c209 = v12456;	// L19220
        } else {
          ap_int<128> v12457 = packed_tmp209;	// L19222
          int64_t v12458 = m209;	// L19223
          int64_t v12459 = v12458 * 8;	// L19224
          ap_int<34> v12460 = m209;	// L19225
          ap_int<34> v12461 = v12460 + 1;	// L19226
          ap_int<66> v12462 = v12461;	// L19227
          ap_int<66> v12463 = v12462 * 8;	// L19228
          ap_int<66> v12464 = v12463 - 1;	// L19229
          int v12465 = v12459;	// L19230
          int v12466 = v12464;	// L19231
          int8_t v12467;
          ap_int<128> v12467_tmp = v12457;
          v12467 = v12467_tmp(v12466, v12465);	// L19232
          ap_int<128> v12468 = packed_c209;	// L19233
          ap_int<128> v12469;
          ap_int<128> v12469_tmp = v12468;
          v12469_tmp(v12466, v12465) = v12467;
          v12469 = v12469_tmp;	// L19243
          packed_c209 = v12469;	// L19244
        }
      }
      ap_int<128> v12470 = packed_c209;	// L19247
      v12418.write(v12470);	// L19248
    }
  }
}

void gemm_14_3(
  hls::stream< int8_t >& v12471,
  hls::stream< int8_t >& v12472,
  hls::stream< int8_t >& v12473,
  hls::stream< int8_t >& v12474,
  hls::stream< ap_uint<128> >& v12475,
  hls::stream< ap_uint<128> >& v12476
) {	// L19253
  l_S_mt_nt_0_mt243: for (int mt243 = 0; mt243 < 4; mt243++) {	// L19261
    l_nt243: for (int nt243 = 0; nt243 < 4; nt243++) {	// L19262
      int8_t c210;	// L19263
      c210 = 0;	// L19264
      l_S_k_0_k243: for (int k243 = 0; k243 < 64; k243++) {	// L19265
        int8_t v12481 = v12471.read();	// L19266
        int8_t a224;	// L19267
        a224 = v12481;	// L19268
        int8_t v12483 = v12472.read();	// L19269
        int8_t b226;	// L19270
        b226 = v12483;	// L19271
        int8_t v12485 = a224;	// L19272
        int8_t v12486 = b226;	// L19273
        int16_t v12487 = v12485;	// L19274
        int16_t v12488 = v12486;	// L19275
        int16_t v12489 = v12487 * v12488;	// L19276
        int8_t v12490 = c210;	// L19277
        ap_int<17> v12491 = v12490;	// L19278
        ap_int<17> v12492 = v12489;	// L19279
        ap_int<17> v12493 = v12491 + v12492;	// L19280
        int8_t v12494 = v12493;	// L19281
        c210 = v12494;	// L19282
        int8_t v12495 = a224;	// L19283
        v12473.write(v12495);	// L19284
        int8_t v12496 = b226;	// L19285
        v12474.write(v12496);	// L19286
      }
      ap_uint<128> packed_tmp210;	// L19288
      ap_uint<128> v12498 = v12475.read();	// L19289
      packed_tmp210 = v12498;	// L19290
      ap_uint<128> packed_c210;	// L19291
      packed_c210 = 0;	// L19292
      l_S_m_1_m210: for (int m210 = 0; m210 < 16; m210++) {	// L19293
        ap_int<33> v12501 = m210;	// L19294
        bool v12502 = v12501 == 13;	// L19295
        if (v12502) {	// L19296
          int8_t v12503 = c210;	// L19297
          ap_int<128> v12504 = packed_c210;	// L19298
          int64_t v12505 = m210;	// L19299
          int64_t v12506 = v12505 * 8;	// L19300
          ap_int<34> v12507 = m210;	// L19301
          ap_int<34> v12508 = v12507 + 1;	// L19302
          ap_int<66> v12509 = v12508;	// L19303
          ap_int<66> v12510 = v12509 * 8;	// L19304
          ap_int<66> v12511 = v12510 - 1;	// L19305
          int v12512 = v12506;	// L19306
          int v12513 = v12511;	// L19307
          ap_int<128> v12514;
          ap_int<128> v12514_tmp = v12504;
          v12514_tmp(v12513, v12512) = v12503;
          v12514 = v12514_tmp;	// L19308
          packed_c210 = v12514;	// L19309
        } else {
          ap_int<128> v12515 = packed_tmp210;	// L19311
          int64_t v12516 = m210;	// L19312
          int64_t v12517 = v12516 * 8;	// L19313
          ap_int<34> v12518 = m210;	// L19314
          ap_int<34> v12519 = v12518 + 1;	// L19315
          ap_int<66> v12520 = v12519;	// L19316
          ap_int<66> v12521 = v12520 * 8;	// L19317
          ap_int<66> v12522 = v12521 - 1;	// L19318
          int v12523 = v12517;	// L19319
          int v12524 = v12522;	// L19320
          int8_t v12525;
          ap_int<128> v12525_tmp = v12515;
          v12525 = v12525_tmp(v12524, v12523);	// L19321
          ap_int<128> v12526 = packed_c210;	// L19322
          ap_int<128> v12527;
          ap_int<128> v12527_tmp = v12526;
          v12527_tmp(v12524, v12523) = v12525;
          v12527 = v12527_tmp;	// L19332
          packed_c210 = v12527;	// L19333
        }
      }
      ap_int<128> v12528 = packed_c210;	// L19336
      v12476.write(v12528);	// L19337
    }
  }
}

void gemm_14_4(
  hls::stream< int8_t >& v12529,
  hls::stream< int8_t >& v12530,
  hls::stream< int8_t >& v12531,
  hls::stream< int8_t >& v12532,
  hls::stream< ap_uint<128> >& v12533,
  hls::stream< ap_uint<128> >& v12534
) {	// L19342
  l_S_mt_nt_0_mt244: for (int mt244 = 0; mt244 < 4; mt244++) {	// L19350
    l_nt244: for (int nt244 = 0; nt244 < 4; nt244++) {	// L19351
      int8_t c211;	// L19352
      c211 = 0;	// L19353
      l_S_k_0_k244: for (int k244 = 0; k244 < 64; k244++) {	// L19354
        int8_t v12539 = v12529.read();	// L19355
        int8_t a225;	// L19356
        a225 = v12539;	// L19357
        int8_t v12541 = v12530.read();	// L19358
        int8_t b227;	// L19359
        b227 = v12541;	// L19360
        int8_t v12543 = a225;	// L19361
        int8_t v12544 = b227;	// L19362
        int16_t v12545 = v12543;	// L19363
        int16_t v12546 = v12544;	// L19364
        int16_t v12547 = v12545 * v12546;	// L19365
        int8_t v12548 = c211;	// L19366
        ap_int<17> v12549 = v12548;	// L19367
        ap_int<17> v12550 = v12547;	// L19368
        ap_int<17> v12551 = v12549 + v12550;	// L19369
        int8_t v12552 = v12551;	// L19370
        c211 = v12552;	// L19371
        int8_t v12553 = a225;	// L19372
        v12531.write(v12553);	// L19373
        int8_t v12554 = b227;	// L19374
        v12532.write(v12554);	// L19375
      }
      ap_uint<128> packed_tmp211;	// L19377
      ap_uint<128> v12556 = v12533.read();	// L19378
      packed_tmp211 = v12556;	// L19379
      ap_uint<128> packed_c211;	// L19380
      packed_c211 = 0;	// L19381
      l_S_m_1_m211: for (int m211 = 0; m211 < 16; m211++) {	// L19382
        ap_int<33> v12559 = m211;	// L19383
        bool v12560 = v12559 == 13;	// L19384
        if (v12560) {	// L19385
          int8_t v12561 = c211;	// L19386
          ap_int<128> v12562 = packed_c211;	// L19387
          int64_t v12563 = m211;	// L19388
          int64_t v12564 = v12563 * 8;	// L19389
          ap_int<34> v12565 = m211;	// L19390
          ap_int<34> v12566 = v12565 + 1;	// L19391
          ap_int<66> v12567 = v12566;	// L19392
          ap_int<66> v12568 = v12567 * 8;	// L19393
          ap_int<66> v12569 = v12568 - 1;	// L19394
          int v12570 = v12564;	// L19395
          int v12571 = v12569;	// L19396
          ap_int<128> v12572;
          ap_int<128> v12572_tmp = v12562;
          v12572_tmp(v12571, v12570) = v12561;
          v12572 = v12572_tmp;	// L19397
          packed_c211 = v12572;	// L19398
        } else {
          ap_int<128> v12573 = packed_tmp211;	// L19400
          int64_t v12574 = m211;	// L19401
          int64_t v12575 = v12574 * 8;	// L19402
          ap_int<34> v12576 = m211;	// L19403
          ap_int<34> v12577 = v12576 + 1;	// L19404
          ap_int<66> v12578 = v12577;	// L19405
          ap_int<66> v12579 = v12578 * 8;	// L19406
          ap_int<66> v12580 = v12579 - 1;	// L19407
          int v12581 = v12575;	// L19408
          int v12582 = v12580;	// L19409
          int8_t v12583;
          ap_int<128> v12583_tmp = v12573;
          v12583 = v12583_tmp(v12582, v12581);	// L19410
          ap_int<128> v12584 = packed_c211;	// L19411
          ap_int<128> v12585;
          ap_int<128> v12585_tmp = v12584;
          v12585_tmp(v12582, v12581) = v12583;
          v12585 = v12585_tmp;	// L19421
          packed_c211 = v12585;	// L19422
        }
      }
      ap_int<128> v12586 = packed_c211;	// L19425
      v12534.write(v12586);	// L19426
    }
  }
}

void gemm_14_5(
  hls::stream< int8_t >& v12587,
  hls::stream< int8_t >& v12588,
  hls::stream< int8_t >& v12589,
  hls::stream< int8_t >& v12590,
  hls::stream< ap_uint<128> >& v12591,
  hls::stream< ap_uint<128> >& v12592
) {	// L19431
  l_S_mt_nt_0_mt245: for (int mt245 = 0; mt245 < 4; mt245++) {	// L19439
    l_nt245: for (int nt245 = 0; nt245 < 4; nt245++) {	// L19440
      int8_t c212;	// L19441
      c212 = 0;	// L19442
      l_S_k_0_k245: for (int k245 = 0; k245 < 64; k245++) {	// L19443
        int8_t v12597 = v12587.read();	// L19444
        int8_t a226;	// L19445
        a226 = v12597;	// L19446
        int8_t v12599 = v12588.read();	// L19447
        int8_t b228;	// L19448
        b228 = v12599;	// L19449
        int8_t v12601 = a226;	// L19450
        int8_t v12602 = b228;	// L19451
        int16_t v12603 = v12601;	// L19452
        int16_t v12604 = v12602;	// L19453
        int16_t v12605 = v12603 * v12604;	// L19454
        int8_t v12606 = c212;	// L19455
        ap_int<17> v12607 = v12606;	// L19456
        ap_int<17> v12608 = v12605;	// L19457
        ap_int<17> v12609 = v12607 + v12608;	// L19458
        int8_t v12610 = v12609;	// L19459
        c212 = v12610;	// L19460
        int8_t v12611 = a226;	// L19461
        v12589.write(v12611);	// L19462
        int8_t v12612 = b228;	// L19463
        v12590.write(v12612);	// L19464
      }
      ap_uint<128> packed_tmp212;	// L19466
      ap_uint<128> v12614 = v12591.read();	// L19467
      packed_tmp212 = v12614;	// L19468
      ap_uint<128> packed_c212;	// L19469
      packed_c212 = 0;	// L19470
      l_S_m_1_m212: for (int m212 = 0; m212 < 16; m212++) {	// L19471
        ap_int<33> v12617 = m212;	// L19472
        bool v12618 = v12617 == 13;	// L19473
        if (v12618) {	// L19474
          int8_t v12619 = c212;	// L19475
          ap_int<128> v12620 = packed_c212;	// L19476
          int64_t v12621 = m212;	// L19477
          int64_t v12622 = v12621 * 8;	// L19478
          ap_int<34> v12623 = m212;	// L19479
          ap_int<34> v12624 = v12623 + 1;	// L19480
          ap_int<66> v12625 = v12624;	// L19481
          ap_int<66> v12626 = v12625 * 8;	// L19482
          ap_int<66> v12627 = v12626 - 1;	// L19483
          int v12628 = v12622;	// L19484
          int v12629 = v12627;	// L19485
          ap_int<128> v12630;
          ap_int<128> v12630_tmp = v12620;
          v12630_tmp(v12629, v12628) = v12619;
          v12630 = v12630_tmp;	// L19486
          packed_c212 = v12630;	// L19487
        } else {
          ap_int<128> v12631 = packed_tmp212;	// L19489
          int64_t v12632 = m212;	// L19490
          int64_t v12633 = v12632 * 8;	// L19491
          ap_int<34> v12634 = m212;	// L19492
          ap_int<34> v12635 = v12634 + 1;	// L19493
          ap_int<66> v12636 = v12635;	// L19494
          ap_int<66> v12637 = v12636 * 8;	// L19495
          ap_int<66> v12638 = v12637 - 1;	// L19496
          int v12639 = v12633;	// L19497
          int v12640 = v12638;	// L19498
          int8_t v12641;
          ap_int<128> v12641_tmp = v12631;
          v12641 = v12641_tmp(v12640, v12639);	// L19499
          ap_int<128> v12642 = packed_c212;	// L19500
          ap_int<128> v12643;
          ap_int<128> v12643_tmp = v12642;
          v12643_tmp(v12640, v12639) = v12641;
          v12643 = v12643_tmp;	// L19510
          packed_c212 = v12643;	// L19511
        }
      }
      ap_int<128> v12644 = packed_c212;	// L19514
      v12592.write(v12644);	// L19515
    }
  }
}

void gemm_14_6(
  hls::stream< int8_t >& v12645,
  hls::stream< int8_t >& v12646,
  hls::stream< int8_t >& v12647,
  hls::stream< int8_t >& v12648,
  hls::stream< ap_uint<128> >& v12649,
  hls::stream< ap_uint<128> >& v12650
) {	// L19520
  l_S_mt_nt_0_mt246: for (int mt246 = 0; mt246 < 4; mt246++) {	// L19528
    l_nt246: for (int nt246 = 0; nt246 < 4; nt246++) {	// L19529
      int8_t c213;	// L19530
      c213 = 0;	// L19531
      l_S_k_0_k246: for (int k246 = 0; k246 < 64; k246++) {	// L19532
        int8_t v12655 = v12645.read();	// L19533
        int8_t a227;	// L19534
        a227 = v12655;	// L19535
        int8_t v12657 = v12646.read();	// L19536
        int8_t b229;	// L19537
        b229 = v12657;	// L19538
        int8_t v12659 = a227;	// L19539
        int8_t v12660 = b229;	// L19540
        int16_t v12661 = v12659;	// L19541
        int16_t v12662 = v12660;	// L19542
        int16_t v12663 = v12661 * v12662;	// L19543
        int8_t v12664 = c213;	// L19544
        ap_int<17> v12665 = v12664;	// L19545
        ap_int<17> v12666 = v12663;	// L19546
        ap_int<17> v12667 = v12665 + v12666;	// L19547
        int8_t v12668 = v12667;	// L19548
        c213 = v12668;	// L19549
        int8_t v12669 = a227;	// L19550
        v12647.write(v12669);	// L19551
        int8_t v12670 = b229;	// L19552
        v12648.write(v12670);	// L19553
      }
      ap_uint<128> packed_tmp213;	// L19555
      ap_uint<128> v12672 = v12649.read();	// L19556
      packed_tmp213 = v12672;	// L19557
      ap_uint<128> packed_c213;	// L19558
      packed_c213 = 0;	// L19559
      l_S_m_1_m213: for (int m213 = 0; m213 < 16; m213++) {	// L19560
        ap_int<33> v12675 = m213;	// L19561
        bool v12676 = v12675 == 13;	// L19562
        if (v12676) {	// L19563
          int8_t v12677 = c213;	// L19564
          ap_int<128> v12678 = packed_c213;	// L19565
          int64_t v12679 = m213;	// L19566
          int64_t v12680 = v12679 * 8;	// L19567
          ap_int<34> v12681 = m213;	// L19568
          ap_int<34> v12682 = v12681 + 1;	// L19569
          ap_int<66> v12683 = v12682;	// L19570
          ap_int<66> v12684 = v12683 * 8;	// L19571
          ap_int<66> v12685 = v12684 - 1;	// L19572
          int v12686 = v12680;	// L19573
          int v12687 = v12685;	// L19574
          ap_int<128> v12688;
          ap_int<128> v12688_tmp = v12678;
          v12688_tmp(v12687, v12686) = v12677;
          v12688 = v12688_tmp;	// L19575
          packed_c213 = v12688;	// L19576
        } else {
          ap_int<128> v12689 = packed_tmp213;	// L19578
          int64_t v12690 = m213;	// L19579
          int64_t v12691 = v12690 * 8;	// L19580
          ap_int<34> v12692 = m213;	// L19581
          ap_int<34> v12693 = v12692 + 1;	// L19582
          ap_int<66> v12694 = v12693;	// L19583
          ap_int<66> v12695 = v12694 * 8;	// L19584
          ap_int<66> v12696 = v12695 - 1;	// L19585
          int v12697 = v12691;	// L19586
          int v12698 = v12696;	// L19587
          int8_t v12699;
          ap_int<128> v12699_tmp = v12689;
          v12699 = v12699_tmp(v12698, v12697);	// L19588
          ap_int<128> v12700 = packed_c213;	// L19589
          ap_int<128> v12701;
          ap_int<128> v12701_tmp = v12700;
          v12701_tmp(v12698, v12697) = v12699;
          v12701 = v12701_tmp;	// L19599
          packed_c213 = v12701;	// L19600
        }
      }
      ap_int<128> v12702 = packed_c213;	// L19603
      v12650.write(v12702);	// L19604
    }
  }
}

void gemm_14_7(
  hls::stream< int8_t >& v12703,
  hls::stream< int8_t >& v12704,
  hls::stream< int8_t >& v12705,
  hls::stream< int8_t >& v12706,
  hls::stream< ap_uint<128> >& v12707,
  hls::stream< ap_uint<128> >& v12708
) {	// L19609
  l_S_mt_nt_0_mt247: for (int mt247 = 0; mt247 < 4; mt247++) {	// L19617
    l_nt247: for (int nt247 = 0; nt247 < 4; nt247++) {	// L19618
      int8_t c214;	// L19619
      c214 = 0;	// L19620
      l_S_k_0_k247: for (int k247 = 0; k247 < 64; k247++) {	// L19621
        int8_t v12713 = v12703.read();	// L19622
        int8_t a228;	// L19623
        a228 = v12713;	// L19624
        int8_t v12715 = v12704.read();	// L19625
        int8_t b230;	// L19626
        b230 = v12715;	// L19627
        int8_t v12717 = a228;	// L19628
        int8_t v12718 = b230;	// L19629
        int16_t v12719 = v12717;	// L19630
        int16_t v12720 = v12718;	// L19631
        int16_t v12721 = v12719 * v12720;	// L19632
        int8_t v12722 = c214;	// L19633
        ap_int<17> v12723 = v12722;	// L19634
        ap_int<17> v12724 = v12721;	// L19635
        ap_int<17> v12725 = v12723 + v12724;	// L19636
        int8_t v12726 = v12725;	// L19637
        c214 = v12726;	// L19638
        int8_t v12727 = a228;	// L19639
        v12705.write(v12727);	// L19640
        int8_t v12728 = b230;	// L19641
        v12706.write(v12728);	// L19642
      }
      ap_uint<128> packed_tmp214;	// L19644
      ap_uint<128> v12730 = v12707.read();	// L19645
      packed_tmp214 = v12730;	// L19646
      ap_uint<128> packed_c214;	// L19647
      packed_c214 = 0;	// L19648
      l_S_m_1_m214: for (int m214 = 0; m214 < 16; m214++) {	// L19649
        ap_int<33> v12733 = m214;	// L19650
        bool v12734 = v12733 == 13;	// L19651
        if (v12734) {	// L19652
          int8_t v12735 = c214;	// L19653
          ap_int<128> v12736 = packed_c214;	// L19654
          int64_t v12737 = m214;	// L19655
          int64_t v12738 = v12737 * 8;	// L19656
          ap_int<34> v12739 = m214;	// L19657
          ap_int<34> v12740 = v12739 + 1;	// L19658
          ap_int<66> v12741 = v12740;	// L19659
          ap_int<66> v12742 = v12741 * 8;	// L19660
          ap_int<66> v12743 = v12742 - 1;	// L19661
          int v12744 = v12738;	// L19662
          int v12745 = v12743;	// L19663
          ap_int<128> v12746;
          ap_int<128> v12746_tmp = v12736;
          v12746_tmp(v12745, v12744) = v12735;
          v12746 = v12746_tmp;	// L19664
          packed_c214 = v12746;	// L19665
        } else {
          ap_int<128> v12747 = packed_tmp214;	// L19667
          int64_t v12748 = m214;	// L19668
          int64_t v12749 = v12748 * 8;	// L19669
          ap_int<34> v12750 = m214;	// L19670
          ap_int<34> v12751 = v12750 + 1;	// L19671
          ap_int<66> v12752 = v12751;	// L19672
          ap_int<66> v12753 = v12752 * 8;	// L19673
          ap_int<66> v12754 = v12753 - 1;	// L19674
          int v12755 = v12749;	// L19675
          int v12756 = v12754;	// L19676
          int8_t v12757;
          ap_int<128> v12757_tmp = v12747;
          v12757 = v12757_tmp(v12756, v12755);	// L19677
          ap_int<128> v12758 = packed_c214;	// L19678
          ap_int<128> v12759;
          ap_int<128> v12759_tmp = v12758;
          v12759_tmp(v12756, v12755) = v12757;
          v12759 = v12759_tmp;	// L19688
          packed_c214 = v12759;	// L19689
        }
      }
      ap_int<128> v12760 = packed_c214;	// L19692
      v12708.write(v12760);	// L19693
    }
  }
}

void gemm_14_8(
  hls::stream< int8_t >& v12761,
  hls::stream< int8_t >& v12762,
  hls::stream< int8_t >& v12763,
  hls::stream< int8_t >& v12764,
  hls::stream< ap_uint<128> >& v12765,
  hls::stream< ap_uint<128> >& v12766
) {	// L19698
  l_S_mt_nt_0_mt248: for (int mt248 = 0; mt248 < 4; mt248++) {	// L19706
    l_nt248: for (int nt248 = 0; nt248 < 4; nt248++) {	// L19707
      int8_t c215;	// L19708
      c215 = 0;	// L19709
      l_S_k_0_k248: for (int k248 = 0; k248 < 64; k248++) {	// L19710
        int8_t v12771 = v12761.read();	// L19711
        int8_t a229;	// L19712
        a229 = v12771;	// L19713
        int8_t v12773 = v12762.read();	// L19714
        int8_t b231;	// L19715
        b231 = v12773;	// L19716
        int8_t v12775 = a229;	// L19717
        int8_t v12776 = b231;	// L19718
        int16_t v12777 = v12775;	// L19719
        int16_t v12778 = v12776;	// L19720
        int16_t v12779 = v12777 * v12778;	// L19721
        int8_t v12780 = c215;	// L19722
        ap_int<17> v12781 = v12780;	// L19723
        ap_int<17> v12782 = v12779;	// L19724
        ap_int<17> v12783 = v12781 + v12782;	// L19725
        int8_t v12784 = v12783;	// L19726
        c215 = v12784;	// L19727
        int8_t v12785 = a229;	// L19728
        v12763.write(v12785);	// L19729
        int8_t v12786 = b231;	// L19730
        v12764.write(v12786);	// L19731
      }
      ap_uint<128> packed_tmp215;	// L19733
      ap_uint<128> v12788 = v12765.read();	// L19734
      packed_tmp215 = v12788;	// L19735
      ap_uint<128> packed_c215;	// L19736
      packed_c215 = 0;	// L19737
      l_S_m_1_m215: for (int m215 = 0; m215 < 16; m215++) {	// L19738
        ap_int<33> v12791 = m215;	// L19739
        bool v12792 = v12791 == 13;	// L19740
        if (v12792) {	// L19741
          int8_t v12793 = c215;	// L19742
          ap_int<128> v12794 = packed_c215;	// L19743
          int64_t v12795 = m215;	// L19744
          int64_t v12796 = v12795 * 8;	// L19745
          ap_int<34> v12797 = m215;	// L19746
          ap_int<34> v12798 = v12797 + 1;	// L19747
          ap_int<66> v12799 = v12798;	// L19748
          ap_int<66> v12800 = v12799 * 8;	// L19749
          ap_int<66> v12801 = v12800 - 1;	// L19750
          int v12802 = v12796;	// L19751
          int v12803 = v12801;	// L19752
          ap_int<128> v12804;
          ap_int<128> v12804_tmp = v12794;
          v12804_tmp(v12803, v12802) = v12793;
          v12804 = v12804_tmp;	// L19753
          packed_c215 = v12804;	// L19754
        } else {
          ap_int<128> v12805 = packed_tmp215;	// L19756
          int64_t v12806 = m215;	// L19757
          int64_t v12807 = v12806 * 8;	// L19758
          ap_int<34> v12808 = m215;	// L19759
          ap_int<34> v12809 = v12808 + 1;	// L19760
          ap_int<66> v12810 = v12809;	// L19761
          ap_int<66> v12811 = v12810 * 8;	// L19762
          ap_int<66> v12812 = v12811 - 1;	// L19763
          int v12813 = v12807;	// L19764
          int v12814 = v12812;	// L19765
          int8_t v12815;
          ap_int<128> v12815_tmp = v12805;
          v12815 = v12815_tmp(v12814, v12813);	// L19766
          ap_int<128> v12816 = packed_c215;	// L19767
          ap_int<128> v12817;
          ap_int<128> v12817_tmp = v12816;
          v12817_tmp(v12814, v12813) = v12815;
          v12817 = v12817_tmp;	// L19777
          packed_c215 = v12817;	// L19778
        }
      }
      ap_int<128> v12818 = packed_c215;	// L19781
      v12766.write(v12818);	// L19782
    }
  }
}

void gemm_14_9(
  hls::stream< int8_t >& v12819,
  hls::stream< int8_t >& v12820,
  hls::stream< int8_t >& v12821,
  hls::stream< int8_t >& v12822,
  hls::stream< ap_uint<128> >& v12823,
  hls::stream< ap_uint<128> >& v12824
) {	// L19787
  l_S_mt_nt_0_mt249: for (int mt249 = 0; mt249 < 4; mt249++) {	// L19795
    l_nt249: for (int nt249 = 0; nt249 < 4; nt249++) {	// L19796
      int8_t c216;	// L19797
      c216 = 0;	// L19798
      l_S_k_0_k249: for (int k249 = 0; k249 < 64; k249++) {	// L19799
        int8_t v12829 = v12819.read();	// L19800
        int8_t a230;	// L19801
        a230 = v12829;	// L19802
        int8_t v12831 = v12820.read();	// L19803
        int8_t b232;	// L19804
        b232 = v12831;	// L19805
        int8_t v12833 = a230;	// L19806
        int8_t v12834 = b232;	// L19807
        int16_t v12835 = v12833;	// L19808
        int16_t v12836 = v12834;	// L19809
        int16_t v12837 = v12835 * v12836;	// L19810
        int8_t v12838 = c216;	// L19811
        ap_int<17> v12839 = v12838;	// L19812
        ap_int<17> v12840 = v12837;	// L19813
        ap_int<17> v12841 = v12839 + v12840;	// L19814
        int8_t v12842 = v12841;	// L19815
        c216 = v12842;	// L19816
        int8_t v12843 = a230;	// L19817
        v12821.write(v12843);	// L19818
        int8_t v12844 = b232;	// L19819
        v12822.write(v12844);	// L19820
      }
      ap_uint<128> packed_tmp216;	// L19822
      ap_uint<128> v12846 = v12823.read();	// L19823
      packed_tmp216 = v12846;	// L19824
      ap_uint<128> packed_c216;	// L19825
      packed_c216 = 0;	// L19826
      l_S_m_1_m216: for (int m216 = 0; m216 < 16; m216++) {	// L19827
        ap_int<33> v12849 = m216;	// L19828
        bool v12850 = v12849 == 13;	// L19829
        if (v12850) {	// L19830
          int8_t v12851 = c216;	// L19831
          ap_int<128> v12852 = packed_c216;	// L19832
          int64_t v12853 = m216;	// L19833
          int64_t v12854 = v12853 * 8;	// L19834
          ap_int<34> v12855 = m216;	// L19835
          ap_int<34> v12856 = v12855 + 1;	// L19836
          ap_int<66> v12857 = v12856;	// L19837
          ap_int<66> v12858 = v12857 * 8;	// L19838
          ap_int<66> v12859 = v12858 - 1;	// L19839
          int v12860 = v12854;	// L19840
          int v12861 = v12859;	// L19841
          ap_int<128> v12862;
          ap_int<128> v12862_tmp = v12852;
          v12862_tmp(v12861, v12860) = v12851;
          v12862 = v12862_tmp;	// L19842
          packed_c216 = v12862;	// L19843
        } else {
          ap_int<128> v12863 = packed_tmp216;	// L19845
          int64_t v12864 = m216;	// L19846
          int64_t v12865 = v12864 * 8;	// L19847
          ap_int<34> v12866 = m216;	// L19848
          ap_int<34> v12867 = v12866 + 1;	// L19849
          ap_int<66> v12868 = v12867;	// L19850
          ap_int<66> v12869 = v12868 * 8;	// L19851
          ap_int<66> v12870 = v12869 - 1;	// L19852
          int v12871 = v12865;	// L19853
          int v12872 = v12870;	// L19854
          int8_t v12873;
          ap_int<128> v12873_tmp = v12863;
          v12873 = v12873_tmp(v12872, v12871);	// L19855
          ap_int<128> v12874 = packed_c216;	// L19856
          ap_int<128> v12875;
          ap_int<128> v12875_tmp = v12874;
          v12875_tmp(v12872, v12871) = v12873;
          v12875 = v12875_tmp;	// L19866
          packed_c216 = v12875;	// L19867
        }
      }
      ap_int<128> v12876 = packed_c216;	// L19870
      v12824.write(v12876);	// L19871
    }
  }
}

void gemm_14_10(
  hls::stream< int8_t >& v12877,
  hls::stream< int8_t >& v12878,
  hls::stream< int8_t >& v12879,
  hls::stream< int8_t >& v12880,
  hls::stream< ap_uint<128> >& v12881,
  hls::stream< ap_uint<128> >& v12882
) {	// L19876
  l_S_mt_nt_0_mt250: for (int mt250 = 0; mt250 < 4; mt250++) {	// L19884
    l_nt250: for (int nt250 = 0; nt250 < 4; nt250++) {	// L19885
      int8_t c217;	// L19886
      c217 = 0;	// L19887
      l_S_k_0_k250: for (int k250 = 0; k250 < 64; k250++) {	// L19888
        int8_t v12887 = v12877.read();	// L19889
        int8_t a231;	// L19890
        a231 = v12887;	// L19891
        int8_t v12889 = v12878.read();	// L19892
        int8_t b233;	// L19893
        b233 = v12889;	// L19894
        int8_t v12891 = a231;	// L19895
        int8_t v12892 = b233;	// L19896
        int16_t v12893 = v12891;	// L19897
        int16_t v12894 = v12892;	// L19898
        int16_t v12895 = v12893 * v12894;	// L19899
        int8_t v12896 = c217;	// L19900
        ap_int<17> v12897 = v12896;	// L19901
        ap_int<17> v12898 = v12895;	// L19902
        ap_int<17> v12899 = v12897 + v12898;	// L19903
        int8_t v12900 = v12899;	// L19904
        c217 = v12900;	// L19905
        int8_t v12901 = a231;	// L19906
        v12879.write(v12901);	// L19907
        int8_t v12902 = b233;	// L19908
        v12880.write(v12902);	// L19909
      }
      ap_uint<128> packed_tmp217;	// L19911
      ap_uint<128> v12904 = v12881.read();	// L19912
      packed_tmp217 = v12904;	// L19913
      ap_uint<128> packed_c217;	// L19914
      packed_c217 = 0;	// L19915
      l_S_m_1_m217: for (int m217 = 0; m217 < 16; m217++) {	// L19916
        ap_int<33> v12907 = m217;	// L19917
        bool v12908 = v12907 == 13;	// L19918
        if (v12908) {	// L19919
          int8_t v12909 = c217;	// L19920
          ap_int<128> v12910 = packed_c217;	// L19921
          int64_t v12911 = m217;	// L19922
          int64_t v12912 = v12911 * 8;	// L19923
          ap_int<34> v12913 = m217;	// L19924
          ap_int<34> v12914 = v12913 + 1;	// L19925
          ap_int<66> v12915 = v12914;	// L19926
          ap_int<66> v12916 = v12915 * 8;	// L19927
          ap_int<66> v12917 = v12916 - 1;	// L19928
          int v12918 = v12912;	// L19929
          int v12919 = v12917;	// L19930
          ap_int<128> v12920;
          ap_int<128> v12920_tmp = v12910;
          v12920_tmp(v12919, v12918) = v12909;
          v12920 = v12920_tmp;	// L19931
          packed_c217 = v12920;	// L19932
        } else {
          ap_int<128> v12921 = packed_tmp217;	// L19934
          int64_t v12922 = m217;	// L19935
          int64_t v12923 = v12922 * 8;	// L19936
          ap_int<34> v12924 = m217;	// L19937
          ap_int<34> v12925 = v12924 + 1;	// L19938
          ap_int<66> v12926 = v12925;	// L19939
          ap_int<66> v12927 = v12926 * 8;	// L19940
          ap_int<66> v12928 = v12927 - 1;	// L19941
          int v12929 = v12923;	// L19942
          int v12930 = v12928;	// L19943
          int8_t v12931;
          ap_int<128> v12931_tmp = v12921;
          v12931 = v12931_tmp(v12930, v12929);	// L19944
          ap_int<128> v12932 = packed_c217;	// L19945
          ap_int<128> v12933;
          ap_int<128> v12933_tmp = v12932;
          v12933_tmp(v12930, v12929) = v12931;
          v12933 = v12933_tmp;	// L19955
          packed_c217 = v12933;	// L19956
        }
      }
      ap_int<128> v12934 = packed_c217;	// L19959
      v12882.write(v12934);	// L19960
    }
  }
}

void gemm_14_11(
  hls::stream< int8_t >& v12935,
  hls::stream< int8_t >& v12936,
  hls::stream< int8_t >& v12937,
  hls::stream< int8_t >& v12938,
  hls::stream< ap_uint<128> >& v12939,
  hls::stream< ap_uint<128> >& v12940
) {	// L19965
  l_S_mt_nt_0_mt251: for (int mt251 = 0; mt251 < 4; mt251++) {	// L19973
    l_nt251: for (int nt251 = 0; nt251 < 4; nt251++) {	// L19974
      int8_t c218;	// L19975
      c218 = 0;	// L19976
      l_S_k_0_k251: for (int k251 = 0; k251 < 64; k251++) {	// L19977
        int8_t v12945 = v12935.read();	// L19978
        int8_t a232;	// L19979
        a232 = v12945;	// L19980
        int8_t v12947 = v12936.read();	// L19981
        int8_t b234;	// L19982
        b234 = v12947;	// L19983
        int8_t v12949 = a232;	// L19984
        int8_t v12950 = b234;	// L19985
        int16_t v12951 = v12949;	// L19986
        int16_t v12952 = v12950;	// L19987
        int16_t v12953 = v12951 * v12952;	// L19988
        int8_t v12954 = c218;	// L19989
        ap_int<17> v12955 = v12954;	// L19990
        ap_int<17> v12956 = v12953;	// L19991
        ap_int<17> v12957 = v12955 + v12956;	// L19992
        int8_t v12958 = v12957;	// L19993
        c218 = v12958;	// L19994
        int8_t v12959 = a232;	// L19995
        v12937.write(v12959);	// L19996
        int8_t v12960 = b234;	// L19997
        v12938.write(v12960);	// L19998
      }
      ap_uint<128> packed_tmp218;	// L20000
      ap_uint<128> v12962 = v12939.read();	// L20001
      packed_tmp218 = v12962;	// L20002
      ap_uint<128> packed_c218;	// L20003
      packed_c218 = 0;	// L20004
      l_S_m_1_m218: for (int m218 = 0; m218 < 16; m218++) {	// L20005
        ap_int<33> v12965 = m218;	// L20006
        bool v12966 = v12965 == 13;	// L20007
        if (v12966) {	// L20008
          int8_t v12967 = c218;	// L20009
          ap_int<128> v12968 = packed_c218;	// L20010
          int64_t v12969 = m218;	// L20011
          int64_t v12970 = v12969 * 8;	// L20012
          ap_int<34> v12971 = m218;	// L20013
          ap_int<34> v12972 = v12971 + 1;	// L20014
          ap_int<66> v12973 = v12972;	// L20015
          ap_int<66> v12974 = v12973 * 8;	// L20016
          ap_int<66> v12975 = v12974 - 1;	// L20017
          int v12976 = v12970;	// L20018
          int v12977 = v12975;	// L20019
          ap_int<128> v12978;
          ap_int<128> v12978_tmp = v12968;
          v12978_tmp(v12977, v12976) = v12967;
          v12978 = v12978_tmp;	// L20020
          packed_c218 = v12978;	// L20021
        } else {
          ap_int<128> v12979 = packed_tmp218;	// L20023
          int64_t v12980 = m218;	// L20024
          int64_t v12981 = v12980 * 8;	// L20025
          ap_int<34> v12982 = m218;	// L20026
          ap_int<34> v12983 = v12982 + 1;	// L20027
          ap_int<66> v12984 = v12983;	// L20028
          ap_int<66> v12985 = v12984 * 8;	// L20029
          ap_int<66> v12986 = v12985 - 1;	// L20030
          int v12987 = v12981;	// L20031
          int v12988 = v12986;	// L20032
          int8_t v12989;
          ap_int<128> v12989_tmp = v12979;
          v12989 = v12989_tmp(v12988, v12987);	// L20033
          ap_int<128> v12990 = packed_c218;	// L20034
          ap_int<128> v12991;
          ap_int<128> v12991_tmp = v12990;
          v12991_tmp(v12988, v12987) = v12989;
          v12991 = v12991_tmp;	// L20044
          packed_c218 = v12991;	// L20045
        }
      }
      ap_int<128> v12992 = packed_c218;	// L20048
      v12940.write(v12992);	// L20049
    }
  }
}

void gemm_14_12(
  hls::stream< int8_t >& v12993,
  hls::stream< int8_t >& v12994,
  hls::stream< int8_t >& v12995,
  hls::stream< int8_t >& v12996,
  hls::stream< ap_uint<128> >& v12997,
  hls::stream< ap_uint<128> >& v12998
) {	// L20054
  l_S_mt_nt_0_mt252: for (int mt252 = 0; mt252 < 4; mt252++) {	// L20062
    l_nt252: for (int nt252 = 0; nt252 < 4; nt252++) {	// L20063
      int8_t c219;	// L20064
      c219 = 0;	// L20065
      l_S_k_0_k252: for (int k252 = 0; k252 < 64; k252++) {	// L20066
        int8_t v13003 = v12993.read();	// L20067
        int8_t a233;	// L20068
        a233 = v13003;	// L20069
        int8_t v13005 = v12994.read();	// L20070
        int8_t b235;	// L20071
        b235 = v13005;	// L20072
        int8_t v13007 = a233;	// L20073
        int8_t v13008 = b235;	// L20074
        int16_t v13009 = v13007;	// L20075
        int16_t v13010 = v13008;	// L20076
        int16_t v13011 = v13009 * v13010;	// L20077
        int8_t v13012 = c219;	// L20078
        ap_int<17> v13013 = v13012;	// L20079
        ap_int<17> v13014 = v13011;	// L20080
        ap_int<17> v13015 = v13013 + v13014;	// L20081
        int8_t v13016 = v13015;	// L20082
        c219 = v13016;	// L20083
        int8_t v13017 = a233;	// L20084
        v12995.write(v13017);	// L20085
        int8_t v13018 = b235;	// L20086
        v12996.write(v13018);	// L20087
      }
      ap_uint<128> packed_tmp219;	// L20089
      ap_uint<128> v13020 = v12997.read();	// L20090
      packed_tmp219 = v13020;	// L20091
      ap_uint<128> packed_c219;	// L20092
      packed_c219 = 0;	// L20093
      l_S_m_1_m219: for (int m219 = 0; m219 < 16; m219++) {	// L20094
        ap_int<33> v13023 = m219;	// L20095
        bool v13024 = v13023 == 13;	// L20096
        if (v13024) {	// L20097
          int8_t v13025 = c219;	// L20098
          ap_int<128> v13026 = packed_c219;	// L20099
          int64_t v13027 = m219;	// L20100
          int64_t v13028 = v13027 * 8;	// L20101
          ap_int<34> v13029 = m219;	// L20102
          ap_int<34> v13030 = v13029 + 1;	// L20103
          ap_int<66> v13031 = v13030;	// L20104
          ap_int<66> v13032 = v13031 * 8;	// L20105
          ap_int<66> v13033 = v13032 - 1;	// L20106
          int v13034 = v13028;	// L20107
          int v13035 = v13033;	// L20108
          ap_int<128> v13036;
          ap_int<128> v13036_tmp = v13026;
          v13036_tmp(v13035, v13034) = v13025;
          v13036 = v13036_tmp;	// L20109
          packed_c219 = v13036;	// L20110
        } else {
          ap_int<128> v13037 = packed_tmp219;	// L20112
          int64_t v13038 = m219;	// L20113
          int64_t v13039 = v13038 * 8;	// L20114
          ap_int<34> v13040 = m219;	// L20115
          ap_int<34> v13041 = v13040 + 1;	// L20116
          ap_int<66> v13042 = v13041;	// L20117
          ap_int<66> v13043 = v13042 * 8;	// L20118
          ap_int<66> v13044 = v13043 - 1;	// L20119
          int v13045 = v13039;	// L20120
          int v13046 = v13044;	// L20121
          int8_t v13047;
          ap_int<128> v13047_tmp = v13037;
          v13047 = v13047_tmp(v13046, v13045);	// L20122
          ap_int<128> v13048 = packed_c219;	// L20123
          ap_int<128> v13049;
          ap_int<128> v13049_tmp = v13048;
          v13049_tmp(v13046, v13045) = v13047;
          v13049 = v13049_tmp;	// L20133
          packed_c219 = v13049;	// L20134
        }
      }
      ap_int<128> v13050 = packed_c219;	// L20137
      v12998.write(v13050);	// L20138
    }
  }
}

void gemm_14_13(
  hls::stream< int8_t >& v13051,
  hls::stream< int8_t >& v13052,
  hls::stream< int8_t >& v13053,
  hls::stream< int8_t >& v13054,
  hls::stream< ap_uint<128> >& v13055,
  hls::stream< ap_uint<128> >& v13056
) {	// L20143
  l_S_mt_nt_0_mt253: for (int mt253 = 0; mt253 < 4; mt253++) {	// L20151
    l_nt253: for (int nt253 = 0; nt253 < 4; nt253++) {	// L20152
      int8_t c220;	// L20153
      c220 = 0;	// L20154
      l_S_k_0_k253: for (int k253 = 0; k253 < 64; k253++) {	// L20155
        int8_t v13061 = v13051.read();	// L20156
        int8_t a234;	// L20157
        a234 = v13061;	// L20158
        int8_t v13063 = v13052.read();	// L20159
        int8_t b236;	// L20160
        b236 = v13063;	// L20161
        int8_t v13065 = a234;	// L20162
        int8_t v13066 = b236;	// L20163
        int16_t v13067 = v13065;	// L20164
        int16_t v13068 = v13066;	// L20165
        int16_t v13069 = v13067 * v13068;	// L20166
        int8_t v13070 = c220;	// L20167
        ap_int<17> v13071 = v13070;	// L20168
        ap_int<17> v13072 = v13069;	// L20169
        ap_int<17> v13073 = v13071 + v13072;	// L20170
        int8_t v13074 = v13073;	// L20171
        c220 = v13074;	// L20172
        int8_t v13075 = a234;	// L20173
        v13053.write(v13075);	// L20174
        int8_t v13076 = b236;	// L20175
        v13054.write(v13076);	// L20176
      }
      ap_uint<128> packed_tmp220;	// L20178
      ap_uint<128> v13078 = v13055.read();	// L20179
      packed_tmp220 = v13078;	// L20180
      ap_uint<128> packed_c220;	// L20181
      packed_c220 = 0;	// L20182
      l_S_m_1_m220: for (int m220 = 0; m220 < 16; m220++) {	// L20183
        ap_int<33> v13081 = m220;	// L20184
        bool v13082 = v13081 == 13;	// L20185
        if (v13082) {	// L20186
          int8_t v13083 = c220;	// L20187
          ap_int<128> v13084 = packed_c220;	// L20188
          int64_t v13085 = m220;	// L20189
          int64_t v13086 = v13085 * 8;	// L20190
          ap_int<34> v13087 = m220;	// L20191
          ap_int<34> v13088 = v13087 + 1;	// L20192
          ap_int<66> v13089 = v13088;	// L20193
          ap_int<66> v13090 = v13089 * 8;	// L20194
          ap_int<66> v13091 = v13090 - 1;	// L20195
          int v13092 = v13086;	// L20196
          int v13093 = v13091;	// L20197
          ap_int<128> v13094;
          ap_int<128> v13094_tmp = v13084;
          v13094_tmp(v13093, v13092) = v13083;
          v13094 = v13094_tmp;	// L20198
          packed_c220 = v13094;	// L20199
        } else {
          ap_int<128> v13095 = packed_tmp220;	// L20201
          int64_t v13096 = m220;	// L20202
          int64_t v13097 = v13096 * 8;	// L20203
          ap_int<34> v13098 = m220;	// L20204
          ap_int<34> v13099 = v13098 + 1;	// L20205
          ap_int<66> v13100 = v13099;	// L20206
          ap_int<66> v13101 = v13100 * 8;	// L20207
          ap_int<66> v13102 = v13101 - 1;	// L20208
          int v13103 = v13097;	// L20209
          int v13104 = v13102;	// L20210
          int8_t v13105;
          ap_int<128> v13105_tmp = v13095;
          v13105 = v13105_tmp(v13104, v13103);	// L20211
          ap_int<128> v13106 = packed_c220;	// L20212
          ap_int<128> v13107;
          ap_int<128> v13107_tmp = v13106;
          v13107_tmp(v13104, v13103) = v13105;
          v13107 = v13107_tmp;	// L20222
          packed_c220 = v13107;	// L20223
        }
      }
      ap_int<128> v13108 = packed_c220;	// L20226
      v13056.write(v13108);	// L20227
    }
  }
}

void gemm_14_14(
  hls::stream< int8_t >& v13109,
  hls::stream< int8_t >& v13110,
  hls::stream< int8_t >& v13111,
  hls::stream< int8_t >& v13112,
  hls::stream< ap_uint<128> >& v13113,
  hls::stream< ap_uint<128> >& v13114
) {	// L20232
  l_S_mt_nt_0_mt254: for (int mt254 = 0; mt254 < 4; mt254++) {	// L20240
    l_nt254: for (int nt254 = 0; nt254 < 4; nt254++) {	// L20241
      int8_t c221;	// L20242
      c221 = 0;	// L20243
      l_S_k_0_k254: for (int k254 = 0; k254 < 64; k254++) {	// L20244
        int8_t v13119 = v13109.read();	// L20245
        int8_t a235;	// L20246
        a235 = v13119;	// L20247
        int8_t v13121 = v13110.read();	// L20248
        int8_t b237;	// L20249
        b237 = v13121;	// L20250
        int8_t v13123 = a235;	// L20251
        int8_t v13124 = b237;	// L20252
        int16_t v13125 = v13123;	// L20253
        int16_t v13126 = v13124;	// L20254
        int16_t v13127 = v13125 * v13126;	// L20255
        int8_t v13128 = c221;	// L20256
        ap_int<17> v13129 = v13128;	// L20257
        ap_int<17> v13130 = v13127;	// L20258
        ap_int<17> v13131 = v13129 + v13130;	// L20259
        int8_t v13132 = v13131;	// L20260
        c221 = v13132;	// L20261
        int8_t v13133 = a235;	// L20262
        v13111.write(v13133);	// L20263
        int8_t v13134 = b237;	// L20264
        v13112.write(v13134);	// L20265
      }
      ap_uint<128> packed_tmp221;	// L20267
      ap_uint<128> v13136 = v13113.read();	// L20268
      packed_tmp221 = v13136;	// L20269
      ap_uint<128> packed_c221;	// L20270
      packed_c221 = 0;	// L20271
      l_S_m_1_m221: for (int m221 = 0; m221 < 16; m221++) {	// L20272
        ap_int<33> v13139 = m221;	// L20273
        bool v13140 = v13139 == 13;	// L20274
        if (v13140) {	// L20275
          int8_t v13141 = c221;	// L20276
          ap_int<128> v13142 = packed_c221;	// L20277
          int64_t v13143 = m221;	// L20278
          int64_t v13144 = v13143 * 8;	// L20279
          ap_int<34> v13145 = m221;	// L20280
          ap_int<34> v13146 = v13145 + 1;	// L20281
          ap_int<66> v13147 = v13146;	// L20282
          ap_int<66> v13148 = v13147 * 8;	// L20283
          ap_int<66> v13149 = v13148 - 1;	// L20284
          int v13150 = v13144;	// L20285
          int v13151 = v13149;	// L20286
          ap_int<128> v13152;
          ap_int<128> v13152_tmp = v13142;
          v13152_tmp(v13151, v13150) = v13141;
          v13152 = v13152_tmp;	// L20287
          packed_c221 = v13152;	// L20288
        } else {
          ap_int<128> v13153 = packed_tmp221;	// L20290
          int64_t v13154 = m221;	// L20291
          int64_t v13155 = v13154 * 8;	// L20292
          ap_int<34> v13156 = m221;	// L20293
          ap_int<34> v13157 = v13156 + 1;	// L20294
          ap_int<66> v13158 = v13157;	// L20295
          ap_int<66> v13159 = v13158 * 8;	// L20296
          ap_int<66> v13160 = v13159 - 1;	// L20297
          int v13161 = v13155;	// L20298
          int v13162 = v13160;	// L20299
          int8_t v13163;
          ap_int<128> v13163_tmp = v13153;
          v13163 = v13163_tmp(v13162, v13161);	// L20300
          ap_int<128> v13164 = packed_c221;	// L20301
          ap_int<128> v13165;
          ap_int<128> v13165_tmp = v13164;
          v13165_tmp(v13162, v13161) = v13163;
          v13165 = v13165_tmp;	// L20311
          packed_c221 = v13165;	// L20312
        }
      }
      ap_int<128> v13166 = packed_c221;	// L20315
      v13114.write(v13166);	// L20316
    }
  }
}

void gemm_14_15(
  hls::stream< int8_t >& v13167,
  hls::stream< int8_t >& v13168,
  hls::stream< int8_t >& v13169,
  hls::stream< int8_t >& v13170,
  hls::stream< ap_uint<128> >& v13171,
  hls::stream< ap_uint<128> >& v13172
) {	// L20321
  l_S_mt_nt_0_mt255: for (int mt255 = 0; mt255 < 4; mt255++) {	// L20329
    l_nt255: for (int nt255 = 0; nt255 < 4; nt255++) {	// L20330
      int8_t c222;	// L20331
      c222 = 0;	// L20332
      l_S_k_0_k255: for (int k255 = 0; k255 < 64; k255++) {	// L20333
        int8_t v13177 = v13167.read();	// L20334
        int8_t a236;	// L20335
        a236 = v13177;	// L20336
        int8_t v13179 = v13168.read();	// L20337
        int8_t b238;	// L20338
        b238 = v13179;	// L20339
        int8_t v13181 = a236;	// L20340
        int8_t v13182 = b238;	// L20341
        int16_t v13183 = v13181;	// L20342
        int16_t v13184 = v13182;	// L20343
        int16_t v13185 = v13183 * v13184;	// L20344
        int8_t v13186 = c222;	// L20345
        ap_int<17> v13187 = v13186;	// L20346
        ap_int<17> v13188 = v13185;	// L20347
        ap_int<17> v13189 = v13187 + v13188;	// L20348
        int8_t v13190 = v13189;	// L20349
        c222 = v13190;	// L20350
        int8_t v13191 = a236;	// L20351
        v13169.write(v13191);	// L20352
        int8_t v13192 = b238;	// L20353
        v13170.write(v13192);	// L20354
      }
      ap_uint<128> packed_tmp222;	// L20356
      ap_uint<128> v13194 = v13171.read();	// L20357
      packed_tmp222 = v13194;	// L20358
      ap_uint<128> packed_c222;	// L20359
      packed_c222 = 0;	// L20360
      l_S_m_1_m222: for (int m222 = 0; m222 < 16; m222++) {	// L20361
        ap_int<33> v13197 = m222;	// L20362
        bool v13198 = v13197 == 13;	// L20363
        if (v13198) {	// L20364
          int8_t v13199 = c222;	// L20365
          ap_int<128> v13200 = packed_c222;	// L20366
          int64_t v13201 = m222;	// L20367
          int64_t v13202 = v13201 * 8;	// L20368
          ap_int<34> v13203 = m222;	// L20369
          ap_int<34> v13204 = v13203 + 1;	// L20370
          ap_int<66> v13205 = v13204;	// L20371
          ap_int<66> v13206 = v13205 * 8;	// L20372
          ap_int<66> v13207 = v13206 - 1;	// L20373
          int v13208 = v13202;	// L20374
          int v13209 = v13207;	// L20375
          ap_int<128> v13210;
          ap_int<128> v13210_tmp = v13200;
          v13210_tmp(v13209, v13208) = v13199;
          v13210 = v13210_tmp;	// L20376
          packed_c222 = v13210;	// L20377
        } else {
          ap_int<128> v13211 = packed_tmp222;	// L20379
          int64_t v13212 = m222;	// L20380
          int64_t v13213 = v13212 * 8;	// L20381
          ap_int<34> v13214 = m222;	// L20382
          ap_int<34> v13215 = v13214 + 1;	// L20383
          ap_int<66> v13216 = v13215;	// L20384
          ap_int<66> v13217 = v13216 * 8;	// L20385
          ap_int<66> v13218 = v13217 - 1;	// L20386
          int v13219 = v13213;	// L20387
          int v13220 = v13218;	// L20388
          int8_t v13221;
          ap_int<128> v13221_tmp = v13211;
          v13221 = v13221_tmp(v13220, v13219);	// L20389
          ap_int<128> v13222 = packed_c222;	// L20390
          ap_int<128> v13223;
          ap_int<128> v13223_tmp = v13222;
          v13223_tmp(v13220, v13219) = v13221;
          v13223 = v13223_tmp;	// L20400
          packed_c222 = v13223;	// L20401
        }
      }
      ap_int<128> v13224 = packed_c222;	// L20404
      v13172.write(v13224);	// L20405
    }
  }
}

void gemm_14_16(
  hls::stream< int8_t >& v13225,
  hls::stream< int8_t >& v13226,
  hls::stream< int8_t >& v13227,
  hls::stream< ap_uint<128> >& v13228,
  hls::stream< ap_uint<128> >& v13229
) {	// L20410
  l_S_mt_nt_0_mt256: for (int mt256 = 0; mt256 < 4; mt256++) {	// L20418
    l_nt256: for (int nt256 = 0; nt256 < 4; nt256++) {	// L20419
      int8_t c223;	// L20420
      c223 = 0;	// L20421
      l_S_k_0_k256: for (int k256 = 0; k256 < 64; k256++) {	// L20422
        int8_t v13234 = v13225.read();	// L20423
        int8_t a237;	// L20424
        a237 = v13234;	// L20425
        int8_t v13236 = v13226.read();	// L20426
        int8_t b239;	// L20427
        b239 = v13236;	// L20428
        int8_t v13238 = a237;	// L20429
        int8_t v13239 = b239;	// L20430
        int16_t v13240 = v13238;	// L20431
        int16_t v13241 = v13239;	// L20432
        int16_t v13242 = v13240 * v13241;	// L20433
        int8_t v13243 = c223;	// L20434
        ap_int<17> v13244 = v13243;	// L20435
        ap_int<17> v13245 = v13242;	// L20436
        ap_int<17> v13246 = v13244 + v13245;	// L20437
        int8_t v13247 = v13246;	// L20438
        c223 = v13247;	// L20439
        int8_t v13248 = b239;	// L20440
        v13227.write(v13248);	// L20441
      }
      ap_uint<128> packed_tmp223;	// L20443
      ap_uint<128> v13250 = v13228.read();	// L20444
      packed_tmp223 = v13250;	// L20445
      ap_uint<128> packed_c223;	// L20446
      packed_c223 = 0;	// L20447
      l_S_m_1_m223: for (int m223 = 0; m223 < 16; m223++) {	// L20448
        ap_int<33> v13253 = m223;	// L20449
        bool v13254 = v13253 == 13;	// L20450
        if (v13254) {	// L20451
          int8_t v13255 = c223;	// L20452
          ap_int<128> v13256 = packed_c223;	// L20453
          int64_t v13257 = m223;	// L20454
          int64_t v13258 = v13257 * 8;	// L20455
          ap_int<34> v13259 = m223;	// L20456
          ap_int<34> v13260 = v13259 + 1;	// L20457
          ap_int<66> v13261 = v13260;	// L20458
          ap_int<66> v13262 = v13261 * 8;	// L20459
          ap_int<66> v13263 = v13262 - 1;	// L20460
          int v13264 = v13258;	// L20461
          int v13265 = v13263;	// L20462
          ap_int<128> v13266;
          ap_int<128> v13266_tmp = v13256;
          v13266_tmp(v13265, v13264) = v13255;
          v13266 = v13266_tmp;	// L20463
          packed_c223 = v13266;	// L20464
        } else {
          ap_int<128> v13267 = packed_tmp223;	// L20466
          int64_t v13268 = m223;	// L20467
          int64_t v13269 = v13268 * 8;	// L20468
          ap_int<34> v13270 = m223;	// L20469
          ap_int<34> v13271 = v13270 + 1;	// L20470
          ap_int<66> v13272 = v13271;	// L20471
          ap_int<66> v13273 = v13272 * 8;	// L20472
          ap_int<66> v13274 = v13273 - 1;	// L20473
          int v13275 = v13269;	// L20474
          int v13276 = v13274;	// L20475
          int8_t v13277;
          ap_int<128> v13277_tmp = v13267;
          v13277 = v13277_tmp(v13276, v13275);	// L20476
          ap_int<128> v13278 = packed_c223;	// L20477
          ap_int<128> v13279;
          ap_int<128> v13279_tmp = v13278;
          v13279_tmp(v13276, v13275) = v13277;
          v13279 = v13279_tmp;	// L20487
          packed_c223 = v13279;	// L20488
        }
      }
      ap_int<128> v13280 = packed_c223;	// L20491
      v13229.write(v13280);	// L20492
    }
  }
}

void gemm_15_0(
  hls::stream< ap_uint<128> >& v13281,
  hls::stream< int8_t >& v13282,
  hls::stream< ap_uint<128> >& v13283
) {	// L20497
  l_S_mt_nt_0_mt257: for (int mt257 = 0; mt257 < 4; mt257++) {	// L20500
    l_nt257: for (int nt257 = 0; nt257 < 4; nt257++) {	// L20501
      l_S_k_0_k257: for (int k257 = 0; k257 < 64; k257++) {	// L20502
        ap_uint<128> v13287 = v13281.read();	// L20503
        ap_uint<128> a238;	// L20504
        a238 = v13287;	// L20505
        ap_int<128> v13289 = a238;	// L20506
        int8_t v13290;
        ap_int<128> v13290_tmp = v13289;
        v13290 = v13290_tmp(119, 112);	// L20507
        v13282.write(v13290);	// L20508
        ap_int<128> v13291 = a238;	// L20509
        v13283.write(v13291);	// L20510
      }
    }
  }
}

void gemm_15_1(
  hls::stream< int8_t >& v13292,
  hls::stream< int8_t >& v13293,
  hls::stream< int8_t >& v13294,
  hls::stream< int8_t >& v13295,
  hls::stream< ap_uint<128> >& v13296,
  hls::stream< ap_uint<128> >& v13297
) {	// L20516
  l_S_mt_nt_0_mt258: for (int mt258 = 0; mt258 < 4; mt258++) {	// L20524
    l_nt258: for (int nt258 = 0; nt258 < 4; nt258++) {	// L20525
      int8_t c224;	// L20526
      c224 = 0;	// L20527
      l_S_k_0_k258: for (int k258 = 0; k258 < 64; k258++) {	// L20528
        int8_t v13302 = v13292.read();	// L20529
        int8_t a239;	// L20530
        a239 = v13302;	// L20531
        int8_t v13304 = v13293.read();	// L20532
        int8_t b240;	// L20533
        b240 = v13304;	// L20534
        int8_t v13306 = a239;	// L20535
        int8_t v13307 = b240;	// L20536
        int16_t v13308 = v13306;	// L20537
        int16_t v13309 = v13307;	// L20538
        int16_t v13310 = v13308 * v13309;	// L20539
        int8_t v13311 = c224;	// L20540
        ap_int<17> v13312 = v13311;	// L20541
        ap_int<17> v13313 = v13310;	// L20542
        ap_int<17> v13314 = v13312 + v13313;	// L20543
        int8_t v13315 = v13314;	// L20544
        c224 = v13315;	// L20545
        int8_t v13316 = a239;	// L20546
        v13294.write(v13316);	// L20547
        int8_t v13317 = b240;	// L20548
        v13295.write(v13317);	// L20549
      }
      ap_uint<128> packed_tmp224;	// L20551
      ap_uint<128> v13319 = v13296.read();	// L20552
      packed_tmp224 = v13319;	// L20553
      ap_uint<128> packed_c224;	// L20554
      packed_c224 = 0;	// L20555
      l_S_m_1_m224: for (int m224 = 0; m224 < 16; m224++) {	// L20556
        ap_int<33> v13322 = m224;	// L20557
        bool v13323 = v13322 == 14;	// L20558
        if (v13323) {	// L20559
          int8_t v13324 = c224;	// L20560
          ap_int<128> v13325 = packed_c224;	// L20561
          int64_t v13326 = m224;	// L20562
          int64_t v13327 = v13326 * 8;	// L20563
          ap_int<34> v13328 = m224;	// L20564
          ap_int<34> v13329 = v13328 + 1;	// L20565
          ap_int<66> v13330 = v13329;	// L20566
          ap_int<66> v13331 = v13330 * 8;	// L20567
          ap_int<66> v13332 = v13331 - 1;	// L20568
          int v13333 = v13327;	// L20569
          int v13334 = v13332;	// L20570
          ap_int<128> v13335;
          ap_int<128> v13335_tmp = v13325;
          v13335_tmp(v13334, v13333) = v13324;
          v13335 = v13335_tmp;	// L20571
          packed_c224 = v13335;	// L20572
        } else {
          ap_int<128> v13336 = packed_tmp224;	// L20574
          int64_t v13337 = m224;	// L20575
          int64_t v13338 = v13337 * 8;	// L20576
          ap_int<34> v13339 = m224;	// L20577
          ap_int<34> v13340 = v13339 + 1;	// L20578
          ap_int<66> v13341 = v13340;	// L20579
          ap_int<66> v13342 = v13341 * 8;	// L20580
          ap_int<66> v13343 = v13342 - 1;	// L20581
          int v13344 = v13338;	// L20582
          int v13345 = v13343;	// L20583
          int8_t v13346;
          ap_int<128> v13346_tmp = v13336;
          v13346 = v13346_tmp(v13345, v13344);	// L20584
          ap_int<128> v13347 = packed_c224;	// L20585
          ap_int<128> v13348;
          ap_int<128> v13348_tmp = v13347;
          v13348_tmp(v13345, v13344) = v13346;
          v13348 = v13348_tmp;	// L20595
          packed_c224 = v13348;	// L20596
        }
      }
      ap_int<128> v13349 = packed_c224;	// L20599
      v13297.write(v13349);	// L20600
    }
  }
}

void gemm_15_2(
  hls::stream< int8_t >& v13350,
  hls::stream< int8_t >& v13351,
  hls::stream< int8_t >& v13352,
  hls::stream< int8_t >& v13353,
  hls::stream< ap_uint<128> >& v13354,
  hls::stream< ap_uint<128> >& v13355
) {	// L20605
  l_S_mt_nt_0_mt259: for (int mt259 = 0; mt259 < 4; mt259++) {	// L20613
    l_nt259: for (int nt259 = 0; nt259 < 4; nt259++) {	// L20614
      int8_t c225;	// L20615
      c225 = 0;	// L20616
      l_S_k_0_k259: for (int k259 = 0; k259 < 64; k259++) {	// L20617
        int8_t v13360 = v13350.read();	// L20618
        int8_t a240;	// L20619
        a240 = v13360;	// L20620
        int8_t v13362 = v13351.read();	// L20621
        int8_t b241;	// L20622
        b241 = v13362;	// L20623
        int8_t v13364 = a240;	// L20624
        int8_t v13365 = b241;	// L20625
        int16_t v13366 = v13364;	// L20626
        int16_t v13367 = v13365;	// L20627
        int16_t v13368 = v13366 * v13367;	// L20628
        int8_t v13369 = c225;	// L20629
        ap_int<17> v13370 = v13369;	// L20630
        ap_int<17> v13371 = v13368;	// L20631
        ap_int<17> v13372 = v13370 + v13371;	// L20632
        int8_t v13373 = v13372;	// L20633
        c225 = v13373;	// L20634
        int8_t v13374 = a240;	// L20635
        v13352.write(v13374);	// L20636
        int8_t v13375 = b241;	// L20637
        v13353.write(v13375);	// L20638
      }
      ap_uint<128> packed_tmp225;	// L20640
      ap_uint<128> v13377 = v13354.read();	// L20641
      packed_tmp225 = v13377;	// L20642
      ap_uint<128> packed_c225;	// L20643
      packed_c225 = 0;	// L20644
      l_S_m_1_m225: for (int m225 = 0; m225 < 16; m225++) {	// L20645
        ap_int<33> v13380 = m225;	// L20646
        bool v13381 = v13380 == 14;	// L20647
        if (v13381) {	// L20648
          int8_t v13382 = c225;	// L20649
          ap_int<128> v13383 = packed_c225;	// L20650
          int64_t v13384 = m225;	// L20651
          int64_t v13385 = v13384 * 8;	// L20652
          ap_int<34> v13386 = m225;	// L20653
          ap_int<34> v13387 = v13386 + 1;	// L20654
          ap_int<66> v13388 = v13387;	// L20655
          ap_int<66> v13389 = v13388 * 8;	// L20656
          ap_int<66> v13390 = v13389 - 1;	// L20657
          int v13391 = v13385;	// L20658
          int v13392 = v13390;	// L20659
          ap_int<128> v13393;
          ap_int<128> v13393_tmp = v13383;
          v13393_tmp(v13392, v13391) = v13382;
          v13393 = v13393_tmp;	// L20660
          packed_c225 = v13393;	// L20661
        } else {
          ap_int<128> v13394 = packed_tmp225;	// L20663
          int64_t v13395 = m225;	// L20664
          int64_t v13396 = v13395 * 8;	// L20665
          ap_int<34> v13397 = m225;	// L20666
          ap_int<34> v13398 = v13397 + 1;	// L20667
          ap_int<66> v13399 = v13398;	// L20668
          ap_int<66> v13400 = v13399 * 8;	// L20669
          ap_int<66> v13401 = v13400 - 1;	// L20670
          int v13402 = v13396;	// L20671
          int v13403 = v13401;	// L20672
          int8_t v13404;
          ap_int<128> v13404_tmp = v13394;
          v13404 = v13404_tmp(v13403, v13402);	// L20673
          ap_int<128> v13405 = packed_c225;	// L20674
          ap_int<128> v13406;
          ap_int<128> v13406_tmp = v13405;
          v13406_tmp(v13403, v13402) = v13404;
          v13406 = v13406_tmp;	// L20684
          packed_c225 = v13406;	// L20685
        }
      }
      ap_int<128> v13407 = packed_c225;	// L20688
      v13355.write(v13407);	// L20689
    }
  }
}

void gemm_15_3(
  hls::stream< int8_t >& v13408,
  hls::stream< int8_t >& v13409,
  hls::stream< int8_t >& v13410,
  hls::stream< int8_t >& v13411,
  hls::stream< ap_uint<128> >& v13412,
  hls::stream< ap_uint<128> >& v13413
) {	// L20694
  l_S_mt_nt_0_mt260: for (int mt260 = 0; mt260 < 4; mt260++) {	// L20702
    l_nt260: for (int nt260 = 0; nt260 < 4; nt260++) {	// L20703
      int8_t c226;	// L20704
      c226 = 0;	// L20705
      l_S_k_0_k260: for (int k260 = 0; k260 < 64; k260++) {	// L20706
        int8_t v13418 = v13408.read();	// L20707
        int8_t a241;	// L20708
        a241 = v13418;	// L20709
        int8_t v13420 = v13409.read();	// L20710
        int8_t b242;	// L20711
        b242 = v13420;	// L20712
        int8_t v13422 = a241;	// L20713
        int8_t v13423 = b242;	// L20714
        int16_t v13424 = v13422;	// L20715
        int16_t v13425 = v13423;	// L20716
        int16_t v13426 = v13424 * v13425;	// L20717
        int8_t v13427 = c226;	// L20718
        ap_int<17> v13428 = v13427;	// L20719
        ap_int<17> v13429 = v13426;	// L20720
        ap_int<17> v13430 = v13428 + v13429;	// L20721
        int8_t v13431 = v13430;	// L20722
        c226 = v13431;	// L20723
        int8_t v13432 = a241;	// L20724
        v13410.write(v13432);	// L20725
        int8_t v13433 = b242;	// L20726
        v13411.write(v13433);	// L20727
      }
      ap_uint<128> packed_tmp226;	// L20729
      ap_uint<128> v13435 = v13412.read();	// L20730
      packed_tmp226 = v13435;	// L20731
      ap_uint<128> packed_c226;	// L20732
      packed_c226 = 0;	// L20733
      l_S_m_1_m226: for (int m226 = 0; m226 < 16; m226++) {	// L20734
        ap_int<33> v13438 = m226;	// L20735
        bool v13439 = v13438 == 14;	// L20736
        if (v13439) {	// L20737
          int8_t v13440 = c226;	// L20738
          ap_int<128> v13441 = packed_c226;	// L20739
          int64_t v13442 = m226;	// L20740
          int64_t v13443 = v13442 * 8;	// L20741
          ap_int<34> v13444 = m226;	// L20742
          ap_int<34> v13445 = v13444 + 1;	// L20743
          ap_int<66> v13446 = v13445;	// L20744
          ap_int<66> v13447 = v13446 * 8;	// L20745
          ap_int<66> v13448 = v13447 - 1;	// L20746
          int v13449 = v13443;	// L20747
          int v13450 = v13448;	// L20748
          ap_int<128> v13451;
          ap_int<128> v13451_tmp = v13441;
          v13451_tmp(v13450, v13449) = v13440;
          v13451 = v13451_tmp;	// L20749
          packed_c226 = v13451;	// L20750
        } else {
          ap_int<128> v13452 = packed_tmp226;	// L20752
          int64_t v13453 = m226;	// L20753
          int64_t v13454 = v13453 * 8;	// L20754
          ap_int<34> v13455 = m226;	// L20755
          ap_int<34> v13456 = v13455 + 1;	// L20756
          ap_int<66> v13457 = v13456;	// L20757
          ap_int<66> v13458 = v13457 * 8;	// L20758
          ap_int<66> v13459 = v13458 - 1;	// L20759
          int v13460 = v13454;	// L20760
          int v13461 = v13459;	// L20761
          int8_t v13462;
          ap_int<128> v13462_tmp = v13452;
          v13462 = v13462_tmp(v13461, v13460);	// L20762
          ap_int<128> v13463 = packed_c226;	// L20763
          ap_int<128> v13464;
          ap_int<128> v13464_tmp = v13463;
          v13464_tmp(v13461, v13460) = v13462;
          v13464 = v13464_tmp;	// L20773
          packed_c226 = v13464;	// L20774
        }
      }
      ap_int<128> v13465 = packed_c226;	// L20777
      v13413.write(v13465);	// L20778
    }
  }
}

void gemm_15_4(
  hls::stream< int8_t >& v13466,
  hls::stream< int8_t >& v13467,
  hls::stream< int8_t >& v13468,
  hls::stream< int8_t >& v13469,
  hls::stream< ap_uint<128> >& v13470,
  hls::stream< ap_uint<128> >& v13471
) {	// L20783
  l_S_mt_nt_0_mt261: for (int mt261 = 0; mt261 < 4; mt261++) {	// L20791
    l_nt261: for (int nt261 = 0; nt261 < 4; nt261++) {	// L20792
      int8_t c227;	// L20793
      c227 = 0;	// L20794
      l_S_k_0_k261: for (int k261 = 0; k261 < 64; k261++) {	// L20795
        int8_t v13476 = v13466.read();	// L20796
        int8_t a242;	// L20797
        a242 = v13476;	// L20798
        int8_t v13478 = v13467.read();	// L20799
        int8_t b243;	// L20800
        b243 = v13478;	// L20801
        int8_t v13480 = a242;	// L20802
        int8_t v13481 = b243;	// L20803
        int16_t v13482 = v13480;	// L20804
        int16_t v13483 = v13481;	// L20805
        int16_t v13484 = v13482 * v13483;	// L20806
        int8_t v13485 = c227;	// L20807
        ap_int<17> v13486 = v13485;	// L20808
        ap_int<17> v13487 = v13484;	// L20809
        ap_int<17> v13488 = v13486 + v13487;	// L20810
        int8_t v13489 = v13488;	// L20811
        c227 = v13489;	// L20812
        int8_t v13490 = a242;	// L20813
        v13468.write(v13490);	// L20814
        int8_t v13491 = b243;	// L20815
        v13469.write(v13491);	// L20816
      }
      ap_uint<128> packed_tmp227;	// L20818
      ap_uint<128> v13493 = v13470.read();	// L20819
      packed_tmp227 = v13493;	// L20820
      ap_uint<128> packed_c227;	// L20821
      packed_c227 = 0;	// L20822
      l_S_m_1_m227: for (int m227 = 0; m227 < 16; m227++) {	// L20823
        ap_int<33> v13496 = m227;	// L20824
        bool v13497 = v13496 == 14;	// L20825
        if (v13497) {	// L20826
          int8_t v13498 = c227;	// L20827
          ap_int<128> v13499 = packed_c227;	// L20828
          int64_t v13500 = m227;	// L20829
          int64_t v13501 = v13500 * 8;	// L20830
          ap_int<34> v13502 = m227;	// L20831
          ap_int<34> v13503 = v13502 + 1;	// L20832
          ap_int<66> v13504 = v13503;	// L20833
          ap_int<66> v13505 = v13504 * 8;	// L20834
          ap_int<66> v13506 = v13505 - 1;	// L20835
          int v13507 = v13501;	// L20836
          int v13508 = v13506;	// L20837
          ap_int<128> v13509;
          ap_int<128> v13509_tmp = v13499;
          v13509_tmp(v13508, v13507) = v13498;
          v13509 = v13509_tmp;	// L20838
          packed_c227 = v13509;	// L20839
        } else {
          ap_int<128> v13510 = packed_tmp227;	// L20841
          int64_t v13511 = m227;	// L20842
          int64_t v13512 = v13511 * 8;	// L20843
          ap_int<34> v13513 = m227;	// L20844
          ap_int<34> v13514 = v13513 + 1;	// L20845
          ap_int<66> v13515 = v13514;	// L20846
          ap_int<66> v13516 = v13515 * 8;	// L20847
          ap_int<66> v13517 = v13516 - 1;	// L20848
          int v13518 = v13512;	// L20849
          int v13519 = v13517;	// L20850
          int8_t v13520;
          ap_int<128> v13520_tmp = v13510;
          v13520 = v13520_tmp(v13519, v13518);	// L20851
          ap_int<128> v13521 = packed_c227;	// L20852
          ap_int<128> v13522;
          ap_int<128> v13522_tmp = v13521;
          v13522_tmp(v13519, v13518) = v13520;
          v13522 = v13522_tmp;	// L20862
          packed_c227 = v13522;	// L20863
        }
      }
      ap_int<128> v13523 = packed_c227;	// L20866
      v13471.write(v13523);	// L20867
    }
  }
}

void gemm_15_5(
  hls::stream< int8_t >& v13524,
  hls::stream< int8_t >& v13525,
  hls::stream< int8_t >& v13526,
  hls::stream< int8_t >& v13527,
  hls::stream< ap_uint<128> >& v13528,
  hls::stream< ap_uint<128> >& v13529
) {	// L20872
  l_S_mt_nt_0_mt262: for (int mt262 = 0; mt262 < 4; mt262++) {	// L20880
    l_nt262: for (int nt262 = 0; nt262 < 4; nt262++) {	// L20881
      int8_t c228;	// L20882
      c228 = 0;	// L20883
      l_S_k_0_k262: for (int k262 = 0; k262 < 64; k262++) {	// L20884
        int8_t v13534 = v13524.read();	// L20885
        int8_t a243;	// L20886
        a243 = v13534;	// L20887
        int8_t v13536 = v13525.read();	// L20888
        int8_t b244;	// L20889
        b244 = v13536;	// L20890
        int8_t v13538 = a243;	// L20891
        int8_t v13539 = b244;	// L20892
        int16_t v13540 = v13538;	// L20893
        int16_t v13541 = v13539;	// L20894
        int16_t v13542 = v13540 * v13541;	// L20895
        int8_t v13543 = c228;	// L20896
        ap_int<17> v13544 = v13543;	// L20897
        ap_int<17> v13545 = v13542;	// L20898
        ap_int<17> v13546 = v13544 + v13545;	// L20899
        int8_t v13547 = v13546;	// L20900
        c228 = v13547;	// L20901
        int8_t v13548 = a243;	// L20902
        v13526.write(v13548);	// L20903
        int8_t v13549 = b244;	// L20904
        v13527.write(v13549);	// L20905
      }
      ap_uint<128> packed_tmp228;	// L20907
      ap_uint<128> v13551 = v13528.read();	// L20908
      packed_tmp228 = v13551;	// L20909
      ap_uint<128> packed_c228;	// L20910
      packed_c228 = 0;	// L20911
      l_S_m_1_m228: for (int m228 = 0; m228 < 16; m228++) {	// L20912
        ap_int<33> v13554 = m228;	// L20913
        bool v13555 = v13554 == 14;	// L20914
        if (v13555) {	// L20915
          int8_t v13556 = c228;	// L20916
          ap_int<128> v13557 = packed_c228;	// L20917
          int64_t v13558 = m228;	// L20918
          int64_t v13559 = v13558 * 8;	// L20919
          ap_int<34> v13560 = m228;	// L20920
          ap_int<34> v13561 = v13560 + 1;	// L20921
          ap_int<66> v13562 = v13561;	// L20922
          ap_int<66> v13563 = v13562 * 8;	// L20923
          ap_int<66> v13564 = v13563 - 1;	// L20924
          int v13565 = v13559;	// L20925
          int v13566 = v13564;	// L20926
          ap_int<128> v13567;
          ap_int<128> v13567_tmp = v13557;
          v13567_tmp(v13566, v13565) = v13556;
          v13567 = v13567_tmp;	// L20927
          packed_c228 = v13567;	// L20928
        } else {
          ap_int<128> v13568 = packed_tmp228;	// L20930
          int64_t v13569 = m228;	// L20931
          int64_t v13570 = v13569 * 8;	// L20932
          ap_int<34> v13571 = m228;	// L20933
          ap_int<34> v13572 = v13571 + 1;	// L20934
          ap_int<66> v13573 = v13572;	// L20935
          ap_int<66> v13574 = v13573 * 8;	// L20936
          ap_int<66> v13575 = v13574 - 1;	// L20937
          int v13576 = v13570;	// L20938
          int v13577 = v13575;	// L20939
          int8_t v13578;
          ap_int<128> v13578_tmp = v13568;
          v13578 = v13578_tmp(v13577, v13576);	// L20940
          ap_int<128> v13579 = packed_c228;	// L20941
          ap_int<128> v13580;
          ap_int<128> v13580_tmp = v13579;
          v13580_tmp(v13577, v13576) = v13578;
          v13580 = v13580_tmp;	// L20951
          packed_c228 = v13580;	// L20952
        }
      }
      ap_int<128> v13581 = packed_c228;	// L20955
      v13529.write(v13581);	// L20956
    }
  }
}

void gemm_15_6(
  hls::stream< int8_t >& v13582,
  hls::stream< int8_t >& v13583,
  hls::stream< int8_t >& v13584,
  hls::stream< int8_t >& v13585,
  hls::stream< ap_uint<128> >& v13586,
  hls::stream< ap_uint<128> >& v13587
) {	// L20961
  l_S_mt_nt_0_mt263: for (int mt263 = 0; mt263 < 4; mt263++) {	// L20969
    l_nt263: for (int nt263 = 0; nt263 < 4; nt263++) {	// L20970
      int8_t c229;	// L20971
      c229 = 0;	// L20972
      l_S_k_0_k263: for (int k263 = 0; k263 < 64; k263++) {	// L20973
        int8_t v13592 = v13582.read();	// L20974
        int8_t a244;	// L20975
        a244 = v13592;	// L20976
        int8_t v13594 = v13583.read();	// L20977
        int8_t b245;	// L20978
        b245 = v13594;	// L20979
        int8_t v13596 = a244;	// L20980
        int8_t v13597 = b245;	// L20981
        int16_t v13598 = v13596;	// L20982
        int16_t v13599 = v13597;	// L20983
        int16_t v13600 = v13598 * v13599;	// L20984
        int8_t v13601 = c229;	// L20985
        ap_int<17> v13602 = v13601;	// L20986
        ap_int<17> v13603 = v13600;	// L20987
        ap_int<17> v13604 = v13602 + v13603;	// L20988
        int8_t v13605 = v13604;	// L20989
        c229 = v13605;	// L20990
        int8_t v13606 = a244;	// L20991
        v13584.write(v13606);	// L20992
        int8_t v13607 = b245;	// L20993
        v13585.write(v13607);	// L20994
      }
      ap_uint<128> packed_tmp229;	// L20996
      ap_uint<128> v13609 = v13586.read();	// L20997
      packed_tmp229 = v13609;	// L20998
      ap_uint<128> packed_c229;	// L20999
      packed_c229 = 0;	// L21000
      l_S_m_1_m229: for (int m229 = 0; m229 < 16; m229++) {	// L21001
        ap_int<33> v13612 = m229;	// L21002
        bool v13613 = v13612 == 14;	// L21003
        if (v13613) {	// L21004
          int8_t v13614 = c229;	// L21005
          ap_int<128> v13615 = packed_c229;	// L21006
          int64_t v13616 = m229;	// L21007
          int64_t v13617 = v13616 * 8;	// L21008
          ap_int<34> v13618 = m229;	// L21009
          ap_int<34> v13619 = v13618 + 1;	// L21010
          ap_int<66> v13620 = v13619;	// L21011
          ap_int<66> v13621 = v13620 * 8;	// L21012
          ap_int<66> v13622 = v13621 - 1;	// L21013
          int v13623 = v13617;	// L21014
          int v13624 = v13622;	// L21015
          ap_int<128> v13625;
          ap_int<128> v13625_tmp = v13615;
          v13625_tmp(v13624, v13623) = v13614;
          v13625 = v13625_tmp;	// L21016
          packed_c229 = v13625;	// L21017
        } else {
          ap_int<128> v13626 = packed_tmp229;	// L21019
          int64_t v13627 = m229;	// L21020
          int64_t v13628 = v13627 * 8;	// L21021
          ap_int<34> v13629 = m229;	// L21022
          ap_int<34> v13630 = v13629 + 1;	// L21023
          ap_int<66> v13631 = v13630;	// L21024
          ap_int<66> v13632 = v13631 * 8;	// L21025
          ap_int<66> v13633 = v13632 - 1;	// L21026
          int v13634 = v13628;	// L21027
          int v13635 = v13633;	// L21028
          int8_t v13636;
          ap_int<128> v13636_tmp = v13626;
          v13636 = v13636_tmp(v13635, v13634);	// L21029
          ap_int<128> v13637 = packed_c229;	// L21030
          ap_int<128> v13638;
          ap_int<128> v13638_tmp = v13637;
          v13638_tmp(v13635, v13634) = v13636;
          v13638 = v13638_tmp;	// L21040
          packed_c229 = v13638;	// L21041
        }
      }
      ap_int<128> v13639 = packed_c229;	// L21044
      v13587.write(v13639);	// L21045
    }
  }
}

void gemm_15_7(
  hls::stream< int8_t >& v13640,
  hls::stream< int8_t >& v13641,
  hls::stream< int8_t >& v13642,
  hls::stream< int8_t >& v13643,
  hls::stream< ap_uint<128> >& v13644,
  hls::stream< ap_uint<128> >& v13645
) {	// L21050
  l_S_mt_nt_0_mt264: for (int mt264 = 0; mt264 < 4; mt264++) {	// L21058
    l_nt264: for (int nt264 = 0; nt264 < 4; nt264++) {	// L21059
      int8_t c230;	// L21060
      c230 = 0;	// L21061
      l_S_k_0_k264: for (int k264 = 0; k264 < 64; k264++) {	// L21062
        int8_t v13650 = v13640.read();	// L21063
        int8_t a245;	// L21064
        a245 = v13650;	// L21065
        int8_t v13652 = v13641.read();	// L21066
        int8_t b246;	// L21067
        b246 = v13652;	// L21068
        int8_t v13654 = a245;	// L21069
        int8_t v13655 = b246;	// L21070
        int16_t v13656 = v13654;	// L21071
        int16_t v13657 = v13655;	// L21072
        int16_t v13658 = v13656 * v13657;	// L21073
        int8_t v13659 = c230;	// L21074
        ap_int<17> v13660 = v13659;	// L21075
        ap_int<17> v13661 = v13658;	// L21076
        ap_int<17> v13662 = v13660 + v13661;	// L21077
        int8_t v13663 = v13662;	// L21078
        c230 = v13663;	// L21079
        int8_t v13664 = a245;	// L21080
        v13642.write(v13664);	// L21081
        int8_t v13665 = b246;	// L21082
        v13643.write(v13665);	// L21083
      }
      ap_uint<128> packed_tmp230;	// L21085
      ap_uint<128> v13667 = v13644.read();	// L21086
      packed_tmp230 = v13667;	// L21087
      ap_uint<128> packed_c230;	// L21088
      packed_c230 = 0;	// L21089
      l_S_m_1_m230: for (int m230 = 0; m230 < 16; m230++) {	// L21090
        ap_int<33> v13670 = m230;	// L21091
        bool v13671 = v13670 == 14;	// L21092
        if (v13671) {	// L21093
          int8_t v13672 = c230;	// L21094
          ap_int<128> v13673 = packed_c230;	// L21095
          int64_t v13674 = m230;	// L21096
          int64_t v13675 = v13674 * 8;	// L21097
          ap_int<34> v13676 = m230;	// L21098
          ap_int<34> v13677 = v13676 + 1;	// L21099
          ap_int<66> v13678 = v13677;	// L21100
          ap_int<66> v13679 = v13678 * 8;	// L21101
          ap_int<66> v13680 = v13679 - 1;	// L21102
          int v13681 = v13675;	// L21103
          int v13682 = v13680;	// L21104
          ap_int<128> v13683;
          ap_int<128> v13683_tmp = v13673;
          v13683_tmp(v13682, v13681) = v13672;
          v13683 = v13683_tmp;	// L21105
          packed_c230 = v13683;	// L21106
        } else {
          ap_int<128> v13684 = packed_tmp230;	// L21108
          int64_t v13685 = m230;	// L21109
          int64_t v13686 = v13685 * 8;	// L21110
          ap_int<34> v13687 = m230;	// L21111
          ap_int<34> v13688 = v13687 + 1;	// L21112
          ap_int<66> v13689 = v13688;	// L21113
          ap_int<66> v13690 = v13689 * 8;	// L21114
          ap_int<66> v13691 = v13690 - 1;	// L21115
          int v13692 = v13686;	// L21116
          int v13693 = v13691;	// L21117
          int8_t v13694;
          ap_int<128> v13694_tmp = v13684;
          v13694 = v13694_tmp(v13693, v13692);	// L21118
          ap_int<128> v13695 = packed_c230;	// L21119
          ap_int<128> v13696;
          ap_int<128> v13696_tmp = v13695;
          v13696_tmp(v13693, v13692) = v13694;
          v13696 = v13696_tmp;	// L21129
          packed_c230 = v13696;	// L21130
        }
      }
      ap_int<128> v13697 = packed_c230;	// L21133
      v13645.write(v13697);	// L21134
    }
  }
}

void gemm_15_8(
  hls::stream< int8_t >& v13698,
  hls::stream< int8_t >& v13699,
  hls::stream< int8_t >& v13700,
  hls::stream< int8_t >& v13701,
  hls::stream< ap_uint<128> >& v13702,
  hls::stream< ap_uint<128> >& v13703
) {	// L21139
  l_S_mt_nt_0_mt265: for (int mt265 = 0; mt265 < 4; mt265++) {	// L21147
    l_nt265: for (int nt265 = 0; nt265 < 4; nt265++) {	// L21148
      int8_t c231;	// L21149
      c231 = 0;	// L21150
      l_S_k_0_k265: for (int k265 = 0; k265 < 64; k265++) {	// L21151
        int8_t v13708 = v13698.read();	// L21152
        int8_t a246;	// L21153
        a246 = v13708;	// L21154
        int8_t v13710 = v13699.read();	// L21155
        int8_t b247;	// L21156
        b247 = v13710;	// L21157
        int8_t v13712 = a246;	// L21158
        int8_t v13713 = b247;	// L21159
        int16_t v13714 = v13712;	// L21160
        int16_t v13715 = v13713;	// L21161
        int16_t v13716 = v13714 * v13715;	// L21162
        int8_t v13717 = c231;	// L21163
        ap_int<17> v13718 = v13717;	// L21164
        ap_int<17> v13719 = v13716;	// L21165
        ap_int<17> v13720 = v13718 + v13719;	// L21166
        int8_t v13721 = v13720;	// L21167
        c231 = v13721;	// L21168
        int8_t v13722 = a246;	// L21169
        v13700.write(v13722);	// L21170
        int8_t v13723 = b247;	// L21171
        v13701.write(v13723);	// L21172
      }
      ap_uint<128> packed_tmp231;	// L21174
      ap_uint<128> v13725 = v13702.read();	// L21175
      packed_tmp231 = v13725;	// L21176
      ap_uint<128> packed_c231;	// L21177
      packed_c231 = 0;	// L21178
      l_S_m_1_m231: for (int m231 = 0; m231 < 16; m231++) {	// L21179
        ap_int<33> v13728 = m231;	// L21180
        bool v13729 = v13728 == 14;	// L21181
        if (v13729) {	// L21182
          int8_t v13730 = c231;	// L21183
          ap_int<128> v13731 = packed_c231;	// L21184
          int64_t v13732 = m231;	// L21185
          int64_t v13733 = v13732 * 8;	// L21186
          ap_int<34> v13734 = m231;	// L21187
          ap_int<34> v13735 = v13734 + 1;	// L21188
          ap_int<66> v13736 = v13735;	// L21189
          ap_int<66> v13737 = v13736 * 8;	// L21190
          ap_int<66> v13738 = v13737 - 1;	// L21191
          int v13739 = v13733;	// L21192
          int v13740 = v13738;	// L21193
          ap_int<128> v13741;
          ap_int<128> v13741_tmp = v13731;
          v13741_tmp(v13740, v13739) = v13730;
          v13741 = v13741_tmp;	// L21194
          packed_c231 = v13741;	// L21195
        } else {
          ap_int<128> v13742 = packed_tmp231;	// L21197
          int64_t v13743 = m231;	// L21198
          int64_t v13744 = v13743 * 8;	// L21199
          ap_int<34> v13745 = m231;	// L21200
          ap_int<34> v13746 = v13745 + 1;	// L21201
          ap_int<66> v13747 = v13746;	// L21202
          ap_int<66> v13748 = v13747 * 8;	// L21203
          ap_int<66> v13749 = v13748 - 1;	// L21204
          int v13750 = v13744;	// L21205
          int v13751 = v13749;	// L21206
          int8_t v13752;
          ap_int<128> v13752_tmp = v13742;
          v13752 = v13752_tmp(v13751, v13750);	// L21207
          ap_int<128> v13753 = packed_c231;	// L21208
          ap_int<128> v13754;
          ap_int<128> v13754_tmp = v13753;
          v13754_tmp(v13751, v13750) = v13752;
          v13754 = v13754_tmp;	// L21218
          packed_c231 = v13754;	// L21219
        }
      }
      ap_int<128> v13755 = packed_c231;	// L21222
      v13703.write(v13755);	// L21223
    }
  }
}

void gemm_15_9(
  hls::stream< int8_t >& v13756,
  hls::stream< int8_t >& v13757,
  hls::stream< int8_t >& v13758,
  hls::stream< int8_t >& v13759,
  hls::stream< ap_uint<128> >& v13760,
  hls::stream< ap_uint<128> >& v13761
) {	// L21228
  l_S_mt_nt_0_mt266: for (int mt266 = 0; mt266 < 4; mt266++) {	// L21236
    l_nt266: for (int nt266 = 0; nt266 < 4; nt266++) {	// L21237
      int8_t c232;	// L21238
      c232 = 0;	// L21239
      l_S_k_0_k266: for (int k266 = 0; k266 < 64; k266++) {	// L21240
        int8_t v13766 = v13756.read();	// L21241
        int8_t a247;	// L21242
        a247 = v13766;	// L21243
        int8_t v13768 = v13757.read();	// L21244
        int8_t b248;	// L21245
        b248 = v13768;	// L21246
        int8_t v13770 = a247;	// L21247
        int8_t v13771 = b248;	// L21248
        int16_t v13772 = v13770;	// L21249
        int16_t v13773 = v13771;	// L21250
        int16_t v13774 = v13772 * v13773;	// L21251
        int8_t v13775 = c232;	// L21252
        ap_int<17> v13776 = v13775;	// L21253
        ap_int<17> v13777 = v13774;	// L21254
        ap_int<17> v13778 = v13776 + v13777;	// L21255
        int8_t v13779 = v13778;	// L21256
        c232 = v13779;	// L21257
        int8_t v13780 = a247;	// L21258
        v13758.write(v13780);	// L21259
        int8_t v13781 = b248;	// L21260
        v13759.write(v13781);	// L21261
      }
      ap_uint<128> packed_tmp232;	// L21263
      ap_uint<128> v13783 = v13760.read();	// L21264
      packed_tmp232 = v13783;	// L21265
      ap_uint<128> packed_c232;	// L21266
      packed_c232 = 0;	// L21267
      l_S_m_1_m232: for (int m232 = 0; m232 < 16; m232++) {	// L21268
        ap_int<33> v13786 = m232;	// L21269
        bool v13787 = v13786 == 14;	// L21270
        if (v13787) {	// L21271
          int8_t v13788 = c232;	// L21272
          ap_int<128> v13789 = packed_c232;	// L21273
          int64_t v13790 = m232;	// L21274
          int64_t v13791 = v13790 * 8;	// L21275
          ap_int<34> v13792 = m232;	// L21276
          ap_int<34> v13793 = v13792 + 1;	// L21277
          ap_int<66> v13794 = v13793;	// L21278
          ap_int<66> v13795 = v13794 * 8;	// L21279
          ap_int<66> v13796 = v13795 - 1;	// L21280
          int v13797 = v13791;	// L21281
          int v13798 = v13796;	// L21282
          ap_int<128> v13799;
          ap_int<128> v13799_tmp = v13789;
          v13799_tmp(v13798, v13797) = v13788;
          v13799 = v13799_tmp;	// L21283
          packed_c232 = v13799;	// L21284
        } else {
          ap_int<128> v13800 = packed_tmp232;	// L21286
          int64_t v13801 = m232;	// L21287
          int64_t v13802 = v13801 * 8;	// L21288
          ap_int<34> v13803 = m232;	// L21289
          ap_int<34> v13804 = v13803 + 1;	// L21290
          ap_int<66> v13805 = v13804;	// L21291
          ap_int<66> v13806 = v13805 * 8;	// L21292
          ap_int<66> v13807 = v13806 - 1;	// L21293
          int v13808 = v13802;	// L21294
          int v13809 = v13807;	// L21295
          int8_t v13810;
          ap_int<128> v13810_tmp = v13800;
          v13810 = v13810_tmp(v13809, v13808);	// L21296
          ap_int<128> v13811 = packed_c232;	// L21297
          ap_int<128> v13812;
          ap_int<128> v13812_tmp = v13811;
          v13812_tmp(v13809, v13808) = v13810;
          v13812 = v13812_tmp;	// L21307
          packed_c232 = v13812;	// L21308
        }
      }
      ap_int<128> v13813 = packed_c232;	// L21311
      v13761.write(v13813);	// L21312
    }
  }
}

void gemm_15_10(
  hls::stream< int8_t >& v13814,
  hls::stream< int8_t >& v13815,
  hls::stream< int8_t >& v13816,
  hls::stream< int8_t >& v13817,
  hls::stream< ap_uint<128> >& v13818,
  hls::stream< ap_uint<128> >& v13819
) {	// L21317
  l_S_mt_nt_0_mt267: for (int mt267 = 0; mt267 < 4; mt267++) {	// L21325
    l_nt267: for (int nt267 = 0; nt267 < 4; nt267++) {	// L21326
      int8_t c233;	// L21327
      c233 = 0;	// L21328
      l_S_k_0_k267: for (int k267 = 0; k267 < 64; k267++) {	// L21329
        int8_t v13824 = v13814.read();	// L21330
        int8_t a248;	// L21331
        a248 = v13824;	// L21332
        int8_t v13826 = v13815.read();	// L21333
        int8_t b249;	// L21334
        b249 = v13826;	// L21335
        int8_t v13828 = a248;	// L21336
        int8_t v13829 = b249;	// L21337
        int16_t v13830 = v13828;	// L21338
        int16_t v13831 = v13829;	// L21339
        int16_t v13832 = v13830 * v13831;	// L21340
        int8_t v13833 = c233;	// L21341
        ap_int<17> v13834 = v13833;	// L21342
        ap_int<17> v13835 = v13832;	// L21343
        ap_int<17> v13836 = v13834 + v13835;	// L21344
        int8_t v13837 = v13836;	// L21345
        c233 = v13837;	// L21346
        int8_t v13838 = a248;	// L21347
        v13816.write(v13838);	// L21348
        int8_t v13839 = b249;	// L21349
        v13817.write(v13839);	// L21350
      }
      ap_uint<128> packed_tmp233;	// L21352
      ap_uint<128> v13841 = v13818.read();	// L21353
      packed_tmp233 = v13841;	// L21354
      ap_uint<128> packed_c233;	// L21355
      packed_c233 = 0;	// L21356
      l_S_m_1_m233: for (int m233 = 0; m233 < 16; m233++) {	// L21357
        ap_int<33> v13844 = m233;	// L21358
        bool v13845 = v13844 == 14;	// L21359
        if (v13845) {	// L21360
          int8_t v13846 = c233;	// L21361
          ap_int<128> v13847 = packed_c233;	// L21362
          int64_t v13848 = m233;	// L21363
          int64_t v13849 = v13848 * 8;	// L21364
          ap_int<34> v13850 = m233;	// L21365
          ap_int<34> v13851 = v13850 + 1;	// L21366
          ap_int<66> v13852 = v13851;	// L21367
          ap_int<66> v13853 = v13852 * 8;	// L21368
          ap_int<66> v13854 = v13853 - 1;	// L21369
          int v13855 = v13849;	// L21370
          int v13856 = v13854;	// L21371
          ap_int<128> v13857;
          ap_int<128> v13857_tmp = v13847;
          v13857_tmp(v13856, v13855) = v13846;
          v13857 = v13857_tmp;	// L21372
          packed_c233 = v13857;	// L21373
        } else {
          ap_int<128> v13858 = packed_tmp233;	// L21375
          int64_t v13859 = m233;	// L21376
          int64_t v13860 = v13859 * 8;	// L21377
          ap_int<34> v13861 = m233;	// L21378
          ap_int<34> v13862 = v13861 + 1;	// L21379
          ap_int<66> v13863 = v13862;	// L21380
          ap_int<66> v13864 = v13863 * 8;	// L21381
          ap_int<66> v13865 = v13864 - 1;	// L21382
          int v13866 = v13860;	// L21383
          int v13867 = v13865;	// L21384
          int8_t v13868;
          ap_int<128> v13868_tmp = v13858;
          v13868 = v13868_tmp(v13867, v13866);	// L21385
          ap_int<128> v13869 = packed_c233;	// L21386
          ap_int<128> v13870;
          ap_int<128> v13870_tmp = v13869;
          v13870_tmp(v13867, v13866) = v13868;
          v13870 = v13870_tmp;	// L21396
          packed_c233 = v13870;	// L21397
        }
      }
      ap_int<128> v13871 = packed_c233;	// L21400
      v13819.write(v13871);	// L21401
    }
  }
}

void gemm_15_11(
  hls::stream< int8_t >& v13872,
  hls::stream< int8_t >& v13873,
  hls::stream< int8_t >& v13874,
  hls::stream< int8_t >& v13875,
  hls::stream< ap_uint<128> >& v13876,
  hls::stream< ap_uint<128> >& v13877
) {	// L21406
  l_S_mt_nt_0_mt268: for (int mt268 = 0; mt268 < 4; mt268++) {	// L21414
    l_nt268: for (int nt268 = 0; nt268 < 4; nt268++) {	// L21415
      int8_t c234;	// L21416
      c234 = 0;	// L21417
      l_S_k_0_k268: for (int k268 = 0; k268 < 64; k268++) {	// L21418
        int8_t v13882 = v13872.read();	// L21419
        int8_t a249;	// L21420
        a249 = v13882;	// L21421
        int8_t v13884 = v13873.read();	// L21422
        int8_t b250;	// L21423
        b250 = v13884;	// L21424
        int8_t v13886 = a249;	// L21425
        int8_t v13887 = b250;	// L21426
        int16_t v13888 = v13886;	// L21427
        int16_t v13889 = v13887;	// L21428
        int16_t v13890 = v13888 * v13889;	// L21429
        int8_t v13891 = c234;	// L21430
        ap_int<17> v13892 = v13891;	// L21431
        ap_int<17> v13893 = v13890;	// L21432
        ap_int<17> v13894 = v13892 + v13893;	// L21433
        int8_t v13895 = v13894;	// L21434
        c234 = v13895;	// L21435
        int8_t v13896 = a249;	// L21436
        v13874.write(v13896);	// L21437
        int8_t v13897 = b250;	// L21438
        v13875.write(v13897);	// L21439
      }
      ap_uint<128> packed_tmp234;	// L21441
      ap_uint<128> v13899 = v13876.read();	// L21442
      packed_tmp234 = v13899;	// L21443
      ap_uint<128> packed_c234;	// L21444
      packed_c234 = 0;	// L21445
      l_S_m_1_m234: for (int m234 = 0; m234 < 16; m234++) {	// L21446
        ap_int<33> v13902 = m234;	// L21447
        bool v13903 = v13902 == 14;	// L21448
        if (v13903) {	// L21449
          int8_t v13904 = c234;	// L21450
          ap_int<128> v13905 = packed_c234;	// L21451
          int64_t v13906 = m234;	// L21452
          int64_t v13907 = v13906 * 8;	// L21453
          ap_int<34> v13908 = m234;	// L21454
          ap_int<34> v13909 = v13908 + 1;	// L21455
          ap_int<66> v13910 = v13909;	// L21456
          ap_int<66> v13911 = v13910 * 8;	// L21457
          ap_int<66> v13912 = v13911 - 1;	// L21458
          int v13913 = v13907;	// L21459
          int v13914 = v13912;	// L21460
          ap_int<128> v13915;
          ap_int<128> v13915_tmp = v13905;
          v13915_tmp(v13914, v13913) = v13904;
          v13915 = v13915_tmp;	// L21461
          packed_c234 = v13915;	// L21462
        } else {
          ap_int<128> v13916 = packed_tmp234;	// L21464
          int64_t v13917 = m234;	// L21465
          int64_t v13918 = v13917 * 8;	// L21466
          ap_int<34> v13919 = m234;	// L21467
          ap_int<34> v13920 = v13919 + 1;	// L21468
          ap_int<66> v13921 = v13920;	// L21469
          ap_int<66> v13922 = v13921 * 8;	// L21470
          ap_int<66> v13923 = v13922 - 1;	// L21471
          int v13924 = v13918;	// L21472
          int v13925 = v13923;	// L21473
          int8_t v13926;
          ap_int<128> v13926_tmp = v13916;
          v13926 = v13926_tmp(v13925, v13924);	// L21474
          ap_int<128> v13927 = packed_c234;	// L21475
          ap_int<128> v13928;
          ap_int<128> v13928_tmp = v13927;
          v13928_tmp(v13925, v13924) = v13926;
          v13928 = v13928_tmp;	// L21485
          packed_c234 = v13928;	// L21486
        }
      }
      ap_int<128> v13929 = packed_c234;	// L21489
      v13877.write(v13929);	// L21490
    }
  }
}

void gemm_15_12(
  hls::stream< int8_t >& v13930,
  hls::stream< int8_t >& v13931,
  hls::stream< int8_t >& v13932,
  hls::stream< int8_t >& v13933,
  hls::stream< ap_uint<128> >& v13934,
  hls::stream< ap_uint<128> >& v13935
) {	// L21495
  l_S_mt_nt_0_mt269: for (int mt269 = 0; mt269 < 4; mt269++) {	// L21503
    l_nt269: for (int nt269 = 0; nt269 < 4; nt269++) {	// L21504
      int8_t c235;	// L21505
      c235 = 0;	// L21506
      l_S_k_0_k269: for (int k269 = 0; k269 < 64; k269++) {	// L21507
        int8_t v13940 = v13930.read();	// L21508
        int8_t a250;	// L21509
        a250 = v13940;	// L21510
        int8_t v13942 = v13931.read();	// L21511
        int8_t b251;	// L21512
        b251 = v13942;	// L21513
        int8_t v13944 = a250;	// L21514
        int8_t v13945 = b251;	// L21515
        int16_t v13946 = v13944;	// L21516
        int16_t v13947 = v13945;	// L21517
        int16_t v13948 = v13946 * v13947;	// L21518
        int8_t v13949 = c235;	// L21519
        ap_int<17> v13950 = v13949;	// L21520
        ap_int<17> v13951 = v13948;	// L21521
        ap_int<17> v13952 = v13950 + v13951;	// L21522
        int8_t v13953 = v13952;	// L21523
        c235 = v13953;	// L21524
        int8_t v13954 = a250;	// L21525
        v13932.write(v13954);	// L21526
        int8_t v13955 = b251;	// L21527
        v13933.write(v13955);	// L21528
      }
      ap_uint<128> packed_tmp235;	// L21530
      ap_uint<128> v13957 = v13934.read();	// L21531
      packed_tmp235 = v13957;	// L21532
      ap_uint<128> packed_c235;	// L21533
      packed_c235 = 0;	// L21534
      l_S_m_1_m235: for (int m235 = 0; m235 < 16; m235++) {	// L21535
        ap_int<33> v13960 = m235;	// L21536
        bool v13961 = v13960 == 14;	// L21537
        if (v13961) {	// L21538
          int8_t v13962 = c235;	// L21539
          ap_int<128> v13963 = packed_c235;	// L21540
          int64_t v13964 = m235;	// L21541
          int64_t v13965 = v13964 * 8;	// L21542
          ap_int<34> v13966 = m235;	// L21543
          ap_int<34> v13967 = v13966 + 1;	// L21544
          ap_int<66> v13968 = v13967;	// L21545
          ap_int<66> v13969 = v13968 * 8;	// L21546
          ap_int<66> v13970 = v13969 - 1;	// L21547
          int v13971 = v13965;	// L21548
          int v13972 = v13970;	// L21549
          ap_int<128> v13973;
          ap_int<128> v13973_tmp = v13963;
          v13973_tmp(v13972, v13971) = v13962;
          v13973 = v13973_tmp;	// L21550
          packed_c235 = v13973;	// L21551
        } else {
          ap_int<128> v13974 = packed_tmp235;	// L21553
          int64_t v13975 = m235;	// L21554
          int64_t v13976 = v13975 * 8;	// L21555
          ap_int<34> v13977 = m235;	// L21556
          ap_int<34> v13978 = v13977 + 1;	// L21557
          ap_int<66> v13979 = v13978;	// L21558
          ap_int<66> v13980 = v13979 * 8;	// L21559
          ap_int<66> v13981 = v13980 - 1;	// L21560
          int v13982 = v13976;	// L21561
          int v13983 = v13981;	// L21562
          int8_t v13984;
          ap_int<128> v13984_tmp = v13974;
          v13984 = v13984_tmp(v13983, v13982);	// L21563
          ap_int<128> v13985 = packed_c235;	// L21564
          ap_int<128> v13986;
          ap_int<128> v13986_tmp = v13985;
          v13986_tmp(v13983, v13982) = v13984;
          v13986 = v13986_tmp;	// L21574
          packed_c235 = v13986;	// L21575
        }
      }
      ap_int<128> v13987 = packed_c235;	// L21578
      v13935.write(v13987);	// L21579
    }
  }
}

void gemm_15_13(
  hls::stream< int8_t >& v13988,
  hls::stream< int8_t >& v13989,
  hls::stream< int8_t >& v13990,
  hls::stream< int8_t >& v13991,
  hls::stream< ap_uint<128> >& v13992,
  hls::stream< ap_uint<128> >& v13993
) {	// L21584
  l_S_mt_nt_0_mt270: for (int mt270 = 0; mt270 < 4; mt270++) {	// L21592
    l_nt270: for (int nt270 = 0; nt270 < 4; nt270++) {	// L21593
      int8_t c236;	// L21594
      c236 = 0;	// L21595
      l_S_k_0_k270: for (int k270 = 0; k270 < 64; k270++) {	// L21596
        int8_t v13998 = v13988.read();	// L21597
        int8_t a251;	// L21598
        a251 = v13998;	// L21599
        int8_t v14000 = v13989.read();	// L21600
        int8_t b252;	// L21601
        b252 = v14000;	// L21602
        int8_t v14002 = a251;	// L21603
        int8_t v14003 = b252;	// L21604
        int16_t v14004 = v14002;	// L21605
        int16_t v14005 = v14003;	// L21606
        int16_t v14006 = v14004 * v14005;	// L21607
        int8_t v14007 = c236;	// L21608
        ap_int<17> v14008 = v14007;	// L21609
        ap_int<17> v14009 = v14006;	// L21610
        ap_int<17> v14010 = v14008 + v14009;	// L21611
        int8_t v14011 = v14010;	// L21612
        c236 = v14011;	// L21613
        int8_t v14012 = a251;	// L21614
        v13990.write(v14012);	// L21615
        int8_t v14013 = b252;	// L21616
        v13991.write(v14013);	// L21617
      }
      ap_uint<128> packed_tmp236;	// L21619
      ap_uint<128> v14015 = v13992.read();	// L21620
      packed_tmp236 = v14015;	// L21621
      ap_uint<128> packed_c236;	// L21622
      packed_c236 = 0;	// L21623
      l_S_m_1_m236: for (int m236 = 0; m236 < 16; m236++) {	// L21624
        ap_int<33> v14018 = m236;	// L21625
        bool v14019 = v14018 == 14;	// L21626
        if (v14019) {	// L21627
          int8_t v14020 = c236;	// L21628
          ap_int<128> v14021 = packed_c236;	// L21629
          int64_t v14022 = m236;	// L21630
          int64_t v14023 = v14022 * 8;	// L21631
          ap_int<34> v14024 = m236;	// L21632
          ap_int<34> v14025 = v14024 + 1;	// L21633
          ap_int<66> v14026 = v14025;	// L21634
          ap_int<66> v14027 = v14026 * 8;	// L21635
          ap_int<66> v14028 = v14027 - 1;	// L21636
          int v14029 = v14023;	// L21637
          int v14030 = v14028;	// L21638
          ap_int<128> v14031;
          ap_int<128> v14031_tmp = v14021;
          v14031_tmp(v14030, v14029) = v14020;
          v14031 = v14031_tmp;	// L21639
          packed_c236 = v14031;	// L21640
        } else {
          ap_int<128> v14032 = packed_tmp236;	// L21642
          int64_t v14033 = m236;	// L21643
          int64_t v14034 = v14033 * 8;	// L21644
          ap_int<34> v14035 = m236;	// L21645
          ap_int<34> v14036 = v14035 + 1;	// L21646
          ap_int<66> v14037 = v14036;	// L21647
          ap_int<66> v14038 = v14037 * 8;	// L21648
          ap_int<66> v14039 = v14038 - 1;	// L21649
          int v14040 = v14034;	// L21650
          int v14041 = v14039;	// L21651
          int8_t v14042;
          ap_int<128> v14042_tmp = v14032;
          v14042 = v14042_tmp(v14041, v14040);	// L21652
          ap_int<128> v14043 = packed_c236;	// L21653
          ap_int<128> v14044;
          ap_int<128> v14044_tmp = v14043;
          v14044_tmp(v14041, v14040) = v14042;
          v14044 = v14044_tmp;	// L21663
          packed_c236 = v14044;	// L21664
        }
      }
      ap_int<128> v14045 = packed_c236;	// L21667
      v13993.write(v14045);	// L21668
    }
  }
}

void gemm_15_14(
  hls::stream< int8_t >& v14046,
  hls::stream< int8_t >& v14047,
  hls::stream< int8_t >& v14048,
  hls::stream< int8_t >& v14049,
  hls::stream< ap_uint<128> >& v14050,
  hls::stream< ap_uint<128> >& v14051
) {	// L21673
  l_S_mt_nt_0_mt271: for (int mt271 = 0; mt271 < 4; mt271++) {	// L21681
    l_nt271: for (int nt271 = 0; nt271 < 4; nt271++) {	// L21682
      int8_t c237;	// L21683
      c237 = 0;	// L21684
      l_S_k_0_k271: for (int k271 = 0; k271 < 64; k271++) {	// L21685
        int8_t v14056 = v14046.read();	// L21686
        int8_t a252;	// L21687
        a252 = v14056;	// L21688
        int8_t v14058 = v14047.read();	// L21689
        int8_t b253;	// L21690
        b253 = v14058;	// L21691
        int8_t v14060 = a252;	// L21692
        int8_t v14061 = b253;	// L21693
        int16_t v14062 = v14060;	// L21694
        int16_t v14063 = v14061;	// L21695
        int16_t v14064 = v14062 * v14063;	// L21696
        int8_t v14065 = c237;	// L21697
        ap_int<17> v14066 = v14065;	// L21698
        ap_int<17> v14067 = v14064;	// L21699
        ap_int<17> v14068 = v14066 + v14067;	// L21700
        int8_t v14069 = v14068;	// L21701
        c237 = v14069;	// L21702
        int8_t v14070 = a252;	// L21703
        v14048.write(v14070);	// L21704
        int8_t v14071 = b253;	// L21705
        v14049.write(v14071);	// L21706
      }
      ap_uint<128> packed_tmp237;	// L21708
      ap_uint<128> v14073 = v14050.read();	// L21709
      packed_tmp237 = v14073;	// L21710
      ap_uint<128> packed_c237;	// L21711
      packed_c237 = 0;	// L21712
      l_S_m_1_m237: for (int m237 = 0; m237 < 16; m237++) {	// L21713
        ap_int<33> v14076 = m237;	// L21714
        bool v14077 = v14076 == 14;	// L21715
        if (v14077) {	// L21716
          int8_t v14078 = c237;	// L21717
          ap_int<128> v14079 = packed_c237;	// L21718
          int64_t v14080 = m237;	// L21719
          int64_t v14081 = v14080 * 8;	// L21720
          ap_int<34> v14082 = m237;	// L21721
          ap_int<34> v14083 = v14082 + 1;	// L21722
          ap_int<66> v14084 = v14083;	// L21723
          ap_int<66> v14085 = v14084 * 8;	// L21724
          ap_int<66> v14086 = v14085 - 1;	// L21725
          int v14087 = v14081;	// L21726
          int v14088 = v14086;	// L21727
          ap_int<128> v14089;
          ap_int<128> v14089_tmp = v14079;
          v14089_tmp(v14088, v14087) = v14078;
          v14089 = v14089_tmp;	// L21728
          packed_c237 = v14089;	// L21729
        } else {
          ap_int<128> v14090 = packed_tmp237;	// L21731
          int64_t v14091 = m237;	// L21732
          int64_t v14092 = v14091 * 8;	// L21733
          ap_int<34> v14093 = m237;	// L21734
          ap_int<34> v14094 = v14093 + 1;	// L21735
          ap_int<66> v14095 = v14094;	// L21736
          ap_int<66> v14096 = v14095 * 8;	// L21737
          ap_int<66> v14097 = v14096 - 1;	// L21738
          int v14098 = v14092;	// L21739
          int v14099 = v14097;	// L21740
          int8_t v14100;
          ap_int<128> v14100_tmp = v14090;
          v14100 = v14100_tmp(v14099, v14098);	// L21741
          ap_int<128> v14101 = packed_c237;	// L21742
          ap_int<128> v14102;
          ap_int<128> v14102_tmp = v14101;
          v14102_tmp(v14099, v14098) = v14100;
          v14102 = v14102_tmp;	// L21752
          packed_c237 = v14102;	// L21753
        }
      }
      ap_int<128> v14103 = packed_c237;	// L21756
      v14051.write(v14103);	// L21757
    }
  }
}

void gemm_15_15(
  hls::stream< int8_t >& v14104,
  hls::stream< int8_t >& v14105,
  hls::stream< int8_t >& v14106,
  hls::stream< int8_t >& v14107,
  hls::stream< ap_uint<128> >& v14108,
  hls::stream< ap_uint<128> >& v14109
) {	// L21762
  l_S_mt_nt_0_mt272: for (int mt272 = 0; mt272 < 4; mt272++) {	// L21770
    l_nt272: for (int nt272 = 0; nt272 < 4; nt272++) {	// L21771
      int8_t c238;	// L21772
      c238 = 0;	// L21773
      l_S_k_0_k272: for (int k272 = 0; k272 < 64; k272++) {	// L21774
        int8_t v14114 = v14104.read();	// L21775
        int8_t a253;	// L21776
        a253 = v14114;	// L21777
        int8_t v14116 = v14105.read();	// L21778
        int8_t b254;	// L21779
        b254 = v14116;	// L21780
        int8_t v14118 = a253;	// L21781
        int8_t v14119 = b254;	// L21782
        int16_t v14120 = v14118;	// L21783
        int16_t v14121 = v14119;	// L21784
        int16_t v14122 = v14120 * v14121;	// L21785
        int8_t v14123 = c238;	// L21786
        ap_int<17> v14124 = v14123;	// L21787
        ap_int<17> v14125 = v14122;	// L21788
        ap_int<17> v14126 = v14124 + v14125;	// L21789
        int8_t v14127 = v14126;	// L21790
        c238 = v14127;	// L21791
        int8_t v14128 = a253;	// L21792
        v14106.write(v14128);	// L21793
        int8_t v14129 = b254;	// L21794
        v14107.write(v14129);	// L21795
      }
      ap_uint<128> packed_tmp238;	// L21797
      ap_uint<128> v14131 = v14108.read();	// L21798
      packed_tmp238 = v14131;	// L21799
      ap_uint<128> packed_c238;	// L21800
      packed_c238 = 0;	// L21801
      l_S_m_1_m238: for (int m238 = 0; m238 < 16; m238++) {	// L21802
        ap_int<33> v14134 = m238;	// L21803
        bool v14135 = v14134 == 14;	// L21804
        if (v14135) {	// L21805
          int8_t v14136 = c238;	// L21806
          ap_int<128> v14137 = packed_c238;	// L21807
          int64_t v14138 = m238;	// L21808
          int64_t v14139 = v14138 * 8;	// L21809
          ap_int<34> v14140 = m238;	// L21810
          ap_int<34> v14141 = v14140 + 1;	// L21811
          ap_int<66> v14142 = v14141;	// L21812
          ap_int<66> v14143 = v14142 * 8;	// L21813
          ap_int<66> v14144 = v14143 - 1;	// L21814
          int v14145 = v14139;	// L21815
          int v14146 = v14144;	// L21816
          ap_int<128> v14147;
          ap_int<128> v14147_tmp = v14137;
          v14147_tmp(v14146, v14145) = v14136;
          v14147 = v14147_tmp;	// L21817
          packed_c238 = v14147;	// L21818
        } else {
          ap_int<128> v14148 = packed_tmp238;	// L21820
          int64_t v14149 = m238;	// L21821
          int64_t v14150 = v14149 * 8;	// L21822
          ap_int<34> v14151 = m238;	// L21823
          ap_int<34> v14152 = v14151 + 1;	// L21824
          ap_int<66> v14153 = v14152;	// L21825
          ap_int<66> v14154 = v14153 * 8;	// L21826
          ap_int<66> v14155 = v14154 - 1;	// L21827
          int v14156 = v14150;	// L21828
          int v14157 = v14155;	// L21829
          int8_t v14158;
          ap_int<128> v14158_tmp = v14148;
          v14158 = v14158_tmp(v14157, v14156);	// L21830
          ap_int<128> v14159 = packed_c238;	// L21831
          ap_int<128> v14160;
          ap_int<128> v14160_tmp = v14159;
          v14160_tmp(v14157, v14156) = v14158;
          v14160 = v14160_tmp;	// L21841
          packed_c238 = v14160;	// L21842
        }
      }
      ap_int<128> v14161 = packed_c238;	// L21845
      v14109.write(v14161);	// L21846
    }
  }
}

void gemm_15_16(
  hls::stream< int8_t >& v14162,
  hls::stream< int8_t >& v14163,
  hls::stream< int8_t >& v14164,
  hls::stream< ap_uint<128> >& v14165,
  hls::stream< ap_uint<128> >& v14166
) {	// L21851
  l_S_mt_nt_0_mt273: for (int mt273 = 0; mt273 < 4; mt273++) {	// L21859
    l_nt273: for (int nt273 = 0; nt273 < 4; nt273++) {	// L21860
      int8_t c239;	// L21861
      c239 = 0;	// L21862
      l_S_k_0_k273: for (int k273 = 0; k273 < 64; k273++) {	// L21863
        int8_t v14171 = v14162.read();	// L21864
        int8_t a254;	// L21865
        a254 = v14171;	// L21866
        int8_t v14173 = v14163.read();	// L21867
        int8_t b255;	// L21868
        b255 = v14173;	// L21869
        int8_t v14175 = a254;	// L21870
        int8_t v14176 = b255;	// L21871
        int16_t v14177 = v14175;	// L21872
        int16_t v14178 = v14176;	// L21873
        int16_t v14179 = v14177 * v14178;	// L21874
        int8_t v14180 = c239;	// L21875
        ap_int<17> v14181 = v14180;	// L21876
        ap_int<17> v14182 = v14179;	// L21877
        ap_int<17> v14183 = v14181 + v14182;	// L21878
        int8_t v14184 = v14183;	// L21879
        c239 = v14184;	// L21880
        int8_t v14185 = b255;	// L21881
        v14164.write(v14185);	// L21882
      }
      ap_uint<128> packed_tmp239;	// L21884
      ap_uint<128> v14187 = v14165.read();	// L21885
      packed_tmp239 = v14187;	// L21886
      ap_uint<128> packed_c239;	// L21887
      packed_c239 = 0;	// L21888
      l_S_m_1_m239: for (int m239 = 0; m239 < 16; m239++) {	// L21889
        ap_int<33> v14190 = m239;	// L21890
        bool v14191 = v14190 == 14;	// L21891
        if (v14191) {	// L21892
          int8_t v14192 = c239;	// L21893
          ap_int<128> v14193 = packed_c239;	// L21894
          int64_t v14194 = m239;	// L21895
          int64_t v14195 = v14194 * 8;	// L21896
          ap_int<34> v14196 = m239;	// L21897
          ap_int<34> v14197 = v14196 + 1;	// L21898
          ap_int<66> v14198 = v14197;	// L21899
          ap_int<66> v14199 = v14198 * 8;	// L21900
          ap_int<66> v14200 = v14199 - 1;	// L21901
          int v14201 = v14195;	// L21902
          int v14202 = v14200;	// L21903
          ap_int<128> v14203;
          ap_int<128> v14203_tmp = v14193;
          v14203_tmp(v14202, v14201) = v14192;
          v14203 = v14203_tmp;	// L21904
          packed_c239 = v14203;	// L21905
        } else {
          ap_int<128> v14204 = packed_tmp239;	// L21907
          int64_t v14205 = m239;	// L21908
          int64_t v14206 = v14205 * 8;	// L21909
          ap_int<34> v14207 = m239;	// L21910
          ap_int<34> v14208 = v14207 + 1;	// L21911
          ap_int<66> v14209 = v14208;	// L21912
          ap_int<66> v14210 = v14209 * 8;	// L21913
          ap_int<66> v14211 = v14210 - 1;	// L21914
          int v14212 = v14206;	// L21915
          int v14213 = v14211;	// L21916
          int8_t v14214;
          ap_int<128> v14214_tmp = v14204;
          v14214 = v14214_tmp(v14213, v14212);	// L21917
          ap_int<128> v14215 = packed_c239;	// L21918
          ap_int<128> v14216;
          ap_int<128> v14216_tmp = v14215;
          v14216_tmp(v14213, v14212) = v14214;
          v14216 = v14216_tmp;	// L21928
          packed_c239 = v14216;	// L21929
        }
      }
      ap_int<128> v14217 = packed_c239;	// L21932
      v14166.write(v14217);	// L21933
    }
  }
}

void gemm_16_0(
  hls::stream< ap_uint<128> >& v14218,
  hls::stream< int8_t >& v14219
) {	// L21938
  l_S_mt_nt_0_mt274: for (int mt274 = 0; mt274 < 4; mt274++) {	// L21941
    l_nt274: for (int nt274 = 0; nt274 < 4; nt274++) {	// L21942
      l_S_k_0_k274: for (int k274 = 0; k274 < 64; k274++) {	// L21943
        ap_uint<128> v14223 = v14218.read();	// L21944
        ap_uint<128> a255;	// L21945
        a255 = v14223;	// L21946
        ap_int<128> v14225 = a255;	// L21947
        int8_t v14226;
        ap_int<128> v14226_tmp = v14225;
        v14226 = v14226_tmp(127, 120);	// L21948
        v14219.write(v14226);	// L21949
      }
    }
  }
}

void gemm_16_1(
  hls::stream< int8_t >& v14227,
  hls::stream< int8_t >& v14228,
  hls::stream< int8_t >& v14229,
  hls::stream< ap_uint<128> >& v14230,
  hls::stream< ap_uint<128> >& v14231
) {	// L21955
  l_S_mt_nt_0_mt275: for (int mt275 = 0; mt275 < 4; mt275++) {	// L21963
    l_nt275: for (int nt275 = 0; nt275 < 4; nt275++) {	// L21964
      int8_t c240;	// L21965
      c240 = 0;	// L21966
      l_S_k_0_k275: for (int k275 = 0; k275 < 64; k275++) {	// L21967
        int8_t v14236 = v14227.read();	// L21968
        int8_t a256;	// L21969
        a256 = v14236;	// L21970
        int8_t v14238 = v14228.read();	// L21971
        int8_t b256;	// L21972
        b256 = v14238;	// L21973
        int8_t v14240 = a256;	// L21974
        int8_t v14241 = b256;	// L21975
        int16_t v14242 = v14240;	// L21976
        int16_t v14243 = v14241;	// L21977
        int16_t v14244 = v14242 * v14243;	// L21978
        int8_t v14245 = c240;	// L21979
        ap_int<17> v14246 = v14245;	// L21980
        ap_int<17> v14247 = v14244;	// L21981
        ap_int<17> v14248 = v14246 + v14247;	// L21982
        int8_t v14249 = v14248;	// L21983
        c240 = v14249;	// L21984
        int8_t v14250 = a256;	// L21985
        v14229.write(v14250);	// L21986
      }
      ap_uint<128> packed_tmp240;	// L21988
      ap_uint<128> v14252 = v14230.read();	// L21989
      packed_tmp240 = v14252;	// L21990
      ap_uint<128> packed_c240;	// L21991
      packed_c240 = 0;	// L21992
      l_S_m_1_m240: for (int m240 = 0; m240 < 16; m240++) {	// L21993
        ap_int<33> v14255 = m240;	// L21994
        bool v14256 = v14255 == 15;	// L21995
        if (v14256) {	// L21996
          int8_t v14257 = c240;	// L21997
          ap_int<128> v14258 = packed_c240;	// L21998
          int64_t v14259 = m240;	// L21999
          int64_t v14260 = v14259 * 8;	// L22000
          ap_int<34> v14261 = m240;	// L22001
          ap_int<34> v14262 = v14261 + 1;	// L22002
          ap_int<66> v14263 = v14262;	// L22003
          ap_int<66> v14264 = v14263 * 8;	// L22004
          ap_int<66> v14265 = v14264 - 1;	// L22005
          int v14266 = v14260;	// L22006
          int v14267 = v14265;	// L22007
          ap_int<128> v14268;
          ap_int<128> v14268_tmp = v14258;
          v14268_tmp(v14267, v14266) = v14257;
          v14268 = v14268_tmp;	// L22008
          packed_c240 = v14268;	// L22009
        } else {
          ap_int<128> v14269 = packed_tmp240;	// L22011
          int64_t v14270 = m240;	// L22012
          int64_t v14271 = v14270 * 8;	// L22013
          ap_int<34> v14272 = m240;	// L22014
          ap_int<34> v14273 = v14272 + 1;	// L22015
          ap_int<66> v14274 = v14273;	// L22016
          ap_int<66> v14275 = v14274 * 8;	// L22017
          ap_int<66> v14276 = v14275 - 1;	// L22018
          int v14277 = v14271;	// L22019
          int v14278 = v14276;	// L22020
          int8_t v14279;
          ap_int<128> v14279_tmp = v14269;
          v14279 = v14279_tmp(v14278, v14277);	// L22021
          ap_int<128> v14280 = packed_c240;	// L22022
          ap_int<128> v14281;
          ap_int<128> v14281_tmp = v14280;
          v14281_tmp(v14278, v14277) = v14279;
          v14281 = v14281_tmp;	// L22032
          packed_c240 = v14281;	// L22033
        }
      }
      ap_int<128> v14282 = packed_c240;	// L22036
      v14231.write(v14282);	// L22037
    }
  }
}

void gemm_16_2(
  hls::stream< int8_t >& v14283,
  hls::stream< int8_t >& v14284,
  hls::stream< int8_t >& v14285,
  hls::stream< ap_uint<128> >& v14286,
  hls::stream< ap_uint<128> >& v14287
) {	// L22042
  l_S_mt_nt_0_mt276: for (int mt276 = 0; mt276 < 4; mt276++) {	// L22050
    l_nt276: for (int nt276 = 0; nt276 < 4; nt276++) {	// L22051
      int8_t c241;	// L22052
      c241 = 0;	// L22053
      l_S_k_0_k276: for (int k276 = 0; k276 < 64; k276++) {	// L22054
        int8_t v14292 = v14283.read();	// L22055
        int8_t a257;	// L22056
        a257 = v14292;	// L22057
        int8_t v14294 = v14284.read();	// L22058
        int8_t b257;	// L22059
        b257 = v14294;	// L22060
        int8_t v14296 = a257;	// L22061
        int8_t v14297 = b257;	// L22062
        int16_t v14298 = v14296;	// L22063
        int16_t v14299 = v14297;	// L22064
        int16_t v14300 = v14298 * v14299;	// L22065
        int8_t v14301 = c241;	// L22066
        ap_int<17> v14302 = v14301;	// L22067
        ap_int<17> v14303 = v14300;	// L22068
        ap_int<17> v14304 = v14302 + v14303;	// L22069
        int8_t v14305 = v14304;	// L22070
        c241 = v14305;	// L22071
        int8_t v14306 = a257;	// L22072
        v14285.write(v14306);	// L22073
      }
      ap_uint<128> packed_tmp241;	// L22075
      ap_uint<128> v14308 = v14286.read();	// L22076
      packed_tmp241 = v14308;	// L22077
      ap_uint<128> packed_c241;	// L22078
      packed_c241 = 0;	// L22079
      l_S_m_1_m241: for (int m241 = 0; m241 < 16; m241++) {	// L22080
        ap_int<33> v14311 = m241;	// L22081
        bool v14312 = v14311 == 15;	// L22082
        if (v14312) {	// L22083
          int8_t v14313 = c241;	// L22084
          ap_int<128> v14314 = packed_c241;	// L22085
          int64_t v14315 = m241;	// L22086
          int64_t v14316 = v14315 * 8;	// L22087
          ap_int<34> v14317 = m241;	// L22088
          ap_int<34> v14318 = v14317 + 1;	// L22089
          ap_int<66> v14319 = v14318;	// L22090
          ap_int<66> v14320 = v14319 * 8;	// L22091
          ap_int<66> v14321 = v14320 - 1;	// L22092
          int v14322 = v14316;	// L22093
          int v14323 = v14321;	// L22094
          ap_int<128> v14324;
          ap_int<128> v14324_tmp = v14314;
          v14324_tmp(v14323, v14322) = v14313;
          v14324 = v14324_tmp;	// L22095
          packed_c241 = v14324;	// L22096
        } else {
          ap_int<128> v14325 = packed_tmp241;	// L22098
          int64_t v14326 = m241;	// L22099
          int64_t v14327 = v14326 * 8;	// L22100
          ap_int<34> v14328 = m241;	// L22101
          ap_int<34> v14329 = v14328 + 1;	// L22102
          ap_int<66> v14330 = v14329;	// L22103
          ap_int<66> v14331 = v14330 * 8;	// L22104
          ap_int<66> v14332 = v14331 - 1;	// L22105
          int v14333 = v14327;	// L22106
          int v14334 = v14332;	// L22107
          int8_t v14335;
          ap_int<128> v14335_tmp = v14325;
          v14335 = v14335_tmp(v14334, v14333);	// L22108
          ap_int<128> v14336 = packed_c241;	// L22109
          ap_int<128> v14337;
          ap_int<128> v14337_tmp = v14336;
          v14337_tmp(v14334, v14333) = v14335;
          v14337 = v14337_tmp;	// L22119
          packed_c241 = v14337;	// L22120
        }
      }
      ap_int<128> v14338 = packed_c241;	// L22123
      v14287.write(v14338);	// L22124
    }
  }
}

void gemm_16_3(
  hls::stream< int8_t >& v14339,
  hls::stream< int8_t >& v14340,
  hls::stream< int8_t >& v14341,
  hls::stream< ap_uint<128> >& v14342,
  hls::stream< ap_uint<128> >& v14343
) {	// L22129
  l_S_mt_nt_0_mt277: for (int mt277 = 0; mt277 < 4; mt277++) {	// L22137
    l_nt277: for (int nt277 = 0; nt277 < 4; nt277++) {	// L22138
      int8_t c242;	// L22139
      c242 = 0;	// L22140
      l_S_k_0_k277: for (int k277 = 0; k277 < 64; k277++) {	// L22141
        int8_t v14348 = v14339.read();	// L22142
        int8_t a258;	// L22143
        a258 = v14348;	// L22144
        int8_t v14350 = v14340.read();	// L22145
        int8_t b258;	// L22146
        b258 = v14350;	// L22147
        int8_t v14352 = a258;	// L22148
        int8_t v14353 = b258;	// L22149
        int16_t v14354 = v14352;	// L22150
        int16_t v14355 = v14353;	// L22151
        int16_t v14356 = v14354 * v14355;	// L22152
        int8_t v14357 = c242;	// L22153
        ap_int<17> v14358 = v14357;	// L22154
        ap_int<17> v14359 = v14356;	// L22155
        ap_int<17> v14360 = v14358 + v14359;	// L22156
        int8_t v14361 = v14360;	// L22157
        c242 = v14361;	// L22158
        int8_t v14362 = a258;	// L22159
        v14341.write(v14362);	// L22160
      }
      ap_uint<128> packed_tmp242;	// L22162
      ap_uint<128> v14364 = v14342.read();	// L22163
      packed_tmp242 = v14364;	// L22164
      ap_uint<128> packed_c242;	// L22165
      packed_c242 = 0;	// L22166
      l_S_m_1_m242: for (int m242 = 0; m242 < 16; m242++) {	// L22167
        ap_int<33> v14367 = m242;	// L22168
        bool v14368 = v14367 == 15;	// L22169
        if (v14368) {	// L22170
          int8_t v14369 = c242;	// L22171
          ap_int<128> v14370 = packed_c242;	// L22172
          int64_t v14371 = m242;	// L22173
          int64_t v14372 = v14371 * 8;	// L22174
          ap_int<34> v14373 = m242;	// L22175
          ap_int<34> v14374 = v14373 + 1;	// L22176
          ap_int<66> v14375 = v14374;	// L22177
          ap_int<66> v14376 = v14375 * 8;	// L22178
          ap_int<66> v14377 = v14376 - 1;	// L22179
          int v14378 = v14372;	// L22180
          int v14379 = v14377;	// L22181
          ap_int<128> v14380;
          ap_int<128> v14380_tmp = v14370;
          v14380_tmp(v14379, v14378) = v14369;
          v14380 = v14380_tmp;	// L22182
          packed_c242 = v14380;	// L22183
        } else {
          ap_int<128> v14381 = packed_tmp242;	// L22185
          int64_t v14382 = m242;	// L22186
          int64_t v14383 = v14382 * 8;	// L22187
          ap_int<34> v14384 = m242;	// L22188
          ap_int<34> v14385 = v14384 + 1;	// L22189
          ap_int<66> v14386 = v14385;	// L22190
          ap_int<66> v14387 = v14386 * 8;	// L22191
          ap_int<66> v14388 = v14387 - 1;	// L22192
          int v14389 = v14383;	// L22193
          int v14390 = v14388;	// L22194
          int8_t v14391;
          ap_int<128> v14391_tmp = v14381;
          v14391 = v14391_tmp(v14390, v14389);	// L22195
          ap_int<128> v14392 = packed_c242;	// L22196
          ap_int<128> v14393;
          ap_int<128> v14393_tmp = v14392;
          v14393_tmp(v14390, v14389) = v14391;
          v14393 = v14393_tmp;	// L22206
          packed_c242 = v14393;	// L22207
        }
      }
      ap_int<128> v14394 = packed_c242;	// L22210
      v14343.write(v14394);	// L22211
    }
  }
}

void gemm_16_4(
  hls::stream< int8_t >& v14395,
  hls::stream< int8_t >& v14396,
  hls::stream< int8_t >& v14397,
  hls::stream< ap_uint<128> >& v14398,
  hls::stream< ap_uint<128> >& v14399
) {	// L22216
  l_S_mt_nt_0_mt278: for (int mt278 = 0; mt278 < 4; mt278++) {	// L22224
    l_nt278: for (int nt278 = 0; nt278 < 4; nt278++) {	// L22225
      int8_t c243;	// L22226
      c243 = 0;	// L22227
      l_S_k_0_k278: for (int k278 = 0; k278 < 64; k278++) {	// L22228
        int8_t v14404 = v14395.read();	// L22229
        int8_t a259;	// L22230
        a259 = v14404;	// L22231
        int8_t v14406 = v14396.read();	// L22232
        int8_t b259;	// L22233
        b259 = v14406;	// L22234
        int8_t v14408 = a259;	// L22235
        int8_t v14409 = b259;	// L22236
        int16_t v14410 = v14408;	// L22237
        int16_t v14411 = v14409;	// L22238
        int16_t v14412 = v14410 * v14411;	// L22239
        int8_t v14413 = c243;	// L22240
        ap_int<17> v14414 = v14413;	// L22241
        ap_int<17> v14415 = v14412;	// L22242
        ap_int<17> v14416 = v14414 + v14415;	// L22243
        int8_t v14417 = v14416;	// L22244
        c243 = v14417;	// L22245
        int8_t v14418 = a259;	// L22246
        v14397.write(v14418);	// L22247
      }
      ap_uint<128> packed_tmp243;	// L22249
      ap_uint<128> v14420 = v14398.read();	// L22250
      packed_tmp243 = v14420;	// L22251
      ap_uint<128> packed_c243;	// L22252
      packed_c243 = 0;	// L22253
      l_S_m_1_m243: for (int m243 = 0; m243 < 16; m243++) {	// L22254
        ap_int<33> v14423 = m243;	// L22255
        bool v14424 = v14423 == 15;	// L22256
        if (v14424) {	// L22257
          int8_t v14425 = c243;	// L22258
          ap_int<128> v14426 = packed_c243;	// L22259
          int64_t v14427 = m243;	// L22260
          int64_t v14428 = v14427 * 8;	// L22261
          ap_int<34> v14429 = m243;	// L22262
          ap_int<34> v14430 = v14429 + 1;	// L22263
          ap_int<66> v14431 = v14430;	// L22264
          ap_int<66> v14432 = v14431 * 8;	// L22265
          ap_int<66> v14433 = v14432 - 1;	// L22266
          int v14434 = v14428;	// L22267
          int v14435 = v14433;	// L22268
          ap_int<128> v14436;
          ap_int<128> v14436_tmp = v14426;
          v14436_tmp(v14435, v14434) = v14425;
          v14436 = v14436_tmp;	// L22269
          packed_c243 = v14436;	// L22270
        } else {
          ap_int<128> v14437 = packed_tmp243;	// L22272
          int64_t v14438 = m243;	// L22273
          int64_t v14439 = v14438 * 8;	// L22274
          ap_int<34> v14440 = m243;	// L22275
          ap_int<34> v14441 = v14440 + 1;	// L22276
          ap_int<66> v14442 = v14441;	// L22277
          ap_int<66> v14443 = v14442 * 8;	// L22278
          ap_int<66> v14444 = v14443 - 1;	// L22279
          int v14445 = v14439;	// L22280
          int v14446 = v14444;	// L22281
          int8_t v14447;
          ap_int<128> v14447_tmp = v14437;
          v14447 = v14447_tmp(v14446, v14445);	// L22282
          ap_int<128> v14448 = packed_c243;	// L22283
          ap_int<128> v14449;
          ap_int<128> v14449_tmp = v14448;
          v14449_tmp(v14446, v14445) = v14447;
          v14449 = v14449_tmp;	// L22293
          packed_c243 = v14449;	// L22294
        }
      }
      ap_int<128> v14450 = packed_c243;	// L22297
      v14399.write(v14450);	// L22298
    }
  }
}

void gemm_16_5(
  hls::stream< int8_t >& v14451,
  hls::stream< int8_t >& v14452,
  hls::stream< int8_t >& v14453,
  hls::stream< ap_uint<128> >& v14454,
  hls::stream< ap_uint<128> >& v14455
) {	// L22303
  l_S_mt_nt_0_mt279: for (int mt279 = 0; mt279 < 4; mt279++) {	// L22311
    l_nt279: for (int nt279 = 0; nt279 < 4; nt279++) {	// L22312
      int8_t c244;	// L22313
      c244 = 0;	// L22314
      l_S_k_0_k279: for (int k279 = 0; k279 < 64; k279++) {	// L22315
        int8_t v14460 = v14451.read();	// L22316
        int8_t a260;	// L22317
        a260 = v14460;	// L22318
        int8_t v14462 = v14452.read();	// L22319
        int8_t b260;	// L22320
        b260 = v14462;	// L22321
        int8_t v14464 = a260;	// L22322
        int8_t v14465 = b260;	// L22323
        int16_t v14466 = v14464;	// L22324
        int16_t v14467 = v14465;	// L22325
        int16_t v14468 = v14466 * v14467;	// L22326
        int8_t v14469 = c244;	// L22327
        ap_int<17> v14470 = v14469;	// L22328
        ap_int<17> v14471 = v14468;	// L22329
        ap_int<17> v14472 = v14470 + v14471;	// L22330
        int8_t v14473 = v14472;	// L22331
        c244 = v14473;	// L22332
        int8_t v14474 = a260;	// L22333
        v14453.write(v14474);	// L22334
      }
      ap_uint<128> packed_tmp244;	// L22336
      ap_uint<128> v14476 = v14454.read();	// L22337
      packed_tmp244 = v14476;	// L22338
      ap_uint<128> packed_c244;	// L22339
      packed_c244 = 0;	// L22340
      l_S_m_1_m244: for (int m244 = 0; m244 < 16; m244++) {	// L22341
        ap_int<33> v14479 = m244;	// L22342
        bool v14480 = v14479 == 15;	// L22343
        if (v14480) {	// L22344
          int8_t v14481 = c244;	// L22345
          ap_int<128> v14482 = packed_c244;	// L22346
          int64_t v14483 = m244;	// L22347
          int64_t v14484 = v14483 * 8;	// L22348
          ap_int<34> v14485 = m244;	// L22349
          ap_int<34> v14486 = v14485 + 1;	// L22350
          ap_int<66> v14487 = v14486;	// L22351
          ap_int<66> v14488 = v14487 * 8;	// L22352
          ap_int<66> v14489 = v14488 - 1;	// L22353
          int v14490 = v14484;	// L22354
          int v14491 = v14489;	// L22355
          ap_int<128> v14492;
          ap_int<128> v14492_tmp = v14482;
          v14492_tmp(v14491, v14490) = v14481;
          v14492 = v14492_tmp;	// L22356
          packed_c244 = v14492;	// L22357
        } else {
          ap_int<128> v14493 = packed_tmp244;	// L22359
          int64_t v14494 = m244;	// L22360
          int64_t v14495 = v14494 * 8;	// L22361
          ap_int<34> v14496 = m244;	// L22362
          ap_int<34> v14497 = v14496 + 1;	// L22363
          ap_int<66> v14498 = v14497;	// L22364
          ap_int<66> v14499 = v14498 * 8;	// L22365
          ap_int<66> v14500 = v14499 - 1;	// L22366
          int v14501 = v14495;	// L22367
          int v14502 = v14500;	// L22368
          int8_t v14503;
          ap_int<128> v14503_tmp = v14493;
          v14503 = v14503_tmp(v14502, v14501);	// L22369
          ap_int<128> v14504 = packed_c244;	// L22370
          ap_int<128> v14505;
          ap_int<128> v14505_tmp = v14504;
          v14505_tmp(v14502, v14501) = v14503;
          v14505 = v14505_tmp;	// L22380
          packed_c244 = v14505;	// L22381
        }
      }
      ap_int<128> v14506 = packed_c244;	// L22384
      v14455.write(v14506);	// L22385
    }
  }
}

void gemm_16_6(
  hls::stream< int8_t >& v14507,
  hls::stream< int8_t >& v14508,
  hls::stream< int8_t >& v14509,
  hls::stream< ap_uint<128> >& v14510,
  hls::stream< ap_uint<128> >& v14511
) {	// L22390
  l_S_mt_nt_0_mt280: for (int mt280 = 0; mt280 < 4; mt280++) {	// L22398
    l_nt280: for (int nt280 = 0; nt280 < 4; nt280++) {	// L22399
      int8_t c245;	// L22400
      c245 = 0;	// L22401
      l_S_k_0_k280: for (int k280 = 0; k280 < 64; k280++) {	// L22402
        int8_t v14516 = v14507.read();	// L22403
        int8_t a261;	// L22404
        a261 = v14516;	// L22405
        int8_t v14518 = v14508.read();	// L22406
        int8_t b261;	// L22407
        b261 = v14518;	// L22408
        int8_t v14520 = a261;	// L22409
        int8_t v14521 = b261;	// L22410
        int16_t v14522 = v14520;	// L22411
        int16_t v14523 = v14521;	// L22412
        int16_t v14524 = v14522 * v14523;	// L22413
        int8_t v14525 = c245;	// L22414
        ap_int<17> v14526 = v14525;	// L22415
        ap_int<17> v14527 = v14524;	// L22416
        ap_int<17> v14528 = v14526 + v14527;	// L22417
        int8_t v14529 = v14528;	// L22418
        c245 = v14529;	// L22419
        int8_t v14530 = a261;	// L22420
        v14509.write(v14530);	// L22421
      }
      ap_uint<128> packed_tmp245;	// L22423
      ap_uint<128> v14532 = v14510.read();	// L22424
      packed_tmp245 = v14532;	// L22425
      ap_uint<128> packed_c245;	// L22426
      packed_c245 = 0;	// L22427
      l_S_m_1_m245: for (int m245 = 0; m245 < 16; m245++) {	// L22428
        ap_int<33> v14535 = m245;	// L22429
        bool v14536 = v14535 == 15;	// L22430
        if (v14536) {	// L22431
          int8_t v14537 = c245;	// L22432
          ap_int<128> v14538 = packed_c245;	// L22433
          int64_t v14539 = m245;	// L22434
          int64_t v14540 = v14539 * 8;	// L22435
          ap_int<34> v14541 = m245;	// L22436
          ap_int<34> v14542 = v14541 + 1;	// L22437
          ap_int<66> v14543 = v14542;	// L22438
          ap_int<66> v14544 = v14543 * 8;	// L22439
          ap_int<66> v14545 = v14544 - 1;	// L22440
          int v14546 = v14540;	// L22441
          int v14547 = v14545;	// L22442
          ap_int<128> v14548;
          ap_int<128> v14548_tmp = v14538;
          v14548_tmp(v14547, v14546) = v14537;
          v14548 = v14548_tmp;	// L22443
          packed_c245 = v14548;	// L22444
        } else {
          ap_int<128> v14549 = packed_tmp245;	// L22446
          int64_t v14550 = m245;	// L22447
          int64_t v14551 = v14550 * 8;	// L22448
          ap_int<34> v14552 = m245;	// L22449
          ap_int<34> v14553 = v14552 + 1;	// L22450
          ap_int<66> v14554 = v14553;	// L22451
          ap_int<66> v14555 = v14554 * 8;	// L22452
          ap_int<66> v14556 = v14555 - 1;	// L22453
          int v14557 = v14551;	// L22454
          int v14558 = v14556;	// L22455
          int8_t v14559;
          ap_int<128> v14559_tmp = v14549;
          v14559 = v14559_tmp(v14558, v14557);	// L22456
          ap_int<128> v14560 = packed_c245;	// L22457
          ap_int<128> v14561;
          ap_int<128> v14561_tmp = v14560;
          v14561_tmp(v14558, v14557) = v14559;
          v14561 = v14561_tmp;	// L22467
          packed_c245 = v14561;	// L22468
        }
      }
      ap_int<128> v14562 = packed_c245;	// L22471
      v14511.write(v14562);	// L22472
    }
  }
}

void gemm_16_7(
  hls::stream< int8_t >& v14563,
  hls::stream< int8_t >& v14564,
  hls::stream< int8_t >& v14565,
  hls::stream< ap_uint<128> >& v14566,
  hls::stream< ap_uint<128> >& v14567
) {	// L22477
  l_S_mt_nt_0_mt281: for (int mt281 = 0; mt281 < 4; mt281++) {	// L22485
    l_nt281: for (int nt281 = 0; nt281 < 4; nt281++) {	// L22486
      int8_t c246;	// L22487
      c246 = 0;	// L22488
      l_S_k_0_k281: for (int k281 = 0; k281 < 64; k281++) {	// L22489
        int8_t v14572 = v14563.read();	// L22490
        int8_t a262;	// L22491
        a262 = v14572;	// L22492
        int8_t v14574 = v14564.read();	// L22493
        int8_t b262;	// L22494
        b262 = v14574;	// L22495
        int8_t v14576 = a262;	// L22496
        int8_t v14577 = b262;	// L22497
        int16_t v14578 = v14576;	// L22498
        int16_t v14579 = v14577;	// L22499
        int16_t v14580 = v14578 * v14579;	// L22500
        int8_t v14581 = c246;	// L22501
        ap_int<17> v14582 = v14581;	// L22502
        ap_int<17> v14583 = v14580;	// L22503
        ap_int<17> v14584 = v14582 + v14583;	// L22504
        int8_t v14585 = v14584;	// L22505
        c246 = v14585;	// L22506
        int8_t v14586 = a262;	// L22507
        v14565.write(v14586);	// L22508
      }
      ap_uint<128> packed_tmp246;	// L22510
      ap_uint<128> v14588 = v14566.read();	// L22511
      packed_tmp246 = v14588;	// L22512
      ap_uint<128> packed_c246;	// L22513
      packed_c246 = 0;	// L22514
      l_S_m_1_m246: for (int m246 = 0; m246 < 16; m246++) {	// L22515
        ap_int<33> v14591 = m246;	// L22516
        bool v14592 = v14591 == 15;	// L22517
        if (v14592) {	// L22518
          int8_t v14593 = c246;	// L22519
          ap_int<128> v14594 = packed_c246;	// L22520
          int64_t v14595 = m246;	// L22521
          int64_t v14596 = v14595 * 8;	// L22522
          ap_int<34> v14597 = m246;	// L22523
          ap_int<34> v14598 = v14597 + 1;	// L22524
          ap_int<66> v14599 = v14598;	// L22525
          ap_int<66> v14600 = v14599 * 8;	// L22526
          ap_int<66> v14601 = v14600 - 1;	// L22527
          int v14602 = v14596;	// L22528
          int v14603 = v14601;	// L22529
          ap_int<128> v14604;
          ap_int<128> v14604_tmp = v14594;
          v14604_tmp(v14603, v14602) = v14593;
          v14604 = v14604_tmp;	// L22530
          packed_c246 = v14604;	// L22531
        } else {
          ap_int<128> v14605 = packed_tmp246;	// L22533
          int64_t v14606 = m246;	// L22534
          int64_t v14607 = v14606 * 8;	// L22535
          ap_int<34> v14608 = m246;	// L22536
          ap_int<34> v14609 = v14608 + 1;	// L22537
          ap_int<66> v14610 = v14609;	// L22538
          ap_int<66> v14611 = v14610 * 8;	// L22539
          ap_int<66> v14612 = v14611 - 1;	// L22540
          int v14613 = v14607;	// L22541
          int v14614 = v14612;	// L22542
          int8_t v14615;
          ap_int<128> v14615_tmp = v14605;
          v14615 = v14615_tmp(v14614, v14613);	// L22543
          ap_int<128> v14616 = packed_c246;	// L22544
          ap_int<128> v14617;
          ap_int<128> v14617_tmp = v14616;
          v14617_tmp(v14614, v14613) = v14615;
          v14617 = v14617_tmp;	// L22554
          packed_c246 = v14617;	// L22555
        }
      }
      ap_int<128> v14618 = packed_c246;	// L22558
      v14567.write(v14618);	// L22559
    }
  }
}

void gemm_16_8(
  hls::stream< int8_t >& v14619,
  hls::stream< int8_t >& v14620,
  hls::stream< int8_t >& v14621,
  hls::stream< ap_uint<128> >& v14622,
  hls::stream< ap_uint<128> >& v14623
) {	// L22564
  l_S_mt_nt_0_mt282: for (int mt282 = 0; mt282 < 4; mt282++) {	// L22572
    l_nt282: for (int nt282 = 0; nt282 < 4; nt282++) {	// L22573
      int8_t c247;	// L22574
      c247 = 0;	// L22575
      l_S_k_0_k282: for (int k282 = 0; k282 < 64; k282++) {	// L22576
        int8_t v14628 = v14619.read();	// L22577
        int8_t a263;	// L22578
        a263 = v14628;	// L22579
        int8_t v14630 = v14620.read();	// L22580
        int8_t b263;	// L22581
        b263 = v14630;	// L22582
        int8_t v14632 = a263;	// L22583
        int8_t v14633 = b263;	// L22584
        int16_t v14634 = v14632;	// L22585
        int16_t v14635 = v14633;	// L22586
        int16_t v14636 = v14634 * v14635;	// L22587
        int8_t v14637 = c247;	// L22588
        ap_int<17> v14638 = v14637;	// L22589
        ap_int<17> v14639 = v14636;	// L22590
        ap_int<17> v14640 = v14638 + v14639;	// L22591
        int8_t v14641 = v14640;	// L22592
        c247 = v14641;	// L22593
        int8_t v14642 = a263;	// L22594
        v14621.write(v14642);	// L22595
      }
      ap_uint<128> packed_tmp247;	// L22597
      ap_uint<128> v14644 = v14622.read();	// L22598
      packed_tmp247 = v14644;	// L22599
      ap_uint<128> packed_c247;	// L22600
      packed_c247 = 0;	// L22601
      l_S_m_1_m247: for (int m247 = 0; m247 < 16; m247++) {	// L22602
        ap_int<33> v14647 = m247;	// L22603
        bool v14648 = v14647 == 15;	// L22604
        if (v14648) {	// L22605
          int8_t v14649 = c247;	// L22606
          ap_int<128> v14650 = packed_c247;	// L22607
          int64_t v14651 = m247;	// L22608
          int64_t v14652 = v14651 * 8;	// L22609
          ap_int<34> v14653 = m247;	// L22610
          ap_int<34> v14654 = v14653 + 1;	// L22611
          ap_int<66> v14655 = v14654;	// L22612
          ap_int<66> v14656 = v14655 * 8;	// L22613
          ap_int<66> v14657 = v14656 - 1;	// L22614
          int v14658 = v14652;	// L22615
          int v14659 = v14657;	// L22616
          ap_int<128> v14660;
          ap_int<128> v14660_tmp = v14650;
          v14660_tmp(v14659, v14658) = v14649;
          v14660 = v14660_tmp;	// L22617
          packed_c247 = v14660;	// L22618
        } else {
          ap_int<128> v14661 = packed_tmp247;	// L22620
          int64_t v14662 = m247;	// L22621
          int64_t v14663 = v14662 * 8;	// L22622
          ap_int<34> v14664 = m247;	// L22623
          ap_int<34> v14665 = v14664 + 1;	// L22624
          ap_int<66> v14666 = v14665;	// L22625
          ap_int<66> v14667 = v14666 * 8;	// L22626
          ap_int<66> v14668 = v14667 - 1;	// L22627
          int v14669 = v14663;	// L22628
          int v14670 = v14668;	// L22629
          int8_t v14671;
          ap_int<128> v14671_tmp = v14661;
          v14671 = v14671_tmp(v14670, v14669);	// L22630
          ap_int<128> v14672 = packed_c247;	// L22631
          ap_int<128> v14673;
          ap_int<128> v14673_tmp = v14672;
          v14673_tmp(v14670, v14669) = v14671;
          v14673 = v14673_tmp;	// L22641
          packed_c247 = v14673;	// L22642
        }
      }
      ap_int<128> v14674 = packed_c247;	// L22645
      v14623.write(v14674);	// L22646
    }
  }
}

void gemm_16_9(
  hls::stream< int8_t >& v14675,
  hls::stream< int8_t >& v14676,
  hls::stream< int8_t >& v14677,
  hls::stream< ap_uint<128> >& v14678,
  hls::stream< ap_uint<128> >& v14679
) {	// L22651
  l_S_mt_nt_0_mt283: for (int mt283 = 0; mt283 < 4; mt283++) {	// L22659
    l_nt283: for (int nt283 = 0; nt283 < 4; nt283++) {	// L22660
      int8_t c248;	// L22661
      c248 = 0;	// L22662
      l_S_k_0_k283: for (int k283 = 0; k283 < 64; k283++) {	// L22663
        int8_t v14684 = v14675.read();	// L22664
        int8_t a264;	// L22665
        a264 = v14684;	// L22666
        int8_t v14686 = v14676.read();	// L22667
        int8_t b264;	// L22668
        b264 = v14686;	// L22669
        int8_t v14688 = a264;	// L22670
        int8_t v14689 = b264;	// L22671
        int16_t v14690 = v14688;	// L22672
        int16_t v14691 = v14689;	// L22673
        int16_t v14692 = v14690 * v14691;	// L22674
        int8_t v14693 = c248;	// L22675
        ap_int<17> v14694 = v14693;	// L22676
        ap_int<17> v14695 = v14692;	// L22677
        ap_int<17> v14696 = v14694 + v14695;	// L22678
        int8_t v14697 = v14696;	// L22679
        c248 = v14697;	// L22680
        int8_t v14698 = a264;	// L22681
        v14677.write(v14698);	// L22682
      }
      ap_uint<128> packed_tmp248;	// L22684
      ap_uint<128> v14700 = v14678.read();	// L22685
      packed_tmp248 = v14700;	// L22686
      ap_uint<128> packed_c248;	// L22687
      packed_c248 = 0;	// L22688
      l_S_m_1_m248: for (int m248 = 0; m248 < 16; m248++) {	// L22689
        ap_int<33> v14703 = m248;	// L22690
        bool v14704 = v14703 == 15;	// L22691
        if (v14704) {	// L22692
          int8_t v14705 = c248;	// L22693
          ap_int<128> v14706 = packed_c248;	// L22694
          int64_t v14707 = m248;	// L22695
          int64_t v14708 = v14707 * 8;	// L22696
          ap_int<34> v14709 = m248;	// L22697
          ap_int<34> v14710 = v14709 + 1;	// L22698
          ap_int<66> v14711 = v14710;	// L22699
          ap_int<66> v14712 = v14711 * 8;	// L22700
          ap_int<66> v14713 = v14712 - 1;	// L22701
          int v14714 = v14708;	// L22702
          int v14715 = v14713;	// L22703
          ap_int<128> v14716;
          ap_int<128> v14716_tmp = v14706;
          v14716_tmp(v14715, v14714) = v14705;
          v14716 = v14716_tmp;	// L22704
          packed_c248 = v14716;	// L22705
        } else {
          ap_int<128> v14717 = packed_tmp248;	// L22707
          int64_t v14718 = m248;	// L22708
          int64_t v14719 = v14718 * 8;	// L22709
          ap_int<34> v14720 = m248;	// L22710
          ap_int<34> v14721 = v14720 + 1;	// L22711
          ap_int<66> v14722 = v14721;	// L22712
          ap_int<66> v14723 = v14722 * 8;	// L22713
          ap_int<66> v14724 = v14723 - 1;	// L22714
          int v14725 = v14719;	// L22715
          int v14726 = v14724;	// L22716
          int8_t v14727;
          ap_int<128> v14727_tmp = v14717;
          v14727 = v14727_tmp(v14726, v14725);	// L22717
          ap_int<128> v14728 = packed_c248;	// L22718
          ap_int<128> v14729;
          ap_int<128> v14729_tmp = v14728;
          v14729_tmp(v14726, v14725) = v14727;
          v14729 = v14729_tmp;	// L22728
          packed_c248 = v14729;	// L22729
        }
      }
      ap_int<128> v14730 = packed_c248;	// L22732
      v14679.write(v14730);	// L22733
    }
  }
}

void gemm_16_10(
  hls::stream< int8_t >& v14731,
  hls::stream< int8_t >& v14732,
  hls::stream< int8_t >& v14733,
  hls::stream< ap_uint<128> >& v14734,
  hls::stream< ap_uint<128> >& v14735
) {	// L22738
  l_S_mt_nt_0_mt284: for (int mt284 = 0; mt284 < 4; mt284++) {	// L22746
    l_nt284: for (int nt284 = 0; nt284 < 4; nt284++) {	// L22747
      int8_t c249;	// L22748
      c249 = 0;	// L22749
      l_S_k_0_k284: for (int k284 = 0; k284 < 64; k284++) {	// L22750
        int8_t v14740 = v14731.read();	// L22751
        int8_t a265;	// L22752
        a265 = v14740;	// L22753
        int8_t v14742 = v14732.read();	// L22754
        int8_t b265;	// L22755
        b265 = v14742;	// L22756
        int8_t v14744 = a265;	// L22757
        int8_t v14745 = b265;	// L22758
        int16_t v14746 = v14744;	// L22759
        int16_t v14747 = v14745;	// L22760
        int16_t v14748 = v14746 * v14747;	// L22761
        int8_t v14749 = c249;	// L22762
        ap_int<17> v14750 = v14749;	// L22763
        ap_int<17> v14751 = v14748;	// L22764
        ap_int<17> v14752 = v14750 + v14751;	// L22765
        int8_t v14753 = v14752;	// L22766
        c249 = v14753;	// L22767
        int8_t v14754 = a265;	// L22768
        v14733.write(v14754);	// L22769
      }
      ap_uint<128> packed_tmp249;	// L22771
      ap_uint<128> v14756 = v14734.read();	// L22772
      packed_tmp249 = v14756;	// L22773
      ap_uint<128> packed_c249;	// L22774
      packed_c249 = 0;	// L22775
      l_S_m_1_m249: for (int m249 = 0; m249 < 16; m249++) {	// L22776
        ap_int<33> v14759 = m249;	// L22777
        bool v14760 = v14759 == 15;	// L22778
        if (v14760) {	// L22779
          int8_t v14761 = c249;	// L22780
          ap_int<128> v14762 = packed_c249;	// L22781
          int64_t v14763 = m249;	// L22782
          int64_t v14764 = v14763 * 8;	// L22783
          ap_int<34> v14765 = m249;	// L22784
          ap_int<34> v14766 = v14765 + 1;	// L22785
          ap_int<66> v14767 = v14766;	// L22786
          ap_int<66> v14768 = v14767 * 8;	// L22787
          ap_int<66> v14769 = v14768 - 1;	// L22788
          int v14770 = v14764;	// L22789
          int v14771 = v14769;	// L22790
          ap_int<128> v14772;
          ap_int<128> v14772_tmp = v14762;
          v14772_tmp(v14771, v14770) = v14761;
          v14772 = v14772_tmp;	// L22791
          packed_c249 = v14772;	// L22792
        } else {
          ap_int<128> v14773 = packed_tmp249;	// L22794
          int64_t v14774 = m249;	// L22795
          int64_t v14775 = v14774 * 8;	// L22796
          ap_int<34> v14776 = m249;	// L22797
          ap_int<34> v14777 = v14776 + 1;	// L22798
          ap_int<66> v14778 = v14777;	// L22799
          ap_int<66> v14779 = v14778 * 8;	// L22800
          ap_int<66> v14780 = v14779 - 1;	// L22801
          int v14781 = v14775;	// L22802
          int v14782 = v14780;	// L22803
          int8_t v14783;
          ap_int<128> v14783_tmp = v14773;
          v14783 = v14783_tmp(v14782, v14781);	// L22804
          ap_int<128> v14784 = packed_c249;	// L22805
          ap_int<128> v14785;
          ap_int<128> v14785_tmp = v14784;
          v14785_tmp(v14782, v14781) = v14783;
          v14785 = v14785_tmp;	// L22815
          packed_c249 = v14785;	// L22816
        }
      }
      ap_int<128> v14786 = packed_c249;	// L22819
      v14735.write(v14786);	// L22820
    }
  }
}

void gemm_16_11(
  hls::stream< int8_t >& v14787,
  hls::stream< int8_t >& v14788,
  hls::stream< int8_t >& v14789,
  hls::stream< ap_uint<128> >& v14790,
  hls::stream< ap_uint<128> >& v14791
) {	// L22825
  l_S_mt_nt_0_mt285: for (int mt285 = 0; mt285 < 4; mt285++) {	// L22833
    l_nt285: for (int nt285 = 0; nt285 < 4; nt285++) {	// L22834
      int8_t c250;	// L22835
      c250 = 0;	// L22836
      l_S_k_0_k285: for (int k285 = 0; k285 < 64; k285++) {	// L22837
        int8_t v14796 = v14787.read();	// L22838
        int8_t a266;	// L22839
        a266 = v14796;	// L22840
        int8_t v14798 = v14788.read();	// L22841
        int8_t b266;	// L22842
        b266 = v14798;	// L22843
        int8_t v14800 = a266;	// L22844
        int8_t v14801 = b266;	// L22845
        int16_t v14802 = v14800;	// L22846
        int16_t v14803 = v14801;	// L22847
        int16_t v14804 = v14802 * v14803;	// L22848
        int8_t v14805 = c250;	// L22849
        ap_int<17> v14806 = v14805;	// L22850
        ap_int<17> v14807 = v14804;	// L22851
        ap_int<17> v14808 = v14806 + v14807;	// L22852
        int8_t v14809 = v14808;	// L22853
        c250 = v14809;	// L22854
        int8_t v14810 = a266;	// L22855
        v14789.write(v14810);	// L22856
      }
      ap_uint<128> packed_tmp250;	// L22858
      ap_uint<128> v14812 = v14790.read();	// L22859
      packed_tmp250 = v14812;	// L22860
      ap_uint<128> packed_c250;	// L22861
      packed_c250 = 0;	// L22862
      l_S_m_1_m250: for (int m250 = 0; m250 < 16; m250++) {	// L22863
        ap_int<33> v14815 = m250;	// L22864
        bool v14816 = v14815 == 15;	// L22865
        if (v14816) {	// L22866
          int8_t v14817 = c250;	// L22867
          ap_int<128> v14818 = packed_c250;	// L22868
          int64_t v14819 = m250;	// L22869
          int64_t v14820 = v14819 * 8;	// L22870
          ap_int<34> v14821 = m250;	// L22871
          ap_int<34> v14822 = v14821 + 1;	// L22872
          ap_int<66> v14823 = v14822;	// L22873
          ap_int<66> v14824 = v14823 * 8;	// L22874
          ap_int<66> v14825 = v14824 - 1;	// L22875
          int v14826 = v14820;	// L22876
          int v14827 = v14825;	// L22877
          ap_int<128> v14828;
          ap_int<128> v14828_tmp = v14818;
          v14828_tmp(v14827, v14826) = v14817;
          v14828 = v14828_tmp;	// L22878
          packed_c250 = v14828;	// L22879
        } else {
          ap_int<128> v14829 = packed_tmp250;	// L22881
          int64_t v14830 = m250;	// L22882
          int64_t v14831 = v14830 * 8;	// L22883
          ap_int<34> v14832 = m250;	// L22884
          ap_int<34> v14833 = v14832 + 1;	// L22885
          ap_int<66> v14834 = v14833;	// L22886
          ap_int<66> v14835 = v14834 * 8;	// L22887
          ap_int<66> v14836 = v14835 - 1;	// L22888
          int v14837 = v14831;	// L22889
          int v14838 = v14836;	// L22890
          int8_t v14839;
          ap_int<128> v14839_tmp = v14829;
          v14839 = v14839_tmp(v14838, v14837);	// L22891
          ap_int<128> v14840 = packed_c250;	// L22892
          ap_int<128> v14841;
          ap_int<128> v14841_tmp = v14840;
          v14841_tmp(v14838, v14837) = v14839;
          v14841 = v14841_tmp;	// L22902
          packed_c250 = v14841;	// L22903
        }
      }
      ap_int<128> v14842 = packed_c250;	// L22906
      v14791.write(v14842);	// L22907
    }
  }
}

void gemm_16_12(
  hls::stream< int8_t >& v14843,
  hls::stream< int8_t >& v14844,
  hls::stream< int8_t >& v14845,
  hls::stream< ap_uint<128> >& v14846,
  hls::stream< ap_uint<128> >& v14847
) {	// L22912
  l_S_mt_nt_0_mt286: for (int mt286 = 0; mt286 < 4; mt286++) {	// L22920
    l_nt286: for (int nt286 = 0; nt286 < 4; nt286++) {	// L22921
      int8_t c251;	// L22922
      c251 = 0;	// L22923
      l_S_k_0_k286: for (int k286 = 0; k286 < 64; k286++) {	// L22924
        int8_t v14852 = v14843.read();	// L22925
        int8_t a267;	// L22926
        a267 = v14852;	// L22927
        int8_t v14854 = v14844.read();	// L22928
        int8_t b267;	// L22929
        b267 = v14854;	// L22930
        int8_t v14856 = a267;	// L22931
        int8_t v14857 = b267;	// L22932
        int16_t v14858 = v14856;	// L22933
        int16_t v14859 = v14857;	// L22934
        int16_t v14860 = v14858 * v14859;	// L22935
        int8_t v14861 = c251;	// L22936
        ap_int<17> v14862 = v14861;	// L22937
        ap_int<17> v14863 = v14860;	// L22938
        ap_int<17> v14864 = v14862 + v14863;	// L22939
        int8_t v14865 = v14864;	// L22940
        c251 = v14865;	// L22941
        int8_t v14866 = a267;	// L22942
        v14845.write(v14866);	// L22943
      }
      ap_uint<128> packed_tmp251;	// L22945
      ap_uint<128> v14868 = v14846.read();	// L22946
      packed_tmp251 = v14868;	// L22947
      ap_uint<128> packed_c251;	// L22948
      packed_c251 = 0;	// L22949
      l_S_m_1_m251: for (int m251 = 0; m251 < 16; m251++) {	// L22950
        ap_int<33> v14871 = m251;	// L22951
        bool v14872 = v14871 == 15;	// L22952
        if (v14872) {	// L22953
          int8_t v14873 = c251;	// L22954
          ap_int<128> v14874 = packed_c251;	// L22955
          int64_t v14875 = m251;	// L22956
          int64_t v14876 = v14875 * 8;	// L22957
          ap_int<34> v14877 = m251;	// L22958
          ap_int<34> v14878 = v14877 + 1;	// L22959
          ap_int<66> v14879 = v14878;	// L22960
          ap_int<66> v14880 = v14879 * 8;	// L22961
          ap_int<66> v14881 = v14880 - 1;	// L22962
          int v14882 = v14876;	// L22963
          int v14883 = v14881;	// L22964
          ap_int<128> v14884;
          ap_int<128> v14884_tmp = v14874;
          v14884_tmp(v14883, v14882) = v14873;
          v14884 = v14884_tmp;	// L22965
          packed_c251 = v14884;	// L22966
        } else {
          ap_int<128> v14885 = packed_tmp251;	// L22968
          int64_t v14886 = m251;	// L22969
          int64_t v14887 = v14886 * 8;	// L22970
          ap_int<34> v14888 = m251;	// L22971
          ap_int<34> v14889 = v14888 + 1;	// L22972
          ap_int<66> v14890 = v14889;	// L22973
          ap_int<66> v14891 = v14890 * 8;	// L22974
          ap_int<66> v14892 = v14891 - 1;	// L22975
          int v14893 = v14887;	// L22976
          int v14894 = v14892;	// L22977
          int8_t v14895;
          ap_int<128> v14895_tmp = v14885;
          v14895 = v14895_tmp(v14894, v14893);	// L22978
          ap_int<128> v14896 = packed_c251;	// L22979
          ap_int<128> v14897;
          ap_int<128> v14897_tmp = v14896;
          v14897_tmp(v14894, v14893) = v14895;
          v14897 = v14897_tmp;	// L22989
          packed_c251 = v14897;	// L22990
        }
      }
      ap_int<128> v14898 = packed_c251;	// L22993
      v14847.write(v14898);	// L22994
    }
  }
}

void gemm_16_13(
  hls::stream< int8_t >& v14899,
  hls::stream< int8_t >& v14900,
  hls::stream< int8_t >& v14901,
  hls::stream< ap_uint<128> >& v14902,
  hls::stream< ap_uint<128> >& v14903
) {	// L22999
  l_S_mt_nt_0_mt287: for (int mt287 = 0; mt287 < 4; mt287++) {	// L23007
    l_nt287: for (int nt287 = 0; nt287 < 4; nt287++) {	// L23008
      int8_t c252;	// L23009
      c252 = 0;	// L23010
      l_S_k_0_k287: for (int k287 = 0; k287 < 64; k287++) {	// L23011
        int8_t v14908 = v14899.read();	// L23012
        int8_t a268;	// L23013
        a268 = v14908;	// L23014
        int8_t v14910 = v14900.read();	// L23015
        int8_t b268;	// L23016
        b268 = v14910;	// L23017
        int8_t v14912 = a268;	// L23018
        int8_t v14913 = b268;	// L23019
        int16_t v14914 = v14912;	// L23020
        int16_t v14915 = v14913;	// L23021
        int16_t v14916 = v14914 * v14915;	// L23022
        int8_t v14917 = c252;	// L23023
        ap_int<17> v14918 = v14917;	// L23024
        ap_int<17> v14919 = v14916;	// L23025
        ap_int<17> v14920 = v14918 + v14919;	// L23026
        int8_t v14921 = v14920;	// L23027
        c252 = v14921;	// L23028
        int8_t v14922 = a268;	// L23029
        v14901.write(v14922);	// L23030
      }
      ap_uint<128> packed_tmp252;	// L23032
      ap_uint<128> v14924 = v14902.read();	// L23033
      packed_tmp252 = v14924;	// L23034
      ap_uint<128> packed_c252;	// L23035
      packed_c252 = 0;	// L23036
      l_S_m_1_m252: for (int m252 = 0; m252 < 16; m252++) {	// L23037
        ap_int<33> v14927 = m252;	// L23038
        bool v14928 = v14927 == 15;	// L23039
        if (v14928) {	// L23040
          int8_t v14929 = c252;	// L23041
          ap_int<128> v14930 = packed_c252;	// L23042
          int64_t v14931 = m252;	// L23043
          int64_t v14932 = v14931 * 8;	// L23044
          ap_int<34> v14933 = m252;	// L23045
          ap_int<34> v14934 = v14933 + 1;	// L23046
          ap_int<66> v14935 = v14934;	// L23047
          ap_int<66> v14936 = v14935 * 8;	// L23048
          ap_int<66> v14937 = v14936 - 1;	// L23049
          int v14938 = v14932;	// L23050
          int v14939 = v14937;	// L23051
          ap_int<128> v14940;
          ap_int<128> v14940_tmp = v14930;
          v14940_tmp(v14939, v14938) = v14929;
          v14940 = v14940_tmp;	// L23052
          packed_c252 = v14940;	// L23053
        } else {
          ap_int<128> v14941 = packed_tmp252;	// L23055
          int64_t v14942 = m252;	// L23056
          int64_t v14943 = v14942 * 8;	// L23057
          ap_int<34> v14944 = m252;	// L23058
          ap_int<34> v14945 = v14944 + 1;	// L23059
          ap_int<66> v14946 = v14945;	// L23060
          ap_int<66> v14947 = v14946 * 8;	// L23061
          ap_int<66> v14948 = v14947 - 1;	// L23062
          int v14949 = v14943;	// L23063
          int v14950 = v14948;	// L23064
          int8_t v14951;
          ap_int<128> v14951_tmp = v14941;
          v14951 = v14951_tmp(v14950, v14949);	// L23065
          ap_int<128> v14952 = packed_c252;	// L23066
          ap_int<128> v14953;
          ap_int<128> v14953_tmp = v14952;
          v14953_tmp(v14950, v14949) = v14951;
          v14953 = v14953_tmp;	// L23076
          packed_c252 = v14953;	// L23077
        }
      }
      ap_int<128> v14954 = packed_c252;	// L23080
      v14903.write(v14954);	// L23081
    }
  }
}

void gemm_16_14(
  hls::stream< int8_t >& v14955,
  hls::stream< int8_t >& v14956,
  hls::stream< int8_t >& v14957,
  hls::stream< ap_uint<128> >& v14958,
  hls::stream< ap_uint<128> >& v14959
) {	// L23086
  l_S_mt_nt_0_mt288: for (int mt288 = 0; mt288 < 4; mt288++) {	// L23094
    l_nt288: for (int nt288 = 0; nt288 < 4; nt288++) {	// L23095
      int8_t c253;	// L23096
      c253 = 0;	// L23097
      l_S_k_0_k288: for (int k288 = 0; k288 < 64; k288++) {	// L23098
        int8_t v14964 = v14955.read();	// L23099
        int8_t a269;	// L23100
        a269 = v14964;	// L23101
        int8_t v14966 = v14956.read();	// L23102
        int8_t b269;	// L23103
        b269 = v14966;	// L23104
        int8_t v14968 = a269;	// L23105
        int8_t v14969 = b269;	// L23106
        int16_t v14970 = v14968;	// L23107
        int16_t v14971 = v14969;	// L23108
        int16_t v14972 = v14970 * v14971;	// L23109
        int8_t v14973 = c253;	// L23110
        ap_int<17> v14974 = v14973;	// L23111
        ap_int<17> v14975 = v14972;	// L23112
        ap_int<17> v14976 = v14974 + v14975;	// L23113
        int8_t v14977 = v14976;	// L23114
        c253 = v14977;	// L23115
        int8_t v14978 = a269;	// L23116
        v14957.write(v14978);	// L23117
      }
      ap_uint<128> packed_tmp253;	// L23119
      ap_uint<128> v14980 = v14958.read();	// L23120
      packed_tmp253 = v14980;	// L23121
      ap_uint<128> packed_c253;	// L23122
      packed_c253 = 0;	// L23123
      l_S_m_1_m253: for (int m253 = 0; m253 < 16; m253++) {	// L23124
        ap_int<33> v14983 = m253;	// L23125
        bool v14984 = v14983 == 15;	// L23126
        if (v14984) {	// L23127
          int8_t v14985 = c253;	// L23128
          ap_int<128> v14986 = packed_c253;	// L23129
          int64_t v14987 = m253;	// L23130
          int64_t v14988 = v14987 * 8;	// L23131
          ap_int<34> v14989 = m253;	// L23132
          ap_int<34> v14990 = v14989 + 1;	// L23133
          ap_int<66> v14991 = v14990;	// L23134
          ap_int<66> v14992 = v14991 * 8;	// L23135
          ap_int<66> v14993 = v14992 - 1;	// L23136
          int v14994 = v14988;	// L23137
          int v14995 = v14993;	// L23138
          ap_int<128> v14996;
          ap_int<128> v14996_tmp = v14986;
          v14996_tmp(v14995, v14994) = v14985;
          v14996 = v14996_tmp;	// L23139
          packed_c253 = v14996;	// L23140
        } else {
          ap_int<128> v14997 = packed_tmp253;	// L23142
          int64_t v14998 = m253;	// L23143
          int64_t v14999 = v14998 * 8;	// L23144
          ap_int<34> v15000 = m253;	// L23145
          ap_int<34> v15001 = v15000 + 1;	// L23146
          ap_int<66> v15002 = v15001;	// L23147
          ap_int<66> v15003 = v15002 * 8;	// L23148
          ap_int<66> v15004 = v15003 - 1;	// L23149
          int v15005 = v14999;	// L23150
          int v15006 = v15004;	// L23151
          int8_t v15007;
          ap_int<128> v15007_tmp = v14997;
          v15007 = v15007_tmp(v15006, v15005);	// L23152
          ap_int<128> v15008 = packed_c253;	// L23153
          ap_int<128> v15009;
          ap_int<128> v15009_tmp = v15008;
          v15009_tmp(v15006, v15005) = v15007;
          v15009 = v15009_tmp;	// L23163
          packed_c253 = v15009;	// L23164
        }
      }
      ap_int<128> v15010 = packed_c253;	// L23167
      v14959.write(v15010);	// L23168
    }
  }
}

void gemm_16_15(
  hls::stream< int8_t >& v15011,
  hls::stream< int8_t >& v15012,
  hls::stream< int8_t >& v15013,
  hls::stream< ap_uint<128> >& v15014,
  hls::stream< ap_uint<128> >& v15015
) {	// L23173
  l_S_mt_nt_0_mt289: for (int mt289 = 0; mt289 < 4; mt289++) {	// L23181
    l_nt289: for (int nt289 = 0; nt289 < 4; nt289++) {	// L23182
      int8_t c254;	// L23183
      c254 = 0;	// L23184
      l_S_k_0_k289: for (int k289 = 0; k289 < 64; k289++) {	// L23185
        int8_t v15020 = v15011.read();	// L23186
        int8_t a270;	// L23187
        a270 = v15020;	// L23188
        int8_t v15022 = v15012.read();	// L23189
        int8_t b270;	// L23190
        b270 = v15022;	// L23191
        int8_t v15024 = a270;	// L23192
        int8_t v15025 = b270;	// L23193
        int16_t v15026 = v15024;	// L23194
        int16_t v15027 = v15025;	// L23195
        int16_t v15028 = v15026 * v15027;	// L23196
        int8_t v15029 = c254;	// L23197
        ap_int<17> v15030 = v15029;	// L23198
        ap_int<17> v15031 = v15028;	// L23199
        ap_int<17> v15032 = v15030 + v15031;	// L23200
        int8_t v15033 = v15032;	// L23201
        c254 = v15033;	// L23202
        int8_t v15034 = a270;	// L23203
        v15013.write(v15034);	// L23204
      }
      ap_uint<128> packed_tmp254;	// L23206
      ap_uint<128> v15036 = v15014.read();	// L23207
      packed_tmp254 = v15036;	// L23208
      ap_uint<128> packed_c254;	// L23209
      packed_c254 = 0;	// L23210
      l_S_m_1_m254: for (int m254 = 0; m254 < 16; m254++) {	// L23211
        ap_int<33> v15039 = m254;	// L23212
        bool v15040 = v15039 == 15;	// L23213
        if (v15040) {	// L23214
          int8_t v15041 = c254;	// L23215
          ap_int<128> v15042 = packed_c254;	// L23216
          int64_t v15043 = m254;	// L23217
          int64_t v15044 = v15043 * 8;	// L23218
          ap_int<34> v15045 = m254;	// L23219
          ap_int<34> v15046 = v15045 + 1;	// L23220
          ap_int<66> v15047 = v15046;	// L23221
          ap_int<66> v15048 = v15047 * 8;	// L23222
          ap_int<66> v15049 = v15048 - 1;	// L23223
          int v15050 = v15044;	// L23224
          int v15051 = v15049;	// L23225
          ap_int<128> v15052;
          ap_int<128> v15052_tmp = v15042;
          v15052_tmp(v15051, v15050) = v15041;
          v15052 = v15052_tmp;	// L23226
          packed_c254 = v15052;	// L23227
        } else {
          ap_int<128> v15053 = packed_tmp254;	// L23229
          int64_t v15054 = m254;	// L23230
          int64_t v15055 = v15054 * 8;	// L23231
          ap_int<34> v15056 = m254;	// L23232
          ap_int<34> v15057 = v15056 + 1;	// L23233
          ap_int<66> v15058 = v15057;	// L23234
          ap_int<66> v15059 = v15058 * 8;	// L23235
          ap_int<66> v15060 = v15059 - 1;	// L23236
          int v15061 = v15055;	// L23237
          int v15062 = v15060;	// L23238
          int8_t v15063;
          ap_int<128> v15063_tmp = v15053;
          v15063 = v15063_tmp(v15062, v15061);	// L23239
          ap_int<128> v15064 = packed_c254;	// L23240
          ap_int<128> v15065;
          ap_int<128> v15065_tmp = v15064;
          v15065_tmp(v15062, v15061) = v15063;
          v15065 = v15065_tmp;	// L23250
          packed_c254 = v15065;	// L23251
        }
      }
      ap_int<128> v15066 = packed_c254;	// L23254
      v15015.write(v15066);	// L23255
    }
  }
}

void gemm_16_16(
  hls::stream< int8_t >& v15067,
  hls::stream< int8_t >& v15068,
  hls::stream< ap_uint<128> >& v15069,
  hls::stream< ap_uint<128> >& v15070
) {	// L23260
  l_S_mt_nt_0_mt290: for (int mt290 = 0; mt290 < 4; mt290++) {	// L23268
    l_nt290: for (int nt290 = 0; nt290 < 4; nt290++) {	// L23269
      int8_t c255;	// L23270
      c255 = 0;	// L23271
      l_S_k_0_k290: for (int k290 = 0; k290 < 64; k290++) {	// L23272
        int8_t v15075 = v15067.read();	// L23273
        int8_t a271;	// L23274
        a271 = v15075;	// L23275
        int8_t v15077 = v15068.read();	// L23276
        int8_t b271;	// L23277
        b271 = v15077;	// L23278
        int8_t v15079 = a271;	// L23279
        int8_t v15080 = b271;	// L23280
        int16_t v15081 = v15079;	// L23281
        int16_t v15082 = v15080;	// L23282
        int16_t v15083 = v15081 * v15082;	// L23283
        int8_t v15084 = c255;	// L23284
        ap_int<17> v15085 = v15084;	// L23285
        ap_int<17> v15086 = v15083;	// L23286
        ap_int<17> v15087 = v15085 + v15086;	// L23287
        int8_t v15088 = v15087;	// L23288
        c255 = v15088;	// L23289
      }
      ap_uint<128> packed_tmp255;	// L23291
      ap_uint<128> v15090 = v15069.read();	// L23292
      packed_tmp255 = v15090;	// L23293
      ap_uint<128> packed_c255;	// L23294
      packed_c255 = 0;	// L23295
      l_S_m_1_m255: for (int m255 = 0; m255 < 16; m255++) {	// L23296
        ap_int<33> v15093 = m255;	// L23297
        bool v15094 = v15093 == 15;	// L23298
        if (v15094) {	// L23299
          int8_t v15095 = c255;	// L23300
          ap_int<128> v15096 = packed_c255;	// L23301
          int64_t v15097 = m255;	// L23302
          int64_t v15098 = v15097 * 8;	// L23303
          ap_int<34> v15099 = m255;	// L23304
          ap_int<34> v15100 = v15099 + 1;	// L23305
          ap_int<66> v15101 = v15100;	// L23306
          ap_int<66> v15102 = v15101 * 8;	// L23307
          ap_int<66> v15103 = v15102 - 1;	// L23308
          int v15104 = v15098;	// L23309
          int v15105 = v15103;	// L23310
          ap_int<128> v15106;
          ap_int<128> v15106_tmp = v15096;
          v15106_tmp(v15105, v15104) = v15095;
          v15106 = v15106_tmp;	// L23311
          packed_c255 = v15106;	// L23312
        } else {
          ap_int<128> v15107 = packed_tmp255;	// L23314
          int64_t v15108 = m255;	// L23315
          int64_t v15109 = v15108 * 8;	// L23316
          ap_int<34> v15110 = m255;	// L23317
          ap_int<34> v15111 = v15110 + 1;	// L23318
          ap_int<66> v15112 = v15111;	// L23319
          ap_int<66> v15113 = v15112 * 8;	// L23320
          ap_int<66> v15114 = v15113 - 1;	// L23321
          int v15115 = v15109;	// L23322
          int v15116 = v15114;	// L23323
          int8_t v15117;
          ap_int<128> v15117_tmp = v15107;
          v15117 = v15117_tmp(v15116, v15115);	// L23324
          ap_int<128> v15118 = packed_c255;	// L23325
          ap_int<128> v15119;
          ap_int<128> v15119_tmp = v15118;
          v15119_tmp(v15116, v15115) = v15117;
          v15119 = v15119_tmp;	// L23335
          packed_c255 = v15119;	// L23336
        }
      }
      ap_int<128> v15120 = packed_c255;	// L23339
      v15070.write(v15120);	// L23340
    }
  }
}

void gemm_17_1(
  hls::stream< ap_uint<128> >& v15121,
  hls::stream< ap_uint<128> >& v15122
) {	// L23345
  l_S_mt_nt_0_mt291: for (int mt291 = 0; mt291 < 4; mt291++) {	// L23346
    l_nt291: for (int nt291 = 0; nt291 < 4; nt291++) {	// L23347
      ap_uint<128> v15125 = v15121.read();	// L23348
      ap_uint<128> c_C;	// L23349
      c_C = v15125;	// L23350
      ap_int<128> v15127 = c_C;	// L23351
      v15122.write(v15127);	// L23352
    }
  }
}

void gemm_17_2(
  hls::stream< ap_uint<128> >& v15128,
  hls::stream< ap_uint<128> >& v15129,
  hls::stream< ap_uint<128> >& v15130
) {	// L23357
  l_S_mt_nt_0_mt292: for (int mt292 = 0; mt292 < 4; mt292++) {	// L23358
    l_nt292: for (int nt292 = 0; nt292 < 4; nt292++) {	// L23359
      ap_uint<128> v15133 = v15128.read();	// L23360
      ap_uint<128> c_C1;	// L23361
      c_C1 = v15133;	// L23362
      ap_int<128> v15135 = c_C1;	// L23363
      v15129.write(v15135);	// L23364
      l_S_ind_0_ind: for (int ind = 0; ind < 1; ind++) {	// L23365
        ap_uint<128> v15137 = v15130.read();	// L23366
        v15129.write(v15137);	// L23367
      }
    }
  }
}

void gemm_17_3(
  hls::stream< ap_uint<128> >& v15138,
  hls::stream< ap_uint<128> >& v15139,
  hls::stream< ap_uint<128> >& v15140
) {	// L23373
  l_S_mt_nt_0_mt293: for (int mt293 = 0; mt293 < 4; mt293++) {	// L23374
    l_nt293: for (int nt293 = 0; nt293 < 4; nt293++) {	// L23375
      ap_uint<128> v15143 = v15138.read();	// L23376
      ap_uint<128> c_C2;	// L23377
      c_C2 = v15143;	// L23378
      ap_int<128> v15145 = c_C2;	// L23379
      v15139.write(v15145);	// L23380
      l_S_ind_0_ind1: for (int ind1 = 0; ind1 < 2; ind1++) {	// L23381
        ap_uint<128> v15147 = v15140.read();	// L23382
        v15139.write(v15147);	// L23383
      }
    }
  }
}

void gemm_17_4(
  hls::stream< ap_uint<128> >& v15148,
  hls::stream< ap_uint<128> >& v15149,
  hls::stream< ap_uint<128> >& v15150
) {	// L23389
  l_S_mt_nt_0_mt294: for (int mt294 = 0; mt294 < 4; mt294++) {	// L23390
    l_nt294: for (int nt294 = 0; nt294 < 4; nt294++) {	// L23391
      ap_uint<128> v15153 = v15148.read();	// L23392
      ap_uint<128> c_C3;	// L23393
      c_C3 = v15153;	// L23394
      ap_int<128> v15155 = c_C3;	// L23395
      v15149.write(v15155);	// L23396
      l_S_ind_0_ind2: for (int ind2 = 0; ind2 < 3; ind2++) {	// L23397
        ap_uint<128> v15157 = v15150.read();	// L23398
        v15149.write(v15157);	// L23399
      }
    }
  }
}

void gemm_17_5(
  hls::stream< ap_uint<128> >& v15158,
  hls::stream< ap_uint<128> >& v15159,
  hls::stream< ap_uint<128> >& v15160
) {	// L23405
  l_S_mt_nt_0_mt295: for (int mt295 = 0; mt295 < 4; mt295++) {	// L23406
    l_nt295: for (int nt295 = 0; nt295 < 4; nt295++) {	// L23407
      ap_uint<128> v15163 = v15158.read();	// L23408
      ap_uint<128> c_C4;	// L23409
      c_C4 = v15163;	// L23410
      ap_int<128> v15165 = c_C4;	// L23411
      v15159.write(v15165);	// L23412
      l_S_ind_0_ind3: for (int ind3 = 0; ind3 < 4; ind3++) {	// L23413
        ap_uint<128> v15167 = v15160.read();	// L23414
        v15159.write(v15167);	// L23415
      }
    }
  }
}

void gemm_17_6(
  hls::stream< ap_uint<128> >& v15168,
  hls::stream< ap_uint<128> >& v15169,
  hls::stream< ap_uint<128> >& v15170
) {	// L23421
  l_S_mt_nt_0_mt296: for (int mt296 = 0; mt296 < 4; mt296++) {	// L23422
    l_nt296: for (int nt296 = 0; nt296 < 4; nt296++) {	// L23423
      ap_uint<128> v15173 = v15168.read();	// L23424
      ap_uint<128> c_C5;	// L23425
      c_C5 = v15173;	// L23426
      ap_int<128> v15175 = c_C5;	// L23427
      v15169.write(v15175);	// L23428
      l_S_ind_0_ind4: for (int ind4 = 0; ind4 < 5; ind4++) {	// L23429
        ap_uint<128> v15177 = v15170.read();	// L23430
        v15169.write(v15177);	// L23431
      }
    }
  }
}

void gemm_17_7(
  hls::stream< ap_uint<128> >& v15178,
  hls::stream< ap_uint<128> >& v15179,
  hls::stream< ap_uint<128> >& v15180
) {	// L23437
  l_S_mt_nt_0_mt297: for (int mt297 = 0; mt297 < 4; mt297++) {	// L23438
    l_nt297: for (int nt297 = 0; nt297 < 4; nt297++) {	// L23439
      ap_uint<128> v15183 = v15178.read();	// L23440
      ap_uint<128> c_C6;	// L23441
      c_C6 = v15183;	// L23442
      ap_int<128> v15185 = c_C6;	// L23443
      v15179.write(v15185);	// L23444
      l_S_ind_0_ind5: for (int ind5 = 0; ind5 < 6; ind5++) {	// L23445
        ap_uint<128> v15187 = v15180.read();	// L23446
        v15179.write(v15187);	// L23447
      }
    }
  }
}

void gemm_17_8(
  hls::stream< ap_uint<128> >& v15188,
  hls::stream< ap_uint<128> >& v15189,
  hls::stream< ap_uint<128> >& v15190
) {	// L23453
  l_S_mt_nt_0_mt298: for (int mt298 = 0; mt298 < 4; mt298++) {	// L23454
    l_nt298: for (int nt298 = 0; nt298 < 4; nt298++) {	// L23455
      ap_uint<128> v15193 = v15188.read();	// L23456
      ap_uint<128> c_C7;	// L23457
      c_C7 = v15193;	// L23458
      ap_int<128> v15195 = c_C7;	// L23459
      v15189.write(v15195);	// L23460
      l_S_ind_0_ind6: for (int ind6 = 0; ind6 < 7; ind6++) {	// L23461
        ap_uint<128> v15197 = v15190.read();	// L23462
        v15189.write(v15197);	// L23463
      }
    }
  }
}

void gemm_17_9(
  hls::stream< ap_uint<128> >& v15198,
  hls::stream< ap_uint<128> >& v15199,
  hls::stream< ap_uint<128> >& v15200
) {	// L23469
  l_S_mt_nt_0_mt299: for (int mt299 = 0; mt299 < 4; mt299++) {	// L23470
    l_nt299: for (int nt299 = 0; nt299 < 4; nt299++) {	// L23471
      ap_uint<128> v15203 = v15198.read();	// L23472
      ap_uint<128> c_C8;	// L23473
      c_C8 = v15203;	// L23474
      ap_int<128> v15205 = c_C8;	// L23475
      v15199.write(v15205);	// L23476
      l_S_ind_0_ind7: for (int ind7 = 0; ind7 < 8; ind7++) {	// L23477
        ap_uint<128> v15207 = v15200.read();	// L23478
        v15199.write(v15207);	// L23479
      }
    }
  }
}

void gemm_17_10(
  hls::stream< ap_uint<128> >& v15208,
  hls::stream< ap_uint<128> >& v15209,
  hls::stream< ap_uint<128> >& v15210
) {	// L23485
  l_S_mt_nt_0_mt300: for (int mt300 = 0; mt300 < 4; mt300++) {	// L23486
    l_nt300: for (int nt300 = 0; nt300 < 4; nt300++) {	// L23487
      ap_uint<128> v15213 = v15208.read();	// L23488
      ap_uint<128> c_C9;	// L23489
      c_C9 = v15213;	// L23490
      ap_int<128> v15215 = c_C9;	// L23491
      v15209.write(v15215);	// L23492
      l_S_ind_0_ind8: for (int ind8 = 0; ind8 < 9; ind8++) {	// L23493
        ap_uint<128> v15217 = v15210.read();	// L23494
        v15209.write(v15217);	// L23495
      }
    }
  }
}

void gemm_17_11(
  hls::stream< ap_uint<128> >& v15218,
  hls::stream< ap_uint<128> >& v15219,
  hls::stream< ap_uint<128> >& v15220
) {	// L23501
  l_S_mt_nt_0_mt301: for (int mt301 = 0; mt301 < 4; mt301++) {	// L23502
    l_nt301: for (int nt301 = 0; nt301 < 4; nt301++) {	// L23503
      ap_uint<128> v15223 = v15218.read();	// L23504
      ap_uint<128> c_C10;	// L23505
      c_C10 = v15223;	// L23506
      ap_int<128> v15225 = c_C10;	// L23507
      v15219.write(v15225);	// L23508
      l_S_ind_0_ind9: for (int ind9 = 0; ind9 < 10; ind9++) {	// L23509
        ap_uint<128> v15227 = v15220.read();	// L23510
        v15219.write(v15227);	// L23511
      }
    }
  }
}

void gemm_17_12(
  hls::stream< ap_uint<128> >& v15228,
  hls::stream< ap_uint<128> >& v15229,
  hls::stream< ap_uint<128> >& v15230
) {	// L23517
  l_S_mt_nt_0_mt302: for (int mt302 = 0; mt302 < 4; mt302++) {	// L23518
    l_nt302: for (int nt302 = 0; nt302 < 4; nt302++) {	// L23519
      ap_uint<128> v15233 = v15228.read();	// L23520
      ap_uint<128> c_C11;	// L23521
      c_C11 = v15233;	// L23522
      ap_int<128> v15235 = c_C11;	// L23523
      v15229.write(v15235);	// L23524
      l_S_ind_0_ind10: for (int ind10 = 0; ind10 < 11; ind10++) {	// L23525
        ap_uint<128> v15237 = v15230.read();	// L23526
        v15229.write(v15237);	// L23527
      }
    }
  }
}

void gemm_17_13(
  hls::stream< ap_uint<128> >& v15238,
  hls::stream< ap_uint<128> >& v15239,
  hls::stream< ap_uint<128> >& v15240
) {	// L23533
  l_S_mt_nt_0_mt303: for (int mt303 = 0; mt303 < 4; mt303++) {	// L23534
    l_nt303: for (int nt303 = 0; nt303 < 4; nt303++) {	// L23535
      ap_uint<128> v15243 = v15238.read();	// L23536
      ap_uint<128> c_C12;	// L23537
      c_C12 = v15243;	// L23538
      ap_int<128> v15245 = c_C12;	// L23539
      v15239.write(v15245);	// L23540
      l_S_ind_0_ind11: for (int ind11 = 0; ind11 < 12; ind11++) {	// L23541
        ap_uint<128> v15247 = v15240.read();	// L23542
        v15239.write(v15247);	// L23543
      }
    }
  }
}

void gemm_17_14(
  hls::stream< ap_uint<128> >& v15248,
  hls::stream< ap_uint<128> >& v15249,
  hls::stream< ap_uint<128> >& v15250
) {	// L23549
  l_S_mt_nt_0_mt304: for (int mt304 = 0; mt304 < 4; mt304++) {	// L23550
    l_nt304: for (int nt304 = 0; nt304 < 4; nt304++) {	// L23551
      ap_uint<128> v15253 = v15248.read();	// L23552
      ap_uint<128> c_C13;	// L23553
      c_C13 = v15253;	// L23554
      ap_int<128> v15255 = c_C13;	// L23555
      v15249.write(v15255);	// L23556
      l_S_ind_0_ind12: for (int ind12 = 0; ind12 < 13; ind12++) {	// L23557
        ap_uint<128> v15257 = v15250.read();	// L23558
        v15249.write(v15257);	// L23559
      }
    }
  }
}

void gemm_17_15(
  hls::stream< ap_uint<128> >& v15258,
  hls::stream< ap_uint<128> >& v15259,
  hls::stream< ap_uint<128> >& v15260
) {	// L23565
  l_S_mt_nt_0_mt305: for (int mt305 = 0; mt305 < 4; mt305++) {	// L23566
    l_nt305: for (int nt305 = 0; nt305 < 4; nt305++) {	// L23567
      ap_uint<128> v15263 = v15258.read();	// L23568
      ap_uint<128> c_C14;	// L23569
      c_C14 = v15263;	// L23570
      ap_int<128> v15265 = c_C14;	// L23571
      v15259.write(v15265);	// L23572
      l_S_ind_0_ind13: for (int ind13 = 0; ind13 < 14; ind13++) {	// L23573
        ap_uint<128> v15267 = v15260.read();	// L23574
        v15259.write(v15267);	// L23575
      }
    }
  }
}

void gemm_17_16(
  hls::stream< ap_uint<128> >& v15268,
  hls::stream< ap_uint<128> >& v15269,
  hls::stream< ap_uint<128> >& v15270
) {	// L23581
  l_S_mt_nt_0_mt306: for (int mt306 = 0; mt306 < 4; mt306++) {	// L23582
    l_nt306: for (int nt306 = 0; nt306 < 4; nt306++) {	// L23583
      ap_uint<128> v15273 = v15268.read();	// L23584
      ap_uint<128> c_C15;	// L23585
      c_C15 = v15273;	// L23586
      ap_int<128> v15275 = c_C15;	// L23587
      v15269.write(v15275);	// L23588
      l_S_ind_0_ind14: for (int ind14 = 0; ind14 < 15; ind14++) {	// L23589
        ap_uint<128> v15277 = v15270.read();	// L23590
        v15269.write(v15277);	// L23591
      }
    }
  }
}

void gemm_17_17(
  hls::stream< ap_uint<128> >& v15278,
  hls::stream< ap_uint<128> >& v15279
) {	// L23597
  l_S_mt_nt_0_mt307: for (int mt307 = 0; mt307 < 4; mt307++) {	// L23598
    l_nt307: for (int nt307 = 0; nt307 < 4; nt307++) {	// L23599
      l_S_n_0_n: for (int n = 0; n < 16; n++) {	// L23600
        ap_uint<128> v15283 = v15278.read();	// L23601
        v15279.write(v15283);	// L23602
      }
    }
  }
}

void offchip_store_0(
  ap_uint<128> v15284[256],
  hls::stream< ap_uint<128> >& v15285
) {	// L23608
  l_S_mt_nt_0_mt308: for (int mt308 = 0; mt308 < 4; mt308++) {	// L23609
    l_nt308: for (int nt308 = 0; nt308 < 4; nt308++) {	// L23610
      l_S_n_0_n1: for (int n1 = 0; n1 < 16; n1++) {	// L23611
        ap_uint<128> v15289 = v15285.read();	// L23612
        v15284[(((mt308 * 64) + (nt308 * 16)) + n1)] = v15289;	// L23613
      }
    }
  }
}

/// This is top function.
void top(
  ap_uint<128> *v15290,
  ap_uint<128> *v15291,
  ap_uint<128> *v15292
) {	// L23619
  #pragma HLS interface m_axi port=v15290 offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=v15291 offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=v15292 offset=slave bundle=gmem2
  #pragma HLS dataflow
  hls::stream< ap_uint<128> > v15293;
  #pragma HLS stream variable=v15293 depth=4	// L23620
  hls::stream< ap_uint<128> > v15294;
  #pragma HLS stream variable=v15294 depth=4	// L23621
  hls::stream< ap_uint<128> > v15295;
  #pragma HLS stream variable=v15295 depth=4	// L23622
  hls::stream< ap_uint<128> > v15296;
  #pragma HLS stream variable=v15296 depth=2	// L23623
  hls::stream< ap_uint<128> > v15297;
  #pragma HLS stream variable=v15297 depth=2	// L23624
  hls::stream< ap_uint<128> > v15298;
  #pragma HLS stream variable=v15298 depth=2	// L23625
  hls::stream< ap_uint<128> > v15299;
  #pragma HLS stream variable=v15299 depth=2	// L23626
  hls::stream< ap_uint<128> > v15300;
  #pragma HLS stream variable=v15300 depth=2	// L23627
  hls::stream< ap_uint<128> > v15301;
  #pragma HLS stream variable=v15301 depth=2	// L23628
  hls::stream< ap_uint<128> > v15302;
  #pragma HLS stream variable=v15302 depth=2	// L23629
  hls::stream< ap_uint<128> > v15303;
  #pragma HLS stream variable=v15303 depth=2	// L23630
  hls::stream< ap_uint<128> > v15304;
  #pragma HLS stream variable=v15304 depth=2	// L23631
  hls::stream< ap_uint<128> > v15305;
  #pragma HLS stream variable=v15305 depth=2	// L23632
  hls::stream< ap_uint<128> > v15306;
  #pragma HLS stream variable=v15306 depth=2	// L23633
  hls::stream< ap_uint<128> > v15307;
  #pragma HLS stream variable=v15307 depth=2	// L23634
  hls::stream< ap_uint<128> > v15308;
  #pragma HLS stream variable=v15308 depth=2	// L23635
  hls::stream< ap_uint<128> > v15309;
  #pragma HLS stream variable=v15309 depth=2	// L23636
  hls::stream< ap_uint<128> > v15310;
  #pragma HLS stream variable=v15310 depth=2	// L23637
  hls::stream< ap_uint<128> > v15311;
  #pragma HLS stream variable=v15311 depth=2	// L23638
  hls::stream< ap_uint<128> > v15312;
  #pragma HLS stream variable=v15312 depth=2	// L23639
  hls::stream< ap_uint<128> > v15313;
  #pragma HLS stream variable=v15313 depth=2	// L23640
  hls::stream< ap_uint<128> > v15314;
  #pragma HLS stream variable=v15314 depth=2	// L23641
  hls::stream< ap_uint<128> > v15315;
  #pragma HLS stream variable=v15315 depth=2	// L23642
  hls::stream< ap_uint<128> > v15316;
  #pragma HLS stream variable=v15316 depth=2	// L23643
  hls::stream< ap_uint<128> > v15317;
  #pragma HLS stream variable=v15317 depth=2	// L23644
  hls::stream< ap_uint<128> > v15318;
  #pragma HLS stream variable=v15318 depth=2	// L23645
  hls::stream< ap_uint<128> > v15319;
  #pragma HLS stream variable=v15319 depth=2	// L23646
  hls::stream< ap_uint<128> > v15320;
  #pragma HLS stream variable=v15320 depth=2	// L23647
  hls::stream< ap_uint<128> > v15321;
  #pragma HLS stream variable=v15321 depth=2	// L23648
  hls::stream< ap_uint<128> > v15322;
  #pragma HLS stream variable=v15322 depth=2	// L23649
  hls::stream< ap_uint<128> > v15323;
  #pragma HLS stream variable=v15323 depth=2	// L23650
  hls::stream< ap_uint<128> > v15324;
  #pragma HLS stream variable=v15324 depth=2	// L23651
  hls::stream< ap_uint<128> > v15325;
  #pragma HLS stream variable=v15325 depth=2	// L23652
  hls::stream< ap_uint<128> > v15326;
  #pragma HLS stream variable=v15326 depth=2	// L23653
  hls::stream< ap_uint<128> > v15327;
  #pragma HLS stream variable=v15327 depth=2	// L23654
  hls::stream< ap_uint<128> > v15328;
  #pragma HLS stream variable=v15328 depth=2	// L23655
  hls::stream< ap_uint<128> > v15329;
  #pragma HLS stream variable=v15329 depth=2	// L23656
  hls::stream< ap_uint<128> > v15330;
  #pragma HLS stream variable=v15330 depth=2	// L23657
  hls::stream< ap_uint<128> > v15331;
  #pragma HLS stream variable=v15331 depth=2	// L23658
  hls::stream< ap_uint<128> > v15332;
  #pragma HLS stream variable=v15332 depth=2	// L23659
  hls::stream< ap_uint<128> > v15333;
  #pragma HLS stream variable=v15333 depth=2	// L23660
  hls::stream< ap_uint<128> > v15334;
  #pragma HLS stream variable=v15334 depth=2	// L23661
  hls::stream< ap_uint<128> > v15335;
  #pragma HLS stream variable=v15335 depth=2	// L23662
  hls::stream< ap_uint<128> > v15336;
  #pragma HLS stream variable=v15336 depth=2	// L23663
  hls::stream< ap_uint<128> > v15337;
  #pragma HLS stream variable=v15337 depth=2	// L23664
  hls::stream< ap_uint<128> > v15338;
  #pragma HLS stream variable=v15338 depth=2	// L23665
  hls::stream< ap_uint<128> > v15339;
  #pragma HLS stream variable=v15339 depth=2	// L23666
  hls::stream< ap_uint<128> > v15340;
  #pragma HLS stream variable=v15340 depth=2	// L23667
  hls::stream< ap_uint<128> > v15341;
  #pragma HLS stream variable=v15341 depth=2	// L23668
  hls::stream< ap_uint<128> > v15342;
  #pragma HLS stream variable=v15342 depth=2	// L23669
  hls::stream< ap_uint<128> > v15343;
  #pragma HLS stream variable=v15343 depth=2	// L23670
  hls::stream< ap_uint<128> > v15344;
  #pragma HLS stream variable=v15344 depth=2	// L23671
  hls::stream< ap_uint<128> > v15345;
  #pragma HLS stream variable=v15345 depth=2	// L23672
  hls::stream< ap_uint<128> > v15346;
  #pragma HLS stream variable=v15346 depth=2	// L23673
  hls::stream< ap_uint<128> > v15347;
  #pragma HLS stream variable=v15347 depth=2	// L23674
  hls::stream< ap_uint<128> > v15348;
  #pragma HLS stream variable=v15348 depth=2	// L23675
  hls::stream< ap_uint<128> > v15349;
  #pragma HLS stream variable=v15349 depth=2	// L23676
  hls::stream< ap_uint<128> > v15350;
  #pragma HLS stream variable=v15350 depth=2	// L23677
  hls::stream< ap_uint<128> > v15351;
  #pragma HLS stream variable=v15351 depth=2	// L23678
  hls::stream< ap_uint<128> > v15352;
  #pragma HLS stream variable=v15352 depth=2	// L23679
  hls::stream< ap_uint<128> > v15353;
  #pragma HLS stream variable=v15353 depth=2	// L23680
  hls::stream< ap_uint<128> > v15354;
  #pragma HLS stream variable=v15354 depth=2	// L23681
  hls::stream< ap_uint<128> > v15355;
  #pragma HLS stream variable=v15355 depth=2	// L23682
  hls::stream< ap_uint<128> > v15356;
  #pragma HLS stream variable=v15356 depth=2	// L23683
  hls::stream< ap_uint<128> > v15357;
  #pragma HLS stream variable=v15357 depth=2	// L23684
  hls::stream< ap_uint<128> > v15358;
  #pragma HLS stream variable=v15358 depth=2	// L23685
  hls::stream< ap_uint<128> > v15359;
  #pragma HLS stream variable=v15359 depth=2	// L23686
  hls::stream< ap_uint<128> > v15360;
  #pragma HLS stream variable=v15360 depth=2	// L23687
  hls::stream< ap_uint<128> > v15361;
  #pragma HLS stream variable=v15361 depth=2	// L23688
  hls::stream< ap_uint<128> > v15362;
  #pragma HLS stream variable=v15362 depth=2	// L23689
  hls::stream< ap_uint<128> > v15363;
  #pragma HLS stream variable=v15363 depth=2	// L23690
  hls::stream< ap_uint<128> > v15364;
  #pragma HLS stream variable=v15364 depth=2	// L23691
  hls::stream< ap_uint<128> > v15365;
  #pragma HLS stream variable=v15365 depth=2	// L23692
  hls::stream< ap_uint<128> > v15366;
  #pragma HLS stream variable=v15366 depth=2	// L23693
  hls::stream< ap_uint<128> > v15367;
  #pragma HLS stream variable=v15367 depth=2	// L23694
  hls::stream< ap_uint<128> > v15368;
  #pragma HLS stream variable=v15368 depth=2	// L23695
  hls::stream< ap_uint<128> > v15369;
  #pragma HLS stream variable=v15369 depth=2	// L23696
  hls::stream< ap_uint<128> > v15370;
  #pragma HLS stream variable=v15370 depth=2	// L23697
  hls::stream< ap_uint<128> > v15371;
  #pragma HLS stream variable=v15371 depth=2	// L23698
  hls::stream< ap_uint<128> > v15372;
  #pragma HLS stream variable=v15372 depth=2	// L23699
  hls::stream< ap_uint<128> > v15373;
  #pragma HLS stream variable=v15373 depth=2	// L23700
  hls::stream< ap_uint<128> > v15374;
  #pragma HLS stream variable=v15374 depth=2	// L23701
  hls::stream< ap_uint<128> > v15375;
  #pragma HLS stream variable=v15375 depth=2	// L23702
  hls::stream< ap_uint<128> > v15376;
  #pragma HLS stream variable=v15376 depth=2	// L23703
  hls::stream< ap_uint<128> > v15377;
  #pragma HLS stream variable=v15377 depth=2	// L23704
  hls::stream< ap_uint<128> > v15378;
  #pragma HLS stream variable=v15378 depth=2	// L23705
  hls::stream< ap_uint<128> > v15379;
  #pragma HLS stream variable=v15379 depth=2	// L23706
  hls::stream< ap_uint<128> > v15380;
  #pragma HLS stream variable=v15380 depth=2	// L23707
  hls::stream< ap_uint<128> > v15381;
  #pragma HLS stream variable=v15381 depth=2	// L23708
  hls::stream< ap_uint<128> > v15382;
  #pragma HLS stream variable=v15382 depth=2	// L23709
  hls::stream< ap_uint<128> > v15383;
  #pragma HLS stream variable=v15383 depth=2	// L23710
  hls::stream< ap_uint<128> > v15384;
  #pragma HLS stream variable=v15384 depth=2	// L23711
  hls::stream< ap_uint<128> > v15385;
  #pragma HLS stream variable=v15385 depth=2	// L23712
  hls::stream< ap_uint<128> > v15386;
  #pragma HLS stream variable=v15386 depth=2	// L23713
  hls::stream< ap_uint<128> > v15387;
  #pragma HLS stream variable=v15387 depth=2	// L23714
  hls::stream< ap_uint<128> > v15388;
  #pragma HLS stream variable=v15388 depth=2	// L23715
  hls::stream< ap_uint<128> > v15389;
  #pragma HLS stream variable=v15389 depth=2	// L23716
  hls::stream< ap_uint<128> > v15390;
  #pragma HLS stream variable=v15390 depth=2	// L23717
  hls::stream< ap_uint<128> > v15391;
  #pragma HLS stream variable=v15391 depth=2	// L23718
  hls::stream< ap_uint<128> > v15392;
  #pragma HLS stream variable=v15392 depth=2	// L23719
  hls::stream< ap_uint<128> > v15393;
  #pragma HLS stream variable=v15393 depth=2	// L23720
  hls::stream< ap_uint<128> > v15394;
  #pragma HLS stream variable=v15394 depth=2	// L23721
  hls::stream< ap_uint<128> > v15395;
  #pragma HLS stream variable=v15395 depth=2	// L23722
  hls::stream< ap_uint<128> > v15396;
  #pragma HLS stream variable=v15396 depth=2	// L23723
  hls::stream< ap_uint<128> > v15397;
  #pragma HLS stream variable=v15397 depth=2	// L23724
  hls::stream< ap_uint<128> > v15398;
  #pragma HLS stream variable=v15398 depth=2	// L23725
  hls::stream< ap_uint<128> > v15399;
  #pragma HLS stream variable=v15399 depth=2	// L23726
  hls::stream< ap_uint<128> > v15400;
  #pragma HLS stream variable=v15400 depth=2	// L23727
  hls::stream< ap_uint<128> > v15401;
  #pragma HLS stream variable=v15401 depth=2	// L23728
  hls::stream< ap_uint<128> > v15402;
  #pragma HLS stream variable=v15402 depth=2	// L23729
  hls::stream< ap_uint<128> > v15403;
  #pragma HLS stream variable=v15403 depth=2	// L23730
  hls::stream< ap_uint<128> > v15404;
  #pragma HLS stream variable=v15404 depth=2	// L23731
  hls::stream< ap_uint<128> > v15405;
  #pragma HLS stream variable=v15405 depth=2	// L23732
  hls::stream< ap_uint<128> > v15406;
  #pragma HLS stream variable=v15406 depth=2	// L23733
  hls::stream< ap_uint<128> > v15407;
  #pragma HLS stream variable=v15407 depth=2	// L23734
  hls::stream< ap_uint<128> > v15408;
  #pragma HLS stream variable=v15408 depth=2	// L23735
  hls::stream< ap_uint<128> > v15409;
  #pragma HLS stream variable=v15409 depth=2	// L23736
  hls::stream< ap_uint<128> > v15410;
  #pragma HLS stream variable=v15410 depth=2	// L23737
  hls::stream< ap_uint<128> > v15411;
  #pragma HLS stream variable=v15411 depth=2	// L23738
  hls::stream< ap_uint<128> > v15412;
  #pragma HLS stream variable=v15412 depth=2	// L23739
  hls::stream< ap_uint<128> > v15413;
  #pragma HLS stream variable=v15413 depth=2	// L23740
  hls::stream< ap_uint<128> > v15414;
  #pragma HLS stream variable=v15414 depth=2	// L23741
  hls::stream< ap_uint<128> > v15415;
  #pragma HLS stream variable=v15415 depth=2	// L23742
  hls::stream< ap_uint<128> > v15416;
  #pragma HLS stream variable=v15416 depth=2	// L23743
  hls::stream< ap_uint<128> > v15417;
  #pragma HLS stream variable=v15417 depth=2	// L23744
  hls::stream< ap_uint<128> > v15418;
  #pragma HLS stream variable=v15418 depth=2	// L23745
  hls::stream< ap_uint<128> > v15419;
  #pragma HLS stream variable=v15419 depth=2	// L23746
  hls::stream< ap_uint<128> > v15420;
  #pragma HLS stream variable=v15420 depth=2	// L23747
  hls::stream< ap_uint<128> > v15421;
  #pragma HLS stream variable=v15421 depth=2	// L23748
  hls::stream< ap_uint<128> > v15422;
  #pragma HLS stream variable=v15422 depth=2	// L23749
  hls::stream< ap_uint<128> > v15423;
  #pragma HLS stream variable=v15423 depth=2	// L23750
  hls::stream< ap_uint<128> > v15424;
  #pragma HLS stream variable=v15424 depth=2	// L23751
  hls::stream< ap_uint<128> > v15425;
  #pragma HLS stream variable=v15425 depth=2	// L23752
  hls::stream< ap_uint<128> > v15426;
  #pragma HLS stream variable=v15426 depth=2	// L23753
  hls::stream< ap_uint<128> > v15427;
  #pragma HLS stream variable=v15427 depth=2	// L23754
  hls::stream< ap_uint<128> > v15428;
  #pragma HLS stream variable=v15428 depth=2	// L23755
  hls::stream< ap_uint<128> > v15429;
  #pragma HLS stream variable=v15429 depth=2	// L23756
  hls::stream< ap_uint<128> > v15430;
  #pragma HLS stream variable=v15430 depth=2	// L23757
  hls::stream< ap_uint<128> > v15431;
  #pragma HLS stream variable=v15431 depth=2	// L23758
  hls::stream< ap_uint<128> > v15432;
  #pragma HLS stream variable=v15432 depth=2	// L23759
  hls::stream< ap_uint<128> > v15433;
  #pragma HLS stream variable=v15433 depth=2	// L23760
  hls::stream< ap_uint<128> > v15434;
  #pragma HLS stream variable=v15434 depth=2	// L23761
  hls::stream< ap_uint<128> > v15435;
  #pragma HLS stream variable=v15435 depth=2	// L23762
  hls::stream< ap_uint<128> > v15436;
  #pragma HLS stream variable=v15436 depth=2	// L23763
  hls::stream< ap_uint<128> > v15437;
  #pragma HLS stream variable=v15437 depth=2	// L23764
  hls::stream< ap_uint<128> > v15438;
  #pragma HLS stream variable=v15438 depth=2	// L23765
  hls::stream< ap_uint<128> > v15439;
  #pragma HLS stream variable=v15439 depth=2	// L23766
  hls::stream< ap_uint<128> > v15440;
  #pragma HLS stream variable=v15440 depth=2	// L23767
  hls::stream< ap_uint<128> > v15441;
  #pragma HLS stream variable=v15441 depth=2	// L23768
  hls::stream< ap_uint<128> > v15442;
  #pragma HLS stream variable=v15442 depth=2	// L23769
  hls::stream< ap_uint<128> > v15443;
  #pragma HLS stream variable=v15443 depth=2	// L23770
  hls::stream< ap_uint<128> > v15444;
  #pragma HLS stream variable=v15444 depth=2	// L23771
  hls::stream< ap_uint<128> > v15445;
  #pragma HLS stream variable=v15445 depth=2	// L23772
  hls::stream< ap_uint<128> > v15446;
  #pragma HLS stream variable=v15446 depth=2	// L23773
  hls::stream< ap_uint<128> > v15447;
  #pragma HLS stream variable=v15447 depth=2	// L23774
  hls::stream< ap_uint<128> > v15448;
  #pragma HLS stream variable=v15448 depth=2	// L23775
  hls::stream< ap_uint<128> > v15449;
  #pragma HLS stream variable=v15449 depth=2	// L23776
  hls::stream< ap_uint<128> > v15450;
  #pragma HLS stream variable=v15450 depth=2	// L23777
  hls::stream< ap_uint<128> > v15451;
  #pragma HLS stream variable=v15451 depth=2	// L23778
  hls::stream< ap_uint<128> > v15452;
  #pragma HLS stream variable=v15452 depth=2	// L23779
  hls::stream< ap_uint<128> > v15453;
  #pragma HLS stream variable=v15453 depth=2	// L23780
  hls::stream< ap_uint<128> > v15454;
  #pragma HLS stream variable=v15454 depth=2	// L23781
  hls::stream< ap_uint<128> > v15455;
  #pragma HLS stream variable=v15455 depth=2	// L23782
  hls::stream< ap_uint<128> > v15456;
  #pragma HLS stream variable=v15456 depth=2	// L23783
  hls::stream< ap_uint<128> > v15457;
  #pragma HLS stream variable=v15457 depth=2	// L23784
  hls::stream< ap_uint<128> > v15458;
  #pragma HLS stream variable=v15458 depth=2	// L23785
  hls::stream< ap_uint<128> > v15459;
  #pragma HLS stream variable=v15459 depth=2	// L23786
  hls::stream< ap_uint<128> > v15460;
  #pragma HLS stream variable=v15460 depth=2	// L23787
  hls::stream< ap_uint<128> > v15461;
  #pragma HLS stream variable=v15461 depth=2	// L23788
  hls::stream< ap_uint<128> > v15462;
  #pragma HLS stream variable=v15462 depth=2	// L23789
  hls::stream< ap_uint<128> > v15463;
  #pragma HLS stream variable=v15463 depth=2	// L23790
  hls::stream< ap_uint<128> > v15464;
  #pragma HLS stream variable=v15464 depth=2	// L23791
  hls::stream< ap_uint<128> > v15465;
  #pragma HLS stream variable=v15465 depth=2	// L23792
  hls::stream< ap_uint<128> > v15466;
  #pragma HLS stream variable=v15466 depth=2	// L23793
  hls::stream< ap_uint<128> > v15467;
  #pragma HLS stream variable=v15467 depth=2	// L23794
  hls::stream< ap_uint<128> > v15468;
  #pragma HLS stream variable=v15468 depth=2	// L23795
  hls::stream< ap_uint<128> > v15469;
  #pragma HLS stream variable=v15469 depth=2	// L23796
  hls::stream< ap_uint<128> > v15470;
  #pragma HLS stream variable=v15470 depth=2	// L23797
  hls::stream< ap_uint<128> > v15471;
  #pragma HLS stream variable=v15471 depth=2	// L23798
  hls::stream< ap_uint<128> > v15472;
  #pragma HLS stream variable=v15472 depth=2	// L23799
  hls::stream< ap_uint<128> > v15473;
  #pragma HLS stream variable=v15473 depth=2	// L23800
  hls::stream< ap_uint<128> > v15474;
  #pragma HLS stream variable=v15474 depth=2	// L23801
  hls::stream< ap_uint<128> > v15475;
  #pragma HLS stream variable=v15475 depth=2	// L23802
  hls::stream< ap_uint<128> > v15476;
  #pragma HLS stream variable=v15476 depth=2	// L23803
  hls::stream< ap_uint<128> > v15477;
  #pragma HLS stream variable=v15477 depth=2	// L23804
  hls::stream< ap_uint<128> > v15478;
  #pragma HLS stream variable=v15478 depth=2	// L23805
  hls::stream< ap_uint<128> > v15479;
  #pragma HLS stream variable=v15479 depth=2	// L23806
  hls::stream< ap_uint<128> > v15480;
  #pragma HLS stream variable=v15480 depth=2	// L23807
  hls::stream< ap_uint<128> > v15481;
  #pragma HLS stream variable=v15481 depth=2	// L23808
  hls::stream< ap_uint<128> > v15482;
  #pragma HLS stream variable=v15482 depth=2	// L23809
  hls::stream< ap_uint<128> > v15483;
  #pragma HLS stream variable=v15483 depth=2	// L23810
  hls::stream< ap_uint<128> > v15484;
  #pragma HLS stream variable=v15484 depth=2	// L23811
  hls::stream< ap_uint<128> > v15485;
  #pragma HLS stream variable=v15485 depth=2	// L23812
  hls::stream< ap_uint<128> > v15486;
  #pragma HLS stream variable=v15486 depth=2	// L23813
  hls::stream< ap_uint<128> > v15487;
  #pragma HLS stream variable=v15487 depth=2	// L23814
  hls::stream< ap_uint<128> > v15488;
  #pragma HLS stream variable=v15488 depth=2	// L23815
  hls::stream< ap_uint<128> > v15489;
  #pragma HLS stream variable=v15489 depth=2	// L23816
  hls::stream< ap_uint<128> > v15490;
  #pragma HLS stream variable=v15490 depth=2	// L23817
  hls::stream< ap_uint<128> > v15491;
  #pragma HLS stream variable=v15491 depth=2	// L23818
  hls::stream< ap_uint<128> > v15492;
  #pragma HLS stream variable=v15492 depth=2	// L23819
  hls::stream< ap_uint<128> > v15493;
  #pragma HLS stream variable=v15493 depth=2	// L23820
  hls::stream< ap_uint<128> > v15494;
  #pragma HLS stream variable=v15494 depth=2	// L23821
  hls::stream< ap_uint<128> > v15495;
  #pragma HLS stream variable=v15495 depth=2	// L23822
  hls::stream< ap_uint<128> > v15496;
  #pragma HLS stream variable=v15496 depth=2	// L23823
  hls::stream< ap_uint<128> > v15497;
  #pragma HLS stream variable=v15497 depth=2	// L23824
  hls::stream< ap_uint<128> > v15498;
  #pragma HLS stream variable=v15498 depth=2	// L23825
  hls::stream< ap_uint<128> > v15499;
  #pragma HLS stream variable=v15499 depth=2	// L23826
  hls::stream< ap_uint<128> > v15500;
  #pragma HLS stream variable=v15500 depth=2	// L23827
  hls::stream< ap_uint<128> > v15501;
  #pragma HLS stream variable=v15501 depth=2	// L23828
  hls::stream< ap_uint<128> > v15502;
  #pragma HLS stream variable=v15502 depth=2	// L23829
  hls::stream< ap_uint<128> > v15503;
  #pragma HLS stream variable=v15503 depth=2	// L23830
  hls::stream< ap_uint<128> > v15504;
  #pragma HLS stream variable=v15504 depth=2	// L23831
  hls::stream< ap_uint<128> > v15505;
  #pragma HLS stream variable=v15505 depth=2	// L23832
  hls::stream< ap_uint<128> > v15506;
  #pragma HLS stream variable=v15506 depth=2	// L23833
  hls::stream< ap_uint<128> > v15507;
  #pragma HLS stream variable=v15507 depth=2	// L23834
  hls::stream< ap_uint<128> > v15508;
  #pragma HLS stream variable=v15508 depth=2	// L23835
  hls::stream< ap_uint<128> > v15509;
  #pragma HLS stream variable=v15509 depth=2	// L23836
  hls::stream< ap_uint<128> > v15510;
  #pragma HLS stream variable=v15510 depth=2	// L23837
  hls::stream< ap_uint<128> > v15511;
  #pragma HLS stream variable=v15511 depth=2	// L23838
  hls::stream< ap_uint<128> > v15512;
  #pragma HLS stream variable=v15512 depth=2	// L23839
  hls::stream< ap_uint<128> > v15513;
  #pragma HLS stream variable=v15513 depth=2	// L23840
  hls::stream< ap_uint<128> > v15514;
  #pragma HLS stream variable=v15514 depth=2	// L23841
  hls::stream< ap_uint<128> > v15515;
  #pragma HLS stream variable=v15515 depth=2	// L23842
  hls::stream< ap_uint<128> > v15516;
  #pragma HLS stream variable=v15516 depth=2	// L23843
  hls::stream< ap_uint<128> > v15517;
  #pragma HLS stream variable=v15517 depth=2	// L23844
  hls::stream< ap_uint<128> > v15518;
  #pragma HLS stream variable=v15518 depth=2	// L23845
  hls::stream< ap_uint<128> > v15519;
  #pragma HLS stream variable=v15519 depth=2	// L23846
  hls::stream< ap_uint<128> > v15520;
  #pragma HLS stream variable=v15520 depth=2	// L23847
  hls::stream< ap_uint<128> > v15521;
  #pragma HLS stream variable=v15521 depth=2	// L23848
  hls::stream< ap_uint<128> > v15522;
  #pragma HLS stream variable=v15522 depth=2	// L23849
  hls::stream< ap_uint<128> > v15523;
  #pragma HLS stream variable=v15523 depth=2	// L23850
  hls::stream< ap_uint<128> > v15524;
  #pragma HLS stream variable=v15524 depth=2	// L23851
  hls::stream< ap_uint<128> > v15525;
  #pragma HLS stream variable=v15525 depth=2	// L23852
  hls::stream< ap_uint<128> > v15526;
  #pragma HLS stream variable=v15526 depth=2	// L23853
  hls::stream< ap_uint<128> > v15527;
  #pragma HLS stream variable=v15527 depth=2	// L23854
  hls::stream< ap_uint<128> > v15528;
  #pragma HLS stream variable=v15528 depth=2	// L23855
  hls::stream< ap_uint<128> > v15529;
  #pragma HLS stream variable=v15529 depth=2	// L23856
  hls::stream< ap_uint<128> > v15530;
  #pragma HLS stream variable=v15530 depth=2	// L23857
  hls::stream< ap_uint<128> > v15531;
  #pragma HLS stream variable=v15531 depth=2	// L23858
  hls::stream< ap_uint<128> > v15532;
  #pragma HLS stream variable=v15532 depth=2	// L23859
  hls::stream< ap_uint<128> > v15533;
  #pragma HLS stream variable=v15533 depth=2	// L23860
  hls::stream< ap_uint<128> > v15534;
  #pragma HLS stream variable=v15534 depth=2	// L23861
  hls::stream< ap_uint<128> > v15535;
  #pragma HLS stream variable=v15535 depth=2	// L23862
  hls::stream< ap_uint<128> > v15536;
  #pragma HLS stream variable=v15536 depth=2	// L23863
  hls::stream< ap_uint<128> > v15537;
  #pragma HLS stream variable=v15537 depth=2	// L23864
  hls::stream< ap_uint<128> > v15538;
  #pragma HLS stream variable=v15538 depth=2	// L23865
  hls::stream< ap_uint<128> > v15539;
  #pragma HLS stream variable=v15539 depth=2	// L23866
  hls::stream< ap_uint<128> > v15540;
  #pragma HLS stream variable=v15540 depth=2	// L23867
  hls::stream< ap_uint<128> > v15541;
  #pragma HLS stream variable=v15541 depth=2	// L23868
  hls::stream< ap_uint<128> > v15542;
  #pragma HLS stream variable=v15542 depth=2	// L23869
  hls::stream< ap_uint<128> > v15543;
  #pragma HLS stream variable=v15543 depth=2	// L23870
  hls::stream< ap_uint<128> > v15544;
  #pragma HLS stream variable=v15544 depth=2	// L23871
  hls::stream< ap_uint<128> > v15545;
  #pragma HLS stream variable=v15545 depth=2	// L23872
  hls::stream< ap_uint<128> > v15546;
  #pragma HLS stream variable=v15546 depth=2	// L23873
  hls::stream< ap_uint<128> > v15547;
  #pragma HLS stream variable=v15547 depth=2	// L23874
  hls::stream< ap_uint<128> > v15548;
  #pragma HLS stream variable=v15548 depth=2	// L23875
  hls::stream< ap_uint<128> > v15549;
  #pragma HLS stream variable=v15549 depth=2	// L23876
  hls::stream< ap_uint<128> > v15550;
  #pragma HLS stream variable=v15550 depth=2	// L23877
  hls::stream< ap_uint<128> > v15551;
  #pragma HLS stream variable=v15551 depth=2	// L23878
  hls::stream< ap_uint<128> > v15552;
  #pragma HLS stream variable=v15552 depth=2	// L23879
  hls::stream< ap_uint<128> > v15553;
  #pragma HLS stream variable=v15553 depth=2	// L23880
  hls::stream< ap_uint<128> > v15554;
  #pragma HLS stream variable=v15554 depth=2	// L23881
  hls::stream< ap_uint<128> > v15555;
  #pragma HLS stream variable=v15555 depth=2	// L23882
  hls::stream< ap_uint<128> > v15556;
  #pragma HLS stream variable=v15556 depth=2	// L23883
  hls::stream< ap_uint<128> > v15557;
  #pragma HLS stream variable=v15557 depth=2	// L23884
  hls::stream< ap_uint<128> > v15558;
  #pragma HLS stream variable=v15558 depth=2	// L23885
  hls::stream< ap_uint<128> > v15559;
  #pragma HLS stream variable=v15559 depth=2	// L23886
  hls::stream< ap_uint<128> > v15560;
  #pragma HLS stream variable=v15560 depth=2	// L23887
  hls::stream< ap_uint<128> > v15561;
  #pragma HLS stream variable=v15561 depth=2	// L23888
  hls::stream< ap_uint<128> > v15562;
  #pragma HLS stream variable=v15562 depth=2	// L23889
  hls::stream< ap_uint<128> > v15563;
  #pragma HLS stream variable=v15563 depth=2	// L23890
  hls::stream< ap_uint<128> > v15564;
  #pragma HLS stream variable=v15564 depth=2	// L23891
  hls::stream< ap_uint<128> > v15565;
  #pragma HLS stream variable=v15565 depth=2	// L23892
  hls::stream< ap_uint<128> > v15566;
  #pragma HLS stream variable=v15566 depth=2	// L23893
  hls::stream< ap_uint<128> > v15567;
  #pragma HLS stream variable=v15567 depth=2	// L23894
  hls::stream< ap_uint<128> > v15568;
  #pragma HLS stream variable=v15568 depth=2	// L23895
  hls::stream< ap_uint<128> > v15569;
  #pragma HLS stream variable=v15569 depth=2	// L23896
  hls::stream< ap_uint<128> > v15570;
  #pragma HLS stream variable=v15570 depth=2	// L23897
  hls::stream< ap_uint<128> > v15571;
  #pragma HLS stream variable=v15571 depth=2	// L23898
  hls::stream< ap_uint<128> > v15572;
  #pragma HLS stream variable=v15572 depth=2	// L23899
  hls::stream< ap_uint<128> > v15573;
  #pragma HLS stream variable=v15573 depth=2	// L23900
  hls::stream< ap_uint<128> > v15574;
  #pragma HLS stream variable=v15574 depth=2	// L23901
  hls::stream< ap_uint<128> > v15575;
  #pragma HLS stream variable=v15575 depth=2	// L23902
  hls::stream< ap_uint<128> > v15576;
  #pragma HLS stream variable=v15576 depth=2	// L23903
  hls::stream< ap_uint<128> > v15577;
  #pragma HLS stream variable=v15577 depth=2	// L23904
  hls::stream< ap_uint<128> > v15578;
  #pragma HLS stream variable=v15578 depth=2	// L23905
  hls::stream< ap_uint<128> > v15579;
  #pragma HLS stream variable=v15579 depth=2	// L23906
  hls::stream< ap_uint<128> > v15580;
  #pragma HLS stream variable=v15580 depth=2	// L23907
  hls::stream< ap_uint<128> > v15581;
  #pragma HLS stream variable=v15581 depth=2	// L23908
  hls::stream< ap_uint<128> > v15582;
  #pragma HLS stream variable=v15582 depth=2	// L23909
  hls::stream< ap_uint<128> > v15583;
  #pragma HLS stream variable=v15583 depth=2	// L23910
  hls::stream< ap_uint<128> > v15584;
  #pragma HLS stream variable=v15584 depth=2	// L23911
  hls::stream< ap_uint<128> > v15585;
  #pragma HLS stream variable=v15585 depth=2	// L23912
  hls::stream< ap_uint<128> > v15586;
  #pragma HLS stream variable=v15586 depth=2	// L23913
  hls::stream< ap_uint<128> > v15587;
  #pragma HLS stream variable=v15587 depth=2	// L23914
  hls::stream< ap_uint<128> > v15588;
  #pragma HLS stream variable=v15588 depth=2	// L23915
  hls::stream< ap_uint<128> > v15589;
  #pragma HLS stream variable=v15589 depth=2	// L23916
  hls::stream< ap_uint<128> > v15590;
  #pragma HLS stream variable=v15590 depth=2	// L23917
  hls::stream< ap_uint<128> > v15591;
  #pragma HLS stream variable=v15591 depth=2	// L23918
  hls::stream< ap_uint<128> > v15592;
  #pragma HLS stream variable=v15592 depth=2	// L23919
  hls::stream< ap_uint<128> > v15593;
  #pragma HLS stream variable=v15593 depth=2	// L23920
  hls::stream< ap_uint<128> > v15594;
  #pragma HLS stream variable=v15594 depth=2	// L23921
  hls::stream< ap_uint<128> > v15595;
  #pragma HLS stream variable=v15595 depth=2	// L23922
  hls::stream< ap_uint<128> > v15596;
  #pragma HLS stream variable=v15596 depth=2	// L23923
  hls::stream< ap_uint<128> > v15597;
  #pragma HLS stream variable=v15597 depth=2	// L23924
  hls::stream< ap_uint<128> > v15598;
  #pragma HLS stream variable=v15598 depth=2	// L23925
  hls::stream< ap_uint<128> > v15599;
  #pragma HLS stream variable=v15599 depth=2	// L23926
  hls::stream< ap_uint<128> > v15600;
  #pragma HLS stream variable=v15600 depth=2	// L23927
  hls::stream< ap_uint<128> > v15601;
  #pragma HLS stream variable=v15601 depth=2	// L23928
  hls::stream< int8_t > v15602;
  #pragma HLS stream variable=v15602 depth=2	// L23929
  hls::stream< int8_t > v15603;
  #pragma HLS stream variable=v15603 depth=2	// L23930
  hls::stream< int8_t > v15604;
  #pragma HLS stream variable=v15604 depth=2	// L23931
  hls::stream< int8_t > v15605;
  #pragma HLS stream variable=v15605 depth=2	// L23932
  hls::stream< int8_t > v15606;
  #pragma HLS stream variable=v15606 depth=2	// L23933
  hls::stream< int8_t > v15607;
  #pragma HLS stream variable=v15607 depth=2	// L23934
  hls::stream< int8_t > v15608;
  #pragma HLS stream variable=v15608 depth=2	// L23935
  hls::stream< int8_t > v15609;
  #pragma HLS stream variable=v15609 depth=2	// L23936
  hls::stream< int8_t > v15610;
  #pragma HLS stream variable=v15610 depth=2	// L23937
  hls::stream< int8_t > v15611;
  #pragma HLS stream variable=v15611 depth=2	// L23938
  hls::stream< int8_t > v15612;
  #pragma HLS stream variable=v15612 depth=2	// L23939
  hls::stream< int8_t > v15613;
  #pragma HLS stream variable=v15613 depth=2	// L23940
  hls::stream< int8_t > v15614;
  #pragma HLS stream variable=v15614 depth=2	// L23941
  hls::stream< int8_t > v15615;
  #pragma HLS stream variable=v15615 depth=2	// L23942
  hls::stream< int8_t > v15616;
  #pragma HLS stream variable=v15616 depth=2	// L23943
  hls::stream< int8_t > v15617;
  #pragma HLS stream variable=v15617 depth=2	// L23944
  hls::stream< int8_t > v15618;
  #pragma HLS stream variable=v15618 depth=2	// L23945
  hls::stream< int8_t > v15619;
  #pragma HLS stream variable=v15619 depth=2	// L23946
  hls::stream< int8_t > v15620;
  #pragma HLS stream variable=v15620 depth=2	// L23947
  hls::stream< int8_t > v15621;
  #pragma HLS stream variable=v15621 depth=2	// L23948
  hls::stream< int8_t > v15622;
  #pragma HLS stream variable=v15622 depth=2	// L23949
  hls::stream< int8_t > v15623;
  #pragma HLS stream variable=v15623 depth=2	// L23950
  hls::stream< int8_t > v15624;
  #pragma HLS stream variable=v15624 depth=2	// L23951
  hls::stream< int8_t > v15625;
  #pragma HLS stream variable=v15625 depth=2	// L23952
  hls::stream< int8_t > v15626;
  #pragma HLS stream variable=v15626 depth=2	// L23953
  hls::stream< int8_t > v15627;
  #pragma HLS stream variable=v15627 depth=2	// L23954
  hls::stream< int8_t > v15628;
  #pragma HLS stream variable=v15628 depth=2	// L23955
  hls::stream< int8_t > v15629;
  #pragma HLS stream variable=v15629 depth=2	// L23956
  hls::stream< int8_t > v15630;
  #pragma HLS stream variable=v15630 depth=2	// L23957
  hls::stream< int8_t > v15631;
  #pragma HLS stream variable=v15631 depth=2	// L23958
  hls::stream< int8_t > v15632;
  #pragma HLS stream variable=v15632 depth=2	// L23959
  hls::stream< int8_t > v15633;
  #pragma HLS stream variable=v15633 depth=2	// L23960
  hls::stream< int8_t > v15634;
  #pragma HLS stream variable=v15634 depth=2	// L23961
  hls::stream< int8_t > v15635;
  #pragma HLS stream variable=v15635 depth=2	// L23962
  hls::stream< int8_t > v15636;
  #pragma HLS stream variable=v15636 depth=2	// L23963
  hls::stream< int8_t > v15637;
  #pragma HLS stream variable=v15637 depth=2	// L23964
  hls::stream< int8_t > v15638;
  #pragma HLS stream variable=v15638 depth=2	// L23965
  hls::stream< int8_t > v15639;
  #pragma HLS stream variable=v15639 depth=2	// L23966
  hls::stream< int8_t > v15640;
  #pragma HLS stream variable=v15640 depth=2	// L23967
  hls::stream< int8_t > v15641;
  #pragma HLS stream variable=v15641 depth=2	// L23968
  hls::stream< int8_t > v15642;
  #pragma HLS stream variable=v15642 depth=2	// L23969
  hls::stream< int8_t > v15643;
  #pragma HLS stream variable=v15643 depth=2	// L23970
  hls::stream< int8_t > v15644;
  #pragma HLS stream variable=v15644 depth=2	// L23971
  hls::stream< int8_t > v15645;
  #pragma HLS stream variable=v15645 depth=2	// L23972
  hls::stream< int8_t > v15646;
  #pragma HLS stream variable=v15646 depth=2	// L23973
  hls::stream< int8_t > v15647;
  #pragma HLS stream variable=v15647 depth=2	// L23974
  hls::stream< int8_t > v15648;
  #pragma HLS stream variable=v15648 depth=2	// L23975
  hls::stream< int8_t > v15649;
  #pragma HLS stream variable=v15649 depth=2	// L23976
  hls::stream< int8_t > v15650;
  #pragma HLS stream variable=v15650 depth=2	// L23977
  hls::stream< int8_t > v15651;
  #pragma HLS stream variable=v15651 depth=2	// L23978
  hls::stream< int8_t > v15652;
  #pragma HLS stream variable=v15652 depth=2	// L23979
  hls::stream< int8_t > v15653;
  #pragma HLS stream variable=v15653 depth=2	// L23980
  hls::stream< int8_t > v15654;
  #pragma HLS stream variable=v15654 depth=2	// L23981
  hls::stream< int8_t > v15655;
  #pragma HLS stream variable=v15655 depth=2	// L23982
  hls::stream< int8_t > v15656;
  #pragma HLS stream variable=v15656 depth=2	// L23983
  hls::stream< int8_t > v15657;
  #pragma HLS stream variable=v15657 depth=2	// L23984
  hls::stream< int8_t > v15658;
  #pragma HLS stream variable=v15658 depth=2	// L23985
  hls::stream< int8_t > v15659;
  #pragma HLS stream variable=v15659 depth=2	// L23986
  hls::stream< int8_t > v15660;
  #pragma HLS stream variable=v15660 depth=2	// L23987
  hls::stream< int8_t > v15661;
  #pragma HLS stream variable=v15661 depth=2	// L23988
  hls::stream< int8_t > v15662;
  #pragma HLS stream variable=v15662 depth=2	// L23989
  hls::stream< int8_t > v15663;
  #pragma HLS stream variable=v15663 depth=2	// L23990
  hls::stream< int8_t > v15664;
  #pragma HLS stream variable=v15664 depth=2	// L23991
  hls::stream< int8_t > v15665;
  #pragma HLS stream variable=v15665 depth=2	// L23992
  hls::stream< int8_t > v15666;
  #pragma HLS stream variable=v15666 depth=2	// L23993
  hls::stream< int8_t > v15667;
  #pragma HLS stream variable=v15667 depth=2	// L23994
  hls::stream< int8_t > v15668;
  #pragma HLS stream variable=v15668 depth=2	// L23995
  hls::stream< int8_t > v15669;
  #pragma HLS stream variable=v15669 depth=2	// L23996
  hls::stream< int8_t > v15670;
  #pragma HLS stream variable=v15670 depth=2	// L23997
  hls::stream< int8_t > v15671;
  #pragma HLS stream variable=v15671 depth=2	// L23998
  hls::stream< int8_t > v15672;
  #pragma HLS stream variable=v15672 depth=2	// L23999
  hls::stream< int8_t > v15673;
  #pragma HLS stream variable=v15673 depth=2	// L24000
  hls::stream< int8_t > v15674;
  #pragma HLS stream variable=v15674 depth=2	// L24001
  hls::stream< int8_t > v15675;
  #pragma HLS stream variable=v15675 depth=2	// L24002
  hls::stream< int8_t > v15676;
  #pragma HLS stream variable=v15676 depth=2	// L24003
  hls::stream< int8_t > v15677;
  #pragma HLS stream variable=v15677 depth=2	// L24004
  hls::stream< int8_t > v15678;
  #pragma HLS stream variable=v15678 depth=2	// L24005
  hls::stream< int8_t > v15679;
  #pragma HLS stream variable=v15679 depth=2	// L24006
  hls::stream< int8_t > v15680;
  #pragma HLS stream variable=v15680 depth=2	// L24007
  hls::stream< int8_t > v15681;
  #pragma HLS stream variable=v15681 depth=2	// L24008
  hls::stream< int8_t > v15682;
  #pragma HLS stream variable=v15682 depth=2	// L24009
  hls::stream< int8_t > v15683;
  #pragma HLS stream variable=v15683 depth=2	// L24010
  hls::stream< int8_t > v15684;
  #pragma HLS stream variable=v15684 depth=2	// L24011
  hls::stream< int8_t > v15685;
  #pragma HLS stream variable=v15685 depth=2	// L24012
  hls::stream< int8_t > v15686;
  #pragma HLS stream variable=v15686 depth=2	// L24013
  hls::stream< int8_t > v15687;
  #pragma HLS stream variable=v15687 depth=2	// L24014
  hls::stream< int8_t > v15688;
  #pragma HLS stream variable=v15688 depth=2	// L24015
  hls::stream< int8_t > v15689;
  #pragma HLS stream variable=v15689 depth=2	// L24016
  hls::stream< int8_t > v15690;
  #pragma HLS stream variable=v15690 depth=2	// L24017
  hls::stream< int8_t > v15691;
  #pragma HLS stream variable=v15691 depth=2	// L24018
  hls::stream< int8_t > v15692;
  #pragma HLS stream variable=v15692 depth=2	// L24019
  hls::stream< int8_t > v15693;
  #pragma HLS stream variable=v15693 depth=2	// L24020
  hls::stream< int8_t > v15694;
  #pragma HLS stream variable=v15694 depth=2	// L24021
  hls::stream< int8_t > v15695;
  #pragma HLS stream variable=v15695 depth=2	// L24022
  hls::stream< int8_t > v15696;
  #pragma HLS stream variable=v15696 depth=2	// L24023
  hls::stream< int8_t > v15697;
  #pragma HLS stream variable=v15697 depth=2	// L24024
  hls::stream< int8_t > v15698;
  #pragma HLS stream variable=v15698 depth=2	// L24025
  hls::stream< int8_t > v15699;
  #pragma HLS stream variable=v15699 depth=2	// L24026
  hls::stream< int8_t > v15700;
  #pragma HLS stream variable=v15700 depth=2	// L24027
  hls::stream< int8_t > v15701;
  #pragma HLS stream variable=v15701 depth=2	// L24028
  hls::stream< int8_t > v15702;
  #pragma HLS stream variable=v15702 depth=2	// L24029
  hls::stream< int8_t > v15703;
  #pragma HLS stream variable=v15703 depth=2	// L24030
  hls::stream< int8_t > v15704;
  #pragma HLS stream variable=v15704 depth=2	// L24031
  hls::stream< int8_t > v15705;
  #pragma HLS stream variable=v15705 depth=2	// L24032
  hls::stream< int8_t > v15706;
  #pragma HLS stream variable=v15706 depth=2	// L24033
  hls::stream< int8_t > v15707;
  #pragma HLS stream variable=v15707 depth=2	// L24034
  hls::stream< int8_t > v15708;
  #pragma HLS stream variable=v15708 depth=2	// L24035
  hls::stream< int8_t > v15709;
  #pragma HLS stream variable=v15709 depth=2	// L24036
  hls::stream< int8_t > v15710;
  #pragma HLS stream variable=v15710 depth=2	// L24037
  hls::stream< int8_t > v15711;
  #pragma HLS stream variable=v15711 depth=2	// L24038
  hls::stream< int8_t > v15712;
  #pragma HLS stream variable=v15712 depth=2	// L24039
  hls::stream< int8_t > v15713;
  #pragma HLS stream variable=v15713 depth=2	// L24040
  hls::stream< int8_t > v15714;
  #pragma HLS stream variable=v15714 depth=2	// L24041
  hls::stream< int8_t > v15715;
  #pragma HLS stream variable=v15715 depth=2	// L24042
  hls::stream< int8_t > v15716;
  #pragma HLS stream variable=v15716 depth=2	// L24043
  hls::stream< int8_t > v15717;
  #pragma HLS stream variable=v15717 depth=2	// L24044
  hls::stream< int8_t > v15718;
  #pragma HLS stream variable=v15718 depth=2	// L24045
  hls::stream< int8_t > v15719;
  #pragma HLS stream variable=v15719 depth=2	// L24046
  hls::stream< int8_t > v15720;
  #pragma HLS stream variable=v15720 depth=2	// L24047
  hls::stream< int8_t > v15721;
  #pragma HLS stream variable=v15721 depth=2	// L24048
  hls::stream< int8_t > v15722;
  #pragma HLS stream variable=v15722 depth=2	// L24049
  hls::stream< int8_t > v15723;
  #pragma HLS stream variable=v15723 depth=2	// L24050
  hls::stream< int8_t > v15724;
  #pragma HLS stream variable=v15724 depth=2	// L24051
  hls::stream< int8_t > v15725;
  #pragma HLS stream variable=v15725 depth=2	// L24052
  hls::stream< int8_t > v15726;
  #pragma HLS stream variable=v15726 depth=2	// L24053
  hls::stream< int8_t > v15727;
  #pragma HLS stream variable=v15727 depth=2	// L24054
  hls::stream< int8_t > v15728;
  #pragma HLS stream variable=v15728 depth=2	// L24055
  hls::stream< int8_t > v15729;
  #pragma HLS stream variable=v15729 depth=2	// L24056
  hls::stream< int8_t > v15730;
  #pragma HLS stream variable=v15730 depth=2	// L24057
  hls::stream< int8_t > v15731;
  #pragma HLS stream variable=v15731 depth=2	// L24058
  hls::stream< int8_t > v15732;
  #pragma HLS stream variable=v15732 depth=2	// L24059
  hls::stream< int8_t > v15733;
  #pragma HLS stream variable=v15733 depth=2	// L24060
  hls::stream< int8_t > v15734;
  #pragma HLS stream variable=v15734 depth=2	// L24061
  hls::stream< int8_t > v15735;
  #pragma HLS stream variable=v15735 depth=2	// L24062
  hls::stream< int8_t > v15736;
  #pragma HLS stream variable=v15736 depth=2	// L24063
  hls::stream< int8_t > v15737;
  #pragma HLS stream variable=v15737 depth=2	// L24064
  hls::stream< int8_t > v15738;
  #pragma HLS stream variable=v15738 depth=2	// L24065
  hls::stream< int8_t > v15739;
  #pragma HLS stream variable=v15739 depth=2	// L24066
  hls::stream< int8_t > v15740;
  #pragma HLS stream variable=v15740 depth=2	// L24067
  hls::stream< int8_t > v15741;
  #pragma HLS stream variable=v15741 depth=2	// L24068
  hls::stream< int8_t > v15742;
  #pragma HLS stream variable=v15742 depth=2	// L24069
  hls::stream< int8_t > v15743;
  #pragma HLS stream variable=v15743 depth=2	// L24070
  hls::stream< int8_t > v15744;
  #pragma HLS stream variable=v15744 depth=2	// L24071
  hls::stream< int8_t > v15745;
  #pragma HLS stream variable=v15745 depth=2	// L24072
  hls::stream< int8_t > v15746;
  #pragma HLS stream variable=v15746 depth=2	// L24073
  hls::stream< int8_t > v15747;
  #pragma HLS stream variable=v15747 depth=2	// L24074
  hls::stream< int8_t > v15748;
  #pragma HLS stream variable=v15748 depth=2	// L24075
  hls::stream< int8_t > v15749;
  #pragma HLS stream variable=v15749 depth=2	// L24076
  hls::stream< int8_t > v15750;
  #pragma HLS stream variable=v15750 depth=2	// L24077
  hls::stream< int8_t > v15751;
  #pragma HLS stream variable=v15751 depth=2	// L24078
  hls::stream< int8_t > v15752;
  #pragma HLS stream variable=v15752 depth=2	// L24079
  hls::stream< int8_t > v15753;
  #pragma HLS stream variable=v15753 depth=2	// L24080
  hls::stream< int8_t > v15754;
  #pragma HLS stream variable=v15754 depth=2	// L24081
  hls::stream< int8_t > v15755;
  #pragma HLS stream variable=v15755 depth=2	// L24082
  hls::stream< int8_t > v15756;
  #pragma HLS stream variable=v15756 depth=2	// L24083
  hls::stream< int8_t > v15757;
  #pragma HLS stream variable=v15757 depth=2	// L24084
  hls::stream< int8_t > v15758;
  #pragma HLS stream variable=v15758 depth=2	// L24085
  hls::stream< int8_t > v15759;
  #pragma HLS stream variable=v15759 depth=2	// L24086
  hls::stream< int8_t > v15760;
  #pragma HLS stream variable=v15760 depth=2	// L24087
  hls::stream< int8_t > v15761;
  #pragma HLS stream variable=v15761 depth=2	// L24088
  hls::stream< int8_t > v15762;
  #pragma HLS stream variable=v15762 depth=2	// L24089
  hls::stream< int8_t > v15763;
  #pragma HLS stream variable=v15763 depth=2	// L24090
  hls::stream< int8_t > v15764;
  #pragma HLS stream variable=v15764 depth=2	// L24091
  hls::stream< int8_t > v15765;
  #pragma HLS stream variable=v15765 depth=2	// L24092
  hls::stream< int8_t > v15766;
  #pragma HLS stream variable=v15766 depth=2	// L24093
  hls::stream< int8_t > v15767;
  #pragma HLS stream variable=v15767 depth=2	// L24094
  hls::stream< int8_t > v15768;
  #pragma HLS stream variable=v15768 depth=2	// L24095
  hls::stream< int8_t > v15769;
  #pragma HLS stream variable=v15769 depth=2	// L24096
  hls::stream< int8_t > v15770;
  #pragma HLS stream variable=v15770 depth=2	// L24097
  hls::stream< int8_t > v15771;
  #pragma HLS stream variable=v15771 depth=2	// L24098
  hls::stream< int8_t > v15772;
  #pragma HLS stream variable=v15772 depth=2	// L24099
  hls::stream< int8_t > v15773;
  #pragma HLS stream variable=v15773 depth=2	// L24100
  hls::stream< int8_t > v15774;
  #pragma HLS stream variable=v15774 depth=2	// L24101
  hls::stream< int8_t > v15775;
  #pragma HLS stream variable=v15775 depth=2	// L24102
  hls::stream< int8_t > v15776;
  #pragma HLS stream variable=v15776 depth=2	// L24103
  hls::stream< int8_t > v15777;
  #pragma HLS stream variable=v15777 depth=2	// L24104
  hls::stream< int8_t > v15778;
  #pragma HLS stream variable=v15778 depth=2	// L24105
  hls::stream< int8_t > v15779;
  #pragma HLS stream variable=v15779 depth=2	// L24106
  hls::stream< int8_t > v15780;
  #pragma HLS stream variable=v15780 depth=2	// L24107
  hls::stream< int8_t > v15781;
  #pragma HLS stream variable=v15781 depth=2	// L24108
  hls::stream< int8_t > v15782;
  #pragma HLS stream variable=v15782 depth=2	// L24109
  hls::stream< int8_t > v15783;
  #pragma HLS stream variable=v15783 depth=2	// L24110
  hls::stream< int8_t > v15784;
  #pragma HLS stream variable=v15784 depth=2	// L24111
  hls::stream< int8_t > v15785;
  #pragma HLS stream variable=v15785 depth=2	// L24112
  hls::stream< int8_t > v15786;
  #pragma HLS stream variable=v15786 depth=2	// L24113
  hls::stream< int8_t > v15787;
  #pragma HLS stream variable=v15787 depth=2	// L24114
  hls::stream< int8_t > v15788;
  #pragma HLS stream variable=v15788 depth=2	// L24115
  hls::stream< int8_t > v15789;
  #pragma HLS stream variable=v15789 depth=2	// L24116
  hls::stream< int8_t > v15790;
  #pragma HLS stream variable=v15790 depth=2	// L24117
  hls::stream< int8_t > v15791;
  #pragma HLS stream variable=v15791 depth=2	// L24118
  hls::stream< int8_t > v15792;
  #pragma HLS stream variable=v15792 depth=2	// L24119
  hls::stream< int8_t > v15793;
  #pragma HLS stream variable=v15793 depth=2	// L24120
  hls::stream< int8_t > v15794;
  #pragma HLS stream variable=v15794 depth=2	// L24121
  hls::stream< int8_t > v15795;
  #pragma HLS stream variable=v15795 depth=2	// L24122
  hls::stream< int8_t > v15796;
  #pragma HLS stream variable=v15796 depth=2	// L24123
  hls::stream< int8_t > v15797;
  #pragma HLS stream variable=v15797 depth=2	// L24124
  hls::stream< int8_t > v15798;
  #pragma HLS stream variable=v15798 depth=2	// L24125
  hls::stream< int8_t > v15799;
  #pragma HLS stream variable=v15799 depth=2	// L24126
  hls::stream< int8_t > v15800;
  #pragma HLS stream variable=v15800 depth=2	// L24127
  hls::stream< int8_t > v15801;
  #pragma HLS stream variable=v15801 depth=2	// L24128
  hls::stream< int8_t > v15802;
  #pragma HLS stream variable=v15802 depth=2	// L24129
  hls::stream< int8_t > v15803;
  #pragma HLS stream variable=v15803 depth=2	// L24130
  hls::stream< int8_t > v15804;
  #pragma HLS stream variable=v15804 depth=2	// L24131
  hls::stream< int8_t > v15805;
  #pragma HLS stream variable=v15805 depth=2	// L24132
  hls::stream< int8_t > v15806;
  #pragma HLS stream variable=v15806 depth=2	// L24133
  hls::stream< int8_t > v15807;
  #pragma HLS stream variable=v15807 depth=2	// L24134
  hls::stream< int8_t > v15808;
  #pragma HLS stream variable=v15808 depth=2	// L24135
  hls::stream< int8_t > v15809;
  #pragma HLS stream variable=v15809 depth=2	// L24136
  hls::stream< int8_t > v15810;
  #pragma HLS stream variable=v15810 depth=2	// L24137
  hls::stream< int8_t > v15811;
  #pragma HLS stream variable=v15811 depth=2	// L24138
  hls::stream< int8_t > v15812;
  #pragma HLS stream variable=v15812 depth=2	// L24139
  hls::stream< int8_t > v15813;
  #pragma HLS stream variable=v15813 depth=2	// L24140
  hls::stream< int8_t > v15814;
  #pragma HLS stream variable=v15814 depth=2	// L24141
  hls::stream< int8_t > v15815;
  #pragma HLS stream variable=v15815 depth=2	// L24142
  hls::stream< int8_t > v15816;
  #pragma HLS stream variable=v15816 depth=2	// L24143
  hls::stream< int8_t > v15817;
  #pragma HLS stream variable=v15817 depth=2	// L24144
  hls::stream< int8_t > v15818;
  #pragma HLS stream variable=v15818 depth=2	// L24145
  hls::stream< int8_t > v15819;
  #pragma HLS stream variable=v15819 depth=2	// L24146
  hls::stream< int8_t > v15820;
  #pragma HLS stream variable=v15820 depth=2	// L24147
  hls::stream< int8_t > v15821;
  #pragma HLS stream variable=v15821 depth=2	// L24148
  hls::stream< int8_t > v15822;
  #pragma HLS stream variable=v15822 depth=2	// L24149
  hls::stream< int8_t > v15823;
  #pragma HLS stream variable=v15823 depth=2	// L24150
  hls::stream< int8_t > v15824;
  #pragma HLS stream variable=v15824 depth=2	// L24151
  hls::stream< int8_t > v15825;
  #pragma HLS stream variable=v15825 depth=2	// L24152
  hls::stream< int8_t > v15826;
  #pragma HLS stream variable=v15826 depth=2	// L24153
  hls::stream< int8_t > v15827;
  #pragma HLS stream variable=v15827 depth=2	// L24154
  hls::stream< int8_t > v15828;
  #pragma HLS stream variable=v15828 depth=2	// L24155
  hls::stream< int8_t > v15829;
  #pragma HLS stream variable=v15829 depth=2	// L24156
  hls::stream< int8_t > v15830;
  #pragma HLS stream variable=v15830 depth=2	// L24157
  hls::stream< int8_t > v15831;
  #pragma HLS stream variable=v15831 depth=2	// L24158
  hls::stream< int8_t > v15832;
  #pragma HLS stream variable=v15832 depth=2	// L24159
  hls::stream< int8_t > v15833;
  #pragma HLS stream variable=v15833 depth=2	// L24160
  hls::stream< int8_t > v15834;
  #pragma HLS stream variable=v15834 depth=2	// L24161
  hls::stream< int8_t > v15835;
  #pragma HLS stream variable=v15835 depth=2	// L24162
  hls::stream< int8_t > v15836;
  #pragma HLS stream variable=v15836 depth=2	// L24163
  hls::stream< int8_t > v15837;
  #pragma HLS stream variable=v15837 depth=2	// L24164
  hls::stream< int8_t > v15838;
  #pragma HLS stream variable=v15838 depth=2	// L24165
  hls::stream< int8_t > v15839;
  #pragma HLS stream variable=v15839 depth=2	// L24166
  hls::stream< int8_t > v15840;
  #pragma HLS stream variable=v15840 depth=2	// L24167
  hls::stream< int8_t > v15841;
  #pragma HLS stream variable=v15841 depth=2	// L24168
  hls::stream< int8_t > v15842;
  #pragma HLS stream variable=v15842 depth=2	// L24169
  hls::stream< int8_t > v15843;
  #pragma HLS stream variable=v15843 depth=2	// L24170
  hls::stream< int8_t > v15844;
  #pragma HLS stream variable=v15844 depth=2	// L24171
  hls::stream< int8_t > v15845;
  #pragma HLS stream variable=v15845 depth=2	// L24172
  hls::stream< int8_t > v15846;
  #pragma HLS stream variable=v15846 depth=2	// L24173
  hls::stream< int8_t > v15847;
  #pragma HLS stream variable=v15847 depth=2	// L24174
  hls::stream< int8_t > v15848;
  #pragma HLS stream variable=v15848 depth=2	// L24175
  hls::stream< int8_t > v15849;
  #pragma HLS stream variable=v15849 depth=2	// L24176
  hls::stream< int8_t > v15850;
  #pragma HLS stream variable=v15850 depth=2	// L24177
  hls::stream< int8_t > v15851;
  #pragma HLS stream variable=v15851 depth=2	// L24178
  hls::stream< int8_t > v15852;
  #pragma HLS stream variable=v15852 depth=2	// L24179
  hls::stream< int8_t > v15853;
  #pragma HLS stream variable=v15853 depth=2	// L24180
  hls::stream< int8_t > v15854;
  #pragma HLS stream variable=v15854 depth=2	// L24181
  hls::stream< int8_t > v15855;
  #pragma HLS stream variable=v15855 depth=2	// L24182
  hls::stream< int8_t > v15856;
  #pragma HLS stream variable=v15856 depth=2	// L24183
  hls::stream< int8_t > v15857;
  #pragma HLS stream variable=v15857 depth=2	// L24184
  hls::stream< int8_t > v15858;
  #pragma HLS stream variable=v15858 depth=2	// L24185
  hls::stream< int8_t > v15859;
  #pragma HLS stream variable=v15859 depth=2	// L24186
  hls::stream< int8_t > v15860;
  #pragma HLS stream variable=v15860 depth=2	// L24187
  hls::stream< int8_t > v15861;
  #pragma HLS stream variable=v15861 depth=2	// L24188
  hls::stream< int8_t > v15862;
  #pragma HLS stream variable=v15862 depth=2	// L24189
  hls::stream< int8_t > v15863;
  #pragma HLS stream variable=v15863 depth=2	// L24190
  hls::stream< int8_t > v15864;
  #pragma HLS stream variable=v15864 depth=2	// L24191
  hls::stream< int8_t > v15865;
  #pragma HLS stream variable=v15865 depth=2	// L24192
  hls::stream< int8_t > v15866;
  #pragma HLS stream variable=v15866 depth=2	// L24193
  hls::stream< int8_t > v15867;
  #pragma HLS stream variable=v15867 depth=2	// L24194
  hls::stream< int8_t > v15868;
  #pragma HLS stream variable=v15868 depth=2	// L24195
  hls::stream< int8_t > v15869;
  #pragma HLS stream variable=v15869 depth=2	// L24196
  hls::stream< int8_t > v15870;
  #pragma HLS stream variable=v15870 depth=2	// L24197
  hls::stream< int8_t > v15871;
  #pragma HLS stream variable=v15871 depth=2	// L24198
  hls::stream< int8_t > v15872;
  #pragma HLS stream variable=v15872 depth=2	// L24199
  hls::stream< int8_t > v15873;
  #pragma HLS stream variable=v15873 depth=2	// L24200
  hls::stream< int8_t > v15874;
  #pragma HLS stream variable=v15874 depth=2	// L24201
  hls::stream< int8_t > v15875;
  #pragma HLS stream variable=v15875 depth=2	// L24202
  hls::stream< int8_t > v15876;
  #pragma HLS stream variable=v15876 depth=2	// L24203
  hls::stream< int8_t > v15877;
  #pragma HLS stream variable=v15877 depth=2	// L24204
  hls::stream< int8_t > v15878;
  #pragma HLS stream variable=v15878 depth=2	// L24205
  hls::stream< int8_t > v15879;
  #pragma HLS stream variable=v15879 depth=2	// L24206
  hls::stream< int8_t > v15880;
  #pragma HLS stream variable=v15880 depth=2	// L24207
  hls::stream< int8_t > v15881;
  #pragma HLS stream variable=v15881 depth=2	// L24208
  hls::stream< int8_t > v15882;
  #pragma HLS stream variable=v15882 depth=2	// L24209
  hls::stream< int8_t > v15883;
  #pragma HLS stream variable=v15883 depth=2	// L24210
  hls::stream< int8_t > v15884;
  #pragma HLS stream variable=v15884 depth=2	// L24211
  hls::stream< int8_t > v15885;
  #pragma HLS stream variable=v15885 depth=2	// L24212
  hls::stream< int8_t > v15886;
  #pragma HLS stream variable=v15886 depth=2	// L24213
  hls::stream< int8_t > v15887;
  #pragma HLS stream variable=v15887 depth=2	// L24214
  hls::stream< int8_t > v15888;
  #pragma HLS stream variable=v15888 depth=2	// L24215
  hls::stream< int8_t > v15889;
  #pragma HLS stream variable=v15889 depth=2	// L24216
  hls::stream< int8_t > v15890;
  #pragma HLS stream variable=v15890 depth=2	// L24217
  hls::stream< int8_t > v15891;
  #pragma HLS stream variable=v15891 depth=2	// L24218
  hls::stream< int8_t > v15892;
  #pragma HLS stream variable=v15892 depth=2	// L24219
  hls::stream< int8_t > v15893;
  #pragma HLS stream variable=v15893 depth=2	// L24220
  hls::stream< int8_t > v15894;
  #pragma HLS stream variable=v15894 depth=2	// L24221
  hls::stream< int8_t > v15895;
  #pragma HLS stream variable=v15895 depth=2	// L24222
  hls::stream< int8_t > v15896;
  #pragma HLS stream variable=v15896 depth=2	// L24223
  hls::stream< int8_t > v15897;
  #pragma HLS stream variable=v15897 depth=2	// L24224
  hls::stream< int8_t > v15898;
  #pragma HLS stream variable=v15898 depth=2	// L24225
  hls::stream< int8_t > v15899;
  #pragma HLS stream variable=v15899 depth=2	// L24226
  hls::stream< int8_t > v15900;
  #pragma HLS stream variable=v15900 depth=2	// L24227
  hls::stream< int8_t > v15901;
  #pragma HLS stream variable=v15901 depth=2	// L24228
  hls::stream< int8_t > v15902;
  #pragma HLS stream variable=v15902 depth=2	// L24229
  hls::stream< int8_t > v15903;
  #pragma HLS stream variable=v15903 depth=2	// L24230
  hls::stream< int8_t > v15904;
  #pragma HLS stream variable=v15904 depth=2	// L24231
  hls::stream< int8_t > v15905;
  #pragma HLS stream variable=v15905 depth=2	// L24232
  hls::stream< int8_t > v15906;
  #pragma HLS stream variable=v15906 depth=2	// L24233
  hls::stream< int8_t > v15907;
  #pragma HLS stream variable=v15907 depth=2	// L24234
  hls::stream< int8_t > v15908;
  #pragma HLS stream variable=v15908 depth=2	// L24235
  hls::stream< int8_t > v15909;
  #pragma HLS stream variable=v15909 depth=2	// L24236
  hls::stream< int8_t > v15910;
  #pragma HLS stream variable=v15910 depth=2	// L24237
  hls::stream< int8_t > v15911;
  #pragma HLS stream variable=v15911 depth=2	// L24238
  hls::stream< int8_t > v15912;
  #pragma HLS stream variable=v15912 depth=2	// L24239
  hls::stream< int8_t > v15913;
  #pragma HLS stream variable=v15913 depth=2	// L24240
  hls::stream< int8_t > v15914;
  #pragma HLS stream variable=v15914 depth=2	// L24241
  hls::stream< int8_t > v15915;
  #pragma HLS stream variable=v15915 depth=2	// L24242
  hls::stream< int8_t > v15916;
  #pragma HLS stream variable=v15916 depth=2	// L24243
  hls::stream< int8_t > v15917;
  #pragma HLS stream variable=v15917 depth=2	// L24244
  hls::stream< int8_t > v15918;
  #pragma HLS stream variable=v15918 depth=2	// L24245
  hls::stream< int8_t > v15919;
  #pragma HLS stream variable=v15919 depth=2	// L24246
  hls::stream< int8_t > v15920;
  #pragma HLS stream variable=v15920 depth=2	// L24247
  hls::stream< int8_t > v15921;
  #pragma HLS stream variable=v15921 depth=2	// L24248
  hls::stream< int8_t > v15922;
  #pragma HLS stream variable=v15922 depth=2	// L24249
  hls::stream< int8_t > v15923;
  #pragma HLS stream variable=v15923 depth=2	// L24250
  hls::stream< int8_t > v15924;
  #pragma HLS stream variable=v15924 depth=2	// L24251
  hls::stream< int8_t > v15925;
  #pragma HLS stream variable=v15925 depth=2	// L24252
  hls::stream< int8_t > v15926;
  #pragma HLS stream variable=v15926 depth=2	// L24253
  hls::stream< int8_t > v15927;
  #pragma HLS stream variable=v15927 depth=2	// L24254
  hls::stream< int8_t > v15928;
  #pragma HLS stream variable=v15928 depth=2	// L24255
  hls::stream< int8_t > v15929;
  #pragma HLS stream variable=v15929 depth=2	// L24256
  hls::stream< int8_t > v15930;
  #pragma HLS stream variable=v15930 depth=2	// L24257
  hls::stream< int8_t > v15931;
  #pragma HLS stream variable=v15931 depth=2	// L24258
  hls::stream< int8_t > v15932;
  #pragma HLS stream variable=v15932 depth=2	// L24259
  hls::stream< int8_t > v15933;
  #pragma HLS stream variable=v15933 depth=2	// L24260
  hls::stream< int8_t > v15934;
  #pragma HLS stream variable=v15934 depth=2	// L24261
  hls::stream< int8_t > v15935;
  #pragma HLS stream variable=v15935 depth=2	// L24262
  hls::stream< int8_t > v15936;
  #pragma HLS stream variable=v15936 depth=2	// L24263
  hls::stream< int8_t > v15937;
  #pragma HLS stream variable=v15937 depth=2	// L24264
  hls::stream< int8_t > v15938;
  #pragma HLS stream variable=v15938 depth=2	// L24265
  hls::stream< int8_t > v15939;
  #pragma HLS stream variable=v15939 depth=2	// L24266
  hls::stream< int8_t > v15940;
  #pragma HLS stream variable=v15940 depth=2	// L24267
  hls::stream< int8_t > v15941;
  #pragma HLS stream variable=v15941 depth=2	// L24268
  hls::stream< int8_t > v15942;
  #pragma HLS stream variable=v15942 depth=2	// L24269
  hls::stream< int8_t > v15943;
  #pragma HLS stream variable=v15943 depth=2	// L24270
  hls::stream< int8_t > v15944;
  #pragma HLS stream variable=v15944 depth=2	// L24271
  hls::stream< int8_t > v15945;
  #pragma HLS stream variable=v15945 depth=2	// L24272
  hls::stream< int8_t > v15946;
  #pragma HLS stream variable=v15946 depth=2	// L24273
  hls::stream< int8_t > v15947;
  #pragma HLS stream variable=v15947 depth=2	// L24274
  hls::stream< int8_t > v15948;
  #pragma HLS stream variable=v15948 depth=2	// L24275
  hls::stream< int8_t > v15949;
  #pragma HLS stream variable=v15949 depth=2	// L24276
  hls::stream< int8_t > v15950;
  #pragma HLS stream variable=v15950 depth=2	// L24277
  hls::stream< int8_t > v15951;
  #pragma HLS stream variable=v15951 depth=2	// L24278
  hls::stream< int8_t > v15952;
  #pragma HLS stream variable=v15952 depth=2	// L24279
  hls::stream< int8_t > v15953;
  #pragma HLS stream variable=v15953 depth=2	// L24280
  hls::stream< int8_t > v15954;
  #pragma HLS stream variable=v15954 depth=2	// L24281
  hls::stream< int8_t > v15955;
  #pragma HLS stream variable=v15955 depth=2	// L24282
  hls::stream< int8_t > v15956;
  #pragma HLS stream variable=v15956 depth=2	// L24283
  hls::stream< int8_t > v15957;
  #pragma HLS stream variable=v15957 depth=2	// L24284
  hls::stream< int8_t > v15958;
  #pragma HLS stream variable=v15958 depth=2	// L24285
  hls::stream< int8_t > v15959;
  #pragma HLS stream variable=v15959 depth=2	// L24286
  hls::stream< int8_t > v15960;
  #pragma HLS stream variable=v15960 depth=2	// L24287
  hls::stream< int8_t > v15961;
  #pragma HLS stream variable=v15961 depth=2	// L24288
  hls::stream< int8_t > v15962;
  #pragma HLS stream variable=v15962 depth=2	// L24289
  hls::stream< int8_t > v15963;
  #pragma HLS stream variable=v15963 depth=2	// L24290
  hls::stream< int8_t > v15964;
  #pragma HLS stream variable=v15964 depth=2	// L24291
  hls::stream< int8_t > v15965;
  #pragma HLS stream variable=v15965 depth=2	// L24292
  hls::stream< int8_t > v15966;
  #pragma HLS stream variable=v15966 depth=2	// L24293
  hls::stream< int8_t > v15967;
  #pragma HLS stream variable=v15967 depth=2	// L24294
  hls::stream< int8_t > v15968;
  #pragma HLS stream variable=v15968 depth=2	// L24295
  hls::stream< int8_t > v15969;
  #pragma HLS stream variable=v15969 depth=2	// L24296
  hls::stream< int8_t > v15970;
  #pragma HLS stream variable=v15970 depth=2	// L24297
  hls::stream< int8_t > v15971;
  #pragma HLS stream variable=v15971 depth=2	// L24298
  hls::stream< int8_t > v15972;
  #pragma HLS stream variable=v15972 depth=2	// L24299
  hls::stream< int8_t > v15973;
  #pragma HLS stream variable=v15973 depth=2	// L24300
  hls::stream< int8_t > v15974;
  #pragma HLS stream variable=v15974 depth=2	// L24301
  hls::stream< int8_t > v15975;
  #pragma HLS stream variable=v15975 depth=2	// L24302
  hls::stream< int8_t > v15976;
  #pragma HLS stream variable=v15976 depth=2	// L24303
  hls::stream< int8_t > v15977;
  #pragma HLS stream variable=v15977 depth=2	// L24304
  hls::stream< int8_t > v15978;
  #pragma HLS stream variable=v15978 depth=2	// L24305
  hls::stream< int8_t > v15979;
  #pragma HLS stream variable=v15979 depth=2	// L24306
  hls::stream< int8_t > v15980;
  #pragma HLS stream variable=v15980 depth=2	// L24307
  hls::stream< int8_t > v15981;
  #pragma HLS stream variable=v15981 depth=2	// L24308
  hls::stream< int8_t > v15982;
  #pragma HLS stream variable=v15982 depth=2	// L24309
  hls::stream< int8_t > v15983;
  #pragma HLS stream variable=v15983 depth=2	// L24310
  hls::stream< int8_t > v15984;
  #pragma HLS stream variable=v15984 depth=2	// L24311
  hls::stream< int8_t > v15985;
  #pragma HLS stream variable=v15985 depth=2	// L24312
  hls::stream< int8_t > v15986;
  #pragma HLS stream variable=v15986 depth=2	// L24313
  hls::stream< int8_t > v15987;
  #pragma HLS stream variable=v15987 depth=2	// L24314
  hls::stream< int8_t > v15988;
  #pragma HLS stream variable=v15988 depth=2	// L24315
  hls::stream< int8_t > v15989;
  #pragma HLS stream variable=v15989 depth=2	// L24316
  hls::stream< int8_t > v15990;
  #pragma HLS stream variable=v15990 depth=2	// L24317
  hls::stream< int8_t > v15991;
  #pragma HLS stream variable=v15991 depth=2	// L24318
  hls::stream< int8_t > v15992;
  #pragma HLS stream variable=v15992 depth=2	// L24319
  hls::stream< int8_t > v15993;
  #pragma HLS stream variable=v15993 depth=2	// L24320
  hls::stream< int8_t > v15994;
  #pragma HLS stream variable=v15994 depth=2	// L24321
  hls::stream< int8_t > v15995;
  #pragma HLS stream variable=v15995 depth=2	// L24322
  hls::stream< int8_t > v15996;
  #pragma HLS stream variable=v15996 depth=2	// L24323
  hls::stream< int8_t > v15997;
  #pragma HLS stream variable=v15997 depth=2	// L24324
  hls::stream< int8_t > v15998;
  #pragma HLS stream variable=v15998 depth=2	// L24325
  hls::stream< int8_t > v15999;
  #pragma HLS stream variable=v15999 depth=2	// L24326
  hls::stream< int8_t > v16000;
  #pragma HLS stream variable=v16000 depth=2	// L24327
  hls::stream< int8_t > v16001;
  #pragma HLS stream variable=v16001 depth=2	// L24328
  hls::stream< int8_t > v16002;
  #pragma HLS stream variable=v16002 depth=2	// L24329
  hls::stream< int8_t > v16003;
  #pragma HLS stream variable=v16003 depth=2	// L24330
  hls::stream< int8_t > v16004;
  #pragma HLS stream variable=v16004 depth=2	// L24331
  hls::stream< int8_t > v16005;
  #pragma HLS stream variable=v16005 depth=2	// L24332
  hls::stream< int8_t > v16006;
  #pragma HLS stream variable=v16006 depth=2	// L24333
  hls::stream< int8_t > v16007;
  #pragma HLS stream variable=v16007 depth=2	// L24334
  hls::stream< int8_t > v16008;
  #pragma HLS stream variable=v16008 depth=2	// L24335
  hls::stream< int8_t > v16009;
  #pragma HLS stream variable=v16009 depth=2	// L24336
  hls::stream< int8_t > v16010;
  #pragma HLS stream variable=v16010 depth=2	// L24337
  hls::stream< int8_t > v16011;
  #pragma HLS stream variable=v16011 depth=2	// L24338
  hls::stream< int8_t > v16012;
  #pragma HLS stream variable=v16012 depth=2	// L24339
  hls::stream< int8_t > v16013;
  #pragma HLS stream variable=v16013 depth=2	// L24340
  hls::stream< int8_t > v16014;
  #pragma HLS stream variable=v16014 depth=2	// L24341
  hls::stream< int8_t > v16015;
  #pragma HLS stream variable=v16015 depth=2	// L24342
  hls::stream< int8_t > v16016;
  #pragma HLS stream variable=v16016 depth=2	// L24343
  hls::stream< int8_t > v16017;
  #pragma HLS stream variable=v16017 depth=2	// L24344
  hls::stream< int8_t > v16018;
  #pragma HLS stream variable=v16018 depth=2	// L24345
  hls::stream< int8_t > v16019;
  #pragma HLS stream variable=v16019 depth=2	// L24346
  hls::stream< int8_t > v16020;
  #pragma HLS stream variable=v16020 depth=2	// L24347
  hls::stream< int8_t > v16021;
  #pragma HLS stream variable=v16021 depth=2	// L24348
  hls::stream< int8_t > v16022;
  #pragma HLS stream variable=v16022 depth=2	// L24349
  hls::stream< int8_t > v16023;
  #pragma HLS stream variable=v16023 depth=2	// L24350
  hls::stream< int8_t > v16024;
  #pragma HLS stream variable=v16024 depth=2	// L24351
  hls::stream< int8_t > v16025;
  #pragma HLS stream variable=v16025 depth=2	// L24352
  hls::stream< int8_t > v16026;
  #pragma HLS stream variable=v16026 depth=2	// L24353
  hls::stream< int8_t > v16027;
  #pragma HLS stream variable=v16027 depth=2	// L24354
  hls::stream< int8_t > v16028;
  #pragma HLS stream variable=v16028 depth=2	// L24355
  hls::stream< int8_t > v16029;
  #pragma HLS stream variable=v16029 depth=2	// L24356
  hls::stream< int8_t > v16030;
  #pragma HLS stream variable=v16030 depth=2	// L24357
  hls::stream< int8_t > v16031;
  #pragma HLS stream variable=v16031 depth=2	// L24358
  hls::stream< int8_t > v16032;
  #pragma HLS stream variable=v16032 depth=2	// L24359
  hls::stream< int8_t > v16033;
  #pragma HLS stream variable=v16033 depth=2	// L24360
  hls::stream< int8_t > v16034;
  #pragma HLS stream variable=v16034 depth=2	// L24361
  hls::stream< int8_t > v16035;
  #pragma HLS stream variable=v16035 depth=2	// L24362
  hls::stream< int8_t > v16036;
  #pragma HLS stream variable=v16036 depth=2	// L24363
  hls::stream< int8_t > v16037;
  #pragma HLS stream variable=v16037 depth=2	// L24364
  hls::stream< int8_t > v16038;
  #pragma HLS stream variable=v16038 depth=2	// L24365
  hls::stream< int8_t > v16039;
  #pragma HLS stream variable=v16039 depth=2	// L24366
  hls::stream< int8_t > v16040;
  #pragma HLS stream variable=v16040 depth=2	// L24367
  hls::stream< int8_t > v16041;
  #pragma HLS stream variable=v16041 depth=2	// L24368
  hls::stream< int8_t > v16042;
  #pragma HLS stream variable=v16042 depth=2	// L24369
  hls::stream< int8_t > v16043;
  #pragma HLS stream variable=v16043 depth=2	// L24370
  hls::stream< int8_t > v16044;
  #pragma HLS stream variable=v16044 depth=2	// L24371
  hls::stream< int8_t > v16045;
  #pragma HLS stream variable=v16045 depth=2	// L24372
  hls::stream< int8_t > v16046;
  #pragma HLS stream variable=v16046 depth=2	// L24373
  hls::stream< int8_t > v16047;
  #pragma HLS stream variable=v16047 depth=2	// L24374
  hls::stream< int8_t > v16048;
  #pragma HLS stream variable=v16048 depth=2	// L24375
  hls::stream< int8_t > v16049;
  #pragma HLS stream variable=v16049 depth=2	// L24376
  hls::stream< int8_t > v16050;
  #pragma HLS stream variable=v16050 depth=2	// L24377
  hls::stream< int8_t > v16051;
  #pragma HLS stream variable=v16051 depth=2	// L24378
  hls::stream< int8_t > v16052;
  #pragma HLS stream variable=v16052 depth=2	// L24379
  hls::stream< int8_t > v16053;
  #pragma HLS stream variable=v16053 depth=2	// L24380
  hls::stream< int8_t > v16054;
  #pragma HLS stream variable=v16054 depth=2	// L24381
  hls::stream< int8_t > v16055;
  #pragma HLS stream variable=v16055 depth=2	// L24382
  hls::stream< int8_t > v16056;
  #pragma HLS stream variable=v16056 depth=2	// L24383
  hls::stream< int8_t > v16057;
  #pragma HLS stream variable=v16057 depth=2	// L24384
  hls::stream< int8_t > v16058;
  #pragma HLS stream variable=v16058 depth=2	// L24385
  hls::stream< int8_t > v16059;
  #pragma HLS stream variable=v16059 depth=2	// L24386
  hls::stream< int8_t > v16060;
  #pragma HLS stream variable=v16060 depth=2	// L24387
  hls::stream< int8_t > v16061;
  #pragma HLS stream variable=v16061 depth=2	// L24388
  hls::stream< int8_t > v16062;
  #pragma HLS stream variable=v16062 depth=2	// L24389
  hls::stream< int8_t > v16063;
  #pragma HLS stream variable=v16063 depth=2	// L24390
  hls::stream< int8_t > v16064;
  #pragma HLS stream variable=v16064 depth=2	// L24391
  hls::stream< int8_t > v16065;
  #pragma HLS stream variable=v16065 depth=2	// L24392
  hls::stream< int8_t > v16066;
  #pragma HLS stream variable=v16066 depth=2	// L24393
  hls::stream< int8_t > v16067;
  #pragma HLS stream variable=v16067 depth=2	// L24394
  hls::stream< int8_t > v16068;
  #pragma HLS stream variable=v16068 depth=2	// L24395
  hls::stream< int8_t > v16069;
  #pragma HLS stream variable=v16069 depth=2	// L24396
  hls::stream< int8_t > v16070;
  #pragma HLS stream variable=v16070 depth=2	// L24397
  hls::stream< int8_t > v16071;
  #pragma HLS stream variable=v16071 depth=2	// L24398
  hls::stream< int8_t > v16072;
  #pragma HLS stream variable=v16072 depth=2	// L24399
  hls::stream< int8_t > v16073;
  #pragma HLS stream variable=v16073 depth=2	// L24400
  hls::stream< int8_t > v16074;
  #pragma HLS stream variable=v16074 depth=2	// L24401
  hls::stream< int8_t > v16075;
  #pragma HLS stream variable=v16075 depth=2	// L24402
  hls::stream< int8_t > v16076;
  #pragma HLS stream variable=v16076 depth=2	// L24403
  hls::stream< int8_t > v16077;
  #pragma HLS stream variable=v16077 depth=2	// L24404
  hls::stream< int8_t > v16078;
  #pragma HLS stream variable=v16078 depth=2	// L24405
  hls::stream< int8_t > v16079;
  #pragma HLS stream variable=v16079 depth=2	// L24406
  hls::stream< int8_t > v16080;
  #pragma HLS stream variable=v16080 depth=2	// L24407
  hls::stream< int8_t > v16081;
  #pragma HLS stream variable=v16081 depth=2	// L24408
  hls::stream< int8_t > v16082;
  #pragma HLS stream variable=v16082 depth=2	// L24409
  hls::stream< int8_t > v16083;
  #pragma HLS stream variable=v16083 depth=2	// L24410
  hls::stream< int8_t > v16084;
  #pragma HLS stream variable=v16084 depth=2	// L24411
  hls::stream< int8_t > v16085;
  #pragma HLS stream variable=v16085 depth=2	// L24412
  hls::stream< int8_t > v16086;
  #pragma HLS stream variable=v16086 depth=2	// L24413
  hls::stream< int8_t > v16087;
  #pragma HLS stream variable=v16087 depth=2	// L24414
  hls::stream< int8_t > v16088;
  #pragma HLS stream variable=v16088 depth=2	// L24415
  hls::stream< int8_t > v16089;
  #pragma HLS stream variable=v16089 depth=2	// L24416
  hls::stream< int8_t > v16090;
  #pragma HLS stream variable=v16090 depth=2	// L24417
  hls::stream< int8_t > v16091;
  #pragma HLS stream variable=v16091 depth=2	// L24418
  hls::stream< int8_t > v16092;
  #pragma HLS stream variable=v16092 depth=2	// L24419
  hls::stream< int8_t > v16093;
  #pragma HLS stream variable=v16093 depth=2	// L24420
  hls::stream< int8_t > v16094;
  #pragma HLS stream variable=v16094 depth=2	// L24421
  hls::stream< int8_t > v16095;
  #pragma HLS stream variable=v16095 depth=2	// L24422
  hls::stream< int8_t > v16096;
  #pragma HLS stream variable=v16096 depth=2	// L24423
  hls::stream< int8_t > v16097;
  #pragma HLS stream variable=v16097 depth=2	// L24424
  hls::stream< int8_t > v16098;
  #pragma HLS stream variable=v16098 depth=2	// L24425
  hls::stream< int8_t > v16099;
  #pragma HLS stream variable=v16099 depth=2	// L24426
  hls::stream< int8_t > v16100;
  #pragma HLS stream variable=v16100 depth=2	// L24427
  hls::stream< int8_t > v16101;
  #pragma HLS stream variable=v16101 depth=2	// L24428
  hls::stream< int8_t > v16102;
  #pragma HLS stream variable=v16102 depth=2	// L24429
  hls::stream< int8_t > v16103;
  #pragma HLS stream variable=v16103 depth=2	// L24430
  hls::stream< int8_t > v16104;
  #pragma HLS stream variable=v16104 depth=2	// L24431
  hls::stream< int8_t > v16105;
  #pragma HLS stream variable=v16105 depth=2	// L24432
  hls::stream< int8_t > v16106;
  #pragma HLS stream variable=v16106 depth=2	// L24433
  hls::stream< int8_t > v16107;
  #pragma HLS stream variable=v16107 depth=2	// L24434
  hls::stream< int8_t > v16108;
  #pragma HLS stream variable=v16108 depth=2	// L24435
  hls::stream< int8_t > v16109;
  #pragma HLS stream variable=v16109 depth=2	// L24436
  hls::stream< int8_t > v16110;
  #pragma HLS stream variable=v16110 depth=2	// L24437
  hls::stream< int8_t > v16111;
  #pragma HLS stream variable=v16111 depth=2	// L24438
  hls::stream< int8_t > v16112;
  #pragma HLS stream variable=v16112 depth=2	// L24439
  hls::stream< int8_t > v16113;
  #pragma HLS stream variable=v16113 depth=2	// L24440
  offchip_loadA_0(v15290, v15293);	// L24441
  offchip_loadB_0(v15291, v15294);	// L24442
  gemm_0_0(v15293, v15297, v15294, v15314);	// L24443
  gemm_0_1(v15314, v15858, v15315);	// L24444
  gemm_0_2(v15315, v15859, v15316);	// L24445
  gemm_0_3(v15316, v15860, v15317);	// L24446
  gemm_0_4(v15317, v15861, v15318);	// L24447
  gemm_0_5(v15318, v15862, v15319);	// L24448
  gemm_0_6(v15319, v15863, v15320);	// L24449
  gemm_0_7(v15320, v15864, v15321);	// L24450
  gemm_0_8(v15321, v15865, v15322);	// L24451
  gemm_0_9(v15322, v15866, v15323);	// L24452
  gemm_0_10(v15323, v15867, v15324);	// L24453
  gemm_0_11(v15324, v15868, v15325);	// L24454
  gemm_0_12(v15325, v15869, v15326);	// L24455
  gemm_0_13(v15326, v15870, v15327);	// L24456
  gemm_0_14(v15327, v15871, v15328);	// L24457
  gemm_0_15(v15328, v15872, v15329);	// L24458
  gemm_0_16(v15329, v15873);	// L24459
  gemm_1_0(v15297, v15602, v15298);	// L24460
  gemm_1_1(v15602, v15858, v15603, v15874, v15330);	// L24461
  gemm_1_2(v15603, v15859, v15604, v15875, v15331);	// L24462
  gemm_1_3(v15604, v15860, v15605, v15876, v15332);	// L24463
  gemm_1_4(v15605, v15861, v15606, v15877, v15333);	// L24464
  gemm_1_5(v15606, v15862, v15607, v15878, v15334);	// L24465
  gemm_1_6(v15607, v15863, v15608, v15879, v15335);	// L24466
  gemm_1_7(v15608, v15864, v15609, v15880, v15336);	// L24467
  gemm_1_8(v15609, v15865, v15610, v15881, v15337);	// L24468
  gemm_1_9(v15610, v15866, v15611, v15882, v15338);	// L24469
  gemm_1_10(v15611, v15867, v15612, v15883, v15339);	// L24470
  gemm_1_11(v15612, v15868, v15613, v15884, v15340);	// L24471
  gemm_1_12(v15613, v15869, v15614, v15885, v15341);	// L24472
  gemm_1_13(v15614, v15870, v15615, v15886, v15342);	// L24473
  gemm_1_14(v15615, v15871, v15616, v15887, v15343);	// L24474
  gemm_1_15(v15616, v15872, v15617, v15888, v15344);	// L24475
  gemm_1_16(v15617, v15873, v15889, v15345);	// L24476
  gemm_2_0(v15298, v15618, v15299);	// L24477
  gemm_2_1(v15618, v15874, v15619, v15890, v15330, v15346);	// L24478
  gemm_2_2(v15619, v15875, v15620, v15891, v15331, v15347);	// L24479
  gemm_2_3(v15620, v15876, v15621, v15892, v15332, v15348);	// L24480
  gemm_2_4(v15621, v15877, v15622, v15893, v15333, v15349);	// L24481
  gemm_2_5(v15622, v15878, v15623, v15894, v15334, v15350);	// L24482
  gemm_2_6(v15623, v15879, v15624, v15895, v15335, v15351);	// L24483
  gemm_2_7(v15624, v15880, v15625, v15896, v15336, v15352);	// L24484
  gemm_2_8(v15625, v15881, v15626, v15897, v15337, v15353);	// L24485
  gemm_2_9(v15626, v15882, v15627, v15898, v15338, v15354);	// L24486
  gemm_2_10(v15627, v15883, v15628, v15899, v15339, v15355);	// L24487
  gemm_2_11(v15628, v15884, v15629, v15900, v15340, v15356);	// L24488
  gemm_2_12(v15629, v15885, v15630, v15901, v15341, v15357);	// L24489
  gemm_2_13(v15630, v15886, v15631, v15902, v15342, v15358);	// L24490
  gemm_2_14(v15631, v15887, v15632, v15903, v15343, v15359);	// L24491
  gemm_2_15(v15632, v15888, v15633, v15904, v15344, v15360);	// L24492
  gemm_2_16(v15633, v15889, v15905, v15345, v15361);	// L24493
  gemm_3_0(v15299, v15634, v15300);	// L24494
  gemm_3_1(v15634, v15890, v15635, v15906, v15346, v15362);	// L24495
  gemm_3_2(v15635, v15891, v15636, v15907, v15347, v15363);	// L24496
  gemm_3_3(v15636, v15892, v15637, v15908, v15348, v15364);	// L24497
  gemm_3_4(v15637, v15893, v15638, v15909, v15349, v15365);	// L24498
  gemm_3_5(v15638, v15894, v15639, v15910, v15350, v15366);	// L24499
  gemm_3_6(v15639, v15895, v15640, v15911, v15351, v15367);	// L24500
  gemm_3_7(v15640, v15896, v15641, v15912, v15352, v15368);	// L24501
  gemm_3_8(v15641, v15897, v15642, v15913, v15353, v15369);	// L24502
  gemm_3_9(v15642, v15898, v15643, v15914, v15354, v15370);	// L24503
  gemm_3_10(v15643, v15899, v15644, v15915, v15355, v15371);	// L24504
  gemm_3_11(v15644, v15900, v15645, v15916, v15356, v15372);	// L24505
  gemm_3_12(v15645, v15901, v15646, v15917, v15357, v15373);	// L24506
  gemm_3_13(v15646, v15902, v15647, v15918, v15358, v15374);	// L24507
  gemm_3_14(v15647, v15903, v15648, v15919, v15359, v15375);	// L24508
  gemm_3_15(v15648, v15904, v15649, v15920, v15360, v15376);	// L24509
  gemm_3_16(v15649, v15905, v15921, v15361, v15377);	// L24510
  gemm_4_0(v15300, v15650, v15301);	// L24511
  gemm_4_1(v15650, v15906, v15651, v15922, v15362, v15378);	// L24512
  gemm_4_2(v15651, v15907, v15652, v15923, v15363, v15379);	// L24513
  gemm_4_3(v15652, v15908, v15653, v15924, v15364, v15380);	// L24514
  gemm_4_4(v15653, v15909, v15654, v15925, v15365, v15381);	// L24515
  gemm_4_5(v15654, v15910, v15655, v15926, v15366, v15382);	// L24516
  gemm_4_6(v15655, v15911, v15656, v15927, v15367, v15383);	// L24517
  gemm_4_7(v15656, v15912, v15657, v15928, v15368, v15384);	// L24518
  gemm_4_8(v15657, v15913, v15658, v15929, v15369, v15385);	// L24519
  gemm_4_9(v15658, v15914, v15659, v15930, v15370, v15386);	// L24520
  gemm_4_10(v15659, v15915, v15660, v15931, v15371, v15387);	// L24521
  gemm_4_11(v15660, v15916, v15661, v15932, v15372, v15388);	// L24522
  gemm_4_12(v15661, v15917, v15662, v15933, v15373, v15389);	// L24523
  gemm_4_13(v15662, v15918, v15663, v15934, v15374, v15390);	// L24524
  gemm_4_14(v15663, v15919, v15664, v15935, v15375, v15391);	// L24525
  gemm_4_15(v15664, v15920, v15665, v15936, v15376, v15392);	// L24526
  gemm_4_16(v15665, v15921, v15937, v15377, v15393);	// L24527
  gemm_5_0(v15301, v15666, v15302);	// L24528
  gemm_5_1(v15666, v15922, v15667, v15938, v15378, v15394);	// L24529
  gemm_5_2(v15667, v15923, v15668, v15939, v15379, v15395);	// L24530
  gemm_5_3(v15668, v15924, v15669, v15940, v15380, v15396);	// L24531
  gemm_5_4(v15669, v15925, v15670, v15941, v15381, v15397);	// L24532
  gemm_5_5(v15670, v15926, v15671, v15942, v15382, v15398);	// L24533
  gemm_5_6(v15671, v15927, v15672, v15943, v15383, v15399);	// L24534
  gemm_5_7(v15672, v15928, v15673, v15944, v15384, v15400);	// L24535
  gemm_5_8(v15673, v15929, v15674, v15945, v15385, v15401);	// L24536
  gemm_5_9(v15674, v15930, v15675, v15946, v15386, v15402);	// L24537
  gemm_5_10(v15675, v15931, v15676, v15947, v15387, v15403);	// L24538
  gemm_5_11(v15676, v15932, v15677, v15948, v15388, v15404);	// L24539
  gemm_5_12(v15677, v15933, v15678, v15949, v15389, v15405);	// L24540
  gemm_5_13(v15678, v15934, v15679, v15950, v15390, v15406);	// L24541
  gemm_5_14(v15679, v15935, v15680, v15951, v15391, v15407);	// L24542
  gemm_5_15(v15680, v15936, v15681, v15952, v15392, v15408);	// L24543
  gemm_5_16(v15681, v15937, v15953, v15393, v15409);	// L24544
  gemm_6_0(v15302, v15682, v15303);	// L24545
  gemm_6_1(v15682, v15938, v15683, v15954, v15394, v15410);	// L24546
  gemm_6_2(v15683, v15939, v15684, v15955, v15395, v15411);	// L24547
  gemm_6_3(v15684, v15940, v15685, v15956, v15396, v15412);	// L24548
  gemm_6_4(v15685, v15941, v15686, v15957, v15397, v15413);	// L24549
  gemm_6_5(v15686, v15942, v15687, v15958, v15398, v15414);	// L24550
  gemm_6_6(v15687, v15943, v15688, v15959, v15399, v15415);	// L24551
  gemm_6_7(v15688, v15944, v15689, v15960, v15400, v15416);	// L24552
  gemm_6_8(v15689, v15945, v15690, v15961, v15401, v15417);	// L24553
  gemm_6_9(v15690, v15946, v15691, v15962, v15402, v15418);	// L24554
  gemm_6_10(v15691, v15947, v15692, v15963, v15403, v15419);	// L24555
  gemm_6_11(v15692, v15948, v15693, v15964, v15404, v15420);	// L24556
  gemm_6_12(v15693, v15949, v15694, v15965, v15405, v15421);	// L24557
  gemm_6_13(v15694, v15950, v15695, v15966, v15406, v15422);	// L24558
  gemm_6_14(v15695, v15951, v15696, v15967, v15407, v15423);	// L24559
  gemm_6_15(v15696, v15952, v15697, v15968, v15408, v15424);	// L24560
  gemm_6_16(v15697, v15953, v15969, v15409, v15425);	// L24561
  gemm_7_0(v15303, v15698, v15304);	// L24562
  gemm_7_1(v15698, v15954, v15699, v15970, v15410, v15426);	// L24563
  gemm_7_2(v15699, v15955, v15700, v15971, v15411, v15427);	// L24564
  gemm_7_3(v15700, v15956, v15701, v15972, v15412, v15428);	// L24565
  gemm_7_4(v15701, v15957, v15702, v15973, v15413, v15429);	// L24566
  gemm_7_5(v15702, v15958, v15703, v15974, v15414, v15430);	// L24567
  gemm_7_6(v15703, v15959, v15704, v15975, v15415, v15431);	// L24568
  gemm_7_7(v15704, v15960, v15705, v15976, v15416, v15432);	// L24569
  gemm_7_8(v15705, v15961, v15706, v15977, v15417, v15433);	// L24570
  gemm_7_9(v15706, v15962, v15707, v15978, v15418, v15434);	// L24571
  gemm_7_10(v15707, v15963, v15708, v15979, v15419, v15435);	// L24572
  gemm_7_11(v15708, v15964, v15709, v15980, v15420, v15436);	// L24573
  gemm_7_12(v15709, v15965, v15710, v15981, v15421, v15437);	// L24574
  gemm_7_13(v15710, v15966, v15711, v15982, v15422, v15438);	// L24575
  gemm_7_14(v15711, v15967, v15712, v15983, v15423, v15439);	// L24576
  gemm_7_15(v15712, v15968, v15713, v15984, v15424, v15440);	// L24577
  gemm_7_16(v15713, v15969, v15985, v15425, v15441);	// L24578
  gemm_8_0(v15304, v15714, v15305);	// L24579
  gemm_8_1(v15714, v15970, v15715, v15986, v15426, v15442);	// L24580
  gemm_8_2(v15715, v15971, v15716, v15987, v15427, v15443);	// L24581
  gemm_8_3(v15716, v15972, v15717, v15988, v15428, v15444);	// L24582
  gemm_8_4(v15717, v15973, v15718, v15989, v15429, v15445);	// L24583
  gemm_8_5(v15718, v15974, v15719, v15990, v15430, v15446);	// L24584
  gemm_8_6(v15719, v15975, v15720, v15991, v15431, v15447);	// L24585
  gemm_8_7(v15720, v15976, v15721, v15992, v15432, v15448);	// L24586
  gemm_8_8(v15721, v15977, v15722, v15993, v15433, v15449);	// L24587
  gemm_8_9(v15722, v15978, v15723, v15994, v15434, v15450);	// L24588
  gemm_8_10(v15723, v15979, v15724, v15995, v15435, v15451);	// L24589
  gemm_8_11(v15724, v15980, v15725, v15996, v15436, v15452);	// L24590
  gemm_8_12(v15725, v15981, v15726, v15997, v15437, v15453);	// L24591
  gemm_8_13(v15726, v15982, v15727, v15998, v15438, v15454);	// L24592
  gemm_8_14(v15727, v15983, v15728, v15999, v15439, v15455);	// L24593
  gemm_8_15(v15728, v15984, v15729, v16000, v15440, v15456);	// L24594
  gemm_8_16(v15729, v15985, v16001, v15441, v15457);	// L24595
  gemm_9_0(v15305, v15730, v15306);	// L24596
  gemm_9_1(v15730, v15986, v15731, v16002, v15442, v15458);	// L24597
  gemm_9_2(v15731, v15987, v15732, v16003, v15443, v15459);	// L24598
  gemm_9_3(v15732, v15988, v15733, v16004, v15444, v15460);	// L24599
  gemm_9_4(v15733, v15989, v15734, v16005, v15445, v15461);	// L24600
  gemm_9_5(v15734, v15990, v15735, v16006, v15446, v15462);	// L24601
  gemm_9_6(v15735, v15991, v15736, v16007, v15447, v15463);	// L24602
  gemm_9_7(v15736, v15992, v15737, v16008, v15448, v15464);	// L24603
  gemm_9_8(v15737, v15993, v15738, v16009, v15449, v15465);	// L24604
  gemm_9_9(v15738, v15994, v15739, v16010, v15450, v15466);	// L24605
  gemm_9_10(v15739, v15995, v15740, v16011, v15451, v15467);	// L24606
  gemm_9_11(v15740, v15996, v15741, v16012, v15452, v15468);	// L24607
  gemm_9_12(v15741, v15997, v15742, v16013, v15453, v15469);	// L24608
  gemm_9_13(v15742, v15998, v15743, v16014, v15454, v15470);	// L24609
  gemm_9_14(v15743, v15999, v15744, v16015, v15455, v15471);	// L24610
  gemm_9_15(v15744, v16000, v15745, v16016, v15456, v15472);	// L24611
  gemm_9_16(v15745, v16001, v16017, v15457, v15473);	// L24612
  gemm_10_0(v15306, v15746, v15307);	// L24613
  gemm_10_1(v15746, v16002, v15747, v16018, v15458, v15474);	// L24614
  gemm_10_2(v15747, v16003, v15748, v16019, v15459, v15475);	// L24615
  gemm_10_3(v15748, v16004, v15749, v16020, v15460, v15476);	// L24616
  gemm_10_4(v15749, v16005, v15750, v16021, v15461, v15477);	// L24617
  gemm_10_5(v15750, v16006, v15751, v16022, v15462, v15478);	// L24618
  gemm_10_6(v15751, v16007, v15752, v16023, v15463, v15479);	// L24619
  gemm_10_7(v15752, v16008, v15753, v16024, v15464, v15480);	// L24620
  gemm_10_8(v15753, v16009, v15754, v16025, v15465, v15481);	// L24621
  gemm_10_9(v15754, v16010, v15755, v16026, v15466, v15482);	// L24622
  gemm_10_10(v15755, v16011, v15756, v16027, v15467, v15483);	// L24623
  gemm_10_11(v15756, v16012, v15757, v16028, v15468, v15484);	// L24624
  gemm_10_12(v15757, v16013, v15758, v16029, v15469, v15485);	// L24625
  gemm_10_13(v15758, v16014, v15759, v16030, v15470, v15486);	// L24626
  gemm_10_14(v15759, v16015, v15760, v16031, v15471, v15487);	// L24627
  gemm_10_15(v15760, v16016, v15761, v16032, v15472, v15488);	// L24628
  gemm_10_16(v15761, v16017, v16033, v15473, v15489);	// L24629
  gemm_11_0(v15307, v15762, v15308);	// L24630
  gemm_11_1(v15762, v16018, v15763, v16034, v15474, v15490);	// L24631
  gemm_11_2(v15763, v16019, v15764, v16035, v15475, v15491);	// L24632
  gemm_11_3(v15764, v16020, v15765, v16036, v15476, v15492);	// L24633
  gemm_11_4(v15765, v16021, v15766, v16037, v15477, v15493);	// L24634
  gemm_11_5(v15766, v16022, v15767, v16038, v15478, v15494);	// L24635
  gemm_11_6(v15767, v16023, v15768, v16039, v15479, v15495);	// L24636
  gemm_11_7(v15768, v16024, v15769, v16040, v15480, v15496);	// L24637
  gemm_11_8(v15769, v16025, v15770, v16041, v15481, v15497);	// L24638
  gemm_11_9(v15770, v16026, v15771, v16042, v15482, v15498);	// L24639
  gemm_11_10(v15771, v16027, v15772, v16043, v15483, v15499);	// L24640
  gemm_11_11(v15772, v16028, v15773, v16044, v15484, v15500);	// L24641
  gemm_11_12(v15773, v16029, v15774, v16045, v15485, v15501);	// L24642
  gemm_11_13(v15774, v16030, v15775, v16046, v15486, v15502);	// L24643
  gemm_11_14(v15775, v16031, v15776, v16047, v15487, v15503);	// L24644
  gemm_11_15(v15776, v16032, v15777, v16048, v15488, v15504);	// L24645
  gemm_11_16(v15777, v16033, v16049, v15489, v15505);	// L24646
  gemm_12_0(v15308, v15778, v15309);	// L24647
  gemm_12_1(v15778, v16034, v15779, v16050, v15490, v15506);	// L24648
  gemm_12_2(v15779, v16035, v15780, v16051, v15491, v15507);	// L24649
  gemm_12_3(v15780, v16036, v15781, v16052, v15492, v15508);	// L24650
  gemm_12_4(v15781, v16037, v15782, v16053, v15493, v15509);	// L24651
  gemm_12_5(v15782, v16038, v15783, v16054, v15494, v15510);	// L24652
  gemm_12_6(v15783, v16039, v15784, v16055, v15495, v15511);	// L24653
  gemm_12_7(v15784, v16040, v15785, v16056, v15496, v15512);	// L24654
  gemm_12_8(v15785, v16041, v15786, v16057, v15497, v15513);	// L24655
  gemm_12_9(v15786, v16042, v15787, v16058, v15498, v15514);	// L24656
  gemm_12_10(v15787, v16043, v15788, v16059, v15499, v15515);	// L24657
  gemm_12_11(v15788, v16044, v15789, v16060, v15500, v15516);	// L24658
  gemm_12_12(v15789, v16045, v15790, v16061, v15501, v15517);	// L24659
  gemm_12_13(v15790, v16046, v15791, v16062, v15502, v15518);	// L24660
  gemm_12_14(v15791, v16047, v15792, v16063, v15503, v15519);	// L24661
  gemm_12_15(v15792, v16048, v15793, v16064, v15504, v15520);	// L24662
  gemm_12_16(v15793, v16049, v16065, v15505, v15521);	// L24663
  gemm_13_0(v15309, v15794, v15310);	// L24664
  gemm_13_1(v15794, v16050, v15795, v16066, v15506, v15522);	// L24665
  gemm_13_2(v15795, v16051, v15796, v16067, v15507, v15523);	// L24666
  gemm_13_3(v15796, v16052, v15797, v16068, v15508, v15524);	// L24667
  gemm_13_4(v15797, v16053, v15798, v16069, v15509, v15525);	// L24668
  gemm_13_5(v15798, v16054, v15799, v16070, v15510, v15526);	// L24669
  gemm_13_6(v15799, v16055, v15800, v16071, v15511, v15527);	// L24670
  gemm_13_7(v15800, v16056, v15801, v16072, v15512, v15528);	// L24671
  gemm_13_8(v15801, v16057, v15802, v16073, v15513, v15529);	// L24672
  gemm_13_9(v15802, v16058, v15803, v16074, v15514, v15530);	// L24673
  gemm_13_10(v15803, v16059, v15804, v16075, v15515, v15531);	// L24674
  gemm_13_11(v15804, v16060, v15805, v16076, v15516, v15532);	// L24675
  gemm_13_12(v15805, v16061, v15806, v16077, v15517, v15533);	// L24676
  gemm_13_13(v15806, v16062, v15807, v16078, v15518, v15534);	// L24677
  gemm_13_14(v15807, v16063, v15808, v16079, v15519, v15535);	// L24678
  gemm_13_15(v15808, v16064, v15809, v16080, v15520, v15536);	// L24679
  gemm_13_16(v15809, v16065, v16081, v15521, v15537);	// L24680
  gemm_14_0(v15310, v15810, v15311);	// L24681
  gemm_14_1(v15810, v16066, v15811, v16082, v15522, v15538);	// L24682
  gemm_14_2(v15811, v16067, v15812, v16083, v15523, v15539);	// L24683
  gemm_14_3(v15812, v16068, v15813, v16084, v15524, v15540);	// L24684
  gemm_14_4(v15813, v16069, v15814, v16085, v15525, v15541);	// L24685
  gemm_14_5(v15814, v16070, v15815, v16086, v15526, v15542);	// L24686
  gemm_14_6(v15815, v16071, v15816, v16087, v15527, v15543);	// L24687
  gemm_14_7(v15816, v16072, v15817, v16088, v15528, v15544);	// L24688
  gemm_14_8(v15817, v16073, v15818, v16089, v15529, v15545);	// L24689
  gemm_14_9(v15818, v16074, v15819, v16090, v15530, v15546);	// L24690
  gemm_14_10(v15819, v16075, v15820, v16091, v15531, v15547);	// L24691
  gemm_14_11(v15820, v16076, v15821, v16092, v15532, v15548);	// L24692
  gemm_14_12(v15821, v16077, v15822, v16093, v15533, v15549);	// L24693
  gemm_14_13(v15822, v16078, v15823, v16094, v15534, v15550);	// L24694
  gemm_14_14(v15823, v16079, v15824, v16095, v15535, v15551);	// L24695
  gemm_14_15(v15824, v16080, v15825, v16096, v15536, v15552);	// L24696
  gemm_14_16(v15825, v16081, v16097, v15537, v15553);	// L24697
  gemm_15_0(v15311, v15826, v15312);	// L24698
  gemm_15_1(v15826, v16082, v15827, v16098, v15538, v15554);	// L24699
  gemm_15_2(v15827, v16083, v15828, v16099, v15539, v15555);	// L24700
  gemm_15_3(v15828, v16084, v15829, v16100, v15540, v15556);	// L24701
  gemm_15_4(v15829, v16085, v15830, v16101, v15541, v15557);	// L24702
  gemm_15_5(v15830, v16086, v15831, v16102, v15542, v15558);	// L24703
  gemm_15_6(v15831, v16087, v15832, v16103, v15543, v15559);	// L24704
  gemm_15_7(v15832, v16088, v15833, v16104, v15544, v15560);	// L24705
  gemm_15_8(v15833, v16089, v15834, v16105, v15545, v15561);	// L24706
  gemm_15_9(v15834, v16090, v15835, v16106, v15546, v15562);	// L24707
  gemm_15_10(v15835, v16091, v15836, v16107, v15547, v15563);	// L24708
  gemm_15_11(v15836, v16092, v15837, v16108, v15548, v15564);	// L24709
  gemm_15_12(v15837, v16093, v15838, v16109, v15549, v15565);	// L24710
  gemm_15_13(v15838, v16094, v15839, v16110, v15550, v15566);	// L24711
  gemm_15_14(v15839, v16095, v15840, v16111, v15551, v15567);	// L24712
  gemm_15_15(v15840, v16096, v15841, v16112, v15552, v15568);	// L24713
  gemm_15_16(v15841, v16097, v16113, v15553, v15569);	// L24714
  gemm_16_0(v15312, v15842);	// L24715
  gemm_16_1(v15842, v16098, v15843, v15554, v15570);	// L24716
  gemm_16_2(v15843, v16099, v15844, v15555, v15571);	// L24717
  gemm_16_3(v15844, v16100, v15845, v15556, v15572);	// L24718
  gemm_16_4(v15845, v16101, v15846, v15557, v15573);	// L24719
  gemm_16_5(v15846, v16102, v15847, v15558, v15574);	// L24720
  gemm_16_6(v15847, v16103, v15848, v15559, v15575);	// L24721
  gemm_16_7(v15848, v16104, v15849, v15560, v15576);	// L24722
  gemm_16_8(v15849, v16105, v15850, v15561, v15577);	// L24723
  gemm_16_9(v15850, v16106, v15851, v15562, v15578);	// L24724
  gemm_16_10(v15851, v16107, v15852, v15563, v15579);	// L24725
  gemm_16_11(v15852, v16108, v15853, v15564, v15580);	// L24726
  gemm_16_12(v15853, v16109, v15854, v15565, v15581);	// L24727
  gemm_16_13(v15854, v16110, v15855, v15566, v15582);	// L24728
  gemm_16_14(v15855, v16111, v15856, v15567, v15583);	// L24729
  gemm_16_15(v15856, v16112, v15857, v15568, v15584);	// L24730
  gemm_16_16(v15857, v16113, v15569, v15585);	// L24731
  gemm_17_1(v15585, v15586);	// L24732
  gemm_17_2(v15584, v15587, v15586);	// L24733
  gemm_17_3(v15583, v15588, v15587);	// L24734
  gemm_17_4(v15582, v15589, v15588);	// L24735
  gemm_17_5(v15581, v15590, v15589);	// L24736
  gemm_17_6(v15580, v15591, v15590);	// L24737
  gemm_17_7(v15579, v15592, v15591);	// L24738
  gemm_17_8(v15578, v15593, v15592);	// L24739
  gemm_17_9(v15577, v15594, v15593);	// L24740
  gemm_17_10(v15576, v15595, v15594);	// L24741
  gemm_17_11(v15575, v15596, v15595);	// L24742
  gemm_17_12(v15574, v15597, v15596);	// L24743
  gemm_17_13(v15573, v15598, v15597);	// L24744
  gemm_17_14(v15572, v15599, v15598);	// L24745
  gemm_17_15(v15571, v15600, v15599);	// L24746
  gemm_17_16(v15570, v15601, v15600);	// L24747
  gemm_17_17(v15601, v15295);	// L24748
  offchip_store_0(v15292, v15295);	// L24749
}


} // extern "C"

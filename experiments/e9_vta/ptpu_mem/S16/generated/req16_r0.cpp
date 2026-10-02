
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void req16_r0_0(
  hls::stream< int64_t >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int64_t >& v2,
  hls::stream< int64_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int64_t >& v5
) {	// L2
  int32_t pc;	// L5
  pc = 0;	// L6
  int32_t idx;	// L9
  idx = 0;	// L10
  int64_t cfg;	// L14
  cfg = 0;	// L15
  bool bias;	// L19
  bias = 0;	// L20
  bool reuse;	// L24
  reuse = 0;	// L25
  bool final;	// L29
  final = 0;	// L30
  int32_t kbn;	// L33
  kbn = 0;	// L34
  int32_t nbn;	// L37
  nbn = 0;	// L38
  int32_t tn;	// L41
  tn = 0;	// L42
  int32_t abase;	// L45
  abase = 0;	// L46
  int32_t wbase;	// L49
  wbase = 0;	// L50
  int32_t bbase;	// L53
  bbase = 0;	// L54
  int32_t ybase;	// L57
  ybase = 0;	// L58
  int32_t aptr;	// L61
  aptr = 0;	// L62
  int32_t wptr;	// L65
  wptr = 0;	// L66
  int32_t bptr;	// L69
  bptr = 0;	// L70
  int32_t yptr;	// L73
  yptr = 0;	// L74
  int32_t rows;	// L77
  rows = 0;	// L78
  int32_t abeats;	// L81
  abeats = 0;	// L82
  int32_t kleft;	// L85
  kleft = 0;	// L86
  int32_t nleft;	// L89
  nleft = 0;	// L90
  int32_t tleft;	// L93
  tleft = 0;	// L94
  bool kfirst;	// L98
  kfirst = 0;	// L99
  bool klast;	// L103
  klast = 0;	// L104
  bool nfirst;	// L108
  nfirst = 0;	// L109
  bool nlast;	// L113
  nlast = 0;	// L114
  bool tlast;	// L118
  tlast = 0;	// L119
  int32_t st;	// L122
  st = 0;	// L123
  bool go;	// L127
  go = 1;	// L128
  while (true) {	// L129
    #pragma HLS pipeline II=1 style=flp
    bool v35 = go;	// L130
    if (!(v35)) break;
    bool adv;	// L140
    adv = 0;	// L141
    int32_t v37 = st;	// L142
    bool v38 = v37 == 0;	// L145
    if (v38) {	// L146
      int64_t v39 = v1.read();	// L147
      int64_t la;	// L148
      la = v39;	// L149
      int64_t v41 = la;	// L150
      int64_t v42 = v41 & 1073741823;	// L154
      int32_t v43 = v42;	// L155
      pc = v43;	// L156
      st = 1;	// L159
    } else {
      int32_t v44 = st;	// L161
      bool v45 = v44 == 1;	// L164
      if (v45) {	// L165
        int32_t v46 = pc;	// L166
        int64_t v47 = v46;	// L167
        int64_t p64;	// L168
        p64 = v47;	// L169
        int64_t four;	// L173
        four = 4;	// L174
        int64_t v50 = four;	// L175
        int64_t v51 = v50 << 32;	// L179
        int64_t v52 = p64;	// L180
        int64_t v53 = v51 | v52;	// L181
        v3.write(v53);	// L182
        v4.write(1024);	// L185
        int32_t v54 = pc;	// L186
        ap_int<33> v55 = v54;	// L187
        ap_int<33> v56 = v55 + 32;	// L191
        int32_t v57 = v56;	// L192
        pc = v57;	// L193
        idx = 0;	// L196
        st = 2;	// L199
      } else {
        int32_t v58 = st;	// L201
        bool v59 = v58 == 2;	// L204
        if (v59) {	// L205
          int64_t v60 = v0.read();	// L206
          int64_t w;	// L207
          w = v60;	// L208
          int32_t v62 = idx;	// L209
          bool v63 = v62 == 0;	// L212
          if (v63) {	// L213
            int64_t v64 = w;	// L214
            cfg = v64;	// L215
            int64_t v65 = w;	// L216
            int64_t v66 = v65 >> 6;	// L220
            int64_t v67 = v66 & 1;	// L224
            bool v68 = v67;	// L225
            reuse = v68;	// L226
            int64_t v69 = w;	// L227
            int64_t v70 = v69 >> 7;	// L231
            int64_t v71 = v70 & 1;	// L235
            bool v72 = v71;	// L236
            bias = v72;	// L237
            int64_t v73 = w;	// L238
            int64_t v74 = v73 >> 8;	// L242
            int64_t v75 = v74 & 1;	// L246
            bool v76 = v75;	// L247
            final = v76;	// L248
            int64_t v77 = w;	// L249
            int64_t v78 = v77 >> 16;	// L253
            int64_t v79 = v78 & 65535;	// L257
            int32_t v80 = v79;	// L258
            kbn = v80;	// L259
            int64_t v81 = w;	// L260
            int64_t v82 = v81 >> 32;	// L264
            int64_t v83 = v82 & 4095;	// L268
            int32_t v84 = v83;	// L269
            nbn = v84;	// L270
            int64_t v85 = w;	// L271
            int64_t v86 = v85 >> 52;	// L275
            int64_t v87 = v86 & 4095;	// L279
            int32_t v88 = v87;	// L280
            tn = v88;	// L281
          } else {
            int32_t v89 = idx;	// L283
            bool v90 = v89 == 1;	// L286
            if (v90) {	// L287
              int64_t v91 = w;	// L288
              int64_t v92 = v91 & 1073741823;	// L292
              int32_t v93 = v92;	// L293
              abase = v93;	// L294
              int64_t v94 = w;	// L295
              int64_t v95 = v94 >> 32;	// L299
              int64_t v96 = v95 & 1073741823;	// L303
              int32_t v97 = v96;	// L304
              wbase = v97;	// L305
            } else {
              int32_t v98 = idx;	// L307
              bool v99 = v98 == 2;	// L310
              if (v99) {	// L311
                int64_t v100 = w;	// L312
                int64_t v101 = v100 & 1073741823;	// L316
                int32_t v102 = v101;	// L317
                bbase = v102;	// L318
                int64_t v103 = w;	// L319
                int64_t v104 = v103 >> 32;	// L323
                int64_t v105 = v104 & 1073741823;	// L327
                int32_t v106 = v105;	// L328
                ybase = v106;	// L329
              } else {
                int64_t v107 = w;	// L331
                int64_t v108 = v107 & 65535;	// L335
                int32_t v109 = v108;	// L336
                rows = v109;	// L337
                int64_t v110 = w;	// L338
                int64_t v111 = v110 >> 16;	// L342
                int64_t v112 = v111 & 65535;	// L346
                int32_t v113 = v112;	// L347
                abeats = v113;	// L348
                st = 3;	// L351
              }
            }
          }
          int32_t v114 = idx;	// L355
          ap_int<33> v115 = v114;	// L356
          ap_int<33> v116 = v115 + 1;	// L360
          int32_t v117 = v116;	// L361
          idx = v117;	// L362
        } else {
          int32_t v118 = st;	// L364
          bool v119 = v118 == 3;	// L367
          if (v119) {	// L368
            int64_t v120 = cfg;	// L369
            v2.write(v120);	// L370
            int32_t v121 = abase;	// L371
            aptr = v121;	// L372
            int32_t v122 = wbase;	// L373
            wptr = v122;	// L374
            int32_t v123 = ybase;	// L375
            yptr = v123;	// L376
            int32_t v124 = tn;	// L377
            tleft = v124;	// L378
            tlast = 0;	// L382
            int32_t v125 = tn;	// L383
            bool v126 = v125 == 0;	// L386
            if (v126) {	// L387
              tlast = 1;	// L391
            }
            st = 4;	// L395
          } else {
            int32_t v127 = st;	// L397
            bool v128 = v127 == 4;	// L400
            if (v128) {	// L401
              int64_t last;	// L405
              last = 0;	// L406
              bool v130 = final;	// L407
              bool v131 = tlast;	// L408
              bool v132 = v130 & v131;	// L409
              if (v132) {	// L414
                last = 1;	// L418
              }
              int32_t v133 = rows;	// L420
              int64_t v134 = v133;	// L421
              int64_t r64;	// L422
              r64 = v134;	// L423
              int32_t v136 = yptr;	// L424
              int64_t v137 = v136;	// L425
              int64_t y64;	// L426
              y64 = v137;	// L427
              int64_t v139 = last;	// L428
              int64_t v140 = v139 << 62;	// L432
              int64_t v141 = r64;	// L433
              int64_t v142 = v141 << 32;	// L437
              int64_t v143 = v140 | v142;	// L438
              int64_t v144 = y64;	// L439
              int64_t v145 = v143 | v144;	// L440
              v5.write(v145);	// L441
              int32_t v146 = yptr;	// L442
              int32_t v147 = rows;	// L443
              int32_t v148 = v147 << 4;	// L446
              ap_int<33> v149 = v146;	// L447
              ap_int<33> v150 = v148;	// L448
              ap_int<33> v151 = v149 + v150;	// L449
              int32_t v152 = v151;	// L450
              yptr = v152;	// L451
              bool v153 = reuse;	// L452
              if (v153) {	// L457
                int32_t v154 = wbase;	// L458
                wptr = v154;	// L459
              }
              int32_t v155 = bbase;	// L461
              bptr = v155;	// L462
              int32_t v156 = kbn;	// L463
              kleft = v156;	// L464
              kfirst = 1;	// L468
              klast = 0;	// L472
              int32_t v157 = kbn;	// L473
              bool v158 = v157 == 0;	// L476
              if (v158) {	// L477
                klast = 1;	// L481
              }
              int32_t v159 = nbn;	// L483
              nleft = v159;	// L484
              nfirst = 1;	// L488
              nlast = 0;	// L492
              int32_t v160 = nbn;	// L493
              bool v161 = v160 == 0;	// L496
              if (v161) {	// L497
                nlast = 1;	// L501
              }
              st = 6;	// L505
              bool v162 = bias;	// L506
              if (v162) {	// L511
                st = 5;	// L514
              }
            } else {
              int32_t v163 = st;	// L517
              bool v164 = v163 == 5;	// L520
              if (v164) {	// L521
                int32_t v165 = bptr;	// L522
                int64_t v166 = v165;	// L523
                int64_t b64;	// L524
                b64 = v166;	// L525
                int64_t nb64;	// L529
                nb64 = 8;	// L530
                int64_t v169 = nb64;	// L531
                int64_t v170 = v169 << 32;	// L535
                int64_t v171 = b64;	// L536
                int64_t v172 = v170 | v171;	// L537
                v3.write(v172);	// L538
                v4.write(2057);	// L541
                int32_t v173 = bptr;	// L542
                ap_int<33> v174 = v173;	// L543
                ap_int<33> v175 = v174 + 64;	// L547
                int32_t v176 = v175;	// L548
                bptr = v176;	// L549
                st = 6;	// L552
              } else {
                int32_t v177 = st;	// L554
                bool v178 = v177 == 6;	// L557
                if (v178) {	// L558
                  int32_t v179 = wptr;	// L559
                  int64_t v180 = v179;	// L560
                  int64_t w64;	// L561
                  w64 = v180;	// L562
                  int64_t nw64;	// L566
                  nw64 = 32;	// L567
                  int64_t v183 = nw64;	// L568
                  int64_t v184 = v183 << 32;	// L572
                  int64_t v185 = w64;	// L573
                  int64_t v186 = v184 | v185;	// L574
                  v3.write(v186);	// L575
                  int32_t tg;	// L578
                  tg = 8202;	// L579
                  bool v188 = nfirst;	// L580
                  if (v188) {	// L585
                    st = 7;	// L588
                  } else {
                    adv = 1;	// L593
                    bool v189 = final;	// L594
                    bool v190 = tlast;	// L595
                    bool v191 = v189 & v190;	// L596
                    bool v192 = klast;	// L597
                    bool v193 = v191 & v192;	// L598
                    bool v194 = nlast;	// L599
                    bool v195 = v193 & v194;	// L600
                    if (v195) {	// L605
                      tg = 8198;	// L608
                    }
                  }
                  int32_t v196 = tg;	// L611
                  v4.write(v196);	// L612
                  int32_t v197 = wptr;	// L613
                  ap_int<33> v198 = v197;	// L614
                  ap_int<33> v199 = v198 + 256;	// L618
                  int32_t v200 = v199;	// L619
                  wptr = v200;	// L620
                } else {
                  int32_t v201 = aptr;	// L622
                  int64_t v202 = v201;	// L623
                  int64_t a64;	// L624
                  a64 = v202;	// L625
                  int32_t v204 = abeats;	// L626
                  int64_t v205 = v204;	// L627
                  int64_t na64;	// L628
                  na64 = v205;	// L629
                  int64_t v207 = na64;	// L630
                  int64_t v208 = v207 << 32;	// L634
                  int64_t v209 = a64;	// L635
                  int64_t v210 = v208 | v209;	// L636
                  v3.write(v210);	// L637
                  int32_t v211 = abeats;	// L643
                  int32_t v212 = v211 << 8;	// L646
                  int32_t v213 = v212 | 11;	// L647
                  int32_t tc;	// L648
                  tc = v213;	// L649
                  bool v215 = final;	// L650
                  bool v216 = tlast;	// L651
                  bool v217 = v215 & v216;	// L652
                  bool v218 = klast;	// L653
                  bool v219 = v217 & v218;	// L654
                  bool v220 = nlast;	// L655
                  bool v221 = v219 & v220;	// L656
                  if (v221) {	// L661
                    int32_t v222 = abeats;	// L667
                    int32_t v223 = v222 << 8;	// L670
                    int32_t v224 = v223 | 7;	// L671
                    tc = v224;	// L672
                  }
                  int32_t v225 = tc;	// L674
                  v4.write(v225);	// L675
                  int32_t v226 = aptr;	// L676
                  int32_t v227 = abeats;	// L677
                  int32_t v228 = v227 << 3;	// L680
                  ap_int<33> v229 = v226;	// L681
                  ap_int<33> v230 = v228;	// L682
                  ap_int<33> v231 = v229 + v230;	// L683
                  int32_t v232 = v231;	// L684
                  aptr = v232;	// L685
                  adv = 1;	// L689
                }
              }
            }
          }
        }
      }
    }
    bool v233 = adv;	// L697
    if (v233) {	// L702
      st = 6;	// L705
      bool v234 = nlast;	// L706
      int32_t v235 = v234;	// L707
      bool v236 = v235 == 0;	// L710
      if (v236) {	// L711
        nfirst = 0;	// L715
        int32_t v237 = nleft;	// L716
        bool v238 = v237 == 1;	// L719
        if (v238) {	// L720
          nlast = 1;	// L724
        }
        int32_t v239 = nleft;	// L726
        ap_int<33> v240 = v239;	// L727
        ap_int<33> v241 = v240 - 1;	// L731
        int32_t v242 = v241;	// L732
        nleft = v242;	// L733
        bool v243 = bias;	// L734
        bool v244 = kfirst;	// L735
        bool v245 = v243 & v244;	// L736
        if (v245) {	// L741
          st = 5;	// L744
        }
      } else {
        int32_t v246 = nbn;	// L747
        nleft = v246;	// L748
        nfirst = 1;	// L752
        nlast = 0;	// L756
        int32_t v247 = nbn;	// L757
        bool v248 = v247 == 0;	// L760
        if (v248) {	// L761
          nlast = 1;	// L765
        }
        bool v249 = klast;	// L767
        int32_t v250 = v249;	// L768
        bool v251 = v250 == 0;	// L771
        if (v251) {	// L772
          kfirst = 0;	// L776
          int32_t v252 = kleft;	// L777
          bool v253 = v252 == 1;	// L780
          if (v253) {	// L781
            klast = 1;	// L785
          }
          int32_t v254 = kleft;	// L787
          ap_int<33> v255 = v254;	// L788
          ap_int<33> v256 = v255 - 1;	// L792
          int32_t v257 = v256;	// L793
          kleft = v257;	// L794
        } else {
          bool v258 = tlast;	// L796
          int32_t v259 = v258;	// L797
          bool v260 = v259 == 0;	// L800
          if (v260) {	// L801
            int32_t v261 = tleft;	// L802
            bool v262 = v261 == 1;	// L805
            if (v262) {	// L806
              tlast = 1;	// L810
            }
            int32_t v263 = tleft;	// L812
            ap_int<33> v264 = v263;	// L813
            ap_int<33> v265 = v264 - 1;	// L817
            int32_t v266 = v265;	// L818
            tleft = v266;	// L819
            st = 4;	// L822
          } else {
            bool v267 = final;	// L824
            if (v267) {	// L829
              go = 0;	// L833
            } else {
              st = 1;	// L837
            }
          }
        }
      }
    }
  }
}



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
/// This is top function.
void head4_r0_0(
  hls::stream< hls::vector< int8_t, 4 > >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< hls::vector< int8_t, 9 > >& v3,
  hls::stream< int64_t >& v4,
  hls::stream< int64_t >& v5,
  hls::stream< hls::vector< int8_t, 4 > >& v6
) {	// L2
  static int8_t ab0[64] = {0};	// L6
  static int8_t ab1[64] = {0};	// L11
  static int8_t ab2[64] = {0};	// L16
  static int8_t ab3[64] = {0};	// L21
  int64_t cfg;	// L26
  cfg = 0;	// L27
  bool final;	// L31
  final = 0;	// L32
  bool biased;	// L36
  biased = 0;	// L37
  int32_t kbn;	// L40
  kbn = 0;	// L41
  int32_t nbn;	// L44
  nbn = 0;	// L45
  int32_t ln;	// L48
  ln = 0;	// L49
  bool ksingle;	// L53
  ksingle = 0;	// L54
  bool nsingle;	// L58
  nsingle = 0;	// L59
  bool lsingle;	// L63
  lsingle = 0;	// L64
  int64_t ncfg;	// L68
  ncfg = 0;	// L69
  bool nfinal;	// L73
  nfinal = 0;	// L74
  bool nbiased;	// L78
  nbiased = 0;	// L79
  int32_t nkbn;	// L82
  nkbn = 0;	// L83
  int32_t nnbn;	// L86
  nnbn = 0;	// L87
  int32_t nln;	// L90
  nln = 0;	// L91
  int32_t ntn;	// L94
  ntn = 0;	// L95
  int32_t tleft;	// L98
  tleft = 0;	// L99
  int32_t kleft;	// L102
  kleft = 0;	// L103
  int32_t nleft;	// L106
  nleft = 0;	// L107
  int32_t rl;	// L110
  rl = 0;	// L111
  int32_t r;	// L114
  r = 0;	// L115
  bool tlast;	// L119
  tlast = 0;	// L120
  bool kfirst;	// L124
  kfirst = 0;	// L125
  bool klast;	// L129
  klast = 0;	// L130
  bool nfirst;	// L134
  nfirst = 0;	// L135
  bool nlast;	// L139
  nlast = 0;	// L140
  bool rfirst;	// L144
  rfirst = 0;	// L145
  bool tail;	// L149
  tail = 0;	// L150
  bool isopen;	// L154
  isopen = 1;	// L155
  bool fetched;	// L159
  fetched = 0;	// L160
  int32_t space;	// L163
  space = 2048;	// L164
  bool noroom;	// L168
  noroom = 0;	// L169
  int64_t bb;	// L173
  bb = 0;	// L174
  int32_t st;	// L177
  st = 0;	// L178
  bool go;	// L182
  go = 1;	// L183
  while (true) {	// L184
    #pragma HLS pipeline II=1 style=flp
    bool v50 = go;	// L185
    if (!(v50)) break;
    bool v51 = nlast;	// L192
    bool v52 = klast;	// L193
    bool v53 = v51 & v52;	// L194
    bool v54 = tlast;	// L195
    bool v55 = v53 & v54;	// L196
    bool lastblk;	// L197
    lastblk = v55;	// L198
    bool want;	// L202
    want = 0;	// L203
    int32_t v58 = st;	// L204
    bool v59 = v58 == 0;	// L207
    if (v59) {	// L208
      want = 1;	// L212
    } else {
      int32_t v60 = st;	// L214
      bool v61 = v60 == 2;	// L217
      if (v61) {	// L218
        bool v62 = tail;	// L219
        bool v63 = lastblk;	// L220
        bool v64 = v62 & v63;	// L221
        if (v64) {	// L226
          bool v65 = final;	// L227
          bool v66 = fetched;	// L228
          bool v67 = v65 | v66;	// L229
          bool v68 = isopen;	// L230
          bool v69 = v67 | v68;	// L231
          int32_t v70 = v69;	// L232
          bool v71 = v70 == 0;	// L235
          if (v71) {	// L236
            want = 1;	// L240
          }
        }
      }
    }
    bool v72 = want;	// L245
    if (v72) {	// L250
      int64_t v73 = v4.read();	// L251
      int64_t ins;	// L252
      ins = v73;	// L253
      int64_t v75 = ins;	// L254
      int64_t v76 = v75 & 63;	// L258
      ncfg = v76;	// L259
      int64_t v77 = ins;	// L260
      int64_t v78 = v77 >> 7;	// L264
      int64_t v79 = v78 & 1;	// L268
      bool v80 = v79;	// L269
      nbiased = v80;	// L270
      int64_t v81 = ins;	// L271
      int64_t v82 = v81 >> 8;	// L275
      int64_t v83 = v82 & 1;	// L279
      bool v84 = v83;	// L280
      nfinal = v84;	// L281
      int64_t v85 = ins;	// L282
      int64_t v86 = v85 >> 16;	// L286
      int64_t v87 = v86 & 65535;	// L290
      int32_t v88 = v87;	// L291
      nkbn = v88;	// L292
      int64_t v89 = ins;	// L293
      int64_t v90 = v89 >> 32;	// L297
      int64_t v91 = v90 & 4095;	// L301
      int32_t v92 = v91;	// L302
      nnbn = v92;	// L303
      int64_t v93 = ins;	// L304
      int64_t v94 = v93 >> 44;	// L308
      int64_t v95 = v94 & 255;	// L312
      int32_t v96 = v95;	// L313
      nln = v96;	// L314
      int64_t v97 = ins;	// L315
      int64_t v98 = v97 >> 52;	// L319
      int64_t v99 = v98 & 4095;	// L323
      int32_t v100 = v99;	// L324
      ntn = v100;	// L325
      fetched = 1;	// L329
      int32_t v101 = st;	// L330
      bool v102 = v101 == 0;	// L333
      if (v102) {	// L334
        st = 1;	// L337
      }
    } else {
      int32_t v103 = st;	// L340
      bool v104 = v103 == 1;	// L343
      if (v104) {	// L344
        int64_t v105 = ncfg;	// L345
        cfg = v105;	// L346
        bool v106 = nbiased;	// L347
        if (v106) {	// L352
          int64_t v107 = ncfg;	// L353
          int64_t v108 = v107 | 64;	// L357
          cfg = v108;	// L358
        }
        bool v109 = nfinal;	// L360
        final = v109;	// L361
        bool v110 = nbiased;	// L362
        biased = v110;	// L363
        int32_t v111 = nkbn;	// L364
        kbn = v111;	// L365
        int32_t v112 = nnbn;	// L366
        nbn = v112;	// L367
        int32_t v113 = nln;	// L368
        ln = v113;	// L369
        int32_t v114 = ntn;	// L370
        tleft = v114;	// L371
        tlast = 0;	// L375
        int32_t v115 = ntn;	// L376
        bool v116 = v115 == 0;	// L379
        if (v116) {	// L380
          tlast = 1;	// L384
        }
        int32_t v117 = nkbn;	// L386
        kleft = v117;	// L387
        kfirst = 1;	// L391
        klast = 0;	// L395
        ksingle = 0;	// L399
        int32_t v118 = nkbn;	// L400
        bool v119 = v118 == 0;	// L403
        if (v119) {	// L404
          klast = 1;	// L408
          ksingle = 1;	// L412
        }
        int32_t v120 = nnbn;	// L414
        nleft = v120;	// L415
        nfirst = 1;	// L419
        nlast = 0;	// L423
        nsingle = 0;	// L427
        int32_t v121 = nnbn;	// L428
        bool v122 = v121 == 0;	// L431
        if (v122) {	// L432
          nlast = 1;	// L436
          nsingle = 1;	// L440
        }
        lsingle = 0;	// L445
        int32_t v123 = nln;	// L446
        bool v124 = v123 == 3;	// L449
        if (v124) {	// L450
          lsingle = 1;	// L454
        }
        r = 0;	// L458
        rfirst = 1;	// L462
        int32_t v125 = nln;	// L463
        rl = v125;	// L464
        bool v126 = lsingle;	// L465
        tail = v126;	// L466
        bool v127 = isopen;	// L467
        if (v127) {	// L472
          rl = 3;	// L475
          tail = 1;	// L479
        }
        fetched = 0;	// L484
        st = 2;	// L487
      } else {
        bool blast;	// L492
        blast = 0;	// L493
        int32_t v129 = rl;	// L494
        bool v130 = v129 == 0;	// L497
        if (v130) {	// L498
          blast = 1;	// L502
        }
        bool more;	// L507
        more = 1;	// L508
        bool nextbias;	// L512
        nextbias = 0;	// L513
        bool v133 = isopen;	// L514
        if (v133) {	// L519
          bool v134 = biased;	// L520
          nextbias = v134;	// L521
        } else {
          bool v135 = nlast;	// L523
          int32_t v136 = v135;	// L524
          bool v137 = v136 == 0;	// L527
          if (v137) {	// L528
            bool v138 = kfirst;	// L529
            if (v138) {	// L534
              bool v139 = biased;	// L535
              nextbias = v139;	// L536
            }
          } else {
            bool v140 = klast;	// L539
            if (v140) {	// L544
              bool v141 = tlast;	// L545
              int32_t v142 = v141;	// L546
              bool v143 = v142 == 0;	// L549
              if (v143) {	// L550
                bool v144 = biased;	// L551
                nextbias = v144;	// L552
              } else {
                bool v145 = final;	// L554
                int32_t v146 = v145;	// L555
                bool v147 = v146 == 0;	// L558
                if (v147) {	// L559
                  bool v148 = nbiased;	// L560
                  nextbias = v148;	// L561
                } else {
                  more = 0;	// L566
                }
              }
            }
          }
        }
        bool v149 = tail;	// L572
        bool v150 = more;	// L573
        bool v151 = v149 & v150;	// L574
        bool carry;	// L575
        carry = v151;	// L576
        int8_t e[9];	// L580
        for (int v154 = 0; v154 < 9; v154++) {	// L581
          e[v154] = 0;	// L581
        }
        bool v155 = carry;	// L582
        if (v155) {	// L587
          int8_t v156[4];
          {
            hls::vector< int8_t, 4 > _vec = v6.read();
            for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
              v156[_iv0] = _vec[_iv0];
            }
          }	// L588
          int8_t v157 = v156[0];	// L589
          e[4] = v157;	// L590
          int8_t v158 = v156[1];	// L591
          e[5] = v158;	// L592
          int8_t v159 = v156[2];	// L593
          e[6] = v159;	// L594
          int8_t v160 = v156[3];	// L595
          e[7] = v160;	// L596
        } else {
          e[4] = 0;	// L601
          e[5] = 0;	// L605
          e[6] = 0;	// L609
          e[7] = 0;	// L613
        }
        bool v161 = isopen;	// L615
        if (v161) {	// L620
          e[0] = 0;	// L624
          e[1] = 0;	// L628
          e[2] = 0;	// L632
          e[3] = 0;	// L636
        } else {
          bool v162 = nfirst;	// L638
          if (v162) {	// L643
            int8_t v163[4];
            {
              hls::vector< int8_t, 4 > _vec = v0.read();
              for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
                v163[_iv0] = _vec[_iv0];
              }
            }	// L644
            int8_t v164 = v163[0];	// L645
            int32_t v165 = r;	// L646
            int v166 = v165;	// L647
            ab0[v166] = v164;	// L648
            int8_t v167 = v163[1];	// L649
            int32_t v168 = r;	// L650
            int v169 = v168;	// L651
            ab1[v169] = v167;	// L652
            int8_t v170 = v163[2];	// L653
            int32_t v171 = r;	// L654
            int v172 = v171;	// L655
            ab2[v172] = v170;	// L656
            int8_t v173 = v163[3];	// L657
            int32_t v174 = r;	// L658
            int v175 = v174;	// L659
            ab3[v175] = v173;	// L660
            int8_t v176 = v163[0];	// L661
            e[0] = v176;	// L662
            int8_t v177 = v163[1];	// L663
            e[1] = v177;	// L664
            int8_t v178 = v163[2];	// L665
            e[2] = v178;	// L666
            int8_t v179 = v163[3];	// L667
            e[3] = v179;	// L668
          } else {
            int32_t v180 = r;	// L670
            int v181 = v180;	// L671
            int8_t v182 = ab0[v181];	// L672
            e[0] = v182;	// L673
            int32_t v183 = r;	// L674
            int v184 = v183;	// L675
            int8_t v185 = ab1[v184];	// L676
            e[1] = v185;	// L677
            int32_t v186 = r;	// L678
            int v187 = v186;	// L679
            int8_t v188 = ab2[v187];	// L680
            e[2] = v188;	// L681
            int32_t v189 = r;	// L682
            int v190 = v189;	// L683
            int8_t v191 = ab3[v190];	// L684
            e[3] = v191;	// L685
          }
        }
        int64_t u;	// L691
        u = 0;	// L692
        bool v193 = carry;	// L693
        bool v194 = nextbias;	// L694
        bool v195 = v193 & v194;	// L695
        if (v195) {	// L700
          int64_t v196 = bb;	// L701
          int64_t v197 = v196 >> 32;	// L705
          int64_t imm;	// L706
          imm = v197;	// L707
          int32_t v199 = rl;	// L708
          int32_t v200 = v199 & 1;	// L711
          bool v201 = v200 != 0;	// L714
          if (v201) {	// L715
            int64_t v202 = v1.read();	// L716
            bb = v202;	// L717
            int64_t v203 = bb;	// L718
            int64_t v204 = v203 & 65535;	// L722
            int64_t l0;	// L723
            l0 = v204;	// L724
            int64_t v206 = bb;	// L725
            int64_t v207 = v206 >> 16;	// L729
            int64_t v208 = v207 & 65535;	// L733
            int64_t l1;	// L734
            l1 = v208;	// L735
            int64_t v210 = l1;	// L736
            int64_t v211 = v210 ^ 32768;	// L740
            ap_int<65> v212 = v211;	// L741
            ap_int<65> v213 = v212 - 32768;	// L745
            ap_int<65> v214 = v213 << 16;	// L749
            int64_t v215 = l0;	// L750
            ap_int<65> v216 = v215;	// L751
            ap_int<65> v217 = v214 | v216;	// L752
            int64_t v218 = v217;	// L753
            imm = v218;	// L754
          }
          int32_t v219 = rl;	// L756
          int32_t v220 = v219 & 3;	// L759
          ap_int<33> v221 = v220;	// L763
          ap_int<33> v222 = 3 - v221;	// L764
          int64_t v223 = v222;	// L765
          int64_t q;	// L766
          q = v223;	// L767
          int64_t v225 = imm;	// L768
          int64_t v226 = v225 << 32;	// L772
          int64_t v227 = v226 | 4096;	// L776
          int64_t v228 = q;	// L777
          int64_t v229 = v228 << 13;	// L781
          int64_t v230 = v227 | v229;	// L782
          u = v230;	// L783
        }
        bool v231 = isopen;	// L785
        int32_t v232 = v231;	// L786
        bool v233 = v232 == 0;	// L789
        if (v233) {	// L790
          bool v234 = klast;	// L791
          if (v234) {	// L796
            bool v235 = noroom;	// L797
            if (v235) {	// L802
              int8_t v236 = v2.read();	// L803
              int8_t paid;	// L804
              paid = v236;	// L805
            } else {
              int32_t v238 = space;	// L807
              bool v239 = v238 == 1;	// L810
              if (v239) {	// L811
                noroom = 1;	// L815
              }
              int32_t v240 = space;	// L817
              ap_int<33> v241 = v240;	// L818
              ap_int<33> v242 = v241 - 1;	// L822
              int32_t v243 = v242;	// L823
              space = v243;	// L824
            }
          }
        }
        bool fin;	// L831
        fin = 0;	// L832
        bool v245 = isopen;	// L833
        int32_t v246 = v245;	// L834
        bool v247 = v246 == 0;	// L837
        if (v247) {	// L838
          bool v248 = lastblk;	// L839
          bool v249 = blast;	// L840
          bool v250 = v248 & v249;	// L841
          bool v251 = final;	// L842
          bool v252 = v250 & v251;	// L843
          fin = v252;	// L844
        }
        int32_t fl;	// L848
        fl = 0;	// L849
        bool v254 = blast;	// L850
        if (v254) {	// L855
          fl = 1;	// L858
        }
        bool v255 = fin;	// L860
        if (v255) {	// L865
          int32_t v256 = fl;	// L866
          int32_t v257 = v256 | 2;	// L869
          fl = v257;	// L870
        }
        int32_t v258 = fl;	// L872
        int8_t v259 = v258;	// L873
        e[8] = v259;	// L874
        {
          hls::vector< int8_t, 9 > _vec;
          for (int _iv0 = 0; _iv0 < 9; ++_iv0) {
            _vec[_iv0] = e[_iv0];
          }
          v3.write(_vec);
        }	// L875
        bool v260 = isopen;	// L876
        int32_t v261 = v260;	// L877
        bool v262 = v261 == 0;	// L880
        if (v262) {	// L881
          int64_t v263 = u;	// L882
          int64_t v264 = v263 | 262144;	// L886
          int64_t v265 = cfg;	// L887
          int64_t v266 = v264 | v265;	// L888
          u = v266;	// L889
          bool v267 = kfirst;	// L890
          if (v267) {	// L895
            int64_t v268 = u;	// L896
            int64_t v269 = v268 | 128;	// L900
            u = v269;	// L901
            bool v270 = biased;	// L902
            bool v271 = rfirst;	// L903
            bool v272 = v270 & v271;	// L904
            if (v272) {	// L909
              int64_t v273 = u;	// L910
              int64_t v274 = v273 | 512;	// L914
              u = v274;	// L915
            }
          }
          bool v275 = klast;	// L918
          if (v275) {	// L923
            int64_t v276 = u;	// L924
            int64_t v277 = v276 | 256;	// L928
            u = v277;	// L929
          }
          bool v278 = nlast;	// L931
          bool v279 = blast;	// L932
          bool v280 = v278 & v279;	// L933
          if (v280) {	// L938
            int64_t v281 = u;	// L939
            int64_t v282 = v281 | 2048;	// L943
            u = v282;	// L944
          }
          bool v283 = fin;	// L946
          if (v283) {	// L951
            int64_t v284 = u;	// L952
            int64_t v285 = v284 | 1024;	// L956
            u = v285;	// L957
          }
        }
        int64_t v286 = u;	// L960
        v5.write(v286);	// L961
        bool v287 = blast;	// L962
        int32_t v288 = v287;	// L963
        bool v289 = v288 == 0;	// L966
        if (v289) {	// L967
          bool v290 = isopen;	// L968
          int32_t v291 = v290;	// L969
          bool v292 = v291 == 0;	// L972
          if (v292) {	// L973
            int32_t v293 = rl;	// L974
            bool v294 = v293 == 4;	// L977
            if (v294) {	// L978
              tail = 1;	// L982
            }
            int32_t v295 = r;	// L984
            ap_int<33> v296 = v295;	// L985
            ap_int<33> v297 = v296 + 1;	// L989
            int32_t v298 = v297;	// L990
            r = v298;	// L991
            rfirst = 0;	// L995
          }
          int32_t v299 = rl;	// L997
          ap_int<33> v300 = v299;	// L998
          ap_int<33> v301 = v300 - 1;	// L1002
          int32_t v302 = v301;	// L1003
          rl = v302;	// L1004
        } else {
          int32_t v303 = ln;	// L1006
          rl = v303;	// L1007
          bool v304 = lsingle;	// L1008
          tail = v304;	// L1009
          bool v305 = isopen;	// L1010
          if (v305) {	// L1015
            isopen = 0;	// L1019
          } else {
            r = 0;	// L1023
            rfirst = 1;	// L1027
            bool v306 = nlast;	// L1028
            int32_t v307 = v306;	// L1029
            bool v308 = v307 == 0;	// L1032
            if (v308) {	// L1033
              nfirst = 0;	// L1037
              int32_t v309 = nleft;	// L1038
              bool v310 = v309 == 1;	// L1041
              if (v310) {	// L1042
                nlast = 1;	// L1046
              }
              int32_t v311 = nleft;	// L1048
              ap_int<33> v312 = v311;	// L1049
              ap_int<33> v313 = v312 - 1;	// L1053
              int32_t v314 = v313;	// L1054
              nleft = v314;	// L1055
            } else {
              int32_t v315 = nbn;	// L1057
              nleft = v315;	// L1058
              nfirst = 1;	// L1062
              bool v316 = nsingle;	// L1063
              nlast = v316;	// L1064
              bool v317 = klast;	// L1065
              int32_t v318 = v317;	// L1066
              bool v319 = v318 == 0;	// L1069
              if (v319) {	// L1070
                kfirst = 0;	// L1074
                int32_t v320 = kleft;	// L1075
                bool v321 = v320 == 1;	// L1078
                if (v321) {	// L1079
                  klast = 1;	// L1083
                }
                int32_t v322 = kleft;	// L1085
                ap_int<33> v323 = v322;	// L1086
                ap_int<33> v324 = v323 - 1;	// L1090
                int32_t v325 = v324;	// L1091
                kleft = v325;	// L1092
              } else {
                int32_t v326 = kbn;	// L1094
                kleft = v326;	// L1095
                kfirst = 1;	// L1099
                bool v327 = ksingle;	// L1100
                klast = v327;	// L1101
                bool v328 = tlast;	// L1102
                int32_t v329 = v328;	// L1103
                bool v330 = v329 == 0;	// L1106
                if (v330) {	// L1107
                  int32_t v331 = tleft;	// L1108
                  bool v332 = v331 == 1;	// L1111
                  if (v332) {	// L1112
                    tlast = 1;	// L1116
                  }
                  int32_t v333 = tleft;	// L1118
                  ap_int<33> v334 = v333;	// L1119
                  ap_int<33> v335 = v334 - 1;	// L1123
                  int32_t v336 = v335;	// L1124
                  tleft = v336;	// L1125
                } else {
                  bool v337 = final;	// L1127
                  if (v337) {	// L1132
                    go = 0;	// L1136
                  } else {
                    st = 1;	// L1140
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


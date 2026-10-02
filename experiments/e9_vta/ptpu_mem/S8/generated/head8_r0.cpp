
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
void head8_r0_0(
  hls::stream< hls::vector< int8_t, 8 > >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< hls::vector< int8_t, 17 > >& v3,
  hls::stream< int64_t >& v4,
  hls::stream< int64_t >& v5,
  hls::stream< hls::vector< int8_t, 8 > >& v6
) {	// L2
  static int8_t ab0[64] = {0};	// L6
  static int8_t ab1[64] = {0};	// L11
  static int8_t ab2[64] = {0};	// L16
  static int8_t ab3[64] = {0};	// L21
  static int8_t ab4[64] = {0};	// L26
  static int8_t ab5[64] = {0};	// L31
  static int8_t ab6[64] = {0};	// L36
  static int8_t ab7[64] = {0};	// L41
  int64_t cfg;	// L46
  cfg = 0;	// L47
  bool final;	// L51
  final = 0;	// L52
  bool biased;	// L56
  biased = 0;	// L57
  int32_t kbn;	// L60
  kbn = 0;	// L61
  int32_t nbn;	// L64
  nbn = 0;	// L65
  int32_t ln;	// L68
  ln = 0;	// L69
  bool ksingle;	// L73
  ksingle = 0;	// L74
  bool nsingle;	// L78
  nsingle = 0;	// L79
  bool lsingle;	// L83
  lsingle = 0;	// L84
  int64_t ncfg;	// L88
  ncfg = 0;	// L89
  bool nfinal;	// L93
  nfinal = 0;	// L94
  bool nbiased;	// L98
  nbiased = 0;	// L99
  int32_t nkbn;	// L102
  nkbn = 0;	// L103
  int32_t nnbn;	// L106
  nnbn = 0;	// L107
  int32_t nln;	// L110
  nln = 0;	// L111
  int32_t ntn;	// L114
  ntn = 0;	// L115
  int32_t tleft;	// L118
  tleft = 0;	// L119
  int32_t kleft;	// L122
  kleft = 0;	// L123
  int32_t nleft;	// L126
  nleft = 0;	// L127
  int32_t rl;	// L130
  rl = 0;	// L131
  int32_t r;	// L134
  r = 0;	// L135
  bool tlast;	// L139
  tlast = 0;	// L140
  bool kfirst;	// L144
  kfirst = 0;	// L145
  bool klast;	// L149
  klast = 0;	// L150
  bool nfirst;	// L154
  nfirst = 0;	// L155
  bool nlast;	// L159
  nlast = 0;	// L160
  bool rfirst;	// L164
  rfirst = 0;	// L165
  bool tail;	// L169
  tail = 0;	// L170
  bool isopen;	// L174
  isopen = 1;	// L175
  bool fetched;	// L179
  fetched = 0;	// L180
  int32_t space;	// L183
  space = 1024;	// L184
  bool noroom;	// L188
  noroom = 0;	// L189
  int64_t bb;	// L193
  bb = 0;	// L194
  int32_t st;	// L197
  st = 0;	// L198
  bool go;	// L202
  go = 1;	// L203
  while (true) {	// L204
    #pragma HLS pipeline II=1 style=flp
    bool v58 = go;	// L205
    if (!(v58)) break;
    bool v59 = nlast;	// L212
    bool v60 = klast;	// L213
    bool v61 = v59 & v60;	// L214
    bool v62 = tlast;	// L215
    bool v63 = v61 & v62;	// L216
    bool lastblk;	// L217
    lastblk = v63;	// L218
    bool want;	// L222
    want = 0;	// L223
    int32_t v66 = st;	// L224
    bool v67 = v66 == 0;	// L227
    if (v67) {	// L228
      want = 1;	// L232
    } else {
      int32_t v68 = st;	// L234
      bool v69 = v68 == 2;	// L237
      if (v69) {	// L238
        bool v70 = tail;	// L239
        bool v71 = lastblk;	// L240
        bool v72 = v70 & v71;	// L241
        if (v72) {	// L246
          bool v73 = final;	// L247
          bool v74 = fetched;	// L248
          bool v75 = v73 | v74;	// L249
          bool v76 = isopen;	// L250
          bool v77 = v75 | v76;	// L251
          int32_t v78 = v77;	// L252
          bool v79 = v78 == 0;	// L255
          if (v79) {	// L256
            want = 1;	// L260
          }
        }
      }
    }
    bool v80 = want;	// L265
    if (v80) {	// L270
      int64_t v81 = v4.read();	// L271
      int64_t ins;	// L272
      ins = v81;	// L273
      int64_t v83 = ins;	// L274
      int64_t v84 = v83 & 63;	// L278
      ncfg = v84;	// L279
      int64_t v85 = ins;	// L280
      int64_t v86 = v85 >> 7;	// L284
      int64_t v87 = v86 & 1;	// L288
      bool v88 = v87;	// L289
      nbiased = v88;	// L290
      int64_t v89 = ins;	// L291
      int64_t v90 = v89 >> 8;	// L295
      int64_t v91 = v90 & 1;	// L299
      bool v92 = v91;	// L300
      nfinal = v92;	// L301
      int64_t v93 = ins;	// L302
      int64_t v94 = v93 >> 16;	// L306
      int64_t v95 = v94 & 65535;	// L310
      int32_t v96 = v95;	// L311
      nkbn = v96;	// L312
      int64_t v97 = ins;	// L313
      int64_t v98 = v97 >> 32;	// L317
      int64_t v99 = v98 & 4095;	// L321
      int32_t v100 = v99;	// L322
      nnbn = v100;	// L323
      int64_t v101 = ins;	// L324
      int64_t v102 = v101 >> 44;	// L328
      int64_t v103 = v102 & 255;	// L332
      int32_t v104 = v103;	// L333
      nln = v104;	// L334
      int64_t v105 = ins;	// L335
      int64_t v106 = v105 >> 52;	// L339
      int64_t v107 = v106 & 4095;	// L343
      int32_t v108 = v107;	// L344
      ntn = v108;	// L345
      fetched = 1;	// L349
      int32_t v109 = st;	// L350
      bool v110 = v109 == 0;	// L353
      if (v110) {	// L354
        st = 1;	// L357
      }
    } else {
      int32_t v111 = st;	// L360
      bool v112 = v111 == 1;	// L363
      if (v112) {	// L364
        int64_t v113 = ncfg;	// L365
        cfg = v113;	// L366
        bool v114 = nbiased;	// L367
        if (v114) {	// L372
          int64_t v115 = ncfg;	// L373
          int64_t v116 = v115 | 64;	// L377
          cfg = v116;	// L378
        }
        bool v117 = nfinal;	// L380
        final = v117;	// L381
        bool v118 = nbiased;	// L382
        biased = v118;	// L383
        int32_t v119 = nkbn;	// L384
        kbn = v119;	// L385
        int32_t v120 = nnbn;	// L386
        nbn = v120;	// L387
        int32_t v121 = nln;	// L388
        ln = v121;	// L389
        int32_t v122 = ntn;	// L390
        tleft = v122;	// L391
        tlast = 0;	// L395
        int32_t v123 = ntn;	// L396
        bool v124 = v123 == 0;	// L399
        if (v124) {	// L400
          tlast = 1;	// L404
        }
        int32_t v125 = nkbn;	// L406
        kleft = v125;	// L407
        kfirst = 1;	// L411
        klast = 0;	// L415
        ksingle = 0;	// L419
        int32_t v126 = nkbn;	// L420
        bool v127 = v126 == 0;	// L423
        if (v127) {	// L424
          klast = 1;	// L428
          ksingle = 1;	// L432
        }
        int32_t v128 = nnbn;	// L434
        nleft = v128;	// L435
        nfirst = 1;	// L439
        nlast = 0;	// L443
        nsingle = 0;	// L447
        int32_t v129 = nnbn;	// L448
        bool v130 = v129 == 0;	// L451
        if (v130) {	// L452
          nlast = 1;	// L456
          nsingle = 1;	// L460
        }
        lsingle = 0;	// L465
        int32_t v131 = nln;	// L466
        bool v132 = v131 == 7;	// L469
        if (v132) {	// L470
          lsingle = 1;	// L474
        }
        r = 0;	// L478
        rfirst = 1;	// L482
        int32_t v133 = nln;	// L483
        rl = v133;	// L484
        bool v134 = lsingle;	// L485
        tail = v134;	// L486
        bool v135 = isopen;	// L487
        if (v135) {	// L492
          rl = 7;	// L495
          tail = 1;	// L499
        }
        fetched = 0;	// L504
        st = 2;	// L507
      } else {
        bool blast;	// L512
        blast = 0;	// L513
        int32_t v137 = rl;	// L514
        bool v138 = v137 == 0;	// L517
        if (v138) {	// L518
          blast = 1;	// L522
        }
        bool more;	// L527
        more = 1;	// L528
        bool nextbias;	// L532
        nextbias = 0;	// L533
        bool v141 = isopen;	// L534
        if (v141) {	// L539
          bool v142 = biased;	// L540
          nextbias = v142;	// L541
        } else {
          bool v143 = nlast;	// L543
          int32_t v144 = v143;	// L544
          bool v145 = v144 == 0;	// L547
          if (v145) {	// L548
            bool v146 = kfirst;	// L549
            if (v146) {	// L554
              bool v147 = biased;	// L555
              nextbias = v147;	// L556
            }
          } else {
            bool v148 = klast;	// L559
            if (v148) {	// L564
              bool v149 = tlast;	// L565
              int32_t v150 = v149;	// L566
              bool v151 = v150 == 0;	// L569
              if (v151) {	// L570
                bool v152 = biased;	// L571
                nextbias = v152;	// L572
              } else {
                bool v153 = final;	// L574
                int32_t v154 = v153;	// L575
                bool v155 = v154 == 0;	// L578
                if (v155) {	// L579
                  bool v156 = nbiased;	// L580
                  nextbias = v156;	// L581
                } else {
                  more = 0;	// L586
                }
              }
            }
          }
        }
        bool v157 = tail;	// L592
        bool v158 = more;	// L593
        bool v159 = v157 & v158;	// L594
        bool carry;	// L595
        carry = v159;	// L596
        int8_t e[17];	// L600
        for (int v162 = 0; v162 < 17; v162++) {	// L601
          e[v162] = 0;	// L601
        }
        bool v163 = carry;	// L602
        if (v163) {	// L607
          int8_t v164[8];
          {
            hls::vector< int8_t, 8 > _vec = v6.read();
            for (int _iv0 = 0; _iv0 < 8; ++_iv0) {
              v164[_iv0] = _vec[_iv0];
            }
          }	// L608
          int8_t v165 = v164[0];	// L609
          e[8] = v165;	// L610
          int8_t v166 = v164[1];	// L611
          e[9] = v166;	// L612
          int8_t v167 = v164[2];	// L613
          e[10] = v167;	// L614
          int8_t v168 = v164[3];	// L615
          e[11] = v168;	// L616
          int8_t v169 = v164[4];	// L617
          e[12] = v169;	// L618
          int8_t v170 = v164[5];	// L619
          e[13] = v170;	// L620
          int8_t v171 = v164[6];	// L621
          e[14] = v171;	// L622
          int8_t v172 = v164[7];	// L623
          e[15] = v172;	// L624
        } else {
          e[8] = 0;	// L629
          e[9] = 0;	// L633
          e[10] = 0;	// L637
          e[11] = 0;	// L641
          e[12] = 0;	// L645
          e[13] = 0;	// L649
          e[14] = 0;	// L653
          e[15] = 0;	// L657
        }
        bool v173 = isopen;	// L659
        if (v173) {	// L664
          e[0] = 0;	// L668
          e[1] = 0;	// L672
          e[2] = 0;	// L676
          e[3] = 0;	// L680
          e[4] = 0;	// L684
          e[5] = 0;	// L688
          e[6] = 0;	// L692
          e[7] = 0;	// L696
        } else {
          bool v174 = nfirst;	// L698
          if (v174) {	// L703
            int8_t v175[8];
            {
              hls::vector< int8_t, 8 > _vec = v0.read();
              for (int _iv0 = 0; _iv0 < 8; ++_iv0) {
                v175[_iv0] = _vec[_iv0];
              }
            }	// L704
            int8_t v176 = v175[0];	// L705
            int32_t v177 = r;	// L706
            int v178 = v177;	// L707
            ab0[v178] = v176;	// L708
            int8_t v179 = v175[1];	// L709
            int32_t v180 = r;	// L710
            int v181 = v180;	// L711
            ab1[v181] = v179;	// L712
            int8_t v182 = v175[2];	// L713
            int32_t v183 = r;	// L714
            int v184 = v183;	// L715
            ab2[v184] = v182;	// L716
            int8_t v185 = v175[3];	// L717
            int32_t v186 = r;	// L718
            int v187 = v186;	// L719
            ab3[v187] = v185;	// L720
            int8_t v188 = v175[4];	// L721
            int32_t v189 = r;	// L722
            int v190 = v189;	// L723
            ab4[v190] = v188;	// L724
            int8_t v191 = v175[5];	// L725
            int32_t v192 = r;	// L726
            int v193 = v192;	// L727
            ab5[v193] = v191;	// L728
            int8_t v194 = v175[6];	// L729
            int32_t v195 = r;	// L730
            int v196 = v195;	// L731
            ab6[v196] = v194;	// L732
            int8_t v197 = v175[7];	// L733
            int32_t v198 = r;	// L734
            int v199 = v198;	// L735
            ab7[v199] = v197;	// L736
            int8_t v200 = v175[0];	// L737
            e[0] = v200;	// L738
            int8_t v201 = v175[1];	// L739
            e[1] = v201;	// L740
            int8_t v202 = v175[2];	// L741
            e[2] = v202;	// L742
            int8_t v203 = v175[3];	// L743
            e[3] = v203;	// L744
            int8_t v204 = v175[4];	// L745
            e[4] = v204;	// L746
            int8_t v205 = v175[5];	// L747
            e[5] = v205;	// L748
            int8_t v206 = v175[6];	// L749
            e[6] = v206;	// L750
            int8_t v207 = v175[7];	// L751
            e[7] = v207;	// L752
          } else {
            int32_t v208 = r;	// L754
            int v209 = v208;	// L755
            int8_t v210 = ab0[v209];	// L756
            e[0] = v210;	// L757
            int32_t v211 = r;	// L758
            int v212 = v211;	// L759
            int8_t v213 = ab1[v212];	// L760
            e[1] = v213;	// L761
            int32_t v214 = r;	// L762
            int v215 = v214;	// L763
            int8_t v216 = ab2[v215];	// L764
            e[2] = v216;	// L765
            int32_t v217 = r;	// L766
            int v218 = v217;	// L767
            int8_t v219 = ab3[v218];	// L768
            e[3] = v219;	// L769
            int32_t v220 = r;	// L770
            int v221 = v220;	// L771
            int8_t v222 = ab4[v221];	// L772
            e[4] = v222;	// L773
            int32_t v223 = r;	// L774
            int v224 = v223;	// L775
            int8_t v225 = ab5[v224];	// L776
            e[5] = v225;	// L777
            int32_t v226 = r;	// L778
            int v227 = v226;	// L779
            int8_t v228 = ab6[v227];	// L780
            e[6] = v228;	// L781
            int32_t v229 = r;	// L782
            int v230 = v229;	// L783
            int8_t v231 = ab7[v230];	// L784
            e[7] = v231;	// L785
          }
        }
        int64_t u;	// L791
        u = 0;	// L792
        bool v233 = carry;	// L793
        bool v234 = nextbias;	// L794
        bool v235 = v233 & v234;	// L795
        if (v235) {	// L800
          int64_t v236 = bb;	// L801
          int64_t v237 = v236 >> 32;	// L805
          int64_t imm;	// L806
          imm = v237;	// L807
          int32_t v239 = rl;	// L808
          int32_t v240 = v239 & 1;	// L811
          bool v241 = v240 != 0;	// L814
          if (v241) {	// L815
            int64_t v242 = v1.read();	// L816
            bb = v242;	// L817
            int64_t v243 = bb;	// L818
            int64_t v244 = v243 & 65535;	// L822
            int64_t l0;	// L823
            l0 = v244;	// L824
            int64_t v246 = bb;	// L825
            int64_t v247 = v246 >> 16;	// L829
            int64_t v248 = v247 & 65535;	// L833
            int64_t l1;	// L834
            l1 = v248;	// L835
            int64_t v250 = l1;	// L836
            int64_t v251 = v250 ^ 32768;	// L840
            ap_int<65> v252 = v251;	// L841
            ap_int<65> v253 = v252 - 32768;	// L845
            ap_int<65> v254 = v253 << 16;	// L849
            int64_t v255 = l0;	// L850
            ap_int<65> v256 = v255;	// L851
            ap_int<65> v257 = v254 | v256;	// L852
            int64_t v258 = v257;	// L853
            imm = v258;	// L854
          }
          int32_t v259 = rl;	// L856
          int32_t v260 = v259 & 7;	// L859
          ap_int<33> v261 = v260;	// L863
          ap_int<33> v262 = 7 - v261;	// L864
          int64_t v263 = v262;	// L865
          int64_t q;	// L866
          q = v263;	// L867
          int64_t v265 = imm;	// L868
          int64_t v266 = v265 << 32;	// L872
          int64_t v267 = v266 | 4096;	// L876
          int64_t v268 = q;	// L877
          int64_t v269 = v268 << 13;	// L881
          int64_t v270 = v267 | v269;	// L882
          u = v270;	// L883
        }
        bool v271 = isopen;	// L885
        int32_t v272 = v271;	// L886
        bool v273 = v272 == 0;	// L889
        if (v273) {	// L890
          bool v274 = klast;	// L891
          if (v274) {	// L896
            bool v275 = noroom;	// L897
            if (v275) {	// L902
              int8_t v276 = v2.read();	// L903
              int8_t paid;	// L904
              paid = v276;	// L905
            } else {
              int32_t v278 = space;	// L907
              bool v279 = v278 == 1;	// L910
              if (v279) {	// L911
                noroom = 1;	// L915
              }
              int32_t v280 = space;	// L917
              ap_int<33> v281 = v280;	// L918
              ap_int<33> v282 = v281 - 1;	// L922
              int32_t v283 = v282;	// L923
              space = v283;	// L924
            }
          }
        }
        bool fin;	// L931
        fin = 0;	// L932
        bool v285 = isopen;	// L933
        int32_t v286 = v285;	// L934
        bool v287 = v286 == 0;	// L937
        if (v287) {	// L938
          bool v288 = lastblk;	// L939
          bool v289 = blast;	// L940
          bool v290 = v288 & v289;	// L941
          bool v291 = final;	// L942
          bool v292 = v290 & v291;	// L943
          fin = v292;	// L944
        }
        int32_t fl;	// L948
        fl = 0;	// L949
        bool v294 = blast;	// L950
        if (v294) {	// L955
          fl = 1;	// L958
        }
        bool v295 = fin;	// L960
        if (v295) {	// L965
          int32_t v296 = fl;	// L966
          int32_t v297 = v296 | 2;	// L969
          fl = v297;	// L970
        }
        int32_t v298 = fl;	// L972
        int8_t v299 = v298;	// L973
        e[16] = v299;	// L974
        {
          hls::vector< int8_t, 17 > _vec;
          for (int _iv0 = 0; _iv0 < 17; ++_iv0) {
            _vec[_iv0] = e[_iv0];
          }
          v3.write(_vec);
        }	// L975
        bool v300 = isopen;	// L976
        int32_t v301 = v300;	// L977
        bool v302 = v301 == 0;	// L980
        if (v302) {	// L981
          int64_t v303 = u;	// L982
          int64_t v304 = v303 | 262144;	// L986
          int64_t v305 = cfg;	// L987
          int64_t v306 = v304 | v305;	// L988
          u = v306;	// L989
          bool v307 = kfirst;	// L990
          if (v307) {	// L995
            int64_t v308 = u;	// L996
            int64_t v309 = v308 | 128;	// L1000
            u = v309;	// L1001
            bool v310 = biased;	// L1002
            bool v311 = rfirst;	// L1003
            bool v312 = v310 & v311;	// L1004
            if (v312) {	// L1009
              int64_t v313 = u;	// L1010
              int64_t v314 = v313 | 512;	// L1014
              u = v314;	// L1015
            }
          }
          bool v315 = klast;	// L1018
          if (v315) {	// L1023
            int64_t v316 = u;	// L1024
            int64_t v317 = v316 | 256;	// L1028
            u = v317;	// L1029
          }
          bool v318 = nlast;	// L1031
          bool v319 = blast;	// L1032
          bool v320 = v318 & v319;	// L1033
          if (v320) {	// L1038
            int64_t v321 = u;	// L1039
            int64_t v322 = v321 | 2048;	// L1043
            u = v322;	// L1044
          }
          bool v323 = fin;	// L1046
          if (v323) {	// L1051
            int64_t v324 = u;	// L1052
            int64_t v325 = v324 | 1024;	// L1056
            u = v325;	// L1057
          }
        }
        int64_t v326 = u;	// L1060
        v5.write(v326);	// L1061
        bool v327 = blast;	// L1062
        int32_t v328 = v327;	// L1063
        bool v329 = v328 == 0;	// L1066
        if (v329) {	// L1067
          bool v330 = isopen;	// L1068
          int32_t v331 = v330;	// L1069
          bool v332 = v331 == 0;	// L1072
          if (v332) {	// L1073
            int32_t v333 = rl;	// L1074
            bool v334 = v333 == 8;	// L1077
            if (v334) {	// L1078
              tail = 1;	// L1082
            }
            int32_t v335 = r;	// L1084
            ap_int<33> v336 = v335;	// L1085
            ap_int<33> v337 = v336 + 1;	// L1089
            int32_t v338 = v337;	// L1090
            r = v338;	// L1091
            rfirst = 0;	// L1095
          }
          int32_t v339 = rl;	// L1097
          ap_int<33> v340 = v339;	// L1098
          ap_int<33> v341 = v340 - 1;	// L1102
          int32_t v342 = v341;	// L1103
          rl = v342;	// L1104
        } else {
          int32_t v343 = ln;	// L1106
          rl = v343;	// L1107
          bool v344 = lsingle;	// L1108
          tail = v344;	// L1109
          bool v345 = isopen;	// L1110
          if (v345) {	// L1115
            isopen = 0;	// L1119
          } else {
            r = 0;	// L1123
            rfirst = 1;	// L1127
            bool v346 = nlast;	// L1128
            int32_t v347 = v346;	// L1129
            bool v348 = v347 == 0;	// L1132
            if (v348) {	// L1133
              nfirst = 0;	// L1137
              int32_t v349 = nleft;	// L1138
              bool v350 = v349 == 1;	// L1141
              if (v350) {	// L1142
                nlast = 1;	// L1146
              }
              int32_t v351 = nleft;	// L1148
              ap_int<33> v352 = v351;	// L1149
              ap_int<33> v353 = v352 - 1;	// L1153
              int32_t v354 = v353;	// L1154
              nleft = v354;	// L1155
            } else {
              int32_t v355 = nbn;	// L1157
              nleft = v355;	// L1158
              nfirst = 1;	// L1162
              bool v356 = nsingle;	// L1163
              nlast = v356;	// L1164
              bool v357 = klast;	// L1165
              int32_t v358 = v357;	// L1166
              bool v359 = v358 == 0;	// L1169
              if (v359) {	// L1170
                kfirst = 0;	// L1174
                int32_t v360 = kleft;	// L1175
                bool v361 = v360 == 1;	// L1178
                if (v361) {	// L1179
                  klast = 1;	// L1183
                }
                int32_t v362 = kleft;	// L1185
                ap_int<33> v363 = v362;	// L1186
                ap_int<33> v364 = v363 - 1;	// L1190
                int32_t v365 = v364;	// L1191
                kleft = v365;	// L1192
              } else {
                int32_t v366 = kbn;	// L1194
                kleft = v366;	// L1195
                kfirst = 1;	// L1199
                bool v367 = ksingle;	// L1200
                klast = v367;	// L1201
                bool v368 = tlast;	// L1202
                int32_t v369 = v368;	// L1203
                bool v370 = v369 == 0;	// L1206
                if (v370) {	// L1207
                  int32_t v371 = tleft;	// L1208
                  bool v372 = v371 == 1;	// L1211
                  if (v372) {	// L1212
                    tlast = 1;	// L1216
                  }
                  int32_t v373 = tleft;	// L1218
                  ap_int<33> v374 = v373;	// L1219
                  ap_int<33> v375 = v374 - 1;	// L1223
                  int32_t v376 = v375;	// L1224
                  tleft = v376;	// L1225
                } else {
                  bool v377 = final;	// L1227
                  if (v377) {	// L1232
                    go = 0;	// L1236
                  } else {
                    st = 1;	// L1240
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


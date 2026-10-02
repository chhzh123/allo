
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
void head16_r0_0(
  hls::stream< hls::vector< int8_t, 16 > >& v0,
  hls::stream< int64_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< hls::vector< int8_t, 33 > >& v3,
  hls::stream< int64_t >& v4,
  hls::stream< int64_t >& v5,
  hls::stream< hls::vector< int8_t, 16 > >& v6
) {	// L2
  static int8_t ab0[64] = {0};	// L6
  static int8_t ab1[64] = {0};	// L11
  static int8_t ab2[64] = {0};	// L16
  static int8_t ab3[64] = {0};	// L21
  static int8_t ab4[64] = {0};	// L26
  static int8_t ab5[64] = {0};	// L31
  static int8_t ab6[64] = {0};	// L36
  static int8_t ab7[64] = {0};	// L41
  static int8_t ab8[64] = {0};	// L46
  static int8_t ab9[64] = {0};	// L51
  static int8_t ab10[64] = {0};	// L56
  static int8_t ab11[64] = {0};	// L61
  static int8_t ab12[64] = {0};	// L66
  static int8_t ab13[64] = {0};	// L71
  static int8_t ab14[64] = {0};	// L76
  static int8_t ab15[64] = {0};	// L81
  int64_t cfg;	// L86
  cfg = 0;	// L87
  bool final;	// L91
  final = 0;	// L92
  bool biased;	// L96
  biased = 0;	// L97
  int32_t kbn;	// L100
  kbn = 0;	// L101
  int32_t nbn;	// L104
  nbn = 0;	// L105
  int32_t ln;	// L108
  ln = 0;	// L109
  bool ksingle;	// L113
  ksingle = 0;	// L114
  bool nsingle;	// L118
  nsingle = 0;	// L119
  bool lsingle;	// L123
  lsingle = 0;	// L124
  int64_t ncfg;	// L128
  ncfg = 0;	// L129
  bool nfinal;	// L133
  nfinal = 0;	// L134
  bool nbiased;	// L138
  nbiased = 0;	// L139
  int32_t nkbn;	// L142
  nkbn = 0;	// L143
  int32_t nnbn;	// L146
  nnbn = 0;	// L147
  int32_t nln;	// L150
  nln = 0;	// L151
  int32_t ntn;	// L154
  ntn = 0;	// L155
  int32_t tleft;	// L158
  tleft = 0;	// L159
  int32_t kleft;	// L162
  kleft = 0;	// L163
  int32_t nleft;	// L166
  nleft = 0;	// L167
  int32_t rl;	// L170
  rl = 0;	// L171
  int32_t r;	// L174
  r = 0;	// L175
  bool tlast;	// L179
  tlast = 0;	// L180
  bool kfirst;	// L184
  kfirst = 0;	// L185
  bool klast;	// L189
  klast = 0;	// L190
  bool nfirst;	// L194
  nfirst = 0;	// L195
  bool nlast;	// L199
  nlast = 0;	// L200
  bool rfirst;	// L204
  rfirst = 0;	// L205
  bool tail;	// L209
  tail = 0;	// L210
  bool isopen;	// L214
  isopen = 1;	// L215
  bool fetched;	// L219
  fetched = 0;	// L220
  int32_t space;	// L223
  space = 512;	// L224
  bool noroom;	// L228
  noroom = 0;	// L229
  int64_t bb;	// L233
  bb = 0;	// L234
  int32_t st;	// L237
  st = 0;	// L238
  bool go;	// L242
  go = 1;	// L243
  while (true) {	// L244
    #pragma HLS pipeline II=1 style=flp
    bool v74 = go;	// L245
    if (!(v74)) break;
    bool v75 = nlast;	// L252
    bool v76 = klast;	// L253
    bool v77 = v75 & v76;	// L254
    bool v78 = tlast;	// L255
    bool v79 = v77 & v78;	// L256
    bool lastblk;	// L257
    lastblk = v79;	// L258
    bool want;	// L262
    want = 0;	// L263
    int32_t v82 = st;	// L264
    bool v83 = v82 == 0;	// L267
    if (v83) {	// L268
      want = 1;	// L272
    } else {
      int32_t v84 = st;	// L274
      bool v85 = v84 == 2;	// L277
      if (v85) {	// L278
        bool v86 = tail;	// L279
        bool v87 = lastblk;	// L280
        bool v88 = v86 & v87;	// L281
        if (v88) {	// L286
          bool v89 = final;	// L287
          bool v90 = fetched;	// L288
          bool v91 = v89 | v90;	// L289
          bool v92 = isopen;	// L290
          bool v93 = v91 | v92;	// L291
          int32_t v94 = v93;	// L292
          bool v95 = v94 == 0;	// L295
          if (v95) {	// L296
            want = 1;	// L300
          }
        }
      }
    }
    bool v96 = want;	// L305
    if (v96) {	// L310
      int64_t v97 = v4.read();	// L311
      int64_t ins;	// L312
      ins = v97;	// L313
      int64_t v99 = ins;	// L314
      int64_t v100 = v99 & 63;	// L318
      ncfg = v100;	// L319
      int64_t v101 = ins;	// L320
      int64_t v102 = v101 >> 7;	// L324
      int64_t v103 = v102 & 1;	// L328
      bool v104 = v103;	// L329
      nbiased = v104;	// L330
      int64_t v105 = ins;	// L331
      int64_t v106 = v105 >> 8;	// L335
      int64_t v107 = v106 & 1;	// L339
      bool v108 = v107;	// L340
      nfinal = v108;	// L341
      int64_t v109 = ins;	// L342
      int64_t v110 = v109 >> 16;	// L346
      int64_t v111 = v110 & 65535;	// L350
      int32_t v112 = v111;	// L351
      nkbn = v112;	// L352
      int64_t v113 = ins;	// L353
      int64_t v114 = v113 >> 32;	// L357
      int64_t v115 = v114 & 4095;	// L361
      int32_t v116 = v115;	// L362
      nnbn = v116;	// L363
      int64_t v117 = ins;	// L364
      int64_t v118 = v117 >> 44;	// L368
      int64_t v119 = v118 & 255;	// L372
      int32_t v120 = v119;	// L373
      nln = v120;	// L374
      int64_t v121 = ins;	// L375
      int64_t v122 = v121 >> 52;	// L379
      int64_t v123 = v122 & 4095;	// L383
      int32_t v124 = v123;	// L384
      ntn = v124;	// L385
      fetched = 1;	// L389
      int32_t v125 = st;	// L390
      bool v126 = v125 == 0;	// L393
      if (v126) {	// L394
        st = 1;	// L397
      }
    } else {
      int32_t v127 = st;	// L400
      bool v128 = v127 == 1;	// L403
      if (v128) {	// L404
        int64_t v129 = ncfg;	// L405
        cfg = v129;	// L406
        bool v130 = nbiased;	// L407
        if (v130) {	// L412
          int64_t v131 = ncfg;	// L413
          int64_t v132 = v131 | 64;	// L417
          cfg = v132;	// L418
        }
        bool v133 = nfinal;	// L420
        final = v133;	// L421
        bool v134 = nbiased;	// L422
        biased = v134;	// L423
        int32_t v135 = nkbn;	// L424
        kbn = v135;	// L425
        int32_t v136 = nnbn;	// L426
        nbn = v136;	// L427
        int32_t v137 = nln;	// L428
        ln = v137;	// L429
        int32_t v138 = ntn;	// L430
        tleft = v138;	// L431
        tlast = 0;	// L435
        int32_t v139 = ntn;	// L436
        bool v140 = v139 == 0;	// L439
        if (v140) {	// L440
          tlast = 1;	// L444
        }
        int32_t v141 = nkbn;	// L446
        kleft = v141;	// L447
        kfirst = 1;	// L451
        klast = 0;	// L455
        ksingle = 0;	// L459
        int32_t v142 = nkbn;	// L460
        bool v143 = v142 == 0;	// L463
        if (v143) {	// L464
          klast = 1;	// L468
          ksingle = 1;	// L472
        }
        int32_t v144 = nnbn;	// L474
        nleft = v144;	// L475
        nfirst = 1;	// L479
        nlast = 0;	// L483
        nsingle = 0;	// L487
        int32_t v145 = nnbn;	// L488
        bool v146 = v145 == 0;	// L491
        if (v146) {	// L492
          nlast = 1;	// L496
          nsingle = 1;	// L500
        }
        lsingle = 0;	// L505
        int32_t v147 = nln;	// L506
        bool v148 = v147 == 15;	// L509
        if (v148) {	// L510
          lsingle = 1;	// L514
        }
        r = 0;	// L518
        rfirst = 1;	// L522
        int32_t v149 = nln;	// L523
        rl = v149;	// L524
        bool v150 = lsingle;	// L525
        tail = v150;	// L526
        bool v151 = isopen;	// L527
        if (v151) {	// L532
          rl = 15;	// L535
          tail = 1;	// L539
        }
        fetched = 0;	// L544
        st = 2;	// L547
      } else {
        bool blast;	// L552
        blast = 0;	// L553
        int32_t v153 = rl;	// L554
        bool v154 = v153 == 0;	// L557
        if (v154) {	// L558
          blast = 1;	// L562
        }
        bool more;	// L567
        more = 1;	// L568
        bool nextbias;	// L572
        nextbias = 0;	// L573
        bool v157 = isopen;	// L574
        if (v157) {	// L579
          bool v158 = biased;	// L580
          nextbias = v158;	// L581
        } else {
          bool v159 = nlast;	// L583
          int32_t v160 = v159;	// L584
          bool v161 = v160 == 0;	// L587
          if (v161) {	// L588
            bool v162 = kfirst;	// L589
            if (v162) {	// L594
              bool v163 = biased;	// L595
              nextbias = v163;	// L596
            }
          } else {
            bool v164 = klast;	// L599
            if (v164) {	// L604
              bool v165 = tlast;	// L605
              int32_t v166 = v165;	// L606
              bool v167 = v166 == 0;	// L609
              if (v167) {	// L610
                bool v168 = biased;	// L611
                nextbias = v168;	// L612
              } else {
                bool v169 = final;	// L614
                int32_t v170 = v169;	// L615
                bool v171 = v170 == 0;	// L618
                if (v171) {	// L619
                  bool v172 = nbiased;	// L620
                  nextbias = v172;	// L621
                } else {
                  more = 0;	// L626
                }
              }
            }
          }
        }
        bool v173 = tail;	// L632
        bool v174 = more;	// L633
        bool v175 = v173 & v174;	// L634
        bool carry;	// L635
        carry = v175;	// L636
        int8_t e[33];	// L640
        for (int v178 = 0; v178 < 33; v178++) {	// L641
          e[v178] = 0;	// L641
        }
        bool v179 = carry;	// L642
        if (v179) {	// L647
          int8_t v180[16];
          {
            hls::vector< int8_t, 16 > _vec = v6.read();
            for (int _iv0 = 0; _iv0 < 16; ++_iv0) {
              v180[_iv0] = _vec[_iv0];
            }
          }	// L648
          int8_t v181 = v180[0];	// L649
          e[16] = v181;	// L650
          int8_t v182 = v180[1];	// L651
          e[17] = v182;	// L652
          int8_t v183 = v180[2];	// L653
          e[18] = v183;	// L654
          int8_t v184 = v180[3];	// L655
          e[19] = v184;	// L656
          int8_t v185 = v180[4];	// L657
          e[20] = v185;	// L658
          int8_t v186 = v180[5];	// L659
          e[21] = v186;	// L660
          int8_t v187 = v180[6];	// L661
          e[22] = v187;	// L662
          int8_t v188 = v180[7];	// L663
          e[23] = v188;	// L664
          int8_t v189 = v180[8];	// L665
          e[24] = v189;	// L666
          int8_t v190 = v180[9];	// L667
          e[25] = v190;	// L668
          int8_t v191 = v180[10];	// L669
          e[26] = v191;	// L670
          int8_t v192 = v180[11];	// L671
          e[27] = v192;	// L672
          int8_t v193 = v180[12];	// L673
          e[28] = v193;	// L674
          int8_t v194 = v180[13];	// L675
          e[29] = v194;	// L676
          int8_t v195 = v180[14];	// L677
          e[30] = v195;	// L678
          int8_t v196 = v180[15];	// L679
          e[31] = v196;	// L680
        } else {
          e[16] = 0;	// L685
          e[17] = 0;	// L689
          e[18] = 0;	// L693
          e[19] = 0;	// L697
          e[20] = 0;	// L701
          e[21] = 0;	// L705
          e[22] = 0;	// L709
          e[23] = 0;	// L713
          e[24] = 0;	// L717
          e[25] = 0;	// L721
          e[26] = 0;	// L725
          e[27] = 0;	// L729
          e[28] = 0;	// L733
          e[29] = 0;	// L737
          e[30] = 0;	// L741
          e[31] = 0;	// L745
        }
        bool v197 = isopen;	// L747
        if (v197) {	// L752
          e[0] = 0;	// L756
          e[1] = 0;	// L760
          e[2] = 0;	// L764
          e[3] = 0;	// L768
          e[4] = 0;	// L772
          e[5] = 0;	// L776
          e[6] = 0;	// L780
          e[7] = 0;	// L784
          e[8] = 0;	// L788
          e[9] = 0;	// L792
          e[10] = 0;	// L796
          e[11] = 0;	// L800
          e[12] = 0;	// L804
          e[13] = 0;	// L808
          e[14] = 0;	// L812
          e[15] = 0;	// L816
        } else {
          bool v198 = nfirst;	// L818
          if (v198) {	// L823
            int8_t v199[16];
            {
              hls::vector< int8_t, 16 > _vec = v0.read();
              for (int _iv0 = 0; _iv0 < 16; ++_iv0) {
                v199[_iv0] = _vec[_iv0];
              }
            }	// L824
            int8_t v200 = v199[0];	// L825
            int32_t v201 = r;	// L826
            int v202 = v201;	// L827
            ab0[v202] = v200;	// L828
            int8_t v203 = v199[1];	// L829
            int32_t v204 = r;	// L830
            int v205 = v204;	// L831
            ab1[v205] = v203;	// L832
            int8_t v206 = v199[2];	// L833
            int32_t v207 = r;	// L834
            int v208 = v207;	// L835
            ab2[v208] = v206;	// L836
            int8_t v209 = v199[3];	// L837
            int32_t v210 = r;	// L838
            int v211 = v210;	// L839
            ab3[v211] = v209;	// L840
            int8_t v212 = v199[4];	// L841
            int32_t v213 = r;	// L842
            int v214 = v213;	// L843
            ab4[v214] = v212;	// L844
            int8_t v215 = v199[5];	// L845
            int32_t v216 = r;	// L846
            int v217 = v216;	// L847
            ab5[v217] = v215;	// L848
            int8_t v218 = v199[6];	// L849
            int32_t v219 = r;	// L850
            int v220 = v219;	// L851
            ab6[v220] = v218;	// L852
            int8_t v221 = v199[7];	// L853
            int32_t v222 = r;	// L854
            int v223 = v222;	// L855
            ab7[v223] = v221;	// L856
            int8_t v224 = v199[8];	// L857
            int32_t v225 = r;	// L858
            int v226 = v225;	// L859
            ab8[v226] = v224;	// L860
            int8_t v227 = v199[9];	// L861
            int32_t v228 = r;	// L862
            int v229 = v228;	// L863
            ab9[v229] = v227;	// L864
            int8_t v230 = v199[10];	// L865
            int32_t v231 = r;	// L866
            int v232 = v231;	// L867
            ab10[v232] = v230;	// L868
            int8_t v233 = v199[11];	// L869
            int32_t v234 = r;	// L870
            int v235 = v234;	// L871
            ab11[v235] = v233;	// L872
            int8_t v236 = v199[12];	// L873
            int32_t v237 = r;	// L874
            int v238 = v237;	// L875
            ab12[v238] = v236;	// L876
            int8_t v239 = v199[13];	// L877
            int32_t v240 = r;	// L878
            int v241 = v240;	// L879
            ab13[v241] = v239;	// L880
            int8_t v242 = v199[14];	// L881
            int32_t v243 = r;	// L882
            int v244 = v243;	// L883
            ab14[v244] = v242;	// L884
            int8_t v245 = v199[15];	// L885
            int32_t v246 = r;	// L886
            int v247 = v246;	// L887
            ab15[v247] = v245;	// L888
            int8_t v248 = v199[0];	// L889
            e[0] = v248;	// L890
            int8_t v249 = v199[1];	// L891
            e[1] = v249;	// L892
            int8_t v250 = v199[2];	// L893
            e[2] = v250;	// L894
            int8_t v251 = v199[3];	// L895
            e[3] = v251;	// L896
            int8_t v252 = v199[4];	// L897
            e[4] = v252;	// L898
            int8_t v253 = v199[5];	// L899
            e[5] = v253;	// L900
            int8_t v254 = v199[6];	// L901
            e[6] = v254;	// L902
            int8_t v255 = v199[7];	// L903
            e[7] = v255;	// L904
            int8_t v256 = v199[8];	// L905
            e[8] = v256;	// L906
            int8_t v257 = v199[9];	// L907
            e[9] = v257;	// L908
            int8_t v258 = v199[10];	// L909
            e[10] = v258;	// L910
            int8_t v259 = v199[11];	// L911
            e[11] = v259;	// L912
            int8_t v260 = v199[12];	// L913
            e[12] = v260;	// L914
            int8_t v261 = v199[13];	// L915
            e[13] = v261;	// L916
            int8_t v262 = v199[14];	// L917
            e[14] = v262;	// L918
            int8_t v263 = v199[15];	// L919
            e[15] = v263;	// L920
          } else {
            int32_t v264 = r;	// L922
            int v265 = v264;	// L923
            int8_t v266 = ab0[v265];	// L924
            e[0] = v266;	// L925
            int32_t v267 = r;	// L926
            int v268 = v267;	// L927
            int8_t v269 = ab1[v268];	// L928
            e[1] = v269;	// L929
            int32_t v270 = r;	// L930
            int v271 = v270;	// L931
            int8_t v272 = ab2[v271];	// L932
            e[2] = v272;	// L933
            int32_t v273 = r;	// L934
            int v274 = v273;	// L935
            int8_t v275 = ab3[v274];	// L936
            e[3] = v275;	// L937
            int32_t v276 = r;	// L938
            int v277 = v276;	// L939
            int8_t v278 = ab4[v277];	// L940
            e[4] = v278;	// L941
            int32_t v279 = r;	// L942
            int v280 = v279;	// L943
            int8_t v281 = ab5[v280];	// L944
            e[5] = v281;	// L945
            int32_t v282 = r;	// L946
            int v283 = v282;	// L947
            int8_t v284 = ab6[v283];	// L948
            e[6] = v284;	// L949
            int32_t v285 = r;	// L950
            int v286 = v285;	// L951
            int8_t v287 = ab7[v286];	// L952
            e[7] = v287;	// L953
            int32_t v288 = r;	// L954
            int v289 = v288;	// L955
            int8_t v290 = ab8[v289];	// L956
            e[8] = v290;	// L957
            int32_t v291 = r;	// L958
            int v292 = v291;	// L959
            int8_t v293 = ab9[v292];	// L960
            e[9] = v293;	// L961
            int32_t v294 = r;	// L962
            int v295 = v294;	// L963
            int8_t v296 = ab10[v295];	// L964
            e[10] = v296;	// L965
            int32_t v297 = r;	// L966
            int v298 = v297;	// L967
            int8_t v299 = ab11[v298];	// L968
            e[11] = v299;	// L969
            int32_t v300 = r;	// L970
            int v301 = v300;	// L971
            int8_t v302 = ab12[v301];	// L972
            e[12] = v302;	// L973
            int32_t v303 = r;	// L974
            int v304 = v303;	// L975
            int8_t v305 = ab13[v304];	// L976
            e[13] = v305;	// L977
            int32_t v306 = r;	// L978
            int v307 = v306;	// L979
            int8_t v308 = ab14[v307];	// L980
            e[14] = v308;	// L981
            int32_t v309 = r;	// L982
            int v310 = v309;	// L983
            int8_t v311 = ab15[v310];	// L984
            e[15] = v311;	// L985
          }
        }
        int64_t u;	// L991
        u = 0;	// L992
        bool v313 = carry;	// L993
        bool v314 = nextbias;	// L994
        bool v315 = v313 & v314;	// L995
        if (v315) {	// L1000
          int64_t v316 = bb;	// L1001
          int64_t v317 = v316 >> 32;	// L1005
          int64_t imm;	// L1006
          imm = v317;	// L1007
          int32_t v319 = rl;	// L1008
          int32_t v320 = v319 & 1;	// L1011
          bool v321 = v320 != 0;	// L1014
          if (v321) {	// L1015
            int64_t v322 = v1.read();	// L1016
            bb = v322;	// L1017
            int64_t v323 = bb;	// L1018
            int64_t v324 = v323 & 65535;	// L1022
            int64_t l0;	// L1023
            l0 = v324;	// L1024
            int64_t v326 = bb;	// L1025
            int64_t v327 = v326 >> 16;	// L1029
            int64_t v328 = v327 & 65535;	// L1033
            int64_t l1;	// L1034
            l1 = v328;	// L1035
            int64_t v330 = l1;	// L1036
            int64_t v331 = v330 ^ 32768;	// L1040
            ap_int<65> v332 = v331;	// L1041
            ap_int<65> v333 = v332 - 32768;	// L1045
            ap_int<65> v334 = v333 << 16;	// L1049
            int64_t v335 = l0;	// L1050
            ap_int<65> v336 = v335;	// L1051
            ap_int<65> v337 = v334 | v336;	// L1052
            int64_t v338 = v337;	// L1053
            imm = v338;	// L1054
          }
          int32_t v339 = rl;	// L1056
          int32_t v340 = v339 & 15;	// L1059
          ap_int<33> v341 = v340;	// L1063
          ap_int<33> v342 = 15 - v341;	// L1064
          int64_t v343 = v342;	// L1065
          int64_t q;	// L1066
          q = v343;	// L1067
          int64_t v345 = imm;	// L1068
          int64_t v346 = v345 << 32;	// L1072
          int64_t v347 = v346 | 4096;	// L1076
          int64_t v348 = q;	// L1077
          int64_t v349 = v348 << 13;	// L1081
          int64_t v350 = v347 | v349;	// L1082
          u = v350;	// L1083
        }
        bool v351 = isopen;	// L1085
        int32_t v352 = v351;	// L1086
        bool v353 = v352 == 0;	// L1089
        if (v353) {	// L1090
          bool v354 = klast;	// L1091
          if (v354) {	// L1096
            bool v355 = noroom;	// L1097
            if (v355) {	// L1102
              int8_t v356 = v2.read();	// L1103
              int8_t paid;	// L1104
              paid = v356;	// L1105
            } else {
              int32_t v358 = space;	// L1107
              bool v359 = v358 == 1;	// L1110
              if (v359) {	// L1111
                noroom = 1;	// L1115
              }
              int32_t v360 = space;	// L1117
              ap_int<33> v361 = v360;	// L1118
              ap_int<33> v362 = v361 - 1;	// L1122
              int32_t v363 = v362;	// L1123
              space = v363;	// L1124
            }
          }
        }
        bool fin;	// L1131
        fin = 0;	// L1132
        bool v365 = isopen;	// L1133
        int32_t v366 = v365;	// L1134
        bool v367 = v366 == 0;	// L1137
        if (v367) {	// L1138
          bool v368 = lastblk;	// L1139
          bool v369 = blast;	// L1140
          bool v370 = v368 & v369;	// L1141
          bool v371 = final;	// L1142
          bool v372 = v370 & v371;	// L1143
          fin = v372;	// L1144
        }
        int32_t fl;	// L1148
        fl = 0;	// L1149
        bool v374 = blast;	// L1150
        if (v374) {	// L1155
          fl = 1;	// L1158
        }
        bool v375 = fin;	// L1160
        if (v375) {	// L1165
          int32_t v376 = fl;	// L1166
          int32_t v377 = v376 | 2;	// L1169
          fl = v377;	// L1170
        }
        int32_t v378 = fl;	// L1172
        int8_t v379 = v378;	// L1173
        e[32] = v379;	// L1174
        {
          hls::vector< int8_t, 33 > _vec;
          for (int _iv0 = 0; _iv0 < 33; ++_iv0) {
            _vec[_iv0] = e[_iv0];
          }
          v3.write(_vec);
        }	// L1175
        bool v380 = isopen;	// L1176
        int32_t v381 = v380;	// L1177
        bool v382 = v381 == 0;	// L1180
        if (v382) {	// L1181
          int64_t v383 = u;	// L1182
          int64_t v384 = v383 | 262144;	// L1186
          int64_t v385 = cfg;	// L1187
          int64_t v386 = v384 | v385;	// L1188
          u = v386;	// L1189
          bool v387 = kfirst;	// L1190
          if (v387) {	// L1195
            int64_t v388 = u;	// L1196
            int64_t v389 = v388 | 128;	// L1200
            u = v389;	// L1201
            bool v390 = biased;	// L1202
            bool v391 = rfirst;	// L1203
            bool v392 = v390 & v391;	// L1204
            if (v392) {	// L1209
              int64_t v393 = u;	// L1210
              int64_t v394 = v393 | 512;	// L1214
              u = v394;	// L1215
            }
          }
          bool v395 = klast;	// L1218
          if (v395) {	// L1223
            int64_t v396 = u;	// L1224
            int64_t v397 = v396 | 256;	// L1228
            u = v397;	// L1229
          }
          bool v398 = nlast;	// L1231
          bool v399 = blast;	// L1232
          bool v400 = v398 & v399;	// L1233
          if (v400) {	// L1238
            int64_t v401 = u;	// L1239
            int64_t v402 = v401 | 2048;	// L1243
            u = v402;	// L1244
          }
          bool v403 = fin;	// L1246
          if (v403) {	// L1251
            int64_t v404 = u;	// L1252
            int64_t v405 = v404 | 1024;	// L1256
            u = v405;	// L1257
          }
        }
        int64_t v406 = u;	// L1260
        v5.write(v406);	// L1261
        bool v407 = blast;	// L1262
        int32_t v408 = v407;	// L1263
        bool v409 = v408 == 0;	// L1266
        if (v409) {	// L1267
          bool v410 = isopen;	// L1268
          int32_t v411 = v410;	// L1269
          bool v412 = v411 == 0;	// L1272
          if (v412) {	// L1273
            int32_t v413 = rl;	// L1274
            bool v414 = v413 == 16;	// L1277
            if (v414) {	// L1278
              tail = 1;	// L1282
            }
            int32_t v415 = r;	// L1284
            ap_int<33> v416 = v415;	// L1285
            ap_int<33> v417 = v416 + 1;	// L1289
            int32_t v418 = v417;	// L1290
            r = v418;	// L1291
            rfirst = 0;	// L1295
          }
          int32_t v419 = rl;	// L1297
          ap_int<33> v420 = v419;	// L1298
          ap_int<33> v421 = v420 - 1;	// L1302
          int32_t v422 = v421;	// L1303
          rl = v422;	// L1304
        } else {
          int32_t v423 = ln;	// L1306
          rl = v423;	// L1307
          bool v424 = lsingle;	// L1308
          tail = v424;	// L1309
          bool v425 = isopen;	// L1310
          if (v425) {	// L1315
            isopen = 0;	// L1319
          } else {
            r = 0;	// L1323
            rfirst = 1;	// L1327
            bool v426 = nlast;	// L1328
            int32_t v427 = v426;	// L1329
            bool v428 = v427 == 0;	// L1332
            if (v428) {	// L1333
              nfirst = 0;	// L1337
              int32_t v429 = nleft;	// L1338
              bool v430 = v429 == 1;	// L1341
              if (v430) {	// L1342
                nlast = 1;	// L1346
              }
              int32_t v431 = nleft;	// L1348
              ap_int<33> v432 = v431;	// L1349
              ap_int<33> v433 = v432 - 1;	// L1353
              int32_t v434 = v433;	// L1354
              nleft = v434;	// L1355
            } else {
              int32_t v435 = nbn;	// L1357
              nleft = v435;	// L1358
              nfirst = 1;	// L1362
              bool v436 = nsingle;	// L1363
              nlast = v436;	// L1364
              bool v437 = klast;	// L1365
              int32_t v438 = v437;	// L1366
              bool v439 = v438 == 0;	// L1369
              if (v439) {	// L1370
                kfirst = 0;	// L1374
                int32_t v440 = kleft;	// L1375
                bool v441 = v440 == 1;	// L1378
                if (v441) {	// L1379
                  klast = 1;	// L1383
                }
                int32_t v442 = kleft;	// L1385
                ap_int<33> v443 = v442;	// L1386
                ap_int<33> v444 = v443 - 1;	// L1390
                int32_t v445 = v444;	// L1391
                kleft = v445;	// L1392
              } else {
                int32_t v446 = kbn;	// L1394
                kleft = v446;	// L1395
                kfirst = 1;	// L1399
                bool v447 = ksingle;	// L1400
                klast = v447;	// L1401
                bool v448 = tlast;	// L1402
                int32_t v449 = v448;	// L1403
                bool v450 = v449 == 0;	// L1406
                if (v450) {	// L1407
                  int32_t v451 = tleft;	// L1408
                  bool v452 = v451 == 1;	// L1411
                  if (v452) {	// L1412
                    tlast = 1;	// L1416
                  }
                  int32_t v453 = tleft;	// L1418
                  ap_int<33> v454 = v453;	// L1419
                  ap_int<33> v455 = v454 - 1;	// L1423
                  int32_t v456 = v455;	// L1424
                  tleft = v456;	// L1425
                } else {
                  bool v457 = final;	// L1427
                  if (v457) {	// L1432
                    go = 0;	// L1436
                  } else {
                    st = 1;	// L1440
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


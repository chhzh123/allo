
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void vpu_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t v5 = v0.read();	// L18
  int32_t _st_b;	// L19
  _st_b = v5;	// L20
  int32_t prog[8];	// L21
  for (int v8 = 0; v8 < 8; v8++) {	// L22
    prog[v8] = 0;	// L22
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 8; pc++) {	// L23
    int32_t v10 = v1.read();	// L24
    int32_t word;	// L25
    word = v10;	// L26
    int32_t v12 = word;	// L27
    prog[pc] = v12;	// L28
    int32_t v13 = word;	// L29
    v2.write(v13);	// L30
  }
  l_S_m_1_m: for (int m = 0; m < 6; m++) {	// L32
    int32_t v15 = v3.read();	// L33
    int32_t z;	// L34
    z = v15;	// L35
    int32_t reg[4];	// L36
    for (int v18 = 0; v18 < 4; v18++) {	// L37
      reg[v18] = 0;	// L37
    }
    l_S_step_1_step: for (int step = 0; step < 8; step++) {	// L38
      int32_t v20 = prog[step];	// L39
      int32_t word2;	// L40
      word2 = v20;	// L41
      int32_t v22 = word2;	// L42
      int32_t v23 = v22 >> 24;	// L43
      int32_t v24 = v23 & 255;	// L44
      int32_t opcode;	// L45
      opcode = v24;	// L46
      int32_t v26 = word2;	// L47
      int32_t v27 = v26 >> 20;	// L48
      int32_t v28 = v27 & 15;	// L49
      int32_t dst;	// L50
      dst = v28;	// L51
      int32_t v30 = word2;	// L52
      int32_t v31 = v30 >> 16;	// L53
      int32_t v32 = v31 & 15;	// L54
      int32_t src;	// L55
      src = v32;	// L56
      int32_t v34 = word2;	// L57
      int32_t v35 = v34 & 65535;	// L58
      int32_t imm;	// L59
      imm = v35;	// L60
      int32_t v37 = opcode;	// L61
      bool v38 = v37 == 1;	// L62
      if (v38) {	// L63
        int32_t v39 = z;	// L64
        int32_t v40 = dst;	// L65
        int v41 = v40;	// L66
        reg[v41] = v39;	// L67
      } else {
        int32_t v42 = opcode;	// L69
        bool v43 = v42 == 2;	// L70
        if (v43) {	// L71
          int32_t v44 = _st_b;	// L72
          int32_t v45 = dst;	// L73
          int v46 = v45;	// L74
          reg[v46] = v44;	// L75
        } else {
          int32_t v47 = opcode;	// L77
          bool v48 = v47 == 3;	// L78
          if (v48) {	// L79
            int32_t v49 = imm;	// L80
            int32_t v50 = dst;	// L81
            int v51 = v50;	// L82
            reg[v51] = v49;	// L83
          } else {
            int32_t v52 = opcode;	// L85
            bool v53 = v52 == 4;	// L86
            if (v53) {	// L87
              int32_t v54 = dst;	// L88
              int v55 = v54;	// L89
              int32_t v56 = reg[v55];	// L90
              int32_t v57 = src;	// L91
              int v58 = v57;	// L92
              int32_t v59 = reg[v58];	// L93
              ap_int<33> v60 = v56;	// L94
              ap_int<33> v61 = v59;	// L95
              ap_int<33> v62 = v60 + v61;	// L96
              int32_t v63 = v62;	// L97
              reg[v55] = v63;	// L100
            } else {
              int32_t v64 = opcode;	// L102
              bool v65 = v64 == 5;	// L103
              if (v65) {	// L104
                int32_t v66 = dst;	// L105
                int v67 = v66;	// L106
                int32_t v68 = reg[v67];	// L107
                int32_t v69 = src;	// L108
                int v70 = v69;	// L109
                int32_t v71 = reg[v70];	// L110
                int64_t v72 = v68;	// L111
                int64_t v73 = v71;	// L112
                int64_t v74 = v72 * v73;	// L113
                int32_t v75 = v74;	// L114
                reg[v67] = v75;	// L117
              } else {
                int32_t v76 = opcode;	// L119
                bool v77 = v76 == 6;	// L120
                if (v77) {	// L121
                  int32_t v78 = src;	// L122
                  int v79 = v78;	// L123
                  int32_t v80 = reg[v79];	// L124
                  int32_t v81 = dst;	// L125
                  int v82 = v81;	// L126
                  int32_t v83 = reg[v82];	// L127
                  bool v84 = v80 > v83;	// L128
                  if (v84) {	// L129
                    int32_t v85 = src;	// L130
                    int v86 = v85;	// L131
                    int32_t v87 = reg[v86];	// L132
                    int32_t v88 = dst;	// L133
                    int v89 = v88;	// L134
                    reg[v89] = v87;	// L135
                  }
                } else {
                  int32_t v90 = opcode;	// L138
                  bool v91 = v90 == 7;	// L139
                  if (v91) {	// L140
                    int32_t v92 = dst;	// L141
                    int v93 = v92;	// L142
                    int32_t v94 = reg[v93];	// L143
                    int32_t v95 = imm;	// L144
                    int32_t v96 = v94 >> v95;	// L145
                    reg[v93] = v96;	// L148
                  } else {
                    int32_t v97 = opcode;	// L150
                    bool v98 = v97 == 8;	// L151
                    if (v98) {	// L152
                      int32_t v99 = dst;	// L153
                      int v100 = v99;	// L154
                      int32_t v101 = reg[v100];	// L155
                      v4.write(v101);	// L156
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
}

/// This is top function.
void top(

) {	// L169
  #pragma HLS dataflow
  hls::stream< int32_t > v102;
  #pragma HLS stream variable=v102 depth=2	// L170
  hls::stream< int32_t > v103;
  #pragma HLS stream variable=v103 depth=2	// L171
  hls::stream< int32_t > v104;
  #pragma HLS stream variable=v104 depth=2	// L172
  hls::stream< int32_t > v105;
  #pragma HLS stream variable=v105 depth=2	// L173
  hls::stream< int32_t > v106;
  #pragma HLS stream variable=v106 depth=2	// L174
  vpu_r2_0(v102, v103, v104, v106, v105);	// L175
}


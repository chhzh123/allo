
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
void vpu_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t v5 = v0.read();	// L3
  int32_t _st_b;	// L4
  _st_b = v5;	// L5
  int32_t prog[8];	// L6
  for (int v8 = 0; v8 < 8; v8++) {	// L8
    prog[v8] = 0;	// L8
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 8; pc++) {	// L9
    int32_t v10 = v1.read();	// L10
    int32_t word;	// L11
    word = v10;	// L12
    int32_t v12 = word;	// L13
    prog[pc] = v12;	// L14
    int32_t v13 = word;	// L15
    v2.write(v13);	// L16
  }
  l_S_m_1_m: for (int m = 0; m < 6; m++) {	// L18
    int32_t v15 = v4.read();	// L19
    int32_t z;	// L20
    z = v15;	// L21
    int32_t reg[4];	// L22
    for (int v18 = 0; v18 < 4; v18++) {	// L24
      reg[v18] = 0;	// L24
    }
    l_S_step_1_step: for (int step = 0; step < 8; step++) {	// L25
      int32_t v20 = prog[step];	// L26
      int32_t word2;	// L27
      word2 = v20;	// L28
      int32_t v22 = word2;	// L29
      int32_t v23 = v22 >> 24;	// L32
      int32_t v24 = v23 & 255;	// L35
      int32_t opcode;	// L36
      opcode = v24;	// L37
      int32_t v26 = word2;	// L38
      int32_t v27 = v26 >> 20;	// L41
      int32_t v28 = v27 & 15;	// L44
      int32_t dst;	// L45
      dst = v28;	// L46
      int32_t v30 = word2;	// L47
      int32_t v31 = v30 >> 16;	// L50
      int32_t v32 = v31 & 15;	// L53
      int32_t src;	// L54
      src = v32;	// L55
      int32_t v34 = word2;	// L56
      int32_t v35 = v34 & 65535;	// L59
      int32_t imm;	// L60
      imm = v35;	// L61
      int32_t v37 = opcode;	// L62
      bool v38 = v37 == 1;	// L65
      if (v38) {	// L66
        int32_t v39 = z;	// L67
        int32_t v40 = dst;	// L68
        int v41 = v40;	// L69
        reg[v41] = v39;	// L70
      } else {
        int32_t v42 = opcode;	// L72
        bool v43 = v42 == 2;	// L75
        if (v43) {	// L76
          int32_t v44 = _st_b;	// L77
          int32_t v45 = dst;	// L78
          int v46 = v45;	// L79
          reg[v46] = v44;	// L80
        } else {
          int32_t v47 = opcode;	// L82
          bool v48 = v47 == 3;	// L85
          if (v48) {	// L86
            int32_t v49 = imm;	// L87
            int32_t v50 = dst;	// L88
            int v51 = v50;	// L89
            reg[v51] = v49;	// L90
          } else {
            int32_t v52 = opcode;	// L92
            bool v53 = v52 == 4;	// L95
            if (v53) {	// L96
              int32_t v54 = dst;	// L97
              int v55 = v54;	// L98
              int32_t v56 = reg[v55];	// L99
              int32_t v57 = src;	// L100
              int v58 = v57;	// L101
              int32_t v59 = reg[v58];	// L102
              ap_int<33> v60 = v56;	// L103
              ap_int<33> v61 = v59;	// L104
              ap_int<33> v62 = v60 + v61;	// L105
              int32_t v63 = v62;	// L106
              reg[v55] = v63;	// L109
            } else {
              int32_t v64 = opcode;	// L111
              bool v65 = v64 == 5;	// L114
              if (v65) {	// L115
                int32_t v66 = dst;	// L116
                int v67 = v66;	// L117
                int32_t v68 = reg[v67];	// L118
                int32_t v69 = src;	// L119
                int v70 = v69;	// L120
                int32_t v71 = reg[v70];	// L121
                int64_t v72 = v68;	// L122
                int64_t v73 = v71;	// L123
                int64_t v74 = v72 * v73;	// L124
                int32_t v75 = v74;	// L125
                reg[v67] = v75;	// L128
              } else {
                int32_t v76 = opcode;	// L130
                bool v77 = v76 == 6;	// L133
                if (v77) {	// L134
                  int32_t v78 = src;	// L135
                  int v79 = v78;	// L136
                  int32_t v80 = reg[v79];	// L137
                  int32_t v81 = dst;	// L138
                  int v82 = v81;	// L139
                  int32_t v83 = reg[v82];	// L140
                  bool v84 = v80 > v83;	// L141
                  if (v84) {	// L142
                    int32_t v85 = src;	// L143
                    int v86 = v85;	// L144
                    int32_t v87 = reg[v86];	// L145
                    int32_t v88 = dst;	// L146
                    int v89 = v88;	// L147
                    reg[v89] = v87;	// L148
                  }
                } else {
                  int32_t v90 = opcode;	// L151
                  bool v91 = v90 == 7;	// L154
                  if (v91) {	// L155
                    int32_t v92 = dst;	// L156
                    int v93 = v92;	// L157
                    int32_t v94 = reg[v93];	// L158
                    int32_t v95 = imm;	// L159
                    int32_t v96 = v94 >> v95;	// L160
                    reg[v93] = v96;	// L163
                  } else {
                    int32_t v97 = opcode;	// L165
                    bool v98 = v97 == 8;	// L168
                    if (v98) {	// L169
                      int32_t v99 = dst;	// L170
                      int v100 = v99;	// L171
                      int32_t v101 = reg[v100];	// L172
                      v3.write(v101);	// L173
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



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
void tiled_vpu_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t v5 = v0.read();	// L3
  int32_t _st_b;	// L4
  _st_b = v5;	// L5
  int32_t prog[12];	// L6
  for (int v8 = 0; v8 < 12; v8++) {	// L8
    prog[v8] = 0;	// L8
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 12; pc++) {	// L9
    int32_t v10 = v1.read();	// L10
    int32_t word;	// L11
    word = v10;	// L12
    int32_t v12 = word;	// L13
    prog[pc] = v12;	// L14
    int32_t v13 = word;	// L15
    v2.write(v13);	// L16
  }
  l_S_m_1_m: for (int m = 0; m < 6; m++) {	// L18
    int32_t reg[4];	// L19
    for (int v16 = 0; v16 < 4; v16++) {	// L21
      reg[v16] = 0;	// L21
    }
    l_S_step_1_step: for (int step = 0; step < 12; step++) {	// L22
      int32_t v18 = prog[step];	// L23
      int32_t word2;	// L24
      word2 = v18;	// L25
      int32_t v20 = word2;	// L26
      int32_t v21 = v20 >> 24;	// L29
      int32_t v22 = v21 & 255;	// L32
      int32_t opcode;	// L33
      opcode = v22;	// L34
      int32_t v24 = word2;	// L35
      int32_t v25 = v24 >> 20;	// L38
      int32_t v26 = v25 & 15;	// L41
      int32_t dst;	// L42
      dst = v26;	// L43
      int32_t v28 = word2;	// L44
      int32_t v29 = v28 >> 16;	// L47
      int32_t v30 = v29 & 15;	// L50
      int32_t src;	// L51
      src = v30;	// L52
      int32_t v32 = word2;	// L53
      int32_t v33 = v32 & 65535;	// L56
      int32_t imm;	// L57
      imm = v33;	// L58
      int32_t v35 = opcode;	// L59
      bool v36 = v35 == 9;	// L62
      if (v36) {	// L63
        int32_t v37 = v4.read();	// L64
        int32_t zz;	// L65
        zz = v37;	// L66
        int32_t v39 = dst;	// L67
        int v40 = v39;	// L68
        int32_t v41 = reg[v40];	// L69
        int32_t v42 = zz;	// L70
        ap_int<33> v43 = v41;	// L71
        ap_int<33> v44 = v42;	// L72
        ap_int<33> v45 = v43 + v44;	// L73
        int32_t v46 = v45;	// L74
        reg[v40] = v46;	// L77
      } else {
        int32_t v47 = opcode;	// L79
        bool v48 = v47 == 2;	// L82
        if (v48) {	// L83
          int32_t v49 = _st_b;	// L84
          int32_t v50 = dst;	// L85
          int v51 = v50;	// L86
          reg[v51] = v49;	// L87
        } else {
          int32_t v52 = opcode;	// L89
          bool v53 = v52 == 3;	// L92
          if (v53) {	// L93
            int32_t v54 = imm;	// L94
            int32_t v55 = dst;	// L95
            int v56 = v55;	// L96
            reg[v56] = v54;	// L97
          } else {
            int32_t v57 = opcode;	// L99
            bool v58 = v57 == 4;	// L102
            if (v58) {	// L103
              int32_t v59 = dst;	// L104
              int v60 = v59;	// L105
              int32_t v61 = reg[v60];	// L106
              int32_t v62 = src;	// L107
              int v63 = v62;	// L108
              int32_t v64 = reg[v63];	// L109
              ap_int<33> v65 = v61;	// L110
              ap_int<33> v66 = v64;	// L111
              ap_int<33> v67 = v65 + v66;	// L112
              int32_t v68 = v67;	// L113
              reg[v60] = v68;	// L116
            } else {
              int32_t v69 = opcode;	// L118
              bool v70 = v69 == 5;	// L121
              if (v70) {	// L122
                int32_t v71 = dst;	// L123
                int v72 = v71;	// L124
                int32_t v73 = reg[v72];	// L125
                int32_t v74 = src;	// L126
                int v75 = v74;	// L127
                int32_t v76 = reg[v75];	// L128
                int64_t v77 = v73;	// L129
                int64_t v78 = v76;	// L130
                int64_t v79 = v77 * v78;	// L131
                int32_t v80 = v79;	// L132
                reg[v72] = v80;	// L135
              } else {
                int32_t v81 = opcode;	// L137
                bool v82 = v81 == 6;	// L140
                if (v82) {	// L141
                  int32_t v83 = src;	// L142
                  int v84 = v83;	// L143
                  int32_t v85 = reg[v84];	// L144
                  int32_t v86 = dst;	// L145
                  int v87 = v86;	// L146
                  int32_t v88 = reg[v87];	// L147
                  bool v89 = v85 > v88;	// L148
                  if (v89) {	// L149
                    int32_t v90 = src;	// L150
                    int v91 = v90;	// L151
                    int32_t v92 = reg[v91];	// L152
                    int32_t v93 = dst;	// L153
                    int v94 = v93;	// L154
                    reg[v94] = v92;	// L155
                  }
                } else {
                  int32_t v95 = opcode;	// L158
                  bool v96 = v95 == 7;	// L161
                  if (v96) {	// L162
                    int32_t v97 = dst;	// L163
                    int v98 = v97;	// L164
                    int32_t v99 = reg[v98];	// L165
                    int32_t v100 = imm;	// L166
                    int32_t v101 = v99 >> v100;	// L167
                    reg[v98] = v101;	// L170
                  } else {
                    int32_t v102 = opcode;	// L172
                    bool v103 = v102 == 8;	// L175
                    if (v103) {	// L176
                      int32_t v104 = dst;	// L177
                      int v105 = v104;	// L178
                      int32_t v106 = reg[v105];	// L179
                      v3.write(v106);	// L180
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


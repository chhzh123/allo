
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void tiled_vpu_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t v5 = v0.read();	// L18
  int32_t _st_b;	// L19
  _st_b = v5;	// L20
  int32_t prog[12];	// L21
  for (int v8 = 0; v8 < 12; v8++) {	// L22
    prog[v8] = 0;	// L22
  }
  l_S_pc_0_pc: for (int pc = 0; pc < 12; pc++) {	// L23
    int32_t v10 = v1.read();	// L24
    int32_t word;	// L25
    word = v10;	// L26
    int32_t v12 = word;	// L27
    prog[pc] = v12;	// L28
    int32_t v13 = word;	// L29
    v2.write(v13);	// L30
  }
  l_S_m_1_m: for (int m = 0; m < 6; m++) {	// L32
    int32_t reg[4];	// L33
    for (int v16 = 0; v16 < 4; v16++) {	// L34
      reg[v16] = 0;	// L34
    }
    l_S_step_1_step: for (int step = 0; step < 12; step++) {	// L35
      int32_t v18 = prog[step];	// L36
      int32_t word2;	// L37
      word2 = v18;	// L38
      int32_t v20 = word2;	// L39
      int32_t v21 = v20 >> 24;	// L40
      int32_t v22 = v21 & 255;	// L41
      int32_t opcode;	// L42
      opcode = v22;	// L43
      int32_t v24 = word2;	// L44
      int32_t v25 = v24 >> 20;	// L45
      int32_t v26 = v25 & 15;	// L46
      int32_t dst;	// L47
      dst = v26;	// L48
      int32_t v28 = word2;	// L49
      int32_t v29 = v28 >> 16;	// L50
      int32_t v30 = v29 & 15;	// L51
      int32_t src;	// L52
      src = v30;	// L53
      int32_t v32 = word2;	// L54
      int32_t v33 = v32 & 65535;	// L55
      int32_t imm;	// L56
      imm = v33;	// L57
      int32_t v35 = opcode;	// L58
      bool v36 = v35 == 9;	// L59
      if (v36) {	// L60
        int32_t v37 = v3.read();	// L61
        int32_t zz;	// L62
        zz = v37;	// L63
        int32_t v39 = dst;	// L64
        int v40 = v39;	// L65
        int32_t v41 = reg[v40];	// L66
        int32_t v42 = zz;	// L67
        ap_int<33> v43 = v41;	// L68
        ap_int<33> v44 = v42;	// L69
        ap_int<33> v45 = v43 + v44;	// L70
        int32_t v46 = v45;	// L71
        reg[v40] = v46;	// L74
      } else {
        int32_t v47 = opcode;	// L76
        bool v48 = v47 == 2;	// L77
        if (v48) {	// L78
          int32_t v49 = _st_b;	// L79
          int32_t v50 = dst;	// L80
          int v51 = v50;	// L81
          reg[v51] = v49;	// L82
        } else {
          int32_t v52 = opcode;	// L84
          bool v53 = v52 == 3;	// L85
          if (v53) {	// L86
            int32_t v54 = imm;	// L87
            int32_t v55 = dst;	// L88
            int v56 = v55;	// L89
            reg[v56] = v54;	// L90
          } else {
            int32_t v57 = opcode;	// L92
            bool v58 = v57 == 4;	// L93
            if (v58) {	// L94
              int32_t v59 = dst;	// L95
              int v60 = v59;	// L96
              int32_t v61 = reg[v60];	// L97
              int32_t v62 = src;	// L98
              int v63 = v62;	// L99
              int32_t v64 = reg[v63];	// L100
              ap_int<33> v65 = v61;	// L101
              ap_int<33> v66 = v64;	// L102
              ap_int<33> v67 = v65 + v66;	// L103
              int32_t v68 = v67;	// L104
              reg[v60] = v68;	// L107
            } else {
              int32_t v69 = opcode;	// L109
              bool v70 = v69 == 5;	// L110
              if (v70) {	// L111
                int32_t v71 = dst;	// L112
                int v72 = v71;	// L113
                int32_t v73 = reg[v72];	// L114
                int32_t v74 = src;	// L115
                int v75 = v74;	// L116
                int32_t v76 = reg[v75];	// L117
                int64_t v77 = v73;	// L118
                int64_t v78 = v76;	// L119
                int64_t v79 = v77 * v78;	// L120
                int32_t v80 = v79;	// L121
                reg[v72] = v80;	// L124
              } else {
                int32_t v81 = opcode;	// L126
                bool v82 = v81 == 6;	// L127
                if (v82) {	// L128
                  int32_t v83 = src;	// L129
                  int v84 = v83;	// L130
                  int32_t v85 = reg[v84];	// L131
                  int32_t v86 = dst;	// L132
                  int v87 = v86;	// L133
                  int32_t v88 = reg[v87];	// L134
                  bool v89 = v85 > v88;	// L135
                  if (v89) {	// L136
                    int32_t v90 = src;	// L137
                    int v91 = v90;	// L138
                    int32_t v92 = reg[v91];	// L139
                    int32_t v93 = dst;	// L140
                    int v94 = v93;	// L141
                    reg[v94] = v92;	// L142
                  }
                } else {
                  int32_t v95 = opcode;	// L145
                  bool v96 = v95 == 7;	// L146
                  if (v96) {	// L147
                    int32_t v97 = dst;	// L148
                    int v98 = v97;	// L149
                    int32_t v99 = reg[v98];	// L150
                    int32_t v100 = imm;	// L151
                    int32_t v101 = v99 >> v100;	// L152
                    reg[v98] = v101;	// L155
                  } else {
                    int32_t v102 = opcode;	// L157
                    bool v103 = v102 == 8;	// L158
                    if (v103) {	// L159
                      int32_t v104 = dst;	// L160
                      int v105 = v104;	// L161
                      int32_t v106 = reg[v105];	// L162
                      v4.write(v106);	// L163
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

) {	// L176
  #pragma HLS dataflow
  hls::stream< int32_t > v107;
  #pragma HLS stream variable=v107 depth=2	// L177
  hls::stream< int32_t > v108;
  #pragma HLS stream variable=v108 depth=2	// L178
  hls::stream< int32_t > v109;
  #pragma HLS stream variable=v109 depth=2	// L179
  hls::stream< int32_t > v110;
  #pragma HLS stream variable=v110 depth=2	// L180
  hls::stream< int32_t > v111;
  #pragma HLS stream variable=v111 depth=2	// L181
  tiled_vpu_r2_0(v107, v108, v109, v111, v110);	// L182
}


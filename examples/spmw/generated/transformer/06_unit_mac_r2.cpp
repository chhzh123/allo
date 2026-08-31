
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
void mac_r2_0(
  hls::stream< hls::vector< int8_t, 4 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int32_t >& v5
) {	// L2
  int8_t v6[4];
  {
    hls::vector< int8_t, 4 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
      v6[_iv0] = _vec[_iv0];
    }
  }	// L9
  l_S_step_0_step: for (int step = 0; step < 8; step++) {	// L10
    int32_t v8 = v1.read();	// L11
    int32_t word;	// L12
    word = v8;	// L13
    int32_t v10 = word;	// L14
    v2.write(v10);	// L15
    int32_t v11 = word;	// L16
    int32_t v12 = v11 >> 24;	// L17
    int32_t v13 = v12 & 255;	// L18
    int32_t opcode;	// L19
    opcode = v13;	// L20
    int32_t v15 = word;	// L21
    int32_t v16 = v15 >> 16;	// L22
    int32_t v17 = v16 & 255;	// L23
    int32_t tile;	// L24
    tile = v17;	// L25
    int8_t v19 = v3.read();	// L26
    int8_t a;	// L27
    a = v19;	// L28
    int32_t p;	// L29
    p = 0;	// L30
    int8_t v22 = a;	// L31
    v4.write(v22);	// L32
    int32_t v23 = tile;	// L33
    int v24 = v23;	// L34
    int8_t v25 = v6[v24];	// L35
    int32_t v26 = v25;	// L36
    int32_t wt;	// L37
    wt = v26;	// L38
    int32_t v28 = opcode;	// L39
    bool v29 = v28 == 1;	// L40
    if (v29) {	// L41
      int32_t v30 = p;	// L42
      int8_t v31 = a;	// L43
      int32_t v32 = wt;	// L44
      ap_int<40> v33 = v31;	// L45
      ap_int<40> v34 = v32;	// L46
      ap_int<40> v35 = v33 * v34;	// L47
      ap_int<41> v36 = v30;	// L48
      ap_int<41> v37 = v35;	// L49
      ap_int<41> v38 = v36 + v37;	// L50
      v5.write(v38);	// L51
    } else {
      int32_t v39 = opcode;	// L53
      bool v40 = v39 == 2;	// L54
      if (v40) {	// L55
        int8_t v41 = a;	// L56
        int32_t v42 = wt;	// L57
        ap_int<40> v43 = v41;	// L58
        ap_int<40> v44 = v42;	// L59
        ap_int<40> v45 = v43 * v44;	// L60
        v5.write(v45);	// L61
      } else {
        int32_t v46 = p;	// L63
        v5.write(v46);	// L64
      }
    }
  }
}

/// This is top function.
void top(

) {	// L70
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v47;
  #pragma HLS stream variable=v47 depth=2	// L71
  hls::stream< int8_t > v48;
  #pragma HLS stream variable=v48 depth=2	// L72
  hls::stream< int8_t > v49;
  #pragma HLS stream variable=v49 depth=2	// L73
  hls::stream< int32_t > v50;
  #pragma HLS stream variable=v50 depth=2	// L74
  hls::stream< int32_t > v51;
  #pragma HLS stream variable=v51 depth=2	// L75
  hls::stream< int32_t > v52;
  #pragma HLS stream variable=v52 depth=2	// L76
  mac_r2_0(v47, v50, v51, v48, v49, v52);	// L77
}



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
void mac_r2_0(
  hls::stream< hls::vector< int8_t, 4 > >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5
) {	// L2
  int8_t v6[4];
  {
    hls::vector< int8_t, 4 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
      v6[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v7 = v3.read();	// L4
  int32_t count;	// L5
  count = v7;	// L6
  int32_t v9 = count;	// L7
  v4.write(v9);	// L8
  int32_t v10 = count;	// L9
  int v11 = v10;	// L13
  for (int v12 = 0; v12 < v11; v12 += 1) {	// L17
    int32_t v13 = v3.read();	// L18
    int32_t word;	// L19
    word = v13;	// L20
    int32_t v15 = word;	// L21
    v4.write(v15);	// L22
    int32_t v16 = word;	// L23
    int32_t v17 = v16 >> 24;	// L26
    int32_t v18 = v17 & 255;	// L29
    int32_t opcode;	// L30
    opcode = v18;	// L31
    int32_t v20 = word;	// L32
    int32_t v21 = v20 >> 16;	// L35
    int32_t v22 = v21 & 255;	// L38
    int32_t tile;	// L39
    tile = v22;	// L40
    int8_t v24 = v1.read();	// L41
    int8_t a;	// L42
    a = v24;	// L43
    int32_t p;	// L46
    p = 0;	// L47
    int8_t v27 = a;	// L48
    v2.write(v27);	// L49
    int32_t v28 = tile;	// L50
    int v29 = v28;	// L51
    int8_t v30 = v6[v29];	// L52
    int32_t v31 = v30;	// L53
    int32_t wt;	// L54
    wt = v31;	// L55
    int32_t v33 = opcode;	// L56
    bool v34 = v33 == 1;	// L59
    if (v34) {	// L60
      int32_t v35 = p;	// L61
      int8_t v36 = a;	// L62
      int32_t v37 = wt;	// L63
      ap_int<40> v38 = v36;	// L64
      ap_int<40> v39 = v37;	// L65
      ap_int<40> v40 = v38 * v39;	// L66
      ap_int<41> v41 = v35;	// L67
      ap_int<41> v42 = v40;	// L68
      ap_int<41> v43 = v41 + v42;	// L69
      v5.write(v43);	// L70
    } else {
      int32_t v44 = opcode;	// L72
      bool v45 = v44 == 2;	// L75
      if (v45) {	// L76
        int8_t v46 = a;	// L77
        int32_t v47 = wt;	// L78
        ap_int<40> v48 = v46;	// L79
        ap_int<40> v49 = v47;	// L80
        ap_int<40> v50 = v48 * v49;	// L81
        v5.write(v50);	// L82
      } else {
        int32_t v51 = p;	// L84
        v5.write(v51);	// L85
      }
    }
  }
}


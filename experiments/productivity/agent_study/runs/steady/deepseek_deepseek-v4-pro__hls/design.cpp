#include <hls_stream.h>
#include <ap_int.h>

// PE in column 0: no c_in, writes result to c_out
static void pe_left(
    hls::stream<ap_int<8> > &a_in,
    hls::stream<ap_int<8> > &a_out,
    hls::stream<ap_int<8> > &b_in,
    hls::stream<ap_int<8> > &b_out,
    hls::stream<ap_int<32> > &c_out) {
#pragma HLS inline off
    while (1) {
        ap_int<32> acc = 0;
        for (int k = 0; k < 8; k++) {
#pragma HLS pipeline II=1
            ap_int<8> a = a_in.read();
            ap_int<8> b = b_in.read();
            acc += a * b;
            a_out.write(a);
            b_out.write(b);
        }
        c_out.write(acc);
    }
}

// PE in columns 1..7: forwards 'col' results from west, then injects own
static void pe_rest(
    int col,
    hls::stream<ap_int<8> > &a_in,
    hls::stream<ap_int<8> > &a_out,
    hls::stream<ap_int<8> > &b_in,
    hls::stream<ap_int<8> > &b_out,
    hls::stream<ap_int<32> > &c_in,
    hls::stream<ap_int<32> > &c_out) {
#pragma HLS inline off
    while (1) {
        ap_int<32> acc = 0;
        for (int k = 0; k < 8; k++) {
#pragma HLS pipeline II=1
            ap_int<8> a = a_in.read();
            ap_int<8> b = b_in.read();
            acc += a * b;
            a_out.write(a);
            b_out.write(b);
        }
        // forward 'col' results from the west
        for (int n = 0; n < col; n++) {
#pragma HLS pipeline II=1
            c_out.write(c_in.read());
        }
        c_out.write(acc);
    }
}

// 8-bit delay
static void delay8(hls::stream<ap_int<8> > &in, hls::stream<ap_int<8> > &out) {
#pragma HLS inline off
    while (1) {
#pragma HLS pipeline II=1
        out.write(in.read());
    }
}

// 32-bit pass-through
static void pass32(hls::stream<ap_int<32> > &in, hls::stream<ap_int<32> > &out) {
#pragma HLS inline off
    while (1) {
#pragma HLS pipeline II=1
        out.write(in.read());
    }
}

// Dummy reader: consumes unconnected outputs
static void dummy8(hls::stream<ap_int<8> > &in) {
#pragma HLS inline off
    while (1) {
#pragma HLS pipeline II=1
        in.read();
    }
}

void gemm_tile(
    hls::stream<ap_int<8> > &a_in_0,
    hls::stream<ap_int<8> > &a_in_1,
    hls::stream<ap_int<8> > &a_in_2,
    hls::stream<ap_int<8> > &a_in_3,
    hls::stream<ap_int<8> > &a_in_4,
    hls::stream<ap_int<8> > &a_in_5,
    hls::stream<ap_int<8> > &a_in_6,
    hls::stream<ap_int<8> > &a_in_7,
    hls::stream<ap_int<8> > &b_in_0,
    hls::stream<ap_int<8> > &b_in_1,
    hls::stream<ap_int<8> > &b_in_2,
    hls::stream<ap_int<8> > &b_in_3,
    hls::stream<ap_int<8> > &b_in_4,
    hls::stream<ap_int<8> > &b_in_5,
    hls::stream<ap_int<8> > &b_in_6,
    hls::stream<ap_int<8> > &b_in_7,
    hls::stream<ap_int<32> > &c_out_0,
    hls::stream<ap_int<32> > &c_out_1,
    hls::stream<ap_int<32> > &c_out_2,
    hls::stream<ap_int<32> > &c_out_3,
    hls::stream<ap_int<32> > &c_out_4,
    hls::stream<ap_int<32> > &c_out_5,
    hls::stream<ap_int<32> > &c_out_6,
    hls::stream<ap_int<32> > &c_out_7) {
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS dataflow

    // ---- internal stream arrays ----

    // a_h[i][j] for j=0..7 : input to PE(i,j) from west
    // a_h[i][8]           : output from PE(i,7) to east (dummy)
    hls::stream<ap_int<8> > a_h[8][9];
#pragma HLS stream variable=a_h depth=2

    // b_v[i][j] for i=0..7 : input to PE(i,j) from north
    // b_v[8][j]           : output from PE(7,j) to south (dummy)
    hls::stream<ap_int<8> > b_v[9][8];
#pragma HLS stream variable=b_v depth=2

    // c_h[i][j] for j=0..6 : PE(i,j) output -> PE(i,j+1) input
    // c_h[i][7]            : PE(i,7) output -> c_out_i
    hls::stream<ap_int<32> > c_h[8][8];
#pragma HLS stream variable=c_h depth=2

    // delay-line streams for A inputs
    hls::stream<ap_int<8> > a_delay[8][7];
#pragma HLS stream variable=a_delay depth=2

    // delay-line streams for B inputs
    hls::stream<ap_int<8> > b_delay[8][7];
#pragma HLS stream variable=b_delay depth=2

    // ============================================================
    // A input delay chains  (a_in_i  -- i delays --> a_h[i][0])
    // ============================================================

    // Row 0: 0 delays
    delay8(a_in_0, a_h[0][0]);

    // Row 1: 1 delay
    delay8(a_in_1, a_delay[1][0]);
    delay8(a_delay[1][0], a_h[1][0]);

    // Row 2: 2 delays
    delay8(a_in_2, a_delay[2][0]);
    delay8(a_delay[2][0], a_delay[2][1]);
    delay8(a_delay[2][1], a_h[2][0]);

    // Row 3: 3 delays
    delay8(a_in_3, a_delay[3][0]);
    delay8(a_delay[3][0], a_delay[3][1]);
    delay8(a_delay[3][1], a_delay[3][2]);
    delay8(a_delay[3][2], a_h[3][0]);

    // Row 4: 4 delays
    delay8(a_in_4, a_delay[4][0]);
    delay8(a_delay[4][0], a_delay[4][1]);
    delay8(a_delay[4][1], a_delay[4][2]);
    delay8(a_delay[4][2], a_delay[4][3]);
    delay8(a_delay[4][3], a_h[4][0]);

    // Row 5: 5 delays
    delay8(a_in_5, a_delay[5][0]);
    delay8(a_delay[5][0], a_delay[5][1]);
    delay8(a_delay[5][1], a_delay[5][2]);
    delay8(a_delay[5][2], a_delay[5][3]);
    delay8(a_delay[5][3], a_delay[5][4]);
    delay8(a_delay[5][4], a_h[5][0]);

    // Row 6: 6 delays
    delay8(a_in_6, a_delay[6][0]);
    delay8(a_delay[6][0], a_delay[6][1]);
    delay8(a_delay[6][1], a_delay[6][2]);
    delay8(a_delay[6][2], a_delay[6][3]);
    delay8(a_delay[6][3], a_delay[6][4]);
    delay8(a_delay[6][4], a_delay[6][5]);
    delay8(a_delay[6][5], a_h[6][0]);

    // Row 7: 7 delays
    delay8(a_in_7, a_delay[7][0]);
    delay8(a_delay[7][0], a_delay[7][1]);
    delay8(a_delay[7][1], a_delay[7][2]);
    delay8(a_delay[7][2], a_delay[7][3]);
    delay8(a_delay[7][3], a_delay[7][4]);
    delay8(a_delay[7][4], a_delay[7][5]);
    delay8(a_delay[7][5], a_delay[7][6]);
    delay8(a_delay[7][6], a_h[7][0]);

    // ============================================================
    // B input delay chains  (b_in_j  -- j delays --> b_v[0][j])
    // ============================================================

    // Col 0: 0 delays
    delay8(b_in_0, b_v[0][0]);

    // Col 1: 1 delay
    delay8(b_in_1, b_delay[1][0]);
    delay8(b_delay[1][0], b_v[0][1]);

    // Col 2: 2 delays
    delay8(b_in_2, b_delay[2][0]);
    delay8(b_delay[2][0], b_delay[2][1]);
    delay8(b_delay[2][1], b_v[0][2]);

    // Col 3: 3 delays
    delay8(b_in_3, b_delay[3][0]);
    delay8(b_delay[3][0], b_delay[3][1]);
    delay8(b_delay[3][1], b_delay[3][2]);
    delay8(b_delay[3][2], b_v[0][3]);

    // Col 4: 4 delays
    delay8(b_in_4, b_delay[4][0]);
    delay8(b_delay[4][0], b_delay[4][1]);
    delay8(b_delay[4][1], b_delay[4][2]);
    delay8(b_delay[4][2], b_delay[4][3]);
    delay8(b_delay[4][3], b_v[0][4]);

    // Col 5: 5 delays
    delay8(b_in_5, b_delay[5][0]);
    delay8(b_delay[5][0], b_delay[5][1]);
    delay8(b_delay[5][1], b_delay[5][2]);
    delay8(b_delay[5][2], b_delay[5][3]);
    delay8(b_delay[5][3], b_delay[5][4]);
    delay8(b_delay[5][4], b_v[0][5]);

    // Col 6: 6 delays
    delay8(b_in_6, b_delay[6][0]);
    delay8(b_delay[6][0], b_delay[6][1]);
    delay8(b_delay[6][1], b_delay[6][2]);
    delay8(b_delay[6][2], b_delay[6][3]);
    delay8(b_delay[6][3], b_delay[6][4]);
    delay8(b_delay[6][4], b_delay[6][5]);
    delay8(b_delay[6][5], b_v[0][6]);

    // Col 7: 7 delays
    delay8(b_in_7, b_delay[7][0]);
    delay8(b_delay[7][0], b_delay[7][1]);
    delay8(b_delay[7][1], b_delay[7][2]);
    delay8(b_delay[7][2], b_delay[7][3]);
    delay8(b_delay[7][3], b_delay[7][4]);
    delay8(b_delay[7][4], b_delay[7][5]);
    delay8(b_delay[7][5], b_delay[7][6]);
    delay8(b_delay[7][6], b_v[0][7]);

    // ============================================================
    // Dummy readers for a_h[i][8] and b_v[8][j]
    // ============================================================
    dummy8(a_h[0][8]);
    dummy8(a_h[1][8]);
    dummy8(a_h[2][8]);
    dummy8(a_h[3][8]);
    dummy8(a_h[4][8]);
    dummy8(a_h[5][8]);
    dummy8(a_h[6][8]);
    dummy8(a_h[7][8]);

    dummy8(b_v[8][0]);
    dummy8(b_v[8][1]);
    dummy8(b_v[8][2]);
    dummy8(b_v[8][3]);
    dummy8(b_v[8][4]);
    dummy8(b_v[8][5]);
    dummy8(b_v[8][6]);
    dummy8(b_v[8][7]);

    // ============================================================
    // PE grid  (8 x 8)
    // Column 0: pe_left (no c_in)
    // Columns 1..7: pe_rest (with c_in)
    // ============================================================

    // Row 0
    pe_left(     a_h[0][0], a_h[0][1], b_v[0][0], b_v[1][0], c_h[0][0]);
    pe_rest(1, a_h[0][1], a_h[0][2], b_v[0][1], b_v[1][1], c_h[0][0], c_h[0][1]);
    pe_rest(2, a_h[0][2], a_h[0][3], b_v[0][2], b_v[1][2], c_h[0][1], c_h[0][2]);
    pe_rest(3, a_h[0][3], a_h[0][4], b_v[0][3], b_v[1][3], c_h[0][2], c_h[0][3]);
    pe_rest(4, a_h[0][4], a_h[0][5], b_v[0][4], b_v[1][4], c_h[0][3], c_h[0][4]);
    pe_rest(5, a_h[0][5], a_h[0][6], b_v[0][5], b_v[1][5], c_h[0][4], c_h[0][5]);
    pe_rest(6, a_h[0][6], a_h[0][7], b_v[0][6], b_v[1][6], c_h[0][5], c_h[0][6]);
    pe_rest(7, a_h[0][7], a_h[0][8], b_v[0][7], b_v[1][7], c_h[0][6], c_h[0][7]);

    // Row 1
    pe_left(     a_h[1][0], a_h[1][1], b_v[1][0], b_v[2][0], c_h[1][0]);
    pe_rest(1, a_h[1][1], a_h[1][2], b_v[1][1], b_v[2][1], c_h[1][0], c_h[1][1]);
    pe_rest(2, a_h[1][2], a_h[1][3], b_v[1][2], b_v[2][2], c_h[1][1], c_h[1][2]);
    pe_rest(3, a_h[1][3], a_h[1][4], b_v[1][3], b_v[2][3], c_h[1][2], c_h[1][3]);
    pe_rest(4, a_h[1][4], a_h[1][5], b_v[1][4], b_v[2][4], c_h[1][3], c_h[1][4]);
    pe_rest(5, a_h[1][5], a_h[1][6], b_v[1][5], b_v[2][5], c_h[1][4], c_h[1][5]);
    pe_rest(6, a_h[1][6], a_h[1][7], b_v[1][6], b_v[2][6], c_h[1][5], c_h[1][6]);
    pe_rest(7, a_h[1][7], a_h[1][8], b_v[1][7], b_v[2][7], c_h[1][6], c_h[1][7]);

    // Row 2
    pe_left(     a_h[2][0], a_h[2][1], b_v[2][0], b_v[3][0], c_h[2][0]);
    pe_rest(1, a_h[2][1], a_h[2][2], b_v[2][1], b_v[3][1], c_h[2][0], c_h[2][1]);
    pe_rest(2, a_h[2][2], a_h[2][3], b_v[2][2], b_v[3][2], c_h[2][1], c_h[2][2]);
    pe_rest(3, a_h[2][3], a_h[2][4], b_v[2][3], b_v[3][3], c_h[2][2], c_h[2][3]);
    pe_rest(4, a_h[2][4], a_h[2][5], b_v[2][4], b_v[3][4], c_h[2][3], c_h[2][4]);
    pe_rest(5, a_h[2][5], a_h[2][6], b_v[2][5], b_v[3][5], c_h[2][4], c_h[2][5]);
    pe_rest(6, a_h[2][6], a_h[2][7], b_v[2][6], b_v[3][6], c_h[2][5], c_h[2][6]);
    pe_rest(7, a_h[2][7], a_h[2][8], b_v[2][7], b_v[3][7], c_h[2][6], c_h[2][7]);

    // Row 3
    pe_left(     a_h[3][0], a_h[3][1], b_v[3][0], b_v[4][0], c_h[3][0]);
    pe_rest(1, a_h[3][1], a_h[3][2], b_v[3][1], b_v[4][1], c_h[3][0], c_h[3][1]);
    pe_rest(2, a_h[3][2], a_h[3][3], b_v[3][2], b_v[4][2], c_h[3][1], c_h[3][2]);
    pe_rest(3, a_h[3][3], a_h[3][4], b_v[3][3], b_v[4][3], c_h[3][2], c_h[3][3]);
    pe_rest(4, a_h[3][4], a_h[3][5], b_v[3][4], b_v[4][4], c_h[3][3], c_h[3][4]);
    pe_rest(5, a_h[3][5], a_h[3][6], b_v[3][5], b_v[4][5], c_h[3][4], c_h[3][5]);
    pe_rest(6, a_h[3][6], a_h[3][7], b_v[3][6], b_v[4][6], c_h[3][5], c_h[3][6]);
    pe_rest(7, a_h[3][7], a_h[3][8], b_v[3][7], b_v[4][7], c_h[3][6], c_h[3][7]);

    // Row 4
    pe_left(     a_h[4][0], a_h[4][1], b_v[4][0], b_v[5][0], c_h[4][0]);
    pe_rest(1, a_h[4][1], a_h[4][2], b_v[4][1], b_v[5][1], c_h[4][0], c_h[4][1]);
    pe_rest(2, a_h[4][2], a_h[4][3], b_v[4][2], b_v[5][2], c_h[4][1], c_h[4][2]);
    pe_rest(3, a_h[4][3], a_h[4][4], b_v[4][3], b_v[5][3], c_h[4][2], c_h[4][3]);
    pe_rest(4, a_h[4][4], a_h[4][5], b_v[4][4], b_v[5][4], c_h[4][3], c_h[4][4]);
    pe_rest(5, a_h[4][5], a_h[4][6], b_v[4][5], b_v[5][5], c_h[4][4], c_h[4][5]);
    pe_rest(6, a_h[4][6], a_h[4][7], b_v[4][6], b_v[5][6], c_h[4][5], c_h[4][6]);
    pe_rest(7, a_h[4][7], a_h[4][8], b_v[4][7], b_v[5][7], c_h[4][6], c_h[4][7]);

    // Row 5
    pe_left(     a_h[5][0], a_h[5][1], b_v[5][0], b_v[6][0], c_h[5][0]);
    pe_rest(1, a_h[5][1], a_h[5][2], b_v[5][1], b_v[6][1], c_h[5][0], c_h[5][1]);
    pe_rest(2, a_h[5][2], a_h[5][3], b_v[5][2], b_v[6][2], c_h[5][1], c_h[5][2]);
    pe_rest(3, a_h[5][3], a_h[5][4], b_v[5][3], b_v[6][3], c_h[5][2], c_h[5][3]);
    pe_rest(4, a_h[5][4], a_h[5][5], b_v[5][4], b_v[6][4], c_h[5][3], c_h[5][4]);
    pe_rest(5, a_h[5][5], a_h[5][6], b_v[5][5], b_v[6][5], c_h[5][4], c_h[5][5]);
    pe_rest(6, a_h[5][6], a_h[5][7], b_v[5][6], b_v[6][6], c_h[5][5], c_h[5][6]);
    pe_rest(7, a_h[5][7], a_h[5][8], b_v[5][7], b_v[6][7], c_h[5][6], c_h[5][7]);

    // Row 6
    pe_left(     a_h[6][0], a_h[6][1], b_v[6][0], b_v[7][0], c_h[6][0]);
    pe_rest(1, a_h[6][1], a_h[6][2], b_v[6][1], b_v[7][1], c_h[6][0], c_h[6][1]);
    pe_rest(2, a_h[6][2], a_h[6][3], b_v[6][2], b_v[7][2], c_h[6][1], c_h[6][2]);
    pe_rest(3, a_h[6][3], a_h[6][4], b_v[6][3], b_v[7][3], c_h[6][2], c_h[6][3]);
    pe_rest(4, a_h[6][4], a_h[6][5], b_v[6][4], b_v[7][4], c_h[6][3], c_h[6][4]);
    pe_rest(5, a_h[6][5], a_h[6][6], b_v[6][5], b_v[7][5], c_h[6][4], c_h[6][5]);
    pe_rest(6, a_h[6][6], a_h[6][7], b_v[6][6], b_v[7][6], c_h[6][5], c_h[6][6]);
    pe_rest(7, a_h[6][7], a_h[6][8], b_v[6][7], b_v[7][7], c_h[6][6], c_h[6][7]);

    // Row 7
    pe_left(     a_h[7][0], a_h[7][1], b_v[7][0], b_v[8][0], c_h[7][0]);
    pe_rest(1, a_h[7][1], a_h[7][2], b_v[7][1], b_v[8][1], c_h[7][0], c_h[7][1]);
    pe_rest(2, a_h[7][2], a_h[7][3], b_v[7][2], b_v[8][2], c_h[7][1], c_h[7][2]);
    pe_rest(3, a_h[7][3], a_h[7][4], b_v[7][3], b_v[8][3], c_h[7][2], c_h[7][3]);
    pe_rest(4, a_h[7][4], a_h[7][5], b_v[7][4], b_v[8][4], c_h[7][3], c_h[7][4]);
    pe_rest(5, a_h[7][5], a_h[7][6], b_v[7][5], b_v[8][5], c_h[7][4], c_h[7][5]);
    pe_rest(6, a_h[7][6], a_h[7][7], b_v[7][6], b_v[8][6], c_h[7][5], c_h[7][6]);
    pe_rest(7, a_h[7][7], a_h[7][8], b_v[7][7], b_v[8][7], c_h[7][6], c_h[7][7]);

    // ============================================================
    // C output pass-through  (c_h[i][7] -> c_out_i)
    // ============================================================
    pass32(c_h[0][7], c_out_0);
    pass32(c_h[1][7], c_out_1);
    pass32(c_h[2][7], c_out_2);
    pass32(c_h[3][7], c_out_3);
    pass32(c_h[4][7], c_out_4);
    pass32(c_h[5][7], c_out_5);
    pass32(c_h[6][7], c_out_6);
    pass32(c_h[7][7], c_out_7);
}
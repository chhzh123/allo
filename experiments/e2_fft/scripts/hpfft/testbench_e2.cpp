// E2 extended testbench for HP-FFT RTL cosimulation: E2_NT transforms back-to-back through FFT_TOP.
// Transform 0 is the shipped signal; transforms 1..E2_NT-1 are unit-scale uniform random in [-1,1)
// from a fixed-seed LCG. Inputs and outputs are dumped as IEEE-754 hex words for exact numpy checking.
// The design (FFT.cpp/FFT.h) is untouched.
#include "FFT.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#ifndef E2_NT
#define E2_NT 32
#endif
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint32_t lcg = 0x12345678u;
static float unit_rand() { lcg = lcg * 1664525u + 1013904223u; return (float)((lcg >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f; }
static complex<dtype_test> data[E2_NT][FFT_NUM];
static complex<dtype_test> dataFq[E2_NT][FFT_NUM];
int main() {
    for (int i = 0; i < FFT_NUM; i++) {
        float t = static_cast<float>(i) / FFT_NUM;
        float real_part = std::sin(2.0 * M_PI * 10.0 * t) + 0.5 * std::cos(2.0 * M_PI * 50.0 * t);
        float imag_part = std::exp(-5.0 * t) * std::sin(2.0 * M_PI * 20.0 * t);
        data[0][i] = complex<float>(real_part, imag_part);
    }
    for (int b = 1; b < E2_NT; b++)
        for (int i = 0; i < FFT_NUM; i++) { float re = unit_rand(); float im = unit_rand(); data[b][i] = complex<float>(re, im); }
    FILE* fi = fopen("e2_inputs.txt", "w");
    for (int b = 0; b < E2_NT; b++) for (int i = 0; i < FFT_NUM; i++) fprintf(fi, "%08x %08x\n", f2u(data[b][i].real()), f2u(data[b][i].imag()));
    fclose(fi);
    hls::stream<hls::vector<complex<float>, UF*2>> xn_input_strm;
    hls::stream<hls::vector<complex<float>, UF*2>> xk_output_strm;
    for (int b = 0; b < E2_NT; b++) {
        for (int idx = 0; idx < FFT_NUM/(UF*2); idx++) {
            hls::vector<complex<float>, UF*2> temp;
            for (int u = 0; u < UF*2; u++) temp[u] = data[b][idx*UF*2+u];
            xn_input_strm.write(temp);
        }
        FFT_TOP(xn_input_strm, xk_output_strm);
        for (int idx = 0; idx < FFT_NUM/(UF*2); idx++) {
            hls::vector<complex<float>, UF*2> temp = xk_output_strm.read();
            for (int u = 0; u < UF*2; u++) dataFq[b][idx*UF*2+u] = temp[u];
        }
    }
    FILE* fo = fopen("e2_outputs.txt", "w");
    for (int b = 0; b < E2_NT; b++) for (int i = 0; i < FFT_NUM; i++) fprintf(fo, "%08x %08x\n", f2u(dataFq[b][i].real()), f2u(dataFq[b][i].imag()));
    fclose(fo);
    printf("E2 testbench: %d transforms of %d points, UF=%d, %d beats per transform\n", E2_NT, FFT_NUM, UF, FFT_NUM/(UF*2));
    return 0;
}

// E2 seeded testbench for HP-FFT RTL cosimulation: E2_NT transforms back-to-back through FFT_TOP, inputs read from
// the numpy-generated stimulus file E2_STIM (gen_stimulus.py: uniform [-1,1) complex, numpy seeds 0/1/2).
// Inputs and outputs are dumped as IEEE-754 hex words for exact numpy checking. The design (FFT.cpp/FFT.h) is untouched.
#include "FFT.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#ifndef E2_NT
#define E2_NT 33
#endif
#ifndef E2_STIM
#define E2_STIM "/scratch/hc676/e2_hpfft/n1024/UF1_s/e2_stimulus.txt"
#endif
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static complex<dtype_test> data[E2_NT][FFT_NUM];
static complex<dtype_test> dataFq[E2_NT][FFT_NUM];
int main() {
    FILE* fs = fopen(E2_STIM, "r");
    if (!fs) { fprintf(stderr, "E2 ERROR: cannot open stimulus %s\n", E2_STIM); return 1; }
    for (int b = 0; b < E2_NT; b++) for (int i = 0; i < FFT_NUM; i++) {
        unsigned int ru, iu;
        if (fscanf(fs, "%x %x", &ru, &iu) != 2) { fprintf(stderr, "E2 ERROR: stimulus %s too short at transform %d sample %d\n", E2_STIM, b, i); return 1; }
        data[b][i] = complex<float>(u2f(ru), u2f(iu));
    }
    fclose(fs);
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
    printf("E2 seeded testbench: %d transforms of %d points from %s, UF=%d, %d beats per transform\n", E2_NT, FFT_NUM, E2_STIM, UF, FFT_NUM/(UF*2));
    return 0;
}

// E2 seeded testbench for the Allo MachSuite strided FFT (in-place, DIF radix-2): E2_NT transforms, one call each,
// inputs read from the numpy-generated stimulus file E2_STIM (gen_stimulus.py: uniform [-1,1) complex, numpy seeds
// 0/1/2). Twiddles supplied as inputs: W_N^k = exp(-2*pi*j*k/N), k < N/2 (forward transform). Inputs/outputs dumped
// as IEEE-754 hex words for exact numpy validation. The kernel (kernel.cpp) is untouched.
#include "kernel.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#ifndef E2_N
#define E2_N 256
#endif
#ifndef E2_NT
#define E2_NT 6
#endif
#ifndef E2_STIM
#define E2_STIM "e2_stimulus.txt"
#endif
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static float re[E2_NT][E2_N], im[E2_NT][E2_N], twr[E2_N / 2], twi[E2_N / 2];
int main() {
    FILE* fs = fopen(E2_STIM, "r");
    if (!fs) { fprintf(stderr, "E2 ERROR: cannot open stimulus %s\n", E2_STIM); return 1; }
    for (int b = 0; b < E2_NT; b++) for (int i = 0; i < E2_N; i++) {
        unsigned int ru, iu;
        if (fscanf(fs, "%x %x", &ru, &iu) != 2) { fprintf(stderr, "E2 ERROR: stimulus %s too short at transform %d sample %d\n", E2_STIM, b, i); return 1; }
        re[b][i] = u2f(ru); im[b][i] = u2f(iu);
    }
    fclose(fs);
    for (int k = 0; k < E2_N / 2; k++) { double a = -2.0 * M_PI * k / E2_N; twr[k] = (float)cos(a); twi[k] = (float)sin(a); }
    FILE* fi = fopen("e2_inputs.txt", "w");
    for (int b = 0; b < E2_NT; b++) for (int i = 0; i < E2_N; i++) fprintf(fi, "%08x %08x\n", f2u(re[b][i]), f2u(im[b][i]));
    fclose(fi);
    for (int b = 0; b < E2_NT; b++) fft(re[b], im[b], twr, twi);
    FILE* fo = fopen("e2_outputs.txt", "w");
    for (int b = 0; b < E2_NT; b++) for (int i = 0; i < E2_N; i++) fprintf(fo, "%08x %08x\n", f2u(re[b][i]), f2u(im[b][i]));
    fclose(fo);
    printf("E2 allo seeded tb: %d transforms of %d points from %s\n", E2_NT, E2_N, E2_STIM);
    return 0;
}

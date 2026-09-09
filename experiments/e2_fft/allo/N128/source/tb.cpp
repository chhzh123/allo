// E2 testbench for the Allo MachSuite strided FFT (in-place, DIF radix-2): E2_NT transforms, one call each.
// Transform 0 is the HP-FFT shipped signal; others are unit-scale uniform random in [-1,1) (same LCG as the
// HP-FFT E2 testbench). Twiddles supplied as inputs: W_N^k = exp(-2*pi*j*k/N), k < N/2 (forward transform).
// Inputs/outputs dumped as IEEE-754 hex words for exact numpy validation. The kernel (kernel.cpp) is untouched.
#include "kernel.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#ifndef E2_N
#define E2_N 256
#endif
#ifndef E2_NT
#define E2_NT 32
#endif
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint32_t lcg = 0x12345678u;
static float unit_rand() { lcg = lcg * 1664525u + 1013904223u; return (float)((lcg >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f; }
static float re[E2_NT][E2_N], im[E2_NT][E2_N], twr[E2_N / 2], twi[E2_N / 2];
int main() {
    for (int i = 0; i < E2_N; i++) {
        float t = (float)i / E2_N;
        re[0][i] = (float)(std::sin(2.0 * M_PI * 10.0 * t) + 0.5 * std::cos(2.0 * M_PI * 50.0 * t));
        im[0][i] = (float)(std::exp(-5.0 * t) * std::sin(2.0 * M_PI * 20.0 * t));
    }
    for (int b = 1; b < E2_NT; b++) for (int i = 0; i < E2_N; i++) { re[b][i] = unit_rand(); im[b][i] = unit_rand(); }
    for (int k = 0; k < E2_N / 2; k++) { double a = -2.0 * M_PI * k / E2_N; twr[k] = (float)cos(a); twi[k] = (float)sin(a); }
    FILE* fi = fopen("e2_inputs.txt", "w");
    for (int b = 0; b < E2_NT; b++) for (int i = 0; i < E2_N; i++) fprintf(fi, "%08x %08x\n", f2u(re[b][i]), f2u(im[b][i]));
    fclose(fi);
    for (int b = 0; b < E2_NT; b++) fft(re[b], im[b], twr, twi);
    FILE* fo = fopen("e2_outputs.txt", "w");
    for (int b = 0; b < E2_NT; b++) for (int i = 0; i < E2_N; i++) fprintf(fo, "%08x %08x\n", f2u(re[b][i]), f2u(im[b][i]));
    fclose(fo);
    printf("E2 allo tb: %d transforms of %d points\n", E2_NT, E2_N);
    return 0;
}

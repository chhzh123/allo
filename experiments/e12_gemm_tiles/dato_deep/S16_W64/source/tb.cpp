// Testbench for the Dato paper's array (dato_tiles.sh): one launch on the
// packed operands dato_gen.py wrote, every result byte checked against the
// packed numpy reference. Fails, with a non-zero exit, on any mismatch.
#include <ap_int.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kernel.h"

static const int S = 16, W = 64;
typedef ap_uint<128> word_t;

static bool read_file(const char *name, std::vector<unsigned char> &dst) {
  FILE *f = fopen(name, "rb");
  if (!f) { printf("DATO TB ERROR cannot open %s\n", name); return false; }
  size_t got = fread(dst.data(), 1, dst.size(), f);
  bool whole = got == dst.size() && fgetc(f) == EOF;
  fclose(f);
  if (!whole) printf("DATO TB ERROR %s is not %zu bytes\n", name, dst.size());
  return whole;
}

int main() {
  const size_t bytes = (size_t)W * W, words = bytes / S;
  std::vector<unsigned char> a(bytes), b(bytes), ref(bytes);
  if (!read_file("A0.bin", a) || !read_file("B0.bin", b) || !read_file("Cref0.bin", ref)) return 1;
  int amin = 127, amax = -128;
  for (size_t i = 0; i < bytes; i++) {
    int v = (signed char)a[i];
    if (v < amin) amin = v;
    if (v > amax) amax = v;
  }
  printf("DATO TB A[min,max]=[%d,%d]\n", amin, amax);
  std::vector<word_t> A(words), B(words), C(words);
  for (size_t i = 0; i < words; i++) {
    word_t x = 0, y = 0;
    for (int m = 0; m < S; m++) {
      x.range(8 * m + 7, 8 * m) = a[i * S + m];
      y.range(8 * m + 7, 8 * m) = b[i * S + m];
    }
    A[i] = x;
    B[i] = y;
    C[i] = ~word_t(0) / 3;  // poison: a word the design never writes cannot pass
  }
  top(A.data(), B.data(), C.data());
  long long mism = 0;
  for (size_t i = 0; i < words; i++)
    for (int m = 0; m < S; m++)
      if ((unsigned char)C[i].range(8 * m + 7, 8 * m).to_uint() != ref[i * S + m]) mism++;
  printf("DATO TB %s mismatches=%lld of %zu\n", mism == 0 ? "PASS" : "FAIL", mism, bytes);
  return mism == 0 ? 0 : 1;
}

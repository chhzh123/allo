// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FPGA SYCL/oneAPI baseline: folded radix-2 DIT FFT-1024.
//
// One kernel per stage, connected by pipes, with UF butterflies unrolled inside
// each stage -- the same folding the HLS baseline uses. Stage s pairs indices
// that differ in bit s, so the stages below log2(UF) permute within a vector
// and the stages above it permute between vectors; both cases are written out
// because the compiler cannot infer the distinction from a loop nest.
//
// NOTE: unbuildable on the evaluation machine -- oneAPI 2026.1 removed
// -fintelfpga and ships no FPGA extension headers. Included for Table 5's line
// count only.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <cmath>
#include <complex>

constexpr int N = 1024;
constexpr int LOGN = 10;
constexpr int UF = 32;    // butterflies unrolled per stage
constexpr int VEC = 2 * UF; // points held in one vector beat
constexpr int BEATS = N / VEC;

struct cplx {
  float re;
  float im;
};

struct beat {
  cplx v[VEC];
};

template <int S> class StageId;
template <int S> using StagePipe = sycl::ext::intel::pipe<StageId<S>, beat, 4>;

template <int S> class StageK;

static inline cplx mul(cplx a, cplx b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
static inline cplx add(cplx a, cplx b) { return {a.re + b.re, a.im + b.im}; }
static inline cplx sub(cplx a, cplx b) { return {a.re - b.re, a.im - b.im}; }

// Twiddle for stage s, butterfly index b.
static inline cplx twiddle(int s, int b) {
  const int span = 1 << s;
  const float angle = -2.0f * 3.14159265358979f * float(b % span) / float(2 * span);
  return {std::cos(angle), std::sin(angle)};
}

// A stage whose butterfly partners lie inside one vector beat.
template <int S> struct IntraStage {
  static void submit(sycl::queue &q) {
    q.single_task<StageK<S>>([=] {
      constexpr int span = 1 << S;
      for (int t = 0; t < BEATS; t++) {
        beat in = StagePipe<S>::read();
        beat out;
#pragma unroll
        for (int b = 0; b < UF; b++) {
          const int lo = ((b / span) * 2 * span) + (b % span);
          const int hi = lo + span;
          cplx w = twiddle(S, b);
          cplx prod = mul(in.v[hi], w);
          out.v[lo] = add(in.v[lo], prod);
          out.v[hi] = sub(in.v[lo], prod);
        }
        StagePipe<S + 1>::write(out);
      }
    });
  }
};

// A stage whose butterfly partners lie in different beats: the beat is held
// until its partner arrives, so the stage carries a buffer the intra stages do
// not need.
template <int S> struct InterStage {
  static void submit(sycl::queue &q) {
    q.single_task<StageK<S>>([=] {
      constexpr int span = 1 << S;
      constexpr int stride = span / VEC;
      [[intel::fpga_memory]] beat hold[BEATS];
      for (int t = 0; t < BEATS; t++)
        hold[t] = StagePipe<S>::read();
      for (int t = 0; t < BEATS; t++) {
        const int partner = t ^ stride;
        const bool lower = (t & stride) == 0;
        beat mine = hold[t], other = hold[partner];
        beat out;
#pragma unroll
        for (int i = 0; i < VEC; i++) {
          cplx w = twiddle(S, t * VEC + i);
          cplx prod = mul(other.v[i], w);
          out.v[i] = lower ? add(mine.v[i], prod) : sub(other.v[i], mul(mine.v[i], w));
        }
        StagePipe<S + 1>::write(out);
      }
    });
  }
};

// Pick the right stage kind at compile time and chain them.
template <int S> struct Stage {
  static void submit(sycl::queue &q) {
    if constexpr ((1 << S) < VEC)
      IntraStage<S>::submit(q);
    else
      InterStage<S>::submit(q);
    Stage<S - 1>::submit(q);
  }
};
template <> struct Stage<-1> {
  static void submit(sycl::queue &) {}
};

// Bit-reversal on the way in, and the drain on the way out.
static void feed(sycl::queue &q, const cplx *x) {
  q.single_task([=] {
    for (int t = 0; t < BEATS; t++) {
      beat out;
#pragma unroll
      for (int i = 0; i < VEC; i++) {
        int index = t * VEC + i, rev = 0;
#pragma unroll
        for (int b = 0; b < LOGN; b++)
          rev |= ((index >> b) & 1) << (LOGN - 1 - b);
        out.v[i] = x[rev];
      }
      StagePipe<0>::write(out);
    }
  });
}

static void drain(sycl::queue &q, cplx *y) {
  q.single_task([=] {
    for (int t = 0; t < BEATS; t++) {
      beat in = StagePipe<LOGN>::read();
#pragma unroll
      for (int i = 0; i < VEC; i++)
        y[t * VEC + i] = in.v[i];
    }
  });
}

int main() {
#if defined(FPGA_EMULATOR)
  sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
#else
  sycl::queue q{sycl::ext::intel::fpga_selector_v};
#endif

  auto *x = sycl::malloc_shared<cplx>(N, q);
  auto *y = sycl::malloc_shared<cplx>(N, q);
  for (int i = 0; i < N; i++)
    x[i] = {float(i % 17) - 8.0f, float(i % 11) - 5.0f};

  feed(q, x);
  Stage<LOGN - 1>::submit(q);
  drain(q, y);
  q.wait();

  sycl::free(x, q);
  sycl::free(y, q);
  return 0;
}

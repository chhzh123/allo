// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FPGA SYCL/oneAPI baseline: grouped attention-PV.
//
// The P.V GEMM on a DIMxDIM weight-stationary array cut into GROUPS column
// slabs, with the partial-sum chain serpentining between slabs. GROUPS is a
// compile-time constant because the pipe topology depends on it.
//
// NOTE: unbuildable on the evaluation machine -- oneAPI 2026.1 removed
// -fintelfpga and ships no FPGA extension headers. Included for Table 5's line
// count only.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <cstdint>

constexpr int DIM = 16;
#ifndef GROUPS
#define GROUPS 1
#endif
constexpr int G = GROUPS;
constexpr int SLAB = DIM / G; // output columns per slab == head dimension
constexpr int SHIFT = 2;

using data_t = int8_t;
using acc_t = int32_t;

template <int C, int R> class AId;
template <int R, int C> class PId;

template <int C, int R>
using APipe = sycl::ext::intel::pipe<AId<C, R>, data_t, 4>;
template <int R, int C>
using PPipe = sycl::ext::intel::pipe<PId<R, C>, acc_t, 4>;

template <int R, int C> class PEK;

// The psum destination is the next row, or the top of the next slab at the
// bottom edge. Both are compile-time decisions, so they are template branches.
template <int R, int C, bool LastRow, bool HasNextSlab> struct PsumTo;

template <int R, int C> struct PsumTo<R, C, false, false> {
  static void write(acc_t v) { PPipe<R + 1, C>::write(v); }
};
template <int R, int C> struct PsumTo<R, C, false, true> {
  static void write(acc_t v) { PPipe<R + 1, C>::write(v); }
};
template <int R, int C> struct PsumTo<R, C, true, true> {
  static void write(acc_t v) { PPipe<0, C + SLAB>::write(v); }
};
template <int R, int C> struct PsumTo<R, C, true, false> {
  static void write(acc_t v) { PPipe<DIM, C>::write(v); }
};

template <int R, int C> struct PE {
  static constexpr bool last_row = (R + 1 == DIM);
  static constexpr bool next_slab = (C + SLAB < DIM);
  static constexpr bool forward_a = ((C + 1) % SLAB) != 0;

  static void submit(sycl::queue &q, const data_t *w, int steps) {
    q.single_task<PEK<R, C>>([=] {
      data_t weight = w[R * DIM + C];
      for (int m = 0; m < steps; m++) {
        data_t a = APipe<C, R>::read();
        acc_t p = PPipe<R, C>::read();
        PsumTo<R, C, last_row, next_slab>::write(p + acc_t(a) * acc_t(weight));
        if (forward_a)
          APipe<C + 1, R>::write(a);
      }
    });
  }
};

template <int R, int C> struct Row {
  static void submit(sycl::queue &q, const data_t *w, int steps) {
    PE<R, C>::submit(q, w, steps);
    Row<R, C - 1>::submit(q, w, steps);
  }
};
template <int R> struct Row<R, -1> {
  static void submit(sycl::queue &, const data_t *, int) {}
};

template <int R> struct Grid {
  static void submit(sycl::queue &q, const data_t *w, int steps) {
    Row<R, DIM - 1>::submit(q, w, steps);
    Grid<R - 1>::submit(q, w, steps);
  }
};
template <> struct Grid<-1> {
  static void submit(sycl::queue &, const data_t *, int) {}
};

// Every slab's west column is fed, so there are G * DIM feeds.
template <int I> struct FeedA {
  static void submit(sycl::queue &q, const data_t *pr, int steps) {
    constexpr int slab = I / DIM;
    constexpr int row = I % DIM;
    q.single_task([=] {
      for (int m = 0; m < steps; m++)
        APipe<slab * SLAB, row>::write(pr[m * G * DIM + I]);
    });
    FeedA<I - 1>::submit(q, pr, steps);
  }
};
template <> struct FeedA<-1> {
  static void submit(sycl::queue &, const data_t *, int) {}
};

// Only slab 0's top row is seeded.
template <int C> struct SeedP {
  static void submit(sycl::queue &q, int steps) {
    q.single_task([=] {
      for (int m = 0; m < steps; m++)
        PPipe<0, C>::write(0);
    });
    SeedP<C - 1>::submit(q, steps);
  }
};
template <> struct SeedP<-1> {
  static void submit(sycl::queue &, int) {}
};

// Only the last slab's bottom row drains, through a SLAB-wide activation unit.
template <int C> struct Drain {
  static void submit(sycl::queue &q, data_t *y, int steps) {
    constexpr int col = DIM - SLAB + C;
    q.single_task([=] {
      for (int m = 0; m < steps; m++) {
        acc_t z = PPipe<DIM, col>::read();
        acc_t r = z > 0 ? z : 0;
        y[m * SLAB + C] = data_t(r >> SHIFT);
      }
    });
    Drain<C - 1>::submit(q, y, steps);
  }
};
template <> struct Drain<-1> {
  static void submit(sycl::queue &, data_t *, int) {}
};

int main(int argc, char **argv) {
  const int steps = argc > 1 ? std::atoi(argv[1]) : 64;

#if defined(FPGA_EMULATOR)
  sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
#else
  sycl::queue q{sycl::ext::intel::fpga_selector_v};
#endif

  auto *pr = sycl::malloc_shared<data_t>(steps * G * DIM, q);
  auto *v = sycl::malloc_shared<data_t>(DIM * DIM, q);
  auto *y = sycl::malloc_shared<data_t>(steps * SLAB, q);
  for (int i = 0; i < steps * G * DIM; i++)
    pr[i] = data_t((i % 7) - 3);
  for (int i = 0; i < DIM * DIM; i++)
    v[i] = data_t((i % 7) - 3);

  FeedA<G * DIM - 1>::submit(q, pr, steps);
  SeedP<SLAB - 1>::submit(q, steps);
  Grid<DIM - 1>::submit(q, v, steps);
  Drain<SLAB - 1>::submit(q, y, steps);
  q.wait();

  sycl::free(pr, q);
  sycl::free(v, q);
  sycl::free(y, q);
  return 0;
}

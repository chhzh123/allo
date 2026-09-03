// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FPGA SYCL/oneAPI baseline: daisy-chain multi-cache GEMM.
//
// Same architecture as the Vitis HLS multi-cache baseline: a DIMxDIM
// output-stationary mesh whose results leave down a chain of column vectors
// rather than through a port per PE. int16 throughout.
//
// NOTE: unbuildable on the evaluation machine -- oneAPI 2026.1 removed
// -fintelfpga and ships no FPGA extension headers. Included for Table 5's line
// count only.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <cstdint>

constexpr int DIM = 16;

using data_t = int16_t;

struct column_t {
  data_t v[DIM];
};

template <int C, int R> class AId;
template <int R, int C> class BId;
template <int R, int C> class CId;

template <int C, int R>
using APipe = sycl::ext::intel::pipe<AId<C, R>, data_t, 4>;
template <int R, int C>
using BPipe = sycl::ext::intel::pipe<BId<R, C>, data_t, 4>;
template <int R, int C>
using CPipe = sycl::ext::intel::pipe<CId<R, C>, column_t, 4>;

template <int R, int C> class PEK;

// One PE: accumulate over K, forward both operands, then splice the
// accumulator into the column passing through.
template <int R, int C> struct PE {
  static void submit(sycl::queue &q, int steps) {
    q.single_task<PEK<R, C>>([=] {
      data_t acc = 0;
      for (int k = 0; k < steps; k++) {
        data_t a = APipe<C, R>::read();
        data_t b = BPipe<R, C>::read();
        acc += a * b;
        APipe<C + 1, R>::write(a);
        BPipe<R + 1, C>::write(b);
      }
      column_t col = CPipe<R, C>::read();
      col.v[R] = acc;
      CPipe<R + 1, C>::write(col);
    });
  }
};

template <int R, int C> struct Row {
  static void submit(sycl::queue &q, int steps) {
    PE<R, C>::submit(q, steps);
    Row<R, C - 1>::submit(q, steps);
  }
};
template <int R> struct Row<R, -1> {
  static void submit(sycl::queue &, int) {}
};

template <int R> struct Grid {
  static void submit(sycl::queue &q, int steps) {
    Row<R, DIM - 1>::submit(q, steps);
    Grid<R - 1>::submit(q, steps);
  }
};
template <> struct Grid<-1> {
  static void submit(sycl::queue &, int) {}
};

// Edge kernels.
template <int R> struct FeedWest {
  static void submit(sycl::queue &q, const data_t *a, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        APipe<0, R>::write(a[R * steps + k]);
    });
    FeedWest<R - 1>::submit(q, a, steps);
  }
};
template <> struct FeedWest<-1> {
  static void submit(sycl::queue &, const data_t *, int) {}
};

template <int C> struct FeedNorth {
  static void submit(sycl::queue &q, const data_t *b, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        BPipe<0, C>::write(b[k * DIM + C]);
    });
    FeedNorth<C - 1>::submit(q, b, steps);
  }
};
template <> struct FeedNorth<-1> {
  static void submit(sycl::queue &, const data_t *, int) {}
};

template <int R> struct SinkEast {
  static void submit(sycl::queue &q, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        (void)APipe<DIM, R>::read();
    });
    SinkEast<R - 1>::submit(q, steps);
  }
};
template <> struct SinkEast<-1> {
  static void submit(sycl::queue &, int) {}
};

template <int C> struct SinkSouth {
  static void submit(sycl::queue &q, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        (void)BPipe<DIM, C>::read();
    });
    SinkSouth<C - 1>::submit(q, steps);
  }
};
template <> struct SinkSouth<-1> {
  static void submit(sycl::queue &, int) {}
};

template <int C> struct Chain {
  static void submit(sycl::queue &q, data_t *c) {
    q.single_task([=] {
      column_t zero;
      for (int i = 0; i < DIM; i++)
        zero.v[i] = 0;
      CPipe<0, C>::write(zero);
    });
    q.single_task([=] {
      column_t col = CPipe<DIM, C>::read();
      for (int r = 0; r < DIM; r++)
        c[r * DIM + C] = col.v[r];
    });
    Chain<C - 1>::submit(q, c);
  }
};
template <> struct Chain<-1> {
  static void submit(sycl::queue &, data_t *) {}
};

int main(int argc, char **argv) {
  const int steps = argc > 1 ? std::atoi(argv[1]) : DIM;

#if defined(FPGA_EMULATOR)
  sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
#else
  sycl::queue q{sycl::ext::intel::fpga_selector_v};
#endif

  auto *a = sycl::malloc_shared<data_t>(DIM * steps, q);
  auto *b = sycl::malloc_shared<data_t>(steps * DIM, q);
  auto *c = sycl::malloc_shared<data_t>(DIM * DIM, q);
  for (int i = 0; i < DIM * steps; i++)
    a[i] = data_t((i % 15) - 7);
  for (int i = 0; i < steps * DIM; i++)
    b[i] = data_t((i % 13) - 6);

  FeedWest<DIM - 1>::submit(q, a, steps);
  FeedNorth<DIM - 1>::submit(q, b, steps);
  Chain<DIM - 1>::submit(q, c);
  Grid<DIM - 1>::submit(q, steps);
  SinkEast<DIM - 1>::submit(q, steps);
  SinkSouth<DIM - 1>::submit(q, steps);
  q.wait();

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return 0;
}

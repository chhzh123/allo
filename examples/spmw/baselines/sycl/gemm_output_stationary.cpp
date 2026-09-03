// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FPGA SYCL/oneAPI baseline: the plain systolic GEMM.
//
// A DIMxDIM output-stationary mesh; operands walk east and south, each PE
// accumulates over K and writes its own element of C. int8 into int32.
//
// NOTE: unbuildable on the evaluation machine -- oneAPI 2026.1 removed
// -fintelfpga and ships no FPGA extension headers. Included for Table 5's line
// count only.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <cstdint>

constexpr int DIM = 16;

using data_t = int8_t;
using acc_t = int32_t;

template <int C, int R> class AId;
template <int R, int C> class BId;

template <int C, int R>
using APipe = sycl::ext::intel::pipe<AId<C, R>, data_t, 4>;
template <int R, int C>
using BPipe = sycl::ext::intel::pipe<BId<R, C>, data_t, 4>;

template <int R, int C> class PEK;

// One PE: accumulate over K, forward both operands, write one element of C.
template <int R, int C> struct PE {
  static void submit(sycl::queue &q, acc_t *c, int steps) {
    q.single_task<PEK<R, C>>([=] {
      acc_t acc = 0;
      for (int k = 0; k < steps; k++) {
        data_t a = APipe<C, R>::read();
        data_t b = BPipe<R, C>::read();
        acc += acc_t(a) * acc_t(b);
        APipe<C + 1, R>::write(a);
        BPipe<R + 1, C>::write(b);
      }
      c[R * DIM + C] = acc;
    });
  }
};

template <int R, int C> struct Row {
  static void submit(sycl::queue &q, acc_t *c, int steps) {
    PE<R, C>::submit(q, c, steps);
    Row<R, C - 1>::submit(q, c, steps);
  }
};
template <int R> struct Row<R, -1> {
  static void submit(sycl::queue &, acc_t *, int) {}
};

template <int R> struct Grid {
  static void submit(sycl::queue &q, acc_t *c, int steps) {
    Row<R, DIM - 1>::submit(q, c, steps);
    Grid<R - 1>::submit(q, c, steps);
  }
};
template <> struct Grid<-1> {
  static void submit(sycl::queue &, acc_t *, int) {}
};

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

int main(int argc, char **argv) {
  const int steps = argc > 1 ? std::atoi(argv[1]) : DIM;

#if defined(FPGA_EMULATOR)
  sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
#else
  sycl::queue q{sycl::ext::intel::fpga_selector_v};
#endif

  auto *a = sycl::malloc_shared<data_t>(DIM * steps, q);
  auto *b = sycl::malloc_shared<data_t>(steps * DIM, q);
  auto *c = sycl::malloc_shared<acc_t>(DIM * DIM, q);
  for (int i = 0; i < DIM * steps; i++)
    a[i] = data_t((i % 15) - 7);
  for (int i = 0; i < steps * DIM; i++)
    b[i] = data_t((i % 13) - 6);

  FeedWest<DIM - 1>::submit(q, a, steps);
  FeedNorth<DIM - 1>::submit(q, b, steps);
  Grid<DIM - 1>::submit(q, c, steps);
  SinkEast<DIM - 1>::submit(q, steps);
  SinkSouth<DIM - 1>::submit(q, steps);
  q.wait();

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return 0;
}

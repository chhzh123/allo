// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FPGA SYCL/oneAPI baseline: 16x16 weight-stationary int8 systolic GEMM.
//
// Functionally identical to the Vitis HLS baseline and to the SPMW mini-TPU
// MXU: int8 operands into an int32 accumulator, then ReLU and an arithmetic
// shift back to int8. Written in the idiom oneAPI actually uses for spatial
// designs -- one kernel per PE, connected by pipes, with the array unrolled by
// template recursion at submit time.
//
// NOTE: this compiles and runs under the oneAPI FPGA emulator, but it cannot be
// synthesised for the Alveo U280. The oneAPI FPGA backend targets Intel
// devices, and the install on the evaluation machine carries no FPGA backend at
// all. It is included for the line-count comparison in Table 5; the throughput
// comparison in Fig. 10 cannot include a SYCL series on this hardware.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <array>
#include <cstdint>
#include <vector>

constexpr int DIM = 16;
constexpr int SHIFT = 6;

using data_t = int8_t;
using acc_t = int32_t;

// Pipe identities. A pipe per array edge, indexed by the two array axes: `A`
// carries activations east, `P` carries partial sums south.
template <int C, int R> class AId;
template <int R, int C> class PId;

template <int C, int R>
using APipe = sycl::ext::intel::pipe<AId<C, R>, data_t, 4>;
template <int R, int C>
using PPipe = sycl::ext::intel::pipe<PId<R, C>, acc_t, 4>;

// Kernel name tags.
template <int R, int C> class PEK;
class FeedA;
class SeedP;
class SinkA;
class Drain;

// One processing element: the weight is stationary, activations walk east and
// partial sums walk south.
template <int R, int C> struct PE {
  static void submit(sycl::queue &q, data_t w, int steps) {
    q.single_task<PEK<R, C>>([=] {
      for (int k = 0; k < steps; k++) {
        data_t a = APipe<C, R>::read();
        acc_t p = PPipe<R, C>::read();
        PPipe<R + 1, C>::write(p + acc_t(a) * acc_t(w));
        APipe<C + 1, R>::write(a);
      }
    });
  }
};

// Unroll the grid at submit time: one kernel per site, as the array demands.
template <int R, int C> struct Row {
  static void submit(sycl::queue &q, const data_t *w, int steps) {
    PE<R, C>::submit(q, w[R * DIM + C], steps);
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

// Edge kernels: feed the west column, seed the north row with zeros, sink the
// east column, and drain the south row through the 16-lane activation unit.
template <int R> struct FeedRow {
  static void submit(sycl::queue &q, const data_t *a, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        APipe<0, R>::write(a[k * DIM + R]);
    });
    FeedRow<R - 1>::submit(q, a, steps);
  }
};
template <> struct FeedRow<-1> {
  static void submit(sycl::queue &, const data_t *, int) {}
};

template <int C> struct SeedCol {
  static void submit(sycl::queue &q, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        PPipe<0, C>::write(0);
    });
    SeedCol<C - 1>::submit(q, steps);
  }
};
template <> struct SeedCol<-1> {
  static void submit(sycl::queue &, int) {}
};

template <int R> struct SinkRow {
  static void submit(sycl::queue &q, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        (void)APipe<DIM, R>::read();
    });
    SinkRow<R - 1>::submit(q, steps);
  }
};
template <> struct SinkRow<-1> {
  static void submit(sycl::queue &, int) {}
};

template <int C> struct DrainCol {
  static void submit(sycl::queue &q, data_t *y, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++) {
        acc_t z = PPipe<DIM, C>::read();
        acc_t r = z > 0 ? z : 0;
        y[k * DIM + C] = data_t(r >> SHIFT);
      }
    });
    DrainCol<C - 1>::submit(q, y, steps);
  }
};
template <> struct DrainCol<-1> {
  static void submit(sycl::queue &, data_t *, int) {}
};

// One tile through the array.
void gemm_tile(sycl::queue &q, const data_t *a, const data_t *w, data_t *y,
               int steps) {
  FeedRow<DIM - 1>::submit(q, a, steps);
  SeedCol<DIM - 1>::submit(q, steps);
  Grid<DIM - 1>::submit(q, w, steps);
  SinkRow<DIM - 1>::submit(q, steps);
  DrainCol<DIM - 1>::submit(q, y, steps);
  q.wait();
}

int main(int argc, char **argv) {
  const int steps = argc > 1 ? std::atoi(argv[1]) : 16;
  const int tiles = argc > 2 ? std::atoi(argv[2]) : 1;

#if defined(FPGA_EMULATOR)
  sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
#else
  sycl::queue q{sycl::ext::intel::fpga_selector_v};
#endif

  auto *a = sycl::malloc_shared<data_t>(steps * DIM, q);
  auto *w = sycl::malloc_shared<data_t>(DIM * DIM, q);
  auto *y = sycl::malloc_shared<data_t>(steps * DIM, q);

  for (int i = 0; i < steps * DIM; i++)
    a[i] = data_t((i % 15) - 7);
  for (int i = 0; i < DIM * DIM; i++)
    w[i] = data_t((i % 13) - 6);

  for (int t = 0; t < tiles; t++)
    gemm_tile(q, a, w, y, steps);

  sycl::free(a, q);
  sycl::free(w, q);
  sycl::free(y, q);
  return 0;
}

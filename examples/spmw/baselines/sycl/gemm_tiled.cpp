// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FPGA SYCL/oneAPI baseline: two-level hierarchical tiled GEMM.
//
// A TILES x TILES grid of tile engines, each an inner PE x PE output-stationary
// mesh, over float. The pipe identities carry the tile index as well as the
// position inside the tile, because two levels of structure have to be
// flattened into one namespace of pipes.
//
// NOTE: unbuildable on the evaluation machine -- oneAPI 2026.1 removed
// -fintelfpga and ships no FPGA extension headers. Included for Table 5's line
// count only.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

constexpr int TILES = 2;
constexpr int PE = 8;
constexpr int SIDE = TILES * PE;

using data_t = float;

template <int T, int C, int R> class AId;
template <int T, int R, int C> class BId;

template <int T, int C, int R>
using APipe = sycl::ext::intel::pipe<AId<T, C, R>, data_t, 4>;
template <int T, int R, int C>
using BPipe = sycl::ext::intel::pipe<BId<T, R, C>, data_t, 4>;

template <int T, int R, int C> class PEK;

// One PE of one tile engine.
template <int T, int R, int C> struct Cell {
  static void submit(sycl::queue &q, data_t *c, int steps, int ti, int tj) {
    q.single_task<PEK<T, R, C>>([=] {
      data_t acc = 0;
      for (int k = 0; k < steps; k++) {
        data_t a = APipe<T, C, R>::read();
        data_t b = BPipe<T, R, C>::read();
        acc += a * b;
        APipe<T, C + 1, R>::write(a);
        BPipe<T, R + 1, C>::write(b);
      }
      c[(ti * PE + R) * SIDE + tj * PE + C] = acc;
    });
  }
};

template <int T, int R, int C> struct Row {
  static void submit(sycl::queue &q, data_t *c, int steps, int ti, int tj) {
    Cell<T, R, C>::submit(q, c, steps, ti, tj);
    Row<T, R, C - 1>::submit(q, c, steps, ti, tj);
  }
};
template <int T, int R> struct Row<T, R, -1> {
  static void submit(sycl::queue &, data_t *, int, int, int) {}
};

template <int T, int R> struct Mesh {
  static void submit(sycl::queue &q, data_t *c, int steps, int ti, int tj) {
    Row<T, R, PE - 1>::submit(q, c, steps, ti, tj);
    Mesh<T, R - 1>::submit(q, c, steps, ti, tj);
  }
};
template <int T> struct Mesh<T, -1> {
  static void submit(sycl::queue &, data_t *, int, int, int) {}
};

// Edges of one tile engine.
template <int T, int R> struct FeedWest {
  static void submit(sycl::queue &q, const data_t *a, int steps, int ti) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        APipe<T, 0, R>::write(a[(ti * PE + R) * steps + k]);
    });
    FeedWest<T, R - 1>::submit(q, a, steps, ti);
  }
};
template <int T> struct FeedWest<T, -1> {
  static void submit(sycl::queue &, const data_t *, int, int) {}
};

template <int T, int C> struct FeedNorth {
  static void submit(sycl::queue &q, const data_t *b, int steps, int tj) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        BPipe<T, 0, C>::write(b[k * SIDE + tj * PE + C]);
    });
    FeedNorth<T, C - 1>::submit(q, b, steps, tj);
  }
};
template <int T> struct FeedNorth<T, -1> {
  static void submit(sycl::queue &, const data_t *, int, int) {}
};

template <int T, int R> struct SinkEast {
  static void submit(sycl::queue &q, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        (void)APipe<T, PE, R>::read();
    });
    SinkEast<T, R - 1>::submit(q, steps);
  }
};
template <int T> struct SinkEast<T, -1> {
  static void submit(sycl::queue &, int) {}
};

template <int T, int C> struct SinkSouth {
  static void submit(sycl::queue &q, int steps) {
    q.single_task([=] {
      for (int k = 0; k < steps; k++)
        (void)BPipe<T, PE, C>::read();
    });
    SinkSouth<T, C - 1>::submit(q, steps);
  }
};
template <int T> struct SinkSouth<T, -1> {
  static void submit(sycl::queue &, int) {}
};

// One tile engine, and then the grid of them.
template <int T> struct Engine {
  static void submit(sycl::queue &q, const data_t *a, const data_t *b,
                     data_t *c, int steps) {
    constexpr int ti = T / TILES;
    constexpr int tj = T % TILES;
    FeedWest<T, PE - 1>::submit(q, a, steps, ti);
    FeedNorth<T, PE - 1>::submit(q, b, steps, tj);
    Mesh<T, PE - 1>::submit(q, c, steps, ti, tj);
    SinkEast<T, PE - 1>::submit(q, steps);
    SinkSouth<T, PE - 1>::submit(q, steps);
    Engine<T - 1>::submit(q, a, b, c, steps);
  }
};
template <> struct Engine<-1> {
  static void submit(sycl::queue &, const data_t *, const data_t *, data_t *,
                     int) {}
};

int main(int argc, char **argv) {
  const int steps = argc > 1 ? std::atoi(argv[1]) : 16;

#if defined(FPGA_EMULATOR)
  sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
#else
  sycl::queue q{sycl::ext::intel::fpga_selector_v};
#endif

  auto *a = sycl::malloc_shared<data_t>(SIDE * steps, q);
  auto *b = sycl::malloc_shared<data_t>(steps * SIDE, q);
  auto *c = sycl::malloc_shared<data_t>(SIDE * SIDE, q);
  for (int i = 0; i < SIDE * steps; i++)
    a[i] = data_t(i % 9) - 4.0f;
  for (int i = 0; i < steps * SIDE; i++)
    b[i] = data_t(i % 7) - 3.0f;

  Engine<TILES * TILES - 1>::submit(q, a, b, c, steps);
  q.wait();

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return 0;
}

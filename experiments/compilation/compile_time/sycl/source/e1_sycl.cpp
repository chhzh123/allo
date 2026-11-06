// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// E1's array in SYCL for an Intel FPGA: C = A B with int8 operands and int32
// results, M = N = K = S, on S x S cells of one multiplier each. As AutoSA
// builds it, every cell is a kernel of its own: it takes A from the west and
// B from the north, passes both on, and after its K steps hands its sum down
// its column, then the sums of the rows above it. One kernel loads A, one
// loads B and one stores C; all of them talk through pipes.
//
//   icpx -fsycl -fintelfpga -DSIDE=16 -DFPGA_EMULATOR e1_sycl.cpp     # check it
//   icpx -fsycl -fintelfpga -DSIDE=16 -Xshardware -fsycl-link=early \
//        -Xstarget=Agilex7 -Xsclock=300MHz e1_sycl.cpp -o e1.a        # to RTL
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#ifndef SIDE
#define SIDE 4
#endif

using namespace sycl;

// Not a macro: the SYCL headers name template parameters S.
constexpr int S = SIDE;

// f(i) for i = 0 .. N-1 with i a compile-time constant, so that it can pick
// a pipe.
template <typename F, int... Is>
void unrolled_impl(F &&f, std::integer_sequence<int, Is...>) {
  (f(std::integral_constant<int, Is>{}), ...);
}
template <int N, typename F> void unrolled(F &&f) {
  unrolled_impl(f, std::make_integer_sequence<int, N>{});
}

template <int I, int J> class APipeId;
template <int I, int J> class BPipeId;
template <int I, int J> class CPipeId;
// Into cell (I, J): A from the west, B from the north, and the sums of the
// rows above I from the cell above. AutoSA's FIFOs are two deep.
template <int I, int J> using APipe = ext::intel::pipe<APipeId<I, J>, int8_t, 2>;
template <int I, int J> using BPipe = ext::intel::pipe<BPipeId<I, J>, int8_t, 2>;
template <int I, int J> using CPipe = ext::intel::pipe<CPipeId<I, J>, int32_t, 2>;

template <int I, int J> class Cell;
class LoadA;
class LoadB;
class StoreC;

template <int I, int J> void cell(queue &q) {
  q.single_task<Cell<I, J>>([=]() {
    int32_t acc = 0;
    for (int k = 0; k < S; k++) {
      int8_t a = APipe<I, J>::read();
      int8_t b = BPipe<I, J>::read();
      if constexpr (J + 1 < S)
        APipe<I, J + 1>::write(a);
      if constexpr (I + 1 < S)
        BPipe<I + 1, J>::write(b);
      acc += int32_t(a) * int32_t(b);
    }
    CPipe<I + 1, J>::write(acc);
    if constexpr (I > 0) {
      for (int r = 0; r < I; r++)
        CPipe<I + 1, J>::write(CPipe<I, J>::read());
    }
  });
}

template <int I, int... Js>
void cells_of_row(queue &q, std::integer_sequence<int, Js...>) {
  (cell<I, Js>(q), ...);
}
template <int... Is> void all_cells(queue &q, std::integer_sequence<int, Is...>) {
  (cells_of_row<Is>(q, std::make_integer_sequence<int, S>{}), ...);
}

int main() {
#ifdef FPGA_EMULATOR
  queue q(ext::intel::fpga_emulator_selector_v);
#else
  queue q(ext::intel::fpga_selector_v);
#endif
  std::vector<int8_t> a(S * S), b(S * S);
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> any(-128, 127);
  for (auto &x : a)
    x = int8_t(any(rng));
  for (auto &x : b)
    x = int8_t(any(rng));
  a[0] = -128, a[S * S - 1] = 127, b[0] = 127, b[S * S - 1] = -128;

  int8_t *da = malloc_device<int8_t>(S * S, q);
  int8_t *db = malloc_device<int8_t>(S * S, q);
  int32_t *dc = malloc_device<int32_t>(S * S, q);
  q.memcpy(da, a.data(), S * S).wait();
  q.memcpy(db, b.data(), S * S).wait();

  // A row-major [i][k]: row i goes into the west pipe of row i, a step a k.
  q.single_task<LoadA>([=]() [[intel::kernel_args_restrict]] {
    int8_t buf[S][S];
    for (int x = 0; x < S * S; x++)
      buf[x / S][x % S] = da[x];
    for (int k = 0; k < S; k++)
      unrolled<S>([&](auto i) { APipe<i, 0>::write(buf[i][k]); });
  });
  // B row-major [k][j]: column j goes into the north pipe of column j.
  q.single_task<LoadB>([=]() [[intel::kernel_args_restrict]] {
    int8_t buf[S][S];
    for (int x = 0; x < S * S; x++)
      buf[x / S][x % S] = db[x];
    for (int k = 0; k < S; k++)
      unrolled<S>([&](auto j) { BPipe<0, j>::write(buf[k][j]); });
  });
  all_cells(q, std::make_integer_sequence<int, S>{});
  // A column's sums arrive bottom row first.
  q.single_task<StoreC>([=]() [[intel::kernel_args_restrict]] {
    int32_t buf[S][S];
    for (int r = 0; r < S; r++)
      unrolled<S>([&](auto j) { buf[S - 1 - r][j] = CPipe<S, j>::read(); });
    for (int x = 0; x < S * S; x++)
      dc[x] = buf[x / S][x % S];
  });
  q.wait();

  std::vector<int32_t> c(S * S);
  q.memcpy(c.data(), dc, S * S * sizeof(int32_t)).wait();
  int wrong = 0;
  for (int i = 0; i < S; i++)
    for (int j = 0; j < S; j++) {
      int32_t want = 0;
      for (int k = 0; k < S; k++)
        want += int32_t(a[i * S + k]) * int32_t(b[k * S + j]);
      wrong += c[i * S + j] != want;
    }
  std::printf("E1 SYCL S=%d %s: %d of %d results wrong\n", S, wrong ? "FAIL" : "PASS",
              wrong, S * S);
  free(da, q);
  free(db, q);
  free(dc, q);
  return wrong != 0;
}

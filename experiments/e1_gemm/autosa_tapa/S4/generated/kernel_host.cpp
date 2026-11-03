#include "kernel_host.h"
#include <assert.h>
#include <stdio.h>
#include "kernel_kernel.h"
void kernel0(tapa::mmap<A_t4> A, tapa::mmap<B_t4> B, tapa::mmap<C_t1> C);

#include "kernel.h"

#include <gflags/gflags.h>
int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  data_t A[I][K], B[J][K];
  acc_t  C[I][J], C_golden[I][J];

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++)
      A[i][k] = (data_t)(rand() % 8);

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++)
      B[j][k] = (data_t)(rand() % 8);

  {
    // Allocate memory in host memory
    std::vector<char, tapa::aligned_allocator<char>> dev_A((4) * (4));
    std::vector<char, tapa::aligned_allocator<char>> dev_B((4) * (4));
    std::vector<int, tapa::aligned_allocator<int>> dev_C((4) * (4));

    // Initialize host buffers
    std::copy(reinterpret_cast<char *>(A), reinterpret_cast<char *>(A) + (4) * (4), dev_A.begin());
    std::copy(reinterpret_cast<char *>(B), reinterpret_cast<char *>(B) + (4) * (4), dev_B.begin());

    // Explicitly create TAPA mmap objects
    std::vector<tapa::read_only_mmap<char>> buffer_A;
    std::vector<tapa::read_only_mmap<char>> buffer_B;
    std::vector<tapa::write_only_mmap<int>> buffer_C;
    // Set the direction of the TAPA mmap objects
    for (int i = 0; i < 1; i++) {
      tapa::read_only_mmap<char> buffer_A_tmp(dev_A);
      buffer_A.push_back(std::move(buffer_A_tmp));
    }
    for (int i = 0; i < 1; i++) {
      tapa::read_only_mmap<char> buffer_B_tmp(dev_B);
      buffer_B.push_back(std::move(buffer_B_tmp));
    }
    for (int i = 0; i < 1; i++) {
      tapa::write_only_mmap<int> buffer_C_tmp(dev_C);
      buffer_C.push_back(std::move(buffer_C_tmp));
    }

    // Launch the kernel
    tapa::invoke(kernel0, getenv("TAPA_BITSTREAM") ? getenv("TAPA_BITSTREAM") : "", buffer_A[0].vectorized<4>(), buffer_B[0].vectorized<4>(), std::move(buffer_C[0]));
    // Restore data from host buffers
    std::copy(dev_C.begin(), dev_C.end(), reinterpret_cast<int *>(C));
  }

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C_golden[i][j] = 0;
      for (int k = 0; k < K; k++) {
        C_golden[i][j] = C_golden[i][j] + A[i][k] * B[j][k];
      }
    }

  int err = 0;
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      if (C_golden[i][j] != C[i][j]) err++;

  if (err) printf("Failed with %d errors!\n", err);
  else printf("Passed!\n");
  return 0;
}

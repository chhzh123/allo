#include "kernel_host.h"
#include <assert.h>
#include <stdio.h>
#include "kernel_kernel.h"
void kernel0(tapa::mmap<A_t64> A, tapa::mmap<B_t64> B, tapa::mmap<C_t16> C);

#include "kernel.h"

#include <gflags/gflags.h>
int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  data_t A[I][K], B[J][K];
  acc_t  C[I][J], C_golden[I][J];

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++)
      A[i][k] = (data_t)(rand() % 256 - 128);

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++)
      B[j][k] = (data_t)(rand() % 256 - 128);

  {
    // Allocate memory in host memory
    std::vector<char, tapa::aligned_allocator<char>> dev_A_unserialized((64) * (64));
    std::vector<char, tapa::aligned_allocator<char>> dev_A(4096);
    std::vector<char, tapa::aligned_allocator<char>> dev_B_unserialized((64) * (64));
    std::vector<char, tapa::aligned_allocator<char>> dev_B(4096);
    std::vector<int, tapa::aligned_allocator<int>> dev_C_unserialized((64) * (64));
    std::vector<int, tapa::aligned_allocator<int>> dev_C(4096);

    // Initialize host buffers
    std::copy(reinterpret_cast<char *>(A), reinterpret_cast<char *>(A) + (64) * (64), dev_A_unserialized.begin());
    std::copy(reinterpret_cast<char *>(B), reinterpret_cast<char *>(B) + (64) * (64), dev_B_unserialized.begin());
    host_serialize_A(dev_A, dev_A_unserialized);
    host_serialize_B(dev_B, dev_B_unserialized);

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
    tapa::invoke(kernel0, getenv("TAPA_BITSTREAM") ? getenv("TAPA_BITSTREAM") : "", buffer_A[0].vectorized<64>(), buffer_B[0].vectorized<64>(), buffer_C[0].vectorized<16>());
    host_deserialize_C(dev_C_unserialized, dev_C);
    // Restore data from host buffers
    std::copy(dev_C_unserialized.begin(), dev_C_unserialized.end(), reinterpret_cast<int *>(C));
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

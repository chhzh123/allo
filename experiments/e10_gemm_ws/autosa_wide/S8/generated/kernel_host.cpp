#include <assert.h>
#include <stdio.h>
#include "kernel_kernel.h"

#include "kernel.h"

int main(int argc, char **argv) {
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
    char *dev_A_unserialized = (char *)malloc((8) * (8) * sizeof(char));
    char *dev_A = (char *)malloc(64 * sizeof(char));
    char *dev_B_unserialized = (char *)malloc((8) * (8) * sizeof(char));
    char *dev_B = (char *)malloc(64 * sizeof(char));
    int *dev_C_unserialized = (int *)malloc((8) * (8) * sizeof(int));
    int *dev_C = (int *)malloc(64 * sizeof(int));

    // Initialize host buffers
    memcpy(dev_A_unserialized, A, (8) * (8) * sizeof(char));
    memcpy(dev_B_unserialized, B, (8) * (8) * sizeof(char));
    host_serialize_A(dev_A, dev_A_unserialized);
    host_serialize_B(dev_B, dev_B_unserialized);

    // Allocate buffers in device memory
    std::vector<A_t64 *> buffer_A;
    std::vector<B_t64 *> buffer_B;
    std::vector<C_t16 *> buffer_C;
    for (int i = 0; i < 1; i++) {
      A_t64 *buffer_A_tmp = (A_t64 *)malloc((64) * sizeof(char));
      buffer_A.push_back(buffer_A_tmp);
    }
    for (int i = 0; i < 1; i++) {
      B_t64 *buffer_B_tmp = (B_t64 *)malloc((64) * sizeof(char));
      buffer_B.push_back(buffer_B_tmp);
    }
    for (int i = 0; i < 1; i++) {
      C_t16 *buffer_C_tmp = (C_t16 *)malloc((64) * sizeof(int));
      buffer_C.push_back(buffer_C_tmp);
    }

    for (int i = 0; i < 1; i++) {
      memcpy(buffer_A[i], dev_A, (64) * sizeof(char));
    }

    for (int i = 0; i < 1; i++) {
      memcpy(buffer_B[i], dev_B, (64) * sizeof(char));
    }

    {
      // Launch the kernel
      kernel0(buffer_A[0], buffer_B[0], buffer_C[0]);
    }
    for (int i = 0; i < 1; i++) {
      memcpy(dev_C, buffer_C[i], (64) * sizeof(int));
    }

    host_deserialize_C(dev_C_unserialized, dev_C);
    // Restore data from host buffers
    memcpy(C, dev_C_unserialized, (8) * (8) * sizeof(int));

    // Clean up resources
    for (int i = 0; i < 1; i++) {
      free(buffer_A[i]);
    }
    for (int i = 0; i < 1; i++) {
      free(buffer_B[i]);
    }
    for (int i = 0; i < 1; i++) {
      free(buffer_C[i]);
    }
    free(dev_A);
    free(dev_A_unserialized);
    free(dev_B);
    free(dev_B_unserialized);
    free(dev_C);
    free(dev_C_unserialized);
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

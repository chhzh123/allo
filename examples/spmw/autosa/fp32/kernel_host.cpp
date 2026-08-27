#include <assert.h>
#include <stdio.h>
#include "kernel_kernel.h"

#include "kernel.h"

int main(int argc, char **argv) {
  data_t A[I][K], B[J][K], C[I][J], C_golden[I][J];  

  for (int i = 0; i < I; i++) 
    for (int k = 0; k < K; k++) {
      A[i][k] = (data_t)rand() / RAND_MAX;
    }

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++) {
      B[j][k] = (data_t)rand() / RAND_MAX;      
    }

  {
    // Allocate memory in host memory
    float *dev_A = (float *)malloc((16) * (16) * sizeof(float));
    float *dev_B = (float *)malloc((16) * (16) * sizeof(float));
    float *dev_C = (float *)malloc((16) * (16) * sizeof(float));

    // Initialize host buffers
    memcpy(dev_A, A, (16) * (16) * sizeof(float));
    memcpy(dev_B, B, (16) * (16) * sizeof(float));
    memcpy(dev_C, C, (16) * (16) * sizeof(float));

    // Allocate buffers in device memory
    std::vector<A_t16 *> buffer_A;
    std::vector<B_t16 *> buffer_B;
    std::vector<float *> buffer_C;
    for (int i = 0; i < 1; i++) {
      A_t16 *buffer_A_tmp = (A_t16 *)malloc((16) * (16) * sizeof(float));
      buffer_A.push_back(buffer_A_tmp);
    }
    for (int i = 0; i < 1; i++) {
      B_t16 *buffer_B_tmp = (B_t16 *)malloc((16) * (16) * sizeof(float));
      buffer_B.push_back(buffer_B_tmp);
    }
    for (int i = 0; i < 1; i++) {
      float *buffer_C_tmp = (float *)malloc((16) * (16) * sizeof(float));
      buffer_C.push_back(buffer_C_tmp);
    }

    for (int i = 0; i < 1; i++) {
      memcpy(buffer_A[i], dev_A, (16) * (16) * sizeof(float));
    }

    for (int i = 0; i < 1; i++) {
      memcpy(buffer_B[i], dev_B, (16) * (16) * sizeof(float));
    }

    for (int i = 0; i < 1; i++) {
      memcpy(buffer_C[i], dev_C, (16) * (16) * sizeof(float));
    }

    {
      // Launch the kernel
      kernel0(buffer_A[0], buffer_B[0], buffer_C[0]);
    }
    for (int i = 0; i < 1; i++) {
      memcpy(dev_C, buffer_C[i], (16) * (16) * sizeof(float));
    }

    // Restore data from host buffers
    memcpy(C, dev_C, (16) * (16) * sizeof(float));

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
    free(dev_B);
    free(dev_C);
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
    for (int j = 0; j < J; j++) {
      if (fabs((float)C_golden[i][j] - (float)C[i][j]) > 0.001)
        err++;
    }

  if (err)
    printf("Failed with %d errors!\n", err);
  else
    printf("Passed!\n");

  return 0;
}

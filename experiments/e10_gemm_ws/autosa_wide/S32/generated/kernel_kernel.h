#include <ap_int.h>
#include <hls_stream.h>

#define min(x,y) ((x < y) ? x : y)
#define max(x,y) ((x > y) ? x : y)

/* Data Type */
typedef char A_t1;
typedef char B_t1;
typedef int C_t1;
typedef ap_uint<512> A_t64;
typedef ap_uint<512> B_t64;
typedef ap_uint<512> C_t16;
/* Data Type */

/* Helper Function */
inline void host_serialize_A(char *A_to, char *A_from){
  /* Variable Declaration */
  unsigned int cnt = 0;
  /* Variable Declaration */

  // array
  // io_L3
  for (int c3 = 0; c3 <= 31; c3 += 1) {
    // io_L2
    for (int c4 = 0; c4 <= 31; c4 += 1)
      A_to[cnt++] = A_from[c4 * 32 + c3];
  }
}
/* Helper Function */

/* Helper Function */
inline void host_serialize_B(char *B_to, char *B_from){
  /* Variable Declaration */
  unsigned int cnt = 0;
  /* Variable Declaration */

  // array
  // io_L3
  for (int c3 = 0; c3 <= 31; c3 += 1) {
    // io_L2
    for (int c4 = 0; c4 <= 31; c4 += 1) {
      // io_L1
      // pe
      B_to[cnt++] = B_from[c4 * 32 + c3];
    }
  }
}
/* Helper Function */

/* Helper Function */
inline void host_deserialize_C(int *C_to, int *C_from){
  /* Variable Declaration */
  unsigned int cnt = 0;
  /* Variable Declaration */

  // array
  // io_L3
  for (int c3 = 0; c3 <= 31; c3 += 1) {
    // io_L2
    for (int c4 = 0; c4 <= 31; c4 += 1)
      C_to[c4 * 32 + c3] = C_from[cnt++];
  }
}
/* Helper Function */

void kernel0(A_t64 *A, B_t64 *B, C_t16 *C);
void A_IO_L2_in_intra_trans(int idx, char local_A[32][1], hls::stream<char> &fifo_A_local_out, bool intra_trans_en);
void A_IO_L2_in_inter_trans(int idx, char local_A[32][1], hls::stream<char> &fifo_A_in, hls::stream<char> &fifo_A_out, bool inter_trans_en);
void A_IO_L2_in_inter_trans_boundary(int idx, char local_A[32][1], hls::stream<char> &fifo_A_in, bool inter_trans_en);
void B_IO_L1_in_intra_trans(int idx, int idy, char local_B[1][1], hls::stream<char> &fifo_B_local_out, bool intra_trans_en);
void B_IO_L1_in_inter_trans(int idx, int idy, char local_B[1][1], hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_out, bool inter_trans_en);
void B_IO_L1_in_inter_trans_boundary(int idx, int idy, char local_B[1][1], hls::stream<char> &fifo_B_in, bool inter_trans_en);
void B_IO_L1_in_wrapper(int idx, int idy, hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_out, hls::stream<char> &fifo_B_local_out);
void B_IO_L1_in_boundary_wrapper(int idx, int idy, hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_local_out);
void PE_wrapper(int idx, int idy, hls::stream<char> &fifo_A_in, hls::stream<char> &fifo_A_out, hls::stream<char> &fifo_B_in, hls::stream<int> &fifo_C_1_in, hls::stream<int> &fifo_C_1_out);
void C_1_IO_L2_out_intra_trans(int idx, int local_C_1[32][1], hls::stream<int> &fifo_C_1_local_in, bool intra_trans_en);
void C_1_IO_L2_out_inter_trans(int idx, int local_C_1[32][1], hls::stream<int> &fifo_C_1_in, hls::stream<int> &fifo_C_1_out, bool inter_trans_en);
void C_1_IO_L2_out_inter_trans_boundary(int idx, int local_C_1[32][1], hls::stream<int> &fifo_C_1_out, bool inter_trans_en);

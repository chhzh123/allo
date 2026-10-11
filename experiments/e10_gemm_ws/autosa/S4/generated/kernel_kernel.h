#include <ap_int.h>
#include <hls_stream.h>

#define min(x,y) ((x < y) ? x : y)
#define max(x,y) ((x > y) ? x : y)

/* Data Type */
typedef char A_t1;
typedef char B_t1;
typedef int C_t1;
/* Data Type */

void kernel0(A_t1 *A, B_t1 *B, C_t1 *C);
void A_IO_L2_in_intra_trans(int idx, char local_A[4][1], hls::stream<char> &fifo_A_local_out, bool intra_trans_en);
void A_IO_L2_in_inter_trans(int idx, char local_A[4][1], hls::stream<char> &fifo_A_in, hls::stream<char> &fifo_A_out, bool inter_trans_en);
void A_IO_L2_in_inter_trans_boundary(int idx, char local_A[4][1], hls::stream<char> &fifo_A_in, bool inter_trans_en);
void B_IO_L1_in_intra_trans(int idx, int idy, char local_B[1][1], hls::stream<char> &fifo_B_local_out, bool intra_trans_en);
void B_IO_L1_in_inter_trans(int idx, int idy, char local_B[1][1], hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_out, bool inter_trans_en);
void B_IO_L1_in_inter_trans_boundary(int idx, int idy, char local_B[1][1], hls::stream<char> &fifo_B_in, bool inter_trans_en);
void B_IO_L1_in_wrapper(int idx, int idy, hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_out, hls::stream<char> &fifo_B_local_out);
void B_IO_L1_in_boundary_wrapper(int idx, int idy, hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_local_out);
void PE_wrapper(int idx, int idy, hls::stream<char> &fifo_A_in, hls::stream<char> &fifo_A_out, hls::stream<char> &fifo_B_in, hls::stream<int> &fifo_C_1_in, hls::stream<int> &fifo_C_1_out);
void C_1_IO_L2_out_intra_trans(int idx, int local_C_1[4][1], hls::stream<int> &fifo_C_1_local_in, bool intra_trans_en);
void C_1_IO_L2_out_inter_trans(int idx, int local_C_1[4][1], hls::stream<int> &fifo_C_1_in, hls::stream<int> &fifo_C_1_out, bool inter_trans_en);
void C_1_IO_L2_out_inter_trans_boundary(int idx, int local_C_1[4][1], hls::stream<int> &fifo_C_1_out, bool inter_trans_en);

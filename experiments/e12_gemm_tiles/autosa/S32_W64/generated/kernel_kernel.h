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
/* Data Type */

void kernel0(A_t64 *A, B_t64 *B, C_t1 *C);
void A_IO_L2_in_intra_trans(int idx, int c0, A_t64 local_A[1][1], hls::stream<char> &fifo_A_local_out, bool intra_trans_en);
void A_IO_L2_in_inter_trans(int idx, int c0, A_t64 local_A[1][1], hls::stream<A_t64> &fifo_A_in, hls::stream<A_t64> &fifo_A_out, bool inter_trans_en);
void A_IO_L2_in_inter_trans_boundary(int idx, int c0, A_t64 local_A[1][1], hls::stream<A_t64> &fifo_A_in, bool inter_trans_en);
void B_IO_L2_in_intra_trans(int idx, int c0, int c1, B_t64 local_B[1][1], hls::stream<char> &fifo_B_local_out, bool intra_trans_en);
void B_IO_L2_in_inter_trans(int idx, int c0, int c1, B_t64 local_B[1][1], hls::stream<B_t64> &fifo_B_in, hls::stream<B_t64> &fifo_B_out, bool inter_trans_en);
void B_IO_L2_in_inter_trans_boundary(int idx, int c0, int c1, B_t64 local_B[1][1], hls::stream<B_t64> &fifo_B_in, bool inter_trans_en);
void PE_wrapper(int idx, int idy, hls::stream<char> &fifo_A_in, hls::stream<char> &fifo_A_out, hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_out, hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int c0, int c1, int local_C[1][1], hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int c0, int c1, int local_C[1][1], hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_inter_trans_boundary(int idx, int idy, int c0, int c1, int local_C[1][1], hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_wrapper(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_boundary_wrapper(int idx, int idy, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);

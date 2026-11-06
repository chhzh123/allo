#include <ap_int.h>
#include <hls_stream.h>

#define min(x,y) ((x < y) ? x : y)
#define max(x,y) ((x > y) ? x : y)

/* Data Type */
typedef char A_t1;
typedef char B_t1;
typedef int C_t1;
typedef ap_uint<128> A_t16;
typedef ap_uint<128> B_t16;
/* Data Type */

void kernel0(A_t16 *A, B_t16 *B, C_t1 *C);
void A_IO_L2_in_intra_trans(int idx, A_t16 local_A[1][1], hls::stream<char> &fifo_A_local_out, bool intra_trans_en);
void A_IO_L2_in_inter_trans(int idx, A_t16 local_A[1][1], hls::stream<A_t16> &fifo_A_in, hls::stream<A_t16> &fifo_A_out, bool inter_trans_en);
void A_IO_L2_in_inter_trans_boundary(int idx, A_t16 local_A[1][1], hls::stream<A_t16> &fifo_A_in, bool inter_trans_en);
void B_IO_L2_in_intra_trans(int idx, B_t16 local_B[1][1], hls::stream<char> &fifo_B_local_out, bool intra_trans_en);
void B_IO_L2_in_inter_trans(int idx, B_t16 local_B[1][1], hls::stream<B_t16> &fifo_B_in, hls::stream<B_t16> &fifo_B_out, bool inter_trans_en);
void B_IO_L2_in_inter_trans_boundary(int idx, B_t16 local_B[1][1], hls::stream<B_t16> &fifo_B_in, bool inter_trans_en);
void PE_wrapper(int idx, int idy, hls::stream<char> &fifo_A_in, hls::stream<char> &fifo_A_out, hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_out, hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int local_C[1][1], hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int local_C[1][1], hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_inter_trans_boundary(int idx, int idy, int local_C[1][1], hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_wrapper(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_boundary_wrapper(int idx, int idy, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);

#include <tapa.h>
#include <ap_int.h>

/* Data Type */
typedef char A_t1;
typedef char B_t1;
typedef int C_t1;
typedef tapa::vec_t<char, 64> A_t64;
typedef tapa::vec_t<char, 64> B_t64;
typedef tapa::vec_t<int, 16> C_t16;
typedef tapa::vec_t<int, 4> C_t4;
/* Data Type */

void A_IO_L2_in_intra_trans(int idx, int c0, int c1, int c2, A_t64 local_A[4][1], tapa::ostream<char> &fifo_A_local_out, bool intra_trans_en);
void A_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, A_t64 local_A[4][1], tapa::istream<A_t64> &fifo_A_in, tapa::ostream<A_t64> &fifo_A_out, bool inter_trans_en);
void A_IO_L2_in_inter_trans_boundary(int idx, int c0, int c1, int c2, A_t64 local_A[4][1], tapa::istream<A_t64> &fifo_A_in, bool inter_trans_en);
void B_IO_L2_in_intra_trans(int idx, int c0, int c1, int c2, B_t64 local_B[4][1], tapa::ostream<char> &fifo_B_local_out, bool intra_trans_en);
void B_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, B_t64 local_B[4][1], tapa::istream<B_t64> &fifo_B_in, tapa::ostream<B_t64> &fifo_B_out, bool inter_trans_en);
void B_IO_L2_in_inter_trans_boundary(int idx, int c0, int c1, int c2, B_t64 local_B[4][1], tapa::istream<B_t64> &fifo_B_in, bool inter_trans_en);
void PE_wrapper(int idx, int idy, tapa::istream<char> &fifo_A_in, tapa::ostream<char> &fifo_A_out, tapa::istream<char> &fifo_B_in, tapa::ostream<char> &fifo_B_out, tapa::ostream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int c0, int c1, C_t4 local_C[4][1], tapa::istream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int c0, int c1, C_t4 local_C[4][1], tapa::istream<C_t4> &fifo_C_drain_in, tapa::ostream<C_t4> &fifo_C_drain_out);
void C_drain_IO_L1_out_inter_trans_boundary(int idx, int idy, int c0, int c1, C_t4 local_C[4][1], tapa::ostream<C_t4> &fifo_C_drain_out);
void C_drain_IO_L1_out_wrapper(int idx, int idy, tapa::istream<C_t4> &fifo_C_drain_in, tapa::ostream<C_t4> &fifo_C_drain_out, tapa::istream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_boundary_wrapper(int idx, int idy, tapa::ostream<C_t4> &fifo_C_drain_out, tapa::istream<int> &fifo_C_drain_local_in);

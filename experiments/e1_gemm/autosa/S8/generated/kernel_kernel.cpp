#include "kernel_kernel.h"
/* Module Definition */
void A_IO_L3_in(A_t8 *A, hls::stream<A_t8> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<4> c3 = 0; c3 <= 7; c3 += 1) {
  #pragma HLS PIPELINE II=1
    // io_L2
    // access_coalesce
    {
      A_t8 in_data;
      A_t8 out_data;
      in_data = A[c3];
      out_data = in_data;
      fifo_A_local_out.write(out_data);
    }
  }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_intra_trans(int idx, A_t8 local_A[1][1], hls::stream<char> &fifo_A_local_out, bool intra_trans_en) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  ap_uint<8> data_split[8];
  #pragma HLS ARRAY_PARTITION variable=data_split complete
  /* Variable Declaration */

  if (!intra_trans_en) return;


  // array
  // io_L3
  // io_L2
  // io_L1
  // pe
  for (ap_uint<4> c5 = 0; c5 <= 7; c5 += 1) {
  #pragma HLS PIPELINE II=1
    // hls_pipeline
    {
      A_t8 in_data;
      A_t1 out_data;
      in_data = local_A[0][c5 / 8];
      for (ap_uint<4> n = 0; n < 8; n++) {
      #pragma HLS UNROLL
        data_split[n] = in_data(7, 0);
        in_data = in_data >> 8;
      }
      int split_idx = (c5) % 8;
      union {unsigned int ui; char ut;} u;
      u.ui = (unsigned int)data_split[split_idx];
      out_data = u.ut;
      fifo_A_local_out.write(out_data);
    }
  }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_inter_trans(int idx, A_t8 local_A[1][1], hls::stream<A_t8> &fifo_A_in, hls::stream<A_t8> &fifo_A_out, bool inter_trans_en) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  /* Variable Declaration */

  if (!inter_trans_en) return;

  for (ap_uint<4> c3 = p0; c3 <= 7; c3 += 1) {
  #pragma HLS PIPELINE II=1
    // io_L2
    if (c3 == p0) {
      // access_coalesce
      {
        A_t8 in_data;
        A_t8 out_data;
        in_data = fifo_A_in.read();
        out_data = in_data;
        local_A[0][0] = out_data;
      }
    } else {
      // access_coalesce
      // hls_pipeline
      {
        A_t8 in_data;
        A_t8 out_data;
        in_data = fifo_A_in.read();
        out_data = in_data;
        fifo_A_out.write(out_data);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_inter_trans_boundary(int idx, A_t8 local_A[1][1], hls::stream<A_t8> &fifo_A_in, bool inter_trans_en) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  /* Variable Declaration */

  if (!inter_trans_en) return;

  for (ap_uint<4> c3 = p0; c3 <= 7; c3 += 1)
  #pragma HLS PIPELINE II=1
    if (c3 == p0) {
      // io_L2
      // access_coalesce
      {
        A_t8 in_data;
        A_t8 out_data;
        in_data = fifo_A_in.read();
        out_data = in_data;
        local_A[0][0] = out_data;
      }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in(int idx, hls::stream<A_t8> &fifo_A_in, hls::stream<A_t8> &fifo_A_out, hls::stream<char> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  A_t8 local_A_ping[1][1];
  #pragma HLS RESOURCE variable=local_A_ping core=RAM_2P_BRAM
  A_t8 local_A_pong[1][1];
  #pragma HLS RESOURCE variable=local_A_pong core=RAM_2P_BRAM
  bool arb = 0;
  bool inter_trans_en = 1;
  bool intra_trans_en = 0;
  /* Variable Declaration */

  {
    // array
    // io_L3
    {
      if (arb == 0) {
        A_IO_L2_in_inter_trans(
          /* module id */ idx, 
          /* array */ local_A_pong, 
          /* fifo */ fifo_A_in, 
          /* fifo */ fifo_A_out, 
          /* enable */ inter_trans_en
        );
        A_IO_L2_in_intra_trans(
          /* module id */ idx, 
          /* array */ local_A_ping, 
          /* fifo */ fifo_A_local_out, 
          /* enable */ intra_trans_en
        );
      } else {
        A_IO_L2_in_inter_trans(
          /* module id */ idx, 
          /* array */ local_A_ping, 
          /* fifo */ fifo_A_in, 
          /* fifo */ fifo_A_out, 
          /* enable */ inter_trans_en
        );
        A_IO_L2_in_intra_trans(
          /* module id */ idx, 
          /* array */ local_A_pong, 
          /* fifo */ fifo_A_local_out, 
          /* enable */ intra_trans_en
        );
      }
      intra_trans_en = 1;
      arb = !arb;
    }
    if (arb == 0) {
      A_IO_L2_in_intra_trans(
        /* module id */ idx, 
        /* array */ local_A_ping, 
        /* fifo */ fifo_A_local_out, 
        /* enable */ intra_trans_en
      );
    } else {
      A_IO_L2_in_intra_trans(
        /* module id */ idx, 
        /* array */ local_A_pong, 
        /* fifo */ fifo_A_local_out, 
        /* enable */ intra_trans_en
      );
    }
  }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_boundary(int idx, hls::stream<A_t8> &fifo_A_in, hls::stream<char> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  A_t8 local_A_ping[1][1];
  #pragma HLS RESOURCE variable=local_A_ping core=RAM_2P_BRAM
  A_t8 local_A_pong[1][1];
  #pragma HLS RESOURCE variable=local_A_pong core=RAM_2P_BRAM
  bool arb = 0;
  bool inter_trans_en = 1;
  bool intra_trans_en = 0;
  /* Variable Declaration */

  {
    // array
    // io_L3
    {
      if (arb == 0) {
        A_IO_L2_in_inter_trans_boundary(
          /* module id */ idx, 
          /* array */ local_A_pong, 
          /* fifo */ fifo_A_in, 
          /* enable */ inter_trans_en
        );
        A_IO_L2_in_intra_trans(
          /* module id */ idx, 
          /* array */ local_A_ping, 
          /* fifo */ fifo_A_local_out, 
          /* enable */ intra_trans_en
        );
      } else {
        A_IO_L2_in_inter_trans_boundary(
          /* module id */ idx, 
          /* array */ local_A_ping, 
          /* fifo */ fifo_A_in, 
          /* enable */ inter_trans_en
        );
        A_IO_L2_in_intra_trans(
          /* module id */ idx, 
          /* array */ local_A_pong, 
          /* fifo */ fifo_A_local_out, 
          /* enable */ intra_trans_en
        );
      }
      intra_trans_en = 1;
      arb = !arb;
    }
    if (arb == 0) {
      A_IO_L2_in_intra_trans(
        /* module id */ idx, 
        /* array */ local_A_ping, 
        /* fifo */ fifo_A_local_out, 
        /* enable */ intra_trans_en
      );
    } else {
      A_IO_L2_in_intra_trans(
        /* module id */ idx, 
        /* array */ local_A_pong, 
        /* fifo */ fifo_A_local_out, 
        /* enable */ intra_trans_en
      );
    }
  }
}
/* Module Definition */

/* Module Definition */
void B_IO_L3_in(B_t8 *B, hls::stream<B_t8> &fifo_B_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<4> c3 = 0; c3 <= 7; c3 += 1) {
  #pragma HLS PIPELINE II=1
    // io_L2
    // access_coalesce
    {
      B_t8 in_data;
      B_t8 out_data;
      in_data = B[c3];
      out_data = in_data;
      fifo_B_local_out.write(out_data);
    }
  }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_intra_trans(int idx, B_t8 local_B[1][1], hls::stream<char> &fifo_B_local_out, bool intra_trans_en) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  ap_uint<8> data_split[8];
  #pragma HLS ARRAY_PARTITION variable=data_split complete
  /* Variable Declaration */

  if (!intra_trans_en) return;


  // array
  // io_L3
  // io_L2
  // io_L1
  // pe
  for (ap_uint<4> c5 = 0; c5 <= 7; c5 += 1) {
  #pragma HLS PIPELINE II=1
    // hls_pipeline
    {
      B_t8 in_data;
      B_t1 out_data;
      in_data = local_B[0][c5 / 8];
      for (ap_uint<4> n = 0; n < 8; n++) {
      #pragma HLS UNROLL
        data_split[n] = in_data(7, 0);
        in_data = in_data >> 8;
      }
      int split_idx = (c5) % 8;
      union {unsigned int ui; char ut;} u;
      u.ui = (unsigned int)data_split[split_idx];
      out_data = u.ut;
      fifo_B_local_out.write(out_data);
    }
  }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_inter_trans(int idx, B_t8 local_B[1][1], hls::stream<B_t8> &fifo_B_in, hls::stream<B_t8> &fifo_B_out, bool inter_trans_en) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  /* Variable Declaration */

  if (!inter_trans_en) return;

  for (ap_uint<4> c3 = p0; c3 <= 7; c3 += 1) {
  #pragma HLS PIPELINE II=1
    // io_L2
    if (c3 == p0) {
      // access_coalesce
      {
        B_t8 in_data;
        B_t8 out_data;
        in_data = fifo_B_in.read();
        out_data = in_data;
        local_B[0][0] = out_data;
      }
    } else {
      // access_coalesce
      // hls_pipeline
      {
        B_t8 in_data;
        B_t8 out_data;
        in_data = fifo_B_in.read();
        out_data = in_data;
        fifo_B_out.write(out_data);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_inter_trans_boundary(int idx, B_t8 local_B[1][1], hls::stream<B_t8> &fifo_B_in, bool inter_trans_en) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  /* Variable Declaration */

  if (!inter_trans_en) return;

  for (ap_uint<4> c3 = p0; c3 <= 7; c3 += 1)
  #pragma HLS PIPELINE II=1
    if (c3 == p0) {
      // io_L2
      // access_coalesce
      {
        B_t8 in_data;
        B_t8 out_data;
        in_data = fifo_B_in.read();
        out_data = in_data;
        local_B[0][0] = out_data;
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in(int idx, hls::stream<B_t8> &fifo_B_in, hls::stream<B_t8> &fifo_B_out, hls::stream<char> &fifo_B_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  B_t8 local_B_ping[1][1];
  #pragma HLS RESOURCE variable=local_B_ping core=RAM_2P_BRAM
  B_t8 local_B_pong[1][1];
  #pragma HLS RESOURCE variable=local_B_pong core=RAM_2P_BRAM
  bool arb = 0;
  bool inter_trans_en = 1;
  bool intra_trans_en = 0;
  /* Variable Declaration */

  {
    // array
    // io_L3
    {
      if (arb == 0) {
        B_IO_L2_in_inter_trans(
          /* module id */ idx, 
          /* array */ local_B_pong, 
          /* fifo */ fifo_B_in, 
          /* fifo */ fifo_B_out, 
          /* enable */ inter_trans_en
        );
        B_IO_L2_in_intra_trans(
          /* module id */ idx, 
          /* array */ local_B_ping, 
          /* fifo */ fifo_B_local_out, 
          /* enable */ intra_trans_en
        );
      } else {
        B_IO_L2_in_inter_trans(
          /* module id */ idx, 
          /* array */ local_B_ping, 
          /* fifo */ fifo_B_in, 
          /* fifo */ fifo_B_out, 
          /* enable */ inter_trans_en
        );
        B_IO_L2_in_intra_trans(
          /* module id */ idx, 
          /* array */ local_B_pong, 
          /* fifo */ fifo_B_local_out, 
          /* enable */ intra_trans_en
        );
      }
      intra_trans_en = 1;
      arb = !arb;
    }
    if (arb == 0) {
      B_IO_L2_in_intra_trans(
        /* module id */ idx, 
        /* array */ local_B_ping, 
        /* fifo */ fifo_B_local_out, 
        /* enable */ intra_trans_en
      );
    } else {
      B_IO_L2_in_intra_trans(
        /* module id */ idx, 
        /* array */ local_B_pong, 
        /* fifo */ fifo_B_local_out, 
        /* enable */ intra_trans_en
      );
    }
  }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_boundary(int idx, hls::stream<B_t8> &fifo_B_in, hls::stream<char> &fifo_B_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  B_t8 local_B_ping[1][1];
  #pragma HLS RESOURCE variable=local_B_ping core=RAM_2P_BRAM
  B_t8 local_B_pong[1][1];
  #pragma HLS RESOURCE variable=local_B_pong core=RAM_2P_BRAM
  bool arb = 0;
  bool inter_trans_en = 1;
  bool intra_trans_en = 0;
  /* Variable Declaration */

  {
    // array
    // io_L3
    {
      if (arb == 0) {
        B_IO_L2_in_inter_trans_boundary(
          /* module id */ idx, 
          /* array */ local_B_pong, 
          /* fifo */ fifo_B_in, 
          /* enable */ inter_trans_en
        );
        B_IO_L2_in_intra_trans(
          /* module id */ idx, 
          /* array */ local_B_ping, 
          /* fifo */ fifo_B_local_out, 
          /* enable */ intra_trans_en
        );
      } else {
        B_IO_L2_in_inter_trans_boundary(
          /* module id */ idx, 
          /* array */ local_B_ping, 
          /* fifo */ fifo_B_in, 
          /* enable */ inter_trans_en
        );
        B_IO_L2_in_intra_trans(
          /* module id */ idx, 
          /* array */ local_B_pong, 
          /* fifo */ fifo_B_local_out, 
          /* enable */ intra_trans_en
        );
      }
      intra_trans_en = 1;
      arb = !arb;
    }
    if (arb == 0) {
      B_IO_L2_in_intra_trans(
        /* module id */ idx, 
        /* array */ local_B_ping, 
        /* fifo */ fifo_B_local_out, 
        /* enable */ intra_trans_en
      );
    } else {
      B_IO_L2_in_intra_trans(
        /* module id */ idx, 
        /* array */ local_B_pong, 
        /* fifo */ fifo_B_local_out, 
        /* enable */ intra_trans_en
      );
    }
  }
}
/* Module Definition */

/* Module Definition */
void PE(int idx, int idy, hls::stream<char> &fifo_A_in, hls::stream<char> &fifo_A_out, hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_out, hls::stream<int> &fifo_C_drain_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx, p1 = idy; // module id
  A_t1 local_A[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_A dim=0 complete
  B_t1 local_B[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_B dim=0 complete
  C_t1 local_C[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_C dim=0 complete
  /* Variable Declaration */

  {
    // array
    // pe
    // hls_pipeline
    local_C[0][0] = 0;
    // array
    // pe
    for (ap_uint<4> c5 = 0; c5 <= 7; c5 += 1) {
    #pragma HLS PIPELINE II=1
      {
        local_A[0][0] = fifo_A_in.read();
        local_B[0][0] = fifo_B_in.read();
        local_C[0][0] = (local_C[0][0] + (local_A[0][0] * local_B[0][0]));
        if (c5 == 7)
          fifo_C_drain_out.write(local_C[0][0]);
        fifo_B_out.write(local_B[0][0]);
        fifo_A_out.write(local_A[0][0]);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
void PE_wrapper(int idx, int idy, hls::stream<char> &fifo_A_in, hls::stream<char> &fifo_A_out, hls::stream<char> &fifo_B_in, hls::stream<char> &fifo_B_out, hls::stream<int> &fifo_C_drain_out)
 {
  PE(
    /* module id */ idx, 
    /* module id */ idy, 
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_A_out, 
    /* fifo */ fifo_B_in, 
    /* fifo */ fifo_B_out, 
    /* fifo */ fifo_C_drain_out);
}
/* Module Definition */

/* Module Definition */
void A_PE_dummy_in(int idx, int idy, hls::stream<char> &fifo_A_in) {
  /* Variable Declaration */
  int p0 = idx, p1 = idy; // module id
  /* Variable Declaration */

  {
    // array
    {
    }
    // array
    // pe
    for (ap_uint<4> c5 = 0; c5 <= 7; c5 += 1) {
    #pragma HLS PIPELINE II=1
      A_t1 fifo_data;
      fifo_data = fifo_A_in.read();
    }
  }
}
/* Module Definition */

/* Module Definition */
void B_PE_dummy_in(int idx, int idy, hls::stream<char> &fifo_B_in) {
  /* Variable Declaration */
  int p0 = idx, p1 = idy; // module id
  /* Variable Declaration */

  {
    // array
    {
    }
    // array
    // pe
    for (ap_uint<4> c5 = 0; c5 <= 7; c5 += 1) {
    #pragma HLS PIPELINE II=1
      B_t1 fifo_data;
      fifo_data = fifo_B_in.read();
    }
  }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int local_C[1][1], hls::stream<int> &fifo_C_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  int p0 = idx, p1 = idy; // module id
  /* Variable Declaration */


  // io_L1
  // pe
  // hls_pipeline
  // hls_pipeline
  {
    C_t1 in_data;
    C_t1 out_data;
    in_data = fifo_C_drain_local_in.read();
    out_data = in_data;
    local_C[0][0] = out_data;
  }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int local_C[1][1], hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  int p0 = idx, p1 = idy; // module id
  /* Variable Declaration */

  for (ap_uint<4> c4 = p1; c4 <= 7; c4 += 1) {
  #pragma HLS PIPELINE II=1
    // io_L1
    if (c4 == p1) {
      // access_coalesce
      {
        C_t1 in_data;
        C_t1 out_data;
        in_data = local_C[0][0];
        out_data = in_data;
        fifo_C_drain_out.write(out_data);
      }
    } else {
      // access_coalesce
      // hls_pipeline
      {
        C_t1 in_data;
        C_t1 out_data;
        in_data = fifo_C_drain_in.read();
        out_data = in_data;
        fifo_C_drain_out.write(out_data);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_inter_trans_boundary(int idx, int idy, int local_C[1][1], hls::stream<int> &fifo_C_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  int p0 = idx, p1 = idy; // module id
  /* Variable Declaration */

  for (ap_uint<4> c4 = p1; c4 <= 7; c4 += 1)
  #pragma HLS PIPELINE II=1
    if (c4 == p1) {
      // io_L1
      // access_coalesce
      {
        C_t1 in_data;
        C_t1 out_data;
        in_data = local_C[0][0];
        out_data = in_data;
        fifo_C_drain_out.write(out_data);
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx, p1 = idy; // module id
  C_t1 local_C[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_C dim=0 complete
  /* Variable Declaration */

  // array
  // io_L3
  // io_L2
  C_drain_IO_L1_out_intra_trans(
    /* module id */ idx, 
    /* module id */ idy, 
    /* array */ local_C, 
    /* fifo */ fifo_C_drain_local_in
  );
  C_drain_IO_L1_out_inter_trans(
    /* module id */ idx, 
    /* module id */ idy, 
    /* array */ local_C, 
    /* fifo */ fifo_C_drain_in, 
    /* fifo */ fifo_C_drain_out
  );
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_wrapper(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
 {
  C_drain_IO_L1_out(
    /* module id */ idx, 
    /* module id */ idy, 
    /* fifo */ fifo_C_drain_in, 
    /* fifo */ fifo_C_drain_out, 
    /* fifo */ fifo_C_drain_local_in);
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_boundary(int idx, int idy, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  int p0 = idx, p1 = idy; // module id
  C_t1 local_C[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_C dim=0 complete
  /* Variable Declaration */

  // array
  // io_L3
  // io_L2
  C_drain_IO_L1_out_intra_trans(
    /* module id */ idx, 
    /* module id */ idy, 
    /* array */ local_C, 
    /* fifo */ fifo_C_drain_local_in
  );
  C_drain_IO_L1_out_inter_trans_boundary(
    /* module id */ idx, 
    /* module id */ idy, 
    /* array */ local_C, 
    /* fifo */ fifo_C_drain_out
  );
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_boundary_wrapper(int idx, int idy, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
 {
  C_drain_IO_L1_out_boundary(
    /* module id */ idx, 
    /* module id */ idy, 
    /* fifo */ fifo_C_drain_out, 
    /* fifo */ fifo_C_drain_local_in);
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L2_out(int idx, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<4> c3 = p0; c3 <= 7; c3 += 1) {
    // io_L2
    if (c3 == p0) {
      for (ap_uint<4> c4 = 0; c4 <= 7; c4 += 1) {
      #pragma HLS PIPELINE II=1
        // io_L1
        // access_coalesce
        {
          C_t1 in_data;
          C_t1 out_data;
          in_data = fifo_C_drain_local_in.read();
          out_data = in_data;
          fifo_C_drain_out.write(out_data);
        }
      }
    } else {
      for (ap_uint<4> c4 = 0; c4 <= 7; c4 += 1) {
      #pragma HLS PIPELINE II=1
        // io_L1
        // access_coalesce
        {
          C_t1 in_data;
          C_t1 out_data;
          in_data = fifo_C_drain_in.read();
          out_data = in_data;
          fifo_C_drain_out.write(out_data);
        }
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L2_out_boundary(int idx, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  int p0 = idx; // module id
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<4> c3 = p0; c3 <= 7; c3 += 1)
    if (c3 == p0) {
      // io_L2
      for (ap_uint<4> c4 = 0; c4 <= 7; c4 += 1) {
      #pragma HLS PIPELINE II=1
        // io_L1
        // access_coalesce
        {
          C_t1 in_data;
          C_t1 out_data;
          in_data = fifo_C_drain_local_in.read();
          out_data = in_data;
          fifo_C_drain_out.write(out_data);
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L3_out(C_t1 *C, hls::stream<int> &fifo_C_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<4> c3 = 0; c3 <= 7; c3 += 1) {
    // io_L2
    for (ap_uint<4> c4 = 0; c4 <= 7; c4 += 1) {
    #pragma HLS PIPELINE II=1
      // io_L1
      // access_coalesce
      {
        C_t1 in_data;
        C_t1 out_data;
        in_data = fifo_C_drain_local_in.read();
        out_data = in_data;
        C[c3 + 8*c4] = out_data;
      }
    }
  }
}
/* Module Definition */

void kernel0(A_t8 *A, B_t8 *B, C_t1 *C)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem_A
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem_B
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem_C
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

  /* FIFO Declaration */
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_0 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_1 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_2 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_3 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_4 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_5 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_6 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_7 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<A_t8> fifo_A_A_IO_L2_in_8;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_8 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_0;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_0 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_1;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_1 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_2;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_2 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_3;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_3 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_4;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_4 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_5;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_5 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_6;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_6 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_7;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_7 core=FIFO_SRL
  /* B_IO_L2_in fifo */ hls::stream<B_t8> fifo_B_B_IO_L2_in_8;
  #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_8 depth=2
  #pragma HLS RESOURCE variable=fifo_B_B_IO_L2_in_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_0;
  #pragma HLS STREAM variable=fifo_A_PE_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_1;
  #pragma HLS STREAM variable=fifo_A_PE_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_2;
  #pragma HLS STREAM variable=fifo_A_PE_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_3;
  #pragma HLS STREAM variable=fifo_A_PE_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_4;
  #pragma HLS STREAM variable=fifo_A_PE_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_5;
  #pragma HLS STREAM variable=fifo_A_PE_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_6;
  #pragma HLS STREAM variable=fifo_A_PE_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_7;
  #pragma HLS STREAM variable=fifo_A_PE_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_0_8;
  #pragma HLS STREAM variable=fifo_A_PE_0_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_0;
  #pragma HLS STREAM variable=fifo_A_PE_1_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_1;
  #pragma HLS STREAM variable=fifo_A_PE_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_2;
  #pragma HLS STREAM variable=fifo_A_PE_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_3;
  #pragma HLS STREAM variable=fifo_A_PE_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_4;
  #pragma HLS STREAM variable=fifo_A_PE_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_5;
  #pragma HLS STREAM variable=fifo_A_PE_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_6;
  #pragma HLS STREAM variable=fifo_A_PE_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_7;
  #pragma HLS STREAM variable=fifo_A_PE_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_1_8;
  #pragma HLS STREAM variable=fifo_A_PE_1_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_0;
  #pragma HLS STREAM variable=fifo_A_PE_2_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_1;
  #pragma HLS STREAM variable=fifo_A_PE_2_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_2;
  #pragma HLS STREAM variable=fifo_A_PE_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_3;
  #pragma HLS STREAM variable=fifo_A_PE_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_4;
  #pragma HLS STREAM variable=fifo_A_PE_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_5;
  #pragma HLS STREAM variable=fifo_A_PE_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_6;
  #pragma HLS STREAM variable=fifo_A_PE_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_7;
  #pragma HLS STREAM variable=fifo_A_PE_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_2_8;
  #pragma HLS STREAM variable=fifo_A_PE_2_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_0;
  #pragma HLS STREAM variable=fifo_A_PE_3_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_1;
  #pragma HLS STREAM variable=fifo_A_PE_3_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_2;
  #pragma HLS STREAM variable=fifo_A_PE_3_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_3;
  #pragma HLS STREAM variable=fifo_A_PE_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_4;
  #pragma HLS STREAM variable=fifo_A_PE_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_5;
  #pragma HLS STREAM variable=fifo_A_PE_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_6;
  #pragma HLS STREAM variable=fifo_A_PE_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_7;
  #pragma HLS STREAM variable=fifo_A_PE_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_3_8;
  #pragma HLS STREAM variable=fifo_A_PE_3_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_0;
  #pragma HLS STREAM variable=fifo_A_PE_4_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_1;
  #pragma HLS STREAM variable=fifo_A_PE_4_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_2;
  #pragma HLS STREAM variable=fifo_A_PE_4_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_3;
  #pragma HLS STREAM variable=fifo_A_PE_4_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_4;
  #pragma HLS STREAM variable=fifo_A_PE_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_5;
  #pragma HLS STREAM variable=fifo_A_PE_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_6;
  #pragma HLS STREAM variable=fifo_A_PE_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_7;
  #pragma HLS STREAM variable=fifo_A_PE_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_4_8;
  #pragma HLS STREAM variable=fifo_A_PE_4_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_0;
  #pragma HLS STREAM variable=fifo_A_PE_5_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_1;
  #pragma HLS STREAM variable=fifo_A_PE_5_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_2;
  #pragma HLS STREAM variable=fifo_A_PE_5_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_3;
  #pragma HLS STREAM variable=fifo_A_PE_5_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_4;
  #pragma HLS STREAM variable=fifo_A_PE_5_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_5;
  #pragma HLS STREAM variable=fifo_A_PE_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_6;
  #pragma HLS STREAM variable=fifo_A_PE_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_7;
  #pragma HLS STREAM variable=fifo_A_PE_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_5_8;
  #pragma HLS STREAM variable=fifo_A_PE_5_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_0;
  #pragma HLS STREAM variable=fifo_A_PE_6_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_1;
  #pragma HLS STREAM variable=fifo_A_PE_6_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_2;
  #pragma HLS STREAM variable=fifo_A_PE_6_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_3;
  #pragma HLS STREAM variable=fifo_A_PE_6_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_4;
  #pragma HLS STREAM variable=fifo_A_PE_6_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_5;
  #pragma HLS STREAM variable=fifo_A_PE_6_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_6;
  #pragma HLS STREAM variable=fifo_A_PE_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_7;
  #pragma HLS STREAM variable=fifo_A_PE_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_6_8;
  #pragma HLS STREAM variable=fifo_A_PE_6_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_0;
  #pragma HLS STREAM variable=fifo_A_PE_7_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_1;
  #pragma HLS STREAM variable=fifo_A_PE_7_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_2;
  #pragma HLS STREAM variable=fifo_A_PE_7_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_3;
  #pragma HLS STREAM variable=fifo_A_PE_7_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_4;
  #pragma HLS STREAM variable=fifo_A_PE_7_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_5;
  #pragma HLS STREAM variable=fifo_A_PE_7_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_6;
  #pragma HLS STREAM variable=fifo_A_PE_7_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_7;
  #pragma HLS STREAM variable=fifo_A_PE_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_A_PE_7_8;
  #pragma HLS STREAM variable=fifo_A_PE_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_0_0;
  #pragma HLS STREAM variable=fifo_B_PE_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_0_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_1_0;
  #pragma HLS STREAM variable=fifo_B_PE_1_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_1_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_2_0;
  #pragma HLS STREAM variable=fifo_B_PE_2_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_2_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_3_0;
  #pragma HLS STREAM variable=fifo_B_PE_3_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_3_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_4_0;
  #pragma HLS STREAM variable=fifo_B_PE_4_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_4_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_5_0;
  #pragma HLS STREAM variable=fifo_B_PE_5_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_5_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_6_0;
  #pragma HLS STREAM variable=fifo_B_PE_6_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_6_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_7_0;
  #pragma HLS STREAM variable=fifo_B_PE_7_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_7_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_8_0;
  #pragma HLS STREAM variable=fifo_B_PE_8_0 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_8_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_0_1;
  #pragma HLS STREAM variable=fifo_B_PE_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_0_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_1_1;
  #pragma HLS STREAM variable=fifo_B_PE_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_1_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_2_1;
  #pragma HLS STREAM variable=fifo_B_PE_2_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_2_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_3_1;
  #pragma HLS STREAM variable=fifo_B_PE_3_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_3_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_4_1;
  #pragma HLS STREAM variable=fifo_B_PE_4_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_4_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_5_1;
  #pragma HLS STREAM variable=fifo_B_PE_5_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_5_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_6_1;
  #pragma HLS STREAM variable=fifo_B_PE_6_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_6_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_7_1;
  #pragma HLS STREAM variable=fifo_B_PE_7_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_7_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_8_1;
  #pragma HLS STREAM variable=fifo_B_PE_8_1 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_8_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_0_2;
  #pragma HLS STREAM variable=fifo_B_PE_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_0_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_1_2;
  #pragma HLS STREAM variable=fifo_B_PE_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_1_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_2_2;
  #pragma HLS STREAM variable=fifo_B_PE_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_2_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_3_2;
  #pragma HLS STREAM variable=fifo_B_PE_3_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_3_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_4_2;
  #pragma HLS STREAM variable=fifo_B_PE_4_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_4_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_5_2;
  #pragma HLS STREAM variable=fifo_B_PE_5_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_5_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_6_2;
  #pragma HLS STREAM variable=fifo_B_PE_6_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_6_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_7_2;
  #pragma HLS STREAM variable=fifo_B_PE_7_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_7_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_8_2;
  #pragma HLS STREAM variable=fifo_B_PE_8_2 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_8_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_0_3;
  #pragma HLS STREAM variable=fifo_B_PE_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_0_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_1_3;
  #pragma HLS STREAM variable=fifo_B_PE_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_1_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_2_3;
  #pragma HLS STREAM variable=fifo_B_PE_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_2_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_3_3;
  #pragma HLS STREAM variable=fifo_B_PE_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_3_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_4_3;
  #pragma HLS STREAM variable=fifo_B_PE_4_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_4_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_5_3;
  #pragma HLS STREAM variable=fifo_B_PE_5_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_5_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_6_3;
  #pragma HLS STREAM variable=fifo_B_PE_6_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_6_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_7_3;
  #pragma HLS STREAM variable=fifo_B_PE_7_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_7_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_8_3;
  #pragma HLS STREAM variable=fifo_B_PE_8_3 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_8_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_0_4;
  #pragma HLS STREAM variable=fifo_B_PE_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_0_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_1_4;
  #pragma HLS STREAM variable=fifo_B_PE_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_1_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_2_4;
  #pragma HLS STREAM variable=fifo_B_PE_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_2_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_3_4;
  #pragma HLS STREAM variable=fifo_B_PE_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_3_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_4_4;
  #pragma HLS STREAM variable=fifo_B_PE_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_4_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_5_4;
  #pragma HLS STREAM variable=fifo_B_PE_5_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_5_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_6_4;
  #pragma HLS STREAM variable=fifo_B_PE_6_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_6_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_7_4;
  #pragma HLS STREAM variable=fifo_B_PE_7_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_7_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_8_4;
  #pragma HLS STREAM variable=fifo_B_PE_8_4 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_8_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_0_5;
  #pragma HLS STREAM variable=fifo_B_PE_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_0_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_1_5;
  #pragma HLS STREAM variable=fifo_B_PE_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_1_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_2_5;
  #pragma HLS STREAM variable=fifo_B_PE_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_2_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_3_5;
  #pragma HLS STREAM variable=fifo_B_PE_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_3_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_4_5;
  #pragma HLS STREAM variable=fifo_B_PE_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_4_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_5_5;
  #pragma HLS STREAM variable=fifo_B_PE_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_5_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_6_5;
  #pragma HLS STREAM variable=fifo_B_PE_6_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_6_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_7_5;
  #pragma HLS STREAM variable=fifo_B_PE_7_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_7_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_8_5;
  #pragma HLS STREAM variable=fifo_B_PE_8_5 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_8_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_0_6;
  #pragma HLS STREAM variable=fifo_B_PE_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_0_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_1_6;
  #pragma HLS STREAM variable=fifo_B_PE_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_1_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_2_6;
  #pragma HLS STREAM variable=fifo_B_PE_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_2_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_3_6;
  #pragma HLS STREAM variable=fifo_B_PE_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_3_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_4_6;
  #pragma HLS STREAM variable=fifo_B_PE_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_4_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_5_6;
  #pragma HLS STREAM variable=fifo_B_PE_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_5_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_6_6;
  #pragma HLS STREAM variable=fifo_B_PE_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_6_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_7_6;
  #pragma HLS STREAM variable=fifo_B_PE_7_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_7_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_8_6;
  #pragma HLS STREAM variable=fifo_B_PE_8_6 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_8_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_0_7;
  #pragma HLS STREAM variable=fifo_B_PE_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_0_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_1_7;
  #pragma HLS STREAM variable=fifo_B_PE_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_1_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_2_7;
  #pragma HLS STREAM variable=fifo_B_PE_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_2_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_3_7;
  #pragma HLS STREAM variable=fifo_B_PE_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_3_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_4_7;
  #pragma HLS STREAM variable=fifo_B_PE_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_4_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_5_7;
  #pragma HLS STREAM variable=fifo_B_PE_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_5_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_6_7;
  #pragma HLS STREAM variable=fifo_B_PE_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_6_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_7_7;
  #pragma HLS STREAM variable=fifo_B_PE_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_7_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<char> fifo_B_PE_8_7;
  #pragma HLS STREAM variable=fifo_B_PE_8_7 depth=2
  #pragma HLS RESOURCE variable=fifo_B_PE_8_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_0;
  #pragma HLS STREAM variable=fifo_C_drain_PE_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_0_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_0;
  #pragma HLS STREAM variable=fifo_C_drain_PE_1_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_1_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_0;
  #pragma HLS STREAM variable=fifo_C_drain_PE_2_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_2_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_0;
  #pragma HLS STREAM variable=fifo_C_drain_PE_3_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_3_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_4_0;
  #pragma HLS STREAM variable=fifo_C_drain_PE_4_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_4_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_5_0;
  #pragma HLS STREAM variable=fifo_C_drain_PE_5_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_5_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_6_0;
  #pragma HLS STREAM variable=fifo_C_drain_PE_6_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_6_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_7_0;
  #pragma HLS STREAM variable=fifo_C_drain_PE_7_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_7_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_1;
  #pragma HLS STREAM variable=fifo_C_drain_PE_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_0_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_1;
  #pragma HLS STREAM variable=fifo_C_drain_PE_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_1_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_1;
  #pragma HLS STREAM variable=fifo_C_drain_PE_2_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_2_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_1;
  #pragma HLS STREAM variable=fifo_C_drain_PE_3_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_3_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_4_1;
  #pragma HLS STREAM variable=fifo_C_drain_PE_4_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_4_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_5_1;
  #pragma HLS STREAM variable=fifo_C_drain_PE_5_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_5_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_6_1;
  #pragma HLS STREAM variable=fifo_C_drain_PE_6_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_6_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_7_1;
  #pragma HLS STREAM variable=fifo_C_drain_PE_7_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_7_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_2;
  #pragma HLS STREAM variable=fifo_C_drain_PE_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_0_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_2;
  #pragma HLS STREAM variable=fifo_C_drain_PE_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_1_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_2;
  #pragma HLS STREAM variable=fifo_C_drain_PE_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_2_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_2;
  #pragma HLS STREAM variable=fifo_C_drain_PE_3_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_3_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_4_2;
  #pragma HLS STREAM variable=fifo_C_drain_PE_4_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_4_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_5_2;
  #pragma HLS STREAM variable=fifo_C_drain_PE_5_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_5_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_6_2;
  #pragma HLS STREAM variable=fifo_C_drain_PE_6_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_6_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_7_2;
  #pragma HLS STREAM variable=fifo_C_drain_PE_7_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_7_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_3;
  #pragma HLS STREAM variable=fifo_C_drain_PE_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_0_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_3;
  #pragma HLS STREAM variable=fifo_C_drain_PE_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_1_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_3;
  #pragma HLS STREAM variable=fifo_C_drain_PE_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_2_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_3;
  #pragma HLS STREAM variable=fifo_C_drain_PE_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_3_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_4_3;
  #pragma HLS STREAM variable=fifo_C_drain_PE_4_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_4_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_5_3;
  #pragma HLS STREAM variable=fifo_C_drain_PE_5_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_5_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_6_3;
  #pragma HLS STREAM variable=fifo_C_drain_PE_6_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_6_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_7_3;
  #pragma HLS STREAM variable=fifo_C_drain_PE_7_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_7_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_4;
  #pragma HLS STREAM variable=fifo_C_drain_PE_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_0_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_4;
  #pragma HLS STREAM variable=fifo_C_drain_PE_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_1_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_4;
  #pragma HLS STREAM variable=fifo_C_drain_PE_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_2_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_4;
  #pragma HLS STREAM variable=fifo_C_drain_PE_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_3_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_4_4;
  #pragma HLS STREAM variable=fifo_C_drain_PE_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_4_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_5_4;
  #pragma HLS STREAM variable=fifo_C_drain_PE_5_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_5_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_6_4;
  #pragma HLS STREAM variable=fifo_C_drain_PE_6_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_6_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_7_4;
  #pragma HLS STREAM variable=fifo_C_drain_PE_7_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_7_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_5;
  #pragma HLS STREAM variable=fifo_C_drain_PE_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_0_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_5;
  #pragma HLS STREAM variable=fifo_C_drain_PE_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_1_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_5;
  #pragma HLS STREAM variable=fifo_C_drain_PE_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_2_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_5;
  #pragma HLS STREAM variable=fifo_C_drain_PE_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_3_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_4_5;
  #pragma HLS STREAM variable=fifo_C_drain_PE_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_4_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_5_5;
  #pragma HLS STREAM variable=fifo_C_drain_PE_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_5_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_6_5;
  #pragma HLS STREAM variable=fifo_C_drain_PE_6_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_6_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_7_5;
  #pragma HLS STREAM variable=fifo_C_drain_PE_7_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_7_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_6;
  #pragma HLS STREAM variable=fifo_C_drain_PE_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_0_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_6;
  #pragma HLS STREAM variable=fifo_C_drain_PE_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_1_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_6;
  #pragma HLS STREAM variable=fifo_C_drain_PE_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_2_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_6;
  #pragma HLS STREAM variable=fifo_C_drain_PE_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_3_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_4_6;
  #pragma HLS STREAM variable=fifo_C_drain_PE_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_4_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_5_6;
  #pragma HLS STREAM variable=fifo_C_drain_PE_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_5_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_6_6;
  #pragma HLS STREAM variable=fifo_C_drain_PE_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_6_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_7_6;
  #pragma HLS STREAM variable=fifo_C_drain_PE_7_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_7_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_7;
  #pragma HLS STREAM variable=fifo_C_drain_PE_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_0_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_7;
  #pragma HLS STREAM variable=fifo_C_drain_PE_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_1_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_7;
  #pragma HLS STREAM variable=fifo_C_drain_PE_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_2_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_7;
  #pragma HLS STREAM variable=fifo_C_drain_PE_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_3_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_4_7;
  #pragma HLS STREAM variable=fifo_C_drain_PE_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_4_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_5_7;
  #pragma HLS STREAM variable=fifo_C_drain_PE_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_5_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_6_7;
  #pragma HLS STREAM variable=fifo_C_drain_PE_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_6_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<int> fifo_C_drain_PE_7_7;
  #pragma HLS STREAM variable=fifo_C_drain_PE_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_PE_7_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_0 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_1 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_2 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_3 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_4 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_5 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_6 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_0_8 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_0 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_1 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_2 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_3 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_4 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_5 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_6 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_1_8 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_0 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_1 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_2 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_3 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_4 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_5 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_6 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_2_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_2_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_2_8 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_0 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_1 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_2 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_3 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_4 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_5 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_6 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_3_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_3_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_3_8 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_0 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_1 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_2 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_3 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_4 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_5 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_6 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_4_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_4_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_4_8 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_0 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_1 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_2 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_3 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_4 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_5 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_6 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_5_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_5_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_5_8 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_0 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_1 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_2 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_3 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_4 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_5 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_6 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_6_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_6_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_6_8 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_0 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_1 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_2 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_3 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_4 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_5 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_6 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_7 core=FIFO_SRL
  /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_7_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L1_out_7_8 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_0;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_0 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_0 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_1;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_1 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_1 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_2;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_2 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_2 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_3;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_3 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_3 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_4;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_4 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_4 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_5;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_5 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_5 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_6;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_6 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_6 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_7;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_7 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_7 core=FIFO_SRL
  /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_8;
  #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_8 depth=2
  #pragma HLS RESOURCE variable=fifo_C_drain_C_drain_IO_L2_out_8 core=FIFO_SRL
  /* FIFO Declaration */

  /* Module Call */
  A_IO_L3_in(
    /* array */ A,
    /* fifo */ fifo_A_A_IO_L2_in_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in(
    /* module id */ 0,
    /* fifo */ fifo_A_A_IO_L2_in_0,
    /* fifo */ fifo_A_A_IO_L2_in_1,
    /* fifo */ fifo_A_PE_0_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in(
    /* module id */ 1,
    /* fifo */ fifo_A_A_IO_L2_in_1,
    /* fifo */ fifo_A_A_IO_L2_in_2,
    /* fifo */ fifo_A_PE_1_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in(
    /* module id */ 2,
    /* fifo */ fifo_A_A_IO_L2_in_2,
    /* fifo */ fifo_A_A_IO_L2_in_3,
    /* fifo */ fifo_A_PE_2_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in(
    /* module id */ 3,
    /* fifo */ fifo_A_A_IO_L2_in_3,
    /* fifo */ fifo_A_A_IO_L2_in_4,
    /* fifo */ fifo_A_PE_3_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in(
    /* module id */ 4,
    /* fifo */ fifo_A_A_IO_L2_in_4,
    /* fifo */ fifo_A_A_IO_L2_in_5,
    /* fifo */ fifo_A_PE_4_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in(
    /* module id */ 5,
    /* fifo */ fifo_A_A_IO_L2_in_5,
    /* fifo */ fifo_A_A_IO_L2_in_6,
    /* fifo */ fifo_A_PE_5_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in(
    /* module id */ 6,
    /* fifo */ fifo_A_A_IO_L2_in_6,
    /* fifo */ fifo_A_A_IO_L2_in_7,
    /* fifo */ fifo_A_PE_6_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in_boundary(
    /* module id */ 7,
    /* fifo */ fifo_A_A_IO_L2_in_7,
    /* fifo */ fifo_A_PE_7_0
  );
  /* Module Call */

  /* Module Call */
  B_IO_L3_in(
    /* array */ B,
    /* fifo */ fifo_B_B_IO_L2_in_0
  );
  /* Module Call */

  /* Module Call */
  B_IO_L2_in(
    /* module id */ 0,
    /* fifo */ fifo_B_B_IO_L2_in_0,
    /* fifo */ fifo_B_B_IO_L2_in_1,
    /* fifo */ fifo_B_PE_0_0
  );
  /* Module Call */

  /* Module Call */
  B_IO_L2_in(
    /* module id */ 1,
    /* fifo */ fifo_B_B_IO_L2_in_1,
    /* fifo */ fifo_B_B_IO_L2_in_2,
    /* fifo */ fifo_B_PE_0_1
  );
  /* Module Call */

  /* Module Call */
  B_IO_L2_in(
    /* module id */ 2,
    /* fifo */ fifo_B_B_IO_L2_in_2,
    /* fifo */ fifo_B_B_IO_L2_in_3,
    /* fifo */ fifo_B_PE_0_2
  );
  /* Module Call */

  /* Module Call */
  B_IO_L2_in(
    /* module id */ 3,
    /* fifo */ fifo_B_B_IO_L2_in_3,
    /* fifo */ fifo_B_B_IO_L2_in_4,
    /* fifo */ fifo_B_PE_0_3
  );
  /* Module Call */

  /* Module Call */
  B_IO_L2_in(
    /* module id */ 4,
    /* fifo */ fifo_B_B_IO_L2_in_4,
    /* fifo */ fifo_B_B_IO_L2_in_5,
    /* fifo */ fifo_B_PE_0_4
  );
  /* Module Call */

  /* Module Call */
  B_IO_L2_in(
    /* module id */ 5,
    /* fifo */ fifo_B_B_IO_L2_in_5,
    /* fifo */ fifo_B_B_IO_L2_in_6,
    /* fifo */ fifo_B_PE_0_5
  );
  /* Module Call */

  /* Module Call */
  B_IO_L2_in(
    /* module id */ 6,
    /* fifo */ fifo_B_B_IO_L2_in_6,
    /* fifo */ fifo_B_B_IO_L2_in_7,
    /* fifo */ fifo_B_PE_0_6
  );
  /* Module Call */

  /* Module Call */
  B_IO_L2_in_boundary(
    /* module id */ 7,
    /* fifo */ fifo_B_B_IO_L2_in_7,
    /* fifo */ fifo_B_PE_0_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 0,
    /* module id */ 0,
    /* fifo */ fifo_A_PE_0_0,
    /* fifo */ fifo_A_PE_0_1,
    /* fifo */ fifo_B_PE_0_0,
    /* fifo */ fifo_B_PE_1_0,
    /* fifo */ fifo_C_drain_PE_0_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 0,
    /* module id */ 1,
    /* fifo */ fifo_A_PE_0_1,
    /* fifo */ fifo_A_PE_0_2,
    /* fifo */ fifo_B_PE_0_1,
    /* fifo */ fifo_B_PE_1_1,
    /* fifo */ fifo_C_drain_PE_0_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 0,
    /* module id */ 2,
    /* fifo */ fifo_A_PE_0_2,
    /* fifo */ fifo_A_PE_0_3,
    /* fifo */ fifo_B_PE_0_2,
    /* fifo */ fifo_B_PE_1_2,
    /* fifo */ fifo_C_drain_PE_0_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 0,
    /* module id */ 3,
    /* fifo */ fifo_A_PE_0_3,
    /* fifo */ fifo_A_PE_0_4,
    /* fifo */ fifo_B_PE_0_3,
    /* fifo */ fifo_B_PE_1_3,
    /* fifo */ fifo_C_drain_PE_0_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 0,
    /* module id */ 4,
    /* fifo */ fifo_A_PE_0_4,
    /* fifo */ fifo_A_PE_0_5,
    /* fifo */ fifo_B_PE_0_4,
    /* fifo */ fifo_B_PE_1_4,
    /* fifo */ fifo_C_drain_PE_0_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 0,
    /* module id */ 5,
    /* fifo */ fifo_A_PE_0_5,
    /* fifo */ fifo_A_PE_0_6,
    /* fifo */ fifo_B_PE_0_5,
    /* fifo */ fifo_B_PE_1_5,
    /* fifo */ fifo_C_drain_PE_0_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 0,
    /* module id */ 6,
    /* fifo */ fifo_A_PE_0_6,
    /* fifo */ fifo_A_PE_0_7,
    /* fifo */ fifo_B_PE_0_6,
    /* fifo */ fifo_B_PE_1_6,
    /* fifo */ fifo_C_drain_PE_0_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 0,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_0_7,
    /* fifo */ fifo_A_PE_0_8,
    /* fifo */ fifo_B_PE_0_7,
    /* fifo */ fifo_B_PE_1_7,
    /* fifo */ fifo_C_drain_PE_0_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 1,
    /* module id */ 0,
    /* fifo */ fifo_A_PE_1_0,
    /* fifo */ fifo_A_PE_1_1,
    /* fifo */ fifo_B_PE_1_0,
    /* fifo */ fifo_B_PE_2_0,
    /* fifo */ fifo_C_drain_PE_1_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 1,
    /* module id */ 1,
    /* fifo */ fifo_A_PE_1_1,
    /* fifo */ fifo_A_PE_1_2,
    /* fifo */ fifo_B_PE_1_1,
    /* fifo */ fifo_B_PE_2_1,
    /* fifo */ fifo_C_drain_PE_1_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 1,
    /* module id */ 2,
    /* fifo */ fifo_A_PE_1_2,
    /* fifo */ fifo_A_PE_1_3,
    /* fifo */ fifo_B_PE_1_2,
    /* fifo */ fifo_B_PE_2_2,
    /* fifo */ fifo_C_drain_PE_1_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 1,
    /* module id */ 3,
    /* fifo */ fifo_A_PE_1_3,
    /* fifo */ fifo_A_PE_1_4,
    /* fifo */ fifo_B_PE_1_3,
    /* fifo */ fifo_B_PE_2_3,
    /* fifo */ fifo_C_drain_PE_1_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 1,
    /* module id */ 4,
    /* fifo */ fifo_A_PE_1_4,
    /* fifo */ fifo_A_PE_1_5,
    /* fifo */ fifo_B_PE_1_4,
    /* fifo */ fifo_B_PE_2_4,
    /* fifo */ fifo_C_drain_PE_1_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 1,
    /* module id */ 5,
    /* fifo */ fifo_A_PE_1_5,
    /* fifo */ fifo_A_PE_1_6,
    /* fifo */ fifo_B_PE_1_5,
    /* fifo */ fifo_B_PE_2_5,
    /* fifo */ fifo_C_drain_PE_1_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 1,
    /* module id */ 6,
    /* fifo */ fifo_A_PE_1_6,
    /* fifo */ fifo_A_PE_1_7,
    /* fifo */ fifo_B_PE_1_6,
    /* fifo */ fifo_B_PE_2_6,
    /* fifo */ fifo_C_drain_PE_1_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 1,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_1_7,
    /* fifo */ fifo_A_PE_1_8,
    /* fifo */ fifo_B_PE_1_7,
    /* fifo */ fifo_B_PE_2_7,
    /* fifo */ fifo_C_drain_PE_1_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 2,
    /* module id */ 0,
    /* fifo */ fifo_A_PE_2_0,
    /* fifo */ fifo_A_PE_2_1,
    /* fifo */ fifo_B_PE_2_0,
    /* fifo */ fifo_B_PE_3_0,
    /* fifo */ fifo_C_drain_PE_2_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 2,
    /* module id */ 1,
    /* fifo */ fifo_A_PE_2_1,
    /* fifo */ fifo_A_PE_2_2,
    /* fifo */ fifo_B_PE_2_1,
    /* fifo */ fifo_B_PE_3_1,
    /* fifo */ fifo_C_drain_PE_2_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 2,
    /* module id */ 2,
    /* fifo */ fifo_A_PE_2_2,
    /* fifo */ fifo_A_PE_2_3,
    /* fifo */ fifo_B_PE_2_2,
    /* fifo */ fifo_B_PE_3_2,
    /* fifo */ fifo_C_drain_PE_2_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 2,
    /* module id */ 3,
    /* fifo */ fifo_A_PE_2_3,
    /* fifo */ fifo_A_PE_2_4,
    /* fifo */ fifo_B_PE_2_3,
    /* fifo */ fifo_B_PE_3_3,
    /* fifo */ fifo_C_drain_PE_2_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 2,
    /* module id */ 4,
    /* fifo */ fifo_A_PE_2_4,
    /* fifo */ fifo_A_PE_2_5,
    /* fifo */ fifo_B_PE_2_4,
    /* fifo */ fifo_B_PE_3_4,
    /* fifo */ fifo_C_drain_PE_2_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 2,
    /* module id */ 5,
    /* fifo */ fifo_A_PE_2_5,
    /* fifo */ fifo_A_PE_2_6,
    /* fifo */ fifo_B_PE_2_5,
    /* fifo */ fifo_B_PE_3_5,
    /* fifo */ fifo_C_drain_PE_2_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 2,
    /* module id */ 6,
    /* fifo */ fifo_A_PE_2_6,
    /* fifo */ fifo_A_PE_2_7,
    /* fifo */ fifo_B_PE_2_6,
    /* fifo */ fifo_B_PE_3_6,
    /* fifo */ fifo_C_drain_PE_2_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 2,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_2_7,
    /* fifo */ fifo_A_PE_2_8,
    /* fifo */ fifo_B_PE_2_7,
    /* fifo */ fifo_B_PE_3_7,
    /* fifo */ fifo_C_drain_PE_2_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 3,
    /* module id */ 0,
    /* fifo */ fifo_A_PE_3_0,
    /* fifo */ fifo_A_PE_3_1,
    /* fifo */ fifo_B_PE_3_0,
    /* fifo */ fifo_B_PE_4_0,
    /* fifo */ fifo_C_drain_PE_3_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 3,
    /* module id */ 1,
    /* fifo */ fifo_A_PE_3_1,
    /* fifo */ fifo_A_PE_3_2,
    /* fifo */ fifo_B_PE_3_1,
    /* fifo */ fifo_B_PE_4_1,
    /* fifo */ fifo_C_drain_PE_3_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 3,
    /* module id */ 2,
    /* fifo */ fifo_A_PE_3_2,
    /* fifo */ fifo_A_PE_3_3,
    /* fifo */ fifo_B_PE_3_2,
    /* fifo */ fifo_B_PE_4_2,
    /* fifo */ fifo_C_drain_PE_3_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 3,
    /* module id */ 3,
    /* fifo */ fifo_A_PE_3_3,
    /* fifo */ fifo_A_PE_3_4,
    /* fifo */ fifo_B_PE_3_3,
    /* fifo */ fifo_B_PE_4_3,
    /* fifo */ fifo_C_drain_PE_3_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 3,
    /* module id */ 4,
    /* fifo */ fifo_A_PE_3_4,
    /* fifo */ fifo_A_PE_3_5,
    /* fifo */ fifo_B_PE_3_4,
    /* fifo */ fifo_B_PE_4_4,
    /* fifo */ fifo_C_drain_PE_3_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 3,
    /* module id */ 5,
    /* fifo */ fifo_A_PE_3_5,
    /* fifo */ fifo_A_PE_3_6,
    /* fifo */ fifo_B_PE_3_5,
    /* fifo */ fifo_B_PE_4_5,
    /* fifo */ fifo_C_drain_PE_3_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 3,
    /* module id */ 6,
    /* fifo */ fifo_A_PE_3_6,
    /* fifo */ fifo_A_PE_3_7,
    /* fifo */ fifo_B_PE_3_6,
    /* fifo */ fifo_B_PE_4_6,
    /* fifo */ fifo_C_drain_PE_3_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 3,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_3_7,
    /* fifo */ fifo_A_PE_3_8,
    /* fifo */ fifo_B_PE_3_7,
    /* fifo */ fifo_B_PE_4_7,
    /* fifo */ fifo_C_drain_PE_3_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 4,
    /* module id */ 0,
    /* fifo */ fifo_A_PE_4_0,
    /* fifo */ fifo_A_PE_4_1,
    /* fifo */ fifo_B_PE_4_0,
    /* fifo */ fifo_B_PE_5_0,
    /* fifo */ fifo_C_drain_PE_4_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 4,
    /* module id */ 1,
    /* fifo */ fifo_A_PE_4_1,
    /* fifo */ fifo_A_PE_4_2,
    /* fifo */ fifo_B_PE_4_1,
    /* fifo */ fifo_B_PE_5_1,
    /* fifo */ fifo_C_drain_PE_4_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 4,
    /* module id */ 2,
    /* fifo */ fifo_A_PE_4_2,
    /* fifo */ fifo_A_PE_4_3,
    /* fifo */ fifo_B_PE_4_2,
    /* fifo */ fifo_B_PE_5_2,
    /* fifo */ fifo_C_drain_PE_4_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 4,
    /* module id */ 3,
    /* fifo */ fifo_A_PE_4_3,
    /* fifo */ fifo_A_PE_4_4,
    /* fifo */ fifo_B_PE_4_3,
    /* fifo */ fifo_B_PE_5_3,
    /* fifo */ fifo_C_drain_PE_4_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 4,
    /* module id */ 4,
    /* fifo */ fifo_A_PE_4_4,
    /* fifo */ fifo_A_PE_4_5,
    /* fifo */ fifo_B_PE_4_4,
    /* fifo */ fifo_B_PE_5_4,
    /* fifo */ fifo_C_drain_PE_4_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 4,
    /* module id */ 5,
    /* fifo */ fifo_A_PE_4_5,
    /* fifo */ fifo_A_PE_4_6,
    /* fifo */ fifo_B_PE_4_5,
    /* fifo */ fifo_B_PE_5_5,
    /* fifo */ fifo_C_drain_PE_4_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 4,
    /* module id */ 6,
    /* fifo */ fifo_A_PE_4_6,
    /* fifo */ fifo_A_PE_4_7,
    /* fifo */ fifo_B_PE_4_6,
    /* fifo */ fifo_B_PE_5_6,
    /* fifo */ fifo_C_drain_PE_4_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 4,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_4_7,
    /* fifo */ fifo_A_PE_4_8,
    /* fifo */ fifo_B_PE_4_7,
    /* fifo */ fifo_B_PE_5_7,
    /* fifo */ fifo_C_drain_PE_4_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 5,
    /* module id */ 0,
    /* fifo */ fifo_A_PE_5_0,
    /* fifo */ fifo_A_PE_5_1,
    /* fifo */ fifo_B_PE_5_0,
    /* fifo */ fifo_B_PE_6_0,
    /* fifo */ fifo_C_drain_PE_5_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 5,
    /* module id */ 1,
    /* fifo */ fifo_A_PE_5_1,
    /* fifo */ fifo_A_PE_5_2,
    /* fifo */ fifo_B_PE_5_1,
    /* fifo */ fifo_B_PE_6_1,
    /* fifo */ fifo_C_drain_PE_5_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 5,
    /* module id */ 2,
    /* fifo */ fifo_A_PE_5_2,
    /* fifo */ fifo_A_PE_5_3,
    /* fifo */ fifo_B_PE_5_2,
    /* fifo */ fifo_B_PE_6_2,
    /* fifo */ fifo_C_drain_PE_5_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 5,
    /* module id */ 3,
    /* fifo */ fifo_A_PE_5_3,
    /* fifo */ fifo_A_PE_5_4,
    /* fifo */ fifo_B_PE_5_3,
    /* fifo */ fifo_B_PE_6_3,
    /* fifo */ fifo_C_drain_PE_5_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 5,
    /* module id */ 4,
    /* fifo */ fifo_A_PE_5_4,
    /* fifo */ fifo_A_PE_5_5,
    /* fifo */ fifo_B_PE_5_4,
    /* fifo */ fifo_B_PE_6_4,
    /* fifo */ fifo_C_drain_PE_5_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 5,
    /* module id */ 5,
    /* fifo */ fifo_A_PE_5_5,
    /* fifo */ fifo_A_PE_5_6,
    /* fifo */ fifo_B_PE_5_5,
    /* fifo */ fifo_B_PE_6_5,
    /* fifo */ fifo_C_drain_PE_5_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 5,
    /* module id */ 6,
    /* fifo */ fifo_A_PE_5_6,
    /* fifo */ fifo_A_PE_5_7,
    /* fifo */ fifo_B_PE_5_6,
    /* fifo */ fifo_B_PE_6_6,
    /* fifo */ fifo_C_drain_PE_5_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 5,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_5_7,
    /* fifo */ fifo_A_PE_5_8,
    /* fifo */ fifo_B_PE_5_7,
    /* fifo */ fifo_B_PE_6_7,
    /* fifo */ fifo_C_drain_PE_5_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 6,
    /* module id */ 0,
    /* fifo */ fifo_A_PE_6_0,
    /* fifo */ fifo_A_PE_6_1,
    /* fifo */ fifo_B_PE_6_0,
    /* fifo */ fifo_B_PE_7_0,
    /* fifo */ fifo_C_drain_PE_6_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 6,
    /* module id */ 1,
    /* fifo */ fifo_A_PE_6_1,
    /* fifo */ fifo_A_PE_6_2,
    /* fifo */ fifo_B_PE_6_1,
    /* fifo */ fifo_B_PE_7_1,
    /* fifo */ fifo_C_drain_PE_6_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 6,
    /* module id */ 2,
    /* fifo */ fifo_A_PE_6_2,
    /* fifo */ fifo_A_PE_6_3,
    /* fifo */ fifo_B_PE_6_2,
    /* fifo */ fifo_B_PE_7_2,
    /* fifo */ fifo_C_drain_PE_6_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 6,
    /* module id */ 3,
    /* fifo */ fifo_A_PE_6_3,
    /* fifo */ fifo_A_PE_6_4,
    /* fifo */ fifo_B_PE_6_3,
    /* fifo */ fifo_B_PE_7_3,
    /* fifo */ fifo_C_drain_PE_6_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 6,
    /* module id */ 4,
    /* fifo */ fifo_A_PE_6_4,
    /* fifo */ fifo_A_PE_6_5,
    /* fifo */ fifo_B_PE_6_4,
    /* fifo */ fifo_B_PE_7_4,
    /* fifo */ fifo_C_drain_PE_6_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 6,
    /* module id */ 5,
    /* fifo */ fifo_A_PE_6_5,
    /* fifo */ fifo_A_PE_6_6,
    /* fifo */ fifo_B_PE_6_5,
    /* fifo */ fifo_B_PE_7_5,
    /* fifo */ fifo_C_drain_PE_6_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 6,
    /* module id */ 6,
    /* fifo */ fifo_A_PE_6_6,
    /* fifo */ fifo_A_PE_6_7,
    /* fifo */ fifo_B_PE_6_6,
    /* fifo */ fifo_B_PE_7_6,
    /* fifo */ fifo_C_drain_PE_6_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 6,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_6_7,
    /* fifo */ fifo_A_PE_6_8,
    /* fifo */ fifo_B_PE_6_7,
    /* fifo */ fifo_B_PE_7_7,
    /* fifo */ fifo_C_drain_PE_6_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 7,
    /* module id */ 0,
    /* fifo */ fifo_A_PE_7_0,
    /* fifo */ fifo_A_PE_7_1,
    /* fifo */ fifo_B_PE_7_0,
    /* fifo */ fifo_B_PE_8_0,
    /* fifo */ fifo_C_drain_PE_7_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 7,
    /* module id */ 1,
    /* fifo */ fifo_A_PE_7_1,
    /* fifo */ fifo_A_PE_7_2,
    /* fifo */ fifo_B_PE_7_1,
    /* fifo */ fifo_B_PE_8_1,
    /* fifo */ fifo_C_drain_PE_7_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 7,
    /* module id */ 2,
    /* fifo */ fifo_A_PE_7_2,
    /* fifo */ fifo_A_PE_7_3,
    /* fifo */ fifo_B_PE_7_2,
    /* fifo */ fifo_B_PE_8_2,
    /* fifo */ fifo_C_drain_PE_7_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 7,
    /* module id */ 3,
    /* fifo */ fifo_A_PE_7_3,
    /* fifo */ fifo_A_PE_7_4,
    /* fifo */ fifo_B_PE_7_3,
    /* fifo */ fifo_B_PE_8_3,
    /* fifo */ fifo_C_drain_PE_7_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 7,
    /* module id */ 4,
    /* fifo */ fifo_A_PE_7_4,
    /* fifo */ fifo_A_PE_7_5,
    /* fifo */ fifo_B_PE_7_4,
    /* fifo */ fifo_B_PE_8_4,
    /* fifo */ fifo_C_drain_PE_7_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 7,
    /* module id */ 5,
    /* fifo */ fifo_A_PE_7_5,
    /* fifo */ fifo_A_PE_7_6,
    /* fifo */ fifo_B_PE_7_5,
    /* fifo */ fifo_B_PE_8_5,
    /* fifo */ fifo_C_drain_PE_7_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 7,
    /* module id */ 6,
    /* fifo */ fifo_A_PE_7_6,
    /* fifo */ fifo_A_PE_7_7,
    /* fifo */ fifo_B_PE_7_6,
    /* fifo */ fifo_B_PE_8_6,
    /* fifo */ fifo_C_drain_PE_7_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper(
    /* module id */ 7,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_7_7,
    /* fifo */ fifo_A_PE_7_8,
    /* fifo */ fifo_B_PE_7_7,
    /* fifo */ fifo_B_PE_8_7,
    /* fifo */ fifo_C_drain_PE_7_7
  );
  /* Module Call */

  /* Module Call */
  A_PE_dummy_in(
    /* module id */ 0,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_0_8
  );
  /* Module Call */

  /* Module Call */
  A_PE_dummy_in(
    /* module id */ 1,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_1_8
  );
  /* Module Call */

  /* Module Call */
  A_PE_dummy_in(
    /* module id */ 2,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_2_8
  );
  /* Module Call */

  /* Module Call */
  A_PE_dummy_in(
    /* module id */ 3,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_3_8
  );
  /* Module Call */

  /* Module Call */
  A_PE_dummy_in(
    /* module id */ 4,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_4_8
  );
  /* Module Call */

  /* Module Call */
  A_PE_dummy_in(
    /* module id */ 5,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_5_8
  );
  /* Module Call */

  /* Module Call */
  A_PE_dummy_in(
    /* module id */ 6,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_6_8
  );
  /* Module Call */

  /* Module Call */
  A_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 7,
    /* fifo */ fifo_A_PE_7_8
  );
  /* Module Call */

  /* Module Call */
  B_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 0,
    /* fifo */ fifo_B_PE_8_0
  );
  /* Module Call */

  /* Module Call */
  B_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 1,
    /* fifo */ fifo_B_PE_8_1
  );
  /* Module Call */

  /* Module Call */
  B_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 2,
    /* fifo */ fifo_B_PE_8_2
  );
  /* Module Call */

  /* Module Call */
  B_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 3,
    /* fifo */ fifo_B_PE_8_3
  );
  /* Module Call */

  /* Module Call */
  B_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 4,
    /* fifo */ fifo_B_PE_8_4
  );
  /* Module Call */

  /* Module Call */
  B_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 5,
    /* fifo */ fifo_B_PE_8_5
  );
  /* Module Call */

  /* Module Call */
  B_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 6,
    /* fifo */ fifo_B_PE_8_6
  );
  /* Module Call */

  /* Module Call */
  B_PE_dummy_in(
    /* module id */ 7,
    /* module id */ 7,
    /* fifo */ fifo_B_PE_8_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_boundary_wrapper(
    /* module id */ 0,
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_7,
    /* fifo */ fifo_C_drain_PE_7_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 0,
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_6,
    /* fifo */ fifo_C_drain_PE_6_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 0,
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_5,
    /* fifo */ fifo_C_drain_PE_5_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 0,
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_4,
    /* fifo */ fifo_C_drain_PE_4_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 0,
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_3,
    /* fifo */ fifo_C_drain_PE_3_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 0,
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_2,
    /* fifo */ fifo_C_drain_PE_2_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 0,
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1,
    /* fifo */ fifo_C_drain_PE_1_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 0,
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0,
    /* fifo */ fifo_C_drain_PE_0_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_boundary_wrapper(
    /* module id */ 1,
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_7,
    /* fifo */ fifo_C_drain_PE_7_1
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 1,
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_6,
    /* fifo */ fifo_C_drain_PE_6_1
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 1,
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_5,
    /* fifo */ fifo_C_drain_PE_5_1
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 1,
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_4,
    /* fifo */ fifo_C_drain_PE_4_1
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 1,
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_3,
    /* fifo */ fifo_C_drain_PE_3_1
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 1,
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_2,
    /* fifo */ fifo_C_drain_PE_2_1
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 1,
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1,
    /* fifo */ fifo_C_drain_PE_1_1
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 1,
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0,
    /* fifo */ fifo_C_drain_PE_0_1
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_boundary_wrapper(
    /* module id */ 2,
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_7,
    /* fifo */ fifo_C_drain_PE_7_2
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 2,
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_6,
    /* fifo */ fifo_C_drain_PE_6_2
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 2,
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_5,
    /* fifo */ fifo_C_drain_PE_5_2
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 2,
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_4,
    /* fifo */ fifo_C_drain_PE_4_2
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 2,
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_3,
    /* fifo */ fifo_C_drain_PE_3_2
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 2,
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_2,
    /* fifo */ fifo_C_drain_PE_2_2
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 2,
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_1,
    /* fifo */ fifo_C_drain_PE_1_2
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 2,
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_0,
    /* fifo */ fifo_C_drain_PE_0_2
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_boundary_wrapper(
    /* module id */ 3,
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_7,
    /* fifo */ fifo_C_drain_PE_7_3
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 3,
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_6,
    /* fifo */ fifo_C_drain_PE_6_3
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 3,
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_5,
    /* fifo */ fifo_C_drain_PE_5_3
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 3,
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_4,
    /* fifo */ fifo_C_drain_PE_4_3
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 3,
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_3,
    /* fifo */ fifo_C_drain_PE_3_3
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 3,
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_2,
    /* fifo */ fifo_C_drain_PE_2_3
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 3,
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_1,
    /* fifo */ fifo_C_drain_PE_1_3
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 3,
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_0,
    /* fifo */ fifo_C_drain_PE_0_3
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_boundary_wrapper(
    /* module id */ 4,
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_7,
    /* fifo */ fifo_C_drain_PE_7_4
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 4,
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_6,
    /* fifo */ fifo_C_drain_PE_6_4
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 4,
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_5,
    /* fifo */ fifo_C_drain_PE_5_4
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 4,
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_4,
    /* fifo */ fifo_C_drain_PE_4_4
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 4,
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_3,
    /* fifo */ fifo_C_drain_PE_3_4
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 4,
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_2,
    /* fifo */ fifo_C_drain_PE_2_4
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 4,
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_1,
    /* fifo */ fifo_C_drain_PE_1_4
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 4,
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_0,
    /* fifo */ fifo_C_drain_PE_0_4
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_boundary_wrapper(
    /* module id */ 5,
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_7,
    /* fifo */ fifo_C_drain_PE_7_5
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 5,
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_6,
    /* fifo */ fifo_C_drain_PE_6_5
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 5,
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_5,
    /* fifo */ fifo_C_drain_PE_5_5
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 5,
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_4,
    /* fifo */ fifo_C_drain_PE_4_5
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 5,
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_3,
    /* fifo */ fifo_C_drain_PE_3_5
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 5,
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_2,
    /* fifo */ fifo_C_drain_PE_2_5
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 5,
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_1,
    /* fifo */ fifo_C_drain_PE_1_5
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 5,
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_0,
    /* fifo */ fifo_C_drain_PE_0_5
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_boundary_wrapper(
    /* module id */ 6,
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_7,
    /* fifo */ fifo_C_drain_PE_7_6
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 6,
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_6,
    /* fifo */ fifo_C_drain_PE_6_6
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 6,
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_5,
    /* fifo */ fifo_C_drain_PE_5_6
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 6,
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_4,
    /* fifo */ fifo_C_drain_PE_4_6
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 6,
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_3,
    /* fifo */ fifo_C_drain_PE_3_6
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 6,
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_2,
    /* fifo */ fifo_C_drain_PE_2_6
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 6,
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_1,
    /* fifo */ fifo_C_drain_PE_1_6
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 6,
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_0,
    /* fifo */ fifo_C_drain_PE_0_6
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_boundary_wrapper(
    /* module id */ 7,
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_7,
    /* fifo */ fifo_C_drain_PE_7_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 7,
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_6,
    /* fifo */ fifo_C_drain_PE_6_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 7,
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_5,
    /* fifo */ fifo_C_drain_PE_5_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 7,
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_4,
    /* fifo */ fifo_C_drain_PE_4_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 7,
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_3,
    /* fifo */ fifo_C_drain_PE_3_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 7,
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_2,
    /* fifo */ fifo_C_drain_PE_2_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 7,
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_1,
    /* fifo */ fifo_C_drain_PE_1_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L1_out_wrapper(
    /* module id */ 7,
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_0,
    /* fifo */ fifo_C_drain_PE_0_7
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L2_out_boundary(
    /* module id */ 7,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_7_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L2_out(
    /* module id */ 6,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_7,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_6_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L2_out(
    /* module id */ 5,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_6,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_5_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L2_out(
    /* module id */ 4,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_5,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_4_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L2_out(
    /* module id */ 3,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_4,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_3_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L2_out(
    /* module id */ 2,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_3,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_2_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L2_out(
    /* module id */ 1,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_2,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L2_out(
    /* module id */ 0,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_0,
    /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0
  );
  /* Module Call */

  /* Module Call */
  C_drain_IO_L3_out(
    /* array */ C,
    /* fifo */ fifo_C_drain_C_drain_IO_L2_out_0
  );
  /* Module Call */

}

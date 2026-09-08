#define AP_INT_MAX_W 1024
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// vpu_op_in_bind: 1 channel(s) x `steps` token(s) of 32 bits, from
// `VProg`, a dense step-major channel-minor stream read 512 bits
// at a time. 17 steps is what this design was elaborated with and
// sizes the burst depth; the loop itself runs as far as the host says.
extern "C" {
void feed_vpu_op_in_bind(const ap_uint<512> *src, int steps, hls::stream<ap_uint<32>> out[1]) {
#pragma HLS interface m_axi port=src offset=direct bundle=gmem depth=2
#pragma HLS interface ap_none port=steps
  const int total = steps * 1;
  ap_uint<512> beat = 0;
  for (int n = 0; n < total; n += 1) {
#pragma HLS pipeline II=1
    if (n % 16 == 0)
      beat = src[n / 16];
    for (int k = 0; k < 1; k++) {
#pragma HLS unroll
      int m = n + k;
      out[m % 1].write(
          beat.range((m % 16) * 32 + 31, (m % 16) * 32));
    }
  }
}
}

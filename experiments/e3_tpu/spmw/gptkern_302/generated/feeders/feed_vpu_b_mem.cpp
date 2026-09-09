#define AP_INT_MAX_W 1024
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// vpu_b_mem: 16 channel(s) x `steps` token(s) of 64 bits, from
// `Bias`, a dense step-major channel-minor stream read 512 bits
// at a time. 1 steps is what this design was elaborated with and
// sizes the burst depth; the loop itself runs as far as the host says.
extern "C" {
void feed_vpu_b_mem(const ap_uint<512> *src, int steps, hls::stream<ap_uint<64>> out[16]) {
#pragma HLS interface m_axi port=src offset=direct bundle=gmem depth=2
#pragma HLS interface ap_none port=steps
  const int total = steps * 16;
  ap_uint<512> beat = 0;
  for (int n = 0; n < total; n += 8) {
#pragma HLS pipeline II=1
    if (n % 8 == 0)
      beat = src[n / 8];
    for (int k = 0; k < 8; k++) {
#pragma HLS unroll
      int m = n + k;
      out[m % 16].write(
          beat.range((m % 8) * 64 + 63, (m % 8) * 64));
    }
  }
}
}

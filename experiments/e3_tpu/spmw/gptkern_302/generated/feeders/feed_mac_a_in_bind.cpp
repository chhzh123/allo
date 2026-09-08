#define AP_INT_MAX_W 1024
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// mac_a_in_bind: 16 channel(s) x `steps` token(s) of 8 bits, from
// `A`, a dense step-major channel-minor stream read 512 bits
// at a time. 32768 steps is what this design was elaborated with and
// sizes the burst depth; the loop itself runs as far as the host says.
extern "C" {
void feed_mac_a_in_bind(const ap_uint<512> *src, int steps, hls::stream<ap_uint<8>> out[16]) {
#pragma HLS interface m_axi port=src offset=direct bundle=gmem depth=8192
#pragma HLS interface ap_none port=steps
  const int total = steps * 16;
  ap_uint<512> beat = 0;
  for (int n = 0; n < total; n += 16) {
#pragma HLS pipeline II=1
    if (n % 64 == 0)
      beat = src[n / 64];
    for (int k = 0; k < 16; k++) {
#pragma HLS unroll
      int m = n + k;
      out[m % 16].write(
          beat.range((m % 64) * 8 + 7, (m % 64) * 8));
    }
  }
}
}

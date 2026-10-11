#define AP_INT_MAX_W 1024
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// mac_p_out_bind: 4 channel(s) x `steps` token(s) of 32 bits, from
// `C`, a dense step-major channel-minor stream read 512 bits
// at a time. 4 steps is what this design was elaborated with and
// sizes the burst depth; the loop itself runs as far as the host says.
extern "C" {
void drain_mac_p_out_bind(ap_uint<512> *dst, int steps, hls::stream<ap_uint<32>> in[4]) {
#pragma HLS interface m_axi port=dst offset=direct bundle=gmem depth=1
#pragma HLS interface ap_none port=steps
  const int total = steps * 4;
  ap_uint<512> beat = 0;
  for (int n = 0; n < total; n += 4) {
#pragma HLS pipeline II=1
    for (int k = 0; k < 4; k++) {
#pragma HLS unroll
      int m = n + k;
      beat.range((m % 16) * 32 + 31, (m % 16) * 32) = in[m % 4].read();
    }
    if ((n + 4) % 16 == 0 || n + 4 >= total)
      dst[n / 16] = beat;
  }
}
}

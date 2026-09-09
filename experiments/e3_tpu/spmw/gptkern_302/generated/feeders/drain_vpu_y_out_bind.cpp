#define AP_INT_MAX_W 1024
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// vpu_y_out_bind: 16 channel(s) x `steps` token(s) of 32 bits, from
// `Y`, a dense step-major channel-minor stream read 512 bits
// at a time. 512 steps is what this design was elaborated with and
// sizes the burst depth; the loop itself runs as far as the host says.
extern "C" {
void drain_vpu_y_out_bind(ap_uint<512> *dst, int steps, hls::stream<ap_uint<32>> in[16]) {
#pragma HLS interface m_axi port=dst offset=direct bundle=gmem depth=512
#pragma HLS interface ap_none port=steps
  const int total = steps * 16;
  ap_uint<512> beat = 0;
  for (int n = 0; n < total; n += 16) {
#pragma HLS pipeline II=1
    for (int k = 0; k < 16; k++) {
#pragma HLS unroll
      int m = n + k;
      beat.range((m % 16) * 32 + 31, (m % 16) * 32) = in[m % 16].read();
    }
    if ((n + 16) % 16 == 0 || n + 16 >= total)
      dst[n / 16] = beat;
  }
}
}

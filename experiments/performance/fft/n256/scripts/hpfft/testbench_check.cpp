// csim gate: 3 transforms through FFT_TOP against a double-precision DFT gold.
// Returns non-zero if the max absolute error exceeds TOL, so csim failing means
// the transform is wrong rather than merely that the program ran.
#include "FFT.h"
#define NTX 3
#define TOL 2e-3

static void fft_sw_gold(complex<dtype_gold>* input, complex<dtype_gold>* output) {
    for (int k = 0; k < FFT_NUM; ++k) {
        complex<dtype_gold> sum = 0;
        for (int n = 0; n < FFT_NUM; ++n) {
            double angle = -2.0 * PI * ((double)k) * ((double)n) / FFT_NUM;
            sum += input[n] * complex<dtype_gold>(cos(angle), sin(angle));
        }
        output[k] = sum;
    }
}

int main() {
    static complex<dtype_test> din[NTX][FFT_NUM], dout[NTX][FFT_NUM];
    static complex<dtype_gold> gin[NTX][FFT_NUM], gout[NTX][FFT_NUM];
    for (int b = 0; b < NTX; b++) {
        for (int i = 0; i < FFT_NUM; i++) {
            float t = static_cast<float>(i) / FFT_NUM;
            float re = std::sin(2.0 * M_PI * (10.0 + 7.0 * b) * t) + 0.5f * std::cos(2.0 * M_PI * 50.0 * t);
            float im = std::exp(-5.0 * t) * std::sin(2.0 * M_PI * (20.0 + 3.0 * b) * t);
            din[b][i] = complex<float>(re, im);
            gin[b][i] = complex<dtype_gold>(re, im);
        }
    }
    hls::stream<hls::vector<complex<float>, UF*2>> xn, xk;
    const int BEATS = FFT_NUM / (UF * 2);
    for (int b = 0; b < NTX; b++) {
        for (int idx = 0; idx < BEATS; idx++) {
            hls::vector<complex<float>, UF*2> t;
            for (int u = 0; u < UF*2; u++) t[u] = din[b][idx*UF*2 + u];
            xn.write(t);
        }
        FFT_TOP(xn, xk);
        for (int idx = 0; idx < BEATS; idx++) {
            hls::vector<complex<float>, UF*2> t = xk.read();
            for (int u = 0; u < UF*2; u++) dout[b][idx*UF*2 + u] = t[u];
        }
    }
    for (int b = 0; b < NTX; b++) fft_sw_gold(gin[b], gout[b]);
    double worst = 0.0; int wb = -1, wi = -1;
    for (int b = 0; b < NTX; b++) {
        for (int i = 0; i < FFT_NUM; i++) {
            double e = std::abs(complex<dtype_gold>(dout[b][i]) - gout[b][i]);
            if (e > worst) { worst = e; wb = b; wi = i; }
        }
    }
    cout << "CSIM_CHECK N=" << FFT_NUM << " UF=" << UF << " transforms=" << NTX
         << " max_abs_err=" << scientific << setprecision(4) << worst
         << " at[" << wb << "][" << wi << "] tol=" << TOL << endl;
    if (!(worst <= TOL)) { cout << "CSIM_CHECK FAIL" << endl; return 1; }
    cout << "CSIM_CHECK PASS" << endl;
    return 0;
}

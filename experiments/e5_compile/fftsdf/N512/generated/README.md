E5 changes only how many Vitis HLS projects are compiled and how many run at
once. **The generated C++ is byte identical in all three modes**, so there is
no per-mode hardware to keep here, and the hardware itself is the hardware of
the other experiments: this design at this size lives in
`experiments/e2_fft/spmw/N512/generated/`.

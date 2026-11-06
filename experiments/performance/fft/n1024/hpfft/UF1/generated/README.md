HP-FFT is hand-written HLS C++, so it has no generation step: what Vitis
compiles is exactly what is in `../source/`. This directory exists only to
keep the layout the same as SPMW's, where `generated/` holds what its
compiler emitted from a higher-level input.

import sys, os
sys.path.insert(0, "/scratch/hc676/allo/examples/machsuite/fft/strided")
import allo, strided_fft
A = "/scratch/hc676/e2_allo"
for N in [128, 256, 512, 1024]:
    for wrap in [True, False]:
        strided_fft.FFT_SIZE = N; strided_fft.FFT_SIZE_HALF = N // 2
        s = allo.customize(strided_fft.fft)
        prj = f"{A}/strided_n{N}_{'wrap' if wrap else 'raw'}"
        mod = s.build(target="vitis_hls", mode="csyn", project=prj, configs={"frequency": 300, "device": "u280", "num_output_args": 2}, wrap_io=wrap)
        print(prj, sorted(os.listdir(prj)))

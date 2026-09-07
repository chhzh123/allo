
### RTL cosimulation (xsim), cycles at 3.333 ns -- primary rows

| N | config | status | stimulus | RTL vs numpy | max abs err | first out (cyc) | 1st transform complete (cyc) | steady interval (cyc) | fed | completed | run by | failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | strided_fft FFT_SIZE=128 wrap_io=True | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 4.872e-06 |  | 11586 | 11559.0 | 4 | 4 | previous_agent |  |
| 128 | strided_fft FFT_SIZE=128 wrap_io=False | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 4.872e-06 |  | 27430 | 27403.0 | 4 | 4 | previous_agent |  |
| 256 | strided_fft FFT_SIZE=256 wrap_io=True | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 9.707e-06 |  | 26261 | 26234.0 | 4 | 4 | previous_agent |  |
| 256 | strided_fft FFT_SIZE=256 wrap_io=False | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 9.707e-06 |  | 63849 | 63822.0 | 4 | 4 | previous_agent |  |
| 512 | strided_fft FFT_SIZE=512 wrap_io=True | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 1.453e-05 |  | 58808 | 58781.0 | 4 | 4 | previous_agent |  |
| 512 | strided_fft FFT_SIZE=512 wrap_io=False | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 1.453e-05 |  | 145772 | 145745.0 | 4 | 4 | previous_agent |  |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=True | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 2.187e-05 |  | 130299 | 130272.0 | 4 | 4 | previous_agent |  |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=False | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 2.187e-05 |  | 327791 | 327764.0 | 4 | 4 | previous_agent |  |

### Vivado 2023.2 out-of-context synth + place + route at 3.333 ns (post-route numbers)

| N | config | status | LUT | FF | DSP | BRAM18-eq | RAMB36 | RAMB18 | URAM | WNS ns | TNS ns | unrouted | synth s | place s | route s | total s | run by | failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | strided_fft FFT_SIZE=128 wrap_io=True | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 102_pnrallo_128_wrap; optional extra) |
| 128 | strided_fft FFT_SIZE=128 wrap_io=False | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 115_pnrallo_128_raw; optional extra) |
| 256 | strided_fft FFT_SIZE=256 wrap_io=True | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 100_pnrallo_256_wrap; optional extra) |
| 256 | strided_fft FFT_SIZE=256 wrap_io=False | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 113_pnrallo_256_raw; optional extra) |
| 512 | strided_fft FFT_SIZE=512 wrap_io=True | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 103_pnrallo_512_wrap; optional extra) |
| 512 | strided_fft FFT_SIZE=512 wrap_io=False | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 116_pnrallo_512_raw; optional extra) |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=True | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 101_pnrallo_1024_wrap; optional extra) |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=False | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 114_pnrallo_1024_raw; optional extra) |

### Vitis HLS 2023.2 csynth estimates (hls_estimates.csv; not routed numbers)

| N | config | latency (cyc) | interval (cyc) | LUT | FF | DSP | BRAM18 | URAM | HLS est. ns | budget ns | hls s | csim | C model vs numpy | run by | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | strided_fft FFT_SIZE=128 wrap_io=True |  |  | 6608 | 7802 | 16 | 8 | 0 | 2.431 | 2.43 | 245 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 128 | strided_fft FFT_SIZE=128 wrap_io=False |  |  | 5883 | 6259 | 20 | 8 | 0 | 2.431 | 2.43 | 307 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 256 | strided_fft FFT_SIZE=256 wrap_io=True |  |  | 6562 | 7756 | 16 | 10 | 0 | 2.431 | 2.43 | 291 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 256 | strided_fft FFT_SIZE=256 wrap_io=False |  |  | 5891 | 6265 | 20 | 8 | 0 | 2.431 | 2.43 | 370 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 512 | strided_fft FFT_SIZE=512 wrap_io=True |  |  | 6448 | 7710 | 16 | 12 | 0 | 2.516 | 2.43 | 463 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 512 | strided_fft FFT_SIZE=512 wrap_io=False |  |  | 5895 | 6269 | 20 | 8 | 0 | 2.431 | 2.43 | 554 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=True |  |  | 6464 | 7727 | 16 | 14 | 0 | 2.515 | 2.43 | 723 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=False |  |  | 5899 | 6271 | 20 | 8 | 0 | 2.431 | 2.43 | 958 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |

### Directives Vitis HLS 2023.2 ignored, removed or could not honour (hls.log / csynth.rpt)

| N | config | warnings (code, meaning, count) |
|---|---|---|
| 128 | strided_fft FFT_SIZE=128 wrap_io=True | HLS 200-880 II-violation(memory-dependence) x7; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 128 | strided_fft FFT_SIZE=128 wrap_io=False | HLS 200-880 II-violation(memory-dependence) x16; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 256 | strided_fft FFT_SIZE=256 wrap_io=True | HLS 200-880 II-violation(memory-dependence) x7; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 256 | strided_fft FFT_SIZE=256 wrap_io=False | HLS 200-880 II-violation(memory-dependence) x16; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 512 | strided_fft FFT_SIZE=512 wrap_io=True | HLS 200-871 estimated-clock-exceeds-target x2; HLS 200-880 II-violation(memory-dependence) x7; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 512 | strided_fft FFT_SIZE=512 wrap_io=False | HLS 200-880 II-violation(memory-dependence) x16; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=True | HLS 200-871 estimated-clock-exceeds-target x2; HLS 200-880 II-violation(memory-dependence) x7; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=False | HLS 200-880 II-violation(memory-dependence) x16; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |

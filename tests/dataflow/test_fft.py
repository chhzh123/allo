# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# FFT implementations in Allo dataflow:
#
#  1. Scalar HP-FFT  (get_fft_top / test_fft)
#     - One PE per input point and one PE per butterfly
#     - Simple interface: mapping=[N] / mapping=[LOG2_N, N//2]
#     - Validates functional correctness for any N
#
#  2. Vectorized FFT-256 with F2 bank swizzle  (get_fft_256_vectorized / test_fft_256_*)
#     - Mirrors gemini-fft.prj/kernel.cpp architecture (N=256, WIDTH=32)
#     - Streams carry float32[WIDTH] tokens (WIDTH complex floats split into re/im)
#     - bit_rev_stage: reads input arrays, outputs bit-reversed chunks
#     - intra stages 0-4 (STRIDE=2^s < WIDTH): butterflies within each chunk
#     - inter stages 5-7 (STRIDE=2^s >= WIDTH): 2D partitioned buffer with
#       F2-computed XOR-swizzle for conflict-free bank access
#     - output_stage: reads final stream, writes result arrays
#
#     F2 layout synthesis (allo.transform.f2_layout.F2LayoutSolver):
#       - Computes swizzle matrix S over GF(2) for each inter-vector stage
#       - Swizzle: bank(idx) = (idx & (WIDTH-1)) ^ (((idx >> s) & 1) << (LOG2_WIDTH-1))
#       - Applied via s.partition() after df.customize() for ARRAY_PARTITION pragma

import os
import tempfile
from math import log2, cos, sin, pi

import allo
from allo.ir.types import float32, int32, uint32, Stream, ConstExpr
from allo.customize import Partition
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

from allo.transform.f2_layout import fft_swizzle

# ---------------------------------------------------------------------------
# Configuration for vectorized FFT
# ---------------------------------------------------------------------------
N = 256
WIDTH = 32
LOG2_N = 8
LOG2_WIDTH = 5
NUM_VECS = N // WIDTH  # 8
LOG2_NUM_VECS = LOG2_N - LOG2_WIDTH  # 3

# Full twiddle LUT: W_N^k = exp(-2*pi*i*k/N) for k=0..N/2-1
full_twr = np.array([cos(-2.0 * pi * k / N) for k in range(N // 2)], dtype=np.float32)
full_twi = np.array([sin(-2.0 * pi * k / N) for k in range(N // 2)], dtype=np.float32)
# Snap values within float32 epsilon to exactly 0.0 so HLS can constant-fold multiplications.
# cos(π/2) = 6.12e-17 in float64 is representable as float32 (non-zero), which forces
# HLS to generate DSP multipliers in stage-1 butterflies even though tw=(1,0)/(0,-1).
_snap_eps = float(np.finfo(np.float32).eps)
full_twr = np.where(np.abs(full_twr) < _snap_eps, np.float32(0.0), full_twr).astype(np.float32)
full_twi = np.where(np.abs(full_twi) < _snap_eps, np.float32(0.0), full_twi).astype(np.float32)


# ---------------------------------------------------------------------------
# Part 1: Scalar HP-FFT (functional reference for any power-of-2 N)
# ---------------------------------------------------------------------------


def bit_reverse(x, bits):
    """Reverse bits of x; called at Python compile time during kernel expansion."""
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def get_upper_idx(stage, butterfly):
    span = 1 << stage
    group_size = span << 1
    group = butterfly // span
    pos_in_group = butterfly % span
    return group * group_size + pos_in_group


def get_lower_idx(stage, butterfly):
    return get_upper_idx(stage, butterfly) + (1 << stage)


def get_tw_real(stage, butterfly, N_):
    span = 1 << stage
    group_size = span << 1
    pos_in_group = butterfly % span
    tw_idx = pos_in_group * (N_ // group_size)
    return cos(-2.0 * pi * tw_idx / N_)


def get_tw_imag(stage, butterfly, N_):
    span = 1 << stage
    group_size = span << 1
    pos_in_group = butterfly % span
    tw_idx = pos_in_group * (N_ // group_size)
    return sin(-2.0 * pi * tw_idx / N_)


def get_fft_top(N_):
    """Generate an N-point scalar HP-FFT dataflow region.

    Architecture:
      - input_loader:  N PEs, each loads one element with bit-reversal
      - butterfly:     LOG2_N * N//2 PEs, each computes one butterfly
      - output_store:  N PEs, each stores one output element
    """
    assert N_ > 0 and (N_ & (N_ - 1)) == 0, "N must be a power of 2"
    LOG2_N_ = int(log2(N_))
    HALF_N = N_ // 2

    @df.region()
    def top(
        inp_real: float32[N_],
        inp_imag: float32[N_],
        out_real: float32[N_],
        out_imag: float32[N_],
    ):
        stage_real: Stream[float32, 4][LOG2_N_ + 1, N_]
        stage_imag: Stream[float32, 4][LOG2_N_ + 1, N_]

        @df.kernel(mapping=[N_], args=[inp_real, inp_imag])
        def input_loader(local_inp_real: float32[N_], local_inp_imag: float32[N_]):
            idx = df.get_pid()
            val_real: float32 = local_inp_real[idx]
            val_imag: float32 = local_inp_imag[idx]
            stage_real[0, bit_reverse(idx, LOG2_N_)].put(val_real)
            stage_imag[0, bit_reverse(idx, LOG2_N_)].put(val_imag)

        @df.kernel(mapping=[LOG2_N_, HALF_N])
        def butterfly():
            s, b = df.get_pid()
            upper: ConstExpr[int32] = get_upper_idx(s, b)
            lower: ConstExpr[int32] = get_lower_idx(s, b)
            tw_r: ConstExpr[float32] = get_tw_real(s, b, N_)
            tw_i: ConstExpr[float32] = get_tw_imag(s, b, N_)
            a_real: float32 = stage_real[s, upper].get()
            a_imag: float32 = stage_imag[s, upper].get()
            b_real: float32 = stage_real[s, lower].get()
            b_imag: float32 = stage_imag[s, lower].get()
            bw_real: float32 = b_real * tw_r - b_imag * tw_i
            bw_imag: float32 = b_real * tw_i + b_imag * tw_r
            stage_real[s + 1, upper].put(a_real + bw_real)
            stage_imag[s + 1, upper].put(a_imag + bw_imag)
            stage_real[s + 1, lower].put(a_real - bw_real)
            stage_imag[s + 1, lower].put(a_imag - bw_imag)

        @df.kernel(mapping=[N_], args=[out_real, out_imag])
        def output_store(local_out_real: float32[N_], local_out_imag: float32[N_]):
            idx = df.get_pid()
            local_out_real[idx] = stage_real[LOG2_N_, idx].get()
            local_out_imag[idx] = stage_imag[LOG2_N_, idx].get()

    return top


def test_fft(N_=8):
    """Test scalar HP-FFT against NumPy reference."""
    LOG2_N_ = int(log2(N_))
    np.random.seed(42)
    inp_real = np.random.rand(N_).astype(np.float32)
    inp_imag = np.zeros(N_, dtype=np.float32)
    out_real = np.zeros(N_, dtype=np.float32)
    out_imag = np.zeros(N_, dtype=np.float32)
    print(f"Building {N_}-point scalar FFT ({LOG2_N_} stages)...")
    top = get_fft_top(N_)
    sim_mod = df.build(top, target="simulator")
    print("Running simulator...")
    sim_mod(inp_real, inp_imag, out_real, out_imag)
    ref = np.fft.fft(inp_real + 1j * inp_imag)
    np.testing.assert_allclose(out_real, ref.real.astype(np.float32), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_imag, ref.imag.astype(np.float32), rtol=1e-4, atol=1e-4)
    print(f"✅ HP-FFT {N_}-point Simulator Test PASSED!")


# ---------------------------------------------------------------------------
# Part 2: Vectorized FFT-256 with F2 bank swizzle
#
# F2 layout synthesis computes, per inter-vector stage s:
#   swizzle matrix S s.t. bank(idx) = S @ addr_bits(idx)
#   = (idx & (WIDTH-1)) ^ (((idx >> s) & 1) << (LOG2_WIDTH-1))
#
# This resolves butterfly bank conflicts (idx and idx+STRIDE share same lower
# bits when STRIDE=2^s is a multiple of WIDTH=32).
#
# Schedule: after df.customize(), call s.partition(kernel:buf, dim=1) to
# generate  #pragma HLS array_partition complete dim=1  for each 2D buffer.
# ---------------------------------------------------------------------------

@df.region()
def fft_256(
    inp_re: Stream[float32[WIDTH], 2],
    inp_im: Stream[float32[WIDTH], 2],
    out_re: Stream[float32[WIDTH], 2],
    out_im: Stream[float32[WIDTH], 2],
):
    """Vectorized 256-point FFT with F2-swizzled inter-stage buffers."""

    # 8 intermediate streams (s[0]..s[7]); each token is WIDTH float32 values
    # depth=2 matches the reference design (FIFO_SRL, 0.87ns) for min pipeline latency
    s_re: Stream[float32[WIDTH], 2][LOG2_N + 1]
    s_im: Stream[float32[WIDTH], 2][LOG2_N + 1]

    # ------------------------------------------------------------------
    # Bit-reversal stage: inline 8-bit reversal (N=256, LOG2_N=8)
    # ------------------------------------------------------------------
    @df.kernel(mapping=[1])
    def bit_rev_stage():
        """Bit-reversal stage: rearranges N=256 inputs for DIT FFT with II=1.

        Uses a 2D swizzled buffer [WIDTH, NUM_VECS] = [32, 8] with complete dim=1:
          LOAD (8 iters, II=1): read WIDTH=32 elements from stream, write to 32 distinct banks.
            bank  = bit_rev5(kk):  5-bit reversal of lane → unique bank per lane (no conflicts)
            offset = bit_rev3(ii): 3-bit reversal of vector index

          WRITE (8 iters, II=1): read back in sequential (jj, mm) order.
            buf_re[(jj*32+mm)/8][(jj*32+mm)%8] = input[bit_rev(jj*32+mm)]  (verified)
        """
        # 2D buffer: dim-0 = WIDTH banks (complete partition), dim-1 = NUM_VECS depth
        buf_re: float32[WIDTH, NUM_VECS]
        buf_im: float32[WIDTH, NUM_VECS]

        # LOAD: read WIDTH-element chunks from input streams, then swizzle into 2D buffer.
        # bit_rev5(kk) is unique per kk → 32 parallel writes to 32 distinct banks
        for ii in range(NUM_VECS):
            chunk_in_re: float32[WIDTH] = inp_re.get()
            chunk_in_im: float32[WIDTH] = inp_im.get()
            for kk in range(WIDTH):
                bank: int32 = (
                    ((kk & 1) << 4)
                    | ((kk & 2) << 2)
                    | (kk & 4)
                    | ((kk & 8) >> 2)
                    | ((kk & 16) >> 4)
                )
                offset: int32 = ((ii & 4) >> 2) | (ii & 2) | ((ii & 1) << 2)
                buf_re[bank, offset] = chunk_in_re[kk]
                buf_im[bank, offset] = chunk_in_im[kk]

        # WRITE: sequential 2D read order produces the bit-reversed permutation
        for jj in range(NUM_VECS):
            chunk_re: float32[WIDTH]
            chunk_im: float32[WIDTH]
            for mm in range(WIDTH):
                # rd_bank = (jj*32+mm) >> 3 = (jj<<2) | (mm>>3)
                # rd_off  = (jj*32+mm) &  7 = mm & 7  (jj*32 divisible by 8)
                rd_bank: int32 = (jj << 2) | (mm >> LOG2_NUM_VECS)
                rd_off: int32 = mm & 7
                chunk_re[mm] = buf_re[rd_bank, rd_off]
                chunk_im[mm] = buf_im[rd_bank, rd_off]
            s_re[0].put(chunk_re)
            s_im[0].put(chunk_im)

    # ------------------------------------------------------------------
    # Intra-vector stages 0-4  (STRIDE=2^s < WIDTH=32)
    # All butterflies within one WIDTH-element chunk; no bank conflicts.
    # ------------------------------------------------------------------
    @df.kernel(mapping=[5])
    def intra():
        s = df.get_pid()
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        for _i in range(NUM_VECS):
            c_re: float32[WIDTH] = s_re[s].get()
            c_im: float32[WIDTH] = s_im[s].get()
            o_re: float32[WIDTH]
            o_im: float32[WIDTH]
            stride: int32 = 1 << s
            for k in range(16):
                il: int32 = ((k >> s) << (s + 1)) | (k & (stride - 1))
                iu: int32 = il | stride
                tw_k: int32 = (k & (stride - 1)) << (7 - s)
                
                a_re = c_re[il]
                a_im = c_im[il]
                b_re = c_re[iu]
                b_im = c_im[iu]

                if tw_k == 0:
                    o_re[il] = a_re + b_re
                    o_im[il] = a_im + b_im
                    o_re[iu] = a_re - b_re
                    o_im[iu] = a_im - b_im
                elif tw_k == 64:
                    o_re[il] = a_re + b_im
                    o_im[il] = a_im - b_re
                    o_re[iu] = a_re - b_im
                    o_im[iu] = a_im + b_re
                else:
                    tr = twr[tw_k]
                    ti = twi[tw_k]
                    bw_re: float32 = b_re * tr - b_im * ti
                    bw_im: float32 = b_re * ti + b_im * tr
                    o_re[il] = a_re + bw_re
                    o_im[il] = a_im + bw_im
                    o_re[iu] = a_re - bw_re
                    o_im[iu] = a_im - bw_im
            s_re[s + 1].put(o_re)
            s_im[s + 1].put(o_im)

    # ------------------------------------------------------------------
    # Inter-vector stages 5-7  (STRIDE=2^s >= WIDTH=32)
    #
    # F2 swizzle: bank(idx) = (idx & 31) ^ (((idx >> s) & 1) << 4)
    #             offset(idx) = idx >> 5
    #
    # Buffer shape: float32[WIDTH, NUM_VECS] = float32[32, 8]
    # s.partition(kernel:buf, dim=1) → #pragma HLS array_partition complete dim=1
    # ------------------------------------------------------------------

    @df.kernel(mapping=[3])
    def inter():
        s_rel = df.get_pid()
        # s = s_rel + 5
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        in_re: float32[WIDTH, NUM_VECS]
        in_im: float32[WIDTH, NUM_VECS]
        out_re_b: float32[WIDTH, NUM_VECS]
        out_im_b: float32[WIDTH, NUM_VECS]

        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = s_re[s_rel + 5].get()
            chunk_im: float32[WIDTH] = s_im[s_rel + 5].get()
            for k in range(WIDTH):
                bank: int32 = (k & 15) | (((k >> 4) ^ ((i >> s_rel) & 1)) << 4)
                in_re[bank, i] = chunk_re[k]
                in_im[bank, i] = chunk_im[k]

        for i in range(NUM_VECS):
            for k in range(16):
                bg: uint32 = (i << 4) | k
                bank_il: uint32 = k | ((i & 1) << 4)
                bank_iu: uint32 = bank_il ^ 16

                i_shr: uint32 = i >> 1
                low_mask: uint32 = (1 << s_rel) - 1
                low_bits: uint32 = i_shr & low_mask
                high_bits: uint32 = i_shr >> s_rel
                off_il: uint32 = (high_bits << (s_rel + 1)) | low_bits
                stride_off: uint32 = 1 << s_rel
                off_iu: uint32 = off_il | stride_off

                a_re = in_re[bank_il, off_il]
                a_im = in_im[bank_il, off_il]
                b_re = in_re[bank_iu, off_iu]
                b_im = in_im[bank_iu, off_iu]

                tw_k: uint32 = (bg & (((1 << s_rel) << 5) - 1)) << (2 - s_rel)
                tr = twr[tw_k]
                ti = twi[tw_k]
                bw_re: float32 = b_re * tr - b_im * ti
                bw_im: float32 = b_re * ti + b_im * tr
                out_re_b[bank_il, off_il] = a_re + bw_re
                out_im_b[bank_il, off_il] = a_im + bw_im
                out_re_b[bank_iu, off_iu] = a_re - bw_re
                out_im_b[bank_iu, off_iu] = a_im - bw_im

        for i in range(NUM_VECS):
            chunk_re_out: float32[WIDTH]
            chunk_im_out: float32[WIDTH]
            for k in range(WIDTH):
                bank: int32 = (k & 15) | (((k >> 4) ^ ((i >> s_rel) & 1)) << 4)
                chunk_re_out[k] = out_re_b[bank, i]
                chunk_im_out[k] = out_im_b[bank, i]
            s_re[s_rel + 6].put(chunk_re_out)
            s_im[s_rel + 6].put(chunk_im_out)

    @df.kernel(mapping=[1])
    def output_stage():
        for i in range(NUM_VECS):
            out_re.put(s_re[LOG2_N].get())
            out_im.put(s_im[LOG2_N].get())



def _apply_f2_partitions(s):
    """Apply F2-computed ARRAY_PARTITION pragmas to all inter-stage 2D buffers.

    Called after df.customize() to annotate the 2D swizzled buffers in
    inter_5, inter_6, inter_7 with  #pragma HLS array_partition complete dim=1
    so Vitis HLS instantiates WIDTH separate LUTRAM banks enabling II=1.

    # The kernel names are suffixed by the mapping ID: "_0", "_1", "_2"
    """
    # The kernel names are suffixed by the mapping ID: "_0", "_1", "_2"
    inter_kernels = ["inter_0", "inter_1", "inter_2"]
    bufs = ["in_re", "in_im", "out_re_b", "out_im_b"]
    for kn in inter_kernels:
        for bn in bufs:
            s.partition(f"{kn}:{bn}", partition_type=Partition.Complete, dim=1)


def _apply_f2_optimizations(s):
    """Full F2 optimization pass: partition + dataflow + pipeline + unroll.

    Applies all HLS pragmas needed to match the performance of
    gemini-fft.prj/kernel.cpp:
      - ARRAY_PARTITION complete dim=1 on inter-stage 2D buffers
      - DATAFLOW on inter-stage kernels (sub-function pipeline)
      - PIPELINE II=1 on all outer i/_i/src loops
      - UNROLL on all inner k loops
    """
    # 1. Partition inter-stage 2D buffers
    _apply_f2_partitions(s)

    # 1b. Partition twiddle ROMs (global constants) for parallel access
    #     Without this, unrolled k-loops stall waiting for twiddle reads.
    s.partition_global("twr")
    s.partition_global("twi")

    # 2. Sub-function dataflow for inter-vector stages
    #    This lets HLS pipeline load/compute/write sub-loops concurrently,
    #    achieving II=N/WIDTH instead of 3*(N/WIDTH) cycles per inter stage.
    for kn in ["inter_0", "inter_1", "inter_2"]:
        s.dataflow(kn)

    # 3. BIND_STORAGE ram_2p lutram + DEPENDENCE inter false on all inter-stage buffers
    #    ram_2p enables dual-port access needed for HLS DATAFLOW ping-pong buffering.
    #    DEPENDENCE inter false removes conservative false dependencies for II=1.
    inter_kernels = ["inter_0", "inter_1", "inter_2"]
    bufs = ["in_re", "in_im", "out_re_b", "out_im_b"]
    for kn in inter_kernels:
        for bn in bufs:
            s.bind_storage(f"{kn}:{bn}", impl="lutram", storage_type="ram_2p")
            s.dependence(f"{kn}:{bn}")

    # 4a. Partition and annotate bit_rev_stage 2D buffers (complete dim=1 for
    #     32 parallel banks, enabling II=1 for the vectorized LOAD/WRITE phases)
    for bn in ["buf_re", "buf_im"]:
        s.partition("bit_rev_stage_0:" + bn, partition_type=Partition.Complete, dim=1)
        s.bind_storage("bit_rev_stage_0:" + bn, impl="lutram", storage_type="ram_2p")
        s.dependence("bit_rev_stage_0:" + bn)

    # 4b. Enable DATAFLOW within bit_rev_stage so LOAD and WRITE overlap
    #     (achieves II=8 for the stage instead of II=16 sequential)
    s.dataflow("bit_rev_stage_0")

    # 4d. Pipeline outer loops + unroll inner loops for all kernels
    #     bit_rev_stage: vectorized LOAD (ii/kk) and WRITE (jj/mm) phases
    br_loops = s.get_loops("bit_rev_stage_0")
    s.pipeline(br_loops["S_ii_0"]["ii"])
    s.unroll(br_loops["S_ii_0"]["kk"])
    s.pipeline(br_loops["S_jj_2"]["jj"])
    s.unroll(br_loops["S_jj_2"]["mm"])

    # intra stages: single outer _i loop
    for stage in range(5):
        kn = f"intra_{stage}"
        lp = s.get_loops(kn)
        s.pipeline(lp["S__i_0"]["_i"])
        s.unroll(lp["S__i_0"]["k"])

    # inter stages: three loop bands (load=S_i_0, compute=S_i_2, write=S_i_4)
    for stage in range(3):
        kn = f"inter_{stage}"
        lp = s.get_loops(kn)
        for band in ["S_i_0", "S_i_2", "S_i_4"]:
            s.pipeline(lp[band]["i"])
            s.unroll(lp[band]["k"])

    # output stages: single outer loop
    lp_out = s.get_loops("output_stage_0")
    s.pipeline(lp_out["S_i_0"]["i"])



# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fft_8():
    """Test scalar HP-FFT with N=8."""
    test_fft(8)


def test_fft_256_vectorized():
    """Test vectorized FFT-256 with F2 swizzle (HLS codegen check).

    The top-level interface uses hls::stream<hls::vector<float, 32>> args,
    matching gemini-fft's interface exactly. The LLVM simulator does not support
    stream-typed top-level args, so this test verifies HLS codegen correctness.
    For functional simulation, use Vitis HLS csim.
    """
    s = df.customize(fft_256)
    _apply_f2_optimizations(s)
    mod = s.build(target="vitis_hls", configs=_BUILD_CONFIGS)
    code = mod.hls_code
    assert "hls::stream" in code and "hls::vector" in code, (
        "Expected stream-typed top-level I/O args in fft_256 signature"
    )
    # Top function must accept stream-typed args (passed by reference with '&')
    assert "hls::stream< hls::vector< float, 32 > >& " in code, (
        "Expected hls::stream<hls::vector<float,32>>& args in fft_256 signature"
    )
    assert "bit_rev_stage" in code
    assert "inter_2" in code
    print("✅ FFT-256 Vectorized HLS Codegen Test PASSED!")


# Full build configs: bind_op_fabric reduces fadd/fsub latency from ~5 to ~1 cycle.
# Top function now accepts hls::stream<hls::vector<float, 32>> directly — no
# load_buf/store_res wrappers needed.
_BUILD_CONFIGS = {
    "bind_op_fabric": True,  # reduces fadd/fsub latency from ~5 to ~1 cycle
}


def test_fft_256_hls_codegen():
    """Verify that the HLS code contains F2-partitioned 2D buffers and swizzle."""
    s = df.customize(fft_256)
    _apply_f2_optimizations(s)
    mod = s.build(target="vitis_hls", configs=_BUILD_CONFIGS)
    code = mod.hls_code

    # Structural checks
    assert "hls::vector" in code, "Expected vectorized stream in HLS code"
    assert "hls::stream" in code
    assert "bit_rev_stage" in code
    assert "inter_2" in code

    # F2 swizzle is visible as XOR operations on bank indices
    assert "^" in code, "Expected XOR operations for F2 swizzle"

    # Array partition pragma for 2D buffers
    assert "#pragma HLS array_partition" in code, (
        "Expected array_partition pragma for swizzled 2D buffers"
    )

    # Pipeline pragmas on all kernels
    assert "#pragma HLS pipeline" in code, "Expected pipeline pragma for II=1"

    # Sub-function dataflow on inter-vector stages
    assert "#pragma HLS dataflow" in code, (
        "Expected dataflow pragma for inter-stage sub-function pipeline"
    )

    # dependence pragma on inter-stage buffers
    # All bank arrays use complete dim=1 + bind_storage lutram.
    # 'inter false' suppresses cross-iteration false WAW deps (function scope).
    # 'intra false' suppresses within-iteration false WAW among 16 unrolled butterfly
    # writes (injected inside compute loop body by add_compute_loop_dependence_pragmas).
    assert "dependence" in code, "Expected dependence pragma for II=1"
    assert "bind_storage" in code, "Expected bind_storage pragma for LUTRAM (no BRAM)"
    assert "inter false" in code, "Expected inter false pragma for II=1 with LUTRAM"
    assert "intra false" in code, "Expected intra false pragma for II=1 unrolled butterfly"

    # bind_op fabric pragma for float add/sub latency reduction
    assert "#pragma HLS bind_op" in code, (
        "Expected bind_op pragma for float fadd/fsub impl=fabric"
    )

    print("✅ FFT-256 HLS Codegen Test PASSED!")



def test_fft_256_csyn():
    """Run C-synthesis via Vitis HLS (requires Vitis HLS installation)."""
    if not hls.is_available("vitis_hls"):
        import pytest
        pytest.skip("Vitis HLS not available")
    s = df.customize(fft_256)
    _apply_f2_optimizations(s)
    with tempfile.TemporaryDirectory() as tmpdir:
        s.build(
            target="vitis_hls",
            mode="csyn",
            project=tmpdir,
            configs=_BUILD_CONFIGS,
        )
    print("✅ FFT-256 CSyn Passed!")


def test_fft_256_simulator(N_=256):
    import numpy as np
    sim_mod = df.build(fft_256, target="simulator")
    
    np.random.seed(42)
    inp_real = np.random.rand(NUM_VECS, WIDTH).astype(np.float32)
    inp_imag = np.random.rand(NUM_VECS, WIDTH).astype(np.float32)
    out_real = np.zeros((NUM_VECS, WIDTH), dtype=np.float32)
    out_imag = np.zeros((NUM_VECS, WIDTH), dtype=np.float32)

    print("Running simulator...")
    sim_mod(inp_real, inp_imag, out_real, out_imag)
    
    # Verify results
    inp_cmplx = (inp_real + 1j * inp_imag).flatten()
    out_cmplx = (out_real + 1j * out_imag).flatten()
    ref = np.fft.fft(inp_cmplx)
    
    np.testing.assert_allclose(out_cmplx.real, ref.real, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_cmplx.imag, ref.imag, rtol=1e-4, atol=1e-4)
    print("✅ Simulator Passed!")


if __name__ == "__main__":
    import sys

    N_ = 8
    if len(sys.argv) > 1:
        N_ = int(sys.argv[1])
    num_threads = max(64, N_ * 2)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    if len(sys.argv) > 1 and sys.argv[1] == "csyn":
        test_fft_256_csyn()
    else:
        test_fft_256_simulator()

    del os.environ["OMP_NUM_THREADS"]

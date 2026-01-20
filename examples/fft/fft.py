# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# HP-FFT: High-Performance FFT Implementation using Allo Dataflow
# Based on the HP-FFT paper architecture using Cooley-Tukey radix-2 DIT algorithm

# This implementation uses a multi-stage pipelined dataflow architecture where:
# - Each stage contains N/2 butterfly units computing in parallel
# - Stages are connected via stream interfaces (FIFOs)
# - Data is reordered using bit-reversal permutation

# This version is parameterized for power-of-2 FFT sizes

import os
from math import log2, cos, sin, pi
from pathlib import Path
import tempfile
import importlib.util

import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


def bit_reverse(x: int, bits: int) -> int:
    """Reverse the bits of x for bit-reversal permutation."""
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def compute_twiddle(k: int, N: int) -> tuple:
    """Compute twiddle factor W_N^k = exp(-2*pi*j*k/N)."""
    angle = -2.0 * pi * k / N
    return (cos(angle), sin(angle))


def generate_fft_module(N: int, output_path: str = None):
    """
    Generate an N-point FFT module as a Python source file.

    N must be a power of 2.

    Returns the path to the generated file.
    """
    assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"

    LOG2_N = int(log2(N))
    HALF_N = N // 2

    # Pre-compute bit-reversal indices
    bit_rev_indices = [bit_reverse(i, LOG2_N) for i in range(N)]

    # Pre-compute butterfly indices for each stage
    butterfly_indices = []
    for s in range(LOG2_N):
        span = 1 << s
        group_size = span << 1
        stage_butterflies = []
        for b in range(HALF_N):
            group = b // span
            pos_in_group = b % span
            upper_idx = group * group_size + pos_in_group
            lower_idx = upper_idx + span
            tw_idx = pos_in_group * (N // group_size)
            tw = compute_twiddle(tw_idx, N)
            stage_butterflies.append((upper_idx, lower_idx, tw[0], tw[1]))
        butterfly_indices.append(stage_butterflies)

    # Generate the module code
    code_lines = [
        "# Auto-generated HP-FFT module",
        f"# N = {N}, LOG2_N = {LOG2_N}",
        "",
        "import allo",
        "from allo.ir.types import float32, Stream",
        "import allo.dataflow as df",
        "",
        "",
        "@df.region()",
        "def top(",
        f"    inp_real: float32[{N}],",
        f"    inp_imag: float32[{N}],",
        f"    out_real: float32[{N}],",
        f"    out_imag: float32[{N}],",
        "):",
    ]

    # Add stream declarations
    for i in range(LOG2_N + 1):
        code_lines.append(f"    s{i}_real: Stream[float32, 4][{N}]")
        code_lines.append(f"    s{i}_imag: Stream[float32, 4][{N}]")
    code_lines.append("")

    # Add input_loader kernel
    code_lines.extend(
        [
            "    @df.kernel(mapping=[1], args=[inp_real, inp_imag])",
            f"    def input_loader(local_inp_real: float32[{N}], local_inp_imag: float32[{N}]):",
            "        # Load input with bit-reversal permutation",
        ]
    )
    for idx in range(N):
        rev_idx = bit_rev_indices[idx]
        code_lines.append(f"        s0_real[{idx}].put(local_inp_real[{rev_idx}])")
        code_lines.append(f"        s0_imag[{idx}].put(local_inp_imag[{rev_idx}])")
    code_lines.append("")

    # Add stage kernels
    for s in range(LOG2_N):
        code_lines.extend(
            [
                f"    @df.kernel(mapping=[{HALF_N}])",
                f"    def stage{s}():",
                "        b = df.get_pid()",
            ]
        )
        for b, (upper_idx, lower_idx, tw_real, tw_imag) in enumerate(
            butterfly_indices[s]
        ):
            code_lines.extend(
                [
                    f"        with allo.meta_if(b == {b}):",
                    f"            a_real: float32 = s{s}_real[{upper_idx}].get()",
                    f"            a_imag: float32 = s{s}_imag[{upper_idx}].get()",
                    f"            b_real: float32 = s{s}_real[{lower_idx}].get()",
                    f"            b_imag: float32 = s{s}_imag[{lower_idx}].get()",
                    f"            bw_real: float32 = b_real * {tw_real:.16f} - b_imag * {tw_imag:.16f}",
                    f"            bw_imag: float32 = b_real * {tw_imag:.16f} + b_imag * {tw_real:.16f}",
                    f"            s{s+1}_real[{upper_idx}].put(a_real + bw_real)",
                    f"            s{s+1}_imag[{upper_idx}].put(a_imag + bw_imag)",
                    f"            s{s+1}_real[{lower_idx}].put(a_real - bw_real)",
                    f"            s{s+1}_imag[{lower_idx}].put(a_imag - bw_imag)",
                ]
            )
        code_lines.append("")

    # Add output_store kernel
    code_lines.extend(
        [
            "    @df.kernel(mapping=[1], args=[out_real, out_imag])",
            f"    def output_store(local_out_real: float32[{N}], local_out_imag: float32[{N}]):",
            "        # Store output from the final stage",
        ]
    )
    for idx in range(N):
        code_lines.append(
            f"        local_out_real[{idx}] = s{LOG2_N}_real[{idx}].get()"
        )
        code_lines.append(
            f"        local_out_imag[{idx}] = s{LOG2_N}_imag[{idx}].get()"
        )

    code = "\n".join(code_lines)
    print(code)

    # Write to file
    if output_path is None:
        # Use a temp directory
        temp_dir = Path(tempfile.gettempdir()) / "allo_fft"
        temp_dir.mkdir(exist_ok=True)
        output_path = temp_dir / f"fft_{N}.py"
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write(code)

    return str(output_path)


def load_fft_module(module_path: str):
    """Load a generated FFT module and return the 'top' function."""
    spec = importlib.util.spec_from_file_location("fft_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.top


def get_fft_top(N: int):
    """
    Get an N-point FFT dataflow region.

    This generates the FFT module code to a file and loads it.
    """
    module_path = generate_fft_module(N)
    return load_fft_module(module_path)


def test_fft(N: int = 8):
    """Test the FFT implementation against NumPy reference."""
    LOG2_N = int(log2(N))

    # Generate test input
    np.random.seed(42)
    inp_real = np.random.rand(N).astype(np.float32)
    inp_imag = np.zeros(N, dtype=np.float32)  # Real input

    # Output arrays
    out_real = np.zeros(N, dtype=np.float32)
    out_imag = np.zeros(N, dtype=np.float32)

    print(f"Generating {N}-point FFT module ({LOG2_N} stages)...")

    # Build and run the FFT
    top = get_fft_top(N)

    print("Building simulator...")
    sim_mod = df.build(top, target="simulator")

    print("Running simulator...")
    sim_mod(inp_real, inp_imag, out_real, out_imag)

    # NumPy reference
    ref = np.fft.fft(inp_real + 1j * inp_imag)
    ref_real = ref.real.astype(np.float32)
    ref_imag = ref.imag.astype(np.float32)

    # Verify results
    print("=" * 60)
    print("HP-FFT Test Results")
    print("=" * 60)
    print(f"FFT Size: {N} points, {LOG2_N} stages")
    print("-" * 60)

    if N <= 16:
        print("\nInput (real):", inp_real)
        print("\nOutput (real):", out_real)
        print("Reference (real):", ref_real)
        print("\nOutput (imag):", out_imag)
        print("Reference (imag):", ref_imag)
    else:
        print(f"\nInput (first 8 of {N}):", inp_real[:8])
        print(f"\nOutput (first 8 of {N}):", out_real[:8])
        print(f"Reference (first 8 of {N}):", ref_real[:8])

    max_diff_real = np.max(np.abs(out_real - ref_real))
    max_diff_imag = np.max(np.abs(out_imag - ref_imag))

    print("-" * 60)
    print(f"Max difference (real): {max_diff_real:.6e}")
    print(f"Max difference (imag): {max_diff_imag:.6e}")

    try:
        np.testing.assert_allclose(out_real, ref_real, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_imag, ref_imag, rtol=1e-4, atol=1e-4)
        print(f"\n✅ HP-FFT {N}-point Simulator Test PASSED!")
    except AssertionError as e:
        print(f"\n❌ HP-FFT {N}-point Simulator Test FAILED: {e}")

    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Default to 8-point FFT, or use command-line argument
    N = 8
    if len(sys.argv) > 1:
        N = int(sys.argv[1])

    # Set number of threads based on FFT size
    num_threads = N
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    test_fft(N)

    del os.environ["OMP_NUM_THREADS"]

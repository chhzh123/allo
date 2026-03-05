# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=consider-using-with, no-name-in-module, too-many-branches

import os
import re
import io
import subprocess
import time
import numpy as np
from .._mlir.dialects import allo as allo_d
from .._mlir.ir import (
    Context,
    Location,
    Module,
    UnitAttr,
)
from .._mlir.passmanager import PassManager

from .config import DEFAULT_CONFIG, PART_NUMBER
from .vitis import (
    codegen_host,
    postprocess_hls_code,
    generate_description_file,
    write_tensor_to_file,
    read_tensor_from_file,
    generate_hbm_config,
    extract_hls_arg_names,
)
from .pynq import (
    postprocess_hls_code_pynq,
    codegen_pynq_host,
)
from .tapa import (
    codegen_tapa_host,
)
from .catapult import (
    codegen_tcl as codegen_tcl_catapult,
    codegen_host as codegen_host_catapult,
)
from .ip import IPModule
from .report import parse_xml
from ..passes import (
    _mlir_lower_pipeline,
    decompose_library_function,
    generate_input_output_buffers,
)
from ..harness.makefile_gen.makegen import generate_makefile
from ..ir.transform import find_func_in_module
from ..utils import (
    get_func_inputs_outputs,
    c2allo_type,
    get_bitwidth_from_type,
    np_supported_types,
)


def is_available(backend="vivado_hls"):
    if backend == "vivado_hls":
        return os.system("which vivado_hls >> /dev/null") == 0
    if backend == "tapa":
        return os.system("which tapa >> /dev/null") == 0
    return os.system("which vitis_hls >> /dev/null") == 0


def run_process(cmd, pattern=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err:
        raise RuntimeError("Error raised: ", err.decode())
    if pattern:
        return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")


def codegen_tcl(top, configs):
    out_str = """# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=============================================================================
# run.tcl 
#=============================================================================
# Project name
set hls_prj out.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

"""
    out_str += f'# Top function of the design is "{top}"\n'
    out_str += f"set_top {top}\n"
    out_str += """
# Add design and testbench files
add_files kernel.cpp
add_files -tb host.cpp -cflags "-std=gnu++0x"
open_solution "solution1"
"""
    device = configs["device"]
    frequency = configs["frequency"]
    mode = configs["mode"]
    if device not in PART_NUMBER:
        raise RuntimeError(
            f"Device {device} not supported. Available devices: {list(PART_NUMBER.keys())}"
        )
    out_str += f"\n# Target device is {device}\n"
    out_str += f"set_part {{{PART_NUMBER[device]}}}\n\n"
    out_str += "# Target frequency\n"
    out_str += f"create_clock -period {1000 / frequency:.2f}\n\n"
    out_str += "# Run HLS\n"
    if "csim" in mode or "sw_emu" in mode:
        out_str += "csim_design -O\n"
    if "csyn" in mode or "debug" in mode:
        out_str += "csynth_design\n"
    if "cosim" in mode or "hw_emu" in mode:
        out_str += "cosim_design\n"
    if "impl" in mode or "hw" in mode:
        if device in {"ultra96v2", "pynqz2", "zedboard"}:
            # Embedded boards: export IP only, bitstream happens in Python/Vivado later
            out_str += "export_design -rtl verilog -format ip_catalog\n"
        else:
            # Other platforms: run full impl in HLS
            out_str += "export_design -flow impl\n"
    out_str += "\nexit\n"
    return out_str


def copy_ext_libs(ext_libs, project):
    for ext_lib in ext_libs:
        impl_path = ext_lib.impl
        cpp_file = impl_path.split("/")[-1]
        assert cpp_file != "kernel.cpp", "kernel.cpp is reserved for the top function"
        os.system(f"cp {impl_path} {project}/{cpp_file}")


def optimize_stream_reads(hls_code):
    """Replace intermediate float[N] arrays from stream reads with direct hls::vector assignments.

    The HLS code generator emits stream reads as:
        float vN[M];
        {
          hls::vector< float, M > _vec = STREAM.read();
          for (int _iv0 = 0; _iv0 < M; ++_iv0) {
            #pragma HLS unroll
            vN[_iv0] = _vec[_iv0];
          }
        }	// LNNN

    When vN is inside a pipelined loop body, the float[M] array has limited memory ports.
    Multiple pipeline stages writing/reading vN cause II=2 (HLS violation HLS 200-885).
    Fix: replace with ``hls::vector< float, M > vN = STREAM.read();``
    hls::vector supports operator[], so downstream ``vN[k]`` accesses still compile correctly.

    Enabled via ``configs={'optimize_stream_reads': True}`` in ``s.build()``.
    Also enabled automatically when ``bind_op_fabric`` is True.
    """
    pattern = re.compile(
        r"(\s+)float (v\d+)\[(\d+)\];\n"
        r"\1\{\n"
        r"[ \t]+hls::vector< float, \3 > _vec = (\S+)\.read\(\);\n"
        r"[ \t]+for \(int _iv0 = 0; _iv0 < \3; \+\+_iv0\) \{\n"
        r"[ \t]+#pragma HLS unroll\n"
        r"[ \t]+\2\[_iv0\] = _vec\[_iv0\];\n"
        r"[ \t]+\}\n"
        r"[ \t]+\}\t// L\d+",
        re.MULTILINE,
    )

    def _replace(m):
        indent = m.group(1)
        var_name = m.group(2)
        size = m.group(3)
        stream = m.group(4)
        return f"{indent}hls::vector< float, {size} > {var_name} = {stream}.read();"

    return pattern.sub(_replace, hls_code)


def fix_bank_array_partition(hls_code):
    """Fix array_partition pragmas and bind_storage for inter-stage bank arrays.

    Dataflow inter-stage uses two kinds of [BANKS][DEPTH] 2D bank arrays:

    1. **Input banks** (``in_re*``, ``in_im*``): Written by the LOAD sub-loop with 32x
       unrolled writes using runtime-computed bank indices. In HLS 2023.2,
       ``complete`` without ``dim`` only partitions the outermost dimension (same as
       ``dim=1``), leaving each depth-8 sub-array as a BRAM_AUTO 1R1W RAM (2-cycle
       read latency), which causes COMPUTE iter_latency=11 vs reference 10.
       Fix: add ``complete dim=2`` as a second pragma to also partition the inner
       dimension, converting all [BANKS][DEPTH] entries to 256 individual FF registers
       (0-cycle combinational access) with unlimited read/write ports.

    2. **Output banks** (``out_re_b*``, ``out_im_b*``): Written by the COMPUTE sub-loop.
       Keep ``dim=1`` partition (32 separate 8-element sub-arrays) so ``dependence inter
       false`` applies correctly to each sub-array.  **Keep** ``bind_storage lutram``
       so each 8-element sub-array is implemented as LUTRAM (0 BRAM) rather than BRAM
       (the HLS default for small RAM without explicit storage binding).

    Input banks: remove ``bind_storage`` (no longer needed after full FF partition).
    Output banks: keep ``bind_storage lutram`` to prevent BRAM allocation.

    Only modifies ``complete dim=1`` pragmas (not ``dim=2`` used for I/O buffers).
    Enabled automatically when ``optimize_stream_reads`` or ``bind_op_fabric`` is True.
    """
    # Step 1: collect all variables with 'complete dim=1' partition
    all_dim1_vars = set()
    for m in re.finditer(
        r"#pragma HLS array_partition variable=(\S+) complete dim=1\b", hls_code
    ):
        all_dim1_vars.add(m.group(1))

    # Input bank arrays: upgrade to complete (all dims) for II=1 in LOAD loops
    input_bank_pat = re.compile(r"^in_re\d*$|^in_im\d*$|^buf_re\d*$|^buf_im\d*$")
    input_banks = {v for v in all_dim1_vars if input_bank_pat.match(v)}
    # Output bank arrays: keep dim=1 + bind_storage lutram (LUTRAM, not BRAM)
    output_banks = all_dim1_vars - input_banks

    # Step 2a: input banks need BOTH dim=1 and dim=2 complete partition to fully convert
    # [BANKS][DEPTH] 2D arrays to individual FF registers.
    # In HLS 2023.2, 'complete' without dim = partition outermost dim only (same as dim=1),
    # leaving each [DEPTH] sub-array as a BRAM_AUTO RAM (1R1W, 2-cycle read latency).
    # Adding 'complete dim=2' partitions the inner dimension too, producing BANKS*DEPTH
    # individual scalar registers (FF, 0-cycle combinational access).
    for var in input_banks:
        hls_code = hls_code.replace(
            f"#pragma HLS array_partition variable={var} complete dim=1",
            f"#pragma HLS array_partition variable={var} complete dim=1\n"
            f"  #pragma HLS array_partition variable={var} complete dim=2",
        )

    # Step 2b: output banks keep 'complete dim=1' and 'bind_storage lutram' (no change needed)

    # Step 3: remove 'bind_storage ... type=ram_2p impl=lutram' only for INPUT banks
    # (which are now fully register-partitioned and don't need any RAM binding).
    # Output banks KEEP their bind_storage lutram pragma to use LUTRAM instead of BRAM.
    for var in input_banks:
        hls_code = re.sub(
            rf"#pragma HLS bind_storage variable={re.escape(var)} type=ram_2p impl=lutram\n",
            "",
            hls_code,
        )

    return hls_code


def add_local_array_partition_pragmas(hls_code):
    """Add ``complete`` array_partition pragmas for local output arrays in pipelined loops.

    In butterfly compute loops, local arrays like ``o_re[32]`` and ``o_im[32]`` are used to
    accumulate 16x-unrolled butterfly results. Without ``complete`` partition, the 32 writes
    serialize through a 2-port array, causing high iteration_latency (16+ cycles).
    With ``complete`` partition all 32 elements become individual registers, enabling
    all 32 parallel writes and reducing iteration_latency to ~5-7 cycles.

    Enabled via ``configs={'add_local_array_partition': True}`` or automatically with
    ``optimize_stream_reads``.
    """
    lines = hls_code.split("\n")
    result = []
    # Match float arrays declared inside loop bodies: "    float NAME[N];\t// L..."
    # These are local SSA-named arrays (vN) or named arrays (o_re, o_im, etc.)
    pat = re.compile(r"^(\s{4,})(float) (\w+)\[(\d+)\];\t// L\d+$")
    for line in lines:
        result.append(line)
        m = pat.match(line)
        if m:
            name = m.group(3)
            indent = m.group(1)
            # Add complete partition for local compute buffers (not the 2D bank arrays)
            result.append(f"{indent}#pragma HLS array_partition variable={name} complete")
    return "\n".join(result)


def add_compute_loop_dependence_pragmas(hls_code):
    """Add ``dependence inter false`` pragmas INSIDE compute loops for output bank arrays.

    When HLS DATAFLOW extracts ``l_S_i_2_*`` loops as separate sub-functions (via
    XFORM 203-721), ``#pragma HLS dependence`` pragmas declared at the parent function
    scope do not propagate into the extracted sub-function.  This causes HLS to report
    a false WAW carried-dependence violation (HLS 200-880) on ``out_re_b*``/``out_im_b*``
    and produce II=2.

    Fix: scan each ``l_S_i_2_*`` loop body, identify the output bank array names accessed
    (matching ``out_re_b*`` / ``out_im_b*``), and insert ``dependence inter false``
    pragmas immediately after ``#pragma HLS pipeline II=1`` inside the loop.  These
    pragmas travel with the loop when DATAFLOW extraction creates the sub-function.

    Enabled automatically with ``optimize_stream_reads`` or ``bind_op_fabric``.
    """
    lines = hls_code.split("\n")
    result = []

    # Pattern: the compute loop label (DATAFLOW sub-loop extracted as separate proc)
    compute_loop_pat = re.compile(r"^\s+l_S_i_2_\w+: for ")
    pipeline_pat = re.compile(r"^(\s+)#pragma HLS pipeline")
    # Output bank array names referenced in assignments (out_re_b, out_im_b, with optional suffix)
    out_bank_pat = re.compile(r"\b(out_re_b\w*|out_im_b\w*)\[")

    i = 0
    while i < len(lines):
        line = lines[i]
        if compute_loop_pat.match(line):
            # The loop opens with { on this line; next line should be #pragma HLS pipeline
            if i + 1 < len(lines) and pipeline_pat.match(lines[i + 1]):
                pipeline_line = lines[i + 1]
                m_pipe = pipeline_pat.match(pipeline_line)
                indent = m_pipe.group(1)

                # Scan loop body (starting at i+2) to collect output bank array names.
                # We start at depth=1 (inside the loop's opening {).
                depth = 1
                j = i + 2
                out_vars = set()
                while j < len(lines) and depth > 0:
                    depth += lines[j].count("{") - lines[j].count("}")
                    if depth > 0:
                        m_out = out_bank_pat.search(lines[j])
                        if m_out:
                            out_vars.add(m_out.group(1))
                    j += 1

                # Emit loop label and pipeline pragma
                result.append(line)
                result.append(pipeline_line)
                # Add dependence pragmas inside the loop body
                for var in sorted(out_vars):
                    result.append(
                        f"{indent}#pragma HLS dependence variable={var} inter false"
                    )
                i += 2
                continue
        result.append(line)
        i += 1

    return "\n".join(result)


def convert_io_to_streams(hls_code, n_vecs, width, stream_arg_indices=None):
    """Convert top-level array arguments to hls::stream interface.

    Transforms the generated HLS code so that selected top-level I/O arguments
    become ``hls::stream<hls::vector<float, WIDTH>>`` instead of
    ``float varname[N_VECS][WIDTH]``.  The corresponding ``load_buf{N}`` and
    ``store_res{N}`` wrapper functions are updated to read from / write to the
    stream rather than copying from/to a flat array.

    Parameters
    ----------
    hls_code : str
        The HLS C++ source string as emitted by allo_d.emit_vhls.
    n_vecs : int
        Outer dimension of the I/O arrays (e.g. NUM_VECS = N // WIDTH).
    width : int
        Inner dimension / stream vector width (e.g. WIDTH = 32).
    stream_arg_indices : list[int] | None
        Which top-function argument positions to convert.  When *None*, ALL
        array arguments are converted (the common case when stream_io=True).

    Transformation steps
    --------------------
    1. **load_buf{N} / store_res{N} signatures** – replace the first argument
       ``float v{A}[N][W]`` with ``hls::stream<hls::vector<float, W>>& v{A}``.
    2. **load_buf{N} body** – replace the nested copy loop with a stream ``.read()``
       call that unpacks each ``hls::vector`` token into the local buffer row.
    3. **store_res{N} body** – replace the nested copy loop with a stream
       ``.write()`` call that packs each local buffer row into an ``hls::vector``
       token and sends it.
    4. **Top function signature** – replace ``float v{X}[N][W]`` with
       ``hls::stream<hls::vector<float, W>>& v{X}`` for every converted argument.
    5. **Top function body** – remove the ``#pragma HLS array_partition`` pragma
       that was inserted for the (now-gone) array argument, and remove the
       ``load_buf`` / ``store_res`` calls so the kernels read directly from the
       streams passed in from outside.

    Returns the modified HLS source string.
    """
    stream_type = f"hls::stream<hls::vector<float, {width}>>"
    n_str = str(n_vecs)
    w_str = str(width)

    # ------------------------------------------------------------------
    # Step 1 & 2 & 3: Transform load_buf / store_res function bodies.
    # Pattern for a load_buf / store_res function:
    #
    #   void load_bufN(
    #     float vA[N][W],     <- first arg = external array / stream candidate
    #     float vB[N][W]      <- second arg = internal buffer
    #   ) {
    #     #pragma HLS array_partition variable=vA complete dim=2
    #     #pragma HLS array_partition variable=vB complete dim=2
    #     l_S_...: for (int ...l_0 = 0; ...l_0 < N; ...l_0++) {
    #     #pragma HLS pipeline II=1 rewind
    #       l_...: for (int ...l_1 = 0; ...l_1 < W; ...l_1++) {
    #       #pragma HLS unroll
    #         float vT = vA[...l_0][...l_1];
    #         vB[...l_0][...l_1] = vT;
    #       }
    #     }
    #   }
    # ------------------------------------------------------------------
    def _transform_wrapper_func(m):
        """Called for each load_buf / store_res match."""
        func_type = m.group("ftype")   # 'load_buf' or 'store_res'
        func_idx  = m.group("fidx")    # numeric suffix
        first_var = m.group("ext_var")  # first argument variable name
        second_var = m.group("buf_var")  # second argument variable name

        # Only convert this wrapper if the argument is in the target index set.
        arg_idx = int(func_idx)
        if stream_arg_indices is not None and arg_idx not in stream_arg_indices:
            return m.group(0)  # no change

        if func_type == "load_buf":
            # load_buf(external_input, internal_buffer):
            #   first arg is external (becomes stream), second is internal buffer.
            stream_var = first_var
            buf_var = second_var
            new_sig = (
                f"void {func_type}{func_idx}(\n"
                f"  {stream_type}& {stream_var},\n"
                f"  float {buf_var}[{n_str}][{w_str}]\n"
                f") {{"
            )
            new_body = (
                f"\n  #pragma HLS array_partition variable={buf_var} complete dim=2\n\n"
                f"  for (int _si = 0; _si < {n_str}; _si++) {{\n"
                f"  #pragma HLS pipeline II=1 rewind\n"
                f"    hls::vector<float, {w_str}> _vtmp = {stream_var}.read();\n"
                f"    for (int _sk = 0; _sk < {w_str}; _sk++) {{\n"
                f"    #pragma HLS unroll\n"
                f"      {buf_var}[_si][_sk] = _vtmp[_sk];\n"
                f"    }}\n"
                f"  }}\n"
            )
        else:  # store_res
            # store_res(internal_buffer, external_output):
            #   first arg is internal buffer, second is external (becomes stream).
            buf_var = first_var
            stream_var = second_var
            new_sig = (
                f"void {func_type}{func_idx}(\n"
                f"  float {buf_var}[{n_str}][{w_str}],\n"
                f"  {stream_type}& {stream_var}\n"
                f") {{"
            )
            new_body = (
                f"\n  #pragma HLS array_partition variable={buf_var} complete dim=2\n\n"
                f"  for (int _si = 0; _si < {n_str}; _si++) {{\n"
                f"  #pragma HLS pipeline II=1 rewind\n"
                f"    hls::vector<float, {w_str}> _vtmp;\n"
                f"    for (int _sk = 0; _sk < {w_str}; _sk++) {{\n"
                f"    #pragma HLS unroll\n"
                f"      _vtmp[_sk] = {buf_var}[_si][_sk];\n"
                f"    }}\n"
                f"    {stream_var}.write(_vtmp);\n"
                f"  }}\n"
            )

        return new_sig + new_body + "}"

    # Match each load_buf / store_res function definition.
    # The regex captures the two argument variable names and the full function body.
    wrapper_pat = re.compile(
        r"void (?P<ftype>load_buf|store_res)(?P<fidx>\d+)\(\s*"
        r"float (?P<ext_var>\w+)\[" + n_str + r"\]\[" + w_str + r"\],\s*"
        r"float (?P<buf_var>\w+)\[" + n_str + r"\]\[" + w_str + r"\]\s*"
        r"\) \{(?P<body>.*?)\n\}",
        re.DOTALL,
    )
    hls_code = wrapper_pat.sub(_transform_wrapper_func, hls_code)

    # ------------------------------------------------------------------
    # Step 4: Transform the top function signature.
    # The top function may have multiple array arguments; we need to convert
    # only those whose positional index is in stream_arg_indices.
    # We do a single-pass replacement inside the argument list of the top function.
    # ------------------------------------------------------------------
    # Find the top function signature (lines between "void <top>(" and ") {")
    top_sig_pat = re.compile(
        r"(void \w+\(\s*)((?:  float \w+\[" + n_str + r"\]\[" + w_str + r"\][,\s]*\n)+)(\) \{)",
        re.MULTILINE,
    )

    def _transform_top_sig(m):
        prefix = m.group(1)
        args_block = m.group(2)
        suffix = m.group(3)

        # Each arg line looks like:  "  float vXXX[8][32],\n"
        arg_lines = re.findall(
            r"  float (\w+)\[" + n_str + r"\]\[" + w_str + r"\](,?)\n",
            args_block,
        )
        result_lines = []
        for pos, (var_name, comma) in enumerate(arg_lines):
            if stream_arg_indices is None or pos in stream_arg_indices:
                result_lines.append(
                    f"  {stream_type}& {var_name}{comma}\n"
                )
            else:
                result_lines.append(
                    f"  float {var_name}[{n_str}][{w_str}]{comma}\n"
                )
        return prefix + "".join(result_lines) + suffix

    hls_code = top_sig_pat.sub(_transform_top_sig, hls_code)

    # ------------------------------------------------------------------
    # Step 5: In the top function body, remove only the
    #   "#pragma HLS array_partition variable=vXXX complete dim=2" pragmas
    #   for converted stream args (they are no longer arrays, so the pragma
    #   is meaningless and would cause HLS warnings).
    #
    # The load_buf / store_res CALL sites are kept because they are still
    # the mechanism that transfers data between the top-level stream interface
    # and the internal local buffers.  The wrapper functions themselves were
    # already updated in steps 1-3 to read/write hls::stream instead of arrays.
    #
    # The top-function argument variable names are now "hls::stream<...>& vXXX"
    # in the transformed signature - collect them from the whole code string.
    # ------------------------------------------------------------------
    top_stream_vars = set(
        re.findall(re.escape(stream_type) + r"& (\w+)", hls_code)
    )

    if top_stream_vars:
        lines = hls_code.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            # Remove array_partition pragmas for stream-converted top-level args
            if stripped.startswith("#pragma HLS array_partition"):
                var_match = re.search(r"variable=(\w+)\b", stripped)
                if var_match and var_match.group(1) in top_stream_vars:
                    continue
            result.append(line)
        hls_code = "\n".join(result)

    return hls_code


def fix_deduped_global_refs(hls_code):
    """Rewrite placeholder SSA names for deduplicated global constants.

    When the same numpy array is used in multiple kernels (e.g. ``full_twr``
    used by intra_2 … inter_7), the IR builder now emits a single
    ``memref.GlobalOp`` (symbol ``@twr``).  The MLIR HLS emitter assigns the
    first ``GetGlobalOp`` result the name ``twr``, and subsequent ones the
    names ``twr1``, ``twr2``, … — these extra names are not defined anywhere
    in the generated C, causing compile errors.

    This pass finds every ``// placeholder for const float <var>`` comment and,
    if ``<var>`` looks like a deduplicated name (base name with a numeric
    suffix, e.g. ``twr1``), replaces all references to it with the canonical
    base name (``twr``).
    """
    placeholder_re = re.compile(
        r"//\s*placeholder for const \w+ (\w+)"
    )
    replacements = {}
    for m in placeholder_re.finditer(hls_code):
        var = m.group(1)  # e.g. "twr1"
        base = var.rstrip("0123456789")  # e.g. "twr"
        if base and base != var:
            replacements[var] = base
    for var, base in replacements.items():
        hls_code = re.sub(rf"\b{re.escape(var)}\b", base, hls_code)
    return hls_code


def add_const_to_global_arrays(hls_code):
    """Add ``const`` qualifier to global float array declarations.

    Allo's HLS codegen emits twiddle factor lookup tables as mutable global arrays:
        float twr[128] = {1.0, ...};
    Without ``const``, HLS treats the array as potentially modified at runtime and
    will NOT constant-fold reads even when the array index is a compile-time constant
    after loop unrolling.  This prevents HLS from eliminating multiplications by 0.0
    or 1.0 (e.g., in stage-1 intra butterflies where tw=(1,0) or tw=(0,-1)).

    The reference design uses ``static const float twiddle_re[N/2] = {...}`` which
    allows HLS to propagate the twiddle constants through the unrolled multiply chain,
    eliminating DSP usage for trivial twiddle factors.

    This transformation adds ``const`` to all global float array declarations that
    have an initializer list ``= {...}``, making them read-only and enabling HLS
    constant-folding.

    Enabled automatically when ``optimize_stream_reads`` or ``bind_op_fabric`` is True.
    """
    # Match: "float NAME[N] = {..." at the start of a line (global scope, not indented)
    pat = re.compile(r"^(float \w+\[[\d\]\[]*\] = \{)", re.MULTILINE)
    return pat.sub(r"const \1", hls_code)


def add_bind_op_fabric_pragmas(hls_code):
    """Add #pragma HLS bind_op variable=V op=fadd/fsub impl=fabric after float add/sub assignments.

    This reduces floating-point add/sub operation latency from ~5 cycles (DSP-backed) to
    ~1 cycle (LUT-based fabric), reducing pipeline depth in FFT butterfly computations.

    **Important**: The pragma is NOT added to intermediate butterfly results
    (e.g., ``bw_re = b_re*tw_re - b_im*tw_im``) where both inputs come from
    float multiplications.  Those operations use ``FAddSub_primitivedsp`` (Latency=0,
    combinational) by default, which is faster than fabric (Latency=3).  Only the
    final butterfly outputs (``out = a ± bw``) get the ``impl=fabric`` pragma,
    matching the reference design strategy.

    Specifically, the pragma is skipped when the immediately preceding code line is a
    float multiplication (``float vN = vA * vB``), which indicates the current
    add/sub is an intermediate bw computation, not a final butterfly output.

    Enabled via ``configs={'bind_op_fabric': True}`` in ``s.build()``.
    """
    lines = hls_code.split("\n")
    result = []
    # Match SSA-form float add: `  float vN = vA + vB;\t// L...`
    pat_add = re.compile(r"^(\s+)float (v\d+) = v\d+ \+ v\d+;\t// L\d+$")
    # Match SSA-form float sub: `  float vN = vA - vB;\t// L...`
    pat_sub = re.compile(r"^(\s+)float (v\d+) = v\d+ - v\d+;\t// L\d+$")
    # Match float multiplication: `  float vN = vA * vB;  // L...`
    mul_pat = re.compile(r"^\s+float v\d+ = v\d+ \* v\d+;\t// L\d+$")

    prev_code_line = ""
    for line in lines:
        result.append(line)
        m = pat_add.match(line)
        if m:
            if not mul_pat.match(prev_code_line):
                result.append(
                    f"{m.group(1)}#pragma HLS bind_op variable={m.group(2)} op=fadd impl=fabric"
                )
            prev_code_line = line
            continue
        m = pat_sub.match(line)
        if m:
            if not mul_pat.match(prev_code_line):
                result.append(
                    f"{m.group(1)}#pragma HLS bind_op variable={m.group(2)} op=fsub impl=fabric"
                )
            prev_code_line = line
            continue
        # Track previous non-pragma, non-blank code line
        stripped = line.strip()
        if stripped and not stripped.startswith("#pragma") and not stripped.startswith("//"):
            prev_code_line = line
    return "\n".join(result)


def separate_header(hls_code, top=None, extern_c=True):
    func_decl = False
    sig_str = "#ifndef KERNEL_H\n"
    sig_str += "#define KERNEL_H\n\n"
    args = []
    if extern_c:
        sig_str += 'extern "C" {\n'
    for line in hls_code.split("\n"):
        if line.startswith(f"void {top}"):
            func_decl = True
            sig_str += line + "\n"
        elif func_decl and line.startswith(") {"):
            func_decl = False
            sig_str += ");\n"
            break
        elif func_decl:
            arg_type = line.strip()
            _, var = arg_type.rsplit(" ", 1)
            comma = "," if var[-1] == "," else ""
            ele_type = arg_type.split("[")[0].split(" ")[0].strip()
            allo_type = None
            if ele_type in c2allo_type:
                allo_type = c2allo_type[ele_type]
            else:
                pattern = r"^ap_(u?)int<(\d+)>$"
                match = re.match(pattern, ele_type)
                if not match:
                    raise ValueError(f"Fail to resolve ctype {ele_type}")
                unsigned_flag, width = match.groups()
                allo_type = f"{'u' if unsigned_flag else ''}int{int(width)}"
            shape = tuple(s.split("]")[0] for s in arg_type.split("[")[1:])
            args.append((allo_type, shape))
            if "[" in var:  # array
                var = var.split("[")[0]
                sig_str += "  " + ele_type + " *" + var + f"{comma}\n"
            else:  # scalar
                var = var.split(",")[0]
                sig_str += "  " + ele_type + " " + var + f"{comma}\n"
    if extern_c:
        sig_str += '} // extern "C"\n'
    sig_str += "\n#endif // KERNEL_H\n"
    return sig_str, args


class HLSModule:
    def __init__(
        self,
        mod,
        top_func_name,
        platform="vivado_hls",
        mode=None,
        project=None,
        ext_libs=None,
        configs=None,
        func_args=None,
        wrap_io=True,
    ):
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project
        self.platform = platform
        self.ext_libs = [] if ext_libs is None else ext_libs
        self.num_output_args = 0  # Will be set from configs if provided
        if configs is not None:
            new_configs = DEFAULT_CONFIG.copy()
            new_configs.update(configs)
            configs = new_configs
            self.num_output_args = configs.get("num_output_args", 0)
        else:
            configs = DEFAULT_CONFIG.copy()
        if self.mode is not None:
            configs["mode"] = self.mode
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            func = find_func_in_module(self.module, top_func_name)
            func.attributes["top"] = UnitAttr.get()

            if platform in {"vitis_hls", "pynq"}:
                assert func_args is not None, "Need to specify func_args"
                if wrap_io:
                    generate_input_output_buffers(
                        self.module,
                        top_func_name,
                        flatten=True,
                        mappings=configs.get("mappings", None),
                    )

            self.module = decompose_library_function(self.module)
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                # used for lowering tensor.empty
                "empty-tensor-to-alloc-tensor,"
                # translate tensor dialect (virtual) to memref dialect (physical)
                # "one-shot-bufferize{bufferize-function-boundaries},"
                # common lowering passes
                "func.func(convert-linalg-to-affine-loops)"
                # DO NOT LOWER AFFINE DIALECT
                ")"
            )
            pm.run(self.module.operation)
        buf = io.StringIO()
        success = True
        match platform:
            case "tapa":
                success = allo_d.emit_thls(self.module, buf)
            case "intel_hls":
                success = allo_d.emit_ihls(self.module, buf)
            case "catapult":
                success = allo_d.emit_catapult(self.module, buf)
            case _:
                # wrap_io=True has already linearized array indexing in
                # generate_input_output_buffers, so we don't need to do it again.
                # configs["flatten"] overrides the default: use it when wrap_io=False
                # but the top function has multi-dim array args (e.g. dataflow regions).
                if configs is not None and configs.get("flatten") is not None:
                    flatten = configs["flatten"]
                else:
                    flatten = False if platform == "vivado_hls" else (not wrap_io)
                success = allo_d.emit_vhls(self.module, buf, flatten=flatten)

        if not success:
            raise RuntimeError(
                "Failed to emit HLS code. Check error messages above for details. "
                "Common issues: nested functions with multi-dimensional arrays when wrap_io=False."
            )

        buf.seek(0)
        self.hls_code = buf.read()
        if platform in {"vitis_hls", "vivado_hls"}:
            # optimize_stream_reads is enabled explicitly or implicitly via bind_op_fabric
            if configs.get("optimize_stream_reads", False) or configs.get(
                "bind_op_fabric", False
            ):
                self.hls_code = optimize_stream_reads(self.hls_code)
                self.hls_code = fix_bank_array_partition(self.hls_code)
                self.hls_code = add_local_array_partition_pragmas(self.hls_code)
                self.hls_code = add_compute_loop_dependence_pragmas(self.hls_code)
                self.hls_code = add_const_to_global_arrays(self.hls_code)
                self.hls_code = fix_deduped_global_refs(self.hls_code)
            if configs.get("bind_op_fabric", False):
                self.hls_code = add_bind_op_fabric_pragmas(self.hls_code)
            if configs.get("stream_io", False):
                stream_io_cfg = configs["stream_io"]
                # stream_io may be True (convert all) or a dict with keys:
                #   n_vecs, width, indices (list of arg positions to convert)
                if isinstance(stream_io_cfg, dict):
                    n_vecs = stream_io_cfg.get("n_vecs", None)
                    width  = stream_io_cfg.get("width", None)
                    indices = stream_io_cfg.get("indices", None)
                else:
                    # Infer n_vecs and width from the IO mappings if available.
                    # Each mapping entry is (shape, ...) where shape=[n_vecs, width].
                    mappings = configs.get("mappings", None)
                    n_vecs = None
                    width = None
                    if mappings and len(mappings) > 0 and mappings[0] is not None:
                        shape = mappings[0][0]  # e.g. [NUM_VECS, WIDTH]
                        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                            n_vecs = shape[0]
                            width  = shape[1]
                    indices = None
                if n_vecs is not None and width is not None:
                    self.hls_code = convert_io_to_streams(
                        self.hls_code, n_vecs, width, stream_arg_indices=indices
                    )
        if project is not None:
            assert mode is not None, "mode must be specified when project is specified"
            os.makedirs(project, exist_ok=True)
            path = os.path.dirname(__file__)
            path = os.path.join(path, "../harness/")
            if platform in {"vivado_hls", "vitis_hls", "tapa", "pynq", "catapult"}:
                os.system("cp " + path + f"{platform.split('_')[0]}/* " + project)
                with open(f"{project}/run.tcl", "w", encoding="utf-8") as outfile:
                    if platform == "catapult":
                        outfile.write(codegen_tcl_catapult(top_func_name, configs))
                    else:
                        outfile.write(codegen_tcl(top_func_name, configs))
            copy_ext_libs(ext_libs, project)
            if self.platform == "vitis_hls":
                assert self.mode in {
                    "csim",
                    "csyn",
                    "sw_emu",
                    "hw_emu",
                    "hw",
                }, "Invalid mode"
                assert (
                    self.top_func_name != "kernel"
                ), "kernel is a reserved keyword for vitis_hls"
                path = os.path.dirname(__file__)
                path = os.path.join(path, "../harness/")
                dst_path = os.path.join(project, "description.json")
                generate_description_file(
                    self.top_func_name,
                    path + "makefile_gen/description.json",
                    dst_path,
                    frequency=configs["frequency"],
                )
                hbm_mapping = configs.get("hbm_mapping", None)
                generate_makefile(dst_path, project, self.platform, hbm_mapping)
                header, self.args = separate_header(self.hls_code, self.top_func_name)
                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
                self.hls_code = postprocess_hls_code(self.hls_code, self.top_func_name)

                # Generate HBM/DDR configuration file if hbm_mapping is provided
                # This must be done AFTER postprocess_hls_code to get correct arg names
                if hbm_mapping is not None:
                    # Extract HLS argument names from the postprocessed code
                    hls_arg_names = extract_hls_arg_names(
                        self.hls_code, self.top_func_name
                    )
                    # Build mapping from user arg names to HLS arg names
                    user_arg_names = []
                    if func_args is not None and self.top_func_name in func_args:
                        for arg in func_args[self.top_func_name]:
                            if hasattr(arg, "name"):
                                user_arg_names.append(arg.name)
                            else:
                                user_arg_names.append(str(arg))
                    # Add return value name - it becomes the last argument
                    # Use the last HLS arg name count to determine if there's a return
                    if len(hls_arg_names) > len(user_arg_names):
                        # There's a return value, add placeholder names
                        for i in range(len(hls_arg_names) - len(user_arg_names)):
                            user_arg_names.append(f"output_{i}")

                    arg_name_mapping = None
                    if len(user_arg_names) == len(hls_arg_names):
                        arg_name_mapping = dict(zip(user_arg_names, hls_arg_names))

                    cfg_content = generate_hbm_config(
                        self.top_func_name, hbm_mapping, arg_name_mapping
                    )
                    cfg_path = os.path.join(project, f"{self.top_func_name}.cfg")
                    with open(cfg_path, "w", encoding="utf-8") as cfg_file:
                        cfg_file.write(cfg_content)
                for lib in self.ext_libs:
                    cpp_file = lib.impl.split("/")[-1]
                    with open(f"{project}/{cpp_file}", "r", encoding="utf-8") as infile:
                        new_code = postprocess_hls_code(
                            infile.read(), lib.top, pragma=False
                        )
                    with open(
                        f"{project}/{cpp_file}", "w", encoding="utf-8"
                    ) as outfile:
                        outfile.write(new_code)
                self.host_code = codegen_host(
                    self.top_func_name,
                    self.module,
                    num_output_args=self.num_output_args,
                )
            elif self.platform == "catapult":
                assert self.mode in {
                    "csim",
                    "csyn",
                }, "Invalid mode for catapult"

                if self.mode == "csim":
                    self.host_code = codegen_host_catapult(
                        self.top_func_name,
                        self.module,
                    )
                else:
                    self.host_code = ""

                # For Catapult, we don't have separate kernel.h generation logic yet
                # similar to separate_header. The kernel.cpp contains everything needed
                # or headers are handled differently.
                # If we want to support csim, kernel.cpp usually needs a header
                # referenced by host.cpp.
                # allo/backend/catapult.py's codegen_host includes "kernel.h".
                # So we SHOULD generate kernel.h.
                # Re-using separate_header which is generic enough for C-style headers.
                #
                # However, separate_header currently only understands builtin and
                # ap_(u)int<...> types. When Catapult emits ac_int<...> (e.g., for
                # non-standard integer widths), separate_header can raise ValueError.
                # Fall back to including kernel.cpp directly if that happens.
                try:
                    header, self.args = separate_header(
                        self.hls_code, self.top_func_name
                    )
                except ValueError:
                    header = '#pragma once\n#include "kernel.cpp"\n'
                    self.args = []
                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
            elif self.platform == "tapa":
                assert self.mode in {
                    "csim",
                    "fast_hw_emu",
                    "hw_emu",
                    "hw",
                }, "Invalid mode"
                assert (
                    self.top_func_name != "kernel"
                ), "kernel is a reserved keyword for tapa"
                path = os.path.dirname(__file__)
                path = os.path.join(path, "../harness/")
                dst_path = os.path.join(project, "description.json")
                generate_description_file(
                    self.top_func_name,
                    path + "makefile_gen/description.json",
                    dst_path,
                    frequency=configs["frequency"],
                )
                self.args = []
                hbm_mapping = configs.get("hbm_mapping", None)
                generate_makefile(dst_path, project, self.platform, hbm_mapping)
                # Generate HBM/DDR configuration file if hbm_mapping is provided
                if hbm_mapping is not None:
                    # Extract HLS argument names from the code
                    hls_arg_names = extract_hls_arg_names(
                        self.hls_code, self.top_func_name
                    )
                    # Build mapping from user arg names to HLS arg names
                    user_arg_names = []
                    if func_args is not None and self.top_func_name in func_args:
                        for arg in func_args[self.top_func_name]:
                            if hasattr(arg, "name"):
                                user_arg_names.append(arg.name)
                            else:
                                user_arg_names.append(str(arg))
                    # Add placeholder for return values if needed
                    if len(hls_arg_names) > len(user_arg_names):
                        for i in range(len(hls_arg_names) - len(user_arg_names)):
                            user_arg_names.append(f"output_{i}")

                    arg_name_mapping = None
                    if len(user_arg_names) == len(hls_arg_names):
                        arg_name_mapping = dict(zip(user_arg_names, hls_arg_names))

                    cfg_content = generate_hbm_config(
                        self.top_func_name, hbm_mapping, arg_name_mapping
                    )
                    cfg_path = os.path.join(project, f"{self.top_func_name}.cfg")
                    with open(cfg_path, "w", encoding="utf-8") as cfg_file:
                        cfg_file.write(cfg_content)
                # [NOTE] (Shihan): I guess tapa backend do not use this one. I modified codegen_host for vitis, similar logic should be updated for tapa if self.host_code is useful here
                self.host_code = codegen_host(
                    self.top_func_name,
                    self.module,
                )
                self.tapa_host = codegen_tapa_host(
                    self.top_func_name,
                    self.module,
                    self.hls_code,
                )
                with open(f"{project}/tapa_host.cpp", "w", encoding="utf-8") as outfile:
                    outfile.write(self.tapa_host)
            elif self.platform == "pynq":
                assert self.mode in {"csim", "csyn", "impl"}, "Invalid mode for pynq"
                kernel_h = os.path.join(project, "kernel.h")

                # Generate kernel.h
                header, self.args = separate_header(self.hls_code, self.top_func_name)
                with open(kernel_h, "w", encoding="utf-8") as outfile:
                    outfile.write(header)

                # Apply PYNQ-specific HLS code tweaks and write kernel.cpp
                self.hls_code = postprocess_hls_code_pynq(
                    self.hls_code, self.top_func_name
                )
            else:
                self.host_code = ""
            with open(f"{project}/kernel.cpp", "w", encoding="utf-8") as outfile:
                outfile.write(self.hls_code)
            if hasattr(self, "host_code") and self.host_code:
                with open(f"{project}/host.cpp", "w", encoding="utf-8") as outfile:
                    outfile.write(self.host_code)
            if len(ext_libs) > 0:
                for lib in ext_libs:
                    # Update kernel.cpp
                    new_kernel = ""
                    with open(
                        os.path.join(project, "kernel.cpp"), "r", encoding="utf-8"
                    ) as kernel:
                        for line in kernel:
                            new_kernel += line
                            if "#include <stdint.h>" in line:
                                new_kernel += f'#include "{lib.impl.split("/")[-1]}"\n'
                    with open(
                        os.path.join(project, "kernel.cpp"), "w", encoding="utf-8"
                    ) as kernel:
                        kernel.write(new_kernel)
                    # Update tcl file
                    new_tcl = ""
                    with open(
                        os.path.join(project, "run.tcl"), "r", encoding="utf-8"
                    ) as tcl_file:
                        for line in tcl_file:
                            new_tcl += line
                            if "# Add design and testbench files" in line:
                                cpp_file = lib.impl.split("/")[-1]
                                new_tcl += f"add_files {cpp_file}\n"
                    with open(
                        os.path.join(project, "run.tcl"), "w", encoding="utf-8"
                    ) as tcl_file:
                        tcl_file.write(new_tcl)

    def __repr__(self):
        if self.mode is None:
            return self.hls_code
        return f"HLSModule({self.top_func_name}, {self.mode}, {self.project})"

    def __call__(self, *args, shell=True):
        if self.platform == "vivado_hls":
            assert is_available("vivado_hls"), "vivado_hls is not available"
            ver = run_process("g++ --version", r"\d+\.\d+\.\d+")[0].split(".")
            assert (
                int(ver[0]) * 10 + int(ver[1]) >= 48
            ), f"g++ version too old {ver[0]}.{ver[1]}.{ver[2]}"

            cmd = f"cd {self.project}; make "
            if self.mode == "csim":
                cmd += "csim"
                out = run_process(cmd + " 2>&1")
                runtime = [k for k in out.split("\n") if "seconds" in k][0]
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Simulation runtime {runtime}"
                )

            elif "csyn" in self.mode or self.mode == "custom" or self.mode == "debug":
                cmd += self.platform
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")
                if self.mode != "custom":
                    out = parse_xml(
                        self.project,
                        "Vivado HLS",
                        top=self.top_func_name,
                        print_flag=True,
                    )

            else:
                raise RuntimeError(f"{self.platform} does not support {self.mode} mode")
        elif self.platform == "vitis_hls":
            assert is_available("vitis_hls"), "vitis_hls is not available"
            if self.mode == "csim":
                mod = IPModule(
                    top=self.top_func_name,
                    impl=f"{self.project}/kernel.cpp",
                    include_paths=[self.project],
                    link_hls=True,
                )
                mod(*args)
                return
            if self.mode == "csyn":
                cmd = f"cd {self.project}; vitis_hls -f run.tcl"
                assert len(args) == 0, "csyn mode does not need to pass in arguments"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")
                return
            # Use Makefile (sw_emu, hw_emu, hw)
            assert "XDEVICE" in os.environ, "Please set XDEVICE in your environment"
            # prepare data
            func = find_func_in_module(self.module, self.top_func_name)
            inputs, outputs = get_func_inputs_outputs(func)
            assert len(args) == len(inputs) + len(
                outputs
            ), f"Number of arguments mismatch, got {len(args)}, expected {len(inputs) + len(outputs)}"
            for i, ((in_dtype, in_shape), arg) in enumerate(zip(inputs, args)):
                assert (len(in_shape) == 0 and np.isscalar(arg)) or np.prod(
                    arg.shape
                ) == np.prod(
                    in_shape
                ), f"invalid arguemnt {i}, {np.asarray(arg).shape}-{in_shape}"
                ele_bitwidth = get_bitwidth_from_type(in_dtype)
                assert (
                    ele_bitwidth == 1 or ele_bitwidth % 8 == 0
                ), "can only handle bytes"
                # store as byte stream
                with open(f"{self.project}/input{i}.data", "wb") as f:
                    if np.isscalar(arg):
                        arg = np.array(arg, dtype=np_supported_types[in_dtype])
                    f.write(arg.tobytes())
            # check if the build folder exists
            bitstream_folder = f"{self.project}/build_dir.{self.mode}.{os.environ['XDEVICE'].rsplit('/')[-1].split('.')[0]}"
            if not os.path.exists(
                os.path.join(bitstream_folder, f"{self.top_func_name}.xclbin")
            ):
                cmd = (
                    f"cd {self.project}; make run TARGET={self.mode} PLATFORM=$XDEVICE"
                )
                print(cmd)
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to build the project")
            else:
                print("Build folder exists, skip building")
                # run the executable
                prefix = f"cd {self.project};"
                if not os.path.exists(f"{self.project}/{self.top_func_name}"):
                    prefix += " make host PLATFORM=$XDEVICE;"
                prefix += (
                    f" XCL_EMULATION_MODE={self.mode}" if self.mode != "hw" else ""
                )
                cmd = f"{prefix} ./{self.top_func_name} ../{bitstream_folder}/{self.top_func_name}.xclbin"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run the executable")
            # Read output tensors from files
            # Determine how many output files to read
            func = find_func_in_module(self.module, self.top_func_name)
            _, outputs = get_func_inputs_outputs(func)
            if len(outputs) > 0:
                # Original behavior: single output.data file
                if np.isscalar(args[-1]):
                    raise RuntimeError("The output must be a tensor")
                arr = np.fromfile(f"{self.project}/output.data", dtype=args[-1].dtype)
                args[-1][:] = arr.reshape(args[-1].shape)
            else:
                # Multiple output files: output0.data, output1.data, etc.
                num_out = self.num_output_args if self.num_output_args > 0 else 1
                for idx in range(num_out):
                    out_arg_idx = len(inputs) - num_out + idx
                    if out_arg_idx < 0 or out_arg_idx >= len(args):
                        continue
                    out_arg = args[out_arg_idx]
                    if np.isscalar(out_arg):
                        continue
                    arr = np.fromfile(
                        f"{self.project}/output{idx}.data", dtype=out_arg.dtype
                    )
                    out_arg[:] = arr.reshape(out_arg.shape)
            return
        elif self.platform == "pynq":
            # Do not assert PYNQ availability here; the presence of a physical
            # PYNQ device should be checked by callers that need it.
            if self.mode == "csim":
                cwd = os.getcwd()
                mod = IPModule(
                    top=self.top_func_name,
                    impl=f"{cwd}/{self.project}/kernel.cpp",
                    link_hls=True,
                )
                mod(*args)
                return
            if self.mode in {"csyn", "impl"}:
                # HLS synthesis
                cmd = f"cd {self.project}; vitis_hls -f run.tcl"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")

                if self.mode == "impl":
                    # Produce host (deploy.py)
                    host_code = codegen_pynq_host(
                        self.top_func_name,
                        self.module,
                        self.project,
                    )
                    with open(
                        f"{self.project}/deploy.py", "w", encoding="utf-8"
                    ) as outfile:
                        outfile.write(host_code)

                    # Vivado block design
                    bd_script = "block_design.tcl"
                    bd_script = os.path.basename(bd_script)
                    cmd = f"cd {self.project}; vivado -mode batch -source {bd_script}"
                    print(
                        f"[{time.strftime('%H:%M:%S', time.gmtime())}] Running Vivado Block Design ..."
                    )
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    if process.returncode != 0:
                        raise RuntimeError(
                            "Failed to create block design / generate bitstream"
                        )

                    # Package .bit / .hwh / deploy.py into deploy/ folder
                    deploy_dir = os.path.join(self.project, "deploy")
                    cmd = (
                        f"mkdir -p {deploy_dir}; "
                        f"cp {self.project}/build_vivado/project_1.runs/impl_1/project_1_bd_wrapper.bit {deploy_dir}/{self.top_func_name}.bit; "
                        f"cp {self.project}/build_vivado/project_1.gen/sources_1/bd/project_1_bd/hw_handoff/project_1_bd.hwh {deploy_dir}/{self.top_func_name}.hwh; "
                        f"cp {self.project}/deploy.py {deploy_dir}/deploy.py"
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S', time.gmtime())}] Collecting files for deployment ..."
                    )
                    print(f"Files for deployment located in {deploy_dir}")
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    if process.returncode != 0:
                        raise RuntimeError("Failed to collect files")
                return
        elif self.platform == "tapa":
            assert is_available("tapa"), "tapa is not available"
            # Use Makefile (sw_emu, hw_emu, hw)
            assert "XDEVICE" in os.environ, "Please set XDEVICE in your environment"
            # prepare data
            func = find_func_in_module(self.module, self.top_func_name)
            inputs, _ = get_func_inputs_outputs(func)
            for i, ((_, in_shape), arg) in enumerate(zip(inputs, args)):
                write_tensor_to_file(
                    arg,
                    in_shape,
                    f"{self.project}/input{i}.data",
                )
            # check if the build folder exists
            if self.mode in {"csim", "fast_hw_emu"}:
                cmd = f"cd {self.project}; make {self.mode}"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run tapa executable")
                return
            bitstream_folder = f"{self.project}/build_dir.{self.mode}.{os.environ['XDEVICE'].rsplit('/')[-1].split('.')[0]}"
            if not os.path.exists(
                os.path.join(bitstream_folder, f"{self.top_func_name}.xclbin")
            ):
                cmd = (
                    f"cd {self.project}; make run TARGET={self.mode} PLATFORM=$XDEVICE"
                )
                print(cmd)
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to build the project")
            else:
                print("Build folder exists, skip building")
                # run the executable
                prefix = f"cd {self.project};"
                if not os.path.exists(f"{self.project}/{self.top_func_name}"):
                    prefix += " make host PLATFORM=$XDEVICE;"
                prefix += (
                    f" XCL_EMULATION_MODE={self.mode}" if self.mode != "hw" else ""
                )
                cmd = f"{prefix} ./{self.top_func_name} ../{bitstream_folder}/{self.top_func_name}.xclbin"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run the executable")
            # suppose the last argument is the output tensor
            result = read_tensor_from_file(
                inputs[-1][0], args[-1].shape, f"{self.project}/output.data"
            )
            args[-1][:] = result
            return
        if self.platform == "catapult":
            if self.mode == "csim":
                # Check for input arguments
                func = find_func_in_module(self.module, self.top_func_name)
                inputs, outputs = get_func_inputs_outputs(func)
                assert len(args) == len(inputs) + len(
                    outputs
                ), f"Number of arguments mismatch, got {len(args)}, expected {len(inputs) + len(outputs)}"

                # Generate kernel.h
                # self.args might be updated by separate_header if needed, but for csim we use passed args
                header, _ = separate_header(
                    self.hls_code, self.top_func_name, extern_c=False
                )
                with open(
                    os.path.join(self.project, "kernel.h"), "w", encoding="utf-8"
                ) as outfile:
                    outfile.write(header)

                # Write input data
                for i, ((in_dtype, in_shape), arg) in enumerate(
                    zip(inputs, args[: len(inputs)])
                ):
                    write_tensor_to_file(arg, in_shape, f"{self.project}/input{i}.data")

                # Compilation with g++
                # Assuming 'g++' is in PATH.
                # Include path for ac_types
                mgc_home = os.environ.get("MGC_HOME")
                if not mgc_home:
                    raise RuntimeError(
                        "MGC_HOME environment variable is not set. Please set it to the Catapult installation directory."
                    )

                ac_include = os.path.join(mgc_home, "shared/include")
                if not os.path.isdir(ac_include):
                    raise RuntimeError(
                        f"Catapult headers not found at {ac_include}. Check MGC_HOME."
                    )

                cmd = f"cd {self.project}; g++ -std=c++11 -I{ac_include} kernel.cpp host.cpp -o sim"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Compiling with g++ ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(
                        "Failed to compile with g++. Check if g++ is installed and ac_types headers are correct."
                    )

                # Execution
                cmd = f"cd {self.project}; ./sim"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Running simulation ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Simulation failed.")

                # Read outputs
                for i, ((out_dtype, out_shape), out_arg) in enumerate(
                    zip(outputs, args[len(inputs) :])
                ):
                    if not os.path.exists(f"{self.project}/output{i}.data"):
                        raise RuntimeError(
                            f"Output file output{i}.data not found. Simulation might have failed."
                        )
                    result = read_tensor_from_file(
                        out_dtype, out_shape, f"{self.project}/output{i}.data"
                    )
                    out_arg[:] = result
                return

            if self.mode == "csyn":
                catapult_cmd = "catapult"
                if "MGC_HOME" in os.environ:
                    catapult_cmd = os.path.join(os.environ["MGC_HOME"], "bin/catapult")

                cmd = f"cd {self.project}; {catapult_cmd} -shell -f run.tcl"
                assert len(args) == 0, "csyn mode does not need to pass in arguments"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project with Catapult HLS ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(
                        "Failed to synthesize the design with Catapult HLS"
                    )
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Catapult HLS synthesis completed successfully"
                )
                return
            raise RuntimeError(
                f"Catapult backend currently only supports 'csyn' and 'csim' mode, got '{self.mode}'"
            )
        raise RuntimeError("Not implemented")

#!/usr/bin/env python3
"""E4: the SPMW FEATHER port over a whole workload, cosimulated tile by tile.

Registry-free: builds `feather_stream(N, N, NT)` (every operand streamed, a
weight file and a command per tile) or `feather_stream_x(N, N, NT)` (weights
and commands resident, activations streamed) through spmw_build_array's own
stage / synthesise / assemble, then drives the assembled array with a
$readmemh testbench of the same shape as allo.spmw.cosim's -- the same edge
streams, the same channel order (read off boundary_plan), the same PASS /
CYCLES lines -- plus a log of the cycle at which every output channel
completes each tile, so the whole-workload cycle count and the per-tile
completion events are measured, not extrapolated.

    e4_spmw_run.py --N 8 --workload gemm --gemm 128,128,128 --out <dir>
    e4_spmw_run.py --N 4 --workload conv --conv 64,16,16,64 --out <dir>
    e4_spmw_run.py --N 8 --workload gemm --resident --out <dir>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ALLO = "/scratch/hc676/allo"
for p in (ALLO, os.path.join(ALLO, "tests", "dataflow", "spmw"), os.path.join(ALLO, "scripts"), os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, p)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import allo.spmw as spmw  # noqa: E402  pylint: disable=wrong-import-position
from allo.spmw import rtl  # noqa: E402
from allo.spmw.rtl import StructuralEmitter, boundary_plan  # noqa: E402
from allo.spmw.cosim import _SCALAR_BITS, _one  # noqa: E402
from allo.spmw.ports import IN  # noqa: E402
import test_spmw_feather as F  # noqa: E402
import spmw_build_array as B  # noqa: E402
import e4_feather_gen as G  # noqa: E402


# -- the workload, the same tiles the RTL bench gets --------------------------


def workload_tiles(args):
    N = args.N
    rng = np.random.default_rng(args.seed)
    R = G.pattern_range(args.pattern, 0)
    host = {}
    if args.workload == "gemm":
        M, K, Nn = args.gemm
        A, Bm, tiles = G.gemm_workload(M, K, Nn, N, N, rng, 0, 0, R)
        host = {"A": A, "B": Bm, "dims": (M, K, Nn)}
    elif args.workload == "conv":
        Cin, H, Wd, M = args.conv
        x, w, tiles, geom = G.conv_workload(Cin, H, Wd, M, 3, 3, 1, N, N, rng, 0, 0, R)
        host = {"x": x, "w": w, "geom": geom, "dims": (Cin, H, Wd, M)}
    else:
        tiles = []
        for t in range(args.tiles):
            if N == 4:
                inst = G.conv_insts(N)[t % 4]
            elif t % 2 == 0:
                inst = G.gemm_insts(N)
            else:
                inst = rng.integers(0, 4, size=G.birrd_shape(N)).astype(np.int8)
            ia = rng.integers(0, R, size=(N, N)).astype(np.uint8)
            w = rng.integers(0, R, size=(N, N, N)).astype(np.uint8)
            tiles.append((ia, w, inst, (t,)))
    if args.limit:
        tiles = tiles[: args.limit]
    return tiles, host


def stream_arrays(tiles, N, resident):
    """The engine's arguments and the expected outputs for the tiles."""
    AW = AH = N
    P0, P1 = F.birrd_shape(AW)
    T = len(tiles)
    X = np.zeros((T * 2, AH, AW // 2), dtype=np.int32)
    W = None if resident else np.zeros((T * 2 * AH, AH, AW // 2), dtype=np.int32)
    I = None if resident else np.zeros((T, P0, P1), dtype=np.int32)
    Y = np.zeros((T * AH, AW), dtype=np.int32)
    w0 = tiles[0][1].astype(np.int8)
    inst0 = tiles[0][2]
    for t, (ia, w, inst, _) in enumerate(tiles):
        ia8 = ia.astype(np.int8)
        w8 = w0 if resident else w.astype(np.int8)
        inst_t = inst0 if resident else inst
        Xt, Wt, It = F.tile_operands(ia8, w8, inst_t, AW, AH)
        X[2 * t : 2 * t + 2] = Xt.transpose(2, 0, 1)
        if not resident:
            W[2 * AH * t : 2 * AH * (t + 1)] = Wt.transpose(2, 0, 1)
            I[t] = It[:, :, 0]
        Y[t * AH : (t + 1) * AH] = F.feather_ref(ia8, w8, inst_t, AW, AH)
    YL = np.ascontiguousarray(Y[:, 0::2])
    YR = np.ascontiguousarray(Y[:, 1::2])
    if resident:
        _, Wres, Ires = F.tile_operands(tiles[0][0].astype(np.int8), w0, inst0, AW, AH)
        return {"X": X, "W": Wres, "Inst": Ires, "YL": YL, "YR": YR}, Y
    return {"X": X, "W": W, "Inst": I, "YL": YL, "YR": YR}, Y


# -- channels ------------------------------------------------------------------


def channel_specs(design, N, resident):
    """Per boundary family: direction, tensor, and for each channel either
    ('stream', trailing coordinates) -- tokens along the tensor's leading axis
    -- or ('block', full index). Read off boundary_plan of the same design at
    two small NT, which must agree (the order is a property of the sites)."""
    specs = None
    for nt in (2, 3):
        fabric = F.feather_stream_x(N, N, nt) if resident else F.feather_stream(N, N, nt)
        plan = boundary_plan(spmw.elaborate(fabric))
        cur = {}
        for name, entry in plan.items():
            chans = []
            for idx_list in entry["channels"]:
                raw = idx_list[0]
                if len(idx_list) > 1:
                    first = tuple(int(v) for v in raw)
                    coord = first[1:]
                    want = [(i,) + coord for i in range(len(idx_list))]
                    got = [tuple(int(v) for v in idx) for idx in idx_list]
                    assert got == want, (name, got[:3], want[:3])
                    chans.append(("stream", coord))
                else:
                    # a resident file: the site's whole block, indexed with a slice
                    coord = tuple(v if isinstance(v, slice) else int(v) for v in raw)
                    chans.append(("block", coord))
            cur[name] = {"direction": entry["direction"], "tensor": entry["tensor"], "channels": chans}
        if specs is None:
            specs = cur
        else:
            assert specs == cur, "boundary_plan's channel order changed with NT"
    return specs


def pack_block(values, element_bits):
    packed = 0
    for pos, v in enumerate(np.asarray(values).reshape(-1)):
        packed |= (int(v) & ((1 << element_bits) - 1)) << (element_bits * pos)
    return packed


def write_family_hex(path, arr, chans, dtype_name, block):
    """count * depth tokens, channel-major, one hex word a line."""
    element = _SCALAR_BITS[dtype_name]
    if block:
        digits = (element * int(np.prod(block)) + 3) // 4
        with open(path, "w", encoding="utf-8") as f:
            for kind, coord in chans:
                assert kind == "block"
                f.write(f"{pack_block(arr[coord], element):0{digits}x}\n")
        return len(chans), 1
    digits = (element + 3) // 4
    depth = arr.shape[0]
    mask = (1 << element) - 1
    with open(path, "w", encoding="utf-8") as f:
        for kind, coord in chans:
            assert kind == "stream"
            column = np.asarray(arr[(slice(None),) + coord]).reshape(-1).astype(np.int64) & mask
            f.write("\n".join(f"{int(v):0{digits}x}" for v in column))
            f.write("\n")
    return len(chans), depth


# -- the testbench -------------------------------------------------------------


def render_tb(graph, specs, arrays, rows_per_tile, sim_dir, cycles):
    emitter = StructuralEmitter(graph)
    fams = {f.name: f for f in emitter.families()[1]}
    lines = [
        "`timescale 1ns/1ps",
        "",
        "module tb;",
        "  reg clk = 0, rst_n = 0;",
        "  always #5 clk = ~clk;",
        "  integer errors = 0;",
        "  integer produced = 0;",
        "  integer first = -1;",
        "  integer cycount = 0;",
        "  integer fd;",
        "  always @(posedge clk) cycount <= cycount + 1;",
        f"  localparam integer ROWS = {rows_per_tile};",
    ]
    total = 0
    conns = [".ap_clk(clk)", ".ap_rst_n(rst_n)"]
    sizes = {}
    for name, spec in specs.items():
        fam = fams[name]
        dtype_name = str(fam.dtype)
        width = _SCALAR_BITS[dtype_name]
        for extent in fam.block:
            width *= int(extent)
        arr = arrays[spec["tensor"]]
        kind = "src" if spec["direction"] == IN else "exp"
        count, depth = write_family_hex(os.path.join(sim_dir, f"{name}_{kind}.hex"), arr, spec["channels"], dtype_name, tuple(int(e) for e in fam.block))
        sizes[name] = (count, depth)
        if spec["direction"] == IN:
            lines += [
                f"  wire [{width - 1}:0] {name}_dout [0:{count - 1}];",
                f"  wire {name}_empty_n [0:{count - 1}];",
                f"  wire {name}_read [0:{count - 1}];",
                f"  reg [{width - 1}:0] {name}_src [0:{count * depth - 1}];",
                f"  integer {name}_p [0:{count - 1}];",
                f'  initial $readmemh("{name}_src.hex", {name}_src);',
                f"  genvar {name}_gk;",
                f"  generate for ({name}_gk = 0; {name}_gk < {count}; {name}_gk = {name}_gk + 1) begin : {name}_g",
                f"    initial {name}_p[{name}_gk] = 0;",
                f"    assign {name}_dout[{name}_gk] = {name}_src[{name}_gk * {depth} + (({name}_p[{name}_gk] < {depth}) ? {name}_p[{name}_gk] : {depth - 1})];",
                f"    assign {name}_empty_n[{name}_gk] = ({name}_p[{name}_gk] < {depth});",
                f"    always @(posedge clk) if (rst_n && {name}_read[{name}_gk] && {name}_empty_n[{name}_gk]) {name}_p[{name}_gk] <= {name}_p[{name}_gk] + 1;",
                "  end endgenerate",
            ]
            conns += [f".{name}_dout({name}_dout)", f".{name}_empty_n({name}_empty_n)", f".{name}_read({name}_read)"]
        else:
            total += count * depth
            lines += [
                f"  wire [{width - 1}:0] {name}_din [0:{count - 1}];",
                f"  wire {name}_write [0:{count - 1}];",
                f"  wire {name}_full_n [0:{count - 1}];",
                f"  reg [{width - 1}:0] {name}_exp [0:{count * depth - 1}];",
                f"  integer {name}_q [0:{count - 1}];",
                f'  initial $readmemh("{name}_exp.hex", {name}_exp);',
                f"  genvar {name}_gk;",
                f"  generate for ({name}_gk = 0; {name}_gk < {count}; {name}_gk = {name}_gk + 1) begin : {name}_g",
                f"    initial {name}_q[{name}_gk] = 0;",
                f"    assign {name}_full_n[{name}_gk] = 1'b1;",
                f"    always @(posedge clk) if (rst_n && {name}_write[{name}_gk]) begin",
                f"      if ({name}_q[{name}_gk] < {depth}) begin",
                f"        if ({name}_din[{name}_gk] !== {name}_exp[{name}_gk * {depth} + {name}_q[{name}_gk]]) begin",
                "          errors = errors + 1;",
                f'          if (errors <= 20) $display("MISMATCH {name}[%0d] step %0d: got %h want %h", {name}_gk, {name}_q[{name}_gk], {name}_din[{name}_gk], {name}_exp[{name}_gk * {depth} + {name}_q[{name}_gk]]);',
                "        end",
                f'        if ((({name}_q[{name}_gk] + 1) % ROWS) == 0) $fdisplay(fd, "{name} %0d %0d %0d", {name}_gk, ({name}_q[{name}_gk] + 1) / ROWS - 1, cycount);',
                f"        {name}_q[{name}_gk] <= {name}_q[{name}_gk] + 1;",
                "        produced = produced + 1;",
                "      end else begin",
                "        errors = errors + 1;",
                f'        if (errors <= 20) $display("EXTRA TOKEN on {name}[%0d]", {name}_gk);',
                "      end",
                "    end",
                "  end endgenerate",
            ]
            conns += [f".{name}_din({name}_din)", f".{name}_write({name}_write)", f".{name}_full_n({name}_full_n)"]
    lines.insert(11, f"  localparam integer TOTAL = {total};")
    lines += [
        "  spmw_top dut (" + ", ".join(conns) + ");",
        "  initial begin",
        '    fd = $fopen("tiles.log", "w");',
        "    repeat (4) @(posedge clk);",
        "    @(negedge clk) rst_n = 1;",
        f"    for (integer c = 0; c < {cycles}; c = c + 1) begin",
        "      @(posedge clk);",
        "      if (produced > 0 && first < 0) first = c;",
        "      if (produced == TOTAL) begin",
        '        $display("SPMW COSIM %s (%0d/%0d tokens, %0d errors)", errors == 0 ? "PASS" : "FAIL", produced, TOTAL, errors);',
        '        $display("SPMW CYCLES total=%0d first_out=%0d", c + 1, first + 1);',
        '        $display("E4 END cycount=%0d c=%0d", cycount, c);',
        "        $fclose(fd);",
        "        $finish;",
        "      end",
        "    end",
        '    $display("SPMW COSIM TIMEOUT (%0d/%0d tokens, %0d errors)", produced, TOTAL, errors);',
        '    $display("SPMW CYCLES total=-1 first_out=%0d", first + 1);',
        '    $display("E4 END cycount=%0d c=-1", cycount);',
        "    $fclose(fd);",
        "    $finish;",
        "  end",
        "endmodule",
    ]
    return "\n".join(lines) + "\n", sizes, total


# -- the build and the run -------------------------------------------------------


def run_cmd(cmd, cwd, log):
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workload", choices=("tile", "gemm", "conv"), default="tile")
    p.add_argument("--tiles", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="only the first tiles of the workload")
    p.add_argument("--gemm", type=lambda s: [int(v) for v in s.split(",")], default=[128, 128, 128])
    p.add_argument("--conv", type=lambda s: [int(v) for v in s.split(",")], default=[64, 16, 16, 64])
    p.add_argument("--pattern", default="small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resident", action="store_true", help="feather_stream_x: weights and commands resident")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--frequency", type=float, default=300.0)
    p.add_argument("--cycles", type=int, default=0, help="simulation bound (default from the tile count)")
    p.add_argument("--reuse-build", action="store_true", help="skip HLS/assembly if the array is already built here")
    args = p.parse_args()
    N = args.N
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    timing = {}

    t0 = time.time()
    tiles, host = workload_tiles(args)
    NT = len(tiles)
    arrays, Y = stream_arrays(tiles, N, args.resident)
    timing["operands_s"] = round(time.time() - t0, 1)
    np.savez_compressed(os.path.join(out, "operands.npz"), Y=Y, coords=np.array([t[3] for t in tiles]),
                        **{k: v for k, v in host.items() if isinstance(v, np.ndarray)})

    fabric = F.feather_stream_x(N, N, NT) if args.resident else F.feather_stream(N, N, NT)
    graph = spmw.elaborate(fabric)
    cost = rtl.cost(graph)
    rtl.check_netlist(graph)
    names_file = os.path.join(out, "roles.json")
    if args.reuse_build and os.path.exists(names_file):
        names = json.load(open(names_file, encoding="utf-8"))
        timing["hls_wall_s"] = None
        timing["vivado_elab_s"] = None
    else:
        t1 = time.time()
        names = B.stage(graph, out, B.PART, args.frequency)
        timing["stage_s"] = round(time.time() - t1, 1)
        t1 = time.time()
        hls = B.synthesise(out, names, jobs=args.jobs)
        timing["hls_wall_s"] = round(time.time() - t1, 1)
        timing["hls_cpu_s"] = round(sum(hls.values()), 1)
        elapsed, _ = B.assemble(out, B.PART, names)
        timing["vivado_elab_s"] = elapsed
        json.dump(names, open(names_file, "w", encoding="utf-8"))

    # the testbench and its hex files
    t1 = time.time()
    specs = channel_specs("x" if args.resident else "stream", N, args.resident)
    sim = os.path.join(out, "sim")
    os.makedirs(sim, exist_ok=True)
    cycles = args.cycles or (NT * (6 * N + 40) + 4000)
    tb, sizes, total = render_tb(graph, specs, arrays, N, sim, cycles)
    B._write(os.path.join(sim, "tb.sv"), tb)
    timing["tb_s"] = round(time.time() - t1, 1)

    # the simulation models of any Xilinx IP the roles instantiate, then the sources
    t1 = time.time()
    if not os.path.isdir(os.path.join(out, "ipgen")):
        B._write(os.path.join(out, "genip.tcl"), B.IPGEN_TCL.format(out=out, part=B.PART, root=out, roles=" ".join(names)))
        B._run(["vivado", "-mode", "batch", "-source", "genip.tcl", "-nojournal", "-nolog"], out)
    sources = ["spmw_fifo.sv", "spmw_const.sv", "spmw_top.sv", "tb.sv"]
    for name in sources[:-1]:
        shutil.copy(os.path.join(out, name), sim)
    for role in names:
        shutil.copy(os.path.join(out, role, f"{role}.sv"), sim)
        sources.append(f"{role}.sv")
        verilog = os.path.join(out, role, "prj", "sol", "syn", "verilog")
        if os.path.isdir(verilog):
            for name in os.listdir(verilog):
                if name.endswith(".v"):
                    shutil.copy(os.path.join(verilog, name), sim)
    for root, _dirs, files in os.walk(os.path.join(out, "ipgen")):
        if "sources_1" in root and os.sep + "ip" + os.sep in root + os.sep:
            for name in files:
                if name.endswith(".v"):
                    shutil.copy(os.path.join(root, name), sim)
    rc = run_cmd(["xvlog", "-sv"] + sources, sim, os.path.join(sim, "xvlog_sv.log"))
    if rc != 0:
        sys.exit(f"xvlog -sv failed, see {sim}/xvlog_sv.log")
    vfiles = [f for f in os.listdir(sim) if f.endswith(".v")]
    if vfiles:
        rc = run_cmd(["xvlog"] + vfiles, sim, os.path.join(sim, "xvlog_v.log"))
        if rc != 0:
            sys.exit(f"xvlog failed, see {sim}/xvlog_v.log")
    rc = run_cmd(["xelab", "tb", "-s", "tbsim", "-L", "floating_point_v7_1_16", "-L", "unisims_ver", "-L", "unimacro_ver", "-L", "secureip"], sim, os.path.join(sim, "xelab.log"))
    if rc != 0:
        sys.exit(f"xelab failed, see {sim}/xelab.log")
    timing["sim_compile_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    rc = run_cmd(["xsim", "tbsim", "-runall"], sim, os.path.join(sim, "xsim.log"))
    timing["xsim_s"] = round(time.time() - t1, 1)

    # results
    result = {"N": N, "NT": NT, "resident": args.resident, "workload": args.workload, "cost": cost, "timing": timing, "sizes": sizes, "total_tokens": total}
    text = open(os.path.join(sim, "xsim.log"), encoding="utf-8").read()
    for line in text.splitlines():
        if line.startswith("SPMW COSIM"):
            result["cosim"] = line.strip()
        elif line.startswith("SPMW CYCLES"):
            d = dict(kv.split("=") for kv in line.split()[2:])
            result["total_cycles"] = int(d["total"])
            result["first_out_cycles"] = int(d["first_out"])
        elif line.startswith("E4 END"):
            d = dict(kv.split("=") for kv in line.split()[2:])
            result["end_cycount"] = int(d["cycount"])
            result["end_c"] = int(d["c"])
    result["status"] = "pass" if "SPMW COSIM PASS" in text else "fail"
    done = {}
    tl = os.path.join(sim, "tiles.log")
    if os.path.exists(tl):
        for line in open(tl, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 4:
                t = int(parts[2])
                done[t] = max(done.get(t, -1), int(parts[3]))
    if done and "end_cycount" in result and result.get("end_c", -1) >= 0:
        # cycount -> the standard bench's cycle scale (c + 1)
        shift = (result["end_c"] + 1) - result["end_cycount"]
        seq = np.array([done[t] + shift for t in range(NT) if t in done], dtype=np.int64)
        result["tiles_logged"] = int(len(seq))
        result["tile_done_cycles"] = seq.tolist() if NT <= 64 else seq[:8].tolist() + seq[-8:].tolist()
        result["first_tile_done"] = int(seq[0])
        result["last_tile_done"] = int(seq[-1])
        if len(seq) > 1:
            gaps = np.diff(seq)
            result["interval"] = {"min": int(gaps.min()), "median": float(np.median(gaps)), "max": int(gaps.max()), "mean": float(gaps.mean()), "mean_last32": float(gaps[-32:].mean())}
    # The host side: the rows the cosim verified (Y, one [N, N] block a tile,
    # the drivers' column order) reduced as the workload's host reduces them,
    # against numpy on the same int8 tensors (the stored bytes read as int8:
    # with --pattern full that is the whole mixed-sign range).
    result["operands"] = {"pattern": args.pattern, "seed": args.seed, "dtype": "int8 (stored bytes read as two's complement)",
                          "range": f"[0, {G.pattern_range(args.pattern, 0)}) as bytes" + (" = [-128, 127] as int8" if G.pattern_range(args.pattern, 0) == 256 else "")}
    if result["status"] == "pass" and args.workload in ("gemm", "conv") and not args.resident and not args.limit:
        rows = [Y[t * N : (t + 1) * N].astype(np.int64) for t in range(NT)]
        coords = [t[3] for t in tiles]
        if args.workload == "gemm":
            M, K, Nn = host["dims"]
            C = G.gemm_reduce(rows, coords, M, K, Nn, N, N)
            ref = host["A"].astype(np.int8).astype(np.int64) @ host["B"].astype(np.int8).astype(np.int64)
        else:
            P, Q, _ = host["geom"]
            Cin, H, Wd, M = host["dims"]
            C = G.conv_reduce(rows, coords, P, Q, M, N, N)
            ref = G.conv_reference(host["x"].astype(np.int8), host["w"].astype(np.int8), 0, 0, 1)
        result["host_check"] = "pass" if np.array_equal(C, ref) else "fail"
        result["host_bad"] = int((C != ref).sum())
        result["host_out_range"] = [int(ref.min()), int(ref.max())]
    elif args.resident:
        result["host_check"] = None  # every tile through tile 0's weights: not the workload's result
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=int)
    brief = {k: result.get(k) for k in ("status", "cosim", "host_check", "host_bad", "NT", "total_cycles", "first_out_cycles", "first_tile_done", "last_tile_done", "interval", "timing")}
    print(json.dumps(brief, default=int))


if __name__ == "__main__":
    main()

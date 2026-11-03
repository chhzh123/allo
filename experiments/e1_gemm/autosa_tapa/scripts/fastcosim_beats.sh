#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Stand-in for tapa-fast-cosim (host flag --xosim_executable): set the simulation up,
# add a beat counter to the testbench, run it, and return so the host checks C itself.
# The counter: cycles from the first memory beat (A or B read, C write) to the last,
# sampled at the rising edge on the values before it -- E1's rule
# (experiments/e1_gemm/scripts/beats.py).
set -u
echo "fastcosim_beats.sh args: $*" >&2
OUT=""; prev=""
for a in "$@"; do
  [ "$prev" = --tb_output_dir ] && OUT=$a
  case "$a" in --tb_output_dir=*) OUT=${a#--tb_output_dir=};; esac
  prev=$a
done
[ -n "$OUT" ] || { echo "no --tb_output_dir" >&2; exit 1; }
args=(); for a in "$@"; do [ "$a" = --launch_simulation ] || args+=("$a"); done
tapa-fast-cosim "${args[@]}" --setup_only || exit 1
[ -f "$OUT/tb.sv" ] || { echo "no $OUT/tb.sv" >&2; exit 1; }
python3 - $OUT/tb.sv <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
mon = """
  // beat counter (added): first to last memory beat, E1's definition
  integer bc_cyc = 0, bc_first = -1, bc_last = -1;
  integer bc_a = 0, bc_b = 0, bc_c = 0, bc_a0 = -1, bc_b0 = -1, bc_c0 = -1, bc_ar = -1;
  integer bc_aw = 0, bc_bresp = 0, bc_wv = 0;
  always @(posedge ap_clk) begin
    bc_cyc <= bc_cyc + 1;
    if ((axi_A_rvalid && axi_A_rready) || (axi_B_rvalid && axi_B_rready) ||
        (axi_C_wvalid && axi_C_wready)) begin
      if (bc_first < 0) bc_first <= bc_cyc;
      bc_last <= bc_cyc;
    end
    if (axi_A_arvalid && axi_A_arready && bc_ar < 0) bc_ar <= bc_cyc;
    if (axi_A_rvalid && axi_A_rready) begin bc_a <= bc_a + 1; if (bc_a0 < 0) bc_a0 <= bc_cyc; end
    if (axi_B_rvalid && axi_B_rready) begin bc_b <= bc_b + 1; if (bc_b0 < 0) bc_b0 <= bc_cyc; end
    if (axi_C_wvalid && axi_C_wready) begin bc_c <= bc_c + 1; if (bc_c0 < 0) bc_c0 <= bc_cyc; end
    if (axi_C_awvalid && axi_C_awready) bc_aw <= bc_aw + 1;
    if (axi_C_bvalid && axi_C_bready) bc_bresp <= bc_bresp + 1;
    if (axi_C_wvalid) bc_wv <= bc_wv + 1;
  end
  final $display("BEATS first %0d last %0d cycles %0d | A %0d from %0d (AR %0d) B %0d from %0d C %0d from %0d | C AW %0d B %0d WVALID-cycles %0d",
                 bc_first, bc_last, bc_last - bc_first + 1, bc_a, bc_a0, bc_ar, bc_b, bc_b0, bc_c, bc_c0, bc_aw, bc_bresp, bc_wv);
"""
i = s.rindex("endmodule")
open(p, "w").write(s[:i] + mon + s[i:])
PY
cd $OUT/run && vivado -mode batch -source run_cosim.tcl > ../xsim_beats.log 2>&1
grep -o "BEATS.*" ../xsim_beats.log

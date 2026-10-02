#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# hier_mem.sh <build dir>: the routed whole engine split by what each part is
# for. `--pnr` routes spmw_harness, whose LFSRs are not the design: `dut` is.
# Writes util_hier.rpt, worst_paths.rpt and area_by_unit.txt in the build.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
B=$1
[ -f "$B/routed.dcp" ] || { echo "HIER NO_DCP $B"; exit 1; }
cat > "$B/hier.tcl" <<TCL
open_checkpoint $B/routed.dcp
report_utilization -hierarchical -hierarchical_depth 3 -file $B/util_hier.rpt
report_timing -max_paths 1 -nworst 1 -file $B/worst_paths.rpt
foreach scope {u_req u_deal u_head u_wreq u_pack u_lane u_etap u_ctap u_tap u_uq u_pe} {
  set cells [get_cells -quiet dut/\${scope}*]
  if {[llength \$cells] == 0} { continue }
  set p [get_timing_paths -quiet -max_paths 1 -through [get_pins -quiet -of \$cells -filter {DIRECTION == OUT}]]
  if {[llength \$p]} {
    puts "SCOPE \$scope slack [get_property SLACK \$p] levels [get_property LOGIC_LEVELS \$p] from [get_property STARTPOINT_PIN \$p] to [get_property ENDPOINT_PIN \$p]"
  }
  set q [get_timing_paths -quiet -max_paths 1 -from [get_cells -quiet -hierarchical -filter "NAME =~ dut/\${scope}* && IS_SEQUENTIAL"] -to [get_cells -quiet -hierarchical -filter "NAME =~ dut/\${scope}* && IS_SEQUENTIAL"]]
  if {[llength \$q]} {
    puts "INSIDE \$scope slack [get_property SLACK \$q] levels [get_property LOGIC_LEVELS \$q] from [get_property STARTPOINT_PIN \$q] to [get_property ENDPOINT_PIN \$q]"
  }
}
puts "HIER OK"
TCL
( cd "$B" && vivado -mode batch -source hier.tcl -nojournal -nolog > hier.log 2>&1 )
python3 - "$B" <<'PY' | tee "$B/area_by_unit.txt"
import re, sys, collections
b = sys.argv[1]
rows = []
for line in open(b + "/util_hier.rpt", errors="replace"):
    cells = [c.strip() for c in line.split("|")]
    if len(cells) < 11 or not cells[3].isdigit():
        continue
    depth = (len(line.split("|")[1]) - len(line.split("|")[1].lstrip())) // 2
    rows.append((depth, cells[1], cells[2]) + tuple(int(c) for c in cells[3:11]))
# columns: total LUT, logic LUT, LUTRAM, SRL, FF, RAMB36, RAMB18, URAM
dut = next(r for r in rows if r[1] == "dut")
print(f"HIER {b.split('/')[-1]} dut LUT {dut[3]} FF {dut[7]} RAMB36 {dut[8]} RAMB18 {dut[9]} LUTRAM {dut[5]}")
GROUPS = (
    ("cells", r"u_pe_"),
    ("links between cells", r"g_pe_(a_out_a_in|w_out_w_in|p_out_p_in)"),
    ("edge taps and their links", r"u_etap_|g_etap_|g_pe_(a_in|w_in)_bind"),
    ("head", r"u_head"),
    ("requester", r"u_req|g_req\d+_ins_in"),
    ("dealer", r"u_deal|g_deal\d+_tag_in"),
    ("operand queues to the head", r"g_head\d+_(a_in|w_in|b_in|op_in)_bind"),
    ("micro-op queue, taps and links", r"u_uq|u_tap_|g_uq_|g_tap_|g_lane_c_in"),
    ("lanes, with their accumulators", r"u_lane_|g_lane_z_in"),
    ("result taps and their links", r"u_ctap_|g_ctap_"),
    ("row buffer and its credits", r"g_pack\d+_row_in|g_head\d+_credit"),
    ("packer", r"u_pack"),
    ("write requester", r"u_wreq|g_wreq"),
)
tot = collections.OrderedDict((g, [0] * 6) for g, _ in GROUPS)
other = []
for d, inst, mod, lut, logic, lutram, srl, ff, r36, r18, uram in rows:
    if d != 2:
        continue
    for g, pat in GROUPS:
        if re.match(pat, inst):
            t = tot[g]
            t[0] += 1; t[1] += lut; t[2] += ff; t[3] += r36; t[4] += r18; t[5] += lutram
            break
    else:
        other.append((inst, lut, ff))
for g, (n, lut, ff, r36, r18, lutram) in tot.items():
    print(f"  {g:34s} n={n:4d} LUT={lut:6d} FF={ff:6d} RAMB36={r36:2d} RAMB18={r18:2d} LUTRAM={lutram}")
print("  sum" + " " * 33 + f"LUT={sum(t[1] for t in tot.values()):6d} FF={sum(t[2] for t in tot.values()):6d}")
for inst, lut, ff in other:
    print(f"  ungrouped {inst} LUT={lut} FF={ff}")
PY
grep -E "^(SCOPE|INSIDE)" "$B/hier.log" | tee -a "$B/area_by_unit.txt"

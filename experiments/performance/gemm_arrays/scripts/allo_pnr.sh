#!/bin/bash
# Place and route Allo at one array size, with a watchdog for the Vitis hang.
#
# export_design -flow impl re-runs RTL testbench generation before it hands off
# to Vivado, and that step hangs indefinitely at 100% CPU while writing nothing
# -- the same pathology this experiment already documented for Allo cosim,
# where a plain retry completed in 131 seconds. So: run it, and if the flow log
# goes STALE_S seconds without a write while the process still burns CPU, kill
# that attempt and start it again from the untouched synthesis. Up to TRIES.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
TRIES=${2:-4}
STALE_S=${3:-900}          # 15 minutes with no write to the flow log
SRC=/scratch/hc676/e1_allo_beats/S$S
W=/scratch/hc676/e1_allo_pnr/S$S

command -v vitis_hls >/dev/null || { echo "ALLO_PNR_S${S} ENVIRONMENT vitis_hls not found"; exit 2; }

for try in $(seq 1 "$TRIES"); do
  echo "ALLO_PNR_S${S} ATTEMPT $try START $(date -Is)"
  rm -rf "$W"; mkdir -p "$(dirname "$W")"
  cp -a "$SRC" "$W" || { echo "ALLO_PNR_S${S} COPY_FAILED"; exit 2; }
  cd "$W" || exit 2
  cat > export_impl.tcl <<'TCL'
open_project prj
open_solution sol
set_top gemm
config_export -vivado_report_level 2 -rtl verilog
export_design -flow impl -rtl verilog -format ip_catalog
puts "E1_ALLO_EXPORT_DONE"
exit
TCL
  t0=$(date +%s)
  vitis_hls -f export_impl.tcl > export.log 2>&1 &
  vpid=$!
  FLOW="$W/prj/sol/.autopilot/db/autopilot.flow.log"
  hung=0
  while kill -0 $vpid 2>/dev/null; do
    sleep 60
    # newest write anywhere under the run; the flow log alone can lag
    last=$(find "$W" -type f -newermt "-${STALE_S} seconds" 2>/dev/null | head -1)
    if [ -z "$last" ]; then
      echo "ALLO_PNR_S${S} ATTEMPT $try STALLED after $(( $(date +%s) - t0 ))s, no write in ${STALE_S}s -- killing"
      # kill only processes whose cwd is inside my own run directory
      for p in $(ps -u "$USER" -o pid --no-headers); do
        c=$(readlink /proc/$p/cwd 2>/dev/null)
        case "$c" in "$W"|"$W"/*) kill -9 "$p" 2>/dev/null ;; esac
      done
      hung=1; break
    fi
  done
  wait $vpid 2>/dev/null; rc=$?
  el=$(( $(date +%s) - t0 ))
  [ $hung -eq 1 ] && { echo "ALLO_PNR_S${S} ATTEMPT $try HUNG after ${el}s"; continue; }

  R="$W/prj/sol/impl/report/verilog/export_impl.rpt"
  if [ -f "$R" ]; then
    echo "ALLO_PNR_S${S} DONE attempt=$try rc=$rc wall_s=$el"
    echo "ALLO_PNR_S${S} REPORT $R"
    grep -iE "SLICE|LUT|FF|DSP|BRAM|URAM|SRL|CP required|CP achieved" "$R" | head -25
    echo "ALLO_PNR_S${S} END $(date -Is)"
    exit 0
  fi
  echo "ALLO_PNR_S${S} ATTEMPT $try NO_REPORT rc=$rc wall_s=$el"
  tail -15 "$W/export.log" | sed 's/^/    /'
done
echo "ALLO_PNR_S${S} EXHAUSTED after $TRIES attempts"
echo "ALLO_PNR_S${S} END $(date -Is)"
exit 1

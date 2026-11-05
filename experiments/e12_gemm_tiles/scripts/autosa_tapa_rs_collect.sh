#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_rs_collect.sh: stage the RapidStream runs on E12's AutoSA (TAPA) row
# under export/autosa_tapa_rapidstream/ in this folder's shape. RapidStream's log
# is 3 MB a size, mostly interface inference, so report/rs_floorplan.log keeps its
# area totals and its search over area limits. Its RTL (2,533 files) and
# updated.xo are not kept; autosa_tapa_rs.sh regenerates them from the .xo.
set -u
E=/scratch/hc676/e22_autosa_tapa/e12rs
X=$E/export/autosa_tapa_rapidstream
rm -rf $X; mkdir -p $X/source $X/logs
cp $E/u280_default.json $E/floorplan_config.json $E/pipeline_config.json $X/source/
cp $E/compile_times.txt $E/rs_results.txt $E/pnr_results.txt $E/cosim_results.txt $X/logs/
for W in 64 128 256 512; do
  D=$X/S16_W$W/report; mkdir -p $D
  grep -E 'Total area:|Start DSE|Usage limit increment|percentage limits|Total design area|BRAM_18K: [0-9]+ / [0-9]+|^INFO   SLOT_X|Solution status|Candidate solution exported|ValueError|No solution found' \
    $E/rs/rs_W$W.log | sed "s|$E|<build>|g" > $D/rs_floorplan.log
done
S=$E/rs/W64/dse/solution_0 P=$E/pnr/W64 G=$X/S16_W64/generated D=$X/S16_W64/report
mkdir -p $G
cp $S/xdc/floorplan.xdc $S/metrics/metric_floorplan.json $S/metrics/metric_pipeline.json $G/
cp $P/pnr.tcl $P/timing.rpt $P/util.rpt $P/util_hier.rpt $P/util_pblocks.rpt $P/route_status.rpt $D/
grep -E '^PBLOCK ' $P/pnr.log > $D/pblocks.txt
du -sh $X

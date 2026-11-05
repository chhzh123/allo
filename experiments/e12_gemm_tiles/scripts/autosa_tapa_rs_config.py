# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RapidStream's inputs for E12's AutoSA (TAPA) kernels, written to the current
directory: its whole-chip U280 model, and floorplan and pipeline configs that are
its defaults except that the memory ports are pinned to the bottom row of slots,
where the U280's HBM is. Run as `/opt/rapidstream/rapidstream <this file>`: that
January 2025 install writes the files the runs used (`source/`), byte for byte,
though its floorplanner fails its license check. The July 2025 install that ran
the floorplanner writes a different device model, its column-to-column wires
capped at 40,320 where these leave them uncapped. RapidStream's wrapper changes
into the script's directory, so the files land beside it."""

from rapidstream import (
    FloorplanConfig,
    PipelineConfig,
    get_u280_default_device_factory,
)

get_u280_default_device_factory().generate_virtual_device("u280_default.json")
FloorplanConfig(port_pre_assignments={".*": "SLOT_X0Y0:SLOT_X1Y0"}).save_to_file(
    "floorplan_config.json"
)
PipelineConfig().save_to_file("pipeline_config.json")

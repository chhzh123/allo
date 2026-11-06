# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The steps of a stamped `icpx -v` SYCL compile up to aoc, in seconds:
python3 split_row.py <build.log>. Prints `device_fe host link spirv aoc_start`."""
import re
import sys

starts = []
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"\s*([\d.]+) \s*\"([^\"]+)\"(.*)", line)
    if not m:
        continue
    t, tool, args = float(m.group(1)), m.group(2).rsplit("/", 1)[-1], m.group(3)
    if tool == "clang" and "-fsycl-is-device" in args:
        starts.append(("device_fe", t))
    elif (
        tool == "clang"
        and "-fsycl-is-host" in args
        and not any(k == "host" for k, _ in starts)
    ):
        starts.append(("host", t))
    elif tool == "clang-offload-wrapper" and not any(k == "link" for k, _ in starts):
        starts.append(("link", t))
    elif tool == "llvm-foreach" and "llvm-spirv" in args:
        starts.append(("spirv", t))
    elif tool == "llvm-foreach" and "/aoc" in args:
        starts.append(("aoc_start", t))
        break
when = dict(starts)
order = ["device_fe", "host", "link", "spirv", "aoc_start"]
out = []
for a, b in zip(order, order[1:]):
    out.append(
        "%s=%.1f" % (a, when[b] - when[a]) if a in when and b in when else "%s=" % a
    )
out.append(
    "aoc_start=%.1f" % when["aoc_start"] if "aoc_start" in when else "aoc_start="
)
print(" ".join(out))

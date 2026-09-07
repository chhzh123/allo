# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Fixture for scripts/spmw_loc2.py: multi-line statements count per physical line.
import os  # KEEP

value = [  # KEEP
    1,  # KEEP
    # a comment inside a bracket does not count
    2,  # KEEP
]  # KEEP

total = (1 +  # KEEP
         2 +  # KEEP
         3)  # KEEP

text = "one line KEEP" \
    " continued"  # KEEP

result = os.path.join(  # KEEP
    "a",  # KEEP
    "b",  # KEEP
)  # KEEP

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Fixture for scripts/spmw_loc2.py: every line that must count carries the marker.
# A comment-only line does not count, whatever its indentation.
x = 1  # KEEP: code with a trailing comment counts

y = "# not a comment"  # KEEP: a hash inside a string literal is code
z = '''# not a comment either'''  # KEEP: a triple-quoted value counts
    # an indented comment-only line does not count
w = (x, y, z)  # KEEP

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix every line read on stdin with the seconds since this started."""
import sys
import time

t0 = time.time()
for line in sys.stdin:
    sys.stdout.write("%9.1f %s" % (time.time() - t0, line))
    sys.stdout.flush()

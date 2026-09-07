// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Fixture for scripts/spmw_loc2.py: every line that must count carries the marker.
/* A block comment
   spanning lines
   does not count. */
#include <cstdio> // KEEP: a directive with a trailing comment counts

/* start of a block */ int a = 1; // KEEP: code after a block comment counts
int b = 2; /* KEEP: a block comment opened after code
   continues onto the next line
   and ends here */ int c = 3; // KEEP: code after the closing marker counts
int d = /* inline */ 4; // KEEP

/*
 * A banner comment.
 */
static int add(int x, int y) { // KEEP
  return x + y; // KEEP
} // KEEP

#define KEEP_MACRO(x) \
  ((x) + 1) /* KEEP: the continuation of a directive is code */

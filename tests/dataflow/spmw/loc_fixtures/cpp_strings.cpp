// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Fixture for scripts/spmw_loc2.py: comment markers inside literals are code.
#include <string> // KEEP
const char *url = "http://example.com/not/a/comment"; // KEEP: "//" inside a string
const char *block = "/* not a comment */"; // KEEP
const char *escaped = "a \"quoted\" // still a string"; // KEEP
char slash = '/'; // KEEP
char star = '*'; // KEEP
char quote = '"'; // KEEP: a double quote as a character literal
char apos = '\''; // KEEP: an escaped apostrophe
const char *raw = R"(raw // string /* with */ markers)"; // KEEP
const char *multi /* KEEP */ = R"delim(
// KEEP: inside a raw string this line is code, not a comment
)delim"; // KEEP
/* "a string inside a comment" is still a comment */
// 'a character inside a comment' is still a comment
int end = 0; // KEEP

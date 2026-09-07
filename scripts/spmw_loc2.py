#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Language-aware source counter for the HLS / SPMW size comparison.

The rule is cloc's: a physical line counts when it carries at least one token
of code -- it is neither blank, nor wholly a comment, nor part of a docstring.
A line of code with a trailing comment counts.  What differs from
``scripts/spmw_loc.py`` is that the rule is applied per language, and only to
what a manifest names:

* Python (``.py``) goes through ``tokenize`` and ``ast``.  Comment-only lines,
  docstrings and any string literal that is a statement on its own are
  excluded, every physical line of them; a string used as a value counts.
* C/C++ (``.c .cc .cpp .cxx .h .hpp .hh``) goes through a comment- and
  string-aware stripper: ``//`` and ``/* */`` comments are removed, string,
  character and raw-string literals are preserved, so a comment marker inside
  a literal is code and a quote inside a comment is not a string.
* Verilog/SystemVerilog (``.v .sv .vh .svh``) uses the same stripper without
  character literals, since a lone ``'`` is a size/base marker there.
* TeX (``.tex``): ``minted`` and ``lstlisting`` environments are extracted
  first and counted as the language the environment names.

The unit of selection is a symbol (function, class, ``#define``, module, ...)
or a line range inside a file, never the file by default, and every part of a
manifest is one category: ``design``, ``config``, ``workload``, ``generated``
or ``test``.  Lines of a touched file that no part selects are reported as a
``remainder`` (test/reference/host code by construction) so that the
categories always add up to the whole-file count.  A symbol that is not found,
or is found twice, is an error, not a zero.

    python3 scripts/spmw_loc2.py MANIFEST [--json OUT] [--markdown OUT]
                                          [--included OUT] [--root name=path]
    python3 scripts/spmw_loc2.py --symbols FILE      # what can be selected
    python3 scripts/spmw_loc2.py --count FILE...     # plain per-file counts
    python3 scripts/spmw_loc2.py --version           # sha256 of this script

This script imports nothing outside the standard library.
"""

import argparse
import ast
import bisect
import difflib
import hashlib
import io
import json
import os
import re
import sys
import tokenize

COUNTER_VERSION = "spmw_loc2 2026-09-06"

LANGUAGES = {
    ".py": "python",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".v": "verilog",
    ".vh": "verilog",
    ".sv": "systemverilog",
    ".svh": "systemverilog",
    ".tex": "tex",
}
C_FAMILY = ("c", "cpp")
HDL = ("verilog", "systemverilog")
CATEGORIES = ("design", "config", "workload", "generated", "test")
LISTING_LANGUAGES = {
    "python": "python",
    "py": "python",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "verilog": "verilog",
    "systemverilog": "systemverilog",
    "sv": "systemverilog",
}


class LocError(Exception):
    """A manifest or a source that cannot be counted as asked."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def language_of(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in LANGUAGES:
        raise LocError(f"{path}: no counter for extension {ext!r}")
    return LANGUAGES[ext]


# -- C family: comments and literals -----------------------------------------

_IDENT = re.compile(r"\w")


def strip_c_family(text, char_literals=True, blank_strings=False):
    """Blank comments out of C-family text, keeping every line in place.

    String and character literals are kept (or, with ``blank_strings``, their
    interior is blanked while the quotes stay), so ``"//"`` is not a comment
    and ``'"'`` does not open a string.  Raw strings ``R"delim(...)delim"`` are
    honoured.  A ``//`` comment continued by a trailing backslash is one
    comment, as in C.  Newlines inside comments are preserved.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = i
            while True:
                k = text.find("\n", j)
                if k == -1:
                    k = n
                    break
                # a backslash before the newline continues the comment
                if text[j:k].rstrip("\r").endswith("\\"):
                    j = k + 1
                    continue
                break
            out.append(_blank_keep_newlines(text[i:k]))
            i = k
        elif c == "/" and nxt == "*":
            k = text.find("*/", i + 2)
            k = n if k == -1 else k + 2
            out.append(_blank_keep_newlines(text[i:k]))
            i = k
        elif c == '"':
            k, raw = _string_end(text, i)
            out.append(_literal(text[i:k], blank_strings, raw))
            i = k
        elif c == "'" and char_literals:
            k = _char_end(text, i)
            out.append(_literal(text[i:k], blank_strings, False))
            i = k
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _blank_keep_newlines(segment):
    return "".join("\n" if ch == "\n" else " " for ch in segment)


def _literal(segment, blank, raw):
    if not blank:
        return segment
    if raw:
        # keep the delimiters' shape so line structure survives
        return segment[0] + _blank_keep_newlines(segment[1:-1]) + segment[-1]
    return segment[0] + _blank_keep_newlines(segment[1:-1]) + segment[-1]


def _is_raw_string(text, i):
    """Is the ``"`` at ``i`` the opening quote of a raw string literal?"""
    if i == 0 or text[i - 1] != "R":
        return False
    j = i - 2
    # optional encoding prefix u8 / u / U / L before the R
    if j >= 0 and text[j] in "uUL":
        j -= 1
        if j >= 0 and text[j] == "u" and text[j + 1] == "8":
            j -= 1
    return j < 0 or not _IDENT.match(text[j])


def _string_end(text, i):
    """Index one past the closing quote of the literal opening at ``i``."""
    n = len(text)
    if _is_raw_string(text, i):
        paren = text.find("(", i)
        if paren == -1:
            raise LocError("unterminated raw string literal")
        delim = text[i + 1 : paren]
        close = text.find(")" + delim + '"', paren)
        if close == -1:
            raise LocError("unterminated raw string literal")
        return close + len(delim) + 2, True
    j = i + 1
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == '"':
            return j + 1, False
        if text[j] == "\n":
            # an unterminated string ends at the line, as a compiler would say
            return j, False
        j += 1
    return n, False


def _char_end(text, i):
    n = len(text)
    j = i + 1
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == "'":
            return j + 1
        if text[j] == "\n":
            return j
        j += 1
    return n


def c_family_code_lines(text, char_literals=True):
    stripped = strip_c_family(text, char_literals=char_literals)
    return frozenset(
        number for number, line in enumerate(stripped.split("\n"), 1) if line.strip()
    )


# -- C family: symbols --------------------------------------------------------

_CONTROL = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "decltype",
    "alignas",
    "alignof",
    "static_assert",
}


class _LineIndex:
    def __init__(self, text):
        self.starts = [0] + [m.end() for m in re.finditer("\n", text)]

    def line(self, index):
        return bisect.bisect_right(self.starts, index)


def c_symbols(text, char_literals=True):
    """Top-level definitions of a C-family text: name -> [(start, end), ...].

    Registered: preprocessor directives (``#define NAME`` also as ``NAME``,
    ``#include <x>`` as ``#include <x>``), ``typedef``/``using``/variable
    declarations, ``struct``/``class``/``union``/``enum`` definitions,
    function definitions, and ``extern "C"``/``namespace`` blocks (which are
    also looked into, transparently).  Prototypes are not registered.
    """
    stripped = strip_c_family(text, char_literals=char_literals, blank_strings=True)
    lines = stripped.split("\n")
    found = {}

    def add(name, start, end):
        found.setdefault(name, []).append((start, end))

    directive = set()
    i = 0
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith("#"):
            start = i
            while lines[i].rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
            words = s[1:].split()
            if words:
                keyword = words[0]
                name = "#" + keyword
                if keyword in ("define", "undef", "ifdef", "ifndef") and len(words) > 1:
                    macro = re.match(r"\w+", words[1])
                    if macro:
                        name += " " + macro.group(0)
                        if keyword == "define":
                            add(macro.group(0), start + 1, i + 1)
                elif keyword == "include" and len(words) > 1:
                    name += " " + words[1]
                elif keyword == "pragma" and len(words) > 1:
                    name += " " + " ".join(words[1:3])
                add(name, start + 1, i + 1)
            directive.update(range(start, i + 1))
        i += 1

    body = "\n".join("" if k in directive else l for k, l in enumerate(lines))
    index = _LineIndex(body)
    _scan_range(body, 0, len(body), index, add)
    return found


def _scan_range(body, start, stop, index, add):
    i = start
    chunk = None
    while i < stop:
        c = body[i]
        if c.isspace():
            i += 1
            continue
        if chunk is None:
            chunk = i
        if c == "{":
            head = body[chunk:i]
            close = _match_brace(body, i, stop)
            kind, name = _classify_head(head)
            if kind == "transparent":
                add(name, index.line(chunk), index.line(close))
                _scan_range(body, i + 1, close, index, add)
                end = close
            elif kind == "aggregate":
                end = close
                after = _next_nonspace(body, close + 1, stop)
                if after is not None and body[after] == ";":
                    end = after
                add(name, index.line(chunk), index.line(end))
            elif kind == "function":
                add(name, index.line(chunk), index.line(close))
                end = close
            else:  # a declaration with a braced initialiser: runs to its ';'
                end = _find_top_level(body, close + 1, stop, ";")
                if end is None:
                    end = close
                name = _decl_name(head)
                if name:
                    add(name, index.line(chunk), index.line(end))
            i = end + 1
            chunk = None
        elif c == ";":
            head = body[chunk:i]
            name = _decl_name(head)
            if name:
                add(name, index.line(chunk), index.line(i))
            i += 1
            chunk = None
        else:
            i += 1


def _next_nonspace(body, i, stop):
    while i < stop and body[i].isspace():
        i += 1
    return i if i < stop else None


def _find_top_level(body, i, stop, char):
    depth = 0
    while i < stop:
        c = body[i]
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == char and depth == 0:
            return i
        i += 1
    return None


def _match_brace(body, i, stop):
    depth = 0
    j = i
    while j < stop:
        if body[j] == "{":
            depth += 1
        elif body[j] == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise LocError("unbalanced braces")


def _classify_head(head):
    h = " ".join(head.split())
    # the literal's interior was blanked for scanning, so any quoted text here
    # is the linkage specification of an ``extern "C" { ... }`` block
    if re.match(r'^extern\s+"[^"]*"$', h):
        return "transparent", 'extern "C"'
    m = re.match(r"^(?:inline\s+)?namespace\s+(\w+)$", h)
    if m:
        return "transparent", "namespace " + m.group(1)
    m = re.match(
        r"^(?:template\s*<.*>\s*)?(?:typedef\s+)?(struct|class|union|enum)"
        r"(?:\s+class|\s+struct)?\s+(\w+)\s*(?::[^(]*)?$",
        h,
    )
    if m:
        return "aggregate", m.group(2)
    if h.endswith("="):
        return "declaration", None
    for m in re.finditer(r"(\w+)\s*\(", h):
        name = m.group(1)
        if name in _CONTROL or name.startswith("__attribute"):
            continue
        return "function", name
    return "declaration", None


def _decl_name(head):
    h = " ".join(head.split())
    if not h or h.startswith("#"):
        return None
    if "=" in h:
        h = h.split("=", 1)[0]
    elif "(" in h:
        return None  # a prototype
    h = re.sub(r"\[[^\]]*\]", "", h)
    words = re.findall(r"\w+", h)
    if not words:
        return None
    if words[0] in ("struct", "class", "union", "enum") and len(words) == 2:
        return None  # a forward declaration
    return words[-1]


# -- Verilog / SystemVerilog symbols -------------------------------------------

_HDL_BLOCKS = {
    "module": "endmodule",
    "macromodule": "endmodule",
    "package": "endpackage",
    "interface": "endinterface",
    "program": "endprogram",
    "function": "endfunction",
    "task": "endtask",
    "class": "endclass",
    "checker": "endchecker",
}
_HDL_OPEN = re.compile(
    r"^(?:virtual\s+|extern\s+)?(module|macromodule|package|interface|program|"
    r"function|task|class|checker)\b(.*)$"
)


def hdl_symbols(text):
    stripped = strip_c_family(text, char_literals=False, blank_strings=True)
    found = {}
    stack = []

    def add(name, start, end):
        found.setdefault(name, []).append((start, end))

    for number, raw in enumerate(stripped.split("\n"), 1):
        s = raw.strip()
        m = re.match(r"^`define\s+(\w+)", s)
        if m:
            add("`define " + m.group(1), number, number)
            add(m.group(1), number, number)
            continue
        m = _HDL_OPEN.match(s)
        if m and not s.startswith("end"):
            keyword, rest = m.group(1), m.group(2)
            rest = re.sub(r"^\s*(automatic|static)\b", "", rest)
            if keyword in ("function", "task"):
                rest = re.split(r"[(;]", rest, 1)[0]
                words = re.findall(r"\w+", rest)
                name = words[-1] if words else None
            else:
                words = re.findall(r"\w+", rest)
                name = words[0] if words else None
            if name:
                stack.append((keyword, name, number))
            continue
        for keyword, closer in _HDL_BLOCKS.items():
            if re.match(r"^" + closer + r"\b", s):
                for k in range(len(stack) - 1, -1, -1):
                    if _HDL_BLOCKS[stack[k][0]] == closer:
                        _, name, start = stack.pop(k)
                        add(name, start, number)
                        break
                break
    return found


# -- Python -------------------------------------------------------------------

_PY_SKIP = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.ENCODING,
}


def python_code_lines(text):
    """Physical lines carrying code, docstrings and standalone strings excluded."""
    tree = ast.parse(text)
    standalone = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            value = node.value
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, (str, bytes))
            ) or isinstance(value, ast.JoinedStr):
                standalone.append(
                    (
                        (node.lineno, node.col_offset),
                        (node.end_lineno, node.end_col_offset),
                    )
                )
    code = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in _PY_SKIP:
            continue
        if any(begin <= tok.start and tok.end <= end for begin, end in standalone):
            continue
        code.update(range(tok.start[0], tok.end[0] + 1))
    return frozenset(code)


def _target_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for e in target.elts for n in _target_names(e)]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, ast.Attribute):
        return [ast.unparse(target)]
    if isinstance(target, ast.Subscript):
        return [ast.unparse(target.value) + "[]"]
    return []


def _is_main_guard(test):
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


def python_symbols(text):
    """Definitions: name -> [(start, end), ...], nested ones as ``outer.inner``.

    Decorators belong to the definition they decorate.  Assignments register
    every target name (``M, N = 4, 4`` under both ``M`` and ``N``); attribute
    targets keep their dotted spelling (``engine.spmw_parts``).
    """
    tree = ast.parse(text)
    found = {}

    def add(name, start, end):
        found.setdefault(name, []).append((start, end))

    def visit(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                add(prefix + node.name, start, node.end_lineno)
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in _target_names(target):
                        add(prefix + name, node.lineno, node.end_lineno)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                for name in _target_names(node.target):
                    add(prefix + name, node.lineno, node.end_lineno)
            elif isinstance(node, ast.If) and not prefix and _is_main_guard(node.test):
                add("__main__", node.lineno, node.end_lineno)

    visit(tree.body, "")
    return found


def python_code_text(text, code_lines):
    """Line -> the line's code with comments removed and whitespace collapsed."""
    comments = {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            comments[tok.start[0]] = tok.start[1]
    out = {}
    for number, line in enumerate(text.split("\n"), 1):
        if number not in code_lines:
            continue
        if number in comments:
            line = line[: comments[number]]
        out[number] = " ".join(line.split())
    return out


# -- TeX listings -------------------------------------------------------------

_BEGIN = re.compile(r"\\begin\{(minted|lstlisting)\}(\[[^\]]*\])?(?:\{(\w+)\})?")


def tex_listings(text):
    """The listings of a TeX file, in order: dicts with lang, code, lines."""
    lines = text.split("\n")
    listings = []
    i = 0
    while i < len(lines):
        m = _BEGIN.search(lines[i])
        if not m:
            i += 1
            continue
        env, options, lang = m.group(1), m.group(2) or "", m.group(3)
        if env == "lstlisting":
            lm = re.search(r"language\s*=\s*\{?\s*([\w+]+)", options)
            lang = lm.group(1) if lm else None
        closer = "\\end{" + env + "}"
        j = i + 1
        while j < len(lines) and closer not in lines[j]:
            j += 1
        if j >= len(lines):
            raise LocError(f"listing opened on line {i + 1} is never closed")
        code = "\n".join(lines[i + 1 : j])
        language = LISTING_LANGUAGES.get((lang or "").lower())
        listings.append(
            {
                "index": len(listings),
                "environment": env,
                "language": language,
                "declared": lang,
                "first_line": i + 2,  # file line of the listing's first line
                "last_line": j,
                "code": code,
            }
        )
        i = j + 1
    return listings


# -- a counted source ---------------------------------------------------------


class Source:
    """One text to count: a file, or a listing extracted from a TeX file."""

    def __init__(self, text, language, label, origin_line=1):
        self.text = text
        self.language = language
        self.label = label
        self.origin_line = origin_line
        self.lines = text.split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()  # a trailing newline is not a line
        self._code = None
        self._symbols = None
        self._code_text = None

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        return cls(text, language_of(path), path)

    @classmethod
    def listing(cls, path, index):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        listings = tex_listings(text)
        if index >= len(listings):
            raise LocError(f"{path}: has {len(listings)} listing(s), no index {index}")
        entry = listings[index]
        if entry["language"] is None:
            raise LocError(
                f"{path}: listing {index} declares language {entry['declared']!r}, "
                "which has no counter"
            )
        return cls(
            entry["code"],
            entry["language"],
            f"{path}#listing{index}",
            origin_line=entry["first_line"],
        )

    @property
    def total_lines(self):
        return len(self.lines)

    @property
    def code_lines(self):
        if self._code is None:
            if self.language == "python":
                self._code = python_code_lines(self.text)
            elif self.language in C_FAMILY:
                self._code = c_family_code_lines(self.text, char_literals=True)
            elif self.language in HDL:
                self._code = c_family_code_lines(self.text, char_literals=False)
            else:
                raise LocError(f"{self.label}: cannot count language {self.language!r}")
        return self._code

    @property
    def symbols(self):
        if self._symbols is None:
            if self.language == "python":
                self._symbols = python_symbols(self.text)
            elif self.language in C_FAMILY:
                self._symbols = c_symbols(self.text, char_literals=True)
            elif self.language in HDL:
                self._symbols = hdl_symbols(self.text)
            else:
                self._symbols = {}
        return self._symbols

    def symbol_range(self, name):
        """The one (start, end) of ``name``; ``name@line`` picks a duplicate."""
        wanted_line = None
        if "@" in name:
            name, wanted_line = name.rsplit("@", 1)
            wanted_line = int(wanted_line)
        ranges = self.symbols.get(name, [])
        if wanted_line is not None:
            ranges = [r for r in ranges if r[0] == wanted_line]
        if not ranges:
            raise LocError(f"{self.label}: symbol {name!r} not found")
        if len(ranges) > 1:
            where = ", ".join(str(r[0]) for r in ranges)
            raise LocError(
                f"{self.label}: symbol {name!r} is defined more than once "
                f"(lines {where}); select it as {name}@LINE"
            )
        return ranges[0]

    def code_text(self):
        """Normalised code per counted line, for diffs between designs."""
        if self._code_text is None:
            if self.language == "python":
                self._code_text = python_code_text(self.text, self.code_lines)
            else:
                stripped = strip_c_family(
                    self.text, char_literals=self.language in C_FAMILY
                ).split("\n")
                self._code_text = {
                    n: " ".join(stripped[n - 1].split()) for n in self.code_lines
                }
        return self._code_text

    def text_of(self, numbers):
        return "\n".join(self.lines[n - 1] for n in sorted(numbers))


# -- selection ------------------------------------------------------------------


def _ranges(numbers):
    """Sorted line numbers -> [[start, end], ...]."""
    out = []
    for n in sorted(numbers):
        if out and out[-1][1] == n - 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def select(source, part):
    """Resolve one manifest part against a source.

    Selectors (unioned): ``all``, ``symbols`` (names, ``name@line`` for a
    duplicate, ``outer.inner`` for nested Python definitions),
    ``symbol_pattern`` (regexes over top-level symbol names), ``lines``
    (``[start, end]`` pairs) and ``match`` (regexes over raw lines).  Then
    ``exclude_lines`` removes ranges that must lie inside the selection.
    """
    lines = set()
    resolved = {"symbols": {}, "symbol_pattern": {}, "lines": [], "match": {}}
    total = source.total_lines
    if part.get("all"):
        lines.update(range(1, total + 1))
        resolved["all"] = True
    for name in part.get("symbols", []):
        start, end = source.symbol_range(name)
        resolved["symbols"][name] = [start, end]
        lines.update(range(start, end + 1))
    for pattern in part.get("symbol_pattern", []):
        hits = {}
        for name, ranges in source.symbols.items():
            nested = source.language == "python" and "." in name
            if nested or not re.search(pattern, name):
                continue
            if len(ranges) != 1:
                raise LocError(
                    f"{source.label}: pattern {pattern!r} hits {name!r}, which is "
                    "defined more than once"
                )
            hits[name] = list(ranges[0])
            lines.update(range(ranges[0][0], ranges[0][1] + 1))
        if not hits:
            raise LocError(
                f"{source.label}: symbol pattern {pattern!r} matches nothing"
            )
        resolved["symbol_pattern"][pattern] = hits
    for start, end in part.get("lines", []):
        if not 1 <= start <= end <= total:
            raise LocError(
                f"{source.label}: line range {start}-{end} is outside 1-{total}"
            )
        resolved["lines"].append([start, end])
        lines.update(range(start, end + 1))
    for pattern in part.get("match", []):
        hits = [n for n, line in enumerate(source.lines, 1) if re.search(pattern, line)]
        if not hits:
            raise LocError(f"{source.label}: match {pattern!r} matches no line")
        resolved["match"][pattern] = _ranges(hits)
        lines.update(hits)
    excluded = set()
    for start, end in part.get("exclude_lines", []):
        span = set(range(start, end + 1))
        if not span <= lines:
            raise LocError(
                f"{source.label}: exclude_lines {start}-{end} is not inside the selection"
            )
        excluded |= span
    lines -= excluded
    if excluded:
        resolved["exclude_lines"] = _ranges(excluded)
    return lines, resolved


# -- the manifest -------------------------------------------------------------


def resolve_path(spec, roots):
    """``root:relative/path`` -> absolute path, or an absolute path as is."""
    if ":" in spec and not os.path.isabs(spec):
        root, rel = spec.split(":", 1)
        if root not in roots:
            raise LocError(f"path {spec!r} names unknown root {root!r}")
        return os.path.normpath(os.path.join(roots[root], rel))
    if not os.path.isabs(spec):
        raise LocError(f"path {spec!r} is neither root-relative nor absolute")
    return os.path.normpath(spec)


class Counter:
    def __init__(self, manifest, roots):
        self.manifest = manifest
        self.roots = roots
        self._sources = {}
        self._file_hashes = {}

    def source(self, part):
        spec = part["file"]
        path = resolve_path(spec, self.roots)
        if path not in self._file_hashes:
            with open(path, "rb") as handle:
                self._file_hashes[path] = sha256_bytes(handle.read())
        key = (path, part.get("listing"))
        if key not in self._sources:
            if part.get("listing") is not None:
                self._sources[key] = Source.listing(path, int(part["listing"]))
            else:
                self._sources[key] = Source.from_file(path)
        return path, self._sources[key]

    def count_side(self, side, side_name, design_id):
        parts_out = []
        per_file = {}  # (path, listing) -> {"source", "selected": set, "path"}
        for number, part in enumerate(side.get("parts", [])):
            category = part.get("category")
            if category not in CATEGORIES:
                raise LocError(
                    f"{design_id}/{side_name} part {number}: category {category!r} "
                    f"is not one of {CATEGORIES}"
                )
            path, source = self.source(part)
            lines, resolved = select(source, part)
            key = (path, part.get("listing"))
            entry = per_file.setdefault(
                key,
                {"source": source, "selected": set(), "path": path, "shared": False},
            )
            # a ``shared`` part borrows from a file whose other symbols belong
            # to another design: the file's remainder is not this side's
            entry["shared"] = entry["shared"] or bool(part.get("shared"))
            overlap = entry["selected"] & lines
            if overlap:
                raise LocError(
                    f"{design_id}/{side_name} part {number} ({source.label}): lines "
                    f"{_ranges(overlap)} are already selected by an earlier part"
                )
            entry["selected"] |= lines
            counted = lines & source.code_lines
            parts_out.append(
                {
                    "part": number,
                    "label": part.get("label", ""),
                    "category": category,
                    "file": path,
                    "file_spec": part["file"],
                    "listing": part.get("listing"),
                    "language": source.language,
                    "selectors": resolved,
                    "ranges": _ranges(lines),
                    "selected_lines": len(lines),
                    "count": len(counted),
                    "sha256_selected": sha256_text(source.text_of(lines)),
                    "sha256_file": self._file_hashes[path],
                    "note": part.get("note", ""),
                }
            )
        remainder_category = side.get("remainder_category", "test")
        remainders = []
        for key, entry in per_file.items():
            source = entry["source"]
            if entry["shared"]:
                remainders.append(
                    {
                        "file": entry["path"],
                        "listing": key[1],
                        "category": remainder_category,
                        "count": 0,
                        "ranges": [],
                        "sha256_selected": "",
                        "shared": True,
                        "whole_file_count": len(source.code_lines),
                        "whole_file_lines": source.total_lines,
                    }
                )
                continue
            rest = set(range(1, source.total_lines + 1)) - entry["selected"]
            counted = rest & source.code_lines
            whole = len(source.code_lines)
            assigned = sum(
                p["count"] for p in parts_out if (p["file"], p["listing"]) == key
            )
            if assigned + len(counted) != whole:
                raise LocError(
                    f"{source.label}: parts count {assigned} + remainder "
                    f"{len(counted)} != whole file {whole}"
                )
            remainders.append(
                {
                    "file": entry["path"],
                    "listing": key[1],
                    "category": remainder_category,
                    "count": len(counted),
                    "ranges": _ranges(rest),
                    "sha256_selected": (
                        sha256_text(source.text_of(rest)) if rest else ""
                    ),
                    "whole_file_count": whole,
                    "whole_file_lines": source.total_lines,
                }
            )
        totals = {c: 0 for c in CATEGORIES}
        for part in parts_out:
            totals[part["category"]] += part["count"]
        totals["remainder"] = sum(r["count"] for r in remainders)
        totals["remainder_category"] = remainder_category
        totals["design_plus_config"] = totals["design"] + totals["config"]
        return {
            "label": side.get("label", side_name),
            "status": side.get("status", "counted"),
            "parts": parts_out,
            "remainder": remainders,
            "totals": totals,
            "note": side.get("note", ""),
        }

    def run(self):
        designs_out = []
        for design in self.manifest["designs"]:
            sides = {}
            for side_name, side in design.get("sides", {}).items():
                if side.get("status") == "missing":
                    sides[side_name] = {
                        "label": side.get("label", side_name),
                        "status": "missing",
                        "reason": side.get("reason", ""),
                        "parts": [],
                        "remainder": [],
                        "totals": None,
                    }
                else:
                    sides[side_name] = self.count_side(side, side_name, design["id"])
            out = {
                "id": design["id"],
                "title": design.get("title", design["id"]),
                "kind": design.get("kind", "design"),
                "table": design.get("table", True),
                "archived": design.get("archived"),
                "notes": design.get("notes", []),
                "sides": sides,
            }
            if "adaptation_of" in design:
                out["adaptation_of"] = design["adaptation_of"]
            designs_out.append(out)
        by_id = {d["id"]: d for d in designs_out}
        for design in designs_out:
            base_id = design.get("adaptation_of")
            if base_id:
                if base_id not in by_id:
                    raise LocError(f"{design['id']}: adaptation_of {base_id!r} unknown")
                design["adaptation"] = self.adaptation(by_id[base_id], design)
        return designs_out

    def _design_code_text(self, design_out, side_name):
        side = design_out["sides"].get(side_name)
        if not side or side["status"] == "missing":
            return None
        texts = []
        for part in side["parts"]:
            if part["category"] != "design":
                continue
            source = self._sources[(part["file"], part["listing"])]
            code = source.code_text()
            for start, end in part["ranges"]:
                for n in range(start, end + 1):
                    if n in code:
                        texts.append(code[n])
        return texts

    def adaptation(self, base, new):
        """Design lines of ``new`` that differ from ``base``, per side (difflib)."""
        out = {}
        for side_name in new["sides"]:
            a = self._design_code_text(base, side_name)
            b = self._design_code_text(new, side_name)
            if a is None or b is None:
                continue
            matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            added = removed = 0
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                removed += i2 - i1
                added += j2 - j1
            out[side_name] = {
                "base": base["id"],
                "base_design_lines": len(a),
                "adapted_design_lines": len(b),
                "unchanged": len(b) - added,
                "added": added,
                "removed": removed,
            }
        return out


# -- reports --------------------------------------------------------------------


def ratio(a, b):
    if a is None or b is None or not b:
        return None
    return round(a / b, 2)


def summary_rows(designs):
    rows = []
    for d in designs:
        hls = d["sides"].get("hls")
        spmw = d["sides"].get("spmw")
        row = {
            "id": d["id"],
            "title": d["title"],
            "kind": d["kind"],
            "table": d["table"],
            "hls_status": hls["status"] if hls else "absent",
            "spmw_status": spmw["status"] if spmw else "absent",
        }
        for name, side in (("hls", hls), ("spmw", spmw)):
            totals = side["totals"] if side and side["totals"] else None
            row[name + "_design"] = totals["design"] if totals else None
            row[name + "_config"] = totals["config"] if totals else None
            row[name + "_workload"] = totals["workload"] if totals else None
            row[name + "_generated"] = totals["generated"] if totals else None
            row[name + "_test"] = (
                (totals["test"] + totals["remainder"]) if totals else None
            )
        row["ratio_design"] = ratio(row["hls_design"], row["spmw_design"])
        row["ratio_design_plus_config"] = ratio(
            (
                None
                if row["hls_design"] is None
                else row["hls_design"] + row["hls_config"]
            ),
            (
                None
                if row["spmw_design"] is None
                else row["spmw_design"] + row["spmw_config"]
            ),
        )
        archived = d.get("archived") or {}
        row["archived_hls"] = archived.get("hls")
        row["archived_spmw"] = archived.get("spmw")
        row["archived_ratio"] = ratio(archived.get("hls"), archived.get("spmw"))
        row["archived_note"] = archived.get("note", "")
        row["adaptation"] = d.get("adaptation")
        rows.append(row)
    return rows


def _cell(value):
    return "--" if value is None else str(value)


def _ratio_cell(value):
    return "--" if value is None else f"{value:.2f}x"


def markdown_report(designs, meta):
    rows = summary_rows(designs)
    out = []
    out.append("# Language-aware source recount (E7)\n")
    out.append(
        f"Counter: `{meta['counter_path']}` (sha256 `{meta['counter_sha256']}`), "
        f"{meta['counter_version']}. Manifest sha256 `{meta['manifest_sha256']}`.\n"
    )
    out.append(
        "Rule: cloc's -- a physical line counts when it holds code; comment-only "
        "lines, docstrings and standalone string statements do not; code with a "
        "trailing comment does. Python via `tokenize`+`ast`, C/C++ and "
        "SystemVerilog via a comment- and string-aware stripper, TeX listings "
        "extracted first. `design` = PE/unit bodies, interfaces, topology/link "
        "rules, loaders/drains/boundary bindings and design-specific interface "
        "code; `config` = imports/includes, constants, type aliases and sizes; "
        "`workload` = programs and their encoders; `test` = tests, references, "
        "operand generators and host code (reported, not counted). The headline "
        "columns are `design` only. Source size is not development time.\n"
    )
    if meta.get("provenance"):
        out.append("## Provenance\n")
        for item in meta["provenance"]:
            out.append(f"- {item}")
        out.append("")
    out.append("## Design-only counts, with the archived provisional numbers\n")
    out.append(
        "| Design | HLS | SPMW | HLS/SPMW | archived HLS | archived SPMW | archived ratio | note |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if not r["table"] or r["kind"] != "design":
            continue
        hls = _cell(r["hls_design"]) if r["hls_status"] == "counted" else "--"
        out.append(
            f"| {r['title']} | {hls} | {_cell(r['spmw_design'])} | "
            f"{_ratio_cell(r['ratio_design'])} | {_cell(r['archived_hls'])} | "
            f"{_cell(r['archived_spmw'])} | {_ratio_cell(r['archived_ratio'])} | "
            f"{r['archived_note']} |"
        )
    out.append("")
    out.append("## Breakdown by category (HLS / SPMW)\n")
    out.append(
        "| Design | design | config | workload | generated | test+host (excluded) | design+config | ratio (design+config) |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["kind"] not in ("design", "variant"):
            continue

        def pair(key):
            return f"{_cell(r['hls_' + key])} / {_cell(r['spmw_' + key])}"

        dpc_h = None if r["hls_design"] is None else r["hls_design"] + r["hls_config"]
        dpc_s = (
            None if r["spmw_design"] is None else r["spmw_design"] + r["spmw_config"]
        )
        out.append(
            f"| {r['title']} | {pair('design')} | {pair('config')} | {pair('workload')} | "
            f"{pair('generated')} | {pair('test')} | {_cell(dpc_h)} / {_cell(dpc_s)} | "
            f"{_ratio_cell(r['ratio_design_plus_config'])} |"
        )
    out.append("")
    frag = [r for r in rows if r["kind"] in ("fragment", "generated")]
    if frag:
        out.append(
            "## Listing fragments and generated code (not comparable to the rows above)\n"
        )
        out.append("| Item | HLS side | SPMW side | what it is |")
        out.append("|---|---:|---:|---|")
        for r in frag:
            d = next(x for x in designs if x["id"] == r["id"])
            cells = []
            for name in ("hls", "spmw"):
                side = d["sides"].get(name)
                if not side or side["status"] != "counted":
                    cells.append("--")
                else:
                    t = side["totals"]
                    cells.append(
                        f"{t['design']} design, {t['generated']} generated, "
                        f"{t['test'] + t['remainder']} test/host"
                    )
            out.append(
                f"| {r['title']} | {cells[0]} | {cells[1]} | {' '.join(d['notes'])} |"
            )
        out.append("")
    adapt = [r for r in rows if r["adaptation"]]
    if adapt:
        out.append("## Adaptation: design lines that differ from the original\n")
        out.append(
            "Computed with `difflib` over comment-stripped, whitespace-collapsed "
            "design lines of the two selections; `added`/`removed` are line "
            "counts, not a measure of effort.\n"
        )
        out.append(
            "| Design | side | original | adapted | unchanged | added | removed |"
        )
        out.append("|---|---|---:|---:|---:|---:|---:|")
        for r in adapt:
            for side_name, a in r["adaptation"].items():
                out.append(
                    f"| {r['title']} | {side_name} | {a['base_design_lines']} | "
                    f"{a['adapted_design_lines']} | {a['unchanged']} | {a['added']} | "
                    f"{a['removed']} |"
                )
        out.append("")
    out.append("## Notes per design\n")
    for d in designs:
        out.append(f"### {d['title']} (`{d['id']}`)\n")
        for side_name, side in d["sides"].items():
            if side["status"] == "missing":
                out.append(
                    f"- **{side['label']}**: no counterpart counted. {side['reason']}"
                )
                continue
            t = side["totals"]
            parts = []
            for p in side["parts"]:
                sel = []
                for name in p["selectors"]["symbols"]:
                    sel.append(name)
                for pat in p["selectors"]["symbol_pattern"]:
                    sel.append(f"/{pat}/")
                for start, end in p["selectors"]["lines"]:
                    sel.append(f"L{start}-{end}")
                for pat in p["selectors"]["match"]:
                    sel.append(f"~{pat}~")
                if p["selectors"].get("all"):
                    sel.append("(whole file)")
                excl = ""
                if p["selectors"].get("exclude_lines"):
                    excl = " minus " + ", ".join(
                        f"L{s}-{e}" for s, e in p["selectors"]["exclude_lines"]
                    )
                where = os.path.basename(p["file"])
                if p["listing"] is not None:
                    where += f"#listing{p['listing']}"
                parts.append(
                    f"  - {p['category']} {p['count']}: `{where}` "
                    f"{', '.join(sel)}{excl}"
                    + (f" -- {p['note']}" if p["note"] else "")
                )
            for rem in side["remainder"]:
                if rem["count"]:
                    parts.append(
                        f"  - {rem['category']} (remainder, excluded) {rem['count']}: "
                        f"`{os.path.basename(rem['file'])}` everything not selected above"
                    )
            out.append(
                f"- **{side['label']}**: design {t['design']}, config {t['config']}, "
                f"workload {t['workload']}, generated {t['generated']}, "
                f"test/host {t['test'] + t['remainder']}"
                + (f". {side['note']}" if side["note"] else "")
            )
            out.extend(parts)
        for note in d.get("notes", []):
            out.append(f"- {note}")
        out.append("")
    return "\n".join(out) + "\n"


def included_report(designs, meta):
    out = [
        f"# included ranges; counter {meta['counter_sha256']} manifest {meta['manifest_sha256']}",
        "# file-sha256  file",
        "#   range-sha256  design/side  category  count  ranges  selectors",
    ]
    seen = set()
    for d in designs:
        for side_name, side in d["sides"].items():
            for p in side["parts"] + side["remainder"]:
                path = p["file"]
                if path not in seen:
                    seen.add(path)
                    file_hash = p.get("sha256_file") or next(
                        q["sha256_file"]
                        for dd in designs
                        for ss in dd["sides"].values()
                        for q in ss["parts"]
                        if q["file"] == path
                    )
                    out.append(f"{file_hash}  {path}")
                if "selectors" in p:
                    sel = list(p["selectors"]["symbols"]) + [
                        f"L{s}-{e}" for s, e in p["selectors"]["lines"]
                    ]
                    sel += [f"~{m}~" for m in p["selectors"]["match"]]
                    sel += [f"/{m}/" for m in p["selectors"]["symbol_pattern"]]
                    if p["selectors"].get("all"):
                        sel.append("(all)")
                    if p["selectors"].get("exclude_lines"):
                        sel.append(
                            "minus "
                            + ",".join(
                                f"L{s}-{e}" for s, e in p["selectors"]["exclude_lines"]
                            )
                        )
                    what = p["category"]
                else:
                    sel = ["(remainder)"]
                    what = f"{p['category']}(remainder)"
                listing = "" if p.get("listing") is None else f"#listing{p['listing']}"
                ranges = ",".join(f"{s}-{e}" for s, e in p["ranges"]) or "-"
                out.append(
                    f"  {p['sha256_selected'] or '-':64s}  {d['id']}/{side_name}{listing}  "
                    f"{what}  {p['count']}  {ranges}  {' '.join(sel)}"
                )
    return "\n".join(out) + "\n"


def print_table(designs):
    rows = summary_rows(designs)
    print(
        f"{'design':44s} {'HLS':>6s} {'SPMW':>6s} {'ratio':>7s}   {'arch HLS':>8s} {'arch SPMW':>9s}"
    )
    for r in rows:
        if r["kind"] not in ("design", "variant"):
            continue
        hls = _cell(r["hls_design"]) if r["hls_status"] == "counted" else "--"
        print(
            f"{r['title'][:44]:44s} {hls:>6s} {_cell(r['spmw_design']):>6s} "
            f"{_ratio_cell(r['ratio_design']):>7s}   {_cell(r['archived_hls']):>8s} "
            f"{_cell(r['archived_spmw']):>9s}"
        )
    print()
    for d in designs:
        for side_name, side in d["sides"].items():
            if side["status"] == "missing":
                print(f"{d['id']}/{side_name}: missing -- {side['reason']}")
                continue
            t = side["totals"]
            print(
                f"{d['id']}/{side_name}: design {t['design']} config {t['config']} "
                f"workload {t['workload']} generated {t['generated']} "
                f"test {t['test']} remainder({t['remainder_category']}) {t['remainder']}"
            )
            for p in side["parts"]:
                where = os.path.basename(p["file"])
                if p["listing"] is not None:
                    where += f"#listing{p['listing']}"
                ranges = ",".join(f"{s}-{e}" for s, e in p["ranges"])
                print(
                    f"    {p['category']:9s} {p['count']:5d}  {where}  [{ranges}]  "
                    f"{p['sha256_selected'][:12]}"
                )
            for rem in side["remainder"]:
                if rem["count"]:
                    print(
                        f"    {rem['category'] + '*':9s} {rem['count']:5d}  "
                        f"{os.path.basename(rem['file'])}  (remainder)"
                    )


# -- entry points ---------------------------------------------------------------


def self_sha256():
    with open(os.path.abspath(__file__), "rb") as handle:
        return sha256_bytes(handle.read())


def count_file(path):
    return len(Source.from_file(path).code_lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", nargs="?", help="JSON manifest to count")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--markdown", help="write the summary table here")
    ap.add_argument("--included", help="write the included-ranges list here")
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override a path root declared in the manifest",
    )
    ap.add_argument("--symbols", metavar="FILE", help="list the symbols of one file")
    ap.add_argument("--count", nargs="+", metavar="FILE", help="count whole files")
    ap.add_argument("--version", action="store_true", help="print this script's sha256")
    args = ap.parse_args(argv)

    if args.version:
        print(f"{COUNTER_VERSION}  sha256 {self_sha256()}")
        return 0
    if args.symbols:
        source = Source.from_file(args.symbols)
        for name, ranges in sorted(source.symbols.items(), key=lambda kv: kv[1][0]):
            for start, end in ranges:
                counted = len(set(range(start, end + 1)) & source.code_lines)
                print(f"{start:5d}-{end:<5d} {counted:4d}  {name}")
        print(f"total code lines: {len(source.code_lines)} of {source.total_lines}")
        return 0
    if args.count:
        for path in args.count:
            if path.endswith(".tex"):
                with open(path, encoding="utf-8") as handle:
                    for entry in tex_listings(handle.read()):
                        src = Source(entry["code"], entry["language"], path)
                        print(
                            f"{len(src.code_lines):6d}  {path}#listing{entry['index']} ({entry['language']})"
                        )
            else:
                print(f"{count_file(path):6d}  {path}")
        return 0
    if not args.manifest:
        ap.error("a manifest is required (or --symbols / --count / --version)")

    with open(args.manifest, "rb") as handle:
        raw = handle.read()
    manifest = json.loads(raw.decode("utf-8"))
    roots = dict(manifest.get("roots", {}))
    for item in args.root:
        name, path = item.split("=", 1)
        roots[name] = path
    designs = Counter(manifest, roots).run()
    meta = {
        "counter_path": os.path.abspath(__file__),
        "counter_version": COUNTER_VERSION,
        "counter_sha256": self_sha256(),
        "manifest_path": os.path.abspath(args.manifest),
        "manifest_sha256": sha256_bytes(raw),
        "roots": roots,
        "python": sys.version.split()[0],
        "provenance": manifest.get("provenance", []),
    }
    print_table(designs)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {"meta": meta, "summary": summary_rows(designs), "designs": designs},
                handle,
                indent=2,
            )
            handle.write("\n")
        print(f"wrote {args.json}")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(markdown_report(designs, meta))
        print(f"wrote {args.markdown}")
    if args.included:
        with open(args.included, "w", encoding="utf-8") as handle:
            handle.write(included_report(designs, meta))
        print(f"wrote {args.included}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LocError as error:
        sys.exit(f"spmw_loc2: {error}")

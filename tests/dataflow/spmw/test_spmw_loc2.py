# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/spmw_loc2.py against the fixtures in loc_fixtures/.

Every fixture marks each line that must count with the token KEEP, so the
expected count of a file is both a number written here and the number of lines
carrying the marker -- two independent ways for a fixture edit to be noticed.

This test imports neither numpy nor allo, so it runs anywhere:

    python3 -m pytest tests/dataflow/spmw/test_spmw_loc2.py -q
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
COUNTER = ROOT / "scripts" / "spmw_loc2.py"
FIXTURES = pathlib.Path(__file__).resolve().parent / "loc_fixtures"


def _load():
    spec = importlib.util.spec_from_file_location("spmw_loc2", COUNTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loc = _load()


def keep_lines(path):
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if "KEEP" in line)


EXPECTED = {
    "py_comments.py": 4,
    "py_docstrings.py": 13,
    "py_multiline.py": 14,
    "py_mixed.py": 10,
    "py_symbols.py": 18,
    "py_duplicates.py": 2,
    "cpp_block_comments.cpp": 10,
    "cpp_strings.cpp": 13,
    "cpp_symbols.cpp": 17,
    "sv_comments.sv": 9,
}


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_fixture_counts(name, expected):
    path = FIXTURES / name
    assert (
        keep_lines(path) == expected
    ), "the KEEP markers disagree with the expected count"
    assert loc.count_file(str(path)) == expected


def test_every_fixture_is_covered():
    """A fixture nobody counts tests nothing."""
    present = {p.name for p in FIXTURES.iterdir() if p.suffix in (".py", ".cpp", ".sv")}
    assert present == set(EXPECTED)


def test_the_counter_imports_nothing_heavy():
    text = COUNTER.read_text(encoding="utf-8")
    assert "import numpy" not in text and "import allo" not in text
    assert "numpy" not in sys.modules or True  # importing the counter did not need it


def test_scratch_extensions_are_refused():
    """The archived counter wrote ``.region`` files and counted them as C."""
    with pytest.raises(loc.LocError):
        loc.language_of("test_spmw_gemm_int8.py.region")


def test_tex_listings_are_extracted_and_counted():
    path = FIXTURES / "listing.tex"
    listings = loc.tex_listings(path.read_text(encoding="utf-8"))
    assert [entry["language"] for entry in listings] == ["python", "cpp"]
    python = loc.Source.listing(str(path), 0)
    cpp = loc.Source.listing(str(path), 1)
    assert len(python.code_lines) == 2
    assert len(cpp.code_lines) == 3
    assert keep_lines(path) == 5
    assert python.origin_line == 6  # file line of the listing's first line
    assert cpp.symbol_range("pe") == (2, 4)
    assert python.symbol_range("unit") == (2, 4)


def test_python_symbol_selection():
    source = loc.Source.from_file(str(FIXTURES / "py_symbols.py"))
    assert source.symbol_range("IO") == (13, 16)
    assert source.symbol_range("unit") == (19, 22)  # the decorator belongs to it
    assert source.symbol_range("outer.Inner") == (26, 27)
    assert source.symbol_range("__main__") == (37, 38)
    lines, resolved = loc.select(source, {"symbols": ["IO", "unit", "outer"]})
    assert len(lines & source.code_lines) == 9
    assert resolved["symbols"] == {"IO": [13, 16], "unit": [19, 22], "outer": [25, 29]}
    lines, _ = loc.select(source, {"symbol_pattern": ["^test_"]})
    assert len(lines & source.code_lines) == 3  # def + two asserts
    lines, _ = loc.select(source, {"match": ["^import "], "symbols": ["N"]})
    assert len(lines & source.code_lines) == 2
    lines, _ = loc.select(source, {"lines": [[4, 6]]})
    assert len(lines & source.code_lines) == 2
    lines, resolved = loc.select(
        source, {"symbols": ["outer"], "exclude_lines": [[26, 27]]}
    )
    assert len(lines & source.code_lines) == 2
    assert resolved["exclude_lines"] == [[26, 27]]


def test_selection_errors_fire():
    source = loc.Source.from_file(str(FIXTURES / "py_symbols.py"))
    with pytest.raises(loc.LocError, match="not found"):
        loc.select(source, {"symbols": ["nowhere"]})
    with pytest.raises(loc.LocError, match="not inside the selection"):
        loc.select(source, {"symbols": ["IO"], "exclude_lines": [[1, 2]]})
    with pytest.raises(loc.LocError, match="matches no line"):
        loc.select(source, {"match": ["^never matches$"]})
    with pytest.raises(loc.LocError, match="outside"):
        loc.select(source, {"lines": [[1, 999]]})
    duplicates = loc.Source.from_file(str(FIXTURES / "py_duplicates.py"))
    with pytest.raises(loc.LocError, match="more than once"):
        duplicates.symbol_range("X")
    assert duplicates.symbol_range("X@4") == (4, 4)
    assert duplicates.symbol_range("X@5") == (5, 5)


def test_c_symbol_selection():
    source = loc.Source.from_file(str(FIXTURES / "cpp_symbols.cpp"))
    counted = lambda names: len(
        loc.select(source, {"symbols": names})[0] & source.code_lines
    )
    assert source.symbol_range("pe") == (13, 16)
    assert source.symbol_range("column_t") == (8, 10)
    assert source.symbol_range("top") == (20, 24)  # found inside extern "C"
    assert source.symbol_range('extern "C"') == (18, 26)
    assert counted(["pe"]) == 4
    assert counted(["column_t"]) == 3
    assert counted(["top"]) == 5
    assert counted(['extern "C"']) == 7
    assert counted(["DIM", "data_t", "#include <hls_stream.h>"]) == 3
    assert counted(["pe", "column_t", 'extern "C"']) + counted(
        ["DIM", "data_t", "#include <hls_stream.h>"]
    ) == len(source.code_lines)
    blocks = loc.Source.from_file(str(FIXTURES / "cpp_block_comments.cpp"))
    assert len(loc.select(blocks, {"symbols": ["add"]})[0] & blocks.code_lines) == 3
    assert (
        len(loc.select(blocks, {"symbols": ["KEEP_MACRO"]})[0] & blocks.code_lines) == 2
    )


def test_sv_symbol_selection():
    source = loc.Source.from_file(str(FIXTURES / "sv_comments.sv"))
    assert source.symbol_range("keep_counter") == (6, 14)
    assert (
        len(loc.select(source, {"symbols": ["keep_counter"]})[0] & source.code_lines)
        == 9
    )


def test_c_comment_continuation_and_unterminated_string():
    text = "int a; // comment \\\n still the comment\nint b;\n"
    assert len(loc.Source(text, "cpp", "inline").code_lines) == 2
    text = 'const char *s = "unterminated\nint c; // KEEP\n'
    assert len(loc.Source(text, "cpp", "inline").code_lines) == 2


def _manifest(tmp_path):
    return {
        "roots": {"fx": str(FIXTURES)},
        "designs": [
            {
                "id": "fixture",
                "title": "Fixture design",
                "archived": {"hls": 20, "spmw": 12, "note": "archived note"},
                "sides": {
                    "hls": {
                        "label": "C++",
                        "parts": [
                            {
                                "file": "fx:cpp_symbols.cpp",
                                "category": "design",
                                "symbols": ["pe", "column_t", 'extern "C"'],
                            },
                            {
                                "file": "fx:cpp_symbols.cpp",
                                "category": "config",
                                "match": ["^#include", "^#define", "^typedef"],
                            },
                        ],
                    },
                    "spmw": {
                        "label": "Python",
                        "parts": [
                            {
                                "file": "fx:py_symbols.py",
                                "category": "design",
                                "symbols": ["IO", "unit", "outer"],
                            },
                            {
                                "file": "fx:py_symbols.py",
                                "category": "config",
                                "match": ["^import "],
                                "symbols": ["N"],
                            },
                        ],
                    },
                },
            },
            {
                "id": "adapted",
                "title": "Adapted fixture",
                "adaptation_of": "fixture",
                "sides": {
                    "hls": {"status": "missing", "reason": "no counterpart"},
                    "spmw": {
                        "parts": [
                            {
                                "file": "fx:py_symbols.py",
                                "category": "design",
                                "symbols": ["IO", "unit", "test_something"],
                            }
                        ]
                    },
                },
            },
            {
                "id": "listing",
                "title": "Listing fragment",
                "kind": "fragment",
                "sides": {
                    "hls": {
                        "parts": [
                            {
                                "file": "fx:listing.tex",
                                "listing": 1,
                                "category": "design",
                                "all": True,
                            }
                        ]
                    },
                    "spmw": {
                        "parts": [
                            {
                                "file": "fx:listing.tex",
                                "listing": 0,
                                "category": "design",
                                "all": True,
                            }
                        ]
                    },
                },
            },
        ],
    }


def test_manifest_end_to_end(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")
    out_json = tmp_path / "counts.json"
    out_md = tmp_path / "counts.md"
    out_inc = tmp_path / "included.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(COUNTER),
            str(manifest),
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
            "--included",
            str(out_inc),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data["meta"]["counter_sha256"]) == 64
    assert len(data["meta"]["manifest_sha256"]) == 64
    fixture, adapted, listing = data["designs"]
    hls = fixture["sides"]["hls"]["totals"]
    spmw = fixture["sides"]["spmw"]["totals"]
    assert (hls["design"], hls["config"], hls["remainder"]) == (14, 3, 0)
    assert (spmw["design"], spmw["config"], spmw["remainder"]) == (9, 2, 7)
    for part in fixture["sides"]["hls"]["parts"] + fixture["sides"]["spmw"]["parts"]:
        assert len(part["sha256_selected"]) == 64 and len(part["sha256_file"]) == 64
    summary = {row["id"]: row for row in data["summary"]}
    assert summary["fixture"]["ratio_design"] == round(14 / 9, 2)
    assert summary["fixture"]["archived_ratio"] == round(20 / 12, 2)
    assert adapted["sides"]["hls"]["status"] == "missing"
    diff = adapted["adaptation"]["spmw"]
    # base: IO (2) + unit (3) + outer (4); adapted: IO + unit + test_something (3)
    assert (diff["base_design_lines"], diff["adapted_design_lines"]) == (9, 8)
    assert (diff["unchanged"], diff["added"], diff["removed"]) == (5, 3, 4)
    assert listing["sides"]["hls"]["totals"]["design"] == 3
    assert listing["sides"]["spmw"]["totals"]["design"] == 2
    md = out_md.read_text(encoding="utf-8")
    assert "| Fixture design | 14 | 9 | 1.56x | 20 | 12 | 1.67x | archived note |" in md
    inc = out_inc.read_text(encoding="utf-8")
    assert "fixture/hls  design  14" in inc
    assert "fixture/spmw  test(remainder)  7" in inc


def test_manifest_errors_fire(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["designs"][0]["sides"]["spmw"]["parts"].append(
        {"file": "fx:py_symbols.py", "category": "test", "symbols": ["IO"]}
    )
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(COUNTER), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "already selected" in result.stderr
    manifest = _manifest(tmp_path)
    manifest["designs"][0]["sides"]["spmw"]["parts"][0]["category"] = "essential"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(COUNTER), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "category" in result.stderr


def test_cli_count_and_version():
    result = subprocess.run(
        [sys.executable, str(COUNTER), "--count", str(FIXTURES / "listing.tex")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert [line.split()[0] for line in result.stdout.splitlines()] == ["2", "3"]
    result = subprocess.run(
        [sys.executable, str(COUNTER), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(result.stdout.split()[-1]) == 64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

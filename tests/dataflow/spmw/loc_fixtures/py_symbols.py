# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Fixture for scripts/spmw_loc2.py: symbol selection.
import os  # KEEP

N = 4  # KEEP


def decorator(fn):  # KEEP
    return fn  # KEEP


class IO:  # KEEP
    """An interface."""

    a = 1  # KEEP


@decorator  # KEEP
def unit(io):  # KEEP
    """Docstring."""
    return io.a  # KEEP


def outer(n):  # KEEP
    class Inner:  # KEEP
        x = n  # KEEP

    return Inner  # KEEP


def test_something():  # KEEP
    assert unit(IO) == 1  # KEEP
    assert os.sep  # KEEP


if __name__ == "__main__":  # KEEP
    test_something()  # KEEP

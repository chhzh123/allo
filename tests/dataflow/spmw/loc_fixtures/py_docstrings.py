# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixture for scripts/spmw_loc2.py: a module docstring does not count.

Neither does its second paragraph; lines that must count carry the marker.
"""


def f(a):  # KEEP
    """A one-line docstring does not count."""
    return a  # KEEP


def g(a):  # KEEP
    """A docstring
    spanning several lines -- with non-ASCII text, §3.1 and an em dash —
    none of which count.
    """
    "a standalone string statement does not count either"
    s = """a triple-quoted value KEEP
    spans two lines and both count KEEP"""
    return s + a  # KEEP


class C:  # KEEP
    """Class docstring."""

    attr = 1  # KEEP

    def m(self):  # KEEP
        '''Single quotes work too.'''
        return self.attr  # KEEP


def h():  # KEEP
    """A string statement after code on the same line leaves the line code."""
    x = 1; "a string after code on the same line"  # KEEP
    return x  # KEEP

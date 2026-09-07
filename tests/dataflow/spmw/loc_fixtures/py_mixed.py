# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Fixture for scripts/spmw_loc2.py: lines mixing code and comments.


def deco(fn):  # KEEP
    return fn  # KEEP


@deco  # KEEP: a decorator line with a comment counts
# a comment between the decorator and the def does not count
def f():  # KEEP
    """Docstring with a # hash and a ''' inside; still a docstring."""
    a = 1  # KEEP: '''not a docstring''' inside a comment
    b = """value with # hash"""  # KEEP
    if a:  # KEEP
        pass  # KEEP
    return a, b  # KEEP


# Comment mentioning """ triple quotes """ does not open a string.
g = f  # KEEP

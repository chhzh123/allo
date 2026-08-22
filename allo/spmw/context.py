# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fabric currently elaborating.

A fabric body runs once to build a graph, and the verbs it calls need to find
that graph.  The registry lives here rather than in :mod:`allo.spmw.graph` so
that bricks and placements can reach it without importing the module that
imports them.
"""

from .errors import SPMWBindingError

_STACK = []


def current_fabric(required=True):
    """The fabric currently elaborating, if any."""
    if _STACK:
        return _STACK[-1]
    if required:
        raise SPMWBindingError(
            "this verb is only meaningful inside an @spmw.fabric body, which is "
            "where structure is declared."
        )
    return None


def push(graph):
    _STACK.append(graph)


def pop():
    return _STACK.pop()

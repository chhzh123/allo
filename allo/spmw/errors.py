# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics raised by the SPMW frontend.

Every check in the design doc's table has a class here, so a caller can catch a
category rather than matching on message text.  All of them derive from
:class:`SPMWError`.
"""


class SPMWError(Exception):
    """Base class for every SPMW frontend diagnostic."""


class SPMWDirectionError(SPMWError):
    """`get` on an Out, `put` on an In, or a write through a MemIn."""


class SPMWTypeError(SPMWError):
    """Element types disagree across the two ends of a link."""


class SPMWOwnershipError(SPMWError):
    """A link names a port symbol that its topology's interface does not own."""


class SPMWTopologyError(SPMWError):
    """A link rule is ill-formed: unpaired edge, many writers on one key, bad rank."""


class SPMWPlacementError(SPMWError):
    """A component does not fit a topology, or role coverage is incomplete."""


class SPMWBindingError(SPMWError):
    """A binding's index map is out of bounds, non-disjoint, or wrongly shaped."""


class SPMWMemoryError(SPMWError):
    """A memory link violates protocol, access, shape, capacity or writer rules."""


class SPMWUnboundError(SPMWError):
    """An `In` port has no producer and no binding covering it."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interfaces -- the contract between a unit and a topology.

An ``Interface`` is a named class of typed, directed ports.  The unit imports it
to create and use its ports; the topology imports it to construct the
interconnection.  Neither side can drift from the other because there is nothing
to drift from -- they share one object.
"""

from .ports import _PortDecl, PortAccessor, PortSymbol, STREAM, MEMORY, IN, OUT


class InterfaceMeta(type):
    """Collects port declarations into a closed, ordered set.

    Subclasses inherit the *same* symbol objects as their bases, so identity
    checks hold across a family and a topology written against a fragment
    accepts every extension of it.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        ports = {}
        # Inherited first, in MRO order, so a redeclaration shadows rather than
        # duplicates.
        for base in reversed(cls.__mro__[1:]):
            ports.update(getattr(base, "__ports__", {}))
        for key, value in namespace.items():
            if isinstance(value, _PortDecl):
                ports[key] = value.symbol
        cls.__ports__ = ports
        return cls

    def __iter__(cls):
        """``for p in MacIO`` walks the closed port set."""
        return iter(cls.__ports__.values())

    def __len__(cls):
        return len(cls.__ports__)

    def __contains__(cls, symbol):
        return symbol in cls.__ports__.values()

    def __repr__(cls):
        return f"<Interface {cls.__name__}: {', '.join(cls.__ports__)}>"

    def owns(cls, symbol):
        """Whether ``symbol`` is one of this interface's ports, by identity."""
        return isinstance(symbol, PortSymbol) and symbol in cls.__ports__.values()

    def ports(cls, protocol=None, direction=None):
        """The declared ports, optionally filtered by protocol and direction."""
        out = []
        for sym in cls.__ports__.values():
            if protocol is not None and sym.protocol != protocol:
                continue
            if direction is not None and sym.direction != direction:
                continue
            out.append(sym)
        return out

    def streams(cls, direction=None):
        return cls.ports(protocol=STREAM, direction=direction)

    def memories(cls):
        return cls.ports(protocol=MEMORY)


class Interface(metaclass=InterfaceMeta):
    """Base class for port contracts.

    An *instance* is the contract bound to one site: ``io.west`` yields an
    accessor for this site's west port.  A misspelling is an ``AttributeError``
    on the line that wrote it.
    """

    def __init__(self, site=None):
        object.__setattr__(self, "_site", site)
        object.__setattr__(self, "_writes", {})

    def _accessor(self, symbol):
        return PortAccessor(symbol, self._site)

    def _write(self, symbol, value):
        self._writes[symbol] = value

    def __getattr__(self, name):
        # Only reached when normal lookup failed, i.e. the name is not a port.
        cls = type(self)
        raise AttributeError(
            f"`{cls.__name__}` has no port `{name}`. "
            f"Declared ports: {', '.join(cls.__ports__) or '(none)'}"
        )


def interface(name="Interface", **ports):
    """Build an interface from keyword declarations.

    The spelling for element-generic contracts, where the ports must be
    constructed inside a function and so cannot be written as a class body::

        def NSEW(T):
            return spmw.interface(west=spmw.In(T), east=spmw.Out(T))
    """
    for key, value in ports.items():
        if not isinstance(value, _PortDecl):
            raise TypeError(
                f"`{key}` is {type(value).__name__}, not a port declaration. "
                f"Use spmw.In/Out/MemIn/MemOut/Mem."
            )
    return InterfaceMeta(name, (Interface,), dict(ports))


def structural(iface):
    """Opt in to structural matching at ``place`` for this interface.

    Nominal-by-identity is the default: port symbols make it the mechanical
    choice, and it lets a mismatch name the shape rather than dump a diff.  This
    wrapper relaxes that to comparison by (name, kind, type).
    """
    return _Structural(iface)


class _Structural:
    """Marker requesting structural rather than nominal interface matching."""

    __slots__ = ("iface",)

    def __init__(self, iface):
        self.iface = iface

    def __repr__(self):
        return f"structural({self.iface.__name__})"


def signature_of(iface):
    """The structural signature used when nominal matching is relaxed."""
    return tuple(
        (sym.name, sym.protocol, sym.direction, sym.access, str(sym.dtype), sym.shape)
        for sym in iface.__ports__.values()
    )


def matches(component_iface, topology_iface):
    """Whether a component's interface fits a topology's.

    Same object, or -- when either side opted in to structural mode --
    structurally identical.
    """
    lhs = (
        component_iface.iface
        if isinstance(component_iface, _Structural)
        else component_iface
    )
    rhs = (
        topology_iface.iface
        if isinstance(topology_iface, _Structural)
        else topology_iface
    )
    if lhs is rhs:
        return True
    relaxed = isinstance(component_iface, _Structural) or isinstance(
        topology_iface, _Structural
    )
    if relaxed:
        return signature_of(lhs) == signature_of(rhs)
    return False


def unwrap(iface):
    """The underlying interface class, dropping any ``structural`` wrapper."""
    return iface.iface if isinstance(iface, _Structural) else iface


__all__ = [
    "Interface",
    "InterfaceMeta",
    "interface",
    "structural",
    "matches",
    "unwrap",
    "signature_of",
    "IN",
    "OUT",
]

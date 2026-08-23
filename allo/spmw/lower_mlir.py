# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emitting the rolled ``spmw.map`` form of an elaborated graph.

Where :mod:`allo.spmw.lower_df` expands a fabric into one kernel instance per
grid point, this states the same design *rolled*: the grid, the channel
families, how each port addresses them, and which body runs where, all as
attributes on a single op.  Nothing is expanded, so the number of bodies that
reach code generation tracks the role count rather than the array's size.

The frontend already computes every piece -- site signatures, per-site routing,
family shapes -- so this is a transcription rather than an analysis.  That is
also why the dialect's two analysis passes are *checks* here rather than the
source of the tables.
"""

from . import channels as ch
from .errors import SPMWBindingError
from .lower_df import Lowering, _wiring_classes
from .ports import STREAM


def _mlir_type(dtype, shape=()):
    """The MLIR spelling of a declared type.

    Allo's scalar types already print as their MLIR names -- ``f32``, ``i8`` --
    so a scalar needs no translation and a block becomes a memref of them.
    """
    base = str(dtype)
    if not shape:
        return base
    extents = "x".join(str(int(s)) for s in shape)
    return f"memref<{extents}x{base}>"


def _stream_type(family):
    return f"!allo.stream<{_mlir_type(family.dtype, family.block)}, {family.depth}>"


def _ints(values):
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def _dense(values, elem="i32"):
    """A per-site table, flat and row-major.

    Kept one-dimensional rather than shaped like the grid: the reader indexes it
    by a linearised coordinate anyway, and a flat literal needs no nesting.
    """
    values = list(values)
    body = ", ".join(str(int(v)) for v in values)
    return f"dense<[{body}]> : tensor<{len(values)}x{elem}>"


class RolledEmitter:
    """Builds the ``spmw.map`` attributes for one placement."""

    def __init__(self, graph):
        self.graph = graph
        self.low = Lowering(graph)

    def placements(self):
        return self.low.placements

    def families(self, placement):
        """Every channel array the placement needs, as ``#spmw.family``."""
        res = self.low.resolutions[placement]
        out = []
        for fam in res.families.values():
            out.append(fam)
        for (pl, _port), fam in self.low.bind_families.items():
            if pl is placement and fam not in out:
                out.append(fam)
        return out

    def family_attrs(self, placement):
        return [
            f'#spmw.family<name = "{fam.name}", type = '
            f"{_mlir_type(fam.dtype, fam.block)}, block = {_ints(fam.block)}, "
            f"depth = {fam.depth}, shape = {_ints(fam.shape)}>"
            for fam in self.families(placement)
        ]

    def port_map_attrs(self, placement):
        """How each port reaches its family: a displacement, or a slot table.

        A port appears once per family it addresses, because routing is a
        per-site question and one port may take part in more than one.
        """
        res = self.low.resolutions[placement]
        grid = placement.grid
        out = []
        for fam in self.families(placement):
            ports = {
                port
                for (_site, port), name in res.site_family.items()
                if name == fam.name
            }
            ports |= {
                port
                for (site, port) in fam.slots
                if (pl_port := port) is not None and pl_port.protocol == STREAM
            }
            for port in sorted(ports, key=lambda p: p.name):
                if fam.kind == ch.AFFINE:
                    offset = fam.offset if port.direction != "in" else (0,) * len(grid)
                    out.append(
                        f'#spmw.port_map<port = "{port.name}", family = '
                        f'"{fam.name}", kind = "affine", offset = {_ints(offset)}>'
                    )
                else:
                    slots = [
                        fam.slots.get((site, port), -1) for site in placement.sites()
                    ]
                    out.append(
                        f'#spmw.port_map<port = "{port.name}", family = '
                        f'"{fam.name}", kind = "table", slots = '
                        f"{_dense(slots)}>"
                    )
        return out

    def role_attrs(self, placement, names):
        """One role per wiring class: the body, its signature, its stream order."""
        res = self.low.resolutions[placement]
        classes = _wiring_classes(placement, res)
        out = []
        for order, (signature, _routing, _sites) in enumerate(classes):
            missing = sorted(
                p.name
                for p in placement.iface.ports()
                if p.protocol == STREAM and p not in signature
            )
            ports = sorted(p.name for p in signature if p.protocol == STREAM)
            out.append(
                f"#spmw.role<unit = @{names[order]}, missing = "
                f"[{', '.join(f'{chr(34)}{m}{chr(34)}' for m in missing)}], "
                f"ports = [{', '.join(f'{chr(34)}{p}{chr(34)}' for p in ports)}]>"
            )
        return out

    def class_table(self, placement):
        """Which role runs at each site, flattened row-major."""
        res = self.low.resolutions[placement]
        classes = _wiring_classes(placement, res)
        index = {}
        for order, (_sig, _routing, sites) in enumerate(classes):
            for site in sites:
                index[site] = order
        return [index[site] for site in placement.sites()]

    def map_attrs(self, placement, role_names):
        """The three attributes `spmw.map` carries, as MLIR text."""
        grid = placement.grid
        topology = (
            f"#spmw.topology<grid = {_ints(grid)}, families = ["
            + ", ".join(self.family_attrs(placement))
            + "], ports = ["
            + ", ".join(self.port_map_attrs(placement))
            + "]>"
        )
        roles = "[" + ", ".join(self.role_attrs(placement, role_names)) + "]"
        classes = _dense(self.class_table(placement))
        return topology, roles, classes


def role_signature(emitter, placement, order, tensors):
    """The calling convention a role body must have.

    Tensors, then one index per grid axis, then one stream per port the site has
    -- which is what the op's verifier checks against.
    """
    res = emitter.low.resolutions[placement]
    signature, _routing, _sites = _wiring_classes(placement, res)[order]
    params = [_mlir_type(t.dtype, t.shape) for t in tensors]
    params += ["index"] * len(placement.grid)
    for port in sorted(
        (p for p in signature if p.protocol == STREAM), key=lambda p: p.name
    ):
        fam = res.families.get(res.site_family.get((_sites[0], port)))
        if fam is None:
            fam = emitter.low.bind_families.get((placement, port))
        if fam is None:
            raise SPMWBindingError(
                f"`{placement.name}.{port.name}` is in a site signature but has "
                f"no family; this is an emission bug."
            )
        params.append(_stream_type(fam))
    return params


def render_module(graph, bodies=None):
    """The rolled program as MLIR text.

    ``bodies`` supplies each role's body; without it the roles are emitted empty,
    which is enough to check that the structure the frontend computed is one the
    dialect accepts.
    """
    emitter = RolledEmitter(graph)
    tensors = list(graph.tensors.values())
    lines = []
    calls = []
    for placement in emitter.placements():
        res = emitter.low.resolutions[placement]
        classes = _wiring_classes(placement, res)
        names = [
            f"{emitter.low.kernel_names[placement]}_r{k}" for k in range(len(classes))
        ]
        for order, name in enumerate(names):
            params = role_signature(emitter, placement, order, tensors)
            args = ", ".join(f"%a{i}: {t}" for i, t in enumerate(params))
            body = (bodies or {}).get(name, "    return")
            lines.append(f"  func.func @{name}({args}) {{\n{body}\n  }}")
        topo, roles, cls = emitter.map_attrs(placement, names)
        operands = ", ".join(f"%t{i}" for i in range(len(tensors)))
        types = ", ".join(_mlir_type(t.dtype, t.shape) for t in tensors)
        calls.append(
            f"    spmw.map ({operands}) topology = {topo} roles = {roles} "
            f"classes = {cls} : {types}"
        )
    params = ", ".join(
        f"%t{i}: {_mlir_type(t.dtype, t.shape)}" for i, t in enumerate(tensors)
    )
    lines.append(
        f"  func.func @top({params}) attributes {{dataflow}} {{\n"
        + "\n".join(calls)
        + "\n    return\n  }"
    )
    return "module {\n" + "\n".join(lines) + "\n}\n"


__all__ = ["RolledEmitter", "role_signature", "render_module"]

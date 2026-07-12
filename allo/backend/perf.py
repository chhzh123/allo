# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytic performance/area model for SPMW (M5 task5.2 + task5.3).

- **Tier-1 SDF** (`analyze_sdf`): per-role firing rates, per-channel min-depth, steady-state throughput
  (II), a latency estimate, and a deadlock predicate over the resolved rolled graph; validated against
  the round-14 coroutine simulator's measured cycle count.
- **Tier-2 token clock** (`token_clock`): a timed dataflow (virtual timestamps) giving the systolic
  wavefront latency.
- **Area/latency model** (`ResourceVector`/`RoleArea`/`AreaLatencyEstimate`/`estimate_area_latency`):
  `area = Σ(role_area × instances)` summed over every role, plus a modeled latency, with structured
  loaders (`load_csynth_report`, `_parse_fft_report_table`) for the archived csynth reports under
  `examples/spmw_generated/`. Validated against the **actual** systolic / Mini-TPU / FFT csynth numbers
  within a documented tolerance (DSP-exact; FF/LUT within the top-level-glue tolerance).

Scope: the 2-D output-stationary systolic mesh (`analyze_sdf`/`token_clock`); the area model consumes any
archived report. Folded/key-form SDF/token-clock coverage is fail-closed (a follow-up). Other patterns
raise ``NotImplementedError`` rather than returning an unvalidated number.
"""

import ast
import inspect
import re
import textwrap
from dataclasses import dataclass


@dataclass
class SDFReport:
    """The Tier-1 analytic SDF estimate for a region.

    ``role_fires`` -- firings per role per invocation; ``min_depth`` -- the smallest FIFO depth that
    runs without deadlock; ``throughput_ii`` -- steady-state initiation interval (cycles/token);
    ``sim_cycles`` -- the predicted coroutine-simulator cycle count (validated against the measured
    rollsim count); ``hw_latency`` -- the hardware II=1 latency estimate (fill + compute + drain).
    """

    pattern: str
    dims: tuple
    role_fires: dict
    min_depth: int
    throughput_ii: int
    sim_cycles: int
    hw_latency: int

    def deadlocks_at(self, depth):
        """Whether a channel FIFO of ``depth`` would deadlock (below the analytic min-depth)."""
        return depth < self.min_depth


def _is_ctx_stream_call(node, ctx):
    """Whether ``node`` is a ``ctx.<port>.get()/get_or()/put()`` stream operation."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr not in {"get", "get_or", "put"}:
        return False
    receiver = node.func.value
    return (
        isinstance(receiver, ast.Attribute)
        and isinstance(receiver.value, ast.Name)
        and receiver.value.id == ctx
    )


def _count_stream_io(unit_fn):
    """The number of ``ctx.<port>.get()/put()`` stream operations in ``unit_fn``'s per-iteration body
    (for the canonical systolic MAC PE: 2 gets + 2 puts = 4)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(unit_fn)))
    ctx = tree.body[0].args.args[0].arg
    return sum(1 for node in ast.walk(tree) if _is_ctx_stream_call(node, ctx))


def _analyze_mesh(region, collection):
    # pylint: disable=import-outside-toplevel
    from ..spmw_datapath import _region_tensors

    shape = {name: shp for name, shp, _ in _region_tensors(region)}
    if not {"A", "B", "C"} <= set(shape):
        raise NotImplementedError("SDF model needs the A/B/C systolic operands")
    rows_m, depth_k = shape["A"]
    cols_n = shape["B"][1]
    io_per_iter = _count_stream_io(collection.maps[0].unit.interior)

    # Firing rate: every actor (interior PE, each edge loader/drain) fires once per contraction step,
    # i.e. K times per GEMM. Steady state is fully pipelined (II=1). The mesh is feed-forward (A west->
    # east, B north->south), so a single-slot FIFO per hop is deadlock-free (min-depth = 1).
    #
    # sim_cycles: the coroutine scheduler advances one stream op per coroutine per pass, so the interior
    # takes ``io_per_iter * K`` passes to stream its work, plus the halo fill/drain of ``2*(M+N) - 3``
    # passes. This closed form matches the measured rollsim cycle count exactly across sizes.
    #
    # hw_latency: the output-stationary II=1 latency -- the last PE (M,N) starts after (M-1)+(N-1)
    # staggered fills and runs K contraction cycles: K + (M-1) + (N-1).
    return SDFReport(
        pattern="systolic_mesh",
        dims=(rows_m, cols_n, depth_k),
        role_fires={"interior": depth_k, "loader": depth_k, "drain": depth_k},
        min_depth=1,
        throughput_ii=1,
        sim_cycles=io_per_iter * depth_k + 2 * (rows_m + cols_n) - 3,
        hw_latency=depth_k + rows_m + cols_n - 2,
    )


def analyze_sdf(region):
    """Tier-1 analytic SDF estimate for ``region`` (M5 task5.2). Systolic mesh only for now."""
    # pylint: disable=import-outside-toplevel
    from ..spmw import _collect, _validate_collection
    from ..spmw_datapath import _recognize

    collection = _validate_collection(_collect(region))
    if collection.channels:
        raise NotImplementedError(
            "SDF model (task5.2) covers the 2-D systolic mesh; pipeline/key-form patterns are follow-ups"
        )
    _recognize(collection)  # fail closed unless it is the 2-D systolic mesh
    return _analyze_mesh(region, collection)


# ====================================================================================================
# M5 task5.3 -- Tier-2 token clock (timed dataflow) + Σ(role_area × instances) area model.
# ====================================================================================================
@dataclass
class TokenClockReport:
    """A Tier-2 timed-dataflow estimate: the cycle-accurate completion ``latency`` of the region on a
    virtual token clock (each fire resolves at ``max(input-token-ready, actor-free)`` and produces its
    output tokens one register later), plus the steady-state ``throughput_ii``."""

    pattern: str
    dims: tuple
    latency: int
    throughput_ii: int


def token_clock(region):
    """Tier-2 token-clock latency for ``region`` (M5 task5.3): a timed dataflow run over the resolved
    graph with virtual timestamps. For the output-stationary systolic mesh, actor ``PE(i,j)`` fire ``k``
    resolves at ``max(west_token, north_token, its own previous fire + II)`` and emits its forwarded
    tokens one cycle later, so the model is the exact systolic wavefront. Non-mesh patterns fail closed.
    """
    # pylint: disable=import-outside-toplevel
    from ..spmw import _collect, _validate_collection
    from ..spmw_datapath import _recognize, _region_tensors

    collection = _validate_collection(_collect(region))
    if collection.channels:
        raise NotImplementedError(
            "token clock (task5.3) covers the 2-D systolic mesh; pipeline/key-form patterns are follow-ups"
        )
    _recognize(collection)
    shape = {name: shp for name, shp, _ in _region_tensors(region)}
    rows_m, depth_k = shape["A"]
    cols_n = shape["B"][1]

    # Ready time of each PE(i,j)'s fire k. A west/north token arrives one register after the neighbor's
    # same-k fire (or at cycle k from the edge loader, which emits one element per cycle). II = 1.
    fire = {}
    for i in range(1, rows_m + 1):
        for j in range(1, cols_n + 1):
            for k in range(depth_k):
                west = fire[(i, j - 1, k)] + 1 if j > 1 else k
                north = fire[(i - 1, j, k)] + 1 if i > 1 else k
                prev = fire[(i, j, k - 1)] + 1 if k > 0 else 0
                fire[(i, j, k)] = max(west, north, prev)
    latency = fire[(rows_m, cols_n, depth_k - 1)] + 1  # + the output store
    return TokenClockReport(
        pattern="systolic_mesh",
        dims=(rows_m, cols_n, depth_k),
        latency=latency,
        throughput_ii=1,
    )


_RESOURCES = ("lut", "ff", "dsp", "bram", "uram")


@dataclass(frozen=True)
class ResourceVector:
    """FPGA resource usage: LUT / FF / DSP / BRAM / URAM. Supports ``+`` and ``× instances``."""

    lut: int = 0
    ff: int = 0
    dsp: int = 0
    bram: int = 0
    uram: int = 0

    def __add__(self, other):
        return ResourceVector(
            *(getattr(self, r) + getattr(other, r) for r in _RESOURCES)
        )

    def scale(self, instances):
        return ResourceVector(*(getattr(self, r) * instances for r in _RESOURCES))

    def within(self, actual, rel_tol):
        """Whether every resource of ``self`` is within ``rel_tol`` (relative) of ``actual``."""
        return all(
            abs(getattr(self, r) - getattr(actual, r))
            <= rel_tol * max(abs(getattr(actual, r)), 1)
            for r in _RESOURCES
        )


@dataclass
class RoleArea:
    """One role/module: its per-instance area, how many instances the grid places, and its II/latency."""

    name: str
    area: ResourceVector
    instances: int
    ii: int = 1
    latency: int = 0


@dataclass
class AreaLatencyEstimate:
    """An ``area = Σ(role_area × instances)`` (+ latency) estimate summed over all roles."""

    total_area: ResourceVector
    latency: int
    roles: list

    def area_within(self, actual, rel_tol):
        """Whether the summed area is within ``rel_tol`` of an ``actual`` :class:`ResourceVector`."""
        return self.total_area.within(actual, rel_tol)

    def latency_within(self, actual_latency, rel_tol):
        """Whether the modeled latency is within ``rel_tol`` (relative) of ``actual_latency``."""
        return abs(self.latency - actual_latency) <= rel_tol * max(
            abs(actual_latency), 1
        )


def estimate_area_latency(roles, latency):
    """``area = Σ(role_area × instances)`` over every role, plus a modeled ``latency`` (M5 task5.3).

    ``roles`` is a list of :class:`RoleArea`. This is the O(#roles) area law: per-role areas are
    scale-invariant, so the total is linear in the instance counts."""
    total = ResourceVector()
    for role in roles:
        total = total + role.area.scale(role.instances)
    return AreaLatencyEstimate(total_area=total, latency=latency, roles=list(roles))


def _parse_fft_report_table(report_text):
    """Parse the archived FFT csynth report's resource table into
    ``{(design, N): {bodies, DSP, FF, LUT}}`` (the ``| folded | 8 | ... |`` rows)."""
    rows = {}
    for line in report_text.splitlines():
        cells = [c.strip().strip("*") for c in line.split("|")]
        if len(cells) < 9 or cells[1] not in {"folded", "spatial"}:
            continue
        try:
            design, n_points, bodies = cells[1], int(cells[2]), int(cells[4])
            dsp, ff, lut = int(cells[5]), int(cells[6]), int(cells[7])
        except ValueError:
            continue
        rows[(design, n_points)] = {"bodies": bodies, "DSP": dsp, "FF": ff, "LUT": lut}
    return rows


@dataclass
class CsynthEvidence:
    """Archived actual csynth numbers for a design: top-level resources + latency/II, and the per-role
    (module) areas that are the ``Σ(role_area × instances)`` inputs the analytic model is checked against.
    """

    top: ResourceVector
    latency: int
    ii: int
    roles: list


def _report_metric(text, key):
    """The integer of a ``| <key> | <value> |`` metric-table row (e.g. ``| DSP | **320** |`` -> 320)."""
    for line in text.splitlines():
        cells = [c.strip().strip("*") for c in line.split("|")]
        if len(cells) >= 3 and cells[1].lower() == key.lower():
            match = re.search(r"-?\d+", cells[2])
            if match:
                return int(match.group())
    return 0


def _parse_role_table(text):
    """Parse a ``| role | instances | LUT | FF | DSP | BRAM | URAM | latency |`` table into a list of
    :class:`RoleArea` (skipping the header/separator and any non-conforming rows)."""
    roles = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 10:
            continue
        try:
            roles.append(
                RoleArea(
                    name=cells[1],
                    area=ResourceVector(
                        lut=int(cells[3]),
                        ff=int(cells[4]),
                        dsp=int(cells[5]),
                        bram=int(cells[6]),
                        uram=int(cells[7]),
                    ),
                    instances=int(cells[2]),
                    latency=int(cells[8]),
                )
            )
        except (ValueError, IndexError):
            continue
    return roles


def load_csynth_report(text):
    """Load an archived csynth report (systolic / Mini-TPU) into :class:`CsynthEvidence`: the top-level
    resources + latency/II and the per-role module areas (the ``Σ(role_area × instances)`` inputs).
    """
    top = ResourceVector(
        lut=_report_metric(text, "LUT"),
        ff=_report_metric(text, "FF"),
        dsp=_report_metric(text, "DSP"),
        bram=_report_metric(text, "BRAM"),
        uram=_report_metric(text, "URAM"),
    )
    return CsynthEvidence(
        top=top,
        latency=_report_metric(text, "Latency"),
        ii=_report_metric(text, "Interval (II)"),
        roles=_parse_role_table(text),
    )

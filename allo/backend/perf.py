# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-1 analytic SDF performance model for SPMW (M5 task5.2).

``analyze_sdf(region)`` reads the resolved rolled SPMW graph -- the topology, the work-unit datapath,
and the resolved per-port FIFO depths -- and returns an :class:`SDFReport` with the synchronous-dataflow
quantities the plan asks for (task5.2): per-role firing **rates**, the per-channel **min-depth** for
deadlock-free operation, the steady-state **throughput** (initiation interval), a **latency** estimate,
and a **deadlock** predicate. It is a static analysis (no execution); its cycle prediction is validated
against the round-14 coroutine simulator's measured cycle count on ``test_systolic`` across sizes.

Scope (task5.2): the 2-D output-stationary systolic mesh (the plan's `test_systolic` gate). Other
patterns raise ``NotImplementedError`` rather than returning an unvalidated number. The Tier-2 token
clock and the ``Σ(role_area × instances)`` area/latency validation vs the archived L3/L4 reports are
task5.3.
"""

import ast
import inspect
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


@dataclass
class AreaEstimate:
    """An ``area = Σ(role_area × instances)`` estimate: ``total`` per resource, plus the ``per_role``
    resource and ``instances`` count it was built from."""

    total: dict
    per_role: dict
    instances: int

    def within(self, actual, resource, rel_tol):
        """Whether the estimate for ``resource`` is within ``rel_tol`` (relative) of ``actual``."""
        est = self.total.get(resource, 0)
        return abs(est - actual) <= rel_tol * max(abs(actual), 1)


def estimate_area(per_role, instances):
    """``Σ(role_area × instances)``: scale a per-role resource dict by the instance count (M5 task5.3).

    The synthesis-time-win corollary is that ``per_role`` is scale-invariant, so the total is linear in
    ``instances`` (O(#roles) for the folded/rolled paths, where ``instances`` = the constant role-body
    count, vs O(P) for the spatial path where it grows)."""
    total = {res: value * instances for res, value in per_role.items()}
    return AreaEstimate(total=total, per_role=dict(per_role), instances=instances)


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

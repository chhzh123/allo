# SPMW reference

SPMW is a Python-embedded language for spatial hardware. You describe one
*unit* of computation, the *grid* it is replicated across, and how neighbouring
copies are *connected*. The compiler generates one hardware block per distinct
wiring class and instantiates it at every position, so a description does not
grow with the size of the array.

Import it as:

    import allo.spmw as spmw
    from allo.ir.types import int8, int16, int32, float32

## Execution model

Every placed unit is its own concurrently running process. A unit's body runs
from top to bottom, repeatedly, for as long as data arrives. Reading a port
blocks until a value is there; writing blocks until there is room. There is no
global clock in the source and no schedule you write: order comes from the data
dependencies you express through ports.

## Interfaces and ports

An interface declares the ports a unit has. It is a class, and its members are
port declarations.

    class ChainIO(spmw.Interface):
        x_in  = spmw.In(int32)        # a stream you read
        x_out = spmw.Out(int32)       # a stream you write
        table = spmw.MemIn(int32[16]) # a memory you read, held locally
        total = spmw.MemOut(int32)    # one value you produce per launch

| Declaration | You may | You may not |
|---|---|---|
| `spmw.In(T)` | `io.p.get()` | write to it |
| `spmw.Out(T)` | `io.p.put(v)` | read from it |
| `spmw.MemIn(T[n])` | `io.p[i]` | assign to it |
| `spmw.MemOut(T)` | `io.p = v` | read it back |

`T` is an Allo scalar type, or a small array type for memories. A stream may
also carry a small fixed-size array, declared `spmw.In(int32[2])`, which
transfers as one token.

## Units

A unit is a function decorated with `@spmw.unit`. Its first parameter is the
interface. Local variables are declared with a type annotation.

    @spmw.unit
    def scale(io: ChainIO):
        acc: int32 = 0
        for k in range(8):
            v = io.x_in.get()
            acc = acc + v
            io.x_out.put(v)
        io.total = acc

Loop bounds may be Python constants captured from the enclosing scope, which
become literals in the generated hardware, or values read from ports.

### Knowing where you are

A unit may take a second parameter and read its own position in the grid:

    @spmw.unit
    def stage(io: ChainIO, site: spmw.Site):
        s = site.rank[0]          # this copy's coordinate on axis 0
        span: int32 = 1 << s      # a per-position constant

The coordinate arrives as an input, so a loop bounded by it is legal and its
trip count varies from copy to copy.

### Roles: positions whose wiring differs

The first copy in a chain has nothing to its west, so its `x_in` is unbound.
Declare a variant body for those positions:

    @stage.role(unbound=(ChainIO.x_in,))
    def stage_first(io: ChainIO):
        io.x_out.put(0)           # must not touch x_in

Touching a port you declared unbound is a compile-time error. If a port is
unbound at some positions and you provide no role for them, elaboration fails
and names the positions.

## Grids and topology

A `Topology` gives a grid a size and a rule saying, for each position, where
each of its output ports goes.

    chain = spmw.Topology(
        ChainIO,
        grid=(4,),
        link=lambda s: {ChainIO.x_out: spmw.to((s + 1,), ChainIO.x_in)},
    )

`spmw.to(coords, port)` names the position and the port a link ends at. A link
whose target falls outside the grid simply does not exist, which is how edges
arise. The `link` lambda takes one argument per grid axis, so a two-dimensional
grid uses `lambda i, j: {...}` and may return several entries.

`spmw.Grid((n,))` is a grid with no links at all, for units connected only to
the boundary. `spmw.mesh(IO, (m, n))` is a helper for the common case of a
two-dimensional grid whose `east` port feeds the western neighbour and whose
`south` port feeds the northern one; it requires the interface to use exactly
those four names.

## Placing units and binding the boundary

Inside a `@spmw.fabric`, `place` puts a unit on a grid and returns a handle
whose attributes are that placement's ports.

    @spmw.fabric
    def engine(X: int32[4, 8], Y: int32[4]):
        P = spmw.place(stage, on=chain)
        spmw.stream_in(X, into=P.x_in, index=(P.rows, ...))
        spmw.gather(Y, from_=P.total)

A fabric's parameters are the tensors that cross the boundary. Bindings say how
a tensor's elements reach the ports that are still unbound after the topology
has done its work.

| Binding | Meaning |
|---|---|
| `stream_in(T, into=P.p, index=...)` | feed tensor `T` into the unbound `p` ports, one token per element |
| `gather(T, from_=P.p, index=...)` | collect tensor `T` from the `p` ports |
| `link(P.p, to=Q.q)` | connect one placement's port to another's |
| `stationary(m, at=P.p)` | give every copy the memory `m` at its `MemIn` port |
| `mem(T[n], init=..., layout=spmw.replicate, name=...)` | a memory the fabric owns |

`index` says which element goes to which channel. `P.rows` and `P.cols` stand
for the grid's axes, and `...` for the remaining tensor axes walked in order.
So `index=(P.rows, ...)` gives channel `i` the whole of row `i`, in order, and
`index=(..., P.cols)` gives channel `j` the whole of column `j`. A binding with
no `index` uses one channel per position.

The number of channels a binding creates, and the order of the tokens on each,
is decided entirely by `index` and the tensor's shape. Get this wrong and the
elements arrive at the wrong ports, which the simulation will show you.

## Errors you will meet

The compiler checks structure before it generates anything. Unknown port names,
reading an `Out`, a link whose ends disagree on type, a port left unbound with
no role, two writers on one channel, and a binding whose shape does not match
the grid are all reported by name and position at elaboration, with the
position that caused them. Read the message: it usually names the fix.

## Worked example: a four-stage scaling chain

Not related to your task. It shows an interface, a unit that reads its own
position, a role for the end of the chain, a topology, and boundary bindings.

    import allo.spmw as spmw
    from allo.ir.types import int32

    NS, NT = 4, 8                       # stages, tokens per launch

    class ScaleIO(spmw.Interface):
        x_in  = spmw.In(int32)
        x_out = spmw.Out(int32)

    line = spmw.Topology(
        ScaleIO, grid=(NS,),
        link=lambda s: {ScaleIO.x_out: spmw.to((s + 1,), ScaleIO.x_in)})

    @spmw.unit
    def stage(io: ScaleIO, site: spmw.Site):
        """Multiply every token by this stage's own coefficient."""
        k: int32 = site.rank[0] + 2
        for _t in range(NT):
            v = io.x_in.get()
            io.x_out.put(v * k)

    @spmw.fabric
    def line_engine(X: int32[NT], Y: int32[NT]):
        P = spmw.place(stage, on=line)
        spmw.stream_in(X, into=P.x_in, index=(...,))
        spmw.gather(Y, from_=P.x_out, index=(...,))

Here the first stage's `x_in` and the last stage's `x_out` are the only unbound
stream ports, so `stream_in` attaches to one and `gather` to the other, each
with a single channel of `NT` tokens.

## Building

    build

compiles every distinct wiring class, assembles the array, and simulates it
against the visible vectors. It prints the compiler's own errors if elaboration
or synthesis fails, then the number of wrong values and the cycle counts.

# Architecture conformance: how a submitted design is judged

Every submitted design is reviewed against the architecture section of the
specification, because a bar a harness cannot check is a suggestion rather than
a requirement. Fifteen artifacts is few enough to read all of them, and each
review is recorded with the evidence for its verdict.

## What the harness checks on its own

| Check | Conforming | Why it bites |
|---|---|---|
| Multipliers | exactly 64 after synthesis | fewer means elements share one, more means the grid is not the grid |
| Latency floor | first product takes at least 22 cycles | a value cannot cross the grid faster than one element per cycle, so a design that finishes sooner moved something further |
| Latency ceiling | first product takes at most 64 cycles | the architecture allows far better, so this only rejects the badly stalled |
| Throughput | at most 16 cycles between product starts | an array that empties itself between products is not pipelined |
| Correctness | every value exact on held-out vectors | |
| Timing | routes at 3.333 ns with non-negative slack | |

These are necessary, not sufficient. A design can pass every one of them and
still broadcast an operand to a whole row.

## What the review checks by reading

Each is answered yes or no, with the lines of the submission that decide it.

1. Are there sixty-four accumulators, one per grid position, each accumulating
   exactly one output element across all eight steps?
2. Does every operand reach an element from a neighbour or an edge port, never
   from a shared bus, a replicated register read by several elements, or an
   array indexed by more than one element in the same cycle?
3. Does `A` move east one element per step, and `B` south?
4. Do results reach an output port through a chain of elements rather than by a
   direct connection from an interior element to the port?
5. Is the grid expressed once and instantiated, rather than sixty-four elements
   written out by hand? *Recorded, not required*: it distinguishes a
   description that scales from one that was unrolled, which is worth knowing
   and is not a pass criterion.

A design that fails any of the first four is reported as **non-conforming**,
counted separately, and never counted as a success, whatever its cycles say.
Its transcript is published with the reason, because a model that produced a
fast wrong-architecture design has told you something real about how the
specification reads in that language.

## Who reviews

Each artifact is reviewed by the same procedure against these five questions,
blind to which arm produced it wherever the language does not give it away, and
the verdicts are published alongside the designs.

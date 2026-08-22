..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

########################################
SPMW: Single-Program, Multiple-Work-unit
########################################

SPMW builds on :doc:`dataflow` for the case a spatial accelerator almost always
is: **one program, replicated over a grid, wired by a fixed rule**.  You write
the work unit once against a declared port contract, declare how the copies are
wired, and let the compiler derive the boundaries.

.. contents:: Table of Contents
   :local:
   :depth: 2


Why
===

An accelerator is rarely irregular.  A systolic array is one PE program,
replicated over a grid, wired to its neighbours by a rule.  Written directly in
:doc:`dataflow`, that regularity gets re-derived by hand in every kernel: a long
``meta_if``/``meta_elif`` chain for the boundary variants, and every data
movement spelled as absolute grid arithmetic such as
``fifo_A[i, j + 1].put(local_A[i - 1, k])``.  The interconnect topology exists
only implicitly, and nothing checks that the kernel and the wiring agree.

SPMW keeps the two axes worth keeping -- one program, and explicit interconnect
via ``put``/``get`` -- and adds a *declared and checked* contract between the
program and the wiring.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Regularity
     - ``allo.dataflow``
     - ``allo.spmw``
   * - Contract
     - implicit: pid chains and FIFO index math
     - one ``Interface``, imported by both the unit and the topology
   * - Replication
     - ``mapping=[P0, P1]``
     - ``place(unit, on=topology)``
   * - Interconnect
     - absolute ``fifo[i, j + 1]`` arithmetic
     - typed links over port symbols, from a rule
   * - Boundary variants
     - ``meta_if`` pid chains
     - computed site signatures; roles and bindings
   * - Data placement
     - manual address math
     - declarative ``shard`` / ``stream_in`` / ``gather``


The five nouns
==============

``Interface``
   The contract: named, directed, typed ports -- streams **and** memories.
``unit``
   A leaf component: one program, written against an Interface.
``fabric``
   A composite: declares memories, places components, wires bindings.
``Topology``
   A grid plus a link rule, *over a given Interface*.
``place``
   Instantiation.  Legal iff the component's Interface matches the topology's.


A systolic GEMM
===============

The interior PE is the whole story; the boundary is bindings.

.. code-block:: python

   import allo.spmw as spmw
   from allo.ir.types import float32

   M, N, K = 4, 4, 4

   class MacIO(spmw.Interface):
       west  = spmw.In(float32)     # stream in
       north = spmw.In(float32)
       east  = spmw.Out(float32)    # stream out
       south = spmw.Out(float32)
       c     = spmw.MemOut(float32) # per-PE result the parent gathers

   @spmw.unit
   def pe(io: MacIO):               # ALL you hand-write
       acc: float32 = 0
       for k in range(K):
           a = io.west.get()
           b = io.north.get()
           acc += a * b
           io.east.put(a)           # systolic forwarding is program, not halo
           io.south.put(b)
       io.c = acc

   @spmw.fabric
   def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
       P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
       spmw.stream_in(A, into=P.west,  index=(P.rows, ...))
       spmw.stream_in(B, into=P.north, index=(..., P.cols))
       spmw.gather   (C, from_=P.c)

Build and run it exactly as a dataflow region:

.. code-block:: python

   import numpy as np

   A = np.random.rand(M, K).astype(np.float32)
   B = np.random.rand(K, N).astype(np.float32)
   C = np.zeros((M, N), dtype=np.float32)

   mod = spmw.build(gemm, target="simulator")
   mod(A, B, C)
   np.testing.assert_allclose(C, A @ B, atol=1e-5)

Every identifier in ``pe`` is a local or a ``MacIO`` member; every identifier in
``gemm`` is an argument, a placement bundle, or a declared port.  A misspelling
such as ``io.wets`` is an ``AttributeError`` where it is written.


Ports
=====

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Kind
     - Protocol
     - Meaning
   * - ``spmw.In(T, depth=2)``
     - stream
     - reading end of a FIFO
   * - ``spmw.Out(T, depth=2)``
     - stream
     - writing end of a FIFO
   * - ``spmw.MemIn(T[shape])``
     - memory
     - an array the site is *given* -- a shard, a stationary weight
   * - ``spmw.MemOut(T[shape])``
     - memory
     - an array the site *produces*, which the parent gathers
   * - ``spmw.Mem(T[shape])``
     - memory
     - shared random access, checked at link time

Members are *assigned*, never annotated, because the declaration must
**construct** the runtime symbol that link dicts key on.  Interfaces compose by
inheritance and share the same symbol objects, so ``MacIO.west is NSEW.west``
holds across a family and a topology written against a fragment accepts every
extension of it.


Topologies
==========

A topology is constructed over an Interface, and its ``link`` function gives,
per site, the far end of that site's ports.

**Coordinate form** covers everything neighbour-based.  Declare each edge once,
from its source; the ``In`` side is derived by inversion, so pairing is symmetric
by construction:

.. code-block:: python

   def mesh(iface, shape):
       return spmw.Topology(iface, grid=shape, link=lambda i, j: {
           iface.east:  spmw.to((i, j + 1), iface.west),
           iface.south: spmw.to((i + 1, j), iface.north),
       })

**Key form** is for permutations and collectives, where the far end is not a
fixed neighbour.  Both ends name a shared label and rendezvous on it:

.. code-block:: python

   def bfly_links(s, b):
       up, lo = bfly_pair(s, b)
       return {BflyIO.up_in:  spmw.key(s, up),     BflyIO.lo_in:  spmw.key(s, lo),
               BflyIO.up_out: spmw.key(s + 1, up), BflyIO.lo_out: spmw.key(s + 1, lo)}

Boundaries follow from the rule rather than being declared.  An ``Out`` whose far
end is off-grid, or that the rule leaves unmapped, is **unbound**: its ``put`` is
a discard.  An ``In`` with no producer is unbound too, and must be covered by a
binding or a role.  Because a rule may *withhold* an edge, unbound ports occur
inside the grid as well as at its rim -- which is how one parameter can split an
array into independently fed column slabs.


Placement and boundaries
========================

``place`` returns a handle exposing the placement's aggregate boundary, and only
that:

* ``P.west`` -- the bundle of unbound ``west`` ports, wherever they occur.  A
  bundle has a shape of its own, along the axes its membership varies over.
* ``P.c`` -- the grid of exported ``c`` bricks.
* ``P.rows``, ``P.cols``, ``P.axes`` -- the grid's axis symbols, which binding
  expressions are written in.  ``spmw.split(P.cols, factor=G)`` names an axis's
  factors.

Site **signatures** -- the set of bound ports -- are what the compiler groups on.
A 2-D mesh yields nine, at any size: interior, four edges, four corners.

**Roles** are variant bodies for sites whose *behaviour* genuinely differs:

.. code-block:: python

   @pe.role(unbound=(MacIO.west,))
   def pe_west(io: MacIO, site: spmw.Site):
       ...

A role body may not touch its declared-unbound ports; that is an error at
declaration, not a deadlock later.  Boundary differences that are only about
*inputs* should be **bindings**, not roles, so the interior body stays total.


Bindings
========

``index=`` spells the *tensor's* subscripts, one entry per tensor axis, written
in the placement's axis symbols.  ``...`` marks the port side's own axes -- a
stream's token counter, a memory port's block axes:

.. code-block:: python

   spmw.stream_in(A, into=P.west, index=(P.rows, ...))   # row i streams A[i, :]
   spmw.shard(V, into=P.w, index=(g * R + k, e))         # PE (k,c) holds V[g.R+k, e]
   spmw.gather(Y, from_=Pact.y_out, index=(..., lane))   # token m from lane n -> Y[m, n]

Every computed index is bounds-checked over the whole (step, site) domain, and a
write must tile its destination exactly.  A lambda under the same keyword is the
escape hatch for maps that are not affine in the axes, such as a bit-reversal.

Other verbs: ``spmw.gather`` and ``spmw.scatter`` (drains and their mirror),
``spmw.link`` (wires two placements' bundles), ``spmw.copy`` (a bulk mover),
``spmw.stationary`` (resident data), and ``spmw.phase`` (fill-then-use epochs,
within which a shared brick's clients must be all readers or a single writer).

A rank-0 source is the constant sequence and costs nothing --
``spmw.stream_in(0, into=P.p_in)`` seeds every chain, and the literal folds into
the consuming sites.


Targets
=======

.. code-block:: python

   spmw.build(fab, target="ref")        # run the graph directly: one task per site
   spmw.build(fab, target="simulator")  # via allo.dataflow, JIT to CPU
   spmw.build(fab, target="vitis_hls", mode="csim", project="top.prj")
   spmw.source(fab)                     # the dataflow program it lowers to

``target="ref"`` needs no compiler at all: it gives each site a thread and each
channel a bounded queue, so blocking *is* the handshake.  It answers "does this
compute the right thing, and does it deadlock?" fastest, and makes a good oracle.

Everything else elaborates the fabric and lowers it to an ``allo.dataflow``
program: placements become kernels dispatched by site signature, channel families
become stream arrays, and movers become their own kernels.  ``spmw.source`` shows
that program, which is worth reading when a build surprises you.


What is checked
===============

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Check
     - Fires at
   * - Unknown port name (``io.wets``)
     - unit declaration
   * - ``get`` on an ``Out``, ``put`` on an ``In``, a memory port used as a stream
     - unit declaration
   * - Link keys and far-end targets owned by the topology's interface
     - ``Topology`` construction
   * - Coordinate links pair ``Out`` to ``In`` with matching element types
     - ``Topology`` construction
   * - One writer per key; broadcast allowed, arbitration never
     - ``Topology`` construction
   * - Component fits topology
     - ``place``
   * - A role body touching a declared-unbound port
     - role declaration
   * - Index maps in bounds over the whole domain; writes tile exactly
     - bind time
   * - Every unbound ``In`` covered; every ``MemIn`` bound
     - build
   * - One writer per shared brick per phase
     - build


Current limits
==============

* ``fold``/``unroll`` at ``place`` are carried but not yet realised.
* A placed fabric is expanded per site, so a tiled design emits one kernel per
  tile rather than instantiating one engine. That is correct but not yet the
  hierarchical IP reuse the model is built for.
* The lowered program is a dataflow region, so HLS still schedules per instance.
  Keeping the rolled form all the way to codegen is the next step, and is what
  turns the constant signature count into a constant synthesis time.

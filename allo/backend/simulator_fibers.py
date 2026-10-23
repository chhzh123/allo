# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, too-many-locals
# pylint: disable=no-value-for-parameter, too-many-function-args
# pylint: disable=unexpected-keyword-arg
"""Run a dataflow design's units as fibers over a pool of threads.

The simulator's default is a thread per unit. That is ideal while the units
fit in the cores, and wasteful beyond it: a thread holding a core is often
waiting on a FIFO whose producer is not scheduled, and each wait costs an OS
transition of a few microseconds.

This gives each OS thread a list of units and round-robins them: a unit that
blocks on a FIFO is suspended, the thread picks up another, and the unit is
resumed -- by any thread -- once the FIFO moves. A switch is a `swapcontext`,
about a hundred nanoseconds, and the thread count is the core count whatever
the design's size.

What it takes, and why the IR has to change:

* A unit must start on a stack of its own, so the scheduler needs it as a
  callable entry point with no arguments. Its arguments in `top` are live
  values -- FIFO structs, tensors, its site indices -- so each one is given a
  `memref.global` and a thunk per unit fetches them back and makes the call.
  The indices are constants and are rebuilt.
* A tensor a unit takes is often `top`'s own argument, whose address the
  caller supplies at run time, so there is nothing to hoist. Its global is
  staged instead: copied in before the units run and out after they finish.
  Nothing else touches it in between, so the run computes the same thing.
* The FIFO struct a unit takes is a 0-d memref of a struct of three memrefs
  (buffer, head, tail). A struct has no global initialiser, so the thunk
  builds its own wrapper around the three hoisted members: a different
  wrapper, the same state.

Opt in with `spmw.build(..., fibers=True)` or `ALLO_SIM_FIBERS=1`.
"""

import os

from .._mlir.ir import (
    Block,
    BlockArgument,
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    IntegerType,
    MemRefType,
    Module,
    StringAttr,
    TypeAttr,
)
from .._mlir.dialects import (
    affine as affine_d,
    allo as allo_d,
    arith as arith_d,
    func as func_d,
    memref as memref_d,
    openmp as openmp_d,
)
from ..spmw.errors import SPMWPlacementError

#: The scheduler. One ucontext fiber a unit, a ready queue, and a table of
#: who waits on which FIFO word, hashed by its address. A notify on a word
#: nobody waits on touches one atomic and returns, which is the common case.
RUNTIME = r"""
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define MAX_FIBERS 8192
#define SLOTS 8192
#define NO_FIBER (-1)

enum { FREE = 0, READY, RUNNING, BLOCKED, DONE };

typedef struct {
  ucontext_t ctx;
  void (*fn)(void);
  char *stack;
  int state;
  int next;            /* in a slot's wait list, or the ready stack */
  intptr_t word;       /* the FIFO word it waits on */
  int32_t blocked_on;  /* the value of that word that means "still blocked" */
} fiber_t;

static fiber_t fibers[MAX_FIBERS];
static int n_fibers;

/* Who waits on each FIFO word. `count` is read without the lock so that a
   notify on an unwatched word costs one atomic load. */
typedef struct {
  int32_t count;
  int head;
  pthread_mutex_t lock;
} slot_t;
static slot_t slots[SLOTS];

static int ready_top = NO_FIBER;
static int live;
static pthread_mutex_t rq_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t rq_wake = PTHREAD_COND_INITIALIZER;
static int reported;

static size_t stack_bytes = 256 * 1024;
static int spins = 64;

static __thread int cur = NO_FIBER;
static __thread ucontext_t sched;

__attribute__((constructor)) static void fiber_init(void) {
  const char *s = getenv("ALLO_SIM_FIBER_STACK");
  if (s)
    stack_bytes = (size_t)atoi(s) * 1024;
  s = getenv("ALLO_SIM_SPIN");
  if (s)
    spins = atoi(s);
  for (int i = 0; i < SLOTS; i++) {
    slots[i].count = 0;
    slots[i].head = NO_FIBER;
    pthread_mutex_init(&slots[i].lock, NULL);
  }
}

static inline slot_t *slot_of(intptr_t addr) {
  return &slots[((addr >> 2) * 2654435761u) & (SLOTS - 1)];
}

static inline void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

void allo_fiber_add(void (*fn)(void)) {
  if (n_fibers >= MAX_FIBERS) {
    fprintf(stderr, "allo: more than %d units to simulate\n", MAX_FIBERS);
    abort();
  }
  int k = n_fibers++;
  fibers[k].fn = fn;
  fibers[k].state = READY;
  fibers[k].stack = NULL;
  fibers[k].next = ready_top;
  ready_top = k;
  live++;
}

/* How many workers to run the units on. Fewer than the cores on purpose: a
   block and a wake both pass through the ready queue, and on a 48-core
   machine 48 workers contend on it badly -- one engine took 9.2 s on 48 and
   1.7 s on 16. A quarter of the cores is about where it flattens; set
   ALLO_SIM_THREADS to measure that for yourself. */
int32_t allo_fiber_threads(void) {
  long want = sysconf(_SC_NPROCESSORS_ONLN) / 4;
  const char *s = getenv("ALLO_SIM_THREADS");
  if (s)
    want = atoi(s);
  if (want < 2)
    want = 2;
  return (int32_t)(n_fibers < want ? n_fibers : want);
}

static void push_ready(int k) {
  pthread_mutex_lock(&rq_lock);
  fibers[k].state = READY;
  fibers[k].next = ready_top;
  ready_top = k;
  pthread_cond_signal(&rq_wake);
  pthread_mutex_unlock(&rq_lock);
}

/* Put a unit that has just suspended onto the list of waiters for its word.
   This runs on the scheduler, *after* the switch, never on the unit itself:
   a unit that enlisted before switching could be resumed by a producer
   before its context was saved, and two threads would run the one stack. */
static void park(int k) {
  intptr_t addr = fibers[k].word;
  slot_t *s = slot_of(addr);
  pthread_mutex_lock(&s->lock);
  /* Enlist first, then look at the word. Both orders have to be safe against
     a producer that moves it just now: one that moves it after the count goes
     up sees the count and waits for this lock, and one that moved it before
     is seen by the load below. Looking first and enlisting second loses the
     wakeup when each misses the other, which is how a 42-unit engine stalled
     with one unit parked on a word that had already moved. */
  fibers[k].next = s->head;
  s->head = k;
  __atomic_add_fetch(&s->count, 1, __ATOMIC_SEQ_CST);
  if (__atomic_load_n((int32_t *)addr, __ATOMIC_ACQUIRE) != fibers[k].blocked_on) {
    /* It moved. Nobody else can have taken it off the list, because that is
       done under this lock, so take it off here and run it again. */
    s->head = fibers[k].next;
    __atomic_sub_fetch(&s->count, 1, __ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&s->lock);
    push_ready(k);
    return;
  }
  pthread_mutex_unlock(&s->lock);
}

static int pop_ready(void) {
  pthread_mutex_lock(&rq_lock);
  for (;;) {
    if (ready_top != NO_FIBER) {
      int k = ready_top;
      ready_top = fibers[k].next;
      fibers[k].state = RUNNING;
      pthread_mutex_unlock(&rq_lock);
      return k;
    }
    if (live == 0) {
      pthread_cond_broadcast(&rq_wake);
      pthread_mutex_unlock(&rq_lock);
      return NO_FIBER;
    }
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_sec += 10;
    if (pthread_cond_timedwait(&rq_wake, &rq_lock, &until) == ETIMEDOUT &&
        ready_top == NO_FIBER && !reported) {
      reported = 1;
      int parked = 0, running = 0, done = 0, ready = 0;
      for (int i = 0; i < n_fibers; i++) {
        parked += fibers[i].state == BLOCKED;
        running += fibers[i].state == RUNNING;
        done += fibers[i].state == DONE;
        ready += fibers[i].state == READY;
      }
      fprintf(stderr,
              "allo: the unit pool has stalled -- units %d, live %d, "
              "parked %d, running %d, ready %d, done %d\n",
              n_fibers, live, parked, running, ready, done);
      int shown = 0, stale = 0;
      for (int i = 0; i < n_fibers; i++) {
        if (fibers[i].state != BLOCKED)
          continue;
        int32_t now = *(int32_t *)fibers[i].word;
        stale += now != fibers[i].blocked_on;
        if (shown++ < 6)
          fprintf(stderr,
                  "allo:   parked unit %d word %p blocked_on %d now %d%s\n", i,
                  (void *)fibers[i].word, fibers[i].blocked_on, now,
                  now != fibers[i].blocked_on ? "  <- should have been woken" : "");
      }
      fprintf(stderr, "allo: %d of %d parked units are already runnable\n",
              stale, parked);
    }
  }
}

static void fiber_entry(void) {
  int k = cur;
  fibers[k].fn();
  pthread_mutex_lock(&rq_lock);
  fibers[k].state = DONE;
  live--;
  if (live == 0)
    pthread_cond_broadcast(&rq_wake);
  pthread_mutex_unlock(&rq_lock);
  setcontext(&sched); /* never returns */
}

void allo_fiber_worker(void) {
  for (;;) {
    int k = pop_ready();
    if (k == NO_FIBER)
      return;
    cur = k;
    if (fibers[k].stack == NULL) {
      fibers[k].stack = (char *)malloc(stack_bytes);
      if (!fibers[k].stack) {
        fprintf(stderr, "allo: no memory for a unit's stack\n");
        abort();
      }
      getcontext(&fibers[k].ctx);
      fibers[k].ctx.uc_stack.ss_sp = fibers[k].stack;
      fibers[k].ctx.uc_stack.ss_size = stack_bytes;
      fibers[k].ctx.uc_link = NULL;
      makecontext(&fibers[k].ctx, fiber_entry, 0);
    }
    swapcontext(&sched, &fibers[k].ctx);
    cur = NO_FIBER;
    if (fibers[k].state == BLOCKED)
      park(k);
  }
}

void allo_fiber_reset(void) {
  for (int i = 0; i < n_fibers; i++) {
    free(fibers[i].stack);
    memset(&fibers[i], 0, sizeof(fibers[i]));
  }
  n_fibers = 0;
  live = 0;
  reported = 0;
  ready_top = NO_FIBER;
  for (int i = 0; i < SLOTS; i++) {
    slots[i].count = 0;
    slots[i].head = NO_FIBER;
  }
}

/* Wait until the word differs from `blocked`. On a fiber this suspends the
   unit and hands the thread back to the scheduler; off one (a design built
   without fibers, or `top` itself) it spins. */
void allo_fifo_wait(intptr_t addr, int32_t blocked) {
  int32_t *word = (int32_t *)addr;
  for (int i = 0; i < spins; i++) {
    if (__atomic_load_n(word, __ATOMIC_ACQUIRE) != blocked)
      return;
    relax();
  }
  if (cur == NO_FIBER) {
    /* Not a unit: `top` itself, or a call the rewrite left outside the pool.
       Nothing can be switched to, so spin -- and say so if it never ends. */
    long long turns = 0;
    while (__atomic_load_n(word, __ATOMIC_ACQUIRE) == blocked) {
      relax();
      if (++turns == 200000000LL) {
        int parked = 0, running = 0, done = 0;
        for (int i = 0; i < n_fibers; i++) {
          parked += fibers[i].state == BLOCKED;
          running += fibers[i].state == RUNNING;
          done += fibers[i].state == DONE;
        }
        fprintf(stderr,
                "allo: a wait outside the fiber pool is stuck on %p "
                "(units %d, live %d, parked %d, running %d, done %d)\n",
                (void *)addr, n_fibers, live, parked, running, done);
      }
    }
    return;
  }
  int k = cur;
  fibers[k].word = addr;
  fibers[k].blocked_on = blocked;
  fibers[k].state = BLOCKED;
  swapcontext(&fibers[k].ctx, &sched);
}

/* The other side moved the word: wake whoever was waiting for it to move. */
void allo_fifo_notify(intptr_t addr) {
  slot_t *s = slot_of(addr);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (__atomic_load_n(&s->count, __ATOMIC_ACQUIRE) == 0)
    return;
  pthread_mutex_lock(&s->lock);
  int prev = NO_FIBER, k = s->head;
  while (k != NO_FIBER) {
    int nxt = fibers[k].next;
    int32_t now = __atomic_load_n((int32_t *)fibers[k].word, __ATOMIC_ACQUIRE);
    if (fibers[k].word == addr && now != fibers[k].blocked_on) {
      if (prev == NO_FIBER)
        s->head = nxt;
      else
        fibers[prev].next = nxt;
      __atomic_sub_fetch(&s->count, 1, __ATOMIC_SEQ_CST);
      push_ready(k); /* slot lock then ready lock, the one order taken */
    } else {
      prev = k;
    }
    k = nxt;
  }
  pthread_mutex_unlock(&s->lock);
}
"""


def enabled(given=None):
    """Whether to build the fiber scheduler rather than a thread a unit."""
    if given is not None:
        return bool(given)
    return os.environ.get("ALLO_SIM_FIBERS", "") not in ("", "0")


def _global_named(module, kind, names):
    """A fresh private `memref.global` of that type, and how to reach it."""
    name = f"allo_sim_g{len(names)}"
    names[name] = kind
    memref_d.GlobalOp(
        sym_name=StringAttr.get(name),
        type_=TypeAttr.get(kind),
        sym_visibility=StringAttr.get("private"),
        initial_value=None,
        constant=None,
        alignment=None,
        ip=InsertionPoint(module.body),
    )
    return name, kind


def _global_for(module, alloc, names):
    """Hoist one `memref.alloc` in `top` into a `memref.global`, zeroed.

    Every user -- `top` itself and the thunks -- then reaches the same state
    through `memref.get_global`, which is what lets a unit be called from a
    function that was not where its buffers were allocated.
    """
    kind = MemRefType(alloc.result.type)
    name, _ = _global_named(module, kind, names)
    fetch = memref_d.GetGlobalOp(
        kind, FlatSymbolRefAttr.get(name), ip=InsertionPoint(alloc)
    )
    alloc.result.replace_all_uses_with(fetch.result)
    alloc.operation.erase()
    return name, kind


def declare(module):
    """The scheduler's entry points, as seen from the IR."""
    thunk = FunctionType.get([], [])
    i32 = IntegerType.get_signless(32, module.context)
    wanted = {
        "allo_fiber_add": FunctionType.get([thunk], []),
        "allo_fiber_threads": FunctionType.get([], [i32]),
        "allo_fiber_worker": thunk,
        "allo_fiber_reset": thunk,
    }
    have = {
        str(op.sym_name).strip('"')
        for op in module.body.operations
        if isinstance(op, func_d.FuncOp)
    }
    for name, kind in wanted.items():
        if name in have:
            continue
        decl = func_d.FuncOp(name=name, type=kind, ip=InsertionPoint(module.body))
        decl.attributes["sym_visibility"] = StringAttr.get("private")


def rewrite(module: Module, top, pe_calls, parts, empty_map):
    """Replace the per-unit sections with a thunk a unit and a pool of workers.

    ``parts`` maps a stream's struct memref value to its (buffer, head, tail)
    allocs, which is what a thunk needs to rebuild the struct it passes.
    ``empty_map`` is the 0-d affine map the struct's store wants.
    """
    declare(module)
    calls = list(pe_calls)
    if not calls:
        return 0
    # An operand is one of four things, and each needs a global the thunk can
    # fetch. The bookkeeping is keyed by *values*, never by operations: a
    # hoist erases its alloc, and a dict keyed by the dead op cannot even be
    # written to afterwards.
    order, ops, staged = {}, [], {}

    def slot(value, op):
        if value not in order:
            order[value] = len(ops)
            ops.append(op)
        return order[value]

    plans = []
    for call in calls:
        plan = []
        for operand in call.operands_:
            if operand in parts:
                trio = parts[operand]
                plan.append(
                    (
                        "stream",
                        MemRefType(operand.type).element_type,
                        [slot(a.result, a) for a in trio],
                    )
                )
                continue
            try:
                BlockArgument(operand)
            except ValueError:
                pass
            else:
                # One of `top`'s own arguments: its address comes from the
                # caller, so it is staged through a global instead.
                staged.setdefault(operand, None)
                plan.append(("staged", operand, None))
                continue
            owner = operand.owner.opview
            if isinstance(owner, memref_d.AllocOp):
                plan.append(("global", slot(operand, owner), None))
            elif isinstance(owner, arith_d.ConstantOp):
                plan.append(("constant", owner, None))
            else:
                raise SPMWPlacementError(
                    f"a unit takes a {owner.operation.name}, which the fiber "
                    f"scheduler has no global for"
                )
        plans.append(plan)
    names = {}
    where = [_global_for(module, op, names) for op in ops]
    for value in list(staged):
        staged[value] = _global_named(module, MemRefType(value.type), names)

    def fetch(where, ip):
        name, kind = where
        return memref_d.GetGlobalOp(kind, FlatSymbolRefAttr.get(name), ip=ip).result

    made = 0
    for call, plan in zip(calls, plans):
        callee = str(call.callee)[1:]
        thunk = func_d.FuncOp(
            name=f"allo_fiber_run{made}",
            type=FunctionType.get([], []),
            ip=InsertionPoint(module.body),
        )
        thunk.attributes["sym_visibility"] = StringAttr.get("private")
        block = Block.create_at_start(thunk.body, [])
        ip = InsertionPoint(block)
        args = []
        for kind, a, b in plan:
            if kind == "stream":
                members = [fetch(where[i], ip) for i in b]
                built = allo_d.StructConstructOp(output=a, input=members, ip=ip)
                box = memref_d.AllocOp(MemRefType.get([], a), [], [], ip=ip)
                affine_d.AffineStoreOp(
                    value=built, memref=box, indices=[], map=empty_map, ip=ip
                )
                args.append(box.result)
            elif kind == "global":
                args.append(fetch(where[a], ip))
            elif kind == "staged":
                args.append(fetch(staged[a], ip))
            else:
                args.append(
                    arith_d.ConstantOp(
                        a.result.type, a.attributes["value"], ip=ip
                    ).result
                )
        func_d.CallOp([], FlatSymbolRefAttr.get(callee), args, ip=ip)
        func_d.ReturnOp([], ip=ip)
        made += 1

    # `top` registers the thunks, runs a pool of workers over them, and frees
    # their stacks. The unit calls themselves are gone.
    ip = InsertionPoint(beforeOperation=calls[0])
    for value, where in staged.items():
        memref_d.CopyOp(value, fetch(where, ip), ip=ip)
    thunk_type = FunctionType.get([], [])
    for k in range(made):
        fn = func_d.ConstantOp(
            thunk_type, FlatSymbolRefAttr.get(f"allo_fiber_run{k}"), ip=ip
        )
        func_d.CallOp([], FlatSymbolRefAttr.get("allo_fiber_add"), [fn.result], ip=ip)
    threads = func_d.CallOp(
        [IntegerType.get_signless(32, module.context)],
        FlatSymbolRefAttr.get("allo_fiber_threads"),
        [],
        ip=ip,
    )
    region = openmp_d.ParallelOp(
        [], [], [], [], num_threads=threads.results[0], ip=ip
    )
    worker_ip = InsertionPoint(Block.create_at_start(region.region, []))
    func_d.CallOp([], FlatSymbolRefAttr.get("allo_fiber_worker"), [], ip=worker_ip)
    openmp_d.TerminatorOp(ip=worker_ip)
    func_d.CallOp([], FlatSymbolRefAttr.get("allo_fiber_reset"), [], ip=ip)
    for value, where in staged.items():
        memref_d.CopyOp(fetch(where, ip), value, ip=ip)
    for call in calls:
        call.operation.erase()
    _ = top
    return made

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The programmable mini-TPU as a whole engine: a memory port in, one out.

`test_spmw_ptpu` is an array and its dispatch. Its operands arrive on streams,
``2S`` bytes a cycle, each sent once for every block that uses it, so it has
no counterpart of the scratchpads, loaders and DMA that Gemmini and VTA are
mostly made of. This is that engine with a memory system: it fetches its
program and its operands through one 64-bit read port and writes its results
through one 64-bit write port, a request and then its beats, as VTA's `Core`
does through its VME and Gemmini through TileLink.

**The order is what makes the memory system small.** A tile, a GEMM of ``L``
rows, is run a K block at a time: for each block of ``S`` of ``K``, for each
block of ``S`` columns, the ``L`` rows. Every weight block is then used once,
as it arrives, and an activation chunk -- ``L`` rows of ``S`` values -- serves
the column blocks of its K block and is done. So nothing has to be kept for
a second use but that one chunk, ``L`` words of ``S`` bytes, and within a
tile no operand crosses the port twice. What persists across K blocks is the
partial sums, one a lane for every output of the tile: a lane keeps them in a
RAM and adds into it, as VTA's accumulator scratchpad does and Gemmini's
`AccumulatorMem`.

The cells are `test_spmw_ptpu`'s, unchanged: they run on the tokens' framing
bits and count nothing. Around them:

* **`req`** owns the read port's requests. It fetches four words an
  instruction and walks the GEMM a request at a time -- per K block a weight
  block, the activation chunk, the rest of the weight blocks, and before a
  block of the first K block its biases. Nothing it does waits on a beat of
  data but those four words, so it runs ahead of memory and a burst follows
  the one before it with no cycle between.
* **`deal`** owns the read port's beats. Each request sends it a tag, and it
  hands the beats that come back to the streams that want them, as far ahead
  of the array as their FIFOs let it, so the loads hide behind the arithmetic.
* **`head`** is the sequencer. One iteration a row: it reads the row's
  activation word, from the dealer for the first column block and from its
  chunk buffer after; a weight word on the last ``S`` rows of a block, which
  is the next block's weights shifting in; and it issues the row's micro-op.
  Whenever a word has not arrived it waits, and the array waits with it: the
  activation and the weights of a row travel as one token, so a pause is a
  bubble in every row at once, which bare-register links carry without loss.
* **`etap`**, one an array row, hands its row a byte of each and passes the
  rest down, a row a cycle, which is the array's own skew.
* **`lane`** adds a row's partial sum into its accumulator, and on a GEMM's
  last K block applies the epilogue. Its bias comes down the micro-op chain,
  a lane at a time on the ``S`` rows before the block that needs it.
* **`ctap`**, one a lane, gathers a row's ``S`` results into one word, and
  **`pack`** packs words into the write port's beats. It pays the head a
  credit for every row it takes, and the head issues a row of results only
  into a place the row buffer is known to have, so the lanes are never held
  by a full one.
* **`wreq`** owns the write port's requests: a tile's results as bursts, each
  asked for while the one before it is still being written, and the launch
  done when the last is acknowledged.

The units that count are written as a hardware designer would write them:
counters that run down to zero and flags set as they move, so that what an
iteration does is read off registers and a row is never a comparison away.

**An instruction** is four 64-bit words, a GEMM of ``T`` tiles:

    (T - 1) << 52 | (L - 1) << 44 | (NB - 1) << 32 | (KB - 1) << 16
        | FINAL << 8 | BIAS | REUSE | RELU | SHIFT
    weights << 32 | activations
    results << 32 | biases
    chunk beats << 16 | rows

The last word is an activation chunk in beats and a tile's rows of results:
what the first word's bounds multiply out to, so that the requester
multiplies nothing.

``REUSE`` reads every tile's weights from the same address, a layer's tokens
taken ``L`` at a time; without it the tiles' weights follow each other, as
E3's microbenchmark has them. The biases are read again for each tile. Memory
holds each tensor in the order the array takes it: activations a K block at
a time, ``[kb][row][S]``; weights a block at a time in shift order,
``[kb][nb][column][S]``; results ``[nb][row][S]``, which is the layout the
next layer's activations want.
"""

import linecache

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int16, int32, int64, uint1

from test_spmw_tpu_micro import INT8_MAX, TILES, stimulus_mkn
from test_spmw_llama_ffn import TOKENS, SLICE, llama_stimulus

#: An instruction's flags, above its five-bit shift.
RELU, REUSE, BIAS, FINAL = 1 << 5, 1 << 6, 1 << 7, 1 << 8
#: A micro-op's bits, above the shift and `RELU` it copies from the instruction:
#: the head sets them and a lane reads them, bits 6 to 12 and 18. `SETB` comes
#: with a lane's number in bits 13-17 and its bias in the upper half.
BIASED, FIRST, EMIT, LOAD, END, WRAP, SETB, ROW = (
    1 << b for b in (6, 7, 8, 9, 10, 11, 12, 18)
)
#: The most beats one request asks for, AXI's longest burst.
BURST = 256
#: The outputs a GEMM tile may have: the lanes' accumulators hold this many.
OUTPUTS = 8192
#: The rows an activation chunk may have: the head's buffer holds this many.
CHUNK = 64


def _unit(name, src, **closure):
    """A unit from generated source, kept where `inspect.getsource` reads it."""
    filename = f"<spmw-{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    scope = {
        "int8": int8,
        "int16": int16,
        "int32": int32,
        "int64": int64,
        "uint1": uint1,
    }
    exec(compile(src, filename, "exec"), scope)  # pylint: disable=exec-used
    fn = scope["make"](**closure)
    fn.__name__ = name
    return spmw.unit(fn)


def _byte(value, k):
    """Byte ``k`` of ``value`` as an int8: what a vector of int8 holds."""
    return f"((({value} >> {8 * k}) & 255) ^ 128) - 128"


def _word(pad, word, halves):
    """Lines filling the vector ``word`` from 64-bit ``halves``, low bytes first."""
    return [
        f"{pad}{word}[{8 * h + i}] = {_byte(half, i)}"
        for h, half in enumerate(halves)
        for i in range(8)
    ]


def _req_source(S):
    """The requester: a GEMM's requests, in the order the array takes its words.

    One request an iteration, and nothing an iteration does depends on a beat
    of data but the four words of an instruction. So it runs ahead of memory:
    the next request is waiting when a burst's last beat leaves, and the read
    port carries a beat every cycle it has somewhere to put one.

    ``st`` is what an iteration does: 0 takes the launch, 1 asks for an
    instruction, 2 takes a word of it, 3 starts its GEMM, 4 starts a tile, and
    5, 6 and 7 ask for a block's biases, a weight block and a K block's
    activation chunk. A block's biases come before its weights and a K block's
    chunk after its first weight block, which is the order the head takes them
    in: so no FIFO between the two has to hold more than a block's worth for
    the next word the head wants to get through.

    Where it is in the GEMM is three counters that run down to zero and the
    flags that say so, set as a counter moves: nothing is compared with a
    bound on the way to a request.
    """
    lg = S.bit_length() - 1
    lines = [
        "def make(IO):",
        "    def req(io: IO):",
        "        pc: int32 = 0",
        "        idx: int32 = 0",
        "        cfg: int64 = 0",
        "        bias: uint1 = 0",
        "        reuse: uint1 = 0",
        "        final: uint1 = 0",
        "        kbn: int32 = 0",
        "        nbn: int32 = 0",
        "        tn: int32 = 0",
        "        abase: int32 = 0",
        "        wbase: int32 = 0",
        "        bbase: int32 = 0",
        "        ybase: int32 = 0",
        "        aptr: int32 = 0",
        "        wptr: int32 = 0",
        "        bptr: int32 = 0",
        "        yptr: int32 = 0",
        "        rows: int32 = 0",
        "        abeats: int32 = 0",
        "        kleft: int32 = 0",
        "        nleft: int32 = 0",
        "        tleft: int32 = 0",
        "        kfirst: uint1 = 0",
        "        klast: uint1 = 0",
        "        nfirst: uint1 = 0",
        "        nlast: uint1 = 0",
        "        tlast: uint1 = 0",
        "        st: int32 = 0",
        "        go: uint1 = 1",
        "        while go != 0:",
        "            adv: uint1 = 0",
        "            if st == 0:",
        "                la: int64 = io.launch.get()",
        "                pc = la & 1073741823",
        "                st = 1",
        "            elif st == 1:",
        "                p64: int64 = pc",
        "                four: int64 = 4",
        "                io.rd_cmd.put((four << 32) | p64)",
        f"                io.tag_out.put({4 << 8})",
        "                pc = pc + 32",
        "                idx = 0",
        "                st = 2",
        "            elif st == 2:",
        "                w: int64 = io.ins_in.get()",
        "                if idx == 0:",
        "                    cfg = w",
        "                    reuse = (w >> 6) & 1",
        "                    bias = (w >> 7) & 1",
        "                    final = (w >> 8) & 1",
        "                    kbn = (w >> 16) & 65535",
        "                    nbn = (w >> 32) & 4095",
        "                    tn = (w >> 52) & 4095",
        "                elif idx == 1:",
        "                    abase = w & 1073741823",
        "                    wbase = (w >> 32) & 1073741823",
        "                elif idx == 2:",
        "                    bbase = w & 1073741823",
        "                    ybase = (w >> 32) & 1073741823",
        "                else:",
        "                    rows = w & 65535",
        "                    abeats = (w >> 16) & 65535",
        "                    st = 3",
        "                idx = idx + 1",
        "            elif st == 3:",
        "                io.op_out.put(cfg)",
        "                aptr = abase",
        "                wptr = wbase",
        "                yptr = ybase",
        "                tleft = tn",
        "                tlast = 0",
        "                if tn == 0:",
        "                    tlast = 1",
        "                st = 4",
        "            elif st == 4:",
        # a tile: where its results go, for the write side
        "                last: int64 = 0",
        "                if (final & tlast) != 0:",
        "                    last = 1",
        "                r64: int64 = rows",
        "                y64: int64 = yptr",
        "                io.y_out.put((last << 62) | (r64 << 32) | y64)",
        f"                yptr = yptr + (rows << {lg})",
        "                if reuse != 0:",
        "                    wptr = wbase",
        "                bptr = bbase",
        "                kleft = kbn",
        "                kfirst = 1",
        "                klast = 0",
        "                if kbn == 0:",
        "                    klast = 1",
        "                nleft = nbn",
        "                nfirst = 1",
        "                nlast = 0",
        "                if nbn == 0:",
        "                    nlast = 1",
        "                st = 6",
        "                if bias != 0:",
        "                    st = 5",
        "            elif st == 5:",
        "                b64: int64 = bptr",
        f"                nb64: int64 = {S // 2}",
        "                io.rd_cmd.put((nb64 << 32) | b64)",
        f"                io.tag_out.put({1 | 8 | ((S // 2) << 8)})",
        f"                bptr = bptr + {4 * S}",
        "                st = 6",
        "            elif st == 6:",
        "                w64: int64 = wptr",
        f"                nw64: int64 = {S * S // 8}",
        "                io.rd_cmd.put((nw64 << 32) | w64)",
        f"                tg: int32 = {2 | 8 | ((S * S // 8) << 8)}",
        "                if nfirst != 0:",
        "                    st = 7",
        "                else:",
        "                    adv = 1",
        "                    if (final & tlast & klast & nlast) != 0:",
        f"                        tg = {2 | 4 | ((S * S // 8) << 8)}",
        "                io.tag_out.put(tg)",
        f"                wptr = wptr + {S * S}",
        "            else:",
        "                a64: int64 = aptr",
        "                na64: int64 = abeats",
        "                io.rd_cmd.put((na64 << 32) | a64)",
        "                tc: int32 = 3 | 8 | (abeats << 8)",
        "                if (final & tlast & klast & nlast) != 0:",
        "                    tc = 3 | 4 | (abeats << 8)",
        "                io.tag_out.put(tc)",
        "                aptr = aptr + (abeats << 3)",
        "                adv = 1",
        # a block is asked for: the next one, the next K block, the next tile
        "            if adv != 0:",
        "                st = 6",
        "                if nlast == 0:",
        "                    nfirst = 0",
        "                    if nleft == 1:",
        "                        nlast = 1",
        "                    nleft = nleft - 1",
        "                    if (bias & kfirst) != 0:",
        "                        st = 5",
        "                else:",
        "                    nleft = nbn",
        "                    nfirst = 1",
        "                    nlast = 0",
        "                    if nbn == 0:",
        "                        nlast = 1",
        "                    if klast == 0:",
        "                        kfirst = 0",
        "                        if kleft == 1:",
        "                            klast = 1",
        "                        kleft = kleft - 1",
        "                    elif tlast == 0:",
        "                        if tleft == 1:",
        "                            tlast = 1",
        "                        tleft = tleft - 1",
        "                        st = 4",
        "                    elif final != 0:",
        "                        go = 0",
        "                    else:",
        "                        st = 1",
        "    return req",
    ]
    return "\n".join(lines) + "\n"


def _deal_source(S):
    """The dealer: each beat memory answers with, to the stream that wants it.

    A request's tag says whose its beats are -- 0 an instruction's words, back
    to the requester; 1 a block's biases; 2 a weight block; 3 an activation
    chunk -- with bit 2 on the launch's last burst, bit 3 where another tag
    follows, and the beats above them: never fewer than two. The next tag is
    taken with a burst's last beat, so a burst follows the one before it with
    no cycle between them. An instruction is the exception: the requester
    cannot ask for what comes after it until it has all four words. Whether a
    beat is its burst's last is a flag kept as the count moves, so the beat's
    iteration compares nothing.

    A word is ``S`` bytes: a beat at 8, two beats at 16, the first kept in
    `lo`, and half a beat at 4, where the second half goes out the cycle after.
    """
    route = [
        "if kind == 3:",
        "    io.a_out.put(wd)",
        "else:",
        "    io.w_out.put(wd)",
    ]
    pad = " " * 20
    if S == 8:
        words = [f"{pad}wd: int8[8]", *_word(pad, "wd", ["beat"])]
        words += [pad + line for line in route] + [f"{pad}ends = lastb"]
    elif S == 16:
        words = [
            f"{pad}if part == 0:",
            f"{pad}    lo = beat",
            f"{pad}    part = 1",
            f"{pad}else:",
            f"{pad}    wd: int8[16]",
            *_word(pad + "    ", "wd", ["lo", "beat"]),
            *[pad + "    " + line for line in route],
            f"{pad}    part = 0",
            f"{pad}ends = lastb",
        ]
    elif S == 4:
        words = [f"{pad}wd: int8[4]"]
        words += [f"{pad}wd[{i}] = {_byte('beat', i)}" for i in range(4)]
        words += [pad + line for line in route]
        words += [f"{pad}lo = beat >> 32", f"{pad}endp = lastb", f"{pad}st = 2"]
    else:
        raise ValueError(f"no word packing for an array of {S}")
    lines = [
        "def make(IO):",
        "    def deal(io: IO):",
        "        kind: int32 = 0",
        "        n: int16 = 0",
        "        last: uint1 = 0",
        "        more: uint1 = 0",
        "        islast: uint1 = 0",
        "        st: int32 = 0",
        "        lo: int64 = 0",
        "        part: uint1 = 0",
        "        endp: uint1 = 0",
        "        go: uint1 = 1",
        "        while go != 0:",
        "            ends: uint1 = 0",
        "            if st == 0:",
        "                t0: int32 = io.tag_in.get()",
        "                kind = t0 & 3",
        "                last = (t0 >> 2) & 1",
        "                n = (t0 >> 8) & 255",
        "                more = (t0 >> 3) & 1",
        "                islast = 0",
        "                st = 1",
    ]
    if S == 4:
        lines += [
            "            elif st == 2:",
            "                wh: int8[4]",
            *[f"                wh[{i}] = {_byte('lo', i)}" for i in range(4)],
            "                if kind == 3:",
            "                    io.a_out.put(wh)",
            "                else:",
            "                    io.w_out.put(wh)",
            "                ends = endp",
            "                st = 1",
        ]
    lines += [
        "            else:",
        "                beat: int64 = io.rd_data.get()",
        "                lastb: uint1 = islast",
        "                islast = 0",
        "                if n == 2:",
        "                    islast = 1",
        "                n = n - 1",
        "                if kind == 0:",
        "                    io.ins_out.put(beat)",
        "                    ends = lastb",
        "                elif kind == 1:",
        "                    io.b_out.put(beat)",
        "                    ends = lastb",
        "                else:",
        *words,
        # a burst's last beat brings the next burst's tag with it
        "            if ends != 0:",
        "                if more != 0:",
        "                    t1: int32 = io.tag_in.get()",
        "                    kind = t1 & 3",
        "                    last = (t1 >> 2) & 1",
        "                    n = (t1 >> 8) & 255",
        "                    more = (t1 >> 3) & 1",
        "                    islast = 0",
        "                    st = 1",
        "                elif last != 0:",
        "                    go = 0",
        "                else:",
        "                    st = 0",
        "    return deal",
    ]
    return "\n".join(lines) + "\n"


def _head_source(S, buffer):
    """The head: one iteration a row, and ``S`` before the launch's first.

    Where the array is in a GEMM is four counters that run down to zero --
    the tiles, the K blocks, the column blocks and the rows left -- and the
    flags that say so, set as a counter moves. A row is decided by the flags
    alone, so nothing between one row and the next is longer than a counter's
    step.

    ``st`` is 0 before the launch's first instruction, 1 while a GEMM's bounds
    are taken up, and 2 for its rows. The launch's first block is preceded by
    ``S`` beats that carry its weights in: a block of ``S`` rows with no
    activations, `isopen`. The instruction after this one is read just before
    the last block's last ``S`` rows, which carry the next GEMM's first weights
    and its first biases, and reading it takes an iteration of its own: a row
    with nothing in it, which every cell of the array sees as a pause.

    A row of results goes into the row buffer, ``buffer`` rows deep, and the
    lanes cannot wait for room there. So the head counts the places down, and
    when they are gone it takes one of the packer's credits before each such
    row: it issues only rows the buffer is known to have a place for.
    """
    edge = 2 * S + 1
    lines = ["def make(IO):", "    def head(io: IO):"]
    lines += [f"        ab{i}: int8[{CHUNK}]" for i in range(S)]
    lines += [
        # the GEMM being run, and the one after it
        "        cfg: int64 = 0",
        "        final: uint1 = 0",
        "        biased: uint1 = 0",
        "        kbn: int32 = 0",
        "        nbn: int32 = 0",
        "        ln: int32 = 0",
        "        ksingle: uint1 = 0",
        "        nsingle: uint1 = 0",
        "        lsingle: uint1 = 0",
        "        ncfg: int64 = 0",
        "        nfinal: uint1 = 0",
        "        nbiased: uint1 = 0",
        "        nkbn: int32 = 0",
        "        nnbn: int32 = 0",
        "        nln: int32 = 0",
        "        ntn: int32 = 0",
        # where the array is
        "        tleft: int32 = 0",
        "        kleft: int32 = 0",
        "        nleft: int32 = 0",
        "        rl: int32 = 0",
        "        r: int32 = 0",
        "        tlast: uint1 = 0",
        "        kfirst: uint1 = 0",
        "        klast: uint1 = 0",
        "        nfirst: uint1 = 0",
        "        nlast: uint1 = 0",
        "        rfirst: uint1 = 0",
        "        tail: uint1 = 0",
        "        isopen: uint1 = 1",
        "        fetched: uint1 = 0",
        f"        space: int32 = {buffer}",
        "        noroom: uint1 = 0",
        "        bb: int64 = 0",
        "        st: int32 = 0",
        "        go: uint1 = 1",
        "        while go != 0:",
        "            lastblk: uint1 = nlast & klast & tlast",
        # an instruction: the launch's first, or the next before the last block hands over
        "            want: uint1 = 0",
        "            if st == 0:",
        "                want = 1",
        "            elif st == 2:",
        "                if (tail & lastblk) != 0:",
        "                    if (final | fetched | isopen) == 0:",
        "                        want = 1",
        "            if want != 0:",
        "                ins: int64 = io.op_in.get()",
        "                ncfg = ins & 63",
        "                nbiased = (ins >> 7) & 1",
        "                nfinal = (ins >> 8) & 1",
        "                nkbn = (ins >> 16) & 65535",
        "                nnbn = (ins >> 32) & 4095",
        "                nln = (ins >> 44) & 255",
        "                ntn = (ins >> 52) & 4095",
        "                fetched = 1",
        "                if st == 0:",
        "                    st = 1",
        "            elif st == 1:",
        "                cfg = ncfg",
        "                if nbiased != 0:",
        f"                    cfg = ncfg | {BIASED}",
        "                final = nfinal",
        "                biased = nbiased",
        "                kbn = nkbn",
        "                nbn = nnbn",
        "                ln = nln",
        "                tleft = ntn",
        "                tlast = 0",
        "                if ntn == 0:",
        "                    tlast = 1",
        "                kleft = nkbn",
        "                kfirst = 1",
        "                klast = 0",
        "                ksingle = 0",
        "                if nkbn == 0:",
        "                    klast = 1",
        "                    ksingle = 1",
        "                nleft = nnbn",
        "                nfirst = 1",
        "                nlast = 0",
        "                nsingle = 0",
        "                if nnbn == 0:",
        "                    nlast = 1",
        "                    nsingle = 1",
        "                lsingle = 0",
        f"                if nln == {S - 1}:",
        "                    lsingle = 1",
        "                r = 0",
        "                rfirst = 1",
        "                rl = nln",
        "                tail = lsingle",
        "                if isopen != 0:",
        f"                    rl = {S - 1}",
        "                    tail = 1",
        "                fetched = 0",
        "                st = 2",
        "            else:",
        "                blast: uint1 = 0",
        "                if rl == 0:",
        "                    blast = 1",
        # is there a block after this one, and does it want biases
        "                more: uint1 = 1",
        "                nextbias: uint1 = 0",
        "                if isopen != 0:",
        "                    nextbias = biased",
        "                elif nlast == 0:",
        "                    if kfirst != 0:",
        "                        nextbias = biased",
        "                elif klast != 0:",
        "                    if tlast == 0:",
        "                        nextbias = biased",
        "                    elif final == 0:",
        "                        nextbias = nbiased",
        "                    else:",
        "                        more = 0",
        "                carry: uint1 = tail & more",
        f"                e: int8[{edge}]",
        # the weight word: the next block's, on a block's last S rows
        "                if carry != 0:",
        f"                    wt: int8[{S}] = io.w_in.get()",
        *[f"                    e[{S + i}] = wt[{i}]" for i in range(S)],
        "                else:",
        *[f"                    e[{S + i}] = 0" for i in range(S)],
        # the activation word: from memory on the first column block, kept for the rest
        "                if isopen != 0:",
        *[f"                    e[{i}] = 0" for i in range(S)],
        "                elif nfirst != 0:",
        f"                    at: int8[{S}] = io.a_in.get()",
        *[f"                    ab{i}[r] = at[{i}]" for i in range(S)],
        *[f"                    e[{i}] = at[{i}]" for i in range(S)],
        "                else:",
        *[f"                    e[{i}] = ab{i}[r]" for i in range(S)],
        # a bias of the next block, one lane a row: two to a beat
        "                u: int64 = 0",
        "                if (carry & nextbias) != 0:",
        "                    imm: int64 = bb >> 32",
        "                    if (rl & 1) != 0:",
        "                        bb = io.b_in.get()",
        "                        l0: int64 = bb & 65535",
        "                        l1: int64 = (bb >> 16) & 65535",
        "                        imm = (((l1 ^ 32768) - 32768) << 16) | l0",
        f"                    q: int64 = {S - 1} - (rl & {S - 1})",
        f"                    u = (imm << 32) | {SETB} | (q << 13)",
        # a row of results needs a place in the row buffer: one of those it
        # started with, or one the packer has since emptied
        "                if isopen == 0:",
        "                    if klast != 0:",
        "                        if noroom != 0:",
        "                            paid: int8 = io.credit.get()",
        "                        else:",
        "                            if space == 1:",
        "                                noroom = 1",
        "                            space = space - 1",
        "                fin: uint1 = 0",
        "                if isopen == 0:",
        "                    fin = lastblk & blast & final",
        "                fl: int32 = 0",
        "                if blast != 0:",
        "                    fl = 1",
        "                if fin != 0:",
        "                    fl = fl | 2",
        f"                e[{2 * S}] = fl",
        "                io.e_out.put(e)",
        "                if isopen == 0:",
        f"                    u = u | {ROW} | cfg",
        "                    if kfirst != 0:",
        f"                        u = u | {FIRST}",
        "                        if (biased & rfirst) != 0:",
        f"                            u = u | {LOAD}",
        "                    if klast != 0:",
        f"                        u = u | {EMIT}",
        "                    if (nlast & blast) != 0:",
        f"                        u = u | {WRAP}",
        "                    if fin != 0:",
        f"                        u = u | {END}",
        "                io.u_out.put(u)",
        # advance
        "                if blast == 0:",
        "                    if isopen == 0:",
        f"                        if rl == {S}:",
        "                            tail = 1",
        "                        r = r + 1",
        "                        rfirst = 0",
        "                    rl = rl - 1",
        "                else:",
        "                    rl = ln",
        "                    tail = lsingle",
        "                    if isopen != 0:",
        "                        isopen = 0",
        "                    else:",
        "                        r = 0",
        "                        rfirst = 1",
        "                        if nlast == 0:",
        "                            nfirst = 0",
        "                            if nleft == 1:",
        "                                nlast = 1",
        "                            nleft = nleft - 1",
        "                        else:",
        "                            nleft = nbn",
        "                            nfirst = 1",
        "                            nlast = nsingle",
        "                            if klast == 0:",
        "                                kfirst = 0",
        "                                if kleft == 1:",
        "                                    klast = 1",
        "                                kleft = kleft - 1",
        "                            else:",
        "                                kleft = kbn",
        "                                kfirst = 1",
        "                                klast = ksingle",
        "                                if tlast == 0:",
        "                                    if tleft == 1:",
        "                                        tlast = 1",
        "                                    tleft = tleft - 1",
        "                                elif final != 0:",
        "                                    go = 0",
        "                                else:",
        "                                    st = 1",
        "    return head",
    ]
    return "\n".join(lines) + "\n"


def _wreq_source(S):
    """The write requester: a tile's results as bursts, a request ahead.

    The requester's token says where a tile's results go and how many rows
    they are. Each burst of them is asked for before the one before it has
    been acknowledged, so the packer's beats never wait for a request; the
    launch is done when its last burst is acknowledged.
    """
    shift = {4: ">> 1", 8: "", 16: "<< 1"}[S]
    lines = [
        "def make(IO, BURST):",
        "    def wreq(io: IO):",
        "        addr: int32 = 0",
        "        left: int32 = 0",
        "        final: uint1 = 0",
        "        owed: uint1 = 0",
        "        st: int32 = 0",
        "        go: uint1 = 1",
        "        while go != 0:",
        "            if st == 0:",
        "                c: int64 = io.y_cmd.get()",
        "                addr = c & 1073741823",
        f"                left = ((c >> 32) & 1073741823) {shift}",
        "                final = (c >> 62) & 1",
        "                st = 1",
        "            elif st == 1:",
        "                n: int32 = left",
        "                if left > BURST:",
        "                    n = BURST",
        "                nn: int64 = n",
        "                a64: int64 = addr",
        "                io.wr_cmd.put((nn << 32) | a64)",
        "                addr = addr + (n << 3)",
        "                left = left - n",
        "                st = 2",
        "            elif st == 2:",
        "                if owed != 0:",
        "                    ack: int64 = io.wr_ack.get()",
        "                owed = 1",
        "                st = 0",
        "                if left != 0:",
        "                    st = 1",
        "                elif final != 0:",
        "                    st = 3",
        "            else:",
        "                ack2: int64 = io.wr_ack.get()",
        "                io.done.put(1)",
        "                go = 0",
        "    return wreq",
    ]
    return "\n".join(lines) + "\n"


def _pack_source(S):
    """The packer: rows of results into beats.

    A row is ``S`` bytes and a flag, set on the launch's last. A 4-byte row is
    half a beat, kept in `lo` until its pair arrives; a 16-byte one is two
    beats, the second sent the cycle after. Every row taken pays the head a
    credit: there is one more place in the row buffer.
    """
    pad = " " * 12
    lines = ["def make(IO):", "    def pack(io: IO):"]
    tail = ["if fl != 0:", "    go = 0"]

    def beat(name, base, at):
        count = 8 if S > 4 else 4
        return (
            [f"{at}{name}_{i}: int64 = t[{base + i}] & 255" for i in range(count)]
            + [f"{at}{name}: int64 = 0"]
            + [f"{at}{name} = {name} | ({name}_{i} << {8 * i})" for i in range(count)]
        )

    if S == 8:
        lines += [
            "        go: uint1 = 1",
            "        while go != 0:",
            f"{pad}t: int8[9] = io.row_in.get()",
            f"{pad}io.credit.put(1)",
            *beat("w0", 0, pad),
            f"{pad}io.wr_data.put(w0)",
            f"{pad}fl: int32 = t[8]",
            *[pad + line for line in tail],
        ]
    elif S == 4:
        lines += [
            "        lo: int64 = 0",
            "        part: uint1 = 0",
            "        go: uint1 = 1",
            "        while go != 0:",
            f"{pad}t: int8[5] = io.row_in.get()",
            f"{pad}io.credit.put(1)",
            *beat("half", 0, pad),
            f"{pad}if part == 0:",
            f"{pad}    lo = half",
            f"{pad}    part = 1",
            f"{pad}else:",
            f"{pad}    io.wr_data.put(lo | (half << 32))",
            f"{pad}    part = 0",
            f"{pad}fl: int32 = t[4]",
            *[pad + line for line in tail],
        ]
    elif S == 16:
        lines += [
            "        hi: int64 = 0",
            "        fl: int32 = 0",
            "        st: uint1 = 0",
            "        go: uint1 = 1",
            "        while go != 0:",
            f"{pad}if st == 0:",
            f"{pad}    t: int8[17] = io.row_in.get()",
            f"{pad}    io.credit.put(1)",
            *beat("w0", 0, pad + "    "),
            *beat("w1", 8, pad + "    "),
            f"{pad}    io.wr_data.put(w0)",
            f"{pad}    hi = w1",
            f"{pad}    fl = t[16]",
            f"{pad}    st = 1",
            f"{pad}else:",
            f"{pad}    io.wr_data.put(hi)",
            f"{pad}    st = 0",
            *[pad + "    " + line for line in tail],
        ]
    else:
        raise ValueError(f"no beat packing for an array of {S}")
    lines += ["    return pack"]
    return "\n".join(lines) + "\n"


def mem_engine(S, nrd, nreq, nwreq, nwr, buffer=None):
    """The whole ``S x S`` engine. The lengths size the harness's tensors only.

    ``buffer`` is the rows of results that may wait for the write port, a
    tile's worth unless told otherwise: results then never hold the array.
    """
    if S not in (4, 8, 16):
        raise ValueError(
            f"the word packing is written for arrays of 4, 8 and 16, not {S}"
        )
    depth = OUTPUTS // S  # the rows a lane's accumulator holds
    buffer = buffer or depth
    edge = 2 * S + 1

    class PEIO(spmw.Interface):
        """The lean cell's links, every one a bare register."""

        a_in = spmw.In(int16, depth=0)
        a_out = spmw.Out(int16, depth=0)
        p_in = spmw.In(int32, depth=0)
        p_out = spmw.Out(int32, depth=0)
        w_in = spmw.In(int8, depth=0)
        w_out = spmw.Out(int8, depth=0)

    class ReqIO(spmw.Interface):
        launch = spmw.In(int64)  # where the program is
        ins_in = spmw.In(int64)  # an instruction's words, from the dealer
        rd_cmd = spmw.Out(int64)  # the read port's requests, beats << 32 | address
        tag_out = spmw.Out(int32)
        op_out = spmw.Out(int64)
        y_out = spmw.Out(int64)

    class DealIO(spmw.Interface):
        tag_in = spmw.In(int32, depth=4)
        rd_data = spmw.In(int64)  # the read port's beats
        ins_out = spmw.Out(int64)
        a_out = spmw.Out(int8[S])
        w_out = spmw.Out(int8[S])
        b_out = spmw.Out(int64)

    class HeadIO(spmw.Interface):
        # The dealer runs ahead into these: a chunk, some blocks, a block's
        # biases. They are what hides a load behind the arithmetic.
        op_in = spmw.In(int64)
        a_in = spmw.In(int8[S], depth=2 * CHUNK)
        w_in = spmw.In(int8[S], depth=4 * CHUNK)
        b_in = spmw.In(int64, depth=16)
        credit = spmw.In(int8, depth=buffer)
        e_out = spmw.Out(int8[edge])
        u_out = spmw.Out(int64)

    class QueueIO(spmw.Interface):
        # A micro-op is issued with its row and used when the row's sums reach
        # the lanes, `S` rows later: this is where it waits.
        u_in = spmw.In(int64, depth=3 * S + 8)
        u_out = spmw.Out(int64)

    class ETapIO(spmw.Interface):
        e_in = spmw.In(int8[edge], depth=0)
        e_out = spmw.Out(int8[edge], depth=0)
        a_out = spmw.Out(int16)
        w_out = spmw.Out(int8)

    class TapIO(spmw.Interface):
        u_in = spmw.In(int64)
        u_out = spmw.Out(int64)
        c_out = spmw.Out(int64)

    class LaneIO(spmw.Interface):
        c_in = spmw.In(int64)
        z_in = spmw.In(int32, depth=2)
        y_out = spmw.Out(int16)

    class CTapIO(spmw.Interface):
        y_in = spmw.In(int16)
        r_in = spmw.In(int8[S + 1])
        r_out = spmw.Out(int8[S + 1])

    class WReqIO(spmw.Interface):
        # A tile's token waits here while the tiles before it are written, so
        # the requester is never held by the write side.
        y_cmd = spmw.In(int64, depth=16)
        wr_ack = spmw.In(int64)
        wr_cmd = spmw.Out(int64)
        done = spmw.Out(int64)

    class PackIO(spmw.Interface):
        row_in = spmw.In(int8[S + 1], depth=buffer)
        wr_data = spmw.Out(int64)
        credit = spmw.Out(int8)

    mxu = spmw.Topology(
        PEIO,
        grid=(S, S),
        link=lambda i, j: {
            PEIO.a_out: spmw.to((i, j + 1), PEIO.a_in),
            PEIO.w_out: spmw.to((i, j + 1), PEIO.w_in),
            PEIO.p_out: spmw.to((i + 1, j), PEIO.p_in),
        },
    )
    erow = spmw.Topology(
        ETapIO,
        grid=(S,),
        link=lambda i: {ETapIO.e_out: spmw.to((i + 1,), ETapIO.e_in)},
    )
    urow = spmw.Topology(
        TapIO,
        grid=(S,),
        link=lambda i: {TapIO.u_out: spmw.to((i + 1,), TapIO.u_in)},
    )
    lanes = spmw.Topology(LaneIO, grid=(S,), link=lambda i: {})
    crow = spmw.Topology(
        CTapIO,
        grid=(S,),
        link=lambda i: {CTapIO.r_out: spmw.to((i + 1,), CTapIO.r_in)},
    )

    def one(iface):
        return spmw.Topology(iface, grid=(1,), link=lambda i: {})

    @spmw.unit
    def pe(io: PEIO):
        # `test_spmw_ptpu`'s cell: run by its tokens, counting nothing.
        nxt: int8 = 0
        cur: int8 = 0
        live: uint1 = 0
        go: uint1 = 1
        while go != 0:
            t: int16 = io.a_in.get()
            io.a_out.put(t)
            if live != 0:
                a: int8 = ((t & 255) ^ 128) - 128
                p = io.p_in.get()
                io.p_out.put(p + a * cur)
            io.w_out.put(nxt)
            nxt = io.w_in.get()
            if ((t >> 8) & 1) != 0:
                cur = nxt
                live = 1
            if ((t >> 9) & 1) != 0:
                go = 0

    @spmw.unit
    def etap(io: ETapIO):
        # A row's byte of each operand, and the rest passed down. The row it
        # is for is always element zero: every tap drops the one it used.
        go: uint1 = 1
        while go != 0:
            t: int8[edge] = io.e_in.get()
            f: int16 = t[2 * S] & 3
            a: int16 = t[0] & 255
            io.a_out.put(a | (f << 8))
            io.w_out.put(t[S])
            for k in range(S - 1):
                t[k] = t[k + 1]
                t[S + k] = t[S + k + 1]
            t[S - 1] = 0
            t[2 * S - 1] = 0
            io.e_out.put(t)
            if ((f >> 1) & 1) != 0:
                go = 0

    @spmw.unit
    def uq(io: QueueIO):
        go: uint1 = 1
        while go != 0:
            u: int64 = io.u_in.get()
            io.u_out.put(u)
            if ((u >> 10) & 1) != 0:
                go = 0

    @spmw.unit
    def tap(io: TapIO):
        go: uint1 = 1
        while go != 0:
            u: int64 = io.u_in.get()
            io.u_out.put(u)
            io.c_out.put(u)
            if ((u >> 10) & 1) != 0:
                go = 0

    @spmw.unit
    def lane(io: LaneIO, site: spmw.Site):
        # One partial sum a row, added into the accumulator at the row's place
        # in the K block's pass over the tile; `p` is that place. A bias comes
        # a block ahead and waits in `bnext` for its block's first row.
        (slot,) = site.rank
        buf: int32[depth]
        bnext: int32 = 0
        bias: int32 = 0
        p: int32 = 0
        go: uint1 = 1
        while go != 0:
            u: int64 = io.c_in.get()
            if ((u >> 9) & 1) != 0:
                bias = bnext
            if ((u >> 12) & 1) != 0:
                if ((u >> 13) & 31) == slot:
                    bnext = u >> 32
            if ((u >> 18) & 1) != 0:
                z: int32 = io.z_in.get()
                s: int32 = buf[p]
                if ((u >> 7) & 1) != 0:
                    s = 0
                    if ((u >> 6) & 1) != 0:
                        s = bias
                v: int32 = s + z
                buf[p] = v
                if ((u >> 8) & 1) != 0:
                    y: int32 = v
                    if ((u >> 5) & 1) != 0:
                        if y < 0:
                            y = 0
                    sh: int32 = u & 31
                    y = y >> sh
                    if y < -128:
                        y = -128
                    if y > INT8_MAX:
                        y = INT8_MAX
                    y = y & 255
                    if ((u >> 10) & 1) != 0:
                        y = y | 256
                    io.y_out.put(y)
                if ((u >> 11) & 1) != 0:
                    p = 0
                else:
                    p = p + 1
            if ((u >> 10) & 1) != 0:
                go = 0

    @spmw.unit
    def ctap(io: CTapIO, site: spmw.Site):
        (slot,) = site.rank
        go: uint1 = 1
        while go != 0:
            y: int16 = io.y_in.get()
            t: int8[S + 1] = io.r_in.get()
            t[slot] = ((y & 255) ^ 128) - 128
            t[S] = (y >> 8) & 1
            io.r_out.put(t)
            if ((y >> 8) & 1) != 0:
                go = 0

    req = _unit(f"req{S}", _req_source(S), IO=ReqIO)
    deal = _unit(f"deal{S}", _deal_source(S), IO=DealIO)
    head = _unit(f"head{S}", _head_source(S, buffer), IO=HeadIO)
    wreq = _unit(f"wreq{S}", _wreq_source(S), IO=WReqIO, BURST=BURST)
    pack = _unit(f"pack{S}", _pack_source(S), IO=PackIO)

    @spmw.fabric
    def engine(
        Launch: int64[1],
        RdData: int64[nrd],
        WrAck: int64[nwreq],
        RdCmd: int64[nreq],
        WrCmd: int64[nwreq],
        WrData: int64[nwr],
        Done: int64[1],
    ):
        P = spmw.place(pe, on=mxu)
        spmw.pipeline(P, ii=1, combinational=True)
        E = spmw.place(etap, on=erow)
        spmw.pipeline(E, ii=1, combinational=True)
        Rq = spmw.place(req, on=one(ReqIO))
        spmw.pipeline(Rq, ii=1)
        Dl = spmw.place(deal, on=one(DealIO))
        spmw.pipeline(Dl, ii=1)
        H = spmw.place(head, on=one(HeadIO))
        spmw.pipeline(H, ii=1)
        for i in range(S):
            spmw.ram(H, f"ab{i}")
        Q = spmw.place(uq, on=one(QueueIO))
        spmw.pipeline(Q, ii=1, combinational=True)
        T = spmw.place(tap, on=urow)
        spmw.pipeline(T, ii=1, combinational=True)
        V = spmw.place(lane, on=lanes)
        spmw.pipeline(V, ii=1)
        # A sum is read again a K block later: a tile's rows, never under four.
        spmw.ram(V, "buf", distance=4)
        C = spmw.place(ctap, on=crow)
        spmw.pipeline(C, ii=1, combinational=True)
        Wq = spmw.place(wreq, on=one(WReqIO))
        spmw.pipeline(Wq, ii=1)
        Pk = spmw.place(pack, on=one(PackIO))
        spmw.pipeline(Pk, ii=1)
        # the read port: requests one way, beats the other
        spmw.stream_in(Launch, into=Rq.launch, index=(...,))
        spmw.gather(RdCmd, from_=Rq.rd_cmd, index=(...,))
        spmw.stream_in(RdData, into=Dl.rd_data, index=(...,))
        spmw.link(Rq.tag_out, to=Dl.tag_in)
        spmw.link(Dl.ins_out, to=Rq.ins_in)
        spmw.link(Rq.op_out, to=H.op_in)
        spmw.link(Rq.y_out, to=Wq.y_cmd)
        spmw.link(Dl.a_out, to=H.a_in)
        spmw.link(Dl.w_out, to=H.w_in)
        spmw.link(Dl.b_out, to=H.b_in)
        # the array's edge, and its dispatch
        spmw.link(H.e_out, to=E.e_in)
        spmw.link(E.a_out, to=P.a_in)
        spmw.link(E.w_out, to=P.w_in)
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=V.z_in)
        spmw.link(H.u_out, to=Q.u_in)
        spmw.link(Q.u_out, to=T.u_in)
        spmw.link(T.c_out, to=V.c_in)
        # the results, and the write port
        spmw.link(V.y_out, to=C.y_in)
        spmw.stream_in(0, into=C.r_in)
        spmw.link(C.r_out, to=Pk.row_in)
        spmw.link(Pk.credit, to=H.credit)
        spmw.stream_in(WrAck, into=Wq.wr_ack, index=(...,))
        spmw.gather(WrCmd, from_=Wq.wr_cmd, index=(...,))
        spmw.gather(WrData, from_=Pk.wr_data, index=(...,))
        spmw.gather(Done, from_=Wq.done, index=(...,))

    engine.spmw_bind_mul_fabric = True
    return engine


# -- writing programs ---------------------------------------------------------


class Gemm:
    """One GEMM instruction and its operands.

    ``x`` is the tiles' activations, ``[T, L, K]``, and ``w`` their weights:
    ``[K, N]`` when every tile uses the same matrix, a layer's tokens taken
    ``L`` at a time, or ``[T, K, N]`` when each has its own, as E3's
    microbenchmark does. A bias is ``[N]``, and with it comes the ReLU or not.
    """

    def __init__(self, x, w, shift=0, bias=None, relu=False):
        self.x = np.asarray(x, dtype=np.int8)
        self.w = np.asarray(w, dtype=np.int8)
        self.shift = int(shift)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.int32)
        self.relu = relu

    @property
    def reuse(self):
        return self.w.ndim == 2

    def weights(self, tile):
        return self.w if self.reuse else self.w[tile]

    def golden(self):
        """The results, ``[T, L, N]`` int8."""
        outs = []
        for tile, x in enumerate(self.x):
            acc = x.astype(np.int64) @ self.weights(tile).astype(np.int64)
            if self.bias is not None:
                acc = acc + self.bias
            if self.relu:
                acc = np.maximum(acc, 0)
            outs.append(np.clip(acc >> self.shift, -128, INT8_MAX))
        return np.array(outs, dtype=np.int8)


class Launch:
    """A program and its operands in memory, and what the two ports carry.

    ``memory`` is the image the engine starts from and ``results`` the bytes
    it must have written, by address. ``rd_cmd`` and ``wr_cmd`` are the
    requests in the order the engine makes them, ``beats << 32 | address``;
    ``rd_data`` is the beats memory answers the reads with and ``wr_data``
    those the engine sends. A burst is at most `BURST` beats.
    """

    def __init__(self, S, gemms):
        self.size = S
        self.gemms = gemms
        image = bytearray(32 * len(gemms))
        reads, writes, results, rows = [], [], {}, 0

        def place(data):
            while len(image) % 8:
                image.append(0)
            at = len(image)
            image.extend(data)
            while len(image) % 8:
                image.append(0)
            return at

        def bursts(table, at, beats):
            while beats:
                n = min(beats, BURST)
                table.append((at, n))
                at, beats = at + 8 * n, beats - n

        for index, g in enumerate(gemms):
            T, L, K = g.x.shape
            N = g.w.shape[-1]
            KB, NB = K // S, N // S
            if K % S or N % S or L < S or L > CHUNK or (L * S) % 8:
                raise ValueError(
                    f"a {L} x {K} x {N} GEMM does not block onto {S} x {S}"
                )
            if NB * L > OUTPUTS // S:
                raise ValueError(
                    f"{NB * L} rows a tile is more than a lane accumulates"
                )
            if g.w.shape[-2] != K or (not g.reuse and len(g.w) != T):
                raise ValueError("the weights do not match the activations")
            # activations a K block at a time: [tile][kb][row][S]
            a = place(g.x.reshape(T, L, KB, S).transpose(0, 2, 1, 3).tobytes())
            # weights a block at a time, columns S-1 .. 0: [tile][kb][nb][column][S]
            w = np.stack([g.weights(t) for t in range(1 if g.reuse else T)])
            w = w.reshape(len(w), KB, S, NB, S).transpose(0, 1, 3, 4, 2)[
                :, :, :, ::-1, :
            ]
            w = place(np.ascontiguousarray(w).tobytes())
            b = 0 if g.bias is None else place(g.bias.astype("<i4").tobytes())
            want = g.golden().reshape(T, L, NB, S).transpose(0, 2, 1, 3)
            y = place(bytes(T * L * N))
            results[y] = np.ascontiguousarray(want).tobytes()
            flags = g.shift | (RELU if g.relu else 0) | (REUSE if g.reuse else 0)
            flags |= BIAS if g.bias is not None else 0
            flags |= FINAL if index == len(gemms) - 1 else 0
            if g.shift > 31 or KB > 1 << 16 or NB > 4096 or T > 4096:
                raise ValueError("a GEMM field does not fit its instruction")
            words = (
                ((T - 1) << 52)
                | ((L - 1) << 44)
                | ((NB - 1) << 32)
                | ((KB - 1) << 16)
                | flags,
                (w << 32) | a,
                (y << 32) | b,
                ((L * S // 8) << 16) | (NB * L),
            )
            image[32 * index : 32 * index + 32] = np.array(words, dtype="<u8").tobytes()
            # the requests: the instruction, then each tile a K block at a time
            bursts(reads, 32 * index, 4)
            aptr, wptr = a, w
            for tile in range(T):
                if g.reuse:
                    wptr = w
                for kb in range(KB):
                    for nb in range(NB):
                        if kb == 0 and g.bias is not None:
                            bursts(reads, b + 4 * S * nb, S // 2)
                        bursts(reads, wptr, S * S // 8)
                        wptr += S * S
                        if nb == 0:
                            bursts(reads, aptr, L * S // 8)
                            aptr += L * S
                bursts(writes, y + tile * L * N, L * N // 8)
            rows += T * KB * NB * L
        if len(image) > 1 << 30:
            raise ValueError("the launch does not fit the engine's 30-bit addresses")
        self.memory = np.frombuffer(bytes(image), dtype=np.uint8).copy()
        self.results = results
        self.rows = rows  # the launch's rows: its floor in cycles
        beats = self.memory.view("<u8")
        self.launch = np.zeros(1, dtype=np.int64)  # the program's address
        self.rd_cmd = np.array([(n << 32) | at for at, n in reads], dtype=np.int64)
        self.rd_data = (
            np.concatenate([beats[at // 8 : at // 8 + n] for at, n in reads])
            .astype(np.uint64)
            .view(np.int64)
        )
        self.wr_cmd = np.array([(n << 32) | at for at, n in writes], dtype=np.int64)
        final = self.memory.copy()
        for at, data in results.items():
            final[at : at + len(data)] = np.frombuffer(data, dtype=np.uint8)
        after = final.view("<u8")
        self.wr_data = (
            np.concatenate([after[at // 8 : at // 8 + n] for at, n in writes])
            .astype(np.uint64)
            .view(np.int64)
        )
        self.final = final


def mem_of(S, gemms, buffer=None):
    """The engine with one launch's memory traffic and golden attached."""
    launch = Launch(S, gemms)
    engine = mem_engine(
        S,
        len(launch.rd_data),
        len(launch.rd_cmd),
        len(launch.wr_cmd),
        len(launch.wr_data),
        buffer=buffer,
    )
    engine.spmw_operands = {
        "Launch": launch.launch,
        "RdData": launch.rd_data,
        "WrAck": np.ones(len(launch.wr_cmd), dtype=np.int64),
    }
    engine.spmw_expected = {
        "RdCmd": launch.rd_cmd,
        "WrCmd": launch.wr_cmd,
        "WrData": launch.wr_data,
        "Done": np.ones(1, dtype=np.int64),
    }
    engine.spmw_cosim_cycles = 4 * launch.rows + 2 * len(launch.rd_data) + 100000
    engine.spmw_mem = {"size": S, "launch": launch}
    return engine


# -- the workloads, as programs -----------------------------------------------


def micro_gemms(tiles=TILES, seed=0):
    """E3's microbenchmark: `tiles` 16x16x16 tiles, each its own weights."""
    A, B, bias, shift, _ = stimulus_mkn(tiles, 16, 16, 16, seed)
    return [Gemm(A, B, shift, bias=bias, relu=True)]


def llama_gemms(L=TOKENS, K=2048, n=SLICE):
    """The gate-and-up slice of `test_spmw_llama_ffn`: requantised, no bias."""
    x, w, shift, _shift2, _gu, _h = llama_stimulus(L, K, n)
    return [Gemm(x[None], w, shift)]


def small_gemm(T, L, K, N, seed, bias=False, relu=False, own=False):
    """A GEMM of random operands, shifted so that its results use int8's range."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (T, L, K)).astype(np.int8)
    w = rng.integers(-128, 128, (T, K, N) if own else (K, N)).astype(np.int8)
    b = rng.integers(-(1 << 20), 1 << 20, N).astype(np.int32) if bias else None
    return Gemm(x, w, shift=9 + int(np.log2(K)) // 2, bias=b, relu=relu)


def mixed_gemms(S):
    """Five GEMMs of five shapes back to back, down to one block of ``S`` rows."""
    return [
        micro_gemms(tiles=2)[0],
        small_gemm(1, S, S, S, 5),
        small_gemm(2, 16, 3 * S, 2 * S, 6, bias=True),
        llama_gemms(L=16, K=48, n=8)[0],
        small_gemm(3, S, S, 2 * S, 7, bias=True, relu=True, own=True),
    ]


def mem_workload(S, name):
    """One of the workloads E9 runs, as a launch of this engine."""
    gemms = {
        "micro": micro_gemms,
        "llama": llama_gemms,
        "dsv4": lambda: llama_gemms(K=7168),
        "mixed": lambda: mixed_gemms(S),
    }[name]()
    return mem_of(S, gemms)


# -- tests --------------------------------------------------------------------


def _run(engine, target):
    launch = engine.spmw_mem["launch"]
    outs = {
        "RdCmd": np.zeros(len(launch.rd_cmd), dtype=np.int64),
        "WrCmd": np.zeros(len(launch.wr_cmd), dtype=np.int64),
        "WrData": np.zeros(len(launch.wr_data), dtype=np.int64),
        "Done": np.zeros(1, dtype=np.int64),
    }
    ops = engine.spmw_operands
    spmw.build(engine, target=target)(
        ops["Launch"].copy(),
        ops["RdData"].copy(),
        ops["WrAck"].copy(),
        outs["RdCmd"],
        outs["WrCmd"],
        outs["WrData"],
        outs["Done"],
    )
    for name, want in engine.spmw_expected.items():
        np.testing.assert_array_equal(outs[name], want, err_msg=name)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_microbenchmark_runs_from_memory(target):
    _run(mem_of(4, micro_gemms(tiles=2)), target)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_a_layer_runs_from_memory(target):
    _run(mem_of(4, llama_gemms(L=16, K=32, n=8)), target)


@pytest.mark.parametrize("target", ["ref", "simulator"])
@pytest.mark.parametrize("size", [4, 8, 16])
def test_several_gemms_run_back_to_back(target, size):
    """Five instructions, five shapes, biases and none, shared weights and own."""
    _run(mem_of(size, mixed_gemms(size)), target)


@pytest.mark.parametrize("target", ["ref", "simulator"])
@pytest.mark.parametrize("size", [4, 16])
def test_results_wait_for_room_in_the_row_buffer(target, size):
    """A buffer far smaller than the results: every row after the first few
    is issued on a credit, and none is lost."""
    gemms = [small_gemm(3, 16, size, 2 * size, 8, bias=True), micro_gemms(tiles=2)[0]]
    _run(mem_of(size, gemms, buffer=4), target)


@pytest.mark.parametrize(
    "size, shape, why",
    [
        (8, (1, 16, 12, 8), "does not block"),  # K is not blocks of S
        (8, (1, 16, 16, 12), "does not block"),  # nor N
        (8, (1, 4, 8, 8), "does not block"),  # fewer rows than the array has
        (8, (1, 72, 8, 8), "does not block"),  # more than a chunk holds
        (4, (1, 5, 4, 4), "does not block"),  # rows that do not fill beats
        (8, (1, 64, 8, 2048), "more than a lane accumulates"),
    ],
)
def test_a_gemm_the_engine_cannot_hold_is_refused(size, shape, why):
    """The bounds are hardware: a chunk's rows, a tile's sums, a word's beats."""
    with pytest.raises(ValueError, match=why):
        Launch(size, [small_gemm(*shape, 1)])


def test_every_operand_crosses_the_port_once():
    """What the K-block order buys: no byte of a layer is asked for twice."""
    launch = Launch(8, llama_gemms(L=64, K=64, n=16))
    asked = np.zeros(len(launch.memory), dtype=np.int64)
    for token in launch.rd_cmd:
        at, beats = int(token) & 0xFFFFFFFF, int(token) >> 32
        asked[at : at + 8 * beats] += 1
    assert asked.max() == 1
    operands = 64 * 64 + 64 * 32
    assert asked.sum() == 32 + operands


def test_a_burst_is_at_most_256_beats():
    launch = Launch(16, llama_gemms(L=64, K=32, n=64))
    assert max(int(t) >> 32 for t in launch.rd_cmd) <= BURST
    assert max(int(t) >> 32 for t in launch.wr_cmd) <= BURST
    # 64 rows of 128 results is 8 KB: four bursts
    assert len(launch.wr_cmd) == 4
    # and a 64-row chunk of 16-byte words is one of 128 beats
    assert 128 in {int(t) >> 32 for t in launch.rd_cmd}

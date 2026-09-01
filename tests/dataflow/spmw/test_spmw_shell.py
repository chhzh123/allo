# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the host has to put in memory for the array's feeders to read.

`shell.host_buffer` is the only description of that layout, and until the array
was packaged as a kernel nothing called it -- so it had a bug that no test could
have caught, because no test existed. It assumed every channel carried a scalar.
Two of the six families do not: a weight cell holds `NW` int8 and a bias `NB`
int32, which the fabric carries as one wider word.

These tests are about the layout only. Whether the array computes the right
answer from it is settled elsewhere, by the cosim replay.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.spmw.shell import families, host_buffer

from test_spmw_transformer import BIG, matmul_prog, requant
from test_spmw_tpu_isa import NB, NW


def _operands():
    shape = BIG
    graph = spmw.elaborate(shape.engine)
    rng = np.random.default_rng(3)
    arrays = {
        "A": rng.integers(-8, 8, (shape.steps, shape.dim)).astype(np.int8),
        "W": rng.integers(-4, 4, (shape.dim, shape.dim, NW)).astype(np.int8),
        "Bias": rng.integers(-9, 9, (shape.dim, NB)).astype(np.int32),
        "MProg": matmul_prog(0, shape=shape),
        "VProg": requant(4, outs=shape.outs),
        "Y": np.zeros((shape.outs, shape.dim), dtype=np.int32),
    }
    return graph, arrays


def test_every_family_fills_exactly_its_buffer():
    """Channels times steps times the word, and not a byte more or less.

    A short buffer is the failure that does not announce itself: the feeder
    walks past the end of what the host wrote and the array consumes whatever
    was next in DRAM.
    """
    graph, arrays = _operands()
    for fam in families(graph):
        buf = host_buffer(fam, arrays)
        want = fam["channels"] * fam["steps"] * (fam["width"] // 8)
        assert isinstance(buf, bytes)
        assert len(buf) == want, fam["name"]


def test_a_packed_channel_carries_its_whole_vector():
    """The weight family is 32 bits of four int8, in element order.

    This is what the scalar assumption got wrong: it tried to store a
    four-element row into one slot and numpy refused, which at least failed
    loudly. Had the family been 8 bits wide it would have silently kept one
    weight in four.
    """
    graph, arrays = _operands()
    weights = [f for f in families(graph) if f["tensor"] == "W"]
    assert weights, "the engine should still load weights from memory"
    fam = weights[0]
    assert fam["width"] == 8 * NW, "a cell's weights travel as one word"

    buf = host_buffer(fam, arrays)
    word = fam["width"] // 8
    # Channel c at step t sits at (t * channels + c) * word, and holds the
    # NW weights of whichever cell the plan says.
    for channel, indices in enumerate(fam["plan"]):
        for step, index in enumerate(indices):
            offset = (step * fam["channels"] + channel) * word
            got = np.frombuffer(buf[offset : offset + word], dtype=np.int8)
            assert np.array_equal(got, arrays["W"][tuple(index)]), (channel, step)


def test_a_scalar_channel_is_unchanged_by_the_packing():
    """The activation family is one int8 per channel per step, as before."""
    graph, arrays = _operands()
    fam = [f for f in families(graph) if f["tensor"] == "A"][0]
    assert fam["width"] == 8
    buf = host_buffer(fam, arrays)
    got = np.frombuffer(buf, dtype=np.int8)
    for channel, indices in enumerate(fam["plan"]):
        for step, index in enumerate(indices):
            assert got[step * fam["channels"] + channel] == arrays["A"][tuple(index)]


def test_a_bias_word_is_two_int32():
    """The other packed family, and a different element size from the weights.

    Two families packing differently is what makes the rule "the token is the
    family's word" rather than "the token is four bytes".
    """
    graph, arrays = _operands()
    fam = [f for f in families(graph) if f["tensor"] == "Bias"][0]
    assert fam["width"] == 32 * NB
    buf = host_buffer(fam, arrays)
    word = fam["width"] // 8
    for channel, indices in enumerate(fam["plan"]):
        for step, index in enumerate(indices):
            offset = (step * fam["channels"] + channel) * word
            got = np.frombuffer(buf[offset : offset + word], dtype=np.int32)
            assert np.array_equal(got, arrays["Bias"][tuple(index)])


def test_a_tensor_whose_element_is_the_wrong_size_is_refused():
    """Silence here would put the array's operands one word out of step.

    The check has to be able to fire, so this hands it a real mismatch: a
    weight tensor built for a cell that holds half as many weights, which is
    what an `NW` the host and the fabric disagreed on would look like.
    """
    graph, arrays = _operands()
    fam = [f for f in families(graph) if f["tensor"] == "W"][0]
    assert NW > 2, "the mismatch below needs a narrower tensor to be possible"
    arrays["W"] = arrays["W"][..., : NW // 2].copy()
    with pytest.raises(ValueError, match="disagree"):
        host_buffer(fam, arrays)

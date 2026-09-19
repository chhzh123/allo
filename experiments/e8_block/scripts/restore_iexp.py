#!/usr/bin/env python3
"""Restore Gemmini's integer exponential, which ships commented out.

    restore_iexp.py <AccumulatorScale.scala>

`AccumulatorScale.iexp` takes `qln2` and `qln2_inv` and ignores them: its live
body is character-for-character `igelu`'s, and the real exponential -- the
I-BERT style `2^-z` decomposition -- sits above it inside a `/* ... */`. So the
SOFTMAX activation as shipped applies the erf polynomial to `e - max` and
scales by `inv_sum_exp`, which is not a softmax.

Both at `master` and at `8c3f992`, the commit this project builds.

Matching that bit-exactly would mean matching a function that is not the one
either design claims to compute, so the block workload restores it instead --
the same call E4 made about FEATHER's shipped controller, and recorded the same
way. The restored body is Gemmini's own code, uncommented, not a rewrite.
"""

import re
import sys

LIVE = """    import ev._

    val zero = q.zero
    def neg(x: T) = zero-x

    // qln2_inv needs scale to be 1 / (2 ** 16) / S
    // qln2_inv / S / (2 ** 16) = 1 / ln2
    // q * qln2_inv = x / S / ln2 * S * (2 ** 16) = x / ln2 * (2 ** 16)
    val neg_q_iexp = neg(q)
    val z_iexp = (neg_q_iexp * qln2_inv).asUInt.do_>>(16).asTypeOf(q) // q is non-positive
    val z_iexp_saturated = Wire(z_iexp.cloneType)
    z_iexp_saturated := Mux((5 until 16).map(z_iexp.asUInt(_)).reduce(_ | _), 32.S.asTypeOf(z_iexp), z_iexp)
    val qp_iexp = q.mac(z_iexp, qln2).withWidthOf(q)
    val q_poly_iexp = qc.mac(qp_iexp + qb, qp_iexp + qb).withWidthOf(q)
    (q_poly_iexp.asUInt.do_>>(z_iexp_saturated.asUInt)).asTypeOf(q)
  }
"""


def main():
    path = sys.argv[1]
    src = open(path, encoding="utf-8").read()
    start = src.index("  def iexp[T <: Data]")
    end = src.index("\n", src.index("  }}", start)) + 1
    body = src[start:end]
    if "/*" not in body:
        print("iexp already restored; nothing to do")
        return 0
    head = body[: body.index("{") + 1]
    src = src[:start] + head + "\n" + LIVE + src[end:]
    open(path, "w", encoding="utf-8").write(src)
    print("restored the integer exponential in iexp")
    return 0


if __name__ == "__main__":
    sys.exit(main())

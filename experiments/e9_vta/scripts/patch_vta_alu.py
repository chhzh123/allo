#!/usr/bin/env python3
"""E9 ablation: give VTA's ALU one more opcode, a multiply, and price it.

VTA already has a *variable* shift -- `io.a >> n`, shifting by the operand --
so a multiply is the only primitive missing for I-BERT's `iexp` and `igelu`,
which are what softmax and GELU need.  Guarded by `VTA_ALU_MUL` so the
baseline elaborates untouched by default; this turns "VTA cannot" into "VTA
plus N lookup tables could", which is a fair thing to say and a measurable
one.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/hc676/vta")
core = ROOT / "hardware/chisel/src/main/scala/core"

p = core / "TensorAlu.scala"
s = p.read_text()
old = """  val fop = Seq(Mux(io.a < io.b, io.a, io.b), Mux(io.a < io.b, io.b, io.a),
    io.a + io.b, io.a >> n, io.a << m)"""
new = """  // E9 ablation, guarded so the baseline is untouched by default.
  val withMul = sys.env.getOrElse("VTA_ALU_MUL", "0") != "0"
  val fop = Seq(Mux(io.a < io.b, io.a, io.b), Mux(io.a < io.b, io.b, io.a),
    io.a + io.b, io.a >> n, io.a << m) ++
    (if (withMul) Seq((io.a * io.b)(aluBits - 1, 0).asSInt) else Seq())"""
if "withMul" not in s:
    assert old in s, "fop sequence not found"
    s = s.replace(old, new)
    s = s.replace("  val opmux = Seq.tabulate(ALU_OP_NUM)(i => ALU_OP(i) -> fop(i))",
                  "  val opmux = fop.zipWithIndex.map { case (f, i) => ALU_OP(i) -> f }")
    p.write_text(s)
    print("TensorAlu.scala: multiply opcode added (guarded)")
else:
    print("TensorAlu.scala: already patched")

p = core / "ISA.scala"
s = p.read_text()
if "VTA_ALU_MUL" not in s:
    old = "  val ALU_OP_NUM = 5"
    assert old in s, "ALU_OP_NUM not found"
    s = s.replace(old,
                  '  val ALU_OP_NUM =\n'
                  '    if (sys.env.getOrElse("VTA_ALU_MUL", "0") != "0") 6 else 5')
    p.write_text(s)
    print("ISA.scala: ALU_OP_NUM made conditional")
else:
    print("ISA.scala: already patched")

import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    b: Stream[int32[2], 2][1]
    op_in: Stream[int32, 2][1]
    op_out: Stream[int32, 2][1]
    y_out: Stream[int32, 2][1]
    z_in: Stream[int32, 2][1]

    @df.kernel(mapping=[1])
    def vpu_r2():
        _st_b: int32[2] = b[0].get()
        prog: int32[16] = 0
        for pc in range(vprog_len):
            word: int32 = op_in[0].get()
            prog[pc] = word
            op_out[0].put(word)
        reg: int32[4] = 0
        for m in range(outs):
            for pc2 in range(vprog_len):
                word2: int32 = prog[pc2]
                opcode: int32 = word2 >> 24 & 255
                dst: int32 = word2 >> 20 & 15
                src: int32 = word2 >> 16 & 15
                imm: int32 = word2 & 65535
                if opcode == ACCZ:
                    zz: int32 = z_in[0].get()
                    reg[dst] = reg[dst] + zz
                elif opcode == LOADZ:
                    z2: int32 = z_in[0].get()
                    reg[dst] = z2
                elif opcode == LOADB:
                    reg[dst] = _st_b[src]
                elif opcode == LOADI:
                    reg[dst] = imm
                elif opcode == ADD:
                    reg[dst] = reg[dst] + reg[src]
                elif opcode == MUL:
                    reg[dst] = reg[dst] * reg[src]
                elif opcode == MAX:
                    if reg[src] > reg[dst]:
                        reg[dst] = reg[src]
                elif opcode == SHR:
                    reg[dst] = reg[dst] >> imm
                elif opcode == SUB:
                    reg[dst] = reg[dst] - reg[src]
                elif opcode == EXP2:
                    e: int32 = reg[dst]
                    if e < 0:
                        e = 0
                    if e > 30:
                        e = 30
                    reg[dst] = 1 << e
                elif opcode == RECIP:
                    d: int32 = reg[dst]
                    if d > 0:
                        reg[dst] = (1 << imm) // d
                    else:
                        reg[dst] = 0
                elif opcode == STORE:
                    y_out[0].put(reg[dst])

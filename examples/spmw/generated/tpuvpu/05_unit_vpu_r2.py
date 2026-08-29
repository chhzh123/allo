import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    b: Stream[int32, 2][1]
    op_in: Stream[int32, 2][1]
    op_out: Stream[int32, 2][1]
    y_out: Stream[int32, 2][1]
    z_in: Stream[int32, 2][1]

    @df.kernel(mapping=[1])
    def vpu_r2():
        _st_b: int32 = b[0].get()
        prog: int32[8] = 0
        for pc in range(NPROG):
            word: int32 = op_in[0].get()
            prog[pc] = word
            op_out[0].put(word)
        for m in range(MT):
            z: int32 = z_in[0].get()
            reg: int32[4] = 0
            for step in range(NPROG):
                word2: int32 = prog[step]
                opcode: int32 = word2 >> 24 & 255
                dst: int32 = word2 >> 20 & 15
                src: int32 = word2 >> 16 & 15
                imm: int32 = word2 & 65535
                if opcode == LOADZ:
                    reg[dst] = z
                elif opcode == LOADB:
                    reg[dst] = _st_b
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
                elif opcode == STORE:
                    y_out[0].put(reg[dst])

import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    w: Stream[_T21[4], 2][1]
    a_in: Stream[_T21, 2][1]
    a_out: Stream[_T21, 2][1]
    op_in: Stream[int32, 2][1]
    op_out: Stream[int32, 2][1]
    p_out: Stream[int32, 2][1]

    @df.kernel(mapping=[1])
    def mac_r2():
        _st_w: _T21[4] = w[0].get()
        for step in range(steps):
            word: int32 = op_in[0].get()
            op_out[0].put(word)
            opcode: int32 = word >> 24 & 255
            tile: int32 = word >> 16 & 255
            a = a_in[0].get()
            p = 0
            a_out[0].put(a)
            wt: int32 = _st_w[tile]
            if opcode == MACC:
                p_out[0].put(p + a * wt)
            elif opcode == MZERO:
                p_out[0].put(a * wt)
            else:
                p_out[0].put(p)

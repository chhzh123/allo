import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    c_in: Stream[int32, 2][1]
    c_out: Stream[int32, 2][1]
    north: Stream[int8, 2][1]
    south: Stream[int8, 2][1]
    west: Stream[int8, 2][1]
    _pid0: Stream[int32, 1][1]
    _pid1: Stream[int32, 1][1]

    @df.kernel(mapping=[1])
    def pe_r2():
        _st__pid0: int32 = _pid0[0].get()
        _st__pid1: int32 = _pid1[0].get()
        row, _col = (_st__pid0, _st__pid1)
        acc: int32 = 0
        for k in range(n):
            a = west[0].get()
            b = north[0].get()
            acc += a * b
            south[0].put(b)
        c_out[0].put(acc)
        for _i in range(row):
            c_out[0].put(c_in[0].get())

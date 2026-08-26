import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    c: Stream[int32, 2][1]
    east: Stream[_T3, 2][1]
    north: Stream[_T3, 2][1]
    west: Stream[_T3, 2][1]

    @df.kernel(mapping=[1])
    def pe_r2():
        acc: int32 = 0
        for k in range(N):
            a = west[0].get()
            b = north[0].get()
            acc += a * b
            east[0].put(a)
        c[0].put(acc)

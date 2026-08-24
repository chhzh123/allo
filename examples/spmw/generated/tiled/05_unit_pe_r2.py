import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    c: Stream[float32, 2][1]
    north: Stream[float32, 2][1]
    west: Stream[float32, 2][1]

    @df.kernel(mapping=[1])
    def pe_r2():
        acc: float32 = 0
        for k in range(K):
            a = west[0].get()
            b = north[0].get()
            acc += a * b
        c[0].put(acc)

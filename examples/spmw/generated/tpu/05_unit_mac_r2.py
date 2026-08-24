import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    w: Stream[int8, 2][1]
    a_in: Stream[int8, 2][1]
    a_out: Stream[int8, 2][1]
    p_out: Stream[_T5, 2][1]

    @df.kernel(mapping=[1])
    def mac_r2():
        _st_w: int8 = w[0].get()
        for m in range(MT):
            a = a_in[0].get()
            p = 0
            p_out[0].put(p + a * _st_w)
            a_out[0].put(a)

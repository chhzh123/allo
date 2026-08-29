import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    w: Stream[_T15[2], 2][1]
    a_in: Stream[_T15, 2][1]
    a_out: Stream[_T15, 2][1]
    p_out: Stream[int32, 2][1]

    @df.kernel(mapping=[1])
    def tiled_mac_r2():
        _st_w: _T15[2] = w[0].get()
        for m in range(MT):
            for t in range(NTILE):
                a = a_in[0].get()
                p = 0
                wt: int32 = _st_w[t]
                p_out[0].put(p + a * wt)
                a_out[0].put(a)

import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    lane: Stream[int8, 2][1]
    up: Stream[int8[4], 2][1]
    _pid0: Stream[int32, 1][1]

    @df.kernel(mapping=[1])
    def feed_3_r2():
        _st__pid0: int32 = _pid0[0].get()
        slot, = (_st__pid0,)
        for k in range(n):
            packed: int8[4] = up[0].get()
            lane[0].put(packed[slot])

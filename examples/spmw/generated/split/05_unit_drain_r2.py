import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    down: Stream[int32, 2][1]
    mine: Stream[int32, 2][1]
    up: Stream[int32, 2][1]
    _pid0: Stream[int32, 1][1]
    _pid1: Stream[int32, 1][1]

    @df.kernel(mapping=[1])
    def drain_r2():
        _st__pid0: int32 = _pid0[0].get()
        _st__pid1: int32 = _pid1[0].get()
        row, _col = (_st__pid0, _st__pid1)
        down[0].put(mine[0].get())
        for _i in range(row):
            down[0].put(up[0].get())

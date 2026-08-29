import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(Ct: int32[4, 4]):
    chan: Stream[int32, 2][1]
    _pid0: Stream[int32, 1][1]

    @df.kernel(mapping=[1], args=[Ct])
    def drain_down_drain_io(local_Ct: int32[4, 4]):
        _st__pid0: int32 = _pid0[0].get()
        _q0 = _st__pid0
        for _t in range(4):
            local_Ct[_q0, _t] = chan[0].get()

import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(Bt: int8[4, 4]):
    chan: Stream[int8[4], 2][1]

    @df.kernel(mapping=[1], args=[Bt])
    def feed_2_up_load_io(local_Bt: int8[4, 4]):
        _q0 = 0
        for _t in range(4):
            _blk: int8[4] = 0
            for _b0 in range(4):
                _blk[_b0] = local_Bt[_t, _b0]
            chan[0].put(_blk)

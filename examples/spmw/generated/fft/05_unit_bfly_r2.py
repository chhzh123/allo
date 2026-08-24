import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top():
    lo_in: Stream[_T4[2], 2][1]
    lo_out: Stream[_T4[2], 2][1]
    up_in: Stream[_T4[2], 2][1]
    up_out: Stream[_T4[2], 2][1]
    _pid0: Stream[_T6, 1][1]
    _pid1: Stream[_T6, 1][1]

    @df.kernel(mapping=[1])
    def bfly_r2():
        _st_tw: _T4[4, 2] = _ROM_brick1_5
        _st__pid0: _T6 = _pid0[0].get()
        _st__pid1: _T6 = _pid1[0].get()
        s, b = (_st__pid0, _st__pid1)
        span = 1 << s
        k = b % span * (HALF // span)
        wr = _st_tw[k, 0]
        wi = _st_tw[k, 1]
        a: _T4[2] = up_in[0].get()
        c: _T4[2] = lo_in[0].get()
        tr = wr * c[0] - wi * c[1]
        ti = wr * c[1] + wi * c[0]
        u: _T4[2] = 0
        l: _T4[2] = 0
        u[0] = a[0] + tr
        u[1] = a[1] + ti
        l[0] = a[0] - tr
        l[1] = a[1] - ti
        up_out[0].put(u)
        lo_out[0].put(l)

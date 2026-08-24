import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(X: _T4[8, 2], Y: _T4[8, 2]):
    bfly_key: Stream[_T4[2], 2][16]
    bfly_up_in_bind: Stream[_T4[2], 2][4]
    bfly_lo_in_bind: Stream[_T4[2], 2][4]
    bfly_up_out_bind: Stream[_T4[2], 2][4]
    bfly_lo_out_bind: Stream[_T4[2], 2][4]

    @df.kernel(mapping=[4], args=[X])
    def bfly_up_in_load(local_X: _T4[8, 2]):
        _q0 = df.get_pid()
        _ix0: _T6[1, 1] = _IX_bfly_up_in_load_5[_q0]
        for _t in range(1):
            _blk: _T4[2] = 0
            for _b0 in range(2):
                _blk[_b0] = local_X[_ix0[_t, 0], _b0]
            bfly_up_in_bind[_q0].put(_blk)

    @df.kernel(mapping=[4], args=[X])
    def bfly_lo_in_load(local_X: _T4[8, 2]):
        _q0 = df.get_pid()
        _ix0: _T6[1, 1] = _IX_bfly_lo_in_load_7[_q0]
        for _t in range(1):
            _blk: _T4[2] = 0
            for _b0 in range(2):
                _blk[_b0] = local_X[_ix0[_t, 0], _b0]
            bfly_lo_in_bind[_q0].put(_blk)

    @df.kernel(mapping=[3, 4])
    def bfly():
        _p0, _p1 = df.get_pid()
        _st_tw: _T4[4, 2] = _ROM_brick1_8
        with allo.meta_if(_ROLE_bfly_17[_p0][_p1] == 0):
            s, b = df.get_pid()
            span = 1 << s
            k = b % span * (HALF // span)
            wr = _st_tw[k, 0]
            wi = _st_tw[k, 1]
            a: _T4[2] = bfly_key[_CH_bfly_key_up_in_9[_p0][_p1]].get()
            c: _T4[2] = bfly_key[_CH_bfly_key_lo_in_10[_p0][_p1]].get()
            tr = wr * c[0] - wi * c[1]
            ti = wr * c[1] + wi * c[0]
            u: _T4[2] = 0
            l: _T4[2] = 0
            u[0] = a[0] + tr
            u[1] = a[1] + ti
            l[0] = a[0] - tr
            l[1] = a[1] - ti
            bfly_key[_CH_bfly_key_up_out_11[_p0][_p1]].put(u)
            bfly_key[_CH_bfly_key_lo_out_12[_p0][_p1]].put(l)
        with allo.meta_elif(_ROLE_bfly_17[_p0][_p1] == 1):
            s, b = df.get_pid()
            span = 1 << s
            k = b % span * (HALF // span)
            wr = _st_tw[k, 0]
            wi = _st_tw[k, 1]
            a: _T4[2] = bfly_key[_CH_bfly_key_up_in_13[_p0][_p1]].get()
            c: _T4[2] = bfly_key[_CH_bfly_key_lo_in_14[_p0][_p1]].get()
            tr = wr * c[0] - wi * c[1]
            ti = wr * c[1] + wi * c[0]
            u: _T4[2] = 0
            l: _T4[2] = 0
            u[0] = a[0] + tr
            u[1] = a[1] + ti
            l[0] = a[0] - tr
            l[1] = a[1] - ti
            bfly_up_out_bind[_p1].put(u)
            bfly_lo_out_bind[_p1].put(l)
        with allo.meta_else():
            s, b = df.get_pid()
            span = 1 << s
            k = b % span * (HALF // span)
            wr = _st_tw[k, 0]
            wi = _st_tw[k, 1]
            a: _T4[2] = bfly_up_in_bind[_p1].get()
            c: _T4[2] = bfly_lo_in_bind[_p1].get()
            tr = wr * c[0] - wi * c[1]
            ti = wr * c[1] + wi * c[0]
            u: _T4[2] = 0
            l: _T4[2] = 0
            u[0] = a[0] + tr
            u[1] = a[1] + ti
            l[0] = a[0] - tr
            l[1] = a[1] - ti
            bfly_key[_CH_bfly_key_up_out_15[_p0][_p1]].put(u)
            bfly_key[_CH_bfly_key_lo_out_16[_p0][_p1]].put(l)

    @df.kernel(mapping=[4], args=[Y])
    def bfly_up_out_drain(local_Y: _T4[8, 2]):
        _q0 = df.get_pid()
        _ix0: _T6[1, 1] = _IX_bfly_up_out_drain_18[_q0]
        for _t in range(1):
            _blk: _T4[2] = bfly_up_out_bind[_q0].get()
            for _b0 in range(2):
                local_Y[_ix0[_t, 0], _b0] = _blk[_b0]

    @df.kernel(mapping=[4], args=[Y])
    def bfly_lo_out_drain(local_Y: _T4[8, 2]):
        _q0 = df.get_pid()
        _ix0: _T6[1, 1] = _IX_bfly_lo_out_drain_19[_q0]
        for _t in range(1):
            _blk: _T4[2] = bfly_lo_out_bind[_q0].get()
            for _b0 in range(2):
                local_Y[_ix0[_t, 0], _b0] = _blk[_b0]
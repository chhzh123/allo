import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(At: int8[4, 4], Bt: int8[4, 4], Ct: int32[4, 4]):
    pe_east_west: Stream[int8, 2][4, 4]
    pe_south_north: Stream[int8, 2][4, 4]
    pe_c_out_c_in: Stream[int32, 2][4, 4]
    feed_down_up: Stream[int8[4], 2][4]
    feed_2_down_up: Stream[int8[4], 2][4]
    feed_up_bind: Stream[int8[4], 2][1]
    feed_2_up_bind: Stream[int8[4], 2][1]
    pe_west_bind: Stream[int8, 2][4]
    pe_north_bind: Stream[int8, 2][4]
    pe_c_out_bind: Stream[int32, 2][4]

    @df.kernel(mapping=[1], args=[At])
    def feed_up_load(local_At: int8[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            _blk: int8[4] = 0
            for _b0 in range(4):
                _blk[_b0] = local_At[_t, _b0]
            feed_up_bind[_q0].put(_blk)

    @df.kernel(mapping=[1], args=[Bt])
    def feed_2_up_load(local_Bt: int8[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            _blk: int8[4] = 0
            for _b0 in range(4):
                _blk[_b0] = local_Bt[_t, _b0]
            feed_2_up_bind[_q0].put(_blk)

    @df.kernel(mapping=[4, 4])
    def pe():
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_pe_4[_p0][_p1] == 0):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_east_west[_p0, _p1].get()
                b = pe_north_bind[_p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(0)
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 1):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(pe_c_out_c_in[_p0, _p1].get())
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 2):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(pe_c_out_c_in[_p0, _p1].get())
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 3):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
            pe_c_out_bind[_p1].put(acc)
            for _i in range(row):
                pe_c_out_bind[_p1].put(pe_c_out_c_in[_p0, _p1].get())
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 4):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_west_bind[_p0].get()
                b = pe_north_bind[_p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(0)
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 5):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_east_west[_p0, _p1].get()
                b = pe_north_bind[_p1].get()
                acc += a * b
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(0)
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 6):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_west_bind[_p0].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(pe_c_out_c_in[_p0, _p1].get())
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 7):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(pe_c_out_c_in[_p0, _p1].get())
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 8):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_west_bind[_p0].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(pe_c_out_c_in[_p0, _p1].get())
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 9):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_south_north[_p0 + 1, _p1].put(b)
            pe_c_out_c_in[_p0 + 1, _p1].put(acc)
            for _i in range(row):
                pe_c_out_c_in[_p0 + 1, _p1].put(pe_c_out_c_in[_p0, _p1].get())
        with allo.meta_elif(_ROLE_pe_4[_p0][_p1] == 10):
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_west_bind[_p0].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
            pe_c_out_bind[_p1].put(acc)
            for _i in range(row):
                pe_c_out_bind[_p1].put(pe_c_out_c_in[_p0, _p1].get())
        with allo.meta_else():
            row, _col = df.get_pid()
            acc: int32 = 0
            for k in range(n):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
            pe_c_out_bind[_p1].put(acc)
            for _i in range(row):
                pe_c_out_bind[_p1].put(pe_c_out_c_in[_p0, _p1].get())

    @df.kernel(mapping=[4])
    def feed():
        _p0 = df.get_pid()
        with allo.meta_if(_ROLE_feed_5[_p0] == 0):
            slot, = df.get_pid()
            for k in range(n):
                packed: int8[4] = feed_down_up[_p0].get()
                pe_west_bind[_p0].put(packed[slot])
                feed_down_up[_p0 + 1].put(packed)
        with allo.meta_elif(_ROLE_feed_5[_p0] == 1):
            slot, = df.get_pid()
            for k in range(n):
                packed: int8[4] = feed_up_bind[0].get()
                pe_west_bind[_p0].put(packed[slot])
                feed_down_up[_p0 + 1].put(packed)
        with allo.meta_else():
            slot, = df.get_pid()
            for k in range(n):
                packed: int8[4] = feed_down_up[_p0].get()
                pe_west_bind[_p0].put(packed[slot])

    @df.kernel(mapping=[4])
    def feed_2():
        _p0 = df.get_pid()
        with allo.meta_if(_ROLE_feed_2_6[_p0] == 0):
            slot, = df.get_pid()
            for k in range(n):
                packed: int8[4] = feed_2_down_up[_p0].get()
                pe_north_bind[_p0].put(packed[slot])
                feed_2_down_up[_p0 + 1].put(packed)
        with allo.meta_elif(_ROLE_feed_2_6[_p0] == 1):
            slot, = df.get_pid()
            for k in range(n):
                packed: int8[4] = feed_2_up_bind[0].get()
                pe_north_bind[_p0].put(packed[slot])
                feed_2_down_up[_p0 + 1].put(packed)
        with allo.meta_else():
            slot, = df.get_pid()
            for k in range(n):
                packed: int8[4] = feed_2_down_up[_p0].get()
                pe_north_bind[_p0].put(packed[slot])

    @df.kernel(mapping=[4], args=[Ct])
    def pe_c_out_drain(local_Ct: int32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            local_Ct[_q0, _t] = pe_c_out_bind[_q0].get()
import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(A: _T2[3, 3], B: _T2[3, 3], C: int32[3, 3]):
    pe_east_west: Stream[_T2, 2][3, 3]
    pe_south_north: Stream[_T2, 2][3, 3]
    pe_west_bind: Stream[_T2, 2][3]
    pe_north_bind: Stream[_T2, 2][3]

    @df.kernel(mapping=[3], args=[A])
    def pe_west_load(local_A: _T2[3, 3]):
        _q0 = df.get_pid()
        for _t in range(3):
            pe_west_bind[_q0].put(local_A[_q0, _t])

    @df.kernel(mapping=[3], args=[B])
    def pe_north_load(local_B: _T2[3, 3]):
        _q0 = df.get_pid()
        for _t in range(3):
            pe_north_bind[_q0].put(local_B[_t, _q0])

    @df.kernel(mapping=[3, 3], args=[C])
    def pe(local_C: int32[3, 3]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_pe_3[_p0][_p1] == 0):
            acc: int32 = 0
            for k in range(size):
                a = pe_west_bind[_p0].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 1):
            acc: int32 = 0
            for k in range(size):
                a = pe_west_bind[_p0].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 2):
            acc: int32 = 0
            for k in range(size):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 3):
            acc: int32 = 0
            for k in range(size):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 4):
            acc: int32 = 0
            for k in range(size):
                a = pe_west_bind[_p0].get()
                b = pe_north_bind[_p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 5):
            acc: int32 = 0
            for k in range(size):
                a = pe_east_west[_p0, _p1].get()
                b = pe_north_bind[_p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 6):
            acc: int32 = 0
            for k in range(size):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 7):
            acc: int32 = 0
            for k in range(size):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
            local_C[_p0, _p1] = acc
        with allo.meta_else():
            acc: int32 = 0
            for k in range(size):
                a = pe_east_west[_p0, _p1].get()
                b = pe_north_bind[_p1].get()
                acc += a * b
                pe_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1] = acc
import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(A: float32[4, 4], B: float32[4, 4], C: float32[4, 4]):
    pe_east_west: Stream[float32, 2][2, 2]
    pe_south_north: Stream[float32, 2][2, 2]
    pe_1_east_west: Stream[float32, 2][2, 2]
    pe_1_south_north: Stream[float32, 2][2, 2]
    pe_2_east_west: Stream[float32, 2][2, 2]
    pe_2_south_north: Stream[float32, 2][2, 2]
    pe_3_east_west: Stream[float32, 2][2, 2]
    pe_3_south_north: Stream[float32, 2][2, 2]
    pe_west_bind: Stream[float32, 2][2]
    pe_north_bind: Stream[float32, 2][2]
    pe_1_west_bind: Stream[float32, 2][2]
    pe_1_north_bind: Stream[float32, 2][2]
    pe_2_west_bind: Stream[float32, 2][2]
    pe_2_north_bind: Stream[float32, 2][2]
    pe_3_west_bind: Stream[float32, 2][2]
    pe_3_north_bind: Stream[float32, 2][2]

    @df.kernel(mapping=[2], args=[A])
    def pe_west_load(local_A: float32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            pe_west_bind[_q0].put(local_A[_q0, _t])

    @df.kernel(mapping=[2], args=[B])
    def pe_north_load(local_B: float32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            pe_north_bind[_q0].put(local_B[_t, _q0])

    @df.kernel(mapping=[2], args=[A])
    def pe_1_west_load(local_A: float32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            pe_1_west_bind[_q0].put(local_A[_q0, _t])

    @df.kernel(mapping=[2], args=[B])
    def pe_1_north_load(local_B: float32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            pe_1_north_bind[_q0].put(local_B[_t, _q0 + 2])

    @df.kernel(mapping=[2], args=[A])
    def pe_2_west_load(local_A: float32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            pe_2_west_bind[_q0].put(local_A[_q0 + 2, _t])

    @df.kernel(mapping=[2], args=[B])
    def pe_2_north_load(local_B: float32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            pe_2_north_bind[_q0].put(local_B[_t, _q0])

    @df.kernel(mapping=[2], args=[A])
    def pe_3_west_load(local_A: float32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            pe_3_west_bind[_q0].put(local_A[_q0 + 2, _t])

    @df.kernel(mapping=[2], args=[B])
    def pe_3_north_load(local_B: float32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            pe_3_north_bind[_q0].put(local_B[_t, _q0 + 2])

    @df.kernel(mapping=[2, 2], args=[C])
    def pe(local_C: float32[4, 4]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_pe_3[_p0][_p1] == 0):
            acc: float32 = 0
            for k in range(K):
                a = pe_west_bind[_p0].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 1):
            acc: float32 = 0
            for k in range(K):
                a = pe_west_bind[_p0].get()
                b = pe_north_bind[_p1].get()
                acc += a * b
                pe_east_west[_p0, _p1 + 1].put(a)
                pe_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1] = acc
        with allo.meta_elif(_ROLE_pe_3[_p0][_p1] == 2):
            acc: float32 = 0
            for k in range(K):
                a = pe_east_west[_p0, _p1].get()
                b = pe_south_north[_p0, _p1].get()
                acc += a * b
            local_C[_p0, _p1] = acc
        with allo.meta_else():
            acc: float32 = 0
            for k in range(K):
                a = pe_east_west[_p0, _p1].get()
                b = pe_north_bind[_p1].get()
                acc += a * b
                pe_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1] = acc

    @df.kernel(mapping=[2, 2], args=[C])
    def pe_1(local_C: float32[4, 4]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_pe_1_4[_p0][_p1] == 0):
            acc: float32 = 0
            for k in range(K):
                a = pe_1_west_bind[_p0].get()
                b = pe_1_south_north[_p0, _p1].get()
                acc += a * b
                pe_1_east_west[_p0, _p1 + 1].put(a)
            local_C[_p0, _p1 + 2] = acc
        with allo.meta_elif(_ROLE_pe_1_4[_p0][_p1] == 1):
            acc: float32 = 0
            for k in range(K):
                a = pe_1_west_bind[_p0].get()
                b = pe_1_north_bind[_p1].get()
                acc += a * b
                pe_1_east_west[_p0, _p1 + 1].put(a)
                pe_1_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1 + 2] = acc
        with allo.meta_elif(_ROLE_pe_1_4[_p0][_p1] == 2):
            acc: float32 = 0
            for k in range(K):
                a = pe_1_east_west[_p0, _p1].get()
                b = pe_1_south_north[_p0, _p1].get()
                acc += a * b
            local_C[_p0, _p1 + 2] = acc
        with allo.meta_else():
            acc: float32 = 0
            for k in range(K):
                a = pe_1_east_west[_p0, _p1].get()
                b = pe_1_north_bind[_p1].get()
                acc += a * b
                pe_1_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0, _p1 + 2] = acc

    @df.kernel(mapping=[2, 2], args=[C])
    def pe_2(local_C: float32[4, 4]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_pe_2_5[_p0][_p1] == 0):
            acc: float32 = 0
            for k in range(K):
                a = pe_2_west_bind[_p0].get()
                b = pe_2_south_north[_p0, _p1].get()
                acc += a * b
                pe_2_east_west[_p0, _p1 + 1].put(a)
            local_C[_p0 + 2, _p1] = acc
        with allo.meta_elif(_ROLE_pe_2_5[_p0][_p1] == 1):
            acc: float32 = 0
            for k in range(K):
                a = pe_2_west_bind[_p0].get()
                b = pe_2_north_bind[_p1].get()
                acc += a * b
                pe_2_east_west[_p0, _p1 + 1].put(a)
                pe_2_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0 + 2, _p1] = acc
        with allo.meta_elif(_ROLE_pe_2_5[_p0][_p1] == 2):
            acc: float32 = 0
            for k in range(K):
                a = pe_2_east_west[_p0, _p1].get()
                b = pe_2_south_north[_p0, _p1].get()
                acc += a * b
            local_C[_p0 + 2, _p1] = acc
        with allo.meta_else():
            acc: float32 = 0
            for k in range(K):
                a = pe_2_east_west[_p0, _p1].get()
                b = pe_2_north_bind[_p1].get()
                acc += a * b
                pe_2_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0 + 2, _p1] = acc

    @df.kernel(mapping=[2, 2], args=[C])
    def pe_3(local_C: float32[4, 4]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_pe_3_6[_p0][_p1] == 0):
            acc: float32 = 0
            for k in range(K):
                a = pe_3_west_bind[_p0].get()
                b = pe_3_south_north[_p0, _p1].get()
                acc += a * b
                pe_3_east_west[_p0, _p1 + 1].put(a)
            local_C[_p0 + 2, _p1 + 2] = acc
        with allo.meta_elif(_ROLE_pe_3_6[_p0][_p1] == 1):
            acc: float32 = 0
            for k in range(K):
                a = pe_3_west_bind[_p0].get()
                b = pe_3_north_bind[_p1].get()
                acc += a * b
                pe_3_east_west[_p0, _p1 + 1].put(a)
                pe_3_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0 + 2, _p1 + 2] = acc
        with allo.meta_elif(_ROLE_pe_3_6[_p0][_p1] == 2):
            acc: float32 = 0
            for k in range(K):
                a = pe_3_east_west[_p0, _p1].get()
                b = pe_3_south_north[_p0, _p1].get()
                acc += a * b
            local_C[_p0 + 2, _p1 + 2] = acc
        with allo.meta_else():
            acc: float32 = 0
            for k in range(K):
                a = pe_3_east_west[_p0, _p1].get()
                b = pe_3_north_bind[_p1].get()
                acc += a * b
                pe_3_south_north[_p0 + 1, _p1].put(b)
            local_C[_p0 + 2, _p1 + 2] = acc
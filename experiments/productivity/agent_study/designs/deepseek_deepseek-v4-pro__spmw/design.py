import allo.spmw as spmw
from allo.ir.types import int8, int32

N = 8


class PEIO(spmw.Interface):
    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    b_in = spmw.In(int8)
    b_out = spmw.Out(int8)
    c_in = spmw.In(int32)
    c_out = spmw.Out(int32)


topo = spmw.Topology(
    PEIO, grid=(N, N),
    link=lambda i, j: {
        PEIO.a_out: spmw.to((i, j+1), PEIO.a_in),
        PEIO.b_out: spmw.to((i+1, j), PEIO.b_in),
        PEIO.c_out: spmw.to((i, j+1), PEIO.c_in),
    }
)


@spmw.unit
def pe(io: PEIO, site: spmw.Site):
    acc: int32 = 0
    for k in range(N):
        b = io.b_in.get()
        a = io.a_in.get()
        acc = acc + a * b
        io.b_out.put(b)
        io.a_out.put(a)

    col: int32 = site.rank[1]
    for _ in range(col):
        v = io.c_in.get()
        io.c_out.put(v)
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.c_in,))
def pe_col0(io: PEIO, site: spmw.Site):
    acc: int32 = 0
    for k in range(N):
        b = io.b_in.get()
        a = io.a_in.get()
        acc = acc + a * b
        io.b_out.put(b)
        io.a_out.put(a)

    io.c_out.put(acc)


@spmw.fabric
def engine(A: int8[N, N], B: int8[N, N], C: int32[N, N]):
    P = spmw.place(pe, on=topo)
    spmw.stream_in(A, into=P.a_in, index=(P.rows, ...))
    spmw.stream_in(B, into=P.b_in, index=(..., P.cols))
    spmw.gather(C, from_=P.c_out, index=(P.rows, ...))
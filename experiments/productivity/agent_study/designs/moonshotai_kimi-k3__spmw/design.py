# Fill in the body. Do not change the signature of `engine`.
import allo.spmw as spmw
from allo.ir.types import int8, int16, int32

N = 8


class PEIO(spmw.Interface):
    a_in = spmw.In(int8)      # A value arriving from the west
    a_out = spmw.Out(int8)    # A value forwarded east
    b_in = spmw.In(int8)      # B value arriving from the north
    b_out = spmw.Out(int8)    # B value forwarded south
    c_in = spmw.In(int32)     # result chain arriving from the west
    c_out = spmw.Out(int32)   # result chain forwarded east


mesh = spmw.Topology(
    PEIO,
    grid=(N, N),
    link=lambda i, j: {
        PEIO.a_out: spmw.to((i, j + 1), PEIO.a_in),
        PEIO.b_out: spmw.to((i + 1, j), PEIO.b_in),
        PEIO.c_out: spmw.to((i, j + 1), PEIO.c_in),
    },
)


@spmw.unit
def pe(io: PEIO, site: spmw.Site):
    # interior PE: every port bound
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        aw: int16 = a
        bw: int16 = b
        acc = acc + aw * bw
        io.a_out.put(a)
        io.b_out.put(b)
    # drain: forward the j results from the west, then our own
    for _f in range(site.rank[1]):
        io.c_out.put(io.c_in.get())
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.c_in,))
def pe_west(io: PEIO, site: spmw.Site):
    # west edge (j=0): nothing to forward, never touch c_in
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        aw: int16 = a
        bw: int16 = b
        acc = acc + aw * bw
        io.a_out.put(a)
        io.b_out.put(b)
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.a_out,))
def pe_east(io: PEIO, site: spmw.Site):
    # east edge (j=7): A falls off the grid, never touch a_out
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        aw: int16 = a
        bw: int16 = b
        acc = acc + aw * bw
        io.b_out.put(b)
    for _f in range(site.rank[1]):
        io.c_out.put(io.c_in.get())
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.b_out,))
def pe_south(io: PEIO, site: spmw.Site):
    # south edge (i=7): B falls off the grid, never touch b_out
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        aw: int16 = a
        bw: int16 = b
        acc = acc + aw * bw
        io.a_out.put(a)
    for _f in range(site.rank[1]):
        io.c_out.put(io.c_in.get())
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.a_out, PEIO.b_out))
def pe_se(io: PEIO, site: spmw.Site):
    # south-east corner (7,7): neither a_out nor b_out
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        aw: int16 = a
        bw: int16 = b
        acc = acc + aw * bw
    for _f in range(site.rank[1]):
        io.c_out.put(io.c_in.get())
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.c_in, PEIO.b_out))
def pe_sw(io: PEIO, site: spmw.Site):
    # south-west corner (7,0): no c_in, no b_out
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        aw: int16 = a
        bw: int16 = b
        acc = acc + aw * bw
        io.a_out.put(a)
    io.c_out.put(acc)


@spmw.fabric
def engine(A: int8[N, N], B: int8[N, N], C: int32[N, N]):
    """A[i][k] arrives on input port i, B[k][j] on input port j,
    and C[i][j] must leave on output port i, in ascending j."""
    P = spmw.place(pe, on=mesh)
    spmw.stream_in(A, into=P.a_in, index=(P.rows, ...))
    spmw.stream_in(B, into=P.b_in, index=(..., P.cols))
    spmw.gather(C, from_=P.c_out, index=(P.rows, ...))

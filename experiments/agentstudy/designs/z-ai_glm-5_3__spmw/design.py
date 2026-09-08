import allo.spmw as spmw
from allo.ir.types import int8, int32

N = 8


class PEIO(spmw.Interface):
    a_in = spmw.In(int8)    # A value arriving from the west
    b_in = spmw.In(int8)    # B value arriving from the north
    a_out = spmw.Out(int8)  # A value passed to the eastern neighbour
    b_out = spmw.Out(int8)  # B value passed to the southern neighbour
    c_in = spmw.In(int32)   # result arriving from the western neighbour
    c_out = spmw.Out(int32)  # result passed to the eastern neighbour


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
    """Output-stationary PE(i,j).

    Consumes the eight A/B operand pairs of one product, multiplying and
    accumulating its own C[i][j] while passing A east and B south.  During
    the last j iterations it also forwards, one hop eastward, the j results
    its row's western neighbours have already finished, so the drain of one
    product overlaps the computation of the next.  Its own accumulator
    follows the forwarded ones onto c_out."""
    j: int32 = site.rank[1]
    acc: int32 = 0
    for k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        acc = acc + a * b
        io.a_out.put(a)
        io.b_out.put(b)
        if k + j >= N:
            r = io.c_in.get()
            io.c_out.put(r)
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.c_in,))
def pe_west(io: PEIO):
    """Western column: nothing to forward, only its own accumulator."""
    acc: int32 = 0
    for k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        acc = acc + a * b
        io.a_out.put(a)
        io.b_out.put(b)
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.a_out,))
def pe_east(io: PEIO):
    """Eastern column: no eastern neighbour on the operand path; forwards
    the seven results of its row then its own accumulator to the boundary."""
    acc: int32 = 0
    for k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        acc = acc + a * b
        io.b_out.put(b)
        if k >= 1:
            r = io.c_in.get()
            io.c_out.put(r)
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.b_out,))
def pe_south(io: PEIO, site: spmw.Site):
    """Southern row: no southern neighbour on the operand path."""
    j: int32 = site.rank[1]
    acc: int32 = 0
    for k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        acc = acc + a * b
        io.a_out.put(a)
        if k + j >= N:
            r = io.c_in.get()
            io.c_out.put(r)
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.b_out, PEIO.c_in))
def pe_sw(io: PEIO):
    """South-west corner."""
    acc: int32 = 0
    for k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        acc = acc + a * b
        io.a_out.put(a)
    io.c_out.put(acc)


@pe.role(unbound=(PEIO.a_out, PEIO.b_out))
def pe_se(io: PEIO):
    """South-east corner: forwards seven results then its own."""
    acc: int32 = 0
    for k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        acc = acc + a * b
        if k >= 1:
            r = io.c_in.get()
            io.c_out.put(r)
    io.c_out.put(acc)


@spmw.fabric
def engine(A: int8[N, N], B: int8[N, N], C: int32[N, N]):
    P = spmw.place(pe, on=mesh)
    spmw.stream_in(A, into=P.a_in, index=(P.rows, ...))
    spmw.stream_in(B, into=P.b_in, index=(..., P.cols))
    spmw.gather(C, from_=P.c_out, index=(P.rows, ...))

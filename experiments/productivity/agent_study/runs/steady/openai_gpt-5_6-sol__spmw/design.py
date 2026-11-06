import allo.spmw as spmw
from allo.ir.types import int8, int32

N = 8


class MACIO(spmw.Interface):
    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    b_in = spmw.In(int8)
    b_out = spmw.Out(int8)
    result = spmw.Out(int32)


mac_mesh = spmw.Topology(
    MACIO,
    grid=(N, N),
    link=lambda i, j: {
        MACIO.a_out: spmw.to((i, j + 1), MACIO.a_in),
        MACIO.b_out: spmw.to((i + 1, j), MACIO.b_in),
    },
)


@spmw.unit
def mac(io: MACIO):
    """The arithmetic half of one output-stationary PE."""
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        io.a_out.put(a)
        io.b_out.put(b)
        aa: int32 = a
        bb: int32 = b
        acc = acc + aa * bb
    io.result.put(acc)


@mac.role(unbound=(MACIO.a_out,))
def mac_east(io: MACIO):
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        io.b_out.put(b)
        aa: int32 = a
        bb: int32 = b
        acc = acc + aa * bb
    io.result.put(acc)


@mac.role(unbound=(MACIO.b_out,))
def mac_south(io: MACIO):
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        io.a_out.put(a)
        aa: int32 = a
        bb: int32 = b
        acc = acc + aa * bb
    io.result.put(acc)


@mac.role(unbound=(MACIO.a_out, MACIO.b_out))
def mac_southeast(io: MACIO):
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        aa: int32 = a
        bb: int32 = b
        acc = acc + aa * bb
    io.result.put(acc)


class DrainIO(spmw.Interface):
    # `own` is the stationary accumulator at this site.  east_in/west_out
    # form the result path through adjacent sites of a row.
    own = spmw.In(int32)
    east_in = spmw.In(int32)
    west_out = spmw.Out(int32)


drain_mesh = spmw.Topology(
    DrainIO,
    grid=(N, N),
    link=lambda i, j: {
        DrainIO.west_out: spmw.to((i, j - 1), DrainIO.east_in),
    },
)


@spmw.unit
def drain(io: DrainIO, site: spmw.Site):
    # Inject this site's result before relaying the sites to its east.  At the
    # west edge this gives C[i,0], C[i,1], ..., C[i,7].  This independently
    # running forwarding half is co-located with the MAC half of each PE, so
    # result movement can overlap the following product's accumulation.
    v = io.own.get()
    io.west_out.put(v)
    for _j in range(N - 1 - site.rank[1]):
        v = io.east_in.get()
        io.west_out.put(v)


@drain.role(unbound=(DrainIO.east_in,))
def drain_east(io: DrainIO):
    v = io.own.get()
    io.west_out.put(v)


@spmw.fabric
def engine(A: int8[N, N], B: int8[N, N], C: int32[N, N]):
    """Eight by eight output-stationary systolic matrix multiplier."""
    P = spmw.place(mac, on=mac_mesh)
    R = spmw.place(drain, on=drain_mesh)

    spmw.stream_in(A, into=P.a_in, index=(P.rows, ...))
    spmw.stream_in(B, into=P.b_in, index=(..., P.cols))
    spmw.link(P.result, to=R.own)
    spmw.gather(C, from_=R.west_out, index=(R.rows, ...))

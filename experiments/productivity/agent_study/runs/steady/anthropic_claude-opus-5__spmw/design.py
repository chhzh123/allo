# Fill in the body. Do not change the signature of `engine`.
#
# An 8x8 output-stationary systolic tile.
#
# PE(i,j) lives at grid site (i,j) and is built from two co-located
# processes, both placed on an 8x8 grid:
#
#   * `mac`   -- the element's one multiplier and its one accumulator.
#                It takes A from the west and passes it east, takes B from
#                the north and passes it south, and hands its finished
#                accumulator to the result-forwarding process at the *same*
#                site.
#   * `carry` -- the element's result-forwarding stage.  It relays the
#                accumulators of the elements west of it to the element
#                immediately east of it, and appends its own.
#
# Together they are one processing element: one multiplier, one accumulator,
# and links only to the four nearest neighbours (A east, B south, C east
# along the row).  Row i's results reach output port i through the row's own
# chain of elements, the last of which is PE(i,7).
import allo.spmw as spmw
from allo.ir.types import int8, int16, int32

N = 8


# ---------------------------------------------------------------------------
# The multiply-accumulate half of a processing element.
# ---------------------------------------------------------------------------
class MacIO(spmw.Interface):
    a_in = spmw.In(int8)  # A from the western neighbour
    a_out = spmw.Out(int8)  # A on to the eastern neighbour
    b_in = spmw.In(int8)  # B from the northern neighbour
    b_out = spmw.Out(int8)  # B on to the southern neighbour
    acc_out = spmw.Out(int32)  # finished C[i][j], to this site's carry stage


mac_mesh = spmw.Topology(
    MacIO,
    grid=(N, N),
    link=lambda i, j: {
        MacIO.a_out: spmw.to((i, j + 1), MacIO.a_in),  # A travels east
        MacIO.b_out: spmw.to((i + 1, j), MacIO.b_in),  # B travels south
    },
)


# interior: has both an eastern and a southern neighbour
@spmw.unit
def mac(io: MacIO):
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        io.a_out.put(a)
        io.b_out.put(b)
        ai: int16 = a
        bi: int16 = b
        p: int16 = ai * bi
        acc = acc + p
    io.acc_out.put(acc)


# eastern edge, j = N-1: nothing further east
@mac.role(unbound=(MacIO.a_out,))
def mac_east(io: MacIO):
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        io.b_out.put(b)
        ai: int16 = a
        bi: int16 = b
        p: int16 = ai * bi
        acc = acc + p
    io.acc_out.put(acc)


# southern edge, i = N-1: nothing further south
@mac.role(unbound=(MacIO.b_out,))
def mac_south(io: MacIO):
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        io.a_out.put(a)
        ai: int16 = a
        bi: int16 = b
        p: int16 = ai * bi
        acc = acc + p
    io.acc_out.put(acc)


# south-east corner
@mac.role(unbound=(MacIO.a_out, MacIO.b_out))
def mac_corner(io: MacIO):
    acc: int32 = 0
    for _k in range(N):
        a = io.a_in.get()
        b = io.b_in.get()
        ai: int16 = a
        bi: int16 = b
        p: int16 = ai * bi
        acc = acc + p
    io.acc_out.put(acc)


# ---------------------------------------------------------------------------
# The result-forwarding half of a processing element: row i's drain chain.
# Site (i,j) relays the j accumulators coming from the west, then its own,
# so PE(i,7) emits C[i][0], C[i][1], ... C[i][7] in ascending j.
# ---------------------------------------------------------------------------
class CarryIO(spmw.Interface):
    c_in = spmw.In(int32)  # results from the western neighbour
    c_out = spmw.Out(int32)  # results on to the eastern neighbour
    acc_in = spmw.In(int32)  # this site's own accumulator


carry_mesh = spmw.Topology(
    CarryIO,
    grid=(N, N),
    link=lambda i, j: {CarryIO.c_out: spmw.to((i, j + 1), CarryIO.c_in)},
)


@spmw.unit
def carry(io: CarryIO, site: spmw.Site):
    west: int32 = site.rank[1]
    for _m in range(west):
        v = io.c_in.get()
        io.c_out.put(v)
    own = io.acc_in.get()
    io.c_out.put(own)


# western edge, j = 0: nothing to the west, so only its own result
@carry.role(unbound=(CarryIO.c_in,))
def carry_head(io: CarryIO):
    own = io.acc_in.get()
    io.c_out.put(own)


@spmw.fabric
def engine(A: int8[N, N], B: int8[N, N], C: int32[N, N]):
    """A[i][k] arrives on input port i, B[k][j] on input port j,
    and C[i][j] must leave on output port i, in ascending j."""
    P = spmw.place(mac, on=mac_mesh)
    D = spmw.place(carry, on=carry_mesh)
    # each element hands its accumulator to its own carry stage
    spmw.link(P.acc_out, to=D.acc_in)
    # row i of A enters PE(i,0) and walks east, one element per step
    spmw.stream_in(A, into=P.a_in, index=(P.rows, ...))
    # column j of B enters PE(0,j) and walks south, one element per step
    spmw.stream_in(B, into=P.b_in, index=(..., P.cols))
    # row i of C leaves the row's last element, C[i][0] first
    spmw.gather(C, from_=D.c_out, index=(D.rows, ...))

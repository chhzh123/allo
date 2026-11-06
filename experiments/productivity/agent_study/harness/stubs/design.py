# Fill in the body. Do not change the signature of `engine`.
import allo.spmw as spmw
from allo.ir.types import int8, int32

N = 8


@spmw.fabric
def engine(A: int8[N, N], B: int8[N, N], C: int32[N, N]):
    """A[i][k] arrives on input port i, B[k][j] on input port j,
    and C[i][j] must leave on output port i, in ascending j."""
    # your design here

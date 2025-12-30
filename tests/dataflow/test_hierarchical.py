import allo
import allo.dataflow as df
from allo.ir.types import Stream, float32

Ty = float32


@df.region()
def comm(M, N, K):
    pipe: Stream[Ty, 4]

    @df.kernel(mapping=[1])
    def producer(A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # load data
            out: Ty = A[i, j]
            # send data
            pipe.put(out)

    @df.kernel(mapping=[1])
    def consumer(B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # receive data
            data = pipe.get()
            # computation
            B[i, j] = data + 1


@df.region()
def top():
    @df.kernel(mapping=[2])
    def inner():
        i = df.get_pid()
        with allo.meta_if(i == 0):
            comm(16, 16, 16)
        with allo.meta_elif(i == 1):
            comm(32, 32, 32)


mod = df.build(top)
print(mod.module)

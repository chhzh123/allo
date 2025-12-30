import allo
import allo.dataflow as df
from allo.ir.types import Stream, float32
from allo.memory import Layout

Ty = float32


@df.region()
def comm[M, N](A: Ty[M, N], B: Ty[M, N]):
    pipe: Stream[Ty, 4]

    @df.kernel(mapping=[2], args=[A])
    def producer(A: Ty[M, N] @ Layout("S0R")):
        for i, j in allo.grid(M // 2, N):
            # load data
            out: Ty = A[i, j]
            # send data
            pipe.put(out)

    @df.kernel(mapping=[2], args=[B])
    def consumer(B: Ty[M, N] @ Layout("S0R")):
        for i, j in allo.grid(M // 2, N):
            # receive data
            data = pipe.get()
            # computation
            B[i, j] = data + 1


@df.region()
def top(A: Ty[16, 16], B: Ty[16, 16], C: Ty[32, 32], D: Ty[32, 32]):
    @df.kernel(mapping=[2], args=[A, B, C, D])
    def inner(A: Ty[16, 16], B: Ty[16, 16], C: Ty[32, 32], D: Ty[32, 32]):
        i = df.get_pid()
        with allo.meta_if(i == 0):
            comm[16, 16](A, B)
        with allo.meta_elif(i == 1):
            comm[32, 32](C, D)


mod = df.build(top)
print(mod.module)

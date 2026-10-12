# The Dato paper's array (tests/dataflow/test_multi_cache_gemm.py), generated
# by dato_gen.py with stream depths "paper".
import allo
from allo.ir.types import int8, int32, Stream, UInt, ConstExpr
import allo.dataflow as df
import allo.dsl as dsl


@df.region()
def MXU[
    Rt, Ct, M, N, K
](
    A_Packed: "UInt(Rt * 8)[M * K // Rt]",
    B_Packed: "UInt(Ct * 8)[K * N // Ct]",
    C_Packed: "UInt(Rt * 8)[M * N // Rt]",
):
    P0: ConstExpr[int32] = Rt + 2
    P1: ConstExpr[int32] = Ct + 2

    L3_A: Stream[UInt(Rt * 8), 4]
    L3_B: Stream[UInt(Ct * 8), 4]
    L3_C: Stream[UInt(Rt * 8), 4]

    L2_A: Stream[UInt(Rt * 8), 2][P0 - 1]
    L2_B: Stream[UInt(Ct * 8), 2][P1 - 1]

    L1_C: Stream[UInt(Rt * 8), 2][Rt, Ct]
    L2_C: Stream[UInt(Rt * 8), 2][Ct]

    fifo_A: Stream[int8, 2][Rt, Ct]
    fifo_B: Stream[int8, 2][Rt, Ct]

    @df.kernel(mapping=[1], args=[A_Packed])
    def offchip_loadA(A_Packed_in: "UInt(Rt * 8)[M * K // Rt]"):
        for mt, nt in dsl.grid(M // Rt, N // Ct):
            for k in range(K):
                L3_A.put(A_Packed_in[mt * K + k])

    @df.kernel(mapping=[1], args=[B_Packed])
    def offchip_loadB(B_Packed_in: "UInt(Ct * 8)[K * N // Ct]"):
        for mt, nt in dsl.grid(M // Rt, N // Ct):
            for k in range(K):
                L3_B.put(B_Packed_in[nt * K + k])

    @df.kernel(mapping=[P0, P1])
    def gemm():
        i, j = df.get_pid()
        # peripheral kernels
        with allo.meta_if(i == 0 and j == 0):
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                for k in range(K):
                    L2_A[1].put(L3_A.get())
                    L2_B[1].put(L3_B.get())

        with allo.meta_elif(i == P0 - 1 and j == P1 - 1):
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                for n in range(Ct):
                    L3_C.put(L2_C[Ct - 1].get())

        with allo.meta_elif(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass

        with allo.meta_elif(j == 0):
            # i > 0, the first column
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                for k in range(K):
                    a = L2_A[i].get()
                    # unpack data
                    fifo_A[i - 1, 0].put(a[8 * (i - 1) : 8 * i])
                    with allo.meta_if(i < Rt):
                        L2_A[i + 1].put(a)

        with allo.meta_elif(i == 0):
            # j > 0, the first row
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                for k in range(K):
                    b = L2_B[j].get()
                    fifo_B[0, j - 1].put(b[8 * (j - 1) : 8 * j])
                    with allo.meta_if(j < Ct):
                        L2_B[j + 1].put(b)

        with allo.meta_elif(i == P0 - 1):
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                c_C = L1_C[i - 2, Ct - j].get()
                L2_C[j - 1].put(c_C)
                with allo.meta_if(j != 1):
                    for ind in range(j - 1):
                        L2_C[j - 1].put(L2_C[j - 2].get())

        with allo.meta_elif(j == P1 - 1):
            pass

        # main body
        with allo.meta_else():
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                c: int8 = 0
                for k in range(K):
                    a: int8 = fifo_A[i - 1, j - 1].get()
                    b: int8 = fifo_B[i - 1, j - 1].get()
                    c += a * b
                    with allo.meta_if(j < Ct):
                        fifo_A[i - 1, j].put(a)
                    with allo.meta_if(i < Rt):
                        fifo_B[i, j - 1].put(b)

                packed_tmp: UInt(Rt * 8)
                with allo.meta_if(i == 1):
                    packed_tmp = 0
                with allo.meta_else():
                    packed_tmp = L1_C[i - 2, j - 1].get()

                packed_c: UInt(Rt * 8) = 0
                for m in range(Rt):
                    if m == i - 1:
                        packed_c[m * 8 : (m + 1) * 8] = c
                    else:
                        packed_c[m * 8 : (m + 1) * 8] = packed_tmp[m * 8 : (m + 1) * 8]
                L1_C[i - 1, j - 1].put(packed_c)

    @df.kernel(mapping=[1], args=[C_Packed])
    def offchip_store(C_Packed_out: "UInt(Rt * 8)[M * N // Rt]"):
        for mt, nt in dsl.grid(M // Rt, N // Ct):
            for n in range(Ct):
                C_Packed_out[mt * N + nt * Ct + n] = L3_C.get()


Rt, Ct = 16, 16
M, N, K = 128, 128, 128


@df.region()
def top(
    A: "UInt(Rt * 8)[M * K // Rt]",
    B: "UInt(Ct * 8)[K * N // Ct]",
    C: "UInt(Rt * 8)[M * N // Rt]",
):
    @df.kernel(mapping=[1], args=[A, B, C])
    def wrapper(
        A_Packed: "UInt(Rt * 8)[M * K // Rt]",
        B_Packed: "UInt(Ct * 8)[K * N // Ct]",
        C_Packed: "UInt(Rt * 8)[M * N // Rt]",
    ):
        MXU[Rt, Ct, M, N, K](A_Packed, B_Packed, C_Packed)

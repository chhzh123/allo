import allo
from allo.ir.types import int8
import numpy as np


def test_linear():
    from allo.library.systolic import systolic

    # L, D = 512, 768
    # M0, M1 = 16, 16
    # L, D = 8, 8
    # M0, M1 = 2, 2
    # L, D = 512, 512
    # M0, M1 = 16, 16
    L, D = 1024, 1024
    M0, M1 = 16, 16
    W_A = np.random.randint(-4, 4, size=(D, D)).astype(np.int8)
    allo_C = np.zeros((L, D), dtype=np.int8)

    def top(X: int8[L, D], W_A: int8[D, D]) -> int8[L, D]:
        Z: int8[L, D]
        systolic[int8, int8, int8, L, D, D, M0, M1](X, W_A, Z)
        return Z

    s = allo.customize(top)
    # CPU testing
    mod = s.build()
    X = np.random.randint(-4, 4, size=(L, D)).astype(np.int8)
    allo_C = mod(X, W_A)
    np_C = X @ W_A
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")
    s.compose(systolic, instantiate=[int8, int8, int8, L, D, D, M0, M1])
    s.dataflow("top")
    hls_mod = s.build(
        target="vitis_hls",
        mode="hw",
        project=f"single_{L}x{D}_tile_{M0}x{M1}_csyn_pipeline.prj",
    )
    hls_mod(X, W_A, allo_C)
    # hls_mod = s.build(
    #     target="vitis_hls",
    #     mode="csyn",
    #     project=f"single_{L}x{D}_tile_{M0}x{M1}_csyn_pipeline.prj",
    # )
    # hls_mod()
    np_C = X @ W_A
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")


if __name__ == "__main__":
    test_linear()

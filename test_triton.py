
import allo
from allo.ir.types import float32
import numpy as np
from allo.backend.triton import build, HAS_TRITON
import pytest

def test_vadd_ir():
    """Test that Triton IR is generated correctly."""
    def vadd(A: float32[128], B: float32[128]) -> float32[128]:
        C: float32[128] = 0.0
        for i in range(128):
            C[i] = A[i] + B[i]
        return C

    s = allo.customize(vadd)
    print("Allo IR:")
    print(s.module)
    
    # Use triton backend
    mod = s.build(target="triton", configs={"block_size": 128})
    
    triton_ir = mod.get_ir()
    print("\nGenerated Triton IR:")
    print(triton_ir)
    
    # Check for triton keywords
    assert "tt.func" in triton_ir
    assert "tt.get_program_id" in triton_ir
    assert "tt.load" in triton_ir
    assert "tt.store" in triton_ir
    assert "tt.make_range" in triton_ir
    print("\nIR generation test passed!")


@pytest.mark.skipif(not HAS_TRITON, reason="Triton/PyTorch not installed")
def test_vadd_execution():
    """Test actual execution of the generated Triton kernel."""
    import torch
    
    def vadd(A: float32[128], B: float32[128], C: float32[128]):
        for i in range(128):
            C[i] = A[i] + B[i]

    s = allo.customize(vadd)
    
    # Build with triton backend
    mod = s.build(target="triton", configs={"block_size": 128})
    
    # Print generated Python Triton code
    print("\nGenerated Python Triton code:")
    print(mod.get_python_code())
    
    # Prepare test data
    A = np.random.rand(128).astype(np.float32)
    B = np.random.rand(128).astype(np.float32)
    C = np.zeros(128, dtype=np.float32)
    
    # Run the kernel
    result = mod(A, B, C)
    
    # Verify result
    print(f"Result device: {result.device}")
    assert "cuda" in str(result.device)
    
    expected = A + B
    result_np = result.cpu().numpy()
    
    np.testing.assert_allclose(result_np, expected, rtol=1e-5)
    print("\nExecution test passed!")
    print(f"Sample results: {result_np[:5]}")
    print(f"Expected:       {expected[:5]}")


@pytest.mark.skipif(not HAS_TRITON, reason="Triton/PyTorch not installed")  
def test_vadd_large():
    """Test with a larger array size."""
    import torch
    
    N = 1024
    
    def vadd_large(A: float32[1024], B: float32[1024], C: float32[1024]):
        for i in range(1024):
            C[i] = A[i] + B[i]

    s = allo.customize(vadd_large)
    mod = s.build(target="triton", configs={"block_size": 256})
    
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    
    result = mod(A, B, C)
    
    # Verify result
    print(f"Result device: {result.device}")
    assert "cuda" in str(result.device)
    
    expected = A + B
    result_np = result.cpu().numpy()
    
    np.testing.assert_allclose(result_np, expected, rtol=1e-5)
    print(f"\nLarge array test passed! (N={N})")


if __name__ == "__main__":
    test_vadd_ir()
    if HAS_TRITON:
        test_vadd_execution()
        test_vadd_large()
    else:
        print("\nSkipping execution tests: Triton/PyTorch not installed")

"""Tests for Hessenberg decomposition."""

import pytest
import scipy.linalg
import torch

from torchscience.linear_algebra.decomposition import (
    HessenbergResult,
    hessenberg,
)


class TestHessenberg:
    """Tests for Hessenberg decomposition."""

    def test_basic(self):
        """Test basic Hessenberg decomposition returns correct shapes and info."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float64,
        )

        result = hessenberg(a)

        assert isinstance(result, HessenbergResult)
        assert result.H.shape == (3, 3)
        assert result.Q.shape == (3, 3)
        assert result.info.shape == ()
        assert result.info.item() == 0

    def test_reconstruction(self):
        """Test that A = Q @ H @ Q.mH holds."""
        torch.manual_seed(42)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = hessenberg(a)

        reconstructed = result.Q @ result.H @ result.Q.mH
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_hessenberg_form(self):
        """Test that H has zeros below the first subdiagonal."""
        torch.manual_seed(123)
        n = 6
        a = torch.randn(n, n, dtype=torch.float64)

        result = hessenberg(a)

        # Check that all entries below the first subdiagonal are zero
        for i in range(2, n):
            for j in range(i - 1):
                assert abs(result.H[i, j].item()) < 1e-10, (
                    f"H[{i}, {j}] = {result.H[i, j].item()} should be zero"
                )

    def test_unitary(self):
        """Test that Q is unitary: Q @ Q.mH = I."""
        torch.manual_seed(456)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)

        result = hessenberg(a)

        identity = torch.eye(n, dtype=torch.float64)
        QQH = result.Q @ result.Q.mH
        torch.testing.assert_close(QQH, identity, rtol=1e-10, atol=1e-10)

        QHQ = result.Q.mH @ result.Q
        torch.testing.assert_close(QHQ, identity, rtol=1e-10, atol=1e-10)

    def test_scipy_comparison(self):
        """Compare eigenvalues with scipy.linalg.hessenberg."""
        torch.manual_seed(789)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = hessenberg(a)

        # Get scipy result
        H_scipy, Q_scipy = scipy.linalg.hessenberg(a.numpy(), calc_q=True)

        # The eigenvalues of H should match those of A (and scipy's H)
        eig_H = torch.linalg.eigvals(result.H)
        eig_scipy = torch.linalg.eigvals(torch.from_numpy(H_scipy))

        # Sort by real part for comparison
        def sort_complex(t):
            return t[torch.argsort(t.real)]

        torch.testing.assert_close(
            sort_complex(eig_H),
            sort_complex(eig_scipy),
            rtol=1e-10,
            atol=1e-10,
        )

        # Also verify eigenvalues match original matrix
        eig_a = torch.linalg.eigvals(a)
        torch.testing.assert_close(
            sort_complex(eig_H),
            sort_complex(eig_a),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_batched(self):
        """Test batched Hessenberg decomposition."""
        torch.manual_seed(101)
        batch_size = 3
        n = 4

        a = torch.randn(batch_size, n, n, dtype=torch.float64)

        result = hessenberg(a)

        assert result.H.shape == (batch_size, n, n)
        assert result.Q.shape == (batch_size, n, n)
        assert result.info.shape == (batch_size,)

        # Verify reconstruction and Hessenberg form for each batch
        for i in range(batch_size):
            # Reconstruction
            reconstructed = result.Q[i] @ result.H[i] @ result.Q[i].mH
            torch.testing.assert_close(
                reconstructed, a[i], rtol=1e-10, atol=1e-10
            )

            # Hessenberg form
            for row in range(2, n):
                for col in range(row - 1):
                    assert abs(result.H[i, row, col].item()) < 1e-10

            # Unitary Q
            identity = torch.eye(n, dtype=torch.float64)
            torch.testing.assert_close(
                result.Q[i] @ result.Q[i].mH, identity, rtol=1e-10, atol=1e-10
            )

    def test_batched_multi_dim(self):
        """Test with multiple batch dimensions."""
        torch.manual_seed(202)
        n = 3

        a = torch.randn(2, 3, n, n, dtype=torch.float64)

        result = hessenberg(a)

        assert result.H.shape == (2, 3, n, n)
        assert result.Q.shape == (2, 3, n, n)
        assert result.info.shape == (2, 3)

        # Check reconstruction for a sample
        reconstructed = result.Q[1, 2] @ result.H[1, 2] @ result.Q[1, 2].mH
        torch.testing.assert_close(
            reconstructed, a[1, 2], rtol=1e-10, atol=1e-10
        )

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        a = torch.randn(4, 4, dtype=torch.float64, device="meta")

        result = hessenberg(a)

        assert result.H.shape == (4, 4)
        assert result.Q.shape == (4, 4)
        assert result.info.shape == ()
        assert result.H.device.type == "meta"
        assert result.Q.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        a = torch.randn(2, 3, 5, 5, dtype=torch.float64, device="meta")

        result = hessenberg(a)

        assert result.H.shape == (2, 3, 5, 5)
        assert result.Q.shape == (2, 3, 5, 5)
        assert result.info.shape == (2, 3)

    def test_compile(self):
        """Test torch.compile compatibility."""
        torch.manual_seed(303)

        @torch.compile
        def compiled_hessenberg(x):
            return hessenberg(x)

        a = torch.randn(4, 4, dtype=torch.float64)

        result = compiled_hessenberg(a)

        # Verify basic correctness
        reconstructed = result.Q @ result.H @ result.Q.mH
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_complex_input(self):
        """Test with complex input."""
        torch.manual_seed(404)
        n = 4
        a = torch.randn(n, n, dtype=torch.complex128)

        result = hessenberg(a)

        assert result.H.dtype == torch.complex128
        assert result.Q.dtype == torch.complex128

        # Reconstruction
        reconstructed = result.Q @ result.H @ result.Q.mH
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

        # Hessenberg form
        for i in range(2, n):
            for j in range(i - 1):
                assert abs(result.H[i, j].item()) < 1e-10

        # Unitary Q
        identity = torch.eye(n, dtype=torch.complex128)
        torch.testing.assert_close(
            result.Q @ result.Q.mH, identity, rtol=1e-10, atol=1e-10
        )

    def test_already_hessenberg(self):
        """Test with an already upper Hessenberg matrix."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [0.0, 9.0, 10.0, 11.0],
                [0.0, 0.0, 12.0, 13.0],
            ],
            dtype=torch.float64,
        )

        result = hessenberg(a)

        # Reconstruction should still work
        reconstructed = result.Q @ result.H @ result.Q.mH
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_upper_triangular(self):
        """Test with an upper triangular matrix (special case of Hessenberg)."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [0.0, 4.0, 5.0],
                [0.0, 0.0, 6.0],
            ],
            dtype=torch.float64,
        )

        result = hessenberg(a)

        # Reconstruction
        reconstructed = result.Q @ result.H @ result.Q.mH
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_2x2_matrix(self):
        """Test with 2x2 matrix (trivial case, already Hessenberg)."""
        a = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float64,
        )

        result = hessenberg(a)

        assert result.H.shape == (2, 2)
        reconstructed = result.Q @ result.H @ result.Q.mH
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_1x1_matrix(self):
        """Test with 1x1 matrix (trivial case)."""
        a = torch.tensor([[5.0]], dtype=torch.float64)

        result = hessenberg(a)

        assert result.H.shape == (1, 1)
        assert result.Q.shape == (1, 1)
        torch.testing.assert_close(result.H, a, rtol=1e-10, atol=1e-10)

    def test_invalid_1d_input(self):
        """Test error on 1D input."""
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="must be at least 2D"):
            hessenberg(a)

    def test_invalid_non_square(self):
        """Test error on non-square input."""
        a = torch.randn(3, 4, dtype=torch.float64)
        with pytest.raises(ValueError, match="must be square"):
            hessenberg(a)

    def test_float32(self):
        """Test with float32 dtype."""
        torch.manual_seed(505)
        n = 4
        a = torch.randn(n, n, dtype=torch.float32)

        result = hessenberg(a)

        assert result.H.dtype == torch.float32
        assert result.Q.dtype == torch.float32

        # Lower tolerance for float32
        reconstructed = result.Q @ result.H @ result.Q.mH
        torch.testing.assert_close(reconstructed, a, rtol=1e-4, atol=1e-4)

    def test_eigenvalue_preservation(self):
        """Test that eigenvalues of H match eigenvalues of A."""
        torch.manual_seed(606)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = hessenberg(a)

        eig_a = torch.linalg.eigvals(a)
        eig_h = torch.linalg.eigvals(result.H)

        # Sort by real part for comparison
        def sort_complex(t):
            return t[torch.argsort(t.real)]

        torch.testing.assert_close(
            sort_complex(eig_a),
            sort_complex(eig_h),
            rtol=1e-10,
            atol=1e-10,
        )

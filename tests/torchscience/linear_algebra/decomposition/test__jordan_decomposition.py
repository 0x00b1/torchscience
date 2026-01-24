"""Tests for Jordan decomposition."""

import pytest
import torch

from torchscience.linear_algebra.decomposition import (
    JordanDecompositionResult,
    jordan_decomposition,
)


class TestJordanDecomposition:
    """Tests for Jordan decomposition."""

    def test_basic(self):
        """Test basic Jordan decomposition returns correct shapes and info."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float64,
        )

        result = jordan_decomposition(a)

        assert isinstance(result, JordanDecompositionResult)
        assert result.J.shape == (3, 3)
        assert result.P.shape == (3, 3)
        assert result.info.shape == ()
        # Output should be complex
        assert result.J.is_complex()
        assert result.P.is_complex()

    def test_reconstruction(self):
        """Test that A = P @ J @ P^{-1} holds."""
        torch.manual_seed(42)
        n = 5
        # Create a well-conditioned random matrix
        a = torch.randn(n, n, dtype=torch.float64)

        result = jordan_decomposition(a)

        # Use solve for better numerical stability: A @ P = P @ J
        # Instead of computing P^{-1} directly
        # Reconstruct: A = P @ J @ P^{-1}
        P_inv = torch.linalg.inv(result.P)
        reconstructed = result.P @ result.J @ P_inv

        # Take real part for comparison since input was real
        torch.testing.assert_close(
            reconstructed.real,
            a,
            rtol=1e-8,
            atol=1e-8,
        )

    def test_eigenvalues(self):
        """Test that diagonal of J contains eigenvalues."""
        torch.manual_seed(123)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)

        result = jordan_decomposition(a)

        # For diagonalizable matrices, J should be diagonal with eigenvalues
        j_diag = torch.diag(result.J)
        expected_eigenvalues = torch.linalg.eigvals(a)

        # Sort both for comparison (eigenvalues may be in different order)
        def sort_complex(t):
            # Sort by real part (sufficient for comparison)
            idx = torch.argsort(t.real)
            return t[idx]

        torch.testing.assert_close(
            sort_complex(j_diag),
            sort_complex(expected_eigenvalues),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_diagonal_matrix(self):
        """Test that J equals A for diagonal input."""
        d = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))

        result = jordan_decomposition(d)

        # Extract diagonal of J
        j_diag = torch.diag(result.J)

        # Should contain the same eigenvalues (1, 2, 3, 4) in some order
        def sort_by_real(t):
            return t[torch.argsort(t.real)]

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex128)
        torch.testing.assert_close(
            sort_by_real(j_diag),
            sort_by_real(expected),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_batched(self):
        """Test batched Jordan decomposition."""
        torch.manual_seed(101)
        batch_size = 3
        n = 4

        a = torch.randn(batch_size, n, n, dtype=torch.float64)

        result = jordan_decomposition(a)

        assert result.J.shape == (batch_size, n, n)
        assert result.P.shape == (batch_size, n, n)
        assert result.info.shape == (batch_size,)

        # Verify reconstruction for each batch
        for i in range(batch_size):
            P_inv = torch.linalg.inv(result.P[i])
            reconstructed = result.P[i] @ result.J[i] @ P_inv
            torch.testing.assert_close(
                reconstructed.real,
                a[i],
                rtol=1e-8,
                atol=1e-8,
            )

    def test_batched_multi_dim(self):
        """Test with multiple batch dimensions."""
        torch.manual_seed(202)
        n = 3

        a = torch.randn(2, 3, n, n, dtype=torch.float64)

        result = jordan_decomposition(a)

        assert result.J.shape == (2, 3, n, n)
        assert result.P.shape == (2, 3, n, n)
        assert result.info.shape == (2, 3)

        # Check reconstruction for a sample
        P_inv = torch.linalg.inv(result.P[1, 2])
        reconstructed = result.P[1, 2] @ result.J[1, 2] @ P_inv
        torch.testing.assert_close(
            reconstructed.real,
            a[1, 2],
            rtol=1e-8,
            atol=1e-8,
        )

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        a = torch.randn(4, 4, dtype=torch.float64, device="meta")

        result = jordan_decomposition(a)

        assert result.J.shape == (4, 4)
        assert result.P.shape == (4, 4)
        assert result.info.shape == ()
        assert result.J.device.type == "meta"
        assert result.P.device.type == "meta"
        # Output should be complex
        assert result.J.dtype == torch.complex128
        assert result.P.dtype == torch.complex128

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        a = torch.randn(2, 3, 5, 5, dtype=torch.float64, device="meta")

        result = jordan_decomposition(a)

        assert result.J.shape == (2, 3, 5, 5)
        assert result.P.shape == (2, 3, 5, 5)
        assert result.info.shape == (2, 3)

    def test_compile(self):
        """Test torch.compile compatibility."""
        torch.manual_seed(303)

        @torch.compile
        def compiled_jordan(x):
            return jordan_decomposition(x)

        a = torch.randn(4, 4, dtype=torch.float64)

        result = compiled_jordan(a)

        # Verify basic correctness
        P_inv = torch.linalg.inv(result.P)
        reconstructed = result.P @ result.J @ P_inv
        torch.testing.assert_close(reconstructed.real, a, rtol=1e-8, atol=1e-8)

    def test_no_gradients(self):
        """Test that gradients don't flow through Jordan decomposition."""
        a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

        result = jordan_decomposition(a)

        # The result should not have requires_grad (detached)
        assert not result.J.requires_grad
        assert not result.P.requires_grad

    def test_complex_input(self):
        """Test with complex input."""
        torch.manual_seed(404)
        n = 4
        a = torch.randn(n, n, dtype=torch.complex128)

        result = jordan_decomposition(a)

        assert result.J.dtype == torch.complex128
        assert result.P.dtype == torch.complex128

        # Reconstruction
        P_inv = torch.linalg.inv(result.P)
        reconstructed = result.P @ result.J @ P_inv
        torch.testing.assert_close(reconstructed, a, rtol=1e-8, atol=1e-8)

    def test_symmetric_matrix(self):
        """Test with symmetric matrix (always diagonalizable with real eigenvalues)."""
        torch.manual_seed(505)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)
        a = (a + a.T) / 2  # Make symmetric

        result = jordan_decomposition(a)

        # Symmetric matrices have real eigenvalues
        j_diag = torch.diag(result.J)
        assert torch.allclose(
            j_diag.imag, torch.zeros_like(j_diag.imag), atol=1e-10
        )

        # Reconstruction
        P_inv = torch.linalg.inv(result.P)
        reconstructed = result.P @ result.J @ P_inv
        torch.testing.assert_close(reconstructed.real, a, rtol=1e-8, atol=1e-8)

    def test_2x2_matrix(self):
        """Test with 2x2 matrix."""
        a = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float64,
        )

        result = jordan_decomposition(a)

        assert result.J.shape == (2, 2)
        assert result.P.shape == (2, 2)

        # Reconstruction
        P_inv = torch.linalg.inv(result.P)
        reconstructed = result.P @ result.J @ P_inv
        torch.testing.assert_close(reconstructed.real, a, rtol=1e-8, atol=1e-8)

    def test_1x1_matrix(self):
        """Test with 1x1 matrix."""
        a = torch.tensor([[5.0]], dtype=torch.float64)

        result = jordan_decomposition(a)

        assert result.J.shape == (1, 1)
        assert result.P.shape == (1, 1)
        torch.testing.assert_close(
            result.J[0, 0].real,
            a[0, 0],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_invalid_1d_input(self):
        """Test error on 1D input."""
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="must be at least 2D"):
            jordan_decomposition(a)

    def test_invalid_non_square(self):
        """Test error on non-square input."""
        a = torch.randn(3, 4, dtype=torch.float64)
        with pytest.raises(ValueError, match="must be square"):
            jordan_decomposition(a)

    def test_float32(self):
        """Test with float32 dtype."""
        torch.manual_seed(606)
        n = 4
        a = torch.randn(n, n, dtype=torch.float32)

        result = jordan_decomposition(a)

        # Output should be complex float
        assert result.J.dtype == torch.complex64
        assert result.P.dtype == torch.complex64

        # Lower tolerance for float32
        P_inv = torch.linalg.inv(result.P)
        reconstructed = result.P @ result.J @ P_inv
        torch.testing.assert_close(
            reconstructed.real,
            a,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_identity_matrix(self):
        """Test with identity matrix."""
        n = 4
        a = torch.eye(n, dtype=torch.float64)

        result = jordan_decomposition(a)

        # Jordan form of identity should be identity
        expected_j = torch.eye(n, dtype=torch.complex128)
        torch.testing.assert_close(
            result.J,
            expected_j,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_upper_triangular(self):
        """Test with upper triangular matrix."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [0.0, 4.0, 5.0],
                [0.0, 0.0, 6.0],
            ],
            dtype=torch.float64,
        )

        result = jordan_decomposition(a)

        # Eigenvalues should be the diagonal entries
        j_diag = torch.diag(result.J)

        def sort_by_real(t):
            return t[torch.argsort(t.real)]

        expected = torch.tensor([1.0, 4.0, 6.0], dtype=torch.complex128)
        torch.testing.assert_close(
            sort_by_real(j_diag),
            sort_by_real(expected),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_eigenvalue_equation(self):
        """Test that A @ v_i = lambda_i @ v_i for each eigenvector."""
        torch.manual_seed(707)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)

        result = jordan_decomposition(a)

        # Convert a to complex for comparison
        a_complex = a.to(torch.complex128)

        # For diagonalizable matrices, each column of P is an eigenvector
        # and each diagonal of J is the corresponding eigenvalue
        for i in range(n):
            v_i = result.P[:, i]
            lambda_i = result.J[i, i]

            # A @ v_i should equal lambda_i * v_i
            lhs = a_complex @ v_i
            rhs = lambda_i * v_i

            torch.testing.assert_close(lhs, rhs, rtol=1e-8, atol=1e-8)

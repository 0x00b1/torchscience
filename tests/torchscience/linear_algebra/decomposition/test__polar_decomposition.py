"""Tests for polar decomposition."""

import pytest
import scipy.linalg
import torch

from torchscience.linear_algebra.decomposition import (
    PolarDecompositionResult,
    polar_decomposition,
)


class TestPolarDecomposition:
    """Tests for polar decomposition."""

    def test_basic(self):
        """Test basic polar decomposition returns correct shapes and info."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float64,
        )

        result = polar_decomposition(a)

        assert isinstance(result, PolarDecompositionResult)
        assert result.U.shape == (3, 3)
        assert result.P.shape == (3, 3)  # Right polar: P is (n, n)
        assert result.info.shape == ()
        assert result.info.item() == 0

    def test_basic_left(self):
        """Test left polar decomposition returns correct shapes."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float64,
        )

        result = polar_decomposition(a, side="left")

        assert isinstance(result, PolarDecompositionResult)
        assert result.U.shape == (3, 3)
        assert result.P.shape == (3, 3)  # Left polar: P is (m, m)
        assert result.info.shape == ()
        assert result.info.item() == 0

    def test_reconstruction_right(self):
        """Test that A = U @ P holds for right polar decomposition."""
        torch.manual_seed(42)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a, side="right")

        reconstructed = result.U @ result.P
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_reconstruction_left(self):
        """Test that A = P @ U holds for left polar decomposition."""
        torch.manual_seed(42)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a, side="left")

        reconstructed = result.P @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_unitary(self):
        """Test that U @ U.mH = I for square invertible matrices."""
        torch.manual_seed(123)
        n = 4
        # Create an invertible matrix
        a = torch.randn(n, n, dtype=torch.float64)
        # Make it well-conditioned
        a = a + 2 * torch.eye(n, dtype=torch.float64)

        result = polar_decomposition(a)

        identity = torch.eye(n, dtype=torch.float64)
        UUH = result.U @ result.U.mH
        torch.testing.assert_close(UUH, identity, rtol=1e-10, atol=1e-10)

        UHU = result.U.mH @ result.U
        torch.testing.assert_close(UHU, identity, rtol=1e-10, atol=1e-10)

    def test_positive_semidefinite(self):
        """Test that eigenvalues of P are >= 0."""
        torch.manual_seed(456)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a)

        eigenvalues = torch.linalg.eigvalsh(result.P)
        # All eigenvalues should be non-negative (within numerical tolerance)
        assert torch.all(eigenvalues >= -1e-10), (
            f"P should be positive semidefinite, got eigenvalues: {eigenvalues}"
        )

    def test_positive_semidefinite_left(self):
        """Test that eigenvalues of P are >= 0 for left polar."""
        torch.manual_seed(456)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a, side="left")

        eigenvalues = torch.linalg.eigvalsh(result.P)
        # All eigenvalues should be non-negative (within numerical tolerance)
        assert torch.all(eigenvalues >= -1e-10), (
            f"P should be positive semidefinite, got eigenvalues: {eigenvalues}"
        )

    def test_hermitian(self):
        """Test that P = P.mH (P is Hermitian)."""
        torch.manual_seed(789)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a)

        torch.testing.assert_close(
            result.P, result.P.mH, rtol=1e-10, atol=1e-10
        )

    def test_hermitian_left(self):
        """Test that P = P.mH for left polar."""
        torch.manual_seed(789)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a, side="left")

        torch.testing.assert_close(
            result.P, result.P.mH, rtol=1e-10, atol=1e-10
        )

    def test_scipy_comparison(self):
        """Compare with scipy.linalg.polar."""
        torch.manual_seed(101)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a, side="right")

        # Get scipy result (right polar is default)
        U_scipy, P_scipy = scipy.linalg.polar(a.numpy(), side="right")

        # Compare U matrices (may differ by sign of columns for SVD-based methods)
        # So we compare via reconstruction instead
        reconstructed_ours = result.U @ result.P
        reconstructed_scipy = torch.from_numpy(U_scipy @ P_scipy)
        torch.testing.assert_close(
            reconstructed_ours, reconstructed_scipy, rtol=1e-10, atol=1e-10
        )

        # P should match closely since it's unique for full-rank matrices
        torch.testing.assert_close(
            result.P, torch.from_numpy(P_scipy), rtol=1e-10, atol=1e-10
        )

    def test_scipy_comparison_left(self):
        """Compare left polar with scipy.linalg.polar."""
        torch.manual_seed(101)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a, side="left")

        # Get scipy result
        U_scipy, P_scipy = scipy.linalg.polar(a.numpy(), side="left")

        # Compare via reconstruction
        reconstructed_ours = result.P @ result.U
        reconstructed_scipy = torch.from_numpy(P_scipy @ U_scipy)
        torch.testing.assert_close(
            reconstructed_ours, reconstructed_scipy, rtol=1e-10, atol=1e-10
        )

        # P should match
        torch.testing.assert_close(
            result.P, torch.from_numpy(P_scipy), rtol=1e-10, atol=1e-10
        )

    def test_batched(self):
        """Test batched polar decomposition."""
        torch.manual_seed(202)
        batch_size = 3
        n = 4

        a = torch.randn(batch_size, n, n, dtype=torch.float64)

        result = polar_decomposition(a)

        assert result.U.shape == (batch_size, n, n)
        assert result.P.shape == (batch_size, n, n)
        assert result.info.shape == (batch_size,)

        # Verify reconstruction, unitarity, and positive semidefiniteness for each batch
        for i in range(batch_size):
            # Reconstruction
            reconstructed = result.U[i] @ result.P[i]
            torch.testing.assert_close(
                reconstructed, a[i], rtol=1e-10, atol=1e-10
            )

            # Hermitian P
            torch.testing.assert_close(
                result.P[i], result.P[i].mH, rtol=1e-10, atol=1e-10
            )

            # Positive semidefinite P
            eigenvalues = torch.linalg.eigvalsh(result.P[i])
            assert torch.all(eigenvalues >= -1e-10)

    def test_batched_multi_dim(self):
        """Test with multiple batch dimensions."""
        torch.manual_seed(303)
        n = 3

        a = torch.randn(2, 3, n, n, dtype=torch.float64)

        result = polar_decomposition(a)

        assert result.U.shape == (2, 3, n, n)
        assert result.P.shape == (2, 3, n, n)
        assert result.info.shape == (2, 3)

        # Check reconstruction for a sample
        reconstructed = result.U[1, 2] @ result.P[1, 2]
        torch.testing.assert_close(
            reconstructed, a[1, 2], rtol=1e-10, atol=1e-10
        )

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        a = torch.randn(4, 4, dtype=torch.float64, device="meta")

        result = polar_decomposition(a)

        assert result.U.shape == (4, 4)
        assert result.P.shape == (4, 4)
        assert result.info.shape == ()
        assert result.U.device.type == "meta"
        assert result.P.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        a = torch.randn(2, 3, 5, 5, dtype=torch.float64, device="meta")

        result = polar_decomposition(a)

        assert result.U.shape == (2, 3, 5, 5)
        assert result.P.shape == (2, 3, 5, 5)
        assert result.info.shape == (2, 3)

    def test_meta_tensor_left(self):
        """Test shape inference for left polar with meta tensors."""
        a = torch.randn(3, 4, dtype=torch.float64, device="meta")

        result = polar_decomposition(a, side="left")

        assert result.U.shape == (3, 4)
        assert result.P.shape == (3, 3)  # Left polar: P is (m, m)
        assert result.info.shape == ()

    def test_compile(self):
        """Test torch.compile compatibility."""
        torch.manual_seed(404)

        @torch.compile
        def compiled_polar(x):
            return polar_decomposition(x)

        a = torch.randn(4, 4, dtype=torch.float64)

        result = compiled_polar(a)

        # Verify basic correctness
        reconstructed = result.U @ result.P
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_complex_input(self):
        """Test with complex input."""
        torch.manual_seed(505)
        n = 4
        a = torch.randn(n, n, dtype=torch.complex128)

        result = polar_decomposition(a)

        assert result.U.dtype == torch.complex128
        assert result.P.dtype == torch.complex128

        # Reconstruction
        reconstructed = result.U @ result.P
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

        # Hermitian P
        torch.testing.assert_close(
            result.P, result.P.mH, rtol=1e-10, atol=1e-10
        )

        # Unitary U
        identity = torch.eye(n, dtype=torch.complex128)
        torch.testing.assert_close(
            result.U @ result.U.mH, identity, rtol=1e-10, atol=1e-10
        )

    def test_rectangular_tall(self):
        """Test with tall rectangular matrix (m > n)."""
        torch.manual_seed(606)
        m, n = 5, 3
        a = torch.randn(m, n, dtype=torch.float64)

        result = polar_decomposition(a, side="right")

        assert result.U.shape == (m, n)
        assert result.P.shape == (n, n)  # Right polar: P is (n, n)

        # Reconstruction
        reconstructed = result.U @ result.P
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

        # P is Hermitian
        torch.testing.assert_close(
            result.P, result.P.mH, rtol=1e-10, atol=1e-10
        )

        # P is positive semidefinite
        eigenvalues = torch.linalg.eigvalsh(result.P)
        assert torch.all(eigenvalues >= -1e-10)

    def test_rectangular_wide(self):
        """Test with wide rectangular matrix (m < n)."""
        torch.manual_seed(707)
        m, n = 3, 5
        a = torch.randn(m, n, dtype=torch.float64)

        result = polar_decomposition(a, side="left")

        assert result.U.shape == (m, n)
        assert result.P.shape == (m, m)  # Left polar: P is (m, m)

        # Reconstruction
        reconstructed = result.P @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

        # P is Hermitian
        torch.testing.assert_close(
            result.P, result.P.mH, rtol=1e-10, atol=1e-10
        )

        # P is positive semidefinite
        eigenvalues = torch.linalg.eigvalsh(result.P)
        assert torch.all(eigenvalues >= -1e-10)

    def test_identity_matrix(self):
        """Test with identity matrix."""
        n = 4
        a = torch.eye(n, dtype=torch.float64)

        result = polar_decomposition(a)

        # For identity: U = I, P = I
        identity = torch.eye(n, dtype=torch.float64)
        torch.testing.assert_close(result.U, identity, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(result.P, identity, rtol=1e-10, atol=1e-10)

    def test_orthogonal_matrix(self):
        """Test with an orthogonal matrix."""
        torch.manual_seed(808)
        n = 4
        # Create orthogonal matrix via QR
        q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))

        result = polar_decomposition(q)

        # For orthogonal matrix: U = Q, P = I
        identity = torch.eye(n, dtype=torch.float64)
        torch.testing.assert_close(result.P, identity, rtol=1e-10, atol=1e-10)

        # U should equal Q (up to sign)
        # Check via U @ U.T = I
        torch.testing.assert_close(
            result.U @ result.U.mH, identity, rtol=1e-10, atol=1e-10
        )

    def test_symmetric_positive_definite(self):
        """Test with symmetric positive definite matrix."""
        torch.manual_seed(909)
        n = 4
        # Create SPD matrix
        a = torch.randn(n, n, dtype=torch.float64)
        a = a @ a.T + torch.eye(n, dtype=torch.float64)

        result = polar_decomposition(a)

        # For SPD matrix: U = I, P = A
        identity = torch.eye(n, dtype=torch.float64)
        torch.testing.assert_close(result.U, identity, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(result.P, a, rtol=1e-10, atol=1e-10)

    def test_invalid_1d_input(self):
        """Test error on 1D input."""
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="must be at least 2D"):
            polar_decomposition(a)

    def test_invalid_side(self):
        """Test error on invalid side parameter."""
        a = torch.randn(3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="side must be"):
            polar_decomposition(a, side="invalid")

    def test_float32(self):
        """Test with float32 dtype."""
        torch.manual_seed(1010)
        n = 4
        a = torch.randn(n, n, dtype=torch.float32)

        result = polar_decomposition(a)

        assert result.U.dtype == torch.float32
        assert result.P.dtype == torch.float32

        # Lower tolerance for float32
        reconstructed = result.U @ result.P
        torch.testing.assert_close(reconstructed, a, rtol=1e-4, atol=1e-4)

    def test_p_squared_equals_ata(self):
        """Test that P^2 = A^T A for right polar."""
        torch.manual_seed(1111)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a, side="right")

        P_squared = result.P @ result.P
        AtA = a.mH @ a
        torch.testing.assert_close(P_squared, AtA, rtol=1e-10, atol=1e-10)

    def test_p_squared_equals_aat(self):
        """Test that P^2 = A A^T for left polar."""
        torch.manual_seed(1111)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)

        result = polar_decomposition(a, side="left")

        P_squared = result.P @ result.P
        AAt = a @ a.mH
        torch.testing.assert_close(P_squared, AAt, rtol=1e-10, atol=1e-10)

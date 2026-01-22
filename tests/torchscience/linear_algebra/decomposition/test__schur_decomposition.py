"""Tests for Schur decomposition."""

import pytest
import scipy.linalg
import torch

from torchscience.linear_algebra.decomposition import schur_decomposition


class TestSchurDecomposition:
    """Tests for Schur decomposition."""

    def test_basic_real_schur(self):
        """Test basic real Schur decomposition."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [0.0, 4.0, 5.0],
                [0.0, 0.0, 6.0],
            ],
            dtype=torch.float64,
        )

        result = schur_decomposition(a, output="real")

        assert result.T.shape == (3, 3)
        assert result.Q.shape == (3, 3)
        assert result.eigenvalues.shape == (3,)
        assert result.info.item() == 0

        # Verify A = Q @ T @ Q^T
        reconstructed = result.Q @ result.T @ result.Q.mH
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_complex_schur(self):
        """Test complex Schur decomposition."""
        a = torch.tensor(
            [
                [0.0, -1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float64,
        )

        result = schur_decomposition(a, output="complex")

        # Eigenvalues should be +/- i
        assert result.T.is_complex()
        reconstructed = result.Q @ result.T @ result.Q.mH
        torch.testing.assert_close(
            reconstructed.real, a, rtol=1e-10, atol=1e-10
        )

    def test_scipy_comparison(self):
        """Compare with scipy.linalg.schur."""
        torch.manual_seed(42)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)

        result = schur_decomposition(a, output="complex")

        T_scipy, Q_scipy = scipy.linalg.schur(a.numpy(), output="complex")

        # Eigenvalues should match (may be in different order)
        torch.testing.assert_close(
            torch.sort(result.eigenvalues.real).values,
            torch.sort(torch.from_numpy(T_scipy.diagonal().real)).values,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_batched(self):
        """Test batched Schur decomposition."""
        torch.manual_seed(123)
        batch_size = 2
        n = 3

        a = torch.randn(batch_size, n, n, dtype=torch.float64)

        result = schur_decomposition(a, output="complex")

        assert result.T.shape == (batch_size, n, n)
        assert result.Q.shape == (batch_size, n, n)
        assert result.eigenvalues.shape == (batch_size, n)

        # Verify reconstruction for each batch
        for i in range(batch_size):
            reconstructed = result.Q[i] @ result.T[i] @ result.Q[i].mH
            torch.testing.assert_close(
                reconstructed.real, a[i], rtol=1e-10, atol=1e-10
            )

    def test_invalid_input_1d(self):
        """Test error on 1D input."""
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="must be at least 2D"):
            schur_decomposition(a)

    def test_invalid_input_non_square(self):
        """Test error on non-square input."""
        a = torch.randn(3, 4)
        with pytest.raises(ValueError, match="must be square"):
            schur_decomposition(a)

    def test_invalid_output_mode(self):
        """Test error on invalid output mode."""
        a = torch.randn(3, 3)
        with pytest.raises(ValueError, match="must be 'real' or 'complex'"):
            schur_decomposition(a, output="invalid")

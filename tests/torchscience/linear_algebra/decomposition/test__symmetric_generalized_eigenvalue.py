"""Tests for symmetric_generalized_eigenvalue decomposition."""

import torch

from torchscience.linear_algebra.decomposition import (
    symmetric_generalized_eigenvalue,
)


class TestSymmetricGeneralizedEigenvalue:
    """Tests for symmetric generalized eigenvalue decomposition."""

    def test_basic_2x2(self):
        """Test basic 2x2 symmetric generalized eigenvalue problem."""
        # A is symmetric, B is symmetric positive definite
        a = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        b = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        )  # Identity

        result = symmetric_generalized_eigenvalue(a, b)

        # With B=I, this reduces to standard eigenvalue problem
        # Eigenvalues of [[2, 1], [1, 3]] are (5 +/- sqrt(5))/2
        expected_eigenvalues = torch.linalg.eigvalsh(a)

        assert result.eigenvalues.shape == (2,)
        assert result.eigenvectors.shape == (2, 2)
        assert result.info.item() == 0
        torch.testing.assert_close(
            torch.sort(result.eigenvalues).values,
            torch.sort(expected_eigenvalues).values,
            rtol=1e-10,
            atol=1e-10,
        )

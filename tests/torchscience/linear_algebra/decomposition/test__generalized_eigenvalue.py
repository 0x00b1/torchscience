"""Tests for generalized eigenvalue decomposition."""

import torch

from torchscience.linear_algebra.decomposition import generalized_eigenvalue


class TestGeneralizedEigenvalue:
    """Tests for generalized eigenvalue decomposition."""

    def test_basic(self):
        """Test basic generalized eigenvalue problem."""
        a = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float64,
        )
        b = torch.tensor(
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=torch.float64,
        )

        result = generalized_eigenvalue(a, b)

        assert result.eigenvalues.shape == (2,)
        assert result.eigenvectors_left.shape == (2, 2)
        assert result.eigenvectors_right.shape == (2, 2)
        assert result.info.item() == 0

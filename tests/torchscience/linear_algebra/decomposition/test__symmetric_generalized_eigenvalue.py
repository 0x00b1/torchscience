"""Tests for symmetric_generalized_eigenvalue decomposition."""

import scipy.linalg
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

    def test_reconstruction(self):
        """Test that A @ v = lambda * B @ v for each eigenpair."""
        a = torch.tensor(
            [
                [4.0, 2.0, 1.0],
                [2.0, 5.0, 2.0],
                [1.0, 2.0, 6.0],
            ],
            dtype=torch.float64,
        )
        b = torch.tensor(
            [
                [2.0, 0.5, 0.0],
                [0.5, 2.0, 0.5],
                [0.0, 0.5, 2.0],
            ],
            dtype=torch.float64,
        )

        result = symmetric_generalized_eigenvalue(a, b)

        # Check A @ v_i = lambda_i * B @ v_i for each eigenvector
        for i in range(3):
            v = result.eigenvectors[:, i]
            lam = result.eigenvalues[i]
            lhs = a @ v
            rhs = lam * (b @ v)
            torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)

    def test_scipy_comparison(self):
        """Compare results with scipy.linalg.eigh."""
        torch.manual_seed(42)
        n = 5

        # Generate random symmetric A and SPD B
        X = torch.randn(n, n, dtype=torch.float64)
        a = X @ X.T + torch.eye(n, dtype=torch.float64)

        Y = torch.randn(n, n, dtype=torch.float64)
        b = Y @ Y.T + torch.eye(n, dtype=torch.float64)

        result = symmetric_generalized_eigenvalue(a, b)

        # Compare with scipy
        scipy_eigenvalues, scipy_eigenvectors = scipy.linalg.eigh(
            a.numpy(), b.numpy()
        )

        torch.testing.assert_close(
            result.eigenvalues,
            torch.from_numpy(scipy_eigenvalues),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_batched(self):
        """Test batched input."""
        batch_size = 3
        n = 4

        torch.manual_seed(123)

        # Generate batch of problems
        a_list = []
        b_list = []
        for _ in range(batch_size):
            X = torch.randn(n, n, dtype=torch.float64)
            a_list.append(X @ X.T + torch.eye(n, dtype=torch.float64))
            Y = torch.randn(n, n, dtype=torch.float64)
            b_list.append(Y @ Y.T + torch.eye(n, dtype=torch.float64))

        a = torch.stack(a_list)
        b = torch.stack(b_list)

        result = symmetric_generalized_eigenvalue(a, b)

        assert result.eigenvalues.shape == (batch_size, n)
        assert result.eigenvectors.shape == (batch_size, n, n)
        assert result.info.shape == (batch_size,)

        # Verify each batch element matches unbatched
        for i in range(batch_size):
            single_result = symmetric_generalized_eigenvalue(a[i], b[i])
            torch.testing.assert_close(
                result.eigenvalues[i],
                single_result.eigenvalues,
                rtol=1e-10,
                atol=1e-10,
            )

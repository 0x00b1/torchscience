"""Tests for generalized eigenvalue decomposition."""

import pytest
import scipy.linalg
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

    def test_reconstruction(self):
        """Test A @ vr = B @ vr @ diag(eigenvalues)."""
        torch.manual_seed(42)
        n = 4

        # Create random matrices
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)
        # Make B invertible by adding identity scaled by condition
        b = b + 2.0 * torch.eye(n, dtype=torch.float64)

        result = generalized_eigenvalue(a, b)

        # Convert a and b to complex for comparison with complex eigenvectors
        a_complex = a.to(torch.complex128)
        b_complex = b.to(torch.complex128)

        # Verify A @ vr[:, i] = eigenvalues[i] * B @ vr[:, i] for each eigenvector
        for i in range(n):
            v = result.eigenvectors_right[:, i]
            lam = result.eigenvalues[i]
            lhs = a_complex @ v
            rhs = lam * (b_complex @ v)
            torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)

    def test_scipy_comparison(self):
        """Compare results against scipy.linalg.eig."""
        torch.manual_seed(123)
        n = 5

        # Create random matrices
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)
        # Make B well-conditioned
        b = b + 3.0 * torch.eye(n, dtype=torch.float64)

        result = generalized_eigenvalue(a, b)

        # Compute using scipy directly
        scipy_eigenvalues, scipy_vl, scipy_vr = scipy.linalg.eig(
            a.numpy(), b.numpy(), left=True, right=True
        )

        # Compare eigenvalues (need to sort since order isn't guaranteed)
        # Sort by real part first, then by imaginary part for stability
        # Use a combined key: real * large_constant + imag to break ties
        our_eigenvalues = result.eigenvalues
        our_sort_key = our_eigenvalues.real * 1e10 + our_eigenvalues.imag
        our_sorted_idx = torch.argsort(our_sort_key)
        our_sorted = our_eigenvalues[our_sorted_idx]

        scipy_eigenvalues_tensor = torch.from_numpy(scipy_eigenvalues)
        scipy_sort_key = (
            scipy_eigenvalues_tensor.real * 1e10
            + scipy_eigenvalues_tensor.imag
        )
        scipy_sorted_idx = torch.argsort(scipy_sort_key)
        scipy_sorted = scipy_eigenvalues_tensor[scipy_sorted_idx]

        torch.testing.assert_close(
            our_sorted, scipy_sorted, rtol=1e-10, atol=1e-10
        )

    def test_batched(self):
        """Test batched input."""
        torch.manual_seed(456)
        batch_shape = (2, 3)
        n = 4

        # Create batched random matrices
        a = torch.randn(*batch_shape, n, n, dtype=torch.float64)
        b = torch.randn(*batch_shape, n, n, dtype=torch.float64)
        # Make B well-conditioned
        b = b + 3.0 * torch.eye(n, dtype=torch.float64)

        result = generalized_eigenvalue(a, b)

        # Verify output shapes
        assert result.eigenvalues.shape == (*batch_shape, n)
        assert result.eigenvectors_left.shape == (*batch_shape, n, n)
        assert result.eigenvectors_right.shape == (*batch_shape, n, n)
        assert result.info.shape == batch_shape

        # Verify results match loop over individual matrices
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                single_result = generalized_eigenvalue(a[i, j], b[i, j])

                # Sort eigenvalues for comparison since order may differ
                # Use a combined key: real * large_constant + imag to break ties
                batch_eigenvalues = result.eigenvalues[i, j]
                batch_sort_key = (
                    batch_eigenvalues.real * 1e10 + batch_eigenvalues.imag
                )
                batch_sorted_idx = torch.argsort(batch_sort_key)
                batch_sorted = batch_eigenvalues[batch_sorted_idx]

                single_eigenvalues = single_result.eigenvalues
                single_sort_key = (
                    single_eigenvalues.real * 1e10 + single_eigenvalues.imag
                )
                single_sorted_idx = torch.argsort(single_sort_key)
                single_sorted = single_eigenvalues[single_sorted_idx]

                torch.testing.assert_close(
                    batch_sorted, single_sorted, rtol=1e-10, atol=1e-10
                )

    def test_complex_input(self):
        """Test with complex-valued matrices."""
        torch.manual_seed(789)
        n = 3

        # Create complex matrices
        a_real = torch.randn(n, n, dtype=torch.float64)
        a_imag = torch.randn(n, n, dtype=torch.float64)
        a = torch.complex(a_real, a_imag)

        b_real = torch.randn(n, n, dtype=torch.float64)
        b_imag = torch.randn(n, n, dtype=torch.float64)
        b = torch.complex(b_real, b_imag)
        # Make B well-conditioned
        b = b + 3.0 * torch.eye(n, dtype=torch.complex128)

        result = generalized_eigenvalue(a, b)

        assert result.eigenvalues.shape == (n,)
        assert result.eigenvectors_left.shape == (n, n)
        assert result.eigenvectors_right.shape == (n, n)
        assert result.info.item() == 0

        # Verify reconstruction: A @ vr[:, i] = eigenvalues[i] * B @ vr[:, i]
        for i in range(n):
            v = result.eigenvectors_right[:, i]
            lam = result.eigenvalues[i]
            lhs = a @ v
            rhs = lam * (b @ v)
            torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)

    def test_singular_b(self):
        """Test behavior when B is singular."""
        n = 3
        a = torch.eye(n, dtype=torch.float64)
        # Create a singular B matrix (rank deficient)
        b = torch.zeros(n, n, dtype=torch.float64)
        b[0, 0] = 1.0
        b[1, 1] = 1.0
        # Last row/column all zeros makes B singular

        result = generalized_eigenvalue(a, b)

        # For singular B, either info should be non-zero OR
        # eigenvalues should contain inf values
        has_inf = torch.any(torch.isinf(result.eigenvalues.real)) or torch.any(
            torch.isinf(result.eigenvalues.imag)
        )
        has_nan = torch.any(torch.isnan(result.eigenvalues.real)) or torch.any(
            torch.isnan(result.eigenvalues.imag)
        )
        info_nonzero = result.info.item() != 0

        # At least one of these conditions should be true for singular B
        assert has_inf or has_nan or info_nonzero

    def test_invalid_input_1d(self):
        """Test error on 1D input."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="must be at least 2D"):
            generalized_eigenvalue(a, b)

    def test_invalid_input_non_square(self):
        """Test error on non-square A."""
        a = torch.randn(3, 4, dtype=torch.float64)
        b = torch.eye(4, dtype=torch.float64)
        with pytest.raises(ValueError, match="must be square"):
            generalized_eigenvalue(a, b)

    def test_size_mismatch(self):
        """Test error when A and B have different sizes."""
        a = torch.eye(3, dtype=torch.float64)
        b = torch.eye(4, dtype=torch.float64)
        with pytest.raises(ValueError, match="must have same size"):
            generalized_eigenvalue(a, b)

    def test_dtype_float32(self):
        """Test float32 input."""
        a = torch.tensor([[2.0, 1.0], [3.0, 4.0]], dtype=torch.float32)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        b = b + 2.0 * torch.eye(2, dtype=torch.float32)

        result = generalized_eigenvalue(a, b)

        assert result.eigenvalues.dtype == torch.complex64
        assert result.info.item() == 0

    def test_backward_not_implemented(self):
        """Test that backward returns None gradients (not yet implemented)."""
        torch.manual_seed(789)
        n = 2
        a = torch.randn(n, n, dtype=torch.float64, requires_grad=True)
        b = torch.randn(n, n, dtype=torch.float64)
        b = b + 2.0 * torch.eye(n, dtype=torch.float64)

        result = generalized_eigenvalue(a, b)
        loss = result.eigenvalues.real.sum()
        loss.backward()

        assert a.grad is None

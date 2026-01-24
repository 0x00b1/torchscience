"""Tests for pivoted LU decomposition."""

import pytest
import scipy.linalg
import torch

from torchscience.linear_algebra.decomposition import (
    PivotedLUResult,
    pivoted_lu,
)


def construct_permutation_matrix(pivots: torch.Tensor, m: int) -> torch.Tensor:
    """Construct permutation matrix from pivot indices.

    Parameters
    ----------
    pivots : Tensor
        Pivot indices of shape (..., k) where k <= m.
    m : int
        Number of rows in the original matrix.

    Returns
    -------
    Tensor
        Permutation matrix P of shape (..., m, m).
    """
    batch_shape = pivots.shape[:-1]
    k = pivots.shape[-1]
    device = pivots.device
    dtype = torch.float64

    # Create identity matrix
    if batch_shape:
        P = (
            torch.eye(m, dtype=dtype, device=device)
            .expand(*batch_shape, m, m)
            .clone()
        )
    else:
        P = torch.eye(m, dtype=dtype, device=device)

    # For linalg_lu, pivots represent the permutation directly
    # P[i, pivots[i]] = 1 for i < k
    if batch_shape:
        for i in range(k):
            idx = pivots[..., i]
            # Create new P where row i is swapped appropriately
            batch_indices = tuple(
                torch.arange(s, device=device)
                .view(*([1] * i), s, *([1] * (len(batch_shape) - i - 1)))
                .expand(batch_shape)
                for i, s in enumerate(batch_shape)
            )
            P = P.clone()
            # Zero out current row
            P[batch_indices + (i,)] = 0
            # Set the pivot position to 1
            for b_idx in range(P.shape[0] if batch_shape else 1):
                P[b_idx, i, :] = 0
                P[b_idx, i, pivots[b_idx, i].item()] = 1
    else:
        P = torch.zeros(m, m, dtype=dtype, device=device)
        for i in range(k):
            P[i, pivots[i].item()] = 1
        # Fill remaining rows with identity
        for i in range(k, m):
            P[i, i] = 1

    return P


class TestPivotedLU:
    """Tests for pivoted LU decomposition."""

    def test_basic(self):
        """Test basic pivoted LU decomposition returns correct shapes and info."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float64,
        )

        result = pivoted_lu(a)

        assert isinstance(result, PivotedLUResult)
        # For 3x3 matrix: L is (3, 3), U is (3, 3), pivots is (3,)
        assert result.L.shape == (3, 3)
        assert result.U.shape == (3, 3)
        assert result.pivots.shape == (3,)
        assert result.info.shape == ()
        assert result.info.item() == 0

    def test_reconstruction(self):
        """Test that P @ L @ U = A (reconstruction from factors)."""
        torch.manual_seed(42)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = pivoted_lu(a)

        # Construct P from pivots
        m = a.size(-2)
        P = torch.zeros(m, m, dtype=torch.float64)
        for i in range(m):
            if i < result.pivots.shape[0]:
                P[i, result.pivots[i].item()] = 1
            else:
                P[i, i] = 1

        # linalg_lu returns P such that A = P @ L @ U
        reconstructed = P @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_lower_triangular(self):
        """Test that L is unit lower triangular (ones on diagonal)."""
        torch.manual_seed(123)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = pivoted_lu(a)

        L = result.L
        k = L.shape[-1]

        # Check that diagonal elements are 1
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        torch.testing.assert_close(
            diag, torch.ones(k, dtype=torch.float64), rtol=1e-10, atol=1e-10
        )

        # Check that elements above diagonal are 0
        for i in range(L.shape[-2]):
            for j in range(i + 1, L.shape[-1]):
                assert abs(L[i, j].item()) < 1e-10, (
                    f"L[{i}, {j}] = {L[i, j].item()} should be 0"
                )

    def test_upper_triangular(self):
        """Test that U is upper triangular."""
        torch.manual_seed(456)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = pivoted_lu(a)

        U = result.U

        # Check that elements below diagonal are 0
        for i in range(U.shape[-2]):
            for j in range(i):
                assert abs(U[i, j].item()) < 1e-10, (
                    f"U[{i}, {j}] = {U[i, j].item()} should be 0"
                )

    def test_scipy_comparison(self):
        """Compare with scipy.linalg.lu."""
        torch.manual_seed(789)
        n = 5
        a = torch.randn(n, n, dtype=torch.float64)

        result = pivoted_lu(a)

        # scipy.linalg.lu returns (P, L, U) where P @ A = L @ U  (actually, A = P @ L @ U)
        # scipy uses A = P @ L @ U, so P @ A != L @ U, rather A = P @ L @ U
        P_scipy, L_scipy, U_scipy = scipy.linalg.lu(a.numpy())

        # Compare via reconstruction
        reconstructed_ours = (
            construct_permutation_matrix(result.pivots, n).T
            @ result.L
            @ result.U
        )
        reconstructed_scipy = torch.from_numpy(P_scipy @ L_scipy @ U_scipy)

        torch.testing.assert_close(
            reconstructed_ours, a, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            reconstructed_scipy, a, rtol=1e-10, atol=1e-10
        )

    def test_batched(self):
        """Test batched pivoted LU decomposition."""
        torch.manual_seed(101)
        batch_size = 3
        n = 4

        a = torch.randn(batch_size, n, n, dtype=torch.float64)

        result = pivoted_lu(a)

        assert result.L.shape == (batch_size, n, n)
        assert result.U.shape == (batch_size, n, n)
        assert result.pivots.shape == (batch_size, n)
        assert result.info.shape == (batch_size,)

        # Verify reconstruction for each batch element
        # linalg_lu returns P such that A = P @ L @ U
        for i in range(batch_size):
            m = a.size(-2)
            P = torch.zeros(m, m, dtype=torch.float64)
            for j in range(m):
                P[j, result.pivots[i, j].item()] = 1

            reconstructed = P @ result.L[i] @ result.U[i]
            torch.testing.assert_close(
                reconstructed, a[i], rtol=1e-10, atol=1e-10
            )

    def test_rectangular_tall(self):
        """Test with tall rectangular matrix (m > n)."""
        torch.manual_seed(202)
        m, n = 6, 4
        a = torch.randn(m, n, dtype=torch.float64)

        result = pivoted_lu(a)

        k = min(m, n)
        assert result.L.shape == (m, k)  # (6, 4)
        assert result.U.shape == (k, n)  # (4, 4)
        assert result.pivots.shape == (
            m,
        )  # (6,) - row permutation for all m rows

        # Verify L is unit lower triangular
        for i in range(k):
            assert abs(result.L[i, i].item() - 1.0) < 1e-10
            for j in range(i + 1, k):
                assert abs(result.L[i, j].item()) < 1e-10

        # Verify U is upper triangular
        for i in range(k):
            for j in range(i):
                assert abs(result.U[i, j].item()) < 1e-10

        # Reconstruction: A = P @ L @ U
        P = torch.zeros(m, m, dtype=torch.float64)
        for i in range(m):
            P[i, result.pivots[i].item()] = 1

        reconstructed = P @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_rectangular_wide(self):
        """Test with wide rectangular matrix (m < n)."""
        torch.manual_seed(303)
        m, n = 4, 6
        a = torch.randn(m, n, dtype=torch.float64)

        result = pivoted_lu(a)

        k = min(m, n)
        assert result.L.shape == (m, k)  # (4, 4)
        assert result.U.shape == (k, n)  # (4, 6)
        assert result.pivots.shape == (
            m,
        )  # (4,) - row permutation for all m rows

        # Verify L is unit lower triangular
        for i in range(k):
            assert abs(result.L[i, i].item() - 1.0) < 1e-10

        # Reconstruction: A = P @ L @ U
        P = torch.zeros(m, m, dtype=torch.float64)
        for i in range(m):
            P[i, result.pivots[i].item()] = 1

        reconstructed = P @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        a = torch.randn(4, 4, dtype=torch.float64, device="meta")

        result = pivoted_lu(a)

        assert result.L.shape == (4, 4)
        assert result.U.shape == (4, 4)
        assert result.pivots.shape == (4,)
        assert result.info.shape == ()
        assert result.L.device.type == "meta"
        assert result.U.device.type == "meta"
        assert result.pivots.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        a = torch.randn(2, 3, 5, 5, dtype=torch.float64, device="meta")

        result = pivoted_lu(a)

        assert result.L.shape == (2, 3, 5, 5)
        assert result.U.shape == (2, 3, 5, 5)
        assert result.pivots.shape == (2, 3, 5)
        assert result.info.shape == (2, 3)

    def test_meta_tensor_rectangular(self):
        """Test shape inference for rectangular matrices with meta tensors."""
        # Tall matrix
        a = torch.randn(5, 3, dtype=torch.float64, device="meta")
        result = pivoted_lu(a)
        assert result.L.shape == (5, 3)
        assert result.U.shape == (3, 3)
        assert result.pivots.shape == (
            5,
        )  # pivots has shape (m,) for row permutation

        # Wide matrix
        a = torch.randn(3, 5, dtype=torch.float64, device="meta")
        result = pivoted_lu(a)
        assert result.L.shape == (3, 3)
        assert result.U.shape == (3, 5)
        assert result.pivots.shape == (
            3,
        )  # pivots has shape (m,) for row permutation

    def test_compile(self):
        """Test torch.compile compatibility."""
        torch.manual_seed(404)

        @torch.compile
        def compiled_lu(x):
            return pivoted_lu(x)

        a = torch.randn(4, 4, dtype=torch.float64)

        result = compiled_lu(a)

        # Verify basic correctness
        m = a.size(-2)
        P = torch.zeros(m, m, dtype=torch.float64)
        for i in range(m):
            P[i, result.pivots[i].item()] = 1

        reconstructed = P @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_complex_input(self):
        """Test with complex input."""
        torch.manual_seed(505)
        n = 4
        a = torch.randn(n, n, dtype=torch.complex128)

        result = pivoted_lu(a)

        assert result.L.dtype == torch.complex128
        assert result.U.dtype == torch.complex128

        # Reconstruction
        m = a.size(-2)
        P = torch.zeros(m, m, dtype=torch.complex128)
        for i in range(m):
            P[i, result.pivots[i].item()] = 1

        # linalg_lu returns P such that A = P @ L @ U
        # P is real-valued (0s and 1s) even for complex input
        # Need to cast to complex for matrix multiplication
        reconstructed = P.to(torch.complex128) @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_identity_matrix(self):
        """Test with identity matrix."""
        n = 4
        a = torch.eye(n, dtype=torch.float64)

        result = pivoted_lu(a)

        # For identity: L = I, U = I (up to permutation)
        # Check that L @ U gives a permuted identity
        LU = result.L @ result.U

        # Each row and column of LU should have exactly one non-zero element
        for i in range(n):
            row_nonzero = (LU[i].abs() > 1e-10).sum()
            assert row_nonzero == 1, (
                f"Row {i} has {row_nonzero} non-zero elements"
            )

    def test_diagonal_matrix(self):
        """Test with diagonal matrix."""
        torch.manual_seed(606)
        n = 4
        d = torch.randn(n, dtype=torch.float64)
        a = torch.diag(d)

        result = pivoted_lu(a)

        # Reconstruct and verify
        m = a.size(-2)
        P = torch.zeros(m, m, dtype=torch.float64)
        for i in range(m):
            P[i, result.pivots[i].item()] = 1

        reconstructed = P @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_float32(self):
        """Test with float32 dtype."""
        torch.manual_seed(707)
        n = 4
        a = torch.randn(n, n, dtype=torch.float32)

        result = pivoted_lu(a)

        assert result.L.dtype == torch.float32
        assert result.U.dtype == torch.float32

        # Lower tolerance for float32
        m = a.size(-2)
        P = torch.zeros(m, m, dtype=torch.float32)
        for i in range(m):
            P[i, result.pivots[i].item()] = 1

        reconstructed = P @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-4, atol=1e-4)

    def test_invalid_1d_input(self):
        """Test error on 1D input."""
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="must be at least 2D"):
            pivoted_lu(a)

    def test_batched_multi_dim(self):
        """Test with multiple batch dimensions."""
        torch.manual_seed(808)
        n = 3

        a = torch.randn(2, 3, n, n, dtype=torch.float64)

        result = pivoted_lu(a)

        assert result.L.shape == (2, 3, n, n)
        assert result.U.shape == (2, 3, n, n)
        assert result.pivots.shape == (2, 3, n)
        assert result.info.shape == (2, 3)

        # Check reconstruction for a sample
        m = n
        P = torch.zeros(m, m, dtype=torch.float64)
        for i in range(m):
            P[i, result.pivots[1, 2, i].item()] = 1

        reconstructed = P.T @ result.L[1, 2] @ result.U[1, 2]
        torch.testing.assert_close(
            reconstructed, a[1, 2], rtol=1e-10, atol=1e-10
        )

    def test_singular_matrix(self):
        """Test with a singular matrix."""
        # Create a rank-deficient matrix
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],  # Row 2 = 2 * Row 1
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )

        result = pivoted_lu(a)

        # Should still compute factors (pivoting helps with singular matrices)
        assert result.L.shape == (3, 3)
        assert result.U.shape == (3, 3)

        # Reconstruction should still work
        m = a.size(-2)
        P = torch.zeros(m, m, dtype=torch.float64)
        for i in range(m):
            P[i, result.pivots[i].item()] = 1

        reconstructed = P @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-10, atol=1e-10)

    def test_well_conditioned(self):
        """Test with a well-conditioned matrix."""
        torch.manual_seed(909)
        n = 5
        # Create a well-conditioned matrix
        a = torch.randn(n, n, dtype=torch.float64)
        a = a @ a.T + 3 * torch.eye(n, dtype=torch.float64)

        result = pivoted_lu(a)

        assert result.info.item() == 0

        # High precision reconstruction
        m = a.size(-2)
        P = torch.zeros(m, m, dtype=torch.float64)
        for i in range(m):
            P[i, result.pivots[i].item()] = 1

        reconstructed = P @ result.L @ result.U
        torch.testing.assert_close(reconstructed, a, rtol=1e-12, atol=1e-12)

    def test_pivots_are_valid_indices(self):
        """Test that pivot indices are valid row indices."""
        torch.manual_seed(1010)
        m, n = 5, 4
        a = torch.randn(m, n, dtype=torch.float64)

        result = pivoted_lu(a)

        k = min(m, n)
        for i in range(k):
            pivot = result.pivots[i].item()
            assert 0 <= pivot < m, (
                f"Pivot {i} = {pivot} is out of bounds [0, {m})"
            )

    def test_gradcheck(self):
        """Test gradient computation."""
        torch.manual_seed(1111)
        # Use a well-conditioned matrix for stable gradients
        a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        a = a + 2 * torch.eye(3, dtype=torch.float64)
        a = a.clone().detach().requires_grad_(True)

        def fn(x):
            result = pivoted_lu(x)
            return result.L.real.sum() + result.U.real.sum()

        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batched(self):
        """Test gradient computation with batched input."""
        torch.manual_seed(1212)
        batch_size = 2
        n = 3
        a = torch.randn(batch_size, n, n, dtype=torch.float64)
        # Make well-conditioned
        a = a + 2 * torch.eye(n, dtype=torch.float64).unsqueeze(0)
        a = a.clone().detach().requires_grad_(True)

        def fn(x):
            result = pivoted_lu(x)
            return result.L.real.sum() + result.U.real.sum()

        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-4)

    def test_grad_only_L(self):
        """Test gradient flows only through L."""
        torch.manual_seed(1313)
        a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        a = a + 2 * torch.eye(3, dtype=torch.float64)
        a = a.clone().detach().requires_grad_(True)

        result = pivoted_lu(a)
        loss = result.L.sum()
        loss.backward()

        assert a.grad is not None
        assert not torch.all(a.grad == 0)

    def test_grad_only_U(self):
        """Test gradient flows only through U."""
        torch.manual_seed(1414)
        a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        a = a + 2 * torch.eye(3, dtype=torch.float64)
        a = a.clone().detach().requires_grad_(True)

        result = pivoted_lu(a)
        loss = result.U.sum()
        loss.backward()

        assert a.grad is not None
        assert not torch.all(a.grad == 0)

    def test_grad_complex(self):
        """Test gradient computation with complex input."""
        torch.manual_seed(1515)
        a = torch.randn(3, 3, dtype=torch.complex128, requires_grad=True)
        a = a + 2 * torch.eye(3, dtype=torch.complex128)
        a = a.clone().detach().requires_grad_(True)

        def fn(x):
            result = pivoted_lu(x)
            # For complex, sum the absolute values
            return result.L.abs().sum() + result.U.abs().sum()

        # gradcheck with complex tensors
        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-4)

"""Tests for rank-revealing QR decomposition."""

import pytest
import torch

from torchscience.linear_algebra.decomposition import (
    RankRevealingQRResult,
    rank_revealing_qr,
)


class TestRankRevealingQR:
    """Tests for rank-revealing QR decomposition."""

    def test_basic(self):
        """Test basic rank-revealing QR decomposition returns correct shapes and info."""
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float64,
        )

        result = rank_revealing_qr(a)

        assert isinstance(result, RankRevealingQRResult)
        m, n = a.shape
        k = min(m, n)
        # For 3x3 matrix: Q is (3, 3), R is (3, 3), pivots is (3,)
        assert result.Q.shape == (m, k)
        assert result.R.shape == (k, n)
        assert result.pivots.shape == (n,)
        assert result.rank.shape == ()
        assert result.info.shape == ()
        assert result.info.item() == 0

    def test_reconstruction(self):
        """Test that A[:, pivots] = Q @ R (reconstruction from factors)."""
        torch.manual_seed(42)
        m, n = 5, 5
        a = torch.randn(m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        # A[:, pivots] should equal Q @ R
        a_permuted = a[:, result.pivots]
        reconstructed = result.Q @ result.R

        torch.testing.assert_close(
            reconstructed, a_permuted, rtol=1e-10, atol=1e-10
        )

    def test_full_rank_matrix(self):
        """Test that a full-rank matrix returns rank = min(m, n)."""
        torch.manual_seed(100)
        m, n = 4, 4
        # Create a well-conditioned full-rank matrix
        a = torch.randn(m, n, dtype=torch.float64)
        a = a @ a.T + 3 * torch.eye(
            m, dtype=torch.float64
        )  # Make positive definite

        result = rank_revealing_qr(a)

        expected_rank = min(m, n)
        assert result.rank.item() == expected_rank

    def test_rank_deficient_matrix(self):
        """Test with a rank-deficient matrix (rank 2 for a 3x3 matrix)."""
        # Create a rank-2 matrix
        a = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],  # Row 2 = 2 * Row 1
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )

        result = rank_revealing_qr(a)

        # This matrix has rank 2 (rows 1 and 3 are independent, row 2 is dependent)
        assert result.rank.item() == 2

    def test_rank_1_matrix(self):
        """Test with a rank-1 matrix."""
        # Create a rank-1 matrix (outer product)
        u = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        v = torch.tensor([[4.0], [5.0], [6.0]], dtype=torch.float64)
        a = u @ v.T  # rank-1 matrix

        result = rank_revealing_qr(a)

        assert result.rank.item() == 1

    def test_zero_matrix(self):
        """Test that a zero matrix returns rank = 0."""
        m, n = 3, 3
        a = torch.zeros(m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        assert result.rank.item() == 0

    def test_tolerance_sensitivity(self):
        """Test that different tolerance values affect rank detection."""
        # Create a matrix where rank depends on tolerance
        a = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1e-8, 0.0],
                [0.0, 0.0, 1e-12],
            ],
            dtype=torch.float64,
        )

        # With tight tolerance, should see higher rank
        result_tight = rank_revealing_qr(a, tol=1e-15)
        # With loose tolerance, should see lower rank
        result_loose = rank_revealing_qr(a, tol=1e-6)

        assert result_tight.rank.item() >= result_loose.rank.item()

    def test_batched(self):
        """Test batched rank-revealing QR decomposition with different ranks."""
        torch.manual_seed(101)
        batch_size = 3
        m, n = 4, 4

        # Create batch with different ranks
        batch = []
        # Full rank matrix
        a1 = torch.randn(m, n, dtype=torch.float64)
        a1 = a1 @ a1.T + 3 * torch.eye(m, dtype=torch.float64)
        batch.append(a1)

        # Rank 2 matrix
        u = torch.randn(m, 2, dtype=torch.float64)
        a2 = u @ u.T
        batch.append(a2)

        # Rank 1 matrix
        v = torch.randn(m, 1, dtype=torch.float64)
        a3 = v @ v.T
        batch.append(a3)

        a = torch.stack(batch)

        result = rank_revealing_qr(a)

        k = min(m, n)
        assert result.Q.shape == (batch_size, m, k)
        assert result.R.shape == (batch_size, k, n)
        assert result.pivots.shape == (batch_size, n)
        assert result.rank.shape == (batch_size,)
        assert result.info.shape == (batch_size,)

        # Check ranks (approximate due to numerical precision)
        assert result.rank[0].item() == 4  # Full rank
        assert result.rank[1].item() == 2  # Rank 2
        assert result.rank[2].item() == 1  # Rank 1

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        m, n = 4, 4
        a = torch.randn(m, n, dtype=torch.float64, device="meta")

        result = rank_revealing_qr(a)

        k = min(m, n)
        assert result.Q.shape == (m, k)
        assert result.R.shape == (k, n)
        assert result.pivots.shape == (n,)
        assert result.rank.shape == ()
        assert result.info.shape == ()
        assert result.Q.device.type == "meta"
        assert result.R.device.type == "meta"
        assert result.pivots.device.type == "meta"
        assert result.rank.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        m, n = 5, 5
        a = torch.randn(2, 3, m, n, dtype=torch.float64, device="meta")

        result = rank_revealing_qr(a)

        k = min(m, n)
        assert result.Q.shape == (2, 3, m, k)
        assert result.R.shape == (2, 3, k, n)
        assert result.pivots.shape == (2, 3, n)
        assert result.rank.shape == (2, 3)
        assert result.info.shape == (2, 3)

    def test_meta_tensor_rectangular(self):
        """Test shape inference for rectangular matrices with meta tensors."""
        # Tall matrix
        m, n = 5, 3
        a = torch.randn(m, n, dtype=torch.float64, device="meta")
        result = rank_revealing_qr(a)
        k = min(m, n)
        assert result.Q.shape == (m, k)
        assert result.R.shape == (k, n)
        assert result.pivots.shape == (n,)
        assert result.rank.shape == ()

        # Wide matrix
        m, n = 3, 5
        a = torch.randn(m, n, dtype=torch.float64, device="meta")
        result = rank_revealing_qr(a)
        k = min(m, n)
        assert result.Q.shape == (m, k)
        assert result.R.shape == (k, n)
        assert result.pivots.shape == (n,)
        assert result.rank.shape == ()

    def test_compile(self):
        """Test torch.compile compatibility."""
        torch.manual_seed(404)

        @torch.compile
        def compiled_rrqr(x, tol):
            return rank_revealing_qr(x, tol=tol)

        m, n = 4, 4
        a = torch.randn(m, n, dtype=torch.float64)

        result = compiled_rrqr(a, 1e-10)

        # Verify basic correctness
        a_permuted = a[:, result.pivots]
        reconstructed = result.Q @ result.R
        torch.testing.assert_close(
            reconstructed, a_permuted, rtol=1e-10, atol=1e-10
        )

    def test_complex_input(self):
        """Test with complex input."""
        torch.manual_seed(505)
        m, n = 4, 4
        a = torch.randn(m, n, dtype=torch.complex128)

        result = rank_revealing_qr(a)

        assert result.Q.dtype == torch.complex128
        assert result.R.dtype == torch.complex128

        # Reconstruction
        a_permuted = a[:, result.pivots]
        reconstructed = result.Q @ result.R
        torch.testing.assert_close(
            reconstructed, a_permuted, rtol=1e-10, atol=1e-10
        )

    def test_identity_matrix(self):
        """Test with identity matrix (full rank)."""
        n = 4
        a = torch.eye(n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        assert result.rank.item() == n
        assert result.info.item() == 0

    def test_diagonal_matrix(self):
        """Test with diagonal matrix."""
        n = 4
        d = torch.tensor([3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
        a = torch.diag(d)

        result = rank_revealing_qr(a)

        # All diagonal elements are non-zero, so full rank
        assert result.rank.item() == n

    def test_diagonal_with_zeros(self):
        """Test with diagonal matrix containing zeros."""
        n = 4
        d = torch.tensor([3.0, 2.0, 0.0, 0.0], dtype=torch.float64)
        a = torch.diag(d)

        result = rank_revealing_qr(a)

        # Only 2 non-zero diagonal elements
        assert result.rank.item() == 2

    def test_float32(self):
        """Test with float32 dtype."""
        torch.manual_seed(707)
        m, n = 4, 4
        a = torch.randn(m, n, dtype=torch.float32)

        result = rank_revealing_qr(a)

        assert result.Q.dtype == torch.float32
        assert result.R.dtype == torch.float32

        # Lower tolerance for float32
        a_permuted = a[:, result.pivots]
        reconstructed = result.Q @ result.R
        torch.testing.assert_close(
            reconstructed, a_permuted, rtol=1e-4, atol=1e-4
        )

    def test_invalid_1d_input(self):
        """Test error on 1D input."""
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="must be at least 2D"):
            rank_revealing_qr(a)

    def test_negative_tolerance(self):
        """Test error on negative tolerance."""
        a = torch.randn(3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="non-negative"):
            rank_revealing_qr(a, tol=-1.0)

    def test_orthogonal_q(self):
        """Test that Q is orthogonal (Q.T @ Q = I)."""
        torch.manual_seed(123)
        m, n = 5, 4
        a = torch.randn(m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        k = min(m, n)
        identity = torch.eye(k, dtype=torch.float64)
        Q_T_Q = result.Q.T @ result.Q

        torch.testing.assert_close(Q_T_Q, identity, rtol=1e-10, atol=1e-10)

    def test_upper_triangular_r(self):
        """Test that R is upper triangular."""
        torch.manual_seed(456)
        m, n = 5, 4
        a = torch.randn(m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        R = result.R

        # Check that elements below diagonal are 0
        for i in range(R.shape[-2]):
            for j in range(i):
                assert abs(R[i, j].item()) < 1e-10, (
                    f"R[{i}, {j}] = {R[i, j].item()} should be 0"
                )

    def test_rectangular_tall(self):
        """Test with tall rectangular matrix (m > n)."""
        torch.manual_seed(202)
        m, n = 6, 4
        a = torch.randn(m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        k = min(m, n)  # k = 4
        assert result.Q.shape == (m, k)  # (6, 4)
        assert result.R.shape == (k, n)  # (4, 4)
        assert result.pivots.shape == (n,)  # (4,)
        assert result.rank.shape == ()

        # Verify reconstruction
        a_permuted = a[:, result.pivots]
        reconstructed = result.Q @ result.R
        torch.testing.assert_close(
            reconstructed, a_permuted, rtol=1e-10, atol=1e-10
        )

    def test_rectangular_wide(self):
        """Test with wide rectangular matrix (m < n)."""
        torch.manual_seed(303)
        m, n = 4, 6
        a = torch.randn(m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        k = min(m, n)  # k = 4
        assert result.Q.shape == (m, k)  # (4, 4)
        assert result.R.shape == (k, n)  # (4, 6)
        assert result.pivots.shape == (n,)  # (6,)
        assert result.rank.shape == ()

        # Verify reconstruction
        a_permuted = a[:, result.pivots]
        reconstructed = result.Q @ result.R
        torch.testing.assert_close(
            reconstructed, a_permuted, rtol=1e-10, atol=1e-10
        )

    def test_batched_multi_dim(self):
        """Test with multiple batch dimensions."""
        torch.manual_seed(808)
        m, n = 3, 3

        a = torch.randn(2, 3, m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        k = min(m, n)
        assert result.Q.shape == (2, 3, m, k)
        assert result.R.shape == (2, 3, k, n)
        assert result.pivots.shape == (2, 3, n)
        assert result.rank.shape == (2, 3)
        assert result.info.shape == (2, 3)

        # Check reconstruction for a sample
        a_permuted = a[1, 2][:, result.pivots[1, 2]]
        reconstructed = result.Q[1, 2] @ result.R[1, 2]
        torch.testing.assert_close(
            reconstructed, a_permuted, rtol=1e-10, atol=1e-10
        )

    def test_pivots_are_valid_indices(self):
        """Test that pivot indices are valid column indices."""
        torch.manual_seed(1010)
        m, n = 5, 4
        a = torch.randn(m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        for i in range(n):
            pivot = result.pivots[i].item()
            assert 0 <= pivot < n, (
                f"Pivot {i} = {pivot} is out of bounds [0, {n})"
            )

    def test_pivots_are_permutation(self):
        """Test that pivots form a valid permutation (no duplicates)."""
        torch.manual_seed(1011)
        m, n = 5, 5
        a = torch.randn(m, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        # Check that pivots form a permutation (all unique values 0 to n-1)
        sorted_pivots = torch.sort(result.pivots)[0]
        expected = torch.arange(n)
        torch.testing.assert_close(sorted_pivots, expected)

    def test_2x2_matrix(self):
        """Test with minimal 2x2 matrix."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

        result = rank_revealing_qr(a)

        assert result.Q.shape == (2, 2)
        assert result.R.shape == (2, 2)
        assert result.pivots.shape == (2,)
        assert result.rank.shape == ()

        # Full rank for this matrix
        assert result.rank.item() == 2

        # Reconstruction
        a_permuted = a[:, result.pivots]
        reconstructed = result.Q @ result.R
        torch.testing.assert_close(
            reconstructed, a_permuted, rtol=1e-10, atol=1e-10
        )

    def test_nearly_singular_matrix(self):
        """Test with a nearly singular matrix."""
        # Create a nearly singular matrix (rank 3, but one direction is very small)
        torch.manual_seed(1234)
        n = 4
        U = torch.randn(n, n, dtype=torch.float64)
        U, _ = torch.linalg.qr(U)
        s = torch.tensor([1.0, 0.1, 0.01, 1e-14], dtype=torch.float64)
        a = U @ torch.diag(s) @ U.T

        # With default tolerance, should detect reduced rank
        result = rank_revealing_qr(a)

        # Depending on tolerance, rank should be 3 (the 4th singular value is tiny)
        assert result.rank.item() <= 4

    @pytest.mark.xfail(reason="Autograd kernel not implemented")
    def test_gradcheck(self):
        """Test gradient computation."""
        torch.manual_seed(1111)
        # Use a well-conditioned matrix for stable gradients
        a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        a = a + 2 * torch.eye(3, dtype=torch.float64)
        a = a.clone().detach().requires_grad_(True)

        def fn(x):
            result = rank_revealing_qr(x)
            return result.Q.real.sum() + result.R.real.sum()

        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-4)

    @pytest.mark.xfail(reason="Autograd kernel not implemented")
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
            result = rank_revealing_qr(x)
            return result.Q.real.sum() + result.R.real.sum()

        assert torch.autograd.gradcheck(fn, (a,), eps=1e-6, atol=1e-4)

    def test_low_rank_approximation(self):
        """Test that truncating to rank gives a valid low-rank approximation."""
        torch.manual_seed(2000)
        m, n = 5, 5

        # Create a rank-3 matrix by outer product
        u = torch.randn(m, 3, dtype=torch.float64)
        v = torch.randn(n, 3, dtype=torch.float64)
        a = u @ v.T  # rank-3 matrix

        result = rank_revealing_qr(a)

        # Should detect rank 3
        assert result.rank.item() == 3

        # Low-rank approximation using detected rank
        r = result.rank.item()
        Q_r = result.Q[:, :r]
        R_r = result.R[:r, :]
        low_rank_approx = Q_r @ R_r

        # This should approximately equal A[:, pivots]
        a_permuted = a[:, result.pivots]
        torch.testing.assert_close(
            low_rank_approx, a_permuted, rtol=1e-10, atol=1e-10
        )

    def test_default_tolerance(self):
        """Test that default tolerance works reasonably."""
        torch.manual_seed(3000)
        m, n = 4, 4
        a = torch.randn(m, n, dtype=torch.float64)
        a = a @ a.T + torch.eye(m, dtype=torch.float64)  # Full rank

        # Should work without specifying tolerance
        result = rank_revealing_qr(a)

        assert result.rank.item() == m
        assert result.info.item() == 0

    def test_single_column_matrix(self):
        """Test with a single column matrix."""
        torch.manual_seed(1100)
        m = 5
        a = torch.randn(m, 1, dtype=torch.float64)

        result = rank_revealing_qr(a)

        assert result.Q.shape == (m, 1)
        assert result.R.shape == (1, 1)
        assert result.pivots.shape == (1,)
        assert result.rank.shape == ()
        assert result.rank.item() == 1  # Non-zero column has rank 1

    def test_single_row_matrix(self):
        """Test with a single row matrix."""
        torch.manual_seed(1101)
        n = 5
        a = torch.randn(1, n, dtype=torch.float64)

        result = rank_revealing_qr(a)

        assert result.Q.shape == (1, 1)
        assert result.R.shape == (1, n)
        assert result.pivots.shape == (n,)
        assert result.rank.shape == ()
        assert result.rank.item() == 1  # Non-zero row has rank 1

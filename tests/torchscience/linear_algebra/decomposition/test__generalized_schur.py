"""Tests for generalized Schur (QZ) decomposition."""

import pytest
import torch

from torchscience.linear_algebra.decomposition import generalized_schur


class TestGeneralizedSchur:
    """Tests for generalized Schur (QZ) decomposition."""

    def test_basic(self):
        """Test basic generalized Schur decomposition returns correct shapes and info."""
        torch.manual_seed(42)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="complex")

        assert result.S.shape == (n, n)
        assert result.T.shape == (n, n)
        assert result.alpha.shape == (n,)
        assert result.beta.shape == (n,)
        assert result.Q.shape == (n, n)
        assert result.Z.shape == (n, n)
        assert result.info.item() == 0

        # Verify complex output
        assert result.S.is_complex()
        assert result.T.is_complex()
        assert result.alpha.is_complex()
        assert result.beta.is_complex()
        assert result.Q.is_complex()
        assert result.Z.is_complex()

    def test_reconstruction_a(self):
        """Verify A = Q @ S @ Z.mH reconstruction."""
        torch.manual_seed(123)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="complex")

        # Verify A = Q @ S @ Z.H
        reconstructed_a = result.Q @ result.S @ result.Z.mH
        a_complex = a.to(torch.complex128)
        torch.testing.assert_close(
            reconstructed_a, a_complex, rtol=1e-10, atol=1e-10
        )

    def test_reconstruction_b(self):
        """Verify B = Q @ T @ Z.mH reconstruction."""
        torch.manual_seed(456)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="complex")

        # Verify B = Q @ T @ Z.H
        reconstructed_b = result.Q @ result.T @ result.Z.mH
        b_complex = b.to(torch.complex128)
        torch.testing.assert_close(
            reconstructed_b, b_complex, rtol=1e-10, atol=1e-10
        )

    def test_unitary_q(self):
        """Verify Q @ Q.mH = I (Q is unitary)."""
        torch.manual_seed(789)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="complex")

        # Verify Q @ Q.H = I
        identity = torch.eye(n, dtype=result.Q.dtype)
        q_unitary = result.Q @ result.Q.mH
        torch.testing.assert_close(q_unitary, identity, rtol=1e-10, atol=1e-10)

    def test_unitary_z(self):
        """Verify Z @ Z.mH = I (Z is unitary)."""
        torch.manual_seed(321)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="complex")

        # Verify Z @ Z.H = I
        identity = torch.eye(n, dtype=result.Z.dtype)
        z_unitary = result.Z @ result.Z.mH
        torch.testing.assert_close(z_unitary, identity, rtol=1e-10, atol=1e-10)

    def test_eigenvalues(self):
        """Verify alpha/beta matches generalized eigenvalues."""
        torch.manual_seed(654)
        n = 3
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="complex")

        # Compute generalized eigenvalues as alpha/beta
        # Avoid division by zero for any zero beta
        nonzero_mask = result.beta.abs() > 1e-10
        computed_eigenvalues = (
            result.alpha[nonzero_mask] / result.beta[nonzero_mask]
        )

        # Compare with eigenvalues from B^{-1}A
        b_inv_a = torch.linalg.solve(b, a)
        reference_eigenvalues, _ = torch.linalg.eig(b_inv_a)

        # Sort by real then imaginary for comparison
        computed_sorted = torch.view_as_real(computed_eigenvalues)
        reference_sorted = torch.view_as_real(reference_eigenvalues)

        # Sort by magnitude
        computed_mags = computed_eigenvalues.abs()
        reference_mags = reference_eigenvalues.abs()

        computed_order = torch.argsort(computed_mags)
        reference_order = torch.argsort(reference_mags)

        torch.testing.assert_close(
            computed_eigenvalues[computed_order].abs(),
            reference_eigenvalues[reference_order].abs(),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_batched(self):
        """Test batched generalized Schur decomposition."""
        torch.manual_seed(111)
        batch_size = 2
        n = 3

        a = torch.randn(batch_size, n, n, dtype=torch.float64)
        b = torch.randn(batch_size, n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="complex")

        assert result.S.shape == (batch_size, n, n)
        assert result.T.shape == (batch_size, n, n)
        assert result.alpha.shape == (batch_size, n)
        assert result.beta.shape == (batch_size, n)
        assert result.Q.shape == (batch_size, n, n)
        assert result.Z.shape == (batch_size, n, n)

        # Verify reconstruction for each batch
        for i in range(batch_size):
            reconstructed_a = result.Q[i] @ result.S[i] @ result.Z[i].mH
            a_complex = a[i].to(torch.complex128)
            torch.testing.assert_close(
                reconstructed_a, a_complex, rtol=1e-10, atol=1e-10
            )

            reconstructed_b = result.Q[i] @ result.T[i] @ result.Z[i].mH
            b_complex = b[i].to(torch.complex128)
            torch.testing.assert_close(
                reconstructed_b, b_complex, rtol=1e-10, atol=1e-10
            )

    def test_meta_tensor(self):
        """Test shape inference with meta tensors."""
        n = 4
        a = torch.randn(n, n, dtype=torch.float64, device="meta")
        b = torch.randn(n, n, dtype=torch.float64, device="meta")

        result = generalized_schur(a, b, output="complex")

        assert result.S.shape == (n, n)
        assert result.T.shape == (n, n)
        assert result.alpha.shape == (n,)
        assert result.beta.shape == (n,)
        assert result.Q.shape == (n, n)
        assert result.Z.shape == (n, n)
        assert result.S.device.type == "meta"

    def test_compile(self):
        """Test torch.compile compatibility."""
        torch.manual_seed(222)
        n = 3
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)

        def compute_schur(x, y):
            result = generalized_schur(x, y, output="complex")
            return result.S, result.T, result.Q, result.Z

        compiled_fn = torch.compile(compute_schur, fullgraph=False)

        S1, T1, Q1, Z1 = compute_schur(a, b)
        S2, T2, Q2, Z2 = compiled_fn(a, b)

        torch.testing.assert_close(S1, S2, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(T1, T2, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(Q1, Q2, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(Z1, Z2, rtol=1e-10, atol=1e-10)

    def test_real_output(self):
        """Test real output mode."""
        torch.manual_seed(333)
        n = 4
        a = torch.randn(n, n, dtype=torch.float64)
        b = torch.randn(n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="real")

        # Output matrices should be real
        assert not result.S.is_complex()
        assert not result.T.is_complex()
        assert not result.Q.is_complex()
        assert not result.Z.is_complex()

        # alpha and beta are always complex
        assert result.alpha.is_complex()
        assert result.beta.is_complex()

    def test_complex_input(self):
        """Test with complex input matrices."""
        torch.manual_seed(444)
        n = 3
        a = torch.randn(n, n, dtype=torch.complex128)
        b = torch.randn(n, n, dtype=torch.complex128)

        result = generalized_schur(a, b)

        # Verify reconstruction
        reconstructed_a = result.Q @ result.S @ result.Z.mH
        torch.testing.assert_close(reconstructed_a, a, rtol=1e-10, atol=1e-10)

        reconstructed_b = result.Q @ result.T @ result.Z.mH
        torch.testing.assert_close(reconstructed_b, b, rtol=1e-10, atol=1e-10)

    def test_broadcast_batches(self):
        """Test broadcasting of batch dimensions."""
        torch.manual_seed(555)
        n = 3

        # a has batch dim (2,), b has batch dim (1,)
        a = torch.randn(2, n, n, dtype=torch.float64)
        b = torch.randn(1, n, n, dtype=torch.float64)

        result = generalized_schur(a, b, output="complex")

        # Result should have broadcast shape (2,)
        assert result.S.shape == (2, n, n)
        assert result.T.shape == (2, n, n)

    def test_invalid_input_1d(self):
        """Test error on 1D input."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="must be at least 2D"):
            generalized_schur(a, b)

    def test_invalid_input_non_square_a(self):
        """Test error on non-square input a."""
        a = torch.randn(3, 4)
        b = torch.randn(3, 3)
        with pytest.raises(ValueError, match="must be square"):
            generalized_schur(a, b)

    def test_invalid_input_non_square_b(self):
        """Test error on non-square input b."""
        a = torch.randn(3, 3)
        b = torch.randn(3, 4)
        with pytest.raises(ValueError, match="must be square"):
            generalized_schur(a, b)

    def test_invalid_input_size_mismatch(self):
        """Test error when a and b have different sizes."""
        a = torch.randn(3, 3)
        b = torch.randn(4, 4)
        with pytest.raises(ValueError, match="same size"):
            generalized_schur(a, b)

    def test_invalid_output_mode(self):
        """Test error on invalid output mode."""
        a = torch.randn(3, 3)
        b = torch.randn(3, 3)
        with pytest.raises(ValueError, match="must be 'real' or 'complex'"):
            generalized_schur(a, b, output="invalid")

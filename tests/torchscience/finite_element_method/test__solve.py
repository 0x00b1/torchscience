"""Tests for FEM linear system solvers."""

import pytest
import torch

from torchscience.finite_element_method import solve_direct


class TestSolveDirect:
    """Tests for solve_direct function."""

    def test_solve_basic_spd_system(self):
        """Test solving a simple SPD system with known solution."""
        # Create a simple 3x3 SPD system: K @ u = f
        # K = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        # Known solution: u = [1, 2, 3]
        # f = K @ u = [6, 10, 8]
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)
        expected_u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        u = solve_direct(K, f)

        assert u.shape == f.shape
        assert torch.allclose(u, expected_u, atol=1e-10)

    def test_solve_sparse_csr_input(self):
        """Test that solve_direct accepts sparse CSR matrices."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        ).to_sparse_csr()
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)
        expected_u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        u = solve_direct(K, f)

        assert u.shape == f.shape
        assert torch.allclose(u, expected_u, atol=1e-10)

    def test_solve_dense_input(self):
        """Test that solve_direct accepts dense matrices."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)
        expected_u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        u = solve_direct(K, f)

        assert u.shape == f.shape
        assert torch.allclose(u, expected_u, atol=1e-10)

    def test_solve_multiple_rhs(self):
        """Test solving for multiple right-hand sides."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        # Two right-hand sides
        f = torch.tensor(
            [[6.0, 2.0], [10.0, 3.0], [8.0, 4.0]], dtype=torch.float64
        )

        u = solve_direct(K, f)

        assert u.shape == f.shape
        # Verify K @ u == f for each column
        residual = K @ u - f
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)

    def test_solve_sparse_multiple_rhs(self):
        """Test solving sparse system with multiple right-hand sides."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        ).to_sparse_csr()
        f = torch.tensor(
            [[6.0, 2.0], [10.0, 3.0], [8.0, 4.0]], dtype=torch.float64
        )

        u = solve_direct(K, f)

        assert u.shape == f.shape
        # Verify K @ u == f for each column
        residual = K.to_dense() @ u - f
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)

    def test_solve_verifies_solution(self):
        """Test that K @ u == f for the solution."""
        # Random SPD matrix
        n = 10
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + torch.eye(n, dtype=torch.float64)  # Make SPD
        f = torch.randn(n, dtype=torch.float64)

        u = solve_direct(K, f)

        residual = K @ u - f
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)

    def test_solve_sparse_verifies_solution(self):
        """Test that K @ u == f for sparse input."""
        n = 10
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + torch.eye(n, dtype=torch.float64)
        K_sparse = K.to_sparse_csr()
        f = torch.randn(n, dtype=torch.float64)

        u = solve_direct(K_sparse, f)

        residual = K @ u - f
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)

    def test_solve_differentiability_wrt_vector(self):
        """Test that gradients flow through solve with respect to vector."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        f = torch.tensor(
            [6.0, 10.0, 8.0], dtype=torch.float64, requires_grad=True
        )

        u = solve_direct(K, f)
        loss = u.sum()
        loss.backward()

        assert f.grad is not None
        assert f.grad.shape == f.shape
        assert not torch.isnan(f.grad).any()

    def test_solve_differentiability_wrt_matrix(self):
        """Test that gradients flow through solve with respect to matrix."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)

        u = solve_direct(K, f)
        loss = u.sum()
        loss.backward()

        assert K.grad is not None
        assert K.grad.shape == K.shape
        assert not torch.isnan(K.grad).any()

    def test_solve_differentiability_both_inputs(self):
        """Test that gradients flow through solve with respect to both inputs."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        f = torch.tensor(
            [6.0, 10.0, 8.0], dtype=torch.float64, requires_grad=True
        )

        u = solve_direct(K, f)
        loss = u.sum()
        loss.backward()

        assert K.grad is not None
        assert f.grad is not None
        assert not torch.isnan(K.grad).any()
        assert not torch.isnan(f.grad).any()

    def test_solve_gradcheck(self):
        """Test gradient correctness using torch.autograd.gradcheck."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        f = torch.tensor(
            [6.0, 10.0, 8.0], dtype=torch.float64, requires_grad=True
        )

        # Use a wrapper function for gradcheck
        def solve_wrapper(K, f):
            return solve_direct(K, f)

        assert torch.autograd.gradcheck(
            solve_wrapper, (K, f), raise_exception=True
        )

    def test_solve_dtype_preserved(self):
        """Test that output dtype matches input dtype."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float32,
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float32)

        u = solve_direct(K, f)

        assert u.dtype == torch.float32

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_solve_device_preserved(self):
        """Test that output device matches input device."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
            device="cuda",
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64, device="cuda")

        u = solve_direct(K, f)

        assert u.device.type == "cuda"

    def test_solve_identity_matrix(self):
        """Test solving with identity matrix (u = f)."""
        n = 5
        K = torch.eye(n, dtype=torch.float64)
        f = torch.randn(n, dtype=torch.float64)

        u = solve_direct(K, f)

        assert torch.allclose(u, f, atol=1e-10)

    def test_solve_diagonal_matrix(self):
        """Test solving with diagonal matrix."""
        diag = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        K = torch.diag(diag)
        f = torch.tensor([2.0, 6.0, 12.0, 20.0], dtype=torch.float64)
        expected_u = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        u = solve_direct(K, f)

        assert torch.allclose(u, expected_u, atol=1e-10)

    def test_solve_large_system(self):
        """Test solving a larger system."""
        n = 100
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + n * torch.eye(
            n, dtype=torch.float64
        )  # Make well-conditioned SPD
        f = torch.randn(n, dtype=torch.float64)

        u = solve_direct(K, f)

        residual = K @ u - f
        relative_error = residual.norm() / f.norm()
        assert relative_error < 1e-10

    def test_solve_sparse_coo_input(self):
        """Test that solve_direct accepts sparse COO matrices."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        ).to_sparse_coo()
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)
        expected_u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        u = solve_direct(K, f)

        assert u.shape == f.shape
        assert torch.allclose(u, expected_u, atol=1e-10)

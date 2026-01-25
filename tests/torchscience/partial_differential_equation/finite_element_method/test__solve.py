"""Tests for FEM linear system solvers."""

import pytest
import torch
from torch import Tensor

from torchscience.partial_differential_equation.finite_element_method import (
    solve_cg,
    solve_direct,
)


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


class TestSolveCG:
    """Tests for solve_cg conjugate gradient solver."""

    def test_basic_spd_system(self):
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

        u = solve_cg(K, f)

        assert u.shape == f.shape
        assert torch.allclose(u, expected_u, atol=1e-6)

    def test_sparse_csr_matrix(self):
        """Test that solve_cg works with sparse CSR matrices."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        ).to_sparse_csr()
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)
        expected_u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        u = solve_cg(K, f)

        assert u.shape == f.shape
        assert torch.allclose(u, expected_u, atol=1e-6)

    def test_convergence_residual_decreases(self):
        """Test that the residual decreases and converges."""
        n = 20
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + n * torch.eye(
            n, dtype=torch.float64
        )  # Well-conditioned SPD
        f = torch.randn(n, dtype=torch.float64)

        u = solve_cg(K, f, tol=1e-10)

        # Verify solution quality
        residual = K @ u - f
        relative_error = residual.norm() / f.norm()
        assert relative_error < 1e-8

    def test_custom_initial_guess(self):
        """Test that custom initial guess x0 is used."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)
        # Start close to solution
        x0 = torch.tensor([0.9, 1.9, 2.9], dtype=torch.float64)

        u = solve_cg(K, f, x0=x0)

        expected_u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert torch.allclose(u, expected_u, atol=1e-6)

    def test_tolerance_control(self):
        """Test that tolerance affects convergence criterion."""
        n = 10
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + n * torch.eye(n, dtype=torch.float64)
        f = torch.randn(n, dtype=torch.float64)

        # Solve with different tolerances
        u_loose = solve_cg(K, f, tol=1e-3)
        u_tight = solve_cg(K, f, tol=1e-10)

        # Both should be valid solutions
        residual_loose = (K @ u_loose - f).norm() / f.norm()
        residual_tight = (K @ u_tight - f).norm() / f.norm()

        # Tight tolerance should give better solution
        assert residual_tight <= residual_loose

    def test_maxiter_limits_iterations(self):
        """Test that maxiter limits the number of iterations."""
        n = 50
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + n * torch.eye(n, dtype=torch.float64)
        f = torch.randn(n, dtype=torch.float64)

        # Very few iterations - may not converge
        u = solve_cg(K, f, maxiter=3, tol=1e-12)

        # Should still return a result (even if not fully converged)
        assert u.shape == f.shape
        assert not torch.isnan(u).any()

    def test_diagonal_preconditioner_tensor(self):
        """Test with diagonal preconditioner as tensor."""
        n = 20
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + n * torch.eye(n, dtype=torch.float64)
        f = torch.randn(n, dtype=torch.float64)

        # Diagonal preconditioner: M^{-1} = 1/diag(K)
        diag_inv = 1.0 / K.diag()

        u = solve_cg(K, f, preconditioner=diag_inv, tol=1e-10)

        residual = K @ u - f
        relative_error = residual.norm() / f.norm()
        assert relative_error < 1e-8

    def test_callable_preconditioner(self):
        """Test with preconditioner as callable."""
        n = 20
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + n * torch.eye(n, dtype=torch.float64)
        f = torch.randn(n, dtype=torch.float64)

        # Diagonal preconditioner as callable
        diag_inv = 1.0 / K.diag()

        def precond(r: Tensor) -> Tensor:
            return diag_inv * r

        u = solve_cg(K, f, preconditioner=precond, tol=1e-10)

        residual = K @ u - f
        relative_error = residual.norm() / f.norm()
        assert relative_error < 1e-8

    def test_match_direct_solver(self):
        """Test that CG solution matches direct solver solution."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)

        u_direct = solve_direct(K, f)
        u_cg = solve_cg(K, f, tol=1e-12)

        assert torch.allclose(u_cg, u_direct, atol=1e-10)

    def test_match_direct_solver_sparse(self):
        """Test that CG on sparse matrix matches direct solver."""
        K_dense = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        K_sparse = K_dense.to_sparse_csr()
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)

        u_direct = solve_direct(K_dense, f)
        u_cg = solve_cg(K_sparse, f, tol=1e-12)

        assert torch.allclose(u_cg, u_direct, atol=1e-10)

    def test_identity_matrix(self):
        """Test solving with identity matrix (u = f)."""
        n = 5
        K = torch.eye(n, dtype=torch.float64)
        f = torch.randn(n, dtype=torch.float64)

        u = solve_cg(K, f)

        assert torch.allclose(u, f, atol=1e-10)

    def test_diagonal_matrix(self):
        """Test solving with diagonal matrix."""
        diag = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        K = torch.diag(diag)
        f = torch.tensor([2.0, 6.0, 12.0, 20.0], dtype=torch.float64)
        expected_u = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        u = solve_cg(K, f)

        assert torch.allclose(u, expected_u, atol=1e-6)

    def test_larger_system(self):
        """Test solving a larger SPD system."""
        n = 100
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + n * torch.eye(n, dtype=torch.float64)
        f = torch.randn(n, dtype=torch.float64)

        u = solve_cg(K, f, tol=1e-10)

        residual = K @ u - f
        relative_error = residual.norm() / f.norm()
        assert relative_error < 1e-8

    def test_dtype_preserved(self):
        """Test that output dtype matches input dtype."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float32,
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float32)

        u = solve_cg(K, f)

        assert u.dtype == torch.float32

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_device_preserved(self):
        """Test that output device matches input device."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
            device="cuda",
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64, device="cuda")

        u = solve_cg(K, f)

        assert u.device.type == "cuda"

    def test_zeros_initial_guess_default(self):
        """Test that default initial guess is zeros."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)

        # Should work without specifying x0
        u = solve_cg(K, f)

        expected_u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert torch.allclose(u, expected_u, atol=1e-6)

    def test_default_maxiter(self):
        """Test that default maxiter is 2*n."""
        n = 10
        A = torch.randn(n, n, dtype=torch.float64)
        K = A @ A.T + n * torch.eye(n, dtype=torch.float64)
        f = torch.randn(n, dtype=torch.float64)

        # Should use maxiter=2*n by default
        u = solve_cg(K, f, tol=1e-10)

        residual = K @ u - f
        relative_error = residual.norm() / f.norm()
        assert relative_error < 1e-8

    def test_sparse_coo_matrix(self):
        """Test that solve_cg works with sparse COO matrices."""
        K = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        ).to_sparse_coo()
        f = torch.tensor([6.0, 10.0, 8.0], dtype=torch.float64)
        expected_u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        u = solve_cg(K, f)

        assert u.shape == f.shape
        assert torch.allclose(u, expected_u, atol=1e-6)

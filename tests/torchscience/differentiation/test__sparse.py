"""Tests for sparse tensor support in differentiation operators."""

import torch

from torchscience.differentiation import (
    biharmonic,
    curl,
    derivative,
    divergence,
    gradient,
    hessian,
    jacobian,
    laplacian,
)


class TestSparseScalarOperators:
    """Tests for differentiation with sparse tensor input on scalar field operators."""

    def test_derivative_sparse_coo_densifies(self):
        """Derivative densifies sparse COO input."""
        dense = torch.randn(16, 16)
        sparse = dense.to_sparse()

        deriv = derivative(sparse, dim=-1, dx=0.1)

        assert not deriv.is_sparse
        # Compare with dense result
        deriv_dense = derivative(dense, dim=-1, dx=0.1)
        torch.testing.assert_close(deriv, deriv_dense)

    def test_gradient_sparse_coo_densifies(self):
        """Gradient densifies sparse COO input."""
        dense = torch.randn(16, 16)
        sparse = dense.to_sparse()

        grad = gradient(sparse, dx=0.1)

        assert not grad.is_sparse
        assert grad.shape == (2, 16, 16)

        # Compare with dense result
        grad_dense = gradient(dense, dx=0.1)
        torch.testing.assert_close(grad, grad_dense)

    def test_laplacian_sparse_coo_densifies(self):
        """Laplacian densifies sparse input."""
        dense = torch.randn(16, 16)
        sparse = dense.to_sparse()

        lap = laplacian(sparse, dx=0.1)

        assert not lap.is_sparse
        # Compare with dense result
        lap_dense = laplacian(dense, dx=0.1)
        torch.testing.assert_close(lap, lap_dense)

    def test_hessian_sparse_coo_densifies(self):
        """Hessian densifies sparse input."""
        dense = torch.randn(16, 16)
        sparse = dense.to_sparse()

        hess = hessian(sparse, dx=0.1)

        assert not hess.is_sparse
        assert hess.shape == (2, 2, 16, 16)
        # Compare with dense result
        hess_dense = hessian(dense, dx=0.1)
        torch.testing.assert_close(hess, hess_dense)

    def test_biharmonic_sparse_coo_densifies(self):
        """Biharmonic densifies sparse input."""
        dense = torch.randn(16, 16)
        sparse = dense.to_sparse()

        biharm = biharmonic(sparse, dx=0.1)

        assert not biharm.is_sparse
        # Compare with dense result
        biharm_dense = biharmonic(dense, dx=0.1)
        torch.testing.assert_close(biharm, biharm_dense)


class TestSparseVectorOperators:
    """Tests for differentiation with sparse tensor input on vector field operators."""

    def test_divergence_sparse_coo_densifies(self):
        """Divergence densifies sparse input."""
        dense = torch.randn(2, 16, 16)
        sparse = dense.to_sparse()

        div = divergence(sparse, dx=0.1)

        assert not div.is_sparse
        assert div.shape == (16, 16)
        # Compare with dense result
        div_dense = divergence(dense, dx=0.1)
        torch.testing.assert_close(div, div_dense)

    def test_curl_sparse_coo_densifies(self):
        """Curl densifies sparse input."""
        dense = torch.randn(3, 10, 10, 10)
        sparse = dense.to_sparse()

        c = curl(sparse, dx=0.1)

        assert not c.is_sparse
        assert c.shape == (3, 10, 10, 10)
        # Compare with dense result
        c_dense = curl(dense, dx=0.1)
        torch.testing.assert_close(c, c_dense)

    def test_jacobian_sparse_coo_densifies(self):
        """Jacobian densifies sparse input."""
        dense = torch.randn(2, 16, 16)
        sparse = dense.to_sparse()

        jac = jacobian(sparse, dx=0.1)

        assert not jac.is_sparse
        assert jac.shape == (2, 2, 16, 16)
        # Compare with dense result
        jac_dense = jacobian(dense, dx=0.1)
        torch.testing.assert_close(jac, jac_dense)

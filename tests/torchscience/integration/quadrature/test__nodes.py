import pytest
import torch
from numpy.polynomial.legendre import leggauss


class TestGaussLegendreNodesWeights:
    @pytest.mark.parametrize("n", [2, 5, 10, 32, 64])
    def test_matches_numpy(self, n):
        """Compare with numpy's Gauss-Legendre implementation"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        nodes, weights = gauss_legendre_nodes_weights(n, dtype=torch.float64)
        np_nodes, np_weights = leggauss(n)

        assert torch.allclose(nodes, torch.tensor(np_nodes), rtol=1e-12)
        assert torch.allclose(weights, torch.tensor(np_weights), rtol=1e-12)

    def test_nodes_in_interval(self):
        """Nodes should be in [-1, 1]"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        nodes, _ = gauss_legendre_nodes_weights(100)

        assert (nodes >= -1).all()
        assert (nodes <= 1).all()

    def test_weights_sum_to_two(self):
        """Weights should sum to 2 (length of [-1, 1])"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        _, weights = gauss_legendre_nodes_weights(50, dtype=torch.float64)

        assert torch.allclose(
            weights.sum(), torch.tensor(2.0, dtype=torch.float64), rtol=1e-10
        )

    def test_exact_for_polynomial(self):
        """Should exactly integrate polynomials up to degree 2n-1"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        n = 5
        nodes, weights = gauss_legendre_nodes_weights(n, dtype=torch.float64)

        # Integrate x^8 from -1 to 1 (degree 8 < 2*5-1=9, so should be exact)
        # integral of x^8 from -1 to 1 = 2/9
        integrand = nodes**8
        result = (integrand * weights).sum()
        expected = torch.tensor(2 / 9, dtype=torch.float64)

        assert torch.allclose(result, expected, rtol=1e-10)

    def test_n_equals_1(self):
        """Single-point quadrature: midpoint rule"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        nodes, weights = gauss_legendre_nodes_weights(1, dtype=torch.float64)

        assert nodes.shape == (1,)
        assert weights.shape == (1,)
        assert torch.allclose(nodes, torch.tensor([0.0], dtype=torch.float64))
        assert torch.allclose(
            weights, torch.tensor([2.0], dtype=torch.float64)
        )

    def test_invalid_n_raises(self):
        """n < 1 should raise ValueError"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        with pytest.raises(ValueError, match="at least 1"):
            gauss_legendre_nodes_weights(0)

    def test_device_placement(self):
        """Test that device parameter works"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        nodes, weights = gauss_legendre_nodes_weights(
            10, device=torch.device("cpu")
        )

        assert nodes.device == torch.device("cpu")
        assert weights.device == torch.device("cpu")


class TestGaussKronrodNodesWeights:
    @pytest.mark.parametrize("order", [15, 21])
    def test_valid_orders(self, order):
        """Test that implemented orders work"""
        from torchscience.integration.quadrature._nodes import (
            gauss_kronrod_nodes_weights,
        )

        nodes, k_weights, g_weights, g_indices = gauss_kronrod_nodes_weights(
            order
        )

        assert nodes.shape == (order,)
        assert k_weights.shape == (order,)
        # Gauss points are at specific indices within Kronrod nodes
        n_gauss = order // 2
        assert g_weights.shape == (n_gauss,)
        assert g_indices.shape == (n_gauss,)

    def test_invalid_order(self):
        """Invalid orders should raise"""
        from torchscience.integration.quadrature._nodes import (
            gauss_kronrod_nodes_weights,
        )

        with pytest.raises(ValueError, match="order must be"):
            gauss_kronrod_nodes_weights(20)

    def test_kronrod_weights_sum(self):
        """Kronrod weights should sum to 2"""
        from torchscience.integration.quadrature._nodes import (
            gauss_kronrod_nodes_weights,
        )

        nodes, k_weights, _, _ = gauss_kronrod_nodes_weights(
            15, dtype=torch.float64
        )

        assert torch.allclose(
            k_weights.sum(), torch.tensor(2.0, dtype=torch.float64), rtol=1e-10
        )

    def test_gauss_weights_sum(self):
        """Gauss weights should sum to 2"""
        from torchscience.integration.quadrature._nodes import (
            gauss_kronrod_nodes_weights,
        )

        _, _, g_weights, _ = gauss_kronrod_nodes_weights(
            15, dtype=torch.float64
        )

        assert torch.allclose(
            g_weights.sum(), torch.tensor(2.0, dtype=torch.float64), rtol=1e-10
        )

    def test_g7k15_integration(self):
        """G7-K15 should integrate polynomials exactly"""
        from torchscience.integration.quadrature._nodes import (
            gauss_kronrod_nodes_weights,
        )

        nodes, k_weights, _, _ = gauss_kronrod_nodes_weights(
            15, dtype=torch.float64
        )

        # Integrate x^6 from -1 to 1 = 2/7
        integrand = nodes**6
        result = (integrand * k_weights).sum()
        expected = torch.tensor(2 / 7, dtype=torch.float64)

        assert torch.allclose(result, expected, rtol=1e-10)

    def test_gauss_indices_correct(self):
        """Gauss nodes should be at the returned indices"""
        from torchscience.integration.quadrature._nodes import (
            gauss_kronrod_nodes_weights,
        )

        nodes, _, g_weights, g_indices = gauss_kronrod_nodes_weights(
            15, dtype=torch.float64
        )

        # G7 nodes from numpy
        np_g_nodes, np_g_weights = leggauss(7)

        # Gauss nodes should match at the specified indices
        gauss_nodes = nodes[g_indices]
        assert torch.allclose(
            gauss_nodes,
            torch.tensor(np_g_nodes, dtype=torch.float64),
            rtol=1e-10,
        )

    def test_nodes_symmetric(self):
        """Nodes should be symmetric about 0"""
        from torchscience.integration.quadrature._nodes import (
            gauss_kronrod_nodes_weights,
        )

        nodes, _, _, _ = gauss_kronrod_nodes_weights(15, dtype=torch.float64)

        # Sum should be approximately 0
        assert torch.allclose(
            nodes.sum(), torch.tensor(0.0, dtype=torch.float64), atol=1e-12
        )

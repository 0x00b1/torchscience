import torch
import torch.testing

import torchscience.special_functions


class TestInverseJacobiEllipticDn:
    """Tests for inverse Jacobi elliptic function arcdn(x, m)."""

    def test_forward_one_argument(self):
        """arcdn(1, m) = 0 for all m."""
        x = torch.tensor([1.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_jacobi_elliptic_dn(
            x, m
        )
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_zero_parameter(self):
        """arcdn(x, 0) = 0 since dn(u, 0) = 1 for all u."""
        x = torch.tensor([1.0], dtype=torch.float64)
        m = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.inverse_jacobi_elliptic_dn(
            x, m
        )
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_forward_inverse_property(self):
        """Verify dn(arcdn(x, m), m) = x."""
        # dn(u, m) ranges from sqrt(1-m) to 1, so x should be in that range
        m_val = 0.5
        x = torch.tensor([0.75, 0.8, 0.9, 0.95, 0.99], dtype=torch.float64)
        m = torch.tensor([m_val], dtype=torch.float64)

        u = torchscience.special_functions.inverse_jacobi_elliptic_dn(x, m)
        x_recovered = torchscience.special_functions.jacobi_elliptic_dn(u, m)

        torch.testing.assert_close(x_recovered, x, rtol=1e-5, atol=1e-5)

    def test_gradient(self):
        """First-order gradient via gradcheck."""
        x = torch.tensor([0.9], dtype=torch.float64, requires_grad=True)
        m = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.inverse_jacobi_elliptic_dn,
            (x, m),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradient_gradient(self):
        """Second-order gradient via gradgradcheck."""
        x = torch.tensor([0.9], dtype=torch.float64, requires_grad=True)
        m = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.inverse_jacobi_elliptic_dn,
            (x, m),
            eps=1e-4,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_broadcasting(self):
        """Test broadcasting of x and m."""
        x = torch.tensor([0.75, 0.85, 0.95], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_jacobi_elliptic_dn(
            x, m
        )
        assert result.shape == torch.Size([3])

    def test_meta_tensor(self):
        """Test meta tensor support for shape inference."""
        x = torch.empty(5, 3, dtype=torch.float64, device="meta")
        m = torch.empty(1, dtype=torch.float64, device="meta")
        result = torchscience.special_functions.inverse_jacobi_elliptic_dn(
            x, m
        )
        assert result.shape == torch.Size([5, 3])

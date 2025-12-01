import torch
import torch.testing

import torchscience.special_functions


class TestInverseJacobiEllipticSd:
    """Tests for inverse Jacobi elliptic function arcsd(x, m)."""

    def test_forward_zero_argument(self):
        """arcsd(0, m) = 0 for all m."""
        x = torch.tensor([0.0], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_jacobi_elliptic_sd(
            x, m
        )
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_inverse_property(self):
        """Verify sd(arcsd(x, m), m) = x."""
        x = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)

        u = torchscience.special_functions.inverse_jacobi_elliptic_sd(x, m)
        x_recovered = torchscience.special_functions.jacobi_elliptic_sd(u, m)

        torch.testing.assert_close(x_recovered, x, rtol=1e-5, atol=1e-5)

    def test_gradient(self):
        """First-order gradient via gradcheck."""
        x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        m = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.inverse_jacobi_elliptic_sd,
            (x, m),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradient_gradient(self):
        """Second-order gradient via gradgradcheck."""
        x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        m = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.inverse_jacobi_elliptic_sd,
            (x, m),
            eps=1e-4,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_broadcasting(self):
        """Test broadcasting of x and m."""
        x = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        m = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.inverse_jacobi_elliptic_sd(
            x, m
        )
        assert result.shape == torch.Size([3])

    def test_meta_tensor(self):
        """Test meta tensor support for shape inference."""
        x = torch.empty(5, 3, dtype=torch.float64, device="meta")
        m = torch.empty(1, dtype=torch.float64, device="meta")
        result = torchscience.special_functions.inverse_jacobi_elliptic_sd(
            x, m
        )
        assert result.shape == torch.Size([5, 3])

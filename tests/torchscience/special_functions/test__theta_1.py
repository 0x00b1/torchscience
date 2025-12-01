import mpmath
import torch
import torch.testing

import torchscience.special_functions


def mpmath_jtheta(n, z, q):
    """Reference implementation using mpmath."""
    with mpmath.workdps(30):
        return float(mpmath.jtheta(n, z, q))


class TestTheta1:
    """Tests for Jacobi theta function theta_1(z, q)."""

    def test_forward_zero_argument(self):
        """theta_1(0, q) = 0 (odd function)."""
        z = torch.tensor([0.0], dtype=torch.float64)
        q = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.theta_1(z, q)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_zero_nome(self):
        """theta_1(z, 0) = 0."""
        z = torch.tensor([1.0], dtype=torch.float64)
        q = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.theta_1(z, q)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_odd_symmetry(self):
        """theta_1(-z, q) = -theta_1(z, q)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        q = torch.tensor([0.3], dtype=torch.float64)
        result_pos = torchscience.special_functions.theta_1(z, q)
        result_neg = torchscience.special_functions.theta_1(-z, q)
        torch.testing.assert_close(
            result_neg, -result_pos, rtol=1e-10, atol=1e-10
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath reference."""
        test_cases = [
            (0.5, 0.1),
            (1.0, 0.2),
            (0.3, 0.5),
            (1.5, 0.3),
        ]
        for z_val, q_val in test_cases:
            z = torch.tensor([z_val], dtype=torch.float64)
            q = torch.tensor([q_val], dtype=torch.float64)
            result = torchscience.special_functions.theta_1(z, q)
            expected = mpmath_jtheta(1, z_val, q_val)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_gradient(self):
        """First-order gradient via gradcheck."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.theta_1,
            (z, q),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradient_gradient(self):
        """Second-order gradient via gradgradcheck."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.theta_1,
            (z, q),
            eps=1e-4,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_broadcasting(self):
        """Test broadcasting of z and q."""
        z = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        q = torch.tensor([0.3], dtype=torch.float64)
        result = torchscience.special_functions.theta_1(z, q)
        assert result.shape == torch.Size([3])

    def test_meta_tensor(self):
        """Test meta tensor support for shape inference."""
        z = torch.empty(5, 3, dtype=torch.float64, device="meta")
        q = torch.empty(1, dtype=torch.float64, device="meta")
        result = torchscience.special_functions.theta_1(z, q)
        assert result.shape == torch.Size([5, 3])

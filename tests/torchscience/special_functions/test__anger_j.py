import torch
from torch.testing import assert_close

import torchscience.special_functions


class TestAngerJ:
    """Tests for the Anger function anger_j(n, z)."""

    def test_forward_integer_order_equals_bessel_j(self):
        """For integer n, Anger function J_n(z) equals Bessel J_n(z)."""
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        for n_val in [0.0, 1.0, 2.0, 3.0]:
            n = torch.full_like(z, n_val)
            anger = torchscience.special_functions.anger_j(n, z)
            bessel = torchscience.special_functions.bessel_j(n, z)
            assert_close(anger, bessel, atol=1e-6, rtol=1e-5)

    def test_forward_small_arguments(self):
        """Test Anger function for small arguments."""
        n = torch.tensor([0.5, 1.5, 0.0, 1.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.2, 0.5, 0.5], dtype=torch.float64)
        result = torchscience.special_functions.anger_j(n, z)
        assert result.shape == z.shape
        assert torch.all(torch.isfinite(result))

    def test_forward_zero_argument(self):
        """Test Anger function at z=0."""
        n = torch.tensor([0.0, 1.0, 2.0, 0.5], dtype=torch.float64)
        z = torch.zeros_like(n)
        result = torchscience.special_functions.anger_j(n, z)
        # J_0(0) = 1, J_n(0) = 0 for n > 0
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert_close(result[:3], expected[:3], atol=1e-6, rtol=1e-5)

    def test_forward_moderate_arguments(self):
        """Test Anger function for moderate arguments."""
        n = torch.tensor([0.0, 1.0, 2.0, 0.5], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 3.0, 2.5], dtype=torch.float64)
        result = torchscience.special_functions.anger_j(n, z)
        assert result.shape == z.shape
        assert torch.all(torch.isfinite(result))

    def test_forward_broadcasting(self):
        """Test that broadcasting works correctly."""
        n = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)  # (3, 1)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # (3,)
        result = torchscience.special_functions.anger_j(n, z)
        assert result.shape == (3, 3)

    def test_forward_batch(self):
        """Test batch computation."""
        n = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        z = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float64)
        result = torchscience.special_functions.anger_j(n, z)
        assert result.shape == (5,)
        assert torch.all(torch.isfinite(result))

    def test_forward_negative_z(self):
        """Test Anger function with negative z."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64)
        result = torchscience.special_functions.anger_j(n, z)
        assert torch.all(torch.isfinite(result))

    def test_forward_dtype_float32(self):
        """Test with float32 dtype."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = torchscience.special_functions.anger_j(n, z)
        assert result.dtype == torch.float32

    def test_forward_dtype_float64(self):
        """Test with float64 dtype."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = torchscience.special_functions.anger_j(n, z)
        assert result.dtype == torch.float64

    def test_backward_z(self):
        """Test backward pass with respect to z."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        result = torchscience.special_functions.anger_j(n, z)
        grad = torch.ones_like(result)
        result.backward(grad)
        assert z.grad is not None
        assert z.grad.shape == z.shape
        assert torch.all(torch.isfinite(z.grad))

    def test_backward_n(self):
        """Test backward pass with respect to n."""
        n = torch.tensor(
            [0.5, 1.5, 2.5], dtype=torch.float64, requires_grad=True
        )
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = torchscience.special_functions.anger_j(n, z)
        grad = torch.ones_like(result)
        result.backward(grad)
        assert n.grad is not None
        assert n.grad.shape == n.shape
        assert torch.all(torch.isfinite(n.grad))

    def test_backward_both(self):
        """Test backward pass with respect to both n and z."""
        n = torch.tensor(
            [0.5, 1.5, 2.5], dtype=torch.float64, requires_grad=True
        )
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        result = torchscience.special_functions.anger_j(n, z)
        grad = torch.ones_like(result)
        result.backward(grad)
        assert n.grad is not None
        assert z.grad is not None
        assert torch.all(torch.isfinite(n.grad))
        assert torch.all(torch.isfinite(z.grad))

    def test_gradcheck_z(self):
        """Test gradient correctness for z using torch.autograd.gradcheck."""
        n = torch.tensor([0.5, 1.5], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(z_):
            return torchscience.special_functions.anger_j(n, z_)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_n(self):
        """Test gradient correctness for n using torch.autograd.gradcheck."""
        n = torch.tensor([0.5, 1.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0, 2.0], dtype=torch.float64)

        def func(n_):
            return torchscience.special_functions.anger_j(n_, z)

        assert torch.autograd.gradcheck(
            func, (n,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradgradcheck_z(self):
        """Test second-order gradient correctness for z."""
        n = torch.tensor([0.5], dtype=torch.float64)
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        def func(z_):
            return torchscience.special_functions.anger_j(n, z_)

        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-6, atol=1e-3, rtol=1e-2
        )

    def test_meta_tensor(self):
        """Test that meta tensors work (shape inference without computation)."""
        n = torch.empty(3, device="meta", dtype=torch.float64)
        z = torch.empty(3, device="meta", dtype=torch.float64)
        result = torchscience.special_functions.anger_j(n, z)
        assert result.shape == (3,)
        assert result.device.type == "meta"

    def test_derivative_formula_z(self):
        """Test that dJ_nu/dz = (J_{nu-1} - J_{nu+1})/2."""
        n = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        result = torchscience.special_functions.anger_j(n, z)
        result.backward()

        # Expected: (J_{n-1}(z) - J_{n+1}(z)) / 2
        j_n_minus_1 = torchscience.special_functions.anger_j(
            n - 1.0, z.detach()
        )
        j_n_plus_1 = torchscience.special_functions.anger_j(
            n + 1.0, z.detach()
        )
        expected_grad = 0.5 * (j_n_minus_1 - j_n_plus_1)
        assert_close(z.grad, expected_grad, atol=1e-4, rtol=1e-3)

    def test_gradcheck_both(self):
        """Test gradient correctness for both n and z using torch.autograd.gradcheck."""
        n = torch.tensor([0.5, 1.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(n_, z_):
            return torchscience.special_functions.anger_j(n_, z_)

        assert torch.autograd.gradcheck(
            func, (n, z), eps=1e-6, atol=1e-4, rtol=1e-3
        )

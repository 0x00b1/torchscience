import math

import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestGeneralHammingWindow:
    """Tests for general_hamming_window and periodic_general_hamming_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        alpha = torch.tensor(0.54, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.general_hamming_window(n, alpha, dtype=torch.float64)
            expected = self._reference_general_hamming(
                n, alpha.item(), periodic=False
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        alpha = torch.tensor(0.54, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_general_hamming_window(
                n, alpha, dtype=torch.float64
            )
            expected = self._reference_general_hamming(
                n, alpha.item(), periodic=True
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_hann_special_case(self):
        """Test that alpha=0.5 produces Hann window."""
        alpha = torch.tensor(0.5, dtype=torch.float64)
        for n in [4, 16, 64]:
            result = wf.general_hamming_window(n, alpha, dtype=torch.float64)
            expected = wf.hann_window(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_hamming_special_case(self):
        """Test that alpha=0.54 produces Hamming window."""
        alpha = torch.tensor(0.54, dtype=torch.float64)
        for n in [4, 16, 64]:
            result = wf.general_hamming_window(n, alpha, dtype=torch.float64)
            expected = wf.hamming_window(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_pytorch_comparison_symmetric(self):
        """Compare with torch.signal.windows.general_hamming (symmetric)."""
        for n in [4, 16, 64]:
            for alpha_val in [0.5, 0.54, 0.6]:
                alpha = torch.tensor(alpha_val, dtype=torch.float64)
                result = wf.general_hamming_window(
                    n, alpha, dtype=torch.float64
                )
                expected = torch.signal.windows.general_hamming(
                    n, alpha=alpha_val, sym=True, dtype=torch.float64
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_pytorch_comparison_periodic(self):
        """Compare with torch.signal.windows.general_hamming (periodic)."""
        for n in [4, 16, 64]:
            for alpha_val in [0.5, 0.54, 0.6]:
                alpha = torch.tensor(alpha_val, dtype=torch.float64)
                result = wf.periodic_general_hamming_window(
                    n, alpha, dtype=torch.float64
                )
                expected = torch.signal.windows.general_hamming(
                    n, alpha=alpha_val, sym=False, dtype=torch.float64
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        alpha = torch.tensor(0.54, dtype=torch.float64, requires_grad=True)

        def func(a):
            return wf.general_hamming_window(32, a, dtype=torch.float64).sum()

        torch.autograd.gradcheck(func, (alpha,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        alpha = torch.tensor(0.54, dtype=torch.float64, requires_grad=True)

        def func(a):
            return wf.periodic_general_hamming_window(
                32, a, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (alpha,), raise_exception=True)

    def test_gradient_flow(self):
        """Test that gradients flow through parameter."""
        alpha = torch.tensor(0.54, dtype=torch.float64, requires_grad=True)
        result = wf.general_hamming_window(32, alpha, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert alpha.grad is not None
        assert not torch.isnan(alpha.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        alpha = torch.tensor(0.54, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_general_hamming_window(
            32, alpha, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert alpha.grad is not None
        assert not torch.isnan(alpha.grad)

    def test_meta_tensor(self):
        """Test meta tensor support."""
        alpha = torch.tensor(0.54, device="meta")
        result = wf.general_hamming_window(64, alpha, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_general_hamming_window(
            64, alpha, device="meta"
        )
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        alpha = torch.tensor(0.54, dtype=dtype)
        result = wf.general_hamming_window(64, alpha, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_general_hamming_window(
            64, alpha, dtype=dtype
        )
        assert result_periodic.dtype == dtype

    def test_output_shape(self):
        """Test output shape is (n,)."""
        alpha = torch.tensor(0.54)
        for n in [0, 1, 5, 100]:
            result = wf.general_hamming_window(n, alpha)
            assert result.shape == (n,)
            result_periodic = wf.periodic_general_hamming_window(n, alpha)
            assert result_periodic.shape == (n,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        alpha = torch.tensor(0.54)
        result = wf.general_hamming_window(0, alpha)
        assert result.shape == (0,)
        result_periodic = wf.periodic_general_hamming_window(0, alpha)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        alpha = torch.tensor(0.54, dtype=torch.float64)
        result = wf.general_hamming_window(1, alpha, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_general_hamming_window(
            1, alpha, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric general Hamming window is symmetric."""
        alpha = torch.tensor(0.54, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.general_hamming_window(n, alpha, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value(self):
        """Test that center of odd-length symmetric window is 1.0."""
        alpha = torch.tensor(0.54, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.general_hamming_window(n, alpha, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_endpoint_value(self):
        """Test that endpoint values depend on alpha."""
        # For general Hamming: w[0] = alpha - (1-alpha)*cos(0) = alpha - (1-alpha) = 2*alpha - 1
        for alpha_val in [0.5, 0.54, 0.6]:
            alpha = torch.tensor(alpha_val, dtype=torch.float64)
            n = 64
            result = wf.general_hamming_window(n, alpha, dtype=torch.float64)
            expected_endpoint = torch.tensor(
                2 * alpha_val - 1, dtype=torch.float64
            )
            torch.testing.assert_close(
                result[0], expected_endpoint, atol=1e-10, rtol=0
            )
            torch.testing.assert_close(
                result[-1], expected_endpoint, atol=1e-10, rtol=0
            )

    def test_float_alpha_input(self):
        """Test that alpha can be passed as float."""
        result = wf.general_hamming_window(64, 0.54, dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_general_hamming_window(
            64, 0.54, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    @staticmethod
    def _reference_general_hamming(
        n: int, alpha: float, periodic: bool
    ) -> torch.Tensor:
        """Reference implementation."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        k = torch.arange(n, dtype=torch.float64)
        return alpha - (1 - alpha) * torch.cos(2 * math.pi * k / denom)

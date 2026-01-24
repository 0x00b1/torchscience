import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestGaussianWindow:
    """Tests for gaussian_window and periodic_gaussian_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        std = torch.tensor(0.4, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.gaussian_window(n, std, dtype=torch.float64)
            expected = self._reference_gaussian(n, std.item(), periodic=False)
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        std = torch.tensor(0.4, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_gaussian_window(n, std, dtype=torch.float64)
            expected = self._reference_gaussian(n, std.item(), periodic=True)
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_pytorch_comparison_symmetric(self):
        """Compare with torch.signal.windows.gaussian (symmetric)."""
        for n in [4, 16, 64]:
            for std_val in [0.3, 0.5, 1.0]:
                std = torch.tensor(std_val, dtype=torch.float64)
                result = wf.gaussian_window(n, std, dtype=torch.float64)
                # torch.signal.windows.gaussian uses sig parameter
                # Our std parameter: sigma = std * center, where center = (n-1)/2
                # torch uses: sig directly as sigma
                # So we need to convert: sig = std * (n-1)/2
                sig = std_val * (n - 1) / 2
                expected = torch.signal.windows.gaussian(
                    n, std=sig, sym=True, dtype=torch.float64
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_pytorch_comparison_periodic(self):
        """Compare with torch.signal.windows.gaussian (periodic)."""
        for n in [4, 16, 64]:
            for std_val in [0.3, 0.5, 1.0]:
                std = torch.tensor(std_val, dtype=torch.float64)
                result = wf.periodic_gaussian_window(
                    n, std, dtype=torch.float64
                )
                # For periodic: sigma = std * center, where center = n/2
                sig = std_val * n / 2
                expected = torch.signal.windows.gaussian(
                    n, std=sig, sym=False, dtype=torch.float64
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        std = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.gaussian_window(32, s, dtype=torch.float64).sum()

        torch.autograd.gradcheck(func, (std,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        std = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.periodic_gaussian_window(
                32, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (std,), raise_exception=True)

    def test_gradient_flow(self):
        """Test that gradients flow through parameter."""
        std = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        result = wf.gaussian_window(32, std, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert std.grad is not None
        assert not torch.isnan(std.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        std = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_gaussian_window(32, std, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert std.grad is not None
        assert not torch.isnan(std.grad)

    def test_meta_tensor(self):
        """Test meta tensor support."""
        std = torch.tensor(0.4, device="meta")
        result = wf.gaussian_window(64, std, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_gaussian_window(64, std, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        std = torch.tensor(0.4, dtype=dtype)
        result = wf.gaussian_window(64, std, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_gaussian_window(64, std, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_output_shape(self):
        """Test output shape is (n,)."""
        std = torch.tensor(0.4)
        for n in [0, 1, 5, 100]:
            result = wf.gaussian_window(n, std)
            assert result.shape == (n,)
            result_periodic = wf.periodic_gaussian_window(n, std)
            assert result_periodic.shape == (n,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        std = torch.tensor(0.4)
        result = wf.gaussian_window(0, std)
        assert result.shape == (0,)
        result_periodic = wf.periodic_gaussian_window(0, std)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        std = torch.tensor(0.4, dtype=torch.float64)
        result = wf.gaussian_window(1, std, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_gaussian_window(
            1, std, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric Gaussian window is symmetric."""
        std = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.gaussian_window(n, std, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value(self):
        """Test that center of odd-length symmetric window is 1.0."""
        std = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.gaussian_window(n, std, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_std_affects_width(self):
        """Test that larger std produces wider window."""
        n = 64
        std_narrow = torch.tensor(0.2, dtype=torch.float64)
        std_wide = torch.tensor(0.5, dtype=torch.float64)
        result_narrow = wf.gaussian_window(n, std_narrow, dtype=torch.float64)
        result_wide = wf.gaussian_window(n, std_wide, dtype=torch.float64)
        # Wider std should have larger values at the edges
        assert result_wide[0] > result_narrow[0]
        assert result_wide[-1] > result_narrow[-1]
        # Sum should be larger for wider window
        assert result_wide.sum() > result_narrow.sum()

    def test_float_std_input(self):
        """Test that std can be passed as float."""
        result = wf.gaussian_window(64, 0.4, dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_gaussian_window(
            64, 0.4, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    @staticmethod
    def _reference_gaussian(
        n: int, std: float, periodic: bool
    ) -> torch.Tensor:
        """Reference implementation."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        center = denom / 2
        sigma = std * center
        k = torch.arange(n, dtype=torch.float64)
        x = (k - center) / sigma
        return torch.exp(-0.5 * x * x)

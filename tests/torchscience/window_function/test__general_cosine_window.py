import math

import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestGeneralCosineWindow:
    """Tests for general_cosine_window and periodic_general_cosine_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        coeffs = torch.tensor([0.5, 0.5], dtype=torch.float64)  # Hann coeffs
        for n in [1, 5, 64, 128]:
            result = wf.general_cosine_window(n, coeffs, dtype=torch.float64)
            expected = self._reference_general_cosine(
                n, coeffs.tolist(), periodic=False
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        coeffs = torch.tensor([0.5, 0.5], dtype=torch.float64)  # Hann coeffs
        for n in [1, 5, 64, 128]:
            result = wf.periodic_general_cosine_window(
                n, coeffs, dtype=torch.float64
            )
            expected = self._reference_general_cosine(
                n, coeffs.tolist(), periodic=True
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_hann_special_case(self):
        """Test that coeffs=[0.5, 0.5] produces Hann window."""
        coeffs = torch.tensor([0.5, 0.5], dtype=torch.float64)
        for n in [4, 16, 64]:
            result = wf.general_cosine_window(n, coeffs, dtype=torch.float64)
            expected = wf.hann_window(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_hamming_special_case(self):
        """Test that coeffs=[0.54, 0.46] produces Hamming window."""
        coeffs = torch.tensor([0.54, 0.46], dtype=torch.float64)
        for n in [4, 16, 64]:
            result = wf.general_cosine_window(n, coeffs, dtype=torch.float64)
            expected = wf.hamming_window(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_blackman_special_case(self):
        """Test that coeffs=[0.42, 0.5, 0.08] produces Blackman window."""
        coeffs = torch.tensor([0.42, 0.5, 0.08], dtype=torch.float64)
        for n in [4, 16, 64]:
            result = wf.general_cosine_window(n, coeffs, dtype=torch.float64)
            expected = wf.blackman_window(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_nuttall_special_case(self):
        """Test that Nuttall coeffs produce Nuttall window."""
        coeffs = torch.tensor(
            [0.355768, 0.487396, 0.144232, 0.012604], dtype=torch.float64
        )
        for n in [4, 16, 64]:
            result = wf.general_cosine_window(n, coeffs, dtype=torch.float64)
            expected = wf.nuttall_window(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_pytorch_comparison_symmetric(self):
        """Compare with torch.signal.windows.general_cosine (symmetric)."""
        coeffs_list = [0.5, 0.5]  # Hann
        for n in [4, 16, 64]:
            coeffs = torch.tensor(coeffs_list, dtype=torch.float64)
            result = wf.general_cosine_window(n, coeffs, dtype=torch.float64)
            expected = torch.signal.windows.general_cosine(
                n, a=coeffs_list, sym=True, dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_pytorch_comparison_periodic(self):
        """Compare with torch.signal.windows.general_cosine (periodic)."""
        coeffs_list = [0.5, 0.5]  # Hann
        for n in [4, 16, 64]:
            coeffs = torch.tensor(coeffs_list, dtype=torch.float64)
            result = wf.periodic_general_cosine_window(
                n, coeffs, dtype=torch.float64
            )
            expected = torch.signal.windows.general_cosine(
                n, a=coeffs_list, sym=False, dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        coeffs = torch.tensor(
            [0.5, 0.5], dtype=torch.float64, requires_grad=True
        )

        def func(c):
            return wf.general_cosine_window(32, c, dtype=torch.float64).sum()

        torch.autograd.gradcheck(func, (coeffs,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        coeffs = torch.tensor(
            [0.5, 0.5], dtype=torch.float64, requires_grad=True
        )

        def func(c):
            return wf.periodic_general_cosine_window(
                32, c, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (coeffs,), raise_exception=True)

    def test_gradient_flow(self):
        """Test that gradients flow through parameter."""
        coeffs = torch.tensor(
            [0.5, 0.5], dtype=torch.float64, requires_grad=True
        )
        result = wf.general_cosine_window(32, coeffs, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert coeffs.grad is not None
        assert not torch.isnan(coeffs.grad).any()

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        coeffs = torch.tensor(
            [0.5, 0.5], dtype=torch.float64, requires_grad=True
        )
        result = wf.periodic_general_cosine_window(
            32, coeffs, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert coeffs.grad is not None
        assert not torch.isnan(coeffs.grad).any()

    def test_meta_tensor(self):
        """Test meta tensor support."""
        coeffs = torch.tensor([0.5, 0.5], device="meta")
        result = wf.general_cosine_window(64, coeffs, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_general_cosine_window(
            64, coeffs, device="meta"
        )
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        coeffs = torch.tensor([0.5, 0.5], dtype=dtype)
        result = wf.general_cosine_window(64, coeffs, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_general_cosine_window(
            64, coeffs, dtype=dtype
        )
        assert result_periodic.dtype == dtype

    def test_output_shape(self):
        """Test output shape is (n,)."""
        coeffs = torch.tensor([0.5, 0.5])
        for n in [0, 1, 5, 100]:
            result = wf.general_cosine_window(n, coeffs)
            assert result.shape == (n,)
            result_periodic = wf.periodic_general_cosine_window(n, coeffs)
            assert result_periodic.shape == (n,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        coeffs = torch.tensor([0.5, 0.5])
        result = wf.general_cosine_window(0, coeffs)
        assert result.shape == (0,)
        result_periodic = wf.periodic_general_cosine_window(0, coeffs)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        coeffs = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = wf.general_cosine_window(1, coeffs, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_general_cosine_window(
            1, coeffs, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric general cosine window is symmetric."""
        coeffs = torch.tensor([0.42, 0.5, 0.08], dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.general_cosine_window(n, coeffs, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value(self):
        """Test that center of odd-length symmetric window is coeffs sum."""
        # At center, all cos terms = -1 due to (-1)^j alternation
        # w[center] = sum(coeffs) when all cos terms are 1
        # Actually at center: cos(pi) = -1, so w = a0 + a1 + a2 + ... = sum(coeffs)
        coeffs = torch.tensor(
            [0.42, 0.5, 0.08], dtype=torch.float64
        )  # Blackman
        for n in [5, 11, 65]:
            result = wf.general_cosine_window(n, coeffs, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_list_coeffs_input(self):
        """Test that coeffs can be passed as list."""
        result = wf.general_cosine_window(64, [0.5, 0.5], dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_general_cosine_window(
            64, [0.5, 0.5], dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_different_coefficient_lengths(self):
        """Test different numbers of coefficients."""
        # 1 coefficient (rectangular-like)
        coeffs1 = torch.tensor([1.0], dtype=torch.float64)
        result1 = wf.general_cosine_window(64, coeffs1, dtype=torch.float64)
        assert result1.shape == (64,)
        # All values should be 1.0 for single coeff = 1
        torch.testing.assert_close(
            result1,
            torch.ones(64, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

        # 2 coefficients (Hann/Hamming family)
        coeffs2 = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result2 = wf.general_cosine_window(64, coeffs2, dtype=torch.float64)
        assert result2.shape == (64,)

        # 3 coefficients (Blackman)
        coeffs3 = torch.tensor([0.42, 0.5, 0.08], dtype=torch.float64)
        result3 = wf.general_cosine_window(64, coeffs3, dtype=torch.float64)
        assert result3.shape == (64,)

        # 4 coefficients (Nuttall)
        coeffs4 = torch.tensor(
            [0.355768, 0.487396, 0.144232, 0.012604], dtype=torch.float64
        )
        result4 = wf.general_cosine_window(64, coeffs4, dtype=torch.float64)
        assert result4.shape == (64,)

    @staticmethod
    def _reference_general_cosine(
        n: int, coeffs: list, periodic: bool
    ) -> torch.Tensor:
        """Reference implementation."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        k = torch.arange(n, dtype=torch.float64)
        result = torch.zeros(n, dtype=torch.float64)
        for j, coeff in enumerate(coeffs):
            sign = (-1) ** j
            result = result + coeff * sign * torch.cos(
                2 * math.pi * j * k / denom
            )
        return result

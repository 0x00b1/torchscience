import math

import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestPolylogarithmLi:
    """Tests for the polylogarithm function Li_s(z)."""

    def test_li_at_zero(self):
        """Test that Li_s(0) = 0 for any s."""
        s = torch.tensor([1.0, 2.0, 3.0, 0.5, -1.0], dtype=torch.float64)
        z = torch.tensor([0.0], dtype=torch.float64)
        # Broadcasting
        result = torchscience.special_functions.polylogarithm_li(s, z)
        expected = torch.zeros(5, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_dilogarithm_known_values(self):
        """Test Li_2 at known values."""
        s = torch.tensor([2.0], dtype=torch.float64)

        # Li_2(0.5) = pi^2/12 - ln(2)^2/2 approximately equals 0.58224052646
        z1 = torch.tensor([0.5], dtype=torch.float64)
        result1 = torchscience.special_functions.polylogarithm_li(s, z1)
        expected1 = math.pi**2 / 12 - math.log(2) ** 2 / 2
        torch.testing.assert_close(
            result1,
            torch.tensor([expected1], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )

        # Li_2(1) = pi^2/6 (Basel problem)
        z2 = torch.tensor([1.0], dtype=torch.float64)
        result2 = torchscience.special_functions.polylogarithm_li(s, z2)
        expected2 = math.pi**2 / 6
        torch.testing.assert_close(
            result2,
            torch.tensor([expected2], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_trilogarithm_known_values(self):
        """Test Li_3 at known values."""
        s = torch.tensor([3.0], dtype=torch.float64)

        # Li_3(0.5) approximately equals 0.5372131936
        z = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.polylogarithm_li(s, z)
        expected = 0.5372131936
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_li_1_is_negative_log(self):
        """Test that Li_1(z) = -ln(1-z)."""
        s = torch.tensor([1.0], dtype=torch.float64)
        z_values = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)

        for z_val in z_values:
            z = z_val.unsqueeze(0)
            result = torchscience.special_functions.polylogarithm_li(s, z)
            expected = -torch.log(1 - z)
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_li_s_at_one_equals_zeta(self):
        """Test that Li_s(1) = zeta(s) for s > 1."""
        s_values = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)

        # Known zeta values
        zeta_values = torch.tensor(
            [
                math.pi**2 / 6,  # zeta(2)
                1.2020569031595943,  # zeta(3) - Apery's constant
                math.pi**4 / 90,  # zeta(4)
                1.0369277551433699,  # zeta(5)
            ],
            dtype=torch.float64,
        )

        for i, s_val in enumerate(s_values):
            s = s_val.unsqueeze(0)
            result = torchscience.special_functions.polylogarithm_li(s, z)
            torch.testing.assert_close(
                result, zeta_values[i].unsqueeze(0), rtol=1e-4, atol=1e-4
            )

    def test_small_z_convergence(self):
        """Test convergence for small z values."""
        s = torch.tensor([2.0], dtype=torch.float64)
        z_values = torch.tensor([0.01, 0.05, 0.1], dtype=torch.float64)

        for z_val in z_values:
            z = z_val.unsqueeze(0)
            result = torchscience.special_functions.polylogarithm_li(s, z)
            # For small z, Li_s(z) approximately equals z (first term of series)
            # More precisely: Li_s(z) = z + z^2/2^s + z^3/3^s + ...
            # For very small z, this is approximately z
            assert torch.isfinite(result).all()
            assert result.item() > 0

    def test_negative_z(self):
        """Test polylogarithm for negative z with |z| <= 1."""
        s = torch.tensor([2.0], dtype=torch.float64)

        # Li_2(-1) = -pi^2/12 approximately equals -0.8225
        z = torch.tensor([-1.0], dtype=torch.float64)
        result = torchscience.special_functions.polylogarithm_li(s, z)
        expected = -(math.pi**2) / 12
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_gradient_finite(self):
        """Test that gradients are finite for valid inputs."""
        s = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.polylogarithm_li(s, z)
        y.sum().backward()
        assert torch.isfinite(s.grad).all()
        assert torch.isfinite(z.grad).all()

    def test_gradient_z_formula(self):
        """Test that dLi_s/dz = Li_{s-1}(z) / z."""
        s = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.polylogarithm_li(s, z)
        y.backward()

        # Expected: Li_2(0.5) / 0.5
        li_sm1 = torchscience.special_functions.polylogarithm_li(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
        )
        expected_grad = li_sm1 / 0.5
        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        s = torch.tensor([2.5, 3.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            torchscience.special_functions.polylogarithm_li,
            (s, z),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        s = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.polylogarithm_li,
            (s, z),
            eps=1e-5,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_float32_precision(self):
        """Test float32 inputs."""
        s = torch.tensor([2.0], dtype=torch.float32)
        z = torch.tensor([0.5], dtype=torch.float32)
        result = torchscience.special_functions.polylogarithm_li(s, z)
        # Li_2(0.5) approximately equals 0.5822
        torch.testing.assert_close(
            result,
            torch.tensor([0.5822], dtype=torch.float32),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        s = torch.tensor([[2.0], [3.0]], dtype=torch.float64)  # shape (2, 1)
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)  # shape (3,)
        result = torchscience.special_functions.polylogarithm_li(s, z)
        assert result.shape == (2, 3)
        assert torch.isfinite(result).all()

    def test_empty_tensor(self):
        """Test empty tensor input."""
        s = torch.tensor([], dtype=torch.float64)
        z = torch.tensor([], dtype=torch.float64)
        result = torchscience.special_functions.polylogarithm_li(s, z)
        assert result.shape == torch.Size([0])

    def test_meta_tensor(self):
        """Test meta tensor support."""
        s = torch.tensor([2.0], dtype=torch.float64, device="meta")
        z = torch.tensor([0.5], dtype=torch.float64, device="meta")
        result = torchscience.special_functions.polylogarithm_li(s, z)
        assert result.device.type == "meta"
        assert result.shape == s.shape

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test CUDA support."""
        s = torch.tensor([2.0, 3.0], dtype=torch.float64, device="cuda")
        z = torch.tensor([0.5, 0.5], dtype=torch.float64, device="cuda")
        result = torchscience.special_functions.polylogarithm_li(s, z)
        assert result.device.type == "cuda"
        expected_cpu = torchscience.special_functions.polylogarithm_li(
            s.cpu(), z.cpu()
        )
        torch.testing.assert_close(
            result.cpu(), expected_cpu, rtol=1e-6, atol=1e-6
        )

    def test_different_s_values(self):
        """Test various values of s."""
        z = torch.tensor([0.5], dtype=torch.float64)

        # Test multiple s values
        s_values = [0.5, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        for s_val in s_values:
            s = torch.tensor([s_val], dtype=torch.float64)
            result = torchscience.special_functions.polylogarithm_li(s, z)
            assert torch.isfinite(result).all(), f"Failed for s={s_val}"

    def test_series_convergence(self):
        """Test that the series gives consistent results for different z."""
        s = torch.tensor([2.0], dtype=torch.float64)
        z_values = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)

        results = []
        for z_val in z_values:
            z = z_val.unsqueeze(0)
            result = torchscience.special_functions.polylogarithm_li(s, z)
            results.append(result.item())

        # Results should be monotonically increasing for z > 0
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"Li_2 should be increasing: Li_2({z_values[i].item()}) = {results[i]} "
                f"should be < Li_2({z_values[i + 1].item()}) = {results[i + 1]}"
            )

    def test_complex_input(self):
        """Test complex input support."""
        s = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([0.5 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.polylogarithm_li(s, z)
        # Should match real result
        expected = torchscience.special_functions.polylogarithm_li(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
        )
        torch.testing.assert_close(result.real, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            result.imag,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )

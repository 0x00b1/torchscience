import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestUltrasphericalWindow:
    """Tests for ultraspherical_window and periodic_ultraspherical_window."""

    def test_symmetric_basic(self):
        """Test basic symmetric window generation."""
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        for n in [4, 8, 16, 32]:
            result = wf.ultraspherical_window(n, mu, x_mu, dtype=torch.float64)
            assert result.shape == (n,)
            assert result.dtype == torch.float64

    def test_periodic_basic(self):
        """Test basic periodic window generation."""
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        for n in [4, 8, 16, 32]:
            result = wf.periodic_ultraspherical_window(
                n, mu, x_mu, dtype=torch.float64
            )
            assert result.shape == (n,)
            assert result.dtype == torch.float64

    def test_output_shape(self):
        """Test output shape is (n,)."""
        mu = torch.tensor(1.0)
        x_mu = torch.tensor(1.5)
        for n in [0, 1, 5, 64, 100]:
            result = wf.ultraspherical_window(n, mu, x_mu)
            assert result.shape == (n,)
            result_periodic = wf.periodic_ultraspherical_window(n, mu, x_mu)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        mu = torch.tensor(1.0, dtype=dtype)
        x_mu = torch.tensor(1.5, dtype=dtype)
        result = wf.ultraspherical_window(64, mu, x_mu, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_ultraspherical_window(
            64, mu, x_mu, dtype=dtype
        )
        assert result_periodic.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        mu = torch.tensor(1.0)
        x_mu = torch.tensor(1.5)
        result = wf.ultraspherical_window(0, mu, x_mu)
        assert result.shape == (0,)
        result_periodic = wf.periodic_ultraspherical_window(0, mu, x_mu)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        result = wf.ultraspherical_window(1, mu, x_mu, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_ultraspherical_window(
            1, mu, x_mu, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric ultraspherical window is symmetric."""
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.ultraspherical_window(n, mu, x_mu, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-5, atol=1e-5)

    def test_symmetry_various_params(self):
        """Test symmetry with various parameter combinations."""
        params = [(0.5, 1.2), (1.0, 2.0), (2.0, 1.5)]
        for mu_val, x_mu_val in params:
            mu = torch.tensor(mu_val, dtype=torch.float64)
            x_mu = torch.tensor(x_mu_val, dtype=torch.float64)
            for n in [8, 11, 16]:
                result = wf.ultraspherical_window(
                    n, mu, x_mu, dtype=torch.float64
                )
                flipped = torch.flip(result, dims=[0])
                torch.testing.assert_close(
                    result, flipped, rtol=1e-5, atol=1e-5
                )

    def test_maximum_value_one(self):
        """Test that maximum value is 1.0 (normalized)."""
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        for n in [8, 16, 32]:
            result = wf.ultraspherical_window(n, mu, x_mu, dtype=torch.float64)
            torch.testing.assert_close(
                result.max(),
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-6,
                rtol=0,
            )

    def test_values_between_zero_and_one(self):
        """Test that all window values are in [0, 1] or close."""
        params = [(0.5, 1.2), (1.0, 1.5), (2.0, 2.0)]
        for mu_val, x_mu_val in params:
            mu = torch.tensor(mu_val, dtype=torch.float64)
            x_mu = torch.tensor(x_mu_val, dtype=torch.float64)
            for n in [8, 16, 32]:
                result = wf.ultraspherical_window(
                    n, mu, x_mu, dtype=torch.float64
                )
                # Allow small numerical errors
                assert result.min() >= -1e-6
                assert result.max() <= 1.0 + 1e-6

    def test_gradient_flow_mu(self):
        """Test that gradients flow through mu parameter."""
        mu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        result = wf.ultraspherical_window(16, mu, x_mu, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert mu.grad is not None
        assert not torch.isnan(mu.grad)

    def test_gradient_flow_x_mu(self):
        """Test that gradients flow through x_mu parameter."""
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        result = wf.ultraspherical_window(16, mu, x_mu, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert x_mu.grad is not None
        assert not torch.isnan(x_mu.grad)

    def test_gradient_flow_both(self):
        """Test that gradients flow through both parameters."""
        mu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        x_mu = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        result = wf.ultraspherical_window(16, mu, x_mu, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert mu.grad is not None
        assert x_mu.grad is not None
        assert not torch.isnan(mu.grad)
        assert not torch.isnan(x_mu.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        mu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        x_mu = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_ultraspherical_window(
            16, mu, x_mu, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert mu.grad is not None
        assert x_mu.grad is not None
        assert not torch.isnan(mu.grad)
        assert not torch.isnan(x_mu.grad)

    def test_gradcheck_mu(self):
        """Test gradient correctness for mu with torch.autograd.gradcheck."""
        mu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        x_mu = torch.tensor(1.5, dtype=torch.float64)

        def func(m):
            return wf.ultraspherical_window(
                8, m, x_mu, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (mu,), raise_exception=True)

    def test_gradcheck_x_mu(self):
        """Test gradient correctness for x_mu with torch.autograd.gradcheck."""
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def func(x):
            return wf.ultraspherical_window(
                8, mu, x, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (x_mu,), raise_exception=True)

    def test_gradcheck_both(self):
        """Test gradient correctness for both params."""
        mu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        x_mu = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def func(m, x):
            return wf.ultraspherical_window(8, m, x, dtype=torch.float64).sum()

        torch.autograd.gradcheck(func, (mu, x_mu), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        mu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        x_mu = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def func(m, x):
            return wf.periodic_ultraspherical_window(
                8, m, x, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (mu, x_mu), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        mu = torch.tensor(1.0)
        x_mu = torch.tensor(1.5)
        with pytest.raises(ValueError):
            wf.ultraspherical_window(-1, mu, x_mu)
        with pytest.raises(ValueError):
            wf.periodic_ultraspherical_window(-1, mu, x_mu)

    def test_invalid_mu_raises(self):
        """Test that mu <= 0 raises error."""
        x_mu = torch.tensor(1.5)
        with pytest.raises(ValueError):
            wf.ultraspherical_window(16, 0.0, x_mu)
        with pytest.raises(ValueError):
            wf.ultraspherical_window(16, -1.0, x_mu)
        with pytest.raises(ValueError):
            wf.periodic_ultraspherical_window(16, 0.0, x_mu)
        with pytest.raises(ValueError):
            wf.periodic_ultraspherical_window(16, -1.0, x_mu)

    def test_invalid_x_mu_raises(self):
        """Test that x_mu <= 1 raises error."""
        mu = torch.tensor(1.0)
        with pytest.raises(ValueError):
            wf.ultraspherical_window(16, mu, 1.0)
        with pytest.raises(ValueError):
            wf.ultraspherical_window(16, mu, 0.5)
        with pytest.raises(ValueError):
            wf.periodic_ultraspherical_window(16, mu, 1.0)
        with pytest.raises(ValueError):
            wf.periodic_ultraspherical_window(16, mu, 0.5)

    def test_float_param_input(self):
        """Test that mu and x_mu can be passed as floats."""
        result = wf.ultraspherical_window(64, 1.0, 1.5, dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_ultraspherical_window(
            64, 1.0, 1.5, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_mu_affects_shape(self):
        """Test that mu affects the window shape."""
        n = 32
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        mu_small = torch.tensor(0.5, dtype=torch.float64)
        mu_large = torch.tensor(2.0, dtype=torch.float64)
        result_small = wf.ultraspherical_window(
            n, mu_small, x_mu, dtype=torch.float64
        )
        result_large = wf.ultraspherical_window(
            n, mu_large, x_mu, dtype=torch.float64
        )
        # Different mu should produce different windows
        assert not torch.allclose(result_small, result_large)

    def test_x_mu_affects_shape(self):
        """Test that x_mu affects the window shape."""
        n = 32
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu_small = torch.tensor(1.2, dtype=torch.float64)
        x_mu_large = torch.tensor(2.0, dtype=torch.float64)
        result_small = wf.ultraspherical_window(
            n, mu, x_mu_small, dtype=torch.float64
        )
        result_large = wf.ultraspherical_window(
            n, mu, x_mu_large, dtype=torch.float64
        )
        # Larger x_mu should produce lower sidelobes (larger values near edges)
        # Check that the edge values differ
        assert not torch.allclose(result_small, result_large)

    def test_periodic_vs_symmetric_difference(self):
        """Test that periodic and symmetric windows differ."""
        n = 16
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        symmetric = wf.ultraspherical_window(n, mu, x_mu, dtype=torch.float64)
        periodic = wf.periodic_ultraspherical_window(
            n, mu, x_mu, dtype=torch.float64
        )
        # They should be different (different denominators)
        assert not torch.allclose(symmetric, periodic)

    def test_saramaki_window_mu_one(self):
        """Test mu=1 case (Saramaki window)."""
        # mu=1 should produce a valid window with specific characteristics
        n = 32
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        result = wf.ultraspherical_window(n, mu, x_mu, dtype=torch.float64)
        # Should be symmetric
        flipped = torch.flip(result, dims=[0])
        torch.testing.assert_close(result, flipped, rtol=1e-5, atol=1e-5)
        # Maximum should be 1
        torch.testing.assert_close(
            result.max(),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-6,
            rtol=0,
        )

    def test_n_equals_two(self):
        """Test specific case n=2."""
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu = torch.tensor(1.5, dtype=torch.float64)
        result = wf.ultraspherical_window(2, mu, x_mu, dtype=torch.float64)
        assert result.shape == (2,)
        # Should be symmetric: both values equal
        torch.testing.assert_close(result[0], result[1], rtol=1e-5, atol=1e-5)

    def test_larger_x_mu_narrower_effective_width(self):
        """Test that larger x_mu creates different sidelobe characteristics."""
        n = 64
        mu = torch.tensor(1.0, dtype=torch.float64)
        x_mu_low = torch.tensor(1.2, dtype=torch.float64)
        x_mu_high = torch.tensor(3.0, dtype=torch.float64)

        result_low = wf.ultraspherical_window(
            n, mu, x_mu_low, dtype=torch.float64
        )
        result_high = wf.ultraspherical_window(
            n, mu, x_mu_high, dtype=torch.float64
        )

        # Both should be normalized to max 1
        torch.testing.assert_close(
            result_low.max(),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            result_high.max(),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-6,
            rtol=0,
        )

        # The windows should be different
        assert not torch.allclose(result_low, result_high)


class TestGegenbauerPolynomial:
    """Tests for the internal _gegenbauer_polynomial function."""

    def test_c0_is_one(self):
        """C_0^mu(x) = 1 for all mu and x."""
        from torchscience.window_function._ultraspherical_window import (
            _gegenbauer_polynomial,
        )

        mu = torch.tensor(1.5, dtype=torch.float64)
        x = torch.tensor([0.0, 0.5, 1.0, -0.5], dtype=torch.float64)
        result = _gegenbauer_polynomial(0, mu, x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_c1_is_2_mu_x(self):
        """C_1^mu(x) = 2*mu*x."""
        from torchscience.window_function._ultraspherical_window import (
            _gegenbauer_polynomial,
        )

        mu = torch.tensor(1.5, dtype=torch.float64)
        x = torch.tensor([0.0, 0.5, 1.0, -0.5], dtype=torch.float64)
        result = _gegenbauer_polynomial(1, mu, x)
        expected = 2.0 * mu * x
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_c2_formula(self):
        """Test C_2^mu(x) using explicit formula.

        C_2^mu(x) = 2*mu*(1+mu)*x^2 - mu
        """
        from torchscience.window_function._ultraspherical_window import (
            _gegenbauer_polynomial,
        )

        mu = torch.tensor(1.5, dtype=torch.float64)
        x = torch.tensor([0.0, 0.5, 1.0, -0.5], dtype=torch.float64)
        result = _gegenbauer_polynomial(2, mu, x)
        expected = 2.0 * mu * (1.0 + mu) * x * x - mu
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_chebyshev_limit_mu_half(self):
        """When mu=0.5, Gegenbauer reduces to Legendre polynomials.

        Actually, C_n^{1/2}(x) = P_n(x) (Legendre polynomial).
        P_2(x) = (3*x^2 - 1) / 2
        """
        from torchscience.window_function._ultraspherical_window import (
            _gegenbauer_polynomial,
        )

        mu = torch.tensor(0.5, dtype=torch.float64)
        x = torch.tensor([0.0, 0.5, 1.0, -0.5], dtype=torch.float64)
        result = _gegenbauer_polynomial(2, mu, x)
        # P_2(x) = (3x^2 - 1) / 2
        expected = (3.0 * x * x - 1.0) / 2.0
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

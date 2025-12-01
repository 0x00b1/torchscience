import pytest
import torch
import torch.testing

import torchscience.signal_processing.window_function as wf


class TestGeneralizedAdaptivePolynomialWindow:
    """Tests for generalized_adaptive_polynomial_window and periodic version."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)
        for n in [1, 2, 5, 64, 128]:
            result = wf.generalized_adaptive_polynomial_window(
                n, alpha, beta, dtype=torch.float64
            )
            expected = self._reference_gap(
                n, alpha.item(), beta.item(), periodic=False
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_generalized_adaptive_polynomial_window(
                n, alpha, beta, dtype=torch.float64
            )
            expected = self._reference_gap(
                n, alpha.item(), beta.item(), periodic=True
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_alpha_beta_variations(self):
        """Test various alpha and beta combinations."""
        n = 32
        test_cases = [
            (1.0, 1.0),  # Triangular-like
            (2.0, 1.0),  # Welch
            (2.0, 0.5),  # Cosine-like
            (3.0, 1.0),  # Steeper
            (2.0, 2.0),  # Narrower
            (1.5, 0.75),  # Mixed
        ]
        for alpha_val, beta_val in test_cases:
            alpha = torch.tensor(alpha_val, dtype=torch.float64)
            beta = torch.tensor(beta_val, dtype=torch.float64)
            result = wf.generalized_adaptive_polynomial_window(
                n, alpha, beta, dtype=torch.float64
            )
            expected = self._reference_gap(
                n, alpha_val, beta_val, periodic=False
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_welch_equivalence(self):
        """alpha=2, beta=1 should produce Welch window."""
        for n in [5, 10, 64]:
            gap = wf.generalized_adaptive_polynomial_window(
                n, alpha=2.0, beta=1.0, dtype=torch.float64
            )
            welch = wf.welch_window(n, dtype=torch.float64)
            torch.testing.assert_close(gap, welch, rtol=1e-10, atol=1e-10)

    def test_periodic_welch_equivalence(self):
        """Periodic with alpha=2, beta=1 should match periodic Welch."""
        for n in [5, 10, 64]:
            gap = wf.periodic_generalized_adaptive_polynomial_window(
                n, alpha=2.0, beta=1.0, dtype=torch.float64
            )
            welch = wf.periodic_welch_window(n, dtype=torch.float64)
            torch.testing.assert_close(gap, welch, rtol=1e-10, atol=1e-10)

    def test_output_shape(self):
        """Test output shape is (n,)."""
        alpha = torch.tensor(2.0)
        beta = torch.tensor(1.0)
        for n in [0, 1, 5, 100]:
            result = wf.generalized_adaptive_polynomial_window(n, alpha, beta)
            assert result.shape == (n,)
            result_periodic = (
                wf.periodic_generalized_adaptive_polynomial_window(
                    n, alpha, beta
                )
            )
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        alpha = torch.tensor(2.0, dtype=dtype)
        beta = torch.tensor(1.0, dtype=dtype)
        result = wf.generalized_adaptive_polynomial_window(
            64, alpha, beta, dtype=dtype
        )
        assert result.dtype == dtype
        result_periodic = wf.periodic_generalized_adaptive_polynomial_window(
            64, alpha, beta, dtype=dtype
        )
        assert result_periodic.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        alpha = torch.tensor(2.0)
        beta = torch.tensor(1.0)
        result = wf.generalized_adaptive_polynomial_window(0, alpha, beta)
        assert result.shape == (0,)
        result_periodic = wf.periodic_generalized_adaptive_polynomial_window(
            0, alpha, beta
        )
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)
        result = wf.generalized_adaptive_polynomial_window(
            1, alpha, beta, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_generalized_adaptive_polynomial_window(
            1, alpha, beta, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_n_equals_two(self):
        """Test specific case n=2."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)
        result = wf.generalized_adaptive_polynomial_window(
            2, alpha, beta, dtype=torch.float64
        )
        # For n=2 symmetric, denom = 1
        # x[0] = 2*0/1 - 1 = -1, w[0] = (1 - |-1|^2)^1 = 0
        # x[1] = 2*1/1 - 1 = 1, w[1] = (1 - |1|^2)^1 = 0
        expected = torch.tensor([0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_symmetry(self):
        """Test that symmetric window is symmetric."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.generalized_adaptive_polynomial_window(
                n, alpha, beta, dtype=torch.float64
            )
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_symmetry_various_params(self):
        """Test symmetry with various parameter combinations."""
        params = [(1.0, 1.0), (3.0, 2.0), (1.5, 0.5)]
        for alpha_val, beta_val in params:
            alpha = torch.tensor(alpha_val, dtype=torch.float64)
            beta = torch.tensor(beta_val, dtype=torch.float64)
            for n in [5, 10, 11]:
                result = wf.generalized_adaptive_polynomial_window(
                    n, alpha, beta, dtype=torch.float64
                )
                flipped = torch.flip(result, dims=[0])
                torch.testing.assert_close(
                    result, flipped, rtol=1e-10, atol=1e-10
                )

    def test_center_value_one_for_odd(self):
        """Test that center value is 1.0 for odd-length symmetric windows."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.generalized_adaptive_polynomial_window(
                n, alpha, beta, dtype=torch.float64
            )
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_zero_endpoints_symmetric(self):
        """Test that symmetric window has zero endpoints."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)
        for n in [5, 10, 64]:
            result = wf.generalized_adaptive_polynomial_window(
                n, alpha, beta, dtype=torch.float64
            )
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )
            torch.testing.assert_close(
                result[-1],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_values_between_zero_and_one(self):
        """Test that all window values are in [0, 1]."""
        params = [(1.0, 1.0), (2.0, 1.0), (2.0, 0.5), (3.0, 2.0)]
        for alpha_val, beta_val in params:
            alpha = torch.tensor(alpha_val, dtype=torch.float64)
            beta = torch.tensor(beta_val, dtype=torch.float64)
            for n in [5, 10, 64]:
                result = wf.generalized_adaptive_polynomial_window(
                    n, alpha, beta, dtype=torch.float64
                )
                assert result.min() >= 0.0
                assert result.max() <= 1.0
                result_periodic = (
                    wf.periodic_generalized_adaptive_polynomial_window(
                        n, alpha, beta, dtype=torch.float64
                    )
                )
                assert result_periodic.min() >= 0.0
                assert result_periodic.max() <= 1.0

    def test_gradient_flow_alpha(self):
        """Test that gradients flow through alpha parameter."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(1.0, dtype=torch.float64)
        result = wf.generalized_adaptive_polynomial_window(
            32, alpha, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert alpha.grad is not None
        assert not torch.isnan(alpha.grad)

    def test_gradient_flow_beta(self):
        """Test that gradients flow through beta parameter."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        result = wf.generalized_adaptive_polynomial_window(
            32, alpha, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert beta.grad is not None
        assert not torch.isnan(beta.grad)

    def test_gradient_flow_both(self):
        """Test that gradients flow through both parameters."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        result = wf.generalized_adaptive_polynomial_window(
            32, alpha, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert alpha.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(alpha.grad)
        assert not torch.isnan(beta.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_generalized_adaptive_polynomial_window(
            32, alpha, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert alpha.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(alpha.grad)
        assert not torch.isnan(beta.grad)

    def test_gradcheck_alpha(self):
        """Test gradient correctness for alpha with torch.autograd.gradcheck."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(1.0, dtype=torch.float64)

        def func(a):
            return wf.generalized_adaptive_polynomial_window(
                16, a, beta, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (alpha,), raise_exception=True)

    def test_gradcheck_beta(self):
        """Test gradient correctness for beta with torch.autograd.gradcheck."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        def func(b):
            return wf.generalized_adaptive_polynomial_window(
                16, alpha, b, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (beta,), raise_exception=True)

    def test_gradcheck_both(self):
        """Test gradient correctness for both params."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def func(a, b):
            return wf.generalized_adaptive_polynomial_window(
                16, a, b, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (alpha, beta), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def func(a, b):
            return wf.periodic_generalized_adaptive_polynomial_window(
                16, a, b, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (alpha, beta), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        alpha = torch.tensor(2.0)
        beta = torch.tensor(1.0)
        with pytest.raises(ValueError):
            wf.generalized_adaptive_polynomial_window(-1, alpha, beta)
        with pytest.raises(ValueError):
            wf.periodic_generalized_adaptive_polynomial_window(-1, alpha, beta)

    def test_float_param_input(self):
        """Test that alpha and beta can be passed as floats."""
        result = wf.generalized_adaptive_polynomial_window(
            64, 2.0, 1.0, dtype=torch.float64
        )
        assert result.shape == (64,)
        result_periodic = wf.periodic_generalized_adaptive_polynomial_window(
            64, 2.0, 1.0, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_alpha_affects_shape(self):
        """Test that larger alpha produces steeper transitions."""
        n = 64
        beta = torch.tensor(1.0, dtype=torch.float64)
        alpha_small = torch.tensor(1.0, dtype=torch.float64)
        alpha_large = torch.tensor(4.0, dtype=torch.float64)
        result_small = wf.generalized_adaptive_polynomial_window(
            n, alpha_small, beta, dtype=torch.float64
        )
        result_large = wf.generalized_adaptive_polynomial_window(
            n, alpha_large, beta, dtype=torch.float64
        )
        # Larger alpha should have larger values near edges (less steep decay)
        # Check a point near the edge (but not at the edge which is 0)
        edge_idx = 5
        assert result_large[edge_idx] > result_small[edge_idx]

    def test_beta_affects_shape(self):
        """Test that larger beta produces narrower window."""
        n = 64
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta_small = torch.tensor(0.5, dtype=torch.float64)
        beta_large = torch.tensor(2.0, dtype=torch.float64)
        result_small = wf.generalized_adaptive_polynomial_window(
            n, alpha, beta_small, dtype=torch.float64
        )
        result_large = wf.generalized_adaptive_polynomial_window(
            n, alpha, beta_large, dtype=torch.float64
        )
        # Larger beta should have smaller values away from center
        mid_edge_idx = n // 4
        assert result_large[mid_edge_idx] < result_small[mid_edge_idx]

    def test_default_parameters(self):
        """Test default parameter values (alpha=2.0, beta=1.0)."""
        n = 32
        result_default = wf.generalized_adaptive_polynomial_window(
            n, dtype=torch.float64
        )
        result_explicit = wf.generalized_adaptive_polynomial_window(
            n, 2.0, 1.0, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_default, result_explicit, rtol=1e-10, atol=1e-10
        )

    def test_periodic_vs_symmetric_difference(self):
        """Test that periodic and symmetric windows differ."""
        n = 10
        alpha = torch.tensor(2.0, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)
        symmetric = wf.generalized_adaptive_polynomial_window(
            n, alpha, beta, dtype=torch.float64
        )
        periodic = wf.periodic_generalized_adaptive_polynomial_window(
            n, alpha, beta, dtype=torch.float64
        )
        # They should be different (different denominators)
        assert not torch.allclose(symmetric, periodic)

    @staticmethod
    def _reference_gap(
        n: int, alpha: float, beta: float, periodic: bool
    ) -> torch.Tensor:
        """Reference implementation of Generalized Adaptive Polynomial window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        denom = float(n) if periodic else float(n - 1)
        k = torch.arange(n, dtype=torch.float64)
        x = 2.0 * k / denom - 1.0

        # w[k] = (1 - |x|^alpha)^beta
        abs_x_alpha = torch.abs(x).pow(alpha)
        return torch.clamp(1.0 - abs_x_alpha, min=0.0).pow(beta)

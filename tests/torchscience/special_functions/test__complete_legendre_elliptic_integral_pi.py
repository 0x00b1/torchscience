import mpmath
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    ToleranceConfig,
)


class TestCompleteLegendreEllipticIntegralPi(OpTestCase):
    """Tests for the complete elliptic integral of the third kind Pi(n, m)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="complete_legendre_elliptic_integral_pi",
            func=torchscience.special_functions.complete_legendre_elliptic_integral_pi,
            arity=2,
            input_specs=[
                InputSpec(
                    name="n",
                    position=0,
                    default_real_range=(-2.0, 0.0),
                ),
                InputSpec(
                    name="m",
                    position=1,
                    default_real_range=(0.05, 0.95),
                ),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradgradcheck",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
                "test_vmap_over_batch",
                "test_compile_smoke",
                "test_low_precision_forward",
            },
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath for Pi(0, m) = K(m).

        When n=0, the complete elliptic integral of the third kind reduces
        to the complete elliptic integral of the first kind K(m), and both
        torchscience and mpmath agree exactly.
        """
        m_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        for m_val in m_vals:
            n = torch.tensor([0.0], dtype=torch.float64)
            m = torch.tensor([m_val], dtype=torch.float64)
            result = torchscience.special_functions.complete_legendre_elliptic_integral_pi(
                n, m
            )
            expected = float(mpmath.ellippi(0.0, mpmath.pi / 2, m_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-8,
                atol=1e-10,
            )

    def test_n_zero_equals_K(self):
        """Pi(0, m) = K(m)."""
        m = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        n = torch.zeros_like(m)
        pi_result = torchscience.special_functions.complete_legendre_elliptic_integral_pi(
            n, m
        )
        K = torchscience.special_functions.complete_legendre_elliptic_integral_k(
            m
        )
        torch.testing.assert_close(pi_result, K, rtol=1e-8, atol=1e-10)

    def test_carlson_relation(self):
        """Pi(n, m) = R_F(0, 1-m, 1) + (n/3)*R_J(0, 1-m, 1, 1-n).

        Verifies internal self-consistency of the Carlson-based computation
        using non-positive n values.
        """
        n_vals = torch.tensor(
            [-1.0, -0.5, 0.0, -1.5, -2.0, -0.1], dtype=torch.float64
        )
        m_vals = torch.tensor(
            [0.5, 0.3, 0.5, 0.4, 0.7, 0.9], dtype=torch.float64
        )
        result = torchscience.special_functions.complete_legendre_elliptic_integral_pi(
            n_vals, m_vals
        )
        zero = torch.zeros_like(n_vals)
        one = torch.ones_like(n_vals)
        rf = torchscience.special_functions.carlson_elliptic_integral_r_f(
            zero, 1.0 - m_vals, one
        )
        rj = torchscience.special_functions.carlson_elliptic_integral_r_j(
            zero, 1.0 - m_vals, one, 1.0 - n_vals
        )
        expected = rf + (n_vals / 3.0) * rj
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

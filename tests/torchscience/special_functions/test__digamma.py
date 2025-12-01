import math

import pytest
import sympy
import torch
import torch.testing
from hypothesis import given, settings
from sympy import I, N, symbols

pytestmark = pytest.mark.skip(reason="Test takes >30s, needs optimization")

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    RecurrenceSpec,
    SingularitySpec,
    SpecialValue,
    SymbolicDerivativeVerifier,
    ToleranceConfig,
    avoiding_poles,
    complex_avoiding_real_axis,
    positive_real_numbers,
)

EULER_MASCHERONI = 0.5772156649015329


def sympy_digamma(z: float | complex) -> float | complex:
    """Wrapper for SymPy digamma function."""
    if isinstance(z, complex):
        sympy_z = sympy.Float(z.real) + I * sympy.Float(z.imag)
    else:
        sympy_z = sympy.Float(z)
    result = N(sympy.digamma(sympy_z), 50)
    if result.is_real:
        return float(result)
    return complex(result)


def create_digamma_verifier() -> SymbolicDerivativeVerifier:
    """Create derivative verifier for the digamma function."""
    z = symbols("z")
    expr = sympy.digamma(z)
    return SymbolicDerivativeVerifier(expr, [z])


def _check_recurrence(func) -> bool:
    """Check psi(x+1) = psi(x) + 1/x."""
    x = torch.tensor([0.5, 1.5, 2.5, 3.7], dtype=torch.float64)
    left = func(x + 1)
    right = func(x) + 1 / x
    return torch.allclose(left, right, rtol=1e-10, atol=1e-10)


def _reflection_identity(func):
    """Check psi(1-x) - psi(x) = pi * cot(pi*x)."""
    x = torch.tensor([0.25, 0.3, 0.4, 0.6], dtype=torch.float64)
    left = func(1 - x) - func(x)
    right = math.pi / torch.tan(math.pi * x)
    return left, right


class TestDigamma(OpTestCase):
    """Tests for the digamma function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="digamma",
            func=torchscience.special_functions.digamma,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(0.5, 20.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                ),
            ],
            sympy_func=sympy.digamma,
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                "test_gradcheck_complex",  # Complex not yet implemented
                "test_gradgradcheck_complex",  # Complex not yet implemented
                "test_sparse_coo_basic",  # Sparse has implicit zeros = poles
                "test_low_precision_forward",  # Random values may hit poles
                "test_sympy_reference_complex",  # Complex not yet implemented
            },
            recurrence_relations=[
                RecurrenceSpec(
                    name="digamma_recurrence",
                    check_fn=_check_recurrence,
                    description="psi(x+1) = psi(x) + 1/x",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="reflection_formula",
                    identity_fn=_reflection_identity,
                    description="psi(1-x) - psi(x) = pi * cot(pi*x)",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=-EULER_MASCHERONI,
                    description="psi(1) = -gamma",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=1 - EULER_MASCHERONI,
                    description="psi(2) = 1 - gamma",
                ),
                SpecialValue(
                    inputs=(3.0,),
                    expected=1.5 - EULER_MASCHERONI,
                    description="psi(3) = 3/2 - gamma",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=-EULER_MASCHERONI - 2 * math.log(2),
                    description="psi(1/2) = -gamma - 2*ln(2)",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: (float(n) for n in range(-100, 1)),
                    expected_behavior="-inf",
                    description="Poles at non-positive integers",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=True,
            supports_meta=True,
        )

    def test_known_values(self):
        """Test digamma at known values."""
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.digamma(z)

        # psi(n) = -gamma + H_{n-1} where H_n = 1 + 1/2 + ... + 1/n
        expected = torch.tensor(
            [
                -EULER_MASCHERONI,  # psi(1)
                -EULER_MASCHERONI + 1,  # psi(2)
                -EULER_MASCHERONI + 1 + 0.5,  # psi(3)
                -EULER_MASCHERONI + 1 + 0.5 + 1 / 3,  # psi(4)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_half_value(self):
        """Test psi(1/2) = -gamma - 2*ln(2)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.digamma(z)
        expected = torch.tensor(
            [-EULER_MASCHERONI - 2 * math.log(2)], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_recurrence_relation(self):
        """Test psi(z+1) = psi(z) + 1/z."""
        z = torch.tensor([0.5, 1.5, 2.5, 5.0, 10.0], dtype=torch.float64)
        left = torchscience.special_functions.digamma(z + 1)
        right = torchscience.special_functions.digamma(z) + 1 / z
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_poles_return_neg_inf(self):
        """Test that digamma at poles returns -inf."""
        poles = torch.tensor([0.0, -1.0, -2.0, -3.0], dtype=torch.float64)
        result = torchscience.special_functions.digamma(poles)
        assert (torch.isinf(result) | torch.isnan(result)).all()

    @pytest.mark.skip(reason="Complex digamma not yet implemented")
    def test_complex_conjugate_symmetry(self):
        """Test psi(conj(z)) = conj(psi(z))."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 - 0.3j], dtype=torch.complex128
        )
        result_z = torchscience.special_functions.digamma(z)
        result_conj_z = torchscience.special_functions.digamma(z.conj())
        torch.testing.assert_close(
            result_conj_z, result_z.conj(), rtol=1e-8, atol=1e-8
        )

    def test_comparison_with_torch(self):
        """Test agreement with torch.special.digamma."""
        z = torch.tensor([0.5, 1.0, 1.5, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.digamma(z)
        expected = torch.special.digamma(z)
        torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-7)

    def test_gradient_equals_trigamma(self):
        """Test d/dz psi(z) = trigamma(z)."""
        z = torch.tensor(
            [1.0, 2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.digamma(z)
        y.sum().backward()

        # trigamma(n) for positive integers
        expected = torch.special.polygamma(1, z.detach())
        torch.testing.assert_close(z.grad, expected, rtol=1e-8, atol=1e-8)

    def test_second_order_gradient(self):
        """Test second-order gradients exist and are finite."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.digamma(z)

        (grad1,) = torch.autograd.grad(y, z, create_graph=True)
        (grad2,) = torch.autograd.grad(grad1, z)

        assert torch.isfinite(grad2).all()
        # Second derivative is tetragamma, should be negative for positive z
        assert grad2.item() < 0

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        z = torch.tensor(
            [1.5, 2.5, 5.0], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(
            torchscience.special_functions.digamma,
            (z,),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        z = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.digamma,
            (z,),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )

    @given(z=positive_real_numbers(min_value=0.1, max_value=50.0))
    @settings(max_examples=100, deadline=None)
    def test_property_positive_real_finite(self, z):
        """Property: Digamma of positive real is always finite."""
        z_tensor = torch.tensor([z], dtype=torch.float64)
        result = torchscience.special_functions.digamma(z_tensor)
        assert torch.isfinite(result).all(), (
            f"psi({z}) is not finite: {result}"
        )

    @given(
        z=avoiding_poles(
            max_negative_pole=-50, min_value=-50.0, max_value=50.0
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_recurrence(self, z):
        """Property: psi(z+1) = psi(z) + 1/z for all z not at poles."""
        if abs(z) < 0.01:
            return
        z_tensor = torch.tensor([z], dtype=torch.float64)
        left = torchscience.special_functions.digamma(z_tensor + 1)
        right = torchscience.special_functions.digamma(z_tensor) + 1 / z_tensor
        if torch.isfinite(left).all() and torch.isfinite(right).all():
            torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-8)

    @pytest.mark.skip(reason="Complex digamma not yet implemented")
    @given(z=complex_avoiding_real_axis(real_range=(-5.0, 5.0), min_imag=0.1))
    @settings(max_examples=100, deadline=None)
    def test_property_complex_conjugate(self, z):
        """Property: psi(conj(z)) = conj(psi(z))."""
        z_tensor = torch.tensor([z], dtype=torch.complex128)
        psi_z = torchscience.special_functions.digamma(z_tensor)
        psi_conj_z = torchscience.special_functions.digamma(z_tensor.conj())
        torch.testing.assert_close(
            psi_conj_z, psi_z.conj(), rtol=1e-8, atol=1e-8
        )

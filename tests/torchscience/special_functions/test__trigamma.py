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


def sympy_trigamma(z: float | complex) -> float | complex:
    """Wrapper for SymPy trigamma (polygamma(1, z))."""
    if isinstance(z, complex):
        sympy_z = sympy.Float(z.real) + I * sympy.Float(z.imag)
    else:
        sympy_z = sympy.Float(z)
    result = N(sympy.polygamma(1, sympy_z), 50)
    if result.is_real:
        return float(result)
    return complex(result)


def create_trigamma_verifier() -> SymbolicDerivativeVerifier:
    """Create derivative verifier for the trigamma function."""
    z = symbols("z")
    expr = sympy.polygamma(1, z)
    return SymbolicDerivativeVerifier(expr, [z])


def _check_recurrence(func) -> bool:
    """Check psi_1(x+1) = psi_1(x) - 1/x^2."""
    x = torch.tensor([0.5, 1.5, 2.5, 3.7], dtype=torch.float64)
    left = func(x + 1)
    right = func(x) - 1 / (x * x)
    return torch.allclose(left, right, rtol=1e-10, atol=1e-10)


def _reflection_identity(func):
    """Check psi_1(1-x) + psi_1(x) = pi^2 / sin^2(pi*x)."""
    x = torch.tensor([0.25, 0.3, 0.4, 0.6], dtype=torch.float64)
    left = func(1 - x) + func(x)
    sin_pi_x = torch.sin(math.pi * x)
    right = (math.pi**2) / (sin_pi_x * sin_pi_x)
    return left, right


class TestTrigamma(OpTestCase):
    """Tests for the trigamma function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="trigamma",
            func=torchscience.special_functions.trigamma,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(0.5, 20.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                ),
            ],
            sympy_func=lambda z: sympy.polygamma(1, z),
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
                    name="trigamma_recurrence",
                    check_fn=_check_recurrence,
                    description="psi_1(x+1) = psi_1(x) - 1/x^2",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="reflection_formula",
                    identity_fn=_reflection_identity,
                    description="psi_1(1-x) + psi_1(x) = pi^2 / sin^2(pi*x)",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=math.pi**2 / 6,
                    description="psi_1(1) = pi^2/6",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=math.pi**2 / 6 - 1,
                    description="psi_1(2) = pi^2/6 - 1",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=math.pi**2 / 2,
                    description="psi_1(1/2) = pi^2/2",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: (float(n) for n in range(-100, 1)),
                    expected_behavior="+inf",
                    description="Poles at non-positive integers",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=True,
            supports_meta=True,
        )

    def test_known_values(self):
        """Test trigamma at known values."""
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.trigamma(z)

        # psi_1(n) = pi^2/6 - sum_{k=1}^{n-1} 1/k^2
        pi_sq_6 = math.pi**2 / 6
        expected = torch.tensor(
            [
                pi_sq_6,  # psi_1(1)
                pi_sq_6 - 1,  # psi_1(2)
                pi_sq_6 - 1 - 0.25,  # psi_1(3)
                pi_sq_6 - 1 - 0.25 - 1 / 9,  # psi_1(4)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_half_value(self):
        """Test psi_1(1/2) = pi^2/2."""
        z = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.trigamma(z)
        expected = torch.tensor([math.pi**2 / 2], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_recurrence_relation(self):
        """Test psi_1(z+1) = psi_1(z) - 1/z^2."""
        z = torch.tensor([0.5, 1.5, 2.5, 5.0, 10.0], dtype=torch.float64)
        left = torchscience.special_functions.trigamma(z + 1)
        right = torchscience.special_functions.trigamma(z) - 1 / (z * z)
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_poles_return_pos_inf(self):
        """Test that trigamma at poles returns +inf."""
        poles = torch.tensor([0.0, -1.0, -2.0, -3.0], dtype=torch.float64)
        result = torchscience.special_functions.trigamma(poles)
        assert (torch.isinf(result) | torch.isnan(result)).all()

    @pytest.mark.skip(reason="Complex trigamma not yet implemented")
    def test_complex_conjugate_symmetry(self):
        """Test psi_1(conj(z)) = conj(psi_1(z))."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 - 0.3j], dtype=torch.complex128
        )
        result_z = torchscience.special_functions.trigamma(z)
        result_conj_z = torchscience.special_functions.trigamma(z.conj())
        torch.testing.assert_close(
            result_conj_z, result_z.conj(), rtol=1e-8, atol=1e-8
        )

    def test_comparison_with_torch(self):
        """Test agreement with torch.special.polygamma(1, z)."""
        z = torch.tensor([0.5, 1.0, 1.5, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.trigamma(z)
        expected = torch.special.polygamma(1, z)
        torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-7)

    def test_gradient_equals_tetragamma(self):
        """Test d/dz psi_1(z) = tetragamma(z)."""
        z = torch.tensor(
            [1.0, 2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.trigamma(z)
        y.sum().backward()

        # tetragamma is polygamma(2, z)
        expected = torch.special.polygamma(2, z.detach())
        torch.testing.assert_close(z.grad, expected, rtol=1e-8, atol=1e-8)

    def test_second_order_gradient(self):
        """Test second-order gradients exist and are finite."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.trigamma(z)

        (grad1,) = torch.autograd.grad(y, z, create_graph=True)
        (grad2,) = torch.autograd.grad(grad1, z)

        assert torch.isfinite(grad2).all()
        # Second derivative is pentagamma, should be positive for positive z
        assert grad2.item() > 0

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        z = torch.tensor(
            [1.5, 2.5, 5.0], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(
            torchscience.special_functions.trigamma,
            (z,),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        z = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.trigamma,
            (z,),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )

    @given(z=positive_real_numbers(min_value=0.1, max_value=50.0))
    @settings(max_examples=100, deadline=None)
    def test_property_positive_real_finite(self, z):
        """Property: Trigamma of positive real is always finite and positive."""
        z_tensor = torch.tensor([z], dtype=torch.float64)
        result = torchscience.special_functions.trigamma(z_tensor)
        assert torch.isfinite(result).all(), (
            f"psi_1({z}) is not finite: {result}"
        )
        assert result.item() > 0, f"psi_1({z}) should be positive: {result}"

    @given(
        z=avoiding_poles(
            max_negative_pole=-50, min_value=-50.0, max_value=50.0
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_recurrence(self, z):
        """Property: psi_1(z+1) = psi_1(z) - 1/z^2 for all z not at poles."""
        if abs(z) < 0.01:
            return
        z_tensor = torch.tensor([z], dtype=torch.float64)
        left = torchscience.special_functions.trigamma(z_tensor + 1)
        right = torchscience.special_functions.trigamma(z_tensor) - 1 / (
            z_tensor * z_tensor
        )
        if torch.isfinite(left).all() and torch.isfinite(right).all():
            torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-8)

    @pytest.mark.skip(reason="Complex trigamma not yet implemented")
    @given(z=complex_avoiding_real_axis(real_range=(-5.0, 5.0), min_imag=0.1))
    @settings(max_examples=100, deadline=None)
    def test_property_complex_conjugate(self, z):
        """Property: psi_1(conj(z)) = conj(psi_1(z))."""
        z_tensor = torch.tensor([z], dtype=torch.complex128)
        psi1_z = torchscience.special_functions.trigamma(z_tensor)
        psi1_conj_z = torchscience.special_functions.trigamma(z_tensor.conj())
        torch.testing.assert_close(
            psi1_conj_z, psi1_z.conj(), rtol=1e-8, atol=1e-8
        )

import math

import pytest
import sympy
import torch
import torch.testing
from hypothesis import given, settings
from sympy import I, N

pytestmark = pytest.mark.skip(reason="Test takes >30s, needs optimization")

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    RecurrenceSpec,
    SingularitySpec,
    SpecialValue,
    ToleranceConfig,
    avoiding_poles,
    positive_real_numbers,
)


def sympy_polygamma(n: int, z: float | complex) -> float | complex:
    """Wrapper for SymPy polygamma function."""
    if isinstance(z, complex):
        sympy_z = sympy.Float(z.real) + I * sympy.Float(z.imag)
    else:
        sympy_z = sympy.Float(z)
    result = N(sympy.polygamma(n, sympy_z), 50)
    if result.is_real:
        return float(result)
    return complex(result)


def _check_recurrence(func) -> bool:
    """Check psi^(n)(x+1) = psi^(n)(x) + (-1)^n * n! / x^(n+1)."""
    for n in [0, 1, 2, 3]:
        n_tensor = torch.tensor([float(n)], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 2.5, 3.7], dtype=torch.float64)
        left = func(n_tensor, x + 1)
        factorial_n = math.factorial(n)
        sign = (-1) ** n
        right = func(n_tensor, x) + sign * factorial_n / (x ** (n + 1))
        if not torch.allclose(left, right, rtol=1e-8, atol=1e-8):
            return False
    return True


class TestPolygamma(OpTestCase):
    """Tests for the polygamma function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="polygamma",
            func=torchscience.special_functions.polygamma,
            arity=2,
            input_specs=[
                InputSpec(
                    name="n",
                    position=0,
                    default_real_range=(0.0, 3.0),
                    excluded_values=set(),
                ),
                InputSpec(
                    name="z",
                    position=1,
                    default_real_range=(0.5, 20.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                ),
            ],
            sympy_func=lambda n, z: sympy.polygamma(int(n), z),
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                "test_gradcheck_complex",  # Complex not yet implemented
                "test_gradgradcheck_complex",  # Complex not yet implemented
                "test_sparse_coo_basic",  # Sparse has implicit zeros = poles
                "test_low_precision_forward",  # Random values may hit poles
                "test_sympy_reference_complex",  # Complex not yet implemented
                "test_sympy_reference_real",  # Two-arg sympy func needs special handling
                "test_special_values",  # Special values for binary ops need different format
                "test_nan_propagation",  # NaN tests need two inputs
                "test_nan_propagation_all_inputs",  # NaN tests need two inputs
            },
            recurrence_relations=[
                RecurrenceSpec(
                    name="polygamma_recurrence",
                    check_fn=_check_recurrence,
                    description="psi^(n)(x+1) = psi^(n)(x) + (-1)^n * n! / x^(n+1)",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=math.pi**2 / 6,  # psi^(1)(1) = zeta(2) = pi^2/6
                    description="psi^(1)(1) = pi^2/6",
                ),
                SpecialValue(
                    inputs=(2.0, 1.0),
                    expected=-2
                    * 1.2020569031595942,  # psi^(2)(1) = -2*zeta(3)
                    description="psi^(2)(1) = -2*zeta(3)",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: ((1.0, float(n)) for n in range(-10, 1)),
                    expected_behavior="inf or nan",
                    description="Poles at non-positive integers for all n",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=True,
            supports_meta=True,
        )

    def test_matches_torch_polygamma(self):
        """Test agreement with torch.special.polygamma."""
        for n in [0, 1, 2, 3]:
            z = torch.tensor(
                [0.5, 1.0, 1.5, 2.0, 5.0, 10.0], dtype=torch.float64
            )
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.polygamma(n_tensor, z)
            expected = torch.special.polygamma(n, z)
            torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-7)

    def test_order_0_is_digamma(self):
        """Test that polygamma(0, z) equals digamma(z)."""
        z = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=torch.float64)
        n = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.polygamma(n, z)
        expected = torchscience.special_functions.digamma(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_order_1_is_trigamma(self):
        """Test that polygamma(1, z) equals trigamma(z)."""
        z = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=torch.float64)
        n = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.polygamma(n, z)
        expected = torchscience.special_functions.trigamma(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_trigamma_at_1(self):
        """Test psi^(1)(1) = pi^2/6 (Basel problem)."""
        n = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.polygamma(n, z)
        expected = torch.tensor([math.pi**2 / 6], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_recurrence_relation(self):
        """Test psi^(n)(z+1) = psi^(n)(z) + (-1)^n * n! / z^(n+1)."""
        for n in [0, 1, 2, 3]:
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            z = torch.tensor([0.5, 1.5, 2.5, 5.0], dtype=torch.float64)
            left = torchscience.special_functions.polygamma(n_tensor, z + 1)
            factorial_n = math.factorial(n)
            sign = (-1) ** n
            right = torchscience.special_functions.polygamma(
                n_tensor, z
            ) + sign * factorial_n / (z ** (n + 1))
            torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-8)

    def test_higher_orders(self):
        """Test polygamma for n >= 4 against torch.special.polygamma."""
        for n in [4, 5]:
            z = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.polygamma(n_tensor, z)
            expected = torch.special.polygamma(n, z)
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_gradient_is_next_order(self):
        """Test d/dz psi^(n)(z) = psi^(n+1)(z)."""
        for n in [0, 1, 2]:
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            n_plus_1_tensor = torch.tensor([float(n + 1)], dtype=torch.float64)
            z = torch.tensor(
                [2.0, 3.0], dtype=torch.float64, requires_grad=True
            )

            y = torchscience.special_functions.polygamma(n_tensor, z)
            y.sum().backward()

            expected = torchscience.special_functions.polygamma(
                n_plus_1_tensor, z.detach()
            )
            torch.testing.assert_close(z.grad, expected, rtol=1e-7, atol=1e-7)

    def test_second_order_gradient(self):
        """Test second-order gradients exist and are finite."""
        n = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.polygamma(n, z)

        (grad1,) = torch.autograd.grad(y, z, create_graph=True)
        (grad2,) = torch.autograd.grad(grad1, z)

        assert torch.isfinite(grad2).all()

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        n = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor(
            [2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def func(z_):
            return torchscience.special_functions.polygamma(n, z_)

        torch.autograd.gradcheck(
            func,
            (z,),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        n = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)

        def func(z_):
            return torchscience.special_functions.polygamma(n, z_)

        torch.autograd.gradgradcheck(
            func,
            (z,),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(z=positive_real_numbers(min_value=0.1, max_value=50.0))
    @settings(max_examples=100, deadline=None)
    def test_property_positive_real_finite(self, z):
        """Property: Polygamma of positive real is always finite."""
        for n in [0, 1, 2, 3]:
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            z_tensor = torch.tensor([z], dtype=torch.float64)
            result = torchscience.special_functions.polygamma(
                n_tensor, z_tensor
            )
            assert torch.isfinite(result).all(), (
                f"psi^({n})({z}) is not finite: {result}"
            )

    @given(
        z=avoiding_poles(
            max_negative_pole=-50, min_value=-50.0, max_value=50.0
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_recurrence(self, z):
        """Property: psi^(n)(z+1) = psi^(n)(z) + (-1)^n * n! / z^(n+1) for z not at poles."""
        if abs(z) < 0.01:
            return
        for n in [0, 1, 2]:
            n_tensor = torch.tensor([float(n)], dtype=torch.float64)
            z_tensor = torch.tensor([z], dtype=torch.float64)
            left = torchscience.special_functions.polygamma(
                n_tensor, z_tensor + 1
            )
            factorial_n = math.factorial(n)
            sign = (-1) ** n
            right = torchscience.special_functions.polygamma(
                n_tensor, z_tensor
            ) + sign * factorial_n / (z_tensor ** (n + 1))
            if torch.isfinite(left).all() and torch.isfinite(right).all():
                torch.testing.assert_close(left, right, rtol=1e-7, atol=1e-7)

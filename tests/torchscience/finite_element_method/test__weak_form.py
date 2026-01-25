"""Tests for WeakForm variational formulation class."""

import torch

from torchscience.finite_element_method import WeakForm


class TestWeakFormCreation:
    """Tests for WeakForm instantiation."""

    def test_create_with_bilinear_and_linear_forms(self):
        """Test creating WeakForm with bilinear and linear forms."""

        def bilinear_form(u, v, x):
            return u * v

        def linear_form(v, x):
            return v

        weak_form = WeakForm(
            bilinear_form=bilinear_form,
            linear_form=linear_form,
        )

        assert weak_form.bilinear_form is bilinear_form
        assert weak_form.linear_form is linear_form
        assert weak_form.boundary_form is None

    def test_create_with_all_forms(self):
        """Test creating WeakForm with bilinear, linear, and boundary forms."""

        def bilinear_form(u, v, x):
            return u * v

        def linear_form(v, x):
            return v

        def boundary_form(v, x, n):
            return v * n

        weak_form = WeakForm(
            bilinear_form=bilinear_form,
            linear_form=linear_form,
            boundary_form=boundary_form,
        )

        assert weak_form.bilinear_form is bilinear_form
        assert weak_form.linear_form is linear_form
        assert weak_form.boundary_form is boundary_form


class TestWeakFormOptionalBoundary:
    """Tests for optional boundary_form parameter."""

    def test_boundary_form_is_optional(self):
        """Test that boundary_form defaults to None."""
        weak_form = WeakForm(
            bilinear_form=lambda u, v, x: u * v,
            linear_form=lambda v, x: v,
        )
        assert weak_form.boundary_form is None

    def test_explicit_none_boundary_form(self):
        """Test explicitly setting boundary_form to None."""
        weak_form = WeakForm(
            bilinear_form=lambda u, v, x: u * v,
            linear_form=lambda v, x: v,
            boundary_form=None,
        )
        assert weak_form.boundary_form is None


class TestWeakFormEvaluation:
    """Tests for evaluating weak form callables."""

    def test_bilinear_form_evaluation(self):
        """Test that bilinear form can be called with appropriate arguments."""

        def bilinear_form(u, v, x):
            # Simple L2 inner product: integral of u * v
            return u.value * v.value

        weak_form = WeakForm(
            bilinear_form=bilinear_form,
            linear_form=lambda v, x: v.value,
        )

        # Create simple mock objects with value attribute
        class MockBasisFunction:
            def __init__(self, value):
                self.value = value

        u = MockBasisFunction(torch.tensor([1.0, 2.0, 3.0]))
        v = MockBasisFunction(torch.tensor([0.5, 0.5, 0.5]))
        x = torch.tensor([[0.0], [0.5], [1.0]])

        result = weak_form.bilinear_form(u, v, x)
        expected = torch.tensor([0.5, 1.0, 1.5])
        assert torch.allclose(result, expected)

    def test_linear_form_evaluation(self):
        """Test that linear form can be called with appropriate arguments."""

        def linear_form(v, x):
            # Source term f(x) * v
            return x[:, 0] * v.value

        weak_form = WeakForm(
            bilinear_form=lambda u, v, x: u.value * v.value,
            linear_form=linear_form,
        )

        class MockBasisFunction:
            def __init__(self, value):
                self.value = value

        v = MockBasisFunction(torch.tensor([1.0, 1.0, 1.0]))
        x = torch.tensor([[0.0], [0.5], [1.0]])

        result = weak_form.linear_form(v, x)
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(result, expected)

    def test_boundary_form_evaluation(self):
        """Test that boundary form can be called with appropriate arguments."""

        def boundary_form(v, x, n):
            # Neumann boundary: g(x) * v where g = 1
            return v.value * n[:, 0]

        weak_form = WeakForm(
            bilinear_form=lambda u, v, x: u.value * v.value,
            linear_form=lambda v, x: v.value,
            boundary_form=boundary_form,
        )

        class MockBasisFunction:
            def __init__(self, value):
                self.value = value

        v = MockBasisFunction(torch.tensor([1.0, 2.0]))
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        n = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # outward normal

        result = weak_form.boundary_form(v, x, n)
        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(result, expected)


class TestPoissonWeakForm:
    """Tests for Poisson problem weak formulation."""

    def test_poisson_weak_form_creation(self):
        """Test creating a Poisson weak form: -nabla^2 u = f.

        The weak form is:
        - Bilinear: integral of grad(u) dot grad(v)
        - Linear: integral of f * v
        """

        def poisson_bilinear(u, v, x):
            # Stiffness form: grad(u) dot grad(v)
            return (u.grad * v.grad).sum(dim=-1)

        def poisson_linear(v, x):
            # Source term: f(x) * v where f = 1 (constant)
            return v.value

        poisson = WeakForm(
            bilinear_form=poisson_bilinear,
            linear_form=poisson_linear,
        )

        assert callable(poisson.bilinear_form)
        assert callable(poisson.linear_form)

    def test_poisson_weak_form_with_variable_source(self):
        """Test Poisson with spatially varying source f(x) = sin(pi*x)."""

        def poisson_bilinear(u, v, x):
            return (u.grad * v.grad).sum(dim=-1)

        def poisson_linear(v, x):
            import math

            # f(x) = sin(pi * x) in 1D
            return torch.sin(math.pi * x[:, 0]) * v.value

        poisson = WeakForm(
            bilinear_form=poisson_bilinear,
            linear_form=poisson_linear,
        )

        # Create mock objects for evaluation
        class MockBasisFunction:
            def __init__(self, value, grad):
                self.value = value
                self.grad = grad

        # Test at quadrature points
        x = torch.tensor([[0.25], [0.5], [0.75]])
        v = MockBasisFunction(
            value=torch.tensor([1.0, 1.0, 1.0]),
            grad=torch.tensor([[1.0], [0.0], [-1.0]]),
        )

        result = poisson.linear_form(v, x)
        import math

        expected = torch.sin(math.pi * x[:, 0])
        assert torch.allclose(result, expected)


class TestHeatEquationWeakForm:
    """Tests for heat equation weak form."""

    def test_heat_equation_weak_form(self):
        """Test heat equation: du/dt - k * nabla^2 u = 0.

        Weak form bilinear has two terms:
        - Mass term: integral of u * v (for time derivative)
        - Stiffness term: k * integral of grad(u) dot grad(v)
        """

        k = 0.1  # thermal diffusivity

        def heat_bilinear(u, v, x):
            # Combined mass + stiffness
            mass = u.value * v.value
            stiffness = k * (u.grad * v.grad).sum(dim=-1)
            return mass + stiffness

        def heat_linear(v, x):
            # No source term
            return torch.zeros_like(v.value)

        heat_weak_form = WeakForm(
            bilinear_form=heat_bilinear,
            linear_form=heat_linear,
        )

        assert callable(heat_weak_form.bilinear_form)
        assert callable(heat_weak_form.linear_form)


class TestElasticityWeakForm:
    """Tests for elasticity weak form (vector-valued problem)."""

    def test_elasticity_weak_form_creation(self):
        """Test creating a linear elasticity weak form.

        For linear elasticity:
        - Bilinear: integral of sigma(u) : epsilon(v)
        - Linear: integral of f dot v (body force)
        """

        def elasticity_bilinear(u, v, x):
            # Simplified: just check it returns a tensor
            # In practice, this would compute strain tensors
            return u.value.sum(dim=-1) * v.value.sum(dim=-1)

        def elasticity_linear(v, x):
            # Body force f = (0, -rho*g)
            rho_g = 9.81
            return -rho_g * v.value[..., 1]

        elasticity = WeakForm(
            bilinear_form=elasticity_bilinear,
            linear_form=elasticity_linear,
        )

        assert callable(elasticity.bilinear_form)
        assert callable(elasticity.linear_form)


class TestRobinBoundaryCondition:
    """Tests for Robin boundary condition in weak form."""

    def test_robin_boundary_form(self):
        """Test Robin BC: du/dn + alpha * u = g on boundary.

        The boundary integral adds:
        - To bilinear form: alpha * integral_boundary of u * v
        - To linear form: integral_boundary of g * v
        """

        alpha = 1.0

        def robin_boundary(v, x, n):
            # Boundary contribution: alpha * u * v + g * v
            # Here we just return the test function contribution
            g = torch.ones_like(x[:, 0])  # g = 1
            return g * v.value

        weak_form = WeakForm(
            bilinear_form=lambda u, v, x: (u.grad * v.grad).sum(dim=-1),
            linear_form=lambda v, x: v.value,
            boundary_form=robin_boundary,
        )

        assert weak_form.boundary_form is not None

        class MockBasisFunction:
            def __init__(self, value):
                self.value = value

        v = MockBasisFunction(torch.tensor([1.0, 1.0]))
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        n = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        result = weak_form.boundary_form(v, x, n)
        expected = torch.tensor([1.0, 1.0])
        assert torch.allclose(result, expected)

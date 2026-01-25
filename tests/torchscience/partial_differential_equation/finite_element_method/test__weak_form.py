"""Tests for WeakForm variational formulation class."""

import torch

from torchscience.partial_differential_equation.finite_element_method import (
    WeakForm,
)


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


class TestAssembleWeakForm:
    """Tests for assemble_weak_form function."""

    def test_assemble_poisson_weak_form_equals_stiffness(self):
        """Poisson weak form should produce same matrix as local_stiffness_matrices."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            WeakForm,
            assemble_matrix,
            assemble_weak_form,
            dof_map,
            local_stiffness_matrices,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        # Define Poisson weak form: a(u, v) = integral(grad(u) dot grad(v))
        def poisson_bilinear(u, v, x):
            # u.grad and v.grad have shape (num_quad, num_dofs, dim)
            # We want to compute grad(u) dot grad(v) for all (i, j) combinations
            # For trial i and test j: grad_u[i] dot grad_v[j]
            return (u.grad * v.grad).sum(dim=-1)

        def poisson_linear(v, x):
            return torch.zeros_like(v.value)

        weak_form = WeakForm(
            bilinear_form=poisson_bilinear,
            linear_form=poisson_linear,
        )

        # Assemble using weak form
        K_weak, f_weak = assemble_weak_form(mesh, dm, weak_form)

        # Assemble using direct method
        K_local = local_stiffness_matrices(mesh, dm)
        K_direct = assemble_matrix(K_local, dm)

        # Convert both to dense for comparison
        K_weak_dense = K_weak.to_dense()
        K_direct_dense = K_direct.to_dense()

        assert torch.allclose(
            K_weak_dense, K_direct_dense, rtol=1e-10, atol=1e-12
        )

    def test_assemble_mass_weak_form_equals_mass_matrices(self):
        """Mass weak form should produce same matrix as local_mass_matrices."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            WeakForm,
            assemble_matrix,
            assemble_weak_form,
            dof_map,
            local_mass_matrices,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        # Define mass weak form: a(u, v) = integral(u * v)
        def mass_bilinear(u, v, x):
            # u.value and v.value have shape (num_quad, num_dofs)
            return u.value * v.value

        def mass_linear(v, x):
            return torch.zeros_like(v.value)

        weak_form = WeakForm(
            bilinear_form=mass_bilinear,
            linear_form=mass_linear,
        )

        # Assemble using weak form
        M_weak, f_weak = assemble_weak_form(mesh, dm, weak_form)

        # Assemble using direct method
        M_local = local_mass_matrices(mesh, dm)
        M_direct = assemble_matrix(M_local, dm)

        # Convert both to dense for comparison
        M_weak_dense = M_weak.to_dense()
        M_direct_dense = M_direct.to_dense()

        assert torch.allclose(
            M_weak_dense, M_direct_dense, rtol=1e-10, atol=1e-12
        )

    def test_assemble_custom_bilinear_form(self):
        """Test custom bilinear form that combines mass and stiffness."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            WeakForm,
            assemble_matrix,
            assemble_weak_form,
            dof_map,
            local_mass_matrices,
            local_stiffness_matrices,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        alpha = 2.5  # mass coefficient
        beta = 0.3  # stiffness coefficient

        # Define combined weak form: a(u, v) = alpha * u * v + beta * grad(u) dot grad(v)
        def combined_bilinear(u, v, x):
            mass_term = alpha * u.value * v.value
            stiffness_term = beta * (u.grad * v.grad).sum(dim=-1)
            return mass_term + stiffness_term

        def combined_linear(v, x):
            return torch.zeros_like(v.value)

        weak_form = WeakForm(
            bilinear_form=combined_bilinear,
            linear_form=combined_linear,
        )

        # Assemble using weak form
        A_weak, _ = assemble_weak_form(mesh, dm, weak_form)

        # Assemble using direct method
        M_local = local_mass_matrices(mesh, dm)
        K_local = local_stiffness_matrices(mesh, dm)
        A_local = alpha * M_local + beta * K_local
        A_direct = assemble_matrix(A_local, dm)

        A_weak_dense = A_weak.to_dense()
        A_direct_dense = A_direct.to_dense()

        assert torch.allclose(
            A_weak_dense, A_direct_dense, rtol=1e-10, atol=1e-12
        )

    def test_symmetric_bilinear_form_produces_symmetric_matrix(self):
        """Symmetric weak form should produce symmetric matrix."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            WeakForm,
            assemble_weak_form,
            dof_map,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        # Stiffness form is symmetric
        def stiffness_bilinear(u, v, x):
            return (u.grad * v.grad).sum(dim=-1)

        def zero_linear(v, x):
            return torch.zeros_like(v.value)

        weak_form = WeakForm(
            bilinear_form=stiffness_bilinear,
            linear_form=zero_linear,
        )

        K, _ = assemble_weak_form(mesh, dm, weak_form)
        K_dense = K.to_dense()

        # Check symmetry
        assert torch.allclose(K_dense, K_dense.T, rtol=1e-10, atol=1e-12)

    def test_sparse_csr_output(self):
        """Output matrix should be in sparse CSR format."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            WeakForm,
            assemble_weak_form,
            dof_map,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        def bilinear(u, v, x):
            return (u.grad * v.grad).sum(dim=-1)

        def linear(v, x):
            return v.value

        weak_form = WeakForm(
            bilinear_form=bilinear,
            linear_form=linear,
        )

        K, f = assemble_weak_form(mesh, dm, weak_form)

        # Check matrix is sparse CSR
        assert K.is_sparse_csr

        # Check vector is dense
        assert not f.is_sparse

        # Check shapes
        assert K.shape == (dm.num_global_dofs, dm.num_global_dofs)
        assert f.shape == (dm.num_global_dofs,)

    def test_linear_form_with_constant_source(self):
        """Test linear form with constant source f = 1."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            WeakForm,
            assemble_vector,
            assemble_weak_form,
            dof_map,
            local_mass_matrices,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        # f(v) = integral(1 * v) = M @ ones
        def zero_bilinear(u, v, x):
            return torch.zeros_like(u.value)

        def constant_linear(v, x):
            # f(x) = 1, so L(v) = 1 * v
            return v.value

        weak_form = WeakForm(
            bilinear_form=zero_bilinear,
            linear_form=constant_linear,
        )

        _, f_weak = assemble_weak_form(mesh, dm, weak_form)

        # Using mass matrix: f = M @ ones
        M_local = local_mass_matrices(mesh, dm)
        ones_local = torch.ones(
            mesh.num_elements, dm.dofs_per_element, dtype=mesh.vertices.dtype
        )
        f_direct = assemble_vector(ones_local * M_local.sum(dim=-1), dm)

        # Alternative: sum of local mass row sums
        # Actually: integral(v_i) = sum_j M_ij which is M @ ones
        # But our assembly needs local vectors, so let's compute:
        # integral(v_i) on element e = sum_j M_e[i,j]
        f_local = M_local.sum(dim=-1)  # (num_elements, dofs_per_element)
        f_direct = assemble_vector(f_local, dm)

        assert torch.allclose(f_weak, f_direct, rtol=1e-10, atol=1e-12)

    def test_assemble_with_quad_order_parameter(self):
        """Test that quad_order parameter is respected."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            WeakForm,
            assemble_weak_form,
            dof_map,
        )

        mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        def bilinear(u, v, x):
            return (u.grad * v.grad).sum(dim=-1)

        def linear(v, x):
            return v.value

        weak_form = WeakForm(
            bilinear_form=bilinear,
            linear_form=linear,
        )

        # Different quadrature orders should work
        K1, f1 = assemble_weak_form(mesh, dm, weak_form, quad_order=2)
        K2, f2 = assemble_weak_form(mesh, dm, weak_form, quad_order=4)

        # For linear elements with linear source, results should be close
        assert K1.to_dense().shape == K2.to_dense().shape
        assert f1.shape == f2.shape

    def test_assemble_with_higher_order_elements(self):
        """Test weak form assembly with P2 elements."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            WeakForm,
            assemble_matrix,
            assemble_weak_form,
            dof_map,
            local_stiffness_matrices,
        )

        mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=2)  # P2 elements

        def poisson_bilinear(u, v, x):
            return (u.grad * v.grad).sum(dim=-1)

        def poisson_linear(v, x):
            return torch.zeros_like(v.value)

        weak_form = WeakForm(
            bilinear_form=poisson_bilinear,
            linear_form=poisson_linear,
        )

        # Assemble using weak form
        K_weak, _ = assemble_weak_form(mesh, dm, weak_form)

        # Assemble using direct method
        K_local = local_stiffness_matrices(mesh, dm)
        K_direct = assemble_matrix(K_local, dm)

        K_weak_dense = K_weak.to_dense()
        K_direct_dense = K_direct.to_dense()

        assert torch.allclose(
            K_weak_dense, K_direct_dense, rtol=1e-10, atol=1e-12
        )


class TestBasisValues:
    """Tests for BasisValues helper class."""

    def test_basis_values_creation(self):
        """Test creating BasisValues with value and grad."""
        from torchscience.partial_differential_equation.finite_element_method import (
            BasisValues,
        )

        value = torch.tensor([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
        grad = torch.randn(2, 3, 2)

        bv = BasisValues(value=value, grad=grad)

        assert torch.equal(bv.value, value)
        assert torch.equal(bv.grad, grad)

    def test_basis_values_attributes(self):
        """Test that BasisValues has expected attributes."""
        from torchscience.partial_differential_equation.finite_element_method import (
            BasisValues,
        )

        value = torch.ones(4, 3)
        grad = torch.ones(4, 3, 2)

        bv = BasisValues(value=value, grad=grad)

        assert hasattr(bv, "value")
        assert hasattr(bv, "grad")
        assert bv.value.shape == (4, 3)
        assert bv.grad.shape == (4, 3, 2)


class TestPoissonForm:
    """Tests for poisson_form helper function."""

    def test_poisson_form_default_diffusivity(self):
        """Test poisson_form with default diffusivity=1.0."""
        from torchscience.partial_differential_equation.finite_element_method import (
            poisson_form,
        )

        wf = poisson_form()
        assert callable(wf.bilinear_form)
        assert callable(wf.linear_form)
        assert wf.boundary_form is None

    def test_poisson_form_custom_diffusivity(self):
        """Test poisson_form with custom diffusivity."""
        from torchscience.partial_differential_equation.finite_element_method import (
            poisson_form,
        )

        kappa = 2.5
        wf = poisson_form(diffusivity=kappa)
        assert callable(wf.bilinear_form)

    def test_poisson_form_tensor_diffusivity(self):
        """Test poisson_form with tensor diffusivity."""
        from torchscience.partial_differential_equation.finite_element_method import (
            poisson_form,
        )

        kappa = torch.tensor(3.0)
        wf = poisson_form(diffusivity=kappa)
        assert callable(wf.bilinear_form)

    def test_poisson_form_matches_local_stiffness(self):
        """Verify poisson_form produces same matrix as local_stiffness_matrices."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            assemble_matrix,
            assemble_weak_form,
            dof_map,
            local_stiffness_matrices,
            poisson_form,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        # Using poisson_form helper
        wf = poisson_form(diffusivity=1.0)
        K_weak, _ = assemble_weak_form(mesh, dm, wf)

        # Using direct method
        K_local = local_stiffness_matrices(mesh, dm, material=1.0)
        K_direct = assemble_matrix(K_local, dm)

        K_weak_dense = K_weak.to_dense()
        K_direct_dense = K_direct.to_dense()

        assert torch.allclose(
            K_weak_dense, K_direct_dense, rtol=1e-10, atol=1e-12
        )

    def test_poisson_form_with_custom_diffusivity_matches(self):
        """Verify poisson_form with custom diffusivity matches local_stiffness_matrices."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            assemble_matrix,
            assemble_weak_form,
            dof_map,
            local_stiffness_matrices,
            poisson_form,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)
        kappa = 2.5

        # Using poisson_form helper
        wf = poisson_form(diffusivity=kappa)
        K_weak, _ = assemble_weak_form(mesh, dm, wf)

        # Using direct method
        K_local = local_stiffness_matrices(mesh, dm, material=kappa)
        K_direct = assemble_matrix(K_local, dm)

        K_weak_dense = K_weak.to_dense()
        K_direct_dense = K_direct.to_dense()

        assert torch.allclose(
            K_weak_dense, K_direct_dense, rtol=1e-10, atol=1e-12
        )

    def test_poisson_form_linear_form_returns_zeros(self):
        """Verify poisson_form linear_form returns zeros."""
        from torchscience.partial_differential_equation.finite_element_method import (
            BasisValues,
            poisson_form,
        )

        wf = poisson_form()

        # Create mock basis values
        v = BasisValues(
            value=torch.ones(4, 3),
            grad=torch.randn(4, 3, 2),
        )
        x = torch.randn(4, 3, 2)

        result = wf.linear_form(v, x)
        assert torch.allclose(result, torch.zeros_like(v.value))


class TestMassForm:
    """Tests for mass_form helper function."""

    def test_mass_form_default_density(self):
        """Test mass_form with default density=1.0."""
        from torchscience.partial_differential_equation.finite_element_method import (
            mass_form,
        )

        wf = mass_form()
        assert callable(wf.bilinear_form)
        assert callable(wf.linear_form)
        assert wf.boundary_form is None

    def test_mass_form_custom_density(self):
        """Test mass_form with custom density."""
        from torchscience.partial_differential_equation.finite_element_method import (
            mass_form,
        )

        rho = 7.85  # steel density
        wf = mass_form(density=rho)
        assert callable(wf.bilinear_form)

    def test_mass_form_tensor_density(self):
        """Test mass_form with tensor density."""
        from torchscience.partial_differential_equation.finite_element_method import (
            mass_form,
        )

        rho = torch.tensor(2.7)  # aluminum density
        wf = mass_form(density=rho)
        assert callable(wf.bilinear_form)

    def test_mass_form_matches_local_mass(self):
        """Verify mass_form produces same matrix as local_mass_matrices."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            assemble_matrix,
            assemble_weak_form,
            dof_map,
            local_mass_matrices,
            mass_form,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)

        # Using mass_form helper
        wf = mass_form(density=1.0)
        M_weak, _ = assemble_weak_form(mesh, dm, wf)

        # Using direct method
        M_local = local_mass_matrices(mesh, dm, density=1.0)
        M_direct = assemble_matrix(M_local, dm)

        M_weak_dense = M_weak.to_dense()
        M_direct_dense = M_direct.to_dense()

        assert torch.allclose(
            M_weak_dense, M_direct_dense, rtol=1e-10, atol=1e-12
        )

    def test_mass_form_with_custom_density_matches(self):
        """Verify mass_form with custom density matches local_mass_matrices."""
        from torchscience.geometry.mesh import rectangle_mesh
        from torchscience.partial_differential_equation.finite_element_method import (
            assemble_matrix,
            assemble_weak_form,
            dof_map,
            local_mass_matrices,
            mass_form,
        )

        mesh = rectangle_mesh(3, 3, bounds=[[0.0, 1.0], [0.0, 1.0]])
        dm = dof_map(mesh, order=1)
        rho = 7.85

        # Using mass_form helper
        wf = mass_form(density=rho)
        M_weak, _ = assemble_weak_form(mesh, dm, wf)

        # Using direct method
        M_local = local_mass_matrices(mesh, dm, density=rho)
        M_direct = assemble_matrix(M_local, dm)

        M_weak_dense = M_weak.to_dense()
        M_direct_dense = M_direct.to_dense()

        assert torch.allclose(
            M_weak_dense, M_direct_dense, rtol=1e-10, atol=1e-12
        )

    def test_mass_form_linear_form_returns_zeros(self):
        """Verify mass_form linear_form returns zeros."""
        from torchscience.partial_differential_equation.finite_element_method import (
            BasisValues,
            mass_form,
        )

        wf = mass_form()

        # Create mock basis values
        v = BasisValues(
            value=torch.ones(4, 3),
            grad=torch.randn(4, 3, 2),
        )
        x = torch.randn(4, 3, 2)

        result = wf.linear_form(v, x)
        assert torch.allclose(result, torch.zeros_like(v.value))

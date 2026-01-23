"""Tests for meta tensor support in differentiation operators."""

import torch

from torchscience.differentiation import (
    biharmonic,
    curl,
    derivative,
    divergence,
    gradient,
    hessian,
    jacobian,
    laplacian,
)


class TestMetaTensorShapes:
    """Tests for meta tensor shape inference."""

    def test_derivative_meta_shape(self):
        """Derivative preserves shape on meta tensors."""
        field = torch.empty(16, 16, device="meta")
        deriv = derivative(field, dim=-1, dx=0.1)
        assert deriv.shape == (16, 16)
        assert deriv.device.type == "meta"

    def test_gradient_meta_shape(self):
        """Gradient adds component dimension on meta tensors."""
        field = torch.empty(16, 16, device="meta")
        grad = gradient(field, dx=0.1)
        assert grad.shape == (2, 16, 16)
        assert grad.device.type == "meta"

    def test_laplacian_meta_shape(self):
        """Laplacian preserves shape on meta tensors."""
        field = torch.empty(16, 16, device="meta")
        lap = laplacian(field, dx=0.1)
        assert lap.shape == (16, 16)
        assert lap.device.type == "meta"

    def test_hessian_meta_shape(self):
        """Hessian adds matrix dimensions on meta tensors."""
        field = torch.empty(16, 16, device="meta")
        hess = hessian(field, dx=0.1)
        assert hess.shape == (2, 2, 16, 16)
        assert hess.device.type == "meta"

    def test_biharmonic_meta_shape(self):
        """Biharmonic preserves shape on meta tensors."""
        field = torch.empty(16, 16, device="meta")
        biharm = biharmonic(field, dx=0.1)
        assert biharm.shape == (16, 16)
        assert biharm.device.type == "meta"

    def test_divergence_meta_shape(self):
        """Divergence removes component dimension on meta tensors."""
        field = torch.empty(2, 16, 16, device="meta")
        div = divergence(field, dx=0.1)
        assert div.shape == (16, 16)
        assert div.device.type == "meta"

    def test_curl_meta_shape(self):
        """Curl preserves shape on meta tensors."""
        field = torch.empty(3, 8, 8, 8, device="meta")
        c = curl(field, dx=0.1)
        assert c.shape == (3, 8, 8, 8)
        assert c.device.type == "meta"

    def test_jacobian_meta_shape(self):
        """Jacobian adds derivative dimension on meta tensors."""
        field = torch.empty(2, 16, 16, device="meta")
        jac = jacobian(field, dx=0.1)
        assert jac.shape == (2, 2, 16, 16)
        assert jac.device.type == "meta"


class TestMetaTensor3D:
    """Tests for 3D meta tensors."""

    def test_gradient_3d_meta(self):
        """3D gradient adds 3 components."""
        field = torch.empty(10, 10, 10, device="meta")
        grad = gradient(field, dx=0.1)
        assert grad.shape == (3, 10, 10, 10)
        assert grad.device.type == "meta"

    def test_laplacian_3d_meta(self):
        """3D laplacian preserves shape."""
        field = torch.empty(10, 10, 10, device="meta")
        lap = laplacian(field, dx=0.1)
        assert lap.shape == (10, 10, 10)
        assert lap.device.type == "meta"

    def test_hessian_3d_meta(self):
        """3D hessian is 3x3 matrix."""
        field = torch.empty(10, 10, 10, device="meta")
        hess = hessian(field, dx=0.1)
        assert hess.shape == (3, 3, 10, 10, 10)
        assert hess.device.type == "meta"

    def test_biharmonic_3d_meta(self):
        """3D biharmonic preserves shape."""
        field = torch.empty(10, 10, 10, device="meta")
        biharm = biharmonic(field, dx=0.1)
        assert biharm.shape == (10, 10, 10)
        assert biharm.device.type == "meta"

    def test_derivative_3d_meta(self):
        """3D derivative preserves shape."""
        field = torch.empty(10, 10, 10, device="meta")
        deriv = derivative(field, dim=0, dx=0.1)
        assert deriv.shape == (10, 10, 10)
        assert deriv.device.type == "meta"

    def test_divergence_3d_meta(self):
        """3D divergence removes component dimension."""
        field = torch.empty(3, 10, 10, 10, device="meta")
        div = divergence(field, dx=0.1)
        assert div.shape == (10, 10, 10)
        assert div.device.type == "meta"

    def test_jacobian_3d_meta(self):
        """3D jacobian has correct shape."""
        field = torch.empty(3, 10, 10, 10, device="meta")
        jac = jacobian(field, dx=0.1)
        assert jac.shape == (3, 3, 10, 10, 10)
        assert jac.device.type == "meta"


class TestMetaTensor1D:
    """Tests for 1D meta tensors."""

    def test_derivative_1d_meta(self):
        """1D derivative preserves shape."""
        field = torch.empty(100, device="meta")
        deriv = derivative(field, dim=0, dx=0.1)
        assert deriv.shape == (100,)
        assert deriv.device.type == "meta"

    def test_gradient_1d_meta(self):
        """1D gradient adds 1 component."""
        field = torch.empty(100, device="meta")
        grad = gradient(field, dx=0.1)
        assert grad.shape == (1, 100)
        assert grad.device.type == "meta"

    def test_laplacian_1d_meta(self):
        """1D laplacian preserves shape."""
        field = torch.empty(100, device="meta")
        lap = laplacian(field, dx=0.1)
        assert lap.shape == (100,)
        assert lap.device.type == "meta"

    def test_hessian_1d_meta(self):
        """1D hessian is 1x1 matrix."""
        field = torch.empty(100, device="meta")
        hess = hessian(field, dx=0.1)
        assert hess.shape == (1, 1, 100)
        assert hess.device.type == "meta"


class TestMetaTensorDtypes:
    """Tests for different dtypes with meta tensors."""

    def test_derivative_float32_meta(self):
        """Derivative works with float32 meta tensors."""
        field = torch.empty(16, 16, dtype=torch.float32, device="meta")
        deriv = derivative(field, dim=-1, dx=0.1)
        assert deriv.dtype == torch.float32
        assert deriv.device.type == "meta"

    def test_derivative_float64_meta(self):
        """Derivative works with float64 meta tensors."""
        field = torch.empty(16, 16, dtype=torch.float64, device="meta")
        deriv = derivative(field, dim=-1, dx=0.1)
        assert deriv.dtype == torch.float64
        assert deriv.device.type == "meta"

    def test_gradient_float32_meta(self):
        """Gradient works with float32 meta tensors."""
        field = torch.empty(16, 16, dtype=torch.float32, device="meta")
        grad = gradient(field, dx=0.1)
        assert grad.dtype == torch.float32
        assert grad.device.type == "meta"

    def test_laplacian_float32_meta(self):
        """Laplacian works with float32 meta tensors."""
        field = torch.empty(16, 16, dtype=torch.float32, device="meta")
        lap = laplacian(field, dx=0.1)
        assert lap.dtype == torch.float32
        assert lap.device.type == "meta"


class TestMetaTensorHigherOrder:
    """Tests for higher-order derivatives with meta tensors."""

    def test_second_derivative_meta(self):
        """Second derivative preserves shape on meta tensors."""
        field = torch.empty(16, 16, device="meta")
        deriv = derivative(field, dim=-1, order=2, dx=0.1)
        assert deriv.shape == (16, 16)
        assert deriv.device.type == "meta"

    def test_third_derivative_meta(self):
        """Third derivative preserves shape on meta tensors."""
        field = torch.empty(32, device="meta")
        deriv = derivative(field, dim=0, order=3, dx=0.1, accuracy=4)
        assert deriv.shape == (32,)
        assert deriv.device.type == "meta"

    def test_fourth_derivative_meta(self):
        """Fourth derivative preserves shape on meta tensors."""
        field = torch.empty(32, device="meta")
        deriv = derivative(field, dim=0, order=4, dx=0.1, accuracy=4)
        assert deriv.shape == (32,)
        assert deriv.device.type == "meta"


class TestMetaTensorBoundaryModes:
    """Tests for different boundary modes with meta tensors."""

    def test_derivative_replicate_boundary_meta(self):
        """Derivative with replicate boundary preserves shape."""
        field = torch.empty(16, 16, device="meta")
        deriv = derivative(field, dim=-1, dx=0.1, boundary="replicate")
        assert deriv.shape == (16, 16)
        assert deriv.device.type == "meta"

    def test_derivative_zeros_boundary_meta(self):
        """Derivative with zeros boundary preserves shape."""
        field = torch.empty(16, 16, device="meta")
        deriv = derivative(field, dim=-1, dx=0.1, boundary="zeros")
        assert deriv.shape == (16, 16)
        assert deriv.device.type == "meta"

    def test_derivative_reflect_boundary_meta(self):
        """Derivative with reflect boundary preserves shape."""
        field = torch.empty(16, 16, device="meta")
        deriv = derivative(field, dim=-1, dx=0.1, boundary="reflect")
        assert deriv.shape == (16, 16)
        assert deriv.device.type == "meta"

    def test_derivative_circular_boundary_meta(self):
        """Derivative with circular boundary preserves shape."""
        field = torch.empty(16, 16, device="meta")
        deriv = derivative(field, dim=-1, dx=0.1, boundary="circular")
        assert deriv.shape == (16, 16)
        assert deriv.device.type == "meta"

    def test_gradient_replicate_boundary_meta(self):
        """Gradient with replicate boundary has correct shape."""
        field = torch.empty(16, 16, device="meta")
        grad = gradient(field, dx=0.1, boundary="replicate")
        assert grad.shape == (2, 16, 16)
        assert grad.device.type == "meta"

    def test_laplacian_zeros_boundary_meta(self):
        """Laplacian with zeros boundary preserves shape."""
        field = torch.empty(16, 16, device="meta")
        lap = laplacian(field, dx=0.1, boundary="zeros")
        assert lap.shape == (16, 16)
        assert lap.device.type == "meta"

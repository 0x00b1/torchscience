import pytest

from torchscience.quadrature import (
    IntegrationError,
    QuadratureWarning,
)


class TestExceptions:
    def test_quadrature_warning_is_user_warning(self):
        assert issubclass(QuadratureWarning, UserWarning)

    def test_integration_error_is_exception(self):
        assert issubclass(IntegrationError, Exception)

    def test_quadrature_warning_can_be_raised(self):
        with pytest.warns(QuadratureWarning, match="test"):
            import warnings

            warnings.warn("test", QuadratureWarning)

    def test_integration_error_can_be_raised(self):
        with pytest.raises(IntegrationError, match="failed"):
            raise IntegrationError("integration failed")

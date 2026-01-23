# tests/torchscience/root_finding/test__exceptions.py
import pytest

from torchscience.root_finding._exceptions import (
    BracketError,
    DerivativeError,
    RootFindingError,
)


class TestExceptions:
    """Tests for root finding exceptions."""

    def test_root_finding_error_is_exception(self):
        """RootFindingError is a base Exception."""
        assert issubclass(RootFindingError, Exception)

    def test_bracket_error_inherits_from_root_finding_error(self):
        """BracketError inherits from RootFindingError."""
        assert issubclass(BracketError, RootFindingError)

    def test_derivative_error_inherits_from_root_finding_error(self):
        """DerivativeError inherits from RootFindingError."""
        assert issubclass(DerivativeError, RootFindingError)

    def test_bracket_error_can_be_raised(self):
        """BracketError can be raised with a message."""
        with pytest.raises(BracketError, match="test message"):
            raise BracketError("test message")

    def test_derivative_error_can_be_raised(self):
        """DerivativeError can be raised with a message."""
        with pytest.raises(DerivativeError, match="zero derivative"):
            raise DerivativeError("zero derivative")

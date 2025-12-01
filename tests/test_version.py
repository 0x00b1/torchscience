"""Test package version."""

import torchscience


def test_version_exists():
    """Verify package has version attribute."""
    assert hasattr(torchscience, "__version__")


def test_version_format():
    """Verify version is a string in semver format."""
    version = torchscience.__version__
    assert isinstance(version, str)
    parts = version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)

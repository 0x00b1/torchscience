from typing import TYPE_CHECKING

import pytest
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class IdentityMixin:
    """Mixin providing functional identity tests."""

    descriptor: "OperatorDescriptor"

    def test_functional_identities(self):
        """Test all configured functional identities."""
        if "test_functional_identities" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for spec in self.descriptor.functional_identities:
            left, right = spec.identity_fn(self.descriptor.func)
            (
                torch.testing.assert_close(
                    left,
                    right,
                    rtol=spec.rtol,
                    atol=spec.atol,
                ),
                f"Identity '{spec.name}' failed: {spec.description}",
            )

from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class VmapMixin:
    """Mixin providing vmap tests."""

    descriptor: "OperatorDescriptor"

    def test_vmap_over_batch(self):
        """Test vmap over batch dimension."""
        if "test_vmap_over_batch" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(shape=(5, 3))

        # vmap over first dimension
        vmapped = torch.vmap(
            self.descriptor.func, in_dims=tuple(0 for _ in inputs)
        )
        result = vmapped(*inputs)

        # Compare with manual loop
        expected = torch.stack(
            [self.descriptor.func(*[t[i] for t in inputs]) for i in range(5)]
        )

        rtol, atol = self.descriptor.tolerances.get_tolerances(inputs[0].dtype)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

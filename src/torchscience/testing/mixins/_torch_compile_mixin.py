from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class TorchCompileMixin:
    """Mixin providing torch.compile tests."""

    descriptor: "OperatorDescriptor"

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_smoke(self):
        """Smoke test for torch.compile compatibility."""
        if "test_compile_smoke" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        @torch.compile
        def compiled_func(*args):
            return self.descriptor.func(*args)

        inputs = self._make_standard_inputs()
        result = compiled_func(*inputs)
        expected = self.descriptor.func(*inputs)

        rtol, atol = self.descriptor.tolerances.get_tolerances(inputs[0].dtype)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

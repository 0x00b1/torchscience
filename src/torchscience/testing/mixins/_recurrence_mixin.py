from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class RecurrenceMixin:
    """Mixin providing recurrence relation tests."""

    descriptor: "OperatorDescriptor"

    def test_recurrence_relations(self):
        """Test all configured recurrence relations."""
        if "test_recurrence_relations" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for spec in self.descriptor.recurrence_relations:
            assert spec.check_fn(self.descriptor.func), (
                f"Recurrence '{spec.name}' failed: {spec.description}"
            )

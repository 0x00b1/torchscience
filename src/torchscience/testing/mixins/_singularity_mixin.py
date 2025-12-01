from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class SingularityMixin:
    """Mixin providing singularity behavior tests for n-ary operators."""

    descriptor: "OperatorDescriptor"

    def _make_singularity_inputs(
        self, singularity_value: float | complex, position: int = 0
    ) -> list[torch.Tensor]:
        """Create test inputs with singularity at specified position."""
        inputs = []
        for i, spec in enumerate(self.descriptor.input_specs):
            if i == position:
                # This input gets the singularity value
                x = torch.tensor([singularity_value], dtype=torch.float64)
            else:
                # Other inputs get normal values from their valid range
                low, high = spec.default_real_range
                x = torch.tensor(
                    [low + (high - low) * 0.5],
                    dtype=torch.float64,
                )
            inputs.append(x)
        return inputs

    def test_pole_behavior(self):
        """Test behavior at poles."""
        if "test_pole_behavior" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        pole_specs = [
            s for s in self.descriptor.singularities if s.type == "pole"
        ]
        if not pole_specs:
            pytest.skip("No pole singularities defined")

        for spec in pole_specs:
            for loc in list(spec.locations())[:10]:  # Test first 10 poles
                # Test pole in first input (position 0)
                inputs = self._make_singularity_inputs(loc, position=0)
                result = self.descriptor.func(*inputs)

                if spec.expected_behavior == "inf":
                    assert (
                        torch.isinf(result).all() or torch.isnan(result).all()
                    ), f"Expected inf/nan at pole {loc}, got {result}"

    def test_branch_cut_behavior(self):
        """Test behavior near branch cuts."""
        if "test_branch_cut_behavior" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        branch_specs = [
            s for s in self.descriptor.singularities if s.type == "branch_cut"
        ]
        if not branch_specs:
            pytest.skip("No branch cut singularities defined")

        for spec in branch_specs:
            for loc in list(spec.locations())[:5]:
                # Test approach from both sides of branch cut
                eps = 1e-10
                if isinstance(loc, complex):
                    above = torch.tensor(
                        [loc + eps * 1j], dtype=torch.complex128
                    )
                    below = torch.tensor(
                        [loc - eps * 1j], dtype=torch.complex128
                    )
                else:
                    above = torch.tensor([loc + eps], dtype=torch.float64)
                    below = torch.tensor([loc - eps], dtype=torch.float64)

                # Create full inputs for the function
                inputs_above = []
                inputs_below = []
                for i, input_spec in enumerate(self.descriptor.input_specs):
                    if i == 0:
                        inputs_above.append(above)
                        inputs_below.append(below)
                    else:
                        low, high = input_spec.default_real_range
                        val = torch.tensor(
                            [low + (high - low) * 0.5],
                            dtype=above.dtype,
                        )
                        inputs_above.append(val)
                        inputs_below.append(val)

                result_above = self.descriptor.func(*inputs_above)
                result_below = self.descriptor.func(*inputs_below)

                # Results should be finite near branch cuts
                assert torch.isfinite(result_above).all(), (
                    f"Expected finite near branch cut at {loc}"
                )
                assert torch.isfinite(result_below).all(), (
                    f"Expected finite near branch cut at {loc}"
                )

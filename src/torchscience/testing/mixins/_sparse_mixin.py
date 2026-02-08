from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class SparseMixin:
    """Mixin providing sparse tensor tests for n-ary operators."""

    descriptor: "OperatorDescriptor"

    def _create_sparse_coo_inputs(
        self, dtype: torch.dtype = torch.float64
    ) -> list[torch.Tensor]:
        """Create sparse COO test inputs based on descriptor's input_specs."""
        inputs = []
        indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
        shape = (3, 3)

        for spec in self.descriptor.input_specs:
            # Generate values within the input's valid range
            low, high = spec.default_real_range
            # Ensure values are within a reasonable sparse range
            low = max(low, -0.99) if low < 0 else low
            high = min(high, 0.99) if high <= 1 else high
            values = torch.tensor(
                [
                    low + (high - low) * 0.3,
                    low + (high - low) * 0.5,
                    low + (high - low) * 0.7,
                ],
                dtype=dtype,
            )
            sparse = torch.sparse_coo_tensor(indices, values, shape)
            inputs.append(sparse)

        return inputs

    def _create_sparse_csr_inputs(
        self, dtype: torch.dtype = torch.float64
    ) -> list[torch.Tensor]:
        """Create sparse CSR test inputs based on descriptor's input_specs."""
        inputs = []
        crow_indices = torch.tensor([0, 1, 2, 3])
        col_indices = torch.tensor([1, 2, 0])
        shape = (3, 3)

        for spec in self.descriptor.input_specs:
            # Generate values within the input's valid range
            low, high = spec.default_real_range
            low = max(low, -0.99) if low < 0 else low
            high = min(high, 0.99) if high <= 1 else high
            values = torch.tensor(
                [
                    low + (high - low) * 0.3,
                    low + (high - low) * 0.5,
                    low + (high - low) * 0.7,
                ],
                dtype=dtype,
            )
            sparse = torch.sparse_csr_tensor(
                crow_indices, col_indices, values, shape
            )
            inputs.append(sparse)

        return inputs

    def test_sparse_coo_basic(self):
        """Test basic sparse COO tensor support."""
        if "test_sparse_coo_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_sparse_coo:
            pytest.skip("Operator does not support sparse COO")

        # Create sparse inputs for all arguments
        sparse_inputs = self._create_sparse_coo_inputs()
        dense_inputs = [s.to_dense() for s in sparse_inputs]

        # Compute with sparse inputs
        result = self.descriptor.func(*sparse_inputs)

        # Verify sparsity is preserved
        assert result.is_sparse
        assert result.shape == sparse_inputs[0].shape

        # Compare with dense computation
        dense_result = self.descriptor.func(*dense_inputs)
        rtol, atol = self.descriptor.tolerances.get_tolerances(torch.float64)
        torch.testing.assert_close(
            result.to_dense(), dense_result, rtol=rtol, atol=atol
        )

    def test_sparse_csr_basic(self):
        """Test basic sparse CSR tensor support."""
        if "test_sparse_csr_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_sparse_csr:
            pytest.skip("Operator does not support sparse CSR")

        # Create sparse inputs for all arguments
        sparse_inputs = self._create_sparse_csr_inputs()
        dense_inputs = [s.to_dense() for s in sparse_inputs]

        # Compute with sparse inputs
        result = self.descriptor.func(*sparse_inputs)

        # Verify sparsity is preserved
        assert result.layout == torch.sparse_csr
        assert result.shape == sparse_inputs[0].shape

        # Compare with dense computation
        dense_result = self.descriptor.func(*dense_inputs)
        rtol, atol = self.descriptor.tolerances.get_tolerances(torch.float64)
        torch.testing.assert_close(
            result.to_dense(), dense_result, rtol=rtol, atol=atol
        )

    def test_sparse_coo_mixed_with_dense(self):
        """Test sparse COO mixed with dense inputs (first arg sparse)."""
        if "test_sparse_coo_mixed_with_dense" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_sparse_coo:
            pytest.skip("Operator does not support sparse COO")

        if self.descriptor.arity < 2:
            pytest.skip("Mixed sparse/dense test requires arity >= 2")

        # Create inputs: first sparse, rest dense
        sparse_inputs = self._create_sparse_coo_inputs()
        mixed_inputs = [sparse_inputs[0]] + [
            s.to_dense() for s in sparse_inputs[1:]
        ]
        dense_inputs = [s.to_dense() for s in sparse_inputs]

        # Compute with mixed inputs
        result = self.descriptor.func(*mixed_inputs)

        # Compare with fully dense computation
        dense_result = self.descriptor.func(*dense_inputs)
        rtol, atol = self.descriptor.tolerances.get_tolerances(torch.float64)

        if result.is_sparse:
            torch.testing.assert_close(
                result.to_dense(), dense_result, rtol=rtol, atol=atol
            )
        else:
            torch.testing.assert_close(
                result, dense_result, rtol=rtol, atol=atol
            )

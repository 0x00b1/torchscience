"""Tests for active information storage."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import (
    active_information_storage,
    mutual_information,
    shannon_entropy,
)


class TestActiveInformationStorageBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Output is scalar for 2D input."""
        joint = torch.rand(4, 4)
        joint = joint / joint.sum()
        result = active_information_storage(joint)
        assert result.dim() == 0

    def test_output_shape_batched(self):
        """Output has batch dimensions for >2D input."""
        joint = torch.rand(2, 3, 4, 4)
        joint = joint / joint.sum(dim=(-2, -1), keepdim=True)
        result = active_information_storage(joint)
        assert result.shape == (2, 3)

    def test_non_negative(self):
        """Active information storage is non-negative (it's MI)."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = active_information_storage(joint)
        assert result >= -1e-6  # Allow small numerical error

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        result = active_information_storage(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = active_information_storage(
            log_joint, input_type="log_probability"
        )
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(2, 3, 3)
        joint = joint / joint.sum(dim=(-2, -1), keepdim=True)
        result = active_information_storage(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(2, 3, 3)
        joint = joint / joint.sum(dim=(-2, -1), keepdim=True)
        result = active_information_storage(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2)
        joint = joint / joint.sum()
        result_nats = active_information_storage(joint, base=None)
        result_bits = active_information_storage(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestActiveInformationStorageCorrectness:
    """Correctness tests."""

    def test_iid_process_zero(self):
        """A(X) = 0 for i.i.d. process (x_t independent of x_{t-1})."""
        # Independent: p(x_t, x_{t-1}) = p(x_t) * p(x_{t-1})
        p_x = torch.tensor([0.3, 0.7])
        joint = p_x.unsqueeze(0) * p_x.unsqueeze(1)  # Outer product

        result = active_information_storage(joint)
        assert torch.abs(result) < 1e-5

    def test_iid_process_zero_nonuniform(self):
        """A(X) = 0 for non-uniform i.i.d. process."""
        p_x = torch.tensor([0.1, 0.2, 0.3, 0.4])
        joint = p_x.unsqueeze(0) * p_x.unsqueeze(1)

        result = active_information_storage(joint)
        assert torch.abs(result) < 1e-5

    def test_deterministic_process_equals_entropy(self):
        """A(X) = H(X) for deterministic process where x_t = x_{t-1}."""
        # Identity transition: p(x_t, x_{t-1}) = p(x) when x_t = x_{t-1}
        # This is a diagonal joint distribution
        joint = torch.zeros(4, 4)
        for i in range(4):
            joint[i, i] = 0.25

        result = active_information_storage(joint)

        # p(x_t) = sum_j p(x_t, x_j) = diagonal value = 0.25 for all states
        p_x = joint.sum(dim=1)  # Should be [0.25, 0.25, 0.25, 0.25]
        h_x = shannon_entropy(p_x)

        assert torch.isclose(result, h_x, rtol=1e-4)

    def test_deterministic_process_in_bits(self):
        """A(X) = H(X) = log2(n) bits for uniform deterministic process."""
        # Uniform deterministic: 4 states, each with prob 0.25
        joint = torch.zeros(4, 4)
        for i in range(4):
            joint[i, i] = 0.25

        result = active_information_storage(joint, base=2.0)

        # H(X) = log2(4) = 2 bits
        expected = torch.tensor(2.0)
        assert torch.isclose(result, expected, rtol=1e-4)

    def test_markov_chain_positive(self):
        """A(X) > 0 for Markov chain with memory."""
        # Create a Markov chain where x_t depends on x_{t-1}
        # Transition: from state 0, likely stay in 0; from state 1, likely stay in 1
        joint = torch.tensor(
            [
                [0.4, 0.1],  # x_t=0: likely when x_{t-1}=0
                [0.1, 0.4],  # x_t=1: likely when x_{t-1}=1
            ]
        )
        # Normalize
        joint = joint / joint.sum()

        result = active_information_storage(joint)
        assert result > 0.01  # Should be positive

    def test_equals_mutual_information(self):
        """A(X) = I(X_t; X_{t-1}) by definition."""
        # Create a random joint distribution
        joint = torch.rand(4, 5)
        joint = joint / joint.sum()

        ais = active_information_storage(joint)
        mi = mutual_information(joint, dims=(-2, -1))

        assert torch.allclose(ais, mi, rtol=1e-4)

    def test_bounded_by_marginal_entropy(self):
        """A(X) <= min(H(X_t), H(X_{t-1}))."""
        torch.manual_seed(42)
        for _ in range(10):
            joint = torch.rand(4, 5)
            joint = joint / joint.sum()

            ais = active_information_storage(joint)
            p_curr = joint.sum(dim=1)  # p(x_t)
            p_prev = joint.sum(dim=0)  # p(x_{t-1})
            h_curr = shannon_entropy(p_curr)
            h_prev = shannon_entropy(p_prev)

            min_entropy = torch.min(h_curr, h_prev)
            assert ais <= min_entropy + 1e-5

    def test_symmetric_joint_gives_symmetric_result(self):
        """For symmetric joint, result is same as mutual_information."""
        # Symmetric joint: p(x, y) = p(y, x)
        joint = torch.rand(3, 3)
        joint = (joint + joint.T) / 2  # Make symmetric
        joint = joint / joint.sum()

        ais = active_information_storage(joint)
        # For symmetric joint, I(X;Y) computed either way should be same
        mi_xy = mutual_information(joint, dims=(0, 1))
        mi_yx = mutual_information(joint, dims=(1, 0))

        assert torch.allclose(ais, mi_xy, rtol=1e-4)
        assert torch.allclose(mi_xy, mi_yx, rtol=1e-4)


class TestActiveInformationStorageGradients:
    """Gradient tests."""

    def test_gradcheck(self):
        """First-order gradients are correct."""
        joint = torch.rand(3, 4, dtype=torch.float64)
        joint = joint / joint.sum()
        # Add small offset to avoid zeros
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return active_information_storage(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = active_information_storage(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = active_information_storage(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape

    def test_backward_batched(self):
        """Backward pass works for batched input."""
        joint = torch.rand(2, 3, 4, requires_grad=True)
        joint_norm = joint / joint.sum(dim=(-2, -1), keepdim=True)
        result = active_information_storage(joint_norm, reduction="sum")
        result.backward()
        assert joint.grad is not None
        assert joint.grad.shape == joint.shape


class TestActiveInformationStorageValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="input_type"):
            active_information_storage(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="reduction"):
            active_information_storage(joint, reduction="invalid")

    def test_invalid_base_negative(self):
        """Negative base raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="base"):
            active_information_storage(joint, base=-1.0)

    def test_invalid_base_one(self):
        """Base=1 raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="base"):
            active_information_storage(joint, base=1.0)

    def test_too_few_dimensions(self):
        """Tensor with <2 dimensions raises ValueError."""
        joint = torch.rand(3)
        with pytest.raises(ValueError, match="2 dimensions"):
            active_information_storage(joint)

    def test_non_tensor_input(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="Tensor"):
            active_information_storage([[0.1, 0.2], [0.3, 0.4]])


class TestActiveInformationStorageMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape_2d(self):
        """Meta tensor produces correct output shape for 2D input."""
        joint = torch.rand(3, 4, device="meta")
        result = active_information_storage(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output for 2D input

    def test_meta_tensor_shape_batched(self):
        """Meta tensor produces correct output shape for batched input."""
        joint = torch.rand(2, 3, 4, 5, device="meta")
        result = active_information_storage(joint)
        assert result.device.type == "meta"
        assert result.shape == (2, 3)

    def test_meta_tensor_reduction(self):
        """Meta tensor with reduction produces scalar."""
        joint = torch.rand(2, 3, 4, device="meta")
        result = active_information_storage(joint, reduction="mean")
        assert result.device.type == "meta"
        assert result.dim() == 0


class TestActiveInformationStorageDtypes:
    """Dtype tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        """Output dtype matches input dtype."""
        joint = torch.rand(4, 4, dtype=dtype)
        joint = joint / joint.sum()
        result = active_information_storage(joint)
        assert result.dtype == dtype

"""Tests for transfer entropy."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import transfer_entropy


class TestTransferEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_3d(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(4, 4, 4)
        joint = joint / joint.sum()
        result = transfer_entropy(joint)
        assert result.dim() == 0

    def test_output_shape_batched(self):
        """Output has batch dimensions for >3D input."""
        joint = torch.rand(2, 3, 4, 4, 4)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = transfer_entropy(joint)
        assert result.shape == (2, 3)

    def test_non_negative(self):
        """Transfer entropy is non-negative."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = transfer_entropy(joint)
        assert result >= -1e-6  # Allow small numerical error

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = transfer_entropy(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = transfer_entropy(log_joint, input_type="log_probability")
        assert torch.isfinite(result)

    def test_input_type_logits(self):
        """Works with logits input type."""
        logits = torch.randn(3, 3, 3)
        result = transfer_entropy(logits, input_type="logits")
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(2, 3, 3, 3)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = transfer_entropy(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(2, 3, 3, 3)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = transfer_entropy(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2, 2)
        joint = joint / joint.sum()
        result_nats = transfer_entropy(joint, base=None)
        result_bits = transfer_entropy(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestTransferEntropyCorrectness:
    """Correctness tests."""

    def test_no_transfer_independent(self):
        """T(X -> Y) = 0 when X provides no info about Y beyond Y's past."""
        # Uniform distribution: X and Y are independent
        joint = torch.ones(2, 2, 2) / 8
        result = transfer_entropy(joint)
        assert torch.abs(result) < 1e-5

    def test_no_transfer_y_independent_of_x(self):
        """T(X -> Y) = 0 when Y_t depends only on Y_{t-1}, not X_{t-1}."""
        # p(y_t | y_prev, x_prev) = p(y_t | y_prev)
        # Create joint where X doesn't influence Y
        joint = torch.zeros(2, 2, 2)
        # For y_prev=0: p(y_t=0|y_prev=0) = 0.8, p(y_t=1|y_prev=0) = 0.2
        # For y_prev=1: p(y_t=0|y_prev=1) = 0.3, p(y_t=1|y_prev=1) = 0.7
        # X is uniform and independent
        for xp in range(2):
            joint[0, 0, xp] = (
                0.8 * 0.5 * 0.5
            )  # p(y_t=0|y_prev=0) * p(y_prev=0) * p(x_prev)
            joint[1, 0, xp] = 0.2 * 0.5 * 0.5
            joint[0, 1, xp] = 0.3 * 0.5 * 0.5
            joint[1, 1, xp] = 0.7 * 0.5 * 0.5

        result = transfer_entropy(joint)
        assert torch.abs(result) < 1e-5

    def test_positive_transfer(self):
        """T(X -> Y) > 0 when X helps predict Y."""
        # X perfectly determines Y_t: when x_prev=0, y_t=0; when x_prev=1, y_t=1
        joint = torch.zeros(2, 2, 2)
        # Uniform over (y_prev, x_prev), but y_t is determined by x_prev
        joint[0, 0, 0] = 0.25  # y_t=0, y_prev=0, x_prev=0
        joint[0, 1, 0] = 0.25  # y_t=0, y_prev=1, x_prev=0
        joint[1, 0, 1] = 0.25  # y_t=1, y_prev=0, x_prev=1
        joint[1, 1, 1] = 0.25  # y_t=1, y_prev=1, x_prev=1

        result = transfer_entropy(joint)
        # T(X -> Y) should be H(Y_t | Y_prev) - H(Y_t | Y_prev, X_prev)
        # = H(Y_t | Y_prev) - 0 = log(2) (since Y_t is uniform given Y_prev)
        expected = torch.log(torch.tensor(2.0))
        assert torch.abs(result - expected) < 1e-5

    def test_equals_conditional_mutual_information(self):
        """T(X -> Y) = I(Y_t; X_{t-1} | Y_{t-1}) by definition."""
        from torchscience.information import conditional_mutual_information

        # Create a non-trivial joint distribution
        joint = torch.rand(3, 3, 4)
        joint = joint / joint.sum()

        # Transfer entropy T(X -> Y) with joint p(y_t, y_prev, x_prev)
        te = transfer_entropy(joint)

        # Conditional MI I(Y_t; X_prev | Y_prev) with dims_x=(0,), dims_y=(2,), dims_z=(1,)
        # Note: CMI takes p(x, y, z) with dims_x, dims_y, dims_z
        # We want I(Y_t; X_prev | Y_prev)
        # So Y_t is X in CMI notation, X_prev is Y in CMI notation, Y_prev is Z
        cmi = conditional_mutual_information(
            joint,
            dims_x=(0,),  # Y_t
            dims_y=(2,),  # X_prev
            dims_z=(1,),  # Y_prev
        )

        assert torch.allclose(te, cmi, rtol=1e-4)

    def test_asymmetry(self):
        """T(X -> Y) != T(Y -> X) in general."""
        # Create asymmetric joint distribution
        joint_xy = torch.rand(3, 3, 4)
        joint_xy = joint_xy / joint_xy.sum()

        # T(X -> Y) from p(y_t, y_prev, x_prev)
        te_xy = transfer_entropy(joint_xy)

        # For T(Y -> X), we need p(x_t, x_prev, y_prev)
        # This is a different distribution, so we create one
        joint_yx = torch.rand(4, 4, 3)  # p(x_t, x_prev, y_prev)
        joint_yx = joint_yx / joint_yx.sum()
        te_yx = transfer_entropy(joint_yx)

        # They should generally be different (unless by coincidence)
        # Just check both are valid
        assert torch.isfinite(te_xy) and torch.isfinite(te_yx)


class TestTransferEntropyGradients:
    """Gradient tests."""

    def test_gradcheck(self):
        """First-order gradients are correct."""
        joint = torch.rand(3, 3, 3, dtype=torch.float64)
        joint = joint / joint.sum()
        # Add small offset to avoid zeros for numerical stability
        joint = joint + 1e-4
        joint = joint / joint.sum()
        joint.requires_grad_(True)

        def func(j):
            return transfer_entropy(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 3, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = transfer_entropy(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = transfer_entropy(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape

    def test_backward_batched(self):
        """Backward pass works for batched input."""
        joint = torch.rand(2, 3, 4, 4, requires_grad=True)
        joint_norm = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = transfer_entropy(joint_norm, reduction="sum")
        result.backward()
        assert joint.grad is not None
        assert joint.grad.shape == joint.shape


class TestTransferEntropyValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="input_type"):
            transfer_entropy(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="reduction"):
            transfer_entropy(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="base"):
            transfer_entropy(joint, base=-1.0)

    def test_too_few_dimensions(self):
        """Tensor with <3 dimensions raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="3 dimensions"):
            transfer_entropy(joint)

    def test_non_tensor_input(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="Tensor"):
            transfer_entropy([[0.1, 0.2], [0.3, 0.4]])


class TestTransferEntropyMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape_3d(self):
        """Meta tensor produces correct output shape for 3D input."""
        joint = torch.rand(3, 4, 5, device="meta")
        result = transfer_entropy(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output for 3D input

    def test_meta_tensor_shape_batched(self):
        """Meta tensor produces correct output shape for batched input."""
        joint = torch.rand(2, 3, 4, 5, 6, device="meta")
        result = transfer_entropy(joint)
        assert result.device.type == "meta"
        assert result.shape == (2, 3)

    def test_meta_tensor_reduction(self):
        """Meta tensor with reduction produces scalar."""
        joint = torch.rand(2, 3, 4, 5, device="meta")
        result = transfer_entropy(joint, reduction="mean")
        assert result.device.type == "meta"
        assert result.dim() == 0

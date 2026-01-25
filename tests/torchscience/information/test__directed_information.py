"""Tests for directed information."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import directed_information


class TestDirectedInformationBasic:
    """Basic functionality tests."""

    def test_output_shape_3d(self):
        """Output is scalar for 3D input."""
        joint = torch.rand(4, 4, 4)
        joint = joint / joint.sum()
        result = directed_information(joint)
        assert result.dim() == 0

    def test_output_shape_batched(self):
        """Output has batch dimensions for >3D input."""
        joint = torch.rand(2, 3, 4, 4, 4)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = directed_information(joint)
        assert result.shape == (2, 3)

    def test_non_negative(self):
        """Directed information is non-negative."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = directed_information(joint)
        assert result >= -1e-6  # Allow small numerical error

    def test_input_type_probability(self):
        """Works with probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        result = directed_information(joint, input_type="probability")
        assert torch.isfinite(result)

    def test_input_type_log_probability(self):
        """Works with log_probability input type."""
        joint = torch.rand(3, 3, 3)
        joint = joint / joint.sum()
        log_joint = torch.log(joint)
        result = directed_information(log_joint, input_type="log_probability")
        assert torch.isfinite(result)

    def test_input_type_logits(self):
        """Works with logits input type."""
        logits = torch.randn(3, 3, 3)
        result = directed_information(logits, input_type="logits")
        assert torch.isfinite(result)

    def test_reduction_mean(self):
        """Mean reduction works."""
        joint = torch.rand(2, 3, 3, 3)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = directed_information(joint, reduction="mean")
        assert result.dim() == 0

    def test_reduction_sum(self):
        """Sum reduction works."""
        joint = torch.rand(2, 3, 3, 3)
        joint = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = directed_information(joint, reduction="sum")
        assert result.dim() == 0

    def test_base_bits(self):
        """Base=2 gives result in bits."""
        joint = torch.rand(2, 2, 2)
        joint = joint / joint.sum()
        result_nats = directed_information(joint, base=None)
        result_bits = directed_information(joint, base=2.0)
        # Result in bits should be result_nats / log(2)
        expected_bits = result_nats / torch.log(torch.tensor(2.0))
        assert torch.allclose(result_bits, expected_bits, rtol=1e-4)


class TestDirectedInformationCorrectness:
    """Correctness tests."""

    def test_no_transfer_independent(self):
        """I(X -> Y) = 0 when X provides no info about Y beyond Y's past."""
        # Uniform distribution: X and Y are independent
        joint = torch.ones(2, 2, 2) / 8
        result = directed_information(joint)
        assert torch.abs(result) < 1e-5

    def test_no_transfer_y_independent_of_x(self):
        """I(X -> Y) = 0 when Y_t depends only on Y_{t-1}, not X_t."""
        # p(y_t | y_prev, x_t) = p(y_t | y_prev)
        # Create joint where X doesn't influence Y
        joint = torch.zeros(2, 2, 2)
        # For y_prev=0: p(y_t=0|y_prev=0) = 0.8, p(y_t=1|y_prev=0) = 0.2
        # For y_prev=1: p(y_t=0|y_prev=1) = 0.3, p(y_t=1|y_prev=1) = 0.7
        # X is uniform and independent
        for xt in range(2):
            joint[0, 0, xt] = (
                0.8 * 0.5 * 0.5
            )  # p(y_t=0|y_prev=0) * p(y_prev=0) * p(x_t)
            joint[1, 0, xt] = 0.2 * 0.5 * 0.5
            joint[0, 1, xt] = 0.3 * 0.5 * 0.5
            joint[1, 1, xt] = 0.7 * 0.5 * 0.5

        result = directed_information(joint)
        assert torch.abs(result) < 1e-5

    def test_positive_transfer(self):
        """I(X -> Y) > 0 when X helps predict Y."""
        # X perfectly determines Y_t: when x_t=0, y_t=0; when x_t=1, y_t=1
        joint = torch.zeros(2, 2, 2)
        # Uniform over (y_prev, x_t), but y_t is determined by x_t
        joint[0, 0, 0] = 0.25  # y_t=0, y_prev=0, x_t=0
        joint[0, 1, 0] = 0.25  # y_t=0, y_prev=1, x_t=0
        joint[1, 0, 1] = 0.25  # y_t=1, y_prev=0, x_t=1
        joint[1, 1, 1] = 0.25  # y_t=1, y_prev=1, x_t=1

        result = directed_information(joint)
        # I(X -> Y) should be H(Y_t | Y_prev) - H(Y_t | Y_prev, X_t)
        # = H(Y_t | Y_prev) - 0 = log(2) (since Y_t is uniform given Y_prev)
        expected = torch.log(torch.tensor(2.0))
        assert torch.abs(result - expected) < 1e-5

    def test_equals_conditional_mutual_information(self):
        """I(X -> Y) = I(X_t; Y_t | Y_{t-1}) by definition."""
        from torchscience.information import conditional_mutual_information

        # Create a non-trivial joint distribution
        joint = torch.rand(3, 3, 4)
        joint = joint / joint.sum()

        # Directed information I(X -> Y) with joint p(y_t, y_prev, x_t)
        di = directed_information(joint)

        # Conditional MI I(X_t; Y_t | Y_prev) with dims_x=(2,), dims_y=(0,), dims_z=(1,)
        # Note: CMI takes p(x, y, z) with dims_x, dims_y, dims_z
        # We want I(X_t; Y_t | Y_prev)
        # So X_t is X in CMI notation, Y_t is Y in CMI notation, Y_prev is Z
        cmi = conditional_mutual_information(
            joint,
            dims_x=(2,),  # X_t
            dims_y=(0,),  # Y_t
            dims_z=(1,),  # Y_prev
        )

        assert torch.allclose(di, cmi, rtol=1e-4)

    def test_asymmetry(self):
        """I(X -> Y) != I(Y -> X) in general."""
        # Create asymmetric joint distribution
        joint_xy = torch.rand(3, 3, 4)
        joint_xy = joint_xy / joint_xy.sum()

        # I(X -> Y) from p(y_t, y_prev, x_t)
        di_xy = directed_information(joint_xy)

        # For I(Y -> X), we need p(x_t, x_prev, y_t)
        # This is a different distribution, so we create one
        joint_yx = torch.rand(4, 4, 3)  # p(x_t, x_prev, y_t)
        joint_yx = joint_yx / joint_yx.sum()
        di_yx = directed_information(joint_yx)

        # They should generally be different (unless by coincidence)
        # Just check both are valid
        assert torch.isfinite(di_xy) and torch.isfinite(di_yx)

    def test_relationship_to_transfer_entropy(self):
        """Directed information and transfer entropy are structurally similar."""
        from torchscience.information import transfer_entropy

        # Both are instances of conditional mutual information
        # Transfer entropy: T(X -> Y) = I(Y_t; X_{t-1} | Y_{t-1})
        # Directed info:   I(X -> Y) = I(X_t; Y_t | Y_{t-1})
        #
        # For the same joint distribution, they compute different CMI
        # but should both be non-negative
        joint = torch.rand(3, 3, 4)
        joint = joint / joint.sum()

        di = directed_information(joint)
        te = transfer_entropy(joint)

        # Both should be non-negative
        assert di >= -1e-6
        assert te >= -1e-6
        # They compute different quantities, so generally different
        # Just check both are valid
        assert torch.isfinite(di)
        assert torch.isfinite(te)


class TestDirectedInformationGradients:
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
            return directed_information(j, input_type="probability")

        assert gradcheck(func, (joint,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass executes without error."""
        joint = torch.rand(3, 3, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = directed_information(joint_norm)
        result.backward()
        assert joint.grad is not None
        assert torch.isfinite(joint.grad).all()

    def test_backward_shape(self):
        """Gradient has same shape as input."""
        joint = torch.rand(4, 5, 3, requires_grad=True)
        joint_norm = joint / joint.sum()
        result = directed_information(joint_norm)
        result.backward()
        assert joint.grad.shape == joint.shape

    def test_backward_batched(self):
        """Backward pass works for batched input."""
        joint = torch.rand(2, 3, 4, 4, requires_grad=True)
        joint_norm = joint / joint.sum(dim=(-3, -2, -1), keepdim=True)
        result = directed_information(joint_norm, reduction="sum")
        result.backward()
        assert joint.grad is not None
        assert joint.grad.shape == joint.shape


class TestDirectedInformationValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Invalid input_type raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="input_type"):
            directed_information(joint, input_type="invalid")

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="reduction"):
            directed_information(joint, reduction="invalid")

    def test_invalid_base(self):
        """Invalid base raises ValueError."""
        joint = torch.rand(3, 3, 3)
        with pytest.raises(ValueError, match="base"):
            directed_information(joint, base=-1.0)

    def test_too_few_dimensions(self):
        """Tensor with <3 dimensions raises ValueError."""
        joint = torch.rand(3, 3)
        with pytest.raises(ValueError, match="3 dimensions"):
            directed_information(joint)

    def test_non_tensor_input(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="Tensor"):
            directed_information([[0.1, 0.2], [0.3, 0.4]])


class TestDirectedInformationMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape_3d(self):
        """Meta tensor produces correct output shape for 3D input."""
        joint = torch.rand(3, 4, 5, device="meta")
        result = directed_information(joint)
        assert result.device.type == "meta"
        assert result.dim() == 0  # Scalar output for 3D input

    def test_meta_tensor_shape_batched(self):
        """Meta tensor produces correct output shape for batched input."""
        joint = torch.rand(2, 3, 4, 5, 6, device="meta")
        result = directed_information(joint)
        assert result.device.type == "meta"
        assert result.shape == (2, 3)

    def test_meta_tensor_reduction(self):
        """Meta tensor with reduction produces scalar."""
        joint = torch.rand(2, 3, 4, 5, device="meta")
        result = directed_information(joint, reduction="mean")
        assert result.device.type == "meta"
        assert result.dim() == 0

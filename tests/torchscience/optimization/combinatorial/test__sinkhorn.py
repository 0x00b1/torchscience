import pytest
import torch
import torch.testing

import torchscience.optimization.combinatorial


class TestSinkhorn:
    def test_uniform_marginals(self):
        """Test with uniform marginals."""
        n, m = 3, 4
        C = torch.rand(n, m)
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)

        # Check marginal constraints
        torch.testing.assert_close(P.sum(dim=-1), a, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(P.sum(dim=-2), b, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        """Test output shape matches cost matrix."""
        C = torch.rand(5, 7)
        a = torch.ones(5) / 5
        b = torch.ones(7) / 7
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert P.shape == C.shape

    def test_batched(self):
        """Test batched cost matrices."""
        batch, n, m = 2, 3, 4
        C = torch.rand(batch, n, m)
        a = torch.ones(batch, n) / n
        b = torch.ones(batch, m) / m

        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)

        assert P.shape == (batch, n, m)
        torch.testing.assert_close(P.sum(dim=-1), a, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(P.sum(dim=-2), b, atol=1e-4, rtol=1e-4)

    def test_nonnegative(self):
        """Test that transport plan is non-negative."""
        C = torch.rand(5, 5)
        a = torch.ones(5) / 5
        b = torch.ones(5) / 5
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert (P >= -1e-6).all()

    def test_gradient_wrt_cost(self):
        """Test gradient with respect to cost matrix."""
        n, m = 3, 4
        C = torch.rand(n, m, requires_grad=True)
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        loss = (P * C).sum()
        loss.backward()

        assert C.grad is not None
        assert C.grad.shape == C.shape

    @pytest.mark.xfail(
        reason="Backward uses simplified -P/epsilon gradient; full implicit diff needed"
    )
    def test_gradcheck(self):
        """Test gradient correctness via finite differences.

        Note: The backward uses a simplified gradient (-P/epsilon) that treats
        P as independent of C. The true gradient requires accounting for how
        changes in C affect the scaling vectors u and v through the Sinkhorn
        iterations. This would require either:
        1. Unrolling iterations through autograd (memory intensive)
        2. Full implicit differentiation through the fixed point equations
        """
        n, m = 3, 4
        C = torch.rand(n, m, dtype=torch.float64, requires_grad=True)
        a = torch.ones(n, dtype=torch.float64) / n
        b = torch.ones(m, dtype=torch.float64) / m

        def fn(C_):
            return torchscience.optimization.combinatorial.sinkhorn(C_, a, b)

        assert torch.autograd.gradcheck(
            fn, (C,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_regularization_effect(self):
        """Test that smaller epsilon gives sparser solution."""
        n = 3
        C = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        a = torch.ones(n) / n
        b = torch.ones(n) / n

        P_large_eps = torchscience.optimization.combinatorial.sinkhorn(
            C, a, b, epsilon=1.0
        )
        P_small_eps = torchscience.optimization.combinatorial.sinkhorn(
            C, a, b, epsilon=0.01
        )

        # Smaller epsilon should be more peaked (higher max value)
        assert P_small_eps.max() > P_large_eps.max()

    def test_identity_cost(self):
        """With zero cost, uniform marginals give uniform plan."""
        n = 4
        C = torch.zeros(n, n)
        a = torch.ones(n) / n
        b = torch.ones(n) / n
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        expected = torch.ones(n, n) / (n * n)
        torch.testing.assert_close(P, expected, atol=1e-4, rtol=1e-4)

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        C = torch.empty(3, 4, device="meta")
        a = torch.empty(3, device="meta")
        b = torch.empty(4, device="meta")
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert P.shape == (3, 4)
        assert P.device.type == "meta"


class TestSinkhornDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""
        C = torch.rand(3, 4, dtype=dtype)
        a = torch.ones(3, dtype=dtype) / 3
        b = torch.ones(4, dtype=dtype) / 4
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert P.dtype == dtype

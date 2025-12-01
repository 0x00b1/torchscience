import torch
from tensordict import TensorDict

from torchscience.integration._tensordict_utils import flatten_state


class TestFlattenUnflatten:
    def test_tensor_passthrough(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        flat, unflatten = flatten_state(y)
        assert torch.equal(flat, y)
        restored = unflatten(flat)
        assert torch.equal(restored, y)

    def test_simple_tensordict(self):
        y = TensorDict(
            {"x": torch.tensor([1.0, 2.0]), "v": torch.tensor([3.0])}
        )
        flat, unflatten = flatten_state(y)
        assert flat.shape == (3,)
        restored = unflatten(flat)
        assert isinstance(restored, TensorDict)
        assert torch.equal(restored["x"], y["x"])
        assert torch.equal(restored["v"], y["v"])

    def test_nested_tensordict(self):
        y = TensorDict(
            {
                "robot": TensorDict(
                    {
                        "joints": torch.tensor([1.0, 2.0]),
                        "velocities": torch.tensor([3.0, 4.0]),
                    }
                ),
                "object": TensorDict({"pose": torch.tensor([5.0, 6.0, 7.0])}),
            }
        )
        flat, unflatten = flatten_state(y)
        assert flat.shape == (7,)
        restored = unflatten(flat)
        assert torch.equal(restored["robot", "joints"], y["robot", "joints"])
        assert torch.equal(restored["object", "pose"], y["object", "pose"])

    def test_batched_tensordict(self):
        y = TensorDict(
            {"x": torch.randn(5, 3), "v": torch.randn(5, 2)}, batch_size=[5]
        )
        flat, unflatten = flatten_state(y)
        assert flat.shape == (5, 5)  # batch_size=5, state_dim=3+2=5
        restored = unflatten(flat)
        assert restored.shape == y.shape
        assert torch.equal(restored["x"], y["x"])

    def test_preserves_gradients(self):
        y = TensorDict(
            {"x": torch.tensor([1.0, 2.0], requires_grad=True)}, batch_size=[]
        )
        flat, unflatten = flatten_state(y)
        assert flat.requires_grad
        restored = unflatten(flat)
        loss = restored["x"].sum()
        loss.backward()
        assert y["x"].grad is not None

    def test_vectorized_unflatten(self):
        """Test that unflatten handles (T, *batch, state_dim) input"""
        y = TensorDict(
            {"x": torch.randn(5, 3), "v": torch.randn(5, 2)}, batch_size=[5]
        )
        flat, unflatten = flatten_state(y)  # flat: (5, 5)

        # Simulate multi-time query: (T=10, B=5, D=5)
        multi_time_flat = torch.randn(10, 5, 5)
        restored = unflatten(multi_time_flat)

        assert isinstance(restored, TensorDict)
        assert restored.batch_size == (10, 5)
        assert restored["x"].shape == (10, 5, 3)
        assert restored["v"].shape == (10, 5, 2)

    def test_unflatten_closure_reuse(self):
        """Verify unflatten closure can be called multiple times efficiently"""
        y = TensorDict({"x": torch.randn(3)}, batch_size=[])
        _, unflatten = flatten_state(y)

        # Call multiple times - should reuse cached metadata
        for _ in range(100):
            result = unflatten(torch.randn(3))
            assert result["x"].shape == (3,)

"""Utilities for flattening and unflattening TensorDict states.

Uses TensorDict's built-in flatten_keys() for reliable flattening,
with custom unflatten logic for vectorized multi-time queries.
"""

from typing import Callable, List, Tuple, Union

import torch
from tensordict import TensorDict


def flatten_state(
    y: Union[torch.Tensor, TensorDict],
) -> Tuple[
    torch.Tensor, Callable[[torch.Tensor], Union[torch.Tensor, TensorDict]]
]:
    """
    Flatten a Tensor or TensorDict to a 1D (or batched 1D) tensor.

    Parameters
    ----------
    y : Tensor or TensorDict
        The state to flatten.

    Returns
    -------
    flat : Tensor
        Flattened state. Shape is (*batch_dims, total_elements).
    unflatten : callable
        Function to restore the original structure from a flat tensor.
        Supports vectorized input: shape (T, *batch_dims, total_elements)
        produces TensorDict with batch_size=(T, *batch_dims).

    Notes
    -----
    Uses TensorDict's flatten_keys() for flattening nested structures.
    The returned unflatten function caches all metadata from the template,
    so it can be called efficiently multiple times without recreating the
    key/shape information.
    """
    if isinstance(y, torch.Tensor):
        # Tensor passthrough - no flattening needed
        def unflatten_tensor(flat: torch.Tensor) -> torch.Tensor:
            return flat

        return y, unflatten_tensor

    # TensorDict case - use built-in flatten_keys for nested support
    y_flat_keys = y.flatten_keys(separator=".")

    # Get all keys in consistent sorted order
    flat_keys = sorted(y_flat_keys.keys())

    # Determine batch dimensions
    batch_size = tuple(y.batch_size)
    n_batch_dims = len(batch_size)

    # Collect shapes and flatten each leaf
    shapes: List[Tuple[str, Tuple[int, ...]]] = []
    flat_parts = []
    for key in flat_keys:
        leaf = y_flat_keys[key]
        # Shape after batch dimensions
        leaf_shape = tuple(leaf.shape[n_batch_dims:])
        shapes.append((key, leaf_shape))
        # Flatten non-batch dimensions
        flat_leaf = leaf.reshape(*batch_size, -1)
        flat_parts.append(flat_leaf)

    # Concatenate along the last dimension
    flat = torch.cat(flat_parts, dim=-1)

    # Cache original structure for unflattening
    original_keys = list(y.keys(include_nested=True, leaves_only=True))

    def unflatten_tensordict(flat_tensor: torch.Tensor) -> TensorDict:
        """
        Restore TensorDict structure from flat tensor.

        Supports vectorized input: if flat_tensor has shape (T, *batch, state_dim),
        the result has batch_size=(T, *batch).
        """
        # Determine the batch dimensions from input
        # The last dimension is always state_dim
        input_batch_shape = flat_tensor.shape[:-1]

        # First unflatten to flat-key structure
        flat_td = TensorDict({}, batch_size=input_batch_shape)

        offset = 0
        for key, shape in shapes:
            numel = 1
            for s in shape:
                numel *= s
            leaf_flat = flat_tensor[..., offset : offset + numel]
            leaf = leaf_flat.reshape(*input_batch_shape, *shape)
            flat_td[key] = leaf
            offset += numel

        # Unflatten keys back to nested structure
        return flat_td.unflatten_keys(separator=".")

    return flat, unflatten_tensordict


def unflatten_state(
    flat: torch.Tensor,
    template: Union[torch.Tensor, TensorDict],
) -> Union[torch.Tensor, TensorDict]:
    """
    Unflatten a flat tensor using a template for structure.

    Parameters
    ----------
    flat : Tensor
        Flattened state tensor.
    template : Tensor or TensorDict
        Template providing the target structure.

    Returns
    -------
    y : Tensor or TensorDict
        Unflattened state matching template structure.

    Notes
    -----
    This function creates a new unflatten closure each time. For repeated
    unflattening, prefer to cache the unflatten function from flatten_state().
    """
    _, unflatten_fn = flatten_state(template)
    return unflatten_fn(flat)

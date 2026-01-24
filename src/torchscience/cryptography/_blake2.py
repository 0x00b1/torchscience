import torch
from torch import Tensor


def blake2b(
    data: Tensor,
    key: Tensor | None = None,
    digest_size: int = 64,
) -> Tensor:
    """Compute BLAKE2b hash.

    BLAKE2b is a cryptographic hash function optimized for 64-bit platforms,
    producing digests of 1-64 bytes. It supports optional keyed hashing for
    use as a MAC (Message Authentication Code).

    Parameters
    ----------
    data : Tensor
        Input bytes as (n,) uint8 tensor.
    key : Tensor | None, optional
        Optional key for keyed hashing as (k,) uint8 tensor where k <= 64.
        Default is None (unkeyed hash).
    digest_size : int, optional
        Output digest size in bytes (1-64). Default is 64.

    Returns
    -------
    Tensor
        (digest_size,) uint8 tensor containing the hash.

    Examples
    --------
    >>> import torch
    >>> from torchscience.cryptography import blake2b
    >>> data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
    >>> blake2b(data, digest_size=32)  # 32-byte hash of "abc"
    """
    if key is None:
        key = torch.tensor([], dtype=torch.uint8, device=data.device)
    return torch.ops.torchscience.blake2b(data, key, digest_size)


def blake2s(
    data: Tensor,
    key: Tensor | None = None,
    digest_size: int = 32,
) -> Tensor:
    """Compute BLAKE2s hash.

    BLAKE2s is a cryptographic hash function optimized for 8-32 bit platforms,
    producing digests of 1-32 bytes. It supports optional keyed hashing for
    use as a MAC (Message Authentication Code).

    Parameters
    ----------
    data : Tensor
        Input bytes as (n,) uint8 tensor.
    key : Tensor | None, optional
        Optional key for keyed hashing as (k,) uint8 tensor where k <= 32.
        Default is None (unkeyed hash).
    digest_size : int, optional
        Output digest size in bytes (1-32). Default is 32.

    Returns
    -------
    Tensor
        (digest_size,) uint8 tensor containing the hash.

    Examples
    --------
    >>> import torch
    >>> from torchscience.cryptography import blake2s
    >>> data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)
    >>> blake2s(data, digest_size=16)  # 16-byte hash of "abc"
    """
    if key is None:
        key = torch.tensor([], dtype=torch.uint8, device=data.device)
    return torch.ops.torchscience.blake2s(data, key, digest_size)

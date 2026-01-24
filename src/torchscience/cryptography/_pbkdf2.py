import torch
from torch import Tensor


def pbkdf2_sha256(
    password: Tensor,
    salt: Tensor,
    iterations: int,
    output_len: int = 32,
) -> Tensor:
    """PBKDF2-HMAC-SHA256 key derivation.

    Derives a cryptographic key from a password using the PBKDF2 algorithm
    with HMAC-SHA256 as the pseudorandom function (RFC 2898/8018).

    Parameters
    ----------
    password : Tensor
        Password bytes as a 1D uint8 tensor.
    salt : Tensor
        Salt bytes as a 1D uint8 tensor. Should be random and unique.
    iterations : int
        Number of iterations. Higher is more secure but slower.
        OWASP recommends at least 600,000 for SHA256.
    output_len : int, optional
        Desired output length in bytes. Default is 32.

    Returns
    -------
    Tensor
        Derived key as a 1D uint8 tensor of length output_len.
    """
    return torch.ops.torchscience.pbkdf2_sha256(
        password, salt, iterations, output_len
    )

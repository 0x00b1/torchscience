import torch
from torch import Tensor


def hkdf_extract_sha256(salt: Tensor, ikm: Tensor) -> Tensor:
    """HKDF-Extract with SHA256.

    Extracts a pseudorandom key (PRK) from input keying material (IKM)
    using an optional salt.

    Parameters
    ----------
    salt : Tensor
        Salt bytes as a 1D uint8 tensor. Can be empty.
    ikm : Tensor
        Input keying material as a 1D uint8 tensor.

    Returns
    -------
    Tensor
        32-byte pseudorandom key (PRK).
    """
    return torch.ops.torchscience.hkdf_extract_sha256(salt, ikm)


def hkdf_expand_sha256(prk: Tensor, info: Tensor, output_len: int) -> Tensor:
    """HKDF-Expand with SHA256.

    Expands a pseudorandom key into output keying material.

    Parameters
    ----------
    prk : Tensor
        Pseudorandom key from hkdf_extract, 32 bytes.
    info : Tensor
        Context/application-specific info as a 1D uint8 tensor.
    output_len : int
        Desired output length (max 8160 bytes).

    Returns
    -------
    Tensor
        Output keying material of length output_len.
    """
    return torch.ops.torchscience.hkdf_expand_sha256(prk, info, output_len)


def hkdf_sha256(
    ikm: Tensor,
    salt: Tensor,
    info: Tensor,
    output_len: int,
) -> Tensor:
    """Combined HKDF (extract + expand) with SHA256.

    Parameters
    ----------
    ikm : Tensor
        Input keying material.
    salt : Tensor
        Salt (can be empty).
    info : Tensor
        Context info (can be empty).
    output_len : int
        Desired output length (max 8160 bytes).

    Returns
    -------
    Tensor
        Derived key of length output_len.
    """
    return torch.ops.torchscience.hkdf_sha256(ikm, salt, info, output_len)

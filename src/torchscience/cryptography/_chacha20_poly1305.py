import torch
from torch import Tensor


def chacha20_poly1305_encrypt(
    plaintext: Tensor,
    key: Tensor,
    nonce: Tensor,
    aad: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Encrypt with ChaCha20-Poly1305 AEAD.

    Parameters
    ----------
    plaintext : Tensor
        (n,) uint8 tensor - data to encrypt.
    key : Tensor
        (32,) uint8 tensor - encryption key.
    nonce : Tensor
        (12,) uint8 tensor - unique nonce.
    aad : Tensor, optional
        Additional authenticated data (not encrypted).

    Returns
    -------
    tuple[Tensor, Tensor]
        (ciphertext, tag) - ciphertext and 16-byte authentication tag.
    """
    if aad is None:
        aad = torch.tensor([], dtype=torch.uint8, device=plaintext.device)
    return torch.ops.torchscience.chacha20_poly1305_encrypt(
        plaintext, key, nonce, aad
    )


def chacha20_poly1305_decrypt(
    ciphertext: Tensor,
    key: Tensor,
    nonce: Tensor,
    tag: Tensor,
    aad: Tensor | None = None,
) -> Tensor:
    """Decrypt with ChaCha20-Poly1305 AEAD.

    Parameters
    ----------
    ciphertext : Tensor
        (n,) uint8 tensor - data to decrypt.
    key : Tensor
        (32,) uint8 tensor - decryption key.
    nonce : Tensor
        (12,) uint8 tensor - same nonce used for encryption.
    tag : Tensor
        (16,) uint8 tensor - authentication tag.
    aad : Tensor, optional
        Additional authenticated data (must match encryption).

    Returns
    -------
    Tensor
        (n,) uint8 tensor - decrypted plaintext.

    Raises
    ------
    RuntimeError
        If authentication fails.
    """
    if aad is None:
        aad = torch.tensor([], dtype=torch.uint8, device=ciphertext.device)
    return torch.ops.torchscience.chacha20_poly1305_decrypt(
        ciphertext, key, nonce, aad, tag
    )

import torch
from torch import Tensor


def aes_encrypt_block(plaintext: Tensor, key: Tensor) -> Tensor:
    """Encrypt a single 16-byte block with AES.

    Parameters
    ----------
    plaintext : Tensor
        (16,) uint8 tensor - single block to encrypt.
    key : Tensor
        (16,) or (32,) uint8 tensor for AES-128 or AES-256.

    Returns
    -------
    Tensor
        (16,) uint8 tensor - encrypted block.
    """
    return torch.ops.torchscience.aes_encrypt_block(plaintext, key)


def aes_decrypt_block(ciphertext: Tensor, key: Tensor) -> Tensor:
    """Decrypt a single 16-byte block with AES.

    Parameters
    ----------
    ciphertext : Tensor
        (16,) uint8 tensor - single block to decrypt.
    key : Tensor
        (16,) or (32,) uint8 tensor for AES-128 or AES-256.

    Returns
    -------
    Tensor
        (16,) uint8 tensor - decrypted block.
    """
    return torch.ops.torchscience.aes_decrypt_block(ciphertext, key)


def aes_ctr(
    data: Tensor,
    key: Tensor,
    nonce: Tensor,
    counter: int = 0,
) -> Tensor:
    """Encrypt or decrypt data using AES-CTR mode.

    Parameters
    ----------
    data : Tensor
        (n,) uint8 tensor - data to encrypt/decrypt.
    key : Tensor
        (16,) or (32,) uint8 tensor for AES-128 or AES-256.
    nonce : Tensor
        (12,) uint8 tensor - nonce/IV.
    counter : int
        Initial counter value. Default: 0.

    Returns
    -------
    Tensor
        (n,) uint8 tensor - encrypted/decrypted data.
    """
    return torch.ops.torchscience.aes_ctr(data, key, nonce, counter)

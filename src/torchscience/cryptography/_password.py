import torch
from torch import Tensor

from torchscience.cryptography._pbkdf2 import pbkdf2_sha256

# OWASP recommended minimum for PBKDF2-SHA256 (2023)
DEFAULT_ITERATIONS = 600000
SALT_LENGTH = 16
HASH_LENGTH = 32


def password_hash(
    password: Tensor | bytes,
    iterations: int = DEFAULT_ITERATIONS,
) -> Tensor:
    """Hash a password for secure storage.

    Uses PBKDF2-SHA256 with a random salt. Returns an encoded tensor
    containing the salt, iteration count, and derived hash.

    Parameters
    ----------
    password : Tensor or bytes
        Password to hash.
    iterations : int
        Number of PBKDF2 iterations. Default is 600,000 per OWASP
        recommendations. Use lower values only for testing.

    Returns
    -------
    Tensor
        Encoded hash containing: salt (16) + iterations (4) + hash (32) = 52 bytes.
        Store this entire tensor - it contains everything needed for verification.

    Examples
    --------
    >>> stored = password_hash(b"my_password")
    >>> # Store `stored` in database
    >>> assert password_verify(b"my_password", stored)
    """
    if isinstance(password, bytes):
        password = torch.tensor(list(password), dtype=torch.uint8)

    # Generate random salt
    salt = torch.randint(
        0, 256, (SALT_LENGTH,), dtype=torch.uint8, device=password.device
    )

    # Derive hash
    derived = pbkdf2_sha256(password, salt, iterations, HASH_LENGTH)

    # Encode iterations as 4 bytes (big-endian)
    iter_bytes = torch.tensor(
        [
            (iterations >> 24) & 0xFF,
            (iterations >> 16) & 0xFF,
            (iterations >> 8) & 0xFF,
            iterations & 0xFF,
        ],
        dtype=torch.uint8,
        device=password.device,
    )

    # Concatenate: salt + iterations + hash
    return torch.cat([salt, iter_bytes, derived])


def password_verify(password: Tensor | bytes, hash_tensor: Tensor) -> bool:
    """Verify a password against a stored hash.

    Parameters
    ----------
    password : Tensor or bytes
        Password to verify.
    hash_tensor : Tensor
        Hash from password_hash().

    Returns
    -------
    bool
        True if password matches, False otherwise.

    Examples
    --------
    >>> stored = password_hash(b"correct_password")
    >>> password_verify(b"correct_password", stored)  # True
    >>> password_verify(b"wrong_password", stored)    # False
    """
    if isinstance(password, bytes):
        password = torch.tensor(
            list(password), dtype=torch.uint8, device=hash_tensor.device
        )

    # Extract components
    salt = hash_tensor[:SALT_LENGTH]
    iter_bytes = hash_tensor[SALT_LENGTH : SALT_LENGTH + 4]
    stored_hash = hash_tensor[SALT_LENGTH + 4 :]

    # Decode iterations from big-endian bytes
    iterations = (
        (int(iter_bytes[0].item()) << 24)
        | (int(iter_bytes[1].item()) << 16)
        | (int(iter_bytes[2].item()) << 8)
        | int(iter_bytes[3].item())
    )

    # Recompute hash
    computed = pbkdf2_sha256(password, salt, iterations, HASH_LENGTH)

    # Constant-time comparison (via torch)
    return bool(torch.all(computed == stored_hash).item())

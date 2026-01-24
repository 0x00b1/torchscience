"""Compatibility alias for the legacy `torchscience.encryption` package.

This module re-exports cryptography primitives from `torchscience.cryptography`
to maintain backward compatibility with tests and user imports that still
reference `torchscience.encryption`.
"""

from torchscience.cryptography import (
    ChaCha20Generator,
    chacha20,
    hmac_sha256,
    sha256,
)

__all__ = [
    "chacha20",
    "ChaCha20Generator",
    "hmac_sha256",
    "sha256",
]

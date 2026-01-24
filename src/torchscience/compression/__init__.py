"""Lossless and lossy compression operators."""

from ._run_length import run_length_decode, run_length_encode

__all__ = [
    "run_length_decode",
    "run_length_encode",
]

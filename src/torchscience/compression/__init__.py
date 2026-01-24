"""Lossless and lossy compression operators."""

from ._huffman import huffman_decode, huffman_encode
from ._lz77 import lz77_decode, lz77_encode
from ._run_length import run_length_decode, run_length_encode

__all__ = [
    "huffman_decode",
    "huffman_encode",
    "lz77_decode",
    "lz77_encode",
    "run_length_decode",
    "run_length_encode",
]

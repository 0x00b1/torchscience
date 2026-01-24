"""Lossless and lossy compression operators."""

from ._arithmetic import arithmetic_decode, arithmetic_encode
from ._huffman import huffman_decode, huffman_encode
from ._lz77 import lz77_decode, lz77_encode
from ._range import range_decode, range_encode
from ._run_length import run_length_decode, run_length_encode

__all__ = [
    "arithmetic_decode",
    "arithmetic_encode",
    "huffman_decode",
    "huffman_encode",
    "lz77_decode",
    "lz77_encode",
    "range_decode",
    "range_encode",
    "run_length_decode",
    "run_length_encode",
]

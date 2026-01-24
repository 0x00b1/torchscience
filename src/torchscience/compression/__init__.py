"""Lossless and lossy compression operators."""

from ._arithmetic import arithmetic_decode, arithmetic_encode
from ._distortion_measure import distortion_measure
from ._dithered_quantize import dithered_quantize
from ._huffman import huffman_decode, huffman_encode
from ._lz77 import lz77_decode, lz77_encode
from ._range import range_decode, range_encode
from ._rans import rans_decode, rans_encode
from ._rate_distortion_lagrangian import (
    estimate_bitrate,
    rate_distortion_lagrangian,
)
from ._run_length import run_length_decode, run_length_encode
from ._scalar_quantize import scalar_quantize
from ._vector_quantize import vector_quantize

__all__ = [
    "arithmetic_decode",
    "arithmetic_encode",
    "distortion_measure",
    "dithered_quantize",
    "estimate_bitrate",
    "huffman_decode",
    "huffman_encode",
    "lz77_decode",
    "lz77_encode",
    "range_decode",
    "range_encode",
    "rans_decode",
    "rans_encode",
    "rate_distortion_lagrangian",
    "run_length_decode",
    "run_length_encode",
    "scalar_quantize",
    "vector_quantize",
]

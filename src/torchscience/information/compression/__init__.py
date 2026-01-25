"""Lossless and lossy compression operators."""

from ._arithmetic import arithmetic_decode, arithmetic_encode
from ._distortion_measure import distortion_measure
from ._dithered_quantize import dithered_quantize
from ._entropy_bottleneck import EntropyBottleneck, entropy_bottleneck
from ._gaussian_conditional import GaussianConditional, gaussian_conditional
from ._huffman import huffman_decode, huffman_encode
from ._importance_map import gain_unit, importance_map
from ._lattice_quantize import lattice_quantize
from ._lz77 import lz77_decode, lz77_encode
from ._noise_quantize import noise_quantize, soft_round, ste_round
from ._optimal_quantizer import optimal_quantizer
from ._perceptual_loss import perceptual_loss, rate_loss
from ._range import range_decode, range_encode
from ._rans import rans_decode, rans_encode
from ._rate_distortion_lagrangian import (
    estimate_bitrate,
    rate_distortion_lagrangian,
)
from ._run_length import run_length_decode, run_length_encode
from ._scalar_quantize import scalar_quantize
from ._transform_code import transform_code
from ._vector_quantize import vector_quantize

__all__ = [
    "arithmetic_decode",
    "arithmetic_encode",
    "distortion_measure",
    "dithered_quantize",
    "EntropyBottleneck",
    "entropy_bottleneck",
    "estimate_bitrate",
    "GaussianConditional",
    "gaussian_conditional",
    "huffman_decode",
    "huffman_encode",
    "gain_unit",
    "importance_map",
    "lattice_quantize",
    "lz77_decode",
    "lz77_encode",
    "noise_quantize",
    "optimal_quantizer",
    "perceptual_loss",
    "range_decode",
    "range_encode",
    "rans_decode",
    "rans_encode",
    "rate_distortion_lagrangian",
    "rate_loss",
    "run_length_decode",
    "run_length_encode",
    "scalar_quantize",
    "soft_round",
    "ste_round",
    "transform_code",
    "vector_quantize",
]

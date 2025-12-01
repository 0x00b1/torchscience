import hashlib
import math
from typing import Tuple

import torch
from torch import Tensor

from torchscience.encryption._chacha20 import chacha20


class ChaCha20Generator:
    """Cryptographically secure random number generator using ChaCha20."""

    def __init__(
        self,
        seed: int | Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._device = torch.device(device if device is not None else "cpu")
        self._counter = 0
        self._nonce = torch.zeros(12, dtype=torch.uint8, device=self._device)

        if seed is None:
            self._key = torch.randint(
                0, 256, (32,), dtype=torch.uint8, device=self._device
            )
        elif isinstance(seed, int):
            self.manual_seed(seed)
        elif isinstance(seed, Tensor):
            if seed.shape != (32,) or seed.dtype != torch.uint8:
                raise ValueError("Tensor seed must be (32,) uint8")
            self._key = seed.to(device=self._device)
        else:
            raise TypeError(
                f"seed must be int, Tensor, or None, got {type(seed)}"
            )

    def manual_seed(self, seed: int) -> "ChaCha20Generator":
        seed_bytes = seed.to_bytes(
            (seed.bit_length() + 7) // 8 or 1,
            byteorder="little",
            signed=(seed < 0),
        )
        key_bytes = hashlib.sha256(seed_bytes).digest()
        self._key = torch.tensor(
            list(key_bytes), dtype=torch.uint8, device=self._device
        )
        self._counter = 0
        return self

    def random_bytes(self, num_bytes: int) -> Tensor:
        output = chacha20(
            self._key, self._nonce, num_bytes=num_bytes, counter=self._counter
        )
        self._counter += (num_bytes + 63) // 64
        return output

    def random(
        self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        numel = math.prod(size)
        if dtype == torch.float64:
            bytes_needed = numel * 8
            raw = self.random_bytes(bytes_needed)
            raw_64 = raw.view(torch.int64)
            raw_64 = raw_64 & ((1 << 53) - 1)
            result = raw_64.to(torch.float64) / (1 << 53)
        elif dtype == torch.float32:
            bytes_needed = numel * 4
            raw = self.random_bytes(bytes_needed)
            raw_32 = raw.view(torch.int32)
            raw_32 = raw_32 & ((1 << 24) - 1)
            result = raw_32.to(torch.float32) / (1 << 24)
        elif dtype == torch.float16:
            bytes_needed = numel * 2
            raw = self.random_bytes(bytes_needed)
            raw_16 = raw.view(torch.int16)
            raw_16 = raw_16 & ((1 << 11) - 1)
            result = raw_16.to(torch.float16) / (1 << 11)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return result.view(size)

    def randn(
        self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        numel = math.prod(size)
        n_pairs = (numel + 1) // 2
        u1 = self.random((n_pairs,), dtype=dtype)
        u2 = self.random((n_pairs,), dtype=dtype)
        u1 = u1.clamp(min=1e-10)
        r = torch.sqrt(-2 * torch.log(u1))
        theta = 2 * math.pi * u2
        z0 = r * torch.cos(theta)
        z1 = r * torch.sin(theta)
        result = torch.stack([z0, z1], dim=-1).flatten()[:numel]
        return result.view(size)

    def get_state(self) -> Tensor:
        counter_tensor = torch.tensor(
            [self._counter], dtype=torch.int64, device=self._device
        )
        return torch.cat([self._key.to(torch.int64), counter_tensor])

    def set_state(self, state: Tensor) -> None:
        self._key = state[:32].to(torch.uint8)
        self._counter = state[32].item()

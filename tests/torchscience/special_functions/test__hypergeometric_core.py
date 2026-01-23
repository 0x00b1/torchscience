import math
import sys
from pathlib import Path

import importlib.util
import torch

REPO_SRC = Path(__file__).resolve().parents[3] / "src"


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


_hcore = _load_module(
    "torchscience.special_functions._hypergeometric_core",
    REPO_SRC / "torchscience" / "special_functions" / "_hypergeometric_core.py",
)

_pfq_series = _hcore._pfq_series


def test__pfq_series__0f1_matches_direct_series_small_z():
    # Validate 0F1(;b;z) against a direct partial sum for small z
    dtype = torch.float64
    device = torch.device("cpu")
    z = torch.tensor([0.0, 1e-6, -1e-6, 1e-4, -1e-4], dtype=dtype, device=device)
    b = torch.tensor(1.75, dtype=dtype, device=device)

    def direct_0f1(b_val: float, z_tensor: torch.Tensor, terms: int = 50) -> torch.Tensor:
        out = torch.zeros_like(z_tensor)
        # Sum_{k=0}^{N-1} z^k / (k! * (b)_k)
        for idx in range(z_tensor.numel()):
            z_i = z_tensor.flatten()[idx]
            s = torch.tensor(0.0, dtype=dtype)
            poch = torch.tensor(1.0, dtype=dtype)
            fact = torch.tensor(1.0, dtype=dtype)
            zpow = torch.tensor(1.0, dtype=dtype)
            for k in range(terms):
                if k > 0:
                    fact = fact * k
                    poch = poch * (b_val + (k - 1))
                    zpow = zpow * z_i
                s = s + zpow / (fact * poch)
            out.view(-1)[idx] = s
        return out

    val_series = _pfq_series(a_params=None, b_params=(b,), z=z, tol=1e-16, max_terms=2000)
    val_direct = direct_0f1(float(b.item()), z, terms=50)

    assert torch.allclose(val_series, val_direct, rtol=1e-12, atol=1e-14)


def test__pfq_series__1f1_terminating_series():
    # M(a,b,z) with a = -n terminates (polynomial of degree n)
    dtype = torch.float64
    z = torch.linspace(-2.0, 2.0, steps=11, dtype=dtype)
    a = -2.0
    b = 3.5

    # Series for M(-2,b,z) = 1 + (-2/b) z + [(-2)(-1)/(b(b+1))] z^2 / 2!
    expected = 1.0 + (-2.0 / b) * z + ((2.0) / (b * (b + 1.0))) * (z * z) / 2.0

    val = _pfq_series(a_params=(a,), b_params=(b,), z=z, tol=1e-15, max_terms=50)

    assert torch.allclose(val, expected, rtol=1e-14, atol=1e-14)

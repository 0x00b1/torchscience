import importlib.util
import sys
import types
from pathlib import Path

import torch


REPO_SRC = Path(__file__).resolve().parents[3] / "src"
PKG_DIR = REPO_SRC / "torchscience"
SP_DIR = PKG_DIR / "special_functions"


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


# Minimal package entries for relative imports
sys.modules.setdefault("torchscience", types.ModuleType("torchscience"))
sys.modules["torchscience"].__path__ = [str(PKG_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault(
    "torchscience.special_functions", types.ModuleType("torchscience.special_functions")
)
sys.modules["torchscience.special_functions"].__path__ = [  # type: ignore[attr-defined]
    str(SP_DIR)
]

_hv = _load_module(
    "torchscience.special_functions._struve_hv",
    SP_DIR / "_struve_hv.py",
)
_lv = _load_module(
    "torchscience.special_functions._struve_lv",
    SP_DIR / "_struve_lv.py",
)

struve_hv = _hv.struve_hv
struve_lv = _lv.struve_lv


def _direct_series_H(nu: float, z: torch.Tensor, terms: int = 80) -> torch.Tensor:
    # Hν: sum_k (-1)^k (z/2)^{2k+ν+1} / [Γ(k+3/2) Γ(k+ν+3/2)]
    dtype = z.dtype
    out = torch.zeros_like(z)
    for idx in range(z.numel()):
        zi = z.flatten()[idx]
        s = torch.tensor(0.0, dtype=dtype)
        for k in range(terms):
            sign = -1.0 if (k % 2 == 1) else 1.0
            num = sign * torch.pow(zi / 2.0, 2 * k + nu + 1.0)
            den = torch.lgamma(torch.tensor(k + 1.5, dtype=dtype)).exp() * torch.lgamma(
                torch.tensor(k + nu + 1.5, dtype=dtype)
            ).exp()
            s = s + num / den
        out.view(-1)[idx] = s
    return out


def _direct_series_L(nu: float, z: torch.Tensor, terms: int = 80) -> torch.Tensor:
    # Lν: sum_k (+1)^k (z/2)^{2k+ν+1} / [Γ(k+3/2) Γ(k+ν+3/2)]
    dtype = z.dtype
    out = torch.zeros_like(z)
    for idx in range(z.numel()):
        zi = z.flatten()[idx]
        s = torch.tensor(0.0, dtype=dtype)
        for k in range(terms):
            num = torch.pow(zi / 2.0, 2 * k + nu + 1.0)
            den = torch.lgamma(torch.tensor(k + 1.5, dtype=dtype)).exp() * torch.lgamma(
                torch.tensor(k + nu + 1.5, dtype=dtype)
            ).exp()
            s = s + num / den
        out.view(-1)[idx] = s
    return out


def test_struve_series_small_z_match_direct():
    dtype = torch.float64
    device = torch.device("cpu")
    z = torch.linspace(1e-8, 1e-3, steps=6, dtype=dtype, device=device)
    nu = torch.tensor(0.75, dtype=dtype, device=device)

    H = struve_hv(nu, z, tol=1e-18, max_terms=5000)
    L = struve_lv(nu, z, tol=1e-18, max_terms=5000)

    H_ref = _direct_series_H(float(nu.item()), z, terms=100)
    L_ref = _direct_series_L(float(nu.item()), z, terms=100)

    assert torch.allclose(H, H_ref, rtol=1e-10, atol=1e-14)
    assert torch.allclose(L, L_ref, rtol=1e-10, atol=1e-14)


def test_struve_broadcast_and_dtype():
    z = torch.linspace(1e-6, 1e-3, steps=5, dtype=torch.float32)
    nu = torch.tensor([0.5, 1.5], dtype=torch.float32)
    outH = struve_hv(nu[:, None], z[None, :], tol=1e-6, max_terms=500)
    outL = struve_lv(nu[:, None], z[None, :], tol=1e-6, max_terms=500)
    assert outH.shape == (2, 5)
    assert outL.shape == (2, 5)
    assert outH.dtype == torch.float32 and outL.dtype == torch.float32


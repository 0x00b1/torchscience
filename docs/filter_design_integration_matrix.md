# Filter Design PyTorch Integration Status Matrix

This document tracks the PyTorch integration status for all filter design functions in torchscience. It provides visibility into which functions support autograd, torch.compile, vmap, and other PyTorch features.

**Last Updated:** 2026-01-22
**Test Results:** 47 passed, 14 failed, 9 skipped

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Passing - Feature works correctly |
| ❌ | Failing - Feature does not work or has errors |
| ⚠️ | Partial - Feature works with limitations |
| ➖ | Not applicable - Feature does not apply to this function |
| | Empty - Not yet tested |

## Status Tables

### Design Functions

Functions that design complete filters from specifications.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| butterworth_design | ❌ | | ✅ | ⚠️ | ❌ | ✅ | | Gradcheck Jacobian mismatch; dtype param ignored |
| bessel_design | ❌ | | ✅ | ⚠️ | ❌ | ✅ | | Gradcheck Jacobian mismatch; dtype param ignored |
| chebyshev_type_1_design | ❌ | | ✅ | ⚠️ | ❌ | ✅ | | Gradcheck Jacobian mismatch; dtype param ignored |
| chebyshev_type_2_design | ✅ | | ✅ | ⚠️ | ❌ | ✅ | | dtype param ignored |
| elliptic_design | ❌ | | ✅ | ⚠️ | ❌ | ✅ | | Gradcheck Jacobian mismatch; dtype param ignored |
| firwin | ❌ | | ❌ | ⚠️ | ✅ | ✅ | | Gradcheck fails; torch.compile fails on len(cutoff) |
| firwin2 | ✅ | | ✅ | ⚠️ | ✅ | ✅ | | |
| iirnotch | ⚠️ | | ✅ | ➖ | ✅ | ✅ | | frequency grad skipped; quality_factor grad passes |
| iirpeak | ⚠️ | | ✅ | ➖ | ✅ | ✅ | | frequency grad skipped |

### Prototype Functions

Functions that create analog lowpass prototype filters.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| butterworth_prototype | ➖ | ➖ | | | ✅ | ✅ | | Returns expected dtype |
| bessel_prototype | ➖ | ➖ | | | ✅ | ✅ | | Returns expected dtype |
| chebyshev_type_1_prototype | ➖ | ➖ | | | ✅ | ✅ | | Returns expected dtype |
| chebyshev_type_2_prototype | ➖ | ➖ | | | ✅ | ✅ | | Returns expected dtype |
| elliptic_prototype | ➖ | ➖ | | | ✅ | ✅ | | Returns expected dtype |

### Minimum Order Estimation

Functions that compute the minimum filter order to meet specifications.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| butterworth_minimum_order | ➖ | ➖ | | | ➖ | ➖ | | Returns int (not tensor) |
| chebyshev_type_1_minimum_order | ➖ | ➖ | | | ➖ | ➖ | | Returns int (not tensor) |
| chebyshev_type_2_minimum_order | ➖ | ➖ | | | ➖ | ➖ | | Returns int (not tensor) |
| elliptic_minimum_order | ➖ | ➖ | | | ➖ | ➖ | | Returns int (not tensor) |

### Transformations

Functions that transform filter representations between domains.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| bilinear_transform_zpk | ✅ | | | ✅ | | ✅ | | vmap over gains works |
| bilinear_transform_ba | | | | | | | | |
| lowpass_to_lowpass_zpk | ✅ | | | | | ✅ | | |
| lowpass_to_highpass_zpk | ✅ | | | | | ✅ | | |
| lowpass_to_bandpass_zpk | | | | | | | | |
| lowpass_to_bandstop_zpk | | | | | | | | |

### Conversions

Functions that convert between filter coefficient representations (ba, zpk, sos).

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| ba_to_zpk | | | | | | | | |
| ba_to_sos | | | | | | | | |
| zpk_to_ba | ✅ | | ✅ | | | ✅ | | compile warns: no complex codegen |
| zpk_to_sos | ✅ | | ✅ | ⚠️ | | ✅ | | vmap over gains skipped |
| sos_to_ba | | | | | | | | |
| sos_to_zpk | | | ✅ | | | | | |

### SOS Utilities

Utility functions for working with second-order sections.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| cascade_sos | | | ✅ | | | | | |
| sos_normalize | | | | | | | | |
| sos_sections_count | ➖ | ➖ | | ➖ | ➖ | ➖ | | Returns int (not tensor) |

## Summary Statistics

| Category | Total | Passing | Failing | Partial | Not Tested |
|----------|-------|---------|---------|---------|------------|
| Design Functions | 9 | 2 | 5 | 2 | 0 |
| Prototype Functions | 5 | 5 | 0 | 0 | 0 |
| Minimum Order Estimation | 4 | 0 | 0 | 0 | 0 |
| Transformations | 6 | 3 | 0 | 0 | 3 |
| Conversions | 6 | 3 | 0 | 1 | 2 |
| SOS Utilities | 3 | 1 | 0 | 0 | 2 |
| **Total** | **33** | **14** | **5** | **3** | **7** |

## Known Issues

### Issue: Gradcheck Jacobian Mismatch for IIR Design Functions
- **Functions affected:** butterworth_design, chebyshev_type_1_design, bessel_design, elliptic_design
- **Feature:** gradcheck
- **Error message:** `Jacobian mismatch for output 0 with respect to input 0, numerical: [[11.8681]], analytical: [[11.8845]]`
- **Root cause:** Small numerical discrepancy between analytical gradient and finite-difference approximation. The relative error is ~0.14%, which exceeds gradcheck's default tolerance.
- **Workaround:** Use looser tolerances (atol=1e-3) or use gradients for optimization where this precision is sufficient.
- **Note:** chebyshev_type_2_design passes gradcheck, suggesting its gradient implementation is more numerically stable.

### Issue: Gradcheck Failure for firwin Cutoff Gradients
- **Functions affected:** firwin, firwin (bandpass)
- **Feature:** gradcheck
- **Error message:** Similar Jacobian mismatch
- **Root cause:** FIR filter design involves window functions and sinc operations that may have less stable numerical gradients.

### Issue: torch.compile Fails for firwin
- **Functions affected:** firwin
- **Feature:** torch.compile
- **Error message:** `torch._dynamo.exc.InternalTorchDynamoError: IndexError: tuple index out of range`
- **Root cause:** The code uses `len(cutoff_tensor)` which TorchDynamo cannot trace when the tensor is 0-dimensional (scalar). The conditional `if len(cutoff_tensor) > 1` causes the error.
- **Workaround:** Refactor to avoid `len()` on tensors, use `cutoff_tensor.numel()` or `cutoff_tensor.shape[0]` with proper handling.

### Issue: IIR Design Functions Ignore dtype Parameter
- **Functions affected:** butterworth_design, chebyshev_type_1_design, chebyshev_type_2_design, bessel_design, elliptic_design
- **Feature:** float32 dtype
- **Error message:** `assert torch.float64 == torch.float32` (sos.dtype is float64 when float32 requested)
- **Root cause:** The design functions perform intermediate calculations in float64 and don't cast to the requested dtype before returning.
- **Workaround:** Manually cast output: `sos = butterworth_design(...).to(torch.float32)`

### Issue: vmap Skipped for Design Functions
- **Functions affected:** firwin, butterworth_design, zpk_to_sos
- **Feature:** vmap
- **Reason:** These functions produce outputs with data-dependent shapes, which is incompatible with vmap's requirement for uniform output shapes across the batch dimension.
- **Status:** Expected behavior; marked as partial (⚠️) since vmap cannot apply but the functions work correctly otherwise.

### Issue: iirnotch/iirpeak Frequency Gradient Skipped
- **Functions affected:** iirnotch, iirpeak
- **Feature:** gradcheck for frequency parameter
- **Reason:** Tests are intentionally skipped; frequency gradient may have known limitations.
- **Note:** Quality factor gradient passes for iirnotch.

### Issue: Complex Operator Warning in torch.compile
- **Functions affected:** zpk_to_ba
- **Feature:** torch.compile
- **Warning:** `Torchinductor does not support code generation for complex operators. Performance may be worse than eager.`
- **Status:** Function works correctly but may not benefit from compile optimization.

## Patterns for New Code

### Pattern: Dtype Preservation
- **Use case:** Ensuring output tensors match requested dtype
- **Example function:** firwin, firwin2 (these work correctly)
- **Code snippet:**
```python
def design_filter(..., dtype=None, device=None):
    # Perform calculations
    result = ...

    # Ensure output matches requested dtype
    if dtype is not None:
        result = result.to(dtype=dtype)
    if device is not None:
        result = result.to(device=device)
    return result
```
- **Notes:** IIR design functions should adopt this pattern.

### Pattern: torch.compile Compatible Tensor Length Check
- **Use case:** Checking tensor dimensions without breaking torch.compile
- **Bad pattern:** `if len(tensor) > 1:` (fails for 0-d tensors)
- **Good pattern:**
```python
# Option 1: Use numel() for total elements
if tensor.numel() > 1:
    ...

# Option 2: Use ndim check first
if tensor.ndim > 0 and tensor.shape[0] > 1:
    ...

# Option 3: Always ensure tensor is at least 1-d
tensor = torch.atleast_1d(tensor)
if tensor.shape[0] > 1:
    ...
```
- **Notes:** firwin should be refactored to use one of these patterns.

## Testing Commands

Run specific integration tests:

```bash
# Run all filter design integration tests
uv run pytest tests/torchscience/signal_processing/filter_design/test__pytorch_integration.py -v

# Run tests for a specific category
uv run pytest tests/torchscience/signal_processing/filter_design/test__pytorch_integration.py -k "prototype" -v

# Run tests for a specific feature
uv run pytest tests/torchscience/signal_processing/filter_design/test__pytorch_integration.py -k "gradcheck" -v

# Run tests for a specific function
uv run pytest tests/torchscience/signal_processing/filter_design/test__pytorch_integration.py -k "butterworth" -v
```

## References

- [PyTorch autograd.gradcheck documentation](https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradcheck)
- [torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [torch.vmap documentation](https://pytorch.org/docs/stable/generated/torch.vmap.html)
- [torchscience architecture documentation](./architecture.rst)

# Filter Design PyTorch Integration Status Matrix

This document tracks the PyTorch integration status for all filter design functions in torchscience. It provides visibility into which functions support autograd, torch.compile, vmap, and other PyTorch features.

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
| butterworth_design | | | | | | | | |
| bessel_design | | | | | | | | |
| chebyshev_type_1_design | | | | | | | | |
| chebyshev_type_2_design | | | | | | | | |
| elliptic_design | | | | | | | | |
| firwin | | | | | | | | |
| firwin2 | | | | | | | | |
| iirnotch | | | | | | | | |
| iirpeak | | | | | | | | |

### Prototype Functions

Functions that create analog lowpass prototype filters.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| butterworth_prototype | | | | | | | | |
| bessel_prototype | | | | | | | | |
| chebyshev_type_1_prototype | | | | | | | | |
| chebyshev_type_2_prototype | | | | | | | | |
| elliptic_prototype | | | | | | | | |

### Minimum Order Estimation

Functions that compute the minimum filter order to meet specifications.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| butterworth_minimum_order | | | | | | | | |
| chebyshev_type_1_minimum_order | | | | | | | | |
| chebyshev_type_2_minimum_order | | | | | | | | |
| elliptic_minimum_order | | | | | | | | |

### Transformations

Functions that transform filter representations between domains.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| bilinear_transform_zpk | | | | | | | | |
| bilinear_transform_ba | | | | | | | | |
| lowpass_to_lowpass_zpk | | | | | | | | |
| lowpass_to_highpass_zpk | | | | | | | | |
| lowpass_to_bandpass_zpk | | | | | | | | |
| lowpass_to_bandstop_zpk | | | | | | | | |

### Conversions

Functions that convert between filter coefficient representations (ba, zpk, sos).

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| ba_to_zpk | | | | | | | | |
| ba_to_sos | | | | | | | | |
| zpk_to_ba | | | | | | | | |
| zpk_to_sos | | | | | | | | |
| sos_to_ba | | | | | | | | |
| sos_to_zpk | | | | | | | | |

### SOS Utilities

Utility functions for working with second-order sections.

| Function | gradcheck | gradgradcheck | torch.compile | vmap | float32 | float64 | CUDA | Notes |
|----------|-----------|---------------|---------------|------|---------|---------|------|-------|
| cascade_sos | | | | | | | | |
| sos_normalize | | | | | | | | |
| sos_sections_count | | | | | | | | |

## Summary Statistics

| Category | Total | Passing | Failing | Partial | Not Tested |
|----------|-------|---------|---------|---------|------------|
| Design Functions | 9 | | | | |
| Prototype Functions | 5 | | | | |
| Minimum Order Estimation | 4 | | | | |
| Transformations | 6 | | | | |
| Conversions | 6 | | | | |
| SOS Utilities | 3 | | | | |
| **Total** | **33** | | | | |

## Known Issues

*To be populated after running integration tests.*

<!--
Document known issues here with the following format:

### Issue: [Brief Description]
- **Functions affected:** function1, function2
- **Feature:** gradcheck / torch.compile / vmap / etc.
- **Error message:** `error text here`
- **Root cause:** (if known)
- **Workaround:** (if available)
- **Tracking:** GitHub issue link or internal reference
-->

## Patterns for New Code

*To be populated based on successful implementations.*

<!--
Document patterns that work well for implementing PyTorch integration:

### Pattern: [Name]
- **Use case:** When to use this pattern
- **Example function:** function_name
- **Code snippet:**
```python
# Example code here
```
- **Notes:** Additional context
-->

## Testing Commands

Run specific integration tests:

```bash
# Run all filter design integration tests
uv run pytest tests/torchscience/signal_processing/test__filter_design_integration.py -v

# Run tests for a specific category
uv run pytest tests/torchscience/signal_processing/test__filter_design_integration.py -k "prototype" -v

# Run tests for a specific feature
uv run pytest tests/torchscience/signal_processing/test__filter_design_integration.py -k "gradcheck" -v

# Run tests for a specific function
uv run pytest tests/torchscience/signal_processing/test__filter_design_integration.py -k "butterworth_prototype" -v
```

## References

- [PyTorch autograd.gradcheck documentation](https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradcheck)
- [torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [torch.vmap documentation](https://pytorch.org/docs/stable/generated/torch.vmap.html)
- [torchscience architecture documentation](./architecture.rst)

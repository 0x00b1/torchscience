import hypothesis.strategies
import torch

all_floating_dtypes = hypothesis.strategies.sampled_from(
    [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)

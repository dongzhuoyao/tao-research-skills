# Triton Fused Operations

## Fused Add + LayerNorm (Triton)

Fuse residual addition with normalization into a single GPU kernel. Eliminates an extra memory read/write pass:

```python
# Bad: two separate ops = two memory passes
residual = residual + self.drop_path(x)
x = self.norm(residual)

# Good: single fused kernel
from mamba_ssm.ops.triton.layernorm import rms_norm_fn, layer_norm_fn

fused_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
x, residual = fused_fn(
    self.drop_path(x), self.norm.weight, self.norm.bias,
    residual=residual, prenorm=True,
    residual_in_fp32=True,  # keep residual stream in FP32
    eps=self.norm.eps,
)
```

When writing your own Triton kernels, use `@triton.autotune` over warp counts for forward/backward:
```python
@triton.autotune(
    configs=[triton.Config({"BLOCK_N": 512}, num_warps=w) for w in [1, 2, 4, 8, 16, 32]],
    key=["N"],
)
@triton.jit
def _layer_norm_fwd_kernel(...):
    ...
```

## Residual in FP32

Keep the residual stream in FP32 even during mixed-precision training. Prevents numerical drift in deep networks:

```python
class Block(nn.Module):
    def __init__(self, ..., residual_in_fp32=True):
        self.residual_in_fp32 = residual_in_fp32

    def forward(self, hidden, residual=None):
        # Fused norm handles fp32 residual internally
        # Or manually:
        if self.residual_in_fp32:
            residual = residual.to(torch.float32) if residual is not None else None
```

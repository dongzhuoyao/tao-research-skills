# Custom Autograd with AMP

## Custom Autograd with AMP

When writing custom `torch.autograd.Function` that must work with mixed precision, use `@custom_fwd`/`@custom_bwd` and explicit dtype casting:

```python
from torch.cuda.amp import custom_fwd, custom_bwd

class FusedOp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias):
        # Explicitly cast to autocast dtype when autocast is active
        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
        out = custom_cuda_kernel.fwd(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        return custom_cuda_kernel.bwd(grad_out, x, weight, bias)
```

Without `@custom_fwd`/`@custom_bwd`, custom ops silently run in FP32 inside autocast regions, killing throughput.

## Multi-Tier Attention Fallback

For custom attention (e.g. cross-attention with conditioning), use a 3-tier fallback:

```python
if hasattr(F, "scaled_dot_product_attention"):
    ATTN_MODE = "flash"          # PyTorch 2.0+ (dispatches to FlashAttention2)
else:
    try:
        import xformers.ops
        ATTN_MODE = "xformers"   # xformers memory-efficient attention
    except ImportError:
        ATTN_MODE = "math"       # Naive O(N^2) fallback

# In forward:
if ATTN_MODE == "flash":
    x = F.scaled_dot_product_attention(q, k, v)
elif ATTN_MODE == "xformers":
    x = xformers.ops.memory_efficient_attention(q, k, v)
else:
    attn = (q @ k.transpose(-2, -1)) * self.scale
    x = attn.softmax(dim=-1) @ v
```

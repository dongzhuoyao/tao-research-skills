---
name: gpu-training-acceleration
description: Use when optimizing PyTorch training speed or memory on CUDA GPUs — global flags, torch.compile, fused optimizers, mixed precision, gradient checkpointing, kernel fusion, memory layout, or latent-space training. Applies to any PyTorch training workload.
---

# GPU Training Acceleration

## When to Use

- Starting a new PyTorch training pipeline and want maximum GPU utilization
- Debugging slow training throughput (GPU util < 90%)
- Running out of GPU memory and need to trade compute for memory
- Deciding whether to use torch.compile and how to apply it safely
- Choosing optimizer kernels (fused vs standard)
- Setting up mixed precision (bf16/fp16)
- Writing custom CUDA/Triton kernels that interact with AMP
- Pre-computing encoder features to skip expensive forward passes during training
- For FID evaluation during training, see `fid-evaluation` skill

## Core Principles

### Config-Gated Acceleration

Every acceleration flag must be controlled by config, not hardcoded. Features like `torch.compile` can break on specific PyTorch versions or model architectures. Config gating lets you toggle without code changes:

```yaml
runtime:
  compile: false        # torch.compile on decoder sub-modules
  fused_adamw: true     # fused AdamW kernel
  precision: bf16       # mixed precision dtype
```

```python
# Good: config-gated
if cfg.training.runtime.compile:
    model.decoder = torch.compile(model.decoder)

# Bad: always-on
model.decoder = torch.compile(model.decoder)  # Breaks when Inductor has bugs
```

### Compile Sub-Modules, Not Full Models

`torch.compile` on the full model triggers recompilation when input shapes change (common with variable-length audio/text). Compile only the fixed-shape decoder sub-modules:

```python
# Good: compile specific sub-modules with stable shapes
model.ctc_adapter = torch.compile(model.ctc_adapter)
model.ar_decoder = torch.compile(model.ar_decoder)
# Leave encoder (variable input length) uncompiled

# Bad: compile entire model
model = torch.compile(model)  # Recompiles on every new input length
```

### Log Acceleration State

Always log which acceleration features are active — essential for debugging throughput differences between runs:

```python
log.info("Acceleration: compile=%s fused_adamw=%s amp=%s amp_dtype=%s",
         compile_enabled, fused_adamw, amp_enabled, amp_dtype)
```

## Patterns

### Global CUDA Flags

Set these once at startup before any CUDA operations:

```python
def set_cuda_acceleration_flags():
    """Enable hardware acceleration features. Call before model creation."""
    # TF32 for matmul (3x faster than FP32, negligible accuracy loss)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    # cuDNN TF32 + autotuner
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Autotuner picks fastest conv algorithm

    # Dynamo cache (for torch.compile)
    torch._dynamo.config.cache_size_limit = 64
```

### Fused AdamW

Single CUDA kernel per optimizer step instead of per-parameter:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    fused=True,  # Requires CUDA, ~10-15% optimizer step speedup
)
```

### Flash Attention (Zero Code)

PyTorch 2.1+ `nn.TransformerEncoder`/`Decoder` auto-dispatch to SDPA/Flash Attention. No code needed — just use the standard modules:

```python
# This automatically uses Flash Attention when available
encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
```

### Mixed Precision with Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer = accelerator.prepare(model, optimizer)
# Forward/backward automatically uses bf16 where safe
```

### GPU Memory Monitoring

Use `memory_reserved()` to match `nvidia-smi`, not `memory_allocated()`:

```python
# memory_allocated(): active tensors only (~5 GB typical)
# memory_reserved(): caching allocator pool (~58 GB, matches nvidia-smi)
gpu_mem_gb = torch.cuda.memory_reserved() / 1e9
```

### Handling torch.compile Failures

`torch.compile` Inductor bugs are version-specific and non-deterministic. Pattern:

1. Gate behind config flag (`compile: true/false`)
2. Default to `false` until verified stable on your PyTorch version
3. Test in a fastrun before enabling in fullrun
4. Log compile state so you can correlate with speed differences

```python
if cfg.training.runtime.compile:
    try:
        model.decoder = torch.compile(model.decoder)
        log.info("torch.compile enabled on decoder")
    except Exception as e:
        log.warning("torch.compile failed, continuing without: %s", e)
        cfg.training.runtime.compile = False
```

### Gradient Checkpointing

Trade ~10% speed for ~50% memory. Wrap repeated blocks (transformer layers, SSM blocks) so activations are recomputed in backward instead of stored:

```python
# Per-block checkpointing (preferred — fine-grained control)
for block in self.blocks:
    if self.use_checkpoint:
        hidden, residual = torch.utils.checkpoint.checkpoint(
            block, hidden, residual, cond, use_reentrant=False
        )
    else:
        hidden, residual = block(hidden, residual=residual, c=cond)
```

Config-gate it:
```yaml
runtime:
  gradient_checkpointing: true  # ~50% memory reduction, ~10% slower
```

### Fused Add + LayerNorm (Triton)

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

### Residual in FP32

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

### Custom Autograd with AMP

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

### Multi-Tier Attention Fallback

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

### Memory Layout & In-Place Operations

CUDA kernels require contiguous memory. Enforce it after permutations, and prefer in-place ops to reduce allocations:

```python
# Enforce contiguous after index/permute ops (required by custom CUDA kernels)
if x.stride(-1) != 1:
    x = x.contiguous()

# In-place ops reduce peak memory
x = x.mul_(0.18215)          # Not x = x * 0.18215
ema_param.mul_(decay).add_(param.data, alpha=1 - decay)  # EMA update
```

### Strategic `empty_cache()`

Call before memory-intensive phases (FID sampling, evaluation) to reclaim the caching allocator pool:

```python
# Before evaluation/sampling that needs a large contiguous allocation
torch.cuda.empty_cache()
fid_samples = generate_samples(model, n=50000)
torch.cuda.empty_cache()  # Reclaim after
```

Don't sprinkle `empty_cache()` in training loops — it forces CUDA to re-allocate and hurts throughput.

### Latent Space Training

Pre-compute encoder/VAE features offline, then train on latents. Eliminates the encoder from the training loop entirely:

```python
# Offline pre-computation (run once, store as WebDataset shards)
with torch.no_grad():
    latent = vae.encode(image).latent_dist.sample().mul_(0.18215)
    save_to_shard(latent, label)

# Training loop — just load pre-computed latents
for batch in latent_dataloader:
    x = batch["latent"]  # Already encoded, no VAE forward pass
    loss = model(x, condition)
```

Config pattern:
```yaml
data:
  use_latent: true         # Load pre-computed latents
  latent_scale: 0.18215    # SD VAE scaling factor
training:
  loader: webdataset       # Streaming for pre-computed shards
```

### NVCC Build Optimization

When compiling custom CUDA extensions, use aggressive flags:

```python
# setup.py for custom CUDA kernels
nvcc_flags = [
    "-O3",                         # Maximum optimization
    "--use_fast_math",             # Fast reciprocal/sqrt on GPU
    "--threads", "4",              # Parallel compilation
    "-U__CUDA_NO_HALF_OPERATORS__",      # Enable fp16 ops
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",  # Enable bf16 ops
]
```

`--use_fast_math` trades strict IEEE compliance for speed — acceptable for training, verify for inference.

## Quick Reference

| Technique | Speed Impact | Memory Impact | Config Key |
|-----------|-------------|---------------|------------|
| TF32 matmul/cuDNN | Up to 3x | None | `torch.backends.cuda.matmul.allow_tf32` |
| `cudnn.benchmark` | 5-20% for convs | None | `torch.backends.cudnn.benchmark` |
| Fused AdamW | ~10-15% optim step | None | `fused=True` |
| Mixed precision (bf16/fp16) | ~2x throughput | ~50% less | `mixed_precision: bf16` |
| Flash Attention (SDPA) | 2-4x attention | O(N) vs O(N^2) | Auto in PyTorch 2.0+ |
| torch.compile (sub-modules) | 20-70% on compiled parts | None | `runtime.compile` |
| Gradient checkpointing | ~10% slower | ~50% less | `runtime.gradient_checkpointing` |
| Fused Add+Norm (Triton) | Eliminates memory pass | Less | `fused_add_norm=True` |
| Residual in FP32 | None | Slight overhead | `residual_in_fp32=True` |
| Latent space training | Skip encoder entirely | Much less | `data.use_latent` |
| In-place operations | Marginal | Less peak | Code pattern |
| `--use_fast_math` (NVCC) | Faster kernels | None | Build flag |

## Anti-Patterns

- **Compiling the full model**: Variable-length inputs cause constant recompilation. Compile stable-shape sub-modules only.
- **Always-on torch.compile**: Inductor bugs are PyTorch-version-specific. Gate behind config, default off, test first.
- **Skipping `cudnn.benchmark`**: Free speedup for conv-heavy models. Only skip if input sizes change every batch (rare in practice with padding).
- **Using `memory_allocated()` for GPU monitoring**: Shows ~5 GB when nvidia-smi shows ~58 GB. Use `memory_reserved()`.
- **Hardcoding acceleration flags**: All flags must be in config. Hardcoded flags can't be toggled for debugging or A/B comparison.
- **Forgetting to log acceleration state**: Two runs with different compile/fused settings look identical in W&B unless you log the flags.
- **`empty_cache()` in training loop**: Forces CUDA caching allocator to re-allocate every iteration. Only use before/after memory-intensive phases.
- **Missing `@custom_fwd`/`@custom_bwd`**: Custom autograd functions silently run in FP32 inside autocast, negating mixed-precision gains.
- **Non-contiguous tensors to CUDA kernels**: Silent wrong results or crashes. Always check `stride(-1) == 1` or call `.contiguous()`.
- **Running encoder during training when latents are available**: Pre-compute once, train on latents. The encoder adds zero learning signal.

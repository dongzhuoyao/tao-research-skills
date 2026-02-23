---
name: gpu-training-acceleration
description: Use when optimizing PyTorch training speed or memory on CUDA GPUs — global flags, torch.compile, fused optimizers, mixed precision, gradient checkpointing, kernel fusion, memory layout, or latent-space training. Applies to any PyTorch training workload. Triggers: "torch.compile", "TF32", "fused optimizer", "mixed precision", "bf16", "fp16", "gradient checkpointing", "Triton kernel", "CUDA flags", "GPU slow", "GPU memory"
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
- For generative model evaluation metrics, see `genai-evaluation-metrics` skill

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

Trade ~10% speed for ~50% memory by recomputing activations in backward. Config-gate with `runtime.gradient_checkpointing`.
See [references/gradient-checkpointing.md](references/gradient-checkpointing.md) for implementation.

### Fused Add + LayerNorm (Triton) & Residual in FP32

Fuse residual addition with normalization into a single Triton kernel, and keep the residual stream in FP32 to prevent numerical drift.
See [references/triton-fused-ops.md](references/triton-fused-ops.md) for Triton kernel patterns and FP32 residual implementation.

### Custom Autograd with AMP & Multi-Tier Attention Fallback

Use `@custom_fwd`/`@custom_bwd` for custom autograd ops under mixed precision, and a 3-tier attention fallback (SDPA > xformers > math).
See [references/custom-autograd-amp.md](references/custom-autograd-amp.md) for implementation details.

### Memory Layout, In-Place Operations, `empty_cache()` & NVCC Build

Contiguous memory enforcement, in-place ops for peak memory reduction, strategic `empty_cache()` placement, and NVCC build flags.
See [references/memory-optimization.md](references/memory-optimization.md) for all memory optimization patterns.

### Latent Space Training

Pre-compute encoder/VAE features offline and train on latents to skip the encoder entirely. Config-gate with `data.use_latent`.
See [references/latent-space-training.md](references/latent-space-training.md) for offline pre-computation and config patterns.

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

## See Also

- `genai-evaluation-metrics` — Evaluation metrics during training (memory management)
- `webdataset-streaming` — Latent-space data loading from tar shards
- `wandb-experiment-tracking` — Logging acceleration state to W&B

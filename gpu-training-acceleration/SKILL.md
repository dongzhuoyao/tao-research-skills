---
name: gpu-training-acceleration
description: Use when optimizing PyTorch training speed on CUDA GPUs — setting global flags, torch.compile strategy, fused optimizers, and mixed precision. Applies to any PyTorch training workload.
---

# GPU Training Acceleration

## When to Use

- Starting a new PyTorch training pipeline and want maximum GPU utilization
- Debugging slow training throughput (GPU util < 90%)
- Deciding whether to use torch.compile and how to apply it safely
- Choosing optimizer kernels (fused vs standard)
- Setting up mixed precision (bf16/fp16)

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

## Anti-Patterns

- **Compiling the full model**: Variable-length inputs cause constant recompilation. Compile stable-shape sub-modules only.
- **Always-on torch.compile**: Inductor bugs are PyTorch-version-specific. Gate behind config, default off, test first.
- **Skipping `cudnn.benchmark`**: Free speedup for conv-heavy models. Only skip if input sizes change every batch (rare in practice with padding).
- **Using `memory_allocated()` for GPU monitoring**: Shows ~5 GB when nvidia-smi shows ~58 GB. Use `memory_reserved()`.
- **Hardcoding acceleration flags**: All flags must be in config. Hardcoded flags can't be toggled for debugging or A/B comparison.
- **Forgetting to log acceleration state**: Two runs with different compile/fused settings look identical in W&B unless you log the flags.

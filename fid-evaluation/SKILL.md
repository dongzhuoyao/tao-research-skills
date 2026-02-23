---
name: fid-evaluation
description: Use when setting up FID or other generative metrics (IS, KID, sFID, FDD, FVD) during training — online evaluation, sample count strategy, distributed gather, memory management, OOM during sampling.
---

# FID Evaluation During Training

## When to Use

- Setting up online FID evaluation during generative model training
- Debugging OOM during FID sampling / evaluation phases
- Choosing between FID libraries (torchmetrics vs clean-fid)
- Configuring sample counts for training-time vs final benchmarks
- Running multi-metric evaluation (FID + IS + KID + sFID + FDD)
- Distributed FID computation with HuggingFace Accelerate

## Patterns

### Library: torchmetrics (not clean-fid)

Use `torchmetrics.image.fid.FrechetInceptionDistance` for online FID during training. It handles distributed sync automatically:

```python
from torchmetrics.image.fid import FrechetInceptionDistance

fid_metric = FrechetInceptionDistance(
    feature=2048,              # InceptionV3 pool layer
    reset_real_features=True,  # Recompute real features each eval
    normalize=False,           # Pass uint8 [0,255], not float [0,1]
    sync_on_compute=True,      # Auto DDP sync
).to(device)
```

### Online FID Pattern

Evaluate periodically using the EMA model (not the training model):

```python
if step % cfg.sample_fid_every == 0 and step > 0:
    with torch.no_grad():
        torch.cuda.empty_cache()
        fid_metric.reset()

        # 1) Feed real images
        for _ in range(n_fid_samples // batch_size):
            real = next(real_data_iter)
            fid_metric.update(real, real=True)

        # 2) Generate and feed fake images (use EMA model)
        for _ in range(n_fid_batches):
            z = torch.randn(batch_size, C, H, W, device=device)
            samples = sample_fn(z, ema_model)

            # Decode latents to pixels if training in latent space
            if use_latent:
                samples = vae.decode(samples / 0.18215).sample
            samples = (samples.clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8)

            # Gather across GPUs before updating metric
            samples = accelerator.gather(samples.contiguous())
            fid_metric.update(samples, real=False)

            del samples, z
            torch.cuda.empty_cache()

        result = fid_metric.compute()
        best_fid = min(result.item(), best_fid)
        torch.cuda.empty_cache()
```

### FID Sample Count Strategy

Use fewer samples during training (fast signal), full count for final reporting:

```yaml
evaluation:
  sample_fid_n: 5000       # Training-time FID (noisy but directional)
  sample_fid_every: 20000  # Steps between FID evals
  sample_fid_bs: 4         # FID sampling batch (must be <= training batch)
  # For 1024px resolution, use sample_fid_bs: 1

# Final offline evaluation
num_fid_samples: 50000     # Standard benchmark count
```

Distribute samples evenly across GPUs:
```python
fid_batches = cfg.sample_fid_n // (cfg.sample_fid_bs * accelerator.num_processes)
```

### Adaptive ODE Solver for Fast Training-Time Sampling

Use `dopri5` (adaptive step) during training for speed, fixed-step Euler for final evaluation:

```python
# Training-time FID: adaptive solver, fewer steps (~50)
sample_fn = transport_sampler.sample_ode(
    sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3
)

# Final evaluation: fixed-step, more steps (250) for reproducibility
sample_fn = transport_sampler.sample_ode(
    sampling_method="euler", num_steps=250
)
```

### Multi-Metric Evaluation

Compute multiple metrics in a single pass over generated samples to amortize sampling cost:

| Metric | Feature Extractor | Dimension | What It Captures |
|--------|------------------|-----------|-----------------|
| FID | InceptionV3 pool | 2048 | Distribution quality |
| IS | InceptionV3 | - | Diversity + quality |
| KID | InceptionV3 pool | 2048 | Unbiased FID alternative |
| sFID | InceptionV3 spatial | 768 | Spatial structure |
| FDD | DINOv2-ViT-L/14 | 1024 | Modern alternative to FID |
| FVD | I3D | 400 | Video quality |

### Memory Management

Key patterns to prevent OOM during evaluation:

```python
# FID batch size must fit within training memory budget
assert cfg.sample_fid_bs <= cfg.batch_size

# Per-batch cleanup in sampling loop
del samples, z
torch.cuda.empty_cache()

# Contiguous before DDP gather
samples_global = accelerator.gather(samples.contiguous())
```

### Distributed FID Scaling Gotcha

torchmetrics FID has a known bug at >= 32 GPUs. Workaround:

```python
if accelerator.num_processes >= 32:
    cfg.sample_fid_n = min(cfg.sample_fid_n, 1000)
    log.warning("Capping FID samples to 1K (torchmetrics bug at >=32 GPUs)")
```

## Quick Reference

| Technique | Speed Impact | Memory Impact | Config Key |
|-----------|-------------|---------------|------------|
| 5K samples (training) | 10x faster than 50K | Less | `sample_fid_n: 5000` |
| dopri5 adaptive ODE | Faster than fixed-step | Same | `sampling_method: dopri5` |
| Latent-space generation | 64x less compute | Much less | `is_latent: true` |
| Multi-metric single pass | Amortizes sampling | Same | Compute FID+IS+KID together |
| EMA model for sampling | Better FID score | Same | Use `ema_model.forward` |
| Per-batch `empty_cache()` | Slight overhead | Prevents OOM | Code pattern |

## Anti-Patterns

- **FID with 50K samples during training**: Use 5K for directional signal, save 50K for final benchmarks.
- **FID sampling batch size > training batch size**: Guarantees OOM. Assert `sample_fid_bs <= batch_size`.
- **Using training model for FID (not EMA)**: EMA model produces better samples. Always sample from EMA for evaluation.
- **Forgetting `accelerator.gather()` before metric update**: Each GPU only sees its local samples, FID is computed on partial data.
- **FID at >=32 GPUs without workaround**: torchmetrics has a known sync bug. Cap samples or verify manually.
- **`normalize=True` with uint8 images**: torchmetrics expects [0,255] uint8 when `normalize=False`. Mixing up conventions gives wrong FID.
- **No `torch.no_grad()` around FID eval**: Builds computation graph for sampling, wastes memory. Always wrap in `no_grad()`.

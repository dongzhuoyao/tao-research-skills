---
name: genai-evaluation-metrics
description: Use when evaluating generative models — choosing metrics (FID, IS, KID, sFID, FDD, FVD, PRDC, LPIPS, SSIM, AuthPct, Vendi), setting up online or offline evaluation, feature extractor selection, distributed computation, memory management during sampling. Triggers: "FID", "IS", "KID", "inception score", "frechet", "LPIPS", "SSIM", "evaluation metrics", "generative evaluation", "FVD"
---

# GenAI Evaluation Metrics

## When to Use

- Setting up online evaluation during generative model training
- Choosing which metrics to compute (image vs video, distribution vs perceptual)
- Selecting feature extractors (InceptionV3 vs DINOv2 vs CLIP)
- Configuring sample counts for training-time vs final benchmarks
- Debugging OOM during evaluation sampling phases
- Distributed metric computation with HuggingFace Accelerate
- Evaluating memorization, diversity, or mode collapse

## Metric Catalog

### Distribution Metrics (Frechet Distance family)

All compute distance between real and generated feature distributions using mean + covariance.

| Metric | Feature Extractor | Dim | What It Captures |
|--------|------------------|-----|-----------------|
| **FID** | InceptionV3 pool3 | 2048 | Overall distribution quality (standard benchmark) |
| **sFID** | InceptionV3 spatial (Mixed_6e) | 2023 | Spatial structure quality |
| **FDD** | DINOv2 ViT-L/14 | 1024 | Modern FID alternative, better on textures |
| **FVD** | I3D | 400 | Video temporal + spatial quality |

### Diversity & Quality Metrics

| Metric | What It Measures |
|--------|-----------------|
| **IS** (Inception Score) | Quality (confident predictions) + diversity (class coverage) |
| **KID** (Kernel Inception Distance) | Unbiased FID alternative using MMD with polynomial kernel |
| **PRDC** (Precision/Recall/Density/Coverage) | Manifold overlap: fidelity (P), mode coverage (R), sample density (D), support coverage (C) |
| **Vendi Score** | Diversity via eigenvalue entropy of similarity matrix |

### Perceptual Metrics (paired, per-sample)

| Metric | Feature Extractor | Use Case |
|--------|------------------|----------|
| **LPIPS** | AlexNet (spatial) | Perceptual similarity between paired images/frames |
| **SSIM** | Gaussian filter | Structural similarity (luminance, contrast, structure) |
| **PSNR** | None (MSE) | Pixel-level reconstruction quality |

### Memorization & Overfitting Metrics

| Metric | What It Detects |
|--------|----------------|
| **AuthPct** | % of generated samples that are "authentic" (not memorized) |
| **CT Score** | Data copying / memorization detection |
| **FLS** (Frechet Likelihood Score) | KDE-based likelihood, sensitive to overfitting |
| **FD-infinity** | FID extrapolated to infinite samples (removes sample-size bias) |

## Patterns

### Metric Orchestrator

Wrap all metrics behind a unified update/compute interface:

```python
class MyMetric:
    def __init__(self, device="cuda", choices=["fid"]):
        if "fid" in choices:
            self._fid = FrechetInceptionDistance(
                feature=2048, reset_real_features=True,
                normalize=False, sync_on_compute=True,
            ).to(device)
        if "is" in choices:
            self._is = InceptionScore().to(device)
        if "kid" in choices:
            self._kid = KernelInceptionDistance(subset_size=50).to(device)
        if "prdc" in choices:
            self._prdc = PRDC(nearest_k=5).to(device)
        if "sfid" in choices:
            self._sfid = sFrechetInceptionDistance().to(device)
        if "fdd" in choices:
            self._fdd = FrechetDinovDistance().to(device)
        if "fvd" in choices:
            self._fvd = FrechetVideoDistance()
        if "dinov2" in choices:
            self._dinov2 = DinoV2_Metric().to(device)

    def update_real(self, imgs):
        # IS only gets fake images — skip real for IS
        for name in self.choices:
            if name != "is":
                getattr(self, f"_{name}").update(imgs, real=True)

    def update_fake(self, imgs):
        for name in self.choices:
            getattr(self, f"_{name}").update(imgs, real=False)

    def compute(self):
        results = {}
        for name in self.choices:
            results.update(getattr(self, f"_{name}").compute())
        return results
```

Image vs video metric selection:
```python
# Images
metric = MyMetric(choices=["fid", "is", "kid", "prdc", "sfid", "fdd", "dinov2"])

# Videos
metric = MyMetric(choices=["fid", "fvd"], video_frame=16)
```

### Online Evaluation (During Training)

Evaluate periodically using the EMA model (not the training model):

```python
if step % cfg.sample_fid_every == 0 and step > 0:
    with torch.no_grad():
        torch.cuda.empty_cache()
        metric.reset()

        # 1) Feed real images
        for _ in range(n_fid_samples // batch_size):
            metric.update_real(next(real_data_iter))

        # 2) Generate fake images with EMA model
        for _ in range(n_fid_batches):
            z = torch.randn(batch_size, C, H, W, device=device)
            samples = sample_fn(z, ema_model)
            if use_latent:
                samples = vae.decode(samples / 0.18215).sample
            samples = (samples.clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8)

            samples = accelerator.gather(samples.contiguous())
            metric.update_fake(samples)

            del samples, z
            torch.cuda.empty_cache()

        results = metric.compute()
        best_fid = min(results["fid"], best_fid)
```

Track multiple bests for checkpointing:
```python
best_fid = min(results["fid"], best_fid)
best_fdd = min(results["fdd"], best_fdd)
best_sfid = min(results["sfid"], best_sfid)
best_dinov2_fid = min(results["dinov2_fid"], best_dinov2_fid)
```

### sFID, FDD, FVD: Distribution Metric Implementations

sFID uses InceptionV3 spatial features (Mixed_6e), FDD uses DINOv2 ViT-L/14, FVD uses I3D for video.
See [references/distribution-metrics.md](references/distribution-metrics.md) for implementation details.

### DINOv2 Multi-Metric (FID+KID+IS+PRDC in one pass)

Compute FID, KID, IS, and PRDC in DINOv2 feature space with a single extraction pass.
See [references/dinov2-multi-metric.md](references/dinov2-multi-metric.md) for implementation.

### PRDC, AuthPct, Vendi Score, FD-infinity

Diversity and memorization metrics: k-NN manifold overlap (PRDC), memorization detection (AuthPct), eigenvalue diversity (Vendi), and sample-size debiased FID (FD-infinity).
See [references/diversity-memorization-metrics.md](references/diversity-memorization-metrics.md) for implementations.

### Video Perceptual Metrics (FVD, LPIPS, SSIM, PSNR)

Frame-by-frame perceptual metrics for video evaluation using I3D, AlexNet, and structural similarity.
See [references/video-perceptual-metrics.md](references/video-perceptual-metrics.md) for implementation details.

## Feature Extractor Selection Guide

| Extractor | Best For | Dim | Speed |
|-----------|---------|-----|-------|
| **InceptionV3** (pool3) | Standard FID benchmarks, paper comparisons | 2048 | Fast |
| **InceptionV3** (spatial) | Spatial structure evaluation (sFID) | 2023 | Fast |
| **DINOv2 ViT-L/14** | Modern alternative, better texture sensitivity | 1024 | Medium |
| **CLIP ViT-L/14** | Text-conditioned generation, cross-modal | varies | Medium |
| **I3D** | Video quality (FVD) | 400 | Slow |
| **AlexNet** (LPIPS) | Perceptual similarity (paired) | spatial | Fast |

## Sample Count Strategy

```yaml
# Online (during training): fast directional signal
evaluation:
  sample_fid_n: 5000
  sample_fid_every: 20000  # steps
  sample_fid_bs: 4         # must be <= training batch_size

# Offline (final benchmark): full standard count
num_fid_samples: 50000
```

Multi-GPU scaling:
```python
fid_batches = cfg.sample_fid_n // (cfg.sample_fid_bs * accelerator.num_processes)

# torchmetrics bug at >= 32 GPUs
if accelerator.num_processes >= 32:
    cfg.sample_fid_n = min(cfg.sample_fid_n, 1000)
```

## Memory Management

```python
with torch.no_grad():
    torch.cuda.empty_cache()
    metric.reset()
    for _ in range(n_batches):
        samples = sample_fn(z, ema_model)
        samples = accelerator.gather(samples.contiguous())
        metric.update_fake(samples)
        del samples, z
        torch.cuda.empty_cache()
```

Key rules:
- `sample_fid_bs <= training batch_size` (or OOM)
- `torch.no_grad()` around entire eval block
- `del` + `empty_cache()` per batch during sampling
- `.contiguous()` before `accelerator.gather()`

## Anti-Patterns

- **Only computing FID**: Use multiple metrics. FID misses spatial structure (sFID), mode collapse (PRDC recall), memorization (AuthPct).
- **FID with 50K samples during training**: Use 5K for directional signal, 50K for final benchmarks only.
- **Using training model for sampling (not EMA)**: EMA produces better samples. Always use EMA for evaluation.
- **Forgetting `accelerator.gather()`**: Each GPU only sees local samples, metrics computed on partial data.
- **`normalize=True` with uint8 images**: torchmetrics expects [0,255] uint8 when `normalize=False`.
- **InceptionV3 for everything**: Consider DINOv2 (FDD) for modern benchmarks, I3D for video (FVD).
- **Ignoring memorization**: High-quality samples may be copied from training data. Add AuthPct or CT score.
- **FID at >=32 GPUs without workaround**: torchmetrics sync bug. Cap samples or verify manually.

## See Also

- `gpu-training-acceleration` — Memory management during evaluation sampling
- `wandb-experiment-tracking` — Logging evaluation metrics to W&B

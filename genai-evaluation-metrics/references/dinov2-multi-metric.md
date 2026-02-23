# DINOv2 Multi-Metric Pattern

Compute multiple metrics in DINOv2 feature space using a single feature extraction:

```python
class DinoV2_Metric(Metric):
    def compute(self):
        results = {}
        results["dinov2_fid"] = compute_FD_with_reps(real_feats, fake_feats)
        results["dinov2_kid"] = compute_mmd(real_feats, fake_feats).mean()
        results["dinov2_is"] = compute_inception_score(fake_feats)
        results.update({"dinov2_" + k: v
            for k, v in compute_prdc(real_feats, fake_feats, nearest_k=5).items()})
        return results
```

Output: `dinov2_fid`, `dinov2_kid`, `dinov2_is`, `dinov2_precision`, `dinov2_recall`, `dinov2_density`, `dinov2_coverage`

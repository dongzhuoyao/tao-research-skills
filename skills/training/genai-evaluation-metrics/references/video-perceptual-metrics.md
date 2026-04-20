# Video Perceptual Metrics

## FVD Detail

Uses I3D model for video quality assessment. Requires minimum 10 frames. Computes Frechet distance in I3D feature space (dim=400):

```python
class FrechetVideoDistance(Metric):
    def __init__(self, feature=400):
        self.inception = VideoDetector()  # I3D TorchScript

    def update(self, videos, real):
        # Input: [B, T, H, W, C] float -> permute to [B, C, T, H, W]
        videos = videos.permute(0, 4, 1, 2, 3)
        features = self.inception(videos)
```

## LPIPS

Frame-by-frame perceptual similarity using AlexNet spatial features (lower = more similar):

```python
loss_fn = lpips.LPIPS(net='alex', spatial=True)
for t in range(num_frames):
    score = loss_fn(video1[:, t] * 2 - 1, video2[:, t] * 2 - 1)  # input: [-1,1]
```

## SSIM

Structural similarity index measuring luminance, contrast, and structure (higher = more similar):

- 11x11 Gaussian window, sigma=1.5
- C1=0.01^2, C2=0.03^2

## PSNR

Peak signal-to-noise ratio based on MSE for pixel-level reconstruction quality.

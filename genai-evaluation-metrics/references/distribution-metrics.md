# Distribution Metrics Implementation

## sFID: Spatial Features from InceptionV3

Uses intermediate spatial features (Mixed_6e layer) instead of global pool:

```python
class NoTrainInceptionV3(nn.Module):
    def forward(self, x):
        if "768" in self.features_list:
            # Extract spatial features: [B, 768, 7, 7] -> [B, 2023]
            features["768"] = x[:, :7, :, :].reshape(x.shape[0], -1).to(torch.float32)
        return features

class sFrechetInceptionDistance(Metric):
    def __init__(self, feature=2023):
        self.inception = NoTrainInceptionV3(features_list=["768"])
```

## FDD: DINOv2 as Feature Extractor

Replace InceptionV3 with DINOv2 for more modern feature representations:

```python
class DINOv2Encoder(nn.Module):
    def setup(self, arch="vitl14"):
        self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{arch}")

    def transform(self, image):
        image = TF.Resize((224, 224), TF.InterpolationMode.BICUBIC)(image)
        return TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image.float())

class FrechetDinovDistance(Metric):
    def __init__(self, feature=1024):  # DINOv2-ViT-L/14 output dim
        self.inception = DINOv2Encoder()
```

## FVD: Video Quality

Uses I3D model, requires minimum 10 frames:

```python
class FrechetVideoDistance(Metric):
    def __init__(self, feature=400):
        self.inception = VideoDetector()  # I3D TorchScript

    def update(self, videos, real):
        # Input: [B, T, H, W, C] float -> permute to [B, C, T, H, W]
        videos = videos.permute(0, 4, 1, 2, 3)
        features = self.inception(videos)
```

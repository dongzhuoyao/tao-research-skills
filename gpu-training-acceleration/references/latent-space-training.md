# Latent Space Training

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

---
name: gemini-generate-img
description: "Use when generating images from text prompts via Google's Gemini image models (\"nano-banana\") with the `google-genai` SDK or OpenRouter. Covers model choice, aspect ratio, prompt patterns, multi-candidate sampling, and retry logic. Triggers: \"generate image\", \"gemini image\", \"nano-banana\", \"text to image\", \"gemini-3-pro-image\", \"gemini-2.5-flash-image\", \"imagen\", \"paper figure generation\""
---

# Gemini Image Generation

## When to Use

- Text-to-image generation via Gemini image models (a.k.a. nano-banana) from Python
- Generating paper figures, teasers, demo assets, or social creative in research pipelines
- Picking between `gemini-3-pro-image-preview` (quality) and `gemini-2.5-flash-image` (speed)
- Running batch / parallel candidate sampling with retry

For editing an existing image or composing multiple images, use `gemini-edit-img`.

## Quick Start

```python
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[types.Part.from_text(text="A quiet reading nook at dawn, watercolor")],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        candidate_count=1,
        image_config=types.ImageConfig(aspect_ratio="16:9", image_size="1k"),
    ),
)

for part in response.candidates[0].content.parts:
    if part.inline_data:
        with open("out.png", "wb") as f:
            f.write(part.inline_data.data)
        break
```

## Models

| Model | Use when | Notes |
|-------|----------|-------|
| `gemini-3-pro-image-preview` | Quality matters — hero figures, paper diagrams | Best text rendering; slower |
| `gemini-2.5-flash-image` | Iteration / many candidates | Fast, cheap, the "nano-banana" default |
| `gemini-3.1-flash-image-preview` | Current fast preview | Use when 2.5-flash-image deprecates |

Default to `gemini-3-pro-image-preview` for finals; switch to `gemini-2.5-flash-image` when `candidate_count` > 4 or for quick drafts.

## Aspect Ratio & Size

Pass through `types.ImageConfig(aspect_ratio=..., image_size=...)`:

| Ratio | Pixel dims | Use |
|-------|-----------|-----|
| `1:1`  | 1024×1024 | logos, square covers |
| `16:9` | 1344×768  | landing hero, slides, YouTube |
| `9:16` | 768×1344  | stories, mobile portrait |
| `4:5`  | 896×1152  | Instagram feed |
| `21:9` | 1536×672  | paper teasers, banners |
| `3:2`  | 1248×832  | photo classic |

`image_size`: `"1k"` (default) | `"2k"` | `"4k"`. Start at 1k; upscale only for finals — 4k is ~10× slower.

## Prompt Patterns

### In-image text

Gemini-3-pro handles text better than any open model but still miscounts long strings.

- Quote the literal text: `a neon sign reading "Quiet Chapter"`
- Keep per-sign text ≤ 6 words
- For multi-sign scenes, list each sign explicitly rather than bundling

### Composition

- Subject → setting → mood → style: `a cartographer's desk at dusk, warm lamplight, tilted angle, watercolor, muted palette`
- Put style qualifiers at the end
- Avoid negative prompts (`no X`) — Gemini ignores most. Rephrase positively.

### Multi-candidate sampling

`candidate_count` is capped at 8 per call. For more, loop externally with parallel requests:

```python
# Reference: PaperBanana/utils/generation_utils.py::call_gemini_with_retry_async
target = 16
config.candidate_count = min(8, target)
results = []
while len(results) < target:
    resp = await client.aio.models.generate_content(model=model, contents=contents, config=config)
    results.extend(img_bytes_from(resp))
```

## Retry Pattern

Image generation returns an empty `parts` list on safety filters and transient errors. Always check before indexing:

```python
import asyncio

async def gen_with_retry(client, model, contents, config, max_attempts=5, base_delay=5):
    for attempt in range(max_attempts):
        try:
            resp = await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
            if resp.candidates and resp.candidates[0].content.parts:
                for part in resp.candidates[0].content.parts:
                    if part.inline_data:
                        return part.inline_data.data
        except Exception as e:
            print(f"attempt {attempt + 1}: {e}")
        await asyncio.sleep(min(base_delay * (2 ** attempt), 30))
    raise RuntimeError("All attempts returned empty parts")
```

Full reference: `PaperBanana/utils/generation_utils.py::call_gemini_with_retry_async`.

## OpenRouter Alternative

Single key for both reasoning and image generation. Send `modalities` and `image_config` in the chat payload:

```python
import httpx

payload = {
    "model": "google/gemini-3-pro-image-preview",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "..."}]}],
    "modalities": ["image", "text"],
    "image_config": {"aspect_ratio": "16:9", "image_size": "1k"},
}
resp = httpx.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    json=payload, timeout=300,
)
```

Image bytes arrive in one of three places in the response — see `PaperBanana/utils/generation_utils.py::call_openrouter_image_generation_with_retry_async` for the full extraction chain (inline_data, `images` field, or base64 data URL in `content`).

## Environment

```bash
export GEMINI_API_KEY="..."          # Direct Gemini (preferred)
# or
export GOOGLE_API_KEY="..."          # Equivalent — google-genai reads either
# or
export OPENROUTER_API_KEY="sk-or-..." # Unified routing via OpenRouter
```

Keys: https://aistudio.google.com/apikey (Gemini) | https://openrouter.ai/keys (OpenRouter).

## Anti-Patterns

- **Using a non-image model** — `gemini-pro`, `gemini-2.5-flash` don't generate images. Model name must contain `-image`.
- **`response_modalities=["TEXT", "IMAGE"]` when you only want pixels** — adds a text caption, slows generation, bloats response. Use `["IMAGE"]`.
- **Indexing `response.candidates[0].content.parts` without checking** — safety filter returns empty parts. Always guard.
- **Long negative prompts** — `"no text, no humans, no watermark"` is mostly ignored. Rewrite positively.
- **`candidate_count > 8` per single call** — API hard-caps at 8. Loop externally.
- **Saving `inline_data.data` as `.jpg`** — it's PNG bytes. Convert with Pillow (`Image.open(BytesIO(b)).convert("RGB").save(..., "JPEG", quality=95)`) if you need JPEG.
- **Not fixing a seed** — Gemini image models are stochastic; for A/B you want `seed=` in `GenerateContentConfig` for reproducibility.

## See Also

- `gemini-edit-img` — Editing / composing existing images with the same Gemini image models
- `genai-evaluation-metrics` — Scoring generated image quality (FID, CLIP-score, LPIPS)
- `academic-deep-research` — Evaluating papers that report Gemini / nano-banana baselines

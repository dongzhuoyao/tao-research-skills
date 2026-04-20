---
name: gemini-edit-img
description: Use when editing, modifying, or composing existing images with Google's Gemini image models ("nano-banana"). Covers single-image edits (add/remove/restyle), multi-image composition (outfit swap, subject-into-scene, style transfer), input encoding, preservation tricks, and iterative refinement. Triggers: "edit image", "gemini edit", "nano-banana edit", "image to image", "image composition", "outfit swap", "subject transfer", "style transfer", "inpaint", "outpaint"
---

# Gemini Image Editing

## When to Use

- Editing an existing image with a text instruction (add / remove / modify / restyle)
- Composing two or more images (subject-into-scene, outfit swap, style reference)
- Iterative refinement: generate → edit → edit, chaining the previous output forward
- Deciding between Gemini image-to-image and a dedicated inpainting model

For pure text-to-image generation (no input images), use `gemini-generate-img`.

## Quick Start

```python
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open("photo.png", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        types.Part.from_text(text="Add a soft rainbow in the sky, keep everything else unchanged"),
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        candidate_count=1,
    ),
)

for part in response.candidates[0].content.parts:
    if part.inline_data:
        with open("edited.png", "wb") as f:
            f.write(part.inline_data.data)
        break
```

## Input Encoding

Any of the following builds a valid image part:

```python
# From disk
types.Part.from_bytes(data=open(path, "rb").read(), mime_type="image/png")

# From PIL — SDK auto-encodes
types.Part(image=pil_image)

# From base64 string (generic dict format used by PaperBanana)
{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_str}}
```

Supported mime types: `image/png`, `image/jpeg`, `image/webp`. Convert HEIC / AVIF / TIFF to PNG first.

Reference for dict-to-Part conversion: `PaperBanana/utils/generation_utils.py::_convert_to_gemini_parts`.

## Multi-Image Composition

Order matters — Gemini treats the **first** image as the base, subsequent ones as references:

```python
contents = [
    types.Part.from_bytes(data=scene_bytes,   mime_type="image/png"),  # base scene
    types.Part.from_bytes(data=subject_bytes, mime_type="image/png"),  # subject ref
    types.Part.from_text(text=(
        "Place the subject from the second image at the door of the first scene. "
        "Match lighting and palette of the scene."
    )),
]
```

Recipes:

| Goal | Images (order) | Prompt |
|------|---------------|--------|
| Outfit swap | person, clothing | `dress the person from image 1 in the outfit from image 2` |
| Style transfer | content, style | `repaint image 1 in the style of image 2` |
| Subject-into-scene | scene, subject | see above |
| Product placement | product, scene | `place the product from image 1 on the table in image 2` |
| Face-keep restyle | portrait, style | `restyle image 1 in the style of image 2; keep the face and expression identical` |

Beyond 2–3 references attention splits and quality tanks — combine into a single collage if you need more.

## Preservation Tricks

Gemini re-renders the full image on every edit. To keep regions stable:

| Goal | Prompt tail |
|------|-------------|
| Keep faces | `keep the person's face, eyes, and expression exactly the same` |
| Keep text | `do not change any visible text; preserve signs and logos exactly` |
| Keep layout | `same composition, same camera angle, same crop` |
| Minimal edit | `make only the specific change above; leave everything else pixel-identical` |

For pixel-perfect masked edits (product backgrounds, UI screenshots), use a dedicated inpainting model (SDXL-inpaint, Flux-fill). Gemini always re-renders.

## Iterative Refinement

Chain edits by feeding the previous output forward — don't rebuild from the original each round:

```python
out1 = edit(base,  "add a rainbow in the sky")
out2 = edit(out1,  "make the rainbow brighter and add a unicorn under it")
out3 = edit(out2,  "shift camera slightly left, keep everything else identical")
```

Each call is stateless; Gemini does not retain edit history. If you want a full conversational edit trail, use `client.chats.create(...)` with a sustained session.

## Retry Pattern

Same semantics as text-to-image: empty `parts` = safety filter or transient failure.

```python
for attempt in range(5):
    resp = await client.aio.models.generate_content(...)
    if resp.candidates and resp.candidates[0].content.parts:
        for part in resp.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data
    await asyncio.sleep(min(5 * (2 ** attempt), 30))
raise RuntimeError("All attempts empty")
```

Reference: `PaperBanana/utils/generation_utils.py::call_gemini_with_retry_async`.

## Size Budget

Each input image counts against the context. Rough guidance:

- 1024×1024 PNG ≈ 1–2 MB encoded, ~1300 tokens after tokenization
- Keep total input under ~8 MB / 6 images for reliable latency
- Downscale very large inputs with Pillow before sending:
  ```python
  img = Image.open(path); img.thumbnail((2048, 2048)); img.save(buf, "PNG")
  ```

## Environment

Same as `gemini-generate-img`: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`, or `OPENROUTER_API_KEY`).

## Anti-Patterns

- **Expecting pixel-perfect preservation** — Gemini re-renders every pixel; faces and logos drift. Use a masked inpainting model if you need exact preservation.
- **Passing 5+ reference images** — attention splits; quality drops sharply. Collage them instead.
- **Omitting `mime_type` on `from_bytes`** — silent failure. Always specify `image/png` or `image/jpeg`.
- **Re-uploading the original when chaining edits** — changes compound oddly. Feed the previous output.
- **"Exact same image but ..."** — Gemini still drifts. Be explicit about what to keep (`face, text, composition`).
- **Bounding boxes or coordinate-based masks in the prompt** — Gemini ignores them. Describe edits in natural language.
- **Uploading source images that contain PII without need** — the edit output may leak identifiable features even when you prompt removal. Crop / blur first if privacy matters.

## See Also

- `gemini-generate-img` — Text-to-image generation with the same Gemini image models
- `genai-evaluation-metrics` — Perceptual metrics (LPIPS, SSIM, DINO-sim) for pre-/post-edit fidelity
- `academic-deep-research` — Finding recent image-editing papers and benchmarks

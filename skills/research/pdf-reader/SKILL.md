---
name: pdf-reader
description: "Use when reading PDF papers, reports, or long documents where text, figures, and tables must all be captured and chunk-summarized without truncation. Converts PDF to a markdown + paper_content.json workspace, extracts figures and tables as standalone files, then delegates to arxiv-latex-reader's progressive two-layer reading (section index + on-demand deep reads). Triggers: \"read pdf\", \"pdf to markdown\", \"summarize pdf\", \"pdf paper\", \"extract figures from pdf\", \"extract tables from pdf\", \"marker pdf\", \"pymupdf\", \"docling\", \"paper digest\", \"pdf reader\""
---

# PDF Reader

## When to Use

- Reading a PDF paper or report (any source — not just arxiv) where figures/tables matter
- Papers with result tables, architecture diagrams, ablation plots that must be preserved
- Long PDFs (20+ pages) that overflow a single `Read` call
- Scanned / image-heavy PDFs where plain `pdftotext` loses structure
- Any workflow where you'd use `arxiv-latex-reader` but the input is a PDF, not LaTeX

For arxiv papers with LaTeX source available, prefer `arxiv-latex-reader` directly — it's cleaner (no conversion loss). Use `pdf-reader` when LaTeX source is unavailable, the PDF is the only artifact, or the document is non-arxiv.

## Architecture

Two stages. Stage 1 is PDF-specific; Stage 2 is identical to `arxiv-latex-reader`.

```
Stage 1 (PDF-specific):
  PDF → pdf_to_markdown.py → workspace/<paper_id>/
                             ├── paper_content.json     # sections + text
                             ├── paper.md               # full markdown
                             ├── figures/               # extracted images + captions
                             └── tables/                # extracted markdown tables

Stage 2 (delegated to arxiv-latex-reader, unchanged):
  workspace/<paper_id>/paper_content.json
    → section_indexer.py → section_index.json (~2k tokens)
    → section_reader.py / paper_navigator.py → on-demand full-section reads
```

The workspace layout is **compatible with `arxiv-latex-reader`**. Once Stage 1 produces `paper_content.json`, every downstream script from `arxiv-latex-reader` works unchanged.

## Stage 1 — Convert PDF to Markdown + Workspace

### Conversion tool fallback chain

PDFs vary wildly — academic LaTeX-rendered, scanned scans, double-column with equations, poster-style layouts. Use the first tool in the chain that installs and produces usable output.

| Tool | Strengths | Weaknesses | Install |
|------|-----------|-----------|---------|
| **marker-pdf** | Best overall: text + figures + tables + equations, preserves structure | Slow (ML models), large download | `pip install marker-pdf` |
| **docling** (IBM) | Excellent tables, layout-aware, LLM-friendly output | Heavy dependencies | `pip install docling` |
| **pymupdf4llm** | Fast, good text + basic tables, Markdown-native | No figure extraction; tables mid-quality | `pip install pymupdf4llm` |
| **pdfplumber** | Best-in-class table extraction with structure | Text only, slower, no figures | `pip install pdfplumber` |
| **pdftotext + pdfimages** (poppler) | Minimal install, fast | No tables, no captions, no structure | `brew install poppler` |
| **Claude Code native `Read`** | Built-in, multimodal, reads figures as images | 20 pages per call; not scriptable | N/A |

**Recommendation:** default to `marker-pdf`. Fallback to `pymupdf4llm` if marker is slow or unavailable. Use `pdfplumber` *in addition* for tables when they're critical.

### The converter script

Run `scripts/pdf_to_markdown.py` to build the workspace:

```bash
python scripts/pdf_to_markdown.py <path-or-url-to.pdf> --paper-id <slug> [--tool marker|pymupdf]
# → workspace/<slug>/paper_content.json
# → workspace/<slug>/paper.md
# → workspace/<slug>/figures/
# → workspace/<slug>/tables/
```

Key behaviors:

- **Never truncate.** The whole PDF is converted, no page cap.
- **Section detection** uses markdown heading levels in the converter output. If the converter emits flat text (no headings), falls back to font-size heuristics (H1 = largest font cluster, H2 = next, etc.) via pymupdf.
- **Figures** are saved as PNG to `figures/fig_<n>.png` with a sidecar `figures/fig_<n>.txt` containing the caption detected below/above the bounding box.
- **Tables** are saved as markdown to `tables/tab_<n>.md` with a first-line `<!-- caption: ... -->` comment. For complex tables, save both `.md` and `.csv`.
- **Math** is kept as LaTeX (`$...$`, `$$...$$`). marker preserves this; pymupdf4llm does not — flag in the log when equations may be lost.

See `scripts/pdf_to_markdown.py` for the implementation. The script writes a `workspace/<paper_id>/conversion.log` with the tool used, page count, figure count, table count, and any dropped content warnings.

### `paper_content.json` schema

Produced by the converter, consumed by `arxiv-latex-reader`:

```json
{
  "paper_id": "2404.12345",
  "source": "pdf",
  "title": "...",
  "authors": [...],
  "sections": [
    {"name": "Abstract",    "text": "...", "level": 1},
    {"name": "Introduction","text": "...", "level": 1, "figures": ["fig_1"], "tables": []},
    {"name": "Method",      "text": "...", "level": 1, "figures": ["fig_2","fig_3"], "tables": ["tab_1"]}
  ],
  "figures": [
    {"id":"fig_1","path":"figures/fig_1.png","caption":"...","page":2}
  ],
  "tables":  [
    {"id":"tab_1","path":"tables/tab_1.md",  "caption":"...","page":5}
  ]
}
```

The `figures` and `tables` arrays per-section are what `pdf-reader` adds on top of `arxiv-latex-reader`'s format — they let the reader pull the right figure into context when deep-reading a section.

## Stage 2 — Progressive Reading (delegated)

Once Stage 1 completes, use `arxiv-latex-reader` as-is:

```bash
# Build the ~2k-token section index (one-time per paper)
python ../arxiv-latex-reader/scripts/section_indexer.py <paper_id>

# Overview or on-demand section reads
python ../arxiv-latex-reader/scripts/paper_navigator.py <paper_id> overview
python ../arxiv-latex-reader/scripts/paper_navigator.py <paper_id> read Method Experiments
python ../arxiv-latex-reader/scripts/paper_navigator.py <paper_id> poster
```

Python API is identical:

```python
from paper_navigator import PaperNavigator
nav = PaperNavigator("2404.12345", workspace_dir="workspace")
overview = nav.get_overview()
sections = nav.read_full(["Method", "Experiments"])
```

**Do not re-implement indexing or navigation here.** `pdf-reader` is a converter front-end, nothing more.

## Figure and Table Handling

PDFs encode figures as raster/vector content with captions placed above or below — the converter must pair the two.

### Classification (done during conversion)

Same classes as `blog-reader` and `arxiv-latex-reader`'s figure_extractor:

| Class | Signals | What the reader does |
|-------|---------|----------------------|
| **Key** | Referenced in body ("Fig. 2"), architecture/overview diagrams, main result tables | `Read` the image multimodally when deep-reading its section |
| **Supporting** | Ablation plots, secondary results | `Read` if budget allows |
| **Decorative** | Hero images, logos | Caption-only; skip multimodal Read |

Classification is stored in each figure's sidecar: `figures/fig_1.txt`:

```
caption: Model architecture overview.
class: key
page: 2
referenced_in: §Method
```

### Multimodal figure Read at deep-read time

When `paper_navigator read §Method` is called and §Method has key figures, the caller should:

```python
# After getting section text, also Read key figures for that section
for fig_id in section["figures"]:
    if figures[fig_id]["class"] == "key":
        # Read via Claude Code's multimodal Read tool
        image = read_image(f"workspace/<paper_id>/figures/{fig_id}.png")
        # → multimodal interpretation inline with section text
```

This is the main value-add over `arxiv-latex-reader`: figures are already PNGs, so you skip the LaTeX-compile-to-PNG step.

### Table handling

Tables are already markdown — include them verbatim alongside the section text. For numeric result tables, preserve every cell value verbatim (no rounding, no summarization).

## Quick Reference

```bash
# Full pipeline (one paper, end to end)
python pdf-reader/scripts/pdf_to_markdown.py paper.pdf --paper-id mypaper
python arxiv-latex-reader/scripts/section_indexer.py mypaper
python arxiv-latex-reader/scripts/paper_navigator.py mypaper overview

# Batch: convert once, then query many times
for pdf in papers/*.pdf; do
  python pdf-reader/scripts/pdf_to_markdown.py "$pdf" --paper-id "$(basename "$pdf" .pdf)"
  python arxiv-latex-reader/scripts/section_indexer.py "$(basename "$pdf" .pdf)"
done
```

## Key Principles

1. **PDF → markdown first, always.** Never send raw PDF bytes into a summarization pipeline — you lose structure and table semantics.
2. **Tool fallback, not tool monoculture.** `marker-pdf` fails on some scanned PDFs; `pymupdf4llm` fails on heavy-equation papers. Document which tool produced the output in `conversion.log`.
3. **Figures and tables are load-bearing.** Extract them as first-class artifacts, not inline captions.
4. **Delegate reading.** Stage 2 is `arxiv-latex-reader` unchanged — do not fork or reimplement.
5. **Never truncate.** Same rule as `arxiv-latex-reader`.

## Anti-Patterns

- **Passing the raw PDF to `WebFetch` or an LLM** — silent truncation, no figure extraction, lossy table representation.
- **Relying only on `pdftotext`** — no structure, no tables. Fine as a last-resort fallback, never the default.
- **Using Claude Code's native `Read` on PDFs ≥20 pages without `pages:`** — fails immediately. Chunk or convert instead.
- **Discarding the figure images after caption extraction** — the whole point of going through PDF-conversion is to recover the images. Keep them in `figures/`.
- **Letting the converter "skip" pages that failed to parse** — set `--strict` so any dropped page becomes an explicit error. Silent page-drops produce confidently-wrong summaries.
- **Round-tripping numeric tables through an LLM summarizer** — always include tables verbatim from `tables/tab_N.md`. Never paraphrase cell values.
- **Re-implementing indexing/navigation in `pdf-reader`** — you'll drift from `arxiv-latex-reader`'s semantics. Delegate.
- **Using one giant marker call on a 300-page PDF without checkpointing** — marker can take 10+ min. Use `--max-pages` and resume from `conversion.log` on failure.

## See Also

- `arxiv-latex-reader` — Stage 2 of this pipeline; use directly when LaTeX source is available
- `blog-reader` — Same coverage-test discipline applied to HTML blog posts
- `academic-deep-research` — Paper scoring and surveys; consumes paper metadata, not full text

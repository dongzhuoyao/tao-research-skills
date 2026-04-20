---
name: github-reader
description: "Use when reading a GitHub repository (especially a research code release with an accompanying paper) and producing a faithful digest that covers the implementation logic, the main insight, and the key reported results. Research-first with graceful fallback for non-paper repos. Handles arXiv link detection and delegates paper reading to arxiv-latex-reader / pdf-reader. Triggers: \"read repo\", \"read this github\", \"analyze github\", \"digest repo\", \"github reader\", \"extract from github\", \"summarize this codebase\", \"read this code\", \"https://github.com/\""
---

# GitHub Reader

## When to Use

- Reading a GitHub research-code release (e.g. a paper implementation on CompVis, HuggingFace, or a lab org) and wanting the core idea + architecture + results in one pass
- Onboarding to an unfamiliar library or CLI tool and wanting a guided walkthrough rather than a file-by-file listing
- Following up an `academic-deep-research` or `arxiv-latex-reader` pass by reading the referenced code
- Any prompt that pastes a `https://github.com/<owner>/<repo>` URL with "read" / "summarize" / "explain" intent

Not for: PR or issue digests, commit-history analysis, private-repo auth flows, or blog-style write-ups (use `blog-reader` for those).

## Output

A single markdown file at `docs/github/YYYY-MM-DD-<owner>-<repo>.md` with **seven fixed sections**:

1. **Overview** — 1 paragraph, venue if paper.
2. **Main Insight** — 2–4 sentences from the paper abstract (if fetched) or README motivation.
3. **Architecture** — tree listing of 5–10 core files with one-line annotations.
4. **Implementation Walkthrough** — 300–500 words tracing **one canonical flow** (training step / sampling call / quickstart) with 10–30-line snippets pinned to `file:line`.
5. **Key Results** — metrics from paper tables when fetched, else README benchmarks.
6. **Reproducibility** — deps, hardware, exact run command.
7. **See Also** — paper link + related skills.

## Workflow

1. **Parse input** — extract `owner/repo` from URL; strip `/tree/<branch>`, query strings, trailing slash.
2. **Fetch metadata** via `gh api repos/<owner>/<repo>` (see `github-cli` skill): stars, description, topics, default branch, `size` (KB), language.
3. **Size gate** — if `size > 500000` (500 MB), ask the user before cloning. Opt-in via `--allow-large`.
4. **Shallow clone**:
   ```bash
   git clone --depth 1 --filter=blob:none \
     https://github.com/<owner>/<repo> /tmp/github-reader/<owner>-<repo>
   ```
5. **Paper detection** — grep `README*`, `CITATION*`, `paper.bib` for `arxiv.org/abs/`, `arxiv.org/pdf/`, or `.pdf` URLs.
6. **Paper digest (if link found)** — delegate:
   - arXiv → `arxiv-latex-reader` (index + abstract + Method + main results table)
   - PDF → `pdf-reader`

   Save to `<workspace>/paper_digest.md`. Pull abstract + top result table into the digest's Main Insight and Key Results sections.
7. **File selection** — `git ls-tree -r --name-only HEAD`, skip the list below, then rank:
   1. README-mentioned `.py` / `.ipynb` paths
   2. Entry points at root: `train*.py`, `main*.py`, `sample*.py`, `inference*.py`, `demo*.py`
   3. Largest `.py` in `model/ models/ nn/ network/ networks/ src/ lib/`
   4. One `configs/*.yaml` or `.json`

   **Skip:** `tests/`, `test_*.py`, `setup.py`, `docs/`, `examples/`, vendored deps, weight files (`.pt`, `.safetensors`, `.bin`), `__init__.py` unless >50 lines.

8. **Read core files** — full file for ≤500 lines; otherwise section-chunk by top-level `class` / `def` and read only the functions referenced by the canonical flow.
9. **Emit digest** — fill the 7-section template.
10. **Verify** — every section non-empty; ≥3 `file:line` pins in the walkthrough; `Key Results` non-empty when a paper was fetched.

## Picking the Canonical Flow

| Repo type | Canonical flow |
|---|---|
| Training code | `python train.py` → dataset → model init → forward → loss → optimizer step |
| Inference / sampling | `python sample.py` → model load → sampling loop → save output |
| Library | README quickstart example → 3 main call sites |
| CLI tool | `main()` entry → one representative command path |

Trace one flow end-to-end with `file:line` pins. Describing files independently turns the walkthrough into a listing and loses control-flow understanding — don't do it.

## Anti-Patterns

- **Cloning a huge repo silently** — always run the size gate; a 2 GB clone burns bandwidth and disk without buying anything.
- **Skipping the paper** — if an arXiv link is in the README, fetch it. Key Results sourced only from README hype is not a faithful digest.
- **File-by-file summary** — describes contents without showing how data flows through them. Use a canonical flow instead.
- **No `file:line` pins** — hand-wavy walkthrough reads authoritative but can't be verified. Always anchor.
- **Reading `tests/` or `__init__.py` and calling it core implementation** — filter those out before the ranking step.
- **Leaving "Key Results" empty** when the repo plainly cites benchmarks — either pull them from the paper or quote the README table.
- **Inventing numbers** — every metric in Key Results must trace to a source in the repo or paper. Cite it.

## Example Invocation

```
Read https://github.com/CompVis/zigma
```

Expected outcome: `docs/github/YYYY-MM-DD-compvis-zigma.md` with all seven sections populated, paper insight sourced from the ECCV 2024 arXiv PDF, walkthrough tracing `train.py` → model class → Mamba block forward pass.

## See Also

- `arxiv-latex-reader` — Invoked for deep-reading the underlying paper when an arXiv link is detected.
- `pdf-reader` — Invoked when the paper link is a direct PDF rather than arXiv.
- `github-cli` — Patterns for `gh api` and authenticated repo metadata.
- `academic-deep-research` — Adjacent skill for surveying a research topic rather than a single repo.
- `blog-reader` — Shares the "fetch-then-digest" pattern but for blog posts rather than code.

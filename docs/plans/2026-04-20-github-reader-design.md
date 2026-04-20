# github-reader — design

**Date:** 2026-04-20
**Category:** `skills/research/`
**Motivation:** Given a GitHub repo URL (e.g. `https://github.com/CompVis/zigma`), produce a faithful digest covering the three things users actually want to know: the basic code implementation logic, the main insight, and the key result. Research-first, with graceful fallback for non-paper repos.

## Scope

| Decision | Chosen | Rejected alternatives |
|---|---|---|
| Target repos | Research-first, graceful fallback for non-paper repos | Research-only (brittle); any repo (loses paper leverage) |
| Fetch strategy | `git clone --depth 1 --filter=blob:none` into a workspace | API-only (10+ HTTP round trips); hybrid (not worth the branching) |
| Output | Saved digest at `docs/github/YYYY-MM-DD-<owner>-<repo>.md` with a fixed 7-section template | Free-form summary (inconsistent); stdout only (non-persistent) |
| File selection | Hard-coded heuristic (README mentions → entry points → core-module dirs → one config) | LLM-picks after tree listing (flexible but opaque) |
| Walkthrough | One canonical flow (train / sample / quickstart) with `file:line` pins | File-by-file summary (reads like a listing, hides control flow) |
| Paper integration | Auto-delegate to `arxiv-latex-reader` / `pdf-reader` when arXiv link is detected | Print link and stop (misses the "main insight" payload) |
| Scripts | None — pure orchestration of `gh`, `git`, sibling skills, and Read/Grep | Custom Python (reinvents what `github-cli` + `arxiv-latex-reader` already do) |

## Workflow

1. **Parse input** — extract `owner/repo` from URL; strip `/tree/<branch>`, query strings, trailing slash.
2. **Fetch metadata** — `gh api repos/<owner>/<repo>` for stars, description, topics, default branch, size, language.
3. **Size gate** — if `size > 500000` (KB = 500 MB), prompt the user before cloning. Opt-in via `--allow-large`.
4. **Shallow clone** — `git clone --depth 1 --filter=blob:none https://github.com/<owner>/<repo> <workspace>/<owner>-<repo>`. Default workspace `/tmp/github-reader/`; `--keep` re-points to `docs/github/workspaces/`.
5. **Paper detection** — grep `README*`, `CITATION*`, `paper.bib` for `arxiv.org/abs/`, `arxiv.org/pdf/`, or `.pdf` URLs. Record matches.
6. **Paper digest (if detected)** — delegate:
   - arXiv → `arxiv-latex-reader` (index + abstract + results table)
   - PDF → `pdf-reader`
   Write output to `<workspace>/<owner>-<repo>/paper_digest.md`.
7. **Tree + file selection** — `git ls-tree -r --name-only HEAD`, filter by skip list, then rank:
   - README-mentioned `.py` / `.ipynb` paths (highest)
   - `train*.py`, `main*.py`, `sample*.py`, `inference*.py`, `demo*.py` at root
   - Largest `.py` file in `model/ models/ nn/ network/ networks/ src/ lib/`
   - One `configs/*.yaml` or `.json`
   Skip: `tests/`, `test_*.py`, `setup.py`, `docs/`, `examples/`, vendored deps, weight files (`.pt`, `.safetensors`, `.bin`), `__init__.py` unless >50 lines.
8. **Read core files** — full file for ≤500 lines; otherwise section-chunk by top-level `class`/`def` and read only functions referenced by the canonical flow.
9. **Emit digest** — fill the 7-section template at `docs/github/YYYY-MM-DD-<owner>-<repo>.md`.
10. **Verify** — every section non-empty, ≥3 `file:line` pins in the walkthrough, `Key Results` non-empty when a paper was fetched.

## Digest template

```markdown
# <owner>/<repo> — <short title>

**Repo:** <url>  ·  **Stars:** N  ·  **Paper:** <arxiv-link>

## Overview
<1 paragraph: what the project does, who built it, venue if applicable>

## Main Insight
<2–4 sentences sourced from paper abstract when available, else README motivation>

## Architecture
<tree listing of 5–10 core files with one-line annotations>

## Implementation Walkthrough
<300–500 words tracing one canonical flow (training step / sampling call / quickstart) with 10–30-line snippets pinned to file:line>

## Key Results
<paper tables when fetched, else README benchmarks; metrics, dataset, baseline deltas>

## Reproducibility
- **Deps:** <top-level deps>
- **Hardware:** <GPUs/RAM from README or configs>
- **Command:** <exact run command>

## See Also
- Paper: <url if exists>
- `arxiv-latex-reader` — deep-read the underlying paper
- `academic-deep-research` — survey related work
```

## Skill file layout

```
skills/research/github-reader/
└── SKILL.md
```

No `references/`, no scripts — the workflow is pure orchestration.

## Anti-patterns

- Cloning repos >500 MB without a size-gate prompt.
- Describing files independently instead of tracing a single flow.
- Leaving "Key Results" empty when a paper exists but was not fetched.
- Reading `tests/` or `__init__.py` and presenting them as core implementation.
- Hand-wavy walkthrough with no `file:line` pins.
- Skipping the paper even when an arXiv link is obvious in the README.

## Cross-references

- `arxiv-latex-reader` — invoked for paper reading when arXiv link found.
- `pdf-reader` — invoked for direct PDF links.
- `github-cli` — used for metadata fetch (`gh api`).
- `academic-deep-research` — adjacent skill for surveying a research topic rather than a single repo.
- `blog-reader` — shares the "fetch-then-digest" pattern but for blog posts instead of code.

## Verification plan

Before committing the skill:

1. Dry-run `meta-init` to confirm it finds 29 skills (`ls skills/*/*/SKILL.md | wc -l == 29`).
2. README updated in two places: badge count 28 → 29; Research table row added.
3. Memory sync clean: `python skills/infra/memory-sync/scripts/memory_sync.py check --repo .`.
4. Run the skill end-to-end against `https://github.com/CompVis/zigma` (the motivating example) and confirm all 7 digest sections are populated.

## Out of scope

- PR/issue digests (a different skill).
- Commit-history analysis or blame-driven attribution.
- Live repo monitoring / watch mode.
- Private repo auth (relies on `gh` being logged in; no bespoke token handling).
- Large-repo streaming without a full clone (deferred; `--allow-large` escape hatch is enough for now).

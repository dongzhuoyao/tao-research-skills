---
name: arxiv-latex-reader
description: "Use when reading large arxiv papers without context overflow. Progressive two-layer reading: index all sections (~2k tokens), then deep-read on demand. Never truncates. Triggers: \"read paper\", \"paper sections\", \"section index\", \"progressive reading\", \"paper_content.json\", \"section summary\""
---

# Arxiv LaTeX Reader

## When to Use

- Reading arxiv papers that exceed context window limits (50k–200k+ chars)
- Building a section-level overview before deep-diving into specific sections
- Extracting poster content from papers (skip Related Work, Appendix)
- Answering questions about a paper without loading the full text
- Any workflow that reads `paper_content.json` and needs to manage token budget

## Architecture

Two-layer progressive reading:

```
Layer 1: Section Index (~2k tokens, always in context)
  § Introduction    3.7k chars  → "• Diffusion models need annotations..."
  § Method          9.5k chars  → "• Three self-annotation granularities..."
  § Experiments    12.0k chars  → "• Self-labeled outperforms GT labels..."

Layer 2: On-demand full section read (no limit)
  Caller: "read §Method and §Experiments"
  → Returns complete text, NEVER truncated
```

## Three Components

### 1. Section Indexer (`scripts/section_indexer.py`)

Runs once per paper. Creates `section_index.json` with LLM summaries.

```bash
python scripts/section_indexer.py <paper_id>
python scripts/section_indexer.py 2403.13802 --force --workspace workspace
```

- Short sections (<32k chars): single LLM call → 2-3 bullet-point summary
- Long sections (≥32k chars): split into ~24k char chunks → summarize each → merge
- NEVER truncates — every character is read and summarized
- Cached: only builds once per paper unless `--force`

### 2. Section Reader (`scripts/section_reader.py`)

Stateless full-text retrieval. No LLM calls.

```bash
python scripts/section_reader.py <paper_id> Method Experiments
python scripts/section_reader.py <paper_id> --all
python scripts/section_reader.py <paper_id> --keyword attention
```

Identifier formats:
- By name: `["Method", "Experiments"]`
- By index: `[0, 2, 4]`
- By keyword: `"attention"` → fuzzy match on section names
- All: `"all"`

Cleans LaTeX artifacts (commands, accents, citations, math delimiters).

### 3. Paper Navigator (`scripts/paper_navigator.py`)

Orchestrator. Loads section index, dispatches reads as needed.

```bash
python scripts/paper_navigator.py <paper_id> overview    # ~2k token view
python scripts/paper_navigator.py <paper_id> poster      # smart poster sections
python scripts/paper_navigator.py <paper_id> read Method  # full section text
```

**`read_for_poster()`** — smart section selection:
- Skips: Related Work, Acknowledgments, Appendix
- Includes: Introduction, Method, Experiments/Results, Conclusion
- Saves ~40% tokens compared to loading full paper

## Python API

```python
from paper_navigator import PaperNavigator

nav = PaperNavigator("2403.13802", workspace_dir="workspace")

# Get overview (~2k tokens — keep in context)
overview = nav.get_overview()

# Deep-read specific sections (caller manages token budget)
sections = nav.read_full(["Method", "Experiments"])

# Smart read for poster extraction
context = nav.read_for_poster()
# → {"overview": "...", "sections": [...], "skipped": ["Related Work"]}

# Fuzzy keyword search
results = nav.read_by_keyword("attention")
```

## Data Flow

```
paper_content.json ──→ section_indexer.py ──→ section_index.json (~2k tokens)
        │                                            │
        │                                            ▼
        └──→ section_reader.py ◄── paper_navigator.py (orchestrator)
                    │
                    ▼
            Full section text (no truncation)
```

## File Layout

```
workspace/<paper_id>/
├── paper_content.json     # Full parsed paper (from poster-agent parse)
└── section_index.json     # Lightweight index (generated, cached)
```

## Dependencies

- Python 3.10+
- `anthropic` pip package (for section_indexer LLM calls)
- `paper_content.json` in workspace (from any LaTeX parser)

## Key Principles

1. **Never truncate** — long sections are chunk-summarized, not cut
2. **Always LLM-summarize** — no heuristic truncation for the index
3. **Cache** — `section_index.json` is built once, reused everywhere
4. **Stateless reader** — `section_reader` is a pure function, no LLM, no state
5. **Caller manages budget** — no limit on deep-reads; caller decides what to load

## Anti-Patterns

- **Loading full paper into context** — use the index to decide what to read first
- **Truncating long sections** — split and summarize recursively instead
- **Heuristic summaries** — always use LLM, even for short sections; heuristic is fallback only
- **Re-building the index** — check for cached `section_index.json` before calling the indexer
- **Hard-coding section names** — use fuzzy matching; paper structures vary widely

## See Also

- `academic-deep-research` — Paper evaluation and topic surveys using venue, citations, reproducibility scoring

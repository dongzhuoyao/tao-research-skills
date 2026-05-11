---
name: cc-pdf-reader
description: "Use when reading PDF papers via Claude Code CLI instead of Python converter tools. Delegates PDF-to-text conversion and summarization to Claude Code's native multimodal Read + WebFetch tools, handling multi-page papers via chunked reads and structured prompts. Triggers: \"cc-pdf-reader\", \"claude code pdf\", \"read pdf with claude\", \"claude pdf reader\", \"cc pdf\", \"claude code paper\""
---

# CC PDF Reader

## When to Use

- Reading a PDF paper when **Claude Code CLI is installed** and Python converters (marker-pdf, pymupdf4llm) are unavailable or too slow
- Quick paper skim — get a structured overview without waiting for full PDF conversion
- Papers under ~80 pages that fit within Claude Code's Read tool page limit
- Scenarios where you want to **avoid installing heavy Python deps** (marker-pdf, docling)
- Reading PDFs one-off without setting up a workspace directory
- Papers with heavy figures that Claude Code's multimodal reading handles natively

For batch processing, large-scale PDF ingestion, or automated pipelines, prefer `pdf-reader` with Python converters — it's faster and more scriptable.

## Architecture

Unlike `pdf-reader` which uses a Python converter pipeline, CC PDF Reader delegates the entire PDF reading task to **Claude Code CLI in print mode (`-p`)**:

```
user prompt "read this PDF"
  │
  ▼
Hermes Agent (you) ──→ claude -p 'Read <pdf> and produce a structured summary'
                             │
                             └──→ Claude Code's native Read tool
                                  │  • Reads PDF pages (up to ~20 per call)
                                  │  • Multimodal figure interpretation
                                  │  • Table extraction from rendered pages
                                  │
                                  └──→ Returns structured markdown output
```

No intermediate workspace, no Python converter scripts, no figure extraction pipeline. Claude Code reads the PDF directly using its built-in tools.

## Workflow

### Method 1: One-Shot Summary (Fastest)

For a quick overview of any PDF:

```bash
claude -p "Read the PDF at <path/to/paper.pdf>. Give me:
1. Title and authors
2. One-paragraph abstract summary
3. Key contributions (as bullet points)
4. Method overview (3-5 sentences)
5. Main results with numbers (table format)
6. My main takeaway

Be thorough — read every section. Use the Read tool to go page by page if needed." --allowedTools "Read" --max-turns 15 --output-format json
```

**When to use:** Quick skims, sanity checks, deciding if a paper is relevant.

### Method 2: Structured Deep Read (Recommended)

For serious paper comprehension, use a structured prompt with section-level coverage:

```bash
claude -p "I need a deep understanding of this PDF: <path/to/paper.pdf>

Read the entire paper carefully, section by section. For each section, tell me:
- **Section name** and **page range**
- **Core claim** (1-2 sentences)
- **Key evidence** (specific numbers, equations, or observations)
- **Figures/tables in this section** — describe what they show

After covering all sections, produce:
1. A **one-sentence summary** of the paper
2. Top 3 **strengths**
3. Top 3 **weaknesses / limitations**
4. **How this relates to** (the user's specific project or context)

Read tool will handle up to ~20 pages per call. If the PDF is longer, take it one chunk at a time and tell me you're continuing." --allowedTools "Read" --max-turns 30 --output-format json
```

**When to use:** Papers you actually want to understand deeply and reference later.

### Method 3: Chunked Multi-Session (Long PDFs, 50+ pages)

For long documents like theses, technical reports, or books, chunk manually:

```bash
# Round 1: Document overview + first chunk
claude -p "Read pages 1-20 of <path/to/long.pdf>. Give me:
- Title, authors, abstract
- Table of contents / structure
- Full content of sections covered in these pages

Then wait for my instruction before continuing." --allowedTools "Read" --max-turns 12 --output-format json
```

```bash
# Round 2: Continue
claude -p "Continue reading <path/to/long.pdf>. Read pages 21-40 now.
Give me full content with the same level of detail." --allowedTools "Read" --max-turns 12 --output-format json
```

**When to use:** Theses, books, multi-hundred-page reports.

### Method 4: Figure-Focused Read

When figures carry the main information (architecture diagrams, result plots):

```bash
claude -p "Read <path/to/paper.pdf> but focus on the **figures and tables**. For each one:
1. What page and figure/table number
2. What it shows (detailed description)
3. What the authors want you to conclude from it
4. Any related text passage that explains it

Also give me: the paper title, and a 2-sentence summary of the paper's goal." --allowedTools "Read" --max-turns 15 --output-format json
```

**When to use:** Architecture papers, benchmark comparisons, poster-style papers where figures are the main artifact.

## Prompt Design Principles

These are the key patterns that make CC PDF Reader prompts effective:

1. **Explicit structure** — Tell Claude the output format you want (sections, bullet points, tables). Don't let it infer.

2. **Page-by-page instruction** — The `Read` tool loads up to ~20 PDF pages per call. Add phrases like "go page by page if needed" or "tell me when you need to continue reading" to prevent Claude from stopping prematurely.

3. **Section coverage** — Explicitly ask for "every section" or list expected sections (Introduction, Method, Experiments, Conclusion). Claude tends to skip sections otherwise.

4. **Figure verbatim rule** — Claude Code's multimodal reader renders PDF figures as images. Tell it to "describe each figure in detail" if figures matter.

5. **Number preservation** — Tell Claude to preserve all numbers verbatim. It has a strong tendency to round or summarize numeric results.

6. **Output formatting** — Request markdown output with clear headings. This makes the result immediately useful for further processing or note-taking.

## Comparison: CC PDF Reader vs. pdf-reader

| Aspect | **pdf-reader** (Python) | **cc-pdf-reader** (Claude Code) |
|--------|------------------------|----------------------------------|
| Dependencies | marker-pdf, pymupdf, poppler, etc. | Claude Code CLI only |
| Setup time | Minutes (pip install) | Instant (if claude already installed) |
| Speed | Fast (seconds on GPU, minutes CPU) | Slow (LLM per-token generation) |
| Cost | Free (Python tools) | ~$0.05-1.00 per paper (Claude API) |
| Figure handling | Extracted as PNG files | Rendered natively by Claude multimodal |
| Table handling | Markdown tables from PDF parser | LLM-interpreted from rendered pages |
| Math/equations | LaTeX preserved (marker-pdf) | LLM-interprets; may lose LaTeX fidelity |
| Batch processing | Easy (loop over files) | Impractical (cost + time per file) |
| Automation / cron | Scriptable, headless | Needs Claude Code CLI available |
| Long PDFs (>80pp) | Handles naturally | Chunking required |
| Output structure | Structured JSON workspace | Free-form markdown (prompt-dependent) |

## When to Choose Which

| Scenario | Use |
|----------|-----|
| One-off paper read | `cc-pdf-reader` (fastest to invoke) |
| Batch of 10+ papers | `pdf-reader` (automated pipeline) |
| Heavy-equation paper (math-heavy) | `pdf-reader` + marker-pdf (preserves LaTeX) |
| Architecture diagram paper | `cc-pdf-reader` (better multimodal figure reading) |
| Figure extraction needed | `pdf-reader` (separate PNG files) |
| Automated daily paper digest | `pdf-reader` (scriptable, cron-friendly) |
| No Python ML libs installed | `cc-pdf-reader` (only needs claude CLI) |
| Pay-per-use budget tight | `pdf-reader` (free converters) |

## Anti-Patterns

- **Using CC PDF Reader for batch processing** — $0.05-1.00 per paper adds up fast; use Python converters for bulk work.
- **Not specifying output structure** — Claude defaults to prose paragraphs which are hard to scan. Always specify bullet points, tables, or sections.
- **Assuming Claude reads every page** — The Read tool has page limits. Say "go page by page" or chunk explicitly.
- **Missing figures** — Claude may skip figures unless you explicitly ask for them. Add "describe each figure and table in detail" to your prompt.
- **Trusting rounded numbers** — Claude rounds experimental results. Always add "preserve all numbers verbatim" to your prompt.
- **Using `--max-turns` too low** — 5 turns is often insufficient for a full paper read. Start with 15-30.
- **Feeding scanned/image-only PDFs** — Claude Code's Read tool depends on text layer. For scanned PDFs, use `pdf-reader` with OCR tools.
- **Expecting LaTeX-level equation fidelity** — Claude can read the rendered equation from the PDF page but won't output LaTeX. If you need the raw LaTeX source, prefer `arxiv-latex-reader` on the arxiv e-print.
- **Running interactively instead of `-p`** — Interactive mode adds dialog overhead. Print mode (`-p`) is cleaner for structured output.
- **Not checking `--allowedTools "Read"`** — Claude Code may try to use Bash or other tools to convert the PDF, adding cost and time. Restrict to `Read` for simple paper reading.

## See Also

- `pdf-reader` — Python-based PDF conversion pipeline (marker-pdf / pymupdf4llm), for batch and automated PDF reading
- `arxiv-latex-reader` — Progressive two-layer paper reading from LaTeX source; preserve equations at source level
- `blog-reader` — Same coverage-test discipline applied to HTML blog posts (section-based, parallel-subagent)
- `claude-code` — Full Claude Code orchestration guide for Hermes agents

---
name: blog-reader
description: "Use when reading a long technical blog post (ML research, engineering deep-dives, Distill/Lil'Log-style posts) and producing a faithful, figure-aware summary. Handles context-over-limit via section-based chunking, captures important figures via multimodal Read, and runs a coverage test to catch missing information. Triggers: \"read this blog\", \"summarize this post\", \"read blog\", \"digest this article\", \"long blog post\", \"read this article\", \"blog summary\", \"distill post\", \"summarize url\", \"chunk and summarize\""
---

# Blog Reader

## When to Use

- Summarizing a long blog post that would overflow a single WebFetch response
- Reading technical posts with many figures (architecture diagrams, result plots) that must be understood, not just listed
- Producing a permanent, faithful note from a blog post (saved to `docs/blogs/`)
- Digesting a series of posts on the same topic (batch mode)

Do **not** use for: short blog posts that fit in one WebFetch call — just call WebFetch directly. For academic papers, use `academic-deep-research` instead.

## Pipeline Overview

```
┌──────────────────────────┐
│ 1. Fetch + normalize     │  WebFetch / curl → clean markdown
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 2. Extract structure     │  headings, figures, code blocks, quotes
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 3. Chunk by section      │  ~4-6k tokens per chunk, never mid-section
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 4. Figure pass           │  classify → Read important ones (multimodal)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 5. Summarize per-chunk   │  parallel subagents, structured output
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 6. Synthesize            │  merge chunk notes + figure notes → final
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 7. Coverage test         │  every section + claim traceable to source
└──────────┬───────────────┘
           ▼
    docs/blogs/YYYY-MM-DD-<slug>.md
```

## Step 1 — Fetch and Normalize

### URL input

```
WebFetch → <url>  with prompt "Return the full page as markdown. Preserve all headings, figure captions, image URLs, code blocks, and block quotes. Do not summarize."
```

If the blog is behind a paywall, rendered client-side, or WebFetch returns truncated content, fall back to:

```bash
curl -sL --compressed "$URL" -o /tmp/blog.html
python3 -c "import html2text, sys; print(html2text.html2text(open('/tmp/blog.html').read()))" > /tmp/blog.md
```

### Local input

```
Read → <path>    # if the user already saved the blog as .md or .html
```

### Save the raw source

Always save the normalized markdown to `/tmp/<slug>.md` **before** chunking. The verification step (Step 7) greps against this file — if you lose the raw source you cannot verify.

## Step 2 — Extract Structure

Scan the normalized markdown and build an index:

```
sections:   list of (level, heading, start_line, end_line)
figures:    list of (figure_num, caption, img_url, nearest_section)
code_blocks: list of (lang, line_count, nearest_section)
quotes:     list of (source, line)
```

Use Grep to extract:
- Headings: `^#{1,6} `
- Images: `!\[.*\]\(.*\)` or `<img [^>]*src="[^"]+"`
- Code fences: ` ``` `
- Block quotes: `^> `

Write the index to `/tmp/<slug>.index.json` for reference.

## Step 3 — Chunk by Section

### Size budget

| Source size | Strategy |
|-------------|----------|
| < 8k tokens (~32 KB) | Single chunk — skip to Step 5 |
| 8-30k tokens | Chunk by H2, one chunk per H2 |
| 30-80k tokens | Chunk by H2, but merge small H2 siblings (<2k tokens) |
| > 80k tokens | Chunk by H2 + parallel subagents (see Step 5) |

### Rules

- Never split mid-section, mid-paragraph, or mid-code-block.
- Every chunk carries the parent H1 and a one-line "what came before" breadcrumb.
- If a single H2 is > 8k tokens, split further by H3, but keep figure + caption + referencing paragraph together.
- Never drop block quotes, equations, or numeric results during chunking — they are verbatim evidence.

### Chunk template

```markdown
<!-- chunk {i}/{n} — {section_heading} -->
<!-- breadcrumb: {H1 title} → {prior H2 if any} -->

## {section_heading}

{section_body_verbatim}
```

Save each chunk as `/tmp/<slug>.chunk.<i>.md`.

## Step 4 — Figure Pass

Figures in technical blogs are **load-bearing** — an architecture diagram or a loss curve often encodes the whole argument. Treat them as first-class content.

### Classify every figure

| Class | Signals | Action |
|-------|---------|--------|
| **Key** | Referenced in body text ("see Figure 2"), architecture/overview diagram, headline result plot | Download + multimodal Read |
| **Supporting** | Secondary result, ablation plot, related work comparison | Download + multimodal Read if budget allows |
| **Decorative** | Hero image, logo, unrelated stock photo, section divider | Skip — keep URL only |

### Download + read

```bash
mkdir -p /tmp/<slug>.figs
for url in <key_urls>; do
  curl -sL -o "/tmp/<slug>.figs/fig_N.png" "$url"
done
```

```
Read → /tmp/<slug>.figs/fig_N.png
      # ask yourself: what does this diagram claim? what axes / nodes / arrows?
      # write a 3-5 bullet summary. quote any in-image text verbatim.
```

Save figure notes to `/tmp/<slug>.fig_notes.md`:

```markdown
### Figure 2 — "Model architecture overview"
- url: <url>
- caption: "<verbatim caption>"
- claims: 3 encoder blocks → shared backbone → 2 decoder heads
- axes / labels: input size 224×224; hidden dim 768
- in-image text: "MoE router", "top-k=2"
- referenced in: §3.1 ("as shown in Figure 2")
```

### When the figure is broken or inaccessible

- Record `status: unreachable` and the URL.
- Do **not** hallucinate what the figure shows. The final summary must flag missing figures explicitly.

## Step 5 — Summarize Per-Chunk

### Solo mode (small blog)

Read the single chunk + figure notes, produce the structured summary from Step 6 in one pass.

### Parallel subagent mode (long blog)

Dispatch one subagent per chunk **in parallel** (single message, multiple Agent calls). Give each subagent:

- The chunk file path
- The relevant figure notes (only figures in that chunk's section)
- The breadcrumb (H1 + prior H2)
- A **structured output contract** — everyone returns the same shape

**Subagent prompt template:**

```
Read /tmp/<slug>.chunk.<i>.md and /tmp/<slug>.fig_notes.md (figures for
section "<section_heading>" only). Produce a structured note with:

1. tldr: 1-2 sentences, what does this section argue?
2. key_points: 3-7 bullets, each a distinct claim or finding
3. definitions: any new terms introduced (term → gloss)
4. numbers: all numeric results, hyperparams, dates, model sizes (verbatim)
5. quotes: 1-3 short verbatim quotes worth preserving
6. figures_used: list figure numbers referenced in this section
7. open_questions: anything the section asserts without justification

Do NOT add information not in the chunk. If unsure, quote directly.
Report under 400 words.
```

Collect each subagent's output into `/tmp/<slug>.notes.<i>.md`.

## Step 6 — Synthesize

Merge chunk notes into the final report (see template below). Rules:

- Preserve **every** numeric result from any chunk's `numbers:` field.
- Preserve **every** figure reference. If Figure N was mentioned, it must appear in the final either as an embedded image or a figure note.
- Preserve at least one verbatim quote per major section.
- Do **not** add synthesis claims that no chunk supports.

### Output template

Save to `docs/blogs/YYYY-MM-DD-<slug>.md`:

```markdown
# <Blog title>

**Source**: <url>
**Author(s)**: <...>
**Published**: <date>
**Read on**: YYYY-MM-DD
**Length**: ~<N> words, <M> figures
**Reading mode**: single | chunked-<N>

## TL;DR

<3-5 bullet executive summary>

## Why it matters

<1-2 sentences: what does this change / claim / ship?>

## Section summaries

### §1 — <heading>
- <key points>
- Figures: <Fig 1, Fig 2>
- Numbers: <verbatim>

### §2 — <heading>
...

## Key figures

### Figure 2 — <caption>
![](<local or remote url>)
<3-5 bullet interpretation from Step 4>

## Notable quotes

> "<verbatim>"
> — §X

## Numbers & results

| Metric | Value | Context |
|--------|-------|---------|
| ... | ... | ... |

## Open questions / gaps

- <things the post asserts without evidence>
- <broken/missing figures>

## Verification log

- sections covered: N/N
- figures covered: K/M (key: all; decorative: skipped)
- claims traced: see Step 7
```

## Step 7 — Coverage Test

This is the **mandatory** final step. Without it you cannot claim the summary is faithful.

### 7a. Section coverage

Every H2 in the original must appear in "Section summaries". Run:

```bash
# headings in source
grep -n '^## ' /tmp/<slug>.md | awk -F'## ' '{print $2}' | sort > /tmp/src.h2
# headings in summary
grep -n '^### §' docs/blogs/YYYY-MM-DD-<slug>.md | sed 's/.*§[0-9]* — //' | sort > /tmp/out.h2
diff /tmp/src.h2 /tmp/out.h2
```

If `diff` is non-empty, go back and add the missing sections.

### 7b. Claim traceability

Pick **every** bullet from the TL;DR and every row in "Numbers & results". For each:

```
Grep → /tmp/<slug>.md  for a distinctive phrase or number from the claim
```

If a claim has no hit in the source, **delete it** or rewrite it to quote the source directly. Log the result:

```markdown
## Verification log
- TL;DR bullets traced: 5/5  ✅
- Numbers traced: 12/12  ✅
- Figure references traced: 8/8  ✅
- Removed during verification: <list, if any>
```

### 7c. Figure coverage

Every figure classified as **key** in Step 4 must appear under "Key figures" in the final report. Missing figure → go back and add (or mark as unreachable).

### 7d. Hallucination check

Re-read the final report end-to-end. For each sentence, ask: *can I point to the source line or figure that supports this?* If no → remove or quote.

## Anti-Patterns

- **Single WebFetch on a 20k-word post** — WebFetch silently truncates. If the normalized markdown is suspiciously short (<2k tokens for a known long-form post), refetch via curl + html2text.
- **Listing figures as URLs without reading them** — an architecture diagram summarized as "Figure 2: architecture overview" is zero information. Use Read for multimodal understanding.
- **Chunking by character count mid-section** — breaks the logical argument and causes per-chunk summaries to misinterpret. Always chunk on section boundaries.
- **Skipping Step 7** — without coverage testing you have no evidence the summary is faithful. A plausible-sounding summary that omits a core section is worse than a short but verified one.
- **Paraphrasing numeric results** — "achieves ~90% accuracy" when the post says "89.3%" is a lossy paraphrase. Quote numbers verbatim.
- **Hallucinating figure contents** — if curl fails or the image is behind auth, mark unreachable. Do not guess what an architecture diagram shows from the caption alone.
- **Losing the raw source** — the raw normalized markdown is your ground truth for verification. Keep it under `/tmp/<slug>.md` until verification passes.
- **One giant subagent for the whole post** — defeats the chunking purpose. One subagent per chunk, dispatched in parallel.
- **Subagents returning free-form prose** — hard to merge. Enforce the structured output contract in Step 5.
- **Treating equations and code blocks as prose** — they must be preserved verbatim, never paraphrased.

## See Also

- `academic-deep-research` — For arxiv papers and academic research, not blog posts
- `github-cli` — For fetching GitHub READMEs and release notes as "blog" content
- `agents-md-writing` — For turning blog findings into project memory files

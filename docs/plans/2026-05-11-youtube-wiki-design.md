# youtube-wiki — design

**Date:** 2026-05-11
**Category:** `skills/research/`
**Motivation:** Given a YouTube URL (typically an AI interview, e.g. `https://youtu.be/ttkd0t5qTD4`), produce a faithful, timestamped wiki entry that the user can return to weeks later — searchable text, clickable timestamps as doors back into the video, verbatim quotes with speaker attribution, and a coverage test that catches paraphrase drift. Sits alongside the existing reader family (`blog-reader`, `pdf-reader`, `arxiv-latex-reader`, `github-reader`).

## Scope

| Decision | Chosen | Rejected alternatives |
|---|---|---|
| Output shape | Faithful digest, blog-reader-shaped, one markdown file per video | Q&A-first interview format (awkward for non-interviews); growing topic-card wiki (too ambitious for v1) |
| Output location | `docs/videos/YYYY-MM-DD-<slug>.md` | Per-channel folders (premature); flat `docs/youtube/` (less semantic) |
| Visual content | Transcript-only; flag `[visual ref]` timestamps inline | Periodic frame screenshots (doubles runtime, marginal gain for talking-head interviews); on-demand frames (still adds complexity) |
| Sectioning | Hybrid: YouTube chapters → LLM topic detection → 10-min fallback | Pure time-based (semantically blind); pure LLM (wastes a pass when chapters exist) |
| Timestamps | Clickable timestamps everywhere — `[MM:SS](https://youtu.be/<id>?t=<sec>)` | Plain text timestamps (loses the door-back-to-video value); none (defeats the purpose) |
| Speaker attribution | Labels on quotes only (`[Host]`/`[Guest]`/names); un-labeled in summaries; `Speaker A/B` fallback when low confidence | Always-labeled (mislabels in panels); never-labeled (misattribution risk on quotes); auto-name-detect everywhere (too brittle) |
| Acquisition | **YouTube subs first** (manual or auto, prefer original-language code) **→ Whisper fallback** only when no subs exist | Always-Whisper (rejected after dogfooding — see "Dogfood findings" below); always-YouTube-subs (rejected — some videos genuinely have no subs) |
| Whisper fallback backends | Three documented choices: (A) `mlx_whisper` CLI, (B) Rapid-MLX server (when already running for summarization LLM), (C) OpenAI Whisper API (cloud, $0.006/min, bypasses local RAM limits) | Single forced backend (loses portability across machines with very different RAM/GPU profiles); mlx-whisper only (Apple-Silicon only, can OOM on 16 GB machines) |
| Default Whisper model (A/B) | `mlx-community/whisper-large-v3-mlx`, with `large-v3-mlx-4bit` as the low-RAM swap | Turbo (loses accuracy on Chinese-English code-switch terms — load-bearing content for AI interviews) |
| Cloud Whisper model (C) | `whisper-1` (only model OpenAI hosts) — accept that brand-name accuracy is weaker than YouTube subs for Chinese podcasts | `gpt-4o-transcribe` (no VTT output, harder to integrate) |
| Failure mode | Hard stop on yt-dlp failure; explicit (announced + header-logged) cascade from YouTube subs to Whisper; never silently switch among Whisper backends mid-pipeline | Silent fallback to a smaller model or to a different backend (violates no-silent-fallbacks rule) |
| Scripts | None — pure orchestration of `yt-dlp`, `ffmpeg`, `mlx_whisper`/`curl`, `jq`, Read/Grep, parallel subagents | Custom Python helpers (existing tools cover acquisition; the rest is summarization logic in the SKILL.md itself) |

## Dogfood findings (2026-05-11)

Tested on `https://youtu.be/ttkd0t5qTD4` (3h48m Mandarin AI interview, Yao Shunyu). All three Whisper backends fully exercised:

- **YouTube subs (en-US auto-track, mislabeled as manual):** 144 k chars, 6878 utterances. Captured every load-bearing English token — "Claude", "Jared Kaplan", "Hinton", "nanoGPT", "SWE-bench", "Scaling Law", "OpenAI".
- **mlx-whisper local `large-v3`:** Failed — Python process stuck at 0 % CPU for 20+ minutes on M4 Mac mini. Model is 3 GB on disk; with Chrome + MCP servers running, available unified memory wasn't enough for mmap. Tiny model worked fine; turbo / 4bit not yet tested.
- **OpenAI `whisper-1` API:** 16 chunks × 15-min, 4 minutes wall-clock, $1.44 total. 125 k chars, 5467 utterances (87 % of YouTube's char count). Failed on the technical English tokens that matter most: transcribed 姚顺宇 as 姚舜宇, said "OpenEye" instead of "OpenAI", dropped "Claude" entirely. Clean for general Mandarin prose; poor for AI-interview code-switching.

→ YouTube subs win for AI-interview content in Mandarin. Whisper is the fallback for videos with no subs at all.

## Pipeline

```
┌───────────────────────────────┐
│ 1. Fetch transcript + meta    │  yt-dlp → vtt + info.json (title, chapters, duration)
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 2. Normalize transcript       │  vtt → timestamped plain lines, dedupe karaoke overlap
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 3. Section the video          │  chapters → LLM topics → 10-min fallback
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 4. Chunk by section           │  one chunk per section, carries timestamps + breadcrumb
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 5. Summarize per-chunk        │  parallel subagents, structured contract
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 6. Synthesize wiki entry      │  TL;DR + sections + quotes + indices
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 7. Coverage test              │  sections, quotes, numbers all traced
└──────────┬────────────────────┘
           ▼
   docs/videos/YYYY-MM-DD-<slug>.md
```

## Acquisition

Cascade: YouTube subs first → Whisper only when subs missing. Both paths produce a VTT in the same shape, so the rest of the pipeline is source-agnostic. The chosen source is recorded in the final wiki header so the reader knows the transcript's provenance.

```bash
# Step 1b — primary: YouTube subs + info.json (no audio download)
yt-dlp --list-subs "$URL" 2>&1 | tail -40            # discover available codes
yt-dlp --write-sub --write-auto-sub \
       --sub-lang "en-US,en-GB,en,zh-Hans,zh-Hant,zh-CN,zh-TW" \
       --skip-download --write-info-json \
       --output "/tmp/yt-wiki/<slug>.%(ext)s" \
       "$URL"
# If any .vtt was written → done, proceed to Step 2.
```

```bash
# Step 1c — fallback: audio + Whisper, only when Step 1b emitted no .vtt
yt-dlp --extract-audio --audio-format m4a \
       --output "/tmp/yt-wiki/<slug>.%(ext)s" "$URL"

# Pick ONE Whisper backend (preflight that at least one is available):
# A) mlx-whisper CLI
mlx_whisper "/tmp/yt-wiki/<slug>.m4a" --model mlx-community/whisper-large-v3-mlx \
            --output-format vtt --output-dir /tmp/yt-wiki --output-name "<slug>"
# B) Rapid-MLX server (when already running for the summarization LLM)
curl -fs http://localhost:8000/v1/audio/transcriptions \
     -F file=@/tmp/yt-wiki/<slug>.m4a -F model=default -F response_format=vtt \
     -F language=auto -o /tmp/yt-wiki/<slug>.vtt
# C) OpenAI Whisper API (chunk to ≤25 MB, transcribe each, concat with offsets)
ffmpeg -y -i /tmp/yt-wiki/<slug>.m4a -f segment -segment_time 900 -c copy \
       -reset_timestamps 1 /tmp/yt-wiki/chunks/<slug>_%03d.m4a
# (then curl POST each chunk, concatenate VTTs with timestamp offsets — see SKILL.md)
```

- On Step 1b failure (region lock, age gate, missing cookies, missing yt-dlp): stop and surface the specific error. No silent fallback to scraping.
- On Step 1c failure (OOM, HF download blocked, OpenAI 401/413): stop and surface. No silent fallback to a smaller model — that would silently degrade wiki quality.
- The YouTube-subs → Whisper cascade itself is **explicit**: it only runs when 1b emitted no `.vtt`, and the chosen source is logged in the wiki's verification block.

## Normalization

YouTube `.vtt` cues repeat — each line shows up 2-3 times as the karaoke highlight moves. Strip via a small dedupe loop, keep one line per distinct utterance with the start timestamp:

```
[00:00:12] welcome back to the show today I'm here with
[00:00:18] our guest who's been working on
[00:00:22] foundation model evaluation at...
```

Save to `/tmp/yt-wiki/<slug>.transcript.txt`. This is the ground-truth file for Step 7 verification — keep until done.

## Sectioning

Three-tier strategy, no silent fallback between them:

1. **YouTube chapters.** Parse `info.json` for a `chapters` array. Each entry has `start_time`, `end_time`, `title`. Use directly when present (most quality channels add these).
2. **LLM topic detection** (only when `chapters` is null/empty). Single subagent reads the full normalized transcript, returns JSON: `[{start: "MM:SS", title: "..."}]`. Aim for 5-12 sections for a 60-90 min video.
3. **Time-based fallback** (when LLM detection fails or returns nonsense). Fixed 10-minute windows labeled `[00:00-10:00]`, etc.

Record which method was used — surfaces in the final report's metadata.

## Chunking

One chunk per section. Header carries breadcrumb so the subagent has context without seeing the rest:

```
<!-- chunk {i}/{n} — {section_title} -->
<!-- range: 00:12:30 → 00:24:15 -->
<!-- video: {title} ({url}) -->
<!-- breadcrumb: prior section = "{prev_title}" -->

[12:34] ...transcript lines verbatim...
```

Saved as `/tmp/yt-wiki/<slug>.chunk.<i>.md`.

## Summarization

Parallel subagents (single message, multiple Agent calls). Structured output contract per chunk:

```yaml
tldr: 1-2 sentences — what is this section about?
key_points: 3-7 bullets — distinct claims/findings/anecdotes
people_mentioned:
  - name: ...
    context: ...
    timestamp: MM:SS
papers_or_works_mentioned:
  - title: ...
    url: (if findable)
    timestamp: MM:SS
quotes:
  - speaker: "[Host]" | "[Guest]" | "[Speaker A]" | name
    text: "verbatim — filler may be elided with [...]"
    timestamp: MM:SS
numbers:
  - claim: "verbatim"
    timestamp: MM:SS
visual_refs:
  - timestamp: MM:SS
    cue: "as you can see" | "this diagram" | etc.
open_questions:
  - claim asserted without justification
```

Rules baked into the subagent prompt:
- Quotes MUST be verbatim. Only filler words (`uh`, `you know`, `I mean`) may be elided as `[…]`. No paraphrase.
- Speaker labels: `[Host]`/`[Guest]` when conversational role is obvious; named labels (`[Alec Radford]`) when name was introduced on-mic; `[Speaker A]`/`[Speaker B]` when low-confidence. Never invent names.
- Section summaries themselves stay un-labeled (per Q4-C).
- Visual refs: timestamp + the verbal cue only. Don't try to describe what's on screen.

Subagent output saved to `/tmp/yt-wiki/<slug>.notes.<i>.md`.

## Output template

Saved to `docs/videos/YYYY-MM-DD-<slug>.md`:

```markdown
# <Video Title>

**Source**: https://youtu.be/<id>
**Channel**: <uploader>
**Published**: YYYY-MM-DD
**Duration**: HH:MM:SS
**Watched on**: YYYY-MM-DD
**Sectioning**: chapters | llm-detected | time-based
**Speakers**: <Host>, <Guest>  (or "Speaker A, Speaker B")

## TL;DR

<3-5 bullets — executive summary of the entire conversation>

## Why it matters

<1-2 sentences — what claim, idea, or framing is worth coming back for?>

## Section summaries

### §1 — <chapter title>  [00:00 → 12:34](https://youtu.be/<id>?t=0)
- <key points>
- People: <names + clickable timestamps>
- Numbers: <verbatim + clickable timestamps>
- Visual refs: [[12:01](url?t=721)], [[12:18](url?t=738)]

### §2 — <chapter title>  [12:34 → 24:15](https://youtu.be/<id>?t=754)
...

## Notable quotes

> "<verbatim quote>"
> — [Guest], §3 [[18:42](https://youtu.be/<id>?t=1122)]

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| ... | ... | [12:34](...) |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| ... | ... | [...] |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| ... | ... | [...] |

## Open questions / gaps

- <claims asserted without evidence>

## Verification log

- sections covered: N/N
- quotes traced verbatim: K/K
- numbers traced: M/M
- sectioning method used: chapters | llm-detected | time-based
```

## Coverage test (mandatory)

7a. **Section coverage.** Every section emerging from Step 3 appears under "Section summaries". `diff` between section titles and emitted H3 headings → empty.

7b. **Quote verbatim check.** For every quote in "Notable quotes", grep the transcript for a distinctive 5-8-word phrase from it. No hit → either paraphrase crept in (delete the quote) or filler-elision lost the anchor (rewrite with a tighter unbroken span).

7c. **Number traceability.** Every row in "Numbers & claims" must have a transcript hit (the number itself or a distinctive surrounding phrase). No hit → delete or quote source directly.

7d. **Hallucination sweep.** Re-read end-to-end. For each TL;DR bullet, point at a section + timestamp that supports it. No support → remove.

Failures stop and surface — never "looks plausible enough."

## Skill file layout

```
skills/research/youtube-wiki/
└── SKILL.md
```

No `references/`, no scripts. yt-dlp + jq + Read/Grep + parallel subagents cover everything; the rest is summarization logic that lives in the SKILL.md itself (mirroring `blog-reader`).

## Anti-patterns

- Paraphrasing quotes. Verbatim only; only filler `[…]` elision allowed.
- Hallucinating what's on screen from a `visual_refs` timestamp.
- Confident speaker attribution when the video is a 3+ person panel.
- Time-based sectioning when `info.json` had chapters all along (always check first).
- Skipping the coverage test.
- Inventing "papers mentioned" when the speaker made only a vague allusion.
- Silent fallback to browser scraping when `yt-dlp` fails. Hard stop, surface the error.
- Truncating long transcripts. Chunk by section; subagents in parallel.
- Single huge subagent for a 90-min interview. One per section.

## Cross-references

- `blog-reader` — direct cousin; same pipeline shape, different acquisition.
- `pdf-reader` / `arxiv-latex-reader` — for paper digests; useful when a video references a paper that's worth following.
- `github-reader` — for repo digests when the interview mentions a project.
- `academic-deep-research` — for evaluating papers a guest mentions.
- `agents-md-writing` — for promoting recurring video findings into project memory.

## Verification plan

Before committing:

1. Structural checks pass:
   - `ls skills/*/*/SKILL.md | wc -l` → 30
   - Frontmatter has exactly `name` and `description`
   - `name` matches dir; `description` starts with "Use when", ends with `Triggers:` list
   - Sections include `## When to Use` (first) and `## See Also` (last)
2. README updated: badge count 29 → 30, "29 self-contained" text updated, new row under Research & Reading.
3. Memory sync clean: `python skills/infra/memory-sync/scripts/memory_sync.py check --repo .`.
4. Dry-run on `https://youtu.be/ttkd0t5qTD4` and inspect the emitted wiki entry against `/tmp/yt-wiki/<slug>.transcript.txt` — sections complete, quotes verbatim, timestamps clickable.

## Out of scope

- Frame screenshots and on-screen content (deferred — YAGNI per Q2-D).
- Cross-video topic indexing into a growing wiki (a future skill, not v1).
- Live stream / unreleased video handling.
- Speaker-diarization audio analysis (transcript-only).
- Channel batch ingestion (one URL at a time).
- Private / paywalled videos beyond what cookies-from-browser can handle.

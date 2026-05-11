---
name: youtube-wiki
description: "Use when reading a YouTube video (especially an AI/ML interview, podcast, or technical talk) and producing a faithful, timestamped wiki entry the user can return to weeks later. Fetches the transcript via yt-dlp, sections by chapters or LLM-detected topics, summarizes per-section with parallel subagents, preserves verbatim quotes with speaker attribution, and runs a coverage test against the raw transcript. Triggers: \"youtube wiki\", \"video wiki\", \"summarize this youtube\", \"watch this interview\", \"read this talk\", \"digest this video\", \"https://youtu.be/\", \"https://www.youtube.com/watch\""
---

# YouTube Wiki

## When to Use

- Reading a long YouTube interview, podcast, or technical talk and wanting a searchable, timestamped wiki entry to return to later
- AI/ML interviews where the substance is in the conversation: claims, anecdotes, paper references, named people
- Any prompt that pastes a `https://youtu.be/<id>` or `https://www.youtube.com/watch?v=<id>` URL with "wiki" / "summarize" / "watch" / "digest" intent

Not for: short clips (just watch them); music videos; tutorial videos where the visuals are the substance (transcript-only would miss the point); private / paywalled streams beyond what `--cookies-from-browser` can handle.

## Output

A single markdown file at `docs/videos/YYYY-MM-DD-<slug>.md` with **fixed sections**:

1. **Header metadata** — source URL, channel, published date, duration, watched date, sectioning method, speakers.
2. **TL;DR** — 3-5 bullets covering the whole conversation.
3. **Why it matters** — 1-2 sentences.
4. **Section summaries** — one H3 per section, with key points, people, numbers, visual refs, and a clickable timestamp range.
5. **Notable quotes** — verbatim, with speaker attribution and clickable timestamps.
6. **People mentioned** — table.
7. **Papers / works mentioned** — table.
8. **Numbers & claims** — table.
9. **Open questions / gaps**.
10. **Verification log** — coverage counts and sectioning method.

Every timestamp is a clickable link: `[MM:SS](https://youtu.be/<id>?t=<seconds>)`. The wiki is a door back into the video, not a replacement for it.

## Pipeline

```
┌───────────────────────────────┐
│ 1. Fetch transcript + meta    │  yt-dlp → vtt + info.json
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 2. Normalize transcript       │  vtt → timestamped plain lines, dedupe overlap
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 3. Section the video          │  chapters → LLM topics → 10-min fallback
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 4. Chunk by section           │  one chunk per section, carries timestamps
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

## Step 1 — Get a transcript: YouTube subs first, Whisper as fallback

The acquisition path cascades — **YouTube subs first, Whisper only when no subs exist**. Rationale (empirical, from dogfooding on a 4-hour Mandarin AI interview): YouTube's own captions — even when they're auto-generated and mislabeled as "manual" — captured technical English brand/model/people names ("Claude", "Jared Kaplan", "Hinton", "nanoGPT", "SWE-bench", "Scaling Law") that OpenAI's `whisper-1` missed. YouTube has been running multilingual ASR for years on this exact content domain; whisper-1 is from March 2023 and hallucinates English brands inside Chinese audio ("OpenEye" for "OpenAI", "舜" for "顺"). For an AI-interview wiki where those English tokens are the load-bearing content, that loss is fatal. Local `large-v3` is better, but on a 16 GB Apple-Silicon Mac it can't fit alongside Chrome + MCP servers and hangs at 0 % CPU on model load.

So: try YouTube first. Fall back loudly when there are no subs.

### 1a. Preflight: required tools

```bash
command -v yt-dlp >/dev/null || { echo "FAIL: install yt-dlp (brew install yt-dlp)"; exit 1; }
command -v ffmpeg >/dev/null || { echo "FAIL: install ffmpeg (brew install ffmpeg)"; exit 1; }
```

A Whisper backend is **only** required if Step 1b finds no YouTube subs. We don't pre-check it here — the fallback is rare on AI-talk content (most major channels have subs), and forcing setup for an unlikely path adds friction.

### 1b. Try YouTube subs first

```bash
mkdir -p /tmp/yt-wiki
# List available subtitle languages so you know which codes to request.
yt-dlp --list-subs "$URL" 2>&1 | tail -40
# Download. Use the EXACT language codes from --list-subs — NOT bare "en".
# YouTube usually returns "en-US"/"en-GB" or "zh-Hans"/"zh-Hant"; `--sub-lang en` will
# silently miss "en-US". Request a comma-list of likely codes.
yt-dlp --write-sub --write-auto-sub \
       --sub-lang "en-US,en-GB,en,zh-Hans,zh-Hant,zh-CN,zh-TW" \
       --skip-download \
       --write-info-json \
       --output "/tmp/yt-wiki/<slug>.%(ext)s" \
       "$URL"
```

Pick `<slug>` after fetching: lowercase kebab-case of the video title (from `info.json`), ~40 chars, strip emoji and special characters. The `info.json` also carries `chapters`, `title`, `uploader`, `upload_date`, `duration`, `id` — used by Step 3.

Subtitle priority: prefer the original-audio language (`en-US`/`en-GB` for English audio; `zh-Hans` for Mandarin audio — even when YouTube labels the original-language track as `en-US` on a Chinese podcast, which it commonly does). If multiple files appear, take the first non-empty one in priority order.

**On failure to find any subs** (no `.vtt` files emitted), proceed to Step 1c. **On other yt-dlp failure** (region lock, age gate, missing cookies, missing yt-dlp): stop and surface the specific error. No silent fallback to scraping.

### 1c. Fallback: transcribe with Whisper (only when no subs)

If Step 1b produced no `.vtt` file, fetch the audio separately and transcribe.

```bash
yt-dlp --extract-audio --audio-format m4a \
       --output "/tmp/yt-wiki/<slug>.%(ext)s" \
       "$URL"
```

Then pick one backend (preflight that at least one is available):

```bash
command -v mlx_whisper >/dev/null \
  || curl -fs http://localhost:8000/v1/models >/dev/null \
  || [ -n "$OPENAI_API_KEY" ] \
  || { cat <<EOF
FAIL: no YouTube subs available and no transcription backend installed. Install one:
  A) mlx-whisper          pip install mlx-whisper
  B) Rapid-MLX (local)    pip install 'rapid-mlx[audio]' && rapid-mlx serve mlx-community/whisper-large-v3-mlx
  C) OpenAI Whisper API   export OPENAI_API_KEY=... (often in ~/lab/<project>/.env on this machine)
EOF
exit 1; }
```

All three backends emit the same VTT shape, so Step 2 (normalize) is backend-agnostic. **Record which backend was used** in the final header — this affects how much you should trust technical English token reproduction.

#### Backend A — `mlx_whisper` CLI (default, simplest)

File-in / file-out, no server needed.

```bash
mlx_whisper "/tmp/yt-wiki/<slug>.m4a" \
            --model mlx-community/whisper-large-v3-mlx \
            --output-format vtt \
            --output-dir /tmp/yt-wiki \
            --output-name "<slug>" \
            --word-timestamps False \
            --language auto
```

Outputs `/tmp/yt-wiki/<slug>.vtt`.

#### Backend B — Rapid-MLX server ([raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX))

Use when you already have a Rapid-MLX server running (e.g., because you're also using it as the local LLM backend for the per-chunk summarization step in Step 5). One server, OpenAI-compatible API, shared cache — efficient if you're processing many videos.

Setup (one time):

```bash
pip install 'rapid-mlx[audio]'
rapid-mlx serve mlx-community/whisper-large-v3-mlx --port 8000
# Wait for: "Ready: http://localhost:8000/v1"
```

Transcribe via the OpenAI-compatible endpoint:

```bash
curl -fs http://localhost:8000/v1/audio/transcriptions \
     -F file=@/tmp/yt-wiki/<slug>.m4a \
     -F model=default \
     -F response_format=vtt \
     -F language=auto \
     -o /tmp/yt-wiki/<slug>.vtt
```

If transcription fails (HTTP non-2xx, malformed VTT, empty body): **stop and surface** the curl exit code + server response. Do not fall back to Backend A silently — they may produce different transcripts, and a silent backend swap mid-pipeline would make verification unreliable.

#### Backend C — OpenAI Whisper API (cloud, simplest, pay-per-use)

Use when local options are blocked by RAM pressure (a real concern on 16 GB Apple-Silicon machines with the `large-v3` model — ask any M4 mini owner who has tried) or when you simply want a one-line transcribe with no setup. Cost: **$0.006 per minute of audio** via `whisper-1` (a 4-hour interview ≈ $1.44). Privacy note: audio leaves your machine.

API key: this skill expects `OPENAI_API_KEY` in the environment. On a typical Mac dev setup the key is usually not in `~/.zshrc` but in a per-project `.env` (e.g. `~/lab/<project>/.env`). Source it explicitly — the skill does not search:

```bash
[ -z "$OPENAI_API_KEY" ] && { echo "set OPENAI_API_KEY first (often in ~/lab/<project>/.env)"; exit 1; }
```

**File-size limit: 25 MB per request.** Most YouTube audio at 189 kbps reaches that around ~18 min, so a 4-hour interview needs ~13–14 chunks. Split on time, transcribe each, then re-offset and concatenate the VTTs:

```bash
# Step 1: split audio into 15-minute chunks (~12 MB each, safely under 25 MB at 189 kbps).
mkdir -p /tmp/yt-wiki/chunks
ffmpeg -y -i /tmp/yt-wiki/<slug>.m4a \
       -f segment -segment_time 900 -c copy -reset_timestamps 1 \
       /tmp/yt-wiki/chunks/<slug>_%03d.m4a 2>&1 | tail -3

# Step 2: transcribe each chunk, naming output by index so we know its offset.
for f in /tmp/yt-wiki/chunks/<slug>_*.m4a; do
  idx=$(basename "$f" .m4a | sed "s/.*_//")
  echo "transcribing chunk $idx..."
  curl -fsS https://api.openai.com/v1/audio/transcriptions \
       -H "Authorization: Bearer $OPENAI_API_KEY" \
       -F file=@"$f" \
       -F model=whisper-1 \
       -F response_format=vtt \
       -F language=auto \
       -o "/tmp/yt-wiki/chunks/${idx}.vtt" \
    || { echo "FAIL on chunk $idx"; exit 1; }
done

# Step 3: concatenate VTTs with timestamp re-offset (each chunk's timestamps restart at 00:00).
python3 - <<'PY'
import re, pathlib, glob
slug = "<slug>"
out = ["WEBVTT", ""]
for i, path in enumerate(sorted(glob.glob(f"/tmp/yt-wiki/chunks/*.vtt"))):
    offset = i * 900  # 15-min segments
    text = pathlib.Path(path).read_text()
    text = text.replace("WEBVTT\n\n", "").replace("WEBVTT\n", "")
    def shift(m):
        h, mn, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        total = h*3600 + mn*60 + s + offset
        return f"{total//3600:02d}:{(total%3600)//60:02d}:{total%60:02d}.{ms}"
    text = re.sub(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", shift, text)
    out.append(text)
pathlib.Path(f"/tmp/yt-wiki/{slug}.vtt").write_text("\n".join(out))
print(f"wrote /tmp/yt-wiki/{slug}.vtt")
PY
```

The result `/tmp/yt-wiki/<slug>.vtt` is a single VTT covering the whole video with correctly offset timestamps — drop-in compatible with Step 2 (normalize).

If any chunk fails (HTTP non-2xx, malformed VTT, empty body): **stop and surface** the curl exit code + server response. Do not skip a chunk silently — the wiki entry's coverage test depends on a complete transcript.

#### Model choice (applies to local backends A and B)

Backend C is fixed to `whisper-1` (OpenAI's hosted Whisper). The local backends accept any MLX-converted Whisper variant:

| Model | When to use | Speed (M-series Mac) |
|---|---|---|
| `mlx-community/whisper-large-v3-mlx` | Default. Best accuracy on Chinese–English code-switching (typical AI interview). | ~3–5× realtime |
| `mlx-community/whisper-large-v3-turbo` | Long videos (>2 h) where you want quick turnaround and audio is clean English. | ~12–20× realtime |
| `mlx-community/whisper-large-v3-mlx-4bit` | Low-RAM machines (<16 GB). | Same speed, lower memory. |

For a 4-hour AI podcast with Mandarin+English code-switching, `large-v3` is the safe default — turbo loses a small amount of accuracy on rare technical English-in-Chinese tokens (paper names, model names) which is exactly the load-bearing content for an AI-interview wiki.

#### Language hint

Pass `--language zh` (A), `-F language=zh` (B), or `-F language=zh` (C) for known-Mandarin audio, or `en` for known-English audio, when `auto` mis-detects (rare, but possible on short multilingual intros — Whisper's auto-detect uses only the first 30 seconds).

#### Common failure modes

- **OOM mid-transcription on Backend A/B** → real on 16 GB Apple-Silicon machines with `large-v3`; the symptom is a Python process stuck at 0 % CPU for 5+ minutes. Switch to `whisper-large-v3-mlx-4bit`, or to Backend C, or chunk the audio: `ffmpeg -i input.m4a -f segment -segment_time 600 -c copy chunk_%03d.m4a`.
- **Audio format unsupported** → all backends use ffmpeg internally; confirm `ffmpeg -i <file>` reads it.
- **HuggingFace download blocked (A/B)** → first run downloads ~3 GB of model weights. Ensure network access, or pre-download with `huggingface-cli download mlx-community/whisper-large-v3-mlx`.
- **OpenAI 401 / quota error (C)** → check `OPENAI_API_KEY` is set and the org has credits.
- **OpenAI 413 file-too-large (C)** → a chunk exceeded 25 MB. Re-split with smaller `--segment_time`.

No silent fallback to a smaller model — accuracy regressions would silently produce a worse wiki entry. The cascade from YouTube subs → Whisper IS a fallback, but it is explicit: it only runs when Step 1b emitted no `.vtt` file, and the chosen source is recorded in the final header so a reader knows which transcript shape was used.

## Step 2 — Normalize transcript

YouTube `.vtt` cues repeat as the karaoke highlight moves. Each spoken line shows up 2-3 times. Strip the duplicates and emit one line per distinct utterance with its start timestamp:

```
[00:00:12] welcome back to the show today I'm here with
[00:00:18] our guest who's been working on
[00:00:22] foundation model evaluation at...
```

Quick dedupe approach (Python or awk):

```bash
python3 - <<'PY'
import re, sys, pathlib
src = pathlib.Path("/tmp/yt-wiki/<slug>.en.vtt").read_text()
out, last = [], None
for block in src.split("\n\n"):
    m = re.search(r"^(\d{2}:\d{2}:\d{2})\.\d{3} -->", block, re.M)
    if not m: continue
    text = "\n".join(l for l in block.splitlines() if "-->" not in l and "WEBVTT" not in l and l.strip())
    text = re.sub(r"<[^>]+>", "", text).strip()
    if text and text != last:
        out.append(f"[{m.group(1)}] {text}")
        last = text
pathlib.Path("/tmp/yt-wiki/<slug>.transcript.txt").write_text("\n".join(out))
PY
```

Save to `/tmp/yt-wiki/<slug>.transcript.txt`. **Keep this file until Step 7 passes** — it is the ground truth for verification.

## Step 3 — Section the video

Three-tier strategy. Never silently degrade between tiers — record which tier was used and surface it in the final header.

### 3a. YouTube chapters (preferred)

```bash
jq '.chapters' /tmp/yt-wiki/<slug>.info.json
```

If non-null, each entry has `start_time` (seconds), `end_time`, `title`. Use these directly as section boundaries.

### 3b. LLM topic detection (when no chapters)

Single subagent: read the full normalized transcript, return JSON of section boundaries.

Prompt:

```
Read /tmp/yt-wiki/<slug>.transcript.txt. Identify conversational topic
boundaries. Return JSON: [{"start": "HH:MM:SS", "title": "..."}].
Aim for 5-12 sections for a 60-90 minute video. Titles should be 3-7
words describing what the conversation is *about* in that span, not
who's speaking. Do NOT invent topics — only mark real shifts in subject.
```

Save the result to `/tmp/yt-wiki/<slug>.sections.json`.

### 3c. Time-based fallback (last resort)

Fixed 10-minute windows. Labels: `[00:00-10:00]`, `[10:00-20:00]`, etc. Use only when 3a and 3b both fail.

## Step 4 — Chunk by section

One chunk per section. Each chunk is the verbatim transcript lines whose timestamps fall within the section's `[start, end)` range, prefixed with a breadcrumb header:

```
<!-- chunk {i}/{n} — {section_title} -->
<!-- range: 00:12:30 → 00:24:15 -->
<!-- video: {title} ({url}) -->
<!-- breadcrumb: prior section = "{prev_title}" -->

[12:34] ...transcript lines verbatim...
[12:38] ...
```

Save as `/tmp/yt-wiki/<slug>.chunk.<i>.md`.

## Step 5 — Summarize per-chunk

Dispatch one subagent per chunk **in parallel** (single message, multiple Agent calls).

### Subagent prompt template

```
Read /tmp/yt-wiki/<slug>.chunk.<i>.md. The video is "<title>" — see the
chunk header for the section title, range, and prior-section breadcrumb.

Return a structured note (YAML) with:

tldr:        1-2 sentences — what is this section about?
key_points:  3-7 bullets — distinct claims, findings, or anecdotes
people_mentioned:
  - name: ...
    context: brief
    timestamp: HH:MM:SS
papers_or_works_mentioned:
  - title: ...
    url: (if findable, else omit)
    timestamp: HH:MM:SS
quotes:
  - speaker: "[Host]" | "[Guest]" | name | "[Speaker A]" | "[Speaker B]"
    text: "verbatim — filler may be elided with [...]"
    timestamp: HH:MM:SS
numbers:
  - claim: "verbatim including the number"
    timestamp: HH:MM:SS
visual_refs:
  - timestamp: HH:MM:SS
    cue: "as you can see" | "this diagram" | "on screen" | ...
open_questions:
  - claim asserted without justification

Rules:
- Quotes MUST be verbatim. Only filler words (uh, you know, I mean) may
  be elided with [...]. NO paraphrase.
- Speaker labels: use [Host]/[Guest] when conversational role is obvious;
  use names if introduced on-mic; [Speaker A]/[Speaker B] when uncertain.
  Never invent names.
- visual_refs: timestamp + verbal cue only. Do NOT describe what's on screen.
- If you don't find an item for a field, omit the field — don't fabricate.

Report under 600 words.
```

Save each subagent's output to `/tmp/yt-wiki/<slug>.notes.<i>.md`.

## Step 6 — Synthesize the wiki entry

Merge chunk notes into `docs/videos/YYYY-MM-DD-<slug>.md`:

```markdown
# <Video Title>

**Source**: https://youtu.be/<id>
**Channel**: <uploader>
**Published**: YYYY-MM-DD
**Duration**: HH:MM:SS
**Watched on**: YYYY-MM-DD
**Sectioning**: chapters | llm-detected | time-based
**Transcript source**: youtube-subs (manual `<lang>`) | youtube-subs (auto `<lang>`) | mlx-whisper (large-v3) | rapid-mlx-server (large-v3) | openai-api (whisper-1)
**Speakers**: <Host>, <Guest>  (or "Speaker A, Speaker B")

## TL;DR

- <bullet 1>
- <bullet 2>
- <bullet 3>

## Why it matters

<1-2 sentences — what claim, idea, or framing is worth coming back for?>

## Section summaries

### §1 — <chapter title>  [00:00 → 12:34](https://youtu.be/<id>?t=0)
- <key point>
- <key point>
- People: [Name](url?t=N), [Name](url?t=N)
- Numbers: "<verbatim>" [[MM:SS](url?t=N)]
- Visual refs: [[12:01](url?t=721)], [[12:18](url?t=738)]

### §2 — <chapter title>  [12:34 → 24:15](https://youtu.be/<id>?t=754)
...

## Notable quotes

> "<verbatim quote>"
> — [Guest], §3 [[18:42](https://youtu.be/<id>?t=1122)]

> "<verbatim quote>"
> — [Host], §5 [[34:11](https://youtu.be/<id>?t=2051)]

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| ... | ... | [MM:SS](url?t=N) |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| ... | ... | [MM:SS](url?t=N) |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| ... | [Guest] | [MM:SS](url?t=N) |

## Open questions / gaps

- <claim asserted without evidence>

## Verification log

- sections covered: N/N
- quotes traced verbatim: K/K
- numbers traced: M/M
- sectioning method used: chapters | llm-detected | time-based
- transcript source: youtube-subs (manual/auto, lang) | mlx-whisper | rapid-mlx-server | openai-api (whisper-1)
```

Timestamp link math: `t=<seconds>` where `<seconds> = HH*3600 + MM*60 + SS`. Compute once and reuse.

Synthesis rules:
- Preserve **every** verbatim quote from the per-chunk `quotes` field unless 7b deletes it.
- Preserve **every** number from `numbers` fields.
- "Notable quotes" is a curated subset (3-8 quotes total) of the strongest pulls across sections.
- "People mentioned" deduplicates across sections; "First mentioned" is the earliest timestamp.
- TL;DR bullets must each be traceable to at least one section's `key_points` (no inventing summary-level claims).

## Step 7 — Coverage test (mandatory)

### 7a. Section coverage

Every section emerging from Step 3 appears under "Section summaries". Compute:

```bash
# section titles from Step 3
jq -r '.[].title' /tmp/yt-wiki/<slug>.sections.json | sort > /tmp/yt-wiki/<slug>.src-sections
# section titles in the emitted file
grep -E '^### §[0-9]+ —' docs/videos/YYYY-MM-DD-<slug>.md | sed 's/.*— //; s/  \[.*//' | sort > /tmp/yt-wiki/<slug>.out-sections
diff /tmp/yt-wiki/<slug>.src-sections /tmp/yt-wiki/<slug>.out-sections
```

Non-empty diff → add the missing sections.

### 7b. Quote verbatim check

For every quote in "Notable quotes", check a distinctive 5–15-character substring against the transcript. **Important:** YouTube splits lines at karaoke-highlight boundaries mid-phrase, so a multi-line quote will not match a single line. Flatten the transcript first by joining utterances with a single space, then `in`-test or `grep -F`:

```bash
python3 - <<'PY'
import re, pathlib
raw = pathlib.Path("/tmp/yt-wiki/<slug>.transcript.txt").read_text()
flat = " ".join(re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", l) for l in raw.splitlines())
flat = re.sub(r"\s+", " ", flat)
pathlib.Path("/tmp/yt-wiki/<slug>.flat.txt").write_text(flat)
PY
grep -F "this is the distinctive phrase from quote" /tmp/yt-wiki/<slug>.flat.txt
```

No hit → paraphrase crept in (delete the quote) or filler-elision broke the anchor (rewrite the quote with a tighter unbroken span).

### 7c. Number traceability

Every row in "Numbers & claims" must have a transcript hit (the number itself or a distinctive surrounding phrase). No hit → delete the row or rewrite to quote source directly.

### 7d. Hallucination sweep

Re-read the final entry end-to-end. For each TL;DR bullet, point at the section + timestamp that supports it. No support → remove.

**Failures stop and surface** — never accept "looks plausible enough." Log results:

```markdown
## Verification log
- sections covered: 8/8 ✅
- quotes traced verbatim: 6/6 ✅
- numbers traced: 4/4 ✅
- removed during verification: <list, if any>
```

## Anti-Patterns

- **Paraphrasing quotes.** Verbatim only. Only filler words (`uh`, `you know`, `I mean`) may be elided with `[…]`. A pithy paraphrase that captures "the gist" is misattribution.
- **Hallucinating what's on screen** from a `visual_refs` timestamp. Flag the timestamp; do NOT describe the diagram.
- **Confident speaker attribution in panels.** When 3+ voices participate, default to `Speaker A/B/C` unless names are clearly introduced on-mic.
- **Time-based sectioning when chapters existed.** Always parse `info.json` first.
- **Skipping the coverage test.** A plausible-sounding wiki entry that omits a section or invents a quote is worse than a short, verified one.
- **Inventing "papers mentioned"** when the speaker made only a vague allusion ("there's that scaling paper"). If the title isn't named on-mic and not in the description, leave it out.
- **Silent fallback to browser scraping** when `yt-dlp` fails. Hard stop, surface the error.
- **Choosing Whisper when YouTube subs are available.** Empirically (dogfooded on a 4-hour Mandarin AI interview): YouTube's track captured technical English brand/model/people names that whisper-1 missed entirely. Whisper is the fallback, not the preferred path. Local `large-v3` may be better than whisper-1, but only if it fits in your RAM — on 16 GB Apple-Silicon it often doesn't.
- **One huge subagent for a 90-min interview.** Defeats the chunking purpose. One subagent per section, dispatched in parallel.
- **Truncating a long transcript** to fit a single context. Section + chunk + parallel subagents.
- **Embedding non-clickable timestamps** (`[12:34]` without a link). The whole point is the door back to the video.
- **Losing the raw transcript** before verification. Keep `/tmp/yt-wiki/<slug>.transcript.txt` until Step 7 passes.

## Example Invocation

```
Make a wiki entry for https://youtu.be/ttkd0t5qTD4
```

Expected outcome: `docs/videos/YYYY-MM-DD-<title-slug>.md` with all sections populated, clickable timestamps throughout, quotes verbatim from the transcript, and a clean verification log.

## See Also

- `blog-reader` — Direct cousin; same pipeline shape (fetch → chunk → parallel-summarize → coverage test) but for written blog posts.
- `pdf-reader` — For papers a guest mentions on-mic.
- `arxiv-latex-reader` — For deep-reading an arXiv paper referenced in the video.
- `github-reader` — For a repo a guest mentions.
- `academic-deep-research` — For evaluating papers / topics the conversation surfaces.
- `agents-md-writing` — For promoting recurring findings from many videos into project memory.

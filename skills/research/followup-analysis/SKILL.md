---
name: followup-analysis
description: "Use when analyzing the follow-up works of a specific paper — the papers that cite it. Resolves the seed paper via Semantic Scholar, paginates the full citing-paper list, triages by impact + recency, dispatches parallel subagents to tag each follow-up (extension / improvement / application / theoretical / criticism), and produces a clustered markdown report under docs/research/. Triggers: \"followup analysis\", \"follow-up works\", \"papers that cite\", \"forward citations\", \"who cites this paper\", \"citation graph\", \"impact analysis\", \"downstream papers\""
---

# Followup Analysis

## When to Use

- You read a paper and want to know what the field has built on top of it
- You need a faithful map of the **citation cone** of a seed paper (who picked it up, in what direction, with what impact)
- You're deciding whether a method is still alive — has anyone followed up, or did it die at publication?
- You want to surface the most-cited follow-ups so you can read those instead of the seed paper's bibliography in raw order
- You want to cluster follow-ups into themes (extends / improves / applies / criticizes) instead of reading 100 abstracts manually

Do **not** use this skill to:

- Evaluate the seed paper itself — use `academic-deep-research`
- Read a single paper deeply — use `arxiv-latex-reader` or `pdf-reader`
- Find papers a given paper cites (backward citations) — that's `/references`, not `/citations`; this skill is forward-only

## Pipeline

```
input (title | arxiv id | arxiv URL | DOI | S2 paperId)
   │
   ▼
[1] Resolve seed paper           → paperId + slug
   │
   ▼
[2] Fetch citing papers          → /tmp/followup/<slug>/citations.json
       (paginated, all pages)
   │
   ▼
[3] Triage                       → top-N by citation count, plus most-recent
       (cap at ~50 total)
   │
   ▼
[4] Parallel subagent batching   → /tmp/followup/<slug>/batch.<i>.md
       (~8 papers per batch, one Agent call per batch, single message)
   │
   ▼
[5] Synthesize                   → docs/research/YYYY-MM-DD-<slug>-followups.md
       (cluster by theme, rank by sub-citation count)
   │
   ▼
[6] Verify                       → no fabricated titles, counts match
```

## Step 1 — Resolve the seed paper

Accept any of: a free-text title, an arxiv id (`2602.04770`), an arxiv URL, a DOI, or a Semantic Scholar paperId.

**Always use `curl --noproxy '*' -A "Mozilla/5.0"`** for Semantic Scholar — `WebFetch` is rate-limit-blocked on the unauthenticated public endpoint, and this machine routes via a proxy that the API rejects.

**Rate-limit behavior (measured):** the unauthenticated public endpoint returns `{"code":"429","message":"Too Many Requests"}` after only 1-2 quick requests, and a 30s sleep is **not** long enough to clear it. Wait **≥90s** before retrying after a 429, and put ≥4s between every successful request. If you have a `SEMANTIC_SCHOLAR_API_KEY` env var, pass it as `-H "x-api-key: $SEMANTIC_SCHOLAR_API_KEY"` and the limit jumps to ~100 RPS.

### By arxiv id
```bash
curl -s --noproxy '*' -A "Mozilla/5.0" \
  "https://api.semanticscholar.org/graph/v1/paper/arXiv:2602.04770?fields=title,year,venue,citationCount,authors,externalIds"
```

### By title
```bash
curl -s --noproxy '*' -A "Mozilla/5.0" \
  "https://api.semanticscholar.org/graph/v1/paper/search?query=Generative+Modeling+via+Drifting&limit=5&fields=title,year,venue,citationCount,externalIds,authors" \
  | python3 -m json.tool
```

Take the top match where the title fuzzy-matches. **Confirm with the user before proceeding if `total` > 1 and the top match looks ambiguous** — picking the wrong seed paper poisons every downstream step.

### Slug
`<first-author-lastname>-<first-content-word>-<arxiv-year>`, e.g. `deng-drifting-2026`.

### Cache the seed metadata to disk
```bash
mkdir -p /tmp/followup/<slug>
curl -s --noproxy '*' -A "Mozilla/5.0" \
  "https://api.semanticscholar.org/graph/v1/paper/<paperId>?fields=title,abstract,year,venue,citationCount,authors,externalIds" \
  > /tmp/followup/<slug>/seed.json
```

## Step 2 — Fetch all citing papers (paginated)

The endpoint returns 100 per page (max `limit=1000`, but in practice 100 is a safer default for unauthenticated requests).

```bash
mkdir -p /tmp/followup/<slug>
PAPER_ID="<paperId>"
OFFSET=0
PAGE=0
while true; do
  curl -s --noproxy '*' -A "Mozilla/5.0" \
    "https://api.semanticscholar.org/graph/v1/paper/${PAPER_ID}/citations?fields=title,year,venue,citationCount,authors,abstract,externalIds,openAccessPdf&limit=100&offset=${OFFSET}" \
    > "/tmp/followup/<slug>/citations.page.${PAGE}.json"
  NEXT=$(python3 -c "import json; d=json.load(open('/tmp/followup/<slug>/citations.page.${PAGE}.json')); print(d.get('next',''))")
  [ -z "$NEXT" ] && break
  OFFSET=$NEXT
  PAGE=$((PAGE+1))
  sleep 4   # respect unauthenticated rate limit
done
```

Then flatten into a single normalized JSON list:

```bash
python3 - <<'PY'
import json, glob, pathlib
slug = "<slug>"
out = []
for p in sorted(glob.glob(f"/tmp/followup/{slug}/citations.page.*.json")):
    for row in json.load(open(p)).get("data", []):
        out.append(row["citingPaper"])
pathlib.Path(f"/tmp/followup/{slug}/citations.json").write_text(json.dumps(out, indent=2))
print(f"flattened {len(out)} citing papers")
PY
```

### Response shape (so you can parse it without re-checking)
```json
{
  "offset": 0,
  "next": 100,
  "data": [
    {"citingPaper": {
      "paperId": "f4cf47e...",
      "externalIds": {"ArXiv": "2605.07907", "DOI": null, "CorpusId": 288148003},
      "title": "Consistency Regularised Gradient Flows for Inverse Problems",
      "venue": "",
      "year": 2026,
      "citationCount": 0,
      "openAccessPdf": {"url": "..."},
      "authors": [{"authorId": "...", "name": "Alessio Spagnoletti"}, ...],
      "abstract": "..."
    }}
  ]
}
```

If `citationCount` on the seed is **0**, abort with "no follow-ups yet" — don't run an empty pipeline. If it's **>500**, ask the user whether to cap at top-N or scan everything (top-N is almost always right).

## Step 3 — Triage

Sort the flattened list by:

1. **Top-cited follow-ups** — `citationCount` descending. Take the top 30.
2. **Most-recent follow-ups** — `year` descending, then `paperId` for stable tie-break. Take the top 20, **excluding** ones already in the top-cited bucket.

Result: ≤50 selected papers. Save the selection list:

```bash
python3 - <<'PY'
import json, pathlib
slug = "<slug>"
cits = json.load(open(f"/tmp/followup/{slug}/citations.json"))
by_cite = sorted([c for c in cits if c.get("abstract")],
                 key=lambda c: (-(c.get("citationCount") or 0), -(c.get("year") or 0)))
top_cited = by_cite[:30]
seen = {c["paperId"] for c in top_cited}
recent = sorted([c for c in cits if c["paperId"] not in seen and c.get("abstract")],
                key=lambda c: -(c.get("year") or 0))[:20]
selected = top_cited + recent
pathlib.Path(f"/tmp/followup/{slug}/selected.json").write_text(json.dumps(selected, indent=2))
print(f"selected {len(selected)} ({len(top_cited)} by impact + {len(recent)} by recency)")
PY
```

**Drop papers with no abstract** before triage — without an abstract the subagent can't tag the follow-up reliably, and a fabricated tag is worse than a missing row.

## Step 4 — Parallel subagent tagging

Split `selected.json` into batches of **8 papers each**. Dispatch one Agent subagent per batch **in a single message** (one Agent tool call per batch, all in one assistant turn). This is the parallel-Agent superpower of Claude Code — fan out 5-7 subagents at once instead of sequential.

### What each subagent does

Give it a self-contained prompt:

> Read these N follow-up papers (title + abstract only, no fetching).
> For each paper, return a JSON line with:
> - `paperId`, `title`, `year`, `citationCount`, `arxiv` (if present)
> - `tag` — exactly one of: `Extends`, `Improves`, `Applies`, `Theoretical`, `Criticizes`, `Benchmarks`, `Mentions`
> - `digest` — 1-2 sentence plain-English summary of what *this paper* did with the seed work
> - `relation_quote` — verbatim phrase from the abstract that names the seed work or its contribution, or "—" if the abstract doesn't mention it explicitly
>
> Write to `/tmp/followup/<slug>/batch.<i>.md`. Return only "OK batch i" — do NOT inline the contents.

Tag definitions to put in the subagent prompt verbatim:

| Tag | Meaning |
|-----|---------|
| `Extends` | New variant, new modality, or new objective built on the seed method |
| `Improves` | Claims to outperform the seed method on the seed's own benchmarks |
| `Applies` | Uses the seed method as a tool in a new domain (no methodological change) |
| `Theoretical` | Analyzes why the seed method works, proves a property, or connects it to known theory |
| `Criticizes` | Reports failures, limitations, or counter-examples |
| `Benchmarks` | Cites it only as a baseline in an unrelated comparison |
| `Mentions` | Passing reference; the seed work is not load-bearing in this paper |

### Why disk-based handoff

Subagent responses come back to you, not the user. If 7 subagents return inline tag lists, your context bloats and the user sees nothing. **Have each subagent write to disk and return only `OK batch i`** (this pattern is also documented in the project CLAUDE.md).

## Step 5 — Synthesize report

Read every `batch.<i>.md`, merge into one list, then cluster by tag and rank within each cluster by `citationCount` desc.

Write the report to `docs/research/YYYY-MM-DD-<slug>-followups.md`. Template:

```markdown
# Followup Analysis: <Seed Paper Title>

**Date**: YYYY-MM-DD
**Seed paper**: <title> ([arxiv](https://arxiv.org/abs/<id>))
**Authors**: <first 3 authors + et al if >3>
**Venue / Year**: <venue>, <year>
**Total citations (Semantic Scholar)**: N
**Follow-ups analyzed**: M of N (top-30 by impact + 20 most-recent, abstracts only)

## TL;DR

- <one bullet per major theme that emerged>
- <which sub-thread has the most-cited follow-ups>
- <whether the method is being extended, criticized, or just used as a baseline>

## Cluster summary

| Tag | Count | Top 3 follow-ups (by their own citation count) |
|-----|-------|------------------------------------------------|
| Extends      | k | <short titles> |
| Improves     | k | ... |
| Applies      | k | ... |
| Theoretical  | k | ... |
| Criticizes   | k | ... |
| Benchmarks   | k | ... |
| Mentions     | k | ... |

## Extensions

### <Follow-up Title> — `cit:N` `yr:YYYY` `<arxiv>`
**Authors**: ...
**Digest**: <1-2 sentences from subagent>
> "<relation_quote verbatim from abstract>"

(repeat per paper, ranked by citationCount desc within the section)

## Improvements

...

## Theoretical analyses

...

## Criticisms / Limitations

...

## Applications

...

## Benchmarks (cited as baseline only)

<one-line bullets — these don't need full sections>

- **<title>** — `cit:N` `<arxiv>` — short note

## Mentions (passing citations)

<bullets, terse>

## What's emerging

2-4 sentences synthesizing the trajectory: which sub-direction is hottest, who's driving it, what's the next obvious open problem. This is **synthesis**, not summary — say something the table doesn't already say.

## Verification log

- Selected: M papers (30 top-cited + 20 most-recent, after dropping K with no abstract)
- Tagged: M (no fabrications — every paperId in this report appears in `/tmp/followup/<slug>/selected.json`)
- Unprocessed: N − M follow-ups not analyzed (full list in `/tmp/followup/<slug>/citations.json`)
- Seed `paperId`: <id>
```

## Step 6 — Verify

Before declaring done:

```bash
slug=<slug>
# Every paperId in the report must exist in selected.json
python3 - <<PY
import json, re, sys
sel = {p["paperId"] for p in json.load(open(f"/tmp/followup/{slug}/selected.json"))}
report = open("docs/research/<DATE>-${slug}-followups.md").read()
# extract any 40-char hex paperIds, and any arxiv ids
arxiv_in_report = set(re.findall(r"\b(\d{4}\.\d{4,5})\b", report))
arxiv_in_sel = {p["externalIds"].get("ArXiv") for p in json.load(open(f"/tmp/followup/{slug}/selected.json")) if p.get("externalIds",{}).get("ArXiv")}
missing = arxiv_in_report - arxiv_in_sel
print("arxiv ids in report not in selected:", missing or "OK")
PY
```

If `missing` is non-empty, the report has a fabricated or mis-copied paper id — fix before saving.

Also confirm:

- The cluster-summary counts sum to M (selected papers analyzed)
- Every `Extensions / Improvements / Theoretical / Criticisms / Applications` heading has at least the top 3 entries unless the cluster genuinely has fewer
- The "What's emerging" section says something not in the table

## Quick Reference

| Step | Command | Output |
|------|---------|--------|
| Resolve | `curl /paper/search?query=...` | `paperId` |
| Fetch citations | `curl /paper/{id}/citations` (paginated) | `/tmp/followup/<slug>/citations.json` |
| Triage | python script above | `selected.json` (≤50 papers) |
| Tag | 5-7 parallel Agents, 8 papers each, single message | `batch.<i>.md` |
| Synthesize | manual, follow template | `docs/research/<date>-<slug>-followups.md` |
| Verify | python id-cross-check | clean diff |

## Anti-Patterns

- **Skipping rate-limit waits.** Semantic Scholar's unauthenticated endpoint returns 429 quickly under bursts. Sleep ≥4s between page requests. Do not switch to a "smarter" library that re-uses connections without throttling.
- **Letting subagents return tag lists inline.** Their output goes into *your* context, not the user's. Write to `/tmp/followup/<slug>/batch.<i>.md` and return `OK batch i`. (Same pattern as `youtube-wiki` and `blog-reader`.)
- **Asking for 1000 follow-ups when the seed has 1000 citations.** Triage to 50. The report becomes unreadable at >80, and the long tail is almost always `Mentions` or `Benchmarks`.
- **Tagging without an abstract.** Drop papers with no abstract before sending to subagents. A guessed tag is worse than an honest omission.
- **Re-implementing forward citations from arxiv.** arxiv has no forward-citation API. Use Semantic Scholar (or OpenAlex as a fallback if Semantic Scholar is down).
- **Conflating `/citations` (forward) with `/references` (backward).** This skill is forward-only. Going backward — what the seed paper itself cites — is a different operation that `academic-deep-research` already covers.
- **Quietly falling back to "couldn't find abstract, made one up from the title."** Per the global No-Silent-Fallbacks rule: drop the paper, log it in the verification log, do not fabricate.
- **Writing the report from memory after reading batch files.** Read the batch files at synthesis time. Don't trust your context — a tag list across 50 papers is exactly the kind of thing that drifts.

## See Also

- `academic-deep-research` — Score the seed paper itself across 8 dimensions; complements this skill (one looks at the paper, the other at its descendants)
- `idea-explore` — Consumes this skill's report as the "already-done" exclusion list when proposing new directions on the seed paper
- `arxiv-latex-reader` — Progressive section-by-section reading once you've picked a follow-up to read in depth
- `pdf-reader` — When a follow-up is only released as PDF, not arxiv tex
- `github-reader` — Drill into the implementation of a high-impact follow-up
- `blog-reader` — Sometimes the most useful follow-up is a blog post, not a paper

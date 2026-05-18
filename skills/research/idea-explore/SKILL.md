---
name: idea-explore
description: "Use when proposing new research ideas grounded in a seed paper or method — surfaces the gaps that the citation cone hasn't filled, the drawbacks the community already complains about in the official repo's GitHub issues, and the adjacent angles the literature suggests. Orchestrates followup-analysis (already-done exclusion), GitHub-issue mining (known drawbacks), and academic-deep-research (adjacent literature) into a ranked candidate-ideas report. Triggers: \"idea explore\", \"idea-explore\", \"propose ideas\", \"explore ideas\", \"new directions\", \"research gaps\", \"what's missing\", \"build on this paper\", \"探索想法\", \"提出新想法\", \"研究空白\""
---

# Idea Explore

## When to Use

- You have a **seed paper or method** and want grounded suggestions for *what to do next* — not a recap of what's been done.
- You want proposed ideas anchored to **evidence**, not novelty mining: each idea must point to a gap in the citation cone, a drawback in the issues tracker, or an adjacent-literature angle the seed hasn't crossed yet.
- You want to **avoid re-inventing** a paper already in the seed's followup cone — the exclusion list is mandatory output.
- You want the community's **known complaints** (GitHub issues tagged as bugs, limitations, "doesn't work for X") surfaced before you commit to a direction.
- You're populating an `idea-box` and need a producer that hands you 5-10 candidate `idea.md` seeds with motivation in one pass.

Do **not** use this skill to:

- Decide whether a *single, already-formed* idea is buildable — that's `idea-feasibility`.
- Survey the follow-ups of a paper without proposing new directions — that's `followup-analysis` directly.
- Read or summarize the seed paper itself — use `arxiv-latex-reader` or `pdf-reader` first, then come back here.
- Evaluate the seed paper's impact / venue / reproducibility — that's `academic-deep-research`.
- Brainstorm ideas with **no** seed paper or method anchor — use `superpowers:brainstorming` or `office-hours` instead. This skill needs evidence to anchor proposals.

## Core principles

- **Anchored, not invented.** Every proposed idea must cite at least one of: a missing branch in the followup cone, a GitHub issue, or an adjacent-literature finding. No anchor → drop the idea.
- **Exclusion before proposal.** Run `followup-analysis` first. Anything already covered by an existing follow-up (especially `Extends`/`Improves` tagged ones) is **forbidden** as a proposed idea. The report must explicitly list these exclusions.
- **Drawbacks come from the community, not your imagination.** Pull them from real GitHub issues with verifiable URLs. Do not paraphrase a hypothetical drawback as if the community had reported it.
- **No silent fallbacks.** If the official GitHub repo is unknown, say so and skip the issue scan with an explicit log entry — do not substitute a Google search for "drawbacks of X". If `followup-analysis` finds 0 citing papers, abort: the seed is too new or unknown to explore from.
- **Subagents write to disk.** Idea synthesis fans out in parallel; each subagent writes structured notes to `/tmp/idea-explore/<slug>/` and returns only `OK §i`. The main agent reads from disk during synthesis.

## Pipeline

```
input (seed paper | method name | arxiv id | arxiv URL | DOI)
   │
   ▼
[1] Resolve seed                    → paperId + slug + official-repo URL
   │
   ▼
[2] Already-done map                → followup-analysis report
       (invoke the skill)              (cite this as the exclusion list)
   │
   ▼
[3] Known drawbacks                 → /tmp/idea-explore/<slug>/issues.json
       (gh issue list on official       (filtered to bug-flavored issues
        repo, scored by signal)          with body + url + comment count)
   │
   ▼
[4] Adjacent angles                 → /tmp/idea-explore/<slug>/adjacent.md
       (academic-deep-research on       (2-3 queries crossing the seed
        adjacent topics)                 with neighboring sub-fields)
   │
   ▼
[5] Parallel idea synthesis         → /tmp/idea-explore/<slug>/ideas.<i>.md
       (3-4 subagents, single             (each emits 3-5 candidate ideas
        message)                            with anchor + risk + delta vs.
                                           seed)
   │
   ▼
[6] Synthesize report               → ./idea_box/<slug>/explore.md            (if inside idea-box)
                                       docs/research/<DATE>-<slug>-explore.md (otherwise)
   │
   ▼
[7] Verify                          → no exclusion-list collisions,
                                       every drawback URL resolves,
                                       every idea has ≥1 anchor
```

## Step 1 — Resolve the seed

Accept any of: free-text title, arxiv id (`2602.04770`), arxiv URL, DOI, Semantic Scholar paperId, or a method name with an obvious paper (e.g. "flow matching", "DPO").

Resolve the same way `followup-analysis` does — via Semantic Scholar with `curl --noproxy '*' -A "Mozilla/5.0"`, respecting the ≥4s inter-request and ≥90s post-429 waits. If the input is a method name with no paper attached, search Semantic Scholar for the canonical paper and **confirm with the user before proceeding** if multiple plausible matches exist.

Slug: `<first-author-lastname>-<first-content-word>-<year>`, e.g. `lipman-flow-2023`.

Then find the **official GitHub repo**:

1. Check the paper's project page / arxiv abs page links — most papers link a repo near the top.
2. Search the paper's first author's GitHub for a matching repo name.
3. As a fallback, search GitHub for the paper's exact title.

If you find a repo, cache its `owner/name` in `/tmp/idea-explore/<slug>/repo.txt`. If you do **not** find an official repo with confidence, write `NONE` to that file — Step 3 will degrade gracefully (no fabricated drawbacks).

**Detect idea-box context:** if `./idea_box/<slug-prefix>*/` exists for the current seed (matching `*<first-content-word>*`), record the destination as `./idea_box/<slug>/explore.md`. Otherwise, destination is `docs/research/<YYYY-MM-DD>-<slug>-explore.md`. Do **not** auto-create an `idea_box/` directory — only write into one that already exists.

```bash
mkdir -p /tmp/idea-explore/<slug>
echo "<owner/name>" > /tmp/idea-explore/<slug>/repo.txt   # or "NONE"
```

## Step 2 — Already-done map (exclusion list)

Invoke the `followup-analysis` skill on the seed paper. Per the project CLAUDE.md, this skill dispatches parallel subagents — let it run to completion, but **redirect its output** to the idea-explore workspace:

- The skill writes to `docs/research/<date>-<slug>-followups.md` by default. If inside an idea-box dir, redirect to `./idea_box/<slug>/followups.md` (per the idea-box cross-skill contract).
- Either way, **also** keep a working copy at `/tmp/idea-explore/<slug>/followups.md` for Step 5 subagents to read.

After the followup report exists, extract the exclusion list:

```bash
slug=<slug>
python3 - <<PY
import json, pathlib, re
report = open("/tmp/idea-explore/$slug/followups.md").read()
# extract titles + tags from the per-cluster sections
done = []
for tag in ["Extensions", "Improvements", "Theoretical analyses", "Applications"]:
    block = re.search(rf"## {re.escape(tag)}\n(.+?)(?=\n## |\Z)", report, re.S)
    if not block: continue
    for m in re.finditer(r"### (.+?) — `cit:(\d+)`", block.group(1)):
        done.append({"tag": tag, "title": m.group(1), "cit": int(m.group(2))})
pathlib.Path("/tmp/idea-explore/$slug/exclusions.json").write_text(json.dumps(done, indent=2))
print(f"exclusion list: {len(done)} follow-ups already cover {{Extensions/Improvements/Theoretical/Applications}}")
PY
```

`Benchmarks` and `Mentions` tags do **not** become exclusions — those papers used the seed as a baseline, they didn't extend or improve it. Only the four directional tags above are exclusion-worthy.

If `citationCount` on the seed is **0** or the followup report has no entries in the four exclusion-worthy tags, the seed is too cold to "explore from" responsibly — surface this to the user and ask whether to proceed with only Steps 3–4 anchoring, or abort.

## Step 3 — Known drawbacks (GitHub issues)

Read `/tmp/idea-explore/<slug>/repo.txt`. If it is `NONE`, write `/tmp/idea-explore/<slug>/issues.json` as `[]` and log it in the report's verification section — do not substitute a Google search.

Otherwise use `gh` to enumerate issues (this skill assumes `gh` is authenticated; the `github-cli` skill covers setup):

```bash
slug=<slug>
repo=$(cat /tmp/idea-explore/$slug/repo.txt)
gh issue list --repo "$repo" --state all --limit 200 \
  --json number,title,labels,state,body,createdAt,closedAt,comments,url,author \
  > /tmp/idea-explore/$slug/issues.raw.json
```

Then filter for **bug-flavored** issues — issues that signal real method limitations, not setup confusion:

```bash
slug=<slug>
python3 - <<'PY'
import json, re, pathlib
raw = json.load(open(f"/tmp/idea-explore/$slug/issues.raw.json"))
BUG_LABELS = {"bug", "limitation", "discussion", "help wanted", "question",
              "enhancement", "performance", "regression"}
BUG_KEYWORDS = re.compile(
    r"\b(fails?|doesn'?t work|broken|cannot|can'?t|unstable|instabilit(y|ies)|"
    r"diverges?|nan|nans?|oom|out of memory|hangs?|slow|degrad(es|ed|ation)|"
    r"limitation|edge.case|not.support(ed|s)|reproduc(e|ible)|wrong|incorrect|"
    r"mismatch|inconsistent|deadlock|race condition)\b",
    re.I,
)
out = []
for it in raw:
    labels = {l["name"].lower() for l in it.get("labels", [])}
    text = (it["title"] or "") + " " + (it["body"] or "")
    keyword_hit = bool(BUG_KEYWORDS.search(text))
    label_hit = bool(labels & BUG_LABELS)
    if not (keyword_hit or label_hit):
        continue
    out.append({
        "number": it["number"],
        "title": it["title"],
        "url": it["url"],
        "state": it["state"],
        "labels": sorted(labels),
        "comments": it["comments"],
        "created": it["createdAt"],
        "body_excerpt": (it["body"] or "")[:600],
    })
# rank by community engagement: high comments + still open > low comments + closed
out.sort(key=lambda x: (-x["comments"], 0 if x["state"] == "OPEN" else 1, -int(x["number"])))
pathlib.Path(f"/tmp/idea-explore/$slug/issues.json").write_text(json.dumps(out[:40], indent=2))
print(f"filtered to {len(out)} bug-flavored issues; kept top 40 by engagement")
PY
```

**Drawback-quality check** before passing to Step 5: an issue with `comments: 0` and a one-line body is almost always a setup question, not a method drawback. The ranking above pushes those to the bottom — the top entries (high comment count, recent, often still open) are the ones that signal a real community-known limitation.

## Step 4 — Adjacent angles (deep research)

Invoke `academic-deep-research` on 2-3 adjacent queries that cross the seed with neighboring sub-fields. The queries should be **derived from the seed's method**, not arbitrary topics. Examples for a seed on "flow matching for image generation":

- `flow matching audio` (cross-modality)
- `consistency models distillation` (sister method)
- `flow matching theory convergence` (theoretical angle)

Do **not** invent more than 3 queries — adjacent literature is anchoring material, not the centerpiece. Write each query's report into `/tmp/idea-explore/<slug>/adjacent/<query-slug>.md`.

For each adjacent report, extract a 1-paragraph "what this branch is doing that the seed isn't" note. Append all of these to `/tmp/idea-explore/<slug>/adjacent.md` for Step 5 consumption.

## Step 5 — Parallel idea synthesis

Dispatch **3-4 Agent subagents in a single message** (one Agent tool call per subagent, all in one assistant turn). Give each subagent a different *lens* — this forces diversity instead of 4 versions of the same idea:

| Subagent | Lens | What it reads |
|---|---|---|
| `extend-gap` | "What's a natural extension that the followup cone *hasn't* covered yet?" | `exclusions.json`, seed abstract |
| `fix-drawback` | "What ideas address the issues the community has actually filed?" | `issues.json`, seed abstract |
| `adjacent-cross` | "What cross-pollination from neighboring sub-fields is open?" | `adjacent.md`, seed abstract |
| `criticism-resolve` | "Of the `Criticizes`-tagged follow-ups, which criticism is unresolved?" | `followups.md` (Criticisms section), seed abstract |

Prompt template for each subagent:

> You are proposing new research ideas anchored to a specific seed paper. Read these files:
> - `/tmp/idea-explore/<slug>/seed.json` — seed paper metadata + abstract
> - `<lens-specific files>`
>
> Emit 3-5 candidate ideas. Each idea must have:
> - **Title** — one-line, ≤12 words, concrete (not "use LLMs for X")
> - **Mechanism** — 2-3 sentences on what the idea actually does technically
> - **Anchor** — verbatim pointer to the followup-cone gap, GitHub issue (`#N url`), or adjacent-lit paragraph that motivates this. **No anchor → drop the idea.**
> - **Delta-vs-seed** — one sentence on what changes from the seed paper's method
> - **Why-not-done-yet** — one sentence; if you can't say this, the idea is probably already done
> - **Risk** — the single most likely failure mode
> - **MVP signal** — the smallest experiment that would prove or kill it
>
> Write to `/tmp/idea-explore/<slug>/ideas.<lens>.md` in the above schema, one idea per `### Title` block. Return only "OK <lens>". Do NOT inline the ideas in your response.

Disk-based handoff is mandatory — same reason as `followup-analysis` and `youtube-wiki`. The user only sees the final report; bloating the main context with 16 inline idea cards loses the point.

## Step 6 — Synthesize report

Read all `/tmp/idea-explore/<slug>/ideas.<lens>.md`, dedupe (two lenses sometimes converge on the same idea — merge anchors and keep one entry), and rank by anchor strength:

1. Ideas anchored to a `Criticizes`-tagged follow-up OR an issue with ≥10 comments → **High anchor**
2. Ideas anchored to a single drawback issue OR adjacent-lit angle → **Medium anchor**
3. Ideas anchored to a "gap in the cone" with no issue support → **Speculative**

Cap at 10 ideas total in the final report — more than that, the user can't act on it. Drop the lowest-anchor speculative entries first.

Write to the destination determined in Step 1. Template:

```markdown
# Idea Explore: <Seed Paper Title>

**Date**: YYYY-MM-DD
**Seed**: <title> ([arxiv](https://arxiv.org/abs/<id>))
**Method**: <one-line technical summary>
**Official repo**: <owner/name> | NONE (justified)
**Idea-box slug** (if any): <slug>

## TL;DR

- <one bullet per High-anchor idea, ≤15 words>
- <one bullet on the strongest *exclusion* — what's already been tried and why we're not proposing it again>
- <one bullet on the strongest *drawback* — what the community is most clearly stuck on>

## Already done (exclusion list)

These directions are covered by existing follow-ups. **New ideas must not duplicate these.**

| Direction | Representative follow-up | Tag | Cites |
|---|---|---|---|
| ... | <title> | Extends | N |
| ... | <title> | Improves | N |

(pulled from `followups.md` `Extensions / Improvements / Theoretical / Applications` clusters)

## Known drawbacks (community-reported)

| # | Issue | State | 💬 | Drawback summary |
|---|---|---|---|---|
| 1 | [#123 title](url) | OPEN | 47 | <one-line summary from issue body> |

(top 5-10 from `issues.json`; if `repo.txt` was `NONE`, this section says so and explains why)

## Adjacent angles

- **<query-slug>** — 1 paragraph: what the adjacent branch is doing that the seed isn't.
- ...

## Candidate ideas

### High-anchor

#### 1. <Title>
- **Mechanism**: ...
- **Anchor**: [#456 (12 comments)](issue-url) — "<verbatim phrase>"  OR  Criticizes: <follow-up title> — "<verbatim phrase>"
- **Delta vs seed**: ...
- **Why not done yet**: ...
- **Risk**: ...
- **MVP signal**: ...

### Medium-anchor

#### N. <Title>
...

### Speculative

#### N. <Title>
...

## What we're betting on

3-5 sentences synthesizing the *direction* (not a list of ideas): which lens produced the highest-anchor ideas, what theme keeps recurring across drawbacks + criticisms, where the user should focus first. This is the part the user reads in six months to remember why they picked direction X over Y.

## Verification log

- followup-analysis report: `<path>` — N follow-ups, K in exclusion-worthy tags
- GitHub repo: `<owner/name>` (or `NONE` — reason: ...)
- Issues filtered: M raw → K bug-flavored (kept top 40)
- Adjacent queries: <list>
- Candidate ideas: P proposed before dedupe, Q after rank-and-cap
- Anchors verified: every idea has ≥1 anchor pointing to a real artifact

## Next step

If idea-box: candidates 1-3 are tagged as ready for `idea-box +new` with the Origin section linking to this file.
Else: pick one and run `idea-feasibility` to score the four hard gates.
```

If destination is `./idea_box/<slug>/explore.md`, also append a row to `./idea_box/INDEX.md` noting `explore: yes` on that idea (or follow the idea-box `+regen-index` convention if `INDEX.md` is structured differently in the user's idea-box repo).

## Step 7 — Verify

Before declaring done:

```bash
slug=<slug>
report=<destination path>
python3 - <<PY
import json, re, pathlib
report = pathlib.Path("$report").read_text()
excl = json.load(open("/tmp/idea-explore/$slug/exclusions.json"))
issues = json.load(open("/tmp/idea-explore/$slug/issues.json"))

# 1. exclusion-list collisions: no idea title should be near-identical to an exclusion title
import difflib
idea_titles = re.findall(r"^####? \d+\.\s*(.+)$", report, re.M)
collisions = []
for t in idea_titles:
    for e in excl:
        ratio = difflib.SequenceMatcher(None, t.lower(), e["title"].lower()).ratio()
        if ratio > 0.75:
            collisions.append((t, e["title"], ratio))
print("exclusion collisions:", collisions or "OK")

# 2. every drawback URL in the report must exist in issues.json
issue_urls = {i["url"] for i in issues}
report_urls = set(re.findall(r"https?://github\.com/[^\s)]+/issues/\d+", report))
fabricated = report_urls - issue_urls
print("fabricated issue URLs:", fabricated or "OK")

# 3. every idea must have an Anchor: line
anchor_lines = re.findall(r"^- \*\*Anchor\*\*:", report, re.M)
print(f"anchors: {len(anchor_lines)} / ideas: {len(idea_titles)}",
      "OK" if len(anchor_lines) >= len(idea_titles) else "MISSING")
PY
```

If exclusion collisions are non-empty, the proposed idea overlaps a known follow-up — re-run Step 5 with a sharper prompt or drop the idea. If fabricated URLs appear, you hallucinated an issue — fix before saving. If anchors are missing, an idea slipped the "no anchor → drop" rule — remove it.

## Quick Reference

| Step | Command | Output |
|------|---------|--------|
| Resolve | `curl /paper/search?query=...` + GitHub repo lookup | `seed.json`, `repo.txt` |
| Already-done | Invoke `followup-analysis` skill | `followups.md`, `exclusions.json` |
| Drawbacks | `gh issue list --repo <owner/name>` + filter | `issues.json` |
| Adjacent | Invoke `academic-deep-research` ×2-3 | `adjacent/<query>.md`, `adjacent.md` |
| Synthesize | 3-4 parallel Agents (single message) | `ideas.<lens>.md` |
| Report | manual, follow template | `explore.md` |
| Verify | python collision + URL + anchor check | clean diff |

## Anti-Patterns

- **Skipping the exclusion step.** Proposing an idea that's already a published follow-up is the worst failure mode of this skill. Always run `followup-analysis` first, even when the user is impatient.
- **Treating every GitHub issue as a method drawback.** Many issues are setup confusion ("how do I install on Mac?") not method limitations. The comment-count + keyword filter exists for this reason — do not skip it.
- **Substituting a Google search for the GitHub-issue scan when the repo is unknown.** Per the global no-silent-fallbacks rule: write `NONE` to `repo.txt`, log it in the verification section, and degrade gracefully. Do not paraphrase blog posts as if they were issues.
- **Hallucinating issue numbers or URLs.** Every `#N` and every `github.com/.../issues/N` URL in the report must come from `issues.json`. The Step 7 verifier catches this — do not bypass it.
- **Inline subagent ideas in the main response.** Bloats your context, the user sees nothing. Subagents write to disk; main agent reads disk at synthesis.
- **More than 3 adjacent-lit queries.** Adjacent literature is anchoring fuel, not the dish. 2-3 queries is plenty; 6 queries wastes the deep-research budget and dilutes the report.
- **Capping ideas above 10.** The user cannot act on 25 proposals. Rank by anchor strength, keep the top 10, drop speculative entries first.
- **Writing the report from memory after reading ideas.<lens>.md.** Read the files at synthesis time. A 16-idea list across 4 lenses is exactly the kind of thing your context will silently re-order or merge.
- **Auto-creating `./idea_box/<slug>/`.** Per the idea-box contract, only write into an idea-box dir that already exists. If the user wants idea-box integration, they should `+new` first and then re-run idea-explore from inside that dir.
- **Proposing ideas without a `Why-not-done-yet`.** If you can't say why the field hasn't already done this, it's probably already been done (and you missed it in Step 2) or it's not actually interesting.

## See Also

- `followup-analysis` — Produces the exclusion list this skill consumes; run it first or let idea-explore invoke it as Step 2
- `idea-feasibility` — Natural next step on the top-ranked proposed idea; scores it against the four hard gates (checkpoint / code / dataset / GPU)
- `idea-box` — Lifecycle home for the candidate ideas; this skill writes into `./idea_box/<slug>/explore.md` when invoked from an idea-box working directory
- `academic-deep-research` — Invoked in Step 4 for adjacent-literature angles; also useful before this skill to confirm the seed paper is worth exploring from
- `arxiv-latex-reader` — Read the seed paper deeply before running this skill; the better you understand the seed, the sharper Step 5's idea synthesis
- `pdf-reader` — Same role as arxiv-latex-reader for non-arXiv seeds
- `github-reader` — Drill into the official repo's implementation when an issue points at a code-level limitation
- `github-cli` — Authentication and patterns for `gh issue list` used in Step 3
- `ml-ablation-design` — Once you've picked an idea and confirmed feasibility, design the actual ablation matrix
- `fail-fast-ml-engineering` — Shares the "anchor or drop, no silent fabrications" stance this skill enforces on every proposed idea

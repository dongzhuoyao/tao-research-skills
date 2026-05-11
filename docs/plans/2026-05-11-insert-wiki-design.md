# insert-wiki — design

**Date:** 2026-05-11
**Category:** `skills/research/`
**Motivation:** Given any URL — most commonly an X tweet, but also a LinkedIn post, HN thread, short blog, or random web article — capture it as a small, persistent markdown entry under `docs/wikis/` that the user can re-read later for motivation or grep across future sessions. This is the "commonplace book" sibling of the existing reader family: `youtube-wiki`, `blog-reader`, `pdf-reader`, `arxiv-latex-reader`, `github-reader` each do heavy digests of a single URL type; `insert-wiki` is the lightweight catch-all for everything else.

## Scope

| Decision | Chosen | Rejected alternatives |
|---|---|---|
| Skill shape | Generic catch-all with motivational flavor (A+D from brainstorming) | Router/dispatcher (adds shell-out complexity for negligible UX gain); X-only specialization (loses the broader value) |
| Output location | `docs/wikis/YYYY-MM-DD-<slug>.md`, one file per entry | Single growing `wiki.md` (poor grep); per-host folders (premature) |
| Per-entry shape | Verbatim blockquote + tags + "Why I saved this" hook | Bare verbatim (loses motivation context); full digest with TL;DR + people/papers tables (heavyweight for a 280-char tweet) |
| Index file | None — rely on filename date prefix + grep over frontmatter tags | `docs/wikis/INDEX.md` (extra bookkeeping for marginal benefit at small scale; revisit if >100 entries) |
| Fetch strategy | WebFetch by default; CDP via Playwright MCP / `gstack` for known login-walled hosts (`x.com`, `twitter.com`, `linkedin.com`) | WebFetch-only (fails on tweets — the primary use case); CDP-only (forces Chrome dependency for plain blog posts) |
| Specialized URLs | Redirect: instruct user to invoke `youtube-wiki` / `arxiv-latex-reader` / `github-reader` instead | Internal dispatch (couples skills; harder to verify; "no silent fallback" rule prefers a hard hand-off) |
| Failure mode | Hard stop on login wall, missing CDP, or empty body; surface the specific cause | Silent WebFetch fallback on login-walled hosts (would capture a login page and call it the tweet) |
| Media attachments | Reference inline as `[image: <alt>](<url>)`; no download | Download to repo (bloats; raises license questions for public repo) |
| "Why I saved this" | Auto-drafted from content + user's framing in the same turn; placeholder if no framing | Always require user-supplied text (interrupts flow); always autogenerate (invents motivation that wasn't there) |
| Scripts | None — pure orchestration of WebFetch, Playwright MCP browser tools, Write | Custom Python helper (the existing tools cover fetch + write; rest is small-shape logic) |

## Pipeline

```
┌───────────────────────────────┐
│ 1. Classify host              │  x.com / linkedin → CDP
└──────────┬────────────────────┘  github / arxiv / youtube → redirect
           ▼                       everything else → WebFetch
┌───────────────────────────────┐
│ 2. Fetch                      │  raw text + author + posted date
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 3. Derive slug                │  <handle>-<first-5-words> | <title-first-5>
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 4. Compose entry              │  frontmatter + verbatim body + Why-I-saved-this
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 5. Write + verify             │  frontmatter parses; body has blockquote
└──────────┬────────────────────┘
           ▼
   docs/wikis/YYYY-MM-DD-<slug>.md
```

## Host routing

| Host | Path | Why |
|---|---|---|
| `x.com`, `twitter.com` | Playwright MCP via existing CDP (port 9222, `chrome-debug-v2` profile) | Login wall; user already has authenticated session |
| `linkedin.com` | Same CDP path | Login wall |
| `youtu.be`, `youtube.com` | **Stop, redirect to `youtube-wiki`** | Specialized pipeline already exists |
| `arxiv.org` | **Stop, redirect to `arxiv-latex-reader`** | Specialized pipeline |
| `github.com/<owner>/<repo>` (repo root) | **Stop, redirect to `github-reader`** | Specialized pipeline |
| `github.com/<owner>/<repo>/issues/...`, gists | WebFetch (small, public, no specialized skill) | Generic public content |
| Everything else | WebFetch | Default |

When CDP is needed but Chrome on 9222 is unreachable: surface the startup command from `~/.claude/CLAUDE.md` and the `login-cdp` skill name. Do not fall back to WebFetch — the captured login wall would silently corrupt the wiki entry.

## Output format

```markdown
---
source: <url>
author: "@handle" | "Author Name"
posted: YYYY-MM-DD          # omit if unknown
captured: YYYY-MM-DD          # today
host: x.com | linkedin.com | <domain>
kind: tweet | thread | blog | article | post
tags: [tag-1, tag-2]          # 2-4, lowercase-kebab
---

# <Title or first ~8 words>

**Source**: [<short-display-url>](<url>) · **By** <author> · **Captured** YYYY-MM-DD

> <verbatim body — full tweet, or full thread joined with blank `>` lines>

**Why I saved this**: <1-2 sentences, or `<add motivation hook>` placeholder>
```

Threads: walk the thread in order, separate each tweet by a blank `>` line. Embedded images: inline `[image: <alt-or-caption>](<url>)`.

## Anti-patterns

- Paraphrasing the captured body — verbatim only, even for tweets where the temptation to "clean up" is high.
- Inventing a motivation hook when the user gave no framing — use the placeholder instead.
- Silent WebFetch fallback on `x.com` / `linkedin.com` — captures a login wall and pretends it's the post.
- Auto-invoking `youtube-wiki` / `arxiv-latex-reader` / `github-reader` from inside this skill — hand off cleanly, don't couple.
- Downloading media — reference inline.
- Overwriting an existing wiki entry — append `-2`, `-3` to the slug.
- Heavy multi-section digest for a 280-char tweet — that's `blog-reader`'s shape, not this one's.

## Verification

```bash
test -f docs/wikis/YYYY-MM-DD-<slug>.md
head -20 docs/wikis/YYYY-MM-DD-<slug>.md | grep -E '^(source|author|captured|host|kind|tags):'
grep -E '^> ' docs/wikis/YYYY-MM-DD-<slug>.md
grep -F 'Why I saved this' docs/wikis/YYYY-MM-DD-<slug>.md
```

All four must pass before the skill reports done. Failures stop and surface the specific check that failed.

## README updates

- Bump badge `skills-30-blue` → `skills-31-blue` and "30 self-contained agent skills" → "31".
- Add `insert-wiki` row to the Research & Reading table.

## Example invocations

```
Insert this tweet: https://x.com/TechShiba/status/2053440457934262776
```
→ `docs/wikis/2026-05-11-techshiba-<first-words>.md`

```
Add to wiki: https://paulgraham.com/taste.html
```
→ `docs/wikis/2026-05-11-pg-taste-essay.md` via WebFetch.

```
Insert https://youtu.be/abc123
```
→ Stop: "This is a YouTube URL — invoke `youtube-wiki` instead. It produces a richer, timestamped digest under `docs/videos/`."

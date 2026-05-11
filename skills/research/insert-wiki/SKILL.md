---
name: insert-wiki
description: "Use when capturing a single URL (X tweet, LinkedIn post, HN thread, short blog, news article, GitHub gist or issue) as a small persistent markdown entry the user can re-read later for motivation or grep across future sessions. Verbatim body + author + tags + a one-line \"Why I saved this\" hook. Writes to docs/wikis/YYYY-MM-DD-<slug>.md. WebFetch by default; CDP via Playwright MCP for login-walled hosts (x.com, twitter.com, linkedin.com). Redirects YouTube / arXiv / GitHub repo URLs to their specialized skills. Triggers: \"insert wiki\", \"add to wiki\", \"insert this\", \"save this tweet\", \"capture this post\", \"add this to my wiki\", \"remember this link\", \"https://x.com/\", \"https://twitter.com/\", \"https://linkedin.com/posts/\""
---

# Insert Wiki

## When to Use

- Capturing a single URL as a small commonplace-book entry: X tweet, LinkedIn post, HN thread, short blog post, news article, GitHub gist or single issue
- Building up a personal wiki under `docs/wikis/` that you can come back to for motivation or grep across future sessions
- Any prompt that pastes a URL with intent like "insert", "add to wiki", "save this", "remember this", "capture this"

Not for:
- YouTube videos → use `youtube-wiki` (richer, timestamped output under `docs/videos/`)
- arXiv papers → use `arxiv-latex-reader`
- GitHub repository roots → use `github-reader`
- Long-form technical blog posts where you want a faithful figure-aware digest → use `blog-reader`
- PDFs → use `pdf-reader`

When the URL matches one of the specialized cases above, **stop and hand off** — do not internally dispatch.

## Output

A single markdown file at `docs/wikis/YYYY-MM-DD-<slug>.md`:

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

# <Title or first ~8 words of body>

**Source**: [<short-display-url>](<url>) · **By** <author> · **Captured** YYYY-MM-DD

> <verbatim body — full tweet, or full thread joined with blank `>` lines>

**Why I saved this**: <1-2 sentences, or `<add motivation hook>` placeholder>
```

The entry is a door back to the source plus a small motivation hook — not a digest. Keep it short on purpose. If you find yourself writing a TL;DR or a key-points list, the wrong skill is loaded; route to `blog-reader` or `youtube-wiki` instead.

## Pipeline

```
┌───────────────────────────────┐
│ 1. Classify host              │  redirect → specialized skill if matched
└──────────┬────────────────────┘  CDP for login-walled hosts
           ▼                       WebFetch for the rest
┌───────────────────────────────┐
│ 2. Fetch                      │  raw text + author + posted date
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 3. Derive slug                │  <handle>-<first-5-words> | <title-first-5>
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 4. Compose entry              │  frontmatter + verbatim body + hook
└──────────┬────────────────────┘
           ▼
┌───────────────────────────────┐
│ 5. Write + verify             │  4-check verification (see below)
└──────────┬────────────────────┘
           ▼
   docs/wikis/YYYY-MM-DD-<slug>.md
```

## Step 1 — Classify the host

Parse the URL and route:

| Host pattern | Action |
|---|---|
| `x.com/<handle>/status/<id>`, `twitter.com/<handle>/status/<id>` | Go to Step 2 via **CDP** |
| `linkedin.com/posts/...`, `linkedin.com/pulse/...` | Go to Step 2 via **CDP** |
| `youtu.be/<id>`, `youtube.com/watch?v=<id>` | **Stop.** "This is a YouTube URL — invoke `youtube-wiki` instead. It produces a richer, timestamped digest under `docs/videos/`." |
| `arxiv.org/abs/<id>`, `arxiv.org/pdf/<id>` | **Stop.** "Use `arxiv-latex-reader` for arXiv papers." |
| `github.com/<owner>/<repo>` (repo root, no path beyond) | **Stop.** "Use `github-reader` for GitHub repositories." |
| `github.com/<owner>/<repo>/issues/...`, `gist.github.com/...` | Go to Step 2 via **WebFetch** |
| Anything else | Go to Step 2 via **WebFetch** |

## Step 2 — Fetch

### 2a. CDP path (login-walled hosts)

Use Playwright MCP browser tools (already wired to the user's authenticated Chrome session on port 9222, profile `chrome-debug-v2`).

```
playwright__browser_navigate(url=<url>)
playwright__browser_snapshot()
```

From the snapshot extract:
- Tweet / post body text (verbatim, preserve line breaks)
- Author display name + handle
- Posted timestamp (often in a `<time datetime="...">` attribute → convert to `YYYY-MM-DD`)
- For X threads: walk the thread by collecting all tweets from the same author posted in reply chain, in order
- Embedded images: capture URL + alt text, render as `[image: <alt>](<url>)`

If CDP is unreachable (browser closed, port 9222 not bound, navigation fails), **stop and surface**:

```
Chrome CDP unavailable on 9222. Start it with:
  /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
    --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-debug-v2"
Then re-run. If logged out, invoke /login-cdp first.
```

**Do not fall back to WebFetch** for login-walled hosts — it will capture a login wall and silently call it the post. Per the repo's no-silent-fallbacks rule.

### 2b. WebFetch path (everything else)

Plain `WebFetch` against the URL. From the returned content extract:
- Main body text (the article / post / issue, not nav or footer)
- Author (from byline, `meta` tags, or HN post header)
- Posted date (best-effort; omit `posted:` from frontmatter if not findable — don't invent)
- Page title

If WebFetch returns an empty body, an obvious login wall, or a paywall stub: **stop and surface** the specific failure. Don't paper over it.

## Step 3 — Derive the slug

Filename: `docs/wikis/YYYY-MM-DD-<slug>.md` where `YYYY-MM-DD` is today's date.

| Source type | Slug recipe |
|---|---|
| X tweet | `<handle>-<first-5-words-of-tweet>` |
| LinkedIn post | `<handle-or-author-last-name>-<first-5-words>` |
| Generic web page | `<first-5-words-of-page-title>` |
| HN thread | `hn-<first-5-words-of-title>` |
| GitHub gist | `gist-<owner>-<first-5-words-of-description>` |

Normalize: lowercase, kebab-case, strip emoji and punctuation, collapse repeated `-`, max 50 chars. Examples:
- `2026-05-11-techshiba-fastest-way-to-learn-agents.md`
- `2026-05-11-paulg-the-best-essays-are-rewrites.md`
- `2026-05-11-hn-show-hn-a-typed-jq-replacement.md`

**Collision**: if the slug already exists at the same date, append `-2`, `-3`. Never overwrite.

## Step 4 — Compose the entry

Fill the template (see "Output" above). Rules:

- **Body must be verbatim.** A tweet is so short that paraphrase is misquote. Preserve line breaks, emoji, and capitalization. Only filler artifacts injected by the scraper (e.g. literal "Show this thread" UI strings) may be stripped.
- **Threads**: each tweet on its own blockquote paragraph, separated by a blank `>` line. Maintain order.
- **Embedded media**: inline `[image: <alt or caption>](<url>)`. Do not download.
- **Author**: prefer `"@handle"` for social hosts, `"Author Name"` for blogs. Quote the value in YAML to keep the `@` safe.
- **Tags**: pick 2-4 lowercase-kebab tokens drawn from the content (e.g. `motivation`, `agentic-eng`, `taste`, `hpc`). These are the grep handles for future sessions.
- **`Why I saved this`**: draft a 1-2 sentence hook synthesizing (a) what the post says and (b) any framing the user gave when handing you the URL (e.g. "I want motivation from this", "this captures taste"). If the user said nothing beyond "insert this", write the literal placeholder `<add motivation hook>` rather than fabricating motivation that wasn't theirs.

## Step 5 — Write + verify

Write the file with `Write`, then run all four checks:

```bash
# 1. File exists at expected path
test -f docs/wikis/YYYY-MM-DD-<slug>.md

# 2. Frontmatter has the required fields
head -20 docs/wikis/YYYY-MM-DD-<slug>.md | grep -E '^(source|author|captured|host|kind|tags):'

# 3. Body contains a verbatim blockquote
grep -E '^> ' docs/wikis/YYYY-MM-DD-<slug>.md

# 4. Motivation hook line present
grep -F 'Why I saved this' docs/wikis/YYYY-MM-DD-<slug>.md
```

Any failure → stop, surface which check failed, do not report success.

Report the relative path of the new file so the user can open it.

## Anti-Patterns

- **Paraphrasing the captured body.** Verbatim only — even when the post is grammatically loose or has typos.
- **Inventing the "Why I saved this" hook.** If the user gave no framing, leave the `<add motivation hook>` placeholder. Don't fabricate motivation.
- **Silent fallback to WebFetch on `x.com` / `linkedin.com`.** Login-walled hosts must use CDP; if CDP is down, hard stop with the startup command.
- **Auto-running `youtube-wiki` / `arxiv-latex-reader` / `github-reader` internally.** When the URL matches a specialized skill's host, tell the user to invoke that skill. Don't couple skills via shell-out.
- **Downloading attachments.** Reference inline (`[image: ...](url)`); no media files in the repo.
- **Overwriting an existing wiki entry.** Append `-2` / `-3` to the slug.
- **Forgetting the date prefix** on the filename — breaks chronological sort and grep-by-date.
- **Heavy multi-section digest** for a 280-char tweet. If you're writing a TL;DR or a People-mentioned table, the wrong skill is loaded — route to `blog-reader` or `youtube-wiki`.
- **Inventing the `posted:` date** when the source doesn't expose it. Omit the field instead.
- **Unquoted `@handle` in YAML.** YAML treats `@` specially at the start of a scalar — always wrap the author value in double quotes.

## Example Invocation

```
Insert this tweet: https://x.com/TechShiba/status/2053440457934262776
```

Expected outcome: `docs/wikis/2026-05-11-techshiba-<first-words>.md` written, frontmatter parses, verbatim tweet body in a blockquote, motivation hook either drafted from context or left as `<add motivation hook>`, all four verification checks pass.

```
Insert https://youtu.be/abc123
```

Expected outcome: **stop**, with: "This is a YouTube URL — invoke `youtube-wiki` instead. It produces a richer, timestamped digest under `docs/videos/`."

## See Also

- `youtube-wiki` — Specialized skill for YouTube videos. Insert-wiki redirects YouTube URLs here.
- `blog-reader` — For long-form technical blog posts that warrant a full figure-aware digest. Insert-wiki is the lightweight cousin for short posts.
- `arxiv-latex-reader` — For arXiv papers. Insert-wiki redirects arXiv URLs here.
- `pdf-reader` — For PDFs. Insert-wiki does not handle PDFs.
- `github-reader` — For GitHub repository roots. Insert-wiki handles only gists and individual issues.
- `login-cdp` — Refresh authenticated CDP sessions when X / LinkedIn capture fails because the session expired.
- `github-cli` — Patterns for `gh api` when capturing GitHub issues without WebFetch flakiness.

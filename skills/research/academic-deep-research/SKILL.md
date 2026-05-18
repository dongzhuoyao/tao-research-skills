---
name: academic-deep-research
description: "Use when evaluating academic papers or surveying a research topic. Gathers venue, citations, GitHub stats, social buzz, reproducibility, and author signals to produce a scored markdown report. Triggers: \"evaluate paper\", \"paper review\", \"research survey\", \"literature review\", \"is this paper good\", \"find papers on\", \"compare papers\", \"paper impact\""
---

# Academic Deep Research

## When to Use

- Evaluating a single paper (given arxiv URL, title, or DOI)
- Surveying a research topic and ranking papers by quality/impact
- Deciding whether to read, cite, or build on a paper
- Comparing multiple papers in a research area

## Two Workflows

### Single Paper Mode

Input: arxiv URL, paper title, or DOI.

- [ ] **Extract metadata** — title, authors, date, abstract from arxiv or Semantic Scholar
- [ ] **Query citations + venue** — Semantic Scholar API
- [ ] **Find GitHub repo** — Papers With Code → paper page → WebSearch fallback chain
- [ ] **Gather code health** — `gh api` for stars, forks, last push, open issues
- [ ] **Check reproducibility** — HuggingFace models/datasets, third-party reimplementations
- [ ] **Search social buzz** — Twitter/X, Reddit, HackerNews via WebSearch
- [ ] **Score all 8 dimensions** — apply scoring rubric below
- [ ] **Write report** — save to `docs/research/YYYY-MM-DD-<paper-slug>.md`

### Topic Survey Mode

Input: research topic or question.

- [ ] **Find candidate papers** — WebSearch + Semantic Scholar, collect 5-15 papers
- [ ] **Evaluate each paper** — run Single Paper Mode per paper (parallelize via subagents)
- [ ] **Rank by composite score** — descending
- [ ] **Write report** — ranked table + per-paper scorecards to `docs/research/YYYY-MM-DD-<topic-slug>-survey.md`

## Scoring Rubric (8 Dimensions, 0-20)

### 1. Venue (0-3)

| Score | Criteria |
|-------|----------|
| 3 | Top-tier oral/spotlight (NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV, ACL, EMNLP, SIGGRAPH, JMLR, TPAMI) |
| 2 | Top-tier poster/accepted |
| 1 | Workshop, second-tier venue (AAAI, WACV, BMVC, COLING), or journal under review |
| 0 | Arxiv-only, no peer review |

### 2. Citations (0-3)

Normalized by paper age (citations per month since publication):

| Score | Criteria |
|-------|----------|
| 3 | >10 citations/month |
| 2 | 3-10 citations/month |
| 1 | 0.5-3 citations/month |
| 0 | <0.5 citations/month |

### 3. Code Available (0-1)

Binary. 1 = official repo exists. 0 = no code released.

### 4. Code Health (0-3)

Only scored if code exists:

| Score | Criteria |
|-------|----------|
| 3 | >1k stars AND last push <30 days |
| 2 | >200 stars OR (>50 stars AND last push <30 days) |
| 1 | Repo exists, <200 stars, last push <6 months |
| 0 | Repo abandoned (last push >6 months) or no repo |

### 5. Reproducibility (0-3)

| Score | Criteria |
|-------|----------|
| 3 | HuggingFace models/datasets + third-party reimplementations |
| 2 | Either HF presence OR third-party reimplementation |
| 1 | Official code runs, but no third-party adoption |
| 0 | No code, no reproductions |

### 6. Social Buzz (0-3)

| Score | Criteria |
|-------|----------|
| 3 | Viral — multiple platforms, >200 likes/upvotes total |
| 2 | Notable — discussed on 1-2 platforms, moderate engagement |
| 1 | Mentioned — a few tweets or a Reddit thread |
| 0 | No detectable discussion |

### 7. Recency (0-2)

| Score | Criteria |
|-------|----------|
| 2 | Published within last 12 months |
| 1 | 1-3 years old |
| 0 | >3 years old |

### 8. Author Signal (0-2)

| Score | Criteria |
|-------|----------|
| 2 | Top lab (FAIR, DeepMind, OpenAI, MSR, etc.) OR first/last author h-index >40 |
| 1 | Known university group OR h-index 15-40 |
| 0 | Unknown affiliation, early-career authors |

### Composite Tiers

| Score | Tier |
|-------|------|
| 16-20 | Landmark paper |
| 11-15 | Strong paper |
| 6-10 | Solid paper |
| 0-5 | Early/niche/weak signal |

## Data Gathering Commands

### Semantic Scholar (citations, venue, authors)

```
WebFetch → https://api.semanticscholar.org/graph/v1/paper/arXiv:{id}?fields=title,year,venue,citationCount,authors,externalIds,publicationVenue,publicationTypes
```

For title search:
```
WebFetch → https://api.semanticscholar.org/graph/v1/paper/search?query={url_encoded_title}&limit=5&fields=title,year,venue,citationCount,authors,externalIds,publicationVenue
```

### GitHub Stats

```bash
gh api repos/{owner}/{repo} --jq '{stars: .stargazers_count, forks: .forks_count, last_push: .pushed_at, open_issues: .open_issues_count}'
```

### Repo Discovery Fallback Chain

Try in order until a repo is found:
1. `WebSearch → "{paper title} site:paperswithcode.com"` — check the Papers With Code page for linked repos
2. `WebFetch` the arxiv abstract page — look for GitHub links in the abstract or comments
3. `WebSearch → "{paper title} github"` — direct search
4. `WebSearch → "{first author name} {method name} github"` — author-based search

### Social Buzz

```
WebSearch → "{paper title}" AND (site:x.com OR site:twitter.com OR site:reddit.com OR site:news.ycombinator.com)
```

If no results, broaden:
```
WebSearch → "{method name} {first author last name}" AND (twitter OR reddit OR "hacker news")
```

### HuggingFace Adoption

```
WebSearch → "{method name}" site:huggingface.co
```

### Topic Survey — Paper Discovery

```
WebSearch → "{topic} survey OR benchmark OR state-of-the-art 2024 2025"
WebFetch → https://api.semanticscholar.org/graph/v1/paper/search?query={topic}&limit=20&fields=title,year,venue,citationCount,authors&sort=citationCount:desc
```

## Report Templates

### Single Paper Report

```markdown
# Paper Evaluation: <Title>

**Date**: YYYY-MM-DD
**Paper**: <arxiv link>
**Authors**: <author list>
**Venue**: <conference/journal or "arxiv preprint">

## Scorecard

| Dimension       | Score | Detail                              |
|-----------------|-------|-------------------------------------|
| Venue           | _/3   |                                     |
| Citations       | _/3   |                                     |
| Code Available  | _/1   |                                     |
| Code Health     | _/3   |                                     |
| Reproducibility | _/3   |                                     |
| Social Buzz     | _/3   |                                     |
| Recency         | _/2   |                                     |
| Author Signal   | _/2   |                                     |
| **Total**       | **_/20** | **<tier>**                       |

## Key Findings

- <2-3 bullet summary of contributions>

## Code & Reproducibility

- Repo: <link> | Stars: X | Forks: X | Last push: X
- HuggingFace: <models/datasets if any>
- Third-party implementations: <links if any>

## Social Discussion

- <Notable tweets, Reddit threads, HN posts with links>

## Verdict

<1-2 sentence recommendation: worth reading/implementing/citing?>
```

### Topic Survey Report

```markdown
# Topic Survey: <Topic>

**Date**: YYYY-MM-DD
**Papers evaluated**: N

## Ranked Papers

| Rank | Paper | Venue | Score | Stars | Citations |
|------|-------|-------|-------|-------|-----------|
| 1    | ...   | ...   | _/20  | ...   | ...       |

## Per-Paper Evaluations

<individual scorecards>
```

## Anti-Patterns

- **Scoring without data**: Never assign a score based on assumption. If a data source is unreachable, mark as "N/A" and note it. Do not guess.
- **Stars as sole proxy for quality**: A viral repo with 10k stars may have no peer review. Stars measure popularity, not correctness. Always cross-check with venue and citations.
- **Ignoring negative signals**: An abandoned repo (last push >1 year), retracted paper, or controversy in discussions are important findings. Report them prominently.
- **Recency bias**: A 3-year-old paper with 2000 citations is more impactful than a 1-month-old paper with 5 stars. Recency is one signal, not the signal.
- **Skipping social search**: Social discussion often surfaces limitations, failed reproductions, and real-world usage that papers don't mention.
- **Over-weighting author prestige**: A paper from an unknown lab that ships working code and gets adopted beats a famous-lab paper with no code release.

## See Also

- `ml-ablation-design` — Evaluating experimental rigor within papers
- `genai-evaluation-metrics` — Understanding evaluation metrics referenced in papers
- `fail-fast-ml-engineering` — Assessing code quality of released implementations
- `arxiv-latex-reader` — Progressive reading of large arxiv papers without context overflow
- `idea-explore` — Invokes this skill for adjacent-literature angles when proposing new directions anchored to a seed paper

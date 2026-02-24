# Design: academic-deep-research skill

**Date**: 2026-02-24
**Status**: Implemented

## Problem

When evaluating academic papers or surveying a research topic, researchers need to check many dimensions manually: venue acceptance, citation count, GitHub activity, social media reception, reproducibility, author reputation. This is tedious and easy to do incompletely.

## Decision

Single monolithic skill (`academic-deep-research/SKILL.md`) with two workflows:

1. **Single Paper Mode** — given an arxiv URL/title/DOI, produce a comprehensive scorecard
2. **Topic Survey Mode** — given a research topic, find and rank 5-15 papers

Both fully automated using Claude Code's built-in tools (WebSearch, WebFetch, `gh` CLI, Semantic Scholar API).

## Scoring Rubric

8 dimensions, 0-20 composite scale:

| Dimension | Max | Source |
|-----------|-----|--------|
| Venue | 3 | Semantic Scholar |
| Citations | 3 | Semantic Scholar (normalized by age) |
| Code Available | 1 | Papers With Code / paper page |
| Code Health | 3 | `gh api` (stars, forks, last push) |
| Reproducibility | 3 | HuggingFace, third-party reimpl |
| Social Buzz | 3 | WebSearch (Twitter, Reddit, HN) |
| Recency | 2 | Arxiv metadata |
| Author Signal | 2 | Semantic Scholar (affiliation, h-index) |

Tiers: Landmark (16-20), Strong (11-15), Solid (6-10), Early/niche (0-5).

## Alternatives Considered

- **Two separate skills** (single paper + topic survey): Rejected — scoring logic would be duplicated
- **External Python script**: Rejected — adds dependencies, less portable, harder to iterate
- **Manual checklist**: Rejected — user wanted full automation

## Output

Markdown reports saved to `docs/research/YYYY-MM-DD-<slug>.md` with scorecard table, key findings, code/reproducibility details, social discussion links, and verdict.

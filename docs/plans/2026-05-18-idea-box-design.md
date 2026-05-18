# idea-box — design

**Date:** 2026-05-18
**Category:** `skills/research/`
**Motivation:** The repo already has skills for *evaluating* an idea (`idea-feasibility`), *designing experiments around it* (`ml-ablation-design`), *surveying its citation cone* (`followup-analysis`), and *reading its supporting evidence* (`arxiv-latex-reader`, `pdf-reader`, `github-reader`, `blog-reader`, `academic-deep-research`). But there is no canonical place where an idea *lives* across these stages — between sessions, the user re-types the idea, re-attaches the same papers, re-runs feasibility from scratch, and loses the decision history that explains why an idea got promoted or killed. `idea-box` is the missing **workflow primitive**: a per-idea on-disk directory plus a hard-gated state machine that pulls the existing reader/evaluator skills into a coherent lifecycle.

Conceptually it is the **idea-side analog of `workspace/`** (the per-paper scratch directory used by `pdf-reader` / `arxiv-latex-reader`): a stable directory layout with a stable file schema, written into by multiple skills, never owned by any one of them.

## Scope

| Decision | Chosen | Rejected alternatives |
|---|---|---|
| Skill shape | Workflow-first lifecycle skill (state machine + hard gates) | Pure directory convention (no enforcement, loses the discipline `idea-feasibility` already insists on); commonplace book of flat .md files (closer to `insert-wiki`; loses per-idea evidence subdirs and cross-skill composition) |
| Data location | `./idea_box/<slug>/` relative to cwd; user `cd`s into `~/lab/idea-box` (a dedicated private repo) before triggering | Env-var `$IDEA_BOX_ROOT` (extra setup); pointer file per project (extra setup); explicit `--root` every time (verbose, doesn't compose with auto-triggers) |
| Public-repo footprint | Ship only the skill (convention + workflow). Zero idea data in `tao-research-skills`. | Ship a starter `idea_box/` (publishes private notes); machine-global `~/.idea_box/` (couples skill to a specific filesystem location) |
| Gate strictness | Hard gates — refuse transitions when preconditions fail; print specific missing fact | Soft gates (warns but overrides; weakens the no-silent-fallbacks rule); no gates (degrades to a journal) |
| Skill ships scripts? | No — pure documentation. Agent enforces gates by reading on-disk files. | Python `idea_box.py` CLI (adds install burden; gates are file-existence + regex checks, agents handle these fine) |
| File schemas | `idea.md` (frontmatter + 5-bullet normalized spec), `STATUS.md` (current_state + history), `feasibility.md`, `ablations.md`, `decisions.md`, `INDEX.md` | One mega-file per idea (loses the producer-skill writes-to-known-filename contract); JSON-only (less human-readable) |
| Cross-skill plumbing | Producer skills (`idea-feasibility`, `ml-ablation-design`, readers) write to canonical filenames under whatever path the agent passes them. No code changes inside those skills — just `## Idea-box integration` notes in their SKILL.md. | Modify each producer skill to know about idea-box (couples them; harder to evolve independently) |
| Resurrection | Killed ideas are permanent. To revive a thread, create a new idea (new slug) with `## Origin` linking back. | Allow `+resurrect` to flip state back (loses the historical signal that the idea was tried and failed) |
| Index file | `./idea_box/INDEX.md` regenerated from `STATUS.md` + `idea.md` across all slugs | No index (poor list/filter UX); per-state index files (over-engineering) |
| Triggers | `idea box`, `idea-box`, `+new`, `+advance`, `+list`, `+kill`, `+resurrect`, `+regen-index`, `+decide`, plus 想法箱 / 新想法 / 推进想法 / 列出想法 | English-only (excludes Chinese flows the user works in) |

## Architecture

```
./idea_box/
├── INDEX.md                            # generated, append-on-+new
└── <YYYY-MM-DD-idea-slug>/
    ├── idea.md                         # canonical normalized spec (frontmatter + 5 bullets)
    ├── STATUS.md                       # current_state + transition history
    ├── evidence/
    │   ├── arxiv/<paper-slug>.md       # written by arxiv-latex-reader / pdf-reader
    │   ├── github/<repo-slug>.md       # written by github-reader
    │   ├── blogs/<post-slug>.md        # written by blog-reader
    │   └── models/<model-slug>.md      # checkpoint / model-card notes
    ├── feasibility.md                  # written by idea-feasibility (gate input)
    ├── ablations.md                    # written by ml-ablation-design (gate input)
    ├── followups.md                    # written by followup-analysis (optional)
    └── decisions.md                    # append-only log; every override / kill reason
```

## State machine

```
                ┌─ feasible ─┐
                │            ├─→ building ─→ built
explored ──────→┤            │              │
                │            │              └─→ killed
                └─ blocked ──┘
                     │
                     └─→ killed
```

`built` and `killed` are terminal. To revisit a killed idea, create a new slug; link via `## Origin`.

## Hard gates

Every transition is gated by on-disk preconditions. If any precondition fails, the skill refuses to write the new `STATUS.md` and prints the specific missing fact — no silent fallback.

| Transition | Hard gate |
|---|---|
| `*` → `explored` | `idea.md` exists with 5 frontmatter fields filled and 5 spec bullets non-empty |
| `explored` → `feasible` | `feasibility.md` exists, has `### Verdict` block, verdict ∈ {High, Medium} |
| `explored` → `blocked` | `feasibility.md` exists, verdict ∈ {Low, Blocked}, at least one named blocker |
| `blocked` → `feasible` | re-run `feasibility.md` with verdict ∈ {High, Medium} AND `decisions.md` explains what changed |
| `feasible` → `building` | `ablations.md` exists with MVP + minimum hardware AND `decisions.md` records the go-ahead |
| `building` → `built` | `decisions.md` has outcome entry referencing commit / run id / eval number AND `INDEX.md` regenerated |
| `building` → `killed` | `decisions.md` has kill entry naming the specific signal that failed |
| any → `killed` | `decisions.md` has kill entry with one-paragraph reason |

## File schemas

### `idea.md`

```markdown
---
slug: 2026-05-18-flow-matching-text-encoder
created: 2026-05-18
goal_type: research-paper | student-project | prototype | product
language: en | zh
tags: [diffusion, text-encoder, flow-matching]
---

# <One-line title>

- **Problem:** ...
- **Mechanism:** ...
- **Target setting:** ...
- **Expected gain:** ...
- **Required dependencies:** checkpoint(s), code, dataset(s), GPU budget

## Origin

Where the idea came from — 1-3 sentences.
```

Matches the "Normalize the idea" output shape of `idea-feasibility`, so a feasibility run can read `idea.md` directly instead of re-asking.

### `STATUS.md`

```markdown
---
current_state: explored | feasible | blocked | building | built | killed
last_transition_at: 2026-05-18T14:32:00+02:00
---

# Status history

- 2026-05-18T10:00 — created (explored)
- 2026-05-18T14:32 — explored → feasible (idea-feasibility verdict: Medium, gate passed)
```

### `feasibility.md`

Verbatim output of `idea-feasibility`. Must contain a `### Verdict\n[High|Medium|Low|Blocked]` line so the gate can grep it.

### `ablations.md`

Output of `ml-ablation-design`: ablation matrix, success/failure metrics, smallest-experiment plan.

### `decisions.md`

Append-only log. One entry = ISO timestamp + 2-4 sentences. Every override, every kill reason, every "we changed scope because X" goes here. This is the file the user reads in six months to remember why something died.

### `INDEX.md`

Generated by `+regen-index`, never hand-edited:

```markdown
| slug | status | last_touched | one-liner |
|---|---|---|---|
| 2026-05-18-flow-matching-text-encoder | feasible | 2026-05-18 | Replace T5 with flow-matched encoder in SDXL |
```

## Cross-skill integration

**Producers** — skills that write into `./idea_box/<slug>/`:

| Skill | Writes to | When |
|---|---|---|
| `superpowers:brainstorming` / `office-hours` | `idea.md` + state → `explored` | End of a brainstorm if the user keeps the idea |
| `idea-feasibility` | `feasibility.md` | When run on an idea-box idea |
| `ml-ablation-design` | `ablations.md` | After feasibility passes; required to enter `building` |
| `academic-deep-research` | `evidence/arxiv/<paper-slug>.md` | When evaluating a paper as evidence |
| `followup-analysis` | `followups.md` + `evidence/arxiv/*` | When mapping a seed paper's citation cone |
| `arxiv-latex-reader`, `pdf-reader`, `github-reader`, `blog-reader` | `evidence/<type>/<slug>.md` | When invoked from inside an idea-box directory |

**Consumers** — `idea-box` itself reads `feasibility.md`, `ablations.md`, `decisions.md`, and `STATUS.md` to evaluate gates and regenerate `INDEX.md`.

**Critical wiring rule** — producer skills don't need to know about idea-box. They write to whatever path the agent passes them. The "magic" is just: when invoked from inside `idea_box/<slug>/`, the agent passes that path as the destination. This is how `workspace/` works today; no plumbing changes are required inside the producer skills, only `## Idea-box integration` notes pointing to the canonical filename for each producer's output.

## Skill commands

| Command | Behavior | Pre-conditions |
|---|---|---|
| `idea-box +new "<title>"` | Bootstrap `./idea_box/<YYYY-MM-DD-slug>/` with `idea.md` skeleton + `STATUS.md` (state=`explored`) + empty `evidence/` subdirs; append row to `INDEX.md` | `./idea_box/` exists (or skill creates after user confirmation); slug doesn't collide |
| `idea-box +list` | Render `INDEX.md`, sorted by state then last_touched. Filters: `--state=feasible`, `--tag=diffusion` | None |
| `idea-box +show <slug>` | Summarize idea statement, current state, last 3 decisions, evidence count, gates blocking next transition | Slug exists |
| `idea-box +advance <slug>` | Try to transition to next forward state; run hard gates; on failure print the specific missing precondition; never silently write | Per-transition gates |
| `idea-box +kill <slug> "<reason>"` | Append kill entry to `decisions.md`; set state=`killed` | Reason non-empty |
| `idea-box +resurrect <slug>` | Always refuses by design; prints the "create a new idea, link via Origin" rule | n/a |
| `idea-box +regen-index` | Rebuild `INDEX.md` from `STATUS.md` + `idea.md` across all slugs; idempotent | None |
| `idea-box +decide <slug> "<note>"` | Append freeform entry to `decisions.md` without changing state | Slug exists |

Every command prints the file paths it touched, the gate checks it ran, and the resulting state. Errors include the specific missing file or failed predicate — never a generic "couldn't advance."

## Failure modes

- **Wrong cwd** — skill is invoked outside an idea-box repo. Detected by absence of `./idea_box/`. Print: `cd into your idea-box repo first (e.g., cd ~/lab/idea-box)`.
- **Slug collision** — `+new` with a slug that already exists. Refuses; suggests appending a disambiguator.
- **Missing feasibility file** — `+advance` from `explored` without `feasibility.md`. Refuses; tells the user to run `idea-feasibility` first.
- **Verdict mismatch** — `+advance` from `explored` with `feasibility.md` verdict = "Blocked". Refuses; suggests `+advance` to `blocked` instead.
- **Resurrect attempt** — always refused with the spec's stated rule.

## Anti-patterns

- Storing idea data in `tao-research-skills/` (defeats the public/private split).
- Silently auto-creating `./idea_box/` when the user is in the wrong cwd (could write to a junk location).
- Allowing `building` without an `ablations.md` (loses the "force MVP / kill-shot" discipline).
- Modifying killed ideas in place instead of forking a new slug (loses the historical signal).
- Re-implementing feasibility or ablation logic inside idea-box (it's a workflow primitive, not a replacement).

---
name: idea-feasibility
description: "Use when evaluating whether a research or product idea is actually feasible — buildable, evaluable, and de-risked by available checkpoints, code, datasets, and GPU budget. Normalizes the idea, gathers primary-source evidence (arXiv, GitHub, project pages, model hosts), scores it against four mandatory hard gates, and emits a verdict + falsifiable MVP. Triggers: \"idea feasibility\", \"is this idea feasible\", \"feasibility check\", \"can we do this\", \"is this practical\", \"worth pursuing\", \"评估想法可行性\", \"可行性分析\", \"这个 idea 能做吗\", \"这个想法靠谱吗\""
---

# Idea Feasibility

## When to Use

- The user provides a **research, paper, student-project, or product idea** and asks whether it is worth pursuing.
- The idea may come alone, or with supporting links (arXiv paper, GitHub repo, project page, HuggingFace model/checkpoint, benchmark page).
- You need a grounded judgment, not novelty mining. The output must answer:
  1. Can this be built?
  2. Can it be evaluated clearly?
  3. Is there enough public scaffolding (data, code, checkpoints) to de-risk it?
  4. What are the main blockers?
  5. What is the fastest proof-of-concept that can falsify it?

Do **not** use this skill to:
- Evaluate a single existing paper deeply — use `academic-deep-research`.
- Read a paper end-to-end — use `arxiv-latex-reader` or `pdf-reader`.
- Survey follow-up works of an already-published paper — use `followup-analysis`.

## Four mandatory feasibility gates

Every feasibility answer must explicitly address these four gates. If any is weak, the verdict must reflect that.

1. **Checkpoint availability**
2. **Code availability**
3. **Dataset availability**
4. **GPU resource realism**

## Input

Required:
- **Idea description**

Optional:
- **Links**: arXiv / GitHub / project page / HuggingFace / checkpoint URL / benchmark page
- **Constraints**: time budget, GPU budget, student level, target venue, target product, deadline
- **Goal type**: research paper, student project, engineering prototype, startup/product direction

If the user gives only idea text, do lightweight external research using the key entities in the idea.

## Source policy

Use **primary sources first**:
- arXiv / official paper page
- official GitHub repo
- official project page
- official model/checkpoint host
- benchmark or dataset official page

If a source claim cannot be verified, say so explicitly. Do **not** silently assume:
- a checkpoint exists
- code is released
- a dataset is public
- evaluation is straightforward
- a baseline is reproducible

## Workflow

### Step 1: Normalize the idea

Rewrite the user idea into a compact spec:

```text
- Problem:
- Proposed mechanism:
- Target setting:
- Expected gain:
- Required dependencies:
```

If the idea is underspecified, make the minimum necessary interpretation and label it as an assumption.

### Step 2: Collect feasibility evidence

Priority order for sources:
1. User-provided links
2. Official paper / project page
3. Official GitHub repo
4. Official checkpoint / model page
5. Dataset / benchmark page
6. Additional web search for missing pieces

For each source, extract only what matters for feasibility:
- what is already public
- what is missing
- whether inference-only or trainable artifacts exist
- whether the released code is enough to reproduce a baseline

For every idea, explicitly build this artifact table:

| Artifact | Status | Notes |
|---|---|---|
| Checkpoint | public / partial / none / unverified | model host, license, inference-only vs finetune-ready |
| Code | official / third-party / none / unverified | training code, inference code, maintenance state |
| Dataset | public / gated / private / synthetic-only / unverified | size, labels, access friction |
| GPU resources | light / moderate / heavy / prohibitive | minimum viable setup and full-run estimate |

This table is mandatory.

### Step 3: Evaluate the four hard gates first

Before any broader discussion, answer directly:

1. **Checkpoint**: Is there a usable public checkpoint, or would the user need to train from scratch?
2. **Code**: Is there official code that covers the critical path, or only fragments?
3. **Dataset**: Is there a public dataset or a credible synthetic-data route?
4. **GPU**: What is the minimum viable GPU budget, and is it realistic for the user's setting?

If the answer to any of these is "no" or "unverified", call it out as a primary blocker.

### Step 4: Score the idea across the core axes

Use a 1-5 score for each axis and explain the score in one sentence.

| Axis | What to check |
|---|---|
| Problem clarity | Is the task concrete enough to execute and measure? |
| Prior-art gap | Is there still room, or is the space already saturated? |
| Data availability | Public datasets, labels, synthetic route, privacy risk |
| Code availability | Official repo, baseline implementations, maintenance state |
| Checkpoint availability | Pretrained models, public weights, adapters, model cards |
| Compute feasibility | Minimum viable GPU/CPU budget for a useful result |
| Evaluation clarity | Benchmarks, metrics, human eval burden, ablation path |
| Integration risk | How many brittle components must work together? |
| Time-to-first-signal | How quickly can a kill-shot prototype falsify it? |

### Step 5: Produce a feasibility verdict

Map the evidence to one of:
- **High feasibility**
- **Medium feasibility**
- **Low feasibility**
- **Blocked by missing prerequisites**

Rules:
- **High**: checkpoint/code/dataset are available or replaceable, GPU budget is realistic, eval path is clear
- **Medium**: plausible, but one or two of checkpoint/code/dataset/GPU are weak
- **Low**: several of checkpoint/code/dataset/GPU are weak or missing
- **Blocked**: one of checkpoint/code/dataset/GPU is a hard prerequisite and cannot currently be verified or obtained

### Step 6: Force an MVP and a kill-shot test

Always define the smallest serious experiment that could validate or kill the idea.

Include:
- **MVP setup**
- **Required artifacts**
- **Minimum hardware**
- **Expected success signal**
- **Fast failure signal**

This is mandatory. A feasibility answer without a falsifiable MVP is incomplete.

## Checkpoint-specific guidance

When the idea depends on a foundation model, explicitly answer:

1. Is there a usable public checkpoint?
2. Is it inference-only, finetune-ready, or training-only code?
3. Is the license compatible with the user's intended use?
4. If no checkpoint exists, how much harder does that make the idea?

If no public checkpoint is found after checking the official repo/project/model host, say:

`No public checkpoint verified from the checked primary sources.`

Do not replace that with a vague statement like "probably trainable from code."

## Code-specific guidance

Explicitly answer:

1. Is there an official repo?
2. Does it include training code, inference code, or both?
3. Does the released code cover the part this idea depends on, or only a neighboring baseline?
4. If only third-party implementations exist, how much extra integration risk does that add?

If no relevant code is verified from primary sources, say:

`No relevant public code verified from the checked primary sources.`

## Dataset-specific guidance

Explicitly answer:

1. Is there a public dataset that matches the task?
2. If not, is there a realistic synthetic-data or weak-supervision route?
3. Is access gated, paid, private, or license-restricted?
4. Is the dataset large/clean enough for the claimed result?

If the idea depends on data that is not publicly available, that should materially lower feasibility unless the user explicitly has private access.

## GPU-resource guidance

Always provide both:

1. **Minimum viable setup**: the smallest hardware budget that can produce a meaningful signal
2. **Serious experiment setup**: the hardware budget for a result strong enough to trust

Use concrete estimates:

```text
Minimum viable: 1x4090 for 2 days
Serious run: 8xA100-80G for 5 days
Total serious budget: ~960 GPU-hours
```

Classify GPU burden:
- **Light**: laptop / single consumer GPU / CPU-heavy but cheap
- **Moderate**: 1-4 high-end GPUs
- **Heavy**: 8-32 datacenter GPUs
- **Prohibitive**: beyond a typical academic lab or startup prototype budget

If the idea only works with a prohibitive setup, the verdict should not be "high feasibility" unless the user explicitly has that access.

## Output format

```markdown
## Idea Feasibility

### Verdict
[High / Medium / Low / Blocked] — 2-4 sentence summary

### Normalized idea
- Problem: ...
- Mechanism: ...
- Target setting: ...
- Expected gain: ...

### Evidence
| Source | What it confirms | What it does not confirm |
|---|---|---|

### Hard gates
| Gate | Status | Why |
|---|---|---|
| Checkpoint | ... | ... |
| Code | ... | ... |
| Dataset | ... | ... |
| GPU resources | ... | ... |

### Scorecard
| Axis | Score (1-5) | Why |
|---|---:|---|

### Main blockers
- ...

### MVP
- Setup: ...
- Artifacts needed: ...
- Minimum hardware: ...
- Serious-run hardware: ...
- Success signal: ...
- Failure signal: ...

### Recommendation
- Pursue now / pursue with narrower scope / wait for missing artifact / drop
```

## Calibration rules

- **Student-project** idea: penalize high compute and vague evaluation more aggressively.
- **Product** idea: penalize licensing, data moat, latency/deployment risk more aggressively.
- **Paper** idea: penalize saturated prior art and weak differentiation more aggressively.
- **Quick take**: still verify the most failure-prone claims — public code, checkpoint, dataset, benchmark.

## Style

- Match the user's language. If ambiguous, prefer Chinese.
- Be direct. The useful answer is often "this is only feasible if X exists, and I could not verify X."
- Separate **verified facts** from **your inference**.
- Keep sources linked in the response when external browsing is used.

## Anti-Patterns

- **Silent assumptions** about checkpoint, code, or dataset existence — say "unverified" instead of pretending.
- **Novelty-mining** that ignores buildability. An idea can be novel and infeasible.
- **High-feasibility verdict** when GPU burden is prohibitive without the user confirming access.
- **No MVP**. A feasibility judgment without a falsifiable smallest-experiment is incomplete.
- **Burying the hard gates** under prose. Surface checkpoint/code/dataset/GPU status in a table, not paragraph form.
- **Hand-waving compute** ("should be cheap") instead of concrete GPU-hour estimates.

## See Also

- `academic-deep-research` — Evaluate a single existing paper (venue, citations, reproducibility) rather than a forward-looking idea
- `followup-analysis` — Forward-citation cone for a published paper; pair with this skill when the idea builds on a known seed
- `idea-explore` — Upstream producer that proposes candidate ideas anchored to a seed paper + GitHub issues + adjacent literature; feed its top-ranked idea into this skill's feasibility check
- `arxiv-latex-reader` — Deep-read the papers surfaced as evidence during Step 2
- `pdf-reader` — Same, for non-arXiv PDFs
- `github-reader` — Verify code/checkpoint claims by reading the official repo
- `ml-ablation-design` — Once the idea is judged feasible, design the actual ablation matrix
- `idea-box` — Per-idea lifecycle that consumes this skill's output as the `explored → feasible/blocked` gate. When invoked from inside `./idea_box/<slug>/`, write the verdict to `./idea_box/<slug>/feasibility.md`
- `fail-fast-ml-engineering` — Shares the "no silent fallbacks / state unverified claims explicitly" stance this skill enforces on evidence

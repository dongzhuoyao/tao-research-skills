---
name: consolidate-skill
description: "Use when turning recent agent-session evidence into durable skill updates. Triggers: \"consolidate lessons\", \"update skills from sessions\", \"prevent repeated fixes\", \"recurring mistakes\", \"capture workflow learnings\""
---

# Consolidate Skill

Turn recent session evidence into small, durable updates to existing skills — or, when clearly needed, a new skill. The goal is to prevent the same avoidable fix from recurring.

## When to Use

- The user asks to consolidate lessons from recent sessions.
- The same fix, correction, or clarification keeps coming up.
- An agent workflow or recurring mistake should become a reusable instruction.
- A recent session contains a durable operational rule that no skill currently owns.

## Workflow

0. **Propose before editing.**
   - Stop after the evidence pass and propose changes before editing any skill or instruction file.
   - Report the evidence, durable lesson, target file, intended patch shape, and validation plan.
   - Ask for explicit confirmation before patching.
   - Patch only after the user confirms with wording such as `yes`, `apply`, `implement`, or `confirm`.

1. **Locate recent session evidence.**
   - Run `scripts/session_scan.py --days 14 --limit 30` from this skill folder.
   - The scanner searches Codex, Kimi/Kimi-Code, Claude, Cline, and Hermes session roots by default. Use `--root <path>` for any additional agent or archived location.
   - For Kimi Code CLI, sessions live under `~/.kimi-code/sessions/<workspace_dir_hash>/session_<uuid>/agents/<agent>/wire.jsonl`. To find the current session without the scanner, match the workspace directory name to the repo slug, then pick the most recent `session_<uuid>` directory. See [`references/session-sources.md`](references/session-sources.md) for the full layout.
   - For broad consolidation requests, narrow the scan with `--contains` terms such as the repo name, related skill name, `update the skill`, `same issue`, `fix again`, or `add to memory`.
   - Prefer the script's ranked results over rewriting ad hoc scanners.
   - If `scripts/session_scan.py` is missing or fails, stop and report the exact cause. Do not substitute an ad-hoc `find` or `rg` scan.
   - If the user names a specific session, log file, branch, workspace, or time window, use that scope instead.
   - If session files cannot be found or read, stop and report the exact path and cause. Do not infer lessons from memory alone.

2. **Select sessions to inspect.**
   - Prefer sessions involving the current repo, the same workspace, or explicit repeated fixes from the user.
   - Include the current conversation context when available.
   - Read enough raw evidence to identify the user request, the agent action, the failure or correction, and the successful fix.
   - Ignore injected `AGENTS.md`/`CLAUDE.md` instruction blocks, developer messages, duplicated event/message copies of the same user text, and pasted skill bodies unless the user text around them states a concrete requested lesson.
   - For subagent or forward-test sessions, require a final report or a parent session summary before treating the test as pass/fail evidence. Progress updates alone are evidence only that the test was attempted.
   - Do not commit raw transcripts, private prompts, secrets, or long excerpts.

3. **Interrogate session evidence with a subagent.**
   - After selecting sessions, dispatch an evidence-interrogating subagent.
   - Give it:
     - the selected session snippets, with private values redacted;
     - the paths of the current skill files most likely related to the lesson;
     - the focused search terms used to select the sessions;
     - and this instruction: "Read the session evidence and the related skill files, then ask probing questions to extract durable lessons. Return a structured report with: (1) user request / agent action / failure / successful fix, (2) probing questions you asked and the evidence-based answers, (3) durable lessons likely to recur, (4) which current skill files already cover each lesson and which do not, (5) recommended target skill and patch shape, and (6) lessons that should be ignored as one-off or duplicates."
   - The main agent must still read the relevant skill files itself and make the final consolidation decision. Do not treat the subagent report as authoritative without verifying it against the raw session evidence and the current skill text.
   - Use the subagent output as structured input for step 4, not as a substitute for step 4.

4. **Extract only durable lessons.**
   - Keep a lesson only when it is supported by evidence from at least one session and is likely to recur.
   - Record the trigger, the wrong behavior, the correct behavior, and the target skill or instruction file.
   - Prefer specific operational rules over broad advice.
   - Ignore one-off accidents, stale environment problems, or lessons that would duplicate an existing instruction.

5. **Propose the target asset.**
   - Propose updating a skill's `SKILL.md` when the lesson changes how an agent should perform a reusable workflow.
   - Propose updating a skill `references/` file only when the detail is too large or too conditional for the main `SKILL.md`.
   - Propose creating a new skill only when no existing skill owns the workflow.
   - Default target is the skill that owns the affected workflow, **not** `consolidate-skill` itself. Only target `consolidate-skill` when the lesson is specifically about consolidation mechanics (e.g., session scanning, evidence filtering, proposal/confirmation behavior, privacy, or validation).
   - **When to promote a skill to computer-level.** If the consolidated skill is reusable across projects on this machine (not project-specific), place the canonical copy in `tao-research-skills/skills/infra/<skill>/`, run `meta-init` to symlink it into `~/.agents/skills/` and `~/.claude/skills/`, remove any project-local `.codex/skills/<skill>/` or `.claude/skills/<skill>/` copy, and reference the canonical path in project memory. Do not keep a project-local copy of a cross-project skill; that causes drift and makes updates scattered.
   - In Codex-first repos, treat `.codex/` as canonical. Do not update `.claude/` unless the user explicitly asks for a Claude mirror.

6. **Patch narrowly after confirmation.**
   - Do not edit until the user explicitly confirms the proposal.
   - Write instructions in imperative form.
   - Add the minimum durable rule needed to prevent the repeated fix.
   - Include precise triggers, required checks, and failure behavior.
   - Do not preserve legacy aliases or fallback behavior unless the user asks for compatibility or an external dependency requires it.
   - Follow the project rule: no silent fallbacks. When a required check, tool call, API request, or validation step fails, stop and surface the specific cause.

7. **Validate.**
   - If a platform-specific skill validator exists, run it on every skill folder you changed.
   - If the current repo's Codex assets changed and `scripts/sync_codex_assets.py` exists, run `uv run python scripts/sync_codex_assets.py --check`.
   - Run `git diff --check`.
   - If any validation fails, stop and report the exact failure. Do not claim consolidation is complete.

## Session Sources

See [`references/session-sources.md`](references/session-sources.md) for where to look when session logs are missing, the default scan returns too many candidates, or the task needs a narrower search strategy.

## Output

Report:

- sessions or artifacts inspected
- durable lessons found
- skills or instruction files changed
- validation commands run and their results
- lessons deliberately not consolidated, with the reason

## Anti-Patterns

- Consolidating from memory instead of reading raw session evidence.
- Treating injected instruction blocks or duplicated user text as lesson evidence.
- Patching a skill before the user confirms the proposal.
- Adding broad advice instead of a specific trigger/action rule.
- Duplicating a lesson that already exists in another skill.
- Committing raw transcripts, secrets, or long excerpts.

## See Also

- `meta-init` — install tao-research-skills globally so this skill is available in every project
- `memory-sync` — keep canonical memory in sync with generated `AGENTS.md`/`CLAUDE.md` wrappers
- `agents-md-writing` — write effective agent instruction files

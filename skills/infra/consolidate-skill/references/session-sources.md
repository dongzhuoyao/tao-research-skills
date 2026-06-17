# Session Sources

Use these sources only to locate evidence. Do not paste raw transcripts into
tracked files.

## Primary Agent Locations

`scripts/session_scan.py` searches these roots by default:

- **Codex**: `~/.codex/sessions/YYYY/MM/DD/*.jsonl`, `~/.codex/archived_sessions/*.jsonl`
- **Kimi / Kimi Code**: `~/.kimi-code/sessions/<workspace_dir_hash>/session_<uuid>/agents/<agent>/wire.jsonl`, `~/.kimi/sessions/<user_hash>/<session_uuid>/wire.jsonl`
- **Claude Code**: `~/.claude/projects/*/*.jsonl`, `~/.claude/sessions/*/*.jsonl`
- **Cline**: `~/.cline/data/sessions/*/*.jsonl`
- **Hermes**: `~/.hermes/sessions/*.jsonl`

Use `scripts/session_scan.py` first. If the script is missing or fails, stop and
report the exact cause; do not silently replace it with an ad-hoc `find`, `ls`,
or `rg` scan.

### Locating the current Kimi Code CLI session

Kimi Code CLI stores sessions under `~/.kimi-code/sessions/<workspace_dir_hash>/`.
The `workspace_dir_hash` directory name embeds the project slug and a directory
hash, for example `wd_deepresearch_2afa7f5f0346`. Inside each workspace directory:

```text
session_<uuid>/
  agents/
    main/wire.jsonl          # primary agent turn stream
    agent-0/wire.jsonl       # subagent turn stream (if any)
    ...
```

To find the current session without the scanner:

1. List `~/.kimi-code/sessions/` and match the directory name to the current
   repo or workspace slug.
2. Under that workspace directory, pick the most recently modified
   `session_<uuid>/` directory.
3. Read `agents/main/wire.jsonl` for the main conversation; inspect
   `agents/agent-*/wire.jsonl` for subagent or delegated work.

Because `scripts/session_scan.py` searches roots recursively, it finds Kimi
sessions regardless of this nested layout.

For broad searches, rank likely evidence directly:

```bash
scripts/session_scan.py --days 14 --limit 20 \
  --contains "deepresearch" \
  --contains "update the skill" \
  --contains "same issue" \
  --contains "fix again"
```

## Narrowing The Search

Prefer evidence in this order:

1. Sessions whose raw text contains the current repo path or workspace name.
2. Sessions from the user-provided date range or the most recent 14 days.
3. Sessions containing phrases like `fix again`, `same issue`, `remember`,
   `skill`, `AGENTS.md`, `CLAUDE.md`, `silent fallback`, `preflight`,
   `dryrun`, `W&B`, or the name of the related skill.
4. Current conversation context, when the latest user request itself states the
   lesson clearly.

## Ignore Boilerplate Matches

Recent sessions often include large injected user-message blocks containing
project `AGENTS.md` instructions, skill bodies, or duplicated event/message
copies. Do not treat those as lesson evidence by themselves.

Keep a hit only when the surrounding task text shows an actual request,
correction, failure, or successful fix. If the same user text appears twice in
one JSONL file, count it once.

## Reading JSONL

Session files are JSONL and may contain different event shapes. Use structured
tools when possible:

```bash
jq -r 'select(type=="object") | [.type, .role, .cwd] | @tsv' path/to/session.jsonl
```

For text search, prefer `rg`:

```bash
rg -n "same issue|silent fallback|preflight|skill|AGENTS.md" ~/.codex/sessions
```

If `jq` is unavailable, use plain `sed` or `rg`; report that `jq` was missing
instead of pretending structured parsing was completed.

## Privacy And Evidence Hygiene

- Treat session logs as private local artifacts.
- Do not commit transcripts, secrets, raw API keys, hostnames, account IDs, or
  copied verification logs.
- Summarize evidence as operational facts, not quotes, unless a short quote is
  necessary to identify the trigger.
- If a lesson depends on private backend values, encode the rule generically and
  point agents to ignored env/runtime files for concrete values.

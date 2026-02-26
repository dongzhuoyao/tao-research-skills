# Tao Research Skills

Reusable AI agent skills for ML research workflows, following the [Open Agent Skills](https://agentskills.io) standard.

**Public repo — never commit private information (paths, usernames, API keys, internal URLs).**

## Structure

```
<skill-name>/
  SKILL.md              # Frontmatter + patterns + anti-patterns
  references/           # (optional) Deep-dive docs for progressive disclosure
```

- Each skill is a self-contained directory with a `SKILL.md`
- No runnable code — documentation and knowledge only
- Installed as a git submodule into other projects

## Conventions

### Frontmatter

Every `SKILL.md` starts with exactly two fields:

```yaml
---
name: <skill-name>
description: Use when <scenario>. <Scope summary>. Triggers: "keyword1", "keyword2"
---
```

- `name` matches directory name exactly (kebab-case)
- `description` is a single line — opens with `Use when ...`, ends with a `Triggers:` list
- Triggers are comma-separated, each in double quotes

### Sections

1. `# Title` — human-readable name
2. `## When to Use` — always first, bullet list
3. Middle sections — varies per skill (Core Principles, Quick Reference, Patterns, etc.)
4. `## Anti-Patterns` — near the end
5. `## See Also` — always last

### See Also

Backtick style only:

```markdown
- `<skill-name>` — Brief description of relevance
```

Do **not** use markdown link style (`[name](../path/)`).

### References

- Plain markdown, no YAML frontmatter
- One focused sub-topic per file
- Linked from `SKILL.md` with relative paths

## Adding a Skill

1. Create `<skill-name>/SKILL.md` with proper frontmatter
2. Add `## When to Use`, patterns, `## Anti-Patterns`, `## See Also`
3. Update `README.md` in **three places**:
   - Badge count (`skills-N-blue`)
   - One-prompt install block (flat bullet list)
   - Available Skills table (under the correct category)
4. Cross-reference from related skills' See Also sections

## Known Issues

- `vercel-cost-optimization` uses markdown link style in See Also instead of backtick style

## Commits

Conventional commits, scoped to skill name:

```
feat: add <skill-name> skill
fix(<skill-name>): correct module load example
docs: update README with new skill count
```

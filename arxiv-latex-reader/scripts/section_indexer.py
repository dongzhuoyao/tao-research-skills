"""Build section_index.json with LLM-generated summaries.

Short sections (<100 chars): use full text as summary (no truncation).
Medium sections (<32k chars): single LLM call → 2-3 bullet summary.
Long sections (≥32k chars): chunk-summarize → merge.
Cached: only builds once per paper unless --force.

CRITICAL: Never truncate. Never use [:N] on text. For long sections,
use chunk-summarization (split → summarize each → merge).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

CHUNK_THRESHOLD = 32_000  # chars before chunking
CHUNK_SIZE = 24_000       # target chunk size

SUMMARY_SYSTEM = """\
You are a research paper analyst. Summarize the given section in 2-3 concise \
bullet points (each ≤80 chars). Focus on key claims, methods, or results. \
Return ONLY bullet points starting with •, no other text."""

MERGE_SYSTEM = """\
You are a research paper analyst. Given multiple chunk summaries from the same \
section, merge them into 2-3 concise bullet points (each ≤80 chars). \
Remove redundancy. Return ONLY bullet points starting with •."""


def _summarize_text(client: anthropic.Anthropic, text: str, model: str) -> str:
    """Summarize a single text block via LLM."""
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text.strip()


def _chunk_summarize(client: anthropic.Anthropic, text: str, model: str) -> str:
    """Split long text into chunks, summarize each, merge."""
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE):
        chunks.append(text[i:i + CHUNK_SIZE])

    chunk_summaries = []
    for chunk in chunks:
        summary = _summarize_text(client, chunk, model)
        chunk_summaries.append(summary)

    merged_input = "Chunk summaries to merge:\n\n" + "\n\n".join(chunk_summaries)
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=MERGE_SYSTEM,
        messages=[{"role": "user", "content": merged_input}],
    )
    return resp.content[0].text.strip()


def build_section_index(
    paper_id: str,
    workspace_dir: str | Path = "workspace",
    force: bool = False,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Build or load cached section_index.json.

    Returns dict with "sections" list, each having name/summary/char_count.
    """
    workspace = Path(workspace_dir) / paper_id
    index_path = workspace / "section_index.json"
    paper_path = workspace / "paper_content.json"

    # Return cached if exists and not forcing rebuild
    if index_path.exists() and not force:
        with open(index_path) as f:
            return json.load(f)

    if not paper_path.exists():
        raise FileNotFoundError(f"Run 'poster-agent parse' first: {paper_path}")

    with open(paper_path) as f:
        paper = json.load(f)

    client = anthropic.Anthropic()
    sections_index = []

    for sec in paper.get("sections", []):
        name = sec["name"]
        text = sec["text"]
        char_count = len(text)

        print(f"  Indexing § {name} ({char_count} chars)...", end=" ", flush=True)

        if char_count < 100:
            # Too short to summarize — use full text as summary
            summary = f"• {text}"
        elif char_count >= CHUNK_THRESHOLD:
            summary = _chunk_summarize(client, text, model)
        else:
            summary = _summarize_text(client, text, model)

        print("done")
        sections_index.append({
            "name": name,
            "summary": summary,
            "char_count": char_count,
            "level": sec.get("level", 1),
        })

    index = {"sections": sections_index}
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index


class SectionIndexer:
    """Wraps build_section_index for OOP usage."""

    def __init__(self, paper_id: str, workspace_dir: str | Path = "workspace",
                 model: str = "claude-sonnet-4-20250514"):
        self.paper_id = paper_id
        self.workspace_dir = Path(workspace_dir)
        self.model = model

    def build(self, force: bool = False) -> dict:
        return build_section_index(
            self.paper_id,
            workspace_dir=self.workspace_dir,
            force=force,
            model=self.model,
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python section_indexer.py <paper_id> [--force] [--workspace DIR]")
        sys.exit(1)

    paper_id = args[0]
    force = "--force" in args
    ws = "workspace"
    if "--workspace" in args:
        ws = args[args.index("--workspace") + 1]

    index = build_section_index(paper_id, workspace_dir=ws, force=force)
    print(f"\nSection index ({len(index['sections'])} sections):")
    for sec in index["sections"]:
        print(f"  § {sec['name']:30s} {sec['char_count']:>6d} chars")
        for line in sec["summary"].split("\n"):
            if line.strip():
                print(f"    {line.strip()}")

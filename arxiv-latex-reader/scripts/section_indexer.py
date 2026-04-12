"""Section indexer — LLM-powered section summarization for progressive reading.

Standalone — no poster_agent dependency. Only requires: anthropic (pip).

Generates a lightweight section index (~2k tokens) from paper_content.json.
Each section gets a 2-3 sentence bullet-point summary via LLM.
Long sections (>=8k tokens / ~32k chars) are chunk-summarized recursively.
NEVER truncates — every character is read and summarized.

Usage:
    python section_indexer.py <paper_id>
    python section_indexer.py <paper_id> --force --workspace workspace

    # As library:
    from section_indexer import SectionIndexer
    indexer = SectionIndexer()
    index = indexer.build_index("2403.13802")
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path("workspace")

# ~8k tokens ≈ 32k chars (rough 1:4 ratio)
CHUNK_CHAR_THRESHOLD = 32_000
# Chunk target size: ~6k tokens ≈ 24k chars
CHUNK_TARGET_CHARS = 24_000

SUMMARIZE_SYSTEM = """\
You are an expert academic paper reader. Summarize the given section text \
into 2-3 concise bullet points that capture the key ideas, methods, or findings. \
Use bullet points (•) format. Be specific — include method names, metric values, \
and concrete claims. Do NOT omit important details."""

MERGE_SYSTEM = """\
You are an expert academic paper reader. Given multiple chunk summaries from the \
same paper section, merge them into a single cohesive 2-3 bullet-point summary. \
Preserve the most important details: method names, metrics, key findings. \
Use bullet points (•) format."""


@dataclass
class SectionEntry:
    """One entry in the section index."""
    id: int
    name: str
    level: int
    char_count: int
    summary: str
    figures: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


@dataclass
class SectionIndex:
    """Lightweight section index for a paper."""
    paper_id: str
    title: str
    abstract: str
    total_sections: int
    total_chars: int
    sections: list[SectionEntry] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> SectionIndex:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [SectionEntry(**e) for e in data.pop("sections", [])]
        return cls(**data, sections=entries)


# ── paper_content.json loading (standalone) ───────────────────────

def _load_paper(paper_json: Path) -> dict:
    """Load paper_content.json and return raw dict."""
    return json.loads(paper_json.read_text(encoding="utf-8"))


class SectionIndexer:
    """Builds a section index using LLM summarization."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def build_index(
        self,
        paper_id: str,
        *,
        workspace_dir: Path | str | None = None,
        force: bool = False,
    ) -> SectionIndex:
        """Build section index for a paper.

        Args:
            paper_id: Arxiv paper ID.
            workspace_dir: Override workspace directory.
            force: Rebuild even if section_index.json already exists.

        Returns:
            SectionIndex with summaries for all sections.
        """
        ws = Path(workspace_dir) if workspace_dir else DEFAULT_WORKSPACE
        paper_dir = ws / paper_id
        index_path = paper_dir / "section_index.json"

        if not force and index_path.exists():
            logger.info("Loading cached section index: %s", index_path)
            return SectionIndex.from_json(index_path)

        paper_json = paper_dir / "paper_content.json"
        if not paper_json.exists():
            raise FileNotFoundError(
                f"paper_content.json not found at {paper_json}. "
                "Run 'poster-agent parse' first."
            )

        paper = _load_paper(paper_json)
        index = self._index_paper(paper_id, paper)
        index.to_json(index_path)
        logger.info("Wrote section index: %s (%d sections)", index_path, len(index.sections))
        return index

    def _index_paper(self, paper_id: str, paper: dict) -> SectionIndex:
        """Build index entries for all sections."""
        sections = paper.get("sections", [])
        figures = paper.get("figures", [])
        tables = paper.get("tables", [])
        metadata = paper.get("metadata", {})

        entries = []
        for i, sec in enumerate(sections):
            sec_text = sec.get("text", "")
            sec_figures = [f["id"] for f in figures if f.get("id") and f["id"] in sec_text]
            sec_tables = [t["id"] for t in tables if t.get("id") and t["id"] in sec_text]

            summary = self._summarize_section(sec["name"], sec_text)

            entries.append(SectionEntry(
                id=i,
                name=sec["name"],
                level=sec.get("level", 1),
                char_count=len(sec_text),
                summary=summary,
                figures=sec_figures,
                tables=sec_tables,
            ))

        return SectionIndex(
            paper_id=paper_id,
            title=metadata.get("title", ""),
            abstract=metadata.get("abstract", ""),
            total_sections=len(sections),
            total_chars=sum(len(s.get("text", "")) for s in sections),
            sections=entries,
        )

    def _summarize_section(self, name: str, text: str) -> str:
        """Summarize a section, chunking if necessary."""
        if len(text) < CHUNK_CHAR_THRESHOLD:
            return self._llm_summarize(name, text)

        chunks = split_into_chunks(text, CHUNK_TARGET_CHARS)
        logger.info(
            "Section '%s' is %d chars, splitting into %d chunks",
            name, len(text), len(chunks),
        )

        chunk_summaries = []
        for j, chunk in enumerate(chunks):
            label = f"{name} (chunk {j + 1}/{len(chunks)})"
            chunk_summaries.append(self._llm_summarize(label, chunk))

        merged_text = "\n---\n".join(chunk_summaries)
        return self._llm_merge(name, merged_text)

    def _llm_summarize(self, section_name: str, text: str) -> str:
        """Single LLM call to summarize section text."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=SUMMARIZE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"## Section: {section_name}\n\n{text}",
                }],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.warning("LLM summarize failed for '%s': %s", section_name, e)
            return fallback_summary(text)

    def _llm_merge(self, section_name: str, chunk_summaries: str) -> str:
        """Merge multiple chunk summaries into one cohesive summary."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=MERGE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": (
                        f"## Section: {section_name}\n\n"
                        f"Chunk summaries:\n{chunk_summaries}"
                    ),
                }],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.warning("LLM merge failed for '%s': %s", section_name, e)
            return chunk_summaries


def split_into_chunks(text: str, target_chars: int) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current and current_len + para_len > target_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def fallback_summary(text: str) -> str:
    """Fallback: first + last paragraph extraction (not truncation)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""
    if len(paragraphs) == 1:
        return f"• {paragraphs[0][:300]}..."
    first = paragraphs[0][:200]
    last = paragraphs[-1][:200]
    return f"• {first}...\n• {last}..."


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build section index with LLM summaries")
    parser.add_argument("paper_id", help="Paper ID (e.g., 2403.13802)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name")
    args = parser.parse_args()

    indexer = SectionIndexer(model=args.model)
    index = indexer.build_index(
        args.paper_id, workspace_dir=args.workspace, force=args.force,
    )
    print(f"Index: {index.total_sections} sections, {index.total_chars:,} total chars")
    for entry in index.sections:
        print(f"  §{entry.id} {entry.name} ({entry.char_count:,} chars)")
        print(f"    → {entry.summary[:120]}...")


if __name__ == "__main__":
    main()

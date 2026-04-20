"""Paper navigator — orchestrates section indexer + reader for progressive reading.

Three modes:
  overview    — ~2k token section index (always in context)
  read_full   — full text of specific sections (no truncation, via section_reader)
  read_for_poster — LLM summaries of poster-relevant sections (from section_index.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from scripts.section_reader import SectionReader
except ImportError:
    from section_reader import SectionReader

# Sections to skip when reading for poster extraction
_SKIP_PATTERNS = [
    "related work", "prior work", "background",
    "acknowledgment", "acknowledgement",
    "appendix", "supplementary", "supplemental",
    "references", "bibliography",
]


def _should_skip(name: str) -> bool:
    """Check if a section should be skipped for poster reading."""
    lower = name.lower().strip()
    return any(pat in lower for pat in _SKIP_PATTERNS)


class PaperNavigator:
    """Orchestrates progressive paper reading."""

    def __init__(self, paper_id: str, workspace_dir: str | Path = "workspace"):
        self.paper_id = paper_id
        self.workspace = Path(workspace_dir) / paper_id
        self._paper_path = self.workspace / "paper_content.json"
        self._index_path = self.workspace / "section_index.json"
        self._reader = SectionReader(self._paper_path)
        self._index = self._load_index()
        self._paper_data = self._load_paper()

    def _load_index(self) -> dict:
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"Run section_indexer first: {self._index_path}\n"
                f"  python scripts/section_indexer.py {self.paper_id}"
            )
        with open(self._index_path) as f:
            return json.load(f)

    def _load_paper(self) -> dict:
        with open(self._paper_path) as f:
            return json.load(f)

    def get_overview(self) -> str:
        """Return compact section index (~2k tokens). Keep in context."""
        lines = [f"# {self._paper_data['metadata']['title']}", ""]
        for sec in self._index["sections"]:
            lines.append(f"§ {sec['name']:30s}  {sec['char_count']:>6,} chars")
            for bullet in sec["summary"].split("\n"):
                b = bullet.strip()
                if b:
                    lines.append(f"  {b}")
            lines.append("")
        return "\n".join(lines)

    def read_full(self, section_names: list[str]) -> list[dict]:
        """Read complete text of named sections. No truncation."""
        return self._reader.read(section_names)

    def read_by_keyword(self, keyword: str) -> list[dict]:
        """Fuzzy search sections by keyword."""
        return self._reader.read_by_keyword(keyword)

    def read_for_poster(self) -> dict:
        """Smart read: skip non-poster sections, return LLM summaries.

        Returns summaries from section_index.json (not raw text).
        LLM summaries are higher quality than truncated text because the
        indexer reads the ENTIRE section and distills key claims/methods/results.

        Returns:
            {
                "overview": str,           # compact index (~2k tokens)
                "sections": [dict],        # LLM summaries of poster-relevant sections
                "skipped": [str],          # names of skipped sections
                "figures": [dict],         # figure metadata from paper
                "tables": [dict],          # table metadata from paper
            }
        """
        skipped = []
        sections = []

        for sec in self._index["sections"]:
            name = sec["name"]
            if _should_skip(name):
                skipped.append(name)
            else:
                # Return summary from index, not raw text
                sections.append({
                    "name": name,
                    "summary": sec["summary"],
                    "char_count": sec["char_count"],
                    "level": sec.get("level", 1),
                })

        return {
            "overview": self.get_overview(),
            "sections": sections,
            "skipped": skipped,
            "figures": self._paper_data.get("figures", []),
            "tables": self._paper_data.get("tables", []),
        }


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python paper_navigator.py <paper_id> <command> [args...]")
        print("Commands: overview, poster, read <section_names...>")
        sys.exit(1)

    paper_id, command = args[0], args[1]
    ws = "workspace"
    if "--workspace" in args:
        ws = args[args.index("--workspace") + 1]

    nav = PaperNavigator(paper_id, workspace_dir=ws)

    if command == "overview":
        print(nav.get_overview())
    elif command == "poster":
        result = nav.read_for_poster()
        print(result["overview"])
        print(f"\n--- {len(result['sections'])} sections (skipped: {result['skipped']}) ---\n")
        for sec in result["sections"]:
            print(f"\n§ {sec['name']} ({sec['char_count']} chars)")
            print(sec["summary"])
    elif command == "read":
        names = args[2:]
        for sec in nav.read_full(names):
            print(f"\n§ {sec['name']} ({sec['char_count']} chars)")
            print(sec["text"])

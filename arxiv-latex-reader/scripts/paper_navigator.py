"""Paper navigator — orchestrator for progressive paper reading.

Standalone — no poster_agent dependency. Imports only sibling scripts.

Keeps the lightweight section index in context (~2k tokens) and dispatches
on-demand full section reads only when needed.

Usage:
    python paper_navigator.py <paper_id> overview
    python paper_navigator.py <paper_id> poster
    python paper_navigator.py <paper_id> read Method Experiments

    # As library:
    from paper_navigator import PaperNavigator
    nav = PaperNavigator("2403.13802")
    overview = nav.get_overview()
    context = nav.read_for_poster()
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from section_indexer import SectionIndex, SectionIndexer
from section_reader import read_sections

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path("workspace")

# Sections typically skipped for poster extraction
_SKIP_FOR_POSTER = {"related work", "acknowledgment", "acknowledgments", "acknowledgements", "appendix"}

# Sections always included for poster extraction
_ALWAYS_FOR_POSTER = {"method", "approach", "model", "experiment", "result", "conclusion", "discussion"}


class PaperNavigator:
    """Orchestrator: index-aware progressive paper reading."""

    def __init__(
        self,
        paper_id: str,
        *,
        workspace_dir: Path | str | None = None,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.paper_id = paper_id
        self.workspace_dir = Path(workspace_dir) if workspace_dir else DEFAULT_WORKSPACE
        self._api_key = api_key
        self._model = model
        self._index: SectionIndex | None = None

    @property
    def index(self) -> SectionIndex:
        """Lazy-load or build the section index."""
        if self._index is None:
            index_path = self.workspace_dir / self.paper_id / "section_index.json"
            if index_path.exists():
                self._index = SectionIndex.from_json(index_path)
            else:
                indexer = SectionIndexer(api_key=self._api_key, model=self._model)
                self._index = indexer.build_index(
                    self.paper_id, workspace_dir=self.workspace_dir,
                )
        return self._index

    def get_overview(self) -> str:
        """Return a formatted overview of the paper structure.

        This is the ~2k token lightweight view that stays in context.
        """
        idx = self.index
        lines = [
            f"# {idx.title}",
            f"**Sections:** {idx.total_sections}  |  **Total chars:** {idx.total_chars:,}",
            f"**Abstract:** {idx.abstract[:300]}{'...' if len(idx.abstract) > 300 else ''}",
            "",
        ]

        for entry in idx.sections:
            figs = f"  Figures: {', '.join(entry.figures)}" if entry.figures else ""
            tabs = f"  Tables: {', '.join(entry.tables)}" if entry.tables else ""
            lines.append(f"§{entry.id} {entry.name} ({entry.char_count:,} chars)")
            lines.append(f"  → {entry.summary}")
            if figs:
                lines.append(figs)
            if tabs:
                lines.append(tabs)
            lines.append("")

        return "\n".join(lines)

    def read_full(self, sections: list[int] | list[str] | str) -> list[dict]:
        """Read full text of specified sections.

        Delegates to section_reader. No limit on count — caller manages budget.
        """
        return read_sections(
            self.paper_id, sections, workspace_dir=self.workspace_dir,
        )

    def read_for_poster(self) -> dict:
        """Smart read for poster extraction.

        Returns a dict with:
            - overview: formatted index overview (always included)
            - sections: list of full section dicts for poster-relevant sections
            - skipped: list of section names that were skipped

        Skips: Related Work, Acknowledgments, Appendix.
        Includes: Method, Experiments/Results, Conclusion, Introduction.
        """
        idx = self.index
        needed_indices = []
        skipped_names = []

        for entry in idx.sections:
            name_lower = entry.name.lower()

            if any(skip in name_lower for skip in _SKIP_FOR_POSTER):
                skipped_names.append(entry.name)
                continue

            if any(key in name_lower for key in _ALWAYS_FOR_POSTER):
                needed_indices.append(entry.id)
                continue

            if "introduction" in name_lower:
                needed_indices.append(entry.id)
                continue

            if entry.char_count < 15_000:
                needed_indices.append(entry.id)
            else:
                skipped_names.append(entry.name)

        full_sections = read_sections(
            self.paper_id, needed_indices, workspace_dir=self.workspace_dir,
        )

        total_chars = sum(len(s["text"]) for s in full_sections)
        logger.info(
            "read_for_poster(%s): %d sections (%d chars), skipped %s",
            self.paper_id, len(full_sections), total_chars, skipped_names,
        )

        return {
            "overview": self.get_overview(),
            "sections": full_sections,
            "skipped": skipped_names,
        }

    def read_by_keyword(self, keyword: str) -> list[dict]:
        """Read sections matching a keyword (fuzzy name match)."""
        return read_sections(
            self.paper_id, keyword, workspace_dir=self.workspace_dir,
        )


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Navigate paper sections progressively")
    parser.add_argument("paper_id", help="Paper ID")
    parser.add_argument("action", choices=["overview", "poster", "read"],
                        help="Action: overview, poster, or read")
    parser.add_argument("sections", nargs="*", help="Section names (for 'read' action)")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    args = parser.parse_args()

    nav = PaperNavigator(args.paper_id, workspace_dir=args.workspace)

    if args.action == "overview":
        print(nav.get_overview())
    elif args.action == "poster":
        result = nav.read_for_poster()
        print(result["overview"])
        print(f"\n--- Deep-read {len(result['sections'])} sections ---")
        for sec in result["sections"]:
            print(f"\n§{sec['index']} {sec['name']} ({len(sec['text']):,} chars)")
        if result["skipped"]:
            print(f"\nSkipped: {', '.join(result['skipped'])}")
    elif args.action == "read":
        if not args.sections:
            print("Specify section names for 'read' action", file=sys.stderr)
            sys.exit(1)
        results = nav.read_full(args.sections)
        for sec in results:
            print(f"\n{'=' * 60}")
            print(f"§{sec['index']} {sec['name']} ({len(sec['text']):,} chars)")
            print("=" * 60)
            print(sec["text"])


if __name__ == "__main__":
    main()

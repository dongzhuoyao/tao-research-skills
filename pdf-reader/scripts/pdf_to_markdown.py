"""Convert a PDF to the workspace layout consumed by arxiv-latex-reader.

Produces:
    workspace/<paper_id>/paper_content.json   # sections + text, schema below
    workspace/<paper_id>/paper.md             # full markdown
    workspace/<paper_id>/figures/fig_<n>.png  # extracted figures
    workspace/<paper_id>/figures/fig_<n>.txt  # caption + class sidecar
    workspace/<paper_id>/tables/tab_<n>.md    # extracted markdown tables
    workspace/<paper_id>/conversion.log       # tool used, counts, warnings

paper_content.json schema:
    {
      "paper_id": "...",
      "source": "pdf",
      "title": "...",
      "sections": [{"name", "text", "level", "figures": [...], "tables": [...]}],
      "figures":  [{"id", "path", "caption", "page", "class"}],
      "tables":   [{"id", "path", "caption", "page"}]
    }

Tool chain (first available wins):
    1. marker-pdf       — text + figures + tables + equations, highest quality
    2. pymupdf4llm      — fast, text + basic tables, no figure extraction
    3. pdftotext+pdfimages — last-resort, poppler only

CRITICAL: never truncate. If a page fails under --strict, raise rather than skip.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse


# --------------------------------------------------------------------------- #
# Dataclasses mirroring the paper_content.json schema
# --------------------------------------------------------------------------- #


@dataclass
class Figure:
    id: str
    path: str
    caption: str
    page: int
    cls: str = "supporting"  # key | supporting | decorative


@dataclass
class Table:
    id: str
    path: str
    caption: str
    page: int


@dataclass
class Section:
    name: str
    text: str
    level: int = 1
    figures: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


@dataclass
class PaperContent:
    paper_id: str
    source: str
    title: str
    sections: list[Section]
    figures: list[Figure]
    tables: list[Table]
    authors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Tool detection
# --------------------------------------------------------------------------- #


def _tool_available(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _binary_available(name: str) -> bool:
    return shutil.which(name) is not None


def choose_tool(preferred: str | None) -> str:
    """Return the first available tool in the fallback chain."""
    chain = ["marker", "pymupdf4llm", "poppler"]
    if preferred:
        chain = [preferred] + [t for t in chain if t != preferred]

    for tool in chain:
        if tool == "marker" and _tool_available("marker"):
            return "marker"
        if tool == "pymupdf4llm" and _tool_available("pymupdf4llm"):
            return "pymupdf4llm"
        if tool == "poppler" and _binary_available("pdftotext") and _binary_available("pdfimages"):
            return "poppler"
    raise RuntimeError(
        "No PDF conversion tool available. Install one of: "
        "`pip install marker-pdf`, `pip install pymupdf4llm`, or `brew install poppler`."
    )


# --------------------------------------------------------------------------- #
# Section parsing from markdown
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def split_markdown_sections(md: str) -> list[Section]:
    """Chunk markdown into Section objects on heading boundaries."""
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [Section(name="Document", text=md, level=1)]

    sections: list[Section] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        name = m.group(2).strip()
        level = len(m.group(1))
        body = md[start:end].strip()
        if not body and i + 1 < len(matches):
            # heading with no body → skip; content belongs to next section
            continue
        sections.append(Section(name=name, text=body, level=level))
    return sections


# --------------------------------------------------------------------------- #
# Tool-specific conversion
# --------------------------------------------------------------------------- #


def convert_with_marker(pdf_path: Path, workspace: Path) -> tuple[str, list[Figure], list[Table]]:
    """Use marker-pdf for text + figure + table extraction.

    Requires: pip install marker-pdf
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    figures_dir = workspace / "figures"
    tables_dir = workspace / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    md_text, _, images = text_from_rendered(rendered)

    figures: list[Figure] = []
    for i, (name, pil_image) in enumerate(images.items(), start=1):
        fig_id = f"fig_{i}"
        path = figures_dir / f"{fig_id}.png"
        pil_image.save(path)
        caption = _caption_near_image(md_text, name)
        page = _page_of_image(name)
        cls = _classify_figure(caption, md_text, fig_id)
        fig = Figure(id=fig_id, path=f"figures/{fig_id}.png", caption=caption, page=page, cls=cls)
        figures.append(fig)
        (figures_dir / f"{fig_id}.txt").write_text(
            f"caption: {caption}\nclass: {cls}\npage: {page}\n"
        )

    # marker encodes tables as markdown inline. Extract them.
    tables = _extract_tables_from_markdown(md_text, tables_dir)

    return md_text, figures, tables


def convert_with_pymupdf4llm(pdf_path: Path, workspace: Path) -> tuple[str, list[Figure], list[Table]]:
    """Use pymupdf4llm for fast text + basic tables.

    Requires: pip install pymupdf4llm pymupdf
    Figures are extracted separately via pymupdf's page.get_images().
    """
    import pymupdf4llm
    import pymupdf

    figures_dir = workspace / "figures"
    tables_dir = workspace / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    # Figures: pymupdf image extraction per page
    figures: list[Figure] = []
    doc = pymupdf.open(str(pdf_path))
    counter = 0
    for page_idx, page in enumerate(doc, start=1):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            counter += 1
            fig_id = f"fig_{counter}"
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n >= 5:  # CMYK → RGB
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            out = figures_dir / f"{fig_id}.png"
            pix.save(out)
            caption = _caption_near_image(md_text, fig_id) or ""
            cls = _classify_figure(caption, md_text, fig_id)
            figures.append(Figure(
                id=fig_id, path=f"figures/{fig_id}.png",
                caption=caption, page=page_idx, cls=cls,
            ))
            (figures_dir / f"{fig_id}.txt").write_text(
                f"caption: {caption}\nclass: {cls}\npage: {page_idx}\n"
            )
    doc.close()

    tables = _extract_tables_from_markdown(md_text, tables_dir)
    return md_text, figures, tables


def convert_with_poppler(pdf_path: Path, workspace: Path) -> tuple[str, list[Figure], list[Table]]:
    """Last-resort: pdftotext + pdfimages. No tables, no captions."""
    figures_dir = workspace / "figures"
    tables_dir = workspace / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    txt_out = workspace / "raw.txt"
    subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), str(txt_out)],
        check=True,
    )
    md_text = txt_out.read_text()

    subprocess.run(
        ["pdfimages", "-png", str(pdf_path), str(figures_dir / "fig")],
        check=True,
    )
    figures: list[Figure] = []
    for i, p in enumerate(sorted(figures_dir.glob("fig-*.png")), start=1):
        new = figures_dir / f"fig_{i}.png"
        p.rename(new)
        figures.append(Figure(
            id=f"fig_{i}", path=f"figures/fig_{i}.png",
            caption="", page=0, cls="supporting",
        ))
        (figures_dir / f"fig_{i}.txt").write_text(f"caption: \nclass: supporting\npage: 0\n")

    return md_text, figures, []


# --------------------------------------------------------------------------- #
# Caption / classification / table extraction helpers
# --------------------------------------------------------------------------- #


_CAPTION_RE = re.compile(
    r"(?:^|\n)\s*(?:Figure|Fig\.?|Table|Tab\.?)\s*\d+[.:\-]\s*([^\n]+)",
    re.IGNORECASE,
)


def _caption_near_image(md_text: str, anchor: str) -> str:
    """Look for a 'Figure N: ...' caption near the anchor (best-effort)."""
    m = _CAPTION_RE.search(md_text)
    return m.group(1).strip() if m else ""


def _page_of_image(name: str) -> int:
    m = re.search(r"page[_-]?(\d+)", name)
    return int(m.group(1)) if m else 0


_KEY_CLUE_RE = re.compile(
    r"\b(architecture|overview|pipeline|framework|teaser|main result)\b",
    re.IGNORECASE,
)


def _classify_figure(caption: str, body: str, fig_id: str) -> str:
    """Key | supporting | decorative.

    Key-ness comes from the caption alone (not the surrounding body text),
    plus fig_1 being the teaser by convention. The body is only used to
    distinguish referenced (supporting) from unreferenced (decorative).
    """
    if caption and _KEY_CLUE_RE.search(caption):
        return "key"
    if fig_id == "fig_1":
        return "key"
    num = re.search(r"\d+", fig_id)
    if num and re.search(rf"\bfig(?:ure)?\.?\s*{num.group()}\b", body, re.IGNORECASE):
        return "supporting"
    return "decorative"


_MD_TABLE_RE = re.compile(
    r"(?:(?:^|\n)\|[^\n]+\|\n\|[\s\-:|]+\|(?:\n\|[^\n]+\|)+)",
    re.MULTILINE,
)


_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:Table|Tab\.?)\s*\d+\s*[.:\-]\s*([^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_tables_from_markdown(md_text: str, tables_dir: Path) -> list[Table]:
    """Pull markdown tables out into their own files with caption hints.

    Caption heuristic: look at the line immediately AFTER the table first
    (paper convention for tables is often caption-above, but markdown renders
    often put it below), then immediately BEFORE. Never scan the whole document.
    """
    tables: list[Table] = []
    for i, m in enumerate(_MD_TABLE_RE.finditer(md_text), start=1):
        table_md = m.group(0).strip()

        after = md_text[m.end():m.end() + 300]
        before = md_text[max(0, m.start() - 300):m.start()]
        caption = ""
        for window in (after, before):
            cap = _TABLE_CAPTION_RE.search(window)
            if cap:
                caption = cap.group(1).strip()
                break

        tab_id = f"tab_{i}"
        out = tables_dir / f"{tab_id}.md"
        out.write_text(f"<!-- caption: {caption} -->\n\n{table_md}\n")
        tables.append(Table(id=tab_id, path=f"tables/{tab_id}.md", caption=caption, page=0))
    return tables


# --------------------------------------------------------------------------- #
# Section ↔ figure/table pairing
# --------------------------------------------------------------------------- #


def pair_references(
    sections: list[Section],
    figures: list[Figure],
    tables: list[Table],
) -> None:
    """For each section, record which figures/tables its text mentions."""
    for sec in sections:
        body = sec.text
        for fig in figures:
            num = re.search(r"\d+", fig.id)
            if num and re.search(
                rf"\bfig(?:ure)?\.?\s*{num.group()}\b", body, re.IGNORECASE
            ):
                sec.figures.append(fig.id)
        for tab in tables:
            num = re.search(r"\d+", tab.id)
            if num and re.search(
                rf"\btab(?:le)?\.?\s*{num.group()}\b", body, re.IGNORECASE
            ):
                sec.tables.append(tab.id)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def fetch_pdf(path_or_url: str, workspace: Path) -> Path:
    """Return a local path to the PDF, downloading if needed."""
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        local = workspace / "input.pdf"
        subprocess.run(
            ["curl", "-sSL", "--compressed", "-o", str(local), path_or_url],
            check=True,
        )
        return local
    p = Path(path_or_url)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def write_workspace(pc: PaperContent, md_text: str, workspace: Path) -> None:
    (workspace / "paper.md").write_text(md_text)
    data = {
        "paper_id": pc.paper_id,
        "source": pc.source,
        "title": pc.title,
        "authors": pc.authors,
        "sections": [asdict(s) for s in pc.sections],
        "figures": [
            {"id": f.id, "path": f.path, "caption": f.caption, "page": f.page, "class": f.cls}
            for f in pc.figures
        ],
        "tables": [asdict(t) for t in pc.tables],
    }
    (workspace / "paper_content.json").write_text(json.dumps(data, indent=2))


def write_log(workspace: Path, tool: str, pc: PaperContent, warnings: list[str]) -> None:
    (workspace / "conversion.log").write_text(
        f"tool: {tool}\n"
        f"sections: {len(pc.sections)}\n"
        f"figures: {len(pc.figures)}\n"
        f"tables: {len(pc.tables)}\n"
        f"warnings:\n" + "\n".join(f"  - {w}" for w in warnings) + ("\n" if warnings else "")
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert PDF to arxiv-latex-reader workspace.")
    ap.add_argument("pdf", help="Path or URL to the PDF.")
    ap.add_argument("--paper-id", required=True, help="Slug for the workspace directory.")
    ap.add_argument("--workspace", default="workspace", help="Workspace root (default: workspace).")
    ap.add_argument("--tool", choices=["marker", "pymupdf4llm", "poppler"], help="Preferred tool.")
    ap.add_argument("--strict", action="store_true",
                    help="Raise on any per-page failure instead of skipping.")
    args = ap.parse_args()

    workspace = Path(args.workspace) / args.paper_id
    workspace.mkdir(parents=True, exist_ok=True)

    tool = choose_tool(args.tool)
    pdf_path = fetch_pdf(args.pdf, workspace)

    warnings: list[str] = []
    try:
        if tool == "marker":
            md_text, figures, tables = convert_with_marker(pdf_path, workspace)
        elif tool == "pymupdf4llm":
            md_text, figures, tables = convert_with_pymupdf4llm(pdf_path, workspace)
            warnings.append("pymupdf4llm: equations may be lost; check paper.md")
        else:
            md_text, figures, tables = convert_with_poppler(pdf_path, workspace)
            warnings.append("poppler: no table extraction; captions missing")
    except Exception as e:
        if args.strict:
            raise
        print(f"[pdf-reader] conversion failed with {tool}: {e}", file=sys.stderr)
        sys.exit(2)

    sections = split_markdown_sections(md_text)
    pair_references(sections, figures, tables)

    title = sections[0].name if sections else args.paper_id
    pc = PaperContent(
        paper_id=args.paper_id,
        source="pdf",
        title=title,
        sections=sections,
        figures=figures,
        tables=tables,
    )
    write_workspace(pc, md_text, workspace)
    write_log(workspace, tool, pc, warnings)

    print(
        f"[pdf-reader] {args.paper_id}: tool={tool} "
        f"sections={len(sections)} figures={len(figures)} tables={len(tables)}"
    )
    print(f"[pdf-reader] workspace: {workspace}")
    print(f"[pdf-reader] next: python arxiv-latex-reader/scripts/section_indexer.py {args.paper_id}")


if __name__ == "__main__":
    main()

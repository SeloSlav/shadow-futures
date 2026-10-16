"""Extract the initial Word plans as Git-reviewable text without rewriting them.

This is a one-time import helper. It refuses to overwrite an edited text draft.
The imported Markdown is the working text; future Word files are reading exports.
"""
from pathlib import Path

from docx import Document

BOOK = Path(__file__).resolve().parents[1]
DESTINATION = BOOK / "planning"
DESTINATION.mkdir(exist_ok=True)

for source_name, target_name in [
    ("Shadow_Futures_Argument_Brief.docx", "argument-brief.md"),
    ("Shadow_Futures_Committed_Book_Outline.docx", "book-outline.md"),
]:
    target = DESTINATION / target_name
    if target.exists():
        raise FileExistsError(f"Preserve the existing working draft: {target}")
    paragraphs = []
    for paragraph in Document(BOOK / "drafts" / source_name).paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        name = paragraph.style.name
        prefix = {"Title": "# ", "Heading 1": "## ", "Heading 2": "### "}.get(name, "")
        paragraphs.append(prefix + text)
    target.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    print(f"Imported {target.relative_to(BOOK)}")

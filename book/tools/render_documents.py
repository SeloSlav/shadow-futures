"""Use the packaged DOCX rasterizer with Word as Windows' PDF converter.

The selected runtime bundles Poppler but no LibreOffice on Windows. Word
exports the actual DOCX; the packaged renderer then produces the page PNGs.
No installed desktop LibreOffice or surrogate PDF layout is used.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
RENDER_TEMP = ROOT / ".tmp" / "book-planning" / "render-temp"
RENDER_TEMP.mkdir(parents=True, exist_ok=True)
assert RENDER_TEMP.resolve().is_relative_to(ROOT.resolve())
tempfile.tempdir = str(RENDER_TEMP)
DEPENDENCIES = Path(r"C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies")
RENDERER = Path(r"C:\Users\Asus\.codex\plugins\cache\openai-primary-runtime\documents\26.1013.11914\skills\documents\render_docx.py")
os.environ["PATH"] = str(DEPENDENCIES / "native" / "poppler" / "Library" / "bin") + os.pathsep + os.environ["PATH"]
spec = importlib.util.spec_from_file_location("packaged_docx_renderer", RENDERER)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def word_conversion(input_path, _profile, output_dir, stem, verbose=False):
    output = str(Path(output_dir) / f"{stem}.pdf")
    process = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "book" / "tools" / "export_word_pdf.ps1"), "-DocumentPath", str(input_path), "-PdfPath", output],
        capture_output=True, text=True, timeout=90,
    )
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    return output, "Microsoft Word fixed-format export\n" + process.stdout


renderer.convert_to_pdf = word_conversion
results = []
for source in sorted((ROOT / "book" / "drafts").glob("*.docx")):
    target = ROOT / ".tmp" / "book-planning" / source.stem
    images = renderer.rasterize(str(source), str(target), dpi=150, verbose=True, emit_pdf=True)
    pdf = target / f"{source.stem}.pdf"
    reader = PdfReader(pdf)
    page_text = [page.extract_text() or "" for page in reader.pages]
    (target / "page-text.json").write_text(json.dumps(page_text, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"document": str(source), "page_count": len(reader.pages), "page_character_counts": list(map(len, page_text)), "images": images, "converter": "Microsoft Word", "rasterizer": str(RENDERER)}
    results.append(result)
    print(json.dumps(result), flush=True)
(ROOT / "book" / "research" / "render_verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

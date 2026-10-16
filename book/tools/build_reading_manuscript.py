"""Assemble the current chapters, endnotes and back matter without editing prose.

The chapter files and curated bibliography JSON are authoritative. This tool is
separate from the original planning-document builder. Counts are an inventory,
not factual, mathematical or editorial certification.
"""
from pathlib import Path
import hashlib
import json
import re

from manuscript_status import ROOT, PLAN, main as update_inventory, words

BIBLIOGRAPHY = ROOT / "book/research/bibliography.json"
REFERENCES = ROOT / "book/manuscript/90-references.md"
OUTPUT = ROOT / "book/drafts/Shadow_Futures_Complete_Manuscript.md"
MANIFEST = ROOT / "book/research/reading-manuscript-manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_references(bibliography):
    if sum(bool(entry.get("canonical_v5_reference")) for entry in bibliography["entries"]) != 24:
        raise ValueError("Bibliography must retain all 24 canonical v5 references")
    lines = [
        "# References", "",
        "The chapter notes identify the passages and versions used, distinguish findings from interpretation, and record access limits. This bibliography consolidates duplicate source identifiers while retaining all 24 references from the canonical v5 paper.",
        "",
    ]
    ids = set()
    for entry in bibliography["entries"]:
        if entry["id"] in ids:
            raise ValueError(f"Duplicate bibliography id: {entry['id']}")
        ids.add(entry["id"])
        for link in entry["links"]:
            if not link["url"] or not link["label"]:
                raise ValueError(f"Empty reference link: {entry['id']}")
            if not link["url"].startswith(("https://", "http://")):
                if not (REFERENCES.parent / link["url"]).is_file():
                    raise FileNotFoundError(f"Reference manuscript missing: {link['url']}")
        links = " ".join(f"[{link['label']}]({link['url']})" for link in entry["links"])
        lines += [f"{entry['citation']} {links}", ""]
    REFERENCES.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def namespace_notes(text, prefix):
    return re.sub(r"\[\^([^\]]+)\]", lambda m: f"[^{prefix}-{m.group(1)}]", text)


def build():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    bibliography = json.loads(BIBLIOGRAPHY.read_text(encoding="utf-8"))
    ledger_keys = {
        source["key"]
        for path in (ROOT / "book/research").glob("*source-notes.json")
        for source in json.loads(path.read_text(encoding="utf-8")).get("sources", [])
    }
    covered_keys = {key for entry in bibliography["entries"] for key in entry["source_keys"]}
    if ledger_keys - covered_keys:
        raise ValueError(f"Sources missing from bibliography: {sorted(ledger_keys - covered_keys)}")
    render_references(bibliography)
    update_inventory()
    progress = json.loads((ROOT / "book/research/manuscript-progress.json").read_text(encoding="utf-8"))
    if progress["main_words"] > plan["main_text_target"]:
        raise ValueError("Main text exceeds the approved 80,000-word budget; rebalance revisions before assembly")
    if progress["back_matter_words"] > plan["back_matter_target_max"]:
        raise ValueError("Back matter exceeds the approved word allowance")
    inputs, bodies, endnotes = [], [], []
    main_total = 0
    toc = []
    for item in plan["components"]:
        path = ROOT / item["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Main-text component missing: {path}")
        text = path.read_text(encoding="utf-8-sig")
        body, separator, notes = text.partition("\n## Notes\n")
        prefix = item["id"]
        body = namespace_notes(body, prefix)
        notes = namespace_notes(notes, prefix)
        main_total += words(body)
        if not separator:
            raise ValueError(f"Missing Notes section: {path}")
        toc.append(f"- [{item['title']}](#{prefix})")
        bodies.append(f'<a id="{prefix}"></a>\n\n{body.strip()}')
        endnotes.append(f"## {item['title']}\n\n{notes.strip()}")
        inputs.append({"file": item["file"], "sha256": digest(path)})

    back = []
    for item in plan["back_matter"]:
        path = ROOT / item["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Back matter missing: {path}")
        toc.append(f"- [{item['id'].replace('_', ' ').title()}](#{item['id']})")
        back.append(f'<a id="{item["id"]}"></a>\n\n{path.read_text(encoding="utf-8-sig").strip()}')
        inputs.append({"file": item["file"], "sha256": digest(path)})

    front = [
        "# Shadow Futures", "", "Martin Erlic", "",
        "Complete working draft · 16 October 2026", "",
        "This reading copy assembles the saved chapters and relocates their notes to the end. The individual manuscript files remain authoritative. All main-text components are drafted; final editorial and source review remains in progress.",
        "", "## Contents", "", *toc, "", "- [Endnotes](#endnotes)", "",
    ]
    # Put endnotes before references/acknowledgments in both reading order and contents.
    front = [line for line in front if line != "- [Endnotes](#endnotes)"]
    ref_position = front.index("- [References](#references)")
    front.insert(ref_position, "- [Endnotes](#endnotes)")
    text = "\n".join(front).rstrip() + "\n\n" + "\n\n---\n\n".join(bodies)
    text += '\n\n---\n\n<a id="endnotes"></a>\n\n# Endnotes\n\n' + "\n\n".join(endnotes)
    text += "\n\n---\n\n" + "\n\n---\n\n".join(back) + "\n"

    definitions = re.findall(r"^\[\^([^\]]+)\]:", text, re.M)
    all_refs = re.findall(r"\[\^([^\]]+)\]", text)
    used = {ref for ref in all_refs if all_refs.count(ref) > 1}
    if len(definitions) != len(set(definitions)):
        raise ValueError("Duplicate endnote ids in assembled manuscript")
    if set(all_refs) != set(definitions) or used != set(definitions):
        raise ValueError("Unmatched or unused endnote in assembled manuscript")
    if "\ufffd" in text:
        raise ValueError("Replacement character in manuscript")
    if main_total != progress["main_words"]:
        raise ValueError("Assembly changed the main-text word inventory")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "status": "complete_draft_assembly_review_pending",
        "file": OUTPUT.relative_to(ROOT).as_posix(),
        "sha256": digest(OUTPUT),
        "inputs": inputs,
        "bibliography_input_sha256": digest(BIBLIOGRAPHY),
        "main_words": progress["main_words"],
        "back_matter_words": progress["back_matter_words"],
        "reference_entries": len(bibliography["entries"]),
        "endnote_count": len(definitions),
        "notes_namespaced": True,
        "interpretation": "Assembly, counts, hashes and note matching do not certify facts, proofs, prose quality or completion of the book review.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Reading draft: {OUTPUT.relative_to(ROOT)}; {len(definitions)} endnotes; {len(bibliography['entries'])} references.")


if __name__ == "__main__":
    build()

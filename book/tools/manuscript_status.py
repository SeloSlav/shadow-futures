"""Count the working book against its approved budget and report citation links.

This is an inventory, not a factual, mathematical, or editorial certification.
Run from any directory. The working chapter files are authoritative.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "book/planning/manuscript-plan.json"


def words(text):
    text = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    text = re.sub(r"\[\^[^\]]+\]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\n)]+\)", r"\1", text)
    return len(text.split())


def inspect_component(item):
    path = ROOT / item["file"]
    row = dict(item)
    row.update(exists=path.exists(), main_words=0, note_words=0)
    if not path.exists():
        return row
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    body, separator, notes = text.partition("\n## Notes\n")
    used = set(re.findall(r"\[\^([^\]]+)\]", body))
    defined = re.findall(r"^\[\^([^\]]+)\]:", notes, re.M)
    row.update(
        main_words=words(body),
        note_words=words(notes) if separator else 0,
        sha256=hashlib.sha256(raw).hexdigest(),
        note_count=len(defined),
        missing_notes=sorted(used - set(defined)),
        unused_notes=sorted(set(defined) - used),
        duplicate_notes=len(defined) != len(set(defined)),
    )
    return row


def main():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert sum(item["target"] for item in plan["components"]) == plan["main_text_target"]
    rows = [inspect_component(item) for item in plan["components"]]
    back_rows = []
    for item in plan["back_matter"]:
        path = ROOT / item["file"]
        back_rows.append(dict(item, exists=path.exists(), words=words(path.read_text(encoding="utf-8-sig")) if path.exists() else 0))
    total = sum(row["main_words"] for row in rows)
    back_total = sum(row["note_words"] for row in rows) + sum(row["words"] for row in back_rows)
    report = {
        "status": "in_progress",
        "main_text_target": plan["main_text_target"],
        "main_words": total,
        "words_remaining_to_target": plan["main_text_target"] - total,
        "back_matter_words": back_total,
        "back_matter_target": [plan["back_matter_target_min"], plan["back_matter_target_max"]],
        "components": rows,
        "back_matter": back_rows,
        "interpretation": "File existence and counts do not prove source accuracy, editorial quality, or completion. See the plan's completion requirements.",
    }
    output = ROOT / "book/research/manuscript-progress.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = [
        "# Shadow Futures manuscript", "",
        f"Martin Erlic · Working manuscript · {total:,} of {plan['main_text_target']:,} main-text words", "",
        "The approved budget is 2,500 words for the prologue, 3,500 for the introduction, twelve chapters averaging 6,000 words (72,000 combined), and a 2,000-word epilogue. Notes, references, and acknowledgments have a separate 10,000–15,000-word allowance.", "",
        "Counts below come from the current chapter files. Drafted sections still require the final whole-book review; missing sections are not represented by placeholder chapters.", "",
        "| Component | Target | Written | Stage |", "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        label = row["title"]
        if row["exists"]:
            label = f"[{label}]({row['file'].removeprefix('book/')})"
        lines.append(f"| {label} | {row['target']:,} | {row['main_words']:,} | {row['stage'].replace('_', ' ')} |")
    lines += ["", f"Current back matter: {back_total:,} words, including chapter notes.", ""]
    if back_rows:
        lines += ["| Back matter | Written | Stage |", "| --- | ---: | --- |"]
        for row in back_rows:
            label = row["id"].replace("_", " ").title()
            if row["exists"]:
                label = f"[{label}]({row['file'].removeprefix('book/')})"
            lines.append(f"| {label} | {row['words']:,} | {row['stage'].replace('_', ' ')} |")
        lines.append("")
    if (ROOT / "book/drafts/Shadow_Futures_Complete_Manuscript.md").is_file():
        lines += ["[Read the continuous working manuscript](drafts/Shadow_Futures_Complete_Manuscript.md), with endnotes, references and acknowledgments. It remains a draft pending the whole-book review.", ""]
    lines += ["Regenerate the inventory with `python book/tools/manuscript_status.py`; rebuild the reading copy with `python book/tools/build_reading_manuscript.py`. The argument brief, committed outline, canonical v5, and source notes govern the substance.", ""]
    (ROOT / "book/MANUSCRIPT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Main text: {total:,} / {plan['main_text_target']:,}. Back matter: {back_total:,}.")
    for row in rows:
        if row["exists"]:
            print(f"  {row['id']}: {row['main_words']:,} main + {row['note_words']:,} notes; {row['stage']}")
            if row["missing_notes"] or row["unused_notes"] or row["duplicate_notes"]:
                raise ValueError(f"Footnote mismatch in {row['file']}")


if __name__ == "__main__":
    main()

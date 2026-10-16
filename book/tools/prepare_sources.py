"""Preserve the book's source identity and its complete v5 reference inventory."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import urllib.request
from datetime import date
from pathlib import Path

from docx import Document
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "book"
RESEARCH = BOOK / "research"
CLASSICS = BOOK / "sources" / "classics"
EXTERNAL = Path(r"C:\Shadow Futures\Final")
STEM = "Shadow_Futures_FINAL_Comparison_Budget_Revised_v5"

# These describe each work's role in v5 and its planned use, rather than claiming
# that every source's complete text has independently been reviewed for the book.
ROLES = [
    ("Akerlof1970", [1, 3], "Hidden quality and adverse selection", "Contrast hidden product quality with missing historical comparisons even when inputs are observed."),
    ("Arthur1989", [4], "Increasing returns and historical selection", "Credit the established account of adoption, increasing returns, and lock-in."),
    ("BarYam2016", [2, 8], "Data and decision-relevant information", "Explain why additional records need not resolve the particular causal question."),
    ("BlackwellDubins1962", [3, 8, 12], "Predictive merging", "Credit the classical prediction result; v5 supplies the allocation setting and attribution conjunction."),
    ("David1985", [4], "Historical path dependence and QWERTY", "Use in the history note with the disputed empirical status of the keyboard example made explicit."),
    ("Gabriyelyan2011", [3], "Probability laws on filtered spaces", "Credit the mathematical foundation in a plain-language chapter endnote; do not reproduce the proof."),
    ("GottfriedGrosskinsky2024", [2, 3], "Nonlinear reinforced urns", "Credit the reinforcement results behind the sufficient strong-feedback example."),
    ("Hayek2002", [4, 10, 12], "Competition as discovery", "Credit the 2002 translation and distinguish its original 1968 lecture from the publication year."),
    ("JacodShiryaev2003", [3], "Stochastic-process probability machinery", "Technical lineage endnote for the main theorem, using the exact v5 edition."),
    ("KabanovLiptserShiryaev1977", [3], "Absolute continuity and singularity", "Technical lineage endnote; the general probability result is not a new result of Shadow Futures."),
    ("Kakutani1948", [3], "Infinite product measures", "Explain the older question of what an indefinitely long record can distinguish."),
    ("KasySautmann2021", [2, 10, 12], "Adaptive assignment and policy learning", "Develop allocation as an experimental-design choice with both learning benefits and opportunity costs."),
    ("KerrNandaRhodesKropf2014", [6, 7, 11], "Entrepreneurship as experimentation", "Present the incentive case for risk-taking and prospective rewards seriously."),
    ("LeCamYang2000", [3], "Limits on statistical precision", "Credit the testing and lower-bound machinery in the finite-record endnote."),
    ("LeGoffSoulier2017", [3], "Prior estimation failure for a reinforced urn", "State explicitly that single-trajectory estimation failure predates v5."),
    ("Meghani2021", [5, 10], "Prior use of epistemic monopoly", "Credit the term and the knowledge-governance argument before defining the book's narrower allocation use."),
    ("Merton1968", [1, 4, 6], "Cumulative advantage in science", "Credit the Matthew effect; do not present success breeding success as a new mechanism."),
    ("Oliveira2009", [2, 3], "Dominance under reinforcement", "Credit the mathematical antecedent for eventual dominance under sufficient feedback."),
    ("Pemantle2007", [2, 3], "Reinforced random processes", "General reinforcement lineage and the distinction between different reinforcement regimes."),
    ("PikettySaezStantcheva2014", [6, 11], "Top-income taxation and behavioral margins", "Explain productive effort, bargaining, and tax responses without attributing an exact-merit requirement to optimal-tax theory."),
    ("RothschildScheuer2016", [6, 11], "Taxation with rent-seeking", "Connect to established productive versus rent-seeking distinctions; v5 limits a stronger exact ex post decomposition."),
    ("SalganikDoddsWatts2006", [0, 1, 3, 12], "Parallel cultural markets", "Opening and closing case; empirical analogue rather than a proof of the finite-budget theorem."),
    ("VanDeRijtEtAl2014", [1, 4, 6], "Field experiments in cumulative success", "Provide causal evidence of success-breeds-success mechanisms, with the domain and intervention specified."),
    ("Zhang2004", [3, 8], "Prediction alongside inconsistent parameter estimation", "Credit the prior distinction in geostatistics; avoid claiming that prediction without parameter recovery is newly discovered."),
]

CLASSIC_SOURCES = [
    {"key": "Blackwell1953", "title": "Equivalent Comparisons of Experiments", "authors": "David Blackwell", "year": 1953,
     "doi": "10.1214/aoms/1177729032", "filename": "Blackwell_1953_Equivalent_Comparisons_of_Experiments.pdf",
     "urls": ["https://www.jstor.org/stable/pdf/2236332.pdf", "https://projecteuclid.org/download/pdf_1/euclid.aoms/1177729032"],
     "v5_reference": False, "role": "Decision-based comparison of experiments; additional foundation for the policy extension."},
    {"key": "BlackwellDubins1962", "title": "Merging of Opinions with Increasing Information", "authors": "David Blackwell and Lester Dubins", "year": 1962,
     "doi": "10.1214/aoms/1177704456", "filename": "Blackwell_Dubins_1962_Merging_of_Opinions.pdf",
     "urls": ["https://www.dklevine.com/archive/refs4565.pdf", "http://www.dklevine.com/archive/refs4565.pdf", "https://citeseerx.ist.psu.edu/document?doi=8043220ae5ef8dabcee283c8409085bc276009f3&repid=rep1&type=pdf"],
     "v5_reference": True, "role": "Classical predictive merging under the relevant absolute-continuity condition."},
    {"key": "Arrow1962", "title": "Economic Welfare and the Allocation of Resources for Invention", "authors": "Kenneth J. Arrow", "year": 1962,
     "filename": "Arrow_1962_Economic_Welfare_and_Invention.pdf",
     "urls": ["https://www.nber.org/system/files/chapters/c2144/c2144.pdf"],
     "v5_reference": False, "role": "Uncertainty, knowledge production, and limits to private appropriation; policy foundation added to the book."},
    {"key": "Arrow1963", "title": "Uncertainty and the Welfare Economics of Medical Care", "authors": "Kenneth J. Arrow", "year": 1963,
     "doi": "10.2307/1812044", "filename": "Arrow_1963_Uncertainty_and_Medical_Care.pdf",
     "urls": ["https://assets.aeaweb.org/asset-server/files/9442.pdf"],
     "v5_reference": False, "role": "How uncertainty and information conditions shape institutions; medical care is a comparison, not evidence for v5's theorem."},
    {"key": "Alchian1950", "title": "Uncertainty, Evolution, and Economic Theory", "authors": "Armen A. Alchian", "year": 1950,
     "doi": "10.1086/256940", "filename": "Alchian_1950_Uncertainty_Evolution_and_Economic_Theory.pdf",
     "urls": ["https://docenti.luiss.it/protected-uploads/623/2016/04/20160405105251-Uncertainty-Evolution-and-Economic-Theory-Alchian-1950.pdf", "https://ios23.classes.ryansafner.com/files/readings/Alchian-1950.pdf"],
     "v5_reference": False, "role": "Profit-based selection under uncertainty; a central intellectual interlocutor added to the book, not an existing v5 reference."},
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(item: dict) -> dict:
    record = dict(item)
    destination = CLASSICS / item["filename"]
    errors = []
    for url in item["urls"]:
        try:
            if not destination.exists():
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research source retrieval)"})
                with urllib.request.urlopen(request, timeout=18) as response:
                    data = response.read()
                if not data.startswith(b"%PDF-"):
                    raise ValueError("Response was not a PDF")
                destination.write_bytes(data)
            reader = PdfReader(destination)
            record.update(status="downloaded", source_url=url, local_path=str(destination.relative_to(ROOT)), sha256=sha256(destination), pages=len(reader.pages), opening_text=(reader.pages[0].extract_text() or "")[:1300])
            print(json.dumps({"source": item["key"], "status": "downloaded", "pages": len(reader.pages)}), flush=True)
            return record
        except Exception as error:
            errors.append(f"{url}: {type(error).__name__}: {error}")
    record.update(status="link_only", errors=errors, source_url=item["urls"][0])
    print(json.dumps({"source": item["key"], "status": "link_only", "errors": errors}), flush=True)
    return record


def main() -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    CLASSICS.mkdir(parents=True, exist_ok=True)
    provenance = []
    for extension, project_dir in [("docx", "docx"), ("pdf", "pdf")]:
        source = EXTERNAL / f"{STEM}.{extension}"
        existing = ROOT / "output" / project_dir / source.name
        if not existing.exists():
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_bytes(source.read_bytes())
        if sha256(source) != sha256(existing):
            raise RuntimeError(f"Source mismatch: {existing}")
        provenance.append({"external_path": str(source), "project_path": str(existing.relative_to(ROOT)), "sha256": sha256(source), "byte_identical": True})
    canonical = ROOT / "output" / "docx" / f"{STEM}.docx"
    paragraphs = [p.text for p in Document(canonical).paragraphs]
    begin = paragraphs.index("References") + 1
    end = paragraphs.index("Technical Appendix")
    bibliography = [p.strip() for p in paragraphs[begin:end] if p.strip()]
    if len(bibliography) != len(ROLES):
        raise RuntimeError(f"Unexpected bibliography count: {len(bibliography)}")
    register = [{"key": role[0], "citation": citation, "planned_chapters": role[1], "subject": role[2], "use_and_boundary": role[3], "inventory_status": "transcribed_from_canonical_v5", "book_full_text_check": "required_before_claim_specific_drafting"} for citation, role in zip(bibliography, ROLES)]
    (RESEARCH / "v5_reference_register.json").write_text(json.dumps({"canonical_source": provenance, "reference_count": len(register), "references": register}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESEARCH / "v5_reference_inventory.txt").write_text("SHADOW FUTURES V5 REFERENCE INVENTORY\n\n" + "\n\n".join(f"[{r['key']}] {r['citation']}\nBook chapters: {', '.join(map(str, r['planned_chapters']))}\nRole: {r['use_and_boundary']}" for r in register), encoding="utf-8")
    (RESEARCH / "v5_source_provenance.json").write_text(json.dumps({"checked_on": "2026-10-16", "sources": provenance}, indent=2), encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        classics = list(executor.map(download, CLASSIC_SOURCES))
    (RESEARCH / "classics_manifest.json").write_text(json.dumps({"retrieved_on": "2026-10-16", "papers": classics}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"v5_references": len(register), "source_identity_verified": True, "downloaded_classics": sum(c["status"] == "downloaded" for c in classics)}), flush=True)


if __name__ == "__main__":
    main()

"""Validate the public evidence ledger and reproduce the paper's pilot counts."""

import calendar
import json
from collections import Counter, defaultdict
from datetime import date
from fractions import Fraction
from pathlib import Path


PATH = Path(__file__).with_name("acquisition-pilot-evidence.json")
SHIP = "confirmed shipment"
STOP = "documented cessation"
UNCERTAIN = {"active only", "unresolved"}
OVERLAP = {"direct", "potential", "no direct overlap established"}
FELL_THROUGH = "fell through"
CLOSED = "closed"


def evidence_interval(value):
    if len(value) == 10:
        day = date.fromisoformat(value)
        return day, day
    month_start = date.fromisoformat(value + "-01")
    month_length = calendar.monthrange(month_start.year, month_start.month)[1]
    return month_start, month_start.replace(day=month_length)


def load_ledger():
    ledger = json.loads(PATH.read_text(encoding="utf-8"))
    deals = ledger["deals"]
    screen = ledger["screening"]
    start, end = screen["disposition_window"]
    cases = screen["candidate_records"]
    if len({case["case_source"] for case in cases}) != len(cases):
        raise ValueError("Duplicate CMA case records")
    if {case["deal"] for case in cases if case["decision"] == "included"} != deals.keys():
        raise ValueError("The eligibility screen and deal ledger disagree")
    if {project["deal"] for project in ledger["projects"]} != deals.keys():
        raise ValueError("An included deal lacks a product record")
    for case in cases:
        if case["decision"] != "included":
            continue
        deal = deals[case["deal"]]
        disposition = case["first_phase2_disposition"]
        if not start <= disposition <= end or disposition != deal["cma_first_phase2_disposition"]:
            raise ValueError(f"Invalid CMA disposition: {case['name']}")
        if case["case_source"] != deal["case_source"]:
            raise ValueError(f"Mismatched CMA case link: {case['name']}")
    return ledger


def load_rows():
    ledger = load_ledger()
    deals = ledger["deals"]
    seen = set()
    for project in ledger["projects"]:
        key = (project["deal"], project["product"])
        if key in seen:
            raise ValueError(f"Duplicate product lineage: {key}")
        seen.add(key)
        deal = deals[project["deal"]]
        start = date.fromisoformat(deal["anchor_date"])
        end_year = start.year + 2
        cutoff = date(end_year, start.month, min(start.day, calendar.monthrange(end_year, start.month)[1]))
        outcome = project["outcome"]
        if deal["original_outcome"] not in {CLOSED, FELL_THROUGH}:
            raise ValueError(f"Invalid deal outcome: {key}")
        if project["overlap"] not in OVERLAP:
            raise ValueError(f"Invalid overlap code: {key}")
        if outcome != "unresolved":
            if outcome not in {SHIP, STOP, "active only"}:
                raise ValueError(f"Invalid outcome code: {key}")
            lo, hi = evidence_interval(project["evidence_date"])
            if not start <= lo <= hi <= cutoff:
                raise ValueError(f"Evidence outside follow-up: {key}")
        elif project["evidence_date"] is not None:
            raise ValueError(f"Unresolved row has a qualifying date: {key}")
        for field in ("overlap_source", "outcome_source"):
            if not project[field].startswith("https://"):
                raise ValueError(f"Missing public evidence URL: {key}, {field}")
        yield {**project, "deal_outcome": deal["original_outcome"], "structural_path": deal["structural_path"]}


def summarize(rows):
    grid = {}
    for outcome in (CLOSED, FELL_THROUGH):
        group = [row for row in rows if row["deal_outcome"] == outcome]
        counts = Counter(row["outcome"] for row in group)
        lower = Fraction(counts[SHIP], len(group))
        upper = Fraction(counts[SHIP] + sum(counts[k] for k in UNCERTAIN), len(group))
        grid[outcome] = {
            "deals": len({row["deal"] for row in group}),
            "projects": len(group),
            "confirmed": counts[SHIP],
            "active_only": counts["active only"],
            "unresolved": counts["unresolved"],
            "cessation": counts[STOP],
            "lower": lower,
            "upper": upper,
        }
    return grid


def contrasts(rows):
    direct = [row for row in rows if row["overlap"] == "direct"]
    potential = [row for row in rows if row["overlap"] == "potential"]
    strict = [
        {**row, "outcome": "active only"}
        if row.get("combined_successor") and row["outcome"] == SHIP else row
        for row in direct
    ]
    exact_line = [
        {**row, "outcome": "active only"}
        if row.get("group_level_extension") and row["outcome"] == SHIP else row
        for row in direct
    ]
    return {
        "all lines": summarize(rows),
        "direct overlap": summarize(direct),
        "direct and potential overlap": summarize(direct + potential),
        "direct without counting the combined FLX successor as a shipment": summarize(strict),
        "direct without counting the Euro Auctions group-level extension as a shipment": summarize(exact_line),
        "direct under both narrower product-line definitions": summarize([
            {**row, "outcome": "active only"}
            if row.get("combined_successor") and row["outcome"] == SHIP else row
            for row in exact_line
        ]),
        "direct excluding alternative-buyer and structural-remedy cases": summarize([
            row for row in direct if not row["structural_path"]
        ]),
    }


def descriptive_checks(rows):
    direct = [row for row in rows if row["overlap"] == "direct"]
    by_deal = defaultdict(list)
    for row in direct:
        by_deal[row["deal"]].append(row)

    deal_weighted = {}
    for disposition in (CLOSED, FELL_THROUGH):
        deal_groups = [group for group in by_deal.values() if group[0]["deal_outcome"] == disposition]
        lower = [Fraction(sum(row["outcome"] == SHIP for row in group), len(group)) for group in deal_groups]
        upper = [
            Fraction(sum(row["outcome"] != STOP for row in group), len(group))
            for group in deal_groups
        ]
        deal_weighted[disposition] = {
            "deals": len(deal_groups),
            "lower": sum(lower) / len(deal_groups),
            "upper": sum(upper) / len(deal_groups),
        }

    omit_one = {}
    for deal in by_deal:
        counts = summarize([row for row in direct if row["deal"] != deal])
        omit_one[deal] = counts[CLOSED]["lower"] - counts[FELL_THROUGH]["lower"]

    counts = summarize(direct)
    complete, abandon = counts[CLOSED], counts[FELL_THROUGH]
    tipping = {}
    for abandoned_extra in range(abandon["active_only"] + abandon["unresolved"] + 1):
        abandoned_rate = Fraction(abandon["confirmed"] + abandoned_extra, abandon["projects"])
        tipping[abandoned_extra] = [
            complete_extra
            for complete_extra in range(complete["active_only"] + complete["unresolved"] + 1)
            if Fraction(complete["confirmed"] + complete_extra, complete["projects"]) > abandoned_rate
        ]
    return {"equal deal weights": deal_weighted, "omit one deal": omit_one, "strict reversal": tipping}


def summarize_independent_review(rows, review_path):
    review = json.loads(review_path.read_text(encoding="utf-8"))
    baseline = {(row["deal"], row["product"]): row for row in rows}
    reviewed = review["products"]
    identifiers = [(row["deal"], row["product"]) for row in reviewed]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(baseline):
        raise ValueError("The independent review must cover every pilot line exactly once")
    for key, row in zip(identifiers, reviewed):
        first = baseline[key]
        if (row["original_code"], row["original_overlap"], row["original_deal_route"]) != (
            first["outcome"], first["overlap"], first["deal_outcome"]
        ):
            raise ValueError(f"Original coding differs between supplements: {key}")
        for code, dated in (("second_ai_code", "second_ai_evidence_date"),
                            ("reconciled_code", "reconciled_evidence_date")):
            if row[code] not in {SHIP, STOP, *UNCERTAIN}:
                raise ValueError(f"Unknown review state: {key}")
            if row[code] != "unresolved":
                lo, hi = evidence_interval(row[dated])
                if not date.fromisoformat(row["anchor"]) <= lo <= hi <= date.fromisoformat(row["window_end"]):
                    raise ValueError(f"Review evidence outside follow-up: {key}")
    summary = review["comparison_summary"]
    agreement = sum(row["original_code"] == row["second_ai_code"] for row in reviewed)
    if summary["same_shipment_state"] != agreement or summary["rows"] != len(reviewed):
        raise ValueError("Independent agreement count does not match the rows")
    print(f"\nIndependent AI review: {agreement}/{len(reviewed)} exact outcome matches")
    for field, label in (("original_code", "original"), ("second_ai_code", "second_ai"),
                         ("reconciled_code", "reconciled")):
        counts = Counter(row[field] for row in reviewed)
        if any(counts[state] != count for state, count in summary[label].items()):
            raise ValueError(f"Review summary differs from underlying rows: {label}")
        print(f"{label}: {dict(counts)}")
        for group in ("all", "direct"):
            for route in (CLOSED, FELL_THROUGH):
                subset = [r for r in reviewed if r["original_deal_route"] == route
                          and (group == "all" or r["original_overlap"] == group)]
                print(f"  {group}, {route}: {sum(r[field] == SHIP for r in subset)}/{len(subset)} confirmed")


if __name__ == "__main__":
    screen = load_ledger()["screening"]
    cases = screen["candidate_records"]
    print(f"{screen['register_count']} CMA merger records, {len(cases)} candidates")
    print(dict(Counter(case["decision"] for case in cases)))
    rows = list(load_rows())
    print(f"{len(rows)} product lines from {len({row['deal'] for row in rows})} deals")
    for label, table in contrasts(rows).items():
        print(f"\n{label}")
        for disposition, entry in table.items():
            print(disposition, json.dumps(entry, default=str))
        closed, abandoned = table[CLOSED], table[FELL_THROUGH]
        diff_lower = closed["lower"] - abandoned["upper"]
        diff_upper = closed["upper"] - abandoned["lower"]
        print(f"Closed minus fell through under missing-outcome bounds: "
              f"{100 * float(diff_lower):.1f} to {100 * float(diff_upper):.1f} pp")
    print("\nAdditional descriptive checks")
    print(json.dumps(descriptive_checks(rows), default=str, indent=2))
    review_path = PATH.with_name("acquisition-independent-review.json")
    if review_path.exists():
        summarize_independent_review(rows, review_path)

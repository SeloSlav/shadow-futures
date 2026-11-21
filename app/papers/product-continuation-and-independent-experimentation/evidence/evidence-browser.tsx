"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import type { AcquisitionEvidenceRow } from "@/lib/acquisition-evidence";
import { downloadText } from "@/lib/download-text";
import { evidenceCsv } from "./evidence-csv";
import styles from "./evidence.module.css";

const dealLabels = {
  closed: "Closed",
  "fell through": "Fell through",
};
const outcomeLabels = {
  "confirmed shipment": "Confirmed shipment",
  "active only": "Active only",
  "documented cessation": "Documented cessation",
  unresolved: "Unresolved",
};
const overlapLabels = {
  direct: "Direct overlap",
  potential: "Potential overlap",
  "no direct overlap established": "No direct overlap established",
};

type LedgerFilters = {
  overlap: "all" | keyof typeof overlapLabels;
  deal: "all" | keyof typeof dealLabels;
  shipment: "all" | keyof typeof outcomeLabels;
  q: string;
};

type LedgerSort = "ledger" | "product-name" | "evidence-date";
type LedgerView = LedgerFilters & { sort: LedgerSort };

const allFilters: LedgerFilters = { overlap: "all", deal: "all", shipment: "all", q: "" };
const dealOutcomes = Object.keys(dealLabels) as (keyof typeof dealLabels)[];
const overlapTypes = Object.keys(overlapLabels) as (keyof typeof overlapLabels)[];
const shipmentOutcomes = Object.keys(outcomeLabels) as (keyof typeof outcomeLabels)[];
const resultsHash = "#results";

function readFilter<T extends string>(value: string | null, options: readonly T[]): T | "all" {
  return options.find((option) => option.replaceAll(" ", "-") === value) ?? "all";
}

function viewHref(pathname: string, query: string, view: Partial<LedgerView>, hash: string) {
  const params = new URLSearchParams(query);
  const { q, sort, ...options } = view;
  for (const [key, value] of Object.entries(options)) {
    if (value === "all") params.delete(key);
    else params.set(key, value.replaceAll(" ", "-"));
  }
  if (q !== undefined) {
    const trimmed = q.trim();
    if (trimmed) params.set("q", trimmed);
    else params.delete("q");
  }
  if (sort === "ledger") params.delete("sort");
  else if (sort) params.set("sort", sort);
  const search = params.toString();
  return `${pathname}${search ? `?${search}` : ""}${hash}`;
}

function jumpToResults(target: HTMLParagraphElement | null) {
  target?.focus({ preventScroll: true });
  target?.scrollIntoView({ behavior: "instant", block: "start" });
}

const monthFormat = new Intl.DateTimeFormat("en-GB", {
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});
const dayFormat = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});

function EvidenceDate({ date }: { date: string }) {
  const monthOnly = date.length === 7;
  const parsed = new Date(monthOnly ? `${date}-01` : date);
  return <time dateTime={date}>{(monthOnly ? monthFormat : dayFormat).format(parsed)}</time>;
}

function SourceLink({ href, label, context }: { href: string; label: string; context?: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={context ? `${label} for ${context}` : undefined}
    >
      {label} <span aria-hidden="true">↗</span>
    </a>
  );
}

function EvidenceRow({ row }: { row: AcquisitionEvidenceRow }) {
  const { deal } = row;
  return (
    <article className={styles.row}>
      <div className={styles.product}>
        <h2>{row.product}</h2>
        <p className={styles.dealName}>{deal.name}</p>
        <p className={styles.meta}>
          <strong>{dealLabels[deal.original_outcome]}</strong>
          <span aria-hidden="true">·</span>
          {overlapLabels[row.overlap]}
        </p>
      </div>
      <div>
        <p className={styles.outcomeMeta}>
          <strong className={row.outcome === "confirmed shipment" ? styles.shipped : undefined}>
            {outcomeLabels[row.outcome]}
          </strong>
          {row.evidence_date && (
            <span>Evidence dated <EvidenceDate date={row.evidence_date} /></span>
          )}
        </p>
        <p className={styles.evidence}>{row.evidence}</p>
        <div className={styles.links}>
          <SourceLink href={row.outcome_source} label="Outcome source" context={row.product} />
          <SourceLink href={deal.case_source} label="CMA case" context={deal.name} />
        </div>
        <details className={styles.details}>
          <summary>Deal and source details</summary>
          <p>
            {dealLabels[deal.original_outcome]} on <EvidenceDate date={deal.anchor_date} />.{" "}
            {deal.ownership_and_remedies}
          </p>
          <p>{row.path}</p>
          <p>{row.overlap_reason}</p>
          {row.review_note && <p>{row.review_note}</p>}
          <ul className={styles.sourceList}>
            <li>
              <SourceLink
                href={deal.anchor_source}
                label={deal.original_outcome === "closed" ? "Closing record" : "Abandonment record"}
              />
            </li>
            <li><SourceLink href={row.overlap_source} label="Overlap source" /></li>
            {row.additional_source && (
              <li><SourceLink href={row.additional_source} label="Supporting evidence" /></li>
            )}
            {row.review_source && (
              <li><SourceLink href={row.review_source} label="Follow-up review source" /></li>
            )}
            {row.review_additional_source && (
              <li><SourceLink href={row.review_additional_source} label="Additional review source" /></li>
            )}
            {deal.ownership_sources.map((href, index, sources) => (
              <li key={href}>
                <SourceLink
                  href={href}
                  label={sources.length === 1 ? "Ownership record" : `Ownership record ${index + 1}`}
                />
              </li>
            ))}
          </ul>
        </details>
      </div>
    </article>
  );
}

function countConfirmedShipments(rows: AcquisitionEvidenceRow[]) {
  return rows.filter((row) => row.outcome === "confirmed shipment").length;
}

const productNameOrder = new Intl.Collator("en-GB", { sensitivity: "base", numeric: true });

function compareProductNames(a: AcquisitionEvidenceRow, b: AcquisitionEvidenceRow) {
  return productNameOrder.compare(a.product, b.product);
}

function compareEvidenceDates(a: AcquisitionEvidenceRow, b: AcquisitionEvidenceRow) {
  // ISO dates put month-only records after exact dates in the same month.
  return (b.evidence_date ?? "").localeCompare(a.evidence_date ?? "", "en-GB") || compareProductNames(a, b);
}

function ResultActions({ href, rows }: { href: string; rows: AcquisitionEvidenceRow[] }) {
  const [result, setResult] = useState<{ copied: true } | { copied: false; url: string } | null>(null);

  async function copyLink() {
    const url = new URL(href, window.location.origin).href;
    try {
      await navigator.clipboard.writeText(url);
      setResult({ copied: true });
    } catch {
      setResult({ copied: false, url });
    }
  }

  return (
    <>
      <div className={styles.resultActions}>
        <button type="button" className={styles.resultButton} onClick={copyLink}>Copy link</button>
        <button
          type="button"
          className={styles.resultButton}
          onClick={() => downloadText("acquisition-evidence.csv", evidenceCsv(rows), "text/csv;charset=utf-8")}
        >
          Download CSV
        </button>
        <span className={styles.copyNotice} aria-live="polite">
          {result && (result.copied ? "Link copied" : "Select and copy the link below.")}
        </span>
      </div>
      {result?.copied === false && (
        <input
          className={styles.manualLink}
          type="url"
          aria-label="Link to these results"
          readOnly
          value={result.url}
          onFocus={(event) => event.currentTarget.select()}
        />
      )}
    </>
  );
}

export default function EvidenceBrowser({ rows }: { rows: AcquisitionEvidenceRow[] }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [nameSearch, setNameSearch] = useState(() => searchParams.get("q") ?? "");
  const query = searchParams.toString();
  const availableShipments = shipmentOutcomes.filter((outcome) =>
    rows.some((row) => row.outcome === outcome),
  );
  const overlap = readFilter(searchParams.get("overlap"), overlapTypes);
  const deal = readFilter(searchParams.get("deal"), dealOutcomes);
  const shipment = readFilter(searchParams.get("shipment"), availableShipments);
  const requestedSort = searchParams.get("sort");
  const sort = requestedSort === "product-name" || requestedSort === "evidence-date" ? requestedSort : "ledger";
  const shareHref = viewHref(pathname, "", { overlap, deal, shipment, q: nameSearch, sort }, resultsHash);
  const resultCountRef = useRef<HTMLParagraphElement>(null);
  const pendingJump = useRef<string | null>(null);
  const searchTerms = nameSearch.toLowerCase().match(/\S+/g) ?? [];
  const matchingRows = rows.filter((row) => {
    if (overlap !== "all" && row.overlap !== overlap) return false;
    const names = `${row.product} ${row.deal.name}`.toLowerCase();
    return searchTerms.every((term) => names.includes(term));
  });
  const visibleRows = matchingRows.filter(
    (row) =>
      (deal === "all" || row.deal.original_outcome === deal) &&
      (shipment === "all" || row.outcome === shipment),
  );
  if (sort !== "ledger") visibleRows.sort(sort === "product-name" ? compareProductNames : compareEvidenceDates);
  const shipmentCount = countConfirmedShipments(visibleRows);

  useEffect(() => {
    if (window.location.hash === resultsHash) jumpToResults(resultCountRef.current);

    function restoreNameSearch() {
      const params = new URLSearchParams(window.location.search);
      setNameSearch(params.get("q") ?? "");
    }

    window.addEventListener("popstate", restoreNameSearch);
    return () => window.removeEventListener("popstate", restoreNameSearch);
  }, []);

  useEffect(() => {
    if (pendingJump.current === query) {
      pendingJump.current = null;
      jumpToResults(resultCountRef.current);
    }
  }, [query]);

  function changeView(view: Partial<LedgerView>, hash = window.location.hash) {
    pendingJump.current = null;
    const { pathname, search } = window.location;
    window.history.pushState(null, "", viewHref(pathname, search, view, hash));
  }

  function changeNameSearch(value: string) {
    setNameSearch(value);
    pendingJump.current = null;
    const { pathname, search, hash } = window.location;
    const href = viewHref(pathname, search, { q: value }, hash);
    if (href !== `${pathname}${search}${hash}`) window.history.replaceState(null, "", href);
  }

  function openShipmentView(href: string) {
    const destination = new URL(href, window.location.href);
    const nextQuery = destination.searchParams.toString();
    pendingJump.current = nextQuery === query ? null : nextQuery;
    if (destination.href !== window.location.href) window.history.pushState(null, "", href);
    if (pendingJump.current === null) jumpToResults(resultCountRef.current);
  }

  return (
    <>
      <label className={styles.nameSearch}>
        Search by product or deal name
        <input
          type="search"
          value={nameSearch}
          onChange={(event) => changeNameSearch(event.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
      </label>
      <section className={styles.comparison} aria-labelledby="shipment-comparison-title">
        <div className={styles.comparisonHeader}>
          <h2 id="shipment-comparison-title">Confirmed shipments</h2>
          <label>
            Overlap
            <select
              value={overlap}
              onChange={(event) => changeView({ overlap: event.target.value as LedgerFilters["overlap"] })}
            >
              <option value="all">All overlap types</option>
              {overlapTypes.map((value) => (
                <option key={value} value={value}>{overlapLabels[value]}</option>
              ))}
            </select>
          </label>
        </div>
        <dl className={styles.comparisonCounts} aria-live="polite" aria-atomic="true">
          {dealOutcomes.map((outcome) => {
            const cohort = matchingRows.filter((row) => row.deal.original_outcome === outcome);
            const shipped = countConfirmedShipments(cohort);
            const count = cohort.length > 0 ? `${shipped} of ${cohort.length}` : "No product lines";
            const href = viewHref(pathname, query, {
              overlap,
              deal: outcome,
              shipment: "confirmed shipment",
              q: nameSearch,
              sort,
            }, resultsHash);
            return (
              <div key={outcome}>
                <dt>{dealLabels[outcome]}</dt>
                <dd>
                  {shipped > 0 ? (
                    <Link
                      href={href}
                      prefetch={false}
                      className={styles.countAction}
                      aria-label={`${count} product lines with confirmed shipments for deals that ${outcome}. View matching products.`}
                      onNavigate={(event) => {
                        event.preventDefault();
                        openShipmentView(href);
                      }}
                    >
                      <span>{count}</span>
                      <span aria-hidden="true">↓</span>
                    </Link>
                  ) : count}
                </dd>
              </div>
            );
          })}
        </dl>
        <p className={styles.legend}>
          Totals reflect the name search and overlap filter.
          Select a shipment count to jump to the matching products.
        </p>
      </section>
      <div className={styles.controls}>
        <label>
          Deal outcome
          <select
            value={deal}
            onChange={(event) => changeView({ deal: event.target.value as LedgerFilters["deal"] })}
          >
            <option value="all">All deals</option>
            {dealOutcomes.map((outcome) => (
              <option key={outcome} value={outcome}>{dealLabels[outcome]}</option>
            ))}
          </select>
        </label>
        <label>
          Product outcome
          <select
            value={shipment}
            onChange={(event) => changeView({ shipment: event.target.value as LedgerFilters["shipment"] })}
          >
            <option value="all">All product outcomes</option>
            {availableShipments.map((value) => (
              <option key={value} value={value}>{outcomeLabels[value]}</option>
            ))}
          </select>
        </label>
        <label className={styles.sortControl}>
          Sort by
          <select value={sort} onChange={(event) => changeView({ sort: event.target.value as LedgerSort })}>
            <option value="ledger">Ledger order</option>
            <option value="product-name">Product name (A–Z)</option>
            <option value="evidence-date">Evidence date (newest first)</option>
          </select>
        </label>
      </div>
      <p className={styles.legend}>
        <strong>Active only</strong> means continued use or maintenance was documented.
        A material release or rollout remains unverified.
      </p>
      <div className={styles.resultHeader}>
        <p id={resultsHash.slice(1)} className={styles.resultCount} role="status" ref={resultCountRef} tabIndex={-1}>
          Showing {visibleRows.length} of {rows.length} product lines
          {" · "}{shipmentCount} confirmed {shipmentCount === 1 ? "shipment" : "shipments"}
        </p>
        <ResultActions key={shareHref} href={shareHref} rows={visibleRows} />
      </div>
      <div className={styles.results}>
        {visibleRows.length === 0 ? (
          <div className={styles.empty}>
            <p>No product lines match these filters.</p>
            <button
              type="button"
              onClick={() => {
                setNameSearch("");
                changeView(allFilters, "");
              }}
            >
              Clear filters
            </button>
          </div>
        ) : visibleRows.map((row) => <EvidenceRow key={row.id} row={row} />)}
      </div>
    </>
  );
}

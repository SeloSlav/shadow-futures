"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useRef } from "react";

import type { AcquisitionEvidenceRow } from "@/lib/acquisition-evidence";
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
};

const allFilters: LedgerFilters = { overlap: "all", deal: "all", shipment: "all" };
const dealOutcomes = Object.keys(dealLabels) as (keyof typeof dealLabels)[];
const overlapTypes = Object.keys(overlapLabels) as (keyof typeof overlapLabels)[];
const shipmentOutcomes = Object.keys(outcomeLabels) as (keyof typeof outcomeLabels)[];
const resultsHash = "#results";

function readFilter<T extends string>(value: string | null, options: readonly T[]): T | "all" {
  return options.find((option) => option.replaceAll(" ", "-") === value) ?? "all";
}

function viewHref(pathname: string, query: string, filters: Partial<LedgerFilters>, hash: string) {
  const params = new URLSearchParams(query);
  for (const [key, value] of Object.entries(filters)) {
    if (value === "all") params.delete(key);
    else params.set(key, value.replaceAll(" ", "-"));
  }
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

export default function EvidenceBrowser({ rows }: { rows: AcquisitionEvidenceRow[] }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const query = searchParams.toString();
  const availableShipments = shipmentOutcomes.filter((outcome) =>
    rows.some((row) => row.outcome === outcome),
  );
  const overlap = readFilter(searchParams.get("overlap"), overlapTypes);
  const deal = readFilter(searchParams.get("deal"), dealOutcomes);
  const shipment = readFilter(searchParams.get("shipment"), availableShipments);
  const resultCountRef = useRef<HTMLParagraphElement>(null);
  const pendingJump = useRef<string | null>(null);
  const overlapRows = overlap === "all" ? rows : rows.filter((row) => row.overlap === overlap);
  const visibleRows = overlapRows.filter(
    (row) =>
      (deal === "all" || row.deal.original_outcome === deal) &&
      (shipment === "all" || row.outcome === shipment),
  );
  const shipmentCount = countConfirmedShipments(visibleRows);

  useEffect(() => {
    if (window.location.hash === resultsHash) jumpToResults(resultCountRef.current);
  }, []);

  useEffect(() => {
    if (pendingJump.current === query) {
      pendingJump.current = null;
      jumpToResults(resultCountRef.current);
    }
  }, [query]);

  function changeFilters(filters: Partial<LedgerFilters>, hash = window.location.hash) {
    pendingJump.current = null;
    const { pathname, search } = window.location;
    window.history.pushState(null, "", viewHref(pathname, search, filters, hash));
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
      <section className={styles.comparison} aria-labelledby="shipment-comparison-title">
        <div className={styles.comparisonHeader}>
          <h2 id="shipment-comparison-title">Confirmed shipments</h2>
          <label>
            Overlap
            <select
              value={overlap}
              onChange={(event) => changeFilters({ overlap: event.target.value as LedgerFilters["overlap"] })}
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
            const cohort = overlapRows.filter((row) => row.deal.original_outcome === outcome);
            const shipped = countConfirmedShipments(cohort);
            const count = cohort.length > 0 ? `${shipped} of ${cohort.length}` : "No product lines";
            const href = viewHref(pathname, query, { overlap, deal: outcome, shipment: "confirmed shipment" }, resultsHash);
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
          Totals include all tracked product lines in the selected overlap group.
          Select a shipment count to jump to the matching products. Copy the address to share the view.
        </p>
      </section>
      <div className={styles.controls}>
        <label>
          Deal outcome
          <select
            value={deal}
            onChange={(event) => changeFilters({ deal: event.target.value as LedgerFilters["deal"] })}
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
            onChange={(event) => changeFilters({ shipment: event.target.value as LedgerFilters["shipment"] })}
          >
            <option value="all">All product outcomes</option>
            {availableShipments.map((value) => (
              <option key={value} value={value}>{outcomeLabels[value]}</option>
            ))}
          </select>
        </label>
      </div>
      <p className={styles.legend}>
        <strong>Active only</strong> means continued use or maintenance was documented.
        A material release or rollout remains unverified.
      </p>
      <p id={resultsHash.slice(1)} className={styles.resultCount} role="status" ref={resultCountRef} tabIndex={-1}>
        Showing {visibleRows.length} of {rows.length} product lines
        {" · "}{shipmentCount} confirmed {shipmentCount === 1 ? "shipment" : "shipments"}
      </p>
      <div className={styles.results}>
        {visibleRows.length === 0 ? (
          <div className={styles.empty}>
            <p>No product lines match these filters.</p>
            <button
              type="button"
              onClick={() => changeFilters(allFilters, "")}
            >
              Clear filters
            </button>
          </div>
        ) : visibleRows.map((row) => <EvidenceRow key={row.id} row={row} />)}
      </div>
    </>
  );
}

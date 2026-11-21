import type { AcquisitionEvidenceRow } from "@/lib/acquisition-evidence";

const columns: [string, (row: AcquisitionEvidenceRow) => string | null | undefined][] = [
  ["product", (row) => row.product],
  ["deal", (row) => row.deal.name],
  ["deal_outcome", (row) => row.deal.original_outcome],
  ["deal_date", (row) => row.deal.anchor_date],
  ["overlap", (row) => row.overlap],
  ["product_outcome", (row) => row.outcome],
  ["evidence_date", (row) => row.evidence_date],
  ["evidence", (row) => row.evidence],
  ["overlap_reason", (row) => row.overlap_reason],
  ["product_path", (row) => row.path],
  ["ownership_and_remedies", (row) => row.deal.ownership_and_remedies],
  ["review_note", (row) => row.review_note],
  ["outcome_source_url", (row) => row.outcome_source],
  ["cma_case_url", (row) => row.deal.case_source],
  ["deal_record_url", (row) => row.deal.anchor_source],
  ["overlap_source_url", (row) => row.overlap_source],
  ["supporting_evidence_url", (row) => row.additional_source],
  ["review_source_url", (row) => row.review_source],
  ["additional_review_source_url", (row) => row.review_additional_source],
  ["ownership_source_urls", (row) => row.deal.ownership_sources.join("\n")],
];

function csvCell(value = "") {
  // Prevent spreadsheet applications from treating evidence text as a formula.
  const text = /^[\t\r\n]|^\s*[=+@-]/.test(value) ? `'${value}` : value;
  return `"${text.replaceAll('"', '""')}"`;
}

export function evidenceCsv(rows: readonly AcquisitionEvidenceRow[]) {
  const header = columns.map(([name]) => csvCell(name)).join(",");
  const records = rows.map((row) => columns.map(([, read]) => csvCell(read(row) ?? "")).join(","));
  return `\uFEFF${[header, ...records].join("\r\n")}\r\n`;
}

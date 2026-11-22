import type { Metadata } from "next";
import Link from "next/link";
import { connection } from "next/server";

import { loadAcquisitionEvidence } from "@/lib/acquisition-evidence";
import { ACQUISITION_PAPER } from "@/lib/papers";
import EvidenceBrowser from "./evidence-browser";
import styles from "./evidence.module.css";

const description =
  "Browse product-line outcomes after UK software and digital marketplace deals closed or fell through, with links to the records behind each outcome.";

export const metadata: Metadata = {
  title: "Acquisition evidence ledger",
  description,
  alternates: { canonical: ACQUISITION_PAPER.evidencePath },
};

export default async function AcquisitionEvidencePage() {
  await connection();
  const { rows, dealCount } = loadAcquisitionEvidence();

  return (
    <main id="main-content" className={styles.page}>
      <div className={styles.inner}>
        <header className={styles.header}>
          <Link href={`/papers#${ACQUISITION_PAPER.id}`} className={styles.backLink}>
            <span aria-hidden="true">←</span> Research papers
          </Link>
          <h1>Acquisition evidence ledger</h1>
          <p>
            {rows.length} product lines across {dealCount} deals reviewed by the UK Competition
            and Markets Authority, followed for two years from closing or public abandonment.
            The ledger shows the original coding. A separate AI review agreed on 27 of 29 lines.
            Reconciliation added one confirmed shipment, bringing the total to 21.
          </p>
          <div className={styles.links}>
            <a href={ACQUISITION_PAPER.pdfPath}>Read Paper 06 (PDF)</a>
            <a href={ACQUISITION_PAPER.ledgerPath} download>Original ledger (JSON)</a>
            <a href={ACQUISITION_PAPER.reviewPath} download>Second AI coding and reconciliation (JSON)</a>
          </div>
        </header>
        <EvidenceBrowser rows={rows} />
      </div>
    </main>
  );
}

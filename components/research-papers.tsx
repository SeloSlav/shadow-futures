import Link from "next/link";

import {
  COMPANION_PAPERS,
  MAIN_RESEARCH_PAPER,
  RESEARCH_AUTHOR,
  RESEARCH_REVISION,
  type ResearchPaper,
} from "@/lib/papers";
import { AUTHOR } from "@/lib/paper/citation";
import styles from "@/components/research-papers.module.css";

function PaperActions({ paper }: { paper: ResearchPaper }) {
  return (
    <div className={styles.actions}>
      <a
        className={styles.viewLink}
        href={paper.pdfPath}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`View PDF: ${paper.title}`}
      >
        View PDF <span aria-hidden="true">↗</span>
      </a>
      <a
        className={styles.downloadLink}
        href={paper.pdfPath}
        download
        aria-label={`Download PDF: ${paper.title}`}
      >
        Download <span aria-hidden="true">↓</span>
      </a>
    </div>
  );
}

export function MainPaperCard() {
  return (
    <article className={styles.featured} aria-labelledby="foundational-paper-title">
      <div className={styles.featuredLabel}>
        <span className={styles.eyebrow}>{MAIN_RESEARCH_PAPER.focus}</span>
        <span className={styles.status}>{MAIN_RESEARCH_PAPER.status}</span>
        <span className={styles.pages}>{MAIN_RESEARCH_PAPER.pages} pages · PDF</span>
      </div>
      <div>
        <h2 id="foundational-paper-title">{MAIN_RESEARCH_PAPER.title}</h2>
        <p>{MAIN_RESEARCH_PAPER.description}</p>
        <PaperActions paper={MAIN_RESEARCH_PAPER} />
        <Link className={styles.detailsLink} href="/paper">
          Abstract, citation and source document <span aria-hidden="true">→</span>
        </Link>
      </div>
    </article>
  );
}

export function CompanionPaperList() {
  return (
    <div className={styles.grid}>
      {COMPANION_PAPERS.map((paper, index) => (
        <article className={styles.card} id={paper.id} key={paper.id}>
          <div className={styles.cardMeta}>
            <span className={styles.number}>{String(index + 1).padStart(2, "0")}</span>
            <span className={styles.focus}>{paper.focus}</span>
          </div>
          <h3>{paper.title}</h3>
          <p>{paper.description}</p>
          <div className={styles.fileMeta}>
            <span>{paper.status}</span>
            <span>{paper.pages} pages · PDF</span>
          </div>
          <PaperActions paper={paper} />
        </article>
      ))}
    </div>
  );
}

export function ResearchPapers() {
  return (
    <section className={styles.homeSection} id="research-papers" aria-labelledby="research-papers-title">
      <div className={styles.inner}>
        <div className={styles.sectionHeader}>
          <div>
            <p className={styles.eyebrow}>The research program</p>
            <h2 id="research-papers-title">Research papers</h2>
          </div>
          <div className={styles.sectionIntro}>
            <p>
              Six companion papers develop the institutional, statistical and policy questions
              opened by Shadow Futures: who can enter, what can be learned, and when a comparison
              is worth creating.
            </p>
            <p className={styles.byline}>
              <Link href={AUTHOR.path}>{RESEARCH_AUTHOR}</Link> · {RESEARCH_REVISION} research drafts
            </p>
            <Link className={styles.catalogueLink} href="/papers">
              See all papers <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
        <CompanionPaperList />
      </div>
    </section>
  );
}

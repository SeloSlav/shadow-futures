import type { Metadata } from "next";
import Link from "next/link";

import { CompanionPaperList, MainPaperCard } from "@/components/research-papers";
import {
  COMPANION_PAPERS,
  MAIN_RESEARCH_PAPER,
  RESEARCH_AUTHOR,
  RESEARCH_PUBLISHED_DATE,
  RESEARCH_REVISION,
} from "@/lib/papers";
import { AUTHOR } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";
import styles from "@/app/papers/page.module.css";

const description =
  "Read Shadow Futures and six companion research drafts on contestability, measurement, counterfactuals, institutions, competition policy and corrective instruments.";

export const metadata: Metadata = {
  title: "Research papers",
  description,
  authors: [{ name: RESEARCH_AUTHOR, url: AUTHOR.path }],
  alternates: { canonical: "/papers" },
  openGraph: {
    type: "website",
    title: "Research papers | Shadow Futures",
    description,
    url: "/papers",
  },
  twitter: {
    card: "summary_large_image",
    title: "Research papers | Shadow Futures",
    description,
  },
};

export default function PapersPage() {
  const papers = [MAIN_RESEARCH_PAPER, ...COMPANION_PAPERS];
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    "@id": `${SITE_ORIGIN}/papers#webpage`,
    url: `${SITE_ORIGIN}/papers`,
    name: "Shadow Futures research papers",
    description,
    dateModified: RESEARCH_PUBLISHED_DATE,
    author: {
      "@type": "Person",
      name: RESEARCH_AUTHOR,
      url: `${SITE_ORIGIN}${AUTHOR.path}`,
    },
    mainEntity: {
      "@type": "ItemList",
      numberOfItems: papers.length,
      itemListElement: papers.map((paper, index) => ({
        "@type": "ListItem",
        position: index + 1,
        item: {
          "@type": "ScholarlyArticle",
          name: paper.title,
          description: paper.description,
          url: `${SITE_ORIGIN}${paper.pdfPath}`,
          creativeWorkStatus: paper.status,
          author: { "@type": "Person", name: RESEARCH_AUTHOR },
          encoding: {
            "@type": "MediaObject",
            contentUrl: `${SITE_ORIGIN}${paper.pdfPath}`,
            encodingFormat: "application/pdf",
          },
          inLanguage: "en",
        },
      })),
    },
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(structuredData).replace(/</g, "\\u003c"),
        }}
      />
      <main className={styles.page} id="main-content">
        <div className={styles.inner}>
          <header className={styles.header}>
            <p className={styles.eyebrow}>Shadow Futures / The research program</p>
            <h1>Research papers</h1>
            <p className={styles.dek}>
              From the limits of one market history to the institutions that can create
              worthwhile comparisons.
            </p>
            <p className={styles.byline}>
              By <Link href={AUTHOR.path}>{RESEARCH_AUTHOR}</Link> · {RESEARCH_REVISION}
            </p>
            <p className={styles.note}>
              The original paper and six companion manuscripts are available below. The
              companions are research drafts, published on this site for scholarly review.
            </p>
            <a className={styles.jumpLink} href="#companion-papers">
              Explore the six companion papers <span aria-hidden="true">↓</span>
            </a>
          </header>

          <MainPaperCard />

          <section id="companion-papers" className={styles.companions} aria-labelledby="companions-title">
            <div className={styles.companionHeader}>
              <h2 id="companions-title">Companion papers</h2>
              <p>October 2026 research drafts · PDFs to read or download</p>
            </div>
            <CompanionPaperList />
          </section>

          <aside className={styles.reviewNote}>
            <p>
              Each paper states its model, evidence requirements and contribution separately.
              The comparison-budget theorem motivates the program; the institutional and
              policy conclusions depend on additional assumptions developed in each draft.
            </p>
            <Link href="/paper#contribute">
              Contribute to the formal proof request on Hunchroom <span aria-hidden="true">→</span>
            </Link>
          </aside>
        </div>
      </main>
    </>
  );
}

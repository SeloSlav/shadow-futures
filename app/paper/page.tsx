import type { Metadata } from "next";
import Link from "next/link";

import { AUTHOR, PAPER } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

const pageUrl = `${SITE_ORIGIN}${PAPER.landingPath}`;
const authorUrl = `${SITE_ORIGIN}${AUTHOR.path}`;

export const metadata: Metadata = {
  title: PAPER.title,
  description: PAPER.abstract,
  authors: [{ name: AUTHOR.name, url: AUTHOR.path }],
  keywords: [...PAPER.keywords],
  alternates: { canonical: PAPER.landingPath },
  openGraph: {
    type: "article",
    title: PAPER.title,
    description: PAPER.abstract,
    url: PAPER.landingPath,
    authors: [AUTHOR.name],
    publishedTime: PAPER.publishedDate,
    modifiedTime: PAPER.modifiedDate,
  },
  twitter: {
    card: "summary_large_image",
    title: PAPER.title,
    description:
      "A formal account of why self-reinforcing markets can exhaust the comparisons needed to identify contribution.",
  },
  other: {
    citation_title: PAPER.title,
    citation_author: AUTHOR.name,
    citation_publication_date: "2025/12",
    citation_pdf_url: `${SITE_ORIGIN}${PAPER.pdfPath}`,
    citation_abstract: PAPER.abstract,
    citation_doi: PAPER.doi,
    citation_keywords: PAPER.keywords.join("; "),
    "DC.title": PAPER.title,
    "DC.creator": AUTHOR.name,
    "DC.issued": PAPER.publishedDate,
    "DC.identifier": PAPER.doiUrl,
  },
};

export default function PaperPage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        "@id": `${pageUrl}#webpage`,
        url: pageUrl,
        name: PAPER.title,
        description: PAPER.abstract,
        isPartOf: { "@id": `${SITE_ORIGIN}/#website` },
        mainEntity: { "@id": `${pageUrl}#article` },
        inLanguage: "en",
      },
      {
        "@type": "ScholarlyArticle",
        "@id": `${pageUrl}#article`,
        url: pageUrl,
        headline: PAPER.title,
        name: PAPER.title,
        description: PAPER.abstract,
        abstract: PAPER.abstract,
        author: {
          "@type": "Person",
          "@id": `${authorUrl}#person`,
          name: AUTHOR.name,
          url: authorUrl,
        },
        datePublished: PAPER.publishedDate,
        dateModified: PAPER.modifiedDate,
        keywords: PAPER.keywords,
        identifier: {
          "@type": "PropertyValue",
          propertyID: "DOI",
          value: PAPER.doi,
        },
        sameAs: [PAPER.doiUrl, PAPER.ssrnUrl],
        encoding: {
          "@type": "MediaObject",
          contentUrl: `${SITE_ORIGIN}${PAPER.pdfPath}`,
          encodingFormat: "application/pdf",
        },
        inLanguage: "en",
      },
    ],
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(structuredData).replace(/</g, "\\u003c"),
        }}
      />
      <main className="method-page paper-page" id="main-content">
        <article>
          <header className="method-page__header paper-hero">
            <p className="eyebrow">Research paper</p>
            <h1>{PAPER.title}</h1>
            <p className="paper-byline">
              By <Link href={AUTHOR.path}>{AUTHOR.name}</Link> · First posted{" "}
              {PAPER.firstPosted} · Revised {PAPER.revised}
            </p>
            <p className="paper-dek">
              A formal account of why a market can observe productive inputs, reward them,
              and still exhaust the comparisons needed to identify what they contributed.
            </p>
            <div className="button-row method-page__actions">
              <a className="button button--primary" href={PAPER.pdfPath}>
                Download the PDF
              </a>
              <a className="button" href={PAPER.sourcePath}>
                Download the DOCX
              </a>
              <a
                className="button button--rust"
                href={PAPER.ssrnUrl}
                target="_blank"
                rel="noreferrer"
              >
                View on SSRN <span aria-hidden="true">↗</span>
              </a>
            </div>
          </header>

          <section className="paper-section" aria-labelledby="paper-abstract">
            <div className="paper-section__label">
              <span>Abstract</span>
              <h2 id="paper-abstract">The research question and result</h2>
            </div>
            <div className="paper-section__content">
              <p className="paper-abstract">{PAPER.abstract}</p>
            </div>
          </section>

          <section className="paper-section" aria-labelledby="paper-details">
            <div className="paper-section__label">
              <span>Indexing details</span>
              <h2 id="paper-details">Research metadata</h2>
            </div>
            <div className="paper-section__content paper-metadata">
              <dl>
                <div>
                  <dt>Author</dt>
                  <dd>
                    <Link href={AUTHOR.path}>{AUTHOR.name}</Link>
                  </dd>
                </div>
                <div>
                  <dt>DOI</dt>
                  <dd>
                    <a href={PAPER.doiUrl} target="_blank" rel="noreferrer">
                      {PAPER.doi}
                    </a>
                  </dd>
                </div>
                <div>
                  <dt>SSRN</dt>
                  <dd>
                    <a href={PAPER.ssrnUrl} target="_blank" rel="noreferrer">
                      Abstract 6003994
                    </a>
                  </dd>
                </div>
                <div>
                  <dt>Version</dt>
                  <dd>
                    First posted {PAPER.firstPosted}; revised {PAPER.revised}
                  </dd>
                </div>
                <div>
                  <dt>Keywords</dt>
                  <dd>{PAPER.keywords.join("; ")}</dd>
                </div>
                <div>
                  <dt>JEL codes</dt>
                  <dd>{PAPER.jel.join("; ")}</dd>
                </div>
              </dl>
            </div>
          </section>

          <section className="paper-section" aria-labelledby="paper-result">
            <div className="paper-section__label">
              <span>Central result</span>
              <h2 id="paper-result">Comparison, not transaction count</h2>
            </div>
            <div className="paper-section__content paper-callout">
              <p>
                The comparison budget is the cumulative probability mass remaining outside
                the currently dominant alternative. Under the paper&apos;s conditions, finite
                total comparison makes distinct contribution parameters generate mutually
                absolutely continuous complete-history laws.
              </p>
              <p>
                The implication is an identification limit: no estimator using one realized
                market can consistently recover every nonconstant contribution functional,
                and no test can separate two contribution parameters with vanishing total
                error. Strong reinforcement is one sharp corollary, not the definition of the
                general result.
              </p>
            </div>
          </section>

          <section className="paper-section paper-section--last" aria-labelledby="paper-citation">
            <div className="paper-section__label">
              <span>Citation</span>
              <h2 id="paper-citation">Cite this paper</h2>
            </div>
            <div className="paper-section__content">
              <p className="paper-citation">{PAPER.citation}</p>
              <details className="method-details paper-bibtex">
                <summary>BibTeX</summary>
                <pre>
                  <code>{PAPER.bibtex}</code>
                </pre>
              </details>
              <p className="paper-machine-links">
                Machine-readable versions: <a href="/paper.md">paper.md</a>,{" "}
                <a href="/llms.txt">llms.txt</a>, and{" "}
                <a href="/llms-full.txt">llms-full.txt</a>. Citation:{" "}
                <a href="/citation.bib">BibTeX file</a>.
              </p>
            </div>
          </section>
        </article>
      </main>
    </>
  );
}

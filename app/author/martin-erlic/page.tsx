import type { Metadata } from "next";
import Link from "next/link";

import { AUTHOR, PAPER } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

const pageUrl = `${SITE_ORIGIN}${AUTHOR.path}`;

export const metadata: Metadata = {
  title: `${AUTHOR.name} — Author and researcher`,
  description:
    "Martin Erlic is an independent author and researcher writing about systems, economics, self-reinforcing markets, and the future.",
  authors: [{ name: AUTHOR.name, url: AUTHOR.path }],
  alternates: { canonical: AUTHOR.path },
  openGraph: {
    type: "profile",
    title: `${AUTHOR.name} — Author of Shadow Futures`,
    description:
      "Independent author and researcher writing about systems, economics, self-reinforcing markets, and the future.",
    url: AUTHOR.path,
  },
};

export default function MartinErlicPage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "ProfilePage",
        "@id": `${pageUrl}#profile`,
        url: pageUrl,
        name: `${AUTHOR.name} — Author and researcher`,
        mainEntity: { "@id": `${pageUrl}#person` },
        isPartOf: { "@id": `${SITE_ORIGIN}/#website` },
        inLanguage: "en",
      },
      {
        "@type": "Person",
        "@id": `${pageUrl}#person`,
        name: AUTHOR.name,
        url: pageUrl,
        description:
          "Independent author and researcher writing about systems, economics, self-reinforcing markets, and the future.",
        sameAs: [AUTHOR.ssrnUrl, AUTHOR.mediumUrl, AUTHOR.xUrl],
        subjectOf: { "@id": `${SITE_ORIGIN}${PAPER.landingPath}#article` },
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
      <main className="method-page author-page" id="main-content">
        <header className="method-page__header">
          <p className="eyebrow">Author and researcher</p>
          <h1>{AUTHOR.name}</h1>
          <p>
            Martin Erlic is an independent author and researcher writing about systems,
            economics, self-reinforcing markets, and the future.
          </p>
        </header>

        <section className="paper-section" aria-labelledby="author-work">
          <div className="paper-section__label">
            <span>Research</span>
            <h2 id="author-work">Shadow Futures</h2>
          </div>
          <div className="paper-section__content">
            <h3>{PAPER.title}</h3>
            <p>
              The paper studies contribution uncertainty in adaptive allocation systems:
              markets where verified inputs affect rewards while past reward changes future
              exposure.
            </p>
            <div className="button-row author-page__actions">
              <Link className="button button--primary" href={PAPER.landingPath}>
                Read the paper
              </Link>
              <a className="button" href={PAPER.ssrnUrl} target="_blank" rel="noreferrer">
                SSRN profile <span aria-hidden="true">↗</span>
              </a>
            </div>
          </div>
        </section>

        <section className="paper-section paper-section--last" aria-labelledby="author-profiles">
          <div className="paper-section__label">
            <span>Elsewhere</span>
            <h2 id="author-profiles">Verified profiles</h2>
          </div>
          <div className="paper-section__content author-links">
            <a href={AUTHOR.ssrnUrl} target="_blank" rel="noreferrer">
              SSRN author page <span aria-hidden="true">↗</span>
            </a>
            <a href={AUTHOR.mediumUrl} target="_blank" rel="noreferrer">
              Medium <span aria-hidden="true">↗</span>
            </a>
            <a href={AUTHOR.xUrl} target="_blank" rel="noreferrer">
              X <span aria-hidden="true">↗</span>
            </a>
          </div>
        </section>
      </main>
    </>
  );
}

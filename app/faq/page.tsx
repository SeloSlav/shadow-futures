import type { Metadata } from "next";
import Link from "next/link";

import { FAQ_GROUPS } from "@/lib/paper/faq";
import { PAPER } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

export const metadata: Metadata = {
  title: "FAQ: Shadow Futures, Contribution Uncertainty, AI, Tax and Competition",
  description:
    "Direct answers about Shadow Futures, contribution uncertainty, AI agents, x402, the agentic economy, antitrust, progressive taxation, UBI, and social dividends.",
  alternates: { canonical: "/faq" },
  openGraph: {
    type: "article",
    title: "Shadow Futures FAQ",
    description:
      "What self-reinforcing markets erase and what the result means for AI agents, x402, competition, progressive taxation, UBI, and redistribution.",
    url: "/faq",
  },
  twitter: {
    card: "summary_large_image",
    title: "Shadow Futures FAQ",
    description:
      "Direct answers about contribution uncertainty in self-reinforcing markets.",
  },
};

export default function FaqPage() {
  const pageUrl = new URL("/faq", SITE_ORIGIN).toString();
  const faqJsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    "@id": `${pageUrl}#faq`,
    url: pageUrl,
    name: "Shadow Futures FAQ",
    description:
      "Questions and answers about contribution uncertainty and self-reinforcing markets.",
    inLanguage: "en",
    author: {
      "@type": "Person",
      name: PAPER.author,
    },
    about: {
      "@type": "ScholarlyArticle",
      name: PAPER.title,
      author: {
        "@type": "Person",
        name: PAPER.author,
      },
      url: PAPER.url,
    },
    mainEntity: FAQ_GROUPS.flatMap((group) =>
      group.entries.map((entry) => ({
        "@type": "Question",
        "@id": `${pageUrl}#${entry.id}`,
        name: entry.question,
        acceptedAnswer: {
          "@type": "Answer",
          text: entry.answer.join("\n\n"),
        },
      })),
    ),
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(faqJsonLd).replace(/</g, "\\u003c"),
        }}
      />
      <main className="method-page faq-page" id="main-content">
        <header className="method-page__header faq-hero">
          <p className="eyebrow">Questions and direct answers</p>
          <h1>Shadow Futures, explained</h1>
          <p>
            What self-reinforcing markets erase, why a million transactions may still be one
            experiment, and what the result means for creators, firms, AI agents, x402,
            competition, progressive taxation, UBI, and social dividends.
          </p>
          <div className="faq-lead">
            <span className="panel__meta">The shortest answer</span>
            <strong>
              Markets do not only distribute rewards. Their rules decide whether society gets
              enough independent histories to explain those rewards.
            </strong>
          </div>
          <div className="button-row method-page__actions">
            <Link className="button button--primary" href="/#shadow-futures">
              See the visual explanation
            </Link>
            <a className="button button--rust" href={PAPER.url} target="_blank" rel="noreferrer">
              Read the paper
            </a>
          </div>
        </header>

        <nav className="faq-jump" aria-label="FAQ topics">
          {FAQ_GROUPS.map((group) => (
            <a href={`#${group.id}`} key={group.id}>
              {group.title}
            </a>
          ))}
        </nav>

        {FAQ_GROUPS.map((group, groupIndex) => (
          <section
            className={`method-section faq-section${
              groupIndex === FAQ_GROUPS.length - 1 ? " method-section--last" : ""
            }`}
            id={group.id}
            key={group.id}
            aria-labelledby={`${group.id}-title`}
          >
            <div className="method-section__label">
              <span>{group.label}</span>
              <h2 id={`${group.id}-title`}>{group.title}</h2>
              <p>{group.intro}</p>
            </div>
            <div className="method-section__content faq-list">
              {group.entries.map((entry) => (
                <article className="faq-item" id={entry.id} key={entry.id}>
                  <h3>{entry.question}</h3>
                  {entry.answer.map((paragraph, paragraphIndex) => {
                    const inlineLink =
                      entry.inlineLink?.paragraphIndex === paragraphIndex
                        ? entry.inlineLink
                        : undefined;
                    const linkStart = inlineLink
                      ? paragraph.indexOf(inlineLink.text)
                      : -1;

                    if (!inlineLink || linkStart < 0) {
                      return <p key={`${entry.id}-${paragraphIndex}`}>{paragraph}</p>;
                    }

                    return (
                      <p key={`${entry.id}-${paragraphIndex}`}>
                        {paragraph.slice(0, linkStart)}
                        <a href={inlineLink.href} target="_blank" rel="noreferrer">
                          {inlineLink.text}
                        </a>
                        {paragraph.slice(linkStart + inlineLink.text.length)}
                      </p>
                    );
                  })}
                  <a className="faq-permalink" href={`#${entry.id}`} aria-label={`Link to: ${entry.question}`}>
                    Direct link
                  </a>
                </article>
              ))}
            </div>
          </section>
        ))}

        <aside className="faq-citation" aria-labelledby="faq-citation-title">
          <div>
            <span className="panel__meta">Primary source</span>
            <h2 id="faq-citation-title">{PAPER.title}</h2>
            <p>
              Martin Erlic · First posted {PAPER.firstPosted} · Revised {PAPER.revised}
            </p>
          </div>
          <a className="button button--rust" href={PAPER.url} target="_blank" rel="noreferrer">
            Open on SSRN <span aria-hidden="true">↗</span>
          </a>
        </aside>
      </main>
    </>
  );
}

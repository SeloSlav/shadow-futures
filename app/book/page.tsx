import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

import { BOOK, BOOK_CHAPTERS, BOOK_DESCRIPTION, BOOK_FULL_TITLE, BOOK_PDF_SIZE, BOOK_SYNOPSIS } from "@/lib/book";
import { BOOK_SCHEMA } from "@/lib/book-schema";
import { AUTHOR, PAPER } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: { absolute: "Shadow Futures by Martin Erlic — Free Book and PDF" },
  description: BOOK_DESCRIPTION,
  alternates: {
    canonical: "/book",
    types: { "text/markdown": "/book.md" },
  },
  openGraph: {
    type: "book",
    title: BOOK_FULL_TITLE,
    description: BOOK_DESCRIPTION,
    url: "/book",
    siteName: "Shadow Futures",
    authors: [AUTHOR.name],
  },
  twitter: {
    card: "summary_large_image",
    title: BOOK_FULL_TITLE,
    description: BOOK_DESCRIPTION,
  },
};

export default function BookPage() {
  const schema = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        "@id": `${SITE_ORIGIN}/book#webpage`,
        url: `${SITE_ORIGIN}/book`,
        name: BOOK_FULL_TITLE,
        description: BOOK_DESCRIPTION,
        isPartOf: { "@id": `${SITE_ORIGIN}/#website` },
        mainEntity: { "@id": BOOK_SCHEMA["@id"] },
        inLanguage: "en",
      },
      {
        ...BOOK_SCHEMA,
        hasPart: BOOK_CHAPTERS.map((chapter, index) => ({
          "@type": "Chapter",
          name: `${chapter.section}: ${chapter.label}`,
          description: chapter.description,
          position: index + 1,
          url: `${SITE_ORIGIN}${BOOK.pdf}#page=${chapter.page}`,
          isPartOf: { "@id": BOOK_SCHEMA["@id"] },
        })),
      },
    ],
  };
  return (
    <main id="main-content" className={styles.page}>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schema).replace(/</g, "\\u003c") }} />
      <header className={styles.intro}>
        <Image src={BOOK.cover} alt="Shadow Futures book cover, with Martin Erlic’s name at the bottom" width={160} height={240} priority className={styles.cover} />
        <div>
          <p className={styles.eyebrow}>The complete book · Free reading edition</p>
          <h1>Shadow Futures</h1>
          <p className={styles.subtitle}>{BOOK.subtitle}</p>
          <p className={styles.author}>By <Link href={AUTHOR.path}>{AUTHOR.name}</Link></p>
          <div className={styles.actions}>
            <a className={styles.download} href={BOOK.pdf} download="Shadow-Futures-Martin-Erlic.pdf">Download the full PDF <span aria-hidden="true">↓</span></a>
            <a className={styles.amazon} href={BOOK.amazon} target="_blank" rel="noopener noreferrer" aria-label="Buy Shadow Futures on Amazon (opens in a new tab)">Buy on Amazon <span aria-hidden="true">↗</span></a>
          </div>
          <p className={styles.meta}>{BOOK.pages} pages · {BOOK_PDF_SIZE} · October 2026</p>
        </div>
      </header>
      <section className={styles.synopsis} aria-labelledby="about-book">
        <p className={styles.eyebrow}>About the book</p>
        <h2 id="about-book">What success leaves unseen.</h2>
        {BOOK_SYNOPSIS.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
        <a href="#read">Start reading <span aria-hidden="true">↓</span></a>
      </section>
      <section className={styles.reader} id="read" aria-label="Read the complete book">
        <div className={styles.toolbar}>
          <span>Read Shadow Futures</span>
          <a href={BOOK.pdf} target="_blank" rel="noreferrer">Open PDF in a new tab ↗</a>
        </div>
        <iframe src={`${BOOK.pdf}#view=FitH`} title="Shadow Futures — complete free reading edition" className={styles.frame} />
        <p className={styles.fallback}>You can also <a href={BOOK.pdf} target="_blank" rel="noreferrer">read the PDF directly</a> or download it for your preferred reader.</p>
      </section>
      <section className={styles.contents} aria-labelledby="contents-title">
        <p className={styles.eyebrow}>Inside the book</p>
        <h2 id="contents-title">From one song to the future of the economy.</h2>
        <div className={styles.chapters}>
          {BOOK_CHAPTERS.map((chapter) => (
            <a key={chapter.section} href={`${BOOK.pdf}#page=${chapter.page}`} target="_blank" rel="noreferrer">
              <small>{chapter.section}</small>
              <h3>{chapter.label}</h3>
              <p>{chapter.description}</p>
              <span className={styles.arrow} aria-hidden="true">↗</span>
            </a>
          ))}
        </div>
      </section>
      <section className={styles.research} aria-labelledby="book-research">
        <p className={styles.eyebrow}>The book and the research</p>
        <h2 id="book-research">Explore the formal argument.</h2>
        <p>
          The book, <cite>{BOOK_FULL_TITLE}</cite>, develops the ideas for a general audience.
          The original paper, <cite>{PAPER.title}</cite>, states the model, assumptions, and
          mathematical results. The book&apos;s AI and policy discussions also draw on further
          evidence and institutional arguments.
        </p>
        <div className={styles.researchLinks}>
          <Link href={PAPER.landingPath}>Read the paper →</Link>
          <Link href="/papers">Browse the research collection →</Link>
          <Link href="/faq">Questions and answers →</Link>
          <a href="/book.md">Book summary in Markdown →</a>
        </div>
      </section>
    </main>
  );
}

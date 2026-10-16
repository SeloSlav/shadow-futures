import type { Metadata } from "next";
import Image from "next/image";

import { BOOK, BOOK_PDF_SIZE } from "@/lib/book";
import { SITE_ORIGIN } from "@/lib/site";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "Read the Book Free",
  description: `${BOOK.title}: ${BOOK.subtitle}. The complete book by Martin Erlic, free to read and download as a formatted PDF.`,
  alternates: { canonical: "/book" },
};

export default function BookPage() {
  const schema = {
    "@context": "https://schema.org", "@type": "Book",
    name: BOOK.title, alternativeHeadline: BOOK.subtitle,
    author: { "@type": "Person", name: BOOK.author, url: `${SITE_ORIGIN}/author/martin-erlic` },
    url: `${SITE_ORIGIN}/book`, image: `${SITE_ORIGIN}${BOOK.cover}`,
    numberOfPages: BOOK.pages, inLanguage: "en", isAccessibleForFree: true,
    bookFormat: "https://schema.org/EBook",
    encoding: { "@type": "MediaObject", contentUrl: `${SITE_ORIGIN}${BOOK.pdf}`, encodingFormat: "application/pdf" },
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
          <p className={styles.author}>Martin Erlic</p>
          <a className={styles.download} href={BOOK.pdf} download="Shadow-Futures-Martin-Erlic.pdf">Download the full PDF <span aria-hidden="true">↓</span></a>
          <p className={styles.meta}>{BOOK.pages} pages · {BOOK_PDF_SIZE} · October 2026</p>
        </div>
      </header>
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
          {BOOK.contents.map((chapter) => (
            <a key={chapter.section} href={`${BOOK.pdf}#page=${chapter.page}`} target="_blank" rel="noreferrer">
              <small>{chapter.section}</small><span>{chapter.label}</span><span className={styles.arrow} aria-hidden="true">↗</span>
            </a>
          ))}
        </div>
      </section>
    </main>
  );
}

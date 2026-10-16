import Image from "next/image";
import Link from "next/link";

import { BOOK, BOOK_PDF_SIZE } from "@/lib/book";
import styles from "@/components/book-hero.module.css";

export function BookHero() {
  return (
    <section className={styles.hero} id="book" aria-labelledby="hero-title">
      <div className={styles.copy}>
        <p className={styles.eyebrow}>The book · Free reading edition</p>
        <h1 id="hero-title">Shadow<br />Futures</h1>
        <p className={styles.subtitle}>{BOOK.subtitle}</p>
        <p className={styles.byline}>By <Link href="/author/martin-erlic">Martin Erlic</Link></p>
        <p className={styles.description}>
          A song finishes first in one world and fortieth in another.
          The recording is the same. What changed?
        </p>
        <p className={styles.description}>
          From parallel music markets to AI, work, and power, a book about
          the winners we see—and the possibilities their success can leave behind.
        </p>
        <div className={styles.actions}>
          <Link className={styles.primary} href="/book">Read the book free <span aria-hidden="true">↗</span></Link>
          <a className={styles.story} href="#novelty">
            Explore the idea <span aria-hidden="true">↓</span>
          </a>
        </div>
        <div className={styles.downloadRow}>
          <a className={styles.download} href={BOOK.pdf} download="Shadow-Futures-Martin-Erlic.pdf">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" /></svg>
            Download PDF
          </a>
          <span className={styles.details}>{BOOK.pages} pages · {BOOK_PDF_SIZE} · No sign-up</span>
        </div>
      </div>

      <Link className={styles.stage} href="/book" aria-label="Open Shadow Futures by Martin Erlic, the complete free book">
        <div className={styles.book} aria-hidden="true">
          <div className={styles.back} />
          <div className={styles.pages} />
          <div className={styles.top} />
          <div className={styles.bottom} />
          <div className={styles.spine}>
            <span>SHADOW FUTURES</span><small>MARTIN ERLIC</small>
          </div>
          <div className={styles.front}>
            <Image src={BOOK.cover} alt="" fill priority sizes="(max-width: 600px) 65vw, 340px" />
          </div>
        </div>
        <span className={styles.open}>Open the book <span aria-hidden="true">↗</span></span>
      </Link>
    </section>
  );
}

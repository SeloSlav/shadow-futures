import { BOOK, BOOK_DESCRIPTION, BOOK_FULL_TITLE } from "@/lib/book";
import { AUTHOR } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

export const BOOK_SCHEMA = {
  "@type": "Book",
  "@id": `${SITE_ORIGIN}/book#book`,
  name: BOOK.title,
  alternativeHeadline: BOOK.subtitle,
  description: BOOK_DESCRIPTION,
  author: {
    "@type": "Person",
    "@id": `${SITE_ORIGIN}${AUTHOR.path}#person`,
    name: AUTHOR.name,
    url: `${SITE_ORIGIN}${AUTHOR.path}`,
  },
  url: `${SITE_ORIGIN}/book`,
  mainEntityOfPage: { "@id": `${SITE_ORIGIN}/book#webpage` },
  image: `${SITE_ORIGIN}${BOOK.cover}`,
  sameAs: BOOK.amazon,
  datePublished: BOOK.date,
  bookEdition: BOOK.edition,
  numberOfPages: BOOK.pages,
  isAccessibleForFree: true,
  inLanguage: "en",
  bookFormat: "https://schema.org/EBook",
  encoding: {
    "@type": "MediaObject",
    contentUrl: `${SITE_ORIGIN}${BOOK.pdf}`,
    encodingFormat: "application/pdf",
    name: `${BOOK_FULL_TITLE} — ${BOOK.edition}`,
  },
};

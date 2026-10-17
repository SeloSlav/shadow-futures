import type { Metadata } from "next";
import { CreatorStoryPage } from "@/components/story/creator-story-page";
import { BOOK, BOOK_DESCRIPTION, BOOK_FULL_TITLE } from "@/lib/book";
import { BOOK_SCHEMA } from "@/lib/book-schema";
import { AUTHOR, PAPER } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

export const metadata: Metadata = {
  title: { absolute: "Shadow Futures by Martin Erlic — Read the Book Free" },
  description: `${BOOK.subtitle}. Read and download Martin Erlic's complete book free, and explore the interactive story behind the comparison budget.`,
  openGraph: {
    title: "Shadow Futures by Martin Erlic",
    description: BOOK.subtitle,
    type: "website",
    url: "/",
    siteName: "Shadow Futures",
  },
  twitter: {
    card: "summary_large_image",
    title: BOOK_FULL_TITLE,
    description: BOOK_DESCRIPTION,
  },
};

export default function HomePage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite",
        "@id": `${SITE_ORIGIN}/#website`,
        url: SITE_ORIGIN,
        name: "Shadow Futures",
        alternateName: ["Shadow Futures", "shadow-futures.vercel.app"],
        description: BOOK_DESCRIPTION,
        inLanguage: "en",
        creator: { "@id": `${SITE_ORIGIN}${AUTHOR.path}#person` },
      },
      {
        "@type": "Person",
        "@id": `${SITE_ORIGIN}${AUTHOR.path}#person`,
        name: AUTHOR.name,
        url: `${SITE_ORIGIN}${AUTHOR.path}`,
        sameAs: [AUTHOR.ssrnUrl, AUTHOR.mediumUrl, AUTHOR.xUrl],
      },
      BOOK_SCHEMA,
      {
        "@type": "ScholarlyArticle",
        "@id": `${SITE_ORIGIN}${PAPER.landingPath}#article`,
        url: `${SITE_ORIGIN}${PAPER.landingPath}`,
        headline: PAPER.title,
        author: { "@id": `${SITE_ORIGIN}${AUTHOR.path}#person` },
        datePublished: PAPER.publishedDate,
        dateModified: PAPER.modifiedDate,
        description: PAPER.abstract,
        keywords: PAPER.keywords,
        identifier: PAPER.doi,
        sameAs: [PAPER.doiUrl, PAPER.ssrnUrl],
        image: `${SITE_ORIGIN}/opengraph-image`,
        isPartOf: { "@id": `${SITE_ORIGIN}/#website` },
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
      <CreatorStoryPage />
    </>
  );
}

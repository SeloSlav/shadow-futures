import type { Metadata } from "next";
import { CreatorStoryPage } from "@/components/story/creator-story-page";
import { BOOK } from "@/lib/book";
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
        description:
          "An interactive explanation of contribution uncertainty in self-reinforcing markets.",
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
      {
        "@type": "Book",
        "@id": `${SITE_ORIGIN}/book#book`,
        name: BOOK.title,
        alternativeHeadline: BOOK.subtitle,
        author: { "@id": `${SITE_ORIGIN}${AUTHOR.path}#person` },
        url: `${SITE_ORIGIN}/book`,
        image: `${SITE_ORIGIN}${BOOK.cover}`,
        numberOfPages: BOOK.pages,
        isAccessibleForFree: true,
        inLanguage: "en",
        encoding: {
          "@type": "MediaObject",
          contentUrl: `${SITE_ORIGIN}${BOOK.pdf}`,
          encodingFormat: "application/pdf",
        },
      },
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

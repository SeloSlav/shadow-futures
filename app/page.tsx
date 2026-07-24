import { CreatorStoryPage } from "@/components/story/creator-story-page";
import { AUTHOR, PAPER } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

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

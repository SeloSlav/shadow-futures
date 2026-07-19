import { CreatorStoryPage } from "@/components/story/creator-story-page";
import { PAPER } from "@/lib/paper/citation";

export default function HomePage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "ScholarlyArticle",
    headline: PAPER.title,
    author: {
      "@type": "Person",
      name: PAPER.author,
    },
    datePublished: "2025-12",
    dateModified: "2026-07",
    description:
      "An interactive creator-platform explanation of how self-reinforcing visibility can reward real work while destroying the comparisons needed to identify its contribution.",
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
      />
      <CreatorStoryPage />
    </>
  );
}

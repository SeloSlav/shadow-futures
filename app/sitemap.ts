import type { MetadataRoute } from "next";

import { BOOK } from "@/lib/book";
import { SITE_ORIGIN } from "@/lib/site";
import { COMPANION_PAPERS, RESEARCH_PUBLISHED_DATE } from "@/lib/papers";
import { PAPER } from "@/lib/paper/citation";

export default function sitemap(): MetadataRoute.Sitemap {
  const discoveryModified = new Date("2026-10-17T00:00:00.000Z");
  const directiveModified = new Date("2026-08-07T00:00:00.000Z");
  const researchModified = new Date(`${RESEARCH_PUBLISHED_DATE}T00:00:00.000Z`);
  const bookPdfModified = new Date(`${BOOK.date}T00:00:00.000Z`);

  return [
    { url: SITE_ORIGIN, lastModified: discoveryModified },
    { url: `${SITE_ORIGIN}/book`, lastModified: discoveryModified },
    { url: `${SITE_ORIGIN}/paper`, lastModified: researchModified },
    { url: `${SITE_ORIGIN}/papers`, lastModified: discoveryModified },
    { url: `${SITE_ORIGIN}/playground`, lastModified: new Date("2026-08-10T00:00:00.000Z") },
    { url: `${SITE_ORIGIN}/for-ai`, lastModified: directiveModified },
    { url: `${SITE_ORIGIN}/faq`, lastModified: new Date("2026-10-08T00:00:00.000Z") },
    { url: `${SITE_ORIGIN}/math`, lastModified: new Date("2026-10-08T00:00:00.000Z") },
    { url: `${SITE_ORIGIN}/methodology`, lastModified: new Date("2026-10-02T00:00:00.000Z") },
    { url: `${SITE_ORIGIN}/author/martin-erlic`, lastModified: discoveryModified },
    { url: `${SITE_ORIGIN}${BOOK.pdf}`, lastModified: bookPdfModified },
    { url: `${SITE_ORIGIN}${PAPER.pdfPath}`, lastModified: researchModified },
    ...COMPANION_PAPERS.map((paper) => ({
      url: `${SITE_ORIGIN}${paper.pdfPath}`,
      lastModified: researchModified,
    })),
  ];
}

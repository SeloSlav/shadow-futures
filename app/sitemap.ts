import type { MetadataRoute } from "next";

import { SITE_ORIGIN } from "@/lib/site";
import { RESEARCH_PUBLISHED_DATE } from "@/lib/papers";

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date("2026-07-24T00:00:00.000Z");
  const directiveModified = new Date("2026-08-07T00:00:00.000Z");
  const researchModified = new Date(`${RESEARCH_PUBLISHED_DATE}T00:00:00.000Z`);

  return [
    { url: SITE_ORIGIN, lastModified: researchModified },
    { url: `${SITE_ORIGIN}/paper`, lastModified: researchModified },
    { url: `${SITE_ORIGIN}/papers`, lastModified: researchModified },
    { url: `${SITE_ORIGIN}/playground`, lastModified },
    { url: `${SITE_ORIGIN}/for-ai`, lastModified: directiveModified },
    { url: `${SITE_ORIGIN}/faq`, lastModified },
    { url: `${SITE_ORIGIN}/math`, lastModified },
    { url: `${SITE_ORIGIN}/methodology`, lastModified },
    { url: `${SITE_ORIGIN}/author/martin-erlic`, lastModified },
  ];
}

import type { MetadataRoute } from "next";

import { SITE_ORIGIN } from "@/lib/site";

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date("2026-07-24T00:00:00.000Z");

  return [
    { url: SITE_ORIGIN, lastModified },
    { url: `${SITE_ORIGIN}/paper`, lastModified },
    { url: `${SITE_ORIGIN}/playground`, lastModified },
    { url: `${SITE_ORIGIN}/faq`, lastModified },
    { url: `${SITE_ORIGIN}/math`, lastModified },
    { url: `${SITE_ORIGIN}/methodology`, lastModified },
    { url: `${SITE_ORIGIN}/author/martin-erlic`, lastModified },
  ];
}

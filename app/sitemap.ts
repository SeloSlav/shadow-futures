import type { MetadataRoute } from "next";

import { SITE_ORIGIN } from "@/lib/site";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    { url: SITE_ORIGIN, changeFrequency: "monthly", priority: 1 },
    { url: `${SITE_ORIGIN}/faq`, changeFrequency: "monthly", priority: 0.9 },
    { url: `${SITE_ORIGIN}/math`, changeFrequency: "monthly", priority: 0.8 },
    { url: `${SITE_ORIGIN}/methodology`, changeFrequency: "monthly", priority: 0.7 },
  ];
}

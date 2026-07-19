import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const origin = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3010";
  return [
    { url: origin, changeFrequency: "monthly", priority: 1 },
    { url: `${origin}/math`, changeFrequency: "monthly", priority: 0.8 },
    { url: `${origin}/methodology`, changeFrequency: "monthly", priority: 0.7 },
  ];
}

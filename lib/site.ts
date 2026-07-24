const vercelOrigin = process.env.VERCEL_PROJECT_PRODUCTION_URL
  ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
  : undefined;

export const SITE_ORIGIN = (
  process.env.NEXT_PUBLIC_SITE_URL ??
  vercelOrigin ??
  "https://shadow-futures.vercel.app"
).replace(/\/+$/, "");

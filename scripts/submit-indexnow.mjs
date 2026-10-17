const key = "8b7f4e8d1c2a49b5a6f03d7e9c4128ab";
const origin = (
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://shadow-futures.vercel.app"
).replace(/\/+$/, "");
const host = new URL(origin).host;
const sitemapUrl = `${origin}/sitemap.xml`;
const sitemapResponse = await fetch(sitemapUrl, {
  signal: AbortSignal.timeout(30_000),
});

if (!sitemapResponse.ok) {
  throw new Error(`Cannot read sitemap: ${sitemapResponse.status} ${sitemapUrl}`);
}

const sitemap = await sitemapResponse.text();
const urlList = [...new Set(
  [...sitemap.matchAll(/<loc>\s*([^<]+?)\s*<\/loc>/g)].map((match) => {
    const url = new URL(match[1].replaceAll("&amp;", "&"));
    if (url.origin !== new URL(origin).origin) {
      throw new Error(`Sitemap contains a URL outside the configured site: ${url}`);
    }
    return url.toString();
  }),
)];

if (urlList.length === 0) {
  throw new Error(`No URLs found in ${sitemapUrl}`);
}

const payload = {
  host,
  key,
  keyLocation: `${origin}/${key}.txt`,
  urlList,
};

// A dry run reads the deployed sitemap and never submits URLs.
if (process.argv.includes("--dry-run")) {
  console.log(JSON.stringify({ sitemap: sitemapUrl, ...payload }, null, 2));
  process.exit(0);
}

const response = await fetch("https://api.indexnow.org/indexnow", {
  method: "POST",
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify(payload),
  signal: AbortSignal.timeout(30_000),
});

if (!response.ok) {
  throw new Error(`IndexNow submission failed: ${response.status} ${await response.text()}`);
}

console.log(`IndexNow accepted ${urlList.length} URLs with status ${response.status}.`);

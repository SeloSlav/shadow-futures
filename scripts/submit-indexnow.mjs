const key = "8b7f4e8d1c2a49b5a6f03d7e9c4128ab";
const origin = (
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://shadow-futures.vercel.app"
).replace(/\/+$/, "");
const host = new URL(origin).host;
const paths = [
  "/",
  "/paper",
  "/playground",
  "/faq",
  "/methodology",
  "/math",
  "/author/martin-erlic",
  "/paper.pdf",
];

const response = await fetch("https://api.indexnow.org/indexnow", {
  method: "POST",
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify({
    host,
    key,
    keyLocation: `${origin}/${key}.txt`,
    urlList: paths.map((path) => new URL(path, `${origin}/`).toString()),
  }),
});

if (!response.ok) {
  throw new Error(`IndexNow submission failed: ${response.status} ${await response.text()}`);
}

console.log(`IndexNow accepted ${paths.length} URLs with status ${response.status}.`);

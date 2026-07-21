import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

const people = [
  ["MrBeast", "mrbeast"],
  ["Taylor Swift", "taylor-swift"],
  ["Rihanna", "rihanna"],
  ["Beyoncé", "beyonce"],
  ["Selena Gomez", "selena-gomez"],
  ["Dwayne Johnson", "dwayne-johnson"],
  ["Cristiano Ronaldo", "cristiano-ronaldo"],
  ["Lionel Messi", "lionel-messi"],
  ["Kylie Jenner", "kylie-jenner"],
  ["Kim Kardashian", "kim-kardashian"],
  ["Oprah Winfrey", "oprah-winfrey"],
  ["Elon Musk", "elon-musk"],
  ["Jeff Bezos", "jeff-bezos"],
  ["Mark Zuckerberg", "mark-zuckerberg"],
  ["Bill Gates", "bill-gates"],
  ["Warren Buffett", "warren-buffett"],
  ["Richard Branson", "richard-branson"],
  ["Jensen Huang", "jensen-huang"],
  ["Satya Nadella", "satya-nadella"],
  ["Sundar Pichai", "sundar-pichai"],
  ["Sara Blakely", "sara-blakely"],
  ["Charli D'Amelio", "charli-damelio"],
  ["Khaby Lame", "khaby-lame"],
  ["Emma Chamberlain", "emma-chamberlain"],
];

const outputDirectory = join(process.cwd(), "public", "creator-portraits");
const temporaryDirectory = join(process.cwd(), ".tmp", "creator-portraits");
const headers = {
  "User-Agent": "ShadowFuturesSite/1.0 (portrait asset attribution build)",
};

mkdirSync(outputDirectory, { recursive: true });
mkdirSync(temporaryDirectory, { recursive: true });

function stripHtml(value = "") {
  return value
    .replace(/<[^>]*>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

async function wikipediaQuery(parameters, attempt = 1) {
  const url = new URL("https://en.wikipedia.org/w/api.php");
  url.search = new URLSearchParams({
    action: "query",
    format: "json",
    formatversion: "2",
    redirects: "1",
    origin: "*",
    ...parameters,
  }).toString();
  const response = await fetch(url, { headers });
  if (response.status === 429 && attempt < 4) {
    const retryAfter = Number(response.headers.get("retry-after")) || attempt * 4;
    await new Promise((resolve) => setTimeout(resolve, retryAfter * 1_000));
    return wikipediaQuery(parameters, attempt + 1);
  }
  if (!response.ok) {
    throw new Error(`Wikipedia request failed (${response.status}): ${url}`);
  }
  return response.json();
}

async function fetchPortrait(url, name, attempt = 1) {
  const response = await fetch(url, { headers });
  if (response.status === 429 && attempt < 7) {
    await response.body?.cancel();
    const retryAfter = Number(response.headers.get("retry-after")) || attempt * 5;
    await new Promise((resolve) => setTimeout(resolve, retryAfter * 1_000));
    return fetchPortrait(url, name, attempt + 1);
  }
  if (!response.ok) {
    throw new Error(`Portrait download failed for ${name} (${response.status})`);
  }
  return response;
}

const credits = [];

const pageData = await wikipediaQuery({
  titles: people.map(([name]) => name).join("|"),
  prop: "pageimages",
  piprop: "thumbnail|name|original",
  pithumbsize: "900",
});
const articlePages = pageData.query?.pages ?? [];
const pagesByTitle = new Map(articlePages.map((page) => [page.title, page]));
const imageData = await wikipediaQuery({
  titles: articlePages.map((page) => `File:${page.pageimage}`).join("|"),
  prop: "imageinfo",
  iiprop: "url|extmetadata",
});
const imageInfoByName = new Map(
  (imageData.query?.pages ?? []).map((page) => [
    page.title.replace(/^File:/, "").replaceAll(" ", "_"),
    page.imageinfo?.[0],
  ]),
);

for (const [name, slug] of people) {
  const page = pagesByTitle.get(name);
  if (!page?.thumbnail?.source || !page?.pageimage) {
    throw new Error(`No portrait found for ${name}`);
  }
  const imageInfo = imageInfoByName.get(page.pageimage);
  const sourcePath = join(temporaryDirectory, `${slug}-${basename(new URL(page.thumbnail.source).pathname)}`);
  const outputPath = join(outputDirectory, `${slug}.webp`);
  if (!existsSync(outputPath)) {
    const response = await fetchPortrait(page.thumbnail.source, name);
    writeFileSync(sourcePath, Buffer.from(await response.arrayBuffer()));
    execFileSync(
      "magick",
      [
        sourcePath,
        "-auto-orient",
        "-thumbnail",
        "480x480^",
        "-gravity",
        "center",
        "-extent",
        "480x480",
        "-strip",
        "-quality",
        "84",
        outputPath,
      ],
      { stdio: "inherit" },
    );
    rmSync(sourcePath, { force: true });
    await new Promise((resolve) => setTimeout(resolve, 1_000));
  }

  const metadata = imageInfo?.extmetadata ?? {};
  credits.push({
    name,
    articleUrl: `https://en.wikipedia.org/wiki/${encodeURIComponent(page.title.replaceAll(" ", "_"))}`,
    filePageUrl: imageInfo?.descriptionurl ?? imageInfo?.url,
    artist: stripHtml(metadata.Artist?.value || metadata.Credit?.value || "See source page"),
    license: stripHtml(metadata.LicenseShortName?.value || metadata.UsageTerms?.value || "See source page"),
    licenseUrl: metadata.LicenseUrl?.value || imageInfo?.descriptionurl,
    sourceImage: page.thumbnail.source,
  });
  process.stdout.write(`Downloaded ${name}\n`);
}

writeFileSync(
  join(outputDirectory, "credits.json"),
  `${JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      note: "Portraits were center-cropped, resized, and converted to WebP for this illustrative cast. No endorsement is implied.",
      credits,
    },
    null,
    2,
  )}\n`,
);

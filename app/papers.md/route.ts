import { COMPANION_PAPERS, MAIN_RESEARCH_PAPER, RESEARCH_AUTHOR, RESEARCH_REVISION } from "@/lib/papers";
import { PAPER } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

export const dynamic = "force-static";

export function GET() {
  const markdown = [
    "# Shadow Futures research collection",
    "",
    `Author: ${RESEARCH_AUTHOR}`,
    `Revision: ${RESEARCH_REVISION}`,
    `Canonical collection: ${SITE_ORIGIN}/papers`,
    "",
    "The collection contains the original Shadow Futures paper and six companion research drafts. The companions develop additional models and assumptions; their results are distinct from the original paper’s theorem.",
    "",
    ...[MAIN_RESEARCH_PAPER, ...COMPANION_PAPERS].flatMap((paper) => [
      `## ${paper.title}`,
      "",
      `Status: ${paper.status}`,
      `Focus: ${paper.focus}`,
      `Pages: ${paper.pages}`,
      "",
      paper.description,
      "",
      `[Read the PDF](${SITE_ORIGIN}${paper.pdfPath})`,
      "",
    ]),
    "## Original paper citation",
    "",
    PAPER.citation,
    "",
    `The DOI ${PAPER.doi} identifies the original paper. It does not identify the companion drafts or the separate book.`,
    "",
    `Book and chapter guide: ${SITE_ORIGIN}/book`,
    "",
  ].join("\n");

  return new Response(markdown, {
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
      Link: `<${SITE_ORIGIN}/papers>; rel="canonical"`,
    },
  });
}

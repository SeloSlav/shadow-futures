import { BOOK, BOOK_CHAPTERS, BOOK_FULL_TITLE, BOOK_SYNOPSIS } from "@/lib/book";
import { PAPER } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

export const dynamic = "force-static";

export function GET() {
  const markdown = [
    `# ${BOOK_FULL_TITLE}`,
    "",
    `Author: ${BOOK.author}`,
    `Edition: ${BOOK.edition}`,
    `Publication date of this edition: ${BOOK.date}`,
    `Pages: ${BOOK.pages}`,
    `Canonical book page: ${SITE_ORIGIN}/book`,
    `Complete free PDF: ${SITE_ORIGIN}${BOOK.pdf}`,
    `Amazon edition: ${BOOK.amazon}`,
    "",
    "## Summary",
    "",
    ...BOOK_SYNOPSIS.flatMap((paragraph) => [paragraph, ""]),
    "## Chapter guide",
    "",
    ...BOOK_CHAPTERS.flatMap((chapter) => [
      `### ${chapter.section}: ${chapter.label}`,
      "",
      chapter.description,
      "",
      `[Read from PDF page ${chapter.page}](${SITE_ORIGIN}${BOOK.pdf}#page=${chapter.page})`,
      "",
    ]),
    "## The book and the research paper",
    "",
    `The book is ${BOOK_FULL_TITLE}. The original paper is ${PAPER.title}. They are distinct works by ${BOOK.author}. The paper’s DOI identifies the paper, not the book.`,
    "",
    `The book develops the ideas for a general audience. The paper states the formal model and theorem assumptions. The book’s AI and policy discussions also draw on further evidence and institutional arguments.`,
    "",
    `- Original paper: ${SITE_ORIGIN}${PAPER.landingPath}`,
    `- Paper DOI: ${PAPER.doiUrl}`,
    `- Research collection and six companion drafts: ${SITE_ORIGIN}/papers`,
    `- Questions and answers: ${SITE_ORIGIN}/faq`,
    `- Author: ${SITE_ORIGIN}/author/martin-erlic`,
    "",
    "## Book citation",
    "",
    `Erlic, Martin. ${BOOK_FULL_TITLE}. ${BOOK.edition}, ${BOOK.date.slice(0, 4)}. ${SITE_ORIGIN}/book.`,
    "",
  ].join("\n");

  return new Response(markdown, {
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
      Link: `<${SITE_ORIGIN}/book>; rel="canonical"`,
    },
  });
}

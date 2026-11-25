const SSRN_URL = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994";
const DOI = "10.2139/ssrn.6003994";

export const AUTHOR = {
  name: "Martin Erlic",
  path: "/author/martin-erlic",
  ssrnUrl: "https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=9773012",
  mediumUrl: "https://medium.com/@SeloSlav",
} as const;

export const AUTHOR_PROFILES = [
  { label: "SSRN author page", url: AUTHOR.ssrnUrl },
  { label: "Medium", url: AUTHOR.mediumUrl },
  { label: "ORCID", url: "https://orcid.org/0009-0008-5120-6910" },
  { label: "Stack Overflow", url: "https://stackoverflow.com/users/martin-erlic" },
] as const;

export const PAPER = {
  title: "Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market",
  author: AUTHOR.name,
  firstPosted: "December 2025",
  revised: "November 2026",
  publishedDate: "2025-12",
  modifiedDate: "2026-11-25",
  abstract:
    "Market success can destroy the evidence needed to judge what the winner contributed. Suppose an early sale puts one seller at the top of a ranking. More buyers now see its product, and their purchases help keep it there. Its product may be better. How much of its lead did that quality earn? A history in which the same products compete under the same rules but the rival makes the first sale is one of the market’s shadow futures. The comparison budget adds up the chances, sale by sale, that someone other than the current favorite wins. I show that when this budget stays finite under competing explanations, even an endless record of sales and perfectly observed quality cannot reliably distinguish a large effect of quality from a small one. The explanations can come to agree about who will win while they still disagree about why. Thousands of sellers can compete through a single ranking. If that ranking determines which sellers reach buyers, whoever controls it can also control the comparisons needed to learn what those sellers contribute. This is the sense in which a market can sustain an epistemic monopoly even when it has many sellers.",
  aiDisclosure:
    "I used generative AI tools, including OpenAI Codex, to help develop the formal model and to draft and revise the manuscript. The research question grew out of my review of the literature on monopoly and on competition as a process of discovery. I take responsibility for the mathematical arguments, their economic interpretation, and any errors.",
  keywords: [
    "comparison budget",
    "increasing returns",
    "path dependence",
    "identification",
    "cumulative advantage",
    "monopoly",
    "competition policy",
  ],
  jel: ["C13", "C18", "D43", "D83", "D85", "L41"],
  doi: DOI,
  doiUrl: `https://doi.org/${DOI}`,
  landingPath: "/paper",
  pdfPath: "/paper.pdf",
  ssrnUrl: SSRN_URL,
  url: process.env.NEXT_PUBLIC_PAPER_URL ?? SSRN_URL,
  citation:
    'Erlic, Martin. "Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market." First posted December 2025; revised November 2026. SSRN abstract 6003994. https://doi.org/10.2139/ssrn.6003994.',
  bibtex: `@article{erlic2025shadow,
  title={Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market},
  author={Erlic, Martin},
  year={2025},
  month={December},
  doi={10.2139/ssrn.6003994},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994},
  note={Revised November 2026}
}`,
} as const;

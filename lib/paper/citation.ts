const SSRN_URL = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994";
const DOI = "10.2139/ssrn.6003994";

export const AUTHOR = {
  name: "Martin Erlic",
  path: "/author/martin-erlic",
  ssrnUrl: "https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=9773012",
  mediumUrl: "https://medium.com/@SeloSlav",
} as const;

export const PAPER = {
  title: "Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market",
  author: AUTHOR.name,
  firstPosted: "December 2025",
  revised: "October 2026",
  publishedDate: "2025-12",
  modifiedDate: "2026-10",
  abstract:
    "Can a market observe productive inputs perfectly yet fail to learn their effects on reward? I study markets where inputs affect reward odds and past rewards shape future exposure. The comparison budget sums, across allocations, the probabilities that someone other than the current favorite wins. In a conditional logit model with bounded input differences, this budget bounds Fisher information about input effects. If the lifetime budget is almost surely finite under every parameter, a single market history cannot distinguish competing effects with vanishing error. Yet models that disagree about those effects converge in their forecasts of future allocations.",
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
    'Erlic, Martin. "Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market." First posted December 2025; revised October 2026. SSRN abstract 6003994. https://doi.org/10.2139/ssrn.6003994.',
  bibtex: `@article{erlic2025shadow,
  title={Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market},
  author={Erlic, Martin},
  year={2025},
  month={December},
  doi={10.2139/ssrn.6003994},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994},
  note={Revised October 2026}
}`,
} as const;

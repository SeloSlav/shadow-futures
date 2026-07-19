const SSRN_URL = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994";

export const PAPER = {
  title: "Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market",
  author: "Martin Erlic",
  firstPosted: "December 2025",
  revised: "July 2026",
  ssrnUrl: SSRN_URL,
  url: process.env.NEXT_PUBLIC_PAPER_URL ?? SSRN_URL,
  bibtex: `@article{erlic2026shadow,
  title={Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market},
  author={Erlic, Martin},
  year={2026},
  month={July},
  note={First posted December 2025; revised July 2026}
}`,
} as const;

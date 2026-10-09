const SSRN_URL = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994";
const DOI = "10.2139/ssrn.6003994";

export const AUTHOR = {
  name: "Martin Erlic",
  path: "/author/martin-erlic",
  ssrnUrl: "https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=9773012",
  mediumUrl: "https://medium.com/@SeloSlav",
  xUrl: "https://x.com/seloslav",
} as const;

export const PAPER = {
  title: "Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market",
  author: AUTHOR.name,
  firstPosted: "December 2025",
  revised: "October 2026",
  publishedDate: "2025-12",
  modifiedDate: "2026-10",
  abstract:
    "Can a market observe productive inputs perfectly yet fail to learn their effects on reward? This paper studies adaptive allocation in which verified inputs enter reward odds while past rewards change future exposure. The comparison budget is cumulative allocation probability outside the current leader. In conditional logit, it bounds Fisher information and finite-horizon statistical separation. Under a common predictable design, local equivalence, and comparison-dominated separation, an almost-surely finite budget yields equivalent complete-history laws. No estimator of a nonconstant contribution functional can then be consistent at every parameter; no test can have both errors vanish. By Blackwell–Dubins, conditional forecasts nevertheless merge: predictive agreement can coexist with permanent uncertainty about input effects. A local two-point lower bound relates attribution precision to the expected comparison budget. Strong reinforcement supplies one sufficient case; fixed inputs with latent position yield a separate failure of point identification. Unrealized comparison paths are shadow futures. The contribution is an economic information bound and its attribution interpretation, using classical probability machinery. Productive inputs can remain causally effective even when their effects cannot be consistently recovered from one market history.",
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

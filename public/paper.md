# Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market

Author: [Martin Erlic](https://shadow-futures.vercel.app/author/martin-erlic)

- First posted: December 2025
- Revised: October 2026
- DOI: [10.2139/ssrn.6003994](https://doi.org/10.2139/ssrn.6003994)
- SSRN: [Abstract 6003994](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994)
- PDF: [Download the full paper](https://shadow-futures.vercel.app/paper.pdf)

## Abstract

Can a market observe productive inputs perfectly yet fail to learn their effects on reward? This paper studies adaptive allocation in which verified inputs enter reward odds while past rewards change future exposure. The comparison budget is cumulative allocation probability outside the current leader. In conditional logit, it bounds Fisher information and finite-horizon statistical separation. Under a common predictable design, local equivalence, and comparison-dominated separation, an almost-surely finite budget yields equivalent complete-history laws. No estimator of a nonconstant contribution functional can then be consistent at every parameter; no test can have both errors vanish. By Blackwell–Dubins, conditional forecasts nevertheless merge: predictive agreement can coexist with permanent uncertainty about input effects. A local two-point lower bound relates attribution precision to the expected comparison budget. Strong reinforcement supplies one sufficient case; fixed inputs with latent position yield a separate failure of point identification. Unrealized comparison paths are shadow futures. The contribution is an economic information bound and its attribution interpretation, using classical probability machinery. Productive inputs can remain causally effective even when their effects cannot be consistently recovered from one market history.

## Keywords

comparison budget; increasing returns; path dependence; identification; cumulative advantage; monopoly; competition policy

## JEL codes

C13; C18; D43; D83; D85; L41

## Central result

The comparison budget is cumulative probability mass remaining outside the currently dominant alternative. Under the paper's common-design, local-equivalence, Hellinger-control, and finite-comparison conditions, distinct contribution parameters generate mutually absolutely continuous complete-history laws.

The implication is an identification limit. No estimator based on one realized market can consistently recover every nonconstant contribution functional, and no test can separate two contribution parameters with vanishing total error.

Strong reinforcement is one sharp corollary because it can exhaust the comparison budget and produce eventual allocation monopoly. With latent position, contribution and position can be exactly observationally equivalent.

## Contribute on Hunchroom

[Join the formal proof request on Hunchroom](https://hunchroom.com/p/44) to contribute a Lean proof, an independent meaning review, or a useful partial attempt. Request #44 formalizes a finite-recipient specialization of Theorem 1 and Appendix B, with scalar contribution functionals.

The posted Lean statement has been typechecked. This establishes that the proposition is well formed; proof verification and review of its correspondence to the paper are separate checks.

## Citation

Erlic, Martin. "Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market." First posted December 2025; revised October 2026. SSRN abstract 6003994. https://doi.org/10.2139/ssrn.6003994.

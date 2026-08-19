# Shadow Futures

**Contribution Uncertainty and the Self-Reinforcing Market**

An interactive economics story based on Martin Erlic's paper about a subtle
failure of self-reinforcing markets: they can keep recording activity while
losing the comparisons needed to explain what caused success.

[Explore the interactive essay](https://shadow-futures.vercel.app/) ·
[Read the paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994) ·
[Download the PDF](https://shadow-futures.vercel.app/paper.pdf) ·
[Open the FAQ](https://shadow-futures.vercel.app/faq)

## A market can observe everything and still fail to learn

Imagine a market that perfectly observes work, quality, effort, judgment,
capital, or risk. Those inputs genuinely affect who gets rewarded. But every
reward also changes who is most likely to be seen and rewarded next.

Early success becomes position. Position becomes exposure. Exposure becomes
more success.

As that loop strengthens, alternatives receive fewer meaningful chances to
compete. The market may process thousands of new transactions without creating
thousands of useful comparisons. Eventually, one realized history is rich in
events but poor in evidence about contribution.

The missing comparison paths are **shadow futures**: unrealized repetitions in
which the same productive inputs meet different accumulated positions. They are
the paths we would need to separate what an input contributed from what the
market had already reinforced.

The paper formalizes this as a one-history learning limit. Under its stated
conditions, if the market has only a finite total amount of remaining
comparison, no estimator can consistently recover every nonconstant
contribution functional from the realized path, and no test can separate two
contribution parameters with vanishing total error.

That is a claim about knowledge, not a claim that work does not matter. A real
causal effect can exist and still be unrecoverable from the history that
rewarded it.

## What the experience shows

The site turns the argument into a visual story about creators, platforms, and
firms. Its simulations let you:

- watch nearly identical creator paths diverge after small early differences;
- compare uninterrupted rankings with periodic resets that restore competition;
- run parallel worlds with the same structure and different seeded histories;
- see why repeatedly rewarding a leader adds activity without necessarily
  adding much identifying information;
- contrast causal questions with Lorenz curves, which describe the final
  distribution but not what produced it; and
- carry the same logic into market concentration, interoperability, antitrust,
  taxation, UBI, social dividends, and portability.

For the formal route, see the
[methodology](https://shadow-futures.vercel.app/methodology),
[mathematics](https://shadow-futures.vercel.app/math), and
[comparison playground](https://shadow-futures.vercel.app/playground).

## The comparison budget

The default simulation gives agent `i` a score based on a verified input and
its accumulated position:

```text
s_it(β) = exp(β x_i) (a + N_i(t))^ρ
p_it(β) = s_it(β) / Σ_j s_jt(β)
```

Here, `β` is the contribution coefficient, `x_i` is the verified input, `a` is
baseline attachment, `N_i(t)` is accumulated reward, and `ρ` controls
reinforcement.

At each date, the probability mass outside the current leader is:

```text
ε_t = 1 - max_i p_it
```

The cumulative sum of `ε_t` is the market's **comparison budget**. When that
budget remains finite, time can continue while effective comparison runs out.
The relevant resource for learning is therefore not transaction count alone,
but how much contestability those transactions preserve.

## Why it matters

The idea applies wherever rewards reshape future opportunity: recommendation
feeds, creator markets, hiring, procurement, finance, technical standards,
cloud platforms, AI, logistics, and other increasing-returns industries.

It changes the question we should ask of concentrated outcomes. Instead of only
asking whether success involved skill, effort, or investment, we must also ask
whether the market preserved enough independent comparison to measure their
contribution.

That distinction matters for competition and distribution policy. It does not
automatically select a merger rule, tax rate, or welfare program; it changes
what the observed market history can legitimately be used to infer.

## What the simulations establish—and what they do not

- The simulations illustrate reinforced allocation and disappearing
  comparison; they do not prove the theorem or estimate any real platform.
- Work, quality, effort, judgment, capital, and risk can genuinely affect
  reward even when their contribution cannot be recovered from one history.
- Path dependence alone is not the impossibility result. The theorem depends on
  common-design, local-equivalence, Hellinger-control, and finite-comparison
  conditions stated in the paper.
- Complete-history laws are mutually absolutely continuous under those
  conditions; they are not claimed to be identical.
- Finite comparison is sufficient for the theorem, not necessary for every
  identification failure.
- Policy implications remain conditional on their economic and normative
  assumptions.

## Reproducible by construction

The simulator supports 2–10 agents, 10–10,000 periods, one- or two-dimensional
inputs, initial-position differences, ranking resets, exploration, independent
channels, and up to 1,000 parallel worlds. Scenario state can be shared by URL
or exported as JSON; histories can be exported as CSV.

Every run uses a seeded Mulberry32 generator. Simulation results never rely on
unseeded `Math.random()`, and large parallel-world batches run in a Web Worker.
The equation registry maps the article and appendix equations to their source
locations, assumptions, variables, explanations, and derivations.

## Run locally

You need Node.js 20.9+ and npm 10+.

```bash
npm install
npm run dev
```

Open [http://localhost:3010](http://localhost:3010).

Useful checks:

```bash
npm run lint
npm test
npm run build
npm run test:e2e
```

The end-to-end suite uses Google Chrome. The app has no backend or database and
can be deployed as a standard Next.js project. On Vercel, import the repository
and keep the default framework settings.

These environment variables are optional:

```env
NEXT_PUBLIC_SITE_URL=https://your-domain.example
NEXT_PUBLIC_PAPER_URL=https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994
```

`NEXT_PUBLIC_PAPER_URL` may instead point to the included `/paper.pdf` or
`/paper.docx`.

## Project map

```text
app/                 Routes, metadata, and page shells
components/story/    Interactive narrative and visualizations
components/equations Equation explanations and controls
lib/model/           Seeded allocation and information simulations
lib/equations/       Paper-to-interface equation registry
lib/scenarios/       Shareable scenario presets
tests/unit/          Deterministic model and state tests
tests/e2e/           Desktop and mobile journeys
public/              Paper, citation, and machine-readable sources
```

The interface is built with Next.js, React, TypeScript, D3, Framer Motion,
KaTeX, Zustand, and Zod.

## Citation

Erlic, Martin. “Shadow Futures: Contribution Uncertainty and the
Self-Reinforcing Market.” First posted December 2025; revised July 2026.
[SSRN abstract 6003994](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994).
[https://doi.org/10.2139/ssrn.6003994](https://doi.org/10.2139/ssrn.6003994).

```bibtex
@article{erlic2025shadow,
  title={Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market},
  author={Erlic, Martin},
  year={2025},
  month={December},
  doi={10.2139/ssrn.6003994},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994},
  note={Revised July 2026}
}
```

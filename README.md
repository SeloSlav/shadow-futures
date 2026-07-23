# Shadow Futures

A production-ready interactive economics story based on Martin Erlic's paper,
*Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market*.

The familiar mechanism is self-reinforcement: early success can make later
success more likely. The paper's distinct contribution is an information
problem. A market can reward real productive inputs while consuming the
independent comparison paths needed to measure what those inputs contributed.
The unrealized repetitions that could have separated contribution from
accumulated position are the shadow futures.

The main experience begins with creators whose work has different modeled
audience appeal competing inside a self-reinforcing recommendation feed, then
extends the same information problem to firms. Three animated graphs show:

1. How ten creator paths diverge after a small early burst of attention, and
   how the second- and third-place paths change under periodic ranking resets.
2. Why repeatedly boosting an early leader teaches us less than regularly
   giving everyone another equal start.
3. Why a Lorenz curve is a final scoreboard rather than evidence of what caused
   the result.

The story connects this mechanism to creator careers across OnlyFans, Fanvue,
YouTube, TikTok, Twitch, Instagram, Patreon, Substack, and online marketplaces,
then extends it to competition among firms in AI, cloud computing,
manufacturing, logistics, software, technical standards, finance, and public
procurement. It carries the argument into merger policy, open standards,
progressive taxation, UBI, social dividends, antitrust, and portability. The
mathematics route presents one combined equation that connects the allocation
rule, the closing contest, and the one-history learning limit.

## Quick start

Requirements:

- Node.js 20.9 or newer
- npm 10 or newer
- Google Chrome for the configured Playwright projects

```bash
npm install
npm run dev
```

Open [http://localhost:3010](http://localhost:3010).

## Commands

```bash
npm run dev
npm run lint
npm run test
npm run test:e2e
npm run build
npm run start
```

`npm run test` runs deterministic model, equation-registry, and URL-state tests with Vitest. `npm run test:e2e` starts the app and runs the desktop and mobile journeys in installed Google Chrome.

## Production build and deployment

```bash
npm install
npm run lint
npm run test
npm run build
npm run start
```

The application has no backend, database, or server-side state. It deploys directly to Vercel:

1. Import this repository in Vercel.
2. Keep the default Next.js framework preset.
3. Set `NEXT_PUBLIC_SITE_URL` to the production origin.
4. Optionally set `NEXT_PUBLIC_PAPER_URL` to an SSRN page, hosted PDF, or other canonical paper URL.
5. Deploy.

All simulation work is client-side. Large parallel-world batches are moved to a Web Worker.

## Paper URL

The repository includes the supplied source document at `public/paper.docx`. No lossy PDF conversion is performed.

```env
NEXT_PUBLIC_PAPER_URL=/paper.docx
NEXT_PUBLIC_SITE_URL=https://your-domain.example
```

If a final PDF becomes available, place it in `public/paper.pdf` and use:

```env
NEXT_PUBLIC_PAPER_URL=/paper.pdf
```

An SSRN or publisher URL can be used instead.

## Simulation model

The default model has:

- 5 agents
- scalar verified inputs `x_i`
- contribution coefficient `β = 1`
- baseline attachment `a = 1`
- reinforcement exponent `ρ = 1.35`
- 500 transactions
- seed 42

At each date:

1. Compute

   ```text
   p_it(β) =
     exp(β x_i) (a + N_i(t))^ρ
     / Σ_j exp(β x_j) (a + N_j(t))^ρ
   ```

2. Compute residual contestability `ε_t = 1 - max_i p_it`.
3. Add `ε_t` to the comparison budget.
4. Compute the conditional Fisher-information trace and its `D_X² ε_t` upper bound.
5. Draw one recipient from Mulberry32.
6. Increment the recipient count.
7. Store the probability vector, counts, recipient, comparison, and information metrics.

The model supports:

- 2–10 agents
- 10–10,000 periods
- scalar or two-dimensional verified inputs
- initial position by agent
- contribution, baseline, reinforcement, exploration, and reset controls
- 2–1,000 structurally identical worlds with distinct derived seeds
- 1–100 independent channels
- reproducible URL state
- CSV history and JSON scenario exports

Simulation code lives in `lib/model`. No simulation result uses unseeded `Math.random()`.

## Equation mapping

`lib/equations/registry.ts` retains all 49 displayed equations from the main
article and technical appendix, in document order, plus three key inline
structural equations. The registry supports source verification and tests; it
is not rendered as a public catalog.

- Main article equations (1)–(11)
- Appendix A information and KL identities, including (A1)
- Appendix B likelihood-ratio and Hellinger steps, including (B1)–(B5)
- Appendix C finite-horizon Le Cam construction
- Appendix D reinforced allocation, clock construction, and common-support proof, including (D1)–(D3)

Every registry entry records:

- a stable ID and title
- paper section and equation number, where numbered
- LaTeX
- plain-language explanation
- assumptions
- variable definitions
- equation role (definition, identity, bound, condition, theorem, or policy implication)
- derivation steps

The `/math` route instead renders one central, paper-faithful chain with a
plain-language breakdown. The `/methodology` route presents the essential
supporting equations and theorem conditions.

## Theorem versus simulation

The simulations illustrate the allocation and information mechanisms. They do not prove the impossibility theorem.

The application preserves these boundaries from the paper:

- Work, quality, effort, judgment, capital, and risk can genuinely affect reward.
- Different realized outcomes under identical inputs do not imply zero causal effect.
- Path dependence alone does not imply the exact impossibility.
- Finite comparison is sufficient under the theorem’s assumptions, not necessary for every identification failure.
- Strong reinforcement is a sharp primitive corollary, not the definition of the phenomenon.
- Complete-history laws are mutually absolutely continuous; they are not claimed to be identical.
- One-history learning impossibility is distinct from exact point non-identification with latent position.
- Any additional parameter-dependent observation must be included in the experiment.
- Independent competition can create identifying variation; nominal firm count need not.
- Merger and tax implications require their stated assumptions and do not automatically determine welfare or tax rates.
- The contribution estimand concerns direct reward contribution inside the allocation mechanism, not total social value or moral desert.

## Accessibility

- Semantic landmark and heading structure
- Skip link
- Keyboard-operable buttons, range controls, chart paths, and details
- Visible focus treatment
- Text alternatives for SVG charts
- Current values exposed by labeled native range inputs
- `prefers-reduced-motion` support that follows the operating-system setting
- Layouts tested from 375px upward without intentional horizontal overflow

## Add a scenario

1. Open `lib/scenarios/presets.ts`.
2. Add a scenario through the `withDefaults` helper.
3. Provide a unique `name`.
4. Ensure `n`, `inputs`, and `initialPositions` agree.
5. Run:

   ```bash
   npm run test
   npm run build
   ```

Scenario state is normalized before simulation and validated with Zod before URL serialization.

## Add an equation card

1. Open `lib/equations/registry.ts`.
2. Add an `eq({ ... })` entry with a stable `id`.
3. Supply its paper location, LaTeX, equation role, explanation, variables, and assumptions.
4. Add meaningful controls only when the statement has an adjustable numerical interpretation.
5. For measure-theoretic results, prefer a finite example or candidate-parameter toggle.
6. Update the expected registry count in `tests/unit/equations.test.ts`.
7. Run the tests and inspect `/math`.

## Architecture

```text
app/
  page.tsx
  math/page.tsx
  methodology/page.tsx
  opengraph-image.tsx
components/
  equations/
  story/
  ui/
lib/
  equations/
  model/
  paper/
  scenarios/
  store/
tests/
  unit/
  e2e/
```

React Server Components provide route shells and metadata. Client components are limited to simulations, controls, motion, chart state, and KaTeX rendering. Custom SVG charts use D3 scales/shapes. Zustand holds the shared market state. Zod validates shareable scenarios.

## Citation

```bibtex
@article{erlic2026shadow,
  title={Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market},
  author={Erlic, Martin},
  year={2026},
  month={July},
  note={First posted December 2025; revised July 2026}
}
```

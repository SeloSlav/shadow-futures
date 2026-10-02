import type { Metadata } from "next";
import Link from "next/link";

import { Math as EquationMath } from "@/components/ui/math";

export const metadata: Metadata = {
  title: "Methodology",
  description:
    "How the comparison budget limits learning from one self-reinforcing market: the model, an everyday example, theorem assumptions, and policy boundaries.",
  alternates: { canonical: "/methodology" },
  openGraph: {
    type: "article",
    title: "Shadow Futures methodology",
    description:
      "The assumptions, evidence, theorem, and policy boundaries behind Shadow Futures.",
    url: "/methodology",
  },
};

export default function MethodologyPage() {
  return (
    <main className="method-page" id="main-content">
      <header className="method-page__header">
        <p className="eyebrow">Methodology and scope</p>
        <h1>How the argument works</h1>
        <p>
          Work, quality and risk can improve someone’s chances. An early win can also improve
          their chances next time. The paper asks whether one market history can tell us how
          much the productive inputs contributed when both forces operate together.
          The animation illustrates the mechanism; the theorem establishes a learning limit
          under stated conditions.
        </p>
        <div className="button-row method-page__actions">
          <Link className="button button--primary" href="/paper">
            Read the source paper
          </Link>
          <Link className="button" href="/math">
            See the central equation
          </Link>
        </div>
      </header>

      <ol className="method-map" aria-label="Methodology overview">
        <li>
          <span>01</span>
          <strong>The market chooses</strong>
          <p>Verified inputs and accumulated position shape who wins next.</p>
        </li>
        <li>
          <span>02</span>
          <strong>The contest can close</strong>
          <p>An early win can make later opportunities increasingly likely.</p>
        </li>
        <li>
          <span>03</span>
          <strong>Evidence can run out</strong>
          <p>A long record can add little evidence about the effect of productive inputs.</p>
        </li>
      </ol>

      <section className="method-section">
        <div className="method-section__label">
          <span>01 / Simulation</span>
          <h2>What the app shows</h2>
        </div>
        <div className="method-section__content">
          <p className="method-section__lead">
            The paper’s general model combines verified productive inputs with accumulated
            position to determine who receives the next reward. The app illustrates one
            special case: previous recommendations increase the chance of another recommendation.
          </p>
          <div className="method-equation">
            <span className="panel__meta">Who receives the next opportunity</span>
            <EquationMath
              latex="p_{it}(\beta)=\frac{\exp(\beta x_i)(a+N_i(t))^\rho}{\sum_j\exp(\beta x_j)(a+N_j(t))^\rho}"
              label="Reinforced allocation probability"
            />
            <p>
              The input multiplier is <EquationMath latex="\exp(\beta x_i)" block={false} />.
              The feedback multiplier grows with the number of earlier rewards,{" "}
              <EquationMath latex="N_i(t)" block={false} />. A positive starting value{" "}
              <EquationMath latex="a" block={false} /> lets everyone enter;{" "}
              <EquationMath latex="\rho" block={false} /> controls feedback strength.
            </p>
          </div>
          <div className="method-facts">
            <div>
              <strong>The story example</strong>
              <p>
                Twenty-four creators with modeled audience-response multipliers from 0.84 to
                1.18 compete for 1,600 recommendations. The feedback strength is fixed at 1.55,
                so both creator differences and accumulated exposure affect the next ranking.
              </p>
            </div>
            <div>
              <strong>From creators to firms</strong>
              <p>
                For a firm, the accumulated advantage might be customers, contracts, an
                installed base or past sales. The exact measure must match the real market.
              </p>
            </div>
          </div>
          <p className="method-note">
            The simulation sets the input multipliers by construction. Fixed random seeds
            make replays reproducible. These are illustrations, not estimates for a real
            platform or a proof of what happens over an infinite history.
          </p>
          <p>
            Here, “contribution” means the difference an input would make to rewards under a
            specified counterfactual, including later feedback. An audit can prove that work
            happened. It cannot, by itself, tell us how much less reward there would have been
            without that work.
          </p>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section__label">
          <span>02 / Evidence</span>
          <h2>What counts as a real comparison</h2>
        </div>
        <div className="method-section__content">
          <p className="method-section__lead">
            Think of a contest whose favorite becomes harder to beat after each win. Another
            win still counts as activity, but adds little evidence about how strongly inputs
            affect the odds. The paper measures the chance left for the next reward to go to
            someone other than the current favorite.
          </p>
          <div className="method-equation">
            <span className="panel__meta">How open the contest remains</span>
            <EquationMath
              latex="\begin{aligned}\varepsilon_t(\beta)&=1-\max_i p_{it}(\beta)\\[4pt] B_T(\beta)&=\sum_{t=0}^{T-1}\varepsilon_t(\beta)\end{aligned}"
              label="Contest openness and total comparison"
            />
            <p>
              <EquationMath latex="\varepsilon_t" block={false} /> is the chance left for a
              competitor other than the favorite. <EquationMath latex="B_T" block={false} />{" "}
              adds those chances over time.
            </p>
          </div>
          <div className="method-facts">
            <div>
              <strong>1,000 allocations at 50–50</strong>
              <p>
                In a two-person contest that stays evenly balanced, each allocation adds
                0.5 to the comparison budget. Total: 500.
              </p>
            </div>
            <div>
              <strong>1,000 allocations at 99.9–0.1</strong>
              <p>
                If the favorite’s chance stays at 99.9%, each allocation adds 0.001.
                Total: 1. The transaction count is the same; the comparison budget is not.
              </p>
            </div>
          </div>
          <p className="method-note">
            These are hypothetical constant odds. The budget adds probabilities; it is not
            the observed number of upsets or a count of independent experiments. More budget
            permits more information, but does not guarantee it: the inputs must also differ
            in informative ways.
          </p>
          <div className="method-equation method-equation--quiet">
            <span className="panel__meta">Why this limits information</span>
            <EquationMath
              latex="\operatorname{tr}I_t(\beta)\le D_X^2\varepsilon_t(\beta)"
              label="Information is bounded by remaining contest openness"
            />
            <p>
              <EquationMath latex="D_X" block={false} /> bounds the distance between input
              profiles. With that bound, information about the contribution parameter is
              at most <EquationMath latex="D_X^2" block={false} /> times the chance left
              outside the favorite. This is an upper bound, not an equality between
              comparison and information.
            </p>
          </div>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section__label">
          <span>03 / The theorem</span>
          <h2>What the paper proves</h2>
        </div>
        <div className="method-section__content">
          <p className="method-section__lead">
            Under the conditions below, if total comparison is finite with probability one
            under every parameter, no method using one market history can consistently learn
            a nonconstant contribution quantity at every parameter. “Consistently” means its
            estimation errors tend to zero as the history grows.
          </p>
          <div className="method-equation method-equation--theorem">
            <span className="panel__meta">The finite-budget implication</span>
            <EquationMath
              latex="\begin{gathered} \mathbb P_\beta\!\left(B_\infty(\beta)<\infty\right)=1 \\ \text{for every }\beta\in\Theta \\[4pt] \Downarrow \\[4pt] \mathbb P_\beta\sim\mathbb P_{\beta'}\quad\text{for every }\beta,\beta'\in\Theta \end{gathered}"
              label="Finite comparison implies equivalent history laws and no universal consistent recovery"
            />
            <p>
              The laws of complete histories have the same probability-zero events, though
              they may give different probabilities to possible outcomes. Data can favor one
              parameter over another. They cannot make a method’s errors vanish at every
              parameter for a contribution quantity that changes with the parameter.
            </p>
          </div>
          <div className="method-facts">
            <div>
              <strong>Smaller chances can still add up forever</strong>
              <p>
                A sequence like 1/2, 1/3, 1/4, … approaches zero, but its sum is infinite.
                Shrinking contestability alone does not establish the theorem’s condition.
              </p>
            </div>
            <div>
              <strong>Finite total comparison is stronger</strong>
              <p>
                Chances of 1/2, 1/4, 1/8, … sum to just 1, even over infinitely many rounds.
                The theorem needs this kind of finite sum under every parameter, together
                with its other assumptions.
              </p>
            </div>
          </div>
          <p className="method-note">
            In the paper’s finite-agent model with fixed inputs and feedback{" "}
            <EquationMath latex="g(z)=z^\rho" block={false} />,{" "}
            <EquationMath latex="\rho>1" block={false} /> implies eventual allocation
            monopoly and a finite budget with probability one. A concentrated chart at a
            finite date does not establish that conclusion by itself.
          </p>
          <details className="method-details">
            <summary>Formal conditions and boundaries</summary>
            <ul>
              <li>
                Parameters share the same initial law and observed history space. Inputs
                and state indices are the same predictable functions of the observed past.
              </li>
              <li>
                For every parameter pair, the one-step laws allow the same recipients after
                each history. This is what “local equivalence” means here.
              </li>
              <li>
                Hellinger separation in both directions is controlled by the remaining
                comparison.
              </li>
              <li>
                Total comparison is finite with probability one under every parameter in
                the stated parameter set.
              </li>
              <li>
                Any additional observed process with parameter-dependent information must be
                included.
              </li>
              <li>
                The conclusion is mutual absolute continuity of complete-history laws, not
                equality of distributions.
              </li>
              <li>
                The theorem also rules out tests whose two error probabilities both vanish,
                and confidence sets that shrink to a point while their coverage tends to one
                at every parameter.
              </li>
            </ul>
          </details>
          <p className="method-note">
            <Link href="/paper">Source paper</Link>: sections 2–3, Proposition 1 and
            Theorem 1; Appendices A–D give the information bounds and proofs.
          </p>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section__label">
          <span>04 / Design</span>
          <h2>What can keep learning alive</h2>
        </div>
        <div className="method-section__content">
          <p className="method-section__lead">
            Extending the same history is different from running fresh comparisons.
            Market and platform design can keep alternatives exposed or create independent
            starts. Learning still requires inputs and observations that distinguish the
            contribution parameters of interest.
          </p>
          <div className="method-options">
            <div>
              <strong>Give newcomers real exposure</strong>
              <p>Repeated randomized exposure can keep a real chance available outside the favorite.</p>
            </div>
            <div>
              <strong>Create independent starts</strong>
              <p>
                Independent markets with a shared contribution parameter can add evidence.
                A reset helps only if it supplies a genuinely fresh comparison.
              </p>
            </div>
            <div>
              <strong>Let people and firms reach buyers elsewhere</strong>
              <p>
                Portability and multihoming can preserve distinct routes to audiences and
                buyers. Several storefronts sharing one ranking need not be independent paths.
              </p>
            </div>
            <div>
              <strong>Limit control over discovery</strong>
              <p>
                Public options, independent procurement and structural separation may preserve
                different paths. Their benefits must be weighed against their costs.
              </p>
            </div>
          </div>
          <p className="method-note">
            Random exposure changes the allocation rule. The paper’s replication result
            requires independent markets whose observable distributions distinguish the
            parameters; more channels alone do not guarantee identification. See section 4
            and Appendix E.1 of the <Link href="/paper">paper</Link>.
          </p>
        </div>
      </section>

      <section className="method-section method-section--last">
        <div className="method-section__label">
          <span>05 / Policy</span>
          <h2>What the result changes</h2>
        </div>
        <div className="method-section__content">
          <div className="method-boundaries">
            <div>
              <span>Competition can produce evidence</span>
              <p>
                Independent routes to market can help us learn why someone succeeds.
                The paper asks how many independent comparison histories remain; it does
                not conclude that every merger is harmful.
              </p>
            </div>
            <div>
              <span>An exact rent tax faces a separate limit</span>
              <p>
                If two economies produce exactly the same observable law but assign
                different contribution to the same reward on records with positive
                probability, a tax based on the record must be the same in both. It cannot
                equal reward minus contribution in both. This is Theorem 2’s additional
                condition.
              </p>
            </div>
          </div>
          <p className="method-note">
            Exact equality of observable laws can arise when unobserved initial position
            offsets a change in the input effect (section 3 and Appendix E.2). That is
            distinct from Theorem 1’s equivalent, potentially different laws. Neither result
            says that work is irrelevant or all income is rent; neither selects a tax rate.
          </p>
        </div>
      </section>
    </main>
  );
}

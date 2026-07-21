import type { Metadata } from "next";
import Link from "next/link";

import { Math as EquationMath } from "@/components/ui/math";
import { PAPER } from "@/lib/paper/citation";

export const metadata: Metadata = {
  title: "Methodology",
  description:
    "A concise guide to how self-reinforcing markets close competition, erase useful comparisons, and weaken claims that rewards measure contribution.",
};

export default function MethodologyPage() {
  return (
    <main className="method-page" id="main-content">
      <header className="method-page__header">
        <p className="eyebrow">Methodology and scope</p>
        <h1>How the argument works</h1>
        <p>
          The animation uses familiar self-reinforcement to show the setup. The theorem asks a
          different question: when that process closes off other paths, does one realized
          market still contain enough comparison to recover contribution? This page shows the
          shortest route from the familiar mechanism to the paper’s distinct result.
        </p>
        <div className="button-row method-page__actions">
          <a className="button button--primary" href={PAPER.url} target="_blank" rel="noreferrer">
            Read the source paper
          </a>
          <Link className="button" href="/math">
            See the central equation
          </Link>
        </div>
      </header>

      <ol className="method-map" aria-label="Methodology overview">
        <li>
          <span>01</span>
          <strong>The market chooses</strong>
          <p>Contribution-related inputs and accumulated advantage shape who wins next.</p>
        </li>
        <li>
          <span>02</span>
          <strong>The contest can close</strong>
          <p>An early favorite may receive nearly all later opportunities.</p>
        </li>
        <li>
          <span>03</span>
          <strong>Evidence can run out</strong>
          <p>More activity may repeat the lead without meaningfully testing anyone else.</p>
        </li>
      </ol>

      <section className="method-section">
        <div className="method-section__label">
          <span>01 / Simulation</span>
          <h2>What the app shows</h2>
        </div>
        <div className="method-section__content">
          <p className="method-section__lead">
            The general model gives each person or firm a score. That score combines a
            verified input with advantage already accumulated. The market turns the scores
            into the probabilities of receiving the next opportunity.
          </p>
          <div className="method-equation">
            <span className="panel__meta">Who receives the next opportunity</span>
            <EquationMath
              latex="p_{it}(\beta)=\frac{\exp(\beta x_i)(a+N_i(t))^\rho}{\sum_j\exp(\beta x_j)(a+N_j(t))^\rho}"
              label="Reinforced allocation probability"
            />
            <p>
              Contribution-related signal × accumulated advantage → chance of winning next.
            </p>
          </div>
          <div className="method-facts">
            <div>
              <strong>The story example</strong>
              <p>
                Twenty-four equally good creators compete for 1,600 recommendations. The
                feedback strength is fixed at 1.55.
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
            Fixed random seeds make every replay reproducible. The numbers illustrate the
            mechanism; they are not forecasts for a real platform or industry.
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
            A recommendation, contract or sale is informative only when more than one
            competitor has a meaningful chance. If the favorite is almost certain to win, the
            market reveals almost nothing about everyone else.
          </p>
          <div className="method-equation">
            <span className="panel__meta">How open the contest remains</span>
            <EquationMath
              latex="\varepsilon_t(\beta)=1-\max_i p_{it}(\beta),\qquad B_T(\beta)=\sum_{t=0}^{T-1}\varepsilon_t(\beta)"
              label="Contest openness and total comparison"
            />
            <p>
              <EquationMath latex="\varepsilon_t" block={false} /> is the chance left for a
              competitor other than the favorite. <EquationMath latex="B_T" block={false} />{" "}
              adds those chances over time.
            </p>
          </div>
          <div className="method-equation method-equation--quiet">
            <span className="panel__meta">Why this limits information</span>
            <EquationMath
              latex="\operatorname{tr}I_t(\beta)\le D_X^2\varepsilon_t(\beta)"
              label="Information is bounded by remaining contest openness"
            />
            <p>
              As the chance left for everyone else approaches zero, the new information about
              contribution must also approach zero.
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
            Under the formal conditions below, finite total comparison means that one complete
            history cannot support a method that consistently learns every nonconstant measure
            of contribution.
          </p>
          <div className="method-equation method-equation--theorem">
            <span className="panel__meta">The result in one line</span>
            <EquationMath
              latex="\begin{aligned} B_\infty(\beta)<\infty\ \text{for every }\beta &\Longrightarrow \mathbb P_\beta\sim\mathbb P_{\beta'} \\[4pt] &\Longrightarrow\ \text{no universal consistent recovery of }F(\beta) \end{aligned}"
              label="Finite comparison implies equivalent history laws and no universal consistent recovery"
            />
            <p>
              The histories are not identical. They overlap too much for one realized history
              to identify every contribution measure consistently.
            </p>
          </div>
          <details className="method-details">
            <summary>Formal conditions and boundaries</summary>
            <ul>
              <li>The design is common and predictable from the same observed past.</li>
              <li>Nearby parameter values have locally equivalent one-step laws.</li>
              <li>
                Hellinger separation in both directions is controlled by the remaining
                comparison.
              </li>
              <li>Total comparison is finite under every parameter being compared.</li>
              <li>
                Any additional observed process with parameter-dependent information must be
                included.
              </li>
              <li>
                The conclusion is mutual absolute continuity of complete-history laws, not
                equality of distributions.
              </li>
            </ul>
          </details>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section__label">
          <span>04 / Design</span>
          <h2>What can keep learning alive</h2>
        </div>
        <div className="method-section__content">
          <p className="method-section__lead">
            The result is not a law of nature. Market and platform design can preserve new
            opportunities to compare people and firms.
          </p>
          <div className="method-options">
            <div>
              <strong>Give newcomers real exposure</strong>
              <p>Random discovery prevents the current favorite from becoming nearly certain.</p>
            </div>
            <div>
              <strong>Create independent starts</strong>
              <p>Resets and separate channels produce evidence that one continuous ranking cannot.</p>
            </div>
            <div>
              <strong>Let people and firms reach buyers elsewhere</strong>
              <p>
                Portability, open standards and multihoming stop one platform or distribution
                channel from becoming the only record.
              </p>
            </div>
            <div>
              <strong>Limit control over discovery</strong>
              <p>Public options and structural separation can create genuinely different paths.</p>
            </div>
          </div>
          <p className="method-note">
            Random exposure is an intervention: it changes the platform rule rather than
            merely measuring the original one.
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
              <span>The economic conclusion</span>
              <p>
                Market rankings cannot by themselves settle moral or political questions about
                desert. They do not reveal a clean earned-versus-unearned split.
              </p>
            </div>
            <div>
              <span>The democratic conclusion</span>
              <p>
                Tax rates, public ownership, UBI and social dividends remain collective
                choices. They should be decided openly around power, security, freedom and
                shared prosperity, not outsourced to a market score.
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

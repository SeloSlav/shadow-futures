import type { Metadata } from "next";
import Link from "next/link";

import { Math as EquationMath } from "@/components/ui/math";

export const metadata: Metadata = {
  title: "The mathematics",
  description:
    "Follow the paper’s model from verified inputs and accumulated position to its precise limit on learning contribution from one market history.",
  alternates: { canonical: "/math" },
  openGraph: {
    type: "article",
    title: "The Shadow Futures mathematics",
    description:
      "The allocation model, comparison budget, and assumptions behind the single-history learning limit.",
    url: "/math",
  },
};

const CENTRAL_EQUATION = String.raw`
\begin{gathered}
\begin{aligned}
w_{it}(\beta)
  &= \exp\bigl(x_{it}^{\top}\beta+s_{it}\bigr) \\[6pt]
p_{it}(\beta)
  &= \frac{w_{it}(\beta)}{\sum_{j=1}^{n} w_{jt}(\beta)} \\[6pt]
\varepsilon_t(\beta)
  &= 1-\max_i p_{it}(\beta) \\[6pt]
B_\infty(\beta)
  &= \sum_{t=0}^{\infty}\varepsilon_t(\beta)
\end{aligned}\\[8pt]
\mathbb P_\beta\!\left(B_\infty(\beta)<\infty\right)
  =1\\[-2pt]
\text{for every }\beta\in\Theta\\[6pt]
\Downarrow\\[4pt]
\text{no estimator of nonconstant }F(\beta)\\[-2pt]
\text{consistent at every }\beta\\[-2pt]
\text{from one history}
\end{gathered}
`;

const SYMBOLS = [
  ["x_{it}", "Person or firm i’s verified input profile at time t: work, quality, effort, or capital at risk."],
  ["\\beta", "The parameter, or set of weights, describing how those inputs affect reward odds."],
  ["s_{it}", "Accumulated position: for example, reputation, ranking or prior sales."],
  ["w_{it}", "A positive allocation weight combining the input effect with accumulated position."],
  ["p_{it}", "The conditional chance that i receives the next reward."],
  ["\\varepsilon_t", "The chance left for someone other than the current favorite."],
  ["B_\\infty", "Those remaining chances added over the entire history: the comparison budget."],
  ["F(\\beta)", "A chosen contribution quantity that changes with the input-effect parameter."],
  ["\\mathbb P_\\beta,\\ \\Theta", "The law of histories under parameter β, and the set of parameters under consideration."],
];

export default function MathPage() {
  return (
    <main className="math-page math-page--focused" id="main-content">
      <header className="math-page__header">
        <p className="eyebrow">The mathematics</p>
        <h1>From winning odds to the learning limit.</h1>
        <p>
          Verified inputs can affect rewards while past rewards affect future access.
          The paper asks whether an observer can learn the input effect from one resulting
          history. Follow the model, the comparison budget and the condition that makes
          consistent learning impossible.
        </p>
      </header>

      <section className="core-equation" aria-labelledby="core-equation-title">
        <div className="core-equation__heading">
          <div>
            <span className="panel__meta">The model and the theorem</span>
            <h2 id="core-equation-title">Read it from top to bottom</h2>
          </div>
          <span>The final implication requires all the assumptions below</span>
        </div>
        <EquationMath
          className="core-equation__formula"
          latex={CENTRAL_EQUATION}
          label="The central Shadow Futures equation"
        />
      </section>

      <section className="equation-reading" aria-labelledby="equation-reading-title">
        <div className="equation-reading__intro">
          <span className="panel__meta">Four steps, four ideas</span>
          <h2 id="equation-reading-title">How the pieces fit together</h2>
        </div>
        <ol>
          <li>
            <span>01</span>
            <div>
              <h3>Combine input effects and position</h3>
              <p>
                <EquationMath latex="x_{it}^{\top}\beta" block={false} /> describes the input
                effect; <EquationMath latex="s_{it}" block={false} /> describes accumulated
                position. They combine into a positive allocation weight{" "}
                <EquationMath latex="w_{it}" block={false} />. A better input can improve
                winning odds even when position also matters.
              </p>
              <p>
                The app uses the special case{" "}
                <EquationMath latex="s_{it}=\rho\log(a+N_i(t))" block={false} />:
                previous rewards raise later winning odds. With fixed inputs and a finite
                set of competitors, <EquationMath latex="\rho>1" block={false} /> gives the
                paper’s strong-reinforcement case.
              </p>
            </div>
          </li>
          <li>
            <span>02</span>
            <div>
              <h3>Choose who receives the next opportunity</h3>
              <p>
                Divide each competitor’s weight by the total weight. The resulting
                probabilities add up to one. At the same observed history, a higher relative
                weight means a better chance of receiving the next recommendation,
                customer, contract or sale.
              </p>
            </div>
          </li>
          <li>
            <span>03</span>
            <div>
              <h3>Measure how open the contest remains</h3>
              <p>
                <EquationMath latex="\varepsilon_t" block={false} /> is the chance that the
                next reward goes to anyone except the current favorite. Add these chances
                over time to obtain the comparison budget. If the favorite has a 99.9%
                chance on each of 1,000 rounds, those rounds add only 1 to the budget.
                At 50–50, a two-person contest adds 500.
              </p>
            </div>
          </li>
          <li>
            <span>04</span>
            <div>
              <h3>Apply the finite-budget condition</h3>
              <p>
                The condition must hold with probability one under every parameter in{" "}
                <EquationMath latex="\Theta" block={false} />. Then, with the other theorem
                assumptions, no method can estimate a nonconstant contribution quantity from
                one history with errors tending to zero at every parameter.
              </p>
              <p>
                Shrinking chances alone are insufficient: 1/2 + 1/3 + 1/4 + … is infinite.
                Finite total comparison is stronger: 1/2 + 1/4 + 1/8 + … equals 1.
              </p>
            </div>
          </li>
        </ol>
      </section>

      <section className="core-equation" aria-labelledby="math-assumptions-title">
        <div className="core-equation__heading">
          <div>
            <span className="panel__meta">The scope of the implication</span>
            <h2 id="math-assumptions-title">What must also be true</h2>
          </div>
        </div>
        <details className="method-details">
          <summary>Theorem 1’s formal assumptions, in words</summary>
          <ul>
            <li>
              All parameters share the same initial law and observed-history space.
              Inputs and position are common predictable functions of the past.
            </li>
            <li>
              After each history, the one-step laws have the same possible recipients
              for every parameter pair (“local equivalence”).
            </li>
            <li>
              For every parameter pair, the one-step Hellinger separation is bounded by
              a finite constant times residual contestability, in both parameter directions.
            </li>
            <li>
              All parameter-dependent information in other observed variables must be
              included in the experiment.
            </li>
          </ul>
        </details>
        <p className="method-note">
          The model is equation (1); the budget is equations (2)–(3); the learning limit is
          Theorem 1 in the <Link href="/paper">paper</Link>. The simulation’s feedback rule
          is a special case of Appendix D.
        </p>
      </section>

      <section className="symbol-key" aria-labelledby="symbol-key-title">
        <div>
          <span className="panel__meta">The symbols</span>
          <h2 id="symbol-key-title">A compact key</h2>
        </div>
        <dl>
          {SYMBOLS.map(([symbol, definition]) => (
            <div key={symbol}>
              <dt>
                <EquationMath latex={symbol} block={false} />
              </dt>
              <dd>{definition}</dd>
            </div>
          ))}
        </dl>
      </section>

      <aside className="math-boundary">
        <span className="panel__meta">Read the conclusion precisely</span>
        <h2>A real input effect need not be recoverable.</h2>
        <p>
          Theorem 1 gives mutually absolutely continuous complete-history laws: they have
          the same probability-zero events, but need not be equal. Observations can favor one
          parameter. The limit is that no estimator works consistently at every parameter for
          any nonconstant contribution quantity, even as that one history grows forever.
        </p>
        <p>
          Exactly identical observable laws arise in a separate model where unobserved
          initial position can offset the input effect. The paper uses that stronger
          non-identification condition in its exact contribution-versus-rent tax result.
          Neither result says that productive inputs have no effect.
        </p>
      </aside>

      <div className="button-row math-page__actions">
        <Link className="button button--primary" href="/methodology">
          See the assumptions
        </Link>
        <Link className="button" href="/paper">
          Read the proof
        </Link>
      </div>
    </main>
  );
}

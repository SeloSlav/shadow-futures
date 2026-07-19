import type { Metadata } from "next";
import Link from "next/link";

import { Math as EquationMath } from "@/components/ui/math";
import { PAPER } from "@/lib/paper/citation";

export const metadata: Metadata = {
  title: "The central equation",
  description:
    "One combined equation explains how contribution, accumulated attention, and disappearing competition limit what can be learned from one market history.",
};

const CENTRAL_EQUATION = String.raw`
\begin{aligned}
s_{it}(\beta)
  &= \exp(\beta x_i)\bigl(a+N_i(t)\bigr)^\rho \\[6pt]
p_{it}(\beta)
  &= \frac{s_{it}(\beta)}{\sum_j s_{jt}(\beta)} \\[6pt]
\varepsilon_t(\beta)
  &= 1-\max_i p_{it}(\beta) \\[6pt]
\sum_{t=0}^{\infty}\varepsilon_t(\beta)<\infty
  &\quad\Longrightarrow\quad
  \begin{gathered}
  \text{no universal consistent recovery}\\[-2pt]
  \text{of }F(\beta)\text{ from one history}
  \end{gathered}
\end{aligned}
`;

const SYMBOLS = [
  ["x_i", "Agent i’s verified contribution-related input."],
  ["\\beta", "How strongly that input affects the chance of winning the next opportunity."],
  ["N_i(t)", "The attention, customers or sales agent i has already accumulated."],
  ["\\rho", "How strongly an accumulated advantage produces more advantage."],
  ["p_{it}", "The chance that agent i receives the next opportunity."],
  ["\\varepsilon_t", "The chance left for an agent other than the current favorite."],
];

export default function MathPage() {
  return (
    <main className="math-page math-page--focused" id="main-content">
      <header className="math-page__header">
        <p className="eyebrow">The mathematics</p>
        <h1>One equation carries the whole argument.</h1>
        <p>
          It connects three ideas: how a platform or market awards the next opportunity, how
          the contest closes, and why one observed history may never reveal the exact
          contribution of a person or firm.
        </p>
      </header>

      <section className="core-equation" aria-labelledby="core-equation-title">
        <div className="core-equation__heading">
          <div>
            <span className="panel__meta">The complete chain</span>
            <h2 id="core-equation-title">Read it from top to bottom</h2>
          </div>
          <span>Under the paper’s theorem assumptions</span>
        </div>
        <EquationMath
          className="core-equation__formula"
          latex={CENTRAL_EQUATION}
          label="The central Shadow Futures equation"
        />
      </section>

      <section className="equation-reading" aria-labelledby="equation-reading-title">
        <div className="equation-reading__intro">
          <span className="panel__meta">Four lines, four ideas</span>
          <h2 id="equation-reading-title">What each line is saying</h2>
        </div>
        <ol>
          <li>
            <span>01</span>
            <div>
              <h3>Build each competitor’s score</h3>
              <p>
                The score combines a contribution-related input with advantage the person or
                firm already has—such as attention, customers or past sales. When{" "}
                <EquationMath latex="\rho>1" block={false} />, that advantage can feed on
                itself strongly.
              </p>
            </div>
          </li>
          <li>
            <span>02</span>
            <div>
              <h3>Choose who receives the next opportunity</h3>
              <p>
                Each score becomes a probability. A higher score means a better chance of
                receiving the next recommendation, customer, contract or sale.
              </p>
            </div>
          </li>
          <li>
            <span>03</span>
            <div>
              <h3>Measure how open the contest remains</h3>
              <p>
                <EquationMath latex="\varepsilon_t" block={false} /> is the chance that the
                next opportunity goes to anyone except the current favorite. Near zero,
                almost no other person or firm gets a real shot.
              </p>
            </div>
          </li>
          <li>
            <span>04</span>
            <div>
              <h3>See what one history cannot tell us</h3>
              <p>
                If those remaining chances add up to only a finite amount, watching forever
                does not create endless new comparisons. Under the theorem’s other conditions,
                no method can consistently recover every nonconstant contribution measure from
                that one history.
              </p>
            </div>
          </li>
        </ol>
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
        <span className="panel__meta">The economic meaning</span>
        <h2>The winner’s contribution does not make the market score a proof.</h2>
        <p>
          The score records contribution together with accumulated advantage. When the
          theorem’s formal conditions hold, the one history cannot recover the exact split—so
          the ranking cannot serve as a neutral certificate of desert.
        </p>
      </aside>

      <div className="button-row math-page__actions">
        <Link className="button button--primary" href="/methodology">
          See the assumptions
        </Link>
        <a className="button" href={PAPER.url} target="_blank" rel="noreferrer">
          Read the proof
        </a>
      </div>
    </main>
  );
}

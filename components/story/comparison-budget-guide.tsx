import Link from "next/link";

export function ComparisonBudgetGuide() {
  return (
    <section className="budget-guide" aria-labelledby="budget-guide-title">
      <header>
        <span className="panel__meta">The comparison budget</span>
        <h3 id="budget-guide-title">Same 1,000 transactions. Different chances to learn.</h3>
        <p>
          At each step, add the chance that someone other than the favorite wins.
          Here’s how much that adds up to over 1,000 transactions if the odds stay fixed.
        </p>
      </header>
      <div className="budget-guide__examples">
        {[
          { leader: "50%", other: "50%", width: "50%", budget: "500", formula: "1,000 × 0.5", label: "An open contest" },
          { leader: "99.9%", other: "0.1%", width: "99.9%", budget: "1", formula: "1,000 × 0.001", label: "A nearly closed contest" },
        ].map((example) => (
          <article key={example.label}>
            <h4>{example.label}</h4>
            <div className="budget-guide__bar" aria-hidden="true">
              <span style={{ width: example.width }} />
            </div>
            <div className="budget-guide__key">
              <span>Favorite <strong>{example.leader}</strong></span>
              <span>Others <strong>{example.other}</strong></span>
            </div>
            <div className="budget-guide__calculation">
              <span>{example.formula} =</span>
              <strong>{example.budget}</strong>
            </div>
            <span className="budget-guide__unit">budget units</span>
          </article>
        ))}
      </div>
      <p className="budget-guide__note">
        A larger comparison budget leaves more room to learn. The budget counts chances
        to compare, and a large total alone can’t establish contribution.
      </p>
      <details className="story-details budget-guide__boundary">
        <summary>Why tiny chances can still add up to an infinite budget</summary>
        <p>
          A constant 0.1% chance keeps adding comparison forever. The theorem needs the
          entire future sum to be finite: for example, chances that halve each round add
          up to only 1 (½ + ¼ + ⅛ + …). Under the paper’s other conditions, no method can
          reliably recover contribution across all possible contribution values from one
          history, even by watching forever. Learning also depends on whether competitors’
          measured inputs differ.
        </p>
        <Link href="/methodology">See the assumptions and the precise result →</Link>
      </details>
    </section>
  );
}

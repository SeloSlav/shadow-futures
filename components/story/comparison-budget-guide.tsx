import Link from "next/link";

export function ComparisonBudgetGuide() {
  return (
    <section className="budget-guide" aria-labelledby="budget-guide-title">
      <header>
        <span className="panel__meta">The comparison budget, without the jargon</span>
        <h3 id="budget-guide-title">Same 1,000 transactions. Different chances to learn.</h3>
        <p>
          At each step, take the chance that the next reward goes to anyone except the
          favorite. Add those chances. That sum is the paper’s comparison budget.
          Here the favorite means whoever has the highest chance of winning next.
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
              <span>Favorite: <strong>{example.leader}</strong></span>
              <span>Everyone else: <strong>{example.other}</strong></span>
            </div>
            <p>If these chances stay fixed for 1,000 transactions:</p>
            <div className="budget-guide__calculation">
              <span>{example.formula} =</span>
              <strong>{example.budget}</strong>
            </div>
            <span className="budget-guide__unit">units of comparison budget</span>
          </article>
        ))}
      </div>
      <p className="budget-guide__note">
        The budget counts chances, not actual wins or independent experiments. In the paper’s
        model it bounds how much information allocations can provide. A larger budget leaves
        more room to learn; it doesn’t guarantee learning, especially if competitors’ measured
        inputs are identical.
      </p>
      <aside className="budget-guide__boundary">
        <strong>Tiny chances alone do not prove impossibility.</strong>
        <p>
          A constant 0.1% chance keeps adding comparison forever. The theorem needs the
          entire future sum to be finite: for example, chances that halve each round add
          up to only 1 (½ + ¼ + ⅛ + …). Under the paper’s other conditions, no method can
          reliably recover contribution across all possible contribution values from one
          history, even by watching forever.
        </p>
        <Link href="/methodology">See the assumptions and the precise result →</Link>
      </aside>
    </section>
  );
}

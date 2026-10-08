export const HUNCHROOM_REQUEST_URL = "https://hunchroom.com/p/44";

export function ResearchInvitation() {
  return (
    <section
      className="paper-section research-invitation"
      id="contribute"
      aria-labelledby="contribute-title"
    >
      <div className="paper-section__label">
        <span>Open research</span>
        <h2 id="contribute-title">Contribute on Hunchroom</h2>
      </div>
      <div className="paper-section__content">
        <p>
          The finite comparison-budget impossibility theorem now has a formal proof
          request on Hunchroom. Contribute a Lean proof, review how the formal statement
          matches the paper, or share a useful partial attempt.
        </p>
        <p className="research-invitation__scope">
          Request #44 covers a finite-recipient version of Theorem 1 and Appendix B,
          with scalar contribution quantities. The Lean statement has been typechecked;
          proof verification is a separate step.
        </p>
        <div className="button-row research-invitation__actions">
          <a
            className="button button--primary"
            href={HUNCHROOM_REQUEST_URL}
            target="_blank"
            rel="noreferrer"
          >
            Open request #44 <span aria-hidden="true">↗</span>
          </a>
        </div>
      </div>
    </section>
  );
}

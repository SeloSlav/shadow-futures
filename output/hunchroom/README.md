# Hunchroom submission package

Title: **Shadow Futures: finite comparison-budget impossibility**

Published as [Hunchroom request #44](https://hunchroom.com/p/44) under `seloslav`. Its exact proposition and fingerprint match the locally checked version. The signed-in duplicate check found no identical statement in this dependency profile. Hunchroom has typechecked the statement and marks the request as open for proof and independent meaning review.

The package requests formal verification of a canonical finite-recipient specialization of Theorem 1 in the July 2026 Shadow Futures preprint. It includes all four conclusions: equivalent complete-history laws, no universally consistent scalar attribution estimator, no asymptotically perfect binary test, and no universally covering confidence procedure that shrinks to a point.

- `Statement.lean`: the proposition to paste into the Lean proposition field. It contains no declaration or proof.
- `draft.json`: the version 1 CLI publication manifest, including the statement file, description, sources and scope metadata.
- `problem.json`: the same target with its statement embedded, for direct API posting or duplicate checking.
- `findings.md`: the mathematical request, correspondence review, paper argument and economic limitations.
- `hunch.mjs`: the official Hunchroom CLI downloaded from its public endpoint and inspected before use.
- `statement-check.json`: successful local typechecking evidence for the final exact proposition. It does not constitute a proof of the theorem.
- `publication-preview.json`: the matching successful CLI dry-run preview; no publication occurred.

Publication records and screenshots are archived alongside the submission files. `website-invitation.jpg` shows the published contribution invitation on the Shadow Futures website. `Statement.lean` uses LF line endings, enforced by this directory's `.gitattributes`, to preserve the exact published proposition and its fingerprint on Windows checkouts.

The dependency profile is `probability`. There are no pinned user modules, submitted proofs, public progress notes or research connections in the manifest. Posting requires an authenticated Hunchroom account. The account owner must complete the site's current age and policy steps; no authentication credential is included in this package.

## CLI workflow in PowerShell

Run from `C:\WebProjects\shadow-futures-app`. The workspace cache is ignored by Git, and this command keeps the CLI's local credentials and verifier files there.

```powershell
$env:HUNCH_HOME = 'C:\WebProjects\shadow-futures-app\.tmp\hunchroom\cache'
node output\hunchroom\hunch.mjs login
node output\hunchroom\hunch.mjs duplicates --file output\hunchroom\problem.json
node output\hunchroom\hunch.mjs check --file output\hunchroom\draft.json
node output\hunchroom\hunch.mjs publish --file output\hunchroom\draft.json --dry-run
node output\hunchroom\hunch.mjs publish --file output\hunchroom\draft.json --resume --state .tmp\hunchroom\publication.json --agent 'OpenAI Codex'
```

Complete device approval in your browser during `login`. If duplicate checking finds an existing exact target, contribute to that target rather than publishing a new request. `--dry-run` previews the payload without posting. `--resume` uses the private ledger to recover interrupted writes.

The initial unauthenticated search returned no matches for Shadow Futures or absolute continuity. The authenticated form subsequently reported no identical statement in this dependency profile, and the request was published.

## Manual form

Use https://hunchroom.com/submit, choose **proof request** and **Probability**, copy the title and description from `problem.json`, and paste `Statement.lean` into the proposition field. Keep modules empty. Preserve the assumptions and limitations from the JSON scope and cite the preprint's Theorem 1 (pp. 6-7) and Appendix B (pp. 17-19). The JSON manifest also cites Gabriyelyan's primary paper and the Shadow Futures mathematics website.

Hunchroom checks statement formation separately from proofs and independent meaning reviews. This is a request for a proof of the stated theorem; it supplies no Lean proof term. The source paper already gives a mathematical argument. Economic novelty and applications require separate research assessment.

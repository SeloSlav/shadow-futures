# Shadow Futures book drafting

Use the bounded workflow in
`tools/skills/bounded-book-writing/SKILL.md` for writing and review.
The user requested economical, finite passes. Start with the latest handoff;
reuse unchanged source checks and stop after the agreed pass. Preserve precise
unresolved flags instead of restarting source-access or whole-book audit loops.

For requested edition updates, use `npm run book:refresh -- --preview "changed phrase"`
once after settling the prose. It builds and validates the website PDF, Kindle
EPUB, and paperback files together, with hash-based reuse when unchanged. Follow
`tools/skills/shadow-futures-publication/SKILL.md`; do not manually rerun all the
underlying builders after a passed export. Publishing still requires user scope.

The user wants to write the book in this repository and preserve progress in Git.
Use `planning/argument-brief.md` and `planning/book-outline.md` as the working
planning text. Write new chapters as Markdown in `manuscript/`. Word files in
`drafts/` are reading exports. The existing build script and JSON preserve the
initial planning version; do not run that builder over subsequent edited drafts.
The text import helper refuses to overwrite existing working copies.

After each completed book-writing request, create a local Git checkpoint of the
book files changed for that request. Inspect the staged changes and stage explicit
paths. Preserve unrelated or unfinished user edits. A local commit does not
authorize pushing to a remote; push only when the user requests it.

Read `README.txt`, the current argument brief, the committed outline, and
`research/v5_reference_register.json` before substantive book work. These record
the user's book direction. Keep an equation-free popular-economics main text.

The approved main-text budget is 80,000 words: prologue 2,500, introduction
3,500, twelve chapters averaging 6,000 (72,000 combined), and epilogue 2,000.
Endnotes, references, and acknowledgments have a separate 10,000–15,000-word
allowance. Use `planning/manuscript-plan.json` and `tools/manuscript_status.py`
to track actual chapter prose separately from back matter. Counts and file
existence do not certify that a chapter or the whole book has passed review.

- Use the October 2026 v5 already at
  `../output/docx/Shadow_Futures_FINAL_Comparison_Budget_Revised_v5.docx`.
  Its Word and PDF copies match `C:\Shadow Futures\Final` byte for byte;
  provenance and SHA256 hashes are in `research/v5_source_provenance.json`.
  Preserve the originals.
- Retain all 24 v5 references and their chapter assignments. Credit relevant
  predecessors where their ideas enter. Put mathematical lineage in readable
  endnotes. Verify the relevant full source and record its page or section before
  finalizing a substantive factual or novelty claim. A bibliography inventory is
  not a completed literature review.
- The first draft requires no interviews. Use public primary records and
  published research. The real recurring object is Lockdown by 52Metro in
  MusicLab. Aster and Birch are explicitly fictional suppliers. Invent no
  dialogue, private reactions, visits, or biography for real people.
- Keep the narrative grounded in documented cases. Weave explanations into
  those cases and return to concrete people, actions, or outcomes frequently;
  avoid long stretches of abstract model exposition. The prologue must sell
  the book's questions. Put detailed literature and theorem conditions in notes
  or the relevant later chapters rather than interrupting its narrative.
- Explain the comparison budget as cumulative modeled allocation chance outside
  the current favorite. It is not actual rival wins, independent trial counts,
  market shares, firm counts, or welfare. Small or shrinking chance alone does
  not establish a finite lifetime budget. A large budget alone does not identify
  input effects. Count all other informative observations in an application.
- Shadow futures preserve the underlying parameter, starting conditions, and
  input and state response rules. Adaptive realized inputs may vary with history.
  Distinguish replays from interventions changing inputs or policies.
- Credit the existing literature on path dependence, cumulative advantage,
  reinforced-urn estimation failure, predictive merging, and prediction alongside
  inconsistent estimation. Credit Meghani's prior use of epistemic monopoly.
  Locate the proposed novelty in the specified comparison-budget bound, precision
  limit, and attribution interpretation. Do not claim completed independent proof
  certification or established scholarly priority.
- Distinguish the single-history finite-budget result from the hidden-position
  observational-equivalence argument. Keep attribution, moral desert, contractual
  entitlement, and incentives separate.
- The only reader-facing self-citation is the original Shadow Futures paper at
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994. Cite it once in
  the introduction and once in the bibliography. Never publish internal version
  labels, companion-paper citations, local research paths, or references to the
  author's other drafts. Keep those files only as historical internal records.
  Ground additional claims in relevant published scholarship. Where no published
  source establishes a result, explain a conditional example or proposal without
  presenting it as a sourced theorem. Never relabel a companion result as a result
  from the SSRN paper or attribute it to a replacement author.
- Write reader-facing endnotes as ordinary citations: author, title, publication,
  date, page or section, and link. Put necessary assumptions and distinctions
  naturally in the body at the relevant claim. Keep editorial instructions,
  audit commentary, source-access reports, and manuscript-status language in
  internal research records. Cite published findings where they enter the prose;
  a citation beside an original example must not imply that its source proves
  the book's proposed policy.
- AI, corrective taxes, and remedy design require additional empirical, causal,
  welfare, and legal premises.
  Include scale benefits, private experimentation, implementation costs, testing
  crowdout, and feasible replacement. Infer no real tax rate, legal liability,
  or infinite-horizon condition from a concentration chart or simulation.
- Alchian (1950), Blackwell (1953), and Arrow (1962 and 1963) are additions to
  the book bibliography, not members of the original 24-reference v5 inventory.
  See `research/classics_manifest.json` for actual downloaded files and link-only
  entries; do not claim every linked paper is locally available.

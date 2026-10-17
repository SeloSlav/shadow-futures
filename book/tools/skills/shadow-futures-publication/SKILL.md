---
name: shadow-futures-publication
description: Refresh Shadow Futures into the website PDF, Kindle EPUB, and paperback KDP files when the user explicitly requests regenerated editions or upload files. Ordinary manuscript edits do not authorize exports; deployment requires the user's authorization.
---

# Shadow Futures publication

Work in `C:/WebProjects/shadow-futures-app`. Read `book/AGENTS.md` and use the bounded book-writing workflow for prose. Keep this skill's tracked copy in `book/tools/skills/shadow-futures-publication/` synchronized with the installed copy when changing its instructions.

Publication is a separate user-requested step. Do not run PDF, EPUB, cover, upload-bundle builders or change website edition metadata merely because manuscript edits are complete. A short prompt such as "regenerate all three" authorizes the build without another confirmation. Authorization for a previous batch does not carry into later editing turns, and a request to commit or push alone does not authorize generation. For manuscript-only work, allow Markdown reader assembly and word/citation checks, record that editions are pending, and stop before export.

Settle the requested prose and relevant source notes before exporting. Research only changed factual claims. Reuse unchanged source checks, cover artwork, formatting, and known KDP specifications. Preserve the 80,000-word main-text cap and separate 15,000-word back-matter cap; make room by tightening repetition in the affected passage. Personal hopes are the author's account, not predictions or evidence of scholarly priority.

After the explicit publication request, run this once after settling the edits:

```powershell
npm run book:refresh -- --preview "a distinctive phrase from the changed passage"
```

This assembles the reader, generates all PDF/EPUB/cover files, packages the four KDP uploads, runs EPUBCheck and the existing PDF integrity check, and saves source/output hashes in `book/research/export-state.json`. It reuses cached EPUBCheck. `BOOK_PYTHON` and `EPUBCHECK_JAR` can override runtime locations. On this Windows host Java requires normal Windows runtime access: request the normal tool escalation for the single command if the sandbox prevents execution; do not burn repeated builds diagnosing an unchanged restriction.

Inspect the changed-page PNGs the command reports. Existing EPUB reflow/browser checks and unchanged cover/layout reviews can be reused for ordinary prose edits. If layout code changes, inspect the affected format. Use `--force` only for a real need to invalidate a cached result. An unchanged run verifies output hashes and skips rendering and validation; it does not repeat the book review.

Deliverables are stable:

- Website: `public/book/shadow-futures.pdf` and `public/book/edition.json`.
- Kindle: `output/kdp/shadow-futures-kindle.epub` and `shadow-futures-kindle-cover.jpg`.
- Paperback: `output/kdp/shadow-futures-paperback-interior.pdf` and `shadow-futures-paperback-cover.pdf`.
- All uploads and instructions: `output/kdp/shadow-futures-kdp-upload.zip`.

Do not run each underlying builder again after a passed pipeline, rebuild for a final answer, retry inaccessible literature, or launch another manuscript audit. On a real failure, fix that cause and rerun. Update one handoff and inspect staged changes. Keep PDF/EPUB/ZIP files binary in Git so their verified bytes survive checkout.

Commit completed work with explicit paths. Push only when publishing/updating the site is requested or already authorized for the active revision. A book export alone does not authorize publishing to KDP. For a site update, push through the existing GitHub/Vercel integration and verify the live PDF hash once after deployment. No local Next.js rebuild is needed for prose-only changes; run it when relevant website code changes.

Use the upload instructions for the actual paper/trim settings and retained cover-resolution limitation. KDP Previewer/print-proof approval remains the author's publication step. Keep KDP Select off while the full free digital edition remains on the site; complete KDP's AI-content disclosure accurately without restoring a deleted public acknowledgment.

# Shadow Futures cover and free reading edition

**Shadow Futures: How Markets Choose Winners—and Lose Sight of the Alternatives**  
**Martin Erlic**

The front cover uses a gold path and unrealized architectural alternatives on a dark petrol ground. The title dominates the top; the author name anchors the bottom. Artwork was created with the built-in image-generation tool, followed by one resolution-only edit request. Both returned a native 1024 × 1536 image. The selected master is saved in `public/book/shadow-futures-cover.png`; prompts are in `cover-generation.json`.

## Files

- `public/book/shadow-futures.pdf`: complete, freely downloadable reading edition, with cover, contents, bookmarks, linked endnotes, references and acknowledgments.
- `public/book/shadow-futures-kindle-cover.jpg`: 1600 × 2560 Kindle marketing cover. The complete design is preserved with narrow matching bands, without cropping its text.
- `output/pdf/shadow-futures-interior-6x9.pdf`: 6 × 9-inch paperback interior; black text with embedded fonts, generous margins, and an even page count.
- `output/pdf/shadow-futures-kdp-paperback-cover.pdf`: full back/spine/front proof, with 0.125-inch outer bleed. Spine uses the actual interior count and KDP's black-ink/white-paper factor of 0.002252 inch per page.
- `output/pdf/shadow-futures-front-cover.pdf`: trimmed 6 × 9 front panel.
- `output/pdf/shadow-futures-print-front-300dpi.jpg`: 1800 × 2700 print-size raster export.

The print-size raster is resampled; it does not have added native detail. These are dimensioned **proof assets**, not a claim of KDP acceptance. Obtain higher-resolution original artwork before a final commercial print upload and check the actual paper/ink choices and KDP preview. The blank box on the back reserves space for KDP's barcode; no ISBN or barcode has been invented.

KDP specifications checked on 16 October 2026: [paperback covers](https://kdp.amazon.com/en_US/help/topic/G201953020), [Kindle covers](https://kdp.amazon.com/en_US/help/topic/G6GTK3T3NUHKLEFX), [trim and margins](https://kdp.amazon.com/en_US/help/topic/GVBQ3CMEQW3W2VL6).

Rebuild with `python book/tools/build_publication.py` using the bundled Python runtime with ReportLab, pypdf, pypdfium2 and Pillow. `book/research/publication-manifest.json` binds the exact manuscript inputs and reports output dimensions, hashes and source-art resolution. Website metadata is generated separately in `public/book/edition.json`. This operation formats the current manuscript; it does not revise or certify its claims.

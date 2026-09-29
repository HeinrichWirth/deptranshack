# Working on Rail Inspector

Read README.md and docs/ARCHITECTURE.md first. This directory is the independent submission repository; the parent workspace is not a runtime dependency.

- Keep `engine/` immutable. `SOURCE_MANIFEST.json` records the accepted Python and configuration sources. Cosmetic edits inside the engine invalidate the freeze just like algorithm edits.
- Modify the web application in `app/` and `web/`. It must never pass future scans or reference maps into inference.
- Preserve original source indices, message timestamps and causal history. No interpolation across missing registration is added by the viewer.
- Automatic 10 m object grouping is isolated in `app/grouping.py`. Keep fixed anchors, independent-frame confirmation and separation between pose generations. `tracks.json`, geometry and source points remain intact.
- Run `docker compose exec inspector python -m unittest discover -s tests -v` after rebuilding. For web changes also run JS syntax checks and the browser test in `tests/ui.cjs`.
- A release needs a real upload/run through `tests/full_http.py`, a cross-build geometry comparison, and updated `docs/RELEASE_VALIDATION.md`.
- Never commit source bags, LAS, run clouds, local credentials or compiled binaries.
- `.gitattributes` disables automatic line-ending conversion because frozen source hashes are byte-based.

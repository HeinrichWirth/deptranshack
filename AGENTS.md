# Working on Rail Inspector

Read README.md and docs/ARCHITECTURE.md first. This directory is the independent submission repository; the parent workspace is not a runtime dependency.

- Keep the verified engine baseline stable. An explicitly requested algorithm update must update `engine/SOURCE_MANIFEST.json`, matching build sources under `native/`, regression tests and validation documentation together. Do not silently reseal unrelated changes.
- Modify the web application in `app/` and `web/`. It must never pass future scans or reference maps into inference.
- Preserve original source indices, message timestamps and causal history. No interpolation across missing registration is added by the viewer.
- Automatic 10 m object grouping is isolated in `app/grouping.py`. Keep fixed anchors, independent-frame confirmation and separation between pose generations. `tracks.json`, geometry and source points remain intact.
- Run `docker compose exec inspector python -m unittest discover -s tests -v` after rebuilding. For web changes also run JS syntax checks and the browser test in `tests/ui.cjs`.
- A release needs a real upload/run through `tests/full_http.py`, a cross-build geometry comparison, and updated `docs/RELEASE_VALIDATION.md`.
- Never commit source bags, LAS, run clouds, local credentials or compiled binaries.
- `.gitattributes` disables automatic line-ending conversion because frozen source hashes are byte-based.

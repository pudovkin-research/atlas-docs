# atlas-ml — documentation

The published documentation for `atlas-ml`, served at
<https://pudovkin-research.github.io/atlas-docs/>.

This repository holds the pages only. They are generated and verified in the library's own
repository — `docs/parameters.md` and `docs/api.md` are produced from the code's signatures by
`docs/build.py`, and a test there fails if they fall out of step — and pushed here by
`scripts/publish_docs.sh`. Edit them there, not here.

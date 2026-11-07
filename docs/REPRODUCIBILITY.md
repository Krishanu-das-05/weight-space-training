# Reproducibility And Archive Policy

## What Is Published

Source modules and legacy tests retain their exact pre-reorganization bytes.
`repository-layout.json` records each original filename, its new location and
SHA-256. Curated reports, figures and aggregate tables are published separately.
Clean notebook copies contain no executed outputs or machine-specific metadata.

This is a source-layout change, not a new experiment or a change to any saved
outcome. The original 23 frozen sources, seven amended sources and original 11
response records are not rewritten to make a source-path check pass.

## What Remains Local

The original flat-layout files are preserved under
`.local/original-layout/`, including original notebook bytes, reports and data
selection files. Its snapshot records byte hashes. Original `results/`,
`models/`, `.cache/`, `.venv/` and Jupyter settings remain where they were.
These are ignored, not deleted, and are never staged.

`scripts/lab.py prepare --with-local-data` constructs `.local/workbench/`.
It copies source/test bytes into the expected flat layout and mirrors existing
local results, public edit data and the approved dataset cache. The large pristine
checkpoint uses a local hardlink to avoid another multi-gigabyte copy. **Do not
edit either checkpoint path:** both names refer to the same immutable file.
No model or dataset is downloaded and no inference occurs.

Preparation refuses to overwrite any differing workbench file. If something
differs, investigate and preserve it; the tool does not delete or repair it.

## Which Checks Mean What

- `python tools/check_repo.py` checks syntax, frozen-source bytes, notebook
  hygiene, publication scope, figure safety and obvious credential patterns.
- `python -m unittest discover -s tests/repository -v` exercises the portable
  packaging/security checks using standard-library fixtures.
- `python scripts/lab.py test` runs **all original 140 offline research tests**
  in the reconstructed workbench. Historical fixtures and the pinned environment
  are required; absence is a failure, not an implicit skip.
- `python scripts/lab.py verify` recomputes the full matched analysis from local
  raw records, checks preserved prior outcomes and independently verifies both
  restoration artifacts. It does not import the model runtime or generate text.

The portable checks are not substitutes for the full research suite. The full
suite and raw-record verification are run locally before publication.

## Boundaries Of The Curated Export

The published aggregate JSON is copied from the saved run and is useful for
displaying tables. It is **not independent re-grading** without the raw records.
Notebook 00 clearly identifies this boundary. The clean full-analysis notebook
07 still reads local raw records by default, with `RUN_EXPERIMENT = False`.

The original decoded-text failure had no saved raw output. The approved amendment
was declared after 11 records, before the remaining 19, and the replacement
attempt cannot be empirically compared with the lost tokens. Its old full
8,192-token charge remains retained.

No cleanup command, new generation, changed grader, weakened source check or
rewritten approval manifest is needed to prepare this repository.

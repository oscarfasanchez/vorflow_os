# Milestone 6 — First PyPI release

**Status:** in progress · **Current target:** TestPyPI `0.1.0rc1` · **Risk:** low · **Behavior change:** none (packaging/metadata only)
**Back to** [ROADMAP.md](../../ROADMAP.md)

## Goal

Publish `vorflow` to PyPI so users can `pip install vorflow`. The name is
**available on PyPI** (verified 2026-07-18: the registry returns 404 for
`vorflow`).

This milestone is being rehearsed on TestPyPI first. The rehearsal stops after
verifying `0.1.0rc1`; real PyPI publication remains a separate approval gate.

## Current state (already publication-grade)

- Modern PEP 621 `pyproject.toml` with setuptools backend and `src/` layout.
- MIT `LICENSE`, classifiers, author metadata, `Repository`/`Issues` URLs.
- Real test suite (11 test files) with a cross-platform CI matrix
  (Linux/Windows/macOS × Python 3.10–3.12) plus a ruff lint job.
- Good README with a runnable Basic Usage example.
- `__version__` resolved via `importlib.metadata` with a fallback.

What is missing is **release plumbing** (build verification, publish workflow,
version/tag flow) and **PyPI-facing polish** (README links, dependency floors,
changelog). Roughly a day of work; nothing is a rewrite.

## Phase 1 — Blockers (must-do before first publish)

- [x] **Merge in-flight work.** `gmshflow_missing` was integrated into local
  `main` after the full test suite passed.
- [ ] **Verify the package builds.** The package has never been built here
  (`build` is not installed in the dev env). Run:

  ```bash
  pip install build twine
  python -m build
  twine check dist/*
  ```

  Inspect the sdist and wheel contents: `.conda/`, `benchmarks/`, `docs/`, and
  the stray `src/vorflow/vorflow.code-workspace` must not leak in (and that
  workspace file should move out of the package directory regardless).
- [ ] **Smoke-test the built wheel in a fresh venv** — `pip install
  dist/vorflow-*.whl`, then run the README Basic Usage example end-to-end.
  This catches missing runtime deps that the editable/conda dev setup hides.
- [ ] **Fix README links for PyPI.** PyPI renders the README but does not
  resolve relative links — `examples/`, `ROADMAP.md`, `LICENSE`, and
  `etc/environment.yml` will all be dead on the PyPI page. Convert them to
  absolute GitHub URLs. Change the Installation section to lead with
  `pip install vorflow`, keeping editable/dev instructions as a development
  subsection.
- [ ] **Add lower bounds to dependencies.** All six runtime deps are unpinned.
  Set floors matching what is actually tested, e.g. `numpy>=1.24`,
  `shapely>=2.0` (Shapely 1.x has an incompatible API), `gmsh>=4.11`,
  plus floors for `pandas`, `geopandas`, `scipy`. No upper caps.
- [ ] **Confirm authorship/attribution.** The LICENSE copyright is `rhugman`
  and they are listed first in `authors` with their email. If `vorflow`
  derives from `gmshflow` that attribution is correct — but confirm rhugman
  agrees to their name/email appearing in public PyPI metadata and to
  publishing under this name.

## Phase 2 — Strongly recommended (do at the same time)

- [ ] **Release/publish workflow.** GitHub Actions workflow using **PyPI
  Trusted Publishing** (OIDC, no API tokens) triggered on `v*` tags:
  build → `twine check` → publish. Dry-run against **TestPyPI** first to
  preview how the project page renders.
- [ ] **Single-source the version + tag flow.** The version currently lives in
  both `pyproject.toml` and the `__init__.py` fallback. Either document
  "bump both" as the release step, or adopt `setuptools-scm` so git tags
  drive the version. Decide the first release number — `0.1.0` signals more
  confidence than `0.0.2`.
- [ ] **Add `CHANGELOG.md`** (Keep-a-Changelog style) and create a GitHub
  Release per tag. One entry for the first release is enough.
- [ ] **Modernize license metadata:** `license = "MIT"` (SPDX string) instead
  of `{text = "MIT"}`, add `license-files = ["LICENSE"]`, drop the deprecated
  `License :: OSI Approved :: MIT License` classifier, and bump the build
  requirement to `setuptools>=77`.
- [ ] **Add `keywords`** to `[project]` (e.g. `modflow`, `voronoi`, `mesh`,
  `groundwater`, `gmsh`) — PyPI search matches on these.

## Phase 3 — Nice-to-have (can follow in later 0.x releases)

- [ ] **Python 3.13/3.14 in the CI matrix and classifiers.** The local dev
  env is already on 3.14, so it is implicitly supported but untested in CI.
- [ ] **Docs site** (mkdocs-material + API reference on GitHub Pages or
  ReadTheDocs) and a `Documentation` URL in `[project.urls]`. The README is
  sufficient for an alpha.
- [ ] **Citation/DOI** (`CITATION.cff` + Zenodo) — worthwhile for a
  research-audience package.
- [ ] **conda-forge feedstock.** The MODFLOW/flopy user base is heavily
  conda-based. Do this after the PyPI release stabilizes, since conda-forge
  builds from the published sdist.

## Release sequence

1. Merge `gmshflow_missing` into `main` and verify the full test suite.
2. One metadata/README commit: README links + install section, dependency
   floors, SPDX license, keywords (Phase 1 items 4–6, Phase 2 items 4–5).
3. Build verification + fresh-venv smoke test (Phase 1 items 2–3).
4. Add the TestPyPI publish workflow and `CHANGELOG.md`; use candidate version
   `0.1.0rc1`.
5. Tag `v0.1.0rc1` → TestPyPI rehearsal → verify the rendered page and clean
   installation → stop before real PyPI.

## Verification

- `twine check dist/*` passes; sdist/wheel contain only the package and
  standard metadata.
- README Basic Usage example runs against the installed wheel in a clean venv.
- TestPyPI project page renders with no broken links before the real upload.
- After the rehearsal: install `vorflow==0.1.0rc1` from TestPyPI in a fresh
  environment and confirm `vorflow.__version__` matches the tag.
- Real PyPI publication remains pending a separate review and approval.

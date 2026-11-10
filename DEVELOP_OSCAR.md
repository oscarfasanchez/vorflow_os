# Oscar's Vorflow development

`main` mirrors rhugman/vorflow main. `develop` mirrors rhugman/vorflow develop.
`develop_oscar` is upstream `develop` with the open pull requests merged in, so they can be used and tested
together before upstream merges them. It is personal development, not a release.

## What is merged (2026-11-10)

| PR | Branch | Content |
|---|---|---|
| #86-#93 | `feat/disu-stacked` ... `feat/disu-quality` | DISU with full connectivity, tracking and views, MODFLOW-USG output, grid per layer and splice, refill, split contacts and sublayers, connection quality |
| #87 | `fix/hanging-vertex-faces` | faces lost to roundoff at hanging vertices (MODFLOW-USG export) |
| #94 | `feat/disu-barriers` | HFB barriers on DISU grids |
| #95 | `feat/disu-gnc` | ghost-node corrections (GNC) for MODFLOW-USG and MODFLOW 6 |
| #64 (Erik) | `eriktoller:feat/on_cell-on_face` | `cells_on`, `faces_on`, `ConceptualMesh.features()` |

Each PR branch was merged as it is, with merge commits (no rebase); the only conflict was the changelog, where both
sides were kept. The first prototype of this branch (30 commits) is kept as `backup/develop_oscar-802b103`, and the
uncommitted work found in its worktree as `backup/develop_oscar-wip`.

## Updating

When a PR changes or upstream merges it, merge upstream `develop` and the updated PR branches into this branch again
(merge commits only). Never rebase or force-push it.

## Known open point

#86 adds `normal_x` and `normal_y` to the `connectivity` table returned by `vorflow.utils.grid_to_mfusg_disu`, an
existing public function; the fix (drop them there) is in `backup/develop_oscar-wip` and not yet in #86.

## Validation

Run the normal pytest and Ruff checks against this checkout. Solver comparisons stay in the main checkout's ignored
`benchmarks/` folder (the GNC study: `benchmarks/gnc-truth-2026-11-10`). Local AGENTS.md and CLAUDE.md define
discussion and publication gates; they are deliberately not tracked.

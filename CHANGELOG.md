# Changelog

## [Unreleased]

## [0.2.3] - 2026-10-09

- Added `ParetoFrontier::try_push`, which returns
  `FrontierError::InconsistentDimensions` or `FrontierError::NonFinite`
  instead of panicking; `push` keeps its documented panic.
- `pareto_indices` and `pareto_indices_2d` now return the same set: a point
  is kept iff no other point dominates it under the shared 1e-9 tolerance,
  and exact duplicates are all kept. Previously `pareto_indices_2d` compared
  exactly and kept only the first of a set of duplicates.
- `pareto_indices_2d` returns indices in ascending order; it previously
  returned them in descending-x sweep order.
- `pareto_indices` no longer skips points it has already removed when
  checking dominance, so its result under the tolerance no longer depends on
  input order.
- Sorting uses `total_cmp`, including the eigenvalue sort in the sensitivity
  module.

## [0.2.2] - 2026-06-11

- Added a worked bandit example to the sensitivity module docs.

## [0.2.1] - 2026-04-06

- Added workspace metadata so the crate builds cleanly as a standalone checkout.

## [0.2.0] - 2026-03-14

- Added `EpsilonArchive` for grid-based epsilon-dominance archiving.
- Added `ideal_point`, `nadir_point`, `normalized_values`, and
  `suggest_ref_point`.
- Added ASF scoring, `best_asf`, `knee_index`, `ranked_indices`,
  `hypervolume_contributions`, `retain`, and `pareto_layers`.
- Added `from_points` and `Extend` for batch construction.
- Added generational distance and inverted generational distance indicators.
- Fixed crowding distance for shared boundary values.
- Specialized 3D hypervolume with a z-sweep implementation.

## [0.1.1]

- Add `sensitivity` module (objective redundancy analysis).
- Improve numerical accuracy for finite-difference Jacobians (central differences) and harden validation/tests.

## [0.1.0]

- Initial release.

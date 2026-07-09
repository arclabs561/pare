# Changelog

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

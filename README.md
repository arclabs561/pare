# pare

[![crates.io](https://img.shields.io/crates/v/pare.svg)](https://crates.io/crates/pare)
[![Documentation](https://docs.rs/pare/badge.svg)](https://docs.rs/pare)
[![CI](https://github.com/arclabs561/pare/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/pare/actions/workflows/ci.yml)

Pareto frontier and skyline query primitives.

```toml
[dependencies]
pare = "0.2.0"
```

## Quick start

```rust
use pare::ParetoFrontier;

// [Relevance, Recency] -- higher is better
let candidates = vec![
    vec![0.9, 0.1], // A
    vec![0.5, 0.5], // B
    vec![0.1, 0.9], // C
    vec![0.4, 0.4], // D (dominated by B)
];

let frontier = ParetoFrontier::try_new(&candidates).unwrap();
assert_eq!(frontier.indices(), vec![0, 1, 2]); // D excluded
```

## Mixed objectives

When some objectives should be minimized, construct with explicit directions:

```rust
use pare::{Direction, ParetoFrontier};

// accuracy (maximize) vs latency_ms (minimize)
let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
f.push(vec![0.92, 120.0], "model_a");
f.push(vec![0.90,  80.0], "model_b");
f.push(vec![0.88, 150.0], "model_c"); // dominated by model_b

assert_eq!(f.len(), 2); // model_c filtered out
```

## Selecting from the frontier

Pick a point with weighted scoring, ASF (Achievement Scalarizing Function), knee detection, or analyze spread with crowding distance:

```rust
use pare::{Direction, ParetoFrontier};

let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
f.push(vec![0.9, 0.1], "A");
f.push(vec![0.5, 0.5], "B");
f.push(vec![0.1, 0.9], "C");

// ASF -- works on non-convex fronts, finds the best compromise point
let ideal = f.ideal_point().unwrap();
let best_asf = f.best_asf(&[1.0, 1.0], &ideal).unwrap();

// Knee point -- highest tradeoff point on the frontier
let knee = f.knee_index().unwrap();

// Weighted linear score -- simple but misses non-convex regions
let best = f.best_index(&[0.7, 0.3]).unwrap();

// Full ranking by score (descending)
let ranked = f.ranked_indices(&[0.5, 0.5]);

// Crowding distance -- points in sparse regions score higher
let distances = f.crowding_distances();

// Hypervolume -- area dominated by the frontier (quality indicator)
let ref_pt = f.suggest_ref_point(0.1).unwrap();
let hv = f.hypervolume(&ref_pt);

// Per-point hypervolume contribution (for indicator-based selection)
let contribs = f.hypervolume_contributions(&ref_pt);
```

## Normalization and scaling

Objectives often have different units (accuracy 0-1, latency 10-500ms, cost $0-$10000). pare handles this:

```rust
use pare::{Direction, ParetoFrontier};

let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
f.push(vec![0.92, 120.0], "model_a");
f.push(vec![0.88,  80.0], "model_b");

// Unit-free normalized values: 0 = worst on front, 1 = best on front
let norm = f.normalized_values(0).unwrap();
// norm[0] = 1.0 (best accuracy), norm[1] = 0.0 (worst latency)

// Ideal (best per-objective) and nadir (worst per-objective)
let ideal = f.ideal_point().unwrap(); // [0.92, 80.0]
let nadir = f.nadir_point().unwrap(); // [0.88, 120.0]

// For stable scoring across frontier changes, use static bounds:
let bounds = vec![(0.0, 1.0), (0.0, 500.0)];
let score = f.scalar_score_static(0, &[1.0, 1.0], &bounds);
```

## Post-hoc filtering

Apply constraints after frontier construction:

```rust
use pare::{Direction, ParetoFrontier};

let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
f.push(vec![0.95, 200.0], "expensive");
f.push(vec![0.90, 50.0],  "moderate");
f.push(vec![0.80, 10.0],  "cheap");

// Keep only points within budget
f.retain(|p| p.values[1] < 100.0);
assert_eq!(f.len(), 2);
```

## Convenience functions

For simple cases where you just need non-dominated indices from `Vec<f32>` points:

```rust
use pare::{pareto_indices, pareto_indices_2d, pareto_indices_k_dominance, pareto_layers};

let points = vec![vec![0.9f32, 0.1], vec![0.5, 0.5], vec![0.4, 0.4]];

let idx = pareto_indices(&points).unwrap();       // general N-d
let idx2 = pareto_indices_2d(&points).unwrap();    // optimized 2-d path

// k-dominance: a point is dominated only if worse in >= k objectives
let relaxed = pareto_indices_k_dominance(&points, 2).unwrap();

// Non-dominated sorting: partition into successive Pareto layers
let layers = pareto_layers(&points).unwrap();
// layers[0] = Pareto front, layers[1] = front of remainder, ...
```

## Epsilon-dominance archiving

For streaming / online optimization where memory must be bounded:

```rust
use pare::{Direction, EpsilonArchive};

let mut archive = EpsilonArchive::new_uniform(
    vec![Direction::Maximize, Direction::Maximize],
    0.1, // grid cell width per dimension
);

// Insert many points; archive stays bounded
for i in 0..1000 {
    let x = (i as f64) / 1000.0;
    archive.push(vec![x, 1.0 - x], i);
}
assert!(archive.len() <= 100); // at most ceil(1/0.1)^2 cells

// Convert to ParetoFrontier for analysis
let frontier = archive.into_frontier();
```

## Quality indicators

Compare fronts with Generational Distance (GD) and Inverted Generational Distance (IGD):

```rust
use pare::{generational_distance, inverted_generational_distance};

let front = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let reference = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];

let gd = generational_distance(&front, &reference).unwrap();  // convergence
let igd = inverted_generational_distance(&front, &reference).unwrap(); // convergence + spread
```

## Sensitivity analysis

The `sensitivity` module computes finite-difference Jacobians and objective redundancy analysis:

```rust
use pare::sensitivity::{finite_difference_jacobian, analyze_redundancy};

let objectives: Vec<fn(&[f64]) -> f64> = vec![
    |x| x[0] * x[0],
    |x| (x[0] - 1.0).powi(2),
];

let jac = finite_difference_jacobian(&[0.5], &objectives, 1e-6);
let analysis = analyze_redundancy(&jac).unwrap();
// analysis.redundant_pairs -- objectives that move together
```

## Examples

[**sensitivity_analysis.rs**](examples/sensitivity_analysis.rs) -- Multi-objective sensitivity analysis for a simulated 3-arm experiment with 9 covariate cells and 6 objectives. Builds a finite-difference Jacobian across the full parameter space, computes the eigenvalue spectrum to find how many objectives actually matter (effective dimensionality), identifies redundant pairs via cosine similarity, and reports which objectives can be dropped without losing decision power. Useful when you suspect your objective set is over-specified.

```bash
cargo run --example sensitivity_analysis
```

For background on dominance, crowding distance, and hypervolume, see
[`TECHNICAL_BACKGROUND.md`](TECHNICAL_BACKGROUND.md).

## License

MIT OR Apache-2.0

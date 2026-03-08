# pare

[![crates.io](https://img.shields.io/crates/v/pare.svg)](https://crates.io/crates/pare)
[![Documentation](https://docs.rs/pare/badge.svg)](https://docs.rs/pare)
[![CI](https://github.com/arclabs561/pare/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/pare/actions/workflows/ci.yml)

Pareto frontier and skyline query primitives for multi-objective optimization.
Filters sets of items to find non-dominated candidates across multiple metrics.

```toml
[dependencies]
pare = "0.1.2"
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

Once you have a frontier, pick a single point with weighted scoring or analyze spread with crowding distance:

```rust
use pare::{Direction, ParetoFrontier};

let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
f.push(vec![0.9, 0.1], "A");
f.push(vec![0.5, 0.5], "B");
f.push(vec![0.1, 0.9], "C");

// Weighted linear score -- pick the point best matching your preferences
let best = f.best_index(&[0.7, 0.3]).unwrap(); // favor first objective

// Crowding distance -- points in sparse regions score higher
let distances = f.crowding_distances();

// Hypervolume -- area dominated by the frontier (quality indicator)
let hv = f.hypervolume(&[0.0, 0.0]);
```

## Convenience functions

For simple cases where you just need non-dominated indices from `Vec<f32>` points:

```rust
use pare::{pareto_indices, pareto_indices_2d, pareto_indices_k_dominance};

let points = vec![vec![0.9f32, 0.1], vec![0.5, 0.5], vec![0.4, 0.4]];

let idx = pareto_indices(&points).unwrap();       // general N-d
let idx2 = pareto_indices_2d(&points).unwrap();    // optimized 2-d path

// k-dominance: a point is dominated only if worse in >= k objectives
let relaxed = pareto_indices_k_dominance(&points, 2).unwrap();
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

# pare

Pareto frontier and skyline query primitives for multi-objective optimization.
Efficiently filters sets of items to find non-dominated candidates across multiple metrics.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/pare) | [docs.rs](https://docs.rs/pare)

For the underlying concepts (dominance/skyline, crowding distance, hypervolume), see
[`TECHNICAL_BACKGROUND.md`](TECHNICAL_BACKGROUND.md).

## Quickstart

```toml
[dependencies]
pare = "0.1.1"
```

```rust
use pare::ParetoFrontier;

// [Relevance, Recency] - higher is better
let candidates = vec![
    vec![0.9, 0.1], // A
    vec![0.5, 0.5], // B
    vec![0.1, 0.9], // C
    vec![0.4, 0.4], // D (dominated by B)
];

let frontier = ParetoFrontier::try_new(&candidates).unwrap().indices();
// indices: [0, 1, 2]
```

`try_new()` is a convenience constructor that assumes **all objectives are maximized** and
stores each point's original index as its associated `data`.

For mixed maximize/minimize objectives, build a `ParetoFrontier` with explicit directions and
insert points with `push()`:

```rust
use pare::{Direction, ParetoFrontier};

// [accuracy (higher is better), latency_ms (lower is better)]
let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
f.push(vec![0.92, 120.0], "model_a");
f.push(vec![0.90,  80.0], "model_b");
```
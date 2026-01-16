# pare

Pareto frontier and skyline query primitives for multi-objective optimization.
Efficiently filters sets of items to find non-dominated candidates across multiple metrics.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/pare) | [docs.rs](https://docs.rs/pare)

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


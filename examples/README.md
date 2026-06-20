# pare examples

## Where to start

| I want to... | Run |
|---|---|
| Check whether objectives are redundant | `sensitivity_analysis` |

```sh
cargo run --example sensitivity_analysis
```

## What to inspect

`sensitivity_analysis` builds a small routing/bandit scenario with six objectives over a 3-by-3 covariate grid. The output reports:

- the eigenvalue spectrum of the objective sensitivity Gram matrix,
- effective Pareto dimension at several thresholds,
- pairwise cosine similarity between objectives,
- nearly redundant objective pairs.

Use it when deciding whether a multi-objective problem really needs all listed objectives, or whether some objectives move together and can be removed before optimizing.

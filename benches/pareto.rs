use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pare::{pareto_indices, pareto_indices_2d, pareto_indices_k_dominance};
use rand::prelude::*;

fn bench_pareto(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto");

    let n = 1000;
    let d = 5;
    let mut rng = StdRng::seed_from_u64(42);

    let points: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f32>()).collect())
        .collect();

    group.bench_function("indices_n1000_d5", |b| {
        b.iter(|| {
            pareto_indices(black_box(&points)).unwrap();
        })
    });

    group.bench_function("kdominance_k3_n1000_d5", |b| {
        b.iter(|| {
            pareto_indices_k_dominance(black_box(&points), 3).unwrap();
        })
    });

    // 2D specialized algorithm vs naive.
    let points_2d: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..2).map(|_| rng.random::<f32>()).collect())
        .collect();

    group.bench_function("indices_naive_n1000_d2", |b| {
        b.iter(|| {
            pareto_indices(black_box(&points_2d)).unwrap();
        })
    });

    group.bench_function("indices_2d_sweep_n1000", |b| {
        b.iter(|| {
            pareto_indices_2d(black_box(&points_2d)).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pareto);
criterion_main!(benches);

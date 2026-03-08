use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pare::{
    pareto_indices, pareto_indices_2d, pareto_indices_k_dominance, Direction, ParetoFrontier,
};
use rand::prelude::*;

fn bench_pareto(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto");

    let n = 1000;
    let d = 5;
    let mut rng = StdRng::seed_from_u64(42);

    let points: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f32>()).collect())
        .collect();

    // Bench ParetoFrontier::push (the online API that muxer actually uses).
    // Uses f64 points and mixed directions to be realistic.
    {
        let f64_points: Vec<Vec<f64>> = (0..100)
            .map(|_| (0..d).map(|_| rng.random::<f64>()).collect())
            .collect();
        let directions = vec![
            Direction::Maximize,
            Direction::Minimize,
            Direction::Maximize,
            Direction::Minimize,
            Direction::Maximize,
        ];

        group.bench_function("push_n100_d5_mixed", |b| {
            b.iter(|| {
                let mut frontier = ParetoFrontier::new(directions.clone());
                for (i, p) in f64_points.iter().enumerate() {
                    frontier.push(black_box(p.clone()), i);
                }
                black_box(frontier.len());
            })
        });
    }

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

    // Hypervolume benchmarks.
    let mut hv_group = c.benchmark_group("hypervolume");

    {
        let mut frontier_2d = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
        let mut rng2 = StdRng::seed_from_u64(99);
        for i in 0..100 {
            let p = vec![rng2.random::<f64>(), rng2.random::<f64>()];
            frontier_2d.push(p, i);
        }
        let ref_2d = [0.0, 0.0];
        hv_group.bench_function("2d_n100", |b| {
            b.iter(|| black_box(frontier_2d.hypervolume(black_box(&ref_2d))))
        });
    }

    {
        let mut frontier_3d = ParetoFrontier::new(vec![
            Direction::Maximize,
            Direction::Maximize,
            Direction::Maximize,
        ]);
        let mut rng3 = StdRng::seed_from_u64(101);
        for i in 0..100 {
            let p = vec![
                rng3.random::<f64>(),
                rng3.random::<f64>(),
                rng3.random::<f64>(),
            ];
            frontier_3d.push(p, i);
        }
        let ref_3d = [0.0, 0.0, 0.0];
        hv_group.bench_function("3d_n100", |b| {
            b.iter(|| black_box(frontier_3d.hypervolume(black_box(&ref_3d))))
        });
    }

    hv_group.finish();

    // Crowding distance benchmark.
    let mut cd_group = c.benchmark_group("crowding_distance");

    {
        let mut frontier_cd = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
        let mut rng_cd = StdRng::seed_from_u64(77);
        for i in 0..200 {
            let p = vec![rng_cd.random::<f64>(), rng_cd.random::<f64>()];
            frontier_cd.push(p, i);
        }
        cd_group.bench_function("2d_n200", |b| {
            b.iter(|| black_box(frontier_cd.crowding_distances()))
        });
    }

    cd_group.finish();
}

criterion_group!(benches, bench_pareto);
criterion_main!(benches);

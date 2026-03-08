use pare::{Direction, ParetoFrontier};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_pareto_invariant(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 2), 1..50)) {
        let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
        for (i, v) in values.into_iter().enumerate() {
            frontier.push(v, i);
        }

        let points = frontier.points();
        for i in 0..points.len() {
            for j in 0..points.len() {
                if i == j { continue; }
                // No point on the frontier should dominate another.
                assert!(!frontier.point_dominates(&points[i].values, &points[j].values),
                    "Point {:?} dominates {:?}", points[i].values, points[j].values);
            }
        }
    }

    #[test]
    fn test_crowding_distance_ordering(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 2), 3..50)) {
        let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
        for (i, v) in values.into_iter().enumerate() {
            frontier.push(v, i);
        }

        let dists = frontier.crowding_distances();
        assert_eq!(dists.len(), frontier.len());

        // At least two points should have infinite distance (the boundaries)
        // if we have at least 2 points.
        if frontier.len() >= 2 {
            let infinite_count = dists.iter().filter(|&&d| d.is_infinite()).count();
            assert!(infinite_count >= 2, "Should have at least 2 boundary points with infinite distance");
        }
    }

    #[test]
    fn test_hypervolume_monotone_2d(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 2), 1..80)) {
        // With maximize objectives and a reference point at the origin, hypervolume is:
        // - always >= 0
        // - monotone non-decreasing as we add points (dominated points are ignored by the frontier)
        let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
        let ref_point = [0.0, 0.0];
        let mut prev = 0.0;

        for (i, v) in values.into_iter().enumerate() {
            frontier.push(v, i);
            let hv = frontier.hypervolume(&ref_point);
            assert!(hv >= -1e-9, "hypervolume must be non-negative");
            assert!(hv + 1e-9 >= prev, "hypervolume must be monotone (prev={prev}, hv={hv})");
            // Upper bound: max rectangle is 1x1 in the unit square.
            assert!(hv <= 1.0 + 1e-9, "hypervolume must be <= 1 in [0,1]^2 (hv={hv})");
            prev = hv;
        }
    }
}

// Cross-check: pareto_indices_2d (O(n log n) sweep) vs pareto_indices (O(n^2) general).
proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

    #[test]
    fn test_pareto_indices_2d_matches_naive(
        values in prop::collection::vec(prop::collection::vec(0.0f32..1.0, 2..=2), 1..100)
    ) {
        let idx_naive = pare::pareto_indices(&values).unwrap();
        let idx_2d = pare::pareto_indices_2d(&values).unwrap();

        let set_naive: std::collections::BTreeSet<usize> = idx_naive.into_iter().collect();
        let set_2d: std::collections::BTreeSet<usize> = idx_2d.into_iter().collect();

        prop_assert_eq!(set_naive, set_2d,
            "pareto_indices and pareto_indices_2d disagree");
    }
}

// 3D hypervolume monotonicity (exercises recursive slicing path).
proptest! {
    #[test]
    fn test_hypervolume_monotone_3d(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 3), 1..40)) {
        let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize, Direction::Maximize]);
        let ref_point = [0.0, 0.0, 0.0];
        let mut prev = 0.0;

        for (i, v) in values.into_iter().enumerate() {
            frontier.push(v, i);
            let hv = frontier.hypervolume(&ref_point);
            assert!(hv >= -1e-9, "hypervolume must be non-negative");
            assert!(hv + 1e-9 >= prev, "hypervolume must be monotone (prev={prev}, hv={hv})");
            assert!(hv <= 1.0 + 1e-9, "hypervolume must be <= 1 in [0,1]^3 (hv={hv})");
            prev = hv;
        }
    }
}

// 3D hypervolume grid cross-check: compare exact recursive slicing against
// brute-force grid sampling. Small point sets + coarse grid keep this fast.
proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    #[test]
    fn test_hypervolume_3d_grid_crosscheck(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 3), 1..10)) {
        let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize, Direction::Maximize]);
        for (i, v) in values.into_iter().enumerate() {
            frontier.push(v, i);
        }
        let ref_point = [0.0, 0.0, 0.0];
        let exact_hv = frontier.hypervolume(&ref_point);

        // Brute-force grid approximation: sample a 20^3 grid in [0,1]^3.
        let res = 20usize;
        let step = 1.0 / res as f64;
        let cell_vol = step * step * step;
        let mut dominated_count = 0usize;

        let points = frontier.points();
        for ix in 0..res {
            let x = (ix as f64 + 0.5) * step;
            for iy in 0..res {
                let y = (iy as f64 + 0.5) * step;
                for iz in 0..res {
                    let z = (iz as f64 + 0.5) * step;
                    // A grid cell center is "dominated" if some frontier point
                    // is >= in all coordinates (maximize).
                    let dom = points.iter().any(|p| {
                        p.values[0] >= x && p.values[1] >= y && p.values[2] >= z
                    });
                    if dom {
                        dominated_count += 1;
                    }
                }
            }
        }

        let approx_hv = dominated_count as f64 * cell_vol;
        // With a 20^3 grid the max discretization error is bounded by
        // surface_area * step. For small frontier sets in [0,1]^3 this stays
        // well within 0.1.
        let tol = 0.1;
        assert!(
            (exact_hv - approx_hv).abs() < tol,
            "exact {exact_hv} vs grid approx {approx_hv} differ by more than {tol}"
        );
    }
}

// Brute-force Pareto membership cross-check: keep cases modest (O(n^2) per case).
proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn test_frontier_matches_bruteforce_3d(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 3), 1..25)) {
        // Brute force: i is on the frontier iff no j dominates i.
        let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize, Direction::Maximize]);
        for (i, v) in values.iter().cloned().enumerate() {
            f.push(v, i);
        }
        let kept: std::collections::BTreeSet<usize> = f.points().iter().map(|p| p.data).collect();

        let dominates = |a: &[f64], b: &[f64]| -> bool {
            let mut strictly = false;
            for (&av, &bv) in a.iter().zip(b.iter()) {
                if av + 1e-12 < bv {
                    return false;
                }
                if av > bv + 1e-12 {
                    strictly = true;
                }
            }
            strictly
        };

        for i in 0..values.len() {
            let mut dom = false;
            for j in 0..values.len() {
                if i == j { continue; }
                if dominates(&values[j], &values[i]) {
                    dom = true;
                    break;
                }
            }
            let brute_kept = !dom;
            assert_eq!(kept.contains(&i), brute_kept);
        }
    }
}

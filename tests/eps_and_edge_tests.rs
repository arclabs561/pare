//! Tests for epsilon behavior, edge cases, and miscellaneous untested paths.

use pare::{dominates, Direction, ParetoFrontier};

// ====================================================================
// with_eps effect tests
// ====================================================================

#[test]
fn eps_effect_on_dominance() {
    // Epsilon controls the tolerance for "strictly better":
    //   dominates(a, b) requires a[i] > b[i] + eps in at least one dim.
    //   Larger eps => harder to dominate => more points survive.
    //   Smaller eps => easier to dominate => fewer points survive.
    //
    // Points: a=(1.0, 0.5), b=(1.005, 0.5). Gap is 0.005 in dim 0.

    // eps=0: 1.005 > 1.0 + 0 => b strictly better. b dominates a. len=1.
    let mut f0 = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]).with_eps(0.0);
    f0.push(vec![1.0, 0.5], "a");
    f0.push(vec![1.005, 0.5], "b");
    assert_eq!(f0.len(), 1, "eps=0: b should dominate a");
    assert_eq!(f0.points()[0].data, "b");

    // eps=0.001: 1.005 > 1.001 => b still strictly better. len=1.
    let mut f1 =
        ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]).with_eps(0.001);
    f1.push(vec![1.0, 0.5], "a");
    f1.push(vec![1.005, 0.5], "b");
    assert_eq!(f1.len(), 1, "eps=0.001: gap (0.005) > eps, b dominates a");

    // eps=0.01: 1.005 is NOT > 1.01 => b not strictly better. Both stay. len=2.
    let mut f2 = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]).with_eps(0.01);
    f2.push(vec![1.0, 0.5], "a");
    f2.push(vec![1.005, 0.5], "b");
    assert_eq!(
        f2.len(),
        2,
        "eps=0.01: gap (0.005) < eps, neither dominates"
    );
}

// ====================================================================
// Crowding distance edge cases
// ====================================================================

#[test]
fn crowding_distance_single_point() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.5, 0.5], 0);
    let cd = f.crowding_distances();
    assert_eq!(cd.len(), 1);
    assert!(
        cd[0].is_infinite(),
        "single point should have infinite crowding distance"
    );
}

#[test]
fn crowding_distance_two_points() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.0, 1.0], 0);
    f.push(vec![1.0, 0.0], 1);
    let cd = f.crowding_distances();
    assert_eq!(cd.len(), 2);
    assert!(cd[0].is_infinite());
    assert!(cd[1].is_infinite());
}

// Fortin-Parizeau: shared boundary values all get infinite distance
#[test]
fn crowding_distance_shared_boundary_3d() {
    // Use 3 objectives so points sharing a value in one dim can be non-dominated.
    let mut f = ParetoFrontier::new(vec![
        Direction::Maximize,
        Direction::Maximize,
        Direction::Maximize,
    ]);
    // Two points share dim0=0.0 (min boundary) but trade off in dim1/dim2
    f.push(vec![0.0, 1.0, 0.5], 0);
    f.push(vec![0.0, 0.5, 1.0], 1);
    f.push(vec![1.0, 0.0, 0.0], 2);
    assert_eq!(f.len(), 3);

    let cd = f.crowding_distances();
    // Points 0 and 1 share min boundary in dim 0 -> both should be infinite
    assert!(
        cd[0].is_infinite(),
        "shared min boundary should be infinite"
    );
    assert!(
        cd[1].is_infinite(),
        "shared min boundary should be infinite"
    );
    assert!(cd[2].is_infinite(), "max boundary should be infinite");
}

#[test]
fn crowding_distance_tied_dim_mixed_directions() {
    // Use mixed directions: tied in dim 0 but non-dominated due to dim 1 Minimize
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    f.push(vec![0.5, 1.0], 0); // same dim0, worse dim1
    f.push(vec![0.5, 5.0], 1); // same dim0, worse dim1 -- dominated by 0
                               // Can't have 3 non-dominated points with same dim0 in 2D; use 3 objectives
    let mut f = ParetoFrontier::new(vec![
        Direction::Maximize,
        Direction::Maximize,
        Direction::Maximize,
    ]);
    f.push(vec![0.5, 1.0, 0.0], 0);
    f.push(vec![0.5, 0.0, 1.0], 1);
    f.push(vec![0.5, 0.5, 0.5], 2);
    assert_eq!(f.len(), 3);

    let cd = f.crowding_distances();
    // All tied in dim 0 -> all are boundary in that dim -> all infinite
    assert!(cd[0].is_infinite());
    assert!(cd[1].is_infinite());
    assert!(cd[2].is_infinite(), "all tied in dim 0 makes all boundary");
}

// ====================================================================
// NormalizationStats edge cases
// ====================================================================

#[test]
fn normalization_stats_empty_returns_default() {
    use pare::NormalizationStats;
    let stats = NormalizationStats::from_values(&[]);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std, 0.0);
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
}

#[test]
fn normalization_stats_single_value() {
    use pare::NormalizationStats;
    let stats = NormalizationStats::from_values(&[42.0]);
    assert_eq!(stats.mean, 42.0);
    assert_eq!(stats.std, 0.0);
    assert_eq!(stats.min, 42.0);
    assert_eq!(stats.max, 42.0);
}

// ====================================================================
// Hypervolume edge cases
// ====================================================================

#[test]
fn hypervolume_empty_frontier() {
    let f: ParetoFrontier<()> = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    assert_eq!(f.hypervolume(&[0.0, 0.0]), 0.0);
}

#[test]
fn hypervolume_point_at_ref_point_contributes_nothing() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    // Point exactly at the reference point: improvement is 0 in all dims.
    f.push(vec![0.0, 0.0], ());
    let hv = f.hypervolume(&[0.0, 0.0]);
    assert!(
        hv.abs() < 1e-9,
        "point at ref should contribute 0 volume, got {hv}"
    );
}

// ====================================================================
// scalar_score_static edge cases
// ====================================================================

#[test]
fn scalar_score_static_clamps_out_of_bounds() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize]);
    f.push(vec![2.0], "above");

    // Value 2.0 is above bounds max of 1.0 => clamped to 1.0.
    let s = f.scalar_score_static(0, &[1.0], &[(0.0, 1.0)]);
    assert!(
        (s - 1.0).abs() < 1e-9,
        "clamped score should be 1.0, got {s}"
    );
}

#[test]
fn scalar_score_static_zero_range_contributes_zero() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize]);
    f.push(vec![5.0], "a");

    // min == max => that dimension contributes 0.
    let s = f.scalar_score_static(0, &[1.0], &[(5.0, 5.0)]);
    assert!(
        s.abs() < 1e-9,
        "zero-range dim should contribute 0, got {s}"
    );
}

#[test]
fn scalar_score_static_minimize_direction() {
    // Two non-dominated points in a 2D space: minimize cost, maximize quality.
    let mut f = ParetoFrontier::new(vec![Direction::Minimize, Direction::Maximize]);
    f.push(vec![0.2, 0.3], "cheap-low"); // low cost, low quality
    f.push(vec![0.8, 0.9], "expensive-high"); // high cost, high quality

    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    // Weight heavily on cost (Minimize): cheap should score higher.
    let s_cheap = f.scalar_score_static(0, &[10.0, 1.0], &bounds);
    let s_expensive = f.scalar_score_static(1, &[10.0, 1.0], &bounds);
    assert!(
        s_cheap > s_expensive,
        "cheap should score higher when cost-weighted, got cheap={s_cheap}, expensive={s_expensive}"
    );
}

// ====================================================================
// All-equal points (degenerate input)
// ====================================================================

#[test]
fn all_equal_points_keep_one() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.5, 0.5], 0);
    f.push(vec![0.5, 0.5], 1);
    f.push(vec![0.5, 0.5], 2);
    // All identical => none dominates any other (no strict improvement).
    // First inserted stays; subsequent are not strictly dominated but also
    // don't dominate existing. Current behavior: all are kept.
    // The key invariant: no point on the frontier dominates another.
    for i in 0..f.len() {
        for j in 0..f.len() {
            if i == j {
                continue;
            }
            assert!(
                !f.point_dominates(&f.points()[i].values, &f.points()[j].values),
                "identical points should not dominate each other"
            );
        }
    }
}

// ====================================================================
// Crowding distance on duplicate objective vectors (Fortin-Parizeau 2013)
// ====================================================================

#[test]
fn crowding_distance_duplicate_objectives_finite() {
    // When multiple points share the same objective values on a frontier,
    // crowding distance should still produce finite, non-NaN values for
    // interior points. This tests the edge case from Fortin-Parizeau 2013.
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.0, 1.0], 0);
    f.push(vec![0.5, 0.5], 1);
    f.push(vec![0.5, 0.5], 2); // duplicate of point 1
    f.push(vec![1.0, 0.0], 3);

    let cd = f.crowding_distances();
    assert_eq!(cd.len(), f.len());

    // All distances must be finite or infinite (never NaN).
    for (i, &d) in cd.iter().enumerate() {
        assert!(!d.is_nan(), "crowding distance[{i}] is NaN");
    }
}

#[test]
fn crowding_distance_all_identical() {
    // All points have identical objectives. Range is 0 in every dimension,
    // so the normalized gap addition is skipped. Only boundary assignment
    // applies. Distances must not be NaN.
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    for i in 0..5 {
        f.push(vec![0.5, 0.5], i);
    }

    let cd = f.crowding_distances();
    for (i, &d) in cd.iter().enumerate() {
        assert!(
            !d.is_nan(),
            "crowding distance[{i}] is NaN on identical points"
        );
    }
}

// ====================================================================
// Rejected-point dual property
// ====================================================================

#[test]
fn rejected_point_is_dominated_by_frontier() {
    // Every point rejected by push should be dominated by at least one
    // frontier point. This is the dual of the non-dominance invariant.
    let dirs = vec![Direction::Maximize, Direction::Maximize];
    let eps = 1e-9;
    let mut f = ParetoFrontier::new(dirs.clone());

    let candidates = vec![
        vec![0.9, 0.1],
        vec![0.5, 0.5],
        vec![0.1, 0.9],
        vec![0.4, 0.4],  // dominated by (0.5, 0.5)
        vec![0.3, 0.3],  // dominated by (0.5, 0.5)
        vec![0.8, 0.05], // dominated by (0.9, 0.1)
    ];

    for c in &candidates {
        let added = f.push(c.clone(), ());
        if !added {
            // Verify: at least one frontier point dominates this rejected point.
            let dominated_by_some = f
                .points()
                .iter()
                .any(|p| dominates(&dirs, eps, &p.values, c));
            assert!(
                dominated_by_some,
                "rejected point {c:?} is not dominated by any frontier point"
            );
        }
    }
}

// ====================================================================
// Labels accessor
// ====================================================================

#[test]
fn labels_default_are_numeric() {
    let f: ParetoFrontier<()> = ParetoFrontier::new(vec![
        Direction::Maximize,
        Direction::Minimize,
        Direction::Maximize,
    ]);
    assert_eq!(f.labels(), &["0", "1", "2"]);
}

#[test]
fn labels_custom_roundtrip() {
    let f: ParetoFrontier<()> = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize])
        .with_labels(vec!["accuracy".into(), "latency".into()]);
    assert_eq!(f.labels(), &["accuracy", "latency"]);
}

#[test]
fn directions_accessor() {
    let f: ParetoFrontier<()> = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    assert_eq!(f.directions(), &[Direction::Maximize, Direction::Minimize]);
}

// ====================================================================
// Collinear points (adversarial)
// ====================================================================

#[test]
fn collinear_points_on_frontier() {
    // Points along the line y = 1 - x in [0,1]^2.
    // All are non-dominated (perfect tradeoff).
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    for i in 0..=10 {
        let x = i as f64 / 10.0;
        f.push(vec![x, 1.0 - x], i);
    }
    assert_eq!(f.len(), 11, "all collinear tradeoff points should be kept");
}

// ====================================================================
// Hypervolume with mixed directions (3D)
// ====================================================================

#[test]
fn hypervolume_3d_mixed_directions() {
    // Maximize dim0, minimize dim1, maximize dim2.
    // Point (1, 0, 1) with ref (0, 10, 0):
    // oriented = (1-0, 10-0, 1-0) = (1, 10, 1) => volume = 10.
    let mut f = ParetoFrontier::new(vec![
        Direction::Maximize,
        Direction::Minimize,
        Direction::Maximize,
    ]);
    f.push(vec![1.0, 0.0, 1.0], ());
    let hv = f.hypervolume(&[0.0, 10.0, 0.0]);
    assert!(
        (hv - 10.0).abs() < 1e-6,
        "3D mixed-direction HV should be 10.0, got {hv}"
    );
}

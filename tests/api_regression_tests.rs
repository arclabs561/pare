use pare::{dominates, pareto_layers, Direction, FrontierError, ParetoFrontier};

#[test]
fn try_new_rejects_empty() {
    let points: Vec<Vec<f64>> = Vec::new();
    let err = ParetoFrontier::try_new(&points).unwrap_err();
    assert_eq!(err, FrontierError::Empty);
}

#[test]
fn try_new_rejects_inconsistent_dimensions() {
    let points = vec![vec![1.0, 2.0], vec![3.0]];
    let err = ParetoFrontier::try_new(&points).unwrap_err();
    assert_eq!(err, FrontierError::InconsistentDimensions);
}

#[test]
fn try_new_rejects_non_finite_values() {
    let points = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
    let err = ParetoFrontier::try_new(&points).unwrap_err();
    assert_eq!(
        err,
        FrontierError::NonFinite {
            point_idx: 0,
            dim_idx: 1
        }
    );
}

#[test]
fn dominates_is_irreflexive() {
    let eps = 1e-12;
    let a = [1.0, 2.0, 3.0];
    let dirs = [
        Direction::Maximize,
        Direction::Maximize,
        Direction::Minimize,
    ];
    assert!(!dominates(&dirs, eps, &a, &a));
}

#[test]
fn mixed_directions_dominance_example() {
    // Maximize accuracy, minimize latency.
    let dirs = [Direction::Maximize, Direction::Minimize];
    let eps = 1e-12;

    let a = [0.92, 80.0];
    let b = [0.90, 80.0];
    let c = [0.92, 120.0];

    assert!(dominates(&dirs, eps, &a, &b)); // better accuracy, equal latency
    assert!(dominates(&dirs, eps, &a, &c)); // equal accuracy, better latency
    assert!(!dominates(&dirs, eps, &c, &a));
}

#[test]
fn push_rejecting_dominated_point_is_noop() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    assert!(f.push(vec![1.0, 1.0], "a"));
    let before = f.points().to_vec();

    // Dominated by (1,1).
    assert!(!f.push(vec![0.5, 0.5], "b"));
    let after = f.points().to_vec();
    assert_eq!(before.len(), after.len());
    assert_eq!(before[0].values, after[0].values);
}

#[test]
fn crowding_distance_boundaries_are_infinite_in_2d_three_points() {
    // Three non-dominated points along a 2D trade-off curve.
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.0, 1.0], 0);
    f.push(vec![0.5, 0.5], 1);
    f.push(vec![1.0, 0.0], 2);

    let d = f.crowding_distances();
    assert_eq!(d.len(), 3);
    let inf = d.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(inf, 2);
    let finite = d.iter().filter(|x| x.is_finite()).count();
    assert_eq!(finite, 1);
}

#[test]
fn hypervolume_orients_minimize_dimensions_against_reference() {
    // Objective 0 is maximize; objective 1 is minimize.
    // Reference point is a "worst acceptable" vector.
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    let ref_point = [0.0, 10.0];

    // Improvement box: (x - ref_x, ref_y - y) = (1, 1) => area 1
    f.push(vec![1.0, 9.0], "a");
    let hv1 = f.hypervolume(&ref_point);
    assert!((hv1 - 1.0).abs() < 1e-9, "expected hv ~ 1, got {hv1}");

    // Add a dominated point (worse in both): should not change hypervolume.
    f.push(vec![0.5, 9.5], "b");
    let hv2 = f.hypervolume(&ref_point);
    assert!((hv2 - hv1).abs() < 1e-9, "hv should be unchanged");
}

// ---- scalar_score / best_index tests ----

#[test]
fn scalar_score_uniform_weights() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.0, 1.0], "low-high");
    f.push(vec![1.0, 0.0], "high-low");

    let s0 = f.scalar_score(0, &[1.0, 1.0]);
    let s1 = f.scalar_score(1, &[1.0, 1.0]);
    // With only 2 points at extremes, both should get similar scores.
    assert!(s0.is_finite());
    assert!(s1.is_finite());
}

#[test]
fn scalar_score_single_weight() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.0, 1.0], 0);
    f.push(vec![1.0, 0.0], 1);

    // Weight only the first dimension.
    let s0 = f.scalar_score(0, &[1.0]);
    let s1 = f.scalar_score(1, &[1.0]);
    assert!(s1 > s0, "point 1 has higher dim-0 value");
}

#[test]
fn best_index_prefers_weighted_dimension() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.0, 1.0], 0);
    f.push(vec![1.0, 0.0], 1);

    // Weight heavily on dim 0 => best should be point 1.
    let best = f.best_index(&[10.0, 0.0]);
    assert_eq!(best, Some(1));

    // Weight heavily on dim 1 => best should be point 0.
    let best = f.best_index(&[0.0, 10.0]);
    assert_eq!(best, Some(0));
}

#[test]
fn best_index_empty() {
    let f: ParetoFrontier<()> = ParetoFrontier::new(vec![Direction::Maximize]);
    assert_eq!(f.best_index(&[1.0]), None);
}

#[test]
#[should_panic(expected = "values len")]
fn push_wrong_dimension_panics() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0], "bad");
}

#[test]
#[should_panic(expected = "finite")]
fn push_nan_panics() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize]);
    f.push(vec![f64::NAN], "bad");
}

// ---- 3D / 4D hypervolume known-value tests ----

#[test]
fn hypervolume_3d_single_point() {
    let mut f = ParetoFrontier::new(vec![
        Direction::Maximize,
        Direction::Maximize,
        Direction::Maximize,
    ]);
    f.push(vec![1.0, 1.0, 1.0], ());
    let hv = f.hypervolume(&[0.0, 0.0, 0.0]);
    assert!(
        (hv - 1.0).abs() < 1e-9,
        "single unit cube should have hv = 1.0, got {hv}"
    );
}

#[test]
fn hypervolume_3d_two_non_dominated_points() {
    // Two non-dominated points: (1.0, 0.5, 0.5) and (0.5, 1.0, 1.0).
    // Box 1 volume = 1.0 * 0.5 * 0.5 = 0.25
    // Box 2 volume = 0.5 * 1.0 * 1.0 = 0.50
    // Intersection  = min(1,0.5) * min(0.5,1) * min(0.5,1) = 0.5 * 0.5 * 0.5 = 0.125
    // Union = 0.25 + 0.50 - 0.125 = 0.625
    let mut f = ParetoFrontier::new(vec![
        Direction::Maximize,
        Direction::Maximize,
        Direction::Maximize,
    ]);
    f.push(vec![1.0, 0.5, 0.5], "a");
    f.push(vec![0.5, 1.0, 1.0], "b");
    let hv = f.hypervolume(&[0.0, 0.0, 0.0]);
    assert!((hv - 0.625).abs() < 1e-9, "expected hv = 0.625, got {hv}");
}

#[test]
fn hypervolume_4d_single_point() {
    let mut f = ParetoFrontier::new(vec![
        Direction::Maximize,
        Direction::Maximize,
        Direction::Maximize,
        Direction::Maximize,
    ]);
    f.push(vec![1.0, 1.0, 1.0, 1.0], ());
    let hv = f.hypervolume(&[0.0, 0.0, 0.0, 0.0]);
    assert!(
        (hv - 1.0).abs() < 1e-9,
        "single unit hypercube in 4D should have hv = 1.0, got {hv}"
    );
}

// ---- back-compat helpers ----

#[test]
fn pareto_indices_basic() {
    use pare::pareto_indices;
    let points = vec![
        vec![1.0f32, 0.0],
        vec![0.0, 1.0],
        vec![0.5, 0.5],
        vec![0.2, 0.2], // dominated by (0.5, 0.5)
    ];
    let idx = pareto_indices(&points).unwrap();
    assert!(idx.contains(&0));
    assert!(idx.contains(&1));
    assert!(idx.contains(&2));
    assert!(!idx.contains(&3));
}

#[test]
fn pareto_indices_2d_basic() {
    use pare::pareto_indices_2d;
    let points = vec![vec![1.0f32, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let idx = pareto_indices_2d(&points).unwrap();
    assert_eq!(idx.len(), 3); // all non-dominated
}

#[test]
fn pareto_indices_k_dominance_basic() {
    use pare::pareto_indices_k_dominance;
    let points = vec![
        vec![1.0f32, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.5, 0.5, 0.5],
    ];
    // k=1: standard Pareto dominance.
    let idx = pareto_indices_k_dominance(&points, 1).unwrap();
    assert!(!idx.is_empty());
}

#[test]
fn dominates_same_length_as_directions() {
    // Verify correct behavior when all lengths match.
    let dirs = vec![Direction::Maximize, Direction::Maximize];
    assert!(dominates(&dirs, 1e-9, &[1.0, 1.0], &[0.5, 0.5]));
}

#[test]
fn pareto_indices_empty() {
    use pare::pareto_indices;
    let idx = pareto_indices(&[]).unwrap();
    assert!(idx.is_empty());
}

#[test]
fn pareto_indices_rejects_nan() {
    use pare::pareto_indices;
    let points = vec![vec![1.0f32, f32::NAN], vec![0.5, 0.5]];
    assert!(pareto_indices(&points).is_none());
}

#[test]
fn pareto_indices_2d_rejects_nan() {
    use pare::pareto_indices_2d;
    let points = vec![vec![1.0f32, f32::NAN], vec![0.5, 0.5]];
    assert!(pareto_indices_2d(&points).is_none());
}

#[test]
fn pareto_indices_k_dominance_rejects_nan() {
    use pare::pareto_indices_k_dominance;
    let points = vec![vec![1.0f32, f32::NAN], vec![0.5, 0.5]];
    assert!(pareto_indices_k_dominance(&points, 1).is_none());
}

#[test]
fn pareto_indices_rejects_inf() {
    use pare::pareto_indices;
    let points = vec![vec![1.0f32, f32::INFINITY], vec![0.5, 0.5]];
    assert!(pareto_indices(&points).is_none());
}

// ---- new API: eps() and stats() accessors ----

#[test]
fn eps_accessor_returns_default() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]);
    assert!((f.eps() - 1e-9).abs() < 1e-15);
}

#[test]
fn eps_accessor_returns_custom() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]).with_eps(0.01);
    assert!((f.eps() - 0.01).abs() < 1e-15);
}

#[test]
fn stats_accessor_after_push() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0, 2.0], ());
    f.push(vec![3.0, 1.0], ());
    let stats = f.stats();
    assert_eq!(stats.len(), 2);
    assert!((stats[0].min - 1.0).abs() < 1e-9);
    assert!((stats[0].max - 3.0).abs() < 1e-9);
}

// ---- new API: ideal_point / nadir_point ----

#[test]
fn ideal_point_empty_returns_none() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]);
    assert!(f.ideal_point().is_none());
}

#[test]
fn ideal_nadir_maximize_only() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.9, 0.1], ());
    f.push(vec![0.1, 0.9], ());

    let ideal = f.ideal_point().unwrap();
    assert!((ideal[0] - 0.9).abs() < 1e-9);
    assert!((ideal[1] - 0.9).abs() < 1e-9);

    let nadir = f.nadir_point().unwrap();
    assert!((nadir[0] - 0.1).abs() < 1e-9);
    assert!((nadir[1] - 0.1).abs() < 1e-9);
}

#[test]
fn ideal_nadir_mixed_directions() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    f.push(vec![0.9, 10.0], ());
    f.push(vec![0.7, 5.0], ());

    let ideal = f.ideal_point().unwrap();
    assert!((ideal[0] - 0.9).abs() < 1e-9); // max of maximize
    assert!((ideal[1] - 5.0).abs() < 1e-9); // min of minimize

    let nadir = f.nadir_point().unwrap();
    assert!((nadir[0] - 0.7).abs() < 1e-9); // min of maximize
    assert!((nadir[1] - 10.0).abs() < 1e-9); // max of minimize
}

// ---- new API: ranked_indices ----

#[test]
fn ranked_indices_order() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.9, 0.1], "A");
    f.push(vec![0.5, 0.5], "B");
    f.push(vec![0.1, 0.9], "C");

    // Weight heavily on dim 0
    let ranked = f.ranked_indices(&[1.0, 0.0]);
    assert_eq!(ranked[0], 0); // A best
    assert_eq!(ranked[2], 2); // C worst

    // Weight heavily on dim 1
    let ranked = f.ranked_indices(&[0.0, 1.0]);
    assert_eq!(ranked[0], 2); // C best
    assert_eq!(ranked[2], 0); // A worst
}

#[test]
fn ranked_indices_empty() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]);
    assert!(f.ranked_indices(&[1.0]).is_empty());
}

// ---- new API: ASF ----

#[test]
fn asf_ideal_point_scores_zero() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0, 1.0], "only");

    let ideal = f.ideal_point().unwrap();
    let score = f.asf(0, &[1.0, 1.0], &ideal);
    assert!(
        score.abs() < 1e-9,
        "point at ideal should have ASF ~0, got {score}"
    );
}

#[test]
fn asf_selects_balanced_point() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.9, 0.1], "A");
    f.push(vec![0.5, 0.5], "B");
    f.push(vec![0.1, 0.9], "C");

    let ideal = f.ideal_point().unwrap();
    let best = f.best_asf(&[1.0, 1.0], &ideal).unwrap();
    // B is the most balanced -- closest to ideal along equal weights
    assert_eq!(f.points()[best].data, "B");
}

#[test]
fn asf_respects_weights() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.9, 0.1], "A");
    f.push(vec![0.5, 0.5], "B");
    f.push(vec![0.1, 0.9], "C");

    let ideal = f.ideal_point().unwrap();
    // Heavily weight dim 0 -> should prefer A
    let best = f.best_asf(&[0.01, 1.0], &ideal).unwrap();
    assert_eq!(f.points()[best].data, "A");
}

#[test]
fn asf_zero_weight_returns_infinity() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize]);
    f.push(vec![0.5], ());
    let score = f.asf(0, &[0.0], &[1.0]);
    assert!(score.is_infinite());
}

#[test]
fn best_asf_empty_returns_none() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]);
    assert!(f.best_asf(&[1.0], &[0.0]).is_none());
}

#[test]
fn asf_mixed_directions() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    f.push(vec![0.9, 10.0], "high-acc");
    f.push(vec![0.7, 5.0], "low-lat");

    let ideal = f.ideal_point().unwrap();
    // Equal weights: should pick the more balanced tradeoff
    let s0 = f.asf(0, &[1.0, 1.0], &ideal);
    let s1 = f.asf(1, &[1.0, 1.0], &ideal);
    // Both should be non-negative (distance from ideal)
    assert!(s0 >= -1e-9);
    assert!(s1 >= -1e-9);
}

// ---- new API: retain ----

#[test]
fn retain_filters_points() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    f.push(vec![0.9, 100.0], "expensive");
    f.push(vec![0.7, 20.0], "cheap");
    f.push(vec![0.5, 5.0], "cheapest");

    f.retain(|p| p.values[1] < 50.0);
    assert_eq!(f.len(), 2);
    assert!(f.points().iter().all(|p| p.values[1] < 50.0));
}

#[test]
fn retain_updates_stats() {
    // Use 2D so all points are non-dominated tradeoffs
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0, 3.0], ());
    f.push(vec![2.0, 2.0], ());
    f.push(vec![3.0, 1.0], ());
    assert_eq!(f.len(), 3);

    f.retain(|p| p.values[0] <= 2.0);
    assert_eq!(f.len(), 2);
    assert!((f.stats()[0].max - 2.0).abs() < 1e-9);
}

#[test]
fn retain_all_removed_yields_empty() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize]);
    f.push(vec![1.0], ());
    f.retain(|_| false);
    assert!(f.is_empty());
}

// ---- new API: pareto_layers ----

#[test]
fn pareto_layers_basic() {
    let points = vec![
        vec![0.9f32, 0.1], // layer 0
        vec![0.5, 0.5],    // layer 0
        vec![0.4, 0.4],    // layer 1
        vec![0.3, 0.3],    // layer 2
    ];
    let layers = pareto_layers(&points).unwrap();
    assert_eq!(layers.len(), 3);
    assert!(layers[0].contains(&0));
    assert!(layers[0].contains(&1));
    assert_eq!(layers[1], vec![2]);
    assert_eq!(layers[2], vec![3]);
}

#[test]
fn pareto_layers_empty() {
    let layers = pareto_layers(&[]).unwrap();
    assert!(layers.is_empty());
}

#[test]
fn pareto_layers_all_nondominated() {
    let points = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
    let layers = pareto_layers(&points).unwrap();
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].len(), 2);
}

#[test]
fn pareto_layers_rejects_nan() {
    let points = vec![vec![1.0f32, f32::NAN]];
    assert!(pareto_layers(&points).is_none());
}

// ---- new API: from_points / Extend ----

#[test]
fn from_points_basic() {
    let items = vec![
        (vec![0.9, 0.1], "A"),
        (vec![0.5, 0.5], "B"),
        (vec![0.4, 0.4], "C"), // dominated by B
    ];
    let f = ParetoFrontier::from_points(vec![Direction::Maximize, Direction::Maximize], items);
    assert_eq!(f.len(), 2);
}

#[test]
fn from_points_empty_iter() {
    let f = ParetoFrontier::<()>::from_points(vec![Direction::Maximize], std::iter::empty());
    assert!(f.is_empty());
}

#[test]
fn extend_adds_nondominated_only() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![0.9, 0.1], "A");

    f.extend(vec![
        (vec![0.1, 0.9], "B"),  // non-dominated tradeoff
        (vec![0.4, 0.05], "C"), // dominated by A
    ]);
    assert_eq!(f.len(), 2); // A and B
}

// ---- new API: normalized_values ----

#[test]
fn normalized_values_extremes() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    f.push(vec![0.9, 10.0], "a"); // best acc, worst lat
    f.push(vec![0.7, 5.0], "b"); // worst acc, best lat

    let na = f.normalized_values(0).unwrap();
    assert!((na[0] - 1.0).abs() < 1e-9); // ideal in dim 0
    assert!((na[1] - 0.0).abs() < 1e-9); // nadir in dim 1

    let nb = f.normalized_values(1).unwrap();
    assert!((nb[0] - 0.0).abs() < 1e-9);
    assert!((nb[1] - 1.0).abs() < 1e-9);
}

#[test]
fn normalized_values_empty_returns_none() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]);
    assert!(f.normalized_values(0).is_none());
}

#[test]
fn normalized_values_single_point_returns_half() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize]);
    f.push(vec![5.0], ());
    let norm = f.normalized_values(0).unwrap();
    assert!((norm[0] - 0.5).abs() < 1e-9); // no spread -> 0.5
}

// ---- new API: suggest_ref_point ----

#[test]
fn suggest_ref_point_is_worse_than_nadir() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    f.push(vec![0.9, 10.0], ());
    f.push(vec![0.7, 5.0], ());

    let nadir = f.nadir_point().unwrap();
    let ref_pt = f.suggest_ref_point(0.1).unwrap();

    // For Maximize: ref < nadir (worse)
    assert!(ref_pt[0] < nadir[0]);
    // For Minimize: ref > nadir (worse)
    assert!(ref_pt[1] > nadir[1]);
}

#[test]
fn suggest_ref_point_produces_positive_hv() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0, 0.0], ());
    f.push(vec![0.0, 1.0], ());

    let ref_pt = f.suggest_ref_point(0.1).unwrap();
    let hv = f.hypervolume(&ref_pt);
    assert!(
        hv > 0.0,
        "HV with suggested ref point should be positive, got {hv}"
    );
}

#[test]
fn suggest_ref_point_empty_returns_none() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]);
    assert!(f.suggest_ref_point(0.1).is_none());
}

// ---- new API: hypervolume_contributions ----

#[test]
fn hv_contributions_are_positive_and_bounded() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0, 0.5], ());
    f.push(vec![0.5, 1.0], ());

    let total = f.hypervolume(&[0.0, 0.0]);
    let contribs = f.hypervolume_contributions(&[0.0, 0.0]);
    assert_eq!(contribs.len(), 2);
    // Each contribution is positive and <= total
    for &c in &contribs {
        assert!(c > 0.0, "contribution should be positive, got {c}");
        assert!(c <= total + 1e-9, "contribution {c} exceeds total {total}");
    }
    // Sum of contributions <= total (shared volume is counted once in total
    // but not in individual contributions)
    let sum: f64 = contribs.iter().sum();
    assert!(sum <= total + 1e-9);
}

#[test]
fn hv_contributions_empty() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]);
    assert!(f.hypervolume_contributions(&[0.0]).is_empty());
}

#[test]
fn hv_contributions_single_point() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0, 1.0], ());

    let contribs = f.hypervolume_contributions(&[0.0, 0.0]);
    assert_eq!(contribs.len(), 1);
    assert!((contribs[0] - 1.0).abs() < 1e-9); // entire HV
}

// ---- new API: knee_index ----

#[test]
fn knee_index_selects_balanced_point() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0, 0.0], "extreme_a");
    f.push(vec![0.6, 0.6], "knee");
    f.push(vec![0.0, 1.0], "extreme_b");

    let knee = f.knee_index().unwrap();
    assert_eq!(f.points()[knee].data, "knee");
}

#[test]
fn knee_index_too_few_points() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    f.push(vec![1.0, 0.0], ());
    f.push(vec![0.0, 1.0], ());
    assert!(f.knee_index().is_none()); // need >= 3 points
}

#[test]
fn knee_index_empty() {
    let f = ParetoFrontier::<()>::new(vec![Direction::Maximize]);
    assert!(f.knee_index().is_none());
}

#[test]
fn knee_index_mixed_directions() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    f.push(vec![0.9, 50.0], "fast");
    f.push(vec![0.7, 20.0], "balanced");
    f.push(vec![0.5, 5.0], "cheap");

    // Should return Some (3 points)
    let knee = f.knee_index().unwrap();
    // The balanced point should be the knee
    assert_eq!(f.points()[knee].data, "balanced");
}

#[test]
fn pareto_layers_total_count_matches_input() {
    let points = vec![
        vec![1.0f32, 0.0],
        vec![0.5, 0.5],
        vec![0.0, 1.0],
        vec![0.3, 0.3],
        vec![0.1, 0.1],
    ];
    let layers = pareto_layers(&points).unwrap();
    let total: usize = layers.iter().map(|l| l.len()).sum();
    assert_eq!(total, points.len());
}

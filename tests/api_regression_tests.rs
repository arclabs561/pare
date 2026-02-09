use pare::{dominates, Direction, FrontierError, ParetoFrontier};

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
fn pareto_indices_empty() {
    use pare::pareto_indices;
    let idx = pareto_indices(&[]).unwrap();
    assert!(idx.is_empty());
}

use pare::sensitivity::{analyze_redundancy, finite_difference_jacobian};
use pare::{Direction, ParetoFrontier};

#[test]
fn opposed_objectives_disprove_the_dimension_bound() {
    let objectives: [fn(&[f64]) -> f64; 2] = [|x| x[0], |x| -x[0]];
    let jacobian = finite_difference_jacobian(&[0.0], &objectives, 1e-4);
    let analysis = analyze_redundancy(&jacobian).unwrap();
    assert_eq!(analysis.effective_dimension(1e-6), 1);
    assert!((analysis.cosine(0, 1).unwrap() + 1.0).abs() < 1e-12);

    // On [-1, 1], x < y implies f(x) < f(y) and g(x) > g(y).
    // Thus the entire injective line segment is Pareto optimal, with
    // dimension one. These samples exercise the actual frontier API.
    let mut frontier = ParetoFrontier::new(vec![Direction::Maximize; 2]);
    for x in [-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
        frontier.push(vec![x, -x], x);
    }
    assert_eq!(frontier.points().len(), 5);
    // Retained for compatibility: this heuristic underestimates the dimension.
    assert_eq!(analysis.pareto_dimension_bound(1e-6), 0);
}

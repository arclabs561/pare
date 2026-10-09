//! `pareto_indices` and `pareto_indices_2d` must return the same set: a point
//! is kept iff no other point dominates it under the default 1e-9 tolerance
//! (the same `dominates` rule `ParetoFrontier` uses). Exact duplicates do not
//! dominate each other, so both copies stay.

use proptest::prelude::*;
use std::collections::BTreeSet;

fn as_set(v: Vec<usize>) -> BTreeSet<usize> {
    v.into_iter().collect()
}

#[test]
fn duplicate_points_are_all_kept_by_both_helpers() {
    let points = vec![vec![0.5f32, 0.5], vec![0.5, 0.5], vec![0.2, 0.9]];
    let want: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
    assert_eq!(as_set(pare::pareto_indices(&points).unwrap()), want);
    assert_eq!(as_set(pare::pareto_indices_2d(&points).unwrap()), want);
}

#[test]
fn differences_within_tolerance_do_not_dominate_in_either_helper() {
    // The next f32 after 0.001 is about 1.2e-10 larger, inside the 1e-9
    // tolerance, so neither point dominates the other.
    let x = 0.001f32;
    let x_next = f32::from_bits(x.to_bits() + 1);
    assert!(((x_next - x) as f64) < 1e-9);
    let points = vec![vec![x, 0.002], vec![x_next, 0.002]];
    let want: BTreeSet<usize> = [0, 1].into_iter().collect();
    assert_eq!(as_set(pare::pareto_indices(&points).unwrap()), want);
    assert_eq!(as_set(pare::pareto_indices_2d(&points).unwrap()), want);
}

fn grid_point() -> impl Strategy<Value = Vec<f32>> {
    // A coarse grid makes ties and duplicates common.
    prop::collection::vec((0u8..6).prop_map(|k| k as f32 / 5.0), 2..=2)
}

proptest! {
    #[test]
    fn helpers_agree_with_ties_and_duplicates(
        points in prop::collection::vec(grid_point(), 1..40)
    ) {
        let general = as_set(pare::pareto_indices(&points).unwrap());
        let sweep = as_set(pare::pareto_indices_2d(&points).unwrap());
        prop_assert_eq!(&general, &sweep);

        // Both match the definition: kept iff no other point dominates it.
        let dirs = vec![pare::Direction::Maximize; 2];
        let f64pts: Vec<Vec<f64>> = points
            .iter()
            .map(|p| p.iter().map(|&v| v as f64).collect())
            .collect();
        let brute: BTreeSet<usize> = (0..points.len())
            .filter(|&i| {
                !(0..points.len())
                    .any(|j| j != i && pare::dominates(&dirs, 1e-9, &f64pts[j], &f64pts[i]))
            })
            .collect();
        prop_assert_eq!(general, brute);
    }
}

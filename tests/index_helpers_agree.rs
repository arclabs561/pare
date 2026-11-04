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

/// A dominates B and B dominates C under the tolerance, but A does not
/// dominate C (C is better in y by more than 1e-9), so only A is on the front.
fn near_tie_chain() -> Vec<Vec<f32>> {
    vec![vec![3e-9, 0.0], vec![0.0, 0.8e-9], vec![-2e-9, 1.6e-9]]
}

fn sorted_layers(layers: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    layers
        .into_iter()
        .map(|mut l| {
            l.sort();
            l
        })
        .collect()
}

#[test]
fn pareto_layers_first_layer_is_pareto_indices_under_tolerance() {
    let pts = near_tie_chain();
    assert_eq!(pare::pareto_indices(&pts).unwrap(), vec![0]);
    let layers = sorted_layers(pare::pareto_layers(&pts).unwrap());
    assert_eq!(layers, vec![vec![0], vec![1], vec![2]]);
}

#[test]
fn pareto_layers_do_not_depend_on_input_order() {
    let pts = near_tie_chain();
    let n = pts.len();
    let rev: Vec<Vec<f32>> = pts.iter().rev().cloned().collect();
    let fwd = sorted_layers(pare::pareto_layers(&pts).unwrap());
    let back = sorted_layers(
        pare::pareto_layers(&rev)
            .unwrap()
            .into_iter()
            .map(|l| l.into_iter().map(|i| n - 1 - i).collect())
            .collect(),
    );
    assert_eq!(fwd, back);
}

#[test]
fn pareto_layers_put_a_dominance_cycle_in_one_layer() {
    // In 3-d the tolerance allows a cycle: a dominates b, b dominates c and
    // c dominates a, so no point is non-dominated.
    let pts = vec![
        vec![0.0f32, 0.0, 0.0],
        vec![-1.2e-9, 0.6e-9, 0.6e-9],
        vec![-0.6e-9, -0.6e-9, 1.2e-9],
    ];
    assert_eq!(pare::pareto_indices(&pts).unwrap(), Vec::<usize>::new());
    let layers = sorted_layers(pare::pareto_layers(&pts).unwrap());
    assert_eq!(layers, vec![vec![0, 1, 2]]);
}

proptest! {
    #[test]
    fn pareto_layers_first_layer_matches_pareto_indices(
        points in prop::collection::vec(grid_point(), 1..40)
    ) {
        let layers = pare::pareto_layers(&points).unwrap();
        prop_assert_eq!(as_set(layers[0].clone()), as_set(pare::pareto_indices(&points).unwrap()));
        let total: usize = layers.iter().map(Vec::len).sum();
        prop_assert_eq!(total, points.len());
    }
}

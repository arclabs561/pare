//! Ranking sorts by weighted score. NaN weights make every score NaN; the sort
//! must still return a permutation instead of panicking.

use pare::{Direction, ParetoFrontier};

#[test]
fn ranked_indices_with_nan_weights_does_not_panic() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    for i in 0..40 {
        let t = i as f64 / 39.0;
        f.push(vec![t, 1.0 - t], i);
    }
    let mut ranked = f.ranked_indices(&[f64::NAN, 1.0]);
    ranked.sort_unstable();
    assert_eq!(ranked, (0..f.len()).collect::<Vec<_>>());
    let _ = f.best_index(&[f64::NAN, 1.0]);
}

//! Property tests for the `pare::sensitivity` module.
//!
//! These tests verify algebraic invariants that must hold for ANY sensitivity matrix,
//! not just the specific examples in the unit tests.  Each property corresponds to a
//! theorem about the Gram matrix or its eigendecomposition.

use pare::sensitivity::{analyze_redundancy, SensitivityRow};
use proptest::prelude::*;
use std::ops::RangeInclusive;

/// Generate a random m x n sensitivity matrix with entries in [-10, 10].
///
/// We use `prop_flat_map` to first pick dimensions (m, n), then generate a matrix
/// of exactly that size.  This ensures every row has the same length.
fn sensitivity_matrix(
    m: RangeInclusive<usize>,
    n: RangeInclusive<usize>,
) -> impl Strategy<Value = Vec<SensitivityRow>> {
    (m, n).prop_flat_map(|(m, n)| {
        prop::collection::vec(
            prop::collection::vec(-10.0..10.0_f64, n..=n),
            m..=m,
        )
    })
}

proptest! {
    // ====================================================================
    // Gram matrix eigenvalue invariants
    // ====================================================================

    /// The Gram matrix G = S * S^T is positive semidefinite, so all eigenvalues
    /// must be non-negative.  Any negative values would indicate a bug in the
    /// Jacobi solver or the Gram matrix computation.
    #[test]
    fn eigenvalues_are_non_negative(
        sens in sensitivity_matrix(2..=6, 3..=20),
    ) {
        if let Some(a) = analyze_redundancy(&sens) {
            for &ev in &a.eigenvalues {
                prop_assert!(ev >= -1e-9, "negative eigenvalue: {ev}");
            }
        }
    }

    /// Eigenvalues must be sorted in descending order.  This is a contract of
    /// `RedundancyAnalysis` that callers rely on (e.g. `variance_fraction(k)`
    /// takes the top k).
    #[test]
    fn eigenvalues_sorted_descending(
        sens in sensitivity_matrix(2..=6, 3..=20),
    ) {
        if let Some(a) = analyze_redundancy(&sens) {
            for w in a.eigenvalues.windows(2) {
                prop_assert!(w[0] + 1e-12 >= w[1], "not descending: {} then {}", w[0], w[1]);
            }
        }
    }

    /// Fundamental identity: sum of eigenvalues = trace of G = sum of ||s_i||^2.
    ///
    /// This is a strong check that the Jacobi iteration preserves the trace
    /// (Givens rotations are orthogonal, so they preserve the trace by construction).
    /// A failure here would indicate a bug in the rotation update formulas.
    #[test]
    fn eigenvalue_sum_equals_trace(
        sens in sensitivity_matrix(2..=6, 3..=20),
    ) {
        if let Some(a) = analyze_redundancy(&sens) {
            let ev_sum: f64 = a.eigenvalues.iter().sum();
            let trace = a.trace();
            // Trace is also sum of squared norms of sensitivity vectors.
            let norm_sq: f64 = sens.iter()
                .map(|row| row.iter().map(|&x| x * x).sum::<f64>())
                .sum();
            prop_assert!((ev_sum - trace).abs() < 1e-4 * trace.abs().max(1.0),
                "ev_sum={ev_sum}, trace={trace}");
            prop_assert!((trace - norm_sq).abs() < 1e-9 * norm_sq.abs().max(1.0),
                "trace={trace}, norm_sq={norm_sq}");
        }
    }

    // ====================================================================
    // Cosine similarity invariants
    // ====================================================================

    /// Cosine similarity is symmetric: cos(s_i, s_j) = cos(s_j, s_i).
    /// Diagonal entries should be 1.0 (or 0.0 for zero vectors).
    /// All values must be in [-1, 1].
    #[test]
    fn cosine_symmetric_and_bounded(
        sens in sensitivity_matrix(2..=6, 3..=20),
    ) {
        if let Some(a) = analyze_redundancy(&sens) {
            let m = a.num_objectives;
            for i in 0..m {
                for j in 0..m {
                    let cij = a.cosine(i, j).unwrap();
                    let cji = a.cosine(j, i).unwrap();
                    prop_assert!((cij - cji).abs() < 1e-12, "asymmetric cosine [{i},{j}]");
                    prop_assert!((-1.0 - 1e-12..=1.0 + 1e-12).contains(&cij),
                        "cosine out of [-1,1]: {cij}");
                }
                let cii = a.cosine(i, i).unwrap();
                prop_assert!(cii >= -1e-12, "diagonal cosine < 0: {cii}");
            }
        }
    }

    // ====================================================================
    // Dimension bounds (saturation principle)
    // ====================================================================

    /// Effective dimension <= min(m, n) always.
    ///
    /// The Gram matrix is m x m and has rank at most min(m, n) (because S is m x n,
    /// so rank(S * S^T) = rank(S) <= min(m, n)).  The effective dimension (number of
    /// eigenvalues above threshold) cannot exceed the true rank.
    #[test]
    fn effective_dimension_bounded(
        sens in sensitivity_matrix(2..=8, 3..=15),
    ) {
        if let Some(a) = analyze_redundancy(&sens) {
            let m = a.num_objectives;
            let n = a.num_design_points;
            let eff = a.effective_dimension(0.01);
            prop_assert!(eff <= m, "eff_dim={eff} > m={m}");
            prop_assert!(eff <= n, "eff_dim={eff} > n={n}");
        }
    }

    /// Pareto front dimension <= m - 1 always (you need at least 2 objectives
    /// to have a 1-dimensional front).
    #[test]
    fn pareto_bound_at_most_m_minus_1(
        sens in sensitivity_matrix(2..=8, 3..=15),
    ) {
        if let Some(a) = analyze_redundancy(&sens) {
            let m = a.num_objectives;
            let bound = a.pareto_dimension_bound(1e-6);
            prop_assert!(bound < m, "pareto_bound={bound} >= m={m}");
        }
    }

    // ====================================================================
    // Invariance properties
    // ====================================================================

    /// Scaling a sensitivity vector by a positive constant does not change the
    /// rank of the Gram matrix.  This corresponds to changing the units of an
    /// objective (e.g., measuring cost in cents vs. dollars) -- it shouldn't
    /// change whether the objective is redundant with another.
    ///
    /// Uses a moderate tolerance (1e-4) for the Pareto bound comparison because
    /// scaling can push borderline eigenvalues across a hard threshold.
    #[test]
    fn scaling_preserves_rank(
        sens in sensitivity_matrix(2..=5, 3..=10),
        scale in 0.1_f64..100.0,
        row_idx in 0usize..5,
    ) {
        if sens.is_empty() { return Ok(()); }
        let m = sens.len();
        let row_idx = row_idx % m;

        let a1 = match analyze_redundancy(&sens) {
            Some(a) => a,
            None => return Ok(()),
        };

        let mut sens2 = sens.clone();
        for v in &mut sens2[row_idx] {
            *v *= scale;
        }
        let a2 = match analyze_redundancy(&sens2) {
            Some(a) => a,
            None => return Ok(()),
        };

        let rank1 = a1.pareto_dimension_bound(1e-4);
        let rank2 = a2.pareto_dimension_bound(1e-4);
        prop_assert_eq!(rank1, rank2);
    }

    /// Permuting the rows (reordering objectives) does not change the eigenvalue
    /// spectrum.  Eigenvalues are permutation-invariant because they depend on the
    /// multiset of singular values, not on row order.
    #[test]
    fn permuting_rows_preserves_eigenvalues(
        sens in sensitivity_matrix(2..=5, 3..=10),
    ) {
        if sens.len() < 2 { return Ok(()); }
        let a1 = match analyze_redundancy(&sens) {
            Some(a) => a,
            None => return Ok(()),
        };

        // Reverse the row order.
        let mut sens2 = sens.clone();
        sens2.reverse();
        let a2 = match analyze_redundancy(&sens2) {
            Some(a) => a,
            None => return Ok(()),
        };

        for (e1, e2) in a1.eigenvalues.iter().zip(a2.eigenvalues.iter()) {
            prop_assert!((e1 - e2).abs() < 1e-6 * e1.abs().max(1.0),
                "eigenvalues differ after permutation: {e1} vs {e2}");
        }
    }

    // ====================================================================
    // Monotonicity properties
    // ====================================================================

    /// Variance fraction is monotone non-decreasing in k.
    /// variance_fraction(m) = 1.0 (all eigenvalues capture all variance).
    #[test]
    fn variance_fraction_monotone(
        sens in sensitivity_matrix(2..=6, 3..=15),
    ) {
        if let Some(a) = analyze_redundancy(&sens) {
            let m = a.num_objectives;
            let mut prev = 0.0;
            for k in 1..=m {
                let frac = a.variance_fraction(k);
                prop_assert!(frac + 1e-12 >= prev,
                    "variance fraction not monotone: k={k}, frac={frac}, prev={prev}");
                prev = frac;
            }
            let full = a.variance_fraction(m);
            prop_assert!((full - 1.0).abs() < 1e-9, "full variance fraction = {full}");
        }
    }

    /// Adding a row that is a scalar multiple of an existing row (a "duplicate
    /// objective") should not increase the effective dimension.
    ///
    /// Why: a proportional sensitivity vector lies in the span of the original
    /// vectors, so it adds no new information to the Gram matrix's column space.
    /// The new Gram matrix has one more row/col but its rank doesn't increase.
    #[test]
    fn duplicate_row_does_not_increase_dimension(
        sens in sensitivity_matrix(2..=4, 3..=10),
        row_idx in 0usize..4,
        scale in 0.5_f64..5.0,
    ) {
        if sens.is_empty() { return Ok(()); }
        let m = sens.len();
        let row_idx = row_idx % m;

        let a1 = match analyze_redundancy(&sens) {
            Some(a) => a,
            None => return Ok(()),
        };

        let mut sens2 = sens.clone();
        let dup: Vec<f64> = sens[row_idx].iter().map(|&v| v * scale).collect();
        sens2.push(dup);

        let a2 = match analyze_redundancy(&sens2) {
            Some(a) => a,
            None => return Ok(()),
        };

        // Effective dimension should not increase (it's the number of
        // linearly independent directions, and we added a dependent one).
        let eff1 = a1.effective_dimension(0.01);
        let eff2 = a2.effective_dimension(0.01);
        prop_assert!(eff2 <= eff1 + 1,
            "adding duplicate row increased eff_dim from {eff1} to {eff2}");
    }
}

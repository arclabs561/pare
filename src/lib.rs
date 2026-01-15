//! # pare
//!
//! Pareto frontier and skyline queries.
//!
//! Finds the set of "non-dominated" points in a dataset. A point is dominated
//! if another point exists that is better (or equal) in all dimensions and
//! strictly better in at least one.
//!
//! ## Mathematical Formulation
//!
//! Given a set of vectors $S \subset \mathbb{R}^d$, a point $p \in S$ is
//! **dominated** by $q \in S$ ($q \succ p$) if:
//!
//! $$ \forall i \in \{1,\dots,d\}, q_i \ge p_i \quad \text{and} \quad \exists j, q_j > p_j $$
//!
//! The **Pareto frontier** is the set of non-dominated points:
//!
//! $$ P = \{ p \in S \mid \nexists q \in S, q \succ p \} $$
//!
//! ## Uses in RAG and Search
//!
//! - **Multi-objective retrieval**: Optimizing for (relevance, recency, diversity).
//! - **Model selection**: Trading off (latency, accuracy, cost).
//! - **Index pruning**: Keeping only "efficient" items.
//!
//! ## Research Context
//!
//! In database literature, this is known as the **Skyline Operator** (Börzsönyi et al., 2001).
//!
//! ### The Curse of Dimensionality
//!
//! In high dimensions ($d > 10$), almost every point becomes part of the Pareto frontier
//! because it is likely to be strictly best in at least one dimension.
//!
//! **Mitigation**: This crate implements **$k$-dominance**, where $p \succ_k q$ if
//! $p$ is no worse than $q$ in **at least `k` dimensions** and strictly better in at least one.
//! (This relaxes classical dominance and usually shrinks the skyline in high dimensions.)
//!
//! ## Example
//!
//! ```rust
//! use pare::ParetoFrontier;
//!
//! // Objectives: [Relevance, Recency] (higher is better)
//! let candidates = vec![
//!     vec![0.9, 0.1], // A: High relevance, old
//!     vec![0.5, 0.5], // B: Balanced
//!     vec![0.1, 0.9], // C: Low relevance, new
//!     vec![0.4, 0.4], // D: Dominated by B
//! ];
//!
//! let frontier = ParetoFrontier::try_new(&candidates).unwrap().indices();
//! // Should contain indices for A, B, C. D is removed.
//! ```

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dimensions mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("empty input")]
    EmptyInput,
    #[error("non-finite value at point {point_idx}, dim {dim_idx}: {value}")]
    NonFiniteValue {
        point_idx: usize,
        dim_idx: usize,
        value: f32,
    },
    #[error("invalid k-dominance parameter: k={k}, dim={dim}")]
    InvalidKDominance { k: usize, dim: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Pareto frontier calculator.
pub struct ParetoFrontier<'a> {
    points: &'a [Vec<f32>],
    dim: usize,
}

impl<'a> ParetoFrontier<'a> {
    /// Create a new analysis for a set of points after validation.
    ///
    /// Assumes all dimensions are "higher is better".
    /// If you have "lower is better" costs, negate them first.
    pub fn try_new(points: &'a [Vec<f32>]) -> Result<Self> {
        if points.is_empty() {
            return Err(Error::EmptyInput);
        }
        let dim = points[0].len();
        if dim == 0 {
            return Err(Error::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        for p in points.iter() {
            if p.len() != dim {
                return Err(Error::DimensionMismatch {
                    expected: dim,
                    got: p.len(),
                });
            }
        }

        // Policy: reject NaN/±inf early.
        //
        // Rationale: dominance comparisons on NaN are ill-defined (all comparisons are false),
        // which can silently poison frontier computation. If you want a different policy (e.g.
        // treat NaN as -inf), it should be explicit and tested.
        for (point_idx, p) in points.iter().enumerate() {
            for (dim_idx, &v) in p.iter().enumerate() {
                if !v.is_finite() {
                    return Err(Error::NonFiniteValue {
                        point_idx,
                        dim_idx,
                        value: v,
                    });
                }
            }
        }
        Ok(Self { points, dim })
    }

    /// Find indices of points on the Pareto frontier.
    ///
    /// Uses a naive O(N^2) comparison for simplicity and correctness in high dimensions.
    /// For d=2 or d=3, faster sweep-line algorithms exist (O(N log N)), but
    /// N^2 is often faster for small N (e.g., re-ranking 100 docs).
    pub fn indices(&self) -> Vec<usize> {
        let mut frontier = Vec::new();

        for i in 0..self.points.len() {
            let mut dominated = false;
            for j in 0..self.points.len() {
                if i == j {
                    continue;
                }

                if self.dominates(&self.points[j], &self.points[i]) {
                    dominated = true;
                    break;
                }
            }

            if !dominated {
                frontier.push(i);
            }
        }

        frontier
    }

    /// Find indices of points on the k-dominance skyline.
    ///
    /// This uses the k-dominance relation:
    ///
    /// - Let \(c = |\{ i : a_i \ge b_i \}|\) be the number of dimensions where `a` is no worse.
    /// - `a` **k-dominates** `b` iff \(c \ge k\) and `a` is strictly better in at least one
    ///   of those \(c\) dimensions.
    ///
    /// Notes:
    /// - For `k = dim`, this reduces to standard Pareto dominance.
    /// - For small `k`, the skyline can be much smaller (useful in high `dim`).
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidKDominance` if `k == 0` or `k > dim`.
    pub fn indices_k_dominance(&self, k: usize) -> Result<Vec<usize>> {
        if k == 0 || k > self.dim {
            return Err(Error::InvalidKDominance { k, dim: self.dim });
        }

        let mut skyline = Vec::new();
        for i in 0..self.points.len() {
            let mut dominated = false;
            for j in 0..self.points.len() {
                if i == j {
                    continue;
                }
                if self.k_dominates(&self.points[j], &self.points[i], k) {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                skyline.push(i);
            }
        }
        Ok(skyline)
    }

    /// Fast 2D skyline indices (O(n log n)).
    ///
    /// This is a specialized implementation for `dim = 2` that preserves the crate's strict
    /// dominance semantics:
    /// - duplicates of a frontier maximum are kept
    /// - points dominated by a strictly better x with equal y are removed
    pub fn indices_2d(&self) -> Result<Vec<usize>> {
        if self.dim != 2 {
            return Err(Error::DimensionMismatch {
                expected: 2,
                got: self.dim,
            });
        }

        // Sort by x descending, then y descending, stable tie-break by index.
        let mut order: Vec<usize> = (0..self.points.len()).collect();
        order.sort_by(|&i, &j| {
            let xi = self.points[i][0];
            let xj = self.points[j][0];
            xj.total_cmp(&xi)
                .then_with(|| self.points[j][1].total_cmp(&self.points[i][1]))
                .then_with(|| i.cmp(&j))
        });

        let mut frontier = Vec::new();
        let mut best_y = f32::NEG_INFINITY;

        // Process equal-x blocks together to correctly handle duplicates.
        let mut idx = 0usize;
        while idx < order.len() {
            let x = self.points[order[idx]][0];
            let mut end = idx + 1;
            while end < order.len() && self.points[order[end]][0] == x {
                end += 1;
            }

            let mut block_max_y = f32::NEG_INFINITY;
            for &p in &order[idx..end] {
                block_max_y = block_max_y.max(self.points[p][1]);
            }

            if block_max_y > best_y {
                for &p in &order[idx..end] {
                    if self.points[p][1] == block_max_y {
                        frontier.push(p);
                    }
                }
                best_y = block_max_y;
            }

            idx = end;
        }

        Ok(frontier)
    }

    /// Check if point A dominates point B.
    ///
    /// A dominates B if A >= B in all dims and A > B in at least one.
    fn dominates(&self, a: &[f32], b: &[f32]) -> bool {
        debug_assert_eq!(a.len(), self.dim);
        debug_assert_eq!(b.len(), self.dim);

        let mut better_in_one = false;
        for (va, vb) in a.iter().zip(b.iter()) {
            if va < vb {
                return false; // A is worse in this dimension
            }
            if va > vb {
                better_in_one = true;
            }
        }

        better_in_one
    }

    fn k_dominates(&self, a: &[f32], b: &[f32], k: usize) -> bool {
        debug_assert_eq!(a.len(), self.dim);
        debug_assert_eq!(b.len(), self.dim);
        debug_assert!(k > 0 && k <= self.dim);

        let mut ge = 0usize;
        let mut strictly_better = false;
        for (va, vb) in a.iter().zip(b.iter()) {
            if va >= vb {
                ge += 1;
                if va > vb {
                    strictly_better = true;
                }
            }
        }
        ge >= k && strictly_better
    }
}

/// Convenience function to get frontier indices.
pub fn pareto_indices(points: &[Vec<f32>]) -> Result<Vec<usize>> {
    Ok(ParetoFrontier::try_new(points)?.indices())
}

/// Convenience function to get k-dominance skyline indices.
pub fn pareto_indices_k_dominance(points: &[Vec<f32>], k: usize) -> Result<Vec<usize>> {
    ParetoFrontier::try_new(points)?.indices_k_dominance(k)
}

/// Convenience function for fast 2D skyline indices.
pub fn pareto_indices_2d(points: &[Vec<f32>]) -> Result<Vec<usize>> {
    ParetoFrontier::try_new(points)?.indices_2d()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_simple_frontier_2d() {
        let points = vec![
            vec![1.0, 0.0], // A: Best x
            vec![0.0, 1.0], // B: Best y
            vec![0.5, 0.5], // C: Tradeoff
            vec![0.4, 0.4], // D: Dominated by C
            vec![0.0, 0.0], // E: Dominated by all
        ];

        let indices = pareto_indices(&points).unwrap();
        assert!(indices.contains(&0)); // A
        assert!(indices.contains(&1)); // B
        assert!(indices.contains(&2)); // C
        assert!(!indices.contains(&3)); // D
        assert!(!indices.contains(&4)); // E
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_duplicates() {
        let points = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0], // Duplicate of optimal
            vec![0.5, 0.5],
        ];

        let indices = pareto_indices(&points).unwrap();
        // Both optimal points should be kept (neither dominates the other strictly)
        // Definition: q > p in at least one dim.
        // (1,1) vs (1,1): >= in all, but not > in any. So not dominated.
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(!indices.contains(&2));
    }

    #[test]
    fn test_high_dim() {
        let points = vec![
            vec![1.0, 1.0, 1.0], // Best
            vec![1.0, 1.0, 0.0], // Dominated
            vec![0.0, 0.0, 0.0], // Dominated
        ];
        let indices = pareto_indices(&points).unwrap();
        assert_eq!(indices, vec![0]);
    }

    proptest! {
        #[test]
        fn prop_frontier_points_are_not_dominated(
            points in prop::collection::vec(prop::collection::vec(-10.0f32..10.0, 3), 1..64)
        ) {
            let frontier = ParetoFrontier::try_new(&points).unwrap();
            let idxs = frontier.indices();
            for &i in &idxs {
                for j in 0..points.len() {
                    if i == j { continue; }
                    prop_assert!(
                        !frontier.dominates(&points[j], &points[i]),
                        "frontier point {} was dominated by {}",
                        i,
                        j
                    );
                }
            }
        }

        #[test]
        fn prop_non_frontier_points_are_dominated_by_someone(
            points in prop::collection::vec(prop::collection::vec(-10.0f32..10.0, 3), 1..64)
        ) {
            let frontier = ParetoFrontier::try_new(&points).unwrap();
            let idxs = frontier.indices();
            let mut is_frontier = vec![false; points.len()];
            for &i in &idxs {
                is_frontier[i] = true;
            }

            for i in 0..points.len() {
                if is_frontier[i] {
                    continue;
                }
                let mut found = false;
                for j in 0..points.len() {
                    if i == j {
                        continue;
                    }
                    if frontier.dominates(&points[j], &points[i]) {
                        found = true;
                        break;
                    }
                }
                prop_assert!(found, "point {i} was not on frontier but no dominator found");
            }
        }

        #[test]
        fn prop_k_dominance_k_equals_dim_matches_standard(
            points in prop::collection::vec(prop::collection::vec(-10.0f32..10.0, 3), 1..64)
        ) {
            let f = ParetoFrontier::try_new(&points).unwrap();
            let std = f.indices();
            let kd = f.indices_k_dominance(3).unwrap();

            let mut std_sorted = std.clone();
            std_sorted.sort_unstable();
            let mut kd_sorted = kd.clone();
            kd_sorted.sort_unstable();
            prop_assert_eq!(std_sorted, kd_sorted);
        }

        #[test]
        fn prop_k_dominance_skyline_points_not_k_dominated(
            points in prop::collection::vec(prop::collection::vec(-10.0f32..10.0, 4), 1..64),
            k in 1usize..=4
        ) {
            let f = ParetoFrontier::try_new(&points).unwrap();
            let idxs = f.indices_k_dominance(k).unwrap();
            for &i in &idxs {
                for j in 0..points.len() {
                    if i == j { continue; }
                    prop_assert!(
                        !f.k_dominates(&points[j], &points[i], k),
                        "k-dominance skyline point {} was k-dominated by {} (k={})",
                        i, j, k
                    );
                }
            }
        }

        #[test]
        fn prop_2d_sweep_matches_naive(
            points in prop::collection::vec(prop::collection::vec(-10.0f32..10.0, 2), 1..200)
        ) {
            let f = ParetoFrontier::try_new(&points).unwrap();
            let mut naive = f.indices();
            naive.sort_unstable();
            let mut fast = f.indices_2d().unwrap();
            fast.sort_unstable();
            prop_assert_eq!(naive, fast);
        }
    }

    #[test]
    fn test_rejects_nan_and_inf() {
        let points = vec![vec![1.0, f32::NAN], vec![0.0, 0.0]];
        assert!(matches!(
            ParetoFrontier::try_new(&points),
            Err(Error::NonFiniteValue { .. })
        ));

        let points = vec![vec![1.0, f32::INFINITY], vec![0.0, 0.0]];
        assert!(matches!(
            ParetoFrontier::try_new(&points),
            Err(Error::NonFiniteValue { .. })
        ));
    }
}

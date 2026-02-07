//! Objective redundancy analysis via the gradient test.
//!
//! # The problem
//!
//! A sequential sampling policy (bandit, adaptive experiment, online router) produces a
//! single data stream.  That stream is simultaneously evaluated by multiple objectives:
//! regret, estimation quality, detection speed, fairness, etc.  The question is: how many
//! of those objectives are genuinely in conflict?
//!
//! If two objectives always improve or worsen together (proportional sensitivity), then
//! optimizing one automatically optimizes the other -- they are **redundant**.  The number
//! of genuinely independent tradeoff axes determines whether multi-objective optimization
//! is necessary or whether a simpler single-objective approach suffices.
//!
//! # The gradient test
//!
//! For each objective `L_i`, define the **sensitivity function** at design point `j`:
//!
//! ```text
//!   s_i(j) = dL_i / d mu(j)
//! ```
//!
//! where `mu(j)` is the sampling weight at design point `j` (an arm, a covariate cell,
//! or an (arm, cell) pair).  This measures: "if I add an infinitesimal amount of sampling
//! at design point `j`, how does objective `i` change?"
//!
//! The m sensitivity vectors `{s_1, ..., s_m}` live in R^n (one component per design
//! point).  Their linear algebra reveals the tradeoff structure:
//!
//! - **Cosine similarity** cos(s_i, s_j) = 1.0 means objectives i and j are perfectly
//!   aligned (redundant).  cos = -1.0 means perfectly opposed (maximally conflicting).
//!   cos ~ 0 means independent (perturbations that help one are irrelevant to the other).
//!
//! - The **Gram matrix** G_{ij} = <s_i, s_j> (inner product over design points) is
//!   symmetric positive semidefinite.  Its eigenvalues are the squared lengths of the
//!   principal axes of the sensitivity ellipsoid.
//!
//! - **Rank of G** = number of linearly independent sensitivity directions.
//!   Pareto front dimension = rank - 1 (at most), since one direction is consumed by
//!   the "move along the front" degree of freedom.
//!
//! - **Effective dimension** at tolerance eps = number of eigenvalues above
//!   `eps * max_eigenvalue`.  This is a soft rank that ignores numerically negligible
//!   tradeoff axes.
//!
//! # Saturation principle
//!
//! The Pareto front dimension is bounded by `min(m - 1, D_eff)` where `D_eff` is the
//! effective dimension of the design space (K-1 for non-contextual with K arms, ~K*M
//! for M covariate cells).  You can name 17 objectives, but if your design has 3
//! degrees of freedom, at most 4 can be simultaneously non-redundant.
//!
//! References: Ehrgott & Nickel 2002 (Helly's theorem applied to Pareto optimality),
//! Zhen et al. 2018 (degenerate Pareto fronts from functional dependence).
//!
//! # Average-case vs. worst-case detection
//!
//! A key structural finding:
//!
//! - **Average-case objectives** (average detection delay, integrated MSE) both have
//!   sensitivity proportional to `1/p(x)^2` at each design point x.  They measure "how
//!   many total observations?" in the same way, so they are structurally redundant
//!   *regardless of the design or the number of covariate cells*.
//!
//! - **Worst-case objectives** (max detection delay over cells) concentrate all
//!   sensitivity on the single bottleneck cell -- the (arm, covariate) pair with the
//!   fewest observations.  This is a point mass in sensitivity space, linearly
//!   independent from the smooth profiles of regret and estimation.
//!
//! This means: adding "average detection delay" to an objective set that already
//! includes "average MSE" creates zero new tradeoff.  Only worst-case (minimax)
//! detection introduces a genuinely new axis.  The non-contextual case has only one
//! cell, so average and worst-case are identical and the distinction is moot.

use std::cmp::Ordering;

/// A row of the sensitivity matrix: one objective's sensitivity across all design points.
///
/// Row `i` has length `n` (the number of design points).  Entry `j` is the partial
/// derivative of objective `i` with respect to the sampling weight at design point `j`.
pub type SensitivityRow = Vec<f64>;

/// Result of objective redundancy analysis.
///
/// All fields are derived from the `m x n` sensitivity matrix S whose rows are the
/// sensitivity vectors of the m objectives over n design points.
#[derive(Debug, Clone)]
pub struct RedundancyAnalysis {
    /// Number of objectives (m).
    pub num_objectives: usize,
    /// Number of design points (n).
    pub num_design_points: usize,
    /// Eigenvalues of the m x m Gram matrix G = S * S^T, sorted descending.
    ///
    /// These are non-negative (G is PSD).  Their sum equals the trace of G,
    /// which equals the sum of squared norms of the sensitivity vectors.
    /// The k-th eigenvalue measures how much objective-space variance is
    /// captured by the k-th principal tradeoff axis.
    pub eigenvalues: Vec<f64>,
    /// Pairwise cosine similarity matrix (m x m, row-major).
    ///
    /// `cosines[i * m + j]` is the cosine similarity between sensitivity vectors
    /// of objectives i and j.  Values near +1 indicate redundancy (proportional
    /// sensitivities), near -1 indicate opposition, near 0 indicate independence.
    pub cosines: Vec<f64>,
    /// The m x m Gram matrix G = S * S^T (row-major).
    ///
    /// G_{ij} = <s_i, s_j> = sum_k s_i(k) * s_j(k).
    /// Exposed for debugging and for computing derived quantities like the
    /// trace (sum of diagonal = sum of eigenvalues = total sensitivity variance).
    pub gram: Vec<f64>,
}

impl RedundancyAnalysis {
    /// Effective objective dimension: number of eigenvalues above `eps * max_eigenvalue`.
    ///
    /// This counts how many independent tradeoff axes carry at least `eps` fraction of
    /// the dominant eigenvalue's magnitude.  With `eps = 0.01`, eigenvalues below 1% of
    /// the maximum are treated as negligible.
    ///
    /// Interpretation: if this returns k, then the Pareto front is *approximately*
    /// (k-1)-dimensional, and the remaining objectives create only tiny perturbations.
    pub fn effective_dimension(&self, eps: f64) -> usize {
        let max_ev = self.eigenvalues.first().copied().unwrap_or(0.0);
        if max_ev <= 0.0 {
            return 0;
        }
        let threshold = eps * max_ev;
        self.eigenvalues.iter().filter(|&&ev| ev > threshold).count()
    }

    /// Pareto front dimension upper bound: `min(rank - 1, m - 1)`, where rank
    /// is the number of eigenvalues above a numerical tolerance.
    ///
    /// This is a hard algebraic bound.  If rank = 1, all objectives are proportional
    /// and the Pareto front is a point (dimension 0).  If rank = m, all objectives
    /// are linearly independent and the front is (m-1)-dimensional.
    pub fn pareto_dimension_bound(&self, tol: f64) -> usize {
        let rank = self.eigenvalues.iter().filter(|&&ev| ev > tol).count();
        if rank == 0 {
            return 0;
        }
        (rank - 1).min(self.num_objectives.saturating_sub(1))
    }

    /// Fraction of total sensitivity variance captured by the top `k` eigenvalues.
    ///
    /// This is the analogue of "explained variance ratio" in PCA.  If the top 2
    /// eigenvalues capture 99% of the variance, then the objective-space geometry
    /// is essentially 2-dimensional regardless of how many objectives were named.
    pub fn variance_fraction(&self, k: usize) -> f64 {
        let total: f64 = self.eigenvalues.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let top_k: f64 = self.eigenvalues.iter().take(k).sum();
        top_k / total
    }

    /// Cosine similarity between objectives `i` and `j`.
    ///
    /// Returns `None` if indices are out of bounds.
    pub fn cosine(&self, i: usize, j: usize) -> Option<f64> {
        let m = self.num_objectives;
        if i >= m || j >= m {
            return None;
        }
        Some(self.cosines[i * m + j])
    }

    /// Trace of the Gram matrix (sum of diagonal entries).
    ///
    /// This equals the sum of eigenvalues and the sum of squared norms of the
    /// sensitivity vectors.  It measures the total "amount of sensitivity" across
    /// all objectives and design points.
    pub fn trace(&self) -> f64 {
        let m = self.num_objectives;
        (0..m).map(|i| self.gram[i * m + i]).sum()
    }

    /// Identify pairs of objectives that are nearly redundant (|cosine| > threshold).
    ///
    /// Returns `(i, j, cosine)` triples for all pairs with `|cos(s_i, s_j)| > threshold`.
    /// Positive cosine means proportional (move together); negative means anti-proportional
    /// (one improves when the other worsens, but they are still "coupled" -- optimizing
    /// one determines the other).
    pub fn redundant_pairs(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        let m = self.num_objectives;
        let mut pairs = Vec::new();
        for i in 0..m {
            for j in (i + 1)..m {
                let c = self.cosines[i * m + j];
                if c.abs() > threshold {
                    pairs.push((i, j, c));
                }
            }
        }
        pairs
    }
}

/// Analyze objective redundancy from a sensitivity matrix.
///
/// # Arguments
///
/// `sensitivities` is a slice of `m` rows, each of length `n` (design points).
/// Row `i` contains the sensitivity `s_i(j)` of objective `i` at design point `j`.
///
/// # Returns
///
/// `None` if the input is empty, any row is empty, or rows have inconsistent lengths.
/// Otherwise returns a [`RedundancyAnalysis`] with eigenvalues, cosines, and the Gram matrix.
///
/// # How to build the sensitivity matrix
///
/// For analytic objectives (closed-form derivatives), compute `s_i(j) = dL_i / d mu(j)`
/// directly.  For black-box objectives, use [`finite_difference_jacobian`] to estimate
/// the Jacobian numerically.
///
/// # Example
///
/// ```
/// use pare::sensitivity::analyze_redundancy;
///
/// // Two objectives over 3 design points.
/// // Objective 0: some sensitivity profile [1, 2, 3].
/// // Objective 1: exactly 2x objective 0 => redundant (cosine = 1.0).
/// let sensitivities = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![2.0, 4.0, 6.0],
/// ];
///
/// let analysis = analyze_redundancy(&sensitivities).unwrap();
///
/// // Only one independent direction (rank 1), so Pareto dimension = 0.
/// assert_eq!(analysis.effective_dimension(0.01), 1);
/// assert!((analysis.cosine(0, 1).unwrap() - 1.0).abs() < 1e-9);
///
/// // Trace of Gram matrix equals sum of eigenvalues.
/// let ev_sum: f64 = analysis.eigenvalues.iter().sum();
/// assert!((analysis.trace() - ev_sum).abs() < 1e-9);
/// ```
pub fn analyze_redundancy(sensitivities: &[SensitivityRow]) -> Option<RedundancyAnalysis> {
    if sensitivities.is_empty() {
        return None;
    }
    let m = sensitivities.len();
    let n = sensitivities[0].len();
    if n == 0 || sensitivities.iter().any(|r| r.len() != n) {
        return None;
    }

    // Compute Gram matrix G[i][j] = <s_i, s_j> = sum_k s_i(k) * s_j(k).
    // G is m x m, symmetric, and positive semidefinite (it's S * S^T).
    let mut gram = vec![0.0; m * m];
    for i in 0..m {
        for j in i..m {
            let dot: f64 = sensitivities[i]
                .iter()
                .zip(sensitivities[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            gram[i * m + j] = dot;
            gram[j * m + i] = dot;
        }
    }

    // Compute norms for cosine similarity.
    // norm_i = sqrt(G[i][i]) = ||s_i||_2.
    let norms: Vec<f64> = (0..m).map(|i| gram[i * m + i].sqrt()).collect();
    let mut cosines = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            if norms[i] > 0.0 && norms[j] > 0.0 {
                cosines[i * m + j] = (gram[i * m + j] / (norms[i] * norms[j])).clamp(-1.0, 1.0);
            } else if i == j {
                // Zero vector has cosine 1.0 with itself by convention.
                cosines[i * m + j] = 1.0;
            }
            // else: cosine with a zero vector is 0.0 (default).
        }
    }

    // Eigenvalue decomposition of the symmetric PSD Gram matrix via Jacobi iteration.
    // For the m x m Gram matrix (m = number of objectives, typically < 20),
    // Jacobi is perfectly adequate and avoids a LAPACK/nalgebra dependency.
    let eigenvalues = symmetric_eigenvalues(&gram, m);

    Some(RedundancyAnalysis {
        num_objectives: m,
        num_design_points: n,
        eigenvalues,
        cosines,
        gram,
    })
}

/// Compute the Jacobian matrix (m x n) from a set of objective functions evaluated
/// at perturbations of a baseline design.
///
/// This is a finite-difference helper: given a baseline design `mu` (length n) and
/// objective functions, compute `J[i][j] = (L_i(mu + eps*e_j) - L_i(mu)) / eps`
/// using forward differences.
///
/// # Trade-offs
///
/// Forward differences are O(n * m) function evaluations.  Central differences
/// (`(f(mu+eps) - f(mu-eps)) / (2*eps)`) are more accurate but cost 2x.  For
/// analytic objectives with known derivatives, skip this function entirely and
/// pass the derivatives directly to [`analyze_redundancy`].
///
/// # Arguments
///
/// - `mu`: baseline design vector (length n).
/// - `objectives`: slice of closures, each mapping a design vector to a scalar.
/// - `eps`: perturbation step size.  Smaller is more accurate for smooth objectives
///   but more susceptible to floating-point cancellation.  1e-7 is a common default.
pub fn finite_difference_jacobian<F>(
    mu: &[f64],
    objectives: &[F],
    eps: f64,
) -> Vec<SensitivityRow>
where
    F: Fn(&[f64]) -> f64,
{
    let n = mu.len();
    let m = objectives.len();
    let base_values: Vec<f64> = objectives.iter().map(|f| f(mu)).collect();

    let mut jacobian = vec![vec![0.0; n]; m];
    let mut perturbed = mu.to_vec();

    for j in 0..n {
        let old = perturbed[j];
        perturbed[j] = old + eps;
        for (i, f) in objectives.iter().enumerate() {
            jacobian[i][j] = (f(&perturbed) - base_values[i]) / eps;
        }
        perturbed[j] = old;
    }

    jacobian
}

// ============================================================================
// Jacobi eigenvalue algorithm for small symmetric matrices
// ============================================================================

/// Compute eigenvalues of a symmetric matrix (given as row-major flat array of size n*n).
/// Returns eigenvalues sorted in descending order.
///
/// Uses the classical Jacobi rotation method: repeatedly find the largest off-diagonal
/// element and zero it out with a Givens rotation.  Each rotation is O(n) work and
/// reduces the off-diagonal Frobenius norm.  Convergence is guaranteed for symmetric
/// matrices; the method is unconditionally stable.
///
/// For the small matrices we encounter (m x m where m = number of objectives, typically
/// 2-20), this is fast enough and avoids adding a linear algebra dependency to `pare`.
///
/// The final eigenvalues are clamped to >= 0 because the input is a Gram matrix (PSD);
/// any negative values are numerical noise from the iteration.
fn symmetric_eigenvalues(a: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * n);
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a[0].max(0.0)];
    }

    // Work on a mutable copy.  The Jacobi iteration progressively zeroes
    // off-diagonal elements until the matrix is (nearly) diagonal.
    let mut mat = a.to_vec();

    // Convergence is typically cubic for symmetric matrices.
    // 100 * n^2 sweeps is very conservative for n < 50.
    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        // Find the largest off-diagonal element |mat[p][q]|.
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = mat[i * n + j].abs();
                if v > max_off {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < tol {
            break; // Off-diagonal is negligible; we're done.
        }

        // Compute the Givens rotation angle that zeroes mat[p][q].
        //
        // The 2x2 subproblem is:
        //   [[app, apq], [apq, aqq]]
        // Rotation by theta diagonalizes it when:
        //   tan(2*theta) = 2*apq / (app - aqq)
        let app = mat[p * n + p];
        let aqq = mat[q * n + q];
        let apq = mat[p * n + q];

        let theta = if (app - aqq).abs() < tol {
            // Diagonal elements are equal; optimal rotation is pi/4.
            core::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply the Givens rotation to rows/cols p and q.
        // This zeroes mat[p][q] and updates the rest of the matrix.
        //
        // We clone and update rather than updating in-place to keep the logic
        // clear.  For m < 50 (typical), the O(m^2) per-iteration cost is fine.
        let mut new_mat = mat.clone();
        for i in 0..n {
            if i != p && i != q {
                let ip = mat[i * n + p];
                let iq = mat[i * n + q];
                new_mat[i * n + p] = c * ip + s * iq;
                new_mat[p * n + i] = new_mat[i * n + p]; // symmetry
                new_mat[i * n + q] = -s * ip + c * iq;
                new_mat[q * n + i] = new_mat[i * n + q]; // symmetry
            }
        }
        new_mat[p * n + p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_mat[q * n + q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_mat[p * n + q] = 0.0;
        new_mat[q * n + p] = 0.0;

        mat = new_mat;
    }

    // Diagonal entries are the eigenvalues.
    // Clamp to >= 0: the input is PSD (Gram matrix), so true eigenvalues are
    // non-negative.  Any negative values are floating-point noise.
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| mat[i * n + i].max(0.0)).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // Basic algebraic properties
    // ====================================================================

    #[test]
    fn proportional_sensitivities_have_rank_one() {
        // If s_1 = alpha * s_0, the Gram matrix has rank 1.
        // Pareto dimension = rank - 1 = 0 (no tradeoff: optimizing one objective
        // automatically optimizes the other).
        let s = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![3.0, 6.0, 9.0, 12.0], // 3x row 0
        ];
        let a = analyze_redundancy(&s).unwrap();
        assert_eq!(a.effective_dimension(0.01), 1);
        assert_eq!(a.pareto_dimension_bound(1e-6), 0);
        assert!((a.cosine(0, 1).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn anti_proportional_sensitivities_have_rank_one() {
        // s_1 = -2 * s_0.  Cosine = -1.0 (perfectly opposed).
        // Rank is still 1: the objectives are coupled (one is a deterministic
        // function of the other), just in opposite directions.
        // Pareto dimension = 0.
        let s = vec![
            vec![1.0, 2.0, 3.0],
            vec![-2.0, -4.0, -6.0],
        ];
        let a = analyze_redundancy(&s).unwrap();
        assert_eq!(a.pareto_dimension_bound(1e-6), 0);
        let cos = a.cosine(0, 1).unwrap();
        assert!(
            (cos - (-1.0)).abs() < 1e-9,
            "expected cosine = -1.0, got {cos}"
        );
    }

    #[test]
    fn orthogonal_sensitivities_have_full_rank() {
        // s_0 and s_1 are orthogonal unit vectors.
        // Rank = 2, Pareto dimension = 1 (a genuine 1D tradeoff curve).
        let s = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let a = analyze_redundancy(&s).unwrap();
        assert_eq!(a.effective_dimension(0.01), 2);
        assert_eq!(a.pareto_dimension_bound(1e-6), 1);
        assert!(a.cosine(0, 1).unwrap().abs() < 1e-9);
    }

    #[test]
    fn three_independent_objectives() {
        // Three nearly-orthogonal sensitivity vectors in R^4.
        // The slight overlap (0.5, 0.3, 0.1) means they aren't perfectly orthogonal
        // but are still linearly independent => rank 3, Pareto dimension 2.
        let s = vec![
            vec![1.0, 0.0, 0.0, 0.5],
            vec![0.0, 1.0, 0.0, 0.3],
            vec![0.0, 0.0, 1.0, 0.1],
        ];
        let a = analyze_redundancy(&s).unwrap();
        assert_eq!(a.effective_dimension(0.01), 3);
        assert_eq!(a.pareto_dimension_bound(1e-6), 2);
    }

    // ====================================================================
    // Eigenvalue invariants
    // ====================================================================

    #[test]
    fn eigenvalue_sum_equals_gram_trace() {
        // Fundamental identity: sum of eigenvalues = trace of G = sum of ||s_i||^2.
        // This must hold for any input.
        let s = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let a = analyze_redundancy(&s).unwrap();
        let ev_sum: f64 = a.eigenvalues.iter().sum();
        let trace = a.trace();
        assert!(
            (ev_sum - trace).abs() < 1e-6,
            "eigenvalue sum = {ev_sum}, trace = {trace}"
        );
        // Also verify trace = sum of squared norms.
        let norm_sq_sum: f64 = s.iter().map(|row| row.iter().map(|&x| x * x).sum::<f64>()).sum();
        assert!(
            (trace - norm_sq_sum).abs() < 1e-9,
            "trace = {trace}, norm_sq_sum = {norm_sq_sum}"
        );
    }

    #[test]
    fn permuting_rows_preserves_eigenvalues() {
        // Reordering objectives should not change the eigenvalue spectrum
        // (it permutes rows/cols of G but eigenvalues are permutation-invariant).
        let s_orig = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 5.0, 0.0],
            vec![4.0, 0.0, 1.0],
        ];
        let s_perm = vec![
            s_orig[2].clone(),
            s_orig[0].clone(),
            s_orig[1].clone(),
        ];
        let a1 = analyze_redundancy(&s_orig).unwrap();
        let a2 = analyze_redundancy(&s_perm).unwrap();
        for (e1, e2) in a1.eigenvalues.iter().zip(a2.eigenvalues.iter()) {
            assert!(
                (e1 - e2).abs() < 1e-9,
                "eigenvalues differ after permutation: {e1} vs {e2}"
            );
        }
    }

    // ====================================================================
    // The non-contextual collapse (bandit theory)
    // ====================================================================

    #[test]
    fn non_contextual_collapse_two_arms() {
        // Setup: K=2 Gaussian arms, gap Delta=0.5, change magnitude delta=0.3.
        // The only design variable is n_2 (pulls of the suboptimal arm).
        //
        // Regret:    R = Delta * n_2         =>  dR/d(n_2) = Delta = 0.5
        // MSE:       MSE = sigma^2 / n_2     =>  dMSE/d(n_2) = -sigma^2 / n_2^2
        // Detection: D = C*T / (n_2*delta^2) =>  dD/d(n_2) = -C*T / (n_2^2 * delta^2)
        //
        // Key observation: dD/d(n_2) = (C*T/delta^2) * dMSE/d(n_2).
        // MSE and Detection have PROPORTIONAL sensitivity (cosine = 1.0).
        // This is the "non-contextual collapse": three objectives, but only two
        // independent directions (regret vs. the MSE/detection axis).
        //
        // Since all three are scalars (1D vectors), the Gram matrix has rank 1.
        // In 1D, all vectors are proportional up to sign.
        let delta = 0.5_f64;
        let delta_det = 0.3_f64;
        let n = 50.0_f64;
        let t = 1000.0_f64;
        let c = 1.0_f64;

        let s_r = delta;
        let s_mse = -1.0 / (n * n);
        let s_d = -c * t / (n * n * delta_det * delta_det);

        let sensitivities = vec![vec![s_r], vec![s_mse], vec![s_d]];
        let a = analyze_redundancy(&sensitivities).unwrap();

        // MSE and detection are proportional (both negative, same shape).
        let cos_mse_d = a.cosine(1, 2).unwrap();
        assert!(
            (cos_mse_d - 1.0).abs() < 1e-9,
            "MSE and Detection should be perfectly correlated, got {cos_mse_d}"
        );
        // Regret is anti-proportional to both (positive vs. negative).
        let cos_r_mse = a.cosine(0, 1).unwrap();
        assert!(
            (cos_r_mse - (-1.0)).abs() < 1e-9,
            "Regret and MSE should be anti-proportional, got {cos_r_mse}"
        );

        // All 1D vectors => Gram rank = 1, Pareto dimension = 0.
        assert_eq!(a.pareto_dimension_bound(1e-6), 0);
    }

    // ====================================================================
    // The contextual revival (bandit theory)
    // ====================================================================

    #[test]
    fn contextual_revival_three_objectives() {
        // Setup: 2 arms on [0,1] with f_1(x) = x, f_2(x) = 1-x.
        // Decision boundary at x = 0.5.
        //
        // Now the design has 5 degrees of freedom (allocation probability at
        // each of 5 covariate cells), and objectives have different shapes:
        //
        // - Regret sensitivity: gap(x) = |2x-1|.  Concentrated NEAR the boundary
        //   (x=0.5), because that's where misrouting costs the most.
        //
        // - D-optimal estimation: leverage = x^2 + (1-x)^2.  Concentrated at the
        //   EXTREMES (x=0 and x=1), because those points are most informative for
        //   estimating a linear model.
        //
        // - Worst-case detection: point mass at the bottleneck cell (the cell with
        //   fewest observations).  Here we put it at x=0.1 to model a scenario
        //   where the allocation is skewed away from low-x regions.
        //
        // These three shapes are linearly independent => rank 3, Pareto dimension 2.
        // This is the "contextual revival": the multi-objective problem is real.
        let cells: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];

        let s_regret: Vec<f64> = cells.iter().map(|&x| (2.0 * x - 1.0).abs()).collect();
        let s_estimation: Vec<f64> = cells
            .iter()
            .map(|&x| -(x * x + (1.0 - x) * (1.0 - x)))
            .collect();
        let mut s_detection = vec![0.0; 5];
        s_detection[0] = -1.0; // bottleneck at cell 0

        let sensitivities = vec![s_regret, s_estimation, s_detection];
        let a = analyze_redundancy(&sensitivities).unwrap();

        assert_eq!(a.effective_dimension(0.01), 3);
        assert_eq!(a.pareto_dimension_bound(1e-6), 2);

        // Regret and estimation have different shapes (not highly correlated).
        let cos_re = a.cosine(0, 1).unwrap();
        assert!(cos_re.abs() < 0.95, "regret-estimation cosine = {cos_re}");
    }

    // ====================================================================
    // Average-case detection is redundant even in contextual settings
    // ====================================================================

    #[test]
    fn average_detection_is_redundant_with_estimation_even_contextual() {
        // The key finding from the research manifesto, Section IV/VIII:
        //
        // Average detection delay and nonparametric IMSE both have sensitivity
        // proportional to -1/p(x)^2 at each design point.  This holds for ANY
        // baseline allocation p(x), not just uniform.  The proportionality constant
        // differs (it depends on change magnitude, false alarm rate, etc.) but the
        // SHAPE is identical.
        //
        // This means: if you already have "estimation quality" as an objective,
        // adding "average detection delay" creates ZERO new tradeoff.  The cosine
        // is exactly 1.0, the Gram matrix rank does not increase, and the Pareto
        // front dimension stays the same.
        //
        // To get a genuinely independent detection objective, you need WORST-CASE
        // detection (which concentrates sensitivity on the bottleneck cell).
        let p2: [f64; 5] = [0.1, 0.2, 0.4, 0.2, 0.1]; // non-uniform allocation

        // IMSE sensitivity: s(x) = -1/p(x)^2
        let s_imse: Vec<f64> = p2.iter().map(|&p| -1.0 / (p * p)).collect();

        // Average detection sensitivity: s(x) = -C/p(x)^2  (same shape, different constant)
        let c: f64 = 3.5;
        let s_d_avg: Vec<f64> = p2.iter().map(|&p| -c / (p * p)).collect();

        // Regret sensitivity: different shape.
        let s_regret: Vec<f64> = p2.iter().map(|&p| 10.0 * p).collect();

        let sensitivities = vec![s_regret, s_imse, s_d_avg];
        let a = analyze_redundancy(&sensitivities).unwrap();

        // IMSE and average detection: perfectly correlated.
        let cos_imse_davg = a.cosine(1, 2).unwrap();
        assert!(
            (cos_imse_davg - 1.0).abs() < 1e-9,
            "IMSE and avg detection should be perfectly correlated, got {cos_imse_davg}"
        );

        // Three objectives but only 2 linearly independent directions.
        assert_eq!(a.pareto_dimension_bound(1e-6), 1);

        // The redundant pair is (1, 2) = (IMSE, avg detection).
        let pairs = a.redundant_pairs(0.999);
        assert_eq!(pairs.len(), 1);
        assert_eq!((pairs[0].0, pairs[0].1), (1, 2));
    }

    // ====================================================================
    // RedundancyAnalysis API tests
    // ====================================================================

    #[test]
    fn variance_fraction_sums_correctly() {
        // Two objectives with very different magnitudes.
        // The larger one dominates the variance.
        let s = vec![
            vec![1.0, 0.0],   // ||s_0|| = 1
            vec![0.0, 0.01],  // ||s_1|| = 0.01
        ];
        let a = analyze_redundancy(&s).unwrap();

        // Top eigenvalue captures almost all variance.
        let frac = a.variance_fraction(1);
        assert!(frac > 0.99, "top eigenvalue should dominate, got {frac}");

        // Both eigenvalues capture everything.
        let frac_all = a.variance_fraction(2);
        assert!((frac_all - 1.0).abs() < 1e-9);
    }

    #[test]
    fn redundant_pairs_identifies_proportional_objectives() {
        let s = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0], // proportional to 0
            vec![0.0, 0.0, 1.0], // independent
        ];
        let a = analyze_redundancy(&s).unwrap();
        let pairs = a.redundant_pairs(0.99);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, 0);
        assert_eq!(pairs[0].1, 1);
    }

    #[test]
    fn trace_matches_gram_diagonal() {
        let s = vec![
            vec![3.0, 4.0], // ||s||^2 = 9 + 16 = 25
            vec![1.0, 0.0], // ||s||^2 = 1
        ];
        let a = analyze_redundancy(&s).unwrap();
        assert!((a.trace() - 26.0).abs() < 1e-9, "trace = {}", a.trace());
    }

    // ====================================================================
    // Edge cases
    // ====================================================================

    #[test]
    fn empty_input_returns_none() {
        assert!(analyze_redundancy(&[]).is_none());
        assert!(analyze_redundancy(&[vec![]]).is_none());
    }

    #[test]
    fn mismatched_row_lengths_returns_none() {
        let s = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(analyze_redundancy(&s).is_none());
    }

    #[test]
    fn single_objective_single_point() {
        // One objective, one design point.  Rank = 1, Pareto dim = 0.
        let s = vec![vec![5.0]];
        let a = analyze_redundancy(&s).unwrap();
        assert_eq!(a.effective_dimension(0.01), 1);
        assert_eq!(a.pareto_dimension_bound(1e-6), 0);
    }

    #[test]
    fn zero_sensitivity_vector() {
        // An objective with zero sensitivity everywhere is uninformative.
        // It shouldn't increase the effective dimension.
        let s = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 0.0, 0.0], // zero vector
        ];
        let a = analyze_redundancy(&s).unwrap();
        assert_eq!(a.effective_dimension(0.01), 1);
    }

    // ====================================================================
    // Jacobi eigenvalue solver tests
    // ====================================================================

    #[test]
    fn jacobi_eigenvalues_identity() {
        let id = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let ev = symmetric_eigenvalues(&id, 3);
        assert_eq!(ev.len(), 3);
        for &e in &ev {
            assert!((e - 1.0).abs() < 1e-9, "eigenvalue = {e}");
        }
    }

    #[test]
    fn jacobi_eigenvalues_known_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let ev = symmetric_eigenvalues(&a, 2);
        assert!((ev[0] - 3.0).abs() < 1e-9, "ev[0] = {}", ev[0]);
        assert!((ev[1] - 1.0).abs() < 1e-9, "ev[1] = {}", ev[1]);
    }

    #[test]
    fn jacobi_eigenvalues_5x5_trace_preserved() {
        // A 5x5 PSD matrix.  Verify eigenvalue sum = trace.
        let mut mat = vec![0.0; 25];
        // Build S * S^T from random-ish rows.
        let rows: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![0.5, 1.5, 2.5],
            vec![3.0, 1.0, 4.0],
        ];
        for i in 0..5 {
            for j in 0..5 {
                let dot: f64 = (0..3).map(|k| rows[i][k] * rows[j][k]).sum();
                mat[i * 5 + j] = dot;
            }
        }
        let trace: f64 = (0..5).map(|i| mat[i * 5 + i]).sum();
        let ev = symmetric_eigenvalues(&mat, 5);
        let ev_sum: f64 = ev.iter().sum();
        assert!(
            (ev_sum - trace).abs() < 1e-6,
            "eigenvalue sum = {ev_sum}, trace = {trace}"
        );
    }

    // ====================================================================
    // Finite-difference Jacobian tests
    // ====================================================================

    #[test]
    fn finite_difference_jacobian_linear() {
        // Linear functions have exact Jacobians regardless of eps.
        // f0(x) = x[0] + 2*x[1], f1(x) = 3*x[0] - x[1]
        // Jacobian: [[1, 2], [3, -1]]
        let f0 = |x: &[f64]| x[0] + 2.0 * x[1];
        let f1 = |x: &[f64]| 3.0 * x[0] - x[1];
        let objectives = [f0, f1];
        let mu = vec![1.0, 1.0];
        let jac = finite_difference_jacobian(&mu, &objectives, 1e-7);
        assert!((jac[0][0] - 1.0).abs() < 1e-4);
        assert!((jac[0][1] - 2.0).abs() < 1e-4);
        assert!((jac[1][0] - 3.0).abs() < 1e-4);
        assert!((jac[1][1] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn finite_difference_jacobian_quadratic() {
        // f(x) = x[0]^2 + x[1]^2.  Jacobian at (3, 4): [6, 8].
        let f0 = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let objectives = [f0];
        let mu = vec![3.0, 4.0];
        let jac = finite_difference_jacobian(&mu, &objectives, 1e-7);
        assert!((jac[0][0] - 6.0).abs() < 1e-3, "J[0][0] = {}", jac[0][0]);
        assert!((jac[0][1] - 8.0).abs() < 1e-3, "J[0][1] = {}", jac[0][1]);
    }
}

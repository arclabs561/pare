//! Pareto frontier and skyline query primitives for multi-objective optimization.
//!
//! Maintains a set of non-dominated points across multiple dimensions,
//! supporting maximization/minimization per dimension and crowding distance
//! for diversity maintenance.

use std::cmp::Ordering;

/// Direction of optimization for a dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Direction {
    /// Higher values are better.
    Maximize,
    /// Lower values are better.
    Minimize,
}

/// A point in multi-objective space.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point<V> {
    /// The actual values across all objectives.
    pub values: Vec<f64>,
    /// Associated data (e.g. an ID or candidate metadata).
    pub data: V,
}

/// Statistics for normalizing objective values.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NormalizationStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl NormalizationStats {
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let std = var.sqrt();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self {
            mean,
            std,
            min,
            max,
        }
    }
}

/// Pareto frontier maintaining a set of non-dominated points.
#[derive(Debug, Clone)]
pub struct ParetoFrontier<V> {
    points: Vec<Point<V>>,
    directions: Vec<Direction>,
    stats: Vec<NormalizationStats>,
    labels: Vec<String>,
    eps: f64,
}

impl<V> ParetoFrontier<V> {
    /// Create a new frontier with specified directions for each objective.
    pub fn new(directions: Vec<Direction>) -> Self {
        let dim = directions.len();
        Self {
            points: Vec::new(),
            directions,
            stats: vec![NormalizationStats::default(); dim],
            labels: (0..dim).map(|i| i.to_string()).collect(),
            eps: 1e-9,
        }
    }

    /// Set labels for objectives (e.g. ["accuracy", "latency"]).
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        debug_assert_eq!(labels.len(), self.directions.len());
        self.labels = labels;
        self
    }

    /// Set the epsilon for dominance comparisons.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Number of points currently on the frontier.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Is the frontier empty?
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get all points on the frontier.
    pub fn points(&self) -> &[Point<V>] {
        &self.points
    }

    /// Mutable access to points (use with caution).
    pub fn points_mut(&mut self) -> &mut Vec<Point<V>> {
        &mut self.points
    }

    /// Add a point to the frontier if it is non-dominated.
    pub fn push(&mut self, values: Vec<f64>, data: V) -> bool {
        debug_assert_eq!(values.len(), self.directions.len());

        // Check if any existing point dominates the new point.
        for existing in &self.points {
            if dominates(&self.directions, self.eps, &existing.values, &values) {
                return false;
            }
        }

        // Remove points dominated by the new point.
        let directions = &self.directions;
        let eps = self.eps;
        self.points
            .retain(|existing| !dominates(directions, eps, &values, &existing.values));

        self.points.push(Point { values, data });
        self.update_stats();
        true
    }

    fn update_stats(&mut self) {
        for i in 0..self.directions.len() {
            let dim_values: Vec<f64> = self.points.iter().map(|p| p.values[i]).collect();
            self.stats[i] = NormalizationStats::from_values(&dim_values);
        }
    }

    /// Returns the normalized score for a point across all dimensions (higher is better).
    pub fn scalar_score(&self, point_idx: usize, weights: &[f64]) -> f64 {
        let p = &self.points[point_idx];
        let mut score = 0.0;
        let mut w_sum = 0.0;

        for (i, &w) in weights.iter().enumerate() {
            let val = p.values[i];
            let stat = self.stats[i];
            let range = stat.max - stat.min;
            let norm = if range > self.eps {
                (val - stat.min) / range
            } else {
                0.5
            };

            let oriented = match self.directions[i] {
                Direction::Maximize => norm,
                Direction::Minimize => 1.0 - norm,
            };

            score += oriented * w;
            w_sum += w;
        }

        if w_sum > 0.0 {
            score / w_sum
        } else {
            0.0
        }
    }

    /// Find the index of the point with the highest weighted scalar score.
    pub fn best_index(&self, weights: &[f64]) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        (0..self.points.len()).max_by(|&a, &b| {
            self.scalar_score(a, weights)
                .partial_cmp(&self.scalar_score(b, weights))
                .unwrap_or(Ordering::Equal)
        })
    }

    /// Check if point A dominates point B.
    pub fn point_dominates(&self, a: &[f64], b: &[f64]) -> bool {
        dominates(&self.directions, self.eps, a, b)
    }

    /// Compute crowding distances for all points on the frontier (NSGA-II style).
    pub fn crowding_distances(&self) -> Vec<f64> {
        let n = self.points.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![f64::INFINITY];
        }

        let mut distances = vec![0.0; n];
        let mut indices: Vec<usize> = (0..n).collect();

        for i in 0..self.directions.len() {
            indices.sort_by(|&a, &b| {
                let av = self.points[a].values[i];
                let bv = self.points[b].values[i];
                av.partial_cmp(&bv).unwrap_or(Ordering::Equal)
            });

            distances[indices[0]] = f64::INFINITY;
            distances[indices[n - 1]] = f64::INFINITY;

            let min_val = self.points[indices[0]].values[i];
            let max_val = self.points[indices[n - 1]].values[i];
            let range = max_val - min_val;

            if range > self.eps {
                for window in indices.windows(3) {
                    let prev = window[0];
                    let curr = window[1];
                    let next = window[2];
                    if distances[curr].is_infinite() {
                        continue;
                    }
                    let val_prev = self.points[prev].values[i];
                    let val_next = self.points[next].values[i];
                    distances[curr] += (val_next - val_prev) / range;
                }
            }
        }
        distances
    }

    /// Calculate the hypervolume of the frontier relative to a reference point.
    pub fn hypervolume(&self, ref_point: &[f64]) -> f64
    {
        let dim = self.directions.len();
        if dim == 0 || self.is_empty() {
            return 0.0;
        }
        debug_assert_eq!(ref_point.len(), dim);

        // Convert to an oriented, all-maximize coordinate system rooted at `ref_point`.
        // For each dimension, we measure "improvement over reference", clamped at 0.
        //
        // This lets us compute hypervolume against the origin in a consistent way:
        // - Maximize:  oriented = value - ref
        // - Minimize:  oriented = ref - value
        //
        // Hypervolume is then the Lebesgue measure of the union of axis-aligned boxes
        // [0, p] for each point p in oriented space.
        let mut oriented: Vec<Vec<f64>> = self
            .points
            .iter()
            .map(|p| {
                p.values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| match self.directions[i] {
                        Direction::Maximize => (v - ref_point[i]).max(0.0),
                        Direction::Minimize => (ref_point[i] - v).max(0.0),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Remove points that contribute no volume (any dim <= 0 means their box has 0 volume).
        oriented.retain(|p| p.iter().all(|&x| x > self.eps));
        if oriented.is_empty() {
            return 0.0;
        }

        // In oriented space, dominance is pure-maximize.
        let oriented = nondominated_max(&oriented, self.eps);
        hypervolume_max_exact(&oriented, dim, self.eps)
    }
}

/// Return a non-dominated subset in maximize space (all objectives are "higher is better").
fn nondominated_max(points: &[Vec<f64>], eps: f64) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    'outer: for p in points {
        // If an existing point dominates p, drop p.
        for q in &out {
            if dominates_max(q, p, eps) {
                continue 'outer;
            }
        }
        // Otherwise, remove points dominated by p and insert p.
        out.retain(|q| !dominates_max(p, q, eps));
        out.push(p.clone());
    }
    out
}

fn dominates_max(a: &[f64], b: &[f64], eps: f64) -> bool {
    let mut strictly_better = false;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        if av + eps < bv {
            return false;
        }
        if av > bv + eps {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Exact hypervolume in maximize space, reference at the origin.
///
/// This uses recursive slicing on the last dimension:
/// \[
/// HV_d(P) = \sum_{k} (y_k - y_{k+1}) \cdot HV_{d-1}(\pi(P_{y \ge y_k}))
/// \]
/// where \(y_k\) are the unique last-coordinate levels (descending), and \(\pi\) drops the last coordinate.
fn hypervolume_max_exact(points: &[Vec<f64>], dim: usize, eps: f64) -> f64 {
    debug_assert!(dim >= 1);
    if points.is_empty() {
        return 0.0;
    }
    if dim == 1 {
        return points
            .iter()
            .map(|p| p[0])
            .fold(0.0, f64::max);
    }
    if dim == 2 {
        return hypervolume_max_2d(points, eps);
    }

    // Collect unique slice levels (positive only).
    let mut levels: Vec<f64> = points.iter().map(|p| p[dim - 1]).filter(|&v| v > eps).collect();
    levels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal)); // descending
    levels.dedup_by(|a, b| (*a - *b).abs() <= eps);

    let mut hv = 0.0;
    for (idx, &level) in levels.iter().enumerate() {
        let next = if idx + 1 < levels.len() { levels[idx + 1] } else { 0.0 };
        let thickness = (level - next).max(0.0);
        if thickness <= eps {
            continue;
        }

        let mut projected: Vec<Vec<f64>> = points
            .iter()
            .filter(|p| p[dim - 1] + eps >= level)
            .map(|p| p[0..dim - 1].to_vec())
            .collect();
        if projected.is_empty() {
            continue;
        }
        projected = nondominated_max(&projected, eps);
        let cross = hypervolume_max_exact(&projected, dim - 1, eps);
        hv += thickness * cross;
    }
    hv
}

fn hypervolume_max_2d(points: &[Vec<f64>], eps: f64) -> f64 {
    // Sort by x ascending. For a non-dominated set, y should be non-increasing with x.
    let mut idxs: Vec<usize> = (0..points.len()).collect();
    idxs.sort_by(|&i, &j| {
        let xi = points[i][0];
        let xj = points[j][0];
        match xi.partial_cmp(&xj).unwrap_or(Ordering::Equal) {
            Ordering::Equal => points[j][1]
                .partial_cmp(&points[i][1])
                .unwrap_or(Ordering::Equal),
            ord => ord,
        }
    });

    let mut area = 0.0;
    let mut prev_x = 0.0;
    for i in idxs {
        let x = points[i][0].max(0.0);
        let y = points[i][1].max(0.0);
        let dx = (x - prev_x).max(0.0);
        if dx > eps && y > eps {
            area += dx * y;
        }
        prev_x = x;
    }
    area
}

/// Standalone dominance check.
pub fn dominates(directions: &[Direction], eps: f64, a: &[f64], b: &[f64]) -> bool {
    let mut strictly_better = false;
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let dir = directions[i];
        match dir {
            Direction::Maximize => {
                if av + eps < bv {
                    return false;
                }
                if av > bv + eps {
                    strictly_better = true;
                }
            }
            Direction::Minimize => {
                if av > bv + eps {
                    return false;
                }
                if av + eps < bv {
                    strictly_better = true;
                }
            }
        }
    }
    strictly_better
}

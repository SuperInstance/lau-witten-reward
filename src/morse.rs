//! Morse theory primitives: critical points, Morse functions, and the Morse complex.

use nalgebra::{DMatrix, DVector, Complex};
use serde::{Serialize, Deserialize};
use std::fmt;

/// A non-degenerate critical point of a Morse function on an n-dimensional manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPoint {
    /// Coordinates in the ambient space.
    pub coords: Vec<f64>,
    /// Morse index (number of negative eigenvalues of the Hessian).
    pub index: usize,
    /// Function value at this critical point.
    pub value: f64,
    /// Optional label (e.g. "reward_minimum", "saddle_1").
    pub label: Option<String>,
}

impl CriticalPoint {
    /// Create a new critical point.
    pub fn new(coords: Vec<f64>, index: usize, value: f64) -> Self {
        Self { coords, index, value, label: None }
    }

    /// Create a labeled critical point.
    pub fn labeled(coords: Vec<f64>, index: usize, value: f64, label: impl Into<String>) -> Self {
        Self { coords, index, value, label: Some(label.into()) }
    }

    /// Dimension of the ambient space.
    pub fn ambient_dim(&self) -> usize {
        self.coords.len()
    }

    /// Whether this is a minimum (index 0).
    pub fn is_minimum(&self) -> bool {
        self.index == 0
    }

    /// Whether this is a maximum.
    pub fn is_maximum(&self) -> bool {
        self.index == self.coords.len()
    }

    /// Whether this is a saddle point.
    pub fn is_saddle(&self) -> bool {
        self.index > 0 && self.index < self.coords.len()
    }
}

impl fmt::Display for CriticalPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or("unnamed");
        write!(f, "CP[{}] idx={} val={:.4} @ {:?}",
               label, self.index, self.value, self.coords)
    }
}

/// A smooth Morse function f: M → ℝ on a discrete approximation of the state space.
///
/// The reward landscape IS a Morse function: it has isolated non-degenerate
/// critical points corresponding to reward basins, saddle points, and peaks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseFunction {
    /// Dimension of the domain.
    pub dim: usize,
    /// Known critical points.
    pub critical_points: Vec<CriticalPoint>,
    /// The function evaluated on a discrete grid (optional).
    pub grid_values: Option<DVector<f64>>,
    /// Grid spacing.
    pub grid_spacing: f64,
}

impl MorseFunction {
    /// Create a Morse function from known critical points.
    pub fn from_critical_points(dim: usize, critical_points: Vec<CriticalPoint>) -> Self {
        Self { dim, critical_points, grid_values: None, grid_spacing: 1.0 }
    }

    /// Create a Morse function from a grid of values, estimating critical points.
    pub fn from_grid(values: DVector<f64>, dim: usize, spacing: f64) -> Self {
        let critical_points = Self::find_critical_points(&values, dim, spacing);
        Self { dim, critical_points, grid_values: Some(values), grid_spacing: spacing }
    }

    /// Find critical points on a discrete grid by looking for local extrema and saddles.
    fn find_critical_points(values: &DVector<f64>, dim: usize, spacing: f64) -> Vec<CriticalPoint> {
        let n = values.len();
        if n < 3 || dim == 0 {
            return Vec::new();
        }

        let mut cps = Vec::new();

        // For 1D: find local extrema
        if dim == 1 {
            for i in 1..n - 1 {
                let prev = values[i - 1];
                let curr = values[i];
                let next = values[i + 1];
                if curr > prev && curr > next {
                    cps.push(CriticalPoint::labeled(
                        vec![i as f64 * spacing], 0, curr, "local_max"));
                } else if curr < prev && curr < next {
                    cps.push(CriticalPoint::labeled(
                        vec![i as f64 * spacing], 1, curr, "local_min"));
                }
            }
            return cps;
        }

        // For higher dimensions, find local extrema by comparing with all grid neighbors
        let side = (n as f64).powf(1.0 / dim as f64).round() as usize;
        if side < 3 {
            return cps;
        }

        // Simplified: check corners and center as approximate critical points
        // In a full implementation, we'd use gradient + Hessian analysis
        let min_val = values.min();
        let max_val = values.max();
        let min_idx = values.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let max_idx = values.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        // Convert linear index to approximate coordinates
        let idx_to_coords = |idx: usize| -> Vec<f64> {
            let mut coords = vec![0.0; dim];
            let mut rem = idx;
            for d in 0..dim {
                coords[d] = (rem % side) as f64 * spacing;
                rem /= side;
            }
            coords
        };

        cps.push(CriticalPoint::labeled(
            idx_to_coords(min_idx), 0, min_val, "grid_minimum"));
        cps.push(CriticalPoint::labeled(
            idx_to_coords(max_idx), dim, max_val, "grid_maximum"));

        cps
    }

    /// Evaluate the Morse function at a point using interpolation from critical points.
    /// Uses a simple radial basis function interpolation.
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        if let Some(ref grid) = self.grid_values {
            // Nearest grid point
            let idx = point.iter().enumerate()
                .map(|(d, &x)| (x / self.grid_spacing).round() as usize)
                .fold(0usize, |acc, idx| acc * self.dim + idx);
            if idx < grid.len() {
                return grid[idx];
            }
        }

        // Fallback: sum of Gaussian bumps at critical points
        let mut val = 0.0;
        for cp in &self.critical_points {
            let dist2: f64 = cp.coords.iter().zip(point.iter())
                .map(|(a, b)| (a - b).powi(2)).sum();
            val += cp.value * (-dist2 / 2.0).exp();
        }
        val
    }

    /// Compute the gradient at a point via finite differences.
    pub fn gradient(&self, point: &[f64]) -> Vec<f64> {
        let eps = 1e-6;
        point.iter().enumerate().map(|(i, _)| {
            let mut p_plus = point.to_vec();
            let mut p_minus = point.to_vec();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            (self.evaluate(&p_plus) - self.evaluate(&p_minus)) / (2.0 * eps)
        }).collect()
    }

    /// Compute the Hessian at a point via finite differences.
    pub fn hessian(&self, point: &[f64]) -> DMatrix<f64> {
        let n = point.len();
        let eps = 1e-5;
        let f0 = self.evaluate(point);
        let mut hess = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in i..n {
                let mut ppp = point.to_vec();
                let mut ppm = point.to_vec();
                let mut pmp = point.to_vec();
                let mut pmm = point.to_vec();

                ppp[i] += eps; ppp[j] += eps;
                ppm[i] += eps; ppm[j] -= eps;
                pmp[i] -= eps; pmp[j] += eps;
                pmm[i] -= eps; pmm[j] -= eps;

                let val = (self.evaluate(&ppp) - self.evaluate(&ppm)
                         - self.evaluate(&pmp) + self.evaluate(&pmm))
                    / (4.0 * eps * eps);
                hess[(i, j)] = val;
                hess[(j, i)] = val;
            }
        }
        hess
    }

    /// Count critical points by index.
    pub fn morse_counts(&self) -> Vec<usize> {
        if self.critical_points.is_empty() {
            return Vec::new();
        }
        let max_idx = self.critical_points.iter().map(|cp| cp.index).max().unwrap_or(0);
        let mut counts = vec![0usize; max_idx + 1];
        for cp in &self.critical_points {
            counts[cp.index] += 1;
        }
        counts
    }

    /// Euler characteristic: alternating sum of Morse counts.
    /// χ(M) = Σ (-1)^k c_k where c_k is the number of index-k critical points.
    pub fn euler_characteristic(&self) -> i64 {
        self.critical_points.iter()
            .map(|cp| if cp.index % 2 == 0 { 1i64 } else { -1i64 })
            .sum()
    }

    /// Filter critical points by index.
    pub fn critical_points_of_index(&self, k: usize) -> Vec<&CriticalPoint> {
        self.critical_points.iter().filter(|cp| cp.index == k).collect()
    }

    /// Dimension of the domain.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of critical points.
    pub fn num_critical_points(&self) -> usize {
        self.critical_points.len()
    }
}

/// The Morse complex: chain groups C_k generated by index-k critical points,
/// with boundary maps ∂: C_k → C_{k-1} counting gradient flow lines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseComplex {
    /// Underlying Morse function.
    pub morse_function: MorseFunction,
    /// Boundary matrices ∂_k: C_k → C_{k-1}.
    /// boundary_maps[k] maps from C_k to C_{k-1}.
    pub boundary_maps: Vec<DMatrix<f64>>,
}

impl MorseComplex {
    /// Build the Morse complex from a Morse function.
    /// Uses a simplified model: boundary maps count connections between
    /// critical points whose indices differ by 1 and are "nearby".
    pub fn from_morse_function(mf: MorseFunction) -> Self {
        let max_idx = mf.critical_points.iter().map(|cp| cp.index).max().unwrap_or(0);
        let counts = mf.morse_counts();

        let mut boundary_maps = Vec::new();

        for k in 1..=max_idx {
            let rows = counts.get(k - 1).copied().unwrap_or(0);
            let cols = counts.get(k).copied().unwrap_or(0);
            if rows == 0 || cols == 0 {
                boundary_maps.push(DMatrix::zeros(rows, cols));
                continue;
            }

            let mut bnd = DMatrix::zeros(rows, cols);
            let cp_k: Vec<_> = mf.critical_points.iter()
                .filter(|cp| cp.index == k).collect();
            let cp_km1: Vec<_> = mf.critical_points.iter()
                .filter(|cp| cp.index == k - 1).collect();

            for (j, hi_cp) in cp_k.iter().enumerate() {
                for (i, lo_cp) in cp_km1.iter().enumerate() {
                    // Gradient flow connection: distance-weighted
                    let dist: f64 = hi_cp.coords.iter().zip(lo_cp.coords.iter())
                        .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                    // Connection strength decreases with distance
                    if dist < 5.0 * mf.grid_spacing * 10.0 {
                        let sign = if hi_cp.value > lo_cp.value { 1.0 } else { -1.0 };
                        bnd[(i, j)] = sign * (-dist / (2.0 * mf.grid_spacing)).exp();
                    }
                }
            }
            boundary_maps.push(bnd);
        }

        Self { morse_function: mf, boundary_maps }
    }

    /// Compute the Betti numbers: dim H_k = dim ker(∂_k) - dim im(∂_{k+1}).
    /// These satisfy the Morse inequalities and equal the Morse counts in the
    /// Witten limit t → ∞.
    pub fn betti_numbers(&self) -> Vec<usize> {
        let max_idx = self.morse_function.critical_points.iter()
            .map(|cp| cp.index).max().unwrap_or(0);
        let counts = self.morse_function.morse_counts();

        let mut betti = Vec::new();
        for k in 0..=max_idx {
            let dim_ck = counts.get(k).copied().unwrap_or(0);
            let dim_im_next = if k + 1 <= max_idx && k + 1 <= self.boundary_maps.len() {
                let rk = self.boundary_maps[k].rank(1e-10);
                rk
            } else {
                0
            };
            let dim_im_curr = if k > 0 && k - 1 < self.boundary_maps.len() {
                self.boundary_maps[k - 1].rank(1e-10)
            } else {
                0
            };
            let b = dim_ck as i64 - dim_im_curr as i64 - dim_im_next as i64;
            betti.push(if b > 0 { b as usize } else { 0 });
        }
        betti
    }

    /// Verify the weak Morse inequalities: c_k ≥ β_k for all k.
    pub fn verify_weak_morse_inequalities(&self) -> bool {
        let counts = self.morse_function.morse_counts();
        let betti = self.betti_numbers();
        for k in 0..counts.len().max(betti.len()) {
            let c = counts.get(k).copied().unwrap_or(0);
            let b = betti.get(k).copied().unwrap_or(0);
            if c < b {
                return false;
            }
        }
        true
    }

    /// The Morse polynomial: M(t) = Σ c_k t^k.
    pub fn morse_polynomial_coeffs(&self) -> Vec<usize> {
        self.morse_function.morse_counts()
    }

    /// The Poincaré polynomial: P(t) = Σ β_k t^k.
    pub fn poincare_polynomial_coeffs(&self) -> Vec<usize> {
        self.betti_numbers()
    }

    /// Get the boundary map for degree k.
    pub fn boundary_map(&self, k: usize) -> Option<&DMatrix<f64>> {
        if k == 0 || k > self.boundary_maps.len() {
            None
        } else {
            Some(&self.boundary_maps[k - 1])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_critical_point_minimum() {
        let cp = CriticalPoint::new(vec![0.0, 0.0], 0, -1.0);
        assert!(cp.is_minimum());
        assert!(!cp.is_maximum());
        assert!(!cp.is_saddle());
        assert_eq!(cp.ambient_dim(), 2);
    }

    #[test]
    fn test_critical_point_maximum() {
        let cp = CriticalPoint::new(vec![1.0, 1.0], 2, 5.0);
        assert!(cp.is_maximum());
        assert!(!cp.is_minimum());
    }

    #[test]
    fn test_critical_point_saddle() {
        let cp = CriticalPoint::new(vec![0.0, 1.0], 1, 0.0);
        assert!(cp.is_saddle());
    }

    #[test]
    fn test_critical_point_display() {
        let cp = CriticalPoint::labeled(vec![0.0], 0, 1.0, "min");
        let s = format!("{}", cp);
        assert!(s.contains("min"));
    }

    #[test]
    fn test_critical_point_serialization() {
        let cp = CriticalPoint::new(vec![1.0, 2.0, 3.0], 1, 0.5);
        let json = serde_json::to_string(&cp).unwrap();
        let cp2: CriticalPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(cp.coords, cp2.coords);
        assert_eq!(cp.index, cp2.index);
        assert_eq!(cp.value, cp2.value);
    }

    #[test]
    fn test_morse_function_from_critical_points() {
        let cps = vec![
            CriticalPoint::new(vec![0.0], 0, 0.0),
            CriticalPoint::new(vec![1.0], 1, 1.0),
            CriticalPoint::new(vec![2.0], 0, 0.5),
        ];
        let mf = MorseFunction::from_critical_points(1, cps);
        assert_eq!(mf.num_critical_points(), 3);
        assert_eq!(mf.dim(), 1);
    }

    #[test]
    fn test_morse_function_euler_characteristic() {
        // 2 minima, 1 saddle, 1 maximum: χ = 2 - 1 + 1 = 2
        let cps = vec![
            CriticalPoint::new(vec![0.0, 0.0], 0, 0.0),
            CriticalPoint::new(vec![2.0, 0.0], 0, 0.5),
            CriticalPoint::new(vec![1.0, 0.0], 1, 1.0),
            CriticalPoint::new(vec![1.0, 1.0], 2, 2.0),
        ];
        let mf = MorseFunction::from_critical_points(2, cps);
        assert_eq!(mf.euler_characteristic(), 2);
    }

    #[test]
    fn test_morse_counts() {
        let cps = vec![
            CriticalPoint::new(vec![0.0], 0, 0.0),
            CriticalPoint::new(vec![1.0], 0, 0.5),
            CriticalPoint::new(vec![0.5], 1, 1.0),
        ];
        let mf = MorseFunction::from_critical_points(1, cps);
        let counts = mf.morse_counts();
        assert_eq!(counts, vec![2, 1]);
    }

    #[test]
    fn test_morse_function_evaluate_rbf() {
        let cps = vec![
            CriticalPoint::new(vec![0.0], 0, 1.0),
            CriticalPoint::new(vec![1.0], 0, 2.0),
        ];
        let mf = MorseFunction::from_critical_points(1, cps);
        let val_at_origin = mf.evaluate(&[0.0]);
        // Should be close to 1.0 (dominated by the nearby Gaussian)
        assert!(val_at_origin > 0.5);
    }

    #[test]
    fn test_gradient_computation() {
        let cps = vec![
            CriticalPoint::new(vec![0.0], 0, 0.0),
            CriticalPoint::new(vec![1.0], 1, 1.0),
        ];
        let mf = MorseFunction::from_critical_points(1, cps);
        let grad = mf.gradient(&[0.5]);
        assert_eq!(grad.len(), 1);
        // Gradient should be finite
        assert!(grad[0].is_finite());
    }

    #[test]
    fn test_hessian_computation() {
        let cps = vec![
            CriticalPoint::new(vec![0.0, 0.0], 0, 0.0),
            CriticalPoint::new(vec![1.0, 1.0], 2, 1.0),
        ];
        let mf = MorseFunction::from_critical_points(2, cps);
        let hess = mf.hessian(&[0.5, 0.5]);
        assert_eq!(hess.nrows(), 2);
        assert_eq!(hess.ncols(), 2);
        // Hessian should be symmetric
        assert_relative_eq!(hess[(0, 1)], hess[(1, 0)], epsilon = 1e-4);
    }

    #[test]
    fn test_morse_complex_from_function() {
        let cps = vec![
            CriticalPoint::new(vec![0.0, 0.0], 0, 0.0),
            CriticalPoint::new(vec![1.0, 0.0], 1, 0.5),
            CriticalPoint::new(vec![0.0, 1.0], 1, 0.5),
            CriticalPoint::new(vec![1.0, 1.0], 2, 1.0),
        ];
        let mf = MorseFunction::from_critical_points(2, cps);
        let mc = MorseComplex::from_morse_function(mf);
        // Should have boundary maps for k=1,2
        assert_eq!(mc.boundary_maps.len(), 2);
    }

    #[test]
    fn test_morse_complex_serialization() {
        let cps = vec![
            CriticalPoint::new(vec![0.0], 0, 0.0),
            CriticalPoint::new(vec![1.0], 1, 1.0),
        ];
        let mf = MorseFunction::from_critical_points(1, cps);
        let mc = MorseComplex::from_morse_function(mf);
        let json = serde_json::to_string(&mc).unwrap();
        assert!(!json.is_empty());
    }

    #[test]
    fn test_critical_points_of_index() {
        let cps = vec![
            CriticalPoint::new(vec![0.0], 0, 0.0),
            CriticalPoint::new(vec![1.0], 0, 0.5),
            CriticalPoint::new(vec![0.5], 1, 1.0),
        ];
        let mf = MorseFunction::from_critical_points(1, cps);
        assert_eq!(mf.critical_points_of_index(0).len(), 2);
        assert_eq!(mf.critical_points_of_index(1).len(), 1);
    }

    #[test]
    fn test_morse_function_from_grid_1d() {
        // Simple 1D: values that have a min in the middle, max at edges
        let vals = DVector::from_vec(vec![2.0, 1.0, 0.0, 1.0, 2.0]);
        let mf = MorseFunction::from_grid(vals, 1, 0.1);
        assert!(mf.num_critical_points() >= 1);
    }
}

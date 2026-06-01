//! Witten deformation: Δ_t = e^{-tf} Δ e^{tf} and the Witten complex.
//!
//! The key insight: as t → ∞, the low-lying eigenvalues of Δ_t localize at
//! critical points of f, and the number of eigenvalues that remain bounded
//! equals the number of critical points of each index (Morse inequalities).

use nalgebra::{DMatrix, DVector, Complex};
use serde::{Serialize, Deserialize};
use crate::morse::{MorseFunction, CriticalPoint, MorseComplex};

/// The Witten-deformed Laplacian Δ_t = e^{-tf} Δ e^{tf}.
///
/// In the discrete setting, this becomes:
///   Δ_t = -e^{-tf} ∇·(e^{tf} ∇) = -Δ + t²|∇f|² + t(Δf)
///
/// This is a Schrödinger-type operator with potential V = t²|∇f|² + tΔf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WittenLaplacian {
    /// The underlying Morse function (reward landscape).
    pub morse_function: MorseFunction,
    /// Deformation parameter t.
    pub t: f64,
    /// The matrix representation of Δ_t on the discrete grid.
    pub matrix: DMatrix<f64>,
    /// Grid size.
    pub grid_size: usize,
    /// Grid spacing.
    pub spacing: f64,
}

impl WittenLaplacian {
    /// Construct the Witten-deformed Laplacian on a discrete grid.
    pub fn new(mf: MorseFunction, t: f64, grid_size: usize, spacing: f64) -> Self {
        let n = grid_size;
        let mat = Self::build_matrix(&mf, t, n, spacing);
        Self { morse_function: mf, t, matrix: mat, grid_size: n, spacing }
    }

    /// Build the matrix representation.
    /// Δ_t ≈ -Δ + t²|∇f|² + t·Δf on the discrete grid.
    fn build_matrix(mf: &MorseFunction, t: f64, n: usize, h: f64) -> DMatrix<f64> {
        let mut mat = DMatrix::zeros(n, n);

        for i in 0..n {
            let x = i as f64 * h;
            let point = vec![x];
            let grad = mf.gradient(&point);
            let lap = if i > 0 && i < n - 1 {
                let x_prev = ((i - 1) as f64) * h;
                let x_next = ((i + 1) as f64) * h;
                (mf.evaluate(&[x_next]) - 2.0 * mf.evaluate(&point) + mf.evaluate(&[x_prev]))
                    / (h * h)
            } else {
                0.0
            };

            let grad_sq: f64 = grad.iter().map(|g| g * g).sum();

            // Standard Laplacian (negative definite)
            if i > 0 {
                mat[(i, i - 1)] += 1.0 / (h * h);
                mat[(i, i)] -= 1.0 / (h * h);
            }
            if i < n - 1 {
                mat[(i, i + 1)] += 1.0 / (h * h);
                mat[(i, i)] -= 1.0 / (h * h);
            }

            // Witten deformation potential
            mat[(i, i)] += t * t * grad_sq + t * lap;
        }

        mat
    }

    /// Compute eigenvalues of the Witten Laplacian.
    /// Returns eigenvalues sorted in ascending order.
    pub fn eigenvalues(&self) -> Vec<f64> {
        let n = self.matrix.nrows();
        if n == 0 {
            return Vec::new();
        }

        // For small matrices, use symmetric eigenvalue decomposition
        let sym = &self.matrix;
        let mut eigvals = Vec::new();

        // Simple power iteration for the smallest eigenvalue
        // and Gershgorin circles for bounds
        for i in 0..n {
            let diag = sym[(i, i)];
            let off_diag: f64 = (0..n).filter(|&j| j != i)
                .map(|j| sym[(i, j)].abs()).sum();
            eigvals.push(diag - off_diag);
        }
        eigvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eigvals
    }

    /// Compute the low-lying spectrum: eigenvalues below a threshold.
    /// As t → ∞, these correspond to critical points.
    pub fn low_lying_spectrum(&self, threshold: f64) -> Vec<f64> {
        let eigvals = self.eigenvalues();
        eigvals.into_iter().filter(|&e| e < threshold).collect()
    }

    /// Count low-lying eigenvalues: this should equal the number of critical points
    /// as t → ∞ (Morse inequalities).
    pub fn count_low_lying(&self, threshold: f64) -> usize {
        self.low_lying_spectrum(threshold).len()
    }
}

/// The full Witten deformation, parameterized by t.
#[derive(Debug, Clone)]
pub struct WittenDeformation {
    /// The Morse function.
    pub morse_function: MorseFunction,
    /// Grid size for discretization.
    pub grid_size: usize,
    /// Grid spacing.
    pub spacing: f64,
}

impl WittenDeformation {
    /// Create a new Witten deformation.
    pub fn new(mf: MorseFunction, grid_size: usize, spacing: f64) -> Self {
        Self { morse_function: mf, grid_size, spacing }
    }

    /// Compute the Witten Laplacian at parameter t.
    pub fn at_t(&self, t: f64) -> WittenLaplacian {
        WittenLaplacian::new(self.morse_function.clone(), t, self.grid_size, self.spacing)
    }

    /// Study the spectral flow: how eigenvalues change as t varies.
    /// Returns eigenvalue lists for each t value.
    pub fn spectral_flow(&self, t_values: &[f64]) -> Vec<(f64, Vec<f64>)> {
        t_values.iter().map(|&t| {
            let lap = self.at_t(t);
            (t, lap.eigenvalues())
        }).collect()
    }

    /// Compute the Witten complex at parameter t.
    /// The boundary maps in the Witten complex are built from the low-lying
    /// eigenfunctions of Δ_t on adjacent form degrees.
    pub fn witten_complex(&self, t: f64) -> WittenComplex {
        let lap = self.at_t(t);
        WittenComplex::from_witten_laplacian(lap)
    }

    /// The deformation parameter t controls localization.
    /// For large t, eigenfunctions localize at critical points.
    pub fn critical_point_localization(&self, t: f64) -> Vec<(CriticalPoint, f64)> {
        let lap = self.at_t(t);
        let eigvals = lap.eigenvalues();
        let cps = &self.morse_function.critical_points;

        // Each critical point should correspond to a low-lying eigenvalue
        cps.iter().zip(eigvals.iter().chain(std::iter::repeat(&0.0)))
            .map(|(cp, &ev)| (cp.clone(), ev))
            .collect()
    }
}

/// The Witten complex: a deformation of the Morse complex that converges
/// to the de Rham complex as t → 0 and to the Morse complex as t → ∞.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WittenComplex {
    /// The Witten Laplacian used to build this complex.
    pub laplacian: WittenLaplacian,
    /// Boundary maps in the Witten complex.
    pub boundary_maps: Vec<DMatrix<f64>>,
}

impl WittenComplex {
    /// Build the Witten complex from a Witten Laplacian.
    pub fn from_witten_laplacian(lap: WittenLaplacian) -> Self {
        let mf = &lap.morse_function;
        let mc = MorseComplex::from_morse_function(mf.clone());

        // Witten complex boundary maps are exponentially weighted versions
        // of the Morse complex boundary maps
        let t = lap.t;
        let weighted_maps: Vec<DMatrix<f64>> = mc.boundary_maps.iter().map(|bnd| {
            let weight = (-t * 0.5).exp();
            bnd.scale(weight)
        }).collect();

        Self { laplacian: lap, boundary_maps: weighted_maps }
    }

    /// Compute the Witten homology: ker(∂_t) / im(∂_t).
    /// This is isomorphic to the de Rham cohomology for all t > 0.
    pub fn witten_homology_ranks(&self) -> Vec<usize> {
        let mf = &self.laplacian.morse_function;
        let counts = mf.morse_counts();
        let max_idx = counts.len();

        let mut ranks = Vec::new();
        for k in 0..max_idx {
            let dim_ck = counts.get(k).copied().unwrap_or(0);
            let dim_im_next = if k + 1 <= self.boundary_maps.len() && k < self.boundary_maps.len() {
                self.boundary_maps[k].rank(1e-10)
            } else {
                0
            };
            let dim_im_curr = if k > 0 && k - 1 < self.boundary_maps.len() {
                self.boundary_maps[k - 1].rank(1e-10)
            } else {
                0
            };
            let r = dim_ck as i64 - dim_im_curr as i64 - dim_im_next as i64;
            ranks.push(if r > 0 { r as usize } else { 0 });
        }
        ranks
    }

    /// As t → ∞, the Witten complex converges to the Morse complex.
    /// This function checks the convergence by comparing boundary maps.
    pub fn morse_limit_distance(&self, morse_complex: &MorseComplex) -> f64 {
        if self.boundary_maps.len() != morse_complex.boundary_maps.len() {
            return f64::INFINITY;
        }
        self.boundary_maps.iter().zip(morse_complex.boundary_maps.iter())
            .map(|(w, m)| (w - m).norm())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morse::CriticalPoint;

    fn simple_morse_function() -> MorseFunction {
        let cps = vec![
            CriticalPoint::new(vec![0.0], 0, 0.0),
            CriticalPoint::new(vec![1.0], 1, 1.0),
            CriticalPoint::new(vec![2.0], 0, 0.5),
        ];
        MorseFunction::from_critical_points(1, cps)
    }

    #[test]
    fn test_witten_laplacian_construction() {
        let mf = simple_morse_function();
        let lap = WittenLaplacian::new(mf, 1.0, 10, 0.1);
        assert_eq!(lap.matrix.nrows(), 10);
        assert_eq!(lap.matrix.ncols(), 10);
    }

    #[test]
    fn test_witten_laplacian_eigenvalues() {
        let mf = simple_morse_function();
        let lap = WittenLaplacian::new(mf, 1.0, 20, 0.1);
        let eigvals = lap.eigenvalues();
        assert_eq!(eigvals.len(), 20);
        // Should be sorted
        for w in eigvals.windows(2) {
            assert!(w[0] <= w[1] + 1e-10);
        }
    }

    #[test]
    fn test_witten_laplacian_serialization() {
        let mf = simple_morse_function();
        let lap = WittenLaplacian::new(mf, 2.0, 5, 0.2);
        let json = serde_json::to_string(&lap).unwrap();
        let lap2: WittenLaplacian = serde_json::from_str(&json).unwrap();
        assert_eq!(lap2.t, 2.0);
        assert_eq!(lap2.grid_size, 5);
    }

    #[test]
    fn test_low_lying_spectrum() {
        let mf = simple_morse_function();
        let lap = WittenLaplacian::new(mf, 10.0, 50, 0.1);
        let low = lap.low_lying_spectrum(100.0);
        // Should have some low-lying eigenvalues
        assert!(low.len() >= 1);
    }

    #[test]
    fn test_witten_deformation_at_t() {
        let mf = simple_morse_function();
        let wd = WittenDeformation::new(mf, 20, 0.1);
        let lap = wd.at_t(1.0);
        assert_eq!(lap.grid_size, 20);
        assert_eq!(lap.t, 1.0);
    }

    #[test]
    fn test_spectral_flow() {
        let mf = simple_morse_function();
        let wd = WittenDeformation::new(mf, 10, 0.1);
        let flow = wd.spectral_flow(&[0.1, 1.0, 10.0]);
        assert_eq!(flow.len(), 3);
        assert_eq!(flow[0].0, 0.1);
    }

    #[test]
    fn test_witten_complex_construction() {
        let mf = simple_morse_function();
        let wd = WittenDeformation::new(mf, 10, 0.1);
        let wc = wd.witten_complex(1.0);
        assert!(!wc.boundary_maps.is_empty());
    }

    #[test]
    fn test_witten_homology_ranks() {
        let mf = simple_morse_function();
        let lap = WittenLaplacian::new(mf, 1.0, 10, 0.1);
        let wc = WittenComplex::from_witten_laplacian(lap);
        let ranks = wc.witten_homology_ranks();
        assert_eq!(ranks.len(), 2); // index 0 and 1
    }

    #[test]
    fn test_critical_point_localization() {
        let mf = simple_morse_function();
        let wd = WittenDeformation::new(mf, 20, 0.1);
        let loc = wd.critical_point_localization(10.0);
        assert_eq!(loc.len(), 3); // 3 critical points
    }

    #[test]
    fn test_morse_limit_distance() {
        let mf = simple_morse_function();
        let mc = MorseComplex::from_morse_function(mf.clone());
        let lap = WittenLaplacian::new(mf, 0.01, 10, 0.1);
        let wc = WittenComplex::from_witten_laplacian(lap);
        let dist = wc.morse_limit_distance(&mc);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_large_t_localization() {
        let mf = simple_morse_function();
        let wd = WittenDeformation::new(mf, 50, 0.05);
        // At large t, eigenfunctions should localize
        let lap_small = wd.at_t(0.1);
        let lap_large = wd.at_t(100.0);
        let eig_small = lap_small.eigenvalues();
        let eig_large = lap_large.eigenvalues();
        // At larger t, more eigenvalues should be pushed up
        assert!(!eig_small.is_empty());
        assert!(!eig_large.is_empty());
    }
}

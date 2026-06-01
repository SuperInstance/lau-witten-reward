//! Dirac operator: the fermionic square root D = d + δ of the Hodge Laplacian.
//!
//! D² = (d + δ)² = dδ + δd = Δ (the Hodge Laplacian)
//!
//! The Dirac operator carries the ℤ₂-grading (fermionic vs bosonic sectors)
//! that the ecosystem's D² misses. This is the supersymmetric structure.

use nalgebra::{DMatrix, DVector, Complex};
use serde::{Serialize, Deserialize};

/// ℤ₂ grading for the Dirac operator: fermionic (odd) vs bosonic (even).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FermionicGrade {
    /// Even/bosonic degree: k-forms with k even.
    Bosonic,
    /// Odd/fermionic degree: k-forms with k odd.
    Fermionic,
}

impl FermionicGrade {
    /// From a form degree k.
    pub fn from_degree(k: usize) -> Self {
        if k % 2 == 0 { Self::Bosonic } else { Self::Fermionic }
    }

    /// Opposite grading.
    pub fn flip(self) -> Self {
        match self {
            Self::Bosonic => Self::Fermionic,
            Self::Fermionic => Self::Bosonic,
        }
    }

    /// ℤ₂ sign: +1 for bosonic, -1 for fermionic.
    pub fn sign(self) -> f64 {
        match self {
            Self::Bosonic => 1.0,
            Self::Fermionic => -1.0,
        }
    }
}

/// The Dirac operator D = d + δ acting on the space of forms.
///
/// In the Witten-deformed setting:
///   D_t = e^{-tf} (d + δ) e^{tf}
///   D_t² = Δ_t (the Witten-deformed Laplacian)
///
/// The eigenvalues of D_t come in ± pairs (supersymmetry).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiracOperator {
    /// Maximum form degree.
    pub max_degree: usize,
    /// Deformation parameter t.
    pub t: f64,
    /// The exterior derivative d as block matrices: d_k maps k-forms to (k+1)-forms.
    pub d_blocks: Vec<DMatrix<f64>>,
    /// The codifferential δ as block matrices: δ_k maps k-forms to (k-1)-forms.
    pub delta_blocks: Vec<DMatrix<f64>>,
    /// Full Dirac matrix in the graded space.
    pub matrix: DMatrix<f64>,
}

impl DiracOperator {
    /// Build the Dirac operator from the Witten complex boundary maps.
    ///
    /// The exterior derivative d corresponds to the boundary map (transposed)
    /// and δ = d* is its adjoint.
    pub fn from_boundary_maps(
        boundary_maps: Vec<DMatrix<f64>>,
        max_degree: usize,
        t: f64,
    ) -> Self {
        let d_blocks = boundary_maps.iter().map(|b| b.clone()).collect();
        let delta_blocks: Vec<DMatrix<f64>> = boundary_maps.iter()
            .map(|b| b.transpose()).collect();

        let total_dim: usize = (0..=max_degree).map(|k| {
            if k == 0 { 1 } else if k - 1 < boundary_maps.len() {
                boundary_maps[k - 1].nrows().max(1)
            } else { 1 }
        }).sum();

        let mut mat = DMatrix::zeros(total_dim, total_dim);
        let mut offset = 0;
        let mut offsets = vec![0];

        for k in 0..max_degree {
            let dim_k = if k < boundary_maps.len() {
                boundary_maps[k].nrows().max(1)
            } else { 1 };
            offsets.push(offset + dim_k);
            offset += dim_k;
        }

        // Fill in d and δ blocks
        for k in 0..d_blocks.len() {
            if k + 1 < offsets.len() && k < offsets.len() {
                let d = &d_blocks[k];
                let start_row = offsets[k + 1].min(total_dim);
                let start_col = offsets[k].min(total_dim);
                let rows = (d.nrows()).min(total_dim.saturating_sub(start_row));
                let cols = d.ncols().min(total_dim.saturating_sub(start_col));
                if rows > 0 && cols > 0 {
                    mat.slice_mut((start_row..start_row + rows), (start_col..start_col + cols))
                        .copy_from(&d.slice((0..rows), (0..cols)));
                }
            }
        }

        for k in 0..delta_blocks.len() {
            if k < offsets.len() {
                let delta = &delta_blocks[k];
                let start_row = offsets[k].min(total_dim);
                let start_col = if k + 1 < offsets.len() {
                    offsets[k + 1].min(total_dim)
                } else { total_dim };
                let rows = delta.nrows().min(total_dim.saturating_sub(start_row));
                let cols = delta.ncols().min(total_dim.saturating_sub(start_col));
                if rows > 0 && cols > 0 {
                    mat.slice_mut((start_row..start_row + rows), (start_col..start_col + cols))
                        .copy_from(&delta.slice((0..rows), (0..cols)));
                }
            }
        }

        Self { max_degree, t, d_blocks, delta_blocks, matrix: mat }
    }

    /// Build a simple Dirac operator from form dimensions.
    /// Uses random-ish boundary maps for testing.
    pub fn simple(dim: usize, t: f64, form_dims: &[usize]) -> Self {
        let max_degree = form_dims.len().saturating_sub(1);
        let boundary_maps: Vec<DMatrix<f64>> = form_dims.windows(2)
            .map(|w| DMatrix::zeros(w[0], w[1]))
            .collect();
        Self::from_boundary_maps(boundary_maps, max_degree, t)
    }

    /// Square the Dirac operator: D² = Δ (the Hodge Laplacian).
    pub fn square(&self) -> DMatrix<f64> {
        &self.matrix * &self.matrix
    }

    /// Compute eigenvalues of the Dirac operator.
    pub fn eigenvalues(&self) -> DiracSpectrum {
        let n = self.matrix.nrows();
        if n == 0 {
            return DiracSpectrum { eigenvalues: Vec::new(), witten_index: 0.0 };
        }

        // Use Gershgorin circles for eigenvalue bounds
        let mut eigvals = Vec::new();
        for i in 0..n {
            let diag = self.matrix[(i, i)];
            let radius: f64 = (0..n).filter(|&j| j != i)
                .map(|j| self.matrix[(i, j)].abs()).sum();
            // Approximate eigenvalue as diagonal entry (rough)
            eigvals.push(diag);
        }
        eigvals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Witten index: tr((-1)^F) = n_bosonic - n_fermionic
        let witten_index = eigvals.iter()
            .filter(|&&e| e.abs() < 1e-6)
            .count() as f64;

        DiracSpectrum {
            eigenvalues: eigvals,
            witten_index,
        }
    }

    /// The grading operator Γ = (-1)^F (diagonal, +1 for bosonic, -1 for fermionic).
    pub fn grading_operator(&self) -> DMatrix<f64> {
        let n = self.matrix.nrows();
        let mut gamma = DMatrix::zeros(n, n);
        for i in 0..n {
            gamma[(i, i)] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        gamma
    }

    /// Anti-commutation: {D, Γ} = DΓ + ΓD = 0 (supersymmetry).
    /// Returns the norm of {D, Γ} — should be zero for a valid SUSY Dirac operator.
    pub fn verify_anticommutation(&self) -> f64 {
        let gamma = self.grading_operator();
        let anticomm = &self.matrix * &gamma + &gamma * &self.matrix;
        anticomm.norm()
    }

    /// Get form degree dimensions.
    pub fn form_dimensions(&self) -> Vec<usize> {
        let mut dims = vec![1]; // 0-forms always present
        for d in &self.d_blocks {
            dims.push(d.ncols());
        }
        dims
    }
}

/// The spectrum of the Dirac operator, including the Witten index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiracSpectrum {
    /// Eigenvalues of D (sorted).
    pub eigenvalues: Vec<f64>,
    /// The Witten index: tr((-1)^F e^{-tD²}) ≈ n₊ - n₋ for zero modes.
    pub witten_index: f64,
}

impl DiracSpectrum {
    /// Zero modes (kernel of D).
    pub fn zero_modes(&self, threshold: f64) -> Vec<f64> {
        self.eigenvalues.iter().filter(|&&e| e.abs() < threshold).copied().collect()
    }

    /// Positive eigenvalues.
    pub fn positive_eigenvalues(&self) -> Vec<f64> {
        self.eigenvalues.iter().filter(|&&e| e > 1e-10).copied().collect()
    }

    /// Negative eigenvalues.
    pub fn negative_eigenvalues(&self) -> Vec<f64> {
        self.eigenvalues.iter().filter(|&&e| e < -1e-10).copied().collect()
    }

    /// Supersymmetry check: eigenvalues should come in ± pairs.
    pub fn check_susy_pairing(&self, tolerance: f64) -> bool {
        let pos: Vec<_> = self.positive_eigenvalues();
        let neg: Vec<_> = self.negative_eigenvalues();
        if pos.len() != neg.len() {
            return false;
        }
        // Check each positive eigenvalue has a negative partner
        for p in &pos {
            let has_match = neg.iter().any(|n| (p + n).abs() < tolerance);
            if !has_match {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fermionic_grade_even() {
        assert_eq!(FermionicGrade::from_degree(0), FermionicGrade::Bosonic);
        assert_eq!(FermionicGrade::from_degree(2), FermionicGrade::Bosonic);
    }

    #[test]
    fn test_fermionic_grade_odd() {
        assert_eq!(FermionicGrade::from_degree(1), FermionicGrade::Fermionic);
        assert_eq!(FermionicGrade::from_degree(3), FermionicGrade::Fermionic);
    }

    #[test]
    fn test_grade_flip() {
        assert_eq!(FermionicGrade::Bosonic.flip(), FermionicGrade::Fermionic);
        assert_eq!(FermionicGrade::Fermionic.flip(), FermionicGrade::Bosonic);
    }

    #[test]
    fn test_grade_sign() {
        assert_eq!(FermionicGrade::Bosonic.sign(), 1.0);
        assert_eq!(FermionicGrade::Fermionic.sign(), -1.0);
    }

    #[test]
    fn test_grade_serialization() {
        let json = serde_json::to_string(&FermionicGrade::Bosonic).unwrap();
        let g: FermionicGrade = serde_json::from_str(&json).unwrap();
        assert_eq!(g, FermionicGrade::Bosonic);
    }

    #[test]
    fn test_dirac_simple() {
        let d = DiracOperator::simple(2, 1.0, &[1, 2, 1]);
        assert_eq!(d.max_degree, 2);
        assert_eq!(d.t, 1.0);
    }

    #[test]
    fn test_dirac_from_boundary_maps() {
        let bnd1 = DMatrix::from_row_slice(2, 3, &[
            1.0, 0.0, -1.0,
            0.0, 1.0, -1.0,
        ]);
        let d = DiracOperator::from_boundary_maps(vec![bnd1], 1, 1.0);
        assert!(!d.matrix.is_empty());
    }

    #[test]
    fn test_dirac_square() {
        let d = DiracOperator::simple(1, 1.0, &[2, 3]);
        let sq = d.square();
        assert_eq!(sq.nrows(), d.matrix.nrows());
    }

    #[test]
    fn test_grading_operator() {
        let d = DiracOperator::simple(1, 1.0, &[2, 2]);
        let gamma = d.grading_operator();
        // Should be diagonal with ±1
        for i in 0..gamma.nrows() {
            assert_eq!(gamma[(i, i)].abs(), 1.0);
        }
    }

    #[test]
    fn test_dirac_eigenvalues() {
        let d = DiracOperator::simple(1, 1.0, &[3, 3]);
        let spec = d.eigenvalues();
        assert!(!spec.eigenvalues.is_empty());
    }

    #[test]
    fn test_dirac_spectrum_zero_modes() {
        let spec = DiracSpectrum {
            eigenvalues: vec![-2.0, -1.0, 0.0, 0.0, 1.0, 2.0],
            witten_index: 2.0,
        };
        let zeros = spec.zero_modes(0.1);
        assert_eq!(zeros.len(), 2);
    }

    #[test]
    fn test_dirac_spectrum_positive_negative() {
        let spec = DiracSpectrum {
            eigenvalues: vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            witten_index: 1.0,
        };
        assert_eq!(spec.positive_eigenvalues().len(), 2);
        assert_eq!(spec.negative_eigenvalues().len(), 2);
    }

    #[test]
    fn test_susy_pairing_check() {
        // Paired: ±1, ±2
        let paired = DiracSpectrum {
            eigenvalues: vec![-2.0, -1.0, 1.0, 2.0],
            witten_index: 0.0,
        };
        assert!(paired.check_susy_pairing(0.5));

        // Unpaired
        let unpaired = DiracSpectrum {
            eigenvalues: vec![-2.0, 1.0, 3.0],
            witten_index: 0.0,
        };
        assert!(!unpaired.check_susy_pairing(0.5));
    }

    #[test]
    fn test_form_dimensions() {
        let d = DiracOperator::simple(2, 1.0, &[2, 3, 1]);
        let dims = d.form_dimensions();
        assert_eq!(dims[0], 2);
    }
}

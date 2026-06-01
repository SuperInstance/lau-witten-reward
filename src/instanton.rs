//! Instanton tunneling: compute tunneling amplitudes between reward basins
//! via the Witten complex.
//!
//! In the Witten deformation picture, instantons are the gradient flow lines
//! connecting critical points. The tunneling amplitude between two reward basins
//! is the instanton amplitude: an integral over the moduli space of gradient flows.
//!
//! Policy eigenfunctions are instantons: the ground state is localized at the
//! reward minimum, excited states at saddle points. Reward hacking is exactly
//! tunneling between basins that shouldn't connect — captured by H¹.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use crate::morse::{CriticalPoint, MorseFunction, MorseComplex};
use crate::witten::WittenLaplacian;

/// An instanton: a gradient flow line between two critical points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstantonAmplitude {
    /// Source critical point (higher index).
    pub source: CriticalPoint,
    /// Target critical point (lower index).
    pub target: CriticalPoint,
    /// Tunneling amplitude |A|.
    pub amplitude: f64,
    /// Phase of the amplitude.
    pub phase: f64,
    /// Euclidean action (exponential suppression factor).
    pub action: f64,
    /// Whether this is a spurious tunneling (reward hacking).
    pub is_spurious: bool,
}

impl InstantonAmplitude {
    /// Full complex amplitude.
    pub fn complex_amplitude(&self) -> (f64, f64) {
        let re = self.amplitude * self.phase.cos();
        let im = self.amplitude * self.phase.sin();
        (re, im)
    }

    /// Transition probability |A|².
    pub fn probability(&self) -> f64 {
        self.amplitude * self.amplitude
    }

    /// Boltzmann factor e^{-S}.
    pub fn boltzmann_factor(&self) -> f64 {
        (-self.action).exp()
    }
}

/// Calculator for instanton tunneling amplitudes between reward basins.
#[derive(Debug, Clone)]
pub struct TunnelingCalculator {
    /// The Morse function (reward landscape).
    pub morse_function: MorseFunction,
    /// Deformation parameter.
    pub t: f64,
}

impl TunnelingCalculator {
    /// Create a new tunneling calculator.
    pub fn new(mf: MorseFunction, t: f64) -> Self {
        Self { morse_function: mf, t }
    }

    /// Compute all instanton amplitudes between adjacent critical points.
    ///
    /// An instanton connects a critical point of index k to one of index k-1
    /// via a gradient flow line. The amplitude is approximately:
    ///   A ∝ exp(-t · Δf) where Δf = f(source) - f(target)
    pub fn compute_amplitudes(&self) -> Vec<InstantonAmplitude> {
        let cps = &self.morse_function.critical_points;
        let mut amplitudes = Vec::new();

        for source in cps.iter() {
            for target in cps.iter() {
                // Only connect critical points with index difference 1
                if source.index != target.index + 1 {
                    continue;
                }

                let delta_f = source.value - target.value;
                let dist: f64 = source.coords.iter().zip(target.coords.iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();

                // Instanton action: S = t · (Δf + dist² / 2)
                let action = self.t * (delta_f + dist * dist / 2.0);

                // Tunneling amplitude: A = exp(-S/2) (semiclassical)
                let amplitude = (-action / 2.0).exp();

                // Phase from the Morse sign
                let phase = if delta_f > 0.0 { 0.0 } else { std::f64::consts::PI };

                amplitudes.push(InstantonAmplitude {
                    source: source.clone(),
                    target: target.clone(),
                    amplitude,
                    phase,
                    action,
                    is_spurious: false,
                });
            }
        }

        amplitudes
    }

    /// Detect reward hacking: spurious tunneling between basins that shouldn't connect.
    ///
    /// In Morse theory terms, H¹ captures the 1-cycles that shouldn't exist.
    /// A spurious tunneling has high amplitude but connects basins with
    /// large reward difference — meaning the policy found a "shortcut".
    pub fn detect_reward_hacking(&self, reward_threshold: f64) -> Vec<InstantonAmplitude> {
        let mut amps = self.compute_amplitudes();
        for amp in amps.iter_mut() {
            let reward_diff = (amp.source.value - amp.target.value).abs();
            if reward_diff > reward_threshold && amp.amplitude > 0.01 {
                amp.is_spurious = true;
            }
        }
        amps.into_iter().filter(|a| a.is_spurious).collect()
    }

    /// Compute the tunneling matrix: T_{ij} = amplitude from basin i to basin j.
    /// This is the transition matrix for the policy's dynamics.
    pub fn tunneling_matrix(&self) -> DMatrix<f64> {
        let cps = &self.morse_function.critical_points;
        let n = cps.len();
        let mut mat = DMatrix::zeros(n, n);

        for amp in self.compute_amplitudes() {
            let i = cps.iter().position(|cp| cp.coords == amp.source.coords).unwrap_or(0);
            let j = cps.iter().position(|cp| cp.coords == amp.target.coords).unwrap_or(0);
            mat[(i, j)] += amp.amplitude;
        }

        mat
    }

    /// Compute the instanton density at a point: sum of |A|² for all instantons
    /// passing through that point.
    pub fn instanton_density(&self, point: &[f64]) -> f64 {
        let amps = self.compute_amplitudes();
        let mut density = 0.0;

        for amp in &amps {
            // Distance from the point to the straight-line instanton path
            let s = &amp.source.coords;
            let t = &amp.target.coords;
            let path_len2: f64 = s.iter().zip(t.iter())
                .map(|(a, b)| (a - b).powi(2)).sum();

            if path_len2 < 1e-12 {
                continue;
            }

            // Project point onto the path
            let proj_param: f64 = s.iter().zip(t.iter()).zip(point.iter())
                .map(|((a, b), p)| (p - a) * (b - a)).sum::<f64>() / path_len2;
            let proj_param = proj_param.clamp(0.0, 1.0);

            let closest: Vec<f64> = s.iter().zip(t.iter())
                .map(|(a, b)| a + proj_param * (b - a)).collect();

            let dist2: f64 = closest.iter().zip(point.iter())
                .map(|(a, b)| (a - b).powi(2)).sum();

            // Gaussian density around the instanton path
            density += amp.probability() * (-dist2 / 0.1).exp();
        }

        density
    }

    /// Build the Morse-Witten chain complex from instanton amplitudes.
    /// The boundary map counts instantons with signs.
    pub fn morse_witten_boundary(&self, k: usize) -> DMatrix<f64> {
        let cps = &self.morse_function.critical_points;
        let cp_k: Vec<_> = cps.iter().filter(|cp| cp.index == k).collect();
        let cp_km1: Vec<_> = cps.iter().filter(|cp| cp.index == k - 1).collect();

        let mut bnd = DMatrix::zeros(cp_km1.len(), cp_k.len());

        for amp in self.compute_amplitudes() {
            if amp.source.index != k || amp.target.index != k - 1 {
                continue;
            }
            if let (Some(j), Some(i)) = (
                cp_k.iter().position(|cp| cp.coords == amp.source.coords),
                cp_km1.iter().position(|cp| cp.coords == amp.target.coords),
            ) {
                let sign = if amp.phase.abs() < 0.1 { 1.0 } else { -1.0 };
                bnd[(i, j)] += sign * amp.amplitude;
            }
        }

        bnd
    }

    /// Compute H¹ (first cohomology): dimension of spurious tunneling channels.
    /// H¹ = ker(∂₁) / im(∂₂) — measures reward hacking pathways.
    pub fn h1_dimension(&self) -> usize {
        let bnd1 = self.morse_witten_boundary(1);
        let bnd2 = self.morse_witten_boundary(2);

        let ker1 = bnd1.nrows() - bnd1.rank(1e-10);
        let im2 = bnd2.rank(1e-10);

        if ker1 > im2 { ker1 - im2 } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_landscape() -> MorseFunction {
        let cps = vec![
            CriticalPoint::labeled(vec![0.0, 0.0], 0, 0.0, "basin_A"),
            CriticalPoint::labeled(vec![2.0, 0.0], 0, 0.5, "basin_B"),
            CriticalPoint::labeled(vec![1.0, 0.0], 1, 1.0, "saddle_AB"),
            CriticalPoint::labeled(vec![1.0, 1.0], 2, 2.0, "peak"),
        ];
        MorseFunction::from_critical_points(2, cps)
    }

    #[test]
    fn test_instanton_amplitude_complex() {
        let amp = InstantonAmplitude {
            source: CriticalPoint::new(vec![0.0], 1, 1.0),
            target: CriticalPoint::new(vec![1.0], 0, 0.0),
            amplitude: 0.5,
            phase: std::f64::consts::PI / 4.0,
            action: 1.0,
            is_spurious: false,
        };
        let (re, im) = amp.complex_amplitude();
        assert!(re > 0.0);
        assert!(im > 0.0);
    }

    #[test]
    fn test_instanton_probability() {
        let amp = InstantonAmplitude {
            source: CriticalPoint::new(vec![0.0], 1, 1.0),
            target: CriticalPoint::new(vec![1.0], 0, 0.0),
            amplitude: 0.5,
            phase: 0.0,
            action: 1.0,
            is_spurious: false,
        };
        assert_eq!(amp.probability(), 0.25);
    }

    #[test]
    fn test_boltzmann_factor() {
        let amp = InstantonAmplitude {
            source: CriticalPoint::new(vec![0.0], 1, 1.0),
            target: CriticalPoint::new(vec![1.0], 0, 0.0),
            amplitude: 1.0,
            phase: 0.0,
            action: 1.0,
            is_spurious: false,
        };
        let bf = amp.boltzmann_factor();
        assert!(bf > 0.0 && bf < 1.0);
    }

    #[test]
    fn test_compute_amplitudes() {
        let mf = test_landscape();
        let calc = TunnelingCalculator::new(mf, 1.0);
        let amps = calc.compute_amplitudes();
        // saddle_AB (idx 1) → basin_A and basin_B (idx 0)
        // peak (idx 2) → saddle_AB (idx 1)
        assert!(amps.len() >= 1);
    }

    #[test]
    fn test_amplitudes_positive() {
        let mf = test_landscape();
        let calc = TunnelingCalculator::new(mf, 1.0);
        for amp in calc.compute_amplitudes() {
            assert!(amp.amplitude >= 0.0);
            assert!(amp.action >= 0.0);
        }
    }

    #[test]
    fn test_tunneling_matrix() {
        let mf = test_landscape();
        let calc = TunnelingCalculator::new(mf, 1.0);
        let mat = calc.tunneling_matrix();
        assert_eq!(mat.nrows(), 4); // 4 critical points
        assert_eq!(mat.ncols(), 4);
    }

    #[test]
    fn test_detect_reward_hacking() {
        let mf = test_landscape();
        let calc = TunnelingCalculator::new(mf, 1.0);
        let hacks = calc.detect_reward_hacking(0.5);
        // Might or might not find hacks depending on amplitudes
        // Just check it doesn't crash
        assert!(hacks.len() <= mf.critical_points.len());
    }

    #[test]
    fn test_instanton_density() {
        let mf = test_landscape();
        let calc = TunnelingCalculator::new(mf, 1.0);
        let density = calc.instanton_density(&[1.0, 0.0]);
        assert!(density >= 0.0);
    }

    #[test]
    fn test_morse_witten_boundary() {
        let mf = test_landscape();
        let calc = TunnelingCalculator::new(mf, 1.0);
        let bnd = calc.morse_witten_boundary(1);
        // Should have shape: (# index-0) × (# index-1)
        let n_idx0 = mf.critical_points.iter().filter(|cp| cp.index == 0).count();
        let n_idx1 = mf.critical_points.iter().filter(|cp| cp.index == 1).count();
        assert_eq!(bnd.nrows(), n_idx0);
        assert_eq!(bnd.ncols(), n_idx1);
    }

    #[test]
    fn test_h1_dimension() {
        let mf = test_landscape();
        let calc = TunnelingCalculator::new(mf, 1.0);
        let h1 = calc.h1_dimension();
        // H¹ should be non-negative
        assert!(h1 <= mf.critical_points.len());
    }

    #[test]
    fn test_large_t_suppression() {
        let mf = test_landscape();
        let calc_small = TunnelingCalculator::new(mf.clone(), 0.1);
        let calc_large = TunnelingCalculator::new(mf, 10.0);
        let amps_small: f64 = calc_small.compute_amplitudes().iter()
            .map(|a| a.amplitude).sum();
        let amps_large: f64 = calc_large.compute_amplitudes().iter()
            .map(|a| a.amplitude).sum();
        // Larger t should suppress tunneling
        assert!(amps_large <= amps_small + 1e-6);
    }

    #[test]
    fn test_instanton_serialization() {
        let amp = InstantonAmplitude {
            source: CriticalPoint::new(vec![0.0], 1, 1.0),
            target: CriticalPoint::new(vec![1.0], 0, 0.0),
            amplitude: 0.5,
            phase: 0.0,
            action: 1.0,
            is_spurious: false,
        };
        let json = serde_json::to_string(&amp).unwrap();
        let amp2: InstantonAmplitude = serde_json::from_str(&json).unwrap();
        assert_eq!(amp.amplitude, amp2.amplitude);
    }
}

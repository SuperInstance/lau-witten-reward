//! # lau-witten-reward
//!
//! Witten deformation on reward landscapes — the fermionic square root the
//! ecosystem was missing. Implements instanton tunneling between reward basins
//! via Morse theory.
//!
//! ## Core ideas
//!
//! - **Witten deformation**: Δ_t = e^{-tf} Δ e^{tf} localizes the low-lying
//!   Laplacian spectrum at critical points of a Morse function f (the reward
//!   landscape).
//! - **Dirac operator**: D = d + δ is the fermionic square root of the
//!   Hodge Laplacian (odd grading).
//! - **Instanton tunneling**: tunneling amplitudes between reward basins are
//!   computed via the Witten complex.
//! - **Morse inequalities**: the low-lying spectrum reproduces the Morse
//!   inequalities; H^k ≅ C^k (Morse cells).
//! - **Reward hacking = spurious tunneling**: H¹ captures spurious
//!   connections between reward basins.

mod morse;
mod witten;
mod dirac;
mod instanton;
mod reward;
mod supersymmetry;

pub use morse::{CriticalPoint, MorseFunction, MorseComplex};
pub use witten::{WittenDeformation, WittenLaplacian, WittenComplex};
pub use dirac::{DiracOperator, DiracSpectrum, FermionicGrade};
pub use instanton::{InstantonAmplitude, TunnelingCalculator};
pub use reward::{RewardLandscape, RewardBasin, PolicyEigenfunction};
pub use supersymmetry::{SUSYHamiltonian, WittenIndex, SUSYPair};

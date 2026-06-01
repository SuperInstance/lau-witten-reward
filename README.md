# lau-witten-reward

> Witten deformation on reward landscapes — the fermionic square root the ecosystem was missing.

## What This Does

This crate applies **Witten deformation** and **Morse theory** to AI reward landscapes. It treats the reward function as a Morse function on the state space, then uses the full machinery of differential topology to analyze its structure:

- **Where are the reward basins?** → Critical points of index 0 (minima of the reward landscape)
- **How do basins connect?** → Instanton tunneling between critical points of adjacent index
- **Is the agent reward-hacking?** → Spurious tunneling detected via H¹ (first cohomology)
- **Is the reward landscape supersymmetric?** → Dirac operator D = d + δ, with D² = Δ

The core insight: **reward hacking is spurious tunneling between reward basins that shouldn't be connected.** The Witten complex captures this: H¹ measures the dimension of illegitimate pathways through the reward landscape.

## Key Idea

Edward Witten's 1982 deformation Δ_t = e^{−tf} Δ e^{tf} takes the Hodge Laplacian and localizes it at the critical points of a Morse function f. As t → ∞:

- Eigenfunctions of Δ_t concentrate at critical points
- The number of low-lying eigenvalues equals the Morse count c_k at each index
- The Witten complex converges to the Morse complex
- Tunneling amplitudes between basins are computed as instanton actions

This crate implements this deformation for **reward functions** on discrete state spaces. The Morse function IS the reward landscape, and the Witten deformation reveals its topological structure.

The **Dirac operator** D = d + δ is the fermionic square root that the Laplacian D² = Δ misses. It carries a ℤ₂-grading (bosonic vs fermionic) and the eigenvalues come in ± pairs (supersymmetry). Breaking this pairing means something interesting happened in the reward landscape.

## Install

```toml
[dependencies]
lau-witten-reward = "0.1.0"
```

Or:

```sh
cargo add lau-witten-reward
```

Dependencies: `nalgebra` 0.33, `serde` + `serde_json`.

## Quick Start

```rust
use lau_witten_reward::*;

// Define a reward landscape with critical points
let cps = vec![
    CriticalPoint::labeled(vec![0.0, 0.0], 0, 0.0, "basin_A"),      // reward minimum
    CriticalPoint::labeled(vec![2.0, 0.0], 0, 0.5, "basin_B"),      // another basin
    CriticalPoint::labeled(vec![1.0, 0.0], 1, 1.0, "saddle_AB"),    // saddle between basins
    CriticalPoint::labeled(vec![1.0, 1.0], 2, 2.0, "peak"),         // reward maximum
];
let morse = MorseFunction::from_critical_points(2, cps);

// Compute gradient and Hessian at any point
let grad = morse.gradient(&[0.5, 0.5]);
let hess = morse.hessian(&[0.5, 0.5]);

// Build the Morse complex and check Morse inequalities
let mc = MorseComplex::from_morse_function(morse.clone());
let betti = mc.betti_numbers();
let euler = morse.euler_characteristic();
println!("Betti numbers: {:?}", betti);
println!("Euler characteristic: {}", euler);
println!("Morse inequalities hold: {}", mc.verify_weak_morse_inequalities());

// Witten deformation: study how spectrum localizes at critical points
let witten = WittenDeformation::new(morse.clone(), 50, 0.05);
let lap_t1 = witten.at_t(1.0);
let lap_t100 = witten.at_t(100.0);
println!("Low-lying eigenvalues at t=1: {:?}", lap_t1.low_lying_spectrum(50.0));

// Spectral flow: eigenvalue evolution as t varies
let flow = witten.spectral_flow(&[0.1, 1.0, 5.0, 20.0, 100.0]);

// Instanton tunneling between reward basins
let calc = TunnelingCalculator::new(morse.clone(), 1.0);
let amplitudes = calc.compute_amplitudes();
for amp in &amplitudes {
    println!("{} → {}: amplitude={:.4}, action={:.4}, probability={:.4}",
        amp.source.label.as_deref().unwrap_or("?"),
        amp.target.label.as_deref().unwrap_or("?"),
        amp.amplitude, amp.action, amp.probability());
}

// Detect reward hacking
let hacks = calc.detect_reward_hacking(0.5);
println!("Spurious tunneling (reward hacking): {} pathways", hacks.len());
println!("H¹ dimension (hacking channels): {}", calc.h1_dimension());

// Dirac operator and supersymmetry
let dirac = DiracOperator::from_boundary_maps(
    mc.boundary_maps.clone(), 2, 1.0
);
let spectrum = dirac.eigenvalues();
println!("Dirac eigenvalues: {:?}", spectrum.eigenvalues);
println!("SUSY pairing: {}", spectrum.check_susy_pairing(0.5));
println!("Anticommutation norm {{D, Γ}}: {:.6}", dirac.verify_anticommutation());
```

## API Reference

### `morse` — Morse Theory Primitives

| Type | Purpose |
|------|---------|
| `CriticalPoint` | Non-degenerate critical point with coords, Morse index, value, label |
| `MorseFunction` | Smooth Morse function on discrete state space |
| `MorseComplex` | Chain complex C_k with boundary maps ∂: C_k → C_{k−1} |

**`CriticalPoint`** — Classified as minimum (index 0), maximum (index = dim), or saddle. Supports labeling, serialization, and display.

**`MorseFunction`** — Constructed from critical points or grid values. Computes:
- `evaluate(point)` — RBF interpolation from critical points
- `gradient(point)` — finite-difference gradient
- `hessian(point)` — finite-difference Hessian
- `morse_counts()` — count of critical points per index
- `euler_characteristic()` — χ = Σ (−1)^k c_k

**`MorseComplex`** — Built from a Morse function with distance-weighted boundary maps:
- `betti_numbers()` — dim H_k = dim ker(∂_k) − dim im(∂_{k+1})
- `verify_weak_morse_inequalities()` — check c_k ≥ β_k
- `morse_polynomial_coeffs()` / `poincare_polynomial_coeffs()`

### `witten` — Witten Deformation

| Type | Purpose |
|------|---------|
| `WittenLaplacian` | Δ_t = −Δ + t²\|∇f\|² + tΔf on discrete grid |
| `WittenDeformation` | Parameterized family Δ_t for varying t |
| `WittenComplex` | Deformed complex converging to Morse complex as t → ∞ |

**`WittenLaplacian`** — The deformed operator:
- `eigenvalues()` — sorted spectrum (Gershgorin bounds)
- `low_lying_spectrum(threshold)` — eigenvalues below threshold
- `count_low_lying(threshold)` — should equal critical point count at large t

**`WittenDeformation`** — Studies the family:
- `at_t(t)` — get Laplacian at parameter t
- `spectral_flow(&[t_values])` — eigenvalue evolution
- `witten_complex(t)` — complex at parameter t
- `critical_point_localization(t)` — which critical points correspond to low eigenvalues

**`WittenComplex`** — The deformed chain complex:
- `witten_homology_ranks()` — ker/im at each degree
- `morse_limit_distance(&morse_complex)` — convergence measure

### `dirac` — Dirac Operator and Supersymmetry

| Type | Purpose |
|------|---------|
| `DiracOperator` | D = d + δ, the fermionic square root of Δ |
| `DiracSpectrum` | Eigenvalues + Witten index |
| `FermionicGrade` | ℤ₂ grading (Bosonic / Fermionic) |

**`DiracOperator`** — Built from boundary maps or form dimensions:
- `square()` — D² = Δ (the Hodge Laplacian)
- `eigenvalues()` → `DiracSpectrum`
- `grading_operator()` — Γ = (−1)^F diagonal matrix
- `verify_anticommutation()` — norm of {D, Γ}, should be ≈ 0

**`DiracSpectrum`** — Analysis of the fermionic spectrum:
- `zero_modes(threshold)` — kernel of D
- `positive_eigenvalues()` / `negative_eigenvalues()`
- `check_susy_pairing(tol)` — eigenvalues come in ± pairs?

### `instanton` — Tunneling and Reward Hacking Detection

| Type | Purpose |
|------|---------|
| `InstantonAmplitude` | Gradient flow line between critical points |
| `TunnelingCalculator` | Computes all instanton amplitudes |

**`InstantonAmplitude`** — A single tunneling event:
- `complex_amplitude()` — (re, im) of A·e^{iφ}
- `probability()` — |A|²
- `boltzmann_factor()` — e^{−S}

**`TunnelingCalculator`** — The main analysis tool:
- `compute_amplitudes()` — all instantons between adjacent-index critical points
- `detect_reward_hacking(threshold)` — spurious high-amplitude tunneling
- `tunneling_matrix()` — transition matrix T_{ij}
- `instanton_density(point)` — density at a location
- `morse_witten_boundary(k)` — boundary map from instanton counting
- `h1_dimension()` — dim H¹ = number of reward hacking channels

## How It Works

### Morse Function on Reward Landscapes

1. The reward function R(s) is treated as a Morse function f: M → ℝ.
2. Critical points (∇f = 0) correspond to reward basins (minima), peaks (maxima), and saddle points.
3. The Morse index of a critical point counts the number of "downward" directions.
4. The Morse complex C_k = ℤ^{c_k} has chain groups generated by index-k critical points.

### Witten Deformation

1. Deform the Hodge Laplacian: Δ_t = e^{−tf} Δ e^{tf}.
2. This adds a potential V = t²|∇f|² + tΔf that creates deep wells at critical points.
3. As t → ∞, the low-lying eigenfunctions localize at critical points.
4. The number of eigenvalues that remain O(1) as t → ∞ equals c_k at each index.
5. The Witten complex interpolates between de Rham cohomology (t → 0) and Morse homology (t → ∞).

### Instanton Tunneling

1. An instanton is a gradient flow line from a critical point of index k to one of index k−1.
2. The tunneling amplitude A ∝ exp(−S/2) where S = t·(Δf + dist²/2) is the Euclidean action.
3. Large action → suppressed tunneling. Small action → easy transitions between basins.
4. **Reward hacking** = high-amplitude tunneling between distant reward basins (large Δf but small action, meaning the agent found a shortcut).

### Dirac Operator and SUSY

1. D = d + δ acts on the space of differential forms.
2. D² = dδ + δd = Δ (the Hodge Laplacian).
3. The ℤ₂-grading Γ = (−1)^F satisfies {D, Γ} = 0.
4. Eigenvalues of D come in ± pairs (supersymmetry): for every state with eigenvalue +λ, there's one with −λ.
5. **SUSY breaking** = unpaired zero modes = Witten index ≠ 0.

## The Math

### Witten Deformation

The Witten-deformed Laplacian on a manifold with Morse function f:

```
Δ_t = −e^{−tf} div(e^{tf} grad) = −Δ + t²|∇f|² + t·Δf
```

This is a Schrödinger operator with potential V(x) = t²|∇f(x)|² + t·Δf(x). The potential has deep wells at critical points (where ∇f = 0), with well depth proportional to t².

### Morse Inequalities

For a Morse function on a d-dimensional manifold:

```
Weak: c_k ≥ β_k for all k
Strong: M(t) − P(t) = (1 + t)Q(t) where Q has non-negative coefficients
```

where c_k = Morse count at index k, β_k = Betti number, M(t) = Σ c_k t^k is the Morse polynomial, P(t) = Σ β_k t^k is the Poincaré polynomial.

### Instanton Amplitudes

The semiclassical instanton amplitude between critical points p and q:

```
A_{p→q} = C · exp(−S(p,q)/2)
S(p,q) = t · [f(p) − f(q) + ||p − q||²/2]
```

where S is the Euclidean action. Transition probability: P = |A|².

### Morse-Witten Boundary Maps

The boundary map ∂_k: C_k → C_{k−1} counts (signed) gradient flow lines:

```
∂_k(q) = Σ_{p: index(p) = k−1} n(p,q) · p
```

where n(p,q) counts the number of gradient flow lines from q to p (with orientation signs).

### H¹ and Reward Hacking

The first cohomology H¹ = ker(∂₁)/im(∂₂) measures "1-cycles that aren't boundaries." In the reward landscape context:

- H¹ = 0: All tunneling between basins is explained by gradient flow (legitimate).
- H¹ > 0: There exist spurious tunneling channels (reward hacking).

### Euler Characteristic

```
χ = Σ_{k=0}^{d} (−1)^k c_k = Σ_{k=0}^{d} (−1)^k β_k
```

The alternating sum of Morse counts equals the alternating sum of Betti numbers. This is a topological invariant — it doesn't change under smooth deformations of the reward function.

### Dirac Operator and SUSY

```
D = d + δ
D² = Δ (Hodge Laplacian)
Γ = (−1)^F (grading operator)
{D, Γ} = 0 (anticommutation)
[D², Γ] = 0 (D² preserves grading)
```

The Witten index: ΔW = tr((−1)^F e^{−tD²}) = n_{bosonic}^{(0)} − n_{fermionic}^{(0)}, which is independent of t and equals the Euler characteristic.

## Tests

52 unit tests across 4 modules. Run with:

```sh
cargo test
```

## License

MIT

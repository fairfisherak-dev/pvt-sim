# Numerical Methods

## Overview

This document specifies the numerical algorithms used throughout PVT-SIM. Each algorithm includes convergence criteria, failure modes, and fallback strategies.

---

## 1. Cubic Equation Solver

### Problem
Solve Z³ + c₂Z² + c₁Z + c₀ = 0 for compressibility factor.

### Method: Cardano's Formula (Analytical)

Transform to depressed cubic t³ + pt + q = 0 via substitution Z = t - c₂/3:

```
p = c₁ - c₂²/3
q = c₀ - c₁c₂/3 + 2c₂³/27
```

Discriminant: D = (q/2)² + (p/3)³

**Case 1: D > 0 (one real root)**
```
t = ∛(-q/2 + √D) + ∛(-q/2 - √D)
Z = t - c₂/3
```

**Case 2: D ≤ 0 (three real roots)**
```
r = √(-p³/27)
θ = arccos(-q / (2r))
t₁ = 2∛r · cos(θ/3)
t₂ = 2∛r · cos((θ + 2π)/3)
t₃ = 2∛r · cos((θ + 4π)/3)
```

### Root Selection
- **Liquid:** Select smallest positive real root > b (covolume)
- **Vapor:** Select largest positive real root
- **Supercritical:** Single real root (D > 0 case)

### Edge Cases
- If Z < B (below covolume), solution is non-physical—retry with different initialization
- If all roots are negative, EOS parameters may be inconsistent

### Implementation Notes
```python
def solve_cubic(c2: float, c1: float, c0: float) -> tuple[float, ...]:
    """
    Returns tuple of real roots in ascending order.
    Uses Cardano's formula with careful handling of near-zero discriminant.
    """
```

---

## 2. Rachford-Rice Solver

### Problem
Find vapor fraction nᵥ ∈ [0, 1] satisfying:

f(nᵥ) = Σᵢ zᵢ(Kᵢ - 1) / [1 + nᵥ(Kᵢ - 1)] = 0

### Method: Brent's Method (Bracketed Root Finding)

Brent's method combines bisection, secant, and inverse quadratic interpolation. It guarantees convergence for bracketed roots and typically converges faster than bisection.

### Bracket Determination
The valid bracket [nᵥ_min, nᵥ_max] where f changes sign:

```
nᵥ_min = max(0, max((Kᵢ·zᵢ - 1)/(Kᵢ - 1) for Kᵢ > 1))
nᵥ_max = min(1, min((1 - zᵢ)/(1 - Kᵢ) for Kᵢ < 1))
```

If nᵥ_min > nᵥ_max, no valid two-phase solution exists at these K-values.

### Convergence Criteria
- |f(nᵥ)| < 10⁻¹² (function value)
- |nᵥ_new - nᵥ_old| < 10⁻¹² (step size)
- Maximum 50 iterations

### Negative Flash Extension
For experiment simulations, allow nᵥ < 0 or nᵥ > 1:
- nᵥ < 0: Feed is subcooled liquid
- nᵥ > 1: Feed is superheated vapor

Extend bracket to [-1, 2] for negative flash mode.

### Implementation Notes
```python
def solve_rachford_rice(
    z: NDArray, 
    K: NDArray, 
    allow_negative: bool = False
) -> float:
    """
    Solve Rachford-Rice equation for vapor fraction.
    
    Raises
    ------
    RachfordRiceError
        If no valid bracket exists or iteration fails.
    """
```

---

## 3. Successive Substitution (Flash)

### Problem
Find K-values satisfying fugacity equality: fᵢᴸ = fᵢⱽ for all components.

### Algorithm

```
1. Initialize K from Wilson correlation:
   Kᵢ = (Pcᵢ/P) · exp[5.373(1 + ωᵢ)(1 - Tcᵢ/T)]

2. LOOP:
   a. Solve Rachford-Rice for nᵥ
   b. Calculate compositions:
      xᵢ = zᵢ / [1 + nᵥ(Kᵢ - 1)]
      yᵢ = Kᵢ · xᵢ
   c. Solve cubic EOS for Zᴸ(x) and Zⱽ(y)
   d. Calculate fugacity coefficients φᵢᴸ, φᵢⱽ
   e. Update K-values:
      Kᵢ_new = φᵢᴸ / φᵢⱽ
   f. Check convergence

3. Convergence criterion:
   Σᵢ (ln Kᵢ_new - ln Kᵢ_old)² < 10⁻¹⁰
   
   OR equivalently:
   Σᵢ (fᵢᴸ - fᵢⱽ)² < 10⁻²⁰
```

### Damping
Near critical point, successive substitution can oscillate. Apply damping:

```
Kᵢ_new = Kᵢ_old · (φᵢᴸ/φᵢⱽ)^ω

where ω = 1.0 initially, reduced to 0.5 if oscillation detected
```

### Failure Modes
1. **Trivial solution:** K → 1 for all components (x = y = z)
   - Detection: |Kᵢ - 1| < 10⁻⁶ for all i
   - Cause: Feed is single-phase; stability analysis should have caught this
   
2. **Divergence:** K values grow without bound
   - Detection: max(Kᵢ) > 10⁶ or min(Kᵢ) < 10⁻⁶
   - Recovery: Reinitialize with different K estimates

3. **Slow convergence:** > 50 iterations without convergence
   - Recovery: Switch to accelerated method (GDEM)

---

## 4. GDEM Acceleration

### Problem
Successive substitution converges slowly (linearly) near critical point where eigenvalues approach unity.

### Method: General Dominant Eigenvalue Method

Track iteration history to estimate dominant eigenvalue and extrapolate:

```
Given sequence: K⁽ⁿ⁻²⁾, K⁽ⁿ⁻¹⁾, K⁽ⁿ⁾

1. Compute differences:
   Δ₁ = ln K⁽ⁿ⁻¹⁾ - ln K⁽ⁿ⁻²⁾
   Δ₂ = ln K⁽ⁿ⁾ - ln K⁽ⁿ⁻¹⁾

2. Estimate eigenvalue:
   λ = (Δ₂ · Δ₂) / (Δ₁ · Δ₂)

3. If 0 < λ < 1 (convergent):
   Extrapolate: ln K_extrap = ln K⁽ⁿ⁾ + Δ₂ / (1 - λ)

4. Accept extrapolation if it reduces objective function
```

### Activation Criteria
- Apply after 5 successive substitution iterations
- Only when |λ| > 0.9 (slow convergence detected)

### Implementation Notes
Store last 3 K-value vectors for eigenvalue estimation.

---

## 5. Michelsen Stability Analysis

### Problem
Determine if mixture z at (P, T) is stable (single phase) or unstable (will split).

### Method: Tangent Plane Distance Minimization

The mixture is unstable if there exists any trial composition w where:

TPD(w) = Σᵢ wᵢ [ln wᵢ + ln φᵢ(w) - ln zᵢ - ln φᵢ(z)] < 0

### Algorithm

```
1. Evaluate reference fugacity:
   dᵢ = ln zᵢ + ln φᵢ(z)

2. For each trial (vapor-like and liquid-like):
   
   a. Initialize:
      Vapor trial: Wᵢ = zᵢ · Kᵢ (Wilson K)
      Liquid trial: Wᵢ = zᵢ / Kᵢ
   
   b. Normalize: wᵢ = Wᵢ / ΣWᵢ
   
   c. LOOP:
      - Solve EOS for Z(w)
      - Calculate φᵢ(w)
      - Update: Wᵢ_new = exp(dᵢ - ln φᵢ(w))
      - Check convergence: Σ(ln Wᵢ_new - ln Wᵢ_old)² < 10⁻¹⁰
   
   d. Evaluate TPD = Σᵢ Wᵢ(ln Wᵢ + ln φᵢ(w) - dᵢ) - 1

3. If any TPD < -10⁻⁸: mixture is UNSTABLE
   Otherwise: mixture is STABLE
```

### Multiple Trials
Beyond Wilson initialization, also try:
- Pure component limits (wᵢ = 1 for lightest, heaviest components)
- Random perturbations (for pathological cases)

### Return Values
- Stability status (bool)
- Converged trial compositions (for flash initialization)
- TPD values for diagnostics

---

## 6. Newton-Raphson with Line Search

### Problem
General nonlinear equation solving with quadratic convergence.

### Algorithm

```
Given f(x) = 0 with Jacobian J:

1. Evaluate f(x⁽ⁿ⁾), J(x⁽ⁿ⁾)

2. Solve linear system:
   J · Δx = -f
   
3. Line search for step length α ∈ (0, 1]:
   Find α that satisfies Armijo condition:
   ||f(x + αΔx)|| < ||f(x)|| · (1 - 10⁻⁴ · α)

4. Update: x⁽ⁿ⁺¹⁾ = x⁽ⁿ⁾ + α · Δx

5. Convergence: ||f|| < tol AND ||Δx|| < tol
```

### Applications
- Bubble/dew point pressure calculation
- Phase envelope tracing
- Regression parameter optimization

### Jacobian Calculation
- Analytical derivatives where feasible (EOS, fugacity)
- Finite difference fallback: ∂fᵢ/∂xⱼ ≈ [fᵢ(x + hεⱼ) - fᵢ(x)] / h

---

## 7. Saturation Point Calculation

### Problem
Find pressure P at fixed T where the first infinitesimal amount of a second
phase appears.

### Primary Method: Stability boundary (TPD root) on Pressure

For vapor–liquid equilibrium, a saturation point is the **stability boundary**
of the corresponding single-phase feed.

We define a scalar stability metric using Michelsen's TPD minimization:

- **Bubble point (liquid feed):** use the **vapor-like** trial
  \( f(P) = \mathrm{TPD}_{\text{vapor trial}}(P; z, T, \text{feed=liquid}) \)

- **Dew point (vapor feed):** use the **liquid-like** trial
  \( f(P) = \mathrm{TPD}_{\text{liquid trial}}(P; z, T, \text{feed=vapor}) \)

Solve \( f(P)=0 \) with a *bracketed* root-finder (Brent). This is robust and
prevents non-physical saturation roots that can occur if incipient-phase
compositions are approximated inconsistently.

### Optional Alternative: Newton on classical saturation equations

For diagnostic or research workflows, the classical relations
\(\sum K_i x_i = 1\) (bubble) and \(\sum y_i/K_i = 1\) (dew) can be used.
However, they require self-consistent K-values (including an iterated
incipient-phase composition). If this option is implemented, it must remain
explicitly selectable.

### Initialization
Wilson K-value estimate:
```
P_bubble_init = Σᵢ xᵢ · Pcᵢ · exp[5.373(1 + ωᵢ)(1 - Tcᵢ/T)]
P_dew_init = 1 / Σᵢ (yᵢ / Pcᵢ) · exp[-5.373(1 + ωᵢ)(1 - Tcᵢ/T)]
```

### Convergence
- |f| < 10⁻⁸
- Maximum 80 iterations (including bracketing expansion)

### Post-check (optional)
After solving for a saturation pressure P* via TPD(P)=0, an optional diagnostic check may be performed to ensure P* separates stable and unstable regions (a true stability boundary).

For bubble (liquid feed): TPD(P*+ΔP) ≥ 0 and TPD(P*-ΔP) < 0.
For dew (vapor feed): TPD(P*-ΔP) ≥ 0 and TPD(P*+ΔP) < 0.

If enabled, the implementation supports either **raise** (hard failure) or **warn** behavior, and uses a relative perturbation ΔP = max(1 kPa, rel_step·P*).

---

## 8. Phase Envelope Tracing

### Failure handling
Envelope tracing can terminate normally when no saturation point exists beyond cricondentherm (PhaseError reason=no_saturation). Other failures (numerical convergence, inconsistent phase state) may either raise immediately or return partial curves, depending on a selectable failure mode.

### Problem
Construct the complete phase envelope (bubble + dew curves) in P-T space.

### Method: Continuation with Natural Parameter

```
1. Start at low P on bubble point curve
2. Step along curve using (P, T, K) as variables
3. At each point, solve:
   - Saturation equation (bubble or dew)
   - K-value relations from fugacity equality
4. Switch from bubble to dew curve at critical point
5. Continue until returning to low P
```

### Adaptive Stepping
- Curvature-based step size control
- Smaller steps near critical point, cricondentherm, cricondenbar
- Detection of inflection points

### Critical Point Location
Where bubble and dew curves meet:
- Condition: Phase compositions become identical (x → y)
- Detection: Monitor |Σᵢ(xᵢ - yᵢ)²| → 0

---

## Convergence Tolerances Summary

| Calculation | Variable | Tolerance |
|-------------|----------|-----------|
| Cubic solver | Z | Machine precision |
| Rachford-Rice | nᵥ | 10⁻¹² |
| Flash (K-values) | ln K | 10⁻¹⁰ |
| Flash (fugacity) | f | 10⁻²⁰ (squared) |
| Stability (TPD) | TPD | -10⁻⁸ (threshold) |
| Saturation P | f, ΔP/P | 10⁻⁸, 10⁻⁶ |
| Confined Pc | ΔPc | 10⁻³ Pa |

---

## Error Recovery Strategy

```
Primary Method Failed?
         │
         ▼
    ┌─────────┐
    │ Damping │ (reduce step size)
    └────┬────┘
         │ Failed?
         ▼
    ┌─────────────┐
    │ Reinitialize│ (different starting point)
    └──────┬──────┘
           │ Failed?
           ▼
    ┌──────────────┐
    │ Switch Method│ (SS → GDEM → Newton)
    └───────┬──────┘
            │ Failed?
            ▼
    ┌────────────────┐
    │ Report Failure │ (with diagnostics)
    └────────────────┘
```

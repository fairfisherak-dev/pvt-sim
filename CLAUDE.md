# PVT-SIM: Production-Grade Phase Behavior Simulator

## Project Overview

A complete, from-scratch PVT simulator targeting commercial-grade functionality for reservoir fluid phase behavior analysis. This is not a student exercise—it's intended as a long-term, extensible tool that will eventually integrate as a callable module for larger systems (including voice assistant integration).

**Primary Goal:** Given a fluid composition at some (P, T), determine what phases exist, their compositions, and their properties.

**Distinguishing Feature:** Full nano-confinement phase behavior modeling with capillary pressure coupling—functionality lacking in most commercial software.

---

## Technical Stack

- **Language:** Python 3.11+
- **Core Dependencies:** NumPy, SciPy
- **Visualization:** Matplotlib
- **Testing:** pytest
- **Type Checking:** mypy (strict mode)
- **Documentation:** Sphinx + NumPy docstring format

---

## Architecture

```
src/pvtcore/           # Core library (no UI dependencies)
├── core/              # Constants, units, typing, errors, numerical methods
├── models/            # Data structures: Component, Mixture, Phase, Results
├── characterization/  # Plus-fraction splitting, SCN properties, lumping
├── correlations/      # Tb, Tc, Pc, Vc, ω, parachor correlations
├── eos/               # PR, SRK, cubic solver, mixing rules, volume translation
├── stability/         # Wilson K-values, Michelsen TPD
├── flash/             # Rachford-Rice, PT flash, saturation points, GDEM
├── envelope/          # Phase envelope tracing, critical point, quality lines
├── properties/        # Density, LBC viscosity, parachor IFT
├── experiments/       # CCE, DL, CVD, separator simulations
├── confinement/       # Capillary pressure, confined flash, shifted envelopes
├── tuning/            # EOS regression against experimental data
└── io/                # Import/export, report generation

src/pvtapp/            # Application layer (UI, plotting)
```

**Design Principle:** `pvtcore` must be fully usable without `pvtapp`. External systems (voice assistant, scripts) call `pvtcore` directly.

---

## Coding Standards

### Type Hints
All functions must have complete type annotations:
```python
def flash_pt(
    composition: NDArray[np.float64],
    pressure: float,
    temperature: float,
    eos: CubicEOS,
) -> FlashResult:
```

### Docstrings
NumPy format with parameters, returns, raises, and references:
```python
def pedersen_split(z_plus: float, mw_plus: float, gamma_plus: float) -> SCNDistribution:
    """
    Split plus fraction using Pedersen exponential distribution.

    Parameters
    ----------
    z_plus : float
        Mole fraction of plus fraction (e.g., 0.25 for 25% C7+)
    mw_plus : float
        Molecular weight of plus fraction [g/mol]
    gamma_plus : float
        Specific gravity of plus fraction (relative to water)

    Returns
    -------
    SCNDistribution
        Object containing z_n, MW_n, gamma_n for each SCN

    Raises
    ------
    CharacterizationError
        If material balance cannot be satisfied

    References
    ----------
    Pedersen, K.S., Thomassen, P., and Fredenslund, A. (1984).
    Ind. Eng. Chem. Process Des. Dev., 23(1), 163-170.
    """
```

### Units Convention
- **Temperature:** Kelvin (K) internally, convert at I/O boundaries
- **Pressure:** Pascal (Pa) internally, convert at I/O boundaries
- **Molar volume:** m³/mol
- **Density:** kg/m³
- **Molecular weight:** g/mol (kg/kmol)
- **Specific gravity:** dimensionless (relative to water at 60°F)

### Error Handling
Custom exceptions with meaningful messages:
```python
class ConvergenceError(PVTError):
    """Raised when iterative solver fails to converge."""

class CharacterizationError(PVTError):
    """Raised when plus-fraction splitting fails material balance."""
```

### Numerical Tolerances
Define as module constants, not magic numbers:
```python
FUGACITY_TOLERANCE: float = 1e-10
MAX_FLASH_ITERATIONS: int = 100
RACHFORD_RICE_TOLERANCE: float = 1e-12
```

---

## Key Equations

### Peng-Robinson EOS
```
P = RT/(V-b) - a·α(T) / [V(V+b) + b(V-b)]

a = 0.45724 · R²Tc² / Pc
b = 0.07780 · RTc / Pc
α(T) = [1 + κ(1 - √(T/Tc))]²
κ = 0.37464 + 1.54226ω - 0.26992ω²
```

### Fugacity Coefficient (PR)
```
ln(φᵢ) = (bᵢ/bₘ)(Z-1) - ln(Z-B) - [A/(2√2·B)] · [2ψᵢ/aₘ - bᵢ/bₘ] · ln[(Z + (1+√2)B) / (Z + (1-√2)B)]

where: ψᵢ = Σⱼ xⱼ·√(aᵢaⱼ)·(1 - kᵢⱼ)
```

### Rachford-Rice
```
f(nᵥ) = Σᵢ zᵢ(Kᵢ - 1) / [1 + nᵥ(Kᵢ - 1)] = 0
```

### Capillary Pressure (Confinement)
```
Pc = 2σ·cos(θ) / r

Modified equilibrium: fᵢᴸ(Pᴸ) = fᵢⱽ(Pⱽ) where Pⱽ = Pᴸ + Pc
```

---

## Validation Targets

| Milestone | Validation Source |
|-----------|-------------------|
| PR EOS | Pure component vapor pressures vs NIST |
| Flash | Published C₁-nC₄ VLE data |
| Phase envelope | Match MI PVT output for identical input |
| CCE/DL/CVD | Experimental data from literature |
| Confinement | Nojabaei et al. (2013) C₁-nC₄ results |

---

## Primary References

### Must-Implement Papers
1. **Peng & Robinson (1976)** - Primary EOS
2. **Rachford & Rice (1952)** - Flash material balance
3. **Michelsen (1982)** - Stability analysis (TPD method)
4. **Pedersen et al. (1984)** - Plus-fraction splitting
5. **Riazi & Daubert (1987)** - Critical property correlations
6. **Lohrenz, Bray & Clark (1964)** - Viscosity correlation
7. **Weinaug & Katz (1943)** - IFT for mixtures
8. **Nojabaei, Johns & Chu (2013)** - Nano-confinement

### Textbooks
- Pedersen, Christensen & Shaikh (2015) - "Phase Behavior of Petroleum Reservoir Fluids"
- Whitson & Brulé (2000) - "Phase Behavior" SPE Monograph

---

## Current Development Phase

**Phase 1: Component Database & Characterization Module**

Building the foundation that everything else depends on:
1. Pure component property database (N₂, CO₂, H₂S, C₁-C₁₀, isomers)
2. Correlation library (Tb, Tc, Pc, Vc, ω, parachor)
3. Plus-fraction splitting (Pedersen, Katz, Lohrenz methods)
4. Material balance enforcement
5. Lumping/delumping capability

---

## Working Conventions for Claude Code

1. **Read before writing.** Always inspect existing files before proposing changes.
2. **Propose approach first.** Explain strategy before generating code.
3. **One task at a time.** No scope creep.
4. **Verify correlations.** Check primary sources before implementing equations.
5. **Test incrementally.** Each function should have a corresponding test.
6. **Document units.** Every physical quantity must have clear unit documentation.

# PVT-SIM Architecture

## System Overview

PVT-SIM is a modular phase behavior simulator designed for both standalone use and integration into larger systems. The architecture separates concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    External Systems                         │
│              (Voice Assistant, Scripts, API)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      pvtapp Layer                           │
│         (UI, Plotting, Interactive Sessions)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     pvtcore Layer                           │
│              (All Computational Logic)                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │ charac- │ │   eos   │ │  flash  │ │ proper- │            │
│  │tertic.  │ │         │ │         │ │  ties   │            │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            │
│       │          │          │          │                    │
│       └──────────┴─────┬────┴──────────┘                    │
│                        │                                    │
│               ┌────────▼────────┐                           │
│               │     models      │                           │
│               │  (data structs) │                           │
│               └────────┬────────┘                           │
│                        │                                    │
│               ┌────────▼────────┐                           │
│               │      core       │                           │
│               │ (units, consts) │                           │
│               └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      data Layer                             │
│        (Component DB, Correlation Coefficients)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### core/

Foundation module with no dependencies on other pvtcore modules.

| File | Purpose |
|------|---------|
| `constants.py` | Physical constants (R, standard conditions, etc.) |
| `units.py` | Unit conversion functions, unit registry |
| `typing.py` | Type aliases, Protocol classes |
| `errors.py` | Custom exception hierarchy |
| `numerics/brent.py` | Brent's method for root finding |
| `numerics/newton.py` | Newton-Raphson with damping |
| `numerics/convergence.py` | Convergence criteria, iteration tracking |

### models/

Data structures representing physical entities. Immutable where possible.

| File | Purpose |
|------|---------|
| `component.py` | `Component` dataclass with Tc, Pc, ω, MW, etc. |
| `mixture.py` | `Mixture` class holding composition + component list |
| `eos_params.py` | EOS-specific parameters (a, b, BIPs) |
| `phase.py` | `Phase` with composition, Z-factor, density |
| `results.py` | Result containers for flash, envelope, experiments |

### characterization/

Fluid characterization from laboratory data.

| File | Purpose |
|------|---------|
| `plus_splitting/pedersen.py` | Exponential distribution: ln(zₙ) = A + B·MWₙ |
| `plus_splitting/katz.py` | Katz correlation: zₙ = 1.38205·z₇₊·exp(-0.25903n) |
| `plus_splitting/lohrenz.py` | Quadratic exponential |
| `plus_splitting/whitson_gamma.py` | Gamma distribution (optional) |
| `scn_properties.py` | Generalized SCN tables (Katz-Firoozabadi) |
| `lumping.py` | Whitson lumping with Lee mixing rules |
| `delumping.py` | K-value interpolation for full composition recovery |

### correlations/

Property estimation correlations for pseudo-components.

| File | Purpose |
|------|---------|
| `tb.py` | Boiling point: Soreide (1989) |
| `critical_props/riazi_daubert.py` | Tc, Pc, Vc from Tb, γ |
| `critical_props/kesler_lee.py` | Tc, Pc, ω from Tb, γ |
| `critical_props/cavett.py` | Alternative correlation |
| `acentric.py` | Edmister, Kesler-Lee methods |
| `parachor.py` | Fanchi correlation from MW |

### eos/

Equation of state implementations.

| File | Purpose |
|------|---------|
| `base.py` | Abstract `CubicEOS` protocol |
| `mixing_rules.py` | van der Waals mixing, BIP handling |
| `cubic_solver.py` | Cardano's formula, root selection |
| `peng_robinson.py` | PR (1976) implementation |
| `srk.py` | SRK (1972) implementation |
| `volume_translation.py` | Peneloux correction for density |

### stability/

Phase stability analysis.

| File | Purpose |
|------|---------|
| `wilson.py` | Wilson K-value correlation for initialization |
| `tpd.py` | Michelsen tangent plane distance method |

### flash/

Phase equilibrium calculations.

| File | Purpose |
|------|---------|
| `rachford_rice.py` | RR equation solver (Brent's method) |
| `pt_flash.py` | Isothermal flash at specified P, T |
| `saturation.py` | Bubble point, dew point calculations |
| `acceleration/gdem.py` | General dominant eigenvalue method |

### envelope/

Phase envelope construction.

| File | Purpose |
|------|---------|
| `trace.py` | Envelope tracing algorithm |
| `critical_point.py` | Critical point location |
| `quality_lines.py` | Iso-volume fraction curves |

### properties/

Transport and interfacial properties.

| File | Purpose |
|------|---------|
| `density.py` | From Z-factor with volume translation |
| `viscosity_lbc.py` | Lohrenz-Bray-Clark correlation |
| `ift_parachor.py` | Parachor method (Weinaug-Katz) |

### experiments/

Laboratory test simulations.

| File | Purpose |
|------|---------|
| `cce.py` | Constant Composition Expansion |
| `dl.py` | Differential Liberation |
| `cvd.py` | Constant Volume Depletion |
| `separators.py` | Multi-stage separator optimization |

### confinement/

Nano-confinement extensions.

| File | Purpose |
|------|---------|
| `capillary.py` | Capillary pressure from IFT and pore radius |
| `confined_flash.py` | Flash with Pⱽ = Pᴸ + Pc iteration |
| `confined_envelope.py` | Shifted phase envelope generation |

### tuning/

EOS parameter regression.

| File | Purpose |
|------|---------|
| `parameters.py` | Tunable parameter definitions, bounds |
| `objectives.py` | Objective function construction |
| `datasets.py` | Experimental data containers |
| `regression.py` | Levenberg-Marquardt optimizer |

### io/

Data import/export.

| File | Purpose |
|------|---------|
| `import_csv.py` | Parse CSV composition files |
| `import_excel.py` | Parse Excel PVT reports |
| `export_csv.py` | Write results to CSV |
| `report_templates/` | Standard PVT report formats |

---

## Data Flow: Flash Calculation

```
Input: composition z[], pressure P, temperature T
                    │
                    ▼
        ┌───────────────────────┐
        │  Stability Analysis   │
        │    (Michelsen TPD)    │
        └───────────┬───────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
     TPD ≥ 0             TPD < 0
     (stable)           (unstable)
          │                   │
          ▼                   ▼
    Return single      ┌─────────────┐
    phase (z, Z)       │  PT Flash   │
                       └──────┬──────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │ Liquid Phase │    │ Vapor Phase  │
            │   x[], Zᴸ    │    │   y[], Zⱽ    │
            └──────┬───────┘    └──────┬───────┘
                   │                   │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │    Properties     │
                   │  ρ, μ, σ for each │
                   └───────────────────┘
                             │
                             ▼
                      FlashResult
```

---

## Data Flow: Nano-Confined Flash

```
Input: composition z[], liquid pressure Pᴸ, temperature T, pore radius r
                    │
                    ▼
        ┌───────────────────────┐
        │   Bulk Flash (Pc=0)   │
        │     Get x, y, ρ       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Calculate IFT (σ)   │◄─────────┐
        │   from x, y, ρᴸ, ρⱽ   │          │
        └───────────┬───────────┘          │
                    │                      │
                    ▼                      │
        ┌───────────────────────┐          │
        │   Pc = 2σ/r           │          │
        └───────────┬───────────┘          │
                    │                      │
                    ▼                      │
        ┌───────────────────────┐          │
        │  Re-flash with        │          │
        │  Pⱽ = Pᴸ + Pc         │          │
        └───────────┬───────────┘          │
                    │                      │
                    ▼                      │
              |Pc_new - Pc_old|            │
                < tolerance?               │
                    │                      │
          ┌─────────┴─────────┐            │
          No                  Yes          │
          │                   │            │
          └───────────────────┘────────────┘
                              │
                              ▼
                    ConfinedFlashResult
```

---

## Interface Design

### Callable Module Pattern

For voice assistant and external system integration:

```python
from pvtcore import Fluid, FlashEngine

# Load fluid from composition
fluid = Fluid.from_composition({
    "N2": 0.005, "CO2": 0.012, "C1": 0.45, "C2": 0.08,
    "C3": 0.055, "C7+": {"z": 0.25, "MW": 215, "gamma": 0.85}
})

# Configure and run flash
engine = FlashEngine(eos="PR", splitting="pedersen")
result = engine.flash(fluid, P=2000e5, T=373.15)

# Access results
print(result.vapor_fraction)      # 0.342
print(result.liquid.density)      # 650.2 kg/m³
print(result.phases["vapor"].composition)  # NDArray
```

### Result Objects

All results are serializable dataclasses:

```python
@dataclass(frozen=True)
class FlashResult:
    converged: bool
    iterations: int
    vapor_fraction: float
    liquid: Phase
    vapor: Phase
    stability_info: StabilityResult
    
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
```

---

## Extensibility Points

1. **New EOS:** Implement `CubicEOS` protocol, register in factory
2. **New Correlations:** Add module to `correlations/`, register in dispatcher
3. **New Splitting Methods:** Add to `plus_splitting/`, implement `SplittingMethod` protocol
4. **New Lab Tests:** Add to `experiments/`, follow CCE/DL pattern
5. **Alternative Optimizers:** Swap in `tuning/regression.py`

---

## Performance Considerations

1. **Vectorization:** Use NumPy arrays for composition vectors; avoid Python loops over components
2. **Caching:** EOS parameters (a, b) don't change during flash iteration—compute once
3. **Lazy Evaluation:** Don't compute viscosity/IFT unless requested
4. **Profiling Points:** Flash inner loop, cubic solver, fugacity coefficient calculation

# Equation of State Module

This module provides implementations of cubic equations of state for phase behavior and thermodynamic property calculations.

## Overview

The EOS module implements the **Peng-Robinson (1976)** equation of state with full support for:
- Pure component calculations
- Multi-component mixtures with van der Waals mixing rules
- Binary interaction parameters
- Compressibility factor calculations
- Fugacity and fugacity coefficient calculations
- Density and molar volume calculations
- Departure functions (enthalpy, entropy, Gibbs energy)

## Peng-Robinson Equation of State

### Formulation

The Peng-Robinson EOS is expressed as:

```
P = RT/(V-b) - a(T)/(V² + 2bV - b²)
```

In compressibility factor form:
```
Z³ - (1-B)Z² + (A-2B-3B²)Z - (AB-B²-B³) = 0
```

where:
- `A = aP/(RT)²` - dimensionless attraction parameter
- `B = bP/(RT)` - dimensionless repulsion parameter

### Parameters

**Critical Point Parameters:**
```python
a_c = 0.45724 R²Tc²/Pc  # Critical attraction parameter
b = 0.07780 RTc/Pc      # Covolume parameter
```

**Temperature Dependence:**
```python
κ = 0.37464 + 1.54226ω - 0.26992ω²  # for ω ≤ 0.49
κ = 0.379642 + 1.48503ω - 0.164423ω² + 0.016666ω³  # for ω > 0.49

α(T) = [1 + κ(1 - √(T/Tc))]²
a(T) = a_c × α(T)
```

**Mixing Rules (van der Waals one-fluid):**
```python
a_mix = ΣΣ xᵢxⱼ√(aᵢaⱼ)(1-kᵢⱼ)
b_mix = Σ xᵢbᵢ
```

### Fugacity Coefficient

For component i in a mixture:

```python
ln(φᵢ) = (bᵢ/b_mix)(Z - 1) - ln(Z - B)
         - (A/(2√2 B)) × [2Σⱼ(xⱼaᵢⱼ)/a_mix - bᵢ/b_mix]
         × ln[(Z + (1+√2)B)/(Z + (1-√2)B)]
```

## Usage Examples

### Pure Component Calculations

```python
import numpy as np
from pvtcore.eos import PengRobinsonEOS
from pvtcore.models import load_components

# Load component database
components = load_components()

# Create PR EOS for methane
eos = PengRobinsonEOS([components['C1']])

# Define conditions
T = 300.0  # K
P = 5e6    # Pa (50 bar)
composition = np.array([1.0])

# Calculate compressibility factor
Z = eos.compressibility(P, T, composition, phase='vapor')
print(f"Z-factor: {Z:.4f}")

# Calculate fugacity coefficient
phi = eos.fugacity_coefficient(P, T, composition, phase='vapor')
print(f"Fugacity coefficient: {phi[0]:.4f}")

# Get complete results
result = eos.calculate(P, T, composition, phase='vapor')
print(f"Phase: {result.phase}")
print(f"Z: {result.Z:.4f}")
print(f"Fugacity coef: {result.fugacity_coef[0]:.4f}")
```

### Binary Mixture Calculations

```python
# Create PR EOS for methane-ethane mixture
binary_eos = PengRobinsonEOS([components['C1'], components['C2']])

# 50-50 mixture
composition = np.array([0.5, 0.5])

# With binary interaction parameter
kij = np.array([[0.0, 0.03],    # C1-C2 kij = 0.03
                [0.03, 0.0]])

Z = binary_eos.compressibility(
    P, T, composition, phase='vapor',
    binary_interaction=kij
)

phi = binary_eos.fugacity_coefficient(
    P, T, composition, phase='vapor',
    binary_interaction=kij
)

print(f"Mixture Z-factor: {Z:.4f}")
print(f"C1 fugacity coef: {phi[0]:.4f}")
print(f"C2 fugacity coef: {phi[1]:.4f}")
```

### Density and Volume Calculations

```python
# Molar density (mol/m³)
rho = eos.density(P, T, composition, phase='vapor')

# Molar volume (m³/mol)
V = eos.molar_volume(P, T, composition, phase='vapor')

print(f"Density: {rho:.2f} mol/m³")
print(f"Molar volume: {V*1e6:.2f} cm³/mol")
```

### Two-Phase Calculations

```python
# At conditions below critical temperature
T = 150.0  # K (below Tc of methane)
P = 2e6    # Pa

# Get all roots (liquid, unstable, vapor)
roots = eos.compressibility(P, T, composition, phase='auto')

if len(roots) == 3:
    Z_liquid = min(roots)
    Z_vapor = max(roots)
    print(f"Liquid Z: {Z_liquid:.4f}")
    print(f"Vapor Z: {Z_vapor:.4f}")
```

### Departure Functions

```python
# Calculate enthalpy, entropy, and Gibbs energy departures
departure = eos.calculate_departure_functions(
    P, T, composition, phase='vapor'
)

print(f"H departure: {departure['enthalpy_departure']:.2f} J/mol")
print(f"S departure: {departure['entropy_departure']:.4f} J/(mol·K)")
print(f"G departure: {departure['gibbs_departure']:.2f} J/mol")
```

## API Reference

### PengRobinsonEOS Class

**Constructor:**
```python
PengRobinsonEOS(components: List[Component])
```

**Key Methods:**

- `compressibility(P, T, composition, phase='vapor', binary_interaction=None)` → float
  - Calculate compressibility factor Z

- `fugacity_coefficient(P, T, composition, phase='vapor', binary_interaction=None)` → np.ndarray
  - Calculate fugacity coefficients for all components

- `fugacity(P, T, composition, phase='vapor', binary_interaction=None)` → np.ndarray
  - Calculate fugacities: fᵢ = φᵢ × xᵢ × P

- `calculate(P, T, composition, phase='vapor', binary_interaction=None)` → EOSResult
  - Complete EOS calculation returning EOSResult object

- `density(P, T, composition, phase='vapor', binary_interaction=None)` → float
  - Calculate molar density (mol/m³)

- `molar_volume(P, T, composition, phase='vapor', binary_interaction=None)` → float
  - Calculate molar volume (m³/mol)

- `calculate_departure_functions(P, T, composition, phase='vapor', binary_interaction=None)` → dict
  - Calculate thermodynamic departure functions

**Parameters:**
- `P`: Pressure (Pa)
- `T`: Temperature (K)
- `composition`: Mole fractions (numpy array)
- `phase`: 'liquid', 'vapor', or 'auto'
- `binary_interaction`: kij matrix (n×n numpy array), optional

### EOSResult Dataclass

Contains complete results from EOS calculation:

```python
@dataclass
class EOSResult:
    Z: float | np.ndarray          # Compressibility factor(s)
    phase: str                      # 'liquid', 'vapor', or 'two-phase'
    fugacity_coef: np.ndarray      # Fugacity coefficients
    A: float                        # Dimensionless attraction parameter
    B: float                        # Dimensionless repulsion parameter
    a_mix: float                    # Mixture attraction parameter (J·m³/mol²)
    b_mix: float                    # Mixture covolume (m³/mol)
    roots: List[float]              # All cubic equation roots
    pressure: float                 # Pressure (Pa)
    temperature: float              # Temperature (K)
```

## Cubic Equation Solver

The module includes a robust cubic equation solver using Cardano's formula:

```python
from pvtcore.core.numerics import solve_cubic, select_root

# Solve Z³ + c₂Z² + c₁Z + c₀ = 0
roots = solve_cubic(c2, c1, c0)

# Select appropriate root for phase
Z_vapor = select_root(roots, root_type='vapor')
Z_liquid = select_root(roots, root_type='liquid')
```

## Validation

The implementation has been validated against:

1. **Critical Point**: Z_c = 0.307 (analytical PR value)
2. **Ideal Gas Limit**: Z → 1 as P → 0
3. **Literature Values**: Within 1-2% of NIST/REFPROP for typical conditions
4. **Thermodynamic Consistency**: Satisfies Gibbs-Duhem relation
5. **Vieta's Formulas**: Cubic roots satisfy algebraic constraints

## Testing

Run comprehensive tests:

```bash
# Install dependencies
pip install numpy pytest

# Run all EOS tests
pytest tests/unit/test_cubic_solver.py -v
pytest tests/unit/test_peng_robinson.py -v
```

Test coverage includes:
- 40+ tests for cubic solver (Cardano's formula)
- 60+ tests for PR EOS (pure components, mixtures, limits)
- Numerical stability tests
- Literature value comparisons

## Performance Notes

- Pure component calculations: ~0.1 ms
- Binary mixture calculations: ~0.2 ms
- Cubic equation solving: ~10 µs (using analytical Cardano's formula)

## References

1. Peng, D.-Y. and Robinson, D. B., "A New Two-Constant Equation of State",
   *Industrial & Engineering Chemistry Fundamentals*, 15(1), 59-64 (1976).

2. Robinson, D. B. and Peng, D.-Y., "The Characterization of the Heptanes and Heavier Fractions for the GPA Peng-Robinson Programs",
   GPA Research Report RR-28 (1978).

3. Michelsen, M. L. and Mollerup, J. M., "Thermodynamic Models: Fundamentals & Computational Aspects", 2nd Ed., Tie-Line Publications (2007).

4. Whitson, C. H. and Brulé, M. R., "Phase Behavior", SPE Monograph Vol. 20 (2000).

## Future Enhancements

Planned additions:
- Soave-Redlich-Kwong (SRK) EOS
- Volume translation for improved liquid density
- Temperature-dependent binary interaction parameters
- Phase stability analysis
- Flash calculations (PT, VT, PS flashes)
- Critical point calculations

## Dependencies

- **Required**: numpy (≥1.20.0)
- **Optional**: scipy (for advanced calculations)

## License

Part of the pvt-sim package. See main repository for license details.

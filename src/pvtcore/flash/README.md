# Flash Calculation Engine

This module provides a complete flash calculation engine for vapor-liquid equilibrium (VLE) calculations in hydrocarbon systems.

## Overview

The flash calculation engine implements:
- **Wilson K-value correlation** for initialization
- **Rachford-Rice equation** solver using Brent's method
- **Successive substitution PT flash** algorithm
- **Phase stability testing** (simplified)

## Components

### 1. Wilson K-Values (`stability/wilson.py`)

Initial estimates for equilibrium ratios without requiring EOS calculations.

**Correlation:**
```
Ki = (Pci/P) × exp[5.373(1 + ωi)(1 - Tci/T)]
```

**Physical Meaning:**
- K > 1: Component prefers vapor phase (volatile)
- K < 1: Component prefers liquid phase (heavy)
- K = 1: Equal distribution

**Example:**
```python
from pvtcore.stability import wilson_k_values
from pvtcore.models import load_components

components = load_components()
binary = [components['C1'], components['C10']]

# Calculate Wilson K-values
K = wilson_k_values(
    pressure=3e6,      # 30 bar
    temperature=300,   # K
    components=binary
)

print(f"C1 K-value: {K[0]:.2f}")  # > 1 (light)
print(f"C10 K-value: {K[1]:.4f}") # < 1 (heavy)
```

### 2. Rachford-Rice Solver (`flash/rachford_rice.py`)

Solves the material balance equation to find vapor fraction.

**Equation:**
```
f(nv) = Σ zi(Ki - 1) / (1 + nv(Ki - 1)) = 0
```

**Solution Method:**
- Brent's method (robust, efficient)
- Automatic bracketing
- Handles edge cases (all vapor/liquid)

**Example:**
```python
from pvtcore.flash import solve_rachford_rice
import numpy as np

K = np.array([3.0, 0.5])  # Light and heavy components
z = np.array([0.6, 0.4])  # Feed composition

# Solve for vapor fraction and compositions
nv, x, y = solve_rachford_rice(K, z)

print(f"Vapor fraction: {nv:.3f}")
print(f"Liquid comp: {x}")
print(f"Vapor comp: {y}")

# Verify material balance
z_calc = (1-nv)*x + nv*y
assert np.allclose(z_calc, z)
```

### 3. PT Flash Algorithm (`flash/pt_flash.py`)

Complete flash calculation using successive substitution.

**Algorithm:**
1. Initialize K-values (Wilson correlation)
2. Solve Rachford-Rice for nv, x, y
3. Calculate φ_L and φ_V using EOS
4. Update K-values: Ki = φi_L / φi_V
5. Check convergence: Σ(ln Ki_new - ln Ki_old)² < tol
6. Repeat until converged

**Convergence Criterion:**
```
Σ(ln Ki_new - ln Ki_old)² < 1e-10
```

**Example:**
```python
from pvtcore.flash import pt_flash
from pvtcore.eos import PengRobinsonEOS
from pvtcore.models import load_components
import numpy as np

# Set up system
components = load_components()
binary = [components['C1'], components['C10']]
eos = PengRobinsonEOS(binary)

# Flash conditions
T = 300.0  # K
P = 3e6    # Pa (30 bar)
z = np.array([0.5, 0.5])

# Perform flash
result = pt_flash(P, T, z, binary, eos)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Vapor fraction: {result.vapor_fraction:.3f}")
print(f"Liquid: {result.liquid_composition}")
print(f"Vapor: {result.vapor_composition}")
print(f"K-values: {result.K_values}")
```

## FlashResult Dataclass

Complete results from flash calculation:

```python
@dataclass
class FlashResult:
    converged: bool              # Convergence flag
    iterations: int              # Number of iterations
    vapor_fraction: float        # nv (0 to 1)
    liquid_composition: ndarray  # x (mole fractions)
    vapor_composition: ndarray   # y (mole fractions)
    K_values: ndarray           # Ki = yi/xi
    liquid_fugacity: ndarray    # φi_L
    vapor_fugacity: ndarray     # φi_V
    phase: str                  # 'two-phase', 'vapor', or 'liquid'
    pressure: float             # P (Pa)
    temperature: float          # T (K)
    feed_composition: ndarray   # z
    residual: float             # Final residual
```

## Complete Workflow Example

```python
from pvtcore.eos import PengRobinsonEOS
from pvtcore.flash import pt_flash
from pvtcore.models import load_components
import numpy as np

# 1. Define system
components = load_components()
system = [components['C1'], components['C3'], components['C10']]
eos = PengRobinsonEOS(system)

# 2. Set conditions
T = 300.0  # K
P = 3e6    # Pa (30 bar)
z = np.array([0.5, 0.3, 0.2])  # Methane, propane, decane

# 3. Optional: Binary interaction parameters
kij = np.array([
    [0.0,  0.02, 0.05],
    [0.02, 0.0,  0.03],
    [0.05, 0.03, 0.0]
])

# 4. Perform flash
result = pt_flash(
    pressure=P,
    temperature=T,
    composition=z,
    components=system,
    eos=eos,
    binary_interaction=kij,
    tolerance=1e-10,
    max_iterations=100
)

# 5. Extract results
if result.converged:
    print("Flash Converged!")
    print(f"Phase: {result.phase}")
    print(f"Vapor fraction: {result.vapor_fraction:.3f}")

    if result.phase == 'two-phase':
        print("\nLiquid phase:")
        for i, comp in enumerate(system):
            print(f"  {comp.name}: {result.liquid_composition[i]:.4f}")

        print("\nVapor phase:")
        for i, comp in enumerate(system):
            print(f"  {comp.name}: {result.vapor_composition[i]:.4f}")

        # Verify fugacity equality
        f_L = result.liquid_fugacity * result.liquid_composition * P
        f_V = result.vapor_fugacity * result.vapor_composition * P

        print("\nFugacity equality check:")
        for i, comp in enumerate(system):
            print(f"  {comp.name}: f_L={f_L[i]:.2e}, f_V={f_V[i]:.2e}")
else:
    print(f"Flash failed to converge after {result.iterations} iterations")
```

## Special Cases

### Single-Phase Systems

```python
# All vapor (high K-values)
K = np.array([3.0, 4.0, 5.0])  # All > 1
z = np.array([0.5, 0.3, 0.2])

result = pt_flash(P, T, z, components, eos)
assert result.phase == 'vapor'
assert result.vapor_fraction == 1.0

# All liquid (low K-values)
P_high = 50e6  # Very high pressure
result = pt_flash(P_high, T, z, components, eos)
assert result.phase == 'liquid'
assert result.vapor_fraction == 0.0
```

### Pure Component Flash

```python
# Pure methane two-phase region
T = 150.0  # Below Tc
P = 2e6
z = np.array([1.0])

result = pt_flash(P, T, z, [components['C1']], eos)
print(f"Vapor fraction: {result.vapor_fraction:.3f}")
```

## Validation and Testing

The flash engine includes comprehensive tests:

### Wilson K-Values Tests
- Light components have K > 1
- Heavy components have K < 1
- K decreases with pressure
- K increases with temperature
- Trivial solution detection

### Rachford-Rice Tests
- Solves simple binary systems
- Handles edge cases (all vapor/liquid)
- Material balance verification
- Monotonicity of objective function

### PT Flash Tests
- Convergence for binary/ternary systems
- Light component enrichment in vapor
- K-value consistency
- Fugacity equality at equilibrium
- Special cases (near critical, pure components)

## Performance

Typical performance metrics:

| System | Conditions | Iterations | Time |
|--------|-----------|-----------|------|
| Binary (C1-C10) | 300K, 30 bar | 5-10 | ~2 ms |
| Ternary | 300K, 30 bar | 8-15 | ~5 ms |
| Near critical | 0.99 Tc | 15-25 | ~8 ms |

## Convergence

The successive substitution algorithm typically converges in 5-20 iterations for:
- Well-behaved systems away from critical point
- Moderate pressure (Pr < 0.8)
- Good initial estimates (Wilson K-values)

**Convergence Acceleration:**
- Under-relaxation in early iterations (α = 0.5)
- Less damping in later iterations (α = 0.7)
- Automatic Wilson initialization

**Difficult Cases:**
- Near critical point: May require 20-50 iterations
- High pressure: May need tighter tolerances
- Multiple phases: May need stability analysis

## Error Handling

The flash calculator includes robust error handling:

```python
from pvtcore.core.errors import (
    ValidationError,
    ConvergenceError,
    PhaseError
)

try:
    result = pt_flash(P, T, z, components, eos)
except ValidationError as e:
    # Invalid input (composition sum, negative P/T, etc.)
    print(f"Input error: {e}")
    print(f"Details: {e.details}")
except ConvergenceError as e:
    # Failed to converge
    print(f"Convergence failure: {e}")
    print(f"Iterations: {e.details['iterations']}")
    print(f"Residual: {e.details['residual']}")
except PhaseError as e:
    # Phase determination failed
    print(f"Phase error: {e}")
```

## References

1. **Wilson Correlation:**
   Wilson, G. M., "A Modified Redlich-Kwong Equation of State", AIChE (1968).

2. **Rachford-Rice Equation:**
   Rachford, H. H. and Rice, J. D., "Procedure for Use of Electronic Digital
   Computers in Calculating Flash Vaporization", JPT, 4(10) (1952).

3. **Successive Substitution:**
   Michelsen, M. L. and Mollerup, J. M., "Thermodynamic Models", 2nd Ed. (2007).

4. **Phase Equilibrium:**
   Whitson, C. H. and Brulé, M. R., "Phase Behavior", SPE Monograph Vol. 20 (2000).

## Future Enhancements

Planned additions:
- Newton-Raphson flash (faster convergence)
- Negative flash (PT to PS conversion)
- Stability analysis (full TPD method)
- Three-phase flash
- Temperature-dependent kij
- Critical point calculations

## Dependencies

- **Required**: numpy, component database, EOS module
- **Recommended**: scipy (for advanced solvers)

## Testing

Run comprehensive tests:

```bash
# Install dependencies
pip install numpy pytest

# Run all flash tests
pytest tests/unit/test_flash.py -v

# Run specific test class
pytest tests/unit/test_flash.py::TestPTFlash -v

# Run with coverage
pytest tests/unit/test_flash.py --cov=src/pvtcore/flash
```

## License

Part of the pvt-sim package. See main repository for license details.

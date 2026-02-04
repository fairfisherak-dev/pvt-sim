# PVT Core Foundation Module

This module provides the foundational elements for PVT calculations in the pvtcore package.

## Modules

### constants.py

Physical and chemical constants for thermodynamic calculations.

**Key Features:**
- Universal gas constant R in multiple unit systems (J/mol·K, bar·L/mol·K, psia·ft³/lbmol·R, etc.)
- Standard conditions definitions (STP, NTP, SC, SATP)
- Physical constants (Avogadro's number, Boltzmann constant, etc.)
- Unit conversion factors

**Example Usage:**
```python
from pvtcore.core import constants

# Use gas constant in different units
print(f"R = {constants.R.J_per_mol_K} J/(mol·K)")
print(f"R = {constants.R.bar_L_per_mol_K} bar·L/(mol·K)")

# Access standard conditions
print(f"STP: T = {constants.STP.T} K, P = {constants.STP.P} Pa")

# Get standard condition by name
sc = constants.get_standard_condition('SC_IMPERIAL')
print(f"{sc.description}")
```

### units.py

Unit conversion functions for common PVT calculations.

**Supported Conversions:**
- **Temperature**: K ↔ °C ↔ °F ↔ °R
- **Pressure**: Pa ↔ bar ↔ psi ↔ atm ↔ MPa ↔ kPa ↔ Torr
- **Volume**: m³ ↔ L ↔ ft³ ↔ bbl ↔ gal, including molar volumes
- **Mass/Molar**: kg ↔ lb, mol ↔ lbmol
- **Energy**: J ↔ cal ↔ BTU ↔ kWh
- **Density**: specific gravity ↔ API gravity ↔ density

**Example Usage:**
```python
from pvtcore.core import units

# Temperature conversions
temp_k = units.celsius_to_kelvin(25)  # 298.15 K
temp_f = units.kelvin_to_fahrenheit(298.15)  # 77°F

# Pressure conversions
p_pa = units.bar_to_pa(1.0)  # 100000 Pa
p_psi = units.pa_to_psi(101325)  # 14.696 psi

# Volume conversions
v_L = units.m3_to_liter(1.0)  # 1000 L
v_ft3 = units.m3_to_ft3(1.0)  # 35.3147 ft³

# API gravity conversions
sg = units.api_to_sg(35)  # Specific gravity from API
api = units.sg_to_api(0.85)  # API from specific gravity
```

### errors.py

Custom exception hierarchy for error handling.

**Exception Classes:**
- `PVTError` - Base exception for all PVT errors
- `ConvergenceError` - Iterative calculation convergence failures
- `CharacterizationError` - Fluid characterization errors
- `CompositionError` - Invalid composition data
- `PhaseError` - Phase equilibrium calculation errors
- `ValidationError` - Input validation failures
- `EOSError` - Equation of state errors
- `PropertyError` - Property calculation errors
- `DataError` - Data loading/processing errors
- `UnitError` - Unit conversion errors
- `ConfigurationError` - Configuration errors

**Example Usage:**
```python
from pvtcore.core import errors

# Raise specific error with context
raise errors.ConvergenceError(
    "Newton-Raphson failed to converge",
    iterations=100,
    residual=1e-3,
    tolerance=1e-6
)

# Catch base exception
try:
    # Some PVT calculation
    pass
except errors.PVTError as e:
    print(f"PVT Error: {e}")
    print(f"Details: {e.details}")
```

## Public API

The core module exports commonly used constants and functions:

```python
from pvtcore.core import (
    # Constants
    R, STP, NTP, SC_IMPERIAL, SC_METRIC,

    # Temperature conversions
    celsius_to_kelvin, kelvin_to_celsius,
    fahrenheit_to_kelvin, kelvin_to_fahrenheit,

    # Pressure conversions
    bar_to_pa, pa_to_bar, psi_to_pa, pa_to_psi,

    # Volume conversions
    liter_to_m3, m3_to_liter, ft3_to_m3, m3_to_ft3,

    # Exceptions
    PVTError, ConvergenceError, ValidationError,
)
```

## Design Philosophy

1. **SI Units as Standard**: All internal calculations use SI units (K, Pa, m³/mol, J/mol·K)
2. **Explicit Conversions**: Unit conversions are explicit functions, not implicit
3. **Type Safety**: Type hints throughout for better IDE support
4. **Clear Error Messages**: Exceptions include context via details dictionary
5. **NumPy Optional**: Core functionality works without NumPy, but supports arrays when available

## Dependencies

- **Required**: None (standard library only for basic functionality)
- **Optional**: NumPy (for array support in unit conversions)

## Testing

The core module is thoroughly tested. To run tests:

```bash
pytest tests/unit/test_components.py -v
```

## Notes

- All constants are based on CODATA 2018 recommended values
- Standard conditions follow IUPAC and petroleum industry conventions
- Unit conversions maintain high precision (no rounding internally)
- Error classes support rich context via the `details` dictionary

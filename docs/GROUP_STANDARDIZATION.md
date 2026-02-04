# PPR78 Group Standardization

## Overview

The component database has been standardized to use a consistent group basis for PPR78 temperature-dependent k_ij(T) calculations. This ensures predictable behavior and extensibility across all components.

## Canonical Group Set

### 1. Structural Groups (Hydrocarbons)
For alkanes, isoalkanes, and substituents on rings/aromatics:
- `CH4` - Methane (whole molecule exception)
- `CH3` - Methyl group (-CH₃)
- `CH2` - Methylene group (-CH₂-)
- `CH` - Methine group (>CH-)
- `C` - Quaternary carbon (>C<)

### 2. Cycloalkane Groups
- `CH2_cyclic` - Cyclic methylene
- `CHcyclic` - Cyclic methine
- `Ccyclic` - Cyclic quaternary carbon (rare)

### 3. Aromatic Groups
- `CHaro` - Aromatic CH (benzene ring carbons with H)
- `Caro` - Substituted aromatic carbon (ring carbons with substituents)

### 4. Sulfur Groups
- `SH` - Thiol/mercaptan group (-SH)
- `S` - Sulfide linkage (-S-)

### 5. Special Species (Whole Molecules)
These are retained as special groups because they represent complete small molecules:
- Inorganics: `N2`, `CO2`, `H2S`, `H2`, `CO`, `He`, `Ar`, `H2O`, `SO2`
- Sulfur compounds: `CS2`, `COS`

## Key Changes

### Ethane (C2) Normalization
**Before:** `{"C2H6": 1}` (whole molecule basis)
**After:** `{"CH3": 2}` (structural basis)

**Rationale:** Consistency with C3+ alkanes, which all use structural basis (CH3 + CH2). While C2H6 exists in published PPR78 interaction parameters, using the structural basis:
1. Maintains consistency across all alkanes ≥ C2
2. Extends naturally to larger molecules
3. Simplifies validation and group counting

**Impact on k_ij:** Using CH3-based interactions instead of C2H6-based interactions will produce slightly different k_ij values for ethane-containing pairs, but maintains thermodynamic consistency and follows the group-contribution philosophy.

### Methane (C1) - No Change
Methane retains the whole-molecule group `CH4` because:
1. It's a fundamental building block with unique behavior
2. Published PPR78 parameters use CH4 extensively
3. Cannot be further decomposed into structural groups

## Deprecation

### C2H6 Group
- **Status:** Deprecated for new components
- **Backward Compatibility:** Retained in `PPR78Group` enum for compatibility with published interaction parameters
- **Recommendation:** New component definitions should use `{"CH3": 2}` instead

## Examples

### Normal Alkanes
```json
"C1": {"groups": {"CH4": 1}},
"C2": {"groups": {"CH3": 2}},
"C3": {"groups": {"CH3": 2, "CH2": 1}},
"C4": {"groups": {"CH3": 2, "CH2": 2}},
"C10": {"groups": {"CH3": 2, "CH2": 8}}
```

### Branched Alkanes
```json
"iC4": {"groups": {"CH3": 3, "CH": 1}},
"iC5": {"groups": {"CH3": 3, "CH2": 1, "CH": 1}},
"neoC5": {"groups": {"CH3": 4, "C": 1}}
```

### Aromatics
```json
"BENZENE": {"groups": {"CHaro": 6}},
"TOLUENE": {"groups": {"CHaro": 5, "Caro": 1, "CH3": 1}},
"O_XYLENE": {"groups": {"CHaro": 4, "Caro": 2, "CH3": 2}}
```

### Cycloalkanes
```json
"CYCLOHEXANE": {"groups": {"CH2_cyclic": 6}},
"MCYCLOHEXANE": {"groups": {"CH2_cyclic": 5, "CHcyclic": 1, "CH3": 1}}
```

### Special Species
```json
"N2": {"groups": {"N2": 1}},
"CO2": {"groups": {"CO2": 1}},
"H2S": {"groups": {"H2S": 1}}
```

### Polar Components (No Standard Groups)
```json
"METHANOL": {"groups": {}},
"ETHANOL": {"groups": {}}
```
These components have no standard PPR78 groups. The k_ij defaults to zero for pairs involving these components.

## Validation Rules

The component validator enforces:
1. **No unknown keys:** All group names must be in the canonical set
2. **Non-negative integers:** Group counts must be ≥ 0
3. **No mixed bases:** C2H6 should not appear in new components
4. **Ring sanity:** Cyclohexane uses only CH2_cyclic, substituted cycloalkanes may include CHcyclic
5. **Aromatic sanity:** Aromatic ring carbons use only CHaro/Caro, substituents use CH3/CH2/CH/C

## Implementation Files

- **Component database:** `data/pure_components/components.json`
- **Group definitions:** `src/pvtcore/eos/groups/definitions.py`
- **Built-in groups:** `src/pvtcore/eos/groups/decomposition.py`
- **PPR78 calculator:** `src/pvtcore/eos/ppr78.py`
- **Tests:** `tests/unit/test_ppr78.py`

## Testing

Run validation after database changes:
```bash
python scripts/validate_components.py
python scripts/test_c2_normalization.py
pytest tests/unit/test_ppr78.py -v
```

## References

1. Jaubert, J.-N., & Mutelet, F. (2004). "VLE predictions with the Peng-Robinson equation of state and temperature dependent kij calculated through a group contribution method." *Fluid Phase Equilibria*, 224(2), 285-304.

2. Qian, J.-W., Jaubert, J.-N., & Privat, R. (2013). "Phase equilibria in hydrogen-containing binary systems modeled with the Peng-Robinson equation of state and temperature-dependent binary interaction parameters calculated through a group-contribution method." *Journal of Supercritical Fluids*, 75, 58-71.

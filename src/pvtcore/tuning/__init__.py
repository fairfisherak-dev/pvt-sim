"""EOS parameter tuning/regression module.

This module provides tools for tuning equation of state parameters
against experimental PVT data. Key capabilities:

- Define experimental data points (saturation pressure, density, etc.)
- Specify tunable parameters (kij, volume shifts, multipliers)
- Run optimization using scipy.optimize
- Analyze regression quality and sensitivity

Typical workflow:
1. Create ExperimentalDataSet from lab measurements
2. Define ParameterSet with parameters to tune
3. Create EOSRegressor and add datasets
4. Call regressor.fit() to optimize
5. Extract optimal parameters

Units Convention:
- Pressure: Pa
- Temperature: K
- Density: kg/m³
- kij: dimensionless
- Volume shift: m³/kmol

References
----------
[1] Pedersen et al. (2015). Phase Behavior of Petroleum Reservoir Fluids.
[2] Whitson & Brule (2000). Phase Behavior. SPE Monograph.
"""

# Objective functions and data structures
from .objectives import (
    DataType,
    ExperimentalPoint,
    ExperimentalDataSet,
    ObjectiveResult,
    ObjectiveFunction,
    calculate_residual,
    calculate_objective_sse,
    calculate_objective_aad,
    create_saturation_objective,
    create_density_objective,
)

# Parameter definitions
from .parameters import (
    ParameterType,
    TunableParameter,
    ParameterSet,
    create_kij_parameters,
    create_volume_shift_parameters,
    create_critical_multipliers,
    merge_parameter_sets,
)

# Regression engine
from .regression import (
    RegressionResult,
    EOSRegressor,
    tune_binary_interactions,
    tune_volume_shifts,
    sensitivity_analysis,
)

__all__ = [
    # Data types
    "DataType",
    "ExperimentalPoint",
    "ExperimentalDataSet",
    "ObjectiveResult",
    "ObjectiveFunction",
    # Objective helpers
    "calculate_residual",
    "calculate_objective_sse",
    "calculate_objective_aad",
    "create_saturation_objective",
    "create_density_objective",
    # Parameter types
    "ParameterType",
    "TunableParameter",
    "ParameterSet",
    # Parameter helpers
    "create_kij_parameters",
    "create_volume_shift_parameters",
    "create_critical_multipliers",
    "merge_parameter_sets",
    # Regression
    "RegressionResult",
    "EOSRegressor",
    "tune_binary_interactions",
    "tune_volume_shifts",
    "sensitivity_analysis",
]

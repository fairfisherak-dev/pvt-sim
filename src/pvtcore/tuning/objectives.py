"""Objective functions for EOS parameter regression.

This module provides objective function calculations for tuning EOS
parameters against experimental PVT data. The objective functions
measure the deviation between model predictions and experimental
measurements.

Supported experimental data types:
- Saturation pressure (bubble/dew point) at various temperatures
- Liquid/vapor density at P, T conditions
- Flash results (vapor fraction, phase compositions)
- GOR and Bo from separator tests
- CCE relative volumes
- CVD liquid dropout

Units Convention:
- Pressure: Pa
- Temperature: K
- Density: kg/m³
- All deviations are relative (fraction or %)

References
----------
[1] Pedersen et al. (2015). Phase Behavior of Petroleum Reservoir Fluids.
[2] Experiment Design and Model Discrimination (SPE literature).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Union
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError


class DataType(Enum):
    """Types of experimental data for regression."""
    SATURATION_PRESSURE = "saturation_pressure"
    LIQUID_DENSITY = "liquid_density"
    VAPOR_DENSITY = "vapor_density"
    VAPOR_FRACTION = "vapor_fraction"
    LIQUID_COMPOSITION = "liquid_composition"
    VAPOR_COMPOSITION = "vapor_composition"
    RELATIVE_VOLUME = "relative_volume"
    LIQUID_DROPOUT = "liquid_dropout"
    GOR = "gor"
    BO = "bo"
    Z_FACTOR = "z_factor"


@dataclass
class ExperimentalPoint:
    """Single experimental data point.

    Attributes:
        data_type: Type of measurement
        temperature: Temperature (K)
        pressure: Pressure (Pa), or None for saturation pressure measurements
        value: Measured value
        uncertainty: Measurement uncertainty (optional)
        composition: Composition for this point (if different from reference)
        weight: Weight in objective function (default 1.0)
        metadata: Additional information (experiment ID, etc.)
    """
    data_type: DataType
    temperature: float
    pressure: Optional[float]
    value: Union[float, NDArray[np.float64]]
    uncertainty: Optional[float] = None
    composition: Optional[NDArray[np.float64]] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalDataSet:
    """Collection of experimental data for regression.

    Attributes:
        name: Dataset identifier
        composition: Reference composition (mole fractions)
        points: List of experimental points
        description: Optional description
    """
    name: str
    composition: NDArray[np.float64]
    points: List[ExperimentalPoint]
    description: str = ""

    def __post_init__(self):
        """Validate dataset."""
        if len(self.points) == 0:
            raise ValidationError("Dataset must contain at least one point", parameter="points")
        self.composition = np.asarray(self.composition, dtype=np.float64)
        self.composition = self.composition / self.composition.sum()

    def get_points_by_type(self, data_type: DataType) -> List[ExperimentalPoint]:
        """Get all points of a specific type."""
        return [p for p in self.points if p.data_type == data_type]

    @property
    def n_points(self) -> int:
        """Total number of data points."""
        return len(self.points)

    @property
    def data_types(self) -> List[DataType]:
        """Unique data types in this dataset."""
        return list(set(p.data_type for p in self.points))


@dataclass
class ObjectiveResult:
    """Results from objective function evaluation.

    Attributes:
        total_objective: Total weighted objective value
        n_points: Number of points evaluated
        residuals: Individual residuals for each point
        relative_errors: Relative errors (%)
        by_type: Breakdown by data type
        points_evaluated: Points that were successfully evaluated
        points_failed: Points that failed to evaluate
    """
    total_objective: float
    n_points: int
    residuals: NDArray[np.float64]
    relative_errors: NDArray[np.float64]
    by_type: Dict[DataType, float]
    points_evaluated: int
    points_failed: int


def calculate_residual(
    calculated: float,
    experimental: float,
    uncertainty: Optional[float] = None,
    relative: bool = True,
) -> float:
    """Calculate residual between calculated and experimental values.

    Parameters
    ----------
    calculated : float
        Model-calculated value.
    experimental : float
        Experimental (measured) value.
    uncertainty : float, optional
        Measurement uncertainty for weighting.
    relative : bool
        If True, return relative residual (as fraction).

    Returns
    -------
    float
        Residual value.
    """
    if experimental == 0:
        if calculated == 0:
            return 0.0
        else:
            return abs(calculated)  # Can't compute relative

    if relative:
        residual = (calculated - experimental) / experimental
    else:
        residual = calculated - experimental

    if uncertainty is not None and uncertainty > 0:
        residual = residual / uncertainty

    return residual


def calculate_objective_sse(residuals: NDArray[np.float64]) -> float:
    """Calculate sum of squared errors objective.

    Parameters
    ----------
    residuals : ndarray
        Array of residuals.

    Returns
    -------
    float
        Sum of squared residuals.
    """
    return float(np.sum(residuals ** 2))


def calculate_objective_aad(residuals: NDArray[np.float64]) -> float:
    """Calculate average absolute deviation objective.

    Parameters
    ----------
    residuals : ndarray
        Array of residuals (relative).

    Returns
    -------
    float
        Average absolute deviation (%).
    """
    return float(np.mean(np.abs(residuals)) * 100)


def calculate_objective_max(residuals: NDArray[np.float64]) -> float:
    """Calculate maximum absolute deviation.

    Parameters
    ----------
    residuals : ndarray
        Array of residuals.

    Returns
    -------
    float
        Maximum absolute residual.
    """
    return float(np.max(np.abs(residuals)))


class ObjectiveFunction:
    """Configurable objective function for EOS regression.

    The objective function evaluates how well the EOS model matches
    experimental data. It supports:
    - Multiple data types with individual weights
    - Different error metrics (SSE, AAD, etc.)
    - Robust handling of calculation failures

    Parameters
    ----------
    datasets : list of ExperimentalDataSet
        Experimental data to match.
    model_function : callable
        Function that calculates model predictions.
        Signature: model_function(point: ExperimentalPoint, params: dict) -> float
    weights : dict, optional
        Weights by data type. Default is equal weights.
    metric : str
        Objective metric: 'sse' (sum squared error), 'aad' (avg abs dev).
    """

    def __init__(
        self,
        datasets: List[ExperimentalDataSet],
        model_function: Callable[[ExperimentalPoint, Dict[str, Any]], float],
        weights: Optional[Dict[DataType, float]] = None,
        metric: str = 'sse',
    ):
        self.datasets = datasets
        self.model_function = model_function
        self.weights = weights or {}
        self.metric = metric.lower()

        if self.metric not in ('sse', 'aad', 'max'):
            raise ValidationError(
                f"Unknown metric '{metric}'. Use 'sse', 'aad', or 'max'.",
                parameter="metric"
            )

        # Count total points
        self.n_total = sum(ds.n_points for ds in datasets)

    def evaluate(
        self,
        params: Dict[str, Any],
        return_details: bool = False,
    ) -> Union[float, ObjectiveResult]:
        """Evaluate objective function.

        Parameters
        ----------
        params : dict
            Current parameter values (kij, volume shifts, etc.).
        return_details : bool
            If True, return detailed ObjectiveResult.

        Returns
        -------
        float or ObjectiveResult
            Objective value, or detailed results if requested.
        """
        all_residuals = []
        by_type: Dict[DataType, List[float]] = {}
        points_failed = 0

        for dataset in self.datasets:
            for point in dataset.points:
                try:
                    # Get composition (point-specific or dataset default)
                    comp = point.composition if point.composition is not None else dataset.composition

                    # Calculate model prediction
                    calculated = self.model_function(point, params)

                    # Handle array values (compositions)
                    if isinstance(point.value, np.ndarray):
                        if isinstance(calculated, np.ndarray):
                            residual = np.mean(np.abs(calculated - point.value) / np.maximum(point.value, 1e-10))
                        else:
                            residual = 1.0  # Mismatch
                    else:
                        residual = calculate_residual(
                            calculated, point.value,
                            uncertainty=point.uncertainty,
                            relative=True,
                        )

                    # Apply weight
                    type_weight = self.weights.get(point.data_type, 1.0)
                    weighted_residual = residual * point.weight * type_weight

                    all_residuals.append(weighted_residual)

                    # Track by type
                    if point.data_type not in by_type:
                        by_type[point.data_type] = []
                    by_type[point.data_type].append(residual)

                except Exception:
                    points_failed += 1
                    # Penalty for failed points
                    all_residuals.append(1.0)  # Large relative error

        residuals = np.array(all_residuals)

        # Calculate objective
        if self.metric == 'sse':
            objective = calculate_objective_sse(residuals)
        elif self.metric == 'aad':
            objective = calculate_objective_aad(residuals)
        else:  # max
            objective = calculate_objective_max(residuals)

        if not return_details:
            return objective

        # Detailed results
        type_objectives = {
            dt: calculate_objective_aad(np.array(res))
            for dt, res in by_type.items()
        }

        return ObjectiveResult(
            total_objective=objective,
            n_points=len(residuals),
            residuals=residuals,
            relative_errors=np.abs(residuals) * 100,
            by_type=type_objectives,
            points_evaluated=len(residuals) - points_failed,
            points_failed=points_failed,
        )

    def __call__(self, params: Dict[str, Any]) -> float:
        """Evaluate objective (for scipy.optimize compatibility)."""
        return self.evaluate(params, return_details=False)


def create_saturation_objective(
    temperatures: NDArray[np.float64],
    pressures_exp: NDArray[np.float64],
    composition: NDArray[np.float64],
    saturation_type: str = 'bubble',
) -> ExperimentalDataSet:
    """Create dataset for saturation pressure regression.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures (K) at which Psat was measured.
    pressures_exp : ndarray
        Experimental saturation pressures (Pa).
    composition : ndarray
        Fluid composition.
    saturation_type : str
        'bubble' or 'dew'.

    Returns
    -------
    ExperimentalDataSet
        Dataset ready for regression.
    """
    points = []
    for T, P in zip(temperatures, pressures_exp):
        points.append(ExperimentalPoint(
            data_type=DataType.SATURATION_PRESSURE,
            temperature=float(T),
            pressure=None,  # This is what we're matching
            value=float(P),
            metadata={'saturation_type': saturation_type},
        ))

    return ExperimentalDataSet(
        name=f"{saturation_type}_pressure",
        composition=composition,
        points=points,
        description=f"Saturation pressure ({saturation_type}) vs temperature",
    )


def create_density_objective(
    temperatures: NDArray[np.float64],
    pressures: NDArray[np.float64],
    densities_exp: NDArray[np.float64],
    composition: NDArray[np.float64],
    phase: str = 'liquid',
) -> ExperimentalDataSet:
    """Create dataset for density regression.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures (K).
    pressures : ndarray
        Pressures (Pa).
    densities_exp : ndarray
        Experimental densities (kg/m³).
    composition : ndarray
        Fluid composition.
    phase : str
        'liquid' or 'vapor'.

    Returns
    -------
    ExperimentalDataSet
        Dataset ready for regression.
    """
    data_type = DataType.LIQUID_DENSITY if phase == 'liquid' else DataType.VAPOR_DENSITY

    points = []
    for T, P, rho in zip(temperatures, pressures, densities_exp):
        points.append(ExperimentalPoint(
            data_type=data_type,
            temperature=float(T),
            pressure=float(P),
            value=float(rho),
        ))

    return ExperimentalDataSet(
        name=f"{phase}_density",
        composition=composition,
        points=points,
        description=f"{phase.capitalize()} density at various P, T",
    )

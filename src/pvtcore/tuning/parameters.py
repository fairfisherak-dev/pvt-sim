"""Parameter definitions for EOS regression.

This module defines the parameters that can be tuned during EOS
regression, including:
- Binary interaction parameters (kij)
- Volume shift parameters (Peneloux correction)
- Critical property multipliers
- Acentric factor adjustments

Each parameter has:
- Initial value
- Bounds (min, max)
- Active flag (whether to tune it)
- Optional constraints

Units Convention:
- kij: dimensionless (-0.2 to +0.2 typical range)
- Volume shift: m³/kmol
- Multipliers: dimensionless (typically 0.9 to 1.1)

References
----------
[1] Pedersen et al. (2015). Phase Behavior of Petroleum Reservoir Fluids.
[2] Whitson & Brule (2000). Phase Behavior. SPE Monograph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError


class ParameterType(Enum):
    """Types of tunable parameters."""
    BINARY_INTERACTION = "kij"
    VOLUME_SHIFT = "volume_shift"
    TC_MULTIPLIER = "Tc_mult"
    PC_MULTIPLIER = "Pc_mult"
    OMEGA_MULTIPLIER = "omega_mult"
    OMEGA_A = "Omega_a"
    OMEGA_B = "Omega_b"


@dataclass
class TunableParameter:
    """Definition of a single tunable parameter.

    Attributes:
        name: Parameter identifier (e.g., "kij_C1_C7")
        param_type: Type of parameter
        initial_value: Starting value for regression
        lower_bound: Minimum allowed value
        upper_bound: Maximum allowed value
        active: If True, parameter will be tuned
        component_i: First component index (for kij)
        component_j: Second component index (for kij)
        component: Component index (for volume shift, multipliers)
        description: Human-readable description
    """
    name: str
    param_type: ParameterType
    initial_value: float
    lower_bound: float
    upper_bound: float
    active: bool = True
    component_i: Optional[int] = None
    component_j: Optional[int] = None
    component: Optional[int] = None
    description: str = ""

    def __post_init__(self):
        """Validate parameter definition."""
        if self.lower_bound > self.upper_bound:
            raise ValidationError(
                f"Lower bound ({self.lower_bound}) > upper bound ({self.upper_bound})",
                parameter=self.name
            )
        if not (self.lower_bound <= self.initial_value <= self.upper_bound):
            raise ValidationError(
                f"Initial value ({self.initial_value}) outside bounds [{self.lower_bound}, {self.upper_bound}]",
                parameter=self.name
            )

    @property
    def bounds(self) -> Tuple[float, float]:
        """Return (lower, upper) bounds tuple."""
        return (self.lower_bound, self.upper_bound)


@dataclass
class ParameterSet:
    """Collection of tunable parameters for regression.

    Attributes:
        parameters: List of parameter definitions
        n_components: Number of components in the system
    """
    parameters: List[TunableParameter] = field(default_factory=list)
    n_components: int = 0

    def add_parameter(self, param: TunableParameter) -> None:
        """Add a parameter to the set."""
        # Check for duplicates
        if any(p.name == param.name for p in self.parameters):
            raise ValidationError(
                f"Parameter '{param.name}' already exists",
                parameter="parameters"
            )
        self.parameters.append(param)

    def get_parameter(self, name: str) -> Optional[TunableParameter]:
        """Get parameter by name."""
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    @property
    def active_parameters(self) -> List[TunableParameter]:
        """Get list of active (tunable) parameters."""
        return [p for p in self.parameters if p.active]

    @property
    def n_active(self) -> int:
        """Number of active parameters."""
        return len(self.active_parameters)

    def get_initial_vector(self) -> NDArray[np.float64]:
        """Get initial values as vector (for optimizer)."""
        return np.array([p.initial_value for p in self.active_parameters])

    def get_bounds_list(self) -> List[Tuple[float, float]]:
        """Get bounds as list of tuples (for scipy.optimize)."""
        return [p.bounds for p in self.active_parameters]

    def vector_to_dict(self, x: NDArray[np.float64]) -> Dict[str, float]:
        """Convert parameter vector to dictionary.

        Parameters
        ----------
        x : ndarray
            Parameter vector from optimizer.

        Returns
        -------
        dict
            Parameter values keyed by name.
        """
        active = self.active_parameters
        if len(x) != len(active):
            raise ValidationError(
                f"Vector length ({len(x)}) != number of active parameters ({len(active)})",
                parameter="x"
            )
        return {p.name: float(x[i]) for i, p in enumerate(active)}

    def dict_to_vector(self, params: Dict[str, float]) -> NDArray[np.float64]:
        """Convert dictionary to parameter vector.

        Parameters
        ----------
        params : dict
            Parameter values keyed by name.

        Returns
        -------
        ndarray
            Parameter vector for optimizer.
        """
        active = self.active_parameters
        return np.array([params.get(p.name, p.initial_value) for p in active])

    def extract_kij_matrix(
        self,
        params: Dict[str, float],
    ) -> NDArray[np.float64]:
        """Extract binary interaction matrix from parameters.

        Parameters
        ----------
        params : dict
            Current parameter values.

        Returns
        -------
        ndarray
            n_components x n_components kij matrix (symmetric).
        """
        kij = np.zeros((self.n_components, self.n_components))

        for p in self.parameters:
            if p.param_type == ParameterType.BINARY_INTERACTION:
                i = p.component_i
                j = p.component_j
                if i is not None and j is not None:
                    value = params.get(p.name, p.initial_value)
                    kij[i, j] = value
                    kij[j, i] = value  # Symmetric

        return kij

    def extract_volume_shifts(
        self,
        params: Dict[str, float],
    ) -> NDArray[np.float64]:
        """Extract volume shift parameters.

        Parameters
        ----------
        params : dict
            Current parameter values.

        Returns
        -------
        ndarray
            Volume shifts for each component.
        """
        shifts = np.zeros(self.n_components)

        for p in self.parameters:
            if p.param_type == ParameterType.VOLUME_SHIFT:
                if p.component is not None:
                    shifts[p.component] = params.get(p.name, p.initial_value)

        return shifts


def create_kij_parameters(
    n_components: int,
    component_names: Optional[List[str]] = None,
    initial_values: Optional[NDArray[np.float64]] = None,
    active_pairs: Optional[List[Tuple[int, int]]] = None,
    bounds: Tuple[float, float] = (-0.15, 0.15),
) -> ParameterSet:
    """Create parameter set for binary interaction coefficients.

    Parameters
    ----------
    n_components : int
        Number of components.
    component_names : list of str, optional
        Component names for labeling.
    initial_values : ndarray, optional
        Initial kij matrix. Default is zeros.
    active_pairs : list of tuples, optional
        Which (i, j) pairs to tune. Default is all.
    bounds : tuple
        (lower, upper) bounds for all kij.

    Returns
    -------
    ParameterSet
        Parameter set with kij parameters.

    Examples
    --------
    >>> params = create_kij_parameters(3, ['C1', 'C4', 'C10'])
    >>> print(f"Number of kij parameters: {params.n_active}")
    Number of kij parameters: 3
    """
    if component_names is None:
        component_names = [f"C{i+1}" for i in range(n_components)]

    if initial_values is None:
        initial_values = np.zeros((n_components, n_components))

    param_set = ParameterSet(n_components=n_components)

    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Check if this pair should be active
            if active_pairs is not None:
                active = (i, j) in active_pairs or (j, i) in active_pairs
            else:
                active = True

            name = f"kij_{component_names[i]}_{component_names[j]}"
            initial = float(initial_values[i, j])

            param = TunableParameter(
                name=name,
                param_type=ParameterType.BINARY_INTERACTION,
                initial_value=initial,
                lower_bound=bounds[0],
                upper_bound=bounds[1],
                active=active,
                component_i=i,
                component_j=j,
                description=f"Binary interaction: {component_names[i]}-{component_names[j]}",
            )
            param_set.add_parameter(param)

    return param_set


def create_volume_shift_parameters(
    n_components: int,
    component_names: Optional[List[str]] = None,
    initial_values: Optional[NDArray[np.float64]] = None,
    active_components: Optional[List[int]] = None,
    bounds: Tuple[float, float] = (-0.01, 0.01),
) -> ParameterSet:
    """Create parameter set for volume shift (Peneloux) parameters.

    Parameters
    ----------
    n_components : int
        Number of components.
    component_names : list of str, optional
        Component names for labeling.
    initial_values : ndarray, optional
        Initial volume shifts (m³/kmol). Default is zeros.
    active_components : list of int, optional
        Which components to tune. Default is all.
    bounds : tuple
        (lower, upper) bounds for volume shifts.

    Returns
    -------
    ParameterSet
        Parameter set with volume shift parameters.
    """
    if component_names is None:
        component_names = [f"C{i+1}" for i in range(n_components)]

    if initial_values is None:
        initial_values = np.zeros(n_components)

    param_set = ParameterSet(n_components=n_components)

    for i in range(n_components):
        active = active_components is None or i in active_components

        name = f"c_{component_names[i]}"
        initial = float(initial_values[i])

        param = TunableParameter(
            name=name,
            param_type=ParameterType.VOLUME_SHIFT,
            initial_value=initial,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            active=active,
            component=i,
            description=f"Volume shift for {component_names[i]}",
        )
        param_set.add_parameter(param)

    return param_set


def create_critical_multipliers(
    n_components: int,
    component_names: Optional[List[str]] = None,
    tune_Tc: bool = False,
    tune_Pc: bool = False,
    tune_omega: bool = True,
    active_components: Optional[List[int]] = None,
    Tc_bounds: Tuple[float, float] = (0.95, 1.05),
    Pc_bounds: Tuple[float, float] = (0.95, 1.05),
    omega_bounds: Tuple[float, float] = (0.90, 1.10),
) -> ParameterSet:
    """Create parameter set for critical property multipliers.

    These multipliers adjust critical properties for pseudo-components
    (C7+ fractions) where correlations may be inaccurate.

    Parameters
    ----------
    n_components : int
        Number of components.
    component_names : list of str, optional
        Component names.
    tune_Tc : bool
        Include Tc multipliers.
    tune_Pc : bool
        Include Pc multipliers.
    tune_omega : bool
        Include acentric factor multipliers.
    active_components : list of int, optional
        Which components to tune (typically heavy ends).
    Tc_bounds, Pc_bounds, omega_bounds : tuple
        Bounds for each multiplier type.

    Returns
    -------
    ParameterSet
        Parameter set with multiplier parameters.
    """
    if component_names is None:
        component_names = [f"C{i+1}" for i in range(n_components)]

    param_set = ParameterSet(n_components=n_components)

    for i in range(n_components):
        active_base = active_components is None or i in active_components

        if tune_Tc:
            param_set.add_parameter(TunableParameter(
                name=f"Tc_mult_{component_names[i]}",
                param_type=ParameterType.TC_MULTIPLIER,
                initial_value=1.0,
                lower_bound=Tc_bounds[0],
                upper_bound=Tc_bounds[1],
                active=active_base,
                component=i,
                description=f"Tc multiplier for {component_names[i]}",
            ))

        if tune_Pc:
            param_set.add_parameter(TunableParameter(
                name=f"Pc_mult_{component_names[i]}",
                param_type=ParameterType.PC_MULTIPLIER,
                initial_value=1.0,
                lower_bound=Pc_bounds[0],
                upper_bound=Pc_bounds[1],
                active=active_base,
                component=i,
                description=f"Pc multiplier for {component_names[i]}",
            ))

        if tune_omega:
            param_set.add_parameter(TunableParameter(
                name=f"omega_mult_{component_names[i]}",
                param_type=ParameterType.OMEGA_MULTIPLIER,
                initial_value=1.0,
                lower_bound=omega_bounds[0],
                upper_bound=omega_bounds[1],
                active=active_base,
                component=i,
                description=f"Acentric factor multiplier for {component_names[i]}",
            ))

    return param_set


def merge_parameter_sets(*param_sets: ParameterSet) -> ParameterSet:
    """Merge multiple parameter sets into one.

    Parameters
    ----------
    *param_sets : ParameterSet
        Parameter sets to merge.

    Returns
    -------
    ParameterSet
        Combined parameter set.

    Raises
    ------
    ValidationError
        If parameter sets have incompatible n_components.
    """
    if not param_sets:
        return ParameterSet()

    # Check compatibility
    n_components = param_sets[0].n_components
    for ps in param_sets[1:]:
        if ps.n_components != n_components and ps.n_components > 0:
            raise ValidationError(
                f"Incompatible n_components: {ps.n_components} vs {n_components}",
                parameter="param_sets"
            )

    merged = ParameterSet(n_components=n_components)
    for ps in param_sets:
        for param in ps.parameters:
            merged.add_parameter(param)

    return merged

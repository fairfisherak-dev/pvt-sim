"""Wilson K-value correlation for phase equilibrium initialization.

The Wilson correlation provides initial estimates for K-values (vapor-liquid
equilibrium ratios) used to initialize flash calculations.

Reference:
Wilson, G. M., "A Modified Redlich-Kwong Equation of State, Application to
General Physical Data Calculations", AIChE Meeting, Cleveland, OH (1968).
"""

import math
import numpy as np
from typing import List, Union
from ..models.component import Component


def wilson_k_values(
    pressure: float,
    temperature: float,
    components: List[Component]
) -> np.ndarray:
    """Calculate Wilson K-values for vapor-liquid equilibrium initialization.

    The Wilson correlation provides a simple estimate of K-values without
    requiring an equation of state calculation. It's based on reduced
    properties and is sufficiently accurate for flash initialization.

    Formula:
        Ki = (Pci/P) × exp[5.373(1 + ωi)(1 - Tci/T)]

    where:
        Ki = equilibrium ratio for component i (yi/xi)
        Pci = critical pressure of component i
        P = system pressure
        ωi = acentric factor of component i
        Tci = critical temperature of component i
        T = system temperature

    Args:
        pressure: System pressure (Pa)
        temperature: System temperature (K)
        components: List of Component objects

    Returns:
        Array of K-values (dimensionless), one per component

    Example:
        >>> from pvtcore.models import load_components
        >>> components = load_components()
        >>> # Light component (methane) at low pressure
        >>> K = wilson_k_values(1e6, 300, [components['C1']])
        >>> # K > 1 for light component (prefers vapor phase)
        >>> print(K[0] > 1.0)
        True

    Notes:
        - K > 1: component prefers vapor phase (more volatile)
        - K < 1: component prefers liquid phase (less volatile)
        - K = 1: equal distribution between phases
        - Accuracy decreases near critical point
        - Best for reduced pressures Pr < 0.8
    """
    n_components = len(components)
    K = np.zeros(n_components)

    for i, comp in enumerate(components):
        # Reduced temperature
        Tr = temperature / comp.Tc

        # Reduced pressure
        Pr = pressure / comp.Pc

        # Wilson correlation
        # Ki = (Pci/P) × exp[5.373(1 + ωi)(1 - Tci/T)]
        # Rewrite using reduced properties:
        # Ki = (1/Pr) × exp[5.373(1 + ωi)(1 - 1/Tr)]

        exponent = 5.373 * (1.0 + comp.omega) * (1.0 - 1.0 / Tr)
        K[i] = (1.0 / Pr) * math.exp(exponent)

    return K


def wilson_k_value_single(
    pressure: float,
    temperature: float,
    component: Component
) -> float:
    """Calculate Wilson K-value for a single component.

    Convenience function for single component calculations.

    Args:
        pressure: System pressure (Pa)
        temperature: System temperature (K)
        component: Component object

    Returns:
        K-value for the component

    Example:
        >>> from pvtcore.models import get_component
        >>> methane = get_component('C1')
        >>> K = wilson_k_value_single(5e6, 200, methane)
        >>> print(f"K-value: {K:.2f}")
    """
    result = wilson_k_values(pressure, temperature, [component])
    return result[0]


def is_trivial_solution(
    K_values: np.ndarray,
    composition: np.ndarray,
    tolerance: float = 1e-8
) -> tuple[bool, Union[str, None]]:
    """Check if K-values indicate a trivial (single-phase) solution.

    A trivial solution occurs when all K-values are either:
    - All > 1: All components prefer vapor → single vapor phase
    - All < 1: All components prefer liquid → single liquid phase

    Args:
        K_values: Array of K-values
        composition: Feed composition (mole fractions)
        tolerance: Tolerance for checking K near 1.0

    Returns:
        Tuple of (is_trivial, phase) where:
        - is_trivial: True if single phase
        - phase: 'vapor', 'liquid', or None

    Example:
        >>> K = np.array([2.5, 3.0, 4.0])  # All K > 1
        >>> z = np.array([0.5, 0.3, 0.2])
        >>> is_trivial, phase = is_trivial_solution(K, z)
        >>> print(phase)
        'vapor'
    """
    # Check if all K > 1
    if np.all(K_values > 1.0 + tolerance):
        return True, 'vapor'

    # Check if all K < 1
    if np.all(K_values < 1.0 - tolerance):
        return True, 'liquid'

    return False, None


def rachford_rice_bounds(
    K_values: np.ndarray,
    composition: np.ndarray
) -> tuple[float, float]:
    """Calculate valid bounds for Rachford-Rice vapor fraction.

    The vapor fraction nv must satisfy:
        1/(1-K_max) < nv < 1/(1-K_min)

    where K_max and K_min are the maximum and minimum K-values.

    Args:
        K_values: Array of K-values
        composition: Feed composition (not used but kept for consistency)

    Returns:
        Tuple of (nv_min, nv_max) bounds

    Example:
        >>> K = np.array([0.5, 1.0, 2.0])
        >>> z = np.array([0.3, 0.3, 0.4])
        >>> nv_min, nv_max = rachford_rice_bounds(K, z)
        >>> print(f"Valid range: {nv_min:.3f} to {nv_max:.3f}")
    """
    K_min = np.min(K_values)
    K_max = np.max(K_values)

    # Bounds from requirement that 0 < xi < 1 and 0 < yi < 1
    # xi = zi / (1 + nv(Ki-1)) must be in (0,1)
    # yi = Ki×zi / (1 + nv(Ki-1)) must be in (0,1)

    if K_max > 1.0:
        nv_min = 1.0 / (1.0 - K_max)
    else:
        nv_min = 0.0

    if K_min < 1.0:
        nv_max = 1.0 / (1.0 - K_min)
    else:
        nv_max = 1.0

    return nv_min, nv_max


def estimate_vapor_fraction(
    K_values: np.ndarray,
    composition: np.ndarray
) -> float:
    """Provide a simple initial estimate for vapor fraction.

    Uses a weighted average approach based on how volatile the feed is.

    Args:
        K_values: Array of K-values
        composition: Feed composition (mole fractions)

    Returns:
        Initial estimate for vapor fraction (0 to 1)

    Example:
        >>> K = np.array([3.0, 0.5])
        >>> z = np.array([0.7, 0.3])  # Mostly light component
        >>> nv = estimate_vapor_fraction(K, z)
        >>> print(f"Initial nv estimate: {nv:.2f}")
    """
    # Weighted average based on K-values and composition
    # If average K > 1, more vapor; if average K < 1, more liquid

    avg_K = np.sum(composition * K_values)

    if avg_K > 10.0:
        # Very light feed, mostly vapor
        return 0.9
    elif avg_K < 0.1:
        # Very heavy feed, mostly liquid
        return 0.1
    elif avg_K > 1.0:
        # Light feed
        return 0.5 + 0.3 * (avg_K - 1.0) / 9.0
    else:
        # Heavy feed
        return 0.5 * avg_K

    return max(0.1, min(0.9, nv_estimate))


def wilson_correlation_valid(
    pressure: float,
    temperature: float,
    components: List[Component],
    max_reduced_pressure: float = 0.8
) -> tuple[bool, str]:
    """Check if Wilson correlation is valid for given conditions.

    The Wilson correlation is most accurate at moderate pressures.
    It becomes less accurate near the critical point.

    Args:
        pressure: System pressure (Pa)
        temperature: System temperature (K)
        components: List of Component objects
        max_reduced_pressure: Maximum Pr for validity (default: 0.8)

    Returns:
        Tuple of (is_valid, message)

    Example:
        >>> from pvtcore.models import load_components
        >>> components = load_components()
        >>> valid, msg = wilson_correlation_valid(
        ...     5e6, 300, [components['C1'], components['C10']]
        ... )
        >>> print(valid)
        True
    """
    # Check reduced pressure for each component
    for comp in components:
        Pr = pressure / comp.Pc
        Tr = temperature / comp.Tc

        if Pr > max_reduced_pressure:
            return False, (
                f"Reduced pressure {Pr:.2f} exceeds recommended limit "
                f"{max_reduced_pressure} for {comp.name}. "
                "Wilson correlation may be inaccurate."
            )

        if Tr > 2.0:
            return False, (
                f"Reduced temperature {Tr:.2f} is very high for {comp.name}. "
                "Wilson correlation not recommended."
            )

        if Tr < 0.5:
            return False, (
                f"Reduced temperature {Tr:.2f} is very low for {comp.name}. "
                "Wilson correlation not recommended."
            )

    return True, "Wilson correlation is valid for these conditions."

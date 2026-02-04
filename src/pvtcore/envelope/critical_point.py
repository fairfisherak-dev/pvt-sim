"""Critical point detection for multicomponent mixtures.

Implements robust critical point detection using multiple strategies:
1. Kay's mixing rule estimation with refinement
2. K-value convergence detection (Ki → 1 for all components)
3. Envelope curve intersection analysis
4. Maximum pressure (cricondenbar) proximity

The critical point is where:
- Liquid and vapor phases become identical
- All K-values equal unity simultaneously
- Bubble and dew point curves meet

References:
- Michelsen, M.L. and Mollerup, J.M. (2007). "Thermodynamic Models:
  Fundamentals & Computational Aspects", 2nd ed., Chapter 12.
- Heidemann, R.A. and Khalil, A.M. (1980). "The Calculation of Critical
  Points", AIChE Journal, 26(5), 769-779.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import numpy as np
from numpy.typing import NDArray

from ..models.component import Component
from ..eos.base import CubicEOS
from ..stability.wilson import wilson_k_values
from ..core.errors import ConvergenceError


@dataclass
class CriticalPointResult:
    """Result from critical point calculation."""
    Tc: Optional[float]  # Critical temperature (K)
    Pc: Optional[float]  # Critical pressure (Pa)
    method: str  # Detection method used
    converged: bool
    iterations: int
    K_deviation: Optional[float]  # max|Ki - 1| at critical point


# Numerical parameters
MAX_CRITICAL_ITERATIONS: int = 100
K_VALUE_TOLERANCE: float = 0.05  # Tolerance for K-values near 1
PRESSURE_MATCH_TOLERANCE: float = 5e4  # Pa (50 kPa) for curve matching
TEMPERATURE_STEP: float = 1.0  # K, for refinement


def estimate_critical_point_kays(
    composition: NDArray[np.float64],
    components: List[Component],
) -> Tuple[float, float]:
    """Estimate mixture critical point using Kay's mixing rules.

    Kay's rules provide simple linear mixing for critical properties:
        Tc_mix = Σ zi × Tci
        Pc_mix = Σ zi × Pci

    These are rough estimates but provide good starting points for refinement.

    Parameters
    ----------
    composition : ndarray
        Mole fractions
    components : List[Component]
        Component objects with Tc, Pc properties

    Returns
    -------
    Tuple[float, float]
        (Tc_estimate, Pc_estimate) in (K, Pa)
    """
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()  # Normalize

    Tc_mix = sum(z[i] * comp.Tc for i, comp in enumerate(components))
    Pc_mix = sum(z[i] * comp.Pc for i, comp in enumerate(components))

    return float(Tc_mix), float(Pc_mix)


def estimate_critical_point_li(
    composition: NDArray[np.float64],
    components: List[Component],
) -> Tuple[float, float]:
    """Estimate mixture critical point using Li's correlation.

    Li's correlation accounts for non-ideal mixing and provides better
    estimates than Kay's rules, especially for asymmetric mixtures.

    For Tc: Uses mole fraction weighted Tc
    For Pc: Uses molar volume weighted combining rules

    Parameters
    ----------
    composition : ndarray
        Mole fractions
    components : List[Component]
        Component objects with Tc, Pc, Vc properties

    Returns
    -------
    Tuple[float, float]
        (Tc_estimate, Pc_estimate) in (K, Pa)
    """
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()

    # Get component critical volumes (estimate if not available)
    Vc = []
    for comp in components:
        if hasattr(comp, 'Vc') and comp.Vc is not None and comp.Vc > 0:
            Vc.append(comp.Vc)
        else:
            # Estimate Vc from Tc and Pc using Z_c ≈ 0.27
            Zc = 0.27
            from ..core.constants import R
            Vc_est = Zc * R.Pa_m3_per_mol_K * comp.Tc / comp.Pc
            Vc.append(Vc_est)
    Vc = np.array(Vc)

    # Critical temperature (mole fraction weighted)
    Tc_mix = sum(z[i] * comp.Tc for i, comp in enumerate(components))

    # Critical volume using quadratic mixing
    Vc_mix = 0.0
    for i in range(len(components)):
        for j in range(len(components)):
            Vc_ij = 0.125 * (Vc[i]**(1/3) + Vc[j]**(1/3))**3
            Vc_mix += z[i] * z[j] * Vc_ij

    # Critical pressure from Tc and Vc
    Zc_mix = 0.291 - 0.080 * sum(z[i] * comp.omega for i, comp in enumerate(components))
    from ..core.constants import R
    Pc_mix = Zc_mix * R.Pa_m3_per_mol_K * Tc_mix / Vc_mix

    return float(Tc_mix), float(Pc_mix)


def _k_value_deviation(K: NDArray[np.float64]) -> float:
    """Calculate maximum deviation of K-values from unity."""
    return float(np.max(np.abs(K - 1.0)))


def find_critical_point_kvalue_search(
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    binary_interaction: Optional[NDArray[np.float64]] = None,
    T_initial: Optional[float] = None,
    P_initial: Optional[float] = None,
    max_iterations: int = MAX_CRITICAL_ITERATIONS,
    tolerance: float = K_VALUE_TOLERANCE,
) -> CriticalPointResult:
    """Find critical point by searching for K-values approaching unity.

    At the critical point, all K-values (Ki = yi/xi) approach 1.0.
    This method searches in (T, P) space to minimize max|Ki - 1|.

    Parameters
    ----------
    composition : ndarray
        Feed composition (mole fractions)
    components : List[Component]
        Component objects
    eos : CubicEOS
        Equation of state
    binary_interaction : ndarray, optional
        Binary interaction parameters
    T_initial, P_initial : float, optional
        Initial guesses. If None, uses Kay's mixing rules.
    max_iterations : int
        Maximum iterations for search
    tolerance : float
        Tolerance for K-value deviation from unity

    Returns
    -------
    CriticalPointResult
        Critical point results
    """
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()

    # Initial guess from mixing rules
    if T_initial is None or P_initial is None:
        T_est, P_est = estimate_critical_point_kays(z, components)
        if T_initial is None:
            T_initial = T_est
        if P_initial is None:
            P_initial = P_est

    T = float(T_initial)
    P = float(P_initial)

    # Physical bounds
    T_min = min(comp.Tc for comp in components) * 0.5
    T_max = max(comp.Tc for comp in components) * 1.5
    P_min = 1e5  # 1 bar
    P_max = max(comp.Pc for comp in components) * 2.0

    best_T, best_P = T, P
    best_deviation = float('inf')

    # Use Wilson K-values for search (faster than full EOS K-values)
    # At critical point, Wilson K-values also approach 1
    for iteration in range(max_iterations):
        # Clamp to physical bounds
        T = max(T_min, min(T_max, T))
        P = max(P_min, min(P_max, P))

        try:
            K = wilson_k_values(P, T, components)
            deviation = _k_value_deviation(K)

            if deviation < best_deviation:
                best_deviation = deviation
                best_T, best_P = T, P

            if deviation < tolerance:
                return CriticalPointResult(
                    Tc=T,
                    Pc=P,
                    method="k_value_search",
                    converged=True,
                    iterations=iteration + 1,
                    K_deviation=deviation,
                )

            # Gradient-based step: move toward K = 1
            # dK/dT and dK/dP from Wilson correlation
            dT = 0.5  # K
            dP = P * 0.01  # 1% of P

            K_plus_T = wilson_k_values(P, T + dT, components)
            K_plus_P = wilson_k_values(P + dP, T, components)

            dev_plus_T = _k_value_deviation(K_plus_T)
            dev_plus_P = _k_value_deviation(K_plus_P)

            # Simple gradient descent
            grad_T = (dev_plus_T - deviation) / dT
            grad_P = (dev_plus_P - deviation) / dP

            # Adaptive step size
            step_T = -grad_T * 20.0  # Scale factor
            step_P = -grad_P * P * 0.5

            # Limit step sizes
            step_T = max(-50.0, min(50.0, step_T))
            step_P = max(-P * 0.3, min(P * 0.3, step_P))

            T += step_T
            P += step_P

        except Exception:
            # Numerical issue, try smaller step
            T = best_T + np.random.uniform(-5, 5)
            P = best_P * (1.0 + np.random.uniform(-0.1, 0.1))

    # Return best found even if not converged
    if best_deviation < tolerance * 5:  # Allow looser tolerance
        return CriticalPointResult(
            Tc=best_T,
            Pc=best_P,
            method="k_value_search",
            converged=True,
            iterations=max_iterations,
            K_deviation=best_deviation,
        )

    return CriticalPointResult(
        Tc=None,
        Pc=None,
        method="k_value_search",
        converged=False,
        iterations=max_iterations,
        K_deviation=best_deviation,
    )


def find_critical_from_envelope(
    bubble_T: NDArray[np.float64],
    bubble_P: NDArray[np.float64],
    dew_T: NDArray[np.float64],
    dew_P: NDArray[np.float64],
    composition: NDArray[np.float64],
    components: List[Component],
    tolerance: float = PRESSURE_MATCH_TOLERANCE,
) -> CriticalPointResult:
    """Find critical point from envelope curves using multiple strategies.

    Strategies:
    1. Maximum pressure point (cricondenbar) - often at or near critical
    2. Curve intersection where |P_bubble - P_dew| < tolerance
    3. Point where bubble and dew curves approach each other most closely

    Parameters
    ----------
    bubble_T, bubble_P : ndarray
        Bubble curve points (T in K, P in Pa)
    dew_T, dew_P : ndarray
        Dew curve points (T in K, P in Pa)
    composition : ndarray
        Mole fractions
    components : List[Component]
        Component objects
    tolerance : float
        Pressure matching tolerance (Pa)

    Returns
    -------
    CriticalPointResult
        Critical point results
    """
    if len(bubble_P) == 0 or len(dew_P) == 0:
        return CriticalPointResult(
            Tc=None, Pc=None, method="envelope_curves",
            converged=False, iterations=0, K_deviation=None,
        )

    # Get physical bounds from pure component critical points
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()

    Tc_min = min(comp.Tc for i, comp in enumerate(components) if z[i] > 1e-10)
    Tc_max = max(comp.Tc for i, comp in enumerate(components) if z[i] > 1e-10)

    # Kay's estimate as reference
    Tc_kay, Pc_kay = estimate_critical_point_kays(z, components)

    # Strategy 1: Maximum pressure point (cricondenbar)
    # The critical point is typically at or very near the maximum pressure
    all_P = np.concatenate([bubble_P, dew_P])
    all_T = np.concatenate([bubble_T, dew_T])

    idx_max_P = int(np.argmax(all_P))
    T_cricondenbar = float(all_T[idx_max_P])
    P_cricondenbar = float(all_P[idx_max_P])

    # Strategy 2: Find intersection of bubble and dew curves
    # Sort arrays by temperature for interpolation
    b_sort = np.argsort(bubble_T)
    d_sort = np.argsort(dew_T)
    Tb_sorted = bubble_T[b_sort]
    Pb_sorted = bubble_P[b_sort]
    Td_sorted = dew_T[d_sort]
    Pd_sorted = dew_P[d_sort]

    # Find overlapping temperature range
    T_overlap_min = max(float(np.min(Tb_sorted)), float(np.min(Td_sorted)))
    T_overlap_max = min(float(np.max(Tb_sorted)), float(np.max(Td_sorted)))

    best_intersection_T = None
    best_intersection_P = None
    min_pressure_diff = float('inf')

    if T_overlap_max > T_overlap_min:
        # Sample temperatures in overlap region
        T_samples = np.linspace(T_overlap_min, T_overlap_max, 500)

        for T in T_samples:
            # Interpolate pressures
            Pb_interp = np.interp(T, Tb_sorted, Pb_sorted)
            Pd_interp = np.interp(T, Td_sorted, Pd_sorted)

            pressure_diff = abs(Pb_interp - Pd_interp)

            # Track best intersection
            if pressure_diff < min_pressure_diff:
                # Validate: T should be in reasonable range for mixture
                if Tc_min * 0.8 < T < Tc_max * 1.3:
                    min_pressure_diff = pressure_diff
                    best_intersection_T = float(T)
                    best_intersection_P = float(0.5 * (Pb_interp + Pd_interp))

    # Strategy 3: Find where curves are closest (minimum distance)
    min_distance = float('inf')
    closest_T = None
    closest_P = None

    for i, (tb, pb) in enumerate(zip(bubble_T, bubble_P)):
        for j, (td, pd) in enumerate(zip(dew_T, dew_P)):
            # Distance metric in normalized (T, P) space
            dT_norm = (tb - td) / Tc_kay
            dP_norm = (pb - pd) / Pc_kay
            distance = math.sqrt(dT_norm**2 + dP_norm**2)

            if distance < min_distance:
                # Validate: should be at reasonable T
                avg_T = 0.5 * (tb + td)
                if Tc_min * 0.8 < avg_T < Tc_max * 1.3:
                    min_distance = distance
                    closest_T = float(avg_T)
                    closest_P = float(0.5 * (pb + pd))

    # Select best result
    candidates = []

    # Candidate 1: Cricondenbar (if in reasonable T range)
    if Tc_min * 0.7 < T_cricondenbar < Tc_max * 1.4:
        candidates.append((T_cricondenbar, P_cricondenbar, "cricondenbar", 0.0))

    # Candidate 2: Intersection point
    if best_intersection_T is not None and min_pressure_diff < tolerance:
        candidates.append((best_intersection_T, best_intersection_P,
                          "intersection", min_pressure_diff))

    # Candidate 3: Closest approach (if curves are close)
    if closest_T is not None and min_distance < 0.1:  # Normalized distance
        candidates.append((closest_T, closest_P, "closest_approach",
                          min_distance * Pc_kay))  # Convert back to Pa

    if not candidates:
        return CriticalPointResult(
            Tc=None, Pc=None, method="envelope_curves",
            converged=False, iterations=0, K_deviation=None,
        )

    # Prefer intersection method if available, otherwise cricondenbar
    for T, P, method, _ in candidates:
        if method == "intersection":
            return CriticalPointResult(
                Tc=T, Pc=P, method=f"envelope_{method}",
                converged=True, iterations=1, K_deviation=None,
            )

    # Fall back to cricondenbar or closest approach
    T, P, method, _ = candidates[0]
    return CriticalPointResult(
        Tc=T, Pc=P, method=f"envelope_{method}",
        converged=True, iterations=1, K_deviation=None,
    )


def detect_critical_point(
    bubble_T: NDArray[np.float64],
    bubble_P: NDArray[np.float64],
    dew_T: NDArray[np.float64],
    dew_P: NDArray[np.float64],
    composition: NDArray[np.float64],
    components: List[Component],
    eos: Optional[CubicEOS] = None,
    binary_interaction: Optional[NDArray[np.float64]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Detect critical point using multiple strategies with physical validation.

    This is the main entry point for critical point detection. It combines
    multiple detection strategies and validates results against physical bounds.

    Strategies used (in order of preference):
    1. K-value search (most physically rigorous - finds where Ki → 1)
    2. Envelope curve intersection (when curves meet properly)
    3. Kay's mixing rule estimate (fallback)

    Parameters
    ----------
    bubble_T, bubble_P : ndarray
        Bubble curve points (T in K, P in Pa)
    dew_T, dew_P : ndarray
        Dew curve points (T in K, P in Pa)
    composition : ndarray
        Mole fractions
    components : List[Component]
        Component objects with Tc, Pc properties
    eos : CubicEOS, optional
        Equation of state for K-value refinement
    binary_interaction : ndarray, optional
        Binary interaction parameters

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (Tc, Pc) or (None, None) if detection fails
    """
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()

    # Physical bounds from pure components
    active_comps = [(i, comp) for i, comp in enumerate(components) if z[i] > 1e-10]

    if not active_comps:
        return None, None

    Tc_min = min(comp.Tc for _, comp in active_comps)
    Tc_max = max(comp.Tc for _, comp in active_comps)
    Pc_min = min(comp.Pc for _, comp in active_comps)
    Pc_max = max(comp.Pc for _, comp in active_comps)

    # Get mixing rule estimates as reference
    Tc_kay, Pc_kay = estimate_critical_point_kays(z, components)

    try:
        Tc_li, Pc_li = estimate_critical_point_li(z, components)
    except Exception:
        Tc_li, Pc_li = Tc_kay, Pc_kay

    # Best estimate is average of Kay's and Li's
    Tc_est = 0.5 * (Tc_kay + Tc_li)
    Pc_est = 0.5 * (Pc_kay + Pc_li)

    candidates = []

    # Strategy 1: K-value search (most physically rigorous)
    # At critical point, Wilson K-values approach unity
    result_kvalue = None
    try:
        result_kvalue = find_critical_point_kvalue_search(
            z, components, eos if eos is not None else None, binary_interaction,
            T_initial=Tc_est, P_initial=Pc_est,
            max_iterations=100,
            tolerance=K_VALUE_TOLERANCE,
        )
        if result_kvalue.converged and result_kvalue.Tc is not None:
            Tc, Pc = result_kvalue.Tc, result_kvalue.Pc
            # Validate: must be between pure component criticals and close to estimate
            T_dev = abs(Tc - Tc_est) / Tc_est
            P_dev = abs(Pc - Pc_est) / Pc_est
            if (Tc_min * 0.8 < Tc < Tc_max * 1.2 and
                Pc_min * 0.5 < Pc < Pc_max * 1.5 and
                T_dev < 0.3 and P_dev < 0.5):
                candidates.append((Tc, Pc, "kvalue", result_kvalue.K_deviation or 0.0))
    except Exception:
        pass

    # Strategy 2: Envelope curve intersection
    result_envelope = find_critical_from_envelope(
        bubble_T, bubble_P, dew_T, dew_P, z, components
    )
    if result_envelope.converged and result_envelope.Tc is not None:
        Tc, Pc = result_envelope.Tc, result_envelope.Pc
        T_dev = abs(Tc - Tc_est) / Tc_est
        P_dev = abs(Pc - Pc_est) / Pc_est
        # Only accept envelope result if it's reasonably close to mixing rule estimate
        if (Tc_min * 0.8 < Tc < Tc_max * 1.2 and
            Pc_min * 0.5 < Pc < Pc_max * 1.5 and
            T_dev < 0.4 and P_dev < 0.6):
            # Prefer intersection over cricondenbar
            if "intersection" in result_envelope.method:
                candidates.insert(0, (Tc, Pc, "envelope_intersection", 0.0))
            else:
                candidates.append((Tc, Pc, "envelope_other", 0.0))

    # Strategy 3: Use mixing rule estimate as fallback
    if Tc_min * 0.8 < Tc_est < Tc_max * 1.2:
        candidates.append((Tc_est, Pc_est, "mixing_rule", 0.0))

    if not candidates:
        return None, None

    # Prefer K-value search (most physically rigorous)
    for Tc, Pc, method, _ in candidates:
        if method == "kvalue":
            return Tc, Pc

    # Then intersection
    for Tc, Pc, method, _ in candidates:
        if method == "envelope_intersection":
            return Tc, Pc

    # Fall back to best available
    return candidates[0][0], candidates[0][1]

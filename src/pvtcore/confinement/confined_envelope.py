"""Confined phase envelope calculation.

This module generates phase envelopes for nano-confined systems,
accounting for capillary pressure effects that shift bubble and dew curves.

Key effects of confinement:
- Bubble curve shifts to lower pressures (suppression)
- Dew curve shifts to higher pressures (enhancement)
- Two-phase region shrinks
- Cricondentherm and cricondenbar shift
- Critical point may shift

Units Convention:
- Pressure: Pa
- Temperature: K
- Pore radius: nm

References
----------
[1] Nojabaei, B., Johns, R.T., and Chu, L. (2013).
    "Effect of Capillary Pressure on Phase Behavior in Tight Rocks and Shales."
    SPE Reservoir Evaluation & Engineering, 16(3), 281-289. SPE-159258.
[2] Sandoval, D.R., Yan, W., Michelsen, M.L., and Stenby, E.H. (2016).
    "The Phase Envelope of Multicomponent Mixtures in the Presence of
    a Capillary Pressure Difference." Ind. Eng. Chem. Res., 55, 6530-6538.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ConvergenceError, PhaseError, ValidationError
from ..eos.base import CubicEOS
from ..models.component import Component
from ..envelope.phase_envelope import (
    calculate_phase_envelope,
    EnvelopeResult,
)
from .confined_flash import (
    confined_flash,
    confined_bubble_point,
    confined_dew_point,
)
from .capillary import capillary_pressure_simple


# Numerical parameters
DEFAULT_T_START: float = 150.0
DEFAULT_T_STEP: float = 10.0
MIN_T_STEP: float = 2.0
MAX_ENVELOPE_POINTS: int = 100


@dataclass
class ConfinedEnvelopeResult:
    """Results from confined phase envelope calculation.

    Attributes:
        bubble_T: Bubble curve temperatures (K)
        bubble_P: Bubble curve pressures (Pa) - liquid pressure
        bubble_Pc: Capillary pressures along bubble curve (Pa)
        dew_T: Dew curve temperatures (K)
        dew_P: Dew curve pressures (Pa) - liquid pressure
        dew_Pc: Capillary pressures along dew curve (Pa)
        bulk_envelope: Bulk (unconfined) phase envelope for comparison
        pore_radius: Pore radius used (nm)
        critical_T: Estimated confined critical temperature (K)
        critical_P: Estimated confined critical pressure (Pa)
        converged: True if envelope calculation succeeded
    """
    bubble_T: NDArray[np.float64]
    bubble_P: NDArray[np.float64]
    bubble_Pc: NDArray[np.float64]
    dew_T: NDArray[np.float64]
    dew_P: NDArray[np.float64]
    dew_Pc: NDArray[np.float64]
    bulk_envelope: Optional[EnvelopeResult]
    pore_radius: float
    critical_T: Optional[float]
    critical_P: Optional[float]
    converged: bool


def calculate_confined_envelope(
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    pore_radius_nm: float,
    binary_interaction: Optional[NDArray[np.float64]] = None,
    T_start: float = DEFAULT_T_START,
    T_step: float = DEFAULT_T_STEP,
    max_points: int = MAX_ENVELOPE_POINTS,
    include_bulk: bool = True,
) -> ConfinedEnvelopeResult:
    """Calculate phase envelope for confined system.

    Traces bubble and dew curves accounting for capillary pressure
    effects in nanopores.

    Parameters
    ----------
    composition : ndarray
        Feed mole fractions.
    components : list of Component
        Component objects.
    eos : CubicEOS
        Equation of state instance.
    pore_radius_nm : float
        Pore radius in nanometers.
    binary_interaction : ndarray, optional
        Binary interaction parameters.
    T_start : float
        Starting temperature for tracing (K).
    T_step : float
        Initial temperature step (K).
    max_points : int
        Maximum points per curve.
    include_bulk : bool
        If True, also calculate bulk envelope for comparison.

    Returns
    -------
    ConfinedEnvelopeResult
        Confined phase envelope results.

    Notes
    -----
    The confined envelope is narrower than the bulk envelope:
    - Bubble pressures are lower
    - Dew pressures are higher
    - The effect increases with smaller pore radii

    For very small pores (< 5 nm), the two-phase region may
    vanish entirely at some compositions.

    Examples
    --------
    >>> from pvtcore.models.component import load_components
    >>> from pvtcore.eos.peng_robinson import PengRobinsonEOS
    >>> components = load_components()
    >>> binary = [components['C1'], components['C4']]
    >>> eos = PengRobinsonEOS(binary)
    >>> z = np.array([0.7, 0.3])
    >>> result = calculate_confined_envelope(z, binary, eos, pore_radius_nm=10.0)
    >>> # Compare bubble points
    >>> print(f"Bulk Pb at 300K: {result.bulk_envelope.bubble_P[10]/1e6:.2f} MPa")
    >>> print(f"Confined Pb at 300K: {result.bubble_P[10]/1e6:.2f} MPa")
    """
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()

    # Validate inputs
    if pore_radius_nm <= 0:
        raise ValidationError(
            "Pore radius must be positive",
            parameter="pore_radius_nm",
            value=pore_radius_nm,
        )

    # Calculate bulk envelope for comparison and guidance
    bulk_envelope = None
    if include_bulk:
        try:
            bulk_envelope = calculate_phase_envelope(
                z, components, eos,
                binary_interaction=binary_interaction,
                T_start=T_start,
                T_step_initial=T_step,
            )
        except Exception as e:
            warnings.warn(f"Could not calculate bulk envelope: {e}")

    # Get temperature bounds from components
    Tc_max = max(comp.Tc for comp in components)
    T_max = Tc_max * 1.3

    # Trace confined bubble curve
    bubble_T, bubble_P, bubble_Pc = _trace_confined_bubble_curve(
        z, components, eos, pore_radius_nm,
        binary_interaction=binary_interaction,
        T_start=T_start,
        T_step=T_step,
        T_max=T_max,
        max_points=max_points,
        bulk_envelope=bulk_envelope,
    )

    # Trace confined dew curve
    dew_T, dew_P, dew_Pc = _trace_confined_dew_curve(
        z, components, eos, pore_radius_nm,
        binary_interaction=binary_interaction,
        T_start=T_start,
        T_step=T_step,
        T_max=T_max,
        max_points=max_points,
        bulk_envelope=bulk_envelope,
    )

    # Estimate critical point
    critical_T, critical_P = _estimate_confined_critical(
        bubble_T, bubble_P, dew_T, dew_P, bulk_envelope
    )

    converged = len(bubble_T) > 3 or len(dew_T) > 3

    return ConfinedEnvelopeResult(
        bubble_T=np.array(bubble_T),
        bubble_P=np.array(bubble_P),
        bubble_Pc=np.array(bubble_Pc),
        dew_T=np.array(dew_T),
        dew_P=np.array(dew_P),
        dew_Pc=np.array(dew_Pc),
        bulk_envelope=bulk_envelope,
        pore_radius=pore_radius_nm,
        critical_T=critical_T,
        critical_P=critical_P,
        converged=converged,
    )


def _trace_confined_bubble_curve(
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    pore_radius_nm: float,
    binary_interaction: Optional[NDArray[np.float64]],
    T_start: float,
    T_step: float,
    T_max: float,
    max_points: int,
    bulk_envelope: Optional[EnvelopeResult],
) -> Tuple[List[float], List[float], List[float]]:
    """Trace confined bubble curve."""
    T_list: List[float] = []
    P_list: List[float] = []
    Pc_list: List[float] = []

    T = T_start
    P_prev: Optional[float] = None
    consecutive_failures = 0

    for _ in range(max_points):
        if T > T_max:
            break

        try:
            # Get initial guess from bulk envelope if available
            if bulk_envelope is not None and len(bulk_envelope.bubble_T) > 0:
                # Find closest bulk point
                idx = np.argmin(np.abs(np.array(bulk_envelope.bubble_T) - T))
                P_init = float(bulk_envelope.bubble_P[idx])
            elif P_prev is not None:
                P_init = P_prev
            else:
                P_init = None

            # Calculate confined bubble point
            P_bubble, Pc, ift = confined_bubble_point(
                T, composition, components, eos, pore_radius_nm,
                binary_interaction=binary_interaction,
                P_initial=P_init,
            )

            if P_bubble > 0:
                T_list.append(float(T))
                P_list.append(float(P_bubble))
                Pc_list.append(float(Pc))
                P_prev = P_bubble
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        except (ConvergenceError, PhaseError, ValidationError):
            consecutive_failures += 1

        if consecutive_failures >= 5:
            break

        # Adaptive step
        if len(P_list) >= 2:
            dP_dT = abs(P_list[-1] - P_list[-2]) / T_step
            if dP_dT > 1e5:
                T_step = max(T_step * 0.7, MIN_T_STEP)
            elif dP_dT < 1e4:
                T_step = min(T_step * 1.3, 20.0)

        T += T_step

    return T_list, P_list, Pc_list


def _trace_confined_dew_curve(
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    pore_radius_nm: float,
    binary_interaction: Optional[NDArray[np.float64]],
    T_start: float,
    T_step: float,
    T_max: float,
    max_points: int,
    bulk_envelope: Optional[EnvelopeResult],
) -> Tuple[List[float], List[float], List[float]]:
    """Trace confined dew curve."""
    T_list: List[float] = []
    P_list: List[float] = []
    Pc_list: List[float] = []

    T = T_start
    P_prev: Optional[float] = None
    consecutive_failures = 0

    for _ in range(max_points):
        if T > T_max:
            break

        try:
            # Get initial guess from bulk envelope if available
            if bulk_envelope is not None and len(bulk_envelope.dew_T) > 0:
                idx = np.argmin(np.abs(np.array(bulk_envelope.dew_T) - T))
                P_init = float(bulk_envelope.dew_P[idx])
            elif P_prev is not None:
                P_init = P_prev
            else:
                P_init = None

            # Calculate confined dew point
            P_dew, Pc, ift = confined_dew_point(
                T, composition, components, eos, pore_radius_nm,
                binary_interaction=binary_interaction,
                P_initial=P_init,
            )

            if P_dew > 0:
                T_list.append(float(T))
                P_list.append(float(P_dew))
                Pc_list.append(float(Pc))
                P_prev = P_dew
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        except (ConvergenceError, PhaseError, ValidationError):
            consecutive_failures += 1

        if consecutive_failures >= 5:
            break

        if len(P_list) >= 2:
            dP_dT = abs(P_list[-1] - P_list[-2]) / T_step
            if dP_dT > 1e5:
                T_step = max(T_step * 0.7, MIN_T_STEP)
            elif dP_dT < 1e4:
                T_step = min(T_step * 1.3, 20.0)

        T += T_step

    return T_list, P_list, Pc_list


def _estimate_confined_critical(
    bubble_T: List[float],
    bubble_P: List[float],
    dew_T: List[float],
    dew_P: List[float],
    bulk_envelope: Optional[EnvelopeResult],
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate confined critical point."""
    if not bubble_T and not dew_T:
        return None, None

    # Method 1: Find where curves approach each other
    if len(bubble_T) > 2 and len(dew_T) > 2:
        # Find maximum bubble pressure
        idx_max_bubble = np.argmax(bubble_P)
        T_crit_bubble = bubble_T[idx_max_bubble]
        P_crit_bubble = bubble_P[idx_max_bubble]

        # Find maximum dew pressure
        idx_max_dew = np.argmax(dew_P)
        T_crit_dew = dew_T[idx_max_dew]
        P_crit_dew = dew_P[idx_max_dew]

        # Average as estimate
        T_crit = 0.5 * (T_crit_bubble + T_crit_dew)
        P_crit = 0.5 * (P_crit_bubble + P_crit_dew)

        return T_crit, P_crit

    # Method 2: Use bulk critical with adjustment
    if bulk_envelope is not None and bulk_envelope.critical_T is not None:
        # Confined critical is slightly shifted
        # Estimate shift based on typical Pc magnitudes
        T_crit = bulk_envelope.critical_T
        P_crit = bulk_envelope.critical_P
        return T_crit, P_crit

    return None, None


def estimate_envelope_shrinkage(
    pore_radius_nm: float,
    ift_typical: float = 10.0,
) -> dict:
    """Estimate phase envelope shrinkage due to confinement.

    Provides rough estimates of how much the envelope contracts
    for a given pore radius and typical IFT.

    Parameters
    ----------
    pore_radius_nm : float
        Pore radius in nanometers.
    ift_typical : float
        Typical IFT value in mN/m. Default 10 mN/m.

    Returns
    -------
    dict
        Dictionary with estimated shifts:
        - 'Pc_typical': Typical capillary pressure (Pa)
        - 'bubble_suppression': Approximate bubble point suppression (Pa)
        - 'dew_enhancement': Approximate dew point enhancement (Pa)
        - 'shrinkage_percent': Approximate two-phase region shrinkage (%)
    """
    Pc = capillary_pressure_simple(ift_typical, pore_radius_nm)

    return {
        'Pc_typical': Pc,
        'Pc_typical_MPa': Pc / 1e6,
        'bubble_suppression': Pc,  # ΔPb ≈ -Pc
        'dew_enhancement': Pc,  # ΔPd ≈ +Pc
        'shrinkage_percent': min(100.0, Pc / 1e6 * 10),  # Rough estimate
    }


def compare_bulk_confined(
    bulk_envelope: EnvelopeResult,
    confined_envelope: ConfinedEnvelopeResult,
) -> dict:
    """Compare bulk and confined phase envelopes.

    Parameters
    ----------
    bulk_envelope : EnvelopeResult
        Bulk phase envelope.
    confined_envelope : ConfinedEnvelopeResult
        Confined phase envelope.

    Returns
    -------
    dict
        Comparison metrics including shifts and ratios.
    """
    metrics = {}

    # Bubble point shifts at common temperatures
    if len(bulk_envelope.bubble_T) > 0 and len(confined_envelope.bubble_T) > 0:
        # Find common temperature range
        T_min = max(bulk_envelope.bubble_T[0], confined_envelope.bubble_T[0])
        T_max = min(bulk_envelope.bubble_T[-1], confined_envelope.bubble_T[-1])

        if T_max > T_min:
            T_mid = (T_min + T_max) / 2

            # Interpolate pressures at T_mid
            P_bulk = np.interp(T_mid, bulk_envelope.bubble_T, bulk_envelope.bubble_P)
            P_conf = np.interp(T_mid, confined_envelope.bubble_T, confined_envelope.bubble_P)

            metrics['bubble_shift_at_Tmid'] = P_conf - P_bulk
            metrics['bubble_shift_MPa'] = (P_conf - P_bulk) / 1e6
            metrics['T_mid'] = T_mid

    # Critical point shifts
    if bulk_envelope.critical_T and confined_envelope.critical_T:
        metrics['critical_T_shift'] = confined_envelope.critical_T - bulk_envelope.critical_T
        metrics['critical_P_shift'] = confined_envelope.critical_P - bulk_envelope.critical_P

    metrics['pore_radius_nm'] = confined_envelope.pore_radius

    return metrics

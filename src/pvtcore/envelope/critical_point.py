"""Critical-point helpers for multicomponent phase envelopes.

Important contract:
- the public phase-envelope critical-point surface must be **fail-closed**;
- helper estimates may exist for future diagnostics or solver seeding, but they
  must not be plotted as truth unless a stricter envelope-based certifier says a
  critical point is actually resolved.

In the current phase-1 implementation, the public `detect_critical_point(...)`
path uses a strict hot-end envelope-closure contract rather than heuristic
fallbacks such as mixing-rule, closest-approach, or cricondenbar proxies.

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


def _sorted_curve_arrays(
    temperatures: NDArray[np.float64],
    pressures: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return one curve sorted in ascending temperature order."""
    order = np.argsort(np.asarray(temperatures, dtype=np.float64))
    temps = np.asarray(temperatures, dtype=np.float64)[order]
    press = np.asarray(pressures, dtype=np.float64)[order]
    return temps, press


def _median_temperature_step(temperatures: NDArray[np.float64]) -> float:
    """Return a representative positive ΔT for one traced branch."""
    if len(temperatures) < 2:
        return TEMPERATURE_STEP
    diffs = np.diff(np.asarray(temperatures, dtype=np.float64))
    positive = diffs[diffs > 1.0e-12]
    if positive.size == 0:
        return TEMPERATURE_STEP
    return float(np.median(positive))



def find_critical_from_envelope(
    bubble_T: NDArray[np.float64],
    bubble_P: NDArray[np.float64],
    dew_T: NDArray[np.float64],
    dew_P: NDArray[np.float64],
    composition: NDArray[np.float64],
    components: List[Component],
    tolerance: float = PRESSURE_MATCH_TOLERANCE,
) -> CriticalPointResult:
    """Find a critical point only from a strict hot-end branch closure.

    This deliberately avoids heuristic fallbacks such as mixing-rule estimates,
    closest-approach guesses, cricondenbar proxies, or Wilson-K pseudo-critical
    searches. A critical point is returned only when the traced bubble and dew
    branches demonstrably meet at the hot end of their common temperature window.
    Otherwise the function fails closed.
    """
    _ = composition, components  # kept for API compatibility / future strict checks

    if len(bubble_P) == 0 or len(dew_P) == 0:
        return CriticalPointResult(
            Tc=None, Pc=None, method="strict_envelope_intersection",
            converged=False, iterations=0, K_deviation=None,
        )

    Tb_sorted, Pb_sorted = _sorted_curve_arrays(bubble_T, bubble_P)
    Td_sorted, Pd_sorted = _sorted_curve_arrays(dew_T, dew_P)

    T_overlap_min = max(float(np.min(Tb_sorted)), float(np.min(Td_sorted)))
    T_overlap_max = min(float(np.max(Tb_sorted)), float(np.max(Td_sorted)))
    if T_overlap_max <= T_overlap_min:
        return CriticalPointResult(
            Tc=None, Pc=None, method="strict_envelope_intersection",
            converged=False, iterations=0, K_deviation=None,
        )

    bubble_mask = (Tb_sorted >= T_overlap_min) & (Tb_sorted <= T_overlap_max)
    dew_mask = (Td_sorted >= T_overlap_min) & (Td_sorted <= T_overlap_max)
    candidate_temperatures = np.unique(
        np.concatenate([
            Tb_sorted[bubble_mask],
            Td_sorted[dew_mask],
        ])
    )
    if candidate_temperatures.size == 0:
        return CriticalPointResult(
            Tc=None, Pc=None, method="strict_envelope_intersection",
            converged=False, iterations=0, K_deviation=None,
        )

    Pb_interp = np.interp(candidate_temperatures, Tb_sorted, Pb_sorted)
    Pd_interp = np.interp(candidate_temperatures, Td_sorted, Pd_sorted)
    pressure_gap = np.abs(Pb_interp - Pd_interp)
    meeting = pressure_gap <= float(tolerance)
    if not np.any(meeting):
        return CriticalPointResult(
            Tc=None, Pc=None, method="strict_envelope_intersection",
            converged=False, iterations=1, K_deviation=None,
        )

    Tc_candidates = candidate_temperatures[meeting]
    Pb_candidates = Pb_interp[meeting]
    Pd_candidates = Pd_interp[meeting]
    idx_hot = int(np.argmax(Tc_candidates))
    Tcrit = float(Tc_candidates[idx_hot])
    Pcrit = float(0.5 * (Pb_candidates[idx_hot] + Pd_candidates[idx_hot]))

    span = max(T_overlap_max - T_overlap_min, 0.0)
    hot_step_scale = max(
        _median_temperature_step(Tb_sorted[bubble_mask]),
        _median_temperature_step(Td_sorted[dew_mask]),
        TEMPERATURE_STEP,
    )
    hot_gap_allowance = max(1.5 * hot_step_scale, 0.03 * span, 2.0)
    if (T_overlap_max - Tcrit) > hot_gap_allowance:
        return CriticalPointResult(
            Tc=None, Pc=None, method="strict_envelope_intersection",
            converged=False, iterations=1, K_deviation=None,
        )

    return CriticalPointResult(
        Tc=Tcrit,
        Pc=Pcrit,
        method="strict_envelope_intersection",
        converged=True,
        iterations=1,
        K_deviation=None,
    )


def _validate_hk_against_trace(
    hk_Tc: float,
    hk_Pc: float,
    bubble_T: NDArray[np.float64],
    bubble_P: NDArray[np.float64],
    dew_T: NDArray[np.float64],
    dew_P: NDArray[np.float64],
) -> bool:
    """Return True iff H-K's (Tc, Pc) is visually consistent with the traced envelope.

    Rejects results where the bubble and dew branches haven't converged toward
    the reported critical point. This prevents plotting a free-floating
    "critical" marker on an envelope whose branches never close.
    """
    Tb = np.asarray(bubble_T, dtype=np.float64)
    Pb = np.asarray(bubble_P, dtype=np.float64)
    Td = np.asarray(dew_T, dtype=np.float64)
    Pd = np.asarray(dew_P, dtype=np.float64)
    if Tb.size == 0 or Td.size == 0:
        return False

    Tb_max, Td_max = float(np.max(Tb)), float(np.max(Td))
    step = max(
        _median_temperature_step(Tb),
        _median_temperature_step(Td),
        TEMPERATURE_STEP,
    )
    reach_window = max(3.0 * step, 10.0)  # K
    if (hk_Tc - Tb_max) > reach_window or (hk_Tc - Td_max) > reach_window:
        return False

    Tb_sorted, Pb_sorted = _sorted_curve_arrays(Tb, Pb)
    Td_sorted, Pd_sorted = _sorted_curve_arrays(Td, Pd)
    T_eval = min(hk_Tc, Tb_max, Td_max)
    Pb_at = float(np.interp(T_eval, Tb_sorted, Pb_sorted))
    Pd_at = float(np.interp(T_eval, Td_sorted, Pd_sorted))
    P_gap = abs(Pb_at - Pd_at)
    P_scale = max(abs(hk_Pc), 1e5)
    if P_gap > 0.20 * P_scale:
        return False

    P_mid = 0.5 * (Pb_at + Pd_at)
    if abs(hk_Pc - P_mid) > 0.25 * P_scale:
        return False

    return True


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
    """Detect the mixture critical point.

    Primary path is the Heidemann-Khalil thermodynamic solver (needs only
    z, components, and the EOS — no reliance on the envelope trace). The
    traced branches are used only to seed (T, V): if the bubble/dew branches
    meet on their hot end, that meeting point becomes the H-K seed.

    After H-K converges, the result is validated against the traced branches
    via `_validate_hk_against_trace`. If the branches haven't reached H-K's
    (Tc, Pc) — as happens when the tracer terminates before closure on
    asymmetric mixtures — the H-K result is rejected to keep the public
    surface fail-closed and avoid plotting a disconnected critical marker.

    Fallback path is strict envelope-closure — used only when the H-K
    iteration fails, the EOS is not supplied, or H-K fails validation.
    If all paths fail, return (None, None).
    """
    if eos is not None:
        from .hk_critical import compute_critical_point

        T_seed: Optional[float] = None
        seed_result = find_critical_from_envelope(
            bubble_T, bubble_P, dew_T, dew_P, composition, components,
        )
        if seed_result.converged and seed_result.Tc is not None:
            T_seed = float(seed_result.Tc)

        try:
            hk = compute_critical_point(
                composition, components, eos,
                binary_interaction=binary_interaction,
                T_init=T_seed,
            )
        except Exception:
            hk = None

        if (
            hk is not None
            and hk.converged
            and hk.Tc is not None
            and hk.Pc is not None
            and _validate_hk_against_trace(
                float(hk.Tc), float(hk.Pc),
                bubble_T, bubble_P, dew_T, dew_P,
            )
        ):
            return float(hk.Tc), float(hk.Pc)

    result_envelope = find_critical_from_envelope(
        bubble_T, bubble_P, dew_T, dew_P, composition, components,
    )
    if not result_envelope.converged or result_envelope.Tc is None or result_envelope.Pc is None:
        return None, None
    return float(result_envelope.Tc), float(result_envelope.Pc)

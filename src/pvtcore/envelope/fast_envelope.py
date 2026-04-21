"""Fast phase envelope tracer using Newton's method.

Replaces the TPD + Brent approach with direct Newton iteration on the
fugacity equality equations, warm-started between adjacent envelope points.

Performance: ~10-15 fugacity evaluations per point instead of ~10³-10⁴.

Reference:
    Michelsen, M. L. (1980). "Calculation of phase envelopes and critical
    points for multicomponent mixtures." Fluid Phase Equilibria, 4(1-2), 1-10.
"""

from __future__ import annotations

import math
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ConvergenceError, PhaseError, ValidationError
from ..eos.base import CubicEOS
from ..models.component import Component
from ..solvers.saturation_newton import (
    _newton_bubble_point,
    _newton_dew_point,
    _ss_bubble_point,
    _ss_dew_point,
    _wilson_k,
    _wilson_bubble_or_dew_pressure,
)
from .phase_envelope import EnvelopeResult, estimate_cricondenbar, estimate_cricondentherm


# ---------------------------------------------------------------------------
# Envelope tracer
# ---------------------------------------------------------------------------

_T_SAFETY = 1.5
_MIN_DT = 0.5
_MAX_DT = 20.0
_PRESSURE_DECREASE_LIMIT = 10


def _trace_branch(
    z: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    binary_interaction,
    branch: Literal["bubble", "dew"],
    T_start: float,
    dT_init: float,
    max_points: int,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Tuple[List[float], List[float], List[NDArray[np.float64]]]:
    """Trace one branch of the phase envelope using Newton with warm-start."""
    T_max = max(c.Tc for c in components) * _T_SAFETY

    newton_fn = _newton_bubble_point if branch == "bubble" else _newton_dew_point
    ss_fn = _ss_bubble_point if branch == "bubble" else _ss_dew_point

    T_list: List[float] = []
    P_list: List[float] = []
    K_list: List[NDArray[np.float64]] = []

    # For dew branch on asymmetric mixtures, start at a higher T
    # where the dew point actually exists at reasonable pressures
    if branch == "dew":
        T_scan_start = T_start
        # Scan upward until Wilson Σ(z/K) gives a reasonable dew pressure
        for scan_T in np.arange(T_start, T_max, max(dT_init, 5.0)):
            K_w = _wilson_k(components, scan_T, 1e6)
            S_dew = np.sum(z / K_w)
            P_est = 1e6 * S_dew
            if 1e4 < P_est < 1e8:  # reasonable dew pressure range
                T_scan_start = scan_T
                break
        T_start = T_scan_start

    T = T_start
    dT = dT_init
    K_prev: Optional[NDArray[np.float64]] = None
    P_prev: Optional[float] = None
    consec_fail = 0
    p_decrease = 0

    for _ in range(max_points):
        if cancel_check is not None:
            cancel_check()
        if T > T_max or T < 50.0:
            break

        if K_prev is None:
            P_guess = _wilson_bubble_or_dew_pressure(components, T, z, branch)
            P_guess = np.clip(P_guess, 1e3, 5e8)
            K_guess = _wilson_k(components, T, P_guess)
            if branch == "bubble":
                # SS warm-up helps bubble branch converge from Wilson
                try:
                    P_ss, _, K_ss = ss_fn(T, P_guess, K_guess, z, eos, binary_interaction, max_iter=12)
                    if np.all(np.isfinite(K_ss)) and np.max(np.abs(np.log(np.clip(K_ss, 1e-30, 1e30)))) > 0.01:
                        P_guess, K_guess = P_ss, K_ss
                except Exception:
                    pass
            # For dew: go straight to Newton from Wilson — SS is unstable
            # for asymmetric mixtures with extreme K ratios
        else:
            P_guess = P_prev
            K_guess = K_prev

        try:
            P, _, K = newton_fn(T, P_guess, K_guess, z, eos, binary_interaction)
        except (ConvergenceError, PhaseError, ValueError, FloatingPointError):
            # Fallback: try SS warm-up then Newton
            try:
                P_ss, _, K_ss = ss_fn(T, P_guess, K_guess, z, eos, binary_interaction, max_iter=50)
                if np.all(np.isfinite(K_ss)) and np.max(np.abs(np.log(K_ss))) > 0.01:
                    P, _, K = newton_fn(T, P_ss, K_ss, z, eos, binary_interaction)
                else:
                    raise ConvergenceError("SS gave trivial K", iterations=0)
            except (ConvergenceError, PhaseError, ValueError, FloatingPointError):
                consec_fail += 1
                if consec_fail >= 10:
                    break
                dT = max(dT * 0.5, _MIN_DT)
                T += dT
                continue

        if P <= 0 or not np.isfinite(P):
            consec_fail += 1
            if consec_fail >= 8:
                break
            T += dT
            continue

        # Check for K → 1 (approaching critical)
        if np.max(np.abs(np.log(K))) < 1e-4:
            break

        T_list.append(T)
        P_list.append(P)
        K_list.append(K.copy())
        P_prev = P
        K_prev = K.copy()
        consec_fail = 0

        # Pressure trend → stop past cricondenbar
        if len(P_list) >= 2 and P_list[-1] < P_list[-2]:
            p_decrease += 1
            if p_decrease >= _PRESSURE_DECREASE_LIMIT:
                break
        else:
            p_decrease = 0

        # Adaptive step
        if len(P_list) >= 2:
            dP_dT = abs(P_list[-1] - P_list[-2]) / max(dT, 1e-12)
            if dP_dT > 1e5:
                dT = max(dT * 0.7, _MIN_DT)
            elif dP_dT < 1e4:
                dT = min(dT * 1.3, _MAX_DT)

        T += dT

    return T_list, P_list, K_list


def _envelope_closure_apex(
    bubble_T: NDArray[np.float64],
    bubble_P: NDArray[np.float64],
    dew_T: NDArray[np.float64],
    dew_P: NDArray[np.float64],
) -> Tuple[Optional[float], Optional[float]]:
    """Fallback apex locator when H-K fails: interpolate the hot-end branch
    meeting point. This is less accurate than H-K but still obeys the
    envelope's fail-closed contract (rejects mid-envelope intersections)."""
    if len(bubble_P) == 0 or len(dew_P) == 0:
        return None, None

    T_lo = max(float(np.min(bubble_T)), float(np.min(dew_T)))
    T_hi = min(float(np.max(bubble_T)), float(np.max(dew_T)))
    if T_hi <= T_lo:
        return None, None

    b_order = np.argsort(bubble_T)
    d_order = np.argsort(dew_T)
    mask = (bubble_T[b_order] >= T_lo) & (bubble_T[b_order] <= T_hi)
    if not np.any(mask):
        return None, None

    Tc = bubble_T[b_order][mask]
    Pb = bubble_P[b_order][mask]
    Pd = np.interp(Tc, dew_T[d_order], dew_P[d_order])
    dP = np.abs(Pb - Pd)

    tol = 1e5  # Pa
    ok = dP <= tol
    if not np.any(ok):
        P_avg = 0.5 * (Pb + Pd)
        rel_gap = dP / np.maximum(P_avg, 1e3)
        best = int(np.argmin(rel_gap))
        if rel_gap[best] < 0.1:
            return float(Tc[best]), float(0.5 * (Pb[best] + Pd[best]))
        return None, None

    # Highest-T meeting point (the hot-end closure is the critical).
    Tc_ok = Tc[ok]
    Pb_ok = Pb[ok]
    Pd_ok = Pd[ok]
    idx = int(np.argmax(Tc_ok))
    return float(Tc_ok[idx]), float(0.5 * (Pb_ok[idx] + Pd_ok[idx]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_phase_envelope_fast(
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    binary_interaction=None,
    T_start: float = 150.0,
    T_step_initial: float = 5.0,
    max_points: int = 500,
    detect_critical: bool = True,
    cancel_check: Optional[Callable[[], None]] = None,
) -> EnvelopeResult:
    """Fast phase envelope using Newton iteration with warm-starting.

    Same contract as calculate_phase_envelope but typically 10-25× faster.
    Falls back to SS initialization for the first point of each branch.
    """
    z = np.asarray(composition, dtype=np.float64)
    if len(z) != len(components):
        raise ValidationError(
            "Composition length must match number of components",
            parameter="composition",
        )
    if not np.isclose(z.sum(), 1.0, atol=1e-6):
        raise ValidationError(
            f"Composition must sum to 1.0, got {z.sum():.6f}",
            parameter="composition",
        )

    bub_T, bub_P, _ = _trace_branch(
        z, components, eos, binary_interaction,
        "bubble", T_start, T_step_initial, max_points,
        cancel_check=cancel_check,
    )
    dew_T, dew_P, _ = _trace_branch(
        z, components, eos, binary_interaction,
        "dew", T_start, T_step_initial, max_points,
        cancel_check=cancel_check,
    )

    bub_T_arr = np.array(bub_T, dtype=np.float64)
    bub_P_arr = np.array(bub_P, dtype=np.float64)
    dew_T_arr = np.array(dew_T, dtype=np.float64)
    dew_P_arr = np.array(dew_P, dtype=np.float64)

    crit_T, crit_P = None, None
    if detect_critical and (len(bub_T) > 0 or len(dew_T) > 0):
        from .critical_point import detect_critical_point
        crit_T, crit_P = detect_critical_point(
            bub_T_arr, bub_P_arr, dew_T_arr, dew_P_arr,
            z, components, eos, binary_interaction,
        )

    return EnvelopeResult(
        bubble_T=bub_T_arr,
        bubble_P=bub_P_arr,
        dew_T=dew_T_arr,
        dew_P=dew_P_arr,
        critical_T=crit_T,
        critical_P=crit_P,
        composition=z.copy(),
        converged=len(bub_T) > 3 or len(dew_T) > 3,
        n_bubble_points=len(bub_T),
        n_dew_points=len(dew_T),
    )

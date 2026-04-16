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
from typing import List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.constants import R
from ..core.errors import ConvergenceError, PhaseError, ValidationError
from ..eos.base import CubicEOS
from ..models.component import Component
from .phase_envelope import EnvelopeResult, estimate_cricondenbar, estimate_cricondentherm


# ---------------------------------------------------------------------------
# Wilson K-value initial estimate
# ---------------------------------------------------------------------------

def _wilson_bubble_or_dew_pressure(
    components: List[Component], T: float, z: NDArray[np.float64], branch: str,
) -> float:
    """Wilson-correlation estimate of bubble or dew pressure."""
    nc = len(components)
    # K_i(P=1) ∝ Pc_i · exp(5.373·(1+ω_i)·(1-Tc_i/T))
    # Bubble: P_bub = Σ z_i · K_i(P=1) · 1 Pa  →  P_bub = Σ z_i Pc_i exp(...)
    # Dew:    P_dew = 1 / Σ z_i / (K_i(P=1))  →  P_dew = 1 / Σ z_i/(Pc_i exp(...))
    K_at_1Pa = np.array([
        components[i].Pc * math.exp(5.373 * (1.0 + components[i].omega) * (1.0 - components[i].Tc / T))
        for i in range(nc)
    ])
    if branch == "bubble":
        return float(np.sum(z * K_at_1Pa))
    else:
        denom = np.sum(z / K_at_1Pa)
        return float(1.0 / denom) if denom > 1e-30 else 1e6


def _wilson_k(components: List[Component], T: float, P: float) -> NDArray[np.float64]:
    """Wilson correlation K-values for initial estimate."""
    K = np.empty(len(components))
    for i, c in enumerate(components):
        K[i] = (c.Pc / P) * math.exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T))
    return K


# ---------------------------------------------------------------------------
# Newton solver for a single saturation point
# ---------------------------------------------------------------------------

def _newton_bubble_point(
    T: float,
    P_init: float,
    K_init: NDArray[np.float64],
    z: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction=None,
    max_iter: int = 20,
    tol: float = 1e-10,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Solve bubble-point equations by Newton's method.

    Equations (n_c + 1 unknowns: ln(K_1)..ln(K_nc), ln(P)):
        F_i = ln(K_i) + ln(φ_i^V(y)) - ln(φ_i^L(z)) = 0,  i = 1..n_c
        g   = Σ(z_i · K_i) - 1 = 0

    Returns (P, y, K) at convergence.
    """
    nc = len(z)
    ln_K = np.log(K_init)
    ln_P = math.log(P_init)

    for iteration in range(max_iter):
        P = math.exp(ln_P)
        K = np.exp(ln_K)
        y = z * K
        y_sum = y.sum()
        y = y / y_sum

        try:
            ln_phi_L = eos.ln_fugacity_coefficient(P, T, z, "liquid", binary_interaction)
            ln_phi_V = eos.ln_fugacity_coefficient(P, T, y, "vapor", binary_interaction)
        except PhaseError:
            raise ConvergenceError(
                "EOS evaluation failed in Newton bubble-point solver",
                iterations=iteration, temperature=T, pressure=P,
            )

        F = ln_K + ln_phi_V - ln_phi_L
        g = np.sum(z * K) - 1.0

        residual = np.max(np.abs(F))
        if residual < tol and abs(g) < tol:
            return P, y, K

        # Build Jacobian (n_c+1 × n_c+1)
        # Variables: [ln(K_1), ..., ln(K_nc), ln(P)]
        J = np.zeros((nc + 1, nc + 1))

        # ∂F_i/∂ln(K_j): identity + ∂ln(φ_i^V)/∂n_j · ∂y/∂ln(K)
        # y_j = z_j K_j / S where S = Σ z_k K_k
        # ∂y_j/∂ln(K_m) = y_j (δ_jm - y_m)·(z_m K_m / S) ... simplifies to:
        # ∂y_j/∂ln(K_m) = δ_jm·y_j - y_j·y_m  (at S=1 converged) but S≠1 mid-iter
        # More precisely: let S = Σ z_k K_k, y_j = z_j K_j / S
        # ∂y_j/∂ln(K_m) = z_m K_m (δ_jm/S - z_j K_j/S²) = z_m K_m (δ_jm - y_j)/S
        # = (z_m K_m / S) (δ_jm - y_j)
        S = np.sum(z * K)
        dy_dlnK = np.zeros((nc, nc))
        for m in range(nc):
            w_m = z[m] * K[m] / S
            for j in range(nc):
                dy_dlnK[j, m] = w_m * ((1.0 if j == m else 0.0) - y[j])

        # ∂ln(φ_i^V)/∂y_j matrix
        dlnphi_V_dn = eos.d_ln_phi_dn(P, T, y, "vapor", binary_interaction)

        # ∂F_i/∂ln(K_m) = δ_im + Σ_j ∂ln(φ_i^V)/∂n_j · ∂y_j/∂ln(K_m)
        J[:nc, :nc] = np.eye(nc) + dlnphi_V_dn @ dy_dlnK

        # ∂F_i/∂ln(P) = P · (∂ln(φ_i^V)/∂P - ∂ln(φ_i^L)/∂P)
        dlnphi_V_dP = eos.d_ln_phi_dP(P, T, y, "vapor", binary_interaction)
        dlnphi_L_dP = eos.d_ln_phi_dP(P, T, z, "liquid", binary_interaction)
        J[:nc, nc] = P * (dlnphi_V_dP - dlnphi_L_dP)

        # ∂g/∂ln(K_m) = z_m K_m
        J[nc, :nc] = z * K

        # ∂g/∂ln(P) = 0
        J[nc, nc] = 0.0

        rhs = np.empty(nc + 1)
        rhs[:nc] = -F
        rhs[nc] = -g

        try:
            delta = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            raise ConvergenceError(
                "Singular Jacobian in Newton bubble-point solver",
                iterations=iteration, temperature=T, pressure=P,
            )

        # Damping for large steps
        max_step = 2.0
        scale = min(1.0, max_step / (np.max(np.abs(delta)) + 1e-30))
        ln_K += scale * delta[:nc]
        ln_P += scale * delta[nc]

    raise ConvergenceError(
        "Newton bubble-point did not converge",
        iterations=max_iter, temperature=T,
    )


def _newton_dew_point(
    T: float,
    P_init: float,
    K_init: NDArray[np.float64],
    z: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction=None,
    max_iter: int = 20,
    tol: float = 1e-10,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Solve dew-point equations by Newton's method.

    Equations:
        F_i = ln(K_i) + ln(φ_i^V(z)) - ln(φ_i^L(x)) = 0
        g   = 1 - Σ(z_i / K_i) = 0

    Returns (P, x, K) at convergence.
    """
    nc = len(z)
    ln_K = np.log(K_init)
    ln_P = math.log(P_init)

    for iteration in range(max_iter):
        P = math.exp(ln_P)
        K = np.exp(ln_K)
        x = z / K
        x_sum = x.sum()
        x = x / x_sum

        try:
            ln_phi_L = eos.ln_fugacity_coefficient(P, T, x, "liquid", binary_interaction)
            ln_phi_V = eos.ln_fugacity_coefficient(P, T, z, "vapor", binary_interaction)
        except PhaseError:
            raise ConvergenceError(
                "EOS evaluation failed in Newton dew-point solver",
                iterations=iteration, temperature=T, pressure=P,
            )

        F = ln_K + ln_phi_V - ln_phi_L
        g = 1.0 - np.sum(z / K)

        residual = np.max(np.abs(F))
        if residual < tol and abs(g) < tol:
            return P, x, K

        J = np.zeros((nc + 1, nc + 1))

        # x_j = z_j / (K_j · S) where S = Σ z_k / K_k
        # ∂x_j/∂ln(K_m) = -(z_m/K_m)/S · (δ_jm·(z_j/(K_j·S·z_m/K_m)) ... )
        # Simpler: x_j = z_j/(K_j S), ∂x_j/∂ln(K_m) = x_j(x_m - δ_jm)·(z_m/(K_m S))
        S = np.sum(z / K)
        dx_dlnK = np.zeros((nc, nc))
        for m in range(nc):
            w_m = z[m] / (K[m] * S)
            for j in range(nc):
                # ∂x_j/∂ln(K_m) = -w_m·(δ_jm - x_j)  (negative because K in denominator)
                dx_dlnK[j, m] = w_m * (x[j] - (1.0 if j == m else 0.0))

        dlnphi_L_dn = eos.d_ln_phi_dn(P, T, x, "liquid", binary_interaction)

        # ∂F_i/∂ln(K_m) = δ_im - Σ_j ∂ln(φ_i^L)/∂n_j · ∂x_j/∂ln(K_m)
        J[:nc, :nc] = np.eye(nc) - dlnphi_L_dn @ dx_dlnK

        # ∂F_i/∂ln(P) = P · (∂ln(φ_i^V)/∂P - ∂ln(φ_i^L)/∂P)
        dlnphi_V_dP = eos.d_ln_phi_dP(P, T, z, "vapor", binary_interaction)
        dlnphi_L_dP = eos.d_ln_phi_dP(P, T, x, "liquid", binary_interaction)
        J[:nc, nc] = P * (dlnphi_V_dP - dlnphi_L_dP)

        # ∂g/∂ln(K_m) = z_m / K_m
        J[nc, :nc] = z / K

        J[nc, nc] = 0.0

        rhs = np.empty(nc + 1)
        rhs[:nc] = -F
        rhs[nc] = -g

        try:
            delta = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            raise ConvergenceError(
                "Singular Jacobian in Newton dew-point solver",
                iterations=iteration, temperature=T, pressure=P,
            )

        max_step = 2.0
        scale = min(1.0, max_step / (np.max(np.abs(delta)) + 1e-30))
        ln_K += scale * delta[:nc]
        ln_P += scale * delta[nc]

    raise ConvergenceError(
        "Newton dew-point did not converge",
        iterations=max_iter, temperature=T,
    )


# ---------------------------------------------------------------------------
# Successive substitution fallback for first point (robust initialization)
# ---------------------------------------------------------------------------

def _ss_bubble_point(
    T: float,
    P_init: float,
    K_init: NDArray[np.float64],
    z: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction=None,
    max_iter: int = 30,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """A few SS iterations to bring K-values near the solution basin."""
    K = K_init.copy()
    P = P_init
    for _ in range(max_iter):
        y = z * K
        y_sum = y.sum()
        if y_sum <= 0 or not np.isfinite(y_sum):
            break
        y = y / y_sum
        try:
            phi_L = eos.fugacity_coefficient(P, T, z, "liquid", binary_interaction)
            phi_V = eos.fugacity_coefficient(P, T, y, "vapor", binary_interaction)
        except (PhaseError, ValueError):
            break
        if not (np.all(np.isfinite(phi_L)) and np.all(np.isfinite(phi_V))):
            break
        mask = phi_V > 0
        if not np.all(mask):
            break
        K_new = phi_L / phi_V
        if not np.all(np.isfinite(K_new)):
            break
        S = np.sum(z * K_new)
        if S <= 0 or not np.isfinite(S):
            break
        P_new = P / S
        if P_new <= 0 or not np.isfinite(P_new):
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = K_new / K
        if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
            K = K_new
            P = P_new
            break
        change = np.max(np.abs(np.log(ratio)))
        K = K_new
        P = P_new
        if change < 1e-4:
            break
    return P, z * K / max((z * K).sum(), 1e-30), K


def _ss_dew_point(
    T: float,
    P_init: float,
    K_init: NDArray[np.float64],
    z: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction=None,
    max_iter: int = 30,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """SS warm-up for dew point."""
    K = K_init.copy()
    P = P_init
    for _ in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore"):
            x = z / K
        x_sum = x.sum()
        if x_sum <= 0 or not np.isfinite(x_sum):
            break
        x = x / x_sum
        if not np.all(np.isfinite(x)):
            break
        try:
            phi_L = eos.fugacity_coefficient(P, T, x, "liquid", binary_interaction)
            phi_V = eos.fugacity_coefficient(P, T, z, "vapor", binary_interaction)
        except (PhaseError, ValueError):
            break
        if not (np.all(np.isfinite(phi_L)) and np.all(np.isfinite(phi_V))):
            break
        mask = phi_V > 0
        if not np.all(mask):
            break
        K_new = phi_L / phi_V
        if not np.all(np.isfinite(K_new)) or np.any(K_new <= 0):
            break
        S = np.sum(z / K_new)
        if S <= 0 or not np.isfinite(S):
            break
        P_new = P * S
        if P_new <= 0 or not np.isfinite(P_new):
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = K_new / K
        if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
            K = K_new
            P = P_new
            break
        change = np.max(np.abs(np.log(ratio)))
        K = K_new
        P = P_new
        if change < 1e-4:
            break
    x_out = z / K
    x_sum = x_out.sum()
    if x_sum > 0 and np.isfinite(x_sum):
        x_out = x_out / x_sum
    return P, x_out, K


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


def _locate_critical(
    bubble_T: NDArray[np.float64],
    bubble_P: NDArray[np.float64],
    dew_T: NDArray[np.float64],
    dew_P: NDArray[np.float64],
) -> Tuple[Optional[float], Optional[float]]:
    """Find critical point where bubble and dew curves converge.

    The critical point is the highest-pressure point where the bubble and
    dew pressures are within tolerance at the same temperature.  This avoids
    spurious low-pressure intersections that can occur at low T.
    """
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
        # Relax tolerance: use smallest relative pressure gap
        P_avg = 0.5 * (Pb + Pd)
        rel_gap = dP / np.maximum(P_avg, 1e3)
        best = int(np.argmin(rel_gap))
        if rel_gap[best] < 0.1:  # within 10% relative
            return float(Tc[best]), float(0.5 * (Pb[best] + Pd[best]))
        return None, None

    # Among meeting points, pick the one at highest pressure (true critical)
    Tc_ok = Tc[ok]
    Pb_ok = Pb[ok]
    Pd_ok = Pd[ok]
    P_avg_ok = 0.5 * (Pb_ok + Pd_ok)
    idx = int(np.argmax(P_avg_ok))
    return float(Tc_ok[idx]), float(P_avg_ok[idx])


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
    )
    dew_T, dew_P, _ = _trace_branch(
        z, components, eos, binary_interaction,
        "dew", T_start, T_step_initial, max_points,
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

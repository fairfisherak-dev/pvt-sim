"""Shared saturation-point Newton solvers (Michelsen-style).

Jacobian uses analytic :math:`\\partial\\ln\\phi/\\partial P` and
:math:`\\partial\\ln\\phi/\\partial n` from :class:`~pvtcore.eos.base.CubicEOS`.

This module is the single implementation for Newton-first bubble/dew solves;
callers include ``pvtcore.flash`` (bubble/dew point), ``pvtcore.envelope``
(continuation, fast envelope tracer), and any future saturation entrypoints.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ConvergenceError, PhaseError
from ..eos.base import CubicEOS
from ..models.component import Component


# ---------------------------------------------------------------------------
# Wilson K-value initial estimate
# ---------------------------------------------------------------------------


def _wilson_bubble_or_dew_pressure(
    components: List[Component], T: float, z: NDArray[np.float64], branch: str,
) -> float:
    """Wilson-correlation estimate of bubble or dew pressure."""
    nc = len(components)
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

        J = np.zeros((nc + 1, nc + 1))

        S = np.sum(z * K)
        dy_dlnK = np.zeros((nc, nc))
        for m in range(nc):
            w_m = z[m] * K[m] / S
            for j in range(nc):
                dy_dlnK[j, m] = w_m * ((1.0 if j == m else 0.0) - y[j])

        dlnphi_V_dn = eos.d_ln_phi_dn(P, T, y, "vapor", binary_interaction)

        J[:nc, :nc] = np.eye(nc) + dlnphi_V_dn @ dy_dlnK

        dlnphi_V_dP = eos.d_ln_phi_dP(P, T, y, "vapor", binary_interaction)
        dlnphi_L_dP = eos.d_ln_phi_dP(P, T, z, "liquid", binary_interaction)
        J[:nc, nc] = P * (dlnphi_V_dP - dlnphi_L_dP)

        J[nc, :nc] = z * K

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

        S = np.sum(z / K)
        dx_dlnK = np.zeros((nc, nc))
        for m in range(nc):
            w_m = z[m] / (K[m] * S)
            for j in range(nc):
                dx_dlnK[j, m] = w_m * (x[j] - (1.0 if j == m else 0.0))

        dlnphi_L_dn = eos.d_ln_phi_dn(P, T, x, "liquid", binary_interaction)

        J[:nc, :nc] = np.eye(nc) - dlnphi_L_dn @ dx_dlnK

        dlnphi_V_dP = eos.d_ln_phi_dP(P, T, z, "vapor", binary_interaction)
        dlnphi_L_dP = eos.d_ln_phi_dP(P, T, x, "liquid", binary_interaction)
        J[:nc, nc] = P * (dlnphi_V_dP - dlnphi_L_dP)

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

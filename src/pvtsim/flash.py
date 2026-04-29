"""PT flash, bubble point, fugacity convergence (slides 339-345, 383-386)."""

import numpy as np

from pvtsim.constants import FLASH_TOL, MAX_ITER, SAT_TOL
from pvtsim.eos import AB, fugacity_coef, mix_params, phase_params, Z_liquid, Z_vapor
from pvtsim.helpers import K_wilson, solve_rachford_rice


def _K_from_fugacity(x, y, P, T, comps):
    """K = φ_L / φ_V from EOS fugacity coefficients (slide 381)."""
    aL, bL, ai, bi = phase_params(x, comps, T)
    aV, bV, _, _ = phase_params(y, comps, T)
    AL, BL = AB(aL, bL, P, T)
    AV, BV = AB(aV, bV, P, T)
    ZL = Z_liquid(AL, BL)
    ZV = Z_vapor(AV, BV)
    n = len(comps)
    kij = np.zeros((n, n))
    phiL = fugacity_coef(x, AL, BL, ZL, ai, bi, aL, bL, kij)
    phiV = fugacity_coef(y, AV, BV, ZV, ai, bi, aV, bV, kij)
    return phiL / phiV, ZL, ZV


def pt_flash(z, P, T, comps):
    """Two-phase PT flash (slides 386, 387).

    Returns dict with keys: converged, nL, nV, x, y, K, ZL, ZV.
    """
    z = np.asarray(z, dtype=float)
    Tc = np.array([c[0] for c in comps])
    Pc = np.array([c[1] for c in comps])
    w = np.array([c[2] for c in comps])

    K = K_wilson(Pc, Tc, w, P, T)

    for _ in range(MAX_ITER):
        beta = solve_rachford_rice(z, K)
        if beta <= 0.0:
            return {"converged": True, "nL": 1.0, "nV": 0.0, "x": z.copy(),
                    "y": np.zeros_like(z), "K": K, "ZL": np.nan, "ZV": np.nan, "phase": "L"}
        if beta >= 1.0:
            return {"converged": True, "nL": 0.0, "nV": 1.0,
                    "x": np.zeros_like(z), "y": z.copy(),
                    "K": K, "ZL": np.nan, "ZV": np.nan, "phase": "V"}

        x = z / (1.0 + beta * (K - 1.0))
        y = K * x
        x /= x.sum(); y /= y.sum()

        K_new, ZL, ZV = _K_from_fugacity(x, y, P, T, comps)

        if np.sum((np.log(K_new) - np.log(K)) ** 2) < FLASH_TOL:
            return {"converged": True, "nL": 1.0 - beta, "nV": beta,
                    "x": x, "y": y, "K": K_new, "ZL": ZL, "ZV": ZV, "phase": "LV"}
        K = K_new

    return {"converged": False, "nL": 1.0 - beta, "nV": beta,
            "x": x, "y": y, "K": K, "ZL": ZL, "ZV": ZV, "phase": "LV"}


def bubble_point(z, T, comps, P_guess=None):
    """Bubble point pressure at fixed T (slides 341, 383-384).

    Iterate: K from fugacity @ x=z, y∝Kz; update P by Σ Kᵢzᵢ = 1.
    """
    z = np.asarray(z, dtype=float)
    Tc = np.array([c[0] for c in comps])
    Pc = np.array([c[1] for c in comps])
    w = np.array([c[2] for c in comps])

    # Ideal-gas Wilson initial guess for Pb (slide 341)
    if P_guess is None:
        P = 1.0 / np.sum(z / (Pc * np.exp(5.373 * (1 + w) * (1 - Tc / T))))
    else:
        P = P_guess

    for _ in range(MAX_ITER):
        K = K_wilson(Pc, Tc, w, P, T)
        # Inner loop: tighten K via fugacity with x=z
        for _ in range(MAX_ITER):
            y = K * z
            y = y / y.sum()
            K_new, _, _ = _K_from_fugacity(z, y, P, T, comps)
            if np.sum((np.log(K_new) - np.log(K)) ** 2) < FLASH_TOL:
                K = K_new
                break
            K = K_new

        sumKz = np.sum(K * z)
        if abs(sumKz - 1.0) < SAT_TOL:
            return P
        P = P * sumKz

    return P

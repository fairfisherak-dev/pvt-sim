"""
Katz plus-fraction splitting method.

Implements the Katz (1983) exponential splitting model for C7+ fractions.
This is a simpler alternative to Pedersen that uses fixed coefficients.

Contract:
    z_n = A * exp(-B * n)

where A and B are determined from material balance constraints.

References
----------
[1] Katz, D.L. (1983). "Overview of Phase Behavior in Oil and Gas Production."
    Journal of Petroleum Technology, 35(6), 1205-1214.
[2] Whitson, C.H. and Brule, M.R. (2000). "Phase Behavior." SPE Monograph Vol. 20.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class KatzSplitResult:
    """
    Result container for Katz plus-fraction splitting.

    Attributes
    ----------
    n : ndarray
        SCN indices (e.g., 7, 8, 9, ...).
    MW : ndarray
        Molecular weights for each SCN in g/mol.
    z : ndarray
        Mole fractions for each SCN.
    A : float
        Fitted pre-exponential coefficient.
    B : float
        Fitted decay constant.
    """
    n: np.ndarray
    MW: np.ndarray
    z: np.ndarray
    A: float
    B: float


def _default_scn_mw(n: np.ndarray) -> np.ndarray:
    """Default SCN molecular weight: MW_n = 14*n - 4 (paraffin baseline)."""
    return 14.0 * n.astype(float) - 4.0


def split_plus_fraction_katz(
    *,
    z_plus: float,
    MW_plus: float,
    n_start: int = 7,
    n_end: int = 45,
    scn_mw_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> KatzSplitResult:
    """
    Split a plus fraction using Katz exponential distribution.

    The Katz method assumes:
        z_n = A * exp(-B * n)

    where A and B are found by satisfying:
        1. sum(z_n) = z_plus (mole fraction constraint)
        2. sum(z_n * MW_n) = z_plus * MW_plus (MW constraint)

    Parameters
    ----------
    z_plus : float
        Plus fraction mole fraction (must be > 0).
    MW_plus : float
        Plus fraction molecular weight in g/mol (must be > 0).
    n_start : int
        Starting SCN (default 7 for C7+).
    n_end : int
        Ending SCN (default 45).
    scn_mw_fn : callable, optional
        Function to compute MW from SCN indices. If None, uses MW = 14*n - 4.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    KatzSplitResult
        SCN indices, MWs, mole fractions, and fitted A, B.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If Newton iteration fails to converge.

    Notes
    -----
    The original Katz (1983) used fixed coefficients:
        z_n = 1.38205 * z_C7+ * exp(-0.25903 * n)

    This implementation solves for A and B to match both z_plus and MW_plus,
    making it more flexible.

    Examples
    --------
    >>> result = split_plus_fraction_katz(z_plus=0.25, MW_plus=215.0)
    >>> print(f"C7: z={result.z[0]:.4f}, MW={result.MW[0]:.1f}")
    """
    # Validate inputs
    if z_plus <= 0:
        raise ValueError(f"z_plus must be > 0, got {z_plus}")
    if MW_plus <= 0:
        raise ValueError(f"MW_plus must be > 0, got {MW_plus}")
    if n_end < n_start:
        raise ValueError(f"n_end must be >= n_start")

    n = np.arange(n_start, n_end + 1, dtype=int)
    mw_fn = scn_mw_fn or _default_scn_mw
    MW = np.asarray(mw_fn(n), dtype=float)

    if MW.shape != n.shape:
        raise ValueError("scn_mw_fn must return array same shape as n")
    if not np.isfinite(MW).all() or np.any(MW <= 0):
        raise ValueError("MW_n must be finite and > 0 for all SCNs")

    # Target constraints
    target_z = z_plus
    target_zMW = z_plus * MW_plus

    # Initial guess using classic Katz coefficients
    # z_n = 1.38205 * z_plus * exp(-0.25903 * n)
    A = 1.38205 * z_plus
    B = 0.25903

    def compute_z(A: float, B: float) -> np.ndarray:
        """Compute z_n = A * exp(-B * n)."""
        u = -B * n.astype(float)
        u = np.clip(u, -700, 700)  # Guard overflow
        return A * np.exp(u)

    def eval_residuals(A: float, B: float):
        """Evaluate constraint residuals."""
        z = compute_z(A, B)
        F1 = z.sum() - target_z
        F2 = (z * MW).sum() - target_zMW
        return z, F1, F2

    # Newton iteration to satisfy both constraints
    for iteration in range(max_iter):
        z, F1, F2 = eval_residuals(A, B)

        if abs(F1) < tol and abs(F2) < tol:
            break

        # Jacobian:
        # dF1/dA = sum(exp(-B*n)) = sum(z)/A
        # dF1/dB = sum(-n * A * exp(-B*n)) = -sum(n * z)
        # dF2/dA = sum(MW * exp(-B*n)) = sum(z*MW)/A
        # dF2/dB = sum(-n * A * MW * exp(-B*n)) = -sum(n * z * MW)
        dF1_dA = z.sum() / A if A != 0 else 1e10
        dF1_dB = -(n.astype(float) * z).sum()
        dF2_dA = (z * MW).sum() / A if A != 0 else 1e10
        dF2_dB = -(n.astype(float) * z * MW).sum()

        J = np.array([[dF1_dA, dF1_dB],
                      [dF2_dA, dF2_dB]], dtype=float)
        F = np.array([F1, F2], dtype=float)

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Katz split Newton step failed (singular Jacobian) at iteration {iteration}"
            ) from e

        dA, dB = delta

        # Line search with backtracking
        step = 1.0
        r_curr = np.hypot(F1, F2)

        for _ in range(25):
            A_try = A + step * dA
            B_try = B + step * dB

            # Keep A and B positive
            if A_try <= 0:
                step *= 0.5
                continue
            if B_try <= 0:
                B_try = 0.01  # Minimum decay

            _, F1_try, F2_try = eval_residuals(A_try, B_try)
            r_try = np.hypot(F1_try, F2_try)

            if np.isfinite(r_try) and r_try <= r_curr:
                A, B = A_try, B_try
                break

            step *= 0.5
        else:
            raise RuntimeError(
                f"Katz split line search failed at iteration {iteration}, residual={r_curr:.3e}"
            )
    else:
        raise RuntimeError(
            f"Katz split did not converge after {max_iter} iterations"
        )

    # Final calculation with hard normalization
    z = compute_z(A, B)
    z *= (z_plus / z.sum())  # Enforce exact mole fraction sum

    # Sanity checks
    if not np.isfinite(z).all() or np.any(z <= 0):
        raise RuntimeError("Katz split produced non-finite or non-positive z_n")

    return KatzSplitResult(n=n, MW=MW, z=z, A=float(A), B=float(B))


def katz_classic_split(
    *,
    z_plus: float,
    n_start: int = 7,
    n_end: int = 45,
    scn_mw_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> KatzSplitResult:
    """
    Split using the original Katz (1983) fixed coefficients.

    This uses the classic formula:
        z_n = 1.38205 * z_plus * exp(-0.25903 * n)

    No iteration is needed since coefficients are fixed.

    Parameters
    ----------
    z_plus : float
        Plus fraction mole fraction.
    n_start : int
        Starting SCN (default 7).
    n_end : int
        Ending SCN (default 45).
    scn_mw_fn : callable, optional
        Function to compute MW from SCN indices.

    Returns
    -------
    KatzSplitResult
        Split result with classic Katz coefficients.

    Notes
    -----
    This does NOT satisfy material balance constraints exactly.
    Use split_plus_fraction_katz() for constrained splitting.
    """
    if z_plus <= 0:
        raise ValueError(f"z_plus must be > 0, got {z_plus}")

    n = np.arange(n_start, n_end + 1, dtype=int)
    mw_fn = scn_mw_fn or _default_scn_mw
    MW = np.asarray(mw_fn(n), dtype=float)

    # Classic Katz coefficients
    A = 1.38205 * z_plus
    B = 0.25903

    z = A * np.exp(-B * n.astype(float))

    # Normalize to exactly match z_plus
    z *= (z_plus / z.sum())

    return KatzSplitResult(n=n, MW=MW, z=z, A=A, B=B)

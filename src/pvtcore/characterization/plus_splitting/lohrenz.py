"""
Lohrenz plus-fraction splitting method.

Implements the Lohrenz et al. (1964) quadratic-exponential splitting model.
This method uses a quadratic exponent which can provide better fits for
some heavy oil compositions.

Contract:
    z_n = z_6 * exp(A*(n-6)² + B*(n-6))

where A and B are determined from material balance constraints.

References
----------
[1] Lohrenz, J., Bray, B.G., and Clark, C.R. (1964). "Calculating Viscosities
    of Reservoir Fluids from Their Compositions." Journal of Petroleum
    Technology, 16(10), 1171-1176.
[2] Whitson, C.H. and Brule, M.R. (2000). "Phase Behavior." SPE Monograph Vol. 20.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class LohrenzSplitResult:
    """
    Result container for Lohrenz plus-fraction splitting.

    Attributes
    ----------
    n : ndarray
        SCN indices (e.g., 7, 8, 9, ...).
    MW : ndarray
        Molecular weights for each SCN in g/mol.
    z : ndarray
        Mole fractions for each SCN.
    A : float
        Fitted quadratic coefficient.
    B : float
        Fitted linear coefficient.
    z_ref : float
        Reference mole fraction (z_6 or z_n_start).
    """
    n: np.ndarray
    MW: np.ndarray
    z: np.ndarray
    A: float
    B: float
    z_ref: float


def _default_scn_mw(n: np.ndarray) -> np.ndarray:
    """Default SCN molecular weight: MW_n = 14*n - 4 (paraffin baseline)."""
    return 14.0 * n.astype(float) - 4.0


def split_plus_fraction_lohrenz(
    *,
    z_plus: float,
    MW_plus: float,
    z_ref: Optional[float] = None,
    n_ref: int = 6,
    n_start: int = 7,
    n_end: int = 45,
    scn_mw_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> LohrenzSplitResult:
    """
    Split a plus fraction using Lohrenz quadratic-exponential distribution.

    The Lohrenz method assumes:
        z_n = z_ref * exp(A*(n-n_ref)² + B*(n-n_ref))

    where A and B are found by satisfying material balance constraints.

    Parameters
    ----------
    z_plus : float
        Plus fraction mole fraction (must be > 0).
    MW_plus : float
        Plus fraction molecular weight in g/mol (must be > 0).
    z_ref : float, optional
        Reference mole fraction (typically z_6). If None, it's estimated
        from z_plus.
    n_ref : int
        Reference carbon number (default 6).
    n_start : int
        Starting SCN for the plus fraction (default 7 for C7+).
    n_end : int
        Ending SCN (default 45).
    scn_mw_fn : callable, optional
        Function to compute MW from SCN indices.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    LohrenzSplitResult
        SCN indices, MWs, mole fractions, and fitted parameters.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If Newton iteration fails to converge.

    Notes
    -----
    The quadratic term (A < 0 typically) causes the distribution to decay
    faster than pure exponential for very heavy components, which is more
    physically realistic for petroleum systems.

    Examples
    --------
    >>> result = split_plus_fraction_lohrenz(z_plus=0.25, MW_plus=215.0)
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
    n_float = n.astype(float)
    delta_n = n_float - float(n_ref)

    mw_fn = scn_mw_fn or _default_scn_mw
    MW = np.asarray(mw_fn(n), dtype=float)

    if MW.shape != n.shape:
        raise ValueError("scn_mw_fn must return array same shape as n")
    if not np.isfinite(MW).all() or np.any(MW <= 0):
        raise ValueError("MW_n must be finite and > 0 for all SCNs")

    # Estimate z_ref if not provided
    # Use relation: z_ref ~ z_plus / number_of_SCNs * 2 (rough estimate)
    if z_ref is None:
        z_ref = z_plus / len(n) * 3.0  # Starting estimate

    # Target constraints
    target_z = z_plus
    target_zMW = z_plus * MW_plus

    # Initial guess: approximately linear decay
    A = -0.001  # Small negative for slight curvature
    B = -0.3    # Linear decay

    def compute_z(z_r: float, A: float, B: float) -> np.ndarray:
        """Compute z_n = z_ref * exp(A*(n-n_ref)² + B*(n-n_ref))."""
        u = A * (delta_n ** 2) + B * delta_n
        u = np.clip(u, -700, 700)  # Guard overflow
        return z_r * np.exp(u)

    def eval_residuals(z_r: float, A: float, B: float):
        """Evaluate constraint residuals."""
        z = compute_z(z_r, A, B)
        F1 = z.sum() - target_z
        F2 = (z * MW).sum() - target_zMW
        return z, F1, F2

    # We have 3 unknowns (z_ref, A, B) but only 2 constraints.
    # Fix z_ref and solve for A, B in inner loop.
    # Outer loop adjusts z_ref if needed.

    for outer_iter in range(max_iter):
        # Inner Newton for A, B with fixed z_ref
        for inner_iter in range(max_iter):
            z, F1, F2 = eval_residuals(z_ref, A, B)

            if abs(F1) < tol and abs(F2) < tol:
                break

            # Jacobian for A, B (z_ref fixed)
            # dz/dA = z * (n-n_ref)²
            # dz/dB = z * (n-n_ref)
            dz_dA = z * (delta_n ** 2)
            dz_dB = z * delta_n

            dF1_dA = dz_dA.sum()
            dF1_dB = dz_dB.sum()
            dF2_dA = (dz_dA * MW).sum()
            dF2_dB = (dz_dB * MW).sum()

            J = np.array([[dF1_dA, dF1_dB],
                          [dF2_dA, dF2_dB]], dtype=float)
            F = np.array([F1, F2], dtype=float)

            # Check for singular Jacobian
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-20:
                # Singular - adjust z_ref and restart
                z_ref *= 1.1
                A = -0.001
                B = -0.3
                break

            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                z_ref *= 1.1
                A = -0.001
                B = -0.3
                break

            dA, dB = delta

            # Line search
            step = 1.0
            r_curr = np.hypot(F1, F2)

            for _ in range(25):
                A_try = A + step * dA
                B_try = B + step * dB

                _, F1_try, F2_try = eval_residuals(z_ref, A_try, B_try)
                r_try = np.hypot(F1_try, F2_try)

                if np.isfinite(r_try) and r_try <= r_curr:
                    A, B = A_try, B_try
                    break

                step *= 0.5
            else:
                # Line search failed - try adjusting z_ref
                z_ref *= 0.9
                A = -0.001
                B = -0.3
                break
        else:
            # Inner loop converged
            break

        # Check outer convergence
        z, F1, F2 = eval_residuals(z_ref, A, B)
        if abs(F1) < tol and abs(F2) < tol:
            break
    else:
        raise RuntimeError(
            f"Lohrenz split did not converge after {max_iter} outer iterations"
        )

    # Final calculation with hard normalization
    z = compute_z(z_ref, A, B)
    z *= (z_plus / z.sum())  # Enforce exact mole fraction sum

    # Sanity checks
    if not np.isfinite(z).all() or np.any(z <= 0):
        raise RuntimeError("Lohrenz split produced non-finite or non-positive z_n")

    return LohrenzSplitResult(
        n=n,
        MW=MW,
        z=z,
        A=float(A),
        B=float(B),
        z_ref=float(z_ref),
    )


def lohrenz_classic_coefficients(
    *,
    z_6: float,
    n_start: int = 7,
    n_end: int = 45,
    A: float = -0.001,
    B: float = -0.25,
    scn_mw_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> LohrenzSplitResult:
    """
    Split using user-specified Lohrenz coefficients.

    This allows using pre-determined A and B coefficients from
    characterization or regression.

    Parameters
    ----------
    z_6 : float
        Mole fraction of C6 (reference component).
    n_start : int
        Starting SCN (default 7).
    n_end : int
        Ending SCN (default 45).
    A : float
        Quadratic coefficient (typically small negative).
    B : float
        Linear coefficient (typically negative).
    scn_mw_fn : callable, optional
        Function to compute MW from SCN indices.

    Returns
    -------
    LohrenzSplitResult
        Split result with specified coefficients.

    Notes
    -----
    This does NOT satisfy material balance constraints.
    Use split_plus_fraction_lohrenz() for constrained splitting.
    """
    if z_6 <= 0:
        raise ValueError(f"z_6 must be > 0, got {z_6}")

    n = np.arange(n_start, n_end + 1, dtype=int)
    delta_n = n.astype(float) - 6.0

    mw_fn = scn_mw_fn or _default_scn_mw
    MW = np.asarray(mw_fn(n), dtype=float)

    z = z_6 * np.exp(A * (delta_n ** 2) + B * delta_n)

    return LohrenzSplitResult(
        n=n,
        MW=MW,
        z=z,
        A=A,
        B=B,
        z_ref=z_6,
    )

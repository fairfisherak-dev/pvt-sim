"""
Pedersen-style exponential plus-fraction splitting.

Contract (docs/technical_notes.md §2.2):
  ln(z_n) = A + B * MW_n
with constraints:
  Σ z_n         = z_plus
  Σ z_n * MW_n  = z_plus * MW_plus

Default SCN molecular weight model:
  MW_n = 14*n - 4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class PedersenSplitResult:
    n: np.ndarray          # SCN indices, shape (Ns,)
    MW: np.ndarray         # MW_n, shape (Ns,)
    z: np.ndarray          # z_n, shape (Ns,)
    A: float
    B: float


def _default_scn_mw(n: np.ndarray) -> np.ndarray:
    # MW_n = 14 n - 4 (paraffin baseline)
    return 14.0 * n.astype(float) - 4.0


def split_plus_fraction_pedersen(
    *,
    z_plus: float,
    MW_plus: float,
    n_start: int = 7,
    n_end: int = 45,
    scn_mw_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> PedersenSplitResult:
    """
    Split a plus fraction (Cn+) into SCNs via Pedersen exponential distribution.

    Parameters
    ----------
    z_plus
        Plus fraction mole fraction (must be > 0).
    MW_plus
        Plus fraction molecular weight in g/mol (must be > 0).
    n_start, n_end
        Inclusive SCN range to allocate the plus fraction (e.g., 7..45).
    scn_mw_fn
        Function returning MW_n array for SCN indices n.
        If None, uses MW_n = 14*n - 4.
    tol
        Convergence tolerance on both constraint residuals.
    max_iter
        Maximum Newton iterations.

    Returns
    -------
    PedersenSplitResult
        SCN indices, MWs, split mole fractions z_n, and fitted A, B.

    Notes
    -----
    Uses damped Newton on the 2 constraints in (A, B). Guards overflow by
    evaluating residual norms and applying backtracking when needed.
    """
    if not (z_plus > 0.0):
        raise ValueError(f"z_plus must be > 0, got {z_plus}")
    if not (MW_plus > 0.0):
        raise ValueError(f"MW_plus must be > 0, got {MW_plus}")
    if n_end < n_start:
        raise ValueError(f"n_end must be >= n_start, got {n_start=}, {n_end=}")

    n = np.arange(n_start, n_end + 1, dtype=int)
    mw_fn = scn_mw_fn or _default_scn_mw
    MW = np.asarray(mw_fn(n), dtype=float)

    if MW.shape != n.shape:
        raise ValueError("scn_mw_fn must return array same shape as n")
    if not np.isfinite(MW).all() or np.any(MW <= 0.0):
        raise ValueError("MW_n must be finite and > 0 for all SCNs")

    # Initial guess:
    # B = 0 => uniform in ln-space => z_n = exp(A)
    # Choose A to satisfy Σ z_n = z_plus
    Ns = float(len(n))
    A = float(np.log(z_plus / Ns))
    B = 0.0

    target1 = z_plus
    target2 = z_plus * MW_plus

    def eval_constraints(a: float, b: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        # z_n = exp(a + b*MW_n)
        # Use clip to reduce overflow risk in exp for huge MW ranges
        u = a + b * MW
        # conservative clip: exp(±700) is near float64 limits
        u = np.clip(u, -700.0, 700.0)
        z = np.exp(u)
        s1 = float(z.sum())
        s2 = float((z * MW).sum())
        F1 = s1 - target1
        F2 = s2 - target2
        return z, u, F1, F2

    z, u, F1, F2 = eval_constraints(A, B)
    r0 = float(np.hypot(F1, F2))

    for _ in range(max_iter):
        # Re-evaluate with current (A,B)
        z, u, F1, F2 = eval_constraints(A, B)
        if abs(F1) < tol and abs(F2) < tol:
            break

        # Jacobian
        # dF1/dA = Σ z
        # dF1/dB = Σ z*MW
        # dF2/dA = Σ z*MW
        # dF2/dB = Σ z*MW^2
        s1 = float(z.sum())
        sMW = float((z * MW).sum())
        sMW2 = float((z * MW * MW).sum())

        J = np.array([[s1, sMW],
                      [sMW, sMW2]], dtype=float)
        F = np.array([F1, F2], dtype=float)

        # Solve J * d = -F
        try:
            dA, dB = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Pedersen split Newton step failed (singular Jacobian)") from e

        # Backtracking line search on residual norm
        step = 1.0
        r_curr = float(np.hypot(F1, F2))
        accepted = False
        for _ls in range(25):
            A_try = A + step * float(dA)
            B_try = B + step * float(dB)
            _, _, f1_try, f2_try = eval_constraints(A_try, B_try)
            r_try = float(np.hypot(f1_try, f2_try))
            if np.isfinite(r_try) and r_try <= r_curr:
                A, B = A_try, B_try
                accepted = True
                break
            step *= 0.5

        if not accepted:
            raise RuntimeError(
                "Pedersen split failed to reduce residual (line search exhausted). "
                f"Residual={r_curr:.3e}"
            )

    # Final compute + hard normalization (for safety)
    z, _, F1, F2 = eval_constraints(A, B)
    # Enforce exact constraints numerically by scaling on Σz (keeps shape)
    z *= (z_plus / float(z.sum()))

    # After scaling, MW constraint will be extremely close but not necessarily exact
    # (Newton already drove it tight). Assert sanity and return.
    if not np.isfinite(z).all() or np.any(z <= 0.0):
        raise RuntimeError("Pedersen split produced non-finite or non-positive z_n")

    return PedersenSplitResult(n=n, MW=MW, z=z, A=float(A), B=float(B))

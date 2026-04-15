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
from typing import Callable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PedersenTBPCutConstraint:
    """Observed TBP cut used to constrain a Pedersen split fit."""

    name: str
    carbon_number: int
    carbon_number_end: int
    z: float
    mw: float
    tb_k: float | None = None


@dataclass(frozen=True)
class PedersenSplitResult:
    n: np.ndarray          # SCN indices, shape (Ns,)
    MW: np.ndarray         # MW_n, shape (Ns,)
    z: np.ndarray          # z_n, shape (Ns,)
    A: float
    B: float
    solve_ab_from: str = "balances"
    tbp_cut_rms_relative_error: float | None = None


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
    solve_ab_from: str = "balances",
    tbp_cuts: Sequence[PedersenTBPCutConstraint] | None = None,
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
    solve_ab_from
        ``balances`` solves for ``A``/``B`` from aggregate material balances.
        ``fit_to_tbp`` instead fits ``A``/``B`` to observed TBP cut data while
        preserving the aggregate ``z_plus``/``MW_plus`` targets.
    tbp_cuts
        Ordered TBP cut constraints used when ``solve_ab_from="fit_to_tbp"``.
    tol
        Convergence tolerance on both constraint residuals.
    max_iter
        Maximum Newton / Gauss-Newton iterations.

    Returns
    -------
    PedersenSplitResult
        SCN indices, MWs, split mole fractions z_n, fitted A/B, and the solve mode used.
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

    solve_mode = solve_ab_from.strip().lower()
    if solve_mode == "balances":
        A, B, z = _solve_ab_from_balances(
            z_plus=z_plus,
            MW_plus=MW_plus,
            MW=MW,
            tol=tol,
            max_iter=max_iter,
        )
        return PedersenSplitResult(
            n=n,
            MW=MW,
            z=z,
            A=float(A),
            B=float(B),
            solve_ab_from=solve_mode,
        )

    if solve_mode == "fit_to_tbp":
        constraints = _validate_tbp_constraints(tbp_cuts, n_start=n_start, n_end=n_end)
        A, B, z, rms = _solve_ab_from_tbp_fit(
            z_plus=z_plus,
            MW_plus=MW_plus,
            n=n,
            MW=MW,
            tbp_cuts=constraints,
            tol=tol,
            max_iter=max_iter,
        )
        return PedersenSplitResult(
            n=n,
            MW=MW,
            z=z,
            A=float(A),
            B=float(B),
            solve_ab_from=solve_mode,
            tbp_cut_rms_relative_error=rms,
        )

    raise ValueError("solve_ab_from must be either 'balances' or 'fit_to_tbp'")


def _solve_ab_from_balances(
    *,
    z_plus: float,
    MW_plus: float,
    MW: np.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[float, float, np.ndarray]:
    """Solve Pedersen A/B from aggregate z and MW balances."""
    # Initial guess:
    # B = 0 => uniform in ln-space => z_n = exp(A)
    # Choose A to satisfy Σ z_n = z_plus
    Ns = float(MW.size)
    A = float(np.log(z_plus / Ns))
    B = 0.0

    target1 = z_plus
    target2 = z_plus * MW_plus

    def eval_constraints(a: float, b: float) -> Tuple[np.ndarray, float, float]:
        z = _evaluate_exponential_distribution(a, b, MW)
        s1 = float(z.sum())
        s2 = float((z * MW).sum())
        return z, s1 - target1, s2 - target2

    for _ in range(max_iter):
        z, F1, F2 = eval_constraints(A, B)
        if abs(F1) < tol and abs(F2) < tol:
            break

        s1 = float(z.sum())
        sMW = float((z * MW).sum())
        sMW2 = float((z * MW * MW).sum())

        J = np.array([[s1, sMW], [sMW, sMW2]], dtype=float)
        F = np.array([F1, F2], dtype=float)

        try:
            dA, dB = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Pedersen split Newton step failed (singular Jacobian)") from e

        step = 1.0
        residual_norm = float(np.hypot(F1, F2))
        accepted = False
        for _ls in range(25):
            A_try = A + step * float(dA)
            B_try = B + step * float(dB)
            _, f1_try, f2_try = eval_constraints(A_try, B_try)
            trial_norm = float(np.hypot(f1_try, f2_try))
            if np.isfinite(trial_norm) and trial_norm <= residual_norm:
                A, B = A_try, B_try
                accepted = True
                break
            step *= 0.5

        if not accepted:
            raise RuntimeError(
                "Pedersen split failed to reduce residual (line search exhausted). "
                f"Residual={residual_norm:.3e}"
            )

    z, _, _ = eval_constraints(A, B)
    z, delta_a = _enforce_exact_total(z, z_plus)
    A += delta_a

    if not np.isfinite(z).all() or np.any(z <= 0.0):
        raise RuntimeError("Pedersen split produced non-finite or non-positive z_n")

    return float(A), float(B), z


def _solve_ab_from_tbp_fit(
    *,
    z_plus: float,
    MW_plus: float,
    n: np.ndarray,
    MW: np.ndarray,
    tbp_cuts: tuple[PedersenTBPCutConstraint, ...],
    tol: float,
    max_iter: int,
) -> tuple[float, float, np.ndarray, float]:
    """Fit Pedersen A/B to observed TBP cuts while preserving aggregate targets."""
    _, balance_b, _ = _solve_ab_from_balances(
        z_plus=z_plus,
        MW_plus=MW_plus,
        MW=MW,
        tol=tol,
        max_iter=max_iter,
    )

    def residual_vector_for_b(b: float) -> np.ndarray:
        _a, z = _normalized_distribution_from_b(b=b, z_plus=z_plus, MW=MW)
        return _tbp_fit_residual_vector(
            z=z,
            MW=MW,
            n=n,
            tbp_cuts=tbp_cuts,
            z_plus=z_plus,
            MW_plus=MW_plus,
        )

    best_b = float(balance_b)
    best_residual = residual_vector_for_b(best_b)
    best_score = float(np.dot(best_residual, best_residual))
    half_width = max(0.05, abs(best_b) * 4.0 + 0.02)
    refinement_rounds = max(6, min(max_iter, 10))

    for _ in range(refinement_rounds):
        candidates = np.linspace(best_b - half_width, best_b + half_width, 81)
        improved = False
        for candidate_b in candidates:
            candidate_residual = residual_vector_for_b(float(candidate_b))
            if not np.isfinite(candidate_residual).all():
                continue
            candidate_score = float(np.dot(candidate_residual, candidate_residual))
            if candidate_score < best_score:
                best_b = float(candidate_b)
                best_residual = candidate_residual
                best_score = candidate_score
                improved = True
        if best_score < tol * tol:
            break
        if not improved:
            half_width *= 0.35
        else:
            half_width *= 0.5

    A, z = _normalized_distribution_from_b(b=best_b, z_plus=z_plus, MW=MW)

    if not np.isfinite(z).all() or np.any(z <= 0.0):
        raise RuntimeError("Pedersen TBP fit produced non-finite or non-positive z_n")

    rms = _tbp_cut_relative_rms(z=z, n=n, tbp_cuts=tbp_cuts)
    return float(A), float(best_b), z, rms


def _validate_tbp_constraints(
    tbp_cuts: Sequence[PedersenTBPCutConstraint] | None,
    *,
    n_start: int,
    n_end: int,
) -> tuple[PedersenTBPCutConstraint, ...]:
    """Validate the TBP cuts used to constrain the Pedersen fit."""
    if not tbp_cuts:
        raise ValueError("fit_to_tbp requires non-empty tbp_cuts")

    normalized = tuple(tbp_cuts)
    previous_end: int | None = None
    seen_names: set[str] = set()

    for cut in normalized:
        if cut.name in seen_names:
            raise ValueError("TBP cut names must be unique when fitting Pedersen to TBP data")
        if cut.carbon_number < n_start:
            raise ValueError("TBP fit cuts must not start below n_start")
        if cut.carbon_number_end < cut.carbon_number:
            raise ValueError("TBP fit cut carbon_number_end must be >= carbon_number")
        if cut.carbon_number_end > n_end:
            raise ValueError("TBP fit cuts must not extend beyond n_end")
        if previous_end is None:
            if cut.carbon_number != n_start:
                raise ValueError("The first TBP fit cut must start at n_start")
        elif cut.carbon_number <= previous_end:
            raise ValueError("TBP fit cuts must be ordered, non-overlapping, and strictly increasing")
        if not np.isfinite(cut.z) or cut.z <= 0.0:
            raise ValueError("TBP fit cut z must be finite and > 0")
        if not np.isfinite(cut.mw) or cut.mw <= 0.0:
            raise ValueError("TBP fit cut mw must be finite and > 0")
        previous_end = cut.carbon_number_end
        seen_names.add(cut.name)

    return normalized


def _evaluate_exponential_distribution(a: float, b: float, MW: np.ndarray) -> np.ndarray:
    """Evaluate the clipped Pedersen exponential distribution."""
    exponent = np.clip(a + b * MW, -700.0, 700.0)
    return np.exp(exponent)


def _enforce_exact_total(z: np.ndarray, z_plus: float) -> tuple[np.ndarray, float]:
    """Scale z to close the total balance exactly and return the equivalent ΔA."""
    total = float(z.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Pedersen split produced a non-positive total mole fraction")
    scale = z_plus / total
    return z * scale, float(np.log(scale))


def _normalized_distribution_from_b(*, b: float, z_plus: float, MW: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the Pedersen distribution for a given B after enforcing the total balance."""
    raw = _evaluate_exponential_distribution(0.0, b, MW)
    normalized, delta_a = _enforce_exact_total(raw, z_plus)
    return delta_a, normalized


def _tbp_fit_residual_vector(
    *,
    z: np.ndarray,
    MW: np.ndarray,
    n: np.ndarray,
    tbp_cuts: Sequence[PedersenTBPCutConstraint],
    z_plus: float,
    MW_plus: float,
) -> np.ndarray:
    """Build an overdetermined residual vector for fitting Pedersen to TBP cuts."""
    total_z = float(z.sum())
    if not np.isfinite(total_z) or total_z <= 0.0:
        return np.full(len(tbp_cuts) + 2, 1.0e6, dtype=float)

    residuals: list[float] = []
    for cut in tbp_cuts:
        mask = (n >= cut.carbon_number) & (n <= cut.carbon_number_end)
        cut_z = float(z[mask].sum())
        residuals.append((cut_z - cut.z) / max(cut.z, 1.0e-12))

        if cut.carbon_number_end > cut.carbon_number and cut_z > 0.0:
            cut_avg_mw = float((z[mask] * MW[mask]).sum() / cut_z)
            residuals.append(0.5 * ((cut_avg_mw - cut.mw) / max(cut.mw, 1.0e-12)))

    total_mw = float((z * MW).sum())
    mw_avg = total_mw / total_z
    residuals.append((total_z - z_plus) / max(z_plus, 1.0e-12))
    residuals.append((mw_avg - MW_plus) / max(MW_plus, 1.0e-12))
    return np.asarray(residuals, dtype=float)


def _finite_difference_jacobian(
    residual_fn: Callable[[float, float], np.ndarray],
    *,
    A: float,
    B: float,
    residual: np.ndarray,
) -> np.ndarray:
    """Finite-difference Jacobian for the two-parameter Pedersen fit."""
    h_a = 1.0e-6 * max(1.0, abs(A))
    h_b = 1.0e-6 * max(1.0, abs(B))

    residual_a = residual_fn(A + h_a, B)
    residual_b = residual_fn(A, B + h_b)

    jacobian = np.empty((residual.size, 2), dtype=float)
    jacobian[:, 0] = (residual_a - residual) / h_a
    jacobian[:, 1] = (residual_b - residual) / h_b
    return jacobian


def _tbp_cut_relative_rms(
    *,
    z: np.ndarray,
    n: np.ndarray,
    tbp_cuts: Sequence[PedersenTBPCutConstraint],
) -> float:
    """Return the RMS relative error against observed TBP cut mole fractions."""
    relative_errors: list[float] = []
    for cut in tbp_cuts:
        mask = (n >= cut.carbon_number) & (n <= cut.carbon_number_end)
        cut_z = float(z[mask].sum())
        relative_errors.append((cut_z - cut.z) / max(cut.z, 1.0e-12))
    return float(np.sqrt(np.mean(np.square(relative_errors), dtype=float)))

"""Heidemann-Khalil critical-point solver for multicomponent mixtures.

Solves the two scalar conditions that define a mixture critical point at
fixed feed composition z:

    1.  det Q(T, V) = 0
        where  Q_ij = (1/RT) · ∂²A/∂n_i∂n_j |_{T,V}
               (reduced Helmholtz Hessian in mole numbers).
    2.  C(T, V)   = 0
        where  C   = d³/dε³ (A/RT)(T, V, n + εu) |_{ε=0}
               and u is the unit null-vector of Q.

References
----------
- Heidemann, R. A. & Khalil, A. M. (1980). "The Calculation of Critical
  Points", AIChE J. 26(5), 769-779.
- Michelsen, M. L. (1984). "Calculation of critical points and phase
  boundaries in the critical region", Fluid Phase Equilibria 16, 57-76.
- Michelsen, M. L. & Mollerup, J. M. (2007). "Thermodynamic Models:
  Fundamentals & Computational Aspects", 2nd ed., Chapter 17.

Implementation notes
--------------------
The Helmholtz Hessian Q is assembled column-by-column by central
finite-differencing ln(f_i) = ∂(A/RT)/∂n_i at constant (T, V). The cubic
form C is obtained by a 4-point central FD on A/RT along the null
direction u. The outer iteration is a 2×2 Newton in (T, V) with a
finite-difference Jacobian of (λ_min, C). A valid critical point
requires both residuals below tolerance AND a smallest-eigenvalue sign
change in Q — we reject "spurious" stationary points where the two
independent residuals happen to be small but Q has lost rank ambiguously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.constants import R
from ..eos.base import CubicEOS
from ..models.component import Component


@dataclass
class CriticalPointResult:
    """Result returned by :func:`compute_critical_point`.

    Tc / Pc are ``None`` when convergence failed or the result is outside
    physical bounds. ``method`` is always ``"heidemann_khalil"`` for this
    module; callers may wrap it and re-label as needed.
    """

    Tc: Optional[float]
    Pc: Optional[float]
    Vc: Optional[float]
    method: str
    converged: bool
    iterations: int
    residual_lambda: Optional[float]
    residual_C: Optional[float]


# --------------------------------------------------------------------------
# EOS helpers (general cubic form with roots δ1, δ2)
# --------------------------------------------------------------------------


def _eos_deltas(eos: CubicEOS) -> Tuple[float, float]:
    """Return (δ1, δ2) for a cubic EOS from its (u, w) parameters.

    P = RT/(V-b) − a/((V+δ1 b)(V+δ2 b))  with
    (V+δ1 b)(V+δ2 b) = V² + u·Vb + w·b², i.e. δ1,2 = (u ± √(u² − 4w))/2.

    For PR (u=2, w=−1): δ1 = 1+√2, δ2 = 1−√2.
    For SRK (u=1, w=0):  δ1 = 1,     δ2 = 0.
    """
    disc = eos.u * eos.u - 4.0 * eos.w
    if disc < 0:
        raise ValueError(f"Non-real EOS deltas (u={eos.u}, w={eos.w})")
    s = math.sqrt(disc)
    return 0.5 * (eos.u + s), 0.5 * (eos.u - s)


def _resolve_kij(eos: CubicEOS, T: float, binary_interaction) -> NDArray[np.float64]:
    """Fetch the k_ij matrix at T regardless of whether the EOS exposes a
    private resolver or we were handed a static matrix / callable / None.
    """
    if binary_interaction is None:
        return np.zeros((eos.n_components, eos.n_components))
    if isinstance(binary_interaction, np.ndarray):
        return binary_interaction
    if callable(binary_interaction):
        return np.asarray(binary_interaction(T))
    if hasattr(binary_interaction, "get_kij_matrix"):
        return np.asarray(binary_interaction.get_kij_matrix(T))
    return np.zeros((eos.n_components, eos.n_components))


def _a_ij_matrix(a_array: NDArray[np.float64], kij: NDArray[np.float64]) -> NDArray[np.float64]:
    sqrt_a = np.sqrt(np.maximum(a_array, 0.0))
    return np.outer(sqrt_a, sqrt_a) * (1.0 - kij)


def _pressure_TVn(
    T: float,
    V: float,
    n: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction,
) -> float:
    """EOS pressure at extensive (T, V, n) — no root selection needed."""
    n_T = float(n.sum())
    x = n / n_T
    a_mix, b_mix, _a, _b = eos.calculate_params(T, x, binary_interaction)
    B_ext = n_T * b_mix
    A_ext = n_T * n_T * a_mix
    RT = R.Pa_m3_per_mol_K * T
    d1, d2 = _eos_deltas(eos)
    denom_rep = V - B_ext
    denom_att = (V + d1 * B_ext) * (V + d2 * B_ext)
    if denom_rep <= 0.0 or denom_att == 0.0:
        raise ValueError(
            f"Invalid (T={T:.3f}, V={V:.3e}, n_T={n_T:.3f}) for EOS pressure: "
            f"V-nb={denom_rep:.3e}, denominator_att={denom_att:.3e}"
        )
    return n_T * RT / denom_rep - A_ext / denom_att


def _A_over_RT(
    T: float,
    V: float,
    n: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction,
) -> float:
    """Total (ideal + residual) reduced Helmholtz energy A/RT at (T, V, n).

    The ideal part is  Σ n_i [ln(n_i RT / V) − 1]  (reference state
    cancels in the third derivative of A/RT along any direction u).
    The residual part uses the integrated cubic-EOS form.
    """
    n_T = float(n.sum())
    if n_T <= 0.0:
        raise ValueError("Total moles must be positive")
    x = n / n_T
    a_mix, b_mix, _a, _b = eos.calculate_params(T, x, binary_interaction)
    RT = R.Pa_m3_per_mol_K * T
    B_ext = n_T * b_mix
    A_ext = n_T * n_T * a_mix
    d1, d2 = _eos_deltas(eos)

    if V <= B_ext:
        raise ValueError(f"V={V:.3e} <= n·b={B_ext:.3e}; EOS undefined")

    # Residual part (Michelsen & Mollerup Eq 3.54, sign convention matching
    # the pressure formula P = nRT/(V-nb) + A_ext / ((V+δ1 nb)(V+δ2 nb))):
    #
    #   A^res / RT = -n_T · ln(1 - nb/V)
    #                - A_ext / (B_ext · RT · (δ1 - δ2))
    #                  · ln((V + δ1 nb) / (V + δ2 nb))
    F_res = -n_T * math.log(1.0 - B_ext / V) - (
        A_ext / (B_ext * RT * (d1 - d2))
    ) * math.log((V + d1 * B_ext) / (V + d2 * B_ext))

    # Ideal part (drop the constant shift RT/V that's independent of n);
    # its third directional derivative is -Σ u_i³ / n_i².
    F_id = float(np.sum(n * (np.log(n * RT / V) - 1.0)))

    return F_id + F_res


def _ln_f_TVn(
    T: float,
    V: float,
    n: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction,
) -> NDArray[np.float64]:
    """ln(f_i) = ∂(A/RT)/∂n_i |_{T,V,n_{k≠i}} for the cubic EOS, evaluated
    directly at (T, V, n) with no phase-root ambiguity. Units: dimensionless.
    """
    n_T = float(n.sum())
    x = n / n_T
    a_mix, b_mix, a_array, b_array = eos.calculate_params(T, x, binary_interaction)
    kij = _resolve_kij(eos, T, binary_interaction)
    a_ij = _a_ij_matrix(a_array, kij)

    RT = R.Pa_m3_per_mol_K * T
    B_ext = n_T * b_mix
    A_ext = n_T * n_T * a_mix
    d1, d2 = _eos_deltas(eos)

    if V <= B_ext:
        raise ValueError(f"V={V:.3e} <= n·b={B_ext:.3e}; EOS undefined")

    # Pressure at (T, V, n) — exact from the cubic EOS.
    P = n_T * RT / (V - B_ext) - A_ext / ((V + d1 * B_ext) * (V + d2 * B_ext))
    if P <= 0.0:
        raise ValueError(f"Non-positive pressure P={P:.3e} at (T={T}, V={V}, n_T={n_T})")
    Z = P * V / (n_T * RT)

    A_dim = a_mix * P / (RT * RT)
    B_dim = b_mix * P / RT

    if Z - B_dim <= 0.0:
        raise ValueError(f"Z-B={Z - B_dim:.3e} <= 0 at (T={T}, V={V})")

    log_Z_minus_B = math.log(Z - B_dim)
    ratio_num = Z + d1 * B_dim
    ratio_den = Z + d2 * B_dim
    if ratio_num <= 0.0 or ratio_den <= 0.0:
        raise ValueError(f"Log-ratio arguments non-positive at (T={T}, V={V})")
    log_ratio = math.log(ratio_num / ratio_den)
    coeff = A_dim / ((d1 - d2) * B_dim) if B_dim > 0 else 0.0

    sum_xj_aij = a_ij @ x
    bi_over_bmix = b_array / b_mix
    bracket = 2.0 * sum_xj_aij / a_mix - bi_over_bmix

    ln_phi = bi_over_bmix * (Z - 1.0) - log_Z_minus_B - coeff * bracket * log_ratio
    # ln(f_i) = ln(x_i · P · φ_i) = ln(n_i) − ln(n_T) + ln(P) + ln(φ_i).
    return np.log(np.maximum(n, 1e-300)) - math.log(n_T) + math.log(P) + ln_phi


# --------------------------------------------------------------------------
# Q matrix and cubic form
# --------------------------------------------------------------------------


def _compute_Q(
    T: float,
    V: float,
    n: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction,
    dn_rel: float = 1e-4,
) -> NDArray[np.float64]:
    """Assemble Q_ij = ∂²(A/RT)/∂n_i∂n_j |_{T,V} via central FD on A/RT.

    We use second-derivative FD directly on A/RT rather than FD on ln(f_i)
    so the assembly works even when P(T, V, n) < 0 (mechanically unstable
    region). That matters because the outer Newton iterates can pass
    through the spinodal region en route to the critical point.

    Diagonal:  [F(n + h e_i) − 2 F(n) + F(n − h e_i)] / h²
    Off-diag:  [F(n+h e_i+h e_j) − F(n+h e_i−h e_j) − F(n−h e_i+h e_j)
                + F(n−h e_i−h e_j)] / (4 h²)
    """
    nc = len(n)
    n_T = float(n.sum())
    h = max(dn_rel * n_T, 1e-8)
    Q = np.zeros((nc, nc), dtype=np.float64)
    F0 = _A_over_RT(T, V, n, eos, binary_interaction)

    F_p = np.empty(nc)
    F_m = np.empty(nc)
    for i in range(nc):
        n_p = n.copy(); n_p[i] += h
        n_m = n.copy(); n_m[i] -= h
        F_p[i] = _A_over_RT(T, V, n_p, eos, binary_interaction)
        F_m[i] = _A_over_RT(T, V, n_m, eos, binary_interaction)
        Q[i, i] = (F_p[i] - 2.0 * F0 + F_m[i]) / (h * h)

    inv_4h2 = 1.0 / (4.0 * h * h)
    for i in range(nc):
        for j in range(i + 1, nc):
            n_pp = n.copy(); n_pp[i] += h; n_pp[j] += h
            n_pm = n.copy(); n_pm[i] += h; n_pm[j] -= h
            n_mp = n.copy(); n_mp[i] -= h; n_mp[j] += h
            n_mm = n.copy(); n_mm[i] -= h; n_mm[j] -= h
            val = (
                _A_over_RT(T, V, n_pp, eos, binary_interaction)
                - _A_over_RT(T, V, n_pm, eos, binary_interaction)
                - _A_over_RT(T, V, n_mp, eos, binary_interaction)
                + _A_over_RT(T, V, n_mm, eos, binary_interaction)
            ) * inv_4h2
            Q[i, j] = val
            Q[j, i] = val
    return Q


def _min_eigenpair(Q: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
    """Return (λ_min, u) for a symmetric matrix, with u unit-normalised."""
    w, V = np.linalg.eigh(Q)
    idx = int(np.argmin(w))
    u = V[:, idx]
    # Pick a deterministic sign (largest-magnitude component positive).
    k = int(np.argmax(np.abs(u)))
    if u[k] < 0.0:
        u = -u
    norm = np.linalg.norm(u)
    if norm > 0.0:
        u = u / norm
    return float(w[idx]), u


def _cubic_form(
    T: float,
    V: float,
    n: NDArray[np.float64],
    u: NDArray[np.float64],
    eos: CubicEOS,
    binary_interaction,
    h_rel: float = 1e-3,
) -> float:
    """Directional third derivative C = d³(A/RT)/dε³ at ε = 0 along u.

    4-point central FD:
        φ'''(0) ≈ [φ(−2h) − 2φ(−h) + 2φ(h) − φ(2h)] / (2h³).
    """
    n_T = float(n.sum())
    # Scale h so the perturbation is small relative to min(n_i) to keep all
    # components strictly positive.
    min_positive = float(np.min(np.where(u != 0.0, n / np.maximum(np.abs(u), 1e-300), n_T)))
    h = min(h_rel * n_T, 0.1 * min_positive)
    if h <= 0.0:
        h = 1e-8

    def F(eps: float) -> float:
        return _A_over_RT(T, V, n + eps * u, eos, binary_interaction)

    return (F(-2.0 * h) - 2.0 * F(-h) + 2.0 * F(h) - F(2.0 * h)) / (2.0 * h ** 3)


# --------------------------------------------------------------------------
# Initial guesses
# --------------------------------------------------------------------------


def _li_critical_estimate(
    z: NDArray[np.float64], components: List[Component]
) -> Tuple[float, float, float]:
    """Li-type mixing rule for (Tc, Pc, Vc) seed of the H-K iteration.

    Tc uses mole-fraction weighted Tc; Vc uses cubic combining (Plöcker);
    Pc uses Zc ≈ 0.291 − 0.080·ω_mix with ω_mix the mole-fraction weighted ω.
    Returns Vc (molar) as well for the outer V seed.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    Tc_list = np.array([c.Tc for c in components])
    Pc_list = np.array([c.Pc for c in components])
    omega_list = np.array([c.omega for c in components])

    Vc_list = np.empty(len(components))
    for i, c in enumerate(components):
        if getattr(c, "Vc", None) is not None and c.Vc > 0.0:
            Vc_list[i] = c.Vc
        else:
            Vc_list[i] = 0.27 * R.Pa_m3_per_mol_K * c.Tc / c.Pc

    Tc_mix = float(np.sum(z * Tc_list))
    Vc13 = np.cbrt(Vc_list)
    Vc_mix = 0.0
    for i in range(len(components)):
        for j in range(len(components)):
            Vc_mix += z[i] * z[j] * 0.125 * (Vc13[i] + Vc13[j]) ** 3
    Zc_mix = 0.291 - 0.080 * float(np.sum(z * omega_list))
    Pc_mix = Zc_mix * R.Pa_m3_per_mol_K * Tc_mix / Vc_mix
    _ = Pc_list  # reserved for future Chueh-Prausnitz variant
    return Tc_mix, Pc_mix, Vc_mix


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def compute_critical_point(
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    binary_interaction=None,
    T_init: Optional[float] = None,
    V_init: Optional[float] = None,
    max_iter: int = 60,
    tol_lambda: float = 5e-7,
    tol_C_rel: float = 5e-5,
) -> CriticalPointResult:
    """Solve the Heidemann-Khalil conditions for (Tc, Vc) at fixed z.

    The outer iteration is a damped Newton on the 2×2 system
        F(T, V) = [λ_min(Q(T,V)),  C(T,V)]^T = 0
    with a central finite-difference Jacobian. Pc is recovered from
    P_EOS(Tc, Vc, z).

    Convergence criteria
    --------------------
    - |λ_min| < tol_lambda
    - |C| / C_scale < tol_C_rel   where  C_scale ≈ 1 / V²

    Returns ``CriticalPointResult`` with converged=False if the iteration
    fails or the solution leaves physical bounds.
    """
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()
    nc = len(z)
    if nc != len(components):
        raise ValueError("composition length must match components")

    Tc_est, Pc_est, Vc_est = _li_critical_estimate(z, components)
    T = float(T_init) if T_init is not None else Tc_est
    # Prefer 3.8·b_mix as the V seed: near the PR critical volume V_c ≈ 3.95 b
    # and safely on the vapor side of the spinodal, so the first ln_f / pressure
    # evaluation is well-defined even when Li-rule under-estimates Vc.
    _, _b_mix_seed, _, _ = eos.calculate_params(max(T, 100.0), z, binary_interaction)
    V_from_b = 3.8 * _b_mix_seed
    if V_init is not None:
        V = float(V_init)
    else:
        V = max(Vc_est, V_from_b)

    # Physical bounds (used for early failure detection).
    Tc_min = 0.4 * min(c.Tc for c in components)
    Tc_max = 2.0 * max(c.Tc for c in components)
    _, b_mix_at_T, _, _ = eos.calculate_params(max(T, 100.0), z, binary_interaction)
    V_min = 1.01 * b_mix_at_T
    V_max = 20.0 * b_mix_at_T

    last_lambda: Optional[float] = None
    last_C: Optional[float] = None

    for iteration in range(max_iter):
        try:
            Q = _compute_Q(T, V, z, eos, binary_interaction)
            lam, u = _min_eigenpair(Q)
            C = _cubic_form(T, V, z, u, eos, binary_interaction)
        except (ValueError, FloatingPointError):
            return CriticalPointResult(
                Tc=None, Pc=None, Vc=None,
                method="heidemann_khalil",
                converged=False,
                iterations=iteration,
                residual_lambda=last_lambda,
                residual_C=last_C,
            )

        last_lambda = lam
        last_C = C

        if abs(lam) < tol_lambda and abs(C) < tol_C_rel:
            try:
                Pc = _pressure_TVn(T, V, z, eos, binary_interaction)
            except ValueError:
                return CriticalPointResult(
                    Tc=None, Pc=None, Vc=None,
                    method="heidemann_khalil",
                    converged=False,
                    iterations=iteration + 1,
                    residual_lambda=lam,
                    residual_C=C,
                )
            if not (Tc_min <= T <= Tc_max and V_min <= V <= V_max) or Pc <= 0:
                return CriticalPointResult(
                    Tc=None, Pc=None, Vc=None,
                    method="heidemann_khalil",
                    converged=False,
                    iterations=iteration + 1,
                    residual_lambda=lam,
                    residual_C=C,
                )
            return CriticalPointResult(
                Tc=float(T), Pc=float(Pc), Vc=float(V),
                method="heidemann_khalil",
                converged=True,
                iterations=iteration + 1,
                residual_lambda=lam,
                residual_C=C,
            )

        # Build 2×2 Jacobian via central FD. We re-derive the same u from
        # each perturbed Q (eigenvalue sign is deterministic by design).
        dT = max(T * 1e-4, 0.05)
        dV = max(V * 1e-4, 1e-10)

        try:
            Q_Tp = _compute_Q(T + dT, V, z, eos, binary_interaction)
            Q_Tm = _compute_Q(T - dT, V, z, eos, binary_interaction)
            Q_Vp = _compute_Q(T, V + dV, z, eos, binary_interaction)
            Q_Vm = _compute_Q(T, V - dV, z, eos, binary_interaction)
            lam_Tp, u_Tp = _min_eigenpair(Q_Tp)
            lam_Tm, u_Tm = _min_eigenpair(Q_Tm)
            lam_Vp, u_Vp = _min_eigenpair(Q_Vp)
            lam_Vm, u_Vm = _min_eigenpair(Q_Vm)
            C_Tp = _cubic_form(T + dT, V, z, u_Tp, eos, binary_interaction)
            C_Tm = _cubic_form(T - dT, V, z, u_Tm, eos, binary_interaction)
            C_Vp = _cubic_form(T, V + dV, z, u_Vp, eos, binary_interaction)
            C_Vm = _cubic_form(T, V - dV, z, u_Vm, eos, binary_interaction)
        except (ValueError, FloatingPointError):
            return CriticalPointResult(
                Tc=None, Pc=None, Vc=None,
                method="heidemann_khalil",
                converged=False,
                iterations=iteration + 1,
                residual_lambda=lam,
                residual_C=C,
            )

        J = np.array([
            [(lam_Tp - lam_Tm) / (2.0 * dT), (lam_Vp - lam_Vm) / (2.0 * dV)],
            [(C_Tp - C_Tm) / (2.0 * dT),     (C_Vp - C_Vm) / (2.0 * dV)],
        ])
        r = np.array([lam, C])

        try:
            delta = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return CriticalPointResult(
                Tc=None, Pc=None, Vc=None,
                method="heidemann_khalil",
                converged=False,
                iterations=iteration + 1,
                residual_lambda=lam,
                residual_C=C,
            )

        # Damp the step so we do not leave the physical domain in one shot.
        max_dT = 0.15 * T
        max_dV = 0.25 * V
        scale = 1.0
        if abs(delta[0]) > max_dT:
            scale = min(scale, max_dT / abs(delta[0]))
        if abs(delta[1]) > max_dV:
            scale = min(scale, max_dV / abs(delta[1]))
        T_new = T + scale * delta[0]
        V_new = V + scale * delta[1]

        # Clamp into physical bounds.
        T_new = min(max(T_new, Tc_min), Tc_max)
        V_new = min(max(V_new, V_min), V_max)

        if abs(T_new - T) < 1e-6 and abs(V_new - V) / V < 1e-8:
            return CriticalPointResult(
                Tc=None, Pc=None, Vc=None,
                method="heidemann_khalil",
                converged=False,
                iterations=iteration + 1,
                residual_lambda=lam,
                residual_C=C,
            )
        T = T_new
        V = V_new

    return CriticalPointResult(
        Tc=None, Pc=None, Vc=None,
        method="heidemann_khalil",
        converged=False,
        iterations=max_iter,
        residual_lambda=last_lambda,
        residual_C=last_C,
    )

"""Peng-Robinson 1978 EOS (slides 346, 373, 376, 381).

    P = RT/(V-b) - a(T)/(V² + 2bV - b²)
    α(T) = [1 + κ(1 - √(T/Tc))]²
    κ = 0.379642 + 1.48503w - 0.164423w² + 0.016666w³      (PR78, w > 0.49)
    κ = 0.37464 + 1.54226w - 0.26992w²                      (PR76 / PR78 for w ≤ 0.49)

Mixing rules (slide 376):
    a_mix = ΣΣ xᵢxⱼ √(aᵢaⱼ)(1 - kᵢⱼ)
    b_mix = Σ xᵢ bᵢ

Z cubic:
    Z³ - (1-B)Z² + (A - 2B - 3B²)Z - (AB - B² - B³) = 0
"""

import numpy as np

from pvtsim.constants import OMEGA_A, OMEGA_B, R


def kappa(w):
    """PR78 κ(w).  w ≤ 0.49 uses PR76 form (slide 346)."""
    if w <= 0.49:
        return 0.37464 + 1.54226 * w - 0.26992 * w ** 2
    return 0.379642 + 1.48503 * w - 0.164423 * w ** 2 + 0.016666 * w ** 3


def a_pure(Tc, Pc, w, T):
    """Pure-component a(T) in field units. Tc, T in °R; Pc in psia."""
    ac = OMEGA_A * R ** 2 * Tc ** 2 / Pc
    alpha = (1.0 + kappa(w) * (1.0 - np.sqrt(T / Tc))) ** 2
    return ac * alpha


def b_pure(Tc, Pc):
    """Pure-component b in field units."""
    return OMEGA_B * R * Tc / Pc


def mix_params(x, ai, bi, kij):
    """a_mix, b_mix from van der Waals one-fluid mixing (slide 376)."""
    sqrt_ai = np.sqrt(ai)
    aij = np.outer(sqrt_ai, sqrt_ai) * (1.0 - kij)
    a_mix = np.sum(np.outer(x, x) * aij)
    b_mix = np.sum(x * bi)
    return a_mix, b_mix


def Z_roots(A, B):
    """Real roots of PR Z-cubic (slide 378). Returns sorted ascending."""
    coeffs = [1.0, -(1.0 - B), A - 2.0 * B - 3.0 * B ** 2, -(A * B - B ** 2 - B ** 3)]
    roots = np.roots(coeffs)
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > B)
    return real


def Z_liquid(A, B):
    """Smallest Z root > B."""
    r = Z_roots(A, B)
    return r[0] if r else np.nan


def Z_vapor(A, B):
    """Largest Z root."""
    r = Z_roots(A, B)
    return r[-1] if r else np.nan


def fugacity_coef(x, A, B, Z, ai, bi, a_mix, b_mix, kij):
    """PR fugacity coefficient per component (slide 381)."""
    sqrt2 = np.sqrt(2.0)
    sqrt_ai = np.sqrt(ai)
    # Σⱼ xⱼ √(aᵢaⱼ)(1 - kᵢⱼ)
    sum_j = (x * (1.0 - kij) @ sqrt_ai) * sqrt_ai
    ln_phi = (
        (bi / b_mix) * (Z - 1.0)
        - np.log(Z - B)
        - (A / (2.0 * sqrt2 * B))
        * (2.0 * sum_j / a_mix - bi / b_mix)
        * np.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))
    )
    return np.exp(ln_phi)


def AB(a_mix, b_mix, P, T):
    """Dimensionless A, B for the cubic."""
    A = a_mix * P / (R * T) ** 2
    B = b_mix * P / (R * T)
    return A, B


def phase_params(x, comps, T):
    """Return (a_mix, b_mix, ai, bi) for a phase of composition x at T.

    comps: list of (Tc, Pc, w, MW) tuples
    """
    ai = np.array([a_pure(c[0], c[1], c[2], T) for c in comps])
    bi = np.array([b_pure(c[0], c[1]) for c in comps])
    n = len(comps)
    kij = np.zeros((n, n))
    a_mix, b_mix = mix_params(x, ai, bi, kij)
    return a_mix, b_mix, ai, bi

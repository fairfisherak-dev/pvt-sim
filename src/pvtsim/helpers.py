"""Shared PVT calculation helpers.

Notation (from LECTURE_SLIDES_MERGED.md):
    P, T       pressure [psia], temperature [°R]
    Tc, Pc     critical temperature [°R], pressure [psia]
    w          acentric factor
    z          overall feed mole fraction
    x, y       liquid, vapor mole fraction (per component)
    nL, nV     moles of liquid, vapor (basis n = nL + nV = 1 per step)
    K          vapor/liquid equilibrium ratio
    Z          compressibility factor
    phi        fugacity coefficient
    f          fugacity [psia]
    VL, VV     liquid, vapor volume [ft³]
    VoSC       stock-tank oil volume [ft³]
    Vgas_scf   total liberated gas at SC [scf]
    Rs         solution GOR [scf/STB]
    RsDb       Rs at Pb [scf/STB]
    Bo         oil FVF [bbl/STB]
    Bg         gas FVF [bbl/scf]
    BtD        total FVF [bbl/STB]
    beta       vapor mole fraction (Rachford-Rice root)
"""

import numpy as np

from pvtsim.constants import (
    FT3_PER_BBL, R, SCF_PER_LBMOL, SC_P, SC_T_R,
)


def T_R(T_F):
    """°R from °F."""
    return T_F + 459.67


def K_wilson(Pc, Tc, w, P, T):
    """Wilson K-value for flash initialization (slide 345)."""
    return (Pc / P) * np.exp(5.373 * (1 + w) * (1 - Tc / T))


def rachford_rice(beta, z, K):
    """Rachford-Rice function F(beta) (slide 340)."""
    return np.sum(z * (K - 1) / (1 + beta * (K - 1)))


def solve_rachford_rice(z, K, tol=1e-12, maxit=100):
    """Bracketed bisection on beta ∈ (0, 1) for RR = 0."""
    lo, hi = 1e-10, 1.0 - 1e-10
    f_lo = rachford_rice(lo, z, K)
    f_hi = rachford_rice(hi, z, K)
    if f_lo * f_hi > 0:
        return 0.0 if f_lo < 0 else 1.0  # single-phase boundary
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f = rachford_rice(mid, z, K)
        if abs(f) < tol:
            return mid
        if f_lo * f < 0:
            hi, f_hi = mid, f
        else:
            lo, f_lo = mid, f
    return mid


def Bg(Z, T, P):
    """Bg [bbl/scf] = 0.005035·Z·T/P  (slide 428). T in °R, P in psia."""
    return 0.005035 * Z * T / P


def Bo(VL, VoSC):
    """Bo [bbl/STB] = VL / VoSC  (slide 428)."""
    return VL / VoSC


def Rs(Vgas_scf, VoSC):
    """Rs [scf/STB] = (Vgas_scf / VoSC) · 5.615  (slide 428)."""
    return (Vgas_scf / VoSC) * FT3_PER_BBL


def BtD(Bo_, Bg_, RsDb_, RsD_):
    """BtD = Bo + Bg·(RsDb - RsD)  (slide 285)."""
    return Bo_ + Bg_ * (RsDb_ - RsD_)


def Vgas_at_SC(n_lbmol):
    """V_gas [scf] = n · 379.6  (slide 283). Ideal gas at 14.696 psia, 60°F."""
    return n_lbmol * SCF_PER_LBMOL


def V_from_Z(n_lbmol, Z, T, P):
    """V [ft³] = n·Z·R·T/P  (real gas law, slide 378). T in °R, P in psia."""
    return n_lbmol * Z * R * T / P

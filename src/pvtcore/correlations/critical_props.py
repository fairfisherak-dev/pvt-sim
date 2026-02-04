"""
Critical property correlations for petroleum pseudo-components.

Implements correlations for estimating critical temperature (Tc), critical
pressure (Pc), and critical volume (Vc) from molecular weight (MW), specific
gravity (SG), and normal boiling point (Tb).

Methods implemented:
- Riazi-Daubert (1987): Recommended for petroleum fractions
- Kesler-Lee (1976): Classic correlation, good for heavier fractions
- Cavett (1962): API-style correlation using characterization factor

Units Convention (internal):
- Temperature: Kelvin (K)
- Pressure: Pascal (Pa)
- Volume: m³/mol
- Molecular weight: g/mol
- Specific gravity: dimensionless (60°F/60°F relative to water)

References
----------
[1] Riazi, M.R. and Daubert, T.E. (1987). "Characterization Parameters for
    Petroleum Fractions." Ind. Eng. Chem. Res., 26(4), 755-759.
[2] Kesler, M.G. and Lee, B.I. (1976). "Improve Prediction of Enthalpy of
    Fractions." Hydrocarbon Processing, 55(3), 153-158.
[3] Cavett, R.H. (1962). "Physical Data for Distillation Calculations,
    Vapor-Liquid Equilibria." Proc. 27th API Meeting, San Francisco.
[4] Pedersen, K.S., Christensen, P.L., and Shaikh, J.A. (2015). "Phase
    Behavior of Petroleum Reservoir Fluids." 2nd ed., CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class CriticalPropsMethod(Enum):
    """Enumeration of available critical property correlation methods."""
    RIAZI_DAUBERT = auto()
    KESLER_LEE = auto()
    CAVETT = auto()


@dataclass(frozen=True)
class CriticalPropsResult:
    """
    Result container for critical property calculations.

    Attributes
    ----------
    Tc : float
        Critical temperature in Kelvin.
    Pc : float
        Critical pressure in Pascal.
    Vc : float
        Critical molar volume in m³/mol.
    method : CriticalPropsMethod
        The correlation method used.
    """
    Tc: float  # K
    Pc: float  # Pa
    Vc: float  # m³/mol
    method: CriticalPropsMethod

    @property
    def Tc_R(self) -> float:
        """Critical temperature in Rankine."""
        return self.Tc * 1.8

    @property
    def Pc_psia(self) -> float:
        """Critical pressure in psia."""
        return self.Pc / 6894.757

    @property
    def Pc_bar(self) -> float:
        """Critical pressure in bar."""
        return self.Pc / 1e5

    @property
    def Vc_ft3_lbmol(self) -> float:
        """Critical volume in ft³/lbmol."""
        return self.Vc * 1000.0 * 0.0353147  # m³/mol -> L/mol -> ft³/lbmol


# =============================================================================
# Riazi-Daubert (1987) Correlations
# =============================================================================

def riazi_daubert_Tc(
    MW: float,
    SG: float,
    Tb: Optional[float] = None,
) -> float:
    """
    Estimate critical temperature using Riazi-Daubert (1987) correlation.

    Two forms are available:
    1. If Tb is provided: Tc = a * Tb^b * SG^c (more accurate)
    2. If Tb is None: Tc from MW and SG only (less accurate)

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    SG : float
        Specific gravity at 60°F/60°F (dimensionless).
    Tb : float, optional
        Normal boiling point in Kelvin. If provided, uses the more accurate
        Tb-based correlation.

    Returns
    -------
    float
        Critical temperature in Kelvin.

    References
    ----------
    Riazi & Daubert (1987), Eq. 3-5 for Tb-based, Eq. 3-6 for MW-based.
    """
    _validate_inputs(MW=MW, SG=SG, Tb=Tb)

    if Tb is not None:
        # Tb-based correlation (API Technical Data Book form)
        # Tc (K) = Tb / (0.533 + 0.191e-3 * Tb - 0.779e-6 * Tb^2)
        # Alternative Riazi-Daubert form that gives better results
        # Using theta = Tb_R * SG^(-0.0566) * exp(-0.00015 * MW)
        Tb_R = Tb * 1.8  # K to R

        # Riazi-Daubert (1980) simplified form
        # Tc (R) = 35.9413 * (exp(6.9195e-4 * Tb_R)) * Tb_R^0.7293 * SG^0.6667
        # Adjusted coefficients for petroleum fractions
        Tc_R = 24.2787 * (Tb_R ** 0.58848) * (SG ** 0.3596)

        return Tc_R / 1.8  # R to K
    else:
        # MW-SG based correlation (less accurate, no Tb)
        # Riazi-Daubert form: Tc = a * exp(...) * MW^b * SG^c
        # Using simplified approximation
        Tc_K = 231.27 * (MW ** 0.351) * (SG ** 0.582)
        return Tc_K


def riazi_daubert_Pc(
    MW: float,
    SG: float,
    Tb: Optional[float] = None,
) -> float:
    """
    Estimate critical pressure using Riazi-Daubert (1987) correlation.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float, optional
        Normal boiling point in Kelvin.

    Returns
    -------
    float
        Critical pressure in Pascal.

    References
    ----------
    Riazi & Daubert (1987), Eq. 3-5 for Tb-based, Eq. 3-6 for MW-based.
    """
    _validate_inputs(MW=MW, SG=SG, Tb=Tb)

    if Tb is not None:
        # Tb-based correlation
        # Pc = 3.12281e9 * Tb^(-2.3177) * SG^2.4853
        Tb_R = Tb * 1.8
        Pc_psia = 3.12281e9 * (Tb_R ** (-2.3177)) * (SG ** 2.4853)
    else:
        # MW-SG based correlation
        # Pc = 4.5203e4 * MW^(-0.8063) * SG^1.6015
        Pc_psia = 4.5203e4 * (MW ** (-0.8063)) * (SG ** 1.6015)

    return Pc_psia * 6894.757  # psia to Pa


def riazi_daubert_Vc(
    MW: float,
    SG: float,
    Tb: Optional[float] = None,
) -> float:
    """
    Estimate critical molar volume using Riazi-Daubert (1987) correlation.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float, optional
        Normal boiling point in Kelvin.

    Returns
    -------
    float
        Critical molar volume in m³/mol.

    References
    ----------
    Riazi & Daubert (1987).
    """
    _validate_inputs(MW=MW, SG=SG, Tb=Tb)

    if Tb is not None:
        # Tb-based correlation
        # Vc = 7.0434e-7 * Tb^2.3829 * SG^(-1.683)
        Tb_R = Tb * 1.8
        Vc_ft3_lbmol = 7.0434e-7 * (Tb_R ** 2.3829) * (SG ** (-1.683))
    else:
        # MW-SG based correlation
        # Vc = 1.206e-2 * MW^0.935 * SG^(-1.467)
        Vc_ft3_lbmol = 1.206e-2 * (MW ** 0.935) * (SG ** (-1.467))

    # Convert ft³/lbmol to m³/mol
    # 1 ft³ = 0.0283168 m³, 1 lbmol = 453.59 mol (but mol is same)
    # Actually lbmol is just mol conceptually, so ft³/lbmol -> m³/mol
    return Vc_ft3_lbmol * 0.0283168


def riazi_daubert_critical_props(
    MW: float,
    SG: float,
    Tb: Optional[float] = None,
) -> CriticalPropsResult:
    """
    Estimate all critical properties using Riazi-Daubert correlations.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float, optional
        Normal boiling point in Kelvin.

    Returns
    -------
    CriticalPropsResult
        Critical temperature (K), pressure (Pa), and volume (m³/mol).
    """
    return CriticalPropsResult(
        Tc=riazi_daubert_Tc(MW, SG, Tb),
        Pc=riazi_daubert_Pc(MW, SG, Tb),
        Vc=riazi_daubert_Vc(MW, SG, Tb),
        method=CriticalPropsMethod.RIAZI_DAUBERT,
    )


# =============================================================================
# Kesler-Lee (1976) Correlations
# =============================================================================

def kesler_lee_Tc(
    MW: float,
    SG: float,
    Tb: float,
) -> float:
    """
    Estimate critical temperature using Kesler-Lee (1976) correlation.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol (not used directly, for API compatibility).
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float
        Normal boiling point in Kelvin.

    Returns
    -------
    float
        Critical temperature in Kelvin.

    Notes
    -----
    The Kesler-Lee correlation requires Tb. MW is accepted for API consistency
    but the correlation itself uses Tb and SG.

    References
    ----------
    Kesler & Lee (1976), Hydrocarbon Processing.
    """
    _validate_inputs(MW=MW, SG=SG, Tb=Tb)
    if Tb is None:
        raise ValueError("Kesler-Lee correlation requires Tb (boiling point)")

    Tb_R = Tb * 1.8  # K to R

    # Kesler-Lee Tc correlation
    Tc_R = (
        341.7 + 811.0 * SG
        + (0.4244 + 0.1174 * SG) * Tb_R
        + (0.4669 - 3.2623 * SG) * 1e5 / Tb_R
    )

    return Tc_R / 1.8  # R to K


def kesler_lee_Pc(
    MW: float,
    SG: float,
    Tb: float,
) -> float:
    """
    Estimate critical pressure using Kesler-Lee (1976) correlation.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol (not used directly).
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float
        Normal boiling point in Kelvin.

    Returns
    -------
    float
        Critical pressure in Pascal.

    References
    ----------
    Kesler & Lee (1976), Hydrocarbon Processing.
    """
    _validate_inputs(MW=MW, SG=SG, Tb=Tb)
    if Tb is None:
        raise ValueError("Kesler-Lee correlation requires Tb (boiling point)")

    Tb_R = Tb * 1.8  # K to R

    # Kesler-Lee Pc correlation (log10 form)
    ln_Pc = (
        8.3634
        - 0.0566 / SG
        - (0.24244 + 2.2898 / SG + 0.11857 / (SG ** 2)) * 1e-3 * Tb_R
        + (1.4685 + 3.648 / SG + 0.47227 / (SG ** 2)) * 1e-7 * (Tb_R ** 2)
        - (0.42019 + 1.6977 / (SG ** 2)) * 1e-10 * (Tb_R ** 3)
    )

    Pc_psia = np.exp(ln_Pc)
    return Pc_psia * 6894.757  # psia to Pa


def kesler_lee_critical_props(
    MW: float,
    SG: float,
    Tb: float,
) -> CriticalPropsResult:
    """
    Estimate critical properties using Kesler-Lee correlations.

    Note: Kesler-Lee does not provide a Vc correlation, so Riazi-Daubert
    is used for Vc as a fallback.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float
        Normal boiling point in Kelvin.

    Returns
    -------
    CriticalPropsResult
        Critical properties with Kesler-Lee Tc/Pc and Riazi-Daubert Vc.
    """
    return CriticalPropsResult(
        Tc=kesler_lee_Tc(MW, SG, Tb),
        Pc=kesler_lee_Pc(MW, SG, Tb),
        Vc=riazi_daubert_Vc(MW, SG, Tb),  # Kesler-Lee doesn't have Vc
        method=CriticalPropsMethod.KESLER_LEE,
    )


# =============================================================================
# Cavett (1962) Correlations
# =============================================================================

def cavett_Tc(
    MW: float,
    SG: float,
    Tb: float,
) -> float:
    """
    Estimate critical temperature using Cavett (1962) correlation.

    Uses the Watson characterization factor K = Tb^(1/3) / SG.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol (not used directly).
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float
        Normal boiling point in Kelvin.

    Returns
    -------
    float
        Critical temperature in Kelvin.

    References
    ----------
    Cavett (1962), API 27th Meeting.
    """
    _validate_inputs(MW=MW, SG=SG, Tb=Tb)
    if Tb is None:
        raise ValueError("Cavett correlation requires Tb (boiling point)")

    Tb_F = (Tb - 273.15) * 1.8 + 32.0  # K to °F
    API = 141.5 / SG - 131.5

    # Cavett Tc correlation (Tc in °F)
    Tc_F = (
        768.07121
        + 1.7133693 * Tb_F
        - 0.0010834003 * (Tb_F ** 2)
        - 0.0089212579 * API * Tb_F
        + 0.38890584e-6 * (Tb_F ** 3)
        + 0.5309492e-5 * API * (Tb_F ** 2)
        + 0.327116e-7 * (API ** 2) * (Tb_F ** 2)
    )

    # Convert °F to K
    return (Tc_F - 32.0) / 1.8 + 273.15


def cavett_Pc(
    MW: float,
    SG: float,
    Tb: float,
) -> float:
    """
    Estimate critical pressure using Cavett (1962) correlation.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol (not used directly).
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float
        Normal boiling point in Kelvin.

    Returns
    -------
    float
        Critical pressure in Pascal.

    References
    ----------
    Cavett (1962), API 27th Meeting.
    """
    _validate_inputs(MW=MW, SG=SG, Tb=Tb)
    if Tb is None:
        raise ValueError("Cavett correlation requires Tb (boiling point)")

    Tb_F = (Tb - 273.15) * 1.8 + 32.0  # K to °F
    API = 141.5 / SG - 131.5

    # Cavett Pc correlation (log10 Pc in psia)
    log10_Pc = (
        2.8290406
        + 0.94120109e-3 * Tb_F
        - 0.30474749e-5 * (Tb_F ** 2)
        - 0.2087611e-4 * API * Tb_F
        + 0.15184103e-8 * (Tb_F ** 3)
        + 0.11047899e-7 * API * (Tb_F ** 2)
        - 0.48271599e-7 * (API ** 2) * Tb_F
        + 0.13949619e-9 * (API ** 2) * (Tb_F ** 2)
    )

    Pc_psia = 10.0 ** log10_Pc
    return Pc_psia * 6894.757  # psia to Pa


def cavett_critical_props(
    MW: float,
    SG: float,
    Tb: float,
) -> CriticalPropsResult:
    """
    Estimate critical properties using Cavett correlations.

    Note: Cavett does not provide a Vc correlation, so Riazi-Daubert
    is used for Vc as a fallback.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float
        Normal boiling point in Kelvin.

    Returns
    -------
    CriticalPropsResult
        Critical properties with Cavett Tc/Pc and Riazi-Daubert Vc.
    """
    return CriticalPropsResult(
        Tc=cavett_Tc(MW, SG, Tb),
        Pc=cavett_Pc(MW, SG, Tb),
        Vc=riazi_daubert_Vc(MW, SG, Tb),  # Cavett doesn't have Vc
        method=CriticalPropsMethod.CAVETT,
    )


# =============================================================================
# Unified Interface
# =============================================================================

def estimate_critical_props(
    MW: float,
    SG: float,
    Tb: Optional[float] = None,
    method: CriticalPropsMethod = CriticalPropsMethod.RIAZI_DAUBERT,
) -> CriticalPropsResult:
    """
    Estimate critical properties using the specified correlation method.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    SG : float
        Specific gravity at 60°F/60°F.
    Tb : float, optional
        Normal boiling point in Kelvin. Required for Kesler-Lee and Cavett.
    method : CriticalPropsMethod
        Correlation method to use.

    Returns
    -------
    CriticalPropsResult
        Critical temperature (K), pressure (Pa), and volume (m³/mol).

    Raises
    ------
    ValueError
        If Tb is required but not provided, or if method is invalid.

    Examples
    --------
    >>> props = estimate_critical_props(MW=142.0, SG=0.78, Tb=450.0)
    >>> print(f"Tc = {props.Tc:.1f} K, Pc = {props.Pc/1e6:.2f} MPa")
    """
    if method == CriticalPropsMethod.RIAZI_DAUBERT:
        return riazi_daubert_critical_props(MW, SG, Tb)
    elif method == CriticalPropsMethod.KESLER_LEE:
        if Tb is None:
            raise ValueError("Kesler-Lee correlation requires Tb (boiling point)")
        return kesler_lee_critical_props(MW, SG, Tb)
    elif method == CriticalPropsMethod.CAVETT:
        if Tb is None:
            raise ValueError("Cavett correlation requires Tb (boiling point)")
        return cavett_critical_props(MW, SG, Tb)
    else:
        raise ValueError(f"Unknown critical property method: {method}")


# =============================================================================
# Vectorized Versions for Arrays
# =============================================================================

def riazi_daubert_Tc_array(
    MW: NDArray[np.float64],
    SG: NDArray[np.float64],
    Tb: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Vectorized Riazi-Daubert Tc correlation.

    Parameters
    ----------
    MW : ndarray
        Molecular weights in g/mol.
    SG : ndarray
        Specific gravities at 60°F/60°F.
    Tb : ndarray, optional
        Normal boiling points in Kelvin.

    Returns
    -------
    ndarray
        Critical temperatures in Kelvin.
    """
    MW = np.asarray(MW, dtype=np.float64)
    SG = np.asarray(SG, dtype=np.float64)

    if Tb is not None:
        Tb = np.asarray(Tb, dtype=np.float64)
        Tb_R = Tb * 1.8
        Tc_R = 10.6443 * (Tb_R ** 0.81067) * (SG ** 0.53691)
    else:
        Tc_R = 544.4 * (MW ** 0.2998) * (SG ** 1.0555)

    return Tc_R / 1.8


def riazi_daubert_Pc_array(
    MW: NDArray[np.float64],
    SG: NDArray[np.float64],
    Tb: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Vectorized Riazi-Daubert Pc correlation.

    Returns critical pressure in Pascal.
    """
    MW = np.asarray(MW, dtype=np.float64)
    SG = np.asarray(SG, dtype=np.float64)

    if Tb is not None:
        Tb = np.asarray(Tb, dtype=np.float64)
        Tb_R = Tb * 1.8
        Pc_psia = 3.12281e9 * (Tb_R ** (-2.3177)) * (SG ** 2.4853)
    else:
        Pc_psia = 4.5203e4 * (MW ** (-0.8063)) * (SG ** 1.6015)

    return Pc_psia * 6894.757


def riazi_daubert_Vc_array(
    MW: NDArray[np.float64],
    SG: NDArray[np.float64],
    Tb: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Vectorized Riazi-Daubert Vc correlation.

    Returns critical molar volume in m³/mol.
    """
    MW = np.asarray(MW, dtype=np.float64)
    SG = np.asarray(SG, dtype=np.float64)

    if Tb is not None:
        Tb = np.asarray(Tb, dtype=np.float64)
        Tb_R = Tb * 1.8
        Vc_ft3_lbmol = 7.0434e-7 * (Tb_R ** 2.3829) * (SG ** (-1.683))
    else:
        Vc_ft3_lbmol = 1.206e-2 * (MW ** 0.935) * (SG ** (-1.467))

    return Vc_ft3_lbmol * 0.0283168


# =============================================================================
# Input Validation
# =============================================================================

def _validate_inputs(
    MW: Optional[float] = None,
    SG: Optional[float] = None,
    Tb: Optional[float] = None,
) -> None:
    """Validate correlation inputs."""
    if MW is not None:
        if not np.isfinite(MW):
            raise ValueError(f"MW must be finite, got {MW}")
        if MW <= 0:
            raise ValueError(f"MW must be positive, got {MW}")
        if MW < 16 or MW > 2000:
            # Warning range - correlations may be less accurate
            pass

    if SG is not None:
        if not np.isfinite(SG):
            raise ValueError(f"SG must be finite, got {SG}")
        if SG <= 0:
            raise ValueError(f"SG must be positive, got {SG}")
        if SG < 0.5 or SG > 1.1:
            # Warning range - outside typical petroleum range
            pass

    if Tb is not None:
        if not np.isfinite(Tb):
            raise ValueError(f"Tb must be finite, got {Tb}")
        if Tb <= 0:
            raise ValueError(f"Tb must be positive, got {Tb}")
        if Tb < 100 or Tb > 1000:
            # Warning range - outside typical petroleum range
            pass

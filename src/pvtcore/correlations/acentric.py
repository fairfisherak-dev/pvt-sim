"""
Acentric factor correlations for petroleum pseudo-components.

The acentric factor (omega) is a key parameter in cubic equations of state,
characterizing the deviation from simple fluid behavior. It was defined by
Pitzer as:
    omega = -log10(Psat/Pc) - 1    at Tr = 0.7

Methods implemented:
- Edmister (1958): Based on Tb/Tc and Pc
- Kesler-Lee (1976): More complex correlation based on Tb and SG

Units Convention:
- Temperature: Kelvin (K)
- Pressure: Pascal (Pa)
- Acentric factor: dimensionless

References
----------
[1] Edmister, W.C. (1958). "Applied Hydrocarbon Thermodynamics, Part 4:
    Compressibility Factors and Equations of State." Petroleum Refiner, 37(4), 173-179.
[2] Kesler, M.G. and Lee, B.I. (1976). "Improve Prediction of Enthalpy of
    Fractions." Hydrocarbon Processing, 55(3), 153-158.
[3] Pitzer, K.S. (1955). "The Volumetric and Thermodynamic Properties of Fluids."
    J. Am. Chem. Soc., 77(13), 3427-3433.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class AcentricMethod(Enum):
    """Enumeration of available acentric factor correlation methods."""
    EDMISTER = auto()
    KESLER_LEE = auto()


def edmister_omega(
    Tb: float,
    Tc: float,
    Pc: float,
) -> float:
    """
    Estimate acentric factor using Edmister (1958) correlation.

    The Edmister correlation is based on the Clausius-Clapeyron equation
    and uses the reduced normal boiling point:

        omega = (3/7) * [log10(Pc/Pa)] / [Tc/Tb - 1] - 1

    where Pa is atmospheric pressure (101325 Pa).

    Parameters
    ----------
    Tb : float
        Normal boiling point in Kelvin.
    Tc : float
        Critical temperature in Kelvin.
    Pc : float
        Critical pressure in Pascal.

    Returns
    -------
    float
        Acentric factor (dimensionless).

    Notes
    -----
    This correlation works well for hydrocarbons and is simple to apply.
    Accuracy decreases for very heavy or polar compounds.

    References
    ----------
    Edmister (1958), Petroleum Refiner.

    Examples
    --------
    >>> omega = edmister_omega(Tb=371.6, Tc=540.2, Pc=2.74e6)  # n-Heptane
    >>> print(f"omega = {omega:.3f}")
    omega = 0.350
    """
    _validate_inputs(Tb=Tb, Tc=Tc, Pc=Pc)

    if Tc <= Tb:
        raise ValueError(f"Tc must be greater than Tb: Tc={Tc}, Tb={Tb}")

    Pa = 101325.0  # Atmospheric pressure in Pa

    # Edmister correlation
    # omega = (3/7) * log10(Pc/Pa) / (Tc/Tb - 1) - 1
    Tbr = Tb / Tc  # Reduced boiling point

    omega = (3.0 / 7.0) * np.log10(Pc / Pa) / (1.0 / Tbr - 1.0) - 1.0

    # Clamp to physically reasonable range
    omega = max(omega, -0.5)
    omega = min(omega, 2.0)

    return omega


def kesler_lee_omega(
    Tb: float,
    Tc: float,
    Pc: float,
) -> float:
    """
    Estimate acentric factor using Kesler-Lee (1976) correlation.

    This is a more complex correlation that provides better accuracy
    for heavier petroleum fractions.

    Parameters
    ----------
    Tb : float
        Normal boiling point in Kelvin.
    Tc : float
        Critical temperature in Kelvin.
    Pc : float
        Critical pressure in Pascal.

    Returns
    -------
    float
        Acentric factor (dimensionless).

    Notes
    -----
    The Kesler-Lee correlation uses different expressions depending on
    the reduced boiling point Tbr = Tb/Tc:
    - For Tbr <= 0.8: Simple expression
    - For Tbr > 0.8: Extended expression for heavier fractions

    References
    ----------
    Kesler & Lee (1976), Hydrocarbon Processing.
    """
    _validate_inputs(Tb=Tb, Tc=Tc, Pc=Pc)

    if Tc <= Tb:
        raise ValueError(f"Tc must be greater than Tb: Tc={Tc}, Tb={Tb}")

    Tbr = Tb / Tc  # Reduced boiling point
    Pc_atm = Pc / 101325.0  # Pc in atm for this correlation

    if Tbr <= 0.8:
        # Simple correlation for lighter fractions
        # omega = -ln(Pc/Pa) - 5.92714 + 6.09648/Tbr + 1.28862*ln(Tbr) - 0.169347*Tbr^6
        #         ------------------------------------------------------------------
        #                    15.2518 - 15.6875/Tbr - 13.4721*ln(Tbr) + 0.43577*Tbr^6
        ln_Pc_Pa = np.log(Pc / 101325.0)

        num = (
            -ln_Pc_Pa
            - 5.92714
            + 6.09648 / Tbr
            + 1.28862 * np.log(Tbr)
            - 0.169347 * (Tbr ** 6)
        )
        den = (
            15.2518
            - 15.6875 / Tbr
            - 13.4721 * np.log(Tbr)
            + 0.43577 * (Tbr ** 6)
        )

        omega = num / den
    else:
        # Extended correlation for heavier fractions (Tbr > 0.8)
        # Uses the Pitzer definition-based extrapolation
        theta = Tb / Tc

        omega = (
            -7.904
            + 0.1352 * (Tb / Tc) * 1e1
            - 0.007465 * (Tb / Tc) ** 2 * 1e2
            + 8.359 * theta
            + 1.408 / theta
        )

        # Alternative form from Whitson & Brule
        # For heavy fractions, use Edmister as fallback if result is unreasonable
        if omega < -0.5 or omega > 2.0:
            omega = edmister_omega(Tb, Tc, Pc)

    # Clamp to physically reasonable range
    omega = max(omega, -0.5)
    omega = min(omega, 2.0)

    return omega


def estimate_omega(
    Tb: float,
    Tc: float,
    Pc: float,
    method: AcentricMethod = AcentricMethod.EDMISTER,
) -> float:
    """
    Estimate acentric factor using the specified correlation method.

    Parameters
    ----------
    Tb : float
        Normal boiling point in Kelvin.
    Tc : float
        Critical temperature in Kelvin.
    Pc : float
        Critical pressure in Pascal.
    method : AcentricMethod
        Correlation method to use.

    Returns
    -------
    float
        Acentric factor (dimensionless).

    Raises
    ------
    ValueError
        If inputs are invalid or method is unknown.

    Examples
    --------
    >>> omega = estimate_omega(Tb=371.6, Tc=540.2, Pc=2.74e6)
    >>> print(f"omega = {omega:.3f}")
    """
    if method == AcentricMethod.EDMISTER:
        return edmister_omega(Tb, Tc, Pc)
    elif method == AcentricMethod.KESLER_LEE:
        return kesler_lee_omega(Tb, Tc, Pc)
    else:
        raise ValueError(f"Unknown acentric factor method: {method}")


# =============================================================================
# Vectorized Versions
# =============================================================================

def edmister_omega_array(
    Tb: NDArray[np.float64],
    Tc: NDArray[np.float64],
    Pc: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Vectorized Edmister acentric factor correlation.

    Parameters
    ----------
    Tb : ndarray
        Normal boiling points in Kelvin.
    Tc : ndarray
        Critical temperatures in Kelvin.
    Pc : ndarray
        Critical pressures in Pascal.

    Returns
    -------
    ndarray
        Acentric factors (dimensionless).
    """
    Tb = np.asarray(Tb, dtype=np.float64)
    Tc = np.asarray(Tc, dtype=np.float64)
    Pc = np.asarray(Pc, dtype=np.float64)

    Pa = 101325.0
    Tbr = Tb / Tc

    omega = (3.0 / 7.0) * np.log10(Pc / Pa) / (1.0 / Tbr - 1.0) - 1.0

    # Clamp to reasonable range
    omega = np.clip(omega, -0.5, 2.0)

    return omega


def kesler_lee_omega_array(
    Tb: NDArray[np.float64],
    Tc: NDArray[np.float64],
    Pc: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Vectorized Kesler-Lee acentric factor correlation.

    Parameters
    ----------
    Tb : ndarray
        Normal boiling points in Kelvin.
    Tc : ndarray
        Critical temperatures in Kelvin.
    Pc : ndarray
        Critical pressures in Pascal.

    Returns
    -------
    ndarray
        Acentric factors (dimensionless).
    """
    Tb = np.asarray(Tb, dtype=np.float64)
    Tc = np.asarray(Tc, dtype=np.float64)
    Pc = np.asarray(Pc, dtype=np.float64)

    Tbr = Tb / Tc
    ln_Pc_Pa = np.log(Pc / 101325.0)

    # Calculate for all using the Tbr <= 0.8 form
    num = (
        -ln_Pc_Pa
        - 5.92714
        + 6.09648 / Tbr
        + 1.28862 * np.log(Tbr)
        - 0.169347 * (Tbr ** 6)
    )
    den = (
        15.2518
        - 15.6875 / Tbr
        - 13.4721 * np.log(Tbr)
        + 0.43577 * (Tbr ** 6)
    )

    omega = num / den

    # For Tbr > 0.8, use Edmister as fallback
    heavy_mask = Tbr > 0.8
    if np.any(heavy_mask):
        omega_edmister = edmister_omega_array(Tb, Tc, Pc)
        omega = np.where(heavy_mask, omega_edmister, omega)

    # Clamp to reasonable range
    omega = np.clip(omega, -0.5, 2.0)

    return omega


# =============================================================================
# Input Validation
# =============================================================================

def _validate_inputs(
    Tb: Optional[float] = None,
    Tc: Optional[float] = None,
    Pc: Optional[float] = None,
) -> None:
    """Validate acentric factor correlation inputs."""
    if Tb is not None:
        if not np.isfinite(Tb):
            raise ValueError(f"Tb must be finite, got {Tb}")
        if Tb <= 0:
            raise ValueError(f"Tb must be positive, got {Tb}")

    if Tc is not None:
        if not np.isfinite(Tc):
            raise ValueError(f"Tc must be finite, got {Tc}")
        if Tc <= 0:
            raise ValueError(f"Tc must be positive, got {Tc}")

    if Pc is not None:
        if not np.isfinite(Pc):
            raise ValueError(f"Pc must be finite, got {Pc}")
        if Pc <= 0:
            raise ValueError(f"Pc must be positive, got {Pc}")


# =============================================================================
# Utility Functions
# =============================================================================

def omega_from_vapor_pressure(
    Psat_Tr07: float,
    Pc: float,
) -> float:
    """
    Calculate acentric factor from vapor pressure at Tr = 0.7.

    This is the definition of acentric factor by Pitzer:
        omega = -log10(Psat/Pc) - 1   at Tr = 0.7

    Parameters
    ----------
    Psat_Tr07 : float
        Saturation pressure at Tr = 0.7 in Pascal.
    Pc : float
        Critical pressure in Pascal.

    Returns
    -------
    float
        Acentric factor (dimensionless).
    """
    if Psat_Tr07 <= 0 or Pc <= 0:
        raise ValueError("Pressures must be positive")
    if Psat_Tr07 >= Pc:
        raise ValueError("Psat at Tr=0.7 should be less than Pc")

    return -np.log10(Psat_Tr07 / Pc) - 1.0

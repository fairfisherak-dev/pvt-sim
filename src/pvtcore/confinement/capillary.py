"""Capillary pressure calculations for nano-confined systems.

In tight/shale reservoirs, nanometer-scale pores create significant
capillary pressure differences between coexisting liquid and vapor phases.
This module provides calculations for capillary pressure effects.

The Young-Laplace equation relates capillary pressure to interfacial tension:
    Pc = 2σ cos(θ) / r

where:
    Pc = capillary pressure (Pa)
    σ  = interfacial tension (N/m)
    θ  = contact angle (radians)
    r  = pore radius (m)

In petroleum systems:
    - Liquid is typically the wetting phase
    - Vapor pressure is higher: Pv = PL + Pc
    - Complete wetting (θ = 0) is often assumed

Units Convention:
- Pressure: Pa
- Interfacial tension: mN/m (= 10⁻³ N/m)
- Pore radius: nm or m
- Contact angle: degrees (converted internally to radians)

References
----------
[1] Nojabaei, B., Johns, R.T., and Chu, L. (2013).
    "Effect of Capillary Pressure on Phase Behavior in Tight Rocks and Shales."
    SPE Reservoir Evaluation & Engineering, 16(3), 281-289. SPE-159258.
[2] Sandoval, D.R., Yan, W., Michelsen, M.L., and Stenby, E.H. (2016).
    "The Phase Envelope of Multicomponent Mixtures in the Presence of
    a Capillary Pressure Difference." Ind. Eng. Chem. Res., 55, 6530-6538.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np


# Physical constants
COMPLETE_WETTING_ANGLE = 0.0  # degrees (cos(0) = 1)


@dataclass
class CapillaryPressureResult:
    """Results from capillary pressure calculation.

    Attributes:
        Pc: Capillary pressure in Pa
        Pc_bar: Capillary pressure in bar
        Pc_MPa: Capillary pressure in MPa
        ift: Interfacial tension used (mN/m)
        pore_radius: Pore radius used (m)
        contact_angle: Contact angle used (degrees)
    """
    Pc: float
    Pc_bar: float
    Pc_MPa: float
    ift: float
    pore_radius: float
    contact_angle: float


def calculate_capillary_pressure(
    ift: float,
    pore_radius: float,
    contact_angle: float = COMPLETE_WETTING_ANGLE,
    radius_units: str = 'nm',
) -> CapillaryPressureResult:
    """Calculate capillary pressure using Young-Laplace equation.

    Pc = 2σ cos(θ) / r

    Parameters
    ----------
    ift : float
        Interfacial tension in mN/m.
    pore_radius : float
        Pore radius in specified units (default: nm).
    contact_angle : float
        Contact angle in degrees. Default is 0 (complete wetting).
    radius_units : str
        Units for pore_radius. Options: 'nm', 'm', 'um', 'angstrom'.

    Returns
    -------
    CapillaryPressureResult
        Capillary pressure calculation results.

    Raises
    ------
    ValueError
        If inputs are invalid (negative IFT, zero radius, etc.)

    Notes
    -----
    For oil-wet systems, contact angle > 90° gives negative Pc
    (vapor is the wetting phase). This implementation allows
    both positive and negative Pc.

    Examples
    --------
    >>> # 10 nm pore with 10 mN/m IFT
    >>> result = calculate_capillary_pressure(ift=10.0, pore_radius=10.0)
    >>> print(f"Pc = {result.Pc_MPa:.2f} MPa")
    Pc = 2.00 MPa

    >>> # Same with 5 nm pore (doubled Pc)
    >>> result = calculate_capillary_pressure(ift=10.0, pore_radius=5.0)
    >>> print(f"Pc = {result.Pc_MPa:.2f} MPa")
    Pc = 4.00 MPa
    """
    # Validate inputs
    if ift < 0:
        raise ValueError(f"IFT must be non-negative, got {ift}")
    if pore_radius <= 0:
        raise ValueError(f"Pore radius must be positive, got {pore_radius}")
    if not -180 <= contact_angle <= 180:
        raise ValueError(f"Contact angle must be in [-180, 180], got {contact_angle}")

    # Convert pore radius to meters
    r_m = _convert_radius_to_meters(pore_radius, radius_units)

    # Convert IFT from mN/m to N/m
    sigma_N_m = ift * 1e-3

    # Convert contact angle to radians
    theta_rad = math.radians(contact_angle)

    # Young-Laplace equation: Pc = 2σ cos(θ) / r
    Pc = 2.0 * sigma_N_m * math.cos(theta_rad) / r_m

    return CapillaryPressureResult(
        Pc=Pc,
        Pc_bar=Pc / 1e5,
        Pc_MPa=Pc / 1e6,
        ift=ift,
        pore_radius=r_m,
        contact_angle=contact_angle,
    )


def capillary_pressure_simple(
    ift: float,
    pore_radius_nm: float,
) -> float:
    """Simplified capillary pressure for complete wetting.

    Pc = 2σ / r  (assuming cos(θ) = 1)

    Parameters
    ----------
    ift : float
        Interfacial tension in mN/m.
    pore_radius_nm : float
        Pore radius in nanometers.

    Returns
    -------
    float
        Capillary pressure in Pa.

    Examples
    --------
    >>> Pc = capillary_pressure_simple(10.0, 10.0)  # 10 mN/m, 10 nm
    >>> print(f"Pc = {Pc/1e6:.2f} MPa")
    Pc = 2.00 MPa
    """
    if ift < 0:
        raise ValueError(f"IFT must be non-negative, got {ift}")
    if pore_radius_nm <= 0:
        raise ValueError(f"Pore radius must be positive, got {pore_radius_nm}")

    # Convert units: mN/m to N/m, nm to m
    sigma = ift * 1e-3  # N/m
    r = pore_radius_nm * 1e-9  # m

    return 2.0 * sigma / r


def vapor_pressure_from_liquid(
    P_liquid: float,
    Pc: float,
) -> float:
    """Calculate vapor phase pressure from liquid pressure and Pc.

    In confined systems with liquid as the wetting phase:
        Pv = PL + Pc

    Parameters
    ----------
    P_liquid : float
        Liquid phase pressure in Pa.
    Pc : float
        Capillary pressure in Pa.

    Returns
    -------
    float
        Vapor phase pressure in Pa.
    """
    return P_liquid + Pc


def liquid_pressure_from_vapor(
    P_vapor: float,
    Pc: float,
) -> float:
    """Calculate liquid phase pressure from vapor pressure and Pc.

    In confined systems with liquid as the wetting phase:
        PL = Pv - Pc

    Parameters
    ----------
    P_vapor : float
        Vapor phase pressure in Pa.
    Pc : float
        Capillary pressure in Pa.

    Returns
    -------
    float
        Liquid phase pressure in Pa.
    """
    return P_vapor - Pc


def modified_k_value(
    K_bulk: float,
    P_liquid: float,
    P_vapor: float,
) -> float:
    """Calculate modified K-value for confined equilibrium.

    In confined systems, the equilibrium condition becomes:
        xi φi^L(PL) PL = yi φi^V(Pv) Pv

    Leading to modified K-value:
        Ki_confined = Ki_bulk × (PL / Pv)

    Parameters
    ----------
    K_bulk : float
        Bulk K-value (from equal pressure flash).
    P_liquid : float
        Liquid phase pressure in Pa.
    P_vapor : float
        Vapor phase pressure in Pa.

    Returns
    -------
    float
        Modified K-value for confined equilibrium.

    Notes
    -----
    Since Pv > PL in confined systems (wetting liquid phase),
    Ki_confined < Ki_bulk. This leads to:
    - Reduced vapor fraction
    - Suppressed bubble point (lower P)
    - Enhanced dew point (higher P)
    """
    return K_bulk * (P_liquid / P_vapor)


def modified_k_values_array(
    K_bulk: np.ndarray,
    P_liquid: float,
    P_vapor: float,
) -> np.ndarray:
    """Calculate modified K-values for all components.

    Parameters
    ----------
    K_bulk : ndarray
        Bulk K-values for all components.
    P_liquid : float
        Liquid phase pressure in Pa.
    P_vapor : float
        Vapor phase pressure in Pa.

    Returns
    -------
    ndarray
        Modified K-values for confined equilibrium.
    """
    K_bulk = np.asarray(K_bulk)
    pressure_ratio = P_liquid / P_vapor
    return K_bulk * pressure_ratio


def estimate_bubble_point_suppression(
    Pc: float,
    P_bubble_bulk: float,
) -> float:
    """Estimate bubble point suppression due to capillary pressure.

    A first-order approximation for bubble point shift:
        ΔPb ≈ -Pc

    The actual shift depends on composition and temperature,
    but this gives a rough estimate.

    Parameters
    ----------
    Pc : float
        Capillary pressure in Pa.
    P_bubble_bulk : float
        Bulk bubble point pressure in Pa.

    Returns
    -------
    float
        Estimated confined bubble point pressure in Pa.

    Notes
    -----
    This is a rough approximation. The actual bubble point
    shift requires iterative calculation with the confined
    flash algorithm.
    """
    # Bubble point is suppressed (lowered) in confined systems
    return P_bubble_bulk - Pc


def estimate_dew_point_enhancement(
    Pc: float,
    P_dew_bulk: float,
) -> float:
    """Estimate dew point enhancement due to capillary pressure.

    A first-order approximation for dew point shift:
        ΔPd ≈ +Pc

    Parameters
    ----------
    Pc : float
        Capillary pressure in Pa.
    P_dew_bulk : float
        Bulk dew point pressure in Pa.

    Returns
    -------
    float
        Estimated confined dew point pressure in Pa.

    Notes
    -----
    This is a rough approximation. The actual dew point
    shift requires iterative calculation with the confined
    flash algorithm.
    """
    # Dew point is enhanced (raised) in confined systems
    return P_dew_bulk + Pc


def _convert_radius_to_meters(radius: float, units: str) -> float:
    """Convert pore radius to meters."""
    units = units.lower()
    if units == 'm':
        return radius
    elif units == 'nm':
        return radius * 1e-9
    elif units == 'um' or units == 'µm' or units == 'micron':
        return radius * 1e-6
    elif units == 'angstrom' or units == 'å':
        return radius * 1e-10
    else:
        raise ValueError(
            f"Unknown radius units '{units}'. "
            "Use 'nm', 'm', 'um', or 'angstrom'."
        )


def critical_pore_radius(
    ift: float,
    Pc_target: float,
    contact_angle: float = COMPLETE_WETTING_ANGLE,
) -> float:
    """Calculate pore radius required for a target capillary pressure.

    Rearranging Young-Laplace: r = 2σ cos(θ) / Pc

    Parameters
    ----------
    ift : float
        Interfacial tension in mN/m.
    Pc_target : float
        Target capillary pressure in Pa.
    contact_angle : float
        Contact angle in degrees.

    Returns
    -------
    float
        Required pore radius in nanometers.

    Examples
    --------
    >>> # What pore radius gives 1 MPa Pc with 5 mN/m IFT?
    >>> r = critical_pore_radius(ift=5.0, Pc_target=1e6)
    >>> print(f"r = {r:.1f} nm")
    r = 10.0 nm
    """
    if Pc_target <= 0:
        raise ValueError("Target Pc must be positive")

    sigma_N_m = ift * 1e-3
    theta_rad = math.radians(contact_angle)

    r_m = 2.0 * sigma_N_m * math.cos(theta_rad) / Pc_target
    r_nm = r_m * 1e9  # Convert to nm

    return r_nm

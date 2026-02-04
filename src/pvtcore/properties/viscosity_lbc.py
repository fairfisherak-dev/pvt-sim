"""Lohrenz-Bray-Clark (LBC) viscosity correlation.

This module implements the LBC correlation for calculating viscosity
of petroleum reservoir fluids from composition and density.

The LBC method relates viscosity to reduced density through a fourth-degree
polynomial. It requires mixture critical properties and dilute gas viscosity
as inputs.

Units Convention:
- Viscosity: Pa·s (SI), can convert to cp (1 Pa·s = 1000 cp)
- Density: mol/m³ (molar)
- Temperature: K
- Pressure: Pa

References
----------
[1] Lohrenz, J., Bray, B.G., and Clark, C.R. (1964).
    "Calculating Viscosities of Reservoir Fluids From Their Compositions."
    Journal of Petroleum Technology, October 1964, 1171-1176. SPE-915.
[2] Stiel, L.I. and Thodos, G. (1961).
    "The Viscosity of Nonpolar Gases at Normal Pressures."
    AIChE Journal, 7(4), 611-615.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ..core.constants import R
from ..core.errors import PropertyError, ValidationError
from ..models.component import Component


# LBC polynomial coefficients (from original paper)
# (μ - μ*) × ξ = a₀ + a₁ρᵣ + a₂ρᵣ² + a₃ρᵣ³ + a₄ρᵣ⁴
LBC_COEFFICIENTS = {
    'a0': 0.1023,
    'a1': 0.023364,
    'a2': 0.058533,
    'a3': -0.040758,
    'a4': 0.0093324,
}

# Stiel-Thodos constants for dilute gas viscosity
STIEL_THODOS_CONST = 34.0e-5


@dataclass
class ViscosityResult:
    """Results from viscosity calculation.

    Attributes:
        viscosity: Viscosity in Pa·s
        viscosity_cp: Viscosity in centipoise (cp)
        dilute_gas_viscosity: Low-pressure gas viscosity in Pa·s
        reduced_density: ρ / ρc (dimensionless)
        inverse_viscosity_param: ξ parameter (dimensionless)
    """
    viscosity: float
    viscosity_cp: float
    dilute_gas_viscosity: float
    reduced_density: float
    inverse_viscosity_param: float


def calculate_viscosity_lbc(
    molar_density: float,
    temperature: float,
    composition: NDArray[np.float64],
    components: List[Component],
    MW_mix: Optional[float] = None,
) -> ViscosityResult:
    """Calculate viscosity using the Lohrenz-Bray-Clark correlation.

    The LBC correlation computes viscosity from:
        (μ - μ*) × ξ = f(ρᵣ)

    where:
        μ* = dilute gas viscosity (low pressure limit)
        ξ = inverse viscosity parameter = (Tc_mix)^(1/6) / (MW_mix^(1/2) × Pc_mix^(2/3))
        ρᵣ = ρ / ρc = reduced density
        f(ρᵣ) = polynomial in reduced density

    Parameters
    ----------
    molar_density : float
        Molar density in mol/m³.
    temperature : float
        Temperature in K.
    composition : ndarray
        Mole fractions of each component.
    components : list of Component
        Component objects with Tc, Pc, Vc, MW, omega properties.
    MW_mix : float, optional
        Pre-calculated mixture molecular weight (g/mol).
        If not provided, calculated from composition.

    Returns
    -------
    ViscosityResult
        Viscosity calculation results.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    PropertyError
        If viscosity calculation fails.

    Notes
    -----
    The LBC correlation was developed for hydrocarbon mixtures and
    is most accurate for:
    - Reduced density < 3
    - Hydrocarbon systems (less accurate for high CO₂/H₂S content)

    The dilute gas viscosity is estimated using the Stiel-Thodos
    correlation for non-polar gases.

    References
    ----------
    Lohrenz, Bray & Clark (1964), SPE-915.

    Examples
    --------
    >>> from pvtcore.models.component import load_components
    >>> components = load_components()
    >>> binary = [components['C1'], components['C3']]
    >>> z = np.array([0.7, 0.3])
    >>> # Assume molar_density from EOS = 5000 mol/m³
    >>> result = calculate_viscosity_lbc(5000.0, 300.0, z, binary)
    >>> print(f"Viscosity = {result.viscosity_cp:.4f} cp")
    """
    # Input validation
    _validate_viscosity_inputs(molar_density, temperature, composition, components)

    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()  # Normalize
    n = len(components)

    # Calculate mixture properties using LBC mixing rules
    # Tc_mix = Σ xᵢ Tcᵢ
    Tc_mix = sum(z[i] * components[i].Tc for i in range(n))

    # Pc_mix = Σ xᵢ Pcᵢ (in Pa, then convert to atm for LBC)
    Pc_mix_Pa = sum(z[i] * components[i].Pc for i in range(n))
    Pc_mix_atm = Pc_mix_Pa / 101325.0  # Convert to atm

    # Vc_mix = Σ xᵢ Vcᵢ (in m³/mol, then convert to cm³/mol)
    Vc_mix_m3 = sum(z[i] * components[i].Vc for i in range(n))
    Vc_mix_cm3 = Vc_mix_m3 * 1e6  # m³/mol to cm³/mol

    # MW_mix = Σ xᵢ MWᵢ
    if MW_mix is None:
        MW_mix = sum(z[i] * components[i].MW for i in range(n))

    # Critical molar density (mol/cm³, then mol/m³)
    rho_c_mol_cm3 = 1.0 / Vc_mix_cm3  # mol/cm³
    rho_c = rho_c_mol_cm3 * 1e6  # mol/m³

    # Reduced density
    rho_r = molar_density / rho_c

    # Check reduced density range
    if rho_r > 3.0:
        # LBC extrapolation warning - still compute but may be inaccurate
        pass
    if rho_r < 0:
        raise PropertyError(
            "Reduced density cannot be negative",
            property_name="viscosity",
        )

    # Inverse viscosity parameter ξ
    # ξ = (Tc)^(1/6) / (MW^(1/2) × Pc^(2/3))
    # Units: Tc in K, MW in g/mol, Pc in atm → ξ in (cp)⁻¹
    xi = (Tc_mix ** (1.0/6.0)) / (MW_mix ** 0.5 * Pc_mix_atm ** (2.0/3.0))

    # Dilute gas viscosity using Stiel-Thodos correlation
    # For mixture, use mixing rule
    mu_star = _dilute_gas_viscosity_mix(temperature, z, components)

    # LBC polynomial: (μ - μ*) × ξ = f(ρᵣ)
    a = LBC_COEFFICIENTS
    f_rho = (
        a['a0']
        + a['a1'] * rho_r
        + a['a2'] * rho_r ** 2
        + a['a3'] * rho_r ** 3
        + a['a4'] * rho_r ** 4
    )

    # Handle the polynomial giving negative values at low density
    # LBC should give f(ρᵣ) ≥ 0 for physical conditions
    # At very low density, the polynomial can become slightly negative
    # In that limit, μ ≈ μ*
    if f_rho < 0:
        f_rho = 0.0

    # Solve for viscosity in centipoise
    # (μ - μ*) × ξ = f(ρᵣ)
    # μ = μ* + f(ρᵣ) / ξ
    # Note: The LBC polynomial is structured such that when f(ρᵣ) approaches
    # a certain threshold, we need to use the fourth-root form

    # The original LBC correlation actually uses:
    # [(μ - μ*) × ξ + 10⁻⁴]^(1/4) = f(ρᵣ)
    # So: (μ - μ*) × ξ = f(ρᵣ)⁴ - 10⁻⁴

    f_rho_fourth = f_rho ** 4 - 1e-4
    if f_rho_fourth < 0:
        f_rho_fourth = 0.0

    # μ in cp (original LBC formulation)
    mu_cp = mu_star + f_rho_fourth / xi

    # Convert to Pa·s (SI)
    mu_Pa_s = mu_cp / 1000.0

    return ViscosityResult(
        viscosity=mu_Pa_s,
        viscosity_cp=mu_cp,
        dilute_gas_viscosity=mu_star / 1000.0,  # Convert to Pa·s
        reduced_density=rho_r,
        inverse_viscosity_param=xi,
    )


def _dilute_gas_viscosity_stiel_thodos(
    temperature: float,
    Tc: float,
    Pc: float,
    MW: float,
    omega: float,
) -> float:
    """Calculate dilute gas viscosity using Stiel-Thodos correlation.

    For non-polar gases:
        μ* × ξ = 34.0 × 10⁻⁵ × Tr^0.94   (for Tr < 1.5)
        μ* × ξ = 17.78 × 10⁻⁵ × (4.58Tr - 1.67)^0.625   (for Tr ≥ 1.5)

    where ξ = Tc^(1/6) / (MW^(1/2) × Pc^(2/3))

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Tc : float
        Critical temperature in K.
    Pc : float
        Critical pressure in Pa.
    MW : float
        Molecular weight in g/mol.
    omega : float
        Acentric factor.

    Returns
    -------
    float
        Dilute gas viscosity in centipoise (cp).
    """
    Tr = temperature / Tc
    Pc_atm = Pc / 101325.0

    # Inverse viscosity parameter
    xi = (Tc ** (1.0/6.0)) / (MW ** 0.5 * Pc_atm ** (2.0/3.0))

    # Stiel-Thodos correlation (for non-polar gases)
    if Tr < 1.5:
        mu_xi = 34.0e-5 * (Tr ** 0.94)
    else:
        mu_xi = 17.78e-5 * ((4.58 * Tr - 1.67) ** 0.625)

    # Polar correction (simplified - for H₂S, CO₂, etc.)
    # Full implementation would use Chung et al. or other methods
    # For now, apply a simple polar correction based on acentric factor
    if omega > 0.25:
        # Higher acentric factor indicates more polar/complex molecule
        polar_factor = 1.0 + 0.1 * (omega - 0.25)
        mu_xi *= polar_factor

    mu_star = mu_xi / xi  # Viscosity in cp

    return mu_star


def _dilute_gas_viscosity_mix(
    temperature: float,
    composition: NDArray[np.float64],
    components: List[Component],
) -> float:
    """Calculate mixture dilute gas viscosity using Wilke mixing rule.

    The Wilke mixing rule for viscosity:
        μ_mix = Σᵢ (xᵢ μᵢ) / Σⱼ (xⱼ φᵢⱼ)

    where φᵢⱼ is the interaction parameter.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    composition : ndarray
        Mole fractions.
    components : list of Component
        Component objects.

    Returns
    -------
    float
        Mixture dilute gas viscosity in centipoise (cp).
    """
    z = np.asarray(composition)
    n = len(components)

    # Calculate pure component dilute gas viscosities
    mu_pure = np.zeros(n)
    for i, comp in enumerate(components):
        mu_pure[i] = _dilute_gas_viscosity_stiel_thodos(
            temperature, comp.Tc, comp.Pc, comp.MW, comp.omega
        )

    # Wilke mixing rule
    # φᵢⱼ = [1 + (μᵢ/μⱼ)^0.5 × (MWⱼ/MWᵢ)^0.25]² / [8(1 + MWᵢ/MWⱼ)]^0.5
    mu_mix = 0.0
    for i in range(n):
        if z[i] < 1e-10:
            continue

        denominator = 0.0
        for j in range(n):
            if z[j] < 1e-10:
                continue

            MW_ratio = components[j].MW / components[i].MW
            mu_ratio = mu_pure[i] / max(mu_pure[j], 1e-20)

            phi_ij = (
                (1.0 + mu_ratio ** 0.5 * MW_ratio ** 0.25) ** 2
                / (8.0 * (1.0 + components[i].MW / components[j].MW)) ** 0.5
            )
            denominator += z[j] * phi_ij

        mu_mix += z[i] * mu_pure[i] / max(denominator, 1e-20)

    return mu_mix


def calculate_phase_viscosities(
    liquid_molar_density: float,
    vapor_molar_density: float,
    temperature: float,
    liquid_composition: NDArray[np.float64],
    vapor_composition: NDArray[np.float64],
    components: List[Component],
    liquid_MW: Optional[float] = None,
    vapor_MW: Optional[float] = None,
) -> tuple[ViscosityResult, ViscosityResult]:
    """Calculate viscosities for both liquid and vapor phases.

    Parameters
    ----------
    liquid_molar_density : float
        Liquid molar density in mol/m³.
    vapor_molar_density : float
        Vapor molar density in mol/m³.
    temperature : float
        Temperature in K.
    liquid_composition : ndarray
        Liquid phase mole fractions.
    vapor_composition : ndarray
        Vapor phase mole fractions.
    components : list of Component
        Component objects.
    liquid_MW : float, optional
        Liquid mixture molecular weight.
    vapor_MW : float, optional
        Vapor mixture molecular weight.

    Returns
    -------
    tuple of ViscosityResult
        (liquid_viscosity_result, vapor_viscosity_result)
    """
    liquid_result = calculate_viscosity_lbc(
        liquid_molar_density, temperature, liquid_composition,
        components, MW_mix=liquid_MW,
    )

    vapor_result = calculate_viscosity_lbc(
        vapor_molar_density, temperature, vapor_composition,
        components, MW_mix=vapor_MW,
    )

    return liquid_result, vapor_result


def _validate_viscosity_inputs(
    molar_density: float,
    temperature: float,
    composition: NDArray[np.float64],
    components: List[Component],
) -> None:
    """Validate viscosity calculation inputs."""
    if molar_density <= 0:
        raise ValidationError(
            "Molar density must be positive",
            parameter="molar_density",
            value=molar_density,
        )
    if temperature <= 0:
        raise ValidationError(
            "Temperature must be positive",
            parameter="temperature",
            value=temperature,
        )

    z = np.asarray(composition)
    if len(z) != len(components):
        raise ValidationError(
            "Composition length must match number of components",
            parameter="composition",
            value={"got": len(z), "expected": len(components)},
        )

    # Check that all components have required properties
    for i, comp in enumerate(components):
        if comp.Vc <= 0:
            raise ValidationError(
                f"Component {comp.name} has invalid Vc (critical volume)",
                parameter="components",
                value=comp.Vc,
            )

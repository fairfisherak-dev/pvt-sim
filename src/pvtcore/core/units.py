"""Unit conversion functions for PVT calculations.

This module provides functions to convert between different unit systems
commonly used in petroleum engineering and thermodynamics.
"""

from typing import Union
try:
    import numpy as np
    HAS_NUMPY = True
    # Type alias for numeric types
    Numeric = Union[float, int, np.ndarray]
except ImportError:
    HAS_NUMPY = False
    # Type alias for numeric types (without numpy)
    Numeric = Union[float, int]

from . import constants as const


# ============================================================================
# TEMPERATURE CONVERSIONS
# ============================================================================

def celsius_to_kelvin(temp_c: Numeric) -> Numeric:
    """Convert temperature from Celsius to Kelvin.

    Args:
        temp_c: Temperature in degrees Celsius

    Returns:
        Temperature in Kelvin

    Example:
        >>> celsius_to_kelvin(25)
        298.15
    """
    return temp_c + const.ZERO_CELSIUS_IN_KELVIN


def kelvin_to_celsius(temp_k: Numeric) -> Numeric:
    """Convert temperature from Kelvin to Celsius.

    Args:
        temp_k: Temperature in Kelvin

    Returns:
        Temperature in degrees Celsius

    Example:
        >>> kelvin_to_celsius(298.15)
        25.0
    """
    return temp_k - const.ZERO_CELSIUS_IN_KELVIN


def fahrenheit_to_celsius(temp_f: Numeric) -> Numeric:
    """Convert temperature from Fahrenheit to Celsius.

    Args:
        temp_f: Temperature in degrees Fahrenheit

    Returns:
        Temperature in degrees Celsius

    Example:
        >>> fahrenheit_to_celsius(77)
        25.0
    """
    return (temp_f - 32.0) * const.FAHRENHEIT_TO_CELSIUS_SLOPE


def celsius_to_fahrenheit(temp_c: Numeric) -> Numeric:
    """Convert temperature from Celsius to Fahrenheit.

    Args:
        temp_c: Temperature in degrees Celsius

    Returns:
        Temperature in degrees Fahrenheit

    Example:
        >>> celsius_to_fahrenheit(25)
        77.0
    """
    return temp_c * const.CELSIUS_TO_FAHRENHEIT_SLOPE + 32.0


def fahrenheit_to_kelvin(temp_f: Numeric) -> Numeric:
    """Convert temperature from Fahrenheit to Kelvin.

    Args:
        temp_f: Temperature in degrees Fahrenheit

    Returns:
        Temperature in Kelvin

    Example:
        >>> fahrenheit_to_kelvin(77)
        298.15
    """
    return celsius_to_kelvin(fahrenheit_to_celsius(temp_f))


def kelvin_to_fahrenheit(temp_k: Numeric) -> Numeric:
    """Convert temperature from Kelvin to Fahrenheit.

    Args:
        temp_k: Temperature in Kelvin

    Returns:
        Temperature in degrees Fahrenheit

    Example:
        >>> kelvin_to_fahrenheit(298.15)
        77.0
    """
    return celsius_to_fahrenheit(kelvin_to_celsius(temp_k))


def rankine_to_kelvin(temp_r: Numeric) -> Numeric:
    """Convert temperature from Rankine to Kelvin.

    Args:
        temp_r: Temperature in degrees Rankine

    Returns:
        Temperature in Kelvin

    Example:
        >>> rankine_to_kelvin(536.67)
        298.15
    """
    return temp_r * const.RANKINE_TO_KELVIN_FACTOR


def kelvin_to_rankine(temp_k: Numeric) -> Numeric:
    """Convert temperature from Kelvin to Rankine.

    Args:
        temp_k: Temperature in Kelvin

    Returns:
        Temperature in degrees Rankine

    Example:
        >>> kelvin_to_rankine(298.15)
        536.67
    """
    return temp_k / const.RANKINE_TO_KELVIN_FACTOR


# ============================================================================
# PRESSURE CONVERSIONS
# ============================================================================

def bar_to_pa(pressure_bar: Numeric) -> Numeric:
    """Convert pressure from bar to Pascal.

    Args:
        pressure_bar: Pressure in bar

    Returns:
        Pressure in Pascal

    Example:
        >>> bar_to_pa(1.0)
        100000.0
    """
    return pressure_bar * const.BAR_TO_PA


def pa_to_bar(pressure_pa: Numeric) -> Numeric:
    """Convert pressure from Pascal to bar.

    Args:
        pressure_pa: Pressure in Pascal

    Returns:
        Pressure in bar

    Example:
        >>> pa_to_bar(100000)
        1.0
    """
    return pressure_pa / const.BAR_TO_PA


def psi_to_pa(pressure_psi: Numeric) -> Numeric:
    """Convert pressure from psi to Pascal.

    Args:
        pressure_psi: Pressure in psi

    Returns:
        Pressure in Pascal

    Example:
        >>> psi_to_pa(14.696)
        101325.0
    """
    return pressure_psi * const.PSI_TO_PA


def pa_to_psi(pressure_pa: Numeric) -> Numeric:
    """Convert pressure from Pascal to psi.

    Args:
        pressure_pa: Pressure in Pascal

    Returns:
        Pressure in psi

    Example:
        >>> pa_to_psi(101325)
        14.696
    """
    return pressure_pa / const.PSI_TO_PA


def atm_to_pa(pressure_atm: Numeric) -> Numeric:
    """Convert pressure from atm to Pascal.

    Args:
        pressure_atm: Pressure in atm

    Returns:
        Pressure in Pascal

    Example:
        >>> atm_to_pa(1.0)
        101325.0
    """
    return pressure_atm * const.ATM_TO_PA


def pa_to_atm(pressure_pa: Numeric) -> Numeric:
    """Convert pressure from Pascal to atm.

    Args:
        pressure_pa: Pressure in Pascal

    Returns:
        Pressure in atm

    Example:
        >>> pa_to_atm(101325)
        1.0
    """
    return pressure_pa / const.ATM_TO_PA


def torr_to_pa(pressure_torr: Numeric) -> Numeric:
    """Convert pressure from Torr (mmHg) to Pascal.

    Args:
        pressure_torr: Pressure in Torr

    Returns:
        Pressure in Pascal

    Example:
        >>> torr_to_pa(760)
        101325.0
    """
    return pressure_torr * const.TORR_TO_PA


def pa_to_torr(pressure_pa: Numeric) -> Numeric:
    """Convert pressure from Pascal to Torr (mmHg).

    Args:
        pressure_pa: Pressure in Pascal

    Returns:
        Pressure in Torr

    Example:
        >>> pa_to_torr(101325)
        760.0
    """
    return pressure_pa / const.TORR_TO_PA


def mpa_to_pa(pressure_mpa: Numeric) -> Numeric:
    """Convert pressure from MPa to Pascal.

    Args:
        pressure_mpa: Pressure in MPa

    Returns:
        Pressure in Pascal

    Example:
        >>> mpa_to_pa(1.0)
        1000000.0
    """
    return pressure_mpa * 1e6


def pa_to_mpa(pressure_pa: Numeric) -> Numeric:
    """Convert pressure from Pascal to MPa.

    Args:
        pressure_pa: Pressure in Pascal

    Returns:
        Pressure in MPa

    Example:
        >>> pa_to_mpa(1000000)
        1.0
    """
    return pressure_pa / 1e6


def kpa_to_pa(pressure_kpa: Numeric) -> Numeric:
    """Convert pressure from kPa to Pascal.

    Args:
        pressure_kpa: Pressure in kPa

    Returns:
        Pressure in Pascal

    Example:
        >>> kpa_to_pa(101.325)
        101325.0
    """
    return pressure_kpa * 1e3


def pa_to_kpa(pressure_pa: Numeric) -> Numeric:
    """Convert pressure from Pascal to kPa.

    Args:
        pressure_pa: Pressure in Pascal

    Returns:
        Pressure in kPa

    Example:
        >>> pa_to_kpa(101325)
        101.325
    """
    return pressure_pa / 1e3


# Cross-conversions for convenience
def bar_to_psi(pressure_bar: Numeric) -> Numeric:
    """Convert pressure from bar to psi."""
    return pa_to_psi(bar_to_pa(pressure_bar))


def psi_to_bar(pressure_psi: Numeric) -> Numeric:
    """Convert pressure from psi to bar."""
    return pa_to_bar(psi_to_pa(pressure_psi))


def bar_to_atm(pressure_bar: Numeric) -> Numeric:
    """Convert pressure from bar to atm."""
    return pa_to_atm(bar_to_pa(pressure_bar))


def atm_to_bar(pressure_atm: Numeric) -> Numeric:
    """Convert pressure from atm to bar."""
    return pa_to_bar(atm_to_pa(pressure_atm))


# ============================================================================
# VOLUME CONVERSIONS
# ============================================================================

def liter_to_m3(volume_l: Numeric) -> Numeric:
    """Convert volume from liters to cubic meters.

    Args:
        volume_l: Volume in liters

    Returns:
        Volume in cubic meters

    Example:
        >>> liter_to_m3(1000)
        1.0
    """
    return volume_l * const.L_TO_M3


def m3_to_liter(volume_m3: Numeric) -> Numeric:
    """Convert volume from cubic meters to liters.

    Args:
        volume_m3: Volume in cubic meters

    Returns:
        Volume in liters

    Example:
        >>> m3_to_liter(1.0)
        1000.0
    """
    return volume_m3 / const.L_TO_M3


def ft3_to_m3(volume_ft3: Numeric) -> Numeric:
    """Convert volume from cubic feet to cubic meters.

    Args:
        volume_ft3: Volume in cubic feet

    Returns:
        Volume in cubic meters

    Example:
        >>> ft3_to_m3(35.3147)
        1.0
    """
    return volume_ft3 * const.FT3_TO_M3


def m3_to_ft3(volume_m3: Numeric) -> Numeric:
    """Convert volume from cubic meters to cubic feet.

    Args:
        volume_m3: Volume in cubic meters

    Returns:
        Volume in cubic feet

    Example:
        >>> m3_to_ft3(1.0)
        35.3147
    """
    return volume_m3 / const.FT3_TO_M3


def bbl_to_m3(volume_bbl: Numeric) -> Numeric:
    """Convert volume from barrels to cubic meters.

    Args:
        volume_bbl: Volume in barrels

    Returns:
        Volume in cubic meters

    Example:
        >>> bbl_to_m3(6.28981)
        1.0
    """
    return volume_bbl * const.BBL_TO_M3


def m3_to_bbl(volume_m3: Numeric) -> Numeric:
    """Convert volume from cubic meters to barrels.

    Args:
        volume_m3: Volume in cubic meters

    Returns:
        Volume in barrels

    Example:
        >>> m3_to_bbl(1.0)
        6.28981
    """
    return volume_m3 / const.BBL_TO_M3


def gallon_to_m3(volume_gal: Numeric) -> Numeric:
    """Convert volume from US gallons to cubic meters.

    Args:
        volume_gal: Volume in US gallons

    Returns:
        Volume in cubic meters

    Example:
        >>> gallon_to_m3(264.172)
        1.0
    """
    return volume_gal * const.GAL_TO_M3


def m3_to_gallon(volume_m3: Numeric) -> Numeric:
    """Convert volume from cubic meters to US gallons.

    Args:
        volume_m3: Volume in cubic meters

    Returns:
        Volume in US gallons

    Example:
        >>> m3_to_gallon(1.0)
        264.172
    """
    return volume_m3 / const.GAL_TO_M3


# Molar volume conversions
def cm3_per_mol_to_m3_per_mol(volume: Numeric) -> Numeric:
    """Convert molar volume from cm³/mol to m³/mol."""
    return volume * 1e-6


def m3_per_mol_to_cm3_per_mol(volume: Numeric) -> Numeric:
    """Convert molar volume from m³/mol to cm³/mol."""
    return volume * 1e6


def liter_per_mol_to_m3_per_mol(volume: Numeric) -> Numeric:
    """Convert molar volume from L/mol to m³/mol."""
    return volume * 1e-3


def m3_per_mol_to_liter_per_mol(volume: Numeric) -> Numeric:
    """Convert molar volume from m³/mol to L/mol."""
    return volume * 1e3


# ============================================================================
# MASS AND MOLAR CONVERSIONS
# ============================================================================

def lb_to_kg(mass_lb: Numeric) -> Numeric:
    """Convert mass from pounds to kilograms.

    Args:
        mass_lb: Mass in pounds

    Returns:
        Mass in kilograms

    Example:
        >>> lb_to_kg(2.20462)
        1.0
    """
    return mass_lb * const.LB_TO_KG


def kg_to_lb(mass_kg: Numeric) -> Numeric:
    """Convert mass from kilograms to pounds.

    Args:
        mass_kg: Mass in kilograms

    Returns:
        Mass in pounds

    Example:
        >>> kg_to_lb(1.0)
        2.20462
    """
    return mass_kg / const.LB_TO_KG


def lbmol_to_mol(moles_lbmol: Numeric) -> Numeric:
    """Convert moles from lbmol to mol.

    Args:
        moles_lbmol: Moles in lbmol

    Returns:
        Moles in mol

    Example:
        >>> lbmol_to_mol(1.0)
        453.59237
    """
    return moles_lbmol * const.LBMOL_TO_MOL


def mol_to_lbmol(moles_mol: Numeric) -> Numeric:
    """Convert moles from mol to lbmol.

    Args:
        moles_mol: Moles in mol

    Returns:
        Moles in lbmol

    Example:
        >>> mol_to_lbmol(453.59237)
        1.0
    """
    return moles_mol / const.LBMOL_TO_MOL


# ============================================================================
# ENERGY CONVERSIONS
# ============================================================================

def cal_to_joule(energy_cal: Numeric) -> Numeric:
    """Convert energy from calories to joules."""
    return energy_cal * const.CAL_TO_J


def joule_to_cal(energy_j: Numeric) -> Numeric:
    """Convert energy from joules to calories."""
    return energy_j / const.CAL_TO_J


def btu_to_joule(energy_btu: Numeric) -> Numeric:
    """Convert energy from BTU to joules."""
    return energy_btu * const.BTU_TO_J


def joule_to_btu(energy_j: Numeric) -> Numeric:
    """Convert energy from joules to BTU."""
    return energy_j / const.BTU_TO_J


def kwh_to_joule(energy_kwh: Numeric) -> Numeric:
    """Convert energy from kWh to joules."""
    return energy_kwh * const.KWH_TO_J


def joule_to_kwh(energy_j: Numeric) -> Numeric:
    """Convert energy from joules to kWh."""
    return energy_j / const.KWH_TO_J


# ============================================================================
# DENSITY CONVERSIONS
# ============================================================================

def sg_to_density(specific_gravity: Numeric, reference_density: float = 1000.0) -> Numeric:
    """Convert specific gravity to density.

    Args:
        specific_gravity: Specific gravity (dimensionless)
        reference_density: Reference density in kg/m³ (default: 1000 kg/m³ for water)

    Returns:
        Density in kg/m³

    Example:
        >>> sg_to_density(0.85)  # Light oil
        850.0
    """
    return specific_gravity * reference_density


def density_to_sg(density: Numeric, reference_density: float = 1000.0) -> Numeric:
    """Convert density to specific gravity.

    Args:
        density: Density in kg/m³
        reference_density: Reference density in kg/m³ (default: 1000 kg/m³ for water)

    Returns:
        Specific gravity (dimensionless)

    Example:
        >>> density_to_sg(850.0)  # Light oil
        0.85
    """
    return density / reference_density


def api_to_sg(api_gravity: Numeric) -> Numeric:
    """Convert API gravity to specific gravity.

    Args:
        api_gravity: API gravity in degrees

    Returns:
        Specific gravity (dimensionless, 60°F/60°F basis)

    Example:
        >>> api_to_sg(35.0)  # Medium crude oil
        0.8498
    """
    return 141.5 / (131.5 + api_gravity)


def sg_to_api(specific_gravity: Numeric) -> Numeric:
    """Convert specific gravity to API gravity.

    Args:
        specific_gravity: Specific gravity (dimensionless, 60°F/60°F basis)

    Returns:
        API gravity in degrees

    Example:
        >>> sg_to_api(0.8498)  # Medium crude oil
        35.0
    """
    return 141.5 / specific_gravity - 131.5

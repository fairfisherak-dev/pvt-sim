"""Physical and chemical constants for PVT calculations.

This module provides fundamental constants used in thermodynamic calculations,
including the gas constant in various units and standard conditions.
"""

from typing import NamedTuple


class GasConstant(NamedTuple):
    """Universal gas constant in various units.

    Attributes:
        J_per_mol_K: Gas constant in J/(mol·K)
        kJ_per_kmol_K: Gas constant in kJ/(kmol·K)
        Pa_m3_per_mol_K: Gas constant in Pa·m³/(mol·K)
        bar_L_per_mol_K: Gas constant in bar·L/(mol·K)
        psia_ft3_per_lbmol_R: Gas constant in psia·ft³/(lbmol·R)
        atm_L_per_mol_K: Gas constant in atm·L/(mol·K)
        cal_per_mol_K: Gas constant in cal/(mol·K)
        BTU_per_lbmol_R: Gas constant in BTU/(lbmol·R)
    """
    J_per_mol_K: float
    kJ_per_kmol_K: float
    Pa_m3_per_mol_K: float
    bar_L_per_mol_K: float
    psia_ft3_per_lbmol_R: float
    atm_L_per_mol_K: float
    cal_per_mol_K: float
    BTU_per_lbmol_R: float


class StandardConditions(NamedTuple):
    """Standard conditions definition.

    Attributes:
        T: Temperature (K)
        P: Pressure (Pa)
        name: Name of the standard condition
        description: Description of the standard
    """
    T: float
    P: float
    name: str
    description: str


# Universal Gas Constant
# CODATA 2018 recommended value: 8.31446261815324 J/(mol·K)
R = GasConstant(
    J_per_mol_K=8.31446261815324,
    kJ_per_kmol_K=8.31446261815324,  # Same numerical value
    Pa_m3_per_mol_K=8.31446261815324,  # Same as J/(mol·K)
    bar_L_per_mol_K=0.0831446261815324,
    psia_ft3_per_lbmol_R=10.7316,
    atm_L_per_mol_K=0.08205746,
    cal_per_mol_K=1.98720425864083,
    BTU_per_lbmol_R=1.98588
)

# Standard Temperature and Pressure (STP) - IUPAC definition
STP = StandardConditions(
    T=273.15,  # 0°C
    P=101325,  # 1 atm = 101325 Pa
    name="STP",
    description="Standard Temperature and Pressure (IUPAC): 273.15 K, 101325 Pa"
)

# Normal Temperature and Pressure (NTP)
NTP = StandardConditions(
    T=293.15,  # 20°C
    P=101325,  # 1 atm = 101325 Pa
    name="NTP",
    description="Normal Temperature and Pressure: 293.15 K, 101325 Pa"
)

# Standard Conditions (SC) - Common in petroleum engineering
# 60°F and 14.696 psia
SC_IMPERIAL = StandardConditions(
    T=288.71,  # 60°F = 15.56°C
    P=101325,  # 14.696 psia ≈ 101325 Pa
    name="SC",
    description="Standard Conditions (Petroleum): 60°F (288.71 K), 14.696 psia"
)

# Standard Conditions - Metric (15°C and 101325 Pa)
SC_METRIC = StandardConditions(
    T=288.15,  # 15°C
    P=101325,  # 1 atm = 101325 Pa
    name="SC_METRIC",
    description="Standard Conditions (Metric): 15°C (288.15 K), 101325 Pa"
)

# Standard ambient temperature and pressure (SATP)
SATP = StandardConditions(
    T=298.15,  # 25°C
    P=101325,  # 1 bar = 100000 Pa (Note: SATP often uses 1 bar, but 1 atm is also common)
    name="SATP",
    description="Standard Ambient Temperature and Pressure: 298.15 K, 101325 Pa"
)

# Physical Constants
AVOGADRO_NUMBER = 6.02214076e23  # molecules/mol (CODATA 2018)
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K (CODATA 2018)
PLANCK_CONSTANT = 6.62607015e-34  # J·s (CODATA 2018)

# Unit conversion factors
ATM_TO_PA = 101325.0  # 1 atm = 101325 Pa
BAR_TO_PA = 100000.0  # 1 bar = 100000 Pa
PSI_TO_PA = 6894.757293168  # 1 psi = 6894.757293168 Pa
TORR_TO_PA = 133.322387415  # 1 Torr = 133.322387415 Pa
MMHG_TO_PA = 133.322387415  # 1 mmHg = 133.322387415 Pa

# Temperature conversion constants
ZERO_CELSIUS_IN_KELVIN = 273.15  # 0°C = 273.15 K
ZERO_FAHRENHEIT_IN_CELSIUS = -17.777777777777779  # 0°F = -17.78°C
FAHRENHEIT_TO_CELSIUS_SLOPE = 5.0 / 9.0
CELSIUS_TO_FAHRENHEIT_SLOPE = 9.0 / 5.0
RANKINE_TO_KELVIN_FACTOR = 5.0 / 9.0

# Volume conversion factors
L_TO_M3 = 0.001  # 1 L = 0.001 m³
FT3_TO_M3 = 0.028316846592  # 1 ft³ = 0.028316846592 m³
BBL_TO_M3 = 0.158987294928  # 1 barrel = 0.158987294928 m³
GAL_TO_M3 = 0.003785411784  # 1 US gallon = 0.003785411784 m³

# Mass conversion factors
LB_TO_KG = 0.45359237  # 1 lb = 0.45359237 kg
LBMOL_TO_MOL = 453.59237  # 1 lbmol = 453.59237 mol

# Energy conversion factors
CAL_TO_J = 4.184  # 1 cal = 4.184 J
BTU_TO_J = 1055.05585262  # 1 BTU = 1055.05585262 J
KWH_TO_J = 3600000.0  # 1 kWh = 3.6 MJ

# Gravity
STANDARD_GRAVITY = 9.80665  # m/s² (standard gravity)

# Molecular weight of air (dry air at sea level)
MW_AIR = 28.9647  # g/mol

# Triple point of water
WATER_TRIPLE_POINT_T = 273.16  # K (exact definition of Kelvin before 2019)
WATER_TRIPLE_POINT_P = 611.657  # Pa


# Dictionary of all standard conditions for easy access
STANDARD_CONDITIONS = {
    'STP': STP,
    'NTP': NTP,
    'SC_IMPERIAL': SC_IMPERIAL,
    'SC_METRIC': SC_METRIC,
    'SATP': SATP,
}


def get_standard_condition(name: str) -> StandardConditions:
    """Get standard conditions by name.

    Args:
        name: Name of standard condition ('STP', 'NTP', 'SC_IMPERIAL', 'SC_METRIC', 'SATP')

    Returns:
        StandardConditions object

    Raises:
        KeyError: If standard condition name is not recognized

    Example:
        >>> sc = get_standard_condition('STP')
        >>> print(sc.T, sc.P)
        273.15 101325
    """
    if name not in STANDARD_CONDITIONS:
        available = ', '.join(STANDARD_CONDITIONS.keys())
        raise KeyError(
            f"Standard condition '{name}' not recognized. "
            f"Available: {available}"
        )
    return STANDARD_CONDITIONS[name]

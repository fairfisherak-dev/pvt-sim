"""
Binary Interaction Parameter (BIP) correlations for EOS mixing rules.

BIPs (kij) are used in the van der Waals one-fluid mixing rules to
account for unlike-pair interactions:
    a_mix = sum_i sum_j x_i x_j sqrt(a_i a_j) (1 - k_ij)

This module provides:
- Generalized correlations based on critical temperatures
- Default values for common non-hydrocarbon/hydrocarbon pairs
- Methods to build full BIP matrices

Units Convention:
- Temperature: Kelvin (K)
- BIP: dimensionless

References
----------
[1] Chueh, P.L. and Prausnitz, J.M. (1967). "Vapor-Liquid Equilibria at
    High Pressures: Calculation of Partial Molar Volumes in Nonpolar
    Liquid Mixtures." AIChE Journal, 13(6), 1099-1107.
[2] Peng, D.Y. and Robinson, D.B. (1976). "A New Two-Constant Equation
    of State." Ind. Eng. Chem. Fundam., 15(1), 59-64.
[3] Pedersen, K.S., Christensen, P.L., and Shaikh, J.A. (2015). "Phase
    Behavior of Petroleum Reservoir Fluids." 2nd ed., CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class BIPMethod(Enum):
    """Enumeration of BIP correlation methods."""
    ZERO = auto()           # All kij = 0 (ideal mixing)
    CHUEH_PRAUSNITZ = auto()  # Tc-based correlation
    DEFAULT_VALUES = auto()   # Use tabulated defaults + Tc correlation
    PPR78 = auto()          # Temperature-dependent PPR78 group contribution


# =============================================================================
# Default BIP Values for Non-Hydrocarbon Pairs
# =============================================================================

# Default kij values for common non-HC / HC pairs
# Format: (component1, component2): kij
# Note: kij = kji (symmetric)
DEFAULT_BIPS: Dict[Tuple[str, str], float] = {
    # N2 - hydrocarbons (typical values)
    ("N2", "C1"): 0.025,
    ("N2", "C2"): 0.010,
    ("N2", "C3"): 0.090,
    ("N2", "iC4"): 0.095,
    ("N2", "nC4"): 0.095,
    ("N2", "C4"): 0.095,
    ("N2", "iC5"): 0.100,
    ("N2", "nC5"): 0.100,
    ("N2", "C5"): 0.100,
    ("N2", "C6"): 0.110,
    ("N2", "C7"): 0.115,

    # CO2 - hydrocarbons (typical values)
    ("CO2", "C1"): 0.105,
    ("CO2", "C2"): 0.130,
    ("CO2", "C3"): 0.125,
    ("CO2", "iC4"): 0.120,
    ("CO2", "nC4"): 0.115,
    ("CO2", "C4"): 0.115,
    ("CO2", "iC5"): 0.115,
    ("CO2", "nC5"): 0.115,
    ("CO2", "C5"): 0.115,
    ("CO2", "C6"): 0.115,
    ("CO2", "C7"): 0.115,

    # H2S - hydrocarbons (typical values)
    ("H2S", "C1"): 0.080,
    ("H2S", "C2"): 0.070,
    ("H2S", "C3"): 0.070,
    ("H2S", "iC4"): 0.060,
    ("H2S", "nC4"): 0.060,
    ("H2S", "C4"): 0.060,
    ("H2S", "iC5"): 0.050,
    ("H2S", "nC5"): 0.050,
    ("H2S", "C5"): 0.050,
    ("H2S", "C6"): 0.050,
    ("H2S", "C7"): 0.050,

    # Non-HC pairs
    ("N2", "CO2"): -0.020,
    ("N2", "H2S"): 0.170,
    ("CO2", "H2S"): 0.100,
}


def get_default_bip(comp1: str, comp2: str) -> Optional[float]:
    """
    Get default BIP for a pair of components.

    Parameters
    ----------
    comp1, comp2 : str
        Component identifiers (e.g., "N2", "C1", "CO2").

    Returns
    -------
    float or None
        Default kij value, or None if not found.
    """
    # Normalize component names
    comp1 = comp1.upper().replace("NC", "nC").replace("IC", "iC")
    comp2 = comp2.upper().replace("NC", "nC").replace("IC", "iC")

    # Try both orderings
    if (comp1, comp2) in DEFAULT_BIPS:
        return DEFAULT_BIPS[(comp1, comp2)]
    if (comp2, comp1) in DEFAULT_BIPS:
        return DEFAULT_BIPS[(comp2, comp1)]

    return None


# =============================================================================
# Generalized BIP Correlations
# =============================================================================

def chueh_prausnitz_kij(
    Tc_i: float,
    Tc_j: float,
    A: float = 0.0,
    B: float = 1.0,
) -> float:
    """
    Calculate BIP using Chueh-Prausnitz (1967) correlation.

    The correlation is:
        kij = A * (1 - (2 * sqrt(Tc_i * Tc_j) / (Tc_i + Tc_j))^B)

    For B=1, this simplifies to:
        kij = A * (1 - 2*sqrt(Tc_i*Tc_j)/(Tc_i + Tc_j))

    Parameters
    ----------
    Tc_i, Tc_j : float
        Critical temperatures in Kelvin.
    A : float
        Coefficient (default 0.0 gives kij=0 for hydrocarbons).
    B : float
        Exponent (default 1.0).

    Returns
    -------
    float
        Binary interaction parameter (dimensionless).

    Notes
    -----
    For hydrocarbon/hydrocarbon pairs, A is typically 0 to 0.02.
    For N2/HC pairs, A ~ 0.5 gives reasonable results.
    For CO2/HC pairs, A ~ 0.15 is typical.
    """
    if Tc_i <= 0 or Tc_j <= 0:
        raise ValueError("Critical temperatures must be positive")

    # Avoid division by zero
    Tc_sum = Tc_i + Tc_j
    if Tc_sum < 1e-10:
        return 0.0

    ratio = 2.0 * np.sqrt(Tc_i * Tc_j) / Tc_sum
    kij = A * (1.0 - ratio ** B)

    return kij


def estimate_hc_hc_kij(
    Tc_i: float,
    Tc_j: float,
) -> float:
    """
    Estimate BIP for hydrocarbon/hydrocarbon pairs.

    Uses a correlation based on critical temperature difference:
        kij = A * (1 - (2*sqrt(Tci*Tcj)/(Tci+Tcj))^B)

    with A=0.01 and B=3.0, which gives small positive kij values
    for dissimilar hydrocarbons and near-zero for similar ones.

    Parameters
    ----------
    Tc_i, Tc_j : float
        Critical temperatures in Kelvin.

    Returns
    -------
    float
        Estimated kij (typically 0 to 0.02 for HC/HC pairs).
    """
    return chueh_prausnitz_kij(Tc_i, Tc_j, A=0.01, B=3.0)


def estimate_n2_hc_kij(
    Tc_hc: float,
) -> float:
    """
    Estimate BIP for N2/hydrocarbon pair.

    Uses a correlation that accounts for N2's low Tc relative to HCs.

    Parameters
    ----------
    Tc_hc : float
        Critical temperature of hydrocarbon in Kelvin.

    Returns
    -------
    float
        Estimated kij (typically 0.02 to 0.15).
    """
    Tc_N2 = 126.19  # K
    # N2/HC kij increases with HC size (Tc)
    # Approximate correlation from Pedersen (2015)
    kij = 0.025 + 0.0003 * (Tc_hc - 190.0)  # Relative to methane Tc
    kij = max(0.02, min(0.15, kij))  # Clamp to reasonable range
    return kij


def estimate_co2_hc_kij(
    Tc_hc: float,
) -> float:
    """
    Estimate BIP for CO2/hydrocarbon pair.

    Uses a correlation based on HC critical temperature.

    Parameters
    ----------
    Tc_hc : float
        Critical temperature of hydrocarbon in Kelvin.

    Returns
    -------
    float
        Estimated kij (typically 0.10 to 0.15).
    """
    Tc_CO2 = 304.18  # K
    # CO2/HC kij varies with HC type
    # Light HCs: kij ~ 0.10-0.13
    # Heavy HCs: kij ~ 0.11-0.15
    kij = 0.10 + 0.0001 * (Tc_hc - 190.0)
    kij = max(0.10, min(0.15, kij))
    return kij


def estimate_h2s_hc_kij(
    Tc_hc: float,
) -> float:
    """
    Estimate BIP for H2S/hydrocarbon pair.

    Parameters
    ----------
    Tc_hc : float
        Critical temperature of hydrocarbon in Kelvin.

    Returns
    -------
    float
        Estimated kij (typically 0.05 to 0.08).
    """
    # H2S/HC kij is relatively constant
    kij = 0.07 - 0.00005 * (Tc_hc - 190.0)  # Slight decrease with MW
    kij = max(0.05, min(0.10, kij))
    return kij


# =============================================================================
# BIP Matrix Builder
# =============================================================================

@dataclass(frozen=True)
class BIPMatrix:
    """
    Container for binary interaction parameter matrix.

    Attributes
    ----------
    kij : ndarray
        Symmetric BIP matrix of shape (n, n).
    component_ids : list of str
        Component identifiers.
    method : BIPMethod
        Method used to generate the matrix.
    """
    kij: NDArray[np.float64]
    component_ids: List[str]
    method: BIPMethod

    def get_kij(self, i: int, j: int) -> float:
        """Get BIP for component pair by index."""
        return self.kij[i, j]

    def get_kij_by_name(self, name_i: str, name_j: str) -> float:
        """Get BIP for component pair by name."""
        i = self.component_ids.index(name_i)
        j = self.component_ids.index(name_j)
        return self.kij[i, j]


def build_bip_matrix(
    *,
    component_ids: List[str],
    Tc: NDArray[np.float64],
    method: BIPMethod = BIPMethod.DEFAULT_VALUES,
    custom_bips: Optional[Dict[Tuple[str, str], float]] = None,
    temperature: Optional[float] = None,
    groups: Optional[List[Optional[Dict[str, int]]]] = None,
) -> BIPMatrix:
    """
    Build a BIP matrix for a set of components.

    Parameters
    ----------
    component_ids : list of str
        Component identifiers (e.g., ["N2", "CO2", "C1", "C2", ...]).
    Tc : ndarray
        Critical temperatures in Kelvin.
    method : BIPMethod
        Method for generating BIPs.
    custom_bips : dict, optional
        Custom BIP values to override defaults. Keys are (comp1, comp2) tuples.
    temperature : float, optional
        Temperature in Kelvin. Required for PPR78 method.
    groups : list of dict, optional
        PPR78 group decompositions for each component. Each element is a dict
        like {"CH3": 2, "CH2": 4} or None (use built-in lookup).
        Required for PPR78 method if components are not in built-in database.

    Returns
    -------
    BIPMatrix
        Symmetric BIP matrix.

    Examples
    --------
    >>> bips = build_bip_matrix(
    ...     component_ids=["N2", "CO2", "C1", "C2", "C3"],
    ...     Tc=np.array([126.19, 304.18, 190.6, 305.3, 369.9]),
    ...     method=BIPMethod.DEFAULT_VALUES,
    ... )
    >>> print(bips.kij)

    For PPR78 temperature-dependent BIPs:
    >>> bips = build_bip_matrix(
    ...     component_ids=["C1", "CO2"],
    ...     Tc=np.array([190.6, 304.18]),
    ...     method=BIPMethod.PPR78,
    ...     temperature=300.0,
    ... )
    """
    component_ids = [str(c) for c in component_ids]
    Tc = np.asarray(Tc, dtype=np.float64)
    n = len(component_ids)

    if len(Tc) != n:
        raise ValueError(f"Tc length ({len(Tc)}) must match component_ids ({n})")

    kij = np.zeros((n, n), dtype=np.float64)

    if method == BIPMethod.ZERO:
        # All zeros - ideal mixing
        pass

    elif method == BIPMethod.CHUEH_PRAUSNITZ:
        # Pure correlation-based
        for i in range(n):
            for j in range(i + 1, n):
                kij[i, j] = chueh_prausnitz_kij(Tc[i], Tc[j], A=0.01, B=3.0)
                kij[j, i] = kij[i, j]

    elif method == BIPMethod.DEFAULT_VALUES:
        # Use defaults where available, correlations otherwise
        for i in range(n):
            for j in range(i + 1, n):
                comp_i = component_ids[i]
                comp_j = component_ids[j]

                # Check for default value
                default = get_default_bip(comp_i, comp_j)
                if default is not None:
                    kij[i, j] = default
                else:
                    # Identify component types and use appropriate correlation
                    kij[i, j] = _estimate_kij_by_type(comp_i, comp_j, Tc[i], Tc[j])

                kij[j, i] = kij[i, j]

    elif method == BIPMethod.PPR78:
        # Temperature-dependent PPR78 group contribution method
        if temperature is None:
            raise ValueError("PPR78 method requires 'temperature' parameter")

        # Import here to avoid circular imports
        from ..eos.ppr78 import PPR78Calculator

        calc = PPR78Calculator(use_rdkit=False)

        # Register all components
        for i, comp_id in enumerate(component_ids):
            comp_groups = groups[i] if groups and i < len(groups) else None
            calc.register_component(
                component_id=comp_id,
                groups=comp_groups,
            )

        # Get k_ij matrix at the specified temperature
        kij = calc.get_kij_matrix(temperature)

    # Apply custom overrides
    if custom_bips:
        for (comp1, comp2), value in custom_bips.items():
            try:
                i = component_ids.index(comp1)
                j = component_ids.index(comp2)
                kij[i, j] = value
                kij[j, i] = value
            except ValueError:
                pass  # Component not in list

    return BIPMatrix(kij=kij, component_ids=component_ids, method=method)


def _estimate_kij_by_type(
    comp1: str,
    comp2: str,
    Tc1: float,
    Tc2: float,
) -> float:
    """
    Estimate kij based on component types.

    Uses component identifiers to determine type (N2, CO2, H2S, or HC)
    and applies appropriate correlation.
    """
    comp1_upper = comp1.upper()
    comp2_upper = comp2.upper()

    is_n2_1 = comp1_upper == "N2"
    is_n2_2 = comp2_upper == "N2"
    is_co2_1 = comp1_upper == "CO2"
    is_co2_2 = comp2_upper == "CO2"
    is_h2s_1 = comp1_upper == "H2S"
    is_h2s_2 = comp2_upper == "H2S"

    # Check for non-HC / non-HC pairs
    if is_n2_1 and is_co2_2:
        return -0.020
    if is_n2_2 and is_co2_1:
        return -0.020
    if is_n2_1 and is_h2s_2:
        return 0.170
    if is_n2_2 and is_h2s_1:
        return 0.170
    if is_co2_1 and is_h2s_2:
        return 0.100
    if is_co2_2 and is_h2s_1:
        return 0.100

    # Non-HC / HC pairs
    if is_n2_1 or is_n2_2:
        Tc_hc = Tc2 if is_n2_1 else Tc1
        return estimate_n2_hc_kij(Tc_hc)

    if is_co2_1 or is_co2_2:
        Tc_hc = Tc2 if is_co2_1 else Tc1
        return estimate_co2_hc_kij(Tc_hc)

    if is_h2s_1 or is_h2s_2:
        Tc_hc = Tc2 if is_h2s_1 else Tc1
        return estimate_h2s_hc_kij(Tc_hc)

    # HC / HC pair
    return estimate_hc_hc_kij(Tc1, Tc2)


# =============================================================================
# BIP Tuning Support
# =============================================================================

def scale_c7plus_bips(
    kij: NDArray[np.float64],
    component_ids: List[str],
    scale_factor: float,
    c7plus_indices: Optional[List[int]] = None,
) -> NDArray[np.float64]:
    """
    Scale BIPs involving C7+ components.

    This is useful for EOS tuning, where C7+ BIPs are often adjusted
    to match experimental saturation pressure.

    Parameters
    ----------
    kij : ndarray
        Original BIP matrix.
    component_ids : list of str
        Component identifiers.
    scale_factor : float
        Multiplier for C7+ BIPs (e.g., 1.1 for 10% increase).
    c7plus_indices : list of int, optional
        Indices of C7+ components. If None, auto-detected from names.

    Returns
    -------
    ndarray
        Scaled BIP matrix.
    """
    kij = kij.copy()
    n = len(component_ids)

    if c7plus_indices is None:
        # Auto-detect C7+ components (C7, C8, ..., C45, etc.)
        c7plus_indices = []
        for i, name in enumerate(component_ids):
            # Match C7, C8, ..., C99, or any SCN > 6
            name_upper = name.upper()
            if name_upper.startswith("C") and len(name_upper) > 1:
                try:
                    cn = int(name_upper[1:])
                    if cn >= 7:
                        c7plus_indices.append(i)
                except ValueError:
                    pass

    # Scale BIPs involving any C7+ component
    for i in range(n):
        for j in range(n):
            if i in c7plus_indices or j in c7plus_indices:
                if i != j:  # Don't scale diagonal (always 0)
                    kij[i, j] *= scale_factor

    return kij

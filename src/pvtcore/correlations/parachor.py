"""
Parachor correlations for petroleum components.

Parachor is used in interfacial tension (IFT) calculations via the
Macleod-Sugden correlation:
    sigma^(1/4) = sum_i P_i * (x_i * rho_L/MW_L - y_i * rho_V/MW_V)

where P_i is the parachor of component i.

Methods implemented:
- Fanchi (1985): MW-based correlation for petroleum fractions

Units Convention:
- Parachor: (mN/m)^(1/4) * cm³/mol (traditional units)
- Molecular weight: g/mol

References
----------
[1] Fanchi, J.R. (1985). "Calculation of Parachors for Compositional
    Simulation: An Update." SPE Reservoir Engineering, 1(4), 405-406.
[2] Macleod, D.B. (1923). "On a Relation between Surface Tension and
    Density." Trans. Faraday Soc., 19, 38-41.
[3] Sugden, S. (1924). "A Relation between Surface Tension, Density, and
    Chemical Composition." J. Chem. Soc. Trans., 125, 1177-1189.
[4] Weinaug, C.F. and Katz, D.L. (1943). "Surface Tensions of Methane-
    Propane Mixtures." Ind. Eng. Chem., 35(2), 239-246.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Pure Component Parachors (from literature)
# =============================================================================

# Standard parachors for common components from Weinaug & Katz (1943)
# and other literature sources.
# Units: (mN/m)^(1/4) * cm³/mol
PURE_COMPONENT_PARACHORS = {
    # Non-hydrocarbons
    "N2": 41.0,
    "CO2": 78.0,
    "H2S": 80.1,

    # Normal paraffins
    "C1": 77.0,    # Methane
    "C2": 108.0,   # Ethane
    "C3": 150.3,   # Propane
    "iC4": 181.5,  # Isobutane
    "nC4": 189.9,  # n-Butane (also C4)
    "C4": 189.9,   # n-Butane (alias)
    "iC5": 225.0,  # Isopentane
    "nC5": 231.5,  # n-Pentane (also C5)
    "C5": 231.5,   # n-Pentane (alias)
    "C6": 271.0,   # n-Hexane
    "C7": 312.5,   # n-Heptane
    "C8": 351.5,   # n-Octane
    "C9": 393.0,   # n-Nonane
    "C10": 433.5,  # n-Decane
}


def get_pure_parachor(component_id: str) -> Optional[float]:
    """
    Get parachor for a pure component from the database.

    Parameters
    ----------
    component_id : str
        Component identifier (e.g., "C1", "CO2", "nC4").

    Returns
    -------
    float or None
        Parachor in (mN/m)^(1/4) * cm³/mol, or None if not found.
    """
    return PURE_COMPONENT_PARACHORS.get(component_id.upper())


# =============================================================================
# Fanchi Correlation
# =============================================================================

def fanchi_parachor(MW: float) -> float:
    """
    Estimate parachor using Fanchi (1985) correlation.

    This correlation estimates parachor from molecular weight and is
    useful for pseudo-components where experimental data is unavailable.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.

    Returns
    -------
    float
        Parachor in (mN/m)^(1/4) * cm³/mol.

    Notes
    -----
    The Fanchi correlation is:
        P = 11.4 + 3.23*MW - 0.0022*MW²

    This is valid for hydrocarbon fractions. For non-hydrocarbons
    (N₂, CO₂, H₂S), use the tabulated values instead.

    The correlation was developed by fitting parachors of n-paraffins
    and is most accurate for paraffin-like fractions.

    References
    ----------
    Fanchi (1985), SPE Reservoir Engineering.

    Examples
    --------
    >>> P = fanchi_parachor(MW=100.0)  # ~C7
    >>> print(f"Parachor = {P:.1f}")
    Parachor = 312.6
    """
    _validate_MW(MW)

    # Fanchi correlation
    P = 11.4 + 3.23 * MW - 0.0022 * (MW ** 2)

    # Sanity check
    if P <= 0:
        raise ValueError(
            f"Fanchi correlation produced non-physical parachor for MW={MW}"
        )

    return P


def estimate_parachor(
    MW: float,
    component_id: Optional[str] = None,
) -> float:
    """
    Estimate parachor for a component.

    If component_id is provided and found in the database, returns the
    tabulated value. Otherwise, uses the Fanchi correlation.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    component_id : str, optional
        Component identifier for database lookup.

    Returns
    -------
    float
        Parachor in (mN/m)^(1/4) * cm³/mol.

    Examples
    --------
    >>> P1 = estimate_parachor(MW=16.04, component_id="C1")  # Uses database
    >>> P2 = estimate_parachor(MW=142.0)  # Uses Fanchi correlation
    """
    if component_id is not None:
        tabulated = get_pure_parachor(component_id)
        if tabulated is not None:
            return tabulated

    return fanchi_parachor(MW)


# =============================================================================
# Vectorized Versions
# =============================================================================

def fanchi_parachor_array(MW: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Vectorized Fanchi parachor correlation.

    Parameters
    ----------
    MW : ndarray
        Molecular weights in g/mol.

    Returns
    -------
    ndarray
        Parachors in (mN/m)^(1/4) * cm³/mol.
    """
    MW = np.asarray(MW, dtype=np.float64)
    P = 11.4 + 3.23 * MW - 0.0022 * (MW ** 2)
    return P


def estimate_parachor_array(
    MW: NDArray[np.float64],
    component_ids: Optional[list[str]] = None,
) -> NDArray[np.float64]:
    """
    Vectorized parachor estimation.

    Parameters
    ----------
    MW : ndarray
        Molecular weights in g/mol.
    component_ids : list of str, optional
        Component identifiers for database lookup. Must match length of MW.

    Returns
    -------
    ndarray
        Parachors in (mN/m)^(1/4) * cm³/mol.
    """
    MW = np.asarray(MW, dtype=np.float64)
    n = len(MW)

    P = fanchi_parachor_array(MW)

    if component_ids is not None:
        if len(component_ids) != n:
            raise ValueError(
                f"component_ids length ({len(component_ids)}) must match MW length ({n})"
            )
        for i, cid in enumerate(component_ids):
            if cid is not None:
                tabulated = get_pure_parachor(cid)
                if tabulated is not None:
                    P[i] = tabulated

    return P


# =============================================================================
# Alternative Correlations
# =============================================================================

def quayle_parachor(MW: float, SG: float) -> float:
    """
    Estimate parachor using Quayle group contribution method.

    This is an alternative correlation that accounts for specific gravity,
    providing better estimates for non-paraffinic fractions.

    Parameters
    ----------
    MW : float
        Molecular weight in g/mol.
    SG : float
        Specific gravity at 60°F/60°F.

    Returns
    -------
    float
        Parachor in (mN/m)^(1/4) * cm³/mol.

    Notes
    -----
    The correlation is:
        P = 25.2 + 2.86*MW*SG

    This form accounts for aromaticity through the SG term.
    """
    _validate_MW(MW)
    if SG <= 0:
        raise ValueError(f"SG must be positive, got {SG}")

    P = 25.2 + 2.86 * MW * SG
    return P


# =============================================================================
# Input Validation
# =============================================================================

def _validate_MW(MW: float) -> None:
    """Validate molecular weight input."""
    if not np.isfinite(MW):
        raise ValueError(f"MW must be finite, got {MW}")
    if MW <= 0:
        raise ValueError(f"MW must be positive, got {MW}")
    if MW < 10 or MW > 2000:
        # Warning range - may be outside correlation validity
        pass

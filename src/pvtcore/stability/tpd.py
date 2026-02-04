"""Tangent Plane Distance (TPD) function for phase stability analysis.

The tangent plane distance criterion, developed by Michelsen (1982), provides
a rigorous test for phase stability. A negative TPD indicates that the system
can lower its Gibbs free energy by splitting into two phases.

Reference:
Michelsen, M. L., "The Isothermal Flash Problem. Part I. Stability",
Fluid Phase Equilibria, 9(1), 1-19 (1982).
"""

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ..eos.base import CubicEOS
from ..core.errors import CompositionError


def calculate_tpd(
    trial_composition: NDArray[np.float64],
    feed_composition: NDArray[np.float64],
    feed_ln_fugacity_coef: NDArray[np.float64],
    eos: CubicEOS,
    pressure: float,
    temperature: float,
    phase: str = 'vapor',
    binary_interaction: Optional[NDArray[np.float64]] = None
) -> float:
    """Calculate Tangent Plane Distance for a trial composition.

    The TPD function is defined as:
        TPD(W) = Σ wᵢ[ln(wᵢ) + ln(φᵢ(W)) - dᵢ]

    where:
        wᵢ = trial composition (mole fraction)
        φᵢ(W) = fugacity coefficient of component i in trial phase
        dᵢ = ln(zᵢ) + ln(φᵢ(z)) = feed fugacity terms

    The mixture is unstable if min(TPD) < 0 over all possible trial compositions.

    Parameters
    ----------
    trial_composition : NDArray[np.float64]
        Trial composition W (mole fractions, sum to 1)
    feed_composition : NDArray[np.float64]
        Feed composition z (mole fractions, sum to 1)
    feed_ln_fugacity_coef : NDArray[np.float64]
        Natural log of feed fugacity coefficients ln(φᵢ(z))
    eos : CubicEOS
        Equation of state object
    pressure : float
        System pressure (Pa)
    temperature : float
        System temperature (K)
    phase : str, optional
        Phase for trial composition ('vapor' or 'liquid'), default 'vapor'
    binary_interaction : NDArray[np.float64], optional
        Binary interaction parameters kᵢⱼ (n×n matrix)

    Returns
    -------
    float
        Tangent plane distance value (dimensionless)

    Raises
    ------
    CompositionError
        If trial or feed composition is invalid

    Notes
    -----
    - TPD < 0: System is unstable, phase split will occur
    - TPD = 0: System is at phase boundary (saturation)
    - TPD > 0: Trial phase is metastable or unstable
    - Minimum TPD over all trial compositions determines overall stability

    Example
    -------
    >>> from pvtcore.eos import PengRobinsonEOS
    >>> from pvtcore.models import load_components
    >>> components = load_components()
    >>> eos = PengRobinsonEOS([components['C1'], components['C10']])
    >>> z = np.array([0.5, 0.5])
    >>> W = np.array([0.9, 0.1])  # Vapor-like trial
    >>> phi_z = eos.fugacity_coefficient(P, T, z, phase='liquid')
    >>> ln_phi_z = np.log(phi_z)
    >>> tpd = calculate_tpd(W, z, ln_phi_z, eos, P, T, phase='vapor')
    """
    # Validate compositions
    trial_composition = np.asarray(trial_composition, dtype=np.float64)
    feed_composition = np.asarray(feed_composition, dtype=np.float64)

    if len(trial_composition) != len(feed_composition):
        raise CompositionError(
            "Trial and feed compositions must have same length",
            composition={'trial_size': len(trial_composition), 'feed_size': len(feed_composition)}
        )

    if not np.isclose(trial_composition.sum(), 1.0, atol=1e-6):
        raise CompositionError(
            f"Trial composition must sum to 1.0, got {trial_composition.sum():.6f}",
            composition={'trial': trial_composition.tolist()}
        )

    if not np.isclose(feed_composition.sum(), 1.0, atol=1e-6):
        raise CompositionError(
            f"Feed composition must sum to 1.0, got {feed_composition.sum():.6f}",
            composition={'feed': feed_composition.tolist()}
        )

    # Calculate fugacity coefficients for trial composition
    trial_fugacity_coef = eos.fugacity_coefficient(
        pressure, temperature, trial_composition, phase, binary_interaction
    )
    ln_trial_fugacity_coef = np.log(trial_fugacity_coef)

    # Calculate d terms: dᵢ = ln(zᵢ) + ln(φᵢ(z))
    # Avoid log(0) by using small epsilon for zero mole fractions
    epsilon = 1e-100
    ln_feed_composition = np.log(np.maximum(feed_composition, epsilon))
    d_terms = ln_feed_composition + feed_ln_fugacity_coef

    # Calculate TPD: Σ wᵢ[ln(wᵢ) + ln(φᵢ(W)) - dᵢ]
    ln_trial_composition = np.log(np.maximum(trial_composition, epsilon))

    # Only include components with non-zero trial composition
    # (components with zero trial composition contribute zero to sum)
    mask = trial_composition > epsilon
    tpd_terms = np.zeros_like(trial_composition)
    tpd_terms[mask] = trial_composition[mask] * (
        ln_trial_composition[mask] + ln_trial_fugacity_coef[mask] - d_terms[mask]
    )

    tpd = np.sum(tpd_terms)

    return tpd


def calculate_d_terms(
    composition: NDArray[np.float64],
    eos: CubicEOS,
    pressure: float,
    temperature: float,
    phase: str,
    binary_interaction: Optional[NDArray[np.float64]] = None
) -> NDArray[np.float64]:
    """Calculate d terms for TPD function: dᵢ = ln(zᵢ) + ln(φᵢ(z)).

    The d terms represent the chemical potential reference state based on
    the feed composition. They are constant for a given feed and do not
    change during stability analysis iterations.

    Parameters
    ----------
    composition : NDArray[np.float64]
        Composition (mole fractions)
    eos : CubicEOS
        Equation of state object
    pressure : float
        Pressure (Pa)
    temperature : float
        Temperature (K)
    phase : str
        Phase for fugacity calculation ('liquid' or 'vapor')
    binary_interaction : NDArray[np.float64], optional
        Binary interaction parameters kᵢⱼ

    Returns
    -------
    NDArray[np.float64]
        Array of d terms, one per component

    Notes
    -----
    - d terms are calculated once per stability test
    - Used as reference for all trial compositions
    - Avoid log(0) by using small epsilon for zero mole fractions

    Example
    -------
    >>> d = calculate_d_terms(z, eos, P, T, phase='liquid')
    >>> # Use d for multiple TPD evaluations
    >>> tpd1 = calculate_tpd(W1, z, np.log(phi_z), eos, P, T)
    """
    composition = np.asarray(composition, dtype=np.float64)

    # Calculate fugacity coefficients
    fugacity_coef = eos.fugacity_coefficient(
        pressure, temperature, composition, phase, binary_interaction
    )

    # Calculate d terms: dᵢ = ln(zᵢ) + ln(φᵢ(z))
    epsilon = 1e-100
    ln_composition = np.log(np.maximum(composition, epsilon))
    ln_fugacity_coef = np.log(fugacity_coef)

    d_terms = ln_composition + ln_fugacity_coef

    return d_terms

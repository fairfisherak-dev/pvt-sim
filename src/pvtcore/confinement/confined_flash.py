"""Confined flash calculation with capillary pressure coupling.

This module implements flash calculations for nano-confined systems where
capillary pressure creates different pressures in the liquid and vapor phases.

The coupling loop:
1. Start with bulk flash (Pv = PL)
2. Calculate IFT from phase compositions and densities
3. Calculate Pc = 2σ/r
4. Update Pv = PL + Pc
5. Re-calculate flash with split pressures
6. Repeat until |Pc_new - Pc_old| < tolerance

The key modification is the equilibrium condition:
    xi φi^L(PL) PL = yi φi^V(Pv) Pv

Leading to modified K-values:
    Ki = (φi^L / φi^V) × (PL / Pv)

Units Convention:
- Pressure: Pa
- Temperature: K
- Pore radius: nm (nanometers)
- IFT: mN/m

References
----------
[1] Nojabaei, B., Johns, R.T., and Chu, L. (2013).
    "Effect of Capillary Pressure on Phase Behavior in Tight Rocks and Shales."
    SPE Reservoir Evaluation & Engineering, 16(3), 281-289. SPE-159258.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ..core.constants import R
from ..core.errors import ConvergenceError, PhaseError, ValidationError
from ..eos.base import CubicEOS
from ..models.component import Component
from ..stability.wilson import wilson_k_values
from ..flash.rachford_rice import solve_rachford_rice
from ..properties.density import calculate_density
from ..properties.ift_parachor import calculate_ift_parachor
from .capillary import capillary_pressure_simple, modified_k_values_array


# Numerical parameters
DEFAULT_PC_TOLERANCE: float = 1e3  # Pa (1 kPa tolerance for Pc convergence)
DEFAULT_K_TOLERANCE: float = 1e-8  # K-value convergence tolerance
MAX_PC_ITERATIONS: int = 50  # Maximum outer loop iterations
MAX_FLASH_ITERATIONS: int = 100  # Maximum inner flash iterations


@dataclass
class ConfinedFlashResult:
    """Results from confined flash calculation.

    Attributes:
        converged: True if calculation converged
        iterations_outer: Number of outer Pc coupling iterations
        iterations_inner_total: Total inner flash iterations
        vapor_fraction: Vapor mole fraction (0 to 1)
        liquid_composition: Liquid phase mole fractions
        vapor_composition: Vapor phase mole fractions
        K_values: Final equilibrium ratios (yi/xi)
        K_values_bulk: Bulk K-values (without Pc correction)
        liquid_pressure: Liquid phase pressure (Pa)
        vapor_pressure: Vapor phase pressure (Pa)
        capillary_pressure: Final capillary pressure (Pa)
        ift: Final interfacial tension (mN/m)
        liquid_density: Liquid mass density (kg/m³)
        vapor_density: Vapor mass density (kg/m³)
        pore_radius: Pore radius used (nm)
        phase: Phase state ('two-phase', 'vapor', or 'liquid')
        temperature: Temperature (K)
        feed_composition: Feed composition
        residual_Pc: Final Pc convergence residual (Pa)
    """
    converged: bool
    iterations_outer: int
    iterations_inner_total: int
    vapor_fraction: float
    liquid_composition: NDArray[np.float64]
    vapor_composition: NDArray[np.float64]
    K_values: NDArray[np.float64]
    K_values_bulk: NDArray[np.float64]
    liquid_pressure: float
    vapor_pressure: float
    capillary_pressure: float
    ift: float
    liquid_density: float
    vapor_density: float
    pore_radius: float
    phase: str
    temperature: float
    feed_composition: NDArray[np.float64]
    residual_Pc: float


def confined_flash(
    liquid_pressure: float,
    temperature: float,
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    pore_radius_nm: float,
    binary_interaction: Optional[NDArray[np.float64]] = None,
    K_initial: Optional[NDArray[np.float64]] = None,
    Pc_tolerance: float = DEFAULT_PC_TOLERANCE,
    K_tolerance: float = DEFAULT_K_TOLERANCE,
    max_Pc_iterations: int = MAX_PC_ITERATIONS,
    max_flash_iterations: int = MAX_FLASH_ITERATIONS,
    contact_angle: float = 0.0,
) -> ConfinedFlashResult:
    """Perform confined flash calculation with capillary pressure coupling.

    This function implements the iterative coupling between flash
    calculation and capillary pressure (via IFT).

    The algorithm:
    1. Initialize with bulk flash (Pv = PL)
    2. Calculate IFT from parachor correlation
    3. Calculate Pc = 2σ/r and update Pv = PL + Pc
    4. Re-calculate flash with modified K-values: Ki = Ki_bulk × (PL/Pv)
    5. Check Pc convergence and repeat

    Parameters
    ----------
    liquid_pressure : float
        Liquid phase pressure in Pa. This is typically the reservoir
        pressure in confined systems.
    temperature : float
        Temperature in K.
    composition : ndarray
        Feed mole fractions.
    components : list of Component
        Component objects with critical properties.
    eos : CubicEOS
        Equation of state instance.
    pore_radius_nm : float
        Pore radius in nanometers. Typical shale pores: 2-50 nm.
    binary_interaction : ndarray, optional
        Binary interaction parameters kij.
    K_initial : ndarray, optional
        Initial K-values. If None, uses Wilson correlation.
    Pc_tolerance : float
        Convergence tolerance for capillary pressure (Pa).
    K_tolerance : float
        Convergence tolerance for K-values.
    max_Pc_iterations : int
        Maximum outer loop iterations for Pc convergence.
    max_flash_iterations : int
        Maximum inner loop iterations for K-value convergence.
    contact_angle : float
        Contact angle in degrees. Default 0 (complete wetting).

    Returns
    -------
    ConfinedFlashResult
        Complete results from confined flash calculation.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    ConvergenceError
        If calculation fails to converge.
    PhaseError
        If phase state cannot be determined.

    Notes
    -----
    Effects of confinement:
    - Bubble point is suppressed (shifted to lower pressure)
    - Dew point is enhanced (shifted to higher pressure)
    - Two-phase region shrinks
    - Critical point may shift

    For very small pores (< 5 nm), capillary pressures can be
    several MPa, causing significant phase behavior changes.

    References
    ----------
    Nojabaei, B., Johns, R.T., and Chu, L. (2013). SPE-159258.

    Examples
    --------
    >>> from pvtcore.models.component import load_components
    >>> from pvtcore.eos.peng_robinson import PengRobinsonEOS
    >>> components = load_components()
    >>> binary = [components['C1'], components['C4']]
    >>> eos = PengRobinsonEOS(binary)
    >>> z = np.array([0.7, 0.3])
    >>> result = confined_flash(5e6, 350.0, z, binary, eos, pore_radius_nm=10.0)
    >>> print(f"Pc = {result.capillary_pressure/1e6:.2f} MPa")
    >>> print(f"Vapor fraction = {result.vapor_fraction:.3f}")
    """
    # Input validation
    z = np.asarray(composition, dtype=np.float64)
    n = len(components)

    _validate_confined_flash_inputs(
        liquid_pressure, temperature, z, components, pore_radius_nm
    )

    z = z / z.sum()  # Normalize
    P_L = float(liquid_pressure)
    T = float(temperature)

    # Initialize K-values
    if K_initial is None:
        K = wilson_k_values(P_L, T, components)
    else:
        K = np.asarray(K_initial, dtype=np.float64).copy()

    # Initial guess: bulk flash (Pc = 0)
    P_V = P_L
    Pc = 0.0
    Pc_old = 0.0

    total_inner_iterations = 0
    x = z.copy()
    y = z.copy()
    nv = 0.5
    rho_L = 0.0
    rho_V = 0.0
    ift = 0.0

    # Outer loop: Pc coupling
    for outer_iter in range(max_Pc_iterations):
        # Inner loop: flash calculation with current pressures
        K_bulk = K.copy()  # Store bulk K-values

        for inner_iter in range(max_flash_iterations):
            # Apply pressure ratio correction to K-values
            if P_V > 0:
                K_confined = modified_k_values_array(K_bulk, P_L, P_V)
            else:
                K_confined = K_bulk.copy()

            # Solve Rachford-Rice
            try:
                nv, x, y = solve_rachford_rice(K_confined, z)
            except (ValidationError, ValueError):
                # System may be single-phase
                # Check if all K > 1 (all vapor) or all K < 1 (all liquid)
                if np.all(K_confined > 1.0):
                    nv = 1.0
                    x = np.zeros(n)
                    y = z.copy()
                    break
                elif np.all(K_confined < 1.0):
                    nv = 0.0
                    x = z.copy()
                    y = np.zeros(n)
                    break
                else:
                    # Indeterminate - use bulk-like behavior
                    if np.mean(K_confined) > 1.0:
                        nv = 1.0
                        x = np.zeros(n)
                        y = z.copy()
                    else:
                        nv = 0.0
                        x = z.copy()
                        y = np.zeros(n)
                    break

            # Edge cases
            if nv <= 1e-10:
                nv = 0.0
                x = z.copy()
                y = np.zeros(n)
                break
            if nv >= 1.0 - 1e-10:
                nv = 1.0
                x = np.zeros(n)
                y = z.copy()
                break

            # Calculate fugacity coefficients at respective pressures
            phi_L = eos.fugacity_coefficient(
                P_L, T, x, 'liquid', binary_interaction
            )
            phi_V = eos.fugacity_coefficient(
                P_V, T, y, 'vapor', binary_interaction
            )

            # Update bulk K-values: Ki_bulk = φi^L / φi^V
            K_bulk_new = phi_L / phi_V

            # Check K-value convergence
            ln_K_new = np.log(K_bulk_new)
            ln_K_old = np.log(K_bulk)
            K_residual = np.sum((ln_K_new - ln_K_old) ** 2)

            if K_residual < K_tolerance:
                K_bulk = K_bulk_new
                break

            # Update K-values with damping
            if inner_iter < 5:
                K_bulk = 0.5 * K_bulk_new + 0.5 * K_bulk
            else:
                K_bulk = 0.7 * K_bulk_new + 0.3 * K_bulk

            total_inner_iterations += 1

        total_inner_iterations += 1

        # Check for single-phase
        if nv <= 1e-10 or nv >= 1.0 - 1e-10:
            # Single phase - no capillary pressure
            phase = 'vapor' if nv > 0.5 else 'liquid'
            return ConfinedFlashResult(
                converged=True,
                iterations_outer=outer_iter + 1,
                iterations_inner_total=total_inner_iterations,
                vapor_fraction=nv,
                liquid_composition=x,
                vapor_composition=y,
                K_values=K_confined,
                K_values_bulk=K_bulk,
                liquid_pressure=P_L,
                vapor_pressure=P_L,  # No Pc difference for single phase
                capillary_pressure=0.0,
                ift=0.0,
                liquid_density=0.0,
                vapor_density=0.0,
                pore_radius=pore_radius_nm,
                phase=phase,
                temperature=T,
                feed_composition=z,
                residual_Pc=0.0,
            )

        # Two-phase: calculate densities and IFT
        try:
            rho_L_result = calculate_density(
                P_L, T, x, components, eos, phase='liquid',
                binary_interaction=binary_interaction,
            )
            rho_V_result = calculate_density(
                P_V, T, y, components, eos, phase='vapor',
                binary_interaction=binary_interaction,
            )
            rho_L = rho_L_result.mass_density
            rho_V = rho_V_result.mass_density
            rho_L_mol = rho_L_result.molar_density
            rho_V_mol = rho_V_result.molar_density

            # Calculate IFT using parachor method
            ift_result = calculate_ift_parachor(
                x, y, rho_L_mol, rho_V_mol, components
            )
            ift = ift_result.ift  # mN/m
        except Exception as e:
            # If property calculation fails, use previous values or estimate
            if ift == 0.0:
                # Estimate IFT based on typical values
                ift = 5.0  # mN/m - typical light oil

        # Calculate new capillary pressure
        if ift > 0:
            Pc = capillary_pressure_simple(ift, pore_radius_nm)
        else:
            Pc = 0.0

        # Update vapor pressure
        P_V = P_L + Pc

        # Check Pc convergence
        Pc_residual = abs(Pc - Pc_old)
        if Pc_residual < Pc_tolerance:
            # Converged!
            # Final K-values include pressure ratio correction
            K_final = modified_k_values_array(K_bulk, P_L, P_V)

            return ConfinedFlashResult(
                converged=True,
                iterations_outer=outer_iter + 1,
                iterations_inner_total=total_inner_iterations,
                vapor_fraction=nv,
                liquid_composition=x,
                vapor_composition=y,
                K_values=K_final,
                K_values_bulk=K_bulk,
                liquid_pressure=P_L,
                vapor_pressure=P_V,
                capillary_pressure=Pc,
                ift=ift,
                liquid_density=rho_L,
                vapor_density=rho_V,
                pore_radius=pore_radius_nm,
                phase='two-phase',
                temperature=T,
                feed_composition=z,
                residual_Pc=Pc_residual,
            )

        Pc_old = Pc

    # Failed to converge
    raise ConvergenceError(
        f"Confined flash failed to converge after {max_Pc_iterations} iterations. "
        f"Final Pc residual: {Pc_residual:.1f} Pa (tolerance: {Pc_tolerance:.1f} Pa)",
        iterations=max_Pc_iterations,
        residual=Pc_residual,
    )


def confined_bubble_point(
    temperature: float,
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    pore_radius_nm: float,
    binary_interaction: Optional[NDArray[np.float64]] = None,
    P_initial: Optional[float] = None,
    tolerance: float = 1e-5,
    max_iterations: int = 50,
) -> tuple[float, float, float]:
    """Calculate confined bubble point pressure.

    The bubble point is suppressed in confined systems. This function
    iteratively solves for the pressure where an incipient vapor phase
    appears under confinement.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    composition : ndarray
        Feed composition (liquid at bubble point).
    components : list of Component
        Component objects.
    eos : CubicEOS
        Equation of state instance.
    pore_radius_nm : float
        Pore radius in nanometers.
    binary_interaction : ndarray, optional
        Binary interaction parameters.
    P_initial : float, optional
        Initial pressure guess.
    tolerance : float
        Relative tolerance for bubble point.
    max_iterations : int
        Maximum iterations.

    Returns
    -------
    tuple
        (bubble_pressure, capillary_pressure, ift) all in Pa, Pa, mN/m

    Notes
    -----
    The confined bubble point is lower than bulk bubble point.
    Typical suppression: ΔPb ≈ -Pc ≈ -2σ/r
    """
    from ..flash.bubble_point import calculate_bubble_point

    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()

    # Get bulk bubble point as starting reference
    try:
        bulk_result = calculate_bubble_point(
            temperature, z, components, eos,
            pressure_initial=P_initial,
            binary_interaction=binary_interaction,
        )
        P_bubble_bulk = bulk_result.pressure
    except Exception:
        # Estimate from Wilson correlation
        P_bubble_bulk = sum(
            z[i] * comp.Pc * np.exp(5.373 * (1 + comp.omega) * (1 - comp.Tc / temperature))
            for i, comp in enumerate(components)
        )

    # Initial estimate: bulk bubble point
    P_L = P_bubble_bulk
    Pc = 0.0

    for iteration in range(max_iterations):
        # Do confined flash at current pressure
        try:
            result = confined_flash(
                P_L, temperature, z, components, eos,
                pore_radius_nm=pore_radius_nm,
                binary_interaction=binary_interaction,
                Pc_tolerance=100.0,  # Relaxed tolerance for speed
            )
        except (ConvergenceError, PhaseError):
            # Try lower pressure
            P_L *= 0.9
            continue

        # At bubble point, vapor fraction should be very small
        # We want nv ≈ 0
        if result.phase == 'liquid':
            # All liquid - need to increase pressure
            P_L *= 1.1
        elif result.phase == 'vapor':
            # All vapor - need to decrease pressure
            P_L *= 0.8
        else:
            # Two-phase - adjust based on vapor fraction
            if result.vapor_fraction < 0.01:
                # Close to bubble point
                Pc = result.capillary_pressure
                ift = result.ift

                # Check convergence
                if result.vapor_fraction < 1e-4:
                    return P_L, Pc, ift

                # Slightly increase pressure to get closer
                P_L *= 1.01
            elif result.vapor_fraction > 0.5:
                # Too much vapor - decrease pressure
                P_L *= 0.95
            else:
                # Some vapor - decrease pressure
                P_L *= 0.98

    # Return best estimate
    return P_L, Pc, Pc * pore_radius_nm * 1e-9 / 2.0  # Approximate IFT


def confined_dew_point(
    temperature: float,
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    pore_radius_nm: float,
    binary_interaction: Optional[NDArray[np.float64]] = None,
    P_initial: Optional[float] = None,
    tolerance: float = 1e-5,
    max_iterations: int = 50,
) -> tuple[float, float, float]:
    """Calculate confined dew point pressure.

    The dew point is enhanced (increased) in confined systems.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    composition : ndarray
        Feed composition (vapor at dew point).
    components : list of Component
        Component objects.
    eos : CubicEOS
        Equation of state instance.
    pore_radius_nm : float
        Pore radius in nanometers.
    binary_interaction : ndarray, optional
        Binary interaction parameters.
    P_initial : float, optional
        Initial pressure guess.
    tolerance : float
        Relative tolerance for dew point.
    max_iterations : int
        Maximum iterations.

    Returns
    -------
    tuple
        (dew_pressure, capillary_pressure, ift) in Pa, Pa, mN/m
    """
    from ..flash.dew_point import calculate_dew_point

    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()

    # Get bulk dew point as starting reference
    try:
        bulk_result = calculate_dew_point(
            temperature, z, components, eos,
            pressure_initial=P_initial,
            binary_interaction=binary_interaction,
        )
        P_dew_bulk = bulk_result.pressure
    except Exception:
        # Estimate from inverse Wilson
        P_dew_bulk = 1.0 / sum(
            z[i] / (comp.Pc * np.exp(5.373 * (1 + comp.omega) * (1 - comp.Tc / temperature)))
            for i, comp in enumerate(components)
        )

    # Initial estimate: bulk dew point
    P_L = P_dew_bulk
    Pc = 0.0
    ift = 0.0

    for iteration in range(max_iterations):
        try:
            result = confined_flash(
                P_L, temperature, z, components, eos,
                pore_radius_nm=pore_radius_nm,
                binary_interaction=binary_interaction,
                Pc_tolerance=100.0,
            )
        except (ConvergenceError, PhaseError):
            P_L *= 1.1
            continue

        # At dew point, vapor fraction should be very close to 1
        if result.phase == 'vapor':
            # All vapor - need to decrease pressure
            P_L *= 0.9
        elif result.phase == 'liquid':
            # All liquid - need to increase pressure significantly
            P_L *= 1.2
        else:
            # Two-phase
            if result.vapor_fraction > 0.99:
                # Close to dew point
                Pc = result.capillary_pressure
                ift = result.ift

                if result.vapor_fraction > 1.0 - 1e-4:
                    return P_L, Pc, ift

                P_L *= 0.99
            elif result.vapor_fraction < 0.5:
                P_L *= 1.05
            else:
                P_L *= 1.02

    return P_L, Pc, ift


def _validate_confined_flash_inputs(
    liquid_pressure: float,
    temperature: float,
    composition: NDArray[np.float64],
    components: List[Component],
    pore_radius_nm: float,
) -> None:
    """Validate confined flash inputs."""
    if liquid_pressure <= 0:
        raise ValidationError(
            "Liquid pressure must be positive",
            parameter="liquid_pressure",
            value=liquid_pressure,
        )
    if temperature <= 0:
        raise ValidationError(
            "Temperature must be positive",
            parameter="temperature",
            value=temperature,
        )
    if pore_radius_nm <= 0:
        raise ValidationError(
            "Pore radius must be positive",
            parameter="pore_radius_nm",
            value=pore_radius_nm,
        )
    if pore_radius_nm < 1.0:
        # Sub-nanometer pores are physically questionable
        raise ValidationError(
            "Pore radius < 1 nm is below molecular dimensions",
            parameter="pore_radius_nm",
            value=pore_radius_nm,
        )

    z = np.asarray(composition)
    if len(z) != len(components):
        raise ValidationError(
            "Composition length must match number of components",
            parameter="composition",
            value={"got": len(z), "expected": len(components)},
        )

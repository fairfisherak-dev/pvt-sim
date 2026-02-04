"""Constant Volume Depletion (CVD) simulation.

CVD is a standard PVT laboratory test for gas condensate systems.
The cell volume is held constant while pressure is reduced, and
excess gas is removed to maintain constant volume.

At each pressure step:
1. Flash at new pressure
2. Calculate total volume
3. Remove enough gas to restore original volume
4. Update composition

Key measurements:
- Liquid dropout curve
- Gas produced
- Z-factor (two-phase)
- Liquid recovery

Units Convention:
- Pressure: Pa
- Temperature: K
- Volume: relative to initial volume

References
----------
[1] McCain, W.D. (1990). The Properties of Petroleum Fluids.
[2] Pedersen et al. (2015). Phase Behavior of Petroleum Reservoir Fluids.
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
from ..flash.pt_flash import pt_flash
from ..properties.density import calculate_density, mixture_molecular_weight


@dataclass
class CVDStepResult:
    """Results from a single CVD pressure step.

    Attributes:
        pressure: Pressure at this step (Pa)
        temperature: Temperature (K)
        liquid_dropout: Liquid volume / cell volume (fraction)
        gas_produced: Gas removed this step (fraction of initial)
        cumulative_gas_produced: Total gas removed (fraction of initial)
        Z_two_phase: Two-phase compressibility factor
        liquid_density: Liquid density (kg/m³)
        vapor_density: Vapor density (kg/m³)
        liquid_composition: Liquid composition after step
        vapor_composition: Vapor composition
        cell_composition: Overall composition in cell after gas removal
        moles_remaining: Moles remaining in cell (fraction of initial)
    """
    pressure: float
    temperature: float
    liquid_dropout: float
    gas_produced: float
    cumulative_gas_produced: float
    Z_two_phase: float
    liquid_density: float
    vapor_density: float
    liquid_composition: NDArray[np.float64]
    vapor_composition: NDArray[np.float64]
    cell_composition: NDArray[np.float64]
    moles_remaining: float


@dataclass
class CVDResult:
    """Complete results from CVD simulation.

    Attributes:
        temperature: Test temperature (K)
        dew_pressure: Dew point pressure (Pa)
        initial_Z: Single-phase Z at dew point
        steps: List of results for each pressure step
        pressures: Array of pressures (Pa)
        liquid_dropouts: Array of liquid dropout fractions
        cumulative_gas: Array of cumulative gas produced
        Z_values: Array of two-phase Z factors
        feed_composition: Original feed composition
        converged: True if all steps converged
    """
    temperature: float
    dew_pressure: float
    initial_Z: float
    steps: List[CVDStepResult]
    pressures: NDArray[np.float64]
    liquid_dropouts: NDArray[np.float64]
    cumulative_gas: NDArray[np.float64]
    Z_values: NDArray[np.float64]
    feed_composition: NDArray[np.float64]
    converged: bool


def simulate_cvd(
    composition: NDArray[np.float64],
    temperature: float,
    components: List[Component],
    eos: CubicEOS,
    dew_pressure: float,
    pressure_steps: NDArray[np.float64],
    binary_interaction: Optional[NDArray[np.float64]] = None,
) -> CVDResult:
    """Simulate Constant Volume Depletion test.

    The CVD test reduces pressure while maintaining constant cell volume.
    Excess gas is removed at each step to restore original volume.

    Parameters
    ----------
    composition : ndarray
        Initial feed mole fractions (saturated gas at dew point).
    temperature : float
        Reservoir temperature in K.
    components : list of Component
        Component objects.
    eos : CubicEOS
        Equation of state instance.
    dew_pressure : float
        Dew point pressure in Pa.
    pressure_steps : ndarray
        Pressure steps for CVD test (descending from Pd).
    binary_interaction : ndarray, optional
        Binary interaction parameters.

    Returns
    -------
    CVDResult
        Complete CVD test results.

    Notes
    -----
    Key outputs:
    - Liquid dropout: Volume of retrograde liquid
    - Gas produced: Amount removed to maintain volume
    - Z-factor: Two-phase compressibility

    The CVD test is particularly important for:
    - Gas condensate reservoirs
    - Estimating liquid recovery
    - Determining when liquid re-vaporizes

    Examples
    --------
    >>> from pvtcore.models.component import load_components
    >>> from pvtcore.eos.peng_robinson import PengRobinsonEOS
    >>> components = load_components()
    >>> gas_cond = [components['C1'], components['C3'], components['C7']]
    >>> eos = PengRobinsonEOS(gas_cond)
    >>> z = np.array([0.85, 0.10, 0.05])
    >>> P_steps = np.linspace(20e6, 5e6, 15)
    >>> result = simulate_cvd(z, 380.0, gas_cond, eos, 25e6, P_steps)
    >>> print(f"Max liquid dropout: {max(result.liquid_dropouts)*100:.1f}%")
    """
    # Validate inputs
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()
    T = float(temperature)
    P_d = float(dew_pressure)

    _validate_cvd_inputs(z, T, P_d, pressure_steps, components)

    # Initial state: single-phase gas at dew point
    Z_initial = eos.compressibility(P_d, T, z, phase='vapor', binary_interaction=binary_interaction)
    if isinstance(Z_initial, list):
        Z_initial = Z_initial[-1]

    # Initial cell volume (per mole)
    V_cell = Z_initial * R.Pa_m3_per_mol_K * T / P_d

    # Run CVD steps
    steps = []
    current_composition = z.copy()
    n_total = 1.0  # Total moles in cell
    cumulative_gas = 0.0
    all_converged = True

    # Add dew point as first step (no liquid, no gas removal)
    steps.append(CVDStepResult(
        pressure=P_d,
        temperature=T,
        liquid_dropout=0.0,
        gas_produced=0.0,
        cumulative_gas_produced=0.0,
        Z_two_phase=Z_initial,
        liquid_density=0.0,
        vapor_density=calculate_density(P_d, T, z, components, eos, 'vapor', binary_interaction).mass_density,
        liquid_composition=np.zeros_like(z),
        vapor_composition=z.copy(),
        cell_composition=z.copy(),
        moles_remaining=1.0,
    ))

    for P in pressure_steps:
        if P >= P_d:
            continue  # Skip pressures at or above dew point

        try:
            step_result, current_composition, n_total, cumulative_gas = _cvd_step(
                P, T, current_composition, n_total, cumulative_gas, V_cell,
                components, eos, binary_interaction
            )
            steps.append(step_result)
        except (ConvergenceError, PhaseError) as e:
            all_converged = False
            steps.append(CVDStepResult(
                pressure=P,
                temperature=T,
                liquid_dropout=np.nan,
                gas_produced=np.nan,
                cumulative_gas_produced=cumulative_gas,
                Z_two_phase=np.nan,
                liquid_density=np.nan,
                vapor_density=np.nan,
                liquid_composition=current_composition.copy(),
                vapor_composition=np.zeros_like(z),
                cell_composition=current_composition.copy(),
                moles_remaining=n_total,
            ))

    # Extract arrays
    pressures = np.array([s.pressure for s in steps])
    liquid_dropouts = np.array([s.liquid_dropout for s in steps])
    cumulative_gas_arr = np.array([s.cumulative_gas_produced for s in steps])
    Z_values = np.array([s.Z_two_phase for s in steps])

    return CVDResult(
        temperature=T,
        dew_pressure=P_d,
        initial_Z=Z_initial,
        steps=steps,
        pressures=pressures,
        liquid_dropouts=liquid_dropouts,
        cumulative_gas=cumulative_gas_arr,
        Z_values=Z_values,
        feed_composition=z,
        converged=all_converged,
    )


def _cvd_step(
    pressure: float,
    temperature: float,
    cell_composition: NDArray[np.float64],
    n_total: float,
    cumulative_gas: float,
    V_cell: float,
    components: List[Component],
    eos: CubicEOS,
    binary_interaction: Optional[NDArray[np.float64]],
) -> tuple[CVDStepResult, NDArray[np.float64], float, float]:
    """Execute single CVD pressure step."""
    P = float(pressure)
    T = float(temperature)
    z = cell_composition

    # Flash at current P, T
    flash = pt_flash(P, T, z, components, eos, binary_interaction=binary_interaction)

    if flash.phase == 'vapor':
        # Still single-phase gas
        Z = eos.compressibility(P, T, z, phase='vapor', binary_interaction=binary_interaction)
        if isinstance(Z, list):
            Z = Z[-1]

        # Volume at current P with current moles
        V_current = n_total * Z * R.Pa_m3_per_mol_K * T / P

        # Gas to remove to restore V_cell
        # V_cell = (n_total - n_remove) * Z * R * T / P
        n_remove = n_total - V_cell * P / (Z * R.Pa_m3_per_mol_K * T)
        n_remove = max(0.0, n_remove)

        gas_produced = n_remove  # Fraction of initial = n_remove (since initial = 1)
        cumulative_gas += gas_produced
        n_new = n_total - n_remove

        rho_V = calculate_density(P, T, z, components, eos, 'vapor', binary_interaction)

        return (
            CVDStepResult(
                pressure=P,
                temperature=T,
                liquid_dropout=0.0,
                gas_produced=gas_produced,
                cumulative_gas_produced=cumulative_gas,
                Z_two_phase=Z,
                liquid_density=0.0,
                vapor_density=rho_V.mass_density,
                liquid_composition=np.zeros_like(z),
                vapor_composition=z.copy(),
                cell_composition=z.copy(),
                moles_remaining=n_new,
            ),
            z.copy(),
            n_new,
            cumulative_gas,
        )

    # Two-phase: liquid dropout
    nv = flash.vapor_fraction
    x = flash.liquid_composition
    y = flash.vapor_composition

    # Moles of each phase (per mole in cell)
    n_L = (1 - nv) * n_total
    n_V = nv * n_total

    # Phase volumes
    Z_L = eos.compressibility(P, T, x, phase='liquid', binary_interaction=binary_interaction)
    Z_V = eos.compressibility(P, T, y, phase='vapor', binary_interaction=binary_interaction)
    if isinstance(Z_L, list):
        Z_L = Z_L[0]
    if isinstance(Z_V, list):
        Z_V = Z_V[-1]

    V_L = n_L * Z_L * R.Pa_m3_per_mol_K * T / P
    V_V = n_V * Z_V * R.Pa_m3_per_mol_K * T / P
    V_total = V_L + V_V

    # Two-phase Z
    Z_two_phase = P * V_total / (n_total * R.Pa_m3_per_mol_K * T)

    # Gas to remove to restore V_cell
    # Removing gas changes nV but not liquid (assuming liquid doesn't flash)
    # V_cell = V_L + (n_V - n_remove) * Z_V * R * T / P
    # Solve for n_remove:
    if V_V > 0:
        V_excess = V_total - V_cell
        n_remove = V_excess * P / (Z_V * R.Pa_m3_per_mol_K * T)
        n_remove = max(0.0, min(n_remove, n_V))  # Can't remove more gas than exists
    else:
        n_remove = 0.0

    gas_produced = n_remove
    cumulative_gas += gas_produced

    # New vapor moles after removal
    n_V_new = n_V - n_remove
    n_new = n_L + n_V_new

    # New overall composition
    if n_new > 0:
        z_new = (n_L * x + n_V_new * y) / n_new
        z_new = z_new / z_new.sum()  # Normalize
    else:
        z_new = x.copy()

    # Liquid dropout (as fraction of cell volume)
    liquid_dropout = V_L / V_cell

    # Densities
    rho_L = calculate_density(P, T, x, components, eos, 'liquid', binary_interaction)
    rho_V = calculate_density(P, T, y, components, eos, 'vapor', binary_interaction)

    return (
        CVDStepResult(
            pressure=P,
            temperature=T,
            liquid_dropout=liquid_dropout,
            gas_produced=gas_produced,
            cumulative_gas_produced=cumulative_gas,
            Z_two_phase=Z_two_phase,
            liquid_density=rho_L.mass_density,
            vapor_density=rho_V.mass_density,
            liquid_composition=x.copy(),
            vapor_composition=y.copy(),
            cell_composition=z_new,
            moles_remaining=n_new,
        ),
        z_new,
        n_new,
        cumulative_gas,
    )


def _validate_cvd_inputs(
    composition: NDArray[np.float64],
    temperature: float,
    dew_pressure: float,
    pressure_steps: NDArray[np.float64],
    components: List[Component],
) -> None:
    """Validate CVD inputs."""
    if temperature <= 0:
        raise ValidationError("Temperature must be positive", parameter="temperature")
    if dew_pressure <= 0:
        raise ValidationError("Dew pressure must be positive", parameter="dew_pressure")
    if len(composition) != len(components):
        raise ValidationError(
            "Composition length must match number of components",
            parameter="composition"
        )
    if np.any(pressure_steps <= 0):
        raise ValidationError("All pressure steps must be positive", parameter="pressure_steps")

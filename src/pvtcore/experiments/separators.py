"""Multi-stage separator calculations.

Surface separator trains are used to process reservoir fluids into
stock-tank oil and sales gas. The separator conditions (P, T) at
each stage significantly affect the final oil and gas volumes.

Typical separator configurations:
- Single stage: Reservoir → Stock tank (inefficient)
- Two stage: Reservoir → HP separator → Stock tank
- Three stage: Reservoir → HP → LP separator → Stock tank

At each separator:
1. Flash the inlet stream
2. Remove vapor (to gas sales or next stage)
3. Liquid proceeds to next separator or stock tank

Key outputs:
- Formation volume factor Bo (res bbl/STB)
- Solution GOR Rs (scf/STB)
- Gas formation volume factor Bg
- API gravity of stock-tank oil
- Separator gas compositions

Units Convention:
- Pressure: Pa
- Temperature: K
- GOR: m³/m³ (or scf/STB with conversion)
- Bo: dimensionless (reservoir volume / stock tank volume)

References
----------
[1] McCain, W.D. (1990). The Properties of Petroleum Fluids.
[2] Pedersen et al. (2015). Phase Behavior of Petroleum Reservoir Fluids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.constants import R
from ..core.errors import ConvergenceError, PhaseError, ValidationError
from ..eos.base import CubicEOS
from ..models.component import Component
from ..flash.pt_flash import pt_flash
from ..properties.density import calculate_density, mixture_molecular_weight


# Standard conditions for gas volume (15°C, 1 atm)
T_STD = 288.15  # K (15°C / 60°F)
P_STD = 101325.0  # Pa (1 atm)

# Stock tank conditions
T_STOCK_TANK = 288.15  # K
P_STOCK_TANK = 101325.0  # Pa


@dataclass
class SeparatorConditions:
    """Conditions for a single separator stage.

    Attributes:
        pressure: Separator pressure (Pa)
        temperature: Separator temperature (K)
        name: Optional stage name (e.g., "HP Separator")
    """
    pressure: float
    temperature: float
    name: str = ""


@dataclass
class SeparatorStageResult:
    """Results from a single separator stage.

    Attributes:
        stage_number: Stage index (0-based)
        conditions: P, T conditions
        inlet_composition: Feed to this stage
        inlet_moles: Moles fed to this stage
        vapor_fraction: Fraction flashing to vapor
        liquid_composition: Liquid leaving stage
        vapor_composition: Vapor leaving stage
        liquid_moles: Moles of liquid produced
        vapor_moles: Moles of vapor produced
        liquid_density: Liquid density (kg/m³)
        vapor_density: Vapor density (kg/m³)
        Z_liquid: Liquid compressibility factor
        Z_vapor: Vapor compressibility factor
        converged: True if flash converged
    """
    stage_number: int
    conditions: SeparatorConditions
    inlet_composition: NDArray[np.float64]
    inlet_moles: float
    vapor_fraction: float
    liquid_composition: NDArray[np.float64]
    vapor_composition: NDArray[np.float64]
    liquid_moles: float
    vapor_moles: float
    liquid_density: float
    vapor_density: float
    Z_liquid: float
    Z_vapor: float
    converged: bool


@dataclass
class SeparatorTrainResult:
    """Complete results from multi-stage separator calculation.

    Attributes:
        stages: Results for each separator stage
        stock_tank_oil_composition: Final oil composition
        stock_tank_oil_moles: Moles of stock-tank oil per mole feed
        stock_tank_oil_density: Oil density at stock tank (kg/m³)
        stock_tank_oil_MW: Molecular weight of stock-tank oil
        stock_tank_oil_SG: Specific gravity (relative to water)
        API_gravity: API gravity of stock-tank oil
        total_gas_composition: Combined separator gas composition
        total_gas_moles: Total moles of gas per mole feed
        Bo: Oil formation volume factor (res vol / ST vol)
        Rs: Solution GOR (m³ gas at std / m³ oil at std)
        Rs_scf_stb: Solution GOR in scf/STB
        Bg: Gas formation volume factor
        shrinkage: Oil shrinkage factor (1 / Bo)
        converged: True if all stages converged
    """
    stages: List[SeparatorStageResult]
    stock_tank_oil_composition: NDArray[np.float64]
    stock_tank_oil_moles: float
    stock_tank_oil_density: float
    stock_tank_oil_MW: float
    stock_tank_oil_SG: float
    API_gravity: float
    total_gas_composition: NDArray[np.float64]
    total_gas_moles: float
    Bo: float
    Rs: float
    Rs_scf_stb: float
    Bg: float
    shrinkage: float
    converged: bool


def calculate_separator_train(
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    separator_stages: List[SeparatorConditions],
    reservoir_pressure: float,
    reservoir_temperature: float,
    binary_interaction: Optional[NDArray[np.float64]] = None,
    include_stock_tank: bool = True,
) -> SeparatorTrainResult:
    """Calculate multi-stage separator performance.

    Simulates flow through separator train, flashing liquid at each
    stage and tracking gas and oil volumes.

    Parameters
    ----------
    composition : ndarray
        Reservoir fluid mole fractions.
    components : list of Component
        Component objects.
    eos : CubicEOS
        Equation of state instance.
    separator_stages : list of SeparatorConditions
        Separator conditions (P, T) for each stage.
        Should be in decreasing pressure order.
    reservoir_pressure : float
        Reservoir pressure in Pa (for Bo calculation).
    reservoir_temperature : float
        Reservoir temperature in K (for Bo calculation).
    binary_interaction : ndarray, optional
        Binary interaction parameters.
    include_stock_tank : bool
        If True, add final flash at stock tank conditions.

    Returns
    -------
    SeparatorTrainResult
        Complete separator train results.

    Notes
    -----
    The calculation assumes:
    - Gas from each stage is removed (no recombination)
    - Liquid from each stage feeds the next
    - Final liquid is stock-tank oil
    - Stock tank is at standard conditions if include_stock_tank=True

    Examples
    --------
    >>> from pvtcore.models.component import load_components
    >>> from pvtcore.eos.peng_robinson import PengRobinsonEOS
    >>> components = load_components()
    >>> oil = [components['C1'], components['C4'], components['C10']]
    >>> eos = PengRobinsonEOS(oil)
    >>> z = np.array([0.40, 0.35, 0.25])
    >>> stages = [
    ...     SeparatorConditions(pressure=3e6, temperature=320.0, name="HP Sep"),
    ...     SeparatorConditions(pressure=0.5e6, temperature=300.0, name="LP Sep"),
    ... ]
    >>> result = calculate_separator_train(
    ...     z, oil, eos, stages,
    ...     reservoir_pressure=30e6, reservoir_temperature=380.0
    ... )
    >>> print(f"Bo = {result.Bo:.3f}")
    >>> print(f"Rs = {result.Rs_scf_stb:.0f} scf/STB")
    """
    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()
    nc = len(z)

    _validate_separator_inputs(z, components, separator_stages)

    # Build complete stage list
    all_stages = list(separator_stages)
    if include_stock_tank:
        all_stages.append(SeparatorConditions(
            pressure=P_STOCK_TANK,
            temperature=T_STOCK_TANK,
            name="Stock Tank",
        ))

    # Initialize tracking variables
    stage_results: List[SeparatorStageResult] = []
    current_liquid = z.copy()
    current_moles = 1.0  # Start with 1 mole of feed
    total_gas_moles = 0.0
    gas_streams: List[Tuple[NDArray[np.float64], float]] = []  # (composition, moles)
    all_converged = True

    # Process each separator stage
    for i, stage in enumerate(all_stages):
        try:
            result = _separator_stage(
                current_liquid,
                current_moles,
                stage,
                i,
                components,
                eos,
                binary_interaction,
            )
            stage_results.append(result)

            if result.converged:
                # Update for next stage
                current_liquid = result.liquid_composition.copy()
                current_moles = result.liquid_moles

                # Collect gas
                if result.vapor_moles > 0:
                    gas_streams.append((result.vapor_composition.copy(), result.vapor_moles))
                    total_gas_moles += result.vapor_moles
            else:
                all_converged = False

        except (ConvergenceError, PhaseError) as e:
            all_converged = False
            # Create failure result
            stage_results.append(SeparatorStageResult(
                stage_number=i,
                conditions=stage,
                inlet_composition=current_liquid.copy(),
                inlet_moles=current_moles,
                vapor_fraction=np.nan,
                liquid_composition=current_liquid.copy(),
                vapor_composition=np.zeros(nc),
                liquid_moles=current_moles,
                vapor_moles=0.0,
                liquid_density=np.nan,
                vapor_density=np.nan,
                Z_liquid=np.nan,
                Z_vapor=np.nan,
                converged=False,
            ))

    # Stock-tank oil properties
    stock_tank_oil = current_liquid
    stock_tank_moles = current_moles

    # Calculate stock-tank oil properties
    ST_MW = mixture_molecular_weight(stock_tank_oil, components)

    try:
        ST_density_result = calculate_density(
            P_STOCK_TANK, T_STOCK_TANK, stock_tank_oil, components, eos,
            phase='liquid', binary_interaction=binary_interaction
        )
        ST_density = ST_density_result.mass_density
    except Exception:
        # Estimate from ideal
        ST_density = 750.0  # Rough estimate for oil

    # Specific gravity and API
    water_density = 999.0  # kg/m³ at 15°C
    ST_SG = ST_density / water_density
    API_gravity = 141.5 / ST_SG - 131.5

    # Combined gas composition
    if total_gas_moles > 0 and gas_streams:
        total_gas_comp = np.zeros(nc)
        for gas_comp, gas_moles in gas_streams:
            total_gas_comp += gas_comp * gas_moles
        total_gas_comp /= total_gas_moles
    else:
        total_gas_comp = np.zeros(nc)

    # Calculate Bo (formation volume factor)
    Bo, Bg = _calculate_formation_factors(
        z, components, eos,
        reservoir_pressure, reservoir_temperature,
        stock_tank_oil, stock_tank_moles,
        P_STD, T_STD,
        binary_interaction,
    )

    # Calculate Rs (solution GOR)
    # Rs = gas volume at std / oil volume at std
    Rs = _calculate_solution_gor(
        total_gas_moles,
        stock_tank_moles,
        total_gas_comp,
        stock_tank_oil,
        components,
        eos,
        binary_interaction,
    )

    # Convert Rs to scf/STB (1 m³/m³ = 5.615 scf/STB)
    Rs_scf_stb = Rs * 5.615

    return SeparatorTrainResult(
        stages=stage_results,
        stock_tank_oil_composition=stock_tank_oil,
        stock_tank_oil_moles=stock_tank_moles,
        stock_tank_oil_density=ST_density,
        stock_tank_oil_MW=ST_MW,
        stock_tank_oil_SG=ST_SG,
        API_gravity=API_gravity,
        total_gas_composition=total_gas_comp,
        total_gas_moles=total_gas_moles,
        Bo=Bo,
        Rs=Rs,
        Rs_scf_stb=Rs_scf_stb,
        Bg=Bg,
        shrinkage=1.0 / Bo if Bo > 0 else np.nan,
        converged=all_converged,
    )


def _separator_stage(
    inlet_composition: NDArray[np.float64],
    inlet_moles: float,
    conditions: SeparatorConditions,
    stage_number: int,
    components: List[Component],
    eos: CubicEOS,
    binary_interaction: Optional[NDArray[np.float64]],
) -> SeparatorStageResult:
    """Execute single separator stage."""
    P = conditions.pressure
    T = conditions.temperature
    z = inlet_composition

    # Flash at separator conditions
    flash = pt_flash(P, T, z, components, eos, binary_interaction=binary_interaction)

    if flash.phase == 'liquid':
        # All liquid, no gas evolution
        Z_L = eos.compressibility(P, T, z, phase='liquid', binary_interaction=binary_interaction)
        if isinstance(Z_L, list):
            Z_L = Z_L[0]

        rho_L = calculate_density(P, T, z, components, eos, 'liquid', binary_interaction)

        return SeparatorStageResult(
            stage_number=stage_number,
            conditions=conditions,
            inlet_composition=z.copy(),
            inlet_moles=inlet_moles,
            vapor_fraction=0.0,
            liquid_composition=z.copy(),
            vapor_composition=np.zeros_like(z),
            liquid_moles=inlet_moles,
            vapor_moles=0.0,
            liquid_density=rho_L.mass_density,
            vapor_density=0.0,
            Z_liquid=Z_L,
            Z_vapor=0.0,
            converged=True,
        )

    if flash.phase == 'vapor':
        # All vapor - unusual for separator but handle it
        Z_V = eos.compressibility(P, T, z, phase='vapor', binary_interaction=binary_interaction)
        if isinstance(Z_V, list):
            Z_V = Z_V[-1]

        rho_V = calculate_density(P, T, z, components, eos, 'vapor', binary_interaction)

        return SeparatorStageResult(
            stage_number=stage_number,
            conditions=conditions,
            inlet_composition=z.copy(),
            inlet_moles=inlet_moles,
            vapor_fraction=1.0,
            liquid_composition=np.zeros_like(z),
            vapor_composition=z.copy(),
            liquid_moles=0.0,
            vapor_moles=inlet_moles,
            liquid_density=0.0,
            vapor_density=rho_V.mass_density,
            Z_liquid=0.0,
            Z_vapor=Z_V,
            converged=True,
        )

    # Two-phase
    nv = flash.vapor_fraction
    x = flash.liquid_composition
    y = flash.vapor_composition

    n_L = (1 - nv) * inlet_moles
    n_V = nv * inlet_moles

    # Phase compressibilities
    Z_L = eos.compressibility(P, T, x, phase='liquid', binary_interaction=binary_interaction)
    Z_V = eos.compressibility(P, T, y, phase='vapor', binary_interaction=binary_interaction)
    if isinstance(Z_L, list):
        Z_L = Z_L[0]
    if isinstance(Z_V, list):
        Z_V = Z_V[-1]

    # Densities
    rho_L = calculate_density(P, T, x, components, eos, 'liquid', binary_interaction)
    rho_V = calculate_density(P, T, y, components, eos, 'vapor', binary_interaction)

    return SeparatorStageResult(
        stage_number=stage_number,
        conditions=conditions,
        inlet_composition=z.copy(),
        inlet_moles=inlet_moles,
        vapor_fraction=nv,
        liquid_composition=x.copy(),
        vapor_composition=y.copy(),
        liquid_moles=n_L,
        vapor_moles=n_V,
        liquid_density=rho_L.mass_density,
        vapor_density=rho_V.mass_density,
        Z_liquid=Z_L,
        Z_vapor=Z_V,
        converged=True,
    )


def _calculate_formation_factors(
    feed_composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    P_res: float,
    T_res: float,
    oil_composition: NDArray[np.float64],
    oil_moles: float,
    P_std: float,
    T_std: float,
    binary_interaction: Optional[NDArray[np.float64]],
) -> Tuple[float, float]:
    """Calculate oil and gas formation volume factors."""
    # Reservoir oil volume (molar volume × moles)
    try:
        Z_res = eos.compressibility(
            P_res, T_res, feed_composition, phase='liquid',
            binary_interaction=binary_interaction
        )
        if isinstance(Z_res, list):
            Z_res = Z_res[0]
        V_res = Z_res * R.Pa_m3_per_mol_K * T_res / P_res  # m³/mol
    except Exception:
        # Use vapor Z if liquid fails
        Z_res = eos.compressibility(
            P_res, T_res, feed_composition, phase='vapor',
            binary_interaction=binary_interaction
        )
        if isinstance(Z_res, list):
            Z_res = Z_res[-1]
        V_res = Z_res * R.Pa_m3_per_mol_K * T_res / P_res

    # Stock tank oil volume
    try:
        Z_ST = eos.compressibility(
            P_std, T_std, oil_composition, phase='liquid',
            binary_interaction=binary_interaction
        )
        if isinstance(Z_ST, list):
            Z_ST = Z_ST[0]
        V_ST = Z_ST * R.Pa_m3_per_mol_K * T_std / P_std  # m³/mol
    except Exception:
        Z_ST = 0.01  # Rough estimate
        V_ST = Z_ST * R.Pa_m3_per_mol_K * T_std / P_std

    # Bo = V_res(1 mol feed) / V_ST(oil_moles mol oil)
    V_res_total = V_res * 1.0  # 1 mole feed
    V_ST_total = V_ST * oil_moles
    Bo = V_res_total / V_ST_total if V_ST_total > 0 else np.nan

    # Bg (gas formation volume factor)
    # Bg = V_gas(reservoir) / V_gas(standard)
    # For ideal gas: Bg = (P_std * T_res) / (P_res * T_std)
    Bg = (P_std * T_res) / (P_res * T_std)

    return Bo, Bg


def _calculate_solution_gor(
    gas_moles: float,
    oil_moles: float,
    gas_composition: NDArray[np.float64],
    oil_composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    binary_interaction: Optional[NDArray[np.float64]],
) -> float:
    """Calculate solution gas-oil ratio.

    Returns Rs in m³ gas at std / m³ oil at std.
    """
    if oil_moles <= 0 or gas_moles <= 0:
        return 0.0

    # Gas volume at standard conditions
    # Ideal gas: V = nRT/P
    V_gas_std = gas_moles * R.Pa_m3_per_mol_K * T_STD / P_STD

    # Oil volume at standard conditions
    try:
        Z_oil = eos.compressibility(
            P_STD, T_STD, oil_composition, phase='liquid',
            binary_interaction=binary_interaction
        )
        if isinstance(Z_oil, list):
            Z_oil = Z_oil[0]
        V_oil_mol = Z_oil * R.Pa_m3_per_mol_K * T_STD / P_STD
    except Exception:
        # Estimate liquid Z
        Z_oil = 0.01
        V_oil_mol = Z_oil * R.Pa_m3_per_mol_K * T_STD / P_STD

    V_oil_std = oil_moles * V_oil_mol

    Rs = V_gas_std / V_oil_std if V_oil_std > 0 else 0.0
    return Rs


def _validate_separator_inputs(
    composition: NDArray[np.float64],
    components: List[Component],
    stages: List[SeparatorConditions],
) -> None:
    """Validate separator inputs."""
    if len(composition) != len(components):
        raise ValidationError(
            "Composition length must match number of components",
            parameter="composition"
        )
    if not stages:
        raise ValidationError(
            "At least one separator stage required",
            parameter="separator_stages"
        )
    for i, stage in enumerate(stages):
        if stage.pressure <= 0:
            raise ValidationError(
                f"Stage {i} pressure must be positive",
                parameter=f"separator_stages[{i}].pressure"
            )
        if stage.temperature <= 0:
            raise ValidationError(
                f"Stage {i} temperature must be positive",
                parameter=f"separator_stages[{i}].temperature"
            )


def optimize_separator_pressures(
    composition: NDArray[np.float64],
    components: List[Component],
    eos: CubicEOS,
    reservoir_pressure: float,
    reservoir_temperature: float,
    n_stages: int = 2,
    binary_interaction: Optional[NDArray[np.float64]] = None,
    temperature: float = 300.0,
) -> Tuple[List[SeparatorConditions], SeparatorTrainResult]:
    """Find optimal separator pressures to maximize oil recovery.

    Uses a simple geometric spacing optimization. For more sophisticated
    optimization, use scipy.optimize with the objective being either
    maximum Bo (oil volume) or maximum API gravity.

    Parameters
    ----------
    composition : ndarray
        Reservoir fluid composition.
    components : list of Component
        Component objects.
    eos : CubicEOS
        Equation of state.
    reservoir_pressure : float
        Reservoir pressure (Pa).
    reservoir_temperature : float
        Reservoir temperature (K).
    n_stages : int
        Number of separator stages (excluding stock tank).
    binary_interaction : ndarray, optional
        Binary interaction parameters.
    temperature : float
        Separator temperature (assumed constant).

    Returns
    -------
    tuple
        (optimal_stages, result) - Best configuration and its results.

    Notes
    -----
    Optimal separator pressures often follow a geometric progression
    from reservoir pressure to stock-tank pressure. This function
    tests several configurations and returns the best.
    """
    if n_stages < 1:
        raise ValidationError("Need at least 1 separator stage", parameter="n_stages")

    P_high = min(reservoir_pressure * 0.3, 10e6)  # First stage typically 20-30% of Pres
    P_low = 0.2e6  # Last stage typically 2-5 bar

    best_result = None
    best_stages = None
    best_Bo = 0.0

    # Try different pressure ratios
    for ratio in [0.3, 0.4, 0.5]:
        pressures = _geometric_pressures(P_high, P_low, n_stages, ratio)
        stages = [
            SeparatorConditions(pressure=P, temperature=temperature, name=f"Stage {i+1}")
            for i, P in enumerate(pressures)
        ]

        try:
            result = calculate_separator_train(
                composition, components, eos, stages,
                reservoir_pressure, reservoir_temperature,
                binary_interaction=binary_interaction,
            )

            if result.converged and result.Bo > best_Bo:
                best_Bo = result.Bo
                best_stages = stages
                best_result = result
        except Exception:
            continue

    if best_stages is None:
        # Return default geometric spacing
        pressures = _geometric_pressures(P_high, P_low, n_stages, 0.4)
        best_stages = [
            SeparatorConditions(pressure=P, temperature=temperature, name=f"Stage {i+1}")
            for i, P in enumerate(pressures)
        ]
        best_result = calculate_separator_train(
            composition, components, eos, best_stages,
            reservoir_pressure, reservoir_temperature,
            binary_interaction=binary_interaction,
        )

    return best_stages, best_result


def _geometric_pressures(P_high: float, P_low: float, n_stages: int, ratio: float) -> List[float]:
    """Generate geometrically spaced pressures."""
    if n_stages == 1:
        return [np.sqrt(P_high * P_low)]

    log_ratio = np.log(P_low / P_high) / n_stages
    pressures = [P_high * np.exp(log_ratio * (i + 1) * ratio * n_stages / (n_stages - 0.5))
                 for i in range(n_stages)]
    return pressures

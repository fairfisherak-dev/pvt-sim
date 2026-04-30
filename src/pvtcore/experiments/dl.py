"""DL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ConvergenceError, PhaseError, ValidationError
from ..eos.base import CubicEOS
from ..flash.pt_flash import pt_flash
from ..helper_functions import (
    P_sc,
    SCF_PER_STB_PER_SM3_SM3,
    T_sc_petroleum,
    _V,
    _flash_sc,
    _gas_V,
    _scf_stb,
)
from ..models.component import Component
from ..properties.density import calculate_density, mixture_molecular_weight


####################
# DL
####################
@dataclass
class DLStepResult:
    """One DL step."""

    pressure: float
    temperature: float
    Rs: float
    Rs_scf_stb: float
    Bo: float
    oil_density: float
    gas_gravity: float
    gas_Z: float
    Bt: float
    Bg: Optional[float]
    Bg_rb_per_scf: Optional[float]
    liquid_composition: NDArray[np.float64]
    gas_composition: NDArray[np.float64]
    vapor_fraction: float
    cumulative_gas: float
    cumulative_gas_scf_stb: float
    liquid_moles_remaining: float


@dataclass
class DLResult:
    """Full DL result."""

    temperature: float
    bubble_pressure: float
    steps: list[DLStepResult]
    pressures: NDArray[np.float64]
    Rs_values: NDArray[np.float64]
    Bo_values: NDArray[np.float64]
    oil_densities: NDArray[np.float64]
    Bt_values: NDArray[np.float64]
    Rsi: float
    Rsi_scf_stb: float
    Boi: float
    residual_oil_density: float
    feed_composition: NDArray[np.float64]
    converged: bool


@dataclass
class _DLFlashRecord:
    pressure: float
    temperature: float
    oil_volume: float
    oil_density: float
    gas_volume_res: float
    gas_volume_std: float
    gas_gravity: float
    gas_Z: float
    liquid_composition: NDArray[np.float64]
    gas_composition: NDArray[np.float64]
    vapor_fraction: float
    liquid_moles_remaining: float


####################
# HELPER FUNCTIONS
####################
def _validate_dl_inputs(
    z: NDArray[np.float64],
    T: float,
    Pb: float,
    Ps: NDArray[np.float64],
    cs: list[Component],
) -> None:
    if T <= 0.0:
        raise ValidationError("Temperature must be positive", parameter="temperature")
    if Pb <= 0.0:
        raise ValidationError(
            "Bubble pressure must be positive", parameter="bubble_pressure"
        )
    if len(z) != len(cs):
        raise ValidationError(
            "Composition length must match number of components",
            parameter="composition",
        )
    if np.any(Ps <= 0.0):
        raise ValidationError(
            "All pressure steps must be positive", parameter="pressure_steps"
        )


def _dl_step_record(
    P: float,
    T: float,
    z: NDArray[np.float64],
    n: float,
    cs: list[Component],
    eos: CubicEOS,
    kij: Optional[NDArray[np.float64]],
    Psc: float,
    Tsc: float,
) -> tuple[_DLFlashRecord, NDArray[np.float64], float]:
    fl = pt_flash(P, T, z, cs, eos, binary_interaction=kij)

    if fl.phase == "liquid":
        rhoL = calculate_density(P, T, z, cs, eos, "liquid", kij)
        Vo = float(n) * rhoL.molar_volume
        return (
            _DLFlashRecord(
                pressure=P,
                temperature=T,
                oil_volume=Vo,
                oil_density=rhoL.mass_density,
                gas_volume_res=0.0,
                gas_volume_std=0.0,
                gas_gravity=np.nan,
                gas_Z=1.0,
                liquid_composition=z.copy(),
                gas_composition=np.zeros_like(z),
                vapor_fraction=0.0,
                liquid_moles_remaining=n,
            ),
            z.copy(),
            n,
        )

    nV = float(fl.vapor_fraction) * float(n)
    nL = (1.0 - float(fl.vapor_fraction)) * float(n)
    x = fl.liquid_composition
    y = fl.vapor_composition

    Vg_sc, _ = _gas_V(y, nV, Psc, Tsc, cs, eos, kij)

    rhoL = calculate_density(P, T, x, cs, eos, "liquid", kij)
    Vo = nL * rhoL.molar_volume

    MWg = mixture_molecular_weight(y, cs)
    gg = MWg / 28.97

    ZV = eos.compressibility(P, T, y, phase="vapor", binary_interaction=kij)
    if isinstance(ZV, list):
        ZV = ZV[-1]

    Vg = _V(nV, float(ZV), T, P)

    return (
        _DLFlashRecord(
            pressure=P,
            temperature=T,
            oil_volume=Vo,
            oil_density=rhoL.mass_density,
            gas_volume_res=Vg,
            gas_volume_std=Vg_sc,
            gas_gravity=gg,
            gas_Z=float(ZV),
            liquid_composition=x.copy(),
            gas_composition=y.copy(),
            vapor_fraction=float(fl.vapor_fraction),
            liquid_moles_remaining=nL,
        ),
        x.copy(),
        nL,
    )


def simulate_dl(
    composition: NDArray[np.float64],
    temperature: float,
    components: list[Component],
    eos: CubicEOS,
    bubble_pressure: float,
    pressure_steps: NDArray[np.float64],
    binary_interaction: Optional[NDArray[np.float64]] = None,
    standard_temperature: float = T_sc_petroleum,
    standard_pressure: float = P_sc,
) -> DLResult:
    """Run DL."""

    z = np.asarray(composition, dtype=np.float64)
    z = z / z.sum()
    T = float(temperature)
    Pb = float(bubble_pressure)
    Psc = float(standard_pressure)
    Tsc = float(standard_temperature)

    _validate_dl_inputs(z, T, Pb, pressure_steps, components)

    rhoL0 = calculate_density(Pb, T, z, components, eos, "liquid", binary_interaction)
    records: list[_DLFlashRecord] = [
        _DLFlashRecord(
            pressure=Pb,
            temperature=T,
            oil_volume=rhoL0.molar_volume,
            oil_density=rhoL0.mass_density,
            gas_volume_res=0.0,
            gas_volume_std=0.0,
            gas_gravity=np.nan,
            gas_Z=1.0,
            liquid_composition=z.copy(),
            gas_composition=np.zeros_like(z),
            vapor_fraction=0.0,
            liquid_moles_remaining=1.0,
        )
    ]

    x = z.copy()
    n = 1.0
    ok = True

    for P in pressure_steps:
        if P >= Pb:
            continue
        try:
            record, x, n = _dl_step_record(
                float(P),
                T,
                x,
                n,
                components,
                eos,
                binary_interaction,
                Psc,
                Tsc,
            )
            records.append(record)
        except (ConvergenceError, PhaseError):
            ok = False
            records.append(
                _DLFlashRecord(
                    pressure=float(P),
                    temperature=T,
                    oil_volume=np.nan,
                    oil_density=np.nan,
                    gas_volume_res=0.0,
                    gas_volume_std=0.0,
                    gas_gravity=np.nan,
                    gas_Z=np.nan,
                    liquid_composition=x.copy(),
                    gas_composition=np.zeros_like(z),
                    vapor_fraction=np.nan,
                    liquid_moles_remaining=n,
                )
            )

    stock = _flash_sc(x, n, components, eos, binary_interaction, Psc, Tsc)
    Vo_st = float(stock["Vo_st"])
    rho_o_sc = float(stock["rho_o"])
    if not np.isfinite(Vo_st) or Vo_st <= 0.0:
        raise PhaseError(
            "DL stock-tank flash produced non-positive residual oil volume.",
            phase="liquid",
            pressure=Psc,
            temperature=Tsc,
        )

    stock_y = np.asarray(stock["y"], dtype=np.float64)
    stock_x = np.asarray(stock["x"], dtype=np.float64)
    stock_nV = float(stock["nV"])
    stock_nL = float(stock["nL"])
    stock_Zg = float(stock["Zg_sc"])
    if not np.isfinite(stock_Zg):
        stock_Zg = 1.0
    stock_gg = (
        mixture_molecular_weight(stock_y, components) / 28.97
        if stock_nV > 0.0
        else np.nan
    )
    records.append(
        _DLFlashRecord(
            pressure=Psc,
            temperature=Tsc,
            oil_volume=Vo_st,
            oil_density=rho_o_sc,
            gas_volume_res=float(stock["Vg_sc"]),
            gas_volume_std=float(stock["Vg_sc"]),
            gas_gravity=stock_gg,
            gas_Z=stock_Zg,
            liquid_composition=stock_x.copy(),
            gas_composition=stock_y.copy(),
            vapor_fraction=(
                (stock_nV / (stock_nL + stock_nV)) if stock_nL + stock_nV > 0.0 else 0.0
            ),
            liquid_moles_remaining=stock_nL,
        )
    )

    total_std_gas = sum(record.gas_volume_std for record in records[1:])
    Rsi = total_std_gas / Vo_st
    Boi = records[0].oil_volume / Vo_st

    steps: list[DLStepResult] = []
    cumulative_std_gas = 0.0
    for index, record in enumerate(records):
        Bo = record.oil_volume / Vo_st if np.isfinite(record.oil_volume) else np.nan
        if index == 0:
            Rs = Rsi
            Bg = None
            Bt = Bo
            cumulative_gas = 0.0
        else:
            cumulative_std_gas += record.gas_volume_std
            cumulative_gas = cumulative_std_gas / Vo_st
            remaining_std_gas = max(0.0, total_std_gas - cumulative_std_gas)
            Rs = remaining_std_gas / Vo_st
            Bg = (
                record.gas_volume_res / record.gas_volume_std
                if index != len(records) - 1 and record.gas_volume_std > 0.0
                else None
            )
            Bt = Bo + Bg * (Rsi - Rs) if Bg is not None and np.isfinite(Bg) else Bo

        steps.append(
            DLStepResult(
                pressure=record.pressure,
                temperature=record.temperature,
                Rs=Rs,
                Rs_scf_stb=_scf_stb(Rs),
                Bo=Bo,
                oil_density=record.oil_density,
                gas_gravity=record.gas_gravity,
                gas_Z=record.gas_Z,
                Bt=Bt,
                Bg=Bg,
                Bg_rb_per_scf=(
                    (Bg / SCF_PER_STB_PER_SM3_SM3) if Bg is not None else None
                ),
                liquid_composition=record.liquid_composition.copy(),
                gas_composition=record.gas_composition.copy(),
                vapor_fraction=record.vapor_fraction,
                cumulative_gas=cumulative_gas,
                cumulative_gas_scf_stb=_scf_stb(cumulative_gas),
                liquid_moles_remaining=record.liquid_moles_remaining,
            )
        )

    return DLResult(
        temperature=T,
        bubble_pressure=Pb,
        steps=steps,
        pressures=np.array([s.pressure for s in steps]),
        Rs_values=np.array([s.Rs for s in steps]),
        Bo_values=np.array([s.Bo for s in steps]),
        oil_densities=np.array([s.oil_density for s in steps]),
        Bt_values=np.array([s.Bt for s in steps]),
        Rsi=Rsi,
        Rsi_scf_stb=_scf_stb(Rsi),
        Boi=Boi,
        residual_oil_density=rho_o_sc,
        feed_composition=z,
        converged=ok,
    )

"""Transport and interfacial properties for petroleum fluids.

This module provides calculations for:
- Density (from EOS with optional volume translation)
- Viscosity (Lohrenz-Bray-Clark correlation)
- Interfacial tension (Parachor/Weinaug-Katz method)

These properties are typically calculated after a flash calculation
determines the phase compositions and amounts.

Example
-------
>>> from pvtcore.properties import (
...     calculate_density,
...     calculate_viscosity_lbc,
...     calculate_ift_parachor,
... )
>>> from pvtcore.models.component import load_components
>>> from pvtcore.eos.peng_robinson import PengRobinsonEOS
>>> import numpy as np
>>>
>>> # Setup
>>> components = load_components()
>>> binary = [components['C1'], components['C3']]
>>> eos = PengRobinsonEOS(binary)
>>>
>>> # Conditions (after flash)
>>> P, T = 5e6, 300.0  # 5 MPa, 300 K
>>> x = np.array([0.3, 0.7])  # Liquid composition
>>> y = np.array([0.9, 0.1])  # Vapor composition
>>>
>>> # Density
>>> rho_L = calculate_density(P, T, x, binary, eos, phase='liquid')
>>> rho_V = calculate_density(P, T, y, binary, eos, phase='vapor')
>>> print(f"Liquid: {rho_L.mass_density:.1f} kg/m³")
>>> print(f"Vapor: {rho_V.mass_density:.1f} kg/m³")
>>>
>>> # Viscosity
>>> mu_L = calculate_viscosity_lbc(rho_L.molar_density, T, x, binary)
>>> mu_V = calculate_viscosity_lbc(rho_V.molar_density, T, y, binary)
>>> print(f"Liquid viscosity: {mu_L.viscosity_cp:.4f} cp")
>>> print(f"Vapor viscosity: {mu_V.viscosity_cp:.4f} cp")
>>>
>>> # IFT
>>> ift = calculate_ift_parachor(x, y, rho_L.molar_density, rho_V.molar_density, binary)
>>> print(f"IFT: {ift.ift:.2f} mN/m")
"""

from .density import (
    calculate_density,
    calculate_phase_densities,
    mixture_molecular_weight,
    estimate_volume_shift_peneloux,
    DensityResult,
)

from .viscosity_lbc import (
    calculate_viscosity_lbc,
    calculate_phase_viscosities,
    ViscosityResult,
)

from .ift_parachor import (
    calculate_ift_parachor,
    calculate_ift_from_mass_density,
    estimate_critical_ift_scaling,
    IFTResult,
)

__all__ = [
    # Density
    'calculate_density',
    'calculate_phase_densities',
    'mixture_molecular_weight',
    'estimate_volume_shift_peneloux',
    'DensityResult',
    # Viscosity
    'calculate_viscosity_lbc',
    'calculate_phase_viscosities',
    'ViscosityResult',
    # IFT
    'calculate_ift_parachor',
    'calculate_ift_from_mass_density',
    'estimate_critical_ift_scaling',
    'IFTResult',
]

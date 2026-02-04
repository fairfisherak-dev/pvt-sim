"""Nano-confinement phase behavior modeling.

This module provides calculations for phase behavior in nano-confined systems
such as tight/shale reservoirs. Capillary pressure in nanometer-scale pores
creates significant pressure differences between liquid and vapor phases,
leading to:

- Suppressed bubble points (lower pressure)
- Enhanced dew points (higher pressure)
- Shrinkage of the two-phase region
- Shifted critical point

This is the distinguishing feature of PVT-SIM, providing functionality
that is lacking in most commercial PVT software.

Key Equations
-------------
Capillary pressure (Young-Laplace):
    Pc = 2σ cos(θ) / r

Modified equilibrium condition:
    xi φi^L(PL) PL = yi φi^V(Pv) Pv

where:
    Pv = PL + Pc (liquid is wetting phase)

Modified K-values:
    Ki = (φi^L / φi^V) × (PL / Pv)

Coupling Loop
-------------
1. Start with bulk flash (Pv = PL)
2. Calculate IFT from parachor correlation
3. Calculate Pc = 2σ/r
4. Update Pv = PL + Pc
5. Re-run flash with modified K-values
6. Repeat until Pc converges

Example
-------
>>> from pvtcore.confinement import (
...     confined_flash,
...     calculate_confined_envelope,
...     capillary_pressure_simple,
... )
>>> from pvtcore.models.component import load_components
>>> from pvtcore.eos.peng_robinson import PengRobinsonEOS
>>> import numpy as np
>>>
>>> # Setup
>>> components = load_components()
>>> binary = [components['C1'], components['C4']]
>>> eos = PengRobinsonEOS(binary)
>>> z = np.array([0.7, 0.3])
>>>
>>> # Confined flash in 10 nm pore
>>> result = confined_flash(5e6, 350.0, z, binary, eos, pore_radius_nm=10.0)
>>> print(f"Capillary pressure: {result.capillary_pressure/1e6:.2f} MPa")
>>> print(f"Vapor fraction: {result.vapor_fraction:.3f}")
>>>
>>> # Compare bulk vs confined envelope
>>> envelope = calculate_confined_envelope(z, binary, eos, pore_radius_nm=10.0)
>>> print(f"Bulk critical T: {envelope.bulk_envelope.critical_T:.0f} K")
>>> print(f"Confined critical T: {envelope.critical_T:.0f} K")

References
----------
[1] Nojabaei, B., Johns, R.T., and Chu, L. (2013).
    "Effect of Capillary Pressure on Phase Behavior in Tight Rocks and Shales."
    SPE Reservoir Evaluation & Engineering, 16(3), 281-289. SPE-159258.
"""

from .capillary import (
    calculate_capillary_pressure,
    capillary_pressure_simple,
    vapor_pressure_from_liquid,
    liquid_pressure_from_vapor,
    modified_k_value,
    modified_k_values_array,
    estimate_bubble_point_suppression,
    estimate_dew_point_enhancement,
    critical_pore_radius,
    CapillaryPressureResult,
)

from .confined_flash import (
    confined_flash,
    confined_bubble_point,
    confined_dew_point,
    ConfinedFlashResult,
)

from .confined_envelope import (
    calculate_confined_envelope,
    estimate_envelope_shrinkage,
    compare_bulk_confined,
    ConfinedEnvelopeResult,
)

__all__ = [
    # Capillary pressure
    'calculate_capillary_pressure',
    'capillary_pressure_simple',
    'vapor_pressure_from_liquid',
    'liquid_pressure_from_vapor',
    'modified_k_value',
    'modified_k_values_array',
    'estimate_bubble_point_suppression',
    'estimate_dew_point_enhancement',
    'critical_pore_radius',
    'CapillaryPressureResult',
    # Confined flash
    'confined_flash',
    'confined_bubble_point',
    'confined_dew_point',
    'ConfinedFlashResult',
    # Confined envelope
    'calculate_confined_envelope',
    'estimate_envelope_shrinkage',
    'compare_bulk_confined',
    'ConfinedEnvelopeResult',
]

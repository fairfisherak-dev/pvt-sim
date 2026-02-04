"""Phase envelope calculations."""

from .phase_envelope import (
    calculate_phase_envelope,
    EnvelopeResult,
    estimate_cricondentherm,
    estimate_cricondenbar
)

from .critical_point import (
    detect_critical_point,
    estimate_critical_point_kays,
    find_critical_from_envelope,
    CriticalPointResult,
)

__all__ = [
    'calculate_phase_envelope',
    'EnvelopeResult',
    'estimate_cricondentherm',
    'estimate_cricondenbar',
    'detect_critical_point',
    'estimate_critical_point_kays',
    'find_critical_from_envelope',
    'CriticalPointResult',
]

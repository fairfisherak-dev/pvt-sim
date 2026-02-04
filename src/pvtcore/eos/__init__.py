"""Equation of State module for PVT calculations."""

from .base import CubicEOS, EOSResult
from .peng_robinson import PengRobinsonEOS
from .ppr78 import (
    BIPProvider,
    PPR78Calculator,
    PPR78Result,
    PPR78_INTERACTION_PARAMS,
    PPR78_T_REF,
    StaticBIPProvider,
)

__all__ = [
    # Base EOS
    'CubicEOS',
    'EOSResult',
    'PengRobinsonEOS',
    # PPR78 predictive BIP
    'BIPProvider',
    'PPR78Calculator',
    'PPR78Result',
    'PPR78_INTERACTION_PARAMS',
    'PPR78_T_REF',
    'StaticBIPProvider',
]

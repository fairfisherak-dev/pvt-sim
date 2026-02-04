"""
Plus-fraction splitting methods for petroleum characterization.

This module provides methods for splitting C7+ (or other plus fractions)
into individual Single Carbon Number (SCN) groups with associated
mole fractions and molecular weights.

Available methods:
- Pedersen (1984): Exponential distribution with material balance constraints
- Katz (1983): Simple exponential distribution
- Lohrenz (1964): Quadratic-exponential distribution

Usage
-----
>>> from pvtcore.characterization.plus_splitting import split_plus_fraction_pedersen
>>> result = split_plus_fraction_pedersen(z_plus=0.25, MW_plus=215.0)
>>> print(result.z)  # Mole fractions for C7, C8, ..., C45
"""

from .pedersen import (
    PedersenSplitResult,
    split_plus_fraction_pedersen,
)

from .katz import (
    KatzSplitResult,
    split_plus_fraction_katz,
    katz_classic_split,
)

from .lohrenz import (
    LohrenzSplitResult,
    split_plus_fraction_lohrenz,
    lohrenz_classic_coefficients,
)

__all__ = [
    # Pedersen
    "PedersenSplitResult",
    "split_plus_fraction_pedersen",
    # Katz
    "KatzSplitResult",
    "split_plus_fraction_katz",
    "katz_classic_split",
    # Lohrenz
    "LohrenzSplitResult",
    "split_plus_fraction_lohrenz",
    "lohrenz_classic_coefficients",
]

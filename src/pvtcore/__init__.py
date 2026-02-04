"""pvtcore: thermodynamic kernel for the PVT simulator.

This package is intentionally backend-only:
- characterization (plus fractions, pseudoization)
- EOS (currently PR) + fugacity
- stability (TPD)
- flash + saturation + envelopes

Public APIs will be stabilized incrementally; for now, treat submodules as semi-internal.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("pvt-simulator")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

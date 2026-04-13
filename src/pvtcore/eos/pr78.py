"""Peng-Robinson (1978) heavy-end extended EOS implementation."""

from __future__ import annotations

from typing import List

from ..models.component import Component
from .peng_robinson import PengRobinsonEOS


class PR78EOS(PengRobinsonEOS):
    """Peng-Robinson (1978) EOS variant with heavy-end kappa extension.

    This keeps the Peng-Robinson cubic form and mixing rules, but swaps the
    classic 1976 alpha correlation for the 1978 GPA-program heavy-fraction
    extension:

        κ = 0.379642 + 1.48503ω - 0.164423ω² + 0.016666ω³
    """

    def __init__(self, components: List[Component]):
        super().__init__(components)
        self.name = "Peng-Robinson (1978)"

    def _kappa_from_omega(self, omega: float) -> float:
        """Return the PR78 heavy-end extended kappa correlation."""
        return (
            0.379642
            + 1.48503 * omega
            - 0.164423 * omega ** 2
            + 0.016666 * omega ** 3
        )

    def __repr__(self) -> str:
        return f"PR78EOS(n_components={self.n_components})"

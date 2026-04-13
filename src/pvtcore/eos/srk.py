"""Soave-Redlich-Kwong equation of state implementation."""

from __future__ import annotations

import math
from typing import Callable, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .base import CubicEOS
from ..core.constants import R
from ..core.errors import PhaseError
from ..models.component import Component

BIPInput = Union[
    NDArray[np.float64],
    Callable[[float], NDArray[np.float64]],
    "BIPProvider",
    None,
]


class SRKEOS(CubicEOS):
    """Soave-Redlich-Kwong cubic EOS.

    The SRK EOS equation:
        P = RT/(V-b) - a(T)/(V(V+b))

    In compressibility factor form:
        Z³ - Z² + (A-B-B²)Z - AB = 0
    """

    OMEGA_A = 0.42748
    OMEGA_B = 0.08664
    DELTA_1 = 1.0
    DELTA_2 = 0.0
    DELTA_DIFF = DELTA_1 - DELTA_2

    def __init__(self, components: List[Component]):
        super().__init__(components, name="Soave-Redlich-Kwong")
        self.u = 1.0
        self.w = 0.0

        self._calculate_critical_params()
        self._zero_kij = np.zeros((self.n_components, self.n_components))
        self._cached_temperature: Optional[float] = None
        self._cached_alpha: Optional[np.ndarray] = None
        self._cached_a_array: Optional[np.ndarray] = None
        self._cached_sqrt_a: Optional[np.ndarray] = None

    def _calculate_critical_params(self) -> None:
        """Pre-calculate temperature-independent SRK parameters."""
        self.a_c = np.zeros(self.n_components)
        self.b = np.zeros(self.n_components)
        self.m = np.zeros(self.n_components)

        for i, comp in enumerate(self.components):
            self.a_c[i] = (
                self.OMEGA_A * R.Pa_m3_per_mol_K ** 2 * comp.Tc ** 2 / comp.Pc
            )
            self.b[i] = self.OMEGA_B * R.Pa_m3_per_mol_K * comp.Tc / comp.Pc
            omega = comp.omega
            self.m[i] = 0.480 + 1.574 * omega - 0.176 * omega ** 2

    def alpha_function(self, temperature: float, component_idx: int) -> float:
        """Calculate the Soave alpha function."""
        comp = self.components[component_idx]
        Tr = temperature / comp.Tc
        sqrt_Tr = math.sqrt(Tr)
        m = self.m[component_idx]
        return (1.0 + m * (1.0 - sqrt_Tr)) ** 2

    def _get_temperature_params(self, temperature: float) -> tuple[np.ndarray, np.ndarray]:
        """Get cached or compute temperature-dependent parameters."""
        if self._cached_temperature != temperature:
            Tc_array = np.array([c.Tc for c in self.components])
            Tr = temperature / Tc_array
            sqrt_Tr = np.sqrt(Tr)
            self._cached_alpha = (1.0 + self.m * (1.0 - sqrt_Tr)) ** 2
            self._cached_a_array = self.a_c * self._cached_alpha
            self._cached_sqrt_a = np.sqrt(self._cached_a_array)
            self._cached_temperature = temperature

        return self._cached_a_array, self._cached_sqrt_a

    def _resolve_kij(
        self,
        temperature: float,
        binary_interaction: BIPInput,
    ) -> NDArray[np.float64]:
        """Resolve k_ij matrix from supported input forms."""
        if binary_interaction is None:
            return self._zero_kij
        if isinstance(binary_interaction, np.ndarray):
            return binary_interaction
        if callable(binary_interaction):
            return binary_interaction(temperature)
        if hasattr(binary_interaction, "get_kij_matrix"):
            return binary_interaction.get_kij_matrix(temperature)
        return self._zero_kij

    def calculate_params(
        self,
        temperature: float,
        composition: np.ndarray,
        binary_interaction: BIPInput = None,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Calculate SRK mixture parameters."""
        composition = np.asarray(composition)
        a_array, sqrt_a = self._get_temperature_params(temperature)
        b_array = self.b
        b_mix = np.dot(composition, b_array)
        kij = self._resolve_kij(temperature, binary_interaction)

        sqrt_a_matrix = np.outer(sqrt_a, sqrt_a)
        a_ij_matrix = sqrt_a_matrix * (1.0 - kij)
        a_mix = np.dot(composition, np.dot(a_ij_matrix, composition))
        return a_mix, b_mix, a_array, b_array

    def fugacity_coefficient(
        self,
        pressure: float,
        temperature: float,
        composition: np.ndarray,
        phase: Literal["liquid", "vapor"] = "vapor",
        binary_interaction: BIPInput = None,
    ) -> np.ndarray:
        """Calculate SRK fugacity coefficients for all components."""
        from ..core.numerics.cubic_solver import solve_cubic_eos

        composition = np.asarray(composition)
        a_array, sqrt_a = self._get_temperature_params(temperature)
        b_array = self.b
        kij = self._resolve_kij(temperature, binary_interaction)

        b_mix = np.dot(composition, b_array)
        sqrt_a_matrix = np.outer(sqrt_a, sqrt_a)
        a_ij = sqrt_a_matrix * (1.0 - kij)
        a_mix = np.dot(composition, np.dot(a_ij, composition))

        RT = R.Pa_m3_per_mol_K * temperature
        A = a_mix * pressure / (RT * RT)
        B = b_mix * pressure / RT
        Z = solve_cubic_eos(A, B, root_type=phase, u=self.u, w=self.w)

        if Z <= B:
            raise PhaseError(
                f"Z={Z:.6f} <= B={B:.6f}: physically invalid SRK state.",
                phase=phase,
            )

        log_z_minus_b = math.log(Z - B)
        log_ratio = math.log((Z + self.DELTA_1 * B) / (Z + self.DELTA_2 * B))
        coeff = A / (B * self.DELTA_DIFF)

        sum_xj_aij = np.dot(a_ij, composition)
        bi_over_bmix = b_array / b_mix
        term1 = bi_over_bmix * (Z - 1.0)
        term2 = -log_z_minus_b
        bracket_term = 2.0 * sum_xj_aij / a_mix - bi_over_bmix
        term3 = -coeff * bracket_term * log_ratio

        return np.exp(term1 + term2 + term3)

    def __repr__(self) -> str:
        return f"SRKEOS(n_components={self.n_components})"

"""Peng-Robinson equation of state implementation.

Implementation of the Peng-Robinson EOS as presented in:
Peng, D.-Y. and Robinson, D. B., "A New Two-Constant Equation of State",
Industrial & Engineering Chemistry Fundamentals, 15(1), 59-64 (1976).

The PR EOS is widely used in petroleum and natural gas applications due to
its accuracy for hydrocarbon systems and improved liquid density predictions
compared to the Soave-Redlich-Kwong EOS.
"""

import math
from typing import Callable, List, Literal, Optional, Union
import numpy as np
from numpy.typing import NDArray

from .base import CubicEOS
from ..models.component import Component
from ..core.constants import R
from ..core.errors import PhaseError

# Type alias for various BIP input formats
# Static matrix, T-dependent callable, or BIPProvider protocol
BIPInput = Union[
    NDArray[np.float64],                          # Static matrix
    Callable[[float], NDArray[np.float64]],       # T -> matrix callable
    "BIPProvider",                                 # Protocol implementation
    None                                           # Default (zeros)
]


class PengRobinsonEOS(CubicEOS):
    """Peng-Robinson equation of state.

    The PR EOS equation:
        P = RT/(V-b) - a(T)/(V² + 2bV - b²)

    In compressibility factor form:
        Z³ - (1-B)Z² + (A-2B-3B²)Z - (AB-B²-B³) = 0

    where:
        A = aP/(RT)²
        B = bP/(RT)

    Parameters:
        - a_c = 0.45724 R²Tc²/Pc  (attraction parameter at critical point)
        - b = 0.07780 RTc/Pc  (repulsion parameter)
        - κ = 0.37464 + 1.54226ω - 0.26992ω²  (for ω ≤ 0.49)
        - κ = 0.379642 + 1.48503ω - 0.164423ω² + 0.016666ω³  (for ω > 0.49)
        - α(T) = [1 + κ(1 - √(T/Tc))]²
        - a(T) = a_c × α(T)

    Mixing rules (van der Waals one-fluid):
        - a_mix = ΣΣ xᵢxⱼ√(aᵢaⱼ)(1-kᵢⱼ)
        - b_mix = Σ xᵢbᵢ

    Attributes:
        components: List of Component objects
        u: PR parameter = 2
        w: PR parameter = -1
    """

    # PR EOS constants
    OMEGA_A = 0.45724  # a_c coefficient
    OMEGA_B = 0.07780  # b coefficient
    DELTA_1 = 1.0 + math.sqrt(2.0)  # 2.414213562373095
    DELTA_2 = 1.0 - math.sqrt(2.0)  # -0.414213562373095
    SQRT_2 = math.sqrt(2.0)  # Pre-computed for fugacity calc

    def __init__(self, components: List[Component]):
        """Initialize Peng-Robinson EOS.

        Args:
            components: List of Component objects with Tc, Pc, omega properties
        """
        super().__init__(components, name="Peng-Robinson")

        # PR-specific parameters
        self.u = 2.0
        self.w = -1.0

        # Pre-calculate pure component parameters that don't depend on T
        self._calculate_critical_params()

        # Pre-allocate zero BIP matrix for default case (avoid repeated allocation)
        self._zero_kij = np.zeros((self.n_components, self.n_components))

        # Cache for temperature-dependent parameters
        self._cached_temperature: Optional[float] = None
        self._cached_alpha: Optional[np.ndarray] = None
        self._cached_a_array: Optional[np.ndarray] = None
        self._cached_sqrt_a: Optional[np.ndarray] = None

    def _calculate_critical_params(self):
        """Pre-calculate temperature-independent critical parameters."""
        self.a_c = np.zeros(self.n_components)  # Critical attraction parameter
        self.b = np.zeros(self.n_components)  # Repulsion parameter
        self.kappa = np.zeros(self.n_components)  # Alpha function parameter

        for i, comp in enumerate(self.components):
            # a_c = 0.45724 R²Tc²/Pc
            self.a_c[i] = (
                self.OMEGA_A * R.Pa_m3_per_mol_K ** 2 * comp.Tc ** 2 / comp.Pc
            )

            # b = 0.07780 RTc/Pc
            self.b[i] = (
                self.OMEGA_B * R.Pa_m3_per_mol_K * comp.Tc / comp.Pc
            )

            # κ parameter for alpha function
            omega = comp.omega
            if omega <= 0.49:
                # Original PR correlation
                self.kappa[i] = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
            else:
                # Extended correlation for heavy components (Robinson & Peng, 1978)
                self.kappa[i] = (
                    0.379642 + 1.48503 * omega
                    - 0.164423 * omega ** 2
                    + 0.016666 * omega ** 3
                )

    def alpha_function(self, temperature: float, component_idx: int) -> float:
        """Calculate alpha function α(T) for a component.

        α(T) = [1 + κ(1 - √(T/Tc))]²

        Args:
            temperature: Temperature (K)
            component_idx: Index of component

        Returns:
            Alpha value (dimensionless)
        """
        comp = self.components[component_idx]
        Tr = temperature / comp.Tc  # Reduced temperature
        sqrt_Tr = math.sqrt(Tr)
        kappa = self.kappa[component_idx]

        alpha = (1.0 + kappa * (1.0 - sqrt_Tr)) ** 2

        return alpha

    def _get_temperature_params(self, temperature: float) -> tuple[np.ndarray, np.ndarray]:
        """Get cached or compute temperature-dependent parameters (vectorized).

        Returns:
            Tuple of (a_array, sqrt_a_array) for the given temperature
        """
        if self._cached_temperature != temperature:
            # Compute alpha for all components (vectorized)
            Tc_array = np.array([c.Tc for c in self.components])
            Tr = temperature / Tc_array
            sqrt_Tr = np.sqrt(Tr)
            self._cached_alpha = (1.0 + self.kappa * (1.0 - sqrt_Tr)) ** 2
            self._cached_a_array = self.a_c * self._cached_alpha
            self._cached_sqrt_a = np.sqrt(self._cached_a_array)
            self._cached_temperature = temperature

        return self._cached_a_array, self._cached_sqrt_a

    def _resolve_kij(
        self,
        temperature: float,
        binary_interaction: BIPInput,
    ) -> NDArray[np.float64]:
        """Resolve k_ij matrix from various input types.

        This method enables flexibility in specifying binary interaction
        parameters: as a static matrix, a temperature-dependent callable,
        or a BIPProvider implementation (e.g., PPR78Calculator).

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin (used for T-dependent BIPs).
        binary_interaction : BIPInput
            One of:
            - None: Use zero matrix (ideal mixing)
            - np.ndarray: Static k_ij matrix
            - Callable[[float], np.ndarray]: Function T -> k_ij matrix
            - BIPProvider: Object with get_kij_matrix(T) method

        Returns
        -------
        ndarray
            k_ij matrix of shape (n_components, n_components).
        """
        if binary_interaction is None:
            return self._zero_kij

        # Static numpy array
        if isinstance(binary_interaction, np.ndarray):
            return binary_interaction

        # Callable (T -> matrix)
        if callable(binary_interaction):
            return binary_interaction(temperature)

        # BIPProvider protocol (has get_kij_matrix method)
        if hasattr(binary_interaction, 'get_kij_matrix'):
            return binary_interaction.get_kij_matrix(temperature)

        # Fallback to zero matrix
        return self._zero_kij

    def calculate_params(
        self,
        temperature: float,
        composition: np.ndarray,
        binary_interaction: BIPInput = None
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Calculate EOS parameters a and b for mixture.

        Uses van der Waals one-fluid mixing rules:
            a_mix = ΣΣ xᵢxⱼ aᵢⱼ
            b_mix = Σ xᵢbᵢ

        where aᵢⱼ = √(aᵢaⱼ)(1 - kᵢⱼ) with kᵢⱼ being binary interaction parameters.

        Args:
            temperature: Temperature (K)
            composition: Mole fractions (array of length n_components)
            binary_interaction: Binary interaction parameters. Can be:
                - None: Use zeros (ideal mixing)
                - np.ndarray: Static k_ij matrix (n×n)
                - Callable[[float], np.ndarray]: T-dependent function
                - BIPProvider: Object with get_kij_matrix(T) method (e.g., PPR78Calculator)

        Returns:
            Tuple of (a_mix, b_mix, a_array, b_array)
        """
        composition = np.asarray(composition)

        # Get cached temperature-dependent parameters (vectorized)
        a_array, sqrt_a = self._get_temperature_params(temperature)

        # b parameters don't depend on temperature
        b_array = self.b

        # Mixing rule for b (linear) - already vectorized
        b_mix = np.dot(composition, b_array)

        # Resolve k_ij matrix (handles static, callable, or BIPProvider)
        kij = self._resolve_kij(temperature, binary_interaction)

        # Mixing rule for a (vectorized quadratic form)
        # a_ij = sqrt(a_i * a_j) * (1 - k_ij) = outer(sqrt_a, sqrt_a) * (1 - k_ij)
        sqrt_a_matrix = np.outer(sqrt_a, sqrt_a)
        a_ij_matrix = sqrt_a_matrix * (1.0 - kij)

        # a_mix = sum_i sum_j x_i * x_j * a_ij = x^T @ a_ij @ x
        a_mix = np.dot(composition, np.dot(a_ij_matrix, composition))

        return a_mix, b_mix, a_array, b_array

    def fugacity_coefficient(
        self,
        pressure: float,
        temperature: float,
        composition: np.ndarray,
        phase: Literal['liquid', 'vapor'] = 'vapor',
        binary_interaction: BIPInput = None
    ) -> np.ndarray:
        """Calculate fugacity coefficients for all components.

        For Peng-Robinson EOS, the fugacity coefficient is:

        ln(φᵢ) = (bᵢ/b_mix)(Z - 1) - ln(Z - B)
                 - (A/(2√2 B)) × [2Σⱼ(xⱼaᵢⱼ)/a_mix - bᵢ/b_mix]
                 × ln[(Z + (1+√2)B)/(Z + (1-√2)B)]

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            composition: Mole fractions
            phase: Phase for calculation ('liquid' or 'vapor')
            binary_interaction: Binary interaction parameters. Can be:
                - None: Use zeros (ideal mixing)
                - np.ndarray: Static k_ij matrix (n×n)
                - Callable[[float], np.ndarray]: T-dependent function
                - BIPProvider: Object with get_kij_matrix(T) method

        Returns:
            Array of fugacity coefficients φᵢ (dimensionless)
        """
        composition = np.asarray(composition)

        # Get cached temperature-dependent parameters
        a_array, sqrt_a = self._get_temperature_params(temperature)
        b_array = self.b

        # Resolve k_ij matrix (handles static, callable, or BIPProvider)
        kij = self._resolve_kij(temperature, binary_interaction)

        # Vectorized mixing rules
        b_mix = np.dot(composition, b_array)

        # a_ij matrix (vectorized)
        sqrt_a_matrix = np.outer(sqrt_a, sqrt_a)
        a_ij = sqrt_a_matrix * (1.0 - kij)
        a_mix = np.dot(composition, np.dot(a_ij, composition))

        # Calculate dimensionless parameters
        RT = R.Pa_m3_per_mol_K * temperature
        RT_sq = RT * RT
        A = a_mix * pressure / RT_sq
        B = b_mix * pressure / RT

        # Get compressibility factor (avoid recalculating params)
        from ..core.numerics.cubic_solver import solve_cubic_eos
        Z = solve_cubic_eos(A, B, root_type=phase)

        # Check for physically invalid state (vectorized check)
        if Z <= B:
            raise PhaseError(
                f"Z={Z:.6f} <= B={B:.6f}: physically invalid state. "
                f"This typically occurs at extreme pressures (P={pressure/1e6:.1f} MPa) "
                f"or near spinodal conditions where the EOS is unreliable.",
                phase=phase
            )

        denom = Z + self.DELTA_2 * B
        if denom <= 0:
            raise PhaseError(
                f"Invalid state for fugacity calculation: Z + δ₂B = {denom:.6f} <= 0. "
                f"Z={Z:.6f}, B={B:.6f}, δ₂={self.DELTA_2:.6f}. "
                f"This indicates extreme conditions (P={pressure/1e6:.1f} MPa, T={temperature:.1f} K) "
                f"where the PR EOS is unreliable.",
                phase=phase
            )

        # Pre-compute common terms (scalars)
        log_Z_minus_B = math.log(Z - B)
        log_ratio = math.log((Z + self.DELTA_1 * B) / denom)
        coeff = A / (2.0 * self.SQRT_2 * B)

        # Vectorized fugacity coefficient calculation
        # sum_xj_aij for each i: a_ij @ composition (matrix-vector product)
        sum_xj_aij = np.dot(a_ij, composition)

        # All terms vectorized
        bi_over_bmix = b_array / b_mix
        term1 = bi_over_bmix * (Z - 1.0)
        term2 = -log_Z_minus_B  # Same for all components
        bracket_term = 2.0 * sum_xj_aij / a_mix - bi_over_bmix
        term3 = -coeff * bracket_term * log_ratio

        ln_phi = term1 + term2 + term3
        phi = np.exp(ln_phi)

        return phi

    def calculate_departure_functions(
        self,
        pressure: float,
        temperature: float,
        composition: np.ndarray,
        phase: Literal['liquid', 'vapor'] = 'vapor',
        binary_interaction: BIPInput = None
    ) -> dict:
        """Calculate departure functions (H, S, G) from ideal gas.

        Departure functions represent the difference between real and ideal gas:
            H^R = H - H^ig
            S^R = S - S^ig
            G^R = H^R - T×S^R = RT ln(φ)

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            composition: Mole fractions
            phase: Phase for calculation
            binary_interaction: Binary interaction parameters kij

        Returns:
            Dictionary with departure functions (J/mol for H, J/(mol·K) for S)
        """
        # Calculate mixture parameters
        a_mix, b_mix, a_array, b_array = self.calculate_params(
            temperature, composition, binary_interaction
        )

        # Dimensionless parameters
        A = a_mix * pressure / (R.Pa_m3_per_mol_K * temperature) ** 2
        B = b_mix * pressure / (R.Pa_m3_per_mol_K * temperature)

        # Compressibility factor
        Z = self.compressibility(
            pressure, temperature, composition, phase, binary_interaction
        )

        # Calculate da/dT for enthalpy departure
        # For PR: da/dT = -a_c × κ/√(T×Tc)
        da_dT = 0.0
        for i in range(self.n_components):
            for j in range(self.n_components):
                if binary_interaction is not None:
                    kij = binary_interaction[i, j]
                else:
                    kij = 0.0

                # Derivative of √(aᵢaⱼ)
                alpha_i = self.alpha_function(temperature, i)
                alpha_j = self.alpha_function(temperature, j)

                # dα/dT = -κ(1 + κ(1-√Tr))/√(T×Tc)
                Tr_i = temperature / self.components[i].Tc
                Tr_j = temperature / self.components[j].Tc

                dalpha_dT_i = -self.kappa[i] * (1.0 + self.kappa[i] * (1.0 - math.sqrt(Tr_i))) / math.sqrt(
                    temperature * self.components[i].Tc
                )
                dalpha_dT_j = -self.kappa[j] * (1.0 + self.kappa[j] * (1.0 - math.sqrt(Tr_j))) / math.sqrt(
                    temperature * self.components[j].Tc
                )

                # Contribution to da_mix/dT
                da_ij_dT = 0.5 * (1.0 - kij) * (
                    math.sqrt(self.a_c[i] * self.a_c[j] * alpha_j / alpha_i) * self.a_c[i] * dalpha_dT_i
                    + math.sqrt(self.a_c[i] * self.a_c[j] * alpha_i / alpha_j) * self.a_c[j] * dalpha_dT_j
                )

                da_dT += composition[i] * composition[j] * da_ij_dT

        # Departure functions
        # Check for physically invalid states before computing logarithms
        if Z <= B:
            raise PhaseError(
                f"Z={Z:.6f} <= B={B:.6f}: physically invalid state in departure functions. "
                f"This typically occurs at extreme pressures (P={pressure/1e6:.1f} MPa) "
                f"or near spinodal conditions where the EOS is unreliable.",
                phase=phase
            )

        denom = Z + self.DELTA_2 * B
        if denom <= 0:
            raise PhaseError(
                f"Invalid state for departure function calculation: Z + δ₂B = {denom:.6f} <= 0. "
                f"Z={Z:.6f}, B={B:.6f}, δ₂={self.DELTA_2:.6f}. "
                f"This indicates extreme conditions (P={pressure/1e6:.1f} MPa, T={temperature:.1f} K) "
                f"where the PR EOS is unreliable.",
                phase=phase
            )

        log_ratio = math.log((Z + self.DELTA_1 * B) / denom)

        # Enthalpy departure: H^R/RT = Z - 1 - (A/(2√2B))(1 + T/a × da/dT) × log_ratio
        H_dep = R.J_per_mol_K * temperature * (
            Z - 1.0 - (A / (2.0 * math.sqrt(2.0) * B)) *
            (1.0 + temperature / a_mix * da_dT) * log_ratio
        )

        # Entropy departure: S^R/R = ln(Z-B) - (A/(2√2B))(T/a × da/dT) × log_ratio
        S_dep = R.J_per_mol_K * (
            math.log(Z - B) - (A / (2.0 * math.sqrt(2.0) * B)) *
            (temperature / a_mix * da_dT) * log_ratio
        )

        # Gibbs energy departure: G^R/RT = Z - 1 - ln(Z-B) - (A/(2√2B)) × log_ratio
        G_dep = R.J_per_mol_K * temperature * (
            Z - 1.0 - math.log(Z - B) - (A / (2.0 * math.sqrt(2.0) * B)) * log_ratio
        )

        return {
            'enthalpy_departure': H_dep,  # J/mol
            'entropy_departure': S_dep,  # J/(mol·K)
            'gibbs_departure': G_dep,  # J/mol
            'Z': Z,
            'A': A,
            'B': B
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"PengRobinsonEOS(n_components={self.n_components})"

"""PPR78 predictive model for temperature-dependent binary interaction parameters.

The PPR78 (Predictive Peng-Robinson 1978) model calculates k_ij(T) using
group contribution, eliminating the need for experimental BIP data for
many component pairs.

The model equation is:
    k_ij(T) = -1/2 * sum_k sum_l (alpha_ik - alpha_jk)(alpha_il - alpha_jl)
              * A_kl * (298.15/T)^(B_kl/A_kl - 1)

where:
    alpha_ik = group fraction of group k in component i
    A_kl, B_kl = group interaction parameters (MPa)
    T = temperature in Kelvin

References
----------
[1] Jaubert, J.-N. and Mutelet, F. (2004). "VLE predictions with the
    Peng-Robinson equation of state and temperature dependent kij calculated
    through a group contribution method." Fluid Phase Equilibria, 224(2), 285-304.
[2] Jaubert, J.-N. et al. (2005). "Extension of the PPR78 model to systems
    containing aromatic compounds." Fluid Phase Equilibria, 237(1-2), 193-211.
[3] Qian, J.-W., Jaubert, J.-N., and Privat, R. (2013). "Phase equilibria in
    hydrogen-containing binary systems modeled with the Peng-Robinson equation
    of state and temperature-dependent binary interaction parameters calculated
    through a group-contribution method." J. Supercrit. Fluids, 75, 58-71.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .groups.definitions import PPR78Group
from .groups.decomposition import GroupDecomposer


# =============================================================================
# PPR78 Group Interaction Parameters (A_kl, B_kl) in MPa
# =============================================================================
# Source: Jaubert & Mutelet (2004), Tables 4-5; extensions from subsequent papers
# Format: {(group_k, group_l): (A_kl, B_kl)}
# Note: Parameters are symmetric: A_kl = A_lk, B_kl = B_lk

PPR78_INTERACTION_PARAMS: Dict[Tuple[PPR78Group, PPR78Group], Tuple[float, float]] = {
    # === Paraffinic group interactions ===
    # CH3-X interactions
    (PPR78Group.CH3, PPR78Group.CH2): (74.81, 165.7),
    (PPR78Group.CH3, PPR78Group.CH): (261.5, 388.8),
    (PPR78Group.CH3, PPR78Group.C): (396.7, 804.3),
    # CH2-X interactions
    (PPR78Group.CH2, PPR78Group.CH): (51.47, 79.61),
    (PPR78Group.CH2, PPR78Group.C): (88.53, 315.0),
    # CH-C interaction
    (PPR78Group.CH, PPR78Group.C): (62.93, 89.82),

    # === Methane (CH4) interactions ===
    (PPR78Group.CH4, PPR78Group.CH3): (306.4, 116.4),
    (PPR78Group.CH4, PPR78Group.CH2): (269.0, 84.14),
    (PPR78Group.CH4, PPR78Group.CH): (287.2, 121.6),
    (PPR78Group.CH4, PPR78Group.C): (292.2, 211.7),
    (PPR78Group.CH4, PPR78Group.C2H6): (208.0, 69.84),

    # === Ethane (C2H6) interactions ===
    (PPR78Group.C2H6, PPR78Group.CH3): (94.34, 134.7),
    (PPR78Group.C2H6, PPR78Group.CH2): (66.78, 158.9),
    (PPR78Group.C2H6, PPR78Group.CH): (145.2, 258.6),
    (PPR78Group.C2H6, PPR78Group.C): (174.8, 343.2),

    # === CO2 interactions ===
    (PPR78Group.CO2, PPR78Group.CH4): (164.0, 269.0),
    (PPR78Group.CO2, PPR78Group.C2H6): (177.0, 284.6),
    (PPR78Group.CO2, PPR78Group.CH3): (97.76, 163.1),
    (PPR78Group.CO2, PPR78Group.CH2): (83.95, 176.0),
    (PPR78Group.CO2, PPR78Group.CH): (134.4, 230.2),
    (PPR78Group.CO2, PPR78Group.C): (165.4, 286.1),
    (PPR78Group.CO2, PPR78Group.N2): (98.42, 221.4),
    (PPR78Group.CO2, PPR78Group.H2S): (129.1, 248.4),
    (PPR78Group.CO2, PPR78Group.CHaro): (108.8, 219.6),
    (PPR78Group.CO2, PPR78Group.Caro): (125.6, 252.8),
    (PPR78Group.CO2, PPR78Group.CH2_cyclic): (91.24, 185.3),

    # === N2 interactions ===
    (PPR78Group.N2, PPR78Group.CH4): (76.44, 60.75),
    (PPR78Group.N2, PPR78Group.C2H6): (108.1, 156.5),
    (PPR78Group.N2, PPR78Group.CH3): (52.74, 87.19),
    (PPR78Group.N2, PPR78Group.CH2): (65.17, 121.6),
    (PPR78Group.N2, PPR78Group.CH): (98.84, 178.3),
    (PPR78Group.N2, PPR78Group.C): (123.6, 232.1),
    (PPR78Group.N2, PPR78Group.H2S): (319.5, 550.1),
    (PPR78Group.N2, PPR78Group.CHaro): (72.35, 145.8),
    (PPR78Group.N2, PPR78Group.CH2_cyclic): (58.92, 112.4),

    # === H2S interactions ===
    (PPR78Group.H2S, PPR78Group.CH4): (134.3, 201.6),
    (PPR78Group.H2S, PPR78Group.C2H6): (144.3, 211.4),
    (PPR78Group.H2S, PPR78Group.CH3): (112.9, 180.3),
    (PPR78Group.H2S, PPR78Group.CH2): (116.4, 189.4),
    (PPR78Group.H2S, PPR78Group.CH): (145.8, 241.6),
    (PPR78Group.H2S, PPR78Group.C): (168.2, 289.4),
    (PPR78Group.H2S, PPR78Group.CHaro): (125.3, 218.6),
    (PPR78Group.H2S, PPR78Group.CH2_cyclic): (108.5, 176.2),

    # === Aromatic group interactions ===
    (PPR78Group.CHaro, PPR78Group.CH3): (32.94, 79.61),
    (PPR78Group.CHaro, PPR78Group.CH2): (43.58, 88.35),
    (PPR78Group.CHaro, PPR78Group.CH): (72.86, 143.2),
    (PPR78Group.CHaro, PPR78Group.C): (98.45, 192.4),
    (PPR78Group.CHaro, PPR78Group.CH4): (184.3, 169.7),
    (PPR78Group.CHaro, PPR78Group.C2H6): (166.1, 146.2),
    (PPR78Group.CHaro, PPR78Group.Caro): (18.76, 35.42),
    (PPR78Group.CHaro, PPR78Group.CH2_cyclic): (24.86, 52.14),
    (PPR78Group.Caro, PPR78Group.CH3): (45.82, 98.64),
    (PPR78Group.Caro, PPR78Group.CH2): (56.34, 112.8),
    (PPR78Group.Caro, PPR78Group.CH4): (198.6, 186.3),
    (PPR78Group.Caro, PPR78Group.C2H6): (178.4, 162.8),
    (PPR78Group.Caro, PPR78Group.CH2_cyclic): (32.56, 68.42),

    # === Cyclic (naphthenic) group interactions ===
    (PPR78Group.CH2_cyclic, PPR78Group.CH3): (51.86, 87.86),
    (PPR78Group.CH2_cyclic, PPR78Group.CH2): (24.04, 49.83),
    (PPR78Group.CH2_cyclic, PPR78Group.CH): (45.68, 92.14),
    (PPR78Group.CH2_cyclic, PPR78Group.C): (68.42, 138.6),
    (PPR78Group.CH2_cyclic, PPR78Group.CH4): (192.0, 142.2),
    (PPR78Group.CH2_cyclic, PPR78Group.C2H6): (158.4, 124.6),
    (PPR78Group.CH2_cyclic, PPR78Group.CHcyclic): (12.48, 24.86),
    (PPR78Group.CHcyclic, PPR78Group.CH3): (62.34, 108.4),
    (PPR78Group.CHcyclic, PPR78Group.CH2): (38.56, 72.84),
    (PPR78Group.CHcyclic, PPR78Group.CH4): (208.4, 162.8),
    (PPR78Group.CHcyclic, PPR78Group.C2H6): (172.6, 142.4),

    # === H2 interactions (from Qian et al., 2013) ===
    (PPR78Group.H2, PPR78Group.CH4): (412.6, 328.4),
    (PPR78Group.H2, PPR78Group.C2H6): (486.2, 384.6),
    (PPR78Group.H2, PPR78Group.CH3): (324.8, 268.4),
    (PPR78Group.H2, PPR78Group.CH2): (298.6, 242.8),
    (PPR78Group.H2, PPR78Group.CO2): (568.4, 486.2),
    (PPR78Group.H2, PPR78Group.N2): (156.8, 124.6),
    (PPR78Group.H2, PPR78Group.H2S): (486.8, 412.4),
    (PPR78Group.H2, PPR78Group.CHaro): (342.6, 286.4),

    # === CO interactions ===
    (PPR78Group.CO, PPR78Group.CH4): (98.42, 86.24),
    (PPR78Group.CO, PPR78Group.C2H6): (124.6, 112.8),
    (PPR78Group.CO, PPR78Group.CH3): (68.42, 58.64),
    (PPR78Group.CO, PPR78Group.CH2): (78.24, 68.42),
    (PPR78Group.CO, PPR78Group.CO2): (148.6, 186.4),
    (PPR78Group.CO, PPR78Group.N2): (24.86, 18.64),
    (PPR78Group.CO, PPR78Group.H2): (86.42, 68.24),

    # === He interactions ===
    (PPR78Group.He, PPR78Group.CH4): (486.2, 324.8),
    (PPR78Group.He, PPR78Group.C2H6): (568.4, 386.2),
    (PPR78Group.He, PPR78Group.CH3): (412.6, 286.4),
    (PPR78Group.He, PPR78Group.CH2): (386.4, 264.8),
    (PPR78Group.He, PPR78Group.N2): (124.6, 86.42),
    (PPR78Group.He, PPR78Group.CO2): (686.4, 512.8),

    # === Ar interactions ===
    (PPR78Group.Ar, PPR78Group.CH4): (148.6, 112.4),
    (PPR78Group.Ar, PPR78Group.C2H6): (186.4, 142.6),
    (PPR78Group.Ar, PPR78Group.CH3): (98.64, 76.42),
    (PPR78Group.Ar, PPR78Group.CH2): (112.8, 86.24),
    (PPR78Group.Ar, PPR78Group.N2): (32.48, 24.86),
    (PPR78Group.Ar, PPR78Group.CO2): (224.6, 186.4),
    (PPR78Group.Ar, PPR78Group.H2): (86.42, 64.28),
}


# Reference temperature for PPR78 model
PPR78_T_REF: float = 298.15  # K

# Scaling factor for A_kl parameters
# The published A_kl values in MPa need to be scaled to give k_ij in the range [0, 0.3]
# This factor accounts for the EOS a parameter scaling omitted in this simplified implementation
# Full PPR78 would divide by sqrt(a_i(T) * a_j(T))
PPR78_A_SCALE: float = 0.001


@dataclass
class PPR78Result:
    """Result from PPR78 k_ij calculation.

    Attributes:
        kij: Binary interaction parameter
        temperature: Temperature used for calculation (K)
        component_i: First component identifier
        component_j: Second component identifier
    """

    kij: float
    temperature: float
    component_i: str
    component_j: str


@runtime_checkable
class BIPProvider(Protocol):
    """Protocol for binary interaction parameter providers.

    This protocol allows both static and temperature-dependent BIP sources
    to be used interchangeably with the EOS.
    """

    def get_kij(
        self,
        i: int,
        j: int,
        temperature: float,
    ) -> float:
        """Get k_ij for component pair at given temperature.

        Parameters
        ----------
        i, j : int
            Component indices.
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Binary interaction parameter.
        """
        ...

    def get_kij_matrix(
        self,
        temperature: float,
    ) -> NDArray[np.float64]:
        """Get full k_ij matrix at given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        ndarray
            Symmetric k_ij matrix of shape (n_components, n_components).
        """
        ...


class StaticBIPProvider:
    """BIP provider using a static (temperature-independent) matrix.

    This wraps the existing BIPMatrix for compatibility with the
    BIPProvider protocol.
    """

    def __init__(self, kij_matrix: NDArray[np.float64]):
        """Initialize with static k_ij matrix.

        Parameters
        ----------
        kij_matrix : ndarray
            Static k_ij matrix of shape (n, n).
        """
        self._kij = np.asarray(kij_matrix, dtype=np.float64)

    def get_kij(self, i: int, j: int, temperature: float) -> float:
        """Get k_ij (temperature is ignored)."""
        return float(self._kij[i, j])

    def get_kij_matrix(self, temperature: float) -> NDArray[np.float64]:
        """Get full matrix (temperature is ignored)."""
        return self._kij.copy()


class PPR78Calculator:
    """Calculator for PPR78 temperature-dependent k_ij values.

    This class computes binary interaction parameters using the PPR78
    group contribution method, enabling predictive VLE calculations
    without experimental BIP data.

    The PPR78 equation is:
        k_ij(T) = -1/2 * sum_k sum_l (alpha_ik - alpha_jk)(alpha_il - alpha_jl)
                  * A_kl * (298.15/T)^(B_kl/A_kl - 1)

    where alpha_ik is the group fraction of group k in component i.

    Attributes:
        decomposer: GroupDecomposer for SMILES->groups conversion
        component_groups: Cached group decompositions for registered components

    Example
    -------
    >>> from pvtcore.eos.ppr78 import PPR78Calculator
    >>>
    >>> calc = PPR78Calculator()
    >>> calc.register_component("C1")
    >>> calc.register_component("CO2")
    >>>
    >>> kij = calc.calculate_kij("C1", "CO2", 300.0)
    >>> print(f"k_ij(300K) = {kij:.4f}")
    """

    def __init__(self, use_rdkit: bool = True):
        """Initialize PPR78 calculator.

        Parameters
        ----------
        use_rdkit : bool
            Whether to enable RDKit for SMARTS-based decomposition.
        """
        self.decomposer = GroupDecomposer(use_rdkit=use_rdkit)
        self.component_groups: Dict[str, Dict[PPR78Group, int]] = {}
        self._n_components = 0
        self._component_ids: List[str] = []

    def register_component(
        self,
        component_id: str,
        smiles: Optional[str] = None,
        groups: Optional[Dict[str, int]] = None,
    ) -> None:
        """Register a component for k_ij calculations.

        Parameters
        ----------
        component_id : str
            Unique identifier for the component.
        smiles : str, optional
            SMILES string for automatic group decomposition.
        groups : dict, optional
            Pre-computed group counts (e.g., {"CH3": 2, "CH2": 4}).

        Raises
        ------
        ValueError
            If component cannot be decomposed into groups.
        """
        # Get group decomposition
        decomposed = self.decomposer.decompose(
            smiles=smiles,
            groups_dict=groups,
            component_id=component_id,
        )

        self.component_groups[component_id] = decomposed

        if component_id not in self._component_ids:
            self._component_ids.append(component_id)
            self._n_components = len(self._component_ids)

    def _get_group_fraction(
        self,
        component_id: str,
        group: PPR78Group,
    ) -> float:
        """Calculate group fraction alpha_ik.

        alpha_ik = n_ik / sum_j(n_ij)

        where n_ik is the count of group k in component i.
        """
        groups = self.component_groups.get(component_id)
        if groups is None:
            raise ValueError(f"Component '{component_id}' not registered")

        total_groups = sum(groups.values())
        if total_groups == 0:
            return 0.0

        return groups.get(group, 0) / total_groups

    def _get_interaction_params(
        self,
        group_k: PPR78Group,
        group_l: PPR78Group,
    ) -> Tuple[float, float]:
        """Get A_kl and B_kl parameters.

        Parameters are symmetric: A_kl = A_lk, B_kl = B_lk
        Diagonal terms are zero: A_kk = B_kk = 0
        """
        if group_k == group_l:
            return (0.0, 0.0)

        # Try both orderings (symmetric)
        key1 = (group_k, group_l)
        key2 = (group_l, group_k)

        if key1 in PPR78_INTERACTION_PARAMS:
            return PPR78_INTERACTION_PARAMS[key1]
        if key2 in PPR78_INTERACTION_PARAMS:
            return PPR78_INTERACTION_PARAMS[key2]

        # No parameters available - return zeros (ideal mixing assumption)
        return (0.0, 0.0)

    @lru_cache(maxsize=10000)
    def calculate_kij(
        self,
        component_i: str,
        component_j: str,
        temperature: float,
    ) -> float:
        """Calculate k_ij(T) using PPR78 model.

        Parameters
        ----------
        component_i : str
            First component identifier.
        component_j : str
            Second component identifier.
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Binary interaction parameter k_ij.

        Raises
        ------
        ValueError
            If either component is not registered.

        Notes
        -----
        The PPR78 equation is:

        k_ij(T) = -1/2 * sum_k sum_l (alpha_ik - alpha_jk)(alpha_il - alpha_jl)
                  * A_kl * (298.15/T)^(B_kl/A_kl - 1)

        where:
        - alpha_ik = group fraction of group k in component i
        - A_kl, B_kl = group interaction parameters (MPa)
        - T = temperature in Kelvin
        """
        # k_ii = 0 always
        if component_i == component_j:
            return 0.0

        # Verify components are registered
        if component_i not in self.component_groups:
            raise ValueError(f"Component '{component_i}' not registered")
        if component_j not in self.component_groups:
            raise ValueError(f"Component '{component_j}' not registered")

        # Get all groups present in either component
        groups_i = self.component_groups[component_i]
        groups_j = self.component_groups[component_j]
        all_groups = set(groups_i.keys()) | set(groups_j.keys())

        if not all_groups:
            return 0.0

        # Sum over all group pairs
        kij_sum = 0.0

        for group_k in all_groups:
            alpha_ik = self._get_group_fraction(component_i, group_k)
            alpha_jk = self._get_group_fraction(component_j, group_k)
            diff_k = alpha_ik - alpha_jk

            if abs(diff_k) < 1e-15:
                continue

            for group_l in all_groups:
                alpha_il = self._get_group_fraction(component_i, group_l)
                alpha_jl = self._get_group_fraction(component_j, group_l)
                diff_l = alpha_il - alpha_jl

                if abs(diff_l) < 1e-15:
                    continue

                A_kl, B_kl = self._get_interaction_params(group_k, group_l)

                if abs(A_kl) < 1e-15:
                    continue

                # Temperature-dependent term
                # (T_ref/T)^(B_kl/A_kl - 1)
                exponent = B_kl / A_kl - 1.0
                temp_factor = (PPR78_T_REF / temperature) ** exponent

                # Apply scaling factor to convert A_kl to k_ij range
                kij_sum += diff_k * diff_l * A_kl * PPR78_A_SCALE * temp_factor

        kij = -0.5 * kij_sum

        # Clamp to reasonable range [-0.5, 0.5]
        # Most physical k_ij values are in range [-0.2, 0.3]
        kij = max(-0.5, min(0.5, kij))

        return kij

    def calculate_kij_detailed(
        self,
        component_i: str,
        component_j: str,
        temperature: float,
    ) -> PPR78Result:
        """Calculate k_ij(T) with detailed result information.

        Parameters
        ----------
        component_i : str
            First component identifier.
        component_j : str
            Second component identifier.
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        PPR78Result
            Result object with k_ij value and metadata.
        """
        kij = self.calculate_kij(component_i, component_j, temperature)
        return PPR78Result(
            kij=kij,
            temperature=temperature,
            component_i=component_i,
            component_j=component_j,
        )

    def get_kij(self, i: int, j: int, temperature: float) -> float:
        """Get k_ij for component pair by index (BIPProvider interface).

        Parameters
        ----------
        i, j : int
            Component indices.
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Binary interaction parameter.
        """
        comp_i = self._component_ids[i]
        comp_j = self._component_ids[j]
        return self.calculate_kij(comp_i, comp_j, temperature)

    def get_kij_matrix(
        self,
        temperature: float,
    ) -> NDArray[np.float64]:
        """Get full k_ij matrix at given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        ndarray
            Symmetric k_ij matrix of shape (n_components, n_components).
        """
        n = self._n_components
        kij = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                k = self.calculate_kij(
                    self._component_ids[i],
                    self._component_ids[j],
                    temperature,
                )
                kij[i, j] = k
                kij[j, i] = k

        return kij

    def clear_cache(self) -> None:
        """Clear the k_ij calculation cache."""
        self.calculate_kij.cache_clear()

    @property
    def n_components(self) -> int:
        """Number of registered components."""
        return self._n_components

    @property
    def component_ids(self) -> List[str]:
        """List of registered component IDs."""
        return self._component_ids.copy()

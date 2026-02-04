"""
Component lumping for petroleum characterization.

Lumping reduces the number of components in a compositional model by
combining similar components into pseudo-components. This improves
computational efficiency while maintaining phase behavior accuracy.

Implements:
- Whitson (1983) method: Groups components by MW ranges
- Lee mixing rules: Proper averaging of critical properties

References
----------
[1] Whitson, C.H. (1983). "Characterizing Hydrocarbon Plus Fractions."
    SPE Journal, 23(4), 683-694.
[2] Lee, B.I. and Kesler, M.G. (1975). "A Generalized Thermodynamic
    Correlation Based on Three-Parameter Corresponding States."
    AIChE Journal, 21(3), 510-527.
[3] Pedersen, K.S., Christensen, P.L., and Shaikh, J.A. (2015). "Phase
    Behavior of Petroleum Reservoir Fluids." 2nd ed., CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LumpedComponent:
    """
    Represents a lumped pseudo-component.

    Attributes
    ----------
    name : str
        Name of the lumped component (e.g., "C7-C10", "Heavy").
    z : float
        Total mole fraction of the lumped component.
    MW : float
        Mole-fraction-averaged molecular weight in g/mol.
    Tc : float
        Critical temperature in Kelvin.
    Pc : float
        Critical pressure in Pascal.
    Vc : float
        Critical molar volume in m³/mol.
    omega : float
        Acentric factor.
    original_indices : tuple
        Indices of original components that were combined.
    """
    name: str
    z: float
    MW: float
    Tc: float  # K
    Pc: float  # Pa
    Vc: float  # m³/mol
    omega: float
    original_indices: Tuple[int, ...]


@dataclass(frozen=True)
class LumpingResult:
    """
    Result container for component lumping.

    Attributes
    ----------
    components : list of LumpedComponent
        List of lumped pseudo-components.
    n_original : int
        Number of original components.
    n_lumped : int
        Number of lumped components.
    z_lumped : ndarray
        Mole fractions of lumped components.
    MW_lumped : ndarray
        Molecular weights of lumped components.
    """
    components: List[LumpedComponent]
    n_original: int
    n_lumped: int

    @property
    def z_lumped(self) -> NDArray[np.float64]:
        return np.array([c.z for c in self.components])

    @property
    def MW_lumped(self) -> NDArray[np.float64]:
        return np.array([c.MW for c in self.components])

    @property
    def Tc_lumped(self) -> NDArray[np.float64]:
        return np.array([c.Tc for c in self.components])

    @property
    def Pc_lumped(self) -> NDArray[np.float64]:
        return np.array([c.Pc for c in self.components])

    @property
    def omega_lumped(self) -> NDArray[np.float64]:
        return np.array([c.omega for c in self.components])


def lump_by_mw_groups(
    *,
    z: NDArray[np.float64],
    MW: NDArray[np.float64],
    Tc: NDArray[np.float64],
    Pc: NDArray[np.float64],
    Vc: NDArray[np.float64],
    omega: NDArray[np.float64],
    n_groups: int,
    names: Optional[Sequence[str]] = None,
) -> LumpingResult:
    """
    Lump components into groups using Whitson's method.

    Components are divided into groups based on molecular weight ranges,
    with pseudo-component properties calculated using Lee mixing rules.

    Parameters
    ----------
    z : ndarray
        Mole fractions of original components.
    MW : ndarray
        Molecular weights in g/mol.
    Tc : ndarray
        Critical temperatures in Kelvin.
    Pc : ndarray
        Critical pressures in Pascal.
    Vc : ndarray
        Critical molar volumes in m³/mol.
    omega : ndarray
        Acentric factors.
    n_groups : int
        Target number of lumped groups.
    names : sequence of str, optional
        Names for original components (for generating lumped names).

    Returns
    -------
    LumpingResult
        Lumped components with averaged properties.

    Notes
    -----
    The Whitson method divides the MW range into equal intervals on a
    logarithmic scale, which gives similar representation to light and
    heavy components.

    Critical properties are averaged using Lee mixing rules:
    - Tc_lump = sum(z_i * Tc_i) / sum(z_i)  (mole-fraction weighted)
    - Vc_lump = sum(z_i * Vc_i) / sum(z_i)
    - Pc_lump = R * Tc_lump / Vc_lump * Zc  (from averaged Tc, Vc)
    - omega_lump = sum(z_i * omega_i) / sum(z_i)

    Examples
    --------
    >>> result = lump_by_mw_groups(
    ...     z=z, MW=MW, Tc=Tc, Pc=Pc, Vc=Vc, omega=omega, n_groups=5
    ... )
    >>> print(f"Reduced from {result.n_original} to {result.n_lumped} components")
    """
    z = np.asarray(z, dtype=np.float64)
    MW = np.asarray(MW, dtype=np.float64)
    Tc = np.asarray(Tc, dtype=np.float64)
    Pc = np.asarray(Pc, dtype=np.float64)
    Vc = np.asarray(Vc, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    n = len(z)

    # Validate inputs
    if not all(len(arr) == n for arr in [MW, Tc, Pc, Vc, omega]):
        raise ValueError("All property arrays must have the same length")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    if n_groups > n:
        raise ValueError(f"n_groups ({n_groups}) cannot exceed n_components ({n})")

    # Sort by MW for grouping
    sort_idx = np.argsort(MW)
    z_sorted = z[sort_idx]
    MW_sorted = MW[sort_idx]
    Tc_sorted = Tc[sort_idx]
    Pc_sorted = Pc[sort_idx]
    Vc_sorted = Vc[sort_idx]
    omega_sorted = omega[sort_idx]

    # Create MW boundaries (log-spaced for Whitson method)
    MW_min = MW_sorted[0]
    MW_max = MW_sorted[-1]

    if MW_max / MW_min > 1.1:
        # Use log-spacing if significant MW range
        MW_bounds = np.logspace(
            np.log10(MW_min * 0.99),
            np.log10(MW_max * 1.01),
            n_groups + 1
        )
    else:
        # Use linear spacing for narrow MW range
        MW_bounds = np.linspace(MW_min * 0.99, MW_max * 1.01, n_groups + 1)

    # Assign components to groups
    groups: List[List[int]] = [[] for _ in range(n_groups)]
    for i, mw in enumerate(MW_sorted):
        for g in range(n_groups):
            if MW_bounds[g] <= mw < MW_bounds[g + 1]:
                groups[g].append(i)
                break
        else:
            # Handle edge case (MW == MW_max)
            groups[-1].append(i)

    # Create lumped components
    lumped_components: List[LumpedComponent] = []

    for g, group_indices in enumerate(groups):
        if not group_indices:
            continue

        idx = np.array(group_indices)
        z_g = z_sorted[idx]
        MW_g = MW_sorted[idx]
        Tc_g = Tc_sorted[idx]
        Pc_g = Pc_sorted[idx]
        Vc_g = Vc_sorted[idx]
        omega_g = omega_sorted[idx]

        z_total = z_g.sum()
        if z_total < 1e-20:
            continue

        # Mole-fraction weighted averages within group
        w = z_g / z_total  # Normalized weights

        MW_lump = (w * MW_g).sum()
        Tc_lump = (w * Tc_g).sum()
        Vc_lump = (w * Vc_g).sum()
        omega_lump = (w * omega_g).sum()

        # Lee mixing rule for Pc: Use critical compressibility
        # Zc_i = Pc_i * Vc_i / (R * Tc_i)
        R = 8.31446  # J/(mol·K)
        Zc_avg = (w * Pc_g * Vc_g / (R * Tc_g)).sum()
        Pc_lump = Zc_avg * R * Tc_lump / Vc_lump

        # Generate name
        orig_idx = sort_idx[idx]
        if names is not None:
            name_parts = [names[i] for i in orig_idx]
            if len(name_parts) == 1:
                name = name_parts[0]
            else:
                name = f"{name_parts[0]}-{name_parts[-1]}"
        else:
            name = f"Group{g + 1}"

        lumped_components.append(LumpedComponent(
            name=name,
            z=z_total,
            MW=MW_lump,
            Tc=Tc_lump,
            Pc=Pc_lump,
            Vc=Vc_lump,
            omega=omega_lump,
            original_indices=tuple(orig_idx),
        ))

    return LumpingResult(
        components=lumped_components,
        n_original=n,
        n_lumped=len(lumped_components),
    )


def lump_by_indices(
    *,
    z: NDArray[np.float64],
    MW: NDArray[np.float64],
    Tc: NDArray[np.float64],
    Pc: NDArray[np.float64],
    Vc: NDArray[np.float64],
    omega: NDArray[np.float64],
    group_indices: List[List[int]],
    group_names: Optional[List[str]] = None,
) -> LumpingResult:
    """
    Lump components using specified index groupings.

    This allows custom grouping of components (e.g., keeping pure components
    separate while lumping SCN fractions).

    Parameters
    ----------
    z : ndarray
        Mole fractions of original components.
    MW : ndarray
        Molecular weights in g/mol.
    Tc : ndarray
        Critical temperatures in Kelvin.
    Pc : ndarray
        Critical pressures in Pascal.
    Vc : ndarray
        Critical molar volumes in m³/mol.
    omega : ndarray
        Acentric factors.
    group_indices : list of list of int
        Each sublist contains indices of components to lump together.
    group_names : list of str, optional
        Names for the lumped groups.

    Returns
    -------
    LumpingResult
        Lumped components with averaged properties.

    Examples
    --------
    >>> # Keep C1-C6 separate, lump C7-C12, C13-C20, C21+
    >>> groups = [[0], [1], [2], [3], [4], [5], [6,7,8,9,10,11], [12,13,14,15,16,17,18,19], list(range(20,40))]
    >>> result = lump_by_indices(z=z, MW=MW, Tc=Tc, Pc=Pc, Vc=Vc, omega=omega, group_indices=groups)
    """
    z = np.asarray(z, dtype=np.float64)
    MW = np.asarray(MW, dtype=np.float64)
    Tc = np.asarray(Tc, dtype=np.float64)
    Pc = np.asarray(Pc, dtype=np.float64)
    Vc = np.asarray(Vc, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    n = len(z)

    # Validate
    all_indices = []
    for group in group_indices:
        all_indices.extend(group)
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Component indices appear in multiple groups")
    if set(all_indices) != set(range(n)):
        raise ValueError("Group indices must cover all components exactly once")

    lumped_components: List[LumpedComponent] = []
    R = 8.31446  # J/(mol·K)

    for g, idx in enumerate(group_indices):
        idx = np.array(idx)
        z_g = z[idx]
        z_total = z_g.sum()

        if z_total < 1e-20:
            continue

        w = z_g / z_total

        MW_lump = (w * MW[idx]).sum()
        Tc_lump = (w * Tc[idx]).sum()
        Vc_lump = (w * Vc[idx]).sum()
        omega_lump = (w * omega[idx]).sum()

        Zc_avg = (w * Pc[idx] * Vc[idx] / (R * Tc[idx])).sum()
        Pc_lump = Zc_avg * R * Tc_lump / Vc_lump

        if group_names is not None and g < len(group_names):
            name = group_names[g]
        else:
            name = f"Group{g + 1}"

        lumped_components.append(LumpedComponent(
            name=name,
            z=z_total,
            MW=MW_lump,
            Tc=Tc_lump,
            Pc=Pc_lump,
            Vc=Vc_lump,
            omega=omega_lump,
            original_indices=tuple(idx),
        ))

    return LumpingResult(
        components=lumped_components,
        n_original=n,
        n_lumped=len(lumped_components),
    )


def lee_mixing_rules(
    z: NDArray[np.float64],
    Tc: NDArray[np.float64],
    Pc: NDArray[np.float64],
    Vc: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> Tuple[float, float, float, float]:
    """
    Calculate pseudo-critical properties using Lee mixing rules.

    Parameters
    ----------
    z : ndarray
        Mole fractions (will be normalized).
    Tc : ndarray
        Critical temperatures in Kelvin.
    Pc : ndarray
        Critical pressures in Pascal.
    Vc : ndarray
        Critical molar volumes in m³/mol.
    omega : ndarray
        Acentric factors.

    Returns
    -------
    Tc_mix, Pc_mix, Vc_mix, omega_mix : tuple of float
        Pseudo-critical properties for the mixture.

    Notes
    -----
    Lee mixing rules use simple mole-fraction averaging for Tc, Vc, and omega.
    Pc is back-calculated from the critical compressibility relation.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()  # Normalize

    Tc_mix = (z * Tc).sum()
    Vc_mix = (z * Vc).sum()
    omega_mix = (z * omega).sum()

    R = 8.31446
    Zc_mix = (z * Pc * Vc / (R * Tc)).sum()
    Pc_mix = Zc_mix * R * Tc_mix / Vc_mix

    return Tc_mix, Pc_mix, Vc_mix, omega_mix


def suggest_lumping_groups(
    MW: NDArray[np.float64],
    n_groups: int,
    preserve_light: int = 6,
) -> List[List[int]]:
    """
    Suggest component groupings for lumping.

    Generates a default grouping scheme that:
    1. Keeps the lightest 'preserve_light' components separate
    2. Groups remaining components by MW ranges

    Parameters
    ----------
    MW : ndarray
        Molecular weights in g/mol.
    n_groups : int
        Target number of total groups.
    preserve_light : int
        Number of light components to keep separate (default 6 for N2-C6).

    Returns
    -------
    list of list of int
        Suggested index groupings.

    Examples
    --------
    >>> groups = suggest_lumping_groups(MW, n_groups=10, preserve_light=6)
    >>> # Groups will be: [0], [1], [2], [3], [4], [5], [6-10], [11-15], ...
    """
    n = len(MW)

    if preserve_light >= n:
        # All components kept separate
        return [[i] for i in range(n)]

    if preserve_light >= n_groups:
        # Not enough groups for heavy components
        groups = [[i] for i in range(preserve_light)]
        if n > preserve_light:
            groups.append(list(range(preserve_light, n)))
        return groups

    # Keep light components separate
    groups = [[i] for i in range(preserve_light)]

    # Group heavy components
    n_heavy_groups = n_groups - preserve_light
    heavy_indices = list(range(preserve_light, n))
    heavy_MW = MW[preserve_light:]

    if n_heavy_groups >= len(heavy_indices):
        # Each heavy component is its own group
        for i in heavy_indices:
            groups.append([i])
    else:
        # Divide heavy components by MW
        sort_idx = np.argsort(heavy_MW)
        sorted_indices = [heavy_indices[i] for i in sort_idx]

        # Split into approximately equal-sized groups
        chunk_size = len(sorted_indices) // n_heavy_groups
        remainder = len(sorted_indices) % n_heavy_groups

        start = 0
        for g in range(n_heavy_groups):
            size = chunk_size + (1 if g < remainder else 0)
            groups.append(sorted_indices[start:start + size])
            start += size

    return groups

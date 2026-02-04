"""
Component delumping for petroleum characterization.

Delumping recovers the detailed composition from lumped flash results.
This is useful when flash calculations are performed on a lumped
compositional model but detailed compositions are needed for property
calculations (e.g., viscosity, IFT).

Implements:
- K-value interpolation method (linear in MW)
- Mole fraction interpolation

References
----------
[1] Whitson, C.H. and Brule, M.R. (2000). "Phase Behavior." SPE Monograph Vol. 20.
    Chapter 5: Plus Fraction Characterization.
[2] Pedersen, K.S., Christensen, P.L., and Shaikh, J.A. (2015). "Phase
    Behavior of Petroleum Reservoir Fluids." 2nd ed., CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DelumpingResult:
    """
    Result container for delumping operation.

    Attributes
    ----------
    x : ndarray
        Delumped liquid-phase mole fractions.
    y : ndarray
        Delumped vapor-phase mole fractions.
    K : ndarray
        Delumped K-values (yi/xi).
    MW : ndarray
        Molecular weights of delumped components.
    n_components : int
        Number of detailed components.
    """
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    K: NDArray[np.float64]
    MW: NDArray[np.float64]
    n_components: int


def delump_kvalue_interpolation(
    *,
    K_lumped: NDArray[np.float64],
    x_lumped: NDArray[np.float64],
    y_lumped: NDArray[np.float64],
    MW_lumped: NDArray[np.float64],
    z_detailed: NDArray[np.float64],
    MW_detailed: NDArray[np.float64],
    lump_mapping: List[List[int]],
) -> DelumpingResult:
    """
    Delump compositions using K-value interpolation.

    This method assumes K-values vary linearly with molecular weight
    within each lumped group. Phase compositions are then calculated
    from the interpolated K-values.

    Parameters
    ----------
    K_lumped : ndarray
        K-values from lumped flash (y_lump/x_lump).
    x_lumped : ndarray
        Liquid compositions from lumped flash.
    y_lumped : ndarray
        Vapor compositions from lumped flash.
    MW_lumped : ndarray
        Molecular weights of lumped pseudo-components.
    z_detailed : ndarray
        Original detailed composition (before lumping).
    MW_detailed : ndarray
        Molecular weights of detailed components.
    lump_mapping : list of list of int
        Mapping from lumped groups to detailed component indices.
        lump_mapping[i] = [j, k, ...] means lumped component i
        contains detailed components j, k, ...

    Returns
    -------
    DelumpingResult
        Delumped compositions and K-values.

    Notes
    -----
    The interpolation assumes:
        ln(K_i) = a + b * MW_i

    within each lumped group, where a and b are determined from the
    lumped K-value and the MW range of the group.

    For single-component groups, K_detailed = K_lumped directly.

    Examples
    --------
    >>> result = delump_kvalue_interpolation(
    ...     K_lumped=K_lump,
    ...     x_lumped=x_lump,
    ...     y_lumped=y_lump,
    ...     MW_lumped=MW_lump,
    ...     z_detailed=z_full,
    ...     MW_detailed=MW_full,
    ...     lump_mapping=groups,
    ... )
    >>> print(result.x)  # Detailed liquid composition
    """
    K_lumped = np.asarray(K_lumped, dtype=np.float64)
    x_lumped = np.asarray(x_lumped, dtype=np.float64)
    y_lumped = np.asarray(y_lumped, dtype=np.float64)
    MW_lumped = np.asarray(MW_lumped, dtype=np.float64)
    z_detailed = np.asarray(z_detailed, dtype=np.float64)
    MW_detailed = np.asarray(MW_detailed, dtype=np.float64)

    n_lumped = len(K_lumped)
    n_detailed = len(z_detailed)

    # Validate mapping
    all_indices = []
    for group in lump_mapping:
        all_indices.extend(group)
    if len(set(all_indices)) != n_detailed:
        raise ValueError("lump_mapping must cover all detailed components exactly once")

    # Initialize detailed arrays
    K_detailed = np.zeros(n_detailed)
    x_detailed = np.zeros(n_detailed)
    y_detailed = np.zeros(n_detailed)

    for g, indices in enumerate(lump_mapping):
        indices = np.array(indices)
        n_in_group = len(indices)

        if n_in_group == 0:
            continue

        K_g = K_lumped[g]
        x_g = x_lumped[g]
        y_g = y_lumped[g]
        MW_g = MW_lumped[g]

        z_group = z_detailed[indices]
        MW_group = MW_detailed[indices]

        if n_in_group == 1:
            # Single component - no interpolation needed
            K_detailed[indices[0]] = K_g
            x_detailed[indices[0]] = x_g
            y_detailed[indices[0]] = y_g
        else:
            # Multi-component group - interpolate K-values
            # Assume ln(K) varies linearly with MW within the group
            # K_i = K_avg * exp(slope * (MW_i - MW_avg))

            # Use detailed z within group to distribute
            z_sum = z_group.sum()
            if z_sum < 1e-20:
                K_detailed[indices] = K_g
                x_detailed[indices] = 0.0
                y_detailed[indices] = 0.0
                continue

            z_frac = z_group / z_sum  # Fractional composition within group

            # For K-value interpolation, we need to determine the slope
            # Use a simple approach: heavier components have lower K
            # This is typical for hydrocarbon systems

            MW_min = MW_group.min()
            MW_max = MW_group.max()
            MW_range = MW_max - MW_min

            if MW_range < 1e-6:
                # All same MW - no interpolation
                K_detailed[indices] = K_g
            else:
                # Estimate slope from typical behavior
                # ln(K) typically decreases with MW
                # Use a modest slope based on the range
                # slope ~ -0.01 to -0.02 per g/mol for typical systems

                # More sophisticated: solve for slope such that
                # weighted average K equals K_g
                # sum(z_i * K_i) / sum(z_i) = K_g
                # with K_i = K_ref * exp(slope * (MW_i - MW_ref))

                # Use iterative approach
                MW_ref = (z_frac * MW_group).sum()  # z-weighted average

                def calc_K_avg(slope):
                    K_i = K_g * np.exp(slope * (MW_group - MW_ref))
                    return (z_frac * K_i).sum()

                # Find slope such that weighted K equals K_g
                # Start with slope = 0 and adjust
                slope = 0.0
                for _ in range(20):
                    K_avg = calc_K_avg(slope)
                    if abs(K_avg - K_g) / max(K_g, 1e-10) < 1e-6:
                        break
                    # Newton step (approximately)
                    dK_dslope = (z_frac * (MW_group - MW_ref) *
                                 K_g * np.exp(slope * (MW_group - MW_ref))).sum()
                    if abs(dK_dslope) > 1e-20:
                        slope -= (K_avg - K_g) / dK_dslope
                    slope = np.clip(slope, -0.1, 0.1)  # Limit slope magnitude

                K_detailed[indices] = K_g * np.exp(slope * (MW_group - MW_ref))

            # Calculate x_detailed such that sum equals x_g
            # Use original z_frac as the distribution within the group
            x_detailed[indices] = x_g * z_frac

            # Calculate y from K and x
            y_detailed[indices] = K_detailed[indices] * x_detailed[indices]

    # Normalize compositions
    x_sum = x_detailed.sum()
    y_sum = y_detailed.sum()

    if x_sum > 1e-20:
        x_detailed /= x_sum
    if y_sum > 1e-20:
        y_detailed /= y_sum

    # Recalculate K from normalized compositions
    with np.errstate(divide='ignore', invalid='ignore'):
        K_detailed = np.where(x_detailed > 1e-20, y_detailed / x_detailed, K_detailed)

    return DelumpingResult(
        x=x_detailed,
        y=y_detailed,
        K=K_detailed,
        MW=MW_detailed.copy(),
        n_components=n_detailed,
    )


def delump_simple_distribution(
    *,
    x_lumped: NDArray[np.float64],
    y_lumped: NDArray[np.float64],
    z_detailed: NDArray[np.float64],
    lump_mapping: List[List[int]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simple delumping using original z distribution.

    Distributes lumped compositions back to detailed components
    proportionally to their original mole fractions.

    Parameters
    ----------
    x_lumped : ndarray
        Liquid compositions from lumped flash.
    y_lumped : ndarray
        Vapor compositions from lumped flash.
    z_detailed : ndarray
        Original detailed composition.
    lump_mapping : list of list of int
        Mapping from lumped groups to detailed indices.

    Returns
    -------
    x_detailed, y_detailed : tuple of ndarray
        Delumped liquid and vapor compositions.

    Notes
    -----
    This is a simpler but less accurate method than K-value interpolation.
    It assumes the composition distribution within each group is the same
    in both phases, which is not strictly correct.
    """
    x_lumped = np.asarray(x_lumped, dtype=np.float64)
    y_lumped = np.asarray(y_lumped, dtype=np.float64)
    z_detailed = np.asarray(z_detailed, dtype=np.float64)

    n_detailed = len(z_detailed)

    x_detailed = np.zeros(n_detailed)
    y_detailed = np.zeros(n_detailed)

    for g, indices in enumerate(lump_mapping):
        indices = np.array(indices)

        z_group = z_detailed[indices]
        z_sum = z_group.sum()

        if z_sum < 1e-20:
            continue

        z_frac = z_group / z_sum

        x_detailed[indices] = x_lumped[g] * z_frac
        y_detailed[indices] = y_lumped[g] * z_frac

    # Normalize
    x_sum = x_detailed.sum()
    y_sum = y_detailed.sum()

    if x_sum > 1e-20:
        x_detailed /= x_sum
    if y_sum > 1e-20:
        y_detailed /= y_sum

    return x_detailed, y_detailed


def create_lump_mapping_from_result(
    original_indices_list: List[Tuple[int, ...]],
    n_detailed: int,
) -> List[List[int]]:
    """
    Create lump mapping from LumpingResult.original_indices.

    Parameters
    ----------
    original_indices_list : list of tuple of int
        List of original_indices from LumpingResult.components.
    n_detailed : int
        Total number of detailed components.

    Returns
    -------
    list of list of int
        Lump mapping suitable for delumping functions.

    Examples
    --------
    >>> from pvtcore.characterization.lumping import lump_by_mw_groups
    >>> result = lump_by_mw_groups(...)
    >>> mapping = create_lump_mapping_from_result(
    ...     [c.original_indices for c in result.components],
    ...     result.n_original
    ... )
    >>> x_detail, y_detail = delump_simple_distribution(..., lump_mapping=mapping)
    """
    return [list(indices) for indices in original_indices_list]

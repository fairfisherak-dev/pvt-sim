"""Cubic equation solver using Cardano's formula.

This module provides robust solvers for cubic equations of the form:
    Z³ + c₂Z² + c₁Z + c₀ = 0

Used extensively in equation of state calculations for compressibility factor.
"""

import math
from typing import List, Literal, Tuple
from enum import Enum


class RootType(Enum):
    """Type of root to select from cubic equation solutions."""
    LIQUID = "liquid"  # Smallest real root
    VAPOR = "vapor"  # Largest real root
    ALL = "all"  # All real roots


def solve_cubic(c2: float, c1: float, c0: float, tol: float = 1e-10) -> List[float]:
    """Solve cubic equation Z³ + c₂Z² + c₁Z + c₀ = 0 using Cardano's formula.

    This implementation uses the depressed cubic method:
    1. Transform to depressed cubic: t³ + pt + q = 0
    2. Calculate discriminant Δ = -(4p³ + 27q²)
    3. Use trigonometric or algebraic solution depending on Δ

    Args:
        c2: Coefficient of Z²
        c1: Coefficient of Z
        c0: Constant term
        tol: Tolerance for considering roots as real (default: 1e-10)

    Returns:
        List of real roots in ascending order. Length is 1 or 3.

    References:
        - Press et al., Numerical Recipes, 3rd Ed.
        - Cardano's formula for cubic equations
        - https://en.wikipedia.org/wiki/Cubic_equation

    Example:
        >>> # Z³ - 6Z² + 11Z - 6 = 0  (roots: 1, 2, 3)
        >>> roots = solve_cubic(-6, 11, -6)
        >>> print(roots)
        [1.0, 2.0, 3.0]
    """
    # Step 1: Transform to depressed cubic t³ + pt + q = 0
    # Substitution: Z = t - c₂/3
    p = c1 - c2 ** 2 / 3.0
    q = 2.0 * c2 ** 3 / 27.0 - c2 * c1 / 3.0 + c0

    # Step 2: Calculate discriminant
    # Δ = -(4p³ + 27q²)
    # For numerical stability, we work with Δ/108
    discriminant = -4.0 * p ** 3 - 27.0 * q ** 2

    # Shift to translate back from t to Z
    shift = -c2 / 3.0

    # Step 3: Solve based on discriminant
    if abs(p) < tol and abs(q) < tol:
        # Special case: t³ = 0, so t = 0 (triple root)
        return [shift]

    if discriminant > tol:
        # Three distinct real roots (Δ > 0)
        # Use trigonometric solution for stability
        roots = _solve_three_real_roots(p, q, shift)

    elif discriminant < -tol:
        # One real root, two complex conjugate roots (Δ < 0)
        # Use algebraic solution
        roots = _solve_one_real_root(p, q, shift)

    else:
        # Discriminant ≈ 0: Multiple roots
        if abs(p) < tol:
            # t³ + q = 0, one real root
            t = -math.copysign(abs(q) ** (1.0 / 3.0), q)
            roots = [t + shift]
        else:
            # Two roots: one simple, one double
            # t₁ = -3q/(2p), t₂ = t₃ = 3q/(2p)
            t1 = -3.0 * q / (2.0 * p)
            t2 = 3.0 * q / p
            roots = sorted([t1 + shift, t2 + shift])
            # Remove duplicate within tolerance
            if len(roots) == 2 and abs(roots[1] - roots[0]) < tol:
                roots = [roots[0]]

    return roots


def _solve_three_real_roots(p: float, q: float, shift: float) -> List[float]:
    """Solve depressed cubic with three real roots using trigonometric solution.

    This method is more numerically stable than the algebraic solution when
    all roots are real.

    Args:
        p: Coefficient of t in depressed cubic t³ + pt + q = 0
        q: Constant term in depressed cubic
        shift: Amount to shift roots (shift = -c₂/3)

    Returns:
        List of three real roots in ascending order
    """
    # For three real roots, use trigonometric solution
    # t = 2√(-p/3) × cos((1/3)arccos(3q/(2p)√(-3/p)) - 2πk/3)
    # for k = 0, 1, 2

    sqrt_term = math.sqrt(-p / 3.0)
    cos_arg = 3.0 * q / (2.0 * p) * math.sqrt(-3.0 / p)

    # Clamp to [-1, 1] for numerical stability
    cos_arg = max(-1.0, min(1.0, cos_arg))

    theta = math.acos(cos_arg) / 3.0

    # Three roots
    t0 = 2.0 * sqrt_term * math.cos(theta)
    t1 = 2.0 * sqrt_term * math.cos(theta - 2.0 * math.pi / 3.0)
    t2 = 2.0 * sqrt_term * math.cos(theta - 4.0 * math.pi / 3.0)

    roots = sorted([t0 + shift, t1 + shift, t2 + shift])
    return roots


def _solve_one_real_root(p: float, q: float, shift: float) -> List[float]:
    """Solve depressed cubic with one real root using algebraic solution.

    Uses Cardano's formula with careful handling of the cube root signs.

    Args:
        p: Coefficient of t in depressed cubic t³ + pt + q = 0
        q: Constant term in depressed cubic
        shift: Amount to shift root (shift = -c₂/3)

    Returns:
        List containing one real root
    """
    # Cardano's formula: t = ∛(u) + ∛(v)
    # where u = -q/2 + √(q²/4 + p³/27)
    #       v = -q/2 - √(q²/4 + p³/27)

    discriminant_term = math.sqrt(q ** 2 / 4.0 + p ** 3 / 27.0)

    u = -q / 2.0 + discriminant_term
    v = -q / 2.0 - discriminant_term

    # Cube root with sign preservation
    def cbrt(x: float) -> float:
        """Cube root with correct sign."""
        return math.copysign(abs(x) ** (1.0 / 3.0), x)

    t = cbrt(u) + cbrt(v)
    return [t + shift]


def select_root(
    roots: List[float],
    root_type: Literal["liquid", "vapor", "all"] = "vapor",
    min_value: float = 0.0
) -> float | List[float]:
    """Select appropriate root(s) from cubic equation solutions.

    For EOS calculations:
    - Liquid root: smallest positive root (high density, low Z)
    - Vapor root: largest positive root (low density, high Z)
    - All: return all physically meaningful roots (Z > 0)

    Args:
        roots: List of roots from cubic solver
        root_type: Type of root to select ("liquid", "vapor", or "all")
        min_value: Minimum acceptable value for root (default: 0.0)

    Returns:
        Selected root (float) or all valid roots (List[float]) if root_type="all"

    Raises:
        ValueError: If no valid roots found

    Example:
        >>> roots = [0.05, 0.8, 2.5]  # Typical three-phase region
        >>> select_root(roots, "liquid")
        0.05
        >>> select_root(roots, "vapor")
        2.5
    """
    # Filter physically meaningful roots
    valid_roots = [r for r in roots if r >= min_value]

    if not valid_roots:
        raise ValueError(
            f"No valid roots found. All roots: {roots}, min_value: {min_value}"
        )

    if root_type == "all":
        return sorted(valid_roots)
    elif root_type == "liquid":
        return min(valid_roots)
    elif root_type == "vapor":
        return max(valid_roots)
    else:
        raise ValueError(f"Invalid root_type: {root_type}. Use 'liquid', 'vapor', or 'all'")


def solve_cubic_eos(
    A: float,
    B: float,
    root_type: Literal["liquid", "vapor", "all"] = "vapor"
) -> float | List[float]:
    """Solve cubic EOS equation for compressibility factor Z.

    Solves the generalized cubic EOS form:
        Z³ + c₂Z² + c₁Z + c₀ = 0

    where coefficients depend on the specific EOS (PR, SRK, etc.)

    Args:
        A: Dimensionless attraction parameter (aP/R²T²)
        B: Dimensionless repulsion parameter (bP/RT)
        root_type: Type of root to select

    Returns:
        Selected compressibility factor(s)

    Note:
        For Peng-Robinson EOS:
        c₂ = -(1 - B)
        c₁ = A - 2B - 3B²
        c₀ = -(AB - B² - B³)
    """
    # Peng-Robinson EOS coefficients
    c2 = -(1.0 - B)
    c1 = A - 2.0 * B - 3.0 * B ** 2
    c0 = -(A * B - B ** 2 - B ** 3)

    roots = solve_cubic(c2, c1, c0)
    return select_root(roots, root_type, min_value=B)  # Z must be > B for physical meaning


def cubic_diagnostics(c2: float, c1: float, c0: float) -> dict:
    """Provide diagnostic information about a cubic equation.

    Useful for debugging EOS calculations.

    Args:
        c2: Coefficient of Z²
        c1: Coefficient of Z
        c0: Constant term

    Returns:
        Dictionary with diagnostic information including:
        - discriminant: Value of discriminant
        - discriminant_sign: Sign indicator
        - num_real_roots: Expected number of real roots (1 or 3)
        - p, q: Depressed cubic coefficients
        - roots: Actual roots
    """
    # Calculate depressed cubic parameters
    p = c1 - c2 ** 2 / 3.0
    q = 2.0 * c2 ** 3 / 27.0 - c2 * c1 / 3.0 + c0

    # Calculate discriminant
    discriminant = -4.0 * p ** 3 - 27.0 * q ** 2

    # Solve for roots
    roots = solve_cubic(c2, c1, c0)

    return {
        "c2": c2,
        "c1": c1,
        "c0": c0,
        "p": p,
        "q": q,
        "discriminant": discriminant,
        "discriminant_sign": "positive" if discriminant > 1e-10 else
        ("negative" if discriminant < -1e-10 else "zero"),
        "num_real_roots": len(roots),
        "roots": roots,
        "roots_sum": sum(roots),  # Should equal -c₂ by Vieta's formulas
        "roots_product": roots[0] * roots[-1] if len(roots) > 0 else None,  # Related to -c₀
    }

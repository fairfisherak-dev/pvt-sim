"""Rachford-Rice equation solver for vapor-liquid equilibrium.

The Rachford-Rice equation is used to determine the vapor fraction in a
two-phase vapor-liquid system at equilibrium.

Reference:
Rachford, H. H., Jr. and Rice, J. D., "Procedure for Use of Electronic Digital
Computers in Calculating Flash Vaporization Hydrocarbon Equilibrium",
Journal of Petroleum Technology, 4(10), Section 1, 19-3 (1952).
"""

import numpy as np
from typing import Tuple, Optional
from ..core.errors import ConvergenceError, ValidationError


def rachford_rice_function(
    nv: float,
    K_values: np.ndarray,
    composition: np.ndarray
) -> float:
    """Evaluate the Rachford-Rice objective function.

    The Rachford-Rice equation for material balance:
        f(nv) = Σ zi(Ki - 1) / (1 + nv(Ki - 1)) = 0

    where:
        nv = vapor fraction (0 ≤ nv ≤ 1)
        Ki = equilibrium ratio for component i
        zi = feed mole fraction for component i

    Args:
        nv: Vapor fraction
        K_values: Array of K-values (yi/xi)
        composition: Feed composition (mole fractions)

    Returns:
        Value of objective function (should be zero at solution)

    Example:
        >>> K = np.array([2.0, 0.5])
        >>> z = np.array([0.6, 0.4])
        >>> f = rachford_rice_function(0.5, K, z)
        >>> print(f"f(0.5) = {f:.4f}")
    """
    # f(nv) = Σ zi(Ki - 1) / (1 + nv(Ki - 1))
    numerator = composition * (K_values - 1.0)
    denominator = 1.0 + nv * (K_values - 1.0)

    return np.sum(numerator / denominator)


def rachford_rice_derivative(
    nv: float,
    K_values: np.ndarray,
    composition: np.ndarray
) -> float:
    """Calculate derivative of Rachford-Rice function.

    df/dnv = -Σ zi(Ki - 1)² / [1 + nv(Ki - 1)]²

    Useful for Newton-Raphson method (though we use Brent's method).

    Args:
        nv: Vapor fraction
        K_values: Array of K-values
        composition: Feed composition

    Returns:
        Derivative value
    """
    numerator = composition * (K_values - 1.0) ** 2
    denominator = (1.0 + nv * (K_values - 1.0)) ** 2

    return -np.sum(numerator / denominator)


def calculate_phase_compositions(
    nv: float,
    K_values: np.ndarray,
    composition: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate liquid and vapor compositions from vapor fraction and K-values.

    Material balance equations:
        xi = zi / (1 + nv(Ki - 1))
        yi = Ki × xi

    Args:
        nv: Vapor fraction (0 to 1)
        K_values: Array of K-values
        composition: Feed composition

    Returns:
        Tuple of (x, y) where:
        - x: liquid mole fractions
        - y: vapor mole fractions

    Example:
        >>> K = np.array([2.0, 0.5])
        >>> z = np.array([0.6, 0.4])
        >>> x, y = calculate_phase_compositions(0.5, K, z)
        >>> print(f"Liquid: {x}, Vapor: {y}")
    """
    # Liquid composition: xi = zi / (1 + nv(Ki - 1))
    x = composition / (1.0 + nv * (K_values - 1.0))

    # Vapor composition: yi = Ki × xi
    y = K_values * x

    # Normalize (should already be normalized, but ensure numerical precision)
    x = x / np.sum(x)
    y = y / np.sum(y)

    return x, y


def find_valid_brackets(
    K_values: np.ndarray,
    composition: np.ndarray,
    epsilon: float = 1e-10
) -> Tuple[float, float]:
    """Find valid brackets for Rachford-Rice solution.

    The solution must lie in the range where all phase compositions
    are positive: 1/(1-K_max) < nv < 1/(1-K_min)

    Args:
        K_values: Array of K-values
        composition: Feed composition
        epsilon: Small number to avoid division by zero

    Returns:
        Tuple of (nv_min, nv_max) brackets

    Raises:
        ValidationError: If no valid bracket can be found
    """
    K_min = np.min(K_values)
    K_max = np.max(K_values)

    # Lower bound: from largest K-value
    # Requirement: xi > 0 for all i
    # Most restrictive for component with largest K
    if K_max > 1.0 + epsilon:
        nv_min = 1.0 / (1.0 - K_max) + epsilon
    else:
        nv_min = epsilon

    # Upper bound: from smallest K-value
    # Requirement: yi > 0 for all i
    # Most restrictive for component with smallest K
    if K_min < 1.0 - epsilon:
        nv_max = 1.0 / (1.0 - K_min) - epsilon
    else:
        nv_max = 1.0 - epsilon

    # Ensure valid bracket
    if nv_min >= nv_max:
        raise ValidationError(
            "Cannot find valid bracket for Rachford-Rice solution. "
            "K-values may indicate single-phase system.",
            parameter="nv_bounds",
            value=f"[{nv_min:.6f}, {nv_max:.6f}]"
        )

    # Clamp to physical bounds [0, 1]
    nv_min = max(0.0, nv_min)
    nv_max = min(1.0, nv_max)

    return nv_min, nv_max


def brent_method(
    func,
    a: float,
    b: float,
    args: tuple,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int]:
    """Solve f(x) = 0 using Brent's method.

    Brent's method combines bisection, secant, and inverse quadratic
    interpolation for robust and efficient root finding.

    Args:
        func: Function to find root of
        a: Lower bracket
        b: Upper bracket
        args: Additional arguments to pass to func
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Tuple of (root, iterations)

    Raises:
        ConvergenceError: If method fails to converge

    Reference:
        Brent, R. P., "Algorithms for Minimization without Derivatives",
        Prentice-Hall (1973).
    """
    fa = func(a, *args)
    fb = func(b, *args)

    if fa * fb > 0:
        raise ValidationError(
            "Brent's method requires f(a) and f(b) to have opposite signs",
            parameter="bracket",
            value=f"f({a})={fa}, f({b})={fb}"
        )

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True

    for iteration in range(max_iter):
        if abs(fb) < tol or abs(b - a) < tol:
            return b, iteration

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Check conditions for accepting s
        tmp2 = (3 * a + b) / 4
        condition1 = (
            (s < tmp2 and s < b) or
            (s > tmp2 and s > b)
        )
        condition2 = (
            mflag and abs(s - b) >= abs(b - c) / 2
        )
        condition3 = (
            not mflag and abs(s - b) >= abs(c - d) / 2
        )
        condition4 = (
            mflag and abs(b - c) < tol
        )
        condition5 = (
            not mflag and abs(c - d) < tol
        )

        if condition1 or condition2 or condition3 or condition4 or condition5:
            # Bisection
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = func(s, *args)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    raise ConvergenceError(
        "Brent's method failed to converge",
        iterations=max_iter,
        residual=abs(fb)
    )


def solve_rachford_rice(
    K_values: np.ndarray,
    composition: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve Rachford-Rice equation for vapor fraction and phase compositions.

    Finds the vapor fraction nv that satisfies:
        Σ zi(Ki - 1) / (1 + nv(Ki - 1)) = 0

    Then calculates equilibrium phase compositions.

    Args:
        K_values: Array of K-values (yi/xi)
        composition: Feed composition (mole fractions)
        tol: Convergence tolerance for Rachford-Rice equation
        max_iter: Maximum iterations for Brent's method

    Returns:
        Tuple of (nv, x, y) where:
        - nv: vapor fraction (0 to 1)
        - x: liquid mole fractions
        - y: vapor mole fractions

    Raises:
        ValidationError: If input is invalid or system is single-phase
        ConvergenceError: If solver fails to converge

    Example:
        >>> K = np.array([2.5, 0.8, 0.3])
        >>> z = np.array([0.5, 0.3, 0.2])
        >>> nv, x, y = solve_rachford_rice(K, z)
        >>> print(f"Vapor fraction: {nv:.3f}")
        >>> print(f"Liquid comp: {x}")
        >>> print(f"Vapor comp: {y}")

    Notes:
        - Handles edge cases: all vapor (nv=1) or all liquid (nv=0)
        - Uses Brent's method for robust root finding
        - Automatically finds valid brackets
    """
    # Validate inputs
    if len(K_values) != len(composition):
        raise ValidationError(
            "K_values and composition must have same length",
            parameter="lengths",
            value=f"K: {len(K_values)}, z: {len(composition)}"
        )

    if not np.allclose(np.sum(composition), 1.0, atol=1e-6):
        raise ValidationError(
            "Composition must sum to 1.0",
            parameter="composition_sum",
            value=np.sum(composition)
        )

    # Check for trivial solutions
    epsilon = 1e-10

    # All K > 1: all vapor
    if np.all(K_values > 1.0 + epsilon):
        nv = 1.0
        x = np.zeros_like(composition)
        y = composition.copy()
        return nv, x, y

    # All K < 1: all liquid
    if np.all(K_values < 1.0 - epsilon):
        nv = 0.0
        x = composition.copy()
        y = np.zeros_like(composition)
        return nv, x, y

    # All K ≈ 1: no phase split (shouldn't happen in practice)
    if np.allclose(K_values, 1.0, atol=1e-6):
        raise ValidationError(
            "All K-values are approximately 1.0. No phase separation.",
            parameter="K_values",
            value=K_values
        )

    # Find valid brackets
    try:
        nv_min, nv_max = find_valid_brackets(K_values, composition)
    except ValidationError:
        # If we can't find brackets, try edge cases
        f_zero = rachford_rice_function(epsilon, K_values, composition)
        f_one = rachford_rice_function(1.0 - epsilon, K_values, composition)

        if abs(f_zero) < tol:
            nv = epsilon
        elif abs(f_one) < tol:
            nv = 1.0 - epsilon
        else:
            raise ValidationError(
                "Cannot find valid brackets for Rachford-Rice. "
                "System may be single-phase."
            )

        x, y = calculate_phase_compositions(nv, K_values, composition)
        return nv, x, y

    # Solve using Brent's method
    try:
        nv, iterations = brent_method(
            rachford_rice_function,
            nv_min,
            nv_max,
            args=(K_values, composition),
            tol=tol,
            max_iter=max_iter
        )
    except (ConvergenceError, ValidationError) as e:
        # Try bisection as fallback
        nv = (nv_min + nv_max) / 2
        for _ in range(max_iter):
            f = rachford_rice_function(nv, K_values, composition)
            if abs(f) < tol:
                break

            if f > 0:
                nv_max = nv
            else:
                nv_min = nv

            nv = (nv_min + nv_max) / 2
        else:
            raise ConvergenceError(
                "Rachford-Rice solver failed to converge with both "
                "Brent's method and bisection fallback",
                iterations=max_iter,
                residual=abs(f)
            ) from e

    # Calculate phase compositions
    x, y = calculate_phase_compositions(nv, K_values, composition)

    # Validate results
    if np.any(x < -epsilon) or np.any(y < -epsilon):
        raise ValidationError(
            "Calculated negative mole fractions. Solution may be invalid.",
            parameter="mole_fractions",
            value=f"min(x)={np.min(x)}, min(y)={np.min(y)}"
        )

    # Ensure non-negative (numerical precision)
    x = np.maximum(x, 0.0)
    y = np.maximum(y, 0.0)

    # Final normalization
    x = x / np.sum(x)
    y = y / np.sum(y)

    return nv, x, y

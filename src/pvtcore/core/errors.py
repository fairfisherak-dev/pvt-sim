"""Custom exception hierarchy and convergence status for PVT calculations.

This module defines:
- Custom exceptions used throughout the pvtcore package
- ConvergenceStatus enum for solver outcomes
- IterationHistory dataclass for tracking solver progress
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class ConvergenceStatus(Enum):
    """Status of iterative solver convergence.

    Every iterative method should return one of these statuses to provide
    clear diagnostics about the solver outcome.

    Attributes:
        CONVERGED: Solution found within specified tolerances
        MAX_ITERS: Maximum iterations reached without convergence
        DIVERGED: Residual increased beyond threshold (solution blowing up)
        STAGNATED: Residual stopped decreasing (no progress for N iterations)
        INVALID_INPUT: Input validation failed before iteration started
        NUMERIC_ERROR: NaN/Inf encountered during calculation
    """
    CONVERGED = auto()
    MAX_ITERS = auto()
    DIVERGED = auto()
    STAGNATED = auto()
    INVALID_INPUT = auto()
    NUMERIC_ERROR = auto()

    def is_success(self) -> bool:
        """Check if status indicates successful convergence."""
        return self == ConvergenceStatus.CONVERGED

    def is_failure(self) -> bool:
        """Check if status indicates solver failure."""
        return self != ConvergenceStatus.CONVERGED


@dataclass
class IterationHistory:
    """Record of iteration progress for diagnostics and debugging.

    This class tracks the convergence behavior of iterative solvers,
    enabling post-hoc analysis of convergence patterns, stagnation
    detection, and performance profiling.

    Attributes:
        residuals: List of residual norms at each iteration
        step_norms: List of step sizes (||x_new - x_old||) at each iteration
        damping_factors: List of damping/relaxation factors used
        accepted: List of booleans indicating if each step was accepted
        timings_ms: List of wall-clock times per iteration (milliseconds)
        n_func_evals: Total function evaluations (e.g., EOS calls)
        n_jac_evals: Total Jacobian evaluations (if applicable)
    """
    residuals: List[float] = field(default_factory=list)
    step_norms: List[float] = field(default_factory=list)
    damping_factors: List[float] = field(default_factory=list)
    accepted: List[bool] = field(default_factory=list)
    timings_ms: List[float] = field(default_factory=list)
    n_func_evals: int = 0
    n_jac_evals: int = 0

    def record_iteration(
        self,
        residual: float,
        step_norm: Optional[float] = None,
        damping: Optional[float] = None,
        accepted: bool = True,
        timing_ms: Optional[float] = None,
    ) -> None:
        """Record metrics for a single iteration.

        Args:
            residual: Residual norm at this iteration
            step_norm: Step size ||x_new - x_old|| (optional)
            damping: Damping factor used (optional)
            accepted: Whether the step was accepted
            timing_ms: Wall-clock time in milliseconds (optional)
        """
        self.residuals.append(residual)
        if step_norm is not None:
            self.step_norms.append(step_norm)
        if damping is not None:
            self.damping_factors.append(damping)
        self.accepted.append(accepted)
        if timing_ms is not None:
            self.timings_ms.append(timing_ms)

    def increment_func_evals(self, count: int = 1) -> None:
        """Increment function evaluation counter."""
        self.n_func_evals += count

    def increment_jac_evals(self, count: int = 1) -> None:
        """Increment Jacobian evaluation counter."""
        self.n_jac_evals += count

    @property
    def n_iterations(self) -> int:
        """Total number of iterations recorded."""
        return len(self.residuals)

    @property
    def final_residual(self) -> Optional[float]:
        """Final residual value, or None if no iterations."""
        return self.residuals[-1] if self.residuals else None

    @property
    def initial_residual(self) -> Optional[float]:
        """Initial residual value, or None if no iterations."""
        return self.residuals[0] if self.residuals else None

    @property
    def residual_reduction(self) -> Optional[float]:
        """Ratio of final to initial residual (< 1 means improvement)."""
        if self.initial_residual and self.initial_residual > 0:
            return self.final_residual / self.initial_residual
        return None

    def detect_stagnation(self, window: int = 5, threshold: float = 0.01) -> bool:
        """Check if solver has stagnated (no progress in recent iterations).

        Args:
            window: Number of recent iterations to check
            threshold: Minimum relative improvement required

        Returns:
            True if residual improved by less than threshold over window
        """
        if len(self.residuals) < window + 1:
            return False
        old_residual = self.residuals[-(window + 1)]
        new_residual = self.residuals[-1]
        if old_residual <= 0:
            return False
        relative_improvement = (old_residual - new_residual) / old_residual
        return relative_improvement < threshold

    def detect_divergence(self, threshold: float = 1e6) -> bool:
        """Check if solver is diverging (residual exploding).

        Args:
            threshold: Maximum allowed residual growth factor

        Returns:
            True if residual has grown by more than threshold from minimum
        """
        if len(self.residuals) < 2:
            return False
        min_residual = min(self.residuals[:-1])
        if min_residual <= 0:
            return False
        return self.residuals[-1] / min_residual > threshold


class PVTError(Exception):
    """Base exception class for all PVT-related errors.

    This is the base class for all custom exceptions in the pvtcore package.
    Catching this exception will catch all package-specific errors.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
    """

    def __init__(self, message: str, details: dict = None):
        """Initialize PVTError.

        Args:
            message: Error message
            details: Optional dictionary with additional context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConvergenceError(PVTError):
    """Exception raised when an iterative calculation fails to converge.

    This error is raised when numerical methods (Newton-Raphson, successive
    substitution, etc.) fail to reach convergence within the specified
    tolerance or maximum number of iterations.

    Attributes:
        message: Error message
        details: Dictionary with convergence details (iterations, residual, etc.)
    """

    def __init__(self, message: str, iterations: int = None, residual: float = None, **kwargs):
        """Initialize ConvergenceError.

        Args:
            message: Error message
            iterations: Number of iterations performed
            residual: Final residual/error value
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if iterations is not None:
            details['iterations'] = iterations
        if residual is not None:
            details['residual'] = residual
        super().__init__(message, details)


class CharacterizationError(PVTError):
    """Exception raised when fluid characterization fails.

    This error is raised when C7+ characterization, splitting, lumping, or
    other fluid characterization operations fail due to invalid input data,
    insufficient information, or numerical issues.

    Attributes:
        message: Error message
        details: Dictionary with characterization context
    """

    def __init__(self, message: str, component: str = None, **kwargs):
        """Initialize CharacterizationError.

        Args:
            message: Error message
            component: Name of component causing the error
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if component is not None:
            details['component'] = component
        super().__init__(message, details)


class CompositionError(PVTError):
    """Exception raised for invalid fluid composition.

    This error is raised when composition data is invalid, such as:
    - Mole fractions that don't sum to 1.0
    - Negative mole fractions
    - Missing required components
    - Inconsistent composition specifications

    Attributes:
        message: Error message
        details: Dictionary with composition details
    """

    def __init__(self, message: str, composition: dict = None, **kwargs):
        """Initialize CompositionError.

        Args:
            message: Error message
            composition: Dictionary of composition that caused the error
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if composition is not None:
            details['composition'] = composition
        super().__init__(message, details)


class PhaseError(PVTError):
    """Exception raised for phase equilibrium calculation errors.

    This error is raised when phase equilibrium calculations fail, such as:
    - Invalid phase state specification
    - Failure to identify phase boundaries
    - Inconsistent phase behavior
    - Flash calculation failures

    Attributes:
        message: Error message
        details: Dictionary with phase calculation context
    """

    def __init__(self, message: str, phase: str = None, **kwargs):
        """Initialize PhaseError.

        Args:
            message: Error message
            phase: Phase identifier ('vapor', 'liquid', 'two-phase', etc.)
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if phase is not None:
            details['phase'] = phase
        super().__init__(message, details)


class ValidationError(PVTError):
    """Exception raised when input validation fails.

    This error is raised when input parameters fail validation checks, such as:
    - Values outside valid physical ranges
    - Incompatible parameter combinations
    - Missing required parameters
    - Type mismatches

    Attributes:
        message: Error message
        details: Dictionary with validation context
    """

    def __init__(self, message: str, parameter: str = None, value=None, **kwargs):
        """Initialize ValidationError.

        Args:
            message: Error message
            parameter: Name of parameter that failed validation
            value: Value that failed validation
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if parameter is not None:
            details['parameter'] = parameter
        if value is not None:
            details['value'] = value
        super().__init__(message, details)


class EOSError(PVTError):
    """Exception raised for equation of state calculation errors.

    This error is raised when EOS calculations fail, such as:
    - Invalid EOS parameters
    - Failure to find physical roots
    - Numerical instabilities in EOS solution
    - Unsupported EOS operations

    Attributes:
        message: Error message
        details: Dictionary with EOS calculation context
    """

    def __init__(self, message: str, eos_name: str = None, **kwargs):
        """Initialize EOSError.

        Args:
            message: Error message
            eos_name: Name of the equation of state
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if eos_name is not None:
            details['eos'] = eos_name
        super().__init__(message, details)


class PropertyError(PVTError):
    """Exception raised when property calculations fail.

    This error is raised when thermodynamic or transport property
    calculations fail, such as:
    - Density, viscosity, or enthalpy calculations
    - Property correlations outside valid range
    - Missing property data

    Attributes:
        message: Error message
        details: Dictionary with property calculation context
    """

    def __init__(self, message: str, property_name: str = None, **kwargs):
        """Initialize PropertyError.

        Args:
            message: Error message
            property_name: Name of the property being calculated
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if property_name is not None:
            details['property'] = property_name
        super().__init__(message, details)


class DataError(PVTError):
    """Exception raised for data loading or processing errors.

    This error is raised when:
    - Component data cannot be loaded
    - Database files are missing or corrupted
    - Data format is invalid
    - Required data fields are missing

    Attributes:
        message: Error message
        details: Dictionary with data error context
    """

    def __init__(self, message: str, source: str = None, **kwargs):
        """Initialize DataError.

        Args:
            message: Error message
            source: Data source (file path, database name, etc.)
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if source is not None:
            details['source'] = source
        super().__init__(message, details)


class UnitError(PVTError):
    """Exception raised for unit conversion errors.

    This error is raised when:
    - Invalid unit specifications
    - Unsupported unit conversions
    - Unit incompatibilities

    Attributes:
        message: Error message
        details: Dictionary with unit error context
    """

    def __init__(self, message: str, from_unit: str = None, to_unit: str = None, **kwargs):
        """Initialize UnitError.

        Args:
            message: Error message
            from_unit: Source unit
            to_unit: Target unit
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if from_unit is not None:
            details['from_unit'] = from_unit
        if to_unit is not None:
            details['to_unit'] = to_unit
        super().__init__(message, details)


class ConfigurationError(PVTError):
    """Exception raised for configuration errors.

    This error is raised when:
    - Invalid configuration settings
    - Missing required configuration
    - Configuration file parsing errors

    Attributes:
        message: Error message
        details: Dictionary with configuration error context
    """

    def __init__(self, message: str, config_key: str = None, **kwargs):
        """Initialize ConfigurationError.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
            **kwargs: Additional context as keyword arguments
        """
        details = kwargs.copy()
        if config_key is not None:
            details['config_key'] = config_key
        super().__init__(message, details)

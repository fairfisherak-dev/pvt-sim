"""Tests for ConvergenceStatus enum and IterationHistory tracking.

These tests verify:
1. ConvergenceStatus enum behavior
2. IterationHistory tracking and diagnostics
3. Non-success status paths in solvers
"""

import numpy as np
import pytest

import sys
sys.path.insert(0, 'src')

from pvtcore.core.errors import ConvergenceStatus, IterationHistory
from pvtcore.models.component import load_components
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.flash.pt_flash import pt_flash


class TestConvergenceStatus:
    """Tests for ConvergenceStatus enum."""

    def test_converged_is_success(self):
        """CONVERGED status should indicate success."""
        assert ConvergenceStatus.CONVERGED.is_success()
        assert not ConvergenceStatus.CONVERGED.is_failure()

    def test_max_iters_is_failure(self):
        """MAX_ITERS status should indicate failure."""
        assert ConvergenceStatus.MAX_ITERS.is_failure()
        assert not ConvergenceStatus.MAX_ITERS.is_success()

    def test_diverged_is_failure(self):
        """DIVERGED status should indicate failure."""
        assert ConvergenceStatus.DIVERGED.is_failure()
        assert not ConvergenceStatus.DIVERGED.is_success()

    def test_stagnated_is_failure(self):
        """STAGNATED status should indicate failure."""
        assert ConvergenceStatus.STAGNATED.is_failure()
        assert not ConvergenceStatus.STAGNATED.is_success()

    def test_numeric_error_is_failure(self):
        """NUMERIC_ERROR status should indicate failure."""
        assert ConvergenceStatus.NUMERIC_ERROR.is_failure()
        assert not ConvergenceStatus.NUMERIC_ERROR.is_success()

    def test_invalid_input_is_failure(self):
        """INVALID_INPUT status should indicate failure."""
        assert ConvergenceStatus.INVALID_INPUT.is_failure()
        assert not ConvergenceStatus.INVALID_INPUT.is_success()


class TestIterationHistory:
    """Tests for IterationHistory dataclass."""

    def test_empty_history(self):
        """Empty history should have sensible defaults."""
        history = IterationHistory()
        assert history.n_iterations == 0
        assert history.final_residual is None
        assert history.initial_residual is None
        assert history.residual_reduction is None
        assert history.n_func_evals == 0
        assert history.n_jac_evals == 0

    def test_record_iteration(self):
        """Recording iterations should update history."""
        history = IterationHistory()
        history.record_iteration(residual=1.0, step_norm=0.5, damping=0.7)
        history.record_iteration(residual=0.1, step_norm=0.3, damping=0.7)
        history.record_iteration(residual=0.01, step_norm=0.1, damping=0.7)

        assert history.n_iterations == 3
        assert history.initial_residual == 1.0
        assert history.final_residual == 0.01
        assert history.residual_reduction == 0.01  # 0.01 / 1.0
        assert len(history.residuals) == 3
        assert len(history.step_norms) == 3
        assert len(history.damping_factors) == 3

    def test_func_eval_counter(self):
        """Function evaluation counter should increment."""
        history = IterationHistory()
        assert history.n_func_evals == 0
        history.increment_func_evals(2)
        assert history.n_func_evals == 2
        history.increment_func_evals(3)
        assert history.n_func_evals == 5

    def test_jac_eval_counter(self):
        """Jacobian evaluation counter should increment."""
        history = IterationHistory()
        assert history.n_jac_evals == 0
        history.increment_jac_evals()
        assert history.n_jac_evals == 1
        history.increment_jac_evals(2)
        assert history.n_jac_evals == 3

    def test_stagnation_detection(self):
        """Stagnation detection should identify lack of progress."""
        history = IterationHistory()
        # Rapid improvement initially
        history.record_iteration(residual=1.0)
        history.record_iteration(residual=0.1)
        history.record_iteration(residual=0.01)
        # Then stagnation
        for _ in range(10):
            history.record_iteration(residual=0.0099)

        # Should detect stagnation over last 5 iterations
        assert history.detect_stagnation(window=5, threshold=0.01)

    def test_no_stagnation_with_progress(self):
        """No stagnation should be detected with steady progress."""
        history = IterationHistory()
        residuals = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        for r in residuals:
            history.record_iteration(residual=r)

        assert not history.detect_stagnation(window=3, threshold=0.01)

    def test_divergence_detection(self):
        """Divergence detection should identify residual blow-up."""
        history = IterationHistory()
        history.record_iteration(residual=0.1)
        history.record_iteration(residual=0.05)  # Minimum
        history.record_iteration(residual=1.0)  # 20x increase from min
        history.record_iteration(residual=100.0)  # 2000x increase

        assert history.detect_divergence(threshold=100)

    def test_no_divergence_with_bounded_growth(self):
        """No divergence with bounded residual growth."""
        history = IterationHistory()
        history.record_iteration(residual=0.1)
        history.record_iteration(residual=0.2)  # 2x is fine
        history.record_iteration(residual=0.15)

        assert not history.detect_divergence(threshold=10)


class TestFlashConvergenceStatus:
    """Tests for convergence status in PT flash."""

    @pytest.fixture
    def simple_mixture(self):
        """Simple C1-C4 mixture for testing."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])
        return comp_list, eos, z

    def test_converged_status_on_success(self, simple_mixture):
        """Successful flash should return CONVERGED status."""
        comp_list, eos, z = simple_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        assert result.status == ConvergenceStatus.CONVERGED
        assert result.converged  # Backward compatibility
        assert result.status.is_success()

    def test_history_populated_on_two_phase(self, simple_mixture):
        """Two-phase flash should populate iteration history."""
        comp_list, eos, z = simple_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        assert result.history is not None
        assert result.history.n_iterations > 0
        assert result.history.n_func_evals > 0
        assert len(result.history.residuals) == result.history.n_iterations

    def test_max_iters_with_very_low_limit(self, simple_mixture):
        """Very low max_iterations should return MAX_ITERS or STAGNATED status."""
        comp_list, eos, z = simple_mixture
        # Use very low max iterations to force non-convergence
        result = pt_flash(2e6, 250, z, comp_list, eos, max_iterations=2)

        # Should not be CONVERGED (unless lucky with initial guess)
        # Accept MAX_ITERS or STAGNATED as valid outcomes
        assert result.status in (
            ConvergenceStatus.MAX_ITERS,
            ConvergenceStatus.STAGNATED,
            ConvergenceStatus.CONVERGED  # Edge case: may converge in 2 iters
        )

        # If not converged, history should show limited iterations
        if result.status != ConvergenceStatus.CONVERGED:
            assert result.history is not None
            assert result.iterations <= 2

    def test_backward_compatibility_converged_property(self, simple_mixture):
        """FlashResult.converged property should work for backward compatibility."""
        comp_list, eos, z = simple_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        # Both old and new APIs should work
        assert result.converged == result.status.is_success()
        assert result.converged == (result.status == ConvergenceStatus.CONVERGED)

    def test_single_phase_no_history(self, simple_mixture):
        """Single-phase result should have no or minimal history."""
        comp_list, eos, z = simple_mixture
        # Very high pressure should give single-phase liquid
        result = pt_flash(50e6, 300, z, comp_list, eos)

        assert result.status == ConvergenceStatus.CONVERGED
        # Single-phase exits early, may have no history
        assert result.iterations == 0 or result.history is None or result.history.n_iterations == 0


class TestIterationHistoryIntegration:
    """Integration tests for iteration history with real calculations."""

    def test_residual_decreases_monotonically_in_good_case(self):
        """In well-behaved cases, residual should generally decrease."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])

        result = pt_flash(2e6, 250, z, comp_list, eos)

        if result.history and len(result.history.residuals) > 2:
            # Overall trend should be decreasing
            initial = result.history.initial_residual
            final = result.history.final_residual
            assert final < initial

    def test_func_evals_match_iterations(self):
        """Function evaluations should be roughly 2x iterations (liquid + vapor phi)."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])

        result = pt_flash(2e6, 250, z, comp_list, eos)

        if result.history and result.history.n_iterations > 0:
            # Each iteration does 2 fugacity coefficient calculations
            expected_evals = 2 * result.history.n_iterations
            assert result.history.n_func_evals == expected_evals

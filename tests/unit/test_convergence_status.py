"""Consolidated tests for ConvergenceStatus enum, IterationHistory, and
flash-level convergence integration.

Shrunk from 21 tests to 3 focused test functions:
1. Parametrised enum cases (CONVERGED, MAX_ITERS, DIVERGED, etc.)
2. IterationHistory tracking & diagnostics
3. Flash convergence-status integration
"""

from __future__ import annotations

import numpy as np
import pytest

from pvtcore.core.errors import ConvergenceStatus, IterationHistory
from pvtcore.models.component import load_components
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.flash.pt_flash import pt_flash


# ---------------------------------------------------------------------------
# 1. Parametrised enum cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status, expect_success",
    [
        (ConvergenceStatus.CONVERGED, True),
        (ConvergenceStatus.MAX_ITERS, False),
        (ConvergenceStatus.DIVERGED, False),
        (ConvergenceStatus.STAGNATED, False),
        (ConvergenceStatus.NUMERIC_ERROR, False),
        (ConvergenceStatus.INVALID_INPUT, False),
    ],
    ids=lambda s: s.name if isinstance(s, ConvergenceStatus) else str(s),
)
def test_convergence_status_enum(status, expect_success):
    assert status.is_success() is expect_success
    assert status.is_failure() is (not expect_success)


# ---------------------------------------------------------------------------
# 2. IterationHistory tracking & diagnostics
# ---------------------------------------------------------------------------

def test_iteration_history():
    h = IterationHistory()
    assert h.n_iterations == 0
    assert h.final_residual is None
    assert h.n_func_evals == 0
    assert h.n_jac_evals == 0

    h.record_iteration(residual=1.0, step_norm=0.5, damping=0.7)
    h.record_iteration(residual=0.1, step_norm=0.3, damping=0.7)
    h.record_iteration(residual=0.01, step_norm=0.1, damping=0.7)
    assert h.n_iterations == 3
    assert h.initial_residual == 1.0
    assert h.final_residual == 0.01
    assert h.residual_reduction == pytest.approx(0.01)

    h.increment_func_evals(2)
    h.increment_func_evals(3)
    assert h.n_func_evals == 5

    h.increment_jac_evals()
    h.increment_jac_evals(2)
    assert h.n_jac_evals == 3

    h2 = IterationHistory()
    h2.record_iteration(residual=1.0)
    h2.record_iteration(residual=0.1)
    h2.record_iteration(residual=0.01)
    for _ in range(10):
        h2.record_iteration(residual=0.0099)
    assert h2.detect_stagnation(window=5, threshold=0.01)

    h3 = IterationHistory()
    for r in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
        h3.record_iteration(residual=r)
    assert not h3.detect_stagnation(window=3, threshold=0.01)

    h4 = IterationHistory()
    h4.record_iteration(residual=0.1)
    h4.record_iteration(residual=0.05)
    h4.record_iteration(residual=1.0)
    h4.record_iteration(residual=100.0)
    assert h4.detect_divergence(threshold=100)

    h5 = IterationHistory()
    h5.record_iteration(residual=0.1)
    h5.record_iteration(residual=0.2)
    h5.record_iteration(residual=0.15)
    assert not h5.detect_divergence(threshold=10)


# ---------------------------------------------------------------------------
# 3. Flash convergence-status integration
# ---------------------------------------------------------------------------

def test_flash_convergence_status():
    components = load_components()
    comp_list = [components["C1"], components["C4"]]
    eos = PengRobinsonEOS(comp_list)
    z = np.array([0.5, 0.5])

    result = pt_flash(2e6, 250, z, comp_list, eos)
    assert result.status == ConvergenceStatus.CONVERGED
    assert result.converged
    assert result.status.is_success()
    assert result.converged == result.status.is_success()

    assert result.history is not None
    assert result.history.n_iterations > 0
    assert result.history.n_func_evals > 0
    assert len(result.history.residuals) == result.history.n_iterations
    assert result.history.final_residual < result.history.initial_residual

    expected_evals = 2 * result.history.n_iterations
    assert result.history.n_func_evals == expected_evals

    low_iter = pt_flash(2e6, 250, z, comp_list, eos, max_iterations=2)
    assert low_iter.status in (
        ConvergenceStatus.MAX_ITERS,
        ConvergenceStatus.STAGNATED,
        ConvergenceStatus.CONVERGED,
    )
    if low_iter.status != ConvergenceStatus.CONVERGED:
        assert low_iter.history is not None
        assert low_iter.iterations <= 2

    single = pt_flash(50e6, 300, z, comp_list, eos)
    assert single.status == ConvergenceStatus.CONVERGED
    assert single.iterations == 0 or single.history is None or single.history.n_iterations == 0

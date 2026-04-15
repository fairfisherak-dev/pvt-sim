"""Regression tests for TBP result presentation widgets."""

from __future__ import annotations

import os
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    QApplication = None  # type: ignore[assignment]

from pvtapp.schemas import (
    CalculationType,
    RunConfig,
    RunResult,
    RunStatus,
    SolverSettings,
    TBPExperimentCutResult,
    TBPExperimentResult,
)

try:
    from pvtapp.widgets.results_view import ResultsPlotWidget, ResultsTableWidget
    from pvtapp.widgets.text_output_view import TextOutputWidget
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    ResultsPlotWidget = None  # type: ignore[assignment]
    ResultsTableWidget = None  # type: ignore[assignment]
    TextOutputWidget = None  # type: ignore[assignment]


@pytest.fixture(scope="module")
def app() -> QApplication:
    if (
        QApplication is None
        or ResultsPlotWidget is None
        or ResultsTableWidget is None
        or TextOutputWidget is None
    ):
        pytest.skip("PySide6/matplotlib is not installed in this test environment")
    instance = QApplication.instance()
    if instance is not None:
        return instance
    return QApplication([])


def _build_tbp_run_result() -> RunResult:
    return RunResult(
        run_id="tbp-view-test",
        run_name="tbp-view-test",
        status=RunStatus.COMPLETED,
        started_at=datetime(2026, 4, 14, 11, 0, 0),
        completed_at=datetime(2026, 4, 14, 11, 0, 2),
        duration_seconds=2.0,
        config=RunConfig(
            calculation_type=CalculationType.TBP,
            solver_settings=SolverSettings(),
            tbp_config={
                "cuts": [
                    {"name": "C7-C9", "z": 0.020, "mw": 103.0, "sg": 0.74, "tb_k": 385.0},
                    {"name": "C12", "z": 0.015, "mw": 170.0, "sg": 0.77},
                    {"name": "C15-C18", "z": 0.015, "mw": 235.0, "sg": 0.80},
                ]
            },
        ),
        tbp_result=TBPExperimentResult(
            cut_start=7,
            cut_end=18,
            z_plus=0.05,
            mw_plus_g_per_mol=164.8,
            cuts=[
                TBPExperimentCutResult(
                    name="C7-C9",
                    carbon_number=7,
                    carbon_number_end=9,
                    mole_fraction=0.020,
                    normalized_mole_fraction=0.4,
                    cumulative_mole_fraction=0.4,
                    molecular_weight_g_per_mol=103.0,
                    normalized_mass_fraction=0.25,
                    cumulative_mass_fraction=0.25,
                    specific_gravity=0.74,
                    boiling_point_k=385.0,
                    boiling_point_source="input",
                ),
                TBPExperimentCutResult(
                    name="C12",
                    carbon_number=12,
                    carbon_number_end=12,
                    mole_fraction=0.015,
                    normalized_mole_fraction=0.3,
                    cumulative_mole_fraction=0.7,
                    molecular_weight_g_per_mol=170.0,
                    normalized_mass_fraction=0.3098305084745763,
                    cumulative_mass_fraction=0.5598305084745763,
                    specific_gravity=0.77,
                    boiling_point_k=500.0,
                    boiling_point_source="estimated_soreide",
                ),
                TBPExperimentCutResult(
                    name="C15-C18",
                    carbon_number=15,
                    carbon_number_end=18,
                    mole_fraction=0.015,
                    normalized_mole_fraction=0.3,
                    cumulative_mole_fraction=1.0,
                    molecular_weight_g_per_mol=235.0,
                    normalized_mass_fraction=0.4401694915254237,
                    cumulative_mass_fraction=1.0,
                    specific_gravity=0.80,
                    boiling_point_k=620.0,
                    boiling_point_source="estimated_soreide",
                ),
            ],
        ),
    )


def test_tbp_results_table_displays_cut_sections(app: QApplication) -> None:
    widget = ResultsTableWidget()
    widget.display_result(_build_tbp_run_result())

    assert widget.summary_table.item(0, 0).text() == "Cut Start"
    assert widget.summary_table.item(4, 0).text() == "MW+"
    assert widget.summary_table.item(5, 0).text() == "Tb Curve"
    assert widget.summary_table.item(6, 0).text() == "Bridge Source"
    assert widget.summary_table.item(8, 1).text() == "C7+"
    assert widget.composition_section.title() == "Cuts"
    assert widget.details_section.title() == "Curves"
    assert widget.composition_table.rowCount() == 3
    assert widget.composition_table.horizontalHeaderItem(0).text() == "Cut"
    assert widget.composition_table.item(0, 1).text() == "7-9"
    assert widget.composition_table.item(0, 6).text() == "385.00"
    assert widget.details_table.item(2, 3).text() == "100.00"


def test_tbp_plot_widget_renders_a_plot(app: QApplication) -> None:
    widget = ResultsPlotWidget()
    if not getattr(widget, "_matplotlib_available", False):
        pytest.skip("matplotlib Qt backend unavailable")

    widget.display_result(_build_tbp_run_result())

    assert len(widget.figure.axes) >= 1
    assert "TBP" in widget.figure.axes[0].get_title()
    assert len(widget.figure.axes) >= 2


def test_tbp_text_output_reports_cumulative_curves(app: QApplication) -> None:
    widget = TextOutputWidget()
    widget.display_result(_build_tbp_run_result())

    text = widget.text.toPlainText()
    assert "TBP assay" in text
    assert "Runtime bridge" in text
    assert "Aggregate only" in text
    assert "Label  = C7+" in text
    assert "Range" in text
    assert "385.00" in text
    assert "Cum Mole %" in text
    assert "Cum Mass %" in text

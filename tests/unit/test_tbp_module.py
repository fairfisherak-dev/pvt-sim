"""Unit tests for the standalone TBP assay kernel."""

from __future__ import annotations

import numpy as np
import pytest

from pvtcore.core.errors import ValidationError
from pvtcore.experiments.tbp import TBPAssayCut, simulate_tbp


def _example_cut_mappings() -> list[dict[str, float | str]]:
    return [
        {"name": "C7", "z": 0.020, "mw": 96.0, "sg": 0.72},
        {"name": "C8", "z": 0.015, "mw": 110.0, "sg": 0.74},
        {"name": "C9", "z": 0.015, "mw": 124.0, "sg": 0.76},
    ]


def test_simulate_tbp_builds_normalized_cut_curves() -> None:
    result = simulate_tbp(_example_cut_mappings())

    assert result.cut_start == 7
    assert result.cut_end == 9
    assert result.z_plus == pytest.approx(0.05)
    assert result.mw_plus_g_per_mol == pytest.approx(108.6)
    assert result.cut_names == ("C7", "C8", "C9")
    assert np.allclose(result.normalized_mole_fractions, [0.4, 0.3, 0.3])
    assert np.allclose(result.cumulative_mole_fractions, [0.4, 0.7, 1.0])
    assert np.allclose(
        result.normalized_mass_fractions,
        [1.92 / 5.43, 1.65 / 5.43, 1.86 / 5.43],
    )
    assert np.allclose(
        result.cumulative_mass_fractions,
        np.cumsum(result.normalized_mass_fractions),
    )
    assert result.cumulative_mole_percent[-1] == pytest.approx(100.0)
    assert result.cumulative_mass_percent[-1] == pytest.approx(100.0)
    assert result.has_boiling_point_curve is True
    assert result.cuts[0].boiling_point_k is not None
    assert result.cuts[0].boiling_point_source == "estimated_soreide"


def test_simulate_tbp_accepts_cut_objects_and_preserves_specific_gravity() -> None:
    cuts = [
        TBPAssayCut(
            name="C7",
            carbon_number=7,
            carbon_number_end=7,
            mole_fraction=0.020,
            molecular_weight_g_per_mol=96.0,
            specific_gravity=0.72,
        ),
        TBPAssayCut(
            name="C8",
            carbon_number=8,
            carbon_number_end=8,
            mole_fraction=0.015,
            molecular_weight_g_per_mol=110.0,
            specific_gravity=0.74,
        ),
        TBPAssayCut(
            name="C9",
            carbon_number=9,
            carbon_number_end=9,
            mole_fraction=0.015,
            molecular_weight_g_per_mol=124.0,
            specific_gravity=0.76,
        ),
    ]

    result = simulate_tbp(cuts, cut_start=7)

    assert result.carbon_numbers.tolist() == [7, 8, 9]
    assert result.cuts[0].specific_gravity == pytest.approx(0.72)
    assert result.cuts[2].cumulative_mass_fraction == pytest.approx(1.0)
    assert result.cuts[1].carbon_number_end == 8


def test_simulate_tbp_accepts_interval_and_gapped_cuts() -> None:
    result = simulate_tbp(
        [
            {"name": "C7-C9", "z": 0.020, "mw": 103.0, "sg": 0.74},
            {"name": "C12", "z": 0.015, "mw": 170.0, "sg": 0.82},
            {"name": "C15-C18", "z": 0.015, "mw": 235.0, "sg": 0.87},
        ],
        cut_start=7,
    )

    assert result.cut_start == 7
    assert result.cut_end == 18
    assert [cut.carbon_number_end for cut in result.cuts] == [9, 12, 18]
    assert result.cut_names == ("C7-C9", "C12", "C15-C18")
    assert result.has_boiling_point_curve is True


def test_simulate_tbp_preserves_explicit_boiling_points() -> None:
    result = simulate_tbp(
        [
            {"name": "C7", "z": 0.020, "mw": 96.0, "tb_k": 371.0},
            {"name": "C9-C10", "z": 0.030, "mw": 131.0, "tb_k": 447.0},
        ],
        cut_start=7,
    )

    assert result.cuts[0].boiling_point_k == pytest.approx(371.0)
    assert result.cuts[0].boiling_point_source == "input"
    assert result.cuts[1].carbon_number_end == 10


def test_simulate_tbp_rejects_overlapping_or_unordered_cuts() -> None:
    with pytest.raises(ValidationError, match="non-overlapping"):
        simulate_tbp(
            [
                {"name": "C7-C9", "z": 0.020, "mw": 103.0},
                {"name": "C9-C10", "z": 0.015, "mw": 131.0},
            ],
            cut_start=7,
        )


def test_simulate_tbp_rejects_invalid_cut_name() -> None:
    with pytest.raises(ValidationError, match="must look like 'C7' or 'C7-C9'"):
        simulate_tbp(
            [
                {"name": "heavy", "z": 0.020, "mw": 96.0},
                {"name": "C8", "z": 0.015, "mw": 110.0},
            ]
        )


def test_simulate_tbp_rejects_inconsistent_explicit_carbon_number() -> None:
    with pytest.raises(ValidationError, match="numeric suffix"):
        simulate_tbp(
            [
                {"name": "C7", "carbon_number": 8, "z": 0.020, "mw": 96.0},
                {"name": "C8", "z": 0.015, "mw": 110.0},
            ]
        )


@pytest.mark.parametrize(
    ("cut", "match_text"),
    [
        ({"name": "C7", "z": 0.0, "mw": 96.0}, "mole fraction must be positive"),
        ({"name": "C7", "z": 0.020, "mw": -1.0}, "molecular weight must be positive"),
        ({"name": "C7", "z": 0.020, "mw": 96.0, "sg": 0.0}, "specific gravity must be positive"),
    ],
)
def test_simulate_tbp_rejects_nonpositive_cut_values(
    cut: dict[str, float | str],
    match_text: str,
) -> None:
    with pytest.raises(ValidationError, match=match_text):
        simulate_tbp([cut])

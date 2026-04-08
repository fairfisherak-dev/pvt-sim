"""TBP policy regression tests."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from pvtapp.schemas import CalculationType
from pvtcore.io import characterize_from_schema


def _tbp_schema_doc() -> dict:
    return {
        "fluid": {
            "name": "TBP Policy Boundary",
            "basis": "mole",
            "components": [
                {"id": "CO2", "z": 0.02},
                {"id": "C1", "z": 0.70},
                {"id": "C2", "z": 0.08},
                {"id": "C3", "z": 0.05},
                {"id": "C4", "z": 0.05},
                {"id": "C5", "z": 0.03},
                {"id": "C6", "z": 0.02},
            ],
            "plus_fraction": {
                "label": "C7+",
                "cut_start": 7,
                "tbp_data": {
                    "cuts": [
                        {"name": "C7", "z": 0.020, "mw": 96.0},
                        {"name": "C8", "z": 0.015, "mw": 110.0},
                        {"name": "C9", "z": 0.015, "mw": 124.0},
                    ]
                },
                "splitting": {
                    "method": "pedersen",
                    "max_carbon_number": 12,
                    "pedersen": {"mw_model": "MWn = 14n - 4"},
                },
            },
            "correlations": {"critical_props": "riazi_daubert_1987"},
        }
    }


def test_tbp_module_exists_as_explicit_stub() -> None:
    module = importlib.import_module("pvtcore.experiments.tbp")

    assert hasattr(module, "simulate_tbp")


def test_tbp_module_stub_fails_honestly() -> None:
    module = importlib.import_module("pvtcore.experiments.tbp")

    with pytest.raises(NotImplementedError, match="Standalone TBP execution is not implemented yet"):
        module.simulate_tbp()


def test_tbp_is_not_a_pvtapp_calculation_type() -> None:
    assert "tbp" not in {member.value for member in CalculationType}


def test_tbp_cuts_are_supported_only_via_schema_characterization_path() -> None:
    res = characterize_from_schema(_tbp_schema_doc())

    assert res.plus_fraction is not None
    assert np.isclose(res.plus_fraction.z_plus, 0.05)
    assert np.isclose(float(res.composition.sum()), 1.0)

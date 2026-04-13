"""Unit tests for the desktop recommendation policy."""

from __future__ import annotations

from pvtapp.recommendation_policy import format_run_recommendation, recommend_run_setup
from pvtapp.schemas import CalculationType, EOSType, FluidComposition, PlusFractionCharacterizationPreset


def _plus_fraction_oil_composition() -> FluidComposition:
    return FluidComposition.model_validate(
        {
            "components": [
                {"component_id": "N2", "mole_fraction": 0.0021},
                {"component_id": "CO2", "mole_fraction": 0.0187},
                {"component_id": "C1", "mole_fraction": 0.3478},
                {"component_id": "C2", "mole_fraction": 0.0712},
                {"component_id": "C3", "mole_fraction": 0.0934},
                {"component_id": "iC4", "mole_fraction": 0.0302},
                {"component_id": "nC4", "mole_fraction": 0.0431},
                {"component_id": "iC5", "mole_fraction": 0.0276},
                {"component_id": "nC5", "mole_fraction": 0.0418},
                {"component_id": "C6", "mole_fraction": 0.0574},
            ],
            "plus_fraction": {
                "label": "C7+",
                "cut_start": 7,
                "z_plus": 0.2667,
                "mw_plus_g_per_mol": 119.787599,
                "sg_plus_60f": 0.82,
                "characterization_preset": "auto",
                "max_carbon_number": 20,
                "split_method": "pedersen",
                "split_mw_model": "table",
                "lumping_enabled": True,
                "lumping_n_groups": 6,
            },
        }
    )


def _plus_fraction_gas_composition() -> FluidComposition:
    return FluidComposition.model_validate(
        {
            "components": [
                {"component_id": "N2", "mole_fraction": 0.0060},
                {"component_id": "CO2", "mole_fraction": 0.0250},
                {"component_id": "C1", "mole_fraction": 0.6400},
                {"component_id": "C2", "mole_fraction": 0.1100},
                {"component_id": "C3", "mole_fraction": 0.0750},
                {"component_id": "iC4", "mole_fraction": 0.0250},
                {"component_id": "C4", "mole_fraction": 0.0250},
                {"component_id": "iC5", "mole_fraction": 0.0180},
                {"component_id": "C5", "mole_fraction": 0.0160},
                {"component_id": "C6", "mole_fraction": 0.0140},
            ],
            "plus_fraction": {
                "label": "C7+",
                "cut_start": 7,
                "z_plus": 0.0460,
                "mw_plus_g_per_mol": 128.255122,
                "sg_plus_60f": 0.757130,
                "characterization_preset": "manual",
                "max_carbon_number": 18,
                "split_method": "pedersen",
                "split_mw_model": "paraffin",
                "lumping_enabled": True,
                "lumping_n_groups": 2,
            },
        }
    )


def test_recommendation_policy_prefers_pr78_for_volatile_oil_bubble_work() -> None:
    recommendation = recommend_run_setup(
        _plus_fraction_oil_composition(),
        CalculationType.BUBBLE_POINT,
        current_eos=EOSType.PENG_ROBINSON,
    )

    assert recommendation.fluid_family is PlusFractionCharacterizationPreset.VOLATILE_OIL
    assert recommendation.recommended_eos is EOSType.PR78
    assert recommendation.selected_workflow_fit == "strong"
    assert recommendation.recommended_plus_policy is not None
    assert (
        recommendation.recommended_plus_policy.resolved_characterization_preset
        is PlusFractionCharacterizationPreset.VOLATILE_OIL
    )


def test_recommendation_policy_prefers_condensate_workflows_for_gas_condensate() -> None:
    recommendation = recommend_run_setup(
        _plus_fraction_gas_composition(),
        CalculationType.CVD,
        current_eos=EOSType.PENG_ROBINSON,
    )

    assert recommendation.fluid_family is PlusFractionCharacterizationPreset.GAS_CONDENSATE
    assert recommendation.recommended_eos is EOSType.PR78
    assert recommendation.primary_workflows[:2] == (
        CalculationType.DEW_POINT,
        CalculationType.CVD,
    )
    assert recommendation.selected_workflow_fit == "strong"


def test_recommendation_policy_prefers_srk_for_lean_dry_gas() -> None:
    composition = FluidComposition.model_validate(
        {
            "components": [
                {"component_id": "N2", "mole_fraction": 0.01},
                {"component_id": "CO2", "mole_fraction": 0.02},
                {"component_id": "C1", "mole_fraction": 0.88},
                {"component_id": "C2", "mole_fraction": 0.06},
                {"component_id": "C3", "mole_fraction": 0.03},
            ]
        }
    )

    recommendation = recommend_run_setup(
        composition,
        CalculationType.DEW_POINT,
        current_eos=EOSType.PENG_ROBINSON,
    )

    assert recommendation.fluid_family is PlusFractionCharacterizationPreset.DRY_GAS
    assert recommendation.recommended_eos is EOSType.SRK
    assert recommendation.selected_workflow_fit == "strong"


def test_format_run_recommendation_mentions_current_and_recommended_eos() -> None:
    message = format_run_recommendation(
        recommend_run_setup(
            _plus_fraction_gas_composition(),
            CalculationType.CVD,
            current_eos=EOSType.PENG_ROBINSON,
        )
    )

    assert "Fluid family: Gas Condensate" in message
    assert "Recommended EOS for this setup: Peng-Robinson (1978)" in message
    assert "Current EOS: Peng-Robinson (1976)" in message
    assert "Best first-line workflows for this fluid: Dew Point, CVD, Separator, Phase Envelope" in message

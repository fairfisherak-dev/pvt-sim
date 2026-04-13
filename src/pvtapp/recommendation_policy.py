"""Advisory run-setup recommendations for desktop users.

This module classifies the entered fluid into a broad validated family and
surfaces the current best-fit workflow/EOS guidance from the canon app policy.
The output is advisory only; it does not silently change the runtime config.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from pvtcore.models import load_components, resolve_component_id

from pvtapp.capabilities import GUI_CALCULATION_TYPE_LABELS, GUI_EOS_TYPE_LABELS
from pvtapp.plus_fraction_policy import (
    PLUS_FRACTION_PRESET_LABELS,
    describe_plus_fraction_policy,
    infer_plus_fraction_preset,
    resolve_plus_fraction_entry,
)
from pvtapp.schemas import (
    CalculationType,
    EOSType,
    FluidComposition,
    PlusFractionCharacterizationPreset,
    PlusFractionEntry,
)

_LIGHT_COMPONENT_IDS = {"N2", "CO2", "H2S", "C1", "C2", "C3", "IC4", "C4", "IC5", "C5", "C6"}
_C2_TO_C6_IDS = ("C2", "C3", "IC4", "C4", "IC5", "C5", "C6")
_HEAVY_COMPONENT_PATTERN = re.compile(r"^C(\d+)$")


@dataclass(frozen=True)
class FamilyRecommendationProfile:
    """Static advisory profile for a resolved fluid family."""

    recommended_eos: EOSType
    eos_reason: str
    primary_workflows: tuple[CalculationType, ...]
    secondary_workflows: tuple[CalculationType, ...]
    workflow_reason: str


@dataclass(frozen=True)
class RunRecommendation:
    """Advisory recommendation for the current feed and selected workflow."""

    fluid_family: PlusFractionCharacterizationPreset
    family_reason: str
    selected_calculation_type: CalculationType
    selected_workflow_fit: str
    selected_workflow_reason: str
    recommended_eos: EOSType
    eos_reason: str
    primary_workflows: tuple[CalculationType, ...]
    secondary_workflows: tuple[CalculationType, ...]
    current_eos: Optional[EOSType] = None
    recommended_plus_policy: Optional[PlusFractionEntry] = None


_FAMILY_PROFILES: dict[PlusFractionCharacterizationPreset, FamilyRecommendationProfile] = {
    PlusFractionCharacterizationPreset.DRY_GAS: FamilyRecommendationProfile(
        recommended_eos=EOSType.SRK,
        eos_reason=(
            "Lean methane-dominant dry-gas feeds are the clearest fit for SRK on the "
            "current runtime surface. Keep Peng-Robinson (1976) as the conservative fallback "
            "if acid-gas handling or legacy parity matters more than gas-side sharpness."
        ),
        primary_workflows=(
            CalculationType.DEW_POINT,
            CalculationType.PT_FLASH,
            CalculationType.PHASE_ENVELOPE,
        ),
        secondary_workflows=(CalculationType.CVD,),
        workflow_reason=(
            "Dry gases are usually best framed around dew-point, flash, and envelope behavior. "
            "Oil-expansion workflows are normally low-signal for this family."
        ),
    ),
    PlusFractionCharacterizationPreset.CO2_RICH_GAS: FamilyRecommendationProfile(
        recommended_eos=EOSType.PENG_ROBINSON,
        eos_reason=(
            "CO2-rich and sour gas feeds stay on the classic Peng-Robinson (1976) baseline in the "
            "current canon policy. That keeps the recommendation on the conservative acid-gas path "
            "until predictive BIP admission is broader."
        ),
        primary_workflows=(
            CalculationType.DEW_POINT,
            CalculationType.PT_FLASH,
            CalculationType.PHASE_ENVELOPE,
        ),
        secondary_workflows=(CalculationType.CVD,),
        workflow_reason=(
            "Acid-gas systems are still primarily dew-point, flash, and envelope problems. "
            "Treat depletion-style workflows as secondary diagnostics unless liquid dropout is a real concern."
        ),
    ),
    PlusFractionCharacterizationPreset.GAS_CONDENSATE: FamilyRecommendationProfile(
        recommended_eos=EOSType.PR78,
        eos_reason=(
            "Gas-condensate feeds still behave gas-like, but their heavy-end sensitivity is material. "
            "Peng-Robinson (1978) is the best-fit recommendation because the heavier-end alpha extension "
            "is useful once condensate dropout matters."
        ),
        primary_workflows=(
            CalculationType.DEW_POINT,
            CalculationType.CVD,
            CalculationType.SEPARATOR,
            CalculationType.PHASE_ENVELOPE,
        ),
        secondary_workflows=(CalculationType.PT_FLASH,),
        workflow_reason=(
            "Gas condensates usually want dew-point and depletion-style workflows first, with separator "
            "and envelope follow-up once the condensation window is located."
        ),
    ),
    PlusFractionCharacterizationPreset.VOLATILE_OIL: FamilyRecommendationProfile(
        recommended_eos=EOSType.PR78,
        eos_reason=(
            "Volatile oils benefit from the Peng-Robinson (1978) heavy-end extension. It is the current "
            "best-fit recommendation for oil-side saturation and expansion workflows in this runtime surface."
        ),
        primary_workflows=(
            CalculationType.BUBBLE_POINT,
            CalculationType.CCE,
            CalculationType.DL,
            CalculationType.SEPARATOR,
        ),
        secondary_workflows=(
            CalculationType.PT_FLASH,
            CalculationType.PHASE_ENVELOPE,
        ),
        workflow_reason=(
            "Volatile oils are primarily bubble-point and expansion problems. Start there before using "
            "flash or envelope tools as supporting diagnostics."
        ),
    ),
    PlusFractionCharacterizationPreset.BLACK_OIL: FamilyRecommendationProfile(
        recommended_eos=EOSType.PR78,
        eos_reason=(
            "Black-oil-side feeds are heavier and more saturation-sensitive, so Peng-Robinson (1978) is the "
            "current best-fit recommendation for the exposed oil-side workflows."
        ),
        primary_workflows=(
            CalculationType.BUBBLE_POINT,
            CalculationType.CCE,
            CalculationType.DL,
            CalculationType.SEPARATOR,
        ),
        secondary_workflows=(CalculationType.PT_FLASH,),
        workflow_reason=(
            "Black oils are best validated through bubble-point, expansion, and separator workflows. "
            "Gas-side dew diagnostics are usually the wrong first question."
        ),
    ),
    PlusFractionCharacterizationPreset.SOUR_OIL: FamilyRecommendationProfile(
        recommended_eos=EOSType.PENG_ROBINSON,
        eos_reason=(
            "Sour oils keep the recommendation on classic Peng-Robinson (1976). Acid-gas sensitivity matters "
            "more here than the PR78 heavy-end tweak, so the conservative PR76 baseline is preferred."
        ),
        primary_workflows=(
            CalculationType.BUBBLE_POINT,
            CalculationType.CCE,
            CalculationType.DL,
            CalculationType.SEPARATOR,
        ),
        secondary_workflows=(
            CalculationType.PT_FLASH,
            CalculationType.PHASE_ENVELOPE,
        ),
        workflow_reason=(
            "Sour oils are still oil-side saturation and expansion problems first. Use flash or envelope views "
            "as secondary support, not the primary characterization workflow."
        ),
    ),
}


def _canonical_component_fractions(composition: FluidComposition) -> dict[str, float]:
    """Resolve component IDs into canonical fractions without losing inline pseudos."""

    all_components = load_components()
    inline_ids = {spec.component_id.strip().upper() for spec in composition.inline_components}
    fractions: dict[str, float] = {}
    for entry in composition.components:
        raw_id = entry.component_id.strip()
        if raw_id.upper() in inline_ids:
            canonical_id = raw_id.upper()
        else:
            try:
                canonical_id = resolve_component_id(raw_id, all_components).upper()
            except KeyError:
                canonical_id = raw_id.upper()
        fractions[canonical_id] = fractions.get(canonical_id, 0.0) + float(entry.mole_fraction)
    return fractions


def _extract_carbon_number(component_id: str) -> Optional[int]:
    """Return the carbon number for simple SCN-like IDs."""

    match = _HEAVY_COMPONENT_PATTERN.match(component_id)
    if not match:
        return None
    return int(match.group(1))


def _estimate_heavy_fraction(
    composition: FluidComposition,
    fractions: dict[str, float],
) -> float:
    """Estimate the heavy tail when no explicit plus fraction is present."""

    if composition.plus_fraction is not None:
        return float(composition.plus_fraction.z_plus)

    inline_ids = {spec.component_id.strip().upper() for spec in composition.inline_components}
    heavy_total = 0.0
    for component_id, mole_fraction in fractions.items():
        if component_id in inline_ids or component_id.startswith(("PSEUDO", "LUMP")):
            heavy_total += mole_fraction
            continue
        carbon_number = _extract_carbon_number(component_id)
        if carbon_number is not None and carbon_number >= 7:
            heavy_total += mole_fraction
            continue
        if component_id not in _LIGHT_COMPONENT_IDS:
            heavy_total += mole_fraction
    return heavy_total


def _infer_family_without_plus_fraction(
    composition: FluidComposition,
    calculation_type: CalculationType,
) -> tuple[PlusFractionCharacterizationPreset, str]:
    """Infer the closest family from explicit components only."""

    fractions = _canonical_component_fractions(composition)
    methane = fractions.get("C1", 0.0)
    co2 = fractions.get("CO2", 0.0)
    h2s = fractions.get("H2S", 0.0)
    acid = co2 + h2s
    c2_to_c6 = sum(fractions.get(component_id, 0.0) for component_id in _C2_TO_C6_IDS)
    heavy = _estimate_heavy_fraction(composition, fractions)
    gas_like = methane >= 0.55 and heavy <= 0.12

    if gas_like:
        if acid >= 0.20:
            family = PlusFractionCharacterizationPreset.CO2_RICH_GAS
            regime = "gas-like with high acid-gas loading"
        elif heavy >= 0.035 or c2_to_c6 >= 0.20:
            family = PlusFractionCharacterizationPreset.GAS_CONDENSATE
            regime = "gas-like but rich enough in intermediates/heavy tail to condense"
        else:
            family = PlusFractionCharacterizationPreset.DRY_GAS
            regime = "lean and methane-dominant"
    else:
        if h2s >= 0.05 or acid >= 0.10:
            family = PlusFractionCharacterizationPreset.SOUR_OIL
            regime = "oil-like with material acid-gas content"
        elif methane >= 0.25:
            family = PlusFractionCharacterizationPreset.VOLATILE_OIL
            regime = "oil-like but still methane-rich"
        else:
            family = PlusFractionCharacterizationPreset.BLACK_OIL
            regime = "oil-like with a larger heavy fraction"

    reason = (
        f"Methane = {methane:.3f}, acid gas = {acid:.3f}, C2-C6 = {c2_to_c6:.3f}, "
        f"heavy tail = {heavy:.3f}. That reads as {regime} for "
        f"{GUI_CALCULATION_TYPE_LABELS[calculation_type]}."
    )
    return family, reason


def _infer_fluid_family(
    composition: FluidComposition,
    calculation_type: CalculationType,
) -> tuple[PlusFractionCharacterizationPreset, str, Optional[PlusFractionEntry]]:
    """Infer the fluid family and recommended C7+ baseline, if applicable."""

    plus_fraction = composition.plus_fraction
    if plus_fraction is None:
        family, reason = _infer_family_without_plus_fraction(composition, calculation_type)
        return family, reason, None

    family = infer_plus_fraction_preset(composition.components, plus_fraction, calculation_type)
    fractions = _canonical_component_fractions(composition)
    methane = fractions.get("C1", 0.0)
    acid = fractions.get("CO2", 0.0) + fractions.get("H2S", 0.0)
    c2_to_c6 = sum(fractions.get(component_id, 0.0) for component_id in _C2_TO_C6_IDS)
    reason = (
        f"Methane = {methane:.3f}, acid gas = {acid:.3f}, C2-C6 = {c2_to_c6:.3f}, "
        f"C{plus_fraction.cut_start}+ = {float(plus_fraction.z_plus):.3f}. "
        f"That matches the {PLUS_FRACTION_PRESET_LABELS[family]} family for "
        f"{GUI_CALCULATION_TYPE_LABELS[calculation_type]}."
    )

    baseline_source = plus_fraction.model_copy(
        update={
            "characterization_preset": family,
            "resolved_characterization_preset": None,
        }
    )
    recommended_plus = resolve_plus_fraction_entry(
        composition.components,
        baseline_source,
        calculation_type,
    )
    return family, reason, recommended_plus


def _selected_workflow_fit(
    calculation_type: CalculationType,
    profile: FamilyRecommendationProfile,
) -> tuple[str, str]:
    """Rate how well the selected workflow matches the inferred family."""

    family_label = ", ".join(
        GUI_CALCULATION_TYPE_LABELS[item] for item in profile.primary_workflows
    )
    if calculation_type in profile.primary_workflows:
        return (
            "strong",
            f"{GUI_CALCULATION_TYPE_LABELS[calculation_type]} is a strong first-line workflow for this family.",
        )
    if calculation_type in profile.secondary_workflows:
        return (
            "acceptable",
            f"{GUI_CALCULATION_TYPE_LABELS[calculation_type]} is useful here, but the stronger first-line workflows are {family_label}.",
        )
    return (
        "weak",
        f"{GUI_CALCULATION_TYPE_LABELS[calculation_type]} is usually secondary for this family. Start with {family_label}.",
    )


def recommend_run_setup(
    composition: FluidComposition,
    calculation_type: CalculationType,
    *,
    current_eos: Optional[EOSType] = None,
) -> RunRecommendation:
    """Build an advisory workflow/EOS recommendation for the current input."""

    family, family_reason, recommended_plus = _infer_fluid_family(composition, calculation_type)
    profile = _FAMILY_PROFILES[family]
    workflow_fit, workflow_reason = _selected_workflow_fit(calculation_type, profile)
    return RunRecommendation(
        fluid_family=family,
        family_reason=family_reason,
        selected_calculation_type=calculation_type,
        selected_workflow_fit=workflow_fit,
        selected_workflow_reason=workflow_reason,
        recommended_eos=profile.recommended_eos,
        eos_reason=profile.eos_reason,
        primary_workflows=profile.primary_workflows,
        secondary_workflows=profile.secondary_workflows,
        current_eos=current_eos,
        recommended_plus_policy=recommended_plus,
    )


def format_run_recommendation(recommendation: RunRecommendation) -> str:
    """Render a user-facing recommendation block for the desktop dialog."""

    family_label = PLUS_FRACTION_PRESET_LABELS[recommendation.fluid_family]
    selected_calc_label = GUI_CALCULATION_TYPE_LABELS[recommendation.selected_calculation_type]
    recommended_eos_label = GUI_EOS_TYPE_LABELS[recommendation.recommended_eos]

    lines = [
        "Current best-fit recommendation",
        "(heuristic policy guidance; this does not change the current run config)",
        "",
        f"Fluid family: {family_label}",
        recommendation.family_reason,
        "",
        f"Selected workflow: {selected_calc_label}",
        f"Workflow fit: {recommendation.selected_workflow_fit.title()}",
        recommendation.selected_workflow_reason,
        "",
        f"Recommended EOS for this setup: {recommended_eos_label}",
        recommendation.eos_reason,
    ]

    if recommendation.current_eos is not None:
        current_eos_label = GUI_EOS_TYPE_LABELS[recommendation.current_eos]
        if recommendation.current_eos == recommendation.recommended_eos:
            lines.append("Current EOS already matches the recommendation.")
        else:
            lines.append(f"Current EOS: {current_eos_label}")

    primary = ", ".join(
        GUI_CALCULATION_TYPE_LABELS[item] for item in recommendation.primary_workflows
    )
    lines.extend(
        [
            "",
            f"Best first-line workflows for this fluid: {primary}",
        ]
    )

    if recommendation.secondary_workflows:
        secondary = ", ".join(
            GUI_CALCULATION_TYPE_LABELS[item] for item in recommendation.secondary_workflows
        )
        lines.append(f"Useful secondary workflows: {secondary}")

    if recommendation.recommended_plus_policy is not None:
        lines.extend(
            [
                "",
                "Validated C7+ baseline for this family:",
                describe_plus_fraction_policy(recommendation.recommended_plus_policy),
            ]
        )

    lines.extend(
        [
            "",
            "This recommendation is based on the repo's current validated family defaults and exposed runtime methods.",
            "It does not change the current run config unless you manually update the selections yourself.",
        ]
    )
    return "\n".join(lines)
